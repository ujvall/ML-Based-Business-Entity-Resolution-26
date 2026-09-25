"""Reusable on-disk index for Phase 2 multi-view blocking at full test scale.

The index stores a stable 64-bit key hash and a target-row number for each
blocking key. A 64-bit collision is vanishingly unlikely but can only add a
candidate, never silently remove one. No external data or services are used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import struct
import time
import zlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from phase2 import blocking_keys, record_view, selected_source1, source_rows


POSTING = np.dtype([("key", "<u8"), ("rid", "<u4")])
PACK = struct.Struct("<QI")
ID_BYTES = 16
VIEW_BITS = {"NX": 1, "N2": 2, "NA": 4, "HN": 8, "AX": 16,
             "HA": 32, "N1": 64, "NS": 128}
SKETCH = np.dtype([("name", "<u4", (4,)), ("address", "<u4", (3,)),
                   ("number", "<u4"), ("country", "<u4")])


def key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def token_hash(value: str) -> int:
    # Zero is reserved for padding in the fixed-width sketch.
    return zlib.crc32(value.encode("utf-8")) or 0xFFFFFFFF


def sketch_view(view) -> tuple[list[int], list[int], int, int]:
    name, address, number, country = view
    return ([token_hash(x) for x in name] + [0] * (4 - len(name)),
            [token_hash(x) for x in address] + [0] * (3 - len(address)),
            token_hash(number) if number else 0,
            token_hash(country) if country else 0)


def build_sketch(data_dir: Path, partition: str, index_dir: Path) -> dict:
    meta = json.loads((index_dir / f"{partition}_index.json").read_text(encoding="utf-8"))
    path = index_dir / f"{partition}_sketch.bin"
    if path.exists():
        raise FileExistsError(f"Sketch already exists: {path}")
    sketch = np.memmap(path, mode="w+", dtype=SKETCH, shape=(meta["target_count"],))
    began = time.monotonic()
    n = 0
    for row in source_rows(data_dir, partition):
        if n >= meta["target_count"]:
            raise ValueError("More target rows than the index metadata declares")
        name, address, number, country = sketch_view(record_view(row))
        sketch[n] = (name, address, number, country)
        n += 1
        if n % 1_000_000 == 0:
            print(f"sketched {n:,} targets", flush=True)
    sketch.flush()
    del sketch
    if n != meta["target_count"]:
        raise ValueError(f"Sketch has {n} rows; index has {meta['target_count']}")
    result = {"partition": partition, "target_count": n, "sketch_bytes": path.stat().st_size,
              "build_seconds": round(time.monotonic() - began, 2)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def hashed_jaccard(left, right) -> float:
    a = set(int(x) for x in left if x)
    b = set(int(x) for x in right if x)
    return len(a & b) / len(a | b) if a and b else 0.0


def sketch_rank(query, target, shared_count: int) -> float:
    name = hashed_jaccard(query[0], target["name"])
    address = hashed_jaccard(query[1], target["address"])
    score = 4 * name + 2 * address + 0.4 * min(shared_count, 4)
    if query[2] and target["number"]:
        score += 0.8 if query[2] == target["number"] else -0.7
    if query[3] and query[3] == target["country"]:
        score += 0.25
    return score


def build(data_dir: Path, partition: str, index_dir: Path, target_count: int) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / f"{partition}_postings.bin"
    ids_file = index_dir / f"{partition}_ids.bin"
    meta_file = index_dir / f"{partition}_index.json"
    if index_file.exists() or ids_file.exists() or meta_file.exists():
        raise FileExistsError(f"Index files already exist for {partition}; choose a fresh --index-dir")
    ids = np.memmap(ids_file, mode="w+", dtype=f"S{ID_BYTES}", shape=(target_count,))
    began = time.monotonic()
    n = postings = 0
    buffer = bytearray()
    with index_file.open("wb", buffering=8 * 1024 * 1024) as output:
        for row in source_rows(data_dir, partition):
            if n >= target_count:
                raise ValueError("Target count is smaller than the number of source rows")
            eid = row["entity_id"].encode("ascii")
            if len(eid) > ID_BYTES:
                raise ValueError(f"Entity ID exceeds {ID_BYTES} bytes: {eid!r}")
            ids[n] = eid
            for key in blocking_keys(record_view(row)):
                buffer.extend(PACK.pack(key_hash(key), n))
                postings += 1
            n += 1
            if len(buffer) >= 8 * 1024 * 1024:
                output.write(buffer)
                buffer.clear()
            if n % 1_000_000 == 0:
                print(f"indexed {n:,} targets / {postings:,} postings", flush=True)
        if buffer:
            output.write(buffer)
    ids.flush()
    del ids
    if n != target_count:
        raise ValueError(f"Expected {target_count} target rows; found {n}")
    expected_size = postings * POSTING.itemsize
    if index_file.stat().st_size != expected_size:
        raise ValueError("Posting file size mismatch")
    print(f"sorting {postings:,} postings in place", flush=True)
    data = np.memmap(index_file, mode="r+", dtype=POSTING, shape=(postings,))
    data.sort(order="key", kind="quicksort")
    data.flush()
    del data
    meta = {"partition": partition, "target_count": n, "postings": postings,
            "index_bytes": expected_size, "build_seconds": round(time.monotonic() - began, 2)}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def key_weight(key: str) -> float:
    prefix = key.split("|", 1)[0]
    return {"NX": 4.0, "N2": 3.0, "NA": 2.5, "HN": 2.5,
            "AX": 2.0, "HA": 1.5, "N1": 1.0, "NS": 0.7}[prefix]


def retrieve(data_dir: Path, partition: str, index_dir: Path, out: Path,
             seed: int, sample_modulus: int, top_k: int, max_postings: int,
             evidence_out: Path | None = None, split: str = "validation") -> dict:
    meta = json.loads((index_dir / f"{partition}_index.json").read_text(encoding="utf-8"))
    index = np.memmap(index_dir / f"{partition}_postings.bin", mode="r", dtype=POSTING,
                      shape=(meta["postings"],))
    ids = np.memmap(index_dir / f"{partition}_ids.bin", mode="r", dtype=f"S{ID_BYTES}",
                    shape=(meta["target_count"],))
    sketch_path = index_dir / f"{partition}_sketch.bin"
    sketch = np.memmap(sketch_path, mode="r", dtype=SKETCH,
                       shape=(meta["target_count"],)) if sketch_path.exists() else None
    began = time.monotonic()
    queries = emitted = used_blocks = skipped_blocks = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    output_opener = gzip.open if out.suffix == ".gz" else Path.open
    if evidence_out is not None:
        evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_context = evidence_out.open("w", encoding="utf-8", newline="") if evidence_out else nullcontext(None)
    with output_opener(out, "wt" if out.suffix == ".gz" else "w",
                       encoding="utf-8", newline="") as output, evidence_context as evidence:
        output.write("source1_entity_id\tcandidate_entity_ids\n")
        if evidence is not None:
            evidence.write("source1_entity_id\tcandidate_view_masks\n")
        for row in selected_source1(data_dir, partition, seed, sample_modulus, split=split):
            scores = {}
            query_view = record_view(row)
            for key in blocking_keys(query_view):
                hash_value = key_hash(key)
                # Search the contiguous structured records. Searching index["key"]
                # makes NumPy copy its strided view on each call at this scale.
                low = np.array((hash_value, 0), dtype=POSTING)
                high = np.array((hash_value, 0xFFFFFFFF), dtype=POSTING)
                left = int(np.searchsorted(index, low, side="left"))
                right = int(np.searchsorted(index, high, side="right"))
                size = right - left
                if not size:
                    continue
                if size > max_postings:
                    skipped_blocks += 1
                    continue
                used_blocks += 1
                weight = key_weight(key)
                view_bit = VIEW_BITS[key.split("|", 1)[0]]
                for rid in index["rid"][left:right]:
                    n = int(rid)
                    old_weight, old_count, old_mask = scores.get(n, (0.0, 0, 0))
                    scores[n] = (old_weight + weight, old_count + 1, old_mask | view_bit)
            if sketch is None:
                best = heapq.nlargest(top_k, scores, key=lambda n: (scores[n][0], -n))
            else:
                query_sketch = sketch_view(query_view)
                best = heapq.nlargest(top_k, scores,
                    key=lambda n: (sketch_rank(query_sketch, sketch[n], scores[n][1]), -n))
            candidate_ids = [ids[n].decode("ascii") for n in best]
            output.write(row["entity_id"] + "\t" + ",".join(candidate_ids) + "\n")
            if evidence is not None:
                evidence.write(row["entity_id"] + "\t" + ",".join(str(scores[n][2]) for n in best) + "\n")
            queries += 1
            emitted += len(candidate_ids)
            if queries % 25_000 == 0:
                print(f"retrieved for {queries:,} Source 1 records", flush=True)
    report = {"partition": partition, "source1_entities": queries, "candidate_pairs": emitted,
              "candidates_per_source1": emitted / queries if queries else 0,
              "reduction_ratio": 1 - emitted / (queries * meta["target_count"]) if queries else None,
              "used_blocks": used_blocks, "skipped_broad_blocks": skipped_blocks,
              "retrieve_seconds": round(time.monotonic() - began, 2), "top_k": top_k,
              "max_postings": max_postings, "sample_modulus": sample_modulus,
              "ranking": "name_address_sketch" if sketch is not None else "block_weights",
              "evidence_file": str(evidence_out) if evidence_out else None}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "sketch", "retrieve"])
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--partition", choices=["train", "test"], default="train")
    parser.add_argument("--index-dir", type=Path, default=Path("work/block_index"))
    parser.add_argument("--target-count", type=int, default=10_320_219)
    parser.add_argument("--out", type=Path, default=Path("work/phase2_indexed_candidates.tsv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/phase2_indexed_report.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-modulus", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-postings", type=int, default=200)
    parser.add_argument("--evidence-out", type=Path,
                        help="Optional per-candidate retrieval-view masks aligned to candidate IDs")
    parser.add_argument("--split", choices=["validation", "development", "all"], default="validation")
    args = parser.parse_args()
    if args.command == "build":
        build(args.data_dir, args.partition, args.index_dir, args.target_count)
    elif args.command == "sketch":
        build_sketch(args.data_dir, args.partition, args.index_dir)
    else:
        result = retrieve(args.data_dir, args.partition, args.index_dir, args.out,
                          args.seed, args.sample_modulus, args.top_k, args.max_postings,
                          args.evidence_out, split=args.split)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
