"""Parallel candidate retriever for test set inference.

Distributes queries across multiple worker processes on CPU cores.
All workers share the read-only memory-mapped postings and sketch.
Outputs candidate_pairs.tsv in the exact order of test_source1.tsv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from phase1 import EXPECTED_SOURCE, rows
from phase2 import blocking_keys, record_view
from phase2_indexed import (
    POSTING, SKETCH, ID_BYTES, PACK, VIEW_BITS,
    key_hash, sketch_view, sketch_rank, key_weight
)


def worker_retrieve(
    worker_id: int,
    s1_path_str: str,
    partition: str,
    index_dir: str,
    start_idx: int,
    end_idx: int,
    top_k: int,
    max_postings: int,
    out_path: str
) -> dict:
    meta_path = Path(index_dir) / f"{partition}_index.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    index = np.memmap(Path(index_dir) / f"{partition}_postings.bin", mode="r", dtype=POSTING, shape=(meta["postings"],))
    ids = np.memmap(Path(index_dir) / f"{partition}_ids.bin", mode="r", dtype=f"S{ID_BYTES}", shape=(meta["target_count"],))
    sketch_path = Path(index_dir) / f"{partition}_sketch.bin"
    sketch = np.memmap(sketch_path, mode="r", dtype=SKETCH, shape=(meta["target_count"],)) if sketch_path.exists() else None

    began = time.monotonic()
    emitted = 0
    target_count = end_idx - start_idx
    processed = 0

    with open(s1_path_str, "r", encoding="utf-8", newline="") as infile, open(out_path, "w", encoding="utf-8", newline="") as out:
        header = infile.readline()
        for idx, line in enumerate(infile):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            parts = line.rstrip("\r\n").split("\t")
            row = {
                "entity_id": parts[0],
                "business_name": parts[1] if len(parts) > 1 else "",
                "business_address": parts[2] if len(parts) > 2 else "",
                "country": parts[3] if len(parts) > 3 else "",
            }
            scores = {}
            query_view = record_view(row)
            for key in blocking_keys(query_view):
                hash_value = key_hash(key)
                low = np.array((hash_value, 0), dtype=POSTING)
                high = np.array((hash_value, 0xFFFFFFFF), dtype=POSTING)
                left = int(np.searchsorted(index, low, side="left"))
                right = int(np.searchsorted(index, high, side="right"))
                size = right - left
                if not size or size > max_postings:
                    continue
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
                best = heapq.nlargest(top_k, scores, key=lambda n: (sketch_rank(query_sketch, sketch[n], scores[n][1]), -n))

            candidate_ids = [ids[n].decode("ascii") for n in best]
            out.write(row["entity_id"] + "\t" + ",".join(candidate_ids) + "\n")
            emitted += len(candidate_ids)
            processed += 1

            if processed % 20000 == 0:
                print(f"[Worker {worker_id}] Processed {processed:,}/{target_count:,} queries ({emitted:,} candidates)", flush=True)

    elapsed = round(time.monotonic() - began, 2)
    print(f"[Worker {worker_id}] FINISHED {processed:,} queries in {elapsed}s", flush=True)
    return {"worker_id": worker_id, "queries": processed, "candidates": emitted, "seconds": elapsed}


def run_parallel_retrieval(
    data_dir: Path,
    partition: str,
    index_dir: Path,
    out: Path,
    top_k: int = 150,
    max_postings: int = 200,
    n_workers: int = 6
) -> dict:
    s1_path = data_dir / partition / f"{partition}_source1.tsv"
    print(f"Counting records in {s1_path}...", flush=True)
    with open(s1_path, "r", encoding="utf-8") as f:
        n_total = sum(1 for _ in f) - 1
    print(f"Found {n_total:,} Source 1 records.", flush=True)

    # Split into contiguous chunks by index
    chunk_size = (n_total + n_workers - 1) // n_workers
    part_files = []
    temp_dir = out.parent / "temp_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    worker_ranges = []
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_total)
        part_file = temp_dir / f"cand_part_{i}.tsv"
        part_files.append(str(part_file))
        worker_ranges.append((start_idx, end_idx))
        print(f"Worker {i}: rows {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} queries) -> {part_file.name}", flush=True)

    print(f"\nLaunching {n_workers} retrieval workers across CPU cores...", flush=True)
    start_time = time.monotonic()

    with mp.Pool(processes=n_workers) as pool:
        async_results = [
            pool.apply_async(
                worker_retrieve,
                args=(i, str(s1_path), partition, str(index_dir), worker_ranges[i][0], worker_ranges[i][1], top_k, max_postings, part_files[i])
            )
            for i in range(n_workers)
        ]
        results = [res.get() for res in async_results]

    total_time = round(time.monotonic() - start_time, 2)
    print(f"\nAll workers finished in {total_time}s! Merging parts into {out}...", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_pairs = 0
    seen_ids = set()

    with out.open("w", encoding="utf-8", newline="") as outfile:
        outfile.write("source1_entity_id\tcandidate_entity_ids\n")
        for part_path in part_files:
            with open(part_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    eid = line.partition("\t")[0]
                    if eid in seen_ids:
                        raise ValueError(f"Duplicate entity ID in output: {eid}")
                    seen_ids.add(eid)
                    outfile.write(line)
                    total_written += 1
                    cands = line.partition("\t")[2].strip().split(",") if line.partition("\t")[2].strip() else []
                    total_pairs += len(cands)
            # Remove part file
            try:
                os.remove(part_path)
            except OSError:
                pass

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    assert total_written == n_total, f"Expected {n_total} rows, wrote {total_written}"
    print(f"Successfully generated {out}: {total_written:,} entities, {total_pairs:,} candidate pairs in {total_time}s.", flush=True)

    report = {
        "partition": partition,
        "source1_entities": total_written,
        "candidate_pairs": total_pairs,
        "candidates_per_source1": total_pairs / total_written if total_written else 0,
        "total_seconds": total_time,
        "n_workers": n_workers
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--partition", choices=["train", "test"], default="test")
    parser.add_argument("--index-dir", type=Path, default=Path("work/block_index"))
    parser.add_argument("--out", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--max-postings", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    report = run_parallel_retrieval(
        args.data_dir,
        args.partition,
        args.index_dir,
        args.out,
        args.top_k,
        args.max_postings,
        args.workers
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
