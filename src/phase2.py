"""Multi-view, bounded blocking for Source 1 to Sources 2/3.

Uses only supplied TSVs. A query batch scans target files twice: first to find
overlarge blocks, then to retain the best candidate IDs. Batching bounds RAM and
does not require a multi-gigabyte persistent search index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from phase1 import EXPECTED_SOURCE, EXPECTED_TRUTH, rows, validation_member


TOKEN = re.compile(r"[a-z0-9]+")
NUMERIC = re.compile(r"^\d{1,6}$")
ALIASES = {"corp": "corporation", "co": "company", "pvt": "private",
           "ltd": "limited", "st": "street", "rd": "road", "ave": "avenue",
           "blvd": "boulevard", "ctr": "center"}
LEGAL = {"inc", "incorporated", "llc", "corporation", "company", "private", "limited", "plc", "llp"}
ADDRESS_GENERIC = {"street", "road", "avenue", "boulevard", "near", "opposite", "floor", "unit", "building", "block"}


def tokens(value: str) -> tuple[str, ...]:
    if value.isascii():
        ascii_text = value.lower().replace("&", " and ")
    else:
        normalized = unicodedata.normalize("NFKD", value.casefold().replace("&", " and "))
        ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return tuple(ALIASES.get(token, token) for token in TOKEN.findall(ascii_text))


def selected(values: tuple[str, ...], ignored: set[str], limit: int) -> tuple[str, ...]:
    unique = {x for x in values if len(x) >= 3 and x not in ignored and not NUMERIC.fullmatch(x)}
    return tuple(sorted(unique, key=lambda x: (-len(x), x))[:limit])


def record_view(row: dict[str, str]) -> tuple[tuple[str, ...], tuple[str, ...], str | None, str]:
    name = tokens(row["business_name"])
    address = tokens(row["business_address"])
    number = next((x for x in address if NUMERIC.fullmatch(x)), None)
    return selected(name, LEGAL, 4), selected(address, ADDRESS_GENERIC, 3), number, row["country"].strip().casefold()


def blocking_keys(view: tuple[tuple[str, ...], tuple[str, ...], str | None, str]) -> set[str]:
    name, address, number, country = view
    keys = set()
    if name:
        keys.add("NX|" + "|".join(sorted(name)))
        for item in name[:2]:
            if len(item) >= 7:
                keys.add(f"N1|{country}|{item}")
            if len(item) >= 5:
                keys.add(f"NS|{country}|{item[:5]}")
    for left, right in combinations(name, 2):
        keys.add(f"N2|{min(left,right)}|{max(left,right)}")
    for left in name[:2]:
        for right in address[:2]:
            keys.add(f"NA|{left}|{right}")
    if number:
        for item in name[:2]:
            keys.add(f"HN|{number}|{item}")
        for item in address[:2]:
            keys.add(f"HA|{number}|{item}")
    if address:
        keys.add("AX|" + "|".join(sorted(address)))
    return keys


def overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def preliminary_score(query, target, shared_keys: set[str]) -> float:
    qn, qa, qnum, qc = query
    tn, ta, tnum, tc = target
    name_score = overlap(qn, tn)
    address_score = overlap(qa, ta)
    score = 4 * name_score + 2 * address_score + min(len(shared_keys), 4) * 0.4
    if qnum and tnum:
        score += 0.8 if qnum == tnum else -0.7
    if qc and tc and qc == tc:
        score += 0.25
    return score


def source_rows(data_dir: Path, partition: str):
    for source in ("source2", "source3"):
        yield from rows(data_dir / partition / f"{partition}_{source}.tsv", EXPECTED_SOURCE)


def selected_source1(data_dir: Path, partition: str, seed: int, sample_modulus: int, split: str = "validation"):
    for row in rows(data_dir / partition / f"{partition}_source1.tsv", EXPECTED_SOURCE):
        eid = row["entity_id"]
        if partition == "train":
            is_val = validation_member(eid, seed)
            if split == "validation" and not is_val:
                continue
            elif split == "development" and is_val:
                continue
        if sample_modulus > 1:
            digest = hashlib.blake2b(f"sample:{eid}".encode(), digest_size=8).digest()
            if int.from_bytes(digest, "big") % sample_modulus:
                continue
        yield row


def load_truth(data_dir: Path, selected_ids: set[str]) -> dict[str, set[str]]:
    truth = {}
    for row in rows(data_dir / "train" / "train_ground_truth.tsv", EXPECTED_TRUTH):
        eid = row["source1_entity_id"]
        if eid in selected_ids:
            truth[eid] = {x for x in row["matched_entity_ids"].split(",") if x}
    if set(truth) != selected_ids:
        raise ValueError(f"Ground truth missing {len(selected_ids - set(truth))} selected Source 1 IDs")
    return truth


def run_batch(data_dir: Path, partition: str, queries: list[dict[str, str]],
              max_postings: int, top_k: int, out, truth: dict[str, set[str]] | None):
    started = time.monotonic()
    query_views = {row["entity_id"]: record_view(row) for row in queries}
    key_to_queries = defaultdict(list)
    for eid, view in query_views.items():
        for key in blocking_keys(view):
            key_to_queries[key].append(eid)
    # A key shared by too many queries is not selective even if rare in targets.
    key_to_queries = {key: ids for key, ids in key_to_queries.items() if len(ids) <= max_postings}
    frequency = Counter()
    for row in source_rows(data_dir, partition):
        for key in blocking_keys(record_view(row)):
            if key in key_to_queries and frequency[key] <= max_postings:
                frequency[key] += 1
    active = {key: ids for key, ids in key_to_queries.items() if 0 < frequency[key] <= max_postings}
    heaps = defaultdict(list)
    raw_pairs = 0
    for row in source_rows(data_dir, partition):
        target_view = record_view(row)
        touched = defaultdict(set)
        for key in blocking_keys(target_view):
            for eid in active.get(key, ()):
                touched[eid].add(key)
        target_id = row["entity_id"]
        for eid, shared in touched.items():
            raw_pairs += 1
            score = preliminary_score(query_views[eid], target_view, shared)
            item = (score, target_id)
            heap = heaps[eid]
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    truth_total = truth_found = truth_entities_hit = nonempty_truth = 0
    kept_pairs = 0
    for row in queries:
        eid = row["entity_id"]
        candidates = [target_id for _, target_id in sorted(heaps[eid], reverse=True)]
        kept_pairs += len(candidates)
        out.write(eid + "\t" + ",".join(candidates) + "\n")
        if truth is not None:
            gold = truth[eid]
            truth_total += len(gold)
            truth_found += len(gold & set(candidates))
            nonempty_truth += bool(gold)
            truth_entities_hit += bool(gold & set(candidates))
    return {"source1_entities": len(queries), "query_keys": len(key_to_queries),
            "active_keys": len(active), "raw_candidate_pairs": raw_pairs,
            "retained_candidate_pairs": kept_pairs, "true_links": truth_total,
            "true_links_retrieved": truth_found, "nonempty_truth_entities": nonempty_truth,
            "nonempty_truth_entities_with_candidate": truth_entities_hit,
            "seconds": round(time.monotonic() - started, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--partition", choices=["train", "test"], default="train")
    parser.add_argument("--out", type=Path, default=Path("work/phase2_validation_candidates.tsv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/phase2_blocking_report.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-modulus", type=int, default=32,
                        help="1 selects all validation Source 1 records; 32 selects about 1/32.")
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--max-postings", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    selected_rows = list(selected_source1(args.data_dir, args.partition, args.seed, args.sample_modulus))
    truth = load_truth(args.data_dir, {row["entity_id"] for row in selected_rows}) if args.partition == "train" else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    batches = []
    with args.out.open("w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0, len(selected_rows), args.batch_size):
            batch = selected_rows[start:start + args.batch_size]
            metric = run_batch(args.data_dir, args.partition, batch, args.max_postings, args.top_k, out, truth)
            batches.append(metric)
            print(f"batch {len(batches)}: {json.dumps(metric)}", flush=True)
    total = {key: sum(batch[key] for batch in batches) for key in
             ("source1_entities", "raw_candidate_pairs", "retained_candidate_pairs", "true_links",
              "true_links_retrieved", "nonempty_truth_entities", "nonempty_truth_entities_with_candidate", "seconds")}
    total["pair_recall"] = total["true_links_retrieved"] / total["true_links"] if total["true_links"] else None
    total["entity_hit_rate"] = total["nonempty_truth_entities_with_candidate"] / total["nonempty_truth_entities"] if total["nonempty_truth_entities"] else None
    total["candidates_per_source1"] = total["retained_candidate_pairs"] / total["source1_entities"] if total["source1_entities"] else None
    report = {"partition": args.partition, "sample_modulus": args.sample_modulus,
              "batch_size": args.batch_size, "max_postings": args.max_postings, "top_k": args.top_k,
              "method": "multi-view token and numeric blocks, bounded by posting frequency and top-k",
              "totals": total, "batches": batches}
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
