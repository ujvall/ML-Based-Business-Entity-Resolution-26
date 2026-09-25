"""Independently measure blocking recall and its best possible F0.5 ceiling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase1 import EXPECTED_TRUTH, entity_f05, rows, validation_member


def evaluate(candidates_path: Path, truth_path: Path, target_count: int,
             require_full: bool = False, seed: int = 2026) -> dict:
    # Load only held-out labels; stream candidate rows to keep memory bounded.
    truth = {}
    for row in rows(truth_path, EXPECTED_TRUTH):
        eid = row["source1_entity_id"]
        if validation_member(eid, seed):
            truth[eid] = {x for x in row["matched_entity_ids"].split(",") if x}
    pairs = positives = found = ceiling = singleton_count = empty_singletons = 0
    seen = set()
    with candidates_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Unexpected candidate header: {reader.fieldnames}")
        for row in reader:
            eid = row["source1_entity_id"]
            if eid in seen:
                raise ValueError(f"Duplicate Source 1 candidate row: {eid}")
            seen.add(eid)
            if eid not in truth:
                raise ValueError(f"Candidate row is not in validation ground truth: {eid}")
            ids = [x for x in row["candidate_entity_ids"].split(",") if x]
            kept = set(ids)
            if len(ids) != len(kept):
                raise ValueError(f"Duplicate candidate IDs for {eid}")
            gold = truth[eid]
            hit = gold & kept
            pairs += len(kept)
            positives += len(gold)
            found += len(hit)
            ceiling += entity_f05(gold, hit)
            if not gold:
                singleton_count += 1
                empty_singletons += not kept
    if require_full and len(seen) != len(truth):
        raise ValueError(f"Full validation requires {len(truth)} rows; found {len(seen)}")
    n = len(seen)
    return {"source1_entities": n, "target_entities": target_count,
            "candidate_pairs": pairs, "candidates_per_source1": pairs / n,
            "reduction_ratio": 1 - pairs / (n * target_count),
            "true_links": positives, "true_links_retrieved": found,
            "pair_recall": found / positives,
            "oracle_macro_f0.5_ceiling": ceiling / n,
            "singleton_entities": singleton_count,
            "singletons_with_no_candidates": empty_singletons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--target-count", type=int, default=10_320_219)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_quality.json"))
    args = parser.parse_args()
    result = evaluate(args.candidates, args.truth, args.target_count,
                      args.require_full, args.seed)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
