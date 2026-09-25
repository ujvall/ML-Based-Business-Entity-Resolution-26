"""Summarize missed true links in a sampled Phase 2 candidate file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from phase1 import EXPECTED_SOURCE, EXPECTED_TRUTH, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase2_miss_analysis.json"))
    args = parser.parse_args()
    candidate_ids = {}
    with args.candidates.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            candidate_ids[row["source1_entity_id"]] = set(row["candidate_entity_ids"].split(","))
    missing_by_s1 = {}
    link_counts = Counter()
    source_counts = Counter()
    for row in rows(args.data_dir / "train" / "train_ground_truth.tsv", EXPECTED_TRUTH):
        eid = row["source1_entity_id"]
        if eid not in candidate_ids:
            continue
        gold = {x for x in row["matched_entity_ids"].split(",") if x}
        missing = gold - candidate_ids[eid]
        if missing:
            missing_by_s1[eid] = sorted(missing)
            link_counts[len(gold)] += len(missing)
            source_counts.update(x.split("-", 1)[0] for x in missing)
    countries = Counter()
    examples = []
    for row in rows(args.data_dir / "train" / "train_source1.tsv", EXPECTED_SOURCE):
        eid = row["entity_id"]
        if eid in missing_by_s1:
            countries[row["country"]] += len(missing_by_s1[eid])
            if len(examples) < 12:
                examples.append({"source1_entity_id": eid, "name": row["business_name"],
                                 "address": row["business_address"],
                                 "missed_target_ids": missing_by_s1[eid]})
    result = {"source1_entities_with_misses": len(missing_by_s1),
              "missed_links": sum(len(x) for x in missing_by_s1.values()),
              "by_country": dict(countries), "by_target_source": dict(source_counts),
              "by_true_match_count": dict(sorted(link_counts.items())), "examples": examples}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
