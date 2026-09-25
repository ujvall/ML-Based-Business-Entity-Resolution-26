"""Stream the supplied TSVs, freeze a Source-1 split, and score macro F0.5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SOURCE = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_TRUTH = ["source1_entity_id", "matched_entity_ids"]


def rows(path: Path, expected: list[str]):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected {expected}, got {reader.fieldnames}")
        yield from reader


def validation_member(entity_id: str, seed: int = 2026, fraction: float = 0.2) -> bool:
    digest = hashlib.blake2b(f"{seed}:{entity_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") < int(fraction * 2**64)


def entity_f05(true_ids: set[str], predicted_ids: set[str]) -> float:
    if not true_ids:
        return float(not predicted_ids)
    if not predicted_ids:
        return 0.0
    correct = len(true_ids & predicted_ids)
    if not correct:
        return 0.0
    precision = correct / len(predicted_ids)
    recall = correct / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def audit(data_dir: Path, seed: int) -> dict:
    result = {"split": {"method": "blake2b_hash_source1_id", "seed": seed, "validation_fraction": 0.2}, "files": {}}
    for partition in ("train", "test"):
        for source in ("source1", "source2", "source3"):
            name = f"{partition}_{source}.tsv"
            path = data_dir / partition / name
            country = Counter()
            missing = Counter()
            total = 0
            duplicated_ids = 0
            seen = set()
            valid = 0
            for row in rows(path, EXPECTED_SOURCE):
                total += 1
                eid = row["entity_id"]
                if eid in seen:
                    duplicated_ids += 1
                else:
                    seen.add(eid)
                country[row["country"].strip() or "<missing>"] += 1
                for key in ("business_name", "business_address", "country"):
                    if not row[key].strip():
                        missing[key] += 1
                if partition == "train" and source == "source1" and validation_member(eid, seed):
                    valid += 1
            result["files"][name] = {"rows": total, "duplicate_entity_ids": duplicated_ids,
                "country_counts": dict(country), "missing_counts": dict(missing)}
            if partition == "train" and source == "source1":
                result["split"]["validation_entities"] = valid
                result["split"]["development_entities"] = total - valid
    truth_path = data_dir / "train" / "train_ground_truth.tsv"
    truth_ids = set()
    match_counts = Counter()
    singleton = 0
    validation_truth = 0
    truth_rows = 0
    duplicate_truth_rows = 0
    for row in rows(truth_path, EXPECTED_TRUTH):
        truth_rows += 1
        eid = row["source1_entity_id"]
        if eid in truth_ids:
            duplicate_truth_rows += 1
        else:
            truth_ids.add(eid)
        ids = [x for x in row["matched_entity_ids"].split(",") if x]
        match_counts[len(ids)] += 1
        singleton += not ids
        validation_truth += validation_member(eid, seed)
    result["ground_truth"] = {"rows": truth_rows, "duplicate_source1_rows": duplicate_truth_rows,
        "singletons": singleton, "singleton_fraction": singleton / truth_rows,
        "match_count_distribution": {str(k): v for k, v in sorted(match_counts.items())},
        "validation_rows": validation_truth}
    result["metric_self_test"] = {
        "correct_singleton": entity_f05(set(), set()),
        "false_singleton_merge": entity_f05(set(), {"S2-a"}),
        "perfect_pair": entity_f05({"S2-a"}, {"S2-a"}),
        "one_extra_pair": entity_f05({"S2-a"}, {"S2-a", "S2-b"}),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/phase1_audit.json"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = audit(args.data_dir, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
