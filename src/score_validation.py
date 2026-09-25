"""Score predictions for held-out training Source 1 IDs using macro F0.5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from phase1 import EXPECTED_TRUTH, entity_f05, rows, validation_member


def id_set(value: str) -> set[str]:
    return {item for item in value.split(",") if item}


def score(prediction_path: Path, truth_path: Path, seed: int = 2026) -> dict:
    predictions = {}
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Bad prediction columns: {reader.fieldnames}")
        for row in reader:
            eid = row["source1_entity_id"]
            if eid in predictions:
                raise ValueError(f"Duplicate prediction: {eid}")
            if validation_member(eid, seed):
                predictions[eid] = id_set(row["matched_entity_ids"])
    score_sum = singleton_correct = singleton_total = count = 0
    seen = set()
    for row in rows(truth_path, EXPECTED_TRUTH):
        eid = row["source1_entity_id"]
        if not validation_member(eid, seed):
            continue
        if eid not in predictions:
            raise ValueError(f"Missing validation prediction: {eid}")
        gold = id_set(row["matched_entity_ids"])
        predicted = predictions[eid]
        score_sum += entity_f05(gold, predicted)
        count += 1
        if not gold:
            singleton_total += 1
            singleton_correct += not predicted
        seen.add(eid)
    if set(predictions) != seen:
        raise ValueError(f"Predictions contain {len(set(predictions) - seen)} unknown validation IDs")
    return {"validation_entities": count, "macro_f0.5": score_sum / count,
            "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
            "singleton_entities": singleton_total, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(score(args.predictions, args.truth, args.seed), indent=2))


if __name__ == "__main__":
    main()
