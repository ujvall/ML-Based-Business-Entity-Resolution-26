"""Evaluation utilities for Entity Resolution matching models under macro F0.5."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from phase1 import entity_f05, validation_member, EXPECTED_TRUTH, rows


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    truth: Dict[str, Set[str]],
    s1_records: Dict[str, Tuple[str, str, str]] | None = None,
    target_records: Dict[str, Tuple[str, str, str]] | None = None,
    max_examples: int = 5
) -> dict:
    """Evaluate predictions against ground truth and produce detailed metrics."""
    score_sum = 0.0
    singleton_correct = 0
    singleton_total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    count = 0

    fp_examples = []
    fn_examples = []

    for eid, gold in truth.items():
        if eid not in predictions:
            raise ValueError(f"Missing prediction for entity: {eid}")
        pred = predictions[eid]
        score = entity_f05(gold, pred)
        score_sum += score
        count += 1

        if not gold:
            singleton_total += 1
            if not pred:
                singleton_correct += 1
        else:
            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            if fp > 0 and len(fp_examples) < max_examples:
                fp_examples.append({
                    "source1_id": eid,
                    "s1_info": s1_records.get(eid) if s1_records else None,
                    "false_positives": [
                        {"id": x, "info": target_records.get(x) if target_records else None}
                        for x in (pred - gold)
                    ],
                    "true_positives": list(gold & pred),
                    "gold": list(gold)
                })

            if fn > 0 and len(fn_examples) < max_examples:
                fn_examples.append({
                    "source1_id": eid,
                    "s1_info": s1_records.get(eid) if s1_records else None,
                    "false_negatives": [
                        {"id": x, "info": target_records.get(x) if target_records else None}
                        for x in (gold - pred)
                    ],
                    "predicted": list(pred),
                    "gold": list(gold)
                })

    macro_f05 = score_sum / count if count else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 1.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    pair_f05 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total else 1.0
    total_predicted = sum(len(p) for p in predictions.values())

    return {
        "validation_entities": count,
        "macro_f0.5": macro_f05,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f0.5": pair_f05,
        "singleton_accuracy": singleton_acc,
        "singleton_entities": singleton_total,
        "singleton_correct": singleton_correct,
        "total_predicted_matches": total_predicted,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "false_positive_examples": fp_examples,
        "false_negative_examples": fn_examples,
    }
