"""Phase 3 Experiments: Train, validate, tune threshold, and evaluate ER models.

Strict adherence to competition rules:
- Train ONLY on development split
- Zero validation label leakage
- Evaluate using exact macro F0.5
- Detailed error analysis on false merges / false negatives / singletons
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phase1 import EXPECTED_SOURCE, EXPECTED_TRUTH, rows, validation_member
from features import EntityProfile, extract_pair_features, FEATURE_NAMES
from evaluate_matcher import evaluate_predictions
from train_matcher import (
    load_candidate_pairs,
    load_truth_for_entities,
    load_profiles_for_entities,
    build_dataset_from_candidates,
    score_candidates_with_model,
    tune_threshold,
)


def run_experiments(
    data_dir: Path,
    dev_candidates_path: Path,
    val_candidates_path: Path,
    truth_path: Path,
    output_dir: Path,
    seed: int = 2026,
    max_train_entities: int = 8000,
    max_negatives_per_query: int = 25
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_log = {}

    print("=== Step 1: Loading Development and Validation Candidate Pairs ===", flush=True)
    dev_cands, dev_targets = load_candidate_pairs(dev_candidates_path)
    val_cands, val_targets = load_candidate_pairs(val_candidates_path)

    # Subsample development entities if requested
    if max_train_entities and len(dev_cands) > max_train_entities:
        selected_dev_ids = list(dev_cands.keys())[:max_train_entities]
        dev_cands = {k: dev_cands[k] for k in selected_dev_ids}
        dev_targets = {cid for cands in dev_cands.values() for cid in cands}

    print(f"Development entities: {len(dev_cands):,}, unique targets: {len(dev_targets):,}", flush=True)
    print(f"Validation entities: {len(val_cands):,}, unique targets: {len(val_targets):,}", flush=True)

    # Double check zero overlap between dev and val Source 1 IDs
    overlap_ids = set(dev_cands) & set(val_cands)
    if overlap_ids:
        raise ValueError(f"CRITICAL ERROR: {len(overlap_ids)} entities overlap between dev and val!")
    for eid in dev_cands:
        if validation_member(eid, seed):
            raise ValueError(f"CRITICAL ERROR: Dev entity {eid} is in validation split!")
    print("Verification passed: ZERO validation leakage.", flush=True)

    print("\n=== Step 2: Loading Ground Truth ===", flush=True)
    all_s1 = set(dev_cands) | set(val_cands)
    truth = load_truth_for_entities(truth_path, all_s1)
    dev_truth = {k: truth[k] for k in dev_cands if k in truth}
    val_truth = {k: truth[k] for k in val_cands if k in truth}

    print("\n=== Step 3: Loading Entity Profiles ===", flush=True)
    start_load = time.monotonic()
    all_targets = dev_targets | val_targets
    s1_profiles, target_profiles = load_profiles_for_entities(
        data_dir, "train", all_s1, all_targets
    )
    print(f"Profiles loaded in {time.monotonic() - start_load:.1f}s.", flush=True)

    print("\n=== Step 4: Extracting Features for Training Set ===", flush=True)
    start_feat = time.monotonic()
    X_train, y_train = build_dataset_from_candidates(
        dev_cands, dev_truth, s1_profiles, target_profiles,
        max_negatives_per_query=max_negatives_per_query
    )
    print(
        f"Built X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, "
        f"positives: {int(y_train.sum()):,} ({y_train.mean():.2%}), "
        f"in {time.monotonic() - start_feat:.1f}s.",
        flush=True
    )

    models_to_test = {}

    # Model 1: Logistic Regression
    print("\n--- Training Model 1: Logistic Regression ---", flush=True)
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=300, random_state=seed))
    ])
    start_train = time.monotonic()
    lr_pipeline.fit(X_train, y_train)
    print(f"Logistic Regression trained in {time.monotonic() - start_train:.1f}s.", flush=True)
    models_to_test["LogisticRegression"] = lr_pipeline

    # Model 2: HistGradientBoostingClassifier (Default)
    print("\n--- Training Model 2: HistGradientBoostingClassifier ---", flush=True)
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed
    )
    start_train = time.monotonic()
    hgb_clf.fit(X_train, y_train)
    print(f"HistGradientBoostingClassifier trained in {time.monotonic() - start_train:.1f}s.", flush=True)
    models_to_test["HistGradientBoosting"] = hgb_clf

    # Model 3: HistGradientBoosting with deeper trees
    print("\n--- Training Model 3: HistGradientBoosting (Deep / Regularized) ---", flush=True)
    hgb_deep = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=2.0,
        learning_rate=0.08,
        random_state=seed
    )
    start_train = time.monotonic()
    hgb_deep.fit(X_train, y_train)
    print(f"HistGradientBoosting (Deep) trained in {time.monotonic() - start_train:.1f}s.", flush=True)
    models_to_test["HistGradientBoosting_Deep"] = hgb_deep

    print("\n=== Step 5: Evaluating Models on Validation Split ===", flush=True)
    results = {}
    best_overall_model_name = None
    best_overall_f05 = -1.0
    best_overall_threshold = 0.5
    best_overall_eval = {}

    for name, model in models_to_test.items():
        print(f"\nEvaluating {name}...", flush=True)
        start_score = time.monotonic()
        pair_scores = score_candidates_with_model(
            model, val_cands, s1_profiles, target_profiles
        )
        print(f"Scored {len(val_cands):,} entities in {time.monotonic() - start_score:.1f}s.", flush=True)

        best_thresh, best_eval, sweep = tune_threshold(pair_scores, val_truth)
        results[name] = {
            "best_threshold": best_thresh,
            "best_macro_f0.5": best_eval["macro_f0.5"],
            "best_metrics": best_eval,
            "sweep": sweep
        }
        print(f"\n>> {name} Best Threshold: {best_thresh:.2f}, Macro F0.5: {best_eval['macro_f0.5']:.4f}")
        print(f"   Precision: {best_eval['pair_precision']:.4f}, Recall: {best_eval['pair_recall']:.4f}")
        print(f"   Singleton Accuracy: {best_eval['singleton_accuracy']:.4f}")
        print(f"   Matches: {best_eval['total_predicted_matches']:,}")

        if best_eval["macro_f0.5"] > best_overall_f05:
            best_overall_f05 = best_eval["macro_f0.5"]
            best_overall_model_name = name
            best_overall_threshold = best_thresh
            best_overall_eval = best_eval

    # Save model artifacts
    best_model = models_to_test[best_overall_model_name]
    model_path = output_dir / "best_matching_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({
            "model_name": best_overall_model_name,
            "model": best_model,
            "threshold": best_overall_threshold,
            "feature_names": FEATURE_NAMES
        }, f)
    print(f"\nSaved best model ({best_overall_model_name}) to {model_path}", flush=True)

    summary = {
        "best_model_name": best_overall_model_name,
        "best_threshold": best_overall_threshold,
        "best_macro_f0.5": best_overall_f05,
        "validation_entities": best_overall_eval["validation_entities"],
        "pair_precision": best_overall_eval["pair_precision"],
        "pair_recall": best_overall_eval["pair_recall"],
        "singleton_accuracy": best_overall_eval["singleton_accuracy"],
        "total_predicted_matches": best_overall_eval["total_predicted_matches"],
        "false_positive_examples": best_overall_eval["false_positive_examples"][:3],
        "false_negative_examples": best_overall_eval["false_negative_examples"][:3],
        "models_evaluated": {
            k: {
                "best_threshold": v["best_threshold"],
                "best_macro_f0.5": v["best_macro_f0.5"],
                "pair_precision": v["best_metrics"]["pair_precision"],
                "pair_recall": v["best_metrics"]["pair_recall"],
                "singleton_accuracy": v["best_metrics"]["singleton_accuracy"],
                "matches": v["best_metrics"]["total_predicted_matches"]
            }
            for k, v in results.items()
        }
    }

    report_path = output_dir / "phase3_experiment_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {report_path}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--dev-candidates", type=Path, default=Path("work/phase2_development_candidates.tsv"))
    parser.add_argument("--val-candidates", type=Path, default=Path("work/indexed_sketch_200_150.tsv"))
    parser.add_argument("--truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-train-entities", type=int, default=10000)
    parser.add_argument("--max-negatives-per-query", type=int, default=25)
    args = parser.parse_args()

    summary = run_experiments(
        args.data_dir,
        args.dev_candidates,
        args.val_candidates,
        args.truth,
        args.out_dir,
        max_train_entities=args.max_train_entities,
        max_negatives_per_query=args.max_negatives_per_query
    )
    print("\nFINAL EXPERIMENT SUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
