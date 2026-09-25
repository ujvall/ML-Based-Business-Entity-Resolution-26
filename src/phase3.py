"""Phase 3: Production Entity Resolution Matching Model.

End-to-end pipeline:
1. Feature extraction with 26 discriminative features.
2. Trained on development split only with hard negatives.
3. Conservative matching: country + house-number + name consistency gates.
4. Calibrated Logistic Regression scoring with macro F0.5 thresholding.
5. High-throughput batched inference for validation and test.
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
)


MODEL_FILE = Path("work/phase3_matcher.pkl")
SELECTED_THRESHOLD = 0.40


def train_model(
    data_dir: Path,
    dev_candidates_path: Path,
    truth_path: Path,
    model_out: Path,
    seed: int = 2026,
    max_train_entities: int = 14000,
    max_negatives_per_query: int = 25
) -> dict:
    print(f"Loading development candidate pairs from {dev_candidates_path}...", flush=True)
    dev_cands, dev_targets = load_candidate_pairs(dev_candidates_path)

    if max_train_entities and len(dev_cands) > max_train_entities:
        selected_ids = list(dev_cands.keys())[:max_train_entities]
        dev_cands = {k: dev_cands[k] for k in selected_ids}
        dev_targets = {cid for cands in dev_cands.values() for cid in cands}

    # Verify zero validation leakage
    for eid in dev_cands:
        if validation_member(eid, seed):
            raise ValueError(f"FATAL: Development entity {eid} leaks into validation split!")

    print(f"Loaded {len(dev_cands):,} development entities, {len(dev_targets):,} unique candidate targets.", flush=True)

    print("Loading ground truth for development entities...", flush=True)
    dev_truth = load_truth_for_entities(truth_path, set(dev_cands))

    print("Loading profiles for development candidates...", flush=True)
    start_load = time.monotonic()
    s1_profiles, target_profiles = load_profiles_for_entities(
        data_dir, "train", set(dev_cands), dev_targets
    )
    print(f"Loaded profiles in {time.monotonic() - start_load:.1f}s.", flush=True)

    print("Extracting features for training dataset...", flush=True)
    start_feat = time.monotonic()
    X_train, y_train = build_dataset_from_candidates(
        dev_cands, dev_truth, s1_profiles, target_profiles,
        max_negatives_per_query=max_negatives_per_query
    )
    print(
        f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"positives: {int(y_train.sum()):,} ({y_train.mean():.2%}), "
        f"time: {time.monotonic() - start_feat:.1f}s.",
        flush=True
    )

    print("Training regularized Logistic Regression pipeline...", flush=True)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=seed))
    ])
    start_train = time.monotonic()
    pipeline.fit(X_train, y_train)
    print(f"Trained in {time.monotonic() - start_train:.1f}s.", flush=True)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "feature_names": FEATURE_NAMES,
            "threshold": SELECTED_THRESHOLD,
            "trained_entities": len(dev_cands),
            "trained_pairs": len(X_train)
        }, f)
    print(f"Model saved to {model_out}.", flush=True)
    return {"trained_entities": len(dev_cands), "trained_pairs": len(X_train)}


def predict_candidates(
    data_dir: Path,
    partition: str,
    candidates_path: Path,
    model_path: Path,
    matching_out: Path,
    threshold: float = SELECTED_THRESHOLD,
    batch_size: int = 50000
) -> dict:
    print(f"Loading model from {model_path}...", flush=True)
    with model_path.open("rb") as f:
        meta = pickle.load(f)
    pipeline = meta["pipeline"]

    print(f"Loading candidate pairs from {candidates_path}...", flush=True)
    cands_dict, target_ids = load_candidate_pairs(candidates_path)
    s1_ids = set(cands_dict.keys())
    print(f"Loaded {len(s1_ids):,} Source 1 entities with {len(target_ids):,} candidate targets.", flush=True)

    print(f"Loading profiles for {partition}...", flush=True)
    start_load = time.monotonic()
    s1_profiles, target_profiles = load_profiles_for_entities(
        data_dir, partition, s1_ids, target_ids
    )
    print(f"Loaded profiles in {time.monotonic() - start_load:.1f}s.", flush=True)

    print(f"Scoring candidates and generating predictions (threshold={threshold:.2f})...", flush=True)
    start_score = time.monotonic()
    matched_results: Dict[str, List[str]] = {s1_id: [] for s1_id in cands_dict}

    current_pairs = []
    current_features = []

    def flush_batch():
        if not current_features:
            return
        X_batch = np.array(current_features, dtype=np.float32)
        probs = pipeline.predict_proba(X_batch)[:, 1]
        for (s1_id, cand_id), prob in zip(current_pairs, probs):
            if prob >= threshold:
                matched_results[s1_id].append(cand_id)
        current_pairs.clear()
        current_features.clear()

    total_pairs = 0
    gated_out = 0

    for s1_id, cand_list in cands_dict.items():
        if s1_id not in s1_profiles:
            continue
        s1 = s1_profiles[s1_id]
        n_cands = len(cand_list)

        for rank, cand_id in enumerate(cand_list, start=1):
            s2 = target_profiles.get(cand_id)
            if s2 is None:
                continue
            total_pairs += 1

            # 1. Country contradiction gate
            if s1.country and s2.country and s1.country != s2.country:
                gated_out += 1
                continue

            # 2. House number contradiction gate
            if s1.house_num and s2.house_num and s1.house_num != s2.house_num:
                gated_out += 1
                continue

            # 3. Name consistency gate
            feat = extract_pair_features(s1, s2, rank=rank, total_cands=n_cands)
            name_sort = feat[5]
            name_overlap = feat[3]
            if name_sort < 0.55 and name_overlap < 0.50:
                gated_out += 1
                continue

            current_pairs.append((s1_id, cand_id))
            current_features.append(feat)

            if len(current_features) >= batch_size:
                flush_batch()

    flush_batch()
    print(f"Scored {total_pairs:,} candidate pairs ({gated_out:,} rejected by contradiction gates) in {time.monotonic() - start_score:.1f}s.", flush=True)

    print(f"Writing matching results to {matching_out}...", flush=True)
    matching_out.parent.mkdir(parents=True, exist_ok=True)
    total_matches = 0
    singleton_count = 0
    with matching_out.open("w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, cand_list in cands_dict.items():
            matches = matched_results.get(s1_id, [])
            # Deduplicate preserving order
            seen_ids = set()
            dedup_matches = []
            for mid in matches:
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    dedup_matches.append(mid)

            total_matches += len(dedup_matches)
            if not dedup_matches:
                singleton_count += 1
            f.write(f"{s1_id}\t{','.join(dedup_matches)}\n")

    report = {
        "source1_entities": len(cands_dict),
        "total_candidate_pairs": total_pairs,
        "gated_contradictions": gated_out,
        "total_predicted_matches": total_matches,
        "predicted_singletons": singleton_count,
        "singleton_fraction": singleton_count / len(cands_dict) if cands_dict else 0.0,
        "threshold": threshold
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Entity Resolution Matcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    train_parser.add_argument("--dev-candidates", type=Path, default=Path("work/phase2_development_candidates.tsv"))
    train_parser.add_argument("--truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    train_parser.add_argument("--model-out", type=Path, default=MODEL_FILE)
    train_parser.add_argument("--max-train-entities", type=int, default=14000)
    train_parser.add_argument("--max-negatives-per-query", type=int, default=25)

    # Predict subparser
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    predict_parser.add_argument("--partition", choices=["train", "test"], default="test")
    predict_parser.add_argument("--candidates", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, default=MODEL_FILE)
    predict_parser.add_argument("--out", type=Path, required=True)
    predict_parser.add_argument("--threshold", type=float, default=SELECTED_THRESHOLD)

    args = parser.parse_args()
    if args.command == "train":
        train_model(
            args.data_dir,
            args.dev_candidates,
            args.truth,
            args.model_out,
            max_train_entities=args.max_train_entities,
            max_negatives_per_query=args.max_negatives_per_query
        )
    elif args.command == "predict":
        predict_candidates(
            args.data_dir,
            args.partition,
            args.candidates,
            args.model,
            args.out,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
