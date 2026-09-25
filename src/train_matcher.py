"""Phase 3: Train and evaluate Entity Resolution matching models.

Trains strictly on the development split (no validation labels used for training).
Uses hard negatives from Phase 2 candidate retrieval.
Evaluates on frozen validation split with exact macro F0.5 and threshold tuning.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from phase1 import EXPECTED_SOURCE, EXPECTED_TRUTH, rows, validation_member
from features import EntityProfile, extract_pair_features, FEATURE_NAMES
from evaluate_matcher import evaluate_predictions


def load_candidate_pairs(candidates_path: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    candidates = {}
    unique_targets = set()
    with candidates_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            cands = [x for x in row["candidate_entity_ids"].split(",") if x]
            candidates[s1_id] = cands
            unique_targets.update(cands)
    return candidates, unique_targets


def load_truth_for_entities(truth_path: Path, s1_ids: Set[str]) -> Dict[str, Set[str]]:
    truth = {}
    with truth_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            if s1_id in s1_ids:
                truth[s1_id] = {x for x in row["matched_entity_ids"].split(",") if x}
    return truth


def load_profiles_for_entities(
    data_dir: Path,
    partition: str,
    s1_ids: Set[str],
    target_ids: Set[str]
) -> Tuple[Dict[str, EntityProfile], Dict[str, EntityProfile]]:
    print(f"Loading Source 1 profiles for {len(s1_ids):,} entities...", flush=True)
    s1_profiles = {}
    with (data_dir / partition / f"{partition}_source1.tsv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if eid in s1_ids:
                s1_profiles[eid] = EntityProfile(row["business_name"], row["business_address"], row["country"])
                if len(s1_profiles) == len(s1_ids):
                    break

    print(f"Loading Source 2/3 profiles for {len(target_ids):,} candidate targets...", flush=True)
    target_profiles = {}
    needed = set(target_ids)
    for src in (f"{partition}_source2.tsv", f"{partition}_source3.tsv"):
        if not needed:
            break
        with (data_dir / partition / src).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row["entity_id"]
                if eid in needed:
                    target_profiles[eid] = EntityProfile(row["business_name"], row["business_address"], row["country"])
                    needed.remove(eid)

    return s1_profiles, target_profiles


def build_dataset_from_candidates(
    candidates: Dict[str, List[str]],
    truth: Dict[str, Set[str]],
    s1_profiles: Dict[str, EntityProfile],
    target_profiles: Dict[str, EntityProfile],
    max_negatives_per_query: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and label vector y for candidate pairs.
    Includes all positive pairs and up to max_negatives_per_query hard negatives per query.
    """
    X_rows = []
    y_vals = []

    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_profiles:
            continue
        s1 = s1_profiles[s1_id]
        gold = truth.get(s1_id, set())

        neg_count = 0
        for rank, cand_id in enumerate(cand_list, start=1):
            if cand_id not in target_profiles:
                continue
            s2 = target_profiles[cand_id]
            is_pos = cand_id in gold

            if not is_pos:
                neg_count += 1
                if neg_count > max_negatives_per_query:
                    continue

            feat = extract_pair_features(s1, s2, rank=rank, total_cands=len(cand_list))
            X_rows.append(feat)
            y_vals.append(1.0 if is_pos else 0.0)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return X, y


def score_candidates_with_model(
    model,
    candidates: Dict[str, List[str]],
    s1_profiles: Dict[str, EntityProfile],
    target_profiles: Dict[str, EntityProfile],
    batch_size: int = 50000
) -> Dict[str, List[Tuple[str, float]]]:
    """Score all candidate pairs for validation/test using the trained model."""
    pair_scores = {s1_id: [] for s1_id in candidates}

    # Stream in batches to keep RAM low
    current_pairs = []
    current_features = []

    def flush_batch():
        if not current_features:
            return
        X_batch = np.array(current_features, dtype=np.float32)
        # Probability of class 1
        probs = model.predict_proba(X_batch)[:, 1]
        for (s1_id, cand_id), prob in zip(current_pairs, probs):
            pair_scores[s1_id].append((cand_id, float(prob)))
        current_pairs.clear()
        current_features.clear()

    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_profiles:
            continue
        s1 = s1_profiles[s1_id]
        n_cands = len(cand_list)

        for rank, cand_id in enumerate(cand_list, start=1):
            s2 = target_profiles.get(cand_id)
            if s2 is None:
                continue
            # Fast contradiction pre-filter
            if s1.country and s2.country and s1.country != s2.country:
                continue
            if s1.house_num and s2.house_num and s1.house_num != s2.house_num:
                # Strong house number conflict
                continue

            feat = extract_pair_features(s1, s2, rank=rank, total_cands=n_cands)
            current_pairs.append((s1_id, cand_id))
            current_features.append(feat)

            if len(current_features) >= batch_size:
                flush_batch()

    flush_batch()
    return pair_scores


def tune_threshold(
    pair_scores: Dict[str, List[Tuple[str, float]]],
    truth: Dict[str, Set[str]],
    thresholds: List[float] | None = None
) -> Tuple[float, dict, List[dict]]:
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    best_thresh = 0.50
    best_f05 = -1.0
    best_eval = {}
    sweep = []

    print("\nTuning threshold on validation macro F0.5...", flush=True)
    for thresh in thresholds:
        preds = {
            s1_id: {cid for cid, prob in sc_list if prob >= thresh}
            for s1_id, sc_list in pair_scores.items()
        }
        ev = evaluate_predictions(preds, truth)
        sweep.append({
            "threshold": thresh,
            "macro_f0.5": ev["macro_f0.5"],
            "pair_precision": ev["pair_precision"],
            "pair_recall": ev["pair_recall"],
            "singleton_accuracy": ev["singleton_accuracy"],
            "predicted_matches": ev["total_predicted_matches"]
        })
        print(
            f"Thresh={thresh:.2f}: macro F0.5={ev['macro_f0.5']:.4f}, "
            f"Prec={ev['pair_precision']:.4f}, Rec={ev['pair_recall']:.4f}, "
            f"SingletonAcc={ev['singleton_accuracy']:.4f}, Matches={ev['total_predicted_matches']:,}",
            flush=True
        )
        if ev["macro_f0.5"] > best_f05:
            best_f05 = ev["macro_f0.5"]
            best_thresh = thresh
            best_eval = ev

    return best_thresh, best_eval, sweep
