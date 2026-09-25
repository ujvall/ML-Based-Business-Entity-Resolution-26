"""Baseline heuristic matcher for Amazon ML Challenge 2026.

Uses engineered similarity features and conservative filtering
with threshold tuning on macro F0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from phase1 import EXPECTED_SOURCE, EXPECTED_TRUTH, rows, validation_member
from features import EntityProfile, extract_pair_features, FEATURE_NAMES
from evaluate_matcher import evaluate_predictions


def load_profiles(
    path: Path,
    needed_ids: Set[str] | None = None
) -> Dict[str, EntityProfile]:
    profiles = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            if needed_ids is None or eid in needed_ids:
                profiles[eid] = EntityProfile(
                    row["business_name"],
                    row["business_address"],
                    row["country"]
                )
    return profiles


def baseline_score(s1: EntityProfile, s2: EntityProfile) -> float:
    """Compute heuristic matching score between two entities.
    Returns -1.0 if hard contradiction is detected.
    """
    # Hard contradiction: country mismatch
    if s1.country and s2.country and s1.country != s2.country:
        return -1.0

    # Hard contradiction: house number mismatch
    if s1.house_num and s2.house_num and s1.house_num != s2.house_num:
        return -1.0

    # Extract features
    vec = extract_pair_features(s1, s2, rank=1, total_cands=1)
    f = dict(zip(FEATURE_NAMES, vec))

    # Severe name mismatch
    if f["name_token_sort_ratio"] < 0.60 and f["name_exact"] == 0:
        return -1.0

    # Compute composite score
    name_score = 0.50 * f["name_token_sort_ratio"] + 0.30 * f["name_token_set_ratio"] + 0.20 * f["name_jaccard"]
    if f["addr_missing"]:
        addr_score = 0.65  # neutral prior when address is missing
    else:
        addr_score = 0.60 * f["addr_token_sort_ratio"] + 0.40 * f["addr_jaccard"]

    score = 0.60 * name_score + 0.40 * addr_score
    if f["name_exact"]:
        score = max(score, 0.90)
    return score


def run_baseline(
    data_dir: Path,
    candidates_path: Path,
    truth_path: Path,
    seed: int = 2026,
    thresholds: List[float] | None = None
) -> dict:
    if thresholds is None:
        thresholds = [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

    print("Loading candidate pairs...", flush=True)
    candidates: Dict[str, List[str]] = {}
    needed_s23 = set()
    with candidates_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            cands = [x for x in row["candidate_entity_ids"].split(",") if x]
            candidates[s1_id] = cands
            needed_s23.update(cands)

    print(f"Loaded {len(candidates):,} Source 1 entities with {len(needed_s23):,} unique candidate targets.", flush=True)

    print("Loading ground truth...", flush=True)
    truth: Dict[str, Set[str]] = {}
    with truth_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["source1_entity_id"]
            if eid in candidates:
                truth[eid] = {x for x in row["matched_entity_ids"].split(",") if x}

    print("Loading Source 1 profiles...", flush=True)
    s1_profiles = load_profiles(data_dir / "train" / "train_source1.tsv", set(candidates))

    print("Loading Source 2/3 profiles...", flush=True)
    target_profiles = {}
    for src in ("train_source2.tsv", "train_source3.tsv"):
        p = load_profiles(data_dir / "train" / src, needed_s23)
        target_profiles.update(p)

    print(f"Loaded {len(target_profiles):,} target profiles.", flush=True)

    print("Scoring candidate pairs...", flush=True)
    start_time = time.monotonic()
    # Cache scores: dict of s1_id -> list of (cand_id, score)
    pair_scores: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cand_list in candidates.items():
        s1 = s1_profiles[s1_id]
        scores = []
        for cand_id in cand_list:
            s2 = target_profiles.get(cand_id)
            if s2 is None:
                continue
            sc = baseline_score(s1, s2)
            if sc > 0:
                scores.append((cand_id, sc))
        pair_scores[s1_id] = scores

    print(f"Scored in {time.monotonic() - start_time:.1f}s.", flush=True)

    best_thresh = 0.80
    best_f05 = -1.0
    best_eval = {}
    sweep_results = []

    print("Tuning threshold on validation macro F0.5...", flush=True)
    for thresh in thresholds:
        preds = {
            s1_id: {cid for cid, sc in sc_list if sc >= thresh}
            for s1_id, sc_list in pair_scores.items()
        }
        ev = evaluate_predictions(preds, truth)
        sweep_results.append({
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

    # Re-run best with error examples
    best_preds = {
        s1_id: {cid for cid, sc in sc_list if sc >= best_thresh}
        for s1_id, sc_list in pair_scores.items()
    }
    s1_rec_map = {k: (v.raw_name, v.raw_addr, v.country) for k, v in s1_profiles.items()}
    tgt_rec_map = {k: (v.raw_name, v.raw_addr, v.country) for k, v in target_profiles.items()}
    best_eval = evaluate_predictions(best_preds, truth, s1_rec_map, tgt_rec_map, max_examples=5)

    result = {
        "model": "baseline_heuristic_matcher",
        "best_threshold": best_thresh,
        "best_macro_f0.5": best_eval["macro_f0.5"],
        "metrics": best_eval,
        "threshold_sweep": sweep_results
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--candidates", type=Path, default=Path("work/indexed_sketch_200_150.tsv"))
    parser.add_argument("--truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline_experiment.json"))
    args = parser.parse_args()

    result = run_baseline(args.data_dir, args.candidates, args.truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nBest Result:")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
