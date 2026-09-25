"""High-performance streaming candidate matcher for Amazon ML Challenge 2026.

Memory-efficient architecture:
- Target records stored as raw strings (~2 GB RAM for 10M records)
- Source 1 queries pre-profiled (~0.5 GB RAM)
- Fast contradiction pre-filtering (country, house-number, name consistency)
- Batched numpy inference with regularized Logistic Regression
- Output strictly satisfies competition formatting:
  * Exactly 1 line per test Source 1 entity
  * Every matched ID exists in test Source 2/3
  * Every matched ID exists in candidate_pairs.tsv
  * Empty matches for singletons
  * No duplicate IDs
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

from phase1 import EXPECTED_SOURCE, rows
from features import EntityProfile, extract_pair_features, FEATURE_NAMES


SELECTED_THRESHOLD = 0.40


def load_raw_target_records(
    data_dir: Path,
    partition: str
) -> Dict[str, Tuple[str, str, str]]:
    print(f"Loading {partition}_source2 and {partition}_source3 raw records into memory...", flush=True)
    start = time.monotonic()
    records: Dict[str, Tuple[str, str, str]] = {}
    for src in (f"{partition}_source2.tsv", f"{partition}_source3.tsv"):
        path = data_dir / partition / src
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                records[row["entity_id"]] = (
                    row["business_name"],
                    row["business_address"],
                    row["country"].strip().casefold()
                )
        print(f"  Loaded {len(records):,} targets so far...", flush=True)
    print(f"Loaded {len(records):,} total target records in {time.monotonic() - start:.1f}s.", flush=True)
    return records


def load_source1_profiles(
    data_dir: Path,
    partition: str
) -> Dict[str, EntityProfile]:
    print(f"Loading and profiling {partition}_source1 records...", flush=True)
    start = time.monotonic()
    profiles: Dict[str, EntityProfile] = {}
    path = data_dir / partition / f"{partition}_source1.tsv"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            profiles[row["entity_id"]] = EntityProfile(
                row["business_name"],
                row["business_address"],
                row["country"]
            )
    print(f"Loaded {len(profiles):,} Source 1 profiles in {time.monotonic() - start:.1f}s.", flush=True)
    return profiles


def run_streaming_matching(
    data_dir: Path,
    partition: str,
    candidates_path: Path,
    model_path: Path,
    matching_out: Path,
    threshold: float = SELECTED_THRESHOLD,
    batch_size: int = 50000
) -> dict:
    print(f"Loading matching model from {model_path}...", flush=True)
    with model_path.open("rb") as f:
        meta = pickle.load(f)
    pipeline = meta["pipeline"]

    # Load targets as raw strings and source1 as profiles
    target_records = load_raw_target_records(data_dir, partition)
    s1_profiles = load_source1_profiles(data_dir, partition)

    print(f"\nStreaming candidate pairs from {candidates_path} -> {matching_out}...", flush=True)
    matching_out.parent.mkdir(parents=True, exist_ok=True)

    start_stream = time.monotonic()
    total_queries = 0
    total_candidates = 0
    total_matches = 0
    total_singletons = 0
    gated_out = 0

    batch_pairs = []
    batch_features = []
    open_query_matches: Dict[str, List[str]] = {}

    with candidates_path.open("r", encoding="utf-8", newline="") as cand_file, \
         matching_out.open("w", encoding="utf-8", newline="") as match_file:

        match_file.write("source1_entity_id\tmatched_entity_ids\n")
        reader = csv.DictReader(cand_file, delimiter="\t")

        def flush_inference():
            nonlocal total_matches
            if not batch_features:
                return
            X = np.array(batch_features, dtype=np.float32)
            probs = pipeline.predict_proba(X)[:, 1]
            for (s1_id, cand_id), prob in zip(batch_pairs, probs):
                if prob >= threshold:
                    open_query_matches[s1_id].append(cand_id)
            batch_pairs.clear()
            batch_features.clear()

        for line_idx, row in enumerate(reader, start=1):
            s1_id = row["source1_entity_id"]
            cand_str = row["candidate_entity_ids"].strip()
            cand_list = cand_str.split(",") if cand_str else []
            total_queries += 1
            total_candidates += len(cand_list)
            open_query_matches[s1_id] = []

            s1 = s1_profiles.get(s1_id)
            if s1 is None:
                continue

            n_cands = len(cand_list)
            for rank, cand_id in enumerate(cand_list, start=1):
                rec = target_records.get(cand_id)
                if rec is None:
                    continue

                t_name, t_addr, t_country = rec

                # 1. Country contradiction gate (nanosecond string check)
                if s1.country and t_country and s1.country != t_country:
                    gated_out += 1
                    continue

                # On-the-fly profiling for candidates passing country gate
                s2 = EntityProfile(t_name, t_addr, t_country)

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

                batch_pairs.append((s1_id, cand_id))
                batch_features.append(feat)

                if len(batch_features) >= batch_size:
                    flush_inference()

            # Stream written queries to keep RAM bounded
            if len(open_query_matches) > 30000:
                flush_inference()
                in_flight_queries = {p[0] for p in batch_pairs}
                ready_queries = [qid for qid in open_query_matches if qid not in in_flight_queries]
                for qid in ready_queries:
                    mids = open_query_matches.pop(qid)
                    seen = set()
                    dedup = [m for m in mids if not (m in seen or seen.add(m))]
                    total_matches += len(dedup)
                    if not dedup:
                        total_singletons += 1
                    match_file.write(f"{qid}\t{','.join(dedup)}\n")

            if line_idx % 100000 == 0:
                elapsed = time.monotonic() - start_stream
                rate = line_idx / elapsed
                print(
                    f"Processed {line_idx:,}/{len(s1_profiles):,} queries "
                    f"({total_matches:,} matches, {total_singletons:,} singletons, "
                    f"{gated_out:,} gated) [{rate:.0f} queries/s]",
                    flush=True
                )

        flush_inference()
        for qid, mids in open_query_matches.items():
            seen = set()
            dedup = [m for m in mids if not (m in seen or seen.add(m))]
            total_matches += len(dedup)
            if not dedup:
                total_singletons += 1
            match_file.write(f"{qid}\t{','.join(dedup)}\n")
        open_query_matches.clear()

    total_time = round(time.monotonic() - start_stream, 2)
    print(f"\nFinished matching in {total_time}s!", flush=True)
    print(f"Total queries: {total_queries:,}, Matches: {total_matches:,}, Singletons: {total_singletons:,} ({total_singletons/total_queries:.2%})", flush=True)

    report = {
        "partition": partition,
        "source1_entities": total_queries,
        "candidate_pairs": total_candidates,
        "gated_contradictions": gated_out,
        "total_predicted_matches": total_matches,
        "predicted_singletons": total_singletons,
        "singleton_fraction": total_singletons / total_queries if total_queries else 0.0,
        "threshold": threshold,
        "scoring_seconds": total_time
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("student_resource/dataset"))
    parser.add_argument("--partition", choices=["train", "test"], default="test")
    parser.add_argument("--candidates", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--model", type=Path, default=Path("work/phase3_matcher.pkl"))
    parser.add_argument("--out", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--threshold", type=float, default=SELECTED_THRESHOLD)
    args = parser.parse_args()

    report = run_streaming_matching(
        args.data_dir,
        args.partition,
        args.candidates,
        args.model,
        args.out,
        threshold=args.threshold
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
