"""Parallel, memory-safe, ultra-fast streaming candidate matcher.

Features:
- Multi-process parallel evaluation across CPU cores.
- Fast candidate pre-filtering before instantiating full feature objects.
- Country contradiction gate (nanosecond string check).
- House-number contradiction gate (fast token check).
- Name consistency gate (Fast RapidFuzz + token overlap).
- Full 26-feature dense vector extracted ONLY for candidates passing all gates.
- Batched numpy inference with regularized Logistic Regression.
- Output strictly satisfies competition formatting:
  * Exactly 1 line per test Source 1 entity in exact input order
  * Every matched ID exists in test Source 2/3
  * Every matched ID exists in candidate_pairs.tsv
  * Empty matches for singletons
  * No duplicate IDs
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import rapidfuzz.fuzz as fuzz

from features import (
    EntityProfile, extract_pair_features, normalize_tokens,
    NUMERIC, FEATURE_NAMES
)

SELECTED_THRESHOLD = 0.40


def extract_house_num_from_str(addr: str) -> str | None:
    if not addr:
        return None
    for token in addr.split():
        clean = token.strip(" ,.-#/\\")
        if clean.isdigit() and len(clean) <= 6:
            return clean
    return None


def worker_match(
    worker_id: int,
    data_dir_str: str,
    partition: str,
    model_path_str: str,
    candidates_path_str: str,
    start_row: int,
    end_row: int,
    out_part_str: str,
    threshold: float = SELECTED_THRESHOLD,
    batch_size: int = 50000
) -> dict:
    data_dir = Path(data_dir_str)
    print(f"[Worker {worker_id}] Loading matching model...", flush=True)
    with open(model_path_str, "rb") as f:
        meta = pickle.load(f)
    pipeline = meta["pipeline"]

    print(f"[Worker {worker_id}] Loading target records...", flush=True)
    t0 = time.monotonic()
    target_records: Dict[str, Tuple[str, str, str]] = {}
    for src in (f"{partition}_source2.tsv", f"{partition}_source3.tsv"):
        p = data_dir / partition / src
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                target_records[row["entity_id"]] = (
                    row["business_name"],
                    row["business_address"],
                    row["country"].strip().casefold() if row.get("country") else ""
                )
    print(f"[Worker {worker_id}] Loaded {len(target_records):,} targets in {time.monotonic()-t0:.1f}s.", flush=True)

    # Load only this worker's slice of Source 1 profiles
    print(f"[Worker {worker_id}] Loading Source 1 slice (rows {start_row:,} to {end_row:,})...", flush=True)
    t0 = time.monotonic()
    s1_profiles: Dict[str, EntityProfile] = {}
    s1_path = data_dir / partition / f"{partition}_source1.tsv"
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            if idx < start_row:
                continue
            if idx >= end_row:
                break
            parts = line.rstrip("\r\n").split("\t")
            s1_profiles[parts[0]] = EntityProfile(
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "",
                parts[3] if len(parts) > 3 else ""
            )
    print(f"[Worker {worker_id}] Loaded {len(s1_profiles):,} S1 profiles in {time.monotonic()-t0:.1f}s.", flush=True)

    print(f"[Worker {worker_id}] Streaming and matching candidates...", flush=True)
    t_start = time.monotonic()
    total_queries = 0
    total_matches = 0
    total_singletons = 0
    gated_out = 0

    batch_pairs = []
    batch_features = []
    open_query_matches: Dict[str, List[str]] = {}

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

    with open(candidates_path_str, "r", encoding="utf-8", newline="") as cand_file, \
         open(out_part_str, "w", encoding="utf-8", newline="") as out_file:

        header = cand_file.readline()
        for idx, line in enumerate(cand_file):
            if idx < start_row:
                continue
            if idx >= end_row:
                break

            total_queries += 1
            s1_id, sep, cands_str = line.partition("\t")
            s1_id = s1_id.strip()
            cand_list = cands_str.strip().split(",") if cands_str.strip() else []

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

                # Gate 1: Country contradiction
                if s1.country and t_country and s1.country != t_country:
                    gated_out += 1
                    continue

                # Gate 2: House number contradiction
                if s1.house_num:
                    t_num = extract_house_num_from_str(t_addr)
                    if t_num and s1.house_num != t_num:
                        gated_out += 1
                        continue

                # Gate 3: Fast name consistency check
                t_tokens = normalize_tokens(t_name)
                t_set = set(t_tokens)
                inter = len(s1.name_set & t_set)
                min_len = min(len(s1.name_set), len(t_set))
                name_overlap = inter / min_len if min_len else 0.0
                t_name_clean = " ".join(t_tokens)
                name_sort = fuzz.token_sort_ratio(s1.name_clean, t_name_clean) / 100.0

                if name_sort < 0.55 and name_overlap < 0.50:
                    gated_out += 1
                    continue

                # Full profile and feature extraction ONLY for survivors
                s2 = EntityProfile(t_name, t_addr, t_country)
                feat = extract_pair_features(s1, s2, rank=rank, total_cands=n_cands)

                batch_pairs.append((s1_id, cand_id))
                batch_features.append(feat)

                if len(batch_features) >= batch_size:
                    flush_inference()

            # Flush periodically to keep memory bounded
            if len(open_query_matches) > 20000:
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
                    out_file.write(f"{qid}\t{','.join(dedup)}\n")

            if total_queries % 50000 == 0:
                elapsed = time.monotonic() - t_start
                rate = total_queries / elapsed if elapsed else 0.0
                print(
                    f"[Worker {worker_id}] Processed {total_queries:,}/{end_row-start_row:,} "
                    f"({total_matches:,} matches, {total_singletons:,} singletons, {gated_out:,} gated) [{rate:.0f} q/s]",
                    flush=True
                )

        flush_inference()
        for qid, mids in open_query_matches.items():
            seen = set()
            dedup = [m for m in mids if not (m in seen or seen.add(m))]
            total_matches += len(dedup)
            if not dedup:
                total_singletons += 1
            out_file.write(f"{qid}\t{','.join(dedup)}\n")
        open_query_matches.clear()

    total_time = round(time.monotonic() - t_start, 2)
    print(f"[Worker {worker_id}] FINISHED {total_queries:,} queries in {total_time}s!", flush=True)
    return {
        "worker_id": worker_id,
        "queries": total_queries,
        "matches": total_matches,
        "singletons": total_singletons,
        "gated_out": gated_out,
        "seconds": total_time
    }


def run_parallel_matching(
    data_dir: Path,
    partition: str,
    candidates_path: Path,
    model_path: Path,
    out_path: Path,
    threshold: float = SELECTED_THRESHOLD,
    n_workers: int = 3
) -> dict:
    s1_path = data_dir / partition / f"{partition}_source1.tsv"
    with open(s1_path, "r", encoding="utf-8") as f:
        n_total = sum(1 for _ in f) - 1
    print(f"Total Source 1 entities to match: {n_total:,}", flush=True)

    chunk_size = (n_total + n_workers - 1) // n_workers
    temp_dir = out_path.parent / "temp_match_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    part_files = []
    worker_args = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)
        part_file = temp_dir / f"match_part_{i}.tsv"
        part_files.append(str(part_file))
        worker_args.append((
            i, str(data_dir), partition, str(model_path), str(candidates_path),
            start, end, str(part_file), threshold
        ))
        print(f"Worker {i}: rows {start:,} to {end:,} ({end - start:,} queries) -> {part_file.name}", flush=True)

    print(f"\nLaunching {n_workers} parallel matching workers...", flush=True)
    start_time = time.monotonic()

    with mp.Pool(processes=n_workers) as pool:
        async_results = [pool.apply_async(worker_match, args=arg) for arg in worker_args]
        results = [res.get() for res in async_results]

    total_time = round(time.monotonic() - start_time, 2)
    print(f"\nAll {n_workers} workers finished in {total_time}s! Merging results into {out_path}...", flush=True)

    total_written = 0
    total_matches = 0
    total_singletons = 0
    seen_ids = set()

    with open(out_path, "w", encoding="utf-8", newline="") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for part_p in part_files:
            with open(part_p, "r", encoding="utf-8") as f:
                for line in f:
                    s1_id, sep, mids = line.partition("\t")
                    if s1_id in seen_ids:
                        raise ValueError(f"Duplicate entity ID in output: {s1_id}")
                    seen_ids.add(s1_id)
                    out.write(line)
                    total_written += 1
                    clean_mids = mids.strip().split(",") if mids.strip() else []
                    total_matches += len(clean_mids)
                    if not clean_mids:
                        total_singletons += 1
            try:
                os.remove(part_p)
            except OSError:
                pass

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    assert total_written == n_total, f"Expected {n_total} rows, wrote {total_written}"
    print(
        f"Successfully generated {out_path}: {total_written:,} entities, "
        f"{total_matches:,} matches, {total_singletons:,} singletons ({total_singletons/total_written:.2%}) in {total_time}s.",
        flush=True
    )

    report = {
        "partition": partition,
        "source1_entities": total_written,
        "total_predicted_matches": total_matches,
        "predicted_singletons": total_singletons,
        "singleton_fraction": total_singletons / total_written if total_written else 0.0,
        "threshold": threshold,
        "total_seconds": total_time,
        "n_workers": n_workers
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    report = run_parallel_matching(
        args.data_dir,
        args.partition,
        args.candidates,
        args.model,
        args.out,
        args.threshold,
        args.workers
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
