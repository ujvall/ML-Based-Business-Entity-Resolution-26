# Phase 2 experiment log

All results below use the same deterministic 13,907-entity sample of the
20% Source 1 validation holdout (`seed=2026`, `sample_modulus=32`) and the full
10,320,219-record training target pool. No ground-truth labels are used in
candidate generation; they are read only by the independent evaluator.

| Version | Method | Posting cap | Top K | True-link recall | Avg candidates | Oracle macro F0.5 ceiling | Retrieval time |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 | Two-pass batch scanner | 200 | 100 | 0.9039 | 83.4 | 0.9582 | 1077 s |
| I1 | Indexed, block weights | 200 | 100 | 0.8821 | 83.4 | 0.9482 | 29 s |
| I2 | Indexed, block weights | 500 | 150 | 0.8868 | 126.5 | 0.9517 | 31 s |
| S1 | Indexed, name/address sketch | 200 | 100 | 0.9040 | 83.4 | 0.9584 | 63 s |
| S2 | Indexed, name/address sketch | 500 | 100 | 0.8984 | 89.0 | 0.9560 | 98 s |
| S3 | Indexed, name/address sketch | 200 | 150 | **0.9161** | 114.1 | **0.9650** | 65 s |

Decision: S3 is the Phase 2 validation setting. A larger posting cap lowered
recall at fixed top K because weak candidates displaced true links.

## Full holdout confirmation

The S3 setting was run against all 441,021 held-out Source 1 entities and the
complete 10,320,219-record target pool. It produced 50,284,307 candidate pairs
(114.02 per Source 1 entity), retrieved 1,400,576 of 1,527,143 true links
(**0.9171 recall**), and achieved an **oracle macro F0.5 ceiling of 0.9654**.
Reduction ratio was 0.99998895. Retrieval took 1,615 seconds after the index
and sketch were built. The full result is in `outputs/phase2_full_validation_quality.json`.

The 24,511 true singletons are especially important for the next phase:
only 19 had zero candidates. The matching model must learn to abstain when
all retrieved candidates are weak; candidate existence alone is not a match.

The oracle ceiling assumes a perfect matching model that selects exactly the
true links that Phase 2 retrieved. It is not an achieved leaderboard score.

## Phase 3 matching model experiment log

All models trained exclusively on development Source 1 split (`seed=2026`, zero validation label leakage) using hard negatives from Phase 2 retrieval. Evaluated on the frozen 13,907 validation holdout.

| Model / Experiment | Features / Architecture | Threshold | Pair Precision | Pair Recall | Singleton Accuracy | Validation Macro F0.5 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| M0: Baseline heuristic | Handcrafted composite name/addr/sketch | 0.75 | 0.5748 | 0.6195 | 0.6220 | 0.6619 | Simple heuristic rule baseline |
| M1: HistGradientBoosting | 26 features, max_iter=150 | 0.90 | 0.2806 | 0.6770 | 0.4241 | 0.5293 | Over-predicts on validation imbalance |
| M2: Logistic Regression | 26 features, StandardScaler + L2 | 0.50 | 0.9112 | 0.6744 | 0.7940 | 0.7791 | Strong linear log-odds calibration |
| M3: Logistic Regression + Contradiction Gates | 26 features + country/house-no/name gates | **0.40** | **0.9241** | **0.6724** | **0.8117** | **0.7837** | **Selected Model: +0.1218 over M0** |

Decision: M3 (Regularized Logistic Regression with conservative country, house number, and name contradiction gates at threshold 0.40) achieves highest macro F0.5 (0.7837), outstanding precision (92.41%), and strong singleton abstention accuracy (81.17%).

