# Amazon ML Challenge 2026 — progress and handoff

Status: Phases 0, 1, and 2 are complete. Phase 3 (training and evaluating the matching model) has **not** begun. There is no trained matcher, final test prediction, leaderboard score, or portal submission yet. This package is a working handoff, not a finished competition entry.

## What this challenge requires

For every Source 1 business, identify zero or more matching entities from Source 2 and Source 3. The official metric is macro F0.5 per Source 1 entity, including entities with no match. Final files must be tab-separated `matching_results.tsv` and `candidate_pairs.tsv`, cover every test Source 1 ID exactly once, use only valid target IDs, and keep every final match within its candidate list. The supplied guidelines prohibit external business lookups, geocoding, and internet augmentation; model license and size restrictions also apply. See `RULES.md`, the two supplied PDFs, and the official validator for the complete requirements.

## Completed work

| Phase | Result | Main evidence |
|---|---|---|
| 0 — rules and setup | Requirements captured; reproducible configuration and runbook created. | `RULES.md`, `challenge_config.json`, `README.md` |
| 1 — data and evaluation | Full training/test audit, fixed Source 1 holdout, exact macro F0.5 scorer, and correctness tests. | `outputs/phase1_audit.json`, `src/phase1.py`, `src/score_validation.py`, `tests/test_phase1_phase2.py` |
| 2 — retrieval | Local-only, multi-view blocking index and candidate ranking; full held-out candidate evaluation. | `src/phase2_indexed.py`, `outputs/phase2_full_validation_report.json`, `outputs/phase2_full_validation_quality.json`, `EXPERIMENT_LOG.md` |

The audit found 2,206,821 training Source 1 records, 10,320,219 training Source 2/3 records, and 1,732,544 test Source 1 records. The deterministic validation split (`seed=2026`) has 1,765,800 development and 441,021 held-out Source 1 entities. Test includes France, which is absent from training, so country handling must stay open-set.

The selected Phase 2 setting caps block postings at 200 and retains up to 150 candidates per Source 1 entity. On the **full validation holdout**, it generated 50,284,307 candidate pairs (114.02 per entity), retrieved 1,400,576 of 1,527,143 true links (91.71% recall), and produced an **oracle macro F0.5 ceiling of 0.9654**. This ceiling assumes a perfect matcher on retrieved candidates; **it is not an achieved model score or leaderboard result**. In particular, 24,511 validation entities are true singletons, and only 19 of those had no candidates. Phase 3 must learn to abstain, not treat every candidate as a match.

Five metric/blocking tests passed. The full candidate file was independently checked for validation coverage, duplicate IDs, and recall. Small probes also verified compressed output, retrieval-view evidence alignment, and plain TSV output.

## Package contents

The handoff ZIP includes the supplied `student_resource/` data and official utility, both supplied challenge PDFs, root documentation/configuration, `src/` code, `tests/`, all `outputs/` reports and phase-plan PDF, and `work/` experiment files including the full validation candidates. It excludes only duplicate extracted data already represented in the ZIP, the ~2 GB regenerable `work/block_index/` cache, installed `work/python_packages/`, and macOS/bytecode artifacts. Build the index again with the README commands if needed. Keep the package private if competition data-sharing rules require it.

## Continue in order

1. Extract the ZIP and read `RULES.md`, `README.md`, the supplied problem statement and guidelines, and this handoff. Confirm any portal-specific dates or requirements before submitting.
2. Run `python -m unittest discover -s tests` and inspect `outputs/phase1_audit.json` and the Phase 2 quality report.
3. Build Phase 3: derive pair features from only the supplied records; train using **development** Source 1 labels; score on the frozen validation Source 1 IDs; choose a precision-oriented threshold using actual macro F0.5, with explicit empty-list prediction.
4. Compare the actual score with the Phase 2 ceiling, diagnose false merges and misses, and log each deliberate experiment. Avoid broad, expensive sweeps.
5. Freeze the model and regenerate candidates for test with a separate test index. Produce both required plain TSVs, run the official validator in `student_resource/utils/validate_submission.py`, and preserve checksums and the final approach document.
6. Upload only after local validation; record the portal result in `SUBMISSION_LOG.md`. The supplied event rules allow at most five submissions per day.

## Important boundaries

No external data has been used. No test ground truth is present. Candidate retrieval alone is not a final matching solution. The full validation candidate file is for Phase 3 development; validation labels must not be used to train the pair classifier. The supplied phase-plan PDF describes the intended remaining phases, not completed work.
