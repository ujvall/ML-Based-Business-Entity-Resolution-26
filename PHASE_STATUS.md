# Implementation status

## Complete: Phase 0 - rules and reproducibility

- Extracted the supplied challenge and event requirements into `RULES.md` and
  `challenge_config.json`.
- Preserved the original dataset under `student_resource/dataset/`.
- Added pinned Phase 2 dependencies, a README runbook, an experiment log, and
  a portal submission log. No external business lookup data has been used.

## Complete: Phase 1 - audit and validation

- Full audit: 2,206,821 training Source 1 records; 10,320,219 training Source
  2/3 records; 1,732,544 test Source 1 records; no duplicate IDs or truth rows.
- Frozen Source 1 hash split (`seed=2026`): 1,765,800 development and 441,021
  validation entities.
- Implemented exact macro F0.5, including singleton scoring, and a streaming
  validation scorer. Five correctness checks pass.
- Full details: `outputs/phase1_audit.json`.

## Complete: Phase 2 - candidate generation

- Built a reusable local index of 139,518,407 multi-view block postings from
  the training Source 2/3 records, plus a compact name/address ranking sketch.
- Chosen blocking parameters: target posting cap 200; top 150 candidates per
  Source 1 entity; ranking from name/address overlap, numbers, and shared views.
- Full validation: 50,284,307 candidates, 91.71% true-link recall, 0.9654
  oracle macro F0.5 ceiling, 114.02 candidates per Source 1 entity.
- Optional compressed output and per-candidate retrieval-view masks tested on
  405 Source 1 entities; ID/mask alignment passed.

## Next: Phase 3 and later

Build the evidence-ledger matching baseline using the frozen candidate set.
Train and tune only on the provided training labels, then compare actual macro
F0.5 to the Phase 2 ceiling. Do not submit candidates as final matches.
