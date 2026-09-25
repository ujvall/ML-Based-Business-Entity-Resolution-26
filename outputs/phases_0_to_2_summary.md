# Amazon ML Challenge 2026: Phases 0-2 completed

## Outcome

The rules, full data audit, deterministic validation protocol, and local-only
candidate-generation pipeline are implemented. No leaderboard upload has been
made, and no external business lookup data has been used.

## Phase 0 - rules and reproducibility

- Inputs and outputs are TSV with an explicit tab delimiter.
- Every test Source 1 entity must eventually appear exactly once in both
  required outputs. Final matches must be a subset of candidate IDs.
- The challenge prohibits external business lookup, geocoding, and other
  internet augmentation. The final model must meet its license and size rules.
- Portal submissions are limited to five per day; an experiment and submission
  log are present in the workspace.

## Phase 1 - audit and validation

| Measure | Result |
|---|---:|
| Training Source 1 records | 2,206,821 |
| Training Source 2 + 3 records | 10,320,219 |
| Test Source 1 records | 1,732,544 |
| Duplicate source IDs or truth rows | 0 |
| Training singletons | 123,247 (5.58%) |
| Frozen validation Source 1 entities | 441,021 |
| Development Source 1 entities | 1,765,800 |

The split is deterministic by Source 1 ID (`seed=2026`). The exact macro F0.5
scorer includes the specified singleton behavior. Test includes France, which
is absent from training, so the pipeline treats country as open-set text.

## Phase 2 - candidate generation

The selected approach uses multiple local blocking views: normalized name
tokens, name pairs, address tokens, name/address combinations, and numeric
anchors. A reusable on-disk index contains 139,518,407 postings over all
training Source 2/3 records. A compact name/address sketch ranks candidates;
blocks with more than 200 target postings are skipped, and the best 150
candidates per Source 1 entity are retained.

| Full validation measure | Result |
|---|---:|
| Source 1 entities covered | 441,021 / 441,021 |
| Candidate pairs | 50,284,307 |
| Average candidates per Source 1 | 114.02 |
| True links retrieved | 1,400,576 / 1,527,143 |
| True-link recall | **91.71%** |
| Oracle macro F0.5 ceiling | **0.9654** |
| Reduction ratio | 0.99998895 |
| Full retrieval runtime after index build | 1,615 seconds |

The oracle ceiling is the score a perfect matching stage could obtain if it
selected exactly the true links present among these candidates. It is **not**
an achieved matching score or leaderboard result. Among 24,511 validation
singletons, only 19 had no candidates, so the next stage must explicitly
learn when to predict an empty match list.

## Verification

- Five metric/blocking correctness tests pass.
- Independent evaluator confirmed every held-out Source 1 row exactly once,
  no duplicate IDs within candidate lists, and the full recall figures above.
- Compressed temporary output and aligned per-candidate retrieval-view masks
  passed a separate 405-entity probe.
- Plain TSV output with evidence sidecar passed a 103-entity probe.

## Next phase

Build the evidence-ledger matching baseline on these frozen candidates.
Train only from permitted labels, measure actual macro F0.5 on the fixed
holdout, tune a precision-oriented abstention threshold, and produce final
test output only after that model is verified. No competition submission has
been attempted.
