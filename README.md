# Amazon ML Challenge - starter workspace

The official dataset is available in `student_resource/dataset/`. This workspace
keeps the original files untouched and builds a local-only business-entity
resolution pipeline.

## Next safe workflow

1. Run `python src/phase1.py` for the full dataset audit and deterministic
   Source 1 validation split. Results go to `outputs/phase1_audit.json`.
2. Build the reusable Phase 2 index and compact ranking sketch, then retrieve
   candidates. The commands below use the full Source 2/3 training pool.
3. Use `python src/score_validation.py --predictions <matching-results.tsv>`
   once a matching model produces predictions for the full validation split.
4. Validate final test output with the official validator in
   `student_resource/utils/validate_submission.py`.

## Phase 2 commands

Run from this workspace with Python 3.12 and NumPy:

```
python src/phase2_indexed.py build --partition train --target-count 10320219
python src/phase2_indexed.py sketch --partition train
python src/phase2_indexed.py retrieve --partition train --sample-modulus 1 --max-postings 200 --top-k 150 --out work/phase2_full_validation_candidates.tsv --report outputs/phase2_full_validation_report.json
python src/evaluate_candidates.py --candidates work/phase2_full_validation_candidates.tsv --require-full --out outputs/phase2_full_validation_quality.json
```

The index is about 1.67 GB and the text sketch about 0.37 GB. They are
regenerable work files, not final submission artefacts. Candidate retrieval
does not use ground-truth labels. The validation score script reads them only
after candidate output has been frozen.

`src/phase2.py` remains a slower independent batch-scanner baseline. Its
sample report is useful to cross-check the indexed implementation.

For test inference in a later phase, build a separate `test` index with
`--partition test --target-count 9969589`, then retrieve with
`--partition test --sample-modulus 1`. Ensure enough free disk for the index,
sketch, and full `candidate_pairs.tsv` at the same time. An output path ending
in `.gz` streams a compressed temporary TSV; after retrieval, the regenerable
index can be removed and the file expanded to the required plain TSV. The
official submission itself must remain an uncompressed TSV.

## Expected data layout

```
student_resource/dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Run the official validator from the `student_resource/` directory with
`python utils/validate_submission.py --matching output/matching_results.tsv
--candidate output/candidate_pairs.tsv --test-dir dataset/test`.
