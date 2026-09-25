# Extracted challenge requirements

## Objective

Source 1 is the deduplicated reference source. For every Source 1 business,
predict zero or more matching entities from Source 2 and Source 3.

## Non-negotiable constraints

- All input and output files are TSV; always use `sep="\t"`.
- Test coverage includes France although training only covers the US and India.
  Treat country as open-set text and never hard-code training country labels.
- Produce one row for every test Source 1 entity in both output files.
- Match and candidate lists may contain only existing Source 2/3 IDs, have no
  duplicate IDs, and final matches must be a subset of candidates.
- `matching_results.tsv` and `candidate_pairs.tsv` use the exact prescribed
  columns and tab separator.
- No external business lookups, APIs, geocoders, or internet data augmentation.
- Final model must be MIT/Apache-2.0 licensed and at most 8B parameters.

## Metric implication

The score is macro F_0.5 per Source 1 entity, including singletons. A false
merge is costly; tune the final threshold on held-out Source 1 entities and
make the empty prediction the default when confidence is weak.

## Baseline to implement when data arrives

1. Normalize names and addresses locally (Unicode folding, case/punctuation,
   legal suffixes and common street tokens).
2. Retrieve candidates using several local blocks: normalized country + name
   token, postcode/number tokens when present, and character n-gram similarity.
3. Train a pair classifier using only train ground truth and candidate features.
4. Select one precision-oriented threshold by held-out macro F_0.5.
5. Export and validate both TSVs before any leaderboard upload.
