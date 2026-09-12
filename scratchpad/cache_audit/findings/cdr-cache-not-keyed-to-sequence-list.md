# cdr-cache-not-keyed-to-sequence-list

**title:** The CDR-index cache is keyed on the dataset name alone, so a cached list built from a different set of sequences is served without a check

**area:** tests/experiments/similarity/ (s01–s13, _summary_keys.py, _run_similarity_locally.py, similarity_helpers.py), src/kirby/similarity/, tests/experiments/similarity_optimized_hybrid.py, tests/experiments/structural_geometry_survey.py

**class:** A-wrong-answer

**severity as reported:** small

**location:** tests/experiments/similarity/similarity_helpers.py:508-513, tests/experiments/similarity/similarity_helpers.py:524-529, tests/experiments/similarity/similarity_helpers.py:694-699, tests/experiments/similarity/similarity_helpers.py:598-608

## scenario as reported
Run 1: `KIRBY_SMOKE_N=200 python tests/experiments/similarity/_run_similarity_locally.py flab_thermo_tresanco s04`. _shared.py:3124 subsamples the dataset to 200 rows, s04 calls _extract_cdr_sequences(list(heavy_seqs), dataset_name='flab_thermo_tresanco'), ANARCI runs on those 200 sequences, and the 200-element list is pickled to $KIRBY_CACHE_DIR/cdr_indices/flab_thermo_tresanco.pkl (line 556-558). Run 2: the full run on the same dataset, ~1000 rows. Line 525 sees the file exists and line 527 returns the 200-element list, with no comparison against `heavy_seqs` at all. Line 697 then calls antibody_sequence_cliff_sweep(cdr_seqs, labels, is_clf): line 598 sets `n = len(sequences)` = 200, line 599 `y = np.asarray(labels)` = 1000 entries, and line 603 does not fire because 200 < max_compounds=3000, so no resampling realigns them. identify_cliff_compounds_editdist is handed 200 CDR strings from the subsample paired positionally with the first 200 labels of the full dataset — a different molecule per row. The cdr_only cliff counts written into s04_activity_cliffs/flab_thermo_tresanco.json are computed on mismatched pairs, and the only failure path (line 624) records an error string, so if no exception is raised the number is reported as real.

## evidence as reported
similarity_helpers.py:508-513 `_cdr_cache_path` returns `cache_root / 'cdr_indices' / f'{dataset_name}.pkl'` — dataset name only. 524-529: `cache_path = _cdr_cache_path(dataset_name) if dataset_name else None; if cache_path and cache_path.exists(): try: return pickle.loads(cache_path.read_bytes())` — the cached list is returned before `heavy_seqs` is looked at. 598-604 in antibody_sequence_cliff_sweep: `n = len(sequences)` / `y = np.asarray(labels)` / `if n > max_compounds:` — no length equality check between sequences and labels anywhere in the function. The repo already has the right pattern next door: src/kirby/similarity/mcs.py:167-192 `assert_pairs_match_smiles` refuses a pair set whose stored `smiles_sha` does not match the molecule list in hand, and mcs.py:20-27 spells out why ("a pair set only means anything beside the molecule list it was built from, and one dataset name can carry several such lists").

## fix as proposed
Store the sequence-list ordering hash alongside the CDR strings and check it on read: pickle `{'seqs_sha': kirby.similarity.mcs.pairs_ordering_hash(heavy_seqs), 'cdr': cdr_seqs}` at similarity_helpers.py:558, and at line 525-529 recompute the hash of `heavy_seqs` and fall through to a rebuild when it differs or is absent. Add a length assertion at the top of antibody_sequence_cliff_sweep (`len(sequences) == len(labels)`) so a mismatch raises instead of silently pairing. Delete $KIRBY_CACHE_DIR/cdr_indices/ once, since the existing files cannot prove which list they belong to.
