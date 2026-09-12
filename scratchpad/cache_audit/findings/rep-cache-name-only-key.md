# rep-cache-name-only-key

**title:** rep_cache is keyed by dataset and rep name alone, and the only guard is a row-count check

**area:** Experiment-side caches in tests/experiments/_shared.py (create_hybrid_cached / _hybrid_fold_cache_key, rep_cache, the search-reuse helpers, and the model/evaluator lists that ride in the search cache key)

**class:** C-stale-undetected

**severity as reported:** large

**location:** tests/experiments/_shared.py:205-238 (_rep_cache_dir, _cache_rep), 240-254 (_save_rep_npy), 257-315 (_cache_rep_dict), 3205-3225 (the row-count guard and its remediation message)

## scenario as reported
The cache file is rep_cache/<dataset>/<rep>.npy and nothing else enters the key. Any change that keeps the row count but changes the rows or the featuriser is invisible. Concrete: run 1 loads 'logd', get_molecular_representations computes ecfp4 over the smiles list and writes rep_cache/logd/ecfp4.npy. Later the source CSV is edited — one bad molecule dropped and one added, or the rows re-sorted — so run 2's smiles list is a different order with the same length. _cache_rep_dict finds ecfp4.npy, loads it, and hands back run 1's fingerprints against run 2's labels: every molecule is now scored against another molecule's features. The guard at 3216 compares arr.shape[0] to len(labels) only, so both are 5028 and it passes. The same holds for a featuriser change (an rdkit upgrade, a new ESM checkpoint) that keeps the width: rep_shapes in the search meta still matches, labels_hash still matches, and every derived layer serves the old numbers. _save_rep_npy at 243-244 returns early when the file exists, so even a deliberate recompute cannot replace it. Note that fold_scores/ is safe here — _fold_score_cache_path keys on _fold_input_hash, a content hash of the fold's matrices — so this is a rep_cache, searches/ and result_db problem, not a fold_scores one.

## evidence as reported
tests/experiments/_shared.py:216-221
    cache_dir = _rep_cache_dir() / dataset_name
    cache_file = cache_dir / f'{rep_name}.npy'
    if cache_file.exists():
        arr = np.load(cache_file)
tests/experiments/_shared.py:243-244  if cache_file.exists(): return
tests/experiments/_shared.py:3216-3218  _bad = {rn: int(arr.shape[0]) for rn, arr in _reps.items() if ... arr.shape[0] != _n_labels}
tests/experiments/_shared.py:3224  f"python tests/purge_rep_cache.py {dataset_name} --delete"
tests/purge_rep_cache.py:108-109  pkl_dirs = ['searches', 'baselines', 'hybrid_evals', 'hybrid_fold_cache']  — reached only under --all-caches, and result_db is not in the list.

## fix as proposed
This one is a wipe procedure, not a code fix (representation content is deliberately out of every cache key). Two things are worth changing anyway. First, the guard's own message at 3224 tells the reader to run purge_rep_cache without --all-caches, which deletes the reps and leaves searches/, baselines/, hybrid_evals/ and hybrid_fold_cache/ holding numbers computed from the reps just deleted — make the message say --all-caches. Second, purge_rep_cache's pkl_dirs list omits result_db, so a purged dataset keeps its searched-configuration rows; the owner has to decide whether that directory belongs in the list. The underlying same-length content change stays undetectable by design, so the wipe rule needs to be written down: change a loader, a filter, a row order or a featuriser version, and rep_cache/<dataset>/ plus every derived layer for that dataset gets deleted.
