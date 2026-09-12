# alloc-failure-encoding-changed

**title:** A representation that could not be scored is NaN now and was 0.0 before, with nothing in the fold_scores key recording which

**area:** Cache WIPE and SERVER procedure (wipe_allocation_caches.sh, setup.sh, SLURM scripts, result_db schema guard, run_dataset.py resume)

**class:** C-stale-undetected

**severity as reported:** large

**location:** /Users/apunt/repos/KIRBy/src/kirby/find_best_hybrid.py:5535-5549 (the new NaN branch), :5486-5490 (the key), :5551-5554 (the write); compare git HEAD find_best_hybrid.py:5451-5458

## scenario as reported
Run 1, under the committed code, searches logd. On fold 0 the svm evaluator raises on one representation, so 0.0 is stored for it in fold_scores/logd/alloc__f0__svm__<h>.pkl. Run 2, under the working-tree code, asks for the same dataset, folds, seed and evaluator. _alloc_in_hash is built from the fold's representation matrices, its labels and the internal carve indices — none of those moved — so the filename is identical, and the meta is {kind, evaluator, fold_idx, input_hash, metrics}, none of which records the failure encoding. Run 2 is served 0.0. The new code puts NaN there precisely so _require_scored_reps can count a failed representation and refuse the fold; a finite 0.0 is not a failure, so the guard cannot fire, every weight comes out equal, and the budget is split EQUALLY and filed under whatever allocation method was asked for. That is the exact failure the comment at :5535-5546 says the change exists to stop, and a cached hit reinstates it silently.

## evidence as reported
find_best_hybrid.py:5547-5549 `default_scores[rep_name] = float('nan')` / `for mn in _alloc_metrics:` / `metric_scores[mn][rep_name] = float('nan')`, under the comment `Fixed 2026-09-10.` at :5546. git HEAD of the same block: `default_scores[rep_name] = 0.0` / `metric_scores[mn][rep_name] = 0.0`. Key: find_best_hybrid.py:5487-5490 `_am = {'kind': 'fold_alloc', 'evaluator': ev, 'fold_idx': fold_idx, 'input_hash': _alloc_in_hash, 'metrics': tuple(sorted(_alloc_metrics))}` — no encoding field. Payload written at :5551-5554 `_save_cache_with_meta(_ap, _am, {'default_scores': default_scores, 'metric_scores': metric_scores})`.

## fix as proposed
fold_scores must be DELETED BY HAND on the server (it is also missing from the wipe script — see fold-scores-missing-from-wipe). There is no meta or filename change that catches it, and no reason to keep it: the whole layer is per-fold scores that rebuild in one pass. If you want the layer to defend itself in future, add one constant to the alloc meta naming the failure encoding, e.g. `'unscored_encoding': 'nan'`, so a file written under the old encoding is a stale miss instead of a hit. Note this only bites a cache that already ran the 2026-09-09 commit 0564781 (the one that introduced _fold_score_cache_path); if $KIRBY_CACHE_DIR/fold_scores does not exist on the server, nothing is at risk and the wipe-list gap is the only thing to fix.
