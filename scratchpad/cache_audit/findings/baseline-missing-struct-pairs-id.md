# baseline-missing-struct-pairs-id

**title:** The MCS pair set is not in the baseline key, so a struct_fid baseline computed against one pair set is served to a run using another

**area:** baselines/, folds/, fold_scores/ and ft_embeddings/ cache layers (src/kirby/find_best_hybrid.py)

**class:** A-wrong-answer

**severity as reported:** large

**location:** src/kirby/find_best_hybrid.py:176-212 (_baseline_cache_path), :645-688 (_make_baseline_meta), :4700-4709 (_geo_fold_score), :7412-7425 (the search's own _search_struct_pairs_id)

## scenario as reported
Run 1: find_best_hybrid(dataset_name='herg_ki', objective='struct_fid', structural_pairs=A, row_index=r). Every representation's baseline is the Spearman between that representation's row-pair Euclidean distance and the MCS distance of pair set A, written to baselines/herg_ki__ecfp4__rf__struct_fidelity__5f_42__obj-struct_fid.pkl. Run 2: the same call with pair set B (get_mcs_pairs re-drawn at a different n_pairs or seed, or the study .npz accepted this time instead of refused - tests/experiments/noise/_similarity_hybrid_space.py:116-149 does exactly this). The filename is byte-identical and every key in the expected meta matches, so run 2 is served run 1's structural correlations. Those per-representation numbers are the comparator the hybrid arm's delta is subtracted from, and they are also what _predict_top_k_subset ranks the top_k allocations by, so run 2 reports a delta against a quantity measured on pairs it never used.

## evidence as reported
_baseline_cache_path builds the name from dataset/rep/model/metric/n_folds/seed plus optional sec/noise/objective/tune suffixes only - no pair-set term. _make_baseline_meta returns kind, dataset_name, objective, rep_name, rep_shape, rep_hash, model_name, metric_name, n_folds, seed, n_samples, labels_hash, groups_hash, hd_filter_version, hd_filter_threshold, code_timestamp (+secondary_names, +noise_id) - no struct_pairs_id. The value does depend on the pairs: line 4704 `_parts[OBJECTIVE_STRUCT_FID] = _structural_fidelity_score(X_val_scaled, np.asarray(row_index)[val_idx], structural_pairs)`. The repo already treats the pair set as identity everywhere else - line 7412 computes `_search_struct_pairs_id` as a hash of structural_pairs['I'], ['J'] and row_index, feeds it to _make_search_meta and _make_hybrid_eval_meta, and 'struct_pairs_id' is in _OPT_IN_META_KEYS (line 315).

## fix as proposed
Thread the same `_search_struct_pairs_id` into _evaluate_baselines and add it to _make_baseline_meta, emitted only when non-None, and add 'struct_pairs_id' handling so a stored id that the caller did not ask for is rejected (it is already in _OPT_IN_META_KEYS, so writing it into the baseline meta is enough). Delete every baselines/*__obj-struct_fid*.pkl and every baselines file for a blend name containing struct_fid; they cannot be attributed to a pair set.
