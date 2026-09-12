# s09-pools-unchecked-result-files

**title:** s09 aggregates six per-dataset JSONs across datasets without reading the provenance stamps every one of them carries

**area:** tests/experiments/similarity/ (s01–s13, _summary_keys.py, _run_similarity_locally.py, similarity_helpers.py), src/kirby/similarity/, tests/experiments/similarity_optimized_hybrid.py, tests/experiments/structural_geometry_survey.py

**class:** C-stale-undetected

**severity as reported:** large

**location:** tests/experiments/similarity/s09_cross_dataset_summary.py:766-782, tests/experiments/_shared.py:5700-5709, tests/experiments/_shared.py:5713-5738, tests/experiments/_shared.py:5792

## scenario as reported
s02..s08 each write tests/results/<exp>/<ds>.json, and save_experiment_results stamps every one with `distance_metric` (_shared.py:5792) and `input_fingerprint` (n_samples, labels_hash, groups_hash, n_splits, the sorted rep name+width list, and distance_metric — _shared.py:5687-5709). run_all's resume check compares that fingerprint and re-runs on drift (similarity_helpers.py:1085-1090). s09 does neither. Run 1: `run_dataset.py similarity --dataset herg_ki` on 2026-09-04 writes all six herg_ki files under the 10-rep pool at the default manhattan coefficient. Run 2: on 2026-09-08 logd is re-run after its rep pool changes, and separately s02 for logd is re-run with `--distance-metric cosine` into the same tests/results tree, so logd/s02 sits on a different coefficient and a different rep list from logd/s04-s08 and from every herg_ki file. Run 3: `python s09_cross_dataset_summary.py --datasets logd caco2 herg_ki qm9` loads each `<exp>/<ds>.json` (line 772), keeps whatever parses, and builds the per-dataset table and the Wilcoxon signed-rank test across datasets. The published cross-dataset row for logd mixes an SAC_delta measured under cosine on one pool with a kNN_delta measured under manhattan on another, and the summary file records neither. Nothing in the run prints a warning, because the only check is `if r:` (line 773).

## evidence as reported
s09_cross_dataset_summary.py:769-782 is the whole loader: `for exp_id in EXPERIMENT_IDS: for dir_name in accepted_names(exp_id): r = load_experiment_results(args.results_dir, dir_name, ds); if r: ds_results[exp_id] = r; ... break`. grep for input_fingerprint / distance_metric / timestamp / result_status / fingerprint over s09_cross_dataset_summary.py returns nothing. _shared.py:5722-5726 `fingerprint_mismatch` already checks exactly the fields needed ('n_samples','labels_hash','groups_hash','n_splits','distance_metric') and is imported nowhere in s09. _summary_keys.py contains only key-name aliasing and the empty-aggregate refusal; it has no provenance logic either.

## fix as proposed
In s09's loader, read `distance_metric` and `input_fingerprint` off each of the six files as they are loaded. Refuse to aggregate a dataset whose six files disagree on `distance_metric`, or whose `labels_hash`/`n_samples`/rep list disagree, naming the file and the field. Record the agreed distance_metric and the per-source `timestamp` in the summary's own output so a later reader can see which coefficient and which run each row came from. Decision needed from the owner: refuse the dataset outright, or emit the row with a `provenance_mismatch` field — the second keeps partial results but leaves a mixed row in a published table.
