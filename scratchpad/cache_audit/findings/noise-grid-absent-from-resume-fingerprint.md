# noise-grid-absent-from-resume-fingerprint

**title:** The resume stamp records nothing about the noise grid, so changing the levels, conditions or seeds leaves every finished noise result on disk

**area:** tests/experiments/noise/ (n01–n15, noise_helpers.py, _similarity_hybrid_space.py, the _smoke_* files) plus src/kirby/noise_spec.py and src/kirby/utils/noise_estimation.py

**class:** C-stale-undetected

**severity as reported:** large

**location:** tests/experiments/noise/noise_helpers.py:410, 436-465; tests/experiments/_shared.py:5684-5710 (input_fingerprint), 5713-5737 (fingerprint_mismatch); tests/experiments/dataset_policy.py:394

## scenario as reported
Run 1: run_all('logd') runs n11/n12/n13 over dataset_policy._DEFAULT_NOISE_LEVELS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0), writes tests/results/noise/n13_noise_distributions/logd.json stamped with input_fingerprint. Someone then edits _DEFAULT_NOISE_LEVELS (that file's own comments show this grid was replaced wholesale on 2026-08-27), or edits n13.NOISE_SEEDS, or edits REFERENCE_ERROR_LABEL_UNITS which sets n07's default noise_level. Run 2: run_all('logd') reaches noise_helpers.py:436, the JSON exists and holds numbers, and fingerprint_mismatch compares only n_samples, labels_hash, groups_hash, n_groups, n_splits, distance_metric, the rep list and fold_fitted_reps — none of which moved. It prints 'RESUME n13: already done — skipping' and the file, computed over the old level grid, stays and is read as current.

## evidence as reported
_shared.py:5684-5695 — the fingerprint dict holds n_samples, labels_hash, groups_hash, n_groups, reps, fold_fitted_reps; :5703-5707 adds distance_metric with the comment 'Without it, a run under a second coefficient reads the first coefficient's file, sees no change and skips'; :5709 adds only n_splits from config. fingerprint_mismatch iterates the fixed tuple ('n_samples','labels_hash','groups_hash','n_groups','n_splits','distance_metric') at :5723-5725. Nothing in either function reads a noise level, condition, seed or dose unit. noise_helpers.py:456-460 is the branch that keeps the file when no drift is named.

## fix as proposed
Add the sweep constants each noise experiment actually ran under — the level grid, the condition list, the noise seeds, the dose units, and the downstream model list — to input_fingerprint (or to a per-experiment stamp written by each run()), and add those keys to fingerprint_mismatch's comparison list. This makes every existing noise JSON recompute once, which is the point: today there is no way to tell which of them predate the grid change.
