# resume-fingerprint-ignores-the-metric

**title:** The resume fingerprint records the data but not the metric, so a JSON written under r2 is kept and read as RMSE

**area:** tests/experiments/complexity/c01–c19 + complexity_helpers.py (cache behaviour audit)

**class:** C-stale-undetected

**severity as reported:** large

**location:** tests/experiments/_shared.py:5684-5709 (input_fingerprint), :5713-5735 (fingerprint_mismatch); tests/experiments/complexity/complexity_helpers.py:1094-1105; tests/experiments/complexity/c08_calibration.py:81-108

## scenario as reported
Run 1 is the complexity suite on herg_ki before 2026-09-04, when the registry metric was r2. c02's JSON records best_by_model scores in r2 and is stamped metric_name='r2'. The registry then moves herg_ki to rmse (the same change that widened result_cache.IDENTITY_COLS with 'metric' and forced every result_db to be deleted). Run 2 is the suite on herg_ki today: complexity_helpers.py:1094 finds the c02 JSON, fingerprint_mismatch compares n_samples, labels_hash, groups_hash, n_groups, n_splits, distance_metric and the rep list — all unchanged — returns None, and prints "RESUME c02: already done — skipping". The r2 file stands as this run's c02 result. Then c08 (c08:81 load_experiment_results, no check of any kind) reads that file's best_by_model and at c08:104 picks the winner with `(min if lower_is_better else max)` where lower_is_better now comes from today's registry — so it selects the WORST-scoring config from the r2 table and calls it the optimal one, and c09:660 builds its bias-variance curve on the same stale table.

## evidence as reported
_shared.py:5713-5735 — the loop is `for key in ('n_samples','labels_hash','groups_hash','n_groups','n_splits','distance_metric')`; 'metric_name' and 'task_type' are not in it, and input_fingerprint never emits them. _shared.py:5793-5796 does stamp `output['metric_name']` and `output['lower_is_better']` into the file, so the information is already on disk — it is simply not compared. _shared.py:5835-5844 — load_experiment_results is a bare `json.load` with no fingerprint or metric check.

## fix as proposed
Add metric_name and task_type to input_fingerprint and to fingerprint_mismatch's key loop (guard the comparison with `if key in current and key in stored` so files predating the field are not all invalidated at once), and have load_experiment_results refuse — or at least warn loudly — when the upstream file's metric_name differs from the caller's. Separately, this needs a documented wipe: any tests/results/c*/ JSON written before 2026-09-04 is in the old metric and no code change makes it correct.
