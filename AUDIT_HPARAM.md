# AUDIT — experiment_hparam.py (round 5)

Date: 2026-05-08 (Trustii deadline day)
File: `tests/experiments/annitia/experiment_hparam.py`

## Bug diagnosis

Two bugs, both silent (script exited 0).

### Bug 1 — wrong key in BE-summary lookup

`_read_be_top_view_set` reads each BE record with `r.get('view_set', [])`,
but `experiment_be.py` writes those records under the key `'views'`
(see `experiment_be.py:203`: `'config_id': cid, 'views': list(vs), ...`).

Effect: every record's view list became `[]`, so the dictionary
`by_views` had a single key — the empty tuple `()` — and
`best_views = max(...)` returned `()`. Downstream:

```
=== hepatic: top view set = [] ===
X_tr_full = np.concatenate([views_train[v][0] for v in primary], axis=1)
ValueError: need at least one array to concatenate
```

i.e. concatenating an empty list of arrays. The script crashed before any
Optuna search ran.

### Bug 2 — Optuna search spaces reference wrapper kwargs that do not exist

After fixing Bug 1, the rsf search space crashed every trial with
`TypeError: SurvivalRSF.__init__() got an unexpected keyword argument 'max_depth'`,
which Optuna swallowed via `catch=(Exception,)` so all 5 trials returned
0.0 ("best" C=0.0000) and the refit step then died.

Audited every space against the wrappers in `survival.py`:

| space             | bad kwarg(s) passed by `experiment_hparam.py` | wrapper accepts? |
|-------------------|-----------------------------------------------|------------------|
| `_space_rsf`      | `max_depth`                                   | NO               |
| `_space_lgbm_aft` | `min_data_in_leaf`                            | NO               |
| `_space_ngboost`  | `minibatch_frac`                              | NO               |
| `_space_gbsa`     | (all four match)                              | OK               |
| `_space_catboost_cox/_aft` | (all four match)                     | OK               |
| `_space_coxnet`   | (both match)                                  | OK               |

## Fix applied

Conservative — no refactor, only the four functions touched plus the BE
key lookup. Concretely:

1. `_read_be_top_view_set`: now uses `r.get('views', r.get('view_set', []))`,
   skips empty lists, and raises `RuntimeError` instead of returning `[]`
   if no usable record is found (so a future regression fails loudly).
2. `_space_rsf`: dropped `max_depth`; widened `min_samples_leaf` (1–30).
3. `_space_lgbm_aft`: dropped `min_data_in_leaf`; replaced with `sigma`
   (`0.5–2.0`), which the `SurvivalLightGBMAft` wrapper does accept.
4. `_space_ngboost`: dropped `minibatch_frac`; replaced with `dist`
   (`logNormal` vs `exponential`), which the `SurvivalNGBoost` wrapper accepts.

## Sanity-run output

```
$ python tests/experiments/annitia/experiment_hparam.py \
      --endpoint hepatic --models rsf --n-trials 5 \
      --n-folds 3 --n-folds-refit 3 \
      --output-dir /tmp/annitia_runs/hparam_sanity

=== hepatic: top view set = ['snapshot', 'clinical_pro', 'fpca', 'gp_smooth'] ===
  rsf             searching (5 trials × 3 folds)...
    best CV-3 C=0.8625  (375s, params={...})
  rsf             refit-3 C=0.8625±0.0484
DONE in 6.5 min
```

Top view set now resolves correctly (matches the top-CV BE record:
`rsf` on `snapshot+clinical_pro+fpca+gp_smooth` at C=0.8637). Optuna
actually ran. Files written:

```
best_rsf_hepatic.json
cv_summary.json
oof_hepatic_hparam-rsf__snapshot+clinical_pro+fpca+gp_smooth.npy
val_hepatic_hparam-rsf__snapshot+clinical_pro+fpca+gp_smooth.npy
```

## Final-run status

Started in background:
```
python tests/experiments/annitia/experiment_hparam.py \
    --endpoint both \
    --models rsf,catboost_cox,catboost_aft,gbsa,ngboost,coxnet \
    --n-trials 20 --n-folds 3 --n-folds-refit 5 \
    --output-dir tests/results/annitia/hparam
```

PID 92102, log at `/tmp/annitia_runs/round5_hparam_v2.log`.
6 algos × 2 endpoints = 12 (model, endpoint) tuples × 20 trials each.
Estimated wall time 3–5 h on this Mac (RSF dominates at ~25 min/endpoint;
catboost/coxnet are seconds; ngboost / gbsa medium). lgbm_aft skipped per
docstring's 6-algo list.

Status as of writing: **running** (just started). Output dir
`tests/results/annitia/hparam/` was empty before launch, so nothing can
be overwritten.

## New artifacts produced

When complete the script will write, for each (model, endpoint) tuple:

- `tests/results/annitia/hparam/best_<model>_<endpoint>.json` — best Optuna
  params + best CV-3 c-index + chosen view set
- `tests/results/annitia/hparam/oof_<endpoint>_hparam-<model>__<viewset>.npy`
  — OOF predictions from 5-fold refit
- `tests/results/annitia/hparam/val_<endpoint>_hparam-<model>__<viewset>.npy`
  — val_test predictions from refit-on-full

Plus a single `tests/results/annitia/hparam/cv_summary.json` whose
`per_condition` schema matches the rest of the pipeline.

## Will refuse_meta_aware.py pick them up?

**Not automatically.** `refuse_meta_aware.py:184–186` hardcodes
`pool_dirs = [res_root / p for p in ('be', 'cd', 'hep_boost', 'branches', 'external')]`.
The new `hparam` directory is **not** in that list, so the new OOFs and
val_test files will be ignored by the meta-aware fusion stage even
though their schema is compatible.

Two options to wire them in (both one-line edits — not done here, so we
don't bundle scope changes into a fix):

1. Append `'hparam'` to the pool_dirs tuple. Simplest. Recommended for
   the deadline-day run.
2. Flag-controlled: add `--include-hparam` and gate the dir on it.

`build_submission_zip.py` and `build_v2_zips.py` operate on the cv_summary
records too and use `r['views']`, which `experiment_hparam.py` records
already include (rec dict has `'view_set': primary` rather than
`'views': primary`). If those builders also need to read hparam outputs
they will need the same `views`/`view_set` reconciliation. That's a
separate bug — flagging only.

### Naming caveat

User asked about naming `oof_<endpoint>_hparam_<algo>__<viewset>.npy`
(underscore between `hparam` and algo). The script writes
`oof_<endpoint>_hparam-<model>__<viewset>.npy` (dash). This matches
its own docstring and is consistent with the rest of the schema (which
uses the `config_id` verbatim). `refuse_meta_aware.py` discovers files
via the cv_summary's `config_id` field, so the dash/underscore choice
doesn't matter for discovery — only that pool_dirs includes `hparam`.

## Files changed

- `tests/experiments/annitia/experiment_hparam.py`
  - `_read_be_top_view_set`: support `views` key, raise on empty
  - `_space_rsf`, `_space_lgbm_aft`, `_space_ngboost`: dropped invalid kwargs
