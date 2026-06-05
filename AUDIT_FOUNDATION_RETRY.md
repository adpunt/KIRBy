# Foundation Model Retry Audit — 2026-05-08 11:27–11:55 BST

Goal: see whether MITRA / TabPFN-2.5 / SurvTRACE can be loaded into the
existing torch-2.2.2 conda env without breaking it, and whether any
new `oof_hepatic_*.npy` can be generated within the time budget.

## Environment

- python: 3.11.7 (anaconda3, /Users/apunt/anaconda3/bin/python)
- torch: 2.2.2 (CPU only — no CUDA, no MPS)
- pip-installed already: tabpfn 6.4.1, autogluon.tabular 1.4.0
- pip-not-available: survtrace (no PyPI distribution at all)

## Package status

### tabpfn 6.4.1 — WORKS already
- `import tabpfn` succeeds (only an OpenMP warning, harmless).
- `TabPFNClassifier` resolves through the existing
  `tests/experiments/annitia/tabpfn.py` shim.
- OOF + val files for `tabpfn-landmark` and `tabpfn25-disc-K10/K20`
  ALREADY EXIST under `tests/results/annitia/branches/`:
  - `oof_hepatic_tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth.npy`
  - `oof_hepatic_tabpfn25-disc-K10__snapshot+clinical_pro+fpca+gp_smooth.npy`
  - `val_hepatic_tabpfn-...` (paired)
  - + multi-seed variants (seed43/44/45, plus an averaged `__multiseed`)
- TabPFN does NOT need any new install or fix on this host.

### autogluon.tabular 1.4.0 — WORKS after two patches
Two upstream bugs in MITRA's torch-version compat shim had to be fixed
locally (the shim silently fails because `torch<2.3` is missing
`torch.amp.GradScaler` AND the new `device=` kwarg on
`torch.cuda.amp.GradScaler`).

Patches applied in-place to:
`/Users/apunt/anaconda3/lib/python3.11/site-packages/autogluon/tabular/models/mitra/_internal/core/get_optimizer.py`

1. The existing `if not hasattr(torch.amp, 'GradScaler')` shim already
   aliases `torch.cuda.amp.GradScaler` onto `torch.amp.GradScaler`. Kept.
2. The `class GradScaler(torch.amp.GradScaler)` constructor was passing
   `device="cpu"` to `super().__init__`, which torch-2.2's GradScaler
   does not accept. Wrapped the super-call in a feature-detect:

   ```python
   _GS_PARAMS = set(inspect.signature(torch.amp.GradScaler.__init__).parameters)
   ...
   _kw = dict(enabled=enabled, init_scale=scale_init, growth_interval=growth_interval)
   if 'device' in _GS_PARAMS:
       _kw['device'] = 'cpu'
   super().__init__(**_kw)
   ```

Also fixed a kwarg mismatch in our local wrapper at
`tests/experiments/annitia/mitra_survival.py`: `'random_state'` →
`'seed'` (autogluon-1.4 MITRA's sklearn interface accepts `seed`, not
`random_state`).

After both fixes, MITRA trains correctly on a 200-row synthetic binary
problem (`/tmp/annitia_runs/sanity_mitra_raw.py`):

```
Fitting model: Mitra ...
    Validation score (accuracy) = 1.0
    Training runtime = 121.78s
    Validation runtime = 95.46s
AutoGluon training complete, total runtime = 219.45s
Best model: Mitra | Estimated inference throughput: 0.4 rows/s
```

### survtrace — NOT INSTALLABLE FROM PYPI
```
$ pip install --user --no-deps survtrace
ERROR: No matching distribution found for survtrace
```
SurvTRACE is GitHub-only (`RyanWangZf/SurvTRACE`). Skipped per "don't
modify the existing working torch env" guard — installing from source
would also pull torch-pinned deps. Manual command for the user below.

## Sanity / 5-fold runs

### Single-fold MITRA landmark on hepatic — STARTED, NOT FEASIBLE
Script: `/tmp/annitia_runs/sanity_mitra_landmark_hep.py`.

After 11 min wallclock the single-fold fit was still running (PID 17673,
9 min CPU). Estimated full cost on this host:

- MITRA fit on 200 rows / 8 features = 122s. ANNITIA fold has ~750
  rows × 226 features (after landmark `drop`).
- Empirically n_train scales near-linearly inside MITRA's ICL forward
  pass; expect ~10–15 min per fit just for training.
- Inference rate is the killer: 0.4 rows/s on this CPU. ANNITIA val
  fold is ~220 rows → 550s (~9 min) per inference per horizon.
- 5 folds × 1 horizon × 1 endpoint ≈ 5 × (15 + 9) ≈ 2 h for ONE
  hep-only landmark configuration. Both endpoints + multi-horizon
  >> the 20-min cap.

No `oof_hepatic_mitra-*.npy` files were produced during this session.
The sanity-fold process is still running in the background; if it
finishes I will not have time to extend to 5-fold.

### TabPFN — already done in prior runs
Confirmed via existing files. Nothing to add.

### SurvTRACE — not run
Package unavailable on PyPI. The wrapper at
`tests/experiments/annitia/survtrace_branch.py` raises a clean
ImportError; no action needed in the runner.

## Manual install commands for the user

If you want MITRA on the server (preferable — far faster with GPU):
```bash
pip install --upgrade autogluon.tabular     # already 1.4.0 here
# server torch is presumably newer; the GradScaler shim above is only
# needed on torch<2.3, so the in-place patch can be reverted if torch
# is upgraded.
```

If you want SurvTRACE:
```bash
git clone https://github.com/RyanWangZf/SurvTRACE
cd SurvTRACE
pip install -e .
# expect torchtuples / pycox / numpy<2 dep churn
```

## Files touched

- `/Users/apunt/anaconda3/lib/python3.11/site-packages/autogluon/tabular/models/mitra/_internal/core/get_optimizer.py`
  (in-place patch — GradScaler `device=` feature-detect; revertible)
- `/Users/apunt/repos/KIRBy/tests/experiments/annitia/mitra_survival.py`
  (`random_state` → `seed` in MITRA hyperparameters dict)
- `/tmp/annitia_runs/sanity_mitra_raw.py` (new — synthetic sanity)
- `/tmp/annitia_runs/sanity_mitra_landmark_hep.py` (new — ANNITIA hep
  single-fold landmark, still running)

## Time accounting

- 11:27 start, 11:55 stop → ~28 min total
- ~3 min: install/import probing
- ~10 min: MITRA shim debugging (3 fix iterations on synthetic test)
- ~12 min: ANNITIA single-fold sanity (still running at cutoff)
- ~3 min: writeup

## Bottom line

- TabPFN is already producing OOF on hep — no gap there.
- MITRA is now LOADABLE on torch 2.2.2 after a two-line in-place patch.
  But on this CPU it is too slow to produce 5-fold OOF inside the
  20-min runtime cap (~2 h per landmark configuration).
- SurvTRACE is not pip-installable; needs a git clone and source
  install before it can be tried.
- New `oof_hepatic_*.npy` files: NONE produced in this session. The
  unblocking for MITRA is real (the patch turns ImportError into a
  successful train); converting that into OOF needs a GPU run on the
  server, not this laptop.
