# Federated Experiments Smoke Test — 2026-06-03

## CRITICAL BLOCKER: Python execution was denied in this session

The sandbox denied **every** attempt to run the Python interpreter — including a
bare `python -c "print('hello')"`, `python <runner>.py --help`, and the same with
the sandbox override. Read-only shell tools (`ls`, `grep`, `mkdir`, `which`) were
allowed, but the interpreter itself was not.

Consequence: I could **not** smoke-run fed01–fed05, could **not** produce any
output JSON, and therefore could **not** verify finite/non-NaN numbers as the task
requires. I have **not** fabricated any results. The verdicts below are
**UNVERIFIED-NEEDS-RUN** because the empirical run step was impossible here. The
code review and the smoke-gating fixes are real and complete; they simply have not
been executed.

**What the user needs to do**: grant Python execution permission (or run the
commands below yourself), then the smoke runs can be verified.

---

## Smoke-gating added (test-only, gated behind `KIRBY_SMOKE_N`)

All of this is a no-op when `KIRBY_SMOKE_N` is unset — server/production grids are
untouched. New helpers in `tests/experiments/federated/federated_helpers.py`
(lines ~127-160):

- `_smoke_active()` — true only when `KIRBY_SMOKE_N` is a positive integer.
- `smoke_grid(full, smoke)` — returns the smoke grid under smoke mode, else full.
- `smoke_n_folds(n)` — returns 2 under smoke mode, else `n`.
- `smoke_k_silos(k)` — returns 3 under smoke mode, else `k`.

`_get_federated_config()` now, under smoke mode only, shrinks:
`n_splits 5→2`, `budgets [100]→[50]`, `models 6→['rf','knn']`,
`internal_evaluators 5→['match','rf']`.

Per-experiment grid shrink under smoke mode only:
- **fed01**: `K 5→3`, `SIGMA_FRACS` 6→`[0,0.2]`, `SMOOTH_FACTORS` 5→`[0,0.05]`,
  `DETECTORS` 8→`['knn_k5','zscore']`, `N_FOLDS 5→2`.
- **fed02**: `N_FOLDS 5→2` (K is derived from silo reps, not gated).
- **fed03**: all three `n_folds`/`N_FOLDS_C` 5→2.
- **fed04**: `K 5→3`, `SMOOTH_FACTORS` 5→`[0,0.05]`, `N_FOLDS 5→2`.
- **fed05**: no module-level fold/silo constant; its fan-out is bounded by the
  three silo datasets and `load_data`'s `KIRBY_SMOKE_N` row cap.

Per-dataset runners (`caco2.py`, `logd.py`, `herg.py`) gained a `--reps` flag that
forwards an explicit tiny rep list to `run_all(rep_names=...)`, so smoke runs can
restrict to a cheap rep set (e.g. `ecfp4 maccs avalon`) instead of the full general
list (which for caco2 includes heavy reps: grover, mhg_gnn, molformer, mordred).

---

## Bug found and fixed (code-fixable, static): lhasa hERG rep_names handling

`federated_helpers.load_lhasa_herg_data()` calls
`get_molecular_representations(smiles, rep_names)`, which expects a **string**
(a preset name like `'all'` or a comma-separated rep list). But `run_all()` resolves
the lhasa/herg branch with `rep_names = rep_names or []`, i.e. it passes a Python
**list** (`[]` when no override, or a list from `--reps`).

- With the default (`[]`): `[] in REP_PRESETS` is False, then `[].split(',')` raises
  `AttributeError` — or, depending on path, yields **zero representations**. Either
  way the default `herg.py` run was broken.
- With a list override: `list.split(',')` raises `AttributeError`.

Fix (federated_helpers, in `load_lhasa_herg_data`): normalize the argument —
`None`/empty → `'all'`; a list/tuple → comma-joined string. This makes both the
default herg run and `--reps` overrides valid. This is inside a federated file, so
it was in scope to fix.

---

## Delta / sign-convention review (no bug)

The RMSE-lower-is-better trap was checked in every place a "delta"/"improvement"
field is written:

- `fed05:300` `rmse_improvement = ind_rmse - metrics['rmse']` — positive = better.
  Correct.
- `fed03:1391` `rmse_improvement = ind_rmse - agg['rmse_mean']` — positive = better.
  Correct.
- `fed03:1372` `mcc_improvement = agg['mcc_mean'] - ind_mcc` — MCC higher-is-better,
  positive = better. Correct.

No sign bug found.

---

## Per-experiment verdicts (UNVERIFIED — run step was blocked)

For each, the exact copy-pasteable smoke command is given. Output dir is
`tests/results/_smoke/federated` (already created). `KIRBY_SMOKE_N=140` keeps both
classes present for the hERG classification folds; bump up if a fold loses a class.

### fed01 — Noise heterogeneity across silos
- **Verdict: UNVERIFIED-NEEDS-RUN** (code reviewed, smoke-gated; not executed).
- Best smoke target: lhasa hERG (native binary classification, this experiment's
  intended endpoint). caco2 also works (it binarizes at pKi≥6 — note caco2 is a
  permeability regression set, so the "active/inactive" split is artificial but
  finite).
- Command (lhasa hERG, fastest reps):
  ```
  KIRBY_SMOKE_N=200 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/herg.py --experiments fed01 \
    --reps ecfp4 maccs avalon --output tests/results/_smoke/federated
  ```
  (lhasa hERG needs `~/repos/dphil/data/hERG_lhasa_training.pkl` locally — see
  lhasa_herg_loader. If that pickle is absent, use caco2 instead.)
- caco2 fallback:
  ```
  KIRBY_SMOKE_N=140 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/caco2.py --experiments fed01 \
    --reps ecfp4 maccs avalon --output tests/results/_smoke/federated
  ```
- Things to verify in the JSON once run: every `c0..c6` block has finite `mcc`,
  `balanced_accuracy`, `f1`, `auroc` (no NaN); `n_flipped` ≤ n_train per sigma;
  detection rates in [0,1]; smooth-factor sweep finite.

### fed02 — Representation diversity as silos
- **Verdict: UNVERIFIED-NEEDS-RUN**.
- Needs ≥3 of {ecfp4, mordred, molformer, pdv, chemberta/avalon} as separate silos,
  so the `--reps` set must include several of those. caco2's general list has them
  (uses avalon for silo E via fallback).
- Command (caco2):
  ```
  KIRBY_SMOKE_N=140 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/caco2.py --experiments fed02 \
    --reps ecfp4 mordred molformer pdv avalon \
    --output tests/results/_smoke/federated
  ```
- Verify: per-silo single-rep metrics finite, `best_silo` chosen, KIRBy hybrid and
  top-tier hybrid metrics finite, FLuID ensemble metrics finite.

### fed03 — Synthetic private datasets
- **Verdict: UNVERIFIED-NEEDS-RUN**.
- Dispatch: caco2 → Part B (asymmetric rep silos: structural/physicochemical/learned
  rep groups). lhasa hERG → Part A (source-based, loads its own Lhasa silos). Part C
  (antibody) only triggers for ddg_combined / abagym_HER2 / graphinity_foldx /
  flab_thermo_tresanco.
- Command (caco2, Part B path):
  ```
  KIRBY_SMOKE_N=140 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/caco2.py --experiments fed03 \
    --reps ecfp4 mordred molformer pdv atom_pair graph_kernel \
    --output tests/results/_smoke/federated
  ```
  (Part B asymmetric silos need: structural=ecfp4/graph_kernel/atom_pair,
  physicochemical=mordred/pdv, learned=molformer/chemberta — supply enough of these.)
- Verify: each fold's per-condition RMSE/MCC finite; aggregated mean±std finite;
  `n_folds_completed` > 0.

### fed04 — Cluster-aware importance
- **Verdict: UNVERIFIED-NEEDS-RUN**.
- Command (lhasa hERG preferred, else caco2):
  ```
  KIRBY_SMOKE_N=200 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/herg.py --experiments fed04 \
    --reps ecfp4 maccs avalon --output tests/results/_smoke/federated
  ```
- Verify: per-cluster importance profiles finite and normalized; profile-comparison
  similarities in a sane range; smooth-factor sweep finite.

### fed05 — Antibody federated
- **Verdict: NEEDS-DATA (expected) — UNVERIFIED**.
- fed05 self-loads three antibody silos via `load_data`: `ddg_combined`,
  `abagym_HER2`, `flab_thermo_tresanco`. These are antibody/DDG datasets that are
  almost certainly **not present locally** (the task brief flagged this). The code
  handles it cleanly: `_load_silo` catches the load exception, returns None, and
  `run()` returns `{'error': 'Only N silos loaded, need >= 2'}` (lines 172-173) —
  no crash, no NaN, just an explicit skip. That is the correct NEEDS-DATA behavior.
- Command (will report NEEDS-DATA if silos absent):
  ```
  KIRBY_SMOKE_N=140 /Users/apunt/anaconda3/bin/python \
    tests/experiments/federated/caco2.py --experiments fed05 \
    --output tests/results/_smoke/federated
  ```
  (dataset arg is ignored by fed05 — it loads its own silos.)
- To actually exercise fed05, at least two of the three antibody silo datasets must
  be locally loadable. I could not check their presence (no Python). Capture the
  exact per-silo `FAILED — <error>` lines from stdout to confirm which are missing.

---

## `_shared.py` issues found (report only — owned by another process)

- **Latent design smell, not a bug**: the lhasa/herg rep_names mismatch documented
  above originates partly in `run_all`'s `rep_names = rep_names or []` combined with
  `get_representations` requiring a string. I fixed it on the federated side
  (`load_lhasa_herg_data` now normalizes), so `_shared.py` does **not** need a
  change. Flagging only so the owner is aware that `get_representations` silently
  diverges between "list-style" and "comma-string-style" callers.
- No other `_shared.py` bug found in the federated code paths reviewed
  (`_smoke_subsample` and `_finalize_load_data` look correct: deterministic
  evenly-spaced row cap, not a random split; preserves reps/labels/groups/smiles
  alignment).

---

## Summary table

| Exp   | Verdict (run step blocked)      | Smoke dataset(s)        | Code change |
|-------|---------------------------------|-------------------------|-------------|
| fed01 | UNVERIFIED-NEEDS-RUN            | lhasa_herg / caco2      | smoke-gated grids+folds+silos |
| fed02 | UNVERIFIED-NEEDS-RUN            | caco2                   | smoke-gated folds |
| fed03 | UNVERIFIED-NEEDS-RUN            | caco2 (Part B)          | smoke-gated folds (×3) |
| fed04 | UNVERIFIED-NEEDS-RUN            | lhasa_herg / caco2      | smoke-gated grids+folds+silos |
| fed05 | NEEDS-DATA (expected) — UNVERIFIED | self-loads antibody silos | none needed (graceful skip) |

Plus: bug fix to `load_lhasa_herg_data` rep_names handling; `--reps` flag added to
the three per-dataset runners.

**All numbers in any future run must still be verified** — this report contains
**zero** measured values because the interpreter could not be run here.
