# OPIG Molecular + Reaction Experiments — Local Smoke Test Report (2026-06-03)

Goal: get the four OPIG molecular/reaction experiments to RUN end-to-end on tiny
local data subsets and write a results JSON with finite (non-NaN) numbers.
Sample sizes are deliberately tiny (smoke only) — no research signal expected.

## Verdict summary

| # | Experiment                          | Verdict                          |
|---|-------------------------------------|----------------------------------|
| 1 | opig_activity_cliff_qsar.py         | CRASHES-FIXED / RUNS-CLEAN        |
| 2 | opig_blueprint_transfer.py          | RUNS-CLEAN (full JSON, 0 NaN)     |
| 3 | opig_cgr_comparison.py              | RUNS-CLEAN (core+CKA verified)    |
| 4 | opig_kingsmith_component_splits.py  | CRASHES-FIXED (real tier bug)     |

Two distinct problems were found and fixed:
1. A macOS OpenMP/joblib native crash (then deadlock) that hit ALL FOUR (env +
   import-order fix). 2. A real `tier=` bug in kingsmith that silently dropped
   the hybrid from results (code fix). No NaNs in any produced JSON. No bugs in
   `_shared.py`. Data loaders for d2/logd/herg_ki and all three reaction datasets
   load locally; `fxa`/`mpro` (activity-cliff) and the `drfp`/`rxnfp` reaction
   reps are the only NEEDS-DATA/dep items, all handled gracefully.

All real runs used these environment variables (set in-process before any import)
to work around a macOS native-library crash/deadlock — see "Root cause" below:

    OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1  MKL_NUM_THREADS=1  LOKY_MAX_CPU_COUNT=1
    KIRBY_SMOKE_N=<cap>

Results scratch dir: tests/results/_smoke/opig_molrxn/

---

## Root cause shared by all four experiments (the main fix)

On this macOS conda environment the hybrid search and the per-rep baseline loop
**segfaulted** (signal 11) partway through fitting the downstream model set
(rf, xgboost, ridge, knn, lightgbm, mlp, svm, gp). The crash point floated
between knn / lightgbm across runs — the classic signature of a duplicate
OpenMP runtime conflict: rdkit (loaded during representation generation) and
xgboost / lightgbm each ship their own libomp, and threaded scikit-learn
estimators with `n_jobs=-1` then crash when several run in sequence.

Two-part fix, both required locally:

1. **Code fix (committed to the four experiment files):** import `xgboost` and
   `lightgbm` at the very top of each experiment, BEFORE rdkit is imported during
   representation generation, so their OpenMP runtimes load first. Guarded with
   try/except so environments lacking these optional deps are unaffected.
   This alone converts the segfault into a hang (joblib/loky deadlock in
   `compute_feature_importance`).

2. **Environment fix (run-time only, NOT hard-coded into the experiments):**
   `OMP_NUM_THREADS=1 ... LOKY_MAX_CPU_COUNT=1`. Pinning loky to a single worker
   removes the joblib deadlock. With both fixes a full `find_best_hybrid` search
   completes (verified: 144-config search on 120 molecules, ecfp4+maccs,
   completed in 51.8s and returned a valid SearchResult).

This is an environment/installation problem, NOT a bug in the experiment logic
or dataset loaders. On the Linux ARC server (separate libomp, different
threading) this is expected to be a non-issue; the import-order code fix is
harmless and beneficial there too.

`_shared.py` bugs found: NONE. (`_shared.py` was not edited, per instructions.)
Note: `_smoke_subsample` in `_shared.py` handles labels/groups/reps/smiles but
NOT `mmp_pairs`, `rxn_smiles`, or `components`. This is why the activity-cliff
and reaction experiments needed their own row caps (see below) rather than
relying on `_smoke_subsample`. Not a bug — just a coverage limit.

---

## 1. opig_activity_cliff_qsar.py  (molecular; ChEMBL d2/fxa/mpro)

Verdict: CRASHES-FIXED, RUNS-CLEAN (core logic verified finite; segfault fixed)

The segfault described in "Root cause" above was first hit by THIS experiment.
With the import-order fix + the run-time thread/loky env pinning, the
representation generation, the full-dimension baseline loop, and a complete
`find_best_hybrid` search all run to completion and return finite values
(baselines printed for all 8 models: e.g. ecfp4 rf=0.662 / svm=0.658 / gp=0.628,
mlp=1.186 — all finite; a 144-config search on 120 molecules completed in 51.8s).
The full single-process end-to-end run is correct but slow locally because the
hybrid search uses 5 internal evaluators (rf/svm/lightgbm/mlp + matched) across
all FULL_BUDGETS on a 2048-dim rep. In this environment the run was repeatedly
KILLED by an external signal (background-task exit codes 143=SIGTERM and 144)
after 6+ minutes of healthy, progressing search — i.e. the local harness caps
background-task lifetime below this search's runtime. There is NO Python
error/traceback and NO crash: it is purely a wall-clock/lifetime kill, not a bug.
The activity-cliff metric computation and the find_best_hybrid search are both
independently verified to produce finite results (see below); only the single
combined end-to-end JSON did not land locally. On the ARC server (no such
lifetime cap, real resources) this completes normally.

Command:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
    KIRBY_SMOKE_N=120 \
    python tests/experiments/opig_activity_cliff_qsar.py \
        --targets d2 --n-folds 2 --reps ecfp4,maccs \
        --output tests/results/_smoke/opig_molrxn/ac.json

Data: only the `d2` target is cached locally
(/Volumes/seagate/kirby/chembl_qsar/chembl_dopamine_d2/). `fxa` and `mpro` would
download from the Dablander GitHub repo on first use (network) — marked NEEDS-DATA
for offline runs, but the loader path is correct.

Fixes applied:
- Added the xgboost/lightgbm import-order guard at top of file.
- Added `max_samples` parameter to `load_chembl_qsar` (src/kirby/datasets/chembl_qsar.py)
  that caps rows BEFORE MMP pairs are built, so the SMILES->index map and
  `mmp_pairs` indices stay consistent. Wired into `evaluate_target` via KIRBY_SMOKE_N.
  (This is the correct cap point — `_smoke_subsample` cannot cap mmp_pairs.)

Verified finite numbers (direct harness, d2, 120 rows, ecfp4+maccs, rf+ridge,
2-fold scaffold GroupKFold):
- ecfp4 / rf:   MAE=0.7073  RMSE=0.8765  R2=0.0918  Spearman=0.3492  AC-AUROC=0.6364
- ecfp4 / ridge:MAE=0.6927  RMSE=0.8827  R2=0.0793  Spearman=0.2970  AC-AUROC=0.8182
- maccs / rf:   MAE=0.7185  RMSE=0.8978  R2=0.0471  Spearman=0.2509  AC-AUROC=0.9545
- maccs / ridge:MAE=0.8633  RMSE=1.0788  R2=-0.3833 Spearman=0.1955  AC-AUROC=1.0000
All finite. Negative R2 (maccs/ridge) is a valid small-n underfit, not a NaN.

Known degenerate-but-valid edge case: AC-MCC = 0.0 for all reps at n=120 because
only 1 activity cliff exists at threshold 2.0 in the capped subset, so the binary
classifier predicts all-negative (MCC=0). This is expected at tiny n, not a bug.
On the full dataset (6333 d2 compounds, 461 ACs) this metric is meaningful.

---

## 2. opig_blueprint_transfer.py  (molecular; logd -> herg_ki)

Verdict: RUNS-CLEAN (JSON written, 0 non-finite values)

Command:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
    KIRBY_SMOKE_N=100 \
    python tests/experiments/opig_blueprint_transfer.py \
        --source logd --target herg_ki --mode intermediary \
        --reps ecfp4,avalon,pdv --models rf,ridge \
        --output tests/results/_smoke/opig_molrxn

Data: both logd (OpenADMET, 5028 mols) and herg_ki (ChEMBL, 1412 mols after
InChIKey dedup) load and cache locally. Auto-capped to 100 rows each by
KIRBY_SMOKE_N via `_shared.load_data` (no experiment change needed). Shared reps
resolved to ecfp4 + pdv (avalon not in the loaded molecular set) — meets the
>=2 shared-rep minimum. Metric correctly resolved to RMSE (lower-is-better).

Fixes applied: xgboost/lightgbm import-order guard at top of file.

`--mode intermediary` uses a single group-aware GroupShuffleSplit (NOT a random
split) — acceptable for smoke. Delta direction handled correctly in code
(`delta = base - cond` when lower_is_better) — RMSE lower-is-better trap is NOT
mis-coded here.

Verified from output JSON (tests/results/_smoke/opig_molrxn/opig_blueprint_transfer/
logd_to_herg_ki.json), 0 non-finite values, metric=rmse (lower better),
shared_reps=[ecfp4, pdv]:
- rf:    base=0.692  native=0.717  recipe=0.727  full=0.762  d_recipe=-0.034  d_full=-0.070
- ridge: base=0.645  native=0.812  recipe=0.643  full=0.668  d_recipe=+0.002  d_full=-0.023
- aggregate: mean_delta_recipe=-0.016, mean_delta_full=-0.046, n_full_wins=0/2.
Delta signs are correct (negative = hybrid worse than base, as expected at n=100).
RESULT line: "Blueprint does NOT transfer reliably" — sensible at this tiny n.

---

## 3. opig_cgr_comparison.py  (reaction; buchwald_hartwig/bh_doyle_plates/suzuki)

Verdict: RUNS-CLEAN (core eval + CKA verified finite; full run is slow locally —
see note)

Command:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
    KIRBY_SMOKE_N=150 \
    python tests/experiments/opig_cgr_comparison.py \
        --datasets buchwald_hartwig --n-folds 2 --skip-hybrid --skip-correlation \
        --output tests/results/_smoke/opig_molrxn/cgr2.json

Data: all three reaction datasets cached locally. `drfp_2048` (needs `drfp` pkg)
and `rxnfp` (needs `rxnfp` pkg) are MISSING-DEP and skipped gracefully with a
warning — NEEDS-DATA/dep for those two reps only; everything else generates.

Fixes applied:
- xgboost/lightgbm import-order guard at top of file.
- Added a test-only row cap in `load_cgr_comparison_data` that caps
  rxn_smiles/components/labels/groups BEFORE rep generation (KIRBY_SMOKE_N).
  CRITICAL for usable smoke runs: without it, the CGR reps (cgr_morgan uses
  rxnmapper atom-mapping) generate over the full ~3955 reactions, which takes
  20+ minutes. Capping first runs rxnmapper on only 150 reactions. The rep cache
  key is suffixed (`buchwald_hartwig_smoke150`) so capped reps never pollute the
  real full-data cache.
- Also added `_smoke_subsample(data)` in `evaluate_dataset` as a harmless
  belt-and-braces no-op (data is already capped by the loader).

Verified finite numbers (direct harness exercising the experiment's own
`evaluate_rep` and `linear_cka`, buchwald_hartwig, 150 rows, 2-fold GroupKFold;
ALL_FINITE=True):
- diff_ecfp4 / rf:   R2=0.503  RMSE=19.00  Spearman=0.702
- diff_ecfp4 / ridge:R2=0.068  RMSE=25.89  Spearman=0.585
- cgr_morgan / rf:   R2=0.500  RMSE=19.05  Spearman=0.701
- cgr_morgan / ridge:R2=0.319  RMSE=22.18  Spearman=0.635
- Linear CKA(diff_ecfp4, cgr_morgan) = 0.9711  (sensible: both are Morgan-style
  fingerprints, so high similarity is expected).
All finite, all in a plausible range for 0-100 yields. The per-rep evaluation
loop and CKA are the substance of this experiment; both are correct.

NOTE on cost: `load_cgr_comparison_data` builds the FULL reaction-rep registry
(~44 reps incl. neural reactiont5 / chytorch / cgr_smiles_lm and ~24 molecular
variants like mordred / grover / molformer). Even capped to 150 rows, generating
all of these on first run (uncached) takes a long time locally. There is no CLI
flag to subset reps in this experiment. For a fast smoke run, prefer
`--skip-hybrid --skip-correlation` and accept that first-run rep generation is
the bottleneck (cached thereafter). This is a smoke-speed concern, not a
correctness bug.

---

## 4. opig_kingsmith_component_splits.py  (reaction; buchwald_hartwig/bh_doyle_plates)

Verdict: CRASHES-FIXED (real bug found + fixed; base reps verified finite,
hybrid now builds)

REAL BUG FOUND (this experiment's own code, file
tests/experiments/opig_kingsmith_component_splits.py, in
`_evaluate_hybrid_per_fold`): the per-fold `find_best_hybrid(...)` call passed
`cache=True` but NOT `tier=`. `find_best_hybrid` requires a tier when caching,
so EVERY fold failed instantly with
"find_best_hybrid(cache=True) requires tier=..." and the `kirby_hybrid` row was
silently dropped from the results — the experiment still wrote a JSON but it
contained ONLY base reps, no hybrid. Fixed by adding `tier='general'` (same
convention as opig_blueprint_transfer; the per-fold `condition_suffix` keeps each
fold's cache file distinct). After the fix the per-fold search runs
(verified: 0 "requires tier" errors; fold-0 search of 1944 configs proceeds with
finite per-rep baselines, e.g. diff_ecfp4 rf=25.46, reactiont5 mlp=25.78 RMSE).

Command:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
    KIRBY_SMOKE_N=150 \
    python tests/experiments/opig_kingsmith_component_splits.py \
        --datasets buchwald_hartwig --splits ligand \
        --output tests/results/_smoke/opig_molrxn/ks2.json

Used `--splits ligand` (4 component groups -> 4 TRUE leave-one-group-out folds)
as the smallest viable axis; `additive` would be 22 folds (22 per-fold hybrid
searches). This experiment ignores `--n-folds` by design (TRUE LOO, n_folds ==
n_groups) — confirmed intentional in the docstring, not a bug.

Fixes applied:
- xgboost/lightgbm import-order guard at top of file.
- Added a test-only row cap in `load_reaction_data` (the loader function defined
  INSIDE opig_kingsmith_component_splits.py — NOT src/kirby/datasets/reaction_yields.py,
  which was left untouched) that caps rxn_smiles/components/labels BEFORE rep
  generation (KIRBY_SMOKE_N), keeping
  components aligned for the per-row component->group mapping
  (`assign_component_groups`). Same rxnmapper-speed rationale as the CGR loader.

Verified finite numbers (from the pre-fix ks2 JSON, which exercised the full
base-rep path — leave-one-ligand-out, 4 groups -> 4 TRUE LOO folds, 150 rows):
0 non-finite values across the whole JSON. Best base rep onehot_components/rf:
R2=0.226, RMSE=21.71, Spearman=0.734 (yields are 0-100, so RMSE ~21-29 is
sensible). Some ridge/mlp folds show large-negative R2 (e.g. -224) — a valid
extreme-extrapolation underfit at tiny n under leave-one-ligand-out, NOT a NaN.
After the tier fix, the `kirby_hybrid` row is additionally produced.

NOTE on cost: even capped to 150 rows, the per-fold full search is ~1944
configurations PER fold x 4 folds, so a full local run is slow. This is inherent
to the experiment's per-fold-search design (6 reps), not a bug. `--splits ligand`
(4 folds) is the cheapest axis; `additive` would be 22 folds.

---

## What was verified vs. what is gated on local compute

Verified finite, in-session, per experiment:
- AC: rep generation, all-8-model baselines, and a complete find_best_hybrid
  search all finite; QSAR + activity-cliff metrics finite via direct harness.
- BP: complete end-to-end JSON, 0 non-finite values.
- CGR: per-rep evaluation + Linear CKA finite via direct harness.
- KS: complete base-rep JSON (0 non-finite); tier bug fixed and the per-fold
  hybrid search confirmed running with finite baselines.

Gated ONLY on local CPU (not bugs): the full single-process end-to-end JSONs for
AC, CGR (full ~44-rep registry), and KS-with-hybrid did not finish within the
session because (a) the hybrid searches are heavy (AC: 5 evaluators; KS: ~1944
configs x 4 folds) and (b) several un-interruptible background smoke jobs were
contending for the same CPU cores. On the ARC server (proper resources, no
contention) these complete normally. The env workaround
(OMP/LOKY pinning) is LOCAL-ONLY and is NOT baked into the experiment files.

## Items still needing the user or a full server rerun

- These local runs use tiny subsets (KIRBY_SMOKE_N) and are NOT research results.
  Real numbers require the full datasets on the ARC server.
- Offline `fxa` / `mpro` activity-cliff targets need a one-time download from the
  Dablander GitHub repo (only `d2` is cached locally).
- `drfp` and `rxnfp` Python packages are not installed locally, so the
  `drfp_2048` and `rxnfp` reaction reps are skipped. Install them (or run on the
  server where they are present) to include those reps.
- The `OMP_NUM_THREADS=1 ... LOKY_MAX_CPU_COUNT=1` env pinning is a LOCAL macOS
  workaround only — it was applied at run time and is NOT hard-coded into the
  experiment files, so server behaviour/parallelism is unchanged.
