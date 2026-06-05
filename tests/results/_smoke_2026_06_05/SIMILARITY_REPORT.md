# Similarity Family — Local Smoke-Test Report (2026-06-05)

Agent: SIMILARITY family. Lane: `tests/experiments/similarity/*` only.
Goal: prove every similarity experiment runs end-to-end on a small slice of
REAL data before the clean ARC re-run.

## How the runs were done

- Driver written **in-lane**: `tests/experiments/similarity/_run_similarity_locally.py`
  (modeled on `tests/experiments/_run_complexity_locally.py`). It calls
  `similarity_helpers.run_all()` with a tiny config and local-only reps.
- Two datasets cover both task types (both molecular, both expose the `ecfp4`
  rep that the cliff experiments require):
  - **REGRESSION:** `logd` (OpenADMET-LogD, `task_type=regression`), reps
    `ecfp4`+`maccs`, capped to 200 rows via `KIRBY_SMOKE_N=200`.
  - **CLASSIFICATION:** `herg` (FLuID hERG binary blocker, `task_type=classification`),
    reps `ecfp4`+`maccs`, capped to 200 rows.
- Run prefix (every run): `PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  KIRBY_SMOKE_N=200 perl -e 'alarm <secs>; exec @ARGV' <python> <driver> <dataset> <exps...>`
- `PYTHONUNBUFFERED=1` is essential: without it Python block-buffers stdout, so a
  segfault or alarm-kill loses ALL output and looks like a silent hang.

### What the driver shrinks (all SMOKE-ONLY, server runs untouched)

`run_all`'s precompute step does NOT honour `config` for models/importance —
`precompute_hybrid_searches` hardcodes `models=None` + `speed='full'`, so
`find_best_hybrid` pulls the FULL model set and the FULL
`SPEED_TIERS['full']` importance/allocation/evaluator lists regardless of config
(`tests/experiments/_shared.py:2686-2688`; `src/kirby/find_best_hybrid.py:1702-1713`,
`4095-4111`). To make a local smoke run both fast and crash-free the driver does
in-memory overrides (none of which touch `src/` or on-disk config):

- `_shared.FULL_BUDGETS=[25]`, `_shared.DISTANCE_BUDGETS=[25]` (precompute reads these).
- `_shared.DISTANCE_EVALUATORS=['knn']` (run_all passes this list into the distance-tier precompute).
- `find_best_hybrid.DEFAULT_MODELS` trimmed to `rf/knn/ridge` (drops gp/mlp/svm/lightgbm/xgboost).
- `find_best_hybrid.SPEED_TIERS['full']`: importance→`['lightgbm_gain']`,
  allocation→`['performance_weighted','top_2']`, evaluators→`['match','rf']`.
  **Dropping `deeplift` from importance is what removes the TensorFlow import**
  (see shared-code finding #1).
- Blanks `DATASET_REPS[ds]['augmentation']` in memory so the smoke run does not
  spend minutes generating server-only / 3D augmentation reps (selformer,
  pharmacophore3d, molclr, usrcat, atom_pair, top_torsion).

### In-lane edit to `similarity_helpers.py` (gated, server-safe)

Added an `os.environ['KIRBY_SMOKE_N']`-gated block in `run_all` so s11 gets
`max_iterations=8, n_seeds=1` under smoke (overridable via
`KIRBY_SMOKE_S11_ITERS` / `KIRBY_SMOKE_S11_SEEDS`). Without this, s11 runs
11 conditions × 3 seeds × 100 Bayesian-optimization iterations locally (minutes
to hours) and, because it has **no mid-run checkpoint save**, a timeout-kill
leaves no output. Gated on the smoke env var → the full server run is unchanged.
(`tests/experiments/similarity/similarity_helpers.py`, in the per-experiment
kwargs block of `run_all`.)

## 1. SMOKE RESULTS

All 12 single-dataset similarity experiments PASS on BOTH a regression and a
classification dataset. 24/24 result JSONs written under
`tests/results/_smoke_2026_06_05/similarity_out/`.

| Exp | What it does (plain English) | REG `logd` | CLF `herg` | Notes |
|-----|------------------------------|-----------|-----------|-------|
| s01 | Correlate representation distance with activity difference (per rep) | PASS 3.2s | PASS 3.2s | No hybrid search; fast. |
| s02 | Same correlation but for the importance-weighted hybrid representation | PASS 5.8s | PASS 11.5s | Built real hybrid, reported delta-SAC. |
| s03 | Why hybrids help: feature concentration / mutual-info of selected features | PASS 9.7s | PASS 3.8s | See decision #2 (lightgbm/mlp "failed" lines are a smoke artifact). |
| s04 | Prediction accuracy on activity-cliff compounds (ECFP4 Tanimoto cliffs) | PASS 5.9s | PASS 7.4s | Molecular-only by design; cleanly skips non-molecular. |
| s05 | k-nearest-neighbour prediction using the hybrid distance | PASS 1.4s | PASS 1.7s | Distance-tier hybrid lookup worked. |
| s06 | Applicability-domain / confidence vs distance | PASS 4.1s | PASS 2.9s | Conformal step correctly SKIPs for classification. |
| s07 | Do noise detectors flag activity cliffs as false-positive noise | PASS 0.7s | PASS 0.7s | Uses ECFP4 cliffs (molecular) or edit-distance cliffs (antibody). |
| s08 | Best embedding dimension for similarity vs for prediction | PASS 2.7s | PASS 2.9s | Honestly suppresses the optima comparison when only one dim tested. |
| s10 | Use importance scores as feature weights for kNN/GP/SVR/q-RASAR | PASS 15.5s | PASS 11.4s | GP/SVR tasks correctly SKIP for classification. |
| s11 | Bayesian optimization (active learning) in the compact hybrid space | PASS 38.1s | PASS 37.6s | See decision #1 (degenerate on binary classification — not a crash). |
| s12 | Generic cross-domain activity cliffs (Euclidean, any domain) | PASS 5.7s | PASS 5.3s | Found cliffs on both. Works for antibody/TCGA too by design. |
| s13 | Clustering quality + retrieval precision per space | PASS 3.8s | PASS 2.3s | |

s09 (cross-dataset summary) is out of scope (it aggregates the per-dataset JSON
files, not a single-dataset run); it is not in `SINGLE_DATASET_EXPERIMENTS`.

### Fixes / iterations made to get to green (chronology)

1. **Augmentation-rep explosion** — first `logd` run spent its whole budget
   generating server-only augmentation reps (selformer is a foundation model;
   pharmacophore3d/molclr/usrcat/atom_pair/top_torsion are slow) before any
   experiment ran. Fixed in the driver by blanking `DATASET_REPS[ds]['augmentation']`
   in memory.
2. **Buffered output hid failures** — added `PYTHONUNBUFFERED=1` and stopped
   piping to `tail` (the pipe + block-buffering lost everything on a segfault).
3. **`exit 139` segfault during precompute** — the macOS OpenMP/TensorFlow
   double-load (see shared finding #1). Removed it by dropping `deeplift` from
   the smoke importance list (no TF import) AND trimming `DEFAULT_MODELS`. After
   this, both datasets ran clean end-to-end.

### Timeout audit (per experiment, from reading the code)

The expensive, shared cost for EVERY similarity experiment is the one-time
`run_all` precompute, not the experiments themselves. The experiments split into
two shapes:

- **Pure repetition over a grid (safe to perl-alarm-kill after N iterations):**
  - s01 — per-rep loop, plus a 200-iteration permutation test
    (`s01_similarity_baseline.py:80,122`). Each rep is independent; killing after
    a couple of reps still proves the path. (Here it finished in 3s, no kill needed.)
  - s11 — outer `for condition` × `for seed_i` × `for it in range(max_iterations)`
    (`s11_bayesian_optimization.py:928,964,422`). This is the one true "kill is
    fine" candidate, BUT it has no mid-run save, so a kill = no JSON. I instead
    shrank it (iters=8, seeds=1) so it COMPLETES — preferred per the brief.
  - s03/s05/s06/s08/s10/s13 — nested loops over budgets × models × k × folds
    (e.g. `s05_knn_prediction.py:121-142`, `s10:168-220`, `s08:103-187`). All
    finished in <16s at n=200, so no kill was needed.
- **Distinct phases (must complete, do NOT kill mid-way):**
  - s10 runs Tasks 1–4 sequentially (kNN → GP → SVR → q-RASAR,
    `s10:322,330,331,333`); s06 runs an AD phase then a conformal phase
    (`s06:118,332`). Killing mid-way would skip later code paths, so these were
    shrunk to complete instead.

Decision: at n=200 with the trimmed config every experiment completes in well
under a minute (except the s11 BO loop, capped to ~38s), so **no run needed an
actual alarm-kill** — all 24 wrote real output. The perl `alarm` was kept only as
a safety wall.

## 2. DECISIONS FOR USER

### Decision #1 — s11 (Bayesian optimization) is degenerate on classification

- **Experiment:** s11 runs active-learning / Bayesian optimization in the
  compact hybrid space — i.e. "pick the next molecule to test to find the best
  binders fastest."
- **Problem:** on a binary classification dataset (`herg`, labels 0/1) the
  optimizer's "best value found so far" jumps to 1.0 on the first positive it
  picks, so every condition reports an identical saturated score (AUC=8.0 = the
  area under a flat best-so-far curve over 8 iterations). It runs without error,
  but the numbers carry no signal — every method ties. s11's own docstring says
  it is "molecular-only … library-screening framing"
  (`s11_bayesian_optimization.py:783-784`), and it already gates on
  `domain in ('molecular','reaction')` and `n>=200`, but it does NOT gate on
  task type.
- **Recommendation:** either (a) skip s11 for `task_type=='classification'` with
  an explicit flag (cheapest, matches the existing skip-with-flag pattern at
  `s11:786-792`), or (b) keep it but document that classification results are
  uninformative. I did NOT change this — it's a scope/design call, not a bug.
- **Evidence:** `s11_bayesian_optimization.py:786-792` (existing gates),
  `:418,459` (best-so-far = max over labels), herg run lines 482-492 (all
  conditions = 8.0).

### Decision #2 — s03 "find_best_hybrid failed" lines are a SMOKE ARTIFACT (no action, just FYI)

- **Experiment:** s03 explains why hybrids help by sweeping downstream models ×
  budgets.
- **What you'll see in the smoke log:** on `herg`, s03 prints
  `find_best_hybrid failed: ... no pool candidate produced a result for
  model=lightgbm` and `model=mlp` (herg log lines 158, 161). Likewise s08/s11/s12
  print `WARNING: model 'xgboost'/'svm'/'mlp'/'lightgbm' not in DEFAULT_MODELS,
  skipping`.
- **Why it's NOT a real bug:** s03 defaults `downstream_models=['rf','lightgbm','mlp']`
  (`s03_feature_concentration.py:55`), but MY smoke driver trims `DEFAULT_MODELS`
  to `rf/knn/ridge`, so lightgbm/mlp legitimately aren't available locally. On
  the server (full model set) these resolve normally. s03 handles the miss
  gracefully (records `{'error': ...}` for that cell and continues). No action
  needed — flagging so it isn't mis-read as a server-side failure.

## 3. SHARED-CODE ISSUES (for the orchestrator — I could not fix these in-lane)

### Shared #1 — macOS OpenMP/TensorFlow double-load segfault (exit 139) in the precompute path

- **Where:** triggered inside `find_best_hybrid` precompute when the `deeplift`
  importance method imports TensorFlow alongside numpy/sklearn/lightgbm.
  `deeplift` is in the default importance list
  (`src/kirby/find_best_hybrid.py:1702`, `SPEED_TIERS['full']['importance_methods']
  = ['deeplift','lightgbm_gain','treeshap']`), and `precompute_hybrid_searches`
  forces this full list regardless of `config` (`tests/experiments/_shared.py:2686-2688`).
- **Symptom:** with the brief's single-thread env vars STILL set, `logd`
  segfaulted (exit 139) the moment a threaded model fit ran after TF was imported
  (sim_logd_c.log lines 34-37: crash right after "Importance pre-computed",
  immediately following the TensorFlow `cpu_feature_guard` banner).
- **Scope:** this is a macOS-local OpenMP conflict; the Linux ARC nodes are
  likely unaffected, so this is primarily a **local-smoke blocker**, not
  necessarily a server bug. But it means: any local smoke run that exercises the
  default importance set will crash unless `deeplift` (TF) is excluded.
- **Proposed fix (orchestrator/shared):** gate `deeplift` out of the importance
  list when `KIRBY_SMOKE_N` is set (or when TF import fails), inside
  `precompute_hybrid_searches` / `SPEED_TIERS`. I worked around it locally by
  monkeypatching `SPEED_TIERS['full']['importance_methods']` in my driver, but a
  proper shared gate would let every family's smoke run use the real config.

### Shared #2 — precompute ignores `config['models']`, `config['importance_methods']`, `config['allocations']`, `config['internal_evaluators']` (DESIGN, not a bug — but a smoke-cost trap)

- **Where:** `precompute_hybrid_searches` hardcodes `models=None`, `speed='full'`
  (`tests/experiments/_shared.py:2686-2688`); find_best_hybrid then expands the
  full `DEFAULT_MODELS` (8 models incl. GP) and the full `SPEED_TIERS['full']`
  importance/allocation/evaluator lists. The distance-tier precompute also passes
  the full `DISTANCE_EVALUATORS=['knn','rf','mlp','lightgbm','svm']` explicitly
  (`similarity_helpers.py:916,939`).
- **Consequence for the fleet:** the "tiny config" recipe in the brief
  (`config={'models':['rf'], 'importance_methods':[...], ...}`) does NOT actually
  shrink the precompute. Every family that goes through `run_all` will run the
  full 8-model + GP + 3-importance precompute unless it ALSO overrides the module
  constants (`FULL_BUDGETS`, `DEFAULT_MODELS`, `SPEED_TIERS`, `DISTANCE_*`) the
  way this driver does. This is intentional (to match the SLURM cache key on the
  server), so the right local-smoke answer is the override approach — but the
  orchestrator should make sure the OTHER families know the config dict alone is
  not enough.

## Files touched (this lane only)

- `tests/experiments/similarity/_run_similarity_locally.py` — NEW local smoke driver.
- `tests/experiments/similarity/similarity_helpers.py` — added a `KIRBY_SMOKE_N`-gated
  s11 BO-loop shrink in `run_all` (server runs unaffected).

Nothing under `src/`, `_shared.py`, `registry.py`, `find_best_hybrid.py`, or any
other family's directory was edited. Nothing committed.
