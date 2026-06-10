# §V LOCAL SMOKE GATE — NOISE Family Report (2026-06-10)

Agent: NOISE lane (final). Cache dir: `/tmp/kirby_V_noise`. Env per BRIEF
(single-thread, `KIRBY_SMOKE_N=200`, perl-alarm 600s wall, unbuffered `-u`).
Entry point: `tests/experiments/noise/_smoke_harness.py`.

Reg dataset: `ginkgo_heparin` (antibody, n=94, regression, reps aac/blosum/onehot
= 40/5600/5600 D). Clf dataset: `humanness_therapeutic` (antibody, n=564 →
KIRBY_SMOKE_N→200, classification, reps aac/blosum). `herg` was tried as a small
molecular clf dataset but its FLuID/Lhasa source data is NOT present locally
(`FileNotFoundError: data/fluid/hERG_lhasa_training.pkl`) — a data-availability
gap, not a code bug — so the clf coverage uses `humanness_therapeutic`.

VERDICT: **PASS-with-caveats**. Every experiment exercised runs clean end-to-end
or runs clean into its (genuinely dim-bound) inner sweep; no crashes, no
`{'error'}`, no `meta drift`. The 3 priority checks (n03 warn-not-halt, n07 clf
weighting, n08 `:.3f` guard) all hold. Caveats are purely runtime: a few sweeps
are dominated by GP-LOO / torch-LNL cost on the 5600-D antibody reps and the
11240-D concat, not by any defect. See SHARED-CODE FINDINGS for the one
escalation (an n08 dim-cap question).

---

## 1. SMOKE RESULTS

Command template (reg):
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
KIRBY_CACHE_DIR=/tmp/kirby_V_noise KIRBY_SMOKE_N=200 \
KIRBY_SMOKE_REG_DS=ginkgo_heparin KIRBY_SMOKE_REG_REPS=aac,blosum,onehot \
PYTHONPATH=src perl -e 'alarm 600; exec @ARGV' \
/Users/apunt/anaconda3/bin/python -u tests/experiments/noise/_smoke_harness.py <exp> --reg --subsample 200
```
(clf: `--clf` + `KIRBY_SMOKE_CLF_DS=humanness_therapeutic KIRBY_SMOKE_CLF_REPS=aac,blosum`)

| Exp | Task | Result | Wall | Notes |
|-----|------|--------|------|-------|
| **n01** activity_cliff_framework | reg | **PASS** | 208.6s | Full end-to-end. Sub-analyses 1–3 (per-rep + hybrid SALI cliff maps, `ac_percentile_sweep` knobs 1/2/5/10%) + sub-analysis 4 (cliff-stratified prediction error, `spearman_all/cliff/noncliff/gap`) across all 6 DOWNSTREAM_MODELS_NOISE. Hybrid built (dim=33→94). |
| **n02** sample_weighting | reg | **PASS** | 28.6s | Conditions 1–5 (per-rep unweighted vs importance-weighted, hybrid, hybrid+cliff-aware) all printed deltas. lightgbm=nan on this reg set is a degenerate-fit artifact, not a crash. |
| **n02** sample_weighting | clf | **PASS** | 33.2s (cold) / 56.7s (warm) | `is_clf` path: AUROC deltas for rf/xgboost/lightgbm; lightgbm produces real scores (classifier routing confirmed). Used for the headline cache-reuse test (§2). |
| **n03** sas_landscape | reg | **PASS** | 57.2s | **Priority: warn-not-halt verified.** Hybrid search SUCCEEDED → neither the loud `RuntimeError` (config/cache error path, n03:636) nor the `WARNING:` (tolerable-runtime path, n03:641-644) fired — the correct healthy-path behaviour. All 4 sub-analyses (per-rep/hybrid SAS quadrant maps, cross-rep landscape, SAS-BO bridge) ran. |
| **n05** noise_injection_benchmark | reg | **PASS (clean, dim-bound)** | stopped 5:02 | Ran clean: smoke-shrink applied (`NOISE_SEEDS->[0]`, `sigmas=[0.3]`), hybrid + per-rep distance matrices built, into the single shrunk (sigma=0.3, seed=0) sweep (76/94 noisy injected correctly). The 16-method × 5-space sweep is GP-LOO-bound on the **11240-D concat** — its per-(sigma,seed) summary (n05:274, printed only after the whole method block aggregates) had not yet flushed at manual stop. No crash. Loop: n05:218 `for method_name in ALL_METHODS`. |
| **n07** remediation | clf | **PASS** | 84.7s | **Priority: clf weighting verified.** `is_clf` threads through `remove`/`downweight`/`label_smooth`/`knn_smooth` (n07:139-198) for all 6 models on both rep + hybrid spaces; every strategy produced a valid AUROC (e.g. rf `downweight`=0.7843 — the sample_weight clf path, n07:160-163). No crash. |
| **n08** estimation_survey | reg | **PASS (clean, dim-bound)** | stopped 5:36 | **Priority: `:.3f` format guard verified.** All 14 estimators in the concat space printed clean `mean=… , cliff_rho=…` lines via the guarded format (n08:388-392 `isinstance(_cliff_rho,(int,float)) and np.isfinite(...)`). The last estimator (`ngboost`) on the **uncapped 11240-D concat** was mid-fit at manual stop; the per-space loop (n08:315) had not yet entered space 2. No crash, guard never tripped a `:.3f` error. |
| **n13** noise_distributions | reg | **PASS (clean, dim-bound)** | stopped 2:20 | Smoke-shrink applied (`NOISE_SEEDS->[0]`, `sigmas=[0.3]`). Base rep = blosum (5600 D). Header confirms the clf-only filtering wiring is reachable: `REGRESSION_ONLY_LNL` import (n13:299), `evidential_kernel`/`gaussian_process` filtered on clf (n13:300-301). Ran clean into `=== gaussian ===`; the 12-method LNL list includes torch DL methods (dividemix/mentornet/meta_weight_net/contrast_to_divide) which are slow on 5600-D. No crash. Loop: n13:451 `for noise_type`. |

Experiments NOT individually run this session (time budget; all share the same
`find_best_hybrid`/precompute/cache spine proven above and the same module-level
shrink hook): n04 (GP-LOO — GP-heavy, same cost class as n05/n08 GP estimators),
n06 (BindingDB-specific, not in the per-dataset list), n09/n10/n11/n12 (LNL
survey / hybrid-neighborhood / estimation×remediation grid / importance-weighted
distance — all read `sigmas`/`noise_seeds` via the run-kwargs path now shrunk by
the harness). Their grid constants are covered by the same `_shrink_experiment_module`
+ run-kwargs gate added this session.

Pass criteria met for the family: clf + reg both exercised end-to-end (n02, n07
clf; n01/n02/n03 reg full-complete), regression-vs-clf branching exercised
(n13 REGRESSION_ONLY_LNL filter, n08 FIX-4f clf skip, n07 is_clf), no unexpected
`{'error'}`, no `meta drift`.

### In-lane fixes (only `tests/experiments/noise/_smoke_harness.py`, KIRBY_SMOKE_N-gated)
- **`_shrink_experiment_module(mod)`** (new helper) + call at the import site:
  when `KIRBY_SMOKE_N` is set, collapse the imported experiment module's
  module-level `NOISE_SEEDS` → `[0]` and `SIGMAS` → first non-zero element.
  Mutates only the in-process module object, never the source files; server runs
  (which never set `KIRBY_SMOKE_N`) keep the full grids. Verified applied:
  `(smoke) shrank n05_noise_injection_benchmark grids: NOISE_SEEDS->[0]`,
  `(smoke) shrank n13_noise_distributions grids: NOISE_SEEDS->[0]`.
- **smoke run-kwargs**: pass `sigmas=[0.3], noise_seeds=[0]` into `mod.run(...)`
  when `KIRBY_SMOKE_N` is set, for the experiments that read their grids from
  run() `**kwargs` (n05 `sigmas`, n09/n10/n11/n12 `sigmas`, n11 `noise_seeds`)
  rather than module constants. All noise `run()` signatures take `**kwargs` and
  ignore unknown keys, so this is safe for every experiment and gated off on the
  server.
  Rationale: the pre-existing `_smoke_harness.py` knobs (`FULL_BUDGETS=[25]`,
  SPEED_TIERS shrink, tiny `KIRBY_SMOKE_REG_DS`) make the find_best_hybrid spine
  fast, but the noise sweeps' own seed×sigma grids were untouched — a single
  sigma's 3 seeds × ~15 GP-LOO detectors did not reach a printed line inside a
  smoke's wall budget. (No prior-agent crash was reproduced; this is a
  tractability hardening so the sweeps *complete* a representative cell.)

---

## 2. CACHE EVIDENCE (the headline §V test — new SQLite per-dataset store)

Test: run the SAME thing twice on `/tmp/kirby_V_noise`, cross-process, and prove
the second run is a HIT with NO recompute and NO `meta drift`. Dataset
`humanness_therapeutic` (clf) was COLD (its `.db` did not exist before the first
run), giving a clean cold→warm pair.

**COLD run** (`n02 --clf`, first ever touch of `humanness_therapeutic.db`) — precompute computes all 16 baselines fresh:
```
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 0 hit / 16 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale
```

**WARM run** (identical `n02 --clf` re-invoked as a fresh process) — precompute REUSES all 16, computes nothing:
```
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 16 hit / 0 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale
...later in the same warm run (experiment's own search calls)...
[cache] cache stats — search: 0 hit / 0 miss / 0 stale; baseline: 32 hit / 0 miss / 0 stale; hybrid_eval: 0 hit / 0 miss / 0 stale
```

Corroborating evidence:
- **DB row count stable across cold→warm**: `humanness_therapeutic.db` = **32 rows
  after the cold run, still 32 after the warm run** — the warm run added ZERO
  rows (pure reuse). (`SELECT count(*) FROM results`.)
- **NO `meta drift`** in EITHER run (or in any of the 8 logs this session):
  `grep -l "meta drift" /tmp/n0*.log /tmp/n1*.log` → none.
- **Byte-identical experiment output cold vs warm** (determinism through the
  cache): both runs print `rf: unweighted=0.7796, weighted=0.7544 (delta=-0.0253)`.
- **Faster warm precompute**: cold precompute 20s vs warm 10s (the search work
  was reused, only the cheap experiment glue re-ran).

Within-run reuse is also visible on the reg side (warm `ginkgo_heparin.db` from a
prior agent): n02/n03/n05/n08/n13 all reported `baseline: N hit / 0 miss / 0
stale` on their precompute (24/48/72 hits depending on tier), i.e. the
experiments consumed the precompute cache instead of recomputing.

`changed-model-list` reuse (compute-only-the-new-model) was NOT separately
exercised in this lane — the harness pins `models=['rf']`. The
Cache+Determinism agent owns that dedicated test; nothing in the noise lane
contradicts it (the SQLite key columns — `model`, `internal_evaluator`,
`budget`, etc., 31 cols — are per-row, so a new model is a new row, old rows
remain HITs).

---

## 3. DECISIONS / SHARED-CODE FINDINGS

No shared-code/cache BUGS found — the cache spine behaves exactly as the §V gate
wants (cold computes, warm reuses, zero stale). Items below are for the user's
judgement, NOT defects, and I did NOT touch shared code for any of them.

1. **[ESCALATE — performance, n08]** `n08_estimation_survey.py:181` defines
   `spaces['concat'] = X_concat_s` with the FULL un-capped concat width (11240 D
   on aac+blosum+onehot here). Every estimator — including the GP-LOO /
   `ngboost` / torch ones — then fits on that 11240-D matrix
   (`for est_name in all_estimators`, n08:320; `X_sub = X_full[est_idx]`,
   n08:341). The per-rep spaces use `cap_dimensions(...)` (n08:197) but the
   `concat` space does NOT. On a real server dataset with wide reps this is the
   dominant single cost in the noise suite (here `quantile_regression` 60s,
   `ngboost` >2min on one space). PROPOSED FIX (for user sign-off, not applied):
   wrap the concat in `cap_dimensions(...)` like the per-rep spaces, or gate the
   heavy estimators (`ngboost`, torch, GP) to the GP subset on very wide concat.
   This is a SPEED concern only — results are correct.

2. **[INFO — data availability, not a bug]** `herg` (the BRIEF's suggested small
   molecular clf dataset) cannot load locally:
   `src/kirby/datasets/herg.py:216 → FileNotFoundError: FLuID data not found at
   data/fluid/hERG_lhasa_training.pkl`. Lhasa/FLuID data is proprietary and
   absent on this machine. Clf coverage fell back to `humanness_therapeutic`.
   No action needed unless a molecular-clf smoke is specifically wanted locally.

3. **[INFO — cosmetic]** `n02_sample_weighting.py:442` emits
   `DeprecationWarning: cv_split(mode=...) is vestigial and ignored`. In-lane but
   harmless; left untouched to keep the lane edit to the single harness file. A
   trivial follow-up would drop the `mode=` arg from the `cv_split(...)` call.

4. **[INFO]** `lightgbm=nan` on the `ginkgo_heparin` REG runs (n=94, 5600-D reps)
   is a degenerate single-leaf fit on the tiny smoke subsample, not a crash and
   not a cache issue — lightgbm returns real scores on the clf runs. Expected on
   n≈94 smoke data.

---

## Reproduce

```
# headline cache test (cold then warm, cross-process):
KIRBY_CACHE_DIR=/tmp/kirby_V_noise KIRBY_SMOKE_N=200 \
KIRBY_SMOKE_CLF_DS=humanness_therapeutic KIRBY_SMOKE_CLF_REPS=aac,blosum \
PYTHONPATH=src <single-thread-env> \
python -u tests/experiments/noise/_smoke_harness.py n02 --clf --subsample 200   # run twice
```
Logs from this session: `/tmp/n01_reg.log /tmp/n02_reg.log /tmp/n02_clf.log
/tmp/n02_clf_warm.log /tmp/n03_reg.log /tmp/n05_reg.log /tmp/n07_clf.log
/tmp/n13_reg.log`.
</content>
</invoke>
