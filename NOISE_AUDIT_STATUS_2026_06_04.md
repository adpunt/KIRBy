# Noise Experiment Audit + Fix Status (2026-06-04)

Local-only code audit of the noise suite (`tests/experiments/noise/n01–n13`), driven by
(a) the same breakage classes found in the complexity/similarity suites and (b) the
human `AUDIT_NOISE.md` (dated 2026-05-23). **No server was touched; no result JSON was
changed.** Every fix is code-only and smoke-tested locally; results only refresh when the
user reruns on ARC.

## How this was verified
- A tiny local smoke harness was built: `tests/experiments/noise/_smoke_harness.py`. It
  loads a small real dataset with cheap local reps (`buchwald_hartwig` regression /
  `humanness_therapeutic` classification), runs a minimal `find_best_hybrid` precompute,
  and runs an experiment's `run()` end-to-end. It single-threads native libraries
  (avoids a macOS OpenMP `exit 139` segfault) and shrinks the budget grid purely for
  speed — **it is not a results config.**
- Existing result JSONs were triaged as free evidence of degeneracy. **Important:** the
  on-disk JSONs (26–29 May, and 1 Apr for TCGA) predate the committed fix wave
  `1421b3d` (which added the estimator `.fit()`/`.estimate()` API to the shared
  `noise_estimation.py`), so several "empty" results are *stale library output*, not
  current-code bugs — but they still need an ARC rerun.

## Per-experiment verdict

| Exp | Verdict | Headline issue (now fixed in code) |
|-----|---------|------------------------------------|
| n01 | 🔧 fixed | Absolute cliff threshold `sim≥0.55` on Euclidean `1/(1+D)` → **0 cliffs** on non-molecular data; primary metric never computed on 4/5 datasets. Now distribution-relative percentile (top 2%). Antibody branch's `max_edits=5` → percentile too. |
| n02 | 🔧 fixed | Cliff-aware weighting ignored `is_clf` (ran regression noise-scoring on class labels). `compute_reliability_scores` `is_clf` was already threaded. |
| n03 | 🔧 fixed | Hybrid build swallowed a `KeyError('fast')` (removed speed tier) → silently empty hybrid columns; now re-raises config errors loudly. Random-split fallback removed; cache-collision string fixed. |
| n04 | 🔧 fixed | Same absolute-cliff-threshold bug → win-rate was a fake `1.0`. Now percentile + 0-cliff degeneracy guard. Multi-seed (3) injection already present; added `lower_is_better` to output. |
| n05 | 🔧 fixed | Single noise realisation per sigma → now 3-seed mean±std (`f1_mean`/`f1_std`). MAD-based low-sigma ground truth still yields 0 noisy at σ=0.05 (flagged, see below). |
| n06 | ⏸ deferred | BindingDB ground-truth experiment; needs a data download, out of local scope. |
| n07 | 🔧 fixed | Per-rep `is_clf` was already threaded, BUT `_compute_noise_scores` built `ActivityCliffAwareEstimator` without `is_clf` (regression branch on class labels). Now threaded through all call sites. |
| n08 | 🔧 fixed | kNN estimators built without `is_clf` (regression scoring on class labels). Added a signature-aware `_make_estimator`. Degenerate absolute cliff threshold already fixed to percentile near-neighbour. |
| n09 | 🔧 fixed | `speed='fast'` raised `KeyError` (tier removed) → swallowed → entire hybrid arm NaN. Now `'full'`. Weighted-hybrid **scale collapse** (sum-1 weights × ~100 cols ≈ 0.01·N(0,1)) → mean-1 normalisation. |
| n10 | 🔧 fixed | Importance weighting was a **no-op**: `StandardScaler(X*w)` cancels `w`. Fixed to standardize-then-weight (mean-1) in 4 places. MAD ground-truth mask made explicit + configurable (`REGRESSION_NOISE_MASK`), recorded in output. |
| n11 | 🔧 fixed | kNN estimators built without `is_clf` (same class as n08). Grid body verified non-degenerate. Blueprint-on-clean-labels comment corrected. |
| n12 | 🔧 fixed | Existing JSON was empty (`per_importance_method: {}`) — **stale-library output**; current code populates. Swallowed estimator errors now recorded (`_errors`); `_weight_features` confirmed NOT a no-op. |
| n13 | 🔧 fixed | MLP "remediated" rows silently equalled the noisy baseline (sklearn MLP ignores `sample_weight` via `except TypeError`); MLP now excluded from the sample-weight track and flagged. `corrupt_idx` now returned; partial-noise axis made explicit; partial clf already does real subset label-flipping. |

## AUDIT_NOISE.md item status (re-verified against current code; line numbers had moved)

**P0**
- P0#1 (n07 per-rep `is_clf`): **ALREADY-FIXED** in the per-rep eval path; but a *related* gap (`_compute_noise_scores` estimator) was found and fixed.
- P0#2 (n03 hybrid silently None): audit's exact cause **STALE** (args now passed), but a real swallowed `KeyError('fast')` on the same path was found and fixed; production JSONs prove the hybrid populates.
- P0#3 (n02 reliability `is_clf`): **ALREADY-FIXED**.

**P1**
- P1#4 (n09 weighted-hybrid scale collapse): **CONFIRMED-AND-FIXED** (mean-1 normalisation).
- P1#5 (blueprint chosen on clean labels): n09 re-searches per fold on noisy labels (fair); n10/n11/n12 use the clean-label blueprint by design — comments **corrected** to call it a best-case hybrid, not leak-free.
- P1#6 (n10 MAD ground-truth mask): **CONFIRMED-AND-DOCUMENTED** — made explicit/configurable and recorded in output.
- P1#7 (n04 single noise realisation): **ALREADY-FIXED** (3-seed loop with mean±std).
- P1#8 (n13 partial-noise clf semantics): **ALREADY-FIXED** (real subset label-flipping).
- P1#9 (n13 MLP silent unweighted fallback): **CONFIRMED-AND-FIXED**.

**P2**
- n02 top3 overwrite: **ALREADY-FIXED**. n08 `.3f` crash: **STALE** (already guarded). n08 estimators no clf branch: **FIXED**. n13 `corrupt_idx`/partial axis: **FIXED**. n05 single realisation: **FIXED**. n09 retention>100%: verified direction-correct. n10 importance on noisy labels + scaler cancellation: **FIXED** (cancellation) / documented (noisy-label importance is realistic).
- n04/n06 raw dict vs `make_structured_output`: **REFUTED-IN-CONTEXT** — n05/n07–n13 also return flat dicts, so wrapping only n04 would be inconsistent; added the missing `lower_is_better` field instead.

## Still needs the user / lead
1. **ARC rerun of the whole noise suite** — every result JSON is stale (predates these
   code fixes and, for n08/n12, predates the committed shared-library fix). Nothing is
   "fixed" until rerun.
2. **n05 / n10 MAD ground-truth threshold (design decision):** at σ=0.05 the MAD mask
   marks ~0 samples noisy, so detection F1 is 0 there. n10 now exposes
   `REGRESSION_NOISE_MASK` (`'mad'` default vs `'all'`). Decide whether low-sigma should
   use `'all'` (literal injected set) or drop σ=0.05. Not changed unilaterally — it alters
   ground-truth semantics.
3. **Cliff percentile (`SIMILAR_PERCENTILE`/near-neighbour 2–5%)** is a research knob used
   by n01/n04/n08 — review cliff prevalence after the rerun.

## User decisions applied (2026-06-04)
- **MAD ground-truth noise threshold: KEPT** (n05/n10 default; n10 also exposes
  `REGRESSION_NOISE_MASK='all'`). Caveat accepted: σ=0.05 marks ~0 noisy → F1≈0 there.
- **Activity-cliff knob is now SWEPT** in n01/n04/n08: `AC_PERCENTILES=[1,2,5,10]`,
  `PRIMARY_AC_PERCENTILE=2.0`. New additive `ac_percentile_sweep` output reports cliff
  prevalence + the headline cliff metric per knob (reuses computed scores, so cheap; n01's
  expensive per-model stratified prediction stays at the primary knob only). n08's primary
  switched 5%→2% for cross-experiment consistency. Smoke prevalence monotonic
  (e.g. n04 21.5/38.5/65/83.5%). Guard test: `_smoke_ac_sweep.py`.
- **n13 now tests ALL 12 LNL methods** (`LNL_METHODS = list(LNL_REGISTRY.keys())`), not the
  4-method pre-hybrid shortlist — so the noise-distribution × method result isn't biased by a
  shortlist chosen before n09's hybrid-aware reranking. ~3× n13 track-A training cost.
  Regression-only methods still gated out for classification (10 methods run on clf).
- **n09 reuse:** the shared hybrid precompute cache and `lnl_methods.py` are reused
  automatically; n09's trained-model compute is NOT reused by n10/n13 (they train their own);
  n13 no longer depends on n09's ranking at all (full-registry decision above).

## Deep verification pass (2026-06-04, after "test the untested" request)
- **n06 (BindingDB ground-truth) was HARD-DISABLED** and had never produced a result —
  `run()` raised `RuntimeError("N06 requires data['replicate_groups']")` unconditionally, and
  its docstring wrongly claimed TDC pre-aggregates away replicates. In fact the cached
  `BindingDB_Kd` (52,272 rows) retains **3,487 (drug,target) pairs with disagreeing
  replicates** (mean replicate std 0.985 pKd). Fixed: `build_replicate_groups()` derives
  replicate IDs from `metadata['drug_ids']`+`['protein_ids']`; correlation now computed over
  the has-replicate subset (a constructed case proved the old full-vector Spearman gave −0.074
  vs the correct 1.0); GP subsample no longer shatters replicate groups. Ran on a 1000-row real
  slice: `distance_weighted_residual` ρ=0.608 (p=1.7e-64, n=626) vs true replicate disagreement
  — n06's first-ever valid number. Smoke: `_smoke_n06.py` (synthetic, green). Std arithmetic
  hand-checked: {5,7}→1.0, {2,2,8}→√8. Flagged (shared-infra, not edited): the 30K random
  subsample in `_shared.py` shatters replicate groups (1,502 pairs still survive, so n06 works,
  but a replicate-aware subsample would be cleaner) and the stale "pre-aggregated" docstring in
  `datasets/dta.py`.
- **Hand-trace of the under-tested experiments' correctness-critical logic** (`_handtrace_noise.py`,
  19/19 checks): n09 retention (RMSE clean/noisy, R² noisy/clean — both <100% when degraded,
  matched per-condition baseline); n07/n10/n11 recovery `(denoised−noisy)/gap` (direction-correct
  for both metric senses: +80% partial, 0% no-op, −50% worse, +100% perfect); n10 detection F1
  matches sklearn with safe denominators; n08 cliff mask + `is_clf` estimator routing; n04 planted
  cliff. No new bugs found — the formulas are correct, now proven by worked example not just "it ran".

## New local artifacts (not committed by these agents)
- `tests/experiments/noise/_smoke_harness.py` — reusable tiny end-to-end harness.
- Smoke suites (all green locally): `_smoke_n01_n03.py`, `_smoke_n04_n07.py`,
  `_smoke_n08_n12.py`, `_smoke_n09_n13.py`, `_smoke_audit_n02_n03_n07.py`,
  `_smoke_audit_n04_n05_n13.py`, `_smoke_audit_n09_n10.py`.
