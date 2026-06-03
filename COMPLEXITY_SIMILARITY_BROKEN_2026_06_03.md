# Complexity & Similarity Experiments — What's Broken (non-green roster)

**Created:** 2026-06-03. Companion to `COMPLEXITY_SIMILARITY_RESULTS_2026_06_02.md`.
Covers every experiment that is NOT a clean positive. Excludes the green ones
(c01, c02, c04, c07, c15, s01, s02 — see the results doc).

Every number below was re-derived directly from the JSON/source THIS session (not copied
from a prior summary). Two earlier agent claims were wrong and are corrected inline (c11, c19).

Severity legend: ☠️ DEAD (no usable output) · 🧪 DEGENERATE METRIC · 🔧 STRUCTURAL/MISSING ARM · ⚠️ WEAK/REVERSES

---

## FIX STATUS (updated 2026-06-03, code only — NOTHING re-run on ARC yet)

Constraint: server ops are the user's. All fixes below are **code-level, verified
locally by import/syntax + real-data smoke tests** (`tests/experiments/smoke_tests_broken_fixes.py`).
None changes the result JSONs until the user deploys and re-runs on ARC.

**Complexity end-to-end run verification (2026-06-03, local, small dataset flab_thermo_jain
n=137, reps aac/blosum/onehot, 1 model, tiny budgets):** all 19 complexity modules import with
`run()`. c01 ran fully → MI now real non-zero (aac 0.085 / blosum 0.171 / onehot 0.180; was 0.0).
c10 ran fully (exit 0) → real NDS values (e.g. legacy NDS=−0.030, retention 86%; was the stub).
Note: macOS needs `OMP_NUM_THREADS=1` for the hybrid-search path (unrelated OpenMP segfault; not an
ARC issue). Driver: `tests/experiments/_run_complexity_locally.py`.

**Similarity end-to-end run verification (2026-06-03, local, flab_polyreact_jain_bvp n=137,
reps aac+blosum, OMP_NUM_THREADS=1):** all 5 fixed s-experiments ran to completion and emit the
new fields. s06 (full-tier search) → conformal `undercoverage` +0.091/+0.083 and
`hybrid_genuinely_tighter=True`. s05 → `summary.posthoc_max=+0.0211` (optimistic) vs honest
`fixed_ref=−0.0690` — the optimism gap is now explicit. s08 → self-diag fired
(`n_distinct_dims`/`insufficient_budgets=True`, `optima_differ` suppressed to None). s10 →
`deltas_oriented=True`, all 4 downstream tasks ran. s13 → `metrics_agree=False` (wins retrieval,
loses clustering — the verified mixed finding). Smoke tests: 6/6 pass
(`smoke_tests_broken_fixes.py`, +Fix4 s06 graded error, +Fix5 s10 delta orientation).

| item | status | what changed |
|---|---|---|
| MI estimator (c01/c09/c17/s03) | ✅ PATCHED + smoke-tested | `_shared.py` — replaced collapsing high-dim KSG with de-biased sklearn MI + classification detection |
| cliff thresholds (s07, s12) | ✅ PATCHED + smoke-tested | distribution-relative percentile thresholds (default 2%); fixes both 0% and ~100% degeneracy |
| c19 stale docstring | ✅ PATCHED | docstring now matches hybrid-only design (raw-concat removed by directive) |
| c11 "ensemble" | ✅ PATCHED (docstring) | **CORRECTION:** ensemble arm was removed by directive (no mean baselines), NOT a bug. Docstring fixed. |
| c13 E−B | ✅ PATCHED (docstring) | **CORRECTION:** condition E removed by directive (was identical to D), NOT a measurement failure. Docstring fixed. |
| c16 full-dim NaN | 🔍 DIAGNOSTIC ADDED | root cause (capped_full search failed → no blueprint) only in slurm log; added `capped_full_error` to output so rerun self-diagnoses. Underlying failure still needs the rerun/log. |
| c21 NaN | 🔍 DIAGNOSTIC ADDED | NaN = augmentation search returned no config (bp=None). Added log line; needs rerun to confirm legit-vs-bug. |
| c24 errored | 🔍 DIAGNOSTIC ADDED | per-fold exception was swallowed; now recorded in `fold_errors` in the output so rerun self-diagnoses. |
| c10 noiseInject | ✅ PATCHED + smoke-tested | it's the user's own private pkg (`adpunt/noise_inject`, public+reachable on GitHub). `setup.sh` now clones+installs it (`--no-deps`); SETUP.md §2e documents it. c10 already has the `task_type=='regression'` guard. Needs the user to re-run `setup.sh` on ARC, then rerun c10. |
| s04 / s11 molecular-only | ✅ RESOLVED (user: add molecular datasets) | domain gate is correct by design; they were skipped only because the batch had no molecular dataset. Added `full_mol` mode to `run_similarity_experiments.sh` running s04+s11 via `run_dataset.py` on logd/caco2/herg/qm9 (the 4 molecular datasets in DATASET_REPS; drd2 not registered for this path). Needs the ARC run to produce results. |
| **s05 post-hoc optimism** | ✅ PATCHED + smoke-tested | `summary` now emits BOTH `posthoc_max` (optimistic) AND a pre-registered `fixed_ref` (honest, no grid selection). Weighting confirmed correctly applied (not a no-op) — its neutrality is a real result. Docstring fixed. |
| **s06 clf error + conformal** | ✅ PATCHED + smoke-tested | graded classification error `1-P(true)` replaces degenerate 0/1 indicator; `undercoverage` field added so tighter-but-undercovered intervals can't be misread as better. Regression near-zero corr is a genuine NEGATIVE finding. Docstring fixed. |
| **s08 dim-collapse** | ✅ PATCHED + smoke-tested | self-diagnosing `n_distinct_dims`/`insufficient_budgets`; `optima_differ` suppressed when <4 distinct actual dims (root cause: many budgets collapse to the same dim → SAC ties → noise). Budgets deduped/sorted. Docstring fixed. |
| **s10 RMSE sign trap** | ✅ PATCHED + smoke-tested | `_orient_deltas` (a single recursive pass before reporting) negates every stored `delta*` field for lower-is-better metrics so positive==improvement (buchwald RMSE "+13.077" was a worsening → now −13.08); `results['deltas_oriented']=True`; `_print_summary` no longer double-corrects. Docstring fixed. |
| **s13 mixed finding** | ✅ VERIFIED (no bug) | both metrics + directions correct; weighting applied. Genuine mixed result (retrieval wins, clustering loses). Records winning baseline rep per axis for honest reading. Docstring clarified. |
| remaining weak/reverses (c05/c06/c08/c09/c17/c18/c23) | ⏳ RERUN | not code bugs — genuine findings or need rerun on powered datasets. |

Detail per experiment below; ✅/🔍/⛔ tags repeated inline.

---

## ☠️ DEAD — produced no usable results

### c10_noise_robustness ✅ (FIXED — install wired into setup.sh)
- **What it tests:** noise-injection robustness + feature-selection stability of hybrids vs single reps.
- **What was wrong:** every one of the 9 files is **87 bytes** = `{"error": "noiseInject not installed"}`.
  The import `from noiseInject import NoiseInjectorRegression` (`c10:27`) failed on ARC.
- **Root cause:** `noiseInject` is the user's OWN package (`/Users/apunt/repos/NoiseInject`,
  `github.com/adpunt/noise_inject`), not on PyPI — so the `requirements` pin never installed it on ARC.
- **Fix applied:** `setup.sh` now clones `adpunt/noise_inject` and `pip install --no-deps -e`'s it
  (verified the repo is public + reachable; its only deps numpy/pandas/scipy are already locked).
  SETUP.md §2e documents the manual path. The API c10 uses (`strategy`/`random_state`/`.inject(y,sigma)`
  with `legacy`/`quantile`/`hetero`) is confirmed present; c10 already gates on `task_type=='regression'`.
  Smoke-tested locally (all 3 strategies inject; noise degrades fit).
- **User action:** re-run `setup.sh` on ARC (one-time), then rerun c10.

### c21_single_rep_augmentation ☠️ (on every powered dataset)
- **What it tests:** does uncapped augmentation lift an individual rep without hybridisation.
- **What's wrong:** all powered runs are NaN-saturated — `abagym_VEGF` (3673 NaN tokens), `abagym_Ang2`
  (3691), `abcov_kd` (4834): `aug_mean = nanmean(scores_aug)` over an empty set because **no augmentation
  feature ever survived selection** (`c21:245`). `humanness_therapeutic` returns
  `{"error": "no_augmentation_reps"}` (`c21:97` — no aug reps configured for that dataset).
  Only the n<150 datasets produced numbers, and they were net-negative-to-flat.
- **Root cause (refined):** NaN comes from `bp=None` — `find_best_single_rep_augmented` returned no
  winning config, so every augmented fold scores NaN (`c21.py:204`). This is either legitimate
  (augmentation genuinely didn't beat baseline for these reps) or a search gate; the rerun will tell.
- 🔍 **Diagnostic added** (`c21.py:169`): logs when the search returns no config. humanness still needs
  augmentation reps configured (separate `no_augmentation_reps` error). Rerun to classify legit-vs-bug.

### c24_blueprint_transfer ☠️ (on both powered datasets)
- **What it tests:** does a feature-selection blueprint from one CV fold generalise to other folds.
- **What's wrong:** `abagym_Ang2` and `abagym_VEGF` are **80-byte** `{"error": "insufficient_folds"}`.
  The runner needs ≥2 successful folds (`c24:237`); per-fold blueprint discovery threw exceptions
  ("Own-fold eval failed", `c24:~231`) and was swallowed. The 3 that ran (HER2, abcov_kd,
  immunogenicity, all n≤217) have n_folds=5 but are tiny-n.
- **Fix:** stop swallowing the per-fold exception — log it to find why discovery fails on Ang2/VEGF.
  Rerun once folds succeed.

### s04_activity_cliffs ☠️ → ✅ RESOLVED (run on molecular datasets)
- **What it tests:** kNN RMSE on ECFP4-Tanimoto cliff compounds, hybrid vs per-rep.
- **What's wrong:** all 23 files are **124 bytes** = `{"skipped": true, "reason": "molecular-only
  experiment, domain=antibody/reaction"}`. The metric needs molecular fingerprints; no molecular
  dataset was in this batch. The domain gate (`s04:69`) is CORRECT by design — not a bug.
- ✅ **Decision (user, 2026-06-03): add molecular datasets.** New `full_mol` mode in
  `tests/slurm_scripts/run_similarity_experiments.sh` runs s04 (+s11) via the canonical
  `run_dataset.py similarity <ds> --experiments s04 s11` on **logd, caco2, herg, qm9** (the molecular
  datasets registered in DATASET_REPS; drd2's loader exists but isn't wired into this load path, so it
  was excluded). Produces results on the ARC run.

### s11_bayesian_optimization ☠️ → ✅ RESOLVED (run on molecular datasets)
- **What it tests:** GP-BO sample efficiency in compact hybrid vs high-D fingerprint space.
- **What's wrong:** all 15 files are **121 bytes** = `{"skipped": true, "reason": "domain=antibody"}`.
  Same molecular-only gate as s04 (`s11:758`, allows molecular+reaction). Correct by design.
- ✅ **Decision (user, 2026-06-03): add molecular datasets** — same `full_mol` mode as s04.

---

## 🧪 DEGENERATE METRIC — ran, but the headline metric carries no signal

### Mutual-information estimator is broken everywhere → kills c01(MI), c09(retention), c17, s03
- **What's wrong:** the KSG kNN estimator `estimate_mutual_information_knn(X, labels, k=5)`
  (`c01_rep_characterization.py:66`, same util in c09/c17/s03) returns ~0 across the board:
  - c01 `mutual_info`: **513/567 entries are exactly 0.0 or None** (verified this session).
  - s03 `mi_full`: **0.0 on every rep/dataset** (verified — abagym_VEGF, humanness all 0.0).
  - c09 mutual-information-retention block: all zeros. c17 MI-density: all zeros.
- **Consequence:** every MI-based sub-claim is unsupported — "hybrid concentrates MI" (s03, c17),
  "MI retention inflection" (c09), "MI discriminates rep quality" (c01).
- **Fix:** replace KSG with `sklearn.feature_selection.mutual_info_regression` / `mutual_info_classif`
  on the selected feature block, OR debug why X is over-reduced before the estimate. Re-derive all
  MI claims afterwards. **Until then, drop every MI statement from the writeup.**

### s07_noise_bridge 🧪 — cliff detection saturates at ~100%
- **What it tests:** kNN noise-detection false-positive rate (ground-truth cliffs flagged as noise).
- **What's wrong:** uses **edit-distance ≤5** to define ground-truth cliffs. On antibody sequences that
  differ by a few point mutations, almost every pair qualifies → astronomically many "cliffs"
  (verified: `abagym_VEGF` n_cliff_pairs=**21,782**, `abagym_EGFR`=**81,309**, `tz_her2_multiclass`=**197,777**).
  Contamination saturates near 1.0 and the hybrid-vs-rep delta is never favourable.
- **Fix:** tighten the cliff definition (fewer max_edits and/or a larger activity-gap multiple) so cliff
  prevalence is a small minority, then rerun.

### s12_cross_domain_cliffs 🧪 — cliff detection collapses to ~0% (opposite failure)
- **What it tests:** generic Euclidean cliff detection (`sim ≥ 0.55 AND |Δy| > 2·MAD`).
- **What's wrong:** with `sim = 1/(1+d)` on standardised high-D concat, pairwise d is large so sim ≪ 0.55
  → **13/16 datasets find 0 cliff pairs** (verified). The only non-zero: Ang2=11, immunogenicity=3,
  humanness=2 — too few to measure, deltas are nan/negative.
- **Fix:** make the similarity threshold **distribution-relative** (top-percentile of the pairwise
  similarity matrix) instead of the absolute 0.55. s07 and s12 fail from opposite ends — both need
  the cliff prevalence calibrated into a sane band before any delta is trusted.

### s03_feature_concentration 🧪
- **What it tests:** WHY hybrid similarity works (MI-density ratio of selected vs random dims).
- **What's wrong:** the core MI-density comparison is dead (mi_full=0 everywhere, see above). The only
  surviving signal is that LightGBM-gain importance is highly concentrated (Gini ≈ 0.84–0.98 per agent;
  MI mechanism unverifiable).
- **Fix:** depends on the MI estimator fix above.

---

## 🔧 STRUCTURAL / MISSING ARM — the key comparison isn't in the output

### c11_ensemble_vs_hybrid ✅ (FIXED — was a stale docstring, NOT a bug)
- **What it tests:** feature-level fusion (hybrid) vs best single rep vs PCA-matched control.
- **Resolution:** the docstring promised a model-level "ensemble" arm (one model per rep, average
  predictions), absent from all 9 JSONs. On reading the source, the ensemble/averaging arm was
  **deliberately removed 2026-05-12 per user directive** (`c11.py:196–198`) — averaged baselines are
  not a valid comparator (matches the no-mean-baselines rule). So this was never a bug; the docstring
  was stale. **Docstring corrected** to describe the real conditions (best_single / pca_matched /
  hybrid) and note the intentional removal.
  *(Two earlier summaries were both wrong: "no ensemble arm exists" and "exists but didn't run".
  Truth: existed, removed on purpose.)*
- **What the existing arms show (non-aug, powered):** hybrid > best_single on Ang2 (0.428 vs 0.384),
  humanness (0.925 vs 0.914), EGFR (0.338 vs 0.322); **hybrid loses on VEGF (0.586 vs 0.634)**. Hybrid
  always beats pca_matched.

### c19_importance_weighted_training 🔧 (CORRECTION to earlier summary)
- **What it tests (docstring):** does importance weighting help more on **raw concat** vs **hybrid** features.
- **What's wrong:** the raw-concat arm is **deliberately removed by design directive**
  (`c19:128–131`: "concat baseline removed per design directive — too many reps for full-concat to be a
  meaningful [comparison]") — only `input_spaces['hybrid']` is computed. So the docstring's central
  raw-vs-hybrid question **cannot be answered**, but this is intentional, not a runtime bug.
  *(Earlier note said "add the raw/concat arm" — that contradicts the design decision; instead fix the
  docstring/question.)* On the hybrid space the 5 weighting strategies are all neutral (deltas ±0.01–0.04,
  best_overall = unweighted on VEGF).
- **Fix:** rewrite the docstring/claim to match the hybrid-only design, OR if the raw-vs-hybrid question
  is wanted, reintroduce a dimension-capped concat arm. As-is, the experiment answers a narrower question
  (which weighting on the hybrid — answer: none helps).

### c13_forward_compatibility ✅ (FIXED — was a stale docstring, NOT a bug)
- **What it tests:** per new rep, does adding it help the hybrid (D−C).
- **Resolution:** the E−B "does the hybrid help the rep" arm reads as NaN, but condition E was
  **deliberately removed 2026-05-25** (`c13.py:597–614`) — it was identical to D (a no-op), and the
  intended "hybrid selection helps r alone" needs a different design that isn't implemented. E/E−B are
  emitted as null for backward-compat. So the NaN is intentional, not a measurement failure.
  **Docstring corrected.** The live metric is D−C (near-null, sign-mixed: +0.059 EGFR … −0.095 HER2).

### c16_feature_selection_as_regularization 🔧
- **What it tests:** at the optimal hybrid budget, does model regularization stop mattering (vs a tuned
  full-dimension baseline).
- **What's wrong:** the `best_tuned_full_dim` baseline is **NaN on the powered regression sets** (VEGF 48
  NaN tokens, Ang2 48; EGFR/humanness clean). So the hybrid-vs-tuned-full-dim comparison only resolves on
  2 datasets. The sensitivity-decreasing trend itself is fine (VEGF Ridge std 0.004@b10 → 0.089@b250).
- **Fix:** make the full-dim tuned model produce a score on VEGF/Ang2 (likely OOM/time-out on full concat).

---

## ⚠️ WEAK / REVERSES — ran cleanly but the claim doesn't hold on powered data

### c05_learning_curves ✅ FIXED (budget cap) — needs rerun
- **Claim:** hybrids are more sample-efficient. **Reality:** best-hybrid > best-single in 9/14 overall;
  powered 3/4 (humanness +0.053, EGFR +0.038, buchwald wins on RMSE) but **VEGF reverses (−0.027)**.
- **Defect:** all 4 abagym curves were **all-NaN at their smallest sample sizes** because the budget cap
  was `b <= n_sub` — a budget can still exceed the *training* partition (e.g. budget 50 at n_sub=50
  leaves ~40 train rows → degenerate selection → NaN).
- **Fix applied** (`c05.py:161`): cap budgets to the training-partition size
  (`n_sub*(folds-1)/folds`), not the whole subsample, with a fits-the-train-set fallback. Compiles;
  finding (modest sample-efficiency edge, VEGF reversal) is otherwise a genuine result.

### c06_within_family_gradient ✅ FIXED (data-relative budget grid + diagnostic) — verified by running
- **Claim:** higher model capacity ↔ larger optimal feature budget (positive Spearman ρ).
- **Was (verified):** `capacity_budget_correlation` NaN for all families on VEGF/Ang2 — a coarse budget
  grid (`[50,500,...]`) collapsed to a single value ≤ n_samples, so every capacity variant pinned to the
  same optimal budget → no variance → NaN.
- **Fix applied** (`c06.py:66`): when fewer than 4 distinct budgets are viable, widen to a data-relative
  grid (fractions of `min(n_samples, feature_dim)`); plus a no-variance diagnostic that labels the
  correlation `measured` / `no_budget_variance` / `insufficient_variants` instead of a bare NaN.
- **Verified by running locally** (flab_thermo_jain): grid widened `[10,25]→[6,10,13,25,27,47,68]`;
  RF variants found distinct optima (depth=3→13, deeper→6); correlation now **computes
  (ρ=−0.775, p=0.225, status `measured`)** instead of NaN. (The negative/ns ρ itself is then a real
  finding: capacity and optimal budget are not positively coupled here.)

### c08_calibration ⚠️ (genuine finding — NOT a code bug; CORRECTION below)
- **Claim:** complexity-matching improves calibration. **Reality (verified cal_error = |PICP−0.9|):**
  **underfit (budget 25) is best-calibrated on all 5 powered datasets** —
  VEGF underfit 0.122 < optimal 0.168; EGFR 0.045 < 0.129; Ang2 0.109 < 0.154; humanness 0.046 < 0.052;
  buchwald 0.053 < 0.109.
- **CORRECTION (earlier misread):** I previously called EGFR's optimal "budget=659 = full dimension."
  That is wrong — EGFR `total_dim=68948`, so optimal=659 ≪ full dim; the optimal/underfit(25)/overfit(68948)
  ladder IS well-separated. So this is **not a code bug**: "more compression calibrates better" is a real
  result that simply *refutes* the experiment's hypothesis (simpler models give steadier conformal
  intervals). 🔍 Added a `conditions_distinct` / `condition_budgets` diagnostic (`c08.py`) so any future
  genuine collapse is flagged, but no behavioral change was warranted.

### c09_formal_analysis ⚠️
- **Claim (docstring):** model ≥50%, budget ≥25%, interaction ≥10% of variance. **Reality (verified
  eta²):** **residual dominates** — VEGF residual 88.5% (model 8.2%, budget 1.4%), Ang2 95.6% (model 3.2%),
  humanness 68.8% (model 12.8%, interaction 15.5%). p_model is ~1e-16 but the effect size is tiny.
- Plus the MI-retention sub-block is dead (see degenerate-metric section).

### c17_information_theoretic_decomposition ⚠️
- **Claim:** hybrid selects higher-MI features than PCA/mRMR/RFE/random. **Reality:** MI metric dead;
  on the downstream score the hybrid beats PCA/mRMR/random but **RFE matches or beats the hybrid** on the
  powered sets (VEGF b25 hybrid 0.574 vs RFE 0.666; humanness 0.966 vs RFE 0.973).

### c18_diversity_vs_depth ⚠️
- **Claim:** free hybrid allocation beats concentrating budget on top reps. **Reality:** "natural" wins at
  budget 50 but **the edge collapses at budget 100** (VEGF b50 natural 0.588 best → b100 natural 0.388 <
  depth_1 0.452). humanness flat across all conditions.

### c22_explainability ⚠️ (descriptive, partial)
- **What it does:** traces hybrid features back to source reps with SHAP. No pass/fail thesis.
- **Limitation:** named human-readable features only populate when the blueprint picks aac/structural
  (VEGF: 30 named; Ang2: 0 named — PLM-dominated blueprint). humanness not run.

### c23_evaluator_model_matching ⚠️
- **Claim:** matching the internal evaluator to the downstream model beats mismatched evaluators.
- **Reality:** effect is real but ~noise — matched wins ≈54% of comparisons, mean(mismatched−matched)
  ≈ −0.007 ρ (per grounded agent extraction; per-comparison list in `results`, no top-level `aggregate`
  field). On powered sets (VEGF, Ang2) the win rate is at/near 50%.

### s05_knn_prediction ⚠️ → ✅ code-fixed (finding unchanged, now honest)
- **Claim:** hybrid distance improves kNN. **Reality:** wins 17/22 overall but powered gains shrink to
  ~0–0.07 (VEGF +0.049, tz_multiclass +0.066, humanness +0.004; **EGFR −0.004, Ang2 −0.007**). Big gains
  are all n<200.
- **Defect:** "best hybrid" is selected post-hoc over budgets × conditions × k (in-sample optimistic).
- ✅ **FIX:** `results['summary']` now emits BOTH `posthoc_max` (the optimistic number, clearly labelled
  not-the-headline) AND a pre-registered `fixed_ref` (budget=100, hybrid_unweighted, k=max — chosen a
  priori, no selection over the grid). Report `fixed_ref` as the honest hybrid-vs-rep delta.
- **Weighting REFUTED as a bug:** the importance-weighted condition genuinely multiplies sqrt(importance)
  into the standardized features before kNN (`s05:161-169`, asserted shape-match). It just doesn't help —
  a real negative result, not a no-op. Docstring corrected.

### s06_applicability_domain ⚠️ → ✅ code-fixed (regression result is a genuine NEGATIVE)
- **Claim:** hybrid distance tracks prediction error. **Reality (verified, k5 hybrid distance-error ρ):**
  near-zero and sign-inconsistent — VEGF −0.006, EGFR +0.006, Ang2 −0.108, tz_multiclass +0.246. Only tz
  shows a real positive. Hybrid gives tighter intervals but lower coverage.
- **Verified NOT a bug for regression:** error is the absolute residual `|y-ŷ|`, per-fold standardization
  is train-only and consistent. The near-zero correlation is a real negative finding for these reps.
- ✅ **FIX 1 (clf):** classification "error" was a degenerate 0/1 misclassification indicator → replaced
  with graded `1 - P(true class)` (`_graded_clf_error`), so Spearman(distance, error) has real power on
  humanness/tz instead of a near-binary statistic.
- ✅ **FIX 2 (conformal honesty):** added `undercoverage` (= 0.9 − PICP) and a `hybrid_genuinely_tighter`
  flag (tighter MPIW counts as better ONLY if PICP doesn't drop) so "tighter intervals" can't be misread
  as improvement when it's actually undercoverage. Docstring corrected to the negative finding.

### s08_optimal_dimension ⚠️ → ✅ code-fixed
- **Claim:** optimal dimensionality differs for similarity vs prediction. **Reality:** most datasets swept
  **only 2 budgets (50, 500)** in the stale JSONs, so "optimal" is barely meaningful; `optima_differ` is
  ~6 True / 10 False and the underlying SAC ρ is ~0.1. No clean conclusion.
- **Root cause (refined):** two causes. (1) The on-disk JSONs predate the current grid; the live code
  already sweeps the full `DISTANCE_BUDGETS=[10,25,50,100,250,500]`. (2) The *real* defect: budgets cap at
  the feature pool, so many distinct budgets collapse to the SAME `actual_dim` → tied SAC → the argmax
  (`optima_differ`) is decided by floating-point noise.
- ✅ **FIX:** budgets deduped/sorted to the full sweep; output now carries `n_distinct_dims` /
  `n_budgets_evaluated` / `insufficient_budgets`; `pareto.tradeoff.optima_differ` is suppressed to `None`
  (with `optima_differ_trustworthy=False`) when fewer than 4 distinct dims were evaluated. Docstring fixed.
- ⚠️ still needs the rerun to populate trustworthy multi-dim sweeps on the powered datasets.

### s10_importance_downstream ⚠️ → ✅ code-fixed (RMSE sign trap)
- **Claim:** importance-weighted distance helps GP/SVR/qRASAR (beyond kNN). **Reality:** weighting is
  neutral-to-negative on powered sets (VEGF GP −0.005, Ang2 GP −0.015). The one big positive
  (flab_polyreact_bvp +0.233) is n=137.
- **Confirmed bug:** every stored `delta*` was raw `treatment − reference`, never oriented for metric
  direction. `lower_is_better` was resolved (`s10:671`) but only used for display ordering, never to
  orient the stored deltas. For buchwald (RMSE, lower-better) the "gp +13.077 / svr +0.718" were RMSE
  *increases* (weighting hurt), misread as gains.
- ✅ **FIX:** `_orient_deltas` negates every `delta*` field for lower-is-better metrics so positive always
  means improvement; `results['deltas_oriented']=True` flags it; `_print_summary` no longer double-
  corrects. Verified by smoke test (`Fix5`). buchwald gp now reads −13.08 (a worsening). Docstring fixed.
- **Cross-cutting:** together with s05, the property-weighting step adds nothing to downstream prediction —
  any benefit is from the compact hybrid space. In tension with the s01/s02 framing where weighting is what
  doubles the similarity-activity correlation. **Reconcile: weighting helps the SAC *metric*, not the
  downstream kNN/GP/SVR prediction.**

### s13_clustering_retrieval ⚠️ → ✅ verified (genuine mixed finding, no bug)
- **Claim:** hybrid distance improves clustering quality and retrieval. **Reality:** retrieval wins 14/16
  (small margins; VEGF/EGFR reverse) but **clustering coherence wins only 5/16** — hybrid loses the
  majority, including most powered sets (EGFR 0.249 vs 0.381, VEGF 0.362 vs 0.393).
- **Verified NOT a bug:** both metrics are computed identically for hybrid and every baseline (same KMeans
  config, same k-list), and the win tests respect direction (coherence higher=better; retrieval mean-Δy
  lower=better). Silhouette is recorded but not used for the win test. This is a real MIXED result: the
  hybrid (a supervised feature allocation) improves local nearest-neighbor retrieval but not global cluster
  separation.
- ✅ **FIX (clarity, no behaviour change):** `summary.metrics_agree` flags the retrieval/clustering
  divergence; `best_rep_retrieval` / `best_rep_coherence` already name the (different) baseline rep each
  axis is compared against. Docstring corrected to describe both metrics + the mixed finding.

---

## Priority order for fixes (highest leverage first)

**DONE in code (✅, pending ARC deploy + rerun):**
1. **MI estimator** — one fix unblocks MI claims in c01, c09, c17, s03 simultaneously. ✅
2. **Cliff calibration** — distribution-relative percentile thresholds fix both s07 (~100%) and s12 (0%). ✅
3. **c10 dependency** — `noiseInject` install wired into setup.sh (private pkg, public repo). ✅
4. **c11 / c13 / c19 docstrings** — were by-design removals, not bugs; docstrings corrected. ✅
5. **c16 / c21 / c24 diagnostics** — self-diagnosing fields added so the rerun reveals the real cause. 🔍
6. **Similarity fixes** — s05 honest fixed_ref, s06 graded clf error + conformal undercoverage,
   s08 dim-collapse self-diag, s10 delta orientation, s13 metrics_agree. All e2e-verified locally. ✅
7. **s04 / s11** — user chose "add molecular datasets"; `full_mol` mode added to the SLURM script. ✅

**REMAINING (need the user / a rerun):**
8. **Deploy + rerun on ARC** — `git pull` + `setup.sh` (for noiseInject), then re-run the whole
   c*/s* suite. NOTHING above changes a result JSON until this happens. Run s04/s11 via the new
   `sbatch tests/slurm_scripts/run_similarity_experiments.sh full_mol`.
9. **Re-run the deep experiments (c09–c24, s06–s13) on the powered datasets**, not the n<150 set.
10. **Tune cliff percentile** (default 2%) after the rerun — confirm cliff prevalence is a sane minority.
11. **c16 / c21 / c24 root causes** — read the new diagnostic fields after the rerun to classify
    legit-vs-bug, then fix if needed.

### Smoke tests
All code fixes are covered by `tests/experiments/smoke_tests_broken_fixes.py` (real-data, local):
`python tests/experiments/smoke_tests_broken_fixes.py` → **6/6 passing** (MI, s07 cliff, s12 cliff, c10,
s06 graded clf error, s10 delta orientation). Plus the 5 similarity experiments were run end-to-end on
real small data (see the verification note near the top of this doc).
