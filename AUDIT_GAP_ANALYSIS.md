# ANNITIA LB Gap Analysis (0.93 → 0.97 target)

Date: 2026-05-08. Author: audit pass on tests/experiments/annitia.

## TL;DR

- **Our LB 0.9302** decomposes (given Cdeath ≈ 0.967) to **C_hep ≈ 0.914 on the hidden test set**.
  Our CV reports C_hep = 0.864 (best single, RSF) → 0.872 (best fusion). So generalisation
  is actually slightly *better* than CV, not worse.
- To reach LB 0.97 we need both C_hep and C_death ≳ 0.96. Closing C_hep
  from 0.91 → 0.96 is the ~0.035 LB delta we are looking for.
- I found one **near-certain** lift, two **probable** lifts, and a handful of cheap
  experiments that the leading 0.97+ teams almost certainly ran but we have not.

The single biggest finding: a hand-rolled univariate score
`z(fibroscan_max) + z(fibrotest_max)` already achieves **C_hep = 0.849** on
the training cohort with NO model. Our best learned model (RSF, 226 features)
is at 0.864. We are barely above the trivial NIT max-baseline because the
view-set we feed the learner does not contain any per-NIT max/min/std features
(we use only `__last`).

---

## 1. Features we are missing

### 1a. NIT max/min/std/range (CONFIRMED missing from training pipeline)

**File:** `tests/experiments/annitia/experiment_be.py:66-73`. The `VIEW_SETS`
list never includes the `summary` or `nit` views — every entry uses
`snapshot` (`__last` value only). `experiment_hep_boost.py:96-117` then
inherits the top-K view-sets from BE, so the `summary`/`nit` views never
propagate downstream.

Univariate concordance on hep (full cohort, 47 events, 1253 patients):

| Feature                   | C_hep  |
|---------------------------|--------|
| `fibroscan_last`          | 0.764  |
| `fibroscan_MAX`           | **0.815**  |
| `fibrotest_last`          | 0.760  |
| `fibrotest_MAX`           | 0.759  |
| z(fibs_max)+z(ftest_max)  | **0.849**  |
| `fib4_last`               | 0.541  |
| `Age_v1` (baseline age)   | 0.653  |

`fibroscan_MAX` alone outperforms `fibroscan_last` by +0.05. For a patient
who later regressed (post-treatment), the last value is by definition lower
than the worst observed stiffness — but the worst measurement is the actual
fibrosis signal. This is exactly the "use max-ever instead of last" hypothesis
in the prompt, and the `summary` builder already computes it (views.py:139).

### 1b. AGILE 3+ / AGILE 4 are using fake hand-coded coefficients

**File:** `tests/experiments/annitia/views.py:346-368`. The `_agile_3plus`
function admits in the docstring "Because the coefficients for AGILE 3+ are
not always published openly, we use a published *AGILE 3+ proxy* form" — and
then writes a **made-up** linear predictor `-7.18 + 0.013*age + 0.0089*AST + 0.0188*LSM`.
This is not the published AGILE 3+ logistic; it is a guess. Same problem for
`_agile_4`. The actual coefficients are public (Sanyal 2023 Hep Comm,
supplementary Table 2). Replacing with the real coefficients would make
the score discriminative.

### 1c. NIT trajectory features (slope, delta, acceleration) restricted to last-value

The `_nit_view` builds slope/delta/count/missing for the 3 NITs. It is **not in any
VIEW_SETS**. NITs measure progression — slope of fibroscan vs age is exactly
what hepatologists care about and is missing from training.

### 1d. Probable but unverified gaps

- No HOMA-IR, NFS_with_albumin, FAST, MAST, MEFIB — but these need labs
  (insulin, albumin, MRI-PDFF) that ANNITIA does not have. `_clinical_pro_view`
  correctly stubs these.
- `bariatric_surgery_age` is included as a raw number; could be re-encoded
  as `years_since_bariatric` relative to baseline.
- No interaction terms (e.g., `T2DM × fibroscan_max`, `BMI_slope × age`) anywhere.

---

## 2. Models we are missing

**File:** `tests/experiments/annitia/survival.py:649-670`. Registered models:
`cox`, `coxnet`, `rsf`, `gbsa`, `catboost_cox`, `catboost_aft`, `lgbm_aft`,
`ngboost`, optional `dsm` and `pretrained_mlp`. Branches dir adds
`tabpfn-landmark`, `tabicl-landmark`, `synaug-*`.

**Missing entirely:**

1. **XGBoost-survival (`survival:cox`, `survival:aft`).** zero references
   anywhere under `tests/experiments/annitia`. XGB-AFT has a known edge
   over LightGBM-AFT on small heavy-censored data; CatBoost-AFT and
   LGBM-AFT tend to disagree by ~0.01-0.02 C, so adding XGB-AFT is the
   third independent boosted-AFT model.
2. **DeepSurv / DeepHit retrained per-fold** (DeepHit script `deephit.py` exists
   but our LB history shows it was never the winning submission; could
   indicate it failed or wasn't included in fusion).
3. **Cox–LASSO–boosted (Coxboost / mboost)**. Coxnet on 226 features is the
   only sparse-Cox we run; mboost has tighter shrinkage.
4. **TabPFN v2.5 in landmark mode for hep** (we do it for death — branches/cv_summary.json
   shows `tabpfn-landmark` improved death CV from 0.96 → 0.93/0.93. For hep it
   is in `branches` but with mediocre C=0.82 — possibly a bug, worth re-running with
   max-NIT in features).

---

## 3. Pipeline bugs / silent caps

### 3a. StratifiedKFold variance cap on hep CV

**File:** `src/kirby/datasets/annitia_maslg.py:212-223`. 5-fold StratifiedKFold
on 47 events = ~9 events per fold. Best hep config fold-by-fold:

```
folds = [0.929, 0.743, 0.952, 0.925, 0.769]
```

The 0.74 / 0.77 folds are noise from 9-event resampling. Mean is 0.864 but
the true generalisation is closer to 0.91 (matches LB-decomposed Chep).
**Use RepeatedStratifiedKFold(n_splits=5, n_repeats=10)** to stabilise the
selection signal — currently we may be picking the wrong top-3 due to seed noise.

### 3b. Rank-only fusion may discard magnitude information

**File:** `tests/experiments/annitia/fusion.py:37-63`. All fusion modes
(equal/cv_weighted/single/rrf/stack_meta) operate on ordinal ranks. C-index
is rank-invariant *within a single ranking*, but combining branches via
rank-mean throws away calibration. A patient at rank 98/100 in one branch
and rank 95/100 in another → ranks (98, 95) are nearly identical, but the
underlying risks may have differed by 3× hazard. **Try logit-pooling
(probabilistic stacking)** or sum-of-z-scores instead of rank-pooling.
Probably +0.005–0.01.

### 3c. Hand-picked VIEW_SETS, no exhaustive search

**File:** `tests/experiments/annitia/experiment_be.py:66`. The 6 view-sets
are hand-chosen and all anchored on `snapshot`. Even at 14 base views,
exhaustive 2-, 3-, 4-subsets is 91+364+1001 = 1456 combos × 8 models × 5 folds
≈ 60K fits. Not feasible exhaustively, but **a forward-selection search
starting from the 14 univariate-best views (which would put `summary` and
`nit` in the seed) is feasible in 1-2 hours on Mac CPU**.

### 3d. fpca / gp_smooth basis fit on FULL cohort (mild leakage)

**File:** `tests/experiments/annitia/views.py:711-722`. The fpca docstring
explicitly says "the basis is fit on the FULL cohort once (not per fold).
This is a deliberate, mild leakage". It is mild on the *training* CV but
pure on the held-out test (val_test rows are not in the basis fit). So this
inflates train CV slightly but not test. NOT a cap on LB — a cap on our
ability to *select* models from CV.

### 3e. `_clinical_view` divides by `np.maximum(alt_a, 1e-9)` in FIB-4 sqrt

**File:** `tests/experiments/annitia/views.py:272`. `np.sqrt(np.maximum(alt_a, 1e-9))`
is correct *if* ALT is non-negative (which it is). Not a bug.

---

## 4. Top-3 highest-EV interventions

### #1 — Re-run BE with `summary` + `nit` views in VIEW_SETS  (HIGHEST EV)

- **Change:** Add at least three view-sets containing `summary` and `nit`:
  ```python
  VIEW_SETS = [
      ('snapshot',),
      ('snapshot', 'clinical_pro'),
      ('snapshot', 'clinical_pro', 'summary', 'nit'),     # NEW
      ('summary', 'nit', 'clinical_pro'),                 # NEW (no snapshot)
      ('snapshot', 'summary', 'nit', 'clinical_pro', 'fpca'),  # NEW
      ('snapshot', 'clinical_pro', 'fpca', 'gp_smooth'),
      ('snapshot', 'clinical_pro', 'meld_na', 'visit_density'),
      ('snapshot', 'clinical_pro', 'fpca', 'visit_density', 'clinical_slope'),
  ]
  ```
- **Where:** `tests/experiments/annitia/experiment_be.py:66-73`.
- **Time:** ~30 min CPU on Mac (3 new view-sets × 8 models × 5 folds × 2 endpoints).
- **Probability of success (lifts hep CV by ≥ 0.01):** **80%**. Univariate proves
  the signal exists; the question is whether RSF can extract it.
- **Probability of LB ≥ 0.95:** ~30%. Still need death to hold and fusion to behave.

### #2 — Replace `_agile_3plus` / `_agile_4` with published Sanyal 2023 coefficients

- **Change:** Look up Sanyal 2023 Hep Comm supplementary Table 2 (or
  Newsome 2020 for AGILE 3+). Plug real coefficients into `_agile_3plus` and
  `_agile_4`. Same call signature, just correct numbers.
- **Where:** `tests/experiments/annitia/views.py:346-381`.
- **Time:** 15 min coding + 20 min CV re-run = 35 min.
- **Probability of lifting hep CV by ≥ 0.005:** **60%**. The current proxy is
  monotone-similar to the real score, so the lift is bounded — but boosted
  trees pick up the calibrated logit better than the made-up one.

### #3 — Add XGBoost-AFT survival model + RepeatedStratifiedKFold

- **Change:** (a) Register `SurvivalXGBoostAFT` in `survival.py:649-670` using
  XGBoost's `survival:aft` objective with `aft_loss_distribution='normal'`,
  `aft_loss_distribution_scale=1.5` as starting hyperparams.
  (b) Switch `get_cv_folds` (`annitia_maslg.py:212`) to RepeatedStratifiedKFold
  (n_splits=5, n_repeats=5) so model selection is on 25 fold scores not 5.
  Re-run experiment_be with both changes.
- **Where:** `tests/experiments/annitia/survival.py` + `src/kirby/datasets/annitia_maslg.py:212`.
- **Time:** 45 min coding (XGB-AFT wrapper) + 1.5 h CV re-run = ~2.5 h.
- **Probability of lifting hep CV by ≥ 0.005:** **50%**. XGB-AFT is a proven
  third boosted-AFT and is independent enough from LGBM/CatBoost to add
  diversity in fusion. Repeated CV mostly fixes selection noise (~0.005
  better top-1, not the top-N mean).

### Bonus #4 — Stack-level: replace rank-fusion with logit-pooled stacking

- **Change:** In `fusion.py`, add a `logit_stack` mode that converts each
  branch's risk to z-scores, fits a Coxnet meta-learner on the OOF z-scores
  (not ranks), applies it to test z-scores. Compare against `stack_meta`.
- **Where:** `tests/experiments/annitia/fusion.py` + the fusion-mode dispatch in
  experiment_be.py:246-271.
- **Time:** 30 min coding + 30 min eval.
- **Probability of lift ≥ 0.003:** **40%**.

---

## 5. What the 0.97 teams probably do that we don't

Pure speculation, ranked by plausibility:

1. **They use max/IQR/range of NITs across all visits**, not just last. (We can fix this — see #1.)
2. **They engineer interaction features** (T2DM × fibroscan_max, etc.).
3. **They run XGB-AFT or PySurvival's CoxTime / DeepHit.** PySurvival is on PyPI;
   our `pycox_dt.py` exists but never made it into the winning submission per
   `leaderboard_history.json`.
4. **They stack 5–10 diverse models with logit pooling**, not 3 with rank pooling.
5. **They probe the LB** by submitting carefully-permuted versions to triangulate
   the test-set hep/death distribution. We have `lb_reverse_engineer.py` but
   only 14 historical submissions to work from — top teams have 200+.

The bigger issue: **we have a fixed VIEW_SETS list that excludes the views
which contain the strongest univariate signal in the entire cohort.** Fix that
first.
