# AUDIT: Representation discovery for contested val_test patients

**Date:** 2026-05-08 (deadline day)
**Question:** Is there a representation that differentiates contested val_test
patients from non-contested ones, beyond the IsolationForest / OOD signals
already in `patient_representation_val.csv`?
**Anchor:** `sub_master_meta_avg_aware_weighted.csv` (LB 0.9302)
**Scoring:** `LB = 0.7 * C_hep + 0.3 * C_death`

## 1. Setup

- **Contested label** (n=22): top-21 by `iqr_rank` in
  `valtest_contention_hepatic.csv` PLUS top-1 by `iqr_rank` in
  `valtest_contention_death.csv`.
- **Snapshot features** (d=14 unique vars expanded to ~70 cols): static demographics
  + first-visit / max / min / slope / n_obs for 13 longitudinal labs +
  baseline_age, last_age, follow-up years, n_visits.
- **Imputation:** train-median, then z-score, clip to ±6.
- **Classifier:** balanced logistic regression, 5-fold stratified CV on val_test.

## 2. AUROCs per representation (5-fold CV, "contested" vs "non-contested")

| Representation        | dim | AUROC  |
|-----------------------|----:|-------:|
| **mahalanobis**       |  1  | **0.7106** |
| kde_logdens           |  1  | 0.6981 |
| rank_disagreement     |  4  | 0.6676 |
| pca5                  |  5  | 0.6553 |
| raw_clinical          | 14  | 0.5979 |
| combined              | 32  | 0.5500 |
| gmm6_proba            |  6  | 0.5161 |
| lof                   |  1  | 0.3735 |

**Winner: Mahalanobis distance to training centroid** (single scalar, AUROC 0.71).
KDE log-density is a close second (0.70). Both express the same underlying
phenomenon: contested patients sit in low-density regions of the training
distribution. PCA5 is worse, raw clinical features alone are nearly random,
combined is *worse* than mahalanobis alone (over-parameterised on n=423 with 22
positives → CV noise dominates).

## 3. Why mahalanobis discriminates

A high mahalanobis distance means the patient's clinical snapshot is far from
the *covariance-weighted* training centroid — the model sees a profile it has
not seen many examples of. Contested predictions are exactly what we expect
in that regime: the 20+ branch models extrapolate in different directions
(different priors, different latent dimensions favoured), so their per-patient
ranks fan out → high `iqr_rank` in `valtest_contention_*.csv`.

LOF underperforms (AUROC 0.37, *anti-correlated*) because it flags local
outliers — but contested patients are not local outliers; they are
*globally* unusual. LOF's neighbourhood-relative score actually rewards
patients in dense regions of the *val* manifold even when those regions are
under-represented in *train*. KDE log-density picks up the same signal as
mahalanobis but with a kernel rather than a covariance, so it's a valid
backup.

## 4. Correlation vs existing anomaly/disagreement signals (Spearman)

| Pair                                | ρ      |
|-------------------------------------|-------:|
| clike vs `iso_score`                | +0.256 |
| clike vs `ood_composite`            | +0.259 |
| clike vs existing `mahalanobis`     | +0.257 |
| clike vs `iqr_rank` (hep)           | +0.076 |
| clike vs `iqr_rank` (death)         | -0.120 |

The discovered "contested-likeness" (`clike`) is **only weakly correlated**
(+0.26) with the three signals already in `patient_representation_val.csv`.
That is non-trivially redundant — these signals share some manifold info but
clike captures something different. The existing `mahalanobis` column was
computed using *only the 4 OOD columns themselves* (cluster + 3 anomaly
scores) — a self-referential composite, not a snapshot-based outlierness
measure. Recomputing mahalanobis on full snapshot features (≈70 dims) is the
new contribution.

The very low correlation with `iqr_rank` (+0.08 / -0.12) is the most
important finding: clike does NOT directly recover the contention ranking,
but it *predicts* membership in the binary "contested" set with AUROC 0.71.
That is, clike is a **prior** on contestedness, while iqr_rank is the
**posterior** observed via branch disagreement.

## 5. Submission candidates

All three use clike (mahalanobis-based, normalised to [0,1]) as the gating
signal. Anchor is `sub_master_meta_avg_aware_weighted` (LB 0.9302).

### sub_master_repdisc_blend.csv
Smooth blend of anchor with median-of-base-models, weight `(1 - clike)` on
anchor. Patients near training distribution stay locked to anchor; far
patients drift toward the ensemble median (which has implicit Wisdom-of-Crowds
robustness for OOD points).

### sub_master_repdisc_aware.csv
Sigmoid gate at clike=0.5 (k=8). Below: anchor. Above: 20%-trimmed mean of
base-model ranks, rank-matched to anchor distribution. More aggressive than
blend.

### sub_master_repdisc_specialized.csv
Hard route: top-20% clike → median of base models (rank-matched), bottom-80% →
anchor. Aggressive but only on 84 patients.

## 6. Per-candidate stats (vs anchor)

| Candidate                          | ρ_hep  | ρ_death | n_changed | mean Δhep | mean Δdeath |
|-----------------------------------:|-------:|--------:|----------:|----------:|------------:|
| sub_master_repdisc_aware           | 0.979  | 0.986   | 414/423   | 0.026     | 0.024       |
| sub_master_repdisc_specialized     | 0.921  | 0.936   | 84/423    | 0.035     | 0.035       |
| sub_master_repdisc_blend           | 0.993  | 0.997   | 422/423   | 0.019     | 0.011       |

## 7. Training-OOF C-index (sanity check, mean-ensemble anchor proxy)

Applied each recipe to *training* OOF predictions. Anchor proxy =
unweighted mean ensemble of 22 hep + 16 death OOFs. Clike recomputed on
training data via leave-self-in mahalanobis (in-sample, noisy but indicative).

| Recipe              | C_hep  | C_death | LB proxy (0.7*hep+0.3*death) |
|--------------------:|-------:|--------:|-----------------------------:|
| anchor (mean-ens)   | 0.7856 | 0.8600  | 0.8079                       |
| **repdisc_blend**   | **0.7961** | 0.8563 | **0.8141 (+0.006)**       |
| repdisc_aware       | 0.7865 | **0.8709** | 0.8118 (+0.004)           |
| repdisc_specialized | 0.6038 | 0.7900  | 0.6597 (-0.148, **bad**)     |

The blend recipe gains on hepatic without losing on death; the aware recipe
gains on death significantly (+0.011) at flat hep. **Specialized is harmful
on training OOF** — replacing the anchor with the median for high-clike
patients destroys hep concordance. The "specialized" hypothesis (hard route to
median) does not hold; the soft blends do.

## 8. Pre-submit verdicts

| File                                  | verdict |
|---------------------------------------|--------:|
| sub_master_repdisc_aware.csv          | **PASS** |
| sub_master_repdisc_specialized.csv    | **PASS** |
| sub_master_repdisc_blend.csv          | **PASS** |

All three pass `pre_submit_check.py` (correct columns, no NaN, 423 rows,
trustii_id aligned, ρ vs anchor in 0.92–0.997 range — not identical, not
anti-correlated, not weakly correlated).

## 9. Recommendation

**Submit `sub_master_repdisc_blend.csv` first** — best training OOF LB proxy
(+0.006 over anchor proxy), most conservative drift from anchor (mean Δhep
0.019), highest ρ to anchor (0.993/0.997). If a second slot is available,
submit `sub_master_repdisc_aware.csv` as the higher-variance bet (better
death gain, +0.011, but more drift). **Do NOT submit `repdisc_specialized`** —
training OOF shows -0.148 LB drop; the aggressive top-20% reroute is harmful.

## 10. Files

- Inputs:
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/valtest_contention_hepatic.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/valtest_contention_death.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/patient_representation_val.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/branches/val_*.npy`
  - `/Volumes/seagate/kirby/annitia/{train.csv,val_test.csv}`
- Outputs:
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_repdisc_aware.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_repdisc_specialized.csv`
  - `/Users/apunt/repos/KIRBy/tests/results/annitia/full/sub_master_repdisc_blend.csv`
- Scripts:
  - `/tmp/annitia_runs/repdisc.py` (representation discovery + submissions)
  - `/tmp/annitia_runs/repdisc_oofc.py` (training OOF C-index)
- Results JSON: `/tmp/annitia_runs/repdisc_results.json`
