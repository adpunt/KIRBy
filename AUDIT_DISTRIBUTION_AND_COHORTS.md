# ANNITIA distribution & cohort audit

Train n = 1253 (hep-valid 1106, death-valid 966).  Val_test n = 423.
Features compared: 21.  Used in multivariate shift: 20.

## A. Distribution shift summary

- **Domain classifier AUROC** (5-fold OOF logistic regression, standardised features, mean-imputed): **0.534**.
  - >=0.7 indicates a meaningful shift; >=0.8 is severe; ~0.5 means no shift.
- **Mahalanobis (val vs train centroid)**: train median 3.88, val median 3.99, val p90 6.09.
  - Fraction of val patients above train p95 distance: **6.4%** (no shift => ~5%).
- **OT proxy** (mean nearest-neighbour Mahalanobis from val -> train, sub-sampled): mean 3.34, median 3.16, p90 4.93.

## A.1 Most-shifted features (top 15 by KS statistic)

| feature | KS | Wasserstein* | mean_train | mean_val | std_train | std_val |
|---|---|---|---|---|---|---|
| bariatric_surgery_age | 0.164 | 0.245 | 49.54 | 50.58 | 12.41 | 10.32 |
| followup_years | 0.122 | 0.243 | 5.29 | 6.44 | 4.61 | 4.89 |
| base_bilirubin | 0.093 | 0.106 | 10.39 | 11.01 | 6.60 | 6.50 |
| base_plt | 0.073 | 0.130 | 245.23 | 237.28 | 66.13 | 62.40 |
| base_BMI | 0.064 | 0.092 | 29.39 | 28.85 | 6.13 | 6.03 |
| base_fibs_stiffness_med_BM_1 | 0.064 | 0.091 | 7.92 | 7.53 | 5.28 | 5.09 |
| last_followup_age | 0.063 | 0.122 | 59.72 | 61.24 | 12.97 | 11.88 |
| base_aixp_aix_result_BM_3 | 0.063 | 0.080 | 7.23 | 7.04 | 3.60 | 3.44 |
| base_ast | 0.061 | 0.065 | 40.70 | 42.12 | 23.63 | 25.11 |
| base_fibrotest_BM_2 | 0.053 | 0.091 | 0.31 | 0.31 | 0.22 | 0.20 |
| base_ggt | 0.047 | 0.065 | 102.46 | 105.65 | 122.08 | 119.30 |
| base_chol | 0.047 | 0.072 | 4.99 | 5.04 | 1.16 | 1.18 |
| base_triglyc | 0.046 | 0.100 | 1.61 | 1.53 | 1.07 | 0.90 |
| base_alt | 0.043 | 0.039 | 57.39 | 58.80 | 47.43 | 49.23 |
| baseline_age | 0.040 | 0.076 | 54.43 | 54.80 | 12.68 | 11.63 |

*Wasserstein computed on standardised values (pooled SD), so values are comparable across features.

## A.2 Top discriminating features (logistic-regression weights)

Sign convention: positive = more likely val_test. Coefficients on standardised features.

| feature | LR coef |
|---|---|
| followup_years | +0.193 |
| base_plt | -0.106 |
| base_bilirubin | +0.103 |
| base_fibrotest_BM_2 | -0.098 |
| base_ast | +0.090 |
| last_followup_age | +0.087 |
| base_fibs_stiffness_med_BM_1 | -0.083 |
| base_triglyc | -0.082 |
| T2DM | -0.080 |
| Dyslipidaemia | +0.063 |
| gender | -0.057 |
| base_gluc_fast | +0.047 |
| Hypertension | -0.039 |
| base_BMI | -0.026 |
| bariatric_surgery | +0.025 |

## B. Per-cohort C-index (top 15 variants by weighted train OOF)

Top 15 variants ranked by 0.7*C_hep + 0.3*C_death on training OOF:

  - `tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth`
  - `tsfm-gbsa__tsfm`
  - `tsfm-rsf__tsfm`
  - `dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope`
  - `dsm-K4__snapshot+clinical_pro+meld_na+visit_density`
  - `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth`
  - `pretrained-mlp__nhanes_proj`
  - `tsfm-cox__tsfm`
  - `tsfm-coxnet__tsfm`
  - `tsfm-ngboost__tsfm`
  - `tsfm-lgbm_aft__tsfm`

### Cohort sizes (training set)

| cohort | n |
|---|---|
| diabetes_yes | 416 |
| diabetes_no | 701 |
| sex_1 | 655 |
| sex_2 | 598 |
| age_lt50 | 388 |
| age_50_65 | 581 |
| age_ge65 | 284 |
| fib4_low | 469 |
| fib4_intermediate | 298 |
| fib4_high | 55 |
| followup_short | 551 |
| followup_long | 702 |
| ALL | 1253 |

### Weakest cohort drops (variant x cohort), top 15 by LB delta vs ALL

| variant | cohort | n_hep | events_hep | C_hep | C_death | weighted_LB | delta_vs_all |
|---|---|---|---|---|---|---|---|
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | age_lt50 | 347 | 9 | 0.458 | 0.676 | 0.524 | -0.244 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | fib4_low | 465 | 6 | 0.416 | 0.823 | 0.538 | -0.224 |
| tsfm-lgbm_aft__tsfm | followup_short | 409 | 8 | 0.372 | 0.488 | 0.407 | -0.197 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | followup_short | 409 | 8 | 0.552 | 0.617 | 0.572 | -0.191 |
| tsfm-ngboost__tsfm | fib4_high | 53 | 7 | 0.444 | 0.677 | 0.514 | -0.189 |
| pretrained-mlp__nhanes_proj | fib4_low | 465 | 6 | 0.472 | 0.686 | 0.536 | -0.184 |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | followup_short | 409 | 8 | 0.667 | 0.715 | 0.681 | -0.165 |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | fib4_low | 465 | 6 | 0.423 | 0.938 | 0.577 | -0.152 |
| pretrained-mlp__nhanes_proj | fib4_high | 53 | 7 | 0.503 | 0.740 | 0.574 | -0.147 |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | followup_short | 409 | 8 | 0.566 | 0.625 | 0.583 | -0.146 |
| pretrained-mlp__nhanes_proj | age_ge65 | 230 | 8 | 0.605 | 0.602 | 0.604 | -0.116 |
| pretrained-mlp__nhanes_proj | followup_short | 409 | 8 | 0.577 | 0.692 | 0.612 | -0.109 |
| tsfm-coxnet__tsfm | sex_1 | 578 | 20 | 0.532 | 0.757 | 0.600 | -0.108 |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | fib4_low | 465 | 6 | 0.570 | 0.879 | 0.663 | -0.105 |
| tsfm-cox__tsfm | sex_1 | 578 | 20 | 0.544 | 0.750 | 0.606 | -0.104 |

### Cohort-specific failures (LB drop > 0.05 vs ALL)

- `dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope` drops **0.244** on `age_lt50` (n_hep=347, ev_hep=9).
- `dsm-K4__snapshot+clinical_pro+meld_na+visit_density` drops **0.224** on `fib4_low` (n_hep=465, ev_hep=6).
- `tsfm-lgbm_aft__tsfm` drops **0.197** on `followup_short` (n_hep=409, ev_hep=8).
- `dsm-K4__snapshot+clinical_pro+meld_na+visit_density` drops **0.191** on `followup_short` (n_hep=409, ev_hep=8).
- `tsfm-ngboost__tsfm` drops **0.189** on `fib4_high` (n_hep=53, ev_hep=7).
- `pretrained-mlp__nhanes_proj` drops **0.184** on `fib4_low` (n_hep=465, ev_hep=6).
- `tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth` drops **0.165** on `followup_short` (n_hep=409, ev_hep=8).
- `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth` drops **0.152** on `fib4_low` (n_hep=465, ev_hep=6).
- `pretrained-mlp__nhanes_proj` drops **0.147** on `fib4_high` (n_hep=53, ev_hep=7).
- `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth` drops **0.146** on `followup_short` (n_hep=409, ev_hep=8).
- `pretrained-mlp__nhanes_proj` drops **0.116** on `age_ge65` (n_hep=230, ev_hep=8).
- `pretrained-mlp__nhanes_proj` drops **0.109** on `followup_short` (n_hep=409, ev_hep=8).
- `tsfm-coxnet__tsfm` drops **0.108** on `sex_1` (n_hep=578, ev_hep=20).
- `dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope` drops **0.105** on `fib4_low` (n_hep=465, ev_hep=6).
- `tsfm-cox__tsfm` drops **0.104** on `sex_1` (n_hep=578, ev_hep=20).
- `dsm-K4__snapshot+clinical_pro+meld_na+visit_density` drops **0.103** on `age_ge65` (n_hep=230, ev_hep=8).
- `tsfm-gbsa__tsfm` drops **0.098** on `fib4_high` (n_hep=53, ev_hep=7).
- `tsfm-coxnet__tsfm` drops **0.097** on `fib4_low` (n_hep=465, ev_hep=6).
- `tsfm-ngboost__tsfm` drops **0.094** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `tsfm-lgbm_aft__tsfm` drops **0.090** on `fib4_high` (n_hep=53, ev_hep=7).
- `tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth` drops **0.087** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `tsfm-coxnet__tsfm` drops **0.086** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth` drops **0.079** on `diabetes_yes` (n_hep=401, ev_hep=24).
- `tsfm-cox__tsfm` drops **0.078** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth` drops **0.076** on `age_ge65` (n_hep=230, ev_hep=8).
- `tsfm-ngboost__tsfm` drops **0.076** on `age_50_65` (n_hep=529, ev_hep=22).
- `tsfm-lgbm_aft__tsfm` drops **0.067** on `age_50_65` (n_hep=529, ev_hep=22).
- `dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope` drops **0.065** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `pretrained-mlp__nhanes_proj` drops **0.064** on `diabetes_no` (n_hep=676, ev_hep=15).
- `pretrained-mlp__nhanes_proj` drops **0.061** on `fib4_intermediate` (n_hep=290, ev_hep=12).
- `tsfm-cox__tsfm` drops **0.060** on `fib4_low` (n_hep=465, ev_hep=6).
- `tsfm-coxnet__tsfm` drops **0.058** on `fib4_high` (n_hep=53, ev_hep=7).
- `tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth` drops **0.055** on `sex_1` (n_hep=578, ev_hep=20).
- `tsfm-gbsa__tsfm` drops **0.054** on `fib4_low` (n_hep=465, ev_hep=6).
- `dsm-K4__snapshot+clinical_pro+fpca+gp_smooth` drops **0.052** on `age_lt50` (n_hep=347, ev_hep=9).
- `pretrained-mlp__nhanes_proj` drops **0.050** on `age_50_65` (n_hep=529, ev_hep=22).

## C. Per-time-horizon C-index

| variant | horizon | C_hep | C_death | weighted_LB |
|---|---|---|---|---|
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | 1y | 0.477 | nan | nan |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | 3y | 0.659 | 0.771 | 0.693 |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | 5y | 0.648 | 0.852 | 0.709 |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | 10y | 0.669 | 0.874 | 0.730 |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 1y | 0.558 | nan | nan |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 3y | 0.739 | 0.904 | 0.788 |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 5y | 0.696 | 0.911 | 0.761 |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 10y | 0.717 | 0.906 | 0.774 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 1y | 0.582 | nan | nan |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 3y | 0.681 | 0.884 | 0.742 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 5y | 0.699 | 0.873 | 0.751 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 10y | 0.717 | 0.891 | 0.769 |
| pretrained-mlp__nhanes_proj | 1y | 0.551 | nan | nan |
| pretrained-mlp__nhanes_proj | 3y | 0.674 | 0.803 | 0.713 |
| pretrained-mlp__nhanes_proj | 5y | 0.678 | 0.794 | 0.713 |
| pretrained-mlp__nhanes_proj | 10y | 0.697 | 0.764 | 0.717 |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 1y | 0.571 | nan | nan |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 3y | 0.748 | 0.930 | 0.803 |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 5y | 0.763 | 0.953 | 0.820 |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 10y | 0.807 | 0.955 | 0.851 |
| tsfm-cox__tsfm | 1y | 0.453 | nan | nan |
| tsfm-cox__tsfm | 3y | 0.664 | 0.870 | 0.726 |
| tsfm-cox__tsfm | 5y | 0.643 | 0.799 | 0.690 |
| tsfm-cox__tsfm | 10y | 0.690 | 0.772 | 0.714 |
| tsfm-coxnet__tsfm | 1y | 0.454 | nan | nan |
| tsfm-coxnet__tsfm | 3y | 0.644 | 0.909 | 0.724 |
| tsfm-coxnet__tsfm | 5y | 0.628 | 0.810 | 0.683 |
| tsfm-coxnet__tsfm | 10y | 0.681 | 0.779 | 0.711 |
| tsfm-gbsa__tsfm | 1y | 0.854 | nan | nan |
| tsfm-gbsa__tsfm | 3y | 0.835 | 0.970 | 0.875 |
| tsfm-gbsa__tsfm | 5y | 0.855 | 0.886 | 0.864 |
| tsfm-gbsa__tsfm | 10y | 0.839 | 0.852 | 0.843 |
| tsfm-lgbm_aft__tsfm | 1y | 0.280 | nan | nan |
| tsfm-lgbm_aft__tsfm | 3y | 0.455 | 0.763 | 0.547 |
| tsfm-lgbm_aft__tsfm | 5y | 0.467 | 0.702 | 0.538 |
| tsfm-lgbm_aft__tsfm | 10y | 0.530 | 0.745 | 0.595 |
| tsfm-ngboost__tsfm | 1y | 0.326 | nan | nan |
| tsfm-ngboost__tsfm | 3y | 0.555 | 0.980 | 0.683 |
| tsfm-ngboost__tsfm | 5y | 0.598 | 0.818 | 0.664 |
| tsfm-ngboost__tsfm | 10y | 0.639 | 0.838 | 0.699 |
| tsfm-rsf__tsfm | 1y | 0.644 | nan | nan |
| tsfm-rsf__tsfm | 3y | 0.792 | 0.958 | 0.842 |
| tsfm-rsf__tsfm | 5y | 0.816 | 0.826 | 0.819 |
| tsfm-rsf__tsfm | 10y | 0.839 | 0.821 | 0.833 |

### Horizon stability (max - min weighted_LB per variant)

| variant | LB range across horizons |
|---|---|
| pretrained-mlp__nhanes_proj | 0.004 |
| tsfm-rsf__tsfm | 0.023 |
| dsm-K4__snapshot+clinical_pro+meld_na+visit_density | 0.027 |
| dsm-K4__snapshot+clinical_pro+fpca+visit_density+clinical_slope | 0.028 |
| tsfm-gbsa__tsfm | 0.032 |
| tsfm-ngboost__tsfm | 0.035 |
| tsfm-cox__tsfm | 0.036 |
| dsm-K4__snapshot+clinical_pro+fpca+gp_smooth | 0.038 |
| tsfm-coxnet__tsfm | 0.041 |
| tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth | 0.048 |
| tsfm-lgbm_aft__tsfm | 0.057 |

**Most horizon-stable**: pretrained-mlp__nhanes_proj, tsfm-rsf__tsfm, dsm-K4__snapshot+clinical_pro+meld_na+visit_density.
**Most horizon-sensitive**: tsfm-coxnet__tsfm, tabpfn-landmark__snapshot+clinical_pro+fpca+gp_smooth, tsfm-lgbm_aft__tsfm.

## Notes on cohort proxies

- **Diabetes**: T2DM column OR fasting glucose >= 7 mmol/L. (No HbA1c column present in this dataset.)
- **Sex**: `gender` column (binary).
- **FIB-4**: age * AST / (PLT * sqrt(ALT)) using baseline-visit values; cutoffs 1.3 / 2.67.
- **Follow-up**: median split on (last_followup_age - baseline_age).
