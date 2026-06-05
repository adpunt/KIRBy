# Deep Failure Audit — ANNITIA hepatic + death

## Summary

- **Hepatic**: 1106 valid training patients, 47 events. FN-like=5, FP-like=49, trust=1199.
- **Death**:   966 valid training patients, 76 events. FN-like=9, FP-like=21, trust=1223.
- Discriminator (hepatic): FN AUROC=0.715, FP AUROC=0.899
- Discriminator (death):   FN AUROC=0.754, FP AUROC=0.907
- Val buckets (hepatic): trust=399, FN=5, FP=19
- Val buckets (death):   trust=409, FN=5, FP=9

## Hepatic FN-like patients (events the ensemble missed)

| pid | time | rank | base_age | last_age | span | BMI₁ | ALT₁ | AST₁ | PLT₁ | GGT₁ | FibroScan₁ | FibroTest₁ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QPGB7R3OCWBF | 1.0 | 0.23 | 52.0 | 71.0 | 19.0 | 28.1 | 24.3 | 31.1 | 187.1 | 22.5 | 8.8 | 0.9 |
| CTSUFLSEHUDS | 17.0 | 0.36 | 42.0 | 58.0 | 16.0 | 25.4 | 41.9 | 57.3 | 215.7 | 26.1 | 3.2 | 0.1 |
| E13N3BZT9D31 | 16.0 | 0.26 | 55.0 | 72.0 | 17.0 | 26.3 | 79.4 | 86.0 | 295.4 | 85.6 | 6.7 | 0.3 |
| MXLVZ4QKE2GJ | 5.0 | 0.36 | 33.0 | 41.0 | 8.0 | 28.4 | 41.0 | 42.5 | — | 126.6 | 44.3 | 0.8 |
| 97PWDBOLWIMR | 1.0 | 0.32 | 59.0 | 66.0 | 7.0 | 47.0 | 54.0 | 50.1 | — | 387.7 | 10.3 | 0.4 |

## Hepatic feature means (population vs FN vs FP)

| Feature | Population | FN-like | FP-like |
|---|---|---|---|
| baseline_age | 54.15 | 48.20 | 59.47 |
| last_followup_age | 60.10 | 61.60 | 67.76 |
| followup_span | 5.95 | 13.40 | 8.29 |
| BMI_first | 29.43 | 31.04 | 31.51 |
| alt_first | 57.12 | 48.14 | 63.62 |
| ast_first | 40.63 | 53.38 | 54.00 |
| plt_first | 245.93 | 232.70 | 215.33 |
| ggt_first | 101.76 | 129.71 | 143.22 |
| fibs_stiffness_med_BM_1_first | 7.81 | 14.65 | 12.53 |
| fibrotest_BM_2_first | 0.31 | 0.50 | 0.54 |
| gluc_fast_first | 5.88 | 4.79 | 6.65 |

## Death feature means (population vs FN vs FP)

| Feature | Population | FN-like | FP-like |
|---|---|---|---|
| baseline_age | 53.74 | 53.78 | 68.19 |
| last_followup_age | 60.18 | 66.67 | 74.19 |
| followup_span | 6.44 | 12.89 | 6.00 |
| BMI_first | 29.26 | 27.70 | 31.26 |
| alt_first | 58.18 | 84.37 | 50.70 |
| ast_first | 41.43 | 55.77 | 38.84 |
| plt_first | 244.77 | 262.46 | 208.76 |
| ggt_first | 105.52 | 182.30 | 133.01 |
| fibs_stiffness_med_BM_1_first | 7.89 | 7.70 | 11.45 |
| fibrotest_BM_2_first | 0.31 | 0.41 | 0.48 |

## Hepatic FP-like (top 30 by consensus rank, time≥5y)

| pid | time | rank | base_age | span | FibroScan₁ | FibroTest₁ |
|---|---|---|---|---|---|---|
| 27YH6UAMWJPI | 5.0 | 0.89 | 69.0 | 5.0 | 37.1 | 0.6 |
| E9QDI3G13IWP | 5.0 | 0.87 | 43.0 | 5.0 | 23.5 | 0.2 |
| 8JX54PEGJKS4 | 5.0 | 0.83 | 64.0 | 5.0 | 15.7 | 0.5 |
| 8KSZBFFZZDVG | 12.0 | 0.82 | 66.0 | 12.0 | 23.2 | 0.8 |
| LYZJ0AEOORBT | 5.0 | 0.80 | 58.0 | 5.0 | 14.8 | 0.5 |
| LFQF5E8OSAQ3 | 8.0 | 0.78 | 72.0 | 8.0 | 11.2 | 0.8 |
| ZS9MTHD3IGK3 | 7.0 | 0.78 | 50.0 | 7.0 | 33.3 | 0.9 |
| WCYUKND1IVXA | 5.0 | 0.77 | 65.0 | 5.0 | 12.2 | 0.6 |
| K0AXYCELOS6Z | 16.0 | 0.75 | 56.0 | 16.0 | 11.6 | 0.9 |
| OYC3RWQMMU7J | 6.0 | 0.75 | 55.0 | 6.0 | 11.4 | 0.4 |
| IWZOD05JH2MK | 5.0 | 0.74 | 64.0 | 5.0 | 19.1 | 0.5 |
| LPVOWCP77946 | 9.0 | 0.74 | 52.0 | 9.0 | 8.0 | 0.1 |
| WE4C1V47LZN9 | 8.0 | 0.73 | 58.0 | 8.0 | 16.8 | 0.7 |
| QTXHI15CQZCU | 8.0 | 0.72 | 49.0 | 8.0 | 11.0 | 0.8 |
| E64SKBCBDPDX | 6.0 | 0.72 | 64.0 | 6.0 | 9.3 | 0.8 |
| J36TCFKJNAWN | 12.0 | 0.71 | 49.0 | 12.0 | 5.0 | 0.7 |
| R0TOKPK7YSWN | 6.0 | 0.71 | 63.0 | 6.0 | 14.8 | 0.9 |
| CB1KJTGQ01TE | 10.0 | 0.70 | 74.0 | 10.0 | 5.0 | 0.5 |
| RRPZLK3E4XYF | 9.0 | 0.70 | 67.0 | 9.0 | 7.9 | 0.4 |
| IRJUTFVJ839A | 6.0 | 0.70 | 64.0 | 6.0 | 14.3 | 0.3 |
| Y3RZTPLC6PRR | 5.0 | 0.70 | 62.0 | 5.0 | 4.5 | 0.4 |
| PUXQPHR62GDS | 11.0 | 0.70 | 74.0 | 11.0 | 6.9 | 0.5 |
| 5GRO9X49TMY9 | 9.0 | 0.70 | 51.0 | 9.0 | 8.8 | 0.2 |
| I6FQJJA26YFP | 12.0 | 0.70 | 59.0 | 12.0 | 14.0 | 0.6 |
| 8K06MOXSFH4G | 6.0 | 0.70 | 52.0 | 6.0 | 14.2 | 0.9 |
| HAEWABHB7ZHE | 5.0 | 0.69 | 58.0 | 5.0 | 3.0 | — |
| HN3F9ZXQERBK | 7.0 | 0.69 | 59.0 | 7.0 | 36.6 | 0.7 |
| LRZQ8BXY6RKV | 6.0 | 0.69 | 70.0 | 6.0 | 5.8 | 0.7 |
| AR0KO31NZSXP | 5.0 | 0.69 | 61.0 | 5.0 | 6.1 | 0.4 |
| JPZTC09ZGNX3 | 9.0 | 0.69 | 61.0 | 9.0 | 7.1 | 0.5 |

## Patterns

- Hepatic FN: baseline age 48.2y vs pop 54.1y; followup span 13.4y vs pop 6.0y.
- Hepatic FP: baseline age 59.5y vs pop 54.1y; FibroScan baseline 12.5 vs pop 7.8.

## Generated submissions

- `sub_master_deepfn_targeted.csv` — anchor + alpha·(P(FN)−P(FP)) per patient.
- `sub_master_deepfn_specialist.csv` — anchor blended 50/50 with FN-specialist sub-ensemble for FN-prone val patients.
- `sub_master_deepfn_isolation.csv` — anchor blended toward 20%-trimmed-mean val ensemble for FN/FP-prone or anomalous patients.
- `sub_master_deepfn_targeted_death.csv` — death-only targeted adjust.
