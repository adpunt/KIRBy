# Real censoring: ginkgo_pr_cho, with QM9 beside it

Written by Claude on 2026-10-10 from the server run of `tests/experiments/noise/n30b_real_censoring.py`.
Result files: `/Volumes/seagate/kirby_results_2026-10-09_noise_view/noise/n30b_real_censoring/`.
Every table below is printed by `scratchpad/thesis_ch5_8/prcho_tables.py` from those files.
The prose is Claude's reading of the tables. It is not for `thesis.tex` until the author decides.

## What was run

- ginkgo_pr_cho: 58 of 197 PR_CHO labels are exactly 0. Nothing was clipped by us. The 0 is
  treated as a lower limit. Training labels at 0 per fold: 37, 51, 43, 47, 54.
- QM9: a quarter of the training labels were clipped at the top first (n30's injection). About
  1,997 of 7,982 training molecules per fold sit at the limit. Here the true labels are known.
- Both: all 5 outer folds, each dataset's two decided models (ginkgo_pr_cho gp and dnn, after
  the gp's old results were cleared; QM9 dnn and lightgbm). Per fold and model, the best single
  representation (`base_1`) and the first hybrid (`hybrid`) come from that fold's own search.
  The second of each is in the result files and behaves the same way; it is left out here.
- Scores. ginkgo_pr_cho: Spearman on all held-out antibodies, Spearman on the held-out antibodies
  above 0, and Harrell's C (pairs whose order the floor hides are skipped). QM9: RMSE against the
  unclipped held-out labels, lower is better.
- Each cell: the mean over the 5 folds. "Difference" is the treatment minus no treatment, per
  fold, then its mean and its standard deviation over the 5 folds. No significance tests.

## What it shows (Claude's reading)

**QM9, where the truth is known.** Moving the labels at the limit works with both models and
all four feature spaces. Censored EM has the lowest RMSE in every configuration but one: with dnn
on the second-best single representation, the constant lift is lower (0.4490 against 0.4519).
Hybrid with dnn: RMSE
0.5072 with no treatment, 0.3980 with censored EM, 0.3169 with the unclipped labels.

**The flag carries the QM9 gain.** The same move on the same number of labels NOT at the limit
makes RMSE worse. Hybrid with dnn: 0.5072 to 0.5180, 0.5206 and 0.5253 over the three seeds.

**Per-molecule amounts add nothing on QM9.** Dealing the same lifts to the wrong molecules at the
limit scores as well as the right assignment. Hybrid with dnn: 0.4209 right, 0.4135 shuffled.

**Deleting the labels at the limit hurts on QM9**, in every row. Hybrid with lightgbm: 0.5155 to
0.6323.

**ginkgo_pr_cho, where the truth is unknown: almost no treatment helps.** Nearly every mean
difference is smaller than its fold standard deviation, under all three scores. With the gp, the
best single representation goes 0.4980 to 0.4913 Spearman with censored EM. One exception gains:
winsorising at c = 1.25, dnn, first hybrid, 0.4114 to 0.4359 Spearman (fold sd 0.0211). One
loses: the per-molecule lift, gp, first hybrid, 0.4555 to 0.4058 (fold sd 0.0479).

**On ginkgo_pr_cho the not-at-0 control mostly does worse than the real treatments**, but its
fold spread is wide. With the gp on the hybrid: 0.4555 with no treatment, 0.3757 and 0.3561 on
two of the three seeds.

**Hybrid against single representation** (first hybrid against best single). On QM9 the hybrid
has the lower RMSE before and after every treatment, with both models. dnn, censored EM: single 0.4295, hybrid 0.3980. On
ginkgo_pr_cho it depends on the model. gp: single 0.4980, hybrid 0.4555 Spearman with no
treatment. dnn: single 0.3993, hybrid 0.4114.

**Against injected censoring.** QM9 here repeats the injected-censoring result of n30 on all 5
folds, and adds the control n30 never had. The real floor on ginkgo_pr_cho does not repeat it.
Two differences are worth naming. The real floor holds 0.24 to 0.33 of the training labels, all
at one value. The held-out labels carry the same floor, so the scores themselves are censored.

## Tables

One row is one treatment. Each block is one dataset, one model and one score.

#### ginkgo_pr_cho, gp, spearman_all

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.4980 |  | 0.4555 |  |
| Constant lift | 0.4787 | -0.0193 (0.0378) | 0.4288 | -0.0267 (0.0402) |
| Per-molecule lift | 0.4777 | -0.0203 (0.0327) | 0.4058 | -0.0497 (0.0479) |
| Per-molecule lift, shuffled (seed 0) | 0.4811 | -0.0169 (0.0500) | 0.4392 | -0.0163 (0.0394) |
| Truncated-normal mean | 0.4910 | -0.0070 (0.0181) | 0.4516 | -0.0038 (0.0163) |
| Censored EM | 0.4913 | -0.0068 (0.0243) | 0.4495 | -0.0060 (0.0196) |
| Delete labels at the limit | 0.4500 | -0.0480 (0.0964) | 0.4516 | -0.0039 (0.0712) |
| Same move, labels NOT at the limit (seed 0) | 0.4576 | -0.0405 (0.1445) | 0.4317 | -0.0237 (0.0484) |
| Same move, labels NOT at the limit (seed 1) | 0.4404 | -0.0576 (0.1225) | 0.3757 | -0.0798 (0.0766) |
| Same move, labels NOT at the limit (seed 2) | 0.4547 | -0.0433 (0.0922) | 0.3561 | -0.0994 (0.0975) |
| Winsorise, c = 1 | 0.4795 | -0.0185 (0.0360) | 0.4310 | -0.0245 (0.0313) |
| Winsorise, c = 1.25 | 0.4831 | -0.0150 (0.0414) | 0.4363 | -0.0192 (0.0213) |
| Winsorise, c = 1.5 | 0.4877 | -0.0103 (0.0364) | 0.4550 | -0.0005 (0.0152) |

#### ginkgo_pr_cho, dnn, spearman_all

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.3993 |  | 0.4114 |  |
| Constant lift | 0.3726 | -0.0267 (0.0437) | 0.3882 | -0.0231 (0.0911) |
| Per-molecule lift | 0.3694 | -0.0299 (0.0429) | 0.3900 | -0.0214 (0.0861) |
| Per-molecule lift, shuffled (seed 0) | 0.3791 | -0.0202 (0.0416) | 0.3911 | -0.0203 (0.0810) |
| Truncated-normal mean | 0.4131 | +0.0138 (0.0479) | 0.3819 | -0.0294 (0.0820) |
| Censored EM | 0.3923 | -0.0070 (0.0228) | 0.3828 | -0.0286 (0.0840) |
| Delete labels at the limit | 0.3461 | -0.0532 (0.1755) | 0.4247 | +0.0133 (0.1629) |
| Same move, labels NOT at the limit (seed 0) | 0.3624 | -0.0369 (0.0978) | 0.3833 | -0.0281 (0.0259) |
| Same move, labels NOT at the limit (seed 1) | 0.3961 | -0.0032 (0.1119) | 0.3155 | -0.0959 (0.0770) |
| Same move, labels NOT at the limit (seed 2) | 0.3881 | -0.0111 (0.0967) | 0.4338 | +0.0225 (0.0666) |
| Winsorise, c = 1 | 0.4257 | +0.0264 (0.0531) | 0.4360 | +0.0247 (0.0330) |
| Winsorise, c = 1.25 | 0.4253 | +0.0260 (0.0410) | 0.4359 | +0.0245 (0.0211) |
| Winsorise, c = 1.5 | 0.4068 | +0.0075 (0.0330) | 0.4292 | +0.0178 (0.0181) |

#### ginkgo_pr_cho, gp, spearman_above_floor

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.3745 |  | 0.3628 |  |
| Constant lift | 0.3648 | -0.0096 (0.0490) | 0.3489 | -0.0139 (0.0580) |
| Per-molecule lift | 0.3376 | -0.0369 (0.0676) | 0.3197 | -0.0431 (0.0774) |
| Per-molecule lift, shuffled (seed 0) | 0.3717 | -0.0028 (0.0583) | 0.3655 | +0.0027 (0.0531) |
| Truncated-normal mean | 0.3718 | -0.0027 (0.0168) | 0.3589 | -0.0038 (0.0201) |
| Censored EM | 0.3767 | +0.0022 (0.0175) | 0.3542 | -0.0086 (0.0298) |
| Delete labels at the limit | 0.3061 | -0.0684 (0.1210) | 0.3132 | -0.0495 (0.0743) |
| Same move, labels NOT at the limit (seed 0) | 0.3242 | -0.0503 (0.1521) | 0.2978 | -0.0650 (0.0625) |
| Same move, labels NOT at the limit (seed 1) | 0.2929 | -0.0815 (0.1750) | 0.2724 | -0.0903 (0.1327) |
| Same move, labels NOT at the limit (seed 2) | 0.2746 | -0.0998 (0.2390) | 0.2467 | -0.1161 (0.1421) |
| Winsorise, c = 1 | 0.3498 | -0.0247 (0.0539) | 0.3402 | -0.0226 (0.0547) |
| Winsorise, c = 1.25 | 0.3625 | -0.0120 (0.0468) | 0.3462 | -0.0166 (0.0290) |
| Winsorise, c = 1.5 | 0.3618 | -0.0127 (0.0318) | 0.3595 | -0.0032 (0.0300) |

#### ginkgo_pr_cho, dnn, spearman_above_floor

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.3105 |  | 0.2973 |  |
| Constant lift | 0.2771 | -0.0334 (0.0825) | 0.3042 | +0.0068 (0.1050) |
| Per-molecule lift | 0.2939 | -0.0166 (0.0690) | 0.3012 | +0.0038 (0.1050) |
| Per-molecule lift, shuffled (seed 0) | 0.2829 | -0.0276 (0.0692) | 0.2975 | +0.0002 (0.0660) |
| Truncated-normal mean | 0.3071 | -0.0033 (0.0100) | 0.2843 | -0.0130 (0.0751) |
| Censored EM | 0.3089 | -0.0016 (0.0162) | 0.2809 | -0.0164 (0.0786) |
| Delete labels at the limit | 0.2279 | -0.0826 (0.3110) | 0.3062 | +0.0089 (0.2630) |
| Same move, labels NOT at the limit (seed 0) | 0.2392 | -0.0713 (0.2198) | 0.2779 | -0.0195 (0.0766) |
| Same move, labels NOT at the limit (seed 1) | 0.2558 | -0.0547 (0.2524) | 0.2122 | -0.0851 (0.1309) |
| Same move, labels NOT at the limit (seed 2) | 0.2164 | -0.0941 (0.2188) | 0.3251 | +0.0277 (0.0278) |
| Winsorise, c = 1 | 0.3354 | +0.0249 (0.0869) | 0.3245 | +0.0272 (0.0604) |
| Winsorise, c = 1.25 | 0.3474 | +0.0369 (0.0520) | 0.3167 | +0.0194 (0.0305) |
| Winsorise, c = 1.5 | 0.3197 | +0.0093 (0.0364) | 0.3211 | +0.0238 (0.0293) |

#### ginkgo_pr_cho, gp, harrell_c

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.6891 |  | 0.6729 |  |
| Constant lift | 0.6842 | -0.0049 (0.0124) | 0.6622 | -0.0107 (0.0162) |
| Per-molecule lift | 0.6821 | -0.0070 (0.0149) | 0.6533 | -0.0196 (0.0205) |
| Per-molecule lift, shuffled (seed 0) | 0.6846 | -0.0045 (0.0185) | 0.6645 | -0.0083 (0.0171) |
| Truncated-normal mean | 0.6866 | -0.0026 (0.0061) | 0.6703 | -0.0026 (0.0080) |
| Censored EM | 0.6858 | -0.0033 (0.0093) | 0.6696 | -0.0033 (0.0102) |
| Delete labels at the limit | 0.6659 | -0.0232 (0.0344) | 0.6669 | -0.0060 (0.0262) |
| Same move, labels NOT at the limit (seed 0) | 0.6735 | -0.0156 (0.0535) | 0.6644 | -0.0085 (0.0155) |
| Same move, labels NOT at the limit (seed 1) | 0.6640 | -0.0251 (0.0518) | 0.6359 | -0.0370 (0.0304) |
| Same move, labels NOT at the limit (seed 2) | 0.6663 | -0.0228 (0.0368) | 0.6301 | -0.0427 (0.0399) |
| Winsorise, c = 1 | 0.6795 | -0.0096 (0.0166) | 0.6652 | -0.0076 (0.0091) |
| Winsorise, c = 1.25 | 0.6841 | -0.0050 (0.0157) | 0.6660 | -0.0069 (0.0065) |
| Winsorise, c = 1.5 | 0.6850 | -0.0042 (0.0123) | 0.6740 | +0.0012 (0.0079) |

#### ginkgo_pr_cho, dnn, harrell_c

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.6461 |  | 0.6581 |  |
| Constant lift | 0.6416 | -0.0045 (0.0198) | 0.6488 | -0.0093 (0.0361) |
| Per-molecule lift | 0.6380 | -0.0081 (0.0172) | 0.6486 | -0.0095 (0.0328) |
| Per-molecule lift, shuffled (seed 0) | 0.6426 | -0.0035 (0.0168) | 0.6509 | -0.0072 (0.0347) |
| Truncated-normal mean | 0.6503 | +0.0042 (0.0150) | 0.6452 | -0.0129 (0.0335) |
| Censored EM | 0.6463 | +0.0002 (0.0106) | 0.6455 | -0.0126 (0.0336) |
| Delete labels at the limit | 0.6249 | -0.0212 (0.0530) | 0.6626 | +0.0046 (0.0682) |
| Same move, labels NOT at the limit (seed 0) | 0.6311 | -0.0150 (0.0361) | 0.6461 | -0.0120 (0.0084) |
| Same move, labels NOT at the limit (seed 1) | 0.6428 | -0.0034 (0.0375) | 0.6207 | -0.0374 (0.0329) |
| Same move, labels NOT at the limit (seed 2) | 0.6396 | -0.0065 (0.0324) | 0.6626 | +0.0045 (0.0313) |
| Winsorise, c = 1 | 0.6582 | +0.0121 (0.0260) | 0.6672 | +0.0091 (0.0163) |
| Winsorise, c = 1.25 | 0.6603 | +0.0142 (0.0199) | 0.6660 | +0.0079 (0.0115) |
| Winsorise, c = 1.5 | 0.6510 | +0.0049 (0.0178) | 0.6652 | +0.0071 (0.0096) |

#### qm9__injected0.25, dnn, rmse

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.5501 |  | 0.5072 |  |
| Unclipped labels (QM9 only) | 0.3615 | -0.1886 (0.0454) | 0.3169 | -0.1902 (0.0422) |
| Constant lift | 0.4495 | -0.1006 (0.0246) | 0.4176 | -0.0895 (0.0187) |
| Per-molecule lift | 0.4573 | -0.0928 (0.0173) | 0.4209 | -0.0862 (0.0192) |
| Per-molecule lift, shuffled (seed 0) | 0.4528 | -0.0973 (0.0252) | 0.4135 | -0.0936 (0.0202) |
| Truncated-normal mean | 0.4600 | -0.0901 (0.0202) | 0.4219 | -0.0852 (0.0155) |
| Censored EM | 0.4295 | -0.1207 (0.0252) | 0.3980 | -0.1092 (0.0244) |
| Delete labels at the limit | 0.5962 | +0.0461 (0.0132) | 0.5328 | +0.0256 (0.0103) |
| Same move, labels NOT at the limit (seed 0) | 0.5643 | +0.0142 (0.0100) | 0.5253 | +0.0182 (0.0035) |
| Same move, labels NOT at the limit (seed 1) | 0.5626 | +0.0125 (0.0114) | 0.5180 | +0.0108 (0.0055) |
| Same move, labels NOT at the limit (seed 2) | 0.5638 | +0.0136 (0.0034) | 0.5206 | +0.0134 (0.0026) |
| Winsorise, c = 1 | 0.5534 | +0.0033 (0.0117) | 0.5207 | +0.0135 (0.0053) |
| Winsorise, c = 1.25 | 0.5493 | -0.0008 (0.0092) | 0.5166 | +0.0095 (0.0061) |
| Winsorise, c = 1.5 | 0.5487 | -0.0014 (0.0082) | 0.5148 | +0.0076 (0.0062) |

#### qm9__injected0.25, lightgbm, rmse

| Treatment | base_1 mean | base_1 difference (fold sd) | hybrid mean | hybrid difference (fold sd) |
|---|---|---|---|---|
| No treatment | 0.5348 |  | 0.5155 |  |
| Unclipped labels (QM9 only) | 0.3720 | -0.1627 (0.0400) | 0.3468 | -0.1687 (0.0401) |
| Constant lift | 0.4480 | -0.0868 (0.0222) | 0.4321 | -0.0835 (0.0181) |
| Per-molecule lift | 0.4545 | -0.0803 (0.0195) | 0.4389 | -0.0766 (0.0157) |
| Per-molecule lift, shuffled (seed 0) | 0.4489 | -0.0858 (0.0189) | 0.4344 | -0.0811 (0.0198) |
| Truncated-normal mean | 0.4591 | -0.0757 (0.0157) | 0.4412 | -0.0744 (0.0145) |
| Censored EM | 0.4202 | -0.1146 (0.0285) | 0.4031 | -0.1125 (0.0250) |
| Delete labels at the limit | 0.6552 | +0.1205 (0.0341) | 0.6323 | +0.1167 (0.0264) |
| Same move, labels NOT at the limit (seed 0) | 0.5420 | +0.0073 (0.0031) | 0.5226 | +0.0070 (0.0022) |
| Same move, labels NOT at the limit (seed 1) | 0.5407 | +0.0060 (0.0038) | 0.5212 | +0.0057 (0.0033) |
| Same move, labels NOT at the limit (seed 2) | 0.5417 | +0.0070 (0.0026) | 0.5208 | +0.0052 (0.0026) |
| Winsorise, c = 1 | 0.5513 | +0.0166 (0.0053) | 0.5291 | +0.0136 (0.0034) |
| Winsorise, c = 1.25 | 0.5476 | +0.0129 (0.0040) | 0.5274 | +0.0118 (0.0037) |
| Winsorise, c = 1.5 | 0.5469 | +0.0122 (0.0047) | 0.5244 | +0.0089 (0.0025) |
