# Overall: does it work, by group (Chapter 7 signed-off datasets and methods)

Written by Claude 2026-10-09. Fold 0 only; each method's best setting picked on the held-out score.

One row = one dataset x noise condition x model x method, comparing the best single representation and the best hybrid, each before and after the method. "yes" = the method improved both the single representation and the hybrid (and beat its control, where the table has one). "hybrid better after" = the hybrid's score after the method beats the single representation's score after the method (4 decimals). "no score" = one or both of the two cells is n/a. "as delivered" = no noise added (level 0); "added damage" = the main damaged level of each noise kind.

## A. All 13 datasets (QM9 labels-as-delivered rows left out; they are on their own below)

| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |
|---|---|---|---|---|---|---|---|---|---|---|
| noise-robust training | as delivered | 179 | 73 | 57 | 16 | 0 | 31 | 44 | 31 | 0 |
| noise-robust training | added damage | 798 | 380 | 304 | 76 | 0 | 130 | 139 | 149 | 0 |
| noise-robust training | both | 977 | 453 | 361 | 92 | 0 | 161 | 183 | 180 | 0 |
| mitigation, selected by a detector | as delivered | 416 | 144 | 111 | 33 | 0 | 77 | 60 | 131 | 4 |
| mitigation, selected by a detector | added damage | 1349 | 731 | 598 | 132 | 1 | 134 | 163 | 305 | 16 |
| mitigation, selected by a detector | both | 1765 | 875 | 709 | 165 | 1 | 211 | 223 | 436 | 20 |
| mitigation, selected by a known flag | as delivered | 88 | 21 | 21 | 0 | 0 | 14 | 1 | 36 | 16 |
| mitigation, selected by a known flag | added damage | 346 | 165 | 128 | 37 | 0 | 44 | 48 | 87 | 2 |
| mitigation, selected by a known flag | both | 434 | 186 | 149 | 37 | 0 | 58 | 49 | 123 | 18 |
| mitigation, every molecule | as delivered | 8 | 2 | 2 | 0 | 0 | 1 | 0 | 5 | 0 |
| mitigation, every molecule | added damage | 24 | 17 | 16 | 1 | 0 | 0 | 0 | 7 | 0 |
| mitigation, every molecule | both | 32 | 19 | 18 | 1 | 0 | 1 | 0 | 12 | 0 |
| mitigation total (detector + flag + every molecule) | as delivered | 512 | 167 | 134 | 33 | 0 | 92 | 61 | 172 | 20 |
| mitigation total (detector + flag + every molecule) | added damage | 1719 | 913 | 742 | 170 | 1 | 178 | 211 | 399 | 18 |
| mitigation total (detector + flag + every molecule) | both | 2231 | 1080 | 876 | 203 | 1 | 270 | 272 | 571 | 38 |
| all four groups | as delivered | 691 | 240 | 191 | 49 | 0 | 123 | 105 | 203 | 20 |
| all four groups | added damage | 2517 | 1293 | 1046 | 246 | 1 | 308 | 350 | 548 | 18 |
| all four groups | both | 3208 | 1533 | 1237 | 295 | 1 | 431 | 455 | 751 | 38 |

QM9, labels as delivered (no noise added). Any "yes" here is a gain on clean labels, so it is not noise removal.

| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |
|---|---|---|---|---|---|---|---|---|---|---|
| noise-robust training | as delivered | 15 | 3 | 3 | 0 | 0 | 1 | 3 | 8 | 0 |
| mitigation, selected by a detector | as delivered | 11 | 2 | 2 | 0 | 0 | 0 | 2 | 7 | 0 |
| mitigation, selected by a known flag | as delivered | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| mitigation, every molecule | as delivered | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mitigation total | as delivered | 15 | 2 | 2 | 0 | 0 | 0 | 2 | 7 | 4 |

## B. By domain

Small molecules = qm9 (added-damage rows only), caco2, herg_ki, logd, herg, herg_ic50. Reaction = buchwald_hartwig. Antibodies = flab_thermo_tresanco, ddg_combined, ginkgo_titer, ginkgo_pr_cho, ginkgo_colloidal, ginkgo_acsins.

### small molecules

| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |
|---|---|---|---|---|---|---|---|---|---|---|
| noise-robust training | as delivered | 95 | 34 | 33 | 1 | 0 | 13 | 25 | 23 | 0 |
| noise-robust training | added damage | 491 | 236 | 219 | 17 | 0 | 70 | 77 | 108 | 0 |
| noise-robust training | both | 586 | 270 | 252 | 18 | 0 | 83 | 102 | 131 | 0 |
| mitigation, selected by a detector | as delivered | 232 | 70 | 67 | 3 | 0 | 32 | 26 | 104 | 0 |
| mitigation, selected by a detector | added damage | 929 | 512 | 477 | 34 | 1 | 77 | 98 | 242 | 0 |
| mitigation, selected by a detector | both | 1161 | 582 | 544 | 37 | 1 | 109 | 124 | 346 | 0 |
| mitigation, selected by a known flag | as delivered | 62 | 20 | 20 | 0 | 0 | 11 | 1 | 26 | 4 |
| mitigation, selected by a known flag | added damage | 202 | 99 | 97 | 2 | 0 | 22 | 24 | 57 | 0 |
| mitigation, selected by a known flag | both | 264 | 119 | 117 | 2 | 0 | 33 | 25 | 83 | 4 |
| mitigation, every molecule | as delivered | 6 | 1 | 1 | 0 | 0 | 1 | 0 | 4 | 0 |
| mitigation, every molecule | added damage | 24 | 17 | 16 | 1 | 0 | 0 | 0 | 7 | 0 |
| mitigation, every molecule | both | 30 | 18 | 17 | 1 | 0 | 1 | 0 | 11 | 0 |
| mitigation total (detector + flag + every molecule) | as delivered | 300 | 91 | 88 | 3 | 0 | 44 | 27 | 134 | 4 |
| mitigation total (detector + flag + every molecule) | added damage | 1155 | 628 | 590 | 37 | 1 | 99 | 122 | 306 | 0 |
| mitigation total (detector + flag + every molecule) | both | 1455 | 719 | 678 | 40 | 1 | 143 | 149 | 440 | 4 |
| all four groups | as delivered | 395 | 125 | 121 | 4 | 0 | 57 | 52 | 157 | 4 |
| all four groups | added damage | 1646 | 864 | 809 | 54 | 1 | 169 | 199 | 414 | 0 |
| all four groups | both | 2041 | 989 | 930 | 58 | 1 | 226 | 251 | 571 | 4 |

### reaction

| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |
|---|---|---|---|---|---|---|---|---|---|---|
| noise-robust training | as delivered | 19 | 9 | 9 | 0 | 0 | 2 | 4 | 4 | 0 |
| noise-robust training | added damage | 68 | 35 | 26 | 9 | 0 | 8 | 17 | 8 | 0 |
| noise-robust training | both | 87 | 44 | 35 | 9 | 0 | 10 | 21 | 12 | 0 |
| mitigation, selected by a detector | as delivered | 47 | 15 | 7 | 8 | 0 | 5 | 11 | 12 | 4 |
| mitigation, selected by a detector | added damage | 124 | 68 | 35 | 33 | 0 | 8 | 20 | 12 | 16 |
| mitigation, selected by a detector | both | 171 | 83 | 42 | 41 | 0 | 13 | 31 | 24 | 20 |
| mitigation, selected by a known flag | as delivered | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| mitigation, selected by a known flag | added damage | 60 | 38 | 18 | 20 | 0 | 8 | 4 | 8 | 2 |
| mitigation, selected by a known flag | both | 64 | 38 | 18 | 20 | 0 | 8 | 4 | 8 | 6 |
| mitigation, every molecule | as delivered | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| mitigation, every molecule | added damage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mitigation, every molecule | both | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| mitigation total (detector + flag + every molecule) | as delivered | 53 | 16 | 8 | 8 | 0 | 5 | 11 | 13 | 8 |
| mitigation total (detector + flag + every molecule) | added damage | 184 | 106 | 53 | 53 | 0 | 16 | 24 | 20 | 18 |
| mitigation total (detector + flag + every molecule) | both | 237 | 122 | 61 | 61 | 0 | 21 | 35 | 33 | 26 |
| all four groups | as delivered | 72 | 25 | 17 | 8 | 0 | 7 | 15 | 17 | 8 |
| all four groups | added damage | 252 | 141 | 79 | 62 | 0 | 24 | 41 | 28 | 18 |
| all four groups | both | 324 | 166 | 96 | 70 | 0 | 31 | 56 | 45 | 26 |

### antibodies

| group | labels | rows | yes | yes, hybrid better after | yes, single better after | yes, tie after | single only | hybrid only | neither | no score |
|---|---|---|---|---|---|---|---|---|---|---|
| noise-robust training | as delivered | 65 | 30 | 15 | 15 | 0 | 16 | 15 | 4 | 0 |
| noise-robust training | added damage | 239 | 109 | 59 | 50 | 0 | 52 | 45 | 33 | 0 |
| noise-robust training | both | 304 | 139 | 74 | 65 | 0 | 68 | 60 | 37 | 0 |
| mitigation, selected by a detector | as delivered | 137 | 59 | 37 | 22 | 0 | 40 | 23 | 15 | 0 |
| mitigation, selected by a detector | added damage | 296 | 151 | 86 | 65 | 0 | 49 | 45 | 51 | 0 |
| mitigation, selected by a detector | both | 433 | 210 | 123 | 87 | 0 | 89 | 68 | 66 | 0 |
| mitigation, selected by a known flag | as delivered | 22 | 1 | 1 | 0 | 0 | 3 | 0 | 10 | 8 |
| mitigation, selected by a known flag | added damage | 84 | 28 | 13 | 15 | 0 | 14 | 20 | 22 | 0 |
| mitigation, selected by a known flag | both | 106 | 29 | 14 | 15 | 0 | 17 | 20 | 32 | 8 |
| mitigation, every molecule | as delivered | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mitigation, every molecule | added damage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mitigation, every molecule | both | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mitigation total (detector + flag + every molecule) | as delivered | 159 | 60 | 38 | 22 | 0 | 43 | 23 | 25 | 8 |
| mitigation total (detector + flag + every molecule) | added damage | 380 | 179 | 99 | 80 | 0 | 63 | 65 | 73 | 0 |
| mitigation total (detector + flag + every molecule) | both | 539 | 239 | 137 | 102 | 0 | 106 | 88 | 98 | 8 |
| all four groups | as delivered | 224 | 90 | 53 | 37 | 0 | 59 | 38 | 29 | 8 |
| all four groups | added damage | 619 | 288 | 158 | 130 | 0 | 115 | 110 | 106 | 0 |
| all four groups | both | 843 | 378 | 211 | 167 | 0 | 174 | 148 | 135 | 8 |

## C. Hybrid against single representation, before and after the method

One row of this table counts table rows with both scores. "Before" compares the two scores with no method, at the 3 decimals the page prints; "after" is the page's own column, at 4 decimals. QM9 labels-as-delivered rows left out.

| domain | group | labels | rows with scores | hybrid better before and after | hybrid worse before, better after | hybrid better before, worse after | hybrid worse before and after | tie before or after |
|---|---|---|---|---|---|---|---|---|
| small molecules | noise-robust training | as delivered | 95 | 80 | 5 | 4 | 4 | 2 |
| small molecules | noise-robust training | added damage | 491 | 420 | 32 | 19 | 20 | 0 |
| small molecules | noise-robust training | both | 586 | 500 | 37 | 23 | 24 | 2 |
| small molecules | mitigation total | as delivered | 296 | 267 | 17 | 6 | 6 | 0 |
| small molecules | mitigation total | added damage | 1155 | 1034 | 54 | 23 | 35 | 9 |
| small molecules | mitigation total | both | 1451 | 1301 | 71 | 29 | 41 | 9 |
| reaction | noise-robust training | as delivered | 19 | 8 | 10 | 0 | 1 | 0 |
| reaction | noise-robust training | added damage | 68 | 42 | 11 | 10 | 5 | 0 |
| reaction | noise-robust training | both | 87 | 50 | 21 | 10 | 6 | 0 |
| reaction | mitigation total | as delivered | 45 | 22 | 9 | 4 | 10 | 0 |
| reaction | mitigation total | added damage | 166 | 85 | 12 | 3 | 66 | 0 |
| reaction | mitigation total | both | 211 | 107 | 21 | 7 | 76 | 0 |
| antibodies | noise-robust training | as delivered | 65 | 21 | 13 | 18 | 10 | 3 |
| antibodies | noise-robust training | added damage | 239 | 80 | 39 | 38 | 75 | 7 |
| antibodies | noise-robust training | both | 304 | 101 | 52 | 56 | 85 | 10 |
| antibodies | mitigation total | as delivered | 151 | 65 | 13 | 29 | 39 | 5 |
| antibodies | mitigation total | added damage | 380 | 167 | 45 | 33 | 127 | 8 |
| antibodies | mitigation total | both | 531 | 232 | 58 | 62 | 166 | 13 |
| all domains | noise-robust training | as delivered | 179 | 109 | 28 | 22 | 15 | 5 |
| all domains | noise-robust training | added damage | 798 | 542 | 82 | 67 | 100 | 7 |
| all domains | noise-robust training | both | 977 | 651 | 110 | 89 | 115 | 12 |
| all domains | mitigation total | as delivered | 492 | 354 | 39 | 39 | 55 | 5 |
| all domains | mitigation total | added damage | 1701 | 1286 | 111 | 59 | 228 | 17 |
| all domains | mitigation total | both | 2193 | 1640 | 150 | 98 | 283 | 22 |

## D. Per method (signed-off DEEP and BROAD only)

rows = rows with both scores, QM9 labels-as-delivered left out; yes as above. "QM9 clean" = yes of rows on QM9 labels as delivered. Datasets listed are those with at least one yes row (qm9 = a yes under added damage).

| group | exp | method | rows | yes | yes as delivered | yes added damage | QM9 clean: yes of rows | datasets with a yes row |
|---|---|---|---|---|---|---|---|---|
| noise-robust training | n40 | stack | 44 | 39 | 5 | 34 | 0 of 0 | caco2, herg_ki, logd |
| noise-robust training | n40 | seed_avg | 22 | 19 | 2 | 17 | 0 of 0 | caco2, herg_ki, logd |
| noise-robust training | n40 | retuned | 44 | 34 | 4 | 30 | 0 of 0 | caco2, herg_ki, logd |
| noise-robust training | n40 | tuned_clean_reused | 44 | 21 | 4 | 17 | 0 of 0 | caco2, herg_ki, logd |
| noise-robust training | n09 | heteroscedastic_gp | 86 | 44 | 8 | 36 | 1 of 2 | qm9, caco2, herg_ki, logd, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | evidential_kernel | 86 | 39 | 7 | 32 | 0 of 2 | qm9, caco2, herg_ki, logd, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | early_learning | 100 | 40 | 7 | 33 | 0 of 2 | qm9, caco2, herg_ki, logd, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | uncertainty_curriculum | 100 | 40 | 5 | 35 | 0 of 2 | qm9, caco2, herg_ki, logd, herg, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | meta_weight_net | 100 | 37 | 6 | 31 | 1 of 2 | qm9, caco2, herg_ki, logd, herg, BH, ddg, prCHO, colloidal, acsins |
| noise-robust training | n09 | mentornet | 100 | 41 | 4 | 37 | 0 of 2 | qm9, caco2, herg_ki, logd, herg, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | contrast_to_divide | 100 | 35 | 5 | 30 | 1 of 2 | qm9, caco2, herg_ki, logd, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n09 | DivideMix | 58 | 22 | 3 | 19 | 0 of 1 | qm9, caco2, herg_ki, BH, ddg, titer, prCHO, colloidal, acsins |
| noise-robust training | n33 | huber_dnn | 15 | 11 | 2 | 9 | 0 of 0 | caco2, herg_ki, logd |
| noise-robust training | n38 | robust_loss | 20 | 10 | 2 | 8 | 0 of 0 | herg_ki, logd, BH, ddg, titer, colloidal, acsins |
| noise-robust training | n32 | two_learner | 48 | 15 | 3 | 12 | 0 of 0 | BH, tresanco, ddg, titer, colloidal |
| noise-robust training | n29 | per_record | 2 | 2 | 2 | 0 | 0 of 0 | herg_ic50 |
| noise-robust training | n19 | ngboost_count_feature | 2 | 2 | 2 | 0 | 0 of 0 | herg_ic50 |
| noise-robust training | n19 | count_feature | 4 | 1 | 1 | 0 | 0 of 0 | herg_ic50 |
| noise-robust training | n19 | xgboost_interval_aft | 2 | 1 | 1 | 0 | 0 of 0 | herg_ic50 |
| mitigation, selected by a detector | n20 | search and fit on cleaned labels | 46 | 43 | 13 | 30 | 0 of 0 | caco2, herg_ki, logd, herg, BH, ddg, titer, colloidal |
| mitigation, selected by a detector | n21 | best distance | 64 | 52 | 11 | 41 | 0 of 0 | caco2, herg_ki, logd, herg, BH, ddg, colloidal |
| mitigation, selected by a detector | n13 | detector weights | 84 | 58 | 7 | 51 | 1 of 2 | qm9, caco2, herg_ki, logd, herg, BH, ddg, colloidal |
| mitigation, selected by a detector | n23 | activity_cliff_aware | 52 | 38 | 2 | 36 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n23 | gaussian_process | 52 | 36 | 2 | 34 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n23 | hetero_gp | 52 | 32 | 3 | 29 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n23 | heteroscedastic_nn | 52 | 31 | 4 | 27 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n23 | ngboost | 52 | 26 | 3 | 23 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n07 | remove | 78 | 38 | 4 | 34 | 0 of 0 | caco2, herg_ki, logd, herg, BH, titer, colloidal |
| mitigation, selected by a detector | n07 | label_smooth | 78 | 36 | 1 | 35 | 0 of 0 | caco2, herg_ki, logd, herg, BH, titer, colloidal |
| mitigation, selected by a detector | n07 | downweight | 78 | 34 | 3 | 31 | 0 of 0 | caco2, herg_ki, logd, herg, BH, titer, colloidal |
| mitigation, selected by a detector | n07 | oof_smooth | 54 | 26 | 1 | 25 | 0 of 0 | caco2, herg_ki, logd, titer, colloidal |
| mitigation, selected by a detector | n07 | js_shrink | 54 | 27 | 2 | 25 | 0 of 0 | caco2, herg_ki, logd, titer, colloidal |
| mitigation, selected by a detector | n24 | delete | 48 | 21 | 2 | 19 | 0 of 0 | caco2, herg_ki, herg, BH |
| mitigation, selected by a detector | n24 | downweight | 48 | 27 | 1 | 26 | 0 of 0 | caco2, herg_ki, logd, herg, BH |
| mitigation, selected by a detector | n24 | huber_weight | 48 | 28 | 2 | 26 | 0 of 0 | caco2, herg_ki, logd, herg, BH |
| mitigation, selected by a detector | n24 | label_smooth | 48 | 30 | 3 | 27 | 0 of 0 | caco2, herg_ki, logd, herg, BH |
| mitigation, selected by a detector | n12 | importance-scaled features | 105 | 64 | 8 | 56 | 0 of 0 | caco2, herg_ki, logd, herg, BH, tresanco, ddg, titer, colloidal, acsins |
| mitigation, selected by a detector | n12 | unscaled features | 105 | 60 | 7 | 53 | 0 of 0 | caco2, herg_ki, logd, herg, BH, tresanco, ddg, titer, colloidal, acsins |
| mitigation, selected by a detector | n35 | downweight_pairs | 32 | 21 | 6 | 15 | 0 of 0 | caco2, herg_ki, logd, BH, tresanco, ddg, titer, colloidal |
| mitigation, selected by a detector | n31 | downweight | 30 | 18 | 3 | 15 | 0 of 0 | caco2, herg_ki, ddg |
| mitigation, selected by a detector | n18 | remove | 20 | 5 | 5 | 0 | 0 of 0 | tresanco, titer, colloidal |
| mitigation, selected by a detector | n18 | downweight | 19 | 6 | 6 | 0 | 0 of 0 | caco2, logd, herg, ddg, colloidal, acsins |
| mitigation, selected by a detector | n18 | label_smooth | 20 | 4 | 4 | 0 | 0 of 0 | herg_ki, herg, ddg, colloidal |
| mitigation, selected by a detector | n18 | oof_smooth | 18 | 4 | 4 | 0 | 0 of 0 | caco2, logd, tresanco, colloidal |
| mitigation, selected by a detector | n02 | neighbour-agreement weights | 20 | 11 | 11 | 0 | 1 of 2 | caco2, herg, BH, tresanco, titer, prCHO, colloidal, acsins |
| mitigation, selected by a detector | n25 | cliff_put_back | 30 | 12 | 6 | 6 | 0 of 0 | caco2, herg_ki, logd, herg |
| mitigation, selected by a detector | n09 | small_loss | 100 | 24 | 4 | 20 | 0 of 2 | caco2, herg_ki, BH, ddg, titer, prCHO, colloidal |
| mitigation, selected by a detector | n09 | confident_learning | 100 | 34 | 7 | 27 | 0 of 2 | caco2, herg_ki, logd, herg, BH, ddg, titer, prCHO, colloidal, acsins |
| mitigation, selected by a detector | n09 | distance_selection | 58 | 20 | 6 | 14 | 0 of 1 | caco2, herg_ki, logd, herg, BH, ddg, titer, prCHO, colloidal, acsins |
| mitigation, selected by a detector | n09 | multistage_cleaning | 100 | 9 | 3 | 6 | 0 of 2 | BH, titer, prCHO, colloidal |
| mitigation, selected by a known flag | n30 | Censored EM | 22 | 15 | 3 | 12 | 0 of 0 | qm9, caco2, herg_ki, logd, BH, tresanco |
| mitigation, selected by a known flag | n30 | truncated_normal_mean | 22 | 15 | 3 | 12 | 0 of 0 | qm9, caco2, herg_ki, logd, BH, tresanco |
| mitigation, selected by a known flag | n30 | Constant lift | 22 | 12 | 1 | 11 | 0 of 0 | qm9, caco2, herg_ki, logd, BH, tresanco |
| mitigation, selected by a known flag | n30 | drop_clipped | 22 | 5 | 2 | 3 | 0 of 0 | caco2, logd, tresanco, ddg, colloidal |
| mitigation, selected by a known flag | n30 | per_molecule_lift | 22 | 1 | 1 | 0 | 0 of 0 | caco2 |
| mitigation, selected by a known flag | n07 | Constant lift | 18 | 12 | 2 | 10 | 0 of 0 | caco2, herg_ki, logd, titer, colloidal |
| mitigation, selected by a known flag | n18 | Constant lift | 6 | 2 | 2 | 0 | 0 of 0 | caco2, logd |
| mitigation, selected by a known flag | n34 | group_center | 18 | 6 | 0 | 6 | 0 of 0 | caco2, herg_ki, logd |
| mitigation, selected by a known flag | n34 | group_center_3plus | 18 | 7 | 1 | 6 | 0 of 0 | caco2, herg_ki, logd |
| mitigation, selected by a known flag | n34 | random_intercept | 18 | 10 | 0 | 10 | 0 of 0 | caco2, herg_ki, logd |
| mitigation, selected by a known flag | n39 | offset_removed | 54 | 21 | 0 | 21 | 0 of 0 | caco2, herg_ki, logd, BH, colloidal |
| mitigation, selected by a known flag | n39 | offset_removed_within_var_weights | 54 | 27 | 0 | 27 | 0 of 0 | caco2, herg_ki, logd, BH, colloidal |
| mitigation, selected by a known flag | n39 | weights_n19_fitted | 54 | 25 | 0 | 25 | 0 of 0 | caco2, herg_ki, logd, BH, colloidal |
| mitigation, selected by a known flag | n39 | weights_within_var | 54 | 22 | 0 | 22 | 0 of 0 | caco2, herg_ki, logd, BH, colloidal |
| mitigation, selected by a known flag | n19 | count_weighted | 4 | 2 | 2 | 0 | 0 of 0 | herg_ic50 |
| mitigation, selected by a known flag | n19 | spread_pooled_weighted | 4 | 2 | 2 | 0 | 0 of 0 | herg_ic50 |
| mitigation, selected by a known flag | n19 | shrunk | 4 | 2 | 2 | 0 | 0 of 0 | herg_ic50 |
| mitigation, every molecule | n33 | huber_teacher_blend | 32 | 19 | 2 | 17 | 0 of 0 | caco2, herg_ki, logd, BH |

## Rows dropped

- Rows on datasets outside the 13: 126 (humanness_therapeutic 126).
- Rows on the 13 datasets for methods not in scope: 604: n02 cliff-aware weights 22; n07 gp_smooth 18; n07 knn_smooth 78; n11 best repair 10; n18 gp_smooth 10; n18 knn_smooth 18; n19 calibrated 4; n19 invvar_and_count_feature 4; n24 knn_smooth 34; n24 oof_smooth 34; n27 toward_prediction 10; n30 winsor 32; n31 bayes_shrink 30; n31 remove 30; n31 rewrite 30; n32 itlm 48; n32 shrink, one weight 48; n32 shrink, weight per molecule 48; n33 c_mixup 32; n33 target_noise_bagging 32; n33 yeo_johnson 32.
- In-scope methods with no rows on the page: none.
- In-scope rows with no score (counted in the "no score" column, not in yes/neither): 42.

## Method note

1. Built from the raw result files, not the page. meeting_methods.py and meeting_readable.py were copied to answers/_scripts/ in this scratch folder and run there against /Volumes/seagate/kirby_results_2026-10-09_noise_view, so they wrote only here.
2. They ran under guard_run.py, which hides every file whose name contains tcga, annitia, nhanes or pam50 and refuses to open one. Refused opens: 0.
3. On the 13 datasets the rebuilt page and the seagate MEETING_METHODS.md give 3842 rows each, with no differences in any score, verdict or control. The two pages differ only on cut datasets (cdr_flexibility is now in the repo's cut list) and patient data.
4. The rows were then read off the rebuilt page by parse.py (adapted from scratchpad/thesis_ch5_8/ch7m/parse.py) and counted by build.py.
5. The "yes", "single only", "hybrid only" and "no" labels, and "hybrid better after", are the page's own verdicts. They are not recomputed. Where a table has a control column, the page's "yes" already requires beating the control.
6. Hybrid against single before the method (table C) is computed here from the 3-decimal scores the page prints, so near-equal pairs count as ties. "After" uses the page's 4-decimal column.
7. Fold 0 only. Each method's best setting is picked on the held-out score, so every count leans optimistic.
8. The n09 rows include 416 rows from a second run of the same methods inside n13, marked "(n13 run)" on the page. They are counted as separate rows, as the signed-off counts did.
9. For n30 Constant lift and Censored EM, 10 rows each on labels as delivered have no score because the arm is not in the file. QM9 has no as-delivered rows for n40 or n20.
10. Out of scope though not on the CUT list, so dropped: n24 oof_smooth (34 rows) and n31 remove (30 rows). Neither appears in the signed-off block. tuned_clean_reused is the clean-tuned control in n40, and it is counted as a method because the brief lists it.
