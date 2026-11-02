# Decimal places in the thesis: the plan

Written 2026-10-10 by Claude for the style pass. Not applied. It is applied after the session fixing
results has finished, because that session replaces numbers by exact match.

Line numbers are those of thesis.tex as read on 2026-10-10 and will drift. Find each sentence by its
text. If a number in a quoted sentence has changed since, apply the rule to the new number, not this
list. main.tex (title page to abbreviations) has no scores, so nothing changes there.

## 1. The rule (copied from DECIMALS_RULE.md, section 1)

1. Scores and correlations (RMSE, MAE, Spearman, Pearson, AUROC, MCC, R^2, normalized AUC):
   three decimals. RMSE and MAE in label units (log units, pK_i, kcal/mol) also take three.
2. A difference between two such scores takes three decimals, like the scores.
3. Percentages take one decimal. RMSE in percentage points of yield (Buchwald--Hartwig) is a
   percentage: one decimal (20.3381 becomes 20.3).
4. Tie rule. If two compared values print the same at the rule's precision, keep one more
   decimal for that pair only. A difference smaller than 0.010 also keeps four decimals.
5. Literature values keep their source's precision. Counts are integers. Settings, noise levels,
   shares, rates, thresholds and hyperparameters are written as set (0.2, 0.05, 1.0, 10%).
   Constants are written as defined (1.4826). Exact values stay exact (p = 0.0625 = 1/16).
6. Never pad. A value printed with fewer decimals than the rule stays as it is until its raw
   value is read from a result file. Adding a decimal from memory would invent a number.
7. Never round twice. A four-decimal value whose fourth decimal is 5 (0.8375) cannot be rounded
   from the text: the raw value could give 0.837 or 0.838. Take the third decimal from the raw
   result (for Chapter 7, from `ch7_numbers.py`; for tables, from the CSV next to the table in
   `figures_ch5_7/`, read only). If no raw value is at hand, keep four decimals and list it.
8. Inline tables in thesis.tex follow the same rule as prose. Tables `\input` from
   `figures_ch5_7/` are not edited (section 4).

## 2. Counts

| Chapter | Rounded | Kept at four, rule 7 (raw value not found) | Kept, other rule | Tie-rule cases |
|---|---|---|---|---|
| 1 Introduction | 0 | 0 | 0 | 0 |
| 2 Background | 0 | 0 | 0 | 0 |
| 3 Noise robustness | 0 | 0 | 1 | 0 |
| 4 Similarity assumption | 0 | 0 | 0 | 0 |
| 5 Building hybrids | 99 | 7 | 2 | 6 |
| 6 Similarity in hybrids | 67 | 8 | 1 | 0 |
| 7 Label noise | 79 | 0 | 3 | 0 |
| 8 Conclusions | 7 | 0 | 0 | 0 |
| Total | 252 | 15 | 7 | 6 |

"Rounded" counts each printed value once per place it appears (a value repeated in two sentences
counts twice). It includes 13 rule-7 values whose third decimal was taken from a raw value found in
a result CSV or in `tests/analysis/ch7_numbers_expected.json` (marked "rule 7 resolved" below; two
of them are marked "probable" and need checking). Raw values come from the runs read on 2026-10-09
and 2026-10-10; if the results session changes a number, read its raw value again.

Chapters 1, 2 and 4 change nothing. Their scores are at two or three decimals already, and rule 6
keeps two-decimal values as they are (Chapter 4: 50 two-decimal values in prose and 102 in its
inline tables, settings included; DECIMALS_RULE.md section 3 lists the lines). Chapter 3 already
follows the rule; its one four-decimal value is p = 0.0625. The same holds for two-decimal scores in
Chapters 5 to 7: they stay until their raw values are read.

## 3. Tie-rule cases

1. Chapter 5, line 2746, table `tab:tuning_base_reps`, row "Buchwald--Hartwig & LightGBM & Avalon
   (difference) & 20.22 & 20.21". Yield RMSE in percentage points takes one decimal (rule 3); both
   print 20.2. Keep `20.22` and `20.21`. The other three Buchwald--Hartwig rows round to one decimal.
2. Chapter 5, line 2728, same table, "QM9 & DNN & Mol2vec & 0.4867 & 0.4867", and line 2734,
   "hERG $K_i$ & DNN & GROVER & 0.5891 & 0.5891". Default and tuned print the same at four decimals
   too. Read literally, the tie rule keeps four (`0.4867`, `0.5891`), which is the text as it is.
   Rounding both to `0.487` and `0.589` reads the same way: tuning did not change the score.
   This is the author's choice; the plan keeps four.
3. Differences below 0.010 keep four decimals (rule 4):
   - line 2382 (Chapter 5), "the best configuration it removed was 0.0051 to 0.0154 p$K_i$ behind the
     best kept": `0.0051` stays, `0.0154` -> `0.015`. The range then reads "0.0051 to 0.015".
     Keeping both ends at four is the alternative.
   - line 3079 (Chapter 5), "within 0.0020 RMSE": stays.
   - line 3556 (Chapter 6), "The winning margin was at most 0.0015 RMSE": stays.
4. A scan of every sentence and inline table row found no other pair of different values that
   print the same at three decimals.

## 4. Pairs left at mixed precision (rule 7, no raw value found)

In these sentences or rows one value is kept at four decimals and the value it is compared with
becomes three. Read the kept value's raw result before applying, so that both print at three.
- Chapter 5: line 2238 (`0.5475` to `0.629`); line 2398 (`0.547` against `0.5545`); line 2742
  (`0.4945` against `0.474`); line 3064 (`0.670` to `0.5735`); line 3278 (`0.611` to `0.5715`);
  line 3287 (`0.589` to `0.5665`); line 3315 (`0.5085` against `0.390`).
- Chapter 6: line 3704 (`0.0155`, alone); line 3766 (`0.2315` to `0.1535`, both kept); line 3816
  (`0.582` to `0.7115`); line 3843 (`0.3225` in its row); line 3846 (`0.6065` and `0.5765` in their
  row); line 3952 (`0.136` to `0.0295`).
  The weighted columns of `tab:importance_weighting` are not in any CSV in `figures_ch5_7/`; the
  unweighted columns match `tab_nearest_neighbour_prediction.csv`.

## 5. Every value, by chapter

Format: the sentence or table row as it stands, then old -> new for each value in it.

### Chapter 3

Rounded: 0. Kept at four by rule 7 (no raw value found): 0. Kept for another rule (exact value, constant, difference below 0.010): 1. Tie-rule cases: 0.

- line 850 (prose): "smallest attainable $p$ is 0.0625."
  - `0.0625` stays (exact value (p = 1/16), rule 5)

### Chapter 5

Rounded: 99. Kept at four by rule 7 (no raw value found): 7. Kept for another rule (exact value, constant, difference below 0.010): 2. Tie-rule cases: 6.

- line 2152 (prose): "It was also less accurate: in preliminary runs on Jain antibody retention by the neonatal Fc receptor (FcRn) it placed 16th of 17 allocation methods, with a Spearman correlation of 0.3600 against 0.4526 for linear allocation."
  - `0.3600` -> `0.360`; `0.4526` -> `0.453`
- line 2238 (prose): "For example, on hERG $K_i$ with linear allocation at 300 features, five ways of rescaling each representation's score gave RMSE values from 0.5870 to 0.5924 (p$K_i$)."
  - `0.5870` -> `0.587`; `0.5924` -> `0.592`
- line 2238 (prose): "A single rescaling ranged from 0.5475 to 0.6294 across 5 folds, so the rescaling is one fixed rule per metric."
  - `0.5475` stays (rule 7 (fourth decimal is 5; take third from raw value)); `0.6294` -> `0.629`
- line 2382 (prose): "The best had an RMSE of 0.5472 p$K_i$ (SVM, 1,000 features, LightGBM gain, softmax allocation, random forest evaluator, all representations)."
  - `0.5472` -> `0.547`
- line 2382 (prose): "Its standard deviation over the five folds was 0.0443 p$K_i$."
  - `0.0443` -> `0.044`
- line 2382 (prose): "For each of these four cuts, the best configuration it removed was 0.0051 to 0.0154 p$K_i$ behind the best kept, less than the spread over folds."
  - `0.0051` stays (difference below 0.010 (rule 4)); `0.0154` -> `0.015`
- line 2393 (table row): "Full search & 8{,}220 & 100.0 & 0.5472 & -- \\"
  - `0.5472` -> `0.547`
- line 2394 (table row): "Downstream models: SVM and DNN only & 3{,}360 & 28.7 & 0.5472 & 0.5707 \\"
  - `0.5472` -> `0.547`; `0.5707` -> `0.571`
- line 2395 (table row): "Budgets: 250 to 2{,}000 features & 1{,}680 & 20.0 & 0.5472 & 0.5626 \\"
  - `0.5472` -> `0.547`; `0.5626` -> `0.563`
- line 2396 (table row): "Importance: DeepLIFT removed & 1{,}120 & 16.3 & 0.5472 & 0.5523 \\"
  - `0.5472` -> `0.547`; `0.5523` -> `0.552`
- line 2397 (table row): "Allocation: global pool removed & 896 & 4.6 & 0.5472 & 0.5536 \\"
  - `0.5472` -> `0.547`; `0.5536` -> `0.554`
- line 2398 (table row): "Representation count: all only & 224 & 1.4 & 0.5472 & 0.5545 \\"
  - `0.5472` -> `0.547`; `0.5545` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 2717 (prose): "The SVM gained most: on LogD with MHG-GNN, RMSE fell from 0.4751 to 0.4206, and on hERG $K_i$ with ECFP4 from 0.6748 to 0.6399."
  - `0.4751` -> `0.475`; `0.4206` -> `0.421`; `0.6748` -> `0.675`; `0.6399` -> `0.640`
- line 2717 (prose): "Tuning made the random forest worse on Buchwald--Hartwig yield, from 18.04 to 20.14 with the CGR fingerprint."
  - `18.04` -> `18.0`; `20.14` -> `20.1`
- line 2717 (prose): "At default settings, two repeat fits of the DNN on QM9 with Mol2vec scored 0.4867 and 0.4949."
  - `0.4867` -> `0.487`; `0.4949` -> `0.495`
- line 2727 (table row): "QM9 & DNN & Mordred & 0.3999 & 0.3888 \\"
  - `0.3999` -> `0.400`; `0.3888` -> `0.389`
- line 2728 (table row): "QM9 & DNN & Mol2vec & 0.4867 & 0.4867 \\"
  - `0.4867` stays (tie rule, section 3); `0.4867` stays (tie rule, section 3)
- line 2729 (table row): "QM9 & LightGBM & Mordred & 0.4118 & 0.4029 \\"
  - `0.4118` -> `0.412`; `0.4029` -> `0.403`
- line 2730 (table row): "QM9 & LightGBM & Avalon & 0.4594 & 0.4550 \\"
  - `0.4594` -> `0.459`; `0.4550` -> `0.455`
- line 2731 (table row): "hERG $K_i$ & SVM & MHG-GNN & 0.6400 & 0.6184 \\"
  - `0.6400` -> `0.640`; `0.6184` -> `0.618`
- line 2732 (table row): "hERG $K_i$ & SVM & ECFP4 & 0.6748 & 0.6399 \\"
  - `0.6748` -> `0.675`; `0.6399` -> `0.640`
- line 2733 (table row): "hERG $K_i$ & DNN & MHG-GNN & 0.6383 & 0.6279 \\"
  - `0.6383` -> `0.638`; `0.6279` -> `0.628`
- line 2734 (table row): "hERG $K_i$ & DNN & GROVER & 0.5891 & 0.5891 \\"
  - `0.5891` stays (tie rule, section 3); `0.5891` stays (tie rule, section 3)
- line 2735 (table row): "Caco-2 efflux & DNN & MHG-GNN & 0.2759 & 0.2820 \\"
  - `0.2759` -> `0.276`; `0.2820` -> `0.282`
- line 2736 (table row): "Caco-2 efflux & DNN & GROVER & 0.2657 & 0.2610 \\"
  - `0.2657` -> `0.266`; `0.2610` -> `0.261`
- line 2737 (table row): "Caco-2 efflux & SVM & PDV & 0.2761 & 0.2666 \\"
  - `0.2761` -> `0.276`; `0.2666` -> `0.267`
- line 2738 (table row): "Caco-2 efflux & SVM & MoLFormer & 0.2870 & 0.2819 \\"
  - `0.2870` -> `0.287`; `0.2819` -> `0.282`
- line 2739 (table row): "LogD & DNN & MHG-GNN & 0.4313 & 0.4277 \\"
  - `0.4313` -> `0.431`; `0.4277` -> `0.428`
- line 2740 (table row): "LogD & DNN & Mordred & 0.4167 & 0.4088 \\"
  - `0.4167` -> `0.417`; `0.4088` -> `0.409`
- line 2741 (table row): "LogD & SVM & MHG-GNN & 0.4751 & 0.4206 \\"
  - `0.4751` -> `0.475`; `0.4206` -> `0.421`
- line 2742 (table row): "LogD & SVM & MoLFormer & 0.4945 & 0.4737 \\"
  - `0.4945` stays (rule 7 (fourth decimal is 5; take third from raw value)); `0.4737` -> `0.474`
- line 2743 (table row): "Buchwald--Hartwig & Random forest & Avalon (difference) & 18.88 & 19.86 \\"
  - `18.88` -> `18.9`; `19.86` -> `19.9`
- line 2744 (table row): "Buchwald--Hartwig & Random forest & CGR fingerprint & 18.04 & 20.14 \\"
  - `18.04` -> `18.0`; `20.14` -> `20.1`
- line 2745 (table row): "Buchwald--Hartwig & LightGBM & CGR fingerprint & 19.49 & 19.27 \\"
  - `19.49` -> `19.5`; `19.27` -> `19.3`
- line 2746 (table row): "Buchwald--Hartwig & LightGBM & Avalon (difference) & 20.22 & 20.21 \\"
  - `20.22` stays (tie rule, section 3); `20.21` stays (tie rule, section 3)
- line 2963 (prose): "On Caco-2 with the DNN the best single representation in hindsight is PDV, at 0.2980 RMSE, against 0.2775 for the hybrid."
  - `0.2980` -> `0.298`; `0.2775` -> `0.277` (rule 7 resolved: raw 0.277465, figures_ch5_7/tab_hybrid_vs_best_single.csv, Caco-2 NN hybrid1 (hybrid_score))
- line 3042 (prose): "Yet the DNN ensemble reaches 0.7854 Spearman against 0.7348 for the hybrid, ahead on every fold."
  - `0.7854` -> `0.785`; `0.7348` -> `0.735`
- line 3051 (prose): "On hERG $K_i$ with the SVM at 250 features, the top representation alone gives 0.5990 RMSE and the search's hybrid 0.5642."
  - `0.5990` -> `0.599`; `0.5642` -> `0.564`
- line 3064 (prose): "On the rare-scaffold half of LogD the hybrid goes from 0.6698 to 0.5735 RMSE with the DNN, a larger gain than on the common half."
  - `0.6698` -> `0.670`; `0.5735` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3079 (prose): "On LogD this is 0.4302 against 0.4436 RMSE."
  - `0.4302` -> `0.430`; `0.4436` -> `0.444`
- line 3079 (prose): "With the DNN, PCA is ahead on LogD on all 5 folds, 0.3920 against 0.4089 RMSE."
  - `0.3920` -> `0.392`; `0.4089` -> `0.409`
- line 3079 (prose): "On Caco-2 the DNN hybrid is ahead on 2 of 5 folds, 0.2775 against 0.2733 RMSE for PCA."
  - `0.2775` -> `0.277` (rule 7 resolved: probable: same c02 Caco-2 DNN hybrid score as line 2963 (raw 0.277465); check the SOURCE line before applying); `0.2733` -> `0.273`
- line 3079 (prose): "On hERG $K_i$, MDR1--MDCK efflux and rat microsomal clearance the DNN hybrid and PCA are within 0.0020 RMSE."
  - `0.0020` stays (difference below 0.010 (rule 4))
- line 3079 (prose): "There the DNN hybrid is ahead on every fold, 0.4088 against 0.4767 RMSE on LogD at 1,257 features."
  - `0.4088` -> `0.409`; `0.4767` -> `0.477`
- line 3079 (prose): "On the anti-VEGF expression scan PCA is ahead on 4 of 5 folds with the SVM, 0.7781 against 0.7729 Spearman, and on 3 of 5 with the DNN."
  - `0.7781` -> `0.778`; `0.7729` -> `0.773`
- line 3079 (prose): "With the DNN on binding free energy the hybrid is ahead on 4 of 5 folds, 0.3542 against 0.2392 Spearman."
  - `0.3542` -> `0.354`; `0.2392` -> `0.239`
- line 3091 (prose): "On the small antibody datasets the search promises much more than it delivers, by up to 0.1571 Spearman on colloidal stability with the DNN."
  - `0.1571` -> `0.157`
- line 3124 (prose): "On LogD with the SVM at 1,000 features, for example, it reaches 0.4303 RMSE against 0.4452 for recursive elimination, and random features are further behind."
  - `0.4303` -> `0.430`; `0.4452` -> `0.445`
- line 3167 (prose): "On EGFR binding the matched evaluator gives 0.3527 Spearman with the SVM against 0.2993 for a LightGBM evaluator."
  - `0.3527` -> `0.353`; `0.2993` -> `0.299`
- line 3180 (prose): "Feature scaling takes colloidal stability with the DNN from 0.2607 to 0.3571 Spearman, and keeping the above-median features also helps binding free energy with the SVM."
  - `0.2607` -> `0.261`; `0.3571` -> `0.357`
- line 3186 (prose): "It does so on all four small-molecule datasets with a result, for example 0.4114 against 0.4299 RMSE on LogD."
  - `0.4114` -> `0.411`; `0.4299` -> `0.430`
- line 3262 (prose): "With the DNN on factor Xa it goes from 0.7367 to 0.6612 MAE in log units of activity, lower on all five folds."
  - `0.7367` -> `0.737`; `0.6612` -> `0.661`
- line 3278 (prose): "Adding features from ECFP4 and MHG-GNN takes it from 0.6109 to 0.5715 RMSE with the SVM, lower on all five folds, and no two molecules share a row."
  - `0.6109` -> `0.611`; `0.5715` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3287 (prose): "They beat hERG $K_i$'s best single representation with both models (SVM 0.5888 to 0.5665 RMSE, on all five folds)."
  - `0.5888` -> `0.589`; `0.5665` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3299 (prose): "On LogD with the SVM it reaches 0.4382 RMSE, against 0.4598 for the best family, and the best single representation is further behind."
  - `0.4382` -> `0.438`; `0.4598` -> `0.460`
- line 3307 (prose): "The exception is Caco-2 with the DNN, where the ensembles reach 0.2721 and 0.2730 RMSE against 0.2837 for the hybrid."
  - `0.2721` -> `0.272`; `0.2730` -> `0.273`; `0.2837` -> `0.284`
- line 3315 (prose): "With the DNN and no flips it reaches an MCC of 0.5085 against 0.3904 for the better ensemble, and it stays ahead with 30\% flipped."
  - `0.5085` stays (rule 7 (fourth decimal is 5; take third from raw value)); `0.3904` -> `0.390`
- line 3325 (prose): "This matches one global selection (0.5289 against 0.5316 MCC with the DNN), while fitting a separate model per cluster is worse."
  - `0.5289` -> `0.529`; `0.5316` -> `0.532`

### Chapter 6

Rounded: 67. Kept at four by rule 7 (no raw value found): 8. Kept for another rule (exact value, constant, difference below 0.010): 1. Tie-rule cases: 0.

- line 3556 (prose): "The winning margin was at most 0.0015 RMSE, against a spread across folds of 0.02 to 0.12, so we sweep both."
  - `0.0015` stays (difference below 0.010 (rule 4))
- line 3696 (prose): "On the small-molecule datasets, even the best single representation and distance reach at most 0.2789 Spearman, on QM9."
  - `0.2789` -> `0.279`
- line 3704 (prose): "Its best single representation reaches only 0.0155 Spearman, and about half of its combinations of representation and distance are at or below zero."
  - `0.0155` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3737 (prose): "On CHO polyreactivity all four predictive hybrids are lower, for example 0.1056 to 0.0644 Spearman for the network's first hybrid."
  - `0.1056` -> `0.106`; `0.0644` -> `0.064`
- line 3749 (prose): "On LogD, the single representation with weighting reaches 0.2762 Spearman, which is above the unweighted hybrid (Table~\ref{tab:distance_tracks_activity})."
  - `0.2762` -> `0.276`
- line 3766 (prose): "On LogD the network's hybrid predicts better at 2,000 features than at 500, while distance in the same hybrid falls from 0.2315 to 0.1535 Spearman."
  - `0.2315` stays (rule 7 (fourth decimal is 5; take third from raw value)); `0.1535` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3797 (prose): "Picking the setting again for each fold does not change this, and the hybrid still wins on all five small-molecule datasets, for example on LogD from 0.6384 to 0.5648 RMSE."
  - `0.6384` -> `0.638`; `0.5648` -> `0.565`
- line 3805 (prose): "The largest gain that holds in both is on SMAC colloidal stability, from 0.0918 to 0.2906 Spearman with the setting picked again."
  - `0.0918` -> `0.092`; `0.2906` -> `0.291`
- line 3816 (prose): "On hERG $K_i$ the Gaussian process goes from 0.5819 to 0.7115 RMSE, and the support vector regressor also loses there, while on LogD the regressor barely moves."
  - `0.5819` -> `0.582`; `0.7115` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3826 (prose): "With 10 neighbors, weighting by LightGBM gain improves all four regression small-molecule datasets, for example QM9 from 0.5749 to 0.4841 RMSE."
  - `0.5749` -> `0.575`; `0.4841` -> `0.484`
- line 3843 (table row): "Caco-2 & RMSE $\downarrow$ & 0.3386 & 0.3337 & 0.3246 & 0.3225 \\"
  - `0.3386` -> `0.339`; `0.3337` -> `0.334`; `0.3246` -> `0.325`; `0.3225` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3844 (table row): "hERG blocking & AUROC $\uparrow$ & 0.8143 & 0.8229 & 0.8335 & 0.8366 \\"
  - `0.8143` -> `0.814`; `0.8229` -> `0.823`; `0.8335` -> `0.834` (rule 7 resolved: raw 0.833531, figures_ch5_7/tab_nearest_neighbour_prediction.csv, hERG blocking hybrid_value); `0.8366` -> `0.837`
- line 3845 (table row): "hERG $K_i$ & RMSE $\downarrow$ & 0.6280 & 0.5967 & 0.5822 & 0.5804 \\"
  - `0.6280` -> `0.628`; `0.5967` -> `0.597`; `0.5822` -> `0.582`; `0.5804` -> `0.580`
- line 3846 (table row): "LogD & RMSE $\downarrow$ & 0.6766 & 0.6065 & 0.5831 & 0.5765 \\"
  - `0.6766` -> `0.677`; `0.6065` stays (rule 7 (fourth decimal is 5; take third from raw value)); `0.5831` -> `0.583`; `0.5765` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3848 (table row): "EGFR binding & Spearman $\uparrow$ & 0.1760 & 0.2072 & 0.2126 & 0.2228 \\"
  - `0.1760` -> `0.176`; `0.2072` -> `0.207`; `0.2126` -> `0.213`; `0.2228` -> `0.223`
- line 3849 (table row): "Nanobody thermostability & Spearman $\uparrow$ & 0.5005 & 0.5329 & 0.5458 & 0.5520 \\"
  - `0.5005` -> `0.500` (rule 7 resolved: raw 0.500464, figures_ch5_7/tab_nearest_neighbour_prediction.csv, nanobody single_value); `0.5329` -> `0.533`; `0.5458` -> `0.546`; `0.5520` -> `0.552`
- line 3850 (table row): "AC-SINS self-association & Spearman $\uparrow$ & 0.3912 & 0.4163 & 0.3884 & 0.4217 \\"
  - `0.3912` -> `0.391`; `0.4163` -> `0.416`; `0.3884` -> `0.388`; `0.4217` -> `0.422`
- line 3851 (table row): "SMAC colloidal stability & Spearman $\uparrow$ & 0.1268 & 0.1706 & 0.2418 & 0.2339 \\"
  - `0.1268` -> `0.127`; `0.1706` -> `0.171`; `0.2418` -> `0.242`; `0.2339` -> `0.234`
- line 3852 (table row): "CHO polyreactivity & Spearman $\uparrow$ & 0.2059 & 0.2713 & 0.3591 & 0.3574 \\"
  - `0.2059` -> `0.206`; `0.2713` -> `0.271`; `0.3591` -> `0.359`; `0.3574` -> `0.357`
- line 3853 (table row): "Expression titer & Spearman $\uparrow$ & 0.2521 & \textbf{0.2180} & 0.2622 & \textbf{0.2290} \\"
  - `0.2521` -> `0.252`; `0.2180` -> `0.218`; `0.2622` -> `0.262`; `0.2290` -> `0.229`
- line 3854 (table row): "Binding free energy & Spearman $\uparrow$ & 0.1424 & \textbf{0.1243} & 0.0899 & 0.1353 \\"
  - `0.1424` -> `0.142`; `0.1243` -> `0.124`; `0.0899` -> `0.090`; `0.1353` -> `0.135`
- line 3856 (table row): "Buchwald--Hartwig yield & RMSE $\downarrow$ & 14.4572 & 14.2584 & 16.1811 & 16.5264 \\"
  - `14.4572` -> `14.5`; `14.2584` -> `14.3`; `16.1811` -> `16.2`; `16.5264` -> `16.5`
- line 3918 (prose): "On Caco-2, for example, the network goes from 0.3396 to 0.2881 RMSE on cliff molecules."
  - `0.3396` -> `0.340`; `0.2881` -> `0.288`
- line 3927 (prose): "On cliff molecules only two folds hold the 10 a score needs, and there the hybrid is worse for both models, for the network from 0.2422 to 0.2798 RMSE."
  - `0.2422` -> `0.242`; `0.2798` -> `0.280`
- line 3943 (prose): "The largest change is on hERG $K_i$, from 0.1996 to 0.2761 Spearman with the support vector machine's errors, while on LogD it is small."
  - `0.1996` -> `0.200`; `0.2761` -> `0.276`
- line 3952 (prose): "It tracks error worse for both on SMAC colloidal stability, where the network goes from 0.1360 to 0.0295 Spearman, and on Buchwald--Hartwig."
  - `0.1360` -> `0.136`; `0.0295` stays (rule 7 (fourth decimal is 5; take third from raw value))
- line 3960 (prose): "With the setting picked again for each fold, QM9's label gap falls from 0.6008 to 0.3928."
  - `0.6008` -> `0.601`; `0.3928` -> `0.393`
- line 3968 (prose): "On binding free energy the label gap rises from 1.4386 to 1.5072, and Buchwald--Hartwig is worse too."
  - `1.4386` -> `1.439`; `1.5072` -> `1.507`

### Chapter 7

Rounded: 79. Kept at four by rule 7 (no raw value found): 0. Kept for another rule (exact value, constant, difference below 0.010): 3. Tie-rule cases: 0.

- line 4198 (prose): "\caption[Noise mitigation methods.]{Noise mitigation methods, by what selects the molecules. $\tilde s_i$ is the suspicion score rescaled to $[0, 1]$, $\alpha$ is the strength and $\hat\sigma$ is 1.4826 times the median suspicion score."
  - `1.4826` stays (constant, rule 5)
- line 4255 (prose): "Each label moves halfway toward the out-of-fold prediction of a LightGBM model fitted with a Huber loss \citep{Huber1964}, whose threshold is 1.4826 times the median absolute out-of-fold residual."
  - `1.4826` stays (constant, rule 5)
- line 4301 (prose): "The network is also trained with a Huber loss \citep{Huber1964}, whose threshold is $c$ times 1.4826 times the median absolute out-of-fold residual, with $c$ of 0.5, 1, 1.5 or 2."
  - `1.4826` stays (constant, rule 5)
- line 4359 (prose): "On QM9 the network's second hybrid scores between 0.3484 and 0.3618 RMSE on the same fold and the same labels, depending on which experiment fitted it."
  - `0.3484` -> `0.348`; `0.3618` -> `0.362`
- line 4395 (prose): "With the neighbor-label gap as the detector, deleting takes the hybrid to 0.8375 AUROC against 0.8179 for the best of five random deletions of the same size."
  - `0.8375` -> `0.838` (rule 7 resolved: raw 0.837546, ch7_numbers_expected.json N7-03); `0.8179` -> `0.818`
- line 4406 (prose): "For the hybrid, deleting helped most at 10\% flagged, reaching 0.8375 AUROC under both flip kinds."
  - `0.8375` -> `0.838` (rule 7 resolved: raw 0.837546, ch7_numbers_expected.json N7-03)
- line 4406 (prose): "Under the confusion-matrix flips, the hybrid started below the single representation, 0.8028 against 0.8060 AUROC."
  - `0.8028` -> `0.803`; `0.8060` -> `0.806`
- line 4406 (prose): "At 40\% clipped, censored EM took the hybrid from 0.3653 to 0.3171 RMSE, on one fold."
  - `0.3653` -> `0.365`; `0.3171` -> `0.317`
- line 4444 (prose): "This takes the network on the first single representation from 0.6887 to 0.6419 RMSE, and improves the hybrid by a similar amount."
  - `0.6887` -> `0.689`; `0.6419` -> `0.642`
- line 4454 (prose): "With no noise added, neighbor weights took the network's second hybrid from 0.3618 to 0.3476 RMSE at their best setting, and detector weights did about as well."
  - `0.3618` -> `0.362`; `0.3476` -> `0.348`
- line 4506 (prose): "On QM9 with no noise added, the best deletion setting took the network's hybrid from 0.3550 to 0.3931 RMSE."
  - `0.3550` -> `0.355`; `0.3931` -> `0.393`
- line 4518 (prose): "Label smoothing with the neighbor gap takes the network on the first single representation from 0.6380 to 0.5775 RMSE, while every shuffled run makes it slightly worse."
  - `0.6380` -> `0.638`; `0.5775` -> `0.578` (rule 7 resolved: raw 0.577530, ch7_numbers_expected.json N7-09)
- line 4539 (prose): "On the hybrid it scores 18.0417 RMSE in percentage points of yield, against 19.8986 for the best random deletion."
  - `18.0417` -> `18.0`; `19.8986` -> `19.9`
- line 4548 (prose): "Putting the cliff molecules back takes it to 0.5786 RMSE, against 0.5949 with random molecules put back."
  - `0.5786` -> `0.579`; `0.5949` -> `0.595`
- line 4567 (prose): "On Caco-2 under outlier noise its average precision is 0.3381, against a chance level of 0.0512."
  - `0.3381` -> `0.338`; `0.0512` -> `0.051` (chance level of average precision (the share of damaged molecules, computed, not set); rounded as a score here, author may prefer to keep it)
- line 4575 (prose): "Under the shift in 10\%, its median neighbor setting reaches 0.2686 on Caco-2 against 0.2023 for the residual, and every one of its settings is ahead."
  - `0.2686` -> `0.269`; `0.2023` -> `0.202`
- line 4594 (prose): "The largest gain was on Buchwald--Hartwig, 0.4647 to 0.5123 average precision, with smaller gains on LogD and SMAC colloidal stability."
  - `0.4647` -> `0.465`; `0.5123` -> `0.512`
- line 4604 (prose): "Under censoring the nearest-neighbor Shapley value reached 0.3906 on Buchwald--Hartwig against 0.1949 for the residual, and was also ahead on SMAC colloidal stability."
  - `0.3906` -> `0.391`; `0.1949` -> `0.195`
- line 4635 (prose): "Deleting raised the RMSE from 0.6275 to between 0.6913 and 0.7282, on 2 of the 3 noise draws, because the run has not finished."
  - `0.6275` -> `0.627` (rule 7 resolved: raw 0.627470, figures_ch5_7/f10_detection_vs_cleaning_herg_ki.csv no_cleaning_value); `0.6913` -> `0.691`; `0.7282` -> `0.728`
- line 4635 (prose): "Every share again raised the RMSE, from 0.2494 to between 0.2611 and 0.3009."
  - `0.2494` -> `0.249`; `0.2611` -> `0.261`; `0.3009` -> `0.301`
- line 4635 (prose): "There, under outliers, deleting at 5\% with a precision of 0.36 lowered the RMSE from 0.5874 to 0.5571, on 2 of 3 draws."
  - `0.5874` -> `0.587`; `0.5571` -> `0.557`
- line 4708 (prose): "The constant lift is best on hERG $K_i$, where the network on the second single representation scores 0.6662 RMSE after the constant lift and 0.7037 after censored EM."
  - `0.6662` -> `0.666`; `0.7037` -> `0.704`
- line 4726 (prose): "On LogD, the network on the hybrid scores 0.4988 RMSE with each molecule's own lift, against 0.4632 for the best of three shuffles."
  - `0.4988` -> `0.499`; `0.4632` -> `0.463`
- line 4735 (prose): "On binding free energy, the support vector regressor on the first single representation goes 0.1187 before clipping and 0.2480 after."
  - `0.1187` -> `0.119`; `0.2480` -> `0.248`
- line 4750 (prose): "On the first Caco-2 draw the random-intercept model took it from 0.3214 to 0.2796 RMSE."
  - `0.3214` -> `0.321`; `0.2796` -> `0.280`
- line 4759 (prose): "On Caco-2 the network on the hybrid went from 0.2333 to 0.2669 RMSE after subtracting family means, and LogD got worse too."
  - `0.2333` -> `0.233`; `0.2669` -> `0.267`
- line 4770 (prose): "On hERG $K_i$ the network on the hybrid scored 0.7423 RMSE with the plain mean of the measurements and 0.7485 after removing the offsets."
  - `0.7423` -> `0.742`; `0.7485` -> `0.748` (rule 7 resolved: raw 0.748456, ch7_numbers_expected.json N7-28 "offsets removed")
- line 4782 (prose): "Under Gaussian noise at level 1.0 the second hybrid went from 0.7184 to 0.6812 RMSE."
  - `0.7184` -> `0.718`; `0.6812` -> `0.681`
- line 4810 (prose): "Deleting the 20\% of antibodies the forest residual flags takes the support vector regressor on the first single representation from 0.2740 to 0.4034 Spearman."
  - `0.2740` -> `0.274`; `0.4034` -> `0.403`
- line 4821 (prose): "On binding free energy, label smoothing takes the network's first single representation from 0.3414 to 0.4840 Spearman, and also helps its hybrid."
  - `0.3414` -> `0.341`; `0.4840` -> `0.484`
- line 4830 (prose): "On SMAC, weighting by agreement with the neighbors took the network on the first single representation from 0.0651 to 0.2093 Spearman, and also helped it on the hybrid."
  - `0.0651` -> `0.065`; `0.2093` -> `0.209`
- line 4858 (prose): "Weighting each molecule by its number of measurements took the support vector regressor on the hybrid from 0.7598 to 0.7283 RMSE on the first draw."
  - `0.7598` -> `0.760`; `0.7283` -> `0.728`
- line 4878 (prose): "With the cleaning reaching both the search and the fit, the network on the hybrid goes from 0.1550 to 0.5619 Spearman."
  - `0.1550` -> `0.155`; `0.5619` -> `0.562`
- line 4887 (prose): "There the network on the second hybrid goes from $-0.0143$ to 0.5596 Spearman when the labels are shrunk toward an out-of-fold prediction for the search alone."
  - `-0.0143` -> `-0.014`; `0.5596` -> `0.560`
- line 4914 (prose): "On hERG $K_i$ with Gaussian noise at level 1.0, averaging ten network seeds takes the network on the second hybrid from 0.7184 to 0.6557 RMSE."
  - `0.7184` -> `0.718`; `0.6557` -> `0.656`
- line 4922 (prose): "Against the support vector regressor it therefore compares two models with the weaker one alone, for example 0.4751 to 0.4079 RMSE on LogD."
  - `0.4751` -> `0.475`; `0.4079` -> `0.408`
- line 4931 (prose): "On hERG $K_i$ in the example above, the setting tuned on damaged labels scores 0.6911 and the setting tuned on clean labels 0.7028."
  - `0.6911` -> `0.691`; `0.7028` -> `0.703`
- line 4939 (prose): "On Buchwald--Hartwig under the shift in 10\%, the random forest on the first single representation went from 20.3381 to 18.6007 RMSE in percentage points of yield."
  - `20.3381` -> `20.3`; `18.6007` -> `18.6`
- line 4993 (prose): "Here it is the only method that helps on binding free energy's labels as delivered, from 0.3414 to 0.4475 Spearman on the best single representation."
  - `0.3414` -> `0.341`; `0.4475` -> `0.448` (rule 7 resolved: raw 0.447538, ch7_numbers_expected.json N7-43 heteroscedastic_gp)
- line 5003 (prose): "DivideMix was the worst, 0.4517 RMSE on the hybrid against 0.3550 for the plain network, and distance-based selection the least bad."
  - `0.4517` -> `0.452`; `0.3550` -> `0.355`
- line 5020 (prose): "On LogD with outliers, for example, the network's cleaned hybrid scores 0.5560 against 0.5915 for the cleaned single representation."
  - `0.5560` -> `0.556`; `0.5915` -> `0.591` (rule 7 resolved: probable: raw 0.591462 in ch7_numbers_expected.json N7-07 (logd outlier dnn base_1 weighted), but the sentence is in block N7-46; confirm it is the same quantity)
- line 5026 (prose): "On LogD under a scaffold-family shift, the network's single representation goes from 0.7051 to 0.6626 RMSE with neighbors in the hybrid, and gets worse with neighbors in its own space."
  - `0.7051` -> `0.705`; `0.6626` -> `0.663`

### Chapter 8

Rounded: 7. Kept at four by rule 7 (no raw value found): 0. Kept for another rule (exact value, constant, difference below 0.010): 0. Tie-rule cases: 0.

- line 5109 (prose): "On LogD with the SVM, for example, it went from 0.4930 to 0.4302 RMSE."
  - `0.4930` -> `0.493`; `0.4302` -> `0.430`
- line 5112 (prose): "On hERG $K_i$, the Spearman correlation between the distance of two molecules and the difference in their activity went from 0.0729 to 0.2625."
  - `0.0729` -> `0.073`; `0.2625` -> `0.263` (rule 7 resolved: raw 0.262521, figures_ch5_7/tab_distance_tracks_activity.csv hERG Ki hybrid_value)
- line 5115 (prose): "With a quarter of LogD's training labels clipped, censored EM took the network on its best single representation from 0.6354 to 0.4795 RMSE, against 0.4314 before clipping."
  - `0.6354` -> `0.635`; `0.4795` -> `0.479` (rule 7 resolved: raw 0.479469, ch7_numbers_expected.json N7-20); `0.4314` -> `0.431`

## 6. What must change in tests/analysis/ch7_numbers.py

Apply in the same edit as the Chapter 7 and Chapter 8 rounding. Line numbers are those of the file
as read on 2026-10-10 (last commit 647101e).

1. Section anchor, already broken. `thesis_blocks()` (line 215) looks for
   `\section{Results: Does Noise Handling Work at All?}`. That heading no longer exists: it is now
   `\subsection{Overall effect of noise handling}\label{sec:noise:res:overall}` (thesis.tex line
   4348), under a new `\section{Results}`. So `tex.find` returns -1, `BLOCKS` is empty, and
   `BLOCKS.get(self.id, text)` in `check()` (line 267) returns the expected string itself. Every
   "TEXT EDITED?" test passes whatever the thesis says. Change the anchor to
   `\subsection{Overall effect of noise handling}` (once in thesis.tex; `\section{Results}` appears
   five times, and the label string twice, once in a `\ref`). The end anchor `\chapter{Discussion and Conclusions}` still exists (line 5074).
2. Block ends. Line 222 splits a block at `^\\(?:sub)?section\{`, which does not match the
   `\subsubsection` headings Chapter 7 now has. Blocks still end at the next `% SOURCE [N7-xx]:`
   line, so the effect is small; `(?:sub)*section` fixes it.
3. Retype each text string the prose rounds. The table below gives, for each rounded Chapter 7 and
   Chapter 8 value, the lines of ch7_numbers.py where the four-decimal string is typed. Some lines
   use the same string for a different sentence; retype only where the check names the rounded
   sentence. Line 263 is the docstring example ('0.6354', '-0.0143'); update it to three decimals.
4. Run the tool once with the old four-decimal strings, before retyping, and take the third decimal
   of `0.8375`, `0.5775`, `0.7485`, `0.4475`, `0.5915` and `0.4795` from the raw value it
   prints (DECIMALS_RULE.md section 5, step 2). The values in section 5 above came from
   `ch7_numbers_expected.json` and should agree. `0.6275` (line 4635) is not checked by the tool; its
   raw value came from `figures_ch5_7/f10_detection_vs_cleaning_herg_ki.csv`.
5. Comparisons at four decimals become three, so a claim that passes in the tool cannot read as a
   tie in the text: `round(v, 4)` at lines 699, 713, 717, 2987 and 3012 becomes `round(v, 3)`.
   Lines 3424, 3425 and 3442 (sets of distinct scores) and 2773 (`round(x, 6)`) test whether scores
   are identical and stay. Line 2822 already uses three. Lines 474 and 2558 are printouts.
6. Substring match. `check()` tests `text in block`, so `'0.838'` is found inside `'0.8381'`. Use
   a number-boundary match: `re.search(r'(?<![\d.])' + re.escape(t) + r'(?!\d)', block)`.
7. Docstring line 23: "copied from thesis.tex on 2026-10-09" becomes the date of the rounding.
8. Chapter 8 values typed for N7-20. Lines 2078-2079 and 3473-3475 type `'0.6354'`, `'0.4795'`,
   `'0.4314'`. Those numbers now appear only in Chapter 8 (thesis.tex line 5115), outside the part
   of thesis.tex the script reads. Once item 1 is fixed, these checks will print "TEXT EDITED?".
   Either retype them at three decimals and point them at Chapter 8, or put the numbers back in the
   N7-20 paragraph. The author decides.
9. Keep every `% SOURCE [N7-xx]:` line in thesis.tex. The `{x:.4f}` printouts in note and NEW FOLD
   lines are not checks and can stay.

| thesis.tex line | Old text | New text | ch7_numbers.py lines holding the old string |
|---|---|---|---|
| 4359 | `0.3484` | `0.348` | 2115, 3418, 3433 |
| 4359 | `0.3618` | `0.362` | 908, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3432 |
| 4395 | `0.8375` | `0.838` | 439, 641, 3477 |
| 4395 | `0.8179` | `0.818` | 440, 3478 |
| 4406 | `0.8375` | `0.838` | 439, 641, 3477 |
| 4406 | `0.8028` | `0.803` | 433, 441, 641, 902, 3476 |
| 4406 | `0.8060` | `0.806` | 436, 903 |
| 4444 | `0.6887` | `0.689` | 753 |
| 4444 | `0.6419` | `0.642` | 753, 1040, 3487 |
| 4454 | `0.3618` | `0.362` | 908, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3432 |
| 4454 | `0.3476` | `0.348` | 908 |
| 4506 | `0.3550` | `0.355` | 984, 3245 |
| 4506 | `0.3931` | `0.393` | 984 |
| 4518 | `0.6380` | `0.638` | 1040, 3479 |
| 4518 | `0.5775` | `0.578` | 1040, 3480 |
| 4539 | `18.0417` | `18.0` | 1199 |
| 4539 | `19.8986` | `19.9` | 1200 |
| 4548 | `0.5786` | `0.579` | 1234 |
| 4548 | `0.5949` | `0.595` | 1234 |
| 4567 | `0.3381` | `0.338` | 1388 |
| 4567 | `0.0512` | `0.051` | 1388, 1529 |
| 4575 | `0.2686` | `0.269` | 1429 |
| 4575 | `0.2023` | `0.202` | 1431 |
| 4594 | `0.4647` | `0.465` | 1628 |
| 4594 | `0.5123` | `0.512` | 1628 |
| 4604 | `0.3906` | `0.391` | 1740 |
| 4604 | `0.1949` | `0.195` | 1740 |
| 4708 | `0.6662` | `0.666` | 2146 |
| 4708 | `0.7037` | `0.704` | 2146 |
| 4726 | `0.4988` | `0.499` | 2195 |
| 4726 | `0.4632` | `0.463` | 2195 |
| 4735 | `0.1187` | `0.119` | 2217 |
| 4735 | `0.2480` | `0.248` | 2217 |
| 4750 | `0.3214` | `0.321` | 2238 |
| 4750 | `0.2796` | `0.280` | 2240 |
| 4759 | `0.2333` | `0.233` | 2284 |
| 4759 | `0.2669` | `0.267` | 2284 |
| 4770 | `0.7423` | `0.742` | 2307 |
| 4770 | `0.7485` | `0.748` | 2145, 2178, 2307 |
| 4782 | `0.7184` | `0.718` | 2340, 2941 |
| 4782 | `0.6812` | `0.681` | 2340 |
| 4810 | `0.2740` | `0.274` | 2087, 2499 |
| 4810 | `0.4034` | `0.403` | 2499 |
| 4821 | `0.3414` | `0.341` | 2534, 3207 |
| 4821 | `0.4840` | `0.484` | 2534 |
| 4830 | `0.0651` | `0.065` | 2501, 2567 |
| 4830 | `0.2093` | `0.209` | 2567 |
| 4858 | `0.7598` | `0.760` | 2603 |
| 4858 | `0.7283` | `0.728` | 2603 |
| 4878 | `0.1550` | `0.155` | 2720 |
| 4878 | `0.5619` | `0.562` | 2720 |
| 4887 | `-0.0143` | `-0.014` | 263, 2754 |
| 4887 | `0.5596` | `0.560` | 2755 |
| 4914 | `0.7184` | `0.718` | 2340, 2941 |
| 4914 | `0.6557` | `0.656` | 2941 |
| 4922 | `0.4751` | `0.475` | 2976 |
| 4922 | `0.4079` | `0.408` | 2977 |
| 4931 | `0.6911` | `0.691` | 2941, 3002 |
| 4931 | `0.7028` | `0.703` | 3003 |
| 4939 | `20.3381` | `20.3` | 3063 |
| 4939 | `18.6007` | `18.6` | 3064 |
| 4993 | `0.3414` | `0.341` | 2534, 3207 |
| 4993 | `0.4475` | `0.448` | 3208 |
| 5003 | `0.4517` | `0.452` | 3246, 3247 |
| 5003 | `0.3550` | `0.355` | 984, 3245 |
| 5020 | `0.5560` | `0.556` | 905 |
| 5020 | `0.5915` | `0.591` | 904 |
| 5026 | `0.7051` | `0.705` | 3364 |
| 5026 | `0.6626` | `0.663` | 3364 |
| 5115 | `0.6354` | `0.635` | 263, 2078, 3473 |
| 5115 | `0.4795` | `0.479` | 2079, 3474 |
| 5115 | `0.4314` | `0.431` | 2078, 3475 |

Chapter 7 values in section 5 that are not in this table are in paragraphs whose SOURCE line has
no N7 number (the newer F10 and F11 paragraphs, e.g. line 4635). ch7_numbers.py does not check them.

## 7. Tables written by thesis_figures.py

Unchanged from DECIMALS_RULE.md section 4: `tab_hybrid_vs_best_single.tex`,
`tab_distance_tracks_activity.tex` and `tab_nearest_neighbour_prediction.tex` print four decimals
from `_pair_cells`. Chapter 8 line 5109 (`0.4930`, `0.4302`) and line 5112 (`0.0729`, `0.2625`) quote
these tables. After the prose is rounded, the tables and the prose differ in precision until the
figures session changes `_pair_cells` and `_bold_rows`.
