# Decimal rule applied to thesis.tex (2026-10-10)

Rule: `DECIMALS_RULE.md`. The candidates were recomputed from the current thesis.tex; `DECIMALS_PLAN.md` was used only for raw values it had already found. Edits were made with the Edit tool, prose and hand-typed inline tables only. Comments, SOURCE lines, citations, labels, settings, counts and the tables read from `figures_ch5_7/` were not touched.

## Counts per chapter

One count is one printed value in one place. A value printed twice counts twice.

| Chapter | Rounded | Kept at four, rule 7 | Kept, tie rule | Kept, other rule |
|---|---|---|---|---|
| 1 Introduction | 0 | 0 | 0 | 0 |
| 2 Background | 0 | 0 | 0 | 0 |
| 3 Noise robustness | 0 | 0 | 0 | 1 (p = 0.0625, exact) |
| 4 Similarity assumption | 0 | 0 | 0 | 0 |
| 5 Building hybrids | 89 | 2 | 6 | 1 (0.0020, a difference below 0.010) |
| 6 Similarity in hybrids | 67 | 8 | 0 | 1 (0.0015, a difference below 0.010) |
| 7 Label noise | 70 | 0 | 0 | 3 (1.4826, a constant) |
| 8 Conclusions | 0 | 0 | 0 | 0 |
| Total | 226 | 10 | 6 | 6 |

Chapter 5: 41 in prose, 48 in the inline tables `tab:cut_walkthrough` and `tab:tuning_base_reps`.
Chapter 6: 22 in prose, 45 in `tab:importance_weighting`.
Chapter 7: 70 in prose, in 35 sentences.
Chapter 8 prose now holds no four-decimal value; its numbers are in comments.

Buchwald--Hartwig yield RMSE (percentage points) went to one decimal: Chapter 5 table (18.88 to 18.9, and five more), Chapter 6 table (14.4572 to 14.5, and three more), Chapter 7 (18.0417 to 18.0, 19.8986 to 19.9, 20.3381 to 20.3, 18.6007 to 18.6).

Two-decimal scores were left as they are (rule 6, never pad): Chapter 4's prose and tables, Chapter 3's -0.10, and Chapter 6's 1.63 and 1.68 nats.

## Rule 7 resolved from a raw value

The fourth decimal was 5, so the third was taken from the raw result.

- Ch5 "0.2980 against 0.2775" (Caco-2, DNN): 0.2775 to 0.277. Raw 0.277465, `figures_ch5_7/tab_hybrid_vs_best_single.csv`, c02 hybrid1 held-out mean.
- Ch5 "0.6698 to 0.5735" (rare-scaffold half of LogD, DNN): 0.5735 to 0.573. Raw 0.573496, c07 `logd.json` partitions.scaffold_frequency.hard, dnn hybrid1 mean.
- Ch5 "0.6109 to 0.5715" (graph kernel plus ECFP4 and MHG-GNN, SVM): 0.5715 to 0.571. Raw 0.571451, mean of slot_fold_scores, `opig_shared_vectors_remedy.json` row C2 hybrid1 svm.
- Ch5 "0.5888 to 0.5665" (LogD settings reused on hERG Ki, SVM): 0.5665 to 0.567. Raw 0.566531, `logd_to_herg_ki.json` per_model.svm.target.recipe_transfer.mean.
- Ch5 table `tab:tuning_base_reps`, LogD SVM MoLFormer default 0.4945 to 0.495. Raw 0.494544, n40 `logd__gaussian.json` per_level['0'].base_2.fixed.svm.
- Ch6 table `tab:importance_weighting`, hERG blocking distance hybrid unweighted 0.8335 to 0.834. Raw 0.833531, `tab_nearest_neighbour_prediction.csv` hybrid_value.
- Ch6 same table, nanobody single representation unweighted 0.5005 to 0.500. Raw 0.500465, same CSV, single_value.
- Ch7 0.8375 to 0.838 (raw 0.837546, N7-03); 0.5775 to 0.578 (raw 0.577530, N7-09); 0.7485 to 0.748 (raw 0.748456, N7-28 "offsets removed"); 0.4475 to 0.448 (raw 0.447538, N7-43).
- Ch7 "cleaned hybrid scores 0.5560 against 0.5915" (block N7-46): 0.5915 to 0.591. Raw 0.591462 from N7-07, label "logd n21 outlier dnn base_1 weighted"; its partner "hybrid weighted" is 0.555976, so both values of the sentence match that pair. ch7_numbers.py does not check N7-46 itself; the match is by value and description, not by the block's SOURCE line.

## Numbers left at four decimals, and why

Rule 7: the fourth decimal is 5 and no raw value was read.
- Ch5 `tab:cut_walkthrough`, last row, best removed 0.5545. Its raw value is among the 8,220 configurations in `/Volumes/seagate/slim_precompute/herg_ki.slim.db`, but picking it out means replaying the cuts in order. The column now reads 0.571, 0.563, 0.552, 0.554, 0.5545.
- Ch5 "an MCC of 0.5085 against 0.390" (fed01, Caco-2, DNN, site selection). The raw value is a mean over seeds in a partial jsonl file; not computed.
- Ch6 "reaches 0.0155 Spearman" (SMAC colloidal stability, best single representation).
- Ch6 "falls from 0.2315 to 0.1535 Spearman" (LogD, 500 against 2,000 features). Both kept, so the pair stays at the same precision.
- Ch6 "from 0.582 to 0.7115 RMSE" (hERG Ki, Gaussian process, importance weighting).
- Ch6 "from 0.136 to 0.0295 Spearman" (SMAC colloidal stability, network).
- Ch6 `tab:importance_weighting`: Caco-2 hybrid weighted 0.3225; LogD single weighted 0.6065 and hybrid weighted 0.5765. The weighted columns are not in any CSV in `figures_ch5_7/`.

Mixed precision in one sentence or row follows from these: Ch5 0.5085 against 0.390; Ch6 0.582 to 0.7115; Ch6 0.136 to 0.0295; the Ch5 and Ch6 table rows above. Reading the raw value of the kept number would let both print at three decimals.

Tie rule (two compared values print the same at the rule's precision):
- Ch5 `tab:tuning_base_reps`, Buchwald--Hartwig LightGBM Avalon (difference): 20.22 and 20.21 both print 20.2 at one decimal. Kept at two decimals; the other three Buchwald--Hartwig rows went to one.
- Ch5 same table, QM9 DNN Mol2vec 0.4867 and 0.4867, and hERG Ki DNN GROVER 0.5891 and 0.5891. Default and tuned are equal even at four decimals. Kept at four, as the tie rule reads. Rounding both to 0.487 and 0.589 would say the same thing; that is the author's choice. The prose sentence "two repeat fits ... scored 0.487 and 0.495" is not a tie and was rounded, so 0.4867 prints as 0.487 there and as 0.4867 in the table.

Other rules:
- Ch5 "within 0.0020 RMSE" and Ch6 "at most 0.0015 RMSE": differences below 0.010.
- Ch7 1.4826 (three places): a constant.
- Ch3 p = 0.0625: an exact value (1/16).

## tests/analysis/ch7_numbers.py

- 79 typed text strings retyped to the three-decimal (or one-decimal, yield) value the prose now prints, in the function of the N7 ID whose paragraph holds the number (14 of them are N7-01's 0.3618, checked once per result file). Strings for numbers that are only in SOURCE comments stay at four decimals and still match those comments.
- The test for "TEXT EDITED?" now matches whole numbers only, so '0.838' is not found inside '0.8381'.
- The five claims that compared at four decimals (N7-05 "none of the five improved" and "only the cliff-aware gap improved"; the n40 "none improves" and "retuned beat fixed in 1 of 8 pairs") now compare at three.
- `thesis_blocks()` had stopped finding Chapter 7: the headings it searched for were renamed. It now takes Chapter 7 from its `\section{Results}` to the next `\chapter`.
- The docstring notes the retyping date.
- `python3 -I tests/analysis/ch7_numbers.py --help` runs. There is no text-only mode. A full run with `--skip-group-table` against the 2026-10-09 results gives the same outcome before and after the rounding: 565 numbers OK, 0 CHANGED, 0 TEXT DIFFERS FROM RAW, 4 MISSING, 3 TEXT EDITED?. The 3 TEXT EDITED? lines were there before this pass (the counts 53, 1 and 6 are now written in words).
