# Decimal places in the thesis: the rule

Written 2026-10-10 by Claude for the style pass. The author asked for one rule without being asked.
Counts below were made on thesis.tex as it stood on 2026-10-10, prose outside `%` comments.

## 1. The rule

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

## 2. What the three reference theses do

Scores and correlations: three decimals.
- Dablander 2023: "twin networks show the strongest overall MCC-performance of 0.461 ... an
  AC-sensitivity of 0.465 and an AC-precision of 0.468"; "a mean QSAR-MAE on Dtest of 0.426 (in
  pIC50 units)"; "an MCC of 0.199 on Mtest ... an essentially equivalent MCC of 0.200". Numbers
  below 1 in his main text: 29 at three decimals, 13 at two, 13 at one.
- Vost 2025: Table 2.4 gives Pearson correlations at three decimals ("0.805 ± 0.010", "0.601");
  the text repeats them at three ("a PCC of 0.816", "0.805 vs 0.601", "a PCC of 0.803"). Numbers
  below 1: 36 at three decimals, 19 at two, 18 at one.
- Buttenschoen 2025 reports almost no scores in [0, 1]. His one Spearman is two decimals
  ("𝜌 = 0.08"). RMSD in Å is one decimal ("RMSD 1.9 Å").

Percentages: one decimal (or integers for round shares).
- Buttenschoen: 315 percentages at one decimal, 87 at two, 606 integers. "45.8 ± 2.9 % RMSD within
  2 Å"; "from 32.8 % to 81.2 % despite accuracy dropping from 79.7 % to ...".
- Vost: 47 at one decimal, 2 at two, 48 integers. "pass rates of 92.2% and 81.1%".
- Dablander mixes: "approximately 4.5%", "11.37%", "12.7%, 11.91%, and 6.84%". This is the one
  inconsistency in the references; we follow Buttenschoen and Vost.

Differences: given at the precision of the values compared (Vost "0.805 vs 0.601"; Dablander
"0.199 ... 0.200", two values 0.001 apart, both printed at three decimals). Relative improvements
are percentages (Dablander "a relative improvement of the median MAE of approximately 4.5%").

None of the three gives four decimals for a score in the main text.

## 3. What thesis.tex does now (prose outside comments)

Decimals of numbers below 1, per chapter. Settings such as 0.5 or 0.1 are mixed into the one-
and two-decimal counts, so those are upper bounds for scores.

| Chapter | 1 dp | 2 dp | 3 dp | 4 dp | percentages |
|---|---|---|---|---|---|
| 1 Introduction | 0 | 4 | 0 | 0 | integers |
| 2 Background | 1 | 1 | 1 | 0 | 2 at 1 dp, 16 integers |
| 3 Noise robustness | 23 | 23 | 193 | 1 (p = 0.0625) | 17 at 1 dp |
| 4 Similarity assumption | 19 | 38 | 2 | 0 | 24 at 1 dp |
| 5 Building hybrids | 11 | 15 | 2 | 43 | 6 at 1 dp, 29 integers |
| 6 Similarity in hybrids | 3 | 2 | 4 | 25 | 3 integers |
| 7 Label noise | 15 | 12 | 3 | 63 | 56 integers (all settings) |
| 8 Conclusions | 0 | 4 | 0 | 84 | 2 at 1 dp |

Four-decimal values of 1 or more: Chapter 6 has 2, Chapter 7 has 7 (1.4826 is a constant;
20.3381 and 18.6007 are Buchwald--Hartwig yield RMSE), Chapter 8 has 2 (20.3381 again).
Two-decimal values of 1 or more: Chapter 8 "16.17 to 15.62 RMSE in percentage points of yield".

So Chapter 3 already follows the rule (normalized AUC at three decimals, e.g. "between 0.037 and
0.065"). Chapter 4 uses two decimals. Chapters 5 to 8 use four.

Inline tables in thesis.tex (numbers below 1):
- Chapter 3: 15 at three decimals (Clean R^2 table). Follows the rule.
- Chapter 4: 81 at two decimals, 44 at three. tab:flaws:ties (lines ~1431-1470): 22 at two, 44 at
  three. tab:flaws:distance_summary (~1521): 49 at two. tab:flaws:neighbor_overlap (~1547): 28 at
  two.
- Chapter 6: tab:importance_weighting (~3600-3629): 44 at four decimals, 4 Buchwald--Hartwig yield
  RMSE at four (14.4572 ...). Hand-typed in thesis.tex, so it follows the rule here.
- Chapter 5: 11 at four decimals. Chapter 7: none at four.

What the rule changes:
- Rounded from four decimals to three: 55 values in Chapter 5, 76 in 6, 72 in 7, 92 in 8
  (prose and inline tables together), less the exceptions below.
- Kept at four by rule 7 (fourth decimal is 5, no raw value read yet): 7 in Chapter 5, 11 in 6,
  5 in 7, 10 in 8.
- Kept at four by the tie rule: at least one pair found by a scan of "a to b" / "a against b":
  Chapter 8 line ~4808, "0.6899 against 0.6901 Spearman" (both 0.690 at three decimals).
  The margin "at most 0.0015 RMSE" (Chapter 6, line ~3336) keeps four decimals (difference
  below 0.010).
- Left below the rule, listed for the author (rule 6): Chapter 4's two-decimal correlations in
  prose (lines ~1420, 1502, 1508, 1510, 1512, 1514, 1521, 1636, 1735) and in the two Chapter 4
  tables above; Chapter 3's "Spearman correlation of -0.10 (p = 0.73)"; two-decimal scores in
  Chapters 5 to 7. Chapter 1's literature values (0.68, 0.54) keep their source's precision.
- Chapter 8 repeats numbers from Chapters 5 to 7. It must print each at the same decimals as the
  chapter it comes from.

## 4. Tables from figures_ch5_7/ (not ours to change)

These are written by `tests/analysis/thesis_figures.py`, which the figures session edits.

| File | `\input` in thesis.tex | Decimals now |
|---|---|---|
| tab_hybrid_vs_best_single.tex | yes, line ~2791 (Chapter 5) | 80 at four |
| tab_distance_tracks_activity.tex | yes, line ~3489 (Chapter 6) | 28 at four |
| tab_nearest_neighbour_prediction.tex | yes, line ~3554 (Chapter 6) | 28 at four |
| tab_noise_bygroup.tex | no | integer counts only |
| tab_noise_mitigation_broad.tex | no | integer counts only |
| tab_noise_robust_broad.tex | no | integer counts only |

For the author or the figures session: the four-decimal cells come from `_pair_cells` in
`thesis_figures.py` (line ~289, `f"{r['single']:.4f}"`). The bold rule in `_bold_rows`
(line ~277) treats pairs equal at four decimals as ties. To follow the rule, both would format
at three decimals and keep four for a pair that ties at three. Until then, Chapters 5 and 6
print three decimals in prose and four in these three tables.

## 5. tests/analysis/ch7_numbers.py if Chapter 7 is rounded

How it matches the text now:
- `thesis_blocks()` cuts thesis.tex from `\section{Results: Does Noise Handling Work at All?}`
  to `\chapter{Discussion and Conclusions}`, then splits on the 49 `% SOURCE [N7-xx]:` lines.
- Each check passes the value as printed, as a string typed into the script ('0.8375'), either
  in an `EXPECTED` table (23 of them) or inline in `r.check(...)`. The file holds 416 such
  four-decimal strings (39 end in 5), 22 at three, 70 at two, 37 at one.
- `Report.check()` reads the precision from that string (`_decimals(text)`), rounds the raw value
  to it with `_fmt`, and compares. It also looks for the string in the paragraph; if absent it
  prints "TEXT EDITED?".
- The baseline `ch7_numbers_expected.json` stores raw values, not text, so it does not depend on
  the text's precision.

What must change, in the same commit as the Chapter 7 rounding:
1. Every typed text value that the prose rounds: retype it as the new three-decimal string
   ('0.8375' becomes the value the raw file gives at three decimals). Precision then follows
   automatically. Leave tie-rule and rule-7 values at four.
2. For the 39 strings ending in 5, run the tool first with the four-decimal strings, read the
   raw value it prints, and type the third decimal from that. Do not round '0.8375' by hand.
3. Claims that compare at four decimals must compare at the printed precision, or a claim can
   pass in the tool and read as a tie in the text. `round(..., 4)` at lines ~699, ~713, ~717
   (N7-05: "none of the five improved", "only the cliff-aware gap improved"), ~2987 and ~3012
   (N7 retuned/fixed: "none improves", "retuned beat fixed in 1 of 8 pairs") become
   `round(..., 3)`. Lines ~3424-3442 (sets of distinct scores) and ~2773 (`round(x, 6)`) test
   whether scores are identical and stay as they are. Line ~2822 already uses three.
4. The "TEXT EDITED?" test is a plain substring test. '0.838' is also found inside '0.8381', so a
   three-decimal string can pass against a four-decimal tie-rule number. Change it to a
   number-boundary match, e.g. `re.search(r'(?<![\d.])' + re.escape(t) + r'(?!\d)', block)`.
5. Docstring: "copied from thesis.tex on 2026-10-09" becomes the date of the rounding.
6. Section anchors. If the restructure renames "Results: Does Noise Handling Work at All?" or
   "Discussion and Conclusions", update the two `tex.find(...)` strings in `thesis_blocks()`
   (line ~215), or every check prints "TEXT EDITED?". Keep every `% SOURCE [N7-xx]:` line.
7. The `{x:.4f}` in note and NEW FOLD lines are printouts for the author, not checks. They can
   stay.

Order with the results being fixed in another session: Chapter 7 numbers that change there
should be written at three decimals when they are updated, and the matching strings retyped in
the same edit. Rounding Chapter 7 before those updates land means doing it twice.
