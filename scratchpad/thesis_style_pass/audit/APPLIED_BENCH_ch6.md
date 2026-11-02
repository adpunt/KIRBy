# Chapter 6: benchmark fixes applied (2026-10-10)

Source list: `BENCH_AUDIT_ch6.md`. Every passage was re-found in the live `thesis.tex` before editing. All edits are inside paragraphs marked `% TODO: manual review`. No number, table, caption, `\ref` or `\label` was changed. Snapshot before this run: `thesis.tex.before_bench_fixes_2026-10-10`.

## Listed fixes (36 edits)

| Fix | Status | Note |
|---|---|---|
| 1 Results lead-in | Applied, adapted | The sentence sits on the line straight after `\section{Results}`, with no blank line, so the existing `% TODO: manual review` above the heading still covers it. No new marker was added. |
| 2 Merge "The same holds" paragraph | Applied | Checked `tab_distance_tracks_activity.tex`: the hybrid is higher on every antibody dataset and Buchwald--Hartwig. The one-sentence paragraph and its marker were deleted with it. |
| 3 Conclusions caveat sentence | Applied | Cut "The comparison with the setting picked again favors the hybrid..."; Results and Methods keep it. |
| 4a, 4b MDR1 sentence moved to Methods | Applied | Text unchanged since the table pass; table list kept word for word. |
| 5a "So the spread" | Applied | |
| 5b "So a representation's weight" | Applied | |
| 5c "So the weighted distances" | Applied | |
| 5d "so we count it as new" | Applied | |
| 6 QM9 / "On both" | Applied | "On both" became "on both datasets". |
| 7 single representation definition | Applied | |
| 8 antibody distance sentence | Applied | Kept the EGFR binding and Delta Delta G exceptions the table pass added. |
| 9 "In the other models" | Applied | |
| 10a, 10b, 10c "On" openers | Applied | |
| 11 Conclusions "on all six" | Applied | |
| 12a-12d lower/higher is better | Applied | |
| 13a, 13b cliff rule | Applied | |
| 14a-14g short sentences | Applied | All seven. |
| 15 Experiments lead-in | Applied | |
| 16 "from three starting sets" | Applied | |

Skipped: none.

## Found in the second read (4 edits)

- Introduction, paragraph 1: "...alone. This chapter tests that idea." joined into one sentence.
- Distance against prediction error (Methods): the regression and classification error definitions joined with "and".
- Weight formulas paragraph: "No experiment in this chapter varies the exponent. The square root was first written..." became one sentence: "The square root was first written for Euclidean distance and kept for the others, and no experiment in this chapter varies the exponent."
- Importance weighting results: "With neighbor-label features" (a coined phrase) became "With the four extra features from the nearest training molecules", the wording of the Methods.

## Not changed, and why

- Merges that would make a sentence over 30 words were not made: the picked-again win/loss lists (antibody paragraph), the search-choices sentence in "depends far more on the dataset", the inverse-distance weights comparison, the cliff fitting/scoring pair.
- The `itemize` lists and the research questions are counted as the author's and were not touched.
- Open points from the audit for the author or the results chat, not language: "all 13 datasets" against "14 datasets"; "support vector machine" against "support vector regressor" (lines on importance weighting and distance against error); the Conclusions say QM9 has no best-number-of-features result, but `tab:similarity:feature_count` has a QM9 row.
