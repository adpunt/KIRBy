# Chapter 3 benchmark edits (2026-10-10)

Backup: thesis.tex.before_benchmark_edits_2026-10-10

Chapter 3 prose words (comment lines removed, from \chapter{Noise robustness...} to \chapter{Limitations...}):
- backup: 13,050
- now: 13,017

Which paragraphs were in scope: AI-marked paragraphs ranked in the top 150, or in a results, discussion or conclusion section. No AI-marked paragraph in Chapter 3 ranks in the top 150. All conclusion paragraphs are the author's. Chapter 3 has no discussion section. Two AI-marked paragraphs sit in Results. Both were edited. No tables were built.

## 1. "Across all datasets, models struggle to fit with" (Results, Robustness on the three assay datasets)
- Words: 93 before, 73 after.
- Opens with a pointer to Figure fig:assay.
- "Struggle the most ... lower values across the board" is now "gives the lowest \aucnorm{} values on every assay dataset". It is limited to the three noise conditions run on every model, because censoring is worse on QM9 (Table tab:censoring). I did not check "lowest on every assay dataset" against the result files. The claim is carried over from the old text.
- Cut "RF is the dominant model". The rank numbers that support it stay.
- Split the 38-word sentence about noise level 1.0 into two.
- Cut "Both are strong contenders for systematic label noise." It is an appraisal sentence.
- No numbers were changed. The TODO line above is kept.

## 2. "We consider the impact of label noise on" (Results, Robustness on the three assay datasets)
- Words: 74 before, 61 after.
- Cut the opening sentence. It only announced the paragraph.
- Added a pointer to Figure fig:assay.
- "Typically middle-ground in terms of predictive performance" is now "sit in the middle for predictive performance".
- "There is no pattern of noise they are better suited to" is now "no pattern of noise suits them better than another".
- The Caco-2 sentence is unchanged. The TODO line above is kept.
