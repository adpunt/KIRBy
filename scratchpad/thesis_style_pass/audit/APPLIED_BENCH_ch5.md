# Chapter 5: fixes from BENCH_AUDIT_ch5.md, as applied (2026-10-10)

Written by Claude. File: thesis.tex, Chapter 5 only. Snapshot before: thesis.tex.before_bench_fixes_2026-10-10.
Every edit was made with the Edit tool and only inside paragraphs or captions with a `% TODO: manual review` line directly above
(not `-- fixed:` / `-- moved:`). No number, table, caption, \ref or \label was changed. No claim corrected by the table check
(TABLES_ch5.md) was changed. Where a paragraph was split or a new lead-in added, the new part got its own
`% TODO: manual review` line so it is not mistaken for the author's text.

Counts: 57 numbered fixes. 44 applied as written, 10 adapted, 3 skipped. 11 further edits outside the list.

## Structure

- S1 adapted. Lead-in added under `\section{Results}`, written as two sentences (14 and 22 words) instead of one of 36.
- S2 adapted. Lead-in added under "Sources of the gain", plus one sentence naming the three checks that now have headings (S4).
- S3 applied. The noise/Chapters 6-7 pointer moved from "The fourth question asked..." into the last paragraph, with "first".
- S4 applied, option A. Three headings added in Results, as in Methods: "Fewer features or a tuned model", "Adding a new embedding", "Calibration".
- S5 skipped. No longer glued: the new-embeddings longtable now sits between the two paragraphs.

## Introduction

- I1 applied.
- I2 adapted. First merge reads "...one-hot encoded sequences, where each feature is set in only a few samples and so scores low importance on its own." Second merge as written; comment lines kept.
- I3 applied.

## Methods

- M1 applied.
- M2 applied. Nanobody paragraph removed from "The search space" and folded into the Results paragraph (see R14). The author's sentence "as shown below for nanobody thermostability" still holds: the result is now in Results.
- M3, M4, M5, M6 applied.
- M7 adapted. The proposed 45-word sentence was kept as two sentences; only the first "so" removed.
- M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19 applied.
- M20 applied. The LogD SVM pair is in tab:tuning_base_reps (row "LogD & SVM & MHG-GNN & 0.475 & 0.421"), checked. The QM9 pair stays as the paragraph's one pair.
- M21, M22, M23 applied.
- M24 applied. Split before "In the second experiment"; the second part has its own marker.
- M25 adapted. Two sentences instead of one of 40 words.

## Results

- R1 applied.
- R2 applied. Also: "On the Buchwald--Hartwig reactions it does not" (added by the table check) became a clause after a semicolon, unchanged in meaning, and "We then shuffled..." became "To see which representations the hybrid relies on, we shuffled...".
- R3 applied. The two merges were made and the paragraph was split before "In the $\Delta\Delta G$ hybrid"; the NOTE and TODO: CHECK comments stay with the first part. The second part has its own marker. "four" (changed by the table check) is untouched.
- R4 applied, including the optional join with the paragraph before.
- R5 applied. The budget-figure paragraph moved, with its SOURCE line, from "Against the best single representation" to the top of "Dependence on the downstream model". The colon merge was made.
- R6 no change (as listed).
- R7 applied.
- R8 skipped. The table check rewrote the passage, and "Caco-2 and hERG $K_i$ show the same" is gone.
- R9 applied. "0.009 to 0.037 RMSE" kept.
- R10 adapted. New opener as written. The sentence "Recursive elimination ranks all joined features in one list..." was moved up next to it. The "every budget" result now sits directly before its exception.
- R11, R12, R13 applied.
- R14 applied. The opening sentence was cut because the moved R5 paragraph now opens the subsection. The nanobody result from M2 was joined to the Caco-2 sentence. The colon merge was made.
- R15 applied. Tallies were already replaced by the table check.
- R16 applied (three merges).
- R17 adapted. "...on another; an index well below 1 therefore need not mean the hybrid fits noise." A bare "and" would have lost the causal link.
- R18 applied.
- R19 adapted. "...on every dataset and model, although it can build a different number of features. Each site can therefore select on its own...". The audit's "but" put the caveat and the conclusion the wrong way round.
- R20 adapted. The paragraph no longer has numbers (moved to tab:method:site_reps). Only the tally "in three of the four dataset and model pairs" was replaced, with "except on Caco-2 with the DNN, where both ensembles have the lower RMSE".
- R21 adapted. The limitation now opens the paragraph as its own sentence: "Only Caco-2 has finished this experiment, with three seeds for the DNN and two for the SVM. There, with two-class labels...". One sentence of 45 words would have been too long. The table check's sentence about the SVM at 20% flips is untouched.

## Conclusions

- C1 no change (as listed).
- C2 applied, both parts. The implication follows the Results sentence "which is why both are searched".
- C3 applied.
- C4 skipped. It narrows what the sentence claims, so the author has to approve it. The audit says the same.
- C5 applied. The DNN caveat is still in the Results lead-in, and the cut grids are still in "The search tests every combination".
- C6 = S3.

## Further edits, not in the list (all inside marked paragraphs)

1. "The full runs show the same split" (Methods): "On LogD, where counts of 2, 3, 5 and all were searched, the hybrid used...". Two short sentences merged.
2. "Some groupings give very unequal folds": three sentences of 6-9 words merged into one.
3. "Splitting rules": "no group is split between folds, and classification folds are also stratified...".
4. "Overfitting with the budget" (Methods): "...minus the held-out score; a positive value means...".
5. "Feature reduction at the same width": the mRMR cap and the small-molecule budgets merged into one sentence with the same content.
6. "Stability and blueprints": the three SHAP sentences merged into one with a colon list.
7. Separate-site main experiment: "It also compares both with..." folded into the sentence before.
8. PCA paragraph: the sentence opening with "So" now reads "The two experiments therefore differ...". The paragraph (about 250 words, three score pairs) was split before "On antibodies neither method wins throughout", and the new part has its own marker. No number changed. No table holds these, as TABLES_ch5.md notes.
9. Conformal paragraph: "...a few hundred antibodies, and the hybrid chosen on those antibodies is optimistic...".
10. Regularization paragraph: "...does that better, but with a few hundred antibodies neither does so on every dataset."
11. Hindsight small-molecule paragraph: the Buchwald--Hartwig clause and the purpose phrase, listed under R2.

## Left for the author

- C4: the Conclusions sentence "The hybrid also takes in new embeddings added to the pool" says more than Results shows.
- The Conclusions still name three antibody datasets that beat the best representation in hindsight. The table shows four. This is the table check's open TODO: CHECK, not touched here.
- After S3, the "fourth question" Conclusions paragraph is three sentences (about 55 words). It was left as one paragraph because it answers a separate question.
- The PCA paragraph keeps its three score pairs until a table exists.
