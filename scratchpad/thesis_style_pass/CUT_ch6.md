# Chapter 6 cutting pass (2026-10-10)

Prose words in Chapter 6 (chap:similarity), lines starting with % removed, counted with python3 -I:

- Backup (thesis.tex.before_style_apply_2026-10-10): 6504
- Before this cut: 6597
- After this cut: 6169 (5.1% below the backup; target was 6179 or fewer)

Every cut is in a paragraph marked "% TODO: manual review", or in the AI-marked caption of the similarity-experiments table.
No number was added or changed. No table body, \input, \label or SOURCE comment was touched.

## What was cut, one line per paragraph

- Intro, first paragraph: merged the two Chapter 4 sentences into one; folded the two examples into one sentence.
- Intro, datasets paragraph: shortened; the noisy-label pointer to Chapter 7 was removed (it is kept once in the Conclusions).
- Methods, first paragraph: "scaffold-group-aware folds" changed to "folds (grouped by scaffold family for small molecules)"; two scaling sentences merged.
- Methods, seeds paragraph: the exception sentence was shortened.
- Feature spaces: the "which we refer to as" clauses were shortened.
- Similarity in a hybrid, notation paragraph: the list of shared settings was replaced by "all other settings and the folds".
- Zheng/Itskowitz paragraph: "This modifies a published approach" and the repeat of Chapter 4's finding were cut; now two sentences plus the test.
- Neighbor-score paragraph: "scaffold groups" was changed to "scaffold families".
- Inverse-distance paragraph: the modelability comparison was cut to one clause.
- Undefined-folds list: first mention changed to "change in binding free energy ($\Delta\Delta G$) caused by a mutation".
- Nearest-neighbor settings: "scaffold group" was changed to "group (scaffold family, for small molecules)".
- Rank-scheme paragraph: cut the sentence giving the motivation; the comparison that justifies sweeping both is kept.
- Importance weighting, definition: cut "so that important features count for more".
- Importance weighting, formulas: cut the s_f^{1/2} sentence and the duplicate "same neighbors only under cosine" sentence. The four-times example is kept, merged into one sentence.
- Experiments intro: cut the list of the table's columns.
- Similarity-experiments caption: "five scaffold-group-aware outer folds" was changed to "the five outer folds".
- Q-RASAR paragraph: cut "not at every setting".
- Retrieval and clustering method: two k-means sentences merged.
- Three smaller checks: "scaffold groups" was changed to "scaffold families".
- Results, first paragraph: restates Chapter 4, so it now opens "As in Chapter 4"; three sentences merged into two.
- Results, SMAC sentence: merged into one.
- Results, hybrid tracks label difference: cut the sentence ranking the gains, which the table already shows.
- Results, other domains: shortened to one sentence.
- Results, predictive hybrids: two opening sentences merged.
- Results, importance weighting closes the gap: cut the reason the results are missing (the limitation itself is kept).
- Results, hindsight comparison: cut "It matters which single representation we compare against".
- Results, number of features: cut "The number of features matters too, because".
- Results, picked-again comparison: cut the second description of the two ways it favors the hybrid. It now points to Methods, where that description is kept, and it is also kept in the limitations.
- Results, antibody prediction: cut the sentence listing which datasets change sign, which repeats the two lists before it. Changed to $\Delta\Delta G$.
- Results, importance weighting in prediction: the list of losses was replaced by a pointer to the bold cells of the table.
- Results, dataset matters more than settings: cut "giving more weight to the features that matter"; changed to $\Delta\Delta G$.
- Results, retrieval same pattern: three sentences merged; changed to $\Delta\Delta G$.
- Cliffs, shares paragraph: cut the restated 15% rule and the looser held-out rule (both are in Methods), the "does not mean quite the same thing" sentence (it is in the limitations) and the QM9 48 count (it is in the next paragraph and the limitations).
- Cliffs, main result: the LogD exception was merged with the LogD every-fold sentence.
- Prediction error: cut the framing sentence "Distance can also be used to flag...".
- Prediction error, antibodies: shortened "On three datasets"; changed to $\Delta\Delta G$.
- Retrieval: cut the restated definition of retrieval.
- Retrieval, antibodies: merged the two $\Delta\Delta G$ sentences and the two SMAC clustering sentences.
- Label information: cut "We expected ...".
- Conclusions, other domains: merged the last two sentences; changed to $\Delta\Delta G$.
- Conclusions, last paragraph: cut "This chapter leaves open the step that motivated the hybrid in the first place".

## Not changed

- Table row "Binding free energy" in tab:importance_weighting, because table cells are excluded from this pass.
- Methods comment lines and the "binding free energy" text inside "% CUT FROM TEXT" comments.
- Open point, not fixed: the LogD weighted-distance sentence in Results cites tab:distance_tracks_activity. Its caption says no importance weighting, so the reference may point to the wrong table. This was left for the author.
