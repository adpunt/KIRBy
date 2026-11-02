# Chapter 7 (chap:noise) cutting pass, 2026-10-10

Prose words, lines starting with % removed, counted with python3 -I from `\chapter{...}\label{chap:noise}` to the next chapter:

- Backup (thesis.tex.before_style_apply_2026-10-10): 13,190
- Before this cut: 14,259
- After this cut: 12,467 (5.5% below the backup; target was 12,530 or fewer)

Other sessions edited the chapter during this pass, so the after count includes their changes too.

All cuts are from paragraphs marked `% TODO: manual review`. No number was changed. Numbers that left the prose were already in a figure, or were copied into a `% CUT FROM TEXT` comment. No table body, label, \input, \includegraphics or SOURCE line was touched.

## What was cut, one line per paragraph

- Introduction, methods paragraph: merged with the next paragraph; the QM9 explanation is now one sentence pointing to sec:noise:tests.
- Introduction, QM9 paragraph: cut. The same point is made in Evaluation and controls.
- Introduction, summary of findings: cut. The findings are in the Results and the Conclusions.
- Introduction, feature-space paragraph: Chapter 4 background shortened to one sentence with a pointer.
- Noise conditions: the two shift kinds now point to Table tab:noise:kinds instead of restating its units; the level-nesting sentences merged.
- Three settings with no injection: three sentences merged into one.
- Detectors: the three "kinds" sentences that restate the Reads column of the table merged into one.
- Mitigation intro: the four-things list that restates the table columns cut to one sentence.
- Selected by a detector: the reason for the random-deletion control cut (Evaluation states it).
- Order of cleaning and search: two sentences on the four combinations merged.
- Every molecule (Huber blend): the second half of the last sentence cut.
- Changing the model (methods): first sentence shortened.
- Evaluation, five controls: each control was described a second time; now one sentence that lists them. Seed counts stay with each method.
- Evaluation, QM9 control: Chapter 3 aside, the "every table carries QM9" sentence and the list of methods with no QM9 result cut (the list is in the limitations).
- Overall effect: the three example results cut (each is in its own Results subsection); signposting sentence after the table cut.
- Score against share flagged (fig:noise:score_vs_level): sentences that restate the figure cut; numbers 0.8375, 0.8028, 0.8060 kept in a CUT FROM TEXT comment.
- Weighting: "best of many settings" (stated once in the overall caveats) cut; QM9 lead-in and the tree-model clause cut.
- Did the cleaning work (fig. f08): the "best setting picked on the held-out score" sentence cut (overall caveats).
- Labels as delivered, synthesis paragraph: cut to one sentence pointing to the antibody results.
- Labels as delivered, first paragraph: "there is no ground truth" clause cut.
- QM9 deletion: one restating sentence cut.
- Detection, first paragraph: the grading rule (in Methods) replaced by a pointer.
- Cliff-aware gap: its definition, repeated from Methods, cut.
- Distance families: the sentence repeating the Methods set-up cut.
- Break-even precision: last three sentences merged; the repeat of "below it under the shifts" cut.
- Two learners against break-even (fig. f10): axis and line descriptions that restate the caption cut; Caco-2 RMSE range cut from prose (in the % Numbers comment above).
- Detection synthesis paragraph (Chapter 3 link, flag point): cut. Every sentence was already in this section or in the comparison across damage kinds.
- Effect by noise type, signposting paragraph: cut.
- Clipped labels: the method restatement and the list of figure datasets cut; QM9 and Caco-2 sentences merged.
- Deleting clipped molecules: the first sentence (repeat of Methods) cut.
- Per-molecule lift: the restated method sentence cut.
- Scaffold-family offsets, two paragraphs: the Methods restatement and the random-intercept definition cut; one sentence shortened.
- Several sources: the hedged explanation ("likely already close to the truth") cut.
- Huber teacher blend: the no-QM9 sentence cut (in the limitations).
- Comparison across damage kinds: shortened from eight sentences to four; the hERG-folds sentence is in the Results already.
- Effect by dataset, signposting paragraph: cut.
- SMAC: the 53-antibody caveat shortened (it is also in the summary paragraph).
- $\Delta\Delta G$ and titer: the shuffled-control sentence folded into the first sentence.
- Antibody summary: the repeated Spearman-on-53 sentence and the hybrid-about-half sentence cut (the second is in Effect by model and representation).
- Small molecules summary: cut to the hERG K_i and Buchwald--Hartwig sentences; the rest repeated the Results.
- hERG IC50: the interval model's definition and the graded-molecules sentence (both in Methods) cut.
- Cleaning inside the search: the Methods restatement cut; the closing sentence folded into the first.
- Search on clean against damaged labels: set-up shortened to one sentence.
- Changing the model, first paragraph: the "comparison point" sentence (in Methods) cut.
- Retuning: two sentences merged; the no-QM9 sentence cut (in the limitations).
- Expanded sweep: the sentence restating the table caption cut.
- Heteroscedastic GP: the sentence on what earlier methods estimated and the definition of distance-based selection cut.
- Method ranking by feature space: the Chapter 4 restatement cut.
- Effect by model and representation: "so there is no lead to keep" cut.
- Former Discussion paragraph (sec:noise:discussion): cut from eight sentences to three; the rest repeated the Results and the Conclusions.

## Other changes in AI-marked text

- "scaffold group(s)" became "scaffold family/families" in three prose places: the neighbor-setting paragraph, the cleanlab sentence and the retuning description. Table cells and the subsubsection title "A shared offset per scaffold group" were left alone. The title's TODO says the wording is the author's.
- ddg_combined: "the change in binding free energy ($\Delta\Delta G$)" at its first mention in the chapter (distance families). Later prose mentions say "$\Delta\Delta G$". Comment lines were not changed.
