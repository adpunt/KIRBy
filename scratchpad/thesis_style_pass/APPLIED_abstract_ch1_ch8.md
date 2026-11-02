# Applied: abstract, Chapter 1, Chapter 8 (2026-10-10)

Every OLD block of DRAFT_abstract_ch1_ch8.md still matched the files exactly once.

- D1 adapted: abstract replaced in main.tex. The plan's TODO line was shortened. "13 models" became "13 base models", as in the old abstract. "In an exploratory study" became "In an exploratory study carried out earlier in this project". Three long sentences were split. "A newly published embedding is added" became "can be added". The closing sentence "Together, these studies show that label noise in molecular data is not Gaussian, and that the features a model is given limit how well the noise can be handled" was rewritten. The thesis injects noise, so it cannot show that real noise is not Gaussian. The new sentence is "Label noise in molecular data therefore cannot be treated as Gaussian, and the features a model is given change which noise method works."
- D2 applied: comma ("are, of course, sensitive").
- D3 applied: "I"/"my" to "we"/"our", "cheminformatcs" corrected, "noise mitigation techniques" to "methods for learning with noisy labels". Edited-your-text TODO added.
- D4 adapted: the dramatic opening "The second half of this thesis grew out of a failure." was dropped. The repeated (Chapter~\ref{chap:flaws}) on the new sentence was dropped, since the next sentence but one cites it. The end sentence, the questions and both TODO lines were applied as planned.
- D5 applied: "artifical" corrected, stray comma removed.
- D6 adapted: tense fixes, the broken uncertainty sentence and the model list were applied. "We replicated" became "We replicate", her own verb, not "We imitate". "a made-up assay limit" was kept, because informal wording is not an error. The TODO also asks her to check "LightGBM, XGBoost and the plain neural networks", which disagrees with Chapter 3's Conclusions.
- D7 applied: "did worst" sentence. The TODO quotes the new sentence in full.
- D8 applied: "customizes", "generate", "compared with", and two sentences of Chapter 5 findings. Her TODO kept, with an edited-your-text TODO under it.
- D9 adapted: the Chapter 6 paragraph was applied as planned. In the new Chapter 7 paragraph, "depended on the type of noise, the data and the model" became "depended on the task, the data, the model and the type of noise". This matches the abstract and Chapter 8.
- D10 adapted: the edit was applied. "future directions and conclusions" also became "future work and concluding remarks", to match the new Chapter 8 section titles. The TODO says so.
- D11 adapted: "seven noise conditions compared at a matched amount of noise" became "seven noise conditions, all but one at a matched amount of noise". Chapter 1 says one condition is not matched.
- D12 applied: OPIG datasets now cover Chapters 5 to 7.
- D13 applied: author's own work now runs from Chapter 4 to Chapter 7. The TODO also asks her to check that Chapter 3 is outside the range.
- D14 adapted: Chapter 8 was replaced in three Edits (introduction and Summary of findings; Contributions; Limitations to Concluding remarks). Changes from the plan:
  - Intro: "kept failing ... one level further down" became "did poorly, and asking why led us to the representation itself".
  - The one-line answers "It does." and "The type matters." were cut, and the answers were stated directly.
  - The Chapter 4 paragraph now says it covers the configurations that used a distance. It no longer says it explains all of the published methods.
  - "Bayesian networks" became "Bayesian neural networks".
  - "both hold, under conditions" became "they can, under the right conditions, and ... the hybrid keeps its lead".
  - The 0.6354 RMSE value now names the network on its best single representation, as Chapter 7's comment does.
  - For antibodies and reactions, "activity" became "label".
  - The imperative "So before choosing a method, ask ..." became "it therefore helps to ask".
  - Contribution item 2 says "all but one at a matched amount of noise". Item 11 was made one sentence with a chapter reference.
  - Future work and Limitations sentences were split to stay under 30 words.
  - All five old section labels and the old comments are kept (checked by script against thesis.tex.before_style_apply_2026-10-10).

Checks afterwards:
- Labels: every \label of the old Chapter 8 exists in the new one. sec:discussion:noise, antibodies, conditional and simple sit on "Summary of findings". sec:discussion:claim sits on "Concluding remarks".
- \ref: every \ref in thesis.tex and main.tex resolves, with three exceptions that were already there. The app:* labels resolve through xr from scratchpad/thesis_code_todo/build/additional_files.aux; all were checked present. lst:blueprint is set by the lstlisting's label= option. tab:noise:res:bygroup (Chapter 7, line ~4223 placeholder) and tab:noise:res:robust_broad (Chapter 7, line ~4672) are defined only in figures_ch5_7/ files that thesis.tex does not \input. They will print "??" until the figures session inputs them.
- Order and terms of the abstract, Chapter 1 and Chapter 8 match. All three use the same chapter order. All three name the three groups as noise detection, noise mitigation and noise-robust training, in that order. All three say "depended on the task, the data, the model and the type of noise", the known flag (assay limit, scaffold family), and the hybrid's lead through every treatment. The one disagreement left is in her Chapter 1 text: "LightGBM, XGBoost and the plain neural networks lost the most", where Chapter 8 (from Chapter 3's Conclusions) names only LightGBM and XGBoost. It is flagged in a TODO and was not changed.
