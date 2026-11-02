# Chapter 2 cutting pass (2026-10-10)

Prose words in Chapter 2 (chap:background to chap:robustness), lines starting with % removed, counted with python3 -I:

- Backup (thesis.tex.before_style_apply_2026-10-10): 8256
- Before this cut: 8530
- After this cut: 7753 (6.1% below the backup; target was 7843 or fewer)

All cuts are in AI-marked paragraphs. No author text, number, label, table body, figure or SOURCE comment was changed.

## What was cut, one line per paragraph

- Overview: six sentences restating Chapter 1's questions and the chapter's route cut to one sentence listing the topics.
- Support vector machines: kernel sentence shortened; three wrong-label properties merged into one sentence; "Chapter 3 reports its robustness" signpost cut.
- Similar property principle + "difficulty with this principle": merged into one paragraph; "no standard definition" sentence (repeated by the next sentence) cut.
- Activity cliffs: "exceptions to the similar property principle" cut (restates the definition).
- Dimensionality: Beyer1999 restated twice; one restatement cut, rest shortened.
- Scaffold family definition: pointer to Chapters 3 and 5 for stereochemistry and ring-less molecules cut (those chapters give it).
- LNL paragraphs 1 and 2: merged; selectors, actions and groups now point to Table tab:noise_terms instead of repeating all its rows.
- Song2022 paragraph: one-sentence description of each of the five groups cut to a list; last sentence (repeats the mitigation definition) cut.
- No single method / neighbor-based cleaning: two sentences restating the similar property principle merged into one.
- Implementation of the noise conditions: random-number-generator explanation and check list merged into two sentences.
- Regression noise: package settings not used in the thesis (Student-t nu 3, 5, 10; outlier 1% and 5%; zero-base outlier variant) cut.
- Requested noise: "no calibration search is needed", re-seeding detail and the uneven-family-size justification cut.
- Censoring: lightly shortened.
- Injection record + tolerance band: record list shortened ("number of groups, largest share held by one group" now "group sizes"); "finite sample" justification cut; warning sentences merged.
- Noise scale function: three sentences merged into two.
- Residual-unit injector: "sized against the error the model already makes" restatement cut; sentences merged.
- Classification noise: list of six strategies merged into one sentence.
- Robustness metrics: history of the linear slope in earlier NoiseInject versions cut; flag and threshold sentences merged.
- Uncertainty metrics: "The first ... the fourth" enumeration merged; "requires a separate pair of inputs" cut.
- Wrappers: "Example notebooks and scripts" sentence cut.
- Evaluation, splits paragraph: two pointer sentences merged.
- Conclusions: "no split is random" (stated in Evaluation) and "reused in later chapters" cut.

## Added

- Evaluation, first paragraph (already AI-marked): "Outside Chapter 3, results are reported as per-fold scores and their spread across folds, and we use no significance tests." Wording matched to Chapter 5 Methods ("per-fold differences ... and their spread across the 5 folds. We use no significance tests").

## Not changed

- "scaffold group" appears only in the NOTES comment and in a cell of Table tab:noise_terms (table body, not touched). No AI prose used it.
- The antibody ddG dataset is not mentioned in Chapter 2.
