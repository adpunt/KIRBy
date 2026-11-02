# Review: Chapter 1 (chap:intro) and the abstract (main.tex), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 and main.tex.before_style_apply_2026-10-10 by diff, and read in full as they stand on 2026-10-10. her_text_check.py hits in this unit: #1 (comma, thesis.tex para 2), #2 (second-half questions, has its note), #3 (Chapter 2 outline paragraph: spelling and a comma), #5 (Chapter 7 outline paragraph, AI-marked, unchanged since the backup: false positive). Every OLD block below was checked with python3 -I str.count to occur exactly once in its file. Every number in NEW text is already in thesis.tex (13 base models: line 869; six representations: 869; seven noise conditions; all five small-molecule datasets: Chapter 8; 0.44 and 0.54: line 1527).

Checks that passed: every \ref in Chapter 1 resolves (chapter and section labels in thesis.tex; app:availability and app:datasets through xr in the Additional files build aux). Section titles are in sentence case. No "Results:" prefixes. Chapter 1, the abstract and Chapter 8 follow the same order (label-noise questions, the similarity failure, the hybrid, distance, noise handling). Chapter 1 has about 2,000 words against Buttenschoen's 3,870; the style guide accepts this because Chapter 2 holds the background. The abstract has 454 words (Buttenschoen about 390, Dablander 270, Vost 568). Style guide items 18 (Chapter 7 paragraph), 19 ("this distance" now introduced), 20 (questions rebuilt and answered in Chapter 8) and 44 (outline tense) are done.

## Fixes

### V1 — Chapter 1 paragraph 2: unnoted changes, K_i typography, standard deviation called a disagreement
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: comma added after "are" (", of course,"); "pK$_i$" and "K$_i$" set as "p$K_i$" and "$K_i$", as everywhere else in the thesis; "compound-target" given an en dash, as in Chapter 2; "disagree by about" changed to "have a standard deviation of about", because Kramer2012's 0.54 is the standard deviation (the mean difference is 0.44, as Chapter 4 says). Please check that Kalliokoski2013's 0.68 is also a standard deviation, and whether landrum2024's 0.50 and 0.27 are means or medians.
- why: The comma was added in this pass with no note; under the strict rule a comma is not a trivial change. Chemistry guide items 7 and 10: the 0.54 for p$K_i$ is Kramer's standard deviation, and the same source is quoted in Chapter 4 (line 1527) as 0.44 on average with standard deviation 0.54, so Chapter 1 contradicts Chapter 4. Ch1 holds the last two "pK$_i$" and "K$_i$" forms in the thesis body.
OLD:
```latex
QSAR models are, of course, sensitive to the quality of the input data on which they are trained.
```
NEW:
```latex
% TODO: manual review -- edited your text: comma added after "are" (", of course,"); "pK$_i$" and "K$_i$" set as "p$K_i$" and "$K_i$", as everywhere else in the thesis; "compound-target" given an en dash, as in Chapter 2; "disagree by about" changed to "have a standard deviation of about", because Kramer2012's 0.54 is the standard deviation (the mean difference is 0.44, as Chapter 4 says). Please check that Kalliokoski2013's 0.68 is also a standard deviation, and whether landrum2024's 0.50 and 0.27 are means or medians.
QSAR models are, of course, sensitive to the quality of the input data on which they are trained.
```

### V2 — Chapter 1 paragraph 2: the replacements announced in V1's note
- file: thesis.tex
- kind: author-text
- todo_line: covered by V1
- why: See V1. The fold conversions (4.8-fold, 3.5-fold) are kept; they still describe one standard deviation in concentration.
OLD:
```latex
Here pIC$_{50}$ and pK$_i$ are the negative base-10 logarithms of the IC$_{50}$ and K$_i$ in molar units, so a difference of one log unit is a tenfold difference in concentration. Repeat measurements of the same compound-target pair disagree by about $0.68$ log units for pIC$_{50}$ and $0.54$ log units for pK$_i$
```
NEW:
```latex
Here pIC$_{50}$ and p$K_i$ are the negative base-10 logarithms of the IC$_{50}$ and $K_i$ in molar units, so a difference of one log unit is a tenfold difference in concentration. Repeat measurements of the same compound--target pair have a standard deviation of about $0.68$ log units for pIC$_{50}$ and $0.54$ log units for p$K_i$
```

### V3 — Chapter 2 outline paragraph: comma removed with no note
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "artifical" corrected, and the comma before "and evaluation metrics" removed, since the sentence joins two items, not a list.
- why: her_text_check hit #3. The spelling fix is trivial; the comma removal is not, under the strict rule, and had no note. The change itself fixes a real slip, so it stays and gets a note.
OLD:
```latex
Chapter~\ref{chap:background} introduces molecular representations, model families, molecular similarity and activity cliffs, and the sources of noise in molecular data.
```
NEW:
```latex
% TODO: manual review -- edited your text: "artifical" corrected, and the comma before "and evaluation metrics" removed, since the sentence joins two items, not a list.
Chapter~\ref{chap:background} introduces molecular representations, model families, molecular similarity and activity cliffs, and the sources of noise in molecular data.
```

### V4 — Hypothesis paragraph: the added sentence names the wrong data
- file: thesis.tex
- kind: ai-inserted-sentence (inside her paragraph; its note already says two sentences were added)
- todo_line: keep the existing note
- why: The added first sentence says the exploratory study ran "on QSAR data". Chapter 4 (sec:lnl_motivation) says it ran on one dataset, the QM9 HOMO--LUMO gap, with Gaussian noise added to the training labels. QM9 labels are computed, not assay measurements, so "QSAR data" is wrong.
OLD:
```latex
published methods for learning with noisy labels did poorly on QSAR data.
```
NEW:
```latex
published methods for learning with noisy labels did poorly on the QM9 HOMO--LUMO gap with Gaussian noise added to the training labels.
```

### V5 — Chapter 5 outline paragraph: "It" refers to the hybrid
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "It also investigates" changed to "The chapter also investigates", because the sentence before has "The hybrid" as its subject, so "It" read as the hybrid investigating itself.
- why: Broken reference in her sentence (allowed fix). Her earlier note above this paragraph stays.
OLD:
```latex
random features of the same size. It also investigates properties of the hybrid representation
```
NEW:
```latex
random features of the same size. The chapter also investigates properties of the hybrid representation
```
Note for the applier: put the todo_line directly above the paragraph, under the two TODO lines already there (the line starting "% TODO: manual review -- edited your text: as your TODO above asked.").

### V6 — Chapter 7 outline paragraph (AI): vague and informal phrases
- file: thesis.tex
- kind: ai-text
- todo_line: keep "% TODO: manual review"
- why: "labels damaged on purpose" is informal; Chapter 7 says injected noise. "Label noise was handled, but ..." is an empty opener before the actual finding. One sentence joins two separate findings with "and"; split.
OLD:
```latex
It tests noise detection, noise mitigation and noise-robust training on labels damaged on purpose and on labels as delivered. QM9 with no noise added, random deletion and shuffled detector scores serve as controls. Label noise was handled, but which method worked depended on the task, the data, the model and the type of noise. Corrections worked best when a known flag, such as an assay limit or a scaffold family, said which labels were wrong. The hybrid stayed ahead of the single representation on small molecules through every treatment, and the ranking of noise methods changed with the feature space.
```
NEW:
```latex
It tests noise detection, noise mitigation and noise-robust training, on labels with injected noise and on the labels as delivered. QM9 with no noise added, random deletion and shuffled detector scores serve as controls. Which method helped depended on the task, the data, the model and the type of noise. Corrections worked best when a known flag, such as an assay limit or a scaffold family, said which labels were wrong. On small molecules the hybrid stayed ahead of the single representation through every treatment. The ranking of noise methods changed with the feature space.
```

### V7 — Contributions paragraph (AI): unclear "it", "main" as appraisal
- file: thesis.tex
- kind: ai-text
- todo_line: keep "% TODO: manual review"
- why: "with antibody representations built for it" has no clear referent; "the main contributions" ranks without saying by what; "split of the variance in robustness" leaves out what Chapter 8's list says (model family, representation and their pairing). Rewritten from Chapter 8's contributions list, same order.
OLD:
```latex
On label noise, the main contributions are NoiseInject and seven noise conditions, all but one at a matched amount of noise. They also include a split of the variance in robustness between model and representation, and noise-handling methods tested against controls. On the representation, the main contribution is the adaptive hybrid representation, built for prediction and for distance, with antibody representations built for it.
```
NEW:
```latex
On label noise, the contributions are NoiseInject and seven noise conditions, all but one at a matched amount of noise. They also include a split of the variance in noise robustness between model family, representation and their pairing. Noise-handling methods were tested on single representations and hybrids side by side, against controls. On the representation, the contributions are the adaptive hybrid representation, hybrids built for distance, and new antibody representations.
```

### V8 — Abstract paragraph 2 (AI): words not hers, uncertainty claim wrong by implication
- file: main.tex
- kind: ai-text
- todo_line: keep both existing TODO lines
- why: "decided noise robustness" and "barely mattered" are not her words; Chapter 1 says "the main factor" and "mattered more for predictive performance". "outside censoring it did not point to the corrupted labels" implies it did under censoring; Chapter 3 and Chapter 8 say models were most confident about the clipped labels. "on QM9 and three assay datasets" was lost in the rewrite; restored from the earlier abstract.
OLD:
```latex
With it we compared 13 base models and six molecular representations under seven kinds of label noise. Model family decided noise robustness more than the representation did, and random forests were the most consistent model. The shape of the noise distribution barely mattered. Systematic noise did the most damage, for example an offset shared within a scaffold family or labels clipped at an assay limit. Predicted uncertainty rose with noise, but outside censoring it did not point to the corrupted labels.
```
NEW:
```latex
With it we compared 13 base models and six molecular representations under seven kinds of label noise, on QM9 and three assay datasets. Model family was the main factor in noise robustness, and the representation mattered more for predictive performance. Random forests were the most consistent model. The shape of the noise distribution made little difference. Systematic noise did the most damage, for example an offset shared within a scaffold family or labels clipped at an assay limit. Mean predicted uncertainty rose with noise for almost every probabilistic model on QM9, but no model's uncertainty identified the corrupted labels. Under clipping, the models were most confident about the clipped labels.
```

### V9 — Abstract paragraph 3 (AI): order marker misplaced, dataset vague, argument unfinished
- file: main.tex
- kind: ai-text
- todo_line: keep both existing TODO lines
- why: The paragraph opens with no order word and puts "Next" in its third sentence, breaking the "First, ... Next, ... Last" pattern the TODO line promises. "molecular data" hides that the study used one dataset (QM9, Gaussian noise). The last sentence stops before the consequence ("so together they dominate it", her wording in Chapter 1).
OLD:
```latex
In an exploratory study carried out earlier in this project, published methods for learning with noisy labels did poorly on molecular data. Those that compared molecules by a distance did worst. Next, we traced this failure to the representation. On a single dataset, most features of a standard representation do not vary or do not relate to the label, yet each adds to the distance between two molecules.
```
NEW:
```latex
Next, we traced why published methods for learning with noisy labels did poorly in an exploratory study carried out earlier in this project. That study used QM9 with Gaussian noise added to the training labels, and the methods that compared molecules by a distance did worst. On a single dataset, most features of a standard representation do not vary or do not relate to the label. Each still adds a term to the distance between two molecules, so together they dominate it.
```

### V10 — Abstract paragraph 4 (AI): antibody result oversold, "better nearest neighbors" vague
- file: main.tex
- kind: ai-text
- todo_line: keep both existing TODO lines
- why: The paragraph reports the small-molecule gain but not that on antibodies the gain depended on the dataset (Chapter 1 and Chapter 8 both say so). "gave better nearest neighbors" does not say better at what. "for which we built new representations" reads as covering reactions too; Chapter 8 lists new representations for antibodies only.
OLD:
```latex
On every small-molecule dataset it predicted better than the best single representation, and a hybrid built for distance gave better nearest neighbors. The search also runs on reactions and on antibodies, for which we built new representations. A newly published embedding can be added as one more representation to search.
```
NEW:
```latex
On every small-molecule dataset it predicted better than the best single representation. On antibodies the gain depended on the dataset. In a hybrid built for distance, the distance between two molecules followed their label difference more closely, and nearest neighbors predicted the label better. The search also runs on reactions, and for antibodies we built new representations. A newly published embedding can be added to the search as one more representation.
```

### V11 — Abstract paragraph 5 (AI): empty opener, vague controls
- file: main.tex
- kind: ai-text
- todo_line: keep both existing TODO lines
- why: "Label noise was handled, but" says nothing before the finding. "random and shuffled controls" does not say what was random or shuffled; Chapter 1 names them. "the data flagged the wrong labels" is looser than Chapter 7's "known flag". "every cleaning" is not a phrase the thesis uses; Chapter 1 and Chapter 8 say "every treatment".
OLD:
```latex
We tested noise detection, noise mitigation and noise-robust training on single representations and hybrids side by side, against random and shuffled controls. Label noise was handled, but which method worked depended on the task, the data, the model and the type of noise. Corrections worked best when the data flagged the wrong labels, as an assay limit or a scaffold family does. On small molecules the hybrid kept its lead through every cleaning, and the ranking of noise methods changed with the feature space.
```
NEW:
```latex
We tested noise detection, noise mitigation and noise-robust training on single representations and hybrids side by side. The controls were random deletion, shuffled detector scores and QM9 with no noise added. Which method helped depended on the task, the data, the model and the type of noise. Corrections worked best when a known flag, such as an assay limit or a scaffold family, marked the wrong labels. On small molecules the hybrid stayed ahead of the single representation through every treatment, and the ranking of noise methods changed with the feature space.
```

### V12 — Abstract last paragraph (AI): conclusion contradicts the paragraph above
- file: main.tex
- kind: ai-text
- todo_line: keep both existing TODO lines
- why: "Label noise in molecular data therefore cannot be treated as Gaussian" follows a paragraph saying the shape of the noise distribution made little difference, and Chapter 3 tested Gaussian noise throughout; the measured finding is that systematic noise did more damage than Gaussian noise. "therefore" also draws a conclusion the abstract does not show.
OLD:
```latex
Label noise in molecular data therefore cannot be treated as Gaussian, and the features a model is given change which noise method works. No single model, representation or noise method was best on every dataset we tested. We recommend choosing a noise method by how the labels were made, and adapting the features to each dataset.
```
NEW:
```latex
Systematic noise did more damage than Gaussian noise, and the features a model was given changed which noise method worked. No single model, representation or noise method was best on every dataset we tested. We recommend choosing a noise method by how the labels were made, and adapting the features to each dataset.
```

## For the author
- Chapter 1, Chapter 3 outline paragraph: "The models with the highest predictive performance, LightGBM, XGBoost and the plain neural networks lost the most" is your text and still disagrees with Chapter 8 ("LightGBM and XGBoost predicted well on clean labels and lost the most"); Chapter 3's Conclusions give the GP as best on clean labels. Not changed; needs your call.
- Chapter 1, Collaborations: "the experiments in Chapters 4 to 7 are the author's own work" leaves Chapter 3 out. Not changed; only you know the split of work on the NoiseInject paper.
- Chapter 1 paragraph 2: V1/V2 assume Kalliokoski2013's 0.68 is a standard deviation like Kramer2012's 0.54. Not verified against the paper; check before applying V2, or drop V2's pIC$_{50}$ half.
- Chapter 8, "The third question ..." paragraph ends "Label noise in molecular data therefore cannot be treated as Gaussian", the same sentence V12 removes from the abstract. Chapter 8 is not in this unit; it should match whatever you decide for the abstract.
- Chapter 8's "Label noise was handled, but no single method handled it everywhere" is the same empty opener V6 and V11 remove. Not in this unit.
- The abstract has no results numbers since the 2026-10-10 rewrite (the 76% for censored EM on LogD was cut). Buttenschoen's abstract has none; whether to keep one is yours.
- The NOTES comment block under "Outline of the thesis and contributions" is stale: it calls Chapter 6 "Hybrid representations for similarity and label noise", Chapter 7 "Discussion", and asks to name the confirmation report. Comments do not print; left in place because comments are never deleted without you.
- Chemistry guide item 10 proposes one macro for $K_\mathrm{i}$ (upright i, IUPAC) across about 240 uses. V1/V2 only bring Chapter 1 in line with the current form ($K_i$). The macro is a thesis-wide decision.
- First-half question 1 says "when you instead consider"; your text, not wrong, left as is. Several sentences in your first two paragraphs exceed 30 words; left, since the length rule applies to AI text.

## Applied
- V1: applied as planned (TODO note above Chapter 1 paragraph 2).
- V2: applied as planned. The thesis elsewhere gives Kramer2012's 0.54 as a standard deviation; Kalliokoski2013's 0.68 is still unverified, and the V1 note asks the author to check it.
- V3: applied as planned (TODO note above the Chapter 2 outline paragraph).
- V4: adapted. The planned NEW sentence had 32 words, so it was split in two: "we added Gaussian noise to the training labels of the QM9 HOMO--LUMO gap. Published methods for learning with noisy labels did poorly on these labels." Checked against sec:lnl_motivation.
- V5: applied as planned. The TODO was placed under the existing edited-your-text note.
- V6: applied as planned.
- V7: applied as planned. Checked against the Chapter 8 contributions list.
- V8: adapted. "No model's uncertainty identified the corrupted labels" was overbroad. Chapter 3's Conclusions say this only outside censoring, and under censoring, dividing error by uncertainty picked out the clipped labels. So the sentence now reads "Outside clipping, no model's uncertainty identified the corrupted labels.", and the clipping sentence follows it.
- V9: applied as planned.
- V10: adapted. The distance sentence is now limited to "On the same datasets" (the small-molecule datasets), because Chapter 1 says nearest-neighbor prediction was worse in the hybrid on several antibody and reaction datasets. The antibody sentence now follows it and reads "the gain in prediction".
- V11: applied as planned.
- V12: applied as planned.
