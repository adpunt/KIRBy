# Plan: section titles and Results/Conclusions structure, Chapters 1 to 7 (thesis.tex)

Written 2026-10-10 from thesis.tex as it stood then (re-derived; PLAN_titles.md is out of date). Every OLD block below occurs exactly once in thesis.tex (python3 -I, str.count), no two OLD blocks overlap, and applying all of them in order S1 to S74 on a copy kept every OLD unique at the time it is applied, kept every `\label` (same set before and after) and kept every comment line. Chapter 8 is not planned here (another task rewrites it). No `\includegraphics` or `\input` line is touched. No `\paragraph` heading is changed.

## What this plan does

- Chapters 3 to 7 all run Introduction, Methods, Results, Conclusions. Each gets one `\section{Results}` parent. The old results sections become its subsections; their subsections become subsubsections. No fourth heading level is created anywhere.
- Chapter 4 had no Methods or Results. It gets a `\section{Methods}` holding the one subsection that only defines methods ("Distance metrics for continuous molecular descriptors", moved up unchanged), and a `\section{Results}` over its three topic sections.
- Chapter 5 had two methods sections ("Methods: the hybrid search", "Methods: experimental setup"). The first becomes `\section{Methods}`. The second becomes its subsection "Evaluation protocol". The search subsections were not nested under a "The hybrid search" subsection, because two of them already hold subsubsections.
- No separate Discussion. Chapter 7's Discussion heading goes. Its four paragraphs go into the Results subsections they interpret: the labels as delivered (paragraph 3), noise detection (paragraph 2), effect by noise type (paragraph 1), and effect by model and representation (paragraph 4, which already sits there once the heading goes). `\label{sec:noise:discussion}` stays with paragraph 4. Nothing in the thesis refers to it.
- Limitations are paragraphs inside Conclusions, with no heading, in every research chapter. Chapters 3, 4 and 6 already do this (Chapter 3's "The first limitation ...", Chapter 4's "Feature selection cannot fix every flaw", Chapter 6's "Three limitations apply"). Chapters 5 and 7 lose their `\section{Limitations}`. The paragraphs move into Conclusions after the paragraphs that state the findings and before the sentence that points to the next chapter. Each label moves with its paragraphs: `Section~\ref{sec:method:limitations}` in the Chapter 5 Introduction then points to Chapter 5's Conclusions.
- Titles: sentence case throughout; no "Results:" prefix; no questions, claims or clauses as titles; topic noun phrases as in the reference theses (style guide 19: B "Results on the Astex Diverse set", V "Conditioning on internal energy", D "QSAR-Prediction Performance"). Her good short noun-phrase titles are kept. Changed titles go further than PLAN_titles.md where a title was vague or read oddly: "Package description", "Artificial noise conditions", "Uncertainty", "Antibodies and beyond", "Other Uses of a Distance", "What the noise is", "How each group is tested, and QM9 as the clean control", "By Noise Type", "By Dataset", "Model and Representation".
- Every changed heading gets `% TODO: manual review -- title was: <old> -- edited your text: <what and why>` directly above it. The one heading added by an earlier AI session ("Methods for noisy labels", marked CODE INPUT) gets the plain `% TODO: manual review -- title was: <old>`. Every moved block gets a TODO saying where it came from.

## New chapter titles

| Ch | Old | New |
|---|---|---|
| 1 | Introduction | Introduction (kept) |
| 2 | Molecular Representations, Models and Label Noise | Molecular representations, models and label noise |
| 3 | Noise Robustness Across Molecular Representations and Model Architectures | Noise robustness across molecular representations and model architectures |
| 4 | Limitations of the Similarity Assumption | Limitations of the similarity assumption |
| 5 | Building Adaptive Hybrid Representations | Building adaptive hybrid representations |
| 6 | Similarity in an Adaptive Hybrid Representation | Similarity in an adaptive hybrid representation |
| 7 | Label Noise: Detecting It and Doing Something About It | Detecting and handling label noise |

## Old title -> new title, every heading in Chapters 1 to 7

One row per heading, in file order. "Level" is old level, or old -> new. "new sec" rows are headings added by this plan.

| Ch | Level | Old title | New title | Edit |
|---|---|---|---|---|
| 1 | ch | Introduction | Introduction  (kept) |  |
| 1 | sec | Outline of the thesis and contributions | Outline of the thesis and contributions  (kept) |  |
| 1 | sec | Collaborations | Collaborations  (kept) |  |
| 2 | ch | Molecular Representations, Models and Label Noise | Molecular representations, models and label noise | S1 |
| 2 | sec | Overview | Overview  (kept) |  |
| 2 | sec | Molecular representations in QSAR | Molecular representations in QSAR  (kept) |  |
| 2 | sec | Models for molecular property prediction | Models for molecular property prediction  (kept) |  |
| 2 | subsec | Tree ensembles | Tree ensembles  (kept) |  |
| 2 | subsec | Support vector machines | Support vector machines  (kept) |  |
| 2 | subsec | Gaussian processes | Gaussian processes  (kept) |  |
| 2 | subsec | Bayesian neural networks | Bayesian neural networks  (kept) |  |
| 2 | sec | Molecular similarity and activity cliffs | Molecular similarity and activity cliffs  (kept) |  |
| 2 | sec | Noise in molecular data | Noise in molecular data  (kept) |  |
| 2 | subsec | Artificial noise injection | Artificial noise injection  (kept) |  |
| 2 | subsec | Methods for noisy labels | Learning with noisy labels | S2 |
| 2 | sec | NoiseInject: a benchmarking framework | Implementation of the noise conditions | S3 |
| 2 | subsec | Package description | The NoiseInject package | S4 |
| 2 | sec | Evaluation methodology | Evaluation methodology  (kept) |  |
| 2 | sec | Conclusions | Conclusions  (kept) |  |
| 3 | ch | Noise Robustness Across Molecular Representations and Model Architectures | Noise robustness across molecular representations and model architectures | S5 |
| 3 | sec | Introduction | Introduction  (kept) |  |
| 3 | sec | Methods | Methods  (kept) |  |
| 3 | subsec | Datasets | Datasets  (kept) |  |
| 3 | subsec | Representations | Representations  (kept) |  |
| 3 | subsec | Models | Models  (kept) |  |
| 3 | subsec | Noise conditions and levels | Noise conditions and levels  (kept) |  |
| 3 | subsec | QM9 data pipeline | QM9 data pipeline  (kept) |  |
| 3 | subsec | Uncertainty | Uncertainty estimation | S6 |
| 3 | subsec | Evaluation | Evaluation  (kept) |  |
| 3 | new sec | (none) | Results | S7 |
| 3 | sec -> subsec | Variance decomposition | Variance decomposition  (wording kept) | S7 |
| 3 | sec -> subsec | Predictive performance and noise robustness | Predictive performance and noise robustness  (wording kept) | S8 |
| 3 | sec -> subsec | Artificial noise conditions | Noise conditions on the HOMO--LUMO gap | S9 |
| 3 | sec -> subsec | Robustness on the three assay datasets | Robustness on the three assay datasets  (wording kept) | S10 |
| 3 | sec -> subsec | Probabilistic and deterministic counterparts | Probabilistic and deterministic counterparts  (wording kept) | S11 |
| 3 | sec -> subsec | Uncertainty under increasing label noise | Uncertainty under increasing label noise  (wording kept) | S12 |
| 3 | subsec -> subsub | Aleatoric and epistemic attribution | Aleatoric and epistemic attribution  (wording kept) | S13 |
| 3 | subsec -> subsub | Per-sample uncertainty and label noise | Per-sample uncertainty and label noise  (wording kept) | S14 |
| 3 | sec | Conclusions | Conclusions  (kept) |  |
| 4 | ch | Limitations of the Similarity Assumption | Limitations of the similarity assumption | S15 |
| 4 | sec | Introduction | Introduction  (kept) |  |
| 4 | new sec | (none) | Methods | S16 |
| 4 | subsec | Distance metrics for continuous molecular descriptors (moved up from Results) | Distance metrics for continuous molecular descriptors | S16 |
| 4 | new sec | (none) | Results | S16 |
| 4 | sec -> subsec | Flaws in standard representations | Flaws in standard representations  (wording kept) | S16 |
| 4 | subsec -> subsub | Constant and near-constant features | Constant and near-constant features  (wording kept) | S17 |
| 4 | subsec -> subsub | Molecules with identical feature vectors | Molecules with identical feature vectors  (wording kept) | S18 |
| 4 | sec -> subsec | Flaws in similarity and other pairwise computations | Flaws in similarity and other pairwise computations  (wording kept) | S19 |
| 4 | sec -> subsec | Consequences for similarity-based methods | Consequences for similarity-based methods  (wording kept) | S21 |
| 4 | subsec -> subsub | A small sweep of published noise-handling methods | A small sweep of published noise-handling methods  (wording kept) | S22 |
| 4 | subsec -> subsub | Noise detection by neighbors | Noise detection by neighbors  (wording kept) | S23 |
| 4 | sec | Conclusions | Conclusions  (kept) |  |
| 5 | ch | Building Adaptive Hybrid Representations | Building adaptive hybrid representations | S24 |
| 5 | sec | Introduction | Introduction  (kept) |  |
| 5 | subsec | Combining representations | Combining representations  (kept) |  |
| 5 | subsec | The adaptive hybrid representation | The adaptive hybrid representation  (kept) |  |
| 5 | subsec | Blueprints and their chemical interpretation | Blueprints and their chemical interpretation  (kept) |  |
| 5 | subsec | Antibodies and beyond | Applications beyond small molecules | S25 |
| 5 | sec | Methods: the hybrid search | Methods | S26 |
| 5 | subsec | From base representations to a hybrid | From base representations to a hybrid  (kept) |  |
| 5 | subsec | The search space | The search space  (kept) |  |
| 5 | subsub | Feature importance scoring | Feature importance scoring  (kept) |  |
| 5 | subsub | Feature allocation methods | Feature allocation methods  (kept) |  |
| 5 | subsub | Search configuration and internal evaluation | Search configuration and internal evaluation  (kept) |  |
| 5 | subsec | Augmentation | Augmentation  (kept) |  |
| 5 | subsec | Search grid per dataset | Search grid per dataset  (kept) |  |
| 5 | subsub | Adaptive pruning | Adaptive pruning  (kept) |  |
| 5 | sec -> subsec | Methods: experimental setup | Evaluation protocol | S27 |
| 5 | subsec | Datasets and splitting | Datasets and splitting  (kept) |  |
| 5 | subsec | Representations by domain | Representations by domain  (kept) |  |
| 5 | subsec | Models and metrics | Models and metrics  (kept) |  |
| 5 | subsec | Comparators, controls and ablations | Comparators, controls and ablations  (kept) |  |
| 5 | subsec | Transfer, collaboration studies and separate sites | Transfer, collaboration studies and separate sites  (kept) |  |
| 5 | new sec | (none) | Results | S28 |
| 5 | sec -> subsec | Results: when the hybrid helps | Performance of the hybrid | S28 |
| 5 | subsec -> subsub | Against the best single representation | Against the best single representation  (wording kept) | S29 |
| 5 | subsec -> subsub | Against ensembles, and many representations against few | Against ensembles, and many representations against few  (wording kept) | S30 |
| 5 | subsec -> subsub | Does the search overfit? | Overfitting of the search | S31 |
| 5 | sec -> subsec | Results: why the hybrid helps | Sources of the gain | S32 |
| 5 | subsec -> subsub | Selection, not only fewer features | Selection at a fixed number of features | S33 |
| 5 | subsec -> subsub | Different models want different hybrids | Dependence on the downstream model | S34 |
| 5 | subsec -> subsub | Using importance during training | Using importance during training  (wording kept) | S35 |
| 5 | sec -> subsec | Results: blueprints on specific datasets | Blueprints on specific datasets | S36 |
| 5 | sec -> subsec | Results: collaboration studies | Collaboration studies | S37 |
| 5 | sec -> subsec | Results: features chosen across separate sites | Features chosen across separate sites | S38 |
| 5 | sec | Limitations | (heading removed; its five paragraphs move into Conclusions, with the label) | S39, S40 |
| 5 | sec | Conclusions | Conclusions  (kept) |  |
| 6 | ch | Similarity in an Adaptive Hybrid Representation | Similarity in an adaptive hybrid representation | S41 |
| 6 | sec | Introduction | Introduction  (kept) |  |
| 6 | sec | Methods | Methods  (kept) |  |
| 6 | subsec | Feature Spaces Compared | Feature spaces compared | S42 |
| 6 | subsec | Similarity in a Hybrid | Similarity in a hybrid | S43 |
| 6 | subsec | Nearest-Neighbor Settings | Nearest-neighbor settings | S44 |
| 6 | subsec | Experiments | Experiments  (kept) |  |
| 6 | new sec | (none) | Results | S45 |
| 6 | sec -> subsec | Results: Does Distance in a Hybrid Track Activity Better | Distance against activity difference | S45 |
| 6 | sec -> subsec | Results: Prediction from Nearest Neighbors | Prediction from nearest neighbors | S46 |
| 6 | sec -> subsec | Results: Activity Cliffs | Activity cliffs | S47 |
| 6 | sec -> subsec | Results: Other Uses of a Distance | Prediction error, retrieval and clustering | S48 |
| 6 | sec | Conclusions | Conclusions  (kept) |  |
| 7 | ch | Label Noise: Detecting It and Doing Something About It | Detecting and handling label noise | S49 |
| 7 | sec | Introduction | Introduction  (kept) |  |
| 7 | sec | Methods | Methods  (kept) |  |
| 7 | subsec | What the noise is | Noise conditions | S50 |
| 7 | subsec | Noise detection | Noise detection  (kept) |  |
| 7 | subsec | Noise mitigation | Noise mitigation  (kept) |  |
| 7 | subsec | Noise-robust training | Noise-robust training  (kept) |  |
| 7 | subsec | How each group is tested, and QM9 as the clean control | Evaluation and controls | S51 |
| 7 | new sec | (none) | Results | S52 |
| 7 | sec -> subsec | Results: Does Noise Handling Work at All? | Overall effect of noise handling | S52 |
| 7 | sec -> subsec | Results: Changing a Label Against Deleting a Molecule | Changing a label against deleting a molecule | S53 |
| 7 | subsec -> subsub | The labels as delivered | The labels as delivered  (wording kept) | S54 |
| 7 | sec -> subsec | Results: Which Detectors Find the Suspect Labels | Noise detection | S56 |
| 7 | sec -> subsec | Results: By Noise Type | Effect by noise type | S58 |
| 7 | subsec -> subsub | Labels clipped at an assay limit | Labels clipped at an assay limit  (wording kept) | S59 |
| 7 | subsec -> subsub | A shared offset per scaffold group | A shared offset per scaffold group  (wording kept) | S60 |
| 7 | subsec -> subsub | Several sources per molecule | Several sources per molecule  (wording kept) | S61 |
| 7 | subsec -> subsub | Gaussian noise, outliers and shifts in a few labels | Gaussian noise, outliers and shifts in a few labels  (wording kept) | S62 |
| 7 | sec -> subsec | Results: By Dataset | Effect by dataset | S64 |
| 7 | subsec -> subsub | The antibody datasets | The antibody datasets  (wording kept) | S65 |
| 7 | subsec -> subsub | Small molecules and the reaction set | Small molecules and the reaction set  (wording kept) | S66 |
| 7 | subsec -> subsub | Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} | Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}  (wording kept) | S67 |
| 7 | sec -> subsec | Results: Cleaning Inside the Feature Search | Cleaning inside the feature search | S68 |
| 7 | sec -> subsec | Results: Changing the Model Instead of the Labels | Changing the model instead of the labels | S69 |
| 7 | subsec -> subsub | The expanded published sweep | The expanded published sweep  (wording kept) | S70 |
| 7 | sec -> subsec | Results: Model and Representation | Effect by model and representation | S71 |
| 7 | sec | Discussion | (heading removed; paragraphs 1-3 move into Results, paragraph 4 and the label stay at the end of "Effect by model and representation") | S72, S55, S57, S63 |
| 7 | sec | Limitations | (heading removed; its paragraphs become the end of Conclusions, with the label) | S73 |
| 7 | sec | Conclusion | Conclusions (moved above the limitation paragraphs) | S73, S74 |

## How to apply

- Apply in order S1 to S74 with the Edit tool (exact string replacement), re-reading before each edit. Other sessions edit thesis.tex: if an OLD block no longer matches, re-read and rebuild that edit from the current text.
- Moves are cut-and-paste of whole paragraphs. The OLD/NEW blocks carry the text as it stood on 2026-10-10. PLAN_ch5.md (E72 and others) and PLAN_ch7.md (E52-E54, the conclusion paragraph edit) also edit some of these paragraphs. If those plans are applied first, the moved text must be the paragraph as it then stands, not the copy here. If this plan is applied first, those plans' OLD blocks still match, except PLAN_ch7's edit of the whole Chapter 7 conclusion paragraph: S73/S74 split off its last sentence ("Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis."), so that edit must be applied to the first part only.
- S20 (cut) must come after S16 (paste) or before it; both orders work, because S20's OLD starts at the end of the table above the old position, which the pasted copy does not have.
- PLAN_ch7.md's lead-in edits are anchored on `\label{...}` at the end of heading lines, so they still apply after the heading changes here.

## For the author (not changed)

- Chapter 4's Methods holds only the distance definitions. The set-up of each measurement (the activity-cliff rule, distance against label difference, agreement on neighbors, the MCS comparison, the sweep's retention measure, the neighbor-detection study) stays in its Results paragraph, because each of those paragraphs also carries the numbers. Splitting them would rewrite your text; say if you want it.
- Chapter 5: "Every number in this section is a held-out score" now opens the subsection "Performance of the hybrid", though it applies to all of Results. Moving it up to sit directly under `\section{Results}` would make "this section" mean all results.
- Chapter 7: the moved Discussion paragraph 1 overlaps the paragraph above its new place ("Set side by side, the damage kinds sort by how much the treatment knows"). The prose pass should merge them.
- Chapter 7 Results now has a subsection "Noise detection" and Methods has one too. That is deliberate (results named after the methods subsection they report), but say if you prefer "Detection of damaged labels".
- Chapter 6: "Distance against activity difference" keeps "activity", as in the Methods paragraph "Distance against activity." and the experiments table. The datasets include yield, expression and stability, where "activity" is not the right word (chemistry guide item 17). "Distance against label difference" in all three places would fix it.
- Chapter 2's empty `\section{Overview}` is kept; PLAN_ch2.md E1 writes its text.
- A Chapter 1 notes comment names Chapter 4 "The flaws of the similarity assumption"; the title stays "Limitations of the similarity assumption". Comments are not edited.
- Comments in Chapter 6 give typed section numbers ("Sections 6.3-6.6", "Section 6.6"). After the new Results parent these are 6.3.1 to 6.3.4. Comments are not edited.
- THESIS_OUTLINE.md still uses the old titles and the old Chapter 7 Discussion and Limitations sections.

## Edits

### S1
- file: thesis.tex
- chapter: 2
- kind: title
- result: `\chapter{Molecular representations, models and label noise}\label{chap:background}\label{sec2}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\chapter{Molecular Representations, Models and Label Noise}\label{chap:background}\label{sec2}
```
NEW:
```latex
% TODO: manual review -- title was: Molecular Representations, Models and Label Noise -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Molecular representations, models and label noise}\label{chap:background}\label{sec2}
```

### S2
- file: thesis.tex
- chapter: 2
- kind: title
- result: `\subsection{Learning with noisy labels}\label{sec:lnl_background}`
- why: the CODE INPUT comment above asks for this title to be reviewed; learning with noisy labels (LNL) is the name of the problem this subsection reviews

OLD:
```latex
\subsection{Methods for noisy labels}\label{sec:lnl_background}
```
NEW:
```latex
% TODO: manual review -- title was: Methods for noisy labels
\subsection{Learning with noisy labels}\label{sec:lnl_background}
```

### S3
- file: thesis.tex
- chapter: 2
- kind: title
- result: `\section{Implementation of the noise conditions}`
- why: colon title; the section text says the noise conditions are implemented twice (NoiseInject and the QM9 pipeline) and checked against each other, and NoiseInject now names the subsection below

OLD:
```latex
\section{NoiseInject: a benchmarking framework}
```
NEW:
```latex
% TODO: manual review -- title was: NoiseInject: a benchmarking framework -- edited your text: colon title; the section text says the noise conditions are implemented twice (NoiseInject and the QM9 pipeline) and checked against each other, and NoiseInject now names the subsection below
\section{Implementation of the noise conditions}
```

### S4
- file: thesis.tex
- chapter: 2
- kind: title
- result: `\subsection{The NoiseInject package}\label{sec:noiseinject_pkg} % CODE INPUT: label added`
- why: "Package description" did not say which package; the name moved here from the parent section title

OLD:
```latex
\subsection{Package description}\label{sec:noiseinject_pkg} % CODE INPUT: label added
```
NEW:
```latex
% TODO: manual review -- title was: Package description -- edited your text: "Package description" did not say which package; the name moved here from the parent section title
\subsection{The NoiseInject package}\label{sec:noiseinject_pkg} % CODE INPUT: label added
```

### S5
- file: thesis.tex
- chapter: 3
- kind: title
- result: `\chapter{Noise robustness across molecular representations and model architectures}\label{chap:robustness}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\chapter{Noise Robustness Across Molecular Representations and Model Architectures}\label{chap:robustness}
```
NEW:
```latex
% TODO: manual review -- title was: Noise Robustness Across Molecular Representations and Model Architectures -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Noise robustness across molecular representations and model architectures}\label{chap:robustness}
```

### S6
- file: thesis.tex
- chapter: 3
- kind: title
- result: `\subsection{Uncertainty estimation}`
- why: one word did not say what the subsection holds: how each model produces its uncertainty estimate

OLD:
```latex
\subsection{Uncertainty}

```
NEW:
```latex
% TODO: manual review -- title was: Uncertainty -- edited your text: one word did not say what the subsection holds: how each model produces its uncertainty estimate
\subsection{Uncertainty estimation}

```

### S7
- file: thesis.tex
- chapter: 3
- kind: new Results section + level
- result: `\subsection{Variance decomposition}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Variance decomposition}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (chapter order Introduction, Methods, Results, Conclusions; reference theses put topics under one Results section, style guide 19)
\section{Results}

% TODO: manual review -- title was: Variance decomposition -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Variance decomposition}
```

### S8
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsection{Predictive performance and noise robustness}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Predictive performance and noise robustness}
```
NEW:
```latex
% TODO: manual review -- title was: Predictive performance and noise robustness -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Predictive performance and noise robustness}
```

### S9
- file: thesis.tex
- chapter: 3
- kind: title
- result: `\subsection{Noise conditions on the HOMO--LUMO gap}`
- why: now a subsection under the new Results section; every condition in this chapter is artificial, so the old title did not separate this section from the next one ("Robustness on the three assay datasets"); this one compares the conditions on the QM9 HOMO--LUMO gap

OLD:
```latex
\section{Artificial noise conditions}
```
NEW:
```latex
% TODO: manual review -- title was: Artificial noise conditions -- edited your text: now a subsection under the new Results section; every condition in this chapter is artificial, so the old title did not separate this section from the next one ("Robustness on the three assay datasets"); this one compares the conditions on the QM9 HOMO--LUMO gap
\subsection{Noise conditions on the HOMO--LUMO gap}
```

### S10
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsection{Robustness on the three assay datasets}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Robustness on the three assay datasets}
```
NEW:
```latex
% TODO: manual review -- title was: Robustness on the three assay datasets -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Robustness on the three assay datasets}
```

### S11
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsection{Probabilistic and deterministic counterparts}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Probabilistic and deterministic counterparts}
```
NEW:
```latex
% TODO: manual review -- title was: Probabilistic and deterministic counterparts -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Probabilistic and deterministic counterparts}
```

### S12
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsection{Uncertainty under increasing label noise}\label{sec:uq}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Uncertainty under increasing label noise}\label{sec:uq}
```
NEW:
```latex
% TODO: manual review -- title was: Uncertainty under increasing label noise -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Uncertainty under increasing label noise}\label{sec:uq}
```

### S13
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsubsection{Aleatoric and epistemic attribution}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Aleatoric and epistemic attribution}
```
NEW:
```latex
% TODO: manual review -- title was: Aleatoric and epistemic attribution -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Aleatoric and epistemic attribution}
```

### S14
- file: thesis.tex
- chapter: 3
- kind: level
- result: `\subsubsection{Per-sample uncertainty and label noise}\label{sec:uq_per_sample}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Per-sample uncertainty and label noise}\label{sec:uq_per_sample}
```
NEW:
```latex
% TODO: manual review -- title was: Per-sample uncertainty and label noise -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Per-sample uncertainty and label noise}\label{sec:uq_per_sample}
```

### S15
- file: thesis.tex
- chapter: 4
- kind: title
- result: `\chapter{Limitations of the similarity assumption}\label{chap:flaws}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\chapter{Limitations of the Similarity Assumption}\label{chap:flaws}
```
NEW:
```latex
% TODO: manual review -- title was: Limitations of the Similarity Assumption -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Limitations of the similarity assumption}\label{chap:flaws}
```

### S16
- file: thesis.tex
- chapter: 4
- kind: new Methods section (moved text) + new Results section + level
- result: `\subsection{Flaws in standard representations}\label{sec:flaws:representations}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Flaws in standard representations}\label{sec:flaws:representations}
```
NEW:
```latex
% TODO: manual review -- new section: Chapter 4 now runs Introduction, Methods, Results, Conclusions like Chapters 3, 5, 6 and 7. It holds only the distance definitions moved from the old Section sec:flaws:similarity. The set-up of each measurement (cliff rule, distance against label difference, neighbor agreement, MCS) stays in the Results paragraphs, because each of those paragraphs also carries its numbers.
\section{Methods}

% TODO: manual review -- edited your text: subsection moved here unchanged from the end of "Flaws in similarity and other pairwise computations", because it defines the distances the chapter uses (methods, not a result)
\subsection{Distance metrics for continuous molecular descriptors}


While binary fingerprints are compared with the Tanimoto coefficient (Chapter~\ref{chap:background}), we need distance metrics for continuous feature vectors. The research in Section~\ref{sec:lnl_motivation} used the following four:

\begin{itemize}
    \item Euclidean ($L_2$): standard geometric distance, $d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$
    \item Manhattan ($L_1$): sum of absolute differences, $d(x, y) = \sum_i |x_i - y_i|$, more robust to outlier features
    \item Cosine: angle-based, $d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$, invariant to magnitude
    \item Mahalanobis: covariance-adjusted distance, $d(x,y)=\sqrt{(x-y)^\top S^{-1}(x-y)}$, where $S$ is the feature covariance estimated on the training molecules \citep{Mahalanobis1936}
\end{itemize}

All features are standardized to zero mean and unit variance prior to distance computation.

% TODO: review, needs manual review -- rewritten 2026-10-08 to match the code: the similarity objective uses Manhattan only (src/kirby/distance_config.py default); Chapter 6 sweeps Manhattan, cosine and Minkowski p=0.5 up to 5,000 molecules, Manhattan and cosine above (tests/experiments/noise/similarity_space.py GROUP_METRICS); Euclidean was cut 2026-10-05; importance weighting was cut from the noise suite but survives in some similarity experiments, so the old weighting sentence was removed here
The later chapters use a different set, chosen after that study. The hybrid search's nearest-neighbor objective (Chapter~\ref{chap:method}) uses Manhattan distance only. Chapter~\ref{chap:similarity} compares Manhattan, cosine and Minkowski distance with $p=0.5$, $d(x,y)=\left(\sum_i |x_i-y_i|^{0.5}\right)^{2}$, on datasets of up to 5,000 molecules, and Manhattan and cosine on larger ones. The Minkowski distance with $p=0.5$ is a dissimilarity rather than a metric, because it breaks the triangle inequality.


% TODO: manual review -- new parent section: the results sections that follow are now its subsections (chapter order Introduction, Methods, Results, Conclusions; reference theses put topics under one Results section, style guide 19)
\section{Results}

% TODO: manual review -- title was: Flaws in standard representations -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Flaws in standard representations}\label{sec:flaws:representations}
```

### S17
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsubsection{Constant and near-constant features}\label{sec:flaws:constant}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Constant and near-constant features}\label{sec:flaws:constant}
```
NEW:
```latex
% TODO: manual review -- title was: Constant and near-constant features -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Constant and near-constant features}\label{sec:flaws:constant}
```

### S18
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsubsection{Molecules with identical feature vectors}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Molecules with identical feature vectors}
```
NEW:
```latex
% TODO: manual review -- title was: Molecules with identical feature vectors -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Molecules with identical feature vectors}
```

### S19
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsection{Flaws in similarity and other pairwise computations}\label{sec:flaws:similarity}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Flaws in similarity and other pairwise computations}\label{sec:flaws:similarity}
```
NEW:
```latex
% TODO: manual review -- title was: Flaws in similarity and other pairwise computations -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Flaws in similarity and other pairwise computations}\label{sec:flaws:similarity}
```

### S20
- file: thesis.tex
- chapter: 4
- kind: move (cut)
- result: ``
- why: the subsection is pasted into the new Methods section (the edit that adds "\section{Methods}" in Chapter 4); its text and comments move unchanged. The OLD block starts at the end of the table above so that it differs from the pasted copy

OLD:
```latex
\label{tab:flaws:neighbor_overlap}
\end{table}
% ==== END CODE INPUT ====


\subsection{Distance metrics for continuous molecular descriptors}


While binary fingerprints are compared with the Tanimoto coefficient (Chapter~\ref{chap:background}), we need distance metrics for continuous feature vectors. The research in Section~\ref{sec:lnl_motivation} used the following four:

\begin{itemize}
    \item Euclidean ($L_2$): standard geometric distance, $d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$
    \item Manhattan ($L_1$): sum of absolute differences, $d(x, y) = \sum_i |x_i - y_i|$, more robust to outlier features
    \item Cosine: angle-based, $d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$, invariant to magnitude
    \item Mahalanobis: covariance-adjusted distance, $d(x,y)=\sqrt{(x-y)^\top S^{-1}(x-y)}$, where $S$ is the feature covariance estimated on the training molecules \citep{Mahalanobis1936}
\end{itemize}

All features are standardized to zero mean and unit variance prior to distance computation.

% TODO: review, needs manual review -- rewritten 2026-10-08 to match the code: the similarity objective uses Manhattan only (src/kirby/distance_config.py default); Chapter 6 sweeps Manhattan, cosine and Minkowski p=0.5 up to 5,000 molecules, Manhattan and cosine above (tests/experiments/noise/similarity_space.py GROUP_METRICS); Euclidean was cut 2026-10-05; importance weighting was cut from the noise suite but survives in some similarity experiments, so the old weighting sentence was removed here
The later chapters use a different set, chosen after that study. The hybrid search's nearest-neighbor objective (Chapter~\ref{chap:method}) uses Manhattan distance only. Chapter~\ref{chap:similarity} compares Manhattan, cosine and Minkowski distance with $p=0.5$, $d(x,y)=\left(\sum_i |x_i-y_i|^{0.5}\right)^{2}$, on datasets of up to 5,000 molecules, and Manhattan and cosine on larger ones. The Minkowski distance with $p=0.5$ is a dissimilarity rather than a metric, because it breaks the triangle inequality.

```
NEW:
```latex
\label{tab:flaws:neighbor_overlap}
\end{table}
% ==== END CODE INPUT ====

```

### S21
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsection{Consequences for similarity-based methods}\label{sec:flaws:consequences}`
- why: heading level only (section, now subsection), under the new Results section; wording unchanged

OLD:
```latex
\section{Consequences for similarity-based methods}\label{sec:flaws:consequences}
```
NEW:
```latex
% TODO: manual review -- title was: Consequences for similarity-based methods -- edited your text: heading level only (section, now subsection), under the new Results section; wording unchanged
\subsection{Consequences for similarity-based methods}\label{sec:flaws:consequences}
```

### S22
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsubsection{A small sweep of published noise-handling methods}\label{sec:lnl_motivation}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{A small sweep of published noise-handling methods}\label{sec:lnl_motivation}
```
NEW:
```latex
% TODO: manual review -- title was: A small sweep of published noise-handling methods -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{A small sweep of published noise-handling methods}\label{sec:lnl_motivation}
```

### S23
- file: thesis.tex
- chapter: 4
- kind: level
- result: `\subsubsection{Noise detection by neighbors}\label{sec:flaws:neighbors} % CODE INPUT: label added, needs review`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Noise detection by neighbors}\label{sec:flaws:neighbors} % CODE INPUT: label added, needs review
```
NEW:
```latex
% TODO: manual review -- title was: Noise detection by neighbors -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Noise detection by neighbors}\label{sec:flaws:neighbors} % CODE INPUT: label added, needs review
```

### S24
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\chapter{Building adaptive hybrid representations}\label{chap:method}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\chapter{Building Adaptive Hybrid Representations}\label{chap:method}
```
NEW:
```latex
% TODO: manual review -- title was: Building Adaptive Hybrid Representations -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Building adaptive hybrid representations}\label{chap:method}
```

### S25
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Applications beyond small molecules}`
- why: "and beyond" was vague; the subsection says the search applies to any set of feature matrices for the same samples, antibodies being one case

OLD:
```latex
\subsection{Antibodies and beyond}
```
NEW:
```latex
% TODO: manual review -- title was: Antibodies and beyond -- edited your text: "and beyond" was vague; the subsection says the search applies to any set of feature matrices for the same samples, antibodies being one case
\subsection{Applications beyond small molecules}
```

### S26
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\section{Methods}`
- why: one Methods section per chapter (Introduction, Methods, Results, Conclusions). The search subsections that follow stay subsections. The old second methods section "Methods: experimental setup" becomes a subsection of this one. A Methods > The hybrid search > subsection nesting was not used, because "The search space" and "Search grid per dataset" already hold subsubsections and a fourth level would follow

OLD:
```latex
\section{Methods: the hybrid search}
```
NEW:
```latex
% TODO: manual review -- title was: Methods: the hybrid search -- edited your text: one Methods section per chapter (Introduction, Methods, Results, Conclusions). The search subsections that follow stay subsections. The old second methods section "Methods: experimental setup" becomes a subsection of this one. A Methods > The hybrid search > subsection nesting was not used, because "The search space" and "Search grid per dataset" already hold subsubsections and a fourth level would follow
\section{Methods}
```

### S27
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Evaluation protocol}`
- why: now a subsection of the single Methods section (section, now subsection). Its text says how every experiment is split into outer folds and how the single-representation comparator is chosen and scored, which is an evaluation protocol. The datasets, representations and models that follow are its sibling subsections

OLD:
```latex
\section{Methods: experimental setup}
```
NEW:
```latex
% TODO: manual review -- title was: Methods: experimental setup -- edited your text: now a subsection of the single Methods section (section, now subsection). Its text says how every experiment is split into outer folds and how the single-representation comparator is chosen and scored, which is an evaluation protocol. The datasets, representations and models that follow are its sibling subsections
\subsection{Evaluation protocol}
```

### S28
- file: thesis.tex
- chapter: 5
- kind: new Results section + title
- result: `\subsection{Performance of the hybrid}\label{sec:method:results_single}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; the subsections compare the hybrid with the best single representation and with ensembles, and check the search for overfitting, so the title names that topic rather than a claim

OLD:
```latex
\section{Results: when the hybrid helps}\label{sec:method:results_single}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (chapter order Introduction, Methods, Results, Conclusions; reference theses put topics under one Results section, style guide 19)
\section{Results}

% TODO: manual review -- title was: Results: when the hybrid helps -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; the subsections compare the hybrid with the best single representation and with ensembles, and check the search for overfitting, so the title names that topic rather than a claim
\subsection{Performance of the hybrid}\label{sec:method:results_single}
```

### S29
- file: thesis.tex
- chapter: 5
- kind: level
- result: `\subsubsection{Against the best single representation}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Against the best single representation}
```
NEW:
```latex
% TODO: manual review -- title was: Against the best single representation -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Against the best single representation}
```

### S30
- file: thesis.tex
- chapter: 5
- kind: level
- result: `\subsubsection{Against ensembles, and many representations against few}\label{sec:method:results_ensembles}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Against ensembles, and many representations against few}\label{sec:method:results_ensembles}
```
NEW:
```latex
% TODO: manual review -- title was: Against ensembles, and many representations against few -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Against ensembles, and many representations against few}\label{sec:method:results_ensembles}
```

### S31
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsubsection{Overfitting of the search}\label{sec:method:results_overfit}`
- why: subsection, now subsubsection; question title made a topic (none of the reference theses uses a question, style guide 19)

OLD:
```latex
\subsection{Does the search overfit?}\label{sec:method:results_overfit}
```
NEW:
```latex
% TODO: manual review -- title was: Does the search overfit? -- edited your text: subsection, now subsubsection; question title made a topic (none of the reference theses uses a question, style guide 19)
\subsubsection{Overfitting of the search}\label{sec:method:results_overfit}
```

### S32
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Sources of the gain}\label{sec:method:results_why}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped and the "why" clause made a topic

OLD:
```latex
\section{Results: why the hybrid helps}\label{sec:method:results_why}
```
NEW:
```latex
% TODO: manual review -- title was: Results: why the hybrid helps -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped and the "why" clause made a topic
\subsection{Sources of the gain}\label{sec:method:results_why}
```

### S33
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsubsection{Selection at a fixed number of features}`
- why: subsection, now subsubsection; the "not only" contrast made a topic. The subsection compares the hybrid with recursive elimination and random features at the same number of features

OLD:
```latex
\subsection{Selection, not only fewer features}
```
NEW:
```latex
% TODO: manual review -- title was: Selection, not only fewer features -- edited your text: subsection, now subsubsection; the "not only" contrast made a topic. The subsection compares the hybrid with recursive elimination and random features at the same number of features
\subsubsection{Selection at a fixed number of features}
```

### S34
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsubsection{Dependence on the downstream model}`
- why: subsection, now subsubsection; claim title made a topic. The subsection reports how the number of features picked and the score depend on the downstream model

OLD:
```latex
\subsection{Different models want different hybrids}
```
NEW:
```latex
% TODO: manual review -- title was: Different models want different hybrids -- edited your text: subsection, now subsubsection; claim title made a topic. The subsection reports how the number of features picked and the score depend on the downstream model
\subsubsection{Dependence on the downstream model}
```

### S35
- file: thesis.tex
- chapter: 5
- kind: level
- result: `\subsubsection{Using importance during training}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Using importance during training}
```
NEW:
```latex
% TODO: manual review -- title was: Using importance during training -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Using importance during training}
```

### S36
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Blueprints on specific datasets}\label{sec:method:results_blueprints}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; wording kept

OLD:
```latex
\section{Results: blueprints on specific datasets}\label{sec:method:results_blueprints}
```
NEW:
```latex
% TODO: manual review -- title was: Results: blueprints on specific datasets -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; wording kept
\subsection{Blueprints on specific datasets}\label{sec:method:results_blueprints}
```

### S37
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Collaboration studies}\label{sec:method:results_opig}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: collaboration studies}\label{sec:method:results_opig}
```
NEW:
```latex
% TODO: manual review -- title was: Results: collaboration studies -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Collaboration studies}\label{sec:method:results_opig}
```

### S38
- file: thesis.tex
- chapter: 5
- kind: title
- result: `\subsection{Features chosen across separate sites}\label{sec:method:results_sites}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; wording kept

OLD:
```latex
\section{Results: features chosen across separate sites}\label{sec:method:results_sites}
```
NEW:
```latex
% TODO: manual review -- title was: Results: features chosen across separate sites -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; wording kept
\subsection{Features chosen across separate sites}\label{sec:method:results_sites}
```

### S39
- file: thesis.tex
- chapter: 5
- kind: move (cut) + heading removed
- result: `\section{Conclusions}`
- why: no separate Limitations section in Chapters 3-7: limitations are paragraphs inside Conclusions, as Chapters 3, 4 and 6 already have them. The five paragraphs are pasted unchanged into Conclusions by the next edit, with the label

OLD:
```latex
\section{Limitations}\label{sec:method:limitations}

% TODO: manual review
The hybrid needs data. Its gains are consistent on datasets of a few thousand samples and inconsistent on datasets of a few hundred, where the search's internal scores overstate what the chosen hybrid delivers by up to 0.16 Spearman (Section~\ref{sec:method:results_overfit}) and its prediction intervals cover too few held-out samples. Most of the antibody datasets in this chapter are of that size, so the antibody results say as much about dataset size as about antibodies, and the two cannot be separated with these datasets. Whether the best hybrid settings follow measurable properties of a dataset is the question of the experiment still to be reported in Section~\ref{sec:method:results_single}.

% TODO: manual review
The hybrid can only select from what the pool offers. The antibody pools are mostly protein and antibody language models of the same sequence, several of them variants of ESM-2, so the antibody hybrids had few representations of different kinds to combine. Frozen embeddings also set a ceiling: no signal can be added beyond what the inputs contain. Integrating many foundation models, each with its own dependencies and hardware, is difficult engineering, and the burden grows with the number of models.

% TODO: manual review
The search is costly and must be rerun when the data, the representations or the downstream model change. The grid of each dataset was cut to save compute (Section~\ref{sec:method:decided_grid}), so the hybrids reported here are at worst less informative than a full search would find. The settings found on one dataset carry over to a related one only partly, and we tested this on one pair of datasets. A downstream model from outside the fixed list of model families gets no matched evaluator. \textbf{[RESULT NEEDED: hours per dataset for the full search, and the hardware]}

% TODO: manual review
A simpler selection is sometimes enough. Recursive feature elimination over all representations joined beat the hybrid on binding free energy with the SVM and on CHO polyreactivity at every budget, and on hERG $K_i$ with the DNN at the larger budgets. On small molecules, tuning the SVM on a wider hybrid did better than a smaller one. Where one of these is enough, the search costs more than it returns.

% TODO: manual review -- RERUN-SWAP-DNN
Several results are provisional. Most DNN results were computed before the correction of 8 October 2026, the Gaussian-process results will change when that model is replaced, and the comparison with PCA at the same number of features waits for the DNN re-runs. The experiment on adding a new embedding ran on 2,000 molecules of LogD and hERG blocking, and its random control swaps features without a new search. The separate sites are simulated from one public dataset, so the differences between real sites in curation and chemistry are not tested, and only Caco-2 has finished the experiment with flipped labels.

\section{Conclusions}
```
NEW:
```latex
\section{Conclusions}
```

### S40
- file: thesis.tex
- chapter: 5
- kind: move (paste)
- result: `This chapter used the hybrid only as input to a model.`
- why: the limitation paragraphs go after the paragraphs that answer the chapter questions and before the paragraph that points to Chapter 6; the label sec:method:limitations moves with them, so Section~\ref{sec:method:limitations} in the Introduction now points to Conclusions

OLD:
```latex
% TODO: manual review
This chapter used the hybrid only as input to a model.
```
NEW:
```latex
% TODO: manual review -- the Limitations section heading was removed (limitations are now paragraphs inside Conclusions, as in Chapters 3, 4 and 6); the five paragraphs below moved here unchanged, with the label
\label{sec:method:limitations}
% TODO: manual review
The hybrid needs data. Its gains are consistent on datasets of a few thousand samples and inconsistent on datasets of a few hundred, where the search's internal scores overstate what the chosen hybrid delivers by up to 0.16 Spearman (Section~\ref{sec:method:results_overfit}) and its prediction intervals cover too few held-out samples. Most of the antibody datasets in this chapter are of that size, so the antibody results say as much about dataset size as about antibodies, and the two cannot be separated with these datasets. Whether the best hybrid settings follow measurable properties of a dataset is the question of the experiment still to be reported in Section~\ref{sec:method:results_single}.

% TODO: manual review
The hybrid can only select from what the pool offers. The antibody pools are mostly protein and antibody language models of the same sequence, several of them variants of ESM-2, so the antibody hybrids had few representations of different kinds to combine. Frozen embeddings also set a ceiling: no signal can be added beyond what the inputs contain. Integrating many foundation models, each with its own dependencies and hardware, is difficult engineering, and the burden grows with the number of models.

% TODO: manual review
The search is costly and must be rerun when the data, the representations or the downstream model change. The grid of each dataset was cut to save compute (Section~\ref{sec:method:decided_grid}), so the hybrids reported here are at worst less informative than a full search would find. The settings found on one dataset carry over to a related one only partly, and we tested this on one pair of datasets. A downstream model from outside the fixed list of model families gets no matched evaluator. \textbf{[RESULT NEEDED: hours per dataset for the full search, and the hardware]}

% TODO: manual review
A simpler selection is sometimes enough. Recursive feature elimination over all representations joined beat the hybrid on binding free energy with the SVM and on CHO polyreactivity at every budget, and on hERG $K_i$ with the DNN at the larger budgets. On small molecules, tuning the SVM on a wider hybrid did better than a smaller one. Where one of these is enough, the search costs more than it returns.

% TODO: manual review -- RERUN-SWAP-DNN
Several results are provisional. Most DNN results were computed before the correction of 8 October 2026, the Gaussian-process results will change when that model is replaced, and the comparison with PCA at the same number of features waits for the DNN re-runs. The experiment on adding a new embedding ran on 2,000 molecules of LogD and hERG blocking, and its random control swaps features without a new search. The separate sites are simulated from one public dataset, so the differences between real sites in curation and chemistry are not tested, and only Caco-2 has finished the experiment with flipped labels.

% TODO: manual review
This chapter used the hybrid only as input to a model.
```

### S41
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\chapter{Similarity in an adaptive hybrid representation}\label{chap:similarity}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\chapter{Similarity in an Adaptive Hybrid Representation}\label{chap:similarity}
```
NEW:
```latex
% TODO: manual review -- title was: Similarity in an Adaptive Hybrid Representation -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\chapter{Similarity in an adaptive hybrid representation}\label{chap:similarity}
```

### S42
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Feature spaces compared}\label{sec:similarity:spaces}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\subsection{Feature Spaces Compared}\label{sec:similarity:spaces}
```
NEW:
```latex
% TODO: manual review -- title was: Feature Spaces Compared -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\subsection{Feature spaces compared}\label{sec:similarity:spaces}
```

### S43
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Similarity in a hybrid}\label{sec:similarity:hybrid_similarity}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19); Chapter 4 already quotes it as ``Similarity in a hybrid''

OLD:
```latex
\subsection{Similarity in a Hybrid}\label{sec:similarity:hybrid_similarity}
```
NEW:
```latex
% TODO: manual review -- title was: Similarity in a Hybrid -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19); Chapter 4 already quotes it as ``Similarity in a hybrid''
\subsection{Similarity in a hybrid}\label{sec:similarity:hybrid_similarity}
```

### S44
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Nearest-neighbor settings}\label{sec:similarity:knn_settings}`
- why: sentence case, as in the reference theses and the other chapters (style guide 19)

OLD:
```latex
\subsection{Nearest-Neighbor Settings}\label{sec:similarity:knn_settings}
```
NEW:
```latex
% TODO: manual review -- title was: Nearest-Neighbor Settings -- edited your text: sentence case, as in the reference theses and the other chapters (style guide 19)
\subsection{Nearest-neighbor settings}\label{sec:similarity:knn_settings}
```

### S45
- file: thesis.tex
- chapter: 6
- kind: new Results section + title
- result: `\subsection{Distance against activity difference}\label{sec:similarity:results_similarity}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; question (with no question mark) made a topic, in the words of the matching Methods paragraph "Distance against activity."

OLD:
```latex
\section{Results: Does Distance in a Hybrid Track Activity Better}\label{sec:similarity:results_similarity}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (chapter order Introduction, Methods, Results, Conclusions; reference theses put topics under one Results section, style guide 19)
\section{Results}

% TODO: manual review -- title was: Results: Does Distance in a Hybrid Track Activity Better -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question (with no question mark) made a topic, in the words of the matching Methods paragraph "Distance against activity."
\subsection{Distance against activity difference}\label{sec:similarity:results_similarity}
```

### S46
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Prediction from nearest neighbors}\label{sec:similarity:results_knn}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: Prediction from Nearest Neighbors}\label{sec:similarity:results_knn}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Prediction from Nearest Neighbors -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Prediction from nearest neighbors}\label{sec:similarity:results_knn}
```

### S47
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Activity cliffs}\label{sec:similarity:results_cliffs}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: Activity Cliffs}\label{sec:similarity:results_cliffs}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Activity Cliffs -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Activity cliffs}\label{sec:similarity:results_cliffs}
```

### S48
- file: thesis.tex
- chapter: 6
- kind: title
- result: `\subsection{Prediction error, retrieval and clustering}\label{sec:similarity:results_other}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; "other uses" was vague, and the section reports the three uses named in the Methods paragraphs "Distance against prediction error." and "Retrieval and clustering."

OLD:
```latex
\section{Results: Other Uses of a Distance}\label{sec:similarity:results_other}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Other Uses of a Distance -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; "other uses" was vague, and the section reports the three uses named in the Methods paragraphs "Distance against prediction error." and "Retrieval and clustering."
\subsection{Prediction error, retrieval and clustering}\label{sec:similarity:results_other}
```

### S49
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\chapter{Detecting and handling label noise}\label{chap:noise}`
- why: informal next to the other chapter titles (style guide 19); a gerund title saying what the chapter does, like "Building adaptive hybrid representations". "Handling" covers detection, noise mitigation and noise-robust training, all three of which the chapter tests

OLD:
```latex
\chapter{Label Noise: Detecting It and Doing Something About It}\label{chap:noise}
```
NEW:
```latex
% TODO: manual review -- title was: Label Noise: Detecting It and Doing Something About It -- edited your text: informal next to the other chapter titles (style guide 19); a gerund title saying what the chapter does, like "Building adaptive hybrid representations". "Handling" covers detection, noise mitigation and noise-robust training, all three of which the chapter tests
\chapter{Detecting and handling label noise}\label{chap:noise}
```

### S50
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Noise conditions}\label{sec:noise:conditions}`
- why: a clause where its siblings are noun phrases ("Noise detection", "Noise mitigation", "Noise-robust training"); "noise conditions" is the Chapter 3 term and the label already says conditions

OLD:
```latex
\subsection{What the noise is}\label{sec:noise:conditions}
```
NEW:
```latex
% TODO: manual review -- title was: What the noise is -- edited your text: a clause where its siblings are noun phrases ("Noise detection", "Noise mitigation", "Noise-robust training"); "noise conditions" is the Chapter 3 term and the label already says conditions
\subsection{Noise conditions}\label{sec:noise:conditions}
```

### S51
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Evaluation and controls}\label{sec:noise:tests}`
- why: a 10-word clause; the subsection says how each group of methods is scored and which controls apply, QM9 with no noise added among them

OLD:
```latex
\subsection{How each group is tested, and QM9 as the clean control}\label{sec:noise:tests}
```
NEW:
```latex
% TODO: manual review -- title was: How each group is tested, and QM9 as the clean control -- edited your text: a 10-word clause; the subsection says how each group of methods is scored and which controls apply, QM9 with no noise added among them
\subsection{Evaluation and controls}\label{sec:noise:tests}
```

### S52
- file: thesis.tex
- chapter: 7
- kind: new Results section + title
- result: `\subsection{Overall effect of noise handling}\label{sec:noise:res:overall}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic (style guide 19)

OLD:
```latex
\section{Results: Does Noise Handling Work at All?}\label{sec:noise:res:overall}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (chapter order Introduction, Methods, Results, Conclusions; reference theses put topics under one Results section, style guide 19)
\section{Results}

% TODO: manual review -- title was: Results: Does Noise Handling Work at All? -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic (style guide 19)
\subsection{Overall effect of noise handling}\label{sec:noise:res:overall}
```

### S53
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Changing a label against deleting a molecule}\label{sec:noise:res:action}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: Changing a Label Against Deleting a Molecule}\label{sec:noise:res:action}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Changing a Label Against Deleting a Molecule -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Changing a label against deleting a molecule}\label{sec:noise:res:action}
```

### S54
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{The labels as delivered}\label{sec:noise:res:delivered}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{The labels as delivered}\label{sec:noise:res:delivered}
```
NEW:
```latex
% TODO: manual review -- title was: The labels as delivered -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{The labels as delivered}\label{sec:noise:res:delivered}
```

### S55
- file: thesis.tex
- chapter: 7
- kind: move (paste)
- result: `For the labels as delivered there is no truth to grade against, so a g`
- why: third paragraph of the former Discussion section; it interprets the labels-as-delivered results, so it ends that subsubsection (interpretation inside Results, as in Dablander's "Results and Discussion")

OLD:
```latex
% Fold 2 agrees, 0.5370 against 0.5453, and fold 1 does not, 0.5684 against 0.5572.
```
NEW:
```latex
% Fold 2 agrees, 0.5370 against 0.5453, and fold 1 does not, 0.5684 against 0.5572.

% TODO: manual review -- moved here unchanged from the former Discussion section (no separate Discussion in Chapters 3-7)
% TODO: manual review
For the labels as delivered there is no truth to grade against, so a gain can come from removing hard molecules rather than wrong ones. The shuffled-score control separates the two. On small molecules it leaves one clear result, label smoothing on hERG $K_i$, and small gains on Caco-2 and LogD. On three antibody sets, SMAC colloidal stability, binding free energy and expression titer, cleaning passes the control, on SMAC by a wide margin. We read that as real, with the caution that each held-out fold has only about 50 antibodies. Cleaning an experimental dataset with nothing injected is a use case in its own right, and on those three datasets it is the one that paid off most.
```

### S56
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Noise detection}\label{sec:noise:res:detectors}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic, named like the Methods subsection it reports on (sec:noise:detection)

OLD:
```latex
\section{Results: Which Detectors Find the Suspect Labels}\label{sec:noise:res:detectors}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Which Detectors Find the Suspect Labels -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; question made a topic, named like the Methods subsection it reports on (sec:noise:detection)
\subsection{Noise detection}\label{sec:noise:res:detectors}
```

### S57
- file: thesis.tex
- chapter: 7
- kind: move (paste)
- result: `Chapter~\ref{chap:robustness} found that no model's own uncertainty pi`
- why: second paragraph of the former Discussion section; it explains why the residual detector falls to chance under censoring and links that to Chapter 3, so it ends the detector results

OLD:
```latex
% On hERG $K_i$ under the plain shift in 10\%, the top 10\% has a precision of 0.26 against a break-even of 0.68.
```
NEW:
```latex
% On hERG $K_i$ under the plain shift in 10\%, the top 10\% has a precision of 0.26 against a break-even of 0.68.

% TODO: manual review -- moved here unchanged from the former Discussion section (no separate Discussion in Chapters 3-7)
% TODO: manual review
Chapter~\ref{chap:robustness} found that no model's own uncertainty picks out the corrupted labels, and that models are most certain about the labels that were clipped. This chapter agrees from the other side. The residual detector fell to chance under censoring, because a model fitted to clipped labels predicts the clipped values. The three detectors that predict a spread from the features alone could not be graded at all, because they never read a label. And the damage that was most fixable is the damage that needs no detector.
```

### S58
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Effect by noise type}\label{sec:noise:res:kind}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; "By noise type" was a fragment

OLD:
```latex
\section{Results: By Noise Type}\label{sec:noise:res:kind}
```
NEW:
```latex
% TODO: manual review -- title was: Results: By Noise Type -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; "By noise type" was a fragment
\subsection{Effect by noise type}\label{sec:noise:res:kind}
```

### S59
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{Labels clipped at an assay limit}\label{sec:noise:res:censoring}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Labels clipped at an assay limit}\label{sec:noise:res:censoring}
```
NEW:
```latex
% TODO: manual review -- title was: Labels clipped at an assay limit -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Labels clipped at an assay limit}\label{sec:noise:res:censoring}
```

### S60
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}
```
NEW:
```latex
% TODO: manual review -- title was: A shared offset per scaffold group -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}
```

### S61
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{Several sources per molecule}\label{sec:noise:res:sources}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Several sources per molecule}\label{sec:noise:res:sources}
```
NEW:
```latex
% TODO: manual review -- title was: Several sources per molecule -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Several sources per molecule}\label{sec:noise:res:sources}
```

### S62
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{Gaussian noise, outliers and shifts in a few labels}\label{sec:noise:res:other_kinds}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Gaussian noise, outliers and shifts in a few labels}\label{sec:noise:res:other_kinds}
```
NEW:
```latex
% TODO: manual review -- title was: Gaussian noise, outliers and shifts in a few labels -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Gaussian noise, outliers and shifts in a few labels}\label{sec:noise:res:other_kinds}
```

### S63
- file: thesis.tex
- chapter: 7
- kind: move (paste)
- result: `Noise mitigation works in QSAR data, and it works best when something `
- why: first paragraph of the former Discussion section; it compares flag-driven and detector-driven gains across damage kinds, which is what the paragraph above it ("Set side by side, ...") does, so it goes here. The two overlap: a prose pass should merge them

OLD:
```latex
Those were helped too, as Section~\ref{sec:noise:res:action} shows, but the gains were smaller.
```
NEW:
```latex
Those were helped too, as Section~\ref{sec:noise:res:action} shows, but the gains were smaller.

% TODO: manual review -- moved here unchanged from the former Discussion section (no separate Discussion in Chapters 3-7); overlaps the paragraph above, merge in the prose pass
% TODO: manual review
Noise mitigation works in QSAR data, and it works best when something other than a model tells us which labels are wrong. A clipped label is flagged by the assay limit, and a shifted scaffold family by its scaffold. In both cases moving the flagged labels won back about half or more of the lost RMSE on every small-molecule dataset where the damage hurt. When a detector has to infer which labels are wrong, mitigation still helped. Under hERG's label flips it beat random deletion on three folds. But the gains were smaller, and they depended on whether the detector's precision cleared the break-even point for that damage. Under outliers the forest residual's precision was well above it. Under shifts in a few labels it was well below it, and acting on its flags did little.
```

### S64
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Effect by dataset}\label{sec:noise:res:dataset}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; parallel with "Effect by noise type"

OLD:
```latex
\section{Results: By Dataset}\label{sec:noise:res:dataset}
```
NEW:
```latex
% TODO: manual review -- title was: Results: By Dataset -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; parallel with "Effect by noise type"
\subsection{Effect by dataset}\label{sec:noise:res:dataset}
```

### S65
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{The antibody datasets}\label{sec:noise:res:antibodies}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{The antibody datasets}\label{sec:noise:res:antibodies}
```
NEW:
```latex
% TODO: manual review -- title was: The antibody datasets -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{The antibody datasets}\label{sec:noise:res:antibodies}
```

### S66
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{Small molecules and the reaction set}\label{sec:noise:res:smallmol}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Small molecules and the reaction set}\label{sec:noise:res:smallmol}
```
NEW:
```latex
% TODO: manual review -- title was: Small molecules and the reaction set -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Small molecules and the reaction set}\label{sec:noise:res:smallmol}
```

### S67
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}}\label{sec:noise:res:herg_ic50}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}}\label{sec:noise:res:herg_ic50}
```
NEW:
```latex
% TODO: manual review -- title was: Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}}\label{sec:noise:res:herg_ic50}
```

### S68
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Cleaning inside the feature search}\label{sec:noise:res:search}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: Cleaning Inside the Feature Search}\label{sec:noise:res:search}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Cleaning Inside the Feature Search -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Cleaning inside the feature search}\label{sec:noise:res:search}
```

### S69
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Changing the model instead of the labels}\label{sec:noise:res:model}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case

OLD:
```latex
\section{Results: Changing the Model Instead of the Labels}\label{sec:noise:res:model}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Changing the Model Instead of the Labels -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; sentence case
\subsection{Changing the model instead of the labels}\label{sec:noise:res:model}
```

### S70
- file: thesis.tex
- chapter: 7
- kind: level
- result: `\subsubsection{The expanded published sweep}\label{sec:noise:res:sweep}`
- why: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged

OLD:
```latex
\subsection{The expanded published sweep}\label{sec:noise:res:sweep}
```
NEW:
```latex
% TODO: manual review -- title was: The expanded published sweep -- edited your text: heading level only (subsection, now subsubsection), because its parent is now a subsection under the new Results section; wording unchanged
\subsubsection{The expanded published sweep}\label{sec:noise:res:sweep}
```

### S71
- file: thesis.tex
- chapter: 7
- kind: title
- result: `\subsection{Effect by model and representation}\label{sec:noise:res:repr}`
- why: section, now subsection under the new Results section; "Results:" prefix dropped; parallel with "Effect by noise type" and "Effect by dataset"

OLD:
```latex
\section{Results: Model and Representation}\label{sec:noise:res:repr}
```
NEW:
```latex
% TODO: manual review -- title was: Results: Model and Representation -- edited your text: section, now subsection under the new Results section; "Results:" prefix dropped; parallel with "Effect by noise type" and "Effect by dataset"
\subsection{Effect by model and representation}\label{sec:noise:res:repr}
```

### S72
- file: thesis.tex
- chapter: 7
- kind: heading removed + move (cut)
- result: `Was fixing the feature set the right place to start? On small molecule`
- why: no separate Discussion section in Chapters 3-7. Paragraphs 1-3 were pasted by the three edits above. Paragraph 4 (hybrid against single representation, detectors, ranking of methods by feature space) stays where it is, so it now ends the Results subsection "Effect by model and representation"; the label stays with it

OLD:
```latex
\section{Discussion}\label{sec:noise:discussion}

% TODO: manual review
Noise mitigation works in QSAR data, and it works best when something other than a model tells us which labels are wrong. A clipped label is flagged by the assay limit, and a shifted scaffold family by its scaffold. In both cases moving the flagged labels won back about half or more of the lost RMSE on every small-molecule dataset where the damage hurt. When a detector has to infer which labels are wrong, mitigation still helped. Under hERG's label flips it beat random deletion on three folds. But the gains were smaller, and they depended on whether the detector's precision cleared the break-even point for that damage. Under outliers the forest residual's precision was well above it. Under shifts in a few labels it was well below it, and acting on its flags did little.

% TODO: manual review
Chapter~\ref{chap:robustness} found that no model's own uncertainty picks out the corrupted labels, and that models are most certain about the labels that were clipped. This chapter agrees from the other side. The residual detector fell to chance under censoring, because a model fitted to clipped labels predicts the clipped values. The three detectors that predict a spread from the features alone could not be graded at all, because they never read a label. And the damage that was most fixable is the damage that needs no detector.

% TODO: manual review
For the labels as delivered there is no truth to grade against, so a gain can come from removing hard molecules rather than wrong ones. The shuffled-score control separates the two. On small molecules it leaves one clear result, label smoothing on hERG $K_i$, and small gains on Caco-2 and LogD. On three antibody sets, SMAC colloidal stability, binding free energy and expression titer, cleaning passes the control, on SMAC by a wide margin. We read that as real, with the caution that each held-out fold has only about 50 antibodies. Cleaning an experimental dataset with nothing injected is a use case in its own right, and on those three datasets it is the one that paid off most.

% TODO: manual review
Was fixing the feature set the right place to start? On small molecules the hybrid started ahead, and it stays ahead through every kind of cleaning, so a user who builds a hybrid loses nothing by cleaning afterwards. The detectors told a more mixed story. Neighbors found in the hybrid led to better cleaning slightly more often than neighbors in a single representation. For detection alone, though, the importance-weighted features of a single representation did better than the hybrid's. The hybrid search itself turned out to be robust to Gaussian noise: searching on damaged labels instead of clean ones changed Caco-2's score in the third decimal. And the ranking of noise-handling methods changed between the hybrid and the single representations in almost every condition. That is the result that ties this chapter to Chapter~\ref{chap:flaws}. A noise-handling method is not good or bad on its own. It is good or bad in a feature space.
```
NEW:
```latex
% TODO: manual review -- the Discussion section heading was removed (no separate Discussion in Chapters 3-7). Its first three paragraphs moved unchanged into Results: the labels as delivered, noise detection, and effect by noise type. This last one stays here, inside the Results subsection on model and representation, with the old label.
\label{sec:noise:discussion}
% TODO: manual review
Was fixing the feature set the right place to start? On small molecules the hybrid started ahead, and it stays ahead through every kind of cleaning, so a user who builds a hybrid loses nothing by cleaning afterwards. The detectors told a more mixed story. Neighbors found in the hybrid led to better cleaning slightly more often than neighbors in a single representation. For detection alone, though, the importance-weighted features of a single representation did better than the hybrid's. The hybrid search itself turned out to be robust to Gaussian noise: searching on damaged labels instead of clean ones changed Caco-2's score in the third decimal. And the ranking of noise-handling methods changed between the hybrid and the single representations in almost every condition. That is the result that ties this chapter to Chapter~\ref{chap:flaws}. A noise-handling method is not good or bad on its own. It is good or bad in a feature space.
```

### S73
- file: thesis.tex
- chapter: 7
- kind: title + move
- result: `\section{Conclusions}\label{sec:noise:conclusion}`
- why: Conclusions now comes before the limitation paragraphs. The old \section{Limitations} heading is replaced by the Conclusions heading and the conclusion paragraph, without its last sentence; the label sec:noise:limitations marks where the limitation paragraphs start. "Conclusion" becomes "Conclusions", as in Chapters 2-6

OLD:
```latex
\section{Limitations}\label{sec:noise:limitations}
```
NEW:
```latex
% TODO: manual review -- title was: Conclusion -- edited your text: "Conclusions", as in Chapters 2-6; this heading and the paragraph below moved above the limitation paragraphs (limitations are paragraphs inside Conclusions, as in Chapters 3, 4 and 6); the last sentence of the paragraph moved to the end of the section
\section{Conclusions}\label{sec:noise:conclusion}

% TODO: manual review
Noisy labels in QSAR data can be found and, under the right conditions, fixed. Fixing works best with a flag: raising labels clipped at an assay limit won back half to three quarters of the lost RMSE on the small-molecule datasets, and needed no detector. With a detector, deleting or relabeling flagged molecules helped under label flips and outliers, and passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody sets. On clean QM9 labels deletion never helped, so those gains are noise removed. The hybrid stays ahead of the single representation on small molecules through every treatment, and the ranking of noise-handling methods depends on the feature space they run in.

% TODO: manual review -- the Limitations section heading was removed; the three paragraphs below are its text, unchanged, now part of Conclusions
\label{sec:noise:limitations}
```

### S74
- file: thesis.tex
- chapter: 7
- kind: move (cut)
- result: `Chapter~\ref{chap:discussion} brings these results together with the r`
- why: the old Conclusion heading and paragraph were pasted by the edit above; only the last sentence, which points to Chapter 8, stays here, so it ends the chapter after the limitation paragraphs

OLD:
```latex
\section{Conclusion}\label{sec:noise:conclusion}

% TODO: manual review
Noisy labels in QSAR data can be found and, under the right conditions, fixed. Fixing works best with a flag: raising labels clipped at an assay limit won back half to three quarters of the lost RMSE on the small-molecule datasets, and needed no detector. With a detector, deleting or relabeling flagged molecules helped under label flips and outliers, and passed random and shuffled controls on hERG, hERG $K_i$, Buchwald--Hartwig and three antibody sets. On clean QM9 labels deletion never helped, so those gains are noise removed. The hybrid stays ahead of the single representation on small molecules through every treatment, and the ranking of noise-handling methods depends on the feature space they run in. Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis.
```
NEW:
```latex
% TODO: manual review -- last sentence of the conclusion paragraph, moved after the limitation paragraphs
Chapter~\ref{chap:discussion} brings these results together with the rest of the thesis.
```
