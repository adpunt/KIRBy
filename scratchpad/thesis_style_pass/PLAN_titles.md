# Plan: section titles, Chapters 1 to 8 (thesis.tex)

Every OLD block below was checked to occur exactly once in thesis.tex (python3 -I, str.count) on 2026-10-10, and no two OLD blocks overlap. Where a heading line alone is not unique (e.g. `\section{Conclusions}`), the OLD block continues to the next line or to the first sentence of the next paragraph. Apply the edits in any order. No `\label` is changed. No heading has a short title in `[ ]`. The one `\texorpdfstring` (hERG IC$_{50}$) is kept as it is.

## Rationale

The three reference theses use short noun or gerund phrases, and B and V use sentence case at every level. None uses a question, a claim or a "Results:" prefix. All three put "Results" as a parent section with the topics as subsections, and keep sibling titles parallel (style guide 19). So every title goes to sentence case, and question, claim and informal titles become topics. (Checker: wording changes with no style-guide item behind them were reverted; see Checker notes.) Chapters 3, 5, 6 and 7 each get one `\section{Results}`, with the old results sections as subsections and their subsections demoted one level; none of those sections had a subsubsection, so no fourth level is created. Chapter 3 had no "Results:" prefix, but it gets the same parent so that Chapters 3, 5, 6 and 7 all run Introduction, Methods, Results, then Discussion or Conclusions. Chapter 5 has two "Methods:" sections, and both already hold subsubsections, so they keep their level and only lose the prefix. Titles the guide lists as good, and plain ones such as "Introduction", "Models" and "Tree ensembles", are kept. Each changed heading gets `% TODO: manual review -- title was: <old title>` above it, with "(section, now subsection)" where only the level changed.

## Old title -> new title

One row per heading in thesis.tex, in file order. "Level" gives old -> new heading level.

| Ch | Level | Old title | New title |
|---|---|---|---|
| 1 | ch | Introduction | Introduction  (kept) |
| 1 | sec | Outline of the thesis and contributions | Outline of the thesis and contributions  (kept) |
| 1 | sec | Collaborations | Collaborations  (kept) |
| 2 | ch | Molecular Representations, Models and Label Noise | Molecular representations, models and label noise |
| 2 | sec | Overview | Overview  (kept) |
| 2 | sec | Molecular representations in QSAR | Molecular representations in QSAR  (kept) |
| 2 | sec | Models for molecular property prediction | Models for molecular property prediction  (kept) |
| 2 | subsec | Tree ensembles | Tree ensembles  (kept) |
| 2 | subsec | Support vector machines | Support vector machines  (kept) |
| 2 | subsec | Gaussian processes | Gaussian processes  (kept) |
| 2 | subsec | Bayesian neural networks | Bayesian neural networks  (kept) |
| 2 | sec | Molecular similarity and activity cliffs | Molecular similarity and activity cliffs  (kept) |
| 2 | sec | Noise in molecular data | Noise in molecular data  (kept) |
| 2 | subsec | Artificial noise injection | Artificial noise injection  (kept) |
| 2 | subsec | Methods for noisy labels | Learning with noisy labels |
| 2 | sec | NoiseInject: a benchmarking framework | NoiseInject: a benchmarking framework  (kept) |
| 2 | subsec | Package description | Package description  (kept) |
| 2 | sec | Evaluation methodology | Evaluation methodology  (kept) |
| 2 | sec | Conclusions | Conclusions  (kept) |
| 3 | ch | Noise Robustness Across Molecular Representations and Model Architectures | Noise robustness across molecular representations and model architectures |
| 3 | sec | Introduction | Introduction  (kept) |
| 3 | sec | Methods | Methods  (kept) |
| 3 | subsec | Datasets | Datasets  (kept) |
| 3 | subsec | Representations | Representations  (kept) |
| 3 | subsec | Models | Models  (kept) |
| 3 | subsec | Noise conditions and levels | Noise conditions and levels  (kept) |
| 3 | subsec | QM9 data pipeline | QM9 data pipeline  (kept) |
| 3 | subsec | Uncertainty | Uncertainty  (kept) |
| 3 | subsec | Evaluation | Evaluation  (kept) |
| 3 | sec -> subsec | Variance decomposition | Variance decomposition |
| 3 | sec -> subsec | Predictive performance and noise robustness | Predictive performance and noise robustness |
| 3 | sec -> subsec | Artificial noise conditions | Artificial noise conditions |
| 3 | sec -> subsec | Robustness on the three assay datasets | Robustness on the three assay datasets |
| 3 | sec -> subsec | Probabilistic and deterministic counterparts | Probabilistic and deterministic counterparts |
| 3 | sec -> subsec | Uncertainty under increasing label noise | Uncertainty under increasing label noise |
| 3 | subsec -> subsubsec | Aleatoric and epistemic attribution | Aleatoric and epistemic attribution |
| 3 | subsec -> subsubsec | Per-sample uncertainty and label noise | Per-sample uncertainty and label noise |
| 3 | sec | Conclusions | Conclusions  (kept) |
| 4 | ch | Limitations of the Similarity Assumption | Limitations of the similarity assumption |
| 4 | sec | Introduction | Introduction  (kept) |
| 4 | sec | Flaws in standard representations | Flaws in standard representations  (kept) |
| 4 | subsec | Constant and near-constant features | Constant and near-constant features  (kept) |
| 4 | subsec | Molecules with identical feature vectors | Molecules with identical feature vectors  (kept) |
| 4 | sec | Flaws in similarity and other pairwise computations | Flaws in similarity and other pairwise computations  (kept) |
| 4 | subsec | Distance metrics for continuous molecular descriptors | Distance metrics for continuous molecular descriptors  (kept) |
| 4 | sec | Consequences for similarity-based methods | Consequences for similarity-based methods  (kept) |
| 4 | subsec | A small sweep of published noise-handling methods | A small sweep of published noise-handling methods  (kept) |
| 4 | subsec | Noise detection by neighbors | Noise detection by neighbors  (kept) |
| 4 | sec | Conclusions | Conclusions  (kept) |
| 5 | ch | Building Adaptive Hybrid Representations | Building adaptive hybrid representations |
| 5 | sec | Introduction | Introduction  (kept) |
| 5 | subsec | Combining representations | Combining representations  (kept) |
| 5 | subsec | The adaptive hybrid representation | The adaptive hybrid representation  (kept) |
| 5 | subsec | Blueprints and their chemical interpretation | Blueprints and their chemical interpretation  (kept) |
| 5 | subsec | Antibodies and beyond | Antibodies and beyond  (kept) |
| 5 | sec | Methods: the hybrid search | The hybrid search |
| 5 | subsec | From base representations to a hybrid | From base representations to a hybrid  (kept) |
| 5 | subsec | The search space | The search space  (kept) |
| 5 | subsubsec | Feature importance scoring | Feature importance scoring  (kept) |
| 5 | subsubsec | Feature allocation methods | Feature allocation methods  (kept) |
| 5 | subsubsec | Search configuration and internal evaluation | Search configuration and internal evaluation  (kept) |
| 5 | subsec | Augmentation | Augmentation  (kept) |
| 5 | subsec | Search grid per dataset | Search grid per dataset  (kept) |
| 5 | subsubsec | Adaptive pruning | Adaptive pruning  (kept) |
| 5 | sec | Methods: experimental setup | Experimental setup |
| 5 | subsec | Datasets and splitting | Datasets and splitting  (kept) |
| 5 | subsec | Representations by domain | Representations by domain  (kept) |
| 5 | subsec | Models and metrics | Models and metrics  (kept) |
| 5 | subsec | Comparators, controls and ablations | Comparators, controls and ablations  (kept) |
| 5 | subsec | Transfer, collaboration studies and separate sites | Transfer, collaboration studies and separate sites  (kept) |
| 5 | sec -> subsec | Results: when the hybrid helps | Performance of the hybrid |
| 5 | subsec -> subsubsec | Against the best single representation | Against the best single representation |
| 5 | subsec -> subsubsec | Against ensembles, and many representations against few | Against ensembles, and many representations against few |
| 5 | subsec -> subsubsec | Does the search overfit? | Overfitting of the search |
| 5 | sec -> subsec | Results: why the hybrid helps | Sources of the gain |
| 5 | subsec -> subsubsec | Selection, not only fewer features | Selection at a fixed number of features |
| 5 | subsec -> subsubsec | Different models want different hybrids | Dependence on the downstream model |
| 5 | subsec -> subsubsec | Using importance during training | Using importance during training |
| 5 | sec -> subsec | Results: blueprints on specific datasets | Blueprints on specific datasets |
| 5 | sec -> subsec | Results: collaboration studies | Collaboration studies |
| 5 | sec -> subsec | Results: features chosen across separate sites | Features chosen across separate sites |
| 5 | sec | Limitations | Limitations  (kept) |
| 5 | sec | Conclusions | Conclusions  (kept) |
| 6 | ch | Similarity in an Adaptive Hybrid Representation | Similarity in an adaptive hybrid representation |
| 6 | sec | Introduction | Introduction  (kept) |
| 6 | sec | Methods | Methods  (kept) |
| 6 | subsec | Feature Spaces Compared | Feature spaces compared |
| 6 | subsec | Similarity in a Hybrid | Similarity in a hybrid |
| 6 | subsec | Nearest-Neighbor Settings | Nearest-neighbor settings |
| 6 | subsec | Experiments | Experiments  (kept) |
| 6 | sec -> subsec | Results: Does Distance in a Hybrid Track Activity Better | Distance against activity difference |
| 6 | sec -> subsec | Results: Prediction from Nearest Neighbors | Prediction from nearest neighbors |
| 6 | sec -> subsec | Results: Activity Cliffs | Activity cliffs |
| 6 | sec -> subsec | Results: Other Uses of a Distance | Other uses of a distance |
| 6 | sec | Conclusions | Conclusions  (kept) |
| 7 | ch | Label Noise: Detecting It and Doing Something About It | Detecting and handling label noise |
| 7 | sec | Introduction | Introduction  (kept) |
| 7 | sec | Methods | Methods  (kept) |
| 7 | subsec | What the noise is | What the noise is  (kept) |
| 7 | subsec | Noise detection | Noise detection  (kept) |
| 7 | subsec | Noise mitigation | Noise mitigation  (kept) |
| 7 | subsec | Noise-robust training | Noise-robust training  (kept) |
| 7 | subsec | How each group is tested, and QM9 as the clean control | How each group is tested, and QM9 as the clean control  (kept) |
| 7 | sec -> subsec | Results: Does Noise Handling Work at All? | Overall effect of noise handling |
| 7 | sec -> subsec | Results: Changing a Label Against Deleting a Molecule | Changing a label against deleting a molecule |
| 7 | subsec -> subsubsec | The labels as delivered | The labels as delivered |
| 7 | sec -> subsec | Results: Which Detectors Find the Suspect Labels | Noise detection |
| 7 | sec -> subsec | Results: By Noise Type | Effect by noise type |
| 7 | subsec -> subsubsec | Labels clipped at an assay limit | Labels clipped at an assay limit |
| 7 | subsec -> subsubsec | A shared offset per scaffold group | A shared offset per scaffold group |
| 7 | subsec -> subsubsec | Several sources per molecule | Several sources per molecule |
| 7 | subsec -> subsubsec | Gaussian noise, outliers and shifts in a few labels | Gaussian noise, outliers and shifts in a few labels |
| 7 | sec -> subsec | Results: By Dataset | Effect by dataset |
| 7 | subsec -> subsubsec | The antibody datasets | The antibody datasets |
| 7 | subsec -> subsubsec | Small molecules and the reaction set | Small molecules and the reaction set |
| 7 | subsec -> subsubsec | Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} | Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} |
| 7 | sec -> subsec | Results: Cleaning Inside the Feature Search | Cleaning inside the feature search |
| 7 | sec -> subsec | Results: Changing the Model Instead of the Labels | Changing the model instead of the labels |
| 7 | subsec -> subsubsec | The expanded published sweep | The expanded published sweep |
| 7 | sec -> subsec | Results: Model and Representation | Model and representation |
| 7 | sec | Discussion | Discussion  (kept) |
| 7 | sec | Limitations | Limitations  (kept) |
| 7 | sec | Conclusion | Conclusions |
| 8 | ch | Discussion and Conclusions | Discussion and conclusions |
| 8 | sec | Summary | Summary  (kept) |
| 8 | sec | The Shape of the Argument | The shape of the argument |
| 8 | sec | The Central Claim | The central claim |
| 8 | sec | What Is New in This Thesis | What is new in this thesis |
| 8 | sec | Handling Noise Realistically | Handling noise realistically |
| 8 | sec | The Antibodies | The antibodies |
| 8 | sec | No Single Method Fits Every Dataset | Dependence on the dataset |
| 8 | sec | Simple Is Often Better | Simple methods |
| 8 | sec | Limitations | Limitations  (kept) |
| 8 | sec | Future Directions | Future directions |
| 8 | sec | Conclusions | Conclusions  (kept) |
| end | ch* | Summary of Research Contributions | Summary of research contributions |
| end | sec* | Research Papers | Research papers |
| end | sec* | Conference Presentations | Conference presentations |

New headings with no old title: `\section{Results}` in Chapters 3, 5, 6 and 7 (edits marked "new Results section").

## Edits

### E1 — Molecular representations, models and label noise
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Molecular Representations, Models and Label Noise; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Molecular Representations, Models and Label Noise}\label{chap:background}\label{sec2}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Molecular Representations, Models and Label Noise; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Molecular representations, models and label noise}\label{chap:background}\label{sec2}
```

### E2 — Learning with noisy labels
- file: thesis.tex
- kind: title
- why: the CODE INPUT comment above asks for the title to be reviewed; LNL names the problem, and this section reviews it.
- todo_line: % TODO: manual review -- title was: Methods for noisy labels
OLD:
```latex
\subsection{Methods for noisy labels}\label{sec:lnl_background}
```
NEW:
```latex
% TODO: manual review -- title was: Methods for noisy labels
\subsection{Learning with noisy labels}\label{sec:lnl_background}
```

### E4 — Noise robustness across molecular representations and model architectures
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Noise Robustness Across Molecular Representations and Model Architectures; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Noise Robustness Across Molecular Representations and Model Architectures}\label{chap:robustness}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Noise Robustness Across Molecular Representations and Model Architectures; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Noise robustness across molecular representations and model architectures}\label{chap:robustness}
```

### E5 — new Results section + Variance decomposition
- file: thesis.tex
- kind: move
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Variance decomposition (section, now subsection); heading level only, now under the new Results section; wording unchanged (plus the new-section TODO line above \section{Results})
OLD:
```latex
\section{Variance decomposition}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (style guide 19)
\section{Results}

% TODO: manual review -- edited your text: title was: Variance decomposition (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Variance decomposition}
```

### E6 — Predictive performance and noise robustness
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Predictive performance and noise robustness (section, now subsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\section{Predictive performance and noise robustness}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Predictive performance and noise robustness (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Predictive performance and noise robustness}
```

### E7 — Artificial noise conditions
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Artificial noise conditions (section, now subsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\section{Artificial noise conditions}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Artificial noise conditions (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Artificial noise conditions}
```

### E8 — Robustness on the three assay datasets
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Robustness on the three assay datasets (section, now subsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\section{Robustness on the three assay datasets}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Robustness on the three assay datasets (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Robustness on the three assay datasets}
```

### E9 — Probabilistic and deterministic counterparts
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Probabilistic and deterministic counterparts (section, now subsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\section{Probabilistic and deterministic counterparts}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Probabilistic and deterministic counterparts (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Probabilistic and deterministic counterparts}
```

### E10 — Uncertainty under increasing label noise
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Uncertainty under increasing label noise (section, now subsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\section{Uncertainty under increasing label noise}\label{sec:uq}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Uncertainty under increasing label noise (section, now subsection); heading level only, now under the new Results section; wording unchanged
\subsection{Uncertainty under increasing label noise}\label{sec:uq}
```

### E11 — Aleatoric and epistemic attribution
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Aleatoric and epistemic attribution (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Aleatoric and epistemic attribution}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Aleatoric and epistemic attribution (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Aleatoric and epistemic attribution}
```

### E12 — Per-sample uncertainty and label noise
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Per-sample uncertainty and label noise (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Per-sample uncertainty and label noise}\label{sec:uq_per_sample}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Per-sample uncertainty and label noise (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Per-sample uncertainty and label noise}\label{sec:uq_per_sample}
```

### E13 — Limitations of the similarity assumption
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Limitations of the Similarity Assumption; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Limitations of the Similarity Assumption}\label{chap:flaws}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Limitations of the Similarity Assumption; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Limitations of the similarity assumption}\label{chap:flaws}
```

### E16 — Building adaptive hybrid representations
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Building Adaptive Hybrid Representations; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Building Adaptive Hybrid Representations}\label{chap:method}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Building Adaptive Hybrid Representations; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Building adaptive hybrid representations}\label{chap:method}
```

### E19 — The hybrid search
- file: thesis.tex
- kind: title
- why: dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23).
- todo_line: % TODO: manual review -- edited your text: title was: Methods: the hybrid search; dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23)
OLD:
```latex
\section{Methods: the hybrid search}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Methods: the hybrid search; dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23)
\section{The hybrid search}
```

### E20 — Experimental setup
- file: thesis.tex
- kind: title
- why: dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23).
- todo_line: % TODO: manual review -- edited your text: title was: Methods: experimental setup; dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23)
OLD:
```latex
\section{Methods: experimental setup}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Methods: experimental setup; dropped the "Methods:" prefix, to match the "Results:" prefixes being dropped (style guide item 23)
\section{Experimental setup}
```

### E21 — new Results section + Performance of the hybrid
- file: thesis.tex
- kind: move
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic.
- todo_line: % TODO: manual review -- edited your text: title was: Results: when the hybrid helps (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic (plus the new-section TODO line above \section{Results})
OLD:
```latex
\section{Results: when the hybrid helps}\label{sec:method:results_single}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (style guide 19)
\section{Results}

% TODO: manual review -- edited your text: title was: Results: when the hybrid helps (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic
\subsection{Performance of the hybrid}\label{sec:method:results_single}
```

### E22 — Against the best single representation
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Against the best single representation (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Against the best single representation}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Against the best single representation (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Against the best single representation}
```

### E23 — Against ensembles, and many representations against few
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Against ensembles, and many representations against few (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Against ensembles, and many representations against few}\label{sec:method:results_ensembles}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Against ensembles, and many representations against few (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Against ensembles, and many representations against few}\label{sec:method:results_ensembles}
```

### E24 — Overfitting of the search
- file: thesis.tex
- kind: title
- why: now a subsubsection under the new Results section; the question became a topic (style guide gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Does the search overfit? (subsection, now subsubsection); now a subsubsection under the new Results section; the question became a topic (style guide gap item 24)
OLD:
```latex
\subsection{Does the search overfit?}\label{sec:method:results_overfit}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Does the search overfit? (subsection, now subsubsection); now a subsubsection under the new Results section; the question became a topic (style guide gap item 24)
\subsubsection{Overfitting of the search}\label{sec:method:results_overfit}
```

### E25 — Sources of the gain
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic.
- todo_line: % TODO: manual review -- edited your text: title was: Results: why the hybrid helps (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic
OLD:
```latex
\section{Results: why the hybrid helps}\label{sec:method:results_why}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: why the hybrid helps (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the old title stated a finding, the new one names the topic
\subsection{Sources of the gain}\label{sec:method:results_why}
```

### E26 — Selection at a fixed number of features
- file: thesis.tex
- kind: title
- why: now a subsubsection under the new Results section; the old title stated a claim, the new one names the comparison (style guide 19).
- todo_line: % TODO: manual review -- edited your text: title was: Selection, not only fewer features (subsection, now subsubsection); now a subsubsection under the new Results section; the old title stated a claim, the new one names the comparison (style guide 19)
OLD:
```latex
\subsection{Selection, not only fewer features}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Selection, not only fewer features (subsection, now subsubsection); now a subsubsection under the new Results section; the old title stated a claim, the new one names the comparison (style guide 19)
\subsubsection{Selection at a fixed number of features}
```

### E27 — Dependence on the downstream model
- file: thesis.tex
- kind: title
- why: now a subsubsection under the new Results section; the old title stated a claim, the new one names the topic (style guide 19).
- todo_line: % TODO: manual review -- edited your text: title was: Different models want different hybrids (subsection, now subsubsection); now a subsubsection under the new Results section; the old title stated a claim, the new one names the topic (style guide 19)
OLD:
```latex
\subsection{Different models want different hybrids}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Different models want different hybrids (subsection, now subsubsection); now a subsubsection under the new Results section; the old title stated a claim, the new one names the topic (style guide 19)
\subsubsection{Dependence on the downstream model}
```

### E28 — Using importance during training
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Using importance during training (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Using importance during training}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Using importance during training (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Using importance during training}
```

### E29 — Blueprints on specific datasets
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23).
- todo_line: % TODO: manual review -- edited your text: title was: Results: blueprints on specific datasets (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
OLD:
```latex
\section{Results: blueprints on specific datasets}\label{sec:method:results_blueprints}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: blueprints on specific datasets (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
\subsection{Blueprints on specific datasets}\label{sec:method:results_blueprints}
```

### E30 — Collaboration studies
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23).
- todo_line: % TODO: manual review -- edited your text: title was: Results: collaboration studies (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
OLD:
```latex
\section{Results: collaboration studies}\label{sec:method:results_opig}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: collaboration studies (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
\subsection{Collaboration studies}\label{sec:method:results_opig}
```

### E31 — Features chosen across separate sites
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23).
- todo_line: % TODO: manual review -- edited your text: title was: Results: features chosen across separate sites (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
OLD:
```latex
\section{Results: features chosen across separate sites}\label{sec:method:results_sites}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: features chosen across separate sites (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23)
\subsection{Features chosen across separate sites}\label{sec:method:results_sites}
```

### E32 — Similarity in an adaptive hybrid representation
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Similarity in an Adaptive Hybrid Representation; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Similarity in an Adaptive Hybrid Representation}\label{chap:similarity}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Similarity in an Adaptive Hybrid Representation; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Similarity in an adaptive hybrid representation}\label{chap:similarity}
```

### E33 — Feature spaces compared
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Feature Spaces Compared; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\subsection{Feature Spaces Compared}\label{sec:similarity:spaces}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Feature Spaces Compared; sentence case, as in the other chapters (style guide gap item 22)
\subsection{Feature spaces compared}\label{sec:similarity:spaces}
```

### E34 — Similarity in a hybrid
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Similarity in a Hybrid; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\subsection{Similarity in a Hybrid}\label{sec:similarity:hybrid_similarity}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Similarity in a Hybrid; sentence case, as in the other chapters (style guide gap item 22)
\subsection{Similarity in a hybrid}\label{sec:similarity:hybrid_similarity}
```

### E35 — Nearest-neighbor settings
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Nearest-Neighbor Settings; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\subsection{Nearest-Neighbor Settings}\label{sec:similarity:knn_settings}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Nearest-Neighbor Settings; sentence case, as in the other chapters (style guide gap item 22)
\subsection{Nearest-neighbor settings}\label{sec:similarity:knn_settings}
```

### E36 — new Results section + Distance against activity difference
- file: thesis.tex
- kind: move
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question (missing its "?") became a topic (gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Results: Does Distance in a Hybrid Track Activity Better (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question (missing its "?") became a topic (gap item 24) (plus the new-section TODO line above \section{Results})
OLD:
```latex
\section{Results: Does Distance in a Hybrid Track Activity Better}\label{sec:similarity:results_similarity}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (style guide 19)
\section{Results}

% TODO: manual review -- edited your text: title was: Results: Does Distance in a Hybrid Track Activity Better (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question (missing its "?") became a topic (gap item 24)
\subsection{Distance against activity difference}\label{sec:similarity:results_similarity}
```

### E37 — Prediction from nearest neighbors
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Prediction from Nearest Neighbors (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Prediction from Nearest Neighbors}\label{sec:similarity:results_knn}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Prediction from Nearest Neighbors (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Prediction from nearest neighbors}\label{sec:similarity:results_knn}
```

### E38 — Activity cliffs
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Activity Cliffs (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Activity Cliffs}\label{sec:similarity:results_cliffs}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Activity Cliffs (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Activity cliffs}\label{sec:similarity:results_cliffs}
```

### E39 — Other uses of a distance
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Other Uses of a Distance (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Other Uses of a Distance}\label{sec:similarity:results_other}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Other Uses of a Distance (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Other uses of a distance}\label{sec:similarity:results_other}
```

### E40 — Detecting and handling label noise
- file: thesis.tex
- kind: title
- why: informal chapter title made a topic (style guide gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Label Noise: Detecting It and Doing Something About It; informal chapter title made a topic (style guide gap item 24)
OLD:
```latex
\chapter{Label Noise: Detecting It and Doing Something About It}\label{chap:noise}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Label Noise: Detecting It and Doing Something About It; informal chapter title made a topic (style guide gap item 24)
\chapter{Detecting and handling label noise}\label{chap:noise}
```

### E42 — new Results section + Overall effect of noise handling
- file: thesis.tex
- kind: move
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question became a topic (gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Results: Does Noise Handling Work at All? (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question became a topic (gap item 24) (plus the new-section TODO line above \section{Results})
OLD:
```latex
\section{Results: Does Noise Handling Work at All?}\label{sec:noise:res:overall}
```
NEW:
```latex
% TODO: manual review -- new parent section: the results sections that follow are now its subsections (style guide 19)
\section{Results}

% TODO: manual review -- edited your text: title was: Results: Does Noise Handling Work at All? (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question became a topic (gap item 24)
\subsection{Overall effect of noise handling}\label{sec:noise:res:overall}
```

### E43 — Changing a label against deleting a molecule
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Changing a Label Against Deleting a Molecule (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Changing a Label Against Deleting a Molecule}\label{sec:noise:res:action}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Changing a Label Against Deleting a Molecule (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Changing a label against deleting a molecule}\label{sec:noise:res:action}
```

### E44 — The labels as delivered
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: The labels as delivered (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{The labels as delivered}\label{sec:noise:res:delivered}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The labels as delivered (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{The labels as delivered}\label{sec:noise:res:delivered}
```

### E45 — Noise detection
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question form became a topic (gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Results: Which Detectors Find the Suspect Labels (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question form became a topic (gap item 24)
OLD:
```latex
\section{Results: Which Detectors Find the Suspect Labels}\label{sec:noise:res:detectors}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Which Detectors Find the Suspect Labels (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); the question form became a topic (gap item 24)
\subsection{Noise detection}\label{sec:noise:res:detectors}
```

### E46 — Effect by noise type
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Noise Type" had no head noun (style guide 19, parallel form).
- todo_line: % TODO: manual review -- edited your text: title was: Results: By Noise Type (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Noise Type" had no head noun (style guide 19, parallel form)
OLD:
```latex
\section{Results: By Noise Type}\label{sec:noise:res:kind}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: By Noise Type (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Noise Type" had no head noun (style guide 19, parallel form)
\subsection{Effect by noise type}\label{sec:noise:res:kind}
```

### E47 — Labels clipped at an assay limit
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Labels clipped at an assay limit (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Labels clipped at an assay limit}\label{sec:noise:res:censoring}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Labels clipped at an assay limit (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Labels clipped at an assay limit}\label{sec:noise:res:censoring}
```

### E48 — A shared offset per scaffold group
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: A shared offset per scaffold group (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: A shared offset per scaffold group (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{A shared offset per scaffold group}\label{sec:noise:res:offsets}
```

### E49 — Several sources per molecule
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Several sources per molecule (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Several sources per molecule}\label{sec:noise:res:sources}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Several sources per molecule (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Several sources per molecule}\label{sec:noise:res:sources}
```

### E50 — Gaussian noise, outliers and shifts in a few labels
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Gaussian noise, outliers and shifts in a few labels (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Gaussian noise, outliers and shifts in a few labels}\label{sec:noise:res:other_kinds}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Gaussian noise, outliers and shifts in a few labels (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Gaussian noise, outliers and shifts in a few labels}\label{sec:noise:res:other_kinds}
```

### E51 — Effect by dataset
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Dataset" had no head noun (style guide 19, parallel form).
- todo_line: % TODO: manual review -- edited your text: title was: Results: By Dataset (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Dataset" had no head noun (style guide 19, parallel form)
OLD:
```latex
\section{Results: By Dataset}\label{sec:noise:res:dataset}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: By Dataset (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); "By Dataset" had no head noun (style guide 19, parallel form)
\subsection{Effect by dataset}\label{sec:noise:res:dataset}
```

### E52 — The antibody datasets
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: The antibody datasets (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{The antibody datasets}\label{sec:noise:res:antibodies}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The antibody datasets (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{The antibody datasets}\label{sec:noise:res:antibodies}
```

### E53 — Small molecules and the reaction set
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Small molecules and the reaction set (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Small molecules and the reaction set}\label{sec:noise:res:smallmol}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Small molecules and the reaction set (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Small molecules and the reaction set}\label{sec:noise:res:smallmol}
```

### E54 — Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}}\label{sec:noise:res:herg_ic50}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50} (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{Repeat measurements on hERG IC\texorpdfstring{$_{50}$}{50}}\label{sec:noise:res:herg_ic50}
```

### E55 — Cleaning inside the feature search
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Cleaning Inside the Feature Search (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Cleaning Inside the Feature Search}\label{sec:noise:res:search}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Cleaning Inside the Feature Search (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Cleaning inside the feature search}\label{sec:noise:res:search}
```

### E56 — Changing the model instead of the labels
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Changing the Model Instead of the Labels (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Changing the Model Instead of the Labels}\label{sec:noise:res:model}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Changing the Model Instead of the Labels (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Changing the model instead of the labels}\label{sec:noise:res:model}
```

### E57 — The expanded published sweep
- file: thesis.tex
- kind: title
- why: heading level only, now under the new Results section; wording unchanged.
- todo_line: % TODO: manual review -- edited your text: title was: The expanded published sweep (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
OLD:
```latex
\subsection{The expanded published sweep}\label{sec:noise:res:sweep}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The expanded published sweep (subsection, now subsubsection); heading level only, now under the new Results section; wording unchanged
\subsubsection{The expanded published sweep}\label{sec:noise:res:sweep}
```

### E58 — Model and representation
- file: thesis.tex
- kind: title
- why: dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case.
- todo_line: % TODO: manual review -- edited your text: title was: Results: Model and Representation (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
OLD:
```latex
\section{Results: Model and Representation}\label{sec:noise:res:repr}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Results: Model and Representation (section, now subsection); dropped the "Results:" prefix; now a subsection of the new Results section (style guide gap item 23); sentence case
\subsection{Model and representation}\label{sec:noise:res:repr}
```

### E59 — Conclusions
- file: thesis.tex
- kind: title
- why: "Conclusion" made "Conclusions", the name used in every other chapter.
- todo_line: % TODO: manual review -- edited your text: title was: Conclusion; "Conclusion" made "Conclusions", the name used in every other chapter
OLD:
```latex
\section{Conclusion}\label{sec:noise:conclusion}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Conclusion; "Conclusion" made "Conclusions", the name used in every other chapter
\section{Conclusions}\label{sec:noise:conclusion}
```

### E60 — Discussion and conclusions
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Discussion and Conclusions; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter{Discussion and Conclusions}\label{chap:discussion}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Discussion and Conclusions; sentence case, as in the other chapters (style guide gap item 22)
\chapter{Discussion and conclusions}\label{chap:discussion}
```

### E61 — The shape of the argument
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: The Shape of the Argument; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{The Shape of the Argument}\label{sec:discussion:chain}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The Shape of the Argument; sentence case, as in the other chapters (style guide gap item 22)
\section{The shape of the argument}\label{sec:discussion:chain}
```

### E62 — The central claim
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: The Central Claim; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{The Central Claim}\label{sec:discussion:claim}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The Central Claim; sentence case, as in the other chapters (style guide gap item 22)
\section{The central claim}\label{sec:discussion:claim}
```

### E63 — What is new in this thesis
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: What Is New in This Thesis; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{What Is New in This Thesis}\label{sec:discussion:contributions}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: What Is New in This Thesis; sentence case, as in the other chapters (style guide gap item 22)
\section{What is new in this thesis}\label{sec:discussion:contributions}
```

### E64 — Handling noise realistically
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Handling Noise Realistically; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{Handling Noise Realistically}\label{sec:discussion:noise}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Handling Noise Realistically; sentence case, as in the other chapters (style guide gap item 22)
\section{Handling noise realistically}\label{sec:discussion:noise}
```

### E65 — The antibodies
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: The Antibodies; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{The Antibodies}\label{sec:discussion:antibodies}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: The Antibodies; sentence case, as in the other chapters (style guide gap item 22)
\section{The antibodies}\label{sec:discussion:antibodies}
```

### E66 — Dependence on the dataset
- file: thesis.tex
- kind: title
- why: a question or claim made a topic (style guide gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: No Single Method Fits Every Dataset; a question or claim made a topic (style guide gap item 24)
OLD:
```latex
\section{No Single Method Fits Every Dataset}\label{sec:discussion:conditional}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: No Single Method Fits Every Dataset; a question or claim made a topic (style guide gap item 24)
\section{Dependence on the dataset}\label{sec:discussion:conditional}
```

### E67 — Simple methods
- file: thesis.tex
- kind: title
- why: a question or claim made a topic (style guide gap item 24).
- todo_line: % TODO: manual review -- edited your text: title was: Simple Is Often Better; a question or claim made a topic (style guide gap item 24)
OLD:
```latex
\section{Simple Is Often Better}\label{sec:discussion:simple}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Simple Is Often Better; a question or claim made a topic (style guide gap item 24)
\section{Simple methods}\label{sec:discussion:simple}
```

### E68 — Future directions
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Future Directions; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section{Future Directions}\label{sec:discussion:future}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Future Directions; sentence case, as in the other chapters (style guide gap item 22)
\section{Future directions}\label{sec:discussion:future}
```

### E69 — Summary of research contributions
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Summary of Research Contributions; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\chapter*{Summary of Research Contributions}
% [notes moved to thesis_notes_moved.md]
\addcontentsline{toc}{chapter}{Summary of Research Contributions}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Summary of Research Contributions; sentence case, as in the other chapters (style guide gap item 22)
\chapter*{Summary of research contributions}
% [notes moved to thesis_notes_moved.md]
\addcontentsline{toc}{chapter}{Summary of research contributions}
```

### E70 — Research papers
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Research Papers; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section*{Research Papers}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Research Papers; sentence case, as in the other chapters (style guide gap item 22)
\section*{Research papers}
```

### E71 — Conference presentations
- file: thesis.tex
- kind: title
- why: sentence case, as in the other chapters (style guide gap item 22).
- todo_line: % TODO: manual review -- edited your text: title was: Conference Presentations; sentence case, as in the other chapters (style guide gap item 22)
OLD:
```latex
\section*{Conference Presentations}
```
NEW:
```latex
% TODO: manual review -- edited your text: title was: Conference Presentations; sentence case, as in the other chapters (style guide gap item 22)
\section*{Conference presentations}
```

## For the author (not changed)

- Chapter 2 `\section{Overview}` has no text: it is followed straight by `\section{Molecular representations in QSAR}`. Either write the overview or delete the heading; not done here because neither is a title change.
- Two comments in Chapter 6 give typed section numbers that the new Results parent will change: "The label-information result is one sentence at the end of Section 6.6" and "% Numbers repeated from Sections 6.3-6.6 above; sources there." After the edits these are 6.3.4 and 6.3.1 to 6.3.4. Comments are never edited.
- A Chapter 1 notes comment names an old Chapter 4 title: "% Paragraph 3: Chapter 4 (The flaws of the similarity assumption)". Not edited (comment). The title stays "Limitations of the similarity assumption" (sentence case only), so the comment and the title still differ; style guide 19 asks for one name.
- Chapter 4 has no Methods or Results sections; it is ordered by topic (representations, pairwise comparisons, consequences). The titles are kept in that order. Giving it the Introduction/Methods/Results order would move text, which this task does not do.
- Chapter 5 keeps two methods sections ("The hybrid search", "Experimental setup"). One `\section{Methods}` parent would need a fourth heading level, because "The search space" and "Search grid per dataset" already hold subsubsections. If you want one parent, the subsubsections would become `\paragraph` headings.
- `\paragraph` headings are not in scope and are unchanged. They are already in sentence case.
- "NoiseInject: a benchmarking framework" is kept. It is the only colon title left; "The NoiseInject package" would avoid the colon, but its child is "Package description", so the two would repeat.
- Chapter 7 keeps "What the noise is" (the guide lists it as good), so its Methods siblings are not all noun phrases ("Noise detection", "Noise mitigation", ...). "Noise types and settings" would make them parallel if you prefer.
- THESIS_OUTLINE.md uses the old titles. It is not edited by this plan.

## Checker notes

Checked 2026-10-10 against thesis.tex. All 65 remaining OLD blocks occur exactly once (python3 -I, str.count), none overlaps another, and every NEW keeps the same `\label` keys. No new numbers, no banned words. Titles with no AI marker are her text, so wording was changed only where style guide 19 or gap items 22-24 name a fault (title case, "Results:" prefix, question, claim, informal chapter title). Changes that were taste only were reverted to her wording or deleted.

- Every todo_line on her titles now reads `% TODO: manual review -- edited your text: title was: <old>; <what changed and why>`, as the rules require. E2 keeps its plain form: that heading was added by an earlier AI session (its CODE INPUT note asks for the title to be reviewed).
- Deleted (wording change only, nothing wrong): E3 Evaluation methodology, E14 Flaws in similarity and other pairwise computations, E15 A small sweep of published noise-handling methods, E17 Blueprints and their chemical interpretation, E18 Antibodies and beyond, E41 How each group is tested, and QM9 as the clean control.
- Reduced to sentence case only (the reword was taste): E4 Chapter 3 title, E13 Chapter 4 title (the gerund "Testing ..." also did not match what the chapter does, which is show where the assumption fails), E32 Chapter 6 title, E61, E62, E63, E64, E65 (Chapter 8 sections).
- Reduced to heading level only (the reword was taste): E6, E7, E8, E10 (Chapter 3), E23, E28 (Chapter 5), E44, E48, E52, E53, E57 (Chapter 7).
- E31: prefix dropped, her wording kept ("Features chosen across separate sites").
- E43, E56: prefix dropped and sentence case; her wording kept. E56 had been renamed "Noise-robust training", which duplicated the Methods subsection name and lost her wording.
- E55: kept "inside". The plan's "before" was not accurate: the section also reports a second experiment (n15) in which the search is run on clean against damaged labels, with no cleaning.
- E24: why/todo text rewritten (it had said "wording unchanged" for a reworded title).
- Kept as planned: E1, E2, E5, E9, E11, E12, E16, E19, E20, E21, E22, E24 (title), E25, E26, E27, E29, E30, E33-E40, E42, E45, E46, E47, E49, E50, E51, E54, E58-E60, E66-E71.
- Table and Rationale updated to match; the "For the author" line about Chapter 7 and 8 both using "handling" was removed (Chapter 8 now keeps "Handling noise realistically"), and the Chapter 4 line was updated.
- Not changed, for the author: adding a `\section{Results}` parent to Chapter 3 (E5) is not required by gap item 23, which names only the "Results:" prefixes in Chapters 5-7. It is kept for the same chapter order; to keep Chapter 3 as it was, drop E5 to E12 together (E5 adds the parent; E5-E12 change only the heading levels under it).
