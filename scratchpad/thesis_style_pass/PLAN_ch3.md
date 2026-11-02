# Plan: Chapter 3 (chap:robustness), chemistry, transitions, register, captions, long AI sentences

Source: /Users/apunt/repos/KIRBy/thesis.tex as of 2026-10-10. Every OLD block was checked with python3 -I str.count to be an exact, unique substring of the chapter. The edits do not overlap. Edits are in file order.

The paragraph-level edits to the author's text put the new "% TODO: manual review -- edited your text: ..." line above the whole paragraph. So OLD starts at the paragraph start, even when only one sentence changes.

## Edits

### E1 — Look back to Chapter 2 at the chapter opening; fix broken first sentence
- file: thesis.tex
- kind: new-paragraph
- why: Style guide item 1 and gap 21: B and V open every chapter by naming the previous chapter (B: "In the previous chapter, I benchmarked ... In this chapter, I describe ..."); Chapter 3 opens with the literature. The author sentence after it also had "direction comparisons", "responds", "remains" (style guide item 2 typo list); fixed as trivial grammar.
- todo_line: % TODO: manual review   (above the new paragraph only; the grammar fix in the next sentence is trivial)
OLD:
```latex
While there have been many investigations into the effect of noise on molecular property prediction, direction comparisons between how different molecular representations responds to label noise remains limited.
```
NEW:
```latex
% TODO: manual review
Chapter~\ref{chap:background} introduced the representations, the models and the seven noise conditions used in this thesis. Here we ask which half of a predictor, the representation or the model, decides how much predictive performance it keeps as label noise rises.

While there have been many investigations into the effect of noise on molecular property prediction, direct comparisons of how different molecular representations respond to label noise remain limited.
```

### E2 — Follow the "remove duplicate" TODO in the Methods lead-in
- file: thesis.tex
- kind: author-text
- why: The author's own TODO above the paragraph asks for the second sentence to be removed; "variants models" is a typo (style guide item 2).
- todo_line: % TODO: manual review -- edited your text: removed the second sentence, as your TODO above asks, because Noise Conditions and Levels says the same; "variants models" -> "variant models". (goes below the existing TODO line, which stays)
OLD:
```latex
We looked at six representations paired with thirteen models, and several variants models, across four prediction tasks. Every pairing was run under Gaussian, grouped-wider and grouped-shifted noise at seven noise levels, and the other four noise conditions (Section~\ref{sec:noise_injection}) were run on selected pairings. The hyperparameters
```
NEW:
```latex
% TODO: manual review -- edited your text: removed the second sentence, as your TODO above asks, because Noise Conditions and Levels says the same; "variants models" -> "variant models".
We looked at six representations paired with thirteen models, and several variant models, across four prediction tasks. The hyperparameters
```

### E3 — Caco-2 endpoint name
- file: thesis.tex
- kind: author-text
- why: Efflux ratio and permeability are different quantities; Chapter 5 labels the same OpenADMET set "log$_{10}$ efflux ratio" (chemistry guide item 9). Not checked against the Chapter 3 code.
- todo_line: % TODO: manual review -- edited your text: "Caco-2 efflux permeability" became "the Caco-2 efflux ratio", because an efflux ratio is a ratio of two permeabilities, not a permeability; confirm Chapter 3 used the log10 efflux ratio, as Chapter 5 does (chemistry guide item 9).
OLD:
```latex
We repeated these experiments on three molecular property datasets with experimentally determined endpoints. From the OpenADMET initiative \citep{openadmet}, we used LogD and Caco-2 efflux permeability, with $N = 5{,}039$ and $N = 2{,}161$ respectively.
```
NEW:
```latex
% TODO: manual review -- edited your text: "Caco-2 efflux permeability" became "the Caco-2 efflux ratio", because an efflux ratio is a ratio of two permeabilities, not a permeability; confirm Chapter 3 used the log10 efflux ratio, as Chapter 5 does (chemistry guide item 9).
We repeated these experiments on three molecular property datasets with experimentally determined endpoints. From the OpenADMET initiative \citep{openadmet}, we used LogD and the Caco-2 efflux ratio, with $N = 5{,}039$ and $N = 2{,}161$ respectively.
```

### E4 — Define LogD, Caco-2 efflux ratio and hERG at first use in this chapter
- file: thesis.tex
- kind: new-paragraph
- why: LogD, Caco-2 and hERG are used about 100 times before any definition (LogD is defined only in Chapter 5); B and V gloss every assay term at first use (chemistry guide item 16, wording from its section 3).
- todo_line: % TODO: manual review
OLD:
```latex
Unlike QM9's 10 replicates, which are 10 independent draws from one pool, each assay dataset is a single fixed set in which every molecule is tested exactly once and nothing is redrawn.
```
NEW:
```latex
Unlike QM9's 10 replicates, which are 10 independent draws from one pool, each assay dataset is a single fixed set in which every molecule is tested exactly once and nothing is redrawn.

% TODO: manual review
% TODO(cite): one source per endpoint (log D, the Caco-2 efflux ratio, hERG); none added.
Log $D$ is the logarithm of the distribution coefficient of a molecule between octanol and water, usually measured at pH 7.4. It counts every charge state, whereas log $P$ counts only the neutral form, so a basic amine that is protonated at that pH lowers log $D$ below log $P$. Caco-2 cells come from a human colorectal carcinoma and grow as a layer that models absorption in the gut. Permeability across the layer is measured in both directions. The efflux ratio is the permeability from the basolateral to the apical side divided by that in the other direction. A ratio above about 2 suggests the compound is pumped out by a transporter such as P-glycoprotein. The hERG gene encodes a cardiac potassium channel. Drugs that block it can prolong the QT interval and cause arrhythmia, so hERG binding is screened early in drug discovery. The hERG $K_i$ dataset holds binding affinities for this channel, as p$K_i$ values.
```

### E5 — Bemis--Murcko definition and RDKit sanitization
- file: thesis.tex
- kind: author-text
- why: Bemis--Murcko is used throughout without a definition (chemistry guide item 15); "sanitized ... to remove invalid valence states" misstates RDKit, which raises an error on an invalid valence (chemistry guide item 20). Bemis1996 is in refs.bib.
- todo_line: % TODO: manual review -- edited your text: added a one-sentence definition of the Bemis--Murcko scaffold, and corrected what RDKit sanitization does: it rejects a molecule with an invalid valence rather than removing the valence, and it sets aromaticity and bond types.
OLD:
```latex
All splits and CV folds were grouped by the chirality-blind Bemis--Murcko scaffold, implemented such that each distinct acyclic molecule forms its own group, and groups are filled in random order. Prior to modeling, all molecular structures were sanitized and canonicalized using \texttt{RDKit} to remove invalid valence states and standardize atom and bond typing \citep{rdkit}.
```
NEW:
```latex
% TODO: manual review -- edited your text: added a one-sentence definition of the Bemis--Murcko scaffold, and corrected what RDKit sanitization does: it rejects a molecule with an invalid valence rather than removing the valence, and it sets aromaticity and bond types.
All splits and CV folds were grouped by the chirality-blind Bemis--Murcko scaffold, implemented such that each distinct acyclic molecule forms its own group, and groups are filled in random order. The Bemis--Murcko scaffold of a molecule is its ring systems and the chains that link them, with all other side chains removed \citep{Bemis1996}. Prior to modeling, all molecular structures were sanitized with \texttt{RDKit}, which rejects invalid valence states and sets aromaticity and bond types, and were written as canonical SMILES \citep{rdkit}.
```

### E7 — Broken sentence in QM9 data pipeline
- file: thesis.tex
- kind: author-text
- why: Stray full stop and missing words (style guide item 2, "804--806"). The author's TODO line is kept.
- todo_line: % TODO: manual review -- edited your text: joined the two paragraphs and repaired the broken second sentence (". Each of these subsets is split new scaffold split"); the Datasets subsection already says each replicate draws its own subset and scaffold split, which answers your TODO.
OLD:
```latex
% TODO: is this not repetitive? Do I not say this elsewhere? 
Each replicate, using its own random seed, randomly selects a subset from QM9, split using scaffold splits. 

. Each of these subsets is split new scaffold split (Datasets subsection), and the same grouping defines the scaffold families targeted by the grouped noise conditions.
```
NEW:
```latex
% TODO: is this not repetitive? Do I not say this elsewhere? 
% TODO: manual review -- edited your text: joined the two paragraphs and repaired the broken second sentence (". Each of these subsets is split new scaffold split"); the Datasets subsection already says each replicate draws its own subset and scaffold split, which answers your TODO.
Each replicate, using its own random seed, randomly selects a subset from QM9 and splits it by scaffold, as in the Datasets subsection. The same grouping defines the scaffold families targeted by the grouped noise conditions.
```

### E8 — Split 37-word sentence (memory-mapped files)
- file: thesis.tex
- kind: ai-text
- why: AI paragraph; sentence over 30 words.
- todo_line: keep existing
OLD:
```latex
Python writes one file per split, the Rust program reads each file and writes a new one in which the processed labels replace the raw ones, and Python then reads the new files back for training.
```
NEW:
```latex
Python writes one file per split. The Rust program reads each file and writes a new one in which the processed labels replace the raw ones. Python then reads the new files back for training.
```

### E9 — Split two 45-word sentences (record layout)
- file: thesis.tex
- kind: ai-text
- why: AI paragraph; two sentences over 30 words.
- todo_line: keep existing
OLD:
```latex
A record holds the isomeric and canonical SMILES strings in UTF-8 with a length prefix, the raw target value as a 4-byte float, the requested representations as fixed-width blocks, and the processed target value, which includes the noise and normalization, as a 4-byte float. The blocks occupy 2,048 bytes for Sort \& Slice (1,024 16-bit integers), 800 bytes for PDV (200 32-bit floats), 1,536 bytes for ChemBERTa (384 32-bit floats), 4,096 bytes for MHG-GNN (1,024 32-bit floats), and 256 bytes each for Avalon and ECFP4.
```
NEW:
```latex
A record holds the isomeric and canonical SMILES strings in UTF-8 with a length prefix, and the raw target value as a 4-byte float. It then holds the requested representations as fixed-width blocks, and the processed target value, with noise and normalization applied, as a 4-byte float. The blocks occupy 2,048 bytes for Sort \& Slice (1,024 16-bit integers) and 800 bytes for PDV (200 32-bit floats). ChemBERTa takes 1,536 bytes (384 32-bit floats), MHG-GNN 4,096 bytes (1,024 32-bit floats), and Avalon and ECFP4 256 bytes each.
```

### E10 — Split 45- and 32-word sentences (noise draw); drop "region"
- file: thesis.tex
- kind: ai-text
- why: AI paragraph; sentences over 30 words; "its region" is undefined (it means the molecule's own noise scale).
- todo_line: keep existing
OLD:
```latex
Validation labels receive an independent draw, from a seed derived from the
training seed, with the same amount of noise in label units, the same limit and the same affected
families, and the program verifies that validation received as much noise as training. Test labels
are never corrupted, although every split records the per-molecule noise scale its region would
have received, so that held-out molecules can be scored against the pattern applied in training.
```
NEW:
```latex
Validation labels receive an independent draw, from a seed derived from the training seed. They get
the same amount of noise in label units, the same limit and the same affected families, and the
program verifies that validation received as much noise as training. Test labels are never corrupted.
Every split still records the noise scale each molecule would have received, so that held-out
molecules can be scored against the pattern applied in training.
```

### E11 — Split 35-word sentence (per-molecule record); drop "region"
- file: thesis.tex
- kind: ai-text
- why: AI paragraph; sentence over 30 words; "region" undefined.
- todo_line: keep existing
OLD:
```latex
For every molecule in every split, the Rust program records its canonical SMILES, clean label,
injected noise, noise scale, the noise pattern applied to its region, noisy label and the
standardized value written to file.
```
NEW:
```latex
For every molecule in every split, the Rust program records its canonical SMILES, clean label,
noisy label and the standardized value written to file. It also records the injected noise, its
scale and the noise pattern applied.
```

### E12 — QRF uncertainty definition
- file: thesis.tex
- kind: author-text
- why: Broken sentence and a technical slip: it is a leaf, not a tree, whose training labels give the variance.
- todo_line: % TODO: manual review -- edited your text: "each tree holds multiple training molecules" -> each leaf of a tree; "the variance the labels the molecule's given leaf" was missing words; "uncertainty are" -> "is".
OLD:
```latex
The other models follow the same split. For GPs, the epistemic component is the posterior variance of the latent function and the aleatoric component is the learned observation noise \citep{Rasmussen2005}. All variants of the GP produce one number for the whole fit except heteroscedastic. For QRF models, each tree holds multiple training molecules. So for a given molecule, the aleatoric component is the variance the labels the molecule's given leaf. The total aleatoric uncertainty are those variances averaged over the trees.
```
NEW:
```latex
% TODO: manual review -- edited your text: "each tree holds multiple training molecules" -> each leaf of a tree; "the variance the labels the molecule's given leaf" was missing words; "uncertainty are" -> "is".
The other models follow the same split. For GPs, the epistemic component is the posterior variance of the latent function and the aleatoric component is the learned observation noise \citep{Rasmussen2005}. All variants of the GP produce one number for the whole fit except heteroscedastic. For QRF models, each leaf of a tree holds several training molecules. So for a given molecule, the aleatoric component in one tree is the variance of the labels in the leaf the molecule falls into. The total aleatoric uncertainty is those variances averaged over the trees.
```

### E13 — Scaffold name consistency
- file: thesis.tex
- kind: author-text-trivial
- why: "Murcko scaffolds" here, "Bemis--Murcko scaffold" everywhere else in the chapter.
- todo_line: none (trivial)
OLD:
```latex
Each training set was divided into five folds grouped on Murcko scaffolds, and each
```
NEW:
```latex
Each training set was divided into five folds grouped on Bemis--Murcko scaffolds, and each
```

### E14 — Missing article
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
We compute statistics on the held-out molecules, for QM9 the test split and for three assay datasets the held-out fold of the scaffold CV.
```
NEW:
```latex
We compute statistics on the held-out molecules, for QM9 the test split and for the three assay datasets the held-out fold of the scaffold CV.
```

### E15 — Agreement
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar ("models ... its").
- todo_line: none (trivial)
OLD:
```latex
We compared probabilistic models  with its deterministic counterpart using a two-sided Wilcoxon signed-rank test paired on the replicate
```
NEW:
```latex
We compared probabilistic models with their deterministic counterparts using a two-sided Wilcoxon signed-rank test paired on the replicate
```

### E16 — Agreement in the uncertainty questions
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar ("uncertainty rise", "models who").
- todo_line: none (trivial)
OLD:
```latex
First, whether uncertainty rise with label noise, as measured by the mean $u_i$ over held-out molecules at the highest noise level
divided by its value with no added noise, in each replicate or fold. The second pertains to models who report separate aleatoric and epistemic components.
```
NEW:
```latex
First, whether uncertainty rises with label noise, as measured by the mean $u_i$ over held-out molecules at the highest noise level
divided by its value with no added noise, in each replicate or fold. The second pertains to models that report separate aleatoric and epistemic components.
```

### E17 — Agreement
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar ("noises lowers"; style guide item 2).
- todo_line: none (trivial)
OLD:
```latex
For predictive performance, the artificial label noises lowers the impact of the choice of model family
```
NEW:
```latex
For predictive performance, the artificial label noise lowers the impact of the choice of model family
```

### E18 — Caption fig:three_outcomes: dataset name
- file: thesis.tex
- kind: author-text-trivial
- why: "logD" -> "LogD".
- todo_line: none (trivial)
OLD:
```latex
{Share of the variance explained by each term on a) the QM9 HOMO--LUMO gap, b) logD, c) Caco-2 and d) hERG $K_i$,
```
NEW:
```latex
{Share of the variance explained by each term on a) the QM9 HOMO--LUMO gap, b) LogD, c) Caco-2 and d) hERG $K_i$,
```

### E19 — "significantly" without a test
- file: thesis.tex
- kind: author-text
- why: Style guide item 10 and gap 31: no word "significant(ly)" unless a test was run; none was run here.
- todo_line: % TODO: manual review -- edited your text: "varies significantly" -> "varies widely", because no test was run on this comparison; the shares that follow (28.4% and 51.0%) carry the point.
OLD:
```latex
While we observe similar patterns between noise conditions, the predictive performance of grouped-shifted noise in particular varies significantly between replicates, resulting in a larger share for the replicate.
```
NEW:
```latex
% TODO: manual review -- edited your text: "varies significantly" -> "varies widely", because no test was run on this comparison; the shares that follow (28.4% and 51.0%) carry the point.
While we observe similar patterns between noise conditions, the predictive performance under grouped-shifted noise in particular varies widely between replicates, resulting in a larger share for the replicate.
```

### E20 — What is retained
- file: thesis.tex
- kind: author-text
- why: Wrong word: the retention curve measures performance kept, not noise kept.
- todo_line: % TODO: manual review -- edited your text: "the quantity of noise retained" -> "the share of performance retained", since a model retains performance, not noise; "All model's" -> "Every model's".
OLD:
```latex
The variance decomposition tells us what aspect of QSAR modeling affects predictive performance and robustness, however it does not tell us about the characteristics of individual models. All model's performance degrades as the labels get increasingly corrupted with noise (Figure~\ref{fig:curves}). The loss remains small until the artificial noise reaches about half the spread of the clean training labels, and then it begins to drop. The rate it drops and the quantity of noise retained is not necessarily correlated to the model's predictive performance on clean labels.
```
NEW:
```latex
% TODO: manual review -- edited your text: "the quantity of noise retained" -> "the share of performance retained", since a model retains performance, not noise; "All model's" -> "Every model's".
The variance decomposition tells us what aspect of QSAR modeling affects predictive performance and robustness, however it does not tell us about the characteristics of individual models. Every model's performance degrades as the labels get increasingly corrupted with noise (Figure~\ref{fig:curves}). The loss remains small until the artificial noise reaches about half the spread of the clean training labels, and then it begins to drop. The rate at which it drops and the share of performance retained are not necessarily correlated with the model's predictive performance on clean labels.
```

### E21 — Second person and contraction (NGBoost)
- file: thesis.tex
- kind: author-text
- why: Style guide item 2: no "you" and no contractions in the body (none of B, D, V uses them).
- todo_line: % TODO: manual review -- edited your text: removed "you" and the contraction; same meaning.
OLD:
```latex
NGBoost is a standout case in the sense it is highly dependent on its paired molecular representation. Under Gaussian noise it has the highest \aucnorm{} on every one of the six representations, yet also the lowest in predictive performance on four of the six representations. It is one of the worst performing models with respect to $R^2$ on clean data. Its high robustness is partially due to the fact that when you start with a low predictive performance, you don't have as much to lose.
```
NEW:
```latex
% TODO: manual review -- edited your text: removed "you" and the contraction; same meaning.
NGBoost is a standout case in the sense it is highly dependent on its paired molecular representation. Under Gaussian noise it has the highest \aucnorm{} on every one of the six representations, yet also the lowest in predictive performance on four of the six representations. It is one of the worst performing models with respect to $R^2$ on clean data. Its high robustness is partly due to its low predictive performance on clean labels, which leaves it less to lose.
```

### E22 — Condition name consistency
- file: thesis.tex
- kind: author-text-trivial
- why: "group-shifted" for the condition named "grouped-shifted" everywhere else.
- todo_line: none (trivial)
OLD:
```latex
On group-shifted noise, RF has higher \aucnorm{} values than NGBoost on ECFP4, Avalon and ChemBERTa.
```
NEW:
```latex
On grouped-shifted noise, RF has higher \aucnorm{} values than NGBoost on ECFP4, Avalon and ChemBERTa.
```

### E23 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2 (contraction listed at old line 973).
- todo_line: % TODO: manual review -- edited your text: "we've" -> "we have".
OLD:
```latex
As we've seen, the choice of representation can have an impact on noise robustness
```
NEW:
```latex
% TODO: manual review -- edited your text: "we've" -> "we have".
As we have seen, the choice of representation can have an impact on noise robustness
```

### E24 — Caption fig:curves: direction
- file: thesis.tex
- kind: caption
- why: R$^2$ caption without direction (style guide gap 39, old line 978).
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
OLD:
```latex
\caption[Predictive performance against Gaussian label noise on QM9.]{$R^2$ against noise level on the QM9 HOMO--LUMO gap under Gaussian noise, for the eight base models
with the highest \aucnorm{} on ECFP4: a) ECFP4; b) PDV. Medians over 10 replicates. The dashed line marks
the noise level of 1.0 at which $R^2$ is reported.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
\caption[Predictive performance against Gaussian label noise on QM9.]{$R^2$ against noise level on the QM9 HOMO--LUMO gap under Gaussian noise, for the eight base models
with the highest \aucnorm{} on ECFP4: a) ECFP4; b) PDV. Medians over 10 replicates. The dashed line marks
the noise level of 1.0 at which $R^2$ is reported. Higher $R^2$ is better.}
```

### E25 — Caption fig:decoupling: direction and what one point is
- file: thesis.tex
- kind: caption
- why: No direction (gap 39, old line 987).
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
OLD:
```latex
\caption[Clean predictive performance against robustness on QM9.]{Clean $R^2$ against \aucnorm{} for each base model on the QM9 HOMO--LUMO gap with ECFP4, under
a) Gaussian and b) grouped-shifted noise.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
\caption[Clean predictive performance against robustness on QM9.]{Clean $R^2$ against \aucnorm{} for each base model on the QM9 HOMO--LUMO gap with ECFP4, under
a) Gaussian and b) grouped-shifted noise. One point is one base model. Higher is better on both axes.}
```

### E26 — Caption fig:grid
- file: thesis.tex
- kind: caption
- why: No direction (style guide item 16).
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
OLD:
```latex
\caption[Robustness of each base model on each representation on QM9.]{\aucnorm{} of the 13 base models on each representation, on the QM9 HOMO--LUMO gap under a) Gaussian
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness of each base model on each representation on QM9.]{\aucnorm{} (higher is more robust) of the 13 base models on each representation, on the QM9 HOMO--LUMO gap under a) Gaussian
```

### E27 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "it's injected to" -> "it is injected into".
OLD:
```latex
Many studies involving experimental label noise rely on the assumption that noise can be modeled by a Gaussian distribution. We decided to vary the distribution which noise is drawn from, as well as the methodology by which it's injected to specific labels.
```
NEW:
```latex
% TODO: manual review -- edited your text: "it's injected to" -> "it is injected into".
Many studies involving experimental label noise rely on the assumption that noise can be modeled by a Gaussian distribution. We decided to vary the distribution which noise is drawn from, as well as the methodology by which it is injected into specific labels.
```

### E28 — Doubled "from"
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
The shape of the distribution from which the noise was drawn from made little difference.
```
NEW:
```latex
The shape of the distribution from which the noise was drawn made little difference.
```

### E29 — Adverb
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar ("relative stronger").
- todo_line: none (trivial)
OLD:
```latex
NGBoost's relative stronger performance on Gaussian makes sense
```
NEW:
```latex
NGBoost's relatively stronger performance on Gaussian noise makes sense
```

### E30 — Stray "of"
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar (style guide item 2 lists "lowered the \aucnorm{} of across").
- todo_line: none (trivial)
OLD:
```latex
Relative to Gaussian, grouped-shifted noise lowered the \aucnorm{} of across all models and representations (Table~\ref{tab:condition_cost}).
```
NEW:
```latex
Relative to Gaussian, grouped-shifted noise lowered the \aucnorm{} of every model on every representation (Table~\ref{tab:condition_cost}).
```

### E31 — Comma splice
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
Censoring created the largest drop in predictive performance, \aucnorm{} fell by 0.12 to 0.18 for the same QSAR models relative to Gaussian noise (Table~\ref{tab:censoring}).
```
NEW:
```latex
Censoring created the largest drop in predictive performance: \aucnorm{} fell by 0.12 to 0.18 for the same QSAR models relative to Gaussian noise (Table~\ref{tab:censoring}).
```

### E32 — Tense within one comparison
- file: thesis.tex
- kind: author-text-trivial
- why: Style guide item 11: one run reported in past; "started ... was" then "falls".
- todo_line: none (trivial)
OLD:
```latex
Of the 13 base models, RF started ranked seventh, but by level 1.0 it was ranked first. SVM on the other hand falls from first to fifth (Figure~\ref{fig:rank_level}).
```
NEW:
```latex
Of the 13 base models, RF started ranked seventh, but by level 1.0 it was ranked first. SVM on the other hand fell from first to fifth (Figure~\ref{fig:rank_level}).
```

### E33 — Caption fig:deep
- file: thesis.tex
- kind: caption
- why: No direction.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
OLD:
```latex
\caption[Robustness under every noise condition on a subset of models on QM9.]{\aucnorm{} under the noise conditions run on a subset of models, on the QM9 HOMO--LUMO gap:
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under every noise condition on a subset of models on QM9.]{\aucnorm{} (higher is more robust) under the noise conditions run on a subset of models, on the QM9 HOMO--LUMO gap:
```

### E34 — Caption tab:censoring
- file: thesis.tex
- kind: caption
- why: No direction.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
OLD:
```latex
\caption[Robustness under censoring and under Gaussian noise on QM9.]{\aucnorm{} under censoring and under Gaussian noise on the QM9 HOMO--LUMO gap, for the five
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under censoring and under Gaussian noise on QM9.]{\aucnorm{} (higher is more robust) under censoring and under Gaussian noise on the QM9 HOMO--LUMO gap, for the five
```

### E35 — Caption fig:conditions: direction
- file: thesis.tex
- kind: caption
- why: No direction.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
OLD:
```latex
\caption[Random forest under each noise condition on QM9.]{$R^2$ of the random forest against noise level under each noise condition, on the
QM9 HOMO--LUMO gap with ECFP4. Medians over 10 replicates. Censoring is not shown, as its level is a fraction of labels clipped.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
\caption[Random forest under each noise condition on QM9.]{$R^2$ of the random forest against noise level under each noise condition, on the
QM9 HOMO--LUMO gap with ECFP4. Medians over 10 replicates; higher $R^2$ is better. Censoring is not shown, as its level is a fraction of labels clipped.}
```

### E36 — Caption fig:rank_level: direction of rank
- file: thesis.tex
- kind: caption
- why: Does not say whether rank 1 is best.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
OLD:
```latex
\caption[Rank of each base model at each noise level on QM9.]{Median rank of each base model by $R^2$ at each noise level, on the QM9 HOMO--LUMO gap with ECFP4,
under a) Gaussian and b) grouped-shifted noise. Models are ranked within each replicate; tied medians give half ranks.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
\caption[Rank of each base model at each noise level on QM9.]{Median rank of each base model by $R^2$ at each noise level, on the QM9 HOMO--LUMO gap with ECFP4,
under a) Gaussian and b) grouped-shifted noise. Rank 1 is the highest $R^2$. Models are ranked within each replicate; tied medians give half ranks.}
```

### E37 — Broken phrase in assay-section opening
- file: thesis.tex
- kind: author-text-trivial
- why: Missing words ("results coming on HOMO--LUMO gap").
- todo_line: none (trivial)
OLD:
```latex
So far, we have only discussed results coming on HOMO--LUMO gap from QM9, a computed property whose labels carry no measurement error.
```
NEW:
```latex
So far, we have only discussed results on the QM9 HOMO--LUMO gap, a computed property whose labels carry no measurement error.
```

### E38 — p$K_i$ typography
- file: thesis.tex
- kind: author-text-trivial
- why: "pK$_i$" (roman K) against "p$K_i$" in the rest of the thesis (chemistry guide item 10). Checker removed the planned change of 0.54 to "0.44 on average, SD 0.54": the source was not checked, and the next sentence ("about one unit of that estimated laboratory error") is built on 0.54, so it moved to For the author.
- todo_line: none (trivial)
OLD:
```latex
Repeat measurements of the same compound disagree by about 0.54 log units for pK$_i$ \citep{Kalliokoski2013, Kramer2012}.
```
NEW:
```latex
Repeat measurements of the same compound disagree by about 0.54 log units for p$K_i$ \citep{Kalliokoski2013, Kramer2012}.
```

### E39 — Contraction and "worth remembering"
- file: thesis.tex
- kind: author-text
- why: Style guide item 2 (contraction) and the AI-phrasing guide ("it is worth noting").
- todo_line: % TODO: manual review -- edited your text: removed the contraction and "it's worth remembering"; same content.
OLD:
```latex
As on QM9, the noise distribution does not change the predictive capabilities of a model (Table~\ref{tab:assay_shape}). Laplace, Student-$t$ and outlier noise were run on seven models, RF, the two GPs, SVM, NGBoost, BNN-$\alpha$ (var. head) and VBLL-$\beta$ (het.), on ECFP4, PDV and ChemBERTa. Each delta below is measured from the same model's \aucnorm{} under Gaussian noise. RF and the two GPs never lose
more than 0.03 of \aucnorm{} under these three conditions or under grouped-wider noise, on any dataset or representation. SVM and NGBoost again move in opposite directions, SVM never loses \aucnorm{} under any of the four, while NGBoost loses under all four on Caco-2 and hERG $K_i$. The two networks move furthest in both directions. It's worth remembering that while we are injecting artificial noise, the assay measurements carry their own source of noise whose properties are unknown, so no ``clean'' label is truly clean.
```
NEW:
```latex
% TODO: manual review -- edited your text: removed the contraction and "it's worth remembering"; same content.
As on QM9, the noise distribution does not change the predictive capabilities of a model (Table~\ref{tab:assay_shape}). Laplace, Student-$t$ and outlier noise were run on seven models, RF, the two GPs, SVM, NGBoost, BNN-$\alpha$ (var. head) and VBLL-$\beta$ (het.), on ECFP4, PDV and ChemBERTa. Each delta below is measured from the same model's \aucnorm{} under Gaussian noise. RF and the two GPs never lose
more than 0.03 of \aucnorm{} under these three conditions or under grouped-wider noise, on any dataset or representation. SVM and NGBoost again move in opposite directions, SVM never loses \aucnorm{} under any of the four, while NGBoost loses under all four on Caco-2 and hERG $K_i$. The two networks move furthest in both directions. While we inject artificial noise, the assay measurements carry noise of their own whose properties are unknown, so no ``clean'' label is truly clean.
```

### E40 — Table tab:assay_shape: dataset names
- file: thesis.tex
- kind: author-text-trivial
- why: "logD" -> "LogD"; "hERG K$_i$" (roman K) -> "hERG $K_i$" as in the rest of the thesis (chemistry guide items 10, 11).
- todo_line: none (trivial)
OLD:
```latex
logD & RF & 0.935 & +0.018 & +0.006 & +0.007 & +0.021 \\
logD & NGBoost & 0.924 & 0.000 & -0.028 & -0.032 & -0.009 \\
logD & SVM & 0.879 & +0.044 & +0.023 & +0.027 & +0.044 \\
logD & GP & 0.891 & +0.004 & +0.003 & -0.001 & +0.006 \\
logD & GP (het.) & 0.896 & 0.000 & -0.002 & -0.007 & -0.003 \\
logD & BNN-$\alpha$ (var. head) & 0.881 & -0.013 & -0.013 & -0.024 & -0.029 \\
logD & VBLL-$\beta$ (het.) & 0.796 & +0.010 & +0.012 & +0.020 & +0.068 \\
Caco-2 & RF & 0.865 & +0.035 & +0.028 & +0.026 & +0.053 \\
Caco-2 & NGBoost & 0.878 & -0.069 & -0.038 & -0.060 & -0.147 \\
Caco-2 & SVM & 0.841 & +0.049 & +0.051 & +0.031 & +0.018 \\
Caco-2 & GP & 0.809 & +0.039 & +0.034 & +0.020 & +0.023 \\
Caco-2 & GP (het.) & 0.815 & +0.035 & +0.032 & +0.011 & +0.019 \\
Caco-2 & BNN-$\alpha$ (var. head) & 0.787 & -0.110 & +0.092 & -0.043 & -0.017 \\
Caco-2 & VBLL-$\beta$ (het.) & 0.707 & +0.036 & +0.025 & +0.015 & +0.032 \\
hERG K$_i$ & RF & 0.898 & +0.015 & -0.006 & +0.016 & -0.012 \\
hERG K$_i$ & NGBoost & 0.842 & -0.031 & -0.027 & -0.042 & -0.093 \\
hERG K$_i$ & SVM & 0.859 & +0.051 & +0.035 & +0.025 & +0.055 \\
hERG K$_i$ & GP & 0.831 & +0.022 & +0.034 & +0.025 & +0.030 \\
hERG K$_i$ & GP (het.) & 0.843 & +0.010 & +0.036 & +0.015 & +0.021 \\
hERG K$_i$ & BNN-$\alpha$ (var. head) & 0.809 & -0.034 & -0.032 & -0.063 & -0.009 \\
hERG K$_i$ & VBLL-$\beta$ (het.) & 0.836 & +0.007 & -0.047 & +0.005 & -0.002 \\
```
NEW:
```latex
LogD & RF & 0.935 & +0.018 & +0.006 & +0.007 & +0.021 \\
LogD & NGBoost & 0.924 & 0.000 & -0.028 & -0.032 & -0.009 \\
LogD & SVM & 0.879 & +0.044 & +0.023 & +0.027 & +0.044 \\
LogD & GP & 0.891 & +0.004 & +0.003 & -0.001 & +0.006 \\
LogD & GP (het.) & 0.896 & 0.000 & -0.002 & -0.007 & -0.003 \\
LogD & BNN-$\alpha$ (var. head) & 0.881 & -0.013 & -0.013 & -0.024 & -0.029 \\
LogD & VBLL-$\beta$ (het.) & 0.796 & +0.010 & +0.012 & +0.020 & +0.068 \\
Caco-2 & RF & 0.865 & +0.035 & +0.028 & +0.026 & +0.053 \\
Caco-2 & NGBoost & 0.878 & -0.069 & -0.038 & -0.060 & -0.147 \\
Caco-2 & SVM & 0.841 & +0.049 & +0.051 & +0.031 & +0.018 \\
Caco-2 & GP & 0.809 & +0.039 & +0.034 & +0.020 & +0.023 \\
Caco-2 & GP (het.) & 0.815 & +0.035 & +0.032 & +0.011 & +0.019 \\
Caco-2 & BNN-$\alpha$ (var. head) & 0.787 & -0.110 & +0.092 & -0.043 & -0.017 \\
Caco-2 & VBLL-$\beta$ (het.) & 0.707 & +0.036 & +0.025 & +0.015 & +0.032 \\
hERG $K_i$ & RF & 0.898 & +0.015 & -0.006 & +0.016 & -0.012 \\
hERG $K_i$ & NGBoost & 0.842 & -0.031 & -0.027 & -0.042 & -0.093 \\
hERG $K_i$ & SVM & 0.859 & +0.051 & +0.035 & +0.025 & +0.055 \\
hERG $K_i$ & GP & 0.831 & +0.022 & +0.034 & +0.025 & +0.030 \\
hERG $K_i$ & GP (het.) & 0.843 & +0.010 & +0.036 & +0.015 & +0.021 \\
hERG $K_i$ & BNN-$\alpha$ (var. head) & 0.809 & -0.034 & -0.032 & -0.063 & -0.009 \\
hERG $K_i$ & VBLL-$\beta$ (het.) & 0.836 & +0.007 & -0.047 & +0.005 & -0.002 \\
```

### E41 — Caption tab:assay_shape
- file: thesis.tex
- kind: caption
- why: No direction.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
OLD:
```latex
\caption[Robustness under the noise distributions on the three assay datasets.]{\aucnorm{} under Gaussian noise, and its change under each other noise condition, for a subset of models on the three assay datasets with ECFP4; negative is a loss. Medians over five scaffold folds.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39).
\caption[Robustness under the noise distributions on the three assay datasets.]{\aucnorm{} (higher is more robust) under Gaussian noise, and its change under each other noise condition, for a subset of models on the three assay datasets with ECFP4; negative is a loss. Medians over five scaffold folds.}
```

### E42 — Condition name, stray "at", dataset name
- file: thesis.tex
- kind: author-text-trivial
- why: "group-shifted", "at under higher", "logD" against "LogD" elsewhere (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
Across all datasets, models struggle to fit with group-shifted noise the most, producing lower \aucnorm{} values across the board. For grouped-shifted, RF is the dominant model on hERG $K_i$ and logD, where it ranks first or second for \aucnorm{} with every representation. For predictive performance at under higher label noise (noise level 1.0), which is more impacted by the clean label performance, SVM ranks first to third on logD while RF ranks fifth to ninth across datasets.
```
NEW:
```latex
Across all datasets, models struggle to fit with grouped-shifted noise the most, producing lower \aucnorm{} values across the board. For grouped-shifted, RF is the dominant model on hERG $K_i$ and LogD, where it ranks first or second for \aucnorm{} with every representation. For predictive performance under higher label noise (noise level 1.0), which is more impacted by the clean label performance, SVM ranks first to third on LogD while RF ranks fifth to ninth across datasets.
```

### E43 — Dataset name "hERG" -> "hERG $K_i$"
- file: thesis.tex
- kind: author-text-trivial
- why: The dataset is "hERG $K_i$" everywhere else; the thesis also has a separate Lhasa hERG set, so bare "hERG" is ambiguous.
- todo_line: none (trivial)
OLD:
```latex
However, on hERG and Caco-2, it loses the most with increased noise, particularly on Gaussian and grouped-shifted noise. The effects are not as pronounced on grouped-wider, although it still ranks among the least robust there.  XGBoost degrades in the same way on hERG and Caco-2.
```
NEW:
```latex
However, on hERG $K_i$ and Caco-2, it loses the most with increased noise, particularly on Gaussian and grouped-shifted noise. The effects are not as pronounced on grouped-wider, although it still ranks among the least robust there. XGBoost degrades in the same way on hERG $K_i$ and Caco-2.
```

### E44 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
In general, the relative \aucnorm{} model rankings match between QM9 and logD, as well as between Caco-2 and hERG $K_i$ with every representation except ChemBERTa.
```
NEW:
```latex
In general, the relative \aucnorm{} model rankings match between QM9 and LogD, as well as between Caco-2 and hERG $K_i$ with every representation except ChemBERTa.
```

### E45 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
second or third on QM9 and first on logD at five of six. The GP has the highest predictive performance on clean labels on all three assay datasets with nearly every representation, and ranks third to ninth for \aucnorm. NGBoost, the most robust model on QM9, ranks first to third on logD,
```
NEW:
```latex
second or third on QM9 and first on LogD at five of six. The GP has the highest predictive performance on clean labels on all three assay datasets with nearly every representation, and ranks third to ninth for \aucnorm. NGBoost, the most robust model on QM9, ranks first to third on LogD,
```

### E46 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
$K_i$ and last on logD at every representation, from one of the lowest clean $R^2$ values.
```
NEW:
```latex
$K_i$ and last on LogD at every representation, from one of the lowest clean $R^2$ values.
```

### E47 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
It is the highest on logD and Caco-2 under grouped-wider noise as well, but under grouped-shifted noise it falls.
```
NEW:
```latex
It is the highest on LogD and Caco-2 under grouped-wider noise as well, but under grouped-shifted noise it falls.
```

### E48 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "it's" -> "it is".
OLD:
```latex
There is a slight consideration with using both \aucnorm{} and predictive performance side-by-side. \aucnorm{} measures the fraction of its own clean $R^2$ that a model keeps, so predictive performance on clean labels works against it by construction. A model that already predicts poorly has little to lose and often scores a higher \aucnorm, while a model that starts with strong performance has more to lose. The GP and NGBoost are the clearest cases. The GP has among the highest predictive performance on clean labels and a middling \aucnorm, while NGBoost has the highest \aucnorm{} across the board from one of the lowest clean $R^2$ values. While NGBoost's predictive performance is highly dependent on its representation, its robustness is not. On QM9 it is both the most robust model and the lowest ranked model for predictive performance up to a noise level of 0.75 with every representation. But with PDV it's able to retain some predictive performance under noise, finishing highest across all the models at noise level 1.5.
```
NEW:
```latex
% TODO: manual review -- edited your text: "it's" -> "it is".
There is a slight consideration with using both \aucnorm{} and predictive performance side-by-side. \aucnorm{} measures the fraction of its own clean $R^2$ that a model keeps, so predictive performance on clean labels works against it by construction. A model that already predicts poorly has little to lose and often scores a higher \aucnorm, while a model that starts with strong performance has more to lose. The GP and NGBoost are the clearest cases. The GP has among the highest predictive performance on clean labels and a middling \aucnorm, while NGBoost has the highest \aucnorm{} across the board from one of the lowest clean $R^2$ values. While NGBoost's predictive performance is highly dependent on its representation, its robustness is not. On QM9 it is both the most robust model and the lowest ranked model for predictive performance up to a noise level of 0.75 with every representation. But with PDV it is able to retain some predictive performance under noise, finishing highest across all the models at noise level 1.5.
```

### E49 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
VBLL networks perform poorly on clean labels, are the least robust to noise on logD, yet are the most robust on Caco-2 and hERG $K_i$.
```
NEW:
```latex
VBLL networks perform poorly on clean labels, are the least robust to noise on LogD, yet are the most robust on Caco-2 and hERG $K_i$.
```

### E50 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
the models are largely unaffected by the noise on the HOMO--LUMO gap and on logD, where every base model with ECFP4 still keeps at least 93\% of its clean $R^2$;
```
NEW:
```latex
the models are largely unaffected by the noise on the HOMO--LUMO gap and on LogD, where every base model with ECFP4 still keeps at least 93\% of its clean $R^2$;
```

### E51 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
VBLL-$\alpha$ is the earliest to fall on logD, dropping below 90\% of its clean $R^2$ at a level of 0.5 and reaching 38\% by 1.5.
```
NEW:
```latex
VBLL-$\alpha$ is the earliest to fall on LogD, dropping below 90\% of its clean $R^2$ at a level of 0.5 and reaching 38\% by 1.5.
```

### E52 — Caption fig:curves_all: direction and dataset name
- file: thesis.tex
- kind: caption
- why: No direction (gap 39, old line 1161); "logD" (chemistry guide item 11).
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39); "logD" -> "LogD".
OLD:
```latex
\caption[Predictive performance against noise level on every dataset.]{$R^2$ against noise level for the 13 base models on every dataset with ECFP4. Rows are
datasets and columns noise conditions: a)--d) QM9 HOMO--LUMO gap; e)--h) logD; i)--l) Caco-2; m)--p) hERG $K_i$.
Medians over 10 replicates on QM9 and five scaffold folds on the assay datasets.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39); "logD" -> "LogD".
\caption[Predictive performance against noise level on every dataset.]{$R^2$ against noise level for the 13 base models on every dataset with ECFP4. Rows are
datasets and columns noise conditions: a)--d) QM9 HOMO--LUMO gap; e)--h) LogD; i)--l) Caco-2; m)--p) hERG $K_i$.
Medians over 10 replicates on QM9 and five scaffold folds on the assay datasets. Higher $R^2$ is better.}
```

### E53 — Caption fig:assay
- file: thesis.tex
- kind: caption
- why: No direction; "logD" (chemistry guide item 11).
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); "logD" -> "LogD".
OLD:
```latex
\caption[Robustness on the three assay datasets.]{\aucnorm{} of the 13 base models with ECFP4 on a) logD, b) Caco-2 and c) hERG $K_i$,
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric: higher \aucnorm{} is more robust (style guide item 16, gap 39); "logD" -> "LogD".
\caption[Robustness on the three assay datasets.]{\aucnorm{} (higher is more robust) of the 13 base models with ECFP4 on a) LogD, b) Caco-2 and c) hERG $K_i$,
```

### E54 — Missing word
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
Extending that concept, we are also interested how these shares change under increasing label noise and compare to the variance decomposition of \aucnorm.
```
NEW:
```latex
Extending that concept, we are also interested in how these shares change under increasing label noise and compare to the variance decomposition of \aucnorm.
```

### E55 — Missing word
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
Conversely, it falls well below the choice model family for robustness.
```
NEW:
```latex
Conversely, it falls well below the choice of model family for robustness.
```

### E56 — Caption fig:transfer: direction of rank
- file: thesis.tex
- kind: caption
- why: Does not say whether rank 1 is most robust.
- todo_line: % TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
OLD:
```latex
\caption[Robustness rank of each model on QM9 and the three assay datasets.]{Rank by \aucnorm{} of each base model on every dataset, with ECFP4 under Gaussian noise.
Models are ordered by their rank on the QM9 HOMO--LUMO gap. Ranks are among all 19 models; the six variant models are not drawn.}
```
NEW:
```latex
% TODO: manual review -- edited your text: added the direction of the metric (style guide item 16, gap 39).
\caption[Robustness rank of each model on QM9 and the three assay datasets.]{Rank by \aucnorm{} of each base model on every dataset, with ECFP4 under Gaussian noise.
Models are ordered by their rank on the QM9 HOMO--LUMO gap. Rank 1 is the most robust. Ranks are among all 19 models; the six variant models are not drawn.}
```

### E57 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
and under Gaussian noise on logD and hERG $K_i$ (Additional file~\ref{app:noise:sec:rf_vs_qrf_assay}).
```
NEW:
```latex
and under Gaussian noise on LogD and hERG $K_i$ (Additional file~\ref{app:noise:sec:rf_vs_qrf_assay}).
```

### E58 — Missing noun
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
The Bayesian transformations that we used to transform feed-forward deterministic NNs into probabilistic produced a slightly larger \aucnorm
```
NEW:
```latex
The Bayesian transformations that we used to transform feed-forward deterministic NNs into probabilistic ones produced a slightly larger \aucnorm
```

### E59 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "we've discussed how much predictive capabilities" -> "we have discussed how much predictive performance".
OLD:
```latex
Until now we've discussed how much predictive capabilities a model is able to retain while its training labels are corrupted with noise.
```
NEW:
```latex
% TODO: manual review -- edited your text: "we've discussed how much predictive capabilities" -> "we have discussed how much predictive performance".
Until now we have discussed how much predictive performance a model is able to retain while its training labels are corrupted with noise.
```

### E60 — Dataset name "logD" -> "LogD"
- file: thesis.tex
- kind: author-text-trivial
- why: One form for the dataset name; "LogD" 138 times, "logD" only in this chapter (chemistry guide item 11).
- todo_line: none (trivial)
OLD:
```latex
Model & QM9 & logD & Caco-2 & hERG $K_i$ & Aleatoric & Epistemic \\
```
NEW:
```latex
Model & QM9 & LogD & Caco-2 & hERG $K_i$ & Aleatoric & Epistemic \\
```

### E61 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "we don't know" -> "it does not tell us".
OLD:
```latex
While this tells us that the models are reportedly less certain under noisy data, we don't know what the model is attributing this uncertainty to.
```
NEW:
```latex
% TODO: manual review -- edited your text: "we don't know" -> "it does not tell us".
While this tells us that the models are reportedly less certain under noisy data, it does not tell us what the model is attributing this uncertainty to.
```

### E62 — Comma splice
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
The full-BNNs with a variance head behave as expected (Figure~\ref{fig:decomposition}), the epistemic component grows by at most 0.43 of the aleatoric growth in every configuration where the ratio can be computed.
```
NEW:
```latex
The full-BNNs with a variance head behave as expected (Figure~\ref{fig:decomposition}): the epistemic component grows by at most 0.43 of the aleatoric growth in every configuration where the ratio can be computed.
```

### E63 — Comma splice and condition name
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar; "group-shifted" -> "grouped-shifted".
- todo_line: none (trivial)
OLD:
```latex
The noise conditions follow the pattern found in robustness, the three conditions which alternate the noise distribution show similar decompositions to those under Gaussian noise, while group-shifted raises the epistemic share
```
NEW:
```latex
The noise conditions follow the pattern found in robustness: the three conditions which alternate the noise distribution show similar decompositions to those under Gaussian noise, while grouped-shifted noise raises the epistemic share
```

### E64 — Contraction
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "doesn't" -> "does not".
OLD:
```latex
The aleatoric component indicates that label noise increased, but it doesn't tell us by how much (Figure~\ref{fig:aleatoric_injected}).
```
NEW:
```latex
% TODO: manual review -- edited your text: "doesn't" -> "does not".
The aleatoric component indicates that label noise increased, but it does not tell us by how much (Figure~\ref{fig:aleatoric_injected}).
```

### E66 — Caption fig:aleatoric_injected: dataset name
- file: thesis.tex
- kind: author-text-trivial
- why: "logD" -> "LogD".
- todo_line: none (trivial)
OLD:
```latex
under Gaussian noise. Rows are PDV (top) and ECFP4 (bottom); columns are the QM9 HOMO--LUMO gap, logD, hERG $K_i$
```
NEW:
```latex
under Gaussian noise. Rows are PDV (top) and ECFP4 (bottom); columns are the QM9 HOMO--LUMO gap, LogD, hERG $K_i$
```

### E67 — Contraction, dataset name and broken closing sentence (per-sample uncertainty)
- file: thesis.tex
- kind: author-text
- why: Style guide item 2 (contraction); "logD" against "LogD" elsewhere (chemistry guide item 11); the last sentence had lost its structure. Checker merged the former E67, E68 and E69 into one whole-paragraph edit so the comment sits above the paragraph, not mid-line.
- todo_line: % TODO: manual review -- edited your text: "It's" -> "It is"; "logD" -> "LogD" twice; the last sentence had two main verbs ("... to identify clipped labels as the labels ... are most likely to have been clipped"), so a colon now splits it.
OLD:
```latex
Since Gaussian, Laplace, Student-$t$ and grouped-shifted conditions add noise from the same scale to every label, there is not much for uncertainty to find. It's the same situation with outlier noise, under which a tenth of the data is chosen at random for extra noise, and with grouped-wider in which each scaffold family falls in a single fold, so a model makes predictions based on unseen scaffolds and has no basis for estimating uncertainty. The only case where per-sample uncertainty tracking is possible is censoring, as the noise is dependent on the value of the label itself. The correlations are then large and mostly negative, meaning the model is least uncertain on the labels that were clipped most. On QM9 with half the labels clipped, on PDV, they run from $-0.84$ to $-0.80$ across folds for NGBoost and from $-0.47$ to $-0.41$ for the GP, and on logD with half the labels clipped every fold of every model except the two GPs is at $-0.29$ or lower. We divided each molecule's error by its uncertainty and found that the division ranked the clipped labels better than the error alone on QM9 and on most of logD. Models are most confident on the clipped labels, probably because the clipping piles many labels into one value, which the model learns to tightly predict. Ironically, per-molecule uncertainty can be used in reverse to identify clipped labels as the labels a model is more certain about, especially when they fall in a certain range, are most likely to have been clipped.
```
NEW:
```latex
% TODO: manual review -- edited your text: "It's" -> "It is"; "logD" -> "LogD" twice; the last sentence had two main verbs ("... to identify clipped labels as the labels ... are most likely to have been clipped"), so a colon now splits it.
Since Gaussian, Laplace, Student-$t$ and grouped-shifted conditions add noise from the same scale to every label, there is not much for uncertainty to find. It is the same situation with outlier noise, under which a tenth of the data is chosen at random for extra noise, and with grouped-wider in which each scaffold family falls in a single fold, so a model makes predictions based on unseen scaffolds and has no basis for estimating uncertainty. The only case where per-sample uncertainty tracking is possible is censoring, as the noise is dependent on the value of the label itself. The correlations are then large and mostly negative, meaning the model is least uncertain on the labels that were clipped most. On QM9 with half the labels clipped, on PDV, they run from $-0.84$ to $-0.80$ across folds for NGBoost and from $-0.47$ to $-0.41$ for the GP, and on LogD with half the labels clipped every fold of every model except the two GPs is at $-0.29$ or lower. We divided each molecule's error by its uncertainty and found that the division ranked the clipped labels better than the error alone on QM9 and on most of LogD. Models are most confident on the clipped labels, probably because the clipping piles many labels into one value, which the model learns to tightly predict. Ironically, per-molecule uncertainty can be used in reverse to identify clipped labels: the labels a model is more certain about, especially when they fall in a certain range, are the most likely to have been clipped.
```

### E70 — Comma splice
- file: thesis.tex
- kind: author-text-trivial
- why: Grammar.
- todo_line: none (trivial)
OLD:
```latex
Low quantities of noise do not make much of an impact, on the HOMO--LUMO gap every base model keeps at least 93\% of its clean $R^2$ up to a noise level of 0.3
```
NEW:
```latex
Low quantities of noise do not make much of an impact: on the HOMO--LUMO gap every base model keeps at least 93\% of its clean $R^2$ up to a noise level of 0.3
```

### E71 — Second person
- file: thesis.tex
- kind: author-text
- why: Style guide item 2.
- todo_line: % TODO: manual review -- edited your text: "data you trust" -> "data whose labels are trusted".
OLD:
```latex
When training on noisy experimental data in the context of QSAR, it is important to keep in mind that a model's robustness to noise is primarily driven by its own mechanism of training, and to an extent the molecular representation the input data comes in, oftentimes with respect to the model. Of the models tested, RF was the most consistent choice across the four datasets, ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition. Where a predictive distribution is wanted, its probabilistic counterpart QRF maintains RF's predictive performance and nearly all of its noise robustness. What QRF cannot do is say where that uncertainty came from. Beyond the forests, the better choice depends on how much noise there is. LightGBM and XGBoost predict well on clean labels and lose the most as noise rises, so they suit data you trust, while NGBoost and the VBLL networks start lower and hold what they have.
```
NEW:
```latex
% TODO: manual review -- edited your text: "data you trust" -> "data whose labels are trusted".
When training on noisy experimental data in the context of QSAR, it is important to keep in mind that a model's robustness to noise is primarily driven by its own mechanism of training, and to an extent the molecular representation the input data comes in, oftentimes with respect to the model. Of the models tested, RF was the most consistent choice across the four datasets, ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition. Where a predictive distribution is wanted, its probabilistic counterpart QRF maintains RF's predictive performance and nearly all of its noise robustness. What QRF cannot do is say where that uncertainty came from. Beyond the forests, the better choice depends on how much noise there is. LightGBM and XGBoost predict well on clean labels and lose the most as noise rises, so they suit data whose labels are trusted, while NGBoost and the VBLL networks start lower and hold what they have.
```

### E72 — Second person in the noise-estimation paragraph
- file: thesis.tex
- kind: author-text
- why: Style guide item 2: "you" four times in a Conclusions paragraph.
- todo_line: % TODO: manual review -- edited your text: removed four uses of "you"; the closing question became a statement, since its grammar did not match a question; "generate a predictive limit" -> "gives a limit on predictive performance".
OLD:
```latex
In this research, we define a noise level as a fraction of the spread of the clean training labels. It is therefore relative to the dataset rather than an absolute measurement error, and cannot be mapped onto a given dataset exactly. There are other mechanisms used to estimate the noise in a dataset directly. When an assay's controls are available, the Z$'$-factor can be used to estimate noise \citep{Zhang1999}. You can also use screening data to test for row, column and plate effects, which produce systematic rather than independent error \citep{Caraus2015}. When you lack controls, replicates, and metadata, a label can be checked against the data itself. Confident learning flags labels by comparing each label to what a model predicts for that molecule when it was left out of training \citep{Northcutt2021}. You can use the point at which the model error matches the data's error to generate a predictive limit \citep{Kolmar2021, Crusius2024}. Beyond formal methods, you should consider how a dataset was assembled. How many independent replicates were done, how many laboratories contributed, whether assay protocols were consistently applied, and how varied are the compounds in chemical space?
```
NEW:
```latex
% TODO: manual review -- edited your text: removed four uses of "you"; the closing question became a statement, since its grammar did not match a question; "generate a predictive limit" -> "gives a limit on predictive performance".
In this research, we define a noise level as a fraction of the spread of the clean training labels. It is therefore relative to the dataset rather than an absolute measurement error, and cannot be mapped onto a given dataset exactly. There are other mechanisms used to estimate the noise in a dataset directly. When an assay's controls are available, the Z$'$-factor can be used to estimate noise \citep{Zhang1999}. Screening data can also be tested for row, column and plate effects, which produce systematic rather than independent error \citep{Caraus2015}. Without controls, replicates or metadata, a label can be checked against the data itself. Confident learning flags labels by comparing each label to what a model predicts for that molecule when it was left out of training \citep{Northcutt2021}. The point at which the model error matches the data's error gives a limit on predictive performance \citep{Kolmar2021, Crusius2024}. Beyond formal methods, it helps to know how a dataset was assembled. That means how many independent replicates were done and how many laboratories contributed. It also means whether assay protocols were applied consistently, and how varied the compounds are in chemical space.
```

### E73 — Second person and contractions (limitations)
- file: thesis.tex
- kind: author-text
- why: Style guide item 2 (old line 1299 listed).
- todo_line: % TODO: manual review -- edited your text: removed "you" and two contractions; same content.
OLD:
```latex
Third, it is hard to draw conclusions that hold across so many conditions. No model or representation wins everywhere, although RF comes closest. When working with cheminformatics data, you don't know in advance how a model and representation will perform on the given prediction task. The underlying data is probably noisy, but you don't know how noisy it is or what shape that noise takes.
```
NEW:
```latex
% TODO: manual review -- edited your text: removed "you" and two contractions; same content.
Third, it is hard to draw conclusions that hold across so many conditions. No model or representation wins everywhere, although RF comes closest. When working with cheminformatics data, we do not know in advance how a model and representation will perform on a given prediction task. The underlying data is probably noisy, but how noisy it is, and what shape that noise takes, is unknown.
```

## For the author (not changed)
- K_i typography. This plan only changes the roman `K$_i$` and `pK$_i$` in this chapter to the thesis-wide `$K_i$` and `p$K_i$`. IUPAC sets the subscript roman (`$K_\mathrm{i}$`). That would need one macro in main.tex and about 200 replacements across all chapters (chemistry guide item 10). It is not planned here.
- "Scaffold family" (used throughout this chapter and in Chapter 2's noise table) against "scaffold group" (chemistry guide item 14). This is a thesis-wide choice, so nothing is changed here. Example: "the same grouping defines the scaffold families targeted by the grouped noise conditions".
- The Tanimoto-kernel GP is in Models ("The second uses the Tanimoto kernel \citep{Ralaivola2005, moss2020} and was run on ECFP4 alone"). It is not among the 13 base models or the six variant models. Figure fig:transfer says "Ranks are among all 19 models". The Methods notes count 20 configurations with seven variants. Say where the Tanimoto GP's results are, or drop it.
- The QM9 pipeline paragraphs disagree on where representations are computed. One says Rust "computes the remaining representations". The next says "The six representations described above are computed in Python."
- RF rank range. The results say "it ranks between first and seventh for \aucnorm{} on every dataset and representation" (Gaussian only). The Conclusions say "ranking between first and eighth for \aucnorm{} on every dataset, representation and noise condition". The scopes differ, so both may be right; check both against the results.
- "clean $R^2$ and \aucnorm{} rank in opposite directions at a Spearman correlation of $-0.10$ ($p = 0.73$)". The Limitations say "across the 13 base models it is unrelated to clean $R^2$". A correlation of -0.10 at p = 0.73 is no relation, so "rank in opposite directions" overstates it.
- "SVM gains \aucnorm{} and NGBoost loses it under all three distributions across all datasets and representations." Table tab:assay_shape has NGBoost on LogD at 0.000 under Laplace. The assay text says NGBoost loses under all four only "on Caco-2 and hERG $K_i$".
- "Censoring created the largest drop ... \aucnorm{} fell by 0.12 to 0.18". The tab:censoring caption says the two \aucnorm{} columns "are on different scales", because a censoring level is a fraction clipped. The censoring row of tab:condition_choice makes the same comparison.
- "On Caco-2 and hERG $K_i$ the replicate ... is the largest share of predictive performance before and after noise is injected." The Conclusions say "Model family was the primary factor in every decomposition except one, Caco-2 under grouped-shifted noise". These disagree unless "decomposition" means robustness only.
- The Conclusions say the assay labels are "pooled from different laboratories". This was not checked for the OpenADMET LogD and Caco-2 sets, which may come from one source.
- "Repeat ... measurements ... \citep{Kalliokoski2013, Kramer2012}". The chemistry guide says Kalliokoski 2013 is the pIC50 source (0.68), so check whether it belongs on a p$K_i$ number.
- "where a third or more of labels can be censored \citep{Svensson2025}". Chapter 2 gives the same source as "eight sit between a quarter and two thirds censored".
- Representations: ECFP4 is given radius and bits but not the chirality setting. Chapter 4 says "In this research, ECFP4 is built with stereochemistry off". This was not checked against the Chapter 3 code, so no edit is planned (chemistry guide, ECFP4 section).
- Optional, not planned: the QM9 description could add that its molecules have at most nine heavy atoms (C, N, O, F), which is why its chemistry differs from the drug-like sets (chemistry guide, QM9).
- "(Datasets subsection)" and "the Models subsection" cite subsections by name. Neither subsection has a \label, so `Section~\ref` cannot be used yet. No label is added here because the package conversion may move headings.
- "On QM9 the Bayesian transformations made no significant change". Here "significant" refers to the Wilcoxon test that was run on QM9, so it stays under the decision to keep Chapter 3's tests.
- The Introduction lists three questions ("how the choice of representation ... what is the impact of the type of label noise, and whether uncertainty estimates track label noise"). Chapter 2's Conclusions say "the four questions set out in Chapter 1".
- "it ranks among the five highest for predictive performance at a noise level on every assay dataset". It does not say which noise level.
- "Because only its output layer keeps a learned observation-noise parameter, VBLL networks report only epistemic uncertainty." Table tab:uncertainty_rise gives VBLL an aleatoric component "one per fit". The existing TODO(cite) above the paragraph already flags this.
- The new definitions paragraph (LogD, Caco-2 efflux ratio, hERG) has no citations; a TODO(cite) line is planned above it. Its Caco-2 wording assumes the Chapter 3 label is the log10 efflux ratio, as in Chapter 5. Confirm that before applying both Caco-2 edits.
- Author sentences over 30 words were left alone under the rules. Example: "On QM9 with half the labels clipped, on PDV, they run from $-0.84$ to $-0.80$ across folds for NGBoost ...".
- The chapter ends by pointing forward ("Chapter~\ref{chap:flaws} asks how well these representations compare molecules on a single dataset."), so no edit is needed there.
- No AI paragraph in this chapter lists many scores, so there are no number-to-table edits (task item 6).
- Censoring clips the top of the label range ("we recorded every label above the $k$th largest clean training label as that limit"). For p$K_i$, real censoring is mostly at the weak end (compounds inactive at the highest concentration tested). The text never says which end it imitates (chemistry guide item 13). A new paragraph saying so was planned and removed by the checker, because it judges the design of the experiment.
- "Repeat measurements of the same compound disagree by about 0.54 log units for pK$_i$". Chapter 4 quotes the same source as "differ by $0.44$ log units on average, with a standard deviation of $0.54$", and Chapter 2's table gives 0.44. The next sentence ("about one unit of that estimated laboratory error") uses 0.54, so changing the number changes that sentence too. Not checked against Kramer 2012.

## Checker notes
Every OLD block was re-checked with python3 -I str.count against thesis.tex on 2026-10-10: all 69 remaining edits are exact and unique, and none overlap. No new number appears that is absent from thesis.tex. No \label changes. Sentences over 30 words in NEW text are all the author's own sentences that received a trivial fix, or caption sentences with a short insertion.

Changed:
- E1: todo_line field now shows the exact comment line; the NEW block was already right.
- E4: "LogD is ... at pH 7.4" became "Log $D$ is ... usually measured at pH 7.4". The quantity is written log $D$ (chemistry guide item 11), and the pH of the OpenADMET assay was not checked.
- E26, E33, E34, E41, E53 (captions): the plan rewrote each opening to "Robustness (\aucnorm{}; higher is more robust)". Opening with a symbol is style, not an error. Now the author's opening stays and only "(higher is more robust)" is inserted after \aucnorm{}. The todo lines now say that.
- E38: reduced to the trivial pK$_i$ -> p$K_i$ fix. The 0.54 -> 0.44/0.54 rewrite rests on a source nobody checked, and the following sentence depends on 0.54. Moved to For the author. It was also a comment inserted mid-line.
- E67, E68, E69: merged into one E67 whose OLD is the whole paragraph. The old E67 put its TODO comment in the middle of a line. The rewrite of the last sentence now keeps the author's words ("Ironically", "a certain range") and only adds a colon to repair it. "It's the same situation" -> "It is the same situation" rather than a rephrase.
- E73: kept "data is" (the author's usage elsewhere, e.g. "The data is then passed") instead of "data are".

Deleted:
- E6 (new paragraph on which end censoring clips): an interpretation of the author's experimental design with no citation. Moved to For the author.
- E65 ("None of the models tested treat" -> "treats"): both forms are correct English, so this was taste.

Kept as planned: E2, E3, E5, E7-E37, E39-E64, E66, E70-E72 (E3 and E4 still carry their "confirm" wording about the Caco-2 endpoint).

