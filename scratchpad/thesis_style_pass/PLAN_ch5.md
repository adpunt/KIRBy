# Plan: Chapter 5 (chap:method, Building Adaptive Hybrid Representations)

Scope: thesis.tex from `\chapter{Building Adaptive Hybrid Representations}` to `\chapter{Similarity in an Adaptive Hybrid Representation}`. Read in full on 2026-10-10. Every OLD block was checked with `python3 -I` (`str.count == 1`) against thesis.tex on 2026-10-10. Guide references: CLG = CHEMISTRY_LANGUAGE_GUIDE.md section 5 item number; SG = THESIS_STYLE_GUIDE.md section or gap-list number.

Order: chemistry first, then transitions, register and cross-references, captions, long AI sentences, numbers in prose.

## Edits

### E1 — hERG Ki row: binding, not inhibition
- file: thesis.tex
- kind: ai-text
- why: A K_i from a binding assay measures binding, not inhibition of channel current (CLG item 25).
- todo_line: keep existing
OLD:
```latex
hERG $K_i$ \citep{Zdrazil2023} & hERG inhibition, p$K_i$, regression &
```
NEW:
```latex
hERG $K_i$ \citep{Zdrazil2023} & hERG binding, p$K_i$, regression &
```

### E2 — LogD label unit in the datasets table
- file: thesis.tex
- kind: ai-text
- why: log D is dimensionless and defined at a stated pH; "log units" is not a unit (CLG item 26). pH 7.4 is already stated for this dataset in the LogD blueprint paragraph. The other unitless labels in this table already say "(no unit)".
- todo_line: keep existing
OLD:
```latex
LogD \citep{openadmet} & LogD (log units) & regression, RMSE $\downarrow$
```
NEW:
```latex
LogD \citep{openadmet} & logD at pH 7.4 (no unit) & regression, RMSE $\downarrow$
```

### E3 — LogD blueprint paragraph: SLogP, PEOE_VSA, bare SMILES, log D definition
- file: thesis.tex
- kind: ai-text
- why: SLogP is a calculated log P, not a measured one (CLG section 3, Descriptors). PEOE_VSA descriptors are surface areas summed within bins of Gasteiger partial charge, not charge-weighted areas. A bare SMILES in prose needs typewriter font and a name (CLG item 28); an ECFP environment written CN(C)C is a nitrogen with three carbon neighbors, so "N,N-dimethylamino" (the guide's suggestion) would be more specific than the bit is. "LogD is the distribution coefficient" omits the logarithm, and the guide asks to add that log D counts all charge states and log P only the neutral form (CLG section 3, LogD). The last sentence was 31 words.
- todo_line: keep existing
OLD:
```latex
Among the descriptors are Crippen's logP (SLogP), the number of basic groups (nBase), several charge-weighted surface areas (PEOE\_VSA) and counts of secondary amine and aromatic nitrogen environments. Among the bits are tertiary amine environments such as CN(C)C and aryl chloride. LogD is the distribution coefficient at pH 7.4, where a basic amine is largely protonated, which lowers it below logP, so a model of LogD should need lipophilicity and basicity.
```
NEW:
```latex
Among the descriptors are Crippen's calculated logP (SLogP), the number of basic groups (nBase) and several surface areas binned by partial charge (PEOE\_VSA). There are also counts of secondary amine and aromatic nitrogen environments. Among the bits are tertiary amine environments, a nitrogen bonded to three carbons (written \texttt{CN(C)C}), and aryl chloride environments. The LogD label is the logarithm of the distribution coefficient at pH 7.4. Unlike logP, which counts only the neutral form of a molecule, logD counts every charge state. At pH 7.4 a basic amine is largely protonated, which lowers logD below logP. A model of LogD should therefore need both lipophilicity and basicity.
```

### E4 — Bemis–Murcko scaffold defined; QM9 is a small-molecule dataset
- file: thesis.tex
- kind: ai-text
- why: Bemis–Murcko is never defined in the thesis (CLG item 15), and "the small-molecule and QM9 datasets" implies QM9 is not small molecules (CLG item 30). Definition wording from CLG section 3, Scaffolds; `Bemis1996` is in refs.bib.
- todo_line: keep existing
OLD:
```latex
\paragraph{Groups by domain.} The small-molecule and QM9 datasets are grouped by Bemis--Murcko scaffold (Section~\ref{sec:robustness:methods}), here keeping stereochemistry.
```
NEW:
```latex
\paragraph{Groups by domain.} The small-molecule datasets, QM9 included, are grouped by Bemis--Murcko scaffold \citep{Bemis1996} (Section~\ref{sec:robustness:methods}). A scaffold keeps the ring systems of a molecule and the chains that link them, and removes all other side chains. Here the scaffold keeps stereochemistry.
```

### E5 — Dablander targets: two from ChEMBL, one from COVID Moonshot (methods)
- file: thesis.tex
- kind: ai-text
- why: In Dablander's setting the SARS-CoV-2 main protease data come from the COVID Moonshot project, not ChEMBL (CLG item 5, checked by the guide against the Dablander text, p. 51).
- todo_line: keep existing
OLD:
```latex
The first repeats the activity-cliff QSAR setting of \citet{Dablander2023} on three ChEMBL targets: dopamine D2 (6,333 compounds), factor Xa (3,605) and the SARS-CoV-2 main protease (1,924), with five scaffold folds.
```
NEW:
```latex
The first repeats the activity-cliff QSAR setting of \citet{Dablander2023} on three targets, with five scaffold folds. Dopamine D2 (6,333 compounds) and factor Xa (3,605) come from ChEMBL, and the SARS-CoV-2 main protease (1,924) comes from the COVID Moonshot project.
```

### E6 — Dablander targets (results), and a 36-word sentence split
- file: thesis.tex
- kind: ai-text
- why: Same factual error as E5 (CLG item 5); the sentence was 36 words.
- todo_line: keep existing
OLD:
```latex
On the three ChEMBL targets of the activity-cliff setting of \citet{Dablander2023}, the hybrid has a lower error than the single representation the same search ranked first for five of the six target and model pairs.
```
NEW:
```latex
On the three targets of the activity-cliff setting of \citet{Dablander2023}, the hybrid has a lower error for five of the six target and model pairs. Two targets come from ChEMBL and one from the COVID Moonshot. The hybrid is compared with the single representation that the same search ranked first.
```

### E7 — Ginkgo assay names spelled out
- file: thesis.tex
- kind: ai-text
- why: HIC, SMAC, AC-SINS and CHO are never expanded anywhere in the thesis (SG gap 13 lists SMAC, HIC, AC-SINS and CHO as missing). The datasets paragraph is where the assays are introduced. The 39-word sentence is also split.
- todo_line: keep existing
OLD:
```latex
The five Ginkgo developability assays are measured on the same panel of antibodies, and we use the five cross-validation folds released with the competition \citep{vanNiekerk2026} as the groups, so their outer folds are the competition's folds.
```
NEW:
```latex
The five Ginkgo developability assays are measured on the same panel of antibodies. They are hydrophobic interaction chromatography (HIC) retention, standup monolayer adsorption chromatography (SMAC) retention, affinity-capture self-interaction nanoparticle spectroscopy (AC-SINS), expression titer, and polyreactivity to Chinese hamster ovary (CHO) cell proteins. We use the five cross-validation folds released with the competition \citep{vanNiekerk2026} as the groups, so their outer folds are the competition's folds.
```

### E8 — CDR expanded at first use in prose
- file: thesis.tex
- kind: author-text
- why: CDR is first expanded in Chapter 8 (CLG section 3, Antibodies and CDRs); this is its first use in Chapter 5's prose.
- todo_line: % TODO: manual review -- edited your text: "CDR" spelled out at its first use in the prose (it was only expanded in Chapter 8), with a short gloss of what the CDRs are.
OLD:
```latex
\paragraph{Antibodies.} Antibody representations (Table~\ref{tab:reps_antibodies}) start from the heavy- and light-chain amino-acid sequences. Besides one-hot and BLOSUM62 encodings \citep{Henikoff1992}, most are frozen language models, either trained on all proteins or on antibody repertoires only. Most models embed each chain separately and join the two vectors, so their width doubles on datasets with both chains. Several representations read one model in different ways, for instance averaging over earlier layers or over the CDR residues only, with CDRs found by IMGT numbering \citep{Lefranc2003} using ANARCI \citep{Dunbar2016ANARCI}.
```
NEW:
```latex
% TODO: manual review -- edited your text: "CDR" spelled out at its first use in the prose (it was only expanded in Chapter 8), with a short gloss of what the CDRs are.
\paragraph{Antibodies.} Antibody representations (Table~\ref{tab:reps_antibodies}) start from the heavy- and light-chain amino-acid sequences. Besides one-hot and BLOSUM62 encodings \citep{Henikoff1992}, most are frozen language models, either trained on all proteins or on antibody repertoires only. Most models embed each chain separately and join the two vectors, so their width doubles on datasets with both chains. Several representations read one model in different ways, for instance averaging over earlier layers or over the complementarity-determining regions (CDRs) only, the loops that vary most between antibodies. CDRs are found by IMGT numbering \citep{Lefranc2003} using ANARCI \citep{Dunbar2016ANARCI}.
```

### E9 — FcRn expanded
- file: thesis.tex
- kind: author-text
- why: Undefined abbreviation; FcRn appears nowhere else in the thesis.
- todo_line: % TODO: manual review -- edited your text: FcRn spelled out (neonatal Fc receptor); it is not defined anywhere else.
OLD:
```latex
The first method of feature allocation we tried was a greedy forward selection, in which the hybrid is built one block of features at a time  \citep{Kohavi1997, Guyon2003}. In each round, the next block is added from representations in order of which improves the internal validation score. This in theory lets it choose features for how well they work together. In practice, this is very expensive. With blocks of 10 features and a budget of 500 and 10 representations, it needs up to 500 model fits per configuration and fold against 10 for a score-based method. It was also less accurate: in preliminary runs on Jain antibody FcRn retention it placed
```
NEW:
```latex
% TODO: manual review -- edited your text: FcRn spelled out (neonatal Fc receptor); it is not defined anywhere else.
The first method of feature allocation we tried was a greedy forward selection, in which the hybrid is built one block of features at a time \citep{Kohavi1997, Guyon2003}. In each round, the next block is added from representations in order of which improves the internal validation score. This in theory lets it choose features for how well they work together. In practice, this is very expensive. With blocks of 10 features and a budget of 500 and 10 representations, it needs up to 500 model fits per configuration and fold against 10 for a score-based method. It was also less accurate: in preliminary runs on Jain antibody retention by the neonatal Fc receptor (FcRn) it placed
```

### E10 — ECFP4 bit set by several environments; second person
- file: thesis.tex
- kind: author-text
- why: A folded ECFP4 bit can be set by several atom environments (CLG item 23). "tell you" is second person (SG section 2).
- todo_line: % TODO: manual review -- edited your text: an ECFP4 bit can be set by several atom environments, so "environment" became plural; "tell you if" became "say whether" (no second person).
OLD:
```latex
Depending on the origin of the feature, some are chemically interpretable. By definition, the PDV contains values that relate back to chemical properties like molecular weight or logP. An ECFP4 bit can be shown as the atom environment that sets it, and a MACCS key stands for a known substructure. Beyond small molecules, the readable unit of an antibody is a sequence position and its residue. This feature-level explainability fails when features are derived from learned embeddings. That will only tell you if the learned embedding itself is useful as a representation, but nothing about the selected feature.
```
NEW:
```latex
% TODO: manual review -- edited your text: an ECFP4 bit can be set by several atom environments, so "environment" became plural; "tell you if" became "say whether" (no second person).
Depending on the origin of the feature, some are chemically interpretable. By definition, the PDV contains values that relate back to chemical properties like molecular weight or logP. An ECFP4 bit can be shown as the atom environments that set it, and a MACCS key stands for a known substructure. Beyond small molecules, the readable unit of an antibody is a sequence position and its residue. This feature-level explainability fails when features are derived from learned embeddings. That will only say whether the learned embedding itself is useful as a representation, but nothing about the selected feature.
```

### E11 — "compounds" to "molecules"; long sentence split (splitting rules)
- file: thesis.tex
- kind: ai-text
- why: "Molecule" is the thesis default noun for dataset members (CLG section 1); the sentence was 36 words.
- todo_line: keep existing
OLD:
```latex
No split is random, because random splitting inflates performance estimates for molecular data by placing structurally similar compounds on both sides of the train/test boundary \citep{Yang2019, Wallach2018}.
```
NEW:
```latex
No split is random. Random splitting inflates performance estimates for molecular data, because it places structurally similar molecules on both sides of the train/test boundary \citep{Yang2019, Wallach2018}.
```

### E12 — EGFR scan rows are variants; "molecules" for antibodies
- file: thesis.tex
- kind: ai-text
- why: Rows of a mutational scan are variants, and a sentence covering antibodies and molecules should say "samples" (CLG section 1 and item 19).
- todo_line: keep existing
OLD:
```latex
Amino-acid composition does the same to 432 of the 659 antibodies in the EGFR scan, which differ by single mutations. None of the hybrids built on these datasets leaves two molecules with the same row.
```
NEW:
```latex
Amino-acid composition does the same to 432 of the 659 variants in the EGFR scan, which differ by single mutations. None of the hybrids built on these datasets leaves two samples with the same row.
```

### E13 — Nearest-neighbor objective: "label", not "activity"
- file: thesis.tex
- kind: ai-text
- why: The objective runs on datasets whose labels are not activities (QM9, LogD, antibodies, yields) (CLG item 17).
- todo_line: keep existing
OLD:
```latex
Under the nearest-neighbor objective, each molecule's activity is predicted from its five nearest neighbors in the feature space (Chapter~\ref{chap:similarity}).
```
NEW:
```latex
Under the nearest-neighbor objective, each sample's label is predicted from its five nearest neighbors in the feature space (Chapter~\ref{chap:similarity}).
```

### E14 — Metrics paragraph: QM9 is a small-molecule dataset
- file: thesis.tex
- kind: ai-text
- why: Same issue as CLG item 30.
- todo_line: keep existing
OLD:
```latex
Regression on small molecules, QM9 and reaction yields uses RMSE, in the units of the label.
```
NEW:
```latex
Regression on small molecules, QM9 included, and on reaction yields uses RMSE, in the units of the label.
```

### E15 — Chapter opening: broken sentence, "significantly", second person
- file: thesis.tex
- kind: author-text
- why: "With the understanding that ... about that data." is a fragment; "vary significantly" names a test that was not run (SG section 10, gap 31, which lists this line); "you" is second person (SG section 2). The opening already looks back to Chapter 4, so the transition stays as it is.
- todo_line: % TODO: manual review -- edited your text: the fragment "With the understanding that ..." became a full sentence by dropping those four words; "significantly" became "widely" (no test was run); "you" removed; comma added after "distance".
OLD:
```latex
So far, Chapter~\ref{chap:flaws} explained how the standard methods of estimating molecular similarity, or conversely distance are flawed and potentially break the SPP. Specifically, the representations used in these similarity calculations contain irrelevant features that either do not vary or more generally do not pertain to the prediction task at hand. This chapter explores a solution. With the understanding that each representation contains a set of features which contain a certain type of information about that data. We aim to select the most informative features across a larger set of representations, while filtering out the irrelevant ones. We create an adaptive hybrid representation, a set of features bounded by a budget that is selected from several base representations by a search using one dataset and downstream model. The goal is to select a set of features that is customized for a given model and prediction task. As discussed in Chapter~\ref{chap:robustness}, performance can vary significantly across datasets and models. Rather than hope the features you give a model will be useful, this process lets you prescreen them.
```
NEW:
```latex
% TODO: manual review -- edited your text: the fragment "With the understanding that ..." became a full sentence by dropping those four words; "significantly" became "widely" (no test was run); "you" removed; comma added after "distance".
So far, Chapter~\ref{chap:flaws} explained how the standard methods of estimating molecular similarity, or conversely distance, are flawed and potentially break the SPP. Specifically, the representations used in these similarity calculations contain irrelevant features that either do not vary or more generally do not pertain to the prediction task at hand. This chapter explores a solution. Each representation contains a set of features which contain a certain type of information about that data. We aim to select the most informative features across a larger set of representations, while filtering out the irrelevant ones. We create an adaptive hybrid representation, a set of features bounded by a budget that is selected from several base representations by a search using one dataset and downstream model. The goal is to select a set of features that is customized for a given model and prediction task. As discussed in Chapter~\ref{chap:robustness}, performance can vary widely across datasets and models. Rather than hoping that the features given to a model will be useful, this process prescreens them.
```

### E17 — Lead-in for "Results: why the hybrid helps"
- file: thesis.tex
- kind: new-paragraph
- why: SG section 4: put one or two sentences between a section heading and its first subsection (B, V). This is the only section in the chapter that goes straight into a subsection.
- todo_line: % TODO: manual review
OLD:
```latex
\section{Results: why the hybrid helps}\label{sec:method:results_why}
```
NEW:
```latex
\section{Results: why the hybrid helps}\label{sec:method:results_why}

% TODO: manual review
This section asks where the gain reported in Section~\ref{sec:method:results_single} comes from. It compares the hybrid with other ways of choosing the same number of features. It then tests how the downstream model and the internal evaluator shape the hybrid, and what weighting by importance, tuning and new embeddings add.
```

### E18 — "doesn't" and second person (dimensionality)
- file: thesis.tex
- kind: author-text
- why: Contraction and second person (SG section 2, gap 25).
- todo_line: % TODO: manual review -- edited your text: "doesn't" expanded and "If you add ... you end up with" rephrased without "you".
OLD:
```latex
Most research in this field relies on a single representation, and few combine features from different representations beyond concatenation. One of the reasons just pooling features from different representations together doesn't work is dimensionality. If you add a standard ECFP4 ($d=2,048$) with Mordred descriptors ($d=1,613$) and an MHG-GNN embedding ($d=1,024$), you end up with 4,685 features,
```
NEW:
```latex
% TODO: manual review -- edited your text: "doesn't" expanded and "If you add ... you end up with" rephrased without "you".
Most research in this field relies on a single representation, and few combine features from different representations beyond concatenation. One of the reasons just pooling features from different representations together does not work is dimensionality. Joining a standard ECFP4 ($d=2,048$) with Mordred descriptors ($d=1,613$) and an MHG-GNN embedding ($d=1,024$) gives 4,685 features,
```

### E19 — Two typos in the concatenation paragraph
- file: thesis.tex
- kind: author-text-trivial
- why: "limitation combinations" (SG gap 26 lists it) and an -ly adverb hyphenated.
- todo_line: none (trivial)
OLD:
```latex
Most of this work involves limitation combinations, adding together standard fingerprints and linear representations \citep{Xie2020, Laufktter2019}. Although the features derived from embeddings are not as conceptually-straightforward, they contain another axis of information.
```
NEW:
```latex
Most of this work involves limited combinations, adding together standard fingerprints and linear representations \citep{Xie2020, Laufktter2019}. Although the features derived from embeddings are not as conceptually straightforward, they contain another axis of information.
```

### E20 — Ensembling named as late fusion; unfinished sentence completed
- file: thesis.tex
- kind: author-text
- why: Follows your TODO "I'm pretty sure I'm doing late fusion but calling it ensembling, make sure its appropriately named below": the previous paragraph defines late fusion as one model per source with pooled predictions, which is what this paragraph calls ensembling, and the methods' Ensembles paragraph already says "(late fusion)". The paragraph also stops mid-sentence at "for each new" (SG gap 26).
- todo_line: % TODO: manual review -- edited your text: per your TODO, ensembling is now named as a form of late fusion; the last sentence stopped at "for each new" and now ends "molecule" (check this is the word you meant); comma added and the long last sentence split.
OLD:
```latex
Beyond simply combining features, there are a range of other methods that handle training on multiple representations or excessive feature sets. Ensembling fits one model per representation and aggregates their predictions, essentially combining the information carried by each representation after training rather than before. Consensus QSAR works this way \citep{Valsecchi2020}. One drawback is that no model in an ensemble sees features from other representations so their interaction is never measured, and every representation and every model must be computed for each new
```
NEW:
```latex
% TODO: manual review -- edited your text: per your TODO, ensembling is now named as a form of late fusion; the last sentence stopped at "for each new" and now ends "molecule" (check this is the word you meant); comma added and the long last sentence split.
Beyond simply combining features, there is a range of other methods that handle training on multiple representations or excessive feature sets. Ensembling across representations is a form of late fusion. It fits one model per representation and aggregates their predictions, essentially combining the information carried by each representation after training rather than before. Consensus QSAR works this way \citep{Valsecchi2020}. One drawback is that no model in an ensemble sees features from other representations, so their interaction is never measured. Every representation and every model must also be computed for each new molecule.
```

### E21 — Dimensionality reduction paragraph: second person
- file: thesis.tex
- kind: author-text
- why: Second person (SG section 2); missing comma before a non-restrictive "which".
- todo_line: % TODO: manual review -- edited your text: "When you want to ... you can use" rephrased without "you"; comma added before "which matters".
OLD:
```latex
When you want to filter a feature set for the most important to the prediction task, you can use a dimensionality reduction technique or feature selection methodology. Principal component analysis (PCA) uses linear combinations of the original features, ordered by the variance they explain \citep{Jolliffe2016}. Their combined features are dependent on the original feature units unless each feature is z-scored first which matters
```
NEW:
```latex
% TODO: manual review -- edited your text: "When you want to ... you can use" rephrased without "you"; comma added before "which matters".
To filter a feature set for the features most important to the prediction task, a dimensionality reduction technique or a feature selection method can be used. Principal component analysis (PCA) uses linear combinations of the original features, ordered by the variance they explain \citep{Jolliffe2016}. Their combined features are dependent on the original feature units unless each feature is z-scored first, which matters
```

### E22 — mRMR definition corrected
- file: thesis.tex
- kind: author-text
- why: Factual: mRMR scores a feature by its mutual information with the label minus its redundancy with features already chosen; the sentence defined mutual information as "the label minus its redundancy". Also "PCA or PLS which builds" (verb agreement) and a 50-word sentence.
- todo_line: % TODO: manual review -- edited your text: the mRMR definition said mutual information was "the label minus its redundancy"; it now says mRMR scores mutual information with the label minus redundancy with chosen features; sentence split.
OLD:
```latex
Feature selection keeps a subset of the original features, unlike PCA or PLS which builds new ones. Of its three families \citep{Guyon2003}, filter methods score features against the labels before any model is fitted, as minimum-redundancy maximum-relevance selection (mRMR) ranks features by their mutual information, defined as the label minus its redundancy with already-chosen features \citep{Peng2005}.
```
NEW:
```latex
% TODO: manual review -- edited your text: the mRMR definition said mutual information was "the label minus its redundancy"; it now says mRMR scores mutual information with the label minus redundancy with chosen features; sentence split.
Feature selection keeps a subset of the original features, unlike PCA or PLS, which build new ones. Of its three families \citep{Guyon2003}, filter methods score features against the labels before any model is fitted. Minimum-redundancy maximum-relevance selection (mRMR), for example, ranks each feature by its mutual information with the label minus its redundancy with features already chosen \citep{Peng2005}.
```

### E23 — Questions paragraph: "you" to "we"
- file: thesis.tex
- kind: author-text
- why: Second person (SG section 2). Only the pronouns change.
- todo_line: % TODO: manual review -- edited your text: "you" changed to "we" in the questions; nothing else changed.
OLD:
```latex
The task of selecting the best features from a set of representations and combining them is not simple. How do you measure what features are ``best''? Are these features ``best'' relative to the representation they came from or to the whole pool of features? Should you select more features from representations that seem to have more signal to begin with, or take a set number of features from every representation? How many base representations should you select features from? How many features should you select? When scoring potential combinations of features, what type of evaluator should you use?
```
NEW:
```latex
% TODO: manual review -- edited your text: "you" changed to "we" in the questions; nothing else changed.
The task of selecting the best features from a set of representations and combining them is not simple. How do we measure what features are ``best''? Are these features ``best'' relative to the representation they came from or to the whole pool of features? Should we select more features from representations that seem to have more signal to begin with, or take a set number of features from every representation? How many base representations should we select features from? How many features should we select? When scoring potential combinations of features, what type of evaluator should we use?
```

### E24 — Search-as-answer paragraph: "I", "you", "downstream evaluators"
- file: thesis.tex
- kind: author-text
- why: "I run" and "as I discovered" against the thesis's "we" (SG section 2, gap 25, which lists this line); "your data", "tells you", "you should use" are second person; "models (downstream evaluators)" mixes the thesis's terms "downstream model" and "internal evaluator". The "dream world" sentence is left (see For the author).
- todo_line: % TODO: manual review -- edited your text: "I" became "we", "you/your" removed, "models (downstream evaluators)" became "downstream models" (evaluator is the internal model's name), commas added around the clause in "The question actually became ...".
OLD:
```latex
In a dream world, there would be a single answer to each one of these questions that applies to every dataset. Unfortunately we live in reality, and as I discovered, the answer to every single one of these questions is ``depends on the data.'' The task of selecting the best features then became a search space optimization challenge. The question actually became for a given prediction task, defined by the dataset, prediction target, and model, what is the best methodology for selecting features? The hybrid search does not output just a list of features, rather it outputs an ideal search configuration to find features for your data. It tells you how to score features (feature importance score), what rules you should use to select them (allocation method), how to score potential hybrid feature combinations (internal evaluator). In this research, I run this hybrid search multiple times across different maximum number of features (budgets), number of input representations, and models (downstream evaluators) on each dataset.
```
NEW:
```latex
% TODO: manual review -- edited your text: "I" became "we", "you/your" removed, "models (downstream evaluators)" became "downstream models" (evaluator is the internal model's name), commas added around the clause in "The question actually became ...".
In a dream world, there would be a single answer to each one of these questions that applies to every dataset. Unfortunately we live in reality, and as we found, the answer to every single one of these questions is ``depends on the data.'' The task of selecting the best features then became a search space optimization challenge. The question actually became, for a given prediction task, defined by the dataset, prediction target, and model, what is the best methodology for selecting features? The hybrid search does not output just a list of features, rather it outputs an ideal search configuration to find features for a given dataset. It sets how to score features (feature importance score), what rules to use to select them (allocation method), and how to score potential hybrid feature combinations (internal evaluator). In this research, we run this hybrid search multiple times across different maximum numbers of features (budgets), numbers of input representations, and downstream models on each dataset.
```

### E25 — Formal definition: contradictory "fixed number ... capped"; second person
- file: thesis.tex
- kind: author-text
- why: "a set of fixed number of features, capped at a fixed budget" says both fixed and capped; the methods say the hybrid can be narrower than its budget. "you want" is second person.
- todo_line: % TODO: manual review -- edited your text: "a set of fixed number of features, capped at a fixed budget" became "a set of features, capped at a fixed budget" (the hybrid can be narrower than its budget); "you want to combine" rephrased.
OLD:
```latex
Formally, an adaptive hybrid representation is a set of fixed number of features, capped at a fixed budget, selected from several base representations of the same molecules. The output of a hybrid search is itself a feature selection method that when run on the given dataset, produces the hybrid representation. This search is run on a single dataset and downstream model, and the resulting hybrid differs by model, dataset, and even split. Its intended use case is when there are multiple base representations that describe different aspects of a molecule, and you want to combine these different features for a single training task.
```
NEW:
```latex
% TODO: manual review -- edited your text: "a set of fixed number of features, capped at a fixed budget" became "a set of features, capped at a fixed budget" (the hybrid can be narrower than its budget); "you want to combine" rephrased.
Formally, an adaptive hybrid representation is a set of features, capped at a fixed budget, selected from several base representations of the same molecules. The output of a hybrid search is itself a feature selection method that, when run on the given dataset, produces the hybrid representation. This search is run on a single dataset and downstream model, and the resulting hybrid differs by model, dataset, and even split. Its intended use case is when there are multiple base representations that describe different aspects of a molecule, and the aim is to combine these different features for a single training task.
```

### E26 — Blueprints paragraph: second person
- file: thesis.tex
- kind: author-text
- why: Second person (SG section 2).
- todo_line: % TODO: manual review -- edited your text: "the search can tell you about your own prediction task" rephrased without "you".
OLD:
```latex
As explained above, the adaptive hybrid representation relies on a search. A completed search is saved as a blueprint, which records the settings needed to select the features that make the hybrid representation. Because each selected feature traces back to its original base representation and relative importance rank, the search can tell you about your own prediction task.
```
NEW:
```latex
% TODO: manual review -- edited your text: "the search can tell you about your own prediction task" rephrased without "you".
As explained above, the adaptive hybrid representation relies on a search. A completed search is saved as a blueprint, which records the settings needed to select the features that make the hybrid representation. Because each selected feature traces back to its original base representation and relative importance rank, the search can also say something about the prediction task itself.
```

### E27 — Contraction "it's"
- file: thesis.tex
- kind: author-text
- why: Contraction (SG section 2).
- todo_line: % TODO: manual review -- edited your text: "it's" expanded to "it is".
OLD:
```latex
While a set of thousands of features may seem overwhelming and irrelevant, identifying substructures or descriptors that recur among the selected features can point to parts of the molecule that may be involved in binding or activity. If the selected features make sense chemically for that endpoint, it's a good sign
```
NEW:
```latex
% TODO: manual review -- edited your text: "it's" expanded to "it is".
While a set of thousands of features may seem overwhelming and irrelevant, identifying substructures or descriptors that recur among the selected features can point to parts of the molecule that may be involved in binding or activity. If the selected features make sense chemically for that endpoint, it is a good sign
```

### E28 — Antibodies-and-beyond paragraph: grammar, missing parentheses, second person
- file: thesis.tex
- kind: author-text
- why: "a set of feature selection method" and "blueprint mappings features" are grammar slips; the chapter reference was missing its parentheses; "cases where you have" is second person.
- todo_line: % TODO: manual review -- edited your text: "a set of feature selection method" became "a feature selection method"; "where you have different measurements" rephrased; "blueprint mappings" became "blueprints mapping"; parentheses added around the chapter reference.
OLD:
```latex
The hybrid search is a generalized search that takes in multiple feature matrices from the same samples and a supervised prediction task, and outputs a set of feature selection method that produces a filtered set of features from those original vectors. The default search objective scores prediction only, meaning the methodology applies to tabular tasks or cases where you have different measurements of the same inputs. Although this was developed with small molecules and QSAR modeling in mind, it is by no means limited to that use case. While there are some functionalities in this research that are tailored to chemical tasks, such as blueprint mappings features back to their chemical interpretations and the use of nearest-neighbor objectives Chapter~\ref{chap:similarity} to aid
```
NEW:
```latex
% TODO: manual review -- edited your text: "a set of feature selection method" became "a feature selection method"; "where you have different measurements" rephrased; "blueprint mappings" became "blueprints mapping"; parentheses added around the chapter reference.
The hybrid search is a generalized search that takes in multiple feature matrices from the same samples and a supervised prediction task, and outputs a feature selection method that produces a filtered set of features from those original vectors. The default search objective scores prediction only, meaning the methodology applies to tabular tasks or to cases with several different measurements of the same inputs. Although this was developed with small molecules and QSAR modeling in mind, it is by no means limited to that use case. While there are some functionalities in this research that are tailored to chemical tasks, such as blueprints mapping features back to their chemical interpretations and the use of nearest-neighbor objectives (Chapter~\ref{chap:similarity}) to aid
```

### E29 — Blueprint reuse: second person
- file: thesis.tex
- kind: author-text
- why: Second person (SG section 2).
- todo_line: % TODO: manual review -- edited your text: "you can reuse the features" rephrased without "you".
OLD:
```latex
The blueprint does not record the specific features that are returned by the hybrid, as it is meant to be re-run using the same settings on new, labeled data. Alternatively, you can reuse the features of a labeled fit and apply them to data with no labels under the assumption that the same features matter between sets.
```
NEW:
```latex
% TODO: manual review -- edited your text: "you can reuse the features" rephrased without "you".
The blueprint does not record the specific features that are returned by the hybrid, as it is meant to be re-run using the same settings on new, labeled data. Alternatively, the features of a labeled fit can be reused on data with no labels, under the assumption that the same features matter between sets.
```

### E30 — Search space opening: contraction, "input hybrids", list grammar
- file: thesis.tex
- kind: author-text
- why: "there's" is a contraction; "input hybrids" is the wrong term (hybrids are the output, the inputs are representations); "select how many ... and how to score" lacks a verb for its last item.
- todo_line: % TODO: manual review -- edited your text: "there's" expanded; "input hybrids" became "input representations"; "and choose" added so the list of steps reads as a sentence.
OLD:
```latex
To run a hybrid search, we first measure feature importance, select how many features to take from each contributing base representation, and how to score a candidate feature set. And for each of those steps, there's no one right way to do it. Different prediction tasks prefer different methods of scoring feature importance or candidate features, different sets of input hybrids, and so forth.
```
NEW:
```latex
% TODO: manual review -- edited your text: "there's" expanded; "input hybrids" became "input representations"; "and choose" added so the list of steps reads as a sentence.
To run a hybrid search, we first measure feature importance, select how many features to take from each contributing base representation, and choose how to score a candidate feature set. And for each of those steps, there is no one right way to do it. Different prediction tasks prefer different methods of scoring feature importance or candidate features, different sets of input representations, and so forth.
```

### E31 — Stale description of the search-settings table
- file: thesis.tex
- kind: author-text
- why: Wrong cross-reference description: Table tab:search_tiers no longer groups values by runtime speed (its fast-tier column was moved into the text on 2026-10-08, per the comment above that table).
- todo_line: % TODO: manual review -- edited your text: "gives default search settings, organized by runtime speed" became "gives the values each setting can take"; that table no longer has speed tiers.
OLD:
```latex
Six settings in total are searched, namely the downstream model, the budget, the representation count, the importance method, the allocation method and the internal evaluator. Table~\ref{tab:search_tiers} gives default search settings, organized by runtime speed, and Table~\ref{tab:decided_grid} the smaller set searched on each dataset in this chapter (Section~\ref{sec:method:decided_grid}).  The importance
```
NEW:
```latex
% TODO: manual review -- edited your text: "gives default search settings, organized by runtime speed" became "gives the values each setting can take"; that table no longer has speed tiers.
Six settings in total are searched, namely the downstream model, the budget, the representation count, the importance method, the allocation method and the internal evaluator. Table~\ref{tab:search_tiers} gives the values each setting can take, and Table~\ref{tab:decided_grid} the smaller set searched on each dataset in this chapter (Section~\ref{sec:method:decided_grid}). The importance
```

### E32 — Representation pre-selection paragraph: contractions and a typo
- file: thesis.tex
- kind: author-text
- why: Two contractions; "within in the pool".
- todo_line: % TODO: manual review -- edited your text: "it's" expanded twice and "within in" corrected; nothing else changed.
OLD:
```latex
We select a set of up to 10 base representations per dataset, chosen prior to the search (Table~\ref{tab:dataset_reps}).
```
NEW:
```latex
% TODO: manual review -- edited your text: "it's" expanded twice and "within in" corrected; nothing else changed.
We select a set of up to 10 base representations per dataset, chosen prior to the search (Table~\ref{tab:dataset_reps}).
```
(Second part of E32, same paragraph.)
OLD:
```latex
Since the features contain different types of information at different scales, pooling the features too early makes comparisons within in the pool challenging. We found that it's better to determine which features are important to each representation first, and go from there.
```
NEW:
```latex
Since the features contain different types of information at different scales, pooling the features too early makes comparisons within the pool challenging. We found that it is better to determine which features are important to each representation first, and go from there.
```

### E33 — Importance scores are zero or more, not positive
- file: thesis.tex
- kind: author-text
- why: Inconsistent with the methods ($s^{(k)}_j \ge 0$) and the importance table ("0 if never split on"); the hybrid's cut at "above 0" depends on zero scores existing.
- todo_line: % TODO: manual review -- edited your text: "a single positive score" became "a single score of zero or more", matching $s \ge 0$ and the zero scores used later; double space removed.
OLD:
```latex
We run a feature importance scoring method on each base representation, which gives each feature a single positive score. These scores remain relative within a given representation, so they are not compared or normalized between representations.  Split-based
```
NEW:
```latex
% TODO: manual review -- edited your text: "a single positive score" became "a single score of zero or more", matching $s \ge 0$ and the zero scores used later; double space removed.
We run a feature importance scoring method on each base representation, which gives each feature a single score of zero or more. These scores remain relative within a given representation, so they are not compared or normalized between representations. Split-based
```

### E34 — Fixed-share methods compute budget shares, not importance shares
- file: thesis.tex
- kind: author-text
- why: Inconsistent term: the shares are numbers of features from the budget, computed from standalone scores, not from importance.
- todo_line: % TODO: manual review -- edited your text: "compute importance shares" became "compute budget shares" (the shares come from standalone scores, not importance).
OLD:
```latex
The four fixed-share methods compute importance shares from the standalone score on internal validation.
```
NEW:
```latex
% TODO: manual review -- edited your text: "compute importance shares" became "compute budget shares" (the shares come from standalone scores, not importance).
The four fixed-share methods compute budget shares from the standalone score on internal validation.
```

### E35 — Internal evaluator paragraph: "twelve model types used in this thesis"; broken sentence
- file: thesis.tex
- kind: author-text
- why: Factual: the code's matched-evaluator map (src/kirby/find_best_hybrid.py, `match_map`, re-read by the checker 2026-10-10) has twelve model types, including CatBoost and a CNN, which no chapter of this thesis uses. The sentence "For a downstream model outside of the pre-defined set, if no matched evaluator exists in the code, and the search uses ..." has a stray "and".
- todo_line: % TODO: manual review -- edited your text: "twelve model types used in this thesis" became "twelve model types the code supports" (the map includes CatBoost and a CNN, which the thesis never uses); stray "and" removed from the last-but-one sentence.
OLD:
```latex
The most useful features for an RF may not be the most useful for an SVM, and vice versa. The relative importance of a feature is dependent on the model \citep{Dong2020}. Within the search grid, we use an internal evaluator to fit on one candidate feature set at a time. This process is separate from the final downstream model evaluation. In a standard search, four internal evaluators are used: RF, DNN, LightGBM, and the matched evaluator. When the internal evaluator is set to \texttt{match}, the search scores each candidate hybrid on the same model type as the downstream model, for each of the twelve model types used in this thesis. So internal evaluators outside the standard sweep may be tested.  For a downstream model outside of the pre-defined set, if no matched evaluator exists in the code, and the search uses the other internal evaluators that were requested.
```
NEW:
```latex
% TODO: manual review -- edited your text: "twelve model types used in this thesis" became "twelve model types the code supports" (the map includes CatBoost and a CNN, which the thesis never uses); stray "and" removed from the last-but-one sentence.
The most useful features for an RF may not be the most useful for an SVM, and vice versa. The relative importance of a feature is dependent on the model \citep{Dong2020}. Within the search grid, we use an internal evaluator to fit on one candidate feature set at a time. This process is separate from the final downstream model evaluation. In a standard search, four internal evaluators are used: RF, DNN, LightGBM, and the matched evaluator. When the internal evaluator is set to \texttt{match}, the search scores each candidate hybrid on the same model type as the downstream model, for each of the twelve model types the code supports. So internal evaluators outside the standard sweep may be tested. For a downstream model outside of the pre-defined set, if no matched evaluator exists in the code, the search uses the other internal evaluators that were requested.
```

### E36 — Comma splice
- file: thesis.tex
- kind: author-text-trivial
- why: Comma splice.
- todo_line: none (trivial)
OLD:
```latex
The internal evaluator ranks candidate feature sets, it explicitly does not rank features within a representation.
```
NEW:
```latex
The internal evaluator ranks candidate feature sets; it explicitly does not rank features within a representation.
```

### E38 — Cross-reference by subsection name, augmentation
- file: thesis.tex
- kind: ai-text
- why: Cross-reference by quoted subsection name (SG section 18).
- todo_line: keep existing
OLD:
```latex
They are added after the hybrid is built (see the Augmentation subsection above), and the same representation
```
NEW:
```latex
They are added after the hybrid is built (Section~\ref{sec:method:augmentation}), and the same representation
```

### E39 — One fold fewer: AUROC dataset reported in RMSE; datasets without a result
- file: thesis.tex
- kind: ai-text
- why: The range 0.009 to 0.037 RMSE covers the regression datasets only (source comment: caco2 0.0090 to herg_ki dnn 0.0372); hERG blocking is scored in AUROC. QM9, MDR1–MDCK and rat microsomal clearance have no result for this experiment (source comment of the results section), so "every small-molecule dataset" needs "with a result". The 37-word sentence is also split.
- todo_line: keep existing
OLD:
```latex
Repeating each fold's search with one more fold of training samples left out makes the held-out score worse on every small-molecule dataset with both models, by 0.009 to 0.037 RMSE. On LogD with the SVM, for example, it is worse by 0.0295 on average over the five held-out folds. The two searches differ in two ways at once, the training samples and the budget each one picks, so this says that the hybrid gains from more data, not which of the two changes causes it.
```
NEW:
```latex
Repeating each fold's search with one more fold of training samples left out makes the held-out score worse on every small-molecule dataset with a result, with both models. On the regression datasets it is worse by 0.009 to 0.037 RMSE, and on LogD with the SVM by 0.0295 on average over the five held-out folds. The two searches differ in two ways at once: the training samples and the budget each one picks. So this says that the hybrid gains from more data, not which of the two changes causes it.
```

### E40 — Rat clearance named consistently; two long sentences split
- file: thesis.tex
- kind: ai-text
- why: The dataset is "rat microsomal clearance" everywhere else; sentences of 41 and 45 words.
- todo_line: keep existing
OLD:
```latex
Across the configurations of a search, the choice of downstream model accounts for more of the variation in score than the budget on 11 of the 16 datasets with a result, for example 48.5\% against 1.4\% of the variance on LogD. The budget still interacts with the model's own settings. With the SVM on LogD, rat clearance, MDR1--MDCK efflux and Caco-2, the search picks 500 features on every fold when the SVM is strongly regularized ($C$ of 0.01 or 0.1), and 1,000 to 2,000 on LogD and rat clearance when it is not ($C = 10$).
```
NEW:
```latex
Across a search's configurations, the downstream model accounts for more of the variation in score than the budget on 11 of the 16 datasets with a result. On LogD, for example, it accounts for 48.5\% of the variance and the budget for 1.4\%. The budget still interacts with the model's own settings. On LogD, rat microsomal clearance, MDR1--MDCK efflux and Caco-2, the search picks 500 features on every fold when the SVM is strongly regularized ($C$ of 0.01 or 0.1). It picks 1,000 to 2,000 on LogD and rat microsomal clearance when it is not ($C = 10$).
```

### E41 — Caption of the best-number-of-features figure does not describe the figure
- file: thesis.tex
- kind: caption
- why: The figure (figures_ch5_7/f02a_best_number_of_features.png, viewed 2026-10-10) shows one point per dataset and downstream model at its best budget, on a budget axis; it does not plot held-out score against budget, as the caption says. SG section 16: say what one point is, the metric and its direction, and the model labels. "NN" in the legend is the DNN. The marked-budget sentence and the fold count are kept.
- todo_line: keep existing
OLD:
```latex
\caption[Held-out score against the number of features.]{Held-out score of the best hybrid at each budget, per dataset and downstream model, mean over five outer folds, in each dataset's own metric. The marked budget is the best of these means, read off the held-out folds.}
```
NEW:
```latex
\caption[Best number of features per dataset and downstream model.]{Best number of features per dataset and downstream model. One point is one dataset and model, placed at the budget whose hybrid had the best mean held-out score over five outer folds. Scores are in the dataset's own metric (lower RMSE, higher Spearman or AUROC). The best budget is read off the held-out folds. NN is the DNN.}
```

### E42 — Caption of the blueprints figure: what one bar and one segment are
- file: thesis.tex
- kind: caption
- why: SG section 16: say what one bar is. In the figure (viewed 2026-10-10) each bar is split into segments by representation and the model is named in parentheses as "NN" for the DNN.
- todo_line: keep existing
OLD:
```latex
\caption[Features taken from each representation.]{Number of features the hybrid took from each base representation, one bar per dataset, on the first outer fold with the dataset's first downstream model.}
```
NEW:
```latex
\caption[Features taken from each representation.]{Number of features the hybrid took from each base representation, on the first outer fold. One bar is one dataset with its first downstream model, named in parentheses (NN is the DNN), and one segment is one base representation.}
```

### E43 — Antibody representation table caption: LM undefined
- file: thesis.tex
- kind: caption
- why: "LM" is used in about twenty rows of the table and never defined.
- todo_line: % TODO: manual review -- edited your text: one sentence added to the caption defining LM.
OLD:
```latex
\caption[Representations used for antibodies.]{Representations used for antibodies. Feature counts marked /chain double on the datasets with heavy and light chains.
```
NEW:
```latex
% TODO: manual review -- edited your text: one sentence added to the caption defining LM.
\caption[Representations used for antibodies.]{Representations used for antibodies. LM is a language model. Feature counts marked /chain double on the datasets with heavy and light chains.
```

### E44 — Main result paragraph: numbers that sit in the table, and a 40-word sentence
- file: thesis.tex
- kind: ai-text
- why: Task item 6 and SG section 3: the Caco-2 and QM9 pairs are already in Table tab:hybrid_vs_best_single (figures_ch5_7/tab_hybrid_vs_best_single.tex, read 2026-10-10), and "The gains run from X to Y ... to A to B" reads as one range. One pair (LogD) and the AUROC pair stay. "The comparison is a fair one" is an appraisal (40 words); it is split into what was done.
- todo_line: keep existing
OLD:
```latex
It scores better than the best single representation on every dataset, with both downstream models (Table~\ref{tab:hybrid_vs_best_single}), and on the six regression datasets it does so on every one of the five folds. The gains run from 0.3036 to 0.2862 RMSE on Caco-2 with the SVM to 0.4930 to 0.4302 on LogD with the SVM, and on QM9 the DNN goes from 0.3619 to 0.3155 eV. On hERG blocking the hybrid is ahead on four of five folds with each model, from 0.8555 to 0.8666 AUROC with LightGBM. The comparison is a fair one: the single representation is the one the search itself ranked first on each fold's training molecules, refit and scored on the same held-out molecules, so the hybrid is not being compared with a representation nobody would have picked.
```
NEW:
```latex
It scores better than the best single representation on every dataset, with both downstream models (Table~\ref{tab:hybrid_vs_best_single}). On the six regression datasets it does so on every one of the five folds. On LogD with the SVM, for example, the RMSE falls from 0.4930 to 0.4302. On hERG blocking the hybrid is ahead on four of five folds with each model, from 0.8555 to 0.8666 AUROC with LightGBM. The single representation is the one the search itself ranked first on each fold's training molecules. It is refit and scored on the same held-out molecules, so the hybrid is not compared with a representation nobody would have picked.
```

### E45 — Cut walk-through paragraph: numbers that sit in the table; long sentences
- file: thesis.tex
- kind: ai-text
- why: Task item 6: the per-cut numbers are in Table tab:cut_walkthrough. The XGBoost and LightGBM best scores (0.5842, 0.5707) are dropped from the prose; the 0.5707 is in the table. Sentences of 34 to 37 words split.
- todo_line: keep existing
OLD:
```latex
The best had an RMSE of 0.5472 p$K_i$ (SVM, 1,000 features, LightGBM gain, softmax allocation, random forest evaluator, all representations), with a standard deviation over the five folds of 0.0443 p$K_i$. Keeping only the SVM and the DNN removed 71\% of the recorded fitting time, mostly XGBoost (35\%) and LightGBM (25\%), whose best configurations scored 0.5842 and 0.5707 p$K_i$. The budget, importance, allocation and representation-count cuts then removed most of the rest. The best configuration removed by each of these four cuts was 0.0051 to 0.0154 p$K_i$ behind the best kept, less than the spread over folds. The final grid keeps 224 configurations and 1.4\% of the fitting time, and still contains the best configuration. The cuts were chosen on this same search, so keeping its best configuration is expected; the walk-through shows how much each cut removes and how close the best removed configuration came.
```
NEW:
```latex
The best had an RMSE of 0.5472 p$K_i$ (SVM, 1,000 features, LightGBM gain, softmax allocation, random forest evaluator, all representations). Its standard deviation over the five folds was 0.0443 p$K_i$. Keeping only the SVM and the DNN removed 71\% of the recorded fitting time, mostly from XGBoost (35\%) and LightGBM (25\%). The budget, importance, allocation and representation-count cuts then removed most of the rest. For each of these four cuts, the best configuration it removed was 0.0051 to 0.0154 p$K_i$ behind the best kept, less than the spread over folds. The final grid keeps 224 configurations and 1.4\% of the fitting time, and still contains the best configuration. The cuts were chosen on this same search, so keeping its best configuration is expected. The walk-through shows how much each cut removes and how close the best removed configuration came.
```

### E46 — Long sentences split: grid choice
- file: thesis.tex
- kind: ai-text
- why: 38-word sentence (SG gap 27).
- todo_line: keep existing
OLD:
```latex
On each dataset we kept the two downstream models with the best scores, and the budgets whose ten best configurations scored within 1\% of the best budget (3\% on datasets under about 1,000 samples), widened to at least three consecutive budgets.
```
NEW:
```latex
On each dataset we kept the two downstream models with the best scores. We also kept the budgets whose ten best configurations scored within 1\% of the best budget (3\% on datasets under about 1,000 samples), widened to at least three consecutive budgets.
```

### E47 — Long sentences split: design experiments example
- file: thesis.tex
- kind: ai-text
- why: 46-word sentence.
- todo_line: keep existing
OLD:
```latex
For example, on hERG $K_i$ with linear allocation at 300 features, five ways of rescaling each representation's score gave RMSE values from 0.5870 to 0.5924 (p$K_i$), while a single rescaling ranged from 0.5475 to 0.6294 across 5 folds, so the rescaling is one fixed rule per metric.
```
NEW:
```latex
For example, on hERG $K_i$ with linear allocation at 300 features, five ways of rescaling each representation's score gave RMSE values from 0.5870 to 0.5924 (p$K_i$). A single rescaling ranged from 0.5475 to 0.6294 across 5 folds, so the rescaling is one fixed rule per metric.
```

### E48 — Long sentences split: residual ranking
- file: thesis.tex
- kind: ai-text
- why: 40-word sentence with an embedded formula.
- todo_line: keep existing
OLD:
```latex
A small random forest (50 trees, depth at most 10) is fitted on the base features, and 3-fold group-aware cross-validation gives out-of-fold residuals $r_i = y_i - \hat{y}^{\mathrm{oof}}_i$ for the internal-training molecules, where for binary tasks $\hat{y}$ is the positive-class probability.
```
NEW:
```latex
A small random forest (50 trees, depth at most 10) is fitted on the base features. Three-fold group-aware cross-validation gives out-of-fold residuals $r_i = y_i - \hat{y}^{\mathrm{oof}}_i$ for the internal-training molecules. For binary tasks, $\hat{y}$ is the positive-class probability.
```

### E49 — Long sentences split: adaptive pruning
- file: thesis.tex
- kind: ai-text
- why: Sentences of 31 and 40 words.
- todo_line: keep existing
OLD:
```latex
The search also has an adaptive pruning option, which we did not use in this research: because of time constraints, the per-dataset cuts above, which remove far more configurations, were used instead. Pruning scores every configuration on the first two of the five folds and stops scoring those that are clearly behind. Budgets are scored from smallest to largest, and once every budget up to 100 features is scored, pairs of allocation and importance method with no configuration in the better half of their group are not searched at larger budgets.
```
NEW:
```latex
The search also has an adaptive pruning option, which we did not use in this research. Because of time constraints, we used the per-dataset cuts above instead, which remove far more configurations. Pruning scores every configuration on the first two of the five folds and stops scoring those that are clearly behind. Budgets are scored from smallest to largest. Once every budget up to 100 features is scored, an allocation and importance pair with no configuration in the better half of its group is not searched at larger budgets.
```

### E50 — Long sentences split: experimental setup, folds and clipping
- file: thesis.tex
- kind: ai-text
- why: Sentences of 43 and 37 words.
- todo_line: keep existing
OLD:
```latex
The code limits some experiments to the first three outer folds on most datasets, but this limit was added on 6 and 7 October 2026, after most results reported here had run on all five, so each result states its number of folds. In the comparison with the best single representation, the easy-and-hard-halves experiment and the comparison of many and few representations, each held-out feature is also limited to the range it takes on the training samples.
```
NEW:
```latex
The code limits some experiments to the first three outer folds on most datasets. This limit was added on 6 and 7 October 2026, after most results reported here had run on all five, so each result states its number of folds. Three experiments also limit each held-out feature to the range it takes on the training samples. They are the comparison with the best single representation, the easy and hard halves, and the comparison of many and few representations.
```

### E51 — Long sentences split: comparator
- file: thesis.tex
- kind: ai-text
- why: Sentences of 41 and 47 words.
- todo_line: keep existing
OLD:
```latex
This comparator is chosen per downstream model and per fold from the search's scores on the outer training samples, then refit on those samples and scored on the same held-out samples as the hybrid, so it can differ between folds. Choosing it on the held-out samples would give it an advantage that the hybrid does not get. The hybrid is held to its budget, plus any augmentation features, so the comparator is often many times wider, and both widths are recorded for every fold, since a difference between the two scores is partly a difference in width as well as in method.
```
NEW:
```latex
This comparator is chosen per downstream model and per fold from the search's scores on the outer training samples. It is then refit on those samples and scored on the same held-out samples as the hybrid, so it can differ between folds. Choosing it on the held-out samples would give it an advantage that the hybrid does not get. The hybrid is held to its budget, plus any augmentation features, so the comparator is often many times wider. Both widths are recorded for every fold, since a difference between the two scores is partly a difference in width as well as in method.
```

### E52 — Long sentence split: model limits
- file: thesis.tex
- kind: ai-text
- why: 38-word sentence.
- todo_line: keep existing
OLD:
```latex
The Gaussian process is skipped above 1,000 molecules, because it factorizes the full kernel matrix, and the SVM above 10,000 molecules (5,000 under augmentation beyond the budget), because a fit no longer finishes in useful time.
```
NEW:
```latex
The Gaussian process is skipped above 1,000 molecules, because it factorizes the full kernel matrix. The SVM is skipped above 10,000 molecules (5,000 under augmentation beyond the budget), because a fit no longer finishes in useful time.
```

### E53 — Long sentences split: 8 October corrections
- file: thesis.tex
- kind: ai-text
- why: Sentences of 40 and 42 words.
- todo_line: keep existing
OLD:
```latex
Before, one molecule with an extreme descriptor value could make a fold's error several orders of magnitude larger on rat microsomal clearance, and the experiments where this happened have no result for that dataset until they are re-run. The Gaussian process now starts its kernel length scale at the square root of the number of features rather than at 1, because on inputs of over 1,000 features a start of 1 left some fits predicting one value for every sample.
```
NEW:
```latex
Before, one molecule with an extreme descriptor value could make a fold's error several orders of magnitude larger on rat microsomal clearance. The experiments where this happened have no result for that dataset until they are re-run. The Gaussian process now starts its kernel length scale at the square root of the number of features rather than at 1. On inputs of over 1,000 features, a start of 1 left some fits predicting one value for every sample.
```

### E54 — Long sentence split: overfitting budgets
- file: thesis.tex
- kind: ai-text
- why: 43-word sentence.
- todo_line: keep existing
OLD:
```latex
The budgets are the standard list from 10 to 2{,}000 features, plus 4{,}000 and 8{,}000, plus 0.5, 0.75, 1, 1.25, 1.5 and 2 times the smallest outer training set, which place a budget where the number of features equals the number of samples.
```
NEW:
```latex
The budgets are the standard list from 10 to 2{,}000 features, plus 4{,}000 and 8{,}000. They also include 0.5, 0.75, 1, 1.25, 1.5 and 2 times the smallest outer training set, which places one budget where the number of features equals the number of samples.
```

### E55 — Long sentence split: fewer features or a tuned model (methods)
- file: thesis.tex
- kind: ai-text
- why: 65-word sentence.
- todo_line: keep existing
OLD:
```latex
For the SVM and the random forest, the best hybrid at a small budget, fitted at the model's default setting, is compared with a hybrid rebuilt at the largest width we allow (the smallest of the pool width, twice the number of training samples and 2{,}048 features), fitted at the setting (the SVM's $C$ or the forest's maximum depth) that scores best on the other folds.
```
NEW:
```latex
For the SVM and the random forest, the best hybrid at a small budget is fitted at the model's default setting. It is compared with a hybrid rebuilt at the largest width we allow: the smallest of the pool width, twice the number of training samples and 2{,}048 features. This wider hybrid is fitted at the setting that scores best on the other folds (the SVM's $C$ or the forest's maximum depth).
```

### E56 — Long sentences split: shuffling result on small molecules
- file: thesis.tex
- kind: ai-text
- why: Sentences of 36, 44 and 31 words; "drops through five to seven representations" does not say what is spread.
- todo_line: keep existing
OLD:
```latex
Every representation in each pool was also scored on its own on the same folds, so we know which single representation was best on the held-out molecules, a choice no user could make in advance. On every small-molecule dataset the hybrid beats that representation too, with both models. On Caco-2 with the DNN the best single representation in hindsight is PDV, at 0.2980 RMSE, against 0.2775 for the hybrid. The reason lies in what the hybrid draws on. When each representation's features are shuffled in turn on the held-out molecules, the score of the hybrid on hERG $K_i$, Caco-2 and MDR1--MDCK efflux drops through five to seven representations about equally, and no single representation carries more than 29\% of it.
```
NEW:
```latex
Every representation in each pool was also scored on its own on the same folds. So we know which single representation was best on the held-out molecules, a choice no user could make in advance. On every small-molecule dataset the hybrid beats that representation too, with both models. On Caco-2 with the DNN the best single representation in hindsight is PDV, at 0.2980 RMSE, against 0.2775 for the hybrid. The reason lies in what the hybrid draws on. We shuffled each representation's features in turn on the held-out molecules. On hERG $K_i$, Caco-2 and MDR1--MDCK efflux, the drop in the hybrid's score is spread about equally over five to seven representations. No single representation carries more than 29\% of it.
```

### E57 — Long sentence split: QM9 and hERG blocking lean on one representation
- file: thesis.tex
- kind: ai-text
- why: 31-word sentence.
- todo_line: keep existing
OLD:
```latex
On QM9 with LightGBM and on hERG blocking the hybrid leans more on one representation, which carries 88\% and about half of its score, and it still beats every single representation.
```
NEW:
```latex
On QM9 with LightGBM and on hERG blocking the hybrid leans more on one representation, which carries 88\% and about half of its score. It still beats every single representation.
```

### E58 — Long sentences split: antibody results
- file: thesis.tex
- kind: ai-text
- why: Sentences of 40, 43 and 46 words.
- todo_line: keep existing
OLD:
```latex
On binding free energy the mean gain is the largest in the chapter, 0.0819 to 0.2178 with the SVM and 0.1317 to 0.3542 with the DNN, though the gain moves by more than its own size from fold to fold.
```
NEW:
```latex
On binding free energy the mean gain is the largest in the chapter, 0.0819 to 0.2178 with the SVM and 0.1317 to 0.3542 with the DNN. The gain moves by more than its own size from fold to fold.
```
(Second part of E58, next paragraph.)
OLD:
```latex
In an antibody representation, as in a molecular fingerprint, the features that matter are few and buried among many that carry nothing: a language model's embedding of the whole sequence mixes the CDRs that vary with a framework that mostly does not. Many of the antibody representations in the pools were built for this thesis to separate the two. They read a model's embedding over the CDRs only, over each CDR or each of the seven IMGT regions separately, or over the most variable positions, and they describe the change in the embedding from the parent antibody and the residues and properties of each mutation (Table~\ref{tab:reps_antibodies}).
```
NEW:
```latex
In an antibody representation, as in a molecular fingerprint, the features that matter are few and buried among many that carry nothing. A language model's embedding of the whole sequence mixes the CDRs that vary with a framework that mostly does not. Many of the antibody representations in the pools were built for this thesis to separate the two. They read a model's embedding over the CDRs only, over each CDR or each of the seven IMGT regions separately, or over the most variable positions. Others describe the change in the embedding from the parent antibody, and the residues and properties of each mutation (Table~\ref{tab:reps_antibodies}).
```
(Third part of E58, same paragraph.)
OLD:
```latex
When each representation's features are shuffled in turn in the binding free energy hybrid, the representations built for this thesis carry 0.83 of the drop in score with the SVM and 0.81 with the DNN.
```
NEW:
```latex
When each representation's features are shuffled in turn in the binding free energy hybrid, the representations built for this thesis carry 0.83 of the drop in score with the SVM. With the DNN they carry 0.81.
```

### E59 — Long sentences split: HIC and Buchwald–Hartwig
- file: thesis.tex
- kind: ai-text
- why: Sentences of 35 and 47 words.
- todo_line: keep existing
OLD:
```latex
On HIC retention it is behind with the Gaussian process, 0.5515 against 0.6000, and here the search picked the same ESM-2 representation on every fold, so the single representation was not the problem.
```
NEW:
```latex
On HIC retention it is behind with the Gaussian process, 0.5515 against 0.6000. Here the search picked the same ESM-2 representation on every fold, so the single representation was not the problem.
```
(Second part of E59.)
OLD:
```latex
The search drew on only two of the ten representations on every fold, the reaction fingerprint and the change in the MHG-GNN embedding from reactants to product, and the folds hold out whole additives, so on this split the reaction representations carry little that the hybrid could combine.
```
NEW:
```latex
The search drew on only two of the ten representations on every fold: the reaction fingerprint and the change in the MHG-GNN embedding from reactants to product. The folds hold out whole additives, so on this split the reaction representations carry little that the hybrid could combine.
```

### E60 — Long sentences split: ensembles
- file: thesis.tex
- kind: ai-text
- why: Sentences of 36, 42 and 35 words.
- todo_line: keep existing
OLD:
```latex
On the anti-VEGF expression scan, where the hybrid beat the best single representation on every fold with the SVM, the DNN ensemble reaches 0.7854 Spearman against 0.7348 for the hybrid, ahead on every fold.
```
NEW:
```latex
On the anti-VEGF expression scan, the hybrid beat the best single representation on every fold with the SVM. Yet the DNN ensemble reaches 0.7854 Spearman against 0.7348 for the hybrid, ahead on every fold.
```
(Second part of E60.)
OLD:
```latex
On antibodies the representations are language models of one sequence that make similar predictions, and averaging several of them smooths out the noise of each model on a small training set, which one model on one feature set cannot do. On small molecules the representations carry different information, and one model that sees the selected features of all of them together can use their interactions, which an average of separate models never sees.
```
NEW:
```latex
On antibodies the representations are language models of one sequence that make similar predictions. Averaging several of them smooths out the noise of each model on a small training set, which one model on one feature set cannot do. On small molecules the representations carry different information. One model that sees the selected features of all of them together can use their interactions, which an average of separate models never sees.
```

### E61 — Long sentence split: many against few representations
- file: thesis.tex
- kind: ai-text
- why: 34-word sentence.
- todo_line: keep existing
OLD:
```latex
With the budget filled from the top-ranked representation alone, the score is worse than the search's hybrid at every budget searched, with both models, on 15 of the 18 datasets with a result.
```
NEW:
```latex
We filled the budget from the top-ranked representation alone. Its score is worse than the search's hybrid at every budget searched, with both models, on 15 of the 18 datasets with a result.
```

### E62 — Long sentences split: overfitting result
- file: thesis.tex
- kind: ai-text
- why: Sentences of 38 and 34 words.
- todo_line: keep existing
OLD:
```latex
The search scores its own pick on the training samples, so that score is optimistic by an amount we can measure: the search's score minus the held-out score of the same pick, with budgets searched up to 8,000 features.
```
NEW:
```latex
The search scores its own pick on the training samples, so that score is optimistic. We measure the optimism as the search's score minus the held-out score of the same pick. Budgets are searched up to 8,000 features.
```
(Second part of E62.)
OLD:
```latex
With a few thousand molecules the search's internal scores are reliable enough to choose between configurations, and with 150 to 500 training antibodies they are not, so the choice it makes is partly noise.
```
NEW:
```latex
With a few thousand molecules the search's internal scores are reliable enough to choose between configurations. With 150 to 500 training antibodies they are not, so the choice it makes is partly noise.
```

### E63 — Long sentence split: Ambroise paragraph
- file: thesis.tex
- kind: ai-text
- why: 38-word sentence.
- todo_line: keep existing
OLD:
```latex
The hybrid search computes all importance scores within each fold, on training data only, and every number in this chapter is scored on molecules the search never saw, so the gains above are not this bias.
```
NEW:
```latex
The hybrid search computes all importance scores within each fold, on training data only. Every number in this chapter is scored on samples the search never saw, so the gains above are not this bias.
```

### E64 — Long sentence split: recursive elimination
- file: thesis.tex
- kind: ai-text
- why: 42-word sentence.
- todo_line: keep existing
OLD:
```latex
Recursive elimination ranks every feature of the joined representations in one list with one forest, which is the pooled ranking the hybrid avoids (Section~\ref{sec:method:searchspace}): features from a 2,048-bit fingerprint, a descriptor vector and a learned embedding are put on one scale.
```
NEW:
```latex
Recursive elimination ranks every feature of the joined representations in one list with one forest, which is the pooled ranking the hybrid avoids (Section~\ref{sec:method:searchspace}). Features from a 2,048-bit fingerprint, a descriptor vector and a learned embedding are put on one scale.
```

### E65 — Long sentences split: budgets by model
- file: thesis.tex
- kind: ai-text
- why: Sentences of 35 and 46 words.
- todo_line: keep existing
OLD:
```latex
The SVM's smaller picks fit what Chapter~\ref{chap:flaws} showed for distances: a kernel computed over many features is diluted by the ones that carry nothing, while a network can learn to ignore them. The small antibody sets pick between 25 and 500 features on angiopoietin-2 and binding free energy, while on expression titer the random forest's pick moves from 250 to 2,000 features between folds, which is the selection noise of the previous section showing up in the budget.
```
NEW:
```latex
The SVM's smaller picks fit what Chapter~\ref{chap:flaws} showed for distances. A kernel computed over many features is diluted by the ones that carry nothing, while a network can learn to ignore them. The small antibody sets pick between 25 and 500 features on angiopoietin-2 and binding free energy. On expression titer the random forest's pick moves from 250 to 2,000 features between folds, which is the selection noise of the previous section showing up in the budget.
```

### E66 — Long sentences split: evaluator, importance in training, tuning
- file: thesis.tex
- kind: ai-text
- why: Sentences of 38, 41 and 38 words.
- todo_line: keep existing
OLD:
```latex
On some antibody datasets it matters: on EGFR binding the matched evaluator gives 0.3527 Spearman with the SVM against 0.2993 for a LightGBM evaluator, and it is also best with both models on expression titer and colloidal stability.
```
NEW:
```latex
On some antibody datasets it matters. On EGFR binding the matched evaluator gives 0.3527 Spearman with the SVM against 0.2993 for a LightGBM evaluator. It is also best with both models on expression titer and colloidal stability.
```
(Second part of E66.)
OLD:
```latex
On antibodies some condition helps on most datasets, and by more: feature scaling takes colloidal stability with the DNN from 0.2607 to 0.3571 Spearman, and keeping the above-median features takes binding free energy with the SVM from 0.2388 to 0.3124.
```
NEW:
```latex
On antibodies some condition helps on most datasets, and by more. Feature scaling takes colloidal stability with the DNN from 0.2607 to 0.3571 Spearman. Keeping the above-median features takes binding free energy with the SVM from 0.2388 to 0.3124.
```
(Third part of E66.)
OLD:
```latex
A hybrid of up to 2,048 features with the SVM's regularization setting tuned beats the search's smaller hybrid at the default setting on all four small-molecule datasets with a result, for example 0.4114 against 0.4299 RMSE on LogD.
```
NEW:
```latex
A hybrid of up to 2,048 features with the SVM's regularization setting tuned beats the search's smaller hybrid at the default setting. It does so on all four small-molecule datasets with a result, for example 0.4114 against 0.4299 RMSE on LogD.
```

### E67 — New embedding: long sentences, and "language model" applied to reactions
- file: thesis.tex
- kind: ai-text
- why: Sentences of 44 and 37 words; the closing clause about "one more language model of the same sequence" was attached to the reaction dataset too, where it does not apply.
- todo_line: keep existing
OLD:
```latex
Taking each embedding in the pool in turn as the new one, its selected features beat Gaussian random features put in the same places in the hybrid for every embedding on Caco-2 and hERG $K_i$, and for seven or eight of eight on LogD. On the antibody and reaction datasets they do so for fewer than half of the embeddings on colloidal stability, expression titer and the Buchwald--Hartwig reactions, where one more language model of the same sequence adds little.
```
NEW:
```latex
Each embedding in the pool was taken in turn as the new one. Its selected features beat Gaussian random features put in the same places in the hybrid. They do so for every embedding on Caco-2 and hERG $K_i$, and for seven or eight of eight on LogD. On colloidal stability, expression titer and the Buchwald--Hartwig reactions, they do so for fewer than half of the embeddings. On the two antibody datasets, one more language model of the same sequence adds little.
```

### E68 — Calibration: long sentence, missing noun
- file: thesis.tex
- kind: ai-text
- why: 40-word sentence; "the hybrid's cover" lacks its noun.
- todo_line: keep existing
OLD:
```latex
With 90\% conformal intervals, the hybrid's cover between 0.886 and 0.909 of the held-out molecules, as the single representation's do, while being narrower: on LogD with the SVM the mean width falls from 1.652 to 1.403 log units.
```
NEW:
```latex
With 90\% conformal intervals, the hybrid's intervals cover between 0.886 and 0.909 of the held-out molecules, as the single representation's do. They are also narrower: on LogD with the SVM the mean width falls from 1.652 to 1.403 log units.
```

### E69 — Long sentences split: blueprints and stability
- file: thesis.tex
- kind: ai-text
- why: Sentences of 45, 39 and 38 words.
- todo_line: keep existing
OLD:
```latex
Elsewhere one or two representations carry the hybrid: on EGFR binding 1,275 of its 1,315 features come from ESM-2 run on the joined chains, and the Buchwald--Hartwig hybrid draws only on the reaction fingerprint and the change in the MHG-GNN embedding from reactants to product.
```
NEW:
```latex
Elsewhere one or two representations carry the hybrid. On EGFR binding 1,275 of its 1,315 features come from ESM-2 run on the joined chains. The Buchwald--Hartwig hybrid draws only on the reaction fingerprint and the change in the MHG-GNN embedding from reactants to product.
```
(Second part of E69.)
OLD:
```latex
Many features are correlated within and across representations, so a feature on one fold can stand in for a different one on another, and an index well below 1 does not mean the hybrid is fitting noise.
```
NEW:
```latex
Many features are correlated within and across representations, so a feature on one fold can stand in for a different one on another. An index well below 1 therefore does not mean the hybrid is fitting noise.
```
(Third part of E69.)
OLD:
```latex
The low antibody values fit the rest of this chapter: features drawn from several language models of one sequence are largely interchangeable, and on a few hundred antibodies which of them is picked changes from fold to fold.
```
NEW:
```latex
The low antibody values fit the rest of this chapter. Features drawn from several language models of one sequence are largely interchangeable, and on a few hundred antibodies which of them is picked changes from fold to fold.
```

### E70 — Long sentences split: shared rows and transfer
- file: thesis.tex
- kind: ai-text
- why: Sentences of 42 and 45 words.
- todo_line: keep existing
OLD:
```latex
On hERG $K_i$, where the graph kernel leaves 105 molecules sharing a row, adding features from ECFP4 and MHG-GNN takes it from 0.6109 to 0.5715 RMSE with the SVM, lower on all five folds, and no two molecules share a row.
```
NEW:
```latex
On hERG $K_i$ the graph kernel leaves 105 molecules sharing a row. Adding features from ECFP4 and MHG-GNN takes it from 0.6109 to 0.5715 RMSE with the SVM, lower on all five folds, and no two molecules share a row.
```
(Second part of E70.)
OLD:
```latex
Search settings found on LogD and reused on hERG $K_i$, with the features selected again on hERG $K_i$, beat hERG $K_i$'s best single representation with both models (SVM 0.5888 to 0.5665 RMSE, on all five folds) but stay behind a search run on hERG $K_i$ itself (0.5631).
```
NEW:
```latex
We reused search settings found on LogD on hERG $K_i$, with the features selected again on hERG $K_i$. They beat hERG $K_i$'s best single representation with both models (SVM 0.5888 to 0.5665 RMSE, on all five folds). They stay behind a search run on hERG $K_i$ itself (0.5631).
```

### E71 — Long sentences split: separate sites
- file: thesis.tex
- kind: ai-text
- why: Sentences of 35, 38, 35 and 38 words.
- todo_line: keep existing
OLD:
```latex
Selecting at each site with an even share of the budget does as well as one selection over all families, within 0.0013 RMSE on every dataset and model, although it can build a different number of features.
```
NEW:
```latex
Selecting at each site with an even share of the budget does as well as one selection over all families, within 0.0013 RMSE on every dataset and model. It can build a different number of features, though.
```
(Second part of E71.)
OLD:
```latex
With two-class labels and up to 30\% of each site's labels flipped, at different rates on different sites, selecting features at each site beats the collaborator's ensembles and their student model at every flip level on Caco-2.
```
NEW:
```latex
In this experiment, labels are two-class and up to 30\% of each site's labels are flipped, at different rates on different sites. Selecting features at each site beats the collaborator's ensembles and their student model at every flip level on Caco-2.
```
(Third part of E71.)
OLD:
```latex
It never beats the same selection made with all molecules pooled (0.5204 and 0.4324), so keeping the data apart still costs something, though with the DNN at 30\% flips the two are within 0.0001.
```
NEW:
```latex
It never beats the same selection made with all molecules pooled (0.5204 and 0.4324), so keeping the data apart still costs something. With the DNN at 30\% flips, though, the two are within 0.0001.
```
(Fourth part of E71.)
OLD:
```latex
Selecting features separately in each of five $k$-means clusters of Caco-2 and fitting one model on all molecules matches one global selection (0.5289 against 0.5316 MCC with the DNN), while fitting a separate model per cluster is worse (0.4859).
```
NEW:
```latex
We selected features separately in each of five $k$-means clusters of Caco-2 and fitted one model on all molecules. This matches one global selection (0.5289 against 0.5316 MCC with the DNN), while fitting a separate model per cluster is worse (0.4859).
```

### E72 — Long sentence split: limitations
- file: thesis.tex
- kind: ai-text
- why: 43-word sentence.
- todo_line: keep existing
OLD:
```latex
Its gains are consistent on datasets of a few thousand samples and inconsistent on datasets of a few hundred, where the search's internal scores overstate what the chosen hybrid delivers by up to 0.16 Spearman (Section~\ref{sec:method:results_overfit}) and its prediction intervals cover too few held-out samples.
```
NEW:
```latex
Its gains are consistent on datasets of a few thousand samples and inconsistent on datasets of a few hundred. On those smaller datasets the search's internal scores overstate what the chosen hybrid delivers by up to 0.16 Spearman (Section~\ref{sec:method:results_overfit}), and its prediction intervals cover too few held-out samples.
```

### E73 — Long sentences split: conclusions
- file: thesis.tex
- kind: ai-text
- why: Sentences of 34, 37, 41, 43, 42 and 42 words. The forward pointer to Chapter 6 at the end of the Conclusions already matches the references (SG section 1; e.g. V, end of §3.5: "Having established the effectiveness of this strategy, we next explore a complementary approach ..."), so it is not changed.
- todo_line: keep existing
OLD:
```latex
On every small-molecule dataset it beats the best single representation with both downstream models, on every fold of the six regression datasets, and it beats even the representation that was best in hindsight. It does so because the representations carry different information: on hERG $K_i$, Caco-2 and MDR1--MDCK efflux the hybrids draw on five to seven of them, and no one holds more than 29\% of the signal.
```
NEW:
```latex
On every small-molecule dataset it beats the best single representation with both downstream models, on every fold of the six regression datasets. It also beats the representation that was best in hindsight. It does so because the representations carry different information. On hERG $K_i$, Caco-2 and MDR1--MDCK efflux the hybrids draw on five to seven of them, and no one holds more than 29\% of the signal.
```
(Second part of E73.)
OLD:
```latex
At the same number of features the hybrid beats recursive elimination and random features on four of the five small-molecule datasets with a result, and drawing on several representations beats drawing on the top one alone on 15 of 18 datasets. How the budget is split matters less than which representations are drawn on. No single setting suits every dataset and model: the DNN wants more features than the SVM, the number of representations ranges from two to all ten, and the cheaper internal evaluators are good enough on small molecules but not on some antibody datasets.
```
NEW:
```latex
At the same number of features the hybrid beats recursive elimination and random features on four of the five small-molecule datasets with a result. Drawing on several representations beats drawing on the top one alone on 15 of 18 datasets. How the budget is split matters less than which representations are drawn on. No single setting suits every dataset and model. The DNN wants more features than the SVM, and the number of representations ranges from two to all ten. The cheaper internal evaluators are good enough on small molecules but not on some antibody datasets.
```
(Third part of E73.)
OLD:
```latex
On binding free energy, AC-SINS self-association and the 4,275-variant expression scan it beats every single representation, and on the first and last of these the best single representation is one built here to read the regions and mutations that vary.
```
NEW:
```latex
On binding free energy, AC-SINS self-association and the 4,275-variant expression scan it beats every single representation. On the first and last of these, the best single representation is one built here to read the regions and mutations that vary.
```
(Fourth part of E73.)
OLD:
```latex
The method also runs on reactions, where it gains nothing on the additive split, on data held at separate sites, where selecting at each site does as well as one selection over all representations, and with new embeddings added to the pool.
```
NEW:
```latex
The method also runs on reactions, where it gains nothing on the additive split. It runs on data held at separate sites, where selecting at each site does as well as one selection over all representations. It also takes in new embeddings added to the pool.
```

## For the author (not changed)

- K_i typography, thesis-wide: this chapter has `$K_i$` (italic i) about 30 times and `$\mathrm{p}K_i$` twice in the transfer study ("the target is hERG $\mathrm{p}K_i$", "RMSE in $\mathrm{p}K_i$ units") against `p$K_i$` elsewhere. CLG section 2 recommends one macro, `\newcommand{\Ki}{\ensuremath{K_\mathrm{i}}}`, in main.tex and a replace-all over thesis.tex. Changing Chapter 5 alone would make the thesis less consistent, so it is not planned here.
- "logP"/"LogD" typography: CLG section 2 recommends log $P$ and log $D$ for the quantity, keeping "LogD" as the dataset name. Chapter 2 writes "logP", so E3 and E10 keep "logP" to match. Decide once for the whole thesis.
- hERG, Caco-2, MDR1–MDCK and LogD are never defined (CLG item 16). Their first use is Chapter 3's Datasets, so the definitions belong there, not here. Suggested wording is in CLG section 3.
- Dataset name "Binding free energy" (about 40 uses, including Table tab:datasets and Table tab:dataset_reps): the label is ΔΔG, the change in binding free energy on mutation (CLG item 8). Renaming it is a thesis-wide change. The sign convention is also not stated.
- Activity-cliff study, "the metric is the mean absolute error (MAE) in log units of activity": CLG item 5 says Dablander's labels are logs of nM and µM values, not molar p-values, so "log units of activity" may mix two units. Not checked against the script in this session.
- Your TODO "double check this is an accurate description of what's happening with large reps - which method???" (above "When combining representations, we are careful to avoid data leakage."): the sentence matches the code. `HIGH_DIM_FILTER_THRESHOLD = 5000` in src/kirby/find_best_hybrid.py; the top 500 features (200 under 1,000 samples) are chosen per fold by a LightGBM model with 50 trees and depth 5; it changes only the standalone score. This is a different step from the importance pre-screen of representations over 2,000 features (2,000 highest-variance features) in Table tab:importance_methods. No edit planned.
- Your TODO "ensure consistent use of embeddings langauge" (above "Not every molecular representation is equal."): "features derived from embedded GNNs and pretrained language models" reads as if the GNNs were embedded. "embeddings from pretrained GNNs and language models" would match the thesis's other uses. Not changed, because the sentence reports Deng2023 and I did not check their wording.
- "In a dream world, there would be a single answer ... Unfortunately we live in reality" is informal (SG section 2 lists it). Left as your phrasing; E24 changed only pronouns and terms in that paragraph.
- Duplicate sentence: "Preliminary design experiments set the values each setting may take (Additional file~\ref{app:add:design_experiments})." repeats the first sentence of the paragraph above it ("Before the full runs, we ran preliminary design experiments ..."). Deleting it would leave its TODO line orphaned, so it is left for you.
- "Unless stated otherwise, an experiment takes from each fold's search, for each of the dataset's two downstream models, the two best single representations and the two best hybrids." repeats the last sentence of "Every experiment below runs the search inside each outer fold, ...". One of the two can go.
- Figure fig:best_number_of_features marks the budget read off held-out scores, while the text beside it gives the budgets each fold picked on its training molecules. The two can differ (the AI TODO above the figure already flags this). E41 makes the caption say what is drawn, but not which source you want.
- Figure fig:blueprints puts binding free energy under a "Proteins" panel, separate from the antibodies, and its legends use code names (ESM2-Concat, CGR-Morgan, Difference-MHG-GNN) that differ from the table names (ESM-2 (joined chains), CGR fingerprint, MHG-GNN (difference)). This needs a figure-script change.
- Generated Table tab:hybrid_vs_best_single labels the DNN "NN", and writes "hERG Ki", "MDR1-MDCK" and "Buchwald-Hartwig" with hyphens instead of en dashes. This needs a figure-script change.
- Table tab:datasets labels without units: "SMAC retention time", "HIC retention time", "AC-SINS shift at pH 7.4", "expression titer", "mutational scan score". The AI TODO above the table says these were not checked in the loaders.
- Tally phrasing in results ("on four of five folds", "on 15 of the 18 datasets", "six of the nine antibody datasets") is left. SG section 10 and gap 33 leave it to you whether fold and dataset tallies are allowed in prose.
- Buchwald–Hartwig: "which aryl halide, ligand, base and additive was used" (Chemical reactions paragraph and Table tab:reps_reactions). CLG section 1 suggests "phosphine ligand" here. Optional.
- "The two results likely follow from what the hybrids draw on." (ensembles paragraph) is an untested explanation, hedged once. It is left as is.
- The Conclusions already end by pointing to Chapter 6 ("Chapter~\ref{chap:similarity} asks whether a feature set chosen for the task also gives better neighbors ..."), and the chapter opening looks back to Chapter 4. Neither needs changing.
- (Checker) Chapter opening, "This chapter measures predictive performance under both standard labels and labels with injected artificial noise." Partly true still: the separate-sites experiment in this chapter flips up to 30% of each site's labels. Decide whether to keep, narrow or drop it. The planned edit (old E16) was removed.
- (Checker) "Scaling and filling of missing values happen inside each fold, on the training rows only, as described in the construction of the hybrid." Section sec:method:build does not describe scaling or filling of missing values, so neither the description nor a \ref to it points at that text. The planned edit (old E37) was removed.

## Checker notes

Checked 2026-10-10 against thesis.tex. Every OLD block (including second, third and fourth parts) is an exact substring with `str.count == 1` (python3 -I). No NEW block adds a number absent from thesis.tex. No banned words added. Labels unchanged. Edit IDs were not renumbered, so E16 and E37 are missing.

Deleted (2):
- E16: author text. Adding which Chapter 1 questions the chapter answers is a style change, not a fault. The noise sentence it removed is not plainly wrong (the separate-sites experiment flips labels). Moved to For the author.
- E37: author text. The cross-reference was informal, not wrong, and the proposed \ref{sec:method:build} points at a section that does not describe scaling or missing values. Moved to For the author.

Fixed (17):
- E2: "log $D$" changed to "logD", to match E3 and the rest of the chapter.
- E3: a 31-word sentence in NEW split in two.
- E4: "A scaffold is the ring systems ..." reworded so subject and verb agree.
- E6: NEW sentence was 34 words; split.
- E8: NEW sentence was 31 words; "the residues of" dropped.
- E15: joining the fragment produced a 41-word sentence. Now the fragment becomes a sentence by dropping "With the understanding that". todo_line updated to say so.
- E17: "the gain of Section" became "the gain reported in Section".
- E20: NEW sentence was 32 words; split at the colon.
- E35: the "why" and todo_line said the thesis uses "six plus kNN". That is wrong: Chapter 3 uses thirteen base models. The fix itself is right, because the twelve-model map in src/kirby/find_best_hybrid.py includes CatBoost and a CNN, and the thesis never uses either (grep). Reason text corrected.
- E40: two NEW sentences of 31 words shortened.
- E41: NEW caption sentence was 34 words; split.
- E44: "The largest gain is on LogD" ranked RMSE values in different units (log units against eV). Replaced with the plan's own fallback wording, "for example". The check-before-applying note was removed.
- E49, E56, E67: NEW sentences of 31, 32 and 32 words split or shortened.
- E58 (third part) and E62 (first part): awkward NEW sentences reworded ("That is with the SVM; ...", "We measure by how much as ...").

Kept as is (54): E1, E5, E7, E9 to E14, E18, E19, E21 to E34, E36, E38, E39, E42, E43, E45 to E48, E50 to E55, E57, E59 to E61, E63 to E66, E68 to E73. Checks made:
- E7: the five Ginkgo assays named match the dataset table (SMAC, HIC, AC-SINS, titer, CHO).
- E5/E6: the COVID Moonshot source is confirmed by the chemistry language guide (Dablander p. 51).
- E9: FcRn occurs nowhere else in the thesis.
- E31: Table tab:search_tiers now lists setting values only, with no speed tiers.
- E33: matches $s^{(k)}_j \ge 0$ and the 0 scores in the importance table.
- E43: "LM" is defined nowhere in thesis.tex or main.tex.
- E41 and E42: checked against the figure PNGs.
- Second-person fixes in author text are kept as convention fixes (the thesis uses "we").
- E25, E28 and E35 still have 32 to 39-word sentences. These are the author's own sentences with a few words changed. Splitting them would be a style change to her text, so they are left.
