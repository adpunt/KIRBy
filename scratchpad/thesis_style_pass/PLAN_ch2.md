# Plan: Chapter 2 (Molecular Representations, Models and Label Noise, chap:background)

Audited against THESIS_STYLE_GUIDE.md and CHEMISTRY_LANGUAGE_GUIDE.md on 2026-10-10. Every OLD block was checked with python3 -I to be an exact substring occurring once in thesis.tex. The "% TODO: manual review" comments on author text sit directly above the paragraph (or above the \caption line for captions).

## Edits

### E1 — Overview: look back to Chapter 1 and say what this chapter does
- file: thesis.tex
- kind: new-paragraph
- why: The section is empty; style guide item 1 (B and V open each chapter by looking back, D gives a roadmap) and item 4 (empty Overview at the old line 216).
- todo_line: % TODO: manual review
OLD:
```latex
\section{Overview}


\section{Molecular representations in QSAR}
```
NEW:
```latex
\section{Overview}

% TODO: manual review
Chapter~\ref{chap:intro} set out the problem of this thesis: predicting molecular properties when the labels used for training are noisy. Its first four questions ask how the representation, the model and the type of noise each affect the result. This chapter gives the background the later chapters rely on. It first describes the molecular representations and the models we use, and then defines molecular similarity and activity cliffs. It then describes where label noise in molecular data comes from, how we inject it, and the methods for training with noisy labels. It ends with NoiseInject, the package that injects the noise and computes the metrics, and with how results are evaluated.

\section{Molecular representations in QSAR}
```

### E2 — Caption of the three-representations figure matches what is drawn
- file: thesis.tex
- kind: caption
- why: Chemistry guide item 4: panel a has teal arrows, no orange, and shows integer identifiers rather than bit positions; the caption described colours and a bit collision the figure does not show.
- todo_line: first line of NEW
OLD:
```latex
    \caption[Three representations of one molecule.]{\label{fig:molecular_representations}Three representations of the molecule \lstinline{O=C1C=CN(C1)C=O}. a) ECFP4, in which circular substructures up to a radius of two bonds are hashed (green arrows) to positions in a bit vector, and two substructures can share a position (bit collision, orange). b) Tokenization and one-hot encoding of the SMILES string. c) Molecular graph, with atoms as nodes and bonds as edges.}
```
NEW:
```latex
    % TODO: manual review -- edited your text: "green arrows" and "bit collision, orange" removed and "hashed to positions in a bit vector" changed to integer identifiers, because panel a shows neither colour nor a bit vector; panel a still draws a different molecule (see For the author).
    \caption[Three representations of one molecule.]{\label{fig:molecular_representations}Three representations of the molecule \lstinline{O=C1C=CN(C1)C=O}. a) ECFP4, in which circular substructures up to a radius of two bonds are hashed (arrows) to integer identifiers. Folding the identifiers into a bit vector can put two substructures at the same position (a bit collision). b) Tokenization and one-hot encoding of the SMILES string. c) Molecular graph, with atoms as nodes and bonds as edges.}
```

### E3 — ECFP default is without chirality
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 1: RDKit Morgan fingerprints leave chirality out unless asked, and the thesis itself builds ECFP4 with stereochemistry off (Chapter 4).
- todo_line: first line of NEW
OLD:
```latex
Another common representation is the molecular fingerprint, a vector containing information about the properties of the molecule in question. In a typical bit vector-based fingerprint, each bit accounts for the presence (1) or absence (0) of a given chemical feature or substructure \citep{Fassio2022}. One common fingerprint is the Molecular ACCess System (MACCS) key, a substructure key-based fingerprint that encodes predefined atom symbols and properties, as well as bond types \citep{Xie2020}. Another is the Extended Connectivity Fingerprint (ECFP), which encodes local neighborhoods around each atom and bond connectivity in molecules \citep{Xie2020}. ECFPs are typically constructed with a radius of 2, a length of 2048 bits, and active chirality flags.
```
NEW:
```latex
% TODO: manual review -- edited your text: "and active chirality flags" changed to "without chirality unless it is requested", because the usual default (and this thesis's ECFP4) leaves stereochemistry out.
Another common representation is the molecular fingerprint, a vector containing information about the properties of the molecule in question. In a typical bit vector-based fingerprint, each bit accounts for the presence (1) or absence (0) of a given chemical feature or substructure \citep{Fassio2022}. One common fingerprint is the Molecular ACCess System (MACCS) key, a substructure key-based fingerprint that encodes predefined atom symbols and properties, as well as bond types \citep{Xie2020}. Another is the Extended Connectivity Fingerprint (ECFP), which encodes local neighborhoods around each atom and bond connectivity in molecules \citep{Xie2020}. ECFPs are typically constructed with a radius of 2 and a length of 2048 bits, without chirality unless it is requested.
```

### E4 — 2D graphs can carry stereochemistry
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 2: 2D graphs leave out 3D geometry but can encode stereochemistry as atom and bond labels, as the last sentence of the same paragraph says.
- todo_line: first line of NEW
OLD:
```latex
One of the more intuitive representations is the molecular graph. These graphs can be categorized into 2D and 3D graphs. 2D graphs rely on adjacency matrices that encode topological atom- and bond-level features but leave out geometric properties such as bond angles or torsion, meaning they may fail to capture stereochemistry.
```
NEW:
```latex
% TODO: manual review -- edited your text: "geometric properties such as bond angles or torsion, meaning they may fail to capture stereochemistry" changed, because a 2D graph can encode stereochemistry as atom and bond labels (your own featurization sentence below says so).
One of the more intuitive representations is the molecular graph. These graphs can be categorized into 2D and 3D graphs. 2D graphs rely on adjacency matrices that encode topological atom- and bond-level features but leave out 3D geometry such as bond angles and torsions. Stereochemistry can still be encoded as atom and bond labels.
```

### E5 — 2D graphs do not hold coordinates
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 3: a 2D graph holds connectivity, not atomic coordinates or bond angles; the sentence contradicted the previous paragraph.
- todo_line: first line of NEW
OLD:
```latex
In the standard molecular graph, atoms are mapped to nodes and bonds to edges \citep{David2020}. These molecular graphs are typically 2D objects that represent 3D information such as atomic coordinates, bond angles, chirality, etc. \citep{David2020}.
```
NEW:
```latex
% TODO: manual review -- edited your text: "represent 3D information such as atomic coordinates, bond angles, chirality" changed to connectivity plus optional stereochemistry labels, because a 2D graph holds no coordinates or bond angles; check that David2020 still supports the corrected sentence.
In the standard molecular graph, atoms are mapped to nodes and bonds to edges \citep{David2020}. These molecular graphs are typically 2D objects that hold connectivity and, optionally, stereochemistry labels, but not atomic coordinates or bond angles \citep{David2020}.
```

### E6 — Activity cliff: potency, and no "significant"
- file: thesis.tex
- kind: author-text
- why: Style guide item 10 ("significant" without a test) and chemistry guide section 1 ("activity" is biological; Section 2.3 defines a cliff by potency).
- todo_line: first line of NEW
OLD:
```latex
All the models up to this point have assumed a vector-based input. Molecular graphs, on the other hand, require different types of predictive modeling. Two standard ML methods on molecular graphs include Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks (GINs). GCNs apply the concept of convolution to graph-structured data. Instead of operating on grid-based data, such as images, GCNs aggregate information from neighbors of a node to update its representation \citep{Kipf2017}. In contrast, GINs address the limitations in GCNs’ ability to differentiate between non-isomorphic graphs. Unlike GCNs, which normalize and weight the sum of neighboring node features, GINs use a simple sum followed by a trainable MLP, giving the model flexibility to learn complex transformations of node features. The GIN model’s design is informed by the Weisfeiler--Lehman test for graph isomorphism, enabling it to distinguish between molecular graphs that are structurally similar but differ in subtle ways \citep{TeachOpenCADD2023, Xu2019}. GIN features have been competitive with or better than fingerprints and descriptors at classifying pairs of molecules as activity cliffs (ACs; Section~\ref{sec:mol_similarity}), where small variations in molecular structure can lead to significant changes in chemical or biological activity
```
NEW:
```latex
% TODO: manual review -- edited your text: "small variations ... significant changes in chemical or biological activity" changed to "a small change ... a large change in potency", to match the cliff definition in the similarity section and to avoid "significant" where no test was run.
All the models up to this point have assumed a vector-based input. Molecular graphs, on the other hand, require different types of predictive modeling. Two standard ML methods on molecular graphs include Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks (GINs). GCNs apply the concept of convolution to graph-structured data. Instead of operating on grid-based data, such as images, GCNs aggregate information from neighbors of a node to update its representation \citep{Kipf2017}. In contrast, GINs address the limitations in GCNs’ ability to differentiate between non-isomorphic graphs. Unlike GCNs, which normalize and weight the sum of neighboring node features, GINs use a simple sum followed by a trainable MLP, giving the model flexibility to learn complex transformations of node features. The GIN model’s design is informed by the Weisfeiler--Lehman test for graph isomorphism, enabling it to distinguish between molecular graphs that are structurally similar but differ in subtle ways \citep{TeachOpenCADD2023, Xu2019}. GIN features have been competitive with or better than fingerprints and descriptors at classifying pairs of molecules as activity cliffs (ACs; Section~\ref{sec:mol_similarity}), where a small change in molecular structure leads to a large change in potency
```

### E7 — i.e. used for examples
- file: thesis.tex
- kind: author-text-trivial
- why: Fingerprints and descriptors are examples of feature vectors, so "e.g." is meant.
- todo_line: none (trivial)
OLD:
```latex
(i.e., fingerprints, descriptors)
```
NEW:
```latex
(e.g., fingerprints, descriptors)
```

### E8 — "Standardized kernels" and the Gauche name
- file: thesis.tex
- kind: author-text
- why: Wrong word: "standardized" means rescaled, "standard" is meant; the library name was written two ways in one paragraph.
- todo_line: first line of NEW
OLD:
```latex
We employed Gaussian process (GP) models using the \texttt{Gauche} framework \citep{gauche}, which is optimized for cheminformatics tasks. Standardized kernels for GPs are not optimized for the chemical landscape. GAUCHE defines
```
NEW:
```latex
% TODO: manual review -- edited your text: "Standardized kernels" changed to "Standard kernels" (standardized means rescaled), and "GAUCHE" written as \texttt{Gauche} like the rest of the paragraph.
We employed Gaussian process (GP) models using the \texttt{Gauche} framework \citep{gauche}, which is optimized for cheminformatics tasks. Standard kernels for GPs are not optimized for the chemical landscape. \texttt{Gauche} defines
```

### E9 — Similar property principle: split 32-word sentence
- file: thesis.tex
- kind: ai-text
- why: Sentence over 30 words; "noise identification algorithms" replaced by the thesis term "noise detection".
- todo_line: keep existing
OLD:
```latex
One of its clearest applications is nearest-neighbor prediction, which appears across cheminformatics workflows and, in this thesis, in the kernels used by GPs and at the basis of many noise identification algorithms.
```
NEW:
```latex
One of its clearest applications is nearest-neighbor prediction, which appears across cheminformatics workflows. In this thesis, the same principle underlies the kernels used by GPs and many noise detection methods.
```

### E11 — Activity cliff paragraph: split, K_i typography, cliff molecules
- file: thesis.tex
- kind: ai-text
- why: Two sentences over 30 words; chemistry guide item 10 (K_i as $K_\mathrm{i}$); "cliff compounds" defined as molecules in at least one cliff pair (guide sections 1 and 3).
- todo_line: keep existing
OLD:
```latex
The sharpest failure of the similar property principle is the activity cliff, a pair of structurally similar molecules with a large difference in potency that is produced by a small
chemical change \citep{Maggiora2006, Stumpfe2012}. Activity cliffs inherit the measure dependence of the principle they violate, and there is no widely accepted definition of one
\citep{vanTilborg2022}. The MoleculeACE benchmark \citep{vanTilborg2022} makes this choice explicit. A pair is considered highly similar if its similarity exceeds 90\% under at least one of
three measures, namely the Tanimoto coefficient on ECFPs, the Tanimoto coefficient on scaffold fingerprints and a scaled Levenshtein distance on SMILES strings. Such a pair forms a cliff if
the two molecules also differ tenfold or more in $K_i$ or EC$_{50}$ \citep{vanTilborg2022}. This soft consensus over three measures is used because each measure captures only part of the
underlying chemistry. Across the thirty MoleculeACE targets, cliff compounds account for between 7\% and 52\% of a dataset \citep{vanTilborg2022}.
```
NEW:
```latex
The sharpest failure of the similar property principle is the activity cliff. This is a pair of structurally similar molecules whose potency differs by a large amount because of a small chemical change \citep{Maggiora2006, Stumpfe2012}. Activity cliffs inherit the measure dependence of the principle they violate, and there is no widely accepted definition of one \citep{vanTilborg2022}. The MoleculeACE benchmark \citep{vanTilborg2022} makes this choice explicit. A pair is considered highly similar if its similarity exceeds 90\% under at least one of three measures. These are the Tanimoto coefficient on ECFPs, the Tanimoto coefficient on scaffold fingerprints and a scaled Levenshtein distance on SMILES strings. Such a pair forms a cliff if the two molecules also differ tenfold or more in $K_\mathrm{i}$ or EC$_{50}$ \citep{vanTilborg2022}. This soft consensus over three measures is used because each measure captures only part of the underlying chemistry. Across the thirty MoleculeACE targets, molecules in at least one cliff pair make up between 7\% and 52\% of a dataset \citep{vanTilborg2022}.
```

### E12 — Dimensionality paragraph: split, features not dimensions
- file: thesis.tex
- kind: ai-text
- why: Two sentences over 30 words; the author's term is "features".
- todo_line: keep existing
OLD:
```latex
Dimensionality itself poses a further difficulty. \citet{Beyer1999} show that as the number of dimensions grows, the distances to the nearest and to the furthest
point become increasingly similar, so that pairwise distances become progressively less discriminative. Molecular representations are routinely high-dimensional and sparse, and when a
large proportion of their dimensions carry no information relevant to the property being modeled, distances computed over them are dominated by those uninformative dimensions.
```
NEW:
```latex
Dimensionality itself poses a further difficulty. \citet{Beyer1999} show that as the number of dimensions grows, the distances to the nearest and to the furthest point become increasingly similar. Pairwise distances therefore tell neighbors apart less and less well. Molecular representations are routinely high-dimensional and sparse. When a large proportion of their features carry no information about the property being modeled, distances computed over them are dominated by those uninformative features.
```

### E13 — "discrete quantities" for point labels
- file: thesis.tex
- kind: author-text
- why: "Discrete" means countable; the point is that a label is treated as one exact value rather than a distribution.
- todo_line: first line of NEW
OLD:
```latex
One key issue with QSAR models is the standard ML assumption that labels represent true values, as most chemical datasets lack sufficient replications to provide a robust statistical representation of the underlying distributions \citep{Kolmar2021}. A single measurement, or even multiple measurements for a given label, will not guarantee that this label accurately reflects the population mean \citep{Kolmar2021}. Most ML models rely on the assumption that the training data is, in fact, an accurate reflection of the population. Therefore, they treat these potentially noisy labels as discrete quantities rather than distributions
```
NEW:
```latex
% TODO: manual review -- edited your text: "discrete quantities" changed to "exact values", because discrete means countable and the point is a single value versus a distribution.
One key issue with QSAR models is the standard ML assumption that labels represent true values, as most chemical datasets lack sufficient replications to provide a robust statistical representation of the underlying distributions \citep{Kolmar2021}. A single measurement, or even multiple measurements for a given label, will not guarantee that this label accurately reflects the population mean \citep{Kolmar2021}. Most ML models rely on the assumption that the training data is, in fact, an accurate reflection of the population. Therefore, they treat these potentially noisy labels as exact values rather than distributions
```

### E14 — Contraction and Anderson--Darling dash
- file: thesis.tex
- kind: author-text
- why: Register: no contractions (style guide item 2); two-name test takes an en dash.
- todo_line: first line of NEW
OLD:
```latex
Studies involving noise, and specifically the quantification of noise robustness, typically approach label noise by adding independent, zero-mean, Gaussian error to the labels. However, there's evidence to suggest that noise from experimental assays takes on different forms. \citet{Kruger2012} rejected normality of the differences in binding affinity between orthologous targets by Anderson-Darling test
```
NEW:
```latex
% TODO: manual review -- edited your text: "there's" changed to "there is" (no contractions in the thesis body); en dash in Anderson--Darling.
Studies involving noise, and specifically the quantification of noise robustness, typically approach label noise by adding independent, zero-mean, Gaussian error to the labels. However, there is evidence to suggest that noise from experimental assays takes on different forms. \citet{Kruger2012} rejected normality of the differences in binding affinity between orthologous targets by Anderson--Darling test
```

### E15 — "you", detection limit, tense for one study
- file: thesis.tex
- kind: author-text
- why: Second person (style guide item 2); chemistry guide item 12 (censoring comes from the range an assay can measure, not only signal detection); past tense for one study (style guide item 11).
- todo_line: first line of NEW
OLD:
```latex
Beyond just the distribution, the assumptions of independence and zero mean may fail as well. \citet{Bentz2013} examined P-glycoprotein transport measurements from 23 laboratories and found that, among the laboratories using cell lines, 62\% of the variance in log digoxin efflux ratio was due to differences between laboratories. When you pool data from different sources, each source has the potential to carry a shared offset. These sources usually contain families of related compounds, breaking the independence assumption. The zero-mean assumption is not assured either, especially when true values fall above or below an assay's detection limit, called censoring. \citet{Svensson2025} count censored labels in 15 industrial assays, of which 13 carry censored labels and eight sit between a quarter and two thirds censored.
```
NEW:
```latex
% TODO: manual review -- edited your text: "When you pool data" made impersonal (no second person), the censoring sentence split, with "an assay's detection limit" changed to the range an assay can measure (censoring comes from the tested range), and "count" put in the past tense for one study.
Beyond just the distribution, the assumptions of independence and zero mean may fail as well. \citet{Bentz2013} examined P-glycoprotein transport measurements from 23 laboratories and found that, among the laboratories using cell lines, 62\% of the variance in log digoxin efflux ratio was due to differences between laboratories. When data from different sources are pooled, each source has the potential to carry a shared offset. These sources usually contain families of related compounds, breaking the independence assumption. The zero-mean assumption is not assured either. When true values fall outside the range an assay can measure, they are reported at the edge of that range, which is called censoring. \citet{Svensson2025} counted censored labels in 15 industrial assays, of which 13 carry censored labels and eight sit between a quarter and two thirds censored.
```

### E16 — Validation labels are noised, not clean
- file: thesis.tex
- kind: author-text
- why: Factual contradiction: the next sentence (and Chapter 3, "Validation labels receive an independent draw") says validation labels get noise.
- todo_line: first line of NEW
OLD:
```latex
Our objective is to evaluate the robustness of QSAR models against label noise. To do so, we ``inject'' artificial noise into the labels during training while validation and test data remain free of noise.
```
NEW:
```latex
% TODO: manual review -- edited your text: "while validation and test data remain free of noise" changed to "while the test labels remain free of noise", because the next sentence and Chapter 3 give validation labels noise.
Our objective is to evaluate the robustness of QSAR models against label noise. To do so, we ``inject'' artificial noise into the training labels, while the test labels remain free of noise.
```

### E17 — Define the Bemis--Murcko scaffold and scaffold family
- file: thesis.tex
- kind: new-paragraph
- why: Chemistry guide items 14 and 15: Bemis--Murcko is used here for the first time in the thesis and never defined; Chapter 3 (chirality-blind, each acyclic molecule its own group) and Chapter 5 (keeping stereochemistry, one empty scaffold) use two versions.
- todo_line: % TODO: manual review
OLD:
```latex
We explored seven different ways of adding noise, which we refer to as noise conditions and which are illustrated in Figure~\ref{fig:noise_conditions} and summarized in Table~\ref{tab:noise_conditions}. Three conditions vary the shape of the distribution from which the error is drawn, and three concentrate the error on some molecules or share it within Bemis--Murcko scaffold families. The seventh condition, censoring, clips labels at a limit.
```
NEW:
```latex
We explored seven different ways of adding noise, which we refer to as noise conditions and which are illustrated in Figure~\ref{fig:noise_conditions} and summarized in Table~\ref{tab:noise_conditions}. Three conditions vary the shape of the distribution from which the error is drawn, and three concentrate the error on some molecules or share it within Bemis--Murcko scaffold families. The seventh condition, censoring, clips labels at a limit.

% TODO: manual review
The Bemis--Murcko scaffold of a molecule is its ring systems and the chains that link them, with all other side chains removed \citep{Bemis1996}. Molecules that share one scaffold form a scaffold family. Chapters~\ref{chap:robustness} and~\ref{chap:method} state whether the scaffold keeps stereochemistry and how molecules with no ring are grouped.
```

### E18 — Equation followed by "Here" after a comma
- file: thesis.tex
- kind: author-text-trivial
- why: The equation ends with a comma, so the sentence continues with "where".
- todo_line: none (trivial)
OLD:
```latex
\end{equation}
Here $\tau$ is the target standard deviation
```
NEW:
```latex
\end{equation}
where $\tau$ is the target standard deviation
```

### E19 — K_i typography, noise sources table
- file: thesis.tex
- kind: author-text-trivial
- why: Chemistry guide item 10: italic K, roman subscript i.
- todo_line: none (trivial)
OLD:
```latex
independent $K_i$ measurements of one compound--target pair
```
NEW:
```latex
independent $K_\mathrm{i}$ measurements of one compound--target pair
```

### E20 — pK_i typography, noise conditions table
- file: thesis.tex
- kind: author-text-trivial
- why: Chemistry guide item 10.
- todo_line: none (trivial)
OLD:
```latex
differences in p$K_i$ between related targets
```
NEW:
```latex
differences in p$K_\mathrm{i}$ between related targets
```

### E21 — British spelling in Table tab:noise_terms
- file: thesis.tex
- kind: author-text-trivial
- why: American spelling ("neighbor" everywhere else).
- todo_line: none (trivial)
OLD:
```latex
is a neighbours' mean, a predicted value
```
NEW:
```latex
is a neighbors' mean, a predicted value
```

### E22 — LNL paragraph: split into three paragraphs, sentences under 30 words
- file: thesis.tex
- kind: ai-text
- why: Four sentences over 30 words, one of 75; the Song2022 list rewritten as one sentence per group. Content unchanged.
- todo_line: keep existing (and "% TODO: manual review" above the two new paragraphs)
OLD:
```latex
The problem of training a model when some of its labels are wrong is called learning with noisy labels (LNL). A method that acts on the training data has two halves: what selects the molecules, and what is then done to them. Three things can select molecules. A detector infers from the features and the labels how wrong each label looks, and the score it gives a molecule is that molecule's suspicion score. A known flag is already recorded in the data, with nothing inferred: a label at the assay limit, how many times a molecule was measured, which source reported it, or which scaffold group it belongs to. The third option is to select every molecule. Three actions can then follow: the molecule is deleted, it is down-weighted, or its label is moved toward a target by some distance. These two halves give three groups of methods, set out in Table~\ref{tab:noise_terms}. Noise detection is a detector alone, with nothing done afterwards. Noise mitigation is one of the three selectors followed by one of the three actions, so that any model can then be fitted to the result; Figure~\ref{fig:data_cleaning} shows how such a method is evaluated. Noise-robust training selects nothing and leaves the data untouched. Instead the loss, the kernel, a noise term, the training loop or the model changes, as in co-teaching \citep{Han2018} and Meta-Weight-Net \citep{shu2019metaweightnet}. A sample-selection step that is taken out of its training loop and used only to delete molecules before a refit is a detector followed by deletion, and so counts as noise mitigation. \citet{Song2022} provide a comprehensive review of deep learning with noisy labels, identifying five main approaches: (i) modifying architectures to better handle noise in training; (ii) employing regularization to reduce overfitting to noise; (iii) adjusting loss functions to be less sensitive to outliers or label flips; (iv) loss adjustment, which corrects, reweights or relabels the loss of each example during training; and (v) sample selection, which identifies examples with correct labels during training, using several networks or several rounds of training. All five act during training, so all five are noise-robust training. A detector that runs once and hands a smaller training set to any model has no place among them.
```
NEW:
```latex
The problem of training a model when some of its labels are wrong is called learning with noisy labels (LNL). A method that acts on the training data has two halves: what selects the molecules, and what is then done to them. Three things can select molecules. A detector infers from the features and the labels how wrong each label looks, and the score it gives a molecule is that molecule's suspicion score. A known flag is already recorded in the data, with nothing inferred. Examples are a label at the assay limit, how many times a molecule was measured, which source reported it, or which scaffold group it belongs to. The third option is to select every molecule. Three actions can then follow: the molecule is deleted, it is down-weighted, or its label is moved toward a target by some distance.

% TODO: manual review
These two halves give three groups of methods, set out in Table~\ref{tab:noise_terms}. Noise detection is a detector alone, with nothing done afterwards. Noise mitigation is one of the three selectors followed by one of the three actions, after which any model can be fitted to the result. Figure~\ref{fig:data_cleaning} shows how such a method is evaluated. Noise-robust training selects nothing and leaves the data untouched. Instead the loss, the kernel, a noise term, the training loop or the model changes, as in co-teaching \citep{Han2018} and Meta-Weight-Net \citep{shu2019metaweightnet}. A sample-selection step can be taken out of its training loop and used only to delete molecules before a refit. It is then a detector followed by deletion, and so counts as noise mitigation.

% TODO: manual review
\citet{Song2022} review deep learning with noisy labels and sort the methods into five groups. Robust architectures change the network to handle noise. Robust regularization reduces overfitting to noise. Robust loss functions are less sensitive to outliers or label flips. Loss adjustment corrects, reweights or relabels the loss of each example during training. Sample selection picks out the examples with correct labels during training, using several networks or several rounds of training. All five act during training, so all five are noise-robust training. A detector that runs once and hands a smaller training set to any model has no place among them.
```

### E24 — Cortes2015 paragraph: split, stale pointer to Chapter 6
- file: thesis.tex
- kind: ai-text
- why: Two sentences over 30 words; neighbor-based noise mitigation on the hybrid is in Chapter 7 (chap:noise), not Chapter 6 (the TODO at the head of Chapter 7 asks for exactly this repointing).
- todo_line: keep existing
OLD:
```latex
\citet{Cortes2015} compared 12 algorithms on 12 bioactivity datasets with increasing Gaussian label noise; gradient boosting was the least tolerant, the other 11 degraded comparably, and two polynomial-kernel models became sensitive only at high noise. This is a recurring theme in LNL research: there is no universal solution to handling noise with experimental endpoints. Neighbor-based cleaning dates back to editing a training set by its nearest neighbors \citep{Wilson1972}, and it relies on similar molecules having similar labels, an assumption that activity cliffs break. QSAR models also frequently fail to predict activity cliffs \citep{Dablander2023}. Chapter~\ref{chap:flaws} tests the assumption that similar molecules have similar labels, and Chapter~\ref{chap:similarity} tests neighbor-based noise mitigation with the hybrid as the feature space.
```
NEW:
```latex
\citet{Cortes2015} compared 12 algorithms on 12 bioactivity datasets with increasing Gaussian label noise. Gradient boosting was the least tolerant, the other 11 degraded comparably, and two polynomial-kernel models became sensitive only at high noise. This is a recurring theme in LNL research: there is no universal solution to handling noise with experimental endpoints. Neighbor-based cleaning dates back to editing a training set by its nearest neighbors \citep{Wilson1972}. It relies on similar molecules having similar labels, an assumption that activity cliffs break. QSAR models also frequently fail to predict activity cliffs \citep{Dablander2023}. Chapter~\ref{chap:flaws} tests the assumption that similar molecules have similar labels, and Chapter~\ref{chap:noise} tests neighbor-based noise mitigation with the hybrid as the feature space.
```

### E25 — NoiseInject opening: split; name which datasets
- file: thesis.tex
- kind: ai-text
- why: 41-word sentence; "the three assay datasets" had not been introduced in this chapter.
- todo_line: keep existing
OLD:
```latex
NoiseInject, an open-source Python package, injects the noise on the three assay datasets and computes the robustness and uncertainty metrics, whereas QM9 uses a separate
pipeline, written in Python and Rust, that carries its own Rust implementation of the same conditions.
```
NEW:
```latex
NoiseInject, an open-source Python package, injects the noise on the three assay datasets of Chapter~\ref{chap:robustness} and computes the robustness and uncertainty metrics. QM9 instead uses a separate pipeline, written in Python and Rust, with its own Rust implementation of the same conditions.
```

### E26 — Grouped-wider sentence split
- file: thesis.tex
- kind: ai-text
- why: 43-word sentence.
- todo_line: keep existing
OLD:
```latex
grouped-wider condition, whole scaffold families are added in random order until they cover as close as possible
to $20\%$ of the training molecules, because a fixed fraction of families
gives a very different fraction of molecules when family sizes are uneven.
```
NEW:
```latex
grouped-wider condition, whole scaffold families are added in random order until they cover as close as possible
to $20\%$ of the training molecules. A fixed fraction of families would give a very different fraction of molecules when family sizes are uneven.
```

### E27 — Censored set sentence split
- file: thesis.tex
- kind: ai-text
- why: 31-word sentence.
- todo_line: keep existing
OLD:
```latex
On the training split, the censored
set is therefore exactly the $k = \mathrm{round}(fn)$ largest labels, whereas on any other split it consists of whichever
molecules lie past the training limit.
```
NEW:
```latex
On the training split, the censored
set is therefore exactly the $k = \mathrm{round}(fn)$ largest labels. On any other split it consists of whichever
molecules lie past the training limit.
```

### E28 — Injection record: split the list
- file: thesis.tex
- kind: ai-text
- why: 56-word sentence.
- todo_line: keep existing
OLD:
```latex
This record contains the target amount of noise, the factor
$G$, the solved scale, the delivered root-mean-square noise (in label units and as a fraction of the
clean label standard deviation), the mean of $\epsilon$, the fraction of molecules affected, the
number of groups and the largest share held by one group, and the random seeds.
```
NEW:
```latex
This record contains the target amount of noise, the factor
$G$, the solved scale and the delivered root-mean-square noise. The delivered noise is given in label units and as a fraction of the
clean label standard deviation. The record also holds the mean of $\epsilon$, the fraction of molecules affected, the
number of groups, the largest share held by one group, and the random seeds.
```

### E29 — Effective count sentence split
- file: thesis.tex
- kind: ai-text
- why: 32-word sentence.
- todo_line: keep existing
OLD:
```latex
The effective count $n_{\mathrm{eff}}$ is the number of
independent contributions over which the delivered amount of noise averages, and it falls below $n$ when the noise is
concentrated on a few molecules.
```
NEW:
```latex
The effective count $n_{\mathrm{eff}}$ is the number of
independent contributions over which the delivered amount of noise averages. It falls below $n$ when the noise is
concentrated on a few molecules.
```

### E30 — Degenerate noise-scale sentence split
- file: thesis.tex
- kind: ai-text
- why: 39-word sentence.
- todo_line: keep existing
OLD:
```latex
Under the
uniform and grouped-shifted targeting rules this scale is the same for every molecule, so the package flags the result
as degenerate, because a rank correlation against a constant is undefined and must not be reported as zero.
```
NEW:
```latex
Under the
uniform and grouped-shifted targeting rules this scale is the same for every molecule, so the package flags the result
as degenerate. A rank correlation against a constant is undefined and must not be reported as zero.
```

### E31 — Residual-sized injector: split three long sentences
- file: thesis.tex
- kind: ai-text
- why: Sentences of 37, 39 and 34 words.
- todo_line: keep existing
OLD:
```latex
In the experiments implemented in the KIRBy repository, this unit is the standard deviation of a model's out-of-fold residuals on
the clean training labels, so that the noise is sized against the error the model already makes. Its
conditions either add Gaussian noise with variance equal to the level times the square of the unit, or move $2\%$,
$5\%$ or $10\%$ of labels by the level times the unit, with a random sign or always upward. They can also add
Gaussian noise whose standard deviation follows either a per-molecule covariate or a hidden
log-normal draw, and both of these are scaled to the same mean variance as the first condition.
```
NEW:
```latex
In the experiments implemented in the KIRBy repository, this unit is the standard deviation of a model's out-of-fold residuals on
the clean training labels. The noise is then sized against the error the model already makes. Its
conditions either add Gaussian noise with variance equal to the level times the square of the unit, or move labels by the level times the unit. The labels moved are $2\%$,
$5\%$ or $10\%$ of the total, with a random sign or always upward. They can also add
Gaussian noise whose standard deviation follows either a per-molecule covariate or a hidden
log-normal draw. Both are scaled to the same mean variance as the first condition.
```

### E32 — Classification noise: split the list
- file: thesis.tex
- kind: ai-text
- why: Sentences of 44 and 31 words.
- todo_line: keep existing
OLD:
```latex
The package provides six label-flipping strategies, namely uniform flips,
flip rates set by class frequency, asymmetric flip rates for binary tasks, a random flip rate for each sample,
a separate flip rate for each class, and flips directed by a confusion matrix. Because the realized
flip rate differs from the nominal flip probability under most of these strategies, a binary search
finds the flip probability that produces a target fraction of changed labels.
```
NEW:
```latex
The package provides six label-flipping strategies. The first three are uniform flips,
flip rates set by class frequency, and asymmetric flip rates for binary tasks. The other three are a random flip rate for each sample,
a separate flip rate for each class, and flips directed by a confusion matrix. Under most of these strategies the realized
flip rate differs from the nominal flip probability. A binary search therefore
finds the flip probability that produces a target fraction of changed labels.
```

### E33 — Robustness metrics: split three long sentences
- file: thesis.tex
- kind: ai-text
- why: Sentences of 38, 32 and 42 words.
- todo_line: keep existing
OLD:
```latex
Robustness is read from the
retention curve $\mathrm{ret}(x) = m(x)/m(0)$, where $m$ is a bounded metric for which higher values are better
($R^2$ for regression; accuracy, F1, precision or recall for classification) and $x$ is the noise
level. The primary summary is the normalized area under this curve, \aucnorm{} (Equation~\ref{eq:aucnorm} in Chapter~\ref{chap:robustness}),
evaluated by the trapezoid rule. It is the mean fraction of clean performance retained across the
noise levels, lies approximately within $[0, 1]$, and is less directly correlated with clean-label $R^2$ than other robustness metrics we tested. A
secondary summary fits $\mathrm{ret}(x) \approx \exp\!\left(-(x/\tau_W)^{\beta}\right)$ to the curve. The shape parameter
$\beta$ separates curves that hold and then collapse ($\beta > 1$) from curves that fall early and
then plateau ($\beta < 1$), while $\tau_W$ sets the scale of the decay, and the fit requires at least four
noise levels.
```
NEW:
```latex
Robustness is read from the
retention curve $\mathrm{ret}(x) = m(x)/m(0)$, where $x$ is the noise
level. Here $m$ is a bounded metric for which higher values are better: $R^2$ for regression, and accuracy, F1, precision or recall for classification. The primary summary is the normalized area under this curve, \aucnorm{} (Equation~\ref{eq:aucnorm} in Chapter~\ref{chap:robustness}),
evaluated by the trapezoid rule. It is the mean fraction of clean performance retained across the
noise levels and lies approximately within $[0, 1]$. It is also less directly correlated with clean-label $R^2$ than other robustness metrics we tested. A
secondary summary fits $\mathrm{ret}(x) \approx \exp\!\left(-(x/\tau_W)^{\beta}\right)$ to the curve. The shape parameter
$\beta$ separates curves that hold and then collapse ($\beta > 1$) from curves that fall early and
then plateau ($\beta < 1$). The parameter $\tau_W$ sets the scale of the decay, and the fit requires at least four
noise levels.
```

### E34 — Uncertainty metrics: split the list and one long sentence
- file: thesis.tex
- kind: ai-text
- why: Sentences of 70 and 33 words.
- todo_line: keep existing
OLD:
```latex
These are the Spearman correlation between $u_i$ and the
absolute error $|y_i - \hat{y}_i|$, the empirical coverage $\frac{1}{n}\sum_i \mathbb{1}\left[|y_i -
\hat{y}_i| \le k u_i\right]$ at $k = 1$ and $k = 2$ against Gaussian targets of $68.3\%$ and
$95.4\%$, the mean interval width $\frac{1}{n}\sum_i 2u_i$, and the expected calibration error (ECE),
\begin{equation}
\mathrm{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{n}\,\left|\bar{u}_b - \overline{|y - \hat{y}|}_b\right|,
\label{eq:ece}
\end{equation}
computed over ten equal-count bins $B_b$ of $u$. All of these metrics are computed on the clean test labels. The Spearman
correlation between $u_i$ and the injected noise $|\epsilon_i|$ can only be computed where the noise
is known, which is on the corrupted molecules, so it requires a separate pair of inputs.
```
NEW:
```latex
The first is the Spearman correlation between $u_i$ and the
absolute error $|y_i - \hat{y}_i|$. The second is the empirical coverage $\frac{1}{n}\sum_i \mathbb{1}\left[|y_i -
\hat{y}_i| \le k u_i\right]$ at $k = 1$ and $k = 2$, against Gaussian targets of $68.3\%$ and
$95.4\%$. The third is the mean interval width $\frac{1}{n}\sum_i 2u_i$, and the fourth is the expected calibration error (ECE),
\begin{equation}
\mathrm{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{n}\,\left|\bar{u}_b - \overline{|y - \hat{y}|}_b\right|,
\label{eq:ece}
\end{equation}
computed over ten equal-count bins $B_b$ of $u$. All of these metrics are computed on the clean test labels. The Spearman
correlation between $u_i$ and the injected noise $|\epsilon_i|$ can only be computed on the corrupted molecules,
where the noise is known. It therefore requires a separate pair of inputs.
```

### E35 — Label for the evaluation section
- file: thesis.tex
- kind: title
- why: Needed so the Conclusions can point to it by Section~\ref instead of by its title (task item 3). Title unchanged.
- todo_line: none
OLD:
```latex
\section{Evaluation methodology}
```
NEW:
```latex
\section{Evaluation methodology}\label{sec:evaluation}
```

### E36 — Evaluation: stale chapter range
- file: thesis.tex
- kind: ai-text
- why: Chapter 7 (chap:noise) also states its metrics; the range stopped at Chapter 6.
- todo_line: keep existing
OLD:
```latex
Chapters~\ref{chap:flaws} to~\ref{chap:similarity} state the metric used for each dataset where it is introduced.
```
NEW:
```latex
Chapters~\ref{chap:flaws} to~\ref{chap:noise} state the metric used for each dataset where it is introduced.
```

### E37 — Conclusions: add similarity, Section ref, end on the next chapter
- file: thesis.tex
- kind: ai-text
- why: Style guide item 1 (end by naming the next chapter's question, as B and V do: "Having established ..., we next explore ..."); the similarity section was not summarized; "the section on evaluation methodology" becomes Section~\ref; stale range "Chapters 4 to 6" (the noise conditions are reused in Chapter 7).
- todo_line: keep existing
OLD:
```latex
This chapter has introduced both halves of a predictor, namely the representation and the model. Because real assay noise is not independent, identically distributed or symmetric, our artificial noise model comprises seven conditions. We have also set out three groups of methods for noisy labels, namely noise detection, noise mitigation and noise-robust training, and noted that no single method wins on every dataset. NoiseInject injects these noise conditions and reports the robustness and uncertainty metrics, and the section on evaluation methodology fixes the metrics. Chapter~\ref{chap:robustness} asks whether the representation or the model architecture governs how strongly a predictor degrades under label noise, through the four questions set out in Chapter~\ref{chap:intro}. The representations, similarity definitions and noise conditions introduced here are reused in Chapters~\ref{chap:flaws} to~\ref{chap:similarity}.
```
NEW:
```latex
This chapter has introduced both halves of a predictor, namely the representation and the model. Similarity between molecules depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle. Because real assay noise is not independent, identically distributed or symmetric, our artificial noise model comprises seven conditions. We have also set out three groups of methods for noisy labels: noise detection, noise mitigation and noise-robust training. No single method is expected to win on every dataset. NoiseInject injects these noise conditions and reports the robustness and uncertainty metrics, and Section~\ref{sec:evaluation} fixes the metrics. The representations, similarity definitions and noise conditions introduced here are reused in the later chapters. Chapter~\ref{chap:robustness} asks whether the representation or the model architecture governs how strongly a predictor degrades under label noise, through the four questions set out in Chapter~\ref{chap:intro}.
```
## For the author (not changed)

- Figure fig:molecular_representations, panel a draws a different molecule: a five-membered ring with a ring oxygen, a nitrogen and a C=O (an oxazolidinone-type ring), while `O=C1C=CN(C1)C=O` has no ring oxygen. Panels b and c do show that SMILES. Redraw panel a for the same molecule (E2 fixes only the caption's colours and bit-vector claim).
- Same figure, panel b: the printed vocabulary is `{'C': 0, '=': 1, 'O': 2}`, but the tokens include 1, N, ( and ). The one-hot matrix has three rows, so it encodes only C, = and O.
- "Scaffold family" against "scaffold group": this chapter uses both, e.g. "which scaffold group it belongs to" (LNL paragraph and Table tab:noise_terms) and "scaffold families" (noise conditions, NoiseInject). The chemistry guide (item 14) recommends "scaffold group" everywhere, because "family" suggests a chemical series. This is a thesis-wide choice, so it is not changed here. E17 defines "scaffold family"; change that sentence too if you pick "group".
- $K_\mathrm{i}$ typography: E19, E20 and E11 fix the three uses in this chapter by hand. A macro (`\newcommand{\Ki}{\ensuremath{K_\mathrm{i}}}`) in the preamble would fix the roughly 200 uses elsewhere in one place (chemistry guide item 10).
- "logP (lipophilicity)" and "such as logP, molecular weight and partial charges" (PDV paragraph): the chemistry guide writes the quantity as log $P$. Chapter 5 writes "logP" too, so pick one form thesis-wide.
- Censoring direction: NoiseInject clips the top ("the censored set is therefore exactly the $k = \mathrm{round}(fn)$ largest labels"). In potency data, real censoring mostly sits at the weak end (low p$K_\mathrm{i}$ or pIC$_{50}$). One sentence saying which end is clipped, and why, belongs in the censoring paragraph or Chapter 3's methods (chemistry guide item 13). Only you know the reason.
- hERG, Caco-2 and LogD are not named in this chapter (E25 now says "the three assay datasets of Chapter 3"). Their first use is in Chapter 1 ("three assay datasets: LogD, Caco-2 and hERG $K_i$"). The one-sentence definitions (chemistry guide items 9 and 16) belong in Chapter 3's datasets paragraph ("we used LogD and Caco-2 efflux permeability"), which also has the efflux ratio against permeability error.
- SMILES as an identity: the SMILES paragraph does not say that one molecule has many valid SMILES strings and that canonicalization picks one. Chapter 7 counts molecules by SMILES string. refs.bib has Weininger1988 but no canonical-SMILES paper (Weininger 1989), so no sentence was added.
- Evaluation methodology section: it does not state the reporting policy (per-fold scores and their spread, no significance tests). Chapter 5 states it ("We use no significance tests and no confidence intervals"). Chapter 3 runs Wilcoxon and Tukey tests, so a chapter-wide statement here would need to exclude Chapter 3. The splits text is also still missing (your TODO above the Conclusions).
- Figure fig:data_cleaning uses brain icons for the random forest and the Bayesian model (style guide item 17). A plain labelled box would not suggest a neural network.
- NoiseInject "version 1.0.0" against Zenodo's v0.3.0: the existing TODO(cite) stands.
- The symbol $\tau$ means two things: the target noise standard deviation (Equation eq:dose_matching) and, as $\tau_W$, the scale of the fitted retention curve. Consider another letter for the second.
- "Decision trees handle data in a ``contributory'' rather than ``consensus'' manner": no citation for these two terms. Check where they come from.
- "However, Bayesian methods such as Gaussian processes (GPs), and conformal prediction, do not make this assumption": conformal prediction is not Bayesian. The commas keep it separate, but a reader can parse it as a Bayesian method.
- Tense in the Sort \& Slice paragraph: "Sort \& Slice fingerprints outperform ECFP" next to "Sort \& Slice outperformed advanced substructure selection schemes". Left, since both readings are acceptable.
- Long author sentences left as written (over 50 words): "PDVs typically consist of hundreds of properties including constitutional descriptors", "\citet{Kruger2012} rejected normality", "However, there are alternative molecular graph representations such as graphs with quantum 3D embeddings", "When the labels are corrupted or the important features are scarce".
- Chapter 1 (out of scope): its outline says Chapter 2 "closes with artifical noise injection, and evaluation metrics". The chapter now closes with NoiseInject and evaluation methodology, and "artifical" is a typo.

## Checker notes

Every OLD block was re-checked against thesis.tex on 2026-10-10 with python3 -I and str.count: all 37 original blocks occurred exactly once. Every ai-text edit has a "% TODO: manual review" or "% TODO: review, needs manual review" marker above its paragraph (for E24 a TODO(cite) line sits between the marker and the paragraph). Panel a of figures/molecular_representations.png was opened: teal arrows, integer identifiers, no orange, no bit vector, so E2 stands. \citep{Bemis1996} exists in refs.bib. sec:evaluation is not yet used anywhere, so E35 adds a new key and changes none.

- E10 deleted: "These molecules" after "activity cliffs" is not wrong, because the molecules in cliffs are what is meant. That is taste on author text.
- E23 deleted: a caption that names $R^2$ without its direction is not wrong. Adding "(higher is better)" is taste on author text.
- E1 fixed: "It asked how the representation, the model and the type of noise each affect the result" described all of Chapter 1, which also asks the second-half questions about the hybrid. It now says "Its first four questions ask".
- E17 fixed: "Each chapter states in its methods whether the scaffold keeps stereochemistry" was unverified for Chapters 4, 6 and 7. It now names only Chapters 3 and 5, which do state it ("chirality-blind Bemis--Murcko scaffold"; "here keeping stereochemistry").
- E32 fixed: "Three set flip rates" was an invented grouping, since all six strategies set flip rates. The list is now split in two sentences with the original names.
- For the author: the hERG/Caco-2/LogD line pointed to E24; the "three assay datasets of Chapter 3" change is E25.
- Kept as is: E2-E9, E11-E16, E18-E22, E24-E31, E33-E37. No new number appears in any NEW block. No NEW sentence is over 30 words. No banned phrasing was found.
- Not changed, but noted: E19, E20 and E11 make $K_\mathrm{i}$ roman in Chapter 2 only, while about 200 uses elsewhere stay $K_i$. Do this only together with the macro listed in For the author, or the thesis will be inconsistent.

