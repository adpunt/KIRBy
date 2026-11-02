# Plan: Chapter 4 (chap:flaws, Limitations of the Similarity Assumption), thesis.tex

Rules applied. A paragraph counts as AI-written only when a "% TODO: manual review" or "% TODO: review, needs manual review" line sits directly above it, or when such a line above a block says it covers "the block" (the ties block that starts "We counted these ties" and the distance block that starts "We measured four things"). Edits to unmarked paragraphs inside those two blocks add "% TODO: manual review" so the marker sits directly above. Every other paragraph and caption is treated as the author's text. Every number in NEW text already appears in thesis.tex, except "off in every molecule" in E5, which is a fact read from scratchpad/thesis_code_todo/constant/constant_counts.json (constant_off = 105, constant_on = 0 for herg_ki/ecfp4) and adds no number.

## Edits

### E1 — Chapter opening: SPP wording, AC defined, chapter question
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 21 (the SPP covers biological activity, which is what this chapter studies). "AC" is used five times in this chapter and is never defined anywhere in the thesis or in main.tex's abbreviation list. The last sentence asked whether methods satisfy the SPP "insofar as they do not", which assumes the answer.
- todo_line: % TODO: manual review -- edited your text: the SPP now covers biological activity, AC is defined at first use, and the last sentence no longer assumes its answer.
OLD:
```latex
The previous chapters have been focused on the impact of label noise. Before we delve into techniques that are meant to train on, identify, and mitigate such label noise, we first focus on a tangential point, molecular similarity. All of the aforementioned noise-adjacent methods, as well as the QSAR modeling itself, rest on the similar property principle (SPP; Section~\ref{sec:mol_similarity}), which posits that structurally similar compounds display similar chemical properties \citep{Johnson1990}. This serves as the conceptual foundation for QSAR modeling. The concept of molecular similarity is also used in various contexts to compare neighboring molecules. This comes into play when identifying noisy molecules that in theory stick out from their neighbors. There are some important exceptions to the SPP, namely activity cliffs \citep{Stumpfe2012, vanTilborg2022}. Furthermore, there is no single way to define similarity between compounds. This is where the principle begins to break down in practice. When defining similarity between two molecules, we typically rely on some pairwise mathematical function, or coefficient, that takes in two molecular representations as input. Two factors vary here, the representation and the coefficient. \citet{Bajusz2015} compared eight coefficients and found Tanimoto, Dice, Cosine and Soergel the best and largely equivalent, while similarities derived from Euclidean and Manhattan distances were not recommended on their own. In this chapter we ask if the standard methods of comparing molecules satisfy the SPP, insofar as they do not sufficiently quantify how similar two molecules actually are.
```
NEW:
```latex
% TODO: manual review -- edited your text: the SPP now covers biological activity, AC is defined at first use, and the last sentence no longer assumes its answer.
The previous chapters have been focused on the impact of label noise. Before we delve into techniques that are meant to train on, identify, and mitigate such label noise, we first focus on a tangential point, molecular similarity. All of the aforementioned noise-adjacent methods, as well as the QSAR modeling itself, rest on the similar property principle (SPP; Section~\ref{sec:mol_similarity}), which posits that structurally similar compounds display similar properties, including biological activity \citep{Johnson1990}. This serves as the conceptual foundation for QSAR modeling. The concept of molecular similarity is also used in various contexts to compare neighboring molecules. This comes into play when identifying noisy molecules that in theory stick out from their neighbors. There are some important exceptions to the SPP, namely activity cliffs (ACs) \citep{Stumpfe2012, vanTilborg2022}. Furthermore, there is no single way to define similarity between compounds. This is where the principle begins to break down in practice. When defining similarity between two molecules, we typically rely on some pairwise mathematical function, or coefficient, that takes in two molecular representations as input. Two factors vary here, the representation and the coefficient. \citet{Bajusz2015} compared eight coefficients and found Tanimoto, Dice, Cosine and Soergel the best and largely equivalent, while similarities derived from Euclidean and Manhattan distances were not recommended on their own. In this chapter we ask whether the standard methods of comparing molecules quantify how similar two molecules are well enough for the SPP to hold.
```

### E2 — Fingerprints and stereochemistry: a default, not a property of the fingerprint
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 22. ECFP can encode stereochemistry when asked; the chapter itself says "In this research, ECFP4 is built with stereochemistry off".
- todo_line: % TODO: manual review -- edited your text: "do not encode stereochemistry" became "are usually computed without stereochemistry", because ECFP and similar fingerprints can include it when asked.
OLD:
```latex
General-purpose molecular representations are designed to work across chemical domains. Consequently, when considering data from a specific domain, many of their features show little to no variance. Some representations assign separate molecules identical feature vectors. Since common fingerprints do not encode stereochemistry \citep{orsi2024map4c}, they are often unable to differentiate stereoisomers,
```
NEW:
```latex
% TODO: manual review -- edited your text: "do not encode stereochemistry" became "are usually computed without stereochemistry", because ECFP and similar fingerprints can include it when asked.
General-purpose molecular representations are designed to work across chemical domains. Consequently, when considering data from a specific domain, many of their features show little to no variance. Some representations assign separate molecules identical feature vectors. Since common fingerprints are usually computed without stereochemistry \citep{orsi2024map4c}, they are often unable to differentiate stereoisomers,
```

### E3 — Second person removed (section lead-in)
- file: thesis.tex
- kind: author-text
- why: Style guide §2 (no "you"/"your" in the body; none of the three references uses it).
- todo_line: % TODO: manual review -- edited your text: "your dataset" became "a dataset" (no second person).
OLD:
```latex
General-purpose representations are built to span a broad range of chemical domains. When your dataset covers only a small region of the chemical space,
```
NEW:
```latex
% TODO: manual review -- edited your text: "your dataset" became "a dataset" (no second person).
General-purpose representations are built to span a broad range of chemical domains. When a dataset covers only a small region of the chemical space,
```

### E4 — MDR1–MDCK, the two microsome clearance datasets and their Biogen source defined at first use
- file: thesis.tex
- kind: author-text
- why: Chemistry guide item 16 and §3 "Caco-2 and MDR1–MDCK". This paragraph is the first place in the thesis where MDR1–MDCK, HLM and RLM appear, and the later text calls them "the three Biogen datasets" before Biogen is named (Biogen and \citep{fang2023} first appear in Chapter 5). Ch5's tables give the labels as log10 efflux ratio and log10 CL_int.
- todo_line: % TODO: manual review -- edited your text: added one sentence saying what MDR1--MDCK efflux and the two microsome clearance datasets measure and that they come from Biogen, since this is their first use in the thesis.
OLD:
```latex
To investigate similarity estimation, we first look at a simple measure: how many features vary within a dataset.
```
NEW:
```latex
% TODO: manual review -- edited your text: added one sentence saying what MDR1--MDCK efflux and the two microsome clearance datasets measure and that they come from Biogen, since this is their first use in the thesis.
To investigate similarity estimation, we first look at a simple measure: how many features vary within a dataset.
```
and, in the same paragraph:
OLD:
```latex
rat liver microsome (RLM) clearance and the Lhasa hERG blocker training set.
```
NEW:
```latex
rat liver microsome (RLM) clearance and the Lhasa hERG blocker training set. MDR1--MDCK efflux and the two clearance datasets come from Biogen \citep{fang2023}. They give the efflux ratio across MDCK cells that express human P-glycoprotein (MDR1), and the intrinsic clearance of each molecule in human or rat liver microsomes.
```

### E5 — Near-constant features are amplified only after standardization; contraction; a feature cannot "define" an AC
- file: thesis.tex
- kind: author-text
- why: "near-constant features are amplified" is true only after each feature is scaled to unit variance (the chapter says so later, "After standardization, uncommon bits take on larger values"). An AC is a pair of molecules (chemistry guide §3), so a feature can mark the change behind one but not define it. "That's" is a contraction (style guide §2). "It does not use the 105 constant ECFP4 bits" holds only because all 105 are off in every molecule, which the reader cannot see (checked: constant_off = 105, constant_on = 0).
- todo_line: % TODO: manual review -- edited your text: said the 105 constant bits are all off, that near-constant features are amplified only after standardization, removed the contraction, and said a feature can mark the change behind an AC rather than define one.
OLD:
```latex
These constant counts are sometimes filtered out by the distance. Tanimoto similarity (Equation~\ref{eq:tanimoto}; the distance is $1-T$) ignores bits off in both molecules, but a bit on in every molecule raises every molecule's measured similarity. It does not use the 105 constant ECFP4 bits on hERG $K_i$. Constant features add zero to a Euclidean or Manhattan distance; however, near-constant features are amplified. That's great behavior if those features define an AC. If not, they are contributing disproportionately to the measure of similarity.
```
NEW:
```latex
% TODO: manual review -- edited your text: said the 105 constant bits are all off, that near-constant features are amplified only after standardization, removed the contraction, and said a feature can mark the change behind an AC rather than define one.
These constant counts are sometimes filtered out by the distance. Tanimoto similarity (Equation~\ref{eq:tanimoto}; the distance is $1-T$) ignores bits off in both molecules, but a bit on in every molecule raises every molecule's measured similarity. It does not use the 105 constant ECFP4 bits on hERG $K_i$, which are off in every molecule. Constant features add zero to a Euclidean or Manhattan distance; however, near-constant features are amplified once each feature is standardized to unit variance. That is useful if such a feature marks the structural change behind an AC. If not, it contributes disproportionately to the measure of similarity.
```

### E7 — Ties paragraph: section reference by label, InChIKey glossed, long sentence split
- file: thesis.tex
- kind: ai-text
- why: Style guide gap item 42 (quoted section title "Augmentation" becomes Section~\ref). The duplicate check only works because a standard InChIKey differs between stereoisomers; one clause says so (chemistry guide §3, SMILES and identity). The first sentence would exceed 30 words with the reference.
- todo_line: keep existing
OLD:
```latex
We counted these ties on hERG $K_i$, LogD and Caco-2, for each dataset's base representations and augmentation features (small feature sets added to the hybrid after it is built, Chapter~\ref{chap:method}, ``Augmentation''). Two molecules are tied when a representation gives them exactly the same feature vector. No tie came from one compound listed twice (checked by InChIKey) or from a row of zeros,
```
NEW:
```latex
We counted these ties on hERG $K_i$, LogD and Caco-2, for each dataset's base representations and augmentation features. Augmentation features are small feature sets added to the hybrid after it is built (Section~\ref{sec:method:augmentation}). Two molecules are tied when a representation gives them exactly the same feature vector. No tie came from one compound listed twice (checked by InChIKey, which differs between stereoisomers) or from a row of zeros,
```

### E8 — Ties paragraph 2: two long sentences split
- file: thesis.tex
- kind: ai-text
- why: Part of the block the TODO above "We counted these ties" calls a rewrite; two sentences of 31 and 39 words. No wording or number changed.
- todo_line: % TODO: manual review
OLD:
```latex
Apart from the MACCS keys, most tied molecules are stereoisomers. Atom pairs, PDV, Mordred and SMI-TED tie only stereoisomers, and ECFP4, Avalon and topological torsions add a few molecules that differ in more than their stereochemistry. On hERG $K_i$, ECFP4, atom pairs, PDV, Mordred and SMI-TED tie the same 56 molecules, so joining any of these five representations leaves all 56 tied. The WL graph kernel also ties 49 hERG $K_i$ and 26 Caco-2 molecules that are not stereoisomers, and the MACCS keys tie the most molecules on every dataset, mostly not stereoisomers. The 3D augmentation features and the learned embeddings ChemBERT, GROVER, MHG-GNN, MoLFormer and GraphMVP tie none, and joining all of a dataset's representations leaves no molecule tied.
```
NEW:
```latex
% TODO: manual review
Apart from the MACCS keys, most tied molecules are stereoisomers. Atom pairs, PDV, Mordred and SMI-TED tie only stereoisomers, and ECFP4, Avalon and topological torsions add a few molecules that differ in more than their stereochemistry. On hERG $K_i$, ECFP4, atom pairs, PDV, Mordred and SMI-TED tie the same 56 molecules, so joining any of these five representations leaves all 56 tied. The WL graph kernel also ties 49 hERG $K_i$ and 26 Caco-2 molecules that are not stereoisomers. The MACCS keys tie the most molecules on every dataset, mostly not stereoisomers. The 3D augmentation features and the learned embeddings ChemBERT, GROVER, MHG-GNN, MoLFormer and GraphMVP tie none. Joining all of a dataset's representations leaves no molecule tied.
```

### E9 — Stereodescriptors in italics
- file: thesis.tex
- kind: author-text-trivial
- why: Chemistry guide §2 and item 27: R and S are italic by IUPAC convention.
- todo_line: none (trivial)
OLD:
```latex
(R in A, S in B)
```
NEW:
```latex
(\textit{R} in A, \textit{S} in B)
```

### E10 — Second person removed (similarity coefficient paragraph)
- file: thesis.tex
- kind: author-text
- why: Style guide §2 (no "you" in the body).
- todo_line: % TODO: manual review -- edited your text: "cannot tell you which features" became "cannot tell which features" (no second person).
OLD:
```latex
A similarity coefficient cannot tell you which features matter to the prediction task at hand.
```
NEW:
```latex
% TODO: manual review -- edited your text: "cannot tell you which features" became "cannot tell which features" (no second person).
A similarity coefficient cannot tell which features matter to the prediction task at hand.
```

### E11 — "We measured four things": split a 48-word sentence; HLM not re-expanded, and "not used elsewhere" corrected
- file: thesis.tex
- kind: ai-text
- why: Writing rule (no sentence over 30 words). HLM clearance is already defined in "Constant and near-constant features", so "not used elsewhere in this thesis" is wrong; it appears only in this chapter (checked: every non-comment use of HLM is in Chapter 4). After E4, Biogen has been named, so "a third Biogen dataset" reads.
- todo_line: keep existing
OLD:
```latex
We measured four things on our datasets: how many activity cliffs a representation finds, whether the distance between two molecules tracks the difference in their labels, whether two representations agree on each molecule's nearest neighbors, and whether a fingerprint distance agrees with a distance based on shared structure. The datasets are the six small-molecule regression datasets of Table~\ref{tab:dataset_reps} (Chapter~\ref{chap:method}) and human liver microsome (HLM) clearance, a third dataset from Biogen not used elsewhere in this thesis.
```
NEW:
```latex
We measured four things on our datasets. The first is how many activity cliffs a representation finds, and the second is whether the distance between two molecules tracks the difference in their labels. The third is whether two representations agree on each molecule's nearest neighbors. The fourth is whether a fingerprint distance agrees with a distance based on shared structure. The datasets are the six small-molecule regression datasets of Table~\ref{tab:dataset_reps} (Chapter~\ref{chap:method}) and HLM clearance, a third Biogen dataset used only in this chapter.
```

### E12 — Activity cliffs: a cliff is a pair; the term is borrowed for labels that are not activities
- file: thesis.tex
- kind: ai-text
- why: Chemistry guide §3 ("A cliff is a property of a pair ... define 'cliff molecule' as a molecule in at least one cliff pair") and item 18 (cliffs on QM9's HOMO–LUMO gap: say once that the term is borrowed). The original "a molecule is a cliff if a similar molecule has a label that differs" computes exactly "a molecule in at least one cliff pair", so no number changes. Also splits a 38-word sentence.
- todo_line: % TODO: manual review
OLD:
```latex
Here we use one. Two molecules are similar if their ECFP4 Tanimoto similarity is at least 0.55 \citep{Hu2015}, and a molecule is a cliff if a similar molecule has a label that differs from its own by more than a set gap. We chose the gap on each dataset so that 15\% of the molecules are cliffs, so the gap follows the spread of the labels (Table~\ref{tab:flaws:distance_summary}). On hERG $K_i$ the gap is 1.76 p$K_i$, and 211 of the 1,412 molecules are cliffs.
```
NEW:
```latex
Here we use one. Two molecules are similar if their ECFP4 Tanimoto similarity is at least 0.55 \citep{Hu2015}. A similar pair whose labels differ by more than a set gap is a cliff, and a cliff molecule is one that belongs to at least one cliff. We chose the gap on each dataset so that 15\% of the molecules are cliff molecules, so the gap follows the spread of the labels (Table~\ref{tab:flaws:distance_summary}). We keep the term activity cliff where the label is not a biological activity, such as the QM9 HOMO--LUMO gap. On hERG $K_i$ the gap is 1.76 p$K_i$, and 211 of the 1,412 molecules are cliff molecules.
```
Put the todo_line directly above the paragraph, i.e. replace
OLD:
```latex
\paragraph{Activity cliffs.} Chapter~\ref{chap:background} gives the MoleculeACE definition
```
NEW:
```latex
% TODO: manual review
\paragraph{Activity cliffs.} Chapter~\ref{chap:background} gives the MoleculeACE definition
```

### E13 — Distance and label difference: two long sentences split
- file: thesis.tex
- kind: ai-text
- why: Part of the distance block; sentences of 41 and 35 words. Same content.
- todo_line: % TODO: manual review
OLD:
```latex
\paragraph{Distance and label difference.} We asked whether distance tracks label difference in two ways. Over all pairs of molecules, we took the Spearman correlation between the distance of a pair and the absolute difference of its labels, a one-number summary of the neighborhood-behavior plot of \citet{Patterson1996}; datasets above 2,000 molecules were sampled down to 2,000. Over each molecule's closest pairs only, we predicted each label as the mean label of the molecule's five nearest neighbors, taken from other scaffold groups, and took the Spearman correlation between predicted and measured labels.
```
NEW:
```latex
% TODO: manual review
\paragraph{Distance and label difference.} We asked whether distance tracks label difference in two ways. The first uses all pairs of molecules: we took the Spearman correlation between the distance of a pair and the absolute difference of its labels. This is a one-number summary of the neighborhood-behavior plot of \citet{Patterson1996}. Datasets above 2,000 molecules were sampled down to 2,000. The second uses only each molecule's closest pairs. We predicted each label as the mean label of the molecule's five nearest neighbors, taken from other scaffold groups, and took the Spearman correlation between predicted and measured labels.
```

### E14 — hERG pair paragraph: one 40-word sentence split
- file: thesis.tex
- kind: ai-text
- why: Part of the distance block; writing rule. Numbers unchanged.
- todo_line: % TODO: manual review
OLD:
```latex
Figure~\ref{fig:flaws:herg_pairs} shows why the two differ. Of the 996,166 hERG $K_i$ pairs, 93\% have an ECFP4 Tanimoto similarity below 0.20, and only 1.1\% reach 0.55. Over all pairs the median label difference is 0.81 p$K_i$, but above a similarity of about 0.4 it is 0.20 to 0.30 p$K_i$, although 8.3\% of the 774 pairs at 0.8 or above still differ by more than 1 p$K_i$.
```
NEW:
```latex
% TODO: manual review
Figure~\ref{fig:flaws:herg_pairs} shows why the two differ. Of the 996,166 hERG $K_i$ pairs, 93\% have an ECFP4 Tanimoto similarity below 0.20, and only 1.1\% reach 0.55. Over all pairs the median label difference is 0.81 p$K_i$, but above a similarity of about 0.4 it is 0.20 to 0.30 p$K_i$. Even so, 8.3\% of the 774 pairs at 0.8 or above still differ by more than 1 p$K_i$.
```

### E15 — Agreement on neighbors: one 44-word sentence split
- file: thesis.tex
- kind: ai-text
- why: Part of the distance block; writing rule. Content unchanged.
- todo_line: % TODO: manual review
OLD:
```latex
\paragraph{Agreement on neighbors.} We compared the five nearest neighbors of each molecule, from other scaffold groups, under ECFP4, SMI-TED and Mordred (PDV on Caco-2, where Mordred is not a base representation; on HLM clearance ECFP4 is not a base representation and its stored features were used).
```
NEW:
```latex
% TODO: manual review
\paragraph{Agreement on neighbors.} We compared the five nearest neighbors of each molecule, from other scaffold groups, under ECFP4, SMI-TED and Mordred. PDV took Mordred's place on Caco-2, where Mordred is not a base representation. On HLM clearance ECFP4 is not a base representation, so its stored features were used.
```

### E16 — Figure caption (hERG pairs): panel (a) is a similarity, not a distance
- file: thesis.tex
- kind: caption
- why: The caption says the difference is plotted "against distance", but panel (a) is ECFP4 Tanimoto similarity, as the caption's own panel description says. Style guide §16 (caption names what is shown).
- todo_line: % TODO: manual review -- edited your text: "against distance" became "against similarity or distance", because panel (a) shows Tanimoto similarity.
OLD:
```latex
\caption[Absolute p$K_i$ difference against pair distance on hERG $K_i$.]{Absolute p$K_i$ difference against distance for all pairs of hERG $K_i$ molecules,
```
NEW:
```latex
% TODO: manual review -- edited your text: "against distance" became "against similarity or distance", because panel (a) shows Tanimoto similarity.
\caption[Absolute p$K_i$ difference against pair distance on hERG $K_i$.]{Absolute p$K_i$ difference against similarity or distance for all pairs of hERG $K_i$ molecules,
```

### E18 — Later distances: cosine distance is not a metric either; long sentence split
- file: thesis.tex
- kind: ai-text
- why: The paragraph says Minkowski with p = 0.5 "is a dissimilarity rather than a metric", which implies cosine distance is a metric. It is not: 1 − cos also breaks the triangle inequality. The 36-word sentence is split.
- todo_line: keep existing
OLD:
```latex
Chapter~\ref{chap:similarity} compares Manhattan, cosine and Minkowski distance with $p=0.5$, $d(x,y)=\left(\sum_i |x_i-y_i|^{0.5}\right)^{2}$, on datasets of up to 5,000 molecules, and Manhattan and cosine on larger ones. The Minkowski distance with $p=0.5$ is a dissimilarity rather than a metric, because it breaks the triangle inequality.
```
NEW:
```latex
Chapter~\ref{chap:similarity} compares Manhattan, cosine and Minkowski distance with $p=0.5$, $d(x,y)=\left(\sum_i |x_i-y_i|^{0.5}\right)^{2}$. It uses all three on datasets of up to 5,000 molecules, and Manhattan and cosine on larger ones. Like cosine distance, the Minkowski distance with $p=0.5$ is a dissimilarity rather than a metric, because it breaks the triangle inequality.
```

### E19 — Contraction "We've"
- file: thesis.tex
- kind: author-text
- why: Style guide §2 (no contractions).
- todo_line: % TODO: manual review -- edited your text: "We've" became "We have" (no contractions).
OLD:
```latex
We've identified some issues within the representations and similarities computed over them.
```
NEW:
```latex
% TODO: manual review -- edited your text: "We've" became "We have" (no contractions).
We have identified some issues within the representations and similarities computed over them.
```

### E20 — Small sweep, opening: Chapter 3 number updated; 41-word sentence split
- file: thesis.tex
- kind: ai-text
- why: The existing "TODO: Ch3 update" asks for this check. "Representation at most 11.2%" no longer appears in Chapter 3. The updated Table tab:variance gives representation 8.1, 7.0 and 7.9% of the variance in \aucnorm{} on QM9, so the largest is 8.1%. Model family is the largest term in all three \aucnorm{} rows, so that half stands.
- todo_line: keep existing
OLD:
```latex
In Chapter~\ref{chap:robustness}, model family explained the largest share of the variance in robustness to label noise, and representation at most 11.2\%, so we asked whether training methods built for noisy labels improve robustness, and whether the representation still matters under them.
```
NEW:
```latex
In Chapter~\ref{chap:robustness}, model family explained the largest share of the variance in robustness to label noise, and representation at most 8.1\% on QM9 (Table~\ref{tab:variance}). We therefore asked whether training methods built for noisy labels improve robustness, and whether the representation still matters under them.
```

### E21 — Small sweep, setup: two 34-word sentences split
- file: thesis.tex
- kind: ai-text
- why: Writing rule. Same content and numbers.
- todo_line: keep existing
OLD:
```latex
The study tested 12 published noise-handling methods on the QM9 HOMO--LUMO gap (10,000 molecules, scaffold splits), adding Gaussian noise with $\sigma$ from 0 to 1.0 in steps of 0.1 to the training labels only. The representations were ECFP4, Sort \& Slice \citep{sns}, PDV, canonical SMILES and randomized SMILES. Depending on the method, a configuration could add the model's own uncertainty, a distance between molecules (Tanimoto, Euclidean, Manhattan, cosine or Mahalanobis), an alternative GP kernel or an alternative loss function, giving 221 configurations.
```
NEW:
```latex
The study tested 12 published noise-handling methods on the QM9 HOMO--LUMO gap (10,000 molecules, scaffold splits). It added Gaussian noise with $\sigma$ from 0 to 1.0 in steps of 0.1 to the training labels only. The representations were ECFP4, Sort \& Slice \citep{sns}, PDV, canonical SMILES and randomized SMILES. Depending on the method, a configuration could also add the model's own uncertainty, a distance between molecules, an alternative GP kernel or an alternative loss function. The distances were Tanimoto, Euclidean, Manhattan, cosine and Mahalanobis. In all there were 221 configurations.
```

### E22 — Small sweep, metric: say what retention is and why it was used (author's own TODO)
- file: thesis.tex
- kind: ai-text
- why: The author's TODO above this subsection asks for "a quick explanation of the noise metric I'm using (explain it predated auc_norm) and is more strongly correlated with R2 ... but for the purposes of this experiment its a good proxy". Chapter 3's Evaluation defines the retention ratio inside Equation eq:aucnorm and says \aucnorm{} is "less directly correlated to clean-label R^2". The quoted title ``Evaluation'' becomes an equation reference, since that subsection has no label (style guide gap item 42). Also splits a 33-word sentence.
- todo_line: keep existing
OLD:
```latex
We measured noise robustness by retention, $R^2$ under noise divided by $R^2$ with clean labels, as a percentage, which is the retention ratio of Chapter~\ref{chap:robustness} (``Evaluation''). Methods that used a distance did worse than those that did not, and in an analysis of variance over the adapted components, whether a distance was used explained the most variance in $R^2$.
```
NEW:
```latex
We measured noise robustness by retention, $R^2$ under noise divided by $R^2$ with clean labels, as a percentage. This is the retention ratio of Chapter~\ref{chap:robustness} (Equation~\ref{eq:aucnorm}). The study predates \aucnorm{}, the area under the retention curve that Chapter~\ref{chap:robustness} uses. Retention is more directly correlated with clean $R^2$ than \aucnorm{} is, so it separates robustness from accuracy less well. It is still a good enough proxy for comparing methods with and without a distance. Methods that used a distance did worse than those that did not. In an analysis of variance over the adapted components, whether a distance was used explained the most variance in $R^2$.
```

### E23 — Small sweep, scope: stale chapter pointer and overclaim
- file: thesis.tex
- kind: ai-text
- why: Style guide gap item 43: the later sweep is in Chapter 7 (chap:noise), not Chapter 6. Chapter 7 says itself that it is "a new implementation of the same twelve, not a rerun of the earlier study", running "on many datasets instead of QM9 alone" and "on hybrids as well as single representations" (Section sec:noise:res:sweep). So "repeats the question" is too strong, as the long TODO above this paragraph says.
- todo_line: keep existing
OLD:
```latex
The study used one dataset, one property and one noise condition (Gaussian noise on every training label), and Chapter~\ref{chap:similarity} repeats the question on more datasets and under other noise conditions (Section~\ref{sec:noise_injection}). It also added the distance step to methods designed without one, so poor adaptation is an alternative explanation.
```
NEW:
```latex
The study used one dataset, one property and one noise condition (Gaussian noise on every training label). It also added the distance step to methods designed without one, so poor adaptation is an alternative explanation. Chapter~\ref{chap:noise} runs a new implementation of the same twelve methods on more datasets and on hybrids (Section~\ref{sec:noise:res:sweep}). That chapter also uses the other noise conditions of Section~\ref{sec:noise_injection}.
```

### E24 — Neighbor detection: definitions now live in Chapter 7
- file: thesis.tex
- kind: author-text
- why: Wrong cross-reference. The scores, the three actions and the recovery rate are defined in Chapter 7 (Sections sec:noise:detection and sec:noise:mitigation, Equation eq:recovery), not Chapter 6, which has no noise methods (style guide gap items 42 and 43; Chapter 7 carries "TODO: check every \ref{chap:similarity} outside Chapter 6 that is about label noise").
- todo_line: % TODO: manual review -- edited your text: the pointer went to a quoted Chapter 6 title, but these definitions are now in Chapter 7, so it names its sections and the recovery equation.
OLD:
```latex
The neighbor-based noise scores, sample removal, sample weighting, label smoothing and the recovery rate are defined in Chapter~\ref{chap:similarity}, ``Noise mitigation methods''.
```
NEW:
```latex
% TODO: manual review -- edited your text: the pointer went to a quoted Chapter 6 title, but these definitions are now in Chapter 7, so it names its sections and the recovery equation.
The neighbor-based noise scores, sample removal, sample weighting, label smoothing and the recovery rate are defined in Chapter~\ref{chap:noise}. The scores are in Section~\ref{sec:noise:detection}, the three actions in Section~\ref{sec:noise:mitigation} and the recovery rate in Equation~\ref{eq:recovery}.
```

### E26 — Neighbor detection: stale pointer to the sample-removal study
- file: thesis.tex
- kind: author-text
- why: Wrong cross-reference: deletion experiments are in Chapter 7 (for example Section sec:noise:res:action), and Chapter 6 has none (style guide gap item 43).
- todo_line: % TODO: manual review -- edited your text: "Chapter~\ref{chap:similarity}" became "Chapter~\ref{chap:noise}", where the sample-removal work now is.
OLD:
```latex
The exploratory study also used neighbors to clean labels,
```
NEW:
```latex
% TODO: manual review -- edited your text: "Chapter~\ref{chap:similarity}" became "Chapter~\ref{chap:noise}", where the sample-removal work now is.
The exploratory study also used neighbors to clean labels,
```
and, at the end of the same paragraph:
OLD:
```latex
This study is separate from the sample-removal study of Chapter~\ref{chap:similarity}.
```
NEW:
```latex
This study is separate from the sample-removal study of Chapter~\ref{chap:noise}.
```

### E27 — Early hybrid paragraph: stale pointer
- file: thesis.tex
- kind: ai-text
- why: Style guide gap item 43. Neighbor detectors in a hybrid under non-Gaussian noise are in Chapter 7.
- todo_line: keep existing
OLD:
```latex
Chapter~\ref{chap:similarity} repeats the measurement in a hybrid and under noise conditions other than Gaussian.
```
NEW:
```latex
Chapter~\ref{chap:noise} repeats the measurement in a hybrid and under noise conditions other than Gaussian.
```

### E29 — Conclusions, noise paragraph: stale pointer; 33-word sentence split
- file: thesis.tex
- kind: ai-text
- why: Style guide gap item 43 (the other noise conditions are tested in Chapter 7, not Chapter 6); writing rule.
- todo_line: keep existing
OLD:
```latex
In the small sweep of published noise-handling methods, configurations that used a distance did worst under label noise (Section~\ref{sec:lnl_motivation}), and judging a label by its neighbors' labels usually made the model worse (Section~\ref{sec:flaws:neighbors}). It did not always fail, since one neighbor-based configuration gave the best recovery at the higher noise level. That noise was Gaussian on every training label, so each molecule's neighbors were noisy too; Chapter~\ref{chap:similarity} turns to the other noise conditions of Chapter~\ref{chap:background}.
```
NEW:
```latex
In the small sweep of published noise-handling methods, configurations that used a distance did worst under label noise (Section~\ref{sec:lnl_motivation}). Judging a label by its neighbors' labels usually made the model worse (Section~\ref{sec:flaws:neighbors}). It did not always fail, since one neighbor-based configuration gave the best recovery at the higher noise level. That noise was Gaussian on every training label, so each molecule's neighbors were noisy too; Chapter~\ref{chap:noise} turns to the other noise conditions of Chapter~\ref{chap:background}.
```

### E30 — Conclusions, last paragraph: point forward with Chapter 5's question; quoted title becomes Section~\ref
- file: thesis.tex
- kind: ai-text
- why: Style guide §1 (V and B end each Conclusions by naming the next chapter's question). The question is taken word for word from Chapter 5's own Conclusions ("how the best set of features can be selected for a given prediction task"). The quoted title "The search space" has the label sec:method:searchspace (style guide gap item 42).
- todo_line: keep existing
OLD:
```latex
These flaws are where the thesis began (Chapter~\ref{chap:intro}), and Chapter~\ref{chap:method} builds the method they led to, the adaptive hybrid representation. The settings that method searches over are listed in Chapter~\ref{chap:method}, ``The search space''.
```
NEW:
```latex
These flaws are where the thesis began (Chapter~\ref{chap:intro}). Chapter~\ref{chap:method} builds the method they led to, the adaptive hybrid representation, and asks how the best set of features can be selected for a given prediction task. The settings that method searches over are listed in Section~\ref{sec:method:searchspace}.
```

## For the author (not changed)

- K_i typography. This chapter uses `$K_i$` 47 times, `p$K_i$` 19 times of those, and no other form, so it is consistent inside itself. The chemistry guide (§2, item 10) recommends `$K_\mathrm{i}$` through one macro across the whole thesis (134 uses of `$K_i$`). Do it as one global replace, not chapter by chapter.
- "LogD (log units)" and "Caco-2 (log units)" in Table tab:flaws:ties. The chemistry guide (item 26) suggests "dimensionless". Left alone: the column holds RMSE and label spread of a logged quantity, for which "log units" is correct.
- Figure fig:flaws:stereo_pair: "(R in A, S in B)". If this carbon is each molecule's only stereocentre, the two are enantiomers; if they have others, they are epimers. The caption could say which (chemistry guide item 27). Not checked.
- "\citet{Huber2026} found that fingerprints which store raw counts are more effective at defining similarity than folded fingerprints". Count against binary and folded against unfolded are two separate choices. Check which comparison the paper made.
- "Regularization alone retained less, 73.0\% for Mixup and 74.0\% for sharpness-aware minimization." Neither Mixup nor sharpness-aware minimization is among the twelve methods listed in the long TODO comment above "The study used one dataset". Say whether they were extra baselines.
- The author's TODO above "A small sweep of published noise-handling methods" asks for "a table of the methods ... [that] clearly mark[s] how similarity/distance is used". Not planned, because no tables are created in this pass.
- Rat microsomal clearance is "rat liver microsome (RLM) clearance" in this chapter and "Rat microsomal clearance" in Chapter 5's tables (for example "Rat microsomal clearance \citep{fang2023, polaris}"). Pick one name.
- Requirements table tab:flaws:requirements and "(Chapter~\ref{chap:method}, ``Combining representations'')" in the Conclusions cite sections by quoted title. "Combining representations" and "Blueprints and their chemical interpretation" in Chapter 5 have no \label, so they cannot become Section~\ref without editing Chapter 5. The other rows could use sec:flaws:constant, sec:flaws:ties (a label that would have to be added to the subsection "Molecules with identical feature vectors"), sec:flaws:similarity, sec:flaws:consequences, sec:flaws:representations, sec:method:importance, sec:method:searchspace and sec:similarity:hybrid_similarity.
- Figure fig:constant_features caption does not define near-constant (one value in at least 95\% of molecules). Table tab:constant_features does. Not wrong, so not changed.
- Literature tense is mixed in the opening: "\citet{Coupry2022} learn", "\citet{Zheng2000} did choose ... They predict", and later "\citet{Martin2021} adapt ... They find". The references use past tense for one study (style guide §11).
- "ECFP4 is the worst offender." The style guide (§2) lists "the worst offender" as an informal phrase none of the references would print.
- "That jumps to 1,997 out of 2,048 near-constant features for the HOMO--LUMO gap on QM9." The count depends on the molecules, not the property, so "on QM9" is enough.
- "292 of MHG-GNN's 1,024 features are zero ..." starts a sentence with a numeral.
- "While binary fingerprints are compared with the Tanimoto coefficient (Chapter~\ref{chap:background}), we need distance metrics for continuous feature vectors." The list after it includes cosine distance, which is not a metric (see E18). "Distance measures" would be exact.
- Activity cliffs: "AC" is missing from main.tex's List of Abbreviations (style guide gap item 13).
- "Only the two baselines differed significantly from the rest." This reports Riniker and Landrum's own result, so it is kept (style guide §10).
- The long TODO above "The study used one dataset" says the later survey never adds a distance to the twelve methods, so the analysis of variance on the distance here has no counterpart in Chapter 7. A sentence saying so would explain why this section stays. Not added; the claim comes from a code comment and was not re-checked in this session.
- Chemistry guide item 7 (0.44 against 0.54 for Kramer 2012) affects Chapters 1 and 3. In this chapter, "differ by $0.44$ log units on average, with a standard deviation of $0.54$" is already the full form.
- "(Chapter~\ref{chap:background}, ``Molecular similarity and activity cliffs'')" and "(``Molecules with identical feature vectors'')" cite sections by quoted title. Section~\ref{sec:mol_similarity} exists; the ties subsection would need a new label.
- Table tab:flaws:distance_summary caption does not say that higher is better for either Spearman column.

## Checker notes

Checked 2026-10-10 against thesis.tex. All 29 remaining OLD blocks occur exactly once (python3 -I, str.count). Every number in NEW text appears in thesis.tex. No banned phrasing found. The only NEW sentence over 30 words is the author's own SPP sentence in E1 (34 words), where only "chemical properties" changed.

Changed:
- E1 fixed: dropped the new first sentence ("Chapter~\ref{chap:robustness} focused on ... relatively little of the variance"). "The previous chapters have been focused on the impact of label noise" is not wrong, so rewriting it was taste. The SPP wording, the ACs definition and the last sentence stay; the todo_line was shortened to match.
- E13 fixed: the new sentence joined the Patterson summary and the 2,000-molecule sampling with "and". They are now two sentences.
- E23 fixed: NEW said Chapter 7 runs the twelve methods on datasets "not including QM9". Chapter 7 now has QM9 results (the [N7-42] comment above Section sec:noise:res:sweep says "QM9 finals for censoring and grouped_shifted only"). Now reads "on more datasets and on hybrids".
- E6 deleted (label only needed by E28).
- E17 deleted: the distance-summary caption has no TODO marker directly above it, so it counts as the author's text, and a missing direction is not an error.
- E25 deleted: "(Chapter~\ref{chap:background}, ``Molecular similarity and activity cliffs'')" names an existing section correctly; changing it to Section~\ref is style.
- E28 deleted: same reason, the quoted title "Molecules with identical feature vectors" is correct.

Checked and kept:
- E5: "off in every molecule" confirmed in scratchpad/thesis_code_todo/constant/constant_counts.json (datasets/herg_ki/reps/ecfp4: constant 105, constant_on 0, constant_off 105).
- E20: Table tab:variance gives Representation 8.1, 7.0 and 7.9\% for \aucnorm{} on QM9; "11.2" appears nowhere else in thesis.tex.
- E22: Equation eq:aucnorm is the normalized area under the retention curve, and Chapter 3 says \aucnorm{} is "less directly correlated to clean-label $R^2$".
- E24, E26: sec:noise:detection, sec:noise:mitigation, eq:recovery and sec:noise:res:action are all in Chapter 7.
- E4: fang2023 is the Biogen citation used in Chapter 5; MDR1--MDCK efflux ratio and microsomal intrinsic clearance wording is correct.
- E7, E30: sec:method:augmentation and sec:method:searchspace exist.
- E3, E10 (second person) and E19 (contraction) kept as register errors of the same class as a contraction.

Added to "For the author": quoted section titles in the noise-detection paragraph and the Conclusions (former E25, E28), and the missing direction in the distance-summary caption (former E17).
