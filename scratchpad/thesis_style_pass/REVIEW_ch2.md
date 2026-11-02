# Review: Chapter 2 (chap:background), after the style pass

Compared with thesis.tex.before_style_apply_2026-10-10 by diff (Chapter 2 is backup lines 131-640, current lines 147-670) and read in full as it stands on 2026-10-10. her_text_check.py on Chapter 2 alone gives three hits: #1 chapter title (has its "title was" note above the \chapter line: false positive), #2 "(i.e.," to "(e.g.," in the neural-network paragraph (no note: V3), #3 "Here" to "where" after Equation dose_matching (no note: V8). Reading the diff found one more change to her text that fixes nothing: "count" to "counted" for Svensson2025 (V9). No paragraph of hers was newly given an AI marker. Every OLD block below was checked with python3 -I str.count to occur exactly once in thesis.tex. NEW text adds no number; every number in Chapter 2 is already in the backup thesis.tex (checked by script).

Checks that passed: every \ref in Chapter 2 resolves (18 labels, each defined once); every \label of the backup is still present, and \label{sec:evaluation} was added. The chapter opens by looking back to Chapter 1 and the Conclusions point to Chapter 3. Chapter and section titles are in sentence case. No "Results:" prefix and no Discussion section (a background chapter has no limitations paragraph to place). Chemistry guide items 1, 2, 3 and 15 are done in this chapter (ECFP chirality, 2D graphs and stereochemistry, 2D graphs hold no coordinates, Bemis--Murcko defined); item 4 is moot because the author removed that figure. Style guide items 17 (Overview filled) and 22 (sentence case) are done. The edited-your-text notes on the ECFP, 2D graph, David2020, GIN, Gauche, Kolmar, Kruger, Bentz and noise-injection paragraphs each fix something wrong, except the tense change in V9.

## Fixes

### V1 — Overview: wrong summary of Chapter 1's first four questions
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: Chapter 1's fourth question asks what label noise does to estimated uncertainty. The Overview says the four questions ask how representation, model and noise type "each affect the result", which leaves the uncertainty question out and does not say what "the result" is. The last sentence names NoiseInject only, but the section it points to now also covers the QM9 pipeline and has a new title.
OLD:
```latex
Its first four questions ask how the representation, the model and the type of noise each affect the result. This chapter gives the background the later chapters rely on. It first describes the molecular representations and the models we use, and then defines molecular similarity and activity cliffs. It then describes where label noise in molecular data comes from, how we inject it, and the methods for training with noisy labels. It ends with NoiseInject, the package that injects the noise and computes the metrics, and with how results are evaluated.
```
NEW:
```latex
Its first four questions ask how the representation, the model and the type of noise affect predictive performance under label noise, and what label noise does to estimated uncertainty. This chapter gives the background the later chapters rely on. It first describes the molecular representations and the models we use, and then defines molecular similarity and activity cliffs. It then describes where label noise in molecular data comes from, how we inject it, and the methods for learning with noisy labels. It ends with how the noise conditions are implemented, including the NoiseInject package, and with how results are evaluated.
```

### V2 — Neural-network paragraph: "(i.e.," changed to "(e.g.," with no note
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "(i.e., fingerprints, descriptors)" changed to "(e.g., fingerprints, descriptors)", because fingerprints and descriptors are examples of feature vectors, not the whole set; "a NN" changed to "an NN", as two sentences earlier.
- why: her_text_check hit #2. The change fixes a real misuse, so it stays and gets the note. The article fix is trivial and rides in the same note.
OLD:
```latex
Neural networks (NNs), a very popular family of models, can also be used for QSAR, although with mixed results.
```
NEW:
```latex
% TODO: manual review -- edited your text: "(i.e., fingerprints, descriptors)" changed to "(e.g., fingerprints, descriptors)", because fingerprints and descriptors are examples of feature vectors, not the whole set; "a NN" changed to "an NN", as two sentences earlier.
Neural networks (NNs), a very popular family of models, can also be used for QSAR, although with mixed results.
```

### V3 — Neural-network paragraph: wrong article
- file: thesis.tex
- kind: author-text (trivial, covered by V2's note)
- todo_line: covered by V2
- why: "a NN" against "An NN" two sentences earlier in the same paragraph.
OLD:
```latex
the multilayer perceptron (MLP), a NN containing
```
NEW:
```latex
the multilayer perceptron (MLP), an NN containing
```

### V4 — Support vector regression: "suggest" and "holds up"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "suggest robustness" is the "suggests that" construction the AI phrasing guide removes; the sentence should say what the three properties do. "how it holds up" is informal, and the last-but-one sentence joins a cited result and a pointer to Chapter 3 with "and".
OLD:
```latex
Three properties of the method suggest robustness to label noise.
```
NEW:
```latex
Three properties of the method limit the effect of a wrong label.
```

### V5 — Support vector regression: split the cited result from the pointer
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: See V4.
OLD:
```latex
degraded smoothly and linearly \citep{Cortes2015}, and Chapter~\ref{chap:robustness} reports how it holds up under each of our noise conditions.
```
NEW:
```latex
degraded smoothly and linearly \citep{Cortes2015}. Chapter~\ref{chap:robustness} reports its robustness under each of our noise conditions.
```

### V6 — Similar property principle: "clearest applications ... across cheminformatics workflows"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "One of its clearest applications" is an appraisal, and "appears across cheminformatics workflows" says nothing concrete. Say what nearest-neighbor prediction computes. Chemistry guide item 21: the principle covers biological activity, the case this thesis studies.
OLD:
```latex
structurally similar molecules tend to have similar properties \citep{Johnson1990, vanTilborg2022}.
One of its clearest applications is nearest-neighbor prediction, which appears across cheminformatics workflows. In this thesis,
```
NEW:
```latex
structurally similar molecules tend to have similar properties, including biological activity \citep{Johnson1990, vanTilborg2022}.
Nearest-neighbor prediction applies the principle directly, predicting a molecule's label from the labels of the molecules closest to it. In this thesis,
```

### V7 — Similarity measure: vague clause and "in this way"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "especially when different molecular embeddings are handled in a computational setting" has no content; every similarity in the thesis is computed. "over-interpreted in this way" points back at nothing specific.
OLD:
```latex
There is no standard definition of similarity between molecules, especially when different molecular embeddings are handled in a computational setting. Two molecules that are near-identical under a substructure fingerprint may be far apart under a PDV, and \citet{Maggiora2014} caution that quantified similarity can be over-interpreted in this way.
```
NEW:
```latex
There is no standard definition of similarity between molecules. Two molecules that are near-identical under a substructure fingerprint may be far apart under a PDV. \citet{Maggiora2014} caution that a computed similarity value is easily over-interpreted.
```

### V8 — "Here" changed to "where" after Equation dose_matching, with no note
- file: thesis.tex
- kind: author-text
- todo_line: % TODO: manual review -- edited your text: "Here" changed to "where", because the equation ends with a comma, so the sentence continues.
- why: her_text_check hit #3. The equation in her text ends with "\sigma_z," so a capital "Here" was a grammar slip. The change stays and gets the note. The note goes on its own line directly above the line it concerns, inside the paragraph.
OLD:
```latex
\end{equation}
where $\tau$ is the target standard deviation of the noise, in the units of the label, and $z_i$ is
```
NEW:
```latex
\end{equation}
% TODO: manual review -- edited your text: "Here" changed to "where", because the equation ends with a comma, so the sentence continues.
where $\tau$ is the target standard deviation of the noise, in the units of the label, and $z_i$ is
```

### V9 — Bentz paragraph: revert "counted" to her "count"
- file: thesis.tex
- kind: author-text revert
- todo_line: existing note, shortened as below
- why: Her "\citet{Svensson2025} count ..., of which 13 carry ... and eight sit" is consistent present tense, which is normal for citing what a paper reports (as in "\citet{Beyer1999} show" in the same chapter). The pass changed only "count" to "counted", which fixes nothing and leaves the sentence in two tenses. Revert the word and remove its clause from the note.
OLD:
```latex
\citet{Svensson2025} counted censored labels in 15 industrial assays
```
NEW:
```latex
\citet{Svensson2025} count censored labels in 15 industrial assays
```

### V10 — Bentz paragraph note: drop the reverted clause
- file: thesis.tex
- kind: note
- todo_line: the note itself
- why: Follows V9.
OLD:
```latex
(censoring comes from the tested range), and "count" put in the past tense for one study.
```
NEW:
```latex
(censoring comes from the tested range).
```

### V11 — Activity cliffs: abstract sentences and an appraisal opener
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "The sharpest failure" is an appraisal. "Activity cliffs inherit the measure dependence of the principle they violate" and "makes this choice explicit" are abstract; "this choice" has no antecedent. "This soft consensus over three measures is used because each measure captures only part of the underlying chemistry" states a reason the source is not cited for. Say what MoleculeACE computes, in the order it computes it. Dablander (p. 104) calls cliffs exceptions to the similarity principle; the rewrite borrows that.
OLD:
```latex
The sharpest failure of the similar property principle is the activity cliff. This is a pair of structurally similar molecules whose potency differs by a large amount because of a small chemical change \citep{Maggiora2006, Stumpfe2012}. Activity cliffs inherit the measure dependence of the principle they violate, and there is no widely accepted definition of one \citep{vanTilborg2022}. The MoleculeACE benchmark \citep{vanTilborg2022} makes this choice explicit. A pair is considered highly similar if its similarity exceeds 90\% under at least one of three measures. These are the Tanimoto coefficient on ECFPs, the Tanimoto coefficient on scaffold fingerprints and a scaled Levenshtein distance on SMILES strings. Such a pair forms a cliff if the two molecules also differ tenfold or more in $K_i$ or EC$_{50}$ \citep{vanTilborg2022}. This soft consensus over three measures is used because each measure captures only part of the underlying chemistry. Across the thirty MoleculeACE targets,
```
NEW:
```latex
An activity cliff is a pair of structurally similar molecules whose potency differs by a large amount because of a small chemical change \citep{Maggiora2006, Stumpfe2012}. Activity cliffs are exceptions to the similar property principle. Like similarity, a cliff depends on the similarity measure, and there is no widely accepted definition of one \citep{vanTilborg2022}. The MoleculeACE benchmark \citep{vanTilborg2022} uses three measures: the Tanimoto coefficient on ECFPs, the Tanimoto coefficient on scaffold fingerprints and a scaled Levenshtein distance on SMILES strings. A pair counts as highly similar if its similarity exceeds 90\% under at least one of them. Such a pair forms a cliff if the two molecules also differ tenfold or more in $K_i$ or EC$_{50}$ \citep{vanTilborg2022}. Across the thirty MoleculeACE targets,
```

### V12 — Learning with noisy labels: "scaffold group" and "assay limit"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: Two paragraphs earlier the chapter defines "scaffold family" (the new Bemis--Murcko paragraph), and Chapter 2 uses "scaffold family" everywhere else; this AI paragraph uses "scaffold group". Chemistry guide item 12: "assay limit" does not say which limit; the Bentz paragraph now defines censoring by the assay's measurable range, so use those words. The thesis-wide choice between "family" and "group" is listed for the author.
OLD:
```latex
Examples are a label at the assay limit, how many times a molecule was measured, which source reported it, or which scaffold group it belongs to.
```
NEW:
```latex
Examples are a label at the edge of the assay's measurable range, how many times a molecule was measured, which source reported it, or its scaffold family.
```

### V13 — Song2022 paragraph: rhetorical last sentence
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "has no place among them" is a rhetorical flourish. Say which group it falls in.
OLD:
```latex
A detector that runs once and hands a smaller training set to any model has no place among them.
```
NEW:
```latex
A detector that runs once and hands a smaller training set to any model falls in none of the five groups.
```

### V14 — "No single method" paragraph: unsupported colon-reveal sentence
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: review, needs manual review (keep)
- why: "This is a recurring theme in LNL research: there is no universal solution to handling noise with experimental endpoints." is a colon-reveal with no citation. Her own TODO(cite) above this paragraph says Cortes2015's abstract reports one clear loser, not that no algorithm is best, so the sentence overstates the cited result. The first sentence (Wolpert1996) already makes the general point.
OLD:
```latex
two polynomial-kernel models became sensitive only at high noise. This is a recurring theme in LNL research: there is no universal solution to handling noise with experimental endpoints. Neighbor-based cleaning
```
NEW:
```latex
two polynomial-kernel models became sensitive only at high noise. Neighbor-based cleaning
```

### V15 — NoiseInject second function: "each molecule's region" and "pattern of noise"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: manual review (keep)
- why: "region" is never defined, and "score held-out molecules against the pattern of noise" does not say what is computed. Chapter 3 (Rust pipeline paragraph) says the same thing: every split records the noise scale each molecule would have received. Use those words.
OLD:
```latex
A second function of the package returns the noise scale that each molecule's region would receive, without drawing any noise.
We use it to score held-out molecules against the pattern of noise to which the training labels were exposed. Under the
```
NEW:
```latex
A second function of the package returns the noise scale that each molecule would receive, without drawing any noise.
We apply it to held-out molecules, which receive no noise, to record the scale each would have received in training. Under the
```

### V16 — Evaluation methodology: the split rule is stated nowhere in the chapter
- file: thesis.tex
- kind: new-paragraph
- todo_line: % TODO: manual review
- why: The chapter's own takeaway says every split is group-aware and never random, and the AI TODO on the Conclusions says "splits text not yet written". The section is one sentence about metrics. This paragraph uses only what Chapters 3 and 5 already say (no group split between folds; the Yang2019 and Wallach2018 reason, as in Chapter 5's "Splitting rules"; Bemis--Murcko groups for small molecules; other domains in Section sec:method:datasets). It adds no number. Both labels exist (sec:robustness:methods, sec:method:datasets).
OLD:
```latex
Chapters~\ref{chap:flaws} to~\ref{chap:noise} state the metric used for each dataset where it is introduced.
```
NEW:
```latex
Chapters~\ref{chap:flaws} to~\ref{chap:noise} state the metric used for each dataset where it is introduced.

% TODO: manual review
No split or fold in this thesis is random, and no group of molecules is split between training and test. Random splitting inflates performance estimates for molecular data, because it places structurally similar molecules on both sides of the train/test boundary \citep{Yang2019, Wallach2018}. For small molecules the group is the Bemis--Murcko scaffold family (Section~\ref{sec:noise_injection}). Chapter~\ref{chap:robustness} gives its splits in Section~\ref{sec:robustness:methods}. Section~\ref{sec:method:datasets} gives the folds of the later chapters and the groups used for antibodies and the other datasets that are not small molecules.
```

### V17 — Conclusions: "symmetric", "fixes the metrics", "namely"
- file: thesis.tex
- kind: ai-paragraph
- todo_line: % TODO: review, needs manual review (keep; the "splits text not yet written" part is answered by V16)
- why: The chapter argues that real noise breaks the independence and zero-mean assumptions (Bentz paragraph), not symmetry, so "symmetric" names a property the chapter never discusses. "Section~\ref{sec:evaluation} fixes the metrics" is false: that section defers each dataset's metric to its chapter. "namely" is filler, used twice. The rewrite states the split rule from V16 instead. If V16 is not applied, drop the clause "and no split in the thesis is random" from the NEW text.
OLD:
```latex
This chapter has introduced both halves of a predictor, namely the representation and the model. Similarity between molecules depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle. Because real assay noise is not independent, identically distributed or symmetric, our artificial noise model comprises seven conditions. We have also set out three groups of methods for noisy labels: noise detection, noise mitigation and noise-robust training. No single method is expected to win on every dataset. NoiseInject injects these noise conditions and reports the robustness and uncertainty metrics, and Section~\ref{sec:evaluation} fixes the metrics.
```
NEW:
```latex
This chapter has introduced the two parts of a predictor, the representation and the model. Similarity between molecules depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle. Real assay noise is often not independent, identically distributed or zero-mean, so our artificial noise model has seven conditions. We have also set out three groups of methods for noisy labels: noise detection, noise mitigation and noise-robust training. No single method is expected to win on every dataset. NoiseInject injects these noise conditions and reports the robustness and uncertainty metrics, and no split in the thesis is random (Section~\ref{sec:evaluation}).
```

## For the author

- No-significance-test policy (style guide item 32): not stated in the evaluation section, because Chapter 3 keeps its Wilcoxon, Tukey and bootstrap results by decision. Decide whether to state the policy for Chapters 4 to 7 only.
- "Scaffold family" (31 uses in thesis.tex) against "scaffold group" (17, including the "Known flag" row of Table tab:noise_terms in Chapter 2); the chemistry guide (items 6 and 14) prefers "scaffold group". One term needs choosing; V12 follows Chapter 2's majority.
- Censoring direction (chemistry guide item 13): NoiseInject clips the largest labels (the censoring paragraph), but real potency censoring sits mostly at the weak end. One sentence with your reason for clipping the top is needed.
- $K_i$ typography (chemistry guide item 10): Chapter 2 is consistent ($K_i$, p$K_i$), but the IUPAC form needs a thesis-wide macro.
- "logP" twice in your PDV paragraph against "log $P$" in Chapter 3's new LogD definition; not changed, since it is a thesis-wide typography choice.
- Table tab:noise_terms still says "a label at the assay limit" and "an assay limit"; it is your table, so not changed with V12.
- Figure fig:data_cleaning caption gives $R^2$ without "higher is better"; captions belong to the figures session.
- Implementation section: "the number of badly corrupted labels" in the cross-implementation check has no threshold; give it or drop it.
- Section title "NoiseInject: a benchmarking framework" became "Implementation of the noise conditions", and "Package description" became "The NoiseInject package"; confirm both.
- "GAUCHE" became \texttt{Gauche} in your GP paragraph; the package's own name is upper case, so check which you want.
- Two of your sentences exceed 50 words (PDV contents list; decision-tree definition); left as written.
- Open TODO(cite) items in this chapter: TeachOpenCADD2023, near-verbatim Fassio2022/Xie2020, David2020, wu2018, Koutsoukas2017/Baskin2008, Student-$t$ $\nu = 5$, Cortes2015 noise levels, NoiseInject version 1.0.0 against Zenodo v0.3.0.

## Applied

All OLD blocks found exactly once in thesis.tex on 2026-10-10; applied with Edit.
- V1 applied (Overview: four questions incl. uncertainty; "learning with noisy labels"; implementation section named).
- V2 applied (edited-your-text note above the NN paragraph).
- V3 applied ("a NN" -> "an NN").
- V4 applied ("suggest robustness" -> "limit the effect of a wrong label").
- V5 applied (Cortes2015 result and Chapter 3 pointer split).
- V6 applied (biological activity added; nearest-neighbor prediction said plainly).
- V7 applied (vague clause and "in this way" removed).
- V8 applied (edited-your-text note for "Here" -> "where").
- V9 applied ("counted" reverted to "count").
- V10 applied (reverted clause dropped from the Bentz note).
- V11 applied (activity cliff paragraph rewritten; unsupported "soft consensus" reason removed).
- V12 applied ("scaffold family"; "edge of the assay's measurable range").
- V13 applied ("falls in none of the five groups").
- V14 applied (uncited colon sentence removed).
- V15 applied (second NoiseInject function described by what it records).
- V16 applied (new split-rule paragraph with % TODO: manual review; labels and Yang2019/Wallach2018 verified to exist).
- V17 applied (Conclusions rewritten; split clause kept since V16 applied).
