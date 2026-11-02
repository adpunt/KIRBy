# Chapter 2 against BENCHMARK_TABLE.md

Chapter 2 is "Molecular representations, models and label noise" (thesis.tex lines 64 to about 490, read 2026-10-10). This is an audit only. thesis.tex was not edited.

## Which paragraphs count as AI-written

The benchmark script (`split_hers_ai.py` with `segment_full_author.py`) finds 21 AI paragraphs in Chapter 2, with 83 sentences. That count misses seven paragraphs:

- Four paragraphs sit under subsubsection headings whose marker is above the heading, not above the paragraph. They are Regression noise (line 315), Classification noise (368), Robustness metrics (376) and Uncertainty metrics (393). Each heading has "% TODO: manual review" directly above it.
- Three are the text after an equation in an AI paragraph, which the segmenter splits off as a new paragraph: "which is the number of bits set..." (119), "where τ̂ is the delivered..." (344) and "computed over ten equal-count bins..." (402).

This audit counts all seven as AI-written. The corrected total is 28 paragraphs and 115 sentences: 52 in Background, 57 in Methods and 6 in Conclusion. One caption is AI-written, the data-cleaning figure at line 284. No table in Chapter 2 has a marker.

Sections were typed the same way as in the benchmark. Overview through "Learning with noisy labels" counts as Background. "Implementation of the noise conditions", NoiseInject and "Evaluation methodology" count as Methods. "Conclusions" counts as Conclusion.

The 858-word Background column in BENCHMARK_TABLE.md is Chapter 2 alone. With the script's paragraph assignment, the numbers below reproduce it exactly.

## Language measures, AI-written paragraphs only

Each reference range runs from the lowest to the highest of Buttenschoen, Dablander and Vost for that section type. "Script" uses the benchmark's paragraph assignment. "Corrected" adds the seven paragraphs above.

The Conclusion is one paragraph of 6 sentences and 115 words. Its rates are reported, but they are too small to judge by. I read the paragraph instead.

### Background (corrected: 12 paragraphs, 52 sentences, 953 words)

| Measure | Reference range | Yours | AI, script | AI, corrected | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 23.2–26.0 | 20.5 | 18.7 | 18.3 | Not a target. Short sentences are your preference. The problem is uniformity (see SD). |
| % sentences over 30 words | 20.5–28.6 | 12.1 | 4.3 | 3.8 | Not done. Two of 52 sentences go over 30 words. Merges 2, 4, 7 and 13 each produce one. |
| Dependent clauses per sentence | 0.59–0.81 | 0.60 | 0.70 | 0.67 | Done |
| Words per paragraph | 91.7–154.7 | 94.5 | 78.0 | 79.4 | Not done. Three fragment paragraphs: dimensions (193), Bemis–Murcko (243) and Song2022 (309). Fixes 6, 9 and 11. |
| Lexical density % | 61.2–62.4 | 62.4 | 58.6 | 57.6 | Not a target. It follows from the defining prose. |
| Nominalizations /100 words | 4.0–4.5 | 5.5 | 3.8 | 3.7 | Not a target |
| % passive | 26.5–40.2 | 27.5 | 21.7 | 23.1 | Not a target |
| we/our/I /100 words | 0.29–0.85 | 0.49 | 0.47 | 0.42 | Done |
| Hedges /100 words | 0.95–1.1 | 1.0 | 0.35 | 0.42 | Not a target. The AI sentences state cited facts, and the rule is to hedge only weak evidence. |
| Boosters /100 words | 0.19–0.39 | 0.19 | 0.0 | 0.0 | Not a target |
| AI-guide phrases /100 words | 0.03–0.20 | 0.16 | 0.0 | 0.0 | Done |
| Numbers /100 words | 0.46–1.2 | 0.49 | 0.70 | 0.63 | Done |
| % sentences with 3+ numbers | 0.83–1.7 | 0.48 | 0.0 | 0.0 | Not a target |
| MTLD | 51.7–124.4 | 101.2 | 95.0 | 86.0 | Done |
| MATTR-50 % | 78.3–84.3 | 82.4 | 81.6 | 80.6 | Done |
| Sentence length SD (words) | 10.3–23.8 | 7.76 | 6.62 | 6.54 | **Not done.** This is the main one. Fixes 1, 2, 4, 5, 7, 8 and 13 merge short sentences and leave the long ones alone. |
| Sentence length CV | 0.46–0.95 | 0.39 | 0.36 | 0.36 | Not done. Same fixes. |
| Mean change in length between consecutive sentences | 11.5–18.7 | 7.99 | 7.40 | 7.10 | Not done. Same fixes. |
| GPT-2 surprisal, bits/token | 6.60–7.04 | 6.95 | 7.80 | 7.72 | Not a target. It is driven by symbols and code names (ε, NoiseInject, Meta-Weight-Net). |
| Surprisal SD | 1.13–1.27 | 1.14 | 1.14 | 1.20 | Done |
| Connectives /100 words | 1.84–2.07 | 1.57 | 1.08 | 1.39 | Not done, mildly. The sentences are chained without links. The merges add "so", ", whereas" and ";", and remove some full stops. |
| Distinct / used connectives | 0.18–0.42 | 0.35 | 0.78 | 0.69 | Not a target. Few connectives, none repeated. |
| % opening with formulaic connector | 1.66–5.43 | 6.28 | 0.0 | 0.0 | Not a target |
| Three-item lists /100 sentences | 3.70–13.6 | 9.18 | 17.4 | 15.4 | Not done. The regex found 8 lists. Two are genuine lists of content with a citation per item (MoleculeACE measures, noise kinds), and those stay. Fixes 1, 3 and 10 remove three. |
| Pronouns and demonstratives /100 words | 3.74–5.02 | 5.44 | 4.91 | 5.03 | Done |

### Methods (corrected: 15 paragraphs, 57 sentences, 1,211 words)

| Measure | Reference range | Yours | AI, script | AI, corrected | Verdict |
|---|---|---|---|---|---|
| Words per sentence | 22.3–28.0 | 20.1 | 19.8 | 21.3 | Not a target |
| % sentences over 30 words | 21.1–32.7 | 12.2 | 6.5 | 10.5 | Not done, mildly. Merges 14, 19 and 22 each give one. |
| Dependent clauses per sentence | 0.46–0.95 | 0.50 | 0.39 | 0.54 | Done |
| Words per paragraph | 76.3–142.4 | 81.9 | 68.3 | 80.7 | Done when corrected. The fragments are "Regression noise" (378, 40 words) and "second function" (418, 45 words). Fixes 15 and 16. |
| Lexical density % | 58.8–59.7 | 56.6 | 57.8 | 57.0 | Not a target |
| Nominalizations /100 words | 3.6–5.0 | 3.8 | 2.8 | 3.0 | Not a target |
| % passive | 40.8–45.6 | 28.2 | 25.8 | 29.8 | Not a target. Package actions are written with the package as the subject, and that is clearer. |
| we/our/I /100 words | 0.83–1.6 | 1.1 | 0.33 | 0.25 | Not a target in a software description |
| Hedges /100 words | 0.36–0.85 | 0.39 | 0.0 | 0.25 | Not a target |
| Boosters /100 words | 0.14–0.26 | 0.10 | 0.0 | 0.08 | Not a target |
| AI-guide phrases /100 words | 0.0–0.10 | 0.01 | 0.0 | 0.0 | Done |
| Numbers /100 words | 1.8–2.1 | 2.5 | 0.81 | 1.5 | Not a target |
| % sentences with 3+ numbers | 3.7–4.7 | 6.0 | 3.2 | 3.5 | Done |
| MTLD | 56.9–81.9 | 69.6 | 62.0 | 55.1 | Not a target. It falls just below because a specification repeats its terms (noise, level, scale). |
| MATTR-50 % | 75.2–79.8 | 78.5 | 77.8 | 76.1 | Done |
| Sentence length SD | 12.1–18.1 | 9.19 | 6.27 | 9.68 | Not done. Fixes 14, 15, 19, 20 and 22. |
| Sentence length CV | 0.56–0.67 | 0.47 | 0.33 | 0.46 | Not done. Same fixes. |
| Mean change in length between consecutive sentences | 12.5–17.1 | 9.13 | 7.32 | 7.24 | Not done. Same fixes. |
| GPT-2 surprisal | 7.04–7.42 | 7.67 | 8.09 | 7.80 | Not a target. Symbols. |
| Surprisal SD | 1.11–1.37 | 1.22 | 0.91 | 1.05 | Not a target |
| Connectives /100 words | 1.16–1.76 | 1.80 | 1.85 | 1.77 | Done, at the edge. "also" appears 5 times. Fix 21 removes "first/second/third". |
| Distinct / used connectives | 0.15–0.28 | 0.17 | 0.73 | 0.48 | Not a target |
| % opening with formulaic connector | 1.01–3.48 | 0.85 | 0.0 | 0.0 | Not a target |
| Three-item lists /100 sentences | 6.97–8.39 | 12.8 | 9.7 | 15.8 | Not a target in most cases. Of the 9 lists, 7 list package options, metrics or strategies, which a specification has to do. Fix 21 changes one. |
| Pronouns and demonstratives /100 words | 3.62–5.50 | 5.04 | 3.03 | 4.21 | Done |

### Conclusion (6 sentences, 115 words; rates shown only for completeness)

| Measure | Reference range | Yours | AI | Verdict |
|---|---|---|---|---|
| Words per sentence | 25.0–32.5 | 21.0 | 19.2 | Not a target |
| Sentence length SD | 8.01–12.6 | 9.24 | 4.52 | Not done. The six sentences run 13 to 25 words. Fix 23 builds one long sentence. |
| Words per paragraph | 95.3–138.9 | 123.1 | 115 | Done |
| we/our/I /100 words | 1.3–1.9 | 0.41 | 2.6 | Done |
| Three-item lists /100 sentences | 9.38–14.3 | 12.8 | 66.7 | Not done, but the count overstates it. The regex counts 4 lists, and 2 of them are pairs that the regex misreads. The other 2 are content: the three assumptions and the three groups of methods. Fix 23 does not add lists. |
| Moves (sentences coded by hand) | | | | |
| method/background restated | 10.4–15.9% | 14.1 | 4 of 6 | Expected at the end of a background chapter |
| claim of strength | 15.1–29.9% | 0.0 | 0 of 6 | **Not done.** Fix 23. |
| limitation | 0–5.5% | 15.3 | 0 of 6 | Done |
| link/transition | 1.4–4.8% | 1.2 | 2 of 6: the opening "We have introduced..." and the closing pointer | Not done. The opening sentence only announces. Fix 23 removes it. The closing pointer stays. |
| recommendation/implication | 16.4–17.9% | 18.8 | 0 of 6 | Not a target in Chapter 2 |

## Structure rows that apply to Chapter 2

| Row | Status | Fix |
|---|---|---|
| Chapter opening links back | **Not done.** The Overview (line 69, AI) does not name Chapter 1. BENCHMARK_TABLE lists only Chapters 3–7 as done. | Fix 1 |
| Chapter ends point forward | Done. The last sentence names Chapter 3's question. | none |
| Lead-in between section title and first subsection | Done. Every section with subsections has text first: Models (her text), Noise in molecular data (hers), Implementation (AI) and NoiseInject (hers). | none |
| Title case | Done for headings. The column headers of Tables `tab:noise_sources` and `tab:noise_conditions` are in Title Case ("Kind of Error", "What It Changes"). Both tables have no marker, so they are yours. | Not a target (your text) |
| Section titles | Done. Topic noun phrases; the AI heading "Learning with noisy labels" fits. | none |
| Limitations | Done. The Conclusions have no limitation sentence. | none |
| Captions | One AI caption (data cleaning, line 288, 45 words; the references average 50–74 words). It does not say which direction is better. | Fix 12 |
| Decimals | Done. No AI paragraph has more than two decimals. | none |
| Figure/table references | Done. "Figure", "Table" and "Equation" are written in full everywhere. | none |
| Statistics, unbacked claims | One AI claim has no pointer: AUCnorm "is also less directly correlated with clean-label R² than other robustness metrics we tested" (line 382). The evidence is in Chapter 3 (lines 621 and 1039). | Fix 18 |
| "you" in body | No AI paragraph uses it. "When you pool data..." (line 142) is your text. | Not a target |
| Naming | "In the KIRBy experiments" (line 423). The thesis uses KIRBy only for the repository. The residual-sized noise is used in Chapter 7 (table at line 4610). | Fix 17 |
| Numbers in results, results pointers | Chapter 2 has no Results section. | Not applicable |

## Fixes

Line numbers are those of thesis.tex on 2026-10-10. Chapter 2 sits above the chapters the other workflow is editing, so they should not shift. None of these fixes changes a meaning or a number.

**1. Overview, "This chapter describes the molecular representations and" (line 69).** Rows: chapter opening links back; three-item lists; sentence length SD.
Old: `This chapter describes the molecular representations and models we use, molecular similarity and activity cliffs, the sources of label noise, how we inject noise, the methods for learning with noisy labels, the NoiseInject package and how results are evaluated.`
New: `Chapter~\ref{chap:intro} set out the questions this thesis asks; this chapter gives the background they rest on. We first describe the molecular representations and models we use, and molecular similarity and activity cliffs. We then turn to label noise: where it comes from in bioactivity data, how we inject it and how methods for noisy labels handle it. The chapter closes with our NoiseInject package and how results are evaluated.`

**2. SVR, "Support vector regression fits a function that" (line 133).** Row: sentence length SD (a 9-word sentence next to a 15-word one).
Old: `The features enter the model only through the kernel. With the radial basis function (RBF) kernel, every feature enters the Euclidean distance between two molecules.`
New: `The features enter the model only through the kernel, so with the radial basis function (RBF) kernel every feature enters the Euclidean distance between two molecules.`

**3. SVR, same paragraph.** Row: three-item lists. Optional, because the three items are real properties.
Old: `Three properties limit the effect of a wrong label: errors inside the $\epsilon$ tube change nothing, the loss outside it grows linearly, and the weight of each molecule is capped at $C$ \citep{Smola2004}.`
New: `A wrong label has limited effect: an error inside the $\epsilon$ tube changes nothing, and outside it the loss grows only linearly and the weight of each molecule is capped at $C$ \citep{Smola2004}.`

**4. Similarity, "Much of cheminformatics rests on the similar" (line 109).** Row: sentence length SD. The 11-word second sentence becomes a clause of the first.
Old: `Much of cheminformatics rests on the similar property principle, which states that structurally similar molecules tend to have similar properties, including biological activity \citep{Johnson1990, vanTilborg2022}.`↵`Nearest-neighbor prediction, GP kernels and many noise detection methods rely on it.`
New: `Much of cheminformatics rests on the similar property principle, which states that structurally similar molecules tend to have similar properties, including biological activity \citep{Johnson1990, vanTilborg2022}; nearest-neighbor prediction, GP kernels and many noise detection methods all rely on it.`

**5. Tanimoto, "The most common measure for binary fingerprints" (lines 114–119).** Rows: sentence length SD; a pointer sentence that only announces. Two parallel sentences both end "counts", and the Chapter 4 pointer stands alone. This fix also takes in the paragraph moved by fix 6.
Old: `In the Tanimoto coefficient, a bit that is off in both molecules contributes nothing, whereas every bit that is set in either molecule counts. In the Euclidean distance, every feature that differs between the two molecules counts. Chapter~\ref{chap:flaws} returns to this point (Equation~\ref{eq:distance_split}).`
New: `A bit that is off in both molecules contributes nothing to the Tanimoto coefficient, whereas every bit set in either molecule counts; in the Euclidean distance, every feature that differs between the two molecules counts. As the number of dimensions grows, the distances to the nearest and to the furthest point become increasingly similar \citep{Beyer1999}. Molecular representations are routinely high-dimensional and sparse, and when many of their features carry no information about the property, distances computed over them are dominated by those features, a point Chapter~\ref{chap:flaws} returns to (Equation~\ref{eq:distance_split}).`

**6. Dimensions, "As the number of dimensions grows, the" (line 130).** Rows: words per paragraph; fragment paragraph. A three-sentence, 50-word paragraph that continues the distance argument of the Tanimoto paragraph.
Old: the whole paragraph, with its marker: `As the number of dimensions grows, ... dominated by those features.`
New: delete it here. Its text moves into fix 5, and the 7-word sentence "Molecular representations are routinely high-dimensional and sparse." is merged with the one after it. Carry the "% TODO: manual review" marker to the Tanimoto paragraph, which already has one. If you prefer this paragraph where it is, use just the merge: `Molecular representations are routinely high-dimensional and sparse, and when many of their features carry no information about the property, distances computed over them are dominated by those features.`

**7. Activity cliff, "An activity cliff is a pair of" (line 127).** Rows: sentence length SD; fragmentation. The "also" has nothing to refer back to.
Old: `A cliff also depends on the similarity measure, and there is no widely accepted definition of one \citep{vanTilborg2022}.`
New: `Whether a pair is a cliff depends on the similarity measure, and there is no widely accepted definition \citep{vanTilborg2022}.`
Old: `A pair counts as highly similar if its similarity exceeds 90\% under at least one of them. Such a pair forms a cliff if the two molecules also differ tenfold or more in $K_i$ or EC$_{50}$ \citep{vanTilborg2022}.`
New: `A pair forms a cliff if its similarity exceeds 90\% under at least one of them and the two molecules differ tenfold or more in $K_i$ or EC$_{50}$ \citep{vanTilborg2022}.`

**8. Noise sources, "These noise sources can interact with each" (line 168).** Rows: link sentence that only announces; sentence length SD. The list of three stays, because each item has its own citation.
Old: `These noise sources can interact with each other and present in more ways than a simple Gaussian. Real noise can be heteroscedastic, larger in some regions of chemical space than others \citep{Heid2023}, shared within a laboratory \citep{Bentz2013}, and non-normal \citep{Kruger2012}. We wanted to address this complexity in our research by examining a variety of ways of both modeling and adding noise to datasets.`
New: `These sources can interact, and the noise they produce departs from a simple Gaussian: it can be heteroscedastic, larger in some regions of chemical space than others \citep{Heid2023}, shared within a laboratory \citep{Bentz2013}, and non-normal \citep{Kruger2012}. We therefore model and add noise in several ways (Section~\ref{sec:noise_injection}).`

**9. Bemis–Murcko, "The Bemis--Murcko scaffold of a molecule is" (line 180).** Rows: words per paragraph; fragment paragraph. It is a two-sentence, 30-word definition between two of your paragraphs, and only AI text may change. One option is a single sentence, kept as its own paragraph. The other is a footnote on "Bemis--Murcko scaffold families" in your paragraph above, which you would need to agree to.
Old: `The Bemis--Murcko scaffold of a molecule is its ring systems and the chains that link them, with all other side chains removed \citep{Bemis1996}. Molecules that share one scaffold form a scaffold family.`
New: `Molecules that share a Bemis--Murcko scaffold, the ring systems and linking chains left once all other side chains are removed \citep{Bemis1996}, form a scaffold family.`

**10. LNL, "The problem of training a model when" (line 243).** Rows: re-describes the table, cut and point to it; fragment sentence; three-item lists. The detector, known-flag, noise-detection and noise-robust-training sentences repeat rows of Table `tab:noise_terms`, which follows straight after. Keep the definitions the table does not give: LNL, the two halves, and the sample-selection exception.
Old: `A detector infers from the features and labels how wrong each label looks; the score it gives a molecule is that molecule's suspicion score. A known flag is recorded in the data, such as a label at the assay limit or the molecule's scaffold family. Noise detection is a detector alone. Noise mitigation is a selector followed by deletion, down-weighting or moving the label, after which any model can be fitted; Figure~\ref{fig:data_cleaning} shows how it is evaluated. Noise-robust training leaves the data untouched and changes the loss, the kernel, a noise term, the training loop or the model, as in co-teaching \citep{Han2018} and Meta-Weight-Net \citep{shu2019metaweightnet}.`
New: `The selector is either a detector, which scores how wrong each label looks (its suspicion score), or a known flag recorded in the data, such as a label at the assay limit. Noise detection is a detector alone; noise mitigation follows a selector with one of the three actions, after which any model can be fitted (Figure~\ref{fig:data_cleaning}). Noise-robust training leaves the data untouched and changes the loss, the model or the training loop, as in co-teaching \citep{Han2018} and Meta-Weight-Net \citep{shu2019metaweightnet}.`
If you would rather keep every definition in the text, apply just the merge of "Noise detection is a detector alone." into the next sentence.

**11. Song2022, "\citet{Song2022} sort deep learning methods for noisy" (line 246).** Rows: fragment paragraph; repeated wording ("all five" twice). Merge it into the LNL paragraph, before the sample-selection sentence, which is about one of Song's groups.
Old (whole paragraph): `\citet{Song2022} sort deep learning methods for noisy labels into five groups: robust architectures, robust regularization, robust loss functions, loss adjustment and sample selection. All five act during training, so all five are noise-robust training.`
plus, in the LNL paragraph: `A sample-selection step taken out of its training loop and used only to delete molecules before a refit counts as noise mitigation.`
New (end of the LNL paragraph): `The five groups into which \citet{Song2022} sort deep learning methods for noisy labels (robust architectures, robust regularization, robust loss functions, loss adjustment and sample selection) all act during training, and so count as noise-robust training. The exception is a sample-selection step taken out of its training loop and used only to delete molecules before a refit, which counts as noise mitigation.`

**12. Caption, "Evaluation of a noise mitigation method, with" (line 288).** Rows: caption (metric, unit, direction); fragment.
Old: `Noise is added to the training labels only. A random forest trained on the noisy labels is compared with one trained on labels corrected by a Bayesian model, and both are scored by $R^2$ on the clean test set.`
New: `A random forest trained on noisy training labels is compared with one trained on labels corrected by a Bayesian model; the test labels stay clean, and both forests are scored on them by $R^2$ (higher is better).`

**13. "No single method is expected to win, since" (line 293).** Rows: sentence length SD; triplet; vary connectives. One 13-word sentence and one 15-word sentence merge into their neighbors.
Old: `\citet{Cortes2015} compared 12 algorithms on 12 bioactivity datasets with increasing Gaussian label noise. Gradient boosting was the least tolerant, the other 11 degraded comparably, and two polynomial-kernel models became sensitive only at high noise.`
New: `\citet{Cortes2015} compared 12 algorithms on 12 bioactivity datasets with increasing Gaussian label noise: gradient boosting was the least tolerant and the other 11 degraded comparably, with two polynomial-kernel models sensitive only at high noise.`
Old: `Neighbor-based cleaning dates back to editing a training set by its nearest neighbors \citep{Wilson1972}. It relies on the similar property principle, which activity cliffs break, and QSAR models frequently fail to predict cliffs \citep{Dablander2023}. Chapter~\ref{chap:flaws} tests the principle, and Chapter~\ref{chap:noise} tests neighbor-based noise mitigation with the hybrid as the feature space.`
New: `Neighbor-based cleaning, which dates back to editing a training set by its nearest neighbors \citep{Wilson1972}, relies on the similar property principle, which activity cliffs break, and QSAR models frequently fail to predict cliffs \citep{Dablander2023}. Chapter~\ref{chap:flaws} tests the principle, and Chapter~\ref{chap:noise} tests neighbor-based noise mitigation with the hybrid representation as the feature space.`

**14. Implementation, "The noise conditions are implemented twice." (line 298).** Rows: sentence length SD (a 6-word opener).
Old: `The noise conditions are implemented twice. NoiseInject, an open-source Python package, injects`
New: `The noise conditions are implemented twice: NoiseInject, an open-source Python package, injects`
Also: `Before any cluster run,` → `Before any run on the computing cluster,`. "cluster run" is a session word.

**15. Regression noise, "The regression injector sets a noise shape" (line 315) and "The requested noise $\tau$ is the" (line 320).** Rows: words per paragraph; fragment paragraph (40 words). Join the two paragraphs and delete the blank line and the second marker between them. The text stays as it is.

**16. "A second function returns the noise scale" (line 355).** Rows: fragment paragraph (45 words); unclear reference. "the result" has no antecedent.
Old: `Under uniform and grouped-shifted targeting every molecule gets the same scale. The package then flags the result as degenerate and does not report a rank correlation of zero.`
New: `Under uniform and grouped-shifted targeting every molecule gets the same scale, so a rank correlation with the scale is undefined, and the package flags it as degenerate instead of reporting zero.`
Then join this paragraph to the next one ("A separate injector...", fix 17). Both describe further functions of the package.

**17. "A separate injector sizes the noise against" (line 360).** Rows: naming; sentence length SD (a 12-word sentence).
Old: `A separate injector sizes the noise against a unit supplied by the user. In the KIRBy experiments this unit is`
New: `A separate injector sizes the noise against a unit supplied by the user; in Chapter~\ref{chap:noise} this unit is`
I checked Chapter 7 only (table at line 4610). If Chapters 5 or 6 also use it, write `Chapters~\ref{chap:method} to~\ref{chap:noise}`.

**18. Robustness metrics, "For regression, the package computes $R^2$, the" (line 382).** Row: statistics, claim without backing.
Old: `It is also less directly correlated with clean-label $R^2$ than other robustness metrics we tested.`
New: `It is also less directly correlated with clean-label $R^2$ than other robustness metrics we tested (Chapter~\ref{chap:robustness}).`

**19. Uncertainty metrics, "For a model that returns a predicted" (line 402).** Row: sentence length SD (two short sentences after the 70-word one).
Old: `computed over ten equal-count bins $B_b$ of $u$. All of these use the clean test labels.`
New: `computed over ten equal-count bins $B_b$ of $u$; all of these use the clean test labels.`

**20. Same paragraph (lines 402–405).** Row: fragment sentence (6 words).
Old: `can only be computed on the corrupted molecules, where the noise is known. It is reported within each noise level.`
New: `can only be computed on the corrupted molecules, where the noise is known, and is reported within each noise level.`

**21. "Three optional wrappers produce $u_i$ from a" (line 410).** Rows: connectives (first/second/third scaffold); three-item list kept as one sentence.
Old: `Three optional wrappers produce $u_i$ from a point predictor. The first applies split-conformal prediction to any scikit-learn-compatible model, optionally scaled by the label spread of each molecule's nearest training neighbors. The second fits an exact Gaussian process with a Tanimoto or RBF kernel through \texttt{Gauche} \citep{gauche}, and the third applies Monte Carlo dropout to any \texttt{PyTorch} network \citep{gal2016}.`
New: `Three optional wrappers produce $u_i$ from a point predictor: split-conformal prediction around any scikit-learn-compatible model, optionally scaled by the label spread of each molecule's nearest training neighbors; an exact Gaussian process with a Tanimoto or RBF kernel through \texttt{Gauche} \citep{gauche}; and Monte Carlo dropout on any \texttt{PyTorch} network \citep{gal2016}.`

**22. Evaluation, "Chapter~\ref{chap:robustness} reports predictive performance as $R^2$" (line 418).** Row: sentence length SD.
Old: `Chapters~\ref{chap:flaws} to~\ref{chap:noise} state the metric used for each dataset where it is introduced. Outside Chapter~\ref{chap:robustness}, results are reported as per-fold scores and their spread across folds, and we use no significance tests.`
New: `Chapters~\ref{chap:flaws} to~\ref{chap:noise} state the metric used for each dataset where it is introduced, and report results as per-fold scores and their spread across folds, with no significance tests.`

**23. Conclusions, "We have introduced both parts of a" (line 426).** Rows: a link sentence that only announces, so cut it; a claim of achievement is missing, so state it with "we"; sentence length SD; three-item lists, none added.
Old: `We have introduced both parts of a predictor, the representation and the model. Similarity depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle.`
New: `A predictor has two parts, the representation and the model, and the representation also fixes what counts as similar: similarity depends on the representation and the coefficient, and activity cliffs and uninformative features break the similar property principle.`
Old: `Our NoiseInject package injects the noise conditions and reports the robustness and uncertainty metrics.`
New: `We implemented these conditions in NoiseInject, an open-source package that injects them and reports the robustness and uncertainty metrics for any model.`
"for any model" restates "model-agnostic" from line 306.

## Checked and left alone

- The record list at line 335 (seven items), the six flipping strategies at line 368 and the metric lists at lines 376–378. A specification has to list these. Not a target.
- The splits paragraph at line 421, the censoring paragraph at line 329 and the classification paragraph. Lengths vary, and nothing matches a benchmark row.
- The section headings, including the AI heading "Learning with noisy labels". Not a target.
