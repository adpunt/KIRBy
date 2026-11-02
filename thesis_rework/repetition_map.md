# Repetition map

Source: part files in `scratchpad/trim/parts/`. Line numbers are part-file line numbers. Only text that appears in the PDF was compared (no `%` comments, no REMOVED blocks). Labels were checked in `thesis.tex.bak_pre_trim_2026-09-30`.

## Legend

- **M** = marked. A comment directly above the paragraph contains TODO or CODE INPUT.
- **M-block** = marked by a surrounding `% ==== CODE INPUT: new, needs review ====` block. No comment sits directly above the paragraph. This occurs in Chapter 4 only, and these paragraphs are treated as editable.
- **U** = unmarked (signed off). Never edit.
- **Ch2/Ch3** = off limits, even where marked.
- **OBVIOUS** = the fix is clear: a pure restatement, or a duplicate version of a paragraph.
- **FENCE** = Adelaide should decide.

## Four things to know before editing

1. **Chapter 5 and 6 subsections have no labels.** None of the subsections of Chapters 5 and 6 has a `\label` ("From Base Representations to a Hybrid", "Similarity in a Hybrid", "Recovery Rate" and so on). The thesis already refers to them by quoted title, for example `Chapter~\ref{chap:similarity}, ``Recovery rate''`. The back-references proposed below follow that convention. Labels that do exist and are used below:
   - `sec:lnl_background`, `sec:noise_injection`, `sec:noiseinject_pkg`
   - `eq:tanimoto`, `eq:tanimoto_kernel`, `tab:noise_conditions`
   - `sec:robustness:methods`, `sec:uq`
   - `sec:flaws:representations`, `sec:flaws:constant`, `sec:flaws:similarity`, `sec:flaws:consequences`, `sec:lnl_motivation`, `sec:flaws:neighbors`
   - `tab:search_tiers`, `tab:pruning_rules`, `tab:aug_conditions`, `tab:datasets`
   - `app:datasets`, `app:datasets:smallmol`, `app:datasets:ch3`

   The subsection "Molecules with Identical Feature Vectors" in Chapter 4 also has no label.
2. **Several back-references point to Chapter 2's "Evaluation Methodology", and that section does not hold what they cite.** The citing paragraphs are c5:760, c5:860 (twice), c5:868 and c5:1140. They cite it for group splits, the Bemis--Murcko scaffold, the rule for datasets with fewer groups than folds, and metric definitions. The section (c2:503--509) defines only RMSE, R², Spearman, coverage and ECE. The scaffold grouping is actually defined in Chapter 3 (c3:39, under `sec:robustness:methods`). Deduplicating the splitting text therefore needs a real target first. This affects item C5-S11.
3. **Some paired paragraphs have an unmarked original.** In these pairs an original paragraph is unmarked and the revision directly below it is marked `% TODO: REVIEW MODIFICATION`. Both render. By the rules the unmarked one cannot be touched, so the only allowed edit is to delete the revision. The pairs are c5:134/137, c5:295/298, c5:300/303, c5:368/371 and c5:892/894. All five are FENCE.
4. **Chapter 5 is planned as a stand-alone paper.** Several OBVIOUS items remove text that repeats Chapter 2, Chapter 4 or the appendix. If Chapter 5 must stand alone, she may prefer to keep those copies. These items are flagged "paper caveat".

---

## c1_intro.tex

**C1-01. Definition of LNL.**
- c1:26, Ch1 prose. "Learning with noisy labels (LNL) is the broad family of methods for..." M.
- c2:341, `sec:lnl_background`. "LNL covers every method for training a model when some of its..." Ch2.
- Canonical: c2:341.
- c1:26 is a one-sentence paragraph. The paragraph that was meant to follow it (the exploratory study) exists only as notes. As a result, c1:45 ("such methods rely") has no antecedent in the PDF.
- Option: shorten c1:26 to the acronym plus `(Section~\ref{sec:lnl_background})`.
- FENCE. The introduction has to name LNL somewhere, and cutting it makes the c1:45 gap worse.

**C1-02. Where the thesis began: the hypothesis about distance, and the first version of the hybrid.** Two claims. First, irrelevant features dominate the distance and grow linearly with the number of features. Second, the first version was built for similarity and turned out to work as a model input.
- c1:45, Ch1 prose. "We hypothesized that the fault lay in the distance between molecules on..." M. Holds both claims.
- c4:262, `sec:flaws:similarity`. "Every irrelevant feature that varies adds one further term, so the irrelevant..." U. Holds the mechanism.
- c4:414, `sec:lnl_motivation`. "There are two possible reasons for the poor performance of the distances." M-block. Its last three sentences.
- c4:487, Ch4 Conclusions. "These flaws are where the thesis began. Our hypothesis was that similarity..." M-block. Restates c1:45 almost sentence for sentence.
- c5:20, Ch5 Introduction. "Chapter~\ref{chap:flaws} showed that standard representations carry many features..." M. Its last sentence: "The selection described here was first built to compute similarity and was later found to work as a model input."
- c6:12, Ch6 opening. "The hybrid representation was first conceived as a way to repair molecular similarity..." M. First clause.
- c6:43, Ch6 Introduction. "The original idea behind the hybrid was to select the features relevant..." M. Repeats c6:12 within the same chapter.
- c7:318, Ch7 Conclusions. "The second half began from a negative result..." M. Its second sentence is nearly verbatim c1:45 and c7:79.
- c5:890, "Representations by Domain". "Any representation that gives one fixed-length vector per sample..." U. "The first version of the method used three molecular representations".
- Canonical: c1:45 for the story, c4:262 for the mechanism.
- c5:20: delete the last sentence. OBVIOUS.
- c6:43: delete the paragraph, since c6:12 already says it. OBVIOUS.
- c4:487: either cut it to one sentence with a back-reference to `Chapter~\ref{chap:intro}`, or keep it as the bridge into Chapter 5 and shorten c1:45 instead. FENCE, because the canonical location is debatable.
- c7:318: see C7-01.

**C1-03. The Chapter 3 findings, stated in words.** Architecture matters more than representation for robustness. The shape of the noise matters little. Grouped and censored noise do the most damage. Uncertainty rises with noise but does not identify corrupted labels.
- c1:138, Outline. "Chapter~\ref{chap:robustness} studies the noise robustness of QSAR models..." M.
- c7:76, Ch7 Summary. "Chapter~\ref{chap:robustness} asked how the representation, the model and the type..." M. Carries numbers.
- c7:314, Ch7 Conclusions. "This thesis set out to predict molecular properties from labels that carry..." M.
- c7:249, Ch7 Limitations. Its uncertainty sentence. M.
- c3:485--498, Ch3 Conclusions. Ch3.
- c4:402, `sec:lnl_motivation`. "This section reuses an exploratory study..." M. One sentence restating the Chapter 3 variance result, with a back-reference. It can stay.
- Canonical: Chapter 3.
- FENCE. This is a structural choice. The thesis has an outline with findings, a per-chapter Summary and a final Conclusions, and all three restate the same four findings. Proposal: keep c1:138 in words (her notes asked for findings) and keep c7:76 with numbers. Then cut c7:314 down to the advice paragraphs (c7:321 and c7:325), since the Summary sits a few pages earlier in the same chapter.

**C1-04. The Chapter 4 argument in three steps.** Features do not vary; similarity inherits this; methods that rely on similarity fail.
- c1:141, Outline. "Chapter~\ref{chap:flaws} motivates the methods of the later chapters..." M.
- c4:20, Ch4 opening. "This chapter asks whether the standard ways of comparing molecules suit the..." M.
- c4:29, Ch4 Introduction. "General-purpose molecular representations are designed to work across chemical domains..." U.
- c4:37 and c4:39, section roadmap. U.
- c4:474--481, Ch4 Conclusions. M-block.
- c7:79, Ch7 Summary. M.
- FENCE. Each copy has its own job (outline, abstract, introduction, conclusions, thesis summary). The obvious sub-cases are C4-10 and C7-01.

**C1-05. The Lhasa hERG sets (7,772 training and 4,383 test molecules).**
- c1:165, Collaborations. "This work was supervised by Garrett Morris and co-supervised..." M.
- c5:1239, separate sites. "hERG channel blockade is a cardiotoxicity liability..." M. It also restates the blocker threshold (pKi ≥ 6) and the InChIKey rule "keeping the public copy".
- c8:114, Appendix A. "The separate-site experiments of Chapter~\ref{chap:method} use two further hERG sets..." M.
- c8:267, Appendix B. "The exception is the hERG blocker data..." M.
- Canonical: c8:114.
- Option: c1:165 can drop the counts. c5:1239 can shorten the curation clauses to `(Appendix~\ref{app:datasets:smallmol})`.
- FENCE, low value; paper caveat applies to c5:1239.

**C1-06. Statement that the work is her own.**
- c1:171, Collaborations. "The adaptive hybrid representation, its software and the experiments in Chapters..." M. Uses "my".
- c8:274, Appendix B. "The adaptive hybrid representation, the KIRBy repository, and the experiments of Chapters..." M.
- FENCE: keep one of the two. The style rule says "we". A contributions statement may be an exception to it.

**C1-07. The software contributions.**
- c1:153, Outline. "The software contributions of this work are NoiseInject..." M.
- c7:90, Ch7 Summary. "Chapter~\ref{chap:robustness} tells a reader which model to choose..." M. Its last two sentences.
- c8:215, Appendix B. "Three pieces of software lie behind this thesis." M.
- FENCE, low value. Each copy points onward, and none repeats details.

**C1-08. Kramer: repeat pKi measurements disagree by 0.54 log units.**
- c1:6, Ch1 prose. "QSAR models are of course, sensitive to the quality of the input..." U.
- c4:171, "Molecules with Identical Feature Vectors". "where $n$ is the number of molecules, $G$ runs over the tied..." M-block.
- c5:1519, Ch5 Limitations. "The adaptive hybrid representation has several limitations..." M.
- Canonical: c1:6.
- c4:171 uses the number as a working threshold. Adding "(Chapter~\ref{chap:intro})" would be optional and tiny. Leave it.
- c5:1519 restates the number and derives R² ≈ 0.81 from it. FENCE.

**C1-09. Kolmar: models treat a label as its true value.**
- c1:6. U.
- c2:60. Ch2.
- c2:65. "One key issue with QSAR models is the standard assumption that labels..." Ch2, U.
- FENCE, for information only. Every copy is locked.

---

## c4_flaws.tex

**C4-01. Definition of the Tanimoto coefficient.**
- c2:118, the kernel in `eq:tanimoto_kernel`. Ch2.
- c2:172--177, `eq:tanimoto`, with the reading "bits set in both divided by bits set in either". Ch2.
- c4:253, `sec:flaws:similarity`. "Chapter~\ref{chap:background} writes out the Tanimoto coefficient $T$ as the ratio inside the..." M. Re-derives T from the kernel and restates the in-words definition.
- Canonical: `eq:tanimoto`.
- c4:253: replace the first sentence with "The Tanimoto coefficient $T$ is defined in Equation~\ref{eq:tanimoto}." Keep "The corresponding distance is $1-T$ \citep{Bajusz2015}..." and the pointer to `sec:flaws:constant`.
- OBVIOUS.

**C4-02. Near-constant features matter more than constant ones, and constant features add nothing to the distance.**
- c4:70, `sec:flaws:constant`. "Which of the two counts matters depends on the distance. A bit..." M-block. Its last sentence: "The argument of this chapter therefore rests on the near-constant count rather than the constant count, and Section~\ref{sec:flaws:similarity} shows the effect on distances."
- c4:262, `sec:flaws:similarity`. "Every irrelevant feature that varies adds one further term..." U. Ends "This is why the near-constant count in Section~\ref{sec:flaws:constant} matters more than the constant count."
- Canonical: c4:262 for the conclusion. c4:70 keeps its worked numbers (4.36 and 37.56).
- c4:70: delete the last sentence.
- OBVIOUS.

**C4-03. Irrelevant features dominate the distance and grow linearly, restated inside the exploratory-study section.**
- c4:414, `sec:lnl_motivation`. M-block. Two sentences. The first is "When a large proportion of features is uninformative, similarity measures become dominated by them." It restates c2:191--193. The second is "Each uninformative feature that varies adds one more term to the distance, so their share of it grows linearly with their number." It restates c4:262 (U).
- c4:414: replace both sentences with a back-reference, "(Section~\ref{sec:flaws:similarity})".
- OBVIOUS.

**C4-04. The two possible reasons the distance fails.**
- c4:414, `sec:lnl_motivation`. "There are two possible reasons for the poor performance of the distances." M-block.
- c4:481, Ch4 Conclusions. "There are two possible reasons for the failure of the distance. A..." M-block. The same two reasons and the same "evidence does not separate them".
- Canonical: c4:414.
- c4:481: cut to one sentence, for example "Section~\ref{sec:lnl_motivation} gives two possible reasons for this failure, which our evidence does not separate; the requirements below address the second."
- OBVIOUS.

**C4-05. The scope caveat on the exploratory study: one dataset, one property, Gaussian noise, and Chapter 6 repeats it.**
- c4:412, `sec:lnl_motivation`. "The study used one dataset, one property and one noise type, namely..." M-block.
- c5:137, "The Adaptive Hybrid Representation". "Selecting useful features matters, but filtering out useless ones matters just as much..." M. Sentences 3--5 restate the result and the caveat.
- c4:479, Ch4 Conclusions. Its last sentence. M-block.
- c4:460, `sec:flaws:neighbors`. Its last sentence. M-block.
- Canonical: c4:412.
- c5:137: keep "In an earlier exploratory study, the noise handling methods that used a distance between molecules performed worst (Section~\ref{sec:lnl_motivation})." Delete "That study served as motivation only..." and "Chapter~\ref{chap:similarity} addresses the label-noise question...". OBVIOUS, but see also C5-P08.
- c4:479 and c4:460 belong to C4-06 and C4-10.

**C4-06. Why neighbor-based cleaning failed: noise was added to every label, so the neighbors are noisy too.**
- c4:431, `sec:flaws:neighbors`. "The method requires both that the distance picks the right neighbors and..." M-block. Clause 2.
- c4:456, `sec:flaws:neighbors`. "Judging a label by its neighbors usually made the model worse and..." M-block. Also gives "+17.2%".
- c4:458. "Overall, and in agreement with Section~\ref{sec:lnl_motivation}, using a sample's neighbors to determine..." M-block. Sentence 1 restates c4:456. Its last sentence repeats the +17.2% best configuration.
- c4:460. "The hybrid in that study combined PDV and MHG-GNN at 100 features..." M-block. "Distance-based methods remain challenging under Gaussian noise regardless of representation, which suggests that the core issue is the noise distribution..." restates c4:458.
- c4:479, Ch4 Conclusions. M-block.
- c6:184 and c6:293. See C6-03.
- Canonical: c4:456.
- c4:458: delete sentence 1, and cut the last sentence to the one new number (+11.4% at σ = 0.3).
- c4:460: delete the "Distance-based methods remain challenging..." sentence.
- OBVIOUS.
- c4:479: FENCE, since this is a conclusions restatement (see C4-10).

**C4-07. Stereoisomers: common fingerprints ignore stereochemistry (orsi2024map4c), and stereoisomers can differ in activity (Nguyen2006).**
- c4:29, Ch4 Introduction. U.
- c4:159, "Molecules with Identical Feature Vectors". "Many fingerprints are computed without stereochemistry..." U. Both citations again.
- c4:20, Ch4 opening. M. "mostly stereoisomers".
- FENCE. Both full copies are unmarked.

**C4-08. Bajusz: the choice of similarity coefficient matters little.**
- c2:158. Ch2.
- c4:27, Ch4 Introduction. "One of the fundamental assumptions in cheminformatics..." U.
- c4:250. "The standard way of comparing two small molecules is the Tanimoto coefficient..." U. Uses a different Bajusz claim.
- FENCE, for information only.

**C4-09. Definition of the similar property principle.**
- c2:164. Ch2.
- c2:157. Ch2. The breakdown.
- c4:27. U.
- c6:189, "Sample Removal Methods". "Distance-based approaches rely on the assumption that similar molecules have similar properties \citep{Cherkasov2014}." U.
- FENCE, for information only. This makes four statements, all locked, and c6:189 cites a different source.

**C4-10. The chapter's own numbers stated three times: opening, section and conclusions.**
- ECFP4 on hERG Ki, 1,815 of 2,048 features near-constant: c4:20 (M), c4:61 (M-block), c4:474 (M-block).
- All 20 configurations with the lowest retention used a distance: c4:20, c4:408 (M-block, canonical), c4:479.
- The all-pairs correlation was below 0.10 in 44 of 70 combinations: c4:284 (canonical) and c4:476.
- ECFP4 ties 56 hERG molecules: c4:174 and c4:474.
- FENCE. The opening summary and the conclusions both carry the numbers. Proposal: keep the numbers in the conclusions, which already give section references, and state the opening summary in words.

**C4-11. Requirements for a better representation, in prose and in a table.**
- c4:483. "A representation that avoids these flaws would need five properties..." M-block. Five properties.
- c4:491 and Table `tab:flaws:requirements`. M-block. Four flaw-to-requirement rows.
- c7:79, last sentence. M.
- c5:20, sentence 2. M.
- FENCE: keep the prose or the table. The two disagree on the count, five against four.

**C4-12. Cliffs can be artifacts of the representation.**
- c4:32 and c4:35 (Medina-Franco). U.
- c4:191. "\citep{Guha2008}. A tied pair has similarity 1..." M-block. "Some cliffs are thus artifacts of the representation."
- FENCE, low value. c4:191 turns the literature claim into a measured one, so I recommend keeping it.

**C4-13. Cortes 2015: twelve algorithms on twelve bioactivity datasets with simulated noise.**
- c2:352. Ch2.
- c2:104. Ch2, on the SVM.
- c1:23. M. Makes a different point, that the representations were fixed.
- c4:404, `sec:lnl_motivation`. "LNL methods and the five groups of \citet{Song2022} are described in Section~\ref{sec:lnl_background}..." M-block. Sentence 2 restates the design and then adds new results (gradient boosting worst, SVM and GP sensitive).
- Canonical for the design: c2:352.
- c4:404: shorten the design clause and keep the results.
- OBVIOUS, low value.

**C4-14. Two adjacent unmarked paragraphs each point to the exploratory study.**
- c4:381. "We have identified flaws in the representations themselves..." U. Ends "...as desribed in Section~\ref{sec:lnl_motivation}". Also uses "I".
- c4:383. "Section~\ref{sec:lnl_motivation} reports the exploratory LNL study, in which every configuration that used..." U.
- FENCE. Both are unmarked.

---

## c5_hybrid.tex

### Two versions of the same paragraph

In each pair, the first paragraph is marked `% TODO: review...` and the second `% TODO: REVIEW MODIFICATION`, and both render. In every pair the second is the later version. "Missing" means facts or citations in the old version that are absent from the new one and from the rest of the thesis.

| id | old (delete) | new (keep) | Missing from the new version | class |
|---|---|---|---|---|
| C5-P01 | c5:40 "The majority of the work in this space deals with combining relatively..." | c5:43 | nothing | OBVIOUS |
| C5-P02 | c5:47 "So far, the most common strategy for combining feature sets has been..." | c5:51 | **\citep{Beaude2025}** for cross-attention. c5:51 names cross-attention without a citation, and Beaude2025 appears nowhere else. Move the citation into c5:51 when deleting c5:47, and keep the inline `% TODO: REVIEW MODIFICATION` comment. | OBVIOUS |
| C5-P03 | c5:54 "In practice, most research groups default to one representation and ignore..." | c5:57 | **\citep{FernandezDelgado2014}** for "kNN and SVM are affected most severely". c5:57 replaced it with a reference to Chapter 4, and the citation appears nowhere else. Optionally add it to c5:57. | OBVIOUS |
| C5-P04 | c5:60 "The pace of new model development shows no sign of slowing..." | c5:63 | nothing | OBVIOUS |
| C5-P05 | c5:69 "For small molecules alone, the landscape now includes extended-connectivity fingerprints..." | c5:72 | c5:72 replaces a list of representation families with pointers to the tables. Lost citations that appear nowhere else: **Zeng2022** (image encodings), **Boiko2025** (stereoelectronics graphs), **Olsen2022** (AbLang). The El Nahhas ensemble result (1.2% AUROC) moved to c5:87, so it is not lost. | OBVIOUS, but check whether she wants the three citations back |
| C5-P06 | c5:75 "Concatenation, the most common method for combining feature sets, gives..." | c5:78 | The **Tishby2015** information-bottleneck argument ("the budget forces the model to keep only what matters"), which appears nowhere else. The multiple kernel learning, pairwiseMKL and consensus content is already in c5:51 and c5:87. | FENCE: keep or drop the information-bottleneck argument |
| C5-P07 | c5:129 "Here, we present KIRBy (Knowledge Integration by Representation), a framework..." | c5:132 | nothing. c5:129 uses the repository name for the method, and its NGBoost claim was flagged by her as invented. c5:132 corrects both. | OBVIOUS |
| C5-P08 | c5:134 "But as important as the selection of useful features is..." **U** | c5:137 M | c5:137 adds only the exploratory-study sentences, which repeat Chapter 4 (C4-05). | FENCE: the original is unmarked. The only allowed edit is to delete c5:137. c5:134 uses "I" and "kitchen sink". |
| C5-P09 | c5:208 "Given $K$ frozen representation matrices..." | c5:211 | nothing. c5:208 wrongly says the budget is allocated "according to baseline performance" for every method. | OBVIOUS |
| C5-P10 | c5:214 "Inputs from different sources differ in scale, distribution, and missingness..." | c5:217 | nothing. c5:214 wrongly says missing values are median-imputed for the hybrid. | OBVIOUS |
| C5-P11 | c5:243 "To be able to better examine the selected features..." | c5:246 | nothing. The step size is in Listing `lst:blueprint`. | OBVIOUS |
| C5-P12 | c5:295 "The blueprint does not store selected feature indices..." **U** | c5:298 M | nothing. The two contradict each other: c5:295 says a refit "re-runs the feature selection process deterministically", while c5:298 and c5:246 say a rebuild can pick other features. | FENCE: the original is unmarked |
| C5-P13 | c5:300 "It also provides a registry of selected features..." **U** | c5:303 M | nothing. c5:303 corrects SMARTS to SMILES fragments for ECFP bits. | FENCE: the original is unmarked. See also C5-S25. |
| C5-P14 | c5:566 "When I began to implement this idea, I expected a straightforward way..." (REVIEW MODIFICATION) | c5:309 (REVIEW MODIFICATION) | nothing. The two are near-identical, and both are revisions. | OBVIOUS: delete c5:566 and keep c5:309, which opens "The Search Space". Both use "I". |
| C5-P15 | c5:333--334 "Not all features from a given base representation are equally informative..." | c5:337 and c5:340 | nothing. Strobl2007 and Ancona2018 are both in c5:337, and the "no clear winner" content is in c5:340. | OBVIOUS |
| C5-P16 | c5:344 "We score features using LightGBM gain, DeepLIFT, and TreeSHAP. ... which...." (a stub) | c5:347 | nothing | OBVIOUS |
| C5-P17 | c5:368 "When searching for the features to use in the hybrid, thse feature importance scoring methods are swept." **U** | c5:371 M | nothing | FENCE: the original is unmarked. Either way, c5:371 should lose its pruning sentences (C5-S07). |
| C5-P18 | c5:415 "The allocation strategy divides the total feature budget among sources..." | c5:418 | nothing | OBVIOUS |
| C5-P19 | c5:446 "The four fixed-share methods are variants of performance-weighted allocation..." | c5:449 | nothing. Its caveat (a strong representation may rely on a few features) is in c5:470. | OBVIOUS |
| C5-P20 | c5:452--460, numbered list of five steps (TODO review) | c5:464, together with c5:223--229 | nothing. The list restates the formal steps of "From Base Representations to a Hybrid", and c5:464 already points there. | OBVIOUS |
| C5-P21 | c5:467 "This method cannot adapt allocation based on feature complementarity..." | c5:470 | nothing | OBVIOUS |
| C5-P22 | c5:534 "Features that improve predictions under RF may not necessarily help a kNN..." | c5:537 | nothing | OBVIOUS |
| C5-P23 | c5:540 "This internal evaluator is used to score each representation..." | c5:543 | nothing | OBVIOUS |
| C5-P24 | c5:551 "The full search tests all combinations of budgets, allocations..." | c5:554 | nothing | OBVIOUS |
| C5-P25 | c5:559 "The complete search result is cached to disk. The search runs once..." | c5:562 | nothing | OBVIOUS |
| C5-P26 | c5:569--570 "We ran a series of experiments to determine the best variables..." (unfinished sentence) | c5:573 | nothing | OBVIOUS |
| C5-P27 | c5:590 "Some features are too low-dimensional or too specialized to take part..." | c5:593 | nothing | OBVIOUS |
| C5-P28 | c5:596 "Augmentation features range from smaller fingerprints like MACCS or Coloumb..." | c5:599 | nothing | OBVIOUS |
| C5-P29 | c5:686 "In adaptive pruning, every configuration is first scored on the first two..." | c5:689 | nothing | OBVIOUS |
| C5-P30 | c5:753 "Different biological domains have established different conventions..." (lists the metrics) | c5:755 | nothing. The metrics are in c5:1140 and Table `tab:datasets`. | OBVIOUS |
| C5-P31 | c5:758 "No random splits are used in this chapter. Random splitting inflates..." | c5:760 | The reason for not splitting randomly, with **Yang2019** and **Wallach2018**. Wallach2018 appears nowhere else. Notin2023 and Vieira2026 are also in c5:874. c5:760 points to Chapter 2 "Evaluation methodology", which does not say this (see "Four things to know", point 2). | FENCE |
| C5-P32 | Table `tab:datasets`, c5:762--804 (TODO review) | Table `tab:datasets_modified`, c5:806--847 (REVIEW MODIFICATION) | The second table repeats every column and adds one column of RESULT NEEDED placeholders for the best single representation. All 7 references in the thesis point to `tab:datasets`, and none points to `tab:datasets_modified`. | FENCE: keep one table. If the modified one is kept, give it the label `tab:datasets`. |
| C5-P33 | c5:863 "The hybrid construction step uses a nested internal train/validation split..." | c5:865 | nothing. The content is in c5:220. | OBVIOUS |
| C5-P34 | c5:892 "For every data set in this study, we pre-selected ten base representations..." **U** | c5:894 M | nothing. c5:894 says "at most ten". There is no blank line between them, so they render as one paragraph that says the same thing twice. | FENCE: the original is unmarked |
| C5-P35 | c5:1126 "For a fair comparison, the pool of downstream models must cover a range..." | c5:1128 | c5:1128 drops the list of all 11 available models (CatBoost, NGBoost, a Monte Carlo dropout BNN and a 1D CNN) and the claim that "we use the full set for the model-matching and noise experiments". | OBVIOUS, but see C5-S10: c5:1128 is itself almost fully repeated |

### Other repetition in Chapter 5

**C5-S01. The list of the six search settings.** The settings are downstream model, budget, representation count, importance method, allocation method and internal evaluator.
- c5:315, "The Search Space". "Six settings are searched, namely..." M. Also Table `tab:search_tiers`.
- c5:12, chapter opening. M.
- c5:146, last two sentences. "This design leaves six choices open..." M.
- c5:309. M.
- c5:554. M.
- c4:489. M-block.
- c7:132, first sentence. "The settings of the hybrid search are the downstream model, the budget..." M.
- Canonical: c5:315 and Table `tab:search_tiers`.
- c7:132: cut to "The six settings of the hybrid search (Table~\ref{tab:search_tiers})". OBVIOUS.
- c5:146: the last two sentences can go. OBVIOUS, low value.
- c4:489 is the bridge out of Chapter 4. FENCE.

**C5-S02. Definition of the hybrid: a budget of features, for one dataset and one downstream model, used as a model input rather than a model.**
- c5:143, "Formally, an adaptive hybrid representation is a feature set built by..." M. Canonical.
- c5:12. M. The chapter opening, so keep it.
- c5:23, sentence 2. "Each hybrid draws on base representations of the same molecules and is specific to one dataset and one downstream model." M.
- c6:40, last two sentences. "Chapter~\ref{chap:method} introduced the adaptive hybrid representation, a budget of features... The hybrid is therefore an input to a model and leaves the model itself unchanged." M.
- c7:83, c7:318. M. Summary and conclusions.
- c5:23: delete sentence 2. OBVIOUS.
- c6:40: cut to "Chapter~\ref{chap:method} introduced the adaptive hybrid representation." OBVIOUS.
- c7:83 and c7:318: FENCE, see C1-03.

**C5-S03. Every feature traces back to its representation, and the best configuration is kept for each pair of downstream model and budget.**
- c5:93, last sentence. M.
- c5:156. M. Canonical for traceability.
- c5:197, the text after the figure. "The search is a grid in which every combination of setting values is built..." M. Repeats c5:232, c5:312, c5:686 and the figure caption.
- c5:211, last sentence. M.
- c5:232. M. Canonical for "winner per pair" and for the nested scoring.
- c5:562. M. "The chosen configuration... is saved as a blueprint."
- c5:197: delete the four sentences and keep the figure. OBVIOUS.
- c5:211: delete the last sentence. OBVIOUS.
- c5:562: delete that sentence. OBVIOUS.

**C5-S04. How the allocation methods work, described inside the literature review.**
- c5:93, sentences 4--6. "Four of the five allocation methods in its search give each representation a share... The fifth, the global pool..." M.
- The allocation section: c5:418, c5:476, c5:519 and Table `tab:alloc_methods`.
- c5:93: replace those sentences with a pointer to "Feature Allocation Methods". Keep the group-lasso contrast.
- OBVIOUS, low value.

**C5-S05. Importance and standalone scores are computed once per fold and reused by every configuration.**
- c5:337, sentence 8. M.
- c5:464, last sentence. M.
- c5:562, sentences 2--3. M. Canonical.
- c5:337 and c5:464: delete those sentences. OBVIOUS.

**C5-S06. A fast search can later be extended to a full one, because each score is stored by its settings.**
- c5:562, sentence 4. M.
- c5:647, last two sentences. M. Canonical.
- c5:562: delete the sentence. OBVIOUS.

**C5-S07. Pruning described in the importance-method section.**
- c5:371, sentences 2--3. "Adaptive pruning is off unless a search requests it. When it is on, importance methods that rarely give the best configurations..." M.
- c5:689, c5:692 and Table `tab:pruning_rules`. c5:692 repeats "Pruning is off unless a search requests it".
- c5:371: keep sentence 1 and the pointer to "Fast tier and adaptive pruning", and delete the rest. OBVIOUS.

**C5-S08. The pruning cut-offs and their removals, in prose and in a table.**
- c5:689. M. States each rule's cut-off, which is also in Table `tab:pruning_rules`.
- c5:726, sentences 6--8. "The chosen rules removed the count of 5 representations on LogD..." M. The same as the table's last column and its checkpoint row.
- c5:726: delete sentences 6--8 and keep the TreeSHAP run-time share and the run times. OBVIOUS.
- c5:689 against the table: FENCE (keep the rules in prose, the table, or both).

**C5-S09. The internal-evaluator list, and the claim that importance depends on the model (Dong2020).**
- c5:315. M. Lists the evaluators and cites Dong2020 for the importance method.
- c5:537. M. Canonical for the evaluator. Also cites Dong2020.
- c5:576, sentence 1. "We search four values of the internal evaluator... and the fast tier searches the same four." M. The same as c5:537 and the table.
- c5:582, sentence 1. "Variable importance depends on the model \citep{Dong2020}..." M. The same as c5:537.
- c5:576: delete sentence 1 and start at "The matched evaluator is an evaluator of the same model family..." OBVIOUS.
- c5:582: delete sentence 1. OBVIOUS.
- Whether c5:315 keeps its evaluator list: FENCE, low value.

**C5-S10. The downstream models, the Gaussian process rule, the size caps and "no per-dataset tuning".**
- c5:318. M. Sentences 2--3 give the Gaussian process rule: under 1,000 molecules, three datasets qualify, and it runs on humanness only.
- c5:1128. M. The size caps, the fixed hyperparameters, and the Gaussian process on humanness.
- c5:1131. M. Canonical for the Gaussian process, with the three small datasets and the run-time reasons.
- c5:1134. M. Canonical for the caps and tuning.
- Table `tab:search_tiers`, footnotes a and b.
- c5:318: delete sentences 2--3. OBVIOUS.
- c5:1128: after the old version c5:1126 is deleted, almost every sentence of c5:1128 is in c5:1131 or c5:1134. Delete it, or keep only its first sentence about the range of inductive biases. OBVIOUS.

**C5-S11. The splitting rules: outer folds are group-aware, stratified for classification, 5 by default, and never random.**
- c5:220, "From Base Representations to a Hybrid". M.
- c5:737, "Methods: Experimental Setup". M.
- c5:760. M.
- c5:860, "Splitting rules". M.
- c3:39. Ch3, U. The scaffold definition.
- FENCE. Choose one home for the splitting rules, most naturally c5:860 "Splitting rules", and reduce the others to a pointer. The Chapter 2 back-references in c5:760 and c5:860 do not land (point 2 at the top).

**C5-S12. The shared small-molecule curation steps, in Chapter 5 and in the appendix.** The steps are RDKit standardization, the median merge with the 1.0 pKi standard-deviation rule, InChIKey deduplication, clipping at the 1st and 99th percentiles, and the QM9 and trastuzumab subsamples.
- c5:850--856, "Pre-processing" list. M.
- c8:95, c8:98, c8:101, c8:104, c8:111, c8:183 and Table `tab:curation_counts`. M. Appendix A.
- c3:37. Ch3. The hERG protocol.
- Canonical: Appendix A. c8:44 already says the appendix does not repeat Chapter 5.
- c5:850: replace the list with one sentence pointing to `Appendix~\ref{app:datasets:smallmol}`.
- OBVIOUS, with the paper caveat.

**C5-S13. A score is the best of many configurations, so it is optimistic.**
- c5:232. M. Canonical, citing Varma2006 and Cawley2010.
- c5:1154, sentence 1. M.
- c5:1157, sentence 1. M. The two sentences are adjacent and say the same.
- c5:729. M. For pruning; different enough to keep.
- c5:1157: start at "Several experiments therefore use leave-one-fold-out reselection...". Varma2006 is already cited in c5:232. OBVIOUS.

**C5-S14. Per-fold scores and their spread are reported, with no tests.**
- c5:740, last sentence. M.
- c5:1143. M. Canonical.
- c5:883, last sentence. M.
- c7:254. M.
- c5:740: delete the last sentence. OBVIOUS, low value.
- The others are fine. c5:883 gives a reason, and c7:254 is a limitation.

**C5-S15. The Nogueira stability index.**
- c5:90. M. Mentions it.
- c5:1183. M. Canonical. The experiment.
- c5:1512, last two sentences. "Finally, stability can be measured directly with the Nogueira index..." M.
- c5:1512: replace the two sentences with a pointer to "Comparators, controls and ablations". OBVIOUS.

**C5-S16. Turkina2025: fusion by feature importance across six non-hashed fingerprints.**
- c5:37, literature review. M. Canonical.
- c5:1174, last two sentences. "In QSAR, \citet{Turkina2025} combined six non-hashed fingerprints..." M.
- c5:1174: delete the two sentences, or reduce them to "(see ``Combining representations and the alternatives'')". OBVIOUS.

**C5-S17. Neighbor-based label filtering (Wilson1972, Martin2021).**
- c4:427, `sec:flaws:neighbors`. M-block. Canonical, with Wilson1972 and Martin2021 in detail.
- c5:1201, importance-during-training experiment. M. Sentences 8 and 10 restate both citations.
- c2:352 and c6:46. These mention Wilson1972 only.
- c5:1201: replace the Wilson and Martin sentences with "Neighbor-based filtering is reviewed in Section~\ref{sec:flaws:neighbors}; here the molecule is down-weighted rather than removed." Keep "We found no QSAR study that down-weights..." and the Meta-Weight-Net sentence, since both are unique.
- OBVIOUS, with the paper caveat.

**C5-S18. The list of methods that use a distance.**
- c4:381, `sec:flaws:consequences`. U. Canonical.
- c5:165, sentence 2. "Chapter~\ref{chap:flaws} argues for several such uses, including noise detection, nearest-neighbor prediction..." M.
- c5:165: cut to "Chapter~\ref{chap:flaws} (Section~\ref{sec:flaws:consequences}) lists the methods that could use it." OBVIOUS.

**C5-S19. How the similarity search differs: the neighbor score replaces the model, the evaluator is dropped, and some datasets are excluded.**
- c6:102. M. Canonical.
- c6:115. M. Canonical for the exclusions.
- c5:149. M. Already defers to Chapter 6, so fine.
- c5:235, sentences 3--4. "This score replaces both the standalone score and the downstream-model score, so no downstream model or internal evaluator is used." M.
- c5:612, sentences 8--10. "Two datasets never run that objective. Breast cancer PAM50 subtype has 4 classes..." M. This **contradicts** c6:115, which excludes three datasets.
- c5:172, last sentence. M. Also restates the last sentence of c6:115.
- c7:274, last two sentences. M. See C7-05.
- c5:235: replace sentences 3--4 with "(Chapter~\ref{chap:similarity}, ``Similarity in a hybrid'')". OBVIOUS.
- c5:612: replace the three sentences with the same pointer. OBVIOUS, and it removes the inconsistency.
- c5:172: FENCE, low value. It is Chapter 5's claim about scope.

**C5-S20. Overview of augmentation, repeated in its own subsections.**
- c5:593. M. Sentences 6--9 describe ranking by correlation with out-of-fold residuals, keeping 0/5/10/20/40, and the uncapped and capped modes.
- c5:603, c5:606, c5:609. M. Canonical.
- c5:612, sentences 5--6. "Without augmentation, the width is at most $B$... Capped augmentation keeps the width at most $B$..." M. The same as c5:609 and Table `tab:aug_conditions`.
- c5:593: keep the definitions of donor and base and delete sentences 6--9. OBVIOUS.
- c5:612: delete sentences 5--6. OBVIOUS.

**C5-S21. The number of base representations per dataset (10, 8 for H3K4me3, 9 for TATA).**
- c5:324, sentence 2. M.
- c5:896. U. Canonical.
- c5:324: delete sentence 2. OBVIOUS, low value.

**C5-S22. Sort & Slice, which uses no labels, beat label-based substructure selection.**
- c2:36. Ch2.
- c5:99. M.
- c5:99: append "(Chapter~\ref{chap:background})" and keep the sentence. It is the premise of the paragraph's argument.
- FENCE, low value.

**C5-S23. Deng2023 (62,820 models).**
- c3:13. Ch3, U.
- c5:66. M. The same study, with an overlapping claim.
- c5:51. M. The "misleading" quote. Distinct.
- FENCE, low value.

**C5-S24. Zheng2000, selecting descriptors with a kNN model.**
- c4:292. M-block. In detail.
- c5:90. M. One clause.
- FENCE, low value. I recommend keeping both.

**C5-S25. Readable blueprint features, used for discovery and confirmation.**
- c5:159 and c5:162. M. Canonical.
- c5:300, last two sentences. U.
- c5:303, last two sentences. "This mapping adds a layer of explainability. It can be used to discover which features are useful for a task and to confirm..." M.
- c5:303: delete the last two sentences. OBVIOUS.
- c5:300: FENCE (see C5-P13).

**C5-S26. The blueprint fields, in prose and as a listing.**
- c5:246, sentences 5--10. M. Enumerates the fields.
- Listing `lst:blueprint`.
- FENCE. Keep the listing and trim the prose, or the reverse.

**C5-S27. Two placeholders ask for the same result: whether transferring settings from a large dataset to a small one works.**
- c5:298. M. "Which of the two approaches transfers better to a smaller related dataset is [RESULT NEEDED]".
- c5:1224. M. The transfer experiment, with the same comparisons.
- c5:298: delete the RESULT sentence. OBVIOUS, low value.

**C5-S28. Two placeholders ask for the same result: whether a concatenation scores between its two parts.**
- c5:57. M. "A concatenation of two representations scored between its two parts... on [RESULT NEEDED]". It also argues "combine and filter at the same time".
- c5:1210. M. The experiment, with the same argument.
- FENCE. c5:57 uses the result as motivation.

**C5-S29. The hybrid stops helping below about 200 samples.**
- c5:181. M. "[RESULT NEEDED ...; about 200 samples is expected]".
- c5:1519. M. States it as fact.
- c7:136. M. A back-reference, so fine.
- FENCE. The two copies disagree in certainty.

**C5-S30. One-hot sequence encodings are a poor choice.**
- c5:175. M. "We recommend language-model sequence embeddings over raw one-hot encodings."
- c5:1519, sentences 5--6. M.
- FENCE, low value.

**C5-S31. The Ambroise and Vabalas warnings about leakage in feature selection.**
- c5:217. M. With citations.
- c5:1512. M. In detail.
- FENCE, low value. Each copy serves its section.

**C5-S32. Noise levels for the Chapter 5 robustness experiment.**
- c5:1207. M. Lists the seven levels and the censoring levels.
- c3:71. Ch3.
- c6:166 and c6:169. M. Identical censoring list.
- FENCE, low value. Each chapter states its own setup.

---

## c6_similarity.tex

**C6-01.** Covered by C1-02: c6:43 repeats c6:12. Delete c6:43. OBVIOUS.

**C6-02. Finding, estimating and correcting noise are three different tasks.**
- c6:12. M. "These are distinct tasks, and a method can succeed at the first while failing at the third."
- c6:93. M. "A method can perform the first task well and still fail at the third."
- c6:93: delete sentence 2, and keep the definition of the three tasks. OBVIOUS.

**C6-03. Why the work moved from removal to weighting and smoothing.**
- c6:184, "Noise Mitigation Methods". "In addition to evaluating hybrid representation quality..." M (gap). "sample removal... limitations when applied to Gaussian noise affecting all samples".
- c6:293, "Label Smoothing and Sample Weighting". "Because all labels contain some noise, we focused on label smoothing in preference to removal." M.
- c4:456. M-block. Canonical for the reason.
- c6:293: delete sentence 1. OBVIOUS.
- c6:184: optionally add `(Section~\ref{sec:flaws:neighbors})`. OBVIOUS, low value.

**C6-04. Chapter 4's finding and the hybrid definition, restated in the introduction.**
- c6:40. M.
- Keep sentence 1 as a back-reference.
- Sentence 2 ("For example, ECFP4 bits compared by Tanimoto similarity mix a few features relevant...") restates Chapter 4. Delete it. OBVIOUS, low value.
- For the last two sentences, see C5-S02. OBVIOUS.

**C6-05. What each noise condition represents, with citations, restated from Chapter 2.**
- c6:166, sentence 5. "Censoring represents the range limits of assays... \citep{Svensson2025}" M.
- c6:172, sentences 2 and 4. "Outlier noise represents annotation and transcription errors \citep{Kalliokoski2013, Hampel2001}"; "Grouped-wider and grouped-shifted noise represent differences between laboratories \citep{Bentz2013}". M.
- Tables `tab:noise_sources` and `tab:noise_conditions`, c2:206. Ch2. Canonical.
- Delete, or point to `Table~\ref{tab:noise_conditions}`.
- OBVIOUS, low value. The copies are short, and she may like the reminder.

**C6-06. The exploratory study's recovery rate (R², one random forest on ECFP4).**
- c6:358, last sentence. "The exploratory study reused in Chapter~\ref{chap:flaws} computed the same ratio in $R^2$, with one random forest on ECFP4..." M.
- c4:433. M-block. Canonical. Already points to Chapter 6 "Recovery rate".
- c6:358: delete the last sentence. OBVIOUS.

**C6-07. Weighting features by importance in the distance.**
- c6:120, "Feature importance weighting". "Beyond simply using the feature importance scores for allocation..." U. $x\odot w$ under Manhattan, $\sqrt{w}$ under Euclidean.
- c5:1196. M. The square root of importance, averaged to 1.
- c4:373. U. Points to Chapter 6.
- FENCE. c6:120 is unmarked. The two may also disagree about when the square root is taken.

**C6-08.** Covered by C4-09: c6:189 restates the similar property principle. FENCE.

**C6-09.** Covered by C5-S19: the datasets excluded from the neighbor score appear in c6:115 (canonical), c5:612 and c7:274.

---

## c7_discussion.tex

**C7-01. Summary against Conclusions within Chapter 7.**
- c7:76, c7:79, c7:83, c7:87, Summary. M.
- c7:314 and c7:318, Conclusions. M.
- c7:318, sentence 2: "On a given dataset, most features of a general-purpose representation do not vary or do not relate to the property, yet a distance counts them all (Chapter~\ref{chap:flaws})." This is nearly verbatim c7:79 sentence 2 and c1:45.
- FENCE for the structure (see C1-03).
- If she keeps both sections, delete c7:318 sentence 2. OBVIOUS, low value.

**C7-02. Distance-based noise handling usually made models worse, stated four times in Chapter 7.**
- c7:79, sentence 3. M.
- c7:90. M. As a contribution.
- c7:203, sentences 1--2. "Several kinds of LNL method did not help. In the exploratory study, the methods that used a distance... did worst... neighbor-based label cleaning usually made the model worse..." M.
- c7:318, sentence 1. M.
- Canonical for the result: c4:408 and c4:456.
- c7:203: shorten sentences 1--2 to "Several kinds of LNL method did not help, beyond those of the exploratory study (Sections~\ref{sec:lnl_motivation} and~\ref{sec:flaws:neighbors})." OBVIOUS.
- c7:90 and c7:318: FENCE (a contribution line and the conclusion).

**C7-03. Uncertainty rises with noise but does not identify noisy labels.**
- c7:76. M.
- c7:249, sentences 9--10. M.
- c7:314. M.
- c1:138. M.
- Chapter 3.
- FENCE. c7:249 uses the finding for a recommendation and gives a back-reference.

**C7-04. The label-noise experiments are almost all regression; only the survey flips class labels.**
- c7:253, sentence 1, Limitations. M.
- c7:274, sentences 1--3, Future Directions. M.
- Keep the facts in c7:253.
- c7:274: start at "The first next step is classification. In classification, a noisy label is a flipped class..." and delete the repeated sentences about the survey.
- OBVIOUS.

**C7-05. AUROC is defined for two classes only, so nearest-neighbor prediction is not scored on multiclass data.**
- c7:274, last two sentences. M.
- c6:115. M. Canonical.
- c7:274: delete the two sentences. OBVIOUS.

**C7-06. Two placeholders ask for the same result: nearest-neighbor prediction in a hybrid against single representations.**
- c7:87. M. "For nearest-neighbor prediction, [RESULT NEEDED]".
- c7:199, sentences 4--5. M. "We also predicted the activity of each molecule from its five nearest neighbors... The hybrid scored [RESULT NEEDED]".
- FENCE, low value. c7:199 uses the result for its "simple model" argument, so keep one placeholder and refer to it from the other.

**C7-07.** Covered by C5-S01: delete the settings list in c7:132. OBVIOUS.

**C7-08. The same Chapter 3 evidence, stated in two paragraphs of the same chapter.**
- c7:143, sentences 4--5. M. "Tree and kernel models kept a similar \aucnorm{} on every representation, whereas... networks and NGBoost depended on the representation. On the three assay datasets, XGBoost and LightGBM ranked lower..."
- c7:195. M. "Their \aucnorm{} varied more across the six representations than that of any other model... Rankings also did not carry over unchanged to the assay datasets."
- FENCE, low value. The two paragraphs overlap without being identical.

**C7-09. Literature claims restated from other chapters.**
- c7:207. M. "Most were built... on large image datasets with flipped class labels \citep{Song2022}". This restates c4:404.
- c7:207. M. "tree models still outperform deep networks \citep{Grinsztajn2022}". This restates c5:1131.
- FENCE, low value. c7:207 is an argument that gathers them.

**C7-10. Why Gaussian noise is hardest, with two references to the same section.**
- c7:24. M. Cites "Comparison across noise types" twice in one paragraph, and restates the noise sources from Chapter 2 with citations (Bentz, Svensson).
- FENCE, low value. This is the chapter's opening hook.

**C7-11. New models enter the search as extra representations.**
- c7:325. M. Final paragraph.
- c5:63. M.
- c7:290. M. Direction 1.
- FENCE, low value. Echoing this in the closing paragraph is reasonable.

---

## c8_summary_appendices.tex

**C8-01.** Covered by C5-S12: the curation steps are in Appendix A and in the list at c5:850.

**C8-02. Data availability, stated twice.**
- c8:122, Appendix A section "The Chapter 3 versions...". "The datasets used in this study are publicly available..." U. It cites `Section~\ref{sec2}` for the hERG curation protocol, and `sec2` is the label on Chapter 2, where that protocol is not described.
- c8:264, Appendix B. "All datasets but one are publicly available..." M. More complete.
- FENCE. c8:122 is unmarked. It reads as left over from the paper and duplicates c8:264.

**C8-03. The Chapter 3 QM9 design: ten replicates of 10,000 molecules, split 80/10/10.**
- c8:125, sentence 4. "The QM9 experiments of Chapter~\ref{chap:robustness} draw ten replicates of 10,000 molecules..." M.
- c3:35. Ch3. Canonical.
- c8:125: replace the sentences with "The QM9 replicates of Chapter~\ref{chap:robustness} are drawn from the same 129,428 parsable molecules (Section~\ref{sec:robustness:methods})."
- OBVIOUS.

**C8-04. Description of NoiseInject.**
- c8:217, Appendix B. "The NoiseInject benchmarking framework, implementing all six regression and six classification..." U.
- c2:372 and c2:492. Ch2. Model-agnostic, MIT license, and wrappers for a Gaussian process, conformal prediction and Monte Carlo dropout.
- FENCE. Every copy is locked. c8:217 also names "noise degradation slope" as a metric, while c2:468 says the slope was replaced.

**C8-05. The QM9 pipeline, mentioned twice in Appendix B.**
- c8:215, sentence 3. M.
- c8:235. M.
- FENCE, low value.

**C8-06.** Covered by C1-05 (the hERG blocker sets) and C1-06 (her own work).

---

## Chapters 2 and 3 only (FENCE, for information; every copy is off limits)

- **X1. Tanimoto defined twice in Chapter 2.** c2:118 (the kernel, `eq:tanimoto_kernel`) and c2:172--177 (`eq:tanimoto`).
- **X2. Activity cliffs and the similar property principle, each defined twice in Chapter 2.** Activity cliffs: c2:159 and c2:183. The breakdown of the principle: c2:157 and c2:164--169. Bajusz is cited at c2:158 and c2:172.
- **X3. Aleatoric and epistemic uncertainty, defined twice.** c2:151 carries her TODO "this paragraph may need to go". c3:142 (U) defines the same two components in more detail, so c2:151 can go without losing anything.
- **X4. Uncertainty metrics, twice in Chapter 2.** c2:475--489 (the NoiseInject metrics) and c2:506--509 (Evaluation methodology: Spearman of uncertainty against error and against noise, coverage, ECE).
- **X5. How noise is drawn, in both chapters.** c2:359 against c3:81--84 (the Rust implementation). c2:389--397 against c3:106--113 (per-split draws, seeds, and affected molecules fixed across levels).
- **X6. The retention ratio and the trapezoid rule, explained twice.** c2:453--472 and c3:170--172. Chapter 3 cites the equation but explains both again.
- **X7. Repeats inside Chapter 3.**
  - The ratio of epistemic rise to aleatoric rise: defined at c3:198--199 and again at c3:458.
  - "GP has a single global observation-noise term": c3:401 and c3:492.
  - The Conclusions (c3:485--498) restate the sections, using names the Methods no longer use ("NDS", "six strategies", "three noise conditions"). This is pending her Chapter 3 update.
- **X8.** Kolmar (c1:6 and c2:60) and labels treated as true values (c1:6 and c2:65). See C1-09.

## Problems noticed along the way (not repetition, but they affect the fixes)

- As noted at the top, Chapter 2's "Evaluation methodology" does not define the splits, the scaffold key, the rule for fewer groups than folds, or AUROC, MCC and F1. Five Chapter 5 back-references rely on it: c5:760, c5:860 (twice), c5:868 and c5:1140.
- c4:358 says the eight distance metrics are compared in `sec:flaws:neighbors`. That section (c4:433) uses three distances and refers the eight-distance study to Chapter 6 "Sample removal methods". The prose there (c6:186--197) does not mention it.
- c5:612 says two datasets never run the neighbor objective, and c6:115 says three.
- c1:45 "such methods" has no antecedent, because the exploratory-study paragraph of the introduction exists only as notes.
- "I" appears in c1:171, c4:381 (U), c5:134 (U), c5:137, c5:309, c5:566 and c8:14.
- c5:890 (U) says every base representation "is never trained on the labels". c5:220 says "a representation learned from the labels is retrained inside each outer training fold".
