# Calibration guide: how much detail Dablander gives, and where Adelaide's thesis runs heavier

Sources read in this session:
- Dablander's DPhil thesis, `/Users/apunt/Documents/Theses/DPhil_Thesis_Markus_Dablander--Submitted.pdf`. That is 189 PDF pages, with Chapter 1 page 1 on PDF page 15. Text was extracted with `pdftotext`. I read all of Chapter 1, Chapter 2 up to the start of ECFPs, Chapter 3's overview, literature, methods and conclusions, Chapter 4's overview and methodology, Chapter 5's introduction, results ending and conclusions, and all of Chapter 7 and the Summary.
- Her thesis, as the split part files in `scratchpad/trim/parts/`. I read the prose lines of c1, c4, c5, c6 and c7 in full. I grepped c2 only for overlaps.

How the numbers were computed:
- Character counts for her chapters use non-comment lines only. They are roughly de-TeXed: `\cite{..}`, `\ref{..}` and `\label{..}` each become one character, other commands are stripped, and whitespace is collapsed. Tables and captions are included.
- Her page estimates use 3,000 characters per page, as requested.
- **Dablander's measured density is lower: about 1,900 to 2,200 characters per page.** For example, Chapter 1 is 15,348 characters over 7 pages. So her typeset page counts will be roughly 1.4 times the 3,000-character figures below. Comparing characters to characters is the fairer test, and both are given.

---

## 1. Dablander's lengths

**Total:** 153 pages of body (Chapters 1 to 7), about 291,000 characters. Front matter adds 14 PDF pages, the Summary of Research Contributions 3 pages and the bibliography 19 pages.

| Chapter | Pages | Characters |
|---|---|---|
| 1 Introduction (3 pages of introduction + 4 pages of "Outline of Thesis and Research Contributions") | 7 | 15,348 |
| 2 Molecular Representations and Featurisations (the background chapter) | 37 | 76,164 |
| 3 QSAR and activity-cliff study (results chapter, published paper) | 32 | 54,226 |
| 4 Twin neural network (results chapter) | 32 | 66,549 |
| 5 Sort & Slice substructure pooling (results chapter) | 33 | 60,278 |
| 6 Future Directions (two proposed methods, no results) | 8 | 12,286 |
| 7 Conclusions and Further Thoughts | 4 (2.5 of text) | 6,368 |

Section lengths inside chapters. Page ranges overlap by one page at each boundary.

| Section | Pages | Characters |
|---|---|---|
| Ch2 Overview | ~1.5 | ~3,900 |
| Ch2 Molecular graphs + SMILES | ~5 | ~12,400 |
| Ch2 Physicochemical-descriptor vectors | ~6 | ~17,000 |
| Ch2 ECFPs | ~8 | ~19,400 |
| Ch2 Graph neural networks | ~15 | ~33,200 |
| Ch3 Overview | ~3 | ~8,400 |
| Ch3 "Introduction to activity cliffs" (topic literature review) | ~3 | ~10,000 |
| Ch3 Experimental methodology | ~14 | ~27,100 |
| Ch3 Results and discussion | ~9 | ~14,500 |
| Ch3 Conclusions | ~2 | ~3,300 |
| Ch4 Overview | ~2.5 | ~7,100 |
| Ch4 Mathematical description (his new method) | ~12 | ~25,900 |
| Ch4 Experimental methodology | ~7 | ~17,600 |
| Ch4 Results | ~8 | ~21,200 |
| Ch4 Conclusions | ~2 | ~4,500 |
| Ch5 Introduction (includes its related work) | ~4.5 | ~12,300 |
| Ch5 Methods (definitions, 4 pooling methods, a proof) | ~14 | ~31,000 |
| Ch5 Experimental setup | ~3 | ~8,400 |
| Ch5 Results and discussion | ~8 | ~10,800 |
| Ch5 Conclusions | ~2.5 | ~6,400 |

**Methods sections run 27,000 to 43,000 characters per chapter.** That is 14 to 20 pages, and it includes full mathematical definitions and proofs of his own methods. His experimental setup proper (datasets, models, hyperparameters, training) is 3 to 7 pages per chapter.

He has no separate Limitations section anywhere, neither in the results chapters nor in Chapter 7. Future work is its own 8-page chapter.

---

## 2. Background on things that are not his contribution: one sentence and a citation

The pattern is consistent. Material that is his contribution gets pages. Standard tools get a name, an implementation citation and at most one line of explanation.

- **Standard ML models are never defined.** Random forest, k-nearest neighbours and MLP are not explained anywhere in the thesis. From Ch3 methods (p. 62):
  > "Each molecular featurisation was used as an input featurisation for three regression techniques: random forests (RFs), k-nearest neigbours (kNNs) and multilayer perceptrons (MLPs). The RF and kNN-models were implemented via scikit-learn [129] and the MLP-models via PyTorch [130]."
- **Metrics get a formula and one sentence each, inside the methods section.** There is no metrics section in the background. Ch3 p. 61:
  > "The MCC represents a summary statistic for the confusion matrix of a binary classification problem that is reasonably robust against imbalanced class labels."
- **Fingerprints outside his focus get one sentence each.** For example: "Well-known structural fingerprints include the 166-bit MACCS fingerprint [74] and the 881-bit PubChem fingerprint [75, 76]." ECFP gets 8 pages only because his Chapter 5 modifies ECFP. GNNs get 15 pages because his Chapters 3 and 4 use them as a method under study.
- **Hyperparameter search and training** take about 2 pages per chapter. They are listed plainly, with no justification of each choice.
- **Preliminary or design experiments are not reported in the main text at all.** Settings are stated. Where he justifies a setting, it is one clause, for example "d_crit = 1.5 ... since this value represents the middle point between the intervals".
- **Software behaviour is never described.** There is nothing on what the code refuses, warns about, caches or stores.

---

## 3. How he opens chapters and sections, and how he refers back

- **No chapter abstract.** A results chapter opens with a 3 to 5 line italic-style note on where it was published or presented. Then comes "Overview" or "Introduction".
- **Chapter Overview: 2.5 to 4.5 pages.** Its first paragraph recaps the previous chapter in 2 to 4 sentences. Examples: "In Chapter 2 we have given a detailed technical description of ... However, perhaps the most pressing question has not yet been addressed ..." and "In Chapter 3 we have seen that the utility of standard QSAR models for the detection of ACs is limited if the activities of both compounds are unknown ..."
- The Overview then states the question, the approach and the contributions. **It ends with a one-paragraph roadmap**: "We start off this chapter by ... We then present ... Finally, we summarise ..."
- **Back-references are short, often a bare section number in parentheses.** Ch3 p. 62:
  > "For molecular featurisation, we used PDVs (2.4) and ECFPs (2.5), both generated from SMILES strings (2.3), as well as GINs (2.6.3) on top of molecular graphs (2.2)."
  > "... were normalised ... via their componentwise cumulative distribution functions derived from the training set as explained in Section 2.4."
- **Where he does repeat himself, he flags it and keeps it brief.** Ch4 reuses Chapter 3's data and split and says so: "We will shortly repeat their definitions here in intuitive terms". That is a 4-bullet recap.
- **One place where he repeats more than her rule allows.** Ch4's Overview re-summarises the activity-cliff literature from Section 3.2 at almost the same length: the 15 methods, condensed graphs of reactions, and no comparison with a QSAR baseline. That is about half a page restated. **Do not use it as a model.** Her rule, a back-reference instead of a restatement, is stricter than his practice.

---

## 4. Related work: background chapter versus results chapters

- **His background chapter covers only the topic common to all chapters**: molecular representations and featurisations. Nothing on activity cliffs, twin networks, feature selection or pooling literature is in Chapter 2.
- **Topic-specific literature sits at the start of the results chapter that needs it, capped at a few pages:**
  - Ch3 Section 3.2, activity cliffs and activity-cliff prediction: about 3 pages, 10,000 characters.
  - Ch4 Overview, twin-network literature: about 1 page.
  - Ch5 Introduction, ECFP uses, GNN pooling, filtered fingerprints (Gütlein and Kramer) and MacDougall's near-identical method: about 1.5 to 2 pages inside a 4.5-page introduction.
- **Results sections contain almost no new literature.** They cite only to place a result, for example "These observations reinforce a growing corpus of literature that suggests ... [6, 7, 8, 10, 11, 12, 13]". That is one sentence.
- **Discussion of prior methods in his results chapters is about the methods he directly compares against**, such as filtering and mutual-information maximisation in Ch5. It is not a survey of adjacent families.

---

## 5. How Chapter 7 relates to the per-chapter conclusions

- Each results chapter ends with **Conclusions of 2 to 2.5 pages**. These restate the findings in words, with almost no numbers, and add the recommendation. Example from Ch3: "we recommend using GINs as an AC-classification baseline since such an agreed-upon benchmark is currently lacking".
- **Chapter 7 is 2.5 pages of text and contains no numbers.** It gives one paragraph listing the three projects as (i), (ii), (iii). Then it points back explicitly:
  > "Detailed conclusions from our main research projects can be found at the respective ends of Chapters 3 to 5."
- It then gives one sentence per chapter on what that chapter shows about the thesis-wide question. The rest (about 1.5 pages) is forward-looking thoughts that go beyond any single chapter.
- Results are therefore stated three times at three depths: the Ch1 outline (one paragraph per chapter, 4 pages in total), the chapter conclusions (2 pages), and Chapter 7 (one sentence each). **Chapter 7 does not re-summarise chapter by chapter with numbers, and it has no Limitations section.**

---

## 6. Her chapters 1, 4, 5, 6, 7 against this

### Totals

| Her chapter | Prose characters | Pages at 3,000 | Pages at Dablander density (~2,000) | Dablander's nearest equivalent |
|---|---|---|---|---|
| 1 Introduction | 13,400 | 4.5 | ~6.5 | Ch1: 15,300 characters, 7 pages |
| 2 Background (off limits) | 51,900 | 17.3 | ~26 | Ch2: 76,200 characters, 37 pages |
| 3 Robustness (off limits) | 53,600 | 17.9 | ~27 | Ch3: 54,200 characters, 32 pages |
| 4 Flaws of the similarity assumption | 58,600 | 19.5 | ~29 | a full results chapter (54,000 to 67,000 characters) |
| **5 Adaptive hybrid** | **188,700** | **62.9** | **~94** | his largest chapter, Ch4, is 66,500 characters **including results** |
| 6 Similarity and label noise | 20,900 | 7.0 | ~10 | Ch5's methods alone are 39,400 characters |
| 7 Discussion and Conclusions | 22,600 | 7.5 | ~11 | Ch7: 6,400 characters, 2.5 pages of text |
| Summary + appendices (c8) | 28,400 | 9.5 | ~14 | no appendices |

- Chapters 1 to 7 of hers already total about 410,000 characters. Dablander's total 291,000.
- **The results sections of Chapters 5 and 6 are still empty.** If Chapter 5's results reach Dablander's results length (11,000 to 21,000 characters), Chapter 5 alone will be about 200,000 characters.
- Chapter 5 is where the length is. Its "Methods: The Hybrid Search" plus "Methods: Experimental Setup" come to about 160,000 characters. That is **about 4 times** Dablander's largest methods section, and his includes proofs.

### Which paragraphs can be trimmed at all

A paragraph counts as marked if a TODO or CODE INPUT comment sits directly above it. My detection is approximate.

| Chapter | Marked prose characters | Unmarked (signed-off) prose characters |
|---|---|---|
| c1 | 10,100 | 3,300 |
| c4 | 5,900 | 52,700 (Ch4 is almost all signed off) |
| **c5** | **163,700** | **24,800** (nearly all of it in the representation tables under "Representations by Domain") |
| c6 | 15,100 | 5,700 |
| c7 | 22,400 | 100 |
| c8 | 24,500 | 3,900 |

**Chapter 5 is both the heaviest chapter and the one where almost everything is marked.** Chapter 4 has little that can be touched.

### The single largest source of length: Chapter 5 prints both versions of 23 paragraphs

In `c5_hybrid.tex`, 23 paragraphs appear twice. An original marked `% TODO: review` (or similar) is followed by a rewrite marked `% TODO: REVIEW MODIFICATION`. **Both are live prose, so both print.** The originals total about 15,900 characters, which is 5.3 pages at 3,000 per page and about 8 typeset pages.

Pairs, as the original's line in c5_hybrid.tex followed by its rewrite:

| Original | Rewrite | Original | Rewrite | Original | Rewrite |
|---|---|---|---|---|---|
| 39 | 42 | 206 | 210 | 539 | 542 |
| 45 | 50 | 213 | 216 | 549 | 553 |
| 53 | 56 | 242 | 245 | 568 | 572 |
| 59 | 62 | 331 | 336 | 588 | 592 |
| 68 | 71 | 342 | 346 | 595 | 598 |
| 74 | 77 | 412 | 417 | 685 | 688 |
| 128 | 131 | 444 | 448 | 749 | 752 |
| | | 466 | 469 | | |
| | | 533 | 536 | | |

Further doubled pairs whose comment tags differ, so the count above missed them. These are line numbers in the prose extract, which match c5_hybrid.tex lines:
- 309 and 566, "When I began to implement this idea", which is nearly word for word.
- 753 and 755, metric choice.
- 758 and 760, no random splits.
- 863 and 865, the internal split.
- 1126 and 1128, downstream models.
- 892 and 894, "pre-selected ten base representations".
- 344 and 347, the importance-method stub and its full version.
- Two dataset tables printing the same 16 rows: `tab:datasets`, and `tab:datasets_modified`, which only adds an empty "Best single representation" column.

In c7_discussion.tex, lines 253 and 274 both state the regression-only / classification-flip limitation.

Which version of each pair to keep is her choice, since the rewrites are AI edits. Either way, one of each pair goes.

### Chapter 1: right-sized

It is 4.5 pages against Dablander's 7, and the outline takes 1.7 pages against his 4. **Nothing is heavy.**
- The opening noise paragraph (c1 line 6) cites Kolmar2021, Kramer2012 and Kalliokoski2013 with numbers. Chapter 2 cites the same sources, but a short numbers-based motivation in Ch1 matches Dablander's practice.
- The origin story of the hybrid at c1 line 45 is the first of five tellings (see the repetition list below). It is the one to keep.

### Chapter 4: about right in total, but mostly signed off

At about 58,600 characters it is a normal results-chapter length for Dablander. What is heavier than he would give:
- **The chapter abstract (line 20), Introduction (lines 27 to 39) and Conclusions (lines 472 to 491) state the same headline numbers three times.** "1,815 of the 2,048 ECFP4 features" appears at lines 20, 61 and 474. Dablander has no chapter abstract. Only line 20 is marked.
- **"Distance Metrics for Continuous Molecular Descriptors" (lines 358 to 373).** This is an 8-item list with formulas for Euclidean, Manhattan, Chebyshev, cosine, correlation, Canberra, Bray–Curtis and Mahalanobis. Dablander gives standard metrics a formula and a line each, and only for metrics he actually uses. Line 373 says Chapters 5 and 6 use only four of these. Line 371 is marked.
- **Two literature paragraphs in the Introduction (lines 32 and 35), about 2,200 characters.** Dablander's closest equivalent is his Ch3 literature section (about 3 pages), so this is in range.
- "Exploratory Study of Learning with Noisy Labels", lines 402 to 414, about 1.9 pages. Line 404 re-describes Cortes2015. Chapter 2, line 265, also describes Cortes2015. Line 414 states "two possible reasons", and line 481 in the Conclusions states the same two reasons again.

### Chapter 5: heavier than Dablander at every level

Introduction: about 31,000 characters, or 10.3 pages at 3,000. **Dablander's Ch5 introduction, related work included, is 12,300 characters.**
- **Combining Representations and the Alternatives: about 17,300 characters, or 5.8 pages at 3,000.** Dablander's equivalent topic literature is 1.5 to 2 pages inside a results chapter, or 3 pages at most (his Ch3 Section 3.2).
  - About 6,000 of those 17,300 characters are the doubled pairs (lines 39 to 77).
  - What remains still surveys six families: concatenation, late fusion, intermediate fusion, multiple kernel learning, flat feature selection and PCA/PLS/t-SNE/UMAP. It adds a comparison table (`tab:hybrid_alternatives`) and a paragraph on the pace of foundation models (lines 66 to 72).
  - Dablander surveys only the methods he compares against. Here that would be ensembles, PCA, mRMR and recursive feature elimination, which are the comparators in "Comparators, Controls and Ablations".
  - Repeated inside the subsection: El Nahhas's 1.2% AUROC (lines 69 and 87), "over 40 molecular foundation models" (lines 69 and 72), and multiple kernel learning with pairwiseMKL (lines 51 and 75).
- **The Adaptive Hybrid Representation, line 132 (marked, about 1,700 characters).** It re-reports Chapter 3's NGBoost R² values per dataset to argue that preferences change with the task. Dablander would write one sentence and a reference to the Chapter 3 section. The paragraph also says NGBoost "appeared to prefer Avalon (0.402), with ChemBERTa at 0.404", which reads backwards. That is unverified against the source.
- Line 137 re-tells the exploratory noisy-label study, which Chapter 4 reports.
- **Blueprints and Their Chemical Interpretation (lines 156 to 165) and Beyond Small Molecules (lines 172 to 181), about 6,000 characters in total.** Methods detail is placed in the introduction. Blueprints are then described again in the methods (lines 243 to 303), and chemical readability a third time at line 303.

"Methods: The Hybrid Search": about 87,000 characters, or 29 pages at 3,000. Dablander's methods for a new method run 26,000 to 31,000 characters and include proofs.
- **From Base Representations to a Hybrid: 17,300 characters.** This is software-level detail Dablander never gives. Examples:
  - Missing values are set to 0 in one case and to the median in another (line 217).
  - A LightGBM pre-screen keeps 500 or 200 features (line 217).
  - A worked example of fold counts on Koenig expression (line 220).
  - The full 30-field blueprint listing (`lst:blueprint`), including a field "Left from a removed method".
  - A rebuild without labels "is refused" (line 298).
  - The explanation step's per-representation naming rules (line 303).
  Candidates to move to the Additional files, or cut, if she agrees.
- **Feature Importance Scoring: 11,900 characters.** It reports two preliminary design experiments with numbers (lines 374 to 383 and `tab:importance_wins`), plus the rules for retaining methods (line 380). Dablander reports no design experiments in the main text.
- **Search Configuration and Internal Evaluation: 12,300 characters.**
  - Line 582 is one paragraph of about 2,900 characters listing preliminary scores for the matched evaluator against the others, and it concludes that every gap is within the fold spread.
  - Line 579 describes what happens when a user supplies an unknown model. That is software behaviour.
  - Line 573 describes the design-experiment procedure.
- **Augmentation: 10,300 characters.** Line 612 alone is about 2,200 characters and mixes the three conditions, the nearest-neighbour objective's exclusions and a separate decoy experiment. The decoy experiment is described again at line 1204.
- **Fast Tier and Adaptive Pruning: 11,200 characters, with two tables.** This is about compute. Dablander covers compute in one clause: "in order to save computational resources", and "a runtime in the order of approximately 10-20 hours per model". The pruning calibration (line 726) gives per-dataset hours.

"Methods: Experimental Setup": about 73,000 characters, or 24 pages at 3,000. Dablander's experimental setup is 8,400 to 17,600 characters.
- **Datasets and Splitting: 13,500 characters.** It includes the duplicated dataset table and duplicated paragraphs on metrics, splitting and the internal split. Per-domain grouping (lines 868 to 883) is at the level Dablander gives his three datasets, but here it covers 16 datasets.
- **Representations by Domain: 20,400 characters.** This is mostly the six representation tables, and they are unmarked. Dablander prints one comparable table (his 200 descriptors). Only the prose around the tables is marked.
- **Models and Metrics: 7,400 characters.** It says again where each model is skipped: line 318 in the search space, and lines 1131 and 1134 here. The metric reasons are given at both 753 and 1140. Line 1131 explains Grinsztajn2022, and Chapter 7 line 207 cites it again.
- **Comparators, Controls and Ablations: 16,400 characters.** Eleven planned experiments are each specified in full before any result exists. Dablander's setup covers one comparison design per chapter.
- **Transfer, Mutational Scans and Separate Sites: 13,200 characters.**
  - Line 1230 is federated-learning background that belongs with Chapter 2's "Noise in molecular data", and it back-references there.
  - Line 1248 is a limitations paragraph.
  - Line 1251 lists three benchmarks "not run here". That is future work, and Chapter 7 has a Future Directions section.
- **Discussion (line 1512).** It repeats the selection-bias literature (Ambroise2002, Vabalas2019, Cawley2010) already cited at lines 217, 232 and 1157.
- **Limitations (line 1519).** It repeats the 200-sample threshold from line 181 and Kramer2012's 0.54 log units from Ch1 line 6 and Ch4 line 171. It also makes an uncited claim that "SchNet outperforms any frozen embedding by a wide margin".

### Chapter 6: shorter than Dablander's methods sections

At 20,900 characters with results empty, **it is not heavy.**
- The chapter abstract (line 12) and the Introduction (lines 40 to 43) restate Chapter 4's findings and the hybrid's origin, which is already told in Ch1, Ch4 and Ch5 (see the repetition list).
- Line 184, "our investigation evolved ... This motivated a pivot", narrates the research process. Dablander never does this.
- Line 197 lists eight further removal methods in one sentence, which matches Dablander's level.

### Chapter 7: about 3.5 times Dablander's

22,600 characters against 6,400.
- **The Summary section (lines 71 to 90) re-reports Chapter 3 with numbers**: 29.1% against 26.6%, 49.5% against 9.2%, median normalised AUC 0.813 against 0.951, and Spearman ±0.16. It also re-tells Chapter 4. Dablander's Chapter 7 has no numbers and points to the chapter conclusions instead.
- **"Simple Models and Small Feature Sets", line 195, is another Chapter 3 results paragraph**: NGBoost ranks, normalised AUC from 0.881 to 0.897, and VBLL at 0.924 and 0.926.
- **The Conclusions section (lines 314 to 325) summarises both halves again.** With the Summary, that makes two summaries in one chapter, besides the Ch1 outline and the chapter conclusions.
- The opening paragraph (line 24) and line 321 give the same advice: ask where the labels came from before choosing a method.
- **Limitations (lines 245 to 254) has no counterpart in Dablander's Chapter 7.** Lines 253 and 274 state the classification limitation twice.

---

## 7. The same statement made in several places

Each item names the places where one fact is stated. Keeping the first full statement and replacing the others with a back-reference is the pattern Dablander uses. Places in c2 or c3, or in unmarked paragraphs, cannot be changed.

1. **The hybrid was built for similarity and later worked as a model input.** Ch1 line 45, Ch4 line 487 (unmarked), Ch5 line 20, Ch6 line 12, Ch6 line 43. That is five tellings.
2. **Chapter 4's exploratory noisy-label result, that configurations using a distance did worst.** Ch4 lines 20, 383, 408 and 479. Ch5 line 137. Ch7 lines 79, 203 and 318.
3. **Chapter 3's headline results.** Ch1 line 138, Ch4 line 402 (unmarked), Ch5 line 132, Ch7 lines 76, 195 and 314.
4. **"Irrelevant features' share of the distance grows linearly with their number."** Ch1 line 45, Ch4 line 262 (unmarked) and Ch4 line 414 (unmarked).
5. **"Two possible reasons" for the distance failing.** Ch4 lines 414 and 481, both unmarked.
6. **Repeat pKi measurements disagree by about 0.54 log units (Kramer2012).** Ch1 line 6, Ch4 line 171, Ch5 line 1519, and Chapter 2.
7. **Selection bias from choosing features or configurations on the same folds** (Ambroise2002, Vabalas2019, Cawley2010, Varma2006). Ch5 lines 217, 232, 1157 and 1512.
8. **The Cortes2015 study (12 algorithms on 12 bioactivity datasets).** Ch1 line 23, Ch2 line 265, Ch4 line 404.
9. **Wilson1972 nearest-neighbour editing.** Ch2 line 265, Ch4 lines 35 and 427, Ch5 line 1201, Ch6.
10. **Grinsztajn2022, trees beating networks on tabular data.** Ch5 line 1131 and Ch7 line 207.
11. **The GP runs on humanness only, with the reasons.** Ch5 lines 318 and 1131, the footnote of `tab:search_tiers`, and line 1128.
12. **Where each downstream model is skipped by molecule count.** Ch5 lines 1128 and 1134, and `tab:search_tiers`.
13. **Metric per domain and why.** Ch5 lines 753 and 1140.
14. **Blueprint contents and chemical readability.** Ch5 lines 156 and 159 in the introduction, lines 243, 246, 298, 300 and 303 in the methods.
15. **The decoy experiment for augmentation.** Ch5 lines 612 and 1204.
16. **The advice to ask where the labels came from first.** Ch7 lines 24 and 321.
17. **The classification limitation.** Ch7 lines 253 and 274.

---

## 8. Style-rule hits found in marked paragraphs while reading

These are not length issues. They are recorded here so editors who touch these paragraphs see them.
- **"I" instead of "we":** Ch5 line 137 ("I decided"), Ch5 lines 309 and 566 ("When I began"), and Ch1 line 171 ("my own work"). Ch4 line 381 ("When I was exploring") is also "I", but it is unmarked.
- **"read-across" and "SAC":** neither appears in the prose I read.
