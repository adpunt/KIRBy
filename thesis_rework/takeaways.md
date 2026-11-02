# Main takeaways of the thesis, for the trimming editors

Sources read in full for this file: `brainDumpOutline.txt`, the six `---- TAKEAWAYS ----` blocks in
`thesis.tex.bak_pre_trim_2026-09-30`, `c1_intro.tex`, `c7_discussion.tex`, the Summary of Research
Contributions in `c8_summary_appendices.tex`, and all prose of `c4_flaws.tex` and `c6_similarity.tex`.
Read in part: `c5_hybrid.tex` (the opening, the Introduction, Discussion, Limitations and the Conclusions
notes; the Methods only by heading), the Chapter 2 Overview and Conclusions, the Chapter 3 Introduction
and Conclusions, and `THESIS_OUTLINE.md` (the Chapter 1, 3.9, 4, 5, 6 and 7 entries).

Every number below is copied from thesis text. Nothing here has been checked against code or results.

How to use this file:
- A marked paragraph that serves no takeaway is a trim candidate.
- A marked paragraph that is the only prose statement of a takeaway must be kept.
- "Canonical home" is the one place a takeaway is stated in full. Elsewhere it should be one clause plus
  `Chapter~\ref{}` or `Section~\ref{}`.
- "M" means the paragraph is marked (TODO or CODE INPUT directly above it). "U" means it is unmarked. U
  paragraphs are signed off and cannot change, even when they repeat something.

Chapters 5 and 6 have almost no results prose yet. Their Results sections hold notes only, with
`[RESULT NEEDED]` placeholders. The takeaways for those chapters come from the TAKEAWAYS blocks and the
notes, and they are marked "(pending)" or "(preliminary)" below.

---

## Chapter 1: Introduction

**Its job.** It sets up the problem: QSAR labels carry experimental noise that is not simple Gaussian
error. It states the questions for both halves and the path from the first half to the second. It
gives a one-paragraph outline of each chapter.

**Main takeaways**
1. QSAR labels are noisy. Repeat measurements disagree by about 0.68 log units for pIC50 and 0.54 for
   pKi. Most noise studies assume Gaussian noise and test representation or model in isolation.
   *Stated in:* opening prose (U, first two paragraphs) and the questions paragraph (M).
   *Canonical home:* Chapter 1 opening (U). Chapter 2 "Noise in Molecular Data" owns the detailed
   taxonomy.
2. The first half asks four questions: how representation and architecture split the variance, what
   probabilistic models add, whether the type of noise matters, and what noise does to uncertainty.
   *Stated in:* the paragraph labelled `sec:intro:questions` (M).
   *Canonical home:* here. Chapter 3's Introduction already points back to it (U).
3. Origin of the second half. The distance-based LNL methods failed. The hypothesis is that irrelevant
   features dominate the distance. A feature set chosen for the task was built to fix the distance. It
   then turned out to work as a model input on its own.
   *Stated in:* "We hypothesized that the fault lay in the distance..." (M).
   *Canonical home for the origin story:* here. The mechanism (irrelevant features dominate, linear
   growth) belongs in Chapter 4, Section `sec:flaws:similarity`, Equation `eq:distance_split`.
4. The second half asks five questions (enumerated list, M).
5. Chapter outline: one paragraph per chapter (all M).

**Should not spend words on**
- Chapter 3 numbers. The outline paragraph already gives findings in words only. Keep it that way.
- The full mechanism of distance dilution. Chapter 4 owns it. The hypothesis paragraph currently says
  "grows linearly with the number of features". That detail is a Chapter 4 statement.
- Defining LNL beyond one clause. Chapter 2, `sec:lnl_background`, owns it.

**Gap to report, not to fill.** The LNL paragraph (M) is one sentence of prose: the definition of LNL.
The exploratory study's result (all 20 lowest-retention configurations used a distance) exists only in
the commented bullets under it. The next paragraph then says "the fault lay in the distance... on which
such methods rely", but the failure it refers to has not been stated in prose. That is a missing
sentence, not a trim. Adelaide should decide whether to add one sentence or a back-reference to
Section `sec:lnl_motivation`.

---

## Chapter 2: Background (off limits; listed for ownership only)

**Its job.** It defines, once, everything later chapters rely on.

**What it owns (so later chapters back-reference it rather than restate it)**
- Representation families and their weaknesses. No single representation captures everything.
- Model families, and how far one wrong label spreads in each.
- The similar property principle, similarity coefficients (Tanimoto), activity cliffs, and the 1-WL
  limit of graph kernels.
- Real noise: heavy tails, laboratory offsets, censoring. The seven artificial noise conditions are
  defined in `sec:noise_injection`.
- LNL and noise mitigation, with the five groups of Song et al. (2022) (`sec:lnl_background`). "No
  single method wins" is already said in the Chapter 2 Conclusions (U).
- NoiseInject (`sec:noiseinject_pkg`), group-aware splits and the metrics.

---

## Chapter 3: Noise robustness (off limits; listed for ownership only)

**Its job.** It measures what makes a QSAR model robust to label noise.

**What it owns.** Canonical homes: the Chapter 3 sections and its Conclusions (U).
- Model architecture governs robustness (`\aucnorm`). Representation explains less than 10% of the
  variance in robustness. For performance, the two explain comparable shares (Chapter 3 Conclusions).
- The shape of the noise distribution matters little. Systematic noise does the most damage: grouped
  within a scaffold family, or censored at an assay limit.
- Uncertainty under label noise (`sec:uq`).
- Validation on LogD, Caco-2 and hERG Ki.

**Disagreements the editors must not copy forward.** Chapter 3's own text is U. Later marked paragraphs
restate it in ways that do not match it.
- **Uncertainty.** Chapter 3 Conclusions (U) says NGBoost "identifies noisy labels through a
  per-sample predicted scale". It also says NGBoost and GPs "often produce uncertainty estimates which
  track per-sample label noise". Chapter 1's outline paragraph (M), the Chapter 7 Summary (M), Chapter 7
  Limitations (M) and Chapter 7 Conclusions (M) all say uncertainty does not identify which labels were
  corrupted. Chapter 7 adds that the correlation stayed between -0.16 and 0.16 under six of the seven
  noise conditions. Chapter 6 line ~287 (U) sides with Chapter 3: "NGBoost in particular was selected
  due to its promising uncertainty values". So the two sides disagree on one point, whether per-molecule
  uncertainty finds the corrupted labels. The outline (3.9) sides with Chapters 1 and 7 and calls the
  old uncertainty result "retracted".
- **Most robust models.** Chapter 3 Conclusions (U) names "NGBoost and SVM, the most noise-robust
  models". Chapter 1 (M) names "NGBoost and random forests" on QM9.
- **Variance shares.** Chapter 7 Summary (M) gives 49.5% (model) and 9.2% (representation) for
  robustness under Gaussian noise. The TODO above that paragraph says Table `tab:anova_decomposition`
  gives 48.7% and 8.2%. The 29.1% / 26.6% split and the medians 0.813 / 0.951 do not yet appear in
  Chapter 3 prose.
- **Clean performance.** Chapter 4 `sec:lnl_motivation` first paragraph (M) says "the choice of
  representation explained more of the variance in clean performance than the choice of model". Chapter
  3 Conclusions (U) says model and representation explain comparable shares. The removed Chapter 1 text
  marks the clean split as `[RESULT NEEDED]`.
- **Noise type and rankings.** Chapter 3 Conclusions (U) says model rankings are concordant across noise
  strategies (Kendall's W = 0.92). The outline says W = 0.92 and NDS are old-study numbers that are not in
  the current paper. Chapter 7 (M) says the best representation changed between Gaussian and
  grouped-shifted noise for five of the thirteen models.

---

## Chapter 4: The Flaws of the Similarity Assumption

**Its job.** It is the motivation chapter and introduces no method. It shows, with small measurements,
that standard representations and the distances computed over them are unfit for a given dataset. It
shows that methods built on those distances fail. It ends with what a representation would need, which
Chapter 5 builds. Her brain dump: results "significantly smaller than noise robustness".

**Main takeaways**
1. On one dataset, most features of a general-purpose representation barely vary. On hERG Ki, 1,815 of
   the 2,048 ECFP4 features take one value in at least 95% of the 1,412 molecules. Learned embeddings
   have none, except the 292 always-zero MHG-GNN features.
   *Stated in:* "Constant and Near-Constant Features".
   *Canonical home:* Section `sec:flaws:constant`.
2. Some representations give different molecules the same vector, mostly stereoisomers. On hERG Ki,
   ECFP4 ties 56 of 1,412 molecules, all stereoisomers. The largest label spread in one tied group is
   1.14 pKi. No model can separate tied molecules.
   *Stated in:* "Molecules with Identical Feature Vectors".
   *Canonical home:* that subsection.
3. A distance sums over every feature, so features irrelevant to the task dilute it. Their share grows
   linearly with their number. Representations also disagree about which molecules are close. On the
   Biogen sets and QM9, two representations share 0.13 to 0.21 of each molecule's five neighbors.
   All-pairs correlation was below 0.10 in 44 of 70 dataset-representation combinations. The
   five-neighbor prediction was higher in all 70.
   *Stated in:* "Flaws in Similarity and Other Pairwise Computations".
   *Canonical home:* Section `sec:flaws:similarity`, Equation `eq:distance_split`. Every other statement
   of the "irrelevant features dominate the distance" argument should point here.
4. Methods that judge a label by its neighbors did worst. In the exploratory study, all 20
   configurations with the lowest retention used a distance. Neighbor-based cleaning had negative mean
   recovery for every representation at both noise levels.
   *Stated in:* `sec:lnl_motivation` and `sec:flaws:neighbors`.
   *Canonical home:* `sec:lnl_motivation` for the 221-configuration study. `sec:flaws:neighbors` for
   cleaning by neighbors.
5. What is needed is a compact set of features chosen for the task from several representations. It must
   be traceable and usable both as a model input and as a similarity space.
   *Stated in:* Conclusions, the "five properties" paragraph and Table `tab:flaws:requirements`.
   *Canonical home:* Chapter 4 Conclusions.

**Should not spend words on**
- Defining Tanimoto, activity cliffs, 1-WL or the similar property principle (Chapter 2 owns them). The
  marked Tanimoto paragraph (line ~253) is already a back-reference. That is the right shape.
- Chapter 3's variance findings. The first paragraph of `sec:lnl_motivation` (M) restates them, and that
  restatement conflicts with Chapter 3 (see above).
- The search axes of the hybrid. The Conclusions paragraph "Building this method raises several
  questions" lists them (U by the strict rule; see below).
- The recommendation of LNL over mitigation. Chapter 7 Limitations owns it, and the Chapter 4 text
  already points there (U).

**Note on marks.** Most of Chapter 4's body is U. The marked paragraphs are few: the chapter-opening
paragraph, the tie-counting method (~165), the Tanimoto back-reference (~253), the activity-cliff count
(~276), the first paragraph of `sec:lnl_motivation` (~402), the first paragraph of `sec:flaws:neighbors`
(~427), and the first sentence of the Conclusions (~474). The block `% ==== CODE INPUT: new, needs
review ====` sits above that first sentence only. The other Conclusions paragraphs have no comment
directly above them, so under the strict rule they are U, even though the block says it covers
"Paragraphs 1-6". The orchestrator should decide which reading applies before any Chapter 4 Conclusions
paragraph is touched.

---

## Chapter 5: Building Adaptive Hybrid Representations

**Its job.** The method chapter and the first hybrid paper. It defines the hybrid and the search. It
describes the datasets and setup, and will report when and why the hybrid helps (results pending).

**Main takeaways**
1. Joining whole representations adds their irrelevant features along with the useful ones. A method
   that joins them must combine and filter at once.
   *Stated in:* Introduction, "In practice, most research groups default to one representation..." (M,
   two versions exist) and "Combining Representations and the Alternatives".
   *Canonical home:* "Combining Representations and the Alternatives", with Table
   `tab:hybrid_alternatives`.
2. A hybrid is an input, not a model. It is up to a budget of features, selected from several base
   representations by a search for one dataset and one downstream model.
   *Stated in:* chapter-opening paragraph (M) and "The Adaptive Hybrid Representation" ("Formally, an
   adaptive hybrid representation is...", M).
   *Canonical home:* the "Formally..." paragraph. Chapter 6 Introduction, Chapter 7 Summary and Chapter 7
   Conclusions should point to it.
3. No setting wins everywhere, so importance method, allocation method, number of representations,
   internal evaluator, budget and downstream model are all searched.
   *Stated in:* chapter opening (M), "The Adaptive Hybrid Representation" ("This design leaves six
   choices open", M), "The Search Space".
   *Canonical home:* "The Search Space" and its subsections. The evidence that settings vary is pending
   ("Results: Blueprints on Specific Datasets" notes).
4. Every selected feature traces back to its representation, so a blueprint can show a chemist which
   substructures and descriptors the model relies on. Embedding features cannot be read this way.
   *Canonical home:* "Blueprints and Their Chemical Interpretation". Chapter 4's paragraph on readable
   and unreadable representations (U, ~241) is the motivation.
5. The method needs only a supervised label and several vector representations of the same samples, so
   it carries over to antibodies, reactions, nucleotides and tumor data.
   *Canonical home:* "Beyond Small Molecules".
6. Where one representation already dominates, the hybrid should show little gain. The search is costly
   and must be rerun when the data, representations or model change. Below about 200 samples it cannot
   reliably beat the best single representation.
   *Canonical home:* "Limitations" (M). Chapter 7 already points there rather than listing the limits.

Preliminary patterns in the notes (not yet prose): search settings found on lipophilicity, with features
re-selected on hERG, beat hERG's best single representation on 3 of 5 downstream models. On an antibody
expression scan the hybrid was ahead of the best single representation on all 5 downstream models.

**Should not spend words on**
- Re-arguing Chapter 4. The Introduction opens with a two-sentence back-reference, which is the right
  size. "Selecting useful features matters..." (M) restates the exploratory study and its limits (one
  dataset, one noise type). Chapter 4 `sec:lnl_motivation` (U) already states both.
- Chapter 3's results. The two NGBoost paragraphs ("Here, we present KIRBy..." and "Here, we present the
  adaptive hybrid representation...", both M) quote Chapter 3 clean R² values per representation. The
  point they serve is that the best representation changes with the dataset. The Chapter 5 Introduction
  already makes that point in one sentence with a back-reference.
- Label-noise methods, distances in a hybrid and noise detection. Chapter 6 owns them, and the
  Introduction says so.
- Uses of a task-specific distance. Chapter 4 `sec:flaws:consequences` (U) lists them. The Blueprints
  subsection (M) repeats the list and then adds generative design.

**Duplicated paragraphs (both versions print).** The Introduction holds pairs of paragraphs: Adelaide's
original, marked `% TODO: review`, and a rewrite marked `% TODO: REVIEW MODIFICATION` directly after it.
Both appear in the PDF.
- "The majority of the work in this space..." (two versions)
- "So far, the most common strategy for combining feature sets..." (two versions)
- "In practice, most research groups default..." (two versions)
- "The pace of new model development..." (two versions)
- "For small molecules alone, the landscape..." (original) and "Representations now exist..." (rewrite)
- "Concatenation, the most common method..." (original) and "The hybrid search starts from all base
  representations..." (rewrite)
- "Here, we present KIRBy..." (original, with her comment "is this right??? I'm making shit up with
  NGBoost") and "Here, we present the adaptive hybrid representation..." (rewrite)
- "But as important as the selection of useful features is..." (original, **U**) and "Selecting useful
  features matters..." (rewrite, M)

The El Nahhas et al. 1.2% AUROC result appears twice, in "For small molecules alone..." and in
"Ensembling combines...". Deng et al.'s 62,820 models appear in "A comparison of 62,820 models..." and
also in the Chapter 3 Introduction (U). The original "Here, we present KIRBy" uses KIRBy as the method's
name. Elsewhere the thesis uses "adaptive hybrid representation", and KIRBy is the software repository.

---

## Chapter 6: Hybrid Representations for Similarity and Label Noise

**Its job.** The second hybrid paper. It returns to the failures of Chapter 4 and asks whether a hybrid
repairs them: nearest-neighbor prediction, noise detection and estimation, and LNL. It asks this one
noise type at a time. Her brain dump: the Introduction should be "relatively short, just tying together
the previous two chapters".

**Main takeaways**
1. The hybrid began as a fix for similarity. This chapter tests whether features chosen for the task
   give better neighbors than the single representations they come from.
   *Stated in:* chapter opening (M), Introduction paragraph 2 (M, "The original idea behind the
   hybrid...").
   *Canonical home for the origin story:* Chapter 1. Here, one clause.
2. A similarity hybrid uses the same search but ranks candidates by predicting each molecule's activity
   from its five nearest neighbors. Neighbors come from other scaffold groups and are weighted by
   inverse distance.
   *Canonical home:* "Similarity in a Hybrid". Chapter 5 already defers to it.
3. Finding noisy labels, estimating how large the noise is, and improving the held-out score are three
   different tasks. A method can do the first and fail the third.
   *Stated in:* chapter opening (M) and Methods ("We distinguish three tasks...", M).
   *Canonical home:* the Methods paragraph.
4. The type of noise decides what can be fixed. Censoring has a known direction and location. In the
   preliminary run, raising every clipped label by one constant on hERG Ki at 10% clipped lowered RMSE on
   all four downstream models tried, on ECFP4. Gaussian noise, the standard assumption, is the hardest
   case.
   *Canonical home:* "Comparison across Noise Types" (notes only so far). Until that prose exists, the
   first paragraph of Chapter 7 (M) is the only prose statement of "Gaussian is hardest". Protect it.
5. Under Gaussian noise, the input moved the score more than the choice of LNL method did (preliminary,
   antibody thermostability, twelve methods against one plain network).
   *Canonical home:* "Gaussian Noise: Methods for Learning with Noisy Labels" (notes only).
6. Better similarity cannot remove the noise on the neighbors' own labels. No one method fits every noise
   type and dataset, so model the noise first.
   *Canonical home:* "Comparison across Noise Types" and the Conclusions (both notes only). Chapter 7
   carries the general form.

**Should not spend words on**
- The noise conditions. The Noise Conditions paragraphs (M) point to `sec:noise_injection`, which is
  right. They should only add what is specific to Chapter 6 (levels, which experiment uses which
  condition). The sentence in the censoring paragraph that censoring represents assay range limits
  repeats Chapter 2.
- Restating Chapter 4's findings. Introduction paragraph 1 (M) restates the ECFP4 and Tanimoto dilution
  point, and then restates the hybrid definition that Chapter 5 owns.
- The distance metric list. Chapter 4 "Distance Metrics for Continuous Molecular Descriptors" (U) owns
  it. Chapter 6 already refers back for Manhattan distance.
- Recovery rate as used in the exploratory study. It is defined here, and Chapter 4 points here. That is
  correct.

---

## Chapter 7: Discussion and Conclusions

**Its job.** It draws the argument together and makes the closing case. The case is to handle noise
realistically, that no single method fits every dataset, and that simple is often better. It then gives
the recommendation of LNL in training over mitigation, the limitations, the future directions and the
conclusions. The outline note says "each chapter's own conclusions carry the detail".

**Main takeaways**
1. Handle label noise realistically: ask where it comes from and what shape it has, not only how much
   there is.
   *Canonical home:* the opening paragraph and the "Conclusions" paragraph "Both halves lead to the same
   advice" (M).
2. Real assay noise is not only Gaussian, and Gaussian noise is the hardest case to correct.
   *Canonical home:* Chapter 6 "Comparison across Noise Types" once written. Chapter 7's opening then
   points there, as it already does.
3. No single method fits every dataset, whether for building the hybrid or for LNL. What the thesis
   offers is a procedure for testing, not a fixed recipe.
   *Canonical home:* "Dataset Dependence of the Best Method" (M).
4. Simple is often better: tree ensembles under noise, fewer features, and neighbors on a small
   task-relevant feature set.
   *Canonical home:* "Simple Models and Small Feature Sets" (M).
5. Correcting labels without ground truth is hard. Prefer LNL methods that change training over noise
   mitigation, except for noise of known structure (censoring, gross outliers, an offset tied to one
   source). Use uncertainty to judge how noisy a dataset is, not which labels are wrong.
   *Canonical home:* "Limitations", second paragraph (M). Chapter 4 points here.
6. Next steps: classification and flipped labels, hybrids as inputs to generative models, and a faster
   search.
   *Canonical home:* "Future Directions".

**Should not spend words on**
- Re-reporting Chapter 3 in detail. Chapter 7 currently restates Chapter 3 four times, all M: the Summary
  paragraph on Chapter 3 (six numbers), the "evidence is spread across the thesis" paragraph, the first
  paragraph of "Simple Models" (eight numbers that its own TODO says are not yet in Chapter 3 prose), and
  the first Conclusions paragraph. The Summary and the Conclusions both state the same four Chapter 3
  findings in words.
- Re-reporting Chapter 4. The Summary paragraph on Chapter 4, the third "Simple Models" paragraph and the
  second Conclusions paragraph each restate the exploratory study and the distance argument.
- Re-arguing why Gaussian noise is hard. That belongs in Chapter 6.
- Listing the hybrid's own limits. It already points to the Chapter 5 Limitations. Keep it that way.
- The Chapter 7 limitation that "the label-noise experiments are almost all regression" appears in
  Limitations paragraph 3 (M) and again at the start of Future Directions paragraph 1 (M).

---

## Summary of Research Contributions

It lists papers, a conference poster and blog posts, and has no takeaways of its own. Chapter 1's
Collaborations paragraph and Chapter 7's Summary already point to it. Nothing here needs protecting
beyond keeping the lists.

---

## The claims the whole thesis stands on

1. **Real label noise in QSAR is structured, not independent Gaussian error.** It has laboratory
   offsets, censoring at assay limits and heavy tails. (Chapter 2 defines it; Chapter 3 shows systematic
   noise does the most damage; Chapter 6 shows the type decides what is fixable.)
2. **Under label noise, the model architecture governs robustness more than the representation.**
   Representation explains less than 10% of the variance in `\aucnorm`. (Chapter 3; see the disagreement
   on exact shares above.)
3. **Predicted uncertainty rises with noise on average but does not identify which labels are
   corrupted.** (Chapter 1, Chapter 7. Chapter 3's own Conclusions (U) say the opposite for NGBoost and
   GPs; this must be resolved by Adelaide, not by an editor.)
4. **On a given dataset, general-purpose representations are mostly uninformative.** Some merge distinct
   molecules, and a distance over them is diluted by features irrelevant to the task. (Chapter 4.)
5. **Methods that judge a label by its neighbors in a standard representation usually made models
   worse.** (Chapter 4, from the exploratory study; Chapter 6 repeats it on more datasets and noise
   types.)
6. **A budget of task-relevant features selected from several representations, the adaptive hybrid
   representation, is a usable model input and a candidate similarity space.** Because the best settings
   differ by dataset and model, they are searched rather than fixed. (Chapter 5; results pending.)
7. **Finding noise, sizing it and removing it are different problems.** Censoring is largely
   correctable. Gaussian noise is the hardest case, and better similarity does not fix it because the
   neighbors' labels are noisy too. (Chapter 6; preliminary.)
8. **No single method fits every dataset, and simple choices often do as well.** Model the noise first,
   and prefer LNL in training over label correction except for noise of known structure. (Chapter 7.)

## Chain of argument, Chapter 1 to Chapter 7

1. **Chapter 1.** QSAR labels are noisy in ways most studies do not model. Two questions follow. What
   makes a model robust to noise? Can the representation itself be adapted to the task?
2. **Chapter 2.** It defines the representations, models, similarity, the seven noise conditions, LNL,
   NoiseInject and the splits.
3. **Chapter 3.** Under noise, the model governs robustness and the representation matters little. The
   shape of the noise matters little, but systematic noise does the most damage. Uncertainty does not
   point to the wrong labels. So a different way to judge each label is needed.
4. **Chapter 4.** Judging a label by its neighbors failed. The cause is upstream. Standard
   representations are mostly uninformative on one dataset, and distances over them are diluted. A
   compact, task-chosen, traceable feature set from several representations is needed.
5. **Chapter 5.** The adaptive hybrid representation is that feature set. It is built by a search, and
   because no setting wins everywhere, every setting is searched. It works as a model input across
   domains (pending).
6. **Chapter 6.** Using the hybrid as the similarity space for nearest-neighbor prediction, noise
   detection and LNL: what can be fixed depends on the noise type. Better neighbors do not remove
   Gaussian noise (preliminary).
7. **Chapter 7.** Both halves give the same advice. Model the noise before choosing a method, and choose
   the features for the task. Search what varies by dataset, and keep the simple, robust parts fixed.

---

## Repetition map: each argument that is restated, with its canonical home

| Argument | Canonical home | Also stated in (mark) |
|---|---|---|
| Irrelevant features dominate a distance, growing linearly | Ch4 `sec:flaws:similarity`, Eq. `eq:distance_split` (U) | Ch1 hypothesis paragraph (M); Ch4 opening paragraph (M); Ch4 Introduction paragraph 2 (U); Ch4 `sec:lnl_motivation` last paragraph (U); Ch4 Conclusions (U by strict rule); Ch6 Introduction paragraph 1 (M); Ch7 Summary on Ch4 (M); Ch7 Conclusions paragraph 2 (M) |
| Origin story: built for similarity, found to work as a model input | Ch1 hypothesis paragraph (M) | Ch4 Conclusions "These flaws are where the thesis began" (U by strict rule); Ch5 Introduction paragraph 1, last sentence (M); Ch6 opening paragraph (M); Ch6 Introduction paragraph 2 (M) |
| Hybrid definition (budget, several representations, one dataset and model, an input not a model) | Ch5 "Formally, an adaptive hybrid representation..." (M) | Ch5 opening paragraph (M); Ch6 Introduction paragraph 1 (M); Ch1 Chapter 5 outline paragraph (M); Ch7 Summary on Ch5 (M); Ch7 Conclusions paragraph 2 (M) |
| The list of searched settings | Ch5 "The Search Space" | Ch5 opening paragraph (M); Ch5 "This design leaves six choices open" (M); Ch4 Conclusions "Building this method raises several questions" (U by strict rule); Ch7 "Dataset Dependence" paragraph 1 (M) |
| Exploratory LNL study result (all 20 lowest-retention configurations used a distance) | Ch4 `sec:lnl_motivation` (U) | Ch4 opening paragraph (M); Ch4 Conclusions (U by strict rule); Ch5 "Selecting useful features matters..." (M); Ch7 Summary on Ch4 (M); Ch7 "Simple Models" paragraph 3 (M, with back-references); Ch7 Conclusions paragraph 2 (M) |
| Limits of the exploratory study (one dataset, one noise type) | Ch4 `sec:lnl_motivation` (U) | Ch5 "Selecting useful features matters..." (M) |
| Chapter 3 findings | Ch3 sections and Conclusions (U) | Ch1 Chapter 3 outline paragraph (M, words only); Ch4 `sec:lnl_motivation` paragraph 1 (M, conflicts); Ch5 NGBoost paragraphs (M, two versions); Ch7 Summary (M); Ch7 "evidence" paragraph (M); Ch7 "Simple Models" paragraph 1 (M); Ch7 Conclusions paragraph 1 (M) |
| Find, estimate and improve are three tasks | Ch6 Methods "We distinguish three tasks..." (M) | Ch6 opening paragraph (M) |
| Most LNL methods come from image classification with flipped labels | Ch2 `sec:lnl_background` (U) | Ch4 `sec:lnl_motivation` (U); Ch7 "LNL methods from other fields" (M) |
| Hybrid result count "on how many of the 16 datasets" (placeholder) | Ch5 Results | Ch5 opening paragraph (M); Ch5 Introduction paragraph 3 (M): the same `[RESULT NEEDED]` twice |
| Regression-only scope of the noise experiments | Ch7 Future Directions paragraph 1 (M) | Ch7 Limitations paragraph 3 (M) |
| Uses of a task-specific distance (applicability domain, clustering, federated weighting...) | Ch4 `sec:flaws:consequences` (U) | Ch5 Blueprints subsection, last paragraph (M) |

## Where `THESIS_OUTLINE.md` disagrees with the current thesis or the brain dump

- Outline 1.5 says "Chapter 4 introduces the adaptive hybrid representation". This is stale. Chapter 4
  is now the flaws chapter, and the thesis text is correct.
- Outline 1.5 says the Chapter 3 paper is "in preparation for the Journal of Cheminformatics". The
  Summary of Research Contributions has that entry commented out ("was not submitted"). The thesis style
  rule says never reference the Chapter 3 paper. Chapter 1's outline paragraph (M) correctly does not
  mention it.
- Outline 4.1 and the Chapter 4 notes say to cite the confirmation report (`Punt2026confirmation`). The
  style rule forbids this. The thesis text says "an exploratory study carried out earlier in this
  project", which follows the rule.
- Outline 3.9 says the Chapter 3 Conclusions carry the old uncertainty chapter's text "appended
  unchanged", and that W = 0.92 and NDS are old-study numbers. Both still appear in the Chapter 3
  Conclusions (U). Chapter 3 is off limits, so this is for Adelaide only.
- The Chapter 7 title in the outline is "Discussion and Future Directions". The thesis now uses
  "Discussion and Conclusions", with a new Conclusions section.
- Page targets in the outline (dated): Chapter 1 8 to 10 pages, Chapter 2 30 to 35, Chapter 4 15 to 20,
  Chapter 5 40 to 50, Chapter 6 25 to 30, Chapter 7 12 to 15.
- Brain dump against thesis: the brain dump says irrelevant information grows "exponentially (is this
  true?)". The thesis says linearly (Chapter 1 M, Chapter 4 U). The thesis resolved this, and no change is
  needed.
- Question lists do not line up. Chapter 1 gives five second-half questions. Two of them are
  noise-robustness of hybrids (`\aucnorm`) and sample weighting or label smoothing against removal.
  Chapter 6 states its own three research questions, and neither of those two appears among them. The
  Chapter 5 Conclusions notes say the `\aucnorm` question is answered in Chapter 5 "only if this
  comparison stays in this chapter". The Chapter 7 Summary notes say to follow the order of
  `sec:intro:questions`.
