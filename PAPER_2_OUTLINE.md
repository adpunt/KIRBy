No scores in this file. Results are still being added, so any number written here goes
stale; name the finding and the example dataset, and read the number from the json when
the text is written. Sources are in `PAPER_OUTLINE_NOTES.md`, the per-experiment
inventory in `PAPER_EXPERIMENT_TABLES.md`, both beside this file.

---

# Paper 2 — the adaptive hybrid representation

A method paper. The method takes several feature matrices on the same samples, a label, a
group label per sample and a budget, and returns a feature selection method tailored to
that dataset and downstream model. Molecules are where it was built and where feature
meaning can be read back, not the claim.

## Takeaways

Held open until the experiment inventory below is filled in. What survives from the
thesis chapter is:

- Choosing the features per dataset and model beats the best single feature set.
  Every small-molecule dataset, both models, every fold of the regression sets.
- It also beats the feature set that turns out best on held-out data, which nobody can
  pick in advance.
- Fewer features is not the explanation. At the same number of features the hybrid is
  lower-error at every budget on QM9, Caco-2, LogD and MDR1--MDCK efflux, against
  recursive elimination and against random features.
- Some models predict better given more features, others given fewer, so the number has
  to be searched per model. The downstream model accounts for much more of the variation
  across a search's configurations than the budget does, QM9 being the clearest case.

Two of the three candidate takeaways now have an answer, and neither is the answer the
method wants:

- How many samples it needs: there is no trend. The hybrid's advantage over the best
  single feature set does not grow or shrink with the number of training rows. What does
  bound it is the number of groups, not the rows — below a handful of groups the search
  returns nothing whatever the row count. Source c05, a null; the group floor belongs in
  limitations.
- What it costs against running one model per feature set: the ensemble is ahead on most
  antibody datasets and the two are close on small molecules, and the hybrid's search
  costs far more than it saves per retrain. Source c11.

The third is still open: where the gain comes from. c18 is the experiment that
answers it, and it has not run under its current design yet -- it holds the number
of representations fixed and varies nothing else, so the gain can be charged to
drawing features from several representations or not.

## Sections

**Introduction**

- Several feature sets describe the same samples, and the choice between them changes the
  score. Which one is best is not a property of the data alone. Over the datasets and
  downstream models here, many different feature sets each turn out best somewhere, none
  of them wins on more than a small share of the dataset and model pairs, and on about
  half the datasets the two downstream models disagree about which single feature set is
  best. Source c01.
- Putting them side by side fails on the number of features, and because every input's
  irrelevant features arrive with its useful ones. A method has to join and filter at once.
- Existing ways to join or reduce: late fusion, intermediate fusion, multiple kernel
  learning, principal components and partial least squares, flat feature selection, group
  lasso. The comparison table from the thesis.
- What the method is, and what the search returns.

**Methods**

- Inputs: K feature matrices on the same rows, a label, one group label per row, the
  metric, the task type, downstream models, budgets. Groups are required and there is no
  random-split fallback.
- The six searched settings and their values: downstream model, budget, representation
  count, importance method, allocation method, internal evaluator.
- **One sentence beside the internal evaluator.** The evaluator that scores each single
  representation during the search was checked against the downstream model itself, a
  near relative, a far one, a wider DNN and the dataset's decided list, each setting
  running its own per-fold search. On small molecules the settings sit within 0.005 of
  each other; on antibodies they differ by up to 0.08 Spearman in no fixed direction,
  less than the fold spread. So the search uses the cheaper evaluator. Table in the
  supplement, as the thesis has it in `app:add:evaluator`. Caveat beside it: each
  setting picks its own budget, so two things differ at once. Source c23.
- The three levels of splitting, every split keeping groups whole.
- How the budget is split, and that it is a maximum rather than a target.
- What is not specific to molecules: no SMILES, fingerprint or scaffold anywhere in the
  search code; the two chemical-sounding parts are a Jaccard kernel on any binary vector
  and an objective taking any pairwise distance.
- Datasets. A dataset is listed here only after Adelaide has decided it is in; the route
  a candidate takes to get here, and where each candidate stands today, is the section
  "The route from this file into a paper" at the top of `NEW_DATASETS.md`.
  **TODO (2026-11-04): Matbench and ProteinGym are named below but have not been
  decided. Matbench is being screened; ProteinGym has not been screened. DroneRF is the
  other way round, being screened and decided on its splits, and in no outline.**
  Small molecules: QM9 HOMO--LUMO gap, hERG $K_i$, Caco-2 efflux, LogD, rat
  microsomal clearance, MDR1--MDCK efflux, hERG blocking, and from outside the thesis
  QM9 dipole moment, QM9 polarizability, hERG IC$_{50}$ (repeat measurements), human
  microsomal clearance and three ChEMBL targets. The Lhasa hERG set is out of both papers
  (Adelaide, 2026-11-02). Antibodies (Adelaide,
  2026-11-01): the anti-VEGF expression and binding scans (Koenig), binding free energy,
  EGFR and angiopoietin-2 binding, the Ginkgo expression titer, AC-SINS, SMAC colloidal
  stability and HIC retention sets, antibody humanness, CDR3 flexibility, anti-coronavirus
  binding, two trastuzumab--HER2 tasks and the Adams expression and binding scans.
  The Adams scans are in (Adelaide, 2026-11-03); their searches run on ordinary compute
  nodes, never htc-g011, which restarted the thesis-run searches about every four hours.
  **The two Adams scans use random folds (Adelaide, 2026-11-03), the only random split in
  either paper.** They are 5,000 variants of one antibody, 4-4-20, carrying 1 to 3
  mutations each among 20 heavy-chain positions, so no grouping holds a mutation out:
  grouping on the set of mutated positions gives 267 groups on expression and 263 on
  binding, and in all five folds every held-out mutation also appears in a training
  variant. Joining any two variants that share a mutated position gives one group of 4,945
  of the 5,000. \citet{Chungyoun2024}, the benchmark these two files come from, splits
  them at random, and the Methods section says so and gives these counts.
  Nanobody thermostability, CHO polyreactivity and the AbAgym VEGF scan are out.
  Two of these follow the group's own handling, and the Methods section cites the source:
  CDR3 flexibility is two datasets, one per loop, the two test sets \citet{Spoendlin2025}
  ship separately; the trastuzumab--HER2 scores are also reported per number of mutations
  from the parent sequence, as \citet{Chinery2024} report them.
  The trastuzumab--HER2 folds group the CDRH3 loops at 80 percent identity
  (single linkage, so every near-identical loop stays on one side), after
  \citet{Chinery2024}'s clonotype split: they report their main result on a random
  70:15:15 split and, beside it, a split by clonotype, clustered by V and J gene and by
  70 percent identity across the CDRH3. Every variant here carries trastuzumab's V and J
  genes, so only the CDRH3 rule applies, and a random split is not used anywhere in this
  work except the two Adams scans (below). The one deviation is the threshold, and the Methods section states it with the
  reason: at 70 percent (at most 3 of the 10 loop positions differing) the groups chain
  into one piece holding 3,950 of the 5,000 variants, and the five folds then hold out 1,
  242, 266, 278 and 4,213 variants. At 80 percent the largest group holds 284 variants,
  there are 3,039 groups, and the folds hold out 740 to 1,354 variants with the binding
  share between 0.29 and 0.39. It is also the threshold the CDR3 flexibility sets use.
  Grouping instead on the number of changed residues, which the folds did until
  2026-11-02, held out 13 to 3,035 variants per fold with the binding share running from
  0.92 at three changes to 0.16 at ten. \citet{Chinery2024} report the same imbalance in
  the full library and treat the number of changes as a shortcut a model can take,
  reporting precision per count on the random split's test set rather than splitting on
  it. Scores are still reported per number of changes under this grouping.
  Nucleotides (Adelaide, cut from 21 to 6 on 2026-07-18, recorded in DATASET_STATUS.md):
  four Nucleotide Transformer Revised histone-mark tasks (H3K27me3, H3K4me3, H3K9ac,
  H3K9me3), the TATA promoter task, and the Genomic Benchmarks human enhancers (Cohn).
  Reactions: Buchwald--Hartwig yield,
  the Doyle plate run. Suzuki coupling is cut (Adelaide, 2026-11-01): no signal, every
  representation but one scored worse than predicting the training mean yield. Benchmarks
  outside chemistry (UCI HAR cut, Kepler cut 2026-11-04): Matbench and ProteinGym. Rows, feature sets and
  metric per dataset go in a table, and so does each feature set's width; both come
  from c01, which also holds every single feature set's score per dataset and downstream
  model.
- **Two or three sentences on importance as a distance weighting.** Each feature is
  multiplied by the square root of its importance, weights scaled to average 1, the
  importance computed on the fold's training rows only. Each held-out sample's label is
  then the average of its 10 nearest training samples, weighted by 1 over distance. No
  model is fitted and no search runs. One more sentence for the control: the same weights
  are permuted across the features, so a permuted run scoring as well as the real one
  would mean the rescaling did the work. Source c04.
- **Two sentences on importance used a second time, during training.** The hybrid is
  trained five ways: unweighted, features multiplied by the square root of their
  importance, samples weighted by how well their label agrees with their 10 nearest
  training samples, both together, and only the features above the median importance.
  Run at every one of the dataset's decided budgets, per fold and downstream model; the
  GP takes no sample weights. Source c19.
- **TODO: a few sentences on adding extra features to one representation and to the
  hybrid.** One strong representation at its full width, and the hybrid, each with a
  handful of extra features added: the dataset's small engineered feature sets, the top
  features of every other representation, or both, with two decoy blocks planted among
  the candidates. Not written until the method is refined and run; the admission rule
  and which datasets keep augmentation are open (Adelaide, 2026-10-31). Source c21.
- **Two sentences on reading what the hybrid draws on.** For each representation in a
  hybrid, the number of its features the hybrid took, and the fall in held-out score
  when all of those features are shuffled together across the held-out molecules, five
  shuffles, no refit (grouped permutation importance; Breiman 2001, Fisher et al. 2019,
  Gregorutti et al. 2015). Per fold and downstream model, in the dataset's own metric.
  Source c22.
- Evaluation: group-aware folds, winners chosen on training rows only, search run inside
  each outer training fold and scored once on the untouched fold.
- Comparators: best single feature set, the one best on held-out data, the budget filled
  from the top-ranked input alone, an even split, principal components on everything
  joined, recursive elimination, random features, mRMR, and one model per input.
- **Three sentences on the principal-components comparator**, since it is the only one of
  these with a table of its own. Features from every input are pooled, standardised on
  the fold's training rows and reduced to each budget of the dataset's grid; the model is
  fitted on the components as they are. On each fold the hybrid is compared at its own
  budget, so neither side's width is chosen on held-out samples. There cannot be more
  components than training rows, so above that the two widths differ.
- **Three sentences on the three other ways of choosing the same number of features**,
  because one of them wins on part of the data and the reader has to know how it was
  run. Recursive elimination ranks every feature of every input in one pooled list,
  fitting a forest and dropping a fixed share of the features at each step. mRMR ranks
  features by their mutual information with the label minus their redundancy with the
  features already taken, and it is capped at a feature count well below the
  small-molecule budgets, so it has no result there -- say the cap and say which datasets
  it therefore covers. Random features draw the budget uniformly from the same pooled
  list. All three are chosen on the fold's training rows, do not depend on the downstream
  model, and are then scored with each of the dataset's models. Source c17.
- **Three or four sentences on carrying a recipe from one dataset to another.** The
  search runs once over every row of a larger source dataset, and its two best recipes
  with distinct selected features are carried to a smaller target. Both datasets are
  reduced to the representations they share. On the target, per shared downstream model
  and per outer fold, four things are fitted on the fold's training rows and scored on
  its held-out rows: the best single representation, the target's own hybrid from a
  search on that fold's training rows, the source recipe with its features re-picked on
  the target's training rows, and the source's exact selected features with nothing
  re-picked. No target row is in the source search. Pairs are within a domain only,
  since small molecules and antibodies share no representations; QM9's other eleven
  properties are datasets of their own so that a QM9 property can be the source for an
  assay. Source opig_blueprint_transfer.
- **Two sentences on whether choosing the features replaces the model's own
  regularization.** Per outer fold, at each decided budget, the fold's recipe is refitted
  under every setting of one regularization parameter of its downstream model: the SVM's
  `C`, the forest's and the boosted trees' depth, the DNN's dropout, the GP's length
  scale. The fold's two best single representations are fitted under the same settings
  at their full width, with no selection. Source c16.
- **Three sentences on carrying a recipe from one fold to another.** For every ordered
  pair of folds, the source recipe is a search on the three folds outside both, and on
  the target fold four things are fitted on its training rows and scored on its held-out
  rows: the fold's own recipe, the source settings with features re-picked, and the
  source's exact selected features. A fifth score is the source search fitted on its own
  three folds, which differs from the fold's own recipe only in training molecules. Fold
  k's own recipe applied directly is also recorded, marked as having seen the target's
  rows in its search, so the size of that leak can be read. Source c24.
- **Two sentences on whether the feature selection overfits.** On each outer fold the
  features are selected on the training data only, then scored on the held-out set, beside
  the same number of random features scored on the same held-out set. A selected feature
  set that works only on the training data shows a held-out score below its training score
  and close to the random score. Source c33.
- **TODO (Adelaide, 2026-11-01): a short subsection, a case study on the reaction
  datasets.** Can the hybrid show whether a new fingerprint carries useful signal, beyond
  its score alone? The fingerprint is cgr_morgan, written for this work: a circular
  fingerprint hashed from the published condensed graph of reaction \citep{Varnek2005,
  Hoonakker2011}, in which a bond that changes in the reaction hashes differently from
  one that does not. It is screened beside the other condensed-graph representations
  (the CGR-SMILES language model \citep{Gerhaher2025}, the Graphormer embedding
  \citep{Nugmanov2022, Ying2021}, Chemprop's condensed-graph mode \citep{HeidGreen2022},
  atom-pair and torsion
  fingerprints on the condensed graph \citep{Carhart1985, Nilakantan1987}) and the datasets' existing representations, on
  Buchwald--Hartwig and the Doyle plates, same folds and models as everything
  else. Counts of bonds formed and broken are too few to stand alone, so they are added
  to each representation as extra features. Then the search is run per fold with and
  without the condensed-graph representations. Which representations go into that search
  waits for the screen. **The reaction datasets may carry more than ten representations
  for this subsection (Adelaide, 2026-11-02), unlike every other dataset**, because the
  seven condensed-graph representations are screened beside the existing ten.
  Source reaction screen (b10), opig_cgr_comparison.
- **TENTATIVE (Adelaide, 2026-11-02): a short subsection, the features a generator's
  predictor sees.** In only if it works. A molecule generator keeps a property predictor
  in its loop, and convention hands that predictor one fingerprint. The generator is
  REINVENT 4 \citep{Loeffler2024}, pinned at one commit, its published reinforcement
  learning configuration copied in unedited and called through its own ExternalProcess
  scoring plugin, so no line of it differs between arms and only the predictor's features
  change. Four arms, same model class (LightGBM), same molecules, same generator settings:
  ecfp4 at full width as the control, the blueprint the search picks, a random Gaussian
  matrix of the same width as that blueprint, and the single representation that scores
  best alone. The predictor is fitted on hERG $K_i$ (1,412 molecules) and steered to
  maximise predicted pKi. The molecules that come out are judged by a separate model
  fitted on the hERG blocking set (7,772 molecules), with the 166 molecules the two sets
  share removed first, so the model that steers is never the model that judges. The
  blueprint is credited over the random matrix of the same width or not at all. REINVENT's
  own three published scoring components stay in every arm; each arm adds one predictor
  beside them. Source opig_generator_scoring_swap, built 2026-09-28, not yet run.
- **TODO (Adelaide, 2026-11-02): a short subsection, handing a published feature
  selector the hybrid's features.** MODNet \citep{DeBreuck2021} ranks features before it
  trains, using two scores that are both normalised mutual information: how much a
  feature tells about the target, and how much it repeats a feature already ranked, which
  pushes the repeat down. Two questions. Does MODNet's error change when both scores use
  a different measure -- the size of the Pearson correlation, the size of the Spearman
  correlation, or distance correlation \citep{Szekely2007} -- instead of mutual
  information. And when MODNet is handed a hybrid's features instead of its own, how does
  its error compare under the same measure. Run on the Matbench v0.1 tasks on Matbench's
  own five folds, with MODNet's network, its settings search and all features ranked held
  the same as the published run (b04). MODNet's own rule of dropping a feature whose
  mutual information with itself is below 0.2 is swept on and off, and its published
  setting -- mutual information with the pair scores read from the file computed on
  Materials Project -- is run as the reference. Recorded per task, fold, input, measure
  and setting: test-fold MAE, how many features were ranked, and the 100 top-ranked
  feature names, so two measures can be compared by what they pick.
  The hybrids are c02's, built into feature files per fold by
  opig_modnet_hybrid_features.py: MODNet 0.1.12 needs numpy 1.19 and runs in its own
  environment, so the features pass between the two as files, and the train and test
  indexes are checked against Matbench's folds. It reads c02's cached search rather than
  searching again, so c02 must have run on these tasks first. Server only.
  Source opig_modnet_feature_scores and opig_modnet_hybrid_features, both built
  2026-11-01, not yet run.
- **TODO (Adelaide, 2026-11-02): a few sentences, the antibody twin of the fingerprint
  case study.** Two representations are built from the mutation rather than the sequence:
  the difference between a variant and the wildtype, as 5 features and as 1,140. The
  question kept is the second one -- when the search can choose from the dataset's ten
  existing representations plus these two, how many of the features it picks come from
  the two. Scoring them alone, against those ten, is the supporting number. Run on
  the three of its four datasets that this paper keeps: flab_expression_koenig and the
  two Adams scans. flab_binding_koenig is conditional on its runs finishing, so it is
  added only if they do. Per outer fold, one search on that fold's training rows over the twelve
  representations on the dataset's decided grid, then per decided model the 2 best single
  representations and the 2 best hybrids, each fitted on the training rows and scored on
  the held-out rows, plus each mutation-difference representation alone with its rank
  among every representation the search scored. The per-representation feature count has
  been recorded since 2026-10-05, which is what makes the kept question answerable; it
  was dropped on 2026-09-11 when it was not. Source opig_scaffold_hybrid, not yet run.
- **TODO (Adelaide, 2026-11-02): a small point, smaller than the thesis gives it, with
  the figure.** Some representations give two different molecules the identical row of
  numbers, which forces any model to predict the same value for both. Joining all of a
  dataset's representations removes this, so keeping original features from several
  sources separates molecules that one source merges. Each count is split by cause: the
  same molecule entered twice (the same full InChIKey), a failed featurisation writing
  zeros, two molecules differing only in stereochemistry (the same first InChIKey block),
  or two different structures. Only the last two are a property of the representation.
  On hERG $K_i$, ECFP4, atom pairs, PDV, Mordred and SMI-TED give the same 56 molecules
  identical rows, so joining any of those five leaves all 56 merged; the graph-kernel
  features merge 105 of 1,412, of which 49 are not stereoisomers; ChemBERT, GROVER,
  MHG-GNN, MoLFormer and GraphMVP merge none. Joining all of a dataset's representations
  leaves no molecule merged. The error floor this sets is small -- at most 0.134 p$K_i$ on
  hERG $K_i$ against a label standard deviation of 0.915.
  **Every number in this bullet is from the thesis run and reruns.** Two faults apply to
  all of them: the neural-net scores predate the clip of predictions to the training-label
  range, and a hybrid is built below its stated budget wherever importance reaches zero,
  so a budget label is not a width.
  **Figure:** the two hERG $K_i$ stereoisomers with identical ECFP4 vectors, from the
  thesis (`figures_code_input/herg_ki_ecfp4_stereo_pair.pdf`, thesis.tex:1049). They
  differ at the carbon bearing the amide nitrogen and are 1.14 p$K_i$ apart, so an ECFP4
  model can at best predict their mean and leave each one 0.57 p$K_i$ out. The panel
  below shows the 55 of 2,048 ECFP4 bits set in both.
  Two faults. Three datasets have hybrids that leave
  more molecules merged than all representations side by side, and on one of those, 35 of
  35 hybrids do. And one field name holds two different quantities between the
  representation rows and the hybrid rows, so those two columns cannot be compared as
  they stand. Source opig_shared_vectors_census: ten datasets complete, one errored, one
  partial, everything alphabetically after binding free energy never reached.
- **TODO (Adelaide, 2026-11-02): the follow-on, why the score goes up.** The bullet above
  counts how many molecules a representation gives one identical row. This one adds other
  representations' features to the one doing the merging and measures two numbers on the
  same outer folds: the held-out error, and how many molecules still share a row in the
  matrix built for that fold. The features of the merging representation stay; the
  partners' features are added beside them, so if a partner tells two molecules apart
  they no longer share a row.
  On hERG $K_i$ (1,412 molecules, 5 folds, two models) the representation being fixed is
  the graph kernel, which merges 105 of the 1,412 and scores 0.6109 RMSE with the support
  vector regressor. Two sets of partners were added, picked so that one of them cannot
  work. ecfp4 with mhg_gnn: mhg_gnn merges no molecule, so it can separate them, and the
  result is 0.5715 RMSE with 0 molecules still merged, lower on all 5 folds. ecfp4 with
  pdv: both ignore stereochemistry, so they cannot separate 56 of the merged pairs, and
  the result is 0.5755 RMSE with 56 still merged, lower on 5 of 5 folds.
  Both sets lower the error by about the same amount, and only one of them fixed the
  merging. On one fold the hybrid kept 244 of the graph kernel's 276 features and the
  count stayed at 105, so what changed came from the partners' features and not from
  dropping the graph kernel's own.
  Caveat to state: the count of merged molecules is over all 1,412, while the error is
  over the 282 held out, so the two numbers are measured on different molecules.
  **Every number in this bullet is from the thesis run and reruns.** Two faults apply to
  all of them: the neural-net scores predate the clip of predictions to the training-label
  range, and a hybrid is built below its stated budget wherever importance reaches zero,
  so a budget label is not a width.
  Runs on Caco-2 and LogD as well as hERG $K_i$ (Adelaide, 2026-11-02), each with the
  same shape of pair: one that cannot separate the stereoisomers and one that can. On
  Caco-2 the representation being fixed is the graph kernel, merging 203 of 2,158, and on
  LogD it is topological torsions, merging 675 of 5,028.
  Source opig_shared_vectors_remedy: hERG $K_i$ has 16 rows from the thesis run; Caco-2
  and LogD have not run.
- **TENTATIVE (Adelaide, 2026-11-02): a few sentences, ranking variants without training
  anything.** In only if there is something to say. c02 already runs the two
  trastuzumab--HER2 tasks like any other dataset. The one thing this adds is a ranking
  that trains no model: a protein language model says how likely each amino acid is at
  each position, so a mutated variant can be scored by how plausible the model finds it
  next to the parent sequence. Rank every variant by that score and see whether the
  binders come out on top. Two representations carry it, esmc_llr and antiberty_llr, and
  the ranking uses the score per residue so that longer sequences are not favoured.
  The question is whether that ranking is anywhere near the trained models' ranking on
  the same held-out variants, since it uses no labels at all. \citet{Chinery2024} asked a
  different question -- which generator designs more binders -- so their numbers are
  context and not a baseline.
  Source opig_trastuzumab_enrichment, never run: its partial file holds a header line and
  no rows. Its datasets were cut on 2026-10-08 and are back in both papers' dataset lists
  as of 2026-11-01.
- **How much label information the kept features hold, against the same number of
  features chosen another way.** One unit is one fold, one downstream model and one
  representation the hybrid drew features from. The features the hybrid kept are compared
  with all of that representation's features, with a random subset of the same number, and
  with principal components of the same number, on the held-out rows.
  What the quantity is, and it must be said wherever a number from it appears: each
  representation's features are reduced to at most 50 principal components, label
  information is estimated for each component on its own, the same estimate under shuffled
  labels is subtracted, and the components are added up. Information that two components
  share about the label is counted twice, so the total is not the mutual information of the
  feature set, and two of these totals cannot be added. A difference between two of them is
  still fair, because both are sums over the same number of components from the same
  representation on the same rows.
  The file measures the same thing on the training rows, which are the rows the selection
  was fitted on. That says what the selection did, not what generalises. One clause at most,
  or leave it out.
  Source s03.
- **TODO (Adelaide, 2026-11-05): a meta-learner over models that each have their own
  hybrid (stacking; Wolpert 1992). No name yet, not built.** Per outer fold: five
  models, each with a hybrid search of its own, so each model has its own recipe and its
  own feature set. Each model predicts from its own feature set, and a meta-learner is
  trained on the five models' predictions on rows other than the fold's held-out rows.
  Poor models are left out on purpose. Candidate models, in her words: RF, DNN, LightGBM,
  SVM, GP, XGBoost, nets of different sizes, NGBoost. Candidate meta-learners, in her
  words: linear regression with constraints or weighted averages, ridge regression,
  gradient boosted trees, random forests, small nets.
  The five models: every candidate is searched, then the top 5 are taken on the fold's
  training rows (Adelaide, 2026-11-05). The datasets: not every dataset. Her words,
  2026-11-05: "there's no reason trying this on datasets where the hybrid didn't work",
  and "On a per-dataset basis determine based on results and runtime if this is worth
  it". She decides after seeing the results of a ledger step; which step is not decided.
  The rows the meta-learner trains on: inner folds on the outer fold's training rows,
  keeping each search's settings and re-picking the features per inner fold (Adelaide,
  2026-11-05). The settings and the top-5 pick have still seen those rows' labels, and the
  Methods section says so.
  She wants a minimum row count for a dataset to qualify (2026-11-05); its value is not
  decided.
  Not decided: which net sizes, how many inner folds, what it is compared with, and how
  the meta-learner that counts is chosen.
  Source meta_learner_own_hybrids (tests/experiments/papers/), built 2026-11-05, not yet run.

**Results**

Headings state the finding. To be written from the inventory below.

- **Table, one row per dataset and downstream model, blocks by metric.** The hybrid
  against the best single representation, and against the single representation that
  turned out best on the held-out samples. Both picked and scored the same way, per fold.
  This is the thesis's `tab:hybrid_vs_best_single`, which gives means over five folds;
  show per-fold scores here, because the thesis carries the fold detail in prose and a
  paper has no room for that. Source c02.
- **One short paragraph, in the text, against published scores on the three ChEMBL
  targets (Adelaide, 2026-11-04).** These three are the only datasets here with a
  published score on the same molecules and labels, so the hybrid's mean absolute error
  is given beside its RMSE for these three only, in the text, and compared with
  \citet{Dablander2023}, whose Figures 4, 5 and 6 report mean absolute error across nine
  representation and model combinations: dopamine D2 0.46 to 0.57 pK$_i$, factor Xa 0.58
  to 0.72 pK$_i$, SARS-CoV-2 main protease 0.42 to 0.55 pIC$_{50}$. Both numbers come
  from the same predictions, so this costs no extra run. Two sentences of caveat beside
  it: their folds are their own, not these, and mean absolute error is below RMSE on the
  same predictions, so the two columns of our own table are not a like-for-like pair.
  Every other dataset in the paper keeps RMSE alone.
- **A negative, named: antibody humanness (Adelaide, 2026-11-01).** One representation
  already solves the task, so the hybrid has nothing to add. In the thesis, best single
  against hybrid, mean over five folds: random forest 0.9659 against 0.9614 AUROC,
  XGBoost 0.9511 against 0.9555; germline identity alone gives 0.968. Say it as the case
  where the search has no headroom, not as a failure of the search. Not in Paper 3.
- **Table, one row per dataset and downstream model, blocks by metric.** The hybrid
  against principal components of all representations joined, at the hybrid's own budget
  on each fold, so neither side's width is chosen on held-out samples. In the thesis this
  is prose only, with no table. Source c02.
- **Table, one row per regression dataset, blocks by metric.** Prediction from the 10
  nearest neighbours in all inputs joined: unweighted, then weighted by each importance
  method, in the dataset's own metric. The importance the search uses to rank features
  also works as a distance weighting. The thesis table shows 13 datasets; 20 have
  results. One sentence of prose beside it: the same weights permuted across the features
  are the control, so the gain is charged to which feature carries which weight and not
  to the rescaling. Three permutations per fold, stored per permutation. No figure.
  Source c04.
- **One short paragraph, no table in the main text; the table goes to the supplement.**
  Used a second time during training, the importance scores add nothing on small
  molecules: the five ways of training land within about 0.01 of each other in every
  dataset's metric, and the one clear change is for the worse. The selection has already
  taken what the scores know. On antibodies the five ways differ more and in both
  directions, with no way winning across datasets or models. Whether this holds at every
  budget is a TODO; the results that exist are at 100 features where that was a decided
  budget and at the fold's best budget elsewhere. Source c19.
- **TODO: shape and finding from the results, once c21 has run.** Some datasets will rely
  heavily on their augmentation features, so this is a missing piece, not a side point.
  If adding the extra features works, that is a headline result on the datasets where it
  does, and it moves up into the Takeaways (Adelaide, 2026-10-31).
  The 10-06 run tested the extra features on SMAC colloidal stability and HIC retention
  only, and there the decoy features were admitted and random extras did as well as
  chosen ones, so nothing from that run is reported. Source c21.
- **One figure, the panel grid, plus one sentence beside it.** Each model's best hybrid
  scored on the held-out rows at every budget, with the per-fold picks underneath and the
  recursive-elimination and random-feature comparators on the same axes. This is the
  thesis's `fig:best_number_of_features` and its antibody twin. The sentence: the
  held-out score barely moves across the whole budget range on most datasets, Caco-2 and
  LogD being the clearest. Source c09 for the curves, c02 for the per-fold picks, c17 for
  the comparators.
- **Table, one row per dataset and downstream model, blocks by metric, plus one short
  paragraph.** The hybrid against the other ways of choosing the same number of features
  -- recursive elimination, random features and mRMR -- at the budget each fold picked,
  one number per cell. The paragraph is the honest split, and it is not uniform: on the
  small-molecule datasets the hybrid is ahead of all three on nearly every dataset,
  downstream model and budget, while on the antibodies recursive elimination is ahead
  more often than not and on CHO polyreactivity it is ahead everywhere, and on the
  Buchwald--Hartwig reactions it is ahead at nearly every budget. mRMR beats the hybrid
  wherever its feature cap lets it run, which is antibodies only. In the thesis this is
  `tab:method:feature_reduction` with mRMR in an additional file; here the comparison is
  one table and the curves are already in the figure above, so it needs no figure of its
  own. Source c17.
- **One figure, held-out score against the number of representations, one panel per
  dataset and downstream model, one line per budget.** Where the gain comes from.
  Three things are scored at the same budget on the same folds, ranked by the same
  importance method and fitted the same way: the budget filled from the best single
  representation; the budget filled from the best two, three, five and all
  representations, their features ranked against each other; and the searched hybrid,
  plotted at however many representations its features came from. The second of those
  is the search's global-pool allocation run by hand, so the three read as one axis:
  no diversity, diversity without a search, the searched hybrid. A hybrid that stops
  before the budget is still at that budget (Adelaide, 2026-10-30). In the thesis this
  is nowhere. Source c18.
- **One sentence beside the budget-as-maximum claim.** The share of all the inputs that
  the selected features still carry stops rising a few hundred features in: on the
  small-molecule sets it is flat from the smallest to the largest budget, Caco-2 being
  the clearest. Measured by predicting the full concatenation's leading principal
  components from the selected features on held-out rows. Source c09.
- **Table, one row per dataset and downstream model, blocks by metric.** The hybrid
  against an ensemble that fits one model per single representation and pools the
  predictions, each picking its members and its pooling rule on the fold's training rows
  only. The ensemble is ahead on most antibody datasets, the two sides are close on small
  molecules, and the hybrid still beats the best single representation on nearly every
  row — so this is the ensemble winning, not the single representations. Same shape as the
  thesis's `tab:method:ensembles`. Source c11.
- **One short paragraph beside that table, no table of its own.** What each one costs to
  deploy and to retrain. The hybrid deploys one model instead of several and stores far
  fewer bytes per sample. It does not escape the featurisers: on nearly every dataset and
  model its features come from as many representations as the ensemble uses. And its
  search costs so many model fits that paying it back would take a number of retrains of
  the same dataset that nobody will ever do, which is the honest answer to the compute
  argument. Source c11.
- **TODO (Adelaide, 2026-11-05): shape and finding once the meta-learner over models with
  their own hybrids has run.** Source meta_learner_own_hybrids.
- **One sentence in the main text, one table in the supplement.** The table: one row
  per dataset, downstream model and representation, with the features the hybrid took
  from it and the fall in score when they are shuffled. The sentence: on the
  small-molecule ADME sets the hybrid's score rests on five or six representations,
  none carrying more than about a third of the fall; on the antibody sets it rests on
  two or three. The thesis has this table as Additional file
  `app:add:hindsight_shuffle`. Source c22.
- **One or two sentences, or a short table if there is room.** Choosing the features
  narrows the 90% prediction interval, not only the error. Split-conformal intervals at
  the hybrid's searched budget against the best single representation, width given as a
  fraction of the held-out label range, one number per dataset and downstream model. In
  the thesis this is nowhere. Coverage stays out of the comparison: split conformal sets
  the interval from the calibration residuals, so it meets its 90% whatever the features
  are, and the conditions cannot be told apart by it. The caveat beside it: on three of
  the antibody binding scans coverage falls well below 90%, so those are reported with
  their coverage beside the width or left out. Source c08.
- **One paragraph, no table in the main text, no figure.** The SVM's `C` couples to the
  budget the search picks and no other model's capacity setting does: a tightly
  constrained SVM picks the same small budget on every fold and a loosely constrained one
  picks the largest, cleanest on rat microsomal clearance. It runs the other way on
  nanobody thermostability. The DNN's hidden-layer size does not move its pick at all.
  The GP cannot be read, because its smallest length scales return no budget on several
  datasets. In the thesis this is one paragraph pointing to an appendix table. Source
  c06.
- **One sentence beside that paragraph, no table in the main text.** The held-out score
  moves as much with the regularization setting at the smallest budget as at the widest,
  and as much as on a single representation with no selection, so choosing the features
  does not replace the model's own regularization. In the thesis this is nowhere. Source
  c16.
- **TODO: shape and finding once c24 has run with the five scores per fold pair
  (Adelaide, 2026-10-31).** The 10-06 run had only the three-fold search against the
  fold's own recipe, which differ in training molecules and not in whose recipe was
  applied, so nothing from it is a transfer result. In the thesis the fold-to-fold
  version is nowhere; the dataset-to-dataset version is opig_blueprint_transfer. Source
  c24.
  <!-- TODO: tentatively expand this using c05's results. Decide together what c05 adds
       before writing it. -->
- **Table, one row per dataset and model: dataset, model, score on training data, score
  on held-out set, random score on held-out set.** On some datasets the training and
  held-out scores match, so there is no evidence of overfitting; on others the held-out
  score is lower, which is overfitting. TODO after the c33 rerun: whether the selected
  features beat random features on the held-out set, and which datasets fall on each side.
  In the thesis this is the overfitting section (`sec:method:results_overfit`). Source c33.
- Whether the budget a dataset picks tracks measurable properties of that dataset:
  effective rank and participation ratio of its joined representations, how linear and
  how locally smooth its labels are. One Spearman per descriptor and per model over the
  datasets, and a split of the budget spread into between datasets and between a
  dataset's two models. General patterns only. How much of this is reported is decided
  from the results (Adelaide, 2026-10-30). The thesis says this was not tested. Source c03.
- **Table, one row per source--target pair and shared downstream model, blocks by
  metric, plus a short paragraph.** Four columns: the best single representation, the
  target's own hybrid, the source recipe re-picked on the target, the source's exact
  features. On LogD to hERG $K_i$ the carried hybrid beats the best single
  representation with both models; with the SVM it matches the target's own hybrid,
  with the DNN it closes about half the gap; re-picking the features on the target or
  keeping the source's makes little difference. The paragraph says which pairs carry
  and which do not, by domain, and whether a QM9 property carries into an assay. The
  pairs are `tests/experiments/blueprint_transfer_pairs.txt`. Source
  opig_blueprint_transfer.
- **TODO (Adelaide, 2026-11-01): case study, does a new fingerprint carry useful signal?
  A short subsection; the finding below is what is expected, not yet measured.**
  cgr_morgan scores close to the best single representation on Buchwald--Hartwig: in
  the 10-06 run it is the best with the RF on most folds and just under the best with
  LightGBM, so on score alone it is a near miss. The hybrid says more than the score does:
  - Table, one row per reaction dataset and downstream model: each representation's
    held-out score on its own. Where cgr_morgan sits beside the other condensed-graph
    representations and the datasets' existing ones.
  - How many of the hybrid's features come from cgr_morgan on each fold, and whether
    those features alone keep its signal: those features scored alone beside cgr_morgan
    at its full width.
  - Where the signal is spread: how much cgr_morgan overlaps with the other
    representations (linear CKA), and the hybrid with and without the condensed-graph
    representations, per fold.
  - One sentence: whether counts of bonds formed and broken, added as extra features,
    help a representation.
  The point to close on is the method, not the fingerprint: the search compares
  fingerprints by what the model takes from each, not only by score under one model.
  Source reaction screen (b10), opig_cgr_comparison.

**Discussion**

- When to use it, as an ordered rule.
- Spread between the kept features does not explain the gain. One sentence: principal
  components are more spread out than the hybrid at the same number of features on every
  dataset, model and budget, and they are the comparator the hybrid beats. Sources c09
  for the spread, c02 for the scores.
- At the same number of features, the hybrid's features leave the model less room to
  memorise. One sentence: with the labels shuffled among the training rows, the model
  fitted on the hybrid's features reaches a far lower training-row R-squared than on
  principal components of the same width, and on the real labels the hybrid is the
  better of the two on the held-out rows. Measured on training rows, so it is a property
  of the feature set and not a held-out result. The only comparator is principal
  components at the same width; there is no number for a single representation. Source
  c15, from the results that already exist -- c15 is not being rerun.

- **At the same number of features, the features the search kept hold no more label
  information than principal components do.** One or two sentences, stated as found: it
  goes against the idea that the search gathers the label information into the features it
  keeps. It is also the one experiment in this group where both sides have the same number
  of features, so it does not carry the caveat the distance comparisons carry. Report the
  kept features minus random and the kept features minus principal components, never a
  value on its own, and say what the quantity is in the same sentence (Methods). In the
  thesis run, on held-out rows the kept features held more than principal components of the
  same number in under half the units on 15 of 18 datasets, and beat a random subset of the
  same number in about half the units on most datasets. Source s03.

- What explains the unevenness between datasets.
- Limitations: the search is costly and must be rerun when the data, inputs or model
  change; each dataset's grid was cut by hand; adaptive pruning exists and was not used;
  the search returns nothing below a handful of groups, whatever the row count.
- One sentence in limitations, from a null. Splitting a dataset into an easier and a
  harder half -- by how common each group's scaffold is, by molecule size, by how much
  the label varies inside a group, by how much the features vary -- and giving each half
  its own search gives the same held-out score as one search on all the rows, in every
  metric. So there is no evidence that the recipe has to be refitted per subpopulation,
  and none that it has to be refitted on the hard part of a dataset. The partition could
  not be built at all on the datasets with only a handful of groups, which is the same
  floor as above. Written from the results that exist; not being rerun. Source c07.

## Experiments to run

c01, c02, c03, c04, c06, c08, c09, c11, c16, c17, c18, c19, c21, c22, c23, c24, c33, opig_blueprint_transfer, opig_cgr_comparison (after the reaction screen), opig_generator_scoring_swap (tentative: in only if the arms separate), opig_modnet_feature_scores (needs c02 on the Matbench tasks first; server only), opig_scaffold_hybrid (flab_expression_koenig and the two Adams scans; flab_binding_koenig if it finishes), opig_shared_vectors_census (resume: the datasets after binding free energy, and the two faults), opig_shared_vectors_remedy (hERG Ki, Caco-2, LogD), opig_trastuzumab_enrichment (tentative: the zero-shot ranking only), s03 (held-out rows only), meta_learner_own_hybrids (name describes what it computes; no name from Adelaide yet; the ledger's `meta_learner` step, only on datasets she names)

## Needs work

<!-- Add to this list. One line each. -->


- c18 — rerun from scratch on all the datasets (Adelaide, 2026-10-30). The code now
  runs the pooled level at 1, 2, 3, 5 and all representations and records how many
  representations the searched hybrid used, so the stored files were written under
  the old design and are not read back.

## Experiment inventory

One bullet per experiment: what it found, whether it reached the thesis, and whether it
should go to the paper. Being filled in from the result files now.

Only four complexity experiments are cited anywhere in `thesis.tex`: c04, c06, c09 and
c23. Coverage is thinner than the folder listing suggests — the run ledger counts
(dataset, condition) units, and for several experiments most units are pending.

<!-- TODO: fill this list, then decide the takeaways from it. -->

## Open questions

<!-- TODO: the six nucleotide datasets have stored hybrid searches (whole-dataset, pulled
     2026-10-03, models in DECISIONS.md) but never made it into the thesis. -->
<!-- TODO: the Buchwald--Hartwig additive-group count is 22 in two places and 23 in the
     loader docstring. -->
<!-- TODO: no benchmark outside chemistry can be registered — registry.py allows six
     domain strings and none fits. c11 has no option for fixed outer folds. -->
