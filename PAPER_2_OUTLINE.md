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
- Datasets. Small molecules: QM9 HOMO--LUMO gap, hERG $K_i$, Caco-2 efflux, LogD, rat
  microsomal clearance, MDR1--MDCK efflux, hERG blocking, and from outside the thesis
  human microsomal clearance, the Lhasa hERG set and three ChEMBL targets. Antibodies: the
  twelve of the thesis, plus anti-coronavirus binding, the Adams binding and expression
  scans, the anti-VEGF binding scan, CDR3 flexibility and two trastuzumab--HER2 tasks.
  Nucleotides: the Nucleotide Transformer Revised histone-mark, promoter, enhancer and
  splice-site tasks and four Genomic Benchmarks tasks. Reactions: Buchwald--Hartwig yield,
  the Doyle plate run and Suzuki coupling. Benchmarks outside chemistry: UCI HAR,
  Matbench, ProteinGym and the Kepler threshold crossing events. Rows, feature sets and
  metric per dataset go in a table, and so does each feature set's width; both come
  from c01, which also holds every single feature set's score per dataset and downstream
  model.
- **Two or three sentences on importance as a distance weighting.** Each feature is
  multiplied by the square root of its importance, weights scaled to average 1, the
  importance computed on the fold's training rows only. Each held-out sample's label is
  then the average of its 10 nearest training samples, weighted by 1 over distance. No
  model is fitted and no search runs. Source c04.
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

**Results**

Headings state the finding. To be written from the inventory below.

- **Table, one row per dataset and downstream model, blocks by metric.** The hybrid
  against the best single representation, and against the single representation that
  turned out best on the held-out samples. Both picked and scored the same way, per fold.
  This is the thesis's `tab:hybrid_vs_best_single`, which gives means over five folds;
  show per-fold scores here, because the thesis carries the fold detail in prose and a
  paper has no room for that. Source c02.
- **Table, one row per dataset and downstream model, blocks by metric.** The hybrid
  against principal components of all representations joined, at the hybrid's own budget
  on each fold, so neither side's width is chosen on held-out samples. In the thesis this
  is prose only, with no table. Source c02.
- **Table, one row per regression dataset, blocks by metric.** Prediction from the 10
  nearest neighbours in all inputs joined: unweighted, then weighted by each importance
  method, in the dataset's own metric. The importance the search uses to rank features
  also works as a distance weighting. The thesis table shows 13 datasets; 20 have
  results. One sentence of prose beside it, and it is the caveat: there is no
  shuffled-weights comparison, so the gain cannot be charged to the importance rather
  than to any rescaling. No figure. Source c04.
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

c01, c02, c03, c04, c06, c08, c09, c11, c16, c17, c18, c19, c21, c22, c23, c24, opig_blueprint_transfer

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
<!-- TODO: author wants the MODNet OPIG experiment in this paper. Not run yet. -->

## Open questions

<!-- TODO: nucleotides have a survey of single feature sets and no hybrid search. -->
<!-- TODO: the Buchwald--Hartwig additive-group count is 22 in two places and 23 in the
     loader docstring. -->
<!-- TODO: no benchmark outside chemistry can be registered — registry.py allows six
     domain strings and none fits. c11 has no option for fixed outer folds. -->
