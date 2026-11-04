Sources behind every number here are recorded in `PAPER_OUTLINE_NOTES.md`, and the
per-experiment inventory is in `PAPER_EXPERIMENT_TABLES.md`, both beside this file.

---

# Paper 3 — detecting and handling noisy labels

Three questions kept apart: can the suspect labels be found, does acting on them improve
the held-out score, and does either answer change with the feature space.

## Takeaways

- Whether a noise-handling method works depends on the model and the feature space it
  runs in, so methods cannot be ranked on their own.
- Most reported gains are better training, not noise removed. A dataset with computed
  labels tells the two apart.
- Where the data record how a label was produced, that record beats every detector.
- Ranking the wrong labels above chance is not enough to improve the score.
- Deleting behaves like removing noise. Changing a label or counting it less behaves like
  regularisation.
- A better feature space helps prediction under noise but does not make wrong labels
  easier to find.

## Sections

**Introduction**

- Noise in measured labels, and why finding it depends on the feature space.
- The three groups: finding the suspect labels, acting on them, changing the model.
- Most of what follows reads a distance: a detector that compares a label with its
  neighbours' labels, a cleanup that spares a molecule because its neighbours agree, a
  nearest-neighbour prediction used as a reference. So the feature space the neighbours are
  found in sets what any of them can do. In every single feature set, the distance between
  two held-out molecules rises with the difference in their labels only weakly, which is
  why that space is a choice and not a given. Source s01.

**Methods**

- The six kinds of injected damage and the five flip rules, with the unit of each level.
- The detectors: what each reads and what it scores.
- The mitigation methods, by what selects the molecules and what is done to them.
- Changing the model instead, and the twelve methods for learning with noisy labels.
  (Note 2026-10-29: the method called uncertainty curriculum is renamed self-paced learning.
  Two variants that use the network's uncertainty were tried on one fold each of three
  datasets and dropped 2026-10-30. Note 2026-10-30: contrast-to-divide is renamed Jo-SRC
  weighting, Jo-SRC's noisy-sample test used as a weight; Jo-SRC as published was built and
  cut, both regression forms of its clean test losing to the plain DNN under noise on one fold
  of hERG Ki and logD. Note 2026-10-31: repeating find-and-delete (the trimmed refit) is cut
  from the detector-by-cleanup experiment; it raised the top score in no more cells than
  deleting the same number at random.)
- The split (2026-10-30): every method with a test and an action inside it is taken apart,
  its test joining the detectors and its action joining the cleanups, so the two can be
  recombined. The published method still runs whole. The recombination is reported as a
  result of its own.
- What counts as a method having raised the score.
- Datasets (Adelaide, 2026-11-01). Small molecules: QM9 HOMO-LUMO gap, QM9 dipole moment,
  QM9 polarizability, hERG Ki, hERG IC$_{50}$ (repeat measurements), Caco-2, LogD, rat
  microsomal clearance, human microsomal clearance, MDR1-MDCK efflux, hERG blocking, and
  three ChEMBL targets from Dablander et al. 2023 (dopamine D2, factor Xa, SARS-CoV-2
  Mpro). Antibodies: anti-VEGF expression and binding scans (Koenig), binding free energy,
  EGFR and angiopoietin-2 binding, Ginkgo expression titer, AC-SINS, SMAC colloidal
  stability, HIC retention, CDR3 flexibility (one dataset per loop), anti-coronavirus binding, two
  trastuzumab--HER2 tasks, and the Adams expression and binding scans (added 2026-11-03).
  The two Adams scans use random folds (Adelaide, 2026-11-03), the only random split in
  either paper, because no grouping holds a mutation out of them: 5,000 variants of one
  antibody carrying 1 to 3 mutations each among 20 heavy-chain positions.
  \citet{Chungyoun2024}, the benchmark these two files come from, splits them at random.
  Paper 2's dataset paragraph gives the counts.
  Antibody humanness is in Paper 2 only: one representation
  solves it, so there is nothing for a noise method to show.
  Scope of the activity cliff work (Adelaide, 2026-11-04): small molecules only, while the
  paper as a whole keeps both small molecules and antibodies. No published work applies
  the term to antibodies or protein variants, and no external antibody cliff set exists to
  check a definition against. The cleanup that spares cliff molecules therefore runs on
  small molecules only, which is a scope statement and drops no dataset from any other
  experiment. None of the benchmarks outside drug discovery carries the cliff work either:
  DroneRF has nothing outside its features to measure similarity with, the
  reaction sets are in Paper 2 only, and five of Matbench's six targets are computed.
  No reaction datasets (Adelaide, 2026-11-02): Buchwald--Hartwig and the Doyle plate run are
  in Paper 2 only.
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
- The controls, and what each can and cannot show. Deleting at random shows whether the
  choice of molecules mattered. QM9's labels are computed, so a method that removes noise
  should leave it alone or make it worse.
- QM9 plays two parts (Adelaide, 2026-11-01). It is one dataset among the others, and it is
  the clean comparison point. Its three properties are the only clean data. Every measured
  dataset carries real noise as delivered, so a cleanup that raises its score with no noise
  added has found real noise: a success, not a failure. The ideal method does that and
  leaves QM9 unchanged or slightly worse. Keeping a random 90% of molecules, best of 20
  draws, is the yardstick for any best-of-several-settings result.
- **The feature space the neighbours are found in, measured on its own before any
  detector runs.** Five entries, all from the similarity suite. They establish what a
  distance can do on each dataset, which is the setup for every detector and cleanup that
  reads neighbours.
- **How far apart two samples are in one representation, against how far apart their
  labels are.** For every dataset, every one of its ten representations and every distance,
  the Spearman correlation between the distance separating two held-out molecules and the
  absolute difference in their labels. No model is fitted and no features are selected.
  Report one number per domain with its spread across folds. Do not say which
  representation came out best: the representations are scored at the number of features
  each one has, from 40 to 17,920, so one beating another is also one being wider than the
  other.
  The file also scores two other things, which the thesis never used and the paper does not
  need: how often a molecule's nearest training molecules have a far-away label, and the
  structure--activity landscape index of each pair. Leave both out.
  Two things to settle before it runs again. The inventory counts 18 datasets and the thesis
  outline counts 17 after one was removed. And the thesis outline records a gap nobody has
  explained: a status page implies that multiplying a single representation's features by
  their importance raises this correlation on hERG $K_i$ to about 0.33, where the experiment
  gives 0.03 to 0.09 without that step.
  Source s01.
- **The same correlation with the distance measured in a hybrid.** The measurement above,
  repeated with the distance taken in a hybrid the search returned, at each of the dataset's
  budgets and under each of the dataset's distances. Each side is scored twice, once on the
  features as they are and once with every feature multiplied by its importance.
  **Two changes to the comparison before it runs again (Adelaide, 2026-11-02).** First,
  choose the single representation it is compared against by this same correlation, measured
  on the fold's training rows. The thesis run chose it by how well a nearest-neighbour model
  predicts in it, so the two sides were chosen on different grounds. The correlation for
  every single representation is already measured in s01, so this changes which
  representation is named, not what is computed. Second, compare importance-multiplied
  features only with importance-multiplied features, and plain with plain.
  State either way: the single representation is scored with all of its features and the
  hybrid with 100 to 2,000, and the hybrid is the best of a search while the single
  representation is one of about ten names. Both of those differences sit inside any lead.
  The part that needs no such caveat is what happens as the hybrid is given more features.
  Source s02.
- **Predicting each molecule from its nearest training molecules, in a hybrid's features
  and in one representation's.** Nothing is fitted: the prediction is the average of the
  labels of the nearest training molecules, so the only thing that changes between the two
  sides is which features the distance is measured in. A setting means one combination of
  distance (Manhattan, cosine, Minkowski p=0.5; two of the three on the larger datasets),
  neighbour count (5, 10 or 20), and how the neighbours are counted (by 1/rank or by 1 over
  the distance squared). Every feature space is scored at every setting, on the features as
  they are and with every feature multiplied by its importance.
  The result is read two ways and they do not agree: at one setting fixed in advance
  (Manhattan, 5 neighbours, 1/rank, features as they are), and with each fold given the
  setting that scored best on the other folds' held-out rows. Report both.
  The two sides do not choose from the same number of options. The hybrid side has 6
  feature spaces -- two hybrids for each of the dataset's two downstream models, the hybrid
  from the search that optimises the distance, and the hybrid from the search that optimises
  how well distance follows the label difference -- against 2 single representations, each
  scored over the 18 settings above (12 where the dataset has two distances). Choosing the
  best of more options raises a score, so the second reading favours the hybrid.
  **One change still to decide (Adelaide, 2026-11-02):** either give the two sides the same
  number of options, or choose each fold's setting on that fold's training rows instead of
  from the other folds' held-out rows. The second removes the imbalance rather than
  describing it. Until one is done, the second reading has to be reported with the option
  counts beside it.
  The search that optimises how well distance follows the label difference could not run on
  binding free energy or the four Ginkgo sets, so the hybrid has one fewer option on those
  five.
  In the thesis run the hybrid was ahead on all five small-molecule datasets under both
  readings, ahead on 8 of 10 antibody sets at the fixed setting and behind on 7 datasets
  when the setting was chosen per fold.
  Source s05.
- **The number of features a model predicts best at, against the number distance follows
  the label difference best at.** The sweep is the question, so both sides are swept over
  the number of features: the dataset's two downstream models on their own hybrids at each
  budget, scored on the held-out rows in the dataset's own metric, and the hybrid from the
  search that optimises distance at each budget, scored by the correlation in the s01 and
  s02 entries above.
  **One change before it runs again (Adelaide, 2026-11-02): sweep both sides over the same
  budgets.** In the thesis run they were swept over different lists -- prediction over the
  dataset's general budgets, distance over its distance budgets. On Caco-2, LogD
  and SMAC colloidal stability the prediction budgets were 500, 1,000 and
  2,000 and the distance budgets 100, 250 and 500, so the two could only agree at 500 and
  prediction could not peak below it. On 7 of 17 datasets the two lists overlapped at one
  number of features only. Until both sides share one list, part of any gap is the two
  lists and not the data.
  Report ties as ties: in the thesis run Caco-2 with the neural net scored 0.2785 RMSE at
  1,000 features and 0.2783 at 2,000, so its peak is not at one number.
  Source s08.
- **Multiplying each feature by its importance, in the four things that read a distance.**
  Small, one entry. Nearest-neighbour prediction, a Gaussian process, a support vector
  regressor, and the dataset's two models given four extra features read off each molecule's
  neighbours, each fitted twice: on the scaled features, and on the same features with every
  feature multiplied by its importance. Same features, same number of features, same fold
  and same setting on both sides, so the comparison carries none of the caveats the entries
  above carry.
  Do not report the Gaussian process arm as a result. It is missing on 5 datasets, and where
  it ran many weighted fits ended at a length-scale bound where every held-out molecule is
  given the training mean -- in the thesis run 121 of 260 fits on Buchwald--Hartwig, which
  is not in this paper; take the count from a dataset that is. Report the support vector
  regressor as the kernel case and say the Gaussian process did not converge under
  weighting, with that count.
  The exponent applied to the weights and whether they are rescaled are both marked open in
  the code, so whatever the rerun uses has to be written down. q-RASAR needs a citation;
  `refs.bib` has none.
  In the thesis run weighting helped nearest-neighbour prediction on 15 of 17 datasets, by
  0.0178 on LogD with 300 of 312 settings better and 0.0270 on QM9, and went the other way on
  expression titer and antibody humanness; it hurt the support vector regressor on 10 of 13
  datasets.
  Source s10.
- Where the code does not compute the published method of the name it carries.

**Results**

Headings state the finding. The sections below are the results, one heading each, in this
order.

**What a distance can do before any detector runs**

The setup for everything that follows. Short: four tables and the paragraphs beside them.

- **One short table and one sentence, the premise the rest of the section is measured
  against.** The highest correlation any single representation and distance reaches on each
  dataset, with its spread across folds, grouped by domain. In the thesis run these ran from
  0.0155 Spearman on SMAC colloidal stability to 0.3885 on antibody humanness, median 0.1387
  over 18 datasets, with 6 of the 18 below 0.10 and eight datasets carrying at least one
  representation at zero or negative, all eight antibody sets. Small molecules: QM9 0.2789,
  Caco-2 0.2292, LogD 0.1470, hERG $K_i$ 0.1170, hERG blocking 0.0412. On the small antibody
  sets the spread across folds is as large as the number: anti-coronavirus binding 0.0942,
  spread 0.1002. Source s01.
- **Table, one row per dataset, the correlation in the hybrid against the correlation in one
  representation, plus one short paragraph.** Both sides with plain features and both sides
  with features multiplied by importance, in separate blocks, and a column for the number of
  features on each side. The paragraph covers how the correlation moves as the hybrid is
  given more features. In the thesis run the hybrid was ahead on 16 of the 17 datasets that
  ran, hERG $K_i$ 0.0729 to 0.2625 under Manhattan distance, and anti-coronavirus binding
  was the exception; against the single representation whose own correlation is highest it
  was lower on Caco-2, anti-coronavirus binding and expression titer, and on LogD and QM9
  multiplying a single representation's features by importance reached a higher correlation
  than the plain hybrid. In the thesis this is `tab:distance_tracks_activity`. Source s02.
- **Table, one row per dataset and downstream model, blocks by metric, plus one short
  paragraph.** Predicting each molecule from its nearest training molecules in the hybrid's
  features against the same in the best single representation's, in the dataset's own
  metric, at the setting fixed in advance and at the setting chosen per fold, both readings
  in the same table. The paragraph states the split by domain, small molecules against
  antibodies, and the number of options each side chose from under
  the second reading. Binding free energy is reported as its five folds, not their mean.
  In the thesis this is `tab:nearest_neighbour_prediction`, which gives one reading per
  dataset. Source s05.
- **Table, one row per dataset and downstream model: the number of features that predicts
  best and the number that tracks the label difference best, plus one short paragraph.** The
  paragraph gives the direction and how wide the gap is, and states whether the two sides
  were swept over the same budgets. In the thesis run, where they were not, the two
  disagreed in 149 of 184 cells and prediction wanted more features in 158 of 184, the same
  number in 22 and fewer in 4; MDR1--MDCK efflux predicted best at 1,000 to 2,000 features
  and tracked best at 100 to 250, and AC-SINS predicted best at 2,000 and tracked best at 50
  to 100. The usable statement is that a hybrid built to predict well is not the hybrid to
  measure distance in, and the gap is hundreds of features wide. Source s08.
- **One small table: multiplying each feature by its importance, weighted against
  unweighted, for nearest-neighbour prediction and for the support vector regressor.** The
  same features on both sides, so no caveat about the number of features or about having
  been searched. One line saying the Gaussian process arm did not converge under weighting
  and is not reported. Source s10.

**Can the suspect labels be found**

- First, before any detector: how accurate must a detector be for deletion to help?
  Flag sets built from the injected noise at a set accuracy (none right, a quarter, half,
  three quarters, all), each cleanup applied, and the accuracy at which the model starts
  to gain. Reported before any real detector, so every detector's accuracy below is read
  against it. (Author, 2026-10-31: this result comes first in the results. It was the
  third bullet of "Does acting on them help".)
- Detection under each kind of damage, with the chance level beside every number.
- Which detector suits which damage, and why a model residual cannot see a label moved to
  a reporting limit.
- The space the neighbours are found in.
- Ranking above chance against knowing which to flag.

**Does acting on them help**

- The clean-label check, sorting every method by what it does to QM9.
- No noise added: each method's result on the measured datasets beside its result on
  QM9's three properties. A gain on a measured dataset with a null or slight loss on QM9
  is the ideal.
- Deleting against changing a label against counting it less.
- The labels as delivered, with the number of held-out samples beside every gain.
- Changing the model instead, and how much of that survives the clean-label check.

**Which labels the features are chosen with**

The labels that rank the features need not be the labels the model is fitted on. Four
combinations.

**Labels clipped at an assay reporting limit**

Short and self-contained. The record is in the data, so no detector is needed. Five
methods were tried under clipping and nothing else, which the paper states as that.

**Labels pooled from different papers: anti-coronavirus binding (Adelaide, 2026-11-01)**

Short and self-contained, like the clipped labels. The binding labels come from different
papers, the antigen is not recorded, and identical sequences differ by up to 1.26 log
units between papers. Pooling measurements from many sources should give a lot of label
noise, so this is a specific case of whether the methods help where the noise is known to
be large. 141 antibodies. The thesis left it out for these label problems; here they are
the point.

**Effect by dataset and by feature space**

- Small molecules.
- Antibodies, with the number of held-out antibodies beside every gain.
- Repeat measurements on hERG IC$_{50}$.
- The hybrid against the best single representation, as three separate answers.

**Discussion**

- Choosing a method for a new dataset, as an ordered rule.
- Limitations: most results on the first of five outer folds; each method's best setting
  picked on the rows that score it; one dataset with computed labels, and three properties
  of it.

## Experiments to run

opig_activity_cliff_qsar, s01, s02 (the single feature set chosen by the correlation being
reported; importance-multiplied compared only with importance-multiplied), s05 (both
readings; the setting imbalance fixed or stated), s08 (both sides swept over the same
budgets), s09 (builds the per-fold hybrid-minus-single-feature-set tables from the others;
run it last), s10 (small; the Gaussian process arm not reported)
opig_moleculeace_cliff_check

## Needs work

<!-- Add to this list. One line each. -->

- Noise grid decisions of 2026-11-04 (kinds cut, `base_1` and hybrid only, checks after results) are in `PAPER_3_NOISE_EXPERIMENTS.md`; the code does not reflect them yet (`HANDOFF_PAPER3_NOISE_IMPLEMENTATION.md`).
- s09 — has never run; no folder exists. It fits nothing and assembles the per-fold
  hybrid-minus-representation tables from s02, s04, s05, s06 and s08, which is the shape
  every similarity table in this outline takes. Two changes needed: s04 and s06 are not in this paper, so those inputs become optional rather than required, and it pairs each
  hybrid with the single representation its input chose, so s02's change of that choice has
  to land before s09 reads it.

## Experiment inventory

<!-- TODO: same list for the noise experiments, after Paper 2's is settled. -->

- opig_activity_cliff_qsar — the three ChEMBL targets of Dablander et al. 2023 (dopamine
  D2, factor Xa, SARS-CoV-2 Mpro). Scores held-out prediction and, separately, whether a
  pair is called a cliff. Moved here from Paper 2 (Adelaide, 2026-11-01).
- opig_moleculeace_cliff_check — every cliff rule in `tests/experiments/noise/cliff_rules.py`
  scored against the `cliff_mol` column MoleculeACE's own authors ship, on their 30 ChEMBL
  targets, 48,714 molecules in the files on disk (the research session quotes 48,707 for the paper). Fits no model and uses no split. `cliff_mol` is one published
  rule's output, so this repo's implementation of that same rule recovering it is a check
  on the code, and every other rule's distance from it measures how far two definitions sit
  apart rather than ranking them. The 30 CSVs are fetched once from molML/MoleculeACE into
  `/Volumes/seagate/moleculeace_benchmark` (Adelaide, 2026-11-02).

<!-- TODO: no published method confirms that a pair is really an activity cliff, and the
     paper should say so where it introduces the cliff work. Searched 2026-11-04. What the
     literature does instead is curate the data BEFORE detecting cliffs, so that false
     cliffs are prevented rather than tested for: equilibrium constants only and never
     IC50, Ki and Kd pooled, because mixing assay types manufactures cliffs out of unit
     differences; censored values dropped, since a bound has no gap; records carrying
     ChEMBL's own data-validity flag excluded; a potency gap of at least two orders of
     magnitude for a "high-confidence" cliff; and, in van Tilborg et al. 2022, a compound
     dropped when the standard deviation of its own repeat measurements exceeds 1 log unit,
     which removed about 4.33% of their molecules over 30 targets. Applied to this paper's
     hERG IC50 that rule drops 69 of the 1,218 molecules measured more than once, 5.67% of
     those and 0.70% of all 9,802 (measured 2026-11-04). The only validation of a cliff in
     the literature is prospective: the SAR Matrix method predicted an MMP-1 cliff, the
     compounds were synthesised and retested, and the predicted partner came out 60-fold
     more potent. Not available here. The consequence for this paper: the test of whether a
     cliff definition works has to be functional -- does naming those molecules improve a
     cleanup, against sparing the same number at random -- and that is not a weaker stand-in
     for an established method, because there is no established method. -->

## Open questions

<!-- TODO: the published similarity cutoff finds almost nothing on this paper's datasets,
     and the paper has to say so. Measured 2026-11-02 by the threshold sweep in
     tests/experiments/noise/cliff_census.py --sweep, 2,100 settings per dataset, written up
     in thesis_provenance/CLIFF_DEFINITIONS_2026-11.md section 3c. At the usual ECFP4
     Tanimoto 0.90, with the activity gap at twice the labels' median absolute deviation:
       hERG Ki, 1,412 molecules: 103 of 996,166 pairs similar, 5 cliff pairs, 0.6% of molecules
       Caco-2, 2,158 molecules: 143 of 2,327,403 pairs similar, 7 cliff pairs, 0.6%
       MDR1-MDCK, 2,641 molecules: 6 of 3,486,120 pairs similar, 0 cliff pairs, 0.0%
     MDR1-MDCK has 2,318 Bemis-Murcko scaffolds for 2,641 molecules, close to one each.
     Taking the most similar 1% of pairs instead gives thousands of pairs on all three, but
     the cliff share then runs 30.5% / 74.5% / 88.9% of molecules. Both halves of that belong
     in the paper: the fixed cutoff is comparable across datasets and nearly empty, the
     per-dataset cutoff is usable and no longer means the same thing on two datasets. -->
<!-- TODO: activity cliffs. The ECFP4 Tanimoto 0.55 test calls 1,740 to 3,056 of
     Buchwald--Hartwig's 3,057 training molecules cliff molecules, and 19 of QM9's 7,982.
     Buchwald--Hartwig is out of this paper since 2026-11-02.
     The random control cannot run on Caco-2, hERG blocking or LogD. Redefining a cliff in
     a hybrid distance needs two decisions the code leaves open: what threshold means
     "similar" in that distance, and whether the spared molecules may come from the same
     features the detector reads. -->
<!-- TODO: stored confident-learning results predate the 2026-10-29 fix. -->
<!-- TODO: the nearest-neighbour Shapley value is the one detector above chance under
     clipping on every dataset tested, and was never used to clean. -->
