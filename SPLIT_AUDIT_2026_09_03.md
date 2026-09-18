# Splits, folds and leakage — audit of find_best_hybrid and the five experiment suites

Read 2026-09-02 to 2026-09-04 on branch `similarity-metrics-study`.
Scope: `src/kirby/`, `tests/experiments/_shared.py`, and the complexity, noise, similarity,
similarity_metrics, OPIG and federated experiments. `tests/experiments/framework/` was excluded.

**Everything below is a reading of the code. Nothing here is a measured result.**

159 defects were raised by 32 readers working over separate files. Each went to a second
reader whose only job was to refute it; 26 fell and are listed at the end so they are not
raised again. 133 stood.

Of those 133: **10 were already gone** when the fixing started (9 fixed by the other session
working in this repo while the audit ran, 1 here), and **81 are fixed here** on 2026-09-04 —
about 60 individual code changes, since several findings covered many call sites each, plus
the 6 allocation-sweep findings (c30, c31, c32, exp_b) fixed in the same day's later pass,
the 10 OPIG findings fixed in a pass after that, and the 18 complexity findings c05 to c24
fixed in a pass after that, and the 8 data-and-plumbing findings fixed in a pass after that
(two group vectors, the TCGA batch correction, the resume check, the fold-0 prune, the
whole-dataset recipe label, f27's recipe representations, s07's held-out noise score), and the
10 engine findings in find_best_hybrid, hybrid and result_cache fixed in a pass after that.

**36 are still open.** By kind: 22 are the winner being picked on the folds it is scored on,
5 are a whole-dataset quantity entering a held-out score, 4 are group vectors, 2 experiments
that do not measure their claim, 1 the budget split, 1 cache, 1 arithmetic. The
random-internal-holdout and stratification categories are empty.

The counts move as each pass lands; the per-entry FIXED lines below are what is authoritative.
Three of these eight needed a decision and all three were made — see sections 3 and 5 above.

Every finding fixed on 2026-09-04 carries a FIXED line under its own entry below. Two of the
OPIG findings needed a decision and both were made: the scaffold-similarity pair sample is
now split by variant, with --n-pairs raised so each half keeps more pairs than the whole
sample used to carry; and opig_graphinity_ddg's within_foldx and within_rosetta conditions
were deleted rather than their datasets re-registered.

---

## The fact that decides everything else

Of the 1496 result files under `tests/results`, every file belonging to complexity, noise,
similarity, OPIG or federated was last written in 2026-06 or earlier. The 23 written since
are all framework experiments (f17 to f29). By modification time, **no publishable result on
disk was produced by the code as it now stands.** They have to be re-run whatever you decide.

So none of the decisions is "do I throw away my results". They are all "what do I fix before
the re-run".

---

## The one thing 53 of the findings have in common

When an experiment reports that the hybrid beat the best single representation by some
amount, both sides of that amount were usually chosen by looking at the same five folds the
amount is measured on. The hybrid side gets to pick the best of roughly 600 recipes — five
allocation methods, three importance methods, five internal evaluators, eight budgets. The
single-representation side picks the best of about ten.

Picking the best of 600 on the same rows you then report always flatters, and it flatters
more than picking the best of ten does. So part of every hybrid-minus-best-representation
number currently in the project is that gap and not the hybrid working. Nothing measures how
big the gap is.

There are two pieces of machinery for this and neither is widely used.

`nested_cv_evaluate` re-runs the whole recipe search inside each outer training fold, so the
winner is chosen without ever seeing the rows it is scored on. Across the five suites, one
file calls it: `opig_scaffold_hybrid.py`. The only other callers are f26, f27 and f28, which
are framework experiments.

`fold_reselection` is the cheap version. It does arithmetic over per-fold scores that already
exist: for each held-out fold it re-picks each side's winner using only the other folds, so
both sides pay the same tax. 18 experiment files call it. It cannot repair the list of
candidates, which the search already pruned using every fold.

A memory note dated 2026-07-13 records `fold_reselection` as deleted and the suite as migrated
onto `nested_cv_evaluate`. Neither is true of the code today. That note has been corrected.


---

## The five things only you can decide

### 1. Whether the re-run pays for an honest hybrid-vs-best-representation number

Right now that number is the best of ~600 recipes against the best of ~10 representations,
both scored on the folds that picked them.

Cheap option: for each held-out fold, re-pick each side's winner using only the other four
folds, then read that fold's score. Costs nothing — the per-fold scores already exist, and
18 files already build the input for it. It fixes which recipe wins. It does not fix the
list of recipes, because the search dropped configurations using fold 0 before the list was
made.

Full option: run the whole recipe search separately inside each outer training fold. Five
outer folds means five searches instead of one, so five times the search cost per experiment,
per dataset.

My recommendation: full option for whatever hybrid-versus-best-representation number goes in
the thesis; cheap option everywhere else.

The affected experiments are c02, c05, c11, c12, c14, c15, c16, c19, c22, c23, c24, c30,
c31, c32, exp_b, n11, n13, s02, s05, s07, s08, s11, s12, s13, the five OPIG experiments,
fed01, fed02, fed03, fed05, and four of the similarity-metric scripts.

### 2. Whether you accept the compute cost of the budget-split fix I already made

Each fold used to be handed representation scores that had been measured over the whole
dataset, and it split its budget with them. So even a per-fold rebuild carried information
from the held-out rows into that fold's features. I changed it: a fold now scores the
representations on its own training rows.

That means every fold rebuild re-scores all ten representations. Your own note from
2026-08-27 measured that at 336 seconds per call on flab_thermo_tresanco, and the earlier
attempt to precompute those scores was reverted because it changed which features got
selected — which is the same defect from the other side.

If the cost is unacceptable, the alternative is to compute the scores once per fold, on that
fold's training rows, and hand them in. That is correct and cheap, but it is a change to how
every experiment calls the rebuild, not a one-line default.

### 3. Two group vectors that are not groups — DECIDED 2026-09-04

**The four Genomic Benchmarks datasets** (`gb_human_nontata_promoters`,
`gb_human_enhancers_cohn`, `gb_demo_coding_vs_intergenomic_seqs`, `gb_demo_human_or_worm`)
assign each sequence to one of five folds by hashing the sequence to a number. A hash carries
no biology, so those five folds are an arbitrary partition — the thing the no-random-splits
rule exists to prevent, wearing a group label. The sibling nucleotide loader already groups
by chromosome. Two of these four datasets ship no coordinates at all through the path the
loader takes, so chromosome grouping is not a drop-in.

Your call: find a real key (chromosome where it exists, sequence-identity clustering where it
does not), or stop presenting these four as group-split results.

**Decided 2026-09-04: k-mer clustering.** Each sequence becomes an L2-normalised count over
its 5-mers and those profiles are clustered into 50 groups; the folds come from the clusters.
50 rather than 5 so GroupKFold can still balance the fold sizes. The shipped train/test split
was the other option and was not taken. `_assign_kmer_groups` in `nucleotide.py`.

**The seven Jain endpoints and `flab_immunogenicity_marks`** cluster antibodies at 95%
sequence identity, which gives one antibody per group — measured 136 groups over 137 samples.
This is *not* leakage: a reader measured the panel's nearest-neighbour identity at 0.823
median, with one pair out of 137 at or above 0.95. The panel genuinely has no near-duplicates.
What it means is that the claim these datasets support is "generalises to an unseen clinical
antibody", not "generalises to an unseen germline family".

Your call: keep the split and describe it accurately, or group by V-gene family — the code
for that already exists in the cross-species loader.

**Decided 2026-09-04: keep the split, describe it accurately.** The claim is unseen-antibody
generalisation. The loader now measures what the clustering produced and returns it as
`group_scheme`, which `load_data` carries into every result file, so a later reader cannot
take these folds for homology families.

### 4. Whether the noise survey's cleaning methods may choose their stopping point on the
labels they are graded against

Seven of the twelve learning-with-noisy-labels methods pick which training epoch to keep by
looking at the held-out labels, and those are the same labels the survey then scores them on.
So the survey is partly measuring how well each method can peek.

Fixing it means every one of the twelve methods carves its own selection slice out of its
training rows and never sees the held-out labels. That is a change to twelve classes and it
will lower every method's score, probably by different amounts — which is the point, but it
also means the current ranking cannot be compared to the new one.

Your call, because it changes what the survey measures.

### 5. Batch correction on the lung cancer dataset uses the class label — DECIDED 2026-09-04

For `tcga_luad_vs_lusc`, the two cohorts are downloaded and batch-corrected separately. So
which correction a sample receives depends on whether it is adenocarcinoma or squamous —
which is the thing being predicted. The correction is also fitted over every sample of a
cohort before any split exists, including the ones later held out.

A reader checked and found this cannot currently run: the batch-correction package is not
installed anywhere in the environment files, and the three cached expression matrices predate
the commit that added the block, so the loader returns the cache before reaching it. It will
bite on the next cold rebuild.

Your call: correct both cohorts together in one matrix and fit inside the training fold, or
drop batch correction and let the tissue-source-site grouping measure the site effect it was
built to measure.

**Decided 2026-09-04: drop it, for every TCGA task.** Fitting it inside the fold is not on
the table while the folds are grouped by tissue source site: a fold holds out whole sites and
ComBat has no batch parameter for a site it never saw, so a correction fitted on training
rows cannot cover the held-out site. The site codes still travel with the data as
`tss_groups`. Anything that wants a corrected matrix has to change the group vector first.


---

## Fixed on 2026-09-04

About 60 code changes across 34 files, covering roughly 55 of the findings. Every file
still parses and imports. Grouped by what they change.

### Folds now respect the dataset's groups where they did not

- **Six places built classification folds without stratifying by class**, so a fold could
  end up with almost none of the minority class. `n01`, `s04`, `s05`, `s06` (two places),
  `s08` now pass the task type, which gives them `StratifiedGroupKFold`.
- **The split that decides each representation's share of the budget was a random 20% cut**
  in c03, c08, c09, c10, c24, n11, n12, n13, s11, fed02, fed04, fed05 and the federated
  helpers. All of them now cut on the fold's own scaffold or identity groups. This changes
  which features get budget, so their cached hybrids are stale.
- **`c08`'s conformal calibration rows were a random 20% of the training fold.** They are
  now cut on groups, so the interval is calibrated on residuals as hard as the fold it is
  reported on.
- **Five neural models cut their early-stopping validation slice at random**: the DNN, CNN
  and Bayesian-NN classifiers and the CNN and Bayesian-NN regressors. A group-aware helper
  for exactly this had been written months ago and imported nowhere; it is now wired in.
  The three noisy-label network trainers were doing the same and now use it too.
- **Two of the ten noise estimators the grid compares split without groups** while the other
  eight used them. Both now take the same grouped out-of-fold pass.
- **Hyperparameter tuning for XGBoost scored its trials on an ungrouped 3-fold split.** Now
  grouped.
- **`--tune` on the molecular and TCGA command-line drivers** picked hyperparameters over
  every scaffold group and then reported cross-validation over those same groups. Tuning
  now happens inside each fold, which the search already supported.

### Whole-dataset quantities no longer reach a fold

- **Each fold's budget split was taken from representation scores measured over the whole
  dataset**, carried inside the recipe. A fold rebuild now re-scores on its own training
  rows unless the caller hands it fold-local scores. This costs a rebuild per fold.
- **The similarity space was being re-standardised per feature** in n11 and n13, which
  divides out the block weights that define that space. Taken as built now.

### Caches can no longer serve a result across a boundary that changes it

- **The per-fold feature cache was keyed on row numbers and representation shapes only.**
  Two datasets of the same size could serve each other's features, and the same rows under
  different labels or a different grouping would hit. Dataset name, labels and groups are
  in the key now, so existing entries miss.
- **A run that prunes nothing would resume from an aggressively-pruned run's saved state**,
  inheriting its dropped configurations while reporting a full search. It now refuses.
- **A run using per-fold fine-tuned representations could be served the frozen run's score**,
  because those representations are not in the key. That cache is now off for such runs.

### Numbers that could be wrong by arithmetic

- **A configuration that failed and scored NaN could take a top-three place** in the
  end-of-first-fold ranking and push out one that actually scored, because NaN sorts as if
  it beat everything. It now sorts last, matching the sibling ranking.
- **The refinement pass built its folds without the task type**, so on a classification
  dataset its rows came from different folds than the first pass and the two were then
  ranked together. It now receives the same fold definition.
- **`bias` subtracted a mean over all folds from a mean over the folds that survived**, and
  the spread fields printed 0.0 when one fold contributed. Both now report nothing rather
  than something wrong.

### Experiments that were measuring the wrong thing, or nothing

- **`s08`'s dimension sweep rebuilt every step at one fixed budget**, so the axis it exists
  to sweep never moved. Each step now rebuilds at its own budget.
- **The transfer arm of the binding-affinity-change experiment had an overlap filter that
  never ran**, because the loader dropped the complex identifiers it needed. The identifiers
  are carried through now, and the arm fails loudly rather than reporting a transfer score
  with the same complexes on both sides.
- **The component-split experiment's inner search silently dropped to two or three folds**
  on two of its axes; that warning was being swallowed by a blanket filter and now gets
  through.
- **`fed05` describes three silos in its saved result even when one failed to load.** It now
  records which loaded and why any did not.
- **`n09` averaged its base and hybrid arms over different folds** when the hybrid failed on
  some. Each arm now records which folds it scored.
- **The seed-invariance flag was written as true for every dataset**, and is false on any
  classification dataset because those folds shuffle with the seed. Written from the task
  type now.
- **The mutational shells used to group two antibody libraries are distances from the modal
  sequence.** In a combinatorial library nothing repeats, so that is whichever row came
  first in the file. Grouping on those shells now refuses instead of proceeding.

### The complexity experiments c05 to c24 — a later pass on 2026-09-04

Seventeen findings, in `c05` `c06` `c07` `c09` `c12` `c14` `c15` `c16` `c17` `c19` `c22`
`c23` `c24`. Every file still parses and imports. The per-entry note under each finding says
what changed there; what follows is what the seventeen add up to.

- **Three experiments now search inside each outer training fold.** `c14` and `c23` go
  through `nested_cv_evaluate` and report `delta_mean_of_folds`; `c24`'s cross-fold cells
  search on rows that exclude both the source and the target fold. That is five searches
  where there was one in `c14` and `c23`, and one search per ordered fold pair in `c24`.
- **Three now select features inside the fold.** `c17` moved all five strategies there,
  `c19` moved the hybrid build there, and `c15` builds on a training split before it scores
  kernel-target alignment on the rest.
- **Two now measure on the rows that did not choose the features.** `c09`'s information
  curves and `c17`'s per-feature information density.
- **Two verdicts moved onto a leave-one-fold-out number** that costs no fits, because the
  per-fold scores already existed: `c12`'s HELPS/HURTS and `c05`'s sample-efficiency curve.
  `c16` computes the same number per model family.
- **`c22`'s attribution scores are matched to each fold's own feature map before averaging**,
  and its selection score no longer carries a name that reads as performance.
- **`c07` gained a label-free partition** beside the one whose halves are decided by the
  labels they are compared on; nothing was cut.
- **`c06` stops borrowing importance scores built at a different fold count.**

- **`c11` searches the recipe inside each outer training fold**, by your decision. Its
  ensemble arm already chose everything inside the fold; the hybrid arm did not, so the gap
  between them ran the hybrid's way. `--c11-recipe-search whole_data` keeps the free path and
  renames every field it writes so none of them reads as held out.

Two things are worth carrying forward. `c17`'s per-fold mRMR ranking multiplies its dominant
cost — one mutual-information estimate per feature in the concatenated space, 16,069 of them
on `caco2` — by the fold count. And `c11`'s default now costs one pool of searches per outer
fold per dataset: about 12,000 hybrid builds and 60,000 model fits more than before, or
36,000 and 180,000 where augmentation representations exist.

---

## Fixed by the other session, while this audit ran

- `src/kirby/find_best_hybrid.py:2989` — A representation's score can average over fewer folds than the ones it's compared against.  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:3593` — Under top_2/top_3/top_5, a representation's train and validation features don't line up  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:3798` — Hybrid recipes are voided by one undefined fold; baseline representations are not.  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:3983` — struct_fid picks its comparison representation by nearest-neighbour score, not MCS fidelity  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:5286` — A NaN-scored configuration can be declared a model's winning recipe by list order, not score.  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:6143` — condition_suffix doesn't reach the cache, so different representation sets can share a score  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:6143` — f27 caches fits by dataset and fold only, so 'both' returns 'general's numbers as its own.  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/find_best_hybrid.py:7584` — search_mean averages folds that hybrid_mean drops when rebuild fails, so bias is not paired  
  Fixed by this audit.
- `src/kirby/find_best_hybrid.py:7686` — Predictive-blend baseline pick selects the worst representation, not the best  
  Fixed by the other session working in this repo, while the audit ran.
- `src/kirby/hybrid.py:4878` — A demoted representation validates on different features than it trained on  
  Fixed by the other session working in this repo, while the audit ran.
---

## Added 2026-09-04, not from the audit — both numbers in one file

Found by grepping this session, after the readers had finished. These files compute the
leave-one-fold-out number AND also write the search-side fields into the same result, so a
downstream reader can take the wrong one. That already happened to c02, whose saved files
carry the search-side delta and whose figures were drawn from it.

- `tests/experiments/similarity/s02_hybrid_similarity.py` — writes 1 search-side field
  beside its honest block. Its open entry below is about a different defect in the same file.
- `tests/experiments/similarity/s08_optimal_dimension.py` — same, 1 field.
- `tests/experiments/similarity/s09_cross_dataset_summary.py` — 2 fields. s09 has no open
  entry: the finding raised against it was refuted, because the number it would corrupt is
  never produced.

Already covered by an entry below, for the record: c11 (honest field still contaminated
there, because the leave-one-fold-out pass only re-picks the budget), c12 (the check exists
in the file and the verdict does not use it), s05 (the reader takes the fixed_ref number
instead of the held-out one).

Fix for all of them: drop the search-side field from the saved result, or rename it so no
reader can mistake it for a held-out score.

### Also added 2026-09-04, while fixing the federated package — fed01 searched labels no condition uses

Found while wiring `label_noise` into fed01. Not raised by any reader; all three are fixed.

- **fed01 ran its recipe search on the dataset's own continuous labels**, with the
  registry's metric, while C0, C5 and C6 all train on `labels_binary`. On caco2 and logd
  that meant a regression search ranked by RMSE picking the recipe for a two-class problem,
  and `metric_fn` scoring the silos inside `federated_feature_selection` on labels that are
  0 and 1 by then. Left alone, the `label_noise` fix would also have raised on both
  datasets and been swallowed: a classification NoiseSpec against `is_clf=False` is a
  ValueError inside `make_noisy_labels_verbose`. fed01 now passes `labels_binary`,
  `task_type='classification'` and `metric='auc'`, at the search and at all four
  `federated_feature_selection` calls.
- **fed01 read `data['precomputed_searches']` for a classification request**, the same
  defect the fed04:185 entry records. It now passes `data=None`.
- **fed01 applied pKi >= 6.0 to caco2 and logd**, the same constant the fed04:146 entry
  records, and the same collapse to one class. Both files now call
  `federated_helpers.binarise_labels`, which holds one cutoff per dataset in that dataset's
  own units and refuses a dataset with no row.
- **A cluster holding one class stopped fed01 and fed04 outright**, on fold 2 of a
  250-row check. `FLuID.cluster_data_space` finishes clustering at FLuID.py:538, then
  draws a chart whose y-values read `groups[(cluster, 'Active')]` for every cluster;
  a cluster with no active molecules has no such key. `cluster_training_only` now
  catches it and rebuilds `cluster_data` with the same slicing FLuID does at line 536.
  Behind it, fed01's teacher loops sent single-class silos to `RandomUnderSampler`,
  which raises on one class; both loops now require two rows and both classes.
- **Every warning in these experiments is silenced.** `_shared.py:45` and
  `find_best_hybrid.py:63` both call `warnings.filterwarnings('ignore')` at import.
  The two notices added here print instead.
- **fed03 Part A's `federated_feature_selection` call had to change too**, as a caller of
  the function the fed_helpers:1100 fix rewrote. It now passes the fold's scaffold groups
  and `task_type`; without them every silo would take the flat score and the budget would
  come out equal across silos.

---

## Swept 2026-09-18 — 3 of the 123 were still true, and all 3 are now fixed

Every entry carrying no FIXED marker was re-checked against the code as it stands.
The per-kind counts below ("53 open", "15 open") are from 2026-09-03 and are STALE:
a great deal was fixed on 2026-09-04 and afterwards without the entries being
closed. Read this section, not those numbers.

**89 of the 123 carry a FIXED marker already.** Of the remaining 34, **31 are
fixed too** and 3 are still true. Each of the 34 was checked by opening the code
the entry describes, not by trusting its line number — several had drifted by
more than fifty lines.

**ALL THREE WERE FIXED THE SAME DAY.** They were one change in substance — a
`create_hybrid` or a `fit` that had a group vector to hand and did not pass it —
and the table below is kept as the record of what each was.

**The 3 that survived the sweep are all kind B, and none contaminated a
held-out score.**
In every one the random carve sits inside the outer training fold, and the test
rows are cut group-aware before it. What changes is which features get admitted,
so a future run reports different numbers; nothing already on disk is wrong.

| entry | what it is | what a fix takes |
|---|---|---|
| `src/kirby/find_best_hybrid.py:2244` | f29's dnn and cnn carve their early-stopping holdout at random, not by scaffold group | a `groups` parameter on f29's `_fit_score`, `groups[tr]` at :296, forwarded to `m.fit` behind the same signature probe `_fit_model` already uses at :1267-1287. Moves every f29 dnn and cnn score in all three arms, so those rows need a re-run to sit beside the tree models. dnn's rank against them could move. |
| `tests/experiments/federated/federated_helpers.py:1103` | fed03's C6 `create_hybrid` gets no group vector, though `pooled_groups` is in scope at fed03:464 | one argument at fed03:740, `groups=pooled_groups[train_idx]`, and a group-aware splitter in place of the KFold at fed03:777. Moves C7's representation weights and so Part A's C7 MCC. |
| `tests/experiments/federated/fed04_cluster_aware_importance.py:89` | the same omission in fed05's four `create_hybrid` calls | four one-line edits at fed05:185, :486, :722 and :835, slicing the silo's group vector to the rows handed in. At :722 that vector is already built as `pool_groups`. Changes which features fed05 admits in a future run. |

⛔ ONE THING NOT ESTABLISHED. No fed05 result file was found in the repository,
so what fed05 last wrote is unknown. The claim that its only outputs are June
smoke runs is carried over from the original entry and was not re-checked.

The three are one change in substance: a `create_hybrid` or a `fit` that has a
group vector available and does not pass it. `internal_holdout_split`
(src/kirby/hybrid.py:5435) already uses GroupKFold whenever groups are present
and at least 2 are distinct, so passing them is all that is needed — that is why
the other eleven entries of this kind are already fixed.

---

## Every finding, in nine kinds

123 entries. Roughly 55 of them are addressed by the 2026-09-04 fixes above — the entry is
kept so you can see what was changed and why. Each gives what the code computes, why that is
wrong, what it touches, and the fix.

- **A** — The winning recipe is chosen by its score on the folds the reported number is measured on. 53 open.
- **B** — An internal holdout is drawn at random on a dataset that has groups. 15 open.
- **C** — The budget split comes from representation scores measured on every row. 7 open.
- **D** — The group vector is degenerate, or is not the group the question needs. 8 open.
- **E** — Classification folds lose stratification because task_type is not passed. 3 open.
- **F** — A quantity computed on every row enters a held-out score. 14 open.
- **G** — A cached result can be served across a boundary that changes it. 8 open.
- **H** — Arithmetic that lets a broken configuration win, or that unpairs the two arms. 8 open.
- **I** — The experiment does not run, or measures something other than its claim. 7 open.

---

## A. The winning recipe is chosen by its score on the folds the reported number is measured on

**53 open.**

**`tests/experiments/_shared.py:952`** · FIXED 2026-09-04  
Fold 0's held-out hybrid score picks from candidates fold 0 already helped filter.

- FIXED: `SearchResult.fold0_pruned` records whether the search filtered its own
  candidate list on fold 0. `build_fold_reselection_input` writes `exclude_folds: [0]`
  into the persisted block when it did, `fold_reselection(exclude_folds=...)` drops that
  fold's term from BOTH arms, and the 17 readers pass `hr_exclude_folds(block)`. A block
  with no such key reads as pruned: five allocation methods against `strategy_top_n=3`
  prunes under every schedule, including `default_pruning`.

- Computes: For each held-out fold, fold_reselection picks the best hybrid recipe from all_results by the other folds' mean score, then reports its held-out-fold score.
- Wrong because: Cluster submitters default to a prune that keeps only the top half of each allocation-importance pair's configs, ranked by fold-0 score. Fold 0's held-out hybrid score then gets picked from those survivors, while the single-representation score never goes through that cut.
- Touches: Affects one fold's term in the hybrid-vs-single-representation delta from c02, c05, c07, and four OPIG experiments' saved fold_reselection blocks. Saved JSON does not record pruning mode, so you cannot tell which files it hit.
- Fix: Either drop fold 0's term when computing the delta from a pruned search, or run with a schedule that prunes nothing (checkpoints=[], tail_drop=0, threshold=0) so nothing is fold-dependent.

**`tests/experiments/_shared.py:4513`** · FIXED 2026-09-04 (labelled, not re-searched)  
A hybrid recipe is picked using every row, then s02 reports its score as held-out.

- FIXED: `get_best_search_for_downstream` now returns `selection_scope: 'whole_dataset'`
  in its metadata, `fold_reselection(recipe_selection=...)` copies it onto every model
  entry, and s09 prints it beside the number as "recipe picked on all rows; only the
  features are refit". s02 and s08 pass it. s04, s05, s07 and s13 no longer need it —
  they search the recipe inside each fold. Routing the rest through `nested_cv_evaluate`
  is decision 1 above and was not taken here.

- Computes: get_best_search_for_downstream ranks each recipe — allocation method, budget, importance method, evaluator, augmentation condition — by its score averaged over inner CV folds on the whole dataset.
- Wrong because: In s02, that same score is then reported as the held-out delta, so the recipe was chosen using rows the delta claims to exclude. s05 and s06 rebuild features per fold from training rows only, so they keep the recipe choice, not contaminated feature weights.
- Touches: s09 already wrote s02's held-out delta into its output file, so that number on disk is affected. s04, s07, s08, s13, n01, n06, c15 use the same pattern; s05, s06, c19 do not.
- Fix: Route these experiments through nested_cv_evaluate, or have precompute_hybrid_searches (similarity_helpers.py:945-958) stamp its output as whole-dataset-selected so the delta is never read as held-out.

**`tests/experiments/_shared.py:4600`** · FIXED 2026-09-04  
s07's held-out noise score is fit on the same labels it is later scored against

- FIXED: s07 builds one hybrid per outer fold — the recipe searched on that fold's
  training rows and the features chosen from the same rows (`_outer_fold_hybrids`), and
  each contamination fold reads the matrix whose selection never saw its held-out rows.

- Computes: get_fitted_pool_hybrid selects hybrid features using every row and every label in the dataset, once, before s07 splits into folds
- Wrong because: s07_noise_bridge scores that one matrix per held-out fold against single-rep baselines that never saw any labels. The hybrid side already used a fold's own labels to pick its features before being scored on that fold.
- Touches: s09's noise_contamination_delta_heldout is mislabeled held-out; that number is already written to disk. c17's best_strategy, picked by mi_density, carries the same bias, but is secondary.
- Fix: Rebuild the hybrid feature matrix per fold in s07 with build_fold_hybrid, as s08 already does, or drop the held-out label on this number.

**`tests/experiments/complexity/c05_learning_curves.py:242`** · needs a decision first  
c05 picks its hybrid winner from far more candidates than its representation winner.

- Computes: At each subsample size, c05 takes the best hybrid config's own fold-mean score, and the best single representation's own fold-mean score, on the same folds.
- Wrong because: The hybrid winner is chosen from about ten times more candidates than the representation winner. Picking a max over more candidates on the same folds inflates the hybrid score most when folds are small.
- Touches: Every c05 result file on disk (28 checked) is missing the honest per-fold block, so today's sample-efficiency numbers, including the abagym_EGFR svm figures quoted in OPIG_SLIDE_CONTENT.md, are the biased ones.
- Fix: Rerun c05 and read the curve from fold_reselection_by_n, which it already writes and which scores every candidate on folds that did not pick it.
- **Fixed 2026-09-04:** c05 now computes the leave-one-fold-out curve from the blocks it already writes and stores it as `hybrid_reselected` / `best_single_reselected` / `reselected_delta_mean_of_folds` per model per sample size. Costs no fits. Still needs the re-run.

**`tests/experiments/complexity/c11_ensemble_vs_hybrid.py:419`** · needs a decision first  
c11's hybrid recipe and budget split are chosen using every fold's held-out labels

- Computes: c11 picks each budget's hybrid recipe, and each representation's budget share, from a score computed over every row of the dataset, not just each fold's training rows.
- Wrong because: The comparison assumes both arms see only their fold's training split, but the hybrid's recipe and per-representation scores come from every row, held-out rows included. fold_reselection only re-picks the budget itself; it doesn't redo the recipe or the per-representation scores.
- Touches: auto_minus_hybrid and accuracy_delta_ensemble_minus_hybrid would be biased toward the hybrid if computed today, but no file on disk carries them — the nine files in tests/results/c11_ensemble_vs_hybrid/ are from May 2026, before these fields existed.
- Fix: Either search for the recipe inside each fold's training split like nested_cv_evaluate does, or keep the full-dataset search and relabel hybrid_reselected as recipe-selected on all folds.
- **Fixed 2026-09-04:** both halves, as decided. c11 now searches the recipe inside each outer training fold by default, sweeping the same augmentation conditions the whole-dataset pool compares, so nothing is dropped for cost — one pool of searches per fold, five without augmentation representations and fifteen with. The ensemble arm already made every choice inside the fold; now both arms do, and the head-to-head is a comparison. `results[].recipes_per_fold` records what each fold picked, and `break_even`'s featuriser count is the union across them. `--c11-recipe-search whole_data` (also `run_dataset.py --c11-recipe-search`) keeps the old free path for a dataset whose number is not quoted; under it the file carries `hybrid_recipe_selected_on_all_rows`, `auto_minus_hybrid_recipe_all_rows` and `accuracy_delta_ensemble_minus_hybrid_recipe_all_rows`, and writes no `hybrid_reselected` or `auto_minus_hybrid` at all. `firstpass_notready.py` prints a recipe column and marks every file without the new marker — the nine on disk — as ALL ROWS.

**`tests/experiments/complexity/c16_feature_selection_as_regularization.py:365`** · needs a decision first  
c16's budget allocation uses whole-dataset scores inside every fold's hybrid

- Computes: `build_fold_hybrid` scores each fold's representation ranking using rep_scores from a search over all rows of the dataset, including that fold's own validation rows.
- Wrong because: Every budget's representation ranking is scored using rows from the fold it will later be tested on, not just its own training rows. That leaks into the heatmap, the sensitivity curve, and the L2-equivalence output — not only the hybrid_wins count.
- Touches: Every existing c16 result file's heatmap, sensitivity curve, and L2-equivalence numbers were computed this way, so all of them are affected.
- Fix: Decide whether per-budget representation searches should run inside each outer training fold instead of once over the whole dataset, then re-run c16.
- **Fixed 2026-09-04:** the budget split half is already fixed upstream: `build_fold_hybrid` no longer falls back to the recipe's whole-dataset representation scores, so every fold re-scores on its own training rows. c16 now stamps `budget_split_selected_on` and `recipe_selected_on` and computes `fold_reselection_result` per family. Still needs the re-run.

**`tests/experiments/complexity/c24_blueprint_transfer.py:181`** · needs a decision first  
c24's blueprint-transfer score is biased by an unfair recipe search, not by leaked features

- Computes: c24_blueprint_transfer.py computes mean_abs_delta, the gap between each fold's own hybrid score and its score using a recipe picked on another fold's training rows.
- Wrong because: The recipe for each cross-fold cell is picked using training rows that include that fold's own validation rows. Only five recipe settings cross folds — features get re-selected fresh on each target fold, so the delta cannot test whether feature selection generalizes.
- Touches: The three stored c24 result files already show mean_abs_delta above the 0.02 stable threshold (0.183, 0.050, one NaN), so no current file claims blueprints generalize. A future run could still show a false pass.
- Fix: Search each source fold's recipe on rows that exclude every target fold's validation rows, or relabel the off-diagonal as recipe sensitivity, not transfer.
- **Fixed 2026-09-04:** two changes together, the second being the transfer fix `EXPERIMENTS_FINALIZED.md` already records as decided for c24. Each off-diagonal cell's recipe AND features are now chosen on rows that exclude both the source fold and the target fold, so nothing that produced a cell has seen the rows it is scored on. Those features are then carried onto the target fold's rows rather than re-selected there, so the selection is what crosses — two folds sharing a recipe no longer give a delta of exactly zero. Both arms go through `build_fold_hybrid` instead of a bare `create_hybrid`, so the fold's groups, the internal evaluator, the augmentation settings and the dataset's own metric all reach the build. The one difference left between the diagonal and the off-diagonal is that the off-diagonal saw one fold less data; recorded per cell in `cell_recipe_provenance`.

**`tests/experiments/complexity/c30_balanced_global_selection.py:121`** · needs a decision first  
c30 picks the target representation using the folds it also scores every method on

- Computes: It ranks every staged representation by mean score on the same 5 folds, then makes the top one the target and the top 5-10 the shortlist for all three methods.
- Wrong because: The target and the shortlist are both picked from the same fold scores used to grade all three methods. That makes every delta reported against the target biased in an unknown direction, though comparisons between the three methods themselves stay valid.
- Touches: No c30 result file exists on disk, and c30 is uncommitted, so nothing reported is affected yet. Any future run's delta-against-target numbers would be unsafe to quote; the method-vs-method comparison stays valid.
- Fix: Rank representations per fold so the target and shortlist are chosen per fold, as c31 does, or state the leak and stop quoting target deltas as clean.
- **Fixed 2026-09-04:** c30 ranks the representations inside each outer fold's training rows, via a 3-fold inner group-aware CV over all ten. That ranking picks the fold's target and its pool5/pool10 membership; the target's score for a fold is that fold's held-out score for the rep its own training rows chose. `target.best_rep_per_fold` and `pool_per_fold` record the picks. c30 also could not run at all — it imported a deleted /private/tmp harness and wrote its results there — so it now uses `_alloc_harness` and writes `tests/results/complexity/c30_balanced_global.json`, which no longer collides with exp_b's filename. Costs about 150 extra fits per dataset. Needs the re-run.

**`tests/experiments/complexity/c32_allocation_strategy_cross.py:204`** · needs a decision first  
c32 leaks one oracle ranking into its N=2/3/5 picks and its comparator column.

- Computes: For each of 32 allocation strategies in a cell, c32 computes an R² difference against one target representation, across the same five outer folds.
- Wrong because: Picking the target from the same five folds it's scored against cancels out when ranking strategies against each other. But the set of 2, 3, or 5 representations each strategy draws also comes from that same ranking, and that does not cancel.
- Touches: C32_ANALYSIS_HANDOFF_2026_08_29.md's top-1/top-2/win-margin tables are unaffected. Its beats-target counts (1339/1728 cells, vs 1522 with the next-best representation) and its N=2/3/5-vs-N=10 comparison are both miscalibrated.
- Fix: Move both the target pick and the top-N representation selection inside each fold's training split, using the inner group-aware CV already built at line 245.
- **Fixed 2026-09-04:** the target is picked per fold from that fold's training rows and scored on its held-out rows; `target.best_rep_per_fold` records which representation each fold chose and `folds_agreeing_with_whole_data` how often that matched the old whole-data pick. The N=2/3/5 subsets come from the same per-fold ranking. Costs no extra fits — the inner group-aware CV was already there to size the slots. The 30 files under `/Volumes/seagate/kirby_cache/c32_allocation_cross/` predate this and are not comparable to anything written after it.

**`tests/experiments/noise/n11_estimation_remediation_grid.py:697`** · needs a decision first  ·  FIXED 2026-09-04
n11's best_config is the top of ~4,800 configs, scored on the same folds it reports on

- Computes: n11 scores about 4,800 space x estimator x remediation x alpha configs, then reports the highest recovery percentage as best_config.
- Wrong because: Every config is scored on the same folds used to report its recovery, so the winning score is partly luck. That printed number is not a fair estimate of how well that specific config recovers out of sample.
- Touches: best_config, top_10, and robust_configs in n11 are affected; pct_positive and alpha_effect are not. The buchwald_hartwig headline number (58.3%/72.5%) printed by firstpass_notready.py is that inflated maximum. The per-config recovery table is unaffected.
- Fix: Pre-register one config before running, split the folds in two to score the winner separately, or relabel these three fields as a search result, not a measured recovery.
- FIXED 2026-09-04: `best_config`, `top_10` and `robust_configs` are renamed `*_search_max` and carry a caveat naming them a search result. Beside them sits `heldout_recovery`: for each fold the winner is re-picked from the OTHER folds and read on this one, then averaged — pure arithmetic over per-fold scores now kept alongside the pooled ones, no extra fitting. It also reports whether one config won on every fold. `read_n11` in firstpass_notready.py prints both columns. On a fixture where one config spikes on a single fold the search maximum reads 68% and the held-out number 22%.

**`tests/experiments/noise/n13_noise_distributions.py:817`** · needs a decision first  ·  FIXED 2026-09-04 (renamed)
n13 tags a fixed recipe as `fold_reselection`; the per-fold reselection it promises never runs.

- Computes: n13 computes `delta_mean_of_folds` between one hybrid recipe and a fixed baseline representation, across the same 5 folds used to pick that recipe, and labels it `fold_reselection`.
- Wrong because: The `fold_reselection` key promises a recipe re-picked per held-out fold, but n13 gives it one recipe, so re-selection never happens. That recipe was chosen on all folds with clean labels, true of every recipe in the suite, and it does not inflate the delta.
- Touches: No analysis script reads `results['fold_reselection']` — read_n13 uses `summary.hybrid_advantage_persists` instead. The one n13 result file on disk predates this code, so nothing already reported changes.
- Fix: Either rename the key to drop its per-fold-selection promise, or feed every searched recipe so the re-selection is real — costing one refit per noise_type x level x fold x seed.
- FIXED 2026-09-04: the key is `per_fold_scores_single_recipe`, and the block carries `reselection: 'none_single_recipe_per_model'`, `recipe_source: 'clean_label_search_pool'` and the recipe count per model. The printed header no longer says per-fold reselection. Confirmed no reader breaks: `read_n13` uses `summary.hybrid_advantage_persists`, and nothing else reads n13's block.

**`tests/experiments/opig_blueprint_transfer.py:184`** · needs a decision first  
Native arm's score is measured on the same folds that picked its recipe

- Computes: For the target dataset it picks the recipe (allocation method, importance method, evaluator, budget) with the highest score across five folds, then scores that recipe on those same five folds.
- Wrong because: Under --mode full the search folds and the scoring folds are the identical 5-fold split, so the target-side comparisons come out biased conservative, not optimistic. The source-dataset check near line 406 picks its winner from a larger search than its baseline, inflating toward the paper's claim instead.
- Touches: Nothing published changes: the only saved file, logd_to_herg_ki.json, is a 200-row, one-fold, three-representation smoke run from 2026-06-05, not a used result.
- Fix: Score the native arm, the base_best baseline, and the source-side check with nested_cv_evaluate, searching and rebuilding the recipe inside each held-out fold.
- **Fixed 2026-09-04:** the native arm, base_best and the source-side check now take their recipe and their single representation from a search run inside each held-out fold's training rows. Costs n_folds searches per dataset instead of one.

**`tests/experiments/similarity/s02_hybrid_similarity.py:170`** · needs a decision first  ·  FIXED 2026-09-04 (features per fold; recipe still whole-dataset)
s02 picks the hybrid's features using rows it later scores as "held-out"

- FIXED 2026-09-04: `_cv_reselected_sac` now calls `build_fold_hybrid` inside each held-out
  fold, so the features come from that fold's training rows instead of a slice of the
  whole-dataset matrix. The recipe is still the pool search's — that is the separate
  `_shared.py:4513` entry above — and the file's LIMITATION note now says exactly which
  of the two is refit per fold.

- Computes: For each held-out fold, s02 scores the hybrid representation's distance-activity correlation, but picks that representation's features and recipe (allocation method, budget, importance method, evaluator) from every row, held-out rows included.
- Wrong because: The single representations it's compared against get no such selection, so the comparison isn't fair. The reported gap for the hybrid is biased upward by that mismatch.
- Touches: No result file on disk carries this correlation yet — all 23 existing s02 result files predate it. It will bias the next similarity run and the s09 cross-dataset table that reads it.
- Fix: Either rebuild each fold's hybrid representation from its recipe using training rows only, like s04/s05 do, or rename the output and add a caveat.

**`tests/experiments/similarity/s05_knn_prediction.py:144`** · needs a decision first  ·  FIXED 2026-09-04
s05 and s04 pick their best hybrid recipe on the same folds they score it on

- FIXED 2026-09-04: both files pick the allocation method, importance method and
  internal evaluator inside each evaluation fold, from a search run on that fold's
  training rows (`get_fold_best_search`, new in `_shared.py`). Folds now come from
  `nested_cv_outer_folds`, the split the per-fold search caches against, so fold i's
  recipe was searched on exactly fold i's training rows; the fold composition is the
  same group-aware split as before. Each cell records `fold_recipes` because the folds
  can pick differently. s05 also builds once per (budget, k, fold) instead of once per
  condition as well, which halves the builds it added. Costs one search per fold per
  dataset instead of one per dataset.

- Computes: For each budget, s05 picks the allocation method, importance method, and evaluator with the highest mean score across the same 5 folds it later reports results on.
- Wrong because: The recipe is chosen and scored on the same 5 folds, so the score is the best of 75 combinations measured on itself. Neither the fixed_ref number nor the leave-one-fold-out re-score fixes this, because re-scoring only re-picks budget, condition, and k, not the recipe.
- Touches: This affects the fixed_ref and "held-out" deltas s09 reads from s05 and s04 on molecular datasets run with --mode full. It does not affect classification runs, whose folds differ.
- Fix: Choose the allocation method, importance method, and evaluator inside each evaluation fold's own training rows, instead of once on the folds used to score results.

**`tests/experiments/similarity/s07_noise_bridge.py:390`** · needs a decision first  ·  FIXED 2026-09-04
s07 scores hybrid noise contamination on features selected using every fold's labels

- FIXED 2026-09-04: `_outer_fold_hybrids` builds one hybrid per outer fold — recipe
  searched on that fold's training rows, features selected from the same rows — and
  `_knn_noise_contamination_cv` takes a `matrix_for_fold` callable instead of one
  matrix. Each contamination fold reads the outer fold's matrix that never saw its
  held-out rows. This covers both fold schemes: a group lives entirely inside one
  outer test fold, so the one-fold-per-group scheme resolves to the same matrix
  without needing a search per group. The weighted arm now takes its weights from
  that fold and is scaled on train rows, instead of being standardised over every row.
  The two blocks get their own sets of hybrids when config's n_splits is under five:
  the k-fold block's folds are its own, and the one-fold-per-group block needs folds
  that hold out every row exactly once, which `--mode intermediary`'s single 80/20
  split does not give — it uses five there. At n_splits=5 both blocks share one set,
  so the usual run builds five hybrids, not ten.

- Computes: s07 computes a per-fold held-out noise-contamination rate for the hybrid representation, over rows sliced from one matrix built on all labels.
- Wrong because: The features in that matrix were picked using every point's label, including the labels each fold later holds out. Fitting the scaler and noise threshold per fold does not undo that earlier choice.
- Touches: No stored result file is affected: all 21 s07_noise_bridge JSONs predate the commit that added this held-out score, so none contain it. The next run will write it, and s09 will label it 'held-out'.
- Fix: Rebuild the hybrid per fold from that fold's train labels, and also rerun the search that picks which representations combine — it too uses every fold's labels.

**`tests/experiments/similarity/s13_clustering_retrieval.py:185`** · needs a decision first  ·  FIXED 2026-09-04 (recipe moved inside the fold)
s13 picks its hybrid recipe on all rows, but its docstring claims otherwise

- FIXED 2026-09-04: the recipe search moved inside the fold loop — each fold searches
  on its own training rows via `get_fold_best_search`. The docstring says so, and the
  result file carries `fold_recipes` (what each fold picked) and
  `top3_base_reps_first_fold`, renamed because that ranking is now fold 0's, not the
  whole dataset's.

- Computes: Before the fold loop, s13 picks one hybrid recipe — allocation method, importance method, evaluator, budget, augmentation — by searching every row.
- Wrong because: The recipe is picked with a kNN score close to the retrieval score s13 reports, so it sees rows it is later scored on. Everything else — the KMeans fit, kNN index, scaler, and column selection — is redone train-only inside each fold.
- Touches: No number changes; s02 already documents this suite-wide setup. s13's docstring falsely claims no free look, and its saved JSON doesn't record that the recipe was picked on all rows.
- Fix: Either move the recipe search inside each fold, or keep it as is but fix the docstring and log which rows chose the recipe.

**`tests/experiments/similarity_metrics/feature_ablation.py:96`** · needs a decision first  ·  FIXED 2026-09-04
The fine2D block is picked to match the MCS score it is later graded on

- Computes: select_fine2D picks the 100 features most correlated with MCS distance, then every subset with that block is scored by correlation to the same MCS values.
- Wrong because: Features chosen to match MCS values will score well on those same values, so fine2D's fidelity number is inflated by an unmeasured amount. A similar check elsewhere, splitting pairs in half, moved one such score from 0.449 to 0.410 on herg_ki.
- Touches: feature_ablation.json's fine2D row (+0.110 herg_ki, +0.206 logd) and the structure column of the subset table carry this, unchecked. The stacked-recipe headline numbers have a pair-split check but not a molecule-split one.
- Fix: Pick the fine2D features on one held-out half of molecules and score them on the other half, as similarity_noise_cleaning.py does at line 177.
- FIXED 2026-09-04: molecules are cut in half by whole scaffold group (_heldout.scaffold_halves); the fine-2D ranking and the per-column z-score mean/sd are fitted on one half and every reported number is measured on the other, over the cached pairs with both endpoints there. The old fit-equals-score numbers are kept as an `insample` arm.

**`tests/experiments/similarity_metrics/ranker_experiment.py:260`** · needs a decision first  ·  FIXED 2026-09-04
Activity feature selection sees the test-fold labels it's later scored against

- Computes: For each activity-target ranker, ranker_experiment.py scores feature pairs using |y_i - y_j| over all rows, then tests on those same rows.
- Wrong because: Every activity-target selector and r5_rrelieff pick features using labels from rows later used as the test fold, inflating their five-nearest-neighbour accuracy score. The structural selector table is untouched — it's scored on MCS distances, which carry no labels.
- Touches: ranker_experiment.json already holds the leaked numbers: herg 0.709→0.724 and logd 0.618→0.704, both activity-target selectors compared against baselines that did no selection.
- Fix: Move feature selection inside each fold: score and allocate features only on training-fold pairs, then score five-nearest-neighbour accuracy on the held-out fold; if too costly, report that accuracy only for the structural selectors.
- FIXED 2026-09-04: every ranker re-runs inside each of the five scaffold folds, scoring and allocating features from training-fold labels and from the cached pairs whose both endpoints are training-fold molecules; the held-out fold is predicted from its five nearest training neighbours and the Spearman is pooled. Applied to the structural selectors too, so activity and structural stay paired. The per-column z-score mean/sd follows the same rule. readacross_rho_insample carries the old number.

**`src/kirby/hybrid.py:4720`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Base features are picked using the same rows that judge whether extra features get kept.

> Fixed: performance_weighted and fixed compute importance from `rep_dict_train`/`train_labels`, the internal training part, matching weighted_global; find_best_hybrid's importance precompute moved onto the same rows so a handed-in score matches.

- Computes: create_hybrid splits its input rows into an internal training part and an internal validation part, then picks each representation's top features using importance computed from both parts combined.
- Wrong because: That validation part then decides whether adding extra features beats the base, so the base looks stronger there than it actually is. Extra features are scored from the training part only, so the comparison isn't fair.
- Touches: This biases the recorded augmentation_baseline, augmentation_final, f24's per-fold claimed delta, and its noise_taken/echo_taken decoy counts, toward keeping the base as-is. f24's outer per-fold scores and delta_mean_of_folds are not touched.
- Fix: For the performance_weighted and fixed allocation methods, compute importance from the training part only, matching how weighted_global already works; or subset the passed-in importance to that part.

**`tests/experiments/complexity/exp_b_balanced_global.py:142`** · re-run and wipe caches  
c30_balanced_global_selection.py picks pool5 and comparator model on folds it scores them on

- Computes: c30_balanced_global_selection.py ranks all ten representations on five-fold held-out scores, then takes the top five as pool5 and the best-scoring model as the comparator, at lines 107 and 121.
- Wrong because: Pool5 and the comparator model are chosen by ranking representations on the same five folds they are later scored against. So the pool5-vs-pool10 comparison is not fair, because pool5 was picked using the scores that grade it.
- Touches: No number on disk changes: exp_b never produced results, is blocked by a missing function, and is superseded. The same selection choice is live and unflagged in c30_balanced_global_selection.py, lines 107 and 121.
- Fix: In c30_balanced_global_selection.py, rank representations and pick pool5 and the comparator model from each fold's training rows only, not from the held-out scores.
- **Fixed 2026-09-04:** done in both files. exp_b picks its best representation, pool5, pool10 and comparator model per fold from a 3-fold inner group-aware CV on that fold's training rows; every arm in a fold uses that fold's model, so the pairing holds. c30 got the same treatment — see the c30:121 entry. exp_b also could not run — deleted harness import, `H._label_signal_ladder` missing, output directory gone — so it now uses `_alloc_harness`, its direction gate is `H.assert_metric_direction`, and it writes into `tests/results/complexity/exp_b/`.

**`tests/experiments/federated/fed02_rep_diversity_silos.py:413`** · re-run and wipe caches  
fed02's C5 selects its three representations using the test fold it is scored on

- Computes: C5 scores a hybrid built from the three representations with the best test-fold MCC or RMSE, on those same test rows.
- Wrong because: The three representations entering C5 are picked by their score on the same test rows C5 is then measured on, every fold. That makes C5 look better than the best single representation and better than using all five, for a reason that isn't real.
- Touches: No fed02 result on disk is affected yet. The only run is a smoke log with three representations, where the choice is empty. It will affect C5 once federated runs with five representations.
- Fix: Rank each silo representation inside the training rows only, using a group-aware inner split, then build C5 from the three that win there.
- FIXED 2026-09-04: `_rank_reps_train_only` scores the five silo representations on a 3-fold group-aware split of the training rows, and C5 takes the three that win there. The scores it chose on are written to the result as `inner_rank_scores`. Costs 5 representations x 3 inner fits per outer fold.

**`tests/experiments/complexity/c02_complexity_grid.py:423`** · re-run one experiment  
c02's hybrid-vs-best-rep number is picked on the same folds it's scored on, in every file on disk

- Computes: For each dataset, c02 picks the best hybrid score and the best single-representation score, both chosen on the same inner folds, then subtracts them.
- Wrong because: The hybrid score is the best of roughly 450 to 570 allocation-method, importance-method, evaluator and budget combinations per model. The single-representation score is the best of about 10 representations, so it is biased upward by an unmeasured amount.
- Touches: Every c02 result file on disk (26, from June, before this fix) carries the inflated delta, including the figure c02_hybrid_vs_baselines.png. None have the corrected number yet.
- Fix: Re-run c02 so it writes the fold-reselection block, then point make_supervisor_plots.py and supervisor_presentation.py at that block instead of the current fields.

**`tests/experiments/complexity/c12_modality_ablation.py:497`** · re-run one experiment  
c12's HELPS/HURTS verdict compares two best scores fit and scored on the same folds

- Computes: For each modality pair, delta_vs_best_single subtracts the best score for expression alone from the best score for the pair, both picked and scored on the same folds.
- Wrong because: The single representations entering a pair, and the single-modality winner it's compared against, are all picked by taking the best score on the same folds they are then scored on. Neither side is a held-out score, so the difference has no honest direction — not a known bias against the pair.
- Touches: The results JSON stores this contaminated delta for every pair. The separate "effect of adding each modality to expression" table repeats the same computation, so it is affected too.
- Fix: Use the leave-one-fold-out check already in the file: pick each modality's winning representation on the other folds, then score it only on the held-out fold.
- **Fixed 2026-09-04:** HELPS/HURTS now reads a leave-one-fold-out delta: the single representation each pair is compared against is re-picked on the other folds. Stored per pair as `reselected_delta`, `reselected_delta_by_model` and a (model, fold) win count. The same-fold `delta_vs_best_single` is kept beside it. The 'adding to expression' table got the same treatment.

**`tests/experiments/complexity/c15_ml_theory_depth.py:291`** · re-run one experiment  
c15b's Align column scores a hybrid against the same labels used to build it

- Computes: For each budget, c15b builds a hybrid representation using every row's label, then scores its kernel alignment against those same labels.
- Wrong because: The hybrid's features are picked to fit these same labels, so its alignment score reflects that fit, not a compact space's shape. The representations beside it in the table were never fit to the labels, so the comparison favors the hybrid unfairly.
- Touches: Only the Align column: kernel_target_alignment in results['kernel_analysis']['per_budget'] and in gamma_sweep['hybrid_b50']. Condition number, kernel CV, and effective rank are unaffected. Sub-analysis 15a is unaffected — its hybrid uses no labels.
- Fix: Split into train and test rows first, build the hybrid on train rows only, then compute alignment on the held-out test rows.
- **Fixed 2026-09-04:** 15b cuts a group-aware 80/20 split, builds the hybrid on the training rows and reads `kernel_target_alignment_heldout` on the rest — for the representations and the gamma sweep too, so the column is comparable. The whole-row number is kept as `kernel_target_alignment_all_rows`.

**`tests/experiments/complexity/c19_importance_weighted_training.py:143`** · re-run one experiment  
c19 picks the hybrid features and recipe using validation labels, reports it as held-out.

- Computes: For each dataset, c19 scores five importance-weighting settings (including unweighted and best_condition) on the hybrid representation, once per fold.
- Wrong because: get_fitted_pool_hybrid picks the winning allocation/importance/budget recipe and the hybrid's features once, using every row's label, including validation rows. The cross-validation loop at line 178 then only refits the scaler and importance weights inside that fixed feature set.
- Touches: The absolute scores in results['per_space']['hybrid']['scores'] are not held-out. The deltas between weighting conditions, which is what c19 asks, are unaffected — every condition shares the same feature set.
- Fix: Fetch the recipe with get_best_search_for_downstream, then call build_fold_hybrid inside the fold loop on the training rows' labels only, as c18 does at line 243.
- **Fixed 2026-09-04:** c19 now fetches only the recipe with `get_best_search_for_downstream` and calls `build_fold_hybrid` inside the fold loop on the training rows' labels, as c18 does. The fold's matrix is built once and shared across the five weighting conditions, so their deltas stay paired.

**`tests/experiments/complexity/c23_evaluator_model_matching.py:102`** · re-run one experiment  
c23's evaluator-match deltas score the winning configuration on the folds that picked it

- Computes: For each downstream model and internal evaluator, c23 reports score_mean: the mean inner-CV fold score of whichever hybrid configuration find_best_hybrid picked as best for that evaluator.
- Wrong because: The configuration is chosen by maximizing score_mean on those same folds, so score_mean is inflated before the matched-minus-mismatched delta is even computed. Condition E searches a superset of condition A's evaluators for rf, lightgbm and xgboost, so its delta can only be zero or negative.
- Touches: The A/B/C deltas in tests/results/c23_evaluator_model_matching/*.json are read by firstpass_notready.py and stay affected; they run about 0.02 against a fold spread of 0.13-0.22. The E-vs-A delta is written but never read.
- Fix: Score each model-evaluator pair with nested_cv_evaluate on held-out outer folds and report delta_mean_of_folds instead of score_mean.
- **Fixed 2026-09-04:** every (model, evaluator) pair now goes through `nested_cv_evaluate` at `config['n_splits']` outer folds, and each matched-minus-mismatched delta is paired fold by fold. The winner's own inner mean is kept as `search_mean` with `selection_bias`. Condition E rows carry `evaluator_set_contains_matched`. Costs five searches per pair instead of one.

**`tests/experiments/complexity/c32_allocation_strategy_cross.py:309`** · re-run one experiment  
c32's N=2/3/5 shortlists are picked using validation-fold scores, not train-only

- Computes: c32 ranks each representation by its mean score across all five held-out folds, then takes the top N of that ranking as the candidates for N=2, 3, and 5.
- Wrong because: Fold i's candidate representations were chosen partly from fold i's own held-out labels, so a win at N=2 or N=3 cannot be told apart from picking winners after seeing the answer. N=10 uses all ten representations, so it is not affected.
- Touches: 34 of the 54 result cells in /Volumes/seagate/kirby_cache/c32_allocation_cross/ are won by N below 10, so the N-axis comparison in C32_ANALYSIS_HANDOFF_2026_08_29.md is biased toward small N. Per-fold budget weights are unaffected — they come from a train-only inner CV.
- Fix: Rank representations inside each fold using fold_hib[i], already computed and train-only, and take the top N of that instead of the whole-data ranking.
- **Fixed 2026-09-04:** each fold takes its top N from `fold_train_score[i]`, the raw per-fold train-only scores behind `fold_hib[i]`, so the representations a strategy draws on were never picked using that fold's held-out labels. `subset_by_n_per_fold` records the subset each fold used. N=10 is every staged representation, so it is unchanged. Per-representation importances are now computed for all ten per fold, since any of them can enter any fold's subset.

**`tests/experiments/federated/fed01_noise_heterogeneity.py:143`** · re-run one experiment  
fed01's settings come from labels it later corrupts; four conditions never see that search.

- Computes: fed01 picks one allocation method, importance method, internal evaluator, and augmentation strategy per dataset, by searching all rows with their real, uncorrupted labels, before folds or noise are applied.
- Wrong because: Those four settings are chosen once, using the same labels fed01 later corrupts, and used in every fold for three conditions only (C0, C5, C6). The other four conditions (C1-C4) run fixed settings and never see that search.
- Touches: Any MCC gap fed01 reports between C0/C5/C6 and C1-C4: C0/C5/C6 had real labels for that search, C1-C4 never did. Unclear if a result file exists yet.
- Fix: Pass label_noise so the search runs on each fold's noisy training labels only, not the whole dataset's clean ones.
- FIXED 2026-09-04: one search per noise level, each passing a `NoiseSpec('uniform')` at that level's mean flip fraction (`sigma_frac / 2`, the mean of the per-silo linspace), so it trains on corrupted labels and scores on clean ones. Six searches per dataset instead of one; C0, C5 and C6 read that level's settings from `settings_by_sigma`. User decision, this session.

**`tests/experiments/federated/fed05_antibody_federated.py:199`** · re-run one experiment  
fed05 picks its hybrid recipe using rows that later become the held-out test set

- Computes: fed05 runs find_best_hybrid on all of a silo's representations and labels to pick allocation method, importance method and evaluator, before the 20% test split.
- Wrong because: The rows used to test the hybrid recipe are the same rows used to pick it. In C6 that recipe is tuned on test rows while the quality-weighted ensemble it's compared against gets no tuning at all.
- Touches: Changes the C6 claim that feature-level fusion beats the quality-weighted ensemble, and every C1 per-silo baseline. Leak size is bounded by measured allocation sensitivity, 0.009-0.018; evaluator and importance choices don't move scores.
- Fix: Run find_best_hybrid on the training rows only, after the split, or fix the allocation method, importance method and evaluator the same way for both sides before splitting.
- FIXED 2026-09-04: every silo's 80/20 split is drawn once, before anything is fitted, and `find_best_hybrid` runs on that silo's training rows only. `_silo_baseline` now takes the split rather than redrawing it, and all seven other readers index `silo_splits` instead of falling back to a fresh draw. Same number of searches, on 80% of the rows.

**`tests/experiments/federated/federated_helpers.py:1100`** · re-run one experiment  
silo_scores measures a feature set on the labels used to pick it

- Computes: silo_scores: an evaluator's score on 20% of each silo's rows, using features create_hybrid picked from all of that silo's rows, including those same 20%.
- Wrong because: Feature selection sees the validation rows' labels before those rows are scored, inflating silo_scores for every silo. Which silo gets inflated more is not established — in the one recorded run the noisy silo scored 0.0 and got the smallest budget.
- Touches: Only fed05's c4_noise_heterogeneity silo_scores/silo_budgets and fed01's Phase 4 importance-noise correlation, both on disk now. fed01 C5's held-out MCC is unaffected — it already trains and scores on separate rows.
- Fix: Carve each silo's validation rows out before create_hybrid, select features on the remaining rows only, then score on the held-out rows; pass groups through from fed05 C4 and from federated_noise_detection (which has no groups parameter yet) so the 80/20 split respects folds, not a random shuffle.
- FIXED 2026-09-04: the silo's 20% is carved out first and `create_hybrid` runs on the other 80%, so no feature is admitted on the labels it is then scored against. One create_hybrid call, not two. `federated_noise_detection` takes `groups` and `task_type` and forwards them; fed01 passes the training fold's groups, fed05 C4 passes sequence-identity groups concatenated across silos with an accumulating offset. A silo with no usable groups takes the flat score rather than a random 20%.

**`tests/experiments/opig_blueprint_transfer.py:215`** · re-run one experiment  
Blueprint transfer picks the recipe and best representation on the folds it scores them on

- Computes: For each target dataset, it picks the allocation method and single representation with the highest score on a 5-fold GroupKFold split, then scores them on that same split.
- Wrong because: The evaluation folds are identical to the folds the search already scored, because GroupKFold is deterministic on the same groups. The recipe and the single representation chosen are the ones already seen during scoring, not held out from it.
- Touches: Biases mean_delta_recipe, mean_delta_full, and the transfer verdict in opig_blueprint_transfer.py, because base_best carries the same look-ahead recipe and full are compared against. No full-mode result is on disk yet, only a 200-sample, n_folds=1 smoke run.
- Fix: Re-select the recipe and the best single representation inside each held-out fold, using training rows only, or fold_reselection's per-fold argmax that excludes the scored fold.
- **Fixed 2026-09-04:** same change — per-fold recipe and per-fold best single representation; each fold's picks are saved as per_fold_native_blueprint and per_fold_base_rep, and the whole-data picks are kept beside them as context only.

**`tests/experiments/opig_cdr_flexibility.py:297`** · re-run one experiment  
opig_cdr_flexibility reports a search maximum as the hybrid AUC, not a held-out score

- Computes: opig_cdr_flexibility.py reports kirby_hybrid_search_score_mean — the winning hybrid configuration's mean AUC across its own search folds — as the hybrid's score for CDR flexibility classification.
- Wrong because: That mean is a maximum picked from hundreds of configurations on those same folds, not a held-out score like the per-representation AUCs beside it. The two numbers share the same metric, so they read as directly comparable when they are not.
- Touches: Affects only whether hybrid augmentation helps CDR flexibility classification. The comparison to ITsFlexible's 0.82 AUC does not use this number and is unaffected. 0.6309 sits in tests/results/comparisons/opig_cdr_flexibility.json today as kirby_hybrid_search_score_mean.
- Fix: Replace it with a leave-one-fold-out re-score, as opig_graphinity_ddg.py does, or label it a search maximum, not a results-table score.
- **Fixed 2026-09-04:** replaced by a leave-one-fold-out re-score against the same baseline representations. The search maximum survives as hybrid_summary.search_best_score_mean, renamed so a reader asking for the old name gets nothing rather than a search maximum.

**`tests/experiments/opig_humanness.py:320`** · re-run one experiment  
kirby_hybrid's recipe and features both see held-out labels before being scored on them

- Computes: opig_humanness.py builds a kirby_hybrid representation, picks its features and its underlying blueprint using every row's label, then scores it on the same folds.
- Wrong because: The blueprint behind kirby_hybrid is chosen by a search ranked over every fold's labels, not held out from them. Its features then come from refitting that same blueprint on the whole dataset, so the folds test on labels already used twice.
- Touches: tests/results/comparisons/opig_humanness.json stores kirby_hybrid rows (rf AUROC 0.9229) built this way, with no honest comparison saved next to them; the honest numbers print but are not written to that file.
- Fix: Save the honest comparison the script already prints at lines 455-477 into that results file, and drop the kirby_hybrid rows built from leaked features.
- **Fixed 2026-09-04:** the kirby_hybrid rows are no longer built; the computed honest comparison is saved to the result file under reselected_comparison, beside the raw per-fold block it comes from.

**`tests/experiments/opig_kingsmith_component_splits.py:674`** · re-run one experiment  
The hybrid-vs-base delta pits 5 hybrid models against 70 base draws.

- Computes: print_analysis takes the largest R2 across every representation and model in a split, subtracts it from the largest hybrid R2, and prints that as the Delta column.
- Wrong because: Both numbers are maxima taken over the same folds they're measured on, so both run optimistic. The base side picks the best of 70 representation-model pairs, the hybrid side the best of only 5.
- Touches: No number from this script has been reported yet, so nothing on disk changes meaning. The fix only changes future runs of this comparison.
- Fix: For each model, subtract the base representation's held-out fold score from the hybrid's, using the same arithmetic fold_reselection already runs, then rerun the experiment keeping per-fold scores.
- **Fixed 2026-09-04:** the Delta is now per downstream model and per held-out group — the base representation is re-picked on each group from the other groups. evaluate_rep and the per-fold hybrid loop keep fold-index-aligned scores (r2_per_fold, rmse_per_fold) so the subtraction is paired.

**`tests/experiments/opig_scaffold_similarity.py:283`** · re-run one experiment  
mode_features picks its feature subset and scores it on the same pairs it picked from

- Computes: It computes greedy_score, a top-k score, an all-14-feature score, and a truncation-curve peak, all from one pair sample drawn once per dataset.
- Wrong because: greedy_forward keeps adding features only while the score on that pair sample improves, so its subset is tuned to those exact pairs. Top-k and all-14 stay fixed, so only greedy's score is tuned to the pairs it is reported on.
- Touches: print_analysis's greedy-vs-top-k comparison does not hold: greedy ties top-k on flab_binding_phillips_h3 (0.757 vs 0.757), both below all-14 (0.762). The delta_if_added column in SCAFFOLD_SIMILARITY_RESULTS.md is expected, not a defect.
- Fix: Split each dataset's pair sample into two halves by variant, choose the feature subset on one half, and report its score from the other.
- **Fixed 2026-09-04:** variants are split in two halves and only within-half pairs are kept; greedy, the top-k ordering and the truncation peak are all chosen on one half and reported from the other. --n-pairs default raised 8000 to 32000 so each half carries about 16000 pairs (measured on flab_expression_koenig: 15254 and 16548; 25 s to load).

**`tests/experiments/opig_trastuzumab_enrichment.py:361`** · re-run one experiment  
The printed hybrid AUC is the search's own best CV score, not a held-out score

- Computes: It picks, for each of 8 budgets, the allocation method, importance measure and evaluator with the highest CV score, then reports the best of those 8 as the hybrid's AUC.
- Wrong because: That score is chosen on the same folds it's measured on, so it's inflated by picking the best of many settings. Every single representation printed next to it is instead a mean AUROC over held-out folds.
- Touches: opig_trastuzumab_enrichment.json stores this inflated number as hybrid_search.rf_cv_auc, printed as "Hybrid (RF search CV AUC)" beside honest per-representation AUROC values in every run's output.
- Fix: Score the hybrid the same way as the representations: build_fold_reselection_input(search) then fold_reselection(...), the two calls opig_humanness.py already makes.
- **Fixed 2026-09-04:** build_fold_reselection_input then fold_reselection, printed and saved as reselected_comparison. The search maximum stays as hybrid_search.search_best_cv_auc, renamed from rf_cv_auc.

**`tests/experiments/similarity/s08_optimal_dimension.py:217`** · re-run one experiment  ·  FIXED 2026-09-04
s08's per-model budget picks aren't a real sweep, and never reach s09's optimal_dim_ratio

- FIXED 2026-09-04: the cache shortcut is gone. Every model is fitted on the hybrid
  rebuilt at that step's budget (`budget_override=budget`, already there), so no entry
  is copied in from the search's own winning configuration and the six budgets give six
  scores. kNN is also scored here now — s08's model list is DOWNSTREAM_MODELS plus
  'knn', local to this file — so `best_model_budgets` carries a 'knn' entry and s09's
  optimal_dim_ratio has a numerator. Needs the 16 datasets re-run; every stored
  best_rf_budget and best_model_budgets predates this.

- Computes: For each dataset, s08 picks one budget per downstream model — rf, xgboost, svm, mlp, lightgbm — meant to be the best-scoring budget across a fold sweep.
- Wrong because: At every budget except the one already picked by the search, s08 rebuilds using the budget the knn search picked, not that loop's budget. So five of six scores are identical, and the sixth is just the search's own winning score, copied in.
- Touches: best_rf_budget and best_model_budgets in the 16 stored s08 files, rf_budget_used in the Pareto block, and the best_rf_budget column in firstpass_notready.py are all this broken comparison. s09's optimal_dim_ratio never runs — knn isn't a downstream model.
- Fix: Score every model at its own loop budget using build_fold_hybrid with a budget_override, and drop the cache shortcut entirely so no entry gets copied in.

**`tests/experiments/similarity/s12_cross_domain_cliffs.py:273`** · re-run one experiment  ·  FIXED 2026-09-04
s12 says it re-picks the hybrid per fold; the recipe is picked once, on all rows

- FIXED 2026-09-04: the recipe is searched per model per fold on that fold's training
  rows (`get_fold_best_search`), so both arms now pay for held-out selection. Folds come
  from `nested_cv_outer_folds` and are shared by the single-representation and hybrid
  loops, which previously rebuilt the splitter separately. Each model's block records
  `fold_recipes`, and `hybrid_cliff_delta_reselected`'s stored 'selection' string says what
  the folds chose. Costs one search per model per fold.
  One side effect, worth knowing before comparing to an older s12 file: the folds now
  come from `nested_cv_outer_folds`, which is told the task type, where `cv_split` was
  called without one and fell back to the integer-label heuristic. On a classification
  dataset whose labels are not integer-typed the folds are now stratified where they were
  not, so the fold composition moves. That is the refuted `s12:310` entry's mechanism,
  settled here by using the authoritative path rather than the fallback.

- Computes: s12's hybrid_cliff_delta_reselected compares held-out cliff-compound error between the hybrid and single representations, both labeled as re-picked per left-out fold.
- Wrong because: The hybrid's features are refit inside each fold, but its recipe — allocation method, budget, importance, evaluator, augmentation — is chosen once, using every row. Single representations are re-picked fold by fold, so only that side pays for held-out selection, not the hybrid.
- Touches: No stored result is affected — hybrid_cliff_delta_reselected isn't in any file yet; the 16 saved s12 files predate it and hold only hybrid_cliff_delta.
- Fix: Either pick the hybrid's recipe inside each fold using only its training rows, or give held-out re-selection several hybrid recipes per fold to choose among.

**`tests/experiments/similarity_metrics/aggregate.py:108`** · re-run one experiment  ·  FIXED 2026-09-04
A6_dca's metric order comes from labels it's scored on, though nothing is ever dropped by it

- Computes: aggregate.py orders A6_dca's six metrics by how well each correlates with the activity gap between molecule pairs, then averages the kept ones.
- Wrong because: The keep order comes from the same activity gaps A6_dca is later scored on. That never matters here: metric correlations top out at 0.724 and 0.717, well under the 0.9 threshold, so all six are always kept.
- Touches: No number in aggregate_experiment.json or the roadmap table changes. A6_dca already equals A1_score to four decimal places on herg_ki and logd, the two datasets run so far.
- Fix: Order the six metrics by their correlation with MCS distance, already stored in the same file, instead of with the activity gap.
- FIXED 2026-09-04: A6_dca orders the six metrics by |Spearman(pair distances, MCS distance)|, and the row records the keep order, anything dropped, and the highest correlation between two metrics. Confirmed by running herg_ki: all six kept, A6_dca still equals A1_score to four decimals.

**`tests/experiments/similarity_metrics/feature_ablation.py:186`** · re-run one experiment  ·  FIXED 2026-09-04
feature_ablation.py reports a search maximum, not a held-out structural-fidelity score

- Computes: desc_drop_one_greedy drops RDKit descriptors one at a time, keeping each drop that most raises the correlation between distance and MCS similarity over the same 8000 pairs it later reports.
- Wrong because: The reported correlation is the best of 143 drops, chosen on the same pairs it is scored on. No molecule or pair is held out, so the number is a search maximum, not a measured fidelity.
- Touches: Only the two standalone scripts: feature_ablation.py and its duplicate in creative_hybrids_v2.py. No find_best_hybrid result is touched — src/kirby/similarity/selection.py already holds out a pair half when it drops descriptors. Direction still holds: similarity_noise_cleaning.json shows dropping descriptors still helps on held-out pairs, just by 0.04–0.07 less than feature_ablation.json's 0.250→0.495 (herg_ki) and 0.400→0.690 (logd) claim.
- Fix: Split molecules by scaffold group into a fit half and a score half, run the drop sequence on the fit half's pairs, and report correlation only on the other half, as similarity_noise_cleaning.py already does.
- FIXED 2026-09-04: the drop sequence is chosen on the fit half's pairs and re-scored after every drop on the score half's pairs. `final_struct` is that held-out number; `final_struct_fit`, the maximum the sequence was steered by, sits beside it.

**`tests/experiments/similarity_metrics/ranker_experiment.py:207`** · re-run one experiment  ·  FIXED 2026-09-04
combo_r1+r2, combo_r2|r1, and r9_greedy structural are scored on the same pairs they select on

- Computes: For structural target, three selector rows (combo_r1+r2, combo_r2|r1, r9_greedy) pick features by correlation to MCS overlap, then struct_fidelity scores that same correlation on the same pairs.
- Wrong because: Baseline rows are never fit to MCS, so their struct_fidelity score is fair while these three rows score themselves on data they were picked with. The roadmap's structural-vs-activity comparison uses these same three rows, so part of that reported gain may be this same effect.
- Touches: Affects ranker_experiment.json's struct_fidelity for combo_r1+r2, combo_r2|r1, and r9_greedy structural. The roadmap's +0.05 to +0.12 structural-over-activity gain is the same size as the in-sample inflation measured elsewhere, so it cannot be claimed yet.
- Fix: Split the pairs once, rank features on one half and score struct_fidelity on the other, for selector rows and baselines alike.
- FIXED 2026-09-04: molecules are cut in half by whole scaffold group; every ranker ranks features on one half and struct_fidelity is taken on the pairs inside the other, baselines measured on that same half so the rows stay comparable. struct_fidelity_insample carries the old number.

**`tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:73`** · re-run one experiment  ·  FIXED 2026-09-04
Winning representation's structural fidelity score is fit to the same pairs it is tested on

- Computes: winning_recipe_vs_standard.py scores each representation's struct_fidelity — a Spearman correlation between its distances and MCS similarity — on the same molecule pairs used to pick its descriptor columns.
- Wrong because: Its descriptor columns and 100 fingerprint columns are chosen by fitting to those same pairs, then fidelity is scored on them again. The three comparator representations select no columns, so only the winning score is inflated this way.
- Touches: struct=0.4372 in winning_recipe_vs_standard.json is real, on disk, in-sample. The 0.4490/0.4102 comparison isn't a result file — it's a roadmap note for a related, differently-weighted representation. How much of 0.4372 is inflation isn't established.
- Fix: Score struct_fidelity on pairs held out from the ones used to pick descriptor columns and fine-2D columns, for the winning representation and comparators alike.
- FIXED 2026-09-04: the descriptor blacklist, the 100 fine-2D columns and the z-score mean/sd are fitted on one scaffold half, and EVERY row — the three comparators included, so the deltas stay like-for-like — is scored on the other. struct_fidelity_insample carries the old in-sample number.

**`tests/experiments/complexity/c05_learning_curves.py:110`** · edit only  
c05/c07 call a blueprint "no leakage" though it's tuned on the same rows it's scored on

- Computes: c05 and c07 pick one blueprint per model by maximising CV score over the whole dataset, then score it on subsample or partition folds from those same rows.
- Wrong because: The comment says this blueprint is re-fit on the subsample only, with no leakage. Only feature selection is re-fit; the allocation method, importance method, evaluator, and budget were chosen using every row the subsample later holds out.
- Touches: No computed result changes meaning today: nothing in the repo reads fulldata_bp or fulldata_blueprints_meta anywhere (checked every .py and .md file). It only matters if c07's global-tuning comparison gets used later.
- Fix: Rewrite the c05:110-111 comment and rename fulldata_bp so it reads as a blueprint chosen with every label, not as a leak-free re-fit.
- **Fixed 2026-09-04:** the comment now says the recipe was chosen with every label, and `fulldata_bp` is `all_rows_bp` in c05 and c07. Nothing outside those two files read either name.

**`tests/experiments/complexity/c09_formal_analysis.py:607`** · edit only  
c09's fallback picks its blueprint from all rows, then scores it on folds of the same rows

- Computes: c09 reports train_r2, val_score and gap per model and budget, using a blueprint picked once by ranking score_mean over every row in the dataset.
- Wrong because: Allocation method, importance method, internal evaluator and augmentation strategy are all chosen by score_mean computed over every row, including rows later scored as validation. So val_score is optimistic and gap is understated, by the spread across those four choices, at every budget alike.
- Touches: The nine json files in tests/results/complexity/c09_formal_analysis/ came from c09's other path, reading c06's train_r2 — none are affected. Every run from now on hits this fallback, since c06 no longer writes train_r2.
- Fix: Write a flag in the results JSON saying this fallback's blueprint was picked using all rows, so val_score is never read as held out.
- **Fixed 2026-09-04:** the fallback now writes `recipe_selected_on: 'all rows'` and `val_score_is_held_out: False`, on the result and on every per-model per-budget cell, and prints the same at the top. The c06 path writes the opposite pair.

**`tests/experiments/complexity/c14_cross_domain_summary.py:279`** · edit only  
c14's hybrid score comes from the same folds that picked the winning recipe.

- Computes: c14 takes a hybrid recipe's mean score over its own five folds, maximized across roughly 600 configurations and five models, as the reported hybrid score.
- Wrong because: The folds that pick the winning recipe are the same folds that score it, so the number is optimistic. The single-representation score is also a maximum, but over about 50 combinations, not several hundred, so the two sides are not comparable.
- Touches: No result file, SLURM script, or citation uses c14 — it has never been run. Nothing on disk is wrong yet; this must be fixed before its first run.
- Fix: Replace the recipe lookup with nested_cv_evaluate, so each fold's winning recipe is chosen only from that fold's training rows, and report delta_mean_of_folds.
- **Fixed 2026-09-04:** the recipe lookup is replaced by `nested_cv_evaluate`; c14 reports `delta_mean_of_folds` and writes the held-out single-representation score, the per-fold deltas, and the search-side mean with its bias beside it.

**`tests/experiments/complexity/c22_explainability.py:104`** · edit only  
c22's search_score is a selection score over all rows, not held-out performance

- Computes: c22 scores one allocation method, importance method, evaluator, and budget on each fold's held-out rows, and saves those five scores plus search_score to JSON.
- Wrong because: The allocation method, importance method, evaluator, and budget it uses were picked by comparing scores computed on every row, including each fold's validation rows. search_score is that whole-dataset selection score, not a held-out performance number.
- Touches: search_score in c22's saved JSON is a selection score, not performance — only tests/analysis/firstpass_notready.py reads it, alongside shap_by_rep. The same recipe-selection applies to c10, c11, c14, and the opig_* experiments.
- Fix: Label search_score as a selection score — rename it or set recipe_source='whole-data search' in get_best_search_for_downstream — so no report treats it as performance.
- **Fixed 2026-09-04:** `search_score` is now `recipe_selection_score`, with `recipe_selected_on: 'all rows'` beside it, and the print says it is not held out. `tests/analysis/firstpass_notready.py` reads the new name (falling back to the old one for existing files) and its column header says selection, not performance.

**`tests/experiments/complexity/c31_perf_weighted_global_selection.py:428`** · edit only  
c31 and c32 pick their baseline representation using the same folds it is later scored on

- Computes: For each of ten representations, c31 and c32 average scores over five outer folds, then use the highest-scoring one as the baseline for every other allocation method in that cell.
- Wrong because: That baseline is chosen from the same five folds it is later scored against, so its score is inflated. Because the baseline is the inflated side, every reported delta understates the true gap, not overstates it.
- Touches: Ranking allocation methods by r2_delta_mean within a cell is unaffected — the same baseline is subtracted from all of them. Only win counts and CI flags are affected. c32 lacks this caveat; c31 already has it.
- Fix: Add this caveat to c32, matching c31; optionally add a per-fold baseline computed from the already-saved fold_hib scores, reported as a separate line.
- **Fixed 2026-09-04:** c32 no longer needs the caveat — its target is picked per fold, see the c32:204 entry. c31 keeps its whole-data target and gains a second one beside it: `target_trainpick` picks the representation per fold from `fold_train_score[i]` and reads its held-out score for that fold, and every arm carries `r2_delta_mean_vs_trainpick` and `native_delta_mean_vs_trainpick`. Costs no fits. Ranking allocation methods within a cell is identical under both baselines.

**`tests/experiments/complexity/c31_perf_weighted_global_selection.py:711`** · edit only  
best_scheme_by_cell's delta is the best of three weighting methods, reported as one

- Computes: c31_perf_weighted_global_selection.py:711 picks, per cell, the weighting method with the highest mean delta across the same five folds it then reports.
- Wrong because: The five folds that pick the winning method are the same five folds that score it, so the reported delta runs high. Its confidence interval, built at line 335 for one fixed method, does not cover choosing the best of three, so it reads narrower than it should.
- Touches: No reported result depends on this field: nothing else in the repo reads best_scheme_by_cell. Every result file already carries the honest per-method deltas, fold-win counts, and confidence intervals for all three methods at lines 697-698.
- Fix: Label best_scheme_by_cell's delta as a maximum across three methods, or replace it with a win count across the three methods and five folds.
- **Fixed 2026-09-04:** `r2_delta_mean` in `best_scheme_by_cell` is renamed `r2_delta_max_over_schemes`, and `r2_delta_ci95` is removed from that record — it was built for one fixed method and does not cover the three-way choice. Added: `scheme_fold_wins_vs_target` out of `scheme_fold_pairs`, and `r2_delta_mean_by_scheme` for all three methods. The per-method deltas and CIs in `weighted_global` are untouched.

**`tests/experiments/federated/fed04_cluster_aware_importance.py:185`** · edit only  
fed04's in-memory search cache returns a regression search when asked for classification

- Computes: get_precomputed_search returns whichever hybrid search is stored under an aug_condition key, regardless of the labels, task type, or metric requested.
- Wrong because: fed04 asks for a classification search after binarising caco2 and logd's labels. This path returns a regression search built on the continuous labels instead, with its allocation, importance and evaluator settings unchanged.
- Touches: No result file is affected. herg and lhasa_herg — fed04's only completed datasets — skip this cache and use the correct on-disk one. caco2 and logd crash first, on a separate bug, before any score is recorded.
- Fix: Pass data=None from fed04 so it falls through to the on-disk cache, or key the in-memory store by labels_hash and task_type.
- FIXED 2026-09-04: fed04 passes `data=None`, so the request goes to the on-disk cache whose identity carries labels_hash. The metric is also now 'auc' rather than the registered 'rmse' — a classifier cannot be ranked by RMSE, and caco2 and logd are registered as regression.

**`tests/experiments/opig_activity_cliff_qsar.py:375`** · edit only  
kirby_hybrid features are selected with each fold's own labels, then tested on those folds

- Computes: Builds kirby_hybrid's features from a refit that used every compound's label, then scores it on the same fold split as every other representation.
- Wrong because: Every held-out fold's labels helped pick which features kirby_hybrid uses, so no fold is scored on features chosen without seeing its own answers. The MAE, RMSE, R2, and AC-MCC printed for kirby_hybrid are measured on data that helped build it.
- Touches: The activity-cliff claim for kirby_hybrid in this file's output isn't supported. Only the kirby_hybrid row and the Hybrid MAE/Delta columns are wrong — an honest version is already computed and printed in the file.
- Fix: Drop the kirby_hybrid row from the per-representation table and the Hybrid MAE and Delta columns from the cross-target summary; report only the honest version already built.
- **Fixed 2026-09-04:** the kirby_hybrid representation is no longer built or evaluated; the Hybrid MAE and Delta columns are gone from the cross-target summary, leaving the fold_reselection table as the only hybrid number.

**`tests/experiments/opig_cgr_comparison.py:379`** · edit only  
The CGR-helps-hybrid verdict is biased toward saying yes, not measuring it

- Computes: For each representation set (SMILES-only, then SMILES+CGR), it fits hybrid feature selection using all labels, scores that hybrid with GroupKFold over every reaction, and subtracts the two RMSEs.
- Wrong because: Both RMSEs already use feature indices chosen with all labels, so both are optimistic. SMILES+CGR has more representations to pick from, so it gets more of that same freedom, and that biases the delta toward showing CGR helped.
- Touches: The printed verdict "Hybrid pool improves/does not improve with CGR" is unreliable. Held-out scores already stored per representation set in the output JSON are unaffected and can give an honest delta.
- Fix: Compute the CGR delta from the held-out scores already stored per representation set, not from the full-data hybrid scoring.
- **Fixed 2026-09-04:** the CGR delta is now the difference of the two pools' leave-one-fold-out hybrid scores, not of their full-data hybrid_eval RMSEs. Both pool scores are printed beside it.

**`tests/experiments/similarity/s05_knn_prediction.py:397`** · edit only  ·  FIXED 2026-09-04
s05's "pre-registered" baseline in fixed_ref is picked from the same folds it's scored on

- FIXED 2026-09-04: `firstpass_notready.py`'s `read_s05` reads
  `fold_reselection_result.knn.delta_mean_of_folds` — the leave-one-fold-out delta, both
  arms re-picked on the other folds — and prints why fixed_ref is not used. fixed_ref
  itself is kept and still written, with its stored note and its printed line corrected
  to say the baseline is picked in-sample; the module docstring no longer calls it the
  honest number.

- Computes: For each dataset, fixed_ref subtracts the best representation's score from one pinned hybrid score; that representation score is already the best of k=5 or k=10 on the reported folds.
- Wrong because: The hybrid score is fixed before folds are scored. The baseline score is chosen afterward, from the best representation and best k on the same folds, so it is inflated.
- Touches: The "+4 ~2 -9 (15 datasets)" tally in PRELIM_RESULTS_ANALYSIS.md:361 and s09's knn_delta_fixed_ref series both use this delta; any loss or near-zero count from it is unsound. Positive counts still hold as a lower bound.
- Fix: Have firstpass_notready.py's read_s05 use the leak-free knn_delta_heldout number s05 already computes and s09 already stores, instead of knn_delta_fixed_ref.

**`tests/experiments/similarity/s11_bayesian_optimization.py:883`** · edit only  ·  FIXED 2026-09-04 (decided: seed rows)
s11 picks the hybrid representation's settings using pool labels the search should reveal only later

- FIXED 2026-09-04, owner chose the seed-row search over the fixed defaults: the five
  settings are searched inside the seed loop on that seed's 5% seed rows, once per seed
  and shared across the hybrid conditions. The cache namespace names the seed
  (`bo_seed<i>_s<seed>`), so a seed's search can never be served from a cross-validation
  fold's entry or from another seed's. A seed whose search returns nothing falls back to
  build_hybrid_features's own fixed defaults, as before. `hybrid_seed_recipes` records
  what each seed picked.

- Computes: s11 picks the hybrid representation's five settings — allocation method, importance method, internal evaluator, augmentation strategy, step size — from a search over all rows, labels included.
- Wrong because: Those rows include the pool rows the search loop is meant to reveal one at a time. So the hybrid representation enters the comparison already knowing labels the other representations never see.
- Touches: This affects the "Hybrid vs ECFP4: AUC gain" number and stored bo_auc for molecular datasets. No molecular s11 result exists yet, so nothing on disk carries this now.
- Fix: Choose the hybrid representation's five settings from seed rows only, inside the loop that runs each seed, or use the fixed defaults already coded at lines 682-689.

---

## B. An internal holdout is drawn at random on a dataset that has groups

**15 open.**

**`src/kirby/hybrid.py:2826`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Candidate-feature ranker splits training rows by random shuffle, not by scaffold group

> Fixed: `_pool_and_rank_aug_features` takes `groups_train` and uses GroupKFold (StratifiedGroupKFold for classification) for the out-of-fold residuals; create_hybrid passes the internal training part's groups through augment_capped and augment_uncapped. Falls back to the shuffled split when there are fewer distinct groups than folds.

- Computes: Ranks candidate features by their correlation with out-of-fold residuals from a forest fit on base features, using a random shuffled split over each fold's training rows.
- Wrong because: The shuffled split lets a molecule's scaffold-mates land in its own training rows, so its residuals shrink toward the fit rather than measuring real error. That is not leakage — no held-out row reaches this step — but it weakens the residual signal the ranking depends on.
- Touches: Affects f24's comparison of residual_correlation against rf_importance for admitting augmented features. No result on disk is wrong — the admission score never sees this split — but the two rules differ less than labeled.
- Fix: Add a groups argument to the ranking function, pass in the scaffold groups for that fold's training rows, and use GroupKFold instead of the shuffled KFold when groups are given.

**`tests/experiments/noise/n13_noise_distributions.py:185`** · needs a decision first  
n13's hybrid arm picks features with a random holdout, not the scaffold split

- Computes: n13's hybrid space builder picks which features are admitted and splits the budget across representations using an internal 20% holdout.
- Wrong because: That holdout is a random 20% draw of training rows, not grouped by scaffold like every other split in the fold. The same loop builds the honest arm's holdout from scaffold groups instead, 400 lines later.
- Touches: No label leaks into training, so n13's held-out hybrid scores and the hybrid_advantage_persists claim stand as reported. Unmeasured: how much feature admission and the budget split shift under a scaffold holdout instead of a random one; no result file reports that number.
- Fix: Pass the fold's scaffold groups into that internal holdout, the same groups the honest arm already passes 400 lines later. A test file for measuring the size of this shift exists but has no recorded run. The same random holdout appears in nearly every other caller of this feature-admission step, so decide whether to fix n13 alone or that convention everywhere.

**`src/kirby/find_best_hybrid.py:2244`** · re-run and wipe caches  
mlp's early-stopping split is random, not by scaffold group; cnn/dnn lose grouping in f29 too

- Computes: mlp — the default model in every models=None search — stops training early using a validation holdout carved by random shuffle of rows, not by scaffold group.
- Wrong because: Its stopping point is chosen on rows that share scaffolds with training rows, so training runs longer than a group-honest split would allow. The holdout sits inside the training fold, so no reported test score changes.
- Touches: No test score is wrong — the holdout stays inside the training fold everywhere. mlp (f26/f27/f28/opig) and dnn/cnn (f29 only) stop training later than a group-honest split would, deflating scores evenly across comparisons.
- Fix: Route mlp's validation split, and dnn/cnn inside f29, through the group-aware split function already written in _es_split.py but never imported.

**`tests/experiments/_shared.py:129`** · re-run and wipe caches  
create_hybrid_cached skips fold groups; c09's hybrid budget split differs from the search's

- Computes: create_hybrid_cached computes each representation's share of the hybrid budget from a random 20% split of the training rows, not the scaffold-grouped fold split.
- Wrong because: This differs from the fold split find_best_hybrid used when it selected the hybrid, because c09 never passes groups. The hybrid c09 profiles is not the hybrid the search built, even though no held-out data is touched.
- Touches: c09's 9b/9c numbers (information retention, effective rank) come from training rows only, so none on disk are optimistic - the hybrid profiled just has a different budget split than the one the search picked.
- Fix: Add a groups parameter to create_hybrid_cached, pass groups[train_idx] from both places c09 calls it, and add groups to the cache key so old entries are not reused.

**`tests/experiments/complexity/c08_calibration.py:220`** · re-run and wipe caches  
c08, c09, c10 skip groups, so feature selection is scored on a scaffold-sharing holdout

- Computes: Five calls in c08, c09, and c10 build the hybrid without passing groups, so its internal validation split comes from a fixed random split, not the first GroupKFold fold.
- Wrong because: That random split can put rows sharing a scaffold in both halves. Allocation and augmentation admission then get scored against a holdout that overlaps its own training rows, so the criterion picking features is optimistic.
- Touches: The outer fold scores already reported for c08, c09, c10 stay honest. No result file on disk needs correction; fixing this only changes which features get picked in future runs.
- Fix: Pass groups=groups[train_idx] to all five calls, and add the group vector to the cache key so old cached features aren't reused.

**`tests/experiments/complexity/c24_blueprint_transfer.py:219`** · re-run and wipe caches  
c24 splits budgets using a random holdout instead of scaffold groups, on 3 saved files

- Computes: The budget split across representations for each cell of the transfer matrix, scored on a held-out slice of each fold's training rows.
- Wrong because: Without groups=, create_hybrid's holdout becomes a random 20% split with a fixed seed. That holdout shares scaffolds with its own training rows, which breaks the no-random-split rule.
- Touches: Three c24 result files on disk used this holdout for every cell. It doesn't shift the own-minus-cross delta, but changes each cell's score and the features nogueira_phi is computed from.
- Fix: Pass groups=groups[train_idx] on both create_hybrid calls, at lines 219 and 302, in c24_blueprint_transfer.py.

**`tests/experiments/federated/fed04_cluster_aware_importance.py:89`** · re-run and wipe caches  
create_hybrid calls across fed01-fed05 admit features on a random split, not a grouped one

- Computes: It computes which representation's features get admitted, using a random 20% split of training rows instead of a split grouped by scaffold, sequence identity, or PDB complex.
- Wrong because: Near-duplicate molecules land on both sides of that split, inflating the admission score. No file in the suite forwards groups into create_hybrid, including calls routed through federated_feature_selection.
- Touches: Any run of fed01 through fed05 has this: features win admission on a random split, not a group split. Test-fold scores don't leak, but admission is optimistic in every one of those experiments.
- Fix: Pass groups (sliced to match each split) into every create_hybrid call in fed01 through fed05, and forward groups through federated_feature_selection to the create_hybrid it calls internally.

**`tests/experiments/federated/federated_helpers.py:1103`** · re-run and wipe caches  
create_hybrid picks features from a random split, not your scaffold groups

- Computes: create_hybrid scores each representation on a held-out 20% slice of the training fold, to decide which features enter a hybrid representation.
- Wrong because: None of the ten create_hybrid calls in fed01, fed02, fed03, or federated_helpers pass scaffold groups, so that 20% slice is drawn at random. Molecules sharing a scaffold can land on both sides of it, so features get admitted using scores measured against their own near-duplicates.
- Touches: No number on disk is wrong from label leakage — the outer train/test split stays scaffold-grouped and correct, so reported federated metrics keep their meaning. What is affected is which features get chosen for the KIRBy representations built inside fed01, fed02, and fed03.
- Fix: Pass the training fold's scaffold groups into every create_hybrid call, give federated_feature_selection those groups the way fed01 already does, and replace fed03 Part A's shuffled KFold with a group-aware splitter.

**`tests/experiments/noise/n11_estimation_remediation_grid.py:199`** · re-run and wipe caches  
n11 and n12 skip scaffold groups when create_hybrid sets each representation's feature budget

- Computes: create_hybrid computes each representation's feature budget from a random 20% split of the training rows, not from the scaffold-grouped folds n11 and n12 use elsewhere.
- Wrong because: That split is not scaffold-grouped, so it can put molecules from the same scaffold on both sides. The budget it hands each representation is optimistically high, and that changes which features enter the noise-detection space.
- Touches: No result on disk changes: the split only draws from training rows already inside a fold, so no outer-fold row leaks and no reported number is wrong today.
- Fix: Pass groups[train_idx] into _build_estimation_spaces (n11) and _build_fold_shared (n12), then into create_hybrid(groups=...), the way n09 already does at line 330.

**`src/kirby/utils/lnl_methods.py:119`** · re-run one experiment  
DNN early-stopping holdout is a random split of the training fold, not grouped by scaffold.

- Computes: Three DNN trainers pick the stopping epoch using a random 20% holdout cut from the training fold's own rows.
- Wrong because: That holdout can share scaffolds with the rows just fitted, so the epoch it picks is optimistic. It only affects the training-side epoch choice; each fold's own validation labels stay untouched.
- Touches: Affects n09_lnl_survey's three baseline conditions and four noisy-label retrain methods, plus n13_noise_distributions' direct call to the same trainer. Whether saved result files already carry this is not stated.
- Fix: Give all three trainers, including _train_clf_dnn_return_proba, a groups argument; split with GroupShuffleSplit as dnn.py already does when groups are passed, threaded through _default_predict and n13's direct call.

**`src/kirby/utils/noise_estimation.py:451`** · re-run one experiment  
cross_val and hybrid split folds without groups; eight other n11 estimators use GroupKFold

- Computes: n11 compares ten noise estimators by how much of the clean-to-noisy gap each one recovers, using their out-of-fold residuals.
- Wrong because: cross_val and hybrid split folds with a plain 5-fold split, not by scaffold group. Their out-of-fold residual for a molecule can come from a model trained on that molecule's own scaffold neighbours, so it looks smaller than it should.
- Touches: n11's ranking of cross_val and hybrid against the other eight, for any run after commit 7a98aa1 (2026-08-19). The only n11 result on disk, buchwald_hartwig.json from 29 May, predates that commit and is not affected.
- Fix: Give CrossValResidualEstimator and HybridNoiseEstimator a groups argument, and route their folds through _oof_apply like the other eight estimators.

**`tests/experiments/complexity/c07_data_complexity_partitions.py:338`** · re-run one experiment  
c07 and c03 pick hybrid features on a different holdout than the blueprint used

- Computes: c07 refits each subpopulation's blueprint and scores it as fulldata_bp_scores; c03 refits SVR/ElasticNet under the chosen allocation for each dataset.
- Wrong because: c07 and c03 pick features for the hybrid using a random 20% holdout instead of the fold split used elsewhere. That random split falls inside the outer fold's training rows, so scores stay honest, but the features picked don't match the blueprint's or c05's rebuild.
- Touches: Affects c07's fulldata_bp_scores and c03's SVR/ElasticNet scores. Only allocation methods that consult this holdout are affected — fixed ignores it. Check fulldata_blueprints_meta per model before re-running.
- Fix: Pass groups (and model, metric_fn) into c07's build_fold_hybrid call and into c03's create_hybrid call, matching c05's rebuild.
- **Fixed 2026-09-04:** c07's `build_fold_hybrid` call now passes `groups=sub_groups[tr_i]`, the fold's own model and the dataset's metric with its direction, matching c05's rebuild. (c03 is not in this package.)

**`tests/experiments/complexity/c08_calibration.py:360`** · re-run one experiment  
c08's calibration rows share scaffolds with training rows, because they're picked at random.

- Computes: For each fold, c08 computes PICP and cal_error from residuals of a calibration set cut out of the training rows.
- Wrong because: The calibration rows are drawn at random, not by scaffold group, so they overlap the scaffolds the model trained on. That shifts PICP below 0.90 on 9 of 11 datasets, but above it on two flab_polyreact_jain datasets instead.
- Touches: OPIG_RESULTS.md's per-dataset coverage numbers come from this, since opig_slide_figures.py and firstpass_notready.py only read the 'optimal' condition's PICP. No result file on disk yet distinguishes over- from under-coverage.
- Fix: Cut the calibration set with make_cv_splitter over groups[train_idx] instead of a random permutation, at both the regression and classification code paths.

**`tests/experiments/similarity/s11_bayesian_optimization.py:696`** · re-run one experiment  
s11 picks hybrid features on a random split, not the grouped split it claims to use

- Computes: For each hybrid condition, s11 splits the seed rows 80/20 at random to decide which features each representation contributes, instead of splitting by scaffold group.
- Wrong because: The 20% holdout that decides which features each representation contributes can share scaffolds with its own training rows. best_single, what the hybrid is compared against, splits the same rows by scaffold group instead.
- Touches: Affects every hybrid condition's reported curve in s11 (hybrid_50d, multi_obj_hybrid, hybrid_ngboost/bnn/gauche_gp), not the pool AUC. No result on disk changes — every s11 result on disk so far is an antibody skip.
- Fix: Pass groups=groups[seed_idx] into every create_hybrid call in build_hybrid_features and _run_bo_with_rebuild, the way build_fold_hybrid already does.

**`tests/experiments/_shared.py:4850`** · edit only  
Tuning xgboost's hyperparameters uses a non-grouped 3-fold split, but no run has used it yet

- Computes: It scores each Optuna trial with a plain 3-fold split over an outer fold's training rows, with no group awareness in that split.
- Wrong because: Rows from the same scaffold group, or near-identical sequences in a single-antibody library, can land on both sides of that 3-fold split. That lets the tuner pick hyperparameters that memorise a group rather than generalise past it.
- Touches: It only runs if you call run_baseline_survey.py with --tune (off by default). No script does, and no result file has an xgb_tuned row — nothing on disk is affected yet.
- Fix: Add a groups argument, slice it to match the training rows, and score each trial with GroupKFold instead of plain KFold — matching how cv_split already does it.

---

## C. The budget split comes from representation scores measured on every row

**7 open.**

**`src/kirby/find_best_hybrid.py:5027`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
The cached budget-split score doesn't match what create_hybrid computes on its own.

> Fixed both halves: the precompute carves the fold with `hybrid.internal_holdout_split`, the function create_hybrid itself calls, and the budget split is scored by `internal_evaluator` on both paths (`hybrid._score_rep_for_budget`). Before, a search scored it with the evaluator through the cache and a direct create_hybrid call scored it with the downstream model. The evaluator axis is untouched: it is now the device everywhere instead of only inside a search.

- Computes: For each fold, it scores every representation once, on a train/validation split of that fold's training rows, and reuses that score to set the budget split.
- Wrong because: The held-out rows and the model used to score each representation are both different from what create_hybrid uses when you call it directly. So a hybrid built outside a search gets a different budget split than the one the search reported.
- Touches: No result file is wrong. Every budget/importance-method/model choice inside one search shares the same split, so comparisons within a search hold. A hybrid rebuilt later by create_hybrid gets a different split.
- Fix: Score representations on the same split create_hybrid uses per fold, and either add the model to the cache key or skip the cache when a model is given.

**`src/kirby/find_best_hybrid.py:7454`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Outer-fold rebuild splits the budget by a different score than the search used

> Decided 2026-09-04: a live score, recomputed on each fold's own training rows. apply_blueprint stops injecting `blueprint.rep_scores` when labels are given, so the outer rebuild scores its representations the way the inner search did. Costs one internal-evaluator fit per representation per rebuild.

- Computes: It computes each representation's best score across models, averaged over every inner fold's training rows, then splits the outer-fold budget by that average.
- Wrong because: The inner search split the budget using scores computed on each inner fold's own training rows. The rebuild instead uses one mean score averaged across the whole inner search, and under top_3 can pick a different set of representations.
- Touches: Changes delta_mean_of_folds in f26 job 12971694 and f28 job 12973831, only for performance_weighted/weighted_global arms under a single objective. Blend cells and f26's fixed/equal arms are unaffected; worst under predictive.
- Fix: Decide which per-representation score should split the outer-fold budget: the mean inner-CV score now used, or a live score recomputed on each outer-train fold like the search did.

**`tests/experiments/_shared.py:3751`** · needs a decision first  
Per-fold hybrid rebuilds split the budget using scores computed on the whole dataset

- Computes: For each per-fold hybrid rebuild, it sets a representation's share of the budget from that representation's best cross-validated score across the entire dataset, not the fold's own training rows.
- Wrong because: The whole-dataset score includes each fold's own held-out rows, so the split uses that fold's own test labels. All five allocation methods shipping today route through this: performance_weighted, power_2, softmax_T1, rank_exp_d0.3, wglobal_power2.
- Touches: Reaches every build_fold_hybrid caller passing a blueprint: s04, s05, s06, s08, s10, s12, s13, n02, n07, n15, c11, c13, c15, c16, c18, c22, opig_blueprint_transfer. Since top_2/top_3/top_5 were dropped 2026-09-01, the live effect is a shift in per-representation weights, not which representations get used. nested_cv_evaluate is unaffected.
- Fix: Decide: keep reproducing the search's own representation choice as now, or score each representation fresh on each fold's own training rows.

**`tests/experiments/noise/n01_activity_cliff_framework.py:398`** · needs a decision first  ·  FIXED 2026-09-04 (upstream)
Each fold's hybrid budget split is set by scores fit on that fold's held-out rows too

- Computes: build_fold_hybrid passes each representation's whole-dataset baseline score into create_hybrid, which turns those scores into each representation's share of the fold's feature budget.
- Wrong because: Those scores are fit on all rows of a dataset, including the rows held out in that fold. So each fold's hybrid is built using numbers that already saw its own held-out rows.
- Touches: n01's hybrid-vs-representation gap is affected directly, with no per-fold reselection to cover it. n02 and n07 put the same composition on both compared sides, so it partly cancels. Whether stored results carry it is unverified.
- Fix: Score representations inside each fold's own training rows (nested cross-validation), or label the hybrid rows as selected on all rows, like n11 and n12 already do.
- FIXED 2026-09-04: no change needed in n01. `_shared.build_fold_hybrid` stopped forwarding the blueprint's whole-dataset `rep_scores`, and n01 passes no `precomputed_alloc_scores`, so `create_hybrid` re-scores every representation on the fold's own training rows. Read both this session: `_shared.py:3955` and `hybrid.allocate_performance_weighted`'s `precomputed_scores=None` path.

**`src/kirby/find_best_hybrid.py:7166`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Hybrid budget split uses a whole-dataset score, not each fold's own training rows

> Fixed: apply_blueprint injects `blueprint.rep_scores` only when `labels is None` — a rebuild on unlabelled rows, where nothing can be scored. Every labelled rebuild scores its representations on the rows it is given. The field stays on the Blueprint as provenance.

- Computes: find_best_hybrid writes each representation's best baseline score over the full dataset as rep_scores on the blueprint, then reuses it in every per-fold rebuild.
- Wrong because: That score decides how many features each representation gets, but it never changes per fold. A fold's own training rows never set its own budget split.
- Touches: Affects s06_applicability_domain.py, c11_ensemble_vs_hybrid.py, and opig_blueprint_transfer.py's native result — each claims leakage-free per-fold rebuilds. c05's full-data-blueprint result is unaffected: it deliberately uses a full-dataset score already; only its code comment is wrong.
- Fix: Stop injecting rep_scores into fold rebuilds unless its rows are disjoint from the search's rows, or use each fold's own score from create_hybrid's internal split instead.

**`tests/experiments/_shared.py:3751`** · re-run and wipe caches  
Per-fold budget split uses rep scores measured on that fold's own validation rows

- Computes: build_fold_hybrid takes the per-representation baseline scores from a whole-dataset search and uses them to split each fold's budget across representations.
- Wrong because: Those scores come from 5-fold CV over all rows, including this fold's own held-out rows. So each fold picks which representations get budget using a number that already saw its validation rows.
- Touches: Affects per-fold numbers in s04, s05, s06, s08, s10, s13, n01, n02, n07, n09 — but only runs after 2026-08-27 (commit b05e416); runs before that date re-scored each fold correctly.
- Fix: Default precomputed_alloc_scores to None in build_fold_hybrid so each fold re-scores representations on its own training rows unless a caller passes fold-computed scores.

**`tests/experiments/complexity/c18_diversity_vs_depth.py:243`** · re-run one experiment  
c18 natural arm's representation weights come from the whole dataset, not from each fold

- Computes: For each fold, depth_1/2/3 and uniform_all rank representations from training rows only, while the natural arm gets representation scores computed once from every row in the dataset.
- Wrong because: Those whole-dataset scores are built from rows that include each fold's own held-out labels. The natural arm's advantage over depth_1/2/3 partly reflects having seen data the other arms never see.
- Touches: No stored c18 result file is affected: this scoring was added 27 August 2026, and every c18 result in tests/results was written between 17 May and 2 June 2026, before it existed.
- Fix: Pass None, not an empty dict, for precomputed_alloc_scores in the build_fold_hybrid call, so each fold scores representations from its own training rows the way depth_1/2/3 already does.

---

## D. The group vector is degenerate, or is not the group the question needs

**8 open.**

**`src/kirby/datasets/nucleotide.py:366`** · FIXED 2026-09-04 (user chose k-mer clustering)  
gb_* genomics folds are set by an MD5 hash of the sequence, not a real group

- FIXED: `_assign_kmer_groups` clusters L2-normalised 5-mer profiles into 50 groups with
  MiniBatchKMeans, and the four Genomic Benchmarks datasets take their folds from that.
  50 groups rather than 5 so GroupKFold can still balance the fold sizes. On a check of
  20 families of ten near-identical sequences, all 20 families landed in one group each.
  `_assign_hash_groups` is left in place with a note and no callers.

- Computes: For gb_human_nontata_promoters, gb_human_enhancers_cohn, gb_demo_coding_vs_intergenomic_seqs and gb_demo_human_or_worm, load_genomic_benchmark assigns each sequence a fold by hashing it to an integer 0-4.
- Wrong because: A hash of the sequence carries no biology, so the five folds are close to a random split of the rows. The hash only keeps byte-identical duplicate sequences together — homologous or overlapping sequences still land in different folds.
- Touches: Any MCC already computed for these four datasets was measured under this hash split, not a real grouped split; whether a result file on disk carries one of these numbers wasn't checked here.
- Fix: Carry chromosome position through where a data mirror has it; for the two demo datasets, which have no coordinate at all, choose sequence-identity clustering or drop the grouped-split claim.

**`tests/antibody_baseline_survey.py:82`** · FIXED 2026-09-04 (user chose the unseen-antibody claim)  
Jain identity split gives one group per antibody, so folds aren't grouped by homology

- FIXED: the split is unchanged and the description is now written down. The loader
  measures what the 95% clustering produced (`describe_identity_grouping`), prints it, and
  returns `group_scheme` alongside `groups`; `load_data` carries both into every result.
  On these panels it reads 'one antibody per group'. V-gene family grouping was
  considered and not taken.

- Computes: For seven Jain datasets and flab_immunogenicity_marks, sequence_identity_clusters groups antibodies at 95% identity to build folds for GroupKFold.
- Wrong because: Each antibody lands in its own group — measured 136/137, 129/130, 207/217 groups to samples. GroupKFold then splits like plain KFold over distinct antibodies, not by homology.
- Touches: No number on disk is affected — median nearest-neighbour identity is 0.823, one antibody pair of 137 reaches 0.95. Only how these eight datasets are described changes: not homology-split, random split over distinct antibodies.
- Fix: Decide whether you want unseen-antibody generalisation (current split) or unseen-germline-family generalisation (V-gene grouping via ANARCI, already in cross_species_ness.py).

**`tests/experiments/similarity_metrics/base_rep_comparison.py:116`** · needs a decision first  ·  CLOSED 2026-09-04 — not a defect
koenig folds hold out whole positions, so two positional reps score near zero by arithmetic

- Computes: For flab_expression_koenig and flab_binding_koenig, base_rep_comparison predicts each held-out variant's measured value from its five nearest training neighbours, with folds grouped by mutated position.
- Wrong because: Onehot marks every position, but two single mutants always differ at exactly two entries, so every train-test distance is 2.0, a constant. dphyschem_pos shares no features between train and test positions, so the same five neighbours are picked for every query.
- Touches: Four of 32 rows in base_rep_comparison.json are wrong: onehot and dphyschem_pos on both koenig datasets. The koenig-vs-adams comparison for positional representations is invalid. Family verdicts are unaffected — neither representation was ever the best.
- Fix: Drop onehot and dphyschem_pos from the koenig family verdict, or add a second fold split that keeps each position present in both train and test.
- CLOSED 2026-09-04, not a defect: the user's call. Neither onehot nor dphyschem_pos is in the signed-off 10-representation pool for flab_expression_koenig (ablang2, esm2, esm2_concat, esm2_early, esm2_midlayers, esm3_cdr, esmc, igt5, prot_t5, saprot), and dphyschem / dphyschem_pos exist nowhere in src/kirby. Only the pruned representations run. The grouping IS now recorded per the scaffold_experiment.py:78 entry, with two flags saying whether a rho is arithmetic — every train-test distance identical, or every variant drawing the same five neighbours.

**`tests/experiments/noise/n06_bindingdb_ground_truth.py:295`** · re-run one experiment  ·  FIXED 2026-09-04
Six of n06's ten noise estimators skip the split-by-protein grouping the other four use.

- Computes: n06_bindingdb_ground_truth.py ranks ten noise estimators by the Spearman correlation between each row's residual and its replicate standard deviation.
- Wrong because: Six of the ten estimators never receive the split-by-protein grouping the other four use, so a repeated row's label can leak into its duplicate's score. That leak inflates the Spearman correlation used to pick the best one.
- Touches: n06's best_estimator pick and its Spearman ranking of all ten estimators. This record doesn't say whether a saved result file already reflects it.
- Fix: Route cross_val_residual through the same protein-grouped split as the other estimators, and either do the same for leave_one_out or drop it from screening.
- FIXED 2026-09-04: half of this entry is stale. Re-checked by constructing all ten estimators: `cross_val_residual` DOES take the protein grouping through the `inspect` check at n06:318, along with ensemble_disagreement, gaussian_process, conformal and bayesian_rf. The five that do not are the four nearest-neighbour ones and `leave_one_out`. `leave_one_out` now holds out whole (drug, target) pairs through `_oof_apply` instead of relying on out-of-bag, which never leaves a row's repeats out.

**`tests/experiments/similarity_metrics/scaffold_experiment.py:78`** · re-run one experiment  ·  FIXED 2026-09-04
No frac_single gate on the five-nearest-neighbour metric; the grouping goes unrecorded

- Computes: For each variant, predicts its activity from its five nearest neighbours, holding out variants by which position mutated first.
- Wrong because: The position_gated metric checks frac_single is at least 0.99 before running; this metric runs with no such check. The written JSON stores no frac_single and no group count, so a later reader can't tell the split was this uneven.
- Touches: No number scaffold_experiment.py writes is currently reported — SIMILARITY_METRICS_ROADMAP.md already excludes phillips_h3, and koenig is unaffected. The same grouping drives a committed verdict in base_rep_comparison.json.
- Fix: Gate this metric on frac_single like position_gated is gated, in both scaffold_experiment.py and base_rep_comparison.py, and write frac_single plus group counts into every output row.
- FIXED 2026-09-04: the five-nearest-neighbour score is refused below FRAC_SINGLE_MIN = 0.99, the same threshold position_gated was already refused on, in both scaffold_experiment.py and base_rep_comparison.py. frac_single, the position-group count, the variant count and the grouping's name go into every output row, with two flags saying whether a computed rho is arithmetic.

**`tests/experiments/similarity_metrics/winning_recipe_reselected.py:37`** · re-run one experiment  ·  FIXED 2026-09-04
winning_recipe_reselected.py splits pairs, not molecules, and normalizes features on both halves

- Computes: It reports structural fidelity on a held-out half of 8000 MCS pairs from up to 1427 molecules.
- Wrong because: Each molecule appears in about eleven pairs, so nearly every molecule sits on both the selection half and the scoring half. The feature z-scoring also uses all molecules, not just the selection half, so splitting the pairs by molecule would not fix it alone.
- Touches: This changes the +0.374 held-out number quoted in SIMILARITY_METRICS_ROADMAP.md and SIMILARITY_STUDY_HANDOFF.md as the study's circularity fix. winning_recipe_reselected.json on disk carries the unfixed number.
- Fix: Split molecules by scaffold group first, keep only pairs with both molecules on one side, and fit the z-score stats on the selection half only.
- FIXED 2026-09-04: the split is now on MOLECULES by whole scaffold group, a cached pair counts only when both endpoints are on one side, and the z-score mean/sd comes from the select half alone (make_D_fit). Read-across is scored on the score half's rows; read_rho_all keeps the every-row number beside it.

**`src/kirby/datasets/flab.py:190`** · edit only  
Phillips folds are grouped by distance from an arbitrary sequence, not wildtype

- Computes: For flab_binding_phillips_h3, it picks the most common heavy sequence as wildtype, then groups folds by how many positions each antibody differs from it.
- Wrong because: Every phillips heavy sequence is unique, so "most common" just returns the first row of the 5000-row subsample — an arbitrary antibody, not the wildtype. Folds end up grouped by distance from that point, so held-out folds still contain the same mutation loads as training.
- Touches: No phillips result file on disk uses this grouping — it was added after all phillips runs, and phillips is off the dataset roster. Two experiment scripts still use it today: antibody_region_ablation.py and antibody_finetune_comparison.py.
- Fix: Take wildtype from the file's genotype column — the all-one row is the mature parent — or set an explicit wildtype per dataset, and raise if it appears only once.

**`tests/experiments/opig_kingsmith_component_splits.py:422`** · edit only  
Inner search silently drops to 2-3 folds on two axes; nothing records or logs it

- Computes: For each outer fold, it picks the best hybrid recipe by an inner cross-validated search over the training rows' components.
- Wrong because: On the base and ligand axes, the inner search silently drops to 2 or 3 folds instead of 5. That reduction is never written to the split's JSON output, and the warning that would show it is silenced by filterwarnings('ignore') at line 92.
- Touches: Affects the cross-axis degradation comparison in docstring question 3 — no results file exists yet, so nothing on disk currently carries this.
- Fix: Write the inner fold count actually used into the split's JSON block per outer fold, and narrow filterwarnings('ignore') so it doesn't swallow the reduction warning or the collision warning.

---

## E. Classification folds lose stratification because task_type is not passed

**3 open.**

**`tests/experiments/noise/n01_activity_cliff_framework.py:280`** · re-run and wipe caches  
n01 skips stratified folds on classification datasets; five other files share the bug

- Computes: n01 builds its 5 cross-validation folds for the cliff and non-cliff scores it reports on classification datasets.
- Wrong because: It calls make_cv_splitter without passing task_type, so it gets plain GroupKFold instead of a stratified splitter already in the codebase. On an imbalanced fold, herg, flab_binding_adams, tz_her2_binary, tz_her2_multiclass, humanness_therapeutic, and both TCGA sets can produce a single-class validation set.
- Touches: On AUC datasets a single-class fold crashes the AUC calculation and n01 is recorded FAILED for that dataset, not silently wrong. On F1 datasets nothing crashes — folds are only worse composed, no number changes.
- Fix: Pass task_type into make_cv_splitter at n01:280, and the same missing argument at s04:173, s05:116, s06:159, s06:460, and s08:127.

**`tests/experiments/similarity/s06_applicability_domain.py:159`** · re-run and wipe caches  
s06 and s08 build classification folds as plain GroupKFold, never stratified

- Computes: s06 and s08 build cross-validation folds from the group labels for every dataset in the similarity roster, by calling make_cv_splitter(n_folds, groups).
- Wrong because: Without task_type, make_cv_splitter returns plain GroupKFold, not StratifiedGroupKFold, on classification datasets. A fold can land only one class; s06 drops that fold as NaN, losing power there without corrupting other folds.
- Touches: Changes fold composition on classification datasets: herg, tz_her2_binary, tz_her2_multiclass, humanness_therapeutic, the TCGA and nucleotide sets, in s06 and s08 only. s10 already stratifies on its own; s06's other unstratified call is a regression-only branch that never runs on these datasets.
- Fix: Pass task_type=data['task_type'] at s06:159 and s08:127.

**`tests/experiments/similarity/s04_activity_cliffs.py:173`** · re-run one experiment  
Classification folds aren't stratified by class in six files, not just two

- Computes: For grouped classification datasets, make_cv_splitter builds folds without balancing classes across folds.
- Wrong because: s04, s05, and four other files call it without passing task_type, so it falls back to plain group folds instead of class-balanced ones. Groups still stay intact, so this doesn't leak rows between train and test — it just makes some folds imbalanced by class.
- Touches: s05 already has AUC numbers on humanness_therapeutic and tz_her2_multiclass computed under these imbalanced folds. s04 has no herg result yet. s06, s08, and n01 have the same bug, unrun status unclear.
- Fix: Pass task_type=task_type into all six calls: s04 line 173, s05 line 116, s06 lines 159 and 460, s08 line 127, n01 line 280.

---

## F. A quantity computed on every row enters a held-out score

**14 open.**

**`src/kirby/datasets/tcga.py:295`** · FIXED 2026-09-04 (user chose to drop ComBat)  
TCGA batch correction is fit on held-out samples, and separately per class for LUAD vs LUSC

- FIXED: the ComBat pass is gone from `_download_xena_expression`, for every TCGA task.
  Fitting it inside the fold is not available while the folds are grouped by tissue source
  site: a fold holds out whole sites and ComBat has no batch parameter for a site it never
  saw. The site codes still travel with the data as `tss_groups`, and the split is what
  handles the site effect. BILL_OF_MATERIALS.md was corrected in place.

- Computes: For tcga_luad_vs_lusc, it fits ComBat batch correction on all LUAD samples in one call and all LUSC samples in a separate call, before any fold split.
- Wrong because: The correction for a site is fit on all of that site's samples, including the held-out ones a fold later tests on. For LUAD vs LUSC, the correction for each sample is fit using only samples with its own class label.
- Touches: No cached file is affected: combat is not installed by default, so this step never ran; the three cached files predate the code. Check if the server rebuilt with combat installed since 27 Feb.
- Fix: Fit the batch correction inside each fold, on training rows only, and correct both LUAD and LUSC samples together in one matrix.

**`src/kirby/find_best_hybrid.py:5958`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Interval coverage passes by construction; feature selection leaks calibration labels

> Decided 2026-09-04: three scaffold-disjoint blocks. The code is in `tests/experiments/cross_species/x_mol_rd_analysis.py`, not in the engine. `_calibration_split` returns fit / width / coverage blocks; the blueprint's recipe is applied to the fit block through apply_blueprint, so only its labels choose features; coverage is counted on the third block. The cached FittedResult, whose selected_indices came from a search over every molecule, is no longer read. The JSON key `calibration_coverage` is renamed `coverage` so an old reader fails rather than comparing the two.

- Computes: It reports a coverage percentage for prediction intervals, calculated on the same molecules whose residuals set the interval width.
- Wrong because: calibration_coverage passes by construction: it is measured on the same molecules whose residuals set the interval width. Those molecules' labels also picked which features feed the model, so real interval widths, and the flags built from them, run narrower than they should.
- Touches: This affects calibration_coverage and the below_potency_floor / species_divergence flags computed by x_mol_rd_analysis.py. No result file exists yet for that experiment, so nothing already computed is invalidated.
- Fix: Hold the calibration fold out before running find_best_hybrid, select features and fit only on the training rows, then calibrate on the fold that was held out.

**`tests/experiments/complexity/c07_data_complexity_partitions.py:156`** · needs a decision first  
c07's target_variance split decides easy/hard using the same labels it later scores.

- Computes: For each scaffold group, c07 computes the variance of its own labels, then splits its rows into easy or hard by the median across groups.
- Wrong because: No label reaches the fitted model, scaler, or importance ranking, since each fold holds the whole group out together. But easy/hard membership needs the label to assign, and that same label variance is also part of what the two halves are compared on.
- Touches: Affects all five stored target_variance files (abagym_Ang2, EGFR, VEGF, HER2, buchwald): the easy/hard gap isn't a generalization measure — membership came from the labels being scored. The singleton-default bug affects no file on disk.
- Fix: Drop target_variance as a partition axis, replace it with a label-independent proxy, or compute each fold's variance cut from its training rows only.
- **Fixed 2026-09-04:** `target_variance` is kept and every block it writes carries `partition_uses_labels: True`. A label-free counterpart `feature_variance` was added beside it: it splits scaffold groups by their spread in representation space, z-scored per representation first. No axis was cut.

**`tests/experiments/noise/_similarity_hybrid_space.py:149`** · needs a decision first  ·  FIXED 2026-09-04
hybrid_sim's features are picked using every molecule, not just each fold's training rows.

- Computes: build_similarity_hybrid picks which physchem and fine-2D features go into the hybrid_sim representation, using structural pairs sampled across every molecule in the dataset, not just each fold's training rows.
- Wrong because: The feature choice sees each fold's held-out molecules, which the codebase elsewhere treats as a leak. The comparator arm isn't fully per-fold either: only its feature computation is, and its recipe is chosen once on the whole dataset using clean labels.
- Touches: No result file on disk is affected: this arm can't even switch on for herg_ki or logd from the server, and the only files that exist (buchwald_hartwig) don't contain hybrid_sim.
- Fix: Restrict the 8000 structural pairs to each fold's training rows before the two feature-selection steps, rebuilding the feature choice per fold; if that's too costly, label the run selection_rows='all'.
- FIXED 2026-09-04: `build_similarity_hybrid` takes `selection_rows`, and n11 and n13 pass each fold's training rows. Only structural pairs with BOTH molecules in those rows choose the physchem and fine-2D blocks, and the z-score mean and standard deviation are fitted there too. The one 8000-pair MCS draw is reused, so a per-fold rebuild pays the greedy descriptor drop and no MCS pass; `similarity_hybrid_available` prints the arm ON/OFF without a throwaway whole-dataset build. Every result records `hybrid_sim.selection_rows` and the pair count each fold selected from.

**`tests/experiments/noise/n01_activity_cliff_framework.py:258`** · needs a decision first  ·  FIXED 2026-09-04
n01's cliff reference on reaction, TCGA, and DTA is built from the representations it grades

- Computes: For reaction, TCGA, small-molecule DTA, and any molecular dataset where RDKit fails, n01 finds cliffs from the closest 2% of pairs after concatenating the ten representations it is comparing.
- Wrong because: The cliff set used to grade a representation is built from that same set of representations. Whichever contributes the most features — methyl_hvg on TCGA, the three fingerprints on buchwald — defines most of the cliffs it's later graded on.
- Touches: No stored result is wrong: reaction, TCGA, and DTA n01 files all returned zero cliffs under the old threshold. The next noise run will produce a gap number tied to the representations it grades.
- Fix: Give reaction, TCGA, and DTA datasets a cliff reference built outside the representations being compared, or drop the non-circularity claim for them before the next run.
- FIXED 2026-09-04: reaction datasets get a reference built from reactant and product Morgan bits off `rxn_smiles`, which `load_data` now carries through; drug-target affinity gets Tanimoto on the drug, zeroed between different proteins so a repeated drug is not a cliff against itself. TCGA has no source outside its own representations, so it keeps the concatenation and every result records `cliff_reference_independent: False` with each representation's feature count. The docstring's non-circularity claim is now conditional on that flag.

**`tests/experiments/noise/n07_remediation.py:502`** · needs a decision first  ·  FIXED 2026-09-04 (features; recipe deferred)
n07 picks hybrid features using clean labels, before the labels it studies are corrupted

- Computes: n07 reports each remediation strategy's recovery percentage, per fold, per noise level, per model, against a noisy baseline.
- Wrong because: That baseline is scored on features chosen using clean labels, before noise is added, which favors all four remediation strategies over it in noise detection. One produced file already has a negative denominator, making its percentages uninterpretable for a separate reason.
- Touches: recovery_pct, per_model_recovery_pct and noise_gap in n07's output are unreliable. The strategy ranking (best_hybrid_strategy, best_strategy_per_model) is unaffected — every strategy in a model's column shares one feature space. Only buchwald_hartwig.json exists, and its gap is already negative.
- Fix: Pick features from each fold's noisy labels, not clean ones, and fetch the noise-aware search instead. Keep a clean-label row too, but label it an oracle, as n15 does.
- FIXED 2026-09-04: each fold now builds its hybrid from that fold and noise draw's NOISY training labels — one build per (fold, noise seed) instead of one per fold. The clean-label space is kept as a named oracle arm, `per_model_per_strategy_oracle_features`, so the head start is readable. The RECIPE still comes from the clean-label pool: the user's decision this session was to fix n15's selection first and read its number before moving n07 and n11 onto a noisy search, which the stored noisy lineage (0.05/0.10/0.20 gaussian) does not cover for n07's uniform p=0.1 condition. Separately, the negative denominator: a recovery percentage is now NaN when the injected noise did not degrade its reference, with the reason under `recovery_undefined`, and the direction comes from the metric so a negative gap on RMSE is not treated as an error.

**`tests/experiments/complexity/c17_information_theoretic_decomposition.py:297`** · re-run and wipe caches  
c17: three of five feature-selection strategies see validation-row labels before scoring

- Computes: For each budget, c17 computes downstream_r2 and mi_density for five feature-selection strategies, all measured over the same rows.
- Wrong because: Hybrid, mRMR, and recursive elimination pick features using every row's label, including validation rows; PCA and the random baseline pick features with no labels at all. mi_density is also measured in-sample for all five strategies alike, so best_strategy favors whichever strategy was built to maximize that same number — not necessarily mRMR, which actually scores 0.0 at every budget in abcov_kd.json.
- Touches: Every dataset file c17 produces (e.g. abcov_kd.json) carries downstream_r2 and best_strategy computed this way. The claim that hybrid keeps more predictive information per feature than mRMR, PCA, recursive elimination, and random at matched budget is not supported by this code.
- Fix: Move feature selection for hybrid, mRMR, and recursive elimination inside each fold: select on training rows only, apply the chosen features to validation rows, then measure mi_density and downstream_r2 there. PCA and random need no change.
- **Fixed 2026-09-04:** all five strategies now select inside each fold on its training rows and are measured on its held-out rows. The mRMR ranking is built per fold (one greedy pass per fold, still prefix-sliced across budgets), recursive elimination runs per fold, PCA and the scalers are fitted on training rows. The hybrid's features come from `build_fold_hybrid`; its recipe still comes from a whole-dataset search, stamped `recipe_selected_on`.

**`tests/experiments/noise/n15_clean_vs_noisy_hybrid.py:184`** · re-run and wipe caches  ·  FIXED 2026-09-04 (upstream)
n15 picks its noisy-label blueprint by scoring candidates against clean labels

- Computes: n15 picks the noisy-label blueprint's winning configuration by scoring each candidate against the clean validation labels in the search's inner folds.
- Wrong because: Both blueprints are chosen using the true labels, so the reported gap only measures worse fitting, not worse selection. A practitioner working with noisy labels has no clean labels to select a configuration with, so a small gap cannot clear n11 and n12's warning.
- Touches: n15's oracle_advantage number, and any decision to drop n11/n12's oracle-bias warning based on it. Any saved n15 result file needs relabeling as a training-only head start, not the full advantage.
- Fix: Score the search's inner validation folds on the noisy labels too when label_noise is set, and keep the clean labels only for n15's own final held-out scoring.
- FIXED 2026-09-04, and it matches the user's decision this session: `find_best_hybrid` takes `score_on_noisy_labels`, and n15's noisy lineage passes it, so its inner validation folds are scored on the noisy labels and the winning recipe is picked with no clean label anywhere. Both `tier` and the noise id carry a `__selnoisy` marker, so the clean-scored and noisy-scored lineages cannot serve each other from the result database or the filename caches, and `get_best_search_for_downstream` refuses the in-memory dicts for a noisy-selection request. The three stored noisy searches per model per dataset (0.05/0.10/0.20) have to rebuild.

**`src/kirby/find_best_hybrid.py:3725`** · re-run one experiment  ·  FIXED 2026-09-04 (P8-engine)
n15's oracle_advantage only isolates the fitting head start, not the selection head start.

> Fixed: `find_best_hybrid(score_on_noisy_labels=True)` scores the inner validation folds on the noisy labels, in `_evaluate_single`, `_evaluate_hybrid_config` and the geometry baselines alike. It requires `label_noise` and rides `__selnoisy` on `tier` and on the noise id, so it shares no cached baseline, hybrid-eval fold or stored row with a clean-scored run. n15 passes it for the noisy lineage, and `get_best_search_for_downstream` stops reading the in-memory noisy pool when it is set. Same fix covers `n15_clean_vs_noisy_hybrid.py:184`.

- Computes: n15 computes oracle_advantage: score of a hybrid built on clean labels minus score of one built on noisy labels, per dataset.
- Wrong because: Both searches pick their winning configuration using the same clean validation labels; only training differs. So oracle_advantage only measures the head start from fitting on clean labels, not from picking with them.
- Touches: Affects n15's conclusion that oracle_advantage near zero lets n11/n12 drop their oracle-bias warnings. Numbers on disk are correct; the conclusion drawn from them is not.
- Fix: Add a mode where find_best_hybrid scores inner folds on the noisy training labels too, so selection gets no clean-label access, and run n15's noisy search under it.

**`src/kirby/utils/lnl_methods.py:426`** · re-run one experiment  
Seven of twelve LNL methods pick their checkpoint using labels n09 scores them against

- Computes: For seven of twelve LNL methods, train_and_predict picks final weights by loss on y_val, the same labels n09 scores predictions against.
- Wrong because: Ranking all twelve methods by win count compares seven that select their checkpoint on y_val to five that don't. A method's own hybrid-versus-base change stays valid, since both sides get the same y_val treatment.
- Touches: n09_lnl_survey.py's summary fields — top3_base, top3_hybrid, best_*_lnl, ranking_changed — rank methods across two checkpoint-selection protocols. Any existing result file from that script carries this, though each method's own base-to-hybrid delta is unaffected.
- Fix: Carve a selection split out of the training rows inside train_and_predict and use it for every checkpoint and meta-gradient, then pass X_val with no labels.

**`tests/experiments/complexity/c09_formal_analysis.py:371`** · re-run one experiment  
c09 measures retained label information on the same rows that picked the features

- Computes: For each fold and budget, c09 picks features using the training rows' labels, then measures how much of those same labels the features carry.
- Wrong because: Features are chosen because they match those labels, so the measured amount is inflated by that same match. Whether the inflation grows or shrinks with the budget is not established.
- Touches: The per-budget curve and results['inflection_budget'] that c09 saves are in-sample numbers, not real retained label information. No other script reads inflection_budget.
- Fix: Keep feature selection on each fold's training rows, but measure the information on that fold's held-out validation rows instead.
- **Fixed 2026-09-04:** 9b now applies each fold's selected features to that fold's held-out rows and measures both MI(X;Y) and the retained-variance number there; the scaler, PCA and ridge are all fitted on the training rows.

**`tests/experiments/noise/n09_lnl_survey.py:418`** · re-run one experiment  
n09: 7 of 12 noise-cleaning methods pick their checkpoint on labels they're then scored on

- Computes: n09 ranks noise-cleaning methods across datasets and reports which method comes out on top, using top3_base, top3_hybrid, best_*_lnl, and ranking_changed.
- Wrong because: 7 of the 12 methods pick their best training epoch by checking loss on the same validation labels that later score them. The other 5 methods, and the baseline runs with no noise-cleaning, never see those labels before scoring.
- Touches: Affected: n09's ranking_changed, top3_base, top3_hybrid, best_*_lnl, and the retention deltas against the no-LNL reference. Same defect recurs in n13_noise_distributions.py:664-666. Not affected: n_hybrid_improves and n_hybrid_harms, which compare each method to itself on base versus hybrid features.
- Fix: Inside each fold, carve an early-stopping holdout out of the training rows, keeping folds group-aware; pass that holdout in place of the validation set to train_and_predict.

**`tests/experiments/complexity/c06_within_family_gradient.py:186`** · edit only  
c06's default mode fits importance scores on rows it later validates on

- Computes: c06 picks each variant's best budget by scoring feature importance on an 80/20 train/test split of each dataset.
- Wrong because: This only happens when you run c06 with no --mode flag, its default single 80/20 split. It reuses importance scores from a 5-fold split, so most of the 80/20 test rows already sat in the training data behind those scores.
- Touches: No result already on the cluster is wrong: the production run (submit_dataset.sh, run_dataset.py) uses 5 folds on both sides. Only c06 runs launched with no --mode flag, or through the local smoke driver, are affected.
- Fix: Pass the fold count into get_precomputed_search at line 126, or skip the cached importance scores when the fold counts don't match.
- **Fixed 2026-09-04:** c06 now asks for the importance scores at its own fold count, and discards the cache when its fold indices do not cover exactly that count.

**`tests/experiments/noise/n06_bindingdb_ground_truth.py:300`** · edit only  ·  FIXED 2026-09-04
n06 scores six of ten noise estimators using their own replicate labels

- Computes: For each row, n06 correlates a noise estimator's score against the standard deviation of that row's own replicate group.
- Wrong because: Six of ten estimators include a row's own replicate rows when computing its score, so part of the correlation is arithmetic, not signal. The other four group folds by protein target, so replicate rows stay in the same fold together and their scores aren't contaminated.
- Touches: No n06 result file exists yet and no SLURM script submits this experiment, so no number on disk is affected. This decides what the first run's best-estimator result will mean.
- Fix: Score each row using only rows from a different drug-target pair, and rebuild the hybrid space on the training side of that split.
- FIXED 2026-09-04, user decision this session: the four nearest-neighbour estimators take `exclude_groups`, so no row draws a neighbour from its own (drug, target) pair — the new `_knn_indices` queries past the largest group and masks. `leave_one_out` holds the whole pair out. On a fixture of 12 pairs x 4 replicates, distance_weighted_residual's correlation with replicate spread fell from +0.68 to +0.04. The hybrid space is also rebuilt protein-fold by protein-fold, each row's features chosen from rows outside its own protein group; `hybrid_space_provenance` records it, or why it fell back.

---

## G. A cached result can be served across a boundary that changes it

**8 open.**

**`src/kirby/result_cache.py:45`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
Search-result cache matches rows without checking metric or task_type.

> Fixed: `metric` and `task_type` are IDENTITY_COLS, supplied from the search meta. That widens the primary key, so every `result_db/*.db` written before 2026-09-04 has to be deleted; `_ensure_schema` raises naming the file and the missing columns instead of failing later inside a SELECT.

- Computes: The result database matches a cached search row using six fields — tier, labels hash, groups hash, folds, seed, and fold-splitter id — not metric or task type.
- Wrong because: task_type picks the fold splitter — classification gets StratifiedGroupKFold, everything else gets GroupKFold — so the same folds split differently. metric sets what the stored score means, but a fetched row's ranking_metric is dropped on read and never checked.
- Touches: No reported number is wrong today — every current caller gets metric and task_type from DATASET_REGISTRY automatically. One experiment, similarity_optimized_hybrid.py line 305, sets metric='rmse' explicitly and skips that check.
- Fix: Add metric and task_type to the cache match, or reject a fetched row whose stored ranking_metric doesn't match the metric you requested.

**`tests/experiments/_shared.py:58`** · re-run and wipe caches  
Per-fold feature cache key skips labels and dataset name, risking a collision across datasets

- Computes: A cache key for the features chosen in one fold, from the training row indices and each representation's shape — not from labels or dataset name.
- Wrong because: Two datasets with the same training row indices, the same representation list, and the same allocation settings get served each other's selected features, even with different labels. The five FLAb Jain datasets share identical heavy and light chain sequences, so they can produce that exact match.
- Touches: No wrong number exists yet — c09 runs one dataset at a time. Running two of the five FLAb Jain datasets in the same cache directory in one session risks a collision, since they share sequences and groups.
- Fix: Add the dataset name and hashes of the labels and groups to the cache key, and hash the full keyword-argument set instead of three chosen entries.

**`src/kirby/find_best_hybrid.py:3793`** · re-run one experiment  
The hybrid_eval cache skips fine-tuned per-fold representations, returns a frozen score.

- Computes: For each fold, the hybrid_eval cache key and metadata are built from the predicted top-k base representations and their feature widths only.
- Wrong because: A search using fine-tuned per-fold representations looks identical to the cache as a frozen search: same dataset, representations, labels, seed, and folds. So it reads the frozen run's cached score instead of computing its own.
- Touches: Affects c02's frozen-vs-fine-tuned comparison under a top_k allocation. Not reachable on HEAD unless top_2/top_3/top_5 is passed explicitly — dropped as default 2026-09-02. Files from before that date still carry it; find_best_hybrid_dta.py hits it with --allocation top_3.
- Fix: Add `and not fold_reps` to the cache-enabled check, so runs with per-fold representations compute their own folds instead of reading the cache.

**`tests/experiments/complexity/c13_forward_compatibility.py:494`** · re-run one experiment  
D and D_random inherit C's recipe, picked before the new representation was included.

- Computes: c13 computes three per-fold hybrid scores per dataset: one using existing representations, one adding a new representation, one adding a random noise matrix.
- Wrong because: D and D_random reuse C's full recipe — budget, allocation method, importance method, evaluator, model — chosen before the new representation existed. A representation whose payoff needs a different budget or allocation method cannot be credited.
- Touches: No result file on disk is affected — every existing c13 result predates the shared result store and is already flagged stale-schema. The claim affected is prospective: c13's future runs, until fixed.
- Fix: Pass cache_namespace (not just condition_suffix) into find_best_hybrid so D and D_random get their own result-store file, separate from C's.

**`tests/experiments/run_dataset.py:126`** · FIXED 2026-09-04  
Resumed runs skip stale checks: file existence only, never folds or representations.

- FIXED: `input_fingerprint(data, config)` records the row count, the label hash, the
  group hash, the group count, the split count and the sorted representation names with
  their widths; `save_experiment_results(fingerprint=...)` stamps it into every result
  file. The complexity, similarity and noise resume checks compare it through
  `fingerprint_mismatch` and recompute on any difference, naming which field moved. A
  file written before the stamp has none and always recomputes. federated stamps the
  fingerprint but has no resume path to read it.

- Computes: run_dataset.py decides, per experiment and dataset, whether to reuse an existing result file instead of recomputing it.
- Wrong because: The check only asks whether the output file exists, or, for noise and similarity, whether it holds a measurement. It never asks whether the fold grouping, labels, or representation list changed since that file was written.
- Touches: Affects c08's calibration comparison and c09's claim that variance is mostly model, then budget — only for a run resumed after a fold-grouping or representation-roster change. No current file on disk is confirmed stale; unverified.
- Fix: Save the fold grouping, labels, split count, and sorted representation list in each result file, and compare them before skipping a recompute.

**`src/kirby/find_best_hybrid.py:5027`** · edit only  ·  FIXED 2026-09-04 (P8-engine)
Precompute sets budgets using a split create_hybrid stopped using

> Fixed by importing rather than copying: the carve is `hybrid.internal_holdout_split`, called by create_hybrid and by both of find_best_hybrid's precomputes. The GroupShuffleSplit copy is gone and the two cannot drift again.

- Computes: find_best_hybrid.py scores each representation on one train/validation split of a fold's training rows, then hands those scores to create_hybrid to set that representation's budget.
- Wrong because: A comment there says this split matches create_hybrid's own split; it no longer does. create_hybrid switched to a different split on 20 August, but every default allocation method still takes its scores from the old one.
- Touches: No result already on disk is wrong: both splits sit inside a fold's training rows, so no reported number changes meaning. It changes which representations get budget in future predictive searches.
- Fix: Build the same split create_hybrid uses now inside the precompute too, or have it import that split, so the two can't drift apart again.

**`src/kirby/find_best_hybrid.py:5221`** · edit only  
Resuming a run accepts a checkpoint left by the other pruning mode, unnoticed

- Computes: _load_checkpoint checks each field the current run expects against the saved checkpoint, but never checks for a field the saved file has that isn't expected.
- Wrong because: Default and aggressive pruning differ in two fields only; the default run's checklist skips both, so it accepts an aggressive checkpoint. That checkpoint holds the aggressive run's pruned config list, so the default run evaluates about 58% fewer configs while reporting a full search.
- Touches: No result file is confirmed corrupted by the record. Risk applies to any dataset/tier resumed locally under default_pruning after a cluster run used aggressive_pruning — the cluster scripts default to aggressive, the python scripts default to default.
- Fix: Make _load_checkpoint reject a checkpoint whose stored fields include a pruning field the current run's expected fields lack, or always write both pruning fields.

**`src/kirby/find_best_hybrid.py:5736`** · edit only  
Refinement rows can land on different folds than phase-1 rows, but that path never runs

- Computes: _run_refinement_phase re-scores the top hybrid configs from phase-1 search at other step sizes through _evaluate_hybrid_config, then merges those rows with phase-1 rows and ranks them together.
- Wrong because: It does not pass task_type, objective, train_labels, noise_id, row_index or structural_pairs into _evaluate_hybrid_config, though phase-1 search passes all six. On a classification dataset this puts phase-1 rows on StratifiedGroupKFold folds and refinement rows on GroupKFold folds, then ranks them together.
- Touches: f04 is the only caller of this code, and it always raises before reaching refinement because it passes no task_type. No result file on disk is affected.
- Fix: Add task_type, objective, train_labels, noise_id, row_index and structural_pairs as parameters of _run_refinement_phase and forward them into that _evaluate_hybrid_config call.

---

## H. Arithmetic that lets a broken configuration win, or that unpairs the two arms

**8 open.**

**`src/kirby/hybrid.py:822`** · re-run and wipe caches  ·  FIXED 2026-09-04 (P8-engine)
One held-out group makes f26's similarity allocation methods collapse to a default

> Fixed: `_require_scored_reps` raises when no representation scores finitely on the held-out block, in allocate_performance_weighted and allocate_weighted_global; `blended_rep_scores` raises when a blend component scores non-finite for every representation, which used to drop that component from the blend in silence. The per-representation score already returned NaN for a degenerate block — it was the allocator that read NaN as a tie.

- Computes: For each representation, create_hybrid scores it by predicting each molecule's activity from its five nearest neighbours, on a held-out block carved from a training fold.
- Wrong because: On tz_her2_binary, that held-out block always contains one CDRH3 edit-distance group, so every representation scores exactly 0.0. Allocation methods then fall back to an equal split of the budget across representations, or to ranking representations by their order in the roster, not by any real comparison.
- Touches: f26's objective='similarity' column for tz_her2_binary: allocation methods there never really competed, all fell to equal split or roster order. Separately, ginkgo_hic's similarity cells cannot be scored at all. logd and herg_ki, with 769 and 2413 scaffold groups, are unaffected.
- Fix: Make the per-representation score return no-score, not 0.0, when the held-out block is degenerate, and have the allocator raise instead of silently defaulting.

**`src/kirby/find_best_hybrid.py:5446`** · re-run one experiment  
A NaN score can grab a top-3 slot and knock out a real allocation method, in f27 only

- Computes: After fold 0, f27 ranks allocation methods within each importance/evaluator/model/budget group by score_mean and gives points to the top 3.
- Wrong because: Python sorts NaN scores as if they beat real numbers, so a NaN result can take a top-3 slot meant for a real one. This only happens when one group mixes NaN and real scores, with the NaN sorted early.
- Touches: f28 never reaches this code: it only ever has 2 or 3 allocation methods, below the 4 needed before this prune runs. Only f27's per-objective allocation comparison is affected; no held-out score is touched.
- Fix: Sort with the same sentinel the other prune already uses: send NaN to worst-possible instead of comparing it directly, so it always sorts last.

**`tests/experiments/complexity/c22_explainability.py:231`** · re-run one experiment  
c22 averages SHAP scores across folds but labels them with the last fold's features

- Computes: c22 computes a SHAP importance score per feature on each fold's validation rows, then averages that score across folds and reports it under one fold's feature names.
- Wrong because: Each fold selects its own features from its own training rows, so position 37 can name a different feature in each fold. Averaging SHAP scores across folds and labeling them with one fold's feature names mixes different features together.
- Touches: The top-30 feature list is mislabeled but unused. The per-rep SHAP breakdown holds for fixed-score allocation methods but not entropy or pairdist-redundancy weighting, and every saved c22 result predates the fix.
- Fix: Compare each fold's selected features per representation, not just the total feature count, and average SHAP scores within matching (representation, feature) pairs instead of by index.
- **Fixed 2026-09-04:** matched per fold, then averaged — the shape `EXPERIMENTS_FINALIZED.md` already records as decided for c22. `aggregate_importance_by_rep` now runs inside the fold loop on that fold's own feature map, and `shap_by_rep` is the mean of those. `shap_share_by_rep` is the mean of each fold's own share, so a fold with a larger total does not outvote the others; `shap_by_rep_folds` and `allocation_by_fold` keep the per-fold numbers. The top-feature list and the split counts are labelled as the last fold's, because no feature position is comparable between folds when the split moves. `firstpass_notready.py` prefers the share.

**`tests/experiments/noise/n11_estimation_remediation_grid.py:522`** · re-run one experiment  
hybrid_sim's per-column rescaling in n11 and n13 cancels its own similarity weights

- Computes: n11 and n13 fit a StandardScaler on training-fold rows of hybrid_sim and use that as the fold's distance space.
- Wrong because: That StandardScaler divides each feature by its own standard deviation, which is the same 1/sqrt(block-width) times per-block weight already built into hybrid_sim. Refitting it removes that weighting, so hybrid_sim becomes a plain z-scored concatenation where each block's influence is just its feature count.
- Touches: This is a design defect, not leakage — the split is fine. Any n11 or n13 result file with hybrid_sim numbers for herg_ki, logd, or adme_solubility measures a feature-count-weighted concatenation, not the similarity recipe.
- Fix: Stop re-fitting a StandardScaler on hybrid_sim after building it; use the training rows as they come, or only re-center them without rescaling variance.

**`tests/experiments/noise/n09_lnl_survey.py:338`** · edit only  
n09 can average base and hybrid scores over different folds when only some folds fail

- Computes: n09 averages base_no_lnl/hybrid_no_lnl and base_lnl/hybrid_lnl scores per condition, then compares those averages to call hybrid better or worse.
- Wrong because: When a fold's hybrid build fails, that fold's score is dropped from the hybrid list but kept in the base list, so the two averages can come from different folds. n_folds is saved per condition but the comparison code never checks it before comparing.
- Touches: No saved result currently shows this. The all-folds-failed case is already caught (ranking_changed reports as None). The one saved n09 file has every fold failing on the hybrid side, not some — a different, already-guarded case.
- Fix: Save the fold index with each score, and compare base and hybrid averages only over folds present in both. Also record which folds fell back to the clean-label blueprint when the noisy search found nothing.

**`tests/experiments/noise/n14_activity_cliff_landscape.py:342`** · edit only  ·  FIXED 2026-09-04
n14's residual-SD outputs are in label-SD units, but compared to a fixed pKi noise floor

- Computes: n14 maps labels through from_linear, then reports log_resid_sd, tier_sd, tier_mean_bias, and overall_log_bias for each molecular regression dataset it runs.
- Wrong because: The Kramer 0.54 pKi noise floor only matches these values when a dataset's label SD is 1.0, which is not true in general. On herg_ki, label SD ~1.2, a reported 0.45 is actually 0.54 pKi — right at the floor, but reads as comfortably below it.
- Touches: n14 has never been run, so no result file on disk is affected today. The same scale problem applies to tier_sd, tier_mean_bias, overall_log_bias, and the bias_flag cutoff.
- Fix: Keep the params from_linear returns, multiply the four residual fields by the saved label SD, and record which scale the 0.54 pKi floor uses.
- FIXED 2026-09-04: `_rescale_stage0` multiplies the residual spread, the five tier spreads, the five tier biases and the overall bias by the label standard deviation `from_linear` returns, and writes them under `*_label_units` next to `kramer_noise_floor_pki`, `resid_sd_over_kramer_floor` and a label-units bias flag. Checked on a herg-like label column with standard deviation 1.198: a reported 0.45 comes out as 0.539 pKi against the 0.54 floor.

**`tests/experiments/similarity_optimized_hybrid.py:886`** · edit only  
folds_are_seed_invariant is hardcoded True but false for herg's folds

- Computes: Each herg.json records a field, folds_are_seed_invariant, set to True for every dataset and seed, describing whether folds are the same across the three seeds.
- Wrong because: On herg both search folds and Payoff-1 scoring folds change across seeds, because herg's integer labels trigger StratifiedGroupKFold with the seed as random state. The field says folds stay fixed; for herg they do not.
- Touches: No reported number changes. Nothing in the repo reads this field back — read_sim_optimized_results.py only prints each interval's mean. herg.json carries one false statement about how its folds are built, not a wrong result.
- Fix: Write folds_are_seed_invariant from task type instead of hardcoding True, so classification datasets like herg record that their folds change per seed.

**`tests/find_best_hybrid_molecular.py:1554`** · edit only  
--tune picks hyperparameters on every group, then scores those same groups as held-out

- Computes: With --tune, the three drivers pick hyperparameters using every group in a dataset, then report fold scores on that same dataset.
- Wrong because: The winning hyperparameters were picked using the same groups each fold later holds out for validation. find_best_hybrid's own documentation names this exact pattern as tuning on the test fold.
- Touches: No result file on disk carries this. No driver ever wrote a tuned results file, and the only scripted --tune calls pass --search fast, an argument no driver defines, so they crash before running.
- Fix: Delete the tune_models block from all three drivers; pass tune=True, tune_n_iter=args.tune_iter into find_best_hybrid so tuning runs per fold.

---

## I. The experiment does not run, or measures something other than its claim

**7 open.**

**`tests/experiments/federated/fed01_noise_heterogeneity.py:523`** · needs a decision first  
fed01 condition 6 differs from condition 5 four ways, so their MCC gap isn't detection alone

- Computes: Computes condition 6's MCC score at each noise level and fold, then compares it to condition 5's MCC in the same table.
- Wrong because: Condition 6 pools every silo's cleaned rows into one feature selection, the same call condition 0 uses, not per-silo like condition 5. It also differs in augmented features, in whether a model is passed, and in how the 100-feature budget is split across silos.
- Touches: No production fed01 results exist yet, only smoke logs, so no reported number changes yet. Fix this before the next full run on caco2 and logd.
- Fix: Rebuild condition 6 with per-silo feature selection on the cleaned rows, like condition 5, and check its silent fallback to raw Morgan fingerprints.
- FIXED 2026-09-04: condition 6 calls `federated_feature_selection`, the same call condition 5 makes, with the flagged rows removed and silo membership remapped to the cleaned rows. Per-silo selection, 100 features split across silos, no augmented features, model passed. The Morgan-fingerprint fallback is gone: a failure now records an error and a traceback instead of an MCC under a `note` key. User decision, this session.

**`tests/experiments/federated/fed04_cluster_aware_importance.py:146`** · needs a decision first  
fed04 crashes on caco2 and logd before writing any result, not a corrupted metric

- Computes: fed04 turns each row's label into a binary class (label >= 6.0) for every row in caco2 and logd, then fits folds and a classifier on that class.
- Wrong because: caco2's label tops out at a log efflux ratio of 1.678 and logd's at 4.500, so every row lands in one class. RandomUnderSampler then raises on that single class before any metric is computed, and run_all records only an error for both datasets.
- Touches: No fed04 result file exists for caco2 or logd today — nothing was ever written, so no reported MCC, BAC, or AUROC needs correcting.
- Fix: Give caco2 and logd their own classification cutoff, or have fed04 refuse to run on a dataset with none defined, instead of defaulting to the hERG pKi threshold.
- FIXED 2026-09-04: `CLASSIFICATION_CUTOFFS` in fed04 holds a (cutoff, source) row per dataset — herg_ki at pKi 6.0, caco2 at log10(2) on the log10 efflux ratio (efflux ratio 2, the FDA in vitro drug interaction P-gp substrate call), logd at LogD(7.4) 3 (Hughes et al. 2008; Gleeson 2008). A dataset with no row is refused by name. A cutoff that puts every row in one class raises and prints the label range.

**`tests/experiments/federated/fed05_antibody_federated.py:67`** · needs a decision first  
fed05's dms silo can't load; the run finishes on two silos but reports three

- Computes: fed05 runs C5 pooled-binding and C2 transfer scores using three silos: ddg_combined, abagym_HER2, and flab_thermo_tresanco.
- Wrong because: abagym_HER2 was removed from the dataset registry, so loading it raises an error that gets caught and dropped silently, leaving two silos instead of three. Results still list all three silo names, so a two-silo run reads as the three-silo design later.
- Touches: C5's pooled-binding result and C2's dms transfer pairs, in any future fed05 run. No current result file is affected — the only fed05 outputs on disk are June smoke runs made before abagym_HER2 was removed.
- Fix: Register abagym_HER2 in the dataset registry with a group strategy, or swap in a different registered DMS dataset for the dms silo.

**`tests/experiments/opig_graphinity_ddg.py:714`** · needs a decision first  
within_foldx and within_rosetta are skipped every run, though the code under them works

- Computes: main() checks each condition's dataset name against a list of 31 registered dataset names, and skips the condition if the name is missing.
- Wrong because: graphinity_foldx and graphinity_rosetta are not in that list, so within_foldx and within_rosetta are skipped every run. evaluate_condition loads both datasets directly and never needs that list, so the check blocks a path that works.
- Touches: No result file exists yet for this experiment, so no reported number is wrong today. Once it runs, question 1 of OPIG B2 has no arm to answer it.
- Fix: Re-register graphinity_foldx and graphinity_rosetta, or drop the two conditions from the file — both datasets were cut as label-unfixable on 2026-07-09, so this needs her decision.
- **Fixed 2026-09-04:** user chose to delete the two conditions. within_foldx and within_rosetta are gone from ALL_CONDITIONS and CONDITION_DATASETS, with the reason recorded in the module docstring; --max-rosetta and the direct graphinity load are removed with them. The two submit scripts' --max-samples 5000 was corrected to --max-foldx 5000 in the same pass.

**`tests/experiments/_shared.py:3257`** · FIXED 2026-09-04  
f27 never builds the similarity recipe representations for its 'recipes' and 'both' pools

- FIXED: `load_data`'s molecular path splits the requested names into whole-dataset
  matrices and per-fold factories, builds the factories through
  `find_best_hybrid_molecular.get_rep_factories`, and raises if a requested one could not
  be built rather than returning a short list. f27 turns them into `outer_rep_builders`
  via `outer_rep_builders_from_factories`, so each sim_ recipe is refit once per outer
  fold on that fold's training rows and is scored on the outer test fold in the hybrid AND
  as a baseline. Passing them as `rep_factories` instead would have reached the inner
  search only, leaving them out of both outer arms without saying so.

- Computes: load_data builds the representation-array dictionary that f27 feeds into nested cross-validation, for whichever pool of representation names — general, recipes, or both — a run requests.
- Wrong because: load_data has no path to build the eleven similarity recipe representations; get_rep_factories, which builds them, is never called from here. So pool='recipes' always errors; pool='both' errors too if the general representations are already cached, or otherwise silently returns just those general representations mislabeled 'both'.
- Touches: Affects f27's three-pool comparison (general, recipes, both) for every molecular dataset. Any completed run reports 'recipes' as skipped and 'both' as skipped or a duplicate of 'general'.
- Fix: Add a path in load_data that calls get_rep_factories for the sim_* names and passes the result to run_nested_cv; until then, make pool='recipes'/'both' refuse outright instead of silently skipping or duplicating.

**`tests/experiments/opig_graphinity_ddg.py:231`** · re-run one experiment  
transfer_foldx's overlap filter never runs, so its Pearson r isn't a transfer measurement

- Computes: This condition trains on FoldX synthetic ddG rows and tests on experimental SKEMPI rows from the same antibody-antigen complexes, reporting a Pearson r.
- Wrong because: The filter meant to drop training rows sharing a complex with the test set never runs, because load_dataset drops pdb_ids. Because these representations ignore the mutation, every row from one complex shares one feature vector, so a test mutant is answered by its complex's average.
- Touches: The transfer_foldx Pearson r and its gap-to-EGNN line (opig_graphinity_ddg.py:592-598) measure same-complex fit, not transfer. This record doesn't say whether any file on disk already reports that number.
- Fix: Add data.get('pdb_ids') to load_dataset's return dict, upper-case PDB IDs on both sides before comparing, and make an empty filter print a warning or fail.

**`tests/experiments/similarity/s08_optimal_dimension.py:200`** · re-run one experiment  
s08 rebuilds every fold's hybrid at one fixed budget, not the budget being swept

- Computes: For each budget in the sweep (10 to 500 features), s08 rebuilds every fold's hybrid representation and scores it for both prediction and held-out similarity.
- Wrong because: Each fold is rebuilt using the best knn configuration across all budgets, not the configuration for the current step. So the held-out similarity score and the refit prediction score are both measured at that one configuration's budget, for every step in the sweep.
- Touches: The full-data similarity score and its best number of features are fine. But s08's comparison against the per-fold prediction score is wrong — that score never varies across the budget sweep. Disk status not stated.
- Fix: Use the configuration already picked for that step's budget (`_meta['best_config']['blueprint']`), or keep the current configuration but pass `budget_override=budget` when rebuilding each fold's hybrid.

---

## What was checked and holds

Read directly, not inferred:

- `_make_cv_splitter` raises rather than falling back to a random split when groups are missing
  or single-valued, and reduces the fold count rather than breaking groups when there are fewer
  groups than folds.
- Inside `nested_cv_evaluate`, the inner search sees the outer training rows only; the scaler and
  the imputer are fit on outer-train; both the hybrid and every single representation are scored
  on every outer fold, so the delta is paired.
- `nested_cv_outer_folds` and `nested_cv_cache_namespace` are the single definition shared by the
  precompute step and the evaluator, so precompute warms the entries the evaluator reads.
- Representation factories receive global row numbers restricted to outer-train, so a fold's
  features are not built from its own held-out rows.
- `nested_cv_evaluate` raises when structural pair indices do not align with the row order of
  `base_reps` and `labels`, instead of scoring mis-aligned pairs.
- `fold_reselection` applies the identical leave-one-fold-out tax to both arms and refuses to guess
  the metric direction, raising when a stored block contradicts `BUILTIN_METRICS`.
- `evaluate_cv` and `evaluate_cv_with_train` pass `groups[train_idx]` into the per-fold rebuild.
- The early-stopping split inside `DNNRegressor` and `PretrainedDNNRegressor` is group-aware.
  `mlp`, `cnn` and `bnn` are not — that is finding B.
- Cold-target grouping for drug-target affinity, PDB-complex grouping for binding-affinity change,
  and tissue-source-site grouping for TCGA are all built as the splitting document describes.

---

## Refuted

These were raised and did not survive. Recorded so they are not raised again.

- **`src/kirby/find_best_hybrid.py:144`** — Baseline cache key omits task_type and cv_splitter, both of which set the folds  
  Refuted: The two omissions are real as text, but neither can produce a wrong baseline from any entry point in the repo.
- **`src/kirby/find_best_hybrid.py:2504`** — tune_models silently falls back to a shuffled KFold when groups is None, where the engine's splitter raises  
  Refuted: Read src/kirby/find_best_hybrid.py:2487-2600 (tune_models in full) and :2867-2925 (_make_cv_splitter), plus every call site.
- **`src/kirby/find_best_hybrid.py:4452`** — A caller-supplied importance cache is reused without checking it came from this run's folds  
  Refuted: The code reading is accurate but the leak is not reachable from any real run.
- **`src/kirby/find_best_hybrid.py:4534`** — Precomputed allocation scores use a different inner split than create_hybrid now draws  
  Refuted: Line numbers first: the GroupShuffleSplit is at src/kirby/find_best_hybrid.py:4612-4616, not 4534, and create_hybrid's GroupKFold carve is at src/kirby/hybrid.py:4459-4463, not 4431 — the auditor quoted a different revision than the one on disk.
- **`src/kirby/find_best_hybrid.py:7217`** — The outer fold is always scored with default hyperparameters, so find_best_hybrid(tune=True) under nested CV reports an untuned number  
  Refuted: The auditor read the committed version, not the code in the working tree. `git show HEAD:src/kirby/find_best_hybrid.py | grep -n tuned_model` returns nothing; the working-tree file (M in git status) already threads the tuned estimator end to end, which is also
- **`src/kirby/similarity/mcs.py:132`** — get_mcs_pairs caches dy under a key that omits the labels it was computed from  
  Refuted: The key-omission is real but nothing reads the field, and the harm scenario the finding describes cannot occur as written.
- **`src/kirby/similarity/selection.py:97`** — select_saboteur_drop's held-out pair half shares molecules with the fit half, so it does not hold out what the docstring says  
  Refuted: The mechanical description is right and the consequence is already written into the code. selection.py:97-99 does permute pair indices, not molecules, so with 8000 uniform pairs over herg_ki's 1412 rows (compute_mcs_pairs samples endpoints with rs.randint over
- **`tests/experiments/_shared.py:577`** — --mode fast/intermediary silently replaces the outer nested CV with one 80/20 split, still reported with a std  
  Refuted: Read the whole chain. The mechanism is literally there — _shared.py:577 takes n_outer_folds from config['n_splits'], _shared.py:5308 sets it to 1 for any non-'full' mode, find_best_hybrid.py:2843-2846 turns n_folds==1 into one GroupShuffleSplit 80/20, nested_c
- **`tests/experiments/_shared.py:1200`** — cv_split decides stratification from label dtype where the framework decides from task_type; c09 has task_type and does not pass it  
  Refuted: The two rules cannot disagree on any dataset the repo can load, and the consequence the finding names does not exist even if they did.
- **`tests/experiments/_shared.py:1512`** — sequence_identity_clusters silently replaces collapsed clusters with sha256 hash buckets  
  Refuted: a random split wearing a groups label — The code the finding quotes is real and reads as described: _shared.py:1511-1519 replaces the cluster labels with int(sha256(seq),16) % max(3, n//5) when n_clusters < min_groups, prints nothing, and returns n_clusters =
- **`tests/experiments/_shared.py:1771`** — evaluate_cv_with_train drops a failed fold silently, so a cell's mean can be over one fold with nothing recording it  
  Refuted: I read the whole of both functions and the caller. The bare `except Exception: continue` is really there (tests/experiments/_shared.py:1771-1772, and the twin at 1689-1690), and the return tuple at 1786-1790 really carries no fold count. What does not hold is
- **`tests/experiments/complexity/c05_learning_curves.py:364`** — c05's sample-efficiency curve compares an argmax-over-150-to-600 arm against an argmax-over-10 arm, and the gap it measures grows as n shrinks  
  Refuted: I read c05_learning_curves.py in full, then the code it calls and the code that reads its output.
- **`tests/experiments/complexity/c10_noise_robustness.py:156`** — c10 measures noise robustness with a hybrid recipe chosen on the clean labels of every row  
  Refuted: The mechanism the auditor describes exists, but the harm they attribute to it runs the wrong way on the number c10 actually reports.
- **`tests/experiments/complexity/c32_allocation_strategy_cross.py:241`** — c32 sizes the candidate pool from whole-data scores when the inner CV cannot run  
  Refuted: The code the auditor quotes exists as written — `hib_whole` at c32_allocation_strategy_cross.py:233 is built from `rep_native[rn][0]`, which is the mean of that representation's five held-out fold scores (set at :195-:197 from `score_selection` over all folds)
- **`tests/experiments/complexity/c43_buchwald_step3.py:19`** — Both c43 files and exp_b import their splitter from a deleted per-session scratchpad, so the folds behind their numbers are not in the repo  
  Refuted: I opened all the cited code. The import path is real but the split claim built on top of it is not.
- **`tests/experiments/federated/fed03_synthetic_private.py:1000`** — fed03 Part C and fed02 choose each arm's recipe with a search over the whole dataset, then report 5-fold numbers on those same rows  
  Refuted: The shape is as described — fed03:995-1011 runs find_best_hybrid on the silo's complete reps/labels and keeps best_by_model[...]['blueprint'], and fed03:1085-1098 feeds that blueprint's allocation_method / importance_method / internal_evaluator into the per-fo
- **`tests/experiments/federated/fed03_synthetic_private.py:1207`** — fed03 Part C pools experimental and synthetic ΔΔG with offset group ids, so one PDB complex can sit in train and test at once  
  Refuted: The C2 pooled block cannot execute, and no number from it exists. It is gated at tests/experiments/federated/fed03_synthetic_private.py:1176 on `if 'ddg_exp' in silo_data and 'ddg_synth' in silo_data`. `silo_data['ddg_synth']` is filled at line 983 by `load_da
- **`tests/experiments/noise/_similarity_hybrid_space.py:149`** — n11/n13's hybrid_sim space picks its columns using every row, including each fold's validation rows  
  Refuted: The code fact is right; the contrast the finding is built on is not, and one of its three named leaks is arithmetically cancelled by the very line it quotes.
- **`tests/experiments/noise/n11_estimation_remediation_grid.py:426`** — n11/n12 use the clean-label blueprint, and the noise-searched lineage exists at levels they never sweep  
  Refuted: Read n11 lines 1-60, 369-460, 705-755; n12 lines 1-60, 300-380, 500-620; n15 lines 1-170; noise_helpers.py 45-110 and 340-400; _shared.py 3844-3980 and 4316-4500; dataset_policy.py 370-400.
- **`tests/experiments/noise/n13_noise_distributions.py:656`** — n13's entire Track A raises NameError on every call, so all 12 LNL methods record nothing  
  Refuted: The finding's single load-bearing premise is that `_seed_combined` "is never bound anywhere in the module (checked by AST over the whole file)". It is bound. tests/experiments/noise/n13_noise_distributions.py:532 reads `_seed_combined = 42 + 1000 * fi + noise_
- **`tests/experiments/similarity/s04_activity_cliffs.py:101`** — s04's cliff compounds and their cliff partners land in the same scaffold group, so the cliff arm never tests the cliff  
  Refuted: The code facts are as cited: s04_activity_cliffs.py:95-101 builds a global cliff_mask from identify_cliff_compounds (similarity_helpers.py:409-465, Tanimoto>=0.55 and |delta_y|>2*MAD) and freezes it as cliff_indices; :173 makes a GroupKFold via make_cv_splitte
- **`tests/experiments/similarity/s09_cross_dataset_summary.py:591`** — s09 reads s08's NaN-compared optima, not the NaN-skipped twins s08 added because those can name an unscored budget  
  Refuted: The NaN-argmax mechanism is described correctly, but the number it is claimed to corrupt is never produced. `optimal_dim_ratio` needs a kNN budget, and s08 has no kNN model.
- **`tests/experiments/similarity/s12_cross_domain_cliffs.py:310`** — s12 calls cv_split without task_type, so classification folds rely on the label-dtype heuristic  
  Refuted: s12:310 and :356 do call cv_split(groups, n_folds, sample_X, labels, mode=mode) with no task_type, and _should_stratify (_shared.py:1131-1137) then falls back to "labels are integer-dtype". The fallback only diverges from the authoritative path when a classifi
- **`tests/experiments/similarity_metrics/feature_type_study.py:141`** — feature_type_study's H3 compares a searched subset with a fixed one on the pairs the search used  
  Refuted: The code description is accurate: greedy_forward (feature_type_study.py:141-155) maximises score_subset over the same SD/truth/ham built once at :160, and run reports greedy_v - topk_v at :217 and peak_n at :188 off that same 8000-pair draw. No pairs are held
- **`tests/experiments/similarity_metrics/nary_similarity.py:190`** — nary_similarity grades roughness against a target built from the same distances and the same labels  
  Refuted: The arithmetic half of the claim is right; the "large" half is not, because the number it attacks was already written down as mechanical and the avenue closed as a negative.
- **`tests/experiments/similarity_metrics/winning_recipe_vs_standard.py:166`** — adme_solubility's scaffold groups are near one-per-molecule, so its read-across GroupKFold is close to a random split  
  Refuted: The counts are right and everything built on them is wrong. I reloaded sim_adme_solubility.npz this session: n=2173, 1885 distinct groups, 1721 singletons, largest 26 — exactly what the auditor reports (herg_ki 1415/751/549/33, logd 5039/2245/1421/118 also mat
---

## Not covered

The completeness reader that was meant to close this list died twice on service errors, so this
section is what I know was left out, not a systematic answer.

- `tests/experiments/framework/` — excluded on your instruction.
- `tests/experiments/annitia/`, `cross_species/`, `pretrain_rep/`, `feature_selectors/`, and the
  loose comparison scripts at the top of `tests/experiments/` (`dta_e2e_comparison.py`,
  `molecular_finetune_comparison.py`, `qm9_3d_comparison.py`, `tcga_foundation_comparison.py`,
  `wsi_abmil_comparison.py`, `structural_geometry_survey.py`, `covariate_shift.py`) — outside the
  five suites named.
- `src/kirby/representations/`, `src/kirby/models/` beyond the early-stopping split, `src/kirby/uq/`
  beyond the conformal finding, and `src/kirby/finetuning/`.
- Whether any pretrained or fine-tuned representation was trained on molecules that sit in the test
  folds. No reader was assigned this and it is a real leakage path for `chemberta`, `molformer`,
  `selformer`, `unimol` and the protein language models.
- Duplicate rows: whether the same molecule or the same antibody appears twice in a dataset under
  different identifiers, which would put it on both sides of any split. Not checked anywhere.
- Fold balance: group vectors that are legitimate but produce grossly unequal fold sizes.
- The `.db` result files and cache directories on the server were not inspected; every cache
  finding here is from the key-construction code, not from what is actually stored.
