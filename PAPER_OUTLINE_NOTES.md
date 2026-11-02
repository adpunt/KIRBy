# Notes behind PAPER_OUTLINES.md

What this file is: the facts the outlines are built on, each with where it was read.
Read this before changing `PAPER_OUTLINES.md`. Nothing here is a result; it is a record
of what the thesis, the code and the result folders say, and where.

Written 2026-10-29.

---

## 0. Chapter numbers

`thesis.tex` chapters in order: 1 Introduction, 2 QSAR modelling and label noise,
3 Noise robustness across molecular representations and model architectures,
4 Limitations of the similar property principle, 5 Building adaptive hybrid
representations, 6 Similarity in an adaptive hybrid representation, 7 Detecting and
handling label noise, 8 Conclusions and further thoughts.

Paper 2 draws on Chapter 5 (`\label{chap:method}`, thesis.tex:1334-2779).
Paper 3 draws on Chapter 7 (`\label{chap:noise}`, thesis.tex:3473-4995).
The old outline said "Chapter 4" for the hybrid chapter. That was wrong.

---

## 1. The hybrid search as the code and Chapter 5 define it

Input (thesis.tex:1432): K frozen feature matrices on the same n samples, a label
vector, one group label per sample for cross-validation, the metric, the task type,
one or more downstream models, a list of budgets.

Six searched settings (thesis.tex:1496):

| Setting | Values | Line |
|---|---|---|
| Downstream model | RF, XGBoost, LightGBM, SVM, DNN, GP; two per dataset | 1502 |
| Budget (max number of features) | 10, 25, 50, 100, 250, 500, 1,000, 2,000 | 1504 |
| Representation count | 2, 3, 5, all | 1508 |
| Importance method | LightGBM gain, TreeSHAP, DeepLIFT | 1524 |
| Allocation method | linear, squared, softmax, rank-decay, global pool | 1532 |
| Internal evaluator | RF, DNN, LightGBM, matched | 1558 |

Full grid = 10,080 configurations with all six downstream models, 8,160 without the GP
(thesis.tex:1572). The searches reported in Chapter 5 scored 108 to 3,360 configurations
per outer fold, because each dataset's grid was cut by hand (thesis.tex:1572, 1617).

Three levels of splitting, every split keeping each group whole (thesis.tex:1440,
figure at 1443): outer test fold; inner test fold that scores each setting; inside the
inner training part, 80% computes importance scores and 20% computes the standalone
score of each single representation, which sets the budget split.

Budget split (thesis.tex:1457): each representation's standalone score is rescaled so 0
means no better than a trivial predictor and 1 means perfect, per metric; the allocation
method turns that into a weight; shares are rounded by largest remainder and capped at
the representation's width. Features are ranked only inside their own representation.
The global pool is the one exception: it ranks features from different representations
against each other (thesis.tex:1536, 1544).

The budget is a maximum, not a target; a hybrid can be narrower (thesis.tex:1432, 1461).

Blueprint (thesis.tex:1469): records the settings, not the features. The explainability
layer runs after a hybrid is built and names each selected feature where it can: ECFP4
bit by the atom environment that sets it, MACCS key by SMARTS pattern, descriptors by
name, protein composition / k-mer / one-hot / BLOSUM / structural / inverse-folding /
PCA components also named. MAP4 and DRFP get a bit index only. Avalon, Mordred,
atom-pair, graph-kernel and learned-embedding features get only the representation name
and a dimension index.

Greedy forward allocation was tried and cut (thesis.tex:1538): blocks of 10 features at
budget 500 over 10 representations needs up to 500 model fits per configuration and fold
against 10 for a score-based method; it placed 16th of 17 allocation methods on Jain
FcRn retention, 0.360 against 0.453 Spearman for linear allocation.

Augmentation features (thesis.tex:1586): donor feature sets added beside the base
hybrid. In an earlier search of 16 datasets neither augmented condition beat no
augmentation by more than 1% on any dataset, or 3% under about 1,000 samples, so
Chapter 5 ran without it. Only the kNN search of Chapter 6 keeps it, on SMAC colloidal
stability and HIC retention.

Adaptive pruning exists and was not used (thesis.tex:1601).

Search cost: summed over five folds, up to 28.2 hours on a small-molecule dataset
(thesis.tex:5043).

---

## 2. Chapter 5 results, with the numbers

Every number below is a held-out score in the dataset's own metric, mean over five outer
folds unless the thesis says otherwise.

**Against the best single representation** (table at thesis.tex:2113). Hybrid better on
every small-molecule dataset with both models, and on every fold of the six regression
datasets. Examples: LogD with the DNN 0.4497 to 0.4089 RMSE; QM9 HOMO-LUMO gap with the
DNN 0.3619 to 0.3155; Caco-2 with the DNN 0.3042 to 0.2775. It also beats the single
representation that was best on the held-out molecules, which no user could pick in
advance (thesis.tex:2107).

Antibodies, same table: ahead on AC-SINS self-association, angiopoietin-2 binding,
anti-VEGF scan binding, anti-VEGF scan expression, binding free energy, EGFR binding,
expression titer; behind or level on CHO polyreactivity, HIC retention, nanobody
thermostability, SMAC colloidal stability, antibody humanness. Largest gain in the
chapter: binding free energy with the DNN, 0.1317 to 0.3542 Spearman, but it moves by
more than its own size from fold to fold (thesis.tex:2184).

Buchwald-Hartwig: RF 14.4056 to 15.2383 RMSE (worse), LightGBM 16.1679 to 15.6211
(better), both by less than the spread across folds (thesis.tex:2196). The search drew on
only two of the ten representations on every fold.

**Ensembles** (table at thesis.tex:2203). An ensemble of one model per representation
beats the hybrid on most antibody datasets with both models. On small molecules the
same-model ensemble beats the hybrid only on hERG blocking and Buchwald-Hartwig. The
hybrid is ahead on every fold of QM9 with the DNN and hERG Ki with the SVM.

**Top representation alone, and an even split** (table at thesis.tex:2290). At a fixed
budget the top representation alone scores worse than the search's hybrid at every budget
with both models on most datasets. An even split over the same representations comes
close to the search's split on small molecules; on antibodies the split matters more.

**Rare scaffolds** (table at thesis.tex:2389). About 2,000 molecules of each dataset,
each half searched separately. The hybrid's gain is larger on the rare half than the
frequent half with both models on LogD, and with the DNN on Caco-2 and QM9.

**PCA** (thesis.tex:2420): hybrid clearly ahead with SVM, LightGBM, XGBoost, close with
the DNN. PCA ahead on LogD with the DNN, 0.392 against 0.409 RMSE.

**Overfitting of the search** (table at thesis.tex:2440). Search's own score against the
held-out score of the same hybrid, with the smallest training set of any fold. Small
datasets are where the gap is large: EGFR binding with the DNN, 0.371 search against
0.152 held out, 517 training samples; expression titer with the DNN, 0.426 against 0.294,
186 training samples. QM9 with the DNN, 0.322 against 0.331, 7,982 training samples;
hERG blocking with LightGBM 0.862 against 0.863, 6,112 training samples.

**One fold fewer** (table at thesis.tex:2458): leaving one more fold of training samples
out of each fold's search makes the held-out score worse on every small-molecule dataset
with a result, with both models. The two searches differ in two things at once, the
training samples and the budget each picks.

Same table gives the stability index of Nogueira2018 across the five folds: QM9 LightGBM
0.577 and hERG Ki with the DNN 0.273 at the top and bottom of the small molecules;
antibody and Buchwald-Hartwig hybrids lower, e.g. HIC retention with the GP 0.063.

**Selection at a fixed number of features** (thesis.tex:2483): the hybrid beats recursive
feature elimination and random features at every budget on QM9, Caco-2, LogD and
MDR1-MDCK efflux. Exceptions: hERG Ki with the DNN (random features ahead at 500, 1,000
and 2,000); Buchwald-Hartwig (recursive elimination ahead at most budgets); CHO
polyreactivity (recursive elimination ahead at every budget). mRMR ran only at 100
features or fewer.

**What moves the score: model or budget** (table at thesis.tex:2518, eta squared in
percent, two-way ANOVA over the configurations of a search, from
`tests/experiments/complexity/c09_formal_analysis.py`). Downstream model / budget /
interaction / residual: QM9 84.1 / 1.0 / 0.2 / 14.8; LogD 48.5 / 1.4 / 1.6 / 48.5;
Caco-2 0.0 / 6.6 / 43.6 / 49.9; hERG blocking 14.3 / 51.2 / 0.0 / 34.5; MDR1 2.0 / 3.0 /
4.0 / 90.9. The residual is the other four settings of the grid.

**Budget against model capacity** (thesis.tex:2497): with the SVM's C at 0.01 or 0.1 the
search picks the same budget on every fold of LogD, rat microsomal clearance, MDR1-MDCK
efflux and Caco-2; at C = 10 it picks larger budgets on LogD and rat microsomal
clearance. The size of the DNN's hidden layers does not move its pick the same way.

**Internal evaluator** (thesis.tex:2567): on most small-molecule datasets the choice
makes almost no difference; the exception is QM9 with LightGBM, 0.363 RMSE with an SVM
evaluator against 0.346 for the matched one. On EGFR binding, HIC retention, expression
titer and SMAC colloidal stability the matched evaluator is best with both models; EGFR
binding with the SVM, 0.353 Spearman matched against 0.299 with a LightGBM evaluator.

**Importance scores reused during training** (table at thesis.tex:2578): the best of four
conditions gains little over the plain hybrid on any small-molecule dataset, and feature
scaling makes LogD with the SVM worse, 0.430 to 0.462 RMSE.

**Conformal intervals** (thesis.tex:2641): 90% intervals cover close to 90% of held-out
molecules on small molecules, and are narrower than the single representation's. On
several small antibody datasets they cover too few.

**Blueprints** (thesis.tex:2663): LogD's SVM hybrids share 372 features on all five outer
folds, 38 Mordred descriptors, 36 ECFP4 bits, the rest from eight representations whose
features carry no names. Among them SLogP, nBase, PEOE_VSA surface areas, secondary
amine and aromatic nitrogen counts; ECFP4 tertiary amine (CN(C)C) and aryl chloride
environments. Lipophilicity and basicity, which is what logD at pH 7.4 needs.

**Extension studies** (thesis.tex:2677): three activity-cliff targets of Dablander2023,
hybrid lower MAE for every target and model except factor Xa with LightGBM. Shared rows
on hERG Ki: graph kernel alone 0.611 RMSE with 105 molecules sharing a row; with ECFP4
and MHG-GNN 0.571 and 0 sharing; with ECFP4 and PDV 0.575 and 56 sharing, so most of the
gain is the added information, not the broken ties. Settings found on LogD reused on
hERG Ki beat hERG Ki's best single representation with both models but stay behind a
search run on hERG Ki itself (0.567 against 0.563 with the SVM).

**Separate sites** (thesis.tex:2739): each site holds one family of representations of
the same molecules. One selection per site does as well as one selection over all
families, on Caco-2 and LogD with both models, and both beat the best family. LogD with
the SVM: 0.43822 per family, 0.43823 all families, 0.460 best family, 0.493 best single
representation.

**Chapter 5's own limitation paragraph** (thesis.tex:2777): the hybrid needs data; gains
are consistent on a few thousand samples and not on a few hundred, where the search's own
scores overstate its results; most antibody datasets are that small, "so their results
say as much about size as about antibodies".

---

## 3. Chapter 7, and what the old Paper 3 takeaways got wrong

Scope: clipped labels are one damage kind of six (table at thesis.tex:3516) and one
results subsection, `sec:noise:res:censoring` at thesis.tex:4373. The six kinds are
Gaussian, outlier, grouped-shifted, censoring, shift in 10%, and shift in 10% sized by
model error. Classification uses five distinct flip rules on a two-class dataset
(thesis.tex:3536).

QM9's role (thesis.tex:3777, 3797): labels are computed, so they are clean. A method that
removes noise should do nothing on QM9 or make it worse. A gain there is a better-trained
model. It is a control described in the methods and used to sort methods in the results.
Not a finding.

Datasets: all of Table `tab:datasets` except seven, plus hERG IC50, the only dataset with
real repeat measurements (thesis.tex:3488).

What was actually measured, by question:

**Detection** (thesis.tex:4163). The Forest residual is well above chance under outliers,
shifted labels and flipped classes on every dataset: Caco-2 under outliers, average
precision 0.338 against a chance level of 0.051. Under censoring it is near or below
chance on the small molecules and reactions, because a model fitted to clipped labels
predicts near the limit.

Cliff-aware gap (thesis.tex:4170): under the shift in 10% on Caco-2 its median neighbor
setting reaches 0.269 average precision against 0.202 for the Forest residual. Under
outliers its best setting is ahead and its median setting behind. Under censoring it
trails on Caco-2, hERG Ki and LogD at every setting.

Hybrid features for detection (thesis.tex:4177): noisy labels were NOT easier to find in
the hybrid's features. Four families compared (plain features, importance-weighted
features, forest proximity, the hybrid's own features); on Caco-2 under outliers all four
reach 0.335 to 0.352 average precision. Importance-weighted features best on most
datasets; the hybrid's own features best on no dataset overall. Averaged over detector
settings instead of best setting, the hybrid comes out ahead more often, so the ranking
depends on how the setting is chosen.

Break-even precision (thesis.tex:4200): the share of truly damaged molecules a flagged
set needs before acting on it helps. Caco-2 under the shift sized by model error, the top
5% by LightGBM residual has precision 0.171 while deleting them pays off only above
0.616.

**Acting on the flags** (thesis.tex:3822). hERG blocking with 10% of labels flipped along
a confusion matrix: every action helped, deletion most, and deletion beat deleting the
same number at random (hybrid 0.838 AUROC against 0.818 for the best of five random
deletions). On LogD with 5% outliers and the DNN the actions differ less and the result
against random deletion is mixed.

**Clipped labels** (thesis.tex:4373). Censored EM lowers the RMSE for every model and
configuration with a quarter of the labels clipped, not back to the level before
clipping; LogD, DNN, best single representation wins back about three quarters of the
lost RMSE. No one lift is best everywhere: censored EM best on LogD, QM9, AC-SINS;
constant lift best on hERG Ki and MDR1-MDCK efflux. The size of each molecule's own lift
does not matter: shuffling the same lifts among the clipped molecules beats the own lift
in every configuration on LogD and Caco-2. Deleting the clipped molecules is worse than
keeping their clipped labels on Caco-2, hERG Ki, QM9 and MDR1-MDCK efflux. On three
antibody datasets clipping did no harm, so there was nothing to fix.

**Labels as delivered** (thesis.tex:4577). Strongest controlled results are on antibody
datasets: SMAC colloidal stability, deleting the 20% flagged by the forest residual
raises the SVM's first single representation from 0.274 to 0.403 Spearman, and every
shuffle of the detector's scores does worse; the held-out fold has 53 antibodies. Binding
free energy and expression titer also pass in all four of their configurations. On small
molecules, two neighbor-mean smoothing gains pass: the DNN on hERG Ki (0.638 to 0.578
RMSE on the first single representation) and XGBoost on hERG blocking. Buchwald-Hartwig
is the one dataset where deleting flagged reactions beats random deletion.

**Cleaning the labels the search uses** (thesis.tex:4661). Four combinations: the labels
the search ranks features with, the labels the final model is fitted on, both, or
neither. Cleaning both does best in about half the rows. Expression titer as delivered,
delete the flagged 20%: DNN on the hybrid 0.155 no cleaning, 0.390 fit only, 0.562 both,
0.423 search only. SMAC as delivered, score-scaled shrink for the search only: -0.014 to
0.560 Spearman on a held-out fold of 53 antibodies. Cleaning before the search also hurt
in places. No random-flag control for this experiment (thesis.tex:4671).

**Hybrid against single representation under noise** (thesis.tex:4950). Three different
questions, which the old takeaway ran together:
  - prediction: on small molecules the hybrid beats the best single representation in
    almost every condition before and after treatment (LogD with outliers, DNN, neighbor
    weights: 0.556 against 0.591 RMSE). On antibodies ahead in about half the conditions
    either way, on the reaction set level.
  - detection: the hybrid's features were best on no dataset overall (above).
  - which method wins: the method that works best with hybrids is rarely the one that
    works best on single representations; three exceptions on the first hybrid.
  - neighbor space for weighting: neither reliably better, largest differences both
    ways, no control.

**Changing the model instead of the labels** (thesis.tex:4753). Seed averaging takes the
DNN's second hybrid on hERG Ki from 0.718 to 0.656 RMSE under Gaussian noise at level
1.0, as much as the best cleaning, with no detector and no label changed. But stacking
and seed averaging also help QM9 with no noise added, so most of their gain is better
training.

**Cliff put-back** (thesis.tex:4155, methods at 3660). A training molecule counts as a
cliff if a partner at ECFP4 Tanimoto 0.55 or more has a label further than twice the
median absolute deviation away. On Caco-2, hERG blocking and LogD the random control
cannot be run at all, because more than half of the training molecules count as cliff
molecules. On QM9 only 19 of 7,982 do. The thesis states that no other representation was
tested for this step and that the count of cliffs depends on the representation.

**Fold count** (thesis.tex:3791, 4995): most results are the first outer fold only.

---

## 4. Known holes

- `tests/experiments/complexity/` holds 22 experiments with result files under
  `/Volumes/seagate/kirby_results_2026-10-10b/complexity/`, 15 to 28 datasets each.
  Only four are cited anywhere in `thesis.tex`: c04, c06, c09, c23 (grep for
  `complexity/c` in thesis.tex). The rest are unreported.
- `c12_modality_ablation` has one result file and it is patient data, so it is out.
- No result folder for c20, c31, c32.

---

## 5. The cliff definition, as the code has it

`identify_cliff_pairs` in `tests/experiments/noise/n01_activity_cliff_framework.py` is
the one reference. A pair is a cliff when the two molecules are similar AND their labels
differ. Similar means ECFP4 Tanimoto at or above 0.55, or inside the top percentile of
pairs; for antibodies, edit distance on the heavy chains; for reactions, Tanimoto on
reactant and product Morgan bits. A molecule is a cliff molecule when it is in at least
one such pair. `n25_cliff_exemption.py` imports it rather than redefining it.

Two things in the code already say the reference is a choice, not a fact:

- Every n01 result records `cliff_reference_independent`. When the cliffs come from the
  same features being graded (the Euclidean fallback), the file says the comparison is
  circular and must not be reported as one
  (`n01_activity_cliff_framework.py:1-30`).
- `n25_cliff_exemption.py:693` holds a TODO that names the two open decisions for moving
  the cliff set onto the similarity axis: which threshold means "similar" in a Manhattan
  or forest distance, a percentile of pairs or "among the k nearest"; and whether the
  exempt set may come from the same features the detector reads, which that file's
  docstring calls contamination. `n01_activity_cliff_framework.py:590` holds the same
  question for the k-nearest translation.
- `n42_similarity_noise_sweep.py:90`: the cliff set is ECFP4 Tanimoto, not on the
  similarity axis.

So redefining a cliff in a hybrid distance is blocked on those two decisions, both of
which belong to the author.

---

## 6. The audit page `~/Documents/ch7_noise_methods_audit.html`

Built 2026-10-22 from `/Volumes/seagate/kirby_results_2026-10-10b`, from the raw result
files rather than from thesis text. One row is one Chapter 7 method, with every
experiment folder that ran it merged into that row. "Raised the top" means the best
treated held-out score beat the best untreated score, across every model and
representation, on the same damaged training labels.

Sections: `#s0` detection (16 method blocks), `#s1` mitigation (38), `#s2` noise-robust
training (25), `#s3` folders with no Chapter 7 method, `#s4` disagreements between the
two checking passes, `#s5` code against the published code, `#s6` the two lists.

### The two lists, `#s6`

List 1, "raised the top score under at least one kind of damaged label other than
censoring": 35 entries, 16 cleaning, 4 selection, 15 training.

List 2, "raised the top score only under censoring": 5 entries. Constant lift,
truncated-normal mean, censored EM, own lift, delete clipped. The page states that every
method in List 2 was run under censoring only, so "only under censoring" means no other
kind of damaged label was tried for it, not that it was tried and lost.

So List 2 is one seventh the size of List 1, and its five entries are the record of what
was tried, not a finding about clipping.

Taken off List 2: the XGBoost accelerated-failure-time interval model. The one dataset
where it raised the top is not a Chapter 7 dataset, and there it also raised the top with
no censoring.

Open decision recorded in `#s6`: source offset and source weights raised the top on
datasets with simulated per-source offsets, which is not one of the listed noise kinds
and not censoring; whether that counts as a kind of damaged label for List 1 is a
decision, not a measurement.

### The random-removal yardstick (`#m52`)

Keeping a random 90% of training molecules, best of 20 draws, raised the top score in 30
of 35 dataset-folds with no noise added. The page calls it "the yardstick for any 'best
of many settings' result". Related, `#m16`: "Gains with chance-level detectors come from
picking the best of about 62,000 fits, not from detection."

### Code against the published code, `#s5`

Over the 79 method blocks: 38 published code found and hers differs, 12 hers calls the
published code, 8 no code by those authors found, 21 no published method to compare
against. Judgement: 11 needs fixing, 49 a deliberate change, 11 moved to another task,
8 settings only.

The 11 needing fixing: cliff-aware gap, distance families for neighbors, close-pair
weights, neighbor-agreement weights, cliff put-back, confident learning, uncertainty
curriculum, DivideMix, MentorNet, contrast-to-divide, heteroscedastic GP (the LNL one).
Four plus MentorNet's duplicated input were fixed on 2026-10-26 and 2026-10-29; the other
seven are the author's decisions, in
`thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md`.

For a method on that list, a result reported under the published name is not that method.

### Stored results the page says are superseded

- Neighbor-agreement weights (`#m27`): every stored n02 and n42 result for the
  cliff-aware weight is on the old footing. OPEN_THREADS A10.
- Confident learning (`#m42`): every stored n09 and n13 result is on the old footing.
  OPEN_THREADS A11.
- Distance families (`#m15`): the 2026-10-29 SMILES fix, OPEN_THREADS A9.
- The cliff-aware gap and cliff put-back fixes move no number.

### The three cliff methods, as the audit has them

- Cliff-aware gap (`#m5`): finds shifts best, plus outliers and class flips, 3 of 4 noise
  kinds graded, but not censoring. No published code implements this score; the nearest
  is RegENN, which compares the gap against one spread times a fixed multiplier and
  deletes the molecule, while hers returns a score for every molecule and deletes
  nothing. The two cut-offs that set the cliff flag are fixed in the code and never
  fitted from the data.
- Neighbor-agreement weights (`#m27`): "The two arms this file compares differ in two
  ways at once, the arithmetic and the lean, so the difference between them cannot be
  read as cliff-awareness." The cliff-aware call that left the strength at its default
  came from a 2026-03-25 commit with no reason, so the mismatch was accidental. Fixed
  2026-10-29, and every stored result predates the fix.
- Cliff put-back (`#m28`): beat plain cleaning under noise on 2 of 4 small-molecule
  datasets, lost on the reaction dataset and QM9; lost under outliers and class flips.
  The reason is in the evidence: the cliff set covers most of each training set,
  1,740 to 3,056 of Buchwald-Hartwig's 3,057 molecules, and every corrupted
  Buchwald-Hartwig molecule is a cliff molecule, so putting cliffs back restores the
  corrupted labels. The random put-back is not size-matched, because there are not
  enough non-cliff molecules to draw from.

### Methods the page records as not tied to noise

Score down-weight, neighbor weights, exponential down-weight, retuning, Meta-Weight-Net,
Huber loss, Yeo-Johnson target transform, the kNN-mean predictor, and cleaning before the
hybrid search ("part of the gain is just rerunning the search"). Each raised QM9 with
clean labels, or gained as often with no noise as with noise.

### Detectors the page records as not working across noise kinds

Forest residual (misses censoring above 0.1 and sparse shifts), LightGBM residual (below
chance under censoring on 4 of 7 datasets), cliff-aware gap (below chance under
censoring), heteroscedastic GP spread (finds no damaged labels, 0 of 3 kinds graded),
variance-head network, NGBoost spread (at chance on almost every kind), cleanlab score,
conformal p-values with Benjamini-Hochberg flags.

The one detector above chance under censoring on every dataset tested, 7 of 7, is the
nearest-neighbor Shapley value (`#m10`). Buchwald-Hartwig 0.391 average precision against
a chance level of 0.250, QM9 0.394 against 0.250. It was never used to clean. Under
censoring forest proximity is its best neighbor space; under shifts, hybrid feature
spaces are.

---

## 7. What the search code is, and what is chemistry in it

Read in full: `src/kirby/find_best_hybrid.py` (11,633 lines, FBH below) and
`src/kirby/hybrid.py` (6,641 lines, H below).

**No chemistry.** Neither file imports RDKit, parses a SMILES, builds a fingerprint or
computes a scaffold. The module docstring states the contract at FBH:1-12: pre-computed
numpy matrices, the caller supplies the representations and the group labels. A
configuration is the tuple at FBH:6869-6885.

Three places carry chemical wording and generic code:
- `TanimotoKernel` (FBH:2679) is Jaccard on any binary vector, used only in the GP
  hyperparameter grid.
- `objective='struct_fid'` (FBH:2337) takes an arbitrary pairwise distance array and
  Spearman-correlates it against Euclidean distance in the selected features. The one
  chemical quantity read by name is an optional atom count, and it feeds a number that is
  reported and never selected on.
- `objective='similarity'` (FBH:2131) needs the group array and nothing else. "Scaffold"
  is only that array's name.

Three real constraints on a new domain, all from the code:
- Group labels are required. `_make_cv_splitter` (FBH:4074-4128) refuses to build a split
  with no groups or one group, at both branches. There is no random-split fallback.
- `task_type` must be given as `'classification'` or `'regression'` (FBH:8798).
- `struct_fid` needs a pairwise distance supplied as index arrays plus values.

**What is implemented against what is searched.** Importance methods implemented:
permutation, treeshap, integrated_gradients, kernelshap, xgboost_gain, lightgbm_gain,
deeplift, gradientshap (H:4602-5350). Default searched: deeplift, lightgbm_gain, treeshap
(FBH:3713). Allocation registry `NAMED_ALLOCATIONS` holds about 45 names over five base
methods (H:517-610); default searched: performance_weighted, power_2, softmax_T1,
rank_exp_d0.3, wglobal_power2 (FBH:3722). The whole greedy family is marked cut and
nothing selects it (H:488-515). Internal evaluators accepted: rf, rf_exact,
lightgbm_exact, xgboost, lightgbm, mlp, svm, gp, knn, catboost, dnn_small, dnn_wide, dnn,
cnn, ngboost, bnn, any callable, plus `'match'` (H:2735-3238); default searched: match,
rf, lightgbm, dnn.

**Importance is used to rank and select, nothing else.** Every consumption site is an
argsort take (H:5436, H:2665, H:2431, H:2468, H:3974, H:3340, H:6334, H:6404, FBH:4328).
No `sample_weight` is passed on any fit path in either file. `FittedResult.feature_weights`
is computed only when `weight_method` is set, which defaults to None, and even then it is
attached to the returned object and never used to scale features or train anything.

**Provenance of a feature** is the pair (representation name, integer column index into
that representation's own matrix), carried in `feature_info` (H:5443-5454) and replayed
on new rows by `apply_feature_selection` (H:5485). Neither file holds a feature name, and
`feature_blacklist` says so outright: "the matrices carry no names — the CALLER resolves
names -> indices" (FBH:8457). Naming is entirely the caller's. What can be named is
listed in the thesis at thesis.tex:1474.

**The budget is a maximum.** `concatenate_features` (H:5400-5436) stops a representation
at its last feature with importance strictly above zero and records `n_requested`,
`n_scored_above_zero`, `n_unfilled`.

**Row-count caps that matter for a large dataset**: SVM skipped above 10,000 rows
(FBH:2850), GP above 1,000 (FBH:2909), rf_l1 above 5,000 (FBH:2985); the SVM under
uncapped augmentation above 5,000 (FBH:3121). Internal svm, mlp and gp evaluators fit on
at most 500 rows and predict on the whole held-out block (H:2690). A representation wider
than 5,000 features is scored as a baseline on its top 500 features, or 200 under 1,000
samples (FBH:4597).

**Cost recorded**: `time_seconds` per configuration (FBH:4836), stored as a row column.
Nothing else per configuration.

---

## 8. Dataset inventory

`src/kirby/registry.py:55` ships an empty registry. The one table that fills it is
`DATASET_REGISTRY_DATA` in `tests/experiments/dataset_policy.py:230-343`, with 75
entries. `tests/experiments/decided_grid.json` holds 48 dataset keys. The thesis dataset
table (thesis.tex:1711-1754) has 20 rows.

Allowed domain strings (`registry.py:30-31`): molecular, antibody, dta, tcga, reaction,
nucleotide. Counts: antibody 35, nucleotide 21, molecular 10, dta 4, reaction 3, patient
data 2 (excluded).

**Small molecule, 10 registry entries.** In the thesis: qm9 (9,978), herg_ki (1,412),
caco2 (2,158), logd (5,028), adme_rlm (3,053), adme_mdr1 (2,641), herg (7,772, AUROC).
Not in the thesis: adme_hlm, lhasa_herg, chembl_qsar with three targets (d2 6,333, fxa
3,605, mpro 1,924). Plus `herg_ic50`, which is in no registry but has a decided grid row,
is in Chapter 7, and is the only dataset with real repeat measurements.

**Antibody, 35 registry entries, 12 in the thesis.** Not in the thesis, with a decided
grid row: abcov_kd, flab_binding_adams (11,052 variants, 7,247 clamped at the Tite-Seq
ceiling), flab_expression_adams, abagym_VEGF, cdr_flexibility, tz_her2_binary,
tz_her2_multiclass. Not in the thesis, no grid row: eight FLAb inter-family keys,
flab_binding_phillips_h3 (32,767 variants, 28,965 on the SPR floor), abagym_HER2 (146),
abagym_lysozyme (627), humanness_species, ginkgo_heparin (94), ginkgo_tm2, ginkgo_purity,
ginkgo_aggregation. `dataset_policy.py:288` records that the last three were missing, the
search raised on them, the exception was swallowed, and the results file was written with
no hybrid rows.

**Nucleotide, 21 registry entries, none in the thesis.** All classification, Matthews
correlation coefficient. Nucleotide Transformer Revised histone marks, promoters,
enhancers and splice sites, 5,274 to 33,000 sequences each, plus four Genomic Benchmarks
tasks, 27,791 to 100,000. Rows grouped by chromosome
(`src/kirby/datasets/nucleotide.py:57`). Six keys have a decided grid row:
nt_H3K27me3, nt_H3K4me3, nt_H3K9ac, nt_H3K9me3, nt_promoter_tata, gb_human_enhancers_cohn,
all at speed fast with three allocation methods and lightgbm_gain as the only importance
method, the narrowest grids in the file. Nucleotide is the one domain where ridge is kept
(`dataset_policy.py:26-37`, decision of 2026-07-18); knn and xgboost are dropped for the
whole domain, and mlp for both augmentation conditions.

What ran: `tests/results/nucleotide_survey/`, 22 files dated 17-18 April 2026. Five outer
folds, seed 42, five downstream models, 17 single representations (six k-mer and
composition sets, six DNA language models). The result shape is single representations
plus pairwise augmentations plus a summary. **No hybrid search rows.** A k-mer count is
the best single representation on all 22; no DNA language model is best on any. Examples
in MCC: nt_H3K4me3 kmer4 0.6474, nt_H3K9me3 kmer4 0.2119, gb_demo_human_or_worm
tetranuc_freq 0.9135, gb_human_enhancers_cohn kmer6 0.4777.

**Reaction, 3 keys.** buchwald_hartwig (about 3,955 reactions, grouped by additive, 22
groups in the thesis table and in `reaction_yields.py:385`, 23 in the loader docstring at
:100 — unresolved), in the thesis. bh_doyle_plates (about 4,599 reactions, 23 additives,
a different plate run from the same lab, not a reprocessing) and suzuki (about 5,760
reactions, 11 ligand groups), both with decided grid rows, both run only in
`phase5_baselines/reaction/` and `forward_compat/exp7_reactiont5/`. A fourth loader
function, `load_eln_buchwald_hartwig`, has no key.

**Drug-target affinity, 4 keys, none in the thesis**: davis, kiba, bindingdb_kd (about
52K pairs), metz (about 35K pairs), all scored by concordance index.

**Outside chemistry, loaders exist with no registry entry**, because none of the six
allowed domain strings fits: six Matbench tasks (`matbench.py:32`), five multi-view sets
(`multiview.py:81`: mfeat, caltech101_7, caltech101_20, nuswide_object, reuters), 217
ProteinGym assays (`proteingym.py:122`). The five multi-view sets have decided grid rows
added 2026-10-23 with accuracy as the metric, Matbench and ProteinGym do not.

**Two mismatches to state plainly.** Six keys in `decided_grid.json` have no registry
entry: herg_ic50, mfeat, caltech101_7, caltech101_20, nuswide_object, reuters. 33
registry entries have no decided grid row, and c11 refuses a dataset without one.

### NEW_DATASETS.md, written 2026-10-26, five benchmarks

The file opens with "Nothing here is wired into KIRBy yet." Same three steps for each:
reproduce the published score under the published protocol, run the hybrid search on the
benchmark's own feature sets on training rows only, rerun the published protocol on the
hybrid's features.

1. **UCI HAR.** One row is a 2.56-second window of phone accelerometer and gyroscope
   signal; 7,352 training windows from 21 people, 2,947 test windows from 9 people, no
   person in both halves. 18 feature sets by signal, 7 to 79 features each, 561 in total.
   Published protocol: one-vs-all multiclass SVM, Gaussian kernel, 10-fold CV on the
   training windows. Published accuracy 0.9637. Reproduced at 0.9667 with random folds and
   0.9654 with each person's windows kept in one fold. No loader, no registry entry, no
   grid row.
2. **mfeat on OpenML.** 2,000 handwritten digits, six feature sets, 649 features in total.
   OpenML fixes the 10 folds and the metric. Published accuracies: factors 0.9800,
   fourier 0.8805, karhunen 0.9855, morphological 0.7385, pixel 0.9815, zernike 0.9070.
   Three reran; morphological did not finish in 40 minutes and pixel and zernike were
   never reached. `load_mfeat` returns all six feature sets but not each row's OpenML
   fold. The decided grid row carries budgets up to 2,000 features, above the 649 that
   exist.
3. **Matbench v0.1.** Six tasks, one compound per row, MAE: jdft2d 636, phonons 1,265,
   expt_gap 4,604, dielectric 4,764, log_gvrh 10,987, perovskites 18,928. Matbench's own
   five test folds, random rather than grouped, approved for Matbench only on 2026-10-23.
   13 representations built, 9 caches missing, one of them wider than every budget. Two
   published protocols. The random forest reran on all six tasks within about 2% of
   published. MODNet ran on one fold of one task in a mode that is not the published
   protocol. No registry domain fits, no grid rows, and c11 has no option for fixed outer
   folds.
4. **ProteinGym v1.3.** 217 assays, each its own task; one row is one single-substitution
   variant; 33 to 16,897 variants per assay, median 1,340. Published Spearman under
   `fold_modulo_5`. 13 assays chosen by a stated rule. Not reproduced: what exists is one
   assay outside the 13 under a different protocol. The position-based folds collapse any
   per-position encoding to one identical pattern across the held-out rows, so Spearman
   comes back undefined (`proteingym.py:21-45`). No registry entry, no grid rows.
5. **Kepler DR24 threshold crossing events.** 15,737 labelled events, binary target, six
   feature sets (light-curve views of 2,001 and 201 points, 7 event scalars, 10 stellar
   parameters). Fixed published split, replayed exactly. Published: linear, fully
   connected and convolutional families, accuracy and AUC. Only the three linear rows
   reproduced. No loader, no registry domain, no grid row, and c11 has no option for a
   fixed published split.

Deliberately out of NEW_DATASETS.md: caltech101_7, caltech101_20, nuswide_object and
reuters have no published protocol, no fixed folds and no code from the papers. They
still have decided grid rows.

---

## 9. The complexity experiments, by the question each asks

Source: every `c*.py` in `tests/experiments/complexity/`. Present: c01-c09, c11-c24, c31,
c32, c33. c10 moved to the noise suite. c30 merged into c31. **c20 is on
disk but unreachable from the runner** — it is in neither `SINGLE_DATASET_EXPERIMENTS`
nor `module_map` (`complexity_helpers.py:850-910`), and it is the only per-representation
saturation-curve experiment, so its question is answered nowhere in the dispatched set.

c01-c24 and c33 were rewritten in the 2026-10-05 per-fold alignment (decided models,
decided grid, seed 42). c20, c31 and c32 are standalone scripts on an
older footing: `StandardScaler`, their own model list (lightgbm, rf, catboost), three
default datasets. Their own caveat lists say files written before 2026-09-04 (c32) and
before 2026-09-12 (c31) are not comparable to later ones.

### How many samples are needed

- **c05 learning curves** is the only experiment whose axis is the number of samples.
  Draws whole scaffold or sequence groups at 10, 25, 50, 75 and 100% of the rows, three
  draws per size, compares the first hybrid against the first single representation at
  each size, and fits `score = asymptote - a/n^b`. Above 5,000 rows it uses absolute
  sizes 250, 500, 1,000, 2,000. It **skips any dataset with fewer than 10 groups**, which
  rules out abagym_Ang2, abagym_EGFR, abagym_VEGF and the four Ginkgo sets, all of which
  have five. Result files exist for 18 datasets.
- **c33 budget overfitting** expresses every budget as a ratio of the smallest outer
  fold's training rows and sorts budgets into under (below 0.75x rows), interpolation
  (0.75 to 1.5x) and over (above 1.5x).
- **c15 §15c** sweeps budgets at 0.25 to 4x the row count and records `p_over_n_train`
  per cell.

### Model capacity against feature budget

- **c02** the model x budget heatmap per fold, with a PCA control at each budget. 27
  datasets have a result file.
- **c06** capacity setting inside one model family against the optimal budget, as a
  Spearman rho. RF depth, SVM C, DNN width, GP length scale, kNN k. This is the
  experiment behind the thesis paragraph on SVM C. Classification is not wired
  (`c06:617-638`).
- **c16** regularisation strength against budget, and `sensitivity_by_budget`: the spread
  of held-out score across regularisation settings at each budget. The question is whether
  the budget subsumes model-level regularisation. Runs only where rf or svm is a decided
  model. 23 datasets.
- **c09 §9a** the variance shares the thesis already reports; **§9d** the train-to-held-out
  gap per model per budget.
- **c23** which internal evaluator family suits which downstream model, with
  `selection_bias` per arm.
- **c18** at a fixed budget, many representations against few: depth_1, depth_2, depth_3,
  uniform over the matched representations, uniform over every representation, and the
  search's own split.

### Ensembles, and what the hybrid costs to run

- **c11** is the only ensemble experiment. Two ensemble styles (one fixed model per
  representation; each representation picks its own model), two representation sets (top
  three, all), eight aggregation rules, a width-matched PCA arm, and self-contained
  `*_auto` arms that pick the representation set and the aggregator on inner folds.
  It also records a **cost model** per arm: `n_featurisers`, `featuriser_dim`,
  `n_models_deployed`, `model_input_dim`, `model_bytes`, `storage_bytes_per_sample`,
  `selection_fits`, `total_fit_time_s`, `total_predict_time_s`; and a `break_even` block:
  `search_cost_model_fits`, `featuriser_reduction`, `storage_reduction`,
  `model_fits_saved_per_retrain`, `break_even_retrains`, `hybrid_rep_sparsity`. 25
  datasets have a result file. `oracle_single` is labelled not a fair arm and explicitly
  not an upper bound.

### Selection against other selections, and what the features carry

- **c17** mutual information per feature on held-out rows for the hybrid against PCA,
  mRMR, recursive elimination and random at the same budget, plus effective rank.
  Regression only. The mRMR cap of 200 features was kept on 2026-10-06, so **mRMR has no
  cell on 20 of 27 regression datasets**.
- **c15** distance concentration, relative contrast, hubness; RBF kernel condition
  number, eigenspectrum and kernel-target alignment; a double-descent probe; and
  memorisation of permuted labels against the sqrt(d/n) prediction. Regression only.
- **c01** intrinsic dimensionality, mutual information and effective rank per
  representation, with no selection at all. 28 datasets.
- **c22 explainability** per-representation SHAP shares, grouped permutation importance
  and the top 30 features of each hybrid. 26 datasets. It records that the SHAP shares and
  the permutation shares are not comparable and must never share a table
  (`c22:361-369`).
- **c04** importance as a distance weighting: does scaling features by the square root of
  importance improve nearest-neighbour prediction and retrieval. Classification returns
  skipped, with an open TODO (`c04:553-562`).

### Overfitting of the search

- **c33** is the dedicated one: the search's own mean against the honest reselected mean,
  the bias per budget, `bias_vs_budget_spearman`, `overfit_gap` and `process_gap`.
  17 datasets. It carries a block saying that for four of five default models the
  interpolation threshold is not at budget equals training rows, so a flat shape says
  nothing about double descent.
- **c24** one fold fewer, plus the Nogueira stability index. Its own block
  (`c24:11-28`) says the code does **not** apply one fold's blueprint to another fold;
  the diagonal and off-diagonal differ in how many rows the search saw, and "whether that
  is the experiment you want is a science decision and has not been made".
- **c13 forward compatibility**, whether the search takes in a newly added embedding and
  whether the gain beats noise placed in the same feature slots. Its own block
  (`c13:13-26`) says the `D_random` arm is **not** the control its conclusion line
  describes, because since 2026-10-06 it runs no search.

### Not in this suite

Nothing transfers a search result from one dataset to another. c03 correlates each
dataset's winning budget with complexity descriptors and its own docstring says that is
not transfer. c24 is cross-fold inside one dataset. c14 lays datasets side by side.

### Other things to know before using these

- **c12 modality ablation is patient data only** and is therefore out. It is the only
  modality-ablation experiment, so there is no replacement in this suite.
- **c21's H1 arm is off on every dataset**, and every extra-feature arm is skipped on 41
  of 43 datasets under the Step 8 decision, so outside ginkgo_colloidal and ginkgo_hic
  only the single-representation baseline half plus H0 runs.
- **c09's 9d-from-c06 path has never kept a cell** (`c09:1090-1122`); every
  bias-variance number comes from c09's own per-fold computation.
- Local compute cuts are read from environment variables in c05, c13, c15, c17, c21, c23
  and c33, and every one prints and stores what was cut with its own note that it is a
  cut, not a finding.

---

## 10. How published methodology papers word a mixed result

Every sentence below was read from the paper's own abstract or full text by the reader
agent. The point of this section is the shape of the sentence, not the paper.

**Count and denominator in the same sentence, with the effect size beside it.**
OpenFE (Zhang et al., ICML 2023): "OpenFE outperforms all baseline methods significantly
on 40 out of these 49 datasets and provides an average accuracy improvement of 1.9% over
the base features across these 49 datasets."

**No universal winner stated as the result, with the recommendation in the same
sentence.** Bommert et al. 2020, Computational Statistics & Data Analysis: "It is
concluded that there is no group of filter methods that always outperforms all other
methods, but recommendations on filter methods that perform well on many of the data sets
are made."

**Condition on a measured property of the data, never on a domain label.** Luecken et al.
2022, Nature Methods: "method performance is dependent on the complexity of the
integration task for RNA and simulation scenarios." A 2025 Nucleic Acids Research
benchmark: "the optimal method depends on the data, especially the SRT platform."

**Average first, named exception second, and the exception named by a property.**
Jiang et al. 2021, Journal of Cheminformatics: "on average the descriptor-based models
outperform the graph-based models ... some of the graph-based models ... can yield
outstanding performance for a fraction of larger or multi-task datasets."

**Claim, caveat, failure regime.** MoleculeNet (Wu et al. 2018, Chemical Science):
"learnable representations ... broadly offer the best performance. However, this result
comes with caveats. Learnable representations still struggle to deal with complex tasks
under data scarcity and highly imbalanced classification."

**Split the audience, not the domains.** Elor & Averbuch-Elor 2022: "Our results support
the known utility of balancing for weak classifiers. However, we find that balancing does
not improve prediction performance for the strong ones."

**Scope the claim inside the sentence that makes it.** Li et al. 2019, NeuroImage: "at
least in the case for young healthy adults." Grinsztajn et al. 2022: "tree-based models
remain state-of-the-art on medium-sized data (~10K samples)."

**Claim never-worse instead of best.** Chemprop (Yang et al. 2019): "consistently matches
or outperforms."

**Say a tie is a tie.** McElfresh et al. 2023: "the performance difference between GBDTs
and NNs is negligible."

### How they handle the unfair comparison a small dataset creates

1. Truncate every training set to a common size and say so. Grinsztajn et al. 2022 has a
   named section, "Removing side issues", before any result: "We truncate the training set
   to 10,000 samples for bigger datasets", then repeats the whole benchmark at 50,000 so
   the reader can see whether the ranking moved.
2. Treat the number of samples as a measured covariate and report the correlation.
   McElfresh et al. 2023 correlate the per-dataset difference between two methods against
   dataset metafeatures, and report the correlation with the log of the dataset size
   (-0.14, -0.20, -0.16 for three pairs). Their explanation is a ratio, not a domain:
   "GBDTs excel when the ratio of dataset size to number of features is high".
3. Print the size range of the benchmark and contrast it with prior work's narrower range.
4. Where a method is structurally limited to small n, re-run the competitors at that n.
5. Say small n widens the error bars rather than that the method lost. Varoquaux 2018,
   NeuroImage: "sample sizes of many neuroimaging studies inherently lead to large error
   bars, eg +/-10% for 100 samples."
6. Make the small-n regime the stated design target, so a small dataset is a claim and
   not a defeat. ComBat (Johnson et al. 2007): "robust to outliers in small sample sizes
   and performs comparable to existing methods for large samples."

### What they claim as the headline when the per-dataset results are mixed

- The protocol and the released artifact rather than the win.
- A selection rule for the reader, ordered: "Harmony is recommended as the first method to
  try, with the other methods as viable alternatives" (Tran et al. 2020).
- The mechanism that explains the unevenness. Grinsztajn et al. 2022 turn a mixed
  benchmark into three stated findings about inductive bias.
- A corrective against a prevailing assumption.
- A property other than accuracy: cost, versatility, interpretability. Harmony (Korsunsky
  et al. 2019): "superior performance ... while requiring fewer computational resources."

### Section structures used

Luecken et al. 2022 names every Results heading as a complete sentence stating the
finding, including the two that state a dependence rather than a winner. Tran et al. 2020
names each Results section for the condition tested, not for the datasets. Grinsztajn et
al. 2022 puts "Removing side issues" and "Aggregating results across datasets" as named
sections before any result, and gives the explanation of the result its own numbered
section separate from the result. Sun et al. 2022 ends with three separate sections:
Summary and takeaways, Discussion, Limitations of current study.

---

## 11. Reading rule for result files

`/Volumes/seagate/kirby_results_2026-10-10b/analysis/*.md` are flat dumps written
2026-10-08 04:12 from the `run_2026_10_06` ledger. They are **stale and truncated**: each
dataset section is cut at about 400 lines and ends in `... N more lines`, and the result
folders beside them have been written to since, as late as 2026-10-10 09:44. The dumps
also print score-shaped keys only, so labelled axes (which model, which budget, which
evaluator) are missing for several experiments.

Use the dumps as an index. Take coverage from `ls` of
`/Volumes/seagate/kirby_results_2026-10-10b/complexity/<experiment>/*.json` and read the
jsons for any number that goes in a paper.

Non-patient result files per complexity experiment, counted from the json folders on
2026-10-29: c01 27, c02 26, c04 26, c05 17, c06 23, c07 25, c08 26, c09 21, c11 24,
c12 0, c13 15, c15 21, c16 22, c17 20, c18 26, c19 26, c21 23, c22 25, c23 19, c24 21,
c33 16. The eight datasets marked `cut by Adelaide 2026-10-08` across every experiment are
bindingdb_kd, kiba, flab_binding_adams, flab_expression_adams, gb_human_enhancers_cohn,
nt_H3K27me3, nt_H3K9ac, nt_H3K9me3.

Correction to an earlier note in this file: QM9 does have complexity results in 10-10b
(`c05_learning_curves/qm9.json` is `ok`). The 10-08 ledger showed it as pending.

---

## 12. What each complexity experiment found

### c01 representation characterization, 27 datasets

Re-read from the jsons on 2026-10-29, which supersedes the earlier numbers in this file
(they were median +0.0727 over 44 pairs; the read below covers all 27 non-patient
datasets and 54 dataset-and-model pairs).

Per dataset, for each of about 10 feature sets: a mutual-information number between that
feature set's leading principal components and the label, an intrinsic dimension, an
effective rank, and the held-out score of each of the dataset's two decided models fitted
on the feature set's full width. Nothing is selected. Five outer folds from
`nested_cv_outer_folds` with seed 42, the same held-out rows as every per-fold search.

Mutual information does not reproduce the held-out ranking. Rank correlation between the
two within a (dataset, model) pair: median +0.1333 over 54 pairs, from -0.9142 to
+0.7697, positive on 37 and negative on 17. The feature set with the highest mutual
information is also the best-scoring one in 8 of 54 pairs.

The clean counterexample is nanobody thermostability, where all ten feature sets are
summed over the same 50 components so the numbers are comparable: `onehot` has the
highest mutual information at 5.8852 and the lowest scores of the ten, neural net 0.3427
and SVM 0.3376 Spearman; `saprot` has the lowest mutual information at 2.5804 and the
highest scores, 0.6939 and 0.6901.

Where it cannot be read as a ranking: on 8 of 27 datasets the mutual-information number
is summed over different numbers of principal components for different feature sets in
the same dataset (abagym_EGFR mixes 40, 48 and 50; abagym_VEGF mixes 28, 48 and 50), and
the count differs between datasets as well. Seven entries are exactly 0.0000. Four score
means are NaN from folds that returned NaN.

Result files are dated 2026-10-06 to 2026-10-09, so all 27 carry the 2026-10-05 scaling
rule and the per-fold alignment.

Gap between the best and the worst single feature set, within a dataset and model, by
metric. There is no pooled figure: these five metrics do not add, and an earlier entry in
this file quoting a median of 0.1718 across all of them was meaningless.

- Spearman, 26 pairs: 0.1036 (koenig expression, SVM) to 0.3755 (SMAC colloidal
  stability, SVM).
- RMSE in log units, 14 pairs: 0.0193 (Caco-2, neural net) to 0.2727 (QM9, LightGBM).
- RMSE in yield percent, 4 pairs: 2.8435 (Doyle plates, random forest) to 4.0476
  (Buchwald-Hartwig, random forest).
- AUROC, 8 pairs: 0.0413 (hERG blocking, XGBoost) to 0.2321 (tz_her2_binary, XGBoost).
- Weighted F1, 2 pairs: 0.0277 to 0.0871 (tz_her2_multiclass).

The gap does not predict what the search wins. Caco-2 has the smallest gap of all 27
datasets at 0.0193 RMSE in log units, and c02 has the hybrid ahead of the best single
feature set on all ten of Caco-2's model-and-fold units at +0.0215 RMSE.

### c02 model against budget grid, 26 datasets

Read directly from the jsons on 2026-10-29. This supersedes the earlier entry in this
file, which said the result was unusable and that fits had diverged widely — both were
read off the truncated `analysis/*.md` dump, and both were wrong about the jsons.

Every one of the 26 non-patient datasets has all five outer folds scored
(`n_outer_folds_scored` is 5 on all 26). Each fold holds the full configuration list
with model, allocation method, importance method, budget, internal evaluator and feature
count on every entry, a model-by-budget heatmap of training-row scores, the base
representation ranking, the standard selection scored on held-out rows, and a
principal-components arm at each budget.

Held-out result. The better of the dataset's two hybrids against the better of its two
single feature sets, per (dataset, model, fold): the hybrid is ahead on 176 of 260 units,
median +0.0221 in the dataset's own metric. Against the best principal-components width
on the same fold: ahead on 174 of 260, median +0.0148.

Ahead on every one of the ten (model, fold) units: MDR1-MDCK efflux (+0.0306 RMSE), rat
microsomal clearance (+0.0393), Caco-2 (+0.0215), LogD (+0.0468), QM9 (+0.0371). Behind
on every unit of tz_her2_binary (median -0.0339 AUROC). Behind on most units of
humanness_therapeutic (3 of 10, -0.0063 AUROC), abcov_kd (4 of 10, -0.0287 Spearman) and
Buchwald-Hartwig (4 of 10, -0.2970 RMSE in yield percent).

The premise in the file's own docstring -- "different models have different optimal
representation budgets" -- is not carried by the numbers. The two models do pick a
different best budget on 81 of 130 folds. But the training-row score moves very little
across the whole budget range: median 0.0226 in the dataset's own metric, against a
median 0.0817 for the same quantity's spread across the five folds. Both numbers are
training-row scores of the best configuration, so the comparison is like for like, and
the budget range exceeds the fold spread in 1 of 52 dataset-and-model units
(abagym_VEGF with the GP, 0.1399 against 0.1271). So which budget wins is mostly fold
noise, and the honest claim is that the budget has to be searched because its best value
is not stable, not that each model has a budget of its own.

Diverged fits, corrected. Across all 26 datasets, 5 of 93,155 configurations have a mean
off the dataset's metric scale. All five are on MDR1-MDCK efflux, worst 1947.38 RMSE
against that dataset's median configuration of 0.4472, driven by single folds up to
9735.2. Rat microsomal clearance, which the earlier entry named, has 0 of 1,575
configurations above 1.0 RMSE; its largest is 0.54. No off-scale configuration was picked
into any budget cell: the worst score chosen into a heatmap cell is 0.4579 on MDR1-MDCK
efflux and 0.5236 on rat microsomal clearance. Nothing has to be fixed before c02 is
used.

The budget is a maximum. On Caco-2 the 2,000-feature cell builds 1,962 features, and the
principal-components arm at the same cell is capped at 1,725 components by the training
row count, so the widest cell is not width-matched between the two arms.

Arms recorded as not run, each with its reason in the file: fine-tuned representations,
a search at the width of a model's best single feature set, the capped augmentation
condition, and any distance hybrid.

### c04 importance as a distance weighting, 26 datasets, 18 with numbers

Four datasets wrote `skipped, reason: classification_not_supported` (cdr_flexibility,
herg, humanness_therapeutic, tz_her2_binary).

Weighting each feature by the square root of its importance before computing the distance
raises the R-squared of a nearest-neighbour prediction. At 10 neighbours, weighted minus
unweighted R-squared: `lightgbm_gain` median +0.0717 over 15 datasets, positive on all 15,
from +0.0022 to +0.2013; `treeshap` median +0.0551 over 16, positive on all 16; `deeplift`
median +0.0081 over 11, positive on 8 and negative on 3. Largest single moves:
Buchwald-Hartwig under treeshap at 10 neighbours, 0.4977 to 0.6229 R-squared; rat
microsomal clearance under lightgbm_gain, 0.2569 to 0.3787.

Cutting to the top-budget features first and then weighting wipes the effect out: median
-0.0034 for lightgbm_gain, +0.0063 for treeshap.

Cannot be read as the dataset's own result: the score is R-squared, not each dataset's
metric, and on several datasets the unweighted side is below zero, so the gain is measured
off a baseline that does not predict (binding free energy five-nearest-neighbour
unweighted -0.6525; still -0.3762 after weighting at 5 neighbours). The three medians are
over different dataset sets, 15, 16 and 11, not one common set. The select-by-one-method,
weight-by-another matrix does not appear in the dump at all.

### c05 learning curves, 17 datasets, 9 with numbers

Four Ginkgo datasets wrote `skipped`: "5 scaffold/sequence groups; whole-group draws
cannot be cross-validated and splitting inside groups would break the no-random-split
rule".

`delta` is the hybrid's held-out score minus the best single feature set's, signed so
positive means the hybrid won. At the largest training set the hybrid is ahead in 14 of
18 (dataset, model) curves. **No curve moves in one direction with the number of
samples.** The gap closes to nothing on hERG blocking (LightGBM 0.0339 at 250 molecules to
0.0054 AUROC at 2,000) and on QM9 (neural net 0.194, 0.104, 0.048, 0.065 at 250, 500,
1,000, 2,000 molecules; LightGBM 0.053, -0.003, 0.005, 0.004). It stays flat on Caco-2 and
LogD, grows from negative to positive on angiopoietin-2 binding and CDR flexibility, and
flips sign on Buchwald-Hartwig with the random forest (+0.5772, +0.1742, -1.2959,
-0.8327).

So the thesis sentence "its gains are consistent on a few thousand samples but not on a
few hundred" is about the search's own score against the held-out score, which is a
different quantity from this held-out gap. c05 does not support a claim that the gap grows
with the number of samples.

Cannot be read in places: the fitted asymptote exceeds the metric ceiling on 4 of 18
curves (1.3009 Spearman on nanobody thermostability with the SVM, 1.1428 AUROC on CDR
flexibility); one fit has `fit_r2 = -2.287e-12`; 6 of 18 fits are below 0.75. The smallest
training size gives no paired comparison on two datasets. Cells where the hybrid came out
with the same features as a single feature set are dropped (18 times on
Buchwald-Hartwig, 14 on hERG blocking). The raw pair behind each delta is not in the dump.

### c19 importance used during training, 26 datasets

Five conditions: unweighted, features scaled by importance, molecules weighted, both, and
only the features above the median importance. Budget 100 features.

Doing nothing wins most often. `most_winning_condition` is `unweighted` on 11 datasets,
`sample_weighted` on 8, `importance_pruned` on 5, `feature_weighted` on 1, `combined` on
1. The win over unweighted is smaller than 0.005 in the dataset's own metric on 12 of 52
dataset-and-model pairs and above 0.05 on 8. Largest: bh_doyle_plates with LightGBM
0.6441 RMSE in yield percent at the space level; anti-VEGF binding scan with the GP 0.19
Spearman.

Cannot be read for one condition: `importance_pruned` is fitted on half the features,
every time (ratio of pruned width to unweighted width, median and maximum both 0.500 over
1,040 measurements). On the 5 datasets where it won, the winner differs from the others in
two ways at once, the weighting and the number of features. The two conditions that weight
individual molecules could not run with the GP at all: 280 of 2,080 slots are NaN with the
reason "the Gaussian process fit takes no per-molecule weights; not run".

### c21 extra features on one feature set against on the hybrid, 23 datasets, 2 with numbers

21 of 23 are empty. Each records the same reason: the Step 8 decision cut augmentation on
that dataset, so the extra-feature arms are skipped, and `aug_reps` is `[]`.

The hybrid half of the comparison was never built. Both files that ran record that the
arm which would add extra features to the hybrid is "off until the owner decides", and
every hybrid entry has `augmentation_strategy: "none"` and `kept_augmentations: []`.

The two that ran disagree in sign. SMAC colloidal stability: 36 of 100 tests improved,
mean -0.0163 Spearman. HIC retention: 83 of 100 improved, mean +0.0646 Spearman, but the
two random controls also rose (+0.0286 and +0.0294 with the neural net), taking a third to
two thirds of the rise.

The planted decoy donors take most of what the selector adds. In the real-donor condition
on SMAC colloidal stability, of 17.6 features added per test, 5.54 are pure-noise features
and 8.69 are copies of features already present: 80.6% of everything added is decoy. On
HIC retention it is 49.1%. The real donors offered 36 features in all (`paratope` 8,
`structural` 28).

### c22 per-feature-set shares of a hybrid, 25 datasets

The top feature set's share of a hybrid's SHAP attribution: median 0.310 over 25 datasets,
minimum 0.166, maximum 0.715, which is a median 1.82 times what an even split would give.
Top two together, median 0.538.

Concentrated: Buchwald-Hartwig 0.715 with the top two covering 1.000, because those
hybrids hold only 2 feature sets; bh_doyle_plates 0.671; QM9 0.589, and 0.816 under
LightGBM alone in a hybrid of 3. Spread: hERG $K_i$ 0.166, Caco-2 0.171, LogD 0.192,
MDR1-MDCK efflux 0.195 — each keeping all 10 feature sets, so an even split would be
0.100.

Cannot be put in one table: the file itself records that the permutation share "can be
negative, not comparable to SHAP shares". Permutation top shares are systematically larger
(medians about 0.45 against about 0.30, reaching 0.881 on QM9 with LightGBM). Shuffling a
feature set's features made the model better for an average of about 3 of 10 feature sets
on four antibody datasets. The five folds are not the same hybrid: on SMAC colloidal
stability with the neural net the folds used budgets 500, 500, 2000, 2000, 2000 with four
allocation methods and three importance methods, and held 4 feature sets in one fold and
10 in another. `base_representations_explained` is false, so there is no single-feature-set
share to compare against.

### c23 matched against mismatched internal evaluator, 19 datasets

Matching the evaluator to the downstream model is ahead more often than not, by a median
of +0.0024 Spearman or AUROC for a near-neighbour evaluator (ahead on 15 of 25 pairs) and
+0.0061 for a far one (ahead on 20 of 25). Largest wins for matching: Buchwald-Hartwig
with LightGBM 0.9693 RMSE in yield percent; EGFR binding with the neural net 0.0971
Spearman. Largest the other way: binding free energy with the neural net -0.0801 Spearman.

Cannot be charged to the evaluator alone: the two sides search their own recipe inside the
fold and build different numbers of features. For the near-neighbour evaluator, 56 of 177
folds have a width gap of 0 features and the rest have a median gap of 260 features and a
maximum of 1,757. On abcov_kd with LightGBM the per-fold gaps are [0, -950, -1000, 1950,
-6] features. The wider-network condition produced a score on 12 pairs and a paired delta
on none.

### c24 one fold fewer, and stability of the selection, 21 datasets

The file's own `delta_meaning` says it is not blueprint transfer: the source fold's
blueprint is never applied off the diagonal, and the two arms differ in how many rows the
search saw.

Taking one fold out of the search costs a median +0.0173 Spearman or AUROC, with the
own-fold search ahead on 24 of 30 such pairs and on all 12 RMSE pairs. But the
off-diagonal arm trains on about 25% fewer molecules (QM9 7,982 against 5,986;
Buchwald-Hartwig about 3,057 against about 2,350), so the number is a training-size
effect.

The Nogueira stability index over the five folds' feature sets, median 0.1219 over 42
pairs, from 0.0107 to 0.5774. It tracks the number of training rows: QM9 at 7,982 rows per
fold reaches 0.4713 to 0.5774, LogD at 4,022 reaches 0.4203 to 0.4972, hERG blocking at
about 6,200 reaches 0.3265 to 0.4470, while abcov_kd at 112 rows per fold is 0.0107 to
0.0224 and CDR flexibility at about 260 is 0.0381 to 0.0469. The index is computed across
feature sets of very unequal size (abcov_kd with LightGBM selected 100, 100, 250, 2000 and
34 features on its five folds).

### c33 what the search over-reports at each budget, 16 datasets

16 budgets from 10 to 8,000 features, with six interpolated at 0.5 to 2.0 times the
training-row count. `bias` is the search's own score minus the honest re-selected score.

The bias is largest at the smallest budget on the smallest datasets: at 10 features,
abcov_kd with LightGBM +0.3036 Spearman, SMAC colloidal stability with the SVM +0.2078,
EGFR binding with the SVM +0.1796, Buchwald-Hartwig with LightGBM +2.9222 RMSE in yield
percent. At 10 features on the large datasets it is near zero: Caco-2 with the neural net
+0.0018 RMSE, CDR flexibility with XGBoost +0.0004 AUROC, LogD with the neural net +0.0022
RMSE, QM9 with the neural net +0.0029 RMSE.

**The bias shrinks as the budget grows, it does not grow.** Rank correlation of bias
against budget: median -0.408 over 32 pairs, from -0.982 to +0.515, negative on 24 of 32.
At the largest budget it has often turned negative, meaning the honest score is higher
than the search's own (abcov_kd with LightGBM -0.0547 Spearman).

`process_gap`, what letting the search choose costs: median 0.0186, largest in Spearman
abcov_kd with LightGBM 0.2437.

No double descent anywhere. `double_descent.shape` is `flat_across_the_threshold` on 23 of
32 pairs, `inconclusive` on 8, `overfits_and_stays_worse` on 1. The file prints the
fold-spread yardstick beside it, and the movement is smaller than the spread between folds
on the pairs called flat.

The requested budget and the realised width come apart above the pool width: a budget of
8,000 built 2,127 features on hERG blocking with LightGBM and 854 on nanobody
thermostability with the SVM, so budgets above the pool width are not distinguishable from
one another.

### c31 and c32

c31 has no result file. What exists is a cache of 33 json files from 2026-08-19 to
2026-08-21 under `/Volumes/seagate/kirby_cache/c31_weighted_global/`, and three empty
result directories.

c32 has two local result files, Buchwald-Hartwig (written 2026-10-02) and hERG $K_i$
(2026-09-29), both `complete: true`, under
`/Volumes/seagate/dirty_local_2026_09_28/results/c32/`. Three folds each. Their own caveat
list says the deliverable is win counts across cells, not per-cell significance, and that
at small representation counts the weighting schemes converge.

### c06 capacity setting against the best budget, 23 datasets, 16 with numbers

Four classification datasets wrote `skipped`: "c06 passes task_type=regression to every
search and has no entry for the Logit family; the classifier variants exist but are not
wired here".

The quantity is a Spearman correlation per dataset, per model family, per outer fold,
across that family's four capacity settings, between the setting and the budget that
scored best on that fold's training rows.

Of 160 fold-and-family correlations, 121 are `measured`, 24 are `no_budget_variance` (all
four settings picked the same budget, so it is undefined, not zero), 15 are
`insufficient_variants`. Of the 121 measured: 68 positive, 11 exactly zero, 42 negative,
median 0.2582.

The SVM is the one family with a direction. 51 measured folds over 12 datasets: 40
positive, 11 negative, median 0.7746. Positive on every fold of MDR1-MDCK efflux, rat
microsomal clearance, LogD, Caco-2, hERG $K_i$ and the Koenig expression scan. Negative
overall on three: EGFR binding, binding free energy, nanobody thermostability. The neural
net has no direction at all (50 folds: 22 positive, 5 zero, 23 negative, median 0.0), and
nor does the GP (8 folds: 0 positive, 4 zero, 4 negative).

Cannot be read as a smooth relation: the best budget barely moves across the four
settings. The number of distinct best budgets over the four settings was 1 in 30 cells and
2 in 92 of the 160 cells, so a four-point correlation can only land on a few values — the
121 measured correlations take 23 distinct values, and 0.7746, 0.8944 and 0.9487 account
for most of the positive ones.

### c07 easy and hard halves, 25 datasets, 18 with scored partitions

Each dataset cut in half four ways: scaffold frequency, heavy-atom count, target variance,
feature variance. Each half searched separately. 49 of 72 halvings scored; the rest
recorded in `partition_skips` (11 for no SMILES, 6 for a half under 50 rows, 1 for a half
with 0 rows).

The hybrid beats the best single feature set on both halves: ahead on the easy half in 70
of 90 (dataset, axis, model) comparisons and on the hard half in 66. **The gap is not
larger on the hard half** — it exceeds the easy-half gap in 49 of 90. By axis: 18 of 30
for target variance, 14 of 32 for feature variance, 9 of 16 for scaffold frequency, 8 of
12 for heavy-atom count.

Sizes: Caco-2 0.0082 to 0.0399 RMSE on both halves; LogD 0.0194 to 0.0775 RMSE easy
against 0.0728 to 0.1134 hard; hERG $K_i$ 0.0156 to 0.0455 easy against 0.0214 to 0.0617
hard. On binding free energy the two models disagree in sign on the same halving.

Cannot be read as an easy-against-hard contrast: `config_differs_by_fold` records a
different model, budget or importance method between the two halves in 175 of 200 fold
comparisons, with a budget difference reaching 1,750 features on EGFR binding. 17 of 49
halvings have one half more than twice the other. And the label "easy" does not match the
measured difficulty — the easy half has the higher mutual information in only 21 of 49
halvings, and 2 of 18 for the feature-variance axis. The target-variance axis is cut using
the labels (`partition_uses_labels` true on all 17 of its halvings).

### c08 calibration, 26 datasets

Conformal intervals at a nominal 90% coverage, under five conditions: the single feature
set's own width, the hybrid's searched budget, 25 features, the summed width of every
feature set, and that same width with the dataset's other model fitted.

Coverage lands near nominal under every condition, and the differences between conditions
are smaller than the differences between datasets. Median coverage over 21 regression
datasets: 0.9000 at the single feature set's width (median 1,490 features), 0.8891 at the
searched budget (843), 0.8888 at 25 features, 0.8993 at the full 5,845, 0.8992 mismatched.
Paired per (dataset, model), the hybrid at its searched budget is closer to nominal than
the first single feature set in 22 of 52 units.

Interval width moves while coverage does not: median mean width 1.6674 single, 1.5347 at
the searched budget, 1.8620 at 25 features, 1.5914 at full width.

On accuracy the ordering is clear: the hybrid at its searched budget beats the first
single feature set in 40 of 52 (dataset, model) units, 25 features beats it in 8 of 52,
the full width in 22 of 52.

Cannot be read for uncertainty against error: `ue_corr` exists in 40 of 420 regression
units, from four datasets only and in each case only for the random forest. For 17 of 21
regression datasets there is no number at all.

### c09 variance shares, 21 datasets

`decomposition_valid` true on all 21. One observation is one configuration's mean over
outer folds of its training-row search score. Shares in percent, model / budget /
interaction / residual:

QM9 84.1 / 1.0 / 0.2 / 14.8. Buchwald-Hartwig 82.6 / 1.6 / 0.2 / 15.6. Rat microsomal
clearance 77.4 / 14.3 / 2.3 / 6.0. EGFR binding 60.5 / 3.1 / 2.6 / 33.8. hERG $K_i$ 59.7 /
1.0 / 2.4 / 37.0. LogD 48.5 / 1.4 / 1.6 / 48.5. HIC retention 47.4 / 6.8 / 1.0 / 44.8.
AC-SINS 37.6 / 17.0 / 0.2 / 45.1. SMAC colloidal 34.2 / 0.2 / 4.8 / 60.7. CHO
polyreactivity 33.6 / 20.0 / 0.7 / 45.7. Anti-VEGF scan 21.1 / 24.8 / 0.9 / 53.2.
Angiopoietin-2 18.3 / 13.8 / 0.0 / 67.8. Expression titer 14.5 / 1.0 / 1.7 / 82.7. hERG
blocking 14.3 / 51.2 / 0.0 / 34.5. Binding free energy 14.2 / 6.1 / 0.4 / 79.3. Nanobody
thermostability 7.2 / 25.4 / 1.9 / 65.5. abcov_kd 6.3 / 6.2 / 1.8 / 85.6. MDR1-MDCK efflux
2.0 / 3.0 / 4.0 / 90.9. Antibody humanness 0.9 / 41.6 / 0.1 / 57.5. CDR flexibility 0.6 /
40.2 / 1.6 / 57.6. Caco-2 0.0002 / 6.6 / 43.6 / 49.9.

So the residual is the largest share on 12 of 21 datasets, the model term on 8, the budget
on 1 (hERG blocking, 51.2%). The model term is above 30% on 10 datasets and below 1% on 3.
The interaction is above 5% on one dataset only, Caco-2, where it is 43.6%.

Cannot be read as model-against-budget alone: the decomposition has no term for the
allocation method, importance method or internal evaluator, so those fall into the
residual — which is why the residual is largest on half the datasets. Two scale problems:
MDR1-MDCK efflux's ANOVA grand mean is 5.3184 RMSE against a held-out score near 0.4408 on
the same dataset, a ratio of 12.06 where every other dataset is between 0.46 and 1.08; and
Caco-2's total sum of squares is 0.0010 over 252 configurations, so its 43.6% interaction
is a share of almost nothing.

Agreement between the two models across configurations is negative on 13 of 21 datasets,
reaching -0.8958 on EGFR binding. Agreement between budget pairs is high and positive.

The bias-variance read from c06 was never usable: every dataset carries
`bias_variance_c06_unusable_reason` saying c06's heatmap cells carry no training-row score.

### c11 ensemble against hybrid, 24 datasets, 48 (dataset, model) pairs

Arms: the fold's best and second-best single feature set, principal components at matched
width, two hybrids, and ensembles of one model per feature set over two member pools and
eight ways of combining predictions, plus an `auto` arm that picks the pool and the
combiner per fold.

**The auto ensemble beats the hybrid on 29 of 48 pairs; the hybrid wins 19.** Most
differences are under 0.03 in the dataset's own metric. Largest for the ensemble:
bh_doyle_plates with LightGBM 1.5268 RMSE, Buchwald-Hartwig with the random forest 0.5843
RMSE, abcov_kd with LightGBM 0.2532 Spearman. Largest for the hybrid: binding free energy
with the neural net 0.1569 Spearman, LogD with the SVM 0.0447 RMSE.

Against fixed ensemble arms the hybrid is ahead on 20 to 27 of 48 depending on the
combiner — never a majority by much.

**The hybrid beats the fold's best single feature set on 36 of 48 pairs.** Against the
single feature set picked on the held-out rows it is ahead on only 20 of 48, and the auto
ensemble on 28 of 48. Against principal components at matched width the hybrid is ahead on
139 of 196 (dataset, model, width) comparisons.

**Cost.** The hybrid deploys 1 model, the auto ensemble 3 to 10. The ensemble's own
selection costs 24 or 30 model fits per fold; the hybrid's search costs 2,700 to 84,000
model fits over five folds. `break_even_retrains`, how many retrains repay the search,
runs from 333.3 on Buchwald-Hartwig with LightGBM to 28,000 on antibody humanness with the
random forest; 700 on Caco-2 and LogD, 583 on hERG blocking, 875 on the two ADME sets.

Storage per sample shrinks between 2.681 and 2,144 times. **Featurisation does not
shrink**: `featurisation_saving_exists` is true in only 9 of 48 pairs, and
`hybrid_rep_sparsity` is 1.0 in 37 of 48, meaning the hybrid draws features from every
feature set in the pool. In 9 of 48 pairs the hybrid needs *more* featurisers than the
ensemble it is compared against.

Cannot be read as a like-for-like comparison: the hybrid holds a median of 500 features
while the auto ensemble's members together see a median of 15,457, and on 14 of 24
datasets the ensemble kept all 10 feature sets at full width. There is no ensemble arm
built at the hybrid's feature count — the width-matched control is principal components,
not an ensemble. Deployed model counts differ, 1 against 3 to 10. The cross-model ensemble
family has no hybrid counterpart at all: its `head_to_head` row records only the ensemble,
with the hybrid and the paired differences `None`. The augmentation axis never ran.

### c13 adding a newly built embedding to an existing pool, 11 datasets final, 9 partial

Each file holds many candidate embeddings, not one: 6 on Caco-2 and hERG $K_i$, 8 on LogD,
12 to 16 on the antibody datasets. Three antibody candidates were never loaded on any
dataset (abmpnn, antiberta2, antifold) and eight molecular candidates were never loaded on
Caco-2, hERG $K_i$ or LogD, each with the reason "the domain loader returned no matrix for
it (no builder written, or weights absent)".

The search does take features from a new embedding, at very different rates. Counting one
(embedding, model, hybrid slot, fold) measurement at a time: it took nothing on 207 of 300
measurements on CDR flexibility and 178 of 320 on expression titer, against 4 of 120 on
hERG $K_i$ and 5 of 120 on Caco-2. Median features taken: 101 on LogD, 67 on Caco-2, 64 on
CHO polyreactivity, 0 on CDR flexibility and expression titer.

The hybrid with the new embedding beats that embedding on its own on all 11 datasets
(median +0.2408 Spearman on binding free energy, +0.1498 RMSE on LogD, +0.0071 AUROC on
CDR flexibility), and is above zero in every measurement on Caco-2, binding free energy,
hERG $K_i$ and LogD.

Against the noise condition the median is positive on 10 of 11 datasets but small:
+0.0205 RMSE on LogD, +0.0165 on hERG $K_i$, +0.0089 Spearman on HIC retention, down to
-0.0040 on SMAC colloidal stability. LogD is above zero in 31 of 32 measurements and hERG
$K_i$ in 23 of 24; CDR flexibility only 39 of 60.

Cannot be read as the control it claims: the script's own caveat, dated 2026-10-07 and
marked kept rather than fixed, says that since 2026-10-06 the Gaussian condition runs no
search, only swaps the new embedding's selected features for Gaussian ones in the same
slots, so the number "must be described as r against noise in r's slots, not as a
selection-bias-free credit for r", and is expected to be larger than under the old
control. A re-run of about 110 job-hours was not made. The embedding-alone comparison is
also not width-matched: hybrid 25 to 2,000 features against the embedding's full 256 to
17,920. Both standalone controls scored 0 of 5 folds under the GP on four antibody
datasets. Binding free energy has no Gaussian condition at all.

### c15 high-dimensional pathologies, 21 datasets, 15 with numbers

Four classification datasets wrote `skipped, reason: regression_only`.

Distance concentration: the ratio of farthest to nearest held-out neighbour distance is
higher for the hybrid than for principal components at the same width in 233 of 248
measurements, median difference +0.6844, per-dataset medians from +0.0202 to +10.8787. The
spread of pairwise distances is also higher in 233 of 248. Hubness skewness goes the other
way, higher for the hybrid in only 71 of 248.

Memorisation of permuted labels, R-squared on training rows, floored at zero: at full
width it reaches 0.9878 on HIC retention, 0.9918 on AC-SINS with LightGBM, 0.9095 on the
anti-VEGF scan. The hybrid at its decided budgets memorises less on every dataset, and
less than principal components at the same width on every dataset (Caco-2 0.0241-0.2891
against 0.0476-0.6934).

Against width-matched random features the hybrid is ahead in 28 of 28 measurements on
AC-SINS (+0.1206 Spearman), HIC retention (+0.1561) and CHO polyreactivity (+0.0818), 23
of 28 on Caco-2, and behind in 2 of 28 on angiopoietin-2 binding and 8 of 28 on EGFR
binding. The random comparison is width-matched in all 412 measurements.

**No double descent.** Budget set to 0.25, 0.8, 0.9, 1.0, 1.1, 1.2 and 4.0 times the
number of molecules. Over 60 (dataset, model, slot) combinations where 0.9, 1.0 and 1.1 all
scored, the score is worse at 1.0 in 30 — a coin flip. Median dip 0.0000, median absolute
dip 0.0147. The best factor is 4.0 in 21 of 60, 0.8 in 11, 0.25 in 9.

Cannot be read for principal components: 247 of 412 measurements have that arm at a
different width from the hybrid, and 53 have no score, because above the number of
training molecules that many components cannot be formed.

### c16 does the budget replace model regularisation, 22 datasets, 12 with numbers

Four datasets wrote `skipped` because neither decided model has a regularisation sweep
here — only the random forest's depth and the SVM's $C$ do. On all twelve that ran, only
one of the two decided models was swept; the other is named in
`decided_models_without_family`.

**Narrowing the budget does not stop regularisation mattering.** The spread of held-out
score across settings is smaller at the narrowest budget than at the widest in 18 of 24
(dataset, hybrid slot) pairs and larger in 6, and the differences are small. hERG $K_i$
0.1111 RMSE at 250 features rising to 0.1278 at 2,048. LogD 0.1689 rising to 0.1962.
Caco-2 0.0518 to 0.0558. The six going the other way include the widest spread in the set,
Buchwald-Hartwig 1.6011 RMSE at 500 features falling to 1.5407 at 2,048.

The spread is driven by the extreme settings, not by a flat surface: on hERG $K_i$ at 250
features the five $C$ values give RMSE 0.8415, 0.6925, 0.5647, 0.5538, 0.5666.

A tuned wide hybrid against a simple narrow one at default settings: the simple one is
better in 11 of 22 pairs, median +0.0005. The simple one wins on Buchwald-Hartwig (14.9529
against 16.1489 RMSE), binding free energy (0.3253 against 0.2653 Spearman) and
angiopoietin-2 binding; the wide one wins on all four RMSE small-molecule datasets. The two
sides are not given the same choices — the wide one is scored at its best of four or five
settings, the simple one at the default only — and they are at different widths.

### c17 what the selected features carry, 20 datasets, 14 with numbers

Three classification datasets wrote `skipped, reason: regression_only`.

**mRMR has a cell on 3 of the 14 datasets.** The cap is 200 features, and the other eleven
datasets' smallest decided budget is 250 or 500.

Mutual information per feature, in nats on held-out rows, hybrid against each method at
the nominal same budget, over 94 (dataset, budget, model) combinations: higher than
principal components in 73 of 94 (median +0.1395); higher than recursive elimination in 37
of 94 (median -0.0171); higher than random in 54 of 94 (median +0.0213); higher than mRMR
in 3 of 12 (median -0.4152).

Held-out score, sign-normalised: the hybrid is better than principal components in 69 of
82 (median +0.1052) and better than random in 77 of 94 (median +0.0292). **It does not
beat recursive feature elimination: better in 46 of 94, median -0.0007.** Against mRMR,
better in 4 of 12.

This is the one that matters for the thesis claim. The thesis figure reports the hybrid
lower-error at every budget against recursive elimination on four small-molecule datasets.
Across the fourteen datasets here the two are a tie, with per-dataset medians between
-0.2748 RMSE on Buchwald-Hartwig and +0.0110 RMSE on hERG $K_i$.

Cannot be read in places: six measurements have RMSE in the tens of thousands (adme_hlm
with the neural net, hybrid 10,545.8 at 1,000 features) and are excluded from every figure
above. 65 of 100 measurements have the principal-components arm at a different width,
because that arm cannot exceed the number of training molecules — 2,000 features against
156.6 components on CHO polyreactivity.

### c18 many feature sets against few, 26 datasets, 22 with numbers

Hybrid's own allocation minus each hand-built condition, sign-normalised, over 184
(dataset, model, budget) combinations:

- against one feature set only: hybrid better in 171 of 184, median +0.0576
- against the budget split over two: better in 157 of 184, median +0.0267
- against three: better in 148 of 184, median +0.0220
- **against an even split over the feature sets it used: better in 87 of 184, median
  -0.0002**
- against the second-best hybrid: better in 93 of 184, median +0.0001

Positive against one feature set on all 22 datasets (+0.2080 Spearman on binding free
energy, +0.2045 RMSE on Buchwald-Hartwig, +0.0102 AUROC on hERG blocking). Against the
even split, eleven datasets positive and eleven negative. The edge over one feature set
does not shrink as the budget grows; the edge over the even split is within 0.03 at every
budget.

So across 22 datasets the allocation rule is not where the gain is. Which feature sets are
drawn on is. That matches what the thesis says for small molecules and extends it.

Cannot be read for the depth conditions: they build exactly the budget every fold while
the hybrid falls short of it in 95 of 184 combinations, so they are wider than the hybrid
in about 95 to 100 of 184. And in 462 of 2,760 (condition, fold) records a depth condition
built more features than scored above zero — 2,000 features built with 200 above zero on
abcov_kd. The script's own note (`c18_diversity_vs_depth.py:185`) says the hybrid stops at
the last positively scored feature while the depth conditions take the budget regardless,
that "which rule both arms should use is the open zero-gain decision", and that this was
left unchanged. The even-split comparison is the only one in this file matched on both
width and importance method.

The second even split, equal allocation over *every* feature set rather than the ones the
hybrid used, is absent from all 22 files: they carry
`result_format: per_fold_uniform_matched_to_natural_2026_10_06`, and the script's current
format adds `uniform_every_rep`, which never ran.

---

## 13. What each similarity experiment found

18 non-patient datasets in the suite. Read from the jsons under
`/Volumes/seagate/kirby_results_2026-10-10b/similarity/`.

Computed files per experiment: s01 18 of 18, s02 17, s03 18, s04 **6 of 18**, s05 17,
s06 17, s08 17, s09 **0, no folder**, s10 17, s11 **4 of 18**, s12 18, s13 17.
flab_expression_koenig is missing from s02, s05, s06, s08, s10, s11, s13.

s04's 12 non-computed files say "molecular-only experiment, domain=antibody" and name s12
as the replacement, with their own note that the stub never checked an s12 result exists.
s11's 13 stubs are domain=antibody (10), `clipped_top_label` (Caco-2, LogD) and
`classification_degenerate` (hERG blocking).

**s01.** Distance tracks label difference weakly in every dataset. Best
representation-and-distance per dataset, in Spearman: antibody humanness 0.3885,
MDR1-MDCK efflux 0.2808, QM9 0.2789, down to hERG blocking 0.0412 and SMAC colloidal
stability 0.0155. Median of the 18 per-dataset bests 0.1387; 5 reach 0.20 or above, 6 stay
below 0.10. Which distance wins is not consistent: cosine on 8 datasets, manhattan on 5,
minkowski_p0.5 on 5. Fold spread is the size of the effect on the small antibody sets
(abcov_kd 0.0942 +/- 0.1002). No missing cells. Representations are compared at native
widths, 40 features to 17,920, so a difference between two representations is also a width
difference.

**s02.** Unweighted, manhattan: the nearest-neighbour hybrid's distance tracks label
difference better than the best single representation on 16 of 17 datasets. QM9 +0.2482,
LogD +0.1922, hERG $K_i$ +0.1896, antibody humanness +0.1764; abcov_kd is the exception at
-0.0525. Under importance weighting on both sides the lead shrinks (QM9 +0.2482 to
+0.1310). The budget sweep is the readable part: the lead falls monotonically with width on
9 of 17 datasets — hERG $K_i$ +0.1896 at 100 features, +0.1528 at 250, +0.1104 at 500; LogD
+0.1922, +0.1747, +0.1190 — and is positive at the narrowest budget on 16 of 17 and at the
widest on 17 of 17. The single side is the rank-1 of about 10 representations at full
width (2,048 features for ECFP4 on hERG $K_i$) while the hybrid is a searched recipe at
100 to 2,000 features, so the lead is confounded with width and with having been searched.

**s03.** The features a hybrid selected against the same number of random features from
the same representation, in nats. On training rows the selected features carry more. On
held-out rows it is close to a coin flip: 372 of 652 cells above zero. Best per-dataset
held-out means: QM9 +0.937, hERG $K_i$ +0.876, antibody humanness +0.865,
Buchwald-Hartwig +0.732. Negative in all four cells on hERG blocking and on LogD; Caco-2
splits by model, -0.094 with the neural net and +0.216 with the SVM. The estimator's own
note says it is not the joint mutual information, so the nats are not an information
budget. This is the one place in the suite where the two sides are matched on width.

**s04.** 6 molecular datasets computed. A held-out molecule is a cliff molecule when at
least two training molecules at ECFP4 Tanimoto 0.55 or above differ in label by more than
the fold's activity gap; its own label is never read. The hybrid beats the best single
representation on cliff molecules in 11 of 12 model-dataset cells — LogD with the neural
net 0.4299 to 0.3876 RMSE, hERG $K_i$ with the SVM 0.5606 to 0.5297 — **and by a similar
amount on the non-cliff molecules**, so the cliff-to-non-cliff gap is not closed in any
consistent way (Caco-2 with the neural net narrows from +0.0473 to +0.0062 RMSE; QM9 with
the neural net widens from +0.1774 to +0.2381). MDR1-MDCK efflux and QM9 have 5 to 14
held-out cliff molecules per fold against 514 and 1,983 non-cliff. hERG blocking uses a
different cliff rule from the other five, `fixed_mad_multiple` at 53.9% of training
molecules against about 15.0% elsewhere, so its cliff set is not the same object. The
hybrid gets 216 kNN candidates against 72 baselines.

**s05.** Predicting each held-out molecule from its nearest training molecules, recipe and
kNN setting both chosen on the fold's training rows. The hybrid is ahead on 10 of 17
datasets and behind on 7. Ahead: SMAC colloidal stability +0.1988 Spearman, QM9 +0.1946
RMSE, EGFR binding +0.0844, LogD +0.0736. Behind: abcov_kd -0.1259, CHO polyreactivity
-0.0901, expression titer -0.0738, Buchwald-Hartwig -0.1858 RMSE. At one fixed setting
instead the hybrid is ahead on 13 of 17, and **the direction on 4 datasets depends on
whether the setting is fixed or re-chosen per fold**. Per-fold deltas swing far wider than
their mean on the small sets. The hybrid is picked from 216 cells against 36 baselines
(144 against 24 on hERG blocking, LogD, QM9). The similarity objective could not run at
all on binding free energy and the four Ginkgo sets.

**s06.** Spearman between a held-out molecule's mean distance to its nearest training
molecules and the model's error on it. Median over cells, hybrid: MDR1-MDCK efflux
+0.2923, hERG $K_i$ +0.2711, antibody humanness +0.1866, LogD +0.1613, Caco-2 +0.1283,
down to SMAC colloidal stability -0.0027, abcov_kd -0.0061, CDR flexibility -0.0524 and
Buchwald-Hartwig -0.0434, where it is negative in all 36 cells. Hybrid against the best
single representation: ahead on 11 datasets (AC-SINS +0.0741, QM9 +0.0415) and behind on
6 (Buchwald-Hartwig -0.1476 with 0 of 36 cells positive). The hybrid's conformal intervals
are narrower than the single representation's on every dataset and cover less often —
nanobody thermostability 0.9033 to 0.7903 coverage at width 27.3160 to 21.3582 label
units — with QM9 the one exception where coverage holds. The conformal control could not
run on the three classification datasets. The reliability deciles are not monotone on any
dataset checked. This is the one experiment whose comparator is close to matched on
settings, 2 options against 1.

**s08.** The number of features that predicts best against the number that makes distance
track label difference best, on the same hybrid. **They disagree in 149 of 184 cells**, and
`optima_differ_trustworthy` is true in all of them. The direction is one-sided: distance
wants fewer features. MDR1-MDCK efflux prediction 1,000 and 2,000 against distance 100 to
250; QM9 prediction 2,000 for both models against distance 250 and 100; AC-SINS prediction
2,000 against distance 50 to 100, a gap of 1,900 features. Caco-2 is the one dataset where
they coincide for a model. On the distance side the hybrid beats the best single
representation on 15 of 17 datasets (QM9 0.1532 to 0.4802 Spearman, antibody humanness
0.2902 to 0.5400). But on 7 of 17 datasets the two budget grids overlap at only one width,
which the file states itself: "the two optima can only coincide at a budget in both".

**s09.** No folder, no results.

**s10.** Multiplying the scaled features by their importance, for four things that use a
distance. Nearest-neighbour prediction: helps on 15 of 17 datasets, small and consistent
(LogD +0.0178 with 300 of 312 cells better, QM9 +0.0270, MDR1-MDCK efflux +0.0052 with 436
of 468); expression titer -0.0052 and antibody humanness -0.0019 go the other way. **The
SVR gets worse on 10 of 13 datasets** and **the Gaussian process on 8 of 11**. q-RASAR has
no direction. The Gaussian-process control is missing on 5 of 17 datasets, and where it
ran, many weighted fits ended at a length-scale bound where every held-out molecule gets
the training mean — 121 of 260 fits on Buchwald-Hartwig, 101 of 405 on abcov_kd. q-RASAR
is thin: no dataset reaches 5 folds in every cell. The exponent and rescale of the weights
are marked open in the file.

**s11.** 4 datasets computed. **Nothing readable.** On 3 of the 4 the hunt was over before
it started: `iters_to_top1pct_mean` is 0.0 with 3 of 3 seeds for every one of the 26
conditions, because the top 1% of labels was already in the seed set. Whole arms had a
flat acquisition function so picks followed row order — 300 of 300 picks on one
Buchwald-Hartwig arm, 285 to 300 on three others. The ranking of hybrid families reverses
between datasets, and on QM9 the gauche-GP arms are next-to-worst while on MDR1-MDCK
efflux they lead. Three seeds, with spreads often larger than the gaps between arms.

**s12.** The same cliff comparison as s04 on all 18 datasets, with percentile-based cliffs
for the antibody and reaction domains. **On 9 of the 18 datasets the cliff stratum is
empty or nearly empty**: 0 folds produced a cliff score on abcov_kd, binding free energy
and the four Ginkgo sets; 1 of 5 folds on EGFR binding and nanobody thermostability; 2 of
5 on MDR1-MDCK efflux and QM9. Where it can be scored the hybrid helps cliff molecules on
some datasets and hurts on others with no pattern — the Koenig expression scan with the
SVM +0.1136 Spearman but with the neural net -0.0147; CDR flexibility negative in all 6
cells; Buchwald-Hartwig loses on both strata with the random forest and wins on both with
LightGBM. s04 and s12 report different numbers for the same comparison because they count
different folds (QM9 with the neural net, s04 0.5359 to 0.5508 over 5 folds, s12 0.2422 to
0.2798 over 2). The three antibody partitions use percentile cutoffs differing by orders of
magnitude, so "similar enough to be a cliff partner" is a different relation in each.

**s13.** Clustering coherence and retrieval cost on 15 spaces. Clustering at 10 clusters,
hybrid minus best single representation: ahead on 12 of 17 (QM9 +0.2272, CHO polyreactivity
+0.2145, LogD +0.1250) and behind on 5 (SMAC colloidal stability -0.0761, EGFR binding
-0.0685). **The sign changes with the number of clusters on 7 of 17 datasets.** Retrieval:
the hybrid is cheaper on 12 of 17 (AC-SINS +0.7358 in 18 of 18 settings, nanobody
thermostability +0.5795 in 18 of 18) and dearer on 5 (binding free energy -0.1043 in 1 of
18 settings). The hybrid is chosen from 18 candidates against 6 baselines for clustering
and 108 against 36 for retrieval. The clusterer matched to each distance was switched off,
so everything is clustered with squared-Euclidean k-means while retrieval uses manhattan,
cosine and minkowski. The retrieval cost has no definition string anywhere in the json.

### The pattern across the similarity suite

Six of the eleven computed experiments compare a hybrid against a single feature set while
giving the hybrid more to choose from and at a different number of features: s02 (a
per-fold search against the rank-1 of ten representations at full width), s04 (216 against
72), s05 (216 against 36), s08 (7 against 2), s13 clustering (18 against 6) and s13
retrieval (108 against 36). The three that are not loaded this way are s06 (2 against 1),
s10 (weighted against unweighted inside one arm) and s03 (selected against random features
from the same representation at the same count).

---

## 14. From `NOISE_AUDIT_2026_09_10.md` and the handoff documents

### The one that matters for the cliff question

§15.5: `activity_cliff_aware` was cut from n11, n13 and n06 on 2026-09-12 because on
hERG $K_i$ in ECFP4, over 1,412 molecules, **it ranks molecules the same way as
`neighbor_residual` at Spearman 0.989**. So on that dataset the cliff-aware detector and
the plain neighbour-residual detector are very nearly the same ranking, which is the
measured version of "the cliff-aware method works but so does the baseline it was
compared with".

### Rulings recorded in the audit

§16.5, the author's ruling of 2026-09-17: the ChEMBL curator flag is **outlier noise
only**. "No number from this experiment may be written up as a result about noise in
general." It is written into every n22 result as `this_is_one_kind_of_noise` and
`it_says_nothing_about_laboratory_variation`.

### Defects that change how a noise number reads

- §13.1: distinct per-molecule noise widths on 2,000 labels at level 0.30 — gaussian 1,
  Student-t 1, Laplace 1, grouped-shifted 1, outlier 2, grouped-wider 2, **censoring 601**.
  Censoring is not one damage size applied to everything.
- §6.9: the detection cut-offs divide by the corrupted spread while the level is a
  fraction of the clean spread. Corrupted over clean across the six levels is 1.0000,
  1.0092, 1.0349, 1.1264, 1.2474, 1.4002, so a cut-off nominally at two spreads sits at
  2.80 clean spreads by the top of the grid.
- §4: n12 divides by zero at level 0.0, so one of the six levels is not-a-number in every
  recovery figure. n07 discards at least 10 uncorrupted molecules in every 100 by
  construction. n11 writes no classification result file at all.
- §4, n13 classification: on 400 synthetic binary labels the realised flip share at level
  0.5 was 0.492 with corrupted-to-clean correlation 0.019; at 0.75 it was -0.411; at 1.0 it
  was -0.995, which is every label inverted.
- §14.10: the repeat-measurement ceiling drew from the wrong population — 1,218 of 9,802
  molecules have been measured more than once, so a ten-in-a-hundred ceiling removed about
  eighty in every hundred of every molecule ever measured twice.
- §14.4: the ceiling ranked on a standard deviation with `ddof=0`, which ranks partly by
  how many measurements a molecule has: two measurements of a wide assay report about 0.56
  of its real width, six measurements of a narrow one about 0.91 of its own.
- §14.5: repeat measurements usable per dataset — bindingdb_kd 845 of 46,090 pairs; SKEMPI
  129 of 6,193 and blocked, because repeats are overwritten in the loader; LogD 0 of
  5,326; Caco-2 and the other Polaris sets 0 of 3,521.
- §7: three experiments outside the shared route use their own levels. One records
  `outlier_p05` while delivering 25 molecules in every 100.
- §16.2: the dead curator filter — records carrying both a comment and a pchembl value, 0
  in the fetch, 482 with a comment and no pchembl value, 373 across the whole database.

### Contradictions inside the documents

- n06: `SIMILARITY_HYBRID_NOT_BUILT_HANDOFF_2026_09_19.md:37-44` says "Every stored n06
  result claims a comparison it did not make" and makes fixing n06 its first instruction.
  `NOISE_AUDIT_2026_09_10.md` §18 says "n06 is NOT one of the six, and the handoff that
  listed it was wrong", and that both spaces have been built since 2026-09-12. Neither
  re-read the code on the same day. Still unverified: whether any n06 result file written
  before 2026-09-12 is still in use.
- The audit's own ledger disagrees with its item lines: §10 says thirteen of fifteen
  decisions are closed with only 9 and 15 open; the item lines say 9 closed 2026-09-16 and
  15 closed 2026-09-18. Item 15 is both closed and open inside itself.
- hERG IC50 record count: 12,010 in one section, 12,008 over 9,802 molecules in two others.
- Censoring: §6.12 says nothing in the noise directory runs censoring; §13.1 says it now
  does. The level 0.50 is in the §3.4 grid and §13.1 says it is gone from every grid.
- The classification grid is described as one grid taking no task type in two sections and
  as a six-level sweep in another.
- Two subsections are both numbered §14.11 and the section order is out of sequence.
- Four different group counts from three grouping rules on the same dataset: 769 (where
  fold and noise grouping coincide), 751, 749, 767.
- The fix for one defect was centralised into a shared helper, but n11, n13, n18 and n21
  keep their own copies, so there are now two implementations.

### Still open, from these documents

- Whether a sixth precompute pass is added or an existing one widened. The author's call.
- n15's label-fingerprint check is still missing, and no stored search record carries a
  noise field at all — no setting name, no seed, no label fingerprint.
- Levels 0.5, 0.75 and 1.0 are not covered by the clean-against-noisy search comparison,
  so whether the clean-label advantage grows or shrinks as noise rises is measured nowhere.
  0.5 is the level the policy names as the reference.
- Two smoke tests fail unfixed.
- n17 rests on one configuration: one representation, one model, three draws, and all three
  unrun axes are the axis the result would be read along.
- SKEMPI needs the same loader change hERG needed. Repeat-measurement counts for three
  other datasets are unmeasured.
- `qm9_graphs_noise_robustness.py` asks a question nothing else asks, is uncalled, raises
  at the first injection, and the roster holds no graph network.

---

## 15. The documentation index, and what in it is still true

234 markdown files at the repo root, 96 `.before_*` backups, about 20 MB more under
`/Volumes/seagate/claude_analysis_2026-10-08/answers/`. Everything below is a dated claim
from a document; the ones marked VERIFIED were re-read in code on 2026-10-29.

### The per-experiment answer pages are the best result source

`/Volumes/seagate/claude_analysis_2026-10-08/answers/` holds one page per experiment —
c01-c33, n01-n41, s01-s13, fed01-fed05 and seven OPIG — rebuilt 2026-10-09 by
`_scripts/<exp>.py` from the per-row result files, 4 KB to 1.0 MB each. Roll-ups beside
them: `QM9_CONTROL.md`, `SUCCESSES.md`, `NOISE_BY_GROUP.md`, `NOISE_BY_DATASET.md`,
`CLEANING_BY_METHOD.md`, `DETECTOR_BY_MITIGATION.md`, `MEETING_METHODS.md`, `FAULTS.md`.
`answers_before_2026-10-09/` is superseded. Use these in preference to the truncated
`analysis/*.md` dumps.

### VERIFIED FIXED: the scaling asymmetry

`FRAMEWORK_FINALIZATION.md` (17 Sep) says fingerprints are standardised in the hybrid arm
and raw in the best-single arm, at 324 sites. **That is closed.** `src/kirby/scaling.py`
records the author's rule of 2026-10-05 — 0/1 features are never scaled, continuous
features are standardised, everywhere including the 0/1 features inside a hybrid — and
`FeatureScaler` is used in both arms, at `find_best_hybrid.py:5255` for the hybrid and
`:4637` for the single representation.

The docstring also records how large the old defect was, which matters for any result
written before 2026-10-05: on hERG $K_i$, the SVM on ECFP4 scored 0.6841 RMSE standardised
against 0.6122 left as 0/1, and kNN on atom pairs 0.6452 against 0.6193. So a pre-10-05
hybrid-against-single comparison on a fingerprint can be wrong by about 0.07 RMSE for that
reason alone.

### Open, and it hits a Paper 2 claim directly

`CHANGES_BEFORE_RERUN.md`: representation native widths reach no result file, so the claim
that the hybrid wins **with fewer features** cannot be stated. Needs checking against the
current files before that claim is used — c01 records `n_features_offered` and s01 records
`dim`, so it may now be available for some experiments.

### Open, and it hits a Paper 3 claim directly

`NOISE_AUDIT_2026_09_10.md`: the labels already carry error before anything is injected.
LogD's clean spread is 1.1905 log units against a published measurement error of 0.48,
already 0.403 of the spread; hERG $K_i$ 0.9143 against 0.54, 0.591 of the spread. **So the
level-0 row is not a clean control on a measured dataset.** QM9 is the only clean control,
which is what the thesis says.

Same file: the roster's random forest is not reproducible at its shipped thread setting —
0 of 39 repeats bit-identical — so any comparison asking whether two scores are the same
has a floor of about 1e-15. Not changed.

### The author's own ruling that inverts a framing

A brief in the answers directory, the fourth of four and overriding the third: a dataset
as delivered is **not** clean, so a cleaning method that improves fits on the labels as
delivered is removing the assay's own noise — the best case, not a warning sign.

### Originality, from `thesis_provenance/NOVELTY_BRIEF.md` (26 Oct)

Across the 79 noise methods: 2 top-grade, 22 second, 49 third, 6 bottom. The two
claimable contributions are **cliff put-back** (`n25_cliff_exemption.py:838-944`) and
**cleaning before the hybrid search** (`n20_clean_before_search.py`). The brief's closing
line: "the noise work carries two originality claims, not a shelf of them." Of 273 commits
traced across the 79 methods, 155 carry no reason in the message.

Four methods named as new in Chapter 8 were searched: cliff put-back, none found after two
passes; the cliff-aware neighbour gap, related but different; the neighbour gap divided by
the neighbours' own spread, the same statistic published outside machine learning and
chemistry as a meteorological check (Baserud et al. 2020); cleaning before the feature
search, related but different.

### The kNN axis decisions, `KNN_SEARCH_SPACE_RECORD.md` (6 Oct)

Datasets split at 5,000 molecules. Euclidean cut: it wins 0 of 14 blocks in one survey and
ranks last in 11. Importance weighting cut from the shared neighbour settings: of 48
dataset-and-representation pairs, 1 gains more than its own fold-to-fold range. Uniform
neighbour counting wins 0 of 688 groups. Four axis decisions carry an asterisk because
hERG blocking has 0 of 60 tasks finished, and on the part that did run, 9 neighbours is
the best count on 81 of its 157 cells and the gaussian scheme on 288 of 471 — both cut.

**Clipping inverts the detection score**: detection area under the curve 0.2768 on hERG
$K_i$, 0.3151 on Caco-2, 0.3936 on MDR1-MDCK efflux. Replacing a clipped label with its
neighbours' mean was worse in every row, 2,550 of 2,550, 2,580 of 2,580 and 2,520 of 2,520.

### Contradictions between documents, worth knowing before quoting any of them

- **Delete clipped** is recorded as a negative result in two handoffs ("deleting at the
  limit hurts"; hERG $K_i$ with the network 0.742 to 0.890 RMSE) and is on the author's
  list of methods that worked in a third.
- The number of noise methods reads 85 in one handoff and 79 in the audit, the novelty
  brief and the git-history file.
- Six published-sweep methods are recorded as having "no noise kind where they work" in one
  handoff and as all being in List 1 in another.
- The configuration count per search reads 8,160, 2,400, 10,080 and "roughly 600" in
  different documents. The code gives 8,160 without the GP and 10,080 with it.
- hERG $K_i$'s scaffold-group count reads 751, 749, 767 or 769 depending on the document
  and the scaffold key.
- `suzuki` is "run" in the precompute decisions and "cut" in the rerun commands, with a
  stated reason. The same split applies to `lhasa_herg`.
- Two screens of the best single representation per dataset agree on no dataset.
- How many experiments clip held-out features to the training range: three, six or twelve
  depending on the document.
- `RUN_COMMANDS.md` is named as a live document by project memory and was rejected and
  deleted by the author. It is absent from the repo.
- QM9: one document has the variance decomposition as robust and the model ranking holding;
  another in the same directory, six days later, says every QM9 number in the repository
  came from runs whose held-out labels were corrupted.

### Open items that bear on the two papers

- Four noise-code decisions unanswered: the heteroscedastic Gaussian process, DivideMix,
  contrast-to-divide, and whether MentorNet keeps its name. Each is fix-or-rename.
- Every stored result for the changed arms is on the old footing and nothing has re-run:
  the noise ladder (ten experiments), forest proximity (fourteen importers), the
  cliff-aware weight, confident learning, the uncertainty curriculum (renamed self-paced
  learning 2026-10-29; two uncertainty variants tried and dropped 2026-10-30), MentorNet, the
  close-pair flag.
- No cleaning or detection method has ever been measured against clipping.
- Whether a matched random-removal arm is added to each experiment, and in which shape.
- The cliff thresholds are not swept, and the per-dataset cliff share is computed in the
  run and collected nowhere.
- Whether 7 of the 12 published noise methods may keep picking their stopping epoch on the
  labels they are graded against.
- 55 stored result files hold no measurement, and none of the 1,541 result files carries a
  status field (dated 2026-08-27).
- The QM9 re-run after the held-out-label fix: nothing regenerated, named as the critical
  path.
- The replacement robustness metric for Paper 1 is undecided.

---

## 16. OPIG experiments

Per-experiment detail is in the OPIG table of `PAPER_EXPERIMENT_TABLES.md`. What is not in
the table:

Fifteen entries: fourteen `tests/experiments/opig_*.py` files plus the
`opig_generator_scoring/` support directory. Six have scores: activity_cliff_qsar,
blueprint_transfer, cgr_comparison (partial, no final file), ginkgo_developability
(partial), graphinity_ddg (scores only in the partial; the final file's score block is
empty), shared_vectors_census (still running), shared_vectors_remedy.

The ledger at `/Volumes/seagate/kirby_results_2026-10-10b/LEDGER.md` lists 11 OPIG units:
ok for blueprint_transfer (7.6 h), activity_cliff_qsar (22.0 h) and shared_vectors_remedy
(0.6 h); running for shared_vectors_census; skipped for cgr_comparison,
ginkgo_developability, graphinity_ddg, humanness, scaffold_hybrid, scaffold_similarity and
trastuzumab_enrichment. Three files have no ledger unit at all: cdr_flexibility,
kingsmith_component_splits, generator_scoring_swap.

Two faults from `answers/FAULTS.md` touch almost every OPIG number that exists:
- A1: every neural-net score in this copy predates the clip of predictions to the
  training-label range.
- A6: hybrids are built below their stated budget wherever importance reaches zero, so a
  budget label is not a width.

Two disagreements between OPIG runs:
- The shared-features census counted two antibody feature sets on binding free energy at
  429 rows each; the graphinity run reported the same two as not computed (FAULTS C3).
  Whether both read the same cache directory is unchecked.
- The census counted nothing on the ChEMBL targets with a stored error, "no cached
  representations", while the activity-cliff run lists 8 built feature sets for each of the
  same three targets.

---

## 17. Noise experiments n01-n21: what is not in the table

Per-experiment detail is in the noise table of `PAPER_EXPERIMENT_TABLES.md`.

**Fold coverage.** Only outer fold 0 of 5 ran in n02, n06, n07, n09, n11, n12, n13, n17,
n18, n19, n20 and n21. n01, n14 and n16 ran all five. n15 ran three to five, unevenly, and
the fold rule is broken in both directions: n15 ran 3 folds on hERG $K_i$, which is on the
all-five list, and 5 on Caco-2, which is not.

**Every stored result predates the fixes.** Files are dated 2026-10-07 to 2026-10-10.
`484f78a` (forest proximity counted over out-of-bag trees; touches n07, n14, n21, n25,
n32, n35), `1f5a4de` (unparseable SMILES in the Tanimoto space; n01, n32), `006c261` (the
n02 cliff-aware weight), `1c12208` and `a2b6330` (confident learning and the uncertainty
curriculum; n09, n13) all landed after.

**The forest-proximity rewrite matters most for n21.** The new code counts a pair only in
the trees whose bootstrap left both molecules out, about 13.5 trees of 100 instead of all
100. Forest proximity is the one distance family that lost in n21 (median -0.0124 average
precision), so that number measures the old quantity.

**Three experiments report a quantity their own files contradict.**
- n11's `heldout_recovery` is null in all four files and for every one of its 8 evaluation
  units, and that is the primary reported claim.
- n17's ceiling never beat its random floor in any of 8 units, so every deletion position
  is undefined.
- n20's identity control fails on hERG $K_i$: with no cleaning the two searches get
  byte-identical labels and must pick the same recipe, and only 4 of 20 matched on each of
  two damage conditions. The spread over arms reaches 0.0252 RMSE there, larger than the
  median ordering cost of every cleanup. The control passes on the two antibody datasets,
  20 of 20.

**n21 contradicts the thesis reading of whether the hybrid helps detection.** Paired on the
same feature set and the same cell, the four hybrid neighbour spaces raise detection by a
median +0.0244, +0.0191, +0.0126 and +0.0119 average precision. The thesis table reports
the hybrid's own features as best on no dataset overall, but that table takes each
detector at its best setting across four space families; n21 is a paired comparison. The
audit page already recorded the same tension: averaged over detector settings instead of
best-setting, the hybrid comes out ahead more often.

**n18 is a harder negative than the thesis reports.** The thesis cites the best
configuration per dataset passing its shuffled control. Across all settings in n18 the real
detector's flags win in fewer than half for four of the six cleanups with a control, and
the median real-minus-shuffled gain is negative for four of six. The one cleanup that beats
its control, the Gaussian-process smoother, scored zero folds on Caco-2, hERG $K_i$ and
LogD without recording a refusal, because the code refuses above 1,000 training molecules.
One cleanup has no shuffled control at all.

**`grouped_shifted` needs a scaffold group built from SMILES**, so it is skipped on every
antibody and reaction dataset in n07, n09, n11, n12, n13, n16 and n20. On one antibody
dataset in n20 every damaged draw failed for that reason and the file still reads `ok`.

**Six experiments rest on hERG IC50 alone** — n17, n19, n22, n26, n29 and n36 — so nothing
in the suite cross-checks that dataset. n17 and n19 grade 405 of its 9,802 molecules.

**The cliff criteria disagree between experiments on the same dataset.** n14 uses a fixed
2.0 label-unit gap and finds 0 cliff compounds on Caco-2, because its whole label range
spans 1.9976 units. n01 uses twice the median absolute deviation (0.4170 on Caco-2) and
finds 1,625. The two counts cannot be read side by side.

**One external reference number exists in the whole suite**: the published 0.54 pKi
repeat-measurement floor, compared in n14 against out-of-fold residual spreads of 0.5870
on hERG $K_i$, 0.5025 on LogD, 0.4657 on MDR1-MDCK efflux, 0.2910 on Caco-2 and 0.3467 on
QM9.

---

## 18. Noise experiments n22-n42: provenance that affects every number

Per-experiment detail is in the noise table of `PAPER_EXPERIMENT_TABLES.md`.

### How many noise files hold all five outer folds

`OUTER_FOLDS_RUN` at `tests/experiments/_shared.py:3929` holds c05, c06, c07, c10, c13,
c15, c23, c33, n15 and n26. Of the twenty experiments n22-n42 only **n26** is in it, at 3
folds, and hERG IC50 is absent from `ALL_FOLDS_DATASETS`, so n26's folds 0-2 are the
configured rule and not a shortfall. **Every other fold shortfall in the noise suite is
unfinished work, not a decision.**

The only files in the whole noise suite holding all five outer folds: n27's seven,
n37's eight, `noise_full/n30/logd`, `noise_full/n24/herg_ki__censoring`, and n30b's two —
and n30b sits outside the result tree with no ledger unit.

### The answer pages were built from a different result folder

`/Volumes/seagate/claude_analysis_2026-10-08/answers/` holds pages n01 to n41, all stamped
09 Oct 14:01, built from `kirby_results_2026-10-09_noise_view` rather than
`kirby_results_2026-10-10b`. There is no page for n42 and none for n30b. On n38 the page
contradicts the current jsons: it says only one dataset has a damaged level and that the
residual detector ranked moved molecules no better than chance, while the 10-10b jsons
hold four antibody datasets at three doses with detection reaching 0.9930 AUROC. So the
answer pages are an index, not the authority — correcting what I told the readers earlier.

`FAULTS.md` is stamped 08 Oct 14:57 and one of its items is wrong: it says cleanlab is not
in `requirements-working.txt`, but line 70 pins `cleanlab==2.9.0`. The missing detector is
real — the module was absent from the environment those passes ran in — and the stated
cause is not.

### The neural-net prediction cap splits the files by date

Commit `c18ee30`, 2026-10-08 13:04:35, capped neural-net predictions to the training-label
range. Finished non-patient files predating it: n40 15 of 32, n24 13 of 35, n23 5 of 48,
n25 5 of 19, n30 4 of 15, n27 3 of 7, n33 2 of 12, n37 2 of 8, n34 1 of 8. None in n22,
n26, n29, n31, n32, n35, n36, n38, n39, n41, n42. One caveat: these files resume per fold
and the json is rewritten whole each time, so a post-cap write time does not prove the fold
was computed after the commit.

A second fix, `ffcd254` at 2026-10-08 11:03:43, made the Gaussian process start its length
scale at the square root of the feature count instead of 1. It touches nothing in n31 to
n34, whose decided model pairs exclude the GP. It does reach n30b's CHO polyreactivity
dataset, where the GP is one of two decided models; that file postdates the commit.

### The cliff result that matters most

**n42's cliff-exemption control is readable where n25's is not**, because in n42 the two
arms hold matched row counts. It says the method's gain comes from the cliff molecules:
putting the cliff molecules' labels back **cancels the method** on every dataset —
Buchwald-Hartwig +0.436978 to +0.000358 RMSE, CDR flexibility -0.024619 to +0.008270 AUROC
— while putting back the same number of randomly chosen molecules leaves it almost
unchanged, +0.443001 and -0.032625.

That is the opposite of what cliff put-back assumes. The premise is that a cliff molecule
carries a correct label that looks wrong to a detector, so sparing it should help. Here
sparing them removes the whole benefit, which means the cliff molecules are the ones the
cleanup was working on.

n25 cannot test this, and the reason is countable. The cliff test flags more than half the
training molecules on Caco-2 (866 to 1,412 of 1,726), Buchwald-Hartwig (1,740 to 3,056 of
3,057), LogD (2,027 to 3,458 of 4,022) and hERG blocking (3,345 to 4,231 of 6,206), so
there are not enough non-cliff molecules left to match and the comparison scored 0 passes
on 13 of 16 files. It could run on hERG $K_i$ only, and there sparing cliffs does not beat
sparing the same number of non-cliff molecules.

### Three more results worth carrying

- **n26 gives the one external ceiling in the suite.** Reliability 0.6432 from the repeat
  measurements implies an R-squared ceiling of 0.4453. The SVM on both single feature sets
  sits below it (0.4118 and 0.4305) and both hybrids under both models sit above it, up to
  0.5103.
- **n30's shuffled-lift control.** Dealing the same lifts to the wrong clipped molecules
  matches or beats giving each molecule its own amount in all 8 cells of the two datasets
  with enough folds. What matters is knowing which labels are at the limit.
- **n30b is the real-data counterweight.** On a dataset with 58 of 197 labels sitting at a
  genuine assay floor, no treatment clears its own fold spread over five folds, while the
  same treatments recover an injected clip on QM9 in 5 of 5 folds. Its control separates
  the two explanations on QM9: moving the same number of labels **not** at the limit hurts
  in every cell and every fold.

### Arms that never ran

- Every Gaussian-process arm in n40: 592 skip entries, all for the 1,000-molecule cap.
- n41's intended calibration set, molecules with repeat measurements: no such dataset
  carries an n41 unit.
- n38's two heteroscedastic conditions, which were the comparison between noise whose size
  follows a feature the model sees and noise whose size it cannot see.
- n39's real-repeats block: the correlated share was never measured on a dataset with
  repeat measurements.
- n35 never ran on the one dataset with real repeats, so its near-duplicate estimate is
  never checked against a measured one.
- n36 has no summary file at all, 1 of 10 planned passes.
