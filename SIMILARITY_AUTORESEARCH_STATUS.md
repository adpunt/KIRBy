# Similarity suite — what has been run, debugged and optimised

Opened 2026-09-20. A live tracker, not a result file. Every number in it is a
local run on three whole datasets: caco2 2,158 molecules, herg_ki 1,412,
adme_mdr1 2,641. No rows were ever subsampled; what was cut is the searched
grid — 7 representations that RDKit builds locally in place of the 10 in
DATASET_REPS, budgets 25/100/250, one downstream model, one importance method,
two allocation methods.

⛔ Nothing here is evidence about which allocation method, importance method or
budget wins. Those axes were cut for compute.

Two categories, and they need different work:

- **investigation** — the question is whether the numbers coming out answer the
  question going in. Done when it runs and its output is readable.
- **method** — something is meant to work better than the alternative. Done when
  it has been optimised against that alternative, or when it has been measured
  well enough to say it cannot be.

---

## s01–s13

| exp | kind | smoke tested / debugged | methods to optimise | optimisation done |
|---|---|---|---|---|
| s01 | investigation | yes, 3/3 datasets | no — it is the comparator s02 is measured against | n/a |
| s02 | method | yes, 3/3 | yes | **yes** — 20 cells per dataset, all three complete |
| s03 | investigation | yes, 3/3 | no — the file stamps `descriptive_only=True` | n/a |
| s04 | method | yes, 3/3 twice | yes | **yes** — four levers, 48 settings |
| s05 | method | yes, 3/3 | yes | **partial** — knobs added, re-run still in flight (2026-09-21) |
| s06 | method | yes, 3/3 twice | yes | **none** |
| s08 | investigation | yes, 3/3 twice | no | n/a |
| s09 | investigation | **yes, ran 2026-09-21 on all three** | no | n/a |
| s10 | method | **partial** — has never completed a full run | yes | **partial** |
| s11 | method | yes, 3/3 at two budgets | yes | **yes** |
| s12 | method | yes, 3/3 twice | yes | **yes** |
| s13 | method | yes, 3/3 | yes | **retrieval yes; clustering CANNOT BE** — its statistic moves 6-34x the delta on a reseed |

## Never opened

| group | scripts | note |
|---|---|---|
| s09 | 1 | opened 2026-09-21. Ran on all three, read all five inputs, wrote 11 series |
| standalone | `similarity_optimized_hybrid` | imports and prints help. `tests/results/sim_optimized_hybrid/` exists and is empty — nothing has ever been written to it |
| standalone | `structural_geometry_survey` | cannot address these three. It reads heavy and light chain sequences, and its registry holds 31 antibody datasets — caco2, herg_ki and adme_mdr1 are not among them |
| `similarity_metrics/` | ~25 per EXPERIMENTS_FINALIZED.md | several are method experiments: nary, triplet, noise-cleaning, creative_hybrids |

---

## What the optimisation found, per method

**s02 — the negative was the comparator, not the hybrid.** At weighting
exponent 0 both arms are unweighted. The hybrid barely moves (herg_ki 0.2669
weighted, 0.2669 unweighted, euclidean); the best single representation falls
from 0.2886 to 0.0779. A 2,048-bit fingerprint is only competitive as a
distance because importance weighting rescues it. Held-out delta, hybrid minus
best single representation:

    coefficient, exponent      herg_ki    caco2   adme_mdr1
    manhattan, 1 (shipped)     -0.0237  +0.0260   -0.0233
    euclidean, 0               +0.1891  +0.1283   +0.0453

**s04 — wins two of three once its levers open.** Held-out RMSE delta on cliff
molecules, positive means the hybrid won:

    setting                                 herg_ki    caco2   adme_mdr1
    shipped: 5 neighbours, sqrt, 100 feat   -0.0184  -0.0088   -0.0074
    neighbours/exponent/budget/family open  -0.0036  +0.0269   +0.0315

**s10 kNN — distance-weighted voting is the reliable win.** It lowered held-out
error in 20 of 21 (dataset, representation) pairs and nothing in the suite uses
it. The hardcoded 0.5 exponent is never best; k needs no widening.

    dataset      no weighting   as shipped   best swept   picked
    caco2            0.3561       0.3420       0.3350     exp 0.25, 7 nbrs, distance vote
    herg_ki          0.6631       0.6550       0.6378     exp 0.25, 5-7 nbrs, distance vote
    adme_mdr1        0.6663       0.5878       0.5814     exp 1.00, 15 nbrs, distance vote

Underneath it: the weighting is mostly feature deletion. lightgbm_gain scores
75-90% of a fingerprint's features at exactly zero, so any positive exponent
deletes them. Dropping always helps; rescaling the survivors helps on two
datasets and hurts on herg_ki by 0.0133 pKi.

**s11 — the mechanism is the distance scale, not the number of features.**
Median Euclidean distance between two caco2 molecules, against an RBF kernel
starting at length scale 1.0: 50 selected features 6.6, maccs 13.8, pdv 16.7,
ecfp4 45.3, PCA-50 83.1, random-50 102.4. The spaces that collapse are the ones
whose median distance sits far above 1. PCA-50 and random-50 have fifty
features each and still collapse. Area under the best-found curve at 100 picks,
lower better: herg_ki hybrid 415.1 against raw ECFP4 429.3; adme_mdr1 hybrid
-87.40 against -55.26; caco2 nothing separates and random picking beats the
hybrid.

**s12 — wins all three under the calibrated cliff rule.** Held-out delta:
caco2 +0.0154, herg_ki +0.0107 (was -0.0024), adme_mdr1 +0.0416.

**s13 retrieval — euclidean beats the shipped manhattan on two of three.**
Held-out delta in label units at exponent 0.5: caco2 manhattan +0.0181 against
euclidean +0.0236; adme_mdr1 +0.0504 against +0.0594; herg_ki +0.0245 against
+0.0023. Manhattan is the default because f18 measured it on herg_ki and logd.

**s13 clustering — the headline subtracted coherence at 3 clusters from
coherence at 10.** Every space is scored at 3, 5 and 10 clusters and each arm
keeps its own best count per fold. Label coherence rises with the count — at 10
clusters it is higher than at 3 in 44 of the 50 (dataset, space) pairs on disk —
because more clusters means fewer molecules per cluster and a smaller
within-cluster standard deviation. On herg_ki the hybrid's best count is 3 and
the best single representation's is 10, so the reported -0.1183 is 0.3866
against 0.5050, at two different granularities. Held-out label coherence delta,
unitless, higher better, positive means the hybrid won, 4 paired folds on both
arms (fold 0 excluded because the pool pruned on its scores):

    clusters         caco2   herg_ki  adme_mdr1
       3           +0.0145   -0.0129    +0.0340
       5           +0.0519   +0.0584    +0.0067
      10           -0.0381   -0.2417    +0.0264
    each arm free  -0.0381   -0.1183    +0.0264

At 5 clusters the hybrid wins all three. ⛔ That win does not survive a reseed —
see the instability note below before quoting it.

**A cluster holding fewer than 2 held-out molecules contributes no term, and
which spaces lose clusters is not even.** Summed over 5 folds on herg_ki at 10
clusters, ecfp4 loses 16 of its 50 clusters and top_torsion 20 of 50, while the
six hybrid spaces lose between 0 and 4. ecfp4 is the single representation that
wins at that count, and its 0.4761 is a mean over the clusters that held 2 or
more held-out molecules — about 34 of 50. The classification purity branch was
given the same 2-molecule floor in `df7c7fe`; all three datasets here are
regression, so that branch never fires on them.

**s13's PCA control was matched on one dataset of three.** `pca_matched` is
fitted at 100 components on every fold. Each hybrid slot's width is whatever
budget its recipe chose, from 25, 100 and 250. Of the 30 (hybrid slot, fold)
pairs per dataset, the control's 100 components equal the slot's feature count
in 29 on adme_mdr1, 15 on herg_ki and 5 on caco2 — 24 of caco2's 30 are 250
features wide. The per-width controls added in `df7c7fe` did not fix it: a width
that occurred on one fold produced a control scored on one fold beside a hybrid
scored on five, and the two means were read side by side.

**⛔ THE CLUSTERING HALF CANNOT RESOLVE THE DELTA IT REPORTS.** Label coherence
is not what k-means minimises, so two clusterings of one space with almost the
same inertia can score far apart, and running k-means again from a different
seed changes the number by much more than the hybrid-minus-best-single gap.
Measured on herg_ki fold 1 at 5 clusters, 1,129 training and 283 held-out
molecules, one space at a time, over four `random_state` values. The column is
the highest coherence minus the lowest, unitless:

    restarts   top_torsion 2048 feat   ecfp4 2048 feat   pdv 200 feat
        1              0.3719               0.4174          0.0293
        2              0.2235               0.3491          0.0453
        5              0.4475               0.2259          0.0464
       10 (shipped)    0.2986               0.3416          0.0176
       20             0.2149               0.1398          0.0124
       40             0.2149                  —            0.0124

The delta the clustering half reports is 0.01 to 0.05 coherence. At the shipped
10 restarts the seed alone moves a 2,048-feature fingerprint by 0.30 and 0.34 —
6 to 34 times the gap being measured. Raising the restarts to 20 halves it and
still leaves 0.14 to 0.21. Only the 200-feature descriptor set settles: 0.0124
from 20 restarts on, and 40 buys nothing over 20. The single representations
the hybrid is scored against are the 2,048-feature fingerprints, so it is the
COMPARATOR that will not hold still.

Two whole-experiment re-runs on herg_ki, same folds and same pool, 10 restarts
against 2, differ by up to 0.19 coherence on one space (rep_top_torsion at 3
clusters, 0.2869 against 0.0925) and flip the sign of the delta at 3, 5 and 10
clusters. Held-out label coherence delta, positive means the hybrid won, 4
paired folds:

    clusters       10 restarts   2 restarts
        3            -0.0129       -0.2240
        5            +0.0584       -0.1368
       10            -0.2417       -0.0785

At 2 restarts on caco2 the delta changes sign at every step of the count sweep
— -0.0517, +0.0174, +0.0343, -0.0466, +0.0492, -0.0567 at 2, 3, 5, 8, 10 and 15
clusters — which is what a measurement looks like when the noise is larger than
the effect. There is no right cluster count to be found until the statistic
stops moving.

**The distance mismatch, measured.** k-medians honours the manhattan
coefficient the retrieval half reads; k-means minimises squared euclidean and
takes no metric argument. Same run, same folds, same 2 restarts, 5 clusters,
held-out delta, positive means the hybrid won:

    dataset      k-means (squared euclidean)   k-medians (manhattan)
    herg_ki                -0.1368                   -0.1090
    caco2                  +0.0343                   -0.0514

Making the two halves agree moves the delta by +0.0278 on herg_ki and -0.0857
on caco2, and flips caco2's sign. Both movements sit inside the seed spread
above, so the mismatch cannot be separated from the noise at this restart
count. `minkowski p=0.5` has no clusterer of this shape at all — there is no
closed form for the point minimising summed L_0.5 distance — and the file now
writes that instead of running euclidean silently.

**s10 GP — a null, and a correction.** Multiplying features by the square root
of their importance destroys an RBF kernel: the weighted arm returned the
training mean on all three datasets (herg_ki label spread 0.9151, weighted arm
0.9160). With the arm alive, weighting makes the GP slightly worse, +0.0111 and
+0.0188 pKi on two folds of herg_ki/pdv.

**s10 q-RASAR — a null.** The four neighbour features carry real signal alone
(pdv's four reach 0.6730 pKi against 0.9160 for a mean predictor) and add
nothing to a forest that already has the features.

---

## What s09 says across three datasets

Ran 2026-09-21 against `tests/results_local_sim`. All five inputs loaded for all
three datasets, all fifteen (dataset, input) pairs stamped
`per_file_2026_09_18`, so no column was held back for mixed selection rules. It
wrote eleven series, each carrying three numbers — one per dataset — and never
averages them.

**The signed-rank test is not worth reading at three datasets.** Enumerated over
all eight sign patterns of three values, a two-sided Wilcoxon can return only
0.25, 0.5, 0.75 or 1.0. The floor of 0.25 is reached whenever all three share a
sign, no matter how large they are: three deltas of +5, +9 and +12 give the same
0.25 as +0.0001, +0.0002 and +0.0003. The p-value therefore carries nothing the
three signs do not already carry. Read the three per-dataset numbers.

s05's held-out kNN delta, the column s09 reads as `knn_delta_heldout`, in RMSE
where positive means the hybrid beat the best single representation:

    caco2 +0.0134, herg_ki -0.0105, adme_mdr1 +0.0292

**The two sides of s09's own `knn_delta` are not picked over the same number of
choices.** The hybrid side takes the best of 36 cells — 3 budgets x 2 search
families x 3 ranks within a family x 2 neighbour counts. The baseline side takes
the best of 7 representations, each already reduced to its own better neighbour
count out of 2, so 14 cells. So the hybrid side maximises over 36 and the
baseline side over 14, and the difference between them is read as a win for the
hybrid. `knn_delta_heldout` does not have this problem: it re-picks both sides on
the other folds.

Four things an earlier audit listed, re-checked against the code on 2026-09-21:
s09 is absent from `SINGLE_DATASET_EXPERIMENTS`, so `run_dataset.py` rejects it —
it takes `--datasets` and is run directly instead. Neither s09's docstring nor
`_summary_keys.py` names s07; the only mention left in the suite is
`similarity_helpers.py:399`, which calls it deleted. The NaN guards are present
and fired: the ratio column, which is exactly 1.00 on all three datasets,
recorded `wilcoxon_skipped` with scipy's own reason rather than a p-value.

---

## Fourteen defects found and committed

Each sat under a number that had been reported as a result.

| commit | what two things disagreed |
|---|---|
| `cb3ef49` | s11's Expected Improvement tied across the pool, so `np.argmax` picked in row order — 89 of 90 picks on herg_ki, 90 of 90 on adme_mdr1. Its one flag counted only fits that raised |
| `3b48515` | `random_50d` is a Gaussian projection with a real acquisition, and was on the list exempt from the degeneracy flags |
| `cb82362` | s04 and s12 used different cliff rules; 250 of 496 herg_ki cliff molecules agreed. s12's percentile sweep could never reach its own 10% target |
| `6506675` | the weighted GP's kernel width was divided away by sum-to-1 weights, so it returned the training mean |
| `318c80b` | the importance exponent was two different numbers under one name: 0.5 in s13, family-implied in `compute_distance_matrix` |
| `9c6f7d7` | the search cache had no column for the distance coefficient; one stored row served both, 0.6754 and 0.7327 under one number |
| `c5752d4` | `objective='similarity'` searches built a hand-written Euclidean whatever `--distance-metric` said, and four callers swallowed a stale-cache error into an empty pool |
| `37c3598` `686c337` `6c3c981` | the GP reported a score for a fit that had run out of room at its length-scale bound, and said nothing |
| — | `build_fold_hybrid` refuses a model with a geometry recipe; s04/s05/s06/s10/s11 passed one unconditionally and each caller's `except` swallowed the raise, so every similarity-family arm was written out as not computed |
| — | `MAIN_MODELS_WITH_KNN`, `DISTANCE_MODELS`, `RIDGE_MODELS`, `DOWNSTREAM_MODELS_NOISE` are built above the block that cuts the model list, so a cut list still named six models |
| — | the `rep_count` axis raised under a geometry objective, killing half the similarity grid |
| — | s03 asked the pool for models a cut search never scored |

---

## Open decisions for the owner

1. **Cliff share.** `CLIFF_TARGET_PCT = 15.0`. It is the smallest share where
   every held-out fold on all three datasets carries at least 30 cliff
   molecules. At 10% herg_ki's thinnest fold is 19 and caco2's 23. At 20%
   adme_mdr1's gap falls to 0.21 x the median absolute deviation. Even at 15%
   adme_mdr1 needs a gap of 0.50 x MAD, which is a weak cliff — it has 2,318
   scaffolds among 2,641 molecules.
2. **The s02 headline cell.** Largest delta comes from no weighting anywhere;
   the hybrid's own best geometry comes from weighting. They pick different
   cells.
3. **Keying `baselines/` and `hybrid_evals/` by coefficient.** Until they are,
   the coefficient axis needs a separate `KIRBY_CACHE_DIR` per coefficient.
   Namespacing them orphans every file already in both directories.
4. **Which end of the label s11 hunts.** It negates whenever the dataset metric
   is lower-is-better, so on all three it chases the LOWEST label — the weakest
   binders, the least effluxed. Nothing in the file says which end is good.

## Two things that change how the suite must be run

- Every similarity-objective number in this campaign was computed under
  Euclidean while the suite's default coefficient is manhattan. Reproducing
  them needs `--distance-metric euclidean`.
- Manhattan is the BETTER coefficient for that objective on caco2, beating
  euclidean on all seven representations by 0.012 to 0.051 Spearman. The code
  was using the worse one by accident.

## Compute flags for the cluster

- s10's Gaussian process task is the longest block in the wave. It builds the GP
  with `n_restarts_optimizer=2`, tripling every fit for a hyperparameter neither
  arm reports; both arms get it, so it cancels out of the comparison. Now
  settable by `KIRBY_S10_GP_RESTARTS`.
- Under a geometry objective the general and general_distance searches compute
  the same numbers — the evaluator and model lists are ignored. Warming both is
  a second full grid per dataset per augmentation condition for nothing.
- A minkowski p=0.5 cell costs 25 to 50 times a manhattan one: scipy's cdist is
  39 s for 1,412 by 1,412 rows at 2,048 features under manhattan and over 7
  minutes under minkowski.
- s11's MC-dropout BNN condition killed three processes at 14 GB of 15 GB swap
  and recorded nothing when it did.

---

# `similarity_metrics/` — the group nothing in this campaign had opened

Opened 2026-09-20. This group does not share anything with s01-s13 except the
idea. Four things about it are different and all four change how it is read.

**It has its own molecules.** Every script here calls `_data.load_prepared`,
which reads `sim_<name>.npz` sitting next to the scripts. Three such files
exist: herg_ki 1,415 molecules whole, logd 1,427 molecules kept from 5,039 by
whole Bemis-Murcko scaffold groups, adme_solubility 2,173 molecules whole.
caco2 and adme_mdr1 have no npz in this directory, so nothing here has ever run
on them. Building one means running `build_local_datasets.py` against a cached
CSV, which rebuilds ten representations and a full molecule-by-molecule ECFP4
Tanimoto matrix from SMILES.

**It never touches `/Volumes/seagate/kirby_simrun_cache/rep_cache/`.** The ten
representations are stored inside each npz. Nothing here calls
`find_best_hybrid`, nothing opens a `result_db`, so the StaleResultDB hazard
does not reach this group.

**It already does the thing s10 found helps.** `simeval.eval_knn_readacross`
builds `KNeighborsRegressor(n_neighbors=k, weights='distance',
metric='precomputed')` at simeval.py:161. Distance-weighted neighbour voting is
the default here, not an unused option.

**Its distance coefficients are not the process-wide ones.** `simeval` offers
Tanimoto, Dice, cosine, Soergel, Tversky and Euclidean and computes them by
hand. `src/kirby/similarity/distance_config.py` and
`KIRBY_IMPORTANCE_EXPONENT` are not read anywhere in this directory.

## Survey — every file in the directory

One row is one file. "Runs" means it was imported in this session and its data
dependency checked; all 31 files on disk import without error. Cost is wall
clock measured in this session where a time is given, and is not an estimate
where it says measured.

| file | asks | kind | runs today | cost on 1,415-2,173 molecules |
|---|---|---|---|---|
| `simeval.py` | nine ways to score one distance matrix against measured activity | library, no main | yes, 16 scripts call `eval_knn_readacross` | n/a |
| `rankers.py` | score a feature by how its pair-difference tracks the activity gap or a structural distance | library, no main | yes, imported by creative_hybrids, creative_hybrids_v2, feature_selectors/selector_prediction | n/a |
| `ranker_experiment.py` | does similarity-aware feature selection beat the count fingerprint under its own Tanimoto | METHOD | **no — the file is not on disk**, deleted in `4ab874c` | — |
| `blacklist.py` | which features inside one representation make its geometry worse | METHOD | **no — the file is not on disk**, deleted in `df7c7fe` | — |
| `scaffold_experiment.py` | does describing an antibody variant by what changed beat describing its whole sequence | METHOD | needs a FLAb library through `kirby.datasets.flab`, not a molecule dataset | not timed |
| `scaffold_ddg_experiment.py` | same question on single-point mutants where mutation count cannot vary | METHOD | needs the graphinity ddG set | not timed |
| `scaffold_feature_isolation.py` | which of five property scales carry the agreement with BLOSUM62 | investigation | needs three FLAb libraries | not timed |
| `scaffold_reps.py` | is blosum_matrix a distance | library, no main | yes | n/a |
| `feature_type_study.py` | do 14 property scales each earn their place in the distance | METHOD | **library only — no `if __name__` block.** Its one caller is `tests/experiments/opig_scaffold_similarity.py` | — |
| `feature_ablation.py` | which of four feature groups help, and does any combination beat ECFP4-count Tanimoto | METHOD | yes | not timed |
| `base_rep_comparison.py` | does describing the edit beat describing the whole sequence at neighbour prediction | METHOD | needs four FLAb libraries | not timed |
| `aggregate.py` | does fusing six distance matrices beat the best single one | METHOD | yes | not timed |
| `creative_hybrids.py` | does stacking descriptors, MACCS and fine 2D features onto ecfp4_count beat it | METHOD, superseded by v2 | yes | not timed |
| `creative_hybrids_v2.py` | same, on three datasets, with a per-dataset greedy descriptor drop | METHOD | yes | measured, see below |
| `winning_recipe_reselected.py` | does the stacked space still win once selection and scoring sit on disjoint molecules | METHOD | **the doc names a file that has never existed in this repository's history.** The file on disk is `winning_recipe_honest.py` | — |
| `winning_recipe_vs_standard.py` | does the stacked space beat two field-standard fingerprint metrics | METHOD, the headline table | yes | not timed |
| `structural_embeddings.py` | do MAP4, TDA or mol2vec beat ecfp4_count Tanimoto | METHOD | yes, `map4` and `mhfp` both import | not timed |
| `structural_yardstick.py` | which representation-and-coefficient pair wins when the judge is shared substructure | investigation | yes | not timed |
| `shape_similarity.py` | do USR, USRCAT or a pharmacophore descriptor beat ecfp4_count Tanimoto | METHOD | yes, conformers cached in `sim_<ds>_spatial.npz` | not timed |
| `aug3d_similarity.py` | do E3FP or Coulomb eigenvalues beat ecfp4_count Tanimoto | METHOD | yes, `e3fp` imports, cached in `sim_<ds>_3daug.npz` | not timed |
| `learned_embeddings.py` | cache one ChemBERTa and one MoLFormer vector per molecule | precompute plus one library function | `torch` and `transformers` both import; needs the model weights downloaded | not timed |
| `extra_fingerprints.py` | do Avalon, SECFP6 or MHFP6 beat count-ECFP4 Tanimoto | METHOD | yes | not timed |
| `augmented_descriptor_reps.py` | does a descriptor block beat count-ECFP4 Tanimoto | METHOD | PDV and Mordred yes, `padelpy` absent | not timed |
| `nary_similarity.py` | does complementary similarity rank activity-vs-neighbour disagreement better than two naive scores, and does reweighting the distance by it help | METHOD | yes | measured, 6.1 s herg_ki, 14.5 s logd |
| `triplet_similarity.py` | does a triplet-trained embedding beat count-ECFP4 Tanimoto on held-out folds | METHOD | yes, `torch` imports | not timed |
| `similarity_noise_cleaning.py` | does a distance chosen for structural fidelity find corrupted activity labels better than count-ECFP4 Tanimoto | METHOD | yes | not timed |
| `run_battery_moleculeace.py` | which representation and coefficient wins the full nine-statistic battery across MoleculeACE targets | METHOD | **no — the `MoleculeACE` package is not installed here, so there are no target CSVs and no npz files to score** | — |
| `mcs_cache.py` | maximum-common-substructure distance for a fixed set of 8,000 molecule pairs | precompute | yes, already cached for herg_ki, logd, adme_solubility | n/a |
| `_data.py` | which molecules, in which row order, every script indexes over | loader | yes | n/a |
| `_heldout.py` | split molecules in half by whole scaffold group | library, no main | yes | n/a |
| `_nanrank.py` | keep an uncomputable correlation out of a ranking and out of an average | library, no main | yes | n/a |
| `build_local_datasets.py` | rebuild herg_ki and logd npz from cached CSVs | build | yes | not timed |
| `build_adme_solubility.py` | build the log-solubility npz | build | **no — the file is not on disk**, deleted in `df7c7fe`. `sim_adme_solubility.npz` exists and cannot be rebuilt from this directory | — |
| `build_moleculeace.py` | turn each MoleculeACE CSV into an npz | build | **no — `MoleculeACE` is not installed**, and it finds the CSVs by globbing inside that installed package | — |

## What the survey found before anything was run

**Four files EXPERIMENTS_FINALIZED.md documents are not on disk.**
`ranker_experiment.py`, `blacklist.py` and `build_adme_solubility.py` were each
deleted in a named commit. `winning_recipe_reselected.py` is different: `git log
--all` over that path returns nothing, so no commit in this repository has ever
carried a file by that name. The file the entry describes is
`winning_recipe_honest.py`.

**A result file outlives its script.** `ranker_experiment.json` and
`ranker_experiment_quick.json` sit in the directory, written 3 and 4 September.
The script that wrote them was deleted in `4ab874c`. Nothing on disk can
reproduce or update those two files.

**Two of the seven method experiments named in the brief cannot be run at all.**
`ranker_experiment` has no file. `sm_battery_moleculeace` has a file but no
data: `MoleculeACE` is not an installed package here, and `build_moleculeace.py`
locates its CSVs by globbing inside that package's install directory.

## Method experiments, run at full molecule count

Every number below is this session's run on whole datasets: herg_ki 1,415
molecules, logd 1,427 kept from 5,039 by whole scaffold groups,
adme_solubility 2,173. No rows were subsampled. The activity yardstick
throughout is `simeval.eval_knn_readacross`: five folds by Bemis-Murcko
scaffold group, each held-out molecule predicted from its five nearest
training molecules weighted by one over distance, one Spearman pooled over
every held-out molecule. Higher is better, unitless. The structural yardstick
is the Spearman between a pair's distance in the space and its
maximum-common-substructure distance, over 8,000 cached molecule pairs. Higher
is better, unitless.

### `winning_recipe_vs_standard.py` — the headline table. Wins structure 3 of 3, wins activity 1 of 3.

This is the only script in the group that fits its feature selection on one half
of the molecules and scores on the other. The molecules split by whole scaffold
group: herg_ki 709 select and 706 score, logd 738 and 689, adme_solubility by
the same rule. A cached pair counts only when both its molecules land on the
score half, so herg_ki scores structure on 2,005 of its 8,000 pairs and drops
4,025 that straddle.

The stacked space is ecfp4_count, the RDKit descriptors that survive a drop-one
selection, 100 atompair/torsion features and a 49-feature 3D pharmacophore
block, each block z-scored, divided by the square root of its own feature count
and concatenated, then Euclidean.

Activity Spearman on the score half, five nearest neighbours:

    space                            herg_ki    logd   adme_solubility
    ecfp4:tanimoto (field default)    0.5781  0.5339   0.2895
    ecfp4_count:tanimoto (baseline)   0.5802  0.5811   0.3439
    ecfp4_count, z-scored Euclidean   0.5789  0.4629   0.1786
    the stacked space                 0.6415  0.4315   0.3366

Structural Spearman on the score half's pairs:

    space                            herg_ki    logd   adme_solubility
    ecfp4:tanimoto (field default)    0.1796  0.4081   0.2483
    ecfp4_count:tanimoto (baseline)   0.1079  0.4170   0.2752
    ecfp4_count, z-scored Euclidean   0.2863  0.1857   0.3011
    the stacked space                 0.4389  0.5851   0.3823

**The script's third question has an answer and it is not the same on the three
datasets.** It asks whether a gain is the added blocks or the switch from
Tanimoto to z-scored Euclidean. On herg_ki the switch costs 0.5802 to 0.5789,
so the whole 0.5802 to 0.6415 is the added blocks. On logd the switch costs
0.5811 to 0.4629 and the stacked space ends at 0.4315, so the coefficient
change is most of the loss and the added blocks make it slightly worse still.
On adme_solubility the switch costs 0.3439 to 0.1786 and the stacked space
recovers to 0.3366, so the added blocks are worth 0.1580 Spearman there and the
coefficient is what sank it.

Wall clock: 377 s, 371 s, 354 s per dataset, on a machine at load average 210
where the process held 8% of one core.

### `creative_hybrids_v2.py` — the same stacking, scored on every molecule, and it disagrees with the held-out table on logd

This one fits the descriptor drop and the fine-2D selection on the same 8,000
pairs it then scores structure on, and scores activity on all n molecules.
Activity Spearman, delta against ecfp4_count Tanimoto in the same column:

    space                                     herg_ki           logd
    count_tanimoto (baseline)                  0.7087          0.6180
    count, z-scored Euclidean                  0.6641 -0.0446  0.5329 -0.0851
    count+descriptors (all 208)                0.7100 +0.0014  0.6293 +0.0113
    count+descriptors (drop-one survivors)     0.6901 -0.0185  0.5866 -0.0314
    count+3D pharmacophore                     0.7317 +0.0230  0.3515 -0.2665
    count+survivors+pharmacophore+fine2D       0.7333 +0.0247  0.5259 -0.0920

The drop-one selection keeps 63 of herg_ki's 208 descriptors and 68 of logd's.
It is chosen to raise the structural Spearman and it does: herg_ki 0.248 to
0.492, logd 0.383 to 0.675 on the pairs it fitted on. On activity it costs
0.0185 on herg_ki and 0.0314 on logd.

**The 3D pharmacophore block is the whole story and it points opposite ways.**
Adding it alone to the count fingerprint is the second-best space on herg_ki at
+0.0230 Spearman and the worst space on logd at -0.2665. Nothing else in the
table moves either dataset by more than 0.05.

### `nary_similarity.py` — the shipped setting is worse than turning the method off, on all three

The method multiplies every pair's distance by `1 + lam * mean outlierness of
the pair`, where outlierness is complementary similarity min-max scaled over
every molecule. At lam = 0 the matrix is the plain count-fingerprint Tanimoto,
so lam = 0 is the method switched off. The file ships lam = 1.0.

Activity Spearman at five neighbours, three datasets:

    lam                 herg_ki    logd   adme_solubility
    0 (method off)       0.7087  0.6180   0.3995
    1.0 (shipped)        0.7041  0.6137   0.3728
    8.0                  0.6863  0.6009   0.3242

Sweeping lam over -0.9, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8 and the neighbour
count over 3, 5, 7, 15, the best cell is lam = -0.50 at 3 neighbours on herg_ki
(0.7132 against 0.7027 at lam = 0 and the same neighbour count), lam = -0.90 at
7 neighbours on logd (0.6195 against 0.6181), and lam = 0 at 15 neighbours on
adme_solubility (0.4325, no lam improves it). **Where lam helps at all its sign
is negative, which shrinks distances toward outliers — the opposite of what the
method is for.** The whole lam axis moves the Spearman by less than 0.011 on
every dataset. The neighbour count moves adme_solubility from 0.3995 at five
neighbours to 0.4325 at fifteen, three times the largest lam effect.

The file's other test ranks molecules whose activity disagrees with their
neighbours. Spearman against that target: complementary similarity 0.1265,
0.0897, -0.2855 on the three datasets; mean distance to all molecules 0.1389,
0.0945, -0.2676; distance of the activity from the dataset median 0.0665,
0.2588, 0.5086. The method is beaten by mean distance on all three and by
activity extremeness on two.

Wall clock: 6.1 s herg_ki, 14.5 s logd, 14.7 s adme_solubility for one setting.
The whole 40-cell sweep took 173 s on adme_solubility.

### `aug3d_similarity.py` — E3FP and Coulomb eigenvalues lose to the count fingerprint on all three

Activity Spearman at five neighbours, against ecfp4_count Tanimoto in the first
row:

    metric                  herg_ki    logd   adme_solubility
    ecfp4_count:tanimoto      0.709   0.618   0.399
    e3fp:tanimoto             0.654   0.548   0.331
    e3fp:cosine               0.651   0.548   0.307
    e3fp:euclidean            0.101   0.315   0.044
    coulomb:cosine            0.556   0.556   0.083
    coulomb:euclidean         0.532   0.430   0.231

E3FP under z-scored Euclidean is the same collapse s11 described: 1,024 bits
whose median Euclidean distance sits far above the scale the neighbour search
expects, and the activity Spearman falls to 0.101 on herg_ki and 0.044 on
adme_solubility while the same bits under Tanimoto reach 0.654 and 0.331.

Wall clock: 39 s per dataset with the conformer cache warm.

### `similarity_noise_cleaning.py` — the structure-selected space does not find corrupted labels better than the count fingerprint

Corrupted labels are injected at three noise widths and three contaminated
shares, three injection seeds each. Detection is an AUROC: does a corrupted
molecule disagree with its neighbours in this space? Recovery is a correlation
between the correction the neighbours apply and the error that was injected.
Both on the held-out half of the molecules, with every fitted step fitted on the
other half.

    herg_ki, scored on 706 held-out molecules and 2,005 of its 8,000 pairs
      space                                      detect AUROC   recovery corr
      ecfp4_count Tanimoto (baseline)                  0.6490          0.7860
      count+descriptor survivors (coarse x0.5)         0.6491          0.7935
      count+survivors+pharmacophore+fine2D             0.6422          0.8002

    adme_solubility, scored on 1,087 held-out molecules and 2,022 pairs
      space                                      detect AUROC   recovery corr
      ecfp4_count Tanimoto (baseline)                  0.6640          0.8043
      count+descriptor survivors (equal)               0.6753          0.7966
      count+survivors+pharmacophore+fine2D             0.6673          0.8103

On herg_ki the structure-selected space loses detection by 0.0068 AUROC and
wins recovery by 0.0142. On adme_solubility it wins detection by 0.0033 AUROC
and recovery by 0.0061. Every one of those four numbers is smaller than the gap
between the two spaces inside the same column, so nothing here separates.

**The drop-one descriptor selection picks a different set on each half of the
molecules.** On herg_ki it keeps 55 of 208 RDKit descriptors when fitted on one
half and 64 of 208 when fitted on the other. On adme_solubility, 54 and 57.
Both halves are the same dataset and the same procedure.

Wall clock: 681 s herg_ki, 1,059 s adme_solubility.

**logd was not run for this one and it is the only script in the group that
would not have read the same molecules.** `similarity_noise_cleaning.run` calls
`load_prepared(dataset, cap=None)` at line 320, so logd arrives as all 5,039
molecules, split 2,541 fit and 2,498 held-out. Every other script in the
directory takes the default cap and gets 1,427. Both cite a user decision dated
2026-09-11 in their own comments. Its distance matrices are 5,039 by 5,039, and
it was stopped at 658 MB resident with the machine at 14.4 GB of 15.4 GB swap.

### `triplet_similarity.py` — the two arms of one method train on triplet counts 155 times apart

Learned distance on herg_ki, five folds by scaffold group, each fold's encoder
trained on its own training scaffolds and scored on the held-out ones. Activity
Spearman at five neighbours, held out:

    fold   count fingerprint   structural triplets   activity triplets
    0                 0.7977                0.7493              0.8410
    1                 0.6821                0.6720              0.7678
    2                 0.6718                0.6347              0.7085
    3                 0.7094                0.6872              0.7568
    4                 0.6445                0.5401              0.6890
    mean              0.7011                0.6567              0.7526

**The activity-defined arm beats the count fingerprint on all five held-out
folds and it is the only method in this group that wins everywhere it was
measured.** Its per-fold gains are 0.0433, 0.0857, 0.0367, 0.0474 and 0.0445
Spearman. The scaffold-defined arm loses on all five, by 0.0484, 0.0101, 0.0371,
0.0222 and 0.1044.

On the structural yardstick the order reverses in part: the scaffold-defined arm
beats the count fingerprint on four of five folds, the activity-defined arm on
three of five, and every one of those structural numbers rests on between 288
and 318 held-out pairs rather than 8,000.

The counts underneath those columns are the finding. A triplet survives when
the anchor's positive is closer than its negative by less than the margin, so a
pass with no surviving triplets teaches the encoder nothing. Per pass, out of
20,000 candidate triples drawn:

    arm                   surviving triplets per pass (min / median / max)
    structural positive             106 / 127 / 154     (fold 0)
    activity positive         19,730 / 19,759 / 19,788  (fold 0)

The structural arm defines a positive as a molecule sharing the anchor's
Bemis-Murcko scaffold. The activity arm defines it as the molecule with the
smaller absolute activity gap. The first finds a usable triple in 127 of 20,000
draws, the second in 19,759 of 20,000. Over 40 passes the structural arm trains
on 5,101 triples and the activity arm on 790,333. So the two arms differ in
what defines a positive and in how many examples the encoder sees, two ways at
once, and the file reports them as one comparison.

Wall clock: about 40 minutes per fold for all three arms, on a machine at load
average 300. All five folds completed.

## The conformer cache costs four hours once, and it is not a defect

`shape_similarity.py` prints `spatial cache has 1415 rows but records no
molecule list -- rebuilding`. `sim_herg_ki_spatial.npz` is on disk, dated 19
July, with the right number of rows. It carries no `smiles_id`, and
`build_spatial_reps` at shape_similarity.py:104 refuses a cache whose molecule
list cannot be checked rather than trusting the row count. That is the right
call and the comment in the file says so. Once the rebuild finishes it writes
`smiles_id` and every later run is a cache hit.

The cost of that one rebuild, measured: the 12-feature USR descriptor on
herg_ki's 1,415 molecules took 5,155 s, and USR is one of three descriptors,
each of which generates its own conformers. So herg_ki alone is roughly four
hours at load average 500, and the run was handed to the supervisor at
`/Volumes/seagate/kirby_simrun_logs/supervisor_jobs.tsv` under the name
`sm_shape_similarity` rather than abandoned.

`aug3d_similarity.py` prints `3d-aug cache hit` on all three datasets and
finishes a dataset in 39 to 47 s. Its cache was written after the molecule-list
field existed, so it is trusted.

## What is broken, in one place

| what | what is wrong |
|---|---|
| `ranker_experiment.py` | not on disk, deleted in `4ab874c`. Its two result files are still in the directory and nothing can regenerate them |
| `blacklist.py` | not on disk, deleted in `df7c7fe`. Its procedure now lives in `src/kirby/similarity/selection.select_saboteur_drop`, which `winning_recipe_honest.py` calls |
| `build_adme_solubility.py` | not on disk, deleted in `df7c7fe`. `sim_adme_solubility.npz` still works, but it cannot be rebuilt from this directory |
| `winning_recipe_reselected.py` | EXPERIMENTS_FINALIZED.md documents it. `git log --all` over that path returns nothing. The file it describes is `winning_recipe_honest.py` |
| `run_battery_moleculeace.py` and `build_moleculeace.py` | the `MoleculeACE` package is not installed here. The builder locates its CSVs by globbing inside that package's install directory, so there is nothing to build from and nothing to score |
| `feature_type_study.py` | no `if __name__` block, so it cannot be run. Its only caller is `tests/experiments/opig_scaffold_similarity.py`, outside this directory |
| `augmented_descriptor_reps.py` | `padelpy` is absent, so its PaDEL arm cannot be built. PDV and Mordred both build |
| `shape_similarity.py` | not broken. Its 19 July conformer cache predates the molecule-list field, so it is rebuilt once at roughly four hours per dataset; see above |

Nothing in this group was fixed by editing it. Every script that runs today ran
unmodified.

### `extra_fingerprints.py` — none of Avalon, MAP4, SECFP6 or MHFP6 beats the count fingerprint on activity, and three of them beat it on structure

Activity Spearman at five neighbours, five folds by scaffold group:

    metric                      herg_ki    logd
    ecfp4_count:tanimoto         0.7087  0.6180
    avalon:tanimoto              0.6786  0.5721
    avalon:cosine                0.6733  0.5729
    map4:tanimoto                0.5049  0.5497
    map4:cosine                  0.4042  0.5426
    secfp6:tanimoto              0.6940  0.6030
    secfp6:cosine                0.6945  0.6025
    mhfp6:minhash_jaccard        0.6872  0.5906

Structural Spearman over the same 8,000 cached pairs:

    metric                      herg_ki    logd
    ecfp4_count:tanimoto         0.0431  0.3661
    avalon:tanimoto             -0.0340  0.2216
    map4:tanimoto               -0.4079 -0.0507
    secfp6:tanimoto              0.1542  0.5280
    mhfp6:minhash_jaccard        0.1911  0.5256

**The two yardsticks disagree and they disagree the same way on both datasets.**
SECFP6 and MHFP6 beat the count fingerprint on structure by 0.11 to 0.16
Spearman and lose to it on activity by 0.014 to 0.027. Nothing here beats it on
activity at all.

MAP4 is the outlier in both directions. Its structural Spearman is -0.4079 on
herg_ki, so its pairwise distance runs backwards against shared substructure,
and its activity Spearman falls to 0.4042 under cosine against the count
fingerprint's 0.7087. MAP4 is a 1,024-slot MinHash folded to bits with a mean
of 514 bits set per molecule on herg_ki, against SECFP6's 75.

Wall clock per dataset: Avalon 97-149 s, MAP4 184-347 s, MHFP6 570-932 s.
