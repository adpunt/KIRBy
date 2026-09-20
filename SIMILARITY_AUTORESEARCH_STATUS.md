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
| s05 | method | yes, 3/3 | yes | **partial** — knobs added, never re-run with them |
| s06 | method | yes, 3/3 twice | yes | **none** |
| s08 | investigation | yes, 3/3 twice | no | n/a |
| s09 | investigation | **never run** | no | n/a |
| s10 | method | **partial** — has never completed a full run | yes | **partial** |
| s11 | method | yes, 3/3 at two budgets | yes | **yes** |
| s12 | method | yes, 3/3 twice | yes | **yes** |
| s13 | method | yes, 3/3 | yes | **partial** — retrieval swept, clustering untouched |

## Never opened

| group | scripts | note |
|---|---|---|
| s09 | 1 | reads s02/s04/s05/s06/s08, all of which now exist on disk |
| standalone | `similarity_optimized_hybrid`, `structural_geometry_survey` | — |
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

**s10 GP — a null, and a correction.** Multiplying features by the square root
of their importance destroys an RBF kernel: the weighted arm returned the
training mean on all three datasets (herg_ki label spread 0.9151, weighted arm
0.9160). With the arm alive, weighting makes the GP slightly worse, +0.0111 and
+0.0188 pKi on two folds of herg_ki/pdv.

**s10 q-RASAR — a null.** The four neighbour features carry real signal alone
(pdv's four reach 0.6730 pKi against 0.9160 for a mean predictor) and add
nothing to a forest that already has the features.

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

