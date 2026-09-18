# The 77 experiments, in three groups

Built from the per-experiment classification in `SELECTION_SHAPE_PLAN_2026_09_18.md`,
which carries a file and a line number for each one.

The shape is: for each downstream model, the top 2 base representations and the top 2
hybrids. Five downstream models gives 20 rows, fewer when the same representation is
top for two models.

| group | experiments |
|---|---|
| no change | 38 |
| a simple change | 6 |
| needs a decision from you | 33 |

---

## No change — 38

The shape does not apply. These are not comparing base representations against hybrids
across downstream models, so imposing it would replace what they measure.

One row is one experiment.

| experiment | what it compares instead |
|---|---|
| c01 | It profiles every representation one at a time -- intrinsic dimensionality, mutual information with the labels… |
| c04 | It measures whether feature-importance weights make a distance space better -- the same prediction of each mol… |
| c06 | It measures, within one model family, whether a variant's capacity index and the budget at which its score pea… |
| c08 | It measures conformal interval coverage and width — PICP, MPIW or prediction-set size, and calibration error —… |
| c09 | c09 compares one hybrid's selected features against a PCA rotation and a random feature subset, on mutual info… |
| c12 | c12 compares one omics assay's representation against two assays' representations placed side by side at full … |
| c13 | c13 scores one named embedding model at a time against a hybrid built over a set that includes it and against … |
| c15 | c15 measures the geometry of a feature space -- how far apart the pairwise distances between molecules spread,… |
| c16 | It compares regularisation strength against hybrid budget inside one model family, and both arms of its compar… |
| c17 | It compares five ways of choosing features out of the concatenation of every representation (c17:267 stacks th… |
| c19 | It compares five ways of feeding importance scores into training -- unweighted, feature_weighted, sample_weigh… |
| c20 | It compares one base representation against itself at different feature counts — 10, 25, 50, 100, 200, 300, 50… |
| c21 | It compares a base representation alone against that same representation plus extra features, and separately a… |
| c22 | It explains ONE hybrid — it traces each hybrid feature back to the representation it came from and reports eac… |
| c23 | It compares a hybrid built under an internal evaluator that matches the downstream model against the same down… |
| c24 | It compares a feature selection built on one set of folds against a selection built on a different set of fold… |
| c43_buchwald_signal | It scores one named representation against predicting the training-fold mean, to decide whether buchwald_hartw… |
| fed01 | It compares seven federated training protocols at six label-noise levels on the same molecules, and no arm in … |
| fed03 | Part A compares three curated data SOURCES (ChEMBL hERG, Lhasa training, Lhasa test) against a naive pool and … |
| fed04 | It asks whether the SAME hybrid recipe splits its 100-feature budget differently in different regions of chemi… |
| fed05 | It compares eight federated conditions across three antibody silos that hold different endpoints -- kcal/mol, … |
| n06 | It measures how well each noise estimator's per-molecule score agrees with real inter-lab disagreement between… |
| n09 | It ranks 12 learning-with-noisy-labels methods against each other (n09:852-864) under one fixed base represent… |
| n11 | Its representation axis is which features the SUSPICION SCORE is computed in, not which features a model is fi… |
| n12 | Its five conditions are five distance geometries used to find suspect labels — base unweighted, base importanc… |
| n14 | It counts activity cliffs: how many molecules sit in a pair that is structurally similar and more than 2 log u… |
| n15 | It compares two hybrids against each other -- the recipe a search found on clean labels against the recipe a s… |
| n16 | It ranks individual features by their importance value and measures how that ordering moves as the labels get … |
| n20 | It compares four arms that differ only in which label column the recipe search and the per-fold feature rankin… |
| n23 | It compares cleaned training sets against the uncleaned one — for each (cleaning arm, downstream model) it cou… |
| opig_scaffold_similarity | It correlates how far apart a representation puts two variants of one antibody against how far apart an indepe… |
| opig_shared_vectors_census | It counts how many molecules get a byte-for-byte identical row from a representation, splits that count by cau… |
| opig_shared_vectors_remedy | It takes ONE named representation and asks whether hybridising or augmenting it moves two numbers on the same … |
| s01 | For each representation it compares pairwise distances between molecules against the activity differences of t… |
| s03 | It measures how much mutual information the selected features carry against the full representation, the same … |
| s09 | It loads the stored JSON of s02, s04, s05, s06 and s08 and runs a signed-rank test down a column of one number… |
| s11 | It runs a Bayesian-optimization screening loop over 11 feature spaces and scores each one by the area under it… |
| s13 | It compares each space by k-means label coherence and by k-nearest-neighbour retrieval cost on held-out rows, … |

---

## A simple change — 6

Every number printed today keeps its value. A second base representation appears beside
the first, or a second hybrid beside the first. Nothing already on disk is superseded.
All six cost 0 new model fits.

| experiment | where | what it gains |
|---|---|---|
| c03 | `c03:164-165` | names a top-2 base representation list per downstream model, which it names none of today |
| c07 | `c07:288` | its `top3_base_reps` becomes 2 representations for each of the 5 downstream models instead of 3 for rf alone |
| c31 | `c31:522` | every arm is measured against the rank-1 base representation; the rank-2 one appears beside it |
| c32 | `c32:391` | the same, across all 64 strategy-scorings per downstream model |
| c33 | `c33:523` | `delta_vs_best_single` is against the rank-1 representation; the rank-2 one appears beside it |
| opig_scaffold_hybrid | `opig_scaffold_hybrid.py:283` | 2 base representations printed per downstream model instead of 1, so 10 rows over the 5 models a full-speed run scores |

---

## Needs a decision from you — 33

Not as 33 separate conversations. They raise six questions, and an experiment can be
under more than one.

### 1. When something counts wins, does the hybrid get one entry or two?

These count how often the hybrid beats the representations. Give it 2 entries instead of 1 and the count rises with nothing measured changing. Either the hybrid keeps 1 entry in the count and 2 in the table, or the count is redefined as a rate.

*7 experiments:* c02, c14, n02, n13, opig_activity_cliff_qsar, opig_blueprint_transfer, opig_ginkgo_developability

### 2. When a number is averaged over representations, over how many?

These average a delta across every representation. Averaging over the top 2 is a different quantity, not a cheaper version of the same one.

*6 experiments:* fed02, n21, opig_ginkgo_developability, opig_graphinity_ddg, opig_humanness, s10

### 3. The honest re-pick: pick from 2 candidates, or from all of them?

These pick the winning representation per held-out fold from the other folds. With 2 candidates there is less to pick from, so the baseline drops a little. It drops because there is less selection optimism to strip, which is the point of the re-pick — so this may be a better number, not a worse one. It still moves.

*17 experiments:* c02, c05, c10, c11, c14, fed02, opig_cdr_flexibility, opig_cgr_comparison, opig_ginkgo_developability, opig_graphinity_ddg, opig_humanness, opig_kingsmith_component_splits, opig_trastuzumab_enrichment, s02, s04, s05, s12

### 4. Ties. Today the winner is whichever was built first.

Baseline scores are rounded to 4 decimals, so exact ties happen. Today a tie is broken by the order representations were built in; the selector breaks it by name. Where a tie flips, the representation named in the output changes and its score does not.

*9 experiments:* c02, c03, c05, n19, n22, opig_blueprint_transfer, opig_cgr_comparison, opig_ginkgo_developability, s06

### 5. Six of them publish a number another file reads.

s09 loads the stored results of s02, s04, s05, s06 and s08 and runs a signed-rank test down a column. Changing any of those five means re-running s09, and mixing an old and a new file in that test compares two different quantities.

*6 experiments:* opig_humanness, s02, s04, s05, s06, s08

### 6. Five cost new model fits rather than zero.

Every other experiment in this document reads stored rows, so adopting the shape costs 0 new model fits. These five have to fit something.

*5 experiments:* c18, c43_buchwald_step3, opig_ginkgo_developability, opig_kingsmith_component_splits, opig_trastuzumab_enrichment

### 7. Four of the noise experiments would stop reporting representations they report today.

These sweep every representation the run loaded and report all of them. Under the shape they
report 2 per downstream model, and a representation outside every model's top 2 stops being
reported. n17 removed a representation cut deliberately once before, so narrowing it back is
a decision, not a tidy-up. n18 goes further: its base space today is whichever representation
sorts first alphabetically, so the shape does not narrow that arm, it replaces the matrix
under every number the file reports.

*4 experiments:* n01, n07, n17, n18

---

## What I would do

Take the six in the simple group now. They cost 0 new model fits, nothing on disk is
superseded, and they answer the ask for c03, c07, c31, c32, c33 and opig_scaffold_hybrid.

Then question 1, on its own. Seven experiments count wins, and the fix is the same in all
seven: the hybrid keeps 1 entry in the count and gets 2 in the table. One decision, seven
files.

Question 5 is the one to settle before touching s02, s04, s05, s06 or s08, because s09 reads
all five and a mixed set of files makes its signed-rank test compare two quantities.
