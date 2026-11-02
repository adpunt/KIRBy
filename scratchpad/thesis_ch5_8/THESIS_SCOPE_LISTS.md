# What exists, per dataset and per noise method (2026-10-09)

Written by Claude on 2026-10-09 for the author to pick from. Nothing here is a selection.
Patient data (TCGA, ANNITIA, NHANES) is left out of every list.

Sources:
- Experiments per dataset: result files under `/Volumes/seagate/kirby_results_2026-10-09/{complexity,similarity,federated,noise}/`. A dataset counts for an experiment if it has a result file that is not a skip record. Promoted runs (`noise_full`) and OPIG files are not counted.
- Noise methods: the "does it work?" column of `/Volumes/seagate/claude_analysis_2026-10-08/answers/MEETING_METHODS.md`, counted row by row. One row is one dataset, one noise type and one model. Fold 0. Each method uses its best setting, picked on the held-out score, as on the page.
- The group, selector and action columns are Claude's reading of the code (`scratchpad/thesis_ch5_8/CH7_METHODS.md`). The page's own group column is not used.

## 1. Every dataset: how many experiments it has results for

One row is one dataset. "exps" counts experiments with a result file. The last two columns count noise methods on MEETING_METHODS.md.

| dataset | domain | complexity exps | similarity exps | federated exps | noise exps | noise methods on the page: with a yes row / tested | yes rows / rows |
|---|---|---|---|---|---|---|---|
| caco2 | small molecule | 20 | 10 | 4 | 30 | 66 / 77 | 333 / 742 |
| herg | small molecule | 15 | 10 | 0 | 15 | 26 / 34 | 176 / 380 |
| herg_ki | small molecule | 20 | 11 | 0 | 30 | 66 / 76 | 284 / 655 |
| herg_ic50 | small molecule | 0 | 0 | 0 | 6 | 8 / 9 | 13 / 30 |
| logd | small molecule | 20 | 10 | 4 | 29 | 57 / 73 | 242 / 528 |
| qm9 | small molecule | 15 | 10 | 0 | 13 | 13 / 21 | 20 / 82 |
| adme_hlm | small molecule | 5 | 0 | 0 | 0 | - | - |
| adme_mdr1 | small molecule | 15 | 0 | 0 | 0 | - | - |
| adme_rlm | small molecule | 9 | 0 | 0 | 0 | - | - |
| buchwald_hartwig | reaction | 20 | 10 | 0 | 24 | 46 / 58 | 199 / 390 |
| bh_doyle_plates | reaction | 9 | 0 | 0 | 0 | - | - |
| abagym_Ang2 | antibody | 10 | 0 | 0 | 0 | - | - |
| abagym_EGFR | antibody | 20 | 9 | 0 | 0 | - | - |
| abagym_VEGF | antibody | 16 | 0 | 0 | 0 | - | - |
| abcov_kd | antibody | 18 | 9 | 0 | 5 | 7 / 11 | 17 / 40 |
| cdr_flexibility | antibody | 14 | 9 | 0 | 13 | 21 / 24 | 99 / 246 |
| ddg_combined | antibody | 20 | 9 | 0 | 12 | 32 / 41 | 106 / 264 |
| flab_binding_koenig | antibody | 7 | 0 | 0 | 0 | - | - |
| flab_expression_koenig | antibody | 16 | 3 | 0 | 0 | - | - |
| flab_thermo_tresanco | antibody | 20 | 9 | 0 | 7 | 15 / 21 | 29 / 62 |
| ginkgo_acsins | antibody | 17 | 9 | 0 | 7 | 16 / 23 | 35 / 85 |
| ginkgo_colloidal | antibody | 18 | 9 | 0 | 22 | 45 / 52 | 163 / 368 |
| ginkgo_hic | antibody | 10 | 0 | 0 | 0 | - | - |
| ginkgo_pr_cho | antibody | 17 | 9 | 0 | 6 | 13 / 14 | 34 / 62 |
| ginkgo_titer | antibody | 18 | 9 | 0 | 12 | 32 / 38 | 108 / 194 |
| humanness_therapeutic | antibody | 15 | 9 | 0 | 6 | 6 / 19 | 30 / 126 |
| tz_her2_binary | antibody | 7 | 0 | 0 | 0 | - | - |
| tz_her2_multiclass | antibody | 8 | 0 | 0 | 0 | - | - |
| nt_promoter_tata | DNA sequence | 2 | 0 | 0 | 0 | - | - |
The outline (5.x, "Cut, not in the thesis") lists bh_doyle_plates, tz_her2_binary, tz_her2_multiclass and nt_promoter_tata as cut, although they have result files. Datasets with results in no category: none. Experiments not counted above: c03, c14, c20 and s09 (no result folder, or they read other experiments' results); c12 (patient data, excluded).

## 2. Every dataset: which experiments

The code in brackets is the result folder, for finding the file.

### caco2 (small molecule)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (10): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); activity cliffs (ECFP4) (s04); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- federated (4): noise differs between sites (fed01); sites hold different representations (fed02); synthetic private data (fed03); allocation by chemical cluster (fed04)
- noise (30): score drop when labels are damaged (c10); error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); activity-cliff landscape (n14); hybrid searched on clean vs damaged labels (n15); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); model-uncertainty detectors (n23); delete or keep, vs random deletion (n24); putting cliff molecules back (n25); removal cost (n27); clipped labels (n30); residual detector quality (n31); trimming, two learners, Shapley (n32); robust losses and label transforms (n33); scaffold-group offset (n34); near-duplicate pairs (n35); how much error is label noise (n37); noise sized by model error (n38); simulated multiple sources (n39); changing the model instead (n40); conformal flags (n41); methods x similarity settings (n42); extrapolating to zero noise (noise)

### herg (small molecule)
- complexity (15): profile of each representation (c01); model x budget grid (c02); learning curves (c05); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (10): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); activity cliffs (ECFP4) (s04); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (15): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); detector x action grid (antibody) (n11); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); model-uncertainty detectors (n23); delete or keep, vs random deletion (n24); putting cliff molecules back (n25); methods x similarity settings (n42); extrapolating to zero noise (noise)

### herg_ki (small molecule)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (11): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); activity cliffs (ECFP4) (s04); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); Bayesian optimisation (s11); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (30): score drop when labels are damaged (c10); error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); activity-cliff landscape (n14); hybrid searched on clean vs damaged labels (n15); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); model-uncertainty detectors (n23); delete or keep, vs random deletion (n24); putting cliff molecules back (n25); removal cost (n27); clipped labels (n30); residual detector quality (n31); trimming, two learners, Shapley (n32); robust losses and label transforms (n33); scaffold-group offset (n34); near-duplicate pairs (n35); how much error is label noise (n37); noise sized by model error (n38); simulated multiple sources (n39); changing the model instead (n40); conformal flags (n41); methods x similarity settings (n42); extrapolating to zero noise (noise)

### herg_ic50 (small molecule)
- noise (6): replicate removal (repeats) (n17); measurement count (n19); noise size from repeats (n26); per-document bias (n29); which molecules to remeasure (n36); extrapolating to zero noise (noise)

### logd (small molecule)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (10): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); activity cliffs (ECFP4) (s04); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- federated (4): noise differs between sites (fed01); sites hold different representations (fed02); synthetic private data (fed03); allocation by chemical cluster (fed04)
- noise (29): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); activity-cliff landscape (n14); hybrid searched on clean vs damaged labels (n15); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); model-uncertainty detectors (n23); delete or keep, vs random deletion (n24); putting cliff molecules back (n25); removal cost (n27); clipped labels (n30); residual detector quality (n31); trimming, two learners, Shapley (n32); robust losses and label transforms (n33); scaffold-group offset (n34); near-duplicate pairs (n35); how much error is label noise (n37); noise sized by model error (n38); simulated multiple sources (n39); changing the model instead (n40); conformal flags (n41); methods x similarity settings (n42); extrapolating to zero noise (noise)

### qm9 (small molecule)
- complexity (15): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); theory checks (double descent etc.) (c15); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23)
- similarity (10): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); activity cliffs (ECFP4) (s04); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (13): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); importance under noise (n16); cleaning the labels as delivered (n18); which distance for neighbours (n21); delete or keep, vs random deletion (n24); removal cost (n27); clipped labels (n30); extrapolating to zero noise (noise)

### adme_hlm (small molecule)
- complexity (5): profile of each representation (c01); model x budget grid (c02); uncertainty calibration (c08); budget as regularisation (c16); explainability (c22)

### adme_mdr1 (small molecule)
- complexity (15): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22)

### adme_rlm (small molecule)
- complexity (9): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); diversity against depth (c18); importance weights in training (c19); explainability (c22)

### buchwald_hartwig (reaction)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (10): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); Bayesian optimisation (s11); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (24): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); detector x action grid (antibody) (n11); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); hybrid searched on clean vs damaged labels (n15); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); model-uncertainty detectors (n23); delete or keep, vs random deletion (n24); removal cost (n27); clipped labels (n30); trimming, two learners, Shapley (n32); robust losses and label transforms (n33); near-duplicate pairs (n35); how much error is label noise (n37); noise sized by model error (n38); simulated multiple sources (n39); conformal flags (n41); extrapolating to zero noise (noise)

### bh_doyle_plates (reaction)
- complexity (9): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); uncertainty calibration (c08); ensemble against hybrid (c11); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22)

### abagym_Ang2 (antibody)
- complexity (10): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22)

### abagym_EGFR (antibody)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)

### abagym_VEGF (antibody)
- complexity (16): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); blueprint transfer across folds (c24)

### abcov_kd (antibody)
- complexity (18): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (5): neighbour-agreement weights (n02); neighbour weights, importance-scaled features (n12); cleaning the labels as delivered (n18); noise sized by model error (n38); extrapolating to zero noise (noise)

### cdr_flexibility (antibody)
- complexity (14): profile of each representation (c01); model x budget grid (c02); learning curves (c05); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (13): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); detector x action grid (antibody) (n11); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); hybrid searched on clean vs damaged labels (n15); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); methods x similarity settings (n42); extrapolating to zero noise (noise)

### ddg_combined (antibody)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (12): score drop when labels are damaged (c10); neighbour-agreement weights (n02); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); cleaning the labels as delivered (n18); which distance for neighbours (n21); clipped labels (n30); residual detector quality (n31); trimming, two learners, Shapley (n32); near-duplicate pairs (n35); noise sized by model error (n38); extrapolating to zero noise (noise)

### flab_binding_koenig (antibody)
- complexity (7): profile of each representation (c01); model x budget grid (c02); capacity within a model family (c06); optimum by scaffold subpopulation (c07); diversity against depth (c18); importance weights in training (c19); explainability (c22)

### flab_expression_koenig (antibody)
- complexity (16): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); ensemble against hybrid (c11); adding a new representation (c13); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24)
- similarity (3): standard distances against activity (s01); feature concentration (s03); activity cliffs, all domains (s12)

### flab_thermo_tresanco (antibody)
- complexity (20): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); learning curves (c05); capacity within a model family (c06); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (7): neighbour-agreement weights (n02); neighbour weights, importance-scaled features (n12); cleaning the labels as delivered (n18); clipped labels (n30); trimming, two learners, Shapley (n32); near-duplicate pairs (n35); extrapolating to zero noise (noise)

### ginkgo_acsins (antibody)
- complexity (17): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (7): error on cliff molecules (n01); neighbour-agreement weights (n02); the twelve published methods (n09); neighbour weights, importance-scaled features (n12); cleaning the labels as delivered (n18); noise sized by model error (n38); extrapolating to zero noise (noise)

### ginkgo_colloidal (antibody)
- complexity (18): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (22): score drop when labels are damaged (c10); error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); detector x action grid (antibody) (n11); neighbour weights, importance-scaled features (n12); twelve methods + detector weights, other damage kinds (n13); hybrid searched on clean vs damaged labels (n15); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); which distance for neighbours (n21); removal cost (n27); clipped labels (n30); residual detector quality (n31); trimming, two learners, Shapley (n32); near-duplicate pairs (n35); how much error is label noise (n37); noise sized by model error (n38); simulated multiple sources (n39); extrapolating to zero noise (noise)

### ginkgo_hic (antibody)
- complexity (10): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22)

### ginkgo_pr_cho (antibody)
- complexity (17): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (6): score drop when labels are damaged (c10); error on cliff molecules (n01); neighbour-agreement weights (n02); the twelve published methods (n09); detector x action grid (antibody) (n11); extrapolating to zero noise (noise)

### ginkgo_titer (antibody)
- complexity (18): profile of each representation (c01); model x budget grid (c02); importance as a distance (c04); capacity within a model family (c06); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); theory checks (double descent etc.) (c15); budget as regularisation (c16); information-theoretic selection (c17); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (12): score drop when labels are damaged (c10); error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action grid (antibody) (n11); neighbour weights, importance-scaled features (n12); importance under noise (n16); cleaning the labels as delivered (n18); cleaning before the feature search (n20); trimming, two learners, Shapley (n32); near-duplicate pairs (n35); noise sized by model error (n38); extrapolating to zero noise (noise)

### humanness_therapeutic (antibody)
- complexity (15): profile of each representation (c01); model x budget grid (c02); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); variance decomposition (ANOVA etc.) (c09); ensemble against hybrid (c11); adding a new representation (c13); budget as regularisation (c16); diversity against depth (c18); importance weights in training (c19); one representation augmented (c21); explainability (c22); evaluator matched to model (c23); blueprint transfer across folds (c24); budgets above the molecule count (c33)
- similarity (9): standard distances against activity (s01); hybrid distance against activity (s02); feature concentration (s03); predicting from five nearest neighbours (s05); applicability domain (s06); dimension for similarity vs prediction (s08); importance-weighted features for distance models (s10); activity cliffs, all domains (s12); clustering and retrieval (s13)
- noise (6): error on cliff molecules (n01); neighbour-agreement weights (n02); detector x action, damaged labels (n07); the twelve published methods (n09); cleaning the labels as delivered (n18); extrapolating to zero noise (noise)

### tz_her2_binary (antibody)
- complexity (7): profile of each representation (c01); model x budget grid (c02); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); adding a new representation (c13); diversity against depth (c18); importance weights in training (c19)

### tz_her2_multiclass (antibody)
- complexity (8): profile of each representation (c01); model x budget grid (c02); optimum by scaffold subpopulation (c07); uncertainty calibration (c08); ensemble against hybrid (c11); diversity against depth (c18); importance weights in training (c19); explainability (c22)

### nt_promoter_tata (DNA sequence)
- complexity (2): diversity against depth (c18); explainability (c22)

## 3. Every noise method on MEETING_METHODS.md, by group and action

One row is one method as named on the page. "yes rows / rows" counts rows whose "does it work?" is yes. Methods with 0 yes rows: cliff-aware weights (0 of 28 rows) and the calibrated measurement-count move (0 of 4).

#### noise mitigation — best of 9 (delete, count less, move)

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| search and fit on cleaned labels | Cleaning before the representation search (n20) | detector or flag | 43/48 | buchwald_hartwig 2/2 caco2 12/12 ddg_combined 4/4 ginkgo_colloidal 8/8 ginkgo_titer 4/4 herg 1/2 herg_ki 10/12 logd 2/2 | cdr_flexibility |

#### noise mitigation — count less

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| downweight | Other cleanups of the real labels (n18) | detector | 8/25 | abcov_kd 1/2 caco2 1/2 cdr_flexibility 1/2 ddg_combined 1/2 ginkgo_acsins 1/1 ginkgo_colloidal 1/2 herg 1/2 logd 1/2 | buchwald_hartwig flab_thermo_tresanco ginkgo_titer herg_ki humanness_therapeutic |
| unscaled features | Neighbour suspicion weights with importance-scaled features (n12) | detector | 77/129 | abcov_kd 4/10 buchwald_hartwig 6/10 caco2 6/14 cdr_flexibility 13/14 ddg_combined 6/10 flab_thermo_tresanco 4/8 ginkgo_acsins 5/5 ginkgo_colloidal 4/10 ginkgo_titer 7/10 herg 14/14 herg_ki 4/12 logd 4/12 |  |
| importance-scaled features | Neighbour suspicion weights with importance-scaled features (n12) | detector | 84/129 | abcov_kd 7/10 buchwald_hartwig 6/10 caco2 6/14 cdr_flexibility 13/14 ddg_combined 6/10 flab_thermo_tresanco 6/8 ginkgo_acsins 5/5 ginkgo_colloidal 4/10 ginkgo_titer 8/10 herg 14/14 herg_ki 5/12 logd 4/12 |  |
| neighbour-agreement weights | Down-weighting molecules that disagree with their neighbours (n02) | detector | 16/28 | abcov_kd 2/2 buchwald_hartwig 2/2 caco2 1/2 cdr_flexibility 2/2 flab_thermo_tresanco 2/2 ginkgo_acsins 1/1 ginkgo_colloidal 1/2 ginkgo_pr_cho 1/1 ginkgo_titer 1/2 herg 2/2 qm9 1/2 | ddg_combined herg_ki humanness_therapeutic logd |
| cliff-aware weights | Down-weighting molecules that disagree with their neighbours (n02) | detector | 0/28 |  | abcov_kd buchwald_hartwig caco2 cdr_flexibility ddg_combined flab_thermo_tresanco ginkgo_acsins ginkgo_colloidal ginkgo_pr_cho ginkgo_titer herg herg_ki humanness_therapeutic logd qm9 |
| downweight | Other cleanups of damaged labels (n07) | detector | 45/106 | buchwald_hartwig 3/10 caco2 6/14 cdr_flexibility 5/14 ginkgo_colloidal 2/10 ginkgo_titer 5/8 herg 10/14 herg_ki 4/14 humanness_therapeutic 6/14 logd 4/8 |  |
| detector weights | Refitting with weights from a noise detector (n13) | detector | 69/100 | buchwald_hartwig 7/10 caco2 7/14 cdr_flexibility 10/14 ddg_combined 8/10 ginkgo_colloidal 9/10 herg 11/14 herg_ki 7/12 logd 8/12 qm9 2/4 |  |
| best distance | Neighbour suspicion weights: which distance (n21) | detector | 61/74 | buchwald_hartwig 8/8 caco2 6/10 cdr_flexibility 9/10 ddg_combined 7/8 ginkgo_colloidal 6/8 herg 14/14 herg_ki 7/8 logd 4/8 |  |
| downweight_pairs | Down-weighting disagreeing near-duplicate pairs (n35) | detector | 21/32 | buchwald_hartwig 3/4 caco2 3/4 ddg_combined 4/4 flab_thermo_tresanco 1/4 ginkgo_colloidal 2/4 ginkgo_titer 3/4 herg_ki 2/4 logd 3/4 |  |
| downweight | Delete or keep the flagged molecules (n24) | detector | 27/48 | buchwald_hartwig 3/8 caco2 4/10 herg 13/14 herg_ki 3/8 logd 4/8 |  |
| huber_weight | Delete or keep the flagged molecules (n24) | detector | 28/48 | buchwald_hartwig 4/8 caco2 4/10 herg 13/14 herg_ki 3/8 logd 4/8 |  |
| downweight | Acting on a residual detector's flags (n31) | detector | 18/30 | caco2 6/10 ddg_combined 8/10 herg_ki 4/10 |  |
| weights_n19_fitted | Several sources per molecule: offsets and weights (n39) | known flag | 25/54 | buchwald_hartwig 9/12 caco2 6/12 ginkgo_colloidal 2/12 herg_ki 4/12 logd 4/6 |  |
| weights_within_var | Several sources per molecule: offsets and weights (n39) | known flag | 22/54 | buchwald_hartwig 5/12 caco2 6/12 ginkgo_colloidal 4/12 herg_ki 3/12 logd 4/6 |  |
| count_weighted | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 2/4 | herg_ic50 2/4 |  |
| spread_pooled_weighted | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 2/4 | herg_ic50 2/4 |  |

#### noise mitigation — count less or move

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| best repair | Detector and repair grid (n11) | detector | 10/10 | ginkgo_colloidal 6/6 ginkgo_titer 4/4 |  |

#### noise mitigation — delete

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| small_loss | Other noisy-label training methods for the neural network (n09, n13) | detector | 26/122 | buchwald_hartwig 9/10 caco2 2/14 cdr_flexibility 1/14 ddg_combined 1/10 ginkgo_colloidal 4/10 ginkgo_pr_cho 3/5 ginkgo_titer 4/5 herg_ki 1/12 humanness_therapeutic 1/6 | ginkgo_acsins herg logd qm9 |
| confident_learning | Other noisy-label training methods for the neural network (n09, n13) | detector | 41/122 | buchwald_hartwig 2/10 caco2 5/14 cdr_flexibility 7/14 ddg_combined 1/10 ginkgo_acsins 2/5 ginkgo_colloidal 4/10 ginkgo_pr_cho 2/5 ginkgo_titer 1/5 herg 3/14 herg_ki 6/12 logd 8/12 | humanness_therapeutic qm9 |
| distance_selection | Other noisy-label training methods for the neural network (n09, n13) | detector | 23/72 | buchwald_hartwig 4/5 caco2 3/7 cdr_flexibility 3/7 ddg_combined 1/5 ginkgo_acsins 1/5 ginkgo_colloidal 3/5 ginkgo_pr_cho 2/5 ginkgo_titer 2/5 herg 1/7 herg_ki 2/6 logd 1/6 | humanness_therapeutic qm9 |
| multistage_cleaning | Other noisy-label training methods for the neural network (n09, n13) | detector | 15/122 | buchwald_hartwig 4/10 cdr_flexibility 5/14 ginkgo_colloidal 2/10 ginkgo_pr_cho 2/5 ginkgo_titer 1/5 humanness_therapeutic 1/6 | caco2 ddg_combined ginkgo_acsins herg herg_ki logd qm9 |
| remove | Other cleanups of damaged labels (n07) | detector | 51/106 | buchwald_hartwig 6/10 caco2 4/14 cdr_flexibility 3/14 ginkgo_colloidal 6/10 ginkgo_titer 7/8 herg 11/14 herg_ki 3/14 humanness_therapeutic 10/14 logd 1/8 |  |
| two_learner | Trimmed and two-learner selection, and shrinking flagged labels (n32) | detector | 15/48 | buchwald_hartwig 6/8 ddg_combined 3/8 flab_thermo_tresanco 1/2 ginkgo_colloidal 2/8 ginkgo_titer 3/6 | caco2 herg_ki logd |
| delete | Delete or keep the flagged molecules (n24) | detector | 21/48 | buchwald_hartwig 7/8 caco2 2/10 herg 11/14 herg_ki 1/8 | logd |
| Detect and remove (checked against random removal) | Detect and remove (checked against random removal) | detector | 6/26 | cdr_flexibility 1/2 flab_thermo_tresanco 1/2 ginkgo_colloidal 2/2 ginkgo_titer 2/2 | abcov_kd buchwald_hartwig caco2 ddg_combined ginkgo_acsins herg herg_ki humanness_therapeutic logd |
| remove | Acting on a residual detector's flags (n31) | detector | 12/30 | caco2 2/10 ddg_combined 6/10 herg_ki 4/10 |  |
| drop_clipped | Other treatments of labels clipped at an assay limit (n30) | known flag | 5/22 | caco2 1/4 ddg_combined 1/2 flab_thermo_tresanco 1/2 ginkgo_colloidal 1/4 logd 1/4 | buchwald_hartwig herg_ki qm9 |

#### noise mitigation — delete or move

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| gaussian_process | Cleaning by a model's own per-molecule uncertainty (n23) | detector | 36/52 | caco2 9/14 herg 12/14 herg_ki 8/12 logd 7/12 |  |
| hetero_gp | Cleaning by a model's own per-molecule uncertainty (n23) | detector | 32/52 | caco2 11/14 herg 2/14 herg_ki 10/12 logd 9/12 |  |
| heteroscedastic_nn | Cleaning by a model's own per-molecule uncertainty (n23) | detector | 31/52 | caco2 12/14 herg 3/14 herg_ki 9/12 logd 7/12 |  |
| ngboost | Cleaning by a model's own per-molecule uncertainty (n23) | detector | 26/52 | caco2 8/14 herg 1/14 herg_ki 10/12 logd 7/12 |  |
| activity_cliff_aware | Cleaning by a model's own per-molecule uncertainty (n23) | detector | 38/52 | caco2 9/14 herg 11/14 herg_ki 9/12 logd 9/12 |  |

#### noise mitigation — move the label

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| label_smooth | Other cleanups of the real labels (n18) | detector | 7/26 | abcov_kd 1/2 cdr_flexibility 1/2 ddg_combined 1/2 ginkgo_colloidal 1/2 herg 1/2 herg_ki 1/2 humanness_therapeutic 1/2 | buchwald_hartwig caco2 flab_thermo_tresanco ginkgo_acsins ginkgo_titer logd |
| knn_smooth | Other cleanups of the real labels (n18) | detector | 4/20 | buchwald_hartwig 1/2 ddg_combined 1/2 ginkgo_acsins 2/2 | abcov_kd caco2 flab_thermo_tresanco ginkgo_colloidal ginkgo_titer herg_ki logd |
| oof_smooth | Other cleanups of the real labels (n18) | detector | 5/20 | abcov_kd 1/2 caco2 1/2 flab_thermo_tresanco 1/2 ginkgo_colloidal 1/2 logd 1/2 | buchwald_hartwig ddg_combined ginkgo_acsins ginkgo_titer herg_ki |
| gp_smooth | Other cleanups of the real labels (n18) | detector | 4/12 | ddg_combined 1/2 ginkgo_colloidal 2/2 ginkgo_titer 1/2 | abcov_kd flab_thermo_tresanco ginkgo_acsins |
| label_smooth | Other cleanups of damaged labels (n07) | detector | 48/106 | buchwald_hartwig 4/10 caco2 8/14 cdr_flexibility 1/14 ginkgo_colloidal 1/10 ginkgo_titer 3/8 herg 10/14 herg_ki 8/14 humanness_therapeutic 11/14 logd 2/8 |  |
| oof_smooth | Other cleanups of damaged labels (n07) | detector | 26/64 | caco2 7/14 ginkgo_colloidal 3/10 ginkgo_titer 5/8 herg_ki 8/14 logd 3/8 | buchwald_hartwig |
| gp_smooth | Other cleanups of damaged labels (n07) | detector | 9/18 | ginkgo_colloidal 4/10 ginkgo_titer 5/8 |  |
| js_shrink | Other cleanups of damaged labels (n07) | detector | 27/64 | caco2 9/14 ginkgo_colloidal 2/10 ginkgo_titer 3/8 herg_ki 9/14 logd 4/8 | buchwald_hartwig |
| shrink, one weight | Trimmed and two-learner selection, and shrinking flagged labels (n32) | detector | 20/48 | buchwald_hartwig 5/8 ddg_combined 4/8 flab_thermo_tresanco 1/2 ginkgo_colloidal 4/8 ginkgo_titer 5/6 herg_ki 1/4 | caco2 logd |
| shrink, weight per molecule | Trimmed and two-learner selection, and shrinking flagged labels (n32) | detector | 19/48 | buchwald_hartwig 5/8 caco2 1/8 ddg_combined 4/8 flab_thermo_tresanco 1/2 ginkgo_colloidal 4/8 ginkgo_titer 3/6 herg_ki 1/4 | logd |
| Label smoothing toward the nearest neighbours | Label smoothing toward the nearest neighbours | detector | 30/48 | buchwald_hartwig 3/8 caco2 6/10 herg 12/14 herg_ki 4/8 logd 5/8 |  |
| kNN label replacement | kNN label replacement | detector | 16/106 | buchwald_hartwig 4/10 ginkgo_colloidal 2/10 ginkgo_titer 7/8 herg_ki 1/14 logd 2/8 | caco2 cdr_flexibility herg humanness_therapeutic |
| knn_smooth | Delete or keep the flagged molecules (n24) | detector | 13/34 | buchwald_hartwig 6/8 caco2 2/10 herg_ki 3/8 logd 2/8 |  |
| oof_smooth | Delete or keep the flagged molecules (n24) | detector | 16/34 | buchwald_hartwig 4/8 caco2 4/10 herg_ki 4/8 logd 4/8 |  |
| rewrite | Acting on a residual detector's flags (n31) | detector | 10/30 | caco2 1/10 ddg_combined 8/10 herg_ki 1/10 |  |
| bayes_shrink | Acting on a residual detector's flags (n31) | detector | 10/30 | caco2 2/10 ddg_combined 7/10 herg_ki 1/10 |  |
| winsor | Other treatments of labels clipped at an assay limit (n30) | detector (residual band) | 10/32 | buchwald_hartwig 1/4 caco2 1/4 ddg_combined 1/4 flab_thermo_tresanco 3/4 ginkgo_colloidal 2/4 herg_ki 2/4 | logd qm9 |
| yeo_johnson | Changing how the model reads the labels (n33) | every molecule | 15/32 | buchwald_hartwig 2/2 caco2 8/12 herg_ki 3/12 logd 2/6 |  |
| huber_teacher_blend | Changing how the model reads the labels (n33) | every molecule | 19/32 | buchwald_hartwig 1/2 caco2 6/12 herg_ki 8/12 logd 4/6 |  |
| toward_prediction | Moving training labels toward an out-of-fold prediction (n27) | every molecule | 3/10 | buchwald_hartwig 1/2 ginkgo_colloidal 1/2 herg_ki 1/2 | caco2 logd |
| Constant lift | Other cleanups of the real labels (n18) | known flag | 2/6 | caco2 1/2 logd 1/2 | ginkgo_colloidal |
| Constant lift | Other cleanups of damaged labels (n07) | known flag | 12/20 | caco2 3/4 ginkgo_colloidal 2/6 ginkgo_titer 2/2 herg_ki 2/2 logd 3/4 | buchwald_hartwig |
| truncated_normal_mean | Other treatments of labels clipped at an assay limit (n30) | known flag | 15/22 | buchwald_hartwig 2/2 caco2 3/4 flab_thermo_tresanco 2/2 herg_ki 2/2 logd 4/4 qm9 2/2 | ddg_combined ginkgo_colloidal |
| per_molecule_lift | Other treatments of labels clipped at an assay limit (n30) | known flag | 1/22 | caco2 1/4 | buchwald_hartwig ddg_combined flab_thermo_tresanco ginkgo_colloidal herg_ki logd qm9 |
| Constant lift | Fixing labels clipped at an assay limit | known flag | 12/32 | buchwald_hartwig 1/4 caco2 3/4 flab_thermo_tresanco 2/4 herg_ki 2/4 logd 2/4 qm9 2/4 | ddg_combined ginkgo_colloidal |
| Censored EM | Fixing labels clipped at an assay limit | known flag | 15/32 | buchwald_hartwig 2/4 caco2 4/4 flab_thermo_tresanco 2/4 herg_ki 2/4 logd 3/4 qm9 2/4 | ddg_combined ginkgo_colloidal |
| offset_removed | Several sources per molecule: offsets and weights (n39) | known flag | 21/54 | buchwald_hartwig 10/12 caco2 5/12 ginkgo_colloidal 4/12 herg_ki 1/12 logd 1/6 |  |
| offset_removed_within_var_weights | Several sources per molecule: offsets and weights (n39) | known flag | 27/54 | buchwald_hartwig 9/12 caco2 6/12 ginkgo_colloidal 6/12 herg_ki 2/12 logd 4/6 |  |
| group_center | Removing a shared offset per scaffold group (n34) | known flag | 6/18 | caco2 2/6 herg_ki 2/6 logd 2/6 |  |
| group_center_3plus | Removing a shared offset per scaffold group (n34) | known flag | 7/18 | caco2 2/6 herg_ki 3/6 logd 2/6 |  |
| random_intercept | Removing a shared offset per scaffold group (n34) | known flag | 10/18 | caco2 4/6 herg_ki 4/6 logd 2/6 |  |
| shrunk | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 2/4 | herg_ic50 2/4 |  |
| calibrated | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 0/4 |  | herg_ic50 |

#### noise mitigation — undo part of a deletion

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| cliff_put_back | Putting activity-cliff molecules back after a cleanup (n25) | detector | 12/30 | caco2 5/10 herg 2/4 herg_ki 3/8 logd 2/8 |  |

#### noise-robust training — -

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| early_learning | Other noisy-label training methods for the neural network (n09, n13) | nothing | 44/122 | buchwald_hartwig 4/10 caco2 7/14 cdr_flexibility 4/14 ddg_combined 4/10 ginkgo_acsins 2/5 ginkgo_colloidal 6/10 ginkgo_pr_cho 3/5 ginkgo_titer 2/5 herg_ki 3/12 logd 8/12 qm9 1/5 | herg humanness_therapeutic |
| uncertainty_curriculum | Other noisy-label training methods for the neural network (n09, n13) | nothing | 45/122 | buchwald_hartwig 6/10 caco2 5/14 cdr_flexibility 5/14 ddg_combined 2/10 ginkgo_acsins 3/5 ginkgo_colloidal 5/10 ginkgo_pr_cho 5/5 ginkgo_titer 2/5 herg 1/14 herg_ki 6/12 logd 4/12 qm9 1/5 | humanness_therapeutic |
| meta_weight_net | Other noisy-label training methods for the neural network (n09, n13) | nothing | 45/122 | buchwald_hartwig 3/10 caco2 3/14 cdr_flexibility 7/14 ddg_combined 4/10 ginkgo_acsins 3/5 ginkgo_colloidal 5/10 ginkgo_pr_cho 2/5 herg 1/14 herg_ki 5/12 logd 10/12 qm9 2/5 | ginkgo_titer humanness_therapeutic |
| mentornet | Other noisy-label training methods for the neural network (n09, n13) | nothing | 44/122 | buchwald_hartwig 5/10 caco2 6/14 cdr_flexibility 3/14 ddg_combined 2/10 ginkgo_acsins 1/5 ginkgo_colloidal 7/10 ginkgo_pr_cho 4/5 ginkgo_titer 3/5 herg 1/14 herg_ki 6/12 logd 5/12 qm9 1/5 | humanness_therapeutic |
| contrast_to_divide | Other noisy-label training methods for the neural network (n09, n13) | nothing | 40/122 | buchwald_hartwig 5/10 caco2 7/14 cdr_flexibility 4/14 ddg_combined 2/10 ginkgo_acsins 2/5 ginkgo_colloidal 7/10 ginkgo_pr_cho 3/5 ginkgo_titer 3/5 herg_ki 3/12 logd 2/12 qm9 2/5 | herg humanness_therapeutic |
| heteroscedastic_gp | Other noisy-label training methods for the neural network (n09, n13) | nothing | 45/88 | buchwald_hartwig 5/10 caco2 8/14 ddg_combined 3/10 ginkgo_acsins 2/5 ginkgo_colloidal 7/10 ginkgo_pr_cho 3/5 ginkgo_titer 1/5 herg_ki 7/12 logd 7/12 qm9 2/5 |  |
| evidential_kernel | Other noisy-label training methods for the neural network (n09, n13) | nothing | 39/88 | buchwald_hartwig 8/10 caco2 8/14 ddg_combined 1/10 ginkgo_acsins 2/5 ginkgo_colloidal 4/10 ginkgo_pr_cho 2/5 ginkgo_titer 4/5 herg_ki 4/12 logd 5/12 qm9 1/5 |  |
| DivideMix (training two networks on each other's clean molecules) | DivideMix (training two networks on each other's clean molecules) | nothing | 23/72 | buchwald_hartwig 1/5 caco2 2/7 cdr_flexibility 1/7 ddg_combined 2/5 ginkgo_acsins 2/5 ginkgo_colloidal 4/5 ginkgo_pr_cho 2/5 ginkgo_titer 4/5 herg_ki 4/6 qm9 1/3 | herg humanness_therapeutic logd |
| robust_loss | Huber loss under damage sized by the model's own error (n38) | nothing | 11/24 | abcov_kd 1/4 buchwald_hartwig 1/4 ddg_combined 1/2 ginkgo_acsins 1/2 ginkgo_colloidal 1/2 ginkgo_titer 2/4 herg_ki 2/2 logd 2/2 | caco2 |
| itlm | Trimmed and two-learner selection, and shrinking flagged labels (n32) | nothing | 13/48 | buchwald_hartwig 4/8 ddg_combined 1/8 flab_thermo_tresanco 1/2 ginkgo_colloidal 5/8 ginkgo_titer 1/6 herg_ki 1/4 | caco2 logd |
| c_mixup | Changing how the model reads the labels (n33) | nothing | 4/32 | caco2 1/12 herg_ki 2/12 logd 1/6 | buchwald_hartwig |
| target_noise_bagging | Changing how the model reads the labels (n33) | nothing | 15/32 | caco2 4/12 herg_ki 8/12 logd 3/6 | buchwald_hartwig |
| huber_dnn | Changing how the model reads the labels (n33) | nothing | 11/15 | caco2 4/6 herg_ki 5/6 logd 2/3 |  |
| retuned | Changing the model instead of the labels (n40) | nothing | 34/44 | caco2 12/16 herg_ki 9/14 logd 13/14 |  |
| tuned_clean_reused | Changing the model instead of the labels (n40) | nothing | 21/44 | caco2 8/16 herg_ki 5/14 logd 8/14 |  |
| seed_avg | Changing the model instead of the labels (n40) | nothing | 19/22 | caco2 7/8 herg_ki 5/7 logd 7/7 |  |
| stack | Changing the model instead of the labels (n40) | nothing | 39/44 | caco2 14/16 herg_ki 11/14 logd 14/14 |  |

#### noise-robust training — none (feature or loss)

| method | from page section | selects | yes rows / rows | datasets with a yes row (yes/rows) | datasets with no yes row |
|---|---|---|---|---|---|
| count_feature | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 1/4 | herg_ic50 1/4 |  |
| invvar_and_count_feature | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 1/4 | herg_ic50 1/4 |  |
| ngboost_count_feature | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 2/2 | herg_ic50 2/2 |  |
| xgboost_interval_aft | Using how often each molecule was measured (n19, herg_ic50 only) | known flag | 1/2 | herg_ic50 1/2 |  |
| per_record | Training on every measurement instead of the mean (n29, herg_ic50 only) | known flag | 2/2 | herg_ic50 2/2 |  |

## 4. Methods with no "does it work?" table on the page

Detectors. They are graded on whether the suspicion score is high on the damaged molecules, not by yes/no:
- out-of-fold forest residual; out-of-bag residual;
- neighbour-label gap; the same divided by the neighbours' spread; cliff-aware gap;
- Gaussian process spread; heteroscedastic Gaussian process; variance-head network; NGBoost;
- Shapley value to a nearest-neighbour predictor;
- conformal p-values with a false-discovery-rate rule; isolation forest and LOF; cleanlab;
- labels at the assay limit (a known flag).

Methods with result files but no table on the page:
- moving every label toward its forest-proximity neighbours' mean (n20, n25);
- NGBoost inverse-variance weights (n33);
- acting on conformal flags (n41);
- the per-document methods other than training on every measurement (n29, herg_ic50 only);
- XGBoost survival loss and LightGBM Huber loss on clipped labels (n30);
- the replicate-removal arms (n17, herg_ic50 only);
- clipping onto a quantile-forest range (n11; three antibody files);
- LightGBM Huber, fair, median, Tukey and expectile losses, NGBoost Student-t, forest absolute-error loss (n33; Buchwald-Hartwig, checkpoint file only, the job did not finish).

No results anywhere: XGBoost pseudo-Huber and absolute-error losses.
