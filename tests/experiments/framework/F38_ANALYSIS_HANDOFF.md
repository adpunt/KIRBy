# Handoff: from the f34–f38 results to the kNN search space the noise and similarity experiments run

Written 2026-10-05 by the chat that built and ran f38. Read `CLAUDE.md` at the repo root first and
follow it in every reply. Read `F38_DECISIONS.md` (in this folder) second: it is everything the
owner has signed off, and nothing in it is to be argued again.

**Verified** below means read in the code or the result files on 2026-10-03 to 2026-10-05.
**Unverified** means re-check it before repeating it. The owner's thesis is due 2026-10-10, so
the order of work matters: decide, implement, run.

---

## 1. What the owner wants from you

1. **Decide the kNN search space** that the noise experiments (n01–n42) and the similarity
   experiments (s01–s13) run. Base it on the framework results f34 onward. The owner decides
   every cut; you bring the options with the numbers.
2. **Implement that search space** in the noise and similarity experiments.
3. **Get the experiments running** in srun. Never sbatch.

Speed matters, but not at the cost of care. Read every result in full before proposing anything.
Give realistic runtimes: count fits, and where you can, time them.

## 2. The questions, one per axis

The noise suite finds neighbours in many cleaning methods, and refits every method once per
neighbour setting. A neighbour setting has these axes:

| axis | what the noise suite sweeps today (verified, `tests/experiments/noise/similarity_space.py:115-141`) |
|---|---|
| feature space | base_1, base_2, base_1_aug_uncapped, base_2_aug_uncapped, hybrid, hybrid__general2, hybrid_similarity |
| importance weighting | none, lightgbm_gain, treeshap, deeplift |
| importance power | not swept: set by the distance, 1 under manhattan, 0.5 under euclidean (`kirby.similarity.preprocessing.weight_exponent`) |
| distance | manhattan, euclidean |
| neighbour count k | 5, 9, 10, 20 (`SWEPT_KS`) |
| how much a neighbour counts | not swept: every neighbour counts the same |
| fixed settings | tanimoto_ecfp4, forest_proximity, model_input (today's behaviour) |

That is 7 × 4 × 2 + 3 = 59 settings × 4 k = 236 refits per method, model, fold, draw and level.

For each axis, answer two questions:
1. What does the evidence say: which values win, which never win, and which give the same
   neighbours as another value?
2. Does it hold for finding and fixing corrupted labels, not only for kNN prediction?
   f34 and f35 measured prediction; f38 measures cleaning.

The owner also asks: **does a setting that predicts well with kNN also clean well?** If yes,
f34 can choose settings for the noise suite. If no, it can't. This is a comparison of f34's
ranking with f38's ranking, on the same datasets and representations.

### The similarity experiments have their own kNN choices

Verified in `F34_F35_HANDOFF.md` and the `f34_neighbour_weighting_cross.py` docstring:
- s04 (`s04_activity_cliffs.py:372`), s05 (`s05_knn_prediction.py:416`) and s10
  (`s10_importance_downstream.py:586`) fit sklearn kNN on raw features and pass no distance. So
  they get sklearn's default, euclidean, whatever `KIRBY_DISTANCE_METRIC` says.
- s05 and s10 read the neighbour weighting from an environment variable
  (`kirby.similarity.neighbour_weighting`, `KIRBY_NEIGHBOUR_WEIGHTING`, default uniform).
- k: s05 scores 5 and 10 by default (`KIRBY_S05_K_VALUES`); s10 scores 5, 10 and 20
  (`KIRBY_S10_K_VALUES`).
- s05, s06, s10 and s13 apply importance at power 0.5 under every distance
  (`_shared.importance_feature_weights`). The noise suite uses 1 under manhattan. The two agree
  only under euclidean and cosine.

Read every similarity experiment in full (s01–s06, s08–s13 in `tests/experiments/similarity/`,
plus `similarity_helpers.py`), and record each kNN choice with its file:line.

## 3. Constraints the analysis must respect

- **Two models per dataset.** `scratchpad/precompute_cuts/DECISIONS.md` step 1 (owner,
  2026-10-03). For f38's four datasets: herg_ki svm+dnn, adme_mdr1 svm+dnn, caco2 dnn+svm,
  herg lightgbm+xgboost. Read every other dataset's pair from that file. Analyse only those
  models. knn is not one of anyone's two models; it gets its own precompute for the
  experiments that need it (step 9).
- **The precompute is being redone per fold on a much smaller grid** (owner, 2026-10-04;
  `DECISIONS.md` steps 2–9, `HANDOFF_PER_FOLD_ALIGNMENT.md` at the repo root). The hybrid
  feature spaces in the noise suite come from that precompute. So any decision about hybrid,
  hybrid__general2 or hybrid_similarity is about spaces the new precompute will produce, not the
  ones f38 tested.
- **One kept list per axis, used for every dataset.** No cuts per dataset. Each dataset's numbers
  are shown side by side and never averaged across datasets.
- **A cut needs a stated reason: degenerate, broken or undefined.** A theory about which value
  will lose is not a reason. The owner decides every cut.
- **No cutoff fixed in advance.** For example, how many neighbours two settings must share to
  count as the same. Show the full range; the owner sets the cutoff.
- **No significance tests.** Report per-fold scores and the spread.
- **The levels trade against the kNN space** (owner, 2026-10-04). f37 decides which noise levels
  differ; how many the owner can afford depends on how small the kNN space gets. Give her the
  refit count under each option.
- **Framework results, local and quick ones included, ARE evidence** for these decisions. The
  "local runs are debug" rule covers the noise and similarity suites' own preliminary runs, not
  framework experiments.

## 4. The framework experiments: what each measured, and where the results are

All paths are relative to the repo root. All local copies were checked complete on 2026-10-04.

### f34_neighbour_weighting_cross
- **Question:** for kNN prediction, cross how much a close neighbour counts (the weighting
  scheme) with which distance decides "near".
- **Grid (second run, 2026-09-30/10-01):**
  - 5 schemes: uniform, inverse, inverse_square, gaussian, rank
  - distances by what the array holds: f18's binary, count and continuous families
  - k = 5, 7, 10, 15, 20, 25
  - 7 representations: atom_pair, avalon, ecfp4, maccs and top_torsion (binary), ecfp4_count,
    and pdv (continuous)
  - herg_ki, caco2 and adme_mdr1, 5 scaffold-grouped folds, rmse
- **Results:** `tests/results/framework/f34_neighbour_weighting_cross/{herg_ki,caco2,adme_mdr1}.json`.
  The first, wrong run (four distances only) is in the `__four_coefficients_2026_09_28` folder.
  Ignore it.
- **Written up:** `tests/experiments/framework/NEIGHBOUR_WEIGHTING_AND_DISTANCE_RESULTS.md`.
  Read its numbers from the json files, not from the write-up. Analyser: `f34_analyse.py`.
- **Headline numbers, from the write-up, unverified until you recount them:**
  - Uniform weighting wins 0 of 264 (representation, distance, k) groups on each dataset.
  - The set-overlap distances (dice 7, tanimoto 4, plus tversky and soergel) win most of the 21
    (dataset, representation) pairs.
  - chebyshev and mahalanobis win nothing; correlation wins 1 of 30 on caco2.
- **Its section "PRELIMINARY implementation conclusions — NOT SIGNED OFF"** holds rules 1–4.
  The owner said "I generally agree" to an earlier form on 2026-09-30. Its old proposed sets
  were derived from the wrong run and must not be used.
- **Also note its result 7:** sklearn's `metric=` can't take tanimoto (use `jaccard` on bits),
  tversky or soergel, or any of them on counts. Those need a precomputed distance matrix.

### f34_full_sweep
- **Question:** the same as above, on the full representation pool plus two hybrids, with more k.
- **Grid:**
  - 6 schemes (the 5 above plus inverse_sqrt)
  - k = 5, 7, 10, 15, 20, 25, 30, 40
  - every distance f18 defines: binary 6, count 6, continuous 10
  - herg (classification), herg_ki, caco2, adme_mdr1
  - each dataset's 10 representations plus `hybrid_general` and `hybrid_general_distance`, each
    at 100 features, recipes from the stored search
  - 5 folds
- **Results:** `tests/results/framework/f34_full_sweep/<dataset>/<rep>__fold<i>__<part>.jsonl`.
  Complete: 110–120 files per dataset, every (representation, fold, part) present.
- **Analysers:** `f34_full_analyse.py`; `f34_k_check.py` (does k = 10, 20, 40 lose anything,
  and does the best k depend on the representation kind or the distance).
- **From the F38 handoff, unverified:** on the 29 continuous (dataset, representation) pairs the
  winning distance was cosine 12, correlation 5, manhattan 4, minkowski p0.5 4, minkowski p0.3 3,
  euclidean 1. The noise suite sweeps manhattan and euclidean only.

### f34_importance_sweep
- **Question:** does weighting features by importance help kNN prediction, and at which power?
- **Grid:**
  - 7 variants: none, plus lightgbm_gain, treeshap and deeplift, each at power 0.5 and 1
  - schemes: rank, inverse_square
  - k = 10, 20, 40, 50, 60, 80, 100, 150, 200
  - distances: a shortlist; binary tanimoto, dice, euclidean; count soergel, tanimoto,
    manhattan; continuous cosine, correlation, manhattan, minkowski p0.5
  - the same 4 datasets and representations as the full sweep
- **Results:** `tests/results/framework/f34_importance_sweep/<dataset>/...jsonl`, plus cached
  importance vectors in `importance/`. Complete: no file short of its 7 rows.
- **Analyser:** `f34_importance_analyse.py`.

### f35_scorer_path_check
- **Question:** does f34's answer survive the repo's other way of computing kNN predictions?
  There are three paths: features with the distance passed; a precomputed matrix; features
  with no distance (what s04, s05 and s10 do today).
- **Results:** `tests/results/framework/f35_scorer_path_check/{adme_mdr1,caco2,herg_ki}.json`,
  rerun 2026-10-04 on the second f34 run's winning settings:
  - adme_mdr1: inverse_square, minkowski p0.5, k=15, on pdv
  - caco2: inverse_square, cosine, k=25, on pdv
  - herg_ki: rank, cosine, k=10, on atom_pair
- The previous run, which checked the first f34 run's winners, is in
  `f35_scorer_path_check__first_f34_run_2026_09_30/`.
- The final log line on herg_ki said: passing the distance at all moves the score by at most
  0.1794 rmse, and by 0.0032 at the median.

### f34_loss_functions
- Not about neighbours: it asks which training loss the models should use. Not needed here.
- It is not in git.

### f36_label_smoothing_portfolio (the owner analyses it)
- **Question:** which label smoothings the noise suite must carry. `F36.md`, `f36_analyse.py`.
- **What you need from her:** does any smoothing that reads the neighbours' labels survive?
  label_smooth, knn_smooth and n02's weights read them. If none survives, the kNN cut no longer
  touches smoothing.

### f37_label_damage_map (the owner analyses it)
- **Question:** which kinds of training-label damage act alike, at 3 levels each, 22 kinds.
- **What you need from her:** the kinds and levels she keeps. f38 uses one level (0.5 of the
  label's standard deviation), so her choice of levels doesn't change f38's numbers.
- `HANDOFF_F37_LABEL_DAMAGE_MAP.md` at the repo root; `f37_analyse.py`.

### f38_neighbour_screen (running on the server; results not in yet)
- **Question:** for finding and fixing corrupted labels, with no refits, how does each neighbour
  setting do, and which settings give the same neighbours?
- **Code:** `tests/experiments/framework/f38_neighbour_screen.py`. 12 unit tests in
  `tests/test_f38_neighbour_screen.py` check that its lists match the noise suite's own, list
  for list.
- **One task:** one (dataset, condition, fold, draw, model). 300 tasks:
  - datasets: herg_ki, caco2, adme_mdr1, herg
  - each dataset's two decided models
  - 1 draw
  - conditions: the regression conditions where some labels are corrupted and some aren't
    (`noise_helpers.DETECTION_IS_DEFINED`), and herg's 6 flip conditions
- **Labels:** corrupted as n07 does. Regression at 0.5 of the label's standard deviation;
  classification at a flip rate of 0.1.
- **Settings scored, per task:**
  - the noise suite's 59
  - cosine, correlation and minkowski p0.5 on the same 7 feature spaces
  - every importance weighting at power 0.5 and 1, besides the power the distance implies
    (minkowski p0.5 implies 2)
  - base_1 and base_2 as raw fingerprints under f18's set-overlap distances, when they are bits
    or counts
  - model_input and forest_proximity once per model input
- **Every setting gets two neighbour lists:** plain, and with the molecule's own scaffold
  group excluded (what n07's neighbour detectors read). k = 5, 9, 10, 20.
- **Row fields:**
  - `auroc_{scheme}_k{k}`: AUROC of the corrupted-molecule list against |noisy label −
    neighbours' weighted mean|, for each of uniform, inverse, inverse_square, gaussian and rank.
    On classification it is the weighted share of neighbours that disagree.
  - `auroc_median_k{k}`: the same with the unweighted median (f23's form).
  - `recovery_{scheme}_k{k}`: RMSE between the neighbours' weighted mean and the CLEAN label on
    the corrupted molecules, in the label's units. On classification: the share whose
    neighbours' weighted majority is the clean class.
  - `recovery_before_cleaning`: the noisy label's RMSE, for comparison.
  - `grouped` is plain or grouped. `rows_with_undefined_distance` counts rows with an undefined
    distance.
  - A `_done` row holds `seconds` and `stages` (per-step timings). A `_skipped` row holds the
    reason.
- **Overlap file** `<task>.overlap.jsonl`: for pairs of settings that differ in one axis, plus
  model_input, tanimoto_ecfp4 and forest_proximity against everything, `shared_k{k}` = the mean
  share of a molecule's k neighbours the two lists have in common. Plain lists only. Its last
  row lists the unavailable spaces and why.
- **Output on the server:**
  `/data/stat-ecr/scat9264/KIRBy/tests/results/framework/f38_neighbour_screen/<dataset>/<condition>__fold<i>__draw<d>__<model>.jsonl`.
  The owner pulls it to this laptop when it's done.
- **Caveats you must carry into the analysis:**
  1. **The hybrid spaces are test spaces.** They're rebuilt from stored rows of the OLD
     whole-dataset searches, partly filled (owner: "you can use the results for testing").
     - hybrid = the best stored recipe for that model, augmentation none or uncapped.
     - hybrid__general2 = the next-best stored recipe that differs from it. NOT the noise
       suite's rank-2 rule, which needs the full search.
     - hybrid_similarity = the best stored similarity recipe.
     - Skipped: recipes over representations no longer loaded, and recipes that used a subset
       of the pool without recording which. The counts are in the overlap file's last row.
  2. base_1 and base_2 come from the cached 5-fold baselines. Where those are missing, they're
     ranked on the run's own folds (`base_reps.json`).
  3. One noise draw per task, not three: the owner's speed decision.
  4. **Conditions where every label moves get no rows** (gaussian, laplace, the student_t ones,
     grouped_shifted). Recovery would still be defined there; f38 doesn't compute it. That's a
     gap to tell the owner about.
  5. **Under censoring, the local debug run gave AUROCs of 0.28–0.32** on herg_ki, svm, base_1:
     corrupted molecules had SMALLER gaps than clean ones. A clipped label may sit near its
     neighbours. Check this on the real results before reading the censoring AUROCs as "bad".
     It could be an inverted score under censoring, not a bad setting.
  6. The first `rm -rf` of the output (2026-10-05) deleted 120 finished tasks. The rerun
     started from scratch.
- **Not built yet:**
  - `f38_analyse.py`
  - the comparison of f38 with n42, which refits after cleaning, to check that the screen's top
    settings contain the setting n42 finds best
  - the comparison of f38 with f34 on f34's datasets and representations
  - activity-cliff prediction. It's blocked on 3 owner decisions in `F38_DECISIONS.md`:
    finding cliff partners or predicting cliff molecules; ECFP4 Tanimoto ≥ 0.55 or matched
    molecular pairs; a gap of 2.0 label units or 2 × the label's median absolute deviation.

### Others to check for relevance before you start
- f18_distance_metric_sweep (distance per engine score) and f30_importance_methods (do the
  importance methods agree; if they rank features alike, their settings are degenerate).
- The owner said f34 onward is what's relevant. Read f18 and f30 only if f34+ leaves a gap they
  answer, and say so before relying on them.

## 5. Where the kNN choices live in the code

- **Noise suite axis:** `tests/experiments/noise/similarity_space.py`. `FEATURE_SPACES`,
  `WEIGHTINGS`, `FEATURE_METRICS`, `SWEPT_KS`; `SimilaritySetting` validates metric ∈
  {manhattan, euclidean} (`_CDIST`, :146); `enumerate_settings`, `settings_from_env`
  (`KIRBY_SIMILARITY_SETTINGS`).
- **n07.sweep_settings / sweep_ks / fold_similarity_spaces / fold_neighbours** in
  `n07_remediation.py` (:1582, :1609, :1634, :1788).
- **Disk cache of neighbour lists:** `noise_cache.cached_neighbours` (noise_cache.py:557).
- **Which experiments read the settings:** from `NOISE_SEARCH_SPACE.md` §1.7, unverified: n01,
  n02, n07, n09, n11, n12, n13, n14, n17, n18, n20–n25, n32, n35, n36, n37, n40, n41, n42. Only
  n07, n23, n32, n35 and n42 were checked method by method.
- **Importance power:** `kirby.similarity.preprocessing.fold_weights` / `weight_exponent`;
  `_shared.importance_feature_weights` (power 0.5 everywhere) in s05, s06, s10 and s13.
- **Neighbour weighting:** `src/kirby/similarity/neighbour_weighting.py` (`WEIGHTING_SCHEMES`,
  `weight_array`, `sklearn_kwargs`).
- **Distances defined per kind:** `f18_distance_metric_sweep.coefficients_for`,
  `classify_representation`.

## 6. Compute: what to expect

- **f38, measured 2026-10-04 on the server:** about 60 minutes per round of 32 tasks on herg_ki,
  caco2 and adme_mdr1, at one thread per task. herg (7772 molecules) is much slower; no herg
  task had finished by the time of writing.
- **Noise suite refits** for a method that reads neighbours, per model, fold, draw and method
  option = kept settings × kept k × levels. Today: 59 × 4 × levels.

  | kept settings × k | 1 level | 3 levels | 5 levels |
  |---|---|---|---|
  | 59 × 4 (today) | 236 | 708 | 1180 |
  | 15 × 2 | 30 | 90 | 150 |
  | 5 × 2 | 10 | 30 | 50 |

  The middle and bottom rows are examples, not proposals.
- **The handoff's hours on herg_ki**: n02 about 260 h with the sweep, n07 about 10,000 h, n12
  about 100,000 h, n11 about 1.5 million h. Unverified except n02. Count each experiment's real
  refits from its code before quoting any of these.
- **Cost traps already found and fixed in f38** (useful when implementing):
  - Thread pile-up: each process must be capped at one thread (OMP, MKL, OpenBLAS, LOKY,
    `KIRBY_LGBM_THREADS`, `KIRBY_RF_JOBS`, torch).
  - scipy's weighted correlation runs one pair at a time in Python.
  - The noise suite recomputes importance for every neighbour list.
  - forest_proximity refits its forest for every list kind.
  - Any search that isn't fully stored must never start inside an experiment
    (`KIRBY_COLD_SEARCH_MAX_CONFIGS=0`).

## 7. How to proceed

1. Read the files above in full: the f34–f37 code, analysers and results, `DECISIONS.md`, every
   similarity experiment, and the noise experiments that read neighbours. No skimming.
2. Write down, per axis, what f34 and f35 settle for prediction. Recount every number from the
   result files.
3. When f38 lands, write `f38_analyse.py`. Per axis, one table: a row is one value, a column is
   one dataset, and a cell is the per-fold scores and their spread. Two tables per axis:
   detection AUROC and recovery. Then the full overlap results. Then f34's ranking beside f38's.
4. Bring the owner the options: one decision per line, each with the numbers, the stated reason
   (degenerate, broken or undefined) and the refits saved. Include the similarity experiments'
   choices (distance, weighting, k, power), with the cost of making them match.
5. After she signs off: implement in `similarity_space.py`, n07 and the similarity experiments.
   Test that each changed list matches its old one wherever the setting is unchanged. Then give
   her srun blocks that are safe to paste again: no `rm`, no rewriting the task list.

## 8. Rules that cost the owner real work when broken

- Never put `rm` or any delete in a command block she pastes. She re-pastes whole blocks.
- srun only: one `srun ... --pty /bin/bash` line, then plain commands. Interactive sruns are
  limited to 4 hours, and she runs several at once.
- Before handing over commands, make sure every file they import is in git. f34_loss_functions
  and f35 were untracked.
- Never write a verdict into her documents. Never invent cut targets. No workflows without her
  explicit go.
- Use her words: representations, folds, budgets, datasets, features. Banned: "read-across",
  "SAC". Every number gets its unit.
