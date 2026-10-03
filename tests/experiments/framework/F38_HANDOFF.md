# F38 handoff: cut the noise suite's neighbour search space, and add activity-cliff prediction

Written 2026-10-03 by the chat that ran the f34 sweeps. Read `CLAUDE.md` at the repo root first
and follow it in every reply. Everything below marked **verified** was read in the code that
day. Everything marked **unverified** is a claim to check before you repeat it.

## What the owner wants

A framework experiment, f38, that:

1. **Compiles** what the existing framework experiments already measured about finding neighbours.
2. **Narrows** the neighbour search space the noise experiments sweep, with the owner deciding
   each cut.
3. **Runs tests** only on the questions the existing results leave open, so the cut is clean.
4. **Adds activity-cliff (AC) prediction** as a second test of each neighbour setting.

It's a framework experiment because it decides what gets built. Nothing here is published.

## The problem it fixes

**Verified**, `tests/experiments/noise/similarity_space.py` and `NOISE_SEARCH_SPACE.md` §1.7:

- Every neighbour-based cleaning method in the noise suite is rerun once per neighbour setting.
- A neighbour setting is one feature space × one feature importance weighting × one distance.
  - Feature spaces (7): base_1, base_2, base_1_aug_uncapped, base_2_aug_uncapped, hybrid
    (top prediction-built), hybrid__general2 (second prediction-built), hybrid_similarity (top
    similarity-built).
  - Feature importance weighting (4): none, lightgbm_gain, treeshap, deeplift. The importance
    is learned from the noisy training labels on each fold.
  - Distance (2): manhattan, euclidean.
  - Plus 2 fixed settings, ECFP4 Tanimoto and forest proximity, and model_input, which is
    today's behaviour.
  - That's 59 settings.
- Each setting runs at 4 neighbour counts: k = 5, 9, 10, 20 (`SWEPT_KS`).
- So each cleaning method is refitted 236 times per model and fold, instead of once.
- Owner decisions of 2026-09-25 set the sweep. The capped augmented spaces were dropped and
  hybrid__general2 added on 2026-10-03.

**Estimated cost on herg_ki, from another chat (unverified):**

| experiment | hours without the neighbour sweep | hours with it |
|---|---|---|
| n02 | about 5 | about 260 |
| n07 | about 1,100 | about 10,000 |
| n12 | about 4,300 | about 100,000 |
| n11 | about 48,000 | about 1.5 million |

The n02 jump was measured: one unit took 25 times longer with 35 of the 59 settings. The rest
are counts divided through, not timed. That chat said the cost is the model refits after
cleaning, not the neighbour search. Neighbour lists are computed once per setting and stored.

## Facts already established (verified 2026-10-03)

1. **k=9 and k=10 differ by one neighbour.** All four counts are read off one list of 20 nearest
   neighbours. k=9 is there because today's label_smooth and knn_smooth average 9: they query 10
   including the molecule itself, then drop it. model_input stores that exact query on its own
   (`exact_by_k`), so it doesn't depend on the other settings carrying k=9.
2. **The importance power depends on the distance.** It's set by
   `kirby.similarity.preprocessing.fold_weights`: power 1 under manhattan, 0.5 under euclidean.
   s05, s06, s10 and s13 use `_shared.importance_feature_weights` instead, which applies power
   0.5 under every distance. The two agree only under euclidean and cosine.
3. **n07 knows which labels it corrupted.** `_inject_or_recorded` returns a mask
   (`n07_*.py` around line 1425). So for injected noise, each setting can be scored on whether it
   flags the corrupted molecules, with no refits.

## Existing framework results to compile

Check each one's status on the server before using it. Read results only from runs the owner
says are real. Local runs are debug runs.

| experiment | what it measures | status |
|---|---|---|
| f34_full_sweep | kNN prediction across 6 neighbour weightings × k 5–40 × every coefficient f18 defines. 4 datasets (herg, herg_ki, caco2, adme_mdr1), every representation plus 2 hybrids at 100 features, 5 folds | **finished**; results in `tests/results/framework/f34_full_sweep` on the server |
| f34_importance_sweep | the same, plus importance weighting: deeplift, lightgbm_gain or treeshap at power 0.5 or 1, or none. rank/inverse_square, k 10–200, shortlisted coefficients | **started 2026-10-03**; analysis `f34_importance_analyse.py` |
| f18_distance_metric_sweep | which distance coefficient each scoring site in the engine should use | unverified |
| f17_weight_normalisation | sum-to-1 vs average-1 vs raw importance weights | memory says average-1 won; unverified |
| f22_similarity_feature_sources | whether the sim_* representations are better similarity spaces | unverified; refuses classification |
| f23_noise_cleaning_metric | **which similarity space finds wrong activity values best**: detection AUROC and recovery under injected Gaussian noise, using `simeval.noise_sweep_scores` | unverified. Closest existing test to what f38 needs, but on f22's spaces, not the noise suite's 7 |
| f30_importance_methods | which importance methods agree with each other | unverified |
| f36_label_smoothing_portfolio | which label smoothings the noise suite must carry | owner's; unverified |
| f37_label_damage_map | which kinds of label damage act alike; the owner is writing it to shrink the damage axis | in progress by the owner |

**What f34_full_sweep found, from its own output:** on the 29 continuous (dataset,
representation) pairs, including both hybrids, the winning coefficient was:

| cosine | correlation | manhattan | minkowski p0.5 | minkowski p0.3 | euclidean |
|---|---|---|---|---|---|
| 12 | 5 | 4 | 4 | 3 | 1 |

The noise suite sweeps manhattan and euclidean only, and not cosine. That's kNN prediction, not
cleaning, so it doesn't settle the noise question. It's one of the questions f38 has to test.

## Questions f38 has to answer, one per axis

For each, compile what exists first. Test only what is still open.

1. **Distance.** manhattan, euclidean, and the candidates f34 found: cosine, correlation,
   minkowski p0.5.
2. **Importance weighting.** none, lightgbm_gain, treeshap, deeplift. Also which power. f34
   importance sweep answers this for prediction.
3. **Feature space.** The 7 spaces. Does it matter which one, or do most give the same
   neighbours? Measure the overlap of neighbour lists between settings. Two settings giving
   nearly the same lists are a degenerate reason to cut.
4. **k.** 5, 9, 10, 20. See fact 1.
5. **Does prediction quality predict cleaning quality?** If the setting that predicts best with
   kNN is also the one that flags corrupted labels best, f34 can choose for the noise suite. If
   not, it can't.

## Tests to build for what's open

- **A screen with no refits.** On each dataset and injected noise condition, score every
  setting at every k on how well the gap between a molecule's noisy label and its neighbours'
  mean flags the corrupted molecules. Use AUROC against the n07 mask.
  - Undefined where no mask exists: real censoring, the curator flag.
- **A check that the screen works.** On one small dataset, run one noise experiment with every
  setting, and check whether the screen's top few contain the full run's winner. This is the
  same check f34 used: "does the shortlist catch the winner". Cost it before running.

## Activity-cliff prediction

Ask the owner which of these she means before building anything:

- **(a) Finding cliffs.** Which neighbour setting puts a molecule's cliff partner among its k
  nearest neighbours.
- **(b) Predicting activity on cliff molecules.** Error on the molecules that sit in a cliff,
  the way MoleculeACE reports it.

**The ground truth decides the result.** A cliff is two very similar molecules with very
different activity. If "similar" means ECFP4 Tanimoto, the tanimoto_ecfp4 setting wins by
construction. Matched molecular pairs don't use any swept feature space. Put this to the owner
as a decision.

Existing code to read first, unverified:
- `tests/experiments/noise/n14_activity_cliff_landscape.py`: ECFP4 similarity threshold 0.55,
  potency gap 2.0.
- n01's cliff definitions.
- `tests/experiments/opig_activity_cliff_qsar.py`.
- `tests/experiments/similarity_metrics/build_moleculeace.py`.

## Rules that cost the owner work when broken

- **No cut without the owner's sign-off.** A cut needs a stated reason: degenerate, broken or
  undefined. A theory about which value will lose is not a reason. Offer options with the
  numbers attached.
- **Run on the server with one `srun ... --pty /bin/bash` line and plain commands inside it.**
  Never sbatch. Make every run resumable: one output file per task, finished work skipped on
  rerun, cheapest tasks first. `tests/experiments/framework/f34_full_sweep.py` is a working
  example.
- No significance tests. Report per-fold scores and the spread.
- Numbers in the dataset's own metric, with units. No word she hasn't used without saying what
  it is.
- Never write a verdict into her documents.
