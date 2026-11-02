# How a neighbour is counted, and which distance finds it — results

Two settings decide every nearest-neighbour number in this repo: how much a close
neighbour counts against a far one, and which coefficient decides what "near" means.
Neither was ever chosen. sklearn supplies its own when nobody passes anything, and
that is what most of the code gets.

RUN TWICE. The second run, 2026-09-30/10-01, is the one to read. Everything below
was read out of `tests/results/framework/f34_neighbour_weighting_cross/` and
`tests/results/framework/f35_scorer_path_check/` on 2026-10-01, not remembered.

⛔ THE FILE NUMBER IS CONTESTED. Two experiments are called f34:
`f34_loss_functions.py`, written 2026-09-28, and `f34_neighbour_weighting_cross.py`,
which produced everything below. `f34_analyse.py` reads the second. Not resolved.

## ⛔ The first run was wrong about the coefficients

The first run, 2026-09-28, swept four: manhattan, euclidean, cosine and minkowski at
p=0.5, which is `distance_config.DISTANCE_METRICS`. It also z-scored every
representation before measuring, which turns a 0/1 fingerprint bit into a signed
number and leaves no set to intersect, so the set-overlap family was excluded by
preprocessing rather than by measurement.

f18 already had the evidence that this was wrong. Reading
`tests/results/f18_distance_metric_sweep/{herg_ki,logd}.json`, on the score closest
to this question — Spearman between a molecule's activity and what its five nearest
neighbours predict — a set-overlap coefficient wins 6 of f18's 14 representation
blocks outright and is in the top three of 12 of them. On herg_ki `atom_pair` the top
three are tversky 0.4559, dice 0.4548 and tanimoto 0.4478, all ahead of cosine 0.4456
and manhattan 0.4391.

The second run fixed both: coefficients now follow f18's families, chosen by what the
array actually holds, and only continuous blocks are z-scored. The first run's results
are kept at
`tests/results/framework/f34_neighbour_weighting_cross__four_coefficients_2026_09_28/`.

## What the second run measured

**Weighting, five schemes.** `uniform` (every neighbour equal, sklearn's own),
`inverse` (one over the distance, sklearn's `'distance'`), `inverse_square`,
`gaussian` (exp(-(d/h)**2), h the median distance among that molecule's own
neighbours) and `rank` (one over the neighbour's position, magnitude ignored).
`inverse_sqrt` was dropped, with a measured reason: in the first run it was the best
scheme in 3 of 504 groups. `uniform` is kept although it wins nothing, because it is
what the code does today and the comparison needs its control.

**Coefficients, by what the representation is.** Decided from the array's values, not
its name, using f18's own functions.

    binary fingerprints   tanimoto, dice, cosine, tversky, euclidean, manhattan
    count fingerprints    the same, with soergel in place of tversky
    continuous features   euclidean, manhattan, minkowski p=0.5 and p=0.3,
                          chebyshev, cosine, correlation, canberra, bray-curtis,
                          mahalanobis

Six neighbour counts — 5, 7, 10, 15, 20, 25 — on seven representations per dataset:
atom_pair, avalon, ecfp4, maccs and top_torsion are binary, ecfp4_count is counts,
pdv is 200 continuous descriptors. Three datasets: caco2 2158 molecules, herg_ki 1412,
adme_mdr1 2641. Five scaffold-grouped folds, seed 42. 180 cells per fingerprint
representation and 240 per continuous one.

300 fold-level cells were skipped per dataset, all one reason: canberra and
bray-curtis need non-negative input and `pdv`'s raw descriptors go negative. That is
f18's rule, and the skip is logged rather than left blank.

Every score is rmse, lower is better, the mean over five folds. Nothing is averaged
across datasets.

## Result 1 — counting every neighbour the same is never best

`uniform` wins 0 of the 264 (representation, coefficient, neighbour count) groups. On
all three datasets. It was also 0 of 168 in the first run, under a different
coefficient list. This is the most stable finding here.

    dataset      uniform  inverse  inverse_square  gaussian  rank   (of 264)
    herg_ki            0        7              56        29   172
    caco2              0       13              84        16   151
    adme_mdr1          0        9             172         8    75

`inverse_square` and `rank` take 11 and 10 of the 21 (dataset, representation) pairs
between them. Nothing else wins a single one.

## Result 2 — the set-overlap family wins most of it

Winner of each of the 21 (dataset, representation) pairs, by coefficient:

    dice 7, cosine 5, tanimoto 4, euclidean 1, tversky 1,
    minkowski_p0.3 1, soergel 1, minkowski_p0.5 1

13 of the 21 go to coefficients the first run never tested. That is the correction
f18 predicted.

## Result 3 — three coefficients win nothing anywhere

Counted only in the groups each was defined in:

    coefficient     herg_ki        caco2        adme_mdr1
    chebyshev        0 of 30       0 of 30       0 of 30
    mahalanobis      0 of 30       0 of 30       0 of 30
    correlation      0 of 30       1 of 30       0 of 30

All three are defined only on continuous features, so each costs one representation's
time, and that representation is the most expensive one.

## Result 4 — euclidean is nearly never best, but not never

It wins 22 of 210, 7 of 210 and 5 of 210 groups, and takes 1 of the 21 representation
pairs: `avalon` on herg_ki, by 0.0000 over second place, which is a tie rather than a
win. An earlier version of this document said "never best"; that was true only of the
four-coefficient list.

## Result 5 — the winner per representation, and how close second place is

    herg_ki        weighting        coefficient      k     rmse      2nd
    atom_pair      rank             cosine          10   0.5963   0.5974
    avalon         rank             euclidean       25   0.6381   0.6381
    ecfp4          rank             tversky         10   0.6111   0.6116
    ecfp4_count    inverse_square   dice            15   0.6111   0.6114
    maccs          inverse_square   dice            10   0.6558   0.6560
    pdv            inverse_square   minkowski_p0.3  20   0.6039   0.6041
    top_torsion    rank             tanimoto        10   0.6149   0.6149

    caco2          weighting        coefficient      k     rmse      2nd
    atom_pair      rank             tanimoto        25   0.3338   0.3338
    avalon         rank             cosine          20   0.3424   0.3426
    ecfp4          inverse_square   dice            15   0.3284   0.3285
    ecfp4_count    inverse_square   cosine          10   0.3246   0.3249
    maccs          rank             cosine          25   0.3417   0.3418
    pdv            inverse_square   cosine          25   0.3189   0.3190
    top_torsion    rank             tanimoto        20   0.3324   0.3324

    adme_mdr1      weighting        coefficient      k     rmse      2nd
    atom_pair      inverse_square   dice            20   0.5375   0.5377
    avalon         inverse_square   dice            25   0.5835   0.5840
    ecfp4          rank             tanimoto        25   0.5803   0.5803
    ecfp4_count    rank             soergel         25   0.5678   0.5700
    maccs          inverse_square   dice            15   0.5586   0.5590
    pdv            inverse_square   minkowski_p0.5  15   0.5111   0.5130
    top_torsion    inverse_square   dice            20   0.5940   0.5950

Six or seven different cells win across the seven representations of each dataset,
and first beats second by 0.0000 to 0.0022 rmse while the fold spread runs 0.02 to
0.12. The exact cell is far below the fold noise; the family it comes from is not.

## Result 6 — the scoring path does not change the answer (f35)

`f35_scorer_path_check.py`, run 2026-09-30, took the first run's winning cell on each
dataset and scored it three ways on the same folds and representations: raw features
with the coefficient passed as sklearn's `metric=`, a distance matrix computed by the
caller with `metric='precomputed'`, and raw features with nothing passed.

    dataset      the two coefficient-honouring paths    passing no coefficient
                        largest gap      median            largest     median
    adme_mdr1              0.000001    0.000000             0.0648     0.0344
    caco2                  0.000597    0.000000             0.0448     0.0126
    herg_ki                0.000120    0.000041             0.0355     0.0221

The two paths that honour the coefficient agree. Not passing the coefficient costs
between 60 and 1000 times their disagreement. The one gap above a rounding error is
caco2 `maccs`, 167 bits, where two molecules sharing a fingerprint is common: its
whole 0.000597 comes from one fold of five, and when a tie straddles the k-th
neighbour the two paths keep different molecules, arbitrarily in both.

⛔ f35 ran on the FIRST run's winning cells, which were minkowski p=0.5 and cosine.
It has not been re-run on a set-overlap winner. The finding is about the path rather
than the coefficient, so it should carry over, and Result 7 is the reason it matters
less than it looks.

## Result 7 — sklearn cannot take two of the winners

Checked against `sklearn.neighbors.VALID_METRICS` on 2026-10-01. sklearn's `metric=`
argument accepts `dice` and `jaccard`; it has no name for `tanimoto`, `tversky` or
`soergel`.

On real MACCS bits, scipy's `jaccard` equals simeval's tanimoto to 1.110e-16 and
scipy's `dice` equals simeval's dice to the same, so the two most frequent winners —
dice at 7 of 21 and tanimoto at 4 of 21 — CAN travel as a plain string on binary
fingerprints, tanimoto under the name `jaccard`.

`tversky` and `soergel` cannot, and they win 1 of 21 each. Neither can any of these
names on a COUNT fingerprint, where simeval's continuous forms are not scipy's
boolean ones.

## PRELIMINARY implementation conclusions — NOT SIGNED OFF

The owner said "I generally agree" to an earlier form of rules 1 to 4 on 2026-09-30,
before the re-run. Nothing has been changed in any file.

1. Every kNN fitted as a model passes a neighbour weighting, never sklearn's implicit
   `uniform`. It wins 0 of 264 groups on every dataset, in both runs.
2. Every such kNN is given a coefficient, never sklearn's implicit Euclidean.
3. The coefficient stays a swept axis, and WHICH coefficients are even defined
   depends on whether the representation holds bits, counts or continuous values.
4. The neighbour count stays swept.

### What is still open and needs the owner

- **The swept sets have not been recomputed since the re-run.** The earlier proposal
  — weighting `inverse_square` and `rank`; coefficient manhattan, cosine and
  minkowski p=0.5; neighbour count 5, 7, 15, 25 — was derived from the
  four-coefficient run and two of its three coefficients are no longer the winners.
  It must not be used.
- **How the coefficient reaches the kNN now splits in two.** On a binary fingerprint
  it can be a string argument. On a count fingerprint, or for tversky and soergel, the
  caller must compute the matrix and pass `metric='precomputed'` — the path
  `simeval.py:161` already uses and which f35 measured as giving the same score.
- **Only base representations.** The two hybrid representations, one from the
  `general` lineage and one from `general_distance` at a budget of 100 features, have
  not run under either version.
- **Three datasets, all regression, all rmse.** Nothing here says what a
  classification set does.

### Cost, measured

The second run took 5 hours 40 minutes across the three datasets on one core.

    representation kind   herg_ki     caco2   adme_mdr1
    each binary            28-84s   95-217s    238-432s
    ecfp4_count (counts)      487s     1186s       3176s
    pdv (continuous)         1281s     2507s       5684s

`pdv` is two and a half hours of the total and `soergel` on `ecfp4_count` another
eighty minutes. soergel is an explicit chunked loop over row pairs rather than a
matrix product, and it scales worse than the square of the molecule count. It earns
its place on evidence — f18 has it winning 2 of 14 blocks, both count fingerprints,
and here it wins 30 of the 30 groups it is defined in on adme_mdr1 — but it is the
single dearest coefficient in the sweep.

## Where the rule is broken, counted from the tree 2026-09-30

46 places in 19 live files. Nothing edited. Re-count before acting; this predates the
re-run.

**The coefficient reaches the kNN, the weighting is `inverse`** — rule 2 holds, rule 1
does not. All pass `weights='distance', metric='precomputed'`:

    src/kirby/find_best_hybrid.py:2183
    src/kirby/utils/noise_mitigation_methods.py:413
    tests/experiments/similarity_metrics/simeval.py:161
    tests/experiments/similarity_metrics/triplet_similarity.py:152
    tests/experiments/similarity_metrics/scaffold_experiment.py:218
    tests/experiments/similarity_metrics/base_rep_comparison.py:141
    tests/experiments/framework/f18_distance_metric_sweep.py:330
    tests/experiments/framework/f19_objective_scoring.py:207
    tests/experiments/framework/f22_similarity_feature_sources.py:281

**Neither reaches it** — sklearn's implicit `uniform` and implicit Euclidean, which
is the pair both runs put last. 27 places:

    src/kirby/hybrid.py:2997,2999            the engine's own kNN evaluator
    tests/experiments/similarity/s04_activity_cliffs.py:372,373
    tests/experiments/complexity/c04_importance_as_metric.py:116,122
    tests/experiments/complexity/c12_modality_ablation.py:489
    tests/experiments/_shared.py:2899,2905   evaluate_importance_weighted_knn
    tests/experiments/_shared.py:3120-3123   the kNN capacity family
    tests/similarity_experiments.py          9 places
    tests/complexity_theory_experiments.py   5 places

**The weighting is read from an environment variable, the coefficient does not reach
it** — s05:416,417 and s10:645,655,743,760,769. Both default to `uniform`.

**`uniform` written out, no coefficient** — n40:418,1041,1048.

Left alone: `tests/archive/` (5 places), `src/clamp/clamp/linear_probe.py` (vendored,
and it already passes cosine with distance weighting), and the engine's
`_HP_SEARCH_SPACES['knn']`, which tunes its own weighting and coefficient per fold —
forcing the rule there would overwrite the tuned choice, which is what
`make_nearest_neighbors` in `tests/experiments/_shared.py:2913` already warns against.

### Two places the rule may not belong, owner to decide

- **f18, f19 and f22 have already run.** Changing their weighting moves their stored
  numbers, and f18's are what this document relies on.
- **`_shared.py:3120-3123` is the kNN capacity family**, where the neighbour counts
  20, 10, 5, 1 ARE the capacity gradient. Giving each rung a weighting and a
  coefficient changes what that family measures.

## Code written for this

    src/kirby/similarity/neighbour_weighting.py   the six schemes, one zero-distance
                                                  guard, env var KIRBY_NEIGHBOUR_WEIGHTING
    tests/test_neighbour_weighting.py             31 tests, passing
    tests/test_f34_scoring_paths.py               29 tests, passing
    tests/experiments/framework/f34_neighbour_weighting_cross.py
    tests/experiments/framework/f34_analyse.py
    tests/experiments/framework/f35_scorer_path_check.py

`uniform` and `inverse` travel to sklearn as its own `'uniform'` and `'distance'`
strings rather than being reimplemented, so the sweep's p=0 and p=1 arms are the code
the suite already runs. One test asserts the `inverse` scheme matches sklearn's own
`_get_weights` on distances containing a zero. The coefficient families are imported
from `f18_distance_metric_sweep` rather than copied, so the two experiments sweep the
same functions under the same names.
