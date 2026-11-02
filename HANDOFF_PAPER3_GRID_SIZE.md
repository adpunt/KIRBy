# Handoff: reduce the grid sizes of the Paper 3 experiments

Written 2026-11-02 by Claude, from Adelaide's ask of that day: limit censoring to a
chosen set of datasets, and then go through every Paper 3 experiment the same way. Every
count below was computed from the code in that session, by the script named with it. No
count here is a measurement of time.

Read `CLAUDE.md` first. The rule that governs this work, quoted:

> Don't cut the search space for compute. Cutting an axis, budget or dataset needs a
> reason: degenerate, broken, or undefined. A theory about why it will lose is not one.

That rule binds Claude, not Adelaide. She is the engineer and decides what to spend. So
every proposal in this file comes with its count and what it buys, and none of it is
applied without her word.

## 1. Where the size comes from

One UNIT of the detector-by-cleanup experiment (`n44_detect_and_clean`) is one
combination of noise kind, noise level, noise draw, model and feature set, on one outer
fold. The unit count is a product of five factors:

| factor | Caco-2 | what it is |
|---|---|---|
| passes | 84 | the (kind, level, draw) combinations; the table in section 2 |
| models | 2 | the dataset's two decided models (`decided_grid.decided_models`) |
| feature sets | 3 | base_1, base_2 and the hybrid: DC's three chains (restructure handoff section 8) |
| folds | 5 | Adelaide, 2026-10-30: all five of the scaffold-grouped split |
| **units** | **2,520** | 84 x 2 x 3 x 5 |

Inside one unit, n44 fits the downstream model once per (detector, cleanup, setting) pair
its pairing rules allow. `_smoke_n44.py` MEASURED that on one unit of 128 random
molecules: 4,032 fits, from 54 detector columns and 23 cleanups. Not every pair is
allowed and each cleanup carries its own settings, so 54 x 23 does not give 4,032
directly.

So Caco-2 is about 2,520 x 4,032 = 10.2 million fits of the downstream model. LogD, which
runs 2 noise levels where Caco-2 runs 4, is 1,380 units and about 5.6 million.

TWO THINGS THAT MAKE 4,032 THE WRONG NUMBER TO PLAN WITH, both to settle first:
- it was measured on 128 molecules. Caco-2 has 2,158 and LogD 5,028, and some detectors
  cost more than linearly in molecules;
- the neighbour settings are not in it. They multiply the neighbour detectors and the
  neighbour cleanups further.
FIRST TASK, cheap: run ONE unit of n44 on LogD and count the rows it writes, so the
per-unit number is real rather than borrowed from a smoke. One unit is minutes to an
hour.

## 2. The passes, kind by kind

One row is one noise kind on a dataset that runs the full grid (Caco-2). Computed
2026-11-02 with `KIRBY_NOISE_PROMOTE=1`.

| kind | levels | draws per non-zero level | passes | the levels |
|---|---|---|---|---|
| gaussian | 4 | 3 | 10 | 0, 0.5, 1.0, 1.5 label SDs |
| outlier_p05 | 4 | 3 | 10 | 0, 0.5, 1.0, 1.5 |
| grouped_shifted | 4 | 3 | 10 | 0, 0.5, 1.0, 1.5 |
| censoring | 4 | 1 | 4 | 0, 0.10, 0.25, 0.40 share of labels clipped |
| shift_only_p10 | 4 | 3 | 10 | 0, 0.5, 1.0, 1.5 |
| sparse_dprime_up_p10 | 4 | 3 | 10 | 0, 1.0, 2.0, 4.0 residual SDs |
| dense_oofmse | 4 | 3 | 10 | 0, 0.1, 0.3, 1.0 x out-of-fold mean squared error |
| hetero_seen | 4 | 3 | 10 | as dense_oofmse |
| hetero_hidden | 4 | 3 | 10 | as dense_oofmse |
| **total** | | | **84** | |

Level 0 is the labels as delivered and runs once; every other level runs 3 draws
(1 under censoring).

## 3. Censoring: capping it saves 5 percent, not more

Adelaide's starting point was to limit censoring to a chosen set of datasets, since
results on it already exist. The count, over the 23 regression datasets in the run list:

| | DC units | censoring's share |
|---|---|---|
| screen levels | 32,940 | 1,380 (4%) |
| full grid | 56,820 | 2,700 (5%) |

Censoring is the CHEAPEST kind in the table, because it runs 1 draw per level where the
others run 3. So capping it is worth doing for the reason Adelaide gave — results on it
exist, and only three datasets have a real floor in their labels as delivered
(`n46.REAL_CENSORING_LIMITS`: ginkgo_pr_cho, adme_hlm, adme_rlm) — but it is not where
the cost is.

WHAT TO DO, if she agrees: name the datasets censoring runs on, and add them as
`CONDITION_LIMITED_TO`-style entry in `noise_helpers` so every experiment reads one list.
`CONDITION_LIMITED_TO` already exists and limits an EXPERIMENT to certain conditions; a
condition limited to certain DATASETS is the mirror of it and does not exist yet.

## 4. Where the cost actually is, largest first

Each line says what the factor multiplies and what removing or halving it would cost in
units. No line is a recommendation.

1. **23 cleanups x their settings, inside every unit.** The single biggest multiplier
   inside a unit. Three of them already carry 3 flag shares and several carry 3
   strengths, so the settings matter as much as the count. The audit's section s6 records
   which raised the top score in the old results; every cleanup in DC is on List 1.
2. **54 detector columns, inside every unit.** 26 registry detectors, n32's six, n41's
   three, the conformal flags at every (calibration share, q), the random control, and
   the oracle flags. The oracle flags already run in one detector space rather than two
   (restructure handoff 9.6).
3. **3 draws per non-zero level.** Halving to 1 draw takes Caco-2's 84 passes to 34 and
   its units from 2,520 to 1,020. This is the cheapest large saving and the most
   expensive scientifically: the draw spread is what says whether a gain is the method or
   the damage.
4. **4 noise levels on a full-grid dataset.** Cutting to 2 (what the screen runs) takes
   84 passes to 48.
5. **9 noise kinds.** The two heteroscedastic kinds and dense_oofmse were added
   2026-11-02 so the comparison between the two units of noise (label spread against
   model error) runs somewhere. 30 of Caco-2's 84 passes are those three kinds.
6. **3 feature sets and 2 models.** Both are the subject of the chapter; cutting either
   removes a comparison rather than a cost.

## 5. The other four experiments: what to check in each

The same five factors apply, but each file has its own multipliers on top. None of these
has been counted yet; counting them is the next task after section 1.

| experiment | its own multipliers to count | first thing to look at |
|---|---|---|
| `n43_noise_aware_training` | 5 feature spaces (n09's, not DC's 3), one dnn search per (kind, level, draw, fold), about 37 methods | the searches: one per pass per fold, on the damaged labels. That is the factor DC does not carry |
| `n45_detector_grading` | 37 detector names x the neighbour settings x every k x both counting rules, with NO screen (grading fits no model, so n06/n13/n22's rule was to sweep them all) | the neighbour sweep without a screen |
| `n46_structure_specific_fixes` | the fixes of each structure x models x 3 model inputs; the sources pass carries its own rho and tau axes and 3 draws | the sources pass, which has axes nothing else has |
| `n47_noise_measurements` | n27's removal curve is 5 shares x 20 draws; n37's Gamma test is per representation x the neighbour settings | n27's 20 draws, and the pairwise fingerprint comparison of section 6 |

## 6. One step that is slow for its own reason

`n01._compute_reaction_tanimoto_sim` builds a fingerprint per molecule and then compares
every pair with `scipy.spatial.distance.cdist`. On Buchwald-Hartwig that is 3,057 x 3,057
pairs, and both n44 and n47 were still inside it when an 8-minute crash check was stopped
(restructure handoff 9.10). It is the cliff reference, computed once per dataset and read
by n44's cliff put-back, n47's cliff measurements and n25. Worth caching per dataset
rather than recomputing; it is not a grid axis and cutting the grid does not touch it.

## 7. Decisions this needs from Adelaide

One per line.

- Which datasets censoring runs on.
- Whether 3 draws per non-zero level stays 3.
- Whether a full-grid dataset keeps 4 noise levels or runs the screen's 2.
- Whether the 23 cleanups and 54 detector columns stay whole in DC.
- Whether the two heteroscedastic kinds and dense_oofmse stay on every dataset or on a
  chosen set, as the censoring question.

## 8. What not to do

- Do not cut an axis without her word. Her rule is quoted at the top.
- Do not report a count as a time.
- Do not use `_smoke_n44.py`'s 4,032 fits as a planning number without section 1's
  measurement on a real dataset.
- Do not touch the 5 folds. Adelaide, 2026-10-30: all five, and the caches are keyed on
  each fold's training rows.
- Do not read any number in this file as a result. They are grid sizes.

## 9. How the numbers here were computed

In a python session with `src`, `tests` and `tests/experiments` on the path:

    from experiments.noise import n44_detect_and_clean as dc
    from experiments.noise.noise_helpers import draws_for
    data = {'task_type': 'regression', 'labels': list(range(2158)), 'reps': {'r': None}}
    for c in dc.default_conditions(data):
        levels = [float(x) for x in dc._condition_levels(data, 'caco2', c)]
        draws = len(draws_for(c, (0, 1, 2)))
        passes = 1 + len([l for l in levels if l > 0]) * draws

`KIRBY_NOISE_PROMOTE=1` gives the full grid; without it the levels are the screen's.
The models come from `decided_grid.decided_models(ds, 'general')`, the datasets from
`orchestrate.plan.dataset_roster()`, and the 4,032 fits per unit from the printed output
of `_smoke_n44.py`.
