Read this whole prompt before doing anything. Then read /Users/apunt/repos/KIRBy/CLAUDE.md, which governs every sentence you write to me.

# What I want

A small framework experiment, f37, that maps how the kinds of label damage differ from each other.
Its result decides how far the noise grid in the noise experiments can be cut.

The noise experiments run every kind of label damage (14 for regression, 16 where detection is
defined) x 7 levels x 3 draws x 5 folds. That grid is why n32, n33 and n38 each come out at
2,000-3,500 hours per dataset. If several kinds of damage do the same thing to a model, the
noise experiments only need one of them. f37 finds out which.

It is a framework experiment (tests/experiments/framework/): its result changes the implementation,
not the thesis. It must be cheap enough to run on my laptop, or in one 4-hour srun. It is a survey
to get the lay of the land, not a precise measurement. Differences between kinds of damage should be
obvious at this scale; if they are not, that is itself the answer.

# The design

- **A few fixed (model, representation) pairs.** No hybrid search at all. Use 3-4 pairs per dataset
  that differ in kind, e.g. rf on ecfp4, svm on a descriptor representation (pdv or mordred),
  lightgbm on a learned embedding (mhg_gnn). Propose the pairs to me before running.
- **Datasets:** 2-3 regression datasets of different sizes, e.g. herg_ki (1,412 molecules) and
  caco2. Propose them to me; I decide.
- **Every kind of label damage** the noise experiments use, at **3 fixed levels** (a low, a middle and
  a high one, from the levels the noise code already uses). The list is `noise_helpers.SWEEP_ORDER`
  (14 regression kinds: grouped_shifted, outlier_only_p10, outlier_p10, gaussian, student_t_nu3,
  censoring, outlier_only_p05, outlier_p05, grouped_wider, laplace, outlier_only_p01, outlier_p01,
  student_t_nu5, student_t_nu10) plus the other kinds in `noise_helpers.DETECTION_IS_DEFINED` (16).
  The full level list is in `tests/experiments/dataset_policy.py`.
- **Damage only the training labels**, through the existing injector
  (`noise_helpers.noise_spec_for` and `inject_train_labels`, noise_helpers.py:312 and :399), so the
  damage is exactly what the noise experiments apply. Score on held-out molecules against their
  clean labels.
- **Folds:** 3 scaffold-group folds. **Draws:** 1, or 2 if it is cheap. Plus one clean-label run
  per pair and fold as the reference.

# What to measure, and how to compare

For each (dataset, pair, kind of damage, level, fold), keep:
1. the held-out score in the dataset's own metric, and its drop from the clean-label score;
2. the held-out predictions themselves.

Then, per dataset and level, compare the kinds of damage with each other in two plain ways:
- **Score drop:** a table of the drop for every kind of damage, one row per kind, one column per
  (pair, level). Two kinds that cost every pair the same amount are candidates to merge.
- **Predictions:** the correlation between the predictions of the model trained under kind A and
  under kind B, on the same held-out molecules. A pair of kinds whose predictions correlate as
  highly as two runs of the same kind with different seeds is doing the same thing to the model.
  Put the same-kind, different-seed correlation next to it as the yardstick. Without that yardstick
  the number cannot be read.

Report which kinds of damage group together, at which levels, with the numbers. Show bare numbers
in the original unit, or simple differences; no significance tests, no confidence intervals, no
new composite metrics. If a heatmap of the correlations helps, make one; one row is one kind of
damage.

# What I need back

1. Before running: the pairs, datasets and levels you propose, and an estimate of the run time in
   hours, measured from a timed run of one (pair, kind, level, fold). Wait for my answer.
2. After running: the groups of damage kinds that behave the same, with the numbers behind each
   group, and the list of kinds that behave differently from all others.
3. One line per decision only I can make, e.g. "keep outlier_p05 and drop outlier_p01 and
   outlier_p10 from the noise grid: their score drops differ by at most X".

# Where things are and the rules

- Code: /Users/apunt/repos/KIRBy, branch similarity-metrics-study. Commit to that branch; one
  branch only, never create another. Push when done.
- Framework experiments live in tests/experiments/framework/. Numbers f00-f15 must never be edited.
  The next free number is f37 (f36_label_smoothing_portfolio.py exists).
- Representations come from the existing caches through the usual loaders. Use
  `KIRBY_LGBM_THREADS=1` and `KIRBY_RF_JOBS=2` locally (torch and lightgbm crash together otherwise).
- The seagate drive (/Volumes/seagate) can disconnect for a few seconds at a time. If a read fails,
  wait and retry. Keep the live log on the internal disk.
- No hybrid search anywhere in this experiment. If something you call would start one, stop and
  tell me.
- Do not run anything heavy before I have answered point 1.
