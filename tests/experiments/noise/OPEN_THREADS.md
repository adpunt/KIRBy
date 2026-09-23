# Noise suite — open threads

Things measured or changed that have a next step nobody owns. Written 2026-09-21.
A thread leaves this file when it is closed, not when someone reads it.

Coverage of what has been optimised is generated, not typed:
`python tests/experiments/noise/coverage.py`. This file is for what is NOT in it.

---

## A. Changed tonight, consequences not yet re-measured

### A1. The noise level ladder gained 1.5 and swapped 0.15 for 0.2
`dataset_policy._DEFAULT_NOISE_LEVELS` is now `(0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)`.
It had copied a row from `NOISE_DESIGN.md` §6.4 that was superseded on 2026-08-27 by
one shared ladder. 1.5 matters because the sibling study keeps it as the only level
where the noise types separate at all, and this suite compares noise types.

**Ten experiments sweep this ladder and every result any of them has on disk is on
the old axis:** n07, n09, n11, n12, n13, n15, n16, n20, n21, n23. The other seven
inject nothing and are untouched.

⛔ NEXT: every stored result from those ten is not comparable to a new one. Nothing
has re-run yet.

### A2. `pick_noise_base_rep` no longer picks the control alphabetically
On `flab_thermo_tresanco` the control is now `onehot`, was `esm2_cdr` — a learned
embedding and 8th of the 10 by how well it predicts, 0.391 spearman with a random
forest against 0.594 for the best.

⛔ NEXT: **every noise number measured on an antibody dataset before 2026-09-21 is
on the old footing.** Two ledger entries for flab already sit against different
baselines, 0.4590 and 0.5985 spearman, for this reason. 8 of the 43 datasets now
STOP rather than guess — six antibody sets and two genomics ones hold nothing but
learned embeddings and need a plain encoding adding to their rep list.

### A3. `reference_noise_level()` returns 0.5 because it is "the middle of the grid"
That is a positional choice, and it was the middle of the OLD grid. The sibling
study sets a REPORTING level of 1.0 on measured rank movement, 0.75 where the clean
score is already weak. n07 at 0.5 produced a clean-to-corrupted gap of 3% of the
clean score on flab, negative on one fold.

MEASURED 2026-09-21, and it weakens the case for moving the level: that 3% was
saprot. In the onehot `pick_noise_base_rep` now returns for flab, the same level
0.5 and the same tenth of the training molecules give a gap of 0.0644 spearman
against a clean score of 0.4068 — 15.8% of it — over 3 folds and 3 corruption
draws. The realised dose is 0.501 of the clean label standard deviation in both.
The ladder in onehot: 15.8% at level 0.5, 30.8% at 1.0, 61.9% at 2.0, 80.2% at
3.0. So the level n07 already runs is readable in its own control arm.

Three consumers: n07 (its single level), n13 (a marker on its curve), and
`noise_helpers`. Changing it moves all three differently.

⛔ NEXT: owner's decision, not taken. Measure what 1.0 gives n07 before changing it.

---

## B. Measured, and the follow-up is not owned

### B1. n07 and n18 measured the same axis and disagree
How far a score-proportional correction moves a label. n07, on injected noise at 3
folds: 0.5 to 1.0 improved all three targets, 21.2%→47.2%, 11.7%→23.5%,
25.0%→31.6% of the gap. n18, on real labels: 1.0 LOSES at every correction —
label_smooth best at 0.25, oof_smooth and gp_smooth at 0.5, downweight at a 4-to-1
row weight ratio.

They differ in three ways: injected against real noise, share-of-gap against
held-out spearman, and saprot against esm2_cdr. `n07_shape_in_esm2_cdr` RAN on
2026-09-21, 3 folds, and the answer REVERSES: in esm2_cdr, rescaling the
suspicion score by its rank recovers 42.8% of a 0.2112 spearman gap against
11.1% for moving each label the whole way in proportion to the raw score. In
saprot the same pair read 19.4% against 47.2%. ⛔ THAT PAIR IS NOT CLEAN: the
saprot numbers were measured before the length-scale fix, so the representation
and whether the Gaussian process was fitting both changed between them.

A THIRD READING, 2026-09-21, and it sides with n18. n23 injects noise and sweeps
the same score-proportional move at 0.25, 0.5 and 1.0 on flab, 3 folds, level
0.5. The gentlest strength 0.25 is the best of the three in 2 of its 3
uncertainty sources, and the furthest strength 1.0 is negative in 2 of the 3.
So injected-against-real is not on its own the reason n07 and n18 disagree.

⛔ NEXT: which one sets the default is undecided. No default has moved.

### B2. CLOSED 2026-09-21. The Gaussian process was not reading its inputs, and
### the seven-way ranking has been re-run with one that does
The RBF length scale started at 1.0 and never left: the three inner folds fitted
1.01, 1.00 and 1.01 with the white-noise level pinned at its own upper bound, so
the posterior mean was one constant per inner fold. Fixed in
`GaussianProcessEstimator.estimate` — the length scale now starts at the median
distance between the training molecules it is fitted on.

The seven-way ranking re-run on flab in saprot, 3 folds, injected level 2.0,
clean 0.5985 spearman against corrupted 0.4111 so the gap is 0.1874: remove
35.7% of the gap, quantile_clip 34.1%, knn_smooth 33.6%, oof_smooth 28.2%,
gp_smooth 27.2%, downweight 18.7%, label_smooth 14.3%. `gp_smooth` is fifth of
seven where it was second from bottom.

⛔ NEXT, and it is not a defect: THE TOP THREE ARE NOT SEPARATED. remove,
quantile_clip and knn_smooth sit 0.0039 spearman apart on the three-fold mean
and each wins a different fold outright. The run's own floor, measured inside
it: two gp arms whose targets differ by 0.0000002 label units scored 0.0402
spearman apart on fold 1.

⛔ ALSO NOT MEASURED HERE: what the fix bought in share of the gap. The bake-off
names two gp arms, one supposedly started at 1.0, but the repo fix had already
landed and both fitted the median-started process — its own log says 358
distinct posterior means either way. `n07_gp_lengthscale_payoff` is with the
supervisor and restores the pinned start inside the fit.

### B3. Importance-weighting a kernel: the two readings disagree, and the newer
### one has more behind it
AGAINST: 0.2709 against 0.3303 spearman at 200 features, 0.2182 against 0.2473
at 50. Two of two, ONE fold, in saprot narrowed by hand, with the length scale
still pinned at 1.0.

FOR, 2026-09-21: in esm2_cdr at the full 1280 features with the length scale
fixed, multiplying the kernel's inputs by the square root of each feature's
lightgbm gain wins the three-fold mean at 4 of 4 shapes — 32.7%, 41.6%, 46.7%
and 31.1% of a 0.2112 spearman gap against 10.9%, 11.1%, 42.8% and 15.9%
unweighted — and 9 of 12 paired fold readings. It loses folds 0 and 2 at the
rank rescale and wins the mean there on fold 1 alone.

⛔ NEXT: nothing in n07 passes `feature_weights`, so neither reading has changed
any default. Whether it is the representation or the fixed length scale that
flipped the sign is not measured.

### B4. n12's largest effect has never been followed up
In its existing result file the unweighted HYBRID beat the unweighted base
representation in 6 of 6 readings under both importance scorers — the largest and
most consistent effect in that file. Two of its eight conditions are secretly one
condition, agreeing to 1e-15, because the blueprint's own importance method is
`lightgbm_gain`.

### B5. n21's table at more passes: the top clears the plain row, the top two
### do not clear each other
The reading that opened this thread, in esm2_cdr at 2 folds and 1 corruption
draw and 5 neighbours, 4 paired readings per comparison: importance-weighted at
exponent 1.0 under lightgbm_gain and euclidean was the ONE comparison whose mean
gain beat the spread of the per-cell gains behind it, +0.0120 average precision
over the plain row, spread 0.0056, ahead in 4 of the 4 cells.

RE-READ 2026-09-21 in `onehot`, which is what `pick_noise_base_rep` now returns
for this dataset, at 2 folds by 3 draws and all three neighbour counts -- 36
paired readings per comparison, in
`/Volumes/seagate/local_noise_run/loop_n21/flab_thermo_tresanco__folds_and_draws__11distances_6passes.json`
-- a copy kept under that name because the live file is being overwritten, see
below:

    imp_pow1|treeshap|euclidean        0.2292 detection average precision
    forest_proximity                   0.2251
    imp_pow0.5|treeshap|euclidean      0.2196
    imp_pow1|treeshap|manhattan        0.2170
    imp_pow1|lightgbm_gain|euclidean   0.2130
    plain|euclidean                    0.1905
    plain|manhattan                    0.1875

4 of the 10 comparisons against a plain row now beat their own spread, where 1
did: treeshap at exponent 1.0 under euclidean +0.0387 (spread 0.0350) and under
manhattan +0.0295 (spread 0.0257), and forest proximity +0.0346 against the
plain euclidean row (spread 0.0326) and +0.0376 against the plain manhattan row
(spread 0.0312).

⛔ THE TOP TWO STILL DO NOT SEPARATE. Treeshap at exponent 1.0 under euclidean
is 0.0041 average precision ahead of forest proximity, the spread of that gain
over the 6 passes is 0.0239, and it wins 4 of the 6. 9 of the 15 passes asked
for are still running.

⛔ NOT TWO READINGS OF ONE THING. The control representation differs, the
neighbour counts differ, and the old table was one corruption draw.

⛔ THE TWO HYBRID DISTANCES ARE MISSING FROM THOSE 11, so the
similarity-objective space -- the one the owner suspects should win, because its
features were chosen so the distance predicts a molecule's activity from its
five nearest neighbours -- has no number in that table. When those 6 passes ran,
every stored search was unusable: flab_thermo_tresanco's was missing the
`distance_metric` column and caco2's and herg_ki's held 0 rows each.

THE STORE CAME BACK AT 08:04 on 2026-09-21. The run restarted after that opens
it, and the recipe lookup now builds a hybrid 499 features wide. So the table
being written from 08:23 has 13 distances, not 11, and it starts from no passes
rather than the 6 -- a table half read with the hybrid distances and half
without is not a table, and the resume refuses it by design. The 11-distance
reading above is the copy named in this section and is not overwritten.

### B5b. The neighbour count only matters for one of the two detectors
From the same 36 cells per distance. Averaged over all 11 distances and the 6
passes, detection average precision is 0.1996 at 5 neighbours, 0.2129 at 10 and
0.2176 at 20. Paired within (distance, pass, detector), 20 neighbours beat 5 by
+0.0180 average precision in 85 of the 132 paired readings.

That average hides the split. For `local_variance_ratio` 20 neighbours beat 5 by
+0.0392 in 53 of 66 readings; for `neighbor_residual` they lose by 0.0032 in 32
of 66. `neighbor_residual` is also the stronger detector, +0.0285 average
precision over `local_variance_ratio` in 140 of 198 paired readings.

⛔ NEXT: nobody owns whether a detector that reads a spread should be given more
neighbours than one that reads a residual. Both are swept, so nothing is
currently decided by this.

### B5c. n21's forest proximity, swept over its shape for the first time — CLOSED 2026-09-23
The forest whose leaves the proximity is read off ships at 100 trees and 5
molecules a leaf. `_loop_n21_forest.py` swept 100, 200 and 300 trees against 1,
5, 20 and 50 molecules a leaf, on flab_thermo_tresanco in onehot, under
outlier_p10 at level 0.5, detection only. It is finished: all 6 passes, one pass
being one (fold, corruption draw), 34 to 46 of the 448 training molecules given
a moved label, in
`/Volumes/seagate/local_noise_run/loop_n21/flab_thermo_tresanco__forest_shape.json`.

Arm means over the 36 cells of (fold, corruption draw, detector, neighbour
count), detection average precision:

    forest, 300 trees, 1 per leaf      0.2438
    forest, 100 trees, 1 per leaf      0.2437
    forest, 200 trees, 5 per leaf      0.2391
    forest, 100 trees, 5 per leaf      0.2366   ← what n21 ships
    forest, 100 trees, 50 per leaf     0.2152
    plain representation, manhattan    0.1879
    plain representation, euclidean    0.1844

⛔ THE TREE COUNT LOST AND 200 HAS BEEN PUT BACK TO 100. It was taken at 3
passes, where it had gained 0.0082 average precision on 3 of the 3 passes then
on disk. Paired on the full 36 cells, 200 trees gains 0.0025 average precision
and wins 20 of the 36 cells, against a spread over those 36 cell gains of
0.0151 — six times the gain. On precision at the number of molecules that
really moved, 0.2637 at 100 trees against 0.2683 at 200, winning 14 of the 36
cells. 300 trees gains 0.0020 on the same cells. The doubled forest fit is not
paid. Recorded in coverage_loops.json with kept false.

⛔ THE LEAF SIZE STAYS AT 5, and the same 6 passes settle it. 50 molecules a
leaf loses 0.0214 average precision in 26 of the 36 cells. 1 molecule a leaf
gains 0.0072 and 20 molecules a leaf gains 0.0033, each inside its own spread
over the 36 cell gains — 0.0446 and 0.0422 — and both lose on precision at the
moved count, 0.0191 and 0.0080 below the shipped 5.

WHAT THE FOREST ITSELF IS WORTH did not change: at the shipped 100 trees and 5
molecules a leaf the proximity beats the plain representation under euclidean
by 0.0522 average precision in 31 of the 36 cells, and that gain is larger than
its own spread of 0.0429 over those cells.

This is one dataset under one condition, detection only; the repair is not
measured here.

### B5d. n21 has never run under a condition its distances could predict —
### RUNNING on caco2
Every n21 reading on record is under `outlier_p10`, which picks the molecules it
damages at random, so no distance that reads the features can have an edge from
structure. `_loop_n21_conditions.py` runs n21's distances on caco2, 2,158
molecules in ecfp4, under `grouped_wider` and `censoring` beside `outlier_p10`,
3 folds by 2 draws, detection only, registered with the supervisor as
`n21_conditions_grouped_and_censoring`. Section E is the reasoning.

⛔ The flab table and this one are on different conditions and different
molecules and neither transfers to the other.

### B6. CLOSED 2026-09-21. n23's cleaning wins folds, and the zero-noise
### control says it is not removing the hard molecules
`flab_thermo_tresanco__outlier_p10_cleanwin3.json`, 672 antibodies, 448 per
training block, onehot at 5600 features, 3 outer folds, rf, one draw.

Under outlier_p10 at level 0.5, with 40 of the 448 training molecules moved:
dropping the 10 percent with the widest `heteroscedastic_nn` spread beat the
uncleaned fit on 3 of 3 folds, 0.3442 spearman uncleaned, mean gain +0.0697
spearman, spread over the three fold gains 0.0223 spearman.

At level 0.0, where the injector moves no label, that same arm loses all 3 folds
at mean -0.0630 spearman. Over the three sources, 2 of 27 settings still beat the
uncleaned fit on most folds at level 0.0 against 8 of 27 at level 0.5.

⛔ BUT THE RANKING IS NOT WHAT WINS IT. Run beside two controls it never had,
same folds, same 45 of the 448 molecules removed: removing them AT RANDOM won 3
of 3 folds at +0.0688 spearman on one seed, 3 of 3 at +0.0349 on a second, 1 of
3 at -0.0628 on a third. Removing the molecules whose labels actually moved --
42, 35 and 43 of the 448 on the three folds -- won 2 of 3 folds at +0.0167
spearman. So +0.0697 sits at the top of the span a random drop of that size
covers and not outside it, and identifying the moved labels is worth less than
a random draw of the same size. The probe returns n23's own two numbers,
-0.1157 for activity_cliff_aware and +0.0697 for heteroscedastic_nn, from the
same folds, which is why the controls beside them can be read.

HOW MANY OF THE MOVED MOLECULES EACH ARM REMOVES, fold 0, 45 removed of 448
where chance takes 4.2 of the 42 that moved: activity_cliff_aware 14,
gaussian_process 11, ngboost 9, heteroscedastic_nn 5. The ordering by how well
an arm finds the moved molecules is the REVERSE of the ordering by folds won.

⛔ NEXT: one draw, one flag rate, three random seeds, 3 folds. A second draw is
with the supervisor as `n23_second_draw`, the random control as
`n23_random_drop_control`. What nobody owns is whether `folds won against no
cleaning` can carry n23's claim at all when a random drop of the same size wins
3 of 3 folds on one seed in three.

### B6b. Detection and cleaning do not pick the same source
`heteroscedastic_nn` has the weakest detection of the three, 0.1041 average
precision against 0.0893 for a coin flip, and owns the only arm that wins every
fold. `gaussian_process` has the strongest detection, 0.2495, its best arm wins 2
of 3 folds at +0.0234 spearman inside its own spread of 0.0544, and it costs 391
seconds per uncertainty estimate against 74 for `heteroscedastic_nn`.

⛔ NEXT: whether ranking the moved molecules is the right thing to select an
uncertainty source on at all. Nobody owns this.

### B6c. hetero_gp is the expensive source at 5600 features, not ngboost
All five measured side by side in one process on one fold of
flab_thermo_tresanco, 448 training molecules, onehot at 5600 features, wall
seconds on a machine carrying a one-minute load of 79: activity_cliff_aware 12,
heteroscedastic_nn 30, ngboost 49, gaussian_process 88, **hetero_gp 1070**.

The earlier reading said the opposite — 42 CPU-seconds at 1280 features,
recorded the same day as "hetero_gp is NOT expensive". At 5600 it is 12 times
the next slowest, and it is why a five-source n23 run in onehot takes hours.
`pick_noise_base_rep` returning `onehot` on every antibody set is what put the
five sources at 5600 features in the first place.

⛔ NEXT: owner's decision. Nothing has been cut.

### B7. n20 answered its own question on 8 cells and that is the whole denominator
On `flab_thermo_tresanco`, 672 antibodies, 2 held-out folds, one corruption draw
at 0.5 of the clean label spread, the recipe search choosing between three of the
ten representations, five of the ten and all ten:

Cleaning the labels BEFORE the search changed the recipe the search selected in 1
of 2 cells under `knn_smooth` and 1 of 2 under `remove`, against 0 of 2 under the
identity control. Both moves are on held-out fold 0, where the search on the
corrupted labels took five of the ten representations, the search on the
knn_smooth-cleaned labels took all ten, and the search on the remove-cleaned
labels took three of the ten.

The held-out score moved the same way: searching on the cleaned labels won 2 of 2
folds under both cleanups, mean +0.0977 spearman under knn_smooth (spread over
the two fold gains 0.0567) and +0.0473 under remove (spread 0.0127).

One cell is one held-out fold and one downstream model, so there are 2 cells per
cleanup and 1 of 2 is one fold, not a rate.

⛔ NEXT: `n20_next_passes` is with the supervisor and takes each cleanup to 6
cells by adding two more corruption draws. It also runs `label_smooth`, the third
cleanup that hands back a full-length cleaned column, which no n20 run has ever
included, and re-runs `search_scores_on=cleaned` so that setting has a record of
which recipe it selected.

### B8. `downweight` cannot be asked this question and is not a null result
`sample_weight` appears zero times in `src/kirby/hybrid.py` and zero times in
`src/kirby/find_best_hybrid.py`, re-grepped 2026-09-21. Nothing in the feature
selection reads a per-row weight, so `downweight`'s two searched-on-cleaned arms
are undefined rather than zero, and n20 names them that way rather than running
them.

⛔ NEXT: nothing, unless the owner wants the feature ranking to read a weight.
That is a change to `hybrid.py`, not to n20.


### B7. `remove` and `knn_smooth` act on exactly the same molecules
Both read the same cut-off: `remove` keeps the training rows whose suspicion
score falls below the (1 - noise_fraction) percentile, and `knn_smooth` rewrites
the rows at or above it. So they are the same selection under two treatments —
drop the flagged tenth, or give it the mean label of its ten nearest molecules.
On flab in saprot at injected level 2.0, 3 folds, that head-to-head is 35.7% of
the gap for dropping and 33.6% for rewriting, which is inside the fold spread.

The consequence for anything that composes them: `knn_smooth` then `remove` IS
`remove`. CHECKED on 120 synthetic molecules — same kept rows, and the largest
difference between the two label columns over those rows is 0.0000000000 label
units. The other four label repairs move EVERY label some distance, so none of
them collapses this way.

⛔ NEXT: nothing is broken. It is written down because a combination run has to
know which pairs are one arm.

---

## C. Defects found and not fixed

### C1. CLOSED 2026-09-21. n11 no longer discards its own primary claim to
### protect a caveated one
It returned `{'error': 'no_blueprint'}` when the pool held no recipe, before its
level loop started. Commit `2f1ceb6` drops the hybrid estimation space instead
and carries on, recording what was dropped and why in
`results['pool_selection']['hybrid_arm']`.

MEASURED: a run that used to end at 0 scored cells now scores the cross. On
`herg_ki`, 1,412 molecules in ecfp4, 941 per training fold, it reached 8 of the
10 estimators at 44 cells each on the first of three folds — 352 cells — before
the machine restarted.

⛔ NEXT, and it is the remaining half: n11 has still never written a result file.
One is being produced under the supervisor as `n11_first_result`. Every estimator
call it finishes is cached to `loop_n11/grid_cache` keyed on the exact arrays it
was given, so a restart resumes rather than refitting; the first attempt lost
8,328 seconds of fits to the crash because nothing was written until the end.

### C1b. n11's `GP_MAX` is a constant nothing reads
`GP_MAX = 1000` at n11:261 is referenced only by a comment. The cap that applies
is hardcoded inside `GaussianProcessEstimator.estimate`, also 1000, per inner
fold. On `herg_ki` each inner fold trains on about 753 of the 941 training
molecules, so nothing is subsampled and the detector costs 4,248 seconds per
fold — 34% of the run's measured estimator time. Not changed: the cost lives in
what the estimator computes, and the estimator axis is the experiment.

### C2. Four callers swallow a halted result store
n18, n23 and two others catch `StaleResultDB` and carry on in the base
representation alone rather than stopping. n18's message reported it as an empty
pool, which reads as a statement about what the pool holds; that one is fixed.

### C3. Six cache stores were moved aside by something outside this repo
At 00:11:53 on 2026-09-21: abagym_EGFR, caco2, ddg_combined, flab_thermo_tresanco,
herg_ki, logd, renamed with a `.stale_<timestamp>` suffix that appears nowhere in
this repository. Nothing was deleted; all six are still on disk. A run looking for
one finds nothing and quietly creates an empty file instead.

⛔ NEXT: owner's decision. Renaming them back could collide with whatever is still
writing there.

### C4. `flab_thermo_tresanco.db` on seagate is unusable and was not deleted
11,692 stored searches, missing the `distance_metric` column, so a run opening it
halts. It was 22.8 hours old when the 24-hour cutoff was applied to the other 50.
The copy in the home cache has the column and zero rows. So that dataset has no
usable stored searches either way, which is why every measurement on it tonight was
base-representation-only.

### C5. n16 pins itself to a single level of 0.3
Every other sweeping experiment reads the ladder. n16 may be right to — an
importance-stability measurement may not want a sweep — but it is the one place
left where an experiment names a level itself.

---

## D. Never measured

- **n09, n11, n17 on the three local datasets**: no result file exists for n11 yet;
  see C1 for the run producing one, and for why it could not before today. n17
  ran only on `herg_ic50`, because the three local datasets have no repeat
  measurements at all.
- **n17's one result is negative**: on `herg_ic50` in ecfp4, the ceiling did not
  beat the floor, and its own read-out says so rather than reporting a position.
- **Every n07 result file ever written has an empty detection block.** Eight
  gradings were recorded into a dict built before the loop that fills it. Fixed in
  `0f83448`, but no file written before that has the numbers.

---

## E. Condition coverage — the two kinds of noise almost nothing runs

Added 2026-09-21 at the owner's request. `noiseInject.CONDITIONS` holds eleven.
What each experiment actually names:

    n07  outlier_p05                  n16  gaussian
    n09  gaussian                     n20  outlier_p10
    n11  gaussian                     n21  outlier_p10
    n12  gaussian                     n23  grouped_wider, outlier_p10
    n13  censoring, gaussian

### E1. CENSORING is run by exactly one experiment
n13, and c10. It is the most PREVALENT real mechanism in the sibling study's
literature review, and the only condition that is not zero-mean: values past the
assay limit are recorded as the limit, so it biases labels in one direction
rather than scattering them. That study measured it as **twelve times more
damaging than any shape difference between the zero-mean types**.

It cannot be dose-matched — no variance parameter — so it has its own axis,
`CENSORING_LEVELS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)`, a fraction of labels
clipped. That is already correct in `dataset_policy`.

⛔ NEXT: no cleaning or detection method has ever been measured against it.
Every method result in this suite is against noise that scatters labels
symmetrically. A detector that scores a molecule by how far its label sits from
its neighbours' mean has no reason to work when every label above a cut-point
was moved DOWN to the same value — n23 measured exactly that and found all three
sources at or below a coin flip under censoring.

### E2. GROUPED_SHIFTED is named in four files and RUN BY NONE
`grep` finds it in comments in c10, n13, n16 and n23. Only n13 sweeps it, via
SWEEPS_ALL_CONDITIONS.

⛔ WHY IT MATTERS MORE THAN THE OTHER TYPES. The two grouped conditions are the
only ones where the noise is PREDICTABLE FROM STRUCTURE — whole scaffold groups
are affected together. Every other condition picks its victims at random, so a
method that reads the features has nothing to find. The sibling study says so
directly of grouped_wider: "the only zero-mean condition where noise is
predictable from structure, so the only one that tests whether a model can
*spot* bad data".

`grouped_shifted` is the sourced one: 62% of real measurement variance sits
BETWEEN laboratories, and laboratory averages differing is an OFFSET, not a
widening. `rho = 0.62` comes from Bentz et al. 2013 Table 7. Its algebra is
`eps_i = sqrt(rho)*tau*b_g(i) + sqrt(1-rho)*tau*e_i`, so it is zero-mean in
expectation but not in any one run, and that asymmetry is its mechanism rather
than a defect.

So the suite's detection and cleaning methods have been measured almost entirely
against noise they cannot in principle predict from features, and the one family
that they could is the one nothing runs.

STARTED 2026-09-21. `n23.NOISE_KEYED_TO_STRUCTURE` was `('grouped_wider',)` and
is now `('grouped_wider', 'grouped_shifted')`, so n23 records its three
feature-reading sources as DEFINED under grouped_shifted. Keyed to structure and
detection-gradeable are two different things and grouped_shifted is the
condition that separates them: on herg_ki at level 0.5 it moves 941 of the 941
training molecules, so `detection_scores` returns None and the only reading is
whether cleaning wins folds. Measured the same day, same fold: grouped_wider
moves 172 of 941, censoring at 0.25 moves 231, outlier_p10 at 0.5 moves 93.
All four inject on herg_ki, which has 769 scaffold groups over its 1412
molecules. A job `n23_conditions_herg` is with the supervisor and runs them in
that order, one condition per invocation, each with its own level 0.0 control.
herg_ki is rmse where LOWER wins, unlike flab's spearman.

### E3. What can run it, and what cannot
`dataset_policy.noise_groups_for` returns no groups for any dataset without
SMILES, and the grouped conditions are then DROPPED rather than run over invented
groups. So:

  - caco2, herg_ki and every other molecular set CAN run both grouped conditions.
  - flab_thermo_tresanco and every antibody, nucleotide and expression set
    CANNOT. n23 hit this and recorded it.

⛔ That splits the suite: the condition most worth measuring is unavailable on
the dataset most of this campaign has used.

### E3b. MEASURED 2026-09-21: grouped_shifted moves EVERY training label, so
### a detection score cannot be read under it
Through `noise_helpers.inject_train_labels`, on 900 synthetic labels in 601
lopsided groups with 675 of them in the training block, 5 corruption draws each:

    grouped_shifted   675, 675, 675, 675, 675 of 675 training labels moved
    gaussian          675, 675, 675, 675, 675
    grouped_wider     120, 106, 113, 116, 112
    outlier_p10        60,  62,  65,  75,  59
    censoring         338, 338, 338, 338, 338

Repeated on caco2 itself, 2158 molecules in 1086 scaffold groups, 1438 of them
in the first training fold, 3 draws each: grouped_shifted moved all 1438 every
time, grouped_wider moved 289, 300 and 238, and outlier_p10 moved 151, 139 and
128.

grouped_shifted's own algebra is why: `eps_i = sqrt(rho)*tau*b_g(i) +
sqrt(1-rho)*tau*e_i` gives every molecule its own second term on top of its
group's shared offset, so every label moves and only the OFFSET is shared.
Average precision against "which molecules moved" is then 1.0 by construction,
which is why `noise_helpers.DETECTION_IS_DEFINED` lists grouped_wider and not
grouped_shifted. That tuple is right and does not need changing.

⛔ So an experiment whose reading is DETECTION cannot use grouped_shifted, and
the structure-linked condition it can use is **grouped_wider**. n12 runs
neither: a job `n12_grouped_vs_random_caco2` is registered with the supervisor
to measure the weighting's margin over the plain space under grouped_wider and
under outlier_p10 on caco2, the same folds and the same draws, reported
separately.

⛔ Reading n12's weighting under grouped_shifted means reading RECOVERY, which
needs a hybrid recipe from the pool and a clean-to-noisy gap above
GAP_MIN_FRAC. Nobody owns that yet.

### E4. Seven experiments can already take a condition and six default to one
`HONOURS_NOISE_CONDITION = {n09, n11, n12, n16, n20, n21, n23}`. The plumbing
exists; only the default is narrow. Adding `grouped_shifted` beside `gaussian`
on a molecular dataset costs one more condition per experiment, not new code.

### E5. FOUR EXPERIMENTS DEFAULT TO A CONDITION WHERE DETECTION CANNOT BE GRADED
Found 2026-09-21 by the n11 loop, verified here.

`DETECTION_IS_DEFINED = ('outlier_p01', 'outlier_p05', 'outlier_p10',
'grouped_wider', 'censoring')` at noise_helpers.py:499. `gaussian` is not in it,
and cannot be: gaussian moves EVERY training label — 941 of the 941 on herg_ki,
at every level of the ladder — so there is no untouched set to score a ranking
against and `detection_scores` returns None.

`NOISE_CONDITION = 'gaussian'` is the default in **n09:95, n11:119, n12:87 and
n16:152**.

⛔ n12's stated purpose is FINDING NOISY LABELS by an importance-weighted
distance. Under its own default its detection block is empty by construction.
Every detection number n12 has ever produced came from a loop running it under
`outlier_p10` by hand.

⛔ SO THE CONDITION PROBLEM IS SHARPER THAN E2 SAID. Of the five conditions
where detection can be graded at all, four are outlier variants that pick their
victims at random, and the fifth — `grouped_wider` — is the only one where the
features could predict the corruption. `grouped_shifted` is not in the list at
all, so the best-sourced structure-linked condition cannot currently grade a
detector even if an experiment ran it.

⛔ NEXT: whether `grouped_shifted` belongs in DETECTION_IS_DEFINED is a question
about the injector, not a preference. It moves every molecule in an affected
group and leaves the rest alone, so a "which molecules moved" set exists for it
in a way it does not for gaussian. Nobody has checked whether the injector
records that set.

### E6. The level that clears the floor, measured on herg_ki
From the n11 loop, 2026-09-21. ecfp4, 3 scaffold-grouped folds, clean 0.6032
rmse, so n11's 2%-of-clean floor is 0.0121 rmse. One row is one injected level.

    level   gap          folds clearing the floor
    0.20    +0.0043      0 of 3
    0.30    +0.0034      0 of 3
    0.50    +0.0137      1 of 3
    0.75    +0.0370      3 of 3
    1.00    +0.0622      3 of 3
    1.50    +0.1417      3 of 3

`reference_noise_level()` returns 0.5. Its three-fold MEAN clears the floor, but
`_heldout_recovery` reads each configuration on one fold at a time against that
fold's own gap, and two of the three folds are +0.0034 and +0.0109 rmse, both
under it. So a recovery number at 0.5 would come from one fold.

This is the same disease as n07's dose (A3), measured independently on a
different dataset and metric. The level costs nothing — the fit count is
identical at every level.

⛔ NEXT: A3's decision is still open, and this is a second dataset pointing the
same way. The n11 loop moved its own run to 0.75 and left
`reference_noise_level()` untouched.

### E5. n17's ceiling can rank on two quantities and the default has not moved
`measurement_spread` is the standard deviation of all of a molecule's repeat
measurements. `training_label_error` is how far the one measurement that became
its training label sits from the mean of its others. Counted on herg_ic50, no
model fitted, at three draws of which measurement trains: of the 65 molecules
the spread ranking takes first, 51 are measured exactly TWICE. Below three
measurements `build_entities` holds nothing out, so those 51 train on the mean
of both their measurements and no single laboratory's error survives whole in
the label. Only 14 of the 65 trained on one measurement. The two rankings pick
13 of 65 molecules in common.

Over the 405 molecules measured three or more times the two agree at 0.96, 0.95
and 0.95 spearman, so the disagreement is about which molecules each can reach,
not how it orders the ones both can.

⛔ NEXT: owner's decision, not taken. It changes which molecules the ceiling AND
`floor_measured` can reach, so it changes what n17's headline distance means. A
run is registered as `n17_ceiling_ranks_label_error` to measure what it does to
the held-out rmse first.

---

## F. ⛔⛔ THE NEGATIVE CONTROL THAT EVERY CLEANING CLAIM IN THIS SUITE NEEDS

Measured 2026-09-21 by the n23 loop, verified from
`/Volumes/seagate/local_noise_run/loop_n23/random_drop_probe.json`.
This is the most consequential result of the campaign so far and it is not about
n23.

flab_thermo_tresanco, 672 antibodies, 448 per training block, onehot at 5600
features, 3 outer folds, random forest, outlier_p10 at level 0.5 with 40 of the
448 training molecules moved. Every arm removes the SAME NUMBER of molecules,
45 of 448, from the SAME folds. Only WHICH 45 differs.

    arm                                   folds won   mean gain, spearman
    heteroscedastic_nn drop                  3 of 3        +0.0697
    random, seed 1                           3 of 3        +0.0688
    random, seed 2                           3 of 3        +0.0349
    random, seed 0                           1 of 3        -0.0628
    oracle: the labels that actually moved   2 of 3        +0.0167
    activity_cliff_aware drop                0 of 3        -0.1157

⛔ THE CLEANING GAIN SITS INSIDE THE SPAN A RANDOM DROP COVERS. One random seed
in three matches it to within 0.0009 spearman. And the ORACLE -- removing
exactly the molecules whose labels moved, the best any detector could do -- is
worth LESS than two of the three random draws.

⛔ THE ORDERING BY DETECTION IS THE REVERSE OF THE ORDERING BY FOLDS WON. On
fold 0, 45 removed of 448 where chance takes 4.2 of the 42 that moved:
activity_cliff_aware 14, gaussian_process 11, ngboost 9, heteroscedastic_nn 5,
hetero_gp 2. The arm that finds the moved molecules at 3.3 times chance loses
every fold; the arm that barely beats chance wins every fold.

So on this dataset the gain is about removing a tenth of the training rows, not
about removing the RIGHT tenth.

### What this means beyond n23
EVERY cleaning method in this suite is scored as "folds won against no
cleaning": n02, n07, n11, n17, n18, n20, n23. None has ever been measured
against a same-size random removal on the same folds.

⛔ A "folds won" count cannot carry a cleaning claim without that control beside
it. A method that removes rows at all may win folds for a reason that has
nothing to do with which rows it removed.

⛔ NEXT, AND IT IS THE OWNER'S: whether a random-drop arm is ADDED to the
experiments. The n23 loop deliberately did not add one -- it is an arm, and
adding arms is not a loop's call. Two shapes are available and they are not the
same:
  - a random arm INSIDE each experiment, run every time, so every folds-won
    count has its control in the same file;
  - a one-off probe per method, cheaper, but it goes stale the moment anything
    about the fits changes.

Note what is NOT in doubt: the DETECTION numbers. How well a source ranks the
molecules that moved is measured against a known set and needs no such control.
It is the step from a good ranking to a better model that this breaks.
