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
held-out spearman, and saprot against esm2_cdr. A job `n07_shape_in_esm2_cdr` is
registered to remove the third.

A THIRD READING, 2026-09-21, and it sides with n18. n23 injects noise and sweeps
the same score-proportional move at 0.25, 0.5 and 1.0 on flab, 3 folds, level
0.5. The gentlest strength 0.25 is the best of the three in 2 of its 3
uncertainty sources, and the furthest strength 1.0 is negative in 2 of the 3.
So injected-against-real is not on its own the reason n07 and n18 disagree.

⛔ NEXT: which one sets the default is undecided. No default has moved.

### B2. The Gaussian process `gp_smooth` corrects toward is degenerate
At n07's width it returns 3 distinct posterior-mean values across 448 molecules —
the train mean. Collapsed at 1280, 361 and 20 features with bit-identical numbers;
fits at 200, 50 and 5. Best measured is 5 features at 0.4360 spearman against clean,
better than 200 by 0.1057.

That the collapsed rows are bit-identical across a sixty-fourfold range of widths
says optimiser fallback, not genuine flatness. A job `n07_gp_kernel_bounds` is
registered to print the fitted length-scale at each width and settle it.

⛔ NEXT: `gp_smooth` ranked second from bottom of the seven corrections at 13.4%
while smoothing toward a constant. The whole seven-way ranking may change.

### B3. Importance-weighting a kernel loses where the process actually fits
0.2709 against 0.3303 spearman at 200 features, 0.2182 against 0.2473 at 50. Two of
two, one fold. This is the opposite of the 29% gain the similarity work measured.

⛔ NEXT: unexplained. One fold.

### B4. n12's largest effect has never been followed up
In its existing result file the unweighted HYBRID beat the unweighted base
representation in 6 of 6 readings under both importance scorers — the largest and
most consistent effect in that file. Two of its eight conditions are secretly one
condition, agreeing to 1e-15, because the blueprint's own importance method is
`lightgbm_gain`.

### B5. Only one n21 comparison beat its own spread
Importance-weighted at exponent 1.0, lightgbm_gain, euclidean: +0.0120 average
precision over plain, spread 0.0056, won 4 of 4. Every other row of that table is
within its own noise at 2 folds and 1 draw.

⛔ NEXT: raise folds and draws until the top of the table separates, or establish
that it does not.

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

⛔ NEXT, and it is not a defect: one draw. The 3-of-3 count is the headline and
nothing has repeated it on a second draw. `hetero_gp` and `ngboost` were not in
that run's source list; the five-source run in onehot is with the supervisor as
`n23_cleaning_vs_no_cleaning`.

### B6b. Detection and cleaning do not pick the same source
`heteroscedastic_nn` has the weakest detection of the three, 0.1041 average
precision against 0.0893 for a coin flip, and owns the only arm that wins every
fold. `gaussian_process` has the strongest detection, 0.2495, its best arm wins 2
of 3 folds at +0.0234 spearman inside its own spread of 0.0544, and it costs 391
seconds per uncertainty estimate against 74 for `heteroscedastic_nn`.

⛔ NEXT: whether ranking the moved molecules is the right thing to select an
uncertainty source on at all. Nobody owns this.

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

---

## C. Defects found and not fixed

### C1. n11 discards its own primary claim to protect a caveated one
It returns `{'error': 'no_blueprint'}` at n11:760 and n11:804 when the pool holds no
recipe, before its level loop starts. The estimator-by-correction recovery needs no
recipe and is computed in every estimation space. Only the hybrid-versus-base
comparison needs one, and n11's own header calls that comparison oracle-biased and
says it must not be headlined.

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

- **n09, n11, n17 on the three local datasets**: no result file exists for n11. n17
  ran only on `herg_ic50`, because the three local datasets have no repeat
  measurements at all.
- **n17's one result is negative**: on `herg_ic50` in ecfp4, the ceiling did not
  beat the floor, and its own read-out says so rather than reporting a position.
- **Every n07 result file ever written has an empty detection block.** Eight
  gradings were recorded into a dict built before the loop that fills it. Fixed in
  `0f83448`, but no file written before that has the numbers.
