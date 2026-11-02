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

### A2. The fixed preference list no longer picked the control alphabetically (2026-09-21)
2026-09-25, owner's decision: the fixed preference list and the function that
applied it are deleted. The single-representation comparison in the noise
experiments is now each downstream model's top 2 single representations by that
model's score in the stored search. The text below records the 2026-09-21 state.

On `flab_thermo_tresanco` the control became `onehot`, was `esm2_cdr` — a learned
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
saprot. In the onehot the fixed preference list then returned for flab (list
deleted 2026-09-25), the same level
0.5 and the same tenth of the training molecules give a gap of 0.0644 spearman
against a clean score of 0.4068 — 15.8% of it — over 3 folds and 3 corruption
draws. The realised dose is 0.501 of the clean label standard deviation in both.
The ladder in onehot: 15.8% at level 0.5, 30.8% at 1.0, 61.9% at 2.0, 80.2% at
3.0. So the level n07 already runs is readable in its own control arm.

Three consumers: n07 (its single level), n13 (a marker on its curve), and
`noise_helpers`. Changing it moves all three differently.

⛔ NEXT: owner's decision, not taken. Measure what 1.0 gives n07 before changing it.

### A4. n35's flag reads the zero-distance intercept, not the closest bin (2026-10-26)
The published test reads the noise variance at distance zero; the flag read the closest
bin's value and the intercept never changed a weight. Fixed in n35; `flag_estimate` and
`flag_estimate_usable` are new fields on each distance space's record.
Every stored n35 result is on the old footing. Nothing has re-run.

### A5. n25's spared count reads against the column's untouched weight (2026-10-26)
`n_spared` compared the cleaned weights against put_back's rescaled column, so under
`downweight` and `huber_weight` every exempt molecule read as touched (open since
`4c85148`). It now reads a molecule as touched when its weight sits below the column's
largest. Scores do not move; `n_cliff_molecules_spared` and `n_random_molecules_spared`
do, for the two weighting cleanups only. Under `downweight` nearly every molecule still
reads as touched, because that weight lowers every molecule but the cleanest.

### A6. The cliff-aware gap's docstring now states the formula the code computes (2026-10-26)
No number moves. The thesis detectors table already carried the code's formula.

### A7. Forest proximity is counted over out-of-bag trees only (2026-10-26)
`forest_proximity_distance` and `similarity_space._forest_distance` count a pair only in
the trees whose bootstrap left both molecules out, divided by that count (Breiman and
Cutler's rule); before, every tree counted. `forest_leaf_ids` and `_forest_leaves` return
the out-of-bag mask too, and `cached_forest_leaf_ids` entries stored before today fail to
unpack (three elements now).
Every stored result that chose or swept the forest_proximity space is on the old footing:
n07, n12, n14, n17, n20, n21, n22, n23, n25, n32, n35, n37, n40, n42 import it. Nothing
has re-run. Open: an unparseable SMILES still gets methane's fingerprint in the Tanimoto
space (decision B in `thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md`).

### A8. MentorNet's seven inputs are seven different numbers (2026-10-26)
The loss history was shifted before the inputs were stacked, so this epoch's loss entered
twice. Every stored n09 and n13 result for `mentornet` is on the old footing. Whether the
arm keeps the name is decision H in the same file.

### A9. An unparseable SMILES is skipped in the Tanimoto space (2026-10-29)
Author's decision 2026-10-29. The loaders drop a SMILES RDKit cannot parse, so one should
never reach an experiment. If one does, `similarity_space._ecfp4_bits` now marks it and the
Tanimoto neighbour search gives it distance +inf to and from every molecule: it takes no
neighbours and is nobody's neighbour, and the count is written into the lists as
`n_unparseable_train` / `n_unparseable_val`. n01's `_compute_ecfp4_tanimoto_sim` gives it
similarity 0 to every molecule (so n14 and n25 put it in no pair), and n32's Tanimoto
distance does the same as similarity_space. Before this, such a molecule was given methane's
fingerprint, so two unparseable molecules were each other's closest neighbour at distance 0.
Every cached Tanimoto neighbour list and every stored result that read one is on the old
footing; the cache has no versioning, so the entries are wiped and rebuilt. Whether any
loader lets one through is not checked.

### A10. n02's two weighting arms use one score-to-weight step (2026-10-29)
Author's decision 2026-10-29 (decision A, option 1). The cliff-aware weight went through the
estimator's own `get_weights`: the raw score, 1 over 1 plus the score, strength 1.0 because
no caller passed one. The agreement weight went through `weights_from_scores`: min-max
scaling to 0..1, strength 3.0. Now both go through `weights_from_scores` at `DEFAULT_STRENGTH`
(n02 `compute_cliff_aware_weights`, the n02 `cliff_aware_weight` cleanup arm, and n42's copy
of it), so the two arms differ in the score only. Every stored n02 and n42 result for the
cliff-aware weight is on the old footing.

### A11. Confident learning calls cleanlab (2026-10-29)
Author's decision 2026-10-29 (decision C, option 1). `ConfidentLearning` now calls cleanlab 2.9.0. Classification: the network's class probabilities for every training molecule, predicted out of fold (5 folds, scaffold-grouped when the experiment passes groups), into `cleanlab.filter.find_label_issues`, which sets one threshold per class and reads the flags off the count table; the suspicion score is 1 minus cleanlab's label quality. Regression: `cleanlab.regression.learn.CleanLearning.find_label_issues`, the same call `noise_helpers.cleanlab_suspicion` makes for the detectors, with this module's network as the refitted model by default (`issue_model='dnn'`, about 80 network fits per call) or LightGBM (`issue_model='lightgbm'`, the detector's call; a LightGBM fit followed by a torch fit in one process crashes on a laptop, so it was not checked locally); the number flagged comes from cleanlab's own search, no percentile. The network is then retrained on the kept molecules. Which issue model the sweep runs is the author's sub-choice. Every stored n09 and n13 result for `confident_learning` is on the old footing. OPEN_THREADS A11.
Sub-choice open: for regression, whether cleanlab refits the sweep's network (the default,
about 80 network fits per fold per configuration; 52 s on 300 synthetic molecules at 40
epochs) or LightGBM (the detector's call) to find the issues. Checked locally: the network
path for regression and classification, and scaffold-grouped inner folds.

### A12. Uncertainty curriculum renamed self-paced learning; stages now reached; two variants tested and dropped (2026-10-29, closed 2026-10-30)
Author's decision 2026-10-29 (decision D). Renamed `SelfPacedLearning`, registry key `self_paced_learning`, citing Kumar, Packer and Koller 2010 with Bengio 2009 for the easy-first schedule; the thesis LNL-sweep row, the two results-table labels and the analysis lists (`ch7_numbers.py`, `ch7_numbers_expected.json`, `group_table/build.py`, `tables_tex.py`) carry the new name. Early stopping: the patience counter restarts at each stage, and running out of patience inside a stage that is not the last opens the next stage instead of ending the run; before, on one fold of buchwald_hartwig the run ended at epoch 40 in stage 1 of 4. `stages_reached` is written into the metadata. Two variants the author asked for are added under local test: `self_paced_uncertainty_weighted` (selection by loss; each molecule's loss divided by the spread of 10 dropout passes, mean 1 over the active set) and `self_paced_uncertainty_ranked` (selection by loss divided by that spread, so a confident mistake comes in last). Whether they stay in the sweep is the author's call after the one-fold check (OPEN_THREADS A12). Every stored n09 and n13 result for `uncertainty_curriculum` is on the old footing.
One-fold check (buchwald_hartwig, diff_ecfp4, fold 0 of 5, gaussian noise, draw 0, rmse,
lower is better; a local debug run, not a result). One row is one arm; the columns are the
noise level as a fraction of the clean training-label SD.

Kept checkpoint = any epoch (today's behaviour, `checkpoint_from='any'`). Every self-paced
arm kept epoch 9, inside the 20-epoch warm-up, at every level, so the staged training
changed nothing and the three arms are one network:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 16.030 | 16.378 | 16.623 |
| self_paced_learning | 15.756 | 15.462 | 16.848 |
| self_paced_uncertainty_weighted | 15.756 | 15.462 | 16.848 |
| self_paced_uncertainty_ranked | 15.756 | 15.462 | 16.848 |

Kept checkpoint = last stage only (`checkpoint_from='last_stage'`), so the prediction is the
self-paced network (kept epochs 81 to 100 of 100):

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 16.030 | 16.378 | 16.623 |
| self_paced_learning | 15.202 | 16.584 | 18.026 |
| self_paced_uncertainty_weighted | 15.563 | 16.434 | 18.404 |
| self_paced_uncertainty_ranked | 15.713 | 16.634 | 18.465 |

What this says: with the kept checkpoint unrestricted, every stored `uncertainty_curriculum`
score was a warm-up network, not the curriculum. With the curriculum's own network, on this
fold the curriculum is below the plain DNN at both noise levels and neither uncertainty
variant is above the plain DNN. Open for the author: which `checkpoint_from` the sweep runs
(default left at 'any'), and whether the two variants stay registered.

**Closed 2026-10-30 (author): only the plain self-paced rule stays; the kept checkpoint is the
last stage.** The two uncertainty variants were also run on one fold each of logd (ecfp4) and
herg_ki (ecfp4), gaussian noise at 0, 0.5 and 1.0 of the clean training-label SD, rmse, lower
is better. One row is one arm.

logd, 4022 train, 1006 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.520 | 0.640 | 0.674 |
| self_paced_learning | 0.536 | 0.651 | 0.843 |
| self_paced_uncertainty_weighted | 0.532 | 0.640 | 0.852 |
| self_paced_uncertainty_ranked | 0.531 | 0.633 | 0.847 |

herg_ki, 1129 train, 283 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.629 | 0.629 | 0.722 |
| self_paced_learning | 0.616 | 0.612 | 0.749 |
| self_paced_uncertainty_weighted | 0.617 | 0.616 | 0.758 |
| self_paced_uncertainty_ranked | 0.614 | 0.612 | 0.754 |

On three datasets the variants never separate from the plain rule by more than 0.01 rmse, and
every self-paced arm loses to the plain DNN at the highest noise level. Under noise the plain
DNN stops within about ten epochs; self-paced learning cannot end before its last stage, so
it trains on the noisy labels for 100 epochs. The author's decision: keep regular self-paced
learning only. `self_paced_uncertainty_weighted`, `self_paced_uncertainty_ranked`, the `mode`
and `checkpoint_from` arguments and the dropout-spread code are removed; the kept checkpoint
is the last stage, which is what makes the prediction the self-paced network. Registry:
`self_paced_learning` only. Every stored `uncertainty_curriculum` score (n09, n13) remains on
the old footing.

---

### A15. Contrast-to-divide renamed Jo-SRC weighting (2026-10-30)
Author's decision 2026-10-30 (`HANDOFF_NOISE_RESTRUCTURE.md` section 1; closes decision G of
`thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md` by renaming). Class `JoSRCWeighting`,
registry key `josrc_weighting`, was `ContrastToDivide` / `contrast_to_divide`. No code changed.
The rule, every five epochs the loss rank times one minus the prediction-variance rank over five
jittered copies of the features, down-weighting by 1/(1 + 3 x product), is the test Jo-SRC (Yao
et al. 2021, CVPR, refs.bib `yao2021josrc`) uses to call a sample in-distribution noisy, used here
as a weight. Left out of Jo-SRC and said so in the thesis row (`tab:noise:robust`): relabelling the
flagged molecules by the mean-teacher copy of the network, the two-view consistency loss, the
threshold schedule. Jo-SRC as published and two Jo-SRC + DivideMix combinations are separate
methods still to build. Changed with the name: the thesis LNL-sweep row and the two results-table
labels, `THESIS_OUTLINE.md`, `ch7_numbers.py` (reads the stored n09 files under the old key),
`ch7_numbers_expected.json`, `group_table/build.py`, `tables_tex.py`, the audit page block m64,
`NOISE_CODE_GIT_HISTORY.md`, `METHOD_PROVENANCE.md`, `NOVELTY_BRIEF.md`. Every stored n09 and n13
result is under the old key and is the same computation.

### A16. The in-loop methods' tests and actions split into detectors and cleanups (2026-10-30)
Author's decision 2026-10-30 (`HANDOFF_NOISE_RESTRUCTURE.md` section 1, step 2; commit
`f238dea`). `src/kirby/utils/lnl_parts.py`. Detectors in `ESTIMATOR_REGISTRY`
(`IN_LOOP_DETECTORS`), scored once after a 10-epoch warm-up of the suite's DNN:
`net_small_loss` (per-molecule loss), `net_loss_mixture` (DivideMix's mixture posterior),
`net_josrc_test` (Jo-SRC's loss rank times one minus the jitter-variance rank),
`net_meta_weight` (one minus Meta-Weight-Net's weight), `net_evidential_spread` (aleatoric
variance). Cleanups as n07 `cleaned_training_set` branches, in `SPLIT_CLEANUPS` and not in
`CLEANUPS`: `score_weight`, `spread_invvar_weight`, `network_rewrite`, `ema_rewrite`. No stored
result is affected: n07's sweep list is unchanged and the five published methods still run
whole. Two method classes changed without changing their prediction: `MetaWeightNet` records
its meta weight per input row; `EvidentialKernel` checkpoints its three spread heads with the
backbone, so its recorded epistemic spread is now from one network. Handoff section 9 records
where the handoff disagreed with the code (early learning, self-paced, the
noise_mitigation_methods filters, which are live in fed01 and molecular.py and were not
deleted).

### A17. MentorNet fitted before training and frozen; the data-driven variant added (2026-10-30)
Author's decision H 2026-10-30 (`thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md`).
`lnl_methods.MentorNet` is Jiang 2018's predefined-curriculum variant; until today its mentor
was fitted one step per epoch during student training. Now: 10 burn-in epochs at weight 1, the
mentor fitted on the burn-in's loss features against the self-paced rule
(`lnl_parts.self_paced_weight`) at five epoch fractions, frozen, then weighted training. The
mentor takes 8 inputs (the epoch fraction is new). `MentorNetDD` (`mentornet_dd`, the 13th
registry method) fits the mentor on a trusted set with half its labels corrupted; by default the
trusted set is the selection carve, which is NOT clean (metadata `trusted_source`), the trusted
set being the author's open choice (`HANDOFF_NOISE_RESTRUCTURE.md` section 8). Split parts:
detector `net_trusted_weight` (DD) in `IN_LOOP_DETECTORS`; `lnl_parts.repeated_weighting` runs
PD's rule as `net_small_loss` repeated every epoch, no mentor. Every stored n09 and n13 result
for `mentornet` is on the old footing; `mentornet_dd` has none.

## B. Measured, and the follow-up is not owned


### A13. Heteroscedastic GP runs the published method (2026-10-30)
Author's decision 2026-10-30 (decision E, option 1, then a comparison). `HeteroscedasticGP`
runs Kersting et al. 2007: a value GP and a noise GP fitted in turn, three passes, at most
1000 training molecules; per-molecule noise = squared residual plus posterior variance of the
value GP, the noise GP fitted to its log, its exponentiated prediction set as the molecule's
noise term in the value GP; the value GP predicts. Before, one GP with a shared noise term
had its in-sample residuals fitted by a small network, and a plain DNN weighted
1 / (1 + predicted residual) predicted, so the arm was a reweighted DNN. Every stored
`heteroscedastic_gp` result (n09, n13, n19, n40) is on the old footing.

One-fold check, 2026-10-30 (local debug run, not a result). Gaussian noise on the training
labels at 0, 0.5 and 1.0 of the clean training-label SD, draw 0, rmse, lower is better. One
row is one arm; "plain GP" is the new class run with one pass, so one shared noise term. The
GP arms fit on at most 1000 training molecules, the cap the old code already had.

herg_ki, ecfp4, 1129 train (1000 in the GP), 283 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.629 | 0.629 | 0.722 |
| plain GP | 0.556 | 0.590 | 0.649 |
| old code (DNN weighted by a network's estimate of the GP residual) | 0.633 | 0.626 | 0.712 |
| heteroscedastic GP (published) | 0.563 | 0.598 | 0.636 |

logd, ecfp4, 4022 train (1000 in the GP), 1006 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.520 | 0.640 | 0.674 |
| plain GP | 0.569 | 0.633 | 0.720 |
| old code | 0.516 | 0.602 | 0.699 |
| heteroscedastic GP (published) | 0.569 | 0.636 | 0.723 |

The published method scores like a plain GP: its per-molecule noise moves the score by at
most 0.013 on either dataset. The old code scores like a plain DNN, within 0.01 on herg_ki,
and beat it once, on logd at noise 0.5. Which of the two wins follows which of GP and DNN
wins on the dataset. The author's decision 2026-10-30: the published method only. The old
code is removed (it was registered as `gp_residual_noise_weighting` for the comparison only).


### A14. DivideMix trains on its noisy set with guessed labels; two faults fixed (2026-10-30)
Author's decision 2026-10-30 (decision F). `DivideMixReg` keeps the molecules its mixture
calls noisy and trains on them with the two networks' average prediction as the label, the
published co-guessing step; before, they were dropped from the epoch, so the arm was
co-teaching with a mixture as the selector. Two faults fixed: the mixture is fitted to the
min-max scaled loss (as published), not the log loss, which with no noise injected called
73% of herg_ki and 84% of logd noisy; and no checkpoint or patience until 10 divide-and-train
epochs after the warm-up, because the kept checkpoint had been epoch 11 in four of six cells.
Co-refinement and pair mixing stay out. Every stored `dividemix` result (n09, n13) is on the
old footing.

One-fold checks, 2026-10-30 (local debug runs, not results). Gaussian noise on the training
labels at 0, 0.5 and 1.0 of the clean training-label SD, draw 0, ecfp4, rmse, lower is better.
One row is one arm. "Old" is the code as it ran (noisy set dropped, mixture on the log loss,
checkpoint from any post-warm-up epoch). "Guessed labels" adds the published co-guessing step
only. "Corrected" is the code now: guessed labels, mixture on the min-max scaled loss, and
no checkpoint or patience until 10 divide-and-train epochs after the 10 warm-up epochs.

herg_ki, 1129 train, 283 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.629 | 0.629 | 0.722 |
| old | 0.618 | 0.610 | 0.718 |
| guessed labels | 0.618 | 0.609 | 0.670 |
| corrected | 0.621 | 0.599 | 0.659 |

logd, 4022 train, 1006 held out:

| arm | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| plain DNN | 0.520 | 0.640 | 0.674 |
| old | 0.527 | 0.613 | 0.745 |
| guessed labels | 0.532 | 0.597 | 0.731 |
| corrected | 0.534 | 0.599 | 0.772 |

Two faults the first comparison showed, both in the old and the guessed-labels code: (1) the
kept checkpoint was epoch 11 in four of six cells, one epoch after the warm-up, so the stored
DivideMix scores are mostly 11-epoch networks; (2) with no noise injected the mixture on the
log loss called 73% of herg_ki and 84% of logd noisy. After the fixes the kept checkpoint is
epoch 21 to 32 and the clean share with no noise is 68% (herg_ki) and 70% (logd). The
corrected code beats the plain DNN at noise 0.5 on both datasets and at 1.0 on herg_ki, and
loses to it at noise 0 on both and at 1.0 on logd.

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

RE-READ 2026-09-21 in `onehot`, which is what the fixed preference list then
returned for this dataset (list deleted 2026-09-25), at 2 folds by 3 draws and all three neighbour counts -- 36
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
The fixed preference list returning `onehot` on every antibody set is what put
the five sources at 5600 features in the first place. That list was deleted
2026-09-25; the comparison is now each downstream model's top 2 single
representations.

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
  At a removal share of 0.01 it beat the floor by +0.0071 rmse in log units of
  pIC50, which since 2026-09-23 is under the smallest distance n17 will divide
  by (2% of the no-cleaning score 0.7399 rmse, so 0.0148 rmse). See E5.
- **Every n17 number rests on ONE draw** of which measurement became the
  training label. The spread over draws printed in every result file is 0.0000
  for that reason, so no n17 number has a wobble to be read against. Three
  draws cost about 320 minutes of wall clock.
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

That run has since returned. On herg_ic50 at a removal share of 0.01 the
ceiling ranking on `training_label_error` comes back 0.7507 rmse in log units of
pIC50 against its floor at 0.7702, a distance of +0.0195 rmse. Under
`measurement_spread` on the same removal share the ceiling comes back 0.7471
against its floor at 0.7542, a distance of +0.0071 rmse. Both ceilings still
lose to no cleaning, which is 0.7399 rmse.

Since 2026-09-23 n17 refuses to divide a removal position by a distance under
`SPAN_MIN_FRAC` of the no-cleaning score, `SPAN_MIN_FRAC` = 0.02, which is
0.0148 rmse here. The +0.0195 rmse distance clears it and the +0.0071 rmse
distance does not. So the shipped default now reports no removal position on
this dataset, and the ranking that is not the default is the only one with a
distance n17 will divide by.

⛔ NEXT: owner's decision, not taken. It changes which molecules the ceiling AND
`floor_measured` can reach, so it changes what n17's headline distance means.

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

### F2. The same control for a MOVE arm, on herg_ki — MEASURED 2026-09-23
Verified from `/Volumes/seagate/local_noise_run/loop_n23/shuffled_move_probe.json`,
written by `loop_n23/shuffled_move_probe.py`. It adds no arm and no source to
n23.

herg_ki, 1412 molecules, 941 in each of the 3 training blocks, ecfp4 at 2048
features, rmse, LOWER IS BETTER, 3 outer folds, random forest, outlier_p10 at
level 0.5, one corruption draw. On this dataset every drop arm loses and
`move_proportional@0.25` wins, for all five uncertainty sources at once.

WHAT THE MOVE ARM DOES. `smooth_by_scores` at noise_estimation.py:1971-1972
rescales the spreads to run from zero to one and sets each training label to
`y + strength * rescaled_spread * (target - y)`, with `strength` 0.25.

THE CONTROL. The spread vector is permuted across the 941 training molecules
before the labels are moved, at 20 seeds. The same multiset of shrink fractions
is handed out, to different molecules. Same folds, same draw, same strength,
same target, same features, same forest seed.

    source                    real ranking        20 shuffled draws
    heteroscedastic_nn     2 of 3 folds won    0 to 3 folds won, 1.50 on average
                           mean +0.00305 rmse  mean gain -0.00602 to +0.00682 rmse
    activity_cliff_aware   2 of 3 folds won    1 to 3 folds won, 1.75 on average
                           mean +0.00197 rmse  mean gain -0.00436 to +0.00758 rmse

⛔ THE RANKING DOES NOT WIN MORE FOLDS THAN A PERMUTATION OF ITSELF. On
heteroscedastic_nn, 9 of the 20 shuffled draws won at least as many of the 3
folds as the real ranking and 3 of the 20 won more. On activity_cliff_aware, 10
of the 20 won at least as many and 5 of the 20 won more. So the herg_ki move
result is the shrinkage, not the detection: what earns the gain is pulling every
label a quarter of the way toward its out-of-fold prediction.

ON MEAN GAIN THE READING IS WEAKER BUT NOT NOTHING. The real ranking's mean gain
sits above 17 of the 20 shuffled draws on heteroscedastic_nn and above 12 of the
20 on activity_cliff_aware. `folds won` is the count n23's headline is stated
in; `mean gain` is not, and neither the mean gain nor the fold count clears the
standard deviation over the three fold gains, 0.00481 and 0.00310 rmse.

⛔ THE CONTROL MATCHES THE SHRINK FRACTIONS AND NOT HOW FAR THE LABELS TRAVEL.
On activity_cliff_aware, fold 0, the real ranking moves a label 0.0400 pKi units
on average and a shuffled draw moves it 0.0238, because the molecules with the
widest spread are also the ones furthest from their neighbour mean. That pair
differs in two ways at once, not one. On heteroscedastic_nn the same pair is
0.0668 against 0.0647 and is close to matched, so that source's reading is the
cleaner of the two.

⛔ THE INJECTED DAMAGE ON ONE FOLD IS BELOW THE FLOOR. Clean to corrupted, per
fold: 0.0429, 0.0060 and 0.0165 rmse against a floor of 2 percent of that fold's
clean score. The middle fold does not clear it. No share of the injected damage
is computed anywhere in the probe or above.

THE PROBE REPRODUCES n23's OWN NUMBERS. All 9 stored scores for these two
sources in `n23_uncertainty_cleaning/herg_ki__outlier_p10_cond3.json` — the
three uncleaned fits and the six move-arm fits — come back bit-identical.

⛔ NEXT, AND IT IS THE OWNER'S: one corruption draw, 3 folds, two of the five
sources, one strength. Whether a shuffled-ranking arm is ADDED to n23 beside the
random-drop one is the same open question section F already puts to the owner.

---

## G. The activity-cliff work: what blocks a reading of it

Added 2026-10-26, from a read of `n25_cliff_exemption.py`, `n01_activity_cliff_framework.py`,
`noise_estimation.py` and `ch7_noise_methods_audit.html` (the audit's numbers are quoted,
not re-measured here).

### G1. The cliff share is a free parameter and nothing has swept it
One pair of thresholds decides which molecules are cliff molecules:
`CLIFF_SIM_THRESHOLD = 0.55` ECFP4 Tanimoto and `CLIFF_ACTIVITY_K = 2.0` median absolute
deviations of the label column, n25:568-569. `--cliff-sim-threshold` and `--cliff-activity-k`
exist (n25:2228-2236) and no stored run passes either.

At that setting the audit records 1,740 to 3,056 cliff molecules out of 3,057 training
molecules on `buchwald_hartwig`, and every molecule the draw corrupted is a cliff molecule.
Putting the cliff molecules back and not cleaning at all are then the same action on that
dataset, so the arm pair `cliff_exempt` against `plain` cannot be read there.

At the other end, clean QM9 holds 19 cliff molecules out of 7,982, and the two arms differ
by about 0.003 rmse, which is the rerun spread.

⛔ NEXT: the cliff share per dataset, per fold, at the shipped setting, before anything else.
It is already computed inside the run (`n_cliff_training` out of `n_training`, n25:123) and
is not collected anywhere. Then the owner's decision: whether the two thresholds become a
swept axis in n25, and if so over what values.

### G2. The third arm is not size-matched where it is needed most
`random_exempt` puts back the same number of molecules, none of them on a cliff
(n25:29-33). On a dataset where most training molecules are cliff molecules there are not
enough non-cliff molecules to match the count, so the arm that exists to separate "cliffs are
special" from "cleaning less helps" is absent exactly where the cliff share is highest.
The audit states the control is not size-matched and does not say on which datasets it fell
short.

⛔ NEXT: record, per pass, how many molecules `random_exempt` could put back against how many
`cliff_exempt` put back. A pass where those two counts differ is not a pair and the audit
should not be quoting it. Fixing G1 may remove this on its own.

### G3. No experiment acts on the plain gap over the neighbour spread
`activity_cliff_aware` divides the gap by the sum of the label spread and the neighbour
spread. `local_variance_ratio` divides it by the neighbour spread alone, NE:425-497. The
second is the method the first is an addition to.

Both are graded side by side in n06, n07, n11, n12, n13, n21 and n22, so their detection
quality is comparable from stored results. Neither n18:350 nor n24:255 acts on either. n25:504
acts on `activity_cliff_aware` and not on `local_variance_ratio`; n42:237 takes n25's tuple.
So no stored result anywhere says whether ACTING on the cliff-aware form beats acting on the
plain form. The two differ in one thing, the extra spread term in the denominator, and that
one thing has never been isolated downstream.

⛔ NEXT: add `local_variance_ratio` to `n25.DETECTORS` (n42 inherits it). It costs one more
detector across n25's grid: one model fit per (pass, space, setting, model, cleanup, arm).
This is the comparison that was asked for on 2026-10-25 and the reason it could not be
answered from disk.

### G4. The detector's docstring and its code are different functions
NE:565 states `noise_score = deviation * (1 - local_variance_normalized)`.
NE:621-623 computes `deviation * 1 / (1 + local_variance_normalized)`.
Both are a damping term and they are not the same number: at a neighbour spread equal to the
label spread the first gives 0 and the second gives 0.5. The code is what ran.

⛔ NEXT: Chapter 8 names this method as new, so the formula written in the thesis has to be
the one at NE:621. Nothing in the thesis has been checked against it.

### G5. Datasets with no n25 result
`adme_mdr1` (no fold scored), `ddg_combined`, and the Ginkgo sets have no cliff-exemption
pass. One antibody dataset ran, `cdr_flexibility`. QM9 cannot carry the question (G1).

⛔ NEXT: whether the claim is small molecules only, or whether the missing datasets run.
