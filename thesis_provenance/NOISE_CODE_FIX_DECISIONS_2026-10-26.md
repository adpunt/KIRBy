# Noise code fixes: what was done on 2026-10-26, and seven decisions that are the owner's

Written by an agent on 2026-10-26 from `HANDOFF_NOISE_CODE_FIXES.md`. Every claim below
comes from reading the named code in this session. Recommendations are an agent's reading,
not a measured result. Nothing here ran on the cluster.

## Part 1. Done

Five of the handoff's eleven entries needed no decision. They are fixed, in one commit, with
the reason in the commit message. The smoke checks that cover them pass:
`_smoke_n21.test_forest_proximity_is_what_it_says`, `_smoke_n35.test_pure`,
`_smoke_n25.test_put_back`.

### 1. Close-pair weights (n35)

What disagreed: the file computes the noise variance two ways, the mean squared label
difference in the closest fifth of the close pairs (halved), and the straight-line intercept
of that quantity at distance zero. The published test (Hertz and Winstein 1988) reads the
intercept. The flag read the closest bin. So the intercept was computed, written to the
result file, and never changed a weight.

What changed: `n35_near_duplicate_disagreement.py`, the flagging call. The flag now reads
`intercept_estimate`. Two fields were added to each distance space's record:
`flag_estimate` says which estimate the flag read, and `flag_estimate_usable` is False when
the intercept is negative or undefined (fewer than two bins), in which case no pair is
flagged. That last rule is the file's existing rule for a negative estimate, unchanged.

### 2. Cliff put-back (n25)

What disagreed: `put_back` restores an exempt molecule's weight to 1 and then divides the
whole weight column by its mean. `n_spared` compared the cleaned column against that
rescaled one, so under both weighting cleanups every exempt molecule differed from its
restored weight and read as touched. Commit `4c85148` recorded this as open.

What changed: `n_spared` now reads a molecule as touched by a weighting cleanup when its
cleaned weight sits below the column's largest weight. Both weighting cleanups
(`downweight`, `huber_weight` in n07) give the molecules they leave alone the column's
largest weight, 1 before the column is divided by its mean, so the largest weight is the
untouched level.

The handoff's own wording, "compare against the un-renormalised column", would not have
done it: the cleaned column is itself divided by its mean, so a molecule the cleanup left
alone carries a weight of 1 over that mean, not 1, and would still have differed from a
restored 1.0.

What the count now says, checked on 120 random molecules with 20 exempt:
`huber_weight` spares 0 of 20 exempt molecules (its weight is flat at 1 inside the scale
and only one molecule of 120 sat beyond it); `downweight` spares 20 of 20, because its
weight falls smoothly from the very first molecule and lowers every molecule but the one
with the smallest score. The second number is a fact about the smooth weight, not a defect
in the count. Scores do not move; only this count does.

### 3. Cliff-aware gap (noise_estimation.py)

What disagreed: the docstring said the score is the gap times one minus the neighbour-spread
ratio. The code computes the gap times one over one plus that ratio. The code is what
produced every number.

What changed: the docstring, with the date and the reason. The code is untouched. The thesis
table (`thesis.tex`, the detectors table, row "Cliff-aware gap") already carried the code's
formula, 1 over 1 plus the ratio; so did `THESIS_OUTLINE.md`. Nothing in the thesis moved.

### 5. Forest proximity (noise_estimation.py, similarity_space.py, n21)

What disagreed: Breiman and Cutler's code counts a pair of molecules in the trees whose
bootstrap left both out, and divides by that count. Hers counted every pair in every tree
and divided by the tree count, on the molecules the forest was grown on.

What changed:
- `forest_leaf_ids` returns the out-of-bag mask beside the leaf ids (one row per molecule,
  one column per tree, True where that tree's bootstrap left the molecule out), read from
  sklearn's public `estimators_samples_`. A new helper `forest_out_of_bag` does the reading.
- `forest_proximity_distance` takes that mask and counts a pair only in the trees that left
  both molecules out, divided by the number of such trees. A pair no tree left out together
  gets distance 1. With 100 trees about 13.5 leave a given pair out together, so the share
  is now a vote over about 13 trees instead of 100.
- `similarity_space._forest_leaves` and `_forest_distance` do the same for the train-by-train
  and train-by-validation rectangles. A validation molecule was in no bootstrap, so a
  train-by-validation pair counts in the trees that left the training molecule out. The
  published code's rule for a test-set molecule was not checked; this is the agent's reading
  of the same leak, stated in the docstring.
- n21's inner held-back forest (`_inner_forest_distance`) now calls that function.
- Every caller updated: n07 `prox_smooth`, n14, n32, n21, `_loop_n21_forest.py`,
  `_smoke_n21.py`, and `noise_cache.cached_forest_leaf_ids`, whose stored value now has
  three elements. An entry stored before today has two and fails to unpack, which is the
  loud failure the no-versioning rule asks for.
- `thesis.tex`, the distance-family paragraph in the noise chapter: one sentence now says the
  share is counted over the trees whose bootstrap left both molecules out, marked
  `% [2026-10-26 fix]`.

Experiments that build a forest-proximity space, from the imports: n07, n12, n14, n17, n20,
n21, n22, n23, n25, n32, n35, n37, n40, n42.

### 11, in part. MentorNet's duplicated input (lnl_methods.py)

What disagreed: the loss history was shifted before the mentor's inputs were stacked, so
its last column was this epoch's loss, which was stacked again as the first input. Two of
the seven inputs held the same values. Now the inputs are this epoch's loss, the loss at
each of the five epochs before it, and the neighbour distance; the shift happens after.
The decision on MentorNet itself is Part 2, item G.

## Part 2. Seven decisions, one per heading

Each one: what the code computes today, what the published method computes, the two ways
to close the gap with what each costs to write, and the agent's recommendation. "Rename"
means the arm keeps its code and is reported under a name that says what it computes, in
the thesis, the outline, n09 and n13.

### A. Neighbour-agreement weights against cliff-aware weights (n02)

**Decided 2026-10-29 (author): option 1.** Built the same day. Both weights now go through `weights_from_scores` (min-max scaling, strength 3.0, divided by the mean): `compute_cliff_aware_weights` reads the estimator's scores and hands them to that function at DEFAULT_STRENGTH, and the n02 cleanup arm and n42's copy of it do the same. The two arms differ in the score only. Every stored n02 and n42 result for the cliff-aware weight is on the old footing. OPEN_THREADS A10.


Today: the agreement weight min-max scales each molecule's deviation from its neighbours'
mean to 0..1 and leans at strength 3.0, a 4-to-1 weight between the most and least suspect
molecule. The cliff-aware weight hands the raw cliff-aware score to 1 over 1 plus the score
at strength 1.0, because the call passes no strength. Two differences at once, the
arithmetic and the lean, so the gap between the two arms cannot be read as cliff-awareness.

The published method: none; both weights are hers. The 3.0 has a recorded reason
(`e0e0301`, from an n18 sweep on the same molecules). The 1.0 has none.

Option 1: send the cliff-aware score through the same function the agreement weight uses
(`weights_from_scores`, min-max scaling, strength 3.0). Three lines in
`compute_cliff_aware_weights`. The arms then differ in one thing, the score.
Option 2: send the agreement deviation through the estimator's own weight, 1 over 1 plus
the raw score at 1.0. Also three lines, but it undoes the 2026-09-18 change that fixed the
near-uniform weights, and 1.0 has no recorded reason.

Recommendation: option 1.

### B. An unparseable SMILES gets methane's fingerprint (similarity_space.py)

**Decided 2026-10-29 (author): skip the molecule.** Built the same day: `_ecfp4_bits` returns
an unparseable mask, the Tanimoto branch of `neighbour_lists` gives those molecules +inf to
and from everyone and records the count; n01's similarity matrix gives them 0 to everyone;
n32's Tanimoto distance does the same as similarity_space. OPEN_THREADS A9.


Today: in the Tanimoto neighbour space, a SMILES RDKit cannot parse is given methane's
fingerprint (`_ecfp4_bits`), copied from n01. Methane's fingerprint has one bit set, so the
molecule's Tanimoto distance to every other molecule is near 1, and two such molecules are
at distance 0 from each other. n35's own fingerprint function (`ecfp4_fingerprints`) returns
None for the same case instead.

How often this happens per dataset is not known; counting it is a local loop over the
loaders' SMILES.

Option 1: the molecule gets distance 1 to every molecule, is nobody's neighbour, and the
count of such molecules is written into the fold's result. About ten lines in `_ecfp4_bits`
and the Tanimoto branch of `neighbour_lists`. Option 2: keep methane and record the count.
Two lines.

Recommendation: count first, then option 1. Note that scipy's jaccard of two all-zero
vectors is 0, so an all-zero fingerprint is not a substitute for an explicit distance of 1.

### C. Confident learning (lnl_methods.py `ConfidentLearning`, n09, n13)

**Decided 2026-10-29 (author): option 1.** Built the same day. `ConfidentLearning` now calls cleanlab 2.9.0. Classification: the network's class probabilities for every training molecule, predicted out of fold (5 folds, scaffold-grouped when the experiment passes groups), into `cleanlab.filter.find_label_issues`, which sets one threshold per class and reads the flags off the count table; the suspicion score is 1 minus cleanlab's label quality. Regression: `cleanlab.regression.learn.CleanLearning.find_label_issues`, the same call `noise_helpers.cleanlab_suspicion` makes for the detectors, with this module's network as the refitted model by default (`issue_model='dnn'`, about 80 network fits per call) or LightGBM (`issue_model='lightgbm'`, the detector's call; a LightGBM fit followed by a torch fit in one process crashes on a laptop, so it was not checked locally); the number flagged comes from cleanlab's own search, no percentile. The network is then retrained on the kept molecules. Which issue model the sweep runs is the author's sub-choice. Every stored n09 and n13 result for `confident_learning` is on the old footing. OPEN_THREADS A11.


Today: trains three networks on every training molecule, predicts the same molecules,
flags those whose error is above the 85th percentile and whose spread across the three
networks is below the 25th, removes them, retrains. Two fixed percentiles, in-sample
predictions.

The published method (cleanlab): one probability threshold per class, a count table of
given class against confident class, and the number removed falls out of the counts. For a
continuous label cleanlab's regression branch scores each label from an out-of-fold residual
scaled by estimated uncertainty. The repository installs cleanlab 2.9.0 and
`noise_helpers.cleanlab_suspicion` already calls `cleanlab.regression.learn.CleanLearning`
with a LightGBM model for the detectors in n32 and n41.

Option 1: call cleanlab. Classification: `cleanlab.filter.find_label_issues` on out-of-fold
class probabilities from the DNN, then remove and retrain. Regression:
`CleanLearning.find_label_issues` as `cleanlab_suspicion` does, with the DNN or with the
LightGBM it already uses. About forty lines; the flagged set then has no fixed percentile.
Option 2: rename the arm "percentile cut on ensemble error and spread" and keep it as its
own method, no code change. Both are defensible; the current state is neither.

Recommendation: option 1, because the published method's own code is installed and already
called elsewhere in the suite, so the arm would be the same call the detectors use.

### D. Uncertainty curriculum (lnl_methods.py `UncertaintyCurriculum`, n09, n13)

**Decided 2026-10-29 (author): rename to self-paced learning, and build two uncertainty variants for a one-fold local check.** Built the same day. Renamed `SelfPacedLearning`, registry key `self_paced_learning`, citing Kumar, Packer and Koller 2010 with Bengio 2009 for the easy-first schedule; the thesis LNL-sweep row, the two results-table labels and the analysis lists (`ch7_numbers.py`, `ch7_numbers_expected.json`, `group_table/build.py`, `tables_tex.py`) carry the new name. Early stopping: the patience counter restarts at each stage, and running out of patience inside a stage that is not the last opens the next stage instead of ending the run; before, on one fold of buchwald_hartwig the run ended at epoch 40 in stage 1 of 4. `stages_reached` is written into the metadata. Two variants the author asked for are added under local test: `self_paced_uncertainty_weighted` (selection by loss; each molecule's loss divided by the spread of 10 dropout passes, mean 1 over the active set) and `self_paced_uncertainty_ranked` (selection by loss divided by that spread, so a confident mistake comes in last). Whether they stay in the sweep is the author's call after the one-fold check (OPEN_THREADS A12). Every stored n09 and n13 result for `uncertainty_curriculum` is on the old footing.

One-fold check, 2026-10-29 (numbers in OPEN_THREADS A12): with the kept checkpoint unrestricted,
every self-paced arm kept epoch 9 of the warm-up, so the three arms were one network and every
stored `uncertainty_curriculum` score was a warm-up network. With the checkpoint restricted to
the last stage (`checkpoint_from='last_stage'`, added, default 'any'), on that fold the
curriculum sits below the plain DNN at noise 0.5 and 1.0 and neither uncertainty variant sits
above the plain DNN. Two sub-choices open: `checkpoint_from` for the sweep; whether the two
variants stay registered.

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


Today: after 20 warm-up epochs, sorts molecules by the loss the network computes on the
rows it trained on, and trains on the lowest quarter, then half, then three quarters, then
all, 45 epochs each. No spread across models, seeds or dropout draws is computed anywhere
in the class. That is self-paced learning (Kumar 2010), not a curriculum by uncertainty.

Second thing, in the same class: early stopping has patience 20 epochs and the fourth
stage starts at epoch 155, so a run can stop before the later stages are reached. Whether
any stored run reached the last stage could not be checked: the local n09 result files
hold no `final_inclusion_pct` field. The field is written now.

Option 1: order by a spread. The class already runs dropout 0.2, so at each stage boundary
the network can predict every training molecule K times with dropout on, and the spread of
those K predictions orders the molecules, most certain first. About fifteen lines; K is a
new constant (the handoff's rule: a new constant needs a reason, so say K and why).
Option 2: rename to self-paced learning, cite Kumar 2010 instead of Bengio 2009. No code
change.

Either way the patience question needs an answer. One possibility: reset the patience
counter and the best loss when the stage changes, so each stage gets its 45 epochs unless
its own 20 epochs show no gain. Three lines.

Recommendation: option 2 plus the patience reset. The arm as written is a clean self-paced
learner and the thesis can say so; building a dropout spread adds a constant with no sweep
behind it.

### E. Heteroscedastic GP (lnl_methods.py `HeteroscedasticGP`, n09, n13, n19, n40)

Today: fits one scikit-learn Gaussian process (one white-noise term for the whole fold, up
to 1,000 molecules), trains a small network on that process's in-sample residuals for 50
passes, then fits a DNN with weights of 1 over 1 plus the predicted noise. The predicted
noise never re-enters the Gaussian process, and the prediction comes from the DNN.

The published method: a second Gaussian process over the log noise, fitted jointly with
the mean process under one likelihood, so the estimated noise changes the mean fit, and the
prediction is the mean process.

Option 1: iterate. Fit the process, take out-of-fold residuals, train the noise network,
refit the process with the predicted noise per molecule on the diagonal (the piece n19
already holds as `gp_per_molecule_noise`, which sets `alpha` per molecule), and repeat two
or three times. Predict with the final process. About forty lines, and the iteration count
is a new constant. The DNN step is then gone.
Option 2: rename to "GP-residual noise weighting": the DNN weighted by a network's estimate
of the Gaussian process's residual. No code change.

Recommendation: option 1, because the piece that is missing already exists in n19 and the
iteration is what the published name promises; without it the arm is a weighted DNN.

**Decided 2026-10-30 (author): option 1, then compared; the published method is kept.** Built
the same day. `HeteroscedasticGP` now runs Kersting, Plagemann, Pfaff and Burgard 2007: a
value GP and a noise GP fitted in turn, three passes, on at most 1000 training molecules. The
first pass fits the value GP with one shared white-noise term. Each later pass takes, per
molecule, the squared residual plus the posterior variance of the value GP at that molecule
(the expected squared residual under the posterior, Kersting's sampled estimate in closed
form), fits the noise GP to the log of those, and puts the exponentiated prediction on the
value GP's diagonal (`alpha` per molecule, the piece n19's gp_per_molecule_noise holds). The
value GP predicts. Labels are z-scored for the fit. `N_ITER = 3` is the one new constant. The
thesis LNL-sweep row (thesis.tex, Ch7) now describes this and cites Kersting without "after".
Every stored `heteroscedastic_gp` result (n09, n13, n19, n40) is on the old footing.
OPEN_THREADS A13.

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

### F. DivideMix (lnl_methods.py `DivideMixReg`, n09, n13)

Today: two networks; each epoch a two-component mixture on each network's log loss splits
the molecules into clean and noisy; each network trains on the other's clean set; the noisy
set is dropped from the epoch.

The published method: keeps the noisy set as unlabelled molecules and trains on them with
labels guessed by both networks, blends each clean label with the network's own averaged
prediction, and mixes sample pairs.

Option 1: train on the noisy set with guessed labels, the average of the two networks'
predictions on it, which is the defining step and the smallest change that earns the name.
About twenty-five lines. The pair mixing and the label blending stay out and are stated as
left out.
Option 2: rename to "loss-mixture co-selection". No code change.

Recommendation: option 1.

**Decided 2026-10-30 (author): option 1, compared, then two faults fixed; the corrected code
stays.** `DivideMixReg` now keeps the molecules its mixture calls noisy and trains on them with
the average of the two networks' predictions as the label (classification: the class with the
highest average probability); each network trains on all molecules, with the other network's
split deciding which label a molecule carries. Co-refinement (blending clean labels with the
network's own prediction) and MixMatch pair mixing stay out; the thesis row says so. Two
faults fixed the same day: the mixture is fitted to the per-molecule loss scaled to [0, 1] by
its min and max, as published, not to the log of the loss, with clean = posterior of the
low-loss component above 0.5; and no checkpoint is kept and no patience counted until
`min_divide_epochs = 10` divide-and-train epochs after the warm-up (`best_epoch` is written to
the metadata). Every stored `dividemix` result (n09, n13) is on the old footing. OPEN_THREADS
A14.

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

### G. Contrast-to-divide (lnl_methods.py `ContrastToDivide`, n09, n13)

Today: one network from random weights; every five epochs it ranks the loss, ranks the
spread of the prediction over five noisy copies of the features, multiplies the two ranks,
and down-weights by the product at strength 3.0. No pretraining, no mixture, one network.

The published method: pretrains the encoder with no labels, fits a two-component mixture
to the scaled per-sample loss, and has each of two networks train on the other's split.
The selection rule here shares no computation with that one.

Option 1: a self-supervised pretraining pass over the features (for a feature matrix, a
denoising autoencoder is the nearest thing to the image contrastive pass), then the mixture
and the second network, which `DivideMixReg` already holds. About a hundred lines and a
pretraining recipe with its own constants.
Option 2: rename to "perturbation-consistency weighting". No code change.

Recommendation: option 2. The published method's contrast is an image-augmentation
pretraining; there is no stated recipe for it on a fixed feature matrix, and inventing one
is design, which is the owner's.

### H. MentorNet (lnl_methods.py `MentorNet`, n09, n13)

**Decided 2026-10-30 (author): fix it.** The code was Jiang 2018's predefined-curriculum variant
(MentorNet PD) with one departure, the mentor fitted during student training instead of before
it. Built the same day: the mentor is now fitted BEFORE weighted training and frozen, as
published. `MentorNet` runs a burn-in of 10 epochs with every weight 1, fits the mentor, then
trains the student under the frozen mentor's weights. PD fits the mentor on the features the
student produced over the burn-in, paired with the self-paced rule's weight at five epoch
fractions (`lnl_parts.self_paced_weight`: weight near 1 below a loss cut-off that rises from the
median to every molecule). The data-driven variant, MentorNet DD, is added as `MentorNetDD`,
registry key `mentornet_dd`: the mentor is fitted on a trusted set with half its labels
corrupted on purpose, weight 1 for an untouched label and 0 for a corrupted one. The trusted
set is `trusted_idx` when given, else the method's own selection carve (the same rows
Meta-Weight-Net uses as its meta set), which carries the training labels' noise; metadata
`trusted_source` says which. Mentor inputs are 8: the loss, its 5 previous values, the
neighbour distance and the epoch fraction. Split parts: DD's test is the detector
`net_trusted_weight`; PD's test is `net_small_loss` under the rounds axis, built as
`lnl_parts.repeated_weighting`. Every stored n09 and n13 result for `mentornet` is on the old
footing. OPEN_THREADS A17.

Local check, herg_ki, ecfp4, fold 0 of 5 (1,129 training molecules, 283 held out), gaussian
noise on every training label at 0, 0.5 and 1.0 of the label SD, RMSE (lower is better). Debug
only, not a result. One row is one noise level.

| noise | plain DNN | MentorNet PD, rebuilt | MentorNet DD | small-loss rule repeated, no mentor |
|---|---|---|---|---|
| 0 | 0.629 | 0.621 | 0.622 | 0.637 |
| 0.5 | 0.629 | 0.620 | 0.620 | 0.627 |
| 1.0 | 0.722 | 0.750 | 0.751 | 0.779 |

All three weighted arms kept the network from epoch 11, the first weighted epoch after the
10-epoch burn-in, at every level, so the weights acted for one epoch before the checkpoint.
Whether a minimum number of weighted epochs is enforced before the checkpoint, as DivideMix
does since decision F (`min_divide_epochs`), is the author's choice; it was not done here.


Today: the mentor is a one-hidden-layer network fit online, one step per epoch, to
reproduce a fixed self-paced formula of the current loss with an annealed threshold. Commit
`7a98aa1` (2026-08-19) changed it to this because the earlier objective, minimise weight
times loss, had all-zero weights as its best answer. The change fixed a real collapse. What
it cost is the learned, transferred curriculum the name refers to.

The published method: the mentor is fit in a separate run against a supplied target weight
per sample, then frozen while the student trains.

Option 1: supply a target weight per molecule and fit the mentor against it before student
training, then freeze it. The target has to come from somewhere; the published code uses
either a predefined curriculum or a held-out clean set, and this suite holds no clean set a
method may read. About fifty lines plus the choice of target. Do not revert the objective.
Option 2: rename to self-paced weighting. No code change.

Recommendation: option 2. The duplicated input is already fixed (Part 1).

### I. Jo-SRC as published: the clean test on a regression label (added 2026-10-30)

Jo-SRC (Yao 2021) calls a sample clean when 1 minus the Jensen-Shannon divergence between
the network's class probabilities and its label is above a threshold that rises from 0 to
0.75 over the warm-up and from 0.75 to 0.95 by the last epoch (official code,
config/cifar100.cfg and main.py). A molecule that fails the test and whose two jittered views
agree is relabelled with the mean-teacher's prediction; one whose views disagree is treated as
out of distribution. On a regression label there are no class probabilities. Built 2026-10-30
as `lnl_methods.JoSRC`, registry key `josrc`, with the regression test as a switch:

- test A, `clean_test='squared_error'`: the squared error on the z-scored label; unbounded,
  so clean = the 70 percent with the smallest error (small-loss selection's share) after the
  warm-up.
- test B, `clean_test='bounded'`: exp(minus that error), in (0, 1], with the published
  threshold schedule as it stands.

Both: the views agree when their gap is at or below the 90th percentile of that gap over the
clean molecules; out-of-distribution molecules enter neither loss term (no regression form
of the flattened target or the push-apart). Classification runs the published test.

One-fold checks, 2026-10-30 (local debug runs, not results). Gaussian noise on the training
labels at 0, 0.5 and 1.0 of the clean training-label SD, draw 0, ecfp4, rmse, lower is
better. One row is one arm. "relabelled" is the share of training molecules the test called
noisy with agreeing views at the last epoch, which the method relabels with the teacher's
prediction.

herg_ki, 1129 train, 283 held out:

| arm | 0.0 | 0.5 | 1.0 | relabelled |
|---|---|---|---|---|
| plain DNN | 0.629 | 0.629 | 0.722 | |
| Jo-SRC, test A | 0.720 | 0.718 | 0.730 | 26 to 28% |
| Jo-SRC, test B | 0.720 | 0.738 | 0.763 | 10 to 30% |

logd, 4022 train, 1006 held out:

| arm | 0.0 | 0.5 | 1.0 | relabelled |
|---|---|---|---|---|
| plain DNN | 0.520 | 0.640 | 0.674 | |
| Jo-SRC, test A | 0.696 | 0.712 | 0.840 | 27 to 29% |
| Jo-SRC, test B | 0.514 | 0.657 | 0.805 | 2 to 17% |

What the numbers show. On herg_ki with no noise, both tests give predictions with a spread
of 0.41 label units against 0.76 for the same loop with nothing relabelled (keep share 1.0,
rmse 0.605, below the plain DNN). The relabelled target is the teacher's own prediction,
which on a regression label sits nearer the mean than the label does, so every relabelled
molecule pulls the fit toward the mean. In the published method the target is a class
distribution, where that pull does not exist. Test B relabels fewer molecules when the fit is
tight (logd, 2 percent with no noise, where it sits 0.006 under the plain DNN) and more as the
threshold rises late in training. Test A relabels a fixed 30 percent whatever the labels.
Neither test beats the plain DNN under injected noise on either dataset on this fold.

**Decided 2026-10-30 (author): Jo-SRC as published is cut, and the two Jo-SRC + DivideMix
combinations are not built.** Reason: both regression forms of its clean test lost to the
plain DNN in every cell with noise on both datasets, and the loss comes from the method's
defining step, relabelling with the teacher's prediction. The class was removed from
`lnl_methods.py` (it is in git history at the commit before this note). What stays:
`josrc_weighting` (Jo-SRC's test as a weight, the arm that ran in n09 and n13), the detector
`net_josrc_test`, and the one-shot cleanup `ema_rewrite` (the flagged labels replaced by the
mean-teacher's prediction), which is a cleanup paired with any detector in the
detector-by-cleanup sweep rather than the in-loop method; the author may cut it too.

## Part 3. What else was written

- `tests/experiments/noise/OPEN_THREADS.md` section A: items A4 to A8, one per fix, saying
  which stored results are on the old footing.
- `thesis_provenance/NOISE_CODE_GIT_HISTORY.md`: the judgement rows and sections for the
  five fixed entries.
- `~/Documents/ch7_noise_methods_audit.html`: the judgement and "What to fix" lines for the
  same five, and the summary table.

Not committed: `OPEN_THREADS.md`, `THESIS_OUTLINE.md` and the `thesis_provenance/` files
other than this one carried uncommitted edits from before this session, so they are left
for the owner to commit with those.
