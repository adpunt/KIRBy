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
