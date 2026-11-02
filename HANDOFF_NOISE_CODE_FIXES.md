# Handoff: fix the noise methods whose code is not the published method

Written 2026-10-26. Hand this whole file to a fresh chat. Nothing in it needs a cluster job.

## Status 2026-10-26

Fixes 1, 2, 3, 5 (the proximity; the methane fallback is open) and the duplicated mentor
input in 11 are done and committed. Fixes 4, 6, 7, 8, 9, 10, 11 and the methane fallback are
the owner's decisions, written one per heading in
`thesis_provenance/NOISE_CODE_FIX_DECISIONS_2026-10-26.md`, with what each option costs to
write and an agent's recommendation.

## What this is

All 79 noise methods in the Chapter 7 audit were compared against the original authors' own
code. Eleven came back with the published method's own mechanism absent or replaced, so a
result filed under the published name is not that method. Two of the eleven were deliberate
changes with the reason recorded in the commit; the rest were written once with no reason in
any commit message.

## Read these first, in this order

1. `~/Documents/ch7_noise_methods_audit.html` — open it and read block 5, "Code against the
   published code", on each of the eleven methods named below. Each block holds the links to
   the authors' own source, the full list of differences, the commits, and a "What to fix" line.
2. `thesis_provenance/NOISE_CODE_GIT_HISTORY.md` — the same eleven in a table at the top,
   then one section per method with the commits that wrote each difference.
3. `thesis_provenance/METHOD_PROVENANCE.md` — the earlier comparison, her code against the
   papers' TEXT. Different question, still useful for what each method was meant to be.
4. The code for whichever fix you are doing. Read the function in full before changing a line.

## Rules for this work

- **The owner decides between fixing and renaming.** For five of these, implementing the
  published mechanism is a build and renaming the arm is a sentence. Put both to her with the
  cost of each. Do not pick.
- **No cluster jobs.** If something has to run, it runs locally, or inside one
  `srun ... --pty /bin/bash` line with plain commands inside. Never `sbatch`.
- **Do not prune the search space** to make a re-run cheaper. Cutting an axis needs a reason:
  degenerate, broken, or undefined.
- **Every stored result for a method you change is on the old footing** and is not comparable
  to a new one. Say so in the commit message and add a line to
  `tests/experiments/noise/OPEN_THREADS.md` section A.
- **No significance tests.** Per-fold score pairs and the spread over folds.
- Write defects and open questions into the documents, not into chat replies.
- Commit messages must say WHY, not only what. 155 of the 273 commits behind these differences
  carry no reason, which is how this list got long.

## The fixes, cheapest first

### 1. Close-pair weights: the intercept is computed and never used
`tests/experiments/noise/n35_near_duplicate_disagreement.py`

The published test reads the noise level by carrying the mean squared label difference out to
zero distance. Her file computes that intercept at n35:326-332 and writes it to the result
file. The flag at n35:870 is handed `closest_bin_estimate` instead, so the intercept never
changes a weight.

Fix: pass the intercept to the flag. One line. Then the arm is the published test.

Cost: every n35 result is on the old footing. The arm is cheap to re-run locally.

### 2. Cliff put-back: the spared-molecule count reads against the wrong column
`tests/experiments/noise/n25_cliff_exemption.py`

`put_back` sets an exempt molecule's weight to 1 and then renormalises the whole weight
column to mean 1, at n25:902-905. `n_spared` at n25:931-935 compares the cleaned column
against that renormalised one, so under every down-weighting cleanup each exempt molecule
reads as touched.

Fix: compare against the un-renormalised column. The put-back itself needs no change.

Note: commit `4c85148` of 2026-09-23 already records this as open, in its own message. The
reason cliff molecules are spared is recorded in the same commit and is not in question.

Cost: the count changes, the scores do not. No refit needed.

### 3. Cliff-aware gap: the docstring and the code are different formulas
`src/kirby/utils/noise_estimation.py`

The docstring at :565 says the score is the gap times one minus the neighbour-spread ratio.
The code at :620-623 computes the gap times one over one plus that ratio. At a neighbour
spread equal to the label spread the first gives 0 and the second gives 0.5.

Fix: correct the docstring to the code. The code is what produced every number.

Then check the thesis: Chapter 8 names this method as new, so the formula written there has
to be the one at :621. Grep the thesis for the formula before and after.

Cost: none. No number moves.

### 4. Neighbour-agreement weights: the two arms differ in two ways at once
`tests/experiments/noise/n02_sample_weighting.py`

The agreement weight min-max scales the deviation and leans at strength 3.0 (n02:359-362).
The cliff-aware weight feeds the raw score into one over one plus the score, and the call at
n02:569-571 passes no strength, so it takes the function default of 1.0. That is a 2-to-1
lean against the agreement weight's 4-to-1. The two arms differ in the arithmetic AND in the
lean, so the gap between them cannot be read as cliff-awareness.

Fix: pass the same scaling and the same strength to both.

The 3.0 has a recorded reason (`e0e0301`, from an n18 sweep). The 1.0 does not. Changing
which one both arms use is the owner's call: state both and let her pick.

Cost: every n02 weighted result is on the old footing.

### 5. Forest proximity: computed on the molecules the forest was grown on
`src/kirby/utils/noise_estimation.py:2234-2277`, `tests/experiments/noise/similarity_space.py`

Breiman and Cutler's code counts a pair of molecules only in the trees that left both out of
the bootstrap, and divides by that count. Hers counts every pair in every tree, on the
molecules the forest was fitted to, and divides by the total tree count. The out-of-bag
restriction is what stops a proximity reporting the trees' memory of their own training rows.

Fix: track which molecules each tree left out, count a pair only in those trees, divide by
that count. sklearn exposes the bootstrap indices through
`sklearn.ensemble._forest._generate_unsampled_indices`.

Second thing in the same entry: `similarity_space.py:646-649` gives an unparseable SMILES
methane's fingerprint, which makes it a near neighbour of everything. Decide with the owner
whether such a molecule is dropped or kept with a flag.

Cost: forest proximity is one of the swept neighbour spaces, so every result that chose it is
on the old footing. This is the most expensive fix on the list. Count the affected experiments
before starting: n02, n07, n12, n13, n21, n23, n24, n25, n32, n42.

### 6. Confident learning: it is a percentile cut under cleanlab's name
`src/kirby/utils/lnl_methods.py:659-745`

cleanlab sets one probability threshold per class and builds a count table of given label
against confident class; how many molecules are removed falls out of those counts. Hers
trains 3 networks on all the training molecules, predicts the same molecules they trained on,
and flags those with error above the 85th percentile and spread below the 25th.

Fix: call `cleanlab.filter.find_label_issues`. The repository already installs cleanlab
(setup.py:136-140) and already calls it for a different detector,
`noise_helpers.cleanlab_suspicion`.

Owner's decision: do this, or rename the arm to a percentile cut and keep it as its own
method. Both are defensible; the current state is neither.

Cost: every n09 and n13 result for this arm is on the old footing.

### 7. Uncertainty curriculum: it orders by training loss, not by a spread
`src/kirby/utils/lnl_methods.py:1074-1231`

No spread across models, seeds or dropout draws is computed anywhere in the class. It sorts
molecules by the loss the network computed on the rows it trained on, and trains on the
lowest quarter, then half, then three quarters, then all.

Fix: compute a spread across the seeds or dropout draws the class already runs, and order by
that. Or rename the arm self-paced learning, which is what it is (Kumar 2010, not Bengio 2009).

Second thing: four stages of 45 epochs run against an early-stopping patience of 20 epochs
(lines 1134-1135, 1209-1211), so a run can finish before the later stages are reached. Check
whether any stored run reached stage 4 at all.

Cost: every n09 and n13 result for this arm is on the old footing.

### 8. Heteroscedastic GP (the training method): the two halves are not coupled
`src/kirby/utils/lnl_methods.py:2012-2145`

The published method fits a second Gaussian process over the log noise jointly with the mean
process, under one likelihood, so the estimated noise changes the mean fit. Hers fits one
scikit-learn process, trains a network on its residuals for 50 passes, then fits a DNN with
weights of one over one plus the predicted noise. The predicted noise never re-enters the
Gaussian process, which keeps one constant white-noise term and is never refitted.

Fix: refit the process with the predicted noise per molecule on the diagonal, and iterate.
`n19_measurement_count.py:1449-1462` already holds `gp_per_molecule_noise`, a process fitted
with a known noise per molecule, so the piece exists in the repository.

Also: the network's target is the residual on the rows the process was fitted on, so it is in
sample for up to 1,000 molecules. Out-of-fold residuals are available through
`NoiseEstimator._oof_apply`.

Cost: every n09, n13, n19 and n40 result for this arm is on the old footing.

### 9. DivideMix: the semi-supervised half is absent
`src/kirby/utils/lnl_methods.py:1238-1433`

The published method keeps the high-loss molecules as an unlabelled set and trains on them
with labels guessed by both networks, blends each clean label with the network's own averaged
prediction, and mixes sample pairs. Hers drops the high-loss molecules from the epoch at
lines 1372-1373 and does none of the rest.

Two options, both the owner's call:
- Implement the guessed-label training on the high-loss molecules. That is the defining step
  and it is the smallest change that earns the name. The pair mixing and the label blending
  can stay out if they are stated as left out.
- Rename the arm loss-mixture co-selection and report it as that. No code changes.

Cost of the first option: a build, plus every n09 and n13 result for this arm.

### 10. Contrast-to-divide: the contrast it is named for is missing
`src/kirby/utils/lnl_methods.py:1835-2005`

The published method pretrains the encoder with no labels, fits a two-component mixture to
the scaled per-sample loss, and has each of two networks train on the other's split. Hers
trains one network from random weights, with no pretraining and no mixture: it ranks the loss,
ranks the variance of the prediction across five noisy copies of the features, multiplies the
two ranks, and down-weights by the product.

Two options, both the owner's call:
- Add a self-supervised pretraining pass and the second network.
- Rename the arm. The selection rule shares no computation with the published one, so the
  name claims something the code does not do.

Cost of the first option: a build, plus every n09 and n13 result for this arm.

### 11. MentorNet: a decision, not a code fix
`src/kirby/utils/lnl_methods.py:1627-1832`

Do not revert this one without reading the commit first. `7a98aa1` of 2026-08-19 changed the
mentor's objective because the old one's "trivial optimum was all-zero weights -> collapse".
The change fixed a real collapse. What it cost is the learned, transferred curriculum the name
refers to: the mentor is now fit online, one step per epoch, to reproduce a fixed self-paced
formula of the current loss.

The owner's decision: supply a target weight per molecule and fit the mentor against it
before student training, which is the published shape, or rename the arm to self-paced
weighting. Renaming costs nothing and is honest.

Smaller thing in the same class, worth fixing either way: the current loss enters the mentor
twice, at lines 1748 and 1751, so two of its seven input features hold the same values.

## When you are done

- One line per fix in `tests/experiments/noise/OPEN_THREADS.md` section A, saying which stored
  results are now on the old footing.
- Update the method's block in `~/Documents/ch7_noise_methods_audit.html`: the "What to fix"
  line becomes what was done, and the judgement line changes.
- Update the row in `thesis_provenance/NOISE_CODE_GIT_HISTORY.md`.
- If an arm was renamed rather than fixed, grep the thesis and `THESIS_OUTLINE.md` for the old
  name and change those too.
