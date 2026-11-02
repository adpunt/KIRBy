# opig_generator_scoring/ — support files for `../opig_generator_scoring_swap.py`

The experiment is one level up, flat with the other twelve `opig_*` scripts:
`tests/experiments/opig_generator_scoring_swap.py`. This folder holds only what it needs:
REINVENT's published configuration, unedited, and the scorer REINVENT calls through a pipe.

## What is being tested

A molecule generator keeps a property predictor inside its loop. That predictor is handed one
fingerprint by convention. This changes the features the predictor sees, changes nothing else,
and measures the molecules that come out.

## The generator is taken, not written

REINVENT 4, Apache 2.0, `github.com/MolecularAI/REINVENT4`, pinned at commit
`660d2c9cec9ea1ced1b452394fd38452e478fc8b` (2026-09-23).

`reinvent_published/` holds three of their files copied byte for byte and never edited:
`staged_learning.toml`, `stage1_scoring.toml`, `stage2_scoring.toml`. Their batch size of 64
molecules per step, their 100 steps per stage, their sigma of 128, their learning rate of
0.0001, their `IdenticalMurckoScaffold` diversity filter and their two-stage termination all
come from those files.

**No line of REINVENT is edited in any arm.** Their `ExternalProcess` scoring component, which
ships in their repo at `reinvent_plugins/components/comp_external_process.py`, hands a batch of
SMILES to an executable on stdin and reads `{"version": 1, "payload": {...}}` from its stdout.
`kirby_reinvent_score.py` is that executable. The only change to their configuration is one
added `[[component]]` block per arm, written by the experiment.

Their prior model files are not in their repo. They are on Zenodo, doi
`10.5281/zenodo.15641296`, and have to be downloaded once.

## The arms

Every arm fits the same model class, `lightgbm`, on the same 1,412 molecules of `herg_ki`, and
runs the generator with the same settings. Their three published scoring components are kept in
every arm. Only the features the predictor sees differ.

| arm | features the predictor sees |
|---|---|
| B | every feature of ecfp4. The convention the claim is about. The control. |
| C | the blueprint `find_best_hybrid` picks, `objective='predictive'`, one per budget |
| D | the same number of features as C, chosen at random from the same representations |
| E | every feature of whichever single representation scores best alone on `herg_ki` |

**Arm A is dropped**, by the owner on 2026-09-28, with the reason recorded. REINVENT's published
scoring function rewards molecular weight between 200 and 500 daltons, scores 14 substructures
as zero, and rewards drug-likeness. It predicts no measured property. An arm steered by it would
differ from arm B in two ways at once, not one, so its number could not be read.

**Arm D is not the brief's arm D**, and this is the one place the build departs from what was
asked. The brief asked for a random Gaussian matrix of the same width as C. A matrix of noise
has no rule mapping a molecule to a row, so a predictor fitted on one cannot score a molecule
the generator invents, and every arm here has to score unseen molecules. So arm D holds the
width and the representations fixed and removes only the search's choice of which features. A
random Gaussian *projection* of the concatenated representations, which is what
`similarity/s11_bayesian_optimization.py` calls `random_50d`, is a different null: it tests compression
rather than selection, and it would be a second arm rather than a replacement.

## The direction, and why it points the way it does

The generator is steered to **maximise predicted hERG pKi**. High hERG affinity is not a
desirable property in a drug. The direction is chosen because the model that scores the output
is a hERG blocker classifier, so the steered property has to be hERG affinity. Both sides move
the same way, and every arm moves the same way.

## What scores the molecules that come out

A model that both steers the generator and scores its output makes a better steering model win
by definition. So the output is scored by a classifier fitted on different molecules: `herg`,
the FLuID binary blocker set.

Counted on 2026-09-28, not assumed:

- `herg_ki` holds 1,412 molecules, after 3 molecules with a repeated InChIKey were dropped from
  the 1,415 it loads.
- `herg` holds 7,772 molecules: 3,630 blockers and 4,142 non-blockers.
- 166 molecules appear in both datasets.
- Those 166 come out of `herg` before the classifier is fitted, leaving 7,606 molecules
  (3,514 blockers, 4,092 non-blockers). `herg_ki` keeps all 1,412 of its own.

The classifier is reported on **two** representations, so the result does not hang on one
choice: ecfp4, which is also what arm B steers on, and whichever of `herg`'s own ten
representations scores best on its own. Its scaffold-grouped held-out AUROC per fold is
reported beside every arm's numbers, so a reader can see how good the thing doing the judging
is.

Docking was considered and is not used. No protein structure file exists anywhere in this repo.
`vina==1.2.6` is pinned at `requirements-working.txt:476` and is imported nowhere. And docking
needs one protein: hERG blockade has no single one. `mpro` is the only single-protein dataset
here and it is not among the 75 datasets in `dataset_policy.py`.

## What is reported, per arm

- The blocker probability of the single best molecule, and the mean over the best 10 and the
  best 100 molecules, under each of the two classifiers.
- How many of the molecules generated were valid, and how many were unique and valid. Both as
  a count out of the number generated. REINVENT's own `SMILES_state` column supplies these:
  0 is invalid, 1 is valid, 2 is duplicate, from
  `reinvent/models/model_factory/sample_batch.py`.
- How many of the unique valid molecules are novel against the 1,412 training molecules, by
  InChIKey, as a count out of the number unique and valid.

Raw counts come before any ratio, and what the top and the bottom of every fraction count is
named in the same sentence.

## How to run it

```bash
# local: fit the classifier and every arm's predictor, write the bundles and TOML files
OMP_NUM_THREADS=1 KIRBY_LGBM_THREADS=1 \
  python ../opig_generator_scoring_swap.py --stage fit

# then, where REINVENT is installed: one command per arm, all written into run_arms.sh
bash <output>/run_arms.sh

# read the molecules back and score them
python ../opig_generator_scoring_swap.py --stage collect
```

`OMP_NUM_THREADS=1` is required locally. Without it the process dies with exit 139 and no
traceback, because torch and LightGBM link two different OpenMP runtimes.

## Why this is not a registered category

`run_dataset.py` runs a category against **one** dataset: its four entries — complexity,
similarity, noise, federated — each supply a `{category}_helpers.py` with `run_all()` and
`SINGLE_DATASET_EXPERIMENTS`, and the runner is invoked as `--category X --dataset Y`.

This experiment takes a **pair** of datasets, one to steer the generator and a different one to
score its output. It has no single dataset argument to hand that runner, so forcing it into the
registry would mean inventing one. It is run directly, flat in `tests/experiments/` with the other twelve `opig_*`
scripts, which are not a registered category either. Only its support files sit in a
subfolder, because REINVENT's configuration is three files rather than one.

## Not yet verified

- No arm has been run. The generator is not installed here, and no prior model file has been
  downloaded.
- Wall-clock for one arm is unmeasured. Their settings generate at most 12,800 molecules per
  arm, and `kirby_reinvent_score.py` is launched once per step, so 200 times per arm, loading
  its pickle each time.
- `chembert` is one of `herg_ki`'s ten representations and has never been checked against the
  code that builds it.
