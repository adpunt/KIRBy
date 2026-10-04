# f38 steps

**Claude's working plan, NOT signed off.** What the owner has signed off is in
`F38_DECISIONS.md`; where the two differ, that file wins.

The handoff is `F38_HANDOFF.md`. This file records what is being done and how. Tick a step only
when it has been done and checked.

## What the owner wants (2026-10-03)

1. **Cut the kNN search space in every noise experiment that sweeps it.**
   - Now: each neighbour-based cleaning method is refitted under 59 neighbour settings × 4 values
     of k, for every model and fold.
   - The cut must be backed by results.
2. **Check every noise experiment against the framework results.**
   - Where a framework experiment already settled a choice the noise suite makes, the noise
     experiment should follow it.

## Rules for this work

- Every server run happens inside one interactive session, with plain commands typed inside it.
  Never sbatch. The session line:
  `srun --clusters=htc --account=stat-cadd -p interactive --cpus-per-task=8 --mem=64G --time=03:00:00 --pty /bin/bash`
  Then, inside it: `cd /data/stat-ecr/scat9264/KIRBy`, then `export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh`,
  then one python command per line.
- One cut list per axis, used for every dataset. No cuts per dataset (owner, 2026-10-03). Each
  dataset's numbers are shown side by side, never averaged.
- A cut needs a stated reason: degenerate, broken or undefined. The owner decides each one.
- No cutoff is fixed in advance. The owner sees the full results, then sets the cutoffs
  (2026-10-03).
- No significance tests. Report per-fold scores and the spread.
- Local runs are debug runs, not results. Never run heavy computation on the laptop.
- Never write a verdict into the owner's documents. Any comparison column Claude writes is
  headed "Claude's reading — check".

## Part 1. What the framework experiments decided

**Input:** f34 onward only (owner, 2026-10-03): f34 (all its files), f35, f36 and f37. f00–f33 are
not relevant to the noise suite's choices.

**Output:** `tests/experiments/framework/F38_FRAMEWORK_INVENTORY.md`. One section per experiment:

| field | content |
|---|---|
| question | one sentence |
| computes | task, model, score and metric, what one output row is, axes and their values, datasets, folds |
| result | where it is (server path), finished or running, and whether the owner counts it as real. Filled from the owner, never assumed |
| what it settles | from its own output, numbers in the dataset's own metric. Empty until the owner says the result is real |
| noise-suite choice it bears on | e.g. distance, importance method, weight scaling, model list, smoothings, noise conditions, or none |

**How:**
- [ ] 1a. Parallel reading agents, each given a block of experiments. Each reads every line of
      its files and cites file:line.
- [ ] 1b. Claude opens each cited line and checks it before it goes in the file.
- [ ] 1c. The owner marks which results are real. f34_importance_sweep and f37 are still running.

## Part 2. What the noise and similarity experiments do

**Input:** every noise experiment, n01–n42 in `tests/experiments/noise/` (39 files), and every
similarity experiment, s01–s13 in `tests/experiments/similarity/` (12 files), each read in full.

**Output:** `tests/experiments/framework/F38_NOISE_INVENTORY.md`. One section per experiment.
For each choice below: the value it uses, and the file:line where it is set.
- neighbour settings and k: does it read `n07.sweep_settings()`, or a fixed setting of its own?
- models
- representations / model inputs
- importance method and power
- noise conditions, levels and draws
- smoothings and cleaning methods
- folds

It also gives the refit count per dataset: settings × k × methods × models × folds × draws ×
conditions.

**How:**
- [ ] 2a. Same as 1a–1b: reading agents, then Claude checks each cited line.

## Part 3. The two joined

**Output:** `tests/experiments/framework/F38_ALIGNMENT.md`.
- One row per (noise experiment, choice).
- Columns: what the noise experiment does (file:line), and the framework evidence (experiment,
  result).
- Plus "Claude's reading — check", one of: agrees / contradicts / no evidence / framework
  result not finished.

- [ ] 3a. Build the table from parts 1 and 2.
- [ ] 3b. Each "contradicts" row goes to the owner as a decision: the choice, what changes either
      way, the numbers.

## Part 4. The neighbour cut

The axes, and what the noise suite sweeps today:

| axis | values |
|---|---|
| distance | manhattan, euclidean |
| importance weighting | none, lightgbm_gain, treeshap, deeplift (power set by the distance: 1 manhattan, 0.5 euclidean) |
| feature space | base_1, base_2, base_1_aug_uncapped, base_2_aug_uncapped, hybrid, hybrid__general2, hybrid_similarity |
| k | 5, 9, 10, 20 |
| fixed settings | tanimoto_ecfp4, forest_proximity, model_input |

**4.1 What existing results already settle per axis.** This comes from part 1, which covers f34
(every file), f35, f36 and f37 only. Everything they don't settle goes to 4.2.

**4.2 The screen, with no refits.** Script: `tests/experiments/framework/f38_neighbour_screen.py`.
- What one task is: one (dataset, noise condition, fold, draw, model). The model matters because
  base_1, base_2 and the hybrids are that model's own.
- What it computes per setting × k:
  - **detection:** the AUROC of the injected-corruption list against |noisy label − neighbours'
    mean|, and the same with the median
  - **recovery:** RMSE between the neighbours' mean and the CLEAN label, on the corrupted
    molecules, in the label's own units. Printed beside the RMSE the noisy label had before any
    cleaning.
  - **overlap:** for each pair of settings that differ in one axis, the share of neighbours they
    have in common
- Settings scored:
  - the noise suite's 59
  - cosine, correlation and minkowski p0.5 on the same 7 spaces
  - each weighting at power 0.5 and 1
  - f34's own spaces (each representation, and its two kNN-built hybrids), so that question 5
    below can be answered
- Conditions: those where some training labels are corrupted and some are not. Detection is
  undefined where every label moves. After f37 finishes, the conditions follow the owner's cut
  of them.
- Datasets: f34's four (herg, herg_ki, caco2, adme_mdr1), plus the noise datasets the owner
  picks after seeing the cost (4.5).
- Output: `tests/results/framework/f38_neighbour_screen/<dataset>/<condition>__fold<i>__draw<d>__<model>.jsonl`,
  and the same with `.overlap.jsonl`. The run is resumable.
- Server commands, inside the srun session:
  `python tests/experiments/framework/f38_neighbour_screen.py --list-tasks f38_tasks.txt --datasets herg herg_ki caco2 adme_mdr1`
  `python tests/experiments/framework/f38_neighbour_screen.py --task caco2 outlier_p05 0 0 rf`

- [x] 4.2a. Detection and overlap written. 5 unit tests pass (`tests/test_f38_neighbour_screen.py`).
- [ ] 4.2b. Recovery score (written, not yet wired into the rows or tested).
- [ ] 4.2c. f34's spaces added.
- [ ] 4.2d. Debug run of one task on the server. The local attempt was stopped: it was building
      3D representations that aren't cached on the laptop.
- [ ] 4.2e. Check which conditions corrupt only part of the labels.

**4.3 Does the screen agree with real cleaning?** This uses n42, which refits each model after
cleaning under every setting and scores it on the held-out fold.
- Command, inside srun:
  `python tests/experiments/noise/n42_similarity_noise_sweep.py <dataset> --output <dir> --noise-condition <condition>`
- New script `tests/experiments/framework/f38_compare_n42.py`. It reads n42's rows and the
  screen's rows, and for each (dataset, condition, model, cleaning method) prints:
  - n42's best setting
  - its rank in the screen's detection list and in the screen's recovery list, out of N
- [ ] 4.3a. Time one n42 unit with every setting.
- [ ] 4.3b. Owner picks the datasets: at least 3, from the cost.
- [ ] 4.3c. Run, then compare.

**4.4 Does prediction quality predict cleaning quality? (question 5)**
- New script `tests/experiments/framework/f38_compare_f34.py`. On f34's four datasets and f34's
  spaces, it puts f34's kNN prediction ranking of the settings beside the screen's ranking.
- If they agree, f34's results can choose for the noise suite. If not, they cannot.
- [ ] 4.4a. Waits on f34_importance_sweep (running) and 4.2c.

**4.5 Cost.**
- [ ] 4.5a. Time one screen task. Report the hours per task and the number of tasks.
- [ ] 4.5b. From part 2's refit counts: refits per noise experiment now, and under each cut
      option.

**4.6 Results the owner sees.** New script `tests/experiments/framework/f38_analyse.py`, writing
`F38_RESULTS.md` (numbers only):
- one table per axis: a row is one value, a column is one dataset, a cell is the per-fold
  scores with their spread
- the full overlap results, per dataset and k
- 4.3's ranks and 4.4's rankings in full

**4.7 Options.** Per axis: the values, the numbers, the stated reason for each possible cut, and
the refits it saves. The owner sets the cutoffs and decides.

## Part 5. Activity-cliff prediction: blocked on the owner

- [ ] 5a. (a) Is a molecule's cliff partner among its k nearest neighbours? or (b) What is the
      prediction error on cliff molecules?
- [ ] 5b. Which pairs count as a cliff:
  - ECFP4 Tanimoto ≥ 0.55; then tanimoto_ecfp4 wins (a) by construction
  - matched molecular pairs, which exist for d2, fxa and mpro only
  - another rule
- [ ] 5c. The activity gap: 2.0 label units (n14), or 2 × the label's median absolute deviation
      (n01).
- [ ] 5d. Build it inside the screen once those are decided.

## Order

1. Parts 1 and 2 in parallel.
2. 4.2b–4.2e alongside them.
3. Part 3 and 4.1 once parts 1 and 2 are checked.
4. 4.5 (cost).
5. Owner picks datasets.
6. 4.2 full run and 4.3.
7. 4.4 once f34_importance_sweep is finished.
8. 4.6, then 4.7.

## How to run the screen (srun only)

In each srun (32 CPUs, 128G, 4 hours), type these three lines, with `--part 1/2` in one srun and
`--part 2/2` in the other:
```
cd /data/stat-ecr/scat9264/KIRBy
export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh
python tests/experiments/framework/f38_neighbour_screen.py --run-tasks f38_tasks.txt --jobs 32 --part 1/2
```
- When an srun hits its time limit, start a new one and type the same three lines. Finished tasks
  are skipped; a half-finished task resumes from its rows.
- Never delete `tests/results/framework/f38_neighbour_screen/`. On 2026-10-05 an `rm -rf` line in
  an earlier version of these instructions was pasted again into a new srun, and it deleted 120
  finished tasks.
- `f38_tasks.txt` was written once with
  `python tests/experiments/framework/f38_neighbour_screen.py --list-tasks f38_tasks.txt --draws 1`
  (300 tasks: 4 datasets, each dataset's two decided models, 1 draw). Don't rewrite it while a run is going.
- Measured 2026-10-04: about 60 minutes per round of 32 tasks on herg_ki, caco2 and adme_mdr1. herg
  (7772 molecules) is much slower, and its tasks come last.
