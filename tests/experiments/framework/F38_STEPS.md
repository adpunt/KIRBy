# f38 steps

The handoff is `F38_HANDOFF.md`. Every server run happens inside one
`srun ... --pty /bin/bash` session, with plain commands typed inside it. Never sbatch.
Tick a step only when it has been done and checked.

## A. Screen script: `f38_neighbour_screen.py`

- [x] A1. Written. It scores each neighbour setting × k on detection: the AUROC of the
      corrupted-molecule list against |noisy label − neighbours' mean|, and the same with the median.
      The settings are the noise suite's 59, plus cosine, correlation and minkowski p0.5, plus
      power 0.5 and 1 for each weighting. It also computes neighbour-list overlap.
- [x] A2. Unit tests: `tests/test_f38_neighbour_screen.py`, 5 pass.
- [ ] A3. Add a recovery score with no refits: how close the neighbours' mean is to the CLEAN
      label on the corrupted molecules. This is what label_smooth, knn_smooth and n02's weights
      depend on.
- [ ] A4. Debug run on caco2 (outlier_p05, fold 0, draw 0, rf). It only checks that the
      script runs, and its numbers are not results. Local attempt stopped 2026-10-03. It
      never reached f38's own code: it spent 15 minutes building caco2's 3D augmentation
      representations, which are not cached on this machine, and grover is missing here.
      Do it on the server in srun instead.
- [ ] A5. Check whether an outlier_* condition corrupts only a subset of the training labels, or
      moves every label. If it moves every label, detection is undefined there.
- [ ] A6. `f38_analyse.py`: per-fold scores and the spread across folds, for each axis
      (distance, weighting and power, feature space, k). No significance tests.

## B. Cost, before anything big runs

- [ ] B1. Time one screen task on the server, in srun. Report the hours per task and the
      number of tasks.
- [ ] B2. Time one n42 unit with every setting, on the server, in srun.
- [ ] B3. Bring both costs to the owner. She decides the datasets and how wide the n42 check is.

## C. Full screen on the server (srun)

- [ ] C1. `--list-tasks`, cheapest first. Every noise dataset, every condition where detection
      is defined, 5 folds, 3 draws, 5 models, unless the owner changes it at B3.
- [ ] C2. Run the tasks. The run is resumable: rerunning skips finished rows.

## D. Does the screen agree with real cleaning?

- [ ] D1. Run n42 with every setting, on the datasets the owner picks at B3. Aim for at least 3.
- [ ] D2. For each dataset and condition, check whether the screen's top settings contain the
      setting whose cleaning gives n42's best held-out score. Report how often.
- [ ] D3. If they disagree, the screen cannot be used to cut anything. Report that and stop.

## E. What existing experiments already found

- [ ] E1. A table with one row per axis: which experiment measured it, which run, and what is
      still open.
- [ ] E2. f34 and f37 are still running (owner, 2026-10-03). Their rows wait until she says the
      runs are finished.

## F. Activity-cliff prediction: blocked on the owner

- [ ] F1. Owner picks (a) cliff partner among the k nearest neighbours, or (b) prediction
      error on cliff molecules.
- [ ] F2. Owner picks which pairs count as a cliff: ECFP4 Tanimoto ≥ 0.55, matched molecular
      pairs (d2, fxa and mpro only), or something else.
- [ ] F3. Owner picks the activity gap: 2.0 label units (n14) or 2 × the label's median
      absolute deviation (n01).
- [ ] F4. Build it once those three are decided.

## G. Cuts

- [ ] G1. Bring the owner the options per axis, with numbers from C and D. Each cut needs a
      stated reason: degenerate, broken or undefined. She decides every one.
