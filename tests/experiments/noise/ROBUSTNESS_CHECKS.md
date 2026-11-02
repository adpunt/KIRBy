# Robustness checks for noise-method results (author decision, 2026-10-26)

The main result for every noise method is whether it raised the top score: the best
held-out score with the method beats the best held-out score without it, across all
models and representations, on the same damaged training labels.

Two checks support that result. They are tests, not the result. Too few have been run
to draw wide conclusions from them, so they must not be used to sort or rank methods.

## Check 1: random comparison

The same action applied to molecules chosen at random, with the same count.

- n24: `delete_at_random` deletes the same number of training molecules as Delete
  flagged, 5 seeds (`RANDOM_DELETION_SEEDS`), each refit on the same models and
  representations and reported on its own.
- n31: the same number of molecules flagged, drawn only from molecules whose labels
  were not damaged (`flags_share<s>_p0/seed<k>`).

Question it answers: did choosing these molecules matter, or would any molecules do?

## Check 2: QM9, clean and with added noise

QM9 labels are computed, not measured, so clean QM9 has no lab error.

- Clean QM9: shows whether the method changes the score when there is no noise.
- QM9 with added noise: shows the method on a known clean baseline.

Currently one target: HOMO-LUMO gap (`src/kirby/datasets/qm9.py`).
To do: add a second QM9 target. Which one is not yet chosen (the loader holds
mu, alpha, homo, lumo, gap, r2, ...).

## Where these have been run (as of 2026-10-26)

Random comparison: n24 (Delete flagged, Score down-weight and other n24 actions),
n31 (LightGBM-residual flags), n32 and n38 (Small-loss selection, co-teaching),
n30 (Own lift), n18 (shuffled scores, labels as delivered). Most methods have none.
