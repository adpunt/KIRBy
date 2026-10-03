# f38: what the owner has signed off

Each entry was stated or confirmed by the owner in the chat of 2026-10-03. Nothing here is
Claude's proposal. `F38_STEPS.md` is Claude's working plan and is NOT signed off.

## The goal

1. Cut the kNN search space across the noise experiments (n01–n42) and the similarity
   experiments (s01–s13).
2. Build the kNN parts of those experiments the way the framework results say.

## Which framework experiments count

3. Only f34 onward is relevant: f34 (every file), f35, f36 and f37. f00–f33 are not relevant to
   this work.
4. The framework experiments' local and quick results ARE the evidence for these decisions.
   They are not to be dismissed as debug runs. That rule applies to the noise and similarity
   suites' own preliminary runs, not to the framework experiments.

## How the work is done

5. Read every noise experiment and every similarity experiment in full, along with the f34–f37
   files and their results.
6. Where f34–f37 did not measure a kNN choice those experiments make, add tests to fill the gap.
   That is what f38 is for.
7. Hold off until the framework results are in. As of 2026-10-03:
   - f34_importance_sweep is still running
   - f37 is still running
   - f34_full_sweep is finished, on the server
   Get the lay of the land first; build nothing until the results the work depends on exist.

## How cuts are made

8. One kept list per axis, used for every dataset. No cuts per dataset. Each dataset's numbers
   are shown side by side and never averaged.
9. No cutoff is fixed in advance. That covers both:
   - how many neighbours two settings must share to count as the same
   - how often the screen must agree with real cleaning
   The owner sees the full results, then sets them.

## How things run

10. Every server run happens inside one `srun ... --pty /bin/bash` session, with plain commands
    typed inside it. Never sbatch.
11. No dynamic workflows (multi-agent runs) without the owner's explicit go.

## Not decided yet

- Activity-cliff prediction: (a) finding the cliff partner among the k nearest neighbours, or
  (b) the prediction error on cliff molecules; which pairs count as a cliff; and the activity
  gap.
- Which datasets f38's tests run on.
- The design of `f38_neighbour_screen.py`. It was written before the lay of the land and is not
  signed off.
