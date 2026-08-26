# SUPERSEDED — the redundancy penalty was built on 2026-08-26

This was a handoff prompt. It is no longer needed: the work it describes was done in a
parallel chat on the same day, while this prompt was being written.

What exists now:

- `pairdist_redundancy`, a new weighting scheme in `src/kirby/hybrid.py`, registered as
  `pairdist_0.5/1.0/2.0` and `pairdist_score_0.5/1.0/2.0`. Redundancy between two
  representations is the Spearman correlation between their z-scored pairwise Euclidean
  distance vectors over a shared set of sampled row pairs, drawn by the same
  `_sample_offdiag_pairs` the geometry scorer uses.
- The existing `entropy` scheme is untouched, so every f01 and f12 number still
  reproduces. `tests/test_pairdist_redundancy.py` asserts it — 6 tests, all passing.
- `tests/experiments/framework/f21_redundancy_penalty.py` and
  `tests/slurm_scripts/run_f21_redundancy_penalty.sh` run 11 arms against 3 objectives
  across the full budget list through `nested_cv_evaluate`.

The measured part so far, redundancy only, no allocation outcome yet: on herg_ki with 8
representations the two ways of measuring redundancy disagree, mean absolute difference
0.319, rank agreement rho −0.153. Per-sample row norm correlates with heavy-atom count at
rho 0.851 on ecfp4 and 0.882 on mordred. logd reproduces the disagreement at 0.303 and
rho −0.270.

One item is still open and is Adelaide's call: `entropy` multiplies a score that
`_sanitize_baseline_scores` has already shifted, so under a geometry objective its worst
representation sits at exactly 0 and the penalty cannot move it. Left in place on purpose
as the historical arm, because fixing it there would change what f01 and f12 measured.

Current state lives in `SIMILARITY_METRICS_ROADMAP.md`, section B.
