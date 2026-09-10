# Fix 1 — the five files that cannot run

Copy below the line into a fresh chat.

---

Five files in `tests/experiments/complexity/` die at import. Make them run.

`c20_signal_saturation.py`, `c30_balanced_global_selection.py`, `exp_b_balanced_global.py`,
`c43_buchwald_signal.py`, `c43_buchwald_step3.py`.

All five do `import round2_harness as H` from a hardcoded `/private/tmp` path. That directory
is gone and `round2_harness` was never committed to git.

`tests/experiments/_alloc_harness.py` is the replacement. Read it first. It defines `_clean`,
`load_full`, `make_folds`, `rank_columns`, `score_matrix` and `assert_metric_direction`, and it
re-exports `get_model` from `kirby.find_best_hybrid` at line 58.
`c31_perf_weighted_global_selection.py:87` already imports it as
`from experiments import _alloc_harness as H` and runs. Copy that pattern, including the path
bootstrap above it.

Then, per file:

- `c43_buchwald_signal.py` — also writes nothing. It prints its result and exits, and
  `c43_buchwald_step3.py:131` hardcodes that printed number as a literal. Give it an output path
  and write the payload it already builds.
- `c43_buchwald_step3.py` — its checkpoint path points into the same dead directory. Give it a
  real one.
- `c30_balanced_global_selection.py` — writes `expB_{ds}.json` into the dead directory. Give it
  a real one. Note `exp_b_balanced_global.py:321` writes that same filename, so if both are
  repaired to one directory they overwrite each other.
- `c20_signal_saturation.py` — reads `H.STAGED_DATASETS` when `--dataset` is omitted.
  `_alloc_harness` does not define it. Either make `--dataset` required, or add the name.
- `exp_b_balanced_global.py` — calls `H._label_signal_ladder`, which exists nowhere in the
  repo. Do not invent it. Report what it would have to compute and stop; this file may be
  deleted instead.

Rules:

- Read the code before changing it. Quote `file:line`.
- Do not change what any experiment measures. This is an import and output-path repair only.
- Do not add a fallback that silently skips a missing name. Raise.
- After each file, run it once locally with `KIRBY_SMOKE_N=60` and a pinned model list
  (`--models rf`), and report the exit code and the first ten lines of output. The default
  ten-model list segfaults on this machine.
- Report anything you find that is wrong beyond the import. Do not fix it.
