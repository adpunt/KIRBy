# Smoke-Test Fleet — Shared Agent Brief (2026-06-05)

You are one agent in a fleet smoke-testing the full KIRBy experiment suite **locally with
real data** before a clean re-run on the Oxford ARC server. The user cannot afford an
error-fix-resubmit loop on the cluster, so every experiment must be proven to run
end-to-end here first.

## What "smoke test" means
- Run each experiment end-to-end on a SMALL subset of REAL data (not synthetic).
- Goal = prove the pipeline / code path executes and produces output. NOT real results.
- Cover BOTH a **classification** and a **regression** dataset for the family, where
  the experiment supports both. Some experiments are single-task — note that.

## Environment (macOS, local)
- Python: `/Users/apunt/anaconda3/bin/python` (3.11.7). Run from repo root `/Users/apunt/repos/KIRBy`.
- ALWAYS prefix runs with single-thread env (prevents a macOS OpenMP double-load segfault,
  exit 139, inside find_best_hybrid):
  ```
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  ```
- There is NO `timeout`/`gtimeout` on this machine. Use perl as the wall-clock wrapper:
  ```
  perl -e 'alarm 600; exec @ARGV' /Users/apunt/anaconda3/bin/python <script> <args>
  ```
  (exit 142 = alarm fired = timed out.)

## The smoke knobs (already built into the codebase — read the code to confirm)
- `KIRBY_SMOKE_N=200` env var → caps dataset rows (via `_shared._smoke_subsample`).
  In **federated** it ALSO shrinks folds / silos / sweep grids (`federated_helpers.smoke_*`).
  In **opig** scripts it ALSO enables `_smoke_search_kwargs` (tiny search).
- IMPORTANT: `KIRBY_SMOKE_N` caps ROWS only for most families — it does NOT shrink the
  find_best_hybrid search grid (budgets/models/folds). `precompute_hybrid_searches`
  ALWAYS uses `_shared.FULL_BUDGETS` and ignores `config['budgets']`. So to make a run
  tractable you usually must ALSO inject a tiny config and shrink FULL_BUDGETS.
- The fast config recipe (proven in `tests/experiments/_run_complexity_locally.py` and
  `tests/experiments/noise/_smoke_harness.py`):
  ```python
  import experiments._shared as _shared
  _shared.FULL_BUDGETS = [25]          # precompute reads this, not config['budgets']
  config = {'n_splits': 2, 'search_speed': 'full',
            'allocations': SPEED_TIERS['full']['allocation_methods'][:2],
            'importance_methods': SPEED_TIERS['full']['importance_methods'][:1],
            'internal_evaluators': ['rf'], 'budgets': [25], 'models': ['rf']}
  ```
  `'full'` is the ONLY valid SPEED_TIERS key (legacy `'fast'` was removed; it now raises).

## Local data: use only locally-computable reps (NO foundation models, NO server)
- Foundation-model reps (esm*, antiberty, esm3, saprot, geneformer, molformer, mole, ...)
  WILL fail locally — that is EXPECTED, not a bug. Pick local reps instead.
- Known-good LOCAL reps: protein/antibody = `aac`, `blosum`, `onehot`;
  molecular/reaction = `diff_maccs`, `onehot_components`, `ecfp`, `maccs`, `reactant_pdv`.
- Known-good SMALL datasets that load locally (verified):
  - regression antibody: `ginkgo_heparin` (n=94), `flab_thermo_jain`
  - classification antibody: `humanness_therapeutic` (n=564)
  - regression reaction: `buchwald_hartwig`
  - classification molecular: `herg` (n=7772 — subsample with KIRBY_SMOKE_N)
  Confirm task_type via `data['task_type']`. Pick ONE small clf + ONE small reg for your family.

## The timeout audit (READ THE ACTUAL CODE — do NOT rely on memory or estimate docs)
The user wants timeouts used to cut CPU, but applied intelligently:
- READ each experiment's source: find the outer loops (grids over budgets / models / sigmas /
  detectors / folds / conditions). Decide, per experiment:
  - If the bulk of runtime is REPETITION of the SAME code path over a grid → it is safe to
    let the smoke run a few iterations then KILL via perl-alarm. "Ran N iterations clean
    before timeout" counts as a PASS for the pipeline. Cite the loop (file:line) in your report.
  - If the experiment has DISTINCT sequential phases (e.g. federated: setup → noise sweep →
    detection → importance → smooth sweep), do NOT kill mid-way — you'd miss later code paths.
    Instead shrink via KIRBY_SMOKE_N + tiny config + a smaller `--experiments`/grid so it
    COMPLETES and writes output.
- Record the timeout decision and its justification (the specific loop) for every experiment.

## Fix-and-retest (this is the core of the job)
- When a smoke run errors: read the traceback, find the bug, FIX IT, re-run. Iterate until
  it runs clean (or completes enough iterations before a justified timeout), or you hit a
  genuine blocker.
- **Lane discipline (hard rule):** you may EDIT only files inside YOUR assigned directory.
  Do NOT edit `tests/experiments/_shared.py`, anything under `src/kirby/`, `registry.py`,
  `find_best_hybrid.py`, or another family's directory. If the bug is in shared code, DO NOT
  fix it — record it precisely in your report (file:line, the error, a concrete proposed fix)
  for the orchestrator to apply and serialize.
- You MAY add KIRBY_SMOKE_N-gated shrink logic to YOUR OWN experiment files if that is the
  cleanest way to make a smoke run tractable — but it MUST be gated so server (full) runs are
  unaffected. Prefer using existing knobs over adding new ones.
- Do NOT touch framework experiments (tests/experiments/framework/, F00–F15).
- Do NOT commit anything. The user is the sign-off gate.

## Deliverable — write `tests/results/_smoke_2026_06_05/<FAMILY>_REPORT.md`
Two sections, plain English, low jargon, every claim backed by a real code reference:
1. **SMOKE RESULTS** — a table/list, one row per experiment × {clf, reg}:
   exact command run, pass / fail / timed-out-clean, what was fixed (file:line), notes.
2. **DECISIONS FOR USER** — anything needing the user's judgment. For EACH item:
   - which experiment it came from + one-line plain-English description of what that experiment does
   - what the problem is
   - recommended solution(s)
   - the code evidence (file:line) you based this on
   Also list every SHARED-CODE fix you could not make (for the orchestrator).

Return a concise final summary: per-experiment pass/fail, count of fixes made in-lane,
count of shared-code issues reported, and the top decisions for the user.
