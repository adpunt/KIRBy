# §V LOCAL SMOKE GATE — Shared Fleet Brief (2026-06-10)

You are one agent in a fleet running the **§V local smoke-test gate** (REPRODUCIBILITY_LEDGER.md
§V). This is the LAST gate before the cache redesign is committed and resubmitted to the Oxford
ARC server. The user cannot afford another faulty full server run.

## What is NEW since the last smoke fleet (2026-06-05)
The whole search-result cache was REDESIGNED into a per-dataset **SQLite store**
(`src/kirby/result_cache.py`), wired into `find_best_hybrid.py`. The OLD `searches/*.pkl` bundle is
gone. §V part 2 (cache reuse) is THE headline test now — it is the test that would have caught the
months-long stall. Every family agent MUST prove the new cache is exercised and reused.

## The §V gate has three parts (your report must address all three for your family)
1. **Determinism** — fold/group functions reproducible across fresh processes + on a row-shuffled
   copy (same partition AND same `groups_hash`, integer groups). (Cache+Determinism agent owns the
   dedicated tests; family agents just must not see `meta drift` from nondeterminism.)
2. **Cache REUSED** — run the SAME thing twice → second run is a HIT, NO `meta drift` lines, and the
   engine prints `ResultCache: N stored configs reused`. Changing the MODEL list must reuse the old
   rows and compute ONLY the new model (the core bug). Each agent must capture evidence of this for
   at least one run in its lane.
3. **Experiments run clean end-to-end** — one small dataset per family, **clf + reg** where
   supported, no unexpected `{'error'}`, regression-only experiments SKIP cleanly on clf data,
   large-n guards do not crash, and the wave-audit would count them complete.

## Environment (macOS, local)
- Python: `/Users/apunt/anaconda3/bin/python` (3.11.7). Run from repo root `/Users/apunt/repos/KIRBy`.
- ALWAYS set `PYTHONPATH=src` (or rely on existing sys.path inserts in the experiment files).
- ALWAYS prefix runs with single-thread env (prevents a macOS OpenMP segfault, exit 139):
  ```
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  ```
- No `timeout`/`gtimeout`. Use perl as the wall-clock wrapper:
  ```
  perl -e 'alarm 900; exec @ARGV' /Users/apunt/anaconda3/bin/python <script> <args>
  ```
  (exit 142 = alarm fired = timed out — "ran N iterations clean before timeout" is a PASS for a
  grid-repetition loop; cite the loop file:line.)

## ★ CACHE ISOLATION — MANDATORY (so agents don't clobber each other or the real cache)
Each agent runs in its OWN cache dir. Put this in every run:
```
KIRBY_CACHE_DIR=/tmp/kirby_V_<FAMILY>
```
Use a unique `<FAMILY>` per agent (e.g. `/tmp/kirby_V_complexity`). The SQLite result DBs live at
`$KIRBY_CACHE_DIR/result_db/<dataset>.db`. For a clean reuse test, START from an empty dir, run
once (cold = computes), run again (warm = reuses). Do NOT delete the real `~/.cache/kirby`.

## The smoke knobs (already built in — read the code to confirm)
- `KIRBY_SMOKE_N=200` caps dataset rows (via `_shared._smoke_subsample`). In federated it also
  shrinks folds/silos/grids; in opig it enables `_smoke_search_kwargs` (tiny search).
- `KIRBY_SMOKE_N` does NOT shrink the find_best_hybrid grid. `precompute_hybrid_searches` ALWAYS
  uses `_shared.FULL_BUDGETS` and ignores `config['budgets']`. To make a run tractable, also inject
  a tiny config and shrink FULL_BUDGETS. Proven recipe (`tests/experiments/_run_complexity_locally.py`):
  ```python
  import experiments._shared as _shared
  _shared.FULL_BUDGETS = [25]
  config = {'n_splits': 2, 'search_speed': 'full',
            'allocations': SPEED_TIERS['full']['allocation_methods'][:2],
            'importance_methods': SPEED_TIERS['full']['importance_methods'][:1],
            'internal_evaluators': ['rf'], 'budgets': [25], 'models': ['rf']}
  ```
  `'full'` is the ONLY valid SPEED_TIERS key.

## Local data: ONLY locally-computable reps (NO foundation models, NO server)
- Foundation-model reps (esm*, antiberty, esm3, saprot, geneformer, molformer, mole, ...) WILL fail
  locally — EXPECTED, not a bug. Pick local reps.
- Known-good LOCAL reps: protein/antibody = `aac`, `blosum`, `onehot`; molecular/reaction =
  `diff_maccs`, `onehot_components`, `ecfp`, `maccs`, `reactant_pdv`.
- Known-good SMALL datasets (verified 06-05):
  - reg antibody: `ginkgo_heparin` (n=94), `flab_thermo_jain`
  - clf antibody: `humanness_therapeutic` (n=564)
  - reg reaction: `buchwald_hartwig`
  - clf molecular: `herg` (n=7772 — subsample with KIRBY_SMOKE_N)
  Confirm task_type via `data['task_type']`. Pick ONE small clf + ONE small reg for your family.

## Lane discipline (hard rule)
- EDIT only files inside YOUR assigned experiment directory. Do NOT edit
  `tests/experiments/_shared.py`, `src/kirby/**`, `registry.py`, `find_best_hybrid.py`,
  `result_cache.py`, or another family's directory.
- If the bug is in SHARED code (esp. the cache / engine), DO NOT fix it. Record it PRECISELY in your
  report (file:line, the error, a concrete proposed fix) for the orchestrator to escalate to the
  user. The cache redesign is uncommitted and under sign-off — a real cache bug is a STOP-the-commit
  finding, exactly what this gate exists to catch.
- You MAY add `KIRBY_SMOKE_N`-gated shrink logic to YOUR OWN experiment files if needed, gated so
  full server runs are unaffected. Prefer existing knobs.
- Do NOT touch framework experiments (tests/experiments/framework/). Do NOT commit anything.

## Deliverable — write `tests/results/_smoke_V_2026_06_10/<FAMILY>_REPORT.md`
1. **SMOKE RESULTS** — one row per experiment × {clf, reg}: exact command, pass/fail/timed-out-clean,
   what was fixed (file:line), notes.
2. **CACHE EVIDENCE** — for at least one run: the cold-run line (computed N), the warm-run line
   (`ResultCache: N stored configs reused`), confirmation of NO `meta drift`, and (if you tested it)
   the changed-model-list reuse. Paste the actual stdout lines.
3. **DECISIONS / SHARED-CODE FINDINGS** — anything needing the user's judgment or any shared-code/cache
   bug you could not fix, with file:line evidence and a proposed fix.

Return a concise final summary: per-experiment pass/fail, in-lane fixes made, shared-code/cache
issues found, and the gate verdict for your family (PASS / PASS-with-caveats / FAIL).
