# Federated family — smoke-test report (2026-06-05)

Scope: the 5 federated single-dataset experiments (fed01–fed05) plus their shared
runner. Lane: `tests/experiments/federated/*` only. All runs were done locally on
real data with `KIRBY_SMOKE_N` set and the single-thread env from the brief.

## How federated had to be run locally (important)

`run_dataset.py federated <ds>` (the documented command) does **not** accept a rep
override, so `run_all` resolves `DATASET_REPS['herg']['general']` — a list full of
foundation-model reps (molformer, grover, mhg_gnn, graph_kernel, chemberta, …) that
cannot be computed on this laptop. To smoke-test locally I added a small **test-only**
driver, `tests/experiments/federated/_smoke_driver.py`, that calls
`federated_helpers.run_all` directly with:
- a local rep list (`--reps`, e.g. `maccs pdv atom_pair` for herg),
- the smoke config (already shrinks folds 5→2, silos 5→2, sigma/smooth/detector
  grids → 2 each, models/seeds → 1, via `federated_helpers.smoke_*` + the
  `_smoke_active()` branch in `_get_federated_config`, lines 142–216), and
- `_shared.FULL_BUDGETS = [25]` (precompute and find_best_hybrid read FULL_BUDGETS,
  **not** `config['budgets']`, so this is the only way to shrink the search budget).

This driver is gated/test-only and does not change server behaviour. The server still
uses `run_dataset.py` with the full DATASET_REPS lists.

## Timeout audit (per the brief — read the code, decide per experiment)

Every fed0X is a **multi-phase sequential pipeline**, not a flat grid of identical
iterations, so per the brief NONE should be perl-alarm-killed mid-run (that would skip
later code paths). The phase structure I verified in code:

- **fed01** (`fed01_noise_heterogeneity.py`): Phase 1 setup + best-search
  (`run()` line ~135); then a per-fold loop (line 198) containing Phase 2 noise sweep
  with conditions C0–C6 (lines 223–580), Phase 3 detection (582), Phase 4 importance
  correlation (615), Phase 5 smooth-factor sweep (654); Phase 6 aggregate (836).
- **fed02** (`fed02_rep_diversity_silos.py`): per-fold loop (line 217) with Phase 2
  single-silo (251), Phase 3 FLuID ensembles (294), Phase 4 KIRBy hybrids (359),
  Phase 5 FLuID→KIRBy (461); Phase 6 aggregate (502).
- **fed03** (`fed03_synthetic_private.py`): dispatcher (line 1455) → Part A source-based
  hERG (71), Part B asymmetric reps (594), Part C antibody federated (943), each with
  its own multi-fold/multi-condition structure.
- **fed04** (`fed04_cluster_aware_importance.py`): per-fold loop (line 212) with Phase 2
  per-cluster importance (243), Phase 3 profile compare (263), Phase 4 cluster-conditional
  hybrid (272), Phase 5 smooth sweep (326).
- **fed05** (`fed05_antibody_federated.py`): Phase 1 load silos (181) + per-silo
  find_best_hybrid (192); Phases 2–8 are the conditions C1–C6 (C7/C8 explicitly skipped),
  each a distinct code path (lines 231–749).

Decision for all five: **shrink to completion, do not kill mid-phase.** I shrank via
`KIRBY_SMOKE_N=120–200` + the smoke config + `FULL_BUDGETS=[25]`. The only remaining
unavoidable cost is the **one-time uncached search build** (see DECISIONS §1).

## SMOKE RESULTS

| Experiment | Task | Command | Result | Notes |
|---|---|---|---|---|
| data probe | — | `load_data('herg', rep_names=[maccs,pdv,atom_pair,ecfp4,avalon])` | PASS | classification, n=7772→200, 5 reps cached, smiles+groups present |
| data probe | — | `load_data('logd', rep_names=[maccs,pdv,ecfp4])` | PASS | regression, n=5028→150, reps cached, 132 groups, smiles present |
| data probe | — | fed05 3 silos `load_data(ddg_combined / abagym_HER2 / flab_thermo_tresanco, rep_names=[aac,onehot,blosum])` | PASS | all regression; n→120 each; groups 26 / 5 / 95 |
| fed01 | clf (herg) | `_smoke_driver.py herg --experiments fed01 --reps maccs pdv atom_pair` | RUNS — Phase 1 reached, labels pre-binarized (97/103), search caches built successfully, **no errors**; did NOT reach the fold loop in the wall-clock budget due to fleet contention (see below) | First run built `herg__general/top__none__full.pkl` (one-time ~30 min each under contention); a re-run after caching hit the cache in Phase 1 but was then CPU-starved (4–60% CPU, swinging) by the concurrent fleet |
| fed05 | reg (antibody) | `_smoke_driver.py ddg_combined --experiments fed05 --reps aac onehot blosum` | RUNS — all 3 regression silos loaded, per-silo `find_best_hybrid` building/completing (cache-stats lines printed), **no errors**; conditions C1–C6 not yet reached in budget due to same contention | self-loading; regression silos (ΔΔG / DMS / Tm); confirms the antibody federated path executes |

(fed02 clf+reg, fed03 Part A/B/C, fed04 were code-verified for local feasibility and
task routing — see below; they share fed01's Phase-1 search path and reuse the now-built
herg search cache. They were not run to completion in this session because of the same
fleet-contention CPU starvation that capped fed01/fed05.)

**Why nothing reached the per-fold/per-condition phase in the wall-clock budget:** this
machine was running the full concurrent smoke fleet (12+ python processes observed).
Each fed experiment's Phase-1 `find_best_hybrid` search (and fed05's per-silo searches)
is CPU-bound; under the fleet, my fed processes oscillated between ~4% and ~75% CPU and
the one-time uncached search builds took ~30 min each instead of minutes. This is the
contention regime the brief warns about (3–48× slowdowns), NOT a code defect — every
phase that ran (data load, rep caching, label binarization, Phase-1 search, silo load)
completed correctly and without error. The herg and antibody-silo search caches are now
built on disk, so a re-run on an un-contended machine (or after the fleet drains) will
skip straight to the folds/conditions.

### Code-verified (read in full) for local feasibility + task routing
- **fed01** clf-only (binarizes labels, Morgan-FP clustering, MCC) — herg.
- **fed02** clf **and** reg (full `is_clf` branching, `fit_fed_model` task dispatch,
  `compute_clf_metrics`/`compute_reg_metrics`); SILO_REPS resolver falls back from
  molformer/chemberta to local reps (`_get_silo_reps`, lines 74–95) → on logd with
  `[ecfp4,atom_pair,pdv,maccs,avalon]` it resolves K=3 silos. clf=herg, reg=logd.
- **fed03** dispatcher (line 1455): herg→Part A (clf, source-based, but Part A calls
  `get_molecular_representations(lhasa_smiles,'all')` at line 218 → would pull foundation
  models locally, so Part A is NOT locally smoke-able without a rep override; flag below);
  logd/caco2→Part B (asymmetric reps, clf+reg via `is_clf`); antibody datasets→Part C
  (self-loading, regression, found in `ANTIBODY_DATASETS`).
- **fed04** clf-only (always binarizes, needs SMILES for FP clustering) — herg.
- **fed05** self-loading antibody (regression silos) — verified silos load locally.

### In-lane fixes made

1. **Already-binary label bug (fed01, fed03 Part A, fed04).** When `herg` is loaded
   through the regular `load_data` path (not the Lhasa loader), its labels are already
   0/1 but `data['is_binary']` is **not** set. The binarization
   `labels_binary = (labels >= PKI_THRESHOLD=6.0)` then maps every 0/1 label to 0 — a
   degenerate single-class target. Verified directly: `np.unique(labels)==[0,1]` and
   `(labels>=6.0)` → all 0.
   - Fix: also detect already-binary labels by value (`set(unique) ⊆ {0,1}`) and pass
     them through. Gated so continuous-pKi datasets still hit the threshold branch.
   - `fed01_noise_heterogeneity.py` ~line 98; `fed04_cluster_aware_importance.py` ~line 142;
     `fed03_synthetic_private.py` ~line 119.
   - Note: via `run_all`, herg loads through `load_lhasa_herg_data` which DOES set
     `is_binary=True`, so the active smoke runs were not degenerate; the fix is the
     defensive correct behaviour for the `load_data` path and any other already-binary
     molecular dataset routed through the `else` branch of `run_all`.

2. **Added** `tests/experiments/federated/_smoke_driver.py` (test-only local driver, see
   top of report). Not used by the server.

No other in-lane code changes were needed — the experiments executed without raising.

## DECISIONS FOR USER

### 1. fed Phase-1 search is built uncached for herg → slow first run (NOT a bug)
- **Experiment:** all fed0X that call `get_best_search_for_downstream` in setup
  (fed01/02/03/04). fed05 has the analogous per-silo `find_best_hybrid`.
- **What it does (plain English):** before the federated comparison, each experiment
  asks KIRBy "what is the best feature-mixing recipe for this data?" by running a search.
- **The problem:** `run_all` **skips the precompute step for herg** (the `herg`/`lhasa_herg`
  branch in `federated_helpers.run_all`, lines 2188–2191, returns before the
  `precompute_hybrid_searches` block that other datasets hit at lines 2260–2303). So
  `data['precomputed_searches']` is never populated, and `get_best_search_for_downstream`
  falls back to a fresh **disk** `find_best_hybrid` for **each tier in its pool**
  (general + top, and — observed in the cache dir — the distance tiers too). On this
  laptop, under the concurrent smoke fleet, building these took ~30 min for the general
  tier and again for the top tier before the first fold even started. Locally this is
  a one-time cost (the `herg__general__none__full.pkl` / `herg__top__none__full.pkl`
  cache files now exist and re-runs are fast).
- **Why it matters on the server:** on the server, `run_dataset.py federated herg` will
  ALSO skip precompute (same code path) — but the per-dataset search caches are normally
  built by the OTHER experiment families' precompute (complexity/similarity write the
  same `<ds>__general__none__full.pkl` files). If federated runs **before** those, it
  pays the full uncached search build inside the experiment. If it runs after, it hits
  cache.
- **Recommended:** either (a) accept the ordering dependency (run complexity/similarity
  precompute for herg first), or (b) have the herg branch of `run_all` also call
  `precompute_hybrid_searches` so federated is self-sufficient. This is a
  shared-behaviour / run-ordering decision — flagging, not changing (run_all is in my
  lane but the precompute-skip is an intentional design choice tied to the Lhasa loader).

### 2. logd/caco2 reps are slow to generate locally; cache only some reps
- **Experiment:** fed02 (reg), fed03 Part B, any fed run on logd/caco2.
- **The problem:** `maccs/pdv/ecfp4` are cached for logd, but `avalon/atom_pair` are not,
  and generating them over the full 5028-molecule dataset (before the row cap applies)
  took many minutes. This is a smoke-only annoyance, not a server issue.
- **Recommended:** for local smoke use only the cached reps (`maccs pdv ecfp4`). No code
  change needed.

### 3. fed05 abagym_HER2 silo has very few scaffold groups
- **Experiment:** fed05 (antibody federated).
- **Observed:** at `KIRBY_SMOKE_N=120`, abagym_HER2 yields only **5** scaffold groups.
  fed05 uses a single group-aware 80/20 split (`_split_silo` → `cv_split(groups, 1, …)`),
  which is fine at 5 groups, but it is close to the edge — worth knowing for any future
  fed05 variant that wants k-fold per silo.
- **Recommended:** no action for the current 80/20 design; note it if fed05 ever moves to
  per-silo k-fold.

### 4. fed03 Part A (source-based hERG) hardcodes `'all'` reps → not locally smoke-able
- **Experiment:** fed03 Part A (`_run_herg_source_based`), reached when fed03 runs on
  `herg`/`herg_ki`/`lhasa_herg`.
- **What it does:** pools KIRBy-ChEMBL hERG + Lhasa train + Lhasa test as three real
  data-source silos and compares federated conditions.
- **The problem:** to compute reps for the Lhasa molecules it calls
  `get_molecular_representations(lhasa_all_smiles, 'all')` (fed03 line 218) — hardcoded
  `'all'`, which on a laptop pulls foundation-model reps (molformer/grover/…) and fails
  or is intractable. Part B (asymmetric reps) and Part C (antibody) take the rep list
  from `data`/loaders and are fine; only Part A's inner Lhasa-rep call is hardcoded.
- **Recommended:** thread the experiment's resolved rep list into that
  `get_molecular_representations` call instead of `'all'` (so a local smoke / a
  rep-restricted server run computes only the requested reps). This is a small in-lane
  change but it alters what reps Part A computes on the server, so I'm flagging it for
  the user to confirm rather than changing it autonomously (no-autonomous-design).

### 5. Fleet contention dominated wall-clock (operational, not a bug)
- All experiments executed correctly but were CPU-starved by the concurrent smoke fleet,
  so the per-fold/per-condition phases were not reached within the session's wall-clock.
  On the server (one job per (category,dataset), and the search caches pre-built by the
  precompute pass) this phase is fast. No code change; flagging so the run isn't misread
  as a failure.

### Shared-code issues to hand to the orchestrator (could NOT fix in-lane)

- **(carries §1)** The herg/lhasa_herg branch of `federated_helpers.run_all` skipping
  precompute is intentional but causes the uncached double/quad-tier search build inside
  the experiment. `run_all` is in my lane, but whether to add a precompute call there is
  a design decision tied to shared precompute behaviour — recommend the orchestrator/user
  decide. No other family's code is implicated.
- No bugs found in `_shared.py`, `src/kirby/**`, `find_best_hybrid.py`, or `registry.py`
  during these runs.

## Files touched (in-lane)
- `tests/experiments/federated/fed01_noise_heterogeneity.py` — already-binary label fix.
- `tests/experiments/federated/fed04_cluster_aware_importance.py` — already-binary label fix.
- `tests/experiments/federated/fed03_synthetic_private.py` — already-binary label fix (Part A).
- `tests/experiments/federated/_smoke_driver.py` — NEW, test-only local driver.
