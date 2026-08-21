# Handoff — Similarity-Objective Framework Integration (fresh-chat context)

**Date:** 2026-08-21 · **Branch:** `similarity-metrics-study` · **Read the companion spec:**
`SIMILARITY_FRAMEWORK_INTEGRATION_SPEC_2026_08_21.md` (the file-by-file build plan).

> This chat was a **set-up for a larger build**, not a place to produce results. The
> throwaway numbers below are only evidence the mechanism works; the deliverable is the
> spec + this handoff. Do NOT start coding — the next chat presents the detailed spec for
> a final check, then builds.

---

## 1. What this is, in one paragraph

KIRBy's thesis: improve the **input representation**, not the model. A separate
"similarity-metrics study" produced a **similarity-objective metric** — a structure-tuned
molecular distance space (the "winning recipe") that measures molecular likeness far better
than the standard count fingerprint. The goal here: make that metric a **first-class part
of the framework** and use it to **clean noisy activity labels** (the "noise-cleaning
loop"), so that every experiment using a similarity metric — including the noise
experiments — compares the regular prediction-tuned hybrid against the structure-tuned
similarity hybrid.

## 2. The winning recipe (what the similarity metric IS)

A stacked, z-scored, **equal-block Euclidean** space of four blocks (count block
down-weighted ×0.5 — the "honest" variant):
`[ecfp4_count(1024) + physchem_descriptors_minus_saboteurs + 3D_pharmacophore(49) + fine2D(top-100 atompair/torsion)]`.
- "**saboteurs**" = RDKit descriptors whose presence *lowers* structural agreement; dropped by a label-blind greedy search against MCS overlap.
- "**fine2D**" = top-100 atompair/torsion columns picked label-blind by correlation with MCS overlap.
- **MCS** = maximum-common-substructure overlap between two molecules — the label-blind structural ground truth.
- All selection is **label-blind** (uses structure, never activity) → using it to judge activity-noise is **not circular** (this is why the cut experiments n08/n10 died and this one doesn't).
Defined in `tests/experiments/similarity_metrics/creative_hybrids_v2.py` (`make_D:82`, `desc_blacklist_greedy:123`, `select_fine2d:114`, `load_pharm3d_full:97`).

## 3. Does it work? (evidence — throwaway, do not "write up")

Light local characterization (`similarity_metrics/similarity_noise_cleaning.py`, using
`simeval.noise_sweep_scores` over σ∈{0.3,0.6,0.9} × rate∈{0.10,0.20,0.30} × 3 seeds):
**the structure-selected metric beats the count-fingerprint baseline on BOTH noise
detection and recovery in all 27 cells (3 σ × 3 rate × 3 datasets: herg_ki, logd,
adme_solubility).** Detection gain grows with noise rate; recovery gain strongest at low
rate; biggest on logd. **Caveat:** the margin is vs the *count fingerprint*; vs an
already-standardized ecfp4 the edge shrinks to near-tie (the n11 result). So: works
universally vs the standard metric; margin depends on how good the pipeline's existing
metric already is.

## 4. Decisions locked by the user (2026-08-21)

- **Path B** (make it a first-class framework representation), NOT Path A (tidy the side study).
- **All** molecular datasets.
- The "similarity hybrid" = **top 3–5 recipes**, ranked by structural fidelity (exact N: 3/4/5 TBD).
- **Every** similarity-using experiment (similarity + noise) carries **both** arms: regular predictive hybrid AND the top-N similarity recipes.
- "Replace the old placeholder" = the **old similarity reps/experiments only**. Do NOT touch the binary fingerprints the rest of the pipeline uses.
- Cache is **already wiped**; a full re-run is expected/fine.
- Build order: nothing first; the **cache-key fix is second**.

## 5. Current state of the code (what's committed vs what's a hack to delete)

Committed on `similarity-metrics-study` this session:
- `7a98aa1` — noise-detector **bug fixes**: out-of-fold rewrite killing in-sample leakage in 8 estimators; MetaWeightNet (learning never ran) and MentorNet (collapsed to zero weights) fixed; scaffold-group plumbing. Verified. **Keep.**
- `a643c23` — the **`hybrid_sim` arm** wired into n11/n13 (the noise experiments). **This is a HACK to be replaced** (Phase 5 of the spec): it reaches out of the framework into the study's private caches and matches molecules by SMILES; works only for the 3 cached datasets.
- `5ceb0fb` — array-safe bugfix in the helper + n11 local-run env switches (`KIRBY_SIMILARITY_HYBRID_ONLY`, `KIRBY_NOISE_LIGHT_ESTIMATORS`). **The env vars become config keys; the helper gets deleted.**

Files to REPLACE/DELETE in the build:
- `tests/experiments/noise/_similarity_hybrid_space.py` (the cross-dir SMILES-join bridge).
- The `hybrid_sim` special-casing + env vars in `n11_estimation_remediation_grid.py` and `n13_noise_distributions.py`.
- The standalone recipe cluster in `tests/experiments/similarity_metrics/` is the **research origin**; the framework-native versions supersede it (the old placeholder similarity pieces retire).

## 6. The two framework homes (critical — don't confuse them)

- `tests/experiments/similarity/` (s01–s07) = the **real registered** similarity framework (same `run(data, config, ...)` convention as noise; runs from the roster).
- `tests/experiments/similarity_metrics/` = the **private standalone corner** (no `__init__.py`, not registered, run by hand, own `_data.load_prepared` row space, only herg_ki/logd/adme_solubility). All the recipe work lives here today.

## 7. How the frameworks actually work (seams the build plugs into)

- **Dispatch:** `tests/experiments/run_dataset.py:31-48` (`CATEGORIES` dict) → `noise_helpers.run_all` (`:124-377`) → each experiment's `run(data, config, output_dir, dataset_name)` (`:359`).
- **Reps:** `DATASET_REPS` in `tests/experiments/complexity/complexity_helpers.py:94+` → `load_data` (`_shared.py:2406`, molecular branch `:2881-2912`, `data['smiles']:2902`). Rep registry: `tests/find_best_hybrid_molecular.py:rep_funcs:568-605`, `REP_PRESETS:145`, `FACTORY_REPS:168` (leakage-safe per-fold reps), `get_rep_factories:653`. Base factories: `src/kirby/representations/molecular.py`.
- **Config:** production config built inline in `run_all:142-163` (NOT `get_config`); thread new flags from `run_dataset.py` (precedent `--pruning-mode:80-86,118-119`).
- **Submit:** `tests/slurm_scripts/submit_all_datasets.sh` (molecular list `:34`, noise prefix `nz:61`) → `submit_dataset.sh:101-106` → `run_dataset.py`.
- **Similarity engine:** `find_best_hybrid(objective='similarity'|'similarity_sac')` — `src/kirby/find_best_hybrid.py:1102-1191,3033-3039,5143-5149`. **`objective` is NOT in the results-cache key** (`result_db_identity:5298-5306`) → predictive/similarity collide → Phase 4 fixes this.

## 8. Hard constraints (do not violate)

- **Reps are NEVER in the cache key** (locked, user was angry) → new reps = wipe + rerun (already wiped).
- **No random splits** ever — scaffold-group CV only; per-fold selection must be train-only (leakage-safe).
- **No core changes** (find_best_hybrid cache key, rep registry, gates/search-space) **without user sign-off.**
- **Similarity is molecular-only** (`data['smiles']` only on the molecular branch).
- Plain-English only with this user; propose options, user decides; don't declare a thread "done" after experiments — decide + implement.
- Claude has **no server access** — user runs all ssh/scp/sbatch and pastes output.

## 9. Immediate next step for the fresh chat

Read `SIMILARITY_FRAMEWORK_INTEGRATION_SPEC_2026_08_21.md`, resolve the three open
micro-decisions (N=3/4/5; exact old-placeholder list; characterization home = noise /
similarity / both), then produce a detailed file-by-file implementation plan for the user
to approve **before writing any code**. Build order: everything else, then the cache-key
fix second.
