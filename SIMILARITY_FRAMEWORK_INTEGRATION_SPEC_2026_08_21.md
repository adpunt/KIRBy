# Similarity-Objective Framework Integration — Build Spec (Path B)

**Date:** 2026-08-21 · **Branch:** `similarity-metrics-study` · **Status:** SPEC ONLY — no code until approved.

## Goal (one paragraph)

Turn the similarity-objective "winning recipe" from a hand-run script in a private
corner (`tests/experiments/similarity_metrics/`, 3 datasets, own data loader) into a
**first-class molecular representation** the framework computes for **any** molecular
dataset. The "similarity hybrid" is the **top 3–5 recipes** (ranked, not a single one).
Every experiment that uses a similarity metric — the similarity experiments **and** the
noise experiments — carries **two arms**: (a) the regular prediction-tuned hybrid used as
a similarity space, and (b) the top 3–5 structure-tuned similarity recipes. Old
placeholder *similarity* pieces are replaced. Everything is precomputed once and cached so
it never recomputes. (The plain binary fingerprints the rest of the pipeline depends on
are NOT touched.)

## Decisions locked by the user (2026-08-21)

- Path B (framework-native), not Path A (tidy the side study).
- **All** molecular datasets.
- Similarity hybrid = **top 3–5 recipes**, ranked (revisit exact N: 3, 4, or 5).
- Every similarity-using experiment uses **both** the regular hybrid AND the top-N similarity recipes.
- "Replace the old placeholder" = the **old similarity reps/experiments only** — leave binary fingerprints alone.
- Cache is **already wiped**; full re-run is expected.
- Build order: **nothing first**; the cache-key fix is **second**.

---

## What already exists vs what's missing (from the code maps)

**Reusable, already in the main framework:**
- 3D pharmacophore rep: `create_pharmacophore3d` (49-D) — `src/kirby/representations/molecular.py:579`; conformer gen `_generate_conformer:67`. (Silently returns zeros on conformer failure `:602-605` — needs a coverage guard.)
- `objective='similarity'` / `'similarity_sac'` search engine — `src/kirby/find_best_hybrid.py:1102-1106,1116-1191,3033-3039,3327-3333,5143-5149`. Read-across needs labels+groups only (NO MCS). Working caller example: `tests/experiments/similarity_optimized_hybrid.py:_search:187`.
- The engine can CONSUME structural pairs as a secondary metric: `nested_cv_evaluate(..., structural_pairs=...)` `find_best_hybrid.py:6136` (param `:6206`, align `:6251`), scored by `_structural_fidelity_score:1194`.

**Missing — only in the standalone study, must be ported to `src/kirby` + registry:**
- Count fingerprints (framework has BINARY only): `create_ecfp4/:295`, `create_atom_pair:392`, `create_top_torsion:421`. Recipe needs hashed COUNT versions (see `similarity_metrics/build_moleculeace.py:55-58`).
- The full raw 208-descriptor block (recipe's `desc`). `create_pdv:836` is close but a different ~200 list + pre-z-scored — not drop-in.
- Molecule-to-molecule structural-overlap (MCS) computation + cache. **No MCS anywhere in `src/kirby/`** — only `similarity_metrics/mcs_cache.py` + `structural_yardstick.mcs_distance`. No framework structural cache exists.
- The two label-blind MCS-based selections: saboteur-drop `creative_hybrids_v2.py:desc_blacklist_greedy:123-161`, fine2D `select_fine2d:114-119`. These are fit-on-data → must run through the leakage-safe per-fold factory path (`find_best_hybrid_molecular.py:FACTORY_REPS:168`, `get_rep_factories:653`), which has no entry for them today.
- The block-stacking algebra: `make_D` z-score + `equal_block` (`/sqrt(d_block)`) + coarse-down weighting `creative_hybrids_v2.py:82-93`.
- `objective` is **NOT** in the results-cache key: `result_db_identity find_best_hybrid.py:5298-5306`; absent from `result_db_config_const:5319-5327`, `_make_search_meta:338-530`, `_result_to_db_row:3496-3535`. Predictive and similarity runs collide.

---

## Build plan (file-by-file, in dependency order)

### Phase 0 — Core representation building blocks
- **0.1 Count fingerprints.** Add `create_ecfp4_count`, `create_atompair_count`, `create_torsion_count` to `src/kirby/representations/molecular.py` (hashed count Morgan / atom-pair / torsion; model on `build_moleculeace.py:55-58`). Register in `tests/find_best_hybrid_molecular.py` `rep_funcs:568-605` + imports `:542-555`; add to `REP_PRESETS['all']:145-164` if they should be pool-eligible.
- **0.2 Full descriptor block.** Add `create_desc208` (raw `Descriptors._descList`, unscaled) — distinct from `create_pdv`. Register as above.
- **0.3 3D block.** Reuse `create_pharmacophore3d`; add a per-dataset coverage guard mirroring `load_pharm3d_full`'s `n_built==n` check (`creative_hybrids_v2.py:97-110`).

### Phase 1 — Structural-overlap (MCS) engine + framework cache  ← deep/novel
- **1.1** New `src/kirby/similarity/mcs.py`: `compute_mcs_pairs(smiles, n_pairs=8000, seed)` → `{I, J, mcs}`. Port from `similarity_metrics/mcs_cache.py:35-59` + `structural_yardstick.mcs_distance`.
- **1.2** A per-dataset **structural cache** keyed by `(dataset_name, smiles_hash)` under `KIRBY_CACHE_DIR`. Compute-once/read-many. This is the new cache the framework lacks. Must expose a train-row-restricted view for Phase 2.
- **1.3** A precompute entry point so the roster can build MCS caches per dataset.

### Phase 2 — Leakage-safe per-fold selection factories  ← deep/novel
- **2.1** Saboteur-drop factory (from `desc_blacklist_greedy:123-161`): fits on **train rows' MCS only**, returns kept descriptor columns, applied to val without refit.
- **2.2** fine2D factory (from `select_fine2d:114-119`): train-MCS only, top-100 of `[atompair_count | torsion]`.
- **2.3** Both registered through `FACTORY_REPS:168` / `get_rep_factories:653` so they never see the test fold.

### Phase 3 — Top 3–5 similarity recipes (generator + ranking + reps)
- **3.1** Recipe generator: build candidate stacked spaces (count / +physchem_bl / +shape / +fine2D × weightings) via the `make_D` "stack + z-score + equal_block + coarse-down" algebra (`creative_hybrids_v2.py:82-93`), in **feature-matrix** form (pre-distance stack).
- **3.2** Rank candidates by **structural fidelity** (label-blind, MCS) — `struct_fidelity:165`. Take top **N (3–5)**.
- **3.3** Register the top-N as similarity reps (`sim_recipe_1..N`), computed via the Phase-0/2 blocks, added to `DATASET_REPS` for all molecular datasets (`tests/experiments/complexity/complexity_helpers.py:94+`).
- **3.4** This REPLACES the old placeholder similarity reps/experiments.

### Phase 4 — `objective` in the results-cache key  ← CORE FILE, needs sign-off; **this is step 2 in build order**
- **4.1** Add `'objective'` to `result_db_identity:5298-5306`, `_result_to_db_row:3496-3535`, `_make_search_meta:338-530`. (Model on how `label_noise` rides in `tier` via `tier_name:5191`, but make `objective` a first-class key field.)
- **4.2** Confirm `find_best_hybrid(objective='similarity')` caches and reads back distinctly from predictive rows.

### Phase 5 — The "both arms" convention + delete the hacks
- **5.1** Shared helper (in `tests/experiments/_shared.py` or new module) that, per dataset, yields BOTH: (a) regular predictive hybrid blueprint (`objective='predictive'` → `apply_blueprint`), (b) the top 3–5 similarity recipes/hybrids. Both cached (Phase 4 makes this safe).
- **5.2** Noise experiments: **delete** `tests/experiments/noise/_similarity_hybrid_space.py`; remove the `hybrid_sim` cross-dir hack and env vars from n11 (`:54,226-233,419-422,443-448,513-516`) and n13 (`:58,392-397,650-655,719,726-727,763`). Consume both arms from `data[]` / the rep pipeline instead. Env vars → **config keys** set in `noise_helpers.py:run_all:142-163`, threaded from a `run_dataset.py` CLI flag (precedent: `--pruning-mode:80-86,118-119`).
- **5.3** Similarity experiments (s01–s07 in `tests/experiments/similarity/`): same both-arms wiring.

### Phase 6 — Characterization experiment, properly homed
- **6.1** The noise-cleaning sweep (currently standalone `similarity_metrics/similarity_noise_cleaning.py`) becomes a **registered** experiment on framework data + the new reps + the `simeval`-style σ×rate×config noise battery (`simeval.py:351-369`). Register in `noise_helpers.py` (`SINGLE_DATASET_EXPERIMENTS:43`, `_import_experiment:91`, `_experiment_full_name:110`).
- **6.2** Retire the standalone script (or mark it as research origin; the framework version is live).

### Phase 7 — Precompute + roster for all molecular datasets
- **7.1** Roster the MCS + 3D + selection precompute per molecular dataset (`tests/slurm_scripts/submit_all_datasets.sh` molecular list `:34`; runs via `submit_dataset.sh:101-106` → `run_dataset.py`).
- **7.2** Full re-run (cache already wiped).

---

## Risks / sign-off gates
- **Core-file edits:** `find_best_hybrid.py` cache key (Phase 4); rep registry + `DATASET_REPS` (Phases 0, 3). Per project rule, no gate/search-space/core changes without user sign-off.
- **Deep novel parts:** MCS engine + structural cache (Phase 1); leakage-safe per-fold selection (Phase 2). Highest bug risk.
- **Molecular-only:** `data['smiles']` exists only on the molecular load branch (`_shared.py:2902`), so similarity reps are inherently molecular-only.
- **Locked rule:** reps are never in the cache key → adding reps = wipe + rerun (already done).

## Open micro-decisions
1. Exactly **N = 3, 4, or 5** top recipes.
2. Confirm the precise list of **old similarity pieces to retire** (reps + any s0x experiments that were placeholders).
3. Characterization experiment: register under **noise**, **similarity**, or **both**.
