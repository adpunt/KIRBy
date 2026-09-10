# Similarity-Objective Framework Integration — Implementation Plan (Path B)

**Date:** 2026-08-21 · **Branch:** `similarity-metrics-study` · **Status:** PLAN — awaiting sign-off, no code yet.
**Companions:** `SIMILARITY_FRAMEWORK_INTEGRATION_SPEC_2026_08_21.md` (build spec), `..._HANDOFF_2026_08_21.md` (context).

All code anchors below were re-verified against the live tree on 2026-08-21 (function names confirmed; line
numbers approximate — the code shifts).

---

## Locked decisions (user, 2026-08-21)

1. **N = 5** — expose the top 5 ranked recipes as reps `sim_recipe_1..5`, ranked by label-blind MCS structural fidelity.
2. **Retire = hacks + standalone only.** Delete the n11/n13 `hybrid_sim` hack + env vars + `_similarity_hybrid_space.py`; supersede the private `tests/experiments/similarity_metrics/` corner. **Keep s01–s13 untouched in substance** — they are real, registered, honest-scored experiments, not placeholders. They gain one arm; every other similarity analysis they run (SAC, cliffs, kNN, applicability domain, retrieval, …) stays.
3. **Characterization experiment home = noise** (register in `noise_helpers` only).
4. **Three arms, side by side, each reported individually (NO averaging):**
   - (a) new structure-tuned **similarity hybrid** (top 5 recipes),
   - (b) regular prediction-tuned **hybrid** used as a similarity space,
   - (c) standard baseline metric — **ECFP4 / Tanimoto**.

**Audit correction to the spec:** there are **no dedicated similarity reps** anywhere — not in `DATASET_REPS`
(`complexity_helpers.py:94`), not in the rep registry (`find_best_hybrid_molecular.py:568 rep_funcs`), not in
`REP_PRESETS:145`. So spec Phase 3.4 "replaces old similarity reps" is a misnomer: nothing rep-shaped exists to
delete. What exists is s02's runtime construction of a similarity space from the **pool-best predictive hybrid** +
importance weighting — that IS arm (b), and it stays.

**Build order (locked):** everything else first; the `objective`-in-cache-key fix is **step 2**.

---

## Sign-off gates (core-file / gated edits — will not touch without explicit approval)

- **G1** `find_best_hybrid.py` results-cache identity — Phase 4.
- **G2** rep registry + `DATASET_REPS` additions — Phases 0, 3.
- **G3** leakage-safe per-fold selection factories — Phase 2 (highest bug risk).
- **G4** deleting the noise-side hacks + rewiring n11/n13 config — Phase 5.

Everything else (MCS engine, recipe generator, characterization experiment, roster) is additive/new-file.

---

## Phase-by-phase plan

### Phase A — Core representation building blocks  *(gate G2)*
New rep builders in `src/kirby/representations/molecular.py`, each registered in
`tests/find_best_hybrid_molecular.py` (`rep_funcs:568`, imports, and `REP_PRESETS['all']:145` if pool-eligible):

- **A.1** `create_ecfp4_count`, `create_atompair_count`, `create_torsion_count` — hashed **count** Morgan / atom-pair
  / torsion. Framework today has BINARY only (`create_ecfp4:295`, `create_atom_pair:392`, `create_top_torsion:421`).
  Model on `similarity_metrics/build_moleculeace.py:55-58`.
- **A.2** `create_desc208` — raw `Descriptors._descList`, **unscaled**. Distinct from `create_pdv:836` (different
  ~200 list, pre-z-scored) — not drop-in, hence a new builder.
- **A.3** Reuse `create_pharmacophore3d:579` (49-D). Add a **per-dataset conformer-coverage guard** mirroring
  `load_pharm3d_full`'s `n_built==n` check (`creative_hybrids_v2.py:97`), because it silently returns zeros on
  conformer failure (`molecular.py:602-605`).
- **A.4** `create_ecfp4_tanimoto` baseline arm (c): standard ECFP4 bits + Tanimoto distance. Likely a thin
  wrapper — ECFP4 already exists; the Tanimoto distance is the metric applied at scoring time, so this may be a
  distance-function registration rather than a rep. Resolved in Phase D wiring.

### Phase B — Structural-overlap (MCS) engine + framework cache  *(new files, deep/novel)*
- **B.1** New `src/kirby/similarity/mcs.py`: `compute_mcs_pairs(smiles, n_pairs=8000, seed)` → `{I, J, mcs, dy}`.
  Port from `similarity_metrics/mcs_cache.py:35-59` + `structural_yardstick.mcs_distance:66` (keep its `timeout=2`).
- **B.2** Per-dataset **structural cache** keyed by `(dataset_name, smiles_hash)` under `KIRBY_CACHE_DIR`.
  Compute-once/read-many — the cache the framework currently lacks. Must expose a **train-row-restricted view**
  for Phase C (only pairs with both endpoints in the train fold).
- **B.3** A precompute entry point so the roster can build the MCS cache per dataset.

### Phase C — Leakage-safe per-fold selection factories  *(gate G3, deep/novel)*
Registered through `find_best_hybrid_molecular.py:FACTORY_REPS:168` / `get_rep_factories:653` so they never see
the test fold (same leakage-safe path the framework already uses for per-fold reps):

- **C.1** Saboteur-drop factory — port `desc_blacklist_greedy:123-161`. Fits on **train rows' MCS only**, returns
  kept descriptor columns, applied to val without refit.
- **C.2** fine2D factory — port `select_fine2d:114-119`. Train-MCS only; top-100 of `[atompair_count | torsion]`.

### Phase D — Top-5 similarity recipes (generator + ranking + reps)  *(gate G2)*
- **D.1** Recipe generator: build candidate stacked spaces (count / +physchem-minus-saboteurs / +3D shape /
  +fine2D × weightings) via the `make_D:82` algebra — stack + z-score + `equal_block` (`/sqrt(d_block)`) +
  coarse-down weighting (`creative_hybrids_v2.py:82-93`), in **feature-matrix** (pre-distance) form.
- **D.2** Rank candidates by **structural fidelity** (label-blind, MCS) — `struct_fidelity:165`. Take **top 5**.
- **D.3** Register the top 5 as `sim_recipe_1..5`, built from the Phase-A/C blocks, added to `DATASET_REPS` for
  **all molecular datasets** (`complexity_helpers.py:94+`).

### Phase E — `objective` in the results-cache key  *(gate G1 — STEP 2 in build order)*
Predictive and similarity runs currently collide because `objective` is absent from the cache identity.

- **E.1** Add `'objective'` to the inline `result_db_identity` dict (`find_best_hybrid.py:5294`), to
  `_result_to_db_row:3494`, and to `_make_search_meta:337`. Make it a **first-class key field** (not riding inside
  `tier` the way `label_noise` does via `tier_name:5187` — that's the precedent to model, but keep objective explicit).
- **E.2** Confirm `find_best_hybrid(objective='similarity')` writes and `fetch_rows` reads back **distinctly** from
  predictive rows (`:4279` read path, `:4322` write path).

### Phase F — The three-arms convention + delete the hacks  *(gate G4)*
- **F.1** Shared helper (in `tests/experiments/_shared.py` or a new small module) that, per molecular dataset,
  yields all three arms: (a) top-5 similarity recipes, (b) predictive hybrid blueprint as a similarity space
  (`get_fitted_pool_hybrid` / `apply_blueprint`), (c) ECFP4/Tanimoto baseline. All cached (Phase E makes it safe).
- **F.2 Noise:** delete `tests/experiments/noise/_similarity_hybrid_space.py`; remove the `hybrid_sim` special-casing
  and env vars from `n11_estimation_remediation_grid.py` (`hybrid_sim`, `KIRBY_SIMILARITY_HYBRID_ONLY`,
  `KIRBY_NOISE_LIGHT_ESTIMATORS` — verified at n11:195,231-232,418-421,445-448,515) and n13. Env vars become
  **config keys** set in `noise_helpers.run_all:142-163`, threaded from a `run_dataset.py` flag (precedent:
  `--pruning-mode`). Consume the three arms from the rep pipeline instead of the cross-dir SMILES join.
- **F.3 Similarity (s01–s13):** add the three-arms wiring **additively**. Do not remove any existing similarity
  mechanism. Where an experiment already builds arm (b) (e.g. s02), keep it and add arms (a) and (c) alongside.

### Phase G — Characterization experiment, homed under noise
- **G.1** Port the noise-cleaning sweep (`similarity_metrics/similarity_noise_cleaning.py`) into a **registered**
  noise experiment on framework data + the new reps + the `simeval`-style σ×rate×config battery
  (`simeval.py:351-369`). Register in `noise_helpers` (`SINGLE_DATASET_EXPERIMENTS`, `_import_experiment`,
  `_experiment_full_name`).
- **G.2** Retire the standalone script (mark as research origin; framework version is live).

### Phase H — Precompute + roster for all molecular datasets
- **H.1** Roster MCS + 3D + selection precompute per molecular dataset
  (`tests/slurm_scripts/submit_all_datasets.sh` molecular list `:34` → `submit_dataset.sh` → `run_dataset.py`).
- **H.2** Full re-run (cache already wiped).

---

## Execution order (with the locked cache-key-second rule)

1. **Phase A** (rep building blocks) + **Phase B** (MCS engine/cache) — foundational, no cache-key dependency.
2. **Phase E** (`objective` in cache key) — **step 2**, so all subsequent similarity runs cache correctly.
3. **Phase C** (leakage-safe factories) → **Phase D** (recipes + reps).
4. **Phase F** (three-arms wiring + delete hacks) → **Phase G** (characterization) → **Phase H** (roster + re-run).

## What I will bring back for sign-off before touching it
- The exact `result_db_identity` diff (G1) before editing `find_best_hybrid.py`.
- The `DATASET_REPS` + registry diff (G2) before adding reps.
- The factory implementations (G3) for a leakage review before wiring.
- The n11/n13 deletion diff (G4) before removing the hacks.

## Open items still to confirm during build
- Whether arm (c) ECFP4/Tanimoto is a rep entry or a scoring-time distance function (Phase A.4 / D wiring).
- Exact candidate grid for the recipe generator (weightings × blocks) before ranking.
