# OPIG Experiments Audit — 2026-05-23

## Files audited

**Experiment files (`tests/experiments/`):**
- `imrie_transfer_learning.py` (422) — C4
- `opig_kingsmith_component_splits.py` (541) — C6
- `opig_trastuzumab_enrichment.py` (490) — B1
- `opig_humanness.py` (458) — B3
- `opig_graphinity_ddg.py` (544) — B2
- `opig_ginkgo_developability.py` (459) — B5
- `opig_activity_cliff_qsar.py` (560) — Dablander
- `opig_cdr_flexibility.py` (423) — Spoendlin
- `opig_cgr_comparison.py` (785) — CGR Phase 3
- `opig_blueprint_transfer.py` (536) — C5

**Loaders (`src/kirby/datasets/`):** `chembl_bioactivity.py` (353), `chembl_qsar.py` (216), `ginkgo_dev.py` (135), `graphinity.py` (365), `humanness.py` (325), `itsflexible.py` (220), `trastuzumab.py` (230)

**Models (`src/kirby/models/`):** `dnn_pretrained.py` (260), `pretrain_chembl_dnn.py` (400)

---

## P0 (broken / silently wrong results)

### P0-1. Imrie leakage filter excludes the wrong hERG dataset
- `src/kirby/datasets/chembl_bioactivity.py:328` — `collect_downstream_inchikeys` defaults to `['logd', 'caco2', 'herg', 'qm9', 'adme_*']`.
- `tests/experiments/imrie_transfer_learning.py:66` — `DEFAULT_DATASETS = ['logd', 'caco2', 'herg_ki']`.
- In `find_best_hybrid_molecular`, `'herg'` = hERG fluidigm loader (`kirby.datasets.herg.load_herg_fluid`), `'herg_ki'` = ChEMBL hERG Ki (`_fetch_chembl_herg_ki`, ~1415 compounds). Different datasets, different InChIKey sets.
- Result: pretraining excludes the wrong hERG compounds → direct **train/eval leakage** for the hERG-Ki fine-tune.
- **Fix:** add `'herg_ki'` to the default list at `chembl_bioactivity.py:328` (and consider keeping `'herg'` too); or pipe explicit downstream names from the Imrie experiment.

### P0-2. Trastuzumab experiment overrides edit_distance groups with sequence identity
- `tests/experiments/opig_trastuzumab_enrichment.py:251-258` rebuilds `groups` from unique heavy_seqs, discarding the `data['groups']` already produced by `load_dataset`.
- `antibody_baseline_survey.DATASET_REGISTRY['tz_her2_binary']` declares the strategy as `'edit_distance'` (line 86) and `_assign_groups` returns proper edit-distance buckets.
- Trastuzumab CDRH3 variants are nearly all unique → identity collapses to ~N singleton groups → `StratifiedGroupKFold` degenerates to stratified random — the **registry's leave-edit-distance-out protocol is bypassed**.
- Contradicts the memory note "StratifiedGroupKFold over edit_distance groups".
- **Fix:** replace lines 251-258 with `groups = data['groups']`.

### P0-3. Cache-key collision in 3 OPIG experiments
Per `project_cache_collision_fix.md`: `find_best_hybrid` / `get_best_search_for_downstream` default `augmentation_strategy='uncapped'`. When `augmentations=None` is passed without an explicit strategy, the cache key collides with real uncapped runs.
- `tests/experiments/opig_humanness.py:240-251` — `augmentations=None`, no strategy.
- `tests/experiments/opig_trastuzumab_enrichment.py:282-293` — same.
- `tests/experiments/opig_cgr_comparison.py:292-304` — same.
- **Fix:** add `augmentation_strategy='none'` to each call.

### P0-4. Trastuzumab + Humanness drop mutation info passed to LM reps
- `tests/experiments/opig_trastuzumab_enrichment.py:262-267` and `tests/experiments/opig_humanness.py:220-225` both call `compute_or_load_reps(...)` without `mutations=data.get('mutations')`.
- `trastuzumab.py` returns rich `mutations`; `rep_cache.py:112` predicates mutation-aware reps on `mutations is not None`.
- `DATASET_REPS['tz_her2_binary']` general list includes `mutation_features, esm2_wt_delta, esm2_mutsite, esm2_per_cdr, esm2_per_region` — silently unreachable from this experiment.
- Same "MUTATION_REPS two-pass" class as the prior audit.
- **Fix:** add `mutations=data.get('mutations')` to both `compute_or_load_reps` calls.

### P0-5. Graphinity transfer condition has no PDB-level leakage check
- `tests/experiments/opig_graphinity_ddg.py:157-251` (`evaluate_transfer`) trains on FoldX synthetic, tests on `ddg_combined` experimental — both keyed by PDB complex.
- Synthetic FoldX is derived from the same SAbDab PDBs SKEMPI uses; nothing filters synthetic rows whose PDB also appears in the experimental test set.
- Reported `pearson_global` overstates transfer because sequences in test are present verbatim in train.
- **Fix:** before training, build `test_pdbs = set(test_data['pdb_ids'])`, drop rows from `train_data` where `pdb` is in `test_pdbs`, print removed count.

---

## P1 (methodology drift / incorrect-but-recoverable)

### P1-1. Pool-best hardcoding across 7 OPIG experiments
All hardcode `tier='general'` with `condition_suffix` to bypass shared precompute:
- `opig_kingsmith_component_splits.py:316-329`
- `opig_graphinity_ddg.py:325-338`
- `opig_activity_cliff_qsar.py:308-321`
- `opig_cdr_flexibility.py:222-235`
- `opig_humanness.py:240-251`
- `opig_trastuzumab_enrichment.py:282-293`
- `opig_ginkgo_developability.py:273-283`
- `imrie_transfer_learning.py:195-201`

Tracked in `POOL_REFACTOR_ROADMAP.md`; comments admit choice is empirical.

### P1-2. Imrie GP silently skipped for high-dim reps
- `imrie_transfer_learning.py:100-107` — GP added only when `rep_dim <= 500`. ECFP4 (2048D) and mordred (1613D) miss GP. The cross-rep table at line 287-302 then shows uneven condition counts per rep; the model-vs-rep trade-off at line 328-339 uses Ridge as proxy. Document or drop GP entirely.

### P1-3. Ginkgo hybrid search uses singleton groups when `groups=None`
- `opig_ginkgo_developability.py:256` — `grp = groups if groups is not None else np.arange(n)`. Loader returns `groups=None`; falls back to `np.arange(n)` (every sample its own group). `_PrecomputedFoldSplitter` overrides the final CV, but the hybrid search's internal evaluator may still consume `groups`, behaving like LeaveOneOut on n=242.
- **Fix:** `groups = fold_assignments` so hierarchical clustering is respected through internal CV too.

### P1-4. Graphinity `_parse_complex_column` regex rejects multi-letter chain IDs
- `graphinity.py:90` — `re.match(r'^([A-Z])([A-Za-z])(\d+)([A-Z])$', mut_str)` allows only single-char chain. Multi-char chain PDB entries will silently drop (lines 263-267 only print a count). For current Graphinity CSVs OK; forward-compat smell.

### P1-5. Graphinity heavy/light chain decision by length heuristic
- `graphinity.py:130-172` — when `ab_chain='HL'` (2 chains), the longer chain is labeled heavy. Generally true for SAbDab but fragile; should use ANARCI / numbering-based typing.

### P1-6. CGR comparison `condition_suffix` contains spaces and `+`
- `opig_cgr_comparison.py:301` — `condition_suffix=pool_name` with values like `'SMILES-only'` / `'SMILES+CGR'`. Cache filename sanitization unverified; rename to `'smiles_only'` / `'smiles_plus_cgr'`.

### P1-7. Humanness PAPER_BASELINES for 'species' is empty dict
- `opig_humanness.py:97-101` — baselines key `'species'` is `{}`, so comparison rows never print for species. Dead path, not a bug.

### P1-8. Trastuzumab `--reps` accepts arbitrary names without validation
- `opig_trastuzumab_enrichment.py:462` — no check that requested reps exist in `ANTIBODY_REPS`/`MUTATION_REPS`. Silent "rep not generated".

### P1-9. Imrie pretrained checkpoint `feat_mean`/`feat_scale` requirement
- `dnn_pretrained.py:97-113` (`_fit_zero_shot`) raises if checkpoint missing those keys. Current `pretrain_chembl_dnn.py:227-234` writes them; older checkpoints on `/Volumes/seagate/kirby/chembl_pretrain` will hard-error — that's the right behaviour. Informational.

### P1-10. `pretrain_chembl_dnn.py:324` inherits the herg/herg_ki mismatch
Direct consequence of P0-1.

### P1-11. Graphinity `transfer_foldx` reports `std=0.0`
- `opig_graphinity_ddg.py:233-236` — single train/test, std hardcoded to 0. Misleading when averaging/comparing to k-fold runs with real std. Mark as n/a or run multiple seeds.

### P1-12. Kingsmith `assign_component_groups` lowercases — could merge case-distinct components
- `opig_kingsmith_component_splits.py:211-217` — `str(val).strip().lower()`. Unlikely with current BH HTE data but worth a comment.

---

## P2 (smells, future-proofing)

### P2-1. Bare except clauses swallow per-fold errors
- `opig_trastuzumab_enrichment.py:188, 204`
- `opig_humanness.py:158, 175`
- `opig_ginkgo_developability.py:211`
- `opig_activity_cliff_qsar.py:141`
- `opig_kingsmith_component_splits.py:253`
- `opig_cgr_comparison.py:236`
All should at least print `rep_name + type(e).__name__`.

### P2-2. `chembl_qsar.py` does no SMILES dedup or InChIKey conversion
- Relies on Dablander's "molecule_data_clean.csv" being clean. No defensive `drop_duplicates`.

### P2-3. `chembl_bioactivity._get_top_targets` exhaustively iterates all IC50 activities (millions of rows)
- `chembl_bioactivity.py:72-82` — slow but correct. Cache per-organism target list to avoid cold-start re-fetch.

### P2-4. Humanness `_find_heavy_fasta` heuristic
- `humanness.py:96-110` — picks first FASTA matching species + "heavy"/"vh" in path. Fragile if archive layout changes; falls back to `candidates[0]` silently. Fail loudly when no heavy candidate.

### P2-5. Ginkgo `fold_assignments` dtype assumption
- `ginkgo_dev.py:118` — `df[fold_col].values.astype(int)` crashes if any NaN. Add `dropna(subset=[fold_col])`.

### P2-6. CGR `linear_cka` no shape assertion
- `opig_cgr_comparison.py:372-397` — assumes `X.shape[0] == Y.shape[0]`; if a rep silently changes row count, returns garbage instead of erroring. Returns `0.0` on `denom < 1e-10` indistinguishably from "no similarity".

### P2-7. `opig_blueprint_transfer.py` source vs target metric not enforced
- `opig_blueprint_transfer.py:65-66` — metric derived from `target_data` only. With domain mismatch, source search uses target's metric for source data. Warning printed (lines 82-84) but run continues.

### P2-8. Trastuzumab LLR rep handling assumes `X.shape[1] <= 3`
- `opig_trastuzumab_enrichment.py:313-314` — brittle if LLR rep dim definition changes. Use a registry-driven check.

### P2-9. `opig_blueprint_transfer.py` uses `create_hybrid` on full source data
- `opig_blueprint_transfer.py:188` — intentional for "publish source's exact feature indices" condition. Documented inline. Noted only because the codebase generally prefers `find_best_hybrid`.

### P2-10. Imrie rank-correlation thresholds at 0.8 / 0.5 are arbitrary
- `imrie_transfer_learning.py:320-325` — hardcoded interpretation thresholds. Document source or mark as opinion.

---

## Verified-clean

- `opig_cdr_flexibility.py` — explicit `augmentation_strategy='none'` (line 228); proper StratifiedGroupKFold over CDR3 sequence identity; classification metrics correct (ROC AUC, PR AUC, MCC, balanced accuracy); no leakage path.
- `opig_blueprint_transfer.py` — single legitimate `create_hybrid` call documented as intentional; explicit `augmentation_strategy='none'`; per-fold `StandardScaler` correctly fit on train only.
- `itsflexible.py` — dedup happens AFTER label/Fv validity filter (lines 132-137), correct order per inline comment.
- `trastuzumab.py` — VH framework hardcoded per paper; stratified subsample on (label, edit_distance) with fixed seed 42; mutations reconstructed correctly from CDRH3 diff.
- `humanness.py` — no Thera-SAbDab fallback path exists (memory note confirmed); BioPhi IMGT TSV+FASTA loader is straightforward.
- `chembl_qsar.py` — pActivity unit conversion (nM × 1e-9, uM × 1e-6) correct; MMP pair indexing rebuilt from SMILES→index dict.
- `dnn_pretrained.py` freeze logic (lines 131-151) — verified: with `freeze_layers=-1` and 3 hidden blocks, freezes all 9 hidden modules and leaves final Linear trainable. Pretrained weights loaded before freeze (lines 192-197). Optimizer constructed only from `requires_grad=True` params (line 200).
- `pretrain_chembl_dnn.py` checkpoint format — includes `feat_mean`/`feat_scale`/`y_mean`/`y_std`/`input_dim`; vet-eval leakage filter union (lines 330-341) handled.
- Global checks: no `best_overall` usage anywhere; no random splits; no Ridge sensitivity; no PCA-reduced variants; no mean baselines. `INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']`. `DOWNSTREAM_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']`.
