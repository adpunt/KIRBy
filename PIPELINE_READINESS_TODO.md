# Pipeline Readiness TODO

**Purpose:** Single operational document for the KIRBy experiment pipeline. Tracks dataset readiness, methodology status, blockers, and the path from preliminary runs to publication-quality results.

**Last updated:** 2026-04-01

### WARNING: Two hERG Datasets
- **`herg_ki`** = ChEMBL hERG Ki (regression, pKi, ~1,412 molecules). Loaded via `find_best_hybrid_molecular.py`. All existing surveys/experiments use this version.
- **`herg`** = Lhasa/TDC hERG blocker (binary classification, IC50 threshold, ~10K molecules). Loaded via `src/kirby/datasets/herg.py`. **No baseline survey yet — NEEDS SURVEY.**
- The classification `herg` dataset is used by the federated experiments (FLuID). The regression `herg_ki` is used by complexity/similarity/noise experiments.

---

## Workflow Vision

The pipeline follows a staged approach:

1. **Preliminary runs:** Once F12/F13 results land and methodology is finalized, immediately run complexity/similarity/noise experiments on the 33 ready datasets. These are screening-level — they use the finalized allocation/evaluator defaults but are not the final paper results.

2. **Incremental onboarding:** As new datasets get prepared (surveys completed, tier decisions made, DATASET_REPS populated), run them through the relevant experiments when they're ready. Complexity, similarity, and noise apply to all datasets. OPIG and federated experiments are specific to certain datasets only.

3. **Refinement:** Modify experiments based on preliminary results. This is expected — experiments start in a preliminary running state, not a final one. Iterate on what's interesting.

4. **Final runs:** When the picture is clear — which datasets tell the story, which experiments produce interesting results — trigger the selected experiments in full search mode on the selected datasets for publication-quality results.

**The goal is seamless flow:** F12/F13 finalize → run experiments on ready datasets → onboard new datasets as their surveys land → refine → select → publish.

---

## Publication Artifacts

**What gets published per dataset.** Each dataset that goes through the pipeline produces a self-contained artifact — a "recipe" any researcher can use to reproduce or apply the best hybrid.

### Per-dataset artifact contents

For each dataset in the registry (`~/.cache/kirby/hybrid_registry.json`):

| Field | Description | Example |
|---|---|---|
| `dataset` | Dataset name (matches DATASET_REPS key) | `caco2` |
| `metric` | Primary evaluation metric | `rmse` |
| **`lower_is_better`** | **CRITICAL: metric direction** | `true` |
| `model` | Best downstream model | `rf` |
| `budget` | Feature budget used | `200` |
| `score_mean` | CV score (in original metric units) | `0.6234` |
| `score_std` | CV standard deviation | `0.0312` |
| `blueprint` | Full Blueprint dict (JSON-serialisable) | see below |

### Blueprint contents (the recipe)

```json
{
  "reps": ["ecfp4", "maccs", "pdv"],
  "allocation_method": "top_3",
  "importance_method": "lightgbm_gain",
  "internal_evaluator": "svm",
  "budget": 200,
  "metric": "rmse",
  "lower_is_better": true,
  "rep_scores": {"ecfp4": 0.82, "maccs": 0.71, "pdv": 0.65}
}
```

### METRIC DIRECTION — NON-NEGOTIABLE RULES

**Every published artifact carries `metric` + `lower_is_better` as an inseparable pair.** This applies to:

1. **Blueprint** — has `metric` and `lower_is_better` fields. `Blueprint.from_dict()` / `Blueprint.to_dict()` round-trip both.
2. **SearchResult** — has `metric_name` and `lower_is_better` fields. Set at construction, never inferred.
3. **Registry entries** — every dataset entry has top-level `metric` and `lower_is_better`.
4. **Notebook output** — every score display includes `(lower is better)` or `(higher is better)`.
5. **JSON exports** — always include both fields.

**Direction-aware comparisons:** Code that selects "best" must always check `lower_is_better`:
```python
cmp = min if lower_is_better else max
best = cmp(candidates, key=lambda x: x['score_mean'])
```

Never write bare `max(scores)` for "best" without checking direction.

### How to use published artifacts

```python
# One-line lookup from registry
from kirby.find_best_hybrid import get_best_hybrid, apply_blueprint

info = get_best_hybrid('caco2', model='rf')  # reads registry
# info['lower_is_better'] tells you the direction — always check it

# Apply the blueprint to your data
X_hybrid, feature_info = apply_blueprint(my_reps, info['blueprint'], labels=my_labels)
```

### Registry population

`precompute_hybrids.py` populates the registry automatically. Each successful `find_best_hybrid` call writes the best config per (dataset, tier, aug_condition, model, budget) to `hybrid_registry.json`. The registry is additive — re-running a dataset updates its entries without touching others.

### Notebook (kirby_quickstart.ipynb)

The notebook demonstrates all three workflows:
1. Full search from scratch
2. One-line lookup by dataset name (`get_best_hybrid`)
3. Apply a blueprint to new data (`apply_blueprint`)

All comparisons are direction-aware. All score displays include the direction label.

---

## Current Blockers

### ~~F12/F13 (framework experiments)~~ — DONE (2026-03-29)

**F12/F13 analysis complete. SPEED_TIERS rebuilt. Hybrid code updated.**

Key findings:
- **Model/evaluator choice dominates** (eta² 0.23–0.89). Strategy matters much less (eta² 0.01–0.07). Importance method barely registers (<0.01).
- **Top strategies (by top-1 wins across 24 dataset×model combos):** pw_top_2 (3 wins), pw_top_3/5, pw_linear, pw_min_floor, pw_entropy_1.0, pw_score_gap_0.2 (2 each), 6 others with 1 win each
- **Cut (2026-04-03):** All strategies that never produced a top-1 winner across any dataset×model combo. 31→13 strategies. All greedy family dropped except greedy_fill_s5 (1 win). Redundant rank_exp variants collapsed (kept d0.5 only).
- **Matching evaluator→model:** weak signal (eta² < 0.003), top configs are overwhelmingly cross
- **Budget:** 300 wins 12/24, 150 wins 7/24, 50 wins 5/24. All three matter.
- **Importance:** dead heat (deeplift 3/6, lightgbm_gain 3/6); treeshap dropped from full tier (not tested in F12)

Code changes applied (2026-03-29 → 2026-04-03):
1. `NAMED_ALLOCATIONS` expanded from 22→39 entries (all F12 variants). `min_floor` added as named allocation.
2. Dead strategies removed: greedy_feature, min_floor_0.05, two_stage, stability_selection (dispatch removed, defs kept)
3. `SPEED_TIERS` → **two tiers only: `fast` and `full`**. No medium/long. Budgets not in tiers (in experiment configs).
   - Fast: 3 allocs (entropy_1.0, top_3, top_2), 2 imp (treeshap, lightgbm_gain), match evaluator, 3-fold
   - Full: 13 allocs (F12-validated winners only), 2 imp (deeplift, lightgbm_gain), match evaluator, 5-fold
4. `MODEL_MAX_BUDGET` added: GP/KNN capped at 2000, SVM/MLP/BNN/DNN/CNN at 3000, tree models uncapped.
5. `top_k` guard: skips top_k strategies when k >= n_reps (would be identical to performance_weighted).
6. Sequential search: sklearn models handle their own C-level parallelism via `n_jobs=-1`.
4. `internal_evaluator` wired through entire pipeline (find_best_hybrid → _run_search → _evaluate_hybrid_config → create_hybrid). Tracked in Blueprint + result dicts. Callable evaluators supported.
5. GP added as internal evaluator (O(n³)-safe with 500-sample cap)
6. `create_hybrid` defaults set from F12 top-2 analysis: `top_3` / `lightgbm_gain` / `svm`
7. All 30 hardcoded `allocation_method='greedy'` across 21 experiment files → `entropy_1.0`
8. `cached_create_hybrid` (full-data, data-leaking) **deleted**. Replaced with fold-aware `create_hybrid_cached` in _shared.py
9. `PRELIMINARY`, `INTERMEDIARY`, `FULL` config dicts **deleted**. All helpers build config from SPEED_TIERS directly.
10. `DEFAULT_MODELS` tier labels updated: `fast` (rf, xgboost) and `full` (all 11)
11. Budget grids restored in `get_config()`: screening [10,25,50,100,200,500,1000,2000], full [10,25,50,100,200,300,500,750,1000,1500,2000]

Remaining:
1. ~~Clear hybrid caches on server~~ **DONE** (2026-03-29)
2. Push code + pull on server (`git push` → `git pull` on ARC)
3. ~~Begin preliminary runs on ready datasets~~ **IN PROGRESS** (2026-04-05) — precompute v6 submitted. **Job tracker: [`PRECOMPUTE_JOB_TRACKER.md`](PRECOMPUTE_JOB_TRACKER.md)** (160 jobs, IDs 11729406–11729565). Medium, 64G, 4 CPUs, full speed (3 imp methods incl treeshap), 4 conditions, no FT, SKIP_INSTALLS, cache v9.

### ~~FT experiments~~ — MOSTLY DONE (2026-04-04)

- **Molecular FT fill:** ALL 4 COMPLETED (qm9, adme_hlm, adme_rlm, adme_mdr1). Results in `tests/results/molecular_ft_fill/`. Pulled locally.
- **Antibody ESM-2 LoRA fill:** ALL 29 datasets COMPLETED (was 11, now full coverage including ginkgo, abagym, DMS, Ab CLF). Results in `tests/results/antibody_ft_fill/`. Pulled locally.
- **E2E SchNet:** caco2 + herg done (pulled locally). All 3 (logd, herg_ki, caco2) resubmitted on HTC long partition 2026-04-04 (previous runs failed exit 127 on ARC). In queue.
- Per-dataset FT swap decisions can now proceed for molecular + antibody. SchNet logd still pending.

### New dataset surveys — status as of 2026-04-04

- ~~Reaction: buchwald_hartwig, suzuki, bh_doyle_plates~~ **DONE** — buchwald_hartwig + bh_doyle_plates reviewed and added to experiments. suzuki DROPPED (R²=0.453, 1 viable rep).
- ~~Ab DMS fill: adams (binding+expression), phillips_h3, koenig (binding+expression)~~ **DONE** — all 5 reviewed and added to experiments.
- ~~Graphinity DDG~~ **ALL DROPPED** — foldx (best ρ=0.052), rosetta (best ρ=0.079), augmented (best ρ=0.057). All R² negative.
- ~~Molecular aug fill~~ **DONE** — caco2, logd, qm9, adme_mdr1 all completed. Confirmed existing augmentation assignments.
- DTA: davis, kiba, bindingdb_kd — resubmitted on HTC long partition 2026-04-04 (ARC runs failed exit 127 / timed out). In queue.
- Nucleotide: 22 datasets — ARC run completed with 7/14 FM reps (7 failed: caduceus, ntv3_100m, splicebert, dnabert_s, gpn, rnafm, rinalmo due to missing packages). Fixed on HTC (installed multimolecule, gpn-genomics, mamba-ssm). All 22 resubmitted on HTC long partition 2026-04-04.
- Ab CLF: tz_her2_multiclass + humanness_species — resubmitted on HTC long partition 2026-04-04. tz_her2_binary DONE.
- herg classification (TDC binary, ~10K): previous run loaded herg_ki (regression) by mistake — server code was stale. Resubmitted on HTC long partition 2026-04-04 after git pull.

### Cache warning

~~Complexity and similarity experiments that ran previously used stale hybrid caches (pre-augmentation defaults: step_size=10, patience=3, augmentation_strategy='greedy_ablation'). Must clear caches after F12/F13 finalization before any new runs.~~ **CLEARED** (2026-03-29). Both `~/.cache/kirby/hybrid_search/` and `~/.cache/kirby/create_hybrid_cache/` deleted on server.

---

## Methodology Status

| Component | Status | Blocked on |
|-----------|--------|-----------|
| SPEED_TIERS | **FINALIZED:** 2 tiers (fast/full). No medium/long. Sequential search. | — |
| Allocation strategy | **FINALIZED:** 13 F12-validated winners. Default: `top_3`. top_k guard skips when k>=n_reps. | — |
| Importance methods | **FINALIZED:** deeplift, lightgbm_gain (full); treeshap, lightgbm_gain (fast). | — |
| Internal evaluator | **FINALIZED:** match (default). Wired through full pipeline. | — |
| Budgets | **FINALIZED:** In experiment config. Per-model caps: GP/KNN≤2000, SVM/MLP≤3000, trees uncapped. | — |
| Augmentation defaults | **FINALIZED:** uncapped, step_size=5, patience=20 | — |
| Hybrid caching | **FINALIZED:** Fold-aware `create_hybrid_cached` in _shared.py. Old full-data cache deleted. | — |
| FT swap rules | **PARTIAL:** molecular 3/7 done, TCGA done, antibody pending | HTC queue |
| Metrics | **FINALIZED:** domain-aware (RMSE/CI/AUROC/F1w/ρ/MCC) | — |
| Classification support | **FINALIZED:** C02/C08/C09/C10/C11/N01/N07 fixed | — |
| Augmentation passing | **FINALIZED:** C11/C16-C19, all similarity, all noise fixed | — |

**FT decisions already locked (from `EXPERIMENT_DESIGN_DISCUSSION.md` §9):**
- Molecular: lora_molformer, contrastive_molformer, graphmvp_ft → general base reps. Per-dataset swap only where FT beats frozen. herg_ki keeps frozen molformer. All chemberta/selformer/freeze_head FT excluded.
- TCGA: scfoundation_ft is FT-only (frozen dead). geneformer_ft general tier (frozen v2 fallback).
- Antibody/DTA FT: pending HTC results.
- Excluded everywhere: antiberty_ft (broken), all DTA protein LoRA (broken), chemberta/selformer FT (worse on all datasets).

---

## Experiment Categories

5 categories. Complexity, similarity, and noise apply to **all** ready datasets. OPIG and federated are dataset-specific.

| Category | Experiments | Datasets | Helpers | Roadmap |
|----------|------------|----------|---------|---------|
| Complexity | C01-C22, C24 (C23 deprioritized 2026-04-15) | All 40 ready | `complexity_helpers.py` | `EXPERIMENT_REFACTOR_ROADMAP.md` |
| Similarity | S01-S13 | All 40 ready | `similarity_helpers.py` | `EXPERIMENT_REFACTOR_ROADMAP.md` |
| Noise | N01-N13 all active (regression datasets only — cut for classification 2026-04-15) | All regression datasets | `noise_helpers.py` | `NOISE_EXPERIMENT_ROADMAP.md` |
| OPIG | 6 standalone scripts | Specific OPIG datasets | — | `OPIG_COLLAB_ROADMAP.md` |
| Federated | fed01-fed05 all implemented | herg, caco2, logd | `federated_helpers.py` | `FEDERATED_EXPERIMENT_ROADMAP.md` |

**C23 deprioritization (2026-04-15):** Evaluator-model matching experiment cut from the active run list. Prior runs confirm the methodology finding (matched > mismatched, especially NNs); ~5× compute cost per dataset makes re-running on 9 antibody classification datasets not worth it. Code retained for future cross-domain robustness appendix if needed. See `EXPERIMENT_REFACTOR_ROADMAP.md` for details.

**Noise on classification (2026-04-15):** Noise category cut for classification datasets. Of 13 scripts, only n02 (sample weighting) is classification-ready; the rest assume continuous targets (Gaussian injection, residual-based detection, continuous noise distributions). n01/n03/n04/n05/n07/n08/n11/n12 could be reframed for classification with label-flip noise models but that's a significant rewrite — deferred. Noise experiments continue on regression datasets (molecular, DTA, reaction, regression antibody).

**Noise (N09-N13):** All implemented. N09 (LNL survey, 12 methods), N10 (hybrid neighborhood LNL), N11 (estimation × remediation grid), N12 (importance-weighted distance), N13 (alt noise distributions). New utility: `src/kirby/utils/lnl_methods.py` (12 training-loop LNL methods). See `NOISE_EXPERIMENT_ROADMAP.md`.

**Federated (fed01-fed05):** All 5 implemented: fed01 (noise heterogeneity), fed02 (rep diversity silos), fed03 (synthetic private datasets), fed04 (cluster-aware importance), fed05 (antibody extension). Uses group-aware single 80/20 split (by design for FLuID comparison). RF hardcoded (matches FLuID). Datasets: herg, caco2, logd. `lhasa_herg_loader.py` loads Lhasa curated data. See `FEDERATED_EXPERIMENT_ROADMAP.md`.

**Impact on per-dataset files:** None — the unified runner handles dispatch. Adding a new experiment to any category just requires registering it in the helpers file. Adding a new category requires 1 helpers file + 1 entry in `run_dataset.py`'s `CATEGORIES` dict.

---

## Per-Dataset Experiment Runner — DONE (2026-03-27)

Replaced ~100 per-dataset boilerplate .py files with a single unified runner:

```bash
python tests/experiments/run_dataset.py complexity herg
python tests/experiments/run_dataset.py noise herg --experiments n01 n02
python tests/experiments/run_dataset.py similarity logd --output /custom/path
python tests/experiments/run_dataset.py federated herg
```

**SLURM:** Unified into 2 scripts (was 6):
```bash
sbatch tests/slurm_scripts/submit_dataset.sh complexity herg
sbatch tests/slurm_scripts/submit_dataset.sh noise herg "n01 n02"
bash tests/slurm_scripts/submit_all_datasets.sh complexity              # all 33
bash tests/slurm_scripts/submit_all_datasets.sh noise --domain molecular # 7
bash tests/slurm_scripts/submit_all_datasets.sh federated               # herg only
```

Override SLURM resources at sbatch time: `sbatch --time=96:00:00 --mem=128G submit_dataset.sh federated herg`

**To add a new dataset:** Add to `DATASET_REPS` in `complexity_helpers.py` + add to domain list in `submit_all_datasets.sh`. 0 new files.

**To add a new experiment:** Create the .py file + register in the category's helpers `module_map`/`name_map` + uncomment in `SINGLE_DATASET_EXPERIMENTS`. 0 other changes.

**To add a new category:** Create `{category}_helpers.py` with `run_all()` + `SINGLE_DATASET_EXPERIMENTS` + add 1 entry to `CATEGORIES` in `run_dataset.py` + add prefix in `submit_all_datasets.sh`.

---

## New Dataset Pipeline

When a dataset becomes ready (survey results land from server), follow this checklist:

1. **Pull results:** rsync survey JSON from server to `tests/results/`
2. **Review data:** Show ALL standalone results sorted by primary metric, all augmentation tests, all supplementary data (FT, forward compat, graph survey, comparisons)
3. **Decide tiers:** User decides top tier, general, augmentation rep cutoffs and viability
4. **Update DATASET_REPS:** Add entry in `complexity_helpers.py` with `general` + `augmentation` + `top_tier` rep lists
5. **Add to submit lists:** Add dataset name to the appropriate domain list in `submit_all_datasets.sh`
6. **Submit preliminary runs:** `sbatch submit_dataset.sh complexity {dataset_name}` (repeat for similarity, noise)
7. **Record here:** Update this document with tier tables and mark DONE

For OPIG-specific datasets: also check `OPIG_COLLAB_ROADMAP.md` for relevant standalone experiments.
For federated-eligible datasets: also add to federated experiment list per `FEDERATED_EXPERIMENT_ROADMAP.md`.

---

## Rules for Each Dataset Review

1. Show ALL standalone results: every rep sorted by primary metric, with dimension, all metrics, best model, rep family
2. Show ALL augmentation test results: base + augment, scores, deltas
3. Show ALL supplementary data that exists (fine-tuning, forward compat, graph survey, comparisons, OPIG, DMS)
4. DISCUSS — explain what I think the cutoff should be and why, then wait for user decisions
5. User decides the cutoff and augmentation selections
6. Write the row ONLY after user confirms
7. Do NOT skip supplementary data — fine-tuning, OPIG, comparisons all inform the decision

**Primary metrics by domain:**
- Molecular: RMSE (lower = better), with R² and ρ as supporting
- Antibody (regression): Spearman ρ (higher = better)
- TCGA classification: AUROC (binary) or F1w (multi-class)
- Reaction: R² (higher = better)
- Nucleotide: MCC (higher = better)
- DTA: CI (higher = better)
- Antibody CLF: AUROC / F1w

**Viability thresholds (starting points, user may override):**
- Molecular: R² > 0.5
- Reaction: R² > 0.2
- TCGA clf: AUROC > 0.6
- TCGA multi-class: F1w > 0.4
- Antibody: ρ > 0.15
- DTA: CI > 0.55

**Small rep rule (dim < 25D → augmentation):** For datasets with N > 1000 samples, reps with fewer than 25 dimensions are placed in augmentation instead of general, regardless of their standalone score. Their value is as supplementary features, not as budget competitors. For datasets with N ≤ 1000, this rule does NOT apply — small reps may be the only viable options and need to participate in allocation.

**FT rep notation:**
- `*` = **FT-only.** The frozen version of this rep is dead (produces no signal). There is no fallback — if FT is turned off, this rep is excluded entirely.
- `**` = **FT improves, frozen fallback exists.** The FT version is better, but the frozen version is a viable base rep in its own right. When FT is disabled, the frozen version substitutes in.

---

## Completed

| # | Dataset | Domain | Samples | Status | BOM Location |
|---|---------|--------|---------|--------|-------------|
| 1 | caco2 | Molecular | 2,158 | DONE | Section 10 (new format) |
| 2 | herg_ki | Molecular (regression) | 1,412 | DONE | Section 10 (new format) |
| 2b | herg | Molecular (classification) | ~10K | **NEEDS SURVEY** | — |
| 3 | logd | Molecular | 5,028 | DONE | Section 10 (new format) |
| 4 | qm9 | Molecular | 9,978 | DONE | Section 10 (new format) |
| 5 | adme_hlm | Molecular | 3,086 | DONE | Section 10 (new format) |
| 6 | adme_rlm | Molecular | 3,053 | DONE | Section 10 (new format) |
| 7 | adme_mdr1 | Molecular | 2,641 | DONE | Section 10 (new format) |
| 8 | adme_solubility | Molecular | 2,172 | DROPPED | R²=0.32 ceiling, not viable |
| 47 | tcga_luad_vs_lusc | TCGA | 1,017 | DONE | Section 10 (new format) |
| 48 | tcga_brca_pam50 | TCGA | 821 | DONE | Section 10 (new format) |
| 49 | tcga_brca_survival | TCGA | 151 | DROPPED | All R² negative |
| 9 | flab_thermo_tresanco | Antibody | 672 | DONE | Section 10 (new format) |
| 10 | flab_pk_jain_heprt | Antibody | 130 | DONE | Section 10 (new format) |
| 11 | flab_pk_jain_fcrnrt | Antibody | 132 | DONE | Section 10 (new format) |
| 12 | flab_polyreact_jain_cic | Antibody | 137 | DONE | Section 10 (new format) |
| 13 | flab_aggregation_jain_hicrt | Antibody | 137 | DONE | Section 10 (new format) |
| 14 | flab_thermo_jain | Antibody | 137 | DONE | Section 10 (new format) |
| 15 | abcov_kd | Antibody | 141 | DONE | Section 10 (new format) |
| 16 | flab_aggregation_jain | Antibody | 137 | DONE | Section 10 (new format) |
| 17 | flab_immunogenicity_marks | Antibody | 217 | DONE | Section 10 (new format) |
| 18 | flab_polyreact_jain_bvp | Antibody | 137 | DONE | Section 10 (new format) |
| 21 | ddg_combined | Antibody | 429 | DONE | Section 10 (new format) |
| 34 | tz_her2_binary | Ab CLF | 5,000 | DONE | Section 10 (new format) |
| 35 | humanness_therapeutic | Ab CLF | 564 | DONE | Section 10 (new format) |
| 38 | ginkgo_acsins | Ginkgo | 242 | DONE | Section 10 (new format) |
| 40 | ginkgo_colloidal | Ginkgo | 242 | DONE | Section 10 (new format) |
| 41 | ginkgo_heparin | Ginkgo | 94 | DONE | Section 10 (new format) |
| 42 | ginkgo_hic | Ginkgo | 242 | DONE | Section 10 (new format) |
| 43 | ginkgo_pr_cho | Ginkgo | 197 | DONE | Section 10 (new format) |
| 45 | ginkgo_titer | Ginkgo | 239 | DONE | Section 10 (new format) |
| 39 | ginkgo_aggregation | Ginkgo | 242 | DROPPED | best ρ=0.181, near random |
| 44 | ginkgo_purity | Ginkgo | 242 | DROPPED | best ρ=0.220, weak + high variance |
| 46 | ginkgo_tm2 | Ginkgo | 193 | DROPPED | best ρ=0.166, near random |
| 28 | abagym_all | AbAgym | 2,247 | DROPPED | best ρ=0.145, below viability |
| 29 | abagym_Ang2 | AbAgym | 397 | DONE | Section 10 (new format) |
| 30 | abagym_EGFR | AbAgym | 659 | DONE | Section 10 (new format) |
| 31 | abagym_HER2 | AbAgym | 146 | DONE | Section 10 (new format) |
| 32 | abagym_lysozyme | AbAgym | 627 | DONE | Section 10 (new format) |
| 33 | abagym_VEGF | AbAgym | 418 | DONE | Section 10 (new format) |
| 23 | flab_binding_koenig | Ab DMS | 4,275 | DONE | Section 10 (new format) |
| 24 | flab_expression_koenig | Ab DMS | 4,275 | DONE | Section 10 (new format) |
| 25 | flab_binding_adams | Ab DMS | 5,000 | DONE | Section 10 (new format) |
| 26 | flab_expression_adams | Ab DMS | 5,000 | DONE | Section 10 (new format) |
| 27 | flab_binding_phillips_h3 | Ab DMS | 5,000 | DONE | Section 10 (new format) |
| 50 | buchwald_hartwig | Reaction | 3,955 | DONE | Section 10 (new format) |
| 52 | bh_doyle_plates | Reaction | 4,312 | DONE | Section 10 (new format) |
| 51 | suzuki | Reaction | 5,760 | DROPPED | R²=0.453 ceiling, 1 viable rep |
| 75 | graphinity_foldx | Graphinity | 48,900 | DROPPED | best ρ=0.052, all R² negative |
| 76 | graphinity_rosetta | Graphinity | ~21K | DROPPED | all R² negative, best ρ=0.079 |
| 77 | graphinity_augmented | Graphinity | 49,329 | DROPPED | best ρ=0.057, all R² negative |

**caco2 (2,158 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | graph_kernel, mol2vec, mhg_gnn, ecfp4, pdv, grover, avalon, mordred |
| General (12) | graph_kernel, mol2vec, mhg_gnn, ecfp4, pdv, grover, avalon, mordred, top_torsion, smited, atom_pair, molformer |
| Augmentation (5) | maccs, pharmacophore3d, usrcat, coulomb_matrix, usr |

**herg_ki (1,412 samples, regression):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4 |
| General (15) | mhg_gnn, mordred, graph_kernel, avalon, grover, pdv, mol2vec, ecfp4, molformer, chemberta, atom_pair, chembert, smited, top_torsion, graphmvp |
| Augmentation (5) | maccs, pharmacophore3d, coulomb_matrix, usr, usrcat |

*Note: lora_molformer and contrastive_molformer excluded from herg_ki — both worse than frozen molformer on this dataset (-0.051 and -0.005 R² respectively). Frozen molformer stays.*

**logd (5,028 samples):**
| Tier | Reps |
|------|------|
| Top Tier (8) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer |
| General (15) | mhg_gnn, mordred, grover, mol2vec, pdv, ecfp4, smited, molformer, avalon, atom_pair, chemberta, top_torsion, chembert, graph_kernel, graphmvp |
| Augmentation (6) | selformer, rdkfp, maccs, pharmacophore3d, molclr, usrcat |

**qm9 (9,978 samples):**
| Tier | Reps |
|------|------|
| Top Tier (6) | mordred, pdv, grover, mhg_gnn, avalon, e2e_schnet* |
| General (16) | mordred, pdv, grover, mhg_gnn, avalon, rdkfp, atom_pair, smited, ecfp4, selformer, maccs, chemberta, mol2vec, chembert, graphmvp, molformer |
| Augmentation (4) | top_torsion, pharmacophore3d, usrcat, coulomb_matrix |

*e2e_schnet: factory rep, requires per-fold 3D fine-tuning (128D)

**adme_hlm (3,086 samples) — hard task, R² ceiling 0.46, viability by ρ > 0.5:**
| Tier | Reps |
|------|------|
| Top Tier (6) | mordred, pdv, mhg_gnn, smited, molformer, grover |
| General (16) | mordred, pdv, mhg_gnn, smited, molformer, grover, mol2vec, chemberta, atom_pair, chembert, graph_kernel, avalon, top_torsion, ecfp4, selformer, graphmvp |
| Augmentation (5) | pharmacophore3d, maccs, usrcat, coulomb_matrix, usr |

**adme_rlm (3,053 samples) — hard task, R² ceiling 0.49, viability by ρ > 0.5:**
| Tier | Reps |
|------|------|
| Top Tier (6) | pdv, mhg_gnn, smited, mordred, molformer, grover |
| General (16) | pdv, mhg_gnn, smited, mordred, molformer, grover, mol2vec, chemberta, chembert, graph_kernel, atom_pair, selformer, avalon, ecfp4, top_torsion, graphmvp |
| Augmentation (5) | pharmacophore3d, maccs, usrcat, coulomb_matrix, usr |

**adme_mdr1 (2,641 samples):**
| Tier | Reps |
|------|------|
| Top Tier (5) | mordred, pdv, mhg_gnn, avalon, grover |
| General (16) | mordred, pdv, mhg_gnn, avalon, grover, smited, molformer, atom_pair, graph_kernel, maccs, chemberta, ecfp4, chembert, selformer, top_torsion, rdkfp |
| Augmentation (3) | pharmacophore3d, maccs, coulomb_matrix |

### Molecular Domain-Level Decisions

**FT reps — FULL RESULTS (2026-04-04, all 7 datasets verified):**

Sources: `comparisons/molecular_finetune_comparison.json` (logd, caco2, herg_ki), `molecular_ft_fill/` (qm9, adme_hlm, adme_rlm, adme_mdr1), `phase5_baselines/molecular/` (frozen baselines). All numbers are RMSE (lower=better).

| Dataset | lora_molformer vs frozen | contrastive vs frozen | graphmvp_ft vs frozen |
|---------|------------------------|-----------------------|----------------------|
| logd | **FT WINS** (0.549 vs 0.573, Δ-0.024) | **FT WINS** (0.557 vs 0.573, Δ-0.016) | **FT WINS** (0.560 vs 0.667, Δ-0.107) |
| caco2 | **FT WINS** (0.322 vs 0.337, Δ-0.015) | ~TIE (0.333 vs 0.337, Δ-0.005) | FT WINS (0.339 vs 0.345, Δ-0.006) |
| qm9 | **FT WINS** (0.534 vs 0.607, Δ-0.073) | MISSING | ~TIE (0.598 vs 0.600, Δ-0.001) |
| herg_ki | FROZEN WINS (0.676 vs 0.643, Δ+0.033) | ~TIE (0.646 vs 0.643, Δ+0.003) | **FT WINS** (0.676 vs 0.702, Δ-0.026) |
| adme_hlm | FROZEN WINS (0.512 vs 0.474, Δ+0.038) | FROZEN WINS (0.511 vs 0.474, Δ+0.037) | FROZEN WINS (0.538 vs 0.523, Δ+0.015) |
| adme_rlm | FROZEN WINS (0.602 vs 0.547, Δ+0.055) | FROZEN WINS (0.606 vs 0.547, Δ+0.058) | FROZEN WINS (0.630 vs 0.607, Δ+0.022) |
| adme_mdr1 | FROZEN WINS (0.533 vs 0.505, Δ+0.028) | FROZEN WINS (0.533 vs 0.505, Δ+0.029) | **FT WINS** (0.541 vs 0.565, Δ-0.024) |

**Per-dataset FT swap decisions (clear wins only, Δ > 0.005):**
- **logd:** lora_molformer replaces molformer, contrastive_molformer added to general, graphmvp_ft replaces graphmvp
- **caco2:** lora_molformer replaces molformer. graphmvp_ft borderline (~TIE).
- **qm9:** lora_molformer replaces molformer
- **herg_ki:** graphmvp_ft replaces graphmvp. Frozen molformer kept.
- **adme_hlm:** No FT reps — all frozen win
- **adme_rlm:** No FT reps — all frozen win
- **adme_mdr1:** graphmvp_ft replaces graphmvp (graphmvp not in current general tier but FT version is strong enough to add)

Pattern: FT helps on logd/caco2/qm9 (larger, cleaner datasets). ADME datasets (harder tasks, lower R² ceiling) → FT hurts (overfitting).

**Implementation blocker — FT reps not wired into experiment pipeline:**
- FT reps need per-fold fine-tuning (different embeddings per fold)
- Experiments currently only handle frozen reps (one matrix per rep)
- No cached FT embeddings exist on server (surveys computed on-the-fly without caching)
- **Plan: Precompute script (option B)**
  1. New script: `tests/precompute_ft_reps.py` — for each (dataset, FT rep, fold), fine-tune and save embeddings to `~/.cache/kirby/finetuned_reps/`
  2. Uses `fold_cache_key()` from `src/kirby/finetuning/_base.py` for deterministic cache keys
  3. Folds must match experiments: `GroupKFold(n_splits=5)` with scaffold groups, same seed
  4. Experiments then construct FT factories as normal — factories hit cache and load pre-saved `.npz` files instead of re-finetuning
  5. `_shared.load_data()` needs to return `rep_factories` dict alongside `reps` for FT rep names
  6. Experiment helpers pass both `base_reps` and `rep_factories` to `find_best_hybrid()`
- **Precompute jobs needed (GPU, ~2h each):**
  - logd: lora_molformer, contrastive_molformer, graphmvp_ft (3 reps × 5 folds)
  - caco2: lora_molformer (1 rep × 5 folds)
  - qm9: lora_molformer (1 rep × 5 folds)
  - herg_ki: graphmvp_ft (1 rep × 5 folds)
  - adme_mdr1: graphmvp_ft (1 rep × 5 folds)
- FT factory constructors: `make_molecular_lora_factory`, `make_molecular_contrastive_factory`, `make_molecular_gnn_factory` in `src/kirby/finetuning/`

All other FT variants (chemberta FT, selformer FT, freeze_head anything) excluded — never beat ecfp4.

**Augmentation (apply to all molecular datasets):**
- pharmacophore3d and maccs are domain-wide augmentation reps for all molecular datasets. pharmacophore3d is the most consistent augment rep tested (13/15 improve in graph survey, positive on 5/6 phase5 datasets). maccs shows small but reliable signal everywhere (6/7 datasets).
- usrcat, coulomb_matrix, usr: now added to all molecular datasets (caco2 and adme_mdr1 were missing, fixed 2026-03-28).

**ACTION items:**
- Run E2E SchNet on logd, herg, caco2 to determine if it generalizes beyond qm9. torch-cluster CUDA bug fixed (2026-03-28) — was producing NaN on all 3D methods due to CPU-only torch-cluster on GPU. Fallback radius graph now used automatically.

### TCGA Domain-Level Decisions

**tcga_luad_vs_lusc (1,017 samples, binary, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (8) | scfoundation_ft*, gene_geneformer_v2, gene_bulkrnabert, methyl_hvg, gene_geneformer, gene_chromosome_arm, wsi_titan, methyl_pca50, cnv_pca50 |
| General (28) | all 28 reps with AUROC > 0.6 + geneformer_ft** |
| Augmentation | none — no augment improved the base |

**tcga_brca_pam50 (821 samples, 4-class, F1w):**
| Tier | Reps |
|------|------|
| Top Tier (3) | scfoundation_ft*, gene_geneformer_v2, gene_geneformer, gene_bulkrnabert |
| General (27) | all 27 reps with F1w > 0.4 + geneformer_ft** |
| Augmentation (6) | hrd_score, mut_tmb, purity_consensus, gene_stats, gene_chromosome_arm, immune_xcell |

**FT reps (apply to both TCGA datasets):**
- scfoundation_ft* — FT-only, frozen scfoundation is dead (AUROC=0.500). Best single result: LUAD 0.988 AUROC, BRCA 0.877 F1w (hybrid). No fallback.
- geneformer_ft** — FT improves (LUAD 0.977, BRCA 0.967 AUROC), but frozen geneformer_v2 (0.946/0.861) is already top tier. Frozen V2 substitutes when FT disabled.

**Dead reps (both datasets):** clinical, gene_scfoundation (frozen), heterogeneity_math, msi_status, wsi_conch, wsi_hoptimus1, wsi_uni2h, wsi_virchow2 — all AUROC=0.500 exactly.

**Pending factory reps:** gene_pca100, gene_hallmark, gene_progeny showed strong results in FT survey (gene_pca100: 0.915 AUROC LUAD, 0.806 F1w BRCA). Not yet in phase5 baselines — will likely enter general tier when available.

### Antibody Inter-Family Domain-Level Decisions

**General tier cutoff: ~60% of best ρ per dataset.** Antibody-specific LMs (ablang2, igt5) and general protein LMs (prot_t5, esm2 variants) dominate. Rankings are highly dataset-specific.

**Domain-wide dead reps:** aido_protein, igbert, igbert_cls, mint, xtrimopglm — NaN or near-zero on all datasets.

**Size-dependent reps:** esm3 is spectacular on tresanco (n=672, ρ=0.805) but dead on n≈130 datasets. blosum/onehot strong on tresanco but weak on small datasets.

**flab_thermo_tresanco (n=672, best ρ=0.805, cutoff=0.48):**
| Tier | Reps |
|------|------|
| Top Tier (9) | esm3, saprot, blosum, esmc, prot_t5, igt5, onehot, prot_t5_cdr, esm2_early |
| General (21) | all reps with ρ > 0.48 |
| Augmentation (1) | mutation_features (+0.012 on esm3) |

**flab_pk_jain_heprt (n=130, best ρ=0.783, cutoff=0.47):**
| Tier | Reps |
|------|------|
| Top Tier (6) | igt5, mutation_features, esm2_concat, ablang2, prot_t5, prot_t5_cdr |
| General (7) | all reps with ρ > 0.47 |
| Augmentation (2) | paratope, structural (+0.019 on esm2_concat) |

**flab_pk_jain_fcrnrt (n=132, best ρ=0.595, cutoff=0.36):**
| Tier | Reps |
|------|------|
| Top Tier (2) | ablang2, igt5 |
| General (11) | all reps with ρ > 0.36 |
| Augmentation (1) | aac (+0.011 on ablang2) |

**flab_polyreact_jain_cic (n=137, best ρ=0.586, cutoff=0.35):**
| Tier | Reps |
|------|------|
| Top Tier (3) | prot_t5, prot_t5_cdr, esm2_early |
| General (12) | all reps with ρ > 0.35 |
| Augmentation | none |

**flab_aggregation_jain_hicrt (n=137, best ρ=0.506, cutoff=0.30):**
| Tier | Reps |
|------|------|
| Top Tier (3) | prot_t5, prot_t5_cdr, esm2_early |
| General (14) | all reps with ρ > 0.30 |
| Augmentation (2) | paratope/structural (+0.032), mutation_features (+0.025) |

**flab_thermo_jain (n=137, best ρ=0.457, cutoff=0.27):**
| Tier | Reps |
|------|------|
| Top Tier (4) | ablang2, abroberta, igt5, esm3 |
| General (16) | all reps with ρ > 0.27 |
| Augmentation (1) | aac (+0.032 on ablang2) |

**abcov_kd (n=141, best ρ=0.438, cutoff=0.26):**
| Tier | Reps |
|------|------|
| Top Tier (8) | esm2_entropy_top20, prot_t5, blosum, antiberty, esm2_midlayers, abroberta, esm2, ablang2 |
| General (28) | all reps with ρ > 0.26 |
| Augmentation | none |

**flab_aggregation_jain (n=137, best ρ=0.425, cutoff=0.26):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_concat, ablang2, igt5 |
| General (25) | all reps with ρ > 0.26 |
| Augmentation (1) | paratope/structural (+0.008 on igt5) |

**flab_immunogenicity_marks (n=217, best ρ=0.399, cutoff=0.24):**
| Tier | Reps |
|------|------|
| Top Tier (8) | antiberty_cdr, antiberty, esm2_entropy_top20, igt5, prot_t5, prot_t5_cdr, abroberta_cdr, abroberta |
| General (29) | all reps with ρ > 0.24 |
| Augmentation | none |

**flab_polyreact_jain_bvp (n=137, best ρ=0.374, cutoff=0.22):**
| Tier | Reps |
|------|------|
| Top Tier (4) | ablang2, igt5, antiberty, antiberty_cdr |
| General (23) | all reps with ρ > 0.22 |
| Augmentation | none |

**ddg_combined (n=429, best ρ=0.293, cutoff=0.18):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_wt_delta, prot_t5, esm2_midlayers |
| General (19) | all reps with ρ > 0.18 |
| Augmentation (1) | aac (+0.026 on esm2_wt_delta) |

### AbAgym Domain-Level Decisions

DMS datasets — mutation_features is the standout rep (top 1-2 on lysozyme, HER2, VEGF). Rankings are antigen-specific. abagym_all (mixed antigens) dropped — signal dilutes when antigens are combined.

**abagym_Ang2 (n=397, best ρ=0.472):**
| Tier | Reps |
|------|------|
| Top Tier (4) | saprot, esm3, esm2_midlayers, esm2_layer24 |
| General (24) | all reps with ρ > 0.28 (through esm2_concat) |
| Augmentation | none |

**abagym_EGFR (n=659, best ρ=0.308):**
| Tier | Reps |
|------|------|
| Top Tier (5) | antiberty, antiberty_cdr, esm2_layer24, esm2_early, esm2_midlayers |
| General (26) | all reps with ρ > 0.18 (through aac) |
| Augmentation | none |

**abagym_HER2 (n=146, best ρ=0.369):**
| Tier | Reps |
|------|------|
| Top Tier (3) | mutation_features, prot_t5_cdr, prot_t5 |
| General (13) | all reps with ρ > 0.24 (through ablang2) |
| Augmentation | none |

**abagym_lysozyme (n=627, best ρ=0.490):**
| Tier | Reps |
|------|------|
| Top Tier (1) | mutation_features |
| General (3) | mutation_features, aac, blosum |
| Augmentation (2) | structural (+0.007), antiberty_llr (+0.009) |

**abagym_VEGF (n=418, best ρ=0.613):**
| Tier | Reps |
|------|------|
| Top Tier (4) | mutation_features, aac, prot_t5_cdr, prot_t5 |
| General (24) | all reps with ρ > 0.39 (through esm2_entropy_top20) |
| Augmentation | none |

### Antibody CLF Domain-Level Decisions

Classification tasks (AUROC). Different character from regression — sequence encodings (blosum, onehot) do well.

**tz_her2_binary (n=5,000, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (2) | blosum, onehot |
| General (29) | all reps with AUROC > 0.80 (through aac) |
| Augmentation | none |

**humanness_therapeutic (n=564, AUROC):**
| Tier | Reps |
|------|------|
| Top Tier (5) | antiberty, igt5, abroberta, ablang2, blosum |
| General (10) | all reps with AUROC > 0.83 (through aac) |
| Augmentation | none |

### Ginkgo Domain-Level Decisions

Developability properties (regression, Spearman ρ, pre-computed hierarchical cluster splits). paratope/structural shows small but consistent augmentation signal across datasets.

**Domain-wide augmentation:** paratope/structural (+0.004 to +0.007 on multiple datasets)

**ginkgo_acsins (n=242, best ρ=0.560):**
| Tier | Reps |
|------|------|
| Top Tier (2) | igt5, ablang2 |
| General (26) | all reps with ρ > 0.24 (through esm2_entropy_top20) |
| Augmentation | paratope/structural |

**ginkgo_colloidal (n=242, best ρ=0.417):**
| Tier | Reps |
|------|------|
| Top Tier (2) | prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.20 (through blosum) |
| Augmentation | paratope/structural |

**ginkgo_heparin (n=94, best ρ=0.739):**
| Tier | Reps |
|------|------|
| Top Tier (4) | esm2_concat, prot_t5, prot_t5_cdr, ablang2 |
| General (29) | all reps with ρ > 0.36 (through blosum) |
| Augmentation | paratope/structural |

**ginkgo_hic (n=242, best ρ=0.604):**
| Tier | Reps |
|------|------|
| Top Tier (3) | esm2_concat, prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.35 (through esm2_layer24) |
| Augmentation | paratope/structural |

**ginkgo_pr_cho (n=197, best ρ=0.506):**
| Tier | Reps |
|------|------|
| Top Tier (2) | prot_t5, prot_t5_cdr |
| General (23) | all reps with ρ > 0.22 (through abroberta) |
| Augmentation | paratope/structural |

**ginkgo_titer (n=239, best ρ=0.399):**
| Tier | Reps |
|------|------|
| Top Tier (2) | saprot, saprot_cdr |
| General (23) | all reps with ρ > 0.19 (through structural) |
| Augmentation | paratope/structural |

---

## In Progress

---

## TODO — Molecular (8 datasets)

| # | Dataset | Samples | Reps | Augs | Supplementary Data | Status |
|---|---------|---------|------|------|--------------------|--------|
| 3 | logd | 5,028 | 24 | 6 | FT (8 files), graph survey, Imrie TL, FT comparison | DONE |
| 4 | qm9 | 9,978 | 22 | 14 | FT N/A, graph survey, qm9_3d_comparison (SchNet/DimeNet++) | DONE |
| 5 | adme_hlm | 3,086 | 24 | 7 | No FT/comparison data | DONE |
| 6 | adme_rlm | 3,053 | 24 | 6 | No FT/comparison data | DONE |
| 7 | adme_mdr1 | 2,641 | 22 | 14 | No FT/comparison data | DONE |
| 8 | adme_solubility | 2,172 | 22 | 14 | No FT/comparison data | DROPPED — R²=0.32 ceiling |

**Molecular supplementary data (applies across multiple datasets):**
- `finetune_survey/`: 8 FT strategies × {logd, caco2, herg_ki} — LoRA, LoRA r2/r4, freeze, contrastive, GNN, GROVER
- `comparisons/molecular_finetune_comparison.json`: frozen vs FT for 25 conditions × {logd, caco2, herg_ki}
- `comparisons/imrie_transfer_learning.json`: ChEMBL DNN pretraining × {logd, caco2, herg_ki}
- `molecular_graph_survey/`: GNN augmentation for {herg_ki, logd, qm9}
- `comparisons/qm9_3d_comparison/`: SchNet, DimeNet++ E2E vs frozen reps (qm9 only)

---

## TODO — Antibody Inter-Family (14 datasets)

Old Section 10G has a bulk summary with 11 datasets. Each needs individual cutoff + augmentation decisions.

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 9 | flab_thermo_tresanco | 672 | 34 | 4 | DONE |
| 10 | flab_pk_jain_heprt | 130 | 37 | 4 | DONE |
| 11 | flab_pk_jain_fcrnrt | 132 | 37 | 4 | DONE |
| 12 | flab_polyreact_jain_cic | 137 | 37 | 4 | DONE |
| 13 | flab_aggregation_jain_hicrt | 137 | 37 | 4 | DONE |
| 14 | flab_thermo_jain | 137 | 37 | 4 | DONE |
| 15 | abcov_kd | 141 | 33 | 3 | DONE |
| 16 | flab_aggregation_jain | 137 | 37 | 4 | DONE |
| 17 | flab_immunogenicity_marks | 217 | 37 | 4 | DONE |
| 18 | flab_polyreact_jain_bvp | 137 | 37 | 4 | DONE |
| 19 | flab_thermo_hutchinson | 192 | 38 | 5 | DROPPED — best ρ=0.289, weak ceiling |
| 20 | flab_aggregation_jain_sas | 137 | 37 | 4 | DROPPED — best ρ=0.230, only 8 viable reps |
| 21 | ddg_combined | 429 | 34 | 4 | DONE — also has OPIG DDG data (opig/ddg/, may differ from phase5) |
| 22 | ddg_combined_full | 1,536 | 30 | 8 | DROPPED — best ρ=0.231, only 1 viable rep |

**Antibody supplementary data:**
- `finetune_survey/ft_ab_*.json`: 4 FT strategies × {abagym_Ang2, flab_binding_koenig, flab_expression_koenig}
- `comparisons/antibody_finetune_comparison.json`: frozen vs FT, 6 datasets
- `comparisons/antibody_region_ablation.json`: region ablation, 5 datasets
- `comparisons/antibody_region_entropy.json`: entropy analysis, 5 datasets
- `forward_compat/exp1_mint/`: MINT on ddg_combined, flab_binding_koenig, flab_expression_koenig, flab_thermo_tresanco
- `forward_compat/exp2_saprot/`: SaProt on same 4 datasets
- `forward_compat/exp4_ablm/`: AbLM reps on 5 datasets
- `forward_compat/exp6_xtrimopglm/`: xTrimoPGLM on 3 datasets
- `forward_compat/exp9_aido_protein/`: AIDO-Protein on 3 datasets
- `antibody_survey/`: 3 ESM2 region reps × 24 datasets
- `dms_baselines/`: 9 DMS datasets × 20 specialized reps (varpos, LLR scores, etc.)
- `geometry_survey/`: 3D geometry reps on 15 antibody datasets (may be empty)

### Antibody ESM-2 LoRA FT Results (2026-04-04, all 29 datasets verified)

Source: `antibody_ft_fill/*.json`. Comparing frozen ESM-2 (best ρ across ridge/rf/xgboost) vs ESM-2 LoRA.

**Summary: 5 FT wins, 20 frozen wins, 4 ties.** LoRA mostly hurts — overfits on small/hard datasets.

**FT wins (esm2_lora replaces or supplements esm2):**

| Dataset | n | Frozen ρ | LoRA ρ | Δ | Notes |
|---------|---|---------|--------|---|-------|
| flab_binding_phillips_h3 | 5,000 | 0.652 | 0.739 | +0.087 | Large DMS, strong improvement |
| abagym_Ang2 | 397 | 0.486 | 0.604 | +0.118 | Largest AbAgym gain |
| abagym_EGFR | 659 | 0.126 | 0.168 | +0.043 | Modest but clear |
| abagym_HER2 | 146 | -0.049 | 0.101 | +0.150 | Frozen was dead, LoRA rescues |
| abcov_kd | 141 | 0.390 | 0.460 | +0.070 | Clear gain |

**~TIE (no change):**

| Dataset | n | Frozen ρ/AUROC | LoRA | Δ |
|---------|---|---------------|------|---|
| flab_thermo_tresanco | 672 | 0.707 | 0.717 | +0.010 |
| flab_binding_adams | 5,000 | 0.571 | 0.553 | -0.017 |
| flab_expression_adams | 5,000 | 0.469 | 0.482 | +0.013 |
| humanness_therapeutic | 564 | 0.915 | 0.897 | -0.017 |

**Frozen wins (20 datasets) — LoRA excluded:**
- All 8 inter-family small datasets (n=130-217): Δ ranges -0.044 to -0.200
- All 6 Ginkgo datasets: Δ ranges -0.050 to -0.245
- Both Koenig DMS datasets: Δ -0.097 and -0.242
- ddg_combined: Δ -0.183
- tz_her2_binary: Δ -0.021
- abagym_VEGF: Δ -0.089
- abagym_lysozyme: Δ -0.082

**Pattern:** LoRA helps on DMS/mutation tasks where per-position fine-tuning adds signal (phillips_h3, AbAgym antigen-specific). Badly overfits on small inter-family (n~130), Ginkgo (n~200), and DDG datasets.

**Per-dataset FT swap decisions:**
- **flab_binding_phillips_h3:** esm2_lora replaces esm2 in general (LoRA ρ=0.739 would be top-2 rep)
- **abagym_Ang2:** esm2_lora added to general (LoRA ρ=0.604 > frozen esm2 0.486)
- **abagym_EGFR:** esm2_lora added to general (modest gain but clear)
- **abagym_HER2:** esm2_lora added to general (frozen esm2 was dead)
- **abcov_kd:** esm2_lora added to general (ρ=0.460 > frozen 0.390)
- All other datasets: frozen esm2 kept, no LoRA

**Same implementation blocker as molecular FT:** needs precompute pipeline (option B). esm2_lora is a per-fold factory rep.

---

## TODO — Antibody DMS / Large DMS (datasets with ≥1000 samples or DMS-specific reps)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 23 | flab_binding_koenig | 4,275 | 34 | 12 | DONE |
| 24 | flab_expression_koenig | 4,275 | 34 | 12 | DONE |
| 25 | flab_binding_adams | 5,000 | 35 | 10 | DONE |
| 26 | flab_expression_adams | 5,000 | 35 | 10 | DONE |
| 27 | flab_binding_phillips_h3 | 5,000 | 35 | 10 | DONE |

**flab_binding_koenig (n=4,275, best ρ=0.475, cutoff=0.28):**
| Tier | Reps |
|------|------|
| Top Tier (8) | ablang2, saprot, saprot_cdr, esm3, esm3_cdr, prot_t5, prot_t5_cdr, igt5 |
| General (26) | all reps with ρ > 0.28 (through antiberty_cdr) |
| Augmentation (4) | esm2_llr, mutation_features, antiberty_llr, aac |

**flab_expression_koenig (n=4,275, best ρ=0.715, cutoff=0.43):**
| Tier | Reps |
|------|------|
| Top Tier (7) | esm3, esm3_cdr, prot_t5_cdr, prot_t5, saprot, saprot_cdr, igt5 |
| General (27) | all reps with ρ > 0.43 (through aac) |
| Augmentation (3) | mutation_features, esm2_llr, aac |

**flab_binding_adams (n=5,000, best ρ=0.669, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (8) | blosum, onehot, esm3, esm3_cdr, saprot, saprot_cdr, esm2_cdr, esm2_cdr3 |
| General (26) | all reps with ρ > 0.50 (through esm2_mutsite) |
| Augmentation (2) | aac, paratope |

**flab_expression_adams (n=5,000, best ρ=0.511, cutoff=0.42):**
| Tier | Reps |
|------|------|
| Top Tier (7) | blosum, saprot, saprot_cdr, esm3, esm3_cdr, prot_t5, prot_t5_cdr |
| General (28) | all reps with ρ > 0.42 (through antiberty_cdr) |
| Augmentation (1) | esm2_llr |

**flab_binding_phillips_h3 (n=5,000, best ρ=0.722, cutoff=0.67):**
| Tier | Reps |
|------|------|
| Top Tier (4) | onehot, blosum, igt5, esm2_early |
| General (29) | all reps with ρ > 0.67 (through abroberta) |
| Augmentation (3) | paratope, aac, mutation_features |

---

## TODO — AbAgym (6 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 28 | abagym_all | 2,247 | 35 | 5 | DROPPED — best ρ=0.145, below viability |
| 29 | abagym_Ang2 | 397 | 36 | 5 | DONE |
| 30 | abagym_EGFR | 659 | 36 | 4 | DONE |
| 31 | abagym_HER2 | 146 | 36 | 4 | DONE |
| 32 | abagym_lysozyme | 627 | 36 | 3 | DONE |
| 33 | abagym_VEGF | 418 | 36 | 4 | DONE |

**Supplementary:** FT comparison (abagym_Ang2), DMS baselines (all 5 individual + all), `pp_survey/` (physicochemical hybrids, 5 individual + koenig×2), `tuned_survey/` (tuned hybrids, same 7 datasets)

---

## TODO — Antibody CLF (3 datasets)

| # | Dataset | Samples | Reps | Augs | Metric | Status |
|---|---------|---------|------|------|--------|--------|
| 34 | tz_her2_binary | 5,000 | 37 | 5 | AUROC/F1w | DONE |
| 35 | humanness_therapeutic | 564 | 16 | 4 | AUROC/F1w | DONE |
| 36 | tz_her2_multiclass | 5,000 | 1 | 0 | F1w | BLOCKED — only 1 rep |
| 37 | humanness_species | — | 0 | 0 | AUROC/F1w | BLOCKED — no file, just submitted |

**Note:** tz_her2_multiclass has only structural rep. humanness_species has no file yet (just submitted 2026-03-24).

---

## TODO — Ginkgo Developability (9 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 38 | ginkgo_acsins | 242 | 34 | 2-3 | DONE |
| 39 | ginkgo_aggregation | 242 | 34 | 2-3 | DROPPED — best ρ=0.181 |
| 40 | ginkgo_colloidal | 242 | 34 | 2-3 | DONE |
| 41 | ginkgo_heparin | 94 | 34 | 2-3 | DONE |
| 42 | ginkgo_hic | 242 | 34 | 2-3 | DONE |
| 43 | ginkgo_pr_cho | 197 | 34 | 2-3 | DONE |
| 44 | ginkgo_purity | 242 | 34 | 2-3 | DROPPED — best ρ=0.220 |
| 45 | ginkgo_titer | 239 | 34 | 2-3 | DONE |
| 46 | ginkgo_tm2 | 193 | 34 | 2-3 | DROPPED — best ρ=0.166 |

---

## TODO — TCGA (3 datasets)

Old Section 10D/10E have full writeups with tiers. Need to reconcile with new format.

| # | Dataset | Samples | Reps | Augs | Metric | Status |
|---|---------|---------|------|------|--------|--------|
| 47 | tcga_luad_vs_lusc | 1,017 | 36 | 20 | AUROC | DONE |
| 48 | tcga_brca_pam50 | 821 | 36 | 20 | F1w | DONE |
| 49 | tcga_brca_survival | 151 | 36 | 20 | R² | DROPPED (10F) — all R² negative |

**TCGA supplementary data:**
- `finetune_survey/ft_tcga_*.json`: 3 strategies (base, geneformer, scfoundation) × 2 datasets
- `comparisons/tcga_foundation_comparison/`: frozen vs FT summary + per-dataset
- `comparisons/wsi_abmil_comparison/`: 6 WSI models × 3 pooling strategies × 2 datasets
- `tcga_new_reps/`: 19 factory reps (earlier run, may be stale)

---

## TODO — Reaction (3 datasets)

Old Sections 10B/10C/12D/12E have writeups. Need new-format review.

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 50 | buchwald_hartwig | 3,955 | 14 | 6 | BLOCKED — re-survey needed with CGR reps (cgr_morgan, chytorch_graphormer ready; cgr_smiles_lm blocked on pretraining) |
| 51 | suzuki | 5,760 | 14 | 4 | DROPPED — R²=0.453 ceiling, only 1 viable rep |
| 52 | bh_doyle_plates | 4,312 | 35 | 24 | BLOCKED — re-survey needed with CGR reps (same as buchwald_hartwig) |

**buchwald_hartwig (n=3,955, best R²=0.718, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (3) | reactant_ecfp4, diff_ecfp4, drfp_2048 |
| General (8) | all reps with R² > 0.50 (through rxnfp) |
| Augmentation | none |

**bh_doyle_plates (n=4,312, best R²=0.638, cutoff=0.50):**
| Tier | Reps |
|------|------|
| Top Tier (6) | reactant_avalon, diff_avalon, diff_mhg_gnn, diff_ecfp4, reactant_grover, reactant_ecfp4 |
| General (17) | all reps with R² > 0.50 (through diff_mordred) |
| Augmentation (6) | product_maccs, reactant_maccs, onehot_components, diff_maccs, product_pharmacophore3d, reactant_pharmacophore3d |

**Reaction supplementary data:**
- `comparisons/reaction_rep_comparison.json`: rep comparison across all 3 datasets
- `forward_compat/exp7_reactiont5/`: 14 reps + reactiont5 on all 3 datasets

---

## TODO — DTA (2-3 datasets)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| 53 | davis_frozen | — | 20 | — | TODO — old dta_survey/ data only |
| 54 | kiba_frozen | — | 20 | — | TODO — old dta_survey/ data |
| 55 | kiba_ft | — | 20 | — | TODO — old dta_survey/ data |

**DTA supplementary data:**
- `dta_survey/`: 20 reps per dataset (older survey)
- `comparisons/dta_e2e_comparison/`: E2E DTA models (davis, kiba)
- `comparisons/dta_finetune_comparison/`: DTA FT comparison
- `structural_protein_survey/kiba.json`: cold-target with structural protein reps (6 reps)
- Phase 5 DTA: PENDING (jobs in queue, no results yet)

---

## TODO — Nucleotide (19 datasets)

**⚠️ 2026-04-15 DOWNGRADE:** All 5 datasets previously marked DONE
(nt_H2AFZ, nt_H3K27ac, nt_H3K27me3, nt_H3K36me3, nt_H3K4me1) are
reverted to survey-review status. The gena_lm rep in those JSONs was
contaminated: `_embed_gena_lm` was returning 32000D LM-head logits
instead of 1024D hidden states, and ridge regression on 32000 features
with ~33000 samples memorized noise — which is why gena_lm was
declared the winner on every H3K* task. Fixed in commit 3a5b5d5.
Targeted gena_lm-only re-eval submitted as jobs 7485363-78 (gena_refix
mode). Tier tables below are INVALID until those jobs complete and
are reviewed. See PRECOMPUTE_JOB_TRACKER.md §Nucleotide gena_lm refix.

| # | Dataset | Samples | Reps | Metric | Status |
|---|---------|---------|------|--------|--------|
| 56 | gb_human_nontata_promoters | 36,131 | 6 | MCC | Survey review needed (gena_lm refix 7485377 PD) |
| 57 | nt_enhancers | — | 6 | MCC | Classical only — survey pending (7483707 PD) |
| 58 | nt_enhancers_types | — | 6 | MCC | Classical only — survey pending (7483708 PD) |
| 59 | nt_H2AFZ | 33,000 | 14 | MCC | **DONE** — gena_lm corrected (1024D, MCC=0.019, dropped). Winner: mistral_dna+ridge=0.4323 |
| 60 | nt_H3K27ac | 31,616 | 17 | MCC | **DONE** — gena_lm corrected (dropped, 4.5% of best). Winner: kmer6+ridge=0.4564 |
| 61 | nt_H3K27me3 | 33,000 | 14 | MCC | **DONE** — gena_lm corrected (dropped, 5.2% of best). Winner: mistral_dna+ridge=0.5487 |
| 62 | nt_H3K36me3 | 33,000 | 14 | MCC | **DONE** — gena_lm corrected (dropped, 3.2% of best). Winner: mistral_dna+mlp=0.5310 |
| 63 | nt_H3K4me1 | 33,000 | 14 | MCC | **DONE** — gena_lm corrected (dropped, 5.2% of best). Winner: grover_dna+ridge=0.4207 |
| 64 | nt_H3K4me2 | 32,138 | 17 | MCC | **DONE** — gena_lm corrected (dropped, 6.6% of best). Winner: mistral_dna+ridge=0.5054 |
| 65 | nt_H3K4me3 | — | 18 | MCC | Survey review needed (gena_lm refix 7485369 PD) |
| 66 | nt_H3K9ac | — | 18 | MCC | Survey review needed (gena_lm refix 7485370 PD) |
| 67 | nt_H3K9me3 | — | 18 | MCC | Survey review needed (gena_lm refix 7485371 PD) |
| 68 | nt_H4K20me1 | — | 18 | MCC | Survey review needed (gena_lm refix 7485372 PD) |
| 69 | nt_promoter_all | 31,584 | 18 | MCC | Survey review needed (gena_lm refix 7485373 PD) |
| 70 | nt_promoter_no_tata | — | 18 | MCC | Survey review needed (gena_lm refix 7485375 PD) |
| 71 | nt_promoter_tata | — | 18 | MCC | Survey review needed (gena_lm refix 7485374 PD) |
| 72 | nt_splice_sites_acceptors | — | 6 | MCC | Classical only — survey pending (7483709 PD) |
| 73 | nt_splice_sites_all | — | 18 | MCC | Survey review needed (gena_lm refix 7485376 PD) |
| 74 | nt_splice_sites_donors | — | 6 | MCC | Classical only — survey pending (7483710 PD) |

**Root cause:** `DATASET_REGISTRY` in `run_baseline_survey.py` had default preset `'fast'` (6 classical reps only) for all nucleotide datasets. Survey was submitted without `--reps all` override, so FM reps never ran. **Fixed 2026-03-28** (commit a2a0d2d): default changed to `'all'` (26 reps: 14 FM + 11 classical + 1 factory).

**Reps implemented (26 total):** 14 foundation models + 11 classical + 1 factory. Code: `src/kirby/representations/nucleotide.py`. SLURM: `submit_nucleotide_survey.sh` (short partition, 11:59:00, GPU).

**Resubmission:** Pull on server, run `bash tests/slurm_scripts/submit_nucleotide_survey.sh`. Existing classical results will merge in automatically. Output: `tests/results/nucleotide_survey/`.

**Preliminary decisions (from classical-only results):**
- DROP nt_splice_sites_all (MCC=0.245) and nt_H3K9me3 (MCC=0.239) — near random (pending confirmation after FM reps land)
- kmer4, kmer6, tetranuc_freq will be general-tier classical baselines
- dinuc_freq, gc_content are lower-tier classical; complexity is augmentation-only

**⚠️ CONTAMINATED TIER TABLES BELOW (2026-04-15)**

The 5 tier tables that follow all declare `gena_lm` as a top-tier / winner
rep. Those claims are **INVALID**. The pre-3a5b5d5 `_embed_gena_lm` returned
32000D LM-head logits instead of 1024D hidden states. Ridge regression on
32000 features × ~33000 samples memorized noise, producing spuriously high
MCC for gena_lm on every H3K* task. The real numbers land after
jobs 7485363-78 complete. Tables below retained for historical record only.

**nt_H2AFZ (33,000 samples, 23 groups, MCC) — 14/25 reps ran (7 failed: splicebert, rnafm, dnabert_s, gpn, rinalmo, caduceus, ntv3_100m):**

CORRECTED 2026-04-17 (gena_refix 7485363 + 7496791). gena_lm now 1024D hidden states.

| Tier | Reps |
|------|------|
| Top Tier (4) | mistral_dna, hyenadna, grover_dna, kmer6 |
| General (11) | mistral_dna, hyenadna, grover_dna, kmer6, nt_v2_250m, nt_v2_500m, tnf_canonical, kmer4, kmer5, tetranuc_freq, kmer3 |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (2) | dnabert2 (84% of best), gena_lm (4% of best — near random at 1024D) |

*Winner: mistral_dna+ridge=0.4323 MCC. Best augmentation: mistral_dna+tnf_canonical=0.4512 (+0.019).*

**nt_H3K27ac (31,616 samples, 23 groups, MCC) — 17/25 reps ran. CORRECTED 2026-04-17 (gena_refix 7496792):**

| Tier | Reps |
|------|------|
| Top Tier (5) | kmer6, grover_dna, mistral_dna, kmer5, tnf_canonical |
| General (10) | kmer6, grover_dna, mistral_dna, kmer5, tnf_canonical, nt_v2_500m, nt_v2_250m, kmer4, hyenadna, tetranuc_freq |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (3) | dnabert2 (84% of best), kmer3 (89% of best), gena_lm (near-random at 1024D) |

*Winner: kmer6+ridge=0.4564 MCC. Best aug (excl. gena_lm): kmer6+complexity=0.4579 (+0.0015, marginal).*

**nt_H3K27me3 (33,000 samples, 23 groups, MCC) — 14/25 reps ran. CORRECTED 2026-04-17 (gena_refix 7496793):**

| Tier | Reps |
|------|------|
| Top Tier (5) | mistral_dna, grover_dna, hyenadna, nt_v2_250m, nt_v2_500m |
| General (11) | mistral_dna, grover_dna, hyenadna, nt_v2_250m, nt_v2_500m, tnf_canonical, kmer4, kmer3, kmer5, tetranuc_freq, kmer6 |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (2) | dnabert2 (84% of best), gena_lm (near-random at 1024D) |

*Winner: mistral_dna+ridge=0.5487 MCC. Best aug: mistral_dna+tnf_canonical=0.5620 (+0.0133).*

**nt_H3K36me3 (33,000 samples, 23 groups, MCC) — 14/25 reps ran. CORRECTED 2026-04-17 (gena_refix 7496794):**

| Tier | Reps |
|------|------|
| Top Tier (5) | mistral_dna, hyenadna, nt_v2_250m, nt_v2_500m, grover_dna |
| General (11) | mistral_dna, hyenadna, nt_v2_250m, nt_v2_500m, grover_dna, kmer3, kmer4, kmer5, kmer6, tetranuc_freq, tnf_canonical |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (2) | dnabert2 (80% of best), gena_lm (near-random at 1024D) |

*Winner: mistral_dna+mlp=0.5310 MCC. Best aug: mistral_dna+tnf_canonical=0.5449 (+0.0139).*

**nt_H3K4me1 (33,000 samples, 23 groups, MCC) — 14/25 reps ran. CORRECTED 2026-04-17 (gena_refix 7496795):**

| Tier | Reps |
|------|------|
| Top Tier (5) | grover_dna, mistral_dna, hyenadna, nt_v2_250m, kmer6 |
| General (11) | grover_dna, mistral_dna, hyenadna, nt_v2_250m, kmer6, kmer5, nt_v2_500m, tnf_canonical, kmer4, tetranuc_freq, kmer3 |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (2) | dnabert2 (82% of best), gena_lm (near-random at 1024D) |

*Winner: grover_dna+ridge=0.4207 MCC. Best aug: grover_dna+tnf_canonical=0.4422 (+0.0215).*

**nt_H3K4me2 (32,138 samples, 23 groups, MCC) — 17/25 reps ran. CORRECTED 2026-04-17 (gena_refix 7496796):**

| Tier | Reps |
|------|------|
| Top Tier (7) | mistral_dna, hyenadna, grover_dna, kmer6, nt_v2_500m, nt_v2_250m, kmer5 |
| General (11) | mistral_dna, hyenadna, grover_dna, kmer6, nt_v2_500m, nt_v2_250m, kmer5, tnf_canonical, kmer4, tetranuc_freq, kmer3 |
| Augmentation (5) | codon_usage, complexity, dinuc_freq, dna_shape, gc_content |
| Dropped (2) | dnabert2 (86% of best), gena_lm (near-random at 1024D) |

*Winner: mistral_dna+ridge=0.5054 MCC. Best aug: mistral_dna+tnf_canonical=0.5270 (+0.0216).*

---

## TODO — OPIG DDG (beyond what's in antibody inter-family)

| # | Dataset | Samples | Reps | Augs | Status |
|---|---------|---------|------|------|--------|
| — | ddg_combined | 429 | 34 | 4 | Listed above (#21) in antibody inter-family |
| — | ddg_combined_full | 1,536 | 30 | 8 | Listed above (#22) |
| 75 | graphinity_foldx | 48,900 | 29 | 0 | DROPPED — best ρ=0.052, all R² negative |
| 76 | graphinity_rosetta | ~21K | 29 | 0 | DROPPED — all R² negative, best ρ=0.079 |
| 77 | graphinity_augmented | 49,329 | 28 | 0 | DROPPED — best ρ=0.057, all R² negative |

---

## Summary Counts

| Domain | Total Datasets | DONE | IN PROGRESS | TODO | BLOCKED | DROPPED |
|--------|---------------|------|-------------|------|---------|---------|
| Molecular | 8 | 7 | 0 | 0 | 0 | 1 (adme_solubility) |
| Molecular CLF | 1 | 0 | 0 | 0 | 1 (herg clf, HTC queue) | 0 |
| Ab Inter-Family | 14 | 11 | 0 | 0 | 0 | 3 (hutchinson, sas, ddg_combined_full) |
| Ab DMS/Large | 5 | 5 | 0 | 0 | 0 | 0 |
| AbAgym | 6 | 5 | 0 | 0 | 0 | 1 (abagym_all) |
| Ab CLF | 4 | 2 | 0 | 0 | 2 (multiclass + species, HTC queue) | 0 |
| Ginkgo | 9 | 6 | 0 | 0 | 0 | 3 (aggregation, purity, tm2) |
| TCGA | 3 | 2 | 0 | 0 | 0 | 1 (survival) |
| Reaction | 3 | 0 | 0 | 0 | 2 (buchwald, doyle — CGR re-survey) | 1 (suzuki) |
| DTA | 3 | 0 | 0 | 0 | 3 (davis, kiba, bindingdb_kd — HTC queue) | 0 |
| Nucleotide | 22 | 0 | 0 | 0 | 22 (gena_lm refix 7485363-78 PD + 6 queued surveys) | 0 |
| Graphinity DDG | 3 | 0 | 0 | 0 | 0 | 3 (rosetta, foldx, augmented) |
| **TOTAL** | **81** | **38** | **0** | **0** | **30** | **13** |

*Note: Previous nucleotide DROP decisions (splice_all, H3K9me3) deferred — FM reps may change the picture. herg clf added as separate entry (was #2b).*

---

## Existing BOM Sections to Reconcile

These sections have older-format writeups that were done before this dataset-by-dataset review process. They need to be checked against new data and converted to the new format (base reps + augmentation signal rows):

- **10A. LogD** — has full tier table, needs new-format row
- **10B. Buchwald-Hartwig** — has tiers, needs new-format row
- **10C. Suzuki** — special augmentation experiment, needs new-format row
- **10D. LUAD vs LUSC** — has full table, needs new-format row
- **10E. BRCA PAM50** — has full table, needs new-format row
- **10G. Antibody Inter-Family** — bulk summary of 11 datasets, each needs individual row
- **12A-12F** — locked pipelines with tiers, these inform but don't replace the readiness rows

---

## Action Items

### IMMEDIATE — When F12/F13 results land

1. **Analyze F12/F13 results** — determine winning (allocation × evaluator × importance) triple
2. **Update 26 experiments** that hardcode `allocation_method='greedy'` (list in `EXPERIMENT_BUGS.md`: C08, C09, C10, C11, C16-C19, S11, S13, N01-N08)
3. **Rebuild SPEED_TIERS** in `find_best_hybrid.py` based on F12 results
4. **Clear hybrid caches on server** — `rm -rf ~/.cache/kirby/hybrid_search/ ~/.cache/kirby/create_hybrid_cache/`
5. **Pull code on server** — `git pull` to get all fixes (classification metrics, augmentation passing, mutation reps, etc.)
6. **Run preliminary experiments** on all 40 ready datasets — complexity, similarity, noise

### NEXT — Jobs in HTC queue (2026-04-04)

All resubmitted on HTC long partition with GPU, 256G, 48-96h wall time.

7. **DTA:** davis, kiba, bindingdb_kd. Output: `tests/results/phase5_baselines/dta/`
8. **Nucleotide:** All 22 datasets with full FM reps (fixed: multimolecule, gpn-genomics, mamba-ssm installed). Output: `tests/results/nucleotide_survey/`
9. **Ab CLF:** tz_her2_multiclass + humanness_species. Output: `tests/results/opig/clf/`
10. **herg clf:** TDC binary blocker (~10K). Output: `tests/results/phase5_baselines/molecular/herg.json`
11. **E2E SchNet:** logd, herg_ki, caco2. Output: `tests/results/schnet_e2e/`
12. **As results land** — follow the New Dataset Pipeline checklist: rsync → review → decide tiers → update DATASET_REPS → add to submit lists

### ~~BLOCKED — FT experiments~~ — DONE (2026-04-04)

- ~~Molecular FT fill~~ **DONE** — qm9, adme_hlm, adme_rlm, adme_mdr1 all completed. Results pulled locally.
- ~~Antibody ESM-2 LoRA fill~~ **DONE** — All 29 datasets completed. Results pulled locally.
- **E2E SchNet** — caco2 + herg pulled locally. All 3 resubmitted on HTC (in queue).
- **When FT results are reviewed:** update per-dataset tier tables (swap frozen→FT only where FT wins).

### IMPLEMENTATION — New experiment categories

13. ~~**Noise N13 (alt noise distributions)**~~ **DONE** (2026-03-28) — N09-N13 all implemented. 5 new experiments + `lnl_methods.py` (12 LNL methods). See `NOISE_EXPERIMENT_ROADMAP.md`.
14. ~~**Federated fed04 + fed05**~~ **DONE** — all 5 federated experiments implemented (fed01-fed05). See `FEDERATED_EXPERIMENT_ROADMAP.md`.

### IMPLEMENTATION — Experiment infrastructure

16. ~~**Implement per-dataset file redesign**~~ **DONE** (2026-03-27) — replaced ~100 boilerplate .py files + 6 SLURM scripts with `run_dataset.py` + `submit_dataset.sh` + `submit_all_datasets.sh`. N10-N13 and fed02-fed05 pre-registered in helpers.
17. **S01-S09 classification metrics** — lower priority, but S04/S05/S06 compute R² on predictions which is wrong for classification datasets. See `EXPERIMENT_BUGS.md` Tier 2 similarity.

### EXPERIMENTS — CGR-SMILES pretraining (Phase 4 CGR_EMBEDDING_RESEARCH.md)

**Goal:** Continue-pretrain ChemBERTa-77M-MLM on CGR-SMILES corpus from USPTO (~1M reactions) for 384D reaction embeddings.

**Status:** First run failed (0/392K conversions) — chython was not installed, and `except Exception` silently swallowed the ImportError. Fixed in commit 9fa01f8. chython now installed. **Resubmit required.**

**Dependency chain for reaction pipeline:**
1. CGR-SMILES pretraining job (~8h) → produces checkpoint
2. Reaction re-survey (buchwald_hartwig, bh_doyle_plates) with all CGR reps → new tier tables
3. Precompute hybrids for reaction datasets → pipeline ready

Note: `cgr_morgan` and `chytorch_graphormer` work without the pretraining checkpoint. Only `cgr_smiles_lm` is blocked. Options: (a) re-survey now with 2/3 CGR reps, add cgr_smiles_lm later; (b) wait for pretraining, survey all 3 at once.

**Two-stage SLURM job** (`tests/slurm_scripts/submit_cgr_smiles_pretrain.sh`):
- **Stage 1 — Corpus generation (CPU):** Download USPTO from Figshare (~200MB 7z), extract, convert ~1M atom-mapped reactions to CGR-SMILES via chython compose. Expects 60-80% success rate (~600-800K lines).
  - Partition: **short**, Time: **2h**, Mem: **64G**, CPUs: **16**, GPU: **none**
  - Deps: `py7zr` (for 7z extraction), `chython`
  - Output: `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/corpus.txt`
- **Stage 2 — MLM pretraining (GPU):** Extend tokenizer with 30 dynamic bond tokens, resize embeddings, train 50K steps with HuggingFace Trainer.
  - Partition: **medium**, Time: **6h**, Mem: **128G**, CPUs: **8**
  - Deps: `transformers`, `datasets`, `torch`
  - Output: `/data/stat-cadd/scat9264/kirby_data/cgr_smiles_pretrain/checkpoint/`
- **Combined job:** Partition: **long**, Time: **8h**, Mem: **128G**, CPUs: **16**
- **NOT HTC** — single job, not per-dataset. Run once, checkpoint reused by all reaction datasets.

**After pretraining:**
- `create_cgr_smiles_lm()` auto-detects checkpoint at `/data/.../cgr_smiles_pretrain/checkpoint/`
- Frozen 384D embeddings available in survey + experiments
- Optional LoRA FT variant via `make_cgr_smiles_lora_factory()` (per-fold, ~2h per dataset with GPU)

### EXPERIMENTS — Supplementary runs needed

18. **E2E SchNet on more molecular datasets** — ft_schnet already wired in `make_ft_factories` (line 632-641 of run_baseline_survey.py). Previous HTC runs failed due to torch-cluster bug (now fixed). Resubmitting herg_ki, caco2, logd on HTC with `--finetune`. Output: `tests/results/schnet_e2e/`.
19. ~~**Augmentation per-dataset assignment (usrcat, coulomb_matrix, usr)**~~ **DONE** (2026-03-28) — Results from `molecular_aug_fill/`: caco2 all 3 positive (usrcat +0.021, usr +0.020, coulomb +0.015), adme_mdr1 coulomb_matrix +0.092, logd/qm9 flat or negative. DATASET_REPS already had correct assignments.
20. ~~**DNA foundation model rep implementation**~~ **DONE** — 14 FM reps implemented in `src/kirby/representations/nucleotide.py`. Survey results on server at `tests/results/nucleotide_survey/`.
21. **Nucleotide tier review** — BLOCKED on gena_lm refix jobs 7485363-78 (all PD as of 2026-04-15). All 5 previously-reviewed datasets (H2AFZ, H3K27ac, H3K27me3, H3K36me3, H3K4me1) are downgraded because gena_lm was contaminated at 32000D. Wait for refix jobs + the 6 queued surveys (7483707-12) to land, then review all 22 datasets together.

### OPEN — Adaptive pruning (F15)

**Goal:** After fold 1 on a given dataset, prune search dimensions (strategies, evaluators, importance methods) that are consistently bad, then run folds 2-5 with reduced search space. Saves ~20-50% compute.

**Status:** F15 experiment script written (`tests/experiments/framework/f15_adaptive_pruning_validation.py`) but has 8 known bugs (see below). CANNOT run independently — must run as part of the precompute pipeline (the precompute IS the search, and adaptive pruning reduces that search).

**Preliminary analysis (F12 data, 6 datasets):**
- Strategies are the only prunable dimension — evaluators and importance methods always produce winners
- "Never top-1 on fold 1" rule prunes avg 4.2/13 strategies per dataset with 0 lost winners
- "Top-3 in <10% of groups" rule prunes 2.2/ds with 0 lost
- "Top-3 in <15%" starts losing winners (6/44)
- These results are from F12's limited search space (24 groups/dataset). Full precompute has ~1,170 groups — rules need re-validation

**Blockers:**
1. F15 script has 8 bugs (missing _impute_nan in alloc cache, missing MODEL_MAX_BUDGET caps, bare except, non-deterministic set(), wrong n_folds, unnecessary import, no matched flag, unverified _resolve_metric import)
2. Need per-fold data from full search space to validate fold stability (does fold 1's "bad" set predict folds 2-5?)
3. Rule selection (top-k threshold, percentage threshold, budget filtering) needs data-driven decision
4. Implementation in find_best_hybrid needs design: where does pruning happen, how is it cached, is it default-on or opt-in

**Decision needed:** Whether to integrate pruning into find_best_hybrid BEFORE the precompute (saves compute but adds risk) or run the precompute without pruning first, then validate pruning rules on the results, then re-run with pruning.

### ANALYSIS — After precompute completes

22. **General hybrid vs base rep analysis** — `tests/analysis/general_hybrid_vs_base.py`. Run on server after `precompute_hybrids.py` finishes:
    ```bash
    # All datasets (reads from hybrid search cache, no recomputation)
    python tests/analysis/general_hybrid_vs_base.py
    # By domain
    python tests/analysis/general_hybrid_vs_base.py --domain molecular
    # Export JSON for local analysis
    python tests/analysis/general_hybrid_vs_base.py --export -o general_analysis.json
    ```
    Outputs: `tests/analysis/general_analysis.json` (full per-dataset baselines + hybrids + comparisons) and `tests/analysis/general_analysis_summary.json` (aggregate stats). To analyze locally from exported JSON:
    ```bash
    python tests/analysis/general_hybrid_vs_base.py --analyze general_analysis.json
    ```
    Uses `--speed fast` and `--aug uncapped` by default (must match precompute settings). Direction-aware: positive delta always means hybrid is better.

### Decisions already locked

- **Molecular FT:** lora_molformer, contrastive_molformer, graphmvp_ft → general base reps. Per-dataset: swap frozen→FT only where FT beats frozen. herg_ki keeps frozen molformer. All other FT variants (chemberta, selformer, freeze_head) excluded.
- **Molecular augmentation (confirmed):** pharmacophore3d, maccs → domain-wide. usrcat, coulomb_matrix, usr → dataset-specific: caco2 (all 3), adme_mdr1 (coulomb_matrix only +0.092), herg_ki (all 3 from original survey). logd/qm9 don't benefit.
- **Molecular drops:** adme_solubility (R²=0.32 ceiling), unimol_v2 (dead everywhere).
- **Nucleotide drops:** nt_splice_sites_all (MCC=0.245), nt_H3K9me3 (MCC=0.239).
- **Augmentation defaults:** uncapped, step_size=5, patience=20. Per-experiment overrides in `EXPERIMENT_BUGS.md`.
- **Budgets:** [10,25,50,100,200,500,1000,2000] screening; [10,25,50,100,200,300,500,750,1000,1500,2000] full.
- **Two-stage search:** ABANDONED (F04 showed worse on 9/10 datasets).
- **Stability selection:** ABANDONED (F01/F11).
- ~~**Expand reaction rep set**~~ **DONE** — 14 original + 24 molecular variants = 38 total reaction reps.

---

## Dependency Chain

```
F12/F13 — DONE (2026-04-03)
  ✓ SPEED_TIERS → fast/full (13 allocs, match eval, 2 imp)
  ✓ Defaults: top_3 / lightgbm_gain / svm
  ✓ 30 greedy hardcodes fixed, all stale configs removed
  ✓ MODEL_MAX_BUDGET: GP/KNN≤2000, SVM/MLP≤3000
  ✓ top_k guard, sequential search
  ✓ Caches cleared
  → git push + pull on server
  → PRELIMINARY RUNS on ready datasets (complexity/similarity/noise)

FT fill (HTC, resubmitting 2026-03-28)
  → Molecular FT: qm9, adme_hlm, adme_rlm, adme_mdr1 (pip install fix)
  → Antibody FT: 20 failed datasets (pip install fix), 11 already completed
  → E2E SchNet: herg, caco2, logd (torch-cluster fix)
  → FT swap decisions blocked until these land
  → TODO: Need more FT hybrids for complexity experiments
  → (Does NOT block preliminary runs — FT is enhancement)

Resubmitted surveys (ARC + HTC, 2026-03-28)
  → DTA: davis (370G), kiba (bug fix), bindingdb_kd (ARC)
  → Nucleotide: 22 datasets with FM reps (ARC, default preset fixed)
  → Graphinity DDG: foldx + augmented (ARC, 370G, NaN fix)
  → Ab CLF: tz_her2_multiclass + humanness_species (HTC, 48h 256G)
  → As results land → review → decide tiers → add to DATASET_REPS

Noise extensions (N09-N13 ALL DONE)
  → N09-N13 deploy alongside N01-N08 on same 40 datasets
  → New utility: lnl_methods.py (12 training-loop LNL methods)

Federated experiments (fed01-fed05 IMPLEMENTED)
  → All 5 experiments have run() functions, wired into federated_helpers.py
  → Splits: group-aware via cv_split/GroupShuffleSplit (single 80/20, by design for FLuID comparison)
  → Models: RF hardcoded (deliberate — matches FLuID's published approach)
  → Known issues from FEDERATED_EXPERIMENT_ROADMAP.md:
    - S1: single split (no CI) — acceptable for federated simulation
    - D1: fed01 lacks centralized KIRBy baseline
    - D2: fed03 feature-space confound (FLuID 1 rep vs KIRBy 15+)
    - D3: fed05 no FLuID vs KIRBy head-to-head for antibodies
  → Datasets: herg, caco2, logd
  → See FEDERATED_EXPERIMENT_ROADMAP.md for full design

OPIG standalone experiments (8 scripts — 6 original + 2 CGR)
  → Original 6 deviate from papers' methodology. Audit: OPIG_EXPERIMENT_AUDIT.md
  → Key issues per experiment:
    - Trastuzumab: paper did zero-shot library design, KIRBy does supervised classification (different task)
    - Graphinity: paper uses CDR identity cutoff analysis, KIRBy only groups by PDB (weaker)
    - CDR flexibility: paper trains on 73K general loops, KIRBy only uses tiny CDR test sets
    - Ginkgo: missing Top-10% recall metric (benchmark's secondary metric)
    - Activity cliff: missing half-AC exclusion, MMP partition analysis, MLP/KNN/GIN models
    - Humanness: audit against original repo NOT YET DONE
  → Cross-cutting: all 6 hardcode Ridge+RF instead of using config; all reimplement eval loops
  → Data loaders are CLEAN (verified against original repos)
  → MUST reconstruct before running
  → NEW — CGR reaction experiments (C7, C8):
    - C7 (CGR comparison): `opig_cgr_comparison.py` — CGR vs SMILES head-to-head on 3 reaction datasets. SLURM: `submit_cgr_comparison.sh` (long, 256G, 24h per dataset). WRITTEN, depends on C8.
    - C8 (CGR-SMILES pretrain): `cgr_smiles_pretrain.py` — USPTO corpus + MLM pretraining. SLURM: `submit_cgr_smiles_pretrain.sh` (long, 128G, 8h, single job). SUBMITTED (job 11747734).
    - C6 (King-Smith component splits): `opig_kingsmith_component_splits.py` — now includes cgr_morgan, chytorch_graphormer, cgr_smiles_lm in generator registry. WRITTEN, NOT YET RUN.
```

---

## Reference: Other Documentation

These documents are separate and maintained independently. Referenced here for navigation:

| Document | What it covers | When to consult |
|----------|---------------|-----------------|
| `EXPERIMENT_DESIGN_DISCUSSION.md` | Closed design issues, F12/F13 design, FT results tables, server job history | Historical decisions and methodology rationale |
| `EXPERIMENT_REFACTOR_ROADMAP.md` | Experiment → thesis chapter mapping, 4-step workflow, full inventory | Understanding what each C/S/N experiment tests |
| `EXPERIMENT_BUGS.md` | Tier-based bug tracker, greedy allocation list, hybrid usage audit | Before updating experiments after F12/F13 |
| `FRAMEWORK_EXPERIMENT_ROADMAP.md` | F00-F13 design and results | F12/F13 analysis |
| `NOISE_EXPERIMENT_ROADMAP.md` | N09-N13 design, LNL survey, noise estimation work | Implementing noise extensions |
| `FEDERATED_EXPERIMENT_ROADMAP.md` | fed01-fed05 design, FLuID integration | Implementing federated experiments |
| `OPIG_COLLAB_ROADMAP.md` | 11 OPIG lab publication items | OPIG-specific dataset decisions |
| `OPIG_EXPERIMENT_AUDIT.md` | 6/6 OPIG experiments audited against original GitHub repos | **Before running or modifying any OPIG experiment** |
| `BILL_OF_MATERIALS.md` | Comprehensive rep/dataset inventory, baseline results | Looking up what reps exist and their dimensions |
| `EXPERIMENT_OVERVIEW.md` | Reference for datasets, prediction tasks, representations | Quick lookup |
