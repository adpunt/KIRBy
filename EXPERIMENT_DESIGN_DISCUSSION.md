# Experiment Design Discussion — Open Issues

**Created:** 2026-03-25
**Updated:** 2026-03-25 (v3 — §1/4/5/6 done, §2/3/4-aug covered in other chats)
**Context:** First batch of experiment runs revealed bugs (fixed in b06932e) and design questions (below). These need user decisions before re-running.

---

## 1. `load_data` routing — ~~CRITICAL~~ DONE

**Status:** FIXED. `_shared.py` already had both fixes (prefixes at line 883-884, `task_type` via `data.get()` at line 1067). The stale copy in `complexity_theory_experiments.py` was updated to match (prefixes + task_type).

---

## 2. Ginkgo / OPIG / Forward Compatibility — what's connected and what isn't

**What exists and works:**
- `src/kirby/datasets/ginkgo_dev.py` — clean loader, returns heavy_seqs/light_seqs/labels/fold_assignments
- `antibody_baseline_survey.py` — routes `ginkgo_*` correctly via DATASET_REGISTRY, uses `pre_split` group strategy
- `tests/results/opig/ginkgo/` — all 9 baseline survey JSONs exist (run via `submit_opig_surveys.sh`)
- `tests/results/opig/clf/` — 3 CLF JSONs (humanness_therapeutic, tz_her2_binary, tz_her2_multiclass)
- `tests/results/opig/ddg/` — 2 DDG JSONs

**What's broken:**
- `_shared.py load_data()` can't route ginkgo/clf/graphinity datasets (§1 above)
- The experiment infrastructure (complexity/similarity/noise) therefore can't run on these datasets
- The per-dataset wrapper scripts (e.g., `complexity/ginkgo_hic.py`) exist but crash on load

**What's missing entirely:**
- No "hybrids improve OPIG datasets" result. C02 (complexity grid) WOULD produce this — it compares best hybrid vs best single rep — but it can't load ginkgo data.
- The OPIG baseline survey results (`tests/results/opig/`) are standalone JSONs that no experiment reads. They're informational only.
- C13 (forward compatibility) has stale `_LEGACY_ANTIBODY` rep names (`ablang`, `protbert` instead of `ablang2`, `prot_t5`) — even if ginkgo loaded, the new/legacy split would be wrong.

**Once §1 is fixed:** The existing per-dataset scripts (complexity/similarity/noise) will work for OPIG datasets. That handles point (1) below.

---

## 3. Standalone comparison experiments — hybrid conditions needed

There are **13 standalone experiment scripts** in `tests/experiments/` that each test a specific question (fine-tuning, covariate shift, 3D models, etc.) using individual reps. NONE of them has a hybrid condition. These are separate from the complexity/similarity/noise per-dataset experiments.

OPIG-affiliated datasets are involved in two things:
1. **The complexity/similarity/noise experiments** (handled by per-dataset scripts, needs §1 routing fix)
2. **These standalone experiments** (need a hybrid condition added to each one)

**The modification for each:** Build a KIRBy hybrid (via `find_best_hybrid` or `create_hybrid`), then evaluate it alongside the individual reps using the same analysis the experiment already does. This shows whether the hybrid changes the experiment's conclusion.

| # | Script | What hybrid condition tests |
|---|--------|---------------------------|
| 1 | `antibody_finetune_comparison.py` | Does hybrid of frozen reps match or beat fine-tuned single reps? If yes → fine-tuning is redundant when you have hybrid. |
| 2 | `antibody_region_ablation.py` | Does hybrid across pooling strategies beat the best single pooling? Does the hybrid discover that CDR+FW regions should be combined? |
| 3 | `covariate_shift.py` | Is the hybrid more robust across split types (scaffold/MW/spectral) than any single rep? Smaller performance delta across splits = more robust. |
| 4 | `dta_e2e_comparison.py` | Does KIRBy hybrid of frozen reps close the gap with end-to-end DeepDTA/GraphDTA? |
| 5 | `dta_finetune_comparison.py` | Does hybrid of frozen reps beat LoRA fine-tuned single reps for DTA? |
| 6 | `imrie_transfer_learning.py` | Does hybrid + TL > hybrid alone? Does hybrid + TL > TL + best single rep? Is TL orthogonal to hybridization? |
| 7 | `molecular_finetune_comparison.py` | Does hybrid of frozen reps beat LoRA/contrastive fine-tuned single reps? |
| 8 | `molecular_graph_survey.py` | Does the hybrid's greedy search select graph/structural reps as augmentations? Does automated selection beat manual concatenation? |
| 9 | `qm9_3d_comparison.py` | Can hybrid of frozen reps approach E2E SchNet/DimeNet++? (Probably not for QM9 — but shows the boundary.) |
| 10 | `reaction_rep_comparison.py` | Does hybrid of reaction reps beat the best single reaction rep? Which reps does greedy allocation select? |
| 11 | `structural_geometry_survey.py` | Does KIRBy automated search discover the same geometry augmentations that manual concatenation found useful? |
| 12 | `tcga_foundation_comparison.py` | Does hybrid of frozen genomic FMs beat fine-tuned single FMs? |
| 13 | `wsi_abmil_comparison.py` | Does hybrid across WSI encoders beat the best single encoder? Does it beat ABMIL aggregation? |

**Implementation for each:** ~20-40 lines per script. In the evaluation loop, add a condition that:
1. Calls `create_hybrid` (or `find_best_hybrid`) on the available reps
2. Evaluates the hybrid using the same model/metric/CV as the other conditions
3. Reports it in the same results table

**This is NOT architectural change — it's adding one more row to each experiment's results table.**

---

## 4. Budgets — DONE (dimension-matched → C20, separate)

**Status:** IMPLEMENTED. Log-spaced grids with human-readable anchors (standard in ML literature):
```python
INTERMEDIARY['budgets'] = [10, 25, 50, 100, 200, 500, 1000, 2000]        # 8 values, screening
FULL['budgets'] = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]  # 11 values
```
Ceiling at 2000 covers dimension-matched comparison with ECFP4 (2048D). Low end (10, 25) covers small antibody datasets.

**Dimension-matched comparison:** Will be a new standalone experiment C20 (see §11).

---

## 4. Augmented features — ~~NOT being used~~ DONE

**Status:** IMPLEMENTED (2026-03-25). Based on F06/F08/F11 results:
- Default `augmentation_strategy='uncapped'` (was `greedy_ablation`)
- Default `step_size=5` (was 10), `patience=20` (was 3)
- `run_all()` in complexity/similarity/noise helpers now loads `DATASET_REPS['augmentation']` reps
- `get_hybrid_features_and_weights()` accepts and passes `augmentations` param
- Dimension-sensitive experiments (C05/C06/C07/C09) use `augmentation_strategy='capped'`
- C02 runs both no-aug and uncapped for comparison
- C10 wired with capped augmentation
- New C21 single-rep augmentation experiment
- `all_concat` baseline removed from `find_best_hybrid` and S01
- Stale caches need clearing on server

---

## 5. MI estimation fails on small datasets (S03, C17) — ~~cosmetic~~ DONE

**Status:** FIXED. `estimate_mutual_information_knn` in `_shared.py` now auto-reduces to `min(15, n//10)` PCA components when d exceeds that limit. Returns 0.0 immediately if n < 10 (too few for any estimate). Callers unchanged.

---

## 6. S11 BO fails on small datasets — DONE

**Status:** FIXED. Guard added at `s11_bayesian_optimization.py:487` — returns `{'error': 'dataset_too_small', ...}` when n < 200. Affects flab_*_jain, abagym_HER2, ginkgo_heparin.

---

## 7. Representative datasets for testing — CONFIRMED

**7 datasets confirmed:**

| Domain | Dataset | n | Why this one |
|--------|---------|---|--------------|
| Molecular | herg | 1,412 | Best molecular results, medium size |
| Ab Inter-Family | flab_thermo_tresanco | 672 | Largest antibody, best ρ=0.805 |
| AbAgym DMS | abagym_VEGF | 418 | Strongest DMS signal ρ=0.613 |
| Ab CLF | tz_her2_binary | 5,000 | Largest clf dataset |
| Ginkgo | ginkgo_hic | 242 | Best ginkgo signal ρ=0.604 |
| TCGA | tcga_luad_vs_lusc | 1,017 | Best TCGA AUROC |
| Small Ab | flab_pk_jain_heprt | 130 | Tests small-n edge cases |

`submit_representative.sh` to be created (7 datasets × 3 experiment sets = 21 jobs).

---

## 8. Internal evaluator — wrong, blocked on framework experiments

**Current state:** `create_hybrid(allocation_method='greedy')` uses RF as the internal evaluator (hardcoded in `_evaluate_allocation` and `_create_hybrid_with_allocation` in `hybrid.py`). F07 showed Ridge wins for antibody datasets and is fastest overall. But the fix was never wired in — still RF everywhere.

**Impact:** Every experiment calling `create_hybrid` or `find_best_hybrid` with greedy allocation uses RF internally to score trial allocations. If RF is suboptimal for the dataset domain, the discovered allocation is suboptimal.

**Additionally:** F01 showed that `greedy` allocation was the WORST-performing strategy overall. `performance_weighted` is the fast-tier default in `find_best_hybrid`, but many experiments hardcode `allocation_method='greedy'` in their direct `create_hybrid` calls (C10, C11, C15, C16, C17, C18, S04, S05, S06, S08, S10, S11, S12, N01, N02, N04, N05, N07).

**This is the single biggest systematic issue with the experiments.** But it's blocked on `find_best_hybrid` finalization. When the methodology is finalized:
1. Update the default allocation in `create_hybrid` calls
2. Update the internal evaluator
3. Clear all caches (`~/.cache/kirby/hybrid_search/` and `~/.cache/kirby/create_hybrid_cache/`)
4. Re-run everything

**Decision needed:** Wait for F03/F04 results, then finalize. All current results are screening-level anyway.

---

## 9. Fine-tuning — compute-once static reps + flag toggle

### What the user wants
FT reps are compute-once, save-as-.npy, load-like-frozen. NOT per-fold factories. Optional flag switches frozen→FT where applicable. A complexity experiment compares hybrids with vs without FT reps.

### What currently exists
- **All FT reps are per-fold factories** (`src/kirby/finetuning/`). No pre-computed .npy files exist.
- FT factories are only invoked via `--finetune` CLI args on survey scripts. Not in static rep dicts.
- Cache dir `~/.cache/kirby/finetuned_reps/` is empty.

### Actual results (verified from server JSONs 2026-03-25)

**Molecular** (3 datasets: logd, caco2, herg — R² shown, RF, scaffold-split 5-fold):

| FT Rep | vs Frozen | logd | caco2 | herg | Verdict |
|--------|-----------|------|-------|------|---------|
| molformer_lora | molformer (0.759/0.399/0.499) | **0.779** (+.020) | **0.451** (+.052) | 0.448 (-.051) | **2/3 win** |
| molformer_contrastive | molformer | **0.773** (+.014) | **0.413** (+.014) | 0.494 (-.005) | **2/3 win** |
| graphmvp_ft | graphmvp (0.674/0.370/0.395) | **0.770** (+.096) | **0.395** (+.025) | **0.448** (+.054) | **3/3 win** |
| chembert_lora | chembert (0.664/0.352/0.398) | **0.724** (+.060) | 0.347 (-.005) | 0.335 (-.063) | 1/3 |
| chembert_contrastive | chembert | **0.743** (+.079) | 0.329 (-.023) | 0.380 (-.018) | 1/3 |
| chemberta_lora | chemberta (0.717/0.406/0.416) | 0.702 (-.015) | 0.347 (-.059) | 0.332 (-.084) | **0/3 — WORSE** |
| selformer_lora | selformer (0.687/0.337/0.464) | 0.578 (-.109) | 0.320 (-.017) | 0.352 (-.112) | **0/3 — WORSE** |
| all freeze_head | — | all worse | all worse | all worse | **DEAD** |
| gp_molformer_* | — | NaN | NaN | NaN | **BROKEN** |
| grover_ft | grover (0.841/0.478/0.506) | NaN | NaN | NaN | **BROKEN** |

**TCGA** (2 datasets: LUAD binary AUROC, BRCA 4-class AUROC):

| FT Rep | Frozen | LUAD | BRCA | Verdict |
|--------|--------|------|------|---------|
| scfoundation_ft | **0.500 (DEAD)** | **0.988** | **0.976** | **FT-only, no fallback** |
| geneformer_ft | geneformer 0.975/0.966 | 0.977 (+.002) | 0.967 (+.001) | Marginal |

**Antibody** (6 datasets, R²):

| FT Rep | Best result | Worst result | Verdict |
|--------|------------|-------------|---------|
| esm2_lora | phillips_h3: 0.978 vs 0.960 frozen (+.018) | tresanco: 0.430 vs 0.484 (-.054) | **Mixed** |
| ablang2_ft | expression_koenig: 0.519 vs 0.431 (+.088) | tresanco: NaN | **Mixed** |
| antiberty_ft | phillips_h3: 0.983 vs 0.924 (+.059) | koenig: **-166M R²** | **BROKEN** |

**DTA** (davis, CI):

| FT Rep | Result | Verdict |
|--------|--------|---------|
| All protein LoRA (esm2/esmc/esm3/saprot) | NaN all folds | **BROKEN** |
| lora_chemberta_esm2 | 0.749 vs 0.748 frozen (+.001) | **Negligible** |
| lora_molformer_esm2 | 0.726 vs 0.748 (-.022) | **WORSE** |

### Decisions made (2026-03-25)

**Per-dataset swap rules (FT replaces frozen ONLY where FT wins):**
- lora_molformer replaces molformer on: logd, caco2. Frozen kept on herg.
- contrastive_molformer replaces molformer on: logd, caco2. Frozen kept on herg.
- graphmvp_ft replaces graphmvp on: logd, caco2, herg (all 3).
- scfoundation_ft: FT-only on both TCGA datasets (frozen is dead).
- geneformer_ft: general tier on both TCGA datasets (frozen v2 fallback).
- Pipeline doc updated with per-dataset tier tables.

**Exclude everywhere:**
- All chemberta FT, selformer FT, freeze_head (worse on all datasets)
- gp_molformer, grover_ft (broken/NaN)
- antiberty_ft (catastrophically broken, -166M R²)
- All DTA protein LoRA (broken), DTA drug LoRA (negligible)

### BLOCKED — waiting on server runs

**Molecular FT fill (4 datasets):** qm9, adme_hlm, adme_rlm, adme_mdr1 need lora_molformer + contrastive_molformer + graphmvp_ft results. Script: `submit_molecular_ft_fill.sh`.

**Antibody ESM-2 LoRA fill (29 datasets):** esm2_lora on all active antibody datasets (inter-family + DMS + CLF + Ginkgo). Script: `submit_antibody_ft_fill.sh`. Per-dataset include/exclude decisions pending results.

**After results land:**
1. Update per-dataset tier tables in pipeline doc (swap frozen→FT only where FT wins)
2. Implement compute-once .npy saving + flag toggle in rep dispatch
3. Write complexity experiment comparing hybrid with/without FT reps

---

## 10. Performance allocation — blocked on F03/F04

**Current state:** F01 results (3/7 tasks):
- `performance_weighted` — safest default, consistent
- `stability_selection` — best on large data
- `greedy` / `greedy_feature` — WORST performers, removed from SPEED_TIERS

F03 (step_size × patience) and F04 (two-phase vs full grid) both timed out and were resubmitted. Results pending.

**Key insight:** Most experiments hardcode `allocation_method='greedy'` which F01 showed is the worst. But `find_best_hybrid` uses `performance_weighted` by default in its fast speed tier. So experiments that call `find_best_hybrid` (C02, C05, C06, C07, C08, C13) use the right allocation, while experiments that call `create_hybrid` directly (C10, C11, C15-C19, S04-S06, S08, S10-S13, N01-N08) use the wrong one.

**Decision needed:** Wait for F03/F04, then update all hardcoded `allocation_method='greedy'` to whatever the finalized default is.

---

## 11. Base + augmentation test (without hybrid) — DONE (C20)

**Status:** IMPLEMENTED as `c20_augmentation_value_decomposition.py`. Registered in `complexity_helpers.py` (module map, name map, SINGLE_DATASET_EXPERIMENTS).

Three conditions per (base_rep, aug_rep) pair:
1. Base alone → score
2. Base + aug concat → score
3. Hybrid(base+aug) at budget=base_dim → score

Hybrid budget = base rep's dimensionality (dimension-matched comparison). Models: RF + Ridge. Reads augmentation reps from `DATASET_REPS[ds]['augmentation']`. Skips datasets with no augmentation reps defined.

---

## Priority ordering

| Priority | Issue | Status |
|----------|-------|--------|
| ~~1~~ | ~~`load_data` routing (§1)~~ | **DONE** |
| ~~2~~ | ~~Budgets (§4)~~ | **DONE** |
| ~~3~~ | ~~MI on small datasets (§5)~~ | **DONE** |
| ~~4~~ | ~~S11 small-n guard (§6)~~ | **DONE** |
| 5 | OPIG / FC wiring (§2) | Covered in other chat |
| 6 | Standalone hybrid conditions (§3) | **DONE** (5/13 have hybrids, remaining 8 don't need them) |
| 7 | Augmented features (§4-aug) | **DONE** |
| 8 | Representative datasets (§7) | **CONFIRMED** |
| 9 | Internal evaluator / greedy (§8) | Blocked on F03/F04 |
| 10 | Fine-tuning in experiments (§9) | **PARTIAL** — pipeline doc updated, scripts written, pending server runs |
| 11 | Performance allocation (§10) | Blocked on F03/F04 |
| 12 | C20 base+aug test (§11) | **DONE** |

---

## 12. Server Job Tracker (2026-03-25)

### What's running and what it blocks

#### ARC — Complexity Experiments (cx_*)
**Jobs:** cx_herg, cx_caco2, cx_logd, cx_qm9, cx_adme_hlm, cx_adme_rlm, cx_adme_mdr1 (7 jobs)
**Output:** `tests/results/complexity/{dataset}/` — per-experiment JSONs (c01, c02, c05, etc.)
**Status:** Running (~13h in). No local results yet.
**Blocks:** Nothing directly — these are the research experiments. But they use stale caches (pre-augmentation defaults). **Must clear caches and re-run after framework finalisation.**

```bash
# When done:
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/complexity/" tests/results/complexity/
```

#### ARC — Framework F01/F03/F04 (CRITICAL — blocks §8 and §10)
**Jobs:** f01_* (8 datasets), f03_* (7 datasets), f04_* (8 datasets) = 23 jobs
**Datasets:** logd, herg, buchwald_hartwig, flab_thermo_tresanco, flab_pk_jain_heprt, abagym_EGFR, tz_her2_binary, ginkgo_hic, (some also tcga_luad, tcga_brca)
**Output:** `tests/results/framework/f01_allocation_strategies/`, `f03_greedy_step_size/`, `f04_two_phase_search/`
**Status:** Running (~1-2h in). F01 has 3/7 locally (tcga_luad, tcga_brca, flab_fcrnrt). F03 and F04 have 0 locally (all prior runs timed out).
**Blocks:**
- §8 Internal evaluator — can't finalize Ridge default until F01 confirms allocation ranking
- §10 Performance allocation — can't update hardcoded `allocation_method='greedy'` until F01/F03/F04 show what replaces it
- F02 (adaptive pruning) — reads F01+F03 JSON outputs
- F09 (cross-domain synthesis) — reads all framework outputs
- **ALL thesis experiments at scale** — SPEED_TIERS need finalizing first

```bash
# When done (pull all framework results):
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/framework/" tests/results/framework/
```

#### ARC — Similarity Experiments (already submitted, some completed)
**Jobs:** Submitted via `submit_all_similarity.sh` — 33 datasets
**Output:** `tests/results/similarity/{exp_id}/{dataset}.json`
**Status:** At least flab_pk_jain_fcrnrt COMPLETED (12 experiments in 732s). flab_immunogenicity_marks and flab_polyreact_jain_bvp were running (slurm 11595868/69). **Unknown how many others finished — check server.**
**Blocks:** Nothing directly — similarity experiments are research output. But like complexity, they use pre-augmentation caches.

```bash
# Check what's done:
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/similarity/"

# Pull everything:
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/similarity/" tests/results/similarity/
```

#### HTC — Antibody Rep Fill (ab_all_dms, ab_all_inter)
**Jobs:** 7343797 (ab_all_dms), 7343798 (ab_all_inter)
**Script:** `submit_missing_htc.sh` (antibody sections, `--skip-existing`)
**Output:** `tests/results/phase5_baselines/antibody/` (merges into existing JSONs)
**Status:** Pending (Priority). Filling missing reps for antibody datasets that already have partial results.
**Blocks:** Pipeline readiness — full rep coverage needed for antibody tier decisions.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/phase5_baselines/antibody/" tests/results/phase5_baselines/antibody/
```

#### HTC — Antibody CLF (abclf_hu)
**Job:** 7349453 (humanness_therapeutic)
**Script:** `submit_missing_htc.sh` run_antibody_clf()
**Output:** `tests/results/opig/clf/`
**Status:** Pending. Already have tz_her2_binary, tz_her2_multiclass, humanness_therapeutic locally. Likely filling missing reps.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/opig/clf/" tests/results/opig/clf/
```

#### HTC — Molecular Augmentation Fill (aug_*)
**Jobs:** aug_qm9 (7355877), aug_adme (7355878 = adme_mdr1)
**Script:** `submit_molecular_aug_fill.sh`
**Output:** `tests/results/molecular_aug_fill/`
**Status:** Pending. Fills missing 3D shape reps (usrcat, coulomb_matrix, usr) for qm9 and adme_mdr1.
**Blocks:** Pipeline readiness augmentation decisions for those molecular datasets.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/molecular_aug_fill/" tests/results/molecular_aug_fill/
```

#### HTC — SchNet E2E Generalization (schnet_*)
**Jobs:** schnet_l (logd), schnet_h (herg), schnet_c (caco2) — appears duplicated (6 jobs total: 7355839-43 + 7355870-74)
**Script:** `submit_schnet_e2e.sh`
**Output:** `tests/results/schnet_e2e/`
**Status:** Pending. Tests whether E2E SchNet (top on QM9 R²=0.95) generalises to non-quantum tasks.
**Blocks:** Molecular fine-tuning decisions (is SchNet worth including as FT rep?).

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/schnet_e2e/" tests/results/schnet_e2e/
```

#### HTC — Reaction Survey (rxn_doyl)
**Job:** 7355734 (bh_doyle_plates)
**Script:** `submit_reaction_survey.sh`
**Output:** `tests/results/reaction_survey/`
**Status:** Pending. Doyle HTE plates — new reaction dataset. buchwald_hartwig and suzuki may have already completed (they were in the same batch).
**Blocks:** Pipeline readiness for reaction domain.

```bash
# Check what's done:
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/reaction_survey/"

# Pull:
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/reaction_survey/" tests/results/reaction_survey/
```

#### HTC — Nucleotide Survey (nuc_*)
**Jobs:** 22 jobs (7359437-58) — one per dataset
**Script:** `submit_nucleotide_survey.sh`
**Output:** `tests/results/nucleotide_survey/`
**Status:** Pending. First-ever nucleotide domain run. 22 datasets × 26 reps × 3 models × 3 folds.
**Blocks:** Entire nucleotide domain — can't make any decisions until these land.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/nucleotide_survey/" tests/results/nucleotide_survey/
```

#### HTC — DTA Baseline Surveys (dta_*)
**Jobs:** dta_davi (7331386, RUNNING 16h + 7359597 PENDING), dta_kiba (7359598), dta_bind (7359599)
**Script:** `submit_missing_htc.sh` run_dta()
**Output:** `tests/results/phase5_baselines/dta/`
**Status:** davis running 16h, kiba+bindingdb pending. No local Phase 5 DTA results.
**Blocks:** Pipeline readiness for all 3 DTA datasets. DTA has been "super buggy" — unknown if these will succeed.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/phase5_baselines/dta/" tests/results/phase5_baselines/dta/
```

#### HTC — Graphinity DDG (ddg_grap)
**Jobs:** 7331380-81 (RUNNING ~19h)
**Script:** `submit_missing_htc.sh` run_ddg_synth()
**Output:** `tests/results/opig/ddg/`
**Status:** Running. graphinity_foldx (50K samples) and graphinity_rosetta (21K). First run ever.
**Blocks:** DDG pipeline readiness.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/opig/ddg/" tests/results/opig/ddg/
```

#### HTC — Molecular FT Fill (molft_*, JUST SUBMITTED)
**Jobs:** 7365622-26 (4 jobs)
**Script:** `submit_molecular_ft_fill.sh`
**Output:** `tests/results/molecular_ft_fill/`
**Status:** Just submitted. lora_molformer + contrastive_molformer + graphmvp_ft on qm9, adme_hlm, adme_rlm, adme_mdr1.
**Blocks:** §9 FT decisions for 4 molecular datasets.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/molecular_ft_fill/" tests/results/molecular_ft_fill/
```

#### HTC — Antibody FT Fill (abft_*, JUST SUBMITTED)
**Jobs:** 7365627-55 (29 jobs)
**Script:** `submit_antibody_ft_fill.sh`
**Output:** `tests/results/antibody_ft_fill/`
**Status:** Just submitted. ESM-2 LoRA on all 29 active antibody datasets.
**Blocks:** §9 FT decisions for antibody — per-dataset include/exclude.

```bash
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/antibody_ft_fill/" tests/results/antibody_ft_fill/
```

### Dependency chain

```
LAYER 1 — Baseline surveys (running/pending)
  DTA surveys → DTA pipeline readiness
  Nucleotide surveys → nucleotide pipeline readiness
  Reaction survey (doyle) → reaction pipeline readiness
  Antibody fill surveys → antibody pipeline readiness
  Graphinity DDG → DDG pipeline readiness
  Mol aug fill → molecular augmentation decisions
  SchNet E2E → molecular FT decisions

LAYER 2 — Fine-tuning fill (just submitted)
  Molecular FT fill → §9 FT swap decisions (4 datasets)
  Antibody FT fill → §9 FT swap decisions (29 datasets)

LAYER 3 — Framework experiments (running on ARC)
  F01 + F03 + F04 → finalize SPEED_TIERS, allocation defaults, internal evaluator
  → unblocks §8 (internal evaluator) and §10 (performance allocation)
  → unblocks F02 (pruning) and F09 (synthesis)

LAYER 4 — Research experiments (running on ARC, but use stale caches)
  Complexity (cx_*) → research results BUT need cache clear + re-run after Layer 3
  Similarity → same issue
```

### What's ALREADY DONE and can be pulled now

**Check server for completed jobs from batches where some are still running:**

```bash
# Reaction survey — buchwald_hartwig and suzuki may be done
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/reaction_survey/"

# Similarity — multiple may have finished (fcrnrt confirmed)
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/similarity/"

# Complexity — check if any finished
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/complexity/"

# F01/F03/F04 — check if any new ones landed
ssh scat9264@gateway.arc.ox.ac.uk "ls /data/stat-cadd/scat9264/KIRBy/tests/results/framework/f01_allocation_strategies/ /data/stat-cadd/scat9264/KIRBy/tests/results/framework/f03_greedy_step_size/ /data/stat-cadd/scat9264/KIRBy/tests/results/framework/f04_two_phase_search/"
```

### OPIG experiments — BLOCKED on framework

5 OPIG experiments (humanness, trastuzumab, ginkgo, graphinity, qsar) are ready but blocked until framework (F01/F03/F04) finalises SPEED_TIERS and caches are cleared.

**Output:** `tests/results/comparisons/opig_*.json`

```bash
# When ready to submit (after framework + cache clear):
bash tests/slurm_scripts/submit_opig_experiments.sh          # all 5
bash tests/slurm_scripts/submit_opig_experiments.sh antibody  # humanness + tz + ginkgo
bash tests/slurm_scripts/submit_opig_experiments.sh molecular # graphinity + qsar

# When done:
scp -r "scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/comparisons/opig_*" tests/results/comparisons/
```

### ⚠️ Cache warning

The complexity and similarity experiments currently running on ARC are using **stale hybrid caches** (pre-augmentation-overhaul defaults: step_size=10, patience=3, augmentation_strategy='greedy_ablation'). Their results are still useful as screening but will need re-running after:
1. Framework experiments (F01/F03/F04) finalize SPEED_TIERS
2. Cache cleared: `rm -rf ~/.cache/kirby/hybrid_search/ ~/.cache/kirby/create_hybrid_cache/`
3. Code pulled: `git pull`
