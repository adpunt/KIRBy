# Experiment Design Discussion — Open Issues

**Created:** 2026-03-25
**Updated:** 2026-03-25 (v2 — full answers, not just restated questions)
**Context:** First batch of experiment runs revealed bugs (fixed in b06932e) and design questions (below). These need user decisions before re-running.

---

## 1. `load_data` routing — CRITICAL, blocks ginkgo/clf/graphinity

**Problem:** `_shared.py` `_ANTIBODY_PREFIXES` = `('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind', 'ddg_')`. Missing: `'ginkgo_'`, `'tz_her2'`, `'humanness_'`, `'graphinity_'`. These all load correctly through `antibody_baseline_survey.py` but `_shared.py:load_data()` doesn't know they're antibody datasets — falls through to the molecular loader and crashes.

**Affected:** All 6 ginkgo datasets, tz_her2_binary, humanness_therapeutic — for ALL experiment sets (complexity, similarity, noise). 8 of 33 datasets are broken.

**Fix:** Add the missing prefixes to `_ANTIBODY_PREFIXES` (line 883 of `_shared.py`):
```python
_ANTIBODY_PREFIXES = ('flab_', 'abcov_', 'sabdab_', 'abagym_', 'abbind', 'ddg_',
                      'ginkgo_', 'tz_her2', 'humanness_', 'graphinity_')
```

**Additional fix needed:** The antibody branch hardcodes `'task_type': 'regression'` (line 1045). For tz_her2_binary and humanness (classification), this must read the actual task_type from `load_ab_dataset()`. The `antibody_baseline_survey.py:load_dataset()` already returns `task_type` in the data dict — just read it:
```python
task_type = data.get('task_type', 'regression')
```

**Verification:** `load_data('ginkgo_hic', rep_preset='all', rep_names=['ablang2'])` should return an antibody data dict with `domain='antibody'`.

**This is a 2-line fix.** Safe because `antibody_baseline_survey.py` already handles all these datasets correctly.

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

## 4. Budgets — redesign needed (was §3)

**Current:** PRELIMINARY config uses `[25, 50, 100, 200, 500]`. Now capped at 1024 after MLP blowup at 18881.

**User's proposal:** Powers-of-2-ish budgets: `[8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 1024, 2048]`, with model-specific caps. Benefits: finer resolution on the complexity matching curve, direct dimension-matched comparisons (hybrid at ecfp4's 2048D dimension), small budgets (8, 16) test extreme compression. Numpy operations don't benefit from powers-of-2 per se, but the finer grid is valuable.

**Proposed budget tiers:**
```python
PRELIMINARY['budgets'] = [16, 32, 64, 128, 256, 512]           # 6 values, screening
FULL['budgets'] = [8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 2048]  # 11 values
```

Model-specific caps for FULL:
- RF/LightGBM/Ridge: no cap (can handle any dimension)
- MLP: max 1024 (weights explode above this)
- SVM: max 1024 (O(n²d) scaling)
- GP: max 512 (O(n³) already limits dataset size)
- kNN: no cap

**Dimension-matched comparison idea:** For each base rep, add a condition where `budget = rep.shape[1]`. This shows whether hybrid feature SELECTION at the same dimensionality beats the raw rep. This could be a column in C02's output or a dedicated mini-experiment.

**Decision needed:** Confirm the two tiers. Confirm whether dimension-matched comparison is a new experiment or a C02 extension.

---

## 4. Augmented features — NOT being used, should be

**Current state:** `find_best_hybrid` accepts `augmentations` parameter. `DATASET_REPS[ds]['augmentation']` lists augmentation reps per dataset. But `run_all()` in all three helpers ONLY loads `DATASET_REPS[ds]['general']` reps — augmentation reps are defined but never loaded or passed.

**User's preference:** Default should be **hybrid with uncapped augmented features**. Only use no-aug or capped-aug when explicitly testing dimension sensitivity.

**What needs to change:**
1. `run_all()` needs to load augmentation reps alongside general reps
2. Pass augmentation reps to `find_best_hybrid` via the `augmentations` parameter
3. The `augment_uncapped` function in `hybrid.py` adds them after the base hybrid is built

**Augmentation strategy choice:** F06 results say `greedy_ablation` wins molecular, `core+all` wins antibody. For screening, `core+all` (test each aug rep individually, keep those that help) is simpler and safer.

**Implementation plan:**
```python
# In run_all(), after loading data:
aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
if aug_rep_names:
    aug_data = load_data(dataset_name, rep_preset='all', rep_names=aug_rep_names)
    augmentation_reps = aug_data['reps']
else:
    augmentation_reps = None
# Then pass augmentation_reps to experiments that call find_best_hybrid
```

**Complication:** The augmentation reps need to be loaded alongside the base reps — ideally in the same `load_data` call to avoid re-loading the dataset. This means combining `general + augmentation` in the rep_names list, then splitting them apart before passing to `find_best_hybrid`.

**Decision needed:** Confirm uncapped augmented as default. Confirm `core+all` strategy for screening.

---

## 5. MI estimation fails on small datasets (S03, C17)

**Problem:** KSG mutual information estimator returns 0.0 for all reps on n=137-217 antibody datasets. KSG needs n >> d; with PCA reduction to 50 components on 137 samples, there aren't enough data points for the k-nearest-neighbor density estimation to work.

**Recommendation:** Option C — reduce PCA to `min(15, n//10)` components for small datasets. This preserves the MI computation's validity at the cost of some information loss. Also add a warning when n < 300 that MI results may be unreliable.

**This is not a bug — it's a fundamental limitation of nonparametric MI estimation.** The experiments still run; they just produce MI=0 and nan density ratios for small datasets. S09 (cross-dataset summary) should exclude datasets where MI=0 from the MI-related aggregations.

---

## 6. S11 BO fails on small datasets

**Problem:** BO seed = 5% of n. For n=137, seed=6 samples. Greedy allocation fails with 6 samples.

**Recommendation:** Skip S11 on datasets with n < 200. Add a guard at the top of `s11.run()`:
```python
if len(labels) < 200:
    print(f"  Skipping S11: n={len(labels)} < 200 (BO needs larger datasets)")
    return {'error': 'dataset_too_small', 'n_samples': len(labels)}
```

This affects: all flab_*_jain datasets (n≈130-137), abagym_HER2 (n=146), ginkgo_heparin (n=94).

---

## 7. Representative datasets for testing

**Proposed 7 datasets (one per domain type, strongest performers):**

| Domain | Dataset | n | Why this one |
|--------|---------|---|--------------|
| Molecular | herg | 1,412 | Best molecular results, medium size |
| Ab Inter-Family | flab_thermo_tresanco | 672 | Largest antibody, best ρ=0.805 |
| AbAgym DMS | abagym_VEGF | 418 | Strongest DMS signal ρ=0.613 |
| Ab CLF | tz_her2_binary | 5,000 | Largest clf dataset |
| Ginkgo | ginkgo_hic | 242 | Best ginkgo signal ρ=0.604 |
| TCGA | tcga_luad_vs_lusc | 1,017 | Best TCGA AUROC |
| Small Ab | flab_pk_jain_heprt | 130 | Tests small-n edge cases |

**Decision needed:** Confirm these 7. Once confirmed, I'll create a `submit_representative.sh` SLURM script that submits just these 7 for all 3 experiment sets (21 jobs total instead of 99).

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

## 9. Fine-tuning — not in experiments, significant scope expansion

**Current state:** Fine-tuned reps (lora_molformer, contrastive_molformer for molecular; geneformer_ft, scfoundation_ft for TCGA) are documented as domain-level decisions in PIPELINE_READINESS_TODO.md. They require per-fold factory fitting (data-dependent transforms), not pre-computed arrays.

**Why they're not in experiments:** `run_all()` loads reps via `load_data(rep_names=...)` which returns pre-computed arrays cached to .npy files. Factory reps need `rep_factories` parameter in `find_best_hybrid`, which creates fresh reps per fold using training data only (to prevent leakage).

**What it would take to add them:**
1. Modify `run_all()` to accept and pass `rep_factories`
2. Define factory functions per dataset (already exist in `find_best_hybrid_molecular.py` FACTORY_REPS and `find_best_hybrid_tcga.py`)
3. Pass them to `find_best_hybrid` alongside base reps

**Impact:** The experiments currently test frozen reps only. Fine-tuned reps showed significant improvements on some datasets (scfoundation_ft: LUAD AUROC 0.988 vs frozen 0.500). Missing these means the hybrid experiments don't test the full pipeline.

**Decision needed:** This is a significant scope expansion. Recommend deferring to after the initial screening run confirms the basic pipeline works.

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

## 11. Base + augmentation test (without hybrid) — NEW EXPERIMENT NEEDED

**User request:** Test base rep + augmentation reps concatenated WITHOUT hybrid feature selection. Separates "does augmentation help at all?" from "does the hybrid's feature selection on augmented features help?"

**Proposed design (C20 or S14):**
For each (base_rep, aug_rep) pair:
1. **Base alone:** RF on base_rep → R²_base
2. **Base + aug concat:** RF on hstack(base_rep, aug_rep) → R²_concat
3. **Hybrid(base + aug):** create_hybrid({base, aug}) → RF → R²_hybrid

Key comparison: R²_concat - R²_base (does aug help raw?) vs R²_hybrid - R²_base (does aug help via hybrid?)

If R²_concat ≈ R²_base but R²_hybrid > R²_base, then the hybrid's feature selection is what makes augmentation valuable — not the augmentation itself.

**Where it fits:** Complexity chapter (it's about what the hybrid adds to augmentation). New experiment C20.

**Decision needed:** Confirm this design. Confirm placement in complexity.

---

## Priority ordering

| Priority | Issue | Blocks | Fix complexity |
|----------|-------|--------|----------------|
| 1 | `load_data` routing (§1) | 8/33 datasets for complexity/sim/noise | 2 lines |
| 2 | Standalone experiment hybrid conditions (§3) | 13 experiments have no hybrid | ~20-40 lines each |
| 3 | Augmented features (§5) | All experiments missing augmentation | Medium (loader change) |
| 4 | Representative datasets (§8) | Fast iteration | Config only |
| 5 | Budgets (§4) | All experiments use suboptimal grid | Config change |
| 6 | Internal evaluator / greedy allocation (§9) | Greedy is worst strategy | Blocked on F03/F04 |
| 7 | Base + aug test (§12) | New experiment | New file |
| 8 | MI on small datasets (§6) | Cosmetic (MI=0) | Small code change |
| 9 | S11 small-n guard (§7) | Cosmetic (BO skips) | 5 lines |
| 10 | Fine-tuning (§10) | Scope expansion | Significant |
| 11 | Performance allocation (§11) | Blocked on F03/F04 | Config change after unblock |
