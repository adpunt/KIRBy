# Experiment Design Discussion — Open Issues

**Created:** 2026-03-25
**Context:** First batch of experiment runs revealed bugs (fixed) and design questions (below).

---

## 1. `load_data` routing — CRITICAL

**Problem:** `_shared.py` `load_data()` doesn't route ginkgo datasets correctly. When passed `ginkgo_colloidal`, it falls through to the molecular loader (`load_molecular_dataset`) which fails with "Unknown dataset". The routing logic checks prefixes for antibody (`flab_`, `abcov_`, etc.) and TCGA (`tcga_`) but ginkgo datasets don't match any prefix — they're antibody-domain datasets loaded via `antibody_baseline_survey.py` but their names start with `ginkgo_`.

**Affected:** All 6 ginkgo datasets for ALL experiment sets (complexity, similarity, noise).

**Fix options:**
- A) Add `'ginkgo_'` to `_ANTIBODY_PREFIXES` in `_shared.py` (1-line fix, may have side effects if ginkgo loading differs from standard antibody)
- B) Add explicit ginkgo routing in `load_data`

**Also affected:** The broader MUTATION_REPS gap — `mutation_features`, `esm2_wt_delta`, `antiberty_llr` are still not loadable through `load_data`. Already documented in `complexity_helpers.py` TODO.

**Decision needed:** Is option A safe? Does `load_ab_dataset('ginkgo_colloidal')` in `antibody_baseline_survey.py` work correctly?

---

## 2. Budgets — redesign needed

**Current:** PRELIMINARY config uses `[25, 50, 100, 200, 500]`. C11 was using `total_dim` (up to 18881). Now capped at 1024.

**User's proposal:** Powers-of-2-ish: `[8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 1024, 2048]` with model-specific caps. Also wants to compare base rep at its native dimension vs hybrid at the same dimension (e.g., ecfp4 at 2048D vs hybrid at 2048D).

**Benefits of more granular budgets:**
- Finer resolution on the complexity matching curve
- Direct dimension-matched comparisons (hybrid at ecfp4's dimension)
- Small budgets (8, 16, 32) test extreme compression
- Computational: operations on `budget × budget` matrices are fast when budget is a power of 2

**Concerns:**
- More budgets = more compute. With 12 budgets instead of 5, experiments take ~2.4× longer.
- Some experiments (C02 grid) already sweep budgets × models × allocations × importance — multiplying budget count compounds.
- For screening runs (PRELIMINARY), fewer budgets is fine. Full runs should have the granular set.

**Proposal:** Two budget tiers:
```python
PRELIMINARY['budgets'] = [32, 64, 128, 256, 512]       # 5 budgets, screening
FULL['budgets'] = [16, 32, 64, 128, 256, 512, 768, 1024]  # 8 budgets, publication
```

**Decision needed:** Confirm budget tiers. Also: should experiments that explicitly test "dimension-matched" comparisons (e.g., C17, C18) use a separate budget list that includes native rep dimensions?

---

## 3. Augmented features — not being used

**Problem:** The hybrid search in these experiments uses `find_best_hybrid` or `create_hybrid` with default settings, which typically means greedy allocation with NO augmented features. The user has `augment_capped`, `augment_uncapped`, and `candidate_ranking` functions in `hybrid.py` that add augmented features after the base hybrid is built.

**User's preference:** Default should be hybrid with uncapped augmented features. Only use no-aug or capped-aug when explicitly testing dimension sensitivity.

**Current state:** Neither `run_all()` in the helpers nor the individual experiments pass `augmentations=` to `find_best_hybrid`. The `find_best_hybrid` function accepts `augmentations` and `rep_factories` parameters.

**Fix:** In the helpers, pass the augmentation reps from `DATASET_REPS[ds]['augmentation']` to `find_best_hybrid` via the `augmentations` parameter. Need to check how `find_best_hybrid` handles this.

**Decision needed:**
- Default: augmented uncapped for all experiments?
- Which augmentation strategy? (greedy_ablation vs core+all — f06 results say greedy_ablation wins molecular, core+all wins antibody)
- Should the augmentation reps be loaded alongside general reps? Currently `load_data` only loads `rep_names` from `DATASET_REPS[ds]['general']`, not augmentation reps.

---

## 4. MI estimation fails on small datasets (S03)

**Problem:** KSG mutual information estimator returns 0.0 for all reps on n=137-217 antibody datasets. This is a known limitation: KSG needs n >> d, and with PCA reduction to 50 components on 137 samples, the estimator is unreliable.

**Options:**
- A) Skip S03 on datasets with n < 300 (or warn that MI results are unreliable)
- B) Use a different MI estimator (e.g., binning-based, which is less accurate but doesn't fail)
- C) Reduce PCA to min(20, n//10) components for small datasets
- D) Accept MI=0 and don't use MI density for small datasets

**Decision needed:** Which option? This also affects C17 (information-theoretic decomposition).

---

## 5. S11 BO fails on small datasets

**Problem:** Bayesian optimization with seed=5% gives only 6 samples on n=137 datasets. Greedy allocation fails with 6 samples because the internal evaluator can't fit.

**Options:**
- A) Skip S11 on datasets with n < 200
- B) Increase seed fraction to 20% for small datasets
- C) Use a simpler hybrid construction for the seed set (e.g., fixed allocation instead of greedy)

**Decision needed:** Which option?

---

## 6. Representative datasets for testing

**User asked for:** One per dataset type (not all 33) for faster iteration.

**Proposed representative set (7 datasets):**

| Domain | Dataset | n | Why |
|--------|---------|---|-----|
| Molecular | herg | 1,412 | Best molecular results, medium size |
| Ab Inter-Family | flab_thermo_tresanco | 672 | Largest antibody dataset, best ρ |
| AbAgym DMS | abagym_VEGF | 418 | Strongest DMS signal (ρ=0.613) |
| Ab CLF | tz_her2_binary | 5,000 | Largest clf dataset |
| Ginkgo | ginkgo_hic | 242 | Best ginkgo signal (ρ=0.604) |
| TCGA | tcga_luad_vs_lusc | 1,017 | Best TCGA AUROC |
| Small Ab | flab_pk_jain_heprt | 130 | Tests small-n edge cases |

This covers: molecular, antibody regression, DMS, classification, developability, TCGA, and small-n.

**Decision needed:** Confirm these 7, or adjust.

---

## 7. Internal evaluator

**Question:** Is the internal evaluator being used correctly?

**Current state:** `find_best_hybrid` uses RF as the internal evaluator (hardcoded in `_evaluate_allocation` and `_create_hybrid_with_allocation`). Framework experiment F07 showed Ridge wins for antibody and is fastest overall. But the internal evaluator hasn't been changed yet — still RF everywhere.

**Impact on experiments:** The greedy allocation loop uses RF internally to score trial allocations. If RF is suboptimal for the dataset, the allocation found may be suboptimal. This affects every experiment that calls `find_best_hybrid` or `create_hybrid` with `allocation_method='greedy'`.

**Decision needed:** Wait for framework experiments to finalize, then update. This is part of the broader "find_best_hybrid not finalized" situation.

---

## 8. Fine-tuning

**Question:** What happened to fine-tuning?

**Current state:** Fine-tuned reps (lora_molformer, contrastive_molformer, geneformer_ft, scfoundation_ft) are documented in PIPELINE_READINESS_TODO.md as domain-level decisions. But they are NOT in `DATASET_REPS` because they require per-fold factory fitting, not pre-computed arrays.

**Impact:** The experiments only test frozen reps. Fine-tuned reps are a different modality of representation that the hybrid could incorporate. But factory reps need special handling in `find_best_hybrid` (via `rep_factories` parameter).

**Decision needed:** Include factory reps in experiments? This is a significant change to how data is loaded and would require modifying `run_all()` to pass `rep_factories` alongside `base_reps`.

---

## 9. Performance allocation improvements

**Question:** Do I need more framework experiments to determine performance allocation?

**Current state:** F01 (allocation strategies) has results for 3/7 tasks. `performance_weighted` is the fast-tier default. `stability_selection` best on large data. `greedy`/`greedy_feature` are worst (removed from speed tiers). F03 (step size) and F04 (two-phase) timed out and were resubmitted.

**Decision needed:** Wait for F03/F04 results before changing allocation defaults. The current experiments use `allocation_method='greedy'` with `importance_method='lightgbm_gain'` as hardcoded defaults in many experiments — if the framework experiments show a better default, all hardcoded values need updating.

---

## 10. Base + augmentation test (without hybrid)

**User request:** Test base rep + augmentation reps concatenated, WITHOUT the hybrid feature selection. This would show whether augmentation helps on its own or only through the hybrid.

**Design:** For each base rep, concatenate with each augmentation rep and evaluate directly (no selection). Compare: base alone, base+aug concat, hybrid(base+aug).

**Where it fits:** Could be a new experiment (C20 or S14) or an extension to C11 (which already compares single vs ensemble vs hybrid).

**Decision needed:** New experiment or extension? Which experiment set (complexity or similarity)?

---

## Priority ordering

1. **load_data routing** (CRITICAL — blocks ginkgo + potentially other datasets)
2. **Augmented features** (changes what the hybrid actually does)
3. **Representative dataset list** (enables faster iteration)
4. **Budgets** (affects all experiments but current ones still produce useful data)
5. **Internal evaluator** (blocked on framework experiments)
6. **MI on small datasets** (cosmetic — experiments still run, just MI=0)
7. **S11 small-n guard** (cosmetic — BO skips gracefully)
8. **Fine-tuning** (significant scope expansion, defer)
9. **Base + aug test** (new experiment, low urgency)
10. **Performance allocation** (blocked on F03/F04)
