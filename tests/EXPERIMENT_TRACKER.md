# KIRBy Experiment Tracker

Last updated: 2026-02-15

## Papers

| Paper | Focus | Key experiments |
|-------|-------|-----------------|
| **Complexity** | Model complexity x rep complexity optimal pairing; ANOVA variance decomposition | Complexity Exp 1-10, Reduction Exp A-C |
| **Similarity** | Hybrid distance improves SAC, kNN, RMSE_cliff, AD; cross-dataset transfer | Similarity Exp 1-9 |
| **Antibody** (extension) | Same framework applied to antibody mutation data; mutation augmentations | Ab Complexity Exp 1-10, Ab Scenarios 1-13 |

---

## Molecular Complexity Experiments

**Script:** `tests/complexity_theory_experiments.py`
**SLURM:** `tests/slurm_scripts/run_complexity_experiments.sh`

| Exp | Name | What it tests | Paper section |
|-----|------|---------------|---------------|
| 1 | Hybrid vs baselines | Hybrid R² vs best individual rep across 6 models | Core result |
| 2 | Budget sweep | Optimal budget per model; budget-performance curves | Complexity interaction |
| 3 | Cross-dataset transfer | Blueprint from dataset A applied to dataset B | Generalizability |
| 4 | Feature weights stability | Weight consistency across CV folds | Robustness |
| 5 | Learning curves | Performance vs training set size at different budgets | Sample efficiency |
| 6 | Capacity-dimension interaction | Fine-grained budget sweep per model; train R² for bias-variance | Core complexity thesis |
| 7 | Subpopulation analysis | Optimal config depends on data subpopulation complexity | Supplementary |
| 8 | Calibration | Complexity matching → better-calibrated uncertainty | Supplementary |
| 9 | Formal theory (4 sub-analyses) | 9a: ANOVA (now with ICC check), 9b: MLE model ID, 9c: cross-context, 9d: bias-variance | Core statistical analysis |
| 10 | Noise degradation (NDS) | Noise resilience across strategies | Robustness |

### Active/Recent Jobs

| Job ID | Mode | Dataset | Reps | Experiments | Status | Notes |
|--------|------|---------|------|-------------|--------|-------|
| **11341476** | full | logd | fast+hf | 1-10 | Running (batch 3) | Full molecular, 6 models |
| **11341477** | full | caco2 | fast+hf | 1-10 | Running (batch 3) | |
| **11341478** | full | herg | fast+hf | 1-10 | Running (batch 3) | |
| **11341479** | full | qm9 | fast+hf | 1-10 | Running (batch 3) | |
| 11339363 | quick | logd | fast | 1,2,6,9,10 | **Completed** | JSON at `complexity_theory_fast/` |
| 11337765 | quick | logd | fast | 1,2,6,9,10 | Completed (batch 2, pre-fix) | train_r2 NaN bug |
| 11337766-69 | full | logd/caco2/herg/qm9 | fast+hf | 1-10 | Completed (batch 2) | train_r2 NaN, no ICC |

### Key Preliminary Results (quick, logd, fast reps)

- Hybrid R² 0.752 vs best single rep (pdv) 0.730 → +3% improvement
- MLP shows clear model-capacity/rep-complexity interaction
- ANOVA: model 53%, budget 28%, interaction 11% — all significant
- **Needs**: ICC check (added), train_r2 fix (added) → rerun on server

---

## Similarity Experiments

**Script:** `tests/similarity_experiments.py`
**SLURM:** `tests/slurm_scripts/run_similarity_experiments.sh`

| Exp | Name | What it tests | Paper section |
|-----|------|---------------|---------------|
| 1 | Distance matrix comparison | Hybrid vs individual rep distance correlations | Core comparison |
| 2 | SAC improvement | Hybrid distance reduces structure-activity cliff errors | Core result |
| 3 | Nearest neighbor preservation | How well hybrid preserves local structure | Supplementary |
| 4 | RMSE_cliff improvement | MoleculeACE-style cliff prediction | Core result |
| 5 | kNN improvement | Hybrid weighted vs unweighted kNN | Core result (rf_baseline removed) |
| 6 | AD correlation | Hybrid distance for applicability domain | Core result |
| 7 | Noise FP reduction | False positive reduction under noise | Supplementary |
| 8 | Optimal dim ratio | Best hybrid dimensionality relative to input | Analysis |
| 9 | Cross-dataset transfer | Distance metric generalization across datasets | Supplementary |

### Active/Recent Jobs

| Job ID | Mode | Dataset | Reps | Experiments | Status | Notes |
|--------|------|---------|------|-------------|--------|-------|
| **11341480** | full | logd | fast+hf | 1-9 | Running (batch 3) | |
| **11341481** | full | caco2 | fast+hf | 1-9 | Running (batch 3) | |
| **11341482** | full | herg | fast+hf | 1-9 | Running (batch 3) | |
| **11341483** | full | qm9 | fast+hf | 1-9 | Running (batch 3) | |
| 11339369 | quick | logd | all | 1,2,4,5,8 | **Completed** | JSON at `similarity_all/` |
| 11337770-74 | full | logd/caco2/herg/qm9 | fast+hf | 1-9 | Completed (batch 2) | rf_baseline still in Exp 5 |

### Key Preliminary Results (quick, logd, all reps)

- SAC: hybrid weighted 0.336 vs best individual 0.152 → **+121% improvement**
- kNN R²: hybrid weighted 0.775 vs best individual 0.730 → +6%
- RMSE_cliff: hybrid 0.683 vs best per-rep 0.698 → 2% better (small effect)
- **Needs**: rf_baseline removal (done) → batch 3 will have clean results

---

## Dimensionality Reduction Experiments

**Script:** `tests/dimensionality_reduction_experiments.py`
**SLURM:** `tests/slurm_scripts/run_reduction_experiments.sh`

| Exp | Name | What it tests | Paper section |
|-----|------|---------------|---------------|
| A | Method comparison | PCA vs UMAP vs t-SNE on hybrid reps | Analysis |
| B | Reduction vs direct hybrid | Reduced-dim hybrid vs direct low-budget hybrid | Complexity paper |
| C | Visualization quality | 2D projection quality metrics | Supplementary |

### Active/Recent Jobs

| Job ID | Mode | Dataset | Reps | Experiments | Status | Notes |
|--------|------|---------|------|-------------|--------|-------|
| **11352486** | — | logd | fast | A,B,C | Running (batch 4) | |
| **11352487** | — | logd | all_no_gk | B,C | Running (batch 4) | |
| 11337762-63 | — | logd | fast/all_no_gk | A-C / B-C | Completed (batch 2) | Pre-fix |

---

## Molecular User Scenarios

**Script:** `tests/test_hybrid_user_scenarios_molecular.py` (via `find_best_hybrid_molecular.py`)
**SLURM:** `tests/slurm_scripts/run_molecular_scenarios.sh`

Tests: full pipeline (data load → rep gen → hybrid search → blueprint → feature weights)

### Active/Recent Jobs

| Job ID | Mode | Dataset | Reps | Status | Notes |
|--------|------|---------|------|--------|-------|
| **11352483** | — | logd | fast | Running (batch 4) | |

---

## Antibody Complexity Experiments

**Script:** `tests/complexity_theory_experiments.py` (with antibody datasets)
**SLURM:** `tests/slurm_scripts/run_antibody_complexity.sh`

Same 10 experiments as molecular, but with FLAb datasets and mutation_fast/mutation_full reps. Mutation augmentations (positional 11D, biophysical_delta 6D, lm_score 5D) are now always appended as augmentations, not competing as base reps.

### Active/Recent Jobs

| Job ID | Mode | Dataset | Reps | Experiments | Status | Notes |
|--------|------|---------|------|-------------|--------|-------|
| 11339380 | quick | flab_expression_koenig | mutation_fast | 1,2,6 | **Completed** | JSON at `ab_complexity_fast/`, pre-augmentation-fix |
| — | — | — | — | — | **Needs resubmission** | After augmentation architecture fix |

### Key Preliminary Results (quick, flab_expression_koenig, mutation_fast)

- Hybrid R² 0.123 vs best single rep (aac) 0.235 → hybrid WORSE
- All reps except aac/kmer had near-zero MI (bad reps dragging down hybrid)
- **Root cause**: positional/biophysical_delta/lm_score were base reps, not augmentations → now fixed
- **Needs**: resubmit with augmentation fix (mutation-aware features as augmentations)

---

## Antibody User Scenarios

**Script:** `tests/test_hybrid_user_scenarios_antibody.py`
**SLURM:** `tests/slurm_scripts/run_antibody_scenarios.sh`

| Scenario | Name | What it tests |
|----------|------|---------------|
| 1 | Basic hybrid search | End-to-end pipeline on antibody data |
| 2 | Budget sweep | Multiple budgets, monotonic improvement |
| 3 | Blueprint reproduction | Exact score reproduction from blueprint |
| 4 | Feature weights | Non-negative, sum-to-1 per rep |
| 5 | Model comparison | Multiple models, different optimal configs |
| 6 | (reserved) | — |
| 7 | Allocation comparison | Greedy vs proportional allocation |
| 8 | Augmented features | find_best_hybrid(augmentations=...) |
| 9 | Delta reps | Base-only vs base+delta comparison |
| 10 | Mutation augmentation validation | Augmentation shapes, improve search |
| 11 | Caching validation | Two runs → same scores, second faster |
| 12 | Multi-dataset comparison | Compare optimal across two datasets |
| 13 | Full pipeline | Delta + augmentations combined |

### Active/Recent Jobs

| Job ID | Dataset | Reps | Flags | Status | Notes |
|--------|---------|------|-------|--------|-------|
| **11352484** | flab_expression_koenig | mutation_fast | --delta | Running (batch 4) | Pre-augmentation-fix code |
| **11352485** | flab_expression_koenig | mutation_full | --delta --augment | Running (batch 4) | Pre-augmentation-fix code |
| 11339377 | flab_expression_koenig | mutation_fast | — | **Completed** | 13/13 passed (pre-fix) |
| 11337136-38 | sabdab_affinity | fast/fast+lm/all_server | — | Completed (batch 2) | All negative R², too few samples (114) |

### Status Notes

- Batch 4 jobs (11352484-85) were submitted before the augmentation architecture fix
- **Needs resubmission** with current code (mutation features as augmentations, not base reps)
- Default changed from `sabdab_affinity` (114 samples) → `flab_expression_koenig` (4,275 samples)

---

## Graph Kernel Analysis

**Script:** `tests/graph_kernel_analysis.py`
**SLURM:** `tests/slurm_scripts/run_complexity_experiments.sh gk`

Investigates WL histogram feature quality; auto-filters >95% sparse features (~71K → ~290).

| Job ID | Status | Notes |
|--------|--------|-------|
| 11339368 | Running (batch 3) | `--mode gk --datasets logd` |
| 11337764 | Completed (batch 2) | Pre-fix |

---

## Code Fixes Pending Commit (2026-02-15)

| Fix | File | Impact |
|-----|------|--------|
| train_r2 NaN | `complexity_theory_experiments.py` | All results had NaN train_r2; `seed` param missing from `evaluate_cv_with_train` |
| ICC check for ANOVA | `complexity_theory_experiments.py` | Added `_compute_icc_check()` before ANOVA decomposition (Exp 9a) |
| rf_baseline removed | `similarity_experiments.py` | Removed RF-vs-kNN comparison from Exp 5 conditions |
| Antibody rep architecture | `find_best_hybrid_antibody.py` | Moved positional/biophysical_delta/lm_score from REP_PRESETS → `get_mutation_augmentations()` |
| Augmentation threading | `test_hybrid_user_scenarios_antibody.py` | All scenarios now receive and pass mutation augmentations |
| Augmentation in load_data | `complexity_theory_experiments.py` | `load_data()` auto-generates mutation augmentations for mutation datasets |
| SLURM defaults | `run_antibody_scenarios.sh` | Changed from fast/sabdab → mutation_fast/flab_expression_koenig |

---

## Batch Timeline

| Batch | Date | Commit | Jobs | Key changes |
|-------|------|--------|------|-------------|
| 1 | 2026-02-11 | various | 11330937-42 | Initial full runs; timed out (short partition) |
| 2 | 2026-02-12 | eeedd85 | 11337136-74 | Long partition, all experiments; train_r2 NaN, no ICC |
| 3 | 2026-02-14 | 6b85cdf | 11339363-80, 11341476-83 | Quick tests completed; full molecular resubmitted |
| 4 | 2026-02-15 | 6b85cdf | 11352483-87 | Scenarios + reduction; pre-augmentation-fix |
| **5** | **pending** | **pending** | — | **Augmentation fix, ICC, train_r2 fix → resubmit antibody** |

---

## Next Steps (Priority Order)

1. **Commit and push** the 7 pending fixes
2. **Resubmit antibody experiments** with fixed augmentation architecture:
   - `sbatch run_antibody_scenarios.sh` (defaults: mutation_fast, flab_expression_koenig)
   - `sbatch run_antibody_scenarios.sh mutation_full flab_expression_koenig --delta --augment`
   - `sbatch run_antibody_complexity.sh quick`
   - `sbatch run_antibody_complexity.sh quick_delta`
3. **Wait for batch 3 full molecular results** (11341476-83) → analyze for papers
4. **Run full antibody complexity** once quick passes:
   - `sbatch run_antibody_complexity.sh full flab_expression_koenig`
   - `sbatch run_antibody_complexity.sh full flab_binding_koenig`
5. **Compile results** for complexity and similarity papers
