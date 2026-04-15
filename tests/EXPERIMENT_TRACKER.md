# KIRBy Experiment Tracker

Last updated: 2026-04-15

---

## Antibody Classification Shakedown — 2026-04-15 (commit `ec0aae4`)

First experimental runs on the 9 antibody classification datasets that finished precompute. Uses the unified `run_dataset.py` entry point — one SLURM job per (category, dataset) pair, each loads data once and runs a filtered experiment list. Noise category cut entirely for classification; c04/c12/c23 excluded from complexity; s04/s09/s11 excluded from similarity.

**Submitted:** 2026-04-15. 18 jobs (9 datasets × 2 categories).

**Complexity experiments per job:** c01 c02 c05 c06 c07 c08 c09 c10 c11 c13 c15 c16 c17 c18 c19 c21 c22 c24
**Similarity experiments per job:** s01 s02 s03 s05 s06 s07 s08 s10 s12 s13

| Job ID | Category | Dataset |
|--------|----------|---------|
| 11763170 | complexity | abagym_Ang2 |
| 11763171 | similarity | abagym_Ang2 |
| 11763172 | complexity | abagym_EGFR |
| 11763173 | similarity | abagym_EGFR |
| 11763174 | complexity | abagym_HER2 |
| 11763175 | similarity | abagym_HER2 |
| 11763176 | complexity | abagym_VEGF |
| 11763177 | similarity | abagym_VEGF |
| 11763178 | complexity | abcov_kd |
| 11763179 | similarity | abcov_kd |
| 11763180 | complexity | flab_immunogenicity_marks |
| 11763181 | similarity | flab_immunogenicity_marks |
| 11763182 | complexity | flab_polyreact_jain_bvp |
| 11763183 | similarity | flab_polyreact_jain_bvp |
| 11763184 | complexity | flab_polyreact_jain_cic |
| 11763185 | similarity | flab_polyreact_jain_cic |
| 11763186 | complexity | humanness_therapeutic |
| 11763187 | similarity | humanness_therapeutic |
| 11763190 | noise (smoke test — n01 n02 n05) | buchwald_hartwig |

**Precondition:** required commits landed before submission — `0e7775c` (c03/c04/c19), `6b311cf` (c15 Rademacher), `f09baed` (c16/c18 clf support), `c26aa07` (s01/s02 sequence cliffs), `f037628` (CDR cache keying), `9fb7681` (c23 deprioritized). All on `main`.

**Monitoring commands:**
```bash
# Queue state
squeue -u $USER -j 11763170-11763187

# Completed jobs summary
sacct -u $USER -j 11763170-11763187 --format=JobID,JobName%40,State,Elapsed,ExitCode

# Output files
ls -lt tests/results/complexity/*/
ls -lt tests/results/similarity/*/
```

---

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

### Active Jobs (Batch 5 — 2026-02-16, commit 110a3e9)

| Job ID | Mode | Dataset | Reps | Experiments | Notes |
|--------|------|---------|------|-------------|-------|
| **11353056** | full_a | logd | fast+hf | 1,2,4,8,9 | |
| **11353057** | full_b | logd | fast+hf | 5,6,7,10 | |
| **11353060** | full_a | caco2 | fast+hf | 1,2,4,8,9 | |
| **11353061** | full_b | caco2 | fast+hf | 5,6,7,10 | |
| **11353064** | full_a | herg | fast+hf | 1,2,4,8,9 | |
| **11353065** | full_b | herg | fast+hf | 5,6,7,10 | |
| **11353068** | full_a | qm9 | fast+hf | 1,2,4,8,9 | |
| **11353069** | full_b | qm9 | fast+hf | 5,6,7,10 | |
| **11353132** | full_cross | all 4 | fast+hf | 3 | |

### Completed

| Job ID | Mode | Dataset | Reps | Experiments | Notes |
|--------|------|---------|------|-------------|-------|
| 11339363 | quick | logd | fast | 1,2,6,9,10 | JSON at `complexity_theory_fast/` |
| 11337765 | quick | logd | fast | 1,2,6,9,10 | Batch 2, pre-fix (train_r2 NaN) |
| 11341476-79 | full | all 4 | all | 1-10 | **Timed out** — stuck on graph_kernel generation, 0 results saved |
| 11337766-69 | full | all 4 | all | 1-10 | Batch 2, timed out on graph_kernel |

### Key Preliminary Results (quick, logd, fast reps)

- Hybrid R² 0.752 vs best single rep (pdv) 0.730 → +3% improvement
- MLP shows clear model-capacity/rep-complexity interaction
- ANOVA: model 53%, budget 28%, interaction 11% — all significant

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

### Active Jobs (Batch 5)

| Job ID | Mode | Dataset | Reps | Experiments | Notes |
|--------|------|---------|------|-------------|-------|
| **11353058** | full_a | logd | fast+hf | 1,2,4,5,6,8 | |
| **11353059** | full_b | logd | fast+hf | 3,7 | |
| **11353062** | full_a | caco2 | fast+hf | 1,2,4,5,6,8 | |
| **11353063** | full_b | caco2 | fast+hf | 3,7 | |
| **11353066** | full_a | herg | fast+hf | 1,2,4,5,6,8 | |
| **11353067** | full_b | herg | fast+hf | 3,7 | |
| **11353070** | full_a | qm9 | fast+hf | 1,2,4,5,6,8 | |
| **11353131** | full_b | qm9 | fast+hf | 3,7 | |
| (not yet submitted) | full_cross | all 4 | fast+hf | 9 | Submit after full_a completes |

### Completed

| Job ID | Mode | Dataset | Reps | Experiments | Notes |
|--------|------|---------|------|-------------|-------|
| 11339369 | quick | logd | all | 1,2,4,5,8 | JSON at `similarity_all/` |
| 11341480-83 | full | all 4 | all | 1-9 | **Timed out** — stuck on graph_kernel, 0 results |
| 11337770-74 | full | all 4 | fast+hf | 1-9 | Batch 2, timed out on graph_kernel |

### Key Preliminary Results (quick, logd, all reps)

- SAC: hybrid weighted 0.336 vs best individual 0.152 → **+121% improvement**
- kNN R²: hybrid weighted 0.775 vs best individual 0.730 → +6%
- RMSE_cliff: hybrid 0.683 vs best per-rep 0.698 → 2% better (small effect)

---

## Pre-cache (Graph Kernel)

**Script:** `tests/slurm_scripts/run_precache_reps.sh`

Generates and caches all 15 reps (incl graph_kernel) per dataset. After completion, experiment jobs can use `all` reps and load from cache.

| Job ID | Dataset | Status | Notes |
|--------|---------|--------|-------|
| **11353199** | logd | Running | |
| **11353200** | caco2 | Running | |
| **11353201** | herg | Running | |
| **11353202** | qm9 | Running | |

After these complete, resubmit complexity/similarity with `all` reps:
```
sbatch run_complexity_experiments.sh full_a logd all
```

---

## Dimensionality Reduction Experiments

**Script:** `tests/dimensionality_reduction_experiments.py`
**SLURM:** `tests/slurm_scripts/run_reduction_experiments.sh`

| Exp | Name | What it tests | Paper section |
|-----|------|---------------|---------------|
| A | Method comparison | PCA vs UMAP vs t-SNE on hybrid reps | Analysis |
| B | Reduction vs direct hybrid | Reduced-dim hybrid vs direct low-budget hybrid | Complexity paper |
| C | Visualization quality | 2D projection quality metrics | Supplementary |

| Job ID | Dataset | Reps | Experiments | Status | Notes |
|--------|---------|------|-------------|--------|-------|
| **11352486** | logd | fast | A,B,C | Running (batch 4) | |
| **11352487** | logd | all_no_gk | B,C | Running (batch 4) | |

---

## Molecular User Scenarios

**Script:** `tests/test_hybrid_user_scenarios_molecular.py`
**SLURM:** `tests/slurm_scripts/run_molecular_scenarios.sh`

| Job ID | Dataset | Reps | Status | Notes |
|--------|---------|------|--------|-------|
| **11352483** | logd | fast | Running (batch 4) | |

---

## Antibody Complexity Experiments

**Script:** `tests/complexity_theory_experiments.py` (with antibody datasets)
**SLURM:** `tests/slurm_scripts/run_antibody_complexity.sh`

Mutation augmentations (positional 11D, biophysical_delta 6D, lm_score 5D) now always appended as augmentations, not base reps.

| Job ID | Mode | Dataset | Reps | Experiments | Status | Notes |
|--------|------|---------|------|-------------|--------|-------|
| **11353136** | quick | flab_expression_koenig | mutation_fast | 1,2,6 | Running (batch 5) | With augmentation fix |
| **11353137** | quick_delta | flab_expression_koenig | mutation_fast+delta | 1,2,6 | Running (batch 5) | With augmentation fix |
| 11339380 | quick | flab_expression_koenig | mutation_fast | 1,2,6 | Completed | Pre-augmentation-fix |

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
| 7 | Allocation comparison | Greedy vs proportional allocation |
| 8 | Augmented features | find_best_hybrid(augmentations=...) |
| 9 | Delta reps | Base-only vs base+delta comparison |
| 10 | Mutation augmentation validation | Augmentation shapes, improve search |
| 11 | Caching validation | Two runs → same scores, second faster |
| 12 | Multi-dataset comparison | Compare optimal across two datasets |
| 13 | Full pipeline | Delta + augmentations combined |

| Job ID | Dataset | Reps | Flags | Status | Notes |
|--------|---------|------|-------|--------|-------|
| **11353134** | flab_expression_koenig | mutation_fast | (defaults) | Running (batch 5) | With augmentation fix |
| **11353135** | flab_expression_koenig | mutation_full | --delta --augment | Running (batch 5) | With augmentation fix |
| 11352484 | flab_expression_koenig | mutation_fast | --delta | Running (batch 4) | Pre-augmentation-fix |
| 11352485 | flab_expression_koenig | mutation_full | --delta --augment | Running (batch 4) | Pre-augmentation-fix |
| 11339377 | flab_expression_koenig | mutation_fast | — | Completed | 13/13 passed (pre-fix) |

---

## Other Running Jobs (not KIRBy experiments)

| Job ID | Name | Notes |
|--------|------|-------|
| 11352483 | mol_scen | Molecular scenarios (batch 4) |
| 11352486-87 | reductio | Reduction experiments (batch 4) |
| 11352465-68 | dnn/mlp_full | External |
| 11352470-73 | ngboost/val_ngbo | External |
| 11352782-84 | paper_fi | Pending (BeginTime) |

---

## Batch Timeline

| Batch | Date | Commit | Key changes |
|-------|------|--------|-------------|
| 1 | 2026-02-11 | various | Initial full runs; timed out (short partition) |
| 2 | 2026-02-12 | eeedd85 | Long partition; all timed out on graph_kernel |
| 3 | 2026-02-14 | 6b85cdf | Quick tests completed; full timed out on graph_kernel |
| 4 | 2026-02-15 | 6b85cdf | Scenarios + reduction (pre-augmentation-fix) |
| **5** | **2026-02-16** | **110a3e9** | **Split A/B, fast+hf default, precache, checkpoints, all fixes** |

## Code Fixes in Batch 5

| Fix | File |
|-----|------|
| train_r2 NaN (missing seed param) | `complexity_theory_experiments.py` |
| ICC check before ANOVA | `complexity_theory_experiments.py` |
| rf_baseline removed from Exp 5 | `similarity_experiments.py` |
| Mutation reps → augmentations | `find_best_hybrid_antibody.py` |
| Augmentation threading (all scenarios) | `test_hybrid_user_scenarios_antibody.py` |
| Incremental checkpoint saving | Both experiment scripts |
| SLURM: split A/B, fast+hf default, precache | All SLURM scripts |

---

## Next Steps

1. **Wait for batch 5 fast+hf results** — first real full molecular results
2. **Wait for precache** (11353199-202) → then resubmit with `all` reps
3. **Check antibody results** from 11353134-37 (augmentation fix)
4. **Full antibody complexity** once quick passes
5. **Compile results** for papers
