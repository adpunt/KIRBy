# Experiment Audit — 2026-04-02

## Audit Checklist (per experiment)
1. **DOWNSTREAM_MODELS** — uses `DOWNSTREAM_MODELS` (or `DOWNSTREAM_MODELS_NOISE` for noise) from `_shared.py`
2. **BASE REP COMPARISON** — top 3 base reps per dataset ranked by RF CV, reported individually
3. **METRIC DIRECTION** — `is_lower_better` used correctly in all comparisons/rankings
4. **INTERNAL EVALUATORS** — uses `INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']`
5. **HYBRID CREATION** — uses `find_best_hybrid` (not raw `create_hybrid`)
6. **BUDGET CAP** — budget hard-capped at `n_train` per fold
7. **LOGIC SANITY** — imports, logic, dead code

---

## SHARED INFRASTRUCTURE

### `_shared.py`
- Added `INTERNAL_EVALUATORS = ['match', 'rf', 'svm', 'lightgbm', 'mlp']` (done)
- `DOWNSTREAM_MODELS = ['rf', 'knn', 'svm', 'mlp', 'xgboost', 'lightgbm', 'gp']` (exists)
- `DOWNSTREAM_MODELS_NOISE = [..., 'ngboost']` (exists)

### Helper files (complexity_helpers, similarity_helpers, noise_helpers, federated_helpers)
All four helpers share these systemic issues:
- **Models hardcoded as `['rf', 'lightgbm', 'mlp']`** in config defaults — should use DOWNSTREAM_MODELS
- **Internal evaluators default to `SPEED_TIERS['fast']['internal_evaluators'] = ['match']`** — should use INTERNAL_EVALUATORS
- **No n_train budget capping** — should cap budget at n_train per fold
- **federated_helpers** additionally: uses `create_hybrid` directly, hardcodes `internal_evaluator='ridge'`

---

## COMPLEXITY EXPERIMENTS

### C01 — Rep Characterization
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config (but config defaults to helper's hardcoded list) |
| BASE REP | N/A | Profiling only, no comparison |
| METRIC DIR | OK | No comparisons made |
| INT EVAL | N/A | No hybrid search |
| HYBRID | N/A | No hybrid creation |
| BUDGET CAP | N/A | No budget |
| LOGIC | OK | Clean |

### C02 — Complexity Grid
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config |
| BASE REP | **BUG** | Uses `>` to find best_single_rep without checking lower_is_better (line ~188) |
| METRIC DIR | **BUG** | Retrieves lower_is_better but ignores it in baseline comparison |
| INT EVAL | OK | Delegates to find_best_hybrid |
| HYBRID | OK | Uses find_best_hybrid |
| BUDGET CAP | OK | Delegated |
| LOGIC | BUG | Metric direction in baseline selection |

### C03 — Cross-Dataset Transfer
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | MIXED | find_best_hybrid defaults + hardcoded SVR/ElasticNet |
| BASE REP | OK | Delegated to library |
| METRIC DIR | OK | Correctly used |
| INT EVAL | OK | Uses DISTANCE_EVALUATORS |
| HYBRID | **WARN** | create_hybrid direct call (line ~121) for applying blueprint |
| BUDGET CAP | OK | Delegated |
| LOGIC | OK | Minor config hardcoding |

### C04 — Importance as Metric
| Check | Status | Notes |
|-------|--------|-------|
| All | N/A | Importance evaluation only, no models/hybrids |
| BUDGET CAP | WARN | budget=25 hardcoded, no n_train check |

### C05 — Learning Curves
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config |
| BASE REP | **BUG** | `max(best_single, r2)` ignores metric direction (line ~133) |
| METRIC DIR | **BUG** | Retrieved but not used in comparison |
| INT EVAL | OK | Delegates to find_best_hybrid |
| HYBRID | OK | Uses find_best_hybrid |
| BUDGET CAP | **OK** | `sub_budgets = [b for b in budgets if b <= n_sub]` |
| LOGIC | BUG | Metric direction |

### C06 — Within-Family Gradient
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Clean experiment, uses config, correct metric handling |

### C07 — Data Complexity Partitions
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | Defaults |
| BASE REP | **BUG** | `score_mean > best['score_mean']` ignores metric direction |
| METRIC DIR | **MISSING** | Doesn't call get_domain_metric at all |
| INT EVAL | OK | Delegates |
| HYBRID | OK | Uses find_best_hybrid |
| BUDGET CAP | WARN | Only minimum 50-sample check |
| LOGIC | BUG | Missing metric direction |

### C08 — Calibration
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF + Ridge only (line ~130) |
| BASE REP | N/A | |
| METRIC DIR | OK | |
| INT EVAL | OK | Delegates |
| HYBRID | **WARN** | create_hybrid direct call (line ~160) |
| BUDGET CAP | WARN | No n_train check |
| LOGIC | WARN | Config hardcoding |

### C09 — Formal Analysis
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config |
| BASE REP | WARN | Depends on C02 quality |
| METRIC DIR | OK | |
| INT EVAL | OK | Delegates |
| HYBRID | **WARN** | create_hybrid direct calls in 9b/9c |
| BUDGET CAP | WARN | Delegated |
| LOGIC | WARN | TODO comment about API |

### C10 — Noise Robustness
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| BASE REP | N/A | Per-rep NDS |
| METRIC DIR | OK | |
| INT EVAL | OK | Via get_precomputed_search |
| HYBRID | **WARN** | create_hybrid direct call (line ~175) |
| BUDGET CAP | WARN | No check |
| LOGIC | OK | |

### C11 — Ensemble vs Hybrid
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config |
| BASE REP | **BUG** | `r2_mean > best_single_r2` ignores metric direction (line ~143) |
| METRIC DIR | **BUG** | Retrieved but not used in comparisons |
| INT EVAL | OK | Via get_precomputed_search |
| HYBRID | WARN | Uses build_fold_hybrid wrapper |
| BUDGET CAP | **OK** | `min(budget, X_train.shape[1], len(y_train) - 1)` |
| LOGIC | BUG | Metric direction in both baseline + hybrid comparison |

### C12 — Modality Ablation
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | From config |
| BASE REP | WARN | Only per-modality best, no top-3 ranking |
| METRIC DIR | OK | Correctly handled |
| INT EVAL | N/A | Direct training, no hybrid search |
| HYBRID | N/A | Direct concatenation |
| BUDGET CAP | **WARN** | No cap; concatenation can exceed n |
| LOGIC | OK | |

### C13 — Forward Compatibility
| Check | Status | Notes |
|-------|--------|-------|
| All checks | OK | Correct metric direction, find_best_hybrid, delegated |
| BASE REP | WARN | No top-3 ranking |

### C14 — Cross-Domain Summary
| Check | Status | Notes |
|-------|--------|-------|
| All checks | OK | Correct metric direction, find_best_hybrid |
| BASE REP | WARN | Only best single rep, no top-3 |

### C15 — ML Theory Depth
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | Implicit |
| INT EVAL | OK | Delegates via get_fitted_hybrid |
| HYBRID | OK | get_fitted_hybrid |
| BUDGET CAP | WARN | No n_train cap |
| LOGIC | **BUG** | Line ~140: stale model_factory variable when re-training best model |

### C16 — Feature Selection as Regularization
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | Ridge/RF/SVM only |
| METRIC DIR | OK | |
| INT EVAL | OK | Delegates |
| HYBRID | OK | get_precomputed_search |
| BUDGET CAP | OK | Capped at 2048 |
| LOGIC | OK | |

### C17 — Information-Theoretic Decomposition
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| BUDGET CAP | WARN | No n_train cap |

### C18 — Diversity vs Depth
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | |
| BUDGET CAP | WARN | No n_train cap |

### C19 — Importance-Weighted Training
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | Filtered from config (weighted models only) |
| METRIC DIR | WARN | line ~230 doesn't explicitly use lower_is_better in best selection |
| HYBRID | OK | get_fitted_hybrid |

### C21 — Single-Rep Augmentation
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['rf', 'ridge', 'xgboost']` |
| METRIC DIR | OK | Correctly uses sign flip |
| HYBRID | OK | find_best_hybrid |

### C22 — Explainability
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['rf', 'xgboost']` |
| METRIC DIR | OK | |
| HYBRID | OK | get_precomputed_search |
| LOGIC | **BUG** | Line ~184: uses only last fold's feature_info for explanation |

---

## SIMILARITY EXPERIMENTS

### S01 — Similarity Baseline
| Check | Status | Notes |
|-------|--------|-------|
| All | N/A/OK | Distance correlation only, no models/hybrids |

### S02 — Hybrid Similarity
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | knn (correct for distance experiment) |
| BASE REP | OK | Per-rep SAC ranking |
| METRIC DIR | OK | |
| INT EVAL | WARN | Relies on defaults |
| HYBRID | OK | get_fitted_hybrid |
| BUDGET CAP | WARN | budget=100 hardcoded |

### S03 — Feature Concentration
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Analysis only, MI density |

### S04 — Activity Cliffs
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | knn |
| BASE REP | OK | Per-rep cliff metrics |
| METRIC DIR | **BUG** | Line ~293: best-rep selection uses negated lambda incorrectly for lower_is_better |
| INT EVAL | OK | DISTANCE_EVALUATORS |
| HYBRID | OK | get_fitted_hybrid |

### S05 — kNN Prediction
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Correct metric direction, DISTANCE_EVALUATORS, get_fitted_hybrid |

### S06 — Applicability Domain
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Correct use of DISTANCE_EVALUATORS, get_fitted_hybrid |

### S07 — Noise Bridge
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Correct FP rate handling, DISTANCE_EVALUATORS |

### S08 — Optimal Dimension
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | DOWNSTREAM_MODELS imported |
| BASE REP | OK | Top-3 per model |
| METRIC DIR | OK | |
| INT EVAL | OK | DISTANCE_EVALUATORS |
| LOGIC | **WARN** | Line ~212: splitter reused across per-rep loop (potential data leakage) |

### S09 — Cross-Dataset Summary
| Check | Status | Notes |
|-------|--------|-------|
| All | N/A | Aggregation only |
| LOGIC | **BUG** | Line ~89: reads 'conditions' key from S05, should be 'per_budget' |

### S10 — Importance Downstream
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | knn for search, 4 task evaluations |
| METRIC DIR | OK | |
| INT EVAL | WARN | API uncertainty on get_precomputed_search params |
| HYBRID | OK | build_fold_hybrid |

### S12 — Cross-Domain Cliffs
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | DOWNSTREAM_MODELS imported |
| BASE REP | OK | Per-rep per-model |
| METRIC DIR | OK | |
| INT EVAL | **WARN** | No DISTANCE_EVALUATORS (should use for cliff analysis) |
| HYBRID | OK | get_precomputed_search |

### S13 — Clustering & Retrieval
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | DISTANCE_EVALUATORS, get_fitted_hybrid, correct budget cap |
| BUDGET CAP | **OK** | `min(budget, total_dim, len(labels)-1)` |

---

## NOISE EXPERIMENTS

### N01 — Activity Cliff Framework
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | DISTANCE_EVALUATORS |
| BASE REP | OK | Top-3 ranking |
| METRIC DIR | OK | |
| INT EVAL | OK | DISTANCE_EVALUATORS |
| HYBRID | OK | get_fitted_hybrid |
| LOGIC | **BUG** | Lines ~386, ~408: `kwargs` undefined in antibody path |

### N02 — Sample Weighting
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | WARN | Hardcoded WEIGHTED_MODELS (intentional) |
| BASE REP | OK | Top-3 ranking |
| METRIC DIR | OK | |
| HYBRID | OK | build_fold_hybrid |
| LOGIC | **BUG** | Lines ~408-410: dict key 'unweighted' should be 'unweighted_mean' — top-3 returns NaN |

### N03 — SAS Landscape
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Correct, DISTANCE_EVALUATORS, get_fitted_hybrid |

### N04 — GP LOO Detection
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Correct, DISTANCE_EVALUATORS, PCA dim cap |

### N05 — Noise Injection Benchmark
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Detector benchmark, get_fitted_hybrid |

### N06 — BindingDB Ground Truth
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Estimator survey, correct subsampling cap |

### N07 — Remediation
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | gap formula (s-noisy)/(clean-noisy) is direction-agnostic |
| HYBRID | OK | build_fold_hybrid |

### N08 — Estimation Survey
| Check | Status | Notes |
|-------|--------|-------|
| All | OK | Estimator survey, correct |

### N09 — LNL Survey
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | WARN | Hardcoded DNN + LNL methods |
| BASE REP | OK | Top-3 ranking per condition |
| METRIC DIR | OK | Correct direction checking |
| INT EVAL | **WARN** | Single evaluator from search, no iteration |
| BUDGET CAP | WARN | No capping |

### N10 — Hybrid Neighborhood LNL
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | Recovery formula is direction-agnostic (verified) |
| HYBRID | OK | build_fold_hybrid for weighted eval |
| BUDGET CAP | WARN | No capping |

### N11 — Estimation × Remediation Grid
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | Recovery formula is direction-agnostic (verified) |
| HYBRID | **WARN** | create_hybrid directly (line ~131) |
| BUDGET CAP | WARN | No capping |

### N12 — Importance-Weighted Distance
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | RF only |
| METRIC DIR | OK | Recovery formula is direction-agnostic (verified) |
| HYBRID | **WARN** | create_hybrid directly (line ~110) |
| BUDGET CAP | WARN | No capping |

### N13 — Alternative Noise Distributions
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **OK** | Uses DOWNSTREAM_MODELS_NOISE |
| METRIC DIR | OK | |
| HYBRID | OK | Proper feature selection |
| BUDGET CAP | WARN | No capping |

---

## FEDERATED EXPERIMENTS

### Fed01 — Noise Heterogeneity
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | RF (consistent across conditions) |
| BASE REP | **MISSING** | No single-rep baselines |
| METRIC DIR | OK | MCC higher-is-better |
| INT EVAL | **HARDCODED** | `'ridge'` fallback everywhere |
| HYBRID | OK | create_hybrid (appropriate for fixed budget) |
| BUDGET CAP | **WARN** | budget=100 may exceed per-silo n_train |

### Fed02 — Rep Diversity Silos
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | RF |
| BASE REP | OK | Per-silo ranking |
| METRIC DIR | OK | RMSE minimized correctly |
| INT EVAL | **HARDCODED** | `'ridge'` fallback |
| BUDGET CAP | OK | Safe for typical datasets |

### Fed03 — Synthetic Private
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | RF |
| BASE REP | OK | Per-silo baselines |
| METRIC DIR | OK | |
| INT EVAL | **HARDCODED** | `'ridge'` fallback |
| BUDGET CAP | OK | Part C has safe capping |

### Fed04 — Cluster-Aware Importance
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | RF |
| BASE REP | **PARTIAL** | No top-3 rep ranking |
| METRIC DIR | OK | MCC |
| INT EVAL | **HARDCODED** | `'ridge'` fallback |
| BUDGET CAP | **WARN** | budget=100 on clusters with min 20 samples |

### Fed05 — Antibody Federated
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | OK | RF |
| BASE REP | OK | Per-silo baselines |
| METRIC DIR | OK | RMSE |
| INT EVAL | **HARDCODED** | `'ridge'` fallback |
| BUDGET CAP | OK | Safe |

---

## OPIG EXPERIMENTS

### opig_graphinity_ddg
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | Single best only, no top-3 |
| METRIC DIR | WARN | Implicit (Pearson r, higher-is-better) |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |

### opig_cdr_flexibility
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | No top-3 |
| METRIC DIR | OK | |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |
| LOGIC | **BUG** | Line ~200: `cdr_name` undefined → NameError |

### opig_ginkgo_developability
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | No top-3 |
| METRIC DIR | OK | Explicit LOWER_IS_BETTER set |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |

### opig_trastuzumab_enrichment
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | No top-3 |
| METRIC DIR | OK | |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |

### opig_humanness
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | No top-3 |
| METRIC DIR | OK | |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |
| LOGIC | **BUG** | Line ~221: `ds_name` undefined → NameError |

### opig_activity_cliff_qsar
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | **HARDCODED** | `['ridge', 'rf']` |
| BASE REP | WARN | No top-3 |
| METRIC DIR | OK | MAE lower-is-better, MCC higher |
| INT EVAL | **HARDCODED** | RF only |
| HYBRID | OK | find_best_hybrid |

### imrie_transfer_learning
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | N/A | Custom DNN variants (intentional) |
| METRIC DIR | OK | RMSE |
| HYBRID | PARTIAL | find_best_hybrid for base only |

### structural_geometry_survey
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | WARN | Uses antibody_baseline_survey's get_models |
| INT EVAL | **HARDCODED** | RF default |
| HYBRID | OK | find_best_hybrid |

### reaction_rep_comparison
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | WARN | Independent model factory implementation |
| BASE REP | WARN | Best per metric, no top-3 |
| METRIC DIR | OK | |
| HYBRID | OK | find_best_hybrid |

### covariate_shift
| Check | Status | Notes |
|-------|--------|-------|
| DOWNSTREAM | WARN | Uses run_baseline_survey's get_default_models |
| METRIC DIR | OK | Dynamic |
| HYBRID | OK | find_best_hybrid |

---

## FRAMEWORK EXPERIMENTS (DEFERRED — documented only)

### Critical bugs:
- **F08**: `allocate_greedy_forward` not importable — crash at runtime
- **F11**: `_create_hybrid_with_allocation` not importable — crash at runtime
- **F12**: Line ~318 indentation bug (incremental save only runs occasionally); Line ~328 NaN check uses `==` instead of `np.isnan()`
- **F14**: Budgets 500-4000 will massively exceed n_train

### Systemic:
- F00, F06, F07, F08, F10, F11, F13, F14: All hardcode downstream models
- F00-F04, F06-F08, F12-F14: No budget vs n_train capping
- F00-F04, F06, F08, F14: Single internal evaluator only

---

## PRIORITY SUMMARY

### P0 — Runtime crashes
1. `opig_cdr_flexibility.py` line ~200: `cdr_name` undefined
2. `opig_humanness.py` line ~221: `ds_name` undefined
3. (FC deferred) F08, F11: broken imports

### P1 — Wrong results (metric direction bugs)
4. C02, C05, C07, C11: baseline comparison uses `>` without checking lower_is_better — **FIXED**
5. S04 line ~293: best-rep selection direction bug — **FIXED**
6. ~~N07 line ~271: gap computation ignores metric direction~~ FALSE POSITIVE (formula is direction-agnostic)
7. ~~N10, N11, N12: recovery % calculation inverted~~ FALSE POSITIVE (formula is direction-agnostic)
8. N02 lines ~408-410: dict key mismatch causes NaN in top-3 — **FIXED**

### P2 — Systemic design issues (need everywhere)
9. **DOWNSTREAM_MODELS**: ~20 experiments hardcode models — **FIXED in helpers (propagates to ~45 experiments), OPIG (7 files), reaction_rep_comparison. Remaining: N07/N10/N11/N12 use RF only (need loop restructure), C08/C16/C22 intentionally limited**
10. **INTERNAL_EVALUATORS**: **FIXED — added to all 4 helpers, 9 direct find_best_hybrid callers, 7 OPIG files**
11. **TOP-3 BASE REPS**: `extract_top3_base_reps()` utility added to _shared.py — **individual experiment calls still needed**
12. **BUDGET CAP**: **FIXED in find_best_hybrid** (caps at n_train estimate, covers all experiments using it). Federated per-silo caps added.
13. **create_hybrid → find_best_hybrid**: ~10 experiments call create_hybrid directly — documented, some are intentional (federated per-silo)

### P3 — Isolated bugs
14. C15 line ~140: stale model_factory variable
15. C22 line ~184: uses only last fold's feature_info
16. S09 line ~89: wrong key for S05 results
17. S08 line ~212: splitter reuse across reps
18. N01 lines ~386, ~408: undefined `kwargs` in antibody path
19. (FC deferred) F12: indentation + NaN check bugs
