# MASTER ISSUE LIST — KIRBy pipeline + experiments (2026-06-04)

> ✅ **FIX-FLEET PASS COMPLETE (2026-06-04 overnight).** 12 file-partitioned agents worked EVERY issue below against the real code + audit docs (not memory) with local smoke tests. Disposition + decisions → **`DECISIONS_FOR_USER.md`** (plain English). Full per-agent reports → **`SYNTHESIS_2026_06_04.md`**. Handoff → **`NEXT_SESSION_PROMPT.md`**. Headlines: (1) the CACHE-1 churn fix is UNCOMMITTED — commit `find_best_hybrid.py` first; (2) GP not SVM is the cost driver; (3) "SCREENING forces rebuild" + "cx/sim 2× precompute" were FALSE alarms (estimate shrinks); (4) much was already fixed in committed history (c24 runs, fed C6 import present, OPIG leakage closed); (5) all gate/search-space cost levers are write-ups awaiting sign-off, NONE applied. New bugs caught: n10 silently-empty hybrids, s11 Tanimoto×sklearn crash, graphinity mutation-position corruption (fixed), 8 unregistered OPIG dataset names.

**One place. Everything wrong.** Status: [OPEN] = unaddressed · [PARTIAL] = half-done · [FIXED] = resolved (kept for the record so nothing is "forgotten") · [DESIGN] = structural cost/behaviour, not a bug.

This is the **immediate hand-built list (53 distinct issues)** drawn from the estimates doc (PARTS I–VI), WAVE_STATUS, the code I read this session, and the audit records in memory. The background workflow `whb39y4kb` is reading **all 23 docs in full** and will EXPAND + dedup this with per-doc counts as proof of coverage. IDs are stable.

---

# ════ WORKFLOW SYNTHESIS whb39y4kb (18 docs mined in full, 2026-06-04) — THE AUTHORITATIVE CORPUS ════
**927 raw issues → 358 distinct (603 master rows after multiplicity), 12 categories.** This SUPERSEDES/subsumes the hand-built A–M sections below (those were the down-payment). The estimate-specific findings (PART T time, PART ORD ordering, PART LIN linear-assumptions, and ESTIMATES PARTs VII–X) are NOT in these docs — they're this session's analysis — so the true union is **358 + those ~40 + the 5 un-mined docs**.

### Per-doc counts = PROOF every doc was opened and mined
| doc | issues | doc | issues |
|---|---|---|---|
| EXPERIMENT_DEBUGGING.md | **370** | AUDIT_FEDERATED.md | 35 |
| EXPERIMENT_AUDIT_2026_04_02.md | **110** | EXPERIMENT_BUGS.md | 34 |
| issues_tcga.md | 47 | issues_preprocessing.md | 34 |
| CACHE_FIX_PLAN.md | 41 | issues_benchmarks.md | 34 |
| EXPERIMENT_OVERVIEW.md | 41 | NOISE_AUDIT_STATUS.md | 32 |
| AUDIT_OPIG.md | 27 | AUDIT_SIMILARITY.md | 23 |
| AUDIT_NOISE.md | 22 | AUDIT_COMPLEXITY.md | 21 |
| issues_antibody.md | 16 | issues_molecular.md | 15 |
| issues_dta_reaction.md | 13 | issues_splitting.md | 12 |

### By category (358 distinct; ID ranges from the synthesis)
| category | count | category | count |
|---|---|---|---|
| experiment-design (EXP) | 117 | model (MODEL) | 67 |
| other (OTHER) | 60 | representation-or-data (REP) | 57 |
| preprocessing (PREP) | 53 | dead-or-degenerate (DEAD) | 53 |
| wrong-metric (METRIC) | 52 | split-validity (SPLIT) | 36 |
| cache (CACHE) | 32 | data-leakage (LEAK) | 30 |
| cost-runtime (COST) | 28 | missing-feature (MISS) | 24 |

### By status (of 603 master rows): ~470 open · ~95 fixed · ~15 partial · ~25 unknown.

### The 6 highest-severity OPEN clusters (most cross-doc-reported)
1. **SPLIT-1/2/3/19** — silent random/identity/single-fold splits (in 4–8 docs each); violate the non-negotiable no-random-splits rule; invalidate swaths of antibody-DMS, reaction, federated results.
2. **METRIC-1…5** — wrong primary metric per domain (R² where field uses ρ/CI/Pearson/AUROC) → non-comparable to all literature.
3. **CACHE-1/2/3/24, EXP-102** — cache-version orphaning + aug-strategy collisions + unverified cache-health → 240 wasted jobs + a band of suspect results needing re-run.
4. **MODEL-16…20, EXP-94** — forbidden Ridge as federated internal evaluator + 'match'→ridge/equal-budget degradation → the headline KIRBy-vs-FLuID quality-weighting claims are structurally invalid.
5. **LEAK-9/10, MODEL-30** — blueprint-on-clean-labels / FittedResult-on-full-labels + matched-evaluator violations inflate hybrid scores across OPIG + noise.
6. **REP-1/2/3** — invalid-molecule row misalignment in MACCS/PDV/mol2vec silently corrupts every result using those reps.

### ⚠️ GAPS (coverage is 18/23 docs + a storage truncation)
- **5 docs FAILED extraction** (agents didn't return structured output): **ISSUES.md** (the master — 26 confirmed issues + 38 directives), **issues_forward_compat.md** (17 models), **PRECOMPUTE_EXPERIMENT_BUGS_2026_04_17.md** (runtime-heavy), **COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md**, **COMPLEXITY_SIMILARITY_RESULTS_2026_06_02.md**. → NOT in the 358; need a targeted re-extract.
- **Stored table truncated to its tail**: the per-row detail for the first 9 categories (LEAK/SPLIT/METRIC/PREP/MODEL/REP/CACHE/COST/EXP) was cut; their **counts + ID ranges survive** (above), and the DEAD-43→54 / MISS / OTHER rows are captured verbatim below. Full per-row detail recoverable from the workflow transcript or a re-run.

### Captured detail — MISSING-FEATURE (24) and OTHER (63), verbatim from the synthesis
> (These extend the hand-built list with genuinely new content — esp. TCGA missing modalities and the EXPERIMENT_DEBUGGING per-experiment c0x/s/n/fed bugs. DEAD-43→54 also captured; DEAD-1→42 truncated.)

**MISS-1** somatic mutation feats (mut_pca50/TMB/mut_pathway/mut_driver/network) absent · **MISS-2** COSMIC signatures (SBS/DBS/ID) absent · **MISS-3** immune deconv (CIBERSORTx/xCell/MCP/quanTIseq/EPIC/TIMER) absent · **MISS-4** stemness indices (mRNAsi/mDNAsi) · **MISS-5** purity (ABSOLUTE/CPE) · **MISS-6** heterogeneity (MATH/DEPTH) · **MISS-7** WSI models (Virchow/UNI/CONCH/TITAN) · **MISS-8** splicing PSI/lncRNA/fusion · **MISS-9** HRD/MSI/CIMP · **MISS-10** joint FMs (MOJO/BMFM-RNA/CancerFoundation) · **MISS-11** CPTAC proteomics · **MISS-12** VAE/survival-aware latents · **MISS-13** consensus-subtype (MOVICS) · **MISS-14** DTA large benchmarks (BindingDB_Kd/Metz) · **MISS-15** reaction reps (rxnfp/onehot) · **MISS-16** TCGA multi-omics fusion absent · **MISS-17** molecular cliffs unflagged (no SALI) · **MISS-18** no inverse-folding Ab models · **MISS-19** n13 partial-noise corrupt_idx unsurfaced [fixed] · **MISS-20** fed04 pruning_mode not threaded · **MISS-21** nucleotide excluded from precompute cache fix · **MISS-22** finetune_antibody_lm() claimed but absent from src/ · **MISS-23** entire fine-tuning capability (OVERVIEW §4) aspirational/unused · **MISS-24** c01 metric direction absent from JSON.

**OTHER (federated/complexity/sim/noise/opig per-file)** — OTHER-1 stale stat-cadd path · OTHER-2 builtins.display clobbered · OTHER-3 FLuID-unavailable deep crash · OTHER-4/5/6 fed05/fed02 zip/offset/sampling bugs · OTHER-7/8 fed02 3× aug recompute + doc mismatch · OTHER-9/10 fed03 total-failure crash + fragile assumptions · OTHER-11/12 fed04/fed01 seed/dead-import/f1-crash · OTHER-13 c14 hybrid_config empty keys · OTHER-14 c19 best_overall name / c07 search-in-try · OTHER-15/16 c12 reimpl + complexity_helpers stale docstring · OTHER-17 c01 --reps ignored / MLE=0.0 dropped · OTHER-18 c02 KeyError + heatmap/PCA divergence · OTHER-19 c03 negative interaction % · OTHER-20 c05 wrong top3 source · OTHER-21 c07 double-parse/seed/NaN · OTHER-22 c08 fold-drop + KeyError crash · OTHER-23 c10 Nogueira SE NaN at M=2 / c24 stability norm · OTHER-24 c11 stale n_comp · OTHER-25 c14 dead output_dir · OTHER-27 c16 mislabelled key · OTHER-28 c18 mislabelled n_features · OTHER-29 c22 misalign/shape/task_type · OTHER-30 c23 misleading comment · OTHER-31 c24 aug={} w/ none · OTHER-32 s08/s13 var shadowing · OTHER-33 s01/s08 unclamped importance NaN risk · OTHER-35 sim cliff-sweep silent errors · OTHER-36 n02 mislabeled top3 · OTHER-37 n08 format crash · OTHER-38 n09/n13 private import · OTHER-39/40 n04/n05 dead/redundant · OTHER-41 n10 mislabelled y_noisy · OTHER-42 n03 KeyError · OTHER-43 n01 KeyError/dead var · OTHER-44 opig_trastuzumab --reps unvalidated · OTHER-45 opig bare-except ×7 · OTHER-46 opig_cgr suffix spaces · OTHER-47–54 opig/imrie cosmetics · OTHER-55 n13 methodology cluster · OTHER-56 s11 refit desync · OTHER-57/58 n04/n06 envelope + stale docstring · OTHER-59 CACHE_FIX_PLAN transcripts omitted · OTHER-60 QM9 size doc(2000) vs code(5000) · OTHER-61/62 OVERVIEW preset-count inconsistencies · OTHER-63 (xref).

**DEAD-43→54** (extends hand-built DEAD): finetune_gnn unused baseline · old SearchResult knn/empty errors · s10 hybrid arm silently skipped · s12 top3 NameError · s07 hybrid failure swallowed · s11 degenerate random curves · N04/N06 raw-dict envelope · n12 empty JSON [fixed] · n13 MLP fake remediation [fixed] · n01 zero-cliffs [fixed] · fed 'herg' dead branch · fed DOWNSTREAM_MODELS unused.

---

## A. Critical pipeline bugs (Phase 0, ISSUES.md)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| CRIT-1 | Row misalignment | reps and labels indexed inconsistently → rows mismatched | preprocessing | bug | [FIXED] |
| CRIT-2 | Random KFold fallback | when groups missing, fell back to random splits — violates no-random-splits | splitting | bug | [FIXED] |
| CRIT-3 | ESOL/MoleculeNet present | dataset the group forbids was wired in | datasets | data-issue | [FIXED] |
| CRIT-4 | SNS rep not wired | drug_sns absent from surveys/hybrid search | survey | missing | [FIXED] |
| CRIT-5 | TCGA dead code path | unreachable/incorrect TCGA branch | tcga | bug | [FIXED] |

## B. Evaluation / methodology (Phase 1)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| EVAL-1 | Wrong primary metric | R²-only / wrong per-domain metric (should be RMSE/CI/AUROC/F1w/ρ per domain) | run_baseline_survey | methodology | [FIXED] |
| EVAL-2 | gene_immune leakage | factory rep fit across folds → train/test leakage | tcga factory | data-leakage | [FIXED] |
| EVAL-3 | Inflated single 80/20 split | single split instead of 3–5 fold grouped CV | splitting | methodology | [PARTIAL] |
| EVAL-4 | Weak downstream models | only RF/XGB; no GP/MLP/CatBoost/DNN/BNN | models | design-gap | [FIXED] |
| EVAL-5 | InChIKey dedup missing | duplicate molecules leak across scaffold folds | preprocessing | data-leakage | [FIXED] |
| EVAL-6 | No scaffold split (molecular) | non-scaffold splitting inflated molecular scores | splitting | methodology | [FIXED] |
| EVAL-7 | best_overall metric | un-matched evaluator selection (removed; matched evaluators enforced) | find_best_hybrid | methodology | [FIXED] |

## C. Cache (the high-impact ones)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| CACHE-1 | Neural-rep hash drift | 6 torch reps (chembert/chemberta/graphmvp/grover/molformer/smited) are non-bit-reproducible GPU forwards → hash mismatch → cache rejected → recompute churn | find_best_hybrid.py:244 | bug | [PARTIAL] (fix staged, UNTESTED) |
| CACHE-2 | aug_strategy default collision | default 'uncapped' collides with real uncapped cache files when augmentations=None | find_best_hybrid | bug | [FIXED] (a13e0f4) |
| CACHE-3 | Cache-key split FULL vs SCREENING | precompute_hybrids uses FULL_BUDGETS, helper path uses SCREENING → two cache families for "same" condition | precompute_hybrids.py:290 vs helpers | bug | [OPEN] |
| CACHE-4 | Molecular DONE re-running | the 4 adme/caco2 "DONE-on-disk" datasets re-search from scratch via CACHE-1 → ~376h wasted (73h cx + 303h sim) | running wave | bug | [OPEN] (until CACHE-1 deployed) |
| CACHE-5 | c09 fold mismatch | c09 uses 3-fold create_hybrid but precompute is 5-fold → guaranteed cache MISS → 120 full recomputes | c09 | bug | [OPEN] |

## D. Cost / runtime drivers (PART VI — connect estimates → cuts/bugs/gates)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| COST-1 | c17 MI uses all n | KSG mutual-info computed over all n samples; converges by ~2000 → c17 ∝ n needlessly (THE large-n wall-buster) | c17:_greedy_mrmr | gate | [OPEN] |
| COST-2 | c17 mRMR re-run per budget | greedy mRMR recomputed from scratch for each of 8 budgets though it's prefix-monotone (compute once, slice) | c17:244 | bug | [OPEN] |
| COST-3 | c17 prefilter=2000 too high | relevance/redundancy MI passes over 2000 features; 500 barely changes result | c17:220 | cut | [OPEN] |
| COST-4 | c05 n_repeats=3 | 15 fresh searches (5 sizes × 3 seeds); seeds are just error bars → ÷3 possible | c05 | cut | [OPEN] |
| COST-5 | c05 full-size can't reuse cache | 1.0n fraction re-runs full uncapped search 3× under a unique key instead of reusing main precompute | c05:190 | bug | [OPEN] |
| COST-6 | SVM O(n²) uncapped | SVM dominates uncapped precompute (herg ~302h); _max_samples=10000 too high | find_best_hybrid.py:1184 | gate | [OPEN] |
| COST-7 | capped 50 A/B refits | capped augmentation refits the model up to max_candidates=50 per config + ranks over ~18000-D aug pool | hybrid.py:1979 | cut | [OPEN] |
| COST-8 | c23 extra-tier models | c23 runs tier='extra' (catboost/dnn/cnn) → wall explosion (HER2 +49857%) | c23 | gate | [OPEN] |
| COST-9 | c21 giant aug-pool dims | fresh search over esm2_per_region 17920D / esm2_per_cdr 7680D aug pool | c21 | gate | [OPEN] |
| COST-10 | Node contention 3–48× | oversubscribed SLURM nodes make identical code run 3–48× slower (HER2 48×); destroys estimate validity | cluster | methodology | [OPEN] |
| COST-11 | SCREENING vs FULL budgets | experiments run SCREENING_BUDGETS(4) not intended FULL_BUDGETS(8) → biased best-budget; fixing ADDS ~20–40% wall | complexity_helpers.py:1046, similarity_helpers.py:647 | bug | [OPEN] |
| COST-12 | Distance-lineage precompute uncounted | sim needs general_distance+top_distance (6 budgets×6 evaluators) never measured/counted | precompute_hybrids.py:295 | design | [PARTIAL] (estimated V.3.1) |

## E. Complexity experiments — per-experiment bugs/design
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| CX-1 | c08 NameError | `y_pred_conf` undefined in regression branch → c08 crashed on EVERY regression dataset | c08:387 | bug | [FIXED] (this session) |
| CX-2 | c08 condition collapse | optimal/underfit/overfit budgets can coincide → degenerate comparison | c08:199 | design-gap | [PARTIAL] |
| CX-3 | c02 design gaps | missing capped/uncapped/top-tier conditions, extended budgets, FT stub | c02 | design-gap | [FIXED] |
| CX-4 | c02 uncached FT recompute | general_ft fine-tune pass not cached → recompute every run | c02 | bug | [OPEN] |
| CX-5 | c10 degenerate | <5s, no measurable signal (residuals meaningless) | c10 | dead-or-degenerate | [OPEN] |
| CX-6 | c24 conditional skip / cache anomaly | early-return <2 folds; cold-vs-warm cache makes wall unpredictable | c24:241 | design | [OPEN] |
| CX-7 | c16 blueprint-identity dependence | cost driven by cached blueprint identity → must read SearchResult to estimate | c16 | design | [OPEN] |
| CX-8 | c12 quadratic in modalities | TCGA modality ablation is C(G,2) pairwise → quadratic in active modalities | c12 | design | [OPEN] |
| CX-9 | c01 rep-characterization cost | cost ∝ rep dims; contention-contaminated (HER2 44× vs bvp) | c01 | design | [OPEN] |
| CX-10 | c07 cache-miss subpopulations | which subpops trigger fresh searches is data-dependent → +500–670% swings | c07 | design | [OPEN] |
| CX-11 | c13 new-rep dependence | cost = #reps not in LEGACY_REPS → varies wildly by dataset | c13 | design | [OPEN] |
| CX-12 | c15 erratic | +956% VEGF swing, no stable driver | c15 | design | [OPEN] |
| CX-13 | Mutation reps not loaded | load_data missed MUTATION_REPS (needed two-pass) | load_data | bug | [FIXED] |
| CX-14 | Classification metrics wrong | C02/C08/C09/C10/C11 used regression metrics on classification | multiple | bug | [FIXED] |
| CX-15 | Augmentation not passed | C11/C16–C19 didn't forward augmentations | multiple | bug | [FIXED] |

## F. Similarity experiments
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| SIM-1 | s11 inconsistent estimate | domain-skip (molecular/reaction only, n≥200) misread as inconsistency | s11:758 | methodology | [FIXED] (resolved V.3.2) |
| SIM-2 | s01 OOM | n×n distance matrices ∝ n²·dim → ~35GB RAM/rep at n≥30k → OOM risk | s01 | cost-runtime | [OPEN] |
| SIM-3 | s03 O(n²) kNN-MI | brute-force kNN mutual-info → ~111h at n=30k (biggest sim cost) | s03 | cost-runtime | [OPEN] |
| SIM-4 | s05 misattribution | shared distance-lineage precompute mis-charged into s05's wall | s05 | methodology | [PARTIAL] |
| SIM-5 | s04/s07/s12 degenerate | ~0 cost, little/no signal | s04/s07/s12 | dead-or-degenerate | [OPEN] |
| SIM-6 | Augmentation not passed (sim) | all similarity experiments didn't forward augmentations | all sim | bug | [FIXED] |

## G. Noise experiments (NOISE_AUDIT_STATUS / project_noise_audit)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| NOISE-1 | cliff-threshold bug | activity-cliff threshold computed wrong | n01 | bug | [FIXED] |
| NOISE-2 | is_clf misdetection | classification/regression task-type detected wrong | n0x | bug | [FIXED] |
| NOISE-3 | scale-collapse | feature/label scaling collapsed signal | noise | bug | [FIXED] |
| NOISE-4 | no-op experiment | experiment effectively did nothing | noise | bug | [FIXED] |
| NOISE-5 | MLP-fallback | silent fallback to MLP masking failures | noise | bug | [FIXED] |
| NOISE-6 | stale-library JSONs | results JSONs produced by old code → need ARC rerun | noise results | data-issue | [OPEN] |
| NOISE-7 | N01/N07 classification metrics | wrong metrics on classification noise tasks | n01/n07 | bug | [FIXED] |

## H. Estimate-methodology issues (this effort's own audit, PART II.6 / III)
| ID | Title | Mechanism | Location | Type | Status |
|---|---|---|---|---|---|
| EST-1 | Proportional n-scaling | every prior estimate scaled cumulative wall by (n/anchor)^k → 2–10× wrong | estimates | methodology | [FIXED] (banned) |
| EST-2 | Group-range guesses | "molecular ~100–250h" with no derivation | estimates | methodology | [FIXED] (superseded) |
| EST-3 | Premature "not estimable" | refused to estimate instead of deriving from measured anchors | estimates IV.0 | methodology | [FIXED] (committed table now) |
| EST-4 | c08-fix adds wall | post-fix c08 runs on regression → measured 35–64h is a slight under-estimate | c08 | design | [OPEN] (decision) |
| EST-5 | Rep dimensionality not recorded | total_dim per dataset not stored anywhere → blocks tight c17/c05/c21 | DATASET_REPS | missing | [OPEN] (item 1.1) |
| EST-6 | Wedge dead/alive unconfirmed | dead-job calls are "almost certain," no log tail confirmation | monitor | methodology | [OPEN] (item 1.2) |

---

---

# ════ AUDIT-SOURCED ISSUES (read in full 2026-06-04: AUDIT_FEDERATED/OPIG/COMPLEXITY/SIMILARITY/NOISE, all dated 2026-05-23) ════

Status default [OPEN as of 05-23 audit] unless memory/this-session records a fix. Pri = P0 broken / P1 recoverable / P2 smell.

## I. Federated experiments (AUDIT_FEDERATED.md — 35)
| ID | Issue | Location | Pri |
|---|---|---|---|
| FED-1 | `create_hybrid` referenced but never imported → NameError → fed02 C6 ("strongest publication angle") has NEVER produced a real number | federated_helpers.py:1436 | P0 |
| FED-2 | augmentations dropped from every create_hybrid/federated_feature_selection call (passed only to blueprint search) → federated hybrids run on smaller feature space than rest of KIRBy | fed01/02/03/04/05 (18 sites) | P0 |
| FED-3 | Ridge (forbidden) used as silo internal evaluator driving budget allocation → every "noisier silo gets less budget" claim is a Ridge artefact | helpers:746,785,1418,1448; fed03:675 | P0 |
| FED-4 | Single 80/20 split everywhere (ISSUE S1 not actually closed; n_splits=1) → every metric one-fold, no CI | fed01-05 (many) | P0 |
| FED-5 | Inner silo validation split not group-aware (ShuffleSplit) → scaffold neighbours both sides → inflated silo scores → skewed budget | helpers:782,1446; fed03:691 | P0 |
| FED-6 | `is_classification = len(unique)<=10` heuristic → regression datasets with few values misrouted to RidgeClassifier | helpers:749 | P0 |
| FED-7 | compute_clf_metrics silently NaNs on multiclass (1D proba to roc_auc_score) | helpers:1004 | P0 |
| FED-8 | Stale `/data/stat-cadd/` server path (moved to stat-ecr) | helpers:49, lhasa_herg_loader:20 | P1 |
| FED-9 | builtins.display permanently overwritten with no-op on non-IPython | helpers:44 | P1 |
| FED-10 | fed03 Part C hardcodes RandomForestRegressor regardless of task_type | fed03:887 | P1 |
| FED-11 | fed05 always RandomForestRegressor regardless of task_type | fed05:113+ | P1 |
| FED-12 | fed03 Part B `silo_indices_all` dead code (implies federation that isn't happening) | fed03:668 | P1 |
| FED-13 | fed01 C0 "centralized upper bound" trained on NOISY labels (not a clean upper bound) | fed01:237 | P1 |
| FED-14 | class_weight='balanced' on FLuID teachers interacts with noise-skewed silos | fed01:226 | P1 |
| FED-15 | Cluster reconstruction may diverge from FLuID smoothed views (smooth_factor>0) | fed01:200 | P1 |
| FED-16 | Student hardcodes random_state=0, ignores caller seed | helpers:698 | P1 |
| FED-17 | predict_fluid_quality_ensemble uses KFold(shuffle=False) on sorted silo → distorted quality weights | helpers:544,1322 | P1 |
| FED-18 | fed05 single seed 42, no replicate sweep | fed05:86 | P1 |
| FED-19 | fed01 "reuse C5 hybrid" actually rebuilds (cosmetic/misleading) | fed01:417 | P1 |
| FED-20 | fed05 fragile insertion-order weight-prediction zip | fed05:670 | P1 |
| FED-21 | Bare `except Exception` in every condition hides NameError/ImportError/shape bugs | all fed | P1 |
| FED-22 | RF(150,15,balanced) hardcoded; DOWNSTREAM_MODELS imported but unused — never honours 7-model catalogue | helpers:70 | P2 |
| FED-23 | K=5 clusters hardcoded, no sweep | fed01:49, fed04:39 | P2 |
| FED-24 | precompute runs in run_all but conditions bypass the cache | helpers:1855 | P2 |
| FED-25 | internal_evaluator='match' passed to create_hybrid (default 'svm'; semantics may silently fall through) | fed01:241 etc | P2 |
| FED-26 | compute_clf_metrics lacks accuracy + f1_weighted (registered metric for antibody clf) | helpers:994 | P2 |
| FED-27 | fed03 Part A C7 cross_val_predict inconsistent (RF n=100 vs 150, cv=3, not group-aware) | fed03:454 | P2 |
| FED-28 | cluster_antibody_silos silently concatenates all reps as fallback (memory bomb) | helpers:1599 | P2 |
| FED-29 | fed04 KMeans centroids overwritten → labels_/inertia_ stale | helpers:309 | P2 |
| FED-30 | Dead `dataset_name=='herg'` branch | helpers:1769 | P2 |
| FED-31 | No per-replicate seed sweep (variance unknown) | all | P2 |
| FED-32 | Cross-task transfer allows mismatched task_type pairs | fed03:976 | P2 |
| FED-33 | compare_importance_profiles returns NaN<3 reps; fed04 treats NaN as "STABLE" → false-positive stability claim | helpers:1548; fed04:225 | P2 |
| FED-34 | DOWNSTREAM_MODELS imported only to populate unused config field | helpers:36 | P2 |
| FED-35 | _FLUID_AVAILABLE=False warns at import but crashes deep later | helpers | P2 |

## J. OPIG experiments (AUDIT_OPIG.md — 27)
| ID | Issue | Location | Pri |
|---|---|---|---|
| OPIG-1 | Imrie leakage filter excludes wrong hERG ('herg' vs 'herg_ki') → direct train/eval LEAKAGE for hERG-Ki fine-tune | chembl_bioactivity.py:328 | P0 |
| OPIG-2 | Trastuzumab overrides edit_distance groups with sequence identity → collapses to singletons → leave-edit-distance-out protocol BYPASSED | opig_trastuzumab:251 | P0 |
| OPIG-3 | Cache-key collision (augmentation_strategy default 'uncapped' with augmentations=None) in 3 OPIG experiments | opig_humanness:240, opig_trastuzumab:282, opig_cgr:292 | P0 |
| OPIG-4 | Trastuzumab+Humanness drop mutations to LM reps → 5 mutation-aware reps silently unreachable | opig_trastuzumab:262, opig_humanness:220 | P0 |
| OPIG-5 | Graphinity transfer has no PDB-level leakage check → train FoldX/test experimental share PDBs → overstates transfer | opig_graphinity:157 | P0 |
| OPIG-6 | Pool-best hardcoding tier='general' across 7 experiments (bypasses shared precompute) | 8 files | P1 |
| OPIG-7 | Imrie GP skipped for rep_dim>500 (ECFP4/mordred miss GP) → uneven condition counts | imrie:100 | P1 |
| OPIG-8 | Ginkgo hybrid uses singleton groups when groups=None → LeaveOneOut-like internal CV | opig_ginkgo:256 | P1 |
| OPIG-9 | Graphinity regex rejects multi-letter chain IDs (silent row drop) | graphinity.py:90 | P1 |
| OPIG-10 | Graphinity heavy/light chain by length heuristic (fragile) | graphinity.py:130 | P1 |
| OPIG-11 | CGR condition_suffix has spaces/'+' → cache-filename sanitization risk | opig_cgr:301 | P1 |
| OPIG-12 | Humanness 'species' PAPER_BASELINES empty dict (dead comparison path) | opig_humanness:97 | P1 |
| OPIG-13 | Trastuzumab --reps accepts arbitrary names w/o validation (silent "rep not generated") | opig_trastuzumab:462 | P1 |
| OPIG-14 | Imrie checkpoint feat_mean/feat_scale hard-error on old checkpoints (correct, informational) | dnn_pretrained:97 | P1 |
| OPIG-15 | pretrain_chembl_dnn inherits herg/herg_ki mismatch (consequence of OPIG-1) | pretrain:324 | P1 |
| OPIG-16 | Graphinity transfer_foldx reports std=0.0 (single train/test, misleading vs k-fold) | opig_graphinity:233 | P1 |
| OPIG-17 | Kingsmith assign_component_groups lowercases → could merge case-distinct components | opig_kingsmith:211 | P1 |
| OPIG-18 | Bare except swallows per-fold errors (7 experiments) | many | P2 |
| OPIG-19 | chembl_qsar no SMILES dedup / InChIKey conversion | chembl_qsar.py | P2 |
| OPIG-20 | _get_top_targets exhaustively iterates millions of IC50 rows (cold-start) | chembl_bioactivity:72 | P2 |
| OPIG-21 | Humanness _find_heavy_fasta heuristic falls back silently | humanness.py:96 | P2 |
| OPIG-22 | Ginkgo fold_assignments astype(int) crashes on NaN | ginkgo_dev:118 | P2 |
| OPIG-23 | CGR linear_cka no shape assertion; 0.0 indistinguishable from "no similarity" | opig_cgr:372 | P2 |
| OPIG-24 | blueprint_transfer source/target metric not enforced on domain mismatch | opig_blueprint:65 | P2 |
| OPIG-25 | Trastuzumab LLR rep handling assumes dim<=3 (brittle) | opig_trastuzumab:313 | P2 |
| OPIG-26 | Imrie rank-correlation thresholds 0.8/0.5 arbitrary | imrie:320 | P2 |
| OPIG-27 | blueprint_transfer uses create_hybrid on full source data (documented intentional) | opig_blueprint:188 | P2 |

## K. Complexity experiments (AUDIT_COMPLEXITY.md — 21; some overlap E/D above)
| ID | Issue | Location | Pri |
|---|---|---|---|
| CXA-1 | c24 `budgets=FULL_BUDGETS` but only SCREENING_BUDGETS imported → NameError every fold → "insufficient_folds", NEVER produces output | c24:158 (import :44) | P0 |
| CXA-2 | c11 line 376 uses undefined loop var `search` (should be last_search) → NameError if first iter fails | c11:376 | P0 |
| CXA-3 | c10 regression-only, no task_type guard → mangles class labels, silently wrong on every clf dataset | c10:28 | P0 |
| CXA-4 | c15 _eval_at_budget* regression-only, hardcodes RFRegressor/Ridge → clf metrics on regressor outputs | c15:592 | P0 |
| CXA-5 | c17 _rfe_select/_evaluate_strategy hardcode RFRegressor; task_type captured but unused | c17:137,168 | P0 |
| CXA-6 | c06 uses get_model_family_variants (hardcoded regressors) → wrong on clf | c06 / _shared:997 | P0 |
| CXA-7 | c13 modality_novelty never computes (feature_info never stored) → always NaN, diversity analysis dead | c13:529 | P1 |
| CXA-8 | c24 dead loop → best_model_src stays None → cross-fold always uses RF, ignores source fold's best model | c24:276 | P1 |
| CXA-9 | c08 silent calibration-point dropout when class missing from train → biased conformal coverage | c08:244 | P1 |
| CXA-10 | c19 nominal-on-ordinal arithmetic on multiclass labels → meaningless sample_weighted/combined | c19:50 | P1 |
| CXA-11 | c14 hybrid_config reads wrong dict keys (blueprint fields nested) → mostly empty strings | c14:251 | P1 |
| CXA-12 | cv_split silently falls back to random KFold when groups None/insufficient (forbidden) | _shared:515 | P1 |
| CXA-13 | c10/c11 leftover unused imports from pool migration | c10:21, c11:39 | P1 |
| CXA-14 | c18 budget sweep permanently locked to [50,100] (kwargs never passed) | c18:124 | P2 |
| CXA-15 | c23 aug_strategy dead lookup → always 'uncapped' | c23:129 | P2 |
| CXA-16 | c19 'best_overall' string key (confusable with removed evaluator) | c19:247 | P2 |
| CXA-17 | c07 `search` defined inside try (future NameError risk) | c07:274 | P2 |
| CXA-18 | c13 hand-maintained LEGACY_REPS will drift | c13:67 | P2 |
| CXA-19 | c12 reimplements shared metric/model stack (fixes won't propagate) | c12:161 | P2 |
| CXA-20 | c02 includes len(labels) as skip-key (peculiar vs allow_budget_exceeds_n) | c02:299 | P2 |
| CXA-21 | c03 svm proxy for SVR/ElasticNet blueprint undocumented | c03:90 | P2 |

## L. Similarity experiments (AUDIT_SIMILARITY.md — 20; some overlap F above)
| ID | Issue | Location | Pri |
|---|---|---|---|
| SIMA-1 | s06 loop var `k` shadowed by dict comprehension → wrong k or TypeError across entire AD analysis | s06:104 | P0 |
| SIMA-2 | s10 `task_type` referenced before bound → NameError swallowed → ALL hybrid branches (Tasks 1-4) silently skipped, reports as if produced hybrid results | s10:700 | P0 |
| SIMA-3 | s12 `top3` NameError when search succeeds but is None → aborts after CV ran | s12:263 | P1 |
| SIMA-4 | s11 hybrid_50d/refit ignore --hybrid-budget, use bp.budget → condition name misadvertises dim | s11:549 | P1 |
| SIMA-5 | s11 create_hybrid drops step_size + never threads augmentations into S11 at all | s11:567 | P1 |
| SIMA-6 | s07 bare except hides hybrid failure → reports baselines as if experiment ran (no skip flag in JSON) | s07:122 | P1 |
| SIMA-7 | s01/s08 pass raw importance to sqrt-weighting w/o np.maximum(.,0) → NaN risk if signed importance ever used (all other sim clamps) | s01:98, s08:127 | P1 |
| SIMA-8 | s09 AD-corr aggregation breaks after first k_key → hidden mix of k across datasets | s09:131 | P1 |
| SIMA-9 | s09 noise FP reduction divide-by-zero skips silently (drops dataset w/o reason) | s09:153 | P1 |
| SIMA-10 | s11 random shuffle fallback when groups None/<20 (forbidden random split) | s11:855 | P1 |
| SIMA-11 | s11 conditions reported as curves even when preconditions absent → degenerate pure-random curve labelled as method | s11:776 | P1 |
| SIMA-12 | s11 refit_interval not aligned between run_bo_loop and _run_bo_with_rebuild | s11:1086 | P2 |
| SIMA-13 | s10 is_self_query np.allclose O(n·d) every query call | s10:99 | P2 |
| SIMA-14 | s04/s12 cliff-membership python-`in` per element (minor) | s04:181, s12:199 | P2 |
| SIMA-15 | similarity_helpers per-threshold silent except (console-silent) | helpers:441 | P2 |
| SIMA-16 | SCREENING_BUDGETS hardcoded in similarity run config | helpers:629 | P2 |
| SIMA-17 | Distance-tier searches (general_distance/top_distance) NOT precomputed by run_all → on-demand high first-run latency for s04/05/06/07/10/13 | helpers:720 | P2 |
| SIMA-18 | Redundant make_cv_splitter construction inside per-budget/k loops | s05:116, s06:84 | P2 |
| SIMA-19 | cv_split `mode` arg accepted but ignored | _shared:486 | P2 |
| SIMA-20 | DISTANCE_EVALUATORS includes 'ridge' (forbidden as sensitivity check; here an internal evaluator) | _shared:261 | P2 |

## M. Noise experiments (AUDIT_NOISE.md 2026-05-23 — 20). Several P0/P1 since FIXED-in-code per the 2026-06-04 noise audit (pending ARC rerun), flagged below.
| ID | Issue | Location | Pri / Status |
|---|---|---|---|
| NA-1 | N07 per-rep eval forgets is_clf → regression noise on clf labels → per-rep junk, makes hybrid look spuriously good | n07:267 | P0 [FIXED-code 06-04] |
| NA-2 | N03 hybrid cache-contract violation (no dataset_name) → ValueError swallowed → hybrid columns silently empty for every dataset | n03:553 | P0 |
| NA-3 | N02 hybrid reliability scores ignore is_clf → nonsense weights on clf datasets | n02:311 | P0 [FIXED-code 06-04] |
| NA-4 | N09 weighted-hybrid scale collapse (features ×~0.01, no rescale) → conditions E/F handicapped by scale not hypothesis | n09:246 | P1 [FIXED-code 06-04] |
| NA-5 | Blueprint selected on CLEAN labels, per-fold rebuild under noise → every hybrid condition biased toward higher retention (6 experiments) | n07/09/10/11/12/13 | P1 |
| NA-6 | N10 ground-truth noise mask uses MAD threshold → shrinks positive class → detection precision/recall on biased subset | n10:515 | P1 |
| NA-7 | N04 noise injected once per fraction (single realisation) → claims conditional on one draw | n04:237 | P1 |
| NA-8 | N13 partial-noise semantics collapse for classification (becomes Gaussian at lower rate) | n13:77 | P1 |
| NA-9 | N13 silent unweighted MLP fallback in Track B → "remediated" == noisy baseline, inflates "no recovery" | n13:400 | P1 [FIXED-code 06-04] |
| NA-10 | N02 top3 overwritten by per-model loop var → top3_base_reps is last model's ranking, not search's | n02:252 | P2 |
| NA-11 | N08 f-string `:.3f` crash if cliff_correlation missing | n08:198 | P2 |
| NA-12 | N05 single noise realisation per σ (seed 42) | n05:140 | P2 |
| NA-13 | N09 retention% >100 for lower-is-better (confusing) | n09:381 | P2 |
| NA-14 | N10 importance recomputed on noisy labels (doubly noisy imp_weighted spaces) | n10:155 | P2 |
| NA-15 | N10 imp_weighted spaces partially undone by StandardScaler whitening | n10:162 | P2 |
| NA-16 | N13 partial-noise corrupt_idx never returned → can't measure detection on partial noise | n13:113 | P2 |
| NA-17 | Shared noise seed across experiments → cross-experiment "agreement" partly shared randomness | all N0x | P2 |
| NA-18 | N13 partial track sweeps partial_fraction at one strength while others sweep strength → mismatched axes | n13:296 | P2 |
| NA-19 | N04/N06 return raw dict not make_structured_output → downstream aggregation misses rows (noted in PRECOMPUTE_BUGS §4.5, still unfixed) | n04:365, n06:183 | P2 |
| NA-20 | N08 estimators (GP/conformal/BayesianRF/…) have no clf branch → fit regressors on integer labels on clf datasets | n08:130 | P2 |

---

# ════ PART T — TIME / RUNTIME / COST (CONSOLIDATED — the #1 priority, in one place) ════

> Answer to "are there sufficient time issues": the correctness audits were light on runtime, so this section systematically catalogues EVERY wall-clock cost driver. **Index of already-listed time issues** (by ID) + **new ones (TIME-*)** not previously captured. This is where to look for cuts/gates.

## T.A — Already listed above (time-relevant), by ID
COST-1 c17 MI uses all n · COST-2 c17 mRMR per-budget · COST-3 c17 prefilter=2000 · COST-4 c05 n_repeats=3 · COST-5 c05 can't reuse main cache · COST-6 SVM O(n²) uncapped · COST-7 capped 50 A/B refits · COST-8 c23 extra-tier models · COST-9 c21 giant aug dims · COST-10 node contention 3–48× · COST-11 SCREENING vs FULL budgets · COST-12 distance-lineage uncounted · CACHE-1 neural-rep hash drift · CACHE-3 FULL/SCREENING two cache families · CACHE-4 molecular re-running ~376h · CACHE-5 c09 3-fold→cache miss 120 recomputes · CX-4 c02 uncached FT recompute · CX-8 c12 quadratic in modalities · SIM-2 s01 OOM · SIM-3 s03 O(n²) kNN-MI · EST-4 c08-fix adds wall · SIMA-17 distance tier not precomputed → on-demand · FED-24 precompute bypassed · OPIG-20 _get_top_targets iterates millions.

## T.B — NEW structural / wave-level time issues
| ID | Issue | Mechanism | Impact |
|---|---|---|---|
| TIME-1 | **cx and sim each re-run the SAME precompute** | every dataset has TWO jobs (monitor shows two job IDs each); both cx `run_all` and sim `run_all` call `precompute_hybrid_searches` for general+top → the expensive precompute is computed ~2× per dataset (3× if the standalone precompute_hybrids.py also ran) | ~2× precompute wall across the wave |
| TIME-2 | **Running wave is 8-CPU; 32-CPU bump doesn't apply** | commit 0a6b746 (cpus 8→32) only affects jobs submitted after it; the 69 running jobs are 8-CPU → tree models (rf/xgb/lgbm, n_jobs=-1) run ~4× slower than a resubmit would | whole wave slower than necessary |
| TIME-3 | **No intra-precompute checkpointing** | a job killed at the 7-day wall loses all in-progress condition work; restart redoes the whole condition | lost partial progress on every timed-out job |
| TIME-4 | **Single conditions exceed the 7-day partition** | e.g. herg general/uncapped SVM ≈302h, kiba uncapped ≫168h — a single condition can't fit one partition → never completes regardless of restarts | those conditions are unachievable as-is |
| TIME-5 | **GP O(n³) dominates small-antibody searches** | GP per_cfg up to 65s (uncapped) / 820s (capped); 40–72% of small-antibody none/uncapped wall. Gated at _max_samples=1000 but unconstrained below it | small-ab precompute GP-bound |
| TIME-6 | **Model gates are coarse / not cost-tuned** | _max_samples (svm 10000, gp 1000) and MODEL_MAX_BUDGET are correctness caps, not cost-tuned; svm at n=5000–10000 still O(n²) | misses the cheap-win gates (see COST-6) |
| TIME-7 | **Experiments' fresh searches use FULL_BUDGETS(8), precompute used SCREENING(4)** | c05/c24 (and c17) hardcode FULL_BUDGETS while precompute ran SCREENING → budget-set mismatch → their internal searches CANNOT reuse the precompute cache → full recompute every time | c05/c24/c17 never hit precompute cache (compounds COST-4/5, CACHE-3) |
| TIME-8 | **Capped condition has no large-n gate beyond n≤250, but c05 invokes it per subsample** | c05 runs capped on every n_sub≤250 → inherits the 50-A/B-refit blowup (COST-7) 15× per dataset | small-ab c05 capped-bound |

## T.C — NEW per-experiment cost drivers (each a candidate for a cut/gate; cx)
| ID | Experiment | Cost driver | Gate applied? |
|---|---|---|---|
| TIME-c01 | c01 rep-characterization | ∝ Σ rep dims; HER2 c01=1.07h vs bvp 0.02h (44×) — pure no-FBH compute over every rep | none |
| TIME-c02 | c02 complexity grid | uncached `general_ft` fine-tune pass recomputes each run (= CX-4) + full budget grid | none |
| TIME-c04 | c04 importance-as-metric | ∝ g (model-family variants) × n; HER2 2.57h | none (regression-gated only) |
| TIME-c07 | c07 data-complexity partitions | fresh cache-MISS searches on data-dependent subpopulations → +500–670% swings | none |
| TIME-c09 | c09 formal analysis | 120 `create_hybrid` calls (volume) + 3-fold≠5-fold cache miss (CACHE-5); 0.8–10.4h | none |
| TIME-c11 | c11 ensemble-vs-hybrid | per-fold hybrid builds × ensemble members; 0.5–1.9h | none |
| TIME-c13 | c13 forward-compat | per-"new"-rep fresh search (#reps not in LEGACY_REPS); 0.2–18.6h (wildly variable) | none |
| TIME-c15 | c15 ML-theory-depth | double-descent + approximation-theory budget sweeps (many evals); +956% swings | none |
| TIME-c16 | c16 feature-selection-as-reg | per-budget pool-hybrid builds; 0.4–2.5h | none |
| TIME-c17 | c17 (the #1 cost) | mRMR — see COST-1/2/3; 6.7–53.4h measured, 198h–thousands derived | none (the gates in COST-1/2/3 are the fix) |
| TIME-c21 | c21 single-rep augmentation | per-base-rep search over giant aug pool (esm2_per_region 17920D); 0.2–19.3h | dim>n skip only |
| TIME-c23 | c23 evaluator-model matching | tier='extra' (catboost/dnn/cnn) refits; HER2 anomaly +49857% | none |
| TIME-c24 | c24 blueprint transfer | 5 fresh FULL_BUDGETS searches + 25 cross-fold cells; can't reuse precompute (TIME-7) | <2-fold early return |
| TIME-c12 | c12 modality ablation (TCGA) | C(G,2) pairwise × 5 models × 5 folds on wide concats (= CX-8) | tcga-only |

## T.D — NEW per-experiment cost drivers (sim)
| ID | Experiment | Cost driver |
|---|---|---|
| TIME-s01 | s01 similarity baseline | n×n distance matrices ∝ n²·Σdim → ~35GB RAM/rep, OOM at n≥30k (= SIM-2) |
| TIME-s03 | s03 feature concentration | brute-force kNN mutual-info O(n²) → ~111h at n=30k (biggest sim cost) (= SIM-3) |
| TIME-s05 | s05 kNN prediction | O(n²) kNN at large n (~16h@30k) |
| TIME-s10 | s10 importance downstream | O(n²) at large n (~25–31h@≥30k) |
| TIME-s13 | s13 clustering/retrieval | n² retrieval (~2h@30k) |
| TIME-sdist | sim distance lineage | general_distance+top_distance precompute (4 conditions, 6 budgets, 6 evaluators) — uncounted (COST-12) AND not precomputed by run_all (SIMA-17) → on-demand recompute |

## T — TIME COVERAGE NOW: ~28 pre-existing (indexed in T.A) + **~28 new (T.B 8 + T.C 14 + T.D 6)** = **~56 distinct time/runtime/cost issues**, consolidated. The dominant ones remain c17 (COST-1/2/3) and the cache/duplicate-precompute waste (CACHE-1/4, TIME-1).

# ════ PART V (issue list) — STILL TO MINE for MORE time issues ════
`PRECOMPUTE_EXPERIMENT_BUGS_2026_04_17.md` (32KB — likely heavy on runtime) and `COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md` and `CACHE_FIX_PLAN.md` are being read by the workflow; expect more time issues from those specifically.

---

# ════ PART ORD — LOOP ORDERING: expensive variant computed LAST (confirmed from code) ════
> 7 of 10 cost-varying for-loops put the highest-compute variant last; the 4 universal precompute loops are ALL expensive-last → every timeout sacrifices exactly the most expensive variant.
| ID | Loop | Order | Expensive-last? | Consequence |
|---|---|---|---|---|
| ORD-1 | find_best_hybrid MODEL loop (`find_best_hybrid.py:3056` `models.items()`) | rf→xgboost→ridge→knn→lightgbm→mlp→svm→gp | 🔴 YES (svm O(n²), gp O(n³) last) | timeout drops gp/svm; "best model" biased to cheap models |
| ORD-2 | find_best_hybrid BUDGET loop (`:3059`, ascending, not re-sorted) | 10→…→2000 | 🔴 YES (2000 last) | timeout drops budget-2000 cells; "best budget" biased low (compounds SCREENING bias) |
| ORD-3 | precompute STRATEGY within tier (precompute_hybrids.py:311-335) | none→uncapped→capped | 🔴 YES (capped last) | capped never reached at wall |
| ORD-4 | precompute CONDITION list order | general/none→…→top/capped | 🔴 YES | missing conditions = systematically the last/most-expensive ones, not random |
| ORD-5 | c05 subsample loop (`c05:110`, sizes sorted asc) | 0.1n→1.0n | 🔴 YES (full-n last) | timeout drops the full-size learning-curve point |
| ORD-6 | c17 budget loop (`c17:244`, ascending) | 10→…→2000 | 🔴 YES | biggest budget mRMR never reached |
| ORD-7 | DISTANCE_BUDGETS sweep | 10→…→500 | 🔴 YES | distance precompute drops high budget |
| ORD-8 | c17 strategy loop (`c17:228`) | hybrid,pca,mrmr,rfe,random | 🟢 NO (random cheapest is last) | — |
| ORD-9 | cx experiment sequence c01→c24 | mixed (c17 ~#14) | 🟡 partial | — |
| ORD-10 | sim experiment sequence s01→s13 | s01/s03 O(n²) FIRST | 🟢 NO (expensive-first) | the one expensive-first loop |
| ORD-NEST | **precompute is quadruply expensive-last**: conditions(capped last) × models(gp last) × budgets(2000 last) | — | 🔴 | does general/none+rf+budget-10 FIRST, gp/top-capped/budget-2000 DEAD LAST |
| ORD-KNOB | `KIRBY_PRECOMPUTE_REVERSE_ORDER=1` reverses CONDITION order only | — | — | inner model/budget loops stay expensive-last regardless |

**Lever (option): reverse/interleave the budget+model loops (expensive-first) → fail-fast on capped/gp/budget-2000 + the expensive variant completes-or-is-ruled-out instead of perpetually half-done. Caveat: at very large n, gp/svm are gated out so the model-tail expense vanishes there (budget-last + capped-last still hold).**

### ORD-REVFLAG — the reverse flag (used for similarity) is the WEAK version; `--condition` per-job is strictly better
The flag (`precompute_hybrids.py:373`) only does `conditions = list(reversed(conditions))` — flips the OUTER condition list. Three limits:
1. **Inner loops untouched** — within each condition, budgets still 10→2000 (`:3059`) and models rf→…→gp (`:3056`) → expensive CELLS still last even when "reversed".
2. **🔴 Premise broken by CACHE-1** — the flag's design ("two waves work on different cache files") assumes caching works; neural-rep hash drift makes the forward wave reject + recompute the reverse wave's cached middle conditions → **the two waves COLLIDE rather than parallelize**. Bites molecular/reaction similarity (neural reps); antibody/TCGA/nucleotide are deterministic → OK there.
3. **Binary + cost-agnostic** — only 2 waves; `reversed()` is append-order flip, not a cost sort.
**Improvement (already in-codebase): `--condition` per-job** (`:348-366`, covers similarity distance conditions `general_distance_*`/`top_distance_*`; already used by `resubmit_missing.sh:208`, `migrate_short_to_medium.sh:228`, `migrate_medium_to_long.sh:244`). A SLURM array of `--condition` jobs = disjoint work (no race, CACHE-1 can't cross-collide), each expensive condition gets its OWN 7-day partition (fixes IX.1 at the condition axis), N-way parallel + prioritizable. The residual cell-level expensive-last still needs the inner-loop cost-ordering (ORD-1/ORD-2).

# ════ PART LIN — FORBIDDEN LINEAR/POWER GROWTH ASSUMPTIONS IN THE ESTIMATES (most important class) ════
> Full detail + classification in ESTIMATES_INVESTIGATION PART VIII. These are places MY OWN estimates assume linear/power growth in dataset size (n) or representation size (dim) — the banned `(n/anchor)^k` family. 🔴=forbidden, 🟠=measured-anchored, 🟢=exact loop count.
| ID | Assumption | Class | Note |
|---|---|---|---|
| LIN-1 | c17 = 53.4h×(n/418)^1.078 (backbone of the cx column) | 🔴 | banned form; 2-point fit extrapolated 70–240×; cx column = order-of-magnitude only |
| LIN-2 | cx total ≈ c17×1.5 | 🔴 | measured pipelines show 4.3–5.2×, not 1.5× |
| LIN-3 | s11 ≈ linear in pool (0.8n) | 🔴 | ignores GP n³ + n_pool×dim predict |
| LIN-4 | capped RF per_cfg ≈ linear in n | 🔴 | same line notes 2× rep-dim scatter at fixed n |
| LIN-5 | c22 SHAP ∝ n_val×features×trees | 🔴 | product-of-linears (minor; c22 cheap) |
| LIN-6 | c04 ∝ g×n | 🔴 | per-unit ×n |
| LIN-8 | c17 ∝ total_dim (M factor) | 🔴 | prefilter O(total_dim) but PCA/greedy/eval not linear in dim |
| LIN-12 | none→uncapped μ multipliers fixed, independent of n | 🔴 | svm O(n²) → ratio grows with n → under-estimates large n |
| LIN-13 | top/none 0.6×, top/uncapped 10×, top/capped 114× gen/none | 🔴 | fixed cross-condition multipliers |
| LIN-14 | distance ≈ 0.9×none + 0.55×uncapped | 🔴 | fixed multipliers, n-independent |
| LIN-15 | c12 t_fit ~constant per cell | 🔴 | per-fit cost grows with n + concat width |
| LIN-16 | sim II.4 cross-size interpolation | 🔴 | smooth-scaling implied across size classes |
| LIN-7 | c24 ≈ 5×measured(gen/none) | 🟠 | 5=folds (exact); base measured → least-bad |
| LIN-9 | c01 ∝ rep dims; s01 ∝ n²·rep_dim | 🟠 | anchored on measured rep-size effect; rep_dim unmeasured |
| LIN-11 | s11 kernel ∝ n_pool×n_labeled×2048 | 🟠 | structure correct; "linear in pool" (LIN-3) is the overreach |

**Net: 11 🔴 (forbidden) + 3 🟠. The cx column's backbone is 🔴 ⇒ cx hours are order-of-magnitude/ordering only. Only directly-measured walls are firm.**

---

# ════ GAP-FILL RE-EXTRACTION (the 5 docs the workflow's StructuredOutput failed on) ════
> Re-extracted via plain-text agents. ISS / PCB / BRK / RES still running — appended as they land.

## issues_forward_compat.md — 37 (FWD-1…37) [DONE]
Mostly forward-compatibility model gaps + the downstream-model-expansion. TIME-relevant: FWD-26/28/33.
- **Models not integrated (open):** FWD-1 MINT (DTA, +29% SKEMPI) · FWD-2 ESM3 · FWD-3 ESM-C 600M/6B [partial] · FWD-4 Boltz-2 (mol+protein) · FWD-5/6 AlphaFold3 (+license/cost caveats) · FWD-7 Chai-1 (needs hook) · FWD-8/9 MolE (+NC license) · FWD-10 GP-MoLFormer · FWD-11 MiniMol · FWD-12 IgBERT/IgT5 · FWD-13 S2ALM · FWD-14 Ab-RoBERTa · FWD-15 IgBlend · FWD-16 SaProt · FWD-17 ProSST · FWD-18 S3F · FWD-19 Protenix.
- **Forward-compat experiments not run:** FWD-20 MINT-DTA · FWD-21 SaProt/ESM3 antibody+DTA · FWD-22 MolE/MiniMol molecular · FWD-23 IgBERT/S2ALM antibody · FWD-24 Boltz-2 cross-domain.
- **Downstream-model expansion (methodology/model):** FWD-25 Ridge is bottom-third (weak baseline by design) · **FWD-26 add SVR-rbf (#1 ranked) — O(n²) memory, subsample >10k** 🕒 · **FWD-27/28 add GP — O(n³), problematic >3–5k** 🕒 · FWD-29 advanced GP (Kermut/GAUCHE Tanimoto) · FWD-30 add MLP/FNN · FWD-31 CatBoost · FWD-32 DNN(3-layer) · **FWD-33 1D-CNN blocked by fixed-vector architecture (needs per-residue input)** · FWD-34 TabPFN · FWD-35 XGBoost ~5%>LGBM/CatBoost · FWD-36 rep quality dominates model choice · FWD-37 per-domain model tiers not encoded.

## ISSUES.md — 85 (ISS-1…85) [DONE] — the master; ~60 FIXED (your Phase 0/1 work), ~25 open
**Fixed Phase-0/1 (record):** ISS-1/2/3 row-misalignment MACCS/PDV/mol2vec · ISS-4 abagym seq-before-NaN · ISS-5 random-KFold fallback · ISS-7 ESOL/MoleculeNet · ISS-8 KIBA polarity · ISS-13/14/15/16/17 missing ρ/RMSE/CI/Pearson/AUROC · ISS-18…25 eight data-leakage reps (PDV/PCA/TCGA/esm2_entropy/tune_models/gene_immune/6-TCGA) · ISS-26/27 reaction random-split leakage · ISS-28 DMS identity→LOO · ISS-31 QSAR-ready preprocessing · ISS-33 BOS/EOS pooling · ISS-34 outlier clipping · ISS-35 IgBERT mean-collapse · ISS-37 Ridge-only · ISS-38/41/42 GP/DNN/CNN added · ISS-52 InChIKey dedup. **Open:** ISS-29/30 small-Ab + DDG 56% leakage [partial] · ISS-44 TabPFN · ISS-45 Chemprop · ISS-53/54 PaDEL/Yield-BERT · ISS-64/65 SKEMPI/TCGA expansion [partial] · ISS-66…74 TCGA Tier3/MOJO/WSI + AlphaFold + forward-compat (S2ALM/Boltz-2-mol/MINT/SaProt/MolE) · ISS-76…80 documentation-needed (GroupKFold-vs-scaffold, DDG/antibody expected-vs-bug) · ISS-81/82/83 **Phase-5 re-run all surveys + rebuild BOM** [open] · ISS-84 StratifiedGroupKFold TCGA · ISS-85 Butina clustering.

## PRECOMPUTE_EXPERIMENT_BUGS — 50 (PCB-1…50) [DONE] — 🕒 the most TIME-relevant doc
**🔴 MAJOR NEW TIME FINDING (precompute-bypass cluster):** **PCB-3 + PCB-30…36 + PCB-40** — synthetic `dataset_name` suffixes (`{real}_{suffix}`) used by **c05, c13, c23, and EVERY OPIG experiment + fed03 herg_asymmetric** are ABSENT from `DATASET_REPS` → precompute **never seeds them → every call is a full COLD recompute** (generalises COST-5 across ~10+ experiments). Plus PCB-31/36 (humanness_species, 3 Ginkgo props not in DATASET_REPS). **PCB-45 names these as the aggregate root cause of "experiments stuck in survey stage."**
**Cache (12+):** PCB-2 `_make_search_meta` omits ~13 params → silent collisions · PCB-4 cv_splitter identity not keyed · PCB-6 rep_factories unhashable/skipped · PCB-7 checkpoint `.ckpt.pkl` no meta-validation · PCB-8 dead folds-cache risk · PCB-14 atomic-write not NFS-safe · PCB-15 KIRBY_CACHE_DIR read in 2 places · PCB-18 c02 custom models→key change · **PCB-19 c09 single-budget→guaranteed miss** (=CACHE-5) · PCB-43 only FULL tier ever written · PCB-50 in-session vs SLURM byte-alignment unverified (=CACHE-3/2.4).
**Cost-runtime:** PCB-25 n03 cache bypassed 3/4 sub-analyses · PCB-43 fast-tier never precomputed · PCB-45 stuck-in-survey aggregate.
**Bugs:** PCB-1 11-site NameError crashes · PCB-5 s04 treats all as regression · PCB-11/12 task_type='auto' heuristic reachable · PCB-26 dead `budget=` arg in 7 noise files · PCB-28 imrie missing task_type · PCB-29 blueprint_transfer wrong task_type.

## COMPLEXITY_SIMILARITY_BROKEN — 48 (BRK-1…48) [DONE] — heavy overlap w/ AUDIT_COMPLEXITY/SIMILARITY + NOISE; new/confirmed:
**MI estimator collapse [FIXED]:** BRK-1…5,15 — KSG MI returned ~0 everywhere → ALL MI claims (c01/c09/c17/s03) were dead; replaced with sklearn mutual_info (this is the `_shared.py:1160` version; **so c17's MI now actually computes — cost analysis stands, prior results were degenerate**). **Dead/total-loss:** BRK-6 **c10 = "noiseInject not installed" → all 9 files are 87-byte error stubs** [fixed=dep added] · BRK-11/12 s04/s11 molecular-only gate → all skipped · BRK-9/10 c24 insufficient_folds on Ang2/VEGF (=CXA-1) · BRK-7/8 c21 NaN-saturated / no aug reps. **Fixed code bugs:** BRK-22 c05 budget>train→NaN · BRK-24 c06 budget-grid collapse→NaN · BRK-13/14 s07/s12 cliff threshold (100% vs 0%) · BRK-36/37 s06 graded-error+undercoverage · BRK-40 s08 dim-collapse · BRK-41 s10 RMSE sign-trap. **Open negatives (real, not bugs):** BRK-17 c11 VEGF loss · BRK-26 c08 underfit best-calibrated (refutes hypothesis) · BRK-28 c09 residual 69-98% · BRK-29 c17 RFE≥hybrid · BRK-30 c18 collapses at b=100 · BRK-38/42 s06/s10 near-zero. **BRK-45 macOS OMP segfault · BRK-46/47/48 ARC rerun + powered-dataset rerun + cliff-percentile retune needed.**

## COMPLEXITY_SIMILARITY_RESULTS — 51 (RES-1…51) [DONE] — results-validity (overlaps BRK; the methodology layer)
**Validity gates:** RES-1 suite dominated by n<200 (5-fold = noise) · RES-2/49 deep experiments (c09–c24/s06–s13) ran on the SMALLEST data → least power → **need rerun on powered sets** · RES-48 "best over budgets×conditions" post-hoc-optimistic (c02/s05). **Powered reversals (the real signal):** RES-4 c02 VEGF −0.027 · RES-5 hybrid gain only +0.01–0.05ρ on powered, near-ceiling on saturated · RES-11 c05 VEGF · RES-14 c08 underfit best on all 4 powered · RES-20 c11 VEGF · RES-30 c23 matched≈noise (−0.0073ρ) · RES-34/38 s05/s10 weighting adds nothing downstream. **Dead/degenerate:** RES-8/17/25/32/47 MI ~0 (c01/c09/c17/s03) · RES-10 c05 abagym all-NaN · RES-12 c06 degenerate · RES-18 c10 total loss · RES-22/24/27 c13/c16/c19 missing arms · RES-28 c21 all-NaN powered · RES-31 c24 errored · RES-33/40 s04/s11 skipped · RES-36/41 s07/s12 cliff degenerate. **Cross-cutting:** RES-46 weighting helps SAC metric but not downstream (s01/s02 vs s05/s10 tension) · RES-50 s04/s11 domain-mismatch decision needed.

---

## RUNNING COUNTS — FINAL (all 23 target docs mined: 18 workflow + 5 gap-fill agents)
- **Workflow whb39y4kb (18 docs):** 927 raw → **358 distinct** (603 master rows). Categories: EXP 117, MODEL 67, OTHER 60, REP 57, PREP 53, DEAD 53, METRIC 52, SPLIT 36, CACHE 32, LEAK 30, COST 28, MISS 24.
- **Gap-fill (5 docs the workflow's StructuredOutput failed on): 271** — ISS 85, PCB 50, RES 51, BRK 48, FWD 37.
- **Session-specific (NOT in any doc — this session's analysis): ~46** — PART T time (~28 new), ORD 10, LIN 14, ESTIMATES EA 16 + WRONG 6 (overlap each other).
- **Raw documented-issue rows mined across all 23 docs: ~1,198** (927 + 271).
- **Honest distinct estimate: ~450–480 documented + ~46 session-specific ≈ ~500.** The 271 gap-fill rows OVERLAP the workflow's 358 heavily (BRK/RES ↔ AUDIT_COMPLEXITY/SIMILARITY/NOISE + EXPERIMENT_DEBUGGING; ISS ↔ EXPERIMENT_BUGS + issues_*; PCB ↔ CACHE_FIX_PLAN). A full cross-doc dedup of the 1,198 union was NOT run — so the precise distinct number needs one final merge pass; the genuinely-new gap-fill content is ISS Phase-5/TCGA-expansion + the **PCB precompute-bypass cluster** (~15 new cost rows) + dead-experiment confirmations (c10/c24/MI).
- **Coverage: 23/23 target docs ✓** (per-doc counts above = the proof).
- **By severity:** the 6 cross-doc clusters (WORKFLOW SYNTHESIS) + ~23 audit P0s are the "silently-wrong-results" tier; the TIME tier is PART T (~56) + PCB precompute-bypass + ORD/LIN/PART-X.
- **NOT YET DONE:** (1) final cross-doc dedup of the 1,198 → exact distinct count + single ranked table; (2) recover the workflow's truncated first-9-category per-row detail (counts survive); (3) merge session-specific + documented into one ranked list.

---

# ════ NEW ISSUES FOUND DURING THE FIX-FLEET (rounds 1+2, 2026-06-04) — LIVE LIST, do NOT lose ════
> Recorded so bugs don't progress because they were forgotten. Coverage map: `COVERAGE_AUDIT_2026_06_04.md`. Decisions: `DECISIONS_FOR_USER.md`. Cache ops: `CACHE_OPERATIONS_2026_06_04.md`.

## FIXED this session (kept for the record)
| ID | Issue | Where | Status |
|---|---|---|---|
| NEW-F1 | Graphinity multi-letter chain regex stole digits from residue position (`YH102A`→chain H10/pos 2) → corrupt mutant sequence → every Graphinity rep wrong | graphinity.py:92 | FIXED + cache purge required (CACHE_OPERATIONS §1) |
| NEW-F2 | s11 TanimotoKernel incompatible with sklearn 1.x → AttributeError crash on any ecfp4 molecular run | s11:68-130 | FIXED (proper sklearn kernel, not disabled) |
| NEW-F3 | `find_best_hybrid._make_cv_splitter` silent random-KFold fallback (1 group / groups=None → `KFold(shuffle=True)`) — violated no-random-splits | find_best_hybrid.py:1899 | FIXED → now RAISES |
| NEW-F4 | ISS-84 production: TCGA clf folds used plain GroupKFold (unstratified) → imbalanced folds → noisy clf metrics | find_best_hybrid._make_cv_splitter + threading | FIXED → StratifiedGroupKFold for clf (task_type threaded) |
| NEW-F5 | ISS-29: antibody exact-identity grouping put single-residue analogs in different folds → 87% of near-dup pairs span train/test on flab_thermo_tresanco (leakage) | antibody_baseline_survey.py:314 | FIXED → sequence_identity_clusters (results-affecting; DECISIONS Group 2) |

## OPEN / DECISION (next pass — each is a real item, none deferred-and-forgotten)
| ID | Issue | Where | Type |
|---|---|---|---|
| NEW-O1 | Rep dispatch is decentralized: molecular/DTA new reps (boltz2_mol, protein_prosst, protein_esmc6b) need a 1-line entry in `tests/find_best_hybrid_{molecular,dta}.py` to be reachable (antibody auto-dispatches) | survey files | wiring (needed at activation) |
| NEW-O2 | `xtrimopglm` + `aido_protein` flagged dead/NaN in complexity_helpers comments but STILL in ANTIBODY_REPS → wasted compute + NaN pollution every antibody run | ANTIBODY_REPS | DECISION: remove (search-space change → signoff) |
| NEW-O3 | ProSST runs sequence-only (neutral zero structure tokens); full structure-aware value needs the ProSST quantizer on PDBs → real `ss_input_ids` | protein.py:2892 | open (wired honestly) |
| NEW-O4 | S2ALM public weights unconfirmed → stub raises NotImplementedError, left unregistered | antibody.py:1271 | open (needs weight-source decision) |
| NEW-O5 | Antibody identity grouping uses HEAVY chain only; for paired datasets pass heavy+light concatenated to catch light-chain-only near-dups (follow-up to NEW-F5) | antibody_baseline_survey.py:254-294 | open (leakage follow-up) |
| NEW-O6 | DTA/flab/ddg cap size via a seeded random ROW subsample (`np.random.choice`) — not a split (split stays group-aware) but a non-grouped draw | _shared.py:2047, flab.py:285, ddg_combined.py:943 | low severity (flag) |
| NEW-O7 | `run_experiment_suite` builds config with no `'budgets'` key → KeyError for any experiment reading `config['budgets']` via that path | _shared.py:3400 | open |
| NEW-O8 | `precompute_hybrid_searches` ignores `config['budgets']` but its docstring claims it reads it (the trap behind the whole SCREENING confusion) | _shared.py:2263 | open (docstring + guard) |
| NEW-O9 | `AUG_LOGIC_VERSION` defined but unconfirmed whether threaded into the hybrid_eval meta → a stale-aug-logic cell could be served | find_best_hybrid.py:157 | open (verify) |
| NEW-O10 | `subset_hashes` neural-rep content drift makes the hybrid_eval FILENAME differ run-to-run → those configs never reuse the dedup across runs (missed optimization, NOT wrong/churn) | find_best_hybrid hybrid_evals | open (optimization) |
| NEW-O11 | n10 `_compute_summary` Q4 hardcodes `[25,100,250]` independently of `HYBRID_BUDGETS` → drifts if HYBRID_BUDGETS changes | n10:Q4 | open (minor) |
| NEW-O12 | s07 weighted-arm skip is silent in the summary (`delta_vs_best_rep_weighted`=NaN, no flag) | s07 | open (minor) |
| NEW-O13 | c11 PCA arm reports last-fold `n_comp` as `total_dim` (loop-var leakage; near-constant, reporting artifact) | c11:251 | open (minor) |
| NEW-O14 | complexity_helpers module docstring still mentions SCREENING_BUDGETS in prose (stale after 3i) | complexity_helpers.py:11-13 | cosmetic |

**Plus the un-applied cost/activation DECISIONS** (full text in DECISIONS_FOR_USER.md §3–4): c17 MI subsample + prefilter (large-n levers), model-catalogue additions (TabPFN/SVR-selective/GAUCHE-GP), forward-compat ACTIVATION lists (which reps→which datasets, cost-flagged), and the server-resource steps for the WIRED-but-blocked foundation models (FWD blocked list).
