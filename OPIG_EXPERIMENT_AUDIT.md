# OPIG Experiment Audit (2026-03-28)

All 6 OPIG experiments audited against original papers/GitHub repos.
Each section notes whether the comparison used the actual code repo or paper only.

**STATUS: 6/6 audited against GitHub repos. All repos cloned to `/Users/apunt/repos/opig_refs/`.**

---

## 1. opig_trastuzumab_enrichment.py (OPIG B1)

**Paper:** Chinery et al. 2024 "Baselining the Buzz: Trastuzumab-HER2 Affinity, and Beyond"
**Repo:** https://github.com/lewis-chinery/Tz_her2_affinity_and_beyond (VERIFIED, fetched)
**Audit source:** GitHub repo README + source code under `src/`, `notebooks/`, `data/`

### What the paper actually does
The paper has TWO distinct phases:
- **Library design (zero-shot, generative):** BLOSUM, AbLang, ESM, ProteinMPNN generate candidate CDRH3 sequences from probability distributions. These are GENERATORS, not feature extractors.
- **CNN classifier:** Trained on 524K labeled data (70/15/15 random split) to predict binder/non-binder. Input = one-hot encoding of 10-residue CDRH3 only (shape 10x20). CNN architecture: Conv1D(400, kernel=5) → Dropout(0.2) → MaxPool → Flatten → Dense(300) → Sigmoid.
- **Enrichment:** Fraction of a computationally-designed library predicted as binders by the CNN. NOT top-K of held-out test set.
- **Experimental validation:** 720 designs tested by SPR. BLOSUM outperformed all 4 DL methods.

### What KIRBy does differently
| Aspect | Paper | KIRBy |
|--------|-------|-------|
| Task | Zero-shot library design → CNN scoring | Supervised classification on existing data |
| Sequence region | CDRH3 only (10 residues) | Full VH (120 residues) + VL |
| Models | CNN (1.1M params), FLAML/XGBoost, EGNN | Ridge, RF(500) |
| Enrichment meaning | % CNN-predicted binders in designed library | % true binders in top-K% of held-out predictions |
| Sample size | 367K train / 78K val / 78K test | 5K subsample, 5-fold CV |
| Split | Fixed random 70/15/15 | StratifiedGroupKFold by sequence identity |
| LM usage | AbLang/ESM as GENERATORS (masked-LM sampling) | AbLang2/ESM2 as FEATURE EXTRACTORS |

### Key problems
1. **Fundamental task mismatch.** Paper evaluates library DESIGN methods. KIRBy evaluates REPRESENTATION quality for classification. Different questions.
2. **PAPER_BASELINES values (lines 89-97) cannot be compared to KIRBy's enrichment** because they measure different things.
3. **Missing:** Iterative enrichment cycle, experimental validation, EGNN, the entire library design evaluation.
4. **Added (correct):** Zero-shot LLR ranking condition was added during this audit session.

---

## 2. opig_graphinity_ddg.py (OPIG B2)

**Paper:** Hummer et al. 2025 "Graphinity" (Nature Computational Science)
**Repo:** https://github.com/oxpig/Graphinity (VERIFIED, fetched)
**Data:** Zenodo 15384945 (VERIFIED)
**Audit source:** GitHub repo config files, training scripts, evaluation notebooks

### What the paper actually does
- Graphinity = E(n) Equivariant GNN on 3D atomic graphs from PDB structures
- Input: WT + mutant 3D structures, nodes = heavy atoms near mutation site
- Key: builds SEPARATE graphs for WT and mutant, takes DIFFERENCE of graph embeddings
- **10-fold CV** with CDR SEQUENCE IDENTITY cutoffs (random/100%/90%/70%)
- Core thesis: random splits inflate performance. Stringent cutoffs reveal true generalization.
- Ensemble of 10 models per fold, 500 epochs

### What KIRBy does differently
| Aspect | Paper | KIRBy |
|--------|-------|-------|
| Model | EGNN on 3D atomic graphs | Ridge, RF on sequence reps |
| Input | 3D C-alpha coordinates from PDB | VH/VL amino acid sequences |
| CV | 10-fold, CDR identity cutoffs (90%/70%) | 5-fold GroupKFold by PDB |
| Metric | Global Pearson r (concatenated folds) | Per-fold averaged Pearson r |
| Exp data | Experimental_DDG_608 or _645 (specific curation) | ddg_combined (KIRBy's own curation) |
| FoldX size | Full 942,723 | Subsampled to 50,000 |
| Ensemble | 10 models per fold | Single model |

### Key problems
1. **Paper's PAPER_BASELINES cite r~0.87 but this is with RANDOM splits** (data leakage). The paper's whole point is this number is inflated. KIRBy cites it as target without this context.
2. **CDR identity cutoff not replicated.** KIRBy groups by PDB (≈cutoff_100) but paper shows cutoff_70 is needed for honest evaluation.
3. **`augmented` condition was missing** from CONDITION_DATASETS — fixed during this audit.
4. **50K subsample** of FoldX (from 942K) is a major divergence.

---

## 3. opig_cdr_flexibility.py (OPIG — Spoendlin et al. 2025)

**Paper:** Spoendlin et al. 2025 "ITsFlexible" (Nature Machine Intelligence)
**Repo:** https://github.com/oxpig/ITsFlexible (VERIFIED, fetched)
**Audit source:** GitHub repo README, model code, data files

### What the paper actually does
- EGNN on 3D loop structures (C-alpha coordinates + residue type)
- Trains on ~73K GENERAL protein loops, tests on CDR-specific held-out sets
- Split by 80% sequence identity clustering
- Primary metric: PR AUC (early stopping monitors PR AUC)
- Tests ZERO sequence-only baselines (all baselines are structure-based: AlphaFold2 pLDDT, ABB2, solvent exposure)

### What KIRBy does differently
| Aspect | Paper | KIRBy |
|--------|-------|-------|
| Model | EGNN on 3D coordinates | Ridge, RF on sequence reps |
| Training data | 73K general loops → CDR test sets | CDR test data only (~34-216 samples) |
| Evaluation | Single train/test split | 5-fold StratifiedGroupKFold on tiny CDR data |
| Baselines | All structure-based | All sequence-based (NOVEL — paper tested none) |
| Per-structure vs per-sequence | Evaluates each PDB structure | Deduplicates by CDR sequence (keep first) |

### Key problems
1. **KIRBy never uses the 73K training loops** — only uses the CDR test sets. Very different protocol.
2. **Very small samples** after dedup: CDRA3=34, CDRB3=37. 5-fold CV on 34 samples is meaningless.
3. **Label inconsistency:** 3 CDRH3 sequences have mixed labels across PDB structures. KIRBy picks first arbitrarily.
4. **CDRA3/CDRB3 are TCR CDRs, not antibody CDRs.** Running antibody-specific LMs on TCR is questionable.
5. **Copy-paste comment** (line 221) says "Augmentation config matches tz_her2_binary" — wrong dataset.
6. **Positive:** KIRBy IS the first to test sequence-only baselines for CDR flexibility. This is genuinely novel.

---

## 4. opig_ginkgo_developability.py (OPIG B5)

**Paper/Benchmark:** van Niekerk et al. 2024, Ginkgo Bioworks
**Repo:** https://github.com/ginkgobioworks/abdev-benchmark (VERIFIED, fetched)
**Audit source:** GitHub repo README, orchestrator code, model configs

### What the benchmark does
- 16 models including: ElasticNet on AbLang2, Ridge/RF/XGBoost on ESM2+TAP, MOE 3D descriptors, DeepSP, Aggrescan3D, AntiFold, DeepViscosity
- Pre-computed 5-fold hierarchical cluster + IgG isotype stratified splits
- 5 core properties: HIC, Tm2, Titer, PR_CHO, AC-SINS
- Metrics: Spearman ρ (primary) + Top-10% recall (secondary)
- Also has 80-antibody held-out test set (labels withheld for competition)

### What KIRBy does differently
| Aspect | Benchmark | KIRBy |
|--------|-----------|-------|
| Models | 16 specialized models | Ridge, RF × 10 reps |
| Fold assignments | Pre-computed hierarchical cluster | **Correctly uses same folds** ✓ |
| Properties | 5 core | 9 (5 core + 4 extra) |
| Metrics | Spearman + Top-10% recall | Spearman, R², Pearson, RMSE (NO Top-10% recall) |
| Missing features | N/A | MOE, TAP, DeepSP, Aggrescan3D (all structure-based) |

### Key problems
1. **Top-10% recall not computed.** This is the benchmark's secondary metric.
2. **PAPER_BASELINES partially inaccurate.** "competition top" numbers (e.g., HIC test_top: 0.708) don't match the README's model tables. May be from the actual competition leaderboard (113 teams), not the open-source models.
3. **Missing structure-based features** (MOE, TAP, DeepSP) that are among the benchmark's best performers.
4. **Positive:** Fold assignments ARE faithfully used. This is correct.

---

## 5. opig_activity_cliff_qsar.py (OPIG — Dablander et al. 2023)

**Paper:** Dablander et al. 2023 "Exploring QSAR models for activity cliff prediction" (J Cheminformatics)
**Repo:** https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments (VERIFIED, fetched)
**Audit source:** GitHub repo main scripts, config, model training code

### What the paper does
- 4 models: RF, KNN, MLP (PyTorch, Optuna-tuned), GIN (graph neural net)
- 2 reps: ECFP (2048-bit) and PDV (CDF-normalized)
- Random 50/50 split × 3 seeds = 6 trials
- AC evaluation on 4 MMP partitions: M_train, M_inter, M_test, M_cores
- AC threshold: |ΔpActivity| ≥ 2.0. **Half-ACs (1-2) EXCLUDED** from evaluation.
- Primary metric: MAE for QSAR, MCC for AC detection

### What KIRBy does differently
| Aspect | Paper | KIRBy |
|--------|-------|-------|
| Models | RF, KNN, MLP, GIN (all tuned) | Ridge, RF (no tuning) |
| Split | Random 50/50 × 3 seeds | Scaffold GroupKFold 5-fold |
| AC threshold | ≥2.0, half-ACs EXCLUDED | >2.0, half-ACs counted as negatives |
| MMP evaluation | 4 separate partitions (M_train/inter/test/cores) | Single pooled evaluation |
| Hyperparams | RandomizedSearchCV (RF/KNN), Optuna (MLP/GIN) | Fixed defaults, no tuning |
| PD task | Property-difference classification (separate) | Missing entirely |
| PDV normalization | CDF-normalized | Raw |

### Key problems
1. **Missing models:** KNN, MLP (paper's best = ECFP+MLP), GIN. Replaced with Ridge.
2. **Half-AC exclusion not replicated.** Including half-ACs as negatives inflates TN, distorts MCC. Results NOT comparable to paper.
3. **MMP partition analysis missing.** Paper's key finding (M_inter vs M_test difference) cannot be reproduced.
4. **No hyperparameter tuning.** Paper carefully tuned all models.
5. **Different split strategy.** Scaffold (KIRBy) vs random (paper). Both valid but produce different results.
6. **Positive:** KIRBy adds extra reps (maccs, mordred, mol2vec, avalon, sns). This IS the KIRBy value.
7. **Positive:** Scaffold splitting is arguably more rigorous for QSAR.

---

## 6. opig_humanness.py (OPIG B3)

**Paper:** Prihoda et al. 2022 "BioPhi" (mAbs) — NOT Marks 2021 as KIRBy claims
**Repo:** https://github.com/Merck/BioPhi-2021-publication (VERIFIED, cloned to `/Users/apunt/repos/opig_refs/BioPhi-2021-publication/`)
**Audit source:** Repo notebook `notebooks/reports/03_oasis_humanness.ipynb`, verified against actual cells

### What BioPhi actually does
BioPhi does NOT train any classifiers. It is a **score discrimination study**:
- Computes 8 pre-existing humanness scoring methods on each antibody
- Evaluates which score best separates human vs non-human therapeutics
- **No train/test split, no CV.** ROC AUC + Max Accuracy computed on full dataset with threshold sweeping.
- Dataset: IMGT/mAb-DB, 553 therapeutic antibodies (after filtering ambiguous species)
- 8 methods compared: Hu-mAb (Marks), OASis (their own, 4 threshold variants), T20, Z-score, MG Score, AbLSTM, IgReconstruct, Germline content
- Both VH and VL scored, then averaged

### What KIRBy does differently
| Aspect | BioPhi | KIRBy |
|--------|--------|-------|
| Task | Score discrimination (threshold sweep, no training) | Supervised classification (train RF/Ridge on reps) |
| Features | 8 scalar humanness scores | 12 high-dim antibody reps (ESM2, AbLang2, etc.) |
| Models | No model training | RidgeClassifier + RF(500) |
| Evaluation | ROC AUC on full data (no CV) | 5-fold StratifiedGroupKFold |
| Chains | VH + VL (averaged) | VH only |
| Dataset filtering | Removes 11 ambiguous species entries | Keeps all 564 |
| Species preset | Not present in BioPhi | From AbNatiV (Ramon 2024), Zenodo 10562968 |

### Key problems
1. **KIRBy cites Marks 2021 as primary reference but uses BioPhi's (Prihoda 2022) data.** These are different groups, different methods. Marks = Hu-mAb RF. BioPhi = OASis peptide prevalence. KIRBy should cite Prihoda 2022 as the data source.
2. **Fundamentally different questions.** BioPhi asks "which humanness SCORE separates species?" KIRBy asks "which REPRESENTATION + classifier predicts species?" Not comparable.
3. **KIRBy PAPER_BASELINES claim AUROC=1.0 for species.** This value is not from the BioPhi repo. Cannot be verified. Likely from Marks 2021 paper on OAS test data (different dataset).
4. **VH-only vs VH+VL.** BioPhi averages both chains. KIRBy ignores light chain.
5. **No filtering of ambiguous species.** BioPhi removes Human/Humanized and Human/Chimeric entries. KIRBy labels them as non-human.
6. **BioPhi's evaluation is not proper ML benchmarking.** No CV, no held-out test. KIRBy's approach is more rigorous but the comparison is invalid since the protocols differ entirely.

---

## Cross-Cutting Issues (All 6 Experiments)

### A. Hardcoded models diverge from KIRBy catalog
All use `Ridge(alpha=1.0)` + `RF(n_estimators=500)`. KIRBy catalog uses `RidgeCV` (auto-tuned) and `RF(n_estimators=100)`. None of the original papers used Ridge — it was added by Claude as a generic baseline.

### B. Duplicated evaluation code
All 6 reimplement their own evaluate_rep function (~60 lines each) instead of calling the centralized survey code.

### C. Config dict ignored
All 6 hardcode everything. No way to override without editing source.

### D. Data loaders are CLEAN
All 7 dataset loaders download from correct original sources. Column names, unit conversions, and preprocessing verified. This part is solid.

### E. What original papers used that KIRBy is missing
- **Trastuzumab:** CNN, EGNN, iterative enrichment, experimental SPR validation
- **Graphinity:** EGNN on 3D structures, CDR identity cutoff analysis, ensemble of 10
- **ITsFlexible:** EGNN on 3D structures, 73K general loop training data
- **Ginkgo:** MOE 3D descriptors, TAP, DeepSP, Top-10% recall metric
- **Dablander:** KNN, MLP (Optuna-tuned), GIN, half-AC exclusion, MMP partitions
- **Humanness:** TBD

---

## What Needs Reconstruction (Priority Order)

1. **Activity cliff (Dablander):** Half-AC exclusion, MMP partition analysis, add MLP model. Repo: `/Users/apunt/repos/opig_refs/QSAR-activity-cliff-experiments/`
2. **Graphinity:** CDR identity cutoff evaluation, fix PAPER_BASELINES context (0.87 is from random splits — misleading). Repo: `/Users/apunt/repos/opig_refs/Graphinity/`
3. **Trastuzumab:** Clearly separate zero-shot (paper's protocol) from supervised (KIRBy extension). Paper did library DESIGN, not classification. Repo: `/Users/apunt/repos/opig_refs/Tz_her2_affinity_and_beyond/`
4. **Ginkgo:** Add Top-10% recall metric (benchmark's secondary metric). Repo: `/Users/apunt/repos/opig_refs/abdev-benchmark/`
5. **CDR flexibility:** Address tiny sample sizes, label inconsistency, TCR handling. Consider using 73K general loop training data. Repo: `/Users/apunt/repos/opig_refs/ITsFlexible/`
6. **Humanness:** Fix citation (Prihoda 2022 not Marks 2021), fix PAPER_BASELINES (AUROC=1.0 unverifiable), add VL support, filter ambiguous species. Repo: `/Users/apunt/repos/opig_refs/BioPhi-2021-publication/`
7. **All:** Replace hardcoded Ridge(1.0)+RF(500) with config-driven models. Use centralized evaluate_rep instead of per-experiment reimplementations.
