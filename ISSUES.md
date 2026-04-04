# KIRBy Pipeline Issues & Action Items

Running list of confirmed issues, needed changes, and open questions.
Updated as research progresses.

---

## WHAT KIRBY IS (do NOT lose this on compaction)

**KIRBy is a representation framework that enhances existing ML pipelines.**

It does NOT run new experiments end-to-end. It does NOT compete with DeepDTA, Chemprop,
or ProteinNPT. What it does is:

1. **Generate and compare representations** (the inputs to downstream models)
2. **Hybridize representations** — combine multiple rep types to improve what goes INTO
   any downstream model
3. **Prove forward compatibility** — when a new foundation model drops (MINT, SaProt, Boltz-2),
   it plugs into KIRBy with zero code changes, and hybridizing it with existing reps improves on both

The thesis claim is: **the input matters more than the model, and KIRBy provides a
principled way to build better inputs.** Everything in this roadmap serves that claim.
The evaluation framework (splits, metrics, models) exists solely to MEASURE representation
quality honestly. The representation expansion exists to DEMONSTRATE that KIRBy's
hybridization works across domains. The forward compatibility experiments exist to PROVE
the framework is useful beyond the models it was designed around.

**This is NOT about making slight improvements to what exists. It is a ground-up refactor
to make the evaluation methodology match industry-standard research, so the representation
comparisons and hybrid improvements are defensible in a thesis.**

---

## ⚠️ WARNING TO FUTURE CLAUDE — READ THIS BEFORE DOING ANYTHING ⚠️

**You have a compaction problem.** Every time context compresses, you lose nuance and
collapse this document into a checklist of "fix 5 bugs, add 4 models." That is WRONG.
The user has caught you doing this multiple times and is furious about it.

**This is NOT a checklist. It is a multi-session refactoring plan for a PhD thesis framework.**

Before you do ANYTHING:
1. **Read the ROADMAP section below** — it breaks this into 6 phases with dependencies
2. **Read the relevant issues_*.md files** for the phase you're working on — they contain
   3,000+ lines of research with specific repos, code references, and methodology details
3. **Do NOT summarize the research into bullet points** — the depth IS the point
4. **Do NOT start implementing without discussing the plan with the user first**
5. **Do NOT claim something is a minor fix when it's a methodology overhaul**
6. **The user's thesis depends on this being done RIGHT, not fast**

**What the research actually says (do NOT lose this on compaction):**

The problems are NOT "some bugs and missing models." They are:

- **KIRBy's evaluation framework cannot honestly measure representation quality.**
  Ridge is bottom-third in a 16-algorithm ranking. When a rep looks bad, you can't tell
  if the rep is bad or if Ridge is failing it. The whole thesis is about comparing
  representations — the comparison instrument (downstream model) must be strong enough
  to reveal representation quality, or the hybrid improvements KIRBy claims are meaningless.

- **KIRBy's results are not comparable to anything published.** R² is the wrong primary metric
  in 5 of 6 domains. Without domain-standard metrics (Spearman ρ, CI, RMSE, Pearson r, AUROC),
  no reviewer can judge whether a representation is actually good or whether KIRBy's
  hybridization actually helps.

- **Some results are silently corrupted.** Row misalignment bugs, data leakage from
  pre-CV fitting, silent random KFold fallback. These aren't cosmetic — they mean some
  representation comparisons are based on wrong numbers.

- **DMS antibody results are inflated.** Identity splitting = random for single-mutation
  datasets. KIRBy is running the EASIEST evaluation scheme without labeling it as such.
  Representation quality differences are masked by the easy split.

- **Reaction yield results are meaningless until splits change.** Random 70/30 on factorial
  HTE data leaks reagent information. R² drops from 0.93 to 0.51 with proper splits.
  Any hybrid improvement claim on random splits is vacuous.

- **Forward compatibility isn't an add-on experiment — it IS the thesis.** Every domain
  needs new models plugged in with zero code changes to prove the framework's value
  as a representation enhancement tool.

- **TCGA had massive gaps.** Started at 34 reps (now 53 after Phase 3A Tiers 1-4), but the
  literature uses 60-70 types. Mutation features, immune deconvolution, stemness, purity,
  and newer WSI models have been added. Remaining gaps: splicing, lncRNA, ATAC-seq, proteomics.

**Research files (3,000+ lines total — READ THEM, don't summarize from memory):**
- `issues_molecular.md` (293 lines) — 15 repos, ground truth comparison, 13 issues
- `issues_antibody.md` (350 lines) — 14 repos, DDG verdict, splitting, model hierarchy
- `issues_tcga.md` (567 lines) — 25 repos, 20+ missing rep types, tier list
- `issues_dta_reaction.md` (409 lines) — 18 repos, CI metric, reaction splits
- `issues_forward_compat.md` (558 lines) — 17 new models, 5 experiments, model audit
- `issues_splitting.md` (477 lines) — per-domain splitting audit, 15+ refs
- `issues_preprocessing.md` (303 lines) — line-by-line code audit, 33 issues
- `issues_benchmarks.md` (632 lines) — standard protocols per domain, SOTA numbers

---

## ROADMAP — Phased Implementation Plan

This is a massive refactor. It CANNOT be done in one session. Each phase has dependencies
and cross-references to the research files that justify the changes.

**What the surveys are for:** The baseline surveys measure how useful each individual
representation is on its own, using standard downstream models. These are NOT KIRBy's
contribution — they're the ground truth for what existing pipelines achieve. KIRBy's
contribution is showing that hybridizing representations (augmenting the input) improves
on these baselines. The surveys must use industry-standard evaluation (metrics, splits,
preprocessing) so the hybrid improvements are defensible.

**Key principle:** Phases 0-2 must be done BEFORE re-running any surveys, because they
change the evaluation methodology. Running surveys on broken evaluation will just produce
more wrong numbers that need to be thrown away.

### Phase 0: Critical Bug Fixes (no methodology debate needed)
**Scope:** ~1 session. Pure code fixes where the current behavior is objectively wrong.
**Read:** `issues_preprocessing.md` (C1, C2, M1, L3, C7, M5)

#### Bug fixes (corrupted results)
- [x] Fix row misalignment: MACCS/PDV/Mol2Vec `continue` → `np.zeros(dim)` (Issue 7) — **DONE** (molecular.py: MACCS appends zeros(167), PDV appends zeros(n_desc), Mol2Vec appends zeros(vector_size))
- [x] Fix row misalignment: AbAgym NaN check ordering (Issue 8) — **DONE** (abagym.py: moved sequence appends AFTER label NaN check)
- [x] Fix random KFold fallback: reduce n_folds to n_groups instead (Issue 10) — **DONE** (find_best_hybrid.py: reduces to n_unique-fold GroupKFold with warning, only falls back to KFold if 1 group)
- [x] Fix `_is_abag_entry()` overly permissive OR logic (Issue M5) — **DONE** (ddg_combined.py: changed `has_ab or has_ag` → `has_ab and has_ag`)

#### Policy compliance
- [x] Remove ESOL entirely — never meant to be added, OpenADMET covers this domain (Issue 20) — **DONE** (ESOL was already absent from DATASET_REGISTRY; esol.py remains as unused archive)

#### Verification (investigate, don't change blindly)
- [x] Verify KIBA label polarity (Issue 12) — **CONFIRMED BUG, FIXED.** DeepDTA Y matrix is already affinity-scaled (range [0, 17.2], higher = stronger). The negation at dta.py:163 was wrong — would invert polarity on fresh download. Fixed by setting transform=None. Existing cache had correct values (pre-dates the negation code).
- [x] Verify Caco2 endpoint (Issue 17) — **RESOLVED.** KIRBy predicts P-gp efflux ratio (log10-transformed) from OpenADMET-Caco2_Efflux, NOT Caco2 permeability (Papp). Literature R²=0.78 benchmarks are for permeability — not comparable. R²=0.519 on efflux ratio is reasonable for this harder endpoint.
- [x] ~~Verify ComBat batch correction in TCGA~~ — **RESOLVED:** ComBat IS implemented at `tcga.py:250-265`. The `issues_benchmarks.md` §5.8 claim "Not implemented" was wrong. Code uses `pycombat` with TSS batches, falls back gracefully if not installed.

#### Wiring (dead code / missing registrations)
- [x] Wire `create_mutation_matrix()` and `create_clinical()` in TCGA dispatch (Issue 21) — **DONE** (find_best_hybrid_tcga.py: added to _REP_DISPATCH and 'all' preset, 34→36 reps)
- [x] Add LightGBM to reaction yield AND DTA survey pipelines — **DONE** (run_reaction_yield_survey.py: added LGBMRegressor; dta_baseline_survey.py: added LGBMRegressor)

**NOT in Phase 0 (moved):** SNS wiring — needs rep_factory for per-fold fitting → moved to Phase 1B.

**Expected impact:** Fixes corrupted representation matrices for MACCS/PDV/Mol2Vec,
prevents silent policy violations from random KFold fallback, removes ESOL/MoleculeNet
dead weight, ensures DDG dataset only contains Ab-Ag complexes. These fixes ensure
the baseline surveys produce correct numbers that KIRBy's hybrid improvements can be
measured against.

### Phase 1: Evaluation Framework Overhaul
**Scope:** ~2-3 sessions. Changes HOW results are measured. Must be complete before
any survey re-runs, because old results won't be comparable to new ones.
**Read:** `issues_benchmarks.md` (cross-cutting issues), `issues_splitting.md`,
`issues_preprocessing.md` (C3-C6, H5, H6)

#### 1A. Metrics (every domain affected)
- [x] Add Spearman ρ to all regression tasks — **DONE** (molecular: secondary_metrics=['r2','spearman','rmse']; DTA: secondary_metrics=['r2','spearman','rmse']; TCGA regression: secondary_metrics=['rmse','spearman']; antibody: spearman computed per fold)
- [x] Add RMSE to all regression tasks (primary for molecular per Tropsha "Beware of R²") — **DONE** (molecular: metric='rmse'; DTA/TCGA/antibody: RMSE computed per fold)
- [x] Add Concordance Index for DTA (every DTA paper reports CI) — **DONE** (`metric='ci'` in DTA survey, `_concordance_index()` in find_best_hybrid.py)
- [x] Add Pearson r for DDG (field standard) — **DONE** (added `pearsonr` to antibody_baseline_survey.py evaluate_rep/evaluate_ft_rep, stored in results)
- [x] Switch TCGA classification from accuracy to AUROC (binary) / weighted F1 (multi-class) (Issue 24) — **DONE** (TCGA survey: binary→metric='auc', multi→metric='f1_weighted', accuracy as secondary)
- [x] Keep R² everywhere as secondary — **DONE** (all surveys compute R²)
- [x] **Reference:** `issues_benchmarks.md` cross-cutting Issue 1, `issues_dta_reaction.md` §1.3

#### 1B. Data Leakage Fixes
- [x] Move PCA fitting inside CV folds (molecular.py, genomic.py) — **DONE** (added `make_pca_factory()` in molecular.py, `make_gene_pca_factory()` in genomic.py; survey scripts use rep_factories instead of pre-computed base_reps)
- [x] Move StandardScaler inside CV folds (genomic.py — ALL TCGA reps affected) — **DONE** (added `make_hvg2000_factory()`, `make_gene_set_score_factory()`, `make_weighted_score_factory()`, `make_combined_pathway_factory()` in genomic.py; TCGA survey uses factories for: gene_hvg2000, gene_pca100, gene_hallmark, gene_progeny, gene_kegg, gene_dorothea, gene_combined_pathway, gene_reactome, gene_oncogenic)
- [x] Move rank/binary_mean transforms inside CV folds (molecular.py) — **DONE** (added `make_pdv_binary_mean_factory()` and `make_pdv_rank_factory()` in molecular.py; molecular survey uses factories)
- [x] Move entropy calculation inside CV folds (antibody.py) — **DONE** (added `make_esm2_entropy_factory()` in antibody.py; antibody survey evaluates via `evaluate_ft_rep()` per-fold)
- [x] Wire SNS into molecular 'all' preset AND ensure per-fold fitting via rep_factory (Issue 14) — **DONE** (added `make_sns_factory()` in molecular.py; SNS added to REP_PRESETS['all'] as factory rep)
- [x] Move `tune_models()` scaler inside CV — **DONE** (wrap each model in `Pipeline(StandardScaler(), model)` inside `RandomizedSearchCV`, extract tuned model for return; `issues_preprocessing.md` H4)
- [x] **Reference:** `issues_preprocessing.md` C3-C6, H4-H6
- [x] **gene_immune factory conversion** — **DONE** (`make_gene_immune_factory()` in genomic.py: ssGSEA pre-computed once (sample-independent), ESTIMATE scores computed per-fold on train subset only. Wired in FACTORY_REPS in find_best_hybrid_tcga.py.)
- [x] **gene_nmf50, gene_ica50, gene_quantile_rank, gene_corr_modules, gene_autoencoder50, gene_sparse_pca50** — **DONE** (all 6 converted to per-fold factories in `genomic.py`: `make_nmf_factory`, `make_ica_factory`, `make_sparse_pca_factory`, `make_quantile_rank_pca_factory`, `make_corr_modules_factory`, `make_autoencoder_factory`. Added to `FACTORY_REPS` and `get_rep_factories()` in `find_best_hybrid_tcga.py`.)

**Architecture:** All per-fold fixes use the existing `rep_factories` mechanism in `find_best_hybrid.py`. Factory reps are defined in `FACTORY_REPS` sets in each survey script. The `get_rep_factories()` function in each survey creates the factory callables. Original `create_*()` functions are preserved for backward compatibility and single-use scenarios.

**Survey integration:**
- `tests/find_best_hybrid_molecular.py`: `FACTORY_REPS` set added. `get_rep_factories()` creates per-fold factories for pdv_binary_mean, pdv_rank, sns, and all *_pca50 reps. Merged with fine-tuning factories before `find_best_hybrid()` call.
- `tests/find_best_hybrid_tcga.py`: `FACTORY_REPS` set added. `get_rep_factories()` creates per-fold factories for gene_hvg2000, gene_pca100, and 7 pathway scoring reps. Merged with fine-tuning factories before `find_best_hybrid()` call.
- `tests/antibody_baseline_survey.py`: esm2_entropy_top20 moved from base rep loop to factory-based evaluation using `evaluate_ft_rep()`.
- `tests/find_best_hybrid_dta.py`: `FACTORY_REPS` set added. `get_rep_factories()` creates per-fold factories for drug_pdv_binary_mean, drug_pdv_rank, and 5 drug_*_pca50 reps. Pre-computes base embeddings on unique drugs, expands to pair-level, then fits PCA/rank/mean per fold. Merged with fine-tuning factories before `find_best_hybrid()` call.
- `tests/run_reaction_yield_survey.py`: **Verified clean** — `evaluate_split()` already fits StandardScaler per-split (lines 90-92). No factory conversion needed.
- **BOM impact:** All re-runs in Phase 5 will use the leakage-free evaluation. Results for PCA/rank/pathway reps will change (slightly lower, more honest). BOM must be rebuilt after Phase 5 re-runs.

#### 1C. Splitting Fixes
- [x] Reaction yields BH: GroupKFold on additive (23 groups) — **DONE** (`reaction_yields.py`: `load_buchwald_hartwig()` returns `groups` array keyed on additive SMILES; `run_reaction_yield_survey.py`: uses GroupKFold via `_make_splits()`, falls back to random if no groups. Old random splits preserved via `--random-splits` flag for comparison.)
- [x] Reaction yields Suzuki: GroupKFold on ligand (12 groups) — **DONE** (`reaction_yields.py`: `load_suzuki()` now downloads original Perera 2018 Excel SI (`aap9112_Data_File_S1.xlsx`), extracts `Ligand_Short_Hand` as group key. 12 ligand groups × 480 reactions each. Falls back to TSV-only loading if Excel fails. `run_reaction_yield_survey.py` picks up `groups` via `_make_splits()` automatically.)
  - **Reference:** `issues_splitting.md` §7, `issues_dta_reaction.md` §2.3
- [x] Antibody DMS: position-based modulo splits for intra-family — **DONE** (`antibody_baseline_survey.py`: added `_assign_position_groups_modulo()` using ProteinGym `fold_modulo_5` scheme — groups by `mutation[0]['pos'] % n_folds`. DMS datasets in `DATASET_REGISTRY` changed from `'identity'` to `'position_modulo'`. Falls back to identity if no mutation data or <3 groups.)
  - FLAb `mutations` field already contains `{'pos': int, 'wt_aa': str, 'mut_aa': str, 'chain': str}` per sample — no alignment needed, positions come from `_extract_mutations()` in `flab.py`
  - AbAgym also returns full mutation data with PDB-mapped positions
  - Contiguous split also implemented (`_assign_position_groups_contiguous()`) but not wired as default — available for supplementary comparison
  - **Wildtype identification (L2):** For DMS datasets the most-common sequence heuristic is reliable because the wildtype appears multiple times as a control. For SSM data where every sequence is unique, this could fail — but the position extraction still works because it compares each sequence against the mode, and even if the "wildtype" is a variant, the relative positions are consistent. Still a minor risk to monitor.
  - **Reference:** `issues_splitting.md` §2, `issues_antibody.md` §5
- [x] Small inter-family datasets (Issue 5): **INVESTIGATED, DOCUMENTED** — For n=130-217 with 2560D embeddings (e.g., `flab_thermo_jain`, `flab_aggregation_jain`), the identity GroupKFold is acceptable because these ARE genuinely diverse sequences from different antibodies. The high n_groups/n_samples ratio (near 1:1) means GroupKFold ≈ LOO, which gives high variance. Mitigation: Phase 2 model expansion (RF/XGBoost handle high-D better than Ridge). No splitting fix needed — the problem is dimensionality vs sample size, not the split strategy.
- [x] Document DDG 56% leakage limitation (Bushuiev 2024) — **DOCUMENTED** below in Issue 3 and Splitting section. KIRBy's Ab-Ag-only filtering and deduplication with AB-Bind partially mitigate the 56% leakage found in full SKEMPI. Dataset too small (~20 PDB complexes) for CATH superfamily splitting. Acknowledged as known limitation.
  - **Reference:** `issues_splitting.md` §4

#### 1D. Preprocessing
- [x] Molecular: add salt stripping, neutralization, tautomer standardization (QSAR-Ready Workflow, Mansouri et al. 2024) — **DONE** (`molecular.py`: added public `standardize_smiles()`; `find_best_hybrid_molecular.py` delegates to it)
- [x] DDG: add reverse mutation augmentation (DDG(M→W) = -DDG(W→M), doubles training data) — **DONE** (`ddg_combined.py`: `load_ddg_combined_dataset(augment_reverse=True)` generates reversed entries with negated DDG, swapped WT↔MUT, same pdb_id for GroupKFold)
- [x] BOS/EOS token exclusion from mean pooling — **DONE** (fixed in ALL affected modules):
  - `antibody.py` `_mean_pool()`: added `exclude_bos`/`exclude_eos` params; callers updated (ESM-2/AMPLIFY: both, ProtT5: EOS only, IgBERT: both)
  - `antibody.py` `embed_esmc`: slices `[:, 1:-1, :]` to exclude BOS/EOS
  - `protein.py` ESM-C: same `[:, 1:-1, :]` slice
  - `protein.py` AMPLIFY: mask manipulation to exclude BOS+EOS
  - `protein.py` ESM-2: already correct — used as reference
- [x] Outlier clipping: make optional or remove (currently inflates molecular R²) (Issue 18) — **DONE** (`find_best_hybrid_molecular.py`: `--no-clip` flag; threaded through `load_dataset` → `_load_openadmet_endpoint`)
- [x] IgBERT mean prediction fix: try [CLS] pooling instead of mean-pool-all-tokens (Issue 4) — **DONE** (`antibody.py`: added `embed_igbert_cls()` with [CLS] pooling, registered as `'igbert_cls'`; existing `embed_igbert` now excludes BOS/EOS)
- [x] TCGA gene variance filtering — **DOCUMENTED** as acceptable: HVG2000 already does variance filtering; other reps use their own dimensionality reduction
- [x] **Reference:** `issues_benchmarks.md` §1.3, §3.4, `issues_preprocessing.md` M6-M7, L10

**Expected impact on results after Phase 1:**
- Molecular: R² will DECREASE slightly (no leakage, no clipping, proper preprocessing)
- Antibody DMS: R² will DECREASE (harder position-based splits)
- Reaction: R² will DECREASE SIGNIFICANTLY (0.93 → ~0.7-0.8 with proper splits)
- TCGA: R² will decrease slightly (no leakage in PCA/scaling)
- DTA: R² unchanged but now CI is reported (which is what matters)
- DDG: unchanged but Pearson r reported (which gives positive correlation vs negative R²)
- IgBERT: may improve from near-zero if [CLS] pooling helps
- **This is GOOD.** Lower but correct numbers > higher but leaky numbers.

### Phase 2: Model Suite Expansion ✅ COMPLETE
**Scope:** ~2 sessions. Changes WHICH downstream models are used to evaluate representations.
These models are NOT KIRBy's product — they are the measurement instrument. A weak instrument
(Ridge alone) can't reveal representation quality. A diverse instrument set (Ridge, RF, XGBoost,
GP, MLP, CatBoost, DNN) ensures that when a hybrid outperforms its components, it's a real
improvement, not a model limitation artifact.
**Read:** `issues_forward_compat.md` Part 2 (model audit), `issues_antibody.md` §4

- [x] Add Gaussian Process — **DONE.** sklearn GPR with RBF+WhiteKernel, n_restarts=3. Size guard _max_samples=5000 auto-skips large datasets (Davis 30K, KIBA 117K). HP search: alpha regularization. Classifier variant included.
  - Advanced GP kernels (GAUCHE Tanimoto, Kermut composite) deferred to Phase 4.
  - **Reference:** `issues_forward_compat.md` §2.2 (GPR entry)
- [x] Upgrade MLP defaults — **DONE.** VespaG-style: hidden_layer_sizes=(256,), early_stopping=True, validation_fraction=0.1. HP search updated to [(128,), (256,), (256,128), (512,)].
  - **Reference:** `issues_forward_compat.md` §2.2 (MLP entry)
- [x] Add CatBoost — **DONE.** iterations=500, lr=0.1, depth=6. ImportError fallback to LightGBM. HP search: iterations, depth, lr, l2_leaf_reg.
- [x] Add DNN — **DONE.** PyTorch sklearn-compatible wrapper in `src/kirby/models/dnn.py`. (256,128,64) hidden, dropout=0.2, Adam+ReduceLROnPlateau+early stopping, target z-normalization. __init__ stores only scalars (deepcopy-safe). Skipped in tune_models (too slow for RandomizedSearchCV). Both DNNRegressor and DNNClassifier.
  - **Reference:** `issues_forward_compat.md` §2.2 (DNN entry), `issues_molecular.md` Issue 2
- [x] CNN prediction head on embeddings (Issue 1 — user specifically requested) — **DONE** (`src/kirby/models/cnn.py`: multi-scale 1D CNN with kernel sizes 3/7/15, BatchNorm, AdaptiveAvgPool, FC head. CNNRegressor + CNNClassifier. Registered as 'cnn' in DEFAULT_MODELS, tier='long'. Treats embedding dims as 1D signal — per-residue support deferred.)
  - **Reference:** `issues_forward_compat.md` §2.2 (1D-CNN entry)
- [x] SVR size guard — **DONE.** Added _max_samples=10000 to existing SVM entry. Auto-skipped for large DTA datasets.
  - **Reference:** `issues_forward_compat.md` §2.2 (SVR entry)
- [ ] TabPFN: experimental addition — **DEFERRED.** User said "discuss first." Not standard enough for thesis yet.
  - **Reference:** `issues_forward_compat.md` §2.2 (TabPFN entry)
- [x] Ridge, RF, XGBoost, LightGBM, KNN — unchanged as planned.
- [x] Added `get_model()` public API — survey scripts now use centralized model definitions instead of duplicating.
- [x] Survey scripts updated: antibody (default models: ridge,lightgbm,gp,mlp), DTA, reaction yields all use `get_model()`.

**Expected impact:** Representations that looked bad under Ridge may look good under GP/MLP.
The model hierarchy matters: Ridge Spearman ~0.48 vs GP ~0.74 on SAME embeddings (ProteinGym).
This is the single biggest improvement for antibody/protein fitness domains.

### Phase 3: Representation Expansion (per domain)
**Scope:** ~3-4 sessions (one per major domain). Adds missing representations and datasets.
This is the core of what KIRBy does — expanding the pool of representations available
for hybridization. Each new rep type is a potential augmentation source. The surveys
establish baselines for individual reps; KIRBy's hybrid search then finds which
combinations improve on those baselines. More reps = more augmentation candidates =
stronger thesis.
**Read:** domain-specific issues files

#### 3A. TCGA (~2 sessions)
Target: 36 → 55-65 reps. Broken into tiers by external dependency.

**Tier 1 — Code from existing MC3 MAF data (no external deps):** ✅ DONE
- [x] `mut_tmb` (5D): Tumor mutational burden — total, log10, per-Mb, missense, truncating counts
- [x] `mut_driver` (46D): 40 binary driver gene indicators (Bailey 2018 + COSMIC CGC) + 6 OG/TSG aggregates
- [x] `mut_pathway` (50D): Pathway-aggregated mutations — fraction of Hallmark pathway genes mutated per sample
- [x] `mut_pca50` (50D): PCA of full binary mutation matrix (genes with >=5 mutated samples)
- [x] `mut_oncogene_ts` — folded into `mut_driver` as cols 40-45 (OG/TSG/missense/truncating aggregates)
- [x] Wire all into `_REP_DISPATCH` and add to all/genomic_all/genomic_diverse presets
- [x] Refactored shared `_load_nonsilent_mutations()` helper + `_NON_SILENT_TYPES`/`_TRUNCATING_TYPES`/`PAN_CANCER_DRIVERS` constants

**Tier 2 — Python packages (pip install, run on existing data):** ✅ DONE
- [x] `sig_sbs` (~78D): COSMIC v3.4 SBS mutational signatures via SigProfilerAssignment on MC3 MAF
- [x] `sig_dbs` (~11D): Doublet-base substitution signatures
- [x] `sig_id` (~18D): Small insertion/deletion signatures
- [x] Added `_download_mutations_full()` in tcga.py (preserves chr/pos/ref/alt for SPA)
- [x] Added `_prepare_sigprofiler_input()` + `_run_cosmic_fit()` helpers in genomic.py
- [x] Wired into `_REP_DISPATCH` and added to all/genomic_all presets
- [x] Added SigProfilerAssignment to setup.py "tcga" extras
- [x] Requires one-time: `pip install SigProfilerAssignment` + `install.install('GRCh37')`

**Tier 3 — Pre-computed downloads (write loaders for published data):** ✅ DONE
- [x] `immune_cibersort` (~22D): CIBERSORTx LM22 fractions from Thorsson et al. (Immunity 2018)
- [x] `immune_xcell` (~64D): xCell scores from Thorsson et al. (Immunity 2018)
- [x] `stemness_mrnasi` (2D): mRNAsi + EREG-mRNAsi from Malta et al. (Cell 2018)
- [x] `purity_consensus` (~4D): ABSOLUTE purity/ploidy/doublings from PanCanAtlas Xena
- [x] `heterogeneity_math` (1D): MATH score computed from MC3 DNA_VAF (no download needed)
- [x] `msi_status` (3D): one-hot MSI from clinical phenotype data
- [x] `hrd_score` (4D): LOH + TAI + LST + total from Knijnenburg et al. (Cell Reports 2018)
- [x] `cimp_status` (2D): CpG island methylator phenotype from existing methylation data
- [x] Added `_download_pancanatlas()` generic helper + 4 specific downloaders in tcga.py
- [x] Extended `_download_mutations_full()` to include DNA_VAF column (for MATH)
- [x] All download caches go to /Volumes/seagate/kirby/tcga/pancanatlas/
- [x] Wired into `_REP_DISPATCH` and added to all/genomic_all/genomic_diverse presets
- [x] TCGA total: 43 → 51 reps

**Tier 4 — WSI foundation models:** ✅ DONE (wired models) / DEFERRED (tile-dependent)
- [x] `wsi_titan` (768D): TITAN slide-level embeddings — already has download/prep code + `_WSI_DIMS` entry, now wired into dispatch + presets
- [x] `wsi_uni2h` (1536D): UNI2-h mean-pooled patch embeddings — already has download/prep code + `_WSI_DIMS` entry, now wired into dispatch + presets
- [x] TCGA total: 51 → 53 reps
- DEFERRED: Virchow2 (1280D) — no pre-computed features, needs raw SVS slides + tile extraction
- DEFERRED: CONCH (512D) — TITAN already uses CONCH v1.5 internally; standalone needs tile extraction
- DEFERRED: BMFM-RNA (IBM) — already have 5 RNA FMs, low marginal value
- DEFERRED: CancerFoundation — scRNA-only, no bulk RNA pathway
- DEFERRED: GeneBag — no public code/weights

**Tier 5 — Multi-omic feature types (code from existing data):** ✅ DONE
- [x] `splicing_psi50` (50D): Splicing factor gene expression + PCA — 123 curated splicing factors (SF3B1, U2AF, SRSF family, hnRNPs, PRPF complex) + known PSI event genes (VEGFA, BCL2L1, FN1, CD44). Factory pattern.
- [x] `lncrna_pca50` (50D): 142 cancer-associated lncRNA expression + PCA — HOTAIR, MALAT1, NEAT1, H19, SNHG family, LINC00xxx series. Factory pattern.
- [x] `gsva_hallmark` (~50D): ssGSEA with MSigDB Hallmark gene sets — proper statistical enrichment vs simple mean scoring. Factory pattern.
- [x] `promoter_meth` (50D): Promoter-region methylation (TSS200/TSS1500 probes) + PCA — 119 curated cancer promoter probes. Factory pattern.
- [x] TCGA total: 53 → 57 reps

**After all tiers:**
- [x] Modality ablation analysis — **DONE** (`tests/experiments/complexity/c12_modality_ablation.py`: evaluates every modality group individually, all pairwise combos, and full combination. Annotates which modalities help/hurt per TCGA task.)
- [ ] **Reference:** `issues_tcga.md` (full tier list, 25 repos)

#### 3B. Molecular — DONE
- [x] Add Mordred 2D descriptors (~1613D, superset of PDV, fills Caco2 gap)
- [x] Add rxnfp for reaction yields (IBM RXN, 256D learned reaction fingerprints)
- [x] Add one-hot component encoding for reactions (factorial baseline, ~46D BH / ~40-60D Suzuki)
- [x] Wire 4 existing unreferenced reaction reps (concat_maccs, concat_pdv, product_ecfp4, reactant_ecfp4) — reaction reps 2→8
- [x] Add mordred_pca50 factory rep for molecular survey
- [x] Add Polaris Hub ADME-Fang benchmarks (biogen/adme-fang-v1): adme_hlm (~3087), adme_rlm (~3054), adme_mdr1 (~2642), adme_solubility (~2173). Molecular datasets 4→8.
- [x] Add Avalon fingerprints (2048-bit) — strong Caco2 performer per CaliciBoost benchmark. Wired into REP_PRESETS['all'].
- [x] Add `deduplicate_by_inchikey()` utility — removes duplicate molecules (same InChIKey, different SMILES) at dataset level. Prevents same molecule appearing in train+test folds.
- [x] Add Doyle lab HTE plates dataset (`load_bh_doyle_plates()`, 4,599 reactions) — different from existing BH. Additive-based GroupKFold. Wired in reaction yield survey.
- ELN Buchwald-Hartwig — informative stub (Saebi et al. JCIM 2023 data not publicly available)
- DEFERRED: Chemprop/D-MPNN — incompatible with KIRBy paradigm (E2E model, no fixed rep). KIRBy's frozen GNN embeddings (GROVER, MHG-GNN, GraphMVP, MolCLR, SchNet, DimeNet++) serve the analogous role.
- DEFERRED: PharmaBench, Benchmark-ADMET-2025 — additional ADMET benchmarks with locked splits/leaderboards. Lower priority now that Polaris ADME is integrated.
- **Reference:** `issues_molecular.md` Issues 5, 10, `issues_dta_reaction.md` §2.3

#### 3C. Antibody — DONE
- [x] Add zero-shot fitness score augmentation (Issue 25) — `esm2_llr` already existed; added `esmc_llr` and `antiberty_llr` (3D each, masked marginal scoring). DONE Phase 3C.
- [x] Layer selection as hyperparameter (Issue 26) — added `esm2_layer24` (single layer 24, 2560D) and `esm2_early` (layers 0-16 avg, 2560D). With existing `esm2_midlayers` (20-33), now 4 ESM-2 layer options. DONE Phase 3C.
- [x] Paired-aware embedding — added `esm2_concat` (H+L as single sequence, 1280D). DONE Phase 3C.
- [x] CDR3-only pooling — added `esm2_cdr3` (IMGT 105-117 only, 2560D). DONE Phase 3C.
- [x] Ab-RoBERTa — added `abroberta` (1536D) + `abroberta_cdr` (1536D CDR-pooled). Was documented in SETUP.md but never implemented. DONE Phase 3C.
- DEFERRED: Inverse folding models (ProteinMPNN, ESM-IF, AntiFold) — require 3D structure input, architecturally incompatible with KIRBy's sequence-in/vector-out pipeline.
- DEFERRED: ProteinGym (217 DMS assays), FLIP2 — general protein, not antibody-specific. Would be a "Phase 3E: General Protein Fitness" effort.
- DEFERRED: AbBiBench (184K), AbRank (380K+) — substantial data integration, different task framing. Separate session.
- DEFERRED: ESM-1v LLR — older model (2021), superseded by ESM-2. Not in codebase.
- **Totals:** 27→33 ANTIBODY_REPS (+6), 4→6 MUTATION_REPS (+2) = 31→39 total.
- **Reference:** `issues_antibody.md` §10

#### 3D. DTA — DONE
- [x] Wire cold-drug evaluation — DONE (`--cold-setting` flag in both survey scripts)
- [x] Add BindingDB_Kd via TDC — DONE (TDC approved for data loading; `_load_from_tdc()` helper)
- [x] Document SOTA CI gap — DONE (dta.py docstring)
- Deferred: Metz — can be trivially added later via same TDC loader (add to `_DATASETS` + `tdc_name_map`)
- Re-categorized: DACUM — DDG source, not DTA. Moved to Phase 3C.
- [x] **Reference:** `issues_dta_reaction.md` §1.3 Issues 3-5

### Phase 4: Forward Compatibility Experiments (CORE THESIS)
**Scope:** ~2-3 sessions. This is not optional — it's the thesis's central argument.
**Read:** `issues_forward_compat.md` Part 1 (all 17 models + 5 experiments)

**Sub-phases:**
- **Phase 4A (DONE):** Sequence-based model integrations — write embedding code, register reps, update deps
  - [x] MiniMol (molecular, pip install minimol)
  - [x] GP-MoLFormer (molecular, HuggingFace ibm-research/GP-MoLFormer-Uniq)
  - [x] MolE (molecular, GitHub + manual weights, CC-BY-NC 4.0)
  - [x] MINT (protein/DTA, GitHub + manual checkpoint)
  - [x] IgT5 (antibody, HuggingFace Exscientia/IgT5)
  - [ ] S2ALM (antibody) — **BLOCKED**: weights not publicly available. Stub written, not registered.
- **Phase 4B (DONE):** Structure-dependent integrations
  - [x] ESM3 (protein + antibody, esm package, 1536D)
  - [x] SaProt (protein + antibody, ProstT5 for 3Di, 1280D)
  - [x] Boltz-2 protein (DTA, CLI wrapper, 384D) — molecular deferred
- **Phase 4C (future):** Run all surveys on server, analyze hybrid results

This is where KIRBy proves its value as a representation enhancement framework.
The protocol for each experiment:
1. A new foundation model is released (after KIRBy was built)
2. Extract embeddings from it (zero KIRBy code changes required)
3. Register as new rep, run baseline survey → individual rep score
4. Run KIRBy hybrid search → find which existing reps augment the new model
5. Result: new model alone < new model + KIRBy augmentation

This proves that KIRBy enhances existing pipelines — you take whatever representation
you're already using, and KIRBy finds complementary information from other rep types
to improve the input your downstream model receives.

**Priority order:**
- [ ] **Experiment 1: MINT** → DTA + antibody (Jan 2026, PPI-aware, 1280D, built on ESM-2)
- [ ] **Experiment 2: SaProt** → DTA + antibody (ICLR 2024, structure-aware, drop-in ESM-2 replacement)
- [ ] **Experiment 3: MolE / MiniMol** → molecular (new foundation models, easy pip install)
- [ ] **Experiment 4: IgBERT-new / S2ALM** → antibody (paired + multimodal AbLMs)
- [ ] **Experiment 5: Boltz-2** → cross-domain (structure prediction as universal embedder)

**What each experiment proves for the thesis:**
- A model released AFTER KIRBy was built plugs in with zero code changes
- Hybridizing it with older representations (PDV, ECFP4, etc.) improves on both
- The framework is genuinely forward-compatible — not just compatible with models it was designed around

**Reference for all:** `issues_forward_compat.md` §1.12 (detailed protocols per experiment)

### Phase 5: Re-run All Surveys, Documentation, and Rebuild Results
**Scope:** ~2-3 sessions (server time + documentation). Depends on ALL previous phases.
**Read:** `BILL_OF_MATERIALS.md` (current results to compare against)

#### 5A. Re-run surveys — SLURM scripts ready
**Scripts:**
- `tests/slurm_scripts/submit_phase5_baselines.sh` — all baseline surveys (~25 jobs)
- `tests/slurm_scripts/submit_phase5_hybrids.sh` — hybrid search per domain (~15 jobs)
- `tests/slurm_scripts/submit_forward_compat.sh` — 5 thesis experiments (~35 jobs)
- `tests/verify_phase5_readiness.py` — pre-flight check for imports/registries

- [ ] Re-run molecular survey (8 datasets, 36 reps, 11 models, RMSE primary)
- [ ] Re-run antibody survey (25 datasets, 40 reps, position-modulo DMS splits)
- [ ] Re-run DTA survey (davis/kiba/bindingdb, 37 reps, CI primary, cold-target+cold-drug)
- [ ] Re-run TCGA survey (3 datasets, 57 reps, AUROC/F1w primary, no leakage)
- [ ] Re-run reaction yield survey (3 datasets, 8 reps, GroupKFold splits)
- [ ] Run forward compatibility experiments (MINT, SaProt, MolE/MiniMol, IgT5, Boltz-2)

#### 5B. Documentation and context
- [ ] Rebuild BILL_OF_MATERIALS.md with all new results
- [ ] Document which results changed and WHY (thesis narrative)
- [ ] Add external SOTA validation numbers for context alongside KIRBy results:
  - Molecular: Chemprop, ADMET-AI, Polaris leaderboard numbers
  - Antibody: ProteinGym supervised leaderboard
  - DDG: FoldX, Flex ddG, CATH-ddG published numbers
  - DTA: DeepDTA, GraphDTA CI numbers (warm vs cold)
  - Reaction: DRFP/rxnfp published R² values (random AND OOS)
  - **Reference:** `issues_benchmarks.md` cross-cutting Issue 4
- [ ] Document known limitations and expected-bad results:
  - DDG R²=-0.137 is expected (sequence-only, leave-complex-out) (Issue 3)
  - FLAb inter-family near-zero is expected per FLAb2 (Issue 6)
  - AbAgym poor ML is a published finding (Issue 6)
  - hERG R²=0.552 is plausible for small scaffold-split dataset
  - QM9 R²=0.139 is expected for frozen 2D + scaffold CV
  - GroupKFold averages ~0.04-0.08 higher than single 80/10/10 split (Issue 19)
  - DTA SOTA CI numbers are inflated by warm-start splits
- [ ] Document intentional design decisions:
  - No deep interaction models for DTA (by design — thesis is about representations)
  - ESM-2 truncates at 1022 AA, ESMFold at 512 (`issues_preprocessing.md` M4, M9)
  - AbAgym double normalization (MinMax then z-score) is defensible but should be noted (`issues_preprocessing.md` M2)

**Expected outcome:** Many baseline numbers will be lower (harder splits, no leakage). Some will be
higher (better models revealing rep quality). ALL will be correct and defensible. The individual
rep baselines establish what existing pipelines achieve. KIRBy's hybrid results then show, on
the same honest evaluation, that augmenting representations with complementary information
improves on those baselines. The thesis narrative is: "here is what each representation does
on its own under standard evaluation; here is what KIRBy does by enhancing the input."

### Phase Dependencies

```
Phase 0 (bug fixes) ──→ can start immediately
Phase 1 (evaluation) ──→ depends on Phase 0
Phase 2 (models) ──────→ can start after Phase 1A (metrics)
Phase 3 (reps) ────────→ can start after Phase 0 (TCGA dead code)
Phase 4 (forward compat) → depends on Phases 1-2 (need correct evaluation)
Phase 5 (re-run + docs) → depends on ALL previous phases
```

Phases 2 and 3 can run in parallel after Phase 1 is done.
Phase 4 can start as soon as Phase 2 is done (models ready).

---

## PROCESS RULES (do not delete on compaction)

1. **All research agent outputs MUST be written to files immediately** — never hold in context
2. **Each domain gets its own issues file** (issues_molecular.md, issues_antibody.md, etc.)
3. **ISSUES.md is expanded as findings come in** — not summarized after
4. **Every repo examined must be listed** with what was found
5. **Every finding must reference specific KIRBy code** (file + line) and specific external code
6. **No vague suggestions** — every issue is: what's wrong, where, what others do, what to fix
7. **Two categories of repos**: (a) source code for models/datasets/methods, (b) research implementations using those methods
8. **Do not hallucinate findings** — if research was lost to compaction, redo it
9. **Do not claim work is done if it isn't**
10. Previous research was lost to context compaction. This session redoes it from scratch.

---

## USER DIRECTIVES (captured from conversation — do not lose on compaction)

These are the user's explicit requirements stated across multiple messages. Every one of
these must be addressed in the final implementation plan. If any are dropped or forgotten,
the user will be justifiably furious.

### Hard constraints (non-negotiable)
1. **NO random splitting** — lab group forbids it. User "lives in fear" of scrutiny. Every domain needs a defensible non-random split.
2. **NO TDC** — banned by group decision (Pat Walters disapproves)
3. **NO MoleculeNet** — banned by group decision
4. **Baselines = individual reps, NOT concat** — emphatic directive
5. **Scaffold splitting for molecular** — PhD group requirement

### Models
6. **Ridge is useful but NOT sufficient** — user says "that was arbitrary, YOU pushed it on me"
7. **Add CNN prediction head** — specifically requested (ProteinGym uses this)
8. **Add Gaussian Process** — specifically requested (Kermut uses this, SOTA on ProteinGym)
9. **Add MLP** — specifically requested
10. **User is "VERY open to other models"** — SVM, CatBoost, DNN also in their qsar_qm_models repo
11. **Keep RidgeCV, RF, XGBoost, LightGBM** — don't remove, just add to them

### Metrics
12. **Keep R² everywhere** but add domain-appropriate metrics:
    - Spearman ρ for antibody/DMS/fitness
    - RMSE for DDG and general regression
    - Concordance Index for DTA
    - Pearson r for DDG
13. **R² only for antibody** — NO Spearman ρ was an earlier directive, but user later said to ADD Spearman. Latest directive: keep R², add Spearman ρ and RMSE where applicable.

### Forward compatibility (HUGE thesis point)
14. **Prove KIRBy's forward compatibility** — take brand-new models, extract embeddings, plug in with zero framework changes
15. **Use AlphaFold** — user specifically wants this integrated ("Oh and I want to use AlphaFold somewhere, if possible")
16. **Brand-new models for the experiment**: MINT (Jan 2026), Boltz-2, ESM3, Chai-1, ESM-C 6B mentioned
17. **Experiment design**: new_model → embeddings → KIRBy baseline → augment with existing reps → show hybrid improvement
18. **This is how KIRBy demonstrates its value**: "if a new model or new technique is created, it can be converted into an embedding and used in KIRBy's framework"

### Research approach
19. **KIRBy enhances existing research, doesn't rewrite it** — understand standard benchmarks and methodologies first
20. **Two categories of GitHub repos to examine**: (a) source code for models/datasets/methods, (b) research implementations USING those methods
21. **Compare against the user's qsar_qm_models repo** — they've been doing this for 3 years
22. **Comprehensive repo lists required** — user will check and will be upset if too short/shallow
23. **Preprocessing must be audited** — user suspects corners have been cut
24. **Benchmarking focus** — understand how real research works in each domain before changing anything
25. **No code changes until research is complete and discussed**

### TCGA specifically
26. **User feels TCGA has insufficient work** — wants MORE features and representations
27. **Somatic mutations, TMB, CIBERSORT, stemness, etc. are all missing**

### Data
28. **Investigate combining SKEMPI data** — can more DDG data be gathered?
29. **Investigate expanding DTA** — beyond Davis/KIBA
30. **Test ALL possible reps** — user wants complete coverage, not partial

### Models (updated)
35. **SVR: selective use only** — "fine in certain use cases but don't make it standard across all surveys" (2026-03-10)
36. **This is a massive multi-session refactor, not a quick fix** — user explicitly said separate sessions with implementation plans connecting back to research (2026-03-10)

### What NOT to do
31. **DO NOT touch noise robustness scripts**
32. **DO NOT LIE ABOUT HAVING DONE SOMETHING** — user has caught this multiple times
33. **DO NOT blame scaffold splitting for bad results** — it's the default everywhere
34. **DO NOT use overly jargoned language** — user knows the field but wants clear plain language
37. **DO NOT collapse this into a checklist on compaction** — the user has caught this pattern and is furious. The depth of the research IS the point. Read the issues_*.md files.
38. **DO NOT start implementing without discussing phase plan with user** — "let's do research, talk it through, then move to implementation plan" (emphatic, repeated)

---

## Research files (written by background agents — do not lose)

Each domain's detailed findings are in separate files to survive context compaction.
**ALL 8 FILES ARE COMPLETE.** Total: 3,589 lines of research. Do NOT redo this work.

- `issues_molecular.md` (293 lines) — 15 repos, ground truth comparison vs qsar_qm_models, 13 issues
- `issues_antibody.md` (350 lines) — 14 repos, DDG verdict (NOT a bug), splitting, model hierarchy
- `issues_tcga.md` (567 lines) — 25 repos, 20+ missing rep types, full tier list
- `issues_dta_reaction.md` (409 lines) — 18 repos, CI metric gap, reaction splits with concrete plan
- `issues_forward_compat.md` (558 lines) — 17 new models with embedding extraction code, 5 designed experiments, full downstream model audit with literature ranking
- `issues_splitting.md` (477 lines) — per-domain splitting audit, 15+ references, code sketches
- `issues_preprocessing.md` (303 lines) — line-by-line KIRBy code audit, 8 CRITICAL + 7 HIGH + 8 MEDIUM + 10 LOW
- `issues_benchmarks.md` (632 lines) — standard protocols, preprocessing, metrics, SOTA per domain, gap analysis tables

---

## Core Thesis: Forward Compatibility

KIRBy's key value proposition is **forward compatibility**: any new model, technique, or
data modality can be converted into a fixed-size embedding vector and immediately plugged
into the framework. A new GNN architecture? Extract embeddings, augment with non-graph
features (physicochemical descriptors, fingerprints), and the hybrid outperforms either alone.

This means:
1. The downstream model should be simple enough that the representation does the work
2. Performance differences between representations are meaningful (not masked by a complex predictor)
3. The augmentation/hybrid mechanism (concatenating vectors) is the core contribution
4. When a rep performs poorly, the question is: "is the representation missing information?"
   not "is the model too weak?"

However, we still need competent downstream models. Ridge alone leaves performance on the
table in domains where the embedding-to-label mapping is nonlinear. The solution is not one
complex model, but a small set of models (Ridge, RF/XGBoost, MLP, GP) that cover different
mapping types while keeping the plug-and-play interface intact.

**AlphaFold integration** fits this thesis perfectly — AlphaFold-derived structural features
(pLDDT, pTM, contact maps, per-residue embeddings) become another vector to concatenate.
Relevant for: DDG (where structure is critical), antibody structural features, DTA protein
representations, and potentially molecular 3D conformer features.

---

## Confirmed Issues

### 1. Ridge-only is insufficient
- ProteinGym ablations show Ridge on mean-pooled embeddings scores Spearman ~0.48 vs CNN ~0.62 vs GP ~0.74 on same embeddings
- Ridge was an arbitrary starting point, not a principled choice
- **Action:** Add CNN head and Gaussian process (at minimum) alongside Ridge. Keep Ridge as one option.

### 2. R² alone is wrong metric for several domains
- DMS / antibody fitness: field uses **Spearman rho**
- DDG prediction: field uses **Pearson r** and **RMSE**
- DTA: field uses **Concordance Index (CI)**
- Molecular QSAR: R² is standard here, keep it
- TCGA classification: accuracy is fine
- **Action:** Report R², Spearman rho, and RMSE for all regression tasks. Add CI for DTA.

### 3. DDG (SKEMPI) is fundamentally structural
- Best sequence-only methods: Pearson r ~ 0.25-0.50
- Best structure-based: Pearson r ~ 0.73
- KIRBy is sequence-only → genuine ceiling exists
- But R² = -0.137 (worse than mean) is still suspicious with 518 samples
- **Action:** Investigate — negative R² should not happen even with a bad method. Check data pipeline, outlier filtering, and group splitting with only ~20 PDB complexes.
- **Known limitation (documented):** Bushuiev et al. 2024 (arXiv:2404.10457) found **56% data leakage** in PDB-code-based splitting of SKEMPI v2.0, because structurally similar interfaces exist across PDB codes. KIRBy's PDB complex GroupKFold is the same strategy. Mitigations: (1) KIRBy filters to Ab-Ag only (~hundreds of entries, not thousands), (2) deduplicates with AB-Bind using normalized mutation keys, (3) the ~20 resulting PDB complexes are too few for CATH superfamily splitting. This limitation should be acknowledged in the thesis. If a reviewer challenges it, point to Ab-Ag filtering and deduplication as partial mitigations — the 56% figure applies to full SKEMPI with all protein types.

### 4. IgBert predicts mean everywhere (R² ~ 0)
- **Root cause identified:** NOT a bug. It's curse of dimensionality. IgBERT has 1024D
  embeddings but many dimensions are noise. One paper (Fab Tm with IgBERT, bioRxiv Jan 2026)
  had to aggressively filter embeddings (only keeping dims with |Pearson r| > 0.5 with target)
  to get ANY predictive signal. With Ridge on full 1024D + small antibody datasets, it
  collapses to mean prediction.
- IgBERT Pearson r ~0.64, R² ~0.31 for binding energy in its own paper — it only explains
  ~31% of variance even in the best case.
- **Action:** Either (a) add feature selection before Ridge for high-D reps, (b) use a model
  that handles irrelevant dimensions better (RF/XGBoost already do this), or (c) accept that
  IgBERT embeddings are weaker than ESM-2 for supervised tasks (confirmed by multiple papers).

### 5. Small inter-family datasets may be pathological
- FLAb inter-family: n=130-217, identity GroupKFold
- If most sequences are unique → GroupKFold degenerates to near-LOO
- High-D embeddings (2560D) with LOO-like CV = extremely unstable
- **Action:** Research proper splitting for small diverse antibody panels.

### 6. Many antibody near-zero results may be EXPECTED (not bugs)
- **FLAb2 (Dec 2025):** Tested 30 AI models on antibody developability. **AI models produce
  NO statistically significant correlations for 80% of datasets.**
  - Expression easiest (avg Spearman ~0.22), aggregation hardest (~-0.07)
  - This matches KIRBy's pattern almost exactly
- **AbAgym:** ML models FAILED across the board. FoldX force field outperforms all ML models.
  This validates our poor AbAgym results as real, not bugged.
- **However:** KIRBy's ddg_combined at R² = -0.137 still needs investigation. Sequence-only
  DDG methods in the literature get Pearson r ~0.25-0.50 (positive), not negative.
- **Action:** For each poor-performing dataset, check whether the result is consistent with
  published benchmarks before assuming a bug. Document which results are "expected bad" vs
  "suspiciously bad."

### 7. CRITICAL BUG: MACCS/PDV/Mol2Vec silently drop invalid molecules — array misalignment
- `create_maccs()` (molecular.py:224), `create_pdv()` (molecular.py:249), `create_mol2vec()` (molecular.py:505) use `continue` on invalid SMILES instead of appending zero vectors
- This produces FEWER rows than input. Labels/groups still have full row count → misaligned arrays
- **Every downstream result using these reps is silently corrupted**
- Compare with `create_ecfp4()` which correctly appends zero vectors
- **Action:** One-line fix each — append `np.zeros(dim)` instead of `continue`
- **Full details:** `issues_preprocessing.md` C1, C2, M1

### 8. CRITICAL BUG: AbAgym loader appends sequences BEFORE NaN check — misalignment
- `abagym.py:278-286`: sequences appended to lists BEFORE label NaN check at line 286
- If label is NaN, `continue` skips the label append but sequences already appended
- `all_heavy_seqs` has more entries than `all_labels`
- **Action:** Move sequence appends to AFTER the NaN check
- **Full details:** `issues_preprocessing.md` L3 (reclassified CRITICAL)

### 9. CRITICAL: Data leakage — PCA/scaler/rank fitted on full dataset before CV
- `create_pdv_binary_mean()` (molecular.py:274): column means on full data
- `create_pdv_rank()` (molecular.py:297): rank transform on full data
- `create_pca_rep()` (molecular.py:322-323): StandardScaler + PCA on full data
- ALL TCGA genomic reps (genomic.py): StandardScaler/PCA on full data before CV
- `esm2_entropy_top20` (antibody.py:838): positional entropy on full data
- **Impact:** Test-set information leaks into feature representations for ALL PCA variants and ALL TCGA reps
- **Action:** Restructure as per-fold operations, or document as intentional approximation
- **Full details:** `issues_preprocessing.md` C3-C6, H5
- **STATUS: FIXED** (Phase 1B) — All affected reps now use per-fold `rep_factory` pattern. Factory helpers added to molecular.py, genomic.py, antibody.py. All survey scripts updated to use FACTORY_REPS mechanism: molecular, TCGA, antibody, and DTA. Reaction yield survey verified clean (per-split scaling already correct).

### 10. CRITICAL: Silent fallback to random KFold when groups insufficient
- `find_best_hybrid.py:545`: falls back to `KFold(shuffle=True)` when `n_unique_groups < n_folds`
- `antibody_baseline_survey.py:293`: same fallback
- **This directly violates the NO RANDOM SPLITS policy**
- **Action:** Reduce n_folds to match n_groups instead. Never fall back to random KFold.
- **Full details:** `issues_preprocessing.md` C7

### 11. HIGH: DMS identity grouping provides no meaningful split
- Each unique DMS variant gets its own group → GroupKFold degenerates to LOO
- Mutations at the same position can appear in both train and test
- ProteinGym specifically warns about this: "training on one mutation while testing on another at the same position tends to overestimate"
- **Action:** Group by mutation POSITION for SSM data, not by sequence hash
- **Full details:** `issues_preprocessing.md` H1, H2
- **STATUS: FIXED** (Phase 1C) — DMS datasets in `DATASET_REGISTRY` changed from `'identity'` to `'position_modulo'`. `_assign_position_groups_modulo()` groups by `mutation_pos % n_folds` (ProteinGym scheme). Contiguous split also available. Falls back to identity if no mutation data.

### 12. HIGH: KIBA label polarity may be inverted
- `dta.py:163`: `labels = -labels` negation may double-invert if DeepDTA already transformed KIBA scores
- R² would be the same but predictions would be directionally wrong
- **Action:** Empirically verify by checking actual range of loaded Y values
- **Full details:** `issues_preprocessing.md` H7

### 13. MEDIUM: Antibody mean pooling includes BOS/EOS special tokens
- `antibody.py` `_mean_pool()` includes ESM-2 BOS/EOS tokens
- `protein.py` correctly excludes them for the protein pipeline
- **Inconsistent behavior between antibody and protein pipelines**
- **Action:** Exclude BOS/EOS from antibody mean pooling to match protein.py
- **Full details:** `issues_preprocessing.md` M6, M7

### 14. HIGH: Sort & Slice (SNS) missing from molecular survey
- `create_sns()` EXISTS in `molecular.py:327` but is NOT in `REP_PRESETS['all']` in `find_best_hybrid_molecular.py:88-96`
- User's own qsar_qm_models: XGBoost+SNS R²=0.756 vs XGBoost+ECFP4 R²=0.697
- OxPIG paper confirms 11.37% MAE improvement over hash-based ECFP
- SNS needs per-fold fitting (rep_factory) — this is probably why it was omitted
- **Action:** Add SNS as rep_factory or document why excluded
- **Full details:** `issues_molecular.md` Issue 1
- **STATUS: FIXED** (Phase 1B) — `make_sns_factory()` added to molecular.py. SNS added to REP_PRESETS['all'] as a factory rep. Fits substructure prevalence on train SMILES only per fold.

### 15. HIGH: No DNN model — largest performance gap vs ground truth
- User's qsar_qm_models: DNN+MHG-GNN R²=0.824 (best on LogD), DNN+SNS R²=0.818, DNN+ECFP4 R²=0.766
- KIRBy's sklearn MLP(100,50) is NOT equivalent to a properly trained PyTorch DNN
- **The DNN model class accounts for the single largest performance gap in ground truth data**
- **Action:** Add proper DNN (PyTorch training loop) — hardest to add but biggest impact
- **Full details:** `issues_molecular.md` Issue 2

### 16. MEDIUM: Missing Chemprop/D-MPNN — literature's #1 QSAR method
- Yang et al. 2019: D-MPNN RMSE 0.542 on lipophilicity scaffold split (vs RF+ECFP 0.655)
- Most cited molecular QSAR method, completely absent from KIRBy
- **Full details:** `issues_molecular.md` Issue 10

### 17. MEDIUM: Caco2 R²=0.519 may be endpoint confusion
- KIRBy predicts Caco2_Efflux (P-gp efflux ratio, log-transformed) — different from Caco2 permeability
- Literature benchmarks (CaliciBoost R²=0.78) are for permeability, not efflux ratio
- Efflux ratio is inherently noisier → R²=0.519 may be reasonable for THIS endpoint
- **Action:** Verify endpoint choice and document clearly
- **Full details:** `issues_molecular.md` Issue 5

### 18. MEDIUM: Outlier clipping at 1st/99th percentile inflates R²
- `find_best_hybrid_molecular.py:185-188` clips ALL datasets at 1st/99th percentile
- Ground truth (qsar_qm_models) does NOT clip
- Removes hardest-to-predict compounds → inflates R²
- **Action:** Document or make optional
- **Full details:** `issues_molecular.md` Issue 8

### 19. MEDIUM: GroupKFold vs single scaffold split — numbers not directly comparable
- KIRBy uses 3-fold or 5-fold GroupKFold (averages across folds)
- Literature uses single 80/10/10 ScaffoldSplitter (more conservative)
- KIRBy R² consistently ~0.04-0.08 higher than ground truth for same rep+model
- **NOT a bug** but must be documented when comparing to published numbers
- **Full details:** `issues_molecular.md` Issue 11

### 20. HIGH: ESOL loaded via DeepChem MoleculeNet — contradicts ban
- `esol.py` uses DeepChem's MoleculeNet loader to fetch the ESOL dataset
- The group has banned MoleculeNet (Pat Walters disapproves, group decision)
- The original Delaney 2004 CSV is publicly available independently
- **Action:** Replace MoleculeNet loader with direct CSV download
- **Full details:** `issues_benchmarks.md` §1.7, Issue 5

### 21. MEDIUM: TCGA dead code — mutation_matrix and clinical never registered
- `create_mutation_matrix()` and `create_clinical()` are fully implemented in genomic.py
- But neither is registered in `_REP_DISPATCH` in `find_best_hybrid_tcga.py`
- **Every serious multi-omics benchmark uses somatic mutation features. KIRBy has ZERO.**
- **Action:** 2-line fix to wire into dispatch. Then extend with TMB, drivers, etc. in Phase 3.
- **Full details:** `issues_tcga.md`

### 22. MEDIUM: Missing molecular preprocessing (QSAR-Ready Workflow)
- Standard steps missing: salt stripping, neutralization, tautomer standardization, InChIKey dedup
- Mansouri et al. (J. Cheminformatics 2024) defines the QSAR-Ready Workflow as the standard
- RDKit provides all necessary functions
- **Action:** Add preprocessing pipeline before feature extraction
- **Full details:** `issues_benchmarks.md` §1.3

### 23. MEDIUM: DDG missing reverse mutation augmentation
- Antisymmetric property: DDG(M→W) = -DDG(W→M)
- Standard practice to double training data while enforcing physical constraint
- KIRBy does not use this
- **Action:** Augment DDG training data with reversed mutations
- **Full details:** `issues_benchmarks.md` §3.4

### 24. MEDIUM: TCGA classification should use AUROC, not accuracy
- KIRBy uses accuracy for TCGA classification tasks (LUAD vs LUSC, PAM50)
- Field standard is AUROC for binary, weighted F1 or macro AUROC for multi-class
- Accuracy can be misleading with class imbalance
- **Action:** Add AUROC to classification evaluation
- **Full details:** `issues_benchmarks.md` §5.4, §5.8

### 25. LOW: Missing zero-shot score augmentation for antibody
- ProteinGym standard: augment supervised Ridge/MLP with zero-shot fitness scores from PLMs
- This is a proven technique that significantly boosts supervised performance
- KIRBy does not include zero-shot scores as input features
- **Action:** Compute ESM-1v/ESM-2 log-likelihood ratios as additional features
- **Full details:** `issues_antibody.md` §10a

### 26. LOW: Antibody layer selection not explored
- "The last layer of PLMs is not necessarily the one yielding the highest performance"
  (Domain-specific vs General PLMs for Immunology paper)
- KIRBy uses last layer for all models except `esm2_midlayers`
- Systematic layer search could improve performance
- **Full details:** `issues_antibody.md` §10b

---

## Needed Additions

### Models — ALL DONE (Phase 2)
- [x] CNN prediction head on embeddings (per ProteinGym protocol) — `src/kirby/models/cnn.py`
- [x] Gaussian process / kernel method (per Kermut, SOTA on ProteinGym)
- [x] MLP baseline (proper PyTorch DNN, NOT sklearn MLPRegressor)
- [x] CatBoost (trivial add, competitive per Boldini 2023)
- [x] SVR — **selective use only** (user: "fine in certain use cases but don't make it standard across all surveys"). Size guard _max_samples=10000.
- [x] DNN with proper training loop — `src/kirby/models/dnn.py`
- [x] Keep RidgeCV, RF, XGBoost, LightGBM as-is

### Metrics — ALL DONE (Phase 1A)
- [x] Spearman rho for all regression tasks
- [x] RMSE for all regression tasks (R²-only is non-standard per Chemprop/ADMETlab/PharmaBench)
- [x] Concordance Index for DTA
- [x] Keep R² everywhere

### Missing Reps — MOSTLY DONE (Phase 3B)
- [x] SNS (Sort & Slice) — wired into 'all' preset via `make_sns_factory()` (Phase 1B)
- [ ] Chemprop/D-MPNN — DEFERRED (E2E model, incompatible with KIRBy fixed-rep paradigm)
- [x] PaDEL/Mordred descriptors — Mordred added (`mordred` + `mordred_pca50`). PaDEL deferred.
- [x] Avalon fingerprints — `create_avalon()` in molecular.py, wired into 'all' preset (Phase 3B)

### Data Expansion — ALL DONE (Phase 3B/3D)
- [x] SKEMPI: `ddg_combined_full` loads all SKEMPI PP entries (not just Ab-Ag), ~6x more data (Phase 3B)
- [x] DTA: BindingDB_Kd via TDC added (~52K pairs). Metz deferred. (Phase 3D)
- [x] TCGA: expanded from 34→57 reps across 5 tiers (Phase 3A)

### TCGA Critical Gaps (from 25-repo audit — see issues_tcga.md)
**Done (Phase 0 + Phase 3A):**
- [x] Wire dead code: `create_mutation_matrix()` and `create_clinical()` (Phase 0)
- [x] `mut_tmb`, `mut_driver`, `mut_pathway`, `mut_pca50` — mutation features (Phase 3A Tier 1)
- [x] COSMIC mutational signatures: `sig_sbs`, `sig_dbs`, `sig_id` (Phase 3A Tier 2)
- [x] CIBERSORTx LM22: `immune_cibersort` (Phase 3A Tier 3)
- [x] xCell scores: `immune_xcell` (Phase 3A Tier 3)
- [x] Cancer stemness: `stemness_mrnasi` (Phase 3A Tier 3)
- [x] Tumor purity: `purity_consensus` (Phase 3A Tier 3)
- [x] MATH heterogeneity: `heterogeneity_math` (Phase 3A Tier 3)
- [x] MSI status: `msi_status` (Phase 3A Tier 3)
- [x] HRD score: `hrd_score` (Phase 3A Tier 3)
- [x] CIMP status: `cimp_status` (Phase 3A Tier 3)
- [x] Newer WSI models: TITAN (768D) + UNI2-h (1536D) wired (Phase 3A Tier 4). Virchow2/CONCH deferred.
**Done (Phase 3A Tier 5):**
- [x] Splicing PSI values: `splicing_psi50` (50D, 123 splicing factors + PCA)
- [x] lncRNA expression features: `lncrna_pca50` (50D, 142 cancer lncRNAs + PCA)
- [x] Promoter methylation: `promoter_meth` (50D, TSS200/TSS1500 probes + PCA)
- [x] ssGSEA Hallmark: `gsva_hallmark` (~50D, proper statistical enrichment)
**Remaining:**
- [ ] MOJO joint RNA+methylation embedding (~256D)
- **Current: 57 reps. Potential: 60-70 with MOJO + remaining Tier 3 items.**

### AlphaFold Integration
- [ ] DDG: AlphaFold-predicted structures as input for structure-based DDG features
- [ ] Antibody: AlphaFold structural confidence (pLDDT) as features
- [ ] DTA: AlphaFold protein embeddings / structural features (esmfold/alphafold_db already registered but untested)
- [ ] Molecular: AlphaFold for protein targets in DTA context
- [ ] General: pLDDT, pTM, PAE, per-residue embeddings from AF2 as representation vectors

### Splitting (CRITICAL — group requirement: NO random splits)

**Research complete.** Summary of what the field uses vs what KIRBy does:

| Domain | KIRBy Current | Field Standard | Leakage Risk | Action |
|--------|--------------|----------------|--------------|--------|
| Molecular QSAR | Murcko scaffold GroupKFold | Murcko scaffold GroupKFold | Low | **Already correct** |
| Antibody DMS (intra) | **Position modulo GroupKFold** | Position-based (contiguous/modulo) | Low | **FIXED (Phase 1C)** |
| Antibody (inter) | Sequence identity / experiment | Cluster-based or random (if diverse) | Low | Acceptable |
| DDG | PDB complex GroupKFold | CATH superfamily splitting | **HIGH (56%)** | **Documented as limitation** |
| DTA | Cold-target GroupKFold | Cold-target / cold-drug | Low | **Already correct** |
| TCGA | TSS GroupKFold | Site-preserved CV (TSS GroupKFold) | Low | **Already correct** |
| **Reaction (BH)** | **Additive GroupKFold** | **Leave-one-additive-out** | Low | **FIXED (Phase 1C)** |
| **Reaction (Suzuki)** | **Ligand GroupKFold** | **Leave-one-ligand-out** | Low | **FIXED (Phase 1C)** |

**Details per domain:**

**Molecular:** Murcko scaffold + GroupKFold is the dominant approach (matches DeepChem ScaffoldSplitter). Butina clustering is more rigorous (recommended by Pat Walters' Practical Cheminformatics blog) but Murcko is defensible. RDKit blog (2024) documents known Murcko limitations. No action needed.

**Antibody DMS (intra-family):** ProteinGym defines 3 supervised CV schemes: (1) random, (2) contiguous (sequence split into 5 segments), (3) modulo (position i → fold i%5). Rationale: mutations at the same position have correlated effects, so position-based splits avoid leaking position information. FLIP uses one_vs_rest / two_vs_rest (train on singles, test on multi-mutants). KIRBy's sequence-identity grouping is fine for inter-family but **not ideal for SSM/DMS data** where most sequences differ by 1 mutation. **Action:** Add position-based splitting option for intra-family DMS.

**DDG:** Bushuiev et al. 2024 (arXiv:2404.10457) found that PDB code-based splitting of SKEMPI v2.0 has **56% data leakage** because structurally similar interfaces exist across PDB codes. CATH superfamily splitting (CATH-ddG paper) achieves 0% leakage. With only ~20 PDB complexes this is a known limitation. **Action:** Document as limitation; CATH splitting ideal but requires structural data.

**DTA:** KIRBy's cold-target GroupKFold matches DeepPurpose's `create_fold_setting_cold_protein()`. Already correct. Could add cold-drug as complementary evaluation.

**TCGA:** TSS GroupKFold matches Howard et al. 2021 (Nature Communications) recommendation for site-preserved CV. Already correct. Consider `StratifiedGroupKFold` for classification tasks to maintain class balance.

**Reaction yields (MUST FIX):** The Buchwald-Hartwig dataset has built-in out-of-sample splits: Test1-4 are **leave-one-additive-out** splits. The review "When Yield Prediction Does Not Yield Prediction" (JCIM 2024) defines: leave-one-cluster-out, leave-one-aryl-halide-out, leave-one-additive-out. Components are already parsed in `load_buchwald_hartwig()` (additive, base, ligand fields). **Action:** Use GroupKFold on additive for BH. For Suzuki, use scaffold split on substrate SMILES.

**Key references:**
- Bushuiev et al. 2024 — 56% leakage in SKEMPI PDB splitting
- ProteinGym — position-based supervised CV
- Howard et al. 2021 — site-preserved CV for TCGA (Nature Comms)
- "When Yield Prediction Does Not Yield Prediction" (JCIM 2024) — reaction split taxonomy
- Practical Cheminformatics blog — Butina > Murcko for molecular

---

## Forward Compatibility Experiments

Prove KIRBy's thesis: take a brand-new model, extract embeddings, plug in with zero
framework changes, show it works, show hybrid augmentation improves it further.

### Tier 1 Candidates (brand new, open source, directly relevant)

| Model | Released | Domain | Why it's perfect |
|-------|---------|--------|-----------------|
| **MINT** | Jan 2026 | Antibody/PPI | PPI-aware LM built on ESM-2 w/ cross-chain attention. Surpasses ESM-2 on Ab-Ag binding & DDG. [github.com/VarunUllanat/mint](https://github.com/VarunUllanat/mint) |
| **Boltz-2** | 2025 | Structure/binding | Joint structure + binding affinity prediction. First DL model to approach FEP accuracy. MIT license. [github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz) |

### Tier 2 Candidates (strong, well-documented extraction methods)

| Model | Released | Domain | Notes |
|-------|---------|--------|-------|
| **SaProt** | 2024 | Protein | ICLR 2024 spotlight. ESM-2 retrained with structure-aware vocabulary (AA + Foldseek 3Di). 1280D. #1 on ProteinGym (Apr 2024). Drop-in ESM-2 replacement. MIT license. |
| **MolE** | Nov 2024 | Molecular | Recursion. DeBERTa on molecular graphs, 842M molecule pretraining. Top on 10/22 TDC ADMET. CC-BY-NC. |
| **MiniMol** | 2024 | Molecular | GINE GNN, 10% parameters of MolE but outperforms on 17 datasets. Easy pip install. |
| **IgBERT/IgT5 (new)** | Dec 2024 | Antibody | Exscientia. Paired H+L aware. Best antibody-specific LMs. 2B unpaired + 2M paired training. |
| **S2ALM** | May 2025 | Antibody | Seq+structure multimodal, built on ESM-2 650M. SOTA on binding affinity. |
| **ESM3** | Jan 2025 | Protein | Multimodal (seq+struct+func), 25x more FLOPs than ESM-2. Open 1.4B model. |
| **Chai-1** | 2025 | Structure | Apache 2.0, predicts protein+ligand+DNA/RNA complexes. Relevant for DTA. |
| **ESM-C 6B** | 2025 | Protein | Already have esmc_600m — upgrading to 6B is trivial, demonstrates scalability. |

### Tier 3 Candidates (interesting, more effort)

| Model | Released | Domain | Notes |
|-------|---------|--------|-------|
| **S3F** | 2024 | Protein fitness | NeurIPS 2024. Seq+struct+surface multi-scale. SOTA on ProteinGym supervised (non-MSA). |
| **ProSST** | 2024 | Protein | NeurIPS 2024. Structure quantization + disentangled attention. |
| **IgBlend** | 2024 | Antibody | Multi-modal (sequence + 3D structure). 512D. |
| **GP-MoLFormer** | 2025 | Molecular | IBM. Generative successor to MoLFormer (already in KIRBy). |
| **AlphaFold3** | 2024 | Structure | 384D embeddings. Restrictive academic license. |
| **Ab-RoBERTa** | 2026 | Antibody | 125M params, 5x faster than IgT5. |

### Experiment Protocol

For each new model:
1. Extract embedding vectors (zero KIRBy code changes needed)
2. Register as new rep, run through existing pipeline → baseline score
3. Augment with existing KIRBy reps → show hybrid improvement
4. Compare: `new_model` alone vs `new_model + old_features` vs `best_existing`

MINT is the #1 pick: 2 months old, directly targets weakest KIRBy domain (antibody/DDG),
built on ESM-2 so comparison is apples-to-apples.

---

## TCGA Representation Gaps (from research)

KIRBy has 53 TCGA reps (up from 34 after Phase 3A Tiers 1-4). Remaining gaps below.

### Tier 1 — High impact, straightforward to implement (ALL DONE in Phase 3A)
- [x] `mutation_binary` → `mutation_matrix` — gene-level binary somatic mutation matrix
- [x] `tmb` → `mut_tmb` — tumor mutational burden (mutations/Mb)
- [x] `mutation_signatures` → `sig_sbs`, `sig_dbs`, `sig_id` — COSMIC signature weights
- [x] `cibersort_fractions` → `immune_cibersort` — 22 immune cell type fractions
- [x] `stemness_index` → `stemness_mrnasi` — mRNAsi/EREG-mRNAsi indices
- [x] `tumor_heterogeneity` → `purity_consensus`, `heterogeneity_math` — purity/ploidy/MATH

### Tier 2 — High impact, moderate effort (ALL DONE)
- [x] `splicing_psi` → `splicing_psi50` — 123 splicing factor genes + PCA to 50D (Phase 3A Tier 5)
- [x] `gsva_c2` → `gsva_hallmark` — ssGSEA with MSigDB Hallmark gene sets (Phase 3A Tier 5). C2 deferred.
- [x] `xcell_scores` → `immune_xcell` — 64 cell type enrichment scores
- [x] `msi_status` — microsatellite instability. FDA-approved immunotherapy biomarker.
- [x] `cimp_status` — CpG island methylator phenotype
- [x] `promoter_methylation` → `promoter_meth` — TSS200/TSS1500 probes + PCA to 50D (Phase 3A Tier 5)

### Tier 3 — Interesting, harder
- [ ] `atac_accessibility` — ATAC-seq peaks (only 410 TCGA samples)
- [ ] `cptac_proteomics` — mass-spec proteomics (separate data source, BRCA/OV/COAD/LUAD/etc.)
- [ ] `fusion_binary` — recurrent gene fusions
- [ ] `neoantigen_load` — predicted neoantigen count + quality
- [ ] `regulon_activity` — SCENIC regulon activity scores
- [x] Newer WSI models: TITAN (768D) + UNI2-h (1536D) wired; Virchow2 + CONCH deferred (need tile extraction)

### Key repos checked
- [greenelab/pancancer](https://github.com/greenelab/pancancer) — pan-cancer ML with mutations
- [chenzRG/Cancer-Multi-Omics-Benchmark (MLOmics)](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark)
- [txWang/MOGONET](https://github.com/txWang/MOGONET) — GCN multi-omics integration
- [zhangxiaoyu11/OmiVAE](https://github.com/zhangxiaoyu11/OmiVAE) — VAE multi-omics
- [xlucpu/MOVICS](https://github.com/xlucpu/MOVICS) — multi-omics clustering
- [BiomedSciAI/biomed-multi-omic](https://github.com/BiomedSciAI/biomed-multi-omic) — IBM BMFM-RNA
- [Cassie07/PathOmics](https://github.com/Cassie07/PathOmics) — pathology-genomics
- [gevaertlab/sequoia-pub](https://github.com/gevaertlab/sequoia-pub) — WSI→transcriptomics

---

## Antibody Supervised Prediction Research (from 20+ repos)

### Key Finding: ESM-2 > antibody-specific models for supervised tasks
Multiple independent studies confirm general protein LMs beat antibody-specific ones:
- arjan-hada benchmark: ESM-2 Spearman 0.48 on OOD, beat AntiBERTa2, AbLang-H
- ft-ESM2 paper: ESM-2 fine-tuned achieves AUROC 0.88 for antibody specificity
- TherML: Fine-tuned AntiBERTy Spearman 0.71 but ESM-1b 0.63 (both beat zero-shot)
- Ginkgo GDPa1: ESM-2 + Ridge for developability prediction, scaled with data size

### Key Finding: Model hierarchy for protein fitness prediction
From ProteinGym and FLIP supervised benchmarks (same ESM embeddings, different downstream):
- Ridge: Spearman ~0.48 (standard baseline, used everywhere)
- FNN/MLP: Spearman ~0.50-0.51 (VespaG)
- CNN: Spearman ~0.62 (ProteinGym)
- ProteinNPT (transformer): Spearman ~0.50-0.57
- **GP (Kermut): Spearman ~0.66 OVERALL, up to 0.817 for stability** ← SOTA

### Key Finding: One-hot can beat PLMs on small datasets
Multiple papers warn that PLM embeddings don't always add value over one-hot encoding.
ProteinGym: Ridge on OHE + zero-shot features can outperform Ridge on embeddings.

### Repos examined for antibody/protein supervised prediction
1. [arjan-hada/antibody-binding-affinity](https://github.com/arjan-hada/antibody-binding-affinity) — ESM2 vs antibody LMs for scFv Kd
2. [ucrbioinfo/AbAffinity](https://github.com/ucrbioinfo/AbAffinity) — LLM fine-tuned for SARS-CoV-2 binding
3. [AmeyaHarmalkar/therML](https://github.com/AmeyaHarmalkar/therML) — scFv thermostability, ESM-1b/AntiBERTy
4. [Merck/AbPROP](https://github.com/Merck/AbPROP) — Developability, GNNs + ProtBERT
5. [Merck/AbLEF](https://github.com/Merck/AbLEF) — Developability with MD ensembles
6. Fab Tm with IgBERT (bioRxiv Jan 2026) — RF on filtered IgBERT embeddings
7. ft-ESM2 for antigen specificity (PLOS Comp Bio 2025) — SVM on ESM2
8. [nicwulab/HA_Abs](https://github.com/nicwulab/HA_Abs) — Influenza Ab specificity
9. [regeneron-mpds/propermab](https://github.com/regeneron-mpds/propermab) — Regeneron developability
10. [Ginkgo GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1) — ESM-2 + Ridge for developability
11. [OATML-Markslab/ProteinNPT](https://github.com/OATML-Markslab/ProteinNPT) — NeurIPS 2023 supervised DMS
12. [petergroth/kermut](https://github.com/petergroth/kermut) — GP on ESM-2, NeurIPS 2024 SOTA
13. [JSchlensok/VespaG](https://github.com/JSchlensok/VespaG) — FNN on ESM-2 per-residue
14. [DeepGraphLearning/S3F](https://github.com/DeepGraphLearning/S3F) — Seq+struct+surface
15. [J-SNACKKB/FLIP](https://github.com/J-SNACKKB/FLIP) — Protein fitness benchmark
16. [Graylab/FLAb](https://github.com/Graylab/FLAb) — Antibody fitness benchmark
17. [Jerome-Alvarez/TEMPRO](https://github.com/Jerome-Alvarez/TEMPRO) — Nanobody Tm with ESM
18. [MSBMI-SAFE/AbBiBench](https://github.com/MSBMI-SAFE/AbBiBench) — Antibody binding benchmark
19. [OATML-Markslab/ProteinGym](https://github.com/OATML-Markslab/ProteinGym) — 217 DMS assays
20. InstructPLM-mu (2025) — ESM2 fine-tuning beats ESM3 on ProteinGym

---

## Molecular QSAR Research (from user's repo + 15 external pipelines)

### User's qsar_qm_models baseline performance
- LogD: R²=0.834 (DNN + MHG-GNN), 0.816 (DNN + SNS), 0.697 (XGBoost + ECFP4)
- Models used: RF, XGBoost, LightGBM, SVM, CatBoost, DNN, GNN, GP
- Reps used: ECFP4, SNS, PDV, Mol2Vec, ChemBERTa, MHG-GNN, MPNN
- Split: scaffold 80/10/10 via DeepChem ScaffoldSplitter

### KIRBy vs literature comparison
| Dataset | KIRBy best | Literature best (scaffold) | Gap | Notes |
|---------|-----------|--------------------------|-----|-------|
| LogD | 0.841 (grover+ridge) | 0.834 (DNN+MHG-GNN, user's repo) | None | **Working correctly** |
| Caco2 | 0.519 (graph_kernel+lgbm) | 0.779 (PaDEL+XGB, CaliciBoost) | **-0.26** | May need better reps (PaDEL 3D) or models |
| hERG | 0.552 (ecfp4+lgbm) | 0.61-0.68 (XGB+descriptors) | **-0.06 to -0.13** | Moderate gap, likely model |
| QM9 | 0.139 (pdv_rank+ridge) | 0.1-0.5 (2D, scaffold, small) | None | **Expected for frozen 2D + scaffold** |

### Key reps KIRBy is missing (from literature) — MOSTLY DONE
- [ ] PaDEL descriptors (2D+3D) — best Caco2 rep in CaliciBoost benchmark (deferred)
- [ ] RDKit 2D descriptors (~200D) — used everywhere, often combined with FPs (deferred — Mordred is superset)
- [x] Sort & Slice (SNS) — wired into 'all' preset via `make_sns_factory()`
- [x] Avalon fingerprints — `create_avalon()` in molecular.py, 2048-bit
- [x] Mordred descriptors — `mordred` (~1613D) + `mordred_pca50` factory

### Repos examined for molecular QSAR
1. User's qsar_qm_models — 7 models, 10 reps, scaffold split
2. [Benchmark-ADMET-2025](https://github.com/DonghaiZHAO-ZJU/Benchmark-ADMET-2025) — 14 endpoints, 21 models, scaffold+OOD splits
3. CaliciBoost (arxiv 2506.08059) — Caco-2 systematic rep comparison
4. [ffstghc/caco2ml](https://github.com/ffstghc/caco2ml) — Bayer Caco-2, LightGBM+RDKit
5. [molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME) — Biogen 6 ADME endpoints
6. [swansonk14/admet_ai](https://github.com/swansonk14/admet_ai) — Chemprop, 41 ADMET datasets
7. [chemprop_benchmark_v2](https://github.com/chemprop/chemprop_benchmark_v2) — D-MPNN benchmarks
8. [PatWalters/qsar](https://github.com/PatWalters/qsar) — Pat Walters' own QSAR pipeline
9. [MolecularAI/QSARtuna](https://github.com/MolecularAI/QSARtuna) — AstraZeneca automated QSAR
10. [mindrank-ai/PharmaBench](https://github.com/mindrank-ai/PharmaBench) — 11 ADMET datasets
11. [PDelre93/hERG-QSAR](https://github.com/PDelre93/hERG-QSAR) — ChEMBL hERG
12. [datagrok-ai/admetica](https://github.com/datagrok-ai/admetica) — 23 ADMET models
13. [scikit-fingerprints](https://github.com/scikit-fingerprints/scikit-fingerprints) — 30+ FPs
14. Pred-hERG, ToxTree-hERG, MultiFP-consensus (literature)
15. [Prospero1988/logD_predictor](https://github.com/Prospero1988/logD_predictor) — NMR-based LogD

---

## Open Research Questions (updated)

1. ~~What do supervised ESM-2 embedding pipelines achieve on antibody DMS data?~~ **ANSWERED** — see antibody section
2. ~~What models/features does the molecular QSAR field actually use?~~ **ANSWERED** — see molecular section
3. ~~What TCGA representations/methods are we missing?~~ **ANSWERED** — see TCGA section
4. ~~What does the FLIP benchmark show?~~ **ANSWERED** — random splits overestimate; 2-vs-rest is harder
5. ~~Why does IgBert fail?~~ **ANSWERED** — curse of dimensionality, not a bug
6. ~~Can more data be added to DDG and DTA pipelines?~~ **ANSWERED** — see below
7. ~~What reaction yield methods exist beyond DRFP?~~ **ANSWERED** — see below
8. ~~What non-random splits are defensible per domain?~~ **ANSWERED** — see Splitting section. 4/6 domains already correct. Reaction MUST change. DDG has 56% leakage risk with PDB-based splitting.

## Reaction Yield Research (from 8 repos)

### KIRBy vs literature
- BH: KIRBy R²=0.934 vs literature 0.95 (random split). **Competitive.**
- Suzuki: KIRBy R²=0.842 vs Yield-BERT 0.81. **Actually above Yield-BERT.**
- Random splitting inflates all these numbers. Leave-one-reagent-out drops BH to ~0.83.

### Missing reps
- [x] `rxnfp` — pretrained reaction fingerprints from IBM RXN (256D). Gets ~0.93 on BH. DONE Phase 3B.
- [x] `onehot_components` — one-hot factorial component encoding (~46D BH). DONE Phase 3B.
- [x] `concat_maccs`, `concat_pdv`, `product_ecfp4`, `reactant_ecfp4` — wired into survey. DONE Phase 3B.
- [ ] `yield_bert` — Yield-BERT embeddings. SOTA on BH at 0.951 (fine-tuned transformer).

### Additional reaction datasets
- Deoxyfluorination HTE (740 reactions)
- C-H Arylation HTE
- ELN Buchwald-Hartwig (781 real-world reactions, Saebi et al.)
- Open Reaction Database (ORD) — large-scale, multiple reaction types

### Splitting fix needed
**Current: random 70/30 × 10. MUST CHANGE.** Options:
- Leave-one-reagent-out (e.g., leave-one-additive-out for BH)
- Leave-one-substrate-out
- Scaffold split on product molecules

### Repos examined
1. [reymond-group/drfp](https://github.com/reymond-group/drfp) — DRFP benchmarks
2. [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields) — YieldBERT
3. [rxn4chemistry/rxnfp](https://github.com/rxn4chemistry/rxnfp) — Reaction FPs
4. [Meteor-han/ReaMVP](https://github.com/Meteor-han/ReaMVP) — Multi-view pretraining
5. [hampusgs/AL-for-reaction-yield-prediction](https://github.com/hampusgs/AL-for-reaction-yield-prediction)
6. [SoftServeInc/yield-paper](https://github.com/SoftServeInc/yield-paper) — GNN for yield
7. [Nikki0526/MetaRF](https://github.com/Nikki0526/MetaRF) — Few-shot yield
8. [sustainable-processes/ORDerly](https://github.com/sustainable-processes/ORDerly) — ORD benchmarks

---

## DTA Data Expansion Research

### Additional datasets to add
| Dataset | Drugs | Proteins | Pairs | Type | Priority |
|---------|-------|----------|-------|------|----------|
| **BindingDB_Kd** | 10,665 | 1,413 | 52,284 | Kd | **HIGH** — 10x Davis, diverse targets |
| **Metz** | 1,423 | 170 | 35,259 | Ki (pKi) | **HIGH** — standard 3rd kinase benchmark |
| BindingDB_Ki | 174,662 | 3,070 | 375,032 | Ki | Medium — very large |
| PDBBind refined | ~3,000 | ~3,000 | ~3,000 | Kd/Ki | Already in KIRBy (needs manual download) |

### DTA metrics: add CI
Field standard is CI + MSE (not R²). KIRBy should report CI for comparability.

### DTA repos examined
- [hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA)
- [thinng/GraphDTA](https://github.com/thinng/GraphDTA)
- [guaguabujianle/MGraphDTA](https://github.com/guaguabujianle/MGraphDTA)
- [MaastrichtU-IDS/predict-drug-target](https://github.com/MaastrichtU-IDS/predict-drug-target) — ESM2+MolTransformer
- [camlab-ethz/GEMS](https://github.com/camlab-ethz/GEMS) — leak-free PDBBind

---

## DDG Data Expansion Research

### Options to increase DDG data
1. **Use ALL of SKEMPI 2.0** (not just Ab-Ag): 7,085 → ~6x current. — **DONE** (`load_ddg_combined_dataset(ab_ag_only=False)`; `ddg_combined_full` in DATASET_REGISTRY; `--include-all-skempi` flag in antibody survey)
2. **AbDesign** (658 CDR-H3 mutants): Independent held-out test set, non-overlapping PDBs. — **STUB** (`load_abdesign_ddg()` raises informative error with data source instructions)
3. **AbBiBench** (155K mutants): Relative binding (not DDG), different task.
4. **Synthetic pre-training** (FoldX-generated ~1M DDG values): Graphinity needed this for robustness.
5. **DO NOT merge ProThermDB/FireProtDB/ThermoMutDB** — these are folding DDG, NOT binding DDG.

### Critical finding
Graphinity paper (oxpig, Nature Comp Sci 2025): With only a few hundred Ab-Ag DDG points,
models overtrain. They needed ~1M synthetic FoldX DDG values for robustness. ML models
trained on SKEMPI/AB-Bind achieve Spearman 0.4-0.7 on training distribution but DROP TO
0.0-0.1 on non-overlapping complexes.

### DDG repos examined
- [oxpig/Graphinity](https://github.com/oxpig/Graphinity) — GNN for Ab-Ag DDG
- [LDeng0205/StaB-ddG](https://github.com/LDeng0205/StaB-ddG) — Binding from folding DDG (ICML 2025)
- [haddocking/DACUM](https://github.com/haddocking/DACUM) — Cleaned SKEMPI subset
- [ak422/CATH-ddG](https://github.com/ak422/CATH-ddG) — CATH-based splitting
- [protddg-bench/protddg-bench](https://github.com/protddg-bench/protddg-bench) — Folding DDG benchmark

---

## Forbidden / Constraints
- **NO MoleculeNet or TDC** (group decision, non-negotiable)
- **NO random splits** (group requirement, will be heavily scrutinized)
- Scaffold splitting for molecular (PhD group requirement)
- Pat Walters' guidelines apply

---

## Research Status — ALL COMPLETE (2026-03-10)

**8 research files, 3,589 lines, 80+ repos examined. Research phase is DONE.**
**Next step: discuss roadmap with user, then begin Phase 0 implementation.**

| Domain | File | Lines | Repos | Status |
|--------|------|-------|-------|--------|
| Molecular QSAR | `issues_molecular.md` | 293 | 15 | **DONE** |
| Antibody/DDG | `issues_antibody.md` | 350 | 14 | **DONE** |
| TCGA | `issues_tcga.md` | 567 | 25 | **DONE** |
| DTA + Reaction | `issues_dta_reaction.md` | 409 | 18 | **DONE** |
| Forward compat + Models | `issues_forward_compat.md` | 558 | 17 models + 8 libs | **DONE** |
| Splitting | `issues_splitting.md` | 477 | 15+ | **DONE** |
| Preprocessing | `issues_preprocessing.md` | 303 | KIRBy code | **DONE** |
| Benchmarks/protocols | `issues_benchmarks.md` | 632 | 30+ | **DONE** |
