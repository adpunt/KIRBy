# KIRBy Antibody & DDG Pipeline Audit

**Date:** 2026-03-10
**Auditor:** Claude (automated research audit)
**Scope:** Comparing KIRBy's antibody pipeline against real GitHub implementations and published benchmarks

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Reference Repos & Papers](#2-reference-repos--papers)
3. [DDG Combined: Why R^2 = -0.137](#3-ddg-combined-why-r2---0137)
4. [Downstream Model Choice: Ridge vs GP vs MLP](#4-downstream-model-choice-ridge-vs-gp-vs-mlp)
5. [Splitting Strategy Issues](#5-splitting-strategy-issues)
6. [IgBERT Predicts Mean Everywhere](#6-igbert-predicts-mean-everywhere)
7. [FLAb Inter-family Near-Zero Results](#7-flab-inter-family-near-zero-results)
8. [AbAgym Poor Performance](#8-abagym-poor-performance)
9. [Metric Choice: R^2 vs Spearman](#9-metric-choice-r2-vs-spearman)
10. [Missing Methodological Elements](#10-missing-methodological-elements)
11. [Summary of Verdicts](#11-summary-of-verdicts)

---

## 1. Executive Summary

KIRBy's antibody pipeline is architecturally reasonable but has several issues when compared against the field:

- **DDG R^2 = -0.137 is EXPECTED, not a bug.** Sequence-only DDG prediction with leave-one-complex-out splitting is an essentially unsolved problem. Even structure-based SOTA achieves Pearson ~0.46 under strict splitting. KIRBy's negative R^2 under GroupKFold-by-complex is consistent with published findings.
- **Ridge regression is underpowered** for supervised protein fitness prediction. The field uses Gaussian Processes (Kermut), MLPs (FLAb2, VespaG, EvolvePro), and Random Forest (EvolvePro). Ridge is used only as a baseline in ProteinGym, not as the primary model.
- **Identity GroupKFold for DMS is incorrect** per ProteinGym standards. DMS datasets have one wild-type with point mutations, so all sequences are near-identical. Grouping by sequence identity is meaningless. ProteinGym uses position-based splits (random/modulo/contiguous).
- **R^2 as sole metric is non-standard.** Every major benchmark (ProteinGym, FLAb, FLAb2, AbBiBench, AbAgym) uses Spearman rho as the primary metric. R^2 penalises scale/offset errors that Spearman ignores.
- **FLAb2 shows one-hot can match billion-parameter PLMs** with enough data (>1000 samples) and supervised fine-tuning, which means KIRBy's current results may be underperforming due to model choice, not representation choice.

---

## 2. Reference Repos & Papers

### Category A: Source Code for Benchmark Methods

| Repo | URL | What | Downstream Model | Splitting | Metric |
|------|-----|------|-----------------|-----------|--------|
| **ProteinGym** | https://github.com/OATML-Markslab/ProteinGym | 217 DMS assays, 2.7M variants, supervised + zero-shot benchmark | Ridge (baseline), ProteinNPT (SOTA) | Random / Modulo / Contiguous position-based 5-fold | Spearman rho, MSE |
| **Kermut** | https://github.com/petergroth/kermut | NeurIPS 2024 spotlight, supervised DMS prediction | Gaussian Process (GPyTorch) on ESM-2 650M embeddings | ProteinGym 3 schemes | Spearman rho (SOTA: 0.605-0.817 by category) |
| **FLAb** | https://github.com/Graylab/FLAb | Antibody fitness benchmark, 6 properties | Zero-shot: perplexity. Few-shot: 2-layer FC network | 80/10/10 train/val/test | Spearman rho |
| **FLAb2** | bioRxiv 2025.12.27.696706 | 4M antibodies, 32 studies, 7 properties | 2-layer FC network | 80/10/10 train/val/test | Spearman rho |
| **VespaG** | https://github.com/JSchlensok/VespaG | Fast fitness prediction from ESM-2 | 1-hidden-layer FFN (256 units) on ESM-2 embeddings | ProteinGym benchmark | Spearman rho (avg 0.48 across 217 DMS) |
| **AbAgym** | https://github.com/3BioCompBio/Abagym | 335K Ab-Ag mutations, 5 DMS experiments | FoldX, BeAtMuSiC, SAAMBE-3D, mCSM-AB, SaProt | Per-complex evaluation | Pearson, Spearman |
| **AbBiBench** | https://github.com/MSBMI-SAFE/AbBiBench | 184K experimental Ab mutants, 14 antibodies, 9 antigens | Zero-shot log-likelihood from ESM-2, AntiBERTy, ProteinMPNN, ESM-IF, etc. | Per-antibody evaluation | Spearman rho, precision@10 |
| **EvolvePro** | https://github.com/mat10d/EvolvePro | Few-shot active learning for protein engineering | Random Forest on ESM-2 15B mean-pooled embeddings | Active learning rounds | Activity improvement (up to 100x) |
| **ProteinNPT** | https://github.com/OATML-Markslab/ProteinNPT | NeurIPS 2023, supervised protein property prediction | Non-Parametric Transformer on MSA-Transformer embeddings | ProteinGym 3 schemes | Spearman rho (SOTA on ProteinGym supervised) |
| **Graphinity** | https://github.com/oxpig/Graphinity | Structure-based Ab-Ag DDG prediction | Equivariant GNN on Ab-Ag complex structures | Leave-complex-out on SKEMPI 2.0 | Pearson (up to 0.87, but overfitting noted) |
| **DDGb_bias** | https://github.com/3BioCompBio/DDGb_bias | Quantification of biases in DDG predictors | Benchmarks 8 predictors | SKEMPI 2.0 | Pearson, with symmetrisation |
| **CATH-ddG** | https://github.com/ak422/CATH-ddG | Robust DDG prediction via CATH superfamily splitting | Structure-informed model with co-evolutionary features | CATH superfamily split (no leakage) | Pearson, Spearman |
| **StaB-ddG** | https://github.com/LDeng0205/StaB-ddG | ICML 2025, binding DDG from folding energy | Pre-trained inverse-folding model | SKEMPI v2.0 | Pearson (matches FoldX accuracy) |
| **IgBERT/IgT5** | Exscientia/IgBert on HuggingFace | Paired Ab language model, 2B unpaired + 2M paired sequences | Evaluated on binding energy, expression, sequence recovery | Per-task evaluation | R^2 (binding), accuracy (recovery) |

### Category B: Research Implementations Using These Methods

| Repo/Paper | URL | Relevant Finding |
|------------|-----|------------------|
| **Domain-specific vs General PLMs for Immunology** | https://www.immunoinformaticsjournal.com/article/S2667-1190(24)00006-5/fulltext | General PLMs (ESM-2) outperform Ab-specific models on expression tasks; Ab-specific better on binding. Layer selection is a critical hyperparameter. |
| **AbDesign** | https://www.tandfonline.com/doi/full/10.1080/19420862.2025.2567319 | 658 CDR-H3 mutants. ML models (DSMBind, RDE-PPI) achieve rho~0.4-0.7 on training data but fail to generalise. FoldX is more robust. |
| **Nature Comp Sci (Hummer et al. 2025)** | https://www.nature.com/articles/s43588-025-00823-8 | "Currently insufficient experimental data to accurately and robustly predict DDG." Orders of magnitude more data needed. |
| **DG-Affinity** | https://pmc.ncbi.nlm.nih.gov/articles/PMC10644518/ | Sequence-based Ab-Ag affinity prediction using PLM embeddings. Pearson >0.65 on independent test, but task is absolute affinity, not DDG. |
| **TopoBind** | https://arxiv.org/html/2508.19632v1 | Multi-modal (sequence + topology) Ab-Ag binding free energy. R^2 = 0.460 with ESM-2 + MLP. |

---

## 3. DDG Combined: Why R^2 = -0.137

### KIRBy's Setup
- **File:** `tests/antibody_baseline_survey.py` line 58, `src/kirby/datasets/ddg_combined.py`
- **Data:** SKEMPI 2.0 + AB-Bind merged, deduplicated, ~800-1200 entries
- **Splitting:** `GroupKFold` by PDB complex ID (`assign_complex_groups`)
- **Reps:** Mean-pooled ESM-2 over full VH/VL sequence (2560D)
- **Models:** RidgeCV, RF, XGBoost, LightGBM
- **Result:** R^2 = -0.137

### Why This is Expected (NOT a bug)

**Evidence 1: Leave-one-complex-out destroys performance.**
The DDGb_bias paper (https://github.com/3BioCompBio/DDGb_bias) and CATH-ddG paper both document that holding out entire complexes causes massive performance drops. TopNetTree's average Pearson drops to **0.17** under leave-one-complex-out. R^2 of 0.17^2 = 0.029 is barely above zero — and that is for structure-based methods. Sequence-only methods would be worse.

**Evidence 2: Hummer et al. (Nature Comp Sci 2025) — "insufficient data".**
Even Graphinity (structure-based equivariant GNN achieving Pearson up to 0.87) shows signs of overfitting. The authors state that "orders of magnitude more data are needed" for robust DDG prediction. With ~800 entries in SKEMPI Ab-Ag subset, there simply is not enough data.

**Evidence 3: ML models systematically fail on Ab-Ag DDG.**
AbAgym (https://github.com/3BioCompBio/Abagym) and AbDesign both independently show that ML models (SAAMBE-3D, mCSM-AB, SaProt, DSMBind, RDE-PPI) "fail to produce statistically meaningful predictions" on Ab-Ag DDG. Only physics-based FoldX generalises.

**Evidence 4: Sequence-only embedding is fundamentally wrong for DDG.**
A single-point mutation changes 1 out of ~120 residues. Mean-pooled ESM-2 embeddings over the full sequence barely change between wildtype and mutant. The DDG signal is buried in noise. This is exactly why KIRBy's `mutation.py` has `esm2_mutsite` (per-residue at mutation site) and `esm2_wt_delta` (wildtype minus mutant embedding difference) — but these are 4 separate mutation reps, not the default `esm2` rep.

### Verdict: NOT A BUG
R^2 = -0.137 is consistent with the published literature. Sequence-only DDG prediction with leave-one-complex-out splitting is an essentially unsolved problem. The negative R^2 means the model is worse than predicting the mean, which is expected when:
1. The splitting removes all examples from the same complex
2. Mean-pooled sequence embeddings cannot capture single-point mutation effects
3. The dataset is too small (~800 entries from ~30 complexes)

### Possible Improvements (if desired)
- Use mutation-aware reps (`esm2_mutsite`, `esm2_wt_delta`, `esm2_llr`) instead of full-sequence `esm2`
- Add FoldX predictions as a feature (requires structure)
- Report Pearson/Spearman instead of R^2 (less sensitive to scale)
- Accept that sequence-only DDG prediction may be fundamentally limited

---

## 4. Downstream Model Choice: Ridge vs GP vs MLP

### What the Field Uses

| Method | Model | Why |
|--------|-------|-----|
| **ProteinGym (baseline)** | Ridge regression (augmented with zero-shot scores) | Simplest baseline; NOT the recommended approach |
| **ProteinNPT (SOTA on ProteinGym)** | Non-Parametric Transformer | Self-attention over both sequences and labels |
| **Kermut (NeurIPS 2024 spotlight)** | Gaussian Process with composite kernel on ESM-2 embeddings | Spearman 0.605-0.817 by category; provides calibrated uncertainties |
| **VespaG** | 1-hidden-layer FFN (256 units) on ESM-2 | Spearman 0.48 on 217 DMS assays; blazingly fast |
| **FLAb/FLAb2** | 2-layer fully connected network | Standard for supervised antibody fitness |
| **EvolvePro** | Random Forest on ESM-2 15B embeddings | Active learning; works with as few as 10 datapoints |
| **AbBiBench** | Zero-shot (no supervised model) | Log-likelihood scoring |

### KIRBy's Approach
- **File:** `tests/antibody_baseline_survey.py` lines 221-253
- **Models:** RidgeCV, RF, XGBoost, LightGBM
- Ridge is the default (line 706: `--models ridge,lightgbm`)

### Verdict: RIDGE IS UNDERPOWERED BUT DEFENSIBLE AS A BASELINE

Ridge is explicitly used as the **floor** in ProteinGym's supervised benchmark. Every competitive method uses something stronger:
- GP (Kermut) for DMS data
- MLP/FFN for antibody fitness (FLAb, FLAb2, VespaG)
- RF for few-shot settings (EvolvePro)

KIRBy already includes RF, XGBoost, and LightGBM, which partially addresses this. However:
1. **No MLP/neural network option.** FLAb2 showed that with >1000 samples, even a fine-tuned one-hot model with a 2-layer FC matches billion-parameter PLMs. KIRBy should consider adding a simple MLP.
2. **No GP option.** Kermut's GP approach is SOTA for DMS, and KIRBy's DMS datasets would benefit from this.
3. Ridge with 2560D ESM-2 embeddings and <500 samples is likely overfitting — RidgeCV's regularisation may not be enough for such high dimensionality relative to sample size.

---

## 5. Splitting Strategy Issues

### Issue 5a: Identity GroupKFold for DMS is Incorrect

**KIRBy's approach:**
- File: `tests/antibody_baseline_survey.py` lines 48-53, 175-176, 196-214
- DMS datasets (`flab_expression_koenig`, `flab_binding_koenig`, all AbAgym single-target) use `'identity'` group strategy
- `_identity_groups()` hashes sequences, but DMS data is one wildtype + point mutations
- All sequences in a DMS dataset are >99% identical (differ by 1-3 AA)
- Result: identity hashing creates thousands of groups (one per unique mutant), which makes GroupKFold essentially equivalent to random splitting

**What ProteinGym does:**
- Three position-based splitting schemes: **Random**, **Modulo**, **Contiguous**
- Random: mutations randomly allocated to folds (easiest — mutations at same position can appear in both train and test)
- Modulo: mutations assigned to folds by position modulo N
- Contiguous: sequence split into N contiguous segments

**Why this matters:**
- In DMS data, if mutations at the same position appear in both train and test (as with random or identity splitting), the model can memorise position-specific effects. This inflates performance.
- The harder (and more realistic) test is whether the model can extrapolate to unseen positions (modulo/contiguous splits).
- KIRBy's identity split is effectively random splitting since each mutant has a unique sequence hash.

**Verdict: SPLITTING IS TOO EASY**
KIRBy's identity GroupKFold for DMS datasets is equivalent to random splitting, which is the easiest regime in ProteinGym. This may inflate KIRBy's DMS results relative to harder position-based splits. For a fair comparison with the field, at least modulo or contiguous splitting should be tested.

### Issue 5b: Complex GroupKFold for DDG is Correct but Harsh

**KIRBy's approach for DDG:**
- File: `src/kirby/datasets/ddg_combined.py` lines 690-695
- Groups by PDB complex ID
- This is a leave-one-complex-out approach

**Verdict: CORRECT BUT EXPLAINS POOR PERFORMANCE**
This is the strictest possible splitting for DDG data. It is methodologically correct (prevents data leakage), but it is why R^2 is negative. Most published DDG methods that report high Pearson correlations use weaker splitting (e.g., random train/test split of individual mutations), which allows information leakage from the same complex.

---

## 6. IgBERT Predicts Mean Everywhere

### Known Issue
IgBERT produces R^2 ~ 0 on all datasets in KIRBy's survey.

### Evidence from Literature
- IgBERT (Exscientia/IgBert) is a paired antibody language model based on RoFormer architecture
- It was pre-trained on unpaired sequences first, then fine-tuned on paired data
- The IgBERT/IgT5 paper (https://arxiv.org/html/2403.17889v1) reports that paired models improve binding energy prediction specifically when given natively paired heavy+light input
- However, the paper evaluates IgBERT on **expression** and **binding** tasks with supervised fine-tuning (not just frozen embeddings + Ridge)

### KIRBy's Implementation
- File: `src/kirby/representations/antibody.py` lines 395-421
- Uses HuggingFace `Exscientia/IgBert`
- Passes heavy and light as a text pair: `tokenizer(h_batch, l_batch, ...)`
- Mean-pools over all tokens (including [SEP] tokens between chains)
- Returns 768D embeddings

### Possible Causes
1. **Mean pooling dilutes signal.** IgBERT uses RoFormer architecture with rotary position embeddings. The [CLS] token or chain-specific pooling might be more appropriate than mean-pooling over all tokens.
2. **Frozen embeddings insufficient.** The IgBERT paper shows performance gains from supervised fine-tuning, not from frozen embedding extraction + linear probe.
3. **Model size.** IgBERT is 110M parameters — much smaller than ESM-2 650M. Its embeddings may be less informative for linear probing.

### Verdict: LIKELY A POOLING/ARCHITECTURE MISMATCH
IgBERT was designed for fine-tuning, not linear probing on frozen embeddings. The mean-pooling approach may not extract useful features from this specific architecture. This is not necessarily a KIRBy bug, but it means IgBERT results are not representative of what IgBERT can actually achieve.

---

## 7. FLAb Inter-family Near-Zero Results

### What KIRBy Observes
Many FLAb inter-family datasets (thermostability, aggregation, polyreactivity, PK, immunogenicity) show near-zero R^2.

### What FLAb2 Shows
FLAb2 (bioRxiv 2025.12.27.696706) explicitly states:
- "Protein AI Models Cannot Yet Consistently Predict Developability Properties"
- "No models are able to correlate with all properties or across multiple datasets of similar properties"
- Fine-tuning improves performance for thermostability, aggregation, and binding with 100+ samples
- But polyreactivity and PK "lack enough data for significance"
- "Training data composition influences performance more than model architecture"
- "Intrinsic biophysical properties (thermostability) are more readily learned than extrinsic properties (immunogenicity, PK)"

### FLAb2 Key Finding
**"Given enough developability data (10^3 points), a fine-tuned one-hot encoding model can match the performance of fine-tuned billion-parameter pretrained models."**

This means:
1. Near-zero R^2 for PK, polyreactivity, and immunogenicity is expected
2. These properties may be fundamentally unpredictable from sequence alone
3. KIRBy's results are consistent with the FLAb2 benchmark

### Verdict: EXPECTED BEHAVIOUR
Near-zero inter-family results on extrinsic properties (PK, polyreactivity, immunogenicity) are consistent with FLAb2. Only thermostability and binding tend to be learnable, and only with enough data.

---

## 8. AbAgym Poor Performance

### What KIRBy Observes
AbAgym datasets show poor ML performance.

### What AbAgym's Authors Found
From https://github.com/3BioCompBio/Abagym and the published paper:
- "Machine learning models fail to produce statistically meaningful predictions"
- "ML models have been overtrained and do not generalize outside their training set"
- FoldX (physics-based) and BeAtMuSiC (statistical) outperform all ML models
- This was independently confirmed by the AbDesign study

### Why ML Fails on AbAgym
1. **AbAgym is Ab-Ag binding DDG** — same fundamental problem as DDG_combined
2. **ML models trained on SKEMPI overfit** to the specific complexes in SKEMPI
3. **Antibody CDR mutations affect binding through 3D contacts** that sequence-only models cannot capture
4. **FoldX uses explicit energy calculations** on the 3D structure

### Verdict: EXPECTED — PUBLISHED FINDING
AbAgym poor performance is a published finding, not a KIRBy bug. Physics-based methods outperform ML for Ab-Ag binding DDG prediction.

---

## 9. Metric Choice: R^2 vs Spearman

### KIRBy's Approach
- Uses R^2 exclusively (file: `tests/antibody_baseline_survey.py` line 318)
- User directive: "R^2 only for antibody — NO Spearman rho"

### What the Field Uses
| Benchmark | Primary Metric | Secondary |
|-----------|---------------|-----------|
| ProteinGym | Spearman rho | MSE |
| FLAb/FLAb2 | Spearman rho | — |
| AbBiBench | Spearman rho | Precision@10 |
| AbAgym | Pearson r | Spearman rho |
| Kermut | Spearman rho | MSE, NLL |
| VespaG | Spearman rho | — |

### Why This Matters
- R^2 penalises predictions that are monotonically correct but wrong in scale/offset
- Spearman only cares about ranking (monotonic relationship)
- For fitness prediction, ranking is usually what matters (which variant is best?)
- R^2 can be negative (worse than mean), Spearman cannot
- A model with Spearman 0.5 (useful ranking) might have R^2 = 0.1 or even R^2 < 0

### Verdict: R^2 IS NON-STANDARD AND PESSIMISTIC
Every major antibody/protein fitness benchmark uses Spearman rho. KIRBy's R^2-only approach makes results look worse than they might be on Spearman. The DDG R^2 = -0.137 might correspond to a positive Spearman (i.e., the ranking is partially correct even though the scale is wrong). The user's directive to use R^2 only is noted, but this creates a comparison gap with all published benchmarks.

---

## 10. Missing Methodological Elements

### 10a. Zero-shot Score Augmentation
- **ProteinGym's supervised baselines** augment Ridge regression with zero-shot fitness scores (from ESM-1v, MSA Transformer, etc.)
- This is a standard technique that significantly improves supervised performance
- KIRBy does not include zero-shot fitness scores as input features

### 10b. Layer Selection as Hyperparameter
- The domain-specific vs general PLMs paper (https://www.immunoinformaticsjournal.com/article/S2667-1190(24)00006-5/fulltext) found that "the last layer of PLMs is not necessarily the one yielding the highest performance"
- KIRBy uses the last layer for all models except `esm2_midlayers` (layers 20-33)
- Systematic layer search could improve performance

### 10c. No Inverse Folding Models
- AbBiBench's top performers are inverse folding models (ProteinMPNN, ESM-IF, AntiFold)
- These models encode the entire Ab-Ag complex as a global representation
- KIRBy does not include any inverse folding models in its antibody pipeline
- Note: These require 3D structure, which may be outside scope

### 10d. No Paired-Aware Models Beyond AbLang2/IgBERT
- The IgBERT/IgT5 paper showed that paired models (trained on natively paired H+L) significantly outperform unpaired models for binding prediction
- KIRBy uses paired input for AbLang2 and IgBERT, but embeds H+L separately for ESM-2, ESM-C, ProtT5, AMPLIFY, AntiBERTy
- Concatenating separately-embedded chains may lose cross-chain interaction information

---

## 11. Summary of Verdicts

| Issue | Verdict | Severity | Action Needed |
|-------|---------|----------|---------------|
| DDG R^2 = -0.137 | **NOT A BUG** — consistent with published literature | Low | Consider reporting Spearman alongside R^2; add note in results |
| Ridge as downstream model | **UNDERPOWERED** — field uses GP/MLP/RF | Medium | Consider adding simple MLP option; KIRBy's RF/XGBoost partially address this |
| Identity GroupKFold for DMS | **SPLITTING TOO EASY** — equivalent to random split | High | Should test position-based splits (modulo/contiguous) for DMS datasets |
| IgBERT predicts mean | **POOLING/ARCHITECTURE MISMATCH** — not representative | Low | Consider [CLS] pooling or supervised fine-tuning for IgBERT |
| FLAb inter-family near-zero | **EXPECTED** — consistent with FLAb2 | None | No action needed |
| AbAgym poor performance | **EXPECTED** — published finding | None | No action needed |
| R^2 as sole metric | **NON-STANDARD AND PESSIMISTIC** | Medium | User directive acknowledged; creates comparison gap with all benchmarks |
| No zero-shot augmentation | **MISSING** — standard ProteinGym technique | Medium | Could significantly improve supervised performance |
| No inverse folding models | **MISSING** — top performers in AbBiBench | Low | Requires structure; may be out of scope |
| Complex GroupKFold for DDG | **CORRECT** — methodologically sound | None | No action needed |

### Key References by Topic

**DDG prediction difficulty:**
- DDGb_bias: https://github.com/3BioCompBio/DDGb_bias
- Hummer et al. 2025: https://www.nature.com/articles/s43588-025-00823-8
- Graphinity: https://github.com/oxpig/Graphinity
- CATH-ddG: https://github.com/ak422/CATH-ddG
- StaB-ddG: https://github.com/LDeng0205/StaB-ddG

**Supervised protein fitness prediction:**
- ProteinGym: https://github.com/OATML-Markslab/ProteinGym
- Kermut: https://github.com/petergroth/kermut
- ProteinNPT: https://github.com/OATML-Markslab/ProteinNPT
- VespaG: https://github.com/JSchlensok/VespaG
- EvolvePro: https://github.com/mat10d/EvolvePro

**Antibody fitness benchmarks:**
- FLAb: https://github.com/Graylab/FLAb
- FLAb2: https://www.biorxiv.org/content/10.64898/2025.12.27.696706v1
- AbBiBench: https://github.com/MSBMI-SAFE/AbBiBench
- AbAgym: https://github.com/3BioCompBio/Abagym

**Domain-specific vs general PLMs:**
- https://www.immunoinformaticsjournal.com/article/S2667-1190(24)00006-5/fulltext

**IgBERT:**
- https://arxiv.org/html/2403.17889v1
