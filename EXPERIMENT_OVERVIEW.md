# KIRBy Experiment Overview

Comprehensive reference for all datasets, prediction tasks, representations, and fine-tuning opportunities across all three domains.

---

## 1. DATASETS & PREDICTION TASKS

### 1A. Small Molecules (Regression)

| Dataset | Property | Samples | Source | Splitting |
|---------|----------|---------|--------|-----------|
| **logd** | LogD (lipophilicity) | ~7,309 | OpenADMET-ExpansionRx (HF) | Scaffold |
| **caco2** | Caco2 efflux (P-gp ratio, log) | ~3,777 | OpenADMET-ExpansionRx (HF) | Scaffold |
| **herg** | hERG Ki (pKi) | ~1,415 | ChEMBL API (CHEMBL240) | Scaffold |
| **qm9** | HOMO-LUMO gap | 2,000 (sampled from 134K) | PyTorch Geometric | Scaffold |
| **cox2** | COX-2 IC50 (pIC50) | ~4,000 | ChEMBL (CHEMBL230) | Scaffold |
| **drd2** | DRD2 Ki (pKi) | ~1,000 | ChEMBL (CHEMBL217) | Scaffold |

**Status**: logd, caco2, herg, qm9 are the primary datasets. cox2/drd2 are optional extras.

### 1B. Antibodies (Regression)

#### Intra-Family / DMS (single-family, variable-position reps shine)

| Dataset | Property | Samples | Family | Mutations/sample |
|---------|----------|---------|--------|------------------|
| **flab_expression_koenig** | Expression (enrichment ratio) | 4,275 | G6 SSM | ~1.0 |
| **flab_binding_koenig** | Binding (Kd, nM) | ~2,000 | G6 SSM | ~1.0 |
| **flab_thermo_hie** | Thermostability (Tm, °C) | ~500 | Hie DMS | ~1.0 |
| **flab_binding_rosace** | Binding (adalimumab DMS) | ~200 | Adalimumab | ~1.0 |
| **abagym_Ang2_2017_G6** | Interface ΔΔG | 471 | G6 anti-Ang2 | ~1 |
| **abagym_EGFR_2013_Cetuximab** | Interface ΔΔG | 659 | Cetuximab | ~1 |
| **abagym_lysozyme_2019_D441** | Interface ΔΔG | 627 | D44.1 | ~1 |
| **abagym_VEGF_2017b_G6** | Interface ΔΔG | 513 | G6 anti-VEGF | ~1 |
| **abagym_HER2_2021_trastuzumab** | Interface ΔΔG | 146 | Trastuzumab | ~1 |

#### Inter-Family (diverse antibodies, full-seq LM embeddings appropriate)

| Dataset | Property | Samples | Source |
|---------|----------|---------|--------|
| **abcov_kd** | Binding (-log Kd) | ~1,780 | Rawat et al. 2022 (SARS-CoV-2) |
| **abcov_ic50** | Neutralization (IC50) | ~1,780 | Rawat et al. 2022 |
| **flab_thermo_jain** | Thermostability (Tm) | ~400 | Jain et al. 2017 |
| **flab_aggregation_jain** | Aggregation (AC-SINS) | ~400 | Jain et al. 2017 |
| **abagym_all** | Pooled interface ΔΔG | 2,247 | 5 families pooled |
| **abbind** | ΔΔG (kcal/mol) | ~230 | Sirin et al. 2016 |
| **sabdab_affinity** | SKEMPI ΔΔG | small | SKEMPI 2.0 |

### 1C. TCGA Cancer Genomics

| Dataset | Task | Samples | Cancer | Source |
|---------|------|---------|--------|--------|
| **tcga_luad_vs_lusc** | Binary classification | ~1,016 | Lung | Xena (2 cohorts) |
| **tcga_brca_pam50** | 4-class classification | ~900 | Breast | Xena + PAM50 labels |
| **tcga_brca_survival** | Regression (OS days) | ~400-500 | Breast | Xena + TCGA-CDR |

---

## 2. REPRESENTATIONS

### 2A. Small Molecules

#### Hand-Crafted (no training)
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **ecfp4** | 2048D | Morgan circular fingerprints (radius=2) |
| **maccs** | 167D | Structural substructure keys |
| **pdv** | 200D | RDKit physicochemical descriptors |

#### Pretrained Embeddings (frozen, no fine-tuning)
| Rep | Dim | Architecture | Training data |
|-----|-----|-------------|---------------|
| **mol2vec** | 300D | Word2Vec on substructures | Substructure vocabulary |
| **chemberta** | 768D | RoBERTa (77M params) | 77M molecules (SMILES) |
| **molformer** | 768D | Transformer (linear attn) | 1.1B molecules (SMILES) |
| **selformer** | 768D | RoBERTa on SELFIES | SELFIES representations |
| **chembert** | 256D | BERT (8 layers) | ChEMBL33 (SMILES) |
| **mhg_gnn** | variable | GNN (IBM) | Graph structures |
| **unimol** | 512D | 3D conformer GNN | 209M conformations |
| **chemformer** | 512D | BART encoder-decoder | SMILES denoising |
| **molclr** | 512D | GIN/GCN contrastive | Graph augmentations |
| **graphmvp** | 300D | 2D+3D GNN | 2D/3D alignment |
| **smited** | variable | Encoder-decoder | 91M SMILES (PubChem) |
| **grover** | 5000D | Graph Transformer | 10M molecules |
| **schnet** | 128D | 3D equivariant GNN | QM9 pretraining |

#### Learnable (fitted on data)
| Rep | Dim | Notes |
|-----|-----|-------|
| **sns** (Sort & Slice) | 1024D | Fitted pooling on training data |
| **graph_kernel** (WL) | ~290D | Fitted vocabulary, sparse |
| **finetune_gnn()** | variable | Train GCN/GAT/MPNN from scratch |

#### Presets
| Preset | Reps | Count |
|--------|------|-------|
| **fast** | ecfp4, maccs, pdv | 3 |
| **fast+hf** | + chemberta, molformer, selformer | 6 |
| **all** | All 15 representations | 15 |

### 2B. Antibodies

#### Full-Sequence Embeddings (pretrained, frozen)
| Rep | Dim | Model | Best for |
|-----|-----|-------|----------|
| **ablang2** | 512D | AbLang-2 (paired) | Inter-family |
| **antiberty** | 768D | AntiBERTy v2 (RoFormer) | Inter-family |
| **esm2** | 1280D | ESM-2 (general protein) | Inter-family |
| **prot_t5** | 1024D | ProtT5-XL | Inter-family |
| **igbert** | 768D | IgBERT | Inter-family |
| **ab_roberta** | 768D | AbRoBERTa | Inter-family |
| **balm** | 512D | BALM (2024) | Inter-family |
| **sapiens** | 1024D | SAPIENS | Inter-family |
| **pigen** | 640D | pIgGen | Inter-family |
| **esmc** | 1536D | ESM-C | Inter-family |

#### Hand-Crafted (classical)
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **kmer** | 400D | Dipeptide frequencies |
| **aac** | 20D | Amino acid composition |
| **blosum62** | 20D | Mean-pooled BLOSUM62 |
| **physicochemical** | 68D | 6 properties × 4 stats × 2 chains + group fractions |
| **imgt_onehot** | 5120D | 128 IMGT positions × 20 AAs × 2 chains |
| **developability** | ~20D | Charge, hydro, liabilities |

#### Mutation-Aware (DMS-specific)
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **positional** | 11D | CDR region, BLOSUM score, position |
| **biophysical_delta** | 6D | ΔCharge, ΔHydro, ΔMW, ΔVol, ΔFlex, ΔPol |
| **lm_score_ablang2** | 5D | AbLang2 log-prob mutation scores |
| **lm_score_esm2** | 5D | ESM-2 log-prob mutation scores |
| **lm_score_prot_t5** | 5D | ProtT5 mutation scores |
| **lm_score_antiberty** | 5D | AntiBERTy mutation scores |
| **esm2_site** | 1280D | ESM-2 hidden state at mutation site |

#### Variable-Position Encodings (NEW — DMS-specific)
| Rep | Dim (flab_expr_koenig) | Region | Encoding |
|-----|----------------------|--------|----------|
| **varpos_onehot** | 4200D | All | 20D one-hot/pos |
| **varpos_blosum** | 4200D | All | 20D BLOSUM row/pos |
| **varpos_physchem** | 1400D | All | 6D physchem/pos |
| **varpos_cdr_onehot** | 987D | CDR only | 20D one-hot/pos |
| **varpos_cdr_physchem** | 329D | CDR only | 6D physchem/pos |
| **varpos_cdrh3_physchem** | 56D | CDR-H3 only | 6D physchem/pos |
| **varpos_fw_physchem** | 1246D | Framework only | 6D physchem/pos |

#### Structural (requires ImmuneBuilder)
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **structural** | 28D | SASA (15D) + confidence (5D) + paratope (8D) |

#### Presets
| Preset | Count | Best for |
|--------|-------|----------|
| **fast** | 4 | Quick baseline (kmer, physco, aac, blosum) |
| **fast+lm** | 7 | + antiberty, ablang2, igbert |
| **dms** | 9 | Intra-family DMS (LM scores + varpos + CDR varpos) |
| **dms_full** | 17 | Full DMS arsenal (all varpos variants + LM scores + esm2_site) |
| **mutation_fast** | 7 | Legacy DMS + varpos_physchem |
| **mutation_full** | 10 | Legacy DMS + varpos_onehot/physchem |
| **inter_family** | 6 | Inter-family (ablang2, esm2, prot_t5, physchem, aac, dev) |
| **all_server** | 13 | All including structural |

### 2C. TCGA

#### Expression-Derived
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **gene_hvg2000** | 2000D | Highly variable genes |
| **gene_pca100** | 100D | PCA of expression |
| **gene_nmf50** | 50D | NMF (parts-based) |
| **gene_ica50** | 50D | ICA (independent components) |
| **gene_stats** | 15D | Global transcriptomic shape |

#### Pathway/Signature Scores
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **gene_hallmark** | 50D | MSigDB Hallmark (ssGSEA) |
| **gene_kegg** | ~186D | KEGG pathways (ssGSEA) |
| **gene_reactome** | ~600D | Reactome pathways (ssGSEA) |
| **gene_progeny** | 14D | 14 signaling pathways |
| **gene_dorothea** | ~300D | TF activity (DoRothEA regulons) |
| **gene_combined_pathway** | ~364D | PROGENy + Hallmark + DoRothEA |
| **gene_immune** | ~27D | 24 immune cell types + ESTIMATE |

#### Multi-Omic
| Rep | Dim | What it captures |
|-----|-----|-----------------|
| **gene_mutation** | 200D | Top-200 mutated genes (binary) |
| **cnv_gistic** | 200D | Copy number (GISTIC2) |
| **cnv_pca50** | 50D | PCA of CNV |
| **rppa** | ~260D | Protein expression (RPPA panel) |
| **mirna_top200** | 200D | Top-200 variable miRNAs |
| **clinical** | 10D | Age, sex, TNM, grade, etc. |

#### WSI / Histopathology (pretrained, frozen)
| Rep | Dim | Model | Availability |
|-----|-----|-------|-------------|
| **wsi_ctranspath** | 512D | GigaSSL | Public (Mendeley) |
| **wsi_phikon** | 512D | GigaSSL | Public |
| **wsi_hoptimus0** | 512D | GigaSSL | Public |
| **wsi_gigapath** | 512D | GigaSSL | Public |
| **wsi_moco** | 512D | GigaSSL (MoCo v2) | Public |
| **wsi_titan** | 768D | TITAN (MahmoodLab) | Gated HF |
| **wsi_uni2h** | 1536D | UNI2-h (MahmoodLab) | Gated HF |
| **wsi_cnn320** | 320D | Custom CNN | Local only |

#### Presets
| Preset | Count | Contents |
|--------|-------|---------|
| **fast** | 5 | pca100, hallmark, progeny, ctranspath, phikon |
| **fast+diverse** | 8 | + nmf50, gene_stats, immune |
| **multimodal** | 12 | Expression + mutation + CNV + RPPA + miRNA + clinical + WSI |
| **fast+learned** | 15 | All expression pathways + 5 WSI |
| **all** | 27 | Everything |
| **genomic_fast** | 3 | No WSI: pca100, hallmark, progeny |
| **genomic_diverse** | 10 | No WSI: multi-omic |
| **genomic_all** | 18 | No WSI: comprehensive |

---

## 3. CURRENT STATUS: WHAT'S PRETRAINED vs. FINE-TUNED

**Every pretrained model in the codebase is used FROZEN. No fine-tuning happens.**

The pattern everywhere is:
```
Load pretrained model → model.eval() → torch.no_grad() → forward pass → mean pool → numpy
```

Fine-tuning infrastructure exists (`finetune_antibody_lm()`, `finetune_gnn()`) but is unused.

---

## 4. FINE-TUNING

### 4.1 The Problem with Frozen Embeddings

Every pretrained model in KIRBy is currently used **frozen** — no gradients, no adaptation.
This works well for inter-family datasets (diverse sequences) but fails badly for
intra-family DMS datasets where 99% of the sequence is invariant. Published evidence:
frozen ESM-2 achieves ρ ≈ 0.08 on DMS fitness prediction.

### 4.2 Three Fine-Tuning Approaches

#### Approach A: Full Sequence In, Site-Specific Extraction Out (Recommended First)

Feed the full sequence to preserve positional context, but extract embeddings
only at variable/mutation positions.

```
Input:  Full 120-AA sequence (preserves context for attention layers)
Model:  ESM-2 / AbLang2 / AntiBERTy (with fine-tuning head)
Head:   Small MLP on site-specific embeddings → predict fitness
Output: Per-residue hidden states at variable positions → concatenate → rep

Fine-tuning loss: MSE(predicted_fitness, actual_fitness)
What changes: The model learns that mutations at certain positions matter
What's preserved: Full sequence context (critical for attention patterns)
```

This is essentially what `esm2_site` does at inference, but frozen. Fine-tuning
teaches the model to produce *task-relevant* site embeddings.

**Pros**: Most correct biologically; preserves positional context; models expect full sequences.
**Cons**: Full sequence through the model = full compute cost per forward pass.

#### Approach B: Masked Mutation Prediction (Zero-Shot Transfer)

Fine-tune the LM's masked language modeling objective on your sequences,
then use log-probabilities as the representation.

```
Input:  Full sequence with mutation site replaced by [MASK]
Model:  ESM-2 / AbLang2 (MLM head)
Task:   Predict the correct AA at the masked position
Output: log P(mutant_AA | context) - log P(wildtype_AA | context) = ΔlogP

Fine-tuning loss: Cross-entropy on masked position prediction
What changes: The model learns dataset-specific context preferences
What's preserved: The MLM objective (no new head needed)
```

This is how ESM-1v (Meier et al. 2021) works zero-shot. Fine-tuning on your
specific family's sequences improves the context model.

**Pros**: No new architecture needed; naturally produces interpretable scores; transfers well.
**Cons**: Only captures local mutation effects; may miss epistatic interactions.

#### Approach C: Contrastive Fine-Tuning (Maximum Discrimination)

Train the model so sequences with similar fitness get similar embeddings,
and sequences with different fitness get pushed apart.

```
Input:  Pairs of full sequences (mutant_i, mutant_j)
Model:  ESM-2 / AbLang2 → mean-pooled embedding
Loss:   Contrastive: ||emb_i - emb_j||² ∝ |fitness_i - fitness_j|
Output: Fine-tuned mean-pooled embeddings (same dim as frozen)

What changes: The embedding space is reorganized around fitness
What's preserved: Full sequence input, model architecture
```

**Pros**: Most flexible; works for any property; embeddings directly encode fitness structure.
**Cons**: Requires careful pair sampling; more compute (pairwise training); harder to tune.

### 4.3 Candidates by Domain

#### Small Molecules

| Model | Dim | Approach | Expected impact | Difficulty |
|-------|-----|----------|-----------------|------------|
| **ChemBERTa** | 768D | A or C (LoRA fine-tune on target property) | Medium | Low (HF native) |
| **MolFormer** | 768D | A or C (LoRA, trained on 1.1B molecules) | Medium | Low (HF native) |
| **SELFormer** | 768D | A or C (SELFIES-based) | Medium | Low |
| **SchNet** | 128D | A (3D GNN, native to QM9) | High for QM9 | Medium |
| **GNN from scratch** | variable | Train fresh (already implemented) | Baseline | Low |

#### Antibodies

| Model | Dim | Approach | Expected impact | Difficulty |
|-------|-----|----------|-----------------|------------|
| **ESM-2** | 1280D | A (site-specific) or B (masked) | **HIGH for DMS** | Low (HF) |
| **AntiBERTy** | 768D | A (infrastructure exists!) | High | Low |
| **AbLang2** | 512D | A or B (antibody-specific) | High | Medium |
| **ProtT5** | 1024D | A (LoRA) | Medium-High | Low |

#### TCGA

| Model | Domain | Approach | Expected impact | Difficulty |
|-------|--------|----------|-----------------|------------|
| **Geneformer** | Gene expression | Fine-tune on cancer classification | High | High |
| **scGPT** | Gene expression | Fine-tune on LUAD/LUSC, PAM50 | High | High |
| **CONCH** | Pathology WSI | Fine-tune slide encoder | Medium-High | High |
| **GigaSSL** | WSI patches | Train attention aggregator on top | Medium | Medium |

### 4.4 Workflow: Do It Once, Save, Reuse

```
┌─────────────────────────────────────────────────────────┐
│ FINE-TUNING WORKFLOW (per dataset)                       │
│                                                         │
│ 1. FINE-TUNE (once per dataset, train fold only)        │
│    model = load_pretrained("esm2")                      │
│    model = finetune(model, train_seqs, train_labels,    │
│                     approach="site_specific",            │
│                     epochs=10, lr=2e-5)                  │
│    save("~/.cache/kirby/finetuned/esm2_flab_expr.pt")   │
│                                                         │
│ 2. EXTRACT EMBEDDINGS (all sequences)                   │
│    model = load("~/.cache/kirby/finetuned/esm2_*.pt")   │
│    embeddings = model.encode(all_sequences)  # (N, D)   │
│    np.save("~/.cache/kirby/reps/<hash>/esm2_ft.npy",    │
│            embeddings)                                   │
│                                                         │
│ 3. USE AS REP (identical to any other cached rep)       │
│    reps["esm2_finetuned"] = np.load("esm2_ft.npy")     │
│    → Participates in hybrid search like any other rep   │
│                                                         │
│ 4. COMPARE                                              │
│    reps["esm2_frozen"]    = 1280D (baseline)            │
│    reps["esm2_finetuned"] = 1280D (should be better)    │
│    → Greedy search picks whichever is more useful       │
└─────────────────────────────────────────────────────────┘
```

**Key rules:**
- Fine-tune on **train fold only** (no leakage)
- Use **inner CV** for hyperparameter selection (lr, epochs, LoRA rank)
- Cache the fine-tuned model checkpoint AND the extracted embeddings
- Report as a **separate rep** alongside the frozen version
- Let the hybrid search decide: does fine-tuned actually win?

### 4.5 Implementation Priority

1. **Antibody Approach A** — highest impact, `finetune_antibody_lm()` already exists
   for AntiBERTy. Wire it in, add ESM-2 equivalent. Test on flab_expression_koenig.
2. **Molecular Approach A** — ChemBERTa/MolFormer LoRA on logd. HF-native, quick.
3. **Antibody Approach B** — masked mutation prediction. Good for DMS, interpretable.
4. **TCGA Geneformer** — biggest lift for genomics, but highest difficulty.

---

## 5. ADAPTIVE SEARCH

### 5.1 The Problem with Current Search

The current search is an **exhaustive grid** over:
`importance_methods × allocation_methods × budgets × models`

Evidence from existing results shows most of this grid is redundant:

| Search axis | Variance in results | Current effort |
|-------------|-------------------|----------------|
| **Which reps** | HIGH | Not searched (fixed by preset) |
| **Which model** | MEDIUM | Searched exhaustively |
| **Budget** | LOW-MEDIUM | Searched exhaustively |
| **Importance method** | ~ZERO (all give identical rankings) | Searched exhaustively |
| **Allocation method** | ~ZERO (greedy wins 85%+) | Searched exhaustively |

The search spends most compute on axes that don't matter, and zero effort on the
axis that matters most (which reps to include).

### 5.2 Adaptive Successive Halving

Instead of exhaustive grid search, use **data-driven pruning** to focus compute
on what matters. Configurable via `rep_survival_percentile` (default: 0.6 = top 60%).

```
┌──────────────────────────────────────────────────────────┐
│ STEP 1: CHEAP INDIVIDUAL SCREEN                          │
│   Cost: N evaluations (N = number of reps)               │
│   For each rep:                                          │
│     score_solo[rep] = RF, 3-fold CV, full-dim            │
│   → Produces: individual ranking of all reps             │
│                                                          │
│ STEP 2: CHEAP GREEDY WITH ALL REPS                       │
│   Cost: 1 evaluation                                     │
│   Run greedy search: RF, 3-fold, budget=100, treeshap    │
│   → Produces: which reps got allocated (complementarity) │
│                                                          │
│ STEP 3: ADAPTIVE PRUNING                                 │
│   A rep SURVIVES if EITHER:                              │
│     (a) Individual viability:                            │
│         score_solo[rep] ≥ percentile(all_scores,         │
│                             rep_survival_percentile)     │
│     (b) Complementarity:                                 │
│         greedy allocated it features in Step 2           │
│   Keep the UNION → preserves diversity                   │
│   Only drops reps that are both poor AND unselected      │
│                                                          │
│ STEP 4: FOCUSED SEARCH WITH SURVIVORS                    │
│   Models: RF, Ridge, LightGBM (the 3 that matter)       │
│   Budgets: 50, 100                                       │
│   Importance: treeshap only (others are identical)       │
│   Allocation: greedy only (others barely differ)         │
│   Reps: survivors from Step 3 only                       │
│   Full CV: 5-fold                                        │
│   Cost: 3 models × 2 budgets × 5 folds = 30 evals      │
│         (vs hundreds in current exhaustive search)       │
└──────────────────────────────────────────────────────────┘
```

### 5.3 Why UNION of Criteria (Not Intersection)

A rep with poor solo R² might be the best *complement* to the top rep because
it captures orthogonal signal. Example from molecular results:
- ecfp4 solo: R²=0.65 (mediocre, 2048D fingerprint)
- molformer solo: R²=0.72 (best individual)
- ecfp4 + molformer hybrid: R²=0.76 (best pair — ecfp4 adds structural coverage)

Using intersection (must be individually good AND greedy-selected) would kill
ecfp4. Using union preserves it because greedy selected it.

### 5.4 Pruning Threshold

```python
rep_survival_percentile = 0.6   # configurable, default top 60%

# Example with 10 reps:
# solo R²: [0.72, 0.70, 0.68, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
# 60th percentile = 0.55
# Individual survivors (≥ 0.55): top 6
# Greedy-selected (got features): maybe 4 reps, possibly including #8 (0.45)
# UNION: 6 + any greedy-selected not already in top 6 = maybe 7 survivors
# PRUNED: 3 reps that are both poor individually and never selected
```

With fewer reps (e.g., 5 in `fast` preset), pruning is gentler — most survive.
With many reps (e.g., 17 in `dms_full`), pruning is more aggressive — exactly
what you want.

### 5.5 Cost Comparison

| Search mode | Configurations evaluated | Notes |
|-------------|------------------------|-------|
| Current `fast` | 8 (2 models × 4 importance × 1 alloc × 1 budget) | Wastes effort on importance |
| Current `medium` | ~360 (4 × 6 × 3 × 3) | Most configs are redundant |
| Current `long` | ~1792 (7 × 8 × 6 × 4) | Massive waste |
| **Adaptive** | N + 1 + 30 (screen + greedy + focused) | Data-driven, ~10× faster |

### 5.6 Configuration

```python
find_best_hybrid(
    base_reps=reps,
    labels=labels,
    speed='adaptive',                   # NEW: triggers adaptive search
    rep_survival_percentile=0.6,        # top 60% individually viable
    # Still respects these if provided:
    models=['rf', 'ridge', 'lightgbm'], # override default model set
    budgets=[50, 100],                  # override default budgets
)
```

---

## 6. SUMMARY TABLE

| Domain | Datasets | Hand-crafted | Pretrained (frozen) | Fine-tunable | Varpos |
|--------|----------|-------------|---------------------|--------------|--------|
| **Molecules** | 6 | 3 | 13 (LMs + GNNs) | ChemBERTa, MolFormer, SELFormer, SchNet, GNN | N/A |
| **Antibodies** | 16 | 7 | 10 (LMs) + 4 (LM scores) | AntiBERTy (exists!), ESM-2, AbLang2, ProtT5 | 7 variants |
| **TCGA** | 3 | 19 | 8 (WSI encoders) | Geneformer, scGPT, WSI aggregator | N/A |
| **TOTAL** | **25** | **29** | **35** | **8+ candidates** | **7** |
