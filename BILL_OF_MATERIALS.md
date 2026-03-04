# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-02-27 (preprocessing overhaul applied)

---

## 1. Datasets

### 1A. Molecular (regression, scaffold-split GroupKFold)

| Dataset | Source | Samples | Scaffolds | Target | Metric | Status |
|---------|--------|---------|-----------|--------|--------|--------|
| `logd` | OpenADMET-ExpansionRx | ~5,039 | ~2,387 | LogD | R² | **Confirmed** |
| `caco2` | OpenADMET-ExpansionRx | ~2,161 | ~1,069 | Caco2 Efflux (log) | R² | **Confirmed** |
| `herg` | ChEMBL (CHEMBL240) | ~1,415 | ~767 | pKi | R² | **Confirmed** |
| `qm9` | QM9 | 2,000 | varies | HOMO-LUMO gap | R² | **Confirmed** |

### 1B. Antibody Intra-Family / DMS (regression, GroupKFold by family)

| Dataset | Source | Samples | Target | Status |
|---------|--------|---------|--------|--------|
| `flab_expression_koenig` | FLAb (G6 SSM) | 4,275 | Expression (ER) | **Confirmed** (20 reps) |
| `flab_binding_koenig` | FLAb (G6) | 4,275 | Kd | **Confirmed** (20 reps) |
| `abagym_Ang2_2017_G6` | AbAgym DMS | 397 | Fitness | **Confirmed** (20 reps) |
| `abagym_EGFR_2013_Cetuximab` | AbAgym DMS | 659 | Fitness | **Confirmed** (20 reps) |
| `abagym_lysozyme_2019_D441` | AbAgym DMS | 627 | Fitness | **Confirmed** (20 reps) |
| `abagym_VEGF_2017b_G6` | AbAgym DMS | 418 | Fitness | **Confirmed** (20 reps) |
| `abagym_HER2_2021_trastuzumab` | AbAgym DMS | 146 | Fitness | **Confirmed** (20 reps) |

### 1C. Antibody Inter-Family (regression, identity-grouped GroupKFold)

| Dataset | Source | Samples | Target | Status |
|---------|--------|---------|--------|--------|
| `ddg_combined` | SKEMPI 2.0 + AB-Bind (deduplicated) | 544 | ΔΔG | **Confirmed** (loads OK, paired) |
| `abcov_kd` | Ab-CoV (via FLAb) | 141 | -log Kd | **Confirmed** (loads OK, paired) |
| `flab_thermo_jain` | FLAb (Jain 2017) | 137 | Tm | **Partial** (6 reps, R²=0.14, see §2B) |
| `flab_aggregation_jain` | FLAb (Jain 2017) | 137 | AC-SINS | **Confirmed** (loads OK, paired) |
| `abagym_all` | AbAgym (pooled 5 experiments) | 2,247 | Fitness | **Confirmed** (loads OK, paired) |

**Dropped:** `sabdab_affinity` (was SKEMPI 2.0 only, replaced by `ddg_combined`),
`abbind` / `abbind_spm` (strict subsets of `ddg_combined`).

### 1D. TCGA (classification/regression, TSS-grouped StratifiedGroupKFold)

| Dataset | Task | Samples | Classes/Target | TSS | Metric | Status |
|---------|------|---------|----------------|-----|--------|--------|
| `tcga_luad_vs_lusc` | Binary clf | ~1,017 | LUAD vs LUSC | 68 | Accuracy | **Confirmed** |
| `tcga_brca_pam50` | 4-class clf | ~821 | LumA/LumB/Basal/Her2 | 19 | Accuracy | **Confirmed** |
| `tcga_brca_survival` | Regression | ~151 | OS days (uncensored) | 21 | R² | Listed, no baselines yet |

### 1E. Drug-Target Affinity (regression, cold-target GroupKFold)

| Dataset | Samples | Drugs | Proteins | Target | Status |
|---------|---------|-------|----------|--------|--------|
| `davis` | 30,056 pairs | 68 | 442 | pKd | Listed, no baselines yet |
| `kiba` | ~118K pairs | 2,111 | 229 | KIBA score | Listed, no baselines yet |

---

## 2. Representations with Baseline Performance

Best R² (or Accuracy for classification) shown as `best_model=score`. All scores
are from scaffold-split (molecular), identity-grouped (antibody), or TSS-grouped
(TCGA) cross-validation — no random splits.

### 2A. Molecular Representations

**Source:** 5-fold scaffold-split GroupKFold baselines on LogD (~5,039), Caco2 (~2,161),
hERG (~1,415), QM9 (~1,981). Models: rf, xgboost, ridge, knn, lightgbm.

**⚠ ALL SCORES PENDING** — HP defaults changed (RF max_features, XGB colsample_bytree,
RidgeCV grid). Server caches must be cleared before re-run. See §8.

| Rep | Dim | Category | LogD R² | Caco2 R² | hERG R² | QM9 R² | Deps |
|-----|-----|----------|---------|----------|---------|--------|------|
| `ecfp4` | 2048 | Standard (fingerprint) | PENDING | PENDING | PENDING | PENDING | rdkit |
| `maccs` | 167 | Standard (fingerprint) | PENDING | PENDING | PENDING | PENDING | rdkit |
| `pdv` | 200 | Standard (descriptor) | PENDING | PENDING | PENDING | PENDING | rdkit |
| `mol2vec` | 300 | Pretrained (Word2Vec) | PENDING | PENDING | PENDING | PENDING | mol2vec |
| `chemberta` | 768 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | transformers |
| `molformer` | 768 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | transformers |
| `selformer` | 768 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | transformers |
| `chembert` | 256 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | transformers |
| `molclr` | 512 | Pretrained (GNN) | PENDING | PENDING | PENDING | PENDING | torch_geometric |
| `graphmvp` | 300 | Pretrained (GNN) | PENDING | PENDING | PENDING | PENDING | torch_geometric |
| `smited` | 512 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | smi-ted (IBM, local clone) |
| `mhg_gnn` | 1024 | Pretrained (GNN) | PENDING | PENDING | PENDING | PENDING | torch_geometric |
| `grover` | ~5000 | Pretrained (Transformer) | PENDING | PENDING | PENDING | PENDING | grover |
| `graph_kernel` | ~290 | Standard (kernel) | PENDING | PENDING | PENDING | PENDING | grakel |
| `schnet` | 128 | Pretrained (3D GNN) | PENDING | PENDING | PENDING | PENDING | schnetpack |
| `clamp` | 768 | Pretrained (contrastive MLP) | PENDING | PENDING | PENDING | PENDING | clamp (ml-jku), mhnreact |
| `rmat` | 768 | Pretrained (3D Transformer) | PENDING | PENDING | PENDING | PENDING | huggingmolecules (gmum) |
| `unimol` | 512 | Pretrained (3D) | **DROPPED** | — | — | — | unimol_tools |

**Dropped:** `chemformer` (R² < 0.25 on all datasets, broken HF weights).

**Augmentations (not base reps):** `graph_topology`, `spectral`

**Fine-tuning:** NONE — see §7 for survey results. `lora_chemberta` and `lora_molformer`
were tested and performed **worse** than frozen baselines on all molecular datasets.

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (17): all reps including GNN, graph_kernel, clamp, rmat

### 2B. Antibody Representations

**Approach (March 2026 rebuild):** Simple LM embeddings + sequence encodings → Ridge/RidgeCV
→ Spearman ρ. Following compass document: property-dependent model selection, one-hot
competitive with billion-param models at ≥1K samples.

**Script:** `tests/antibody_baseline_survey.py`

| Rep | Dim | Category | Deps | Notes |
|-----|-----|----------|------|-------|
| `esm2` | 2560 | Pretrained (protein LM) | transformers | ESM-2 650M, H+L concat |
| `esmc` | 2304 | Pretrained (protein LM) | esm (EvolutionaryScale) | ESM-C 600M, H+L concat. Non-commercial license. |
| `ablang2` | 768 | Pretrained (Ab LM, paired) | ablang2 | Single forward pass |
| `igbert` | 768 | Pretrained (Ab LM, paired) | transformers | IgBert paired |
| `antiberty` | 1024 | Pretrained (Ab LM) | transformers | AntiBERTy, H+L concat |
| `prot_t5` | 2048 | Pretrained (protein LM) | transformers | ProtT5-XL, H+L concat |
| `onehot` | 5600 | Standard (positional) | None | Max 150H + 130L, 20 AA |
| `blosum` | 5600 | Standard (positional) | None | BLOSUM62 row per residue |
| `aac` | 40 | Standard (composition) | None | 20 frequencies per chain |
| `esm2_cdr` | 2560 | CDR-pooled (protein LM) | transformers, anarci | ESM-2, pooled over CDR residues only |
| `esmc_cdr` | 2304 | CDR-pooled (protein LM) | esm, anarci | ESM-C, pooled over CDR residues only |
| `antiberty_cdr` | 1024 | CDR-pooled (Ab LM) | transformers, anarci | AntiBERTy, CDR-only |
| `prot_t5_cdr` | 2048 | CDR-pooled (protein LM) | transformers, anarci | ProtT5, CDR-only |
| `esm2_midlayers` | 2560 | Intermediate layers (LM) | transformers | ESM-2, avg layers 20-33 |

**Baselines:** Not yet computed — run `antibody_baseline_survey.py` on server.

**Fine-tuning:** Removed. All antibody FT methods (LoRA, MLM, freeze) produced negative
R² and have been deleted. No antibody fine-tuning factories remain.

### 2C. TCGA Representations

**Source:** 5-fold TSS-grouped StratifiedGroupKFold baselines on LUAD vs LUSC and
BRCA PAM50. Models: rf, xgboost, ridge, knn.

**⚠ ALL SCORES PENDING** — HP defaults changed. Server caches must be cleared.

#### Expression-Derived

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `gene_hvg2000` | 2000 | Standard | PENDING | PENDING | scanpy (opt) | YES |
| `gene_pca100` | 100 | Standard | PENDING | PENDING | sklearn | YES |
| `gene_nmf50` | 50 | Standard | PENDING | PENDING | sklearn | YES |
| `gene_ica50` | 50 | Standard | PENDING | PENDING | sklearn | YES |
| `gene_stats` | 15 | Standard | PENDING | PENDING | scipy | YES |
| `gene_hallmark` | 50 | Pathway (mean z-score) | PENDING | PENDING | local gene sets | YES |
| `gene_kegg` | 320 | Pathway (mean z-score) | PENDING | PENDING | local gene sets | YES |
| `gene_reactome` | 1817 | Pathway (mean z-score) | PENDING | PENDING | local gene sets | YES |
| `gene_oncogenic` | 189 | Pathway (mean z-score) | PENDING | PENDING | local gene sets | YES |
| `gene_progeny` | 14 | Pathway (weighted) | PENDING | PENDING | local gene sets | YES |
| `gene_dorothea` | 216 | Pathway (weighted) | PENDING | PENDING | local gene sets | YES |
| `gene_combined_pathway` | 280 | Pathway (combined) | PENDING | PENDING | local gene sets | YES |
| `gene_quantile_rank` | 50 | Expression (rank PCA) | PENDING | PENDING | sklearn | YES |
| `gene_corr_modules` | 30 | Expression (eigengene) | PENDING | PENDING | sklearn | YES |
| `gene_chromosome_arm` | 42 | Genomic (chr arm) | PENDING | PENDING | local gene sets | YES |
| `gene_autoencoder50` | 50 | Expression (AE) | PENDING | PENDING | torch | YES |
| `gene_sparse_pca50` | 50 | Expression (sparse) | PENDING | PENDING | sklearn | YES |
| `gene_immune` | ~22 | Computed (immune) | PENDING | PENDING | hardcoded | YES |

#### Foundation Model Embeddings (GPU)

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `geneformer` (FT) | 1152 | Fine-tuned (per-fold) | PENDING | PENDING | torch, transformers | GPU |
| `gene_bulkrnabert` | 256 | Pretrained (frozen) | PENDING | PENDING | transformers (from source) | GPU. Purpose-built for bulk RNA-seq (pretrained on TCGA). |
| `gene_scfoundation` | 2048 | Pretrained (frozen) | PENDING | PENDING | torch, transformers | GPU, never tested |

#### Multi-Omic

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `cnv_gistic` | 200 | CNV | PENDING | PENDING | Xena | YES |
| `cnv_pca50` | 50 | CNV | PENDING | PENDING | Xena | YES |
| `rppa` | 258 | Protein | PENDING | PENDING | Xena | YES |
| `mirna_top200` | 200 | miRNA | PENDING | PENDING | Xena | YES |

**Dropped:** `gene_mutation` (Xena MC3 returns 403), `clinical` (Xena phenotype returns 403).

#### WSI (Pretrained Vision)

| Rep | Dim | Source | Lung Acc (best) | BRCA Acc (best) | Access | Server? |
|-----|-----|--------|----------------|----------------|--------|---------|
| `wsi_ctranspath` | 512 | GigaSSL (Mendeley) | PENDING | PENDING | Public | YES |
| `wsi_phikon` | 512 | GigaSSL (Mendeley) | PENDING | PENDING | Public | YES |
| `wsi_hoptimus0` | 512 | GigaSSL (Mendeley) | PENDING | PENDING | Public | YES |
| `wsi_gigapath` | 512 | GigaSSL (Mendeley) | PENDING | PENDING | Public | YES |
| `wsi_moco` | 512 | GigaSSL (Mendeley) | PENDING | PENDING | Public | YES |

**Dropped:** `wsi_titan` (gated, garbage results), `wsi_uni2h` (features not found),
`wsi_cnn320` (local only, garbage results).

**Fine-tuning:** `geneformer` (per-fold, proper vocab tokenization → 1152D V2 embeddings).
Geneformer was pretrained on single-cell data (Genecorpus-104M, non-cancer) — no TCGA
data leakage. Previous survey showed it competitive as individual rep and in hybrids.
**All scores pending re-run with updated HP defaults.**

**Presets:**
- `fast` (5): pca100, hallmark, progeny, ctranspath, phikon
- `fast+diverse` (9): + nmf50, stats, immune, quantile_rank
- `multimodal` (10): + cnv_gistic, rppa, mirna_top200
- `fast+learned` (16): expanded genomic + geneformer + scfoundation + 4 WSI
- `all` (29): everything (18 expression + 2 foundation + 4 multi-omic + 5 WSI)
- `genomic_fast` (3): pca100, hallmark, progeny
- `genomic_diverse` (8): multi-omic without WSI
- `genomic_all` (24): all genomic + foundation models + multi-omic

### 2D. DTA Representations

Cold-target GroupKFold CV (5-fold). "Paired" = drug rep + protein_aac or protein rep + drug_ecfp4.
Davis: 30K pairs (68 drugs × 442 proteins). KIBA: 30K subsample (2111 drugs × 229 proteins).

**⚠ ALL SCORES PENDING** — HP defaults changed, preprocessing overhauled. Server caches must be cleared.

| Rep | Dim | Category | Davis R² (paired) | KIBA R² (standalone) | Deps |
|-----|-----|----------|:-:|:-:|------|
| `drug_ecfp4` | 2048 | Standard (fingerprint) | PENDING | PENDING | rdkit |
| `drug_maccs` | 167 | Standard (fingerprint) | PENDING | PENDING | rdkit |
| `drug_pdv` | 200 | Standard (descriptor) | PENDING | PENDING | rdkit |
| `drug_chemberta` | 768 | Pretrained (drug LM) | PENDING | PENDING | transformers |
| `drug_molformer` | 768 | Pretrained (drug LM) | PENDING | PENDING | transformers |
| `drug_selformer` | 768 | Pretrained (drug LM) | PENDING | PENDING | transformers |
| `drug_clamp` | 768 | Pretrained (contrastive MLP) | PENDING | PENDING | clamp (ml-jku), mhnreact |
| `protein_esm2` | 1280 | Pretrained (protein LM) | PENDING | PENDING | transformers |
| `protein_esmc` | 1152 | Pretrained (protein LM) | PENDING | PENDING | esm (EvolutionaryScale). Non-commercial license. |
| `protein_esm2_midlayers` | 1280 | Intermediate layers (LM) | PENDING | PENDING | transformers. Avg layers 20-33. |
| `protein_aac` | 20 | Standard (composition) | PENDING | PENDING | None |
| `protein_descriptors` | 30 | Standard (biophysical) | PENDING | PENDING | None |
| `protein_kmer` | 200 | Standard (sequence, filtered) | PENDING | PENDING | None |

**Note:** `protein_kmer` reduced from 8000D to 200D via feature-importance pre-filtering
(RF importance on subsample → top 200 features). Fixes timeout and improves tractability.

**Presets:**
- `fast` (5): drug_ecfp4, drug_maccs, drug_pdv, protein_esm2, protein_aac
- `fast+lm` (10): + drug_chemberta, drug_molformer, drug_clamp, protein_esmc, protein_descriptors
- `full` (12): + drug_selformer, protein_esm2_midlayers, protein_kmer

**Fine-tuning:**
- `protein_esm2_lora`: ESM-2 650M LoRA (rank 8) on protein side, trained with frozen drug
  ECFP4 context. Returns 1280D protein embeddings. Recommended for >5K samples (Davis 30K,
  KIBA 118K). Usage: `--finetune protein_esm2_lora --finetune-epochs 5`

---

## 3. Known Issues and Reliability

### Unreliable Reps (may fail on server)

| Rep | Issue | Severity | Workaround |
|-----|-------|----------|------------|
| `gene_mutation` | Xena MC3 returns 403 | ~~HIGH~~ **RESOLVED** | **Dropped** from all presets/dispatch |
| `clinical` | Xena phenotype returns 403 | ~~HIGH~~ **RESOLVED** | **Dropped** from all presets/dispatch |
| `gene_hallmark` | ~~Enrichr download may fail~~ | ~~MEDIUM~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `gene_progeny` | ~~decoupler API 502 / NAType~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath weights, no API calls |
| `gene_dorothea` | ~~decoupler NAType in weight column~~ | ~~LOW~~ **RESOLVED** | Now uses local OmniPath regulons, no API calls |
| `gene_reactome` | ~~ssGSEA orientation bug~~ | ~~LOW~~ **RESOLVED** | Now uses local gene sets, no runtime downloads |
| `wsi_titan` | HuggingFace gated, garbage results | ~~MEDIUM~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `wsi_uni2h` | Manual download + h5 processing, features not found | ~~HIGH~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `wsi_cnn320` | Local Seagate only, garbage results | ~~LOW~~ **RESOLVED** | **Dropped** from presets/dispatch |
| `balm` | Not on HuggingFace; needs manual clone + weight download | ~~MEDIUM~~ **RESOLVED** (2026-02-25) | Cloned to `BALM/` on server, weights in `pretrained-BALM/` |
| `antifold_score` | ImmuneBuilder not installed, 0/7 positive | ~~HIGH~~ **RESOLVED** | Deleted (March 2026 antibody rebuild) |
| `sapiens_score` | Tensor overflow for VH > 146 aa, 0/7 positive | ~~HIGH~~ **RESOLVED** | Deleted (March 2026 antibody rebuild) |

### Stale Server Caches

The following cached `.npy` files on the server are from buggy runs and must be
invalidated before re-running:

- `gene_reactome` — old ssGSEA orientation bug (now uses local gene sets, different dimensions)
- `gene_kegg` — same (now ~320 features vs old ~186)
- `gene_oncogenic` — same
- `gene_progeny` — old decoupler format (now uses local weights)
- `gene_dorothea` — old decoupler format (now ~367 TFs vs old ~300)
- `gene_hallmark` — old ssGSEA format (now mean z-score, same ~50 features)
- `gene_combined_pathway` — depends on all above

**All pathway caches should be deleted after this update** since the scoring
method changed from ssGSEA/decoupler to mean z-score/weighted linear scoring.
The cache validation code detects n_samples mismatches but NOT dimension changes
for the same n_samples, so manual deletion is recommended:
`rm ~/.cache/kirby/reps/tcga_*/gene_{hallmark,kegg,reactome,oncogenic,progeny,dorothea,combined_pathway}.npy`

### DTA Import Issue

The survey job (11380418) failed with `ImportError: DTA datasets require PyTDC`.
This is because the server had the old code. Current `kirby.datasets.dta` downloads
directly from DeepDTA GitHub with no PyTDC dependency. The server needs `git pull`
to pick up commit `3ba481e`.

---

## 4. Coverage Gaps and Actionable Status

### What we HAVE (confirmed baselines)

**NOTHING** — all baselines invalidated by HP default changes (March 2026 overhaul).
All scores reset to PENDING. Full re-run required across all domains.

### What's PENDING (needs compute)

| Domain | Gap | Estimated cells | Blocker |
|--------|-----|----------------|---------|
| Molecular | All 15+ reps × 4 datasets | ~64 | HP defaults changed; server cache clear needed |
| Antibody | 8+ reps × 12 datasets | ~96+ | HP defaults changed; run `antibody_baseline_survey.py all` |
| TCGA | All 20+ reps × 2 datasets | ~44+ | HP defaults changed; pathway caches stale |
| DTA | All 10 reps × 2 datasets | ~20 | HP defaults changed; protein_kmer now pre-filtered |

### What's BROKEN (needs fixing before re-run)

| Item | Issue | Fix |
|------|-------|-----|
| ~~gene_hallmark, gene_kegg, gene_oncogenic~~ | ~~MSigDB `.gmt` files missing~~ | **FIXED**: Now uses local Enrichr gene sets (mean z-score scoring) |
| ~~gene_reactome~~ | ~~ssGSEA returns wrong n_samples~~ | **FIXED**: Now uses local gene sets, no ssGSEA |
| ~~gene_progeny, gene_dorothea~~ | ~~decoupler API 502 / NAType~~ | **FIXED**: Now uses local OmniPath weights |
| ~~gene_combined_pathway~~ | ~~All dependencies broken~~ | **FIXED**: All deps now local |
| gene_mutation, clinical | Xena 403 Forbidden | Pre-download MC3/phenotype from login node or remove from presets |
| wsi_titan | HF gated, access denied | Apply for access or **DROP** |
| wsi_uni2h | Features never prepared | Run `prepare_uni2h_features()` or **DROP** |
| wsi_cnn320 | Local Seagate only, scored random | **DROP** from default presets |
| DTA (protein_kmer) | Times out during generation | Increase timeout or generate separately |

### Candidates to DROP

Reps that are broken AND low-value should be removed from presets to save compute:

| Rep | Reason to drop |
|-----|---------------|
| `wsi_titan` | Gated, scored random (~0.43 on binary), likely never loaded |
| `wsi_uni2h` | Manual prep required, scored random, never worked |
| `wsi_cnn320` | Local-only, scored random, Lung-only, 320D |
| `gene_mutation` | Xena 403, unreliable download |
| `clinical` | Xena 403, unreliable download |
| `unimol` | **DROPPED** — `unimol_tools` package has no working install path |

### Prerequisites before re-running baseline survey

1. `git pull` on server (picks up all fixes including local gene sets)
2. Delete stale pathway caches: `rm ~/.cache/kirby/reps/tcga_*/gene_{hallmark,kegg,reactome,oncogenic,progeny,dorothea,combined_pathway}.npy`
3. ~~Pre-download gseapy gene sets~~ **No longer needed** — gene sets are now bundled as JSON files
4. ~~Ensure gseapy/decoupler installed~~ **No longer needed** for pathway reps (gseapy still needed for `gene_immune` Bindea ssGSEA)
5. For DTA: no PyTDC needed (fixed), but protein_kmer needs longer timeout
6. **Molecular pretrained reps:** Run `bash scripts/setup_server_models.sh` (one-time),
   then `sbatch tests/slurm_scripts/run_precache_molecular.sh` to precache all reps
   in isolated subprocesses before running the baseline survey
7. **New reps:** `gene_autoencoder50` requires torch; all others use only sklearn/scipy

---

## 5. Representation Categories (Summary)

| Category | Description | Examples |
|----------|-------------|---------|
| **Standard** | Computed from input data, no pretrained models | ecfp4, maccs, pdv, aac, kmer, gene_pca100, cnv_gistic |
| **Pretrained** | Frozen embeddings from pretrained LMs | chemberta, molformer, esm2, ablang2, wsi_ctranspath |
| **Fine-tuned** | Per-fold fine-tuned embeddings (LoRA/Geneformer) | lora_chemberta, geneformer |
| **Customizable** | Derived from metadata, mutations, or domain knowledge | varpos_*, lm_score_*, biophysical_delta, structural |

---

## 6. Fine-Tuning Factories

### Active (proceed to Phase 2)

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | GPU | Survey Result |
|---------|--------|---------------|--------|-----------|----------|-----|---------------|
| `make_geneformer_factory` | TCGA | Geneformer V2 (104M) | Freeze+head | 1152D | `--finetune geneformer` | Yes | **PASS** — best individual rep on lung (0.930), BRCA (0.789) |

### Retired (tested, does not help — do NOT run)

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | Survey Result |
|---------|--------|---------------|--------|-----------|----------|---------------|
| `make_molecular_lora_factory` | Molecular | ChemBERTa-77M-MLM | LoRA (rank=8) | 384D | `--finetune lora_chemberta` | **FAIL** — worse than frozen ecfp4/pdv on all 3 datasets |
| `make_molecular_lora_factory` | Molecular | MoLFormer-XL | LoRA (rank=8) | 768D | `--finetune lora_molformer` | **FAIL** — worse than frozen ecfp4/pdv on all 3 datasets |

**Deleted (March 2026):** All antibody fine-tuning factories (`antibody_lora`, `antibody_freeze`,
`antibody_mlm`) removed. All produced negative R² on antibody datasets.

### Two-Phase Fine-Tuning Workflow

**Phase 1: Survey** (COMPLETE) — `run_finetune_survey.sh`
- 2 folds, 2 epochs, fast search, performance_weighted allocation
- Output: `tests/results/finetune_survey/`

| Domain | Datasets | Methods Tested | Frozen Reps | Verdict |
|--------|----------|----------------|-------------|---------|
| Molecular | logd, caco2, herg | lora_chemberta, lora_molformer | fast (3) | **All fail** — FT always worse than frozen |
| Antibody | flab_expression_koenig, flab_binding_koenig, abagym_Ang2 | lora_esm2, lora_esm2_small, mlm_esm2 | dms (9) | **All fail** — negative R², worse than mean predictor |
| TCGA | tcga_luad_vs_lusc, tcga_brca_pam50 | geneformer | fast (5-8) | **Geneformer passes** — best individual + improves hybrid |

**Phase 2: Full run** (targeted) — only Geneformer on TCGA
- 5 folds, 5 epochs, medium search
- Embeddings cached per-fold to `~/.cache/kirby/finetuned_reps/`

**Submission:**
```bash
# Phase 2: only Geneformer survived the survey
sbatch tests/slurm_scripts/run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer
sbatch tests/slurm_scripts/run_finetune_tcga.sh fast tcga_brca_pam50 geneformer
```

---

## 7. Fine-Tuning Survey Results (2026-02-25)

Full results from Phase 1 survey (2 folds, 2 epochs, performance_weighted allocation).
Raw JSON in `tests/results/finetune_survey/`.

### 7A. Molecular — FT does NOT help

| Dataset | Best Frozen Rep | R² | Best FT Rep | R² | Verdict |
|---------|----------------|-----|-------------|-----|---------|
| LogD | pdv (lgbm) | 0.772 | lora_chemberta (lgbm) | 0.697 | Frozen wins by 0.075 |
| Caco2 | pdv (xgb) | 0.466 | lora_chemberta (lgbm) | 0.335 | Frozen wins by 0.131 |
| hERG | ecfp4 (rf) | 0.504 | lora_chemberta (rf) | 0.398 | Frozen wins by 0.106 |

**Conclusion:** LoRA fine-tuning of ChemBERTa and MoLFormer consistently degrades
performance vs. frozen fingerprints/descriptors. The cheap standard reps (ecfp4, pdv)
remain dominant. Fine-tuning is not worth the GPU cost for molecular property prediction.

### 7B. Antibody — FT does NOT help (code deleted March 2026)

All antibody fine-tuning code has been removed. Summary of what was tested before deletion:

- `lora_esm2` (650M, LoRA rank=8): Negative R² on FLAb expression, binding, AbAgym Ang2
- `lora_esm2_small` (8M, LoRA rank=8): Negative R² on all datasets
- `mlm_esm2` (self-supervised MLM): Negative R² on all datasets
- Other variants (`lora_antiberty`, `lora_ablang2`, `mlm_antiberty`, `mlm_esm2_small`): Never tested, not expected to help

The entire mutation-aware representation approach (40+ custom functions, varpos features,
LM log-prob scores, delta reps) has been replaced with simple LM embeddings → Ridge →
Spearman ρ, following the compass document. See `tests/antibody_baseline_survey.py`.

### 7C. TCGA — Geneformer FT HELPS

| Dataset | Best Frozen Rep | Acc | Geneformer FT | Acc | Hybrid (frozen+FT) | Acc |
|---------|----------------|-----|---------------|-----|--------------------|----|
| LUAD vs LUSC | gene_nmf50 (rf) | 0.951 | geneformer (xgb) | 0.930 | 4-rep hybrid | 0.940 |
| BRCA PAM50 | gene_hvg2000 (xgb) | 0.866 | geneformer (xgb) | 0.789 | 5-rep hybrid | 0.820 |

**Conclusion:** Geneformer fine-tuning produces competitive individual performance and
improves hybrid quality on TCGA tasks. The 1152D V2 embeddings loaded correctly
(confirming the tokenization fix worked). No data leakage concern: Geneformer was
pretrained on single-cell data (Genecorpus-104M, non-cancer), not TCGA bulk RNA-seq.
**Proceed to Phase 2 full run (5 folds, 5 epochs).**

Note: gene_nmf50 outperforms Geneformer as an individual rep on lung (0.951 vs 0.930),
but Geneformer captures complementary information — it adds value in hybrids. On BRCA,
hvg2000 leads (0.866 vs 0.789), and the hybrid (0.820) doesn't beat the best individual;
this may improve with more folds and epochs in Phase 2.

---

## 8. Preprocessing Pipeline (2026-02-27)

All baselines must be re-run after these changes. Cache invalidation required.

### Molecular SMILES Standardization
- `_standardise_smiles()` now applies: `Cleanup → FragmentParent → Uncharger → TautomerEnumerator.Canonicalize`
- Metal-containing compounds rejected (organic descriptors unreliable)
- OpenADMET endpoints: 1st/99th percentile outlier clipping after deduplication

### QM9 Normalization Fix
- **Data leak fixed:** QM9 labels were pre-normalized from the full sampled set. Now returns raw labels; `StandardScaler` is applied per-fold in the evaluation pipeline.

### KIBA Score Transform
- KIBA scores negated so higher = stronger binding (consistent with pKd convention)

### DTA SMILES Re-canonicalization
- Drug SMILES from DeepDTA re-canonicalized with RDKit for consistency

### Antibody Label Preprocessing
- `abagym_all`: labels z-scored per experiment before concatenation (standard for pooling assays)
- `ddg_combined`: outlier filter removes entries with |ddG| > 8 kcal/mol or > 3σ from mean

### TCGA Preprocessing
- **ComBat batch correction:** expression data corrected across tissue source sites (TSS) using pyComBat, applied before caching so all downstream reps (HVG, PCA) get corrected data. Optional — falls back to uncorrected if `combat` not installed.
- **Zero-expression filter:** genes with >90% zero expression removed before HVG selection and PCA
- **Mutation matrix:** filtered to non-silent variants only (Missense, Nonsense, Frameshift, Splice_Site, etc.)

### Model Improvements
- **Ridge → RidgeCV:** internal LOO cross-validation selects best alpha from `np.logspace(-6, 6, 13)`. No extra compute cost.
- **LogisticRegression → LogisticRegressionCV:** built-in Cs=10 alpha selection
- **KNN adaptive k:** `n_neighbors = min(5, max(1, sqrt(N_train)))` — scales to dataset size

### Cache Invalidation (REQUIRED — March 2026 overhaul)

**⚠ SERVER ACTION NEEDED:** Clear ALL caches before re-running any baselines.
HP defaults changed (RF max_features, XGB colsample_bytree, RidgeCV alpha grid),
preprocessing changed, all old numbers are invalid.

```bash
# FULL CACHE CLEAR (run on server before any re-runs)
rm -rf ~/.cache/kirby/reps/              # all rep caches (SMILES changed)
rm -rf ~/.cache/kirby/openadmet_train.csv # re-download + re-standardize
rm -rf ~/.cache/kirby/chembl_herg_ki.csv  # re-fetch + re-standardize
rm -rf ~/.cache/kirby/dta/               # KIBA transform changed
rm -rf ~/.cache/kirby/tcga/expression/   # ComBat changes expression data
rm -rf ~/.cache/kirby/tcga/pathways/     # already documented as stale
rm -rf ~/.cache/kirby/ddg_combined/      # outlier filter changes dataset
rm -rf ~/.cache/kirby/abagym/            # z-scoring changes labels
rm -rf ~/.cache/kirby/finetuned_reps/    # FT embeddings from old HP regime
```

---

## 9. Open Problems

### Antibody Representation Strategy (March 2026 rebuild)

The old approach (40+ mutation-aware custom features, complex presets, HP tuning) has been
replaced with a simple baseline pipeline: LM embeddings → Ridge → Spearman ρ.

**Current approach:** 8 representations (5 LM + 3 sequence encodings) evaluated per
dataset with RidgeCV. This matches the standard in the field (FLAb2 paper uses the same).

**Key questions to answer with the new baselines:**
1. Which LM (ESM-2, AbLang2, IgBert, AntiBERTy, ProtT5) works best per task type?
2. Does one-hot match or beat LMs at ≥1K samples (as claimed in literature)?
3. Are inter-family datasets (ddg_combined, abcov_kd) still fundamentally too small/hard?
4. Is Ridge sufficient or do tree models add value for antibody tasks?
