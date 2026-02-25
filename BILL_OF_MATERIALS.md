# KIRBy Bill of Materials

Comprehensive inventory of datasets, representations, and their measured baseline
performance. This document is the single source of truth for what KIRBy supports
and what actually works. Representations without confirmed performance data should
not be used in experiments until validated.

**Last updated:** 2026-02-24

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
| `flab_expression_koenig` | FLAb (G6 SSM) | ~4,275 | Expression (ER) | **Confirmed** |
| `flab_binding_koenig` | FLAb (G6) | varies | Kd | Listed, no baselines yet |
| `flab_thermo_hie` | FLAb (S309) | varies | Tm | Listed, no baselines yet |
| `flab_binding_rosace` | FLAb (adalimumab) | varies | Kd | Listed, no baselines yet |
| `abagym_Ang2_2017_G6` | AbAgym DMS | varies | Fitness | Listed, no baselines yet |
| `abagym_EGFR_2013_Cetuximab` | AbAgym DMS | varies | Fitness | Listed, no baselines yet |
| `abagym_lysozyme_2019_D441` | AbAgym DMS | varies | Fitness | Listed, no baselines yet |
| `abagym_VEGF_2017b_G6` | AbAgym DMS | varies | Fitness | Listed, no baselines yet |
| `abagym_HER2_2021_trastuzumab` | AbAgym DMS | varies | Fitness | Listed, no baselines yet |

### 1C. Antibody Inter-Family (regression, identity-grouped GroupKFold)

| Dataset | Source | Samples | Target | Status |
|---------|--------|---------|--------|--------|
| `sabdab_affinity` | SKEMPI 2.0 | ~4,000 | ΔΔG | Listed, no baselines yet |
| `abcov_kd` | Ab-CoV (via FLAb) | varies | -log Kd | Listed, no baselines yet |
| `flab_thermo_jain` | FLAb | varies | Tm | Listed, no baselines yet |
| `flab_aggregation_jain` | FLAb | varies | AC-SINS | Listed, no baselines yet |
| `abagym_all` | AbAgym (pooled) | ~2,247 | Fitness | Listed, no baselines yet |
| `abbind` | AbBind | ~230 | ΔΔG | Listed, no baselines yet |

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

**Source:** C01 baselines on LogD (~5,039 samples), hERG (~1,415), Caco2 (~2,161).
Models: rf, xgboost, ridge, knn, lightgbm, mlp, svm.

| Rep | Dim | Category | LogD R² (best) | Caco2 R² | hERG R² | Deps |
|-----|-----|----------|---------------|----------|---------|------|
| `ecfp4` | 2048 | Standard (fingerprint) | lgbm=0.756 | rf=0.482 | rf=0.547 | rdkit |
| `maccs` | 167 | Standard (fingerprint) | xgb=0.674 | rf=0.372 | rf=0.487 | rdkit |
| `pdv` | 200 | Standard (descriptor) | lgbm=0.778 | xgb=0.488 | lgbm=0.543 | rdkit |
| `chemberta` | 384 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `molformer` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `selformer` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `mol2vec` | 300 | Pretrained (Word2Vec) | **PENDING** | — | — | mol2vec |
| `chembert` | 768 | Pretrained (Transformer) | **PENDING** | — | — | transformers |
| `molclr` | 512 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric |
| `graphmvp` | 300 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric |
| `smited` | 256 | Pretrained | **PENDING** | — | — | HuggingFace |
| `mhg_gnn` | 64 | Pretrained (GNN) | **PENDING** | — | — | torch_geometric |
| `grover` | 5000 | Pretrained (Transformer) | **PENDING** | — | — | grover |
| `graph_kernel` | varies | Standard (kernel) | **PENDING** | — | — | grakel |
| `schnet` | 128 | Pretrained (3D GNN) | **PENDING** | — | — | schnetpack |
| `unimol` | 512 | Pretrained (3D) | **PENDING** | — | — | unimol_tools |
| `chemformer` | 256 | Pretrained (Transformer) | **PENDING** | — | — | transformers |

**Augmentations (not base reps):** `graph_topology`, `spectral`

**Fine-tuning:** `lora_chemberta`, `lora_molformer` (LoRA per-fold)

**Presets:**
- `fast` (3): ecfp4, maccs, pdv
- `fast+hf` (6): + chemberta, molformer, selformer
- `all` (15): all reps including GNN and graph_kernel

### 2B. Antibody Representations

**Source:** C01 baselines on FLAb Expression Koenig (~4,275 samples, intra-family DMS).
Models: rf, knn, ridge, mlp, xgboost, lightgbm.

#### Inter-Family Base Reps

| Rep | Dim | Category | FLAb Expr R² (best) | Deps | Notes |
|-----|-----|----------|---------------------|------|-------|
| `kmer` | 400 | Standard (sequence) | lgbm=-0.138 | None | Negative R² = worse than mean |
| `aac` | 20 | Standard (composition) | lgbm=0.217 | None | Low but positive |
| `blosum62` | 20 | Standard (substitution) | mlp=0.184 | None | Low but positive |
| `physicochemical` | 68 | Standard (biophysical) | ridge=0.010 | None | Near-zero on DMS |
| `antiberty` | 512 | Pretrained (Ab LM) | **PENDING** | transformers | |
| `ablang2` | 768 | Pretrained (Ab LM) | **PENDING** | ablang2 | |
| `igbert` | 512 | Pretrained (Ab LM) | **PENDING** | transformers | |
| `esm2` | 1280 | Pretrained (protein LM) | **PENDING** | transformers | |
| `prot_t5` | 1024 | Pretrained (protein LM) | **PENDING** | transformers | |
| `ab_roberta` | 768 | Pretrained (Ab LM) | **PENDING** | transformers | |
| `balm` | 768 | Pretrained (Ab LM) | **PENDING** | transformers | |
| `pigen` | 512 | Pretrained (Ab LM) | **PENDING** | transformers | |
| `esmc` | varies | Pretrained (protein LM) | **PENDING** | esmc | |
| `sapiens` | varies | Pretrained (humanization) | **PENDING** | sapiens | |
| `structural` | 28 | Customizable (3D) | **PENDING** | ImmuneBuilder | Server only |
| `developability` | varies | Customizable | **PENDING** | None | |
| `imgt_onehot` | varies | Standard (positional) | **PENDING** | anarci | |

**Note:** Full-sequence LM embeddings are known to be near-useless for intra-family
DMS (published ρ ≈ 0.08). The `dms` presets use mutation-specific reps instead.

#### DMS / Mutation-Aware Reps

| Rep | Dim | Category | FLAb Expr R² (best) | Notes |
|-----|-----|----------|---------------------|-------|
| `lm_score_ablang2` | 5 | Customizable (LM score) | **PENDING** | Per-model pseudo-likelihood |
| `lm_score_esm2` | 5 | Customizable (LM score) | **PENDING** | |
| `lm_score_prot_t5` | 5 | Customizable (LM score) | **PENDING** | |
| `lm_score_antiberty` | 5 | Customizable (LM score) | **PENDING** | |
| `positional` | 11 | Customizable (mutation) | **PENDING** | CDR region, BLOSUM, position |
| `biophysical_delta` | 6 | Customizable (mutation) | **PENDING** | ΔCharge, ΔHydro, etc. |
| `esm2_site` | 1280 | Customizable (mutation) | **PENDING** | Per-residue at mutation site |
| `varpos_physchem` | 6/pos | Customizable (varpos) | **PENDING** | Physicochemical at variable positions |
| `varpos_onehot` | 20/pos | Customizable (varpos) | **PENDING** | One-hot at variable positions |
| `varpos_blosum` | 20/pos | Customizable (varpos) | **PENDING** | BLOSUM62 at variable positions |
| `varpos_cdr_physchem` | reduced | Customizable (varpos) | **PENDING** | CDR-filtered |
| `varpos_cdrh3_physchem` | minimal | Customizable (varpos) | **PENDING** | CDR-H3 only |
| `varpos_fw_physchem` | varies | Customizable (varpos) | **PENDING** | Framework-only |

**Delta reps:** `--delta` flag generates ΔX = X_mut − X_wt for every base rep, doubling count.

**Fine-tuning:** `antibody_lora` (AbLang2 LoRA), `antibody_mlm` (masked LM)

**Presets:**
- `fast` (4): kmer, physicochemical, aac, blosum62
- `fast+lm` (7): + antiberty, ablang2, igbert
- `all_server` (13): + esm2, prot_t5, ab_roberta, balm, pigen, structural
- `mutation_fast` (7): kmer, aac, blosum62, physicochemical, esm2, prot_t5, varpos_physchem
- `mutation_full` (10): + antiberty, ablang2, varpos_onehot
- `dms` (9): lm_score_ablang2, lm_score_esm2, positional, biophysical_delta, aac, blosum62, varpos_physchem, varpos_cdr_physchem, varpos_cdrh3_physchem
- `dms_full` (17): all mutation-aware + full varpos arsenal

### 2C. TCGA Representations

**Source:** Baselines from SLURM jobs 11379620/11379734-11379737 on LUAD vs LUSC
and BRCA PAM50. Models: rf, xgboost.

#### Expression-Derived

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `gene_hvg2000` | 2000 | Standard | **PENDING** | xgb=0.866 | scanpy (opt) | YES |
| `gene_pca100` | 100 | Standard | rf=0.928 | xgb=0.826 | sklearn | YES |
| `gene_nmf50` | 50 | Standard | rf=0.951 | rf=0.834 | sklearn | YES |
| `gene_ica50` | 50 | Standard | **PENDING** | **PENDING** | sklearn | YES |
| `gene_stats` | 15 | Standard | rf=0.710 | rf=0.502 | scipy | YES |
| `gene_hallmark` | 50 | Pathway (ssGSEA) | **PENDING** | **PENDING** | gseapy | CONDITIONAL |
| `gene_kegg` | ~318 | Pathway (ssGSEA) | **PENDING** | **PENDING** | gseapy | CONDITIONAL |
| `gene_reactome` | ~1000 | Pathway (ssGSEA) | **PENDING** | **PENDING** | gseapy | CONDITIONAL |
| `gene_oncogenic` | ~189 | Pathway (ssGSEA) | **PENDING** | **PENDING** | gseapy | CONDITIONAL |
| `gene_progeny` | 14 | Pathway (decoupler) | **PENDING** | **PENDING** | decoupler | CONDITIONAL |
| `gene_dorothea` | ~300 | Pathway (decoupler) | **PENDING** | **PENDING** | decoupler | CONDITIONAL |
| `gene_combined_pathway` | ~364 | Pathway (combined) | **PENDING** | **PENDING** | gseapy+decoupler | CONDITIONAL |
| `gene_immune` | ~27 | Computed (immune) | rf=0.793 | rf=0.524 | gseapy (hardcoded) | YES |

#### Multi-Omic

| Rep | Dim | Category | Lung Acc (best) | BRCA Acc (best) | Deps | Server? |
|-----|-----|----------|----------------|----------------|------|---------|
| `gene_mutation` | 200 | Somatic | **PENDING** | **PENDING** | Xena MC3 | UNRELIABLE (403) |
| `cnv_gistic` | 200 | CNV | xgb=0.820 | rf=0.596 | Xena | CONDITIONAL |
| `cnv_pca50` | 50 | CNV | **PENDING** | **PENDING** | Xena | CONDITIONAL |
| `rppa` | ~260 | Protein | rf=0.749 | xgb=0.773 | Xena | CONDITIONAL |
| `mirna_top200` | 200 | miRNA | rf=0.888 | rf=0.663 | Xena | CONDITIONAL |
| `clinical` | 10 | Phenotype | **PENDING** | **PENDING** | Xena | UNRELIABLE (403) |

#### WSI (Pretrained Vision)

| Rep | Dim | Source | Lung Acc (best) | BRCA Acc (best) | Access | Server? |
|-----|-----|--------|----------------|----------------|--------|---------|
| `wsi_ctranspath` | 512 | GigaSSL (Mendeley) | xgb=0.783 | rf=0.513 | Public | YES |
| `wsi_phikon` | 512 | GigaSSL (Mendeley) | xgb=0.776 | rf=0.513 | Public | YES |
| `wsi_hoptimus0` | 512 | GigaSSL (Mendeley) | **PENDING** | rf=0.641 | Public | YES |
| `wsi_gigapath` | 512 | GigaSSL (Mendeley) | **PENDING** | **PENDING** | Public | YES |
| `wsi_moco` | 512 | GigaSSL (Mendeley) | **PENDING** | rf=0.611 | Public | YES |
| `wsi_titan` | 768 | HuggingFace (gated) | rf=0.513 | rf=0.513 | Gated | LIMITED |
| `wsi_uni2h` | 1536 | HuggingFace (gated) | **PENDING** | **PENDING** | Manual | LIMITED |
| `wsi_cnn320` | 320 | Local (Seagate) | **PENDING** | N/A | Local | LOCAL ONLY |

**Fine-tuning:** `geneformer` (per-fold, rank-encoded expression)

**Presets:**
- `fast` (5): pca100, hallmark, progeny, ctranspath, phikon
- `fast+diverse` (8): + nmf50, stats, immune
- `multimodal` (12): + mutation, cnv_gistic, rppa, mirna_top200, clinical
- `fast+learned` (15): expanded genomic + 5 WSI
- `all` (27): everything
- `genomic_fast` (3): pca100, hallmark, progeny
- `genomic_diverse` (10): multi-omic without WSI
- `genomic_all` (18): all genomic + multi-omic

### 2D. DTA Representations

**No baselines yet.** Survey failed on old code (PyTDC import). Server needs updated code.

| Rep | Dim | Category | Davis R² | KIBA R² | Deps |
|-----|-----|----------|----------|---------|------|
| `drug_ecfp4` | 2048 | Standard (fingerprint) | **PENDING** | **PENDING** | rdkit |
| `drug_maccs` | 167 | Standard (fingerprint) | **PENDING** | **PENDING** | rdkit |
| `drug_pdv` | 200 | Standard (descriptor) | **PENDING** | **PENDING** | rdkit |
| `drug_chemberta` | 384 | Pretrained (drug LM) | **PENDING** | **PENDING** | transformers |
| `drug_molformer` | 768 | Pretrained (drug LM) | **PENDING** | **PENDING** | transformers |
| `drug_selformer` | 768 | Pretrained (drug LM) | **PENDING** | **PENDING** | transformers |
| `protein_esm2` | 1280 | Pretrained (protein LM) | **PENDING** | **PENDING** | transformers |
| `protein_aac` | 20 | Standard (composition) | **PENDING** | **PENDING** | None |
| `protein_descriptors` | 30 | Standard (biophysical) | **PENDING** | **PENDING** | None |
| `protein_kmer` | varies | Standard (sequence) | **PENDING** | **PENDING** | None |

---

## 3. Known Issues and Reliability

### Unreliable Reps (may fail on server)

| Rep | Issue | Severity | Workaround |
|-----|-------|----------|------------|
| `gene_mutation` | Xena MC3 returns 403 | HIGH | Needs manual pre-download or removal from presets |
| `clinical` | Xena phenotype returns 403 | HIGH | Needs manual pre-download or removal from presets |
| `gene_hallmark` | Enrichr download may fail on compute nodes | MEDIUM | Works if gseapy cached from login node first |
| `gene_progeny` | decoupler column name changes across versions | LOW | Fixed: `_normalize_decoupler_net()` handles v1+v2 |
| `gene_dorothea` | decoupler NAType in weight column | LOW | Fixed: `pd.to_numeric(errors='coerce')` |
| `gene_reactome` | ssGSEA orientation when n_pathways > n_samples | LOW | Fixed: `_extract_ssgsea_matrix()` sample-name matching |
| `wsi_titan` | HuggingFace gated, awaiting access | MEDIUM | Skip unless access granted |
| `wsi_uni2h` | Manual download + h5 processing required | HIGH | `prepare_uni2h_features()` must be run first |
| `wsi_cnn320` | Local Seagate only, LUAD/LUSC only | LOW | Silently skipped if absent |

### Stale Server Caches

The following cached `.npy` files on the server are from buggy runs and must be
invalidated before re-running:

- `gene_reactome` — ssGSEA orientation bug produced (n_pathways, n_samples) instead of (n_samples, n_pathways)
- `gene_kegg` — same bug, only visible when n_pathways > n_samples
- `gene_oncogenic` — same bug
- `gene_progeny` — decoupler column mismatch
- `gene_dorothea` — decoupler NAType crash

The cache validation code in `find_best_hybrid_tcga.py` now detects stale caches
automatically (n_samples mismatch → regenerate).

### DTA Import Issue

The survey job (11380418) failed with `ImportError: DTA datasets require PyTDC`.
This is because the server had the old code. Current `kirby.datasets.dta` downloads
directly from DeepDTA GitHub with no PyTDC dependency. The server needs `git pull`
to pick up commit `3ba481e`.

---

## 4. Coverage Gaps (PENDING Items)

### What the baseline survey will fill

The `run_baseline_survey.sh` script is designed to fill all PENDING cells above in
a single run. It tests:

- **Molecular:** 4 datasets × `all` preset (15 reps) = ~60 baseline cells
- **Antibody intra:** 9 datasets × `dms_full` preset (17 reps) = ~153 baseline cells
- **Antibody inter:** 6 datasets × `all_server` preset (13 reps) = ~78 baseline cells
- **TCGA:** 3 datasets × `all` preset (27 reps) = ~81 baseline cells
- **DTA:** 2 datasets × `full` preset (10 reps) = ~20 baseline cells

**Total: ~392 baseline cells to fill.**

### Prerequisites before re-running

1. `git pull` on server (picks up ssGSEA fix, decoupler fix, DTA loader fix)
2. Delete stale caches: `rm ~/.cache/kirby/reps/tcga_*/gene_reactome.npy` etc.
   (or let the auto-validation handle it)
3. Ensure `gseapy>=1.0`, `decoupler>=1.4`, `anndata` are installed
4. For TCGA: pre-download Xena data from login node (compute nodes may lack internet)
5. For DTA: no special setup needed (downloads from GitHub)

---

## 5. Representation Categories (Summary)

| Category | Description | Examples |
|----------|-------------|---------|
| **Standard** | Computed from input data, no pretrained models | ecfp4, maccs, pdv, aac, kmer, gene_pca100, cnv_gistic |
| **Pretrained** | Frozen embeddings from pretrained LMs | chemberta, molformer, esm2, ablang2, wsi_ctranspath |
| **Fine-tuned** | Per-fold fine-tuned embeddings (LoRA/MLM/Geneformer) | lora_chemberta, antibody_lora, geneformer |
| **Customizable** | Derived from metadata, mutations, or domain knowledge | varpos_*, lm_score_*, biophysical_delta, structural |

---

## 6. Fine-Tuning Factories

| Factory | Domain | Base Model(s) | Method | Output Dim | CLI Flag | GPU |
|---------|--------|---------------|--------|-----------|----------|-----|
| `make_molecular_lora_factory` | Molecular | ChemBERTa-77M-MLM | LoRA (rank=8) | 384D | `--finetune lora_chemberta` | Yes |
| `make_molecular_lora_factory` | Molecular | MoLFormer-XL | LoRA (rank=8) | 768D | `--finetune lora_molformer` | Yes |
| `make_antibody_lora_factory` | Antibody | ESM-2 (650M) | LoRA (rank=8) | 1280D | `--finetune lora_esm2` | Yes |
| `make_antibody_lora_factory` | Antibody | ESM-2 (8M) | LoRA (rank=8) | 320D | `--finetune lora_esm2_small` | Yes |
| `make_antibody_lora_factory` | Antibody | AntiBERTy | LoRA (rank=8) | 512D | `--finetune lora_antiberty` | Yes |
| `make_antibody_lora_factory` | Antibody | AbLang2 | LoRA (rank=8) | 480D | `--finetune lora_ablang2` | Yes |
| `make_antibody_mlm_factory` | Antibody | ESM-2 (650M) | Self-supervised MLM | 5D | `--finetune mlm_esm2` | Yes |
| `make_antibody_mlm_factory` | Antibody | ESM-2 (8M) | Self-supervised MLM | 5D | `--finetune mlm_esm2_small` | Yes |
| `make_antibody_mlm_factory` | Antibody | AntiBERTy | Self-supervised MLM | 5D | `--finetune mlm_antiberty` | Yes |
| `make_geneformer_factory` | TCGA | Geneformer / BERT fallback | Freeze+head | 768/256D | `--finetune geneformer` | Yes |

### Two-Phase Fine-Tuning Workflow

**Phase 1: Survey** (cheap triage) — `run_finetune_survey.sh`
- 2 folds, 2 epochs, fast search
- Tests ALL fine-tuning methods on representative datasets per domain
- Output: `tests/results/finetune_survey/`
- Goal: determine which methods improve over frozen baselines
- Depends on: baseline survey results (to have meaningful frozen baselines to compare against)

| Domain | Datasets | Methods Tested | Frozen Reps |
|--------|----------|----------------|-------------|
| Molecular | logd, caco2, herg | lora_chemberta, lora_molformer | fast (3) |
| Antibody | flab_expression_koenig, flab_binding_koenig, abagym_Ang2 | lora_esm2, lora_esm2_small, mlm_esm2 | dms (9) |
| TCGA | tcga_luad_vs_lusc, tcga_brca_pam50 | geneformer | fast (5-8) |

**Phase 2: Full run** (targeted) — existing `run_finetune_{molecular,antibody,tcga}.sh`
- 5 folds (from speed tier), 5 epochs (default)
- Only for methods that passed the survey
- Embeddings cached per-fold to `~/.cache/kirby/finetuned_reps/`
- Can be reused as frozen reps in subsequent searches

**Submission:**
```bash
# Phase 1: launch all survey jobs
bash tests/slurm_scripts/submit_finetune_survey.sh all

# Phase 2: after reviewing survey results, launch full runs for winners
sbatch tests/slurm_scripts/run_finetune_molecular.sh fast logd lora_chemberta
sbatch tests/slurm_scripts/run_finetune_antibody.sh dms flab_expression_koenig lora_esm2
sbatch tests/slurm_scripts/run_finetune_tcga.sh fast tcga_luad_vs_lusc geneformer
```
