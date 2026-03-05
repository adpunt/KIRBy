# KIRBy Full Setup Guide

From-zero-to-working setup for all molecular, antibody, and TCGA representations.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Base Installation](#2-base-installation)
3. [Molecular Representations](#3-molecular-representations)
4. [Antibody Representations](#4-antibody-representations)
5. [Pretrained Model Downloads](#5-pretrained-model-downloads)
6. [External Storage (Optional)](#6-external-storage-optional)
7. [ARC Server Setup (Oxford)](#7-arc-server-setup-oxford)
8. [Verification](#8-verification)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10-3.11 |
| RAM | 8 GB | 32 GB |
| Disk | 20 GB | 50 GB (or use external storage) |
| GPU | Not required | CUDA-capable (for pretrained models) |
| OS | Linux / macOS | Linux (for ANARCI via bioconda) |

### Disk space breakdown

| Component | Size |
|-----------|------|
| KIRBy repo + code | ~100 MB |
| Core pip dependencies | ~2 GB |
| Molecular pretrained weights (mol2vec, MHG-GNN, CLAMP, R-MAT) | ~1.1 GB |
| Antibody HF models (ESM-2, IgBERT, ProtT5, etc.) | ~16 GB |
| Protein / ESM-C models | ~3 GB |
| QM9 / test data | ~1.5 GB |
| **Total** | **~24 GB** |

---

## 2. Base Installation

### 2a. Create environment

```bash
# Conda (recommended — needed for ANARCI and RDKit)
conda create -n kirby python=3.11
conda activate kirby

# RDKit (only available via conda)
conda install -c conda-forge rdkit
```

### 2b. Clone and install KIRBy

```bash
git clone https://github.com/adpunt/KIRBy.git
cd KIRBy
pip install -e .
```

### 2c. Install extras based on your use case

```bash
# Molecular QSAR (transformers, GNNs, mol2vec, GPs, XGBoost, LightGBM)
pip install -e ".[qsar]"

# Antibody work (transformers, ablang2, biopython)
pip install -e ".[antibody]"

# TCGA genomics/histopathology (scanpy, gseapy, decoupler, pyComBat)
pip install -e ".[tcga]"

# Uncertainty quantification (NGBoost, quantile forests, BNNs)
pip install -e ".[uncertainty]"

# Everything
pip install -e ".[qsar,antibody,tcga,uncertainty,interpret]"

# Reaction yield prediction (optional)
pip install drfp

# openpyxl — required for Buchwald-Hartwig Excel loading (pandas needs >=3.1.0)
pip install --upgrade openpyxl

# Optuna HP tuning (optional)
pip install optuna

# MAP4 fingerprints (optional)
pip install map4

# Uni-Mol v2 molecular embeddings (optional)
pip install unimol-tools

# scGPT single-cell foundation model (optional, for TCGA)
pip install scgpt

# UCE universal cell embeddings (optional, for TCGA)
pip install uce-model

# AMPLIFY protein LM (optional — uses transformers, auto-downloads from HF)
# Model: chandar-lab/AMPLIFY_350M (~350M params, 960D embeddings)

# ESMFold structure prediction (optional, for protein structural features)
# Uses facebook/esmfold_v1 (~700M params, ~3GB download)
pip install transformers torch

# DimeNet++ 3D molecular embeddings (optional, uses torch-geometric)
# Already covered by torch-geometric install for SchNet
```

### 2d. Install conda-only dependencies

```bash
# ANARCI — required for CDR extraction, IMGT one-hot, per-CDR features
conda install -c bioconda anarci

# torch-scatter — required for some GNN representations
# torch-cluster — required for SchNet end-to-end (QM9 3D comparison)
# Match your PyTorch + CUDA version:
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

---

## 3. Molecular Representations

### 3a. mol2vec pretrained weights (required)

```bash
cd KIRBy
wget https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl
```

This file must be in the repo root. KIRBy looks for `model_300dim.pkl` relative to the working directory.

### 3b. MHG-GNN (optional)

MHG-GNN requires a separate repository and pretrained weights:

```bash
# Clone the model code
cd ~/repos  # or any of: ~/materials, ../materials, ../../materials
git clone https://huggingface.co/ibm-research/materials.mhg-ged materials

# Download pretrained weights from:
# https://huggingface.co/ibm-research/materials.mhg-ged
# File: mhggnn_pretrained_model_0724_2023.pickle
# Place in: materials/

# Install MHG dependencies
cd KIRBy
pip install -e ".[gnn]"
```

### 3c. GROVER (optional)

```bash
# Clone GROVER repo
git clone https://github.com/tencent-ailab/grover.git
cd grover
pip install -e .

# Download pretrained weights:
# https://ai.tencent.com/ailab/ml/ml-data/grover-models/grover_large.tar.gz
# Extract to: grover/pretrained/grover_large/
```

### 3d. CLAMP (optional)

CLAMP = contrastive learning over concatenated fingerprints. The ONLY model that
statistically significantly beat ECFP in the Praski 2025 benchmark.

```bash
# Clone the CLAMP repo (sibling of KIRBy src)
cd KIRBy/src
git clone https://github.com/ml-jku/clamp.git
cd clamp && pip install -e .

# Install mhnreact dependency
pip install mhnreact

# Weights auto-download to src/clamp/data/models/clamp_clip/ on first use (~636MB)
```

### 3e. R-MAT (optional)

R-MAT = Relative Molecule Self-Attention Transformer. #2 in Praski 2025 benchmark.

```bash
pip install git+https://github.com/gmum/huggingmolecules.git#subdirectory=src
# Weights auto-download from Google Drive on first use (~50MB)
```

### 3f. Uni-Mol v2 (optional)

```bash
# Uni-Mol v2 (NeurIPS 2024) — 768D 3D molecular embeddings
pip install unimol-tools
# If install fails: pip install unimol-tools --no-deps
```

### Molecular representation summary

| Representation | Type | Dims | Extra Setup |
|----------------|------|------|-------------|
| ECFP4 | Static | 2048D | None (RDKit) |
| MACCS | Static | 167D | None (RDKit) |
| PDV | Static | 200D | None (RDKit) |
| mol2vec | Pretrained | 300D | Download `model_300dim.pkl` |
| Graph Kernel (WL) | Pretrained | Variable | None (grakel) |
| MHG-GNN | Pretrained | 256D | Clone repo + weights |
| ChemBERTa | Fine-tuned | 768D | None (auto-downloads from HF) |
| MoLFormer | Fine-tuned | 768D | None (auto-downloads from HF) |
| SELFormer | Fine-tuned | 768D | None (auto-downloads from HF) |
| ChemBERT | Fine-tuned | 768D | None (auto-downloads from HF) |
| Chemformer | Fine-tuned | 256D | None (auto-downloads from HF) |
| SchNet | Fine-tuned | 256D | None (torch-geometric) |
| MolCLR | Pretrained | 512D | None (auto-downloads from HF) |
| GraphMVP | Pretrained | 300D | None (auto-downloads from HF) |
| SMITED | Pretrained | 512D | None (auto-downloads from HF) |
| GROVER | Pretrained | ~5000D | Clone repo + weights |
| CLAMP | Pretrained | 768D | Clone repo + `mhnreact` |
| R-MAT | Pretrained | 768D | `huggingmolecules` (gmum) |
| MAP4 | Static | 1024D | `map4` |
| Uni-Mol v2 | Pretrained (3D) | 768D | `unimol-tools` |

---

## 4. Antibody Representations

### 4a. Core antibody dependencies

```bash
pip install -e ".[antibody]"

# ANARCI (required for CDR extraction, IMGT features, per-CDR features)
conda install -c bioconda anarci
```

### 4b. AbLang models

```bash
pip install ablang    # AbLang v1 (768D)
pip install ablang2   # AbLang-2 (768D, paired, seqcoding mode)
```

### 4c. Additional packages

```bash
# ProtT5-XL tokenizer dependency
pip install sentencepiece

# AntiBERTy (RoFormer tokenizer dependency)
pip install rjieba

# ESM-C (EvolutionaryScale package v3 — different from old fair-esm)
# REQUIRES Python >=3.10. Will not install on Python 3.9.
# ESM-2 650M is a comparable alternative if ESM-C is unavailable.
pip install esm

# ANARCI — required for CDR-specific embeddings (esm2_cdr, esmc_cdr, etc.)
conda install -c bioconda anarci

# LoRA fine-tuning (for DTA protein-side fine-tuning)
pip install peft
```

### Antibody representation summary

**Current reps (March 2026 rebuild):** 8 base reps (5 LM + 3 sequence) plus
CDR-specific and midlayer variants. See `BILL_OF_MATERIALS.md` §2B for full list.

| Representation | Dims | Category | Package | Notes |
|----------------|------|----------|---------|-------|
| `esm2` | 2560 | Pretrained LM | transformers | ESM-2 650M, H+L concat |
| `esmc` | 2304 | Pretrained LM | esm (v3) | ESM-C 600M, H+L concat. Non-commercial. |
| `ablang2` | 768 | Pretrained Ab LM | ablang2 | Paired, seqcoding mode |
| `igbert` | 768 | Pretrained Ab LM | transformers | IgBert paired |
| `antiberty` | 1024 | Pretrained LM | transformers | AntiBERTy, H+L concat |
| `prot_t5` | 2048 | Pretrained LM | transformers + sentencepiece | ProtT5-XL, H+L concat |
| `onehot` | 5600 | Sequence | None | Max 150H + 130L, 20 AA |
| `blosum` | 5600 | Sequence | None | BLOSUM62 row per residue |
| `aac` | 40 | Sequence | None | 20 frequencies per chain |
| `esm2_cdr` | 2560 | CDR-pooled LM | transformers + anarci | CDR residues only |
| `esmc_cdr` | 2304 | CDR-pooled LM | esm + anarci | CDR residues only |
| `antiberty_cdr` | 1024 | CDR-pooled LM | transformers + anarci | CDR residues only |
| `prot_t5_cdr` | 2048 | CDR-pooled LM | transformers + anarci | CDR residues only |
| `esm2_midlayers` | 2560 | Intermediate LM | transformers | Avg layers 20-33 |
| `amplify` | 1920 | Pretrained LM | transformers | AMPLIFY 350M, H+L concat |
| `amplify_cdr` | 1920 | CDR-pooled LM | transformers + anarci | AMPLIFY 350M, CDR-only |

---

## 5. Pretrained Model Downloads

HuggingFace models auto-download on first use, but you can pre-download them to
avoid delays during experiments. Downloads go to `$HF_HOME/hub/` (default:
`~/.cache/huggingface/hub/`).

### 5a. Set cache location (important if disk is limited)

```bash
# Point to a location with enough space
export HF_HOME=/path/to/large/disk/.cache/huggingface

# Make persistent (add to your shell profile or setup.sh)
echo 'export HF_HOME=/path/to/large/disk/.cache/huggingface' >> ~/.bashrc
```

### 5b. Download HuggingFace models

Run these on a machine with internet access. Each line downloads one model.

```bash
# ESM-2 650M (~2.5 GB) — general protein LM, most important baseline
python -c "from transformers import AutoTokenizer, EsmModel; AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"

# IgBERT (~1.6 GB) — SOTA paired antibody model
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('Exscientia/IgBert'); BertModel.from_pretrained('Exscientia/IgBert')"

# ProtT5-XL (~11 GB) — largest, best general protein encoder
python -c "from transformers import T5Tokenizer, T5EncoderModel; T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False); T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd')"

# Ab-RoBERTa (~500 MB) — antibody-specific with SAA tokenizer
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('mogam-ai/Ab-RoBERTa'); AutoModel.from_pretrained('mogam-ai/Ab-RoBERTa')"

# AntiBERTy (~350 MB) — antibody LM (used by IgFold)
python -c "from transformers import RoFormerTokenizer, RoFormerModel; RoFormerTokenizer.from_pretrained('alchemab/antiberta2'); RoFormerModel.from_pretrained('alchemab/antiberta2')"

# p-IgGen (~90 MB) — paired antibody generative model
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('ollieturnbull/p-IgGen'); AutoModelForCausalLM.from_pretrained('ollieturnbull/p-IgGen')"

# Molecular models (auto-download, but can pre-cache)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('HUBioDataLab/SELFormer')"
```

### 5c. BALM (manual — not on HuggingFace)

```bash
# Clone repo
git clone https://github.com/BEAM-Labs/BALM.git

# Download pretrained weights from Google Drive:
# https://drive.google.com/drive/folders/1foy264CIawBIT3QFTdc6JBVxw6MQfvQd
# Place contents in BALM/pretrained-BALM/

# Usage in Python:
# create_balm_embeddings(sequences, balm_repo_path='/path/to/BALM')
```

### 5d. Model size reference

| Model | HF ID | Size |
|-------|-------|------|
| ProtT5-XL | `Rostlab/prot_t5_xl_bfd` | ~11 GB |
| ESM-2 650M | `facebook/esm2_t33_650M_UR50D` | ~2.5 GB |
| IgBERT | `Exscientia/IgBert` | ~1.6 GB |
| BALM | GitHub / Google Drive | ~600 MB |
| Ab-RoBERTa | `mogam-ai/Ab-RoBERTa` | ~500 MB |
| AntiBERTy | `alchemab/antiberta2` | ~350 MB |
| MoLFormer | `ibm/MoLFormer-XL-both-10pct` | ~180 MB |
| ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1` | ~340 MB |
| p-IgGen | `ollieturnbull/p-IgGen` | ~90 MB |
| **Total (all HF)** | | **~17 GB** |

---

## 6. External Storage (Optional)

If disk space is limited, you can offload large data and model weights to an
external drive using symlinks. All code paths remain unchanged.

### 6a. Move HuggingFace cache

```bash
# Copy cache to external drive
rsync -a ~/.cache/huggingface/hub/ /Volumes/YOUR_DRIVE/huggingface/hub/

# Replace with symlink
rm -rf ~/.cache/huggingface/hub
ln -s /Volumes/YOUR_DRIVE/huggingface/hub ~/.cache/huggingface/hub
```

### 6b. Move large data files

```bash
# Example: move QM9 dataset
rsync -a /path/to/KIRBy/data/QM9/ /Volumes/YOUR_DRIVE/KIRBy/data/QM9/
rm -rf /path/to/KIRBy/data/QM9
ln -s /Volumes/YOUR_DRIVE/KIRBy/data/QM9 /path/to/KIRBy/data/QM9

# Repeat for any large directories:
# tests/data/QM9, tests/tybalt/data, tests/feats_pt.zip, etc.
```

### 6c. Alternative: set HF_HOME directly

Instead of symlinks, you can point HuggingFace to the external drive:

```bash
export HF_HOME=/Volumes/YOUR_DRIVE/huggingface
# Add to shell profile to persist
```

---

## 7. ARC Server Setup (Oxford)

For running on the ARC HPC cluster at Oxford.

### 7a. First-time setup

```bash
ssh scat9264@arc-login.arc.ox.ac.uk
cd /data/stat-cadd/scat9264

# Clone repo
git clone https://github.com/adpunt/KIRBy.git
cd KIRBy

# Create/activate environment
# (use whatever conda/module system ARC provides)
. setup.sh

# Point HF cache to data partition (home has limited quota!)
export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface
echo 'export HF_HOME=/data/stat-cadd/scat9264/.cache/huggingface' >> setup.sh

# Install
pip install -e ".[qsar,antibody,uncertainty]"
conda install -c bioconda anarci
pip install ablang ablang2 sentencepiece
```

### 7b. Download models (on login node — has internet)

Compute nodes may not have internet access. Download everything on the login node:

```bash
cd /data/stat-cadd/scat9264/KIRBy
. setup.sh

# mol2vec weights
wget https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl

# HuggingFace models (run each line — they cache to $HF_HOME)
python -c "from transformers import AutoTokenizer, EsmModel; AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')"
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('Exscientia/IgBert'); BertModel.from_pretrained('Exscientia/IgBert')"
python -c "from transformers import T5Tokenizer, T5EncoderModel; T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False); T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd')"
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('mogam-ai/Ab-RoBERTa'); AutoModel.from_pretrained('mogam-ai/Ab-RoBERTa')"
python -c "from transformers import RoFormerTokenizer, RoFormerModel; RoFormerTokenizer.from_pretrained('alchemab/antiberta2'); RoFormerModel.from_pretrained('alchemab/antiberta2')"
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('ollieturnbull/p-IgGen'); AutoModelForCausalLM.from_pretrained('ollieturnbull/p-IgGen')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')"
python -c "from transformers import AutoModel; AutoModel.from_pretrained('HUBioDataLab/SELFormer')"
```

### 7c. Deploy code updates

```bash
# On your local machine:
git push

# On the server:
cd /data/stat-cadd/scat9264/KIRBy
git pull
. setup.sh
```

### 7d. Submit jobs

```bash
# Example SLURM submission
sbatch tests/slurm_scripts/run_complexity_experiments.sh quick
sbatch tests/slurm_scripts/run_similarity_experiments.sh full
```

---

## 7b. Running Hybrid Search

### Model Hyperparameter Tuning

All CLIs support `--tune` to run `RandomizedSearchCV` per model before the hybrid search.
Tuning uses the concatenation of all base reps as features and tunes on 3-fold CV by default.
Ridge is skipped (uses internal CV). Tuned models are passed directly to `find_best_hybrid()`.

```bash
# Tune models before search (any domain CLI)
python tests/find_best_hybrid_molecular.py logd --reps fast --search fast --tune

# Control iterations (default: 20, capped at grid size)
python tests/find_best_hybrid_molecular.py logd --reps fast --search fast --tune --tune-iter 10

# Full tuned survey on server
bash tests/slurm_scripts/submit_tuned_survey.sh medium 20
```

**Search spaces** (defined in `src/kirby/find_best_hybrid.py`):

| Model | Tuned Hyperparameters |
|-------|----------------------|
| RF | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |
| LightGBM | n_estimators, max_depth, learning_rate, num_leaves |
| KNN | n_neighbors, weights |
| MLP | hidden_layer_sizes, alpha, learning_rate_init |
| SVM | C, gamma |
| Ridge | Skipped (uses RidgeCV/LogisticRegressionCV internally) |

**Output paths:** When `--tune` is used without `--output`, results auto-save to
`results_{dataset}_{search}_tuned.json` (vs `results_{dataset}_{search}.json` for defaults).
The SLURM tuned survey saves to `tests/results/tuned_survey/` (vs `tests/results/pp_survey/`).

**TODO: Compare tuned vs default results.** After running both the default survey
(`submit_post_preprocessing_survey.sh`) and tuned survey (`submit_tuned_survey.sh`),
compare per-dataset best scores. The better of the two (tuned or default) should be
reported in final results. Expected: tuning helps RF/XGB/SVM most; KNN/MLP less so;
Ridge unaffected.

### Molecular

```bash
# Quick local test (3 reps, fast search)
python tests/find_best_hybrid_molecular.py logd --reps fast --search fast

# Full server run (15 reps, incl quality-filtered graph kernel)
python tests/find_best_hybrid_molecular.py logd --reps all --search long
```

### Antibody

```bash
# Quick local test (4 classical reps)
python tests/find_best_hybrid_antibody.py sabdab_affinity --reps fast --search fast

# With language models
python tests/find_best_hybrid_antibody.py sabdab_affinity --reps fast+lm --search medium

# Custom CSV dataset
python tests/find_best_hybrid_antibody.py data.csv --heavy-col VH --light-col VL --target-col ddG --reps all_server

# With augmented features (developability, humanness, CDR lengths, etc.)
python tests/find_best_hybrid_antibody.py sabdab_affinity --reps fast+lm --search medium --augment

# Enable result caching (speeds up repeated runs)
python tests/find_best_hybrid_antibody.py sabdab_affinity --reps fast --search fast --cache
```

### FLAb Datasets (Antibody Fitness Landscapes)

FLAb datasets are auto-downloaded from GitHub on first use. Available presets:

| Name | Property | ~Samples | Intra-family? |
|------|----------|----------|---------------|
| `flab_expression_koenig` | Expression (enrichment ratio) | ~4,000 | Yes (G6 antibody) |
| `flab_binding_koenig` | Binding (Kd) | ~4,000 | Yes (G6 antibody) |
| `flab_thermo_jain` | Thermostability (Tm) | ~137 | No (diverse) |
| `flab_aggregation_jain` | Aggregation (AC-SINS) | ~137 | No (diverse) |
| `flab_thermo_hie` | Thermostability (Tm) | ~500+ | Yes (S309 DMS) |
| `flab_binding_rosace` | Binding (Kd) | ~500+ | Yes (adalimumab DMS) |
| `abcov_kd` | Binding (-log Kd) | ~1,780 | No (diverse CoV abs) |
| `abcov_ic50` | Neutralization (IC50) | ~1,780 | No (diverse CoV abs) |

```bash
# FLAb with mutation-aware representations
python tests/find_best_hybrid_antibody.py flab_expression_koenig --reps mutation_fast --search fast

# With delta representations (ΔX = X_mutant - X_wildtype)
python tests/find_best_hybrid_antibody.py flab_expression_koenig --reps mutation_fast --delta --search fast

# Full mutation-aware preset with delta
python tests/find_best_hybrid_antibody.py flab_binding_koenig --reps mutation_full --delta --search medium

# Ab-CoV (inter-family, no delta)
python tests/find_best_hybrid_antibody.py abcov_kd --reps fast --search fast
```

### AbAgym Datasets (Antibody-Side DMS)

[AbAgym](https://github.com/3BioCompBio/AbAgym) — 68 DMS experiments, 5 mutate the antibody side.
Auto-downloaded from GitHub on first use. Requires BioPython for PDB sequence extraction.

| Name | Target | PDB | ~Mutations |
|------|--------|-----|------------|
| `abagym_all` | All 5 pooled | — | ~29K |
| `abagym_Ang2_2017_G6` | Angiopoietin-2 | 4zfg | ~6K |
| `abagym_EGFR_2013_Cetuximab` | EGFR | 1yy9 | ~12K |
| `abagym_lysozyme_2019_D441` | Lysozyme | 1mlc | ~1.8K |
| `abagym_VEGF_2017b_G6` | VEGF | 4zff | ~9.5K |
| `abagym_HER2_2021_trastuzumab` | HER2 | 8pwh | ~184 |

```bash
# AbAgym pooled
python tests/find_best_hybrid_antibody.py abagym_all --reps mutation_fast --search fast

# Individual experiment
python tests/find_best_hybrid_antibody.py abagym_Ang2_2017_G6 --reps mutation_full --delta --search medium
```

### AbBind Dataset (ΔΔG)

[AB-Bind](https://github.com/sarahsirin/AB-Bind-Database) — ~754 Ab-Ag mutations with
ΔΔG (kcal/mol) after filtering non-Ab-Ag PDBs. Auto-downloaded from GitHub.
Requires BioPython for PDB sequence extraction (~100 PDBs, ~5MB total).

```bash
python tests/find_best_hybrid_antibody.py abbind --reps mutation_fast --search fast
```

### Structural Features (ImmuneBuilder)

Structure-based features (28D) derived from ABodyBuilder2-predicted 3D structures.
Includes SASA (15D), prediction confidence (5D), and paratope features (8D).

**Server-only** — structure prediction is ~60s/structure on CPU, ~5s on GPU.
Without ImmuneBuilder, falls back to sequence-only paratope features (8D).

```bash
# Install ImmuneBuilder (server)
pip install ImmuneBuilder
conda install -c conda-forge openmm pdbfixer
```

### Mutation-Aware Rep Presets

| Preset | Reps | Description |
|--------|------|-------------|
| `mutation_fast` | kmer, aac, blosum62, physicochemical, positional, biophysical_delta | Classical + mutation features (6 reps) |
| `mutation_full` | + antiberty, ablang2, lm_score | + LM embeddings and LM scoring (9 reps) |
| `mutation_structural` | + structural | + structure-based features (10 reps, server-only) |

With `--delta`: auto-adds `delta_*` for every non-mutation base rep, roughly doubling the rep count.

### Antibody Complexity Experiments

```bash
# Quick test (FLAb + mutation reps, Exp 1+2+6)
sbatch tests/slurm_scripts/run_antibody_complexity.sh quick

# With delta reps
sbatch tests/slurm_scripts/run_antibody_complexity.sh quick_delta

# Full run on specific dataset
sbatch tests/slurm_scripts/run_antibody_complexity.sh full flab_expression_koenig
sbatch tests/slurm_scripts/run_antibody_complexity.sh full_delta flab_binding_koenig

# AbAgym / AbBind datasets
sbatch tests/slurm_scripts/run_antibody_complexity.sh full abagym_all
sbatch tests/slurm_scripts/run_antibody_complexity.sh full abbind
```

---

## 7c. TCGA Genomics / Histopathology

### Dependencies

```bash
pip install -e ".[tcga]"
# Installs: scanpy, anndata, gseapy, decoupler, requests, combat (pyComBat)

# BulkRNABert (InstaDeep, purpose-built for bulk RNA-seq)
# Requires transformers from source for custom model architecture
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

### Data Sources

TCGA data is downloaded automatically on first use from UCSC Xena (open access).
No GDC/dbGaP credentials needed. Alternative: `python scripts/download_tcga_gdc.py TCGA-BRCA`

| Data Type | Source | Auto-download? | Cache Size |
|-----------|--------|---------------|------------|
| Gene expression | UCSC Xena (log2 TPM+1) | Yes | ~50-80 MB per cohort |
| Clinical/phenotype | UCSC Xena | Yes | <5 MB |
| Survival endpoints | TCGA-CDR (Liu et al. 2018) | Yes | <1 MB |
| Somatic mutations | MC3 MAF (Xena) | Yes | ~50 MB |
| WSI features | Mendeley GigaSSL | Yes (on first use) | ~50 MB |

### Available Datasets

| Name | Task | ~Samples | Description |
|------|------|----------|-------------|
| `tcga_luad_vs_lusc` | Binary classification | ~1,016 | Lung adenocarcinoma vs squamous cell |
| `tcga_brca_pam50` | 4-class classification | ~900 | Breast cancer PAM50 subtypes |
| `tcga_brca_survival` | Regression | ~400-500 | Overall survival (uncensored) |

### Representation Presets

| Preset | Reps | Description |
|--------|------|-------------|
| `fast` | gene_pca100, gene_hallmark, gene_progeny, wsi_ctranspath, wsi_phikon | 5 reps, CPU-only |
| `fast+learned` | + gene_hvg2000, gene_kegg, gene_dorothea, gene_combined_pathway, wsi_conch, wsi_hoptimus0 | 11 reps |
| `all` | + gene_mutation, wsi_gigapath | 13 reps (all genomic + WSI) |
| `genomic_fast` | gene_pca100, gene_hallmark, gene_progeny | 3 reps, no WSI |
| `genomic_all` | all 8 genomic reps, no WSI | 8 reps |

### Running TCGA Experiments

```bash
# Quick local test (downloads data on first run)
python tests/find_best_hybrid_tcga.py tcga_luad_vs_lusc --reps fast --search fast

# BRCA PAM50 (multi-class)
python tests/find_best_hybrid_tcga.py tcga_brca_pam50 --reps fast --search fast

# Survival regression
python tests/find_best_hybrid_tcga.py tcga_brca_survival --reps fast --search fast

# Genomic-only (skip WSI features)
python tests/find_best_hybrid_tcga.py tcga_luad_vs_lusc --reps fast --genomic-only --search fast

# Full search on server
python tests/find_best_hybrid_tcga.py tcga_luad_vs_lusc --reps all --search long
```

### SLURM (Server)

```bash
# Default: fast reps, LUAD vs LUSC
sbatch tests/slurm_scripts/run_tcga_scenarios.sh

# Custom
sbatch tests/slurm_scripts/run_tcga_scenarios.sh fast+learned tcga_brca_pam50
sbatch tests/slurm_scripts/run_tcga_scenarios.sh all tcga_brca_survival
```

### Cross-Validation Grouping

TCGA experiments use **StratifiedGroupKFold by tissue source site (TSS)** to prevent
data leakage from site-specific confounders (Howard et al. 2021). TSS codes are
extracted from TCGA barcodes automatically.

---

## 8. Verification

Run these to verify your installation works. Each test uses 2-3 sequences and
takes seconds (no GPU needed).

### 8a. Core KIRBy

```bash
python -c "
from kirby.hybrid import create_hybrid
from kirby.find_best_hybrid import find_best_hybrid
import numpy as np
X = {'a': np.random.randn(10, 5), 'b': np.random.randn(10, 3)}
h = create_hybrid(X)
print(f'Hybrid: {h.shape}')
print('Core OK')
"
```

### 8b. Molecular representations

```bash
python -c "
from kirby.representations.molecular import create_ecfp4, create_pdv
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
print(f'ECFP4: {create_ecfp4(smiles).shape}')
print(f'PDV: {create_pdv(smiles).shape}')
print('Molecular OK')
"
```

### 8c. Antibody representations (non-pretrained)

```bash
python -c "
from kirby.representations.antibody import aac_encode, onehot_encode, blosum_encode
heavy = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
light = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']

print(f'AAC:     {aac_encode(heavy, light).shape}')
print(f'One-hot: {onehot_encode(heavy, light).shape}')
print(f'BLOSUM:  {blosum_encode(heavy, light).shape}')
print('All antibody non-pretrained OK')
"
```

### 8d. Antibody pretrained models

```bash
# Test each model individually (each downloads weights on first run)

# IgBERT (768D, ~1.6 GB download)
python -c "
from kirby.representations.antibody import embed_igbert
h = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
l = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']
print(f'IgBERT: {embed_igbert(h, l).shape}')
"

# ESM-2 650M (2560D, ~2.5 GB download)
python -c "
from kirby.representations.antibody import embed_esm2
h = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
l = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']
print(f'ESM-2: {embed_esm2(h, l).shape}')
"

# ProtT5-XL (2048D, ~11 GB download)
python -c "
from kirby.representations.antibody import embed_prot_t5
h = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
l = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']
print(f'ProtT5-XL: {embed_prot_t5(h, l).shape}')
"

# AbLang-2 (768D, bundled with pip package)
python -c "
from kirby.representations.antibody import embed_ablang2
h = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
l = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']
print(f'AbLang-2: {embed_ablang2(h, l).shape}')
"

# AntiBERTy (1024D, ~350 MB download)
python -c "
from kirby.representations.antibody import embed_antiberty
h = ['EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS']
l = ['DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK']
print(f'AntiBERTy: {embed_antiberty(h, l).shape}')
"
```

---

## Troubleshooting

**`T5Tokenizer requires the SentencePiece library`**
```bash
pip install sentencepiece
```

**`ANARCI not available` warnings**
```bash
conda install -c bioconda anarci
# ANARCI is conda-only, not available via pip
```

**`Repository Not Found` when loading a model**
- Check you have internet access
- Check the exact HF model ID (see table in Section 5d)
- For BALM: it's not on HuggingFace — see Section 5c

**`GPTNeoXForCausalLM.forward() got an unexpected keyword argument 'token_type_ids'`**
- This was a bug in p-IgGen loading, fixed in commit `aa05745`
- Make sure you have the latest code: `git pull`

**HuggingFace cache filling up disk**
- Set `HF_HOME` to a larger drive (Section 6)
- Or symlink `~/.cache/huggingface/hub` to external storage

**`Not enough free disk space` for ProtT5-XL**
- ProtT5-XL is ~11 GB. Use an external drive or skip it (it's optional).
- Smaller alternative: use ESM-2 650M (~2.5 GB) which has similar performance

**Server compute nodes can't download models**
- Download everything on the login node first (Section 7b)
- Set `HF_HOME` so compute nodes find the cached weights
