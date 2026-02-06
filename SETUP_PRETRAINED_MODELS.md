# Pretrained Model Setup

KIRBy supports 16 molecular representation models. Most work automatically; a few require manual setup.

## Quick Start

```bash
# Install KIRBy with chemistry dependencies
pip install -e ".[qsar]"

# For RDKit (required for all models)
conda install -c conda-forge rdkit
```

**That's it for most models.** The table below shows what works immediately vs. what needs extra steps.

---

## Model Overview

| Model | Dimension | Setup | Notes |
|-------|-----------|-------|-------|
| ECFP4 | 2048 | None | Morgan fingerprints |
| MACCS | 167 | None | Structural keys |
| PDV | 200 | None | Physicochemical descriptors |
| ChemBERTa | 768 | Auto | HuggingFace download |
| MolFormer | 768 | Auto | HuggingFace download |
| SELFormer | 768 | Auto | HuggingFace download |
| ChemBERT | 256 | Auto | HuggingFace download |
| Uni-Mol | 512 | Auto | Downloads on first use |
| MHG-GNN | 1024 | Auto | HuggingFace download |
| SMI-TED | 768 | Auto | HuggingFace download |
| SchNet | 128 | Auto | 3D GNN (generates conformers) |
| mol2vec | 300 | [Manual](#mol2vec) | Download 1 file |
| GROVER | 4800 | [Manual](#grover) | Clone repo + download weights |
| Chemformer | 512 | [Manual](#chemformer) | Clone repo + download weights |
| MolCLR | 512 | [Manual](#molclr) | Clone repo (weights included) |
| GraphMVP | 300 | [Manual](#graphmvp) | Clone repo + download weights |

---

## Automatic Models (No Setup Required)

These models download weights automatically on first use:

```python
from kirby.representations.molecular import (
    create_ecfp4,      # Fingerprints - instant
    create_maccs,
    create_pdv,
    create_chemberta,  # HuggingFace - downloads ~500MB first time
    create_molformer,
    create_selformer,
    create_chembert,
    create_unimol,     # Downloads weights automatically
    create_mhg_gnn,
    create_smited,
    create_schnet,
)

# Example usage
embeddings = create_chemberta(['CCO', 'c1ccccc1'])
print(embeddings.shape)  # (2, 768)
```

---

## Manual Setup Models

### Environment Variable (Optional)

Set `KIRBY_MODELS_DIR` to specify where model files are stored:

```bash
# Add to ~/.bashrc or ~/.zshrc
export KIRBY_MODELS_DIR="$HOME/kirby_models"
```

If not set, KIRBy searches `~/kirby_models`, `~/repos`, and common locations.

---

### mol2vec

Word2vec embeddings on molecular substructures (300d).

```bash
# Download model file
mkdir -p ~/kirby_models
wget -P ~/kirby_models https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl
```

```python
from kirby.representations.molecular import create_mol2vec
embeddings = create_mol2vec(['CCO', 'c1ccccc1'])  # (2, 300)
```

---

### GROVER

Self-supervised graph transformer pretrained on 10M molecules (4800d).

**1. Clone repository:**
```bash
cd ~/kirby_models  # or ~/repos
git clone https://github.com/tencent-ailab/grover.git
```

**2. Download weights (browser required):**

Go to: https://1drv.ms/u/s!Ak4XFI0qaGjOhdlxC3mGn0LC1NFd6g

Download `grover_large.pt` and place in the grover directory:
```bash
mv ~/Downloads/grover_large.pt ~/kirby_models/grover/
```

**3. Install dependencies:**
```bash
pip install typed-argument-parser descriptastorus
```

**4. Test:**
```python
from kirby.representations.molecular import create_grover
embeddings = create_grover(['CCO', 'c1ccccc1'])  # (2, 4800)
```

---

### Chemformer

BART transformer from AstraZeneca pretrained on SMILES (512d).

**1. Clone repository:**
```bash
cd ~/kirby_models  # or ~/repos
git clone https://github.com/MolecularAI/Chemformer.git
```

**2. Download weights (browser required):**

Go to: https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq

Navigate to `models/pre-trained/combined/` and download `step=1000000.ckpt`.

Place it at:
```bash
mkdir -p ~/kirby_models/Chemformer/models/pre-trained/combined
mv ~/Downloads/step=1000000.ckpt ~/kirby_models/Chemformer/models/pre-trained/combined/
```

**3. Convert to simplified format (recommended for servers):**

The original checkpoint requires pytorch-lightning to unpickle, which can cause version conflicts. Convert it to a simpler format:

```bash
cd ~/kirby_models/Chemformer
python -c "import torch; ckpt = torch.load('models/pre-trained/combined/step=1000000.ckpt', map_location='cpu', weights_only=False); torch.save({'state_dict': ckpt.get('state_dict', ckpt), 'hyper_parameters': ckpt.get('hyper_parameters', {})}, 'chemformer_weights.pt'); print('Saved chemformer_weights.pt')"
```

**4. Test:**
```python
from kirby.representations.molecular import create_chemformer
embeddings = create_chemformer(['CCO', 'c1ccccc1'])  # (2, 512)
```

Note: KIRBy includes a built-in SMILES tokenizer, so pysmilesutils is NOT required.

---

### MolCLR

Contrastive learning on molecular graphs (512d). Weights are included in the repo.

**1. Clone repository:**
```bash
cd ~/kirby_models
git clone https://github.com/yuyangw/MolCLR.git
```

**2. Install dependencies:**
```bash
pip install torch-geometric

# torch-scatter/torch-cluster (match your PyTorch version)
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
# For CUDA: replace +cpu with +cu118 or your CUDA version
```

**3. Test:**
```python
from kirby.representations.molecular import create_molclr
embeddings = create_molclr(['CCO', 'c1ccccc1'])  # (2, 512)
```

---

### GraphMVP

Multi-view pretraining with 2D graphs and 3D conformers (300d).

**1. Clone repository:**
```bash
cd ~/kirby_models
git clone https://github.com/chao1224/GraphMVP.git
```

**2. Download weights:**
```bash
cd ~/kirby_models/GraphMVP
python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('MoleculeSTM_weights/pretrained_GraphMVP/GraphMVP_C', exist_ok=True); hf_hub_download(repo_id='chao1224/MoleculeSTM', filename='pretrained_GraphMVP/GraphMVP_C/model.pth', local_dir='MoleculeSTM_weights')"
```

**3. Install dependencies:**
```bash
pip install torch-geometric ogb ase

# torch-scatter (match your PyTorch version)
# Check version: python -c "import torch; print(torch.__version__)"
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
# For CUDA: replace +cpu with +cu118 or your CUDA version
```

**4. Test:**
```python
from kirby.representations.molecular import create_graphmvp
embeddings = create_graphmvp(['CCO', 'c1ccccc1'])  # (2, 300)
```

---

## Server/HPC Setup

Complete setup for a fresh server environment.

### Step 1: Install all pip dependencies

```bash
# All dependencies in one command (copy-paste friendly)
pip install typed-argument-parser descriptastorus torch-geometric ogb ase gensim mol2vec huggingface_hub --user

# torch-scatter (check your PyTorch version first)
python -c "import torch; print(torch.__version__)"
# Then install matching version (examples):
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html --user
# For CUDA: pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu118.html --user
```

### Step 2: Clone repos and download weights

```bash
mkdir -p ~/kirby_models && cd ~/kirby_models

# Clone all repos
git clone https://github.com/tencent-ailab/grover.git
git clone https://github.com/MolecularAI/Chemformer.git
git clone https://github.com/yuyangw/MolCLR.git
git clone https://github.com/chao1224/GraphMVP.git

# Download mol2vec weights
wget https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl

# Download GraphMVP weights
cd GraphMVP
python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('MoleculeSTM_weights/pretrained_GraphMVP/GraphMVP_C', exist_ok=True); hf_hub_download(repo_id='chao1224/MoleculeSTM', filename='pretrained_GraphMVP/GraphMVP_C/model.pth', local_dir='MoleculeSTM_weights')"
cd ..
```

### Step 3: Copy weights from local machine

GROVER and Chemformer weights require browser download. On your **local machine**:

```bash
# Replace SERVER with your server address (e.g., user@server.edu)
scp ~/kirby_models/grover/grover_large.pt SERVER:~/kirby_models/grover/
scp ~/kirby_models/Chemformer/chemformer_weights.pt SERVER:~/kirby_models/Chemformer/
```

### Complete dependency list

For reference, here are all pip packages needed for all 16 models:

| Package | Models | Install |
|---------|--------|---------|
| torch-geometric | MolCLR, GraphMVP, SchNet | `pip install torch-geometric` |
| torch-scatter | MolCLR, GraphMVP | See version-specific install above |
| ogb | GraphMVP | `pip install ogb` |
| ase | GraphMVP, SchNet | `pip install ase` |
| typed-argument-parser | GROVER | `pip install typed-argument-parser` |
| descriptastorus | GROVER | `pip install descriptastorus` |
| gensim, mol2vec | mol2vec | `pip install gensim mol2vec` |
| huggingface_hub | GraphMVP weights, HF models | `pip install huggingface_hub` |
| transformers | ChemBERTa, MolFormer, etc. | `pip install transformers` |
| selfies | SELFormer | `pip install selfies` |

---

## Verification

Run the verification script to check all models:

```bash
python scripts/verify_models.py
```

Or test individual models:

```python
from kirby.representations.molecular import create_chemberta
emb = create_chemberta(['CCO'])
print(f"ChemBERTa: {emb.shape}")  # (1, 768)
```

---

## Troubleshooting

### HuggingFace models slow on first run
First use downloads 500MB-1GB of weights to `~/.cache/huggingface/`. Subsequent runs use cache.

### GROVER "module not found"
Ensure the GROVER repo contains the `grover/` subdirectory with Python code:
```bash
ls ~/kirby_models/grover/grover/  # Should show data/, util/, etc.
```

### torch-scatter segfault (macOS)
MolCLR and GraphMVP may crash on macOS due to torch-scatter version mismatches. These typically work on Linux. Reinstall with matching versions:
```bash
pip uninstall torch-scatter torch-cluster
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-X.X.X+cpu.html
```

### CUDA out of memory
Reduce batch size:
```python
create_grover(smiles, batch_size=8)
create_chemberta(smiles, batch_size=8)
```

### Tokenizer parallelism warning
Handled automatically by setting `TOKENIZERS_PARALLELISM=false`.

---

## Dependencies by Use Case

```bash
# Minimal (fingerprints only) - just need RDKit
conda install -c conda-forge rdkit

# HuggingFace transformers
pip install torch transformers huggingface_hub selfies

# Full QSAR stack
pip install -e ".[qsar]"

# GROVER
pip install typed-argument-parser descriptastorus

# Chemformer (no extra deps needed - uses built-in tokenizer)

# Graph models (MolCLR, GraphMVP)
pip install torch-geometric ogb ase
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```
