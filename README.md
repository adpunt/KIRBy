# KIRBy: Knowledge Integration by Representation Borrowing

**Automated Representation Optimization Through Fine-Tuned Feature Learning**

## Overview

KIRBy creates task-optimized molecular representations by:
1. Fine-tuning neural architectures (MoLFormer, ChemBERTa, GNN) on YOUR specific data
2. Extracting learned embeddings that are tailored to YOUR property
3. Combining these task-specific embeddings into hybrid representations
4. Training final models on hybrids that capture complementary molecular patterns

**Key Insight**: Each neural architecture learns different aspects of your task. MoLFormer learns sequential patterns, GNNs learn graph topology, ChemBERTa provides alternative SMILES encoding. The hybrid combines these learned insights.

## Version 0.2.0 - Complete Implementation

### Available Representations

**Static (No Training)**:
- ECFP4: Morgan circular fingerprints ✓
- PDV: 200 RDKit physical descriptors ✓

**Pretrained (Frozen)**:
- mol2vec: Word2vec on molecular substructures ✓ (requires model download)
- Graph Kernels: Weisfeiler-Lehman graph similarity ✓
- MHG-GNN: Pretrained GNN encoder ⚠️ (requires local mhg_model code and pretrained weights)

**Fine-Tuned (Task-Specific)**:
- MoLFormer: Transformer on SMILES/SELFIES ✓
- ChemBERTa: BERT-style on SMILES/SELFIES ✓
- GNN: Graph neural networks (GCN/GAT/MPNN) ✓

**Gaussian Processes**:
- GAUCHE GP: GP with graph kernels (includes uncertainty) ✓

**Hybrid**:
- Combine any/all representations above ✓

**Legend**: ✓ = Publicly accessible | ⚠️ = Available but requires manual setup

## Installation

### ONE-TIME SETUP (First Time Only)

**Download mol2vec pretrained weights:**

```bash
cd KIRBY
wget https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl
```

**Optional - MHG-GNN Setup:**

MHG-GNN is NOT installed or downloaded automatically. KIRBy searches for a local directory containing `models/mhg_model` in the following paths:
- `~/repos/materials`
- `~/materials`
- `../materials`
- `../../materials`

**To set up MHG-GNN:**

1. The original IBM GitHub repository (https://github.com/IBM/materials) is deprecated, though some documentation may still reference it. All necessary files are now available on HuggingFace.

2. Download the repository code from HuggingFace:
   ```bash
   cd ~/repos  # or your preferred location from search paths above
   git clone https://huggingface.co/ibm-research/materials.mhg-ged materials
   ```

3. Download the pretrained weights file manually:
   - File: `mhggnn_pretrained_model_0724_2023.pickle`
   - URL: https://huggingface.co/ibm-research/materials.mhg-ged
   - Place in the `materials/` directory

4. Install Python dependencies:
   ```bash
   cd KIRBY
   pip install -e ".[mhg]"
   ```
   Note: This installs only the Python dependencies. The model code and weights must be downloaded separately as described above.

### REGULAR SETUP (Every Time You Work)

**Single command:**

```bash
cd KIRBY
pip install -e .
```

That's it. This installs KIRBy and all publicly available dependencies from PyPI.

## Quick Start

```python
from kirby.datasets.esol import load_esol_data
from kirby.representations.molecular import (
    create_ecfp4,
    finetune_molformer,
    finetune_gnn,
    encode_from_model,
    create_hybrid
)
from sklearn.ensemble import RandomForestRegressor

# Load data
data = load_esol_data(splitter='scaffold')
train_smiles, train_labels = data['train_smiles'], data['train_labels']
test_smiles, test_labels = data['test_smiles'], data['test_labels']

# Split for validation
val_smiles = train_smiles[:200]
val_labels = train_labels[:200]
train_smiles = train_smiles[200:]
train_labels = train_labels[200:]

# 1. Static baseline
ecfp4_train = create_ecfp4(train_smiles)
ecfp4_test = create_ecfp4(test_smiles)

# 2. Fine-tune MoLFormer on YOUR solubility data
molformer_model = finetune_molformer(
    train_smiles, train_labels,
    val_smiles, val_labels,
    epochs=10
)

# 3. Extract task-specific embeddings
molformer_train = encode_from_model(molformer_model, train_smiles)
molformer_test = encode_from_model(molformer_model, test_smiles)

# 4. Fine-tune GNN on YOUR data
gnn_model = finetune_gnn(
    train_smiles, train_labels,
    val_smiles, val_labels,
    gnn_type='gcn',
    epochs=100
)

gnn_train = encode_from_model(gnn_model, train_smiles)
gnn_test = encode_from_model(gnn_model, test_smiles)

# 5. Create hybrid - combines task-optimized representations
hybrid_train = create_hybrid({
    'ecfp4': ecfp4_train,
    'molformer': molformer_train,
    'gnn': gnn_train
})

hybrid_test = create_hybrid({
    'ecfp4': ecfp4_test,
    'molformer': molformer_test,
    'gnn': gnn_test
})

# 6. Train final model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(hybrid_train, train_labels)
predictions = rf.predict(hybrid_test)
```

## Running Tests

```bash
# Test all representations on ESOL (solubility)
python tests/test_esol.py

# Test all representations on QM9 (HOMO energy)
python tests/test_qm9.py
```

Each test:
- Tests 10 individual representations
- Shows 4 hybrid combinations
- Ranks results by performance

## Project Structure

```
kirby/
├── src/kirby/
│   ├── datasets/
│   │   ├── esol.py              # ESOL dataset loader
│   │   └── qm9.py               # QM9 dataset loader
│   └── representations/
│       └── molecular.py         # All molecular representations
├── tests/
│   ├── test_esol.py            # Comprehensive ESOL test
│   └── test_qm9.py             # Comprehensive QM9 test
├── setup.py                     # Package configuration
└── README.md                    # This file
```

## Available Representations

### Static (No Training)

```python
create_ecfp4(smiles_list, radius=2, n_bits=2048)
create_pdv(smiles_list)
```

### Pretrained (Frozen)

```python
create_mol2vec(smiles_list)  # Requires mol2vec weights
create_graph_kernel(smiles_list, kernel='weisfeiler_lehman', n_features=100)
create_mhg_gnn(smiles_list, n_features=256)  # Requires local mhg_model code + weights
```

### Fine-Tuned (Task-Specific)

```python
# MoLFormer
finetune_molformer(train_smiles, train_labels, val_smiles, val_labels,
                   epochs=10, use_selfies=False)

# ChemBERTa
finetune_chemberta(train_smiles, train_labels, val_smiles, val_labels,
                   epochs=10, use_selfies=False)

# GNN
finetune_gnn(train_smiles, train_labels, val_smiles, val_labels,
             gnn_type='gcn', epochs=100)

# Extract embeddings
encode_from_model(finetuned_model, smiles_list)
```

### Gaussian Processes

```python
# Train GP with graph kernel
train_gauche_gp(train_smiles, train_labels, kernel='weisfeiler_lehman')
predict_gauche_gp(gp_dict, test_smiles)
```

### Hybrid

```python
create_hybrid(embeddings_dict, selection_method='concat', n_features=None)
```

## SELFIES Support

Both MoLFormer and ChemBERTa support SELFIES (guaranteed valid molecular structures):

```python
# Use SELFIES instead of SMILES
molformer_model = finetune_molformer(
    train_smiles, train_labels,
    val_smiles, val_labels,
    use_selfies=True  # ← Automatic SMILES→SELFIES conversion
)

chemberta_model = finetune_chemberta(
    train_smiles, train_labels,
    val_smiles, val_labels,
    use_selfies=True
)
```

## Requirements

**Core** (installed automatically by `pip install -e .`):
- Python >=3.8
- numpy >=1.26.4
- scikit-learn >=1.0.0
- rdkit >=2022.03.1
- torch >=2.0.0
- transformers >=4.41.0
- torch-geometric >=2.3.0
- selfies >=2.1.1
- mol2vec >=0.1
- gensim >=4.0.0
- grakel >=0.1.8
- gauche >=0.1.0
- gpytorch >=1.9.0

**Optional** (requires one-time manual setup):
- mol2vec pretrained weights: Download from GitHub (see installation instructions)
- MHG-GNN: Requires local mhg_model code and pretrained weights from HuggingFace

## Key Concepts

### Fine-Tuning vs Frozen Embeddings

**Frozen (Generic)**:
```python
# Embeddings know general molecular patterns
embeddings = pretrained_model.encode(smiles)
```

**Fine-Tuned (Task-Specific)**:
```python
# Model learns YOUR task (e.g., solubility)
model = finetune_molformer(train_smiles, train_labels_SOLUBILITY)

# Embeddings now emphasize solubility-relevant features
embeddings = encode_from_model(model, smiles)
```

### Why Hybrid Works

Each architecture learns different aspects:
- **MoLFormer**: Sequential SMILES patterns
- **ChemBERTa**: Alternative sequential encoding
- **GNN**: Graph topology and connectivity
- **ECFP4**: Substructure presence (static)
- **PDV**: Physical properties (static)

The hybrid combines these complementary views, each optimized for YOUR specific property.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# If using SABdAB:
conda install -c bioconda anarci

# Run tests
pytest tests/

# Format code
black src/ tests/
```

## Citation

(Placeholder for future publication)

Adelaide Punt  
DPhil Student, Protein Informatics Group  
University of Oxford

## License

MIT License