"""
QM9 Dataset Loader

QM9 contains 134k molecules with computed quantum mechanical properties.
This module loads HOMO-LUMO gap prediction task with scaffold splitting support.

References:
- Ramakrishnan et al. (2014). Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data.
- PyTorch Geometric QM9 dataset
"""

import numpy as np
import torch
import os.path as osp


def _patch_torch_load_for_pyg():
    """Patch torch.load for PyTorch 2.6+ compatibility with torch_geometric."""
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version >= (2, 6):
        try:
            from torch_geometric.data.data import Data
            from torch_geometric.data.storage import GlobalStorage
            torch.serialization.add_safe_globals([Data, GlobalStorage])
        except (ImportError, AttributeError):
            pass


def load_qm9(n_samples=1000, property_idx=4, data_dir='data/QM9', random_seed=42):
    """
    Load QM9 dataset and extract SMILES + labels.
    
    Args:
        n_samples: Number of molecules to sample
        property_idx: Which property to predict (0-11)
        data_dir: Directory to store/load QM9 data
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: {'smiles': [...], 'labels': [...], 'property_name': str}
    """
    try:
        from torch_geometric.datasets import QM9
    except ImportError as e:
        raise ImportError(f"QM9 loader requires torch-geometric: {e}")
    
    import os.path as osp
    
    property_names = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
        'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    # Fix PyTorch 2.6+ weights_only compatibility
    _patch_torch_load_for_pyg()

    print(f"Loading QM9 (property: {property_names[property_idx]})...")

    qm9 = QM9(root=data_dir)
    
    n_samples = min(n_samples, len(qm9))
    smiles_list = []
    targets = []

    import random
    random.seed(random_seed)
    random_indices = random.sample(range(len(qm9)), n_samples)

    print(f"  Extracting {n_samples} molecules...")
    for i, index in enumerate(random_indices):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{n_samples}")
        
        data = qm9[index]
        smiles_list.append(data.smiles)
        targets.append(data.y[0, property_idx].item())
    
    targets = np.array(targets)
    
    mean = targets.mean()
    std = targets.std()
    targets_norm = (targets - mean) / std
    
    print(f"  Loaded {len(smiles_list)} molecules")
    print(f"  Range: [{targets.min():.3f}, {targets.max():.3f}], normalized to mean=0, std=1")
    
    return {
        'smiles': smiles_list,
        'labels': targets_norm,
        'property_name': property_names[property_idx],
        'property_idx': property_idx
    }

def get_qm9_splits(data, splitter='scaffold', train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """
    Split QM9 data into train/val/test sets.
    
    Args:
        data: Dictionary from load_qm9()
        splitter: How to split the data (default: 'scaffold')
            'scaffold': Scaffold split (more realistic, harder, tests generalization)
            'random': Random split
            'stratified': Stratified split
        train_ratio: Fraction for training (default: 0.8) - used for random/stratified
        val_ratio: Fraction for validation (default: 0.1) - used for random/stratified
        random_seed: Random seed
    
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'val': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]}
        }
    """
    smiles = data['smiles']
    labels = data['labels']
    n_samples = len(labels)
    
    if splitter == 'scaffold':
        # Use DeepChem's scaffold splitter
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError("Scaffold splitting requires DeepChem: pip install deepchem")
        
        print(f"Splitting {n_samples} molecules using scaffold split...")
        
        # Create DeepChem dataset
        dataset = dc.data.NumpyDataset(
            X=np.zeros((n_samples, 1)),  # Dummy features
            y=labels.reshape(-1, 1),
            ids=smiles
        )
        
        # Scaffold split
        splitter_obj = dc.splits.ScaffoldSplitter()
        train_dataset, val_dataset, test_dataset = splitter_obj.train_valid_test_split(
            dataset,
            frac_train=train_ratio,
            frac_valid=val_ratio,
            frac_test=1.0 - train_ratio - val_ratio,
            seed=random_seed
        )
        
        # Extract SMILES and labels
        train_idx = [smiles.index(s) for s in train_dataset.ids]
        val_idx = [smiles.index(s) for s in val_dataset.ids]
        test_idx = [smiles.index(s) for s in test_dataset.ids]
        
    elif splitter == 'random':
        print(f"Splitting {n_samples} molecules using random split...")
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
    elif splitter == 'stratified':
        print(f"Splitting {n_samples} molecules using stratified split...")
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(n_samples),
            train_size=train_ratio,
            random_state=random_seed,
            stratify=np.digitize(labels, bins=np.percentile(labels, [25, 50, 75]))
        )
        
        # Second split: val vs test
        val_size = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_seed,
            stratify=np.digitize(labels[temp_idx], bins=np.percentile(labels[temp_idx], [25, 50, 75]))
        )
    else:
        raise ValueError(f"Unknown splitter: {splitter}. Use 'scaffold', 'random', or 'stratified'")
    
    print(f"  Train: {len(train_idx)} molecules")
    print(f"  Val:   {len(val_idx)} molecules")
    print(f"  Test:  {len(test_idx)} molecules")
    
    return {
        'train': {
            'smiles': [smiles[i] for i in train_idx],
            'labels': labels[train_idx]
        },
        'val': {
            'smiles': [smiles[i] for i in val_idx],
            'labels': labels[val_idx]
        },
        'test': {
            'smiles': [smiles[i] for i in test_idx],
            'labels': labels[test_idx]
        }
    }


def load_qm9_homo_lumo(n_samples=1000, data_dir='data/QM9', random_seed=42):
    """
    Load QM9 for HOMO-LUMO gap prediction (most common task).
    
    This is a convenience wrapper around load_qm9() with property_idx=4.
    """
    return load_qm9(
        n_samples=n_samples,
        property_idx=4,  # HOMO-LUMO gap
        data_dir=data_dir,
        random_seed=random_seed
    )