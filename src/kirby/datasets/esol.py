"""
ESOL (Delaney) Dataset Loader

ESOL contains 1128 molecules with measured aqueous solubility (logS).
Standard benchmark for molecular property prediction.

References:
- Delaney, J.S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. J. Chem. Inf. Comput. Sci.
- MoleculeNet benchmark suite
- DeepChem implementation
"""

import numpy as np


def load_esol(splitter='scaffold', data_dir=None):
    """
    Load ESOL (Delaney) aqueous solubility dataset.
    
    Args:
        splitter: How to split the data (default: 'scaffold')
            'scaffold': Scaffold split (more realistic, harder, tests generalization)
            'random': Random split
            'stratified': Stratified split
            'index': Index split (chronological)
        data_dir: Directory to cache data (optional, uses DeepChem default if None)
    
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'val': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]}
        }
    """
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError("ESOL loader requires DeepChem: pip install deepchem")
    
    print(f"Loading ESOL dataset (splitter: {splitter})...")
    
    # Load via DeepChem MoleculeNet
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer='Raw',  # We'll create our own features
        splitter=splitter,
        reload=False,
        data_dir=data_dir
    )
    
    train_dataset, val_dataset, test_dataset = datasets
    
    # Extract SMILES and labels
    def extract_dataset(dataset):
        smiles = list(dataset.ids)
        labels = dataset.y.flatten()
        return {'smiles': smiles, 'labels': labels}
    
    splits = {
        'train': extract_dataset(train_dataset),
        'val': extract_dataset(val_dataset),
        'test': extract_dataset(test_dataset)
    }
    
    print(f"  Train: {len(splits['train']['smiles'])} molecules")
    print(f"  Val:   {len(splits['val']['smiles'])} molecules")
    print(f"  Test:  {len(splits['test']['smiles'])} molecules")
    print(f"  Total: {sum(len(s['smiles']) for s in splits.values())} molecules")
    
    # Print statistics
    all_labels = np.concatenate([
        splits['train']['labels'],
        splits['val']['labels'],
        splits['test']['labels']
    ])
    print(f"  LogS range: [{all_labels.min():.2f}, {all_labels.max():.2f}]")
    print(f"  LogS mean: {all_labels.mean():.2f} ± {all_labels.std():.2f}")
    
    return splits


def get_esol_combined_train(splits):
    """
    Combine train and validation sets (common practice for final models).
    Normalizes labels using training set statistics.
    
    Args:
        splits: Dictionary from load_esol()
    
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},  # train + val combined, NORMALIZED
            'test': {'smiles': [...], 'labels': [...]}    # NORMALIZED using train stats
        }
    """
    combined_train_smiles = splits['train']['smiles'] + splits['val']['smiles']
    combined_train_labels = np.concatenate([
        splits['train']['labels'],
        splits['val']['labels']
    ])
    test_labels = splits['test']['labels']
    
    # Normalize using training set statistics
    mean = combined_train_labels.mean()
    std = combined_train_labels.std()
    
    combined_train_labels_norm = (combined_train_labels - mean) / std
    test_labels_norm = (test_labels - mean) / std
    
    print(f"  Normalization: mean={mean:.3f}, std={std:.3f}")
    
    return {
        'train': {
            'smiles': combined_train_smiles,
            'labels': combined_train_labels_norm
        },
        'test': {
            'smiles': splits['test']['smiles'],
            'labels': test_labels_norm
        }
    }


def load_esol_combined(splitter='scaffold', data_dir=None):
    """
    Load ESOL with train+val combined (ready for training).
    
    This is the most common usage pattern - train on train+val, test on test.
    
    Args:
        splitter: How to split the data (default: 'scaffold')
            'scaffold': Scaffold split (more realistic, harder, tests generalization)
            'random': Random split
            'stratified': Stratified split
            'index': Index split (chronological)
        data_dir: Directory to cache data (optional)
    
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]}
        }
    """
    splits = load_esol(splitter=splitter, data_dir=data_dir)
    return get_esol_combined_train(splits)


# Metadata for reference
ESOL_METADATA = {
    'name': 'ESOL (Delaney)',
    'task': 'regression',
    'target': 'aqueous solubility (logS)',
    'n_molecules': 1128,
    'units': 'log(mol/L)',
    'reference': 'Delaney, J.S. (2004). J. Chem. Inf. Comput. Sci.',
    'molnet_url': 'https://moleculenet.org/datasets-1',
    'sota': {
        'method': 'MolPROP (2024)',
        'rmse': 0.77,
        'reference': 'arXiv:2409.11772'
    },
    'baseline': {
        'method': 'Random Forest + ECFP',
        'rmse': 1.74
    }
}


def get_esol_info():
    """Print information about the ESOL dataset."""
    info = ESOL_METADATA
    print("="*70)
    print(f"Dataset: {info['name']}")
    print("="*70)
    print(f"Task:         {info['task']}")
    print(f"Target:       {info['target']}")
    print(f"# Molecules:  {info['n_molecules']}")
    print(f"Units:        {info['units']}")
    print(f"\nSOTA Performance:")
    print(f"  {info['sota']['method']}: {info['sota']['rmse']} RMSE")
    print(f"  Reference: {info['sota']['reference']}")
    print(f"\nBaseline:")
    print(f"  {info['baseline']['method']}: {info['baseline']['rmse']} RMSE")
    print(f"\nReference: {info['reference']}")
    print(f"MolNet: {info['molnet_url']}")
    print("="*70)