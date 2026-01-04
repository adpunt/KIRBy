"""
hERG Cardiotoxicity Dataset Loader

hERG (human Ether-à-go-go-Related Gene) channel blockade is a major cause of
drug-induced cardiotoxicity and QT interval prolongation. This module loads
hERG bioactivity data for binary classification (blocker vs. non-blocker).

Data Sources:
1. FLuID federated dataset from Lhasa Limited (.pkl or .sdf files)
   - hERG_lhasa_training.pkl / hERG_lhasa_training.sdf
   - hERG_lhasa_test.pkl / hERG_lhasa_test.sdf
   - FLuID_small.sdf
2. TDC (Therapeutics Data Commons) - Tox21 hERG dataset (~10K molecules)
3. ChEMBL hERG bioactivity data (~30K+ molecules)

Task: Binary classification
- Positive class: hERG blocker (IC50 ≤ 10 μM)
- Negative class: Non-blocker (IC50 > 10 μM)

References:
- FLuID: https://github.com/LhasaLimited/FLuID_POC
- TDC: https://tdcommons.ai/
- ChEMBL: https://www.ebi.ac.uk/chembl/
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, List


def load_herg_chembl(n_samples: Optional[int] = None,
                     ic50_threshold: float = 10.0,
                     data_dir: str = 'data/hERG',
                     random_seed: int = 42) -> Dict:
    """
    Load hERG bioactivity data from ChEMBL.
    
    Downloads and processes hERG data from ChEMBL database with IC50 values.
    
    Args:
        n_samples: Number of samples to use (default: None = all)
        ic50_threshold: IC50 threshold in μM for blocker classification (default: 10.0)
        data_dir: Directory to cache data
        random_seed: Random seed for sampling
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': Binary labels (1 = blocker, 0 = non-blocker),
            'ic50': IC50 values in μM,
            'source': 'chembl'
        }
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError("ChEMBL client required: pip install chembl_webresource_client")
    
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, 'herg_chembl_cache.csv')
    
    # Load from cache if available
    if os.path.exists(cache_file):
        print(f"Loading cached ChEMBL hERG data from {cache_file}...")
        df = pd.read_csv(cache_file)
    else:
        print("Downloading hERG data from ChEMBL (this may take a few minutes)...")
        
        # Query ChEMBL for hERG target (CHEMBL240)
        activity = new_client.activity
        
        # hERG target ID in ChEMBL
        herg_target_id = 'CHEMBL240'
        
        # Get all activities for hERG
        activities = activity.filter(
            target_chembl_id=herg_target_id,
            standard_type='IC50',
            standard_units='nM',
            standard_relation='='
        ).only(['molecule_chembl_id', 'canonical_smiles', 'standard_value'])
        
        data_list = []
        for act in activities:
            if act['canonical_smiles'] and act['standard_value']:
                data_list.append({
                    'smiles': act['canonical_smiles'],
                    'ic50_nM': float(act['standard_value'])
                })
        
        df = pd.DataFrame(data_list)
        
        # Remove duplicates (keep median IC50)
        df = df.groupby('smiles').agg({'ic50_nM': 'median'}).reset_index()
        
        # Convert nM to μM
        df['ic50_uM'] = df['ic50_nM'] / 1000.0
        
        # Save cache
        df.to_csv(cache_file, index=False)
        print(f"Cached {len(df)} compounds to {cache_file}")
    
    # Binarize using IC50 threshold
    df['label'] = (df['ic50_uM'] <= ic50_threshold).astype(int)
    
    # Sample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed)
    
    print(f"\nChEMBL hERG dataset:")
    print(f"  Total: {len(df)} molecules")
    print(f"  Blockers (IC50 ≤ {ic50_threshold} μM): {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"  Non-blockers: {(1-df['label']).sum()} ({100*(1-df['label'].mean()):.1f}%)")
    print(f"  IC50 range: [{df['ic50_uM'].min():.3f}, {df['ic50_uM'].max():.1f}] μM")
    
    return {
        'smiles': df['smiles'].tolist(),
        'labels': df['label'].values,
        'ic50': df['ic50_uM'].values,
        'source': 'chembl'
    }


def load_herg_tdc(n_samples: Optional[int] = None,
                  data_dir: str = 'data/hERG',
                  random_seed: int = 42) -> Dict:
    """
    Load hERG data from Therapeutics Data Commons (TDC).
    
    Uses the Tox21 hERG dataset from TDC (pre-processed and curated).
    
    Args:
        n_samples: Number of samples to use (default: None = all)
        data_dir: Directory to cache data
        random_seed: Random seed
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': Binary labels (1 = blocker, 0 = non-blocker),
            'source': 'tdc'
        }
    """
    try:
        from tdc.single_pred import Tox
    except ImportError:
        raise ImportError("TDC required: pip install PyTDC")
    
    print("Loading hERG data from TDC...")
    
    # Load hERG central dataset from TDC
    data = Tox(name='hERG', label_name='hERG_at_10uM')
    df = data.get_data()
    
    # TDC uses 'Drug' for SMILES and 'Y' for labels
    df = df.rename(columns={'Drug': 'smiles', 'Y': 'label'})
    
    # Sample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed)
    
    print(f"\nTDC hERG dataset:")
    print(f"  Total: {len(df)} molecules")
    print(f"  Blockers: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"  Non-blockers: {(1-df['label']).sum()} ({100*(1-df['label'].mean()):.1f}%)")
    
    return {
        'smiles': df['smiles'].tolist(),
        'labels': df['label'].values,
        'source': 'tdc'
    }


def load_herg_fluid(data_dir: str = 'data/fluid',
                    use_test: bool = False) -> Dict:
    """
    Load hERG federated data from FLuID (Lhasa Limited).
    
    Uses train/test split from FLuID pickle files.
    Run `bash peek_fluid.sh` to see the actual file structure.
    
    Args:
        data_dir: Directory containing FLuID data
        use_test: If True, load test set; if False, load training set
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': Binary labels,
            'source': 'fluid'
        }
    """
    import pickle
    from rdkit import Chem
    
    # Determine which file to load
    if use_test:
        pkl_file = os.path.join(data_dir, 'hERG_lhasa_test.pkl')
        dataset_type = "test"
    else:
        pkl_file = os.path.join(data_dir, 'hERG_lhasa_training.pkl')
        dataset_type = "training"
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(
            f"FLuID data not found at {pkl_file}\n"
            f"Expected: hERG_lhasa_training.pkl or hERG_lhasa_test.pkl"
        )
    
    print(f"Loading FLuID hERG {dataset_type} data from {pkl_file}...")
    
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    
    # Convert RDKit Mol objects to SMILES
    smiles = [Chem.MolToSmiles(mol) for mol in df['MOLECULE']]
    
    # Convert ACTIVITY ("Active"/"Inactive") to binary (1/0)
    labels = (df['ACTIVITY'] == 'Active').astype(int).values
    
    print(f"\nFLuID hERG {dataset_type} dataset:")
    print(f"  Total: {len(smiles)} molecules")
    print(f"  Blockers (Active): {labels.sum()} ({100*labels.mean():.1f}%)")
    print(f"  Non-blockers (Inactive): {(1-labels).sum()} ({100*(1-labels.mean()):.1f}%)")
    
    return {
        'smiles': smiles,
        'labels': labels,
        'source': 'fluid'
    }


def load_herg(source: str = 'fluid',
              n_samples: Optional[int] = None,
              ic50_threshold: float = 10.0,
              data_dir: str = 'data',
              random_seed: int = 42,
              use_test: bool = False) -> Dict:
    """
    Load hERG cardiotoxicity data from specified source.
    
    Args:
        source: Data source ('tdc', 'chembl', or 'fluid')
        n_samples: Number of samples to use
        ic50_threshold: IC50 threshold for ChEMBL (μM)
        data_dir: Data directory
        random_seed: Random seed
        use_test: For FLuID only - load test set instead of training set
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': Binary labels (1 = blocker, 0 = non-blocker),
            'source': Source name
        }
    """
    if source == 'tdc':
        return load_herg_tdc(n_samples=n_samples, data_dir=os.path.join(data_dir, 'hERG'), random_seed=random_seed)
    elif source == 'chembl':
        return load_herg_chembl(n_samples=n_samples, ic50_threshold=ic50_threshold,
                                data_dir=os.path.join(data_dir, 'hERG'), random_seed=random_seed)
    elif source == 'fluid':
        return load_herg_fluid(data_dir=os.path.join(data_dir, 'fluid'), use_test=use_test)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'tdc', 'chembl', or 'fluid'")


def get_herg_splits(data: Dict,
                    splitter: str = 'scaffold',
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    random_seed: int = 42) -> Dict:
    """
    Split hERG data into train/val/test sets.
    
    Args:
        data: Dictionary from load_herg()
        splitter: Split method ('scaffold', 'random', 'stratified')
        train_ratio: Training fraction
        val_ratio: Validation fraction  
        random_seed: Random seed
        
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'val': {...},
            'test': {...}
        }
    """
    smiles = data['smiles']
    labels = data['labels']
    n_samples = len(smiles)
    
    if splitter == 'scaffold':
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError("Scaffold split requires DeepChem: pip install deepchem")
        
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
        
        # Extract indices
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
        from sklearn.model_selection import train_test_split
        
        print(f"Splitting {n_samples} molecules using stratified split...")
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(n_samples),
            train_size=train_ratio,
            random_state=random_seed,
            stratify=labels
        )
        
        # Second split: val vs test
        val_size = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_seed,
            stratify=labels[temp_idx]
        )
    
    else:
        raise ValueError(f"Unknown splitter: {splitter}")
    
    # Create splits
    splits = {
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
    
    # Print split statistics
    for split_name, split_data in splits.items():
        n = len(split_data['smiles'])
        n_blockers = split_data['labels'].sum()
        print(f"  {split_name:5s}: {n:5d} molecules, {n_blockers:4d} blockers ({100*n_blockers/n:.1f}%)")
    
    return splits


# Metadata
HERG_METADATA = {
    'name': 'hERG Cardiotoxicity',
    'task': 'binary_classification',
    'target': 'hERG channel blockade',
    'positive_class': 'blocker (IC50 ≤ 10 μM)',
    'negative_class': 'non-blocker (IC50 > 10 μM)',
    'clinical_relevance': 'QT interval prolongation, Torsades de Pointes',
    'references': [
        'ChEMBL: https://www.ebi.ac.uk/chembl/',
        'TDC: https://tdcommons.ai/',
        'Sato et al. (2018) Scientific Reports 8:12220'
    ],
    'sota': {
        'method': 'XGBoost + Deep Learning ensemble',
        'accuracy': 0.92,
        'auc': 0.96,
        'reference': 'Various 2020-2025 publications'
    }
}


def get_herg_info():
    """Print information about the hERG dataset."""
    info = HERG_METADATA
    print("="*70)
    print(f"Dataset: {info['name']}")
    print("="*70)
    print(f"Task:         {info['task']}")
    print(f"Target:       {info['target']}")
    print(f"Positive:     {info['positive_class']}")
    print(f"Negative:     {info['negative_class']}")
    print(f"Clinical:     {info['clinical_relevance']}")
    print(f"\nSOTA Performance:")
    print(f"  {info['sota']['method']}: ACC={info['sota']['accuracy']}, AUC={info['sota']['auc']}")
    print(f"\nReferences:")
    for ref in info['references']:
        print(f"  {ref}")
    print("="*70)