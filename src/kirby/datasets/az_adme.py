"""
AstraZeneca ADME Dataset Loaders

High-quality ADME datasets deposited by AstraZeneca to ChEMBL, available via TDC.
These datasets are ideal for noise robustness studies due to consistent experimental
protocols from a single industrial source.

Available Datasets:
1. Lipophilicity_AstraZeneca (4,200 compounds) - logD @ pH 7.4
2. PPBR_AZ (1,797 compounds) - Plasma Protein Binding Rate
3. Clearance_Microsome_AZ (1,102 compounds) - Microsomal Clearance
4. Clearance_Hepatocyte_AZ (1,020 compounds) - Hepatocyte Clearance

All datasets:
- Single source (AstraZeneca) with consistent experimental protocols
- Regression tasks with continuous endpoints
- Scaffold split recommended for realistic evaluation

References:
- TDC: https://tdcommons.ai/single_pred_tasks/adme/
- AstraZeneca (2016) ChEMBL deposition: Experimental in vitro DMPK and 
  physicochemical data on a set of publicly disclosed compounds
- Pat Walters recommends Lipophilicity_AstraZeneca for logD benchmarking
  (GitHub gist: LogD model based on 1.8M datapoints from ChEMBL)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Literal


# =============================================================================
# Dataset Metadata
# =============================================================================

AZ_DATASETS = {
    'lipophilicity': {
        'tdc_name': 'Lipophilicity_AstraZeneca',
        'full_name': 'Lipophilicity (AstraZeneca)',
        'task': 'regression',
        'target': 'logD @ pH 7.4',
        'description': 'Octanol-water distribution coefficient at physiological pH',
        'n_molecules': 4200,
        'units': 'log-ratio (dimensionless)',
        'typical_range': [-3, 6],
        'clinical_relevance': 'Drug absorption, distribution, membrane permeability',
        'metric': 'MAE',
        'reference': 'AstraZeneca ChEMBL deposition (2016)',
        'pat_walters_endorsed': True,
        'sota': {
            'method': 'RTlogD (transfer learning from RT data)',
            'mae': 0.467,
            'reference': 'J Cheminform (2023) - RTlogD paper'
        },
        'baseline': {
            'method': 'Random Forest + ECFP4',
            'mae': 0.65
        }
    },
    'ppbr': {
        'tdc_name': 'PPBR_AZ',
        'full_name': 'Plasma Protein Binding Rate (AstraZeneca)',
        'task': 'regression',
        'target': 'PPBR (%)',
        'description': 'Percentage of drug bound to plasma proteins in blood',
        'n_molecules': 1797,
        'units': '% bound',
        'typical_range': [0, 100],
        'clinical_relevance': 'Free drug concentration, efficacy, drug-drug interactions',
        'metric': 'MAE',
        'reference': 'AstraZeneca ChEMBL deposition (2016)',
        'pat_walters_endorsed': False,
        'sota': {
            'method': 'Chemprop ensemble',
            'mae': 7.8,
            'reference': 'TDC Leaderboard'
        },
        'baseline': {
            'method': 'Random Forest + ECFP4',
            'mae': 9.2
        }
    },
    'clearance_microsome': {
        'tdc_name': 'Clearance_Microsome_AZ',
        'full_name': 'Microsomal Clearance (AstraZeneca)',
        'task': 'regression',
        'target': 'Intrinsic clearance (CLint)',
        'description': 'Rate of drug metabolism by liver microsomes',
        'n_molecules': 1102,
        'units': 'mL/min/g liver',
        'typical_range': [0, 500],
        'clinical_relevance': 'Drug metabolism, half-life prediction, dosing',
        'metric': 'Spearman',  # TDC uses Spearman for clearance
        'reference': 'AstraZeneca ChEMBL deposition (2016)',
        'pat_walters_endorsed': False,
        'sota': {
            'method': 'ADMET-AI',
            'spearman': 0.62,
            'reference': 'Bioinformatics (2024)'
        },
        'baseline': {
            'method': 'Random Forest + ECFP4',
            'spearman': 0.52
        }
    },
    'clearance_hepatocyte': {
        'tdc_name': 'Clearance_Hepatocyte_AZ',
        'full_name': 'Hepatocyte Clearance (AstraZeneca)',
        'task': 'regression',
        'target': 'Intrinsic clearance (CLint)',
        'description': 'Rate of drug metabolism by hepatocytes',
        'n_molecules': 1020,
        'units': 'µL/min/10^6 cells',
        'typical_range': [0, 500],
        'clinical_relevance': 'Drug metabolism, more physiologically relevant than microsomes',
        'metric': 'Spearman',
        'reference': 'AstraZeneca ChEMBL deposition (2016)',
        'pat_walters_endorsed': False,
        'sota': {
            'method': 'ADMET-AI',
            'spearman': 0.54,
            'reference': 'Bioinformatics (2024)'
        },
        'baseline': {
            'method': 'Random Forest + ECFP4',
            'spearman': 0.45
        }
    }
}

DatasetName = Literal['lipophilicity', 'ppbr', 'clearance_microsome', 'clearance_hepatocyte']


# =============================================================================
# Core Loading Functions
# =============================================================================

def load_az_adme(dataset: DatasetName = 'lipophilicity',
                 splitter: str = 'scaffold',
                 n_samples: Optional[int] = None,
                 random_seed: int = 42) -> Dict:
    """
    Load AstraZeneca ADME dataset from TDC.
    
    Args:
        dataset: Which dataset to load:
            'lipophilicity' - logD @ pH 7.4 (4,200 compounds)
            'ppbr' - Plasma Protein Binding Rate (1,797 compounds)
            'clearance_microsome' - Microsomal clearance (1,102 compounds)
            'clearance_hepatocyte' - Hepatocyte clearance (1,020 compounds)
        splitter: How to split the data
            'scaffold': Scaffold split (recommended, tests generalization)
            'random': Random split
        n_samples: Number of samples to use (default: None = all)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'val': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]},
            'metadata': {...}
        }
    """
    try:
        from tdc.single_pred import ADME
    except ImportError:
        raise ImportError("TDC required: pip install PyTDC")
    
    if dataset not in AZ_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(AZ_DATASETS.keys())}")
    
    meta = AZ_DATASETS[dataset]
    tdc_name = meta['tdc_name']
    
    print(f"Loading {meta['full_name']} from TDC...")
    print(f"  TDC name: {tdc_name}")
    print(f"  Target: {meta['target']}")
    print(f"  Units: {meta['units']}")
    
    # Load from TDC
    data = ADME(name=tdc_name)
    
    # Get split using TDC's built-in splitter
    if splitter == 'scaffold':
        split = data.get_split(method='scaffold', seed=random_seed)
    elif splitter == 'random':
        split = data.get_split(method='random', seed=random_seed)
    else:
        raise ValueError(f"Unknown splitter: {splitter}. Use 'scaffold' or 'random'")
    
    # TDC returns DataFrames with 'Drug' and 'Y' columns
    def extract_split(df, name):
        if n_samples is not None and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_seed)
        
        smiles = df['Drug'].tolist()
        labels = df['Y'].values.astype(np.float32)
        
        print(f"  {name:5s}: {len(smiles):5d} molecules, "
              f"Y range: [{labels.min():.2f}, {labels.max():.2f}], "
              f"mean: {labels.mean():.2f} ± {labels.std():.2f}")
        
        return {'smiles': smiles, 'labels': labels}
    
    splits = {
        'train': extract_split(split['train'], 'Train'),
        'val': extract_split(split['valid'], 'Val'),
        'test': extract_split(split['test'], 'Test'),
        'metadata': meta
    }
    
    total = sum(len(s['smiles']) for s in [splits['train'], splits['val'], splits['test']])
    print(f"  Total: {total} molecules")
    
    return splits


def load_az_adme_combined(dataset: DatasetName = 'lipophilicity',
                          splitter: str = 'scaffold',
                          normalize: bool = True,
                          random_seed: int = 42) -> Dict:
    """
    Load AstraZeneca ADME dataset with train+val combined.
    
    This is the most common usage pattern - train on train+val, test on test.
    Optionally normalizes labels using training set statistics.
    
    Args:
        dataset: Which dataset to load (see load_az_adme for options)
        splitter: Split method ('scaffold' or 'random')
        normalize: If True, normalize labels using training set statistics
        random_seed: Random seed
        
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]},
            'metadata': {...},
            'normalization': {'mean': ..., 'std': ...} if normalize=True
        }
    """
    splits = load_az_adme(dataset=dataset, splitter=splitter, random_seed=random_seed)
    
    # Combine train and val
    combined_smiles = splits['train']['smiles'] + splits['val']['smiles']
    combined_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
    test_labels = splits['test']['labels']
    
    result = {
        'metadata': splits['metadata']
    }
    
    if normalize:
        mean = combined_labels.mean()
        std = combined_labels.std()
        
        combined_labels_norm = (combined_labels - mean) / std
        test_labels_norm = (test_labels - mean) / std
        
        print(f"  Normalization: mean={mean:.3f}, std={std:.3f}")
        
        result['train'] = {'smiles': combined_smiles, 'labels': combined_labels_norm}
        result['test'] = {'smiles': splits['test']['smiles'], 'labels': test_labels_norm}
        result['normalization'] = {'mean': mean, 'std': std}
    else:
        result['train'] = {'smiles': combined_smiles, 'labels': combined_labels}
        result['test'] = {'smiles': splits['test']['smiles'], 'labels': test_labels}
    
    print(f"  Combined train: {len(result['train']['smiles'])} molecules")
    print(f"  Test: {len(result['test']['smiles'])} molecules")
    
    return result


# =============================================================================
# Convenience Functions for Each Dataset
# =============================================================================

def load_lipophilicity(splitter: str = 'scaffold', 
                       n_samples: Optional[int] = None,
                       random_seed: int = 42) -> Dict:
    """
    Load Lipophilicity (AstraZeneca) dataset - logD @ pH 7.4.
    
    4,200 compounds with measured octanol-water distribution coefficients.
    Pat Walters-endorsed benchmark for lipophilicity prediction.
    
    Args:
        splitter: 'scaffold' (recommended) or 'random'
        n_samples: Number of samples (default: all)
        random_seed: Random seed
        
    Returns:
        dict with train/val/test splits
    """
    return load_az_adme('lipophilicity', splitter=splitter, 
                        n_samples=n_samples, random_seed=random_seed)


def load_lipophilicity_combined(splitter: str = 'scaffold',
                                normalize: bool = True,
                                random_seed: int = 42) -> Dict:
    """Load Lipophilicity with train+val combined (ready for training)."""
    return load_az_adme_combined('lipophilicity', splitter=splitter,
                                  normalize=normalize, random_seed=random_seed)


def load_ppbr(splitter: str = 'scaffold',
              n_samples: Optional[int] = None,
              random_seed: int = 42) -> Dict:
    """
    Load PPBR (AstraZeneca) dataset - Plasma Protein Binding Rate.
    
    1,797 compounds with measured plasma protein binding percentages.
    Important for understanding free drug concentration.
    
    Args:
        splitter: 'scaffold' (recommended) or 'random'
        n_samples: Number of samples (default: all)
        random_seed: Random seed
        
    Returns:
        dict with train/val/test splits
    """
    return load_az_adme('ppbr', splitter=splitter,
                        n_samples=n_samples, random_seed=random_seed)


def load_ppbr_combined(splitter: str = 'scaffold',
                       normalize: bool = True,
                       random_seed: int = 42) -> Dict:
    """Load PPBR with train+val combined (ready for training)."""
    return load_az_adme_combined('ppbr', splitter=splitter,
                                  normalize=normalize, random_seed=random_seed)


def load_clearance_microsome(splitter: str = 'scaffold',
                             n_samples: Optional[int] = None,
                             random_seed: int = 42) -> Dict:
    """
    Load Microsomal Clearance (AstraZeneca) dataset.
    
    1,102 compounds with measured intrinsic clearance from liver microsomes.
    Key endpoint for predicting drug metabolism and half-life.
    
    Args:
        splitter: 'scaffold' (recommended) or 'random'
        n_samples: Number of samples (default: all)
        random_seed: Random seed
        
    Returns:
        dict with train/val/test splits
    """
    return load_az_adme('clearance_microsome', splitter=splitter,
                        n_samples=n_samples, random_seed=random_seed)


def load_clearance_microsome_combined(splitter: str = 'scaffold',
                                      normalize: bool = True,
                                      random_seed: int = 42) -> Dict:
    """Load Microsomal Clearance with train+val combined."""
    return load_az_adme_combined('clearance_microsome', splitter=splitter,
                                  normalize=normalize, random_seed=random_seed)


def load_clearance_hepatocyte(splitter: str = 'scaffold',
                              n_samples: Optional[int] = None,
                              random_seed: int = 42) -> Dict:
    """
    Load Hepatocyte Clearance (AstraZeneca) dataset.
    
    1,020 compounds with measured intrinsic clearance from hepatocytes.
    More physiologically relevant than microsomal clearance.
    
    Args:
        splitter: 'scaffold' (recommended) or 'random'
        n_samples: Number of samples (default: all)
        random_seed: Random seed
        
    Returns:
        dict with train/val/test splits
    """
    return load_az_adme('clearance_hepatocyte', splitter=splitter,
                        n_samples=n_samples, random_seed=random_seed)


def load_clearance_hepatocyte_combined(splitter: str = 'scaffold',
                                       normalize: bool = True,
                                       random_seed: int = 42) -> Dict:
    """Load Hepatocyte Clearance with train+val combined."""
    return load_az_adme_combined('clearance_hepatocyte', splitter=splitter,
                                  normalize=normalize, random_seed=random_seed)


# =============================================================================
# Information Functions
# =============================================================================

def get_az_adme_info(dataset: Optional[DatasetName] = None):
    """
    Print information about AstraZeneca ADME dataset(s).
    
    Args:
        dataset: Specific dataset name, or None to show all
    """
    if dataset is not None:
        datasets = [dataset]
    else:
        datasets = list(AZ_DATASETS.keys())
    
    print("=" * 75)
    print("AstraZeneca ADME Datasets (via TDC)")
    print("=" * 75)
    print("\nAll datasets from single industrial source with consistent protocols.")
    print("Ideal for noise robustness studies and fair method comparison.\n")
    
    for ds in datasets:
        meta = AZ_DATASETS[ds]
        print("-" * 75)
        print(f"Dataset: {meta['full_name']}")
        print("-" * 75)
        print(f"  TDC Name:      {meta['tdc_name']}")
        print(f"  Task:          {meta['task']}")
        print(f"  Target:        {meta['target']}")
        print(f"  # Molecules:   {meta['n_molecules']}")
        print(f"  Units:         {meta['units']}")
        print(f"  Typical Range: {meta['typical_range']}")
        print(f"  Metric:        {meta['metric']}")
        print(f"  Clinical:      {meta['clinical_relevance']}")
        
        if meta.get('pat_walters_endorsed'):
            print(f"  ⭐ Pat Walters endorsed for benchmarking")
        
        sota = meta['sota']
        print(f"\n  SOTA: {sota['method']}")
        if 'mae' in sota:
            print(f"        MAE = {sota['mae']}")
        if 'spearman' in sota:
            print(f"        Spearman = {sota['spearman']}")
        print(f"        Ref: {sota['reference']}")
        
        baseline = meta['baseline']
        print(f"\n  Baseline: {baseline['method']}")
        if 'mae' in baseline:
            print(f"            MAE = {baseline['mae']}")
        if 'spearman' in baseline:
            print(f"            Spearman = {baseline['spearman']}")
        print()
    
    print("=" * 75)
    print("Reference: AstraZeneca ChEMBL deposition (2016)")
    print("           'Experimental in vitro DMPK and physicochemical data'")
    print("=" * 75)


def list_az_datasets():
    """List available AstraZeneca ADME datasets."""
    print("\nAvailable AstraZeneca ADME Datasets:")
    print("-" * 50)
    for name, meta in AZ_DATASETS.items():
        endorsed = " ⭐" if meta.get('pat_walters_endorsed') else ""
        print(f"  {name:22s} | {meta['n_molecules']:5d} compounds | {meta['target']}{endorsed}")
    print("-" * 50)
    print("Use: load_az_adme(dataset='<name>')")
    print("  or: load_lipophilicity(), load_ppbr(), etc.")


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    # Show available datasets
    list_az_datasets()
    print()
    
    # Show detailed info
    get_az_adme_info('lipophilicity')
    
    # Test loading
    print("\n" + "=" * 75)
    print("Testing data loading...")
    print("=" * 75)
    
    # Load lipophilicity (Pat Walters endorsed)
    lipo = load_lipophilicity(splitter='scaffold')
    
    print("\n" + "-" * 75)
    
    # Load PPBR
    ppbr = load_ppbr(splitter='scaffold')
    
    print("\n" + "-" * 75)
    
    # Load clearance
    clint = load_clearance_microsome(splitter='scaffold')
    
    print("\n" + "=" * 75)
    print("All datasets loaded successfully!")
    print("=" * 75)