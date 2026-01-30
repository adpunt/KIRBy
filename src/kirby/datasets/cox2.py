"""
COX-2 (Cyclooxygenase-2) IC50 Dataset Loader

Extracts COX-2 binding affinity (IC50) data from ChEMBL.

Pat Walters uses COX-2 IC50 in his QSAR tutorials (github.com/PatWalters/qsar):
  - COX-2_train_IC50_uM.smi
  - COX-2_train_pIC50.smi
  - COX-2_test.smi

Note on data quality (Landrum 2023):
  IC50 data is LESS reproducible than Ki. Landrum showed R²=0.51 for IC50 
  (vs R²=0.88 for Ki), with 65% of IC50 measurements differing by >0.3 log 
  units and 27% differing by >1 log unit.
  
  However, COX-2 has abundant IC50 data (~4000+ compounds) while Ki data 
  is sparse (~26 compounds), making IC50 the practical choice.

Quality filters applied:
  - standard_type = 'IC50'
  - standard_units = 'nM'
  - standard_relation = '=' (exact values only)
  - data_validity_comment IS NULL
  - Aggregate duplicates using geometric mean
  - Remove compounds with >1 log unit spread across measurements

Target: COX-2 (Cyclooxygenase-2, PTGS2)
  - ChEMBL ID: CHEMBL230
  - UniProt: P35354
  - Clinical relevance: NSAIDs, COX-2 selective inhibitors (celecoxib)

Task: Regression (pIC50 prediction)

References:
- Walters, P. GitHub QSAR tutorial: https://github.com/PatWalters/qsar
- Landrum, G. (2023). RDKit Blog on IC50 variability
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict


COX2_CHEMBL_ID = 'CHEMBL230'


def load_cox2_chembl(data_dir: str = 'data/COX2',
                     random_seed: int = 42,
                     max_spread_log_units: float = 1.0,
                     min_measurements: int = 1) -> Dict:
    """
    Load COX-2 IC50 data from ChEMBL.
    
    Args:
        data_dir: Directory to cache data
        random_seed: Random seed for reproducibility
        max_spread_log_units: Maximum allowed spread in pIC50 for duplicates
        min_measurements: Minimum measurements required per compound
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': pIC50 values (higher = more potent),
            'ic50_nM': IC50 values in nM,
            'n_measurements': Number of measurements per compound,
            'spread': Spread in pIC50 across measurements,
            'source': 'chembl_cox2_ic50'
        }
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client required: pip install chembl_webresource_client"
        )
    
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, 'cox2_ic50_chembl_raw.csv')
    
    if os.path.exists(cache_file):
        print(f"Loading cached raw ChEMBL COX-2 IC50 data from {cache_file}...")
        df_raw = pd.read_csv(cache_file)
    else:
        print("Downloading COX-2 IC50 data from ChEMBL...")
        print("(This may take a few minutes)")
        print()
        print("Quality filters:")
        print("  - standard_type = 'IC50'")
        print("  - standard_units = 'nM'")
        print("  - standard_relation = '='")
        print("  - data_validity_comment IS NULL")
        print()
        
        activity = new_client.activity
        
        activities = activity.filter(
            target_chembl_id=COX2_CHEMBL_ID,
            standard_type='IC50',
            standard_units='nM',
            standard_relation='=',
            data_validity_comment__isnull=True
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_value',
            'pchembl_value',
            'assay_chembl_id',
            'document_chembl_id'
        ])
        
        data_list = []
        for act in activities:
            if act['canonical_smiles'] and act['standard_value']:
                data_list.append({
                    'chembl_id': act['molecule_chembl_id'],
                    'smiles': act['canonical_smiles'],
                    'ic50_nM': float(act['standard_value']),
                    'pchembl': float(act['pchembl_value']) if act['pchembl_value'] else None,
                    'assay_id': act['assay_chembl_id'],
                    'doc_id': act['document_chembl_id']
                })
        
        df_raw = pd.DataFrame(data_list)
        df_raw.to_csv(cache_file, index=False)
        print(f"Cached {len(df_raw)} raw measurements to {cache_file}")
    
    print(f"\nRaw data: {len(df_raw)} measurements")
    
    # Validate SMILES with RDKit
    print("Validating and canonicalizing SMILES...")
    try:
        from rdkit import Chem
        
        valid_smiles = []
        canonical_smiles = []
        for smi in df_raw['smiles']:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(True)
                canonical_smiles.append(Chem.MolToSmiles(mol))
            else:
                valid_smiles.append(False)
                canonical_smiles.append(None)
        
        df_raw['valid'] = valid_smiles
        df_raw['canonical_smiles'] = canonical_smiles
        
        n_invalid = (~df_raw['valid']).sum()
        if n_invalid > 0:
            print(f"  Removed {n_invalid} invalid SMILES")
        
        df_raw = df_raw[df_raw['valid']].copy()
        df_raw['smiles'] = df_raw['canonical_smiles']
        df_raw = df_raw.drop(columns=['valid', 'canonical_smiles'])
        
    except ImportError:
        print("  Warning: RDKit not available, skipping SMILES validation")
    
    print(f"After SMILES validation: {len(df_raw)} measurements")
    
    # Compute pIC50 from IC50 (nM)
    # pIC50 = -log10(IC50_M) = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
    df_raw['pIC50'] = 9 - np.log10(df_raw['ic50_nM'])
    
    # Aggregate by SMILES
    print("\nAggregating duplicate measurements...")
    print("  Using geometric mean of IC50 values (arithmetic mean of pIC50)")
    
    def aggregate_measurements(group):
        n = len(group)
        ic50_values = group['ic50_nM'].values
        pic50_values = group['pIC50'].values
        
        mean_pic50 = np.mean(pic50_values)
        mean_ic50 = 10 ** (9 - mean_pic50)
        spread = np.max(pic50_values) - np.min(pic50_values)
        
        return pd.Series({
            'smiles': group.name,
            'ic50_nM': mean_ic50,
            'pIC50': mean_pic50,
            'n_measurements': n,
            'spread': spread,
            'assays': ','.join(group['assay_id'].unique())
        })
    
    df_agg = df_raw.groupby('smiles').apply(
        aggregate_measurements, include_groups=False
    ).reset_index(drop=True)
    
    n_compounds_raw = len(df_agg)
    print(f"  {n_compounds_raw} unique compounds")
    
    if min_measurements > 1:
        df_agg = df_agg[df_agg['n_measurements'] >= min_measurements]
        print(f"  After min_measurements >= {min_measurements}: {len(df_agg)} compounds")
    
    n_before_spread = len(df_agg)
    df_agg = df_agg[df_agg['spread'] <= max_spread_log_units]
    n_removed = n_before_spread - len(df_agg)
    if n_removed > 0:
        print(f"  Removed {n_removed} compounds with pIC50 spread > {max_spread_log_units}")
    
    print(f"\nFinal curated dataset: {len(df_agg)} compounds")
    
    print(f"\npIC50 range: [{df_agg['pIC50'].min():.2f}, {df_agg['pIC50'].max():.2f}]")
    print(f"pIC50 mean: {df_agg['pIC50'].mean():.2f} ± {df_agg['pIC50'].std():.2f}")
    print(f"IC50 range: [{df_agg['ic50_nM'].min():.2f}, {df_agg['ic50_nM'].max():.1f}] nM")
    
    n_single = (df_agg['n_measurements'] == 1).sum()
    n_multi = (df_agg['n_measurements'] > 1).sum()
    print(f"\nMeasurement counts:")
    print(f"  Single measurement: {n_single} ({100*n_single/len(df_agg):.1f}%)")
    print(f"  Multiple measurements: {n_multi} ({100*n_multi/len(df_agg):.1f}%)")
    
    if n_multi > 0:
        multi_df = df_agg[df_agg['n_measurements'] > 1]
        print(f"  Mean spread (multi): {multi_df['spread'].mean():.3f} log units")
    
    return {
        'smiles': df_agg['smiles'].tolist(),
        'labels': df_agg['pIC50'].values,
        'ic50_nM': df_agg['ic50_nM'].values,
        'n_measurements': df_agg['n_measurements'].values,
        'spread': df_agg['spread'].values,
        'source': 'chembl_cox2_ic50'
    }


def get_cox2_splits(data: Dict,
                    splitter: str = 'scaffold',
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    random_seed: int = 42) -> Dict:
    """Split COX-2 data into train/val/test sets."""
    smiles = data['smiles']
    labels = data['labels']
    n_samples = len(smiles)
    
    if splitter == 'scaffold':
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError(
                "Scaffold split requires DeepChem: pip install deepchem"
            )
        
        print(f"Splitting {n_samples} molecules using scaffold split...")
        
        dataset = dc.data.NumpyDataset(
            X=np.zeros((n_samples, 1)),
            y=labels.reshape(-1, 1),
            ids=smiles
        )
        
        splitter_obj = dc.splits.ScaffoldSplitter()
        train_dataset, val_dataset, test_dataset = splitter_obj.train_valid_test_split(
            dataset,
            frac_train=train_ratio,
            frac_valid=val_ratio,
            frac_test=1.0 - train_ratio - val_ratio,
            seed=random_seed
        )
        
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
        
    else:
        raise ValueError(f"Unknown splitter: {splitter}")
    
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
    
    for split_name, split_data in splits.items():
        n = len(split_data['smiles'])
        mean_pic50 = split_data['labels'].mean()
        std_pic50 = split_data['labels'].std()
        print(f"  {split_name:5s}: {n:4d} molecules, pIC50 = {mean_pic50:.2f} ± {std_pic50:.2f}")
    
    return splits


def load_cox2(splitter: str = 'scaffold',
              data_dir: str = 'data/COX2',
              random_seed: int = 42) -> Dict:
    """Load COX-2 IC50 data with train/val/test splits."""
    data = load_cox2_chembl(data_dir=data_dir, random_seed=random_seed)
    return get_cox2_splits(data, splitter=splitter, random_seed=random_seed)


COX2_METADATA = {
    'name': 'COX-2 IC50 (ChEMBL)',
    'task': 'regression',
    'target': 'Cyclooxygenase-2 (PTGS2)',
    'target_chembl_id': COX2_CHEMBL_ID,
    'endpoint': 'IC50',
    'units': 'pIC50 (-log10(IC50_M))',
    'clinical_relevance': 'NSAIDs, COX-2 selective inhibitors (celecoxib)',
    'data_quality_note': 'IC50 less reproducible than Ki (Landrum: R²=0.51 vs 0.88)',
    'references': [
        'Walters QSAR tutorial: github.com/PatWalters/qsar'
    ]
}


def get_cox2_info():
    """Print information about the COX-2 dataset."""
    info = COX2_METADATA
    print("=" * 70)
    print(f"Dataset: {info['name']}")
    print("=" * 70)
    print(f"Task:         {info['task']}")
    print(f"Target:       {info['target']}")
    print(f"ChEMBL ID:    {info['target_chembl_id']}")
    print(f"Endpoint:     {info['endpoint']}")
    print(f"Units:        {info['units']}")
    print(f"Clinical:     {info['clinical_relevance']}")
    print(f"Note:         {info['data_quality_note']}")
    print("=" * 70)


if __name__ == '__main__':
    get_cox2_info()
    print()
    data = load_cox2_chembl()