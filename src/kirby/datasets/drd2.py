"""
DRD2 (Dopamine D2 Receptor) Ki Dataset Loader

Extracts DRD2 binding affinity (Ki) data from ChEMBL following Greg Landrum's
protocol for Ki data curation.

DRD2 is one of the most heavily studied drug targets with abundant Ki data
(antipsychotics, Parkinson's drugs).

Data curation follows Greg Landrum's protocol (RDKit blog, June 2023):
  "For Ki data, R² is 0.88 and Spearman's R is 0.94. Less than 10% of points
   differ by more than 0.3 log units... This is hugely better agreement than
   what we saw with the IC50 results."

Quality filters applied:
  - standard_type = 'Ki' (more reproducible than IC50)
  - standard_units = 'nM'
  - standard_relation = '=' (exact values only)
  - data_validity_comment IS NULL
  - Aggregate duplicates using geometric mean
  - Remove compounds with >1 log unit spread across measurements

Target: DRD2 (Dopamine D2 receptor)
  - ChEMBL ID: CHEMBL217
  - UniProt: P14416
  - Clinical relevance: Antipsychotics, Parkinson's disease

Task: Regression (pKi prediction)

References:
- Landrum, G. (2023). "The comparative safety of combining data from Ki assays"
  https://greglandrum.github.io/rdkit-blog/posts/2023-06-17-overlapping-Ki-assays1.html
- Landrum et al. (2024). JCIM. https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049
"""

import numpy as np
import pandas as pd
import os
from typing import Optional, Dict


# DRD2 target ID in ChEMBL
DRD2_CHEMBL_ID = 'CHEMBL217'


def load_drd2_chembl(data_dir: str = 'data/DRD2',
                     random_seed: int = 42,
                     max_spread_log_units: float = 1.0,
                     min_measurements: int = 1) -> Dict:
    """
    Load DRD2 Ki data from ChEMBL following Landrum's curation protocol.
    
    Args:
        data_dir: Directory to cache data
        random_seed: Random seed for reproducibility
        max_spread_log_units: Maximum allowed spread in pKi for duplicate
                              measurements (default: 1.0)
        min_measurements: Minimum measurements required per compound
        
    Returns:
        dict: {
            'smiles': List of SMILES strings,
            'labels': pKi values (higher = more potent),
            'ki_nM': Ki values in nM,
            'n_measurements': Number of measurements per compound,
            'spread': Spread in pKi across measurements,
            'source': 'chembl_drd2_ki'
        }
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "ChEMBL client required: pip install chembl_webresource_client"
        )
    
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, 'drd2_ki_chembl_raw.csv')
    
    if os.path.exists(cache_file):
        print(f"Loading cached raw ChEMBL DRD2 Ki data from {cache_file}...")
        df_raw = pd.read_csv(cache_file)
    else:
        print("Downloading DRD2 Ki data from ChEMBL...")
        print("(This may take a few minutes)")
        print()
        print("Applying Landrum's quality filters:")
        print("  - standard_type = 'Ki'")
        print("  - standard_units = 'nM'")
        print("  - standard_relation = '='")
        print("  - data_validity_comment IS NULL")
        print()
        
        activity = new_client.activity
        
        activities = activity.filter(
            target_chembl_id=DRD2_CHEMBL_ID,
            standard_type='Ki',
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
                    'ki_nM': float(act['standard_value']),
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
    
    # Compute pKi from Ki (nM)
    # pKi = -log10(Ki_M) = -log10(Ki_nM * 1e-9) = 9 - log10(Ki_nM)
    df_raw['pKi'] = 9 - np.log10(df_raw['ki_nM'])
    
    # Aggregate by SMILES
    print("\nAggregating duplicate measurements...")
    print("  Using geometric mean of Ki values (arithmetic mean of pKi)")
    
    def aggregate_measurements(group):
        n = len(group)
        ki_values = group['ki_nM'].values
        pki_values = group['pKi'].values
        
        mean_pki = np.mean(pki_values)
        mean_ki = 10 ** (9 - mean_pki)
        spread = np.max(pki_values) - np.min(pki_values)
        
        return pd.Series({
            'smiles': group.name,
            'ki_nM': mean_ki,
            'pKi': mean_pki,
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
        print(f"  Removed {n_removed} compounds with pKi spread > {max_spread_log_units}")
    
    print(f"\nFinal curated dataset: {len(df_agg)} compounds")
    
    print(f"\npKi range: [{df_agg['pKi'].min():.2f}, {df_agg['pKi'].max():.2f}]")
    print(f"pKi mean: {df_agg['pKi'].mean():.2f} ± {df_agg['pKi'].std():.2f}")
    print(f"Ki range: [{df_agg['ki_nM'].min():.2f}, {df_agg['ki_nM'].max():.1f}] nM")
    
    n_single = (df_agg['n_measurements'] == 1).sum()
    n_multi = (df_agg['n_measurements'] > 1).sum()
    print(f"\nMeasurement counts:")
    print(f"  Single measurement: {n_single} ({100*n_single/len(df_agg):.1f}%)")
    print(f"  Multiple measurements: {n_multi} ({100*n_multi/len(df_agg):.1f}%)")
    
    return {
        'smiles': df_agg['smiles'].tolist(),
        'labels': df_agg['pKi'].values,
        'ki_nM': df_agg['ki_nM'].values,
        'n_measurements': df_agg['n_measurements'].values,
        'spread': df_agg['spread'].values,
        'source': 'chembl_drd2_ki'
    }


def get_drd2_splits(data: Dict,
                    splitter: str = 'scaffold',
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    random_seed: int = 42) -> Dict:
    """
    Split DRD2 data into train/val/test sets.
    """
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
        mean_pki = split_data['labels'].mean()
        std_pki = split_data['labels'].std()
        print(f"  {split_name:5s}: {n:4d} molecules, pKi = {mean_pki:.2f} ± {std_pki:.2f}")
    
    return splits


def load_drd2(splitter: str = 'scaffold',
              data_dir: str = 'data/DRD2',
              random_seed: int = 42) -> Dict:
    """Load DRD2 Ki data with train/val/test splits."""
    data = load_drd2_chembl(data_dir=data_dir, random_seed=random_seed)
    return get_drd2_splits(data, splitter=splitter, random_seed=random_seed)


DRD2_METADATA = {
    'name': 'DRD2 Ki (ChEMBL)',
    'task': 'regression',
    'target': 'Dopamine D2 receptor',
    'target_chembl_id': DRD2_CHEMBL_ID,
    'endpoint': 'Ki',
    'units': 'pKi (-log10(Ki_M))',
    'clinical_relevance': 'Antipsychotics (haloperidol, risperidone), Parkinsons',
    'curation': {
        'protocol': 'Greg Landrum (2023)',
        'filters': [
            'standard_type = Ki',
            'standard_units = nM',
            'standard_relation = =',
            'data_validity_comment IS NULL'
        ],
        'aggregation': 'Geometric mean of Ki',
        'outlier_removal': 'Compounds with >1 log unit spread removed'
    }
}


def get_drd2_info():
    """Print information about the DRD2 dataset."""
    info = DRD2_METADATA
    print("=" * 70)
    print(f"Dataset: {info['name']}")
    print("=" * 70)
    print(f"Task:         {info['task']}")
    print(f"Target:       {info['target']}")
    print(f"ChEMBL ID:    {info['target_chembl_id']}")
    print(f"Endpoint:     {info['endpoint']}")
    print(f"Units:        {info['units']}")
    print(f"Clinical:     {info['clinical_relevance']}")
    print("=" * 70)


if __name__ == '__main__':
    get_drd2_info()
    print()
    data = load_drd2_chembl()