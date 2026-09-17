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
from pathlib import Path
from typing import Optional, Dict, List


def _default_data_dir(subdir: str) -> str:
    """Default cache subdir under $KIRBY_CACHE_DIR (falls back to ~/.cache/kirby)."""
    root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return str(root / subdir)


def load_herg_chembl(n_samples: Optional[int] = None,
                     ic50_threshold: float = 10.0,
                     data_dir: Optional[str] = None,
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

    if data_dir is None:
        data_dir = _default_data_dir('hERG')
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
                  data_dir: Optional[str] = None,
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


def _resolve_fluid_dir(data_dir: Optional[str] = None) -> str:
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    env = os.environ.get('LHASA_DATA_DIR')
    if env and os.path.isdir(env):
        return env
    for candidate in (
        '/data/stat-cadd/scat9264/lhasa_data',
        '/Users/apunt/repos/dphil/data',
        'data/fluid',
    ):
        if os.path.isdir(candidate):
            return candidate
    return data_dir or 'data/fluid'


def load_herg_fluid(data_dir: Optional[str] = None,
                    use_test: bool = False) -> Dict:
    """Load hERG federated data from FLuID (Lhasa Limited)."""
    import pickle
    from rdkit import Chem

    resolved_dir = _resolve_fluid_dir(data_dir)
    fname = 'hERG_lhasa_test.pkl' if use_test else 'hERG_lhasa_training.pkl'
    dataset_type = "test" if use_test else "training"
    pkl_file = os.path.join(resolved_dir, fname)

    if not os.path.exists(pkl_file):
        raise FileNotFoundError(
            f"FLuID data not found at {pkl_file}\n"
            f"Set LHASA_DATA_DIR or place hERG_lhasa_training.pkl / hERG_lhasa_test.pkl in one of:\n"
            f"  /data/stat-cadd/scat9264/lhasa_data (server)\n"
            f"  /Users/apunt/repos/dphil/data (local)"
        )
    
    print(f"Loading FLuID hERG {dataset_type} data from {pkl_file}...")
    
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)

    # Convert RDKit Mol objects to SMILES. The .pkl stores Mol OBJECTS, not SMILES
    # strings; if they were pickled under a different RDKit version than the runtime,
    # Chem.MolToSmiles can return '' (empty, no exception). That silently produced
    # blank SMILES -> every downstream rep failed -> a confusing "No representations
    # could be generated!". Guard it: drop None/empty/unparseable, keep labels aligned,
    # warn on any loss, and fail LOUDLY (with the RDKit version) if nothing survives.
    raw_mols = list(df['MOLECULE'])
    activity = (df['ACTIVITY'] == 'Active').astype(int).values
    smiles, keep = [], []
    n_none = n_bad = 0
    for mol in raw_mols:
        if mol is None:
            n_none += 1; keep.append(False); continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = ''
        if not smi or Chem.MolFromSmiles(smi) is None:
            n_bad += 1; keep.append(False); continue
        smiles.append(smi); keep.append(True)
    keep = np.asarray(keep, dtype=bool)
    labels = activity[keep]
    if n_none or n_bad:
        print(f"  WARNING: load_herg_fluid dropped {n_none} None + {n_bad} empty/invalid "
              f"Mol->SMILES of {len(raw_mols)} molecules")
    if len(smiles) == 0:
        raise ValueError(
            f"load_herg_fluid: 0 of {len(raw_mols)} FLuID Mol objects yielded a valid SMILES "
            f"(all None or empty). The .pkl's RDKit Mol objects are almost certainly incompatible "
            f"with this RDKit ({Chem.rdBase.rdkitVersion}). Fix: re-pickle the FLuID data storing "
            f"SMILES strings instead of Mol objects, or run under the RDKit version the .pkl was "
            f"created with.")
    
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
              data_dir: Optional[str] = None,
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
    if data_dir is None:
        data_dir = _default_data_dir('')  # cache root; sub-dirs appended below
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

# ── hERG IC50, one row per MEASUREMENT ─────────────────────────────────────
# Added 2026-09-16, owner's decision. Every other hERG loader in this repository
# averages a molecule's repeat measurements together before anything sees them,
# which destroys the only real measurement error the dataset carries.
#
# WHY IC50 AND NOT Ki. Counted on 2026-09-16 from full ChEMBL pulls of CHEMBL240,
# with no curator-flag filter:
#
#   hERG Ki    1,815 records over 1,488 molecules; 42 have three or more records,
#              and 22 of those 42 have every record identical. Median disagreement
#              0.0000 log units. Nothing to recover.
#   hERG IC50  12,008 records over 9,802 molecules; 405 have three or more
#              records, 230 of those genuinely disagree, 72 disagree by more than
#              0.5 log units. Largest disagreement 2.51 log units.
#
# WHAT THIS RETURNS. One row per record, not per molecule, so a molecule measured
# four times appears four times with the same SMILES and four different labels.
# That is the same shape the drug-target loader uses for repeated pairs, and it
# is what `n17_replicate_removal` needs. `replicate_groups` says which rows are
# the same molecule, and the caller must group its folds by that too, or a
# molecule sits on both sides of a split.
#
# NOTHING IS DELETED FOR DISAGREEING. The 1.0-log-unit cut the older fetch
# applies would remove 84 of these 9,802 molecules, and those 84 are the ones
# that disagree most -- exactly the molecules a removal experiment is about.

HERG_IC50_TARGET = 'CHEMBL240'


def _herg_ic50_raw_path():
    """Where the per-record pull lives. Written by _probe_chembl_herg.py."""
    import os
    from pathlib import Path
    cache = Path(os.environ.get('KIRBY_CACHE_DIR',
                                str(Path.home() / '.cache' / 'kirby')))
    return cache / 'chembl' / f'chembl_raw_{HERG_IC50_TARGET}_IC50.csv'


def load_herg_ic50_measurements(min_records=1):
    """hERG IC50 as one row per measurement, with the repeat structure kept.

    min_records: keep only molecules measured at least this many times. The
    default keeps everything, because the molecules measured once are still
    worth training on even though they can never be graded by n17.

    Returns a dict with `smiles`, `labels` (pIC50), `replicate_groups` (one
    integer per row, equal for rows of the same molecule), `n_measurements`
    (per row, how many times that molecule was measured) and `task_type`.

    Also `curator_flag` and `curator_flag_text`, one entry per row, when the
    pull carried ChEMBL's `data_validity_comment`. That field is written by
    ChEMBL's own curators -- "Outside typical range", "Potential transcription
    error" -- and it is the only list of suspect records here that a person
    made rather than a script. `curator_flag` is False everywhere on a pull
    written before 2026-09-17, which did not request the column;
    `curator_flag_available` says which of the two this is.

    Raises if the per-record file is absent rather than quietly falling back to
    an averaged file, because the averaged file is what this exists to avoid.
    """
    import numpy as np
    import pandas as pd

    path = _herg_ic50_raw_path()
    if not path.exists():
        raise FileNotFoundError(
            f"hERG IC50 per-record pull not found at {path}. Fetch it with:\n"
            f"  python tests/experiments/noise/_probe_chembl_herg.py "
            f"--standard-type IC50\n"
            f"This loader will NOT fall back to chembl_herg_ki.csv: that file "
            f"holds one averaged number per molecule, which is the thing this "
            f"loader exists to avoid.")

    df = pd.read_csv(path)
    smiles_col = 'smiles_raw' if 'smiles_raw' in df.columns else 'canonical_smiles'
    # Carried through every filter below on the same frame, so the flag cannot
    # come out of step with the labels. A pull written before the column was
    # requested has no flag at all, and that is said rather than filled in
    # with False and left to look like a measurement.
    has_flag_column = 'data_validity_comment' in df.columns
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_value'] > 0]
    df = df.dropna(subset=[smiles_col, 'standard_value']).copy()

    # pIC50, the same transform every other ChEMBL loader here uses.
    df['pActivity'] = 9 - np.log10(df['standard_value'])

    counts = df.groupby(smiles_col)[smiles_col].transform('size')
    if min_records > 1:
        df = df[counts >= int(min_records)].copy()
        counts = df.groupby(smiles_col)[smiles_col].transform('size')

    # Stable ids: first appearance order, so two runs agree.
    codes, _ = pd.factorize(df[smiles_col], sort=False)

    if has_flag_column:
        txt = df['data_validity_comment'].astype('string').str.strip()
        flag = txt.notna() & (txt != '') & (txt.str.lower() != 'none')
        flag_text = txt.where(flag, '').fillna('').tolist()
        curator_flag = flag.to_numpy(dtype=bool)
    else:
        curator_flag = np.zeros(len(df), dtype=bool)
        flag_text = [''] * len(df)

    smiles = df[smiles_col].tolist()
    labels = df['pActivity'].to_numpy(dtype=float)
    groups = np.asarray(codes, dtype=np.int64)
    n_meas = np.asarray(counts, dtype=np.int64)

    n_mol = int(groups.max()) + 1 if len(groups) else 0
    n_3plus = int(pd.Series(n_meas).groupby(groups).first().ge(3).sum())
    print(f"  hERG IC50: {len(labels)} measurements over {n_mol} molecules; "
          f"{n_3plus} molecules measured three or more times")
    if has_flag_column:
        n_flag_mol = int(pd.Series(curator_flag).groupby(groups).any().sum())
        print(f"  {int(curator_flag.sum())} of those {len(labels)} "
              f"measurements carry a ChEMBL curator comment, on "
              f"{n_flag_mol} of the {n_mol} molecules")
    else:
        print(f"  no data_validity_comment column in {path.name}: this pull "
              f"predates the column being requested, so no curator flag is "
              f"available")

    return {
        'smiles': smiles,
        'labels': labels,
        'replicate_groups': groups,
        'n_measurements': n_meas,
        'curator_flag': curator_flag,
        'curator_flag_text': flag_text,
        'curator_flag_available': bool(has_flag_column),
        'task_type': 'regression',
        'source': str(path),
    }
