"""Loader for Lhasa's curated hERG datasets (SDF/PKL format).

Data location: /Users/apunt/repos/dphil/data/
Files:
    hERG_lhasa_training.pkl  — 7,772 molecules (Active/Inactive)
    hERG_lhasa_test.pkl      — 4,383 molecules (Active/Inactive)
    FLuID_small.sdf          — 10,000 unlabeled transfer molecules

The training data uses 'Lhasa Chembl Call' for activity labels.
The test data uses 'Overall Conservative Call' for activity labels.
Both store RDKit Mol objects, 2048-bit Morgan FPs, and InChIKeys.
"""

import os
import warnings
import numpy as np
import pandas as pd

_local_default = '/Users/apunt/repos/dphil/data'
_server_default = '/data/stat-cadd/scat9264/lhasa_data'
_DEFAULT_DATA_DIR = os.environ.get(
    'LHASA_DATA_DIR',
    _server_default if os.path.isdir(_server_default) else _local_default,
)


def _load_pkl_or_sdf(basename, data_dir, activity_col=None):
    """Load from PKL (fast) or fall back to SDF."""
    pkl_path = os.path.join(data_dir, f'{basename}.pkl')
    sdf_path = os.path.join(data_dir, f'{basename}.sdf')

    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        print(f"  Loaded {basename} from PKL: {len(df)} molecules")
        return df

    if os.path.exists(sdf_path):
        from rdkit.Chem import PandasTools, Chem
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        from rdkit.Chem import AllChem, DataStructs

        df = PandasTools.LoadSDF(sdf_path, molColName='MOLECULE')
        if activity_col and activity_col in df.columns:
            df.rename({activity_col: 'ACTIVITY'}, axis=1, inplace=True)

        # Compute fingerprints if not present
        if 'FP' not in df.columns:
            def _compute_fp(mol):
                arr = np.zeros((2048,), dtype=np.int8)
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=2048, useFeatures=True)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr
            df['FP'] = df['MOLECULE'].apply(_compute_fp)

        if 'INCHI' not in df.columns and 'inchi_key' not in df.columns:
            def _inchikey(mol):
                try:
                    return InchiToInchiKey(MolToInchi(mol))
                except Exception:
                    return 'UNKNOWN'
            df['INCHI'] = df['MOLECULE'].apply(_inchikey)
        elif 'inchi_key' in df.columns:
            df.rename({'inchi_key': 'INCHI'}, axis=1, inplace=True)

        if 'CLASS' not in df.columns and 'ACTIVITY' in df.columns:
            df['CLASS'] = (df['ACTIVITY'] != 'Inactive').astype(int)

        print(f"  Loaded {basename} from SDF: {len(df)} molecules")
        return df

    raise FileNotFoundError(
        f"Neither {pkl_path} nor {sdf_path} found. "
        f"Set LHASA_DATA_DIR env var or check path.")


def load_lhasa_herg_training(data_dir=None):
    """Load Lhasa's curated hERG training data.

    Returns:
        (smiles_list, binary_labels, inchikeys, fluid_df)
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    df = _load_pkl_or_sdf('hERG_lhasa_training', data_dir,
                          activity_col='Lhasa Chembl Call')

    # Extract SMILES from Mol objects
    from rdkit import Chem
    smiles = []
    valid_mask = []
    for mol in df['MOLECULE']:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                smiles.append(smi)
                valid_mask.append(True)
            except Exception:
                smiles.append(None)
                valid_mask.append(False)
        else:
            smiles.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    df_valid = df[valid_mask].copy()
    df_valid.index = range(len(df_valid))
    smiles_valid = [s for s, v in zip(smiles, valid_mask) if v]

    labels = np.array(df_valid['CLASS'].values, dtype=int)
    inchikeys = list(df_valid.get('INCHI', ['UNKNOWN'] * len(df_valid)))

    n_active = int(labels.sum())
    print(f"  Training: {len(labels)} molecules "
          f"({n_active} active, {len(labels) - n_active} inactive)")

    return smiles_valid, labels, inchikeys, df_valid


def load_lhasa_herg_test(data_dir=None):
    """Load Lhasa's curated hERG test data.

    Returns:
        (smiles_list, binary_labels, inchikeys, fluid_df)
    """
    data_dir = data_dir or _DEFAULT_DATA_DIR
    df = _load_pkl_or_sdf('hERG_lhasa_test', data_dir,
                          activity_col='Overall Conservative Call')

    from rdkit import Chem
    smiles = []
    valid_mask = []
    for mol in df['MOLECULE']:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                smiles.append(smi)
                valid_mask.append(True)
            except Exception:
                smiles.append(None)
                valid_mask.append(False)
        else:
            smiles.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    df_valid = df[valid_mask].copy()
    df_valid.index = range(len(df_valid))
    smiles_valid = [s for s, v in zip(smiles, valid_mask) if v]

    labels = np.array(df_valid['CLASS'].values, dtype=int)
    inchikeys = list(df_valid.get('INCHI', ['UNKNOWN'] * len(df_valid)))

    n_active = int(labels.sum())
    print(f"  Test: {len(labels)} molecules "
          f"({n_active} active, {len(labels) - n_active} inactive)")

    return smiles_valid, labels, inchikeys, df_valid


def dedup_across_silos(silo_smiles_dict, silo_labels_dict,
                       silo_inchikeys_dict):
    """Remove molecules that appear in multiple silos.

    Keeps the molecule in the FIRST silo encountered (by dict order).
    Returns updated dicts with duplicates removed from later silos.
    """
    seen_keys = set()
    result_smiles = {}
    result_labels = {}
    result_inchikeys = {}
    dedup_stats = {}

    for silo_name in silo_smiles_dict:
        smiles = silo_smiles_dict[silo_name]
        labels = silo_labels_dict[silo_name]
        keys = silo_inchikeys_dict[silo_name]

        keep_mask = []
        n_unknown = 0
        for key in keys:
            if key == 'UNKNOWN':
                # Keep UNKNOWN-keyed molecules (can't dedup, but don't drop)
                keep_mask.append(True)
                n_unknown += 1
            elif key in seen_keys:
                keep_mask.append(False)
            else:
                keep_mask.append(True)
                seen_keys.add(key)

        keep_mask = np.array(keep_mask)
        n_removed = int((~keep_mask).sum())
        if n_unknown > 0:
            warnings.warn(
                f"Silo {silo_name}: {n_unknown} molecules with UNKNOWN "
                f"InChIKey kept (cannot dedup these)")

        result_smiles[silo_name] = [s for s, k in zip(smiles, keep_mask) if k]
        result_labels[silo_name] = labels[keep_mask]
        result_inchikeys[silo_name] = [k for k, m in zip(keys, keep_mask) if m]
        dedup_stats[silo_name] = {
            'before': len(smiles),
            'after': int(keep_mask.sum()),
            'removed': n_removed,
        }

    return result_smiles, result_labels, result_inchikeys, dedup_stats
