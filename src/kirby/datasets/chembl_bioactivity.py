"""Bulk ChEMBL Bioactivity Data Loader for DNN Pre-Training

Downloads pooled IC50 regression data from ChEMBL across the top N targets by
data count. Used for Imrie-style transfer learning: pre-train a DNN on broad
bioactivity data, then fine-tune on specific downstream tasks.

Data pipeline:
  1. Query ChEMBL for all IC50 activities, count per target, take top N
  2. For top N targets: fetch IC50 (standard_type='IC50', units='nM', relation='=')
  3. Standardize SMILES, compute pIC50 = -log10(IC50_nM * 1e-9)
  4. Deduplicate by (InChIKey, target) — geometric mean of pIC50 within same target
  5. Exclude compounds matching an optional InChIKey set (leakage prevention)
  6. Subsample to max_compounds if needed
  7. Cache to disk

Reference:
  Imrie et al. (2018). "Protein Family-Specific Models Using Deep Neural
  Networks and Transfer Learning Improve Virtual Screening and Highlight
  the Need for More Data." JCIM 58(11):2319-2330.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Set


def _get_top_targets(max_targets: int = 200) -> pd.DataFrame:
    """Query ChEMBL for targets ranked by IC50 activity count.

    Iterates all IC50 activities and counts per target_chembl_id.
    Returns DataFrame with columns: target_chembl_id, activity_count
    """
    from chembl_webresource_client.new_client import new_client

    activity_client = new_client.activity

    print("  Counting IC50 activities per target (this may take several minutes)...")
    activities = activity_client.filter(
        standard_type='IC50',
        standard_units='nM',
        standard_relation='=',
    ).only(['target_chembl_id'])

    target_counts = {}
    n_total = 0
    for act in activities:
        tid = act.get('target_chembl_id')
        if tid:
            target_counts[tid] = target_counts.get(tid, 0) + 1
        n_total += 1
        if n_total % 100000 == 0:
            print(f"    Processed {n_total} activities...", flush=True)

    print(f"  Total IC50 activities: {n_total}")
    print(f"  Unique targets with IC50 data: {len(target_counts)}")

    # Sort by count, take top N
    sorted_targets = sorted(target_counts.items(), key=lambda x: -x[1])
    top = sorted_targets[:max_targets]

    df = pd.DataFrame(top, columns=['target_chembl_id', 'activity_count'])
    print(f"  Top {max_targets} targets: {df['activity_count'].sum()} total activities")
    print(f"  Range: {df['activity_count'].iloc[-1]} to "
          f"{df['activity_count'].iloc[0]} per target")
    return df


def _standardize_and_inchikey(smiles_series):
    """Vectorized SMILES standardization and InChIKey computation.

    Args:
        smiles_series: pd.Series of SMILES strings.

    Returns:
        (canonical_smiles, inchikeys) — pd.Series, with None for failures.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    canon = []
    ikeys = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            canon.append(None)
            ikeys.append(None)
            continue
        try:
            c = Chem.MolToSmiles(mol, canonical=True)
            inchi = MolToInchi(mol)
            k = InchiToInchiKey(inchi) if inchi else None
        except Exception:
            canon.append(None)
            ikeys.append(None)
            continue
        canon.append(c)
        ikeys.append(k)

    return pd.Series(canon, index=smiles_series.index), \
           pd.Series(ikeys, index=smiles_series.index)


def download_chembl_bioactivity(
    data_dir: str = '/Volumes/seagate/kirby/chembl',
    max_targets: int = 200,
    max_compounds: int = 500_000,
    exclude_inchikeys: Optional[Set[str]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Download broad ChEMBL IC50 data for DNN pre-training.

    Returns DataFrame with columns: smiles, pActivity, target_chembl_id, inchikey.
    Multiple rows per compound are expected (one per target).
    Caches to {data_dir}/chembl_bioactivity_top{max_targets}.csv.

    Args:
        data_dir: Directory to cache data (large file, use external drive).
        max_targets: Number of top targets by activity count.
        max_compounds: Maximum compounds to include (subsample if needed).
        exclude_inchikeys: Set of InChIKeys to exclude (leakage prevention).
        random_seed: Random seed for subsampling.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, f'chembl_bioactivity_top{max_targets}.csv')

    if os.path.exists(cache_file):
        print(f"Loading cached ChEMBL bioactivity data from {cache_file}...")
        df = pd.read_csv(cache_file)
        print(f"  {len(df)} rows, "
              f"{df['inchikey'].nunique()} unique compounds, "
              f"{df['target_chembl_id'].nunique()} targets")
    else:
        from chembl_webresource_client.new_client import new_client

        # Step 1: Get top targets
        top_targets = _get_top_targets(max_targets)
        target_ids = top_targets['target_chembl_id'].tolist()

        # Step 2: Fetch IC50 activities for each target
        print(f"\nFetching IC50 data for {len(target_ids)} targets...")
        activity_client = new_client.activity
        all_records = []

        for i, tid in enumerate(target_ids):
            activities = activity_client.filter(
                target_chembl_id=tid,
                standard_type='IC50',
                standard_units='nM',
                standard_relation='=',
            ).only(['canonical_smiles', 'standard_value', 'target_chembl_id'])

            n_before = len(all_records)
            for act in activities:
                smi = act.get('canonical_smiles')
                val = act.get('standard_value')
                if smi and val:
                    try:
                        val_f = float(val)
                        if val_f > 0:
                            all_records.append({
                                'smiles': smi,
                                'ic50_nM': val_f,
                                'target_chembl_id': tid,
                            })
                    except (ValueError, TypeError):
                        continue

            n_new = len(all_records) - n_before
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Target {i+1}/{len(target_ids)}: {tid} "
                      f"({n_new} records, {len(all_records)} total)", flush=True)

        print(f"\nRaw records: {len(all_records)}")
        df_raw = pd.DataFrame(all_records)

        # Step 3: Standardize SMILES and compute InChIKey
        print("Standardizing SMILES and computing InChIKeys...")
        df_raw['smiles'], df_raw['inchikey'] = \
            _standardize_and_inchikey(df_raw['smiles'])

        # Drop rows where SMILES or InChIKey failed
        df = df_raw.dropna(subset=['smiles', 'inchikey']).copy()
        print(f"After SMILES validation: {len(df)} records")

        # Step 4: Compute pIC50
        df['pActivity'] = 9 - np.log10(df['ic50_nM'])

        # Step 5: Deduplicate by (InChIKey, target) — geometric mean of pIC50
        # Same compound measured multiple times AT THE SAME TARGET gets averaged.
        # Same compound at DIFFERENT targets keeps separate rows (different bioactivities).
        print("Deduplicating by (InChIKey, target) — geometric mean pIC50...")
        n_before_dedup = len(df)
        df = df.groupby(['inchikey', 'target_chembl_id']).agg({
            'smiles': 'first',
            'pActivity': 'mean',
        }).reset_index()
        print(f"After deduplication: {len(df)} rows "
              f"(removed {n_before_dedup - len(df)} duplicate measurements)")

        # Drop intermediate column
        df = df[['smiles', 'pActivity', 'target_chembl_id', 'inchikey']]

        # Cache
        df.to_csv(cache_file, index=False)
        print(f"Cached to {cache_file}")

    # Step 5b: Exclude downstream compounds (leakage prevention)
    if exclude_inchikeys:
        n_before = len(df)
        df = df[~df['inchikey'].isin(exclude_inchikeys)]
        n_removed = n_before - len(df)
        print(f"Excluded {n_removed} rows matching downstream InChIKeys "
              f"({len(df)} remaining)")

    # Step 6: Subsample if needed (by unique compound, not by row)
    if df['inchikey'].nunique() > max_compounds:
        print(f"Subsampling from {df['inchikey'].nunique()} to "
              f"{max_compounds} unique compounds...")
        rng = np.random.RandomState(random_seed)
        keep_keys = rng.choice(df['inchikey'].unique(), max_compounds, replace=False)
        df = df[df['inchikey'].isin(set(keep_keys))]

    # Filter invalid pActivity
    df = df[np.isfinite(df['pActivity'])].copy()

    print(f"\nFinal ChEMBL bioactivity dataset:")
    print(f"  Rows: {len(df)} (compound-target pairs)")
    print(f"  Unique compounds: {df['inchikey'].nunique()}")
    print(f"  Targets: {df['target_chembl_id'].nunique()}")
    print(f"  pActivity range: [{df['pActivity'].min():.2f}, "
          f"{df['pActivity'].max():.2f}]")
    print(f"  pActivity mean: {df['pActivity'].mean():.2f} "
          f"+/- {df['pActivity'].std():.2f}")

    return df


def collect_downstream_inchikeys(datasets=None) -> Set[str]:
    """Collect InChIKeys from all downstream molecular datasets for leakage prevention.

    Returns a set of InChIKey strings.
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'tests'))

    if datasets is None:
        datasets = ['logd', 'caco2', 'herg', 'qm9',
                    'adme_hlm', 'adme_rlm', 'adme_mdr1', 'adme_solubility']

    all_keys = set()
    for ds_name in datasets:
        try:
            from find_best_hybrid_molecular import load_dataset
            smiles, _, _ = load_dataset(ds_name)
            for smi in smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                try:
                    inchi = MolToInchi(mol)
                    ikey = InchiToInchiKey(inchi) if inchi else None
                    if ikey:
                        all_keys.add(ikey)
                except Exception:
                    continue
            print(f"  {ds_name}: {len(smiles)} SMILES -> "
                  f"{len(all_keys)} cumulative InChIKeys")
        except Exception as e:
            print(f"  WARNING: Failed to load {ds_name}: {e}")

    print(f"\nTotal downstream InChIKeys: {len(all_keys)}")
    return all_keys
