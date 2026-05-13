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
from pathlib import Path
from typing import Optional, Set

from kirby.utils.species import chembl_organism


def _default_data_dir(subdir: str) -> str:
    """Default cache subdir, preferring /Volumes/seagate (Mac) then
    $KIRBY_CACHE_DIR (server) then ~/.cache/kirby."""
    seagate = Path('/Volumes/seagate/kirby') / subdir
    if seagate.parent.parent.exists():
        return str(seagate)
    root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return str(root / subdir)


def _get_top_targets(max_targets: int = 200,
                     organism_latin: Optional[str] = None) -> pd.DataFrame:
    """Query ChEMBL for targets ranked by IC50 activity count.

    Iterates all IC50 activities (optionally filtered by target organism)
    and counts per target_chembl_id.
    Returns DataFrame with columns: target_chembl_id, activity_count.

    Note on cross-species: ChEMBL's activity endpoint exposes the
    denormalized `target_organism` string field, not `tax_id`. The
    Latin binomial in SPECIES_REGISTRY is used directly. If a query
    returns zero rows for a non-human organism, the registry string
    may need to be updated (e.g. `Canis familiaris` vs
    `Canis lupus familiaris` for tax_id 9615).
    """
    from chembl_webresource_client.new_client import new_client

    activity_client = new_client.activity

    filter_kwargs = dict(
        standard_type='IC50',
        standard_units='nM',
        standard_relation='=',
    )
    if organism_latin is not None:
        filter_kwargs['target_organism'] = organism_latin

    org_msg = f" for organism='{organism_latin}'" if organism_latin else ""
    print(f"  Counting IC50 activities per target{org_msg} "
          f"(this may take several minutes)...")
    activities = activity_client.filter(**filter_kwargs).only(['target_chembl_id'])

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
    data_dir: Optional[str] = None,
    max_targets: int = 200,
    max_compounds: int = 500_000,
    exclude_inchikeys: Optional[Set[str]] = None,
    random_seed: int = 42,
    organism: str = 'human',
    organism_strict: bool = True,
) -> pd.DataFrame:
    """Download broad ChEMBL IC50 data for DNN pre-training.

    Returns DataFrame with columns: smiles, pActivity, target_chembl_id, inchikey.
    Multiple rows per compound are expected (one per target).
    Caches to {data_dir}/chembl_bioactivity_top{max_targets}_{organism}.csv.

    Args:
        data_dir: Directory to cache data (large file, use external drive).
        max_targets: Number of top targets by activity count.
        max_compounds: Maximum compounds to include (subsample if needed).
        exclude_inchikeys: Set of InChIKeys to exclude (leakage prevention).
        random_seed: Random seed for subsampling.
        organism: One of 'human', 'dog', 'cat', 'cow', 'horse'. Filters
            ChEMBL activities to targets whose organism matches the
            resolved Latin binomial. Default 'human'.
        organism_strict: If True (default), only activities whose target's
            organism field exactly matches the resolved species. If False,
            accept any non-human assay. Reserved for future use;
            non-strict mode currently raises NotImplementedError.

    Cross-species note: ChEMBL vet coverage is sparse. Per design doc §7,
    expect rich human data, sparse-to-moderate dog, very sparse cat/cow/horse.
    """
    if not organism_strict:
        raise NotImplementedError(
            "organism_strict=False is reserved for future use. "
            "Use organism_strict=True (default) for now."
        )
    organism_latin = chembl_organism(organism)

    if data_dir is None:
        data_dir = _default_data_dir('chembl')
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(
        data_dir,
        f'chembl_bioactivity_top{max_targets}_{organism}.csv',
    )
    # Legacy alias: pre-cross-species runs cached to the unsuffixed filename.
    # Only honor for organism='human' (the original behaviour).
    legacy_cache = os.path.join(data_dir, f'chembl_bioactivity_top{max_targets}.csv')
    if (organism == 'human'
            and not os.path.exists(cache_file)
            and os.path.exists(legacy_cache)):
        print(f"  Using legacy unsuffixed cache: {legacy_cache}")
        cache_file = legacy_cache

    if os.path.exists(cache_file):
        print(f"Loading cached ChEMBL bioactivity data from {cache_file}...")
        df = pd.read_csv(cache_file)
        if 'organism' not in df.columns:
            df['organism'] = organism
        print(f"  {len(df)} rows, "
              f"{df['inchikey'].nunique()} unique compounds, "
              f"{df['target_chembl_id'].nunique()} targets")
    else:
        from chembl_webresource_client.new_client import new_client

        # Step 1: Get top targets within the requested organism
        top_targets = _get_top_targets(max_targets, organism_latin=organism_latin)
        if len(top_targets) == 0:
            raise RuntimeError(
                f"No targets found for organism='{organism}' "
                f"(Latin='{organism_latin}'). ChEMBL may use a different "
                f"organism string for this species (e.g. "
                f"'Canis lupus familiaris' instead of 'Canis familiaris'). "
                f"Check SPECIES_REGISTRY in src/kirby/utils/species.py."
            )
        target_ids = top_targets['target_chembl_id'].tolist()

        # Step 2: Fetch IC50 activities for each target
        print(f"\nFetching IC50 data for {len(target_ids)} targets "
              f"(organism='{organism_latin}')...")
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

        # Tag every row with the organism so concatenated multi-species
        # frames retain provenance for species-as-token augmentation.
        df['organism'] = organism

        # Drop intermediate column
        df = df[['smiles', 'pActivity', 'target_chembl_id', 'inchikey', 'organism']]

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
