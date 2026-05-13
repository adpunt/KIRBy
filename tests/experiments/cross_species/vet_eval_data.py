"""Loader for the 4 vet ChEMBL eval targets used by all cross-species experiments.

Reads canonical (species, target_chembl_id) entries from
`docs/cross_species_vet_eval_targets.json` and fetches activities for each
directly from the ChEMBL REST API, caching one CSV per (species, target) under
`/Volumes/seagate/kirby/cross_species/vet_eval/`.

Returns (smiles, pActivity, inchikey) tuples — same shape as the rest of the
molecular dataset loaders.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
VET_EVAL_JSON = REPO_ROOT / 'docs' / 'cross_species_vet_eval_targets.json'
DEFAULT_CACHE_DIR = '/Volumes/seagate/kirby/cross_species/vet_eval'


def load_vet_targets() -> Dict[str, Dict]:
    """Return the canonical (species, target_chembl_id) entries."""
    if not VET_EVAL_JSON.exists():
        raise FileNotFoundError(
            f"{VET_EVAL_JSON} not found. Run "
            f"`python scripts/lookup_vet_target_ids.py` to regenerate."
        )
    data = json.loads(VET_EVAL_JSON.read_text())
    return {k: v for k, v in data.items() if not k.startswith('_')}


def _fetch_chembl_activities(
    target_chembl_id: str,
    cache_csv: Path,
) -> pd.DataFrame:
    """Fetch IC50/Ki activities for one target_chembl_id from ChEMBL.

    Schema: smiles, pActivity, inchikey, target_chembl_id, organism (provenance).
    """
    if cache_csv.exists():
        return pd.read_csv(cache_csv)

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as e:
        raise ImportError(
            "chembl_webresource_client required: "
            "pip install chembl_webresource_client"
        ) from e
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    print(f"  Fetching activities for {target_chembl_id}...")
    acts = new_client.activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=['IC50', 'Ki'],
        standard_units='nM',
        standard_relation='=',
    ).only(['canonical_smiles', 'standard_value', 'standard_type',
            'target_chembl_id', 'target_organism'])

    rows = []
    for a in acts:
        smi = a.get('canonical_smiles')
        val = a.get('standard_value')
        if not smi or val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        rows.append({
            'smiles': smi,
            'value_nM': v,
            'standard_type': a.get('standard_type'),
            'target_chembl_id': a.get('target_chembl_id'),
            'organism': a.get('target_organism'),
        })

    if not rows:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=['smiles', 'pActivity', 'inchikey',
                                       'target_chembl_id', 'organism'])
        empty.to_csv(cache_csv, index=False)
        return empty

    df = pd.DataFrame(rows)

    # Standardise + InChIKey, drop failures
    canon, ikeys = [], []
    for smi in df['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canon.append(None)
            ikeys.append(None)
            continue
        try:
            canon.append(Chem.MolToSmiles(mol, canonical=True))
            inchi = MolToInchi(mol)
            ikeys.append(InchiToInchiKey(inchi) if inchi else None)
        except Exception:
            canon.append(None)
            ikeys.append(None)

    df['smiles'] = canon
    df['inchikey'] = ikeys
    df = df.dropna(subset=['smiles', 'inchikey']).copy()

    # pActivity = 9 - log10(IC50 or Ki in nM)
    df['pActivity'] = 9 - np.log10(df['value_nM'])

    # Deduplicate by InChIKey: geometric mean of pActivity if measured multiple times
    df = df.groupby('inchikey', as_index=False).agg({
        'smiles': 'first',
        'pActivity': 'mean',
        'target_chembl_id': 'first',
        'organism': 'first',
    })

    df = df[np.isfinite(df['pActivity'])]
    df = df[['smiles', 'pActivity', 'inchikey', 'target_chembl_id', 'organism']]

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"  Cached {len(df)} unique compounds to {cache_csv}")
    return df


def load_vet_dataset(
    key: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[List[str], np.ndarray, str]:
    """Load one vet eval dataset by key (e.g. 'cow_CA4').

    Returns (smiles, pActivity, task_type='regression') — same shape as
    `find_best_hybrid_molecular.load_dataset`.
    """
    targets = load_vet_targets()
    if key not in targets:
        raise KeyError(f"Unknown vet eval key '{key}'. "
                       f"Known: {sorted(targets)}")
    entry = targets[key]
    tid = entry['target_chembl_id']
    cache_csv = Path(cache_dir) / f"{key}_{tid}.csv"
    df = _fetch_chembl_activities(tid, cache_csv)
    return list(df['smiles']), df['pActivity'].to_numpy(dtype=float), 'regression'


def list_vet_keys() -> List[str]:
    return sorted(load_vet_targets().keys())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--key', type=str, default=None,
                    help='Vet eval key to fetch (e.g. cow_CA4).')
    ap.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR)
    args = ap.parse_args()

    if args.list:
        for k, v in load_vet_targets().items():
            print(f"  {k:12s}  {v['target_chembl_id']:12s}  "
                  f"n={v.get('activity_count_strict','?'):>5}  "
                  f"{v['pref_name']}")
        sys.exit(0)

    if not args.key:
        ap.error("Pass --key <name> or --list")
    smiles, y, _ = load_vet_dataset(args.key, cache_dir=args.cache_dir)
    print(f"  {args.key}: {len(smiles)} compounds, "
          f"pActivity {y.min():.2f}..{y.max():.2f}")
