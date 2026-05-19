"""Canine CYP isoform substrate / inhibition data loader (T-Mol.3).

The canonical dog CYP isoforms — CYP1A2, CYP2B11, CYP2C21, CYP2D15, CYP3A12 —
are commercially important for vet ADMET (the cat UGT1A6 deficiency, dog
CYP3A12 vs human CYP3A4, breed-MDR1 stories from the research memo). Per
``docs/cross_species_admet_design.md`` §7 most of this data is **not** in
ChEMBL; it lives in hand-curated literature CSVs at
``data/raw/canine_cyp_<isoform>.csv``. ChEMBL is a supplement when a dog
target ID has been resolved (``scripts/resolve_chembl_targets.py``).

Resolution order per ``(isoform, organism)``:

1. CSV at ``data/raw/canine_cyp_<isoform>{_<organism>}.csv`` if present.
2. ChEMBL target ID from ``docs/cross_species_targets_resolved.json`` →
   live activity query (same machinery as ``cox2.py`` / ``drd2.py``).
3. Documented skip — empty result with a clear sparse-data log.

The loader is intentionally thin; it is **infrastructure** for T-Mol.4, not
a curated dataset on its own. Acceptance per ``CROSS_SPECIES_ROADMAP.md``
§6 Track X-Mol: each isoform either returns >20 compounds **or** a
documented skip — both are valid first-pass outputs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kirby.utils.species import resolve_species


CANINE_CYP_ISOFORMS: List[str] = [
    '1A2',   # CYP1A2 — same name across human/dog (orthologue)
    '2B11',  # CYP2B11 — dog-specific (human orthologue: CYP2B6)
    '2C21',  # CYP2C21 — dog-specific (human orthologue: CYP2C9)
    '2D15',  # CYP2D15 — dog-specific (human orthologue: CYP2D6)
    '3A12',  # CYP3A12 — dog-specific (human orthologue: CYP3A4)
]

# Map canine isoform -> ChEMBL gene key in resolved-targets JSON. The JSON
# uses dog-specific gene symbols where they differ from human.
_ISOFORM_TO_GENE: Dict[str, str] = {
    '1A2': 'CYP1A2',
    '2B11': 'CYP2B11',
    '2C21': 'CYP2C21',
    '2D15': 'CYP2D15',
    '3A12': 'CYP3A12',
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESOLVED_JSON = _REPO_ROOT / 'docs' / 'cross_species_targets_resolved.json'
DEFAULT_DATA_RAW_DIR = _REPO_ROOT / 'data' / 'raw'

SPARSE_THRESHOLD = 20  # design doc §13 decision 4


def _resolved_target_id(gene: str, organism: str) -> Optional[str]:
    """Look up the resolved ChEMBL target ID for `(gene, organism)`.

    Returns None if the JSON is missing, the gene is not in it, or the
    organism has no usable target (best=None or flag in {none, skip, error}).
    """
    if not _RESOLVED_JSON.exists():
        return None
    try:
        data = json.loads(_RESOLVED_JSON.read_text())
    except Exception:
        return None
    entry = data.get(gene, {}).get(organism)
    if not entry:
        return None
    if entry.get('flag') in ('none', 'skip', 'error'):
        return None
    return entry.get('best')


def _csv_candidates(isoform: str, organism: str,
                    data_raw_dir: Path) -> List[Path]:
    """Candidate hand-curated CSV paths for `(isoform, organism)`."""
    base = data_raw_dir / f'canine_cyp_{isoform.lower()}'
    return [
        base.with_name(f'{base.name}_{organism}.csv'),
        base.with_suffix('.csv'),  # organism-agnostic; dog is the default reading
    ]


def _load_csv(isoform: str, organism: str,
              data_raw_dir: Path) -> Optional[pd.DataFrame]:
    """Return the first CSV candidate that exists, else None.

    Expected columns: ``smiles``, ``label``. Extra columns (reference,
    assay_type, …) are passed through untouched.
    """
    for path in _csv_candidates(isoform, organism, data_raw_dir):
        if path.exists():
            df = pd.read_csv(path)
            if 'smiles' not in df.columns or 'label' not in df.columns:
                raise ValueError(
                    f"{path} must contain 'smiles' and 'label' columns; "
                    f"found {list(df.columns)}."
                )
            df = df.dropna(subset=['smiles', 'label']).copy()
            df.attrs['source_path'] = str(path)
            return df
    return None


def _load_chembl(target_chembl_id: str, isoform: str, organism: str,
                 max_spread_log_units: float = 1.0) -> pd.DataFrame:
    """Query ChEMBL activities for a single canine CYP target. IC50+Ki, nM.

    The returned DataFrame is grouped by ``(smiles, standard_type)`` — a
    compound assayed in both IC50 and Ki appears as **two rows** with
    different `label` (pActivity) values. Downstream callers that need
    one-row-per-compound must filter on ``standard_type`` first.
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:
        raise ImportError(
            "ChEMBL client required: pip install chembl_webresource_client"
        ) from exc
    from rdkit import Chem  # type: ignore

    activity_client = new_client.activity
    rows: List[Dict] = []
    for stype in ('IC50', 'Ki'):
        activities = activity_client.filter(
            target_chembl_id=target_chembl_id,
            standard_type=stype,
            standard_units='nM',
            standard_relation='=',
            data_validity_comment__isnull=True,
        ).only(['canonical_smiles', 'standard_value', 'assay_chembl_id'])
        for act in activities:
            smi = act.get('canonical_smiles')
            val = act.get('standard_value')
            if not smi or val is None:
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                continue
            if val_f <= 0:
                continue
            rows.append({
                'smiles': smi,
                'value_nM': val_f,
                'standard_type': stype,
                'assay_id': act.get('assay_chembl_id'),
            })

    if not rows:
        return pd.DataFrame(columns=['smiles', 'label', 'standard_type'])

    df = pd.DataFrame(rows)
    valid_smi, canon_smi = [], []
    for s in df['smiles']:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            valid_smi.append(False)
            canon_smi.append(None)
        else:
            valid_smi.append(True)
            canon_smi.append(Chem.MolToSmiles(mol))
    df['valid'] = valid_smi
    df['canonical_smiles'] = canon_smi
    df = df[df['valid']].copy()
    df['smiles'] = df['canonical_smiles']
    df = df.drop(columns=['valid', 'canonical_smiles'])

    # pActivity per row (consistent with cox2.py / drd2.py)
    df['pActivity'] = 9 - np.log10(df['value_nM'])

    # Aggregate per (smiles, standard_type) — geometric mean of pActivity.
    grouped = df.groupby(['smiles', 'standard_type']).agg(
        label=('pActivity', 'mean'),
        n_measurements=('pActivity', 'size'),
        spread=('pActivity', lambda x: float(x.max() - x.min())),
    ).reset_index()
    grouped = grouped[grouped['spread'] <= max_spread_log_units].copy()
    grouped.attrs['source_path'] = (
        f'chembl:{target_chembl_id}:{organism}:{isoform}'
    )
    return grouped


def load_canine_cyp(isoform: str,
                    organism: str = 'dog',
                    prefer: str = 'csv',
                    csv_dir: Optional[Path] = None) -> Dict:
    """Load canine CYP isoform substrate / inhibition data.

    Args:
        isoform: One of ``CANINE_CYP_ISOFORMS`` (e.g. '3A12').
        organism: Species key from ``SPECIES_REGISTRY``. Default 'dog'.
            'human' is supported when a human orthologue is resolved in the
            JSON and used for cross-species comparison.
        prefer: 'csv' (default) checks the hand-curated CSV first and falls
            back to ChEMBL; 'chembl' inverts the order.
        csv_dir: Override for ``data/raw`` lookup directory (testing).

    Returns:
        dict with: ``smiles`` (list), ``labels`` (np.ndarray), ``organism``,
        ``isoform``, ``source`` ('csv'|'chembl'|'none'), ``source_path``,
        ``n_compounds`` and any extra columns from the CSV. When neither
        source has data, returns the dict with empty arrays and a sparse-
        skip log; callers should treat ``n_compounds < 20`` as
        "documented skip" (design doc §13 decision 4, roadmap §6 Track X-Mol).
    """
    if isoform not in CANINE_CYP_ISOFORMS:
        raise ValueError(
            f"Unknown canine CYP isoform '{isoform}'. "
            f"Available: {CANINE_CYP_ISOFORMS}"
        )
    # Validate organism against the cross-species registry.
    resolve_species(organism)

    data_raw_dir = Path(csv_dir) if csv_dir is not None else DEFAULT_DATA_RAW_DIR

    gene = _ISOFORM_TO_GENE[isoform]
    csv_df: Optional[pd.DataFrame] = None
    chembl_df: Optional[pd.DataFrame] = None

    if prefer == 'csv':
        csv_df = _load_csv(isoform, organism, data_raw_dir)
        if csv_df is None or len(csv_df) == 0:
            tid = _resolved_target_id(gene, organism)
            if tid is not None:
                chembl_df = _load_chembl(tid, isoform, organism)
    elif prefer == 'chembl':
        tid = _resolved_target_id(gene, organism)
        if tid is not None:
            chembl_df = _load_chembl(tid, isoform, organism)
        if chembl_df is None or len(chembl_df) == 0:
            csv_df = _load_csv(isoform, organism, data_raw_dir)
    else:
        raise ValueError(f"prefer must be 'csv' or 'chembl', got '{prefer}'")

    if csv_df is not None and len(csv_df) > 0:
        df = csv_df
        source = 'csv'
    elif chembl_df is not None and len(chembl_df) > 0:
        df = chembl_df
        source = 'chembl'
    else:
        print(
            f"  WARNING: no canine CYP{isoform} data found for "
            f"organism='{organism}'. Checked CSV "
            f"{_csv_candidates(isoform, organism, data_raw_dir)} "
            f"and resolved JSON gene='{gene}'. "
            f"Documented skip per design doc §13."
        )
        return {
            'smiles': [],
            'labels': np.array([], dtype=float),
            'organism': organism,
            'isoform': isoform,
            'source': 'none',
            'source_path': None,
            'n_compounds': 0,
        }

    n = len(df)
    if n < SPARSE_THRESHOLD:
        print(
            f"  WARNING: only {n} compounds for canine CYP{isoform} "
            f"(organism='{organism}', source='{source}') — below the "
            f"{SPARSE_THRESHOLD}-row inclusion threshold (design doc §13). "
            f"Documented skip."
        )

    smiles = df['smiles'].astype(str).tolist()
    labels = np.asarray(df['label'].values, dtype=float)
    result: Dict = {
        'smiles': smiles,
        'labels': labels,
        'organism': organism,
        'isoform': isoform,
        'source': source,
        'source_path': df.attrs.get('source_path'),
        'n_compounds': n,
    }
    # Pass through extra CSV columns (reference, assay_type, ...) for
    # downstream provenance, without polluting the headline fields.
    for col in df.columns:
        if col in ('smiles', 'label'):
            continue
        result.setdefault('extras', {})[col] = df[col].tolist()
    return result


def load_all_canine_cyp(organism: str = 'dog', prefer: str = 'csv',
                        csv_dir: Optional[Path] = None
                        ) -> Dict[str, Dict]:
    """Convenience: load every canine CYP isoform; skipped isoforms log warnings."""
    return {
        iso: load_canine_cyp(iso, organism=organism, prefer=prefer,
                             csv_dir=csv_dir)
        for iso in CANINE_CYP_ISOFORMS
    }
