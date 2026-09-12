"""Reaction yield prediction datasets.

Standard benchmarks for evaluating reaction-aware representations.

Buchwald-Hartwig:     ~3955 C-N coupling reactions (Ahneman et al. Science 2018)
                      Data from: rxn4chemistry/rxn_yields (Schwaller et al. 2021)
Suzuki-Miyaura:       ~5760 cross-coupling reactions (Perera et al. Science 2018)
                      Data from: rxn4chemistry/rxn_yields (Schwaller et al. 2021)
BH Doyle Plates:      ~4599 C-N coupling reactions (Doyle lab HTE plates)
                      Data from: doylelab/rxnpredict (GitHub)
                      DIFFERENT dataset from BH: different aryl halides, additives,
                      and reaction plates. Both are BH C-N coupling.
ELN Buchwald-Hartwig: Stub — Saebi et al. (JCIM 2023) used proprietary pharma ELN
                      data that is not publicly available. Function raises NotAvailableError.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def _cache_dir() -> Path:
    cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return cache_root / 'reaction_yields'


def _download_file(url: str, dest: Path) -> None:
    import requests
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


# Buchwald-Hartwig constants
_BH_AMINE = 'Cc1ccc(N)cc1'
_BH_SMARTS = (
    '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3]'
    '.[NH2;D1;+0:4]-[c:5]'
    '>>'
    '[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
)


def _make_bh_product(aryl_smi: str) -> Optional[str]:
    """Apply C-N coupling template to get product SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rxn = AllChem.ReactionFromSmarts(_BH_SMARTS)
    aryl = Chem.MolFromSmiles(aryl_smi)
    amine = Chem.MolFromSmiles(_BH_AMINE)
    if aryl is None or amine is None:
        return None
    products = rxn.RunReactants((aryl, amine))
    if not products:
        return None
    return Chem.MolToSmiles(products[0][0])


def _parse_rxn_components(rxn_smiles: List[str]) -> List[Dict[str, str]]:
    """Parse reaction SMILES into component dicts.

    Splits reactants by '.' and products by '.', assigns consistent keys
    across all reactions (padding with '' if a reaction has fewer fragments).
    """
    max_reactants = 0
    max_products = 0
    parsed = []
    for rxn in rxn_smiles:
        parts = rxn.split('>>')
        r_frags = parts[0].split('.') if parts[0] else []
        p_frags = parts[1].split('.') if len(parts) > 1 and parts[1] else []
        max_reactants = max(max_reactants, len(r_frags))
        max_products = max(max_products, len(p_frags))
        parsed.append((r_frags, p_frags))

    components = []
    for r_frags, p_frags in parsed:
        comp = {}
        for j in range(max_reactants):
            comp[f'reactant_{j}'] = r_frags[j] if j < len(r_frags) else ''
        for j in range(max_products):
            comp[f'product_{j}'] = p_frags[j] if j < len(p_frags) else ''
        components.append(comp)
    return components


def load_buchwald_hartwig(n_splits: int = 10, data_dir=None) -> Dict:
    """Load Buchwald-Hartwig C-N coupling yield dataset (~3955 reactions).

    Returns dict with:
        rxn_smiles: List[str] — full reaction SMILES (reactants>>product)
        labels: np.ndarray — yields (0-100)
        groups: np.ndarray — additive-based group IDs for GroupKFold (23 groups)
        splits: List[Tuple] — random 70/30 (deprecated, kept for comparison)
        components: List[Dict] — individual component SMILES per reaction
        metadata: dict
    """
    cache = Path(data_dir) if data_dir else _cache_dir()
    xlsx_path = cache / 'buchwald_hartwig' / 'Dreher_and_Doyle_input_data.xlsx'

    if not xlsx_path.exists():
        url = ('https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/'
               'master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx')
        print("  Downloading Buchwald-Hartwig data...")
        _download_file(url, xlsx_path)

    print("  Parsing Buchwald-Hartwig Excel...")
    try:
        df = pd.read_excel(xlsx_path, sheet_name='FullCV_01')
    except ImportError as e:
        if 'openpyxl' in str(e):
            raise ImportError(
                "openpyxl>=3.1.0 required for Excel loading. "
                "Fix: pip install --upgrade openpyxl"
            ) from e
        raise

    # Cache product SMILES per unique aryl halide
    aryl_products: Dict[str, Optional[str]] = {}
    rxn_smiles = []
    components = []
    labels = []

    for _, row in df.iterrows():
        aryl = str(row['Aryl halide']).strip()
        ligand = str(row['Ligand']).strip()
        base = str(row['Base']).strip()
        additive = str(row['Additive']).strip()
        yield_val = float(row['Output'])

        if aryl not in aryl_products:
            aryl_products[aryl] = _make_bh_product(aryl)
        product = aryl_products[aryl]
        if product is None:
            continue

        reactants = f"{aryl}.{_BH_AMINE}.{ligand}.{base}.{additive}"
        rxn_smiles.append(f"{reactants}>>{product}")
        components.append({
            'aryl_halide': aryl, 'amine': _BH_AMINE,
            'ligand': ligand, 'base': base, 'additive': additive,
        })
        labels.append(yield_val)

    labels = np.array(labels)
    n = len(labels)

    # Additive-based groups for GroupKFold (23 unique additives)
    # Literature: leave-one-additive-out is the standard OOS evaluation
    # for this dataset (Doyle 2023, "When Yield Prediction..." JCIM 2024)
    unique_additives = sorted(set(c['additive'] for c in components))
    add_to_id = {a: i for i, a in enumerate(unique_additives)}
    groups = np.array([add_to_id[c['additive']] for c in components])

    # Random 70/30 splits kept for backward compatibility / comparison
    random_splits = []
    for i in range(n_splits):
        rng = np.random.RandomState(42 + i)
        perm = rng.permutation(n)
        n_train = int(0.7 * n)
        random_splits.append((perm[:n_train], perm[n_train:]))

    print(f"  Buchwald-Hartwig: {n} reactions, {len(unique_additives)} additive "
          f"groups, yield range [{labels.min():.1f}, {labels.max():.1f}]")

    return {
        'rxn_smiles': rxn_smiles,
        'labels': labels,
        'groups': groups,
        'splits': random_splits,  # deprecated: random 70/30
        'components': components,
        'task_type': 'regression',
        'metadata': {
            'name': 'buchwald_hartwig',
            'n_reactions': n,
            'n_groups': len(unique_additives),
            'group_key': 'additive',
            'description': 'Buchwald-Hartwig C-N coupling yields '
                           '(Ahneman et al. Science 2018)',
        },
    }


def load_suzuki(n_splits: int = 10, data_dir=None) -> Dict:
    """Load Suzuki-Miyaura coupling yield dataset (~5760 reactions).

    Uses the original Perera et al. Science 2018 supplementary Excel which has
    labeled components (electrophile, ligand, base, solvent). Ligand is used
    as the group key for GroupKFold (11 unique ligands).

    Falls back to pre-processed TSVs (rxn + y only, no groups) if Excel
    loading fails.

    Returns dict with:
        rxn_smiles: List[str] — full reaction SMILES
        labels: np.ndarray — yields (0-100)
        groups: np.ndarray — ligand-based group IDs for GroupKFold
        splits: List[Tuple[np.ndarray, np.ndarray]] — random 70/30 (deprecated)
        components: List[Dict] — labeled component names per reaction
        metadata: dict
    """
    cache = Path(data_dir) if data_dir else _cache_dir()
    suzuki_dir = cache / 'suzuki'

    # Try loading from original Excel with labeled components
    xlsx_path = suzuki_dir / 'aap9112_Data_File_S1.xlsx'
    if not xlsx_path.exists():
        url = ('https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/'
               'master/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx')
        print("  Downloading Suzuki-Miyaura original data (Perera 2018 SI)...")
        _download_file(url, xlsx_path)

    try:
        return _load_suzuki_from_excel(xlsx_path, suzuki_dir, n_splits)
    except Exception as e:
        print(f"  WARNING: Excel loading failed ({e}), falling back to TSVs...")
        return _load_suzuki_from_tsvs(suzuki_dir, n_splits)


def _load_suzuki_from_excel(xlsx_path: Path, suzuki_dir: Path,
                            n_splits: int) -> Dict:
    """Load Suzuki data from original Perera 2018 Excel with labeled components."""
    print("  Parsing Suzuki-Miyaura Excel (Perera 2018 SI)...")
    try:
        df = pd.read_excel(xlsx_path)
    except ImportError as e:
        if 'openpyxl' in str(e):
            raise ImportError(
                "openpyxl>=3.1.0 required for Excel loading. "
                "Fix: pip install --upgrade openpyxl"
            ) from e
        raise

    # Yield column
    yield_col = 'Product_Yield_PCT_Area_UV'
    if yield_col not in df.columns:
        raise ValueError(f"Yield column '{yield_col}' not found. "
                         f"Available: {list(df.columns)}")

    # Drop rows with missing yield
    df = df.dropna(subset=[yield_col]).copy()
    labels = df[yield_col].values.astype(np.float64)

    # Also need reaction SMILES from TSVs for representation generation
    # Download TSVs if needed (they contain the rxn SMILES)
    first_tsv = suzuki_dir / 'random_split_0.tsv'
    if not first_tsv.exists():
        base_url = ('https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/'
                     'master/data/Suzuki-Miyaura/random_splits')
        print("  Downloading Suzuki-Miyaura reaction SMILES...")
        for i in range(10):
            _download_file(f"{base_url}/random_split_{i}.tsv",
                           suzuki_dir / f'random_split_{i}.tsv')

    df_tsv = pd.read_csv(first_tsv, sep='\t')
    if len(df_tsv) != len(df):
        raise ValueError(f"Excel ({len(df)} rows) and TSV ({len(df_tsv)} rows) "
                         f"size mismatch — cannot align reaction SMILES")

    # Excel and TSV rows are in the same order (both from Perera 2018)
    rxn_smiles = list(df_tsv['rxn'])

    # Extract labeled components
    ligand_col = 'Ligand_Short_Hand'
    components = []
    ligand_labels = []

    for _, row in df.iterrows():
        comp = {
            'electrophile': str(row.get('Reactant_1_Name', '')).strip(),
            'nucleophile': str(row.get('Reactant_2_Name', '')).strip(),
            'ligand': str(row.get(ligand_col, '')).strip(),
            'base': str(row.get('Reagent_1_Short_Hand', '')).strip(),
            'solvent': str(row.get('Solvent_1_Short_Hand', '')).strip(),
        }
        components.append(comp)
        ligand_labels.append(comp['ligand'])

    # Build ligand-based groups for GroupKFold
    # Replace NaN/empty ligands with 'none' for grouping
    ligand_labels = [l if l and l != 'nan' else 'none' for l in ligand_labels]
    unique_ligands = sorted(set(ligand_labels))
    lig_to_id = {l: i for i, l in enumerate(unique_ligands)}
    groups = np.array([lig_to_id[l] for l in ligand_labels])

    n = len(labels)

    # Random 70/30 splits kept for backward compatibility / comparison
    n_train = int(0.7 * n)
    random_splits = []
    for i in range(n_splits):
        rng = np.random.RandomState(42 + i)
        perm = rng.permutation(n)
        random_splits.append((perm[:n_train], perm[n_train:]))

    print(f"  Suzuki-Miyaura: {n} reactions, {len(unique_ligands)} ligand "
          f"groups, yield range [{labels.min():.1f}, {labels.max():.1f}]")

    return {
        'rxn_smiles': rxn_smiles,
        'labels': labels,
        'groups': groups,
        'splits': random_splits,  # deprecated: random 70/30
        'components': components,
        'task_type': 'regression',
        'metadata': {
            'name': 'suzuki',
            'n_reactions': n,
            'n_groups': len(unique_ligands),
            'group_key': 'ligand',
            'description': 'Suzuki-Miyaura coupling yields '
                           '(Perera et al. Science 2018)',
        },
    }


def _load_suzuki_from_tsvs(suzuki_dir: Path, n_splits: int) -> Dict:
    """Fallback: load Suzuki from pre-processed TSVs (no labeled components)."""
    first_tsv = suzuki_dir / 'random_split_0.tsv'
    if not first_tsv.exists():
        base_url = ('https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/'
                     'master/data/Suzuki-Miyaura/random_splits')
        print("  Downloading Suzuki-Miyaura splits...")
        for i in range(10):
            _download_file(f"{base_url}/random_split_{i}.tsv",
                           suzuki_dir / f'random_split_{i}.tsv')

    # Canonical data from split 0
    df0 = pd.read_csv(first_tsv, sep='\t')
    rxn_smiles = list(df0['rxn'])
    labels = df0['y'].values * 100  # 0-1 → 0-100 to match BH scale
    n = len(labels)
    n_train = int(0.7 * n)

    # Use pre-shuffled TSV order for splits
    rxn_to_idx = {r: j for j, r in enumerate(rxn_smiles)}
    splits = []
    for i in range(min(n_splits, 10)):
        dfi = pd.read_csv(suzuki_dir / f'random_split_{i}.tsv', sep='\t')
        shuffled_idx = np.array([rxn_to_idx.get(r, -1) for r in dfi['rxn']])
        if (shuffled_idx < 0).any():
            rng = np.random.RandomState(42 + i)
            perm = rng.permutation(n)
            splits.append((perm[:n_train], perm[n_train:]))
        else:
            splits.append((shuffled_idx[:n_train], shuffled_idx[n_train:]))

    components = _parse_rxn_components(rxn_smiles)

    print(f"  Suzuki-Miyaura (TSV fallback): {n} reactions, NO groups, "
          f"yield range [{labels.min():.1f}, {labels.max():.1f}]")

    return {
        'rxn_smiles': rxn_smiles,
        'labels': labels,
        'splits': splits,
        'components': components,
        'task_type': 'regression',
        'metadata': {
            'name': 'suzuki',
            'n_reactions': n,
            'description': 'Suzuki-Miyaura coupling yields '
                           '(Perera et al. Science 2018)',
        },
    }


# ---------------------------------------------------------------------------
# Amine constant for BH Doyle plates (same amine as the original BH dataset)
# ---------------------------------------------------------------------------
_BH_DOYLE_AMINE = 'Cc1ccc(N)cc1'  # 4-methylaniline (p-toluidine)


def load_bh_doyle_plates(n_splits: int = 10, data_dir=None) -> Dict:
    """Load Doyle lab BH C-N coupling plates dataset (~4599 reactions).

    This is a DIFFERENT dataset from load_buchwald_hartwig() (Ahneman et al.
    Science 2018 / Dreher & Doyle). Both are Buchwald-Hartwig C-N coupling
    from the Doyle lab, but from different experimental plate runs:
      - Existing BH:   3955 reactions, 22 additives, Excel source (rxn_yields)
      - This dataset:  4599 reactions, 23 additives, CSV source (rxnpredict)

    Data source: https://github.com/doylelab/rxnpredict
    The data_table.csv contains full reaction SMILES for all components.

    Groups: by additive for GroupKFold (same evaluation protocol as existing BH).

    Returns dict with:
        rxn_smiles: List[str] — full reaction SMILES (reactants>>product)
        labels: np.ndarray — yields (0-100)
        groups: np.ndarray — additive-based group IDs for GroupKFold
        splits: List[Tuple] — random 70/30 (deprecated, kept for comparison)
        components: List[Dict] — individual component SMILES per reaction
        metadata: dict
    """
    cache = Path(data_dir) if data_dir else _cache_dir()
    csv_path = cache / 'bh_doyle_plates' / 'data_table.csv'

    if not csv_path.exists():
        url = ('https://raw.githubusercontent.com/doylelab/rxnpredict/'
               'master/data_table.csv')
        print("  Downloading Doyle lab BH plates data...")
        _download_file(url, csv_path)

    print("  Parsing Doyle lab BH plates CSV...")
    df = pd.read_csv(csv_path)

    required_cols = {
        'aryl_halide_smiles', 'base_smiles', 'ligand_smiles',
        'product_smiles', 'yield', 'additive', 'base', 'ligand',
        'aryl_halide',
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Unexpected data_table.csv format; missing columns: {missing}"
        )

    # Drop rows with missing yield or missing aryl halide SMILES
    df = df.dropna(subset=['yield', 'aryl_halide_smiles']).copy()
    df = df[df['aryl_halide_smiles'].str.strip() != ''].copy()

    rxn_smiles = []
    components = []
    labels_list = []

    for _, row in df.iterrows():
        aryl = str(row['aryl_halide_smiles']).strip()
        base_smi = str(row['base_smiles']).strip()
        ligand_smi = str(row['ligand_smiles']).strip()
        additive_smi = str(row.get('additive_smiles', '')).strip()
        product_smi = str(row['product_smiles']).strip()

        if not aryl or not product_smi or product_smi == 'nan':
            continue

        # Build reactant SMILES: aryl halide + amine + ligand + base [+ additive]
        reactant_parts = [aryl, _BH_DOYLE_AMINE, ligand_smi, base_smi]
        if additive_smi and additive_smi != 'nan':
            reactant_parts.append(additive_smi)
        reactants = '.'.join(p for p in reactant_parts if p)
        rxn_smiles.append(f"{reactants}>>{product_smi}")

        components.append({
            'aryl_halide': aryl,
            'amine': _BH_DOYLE_AMINE,
            'ligand': str(row['ligand']).strip(),
            'ligand_smiles': ligand_smi,
            'base': str(row['base']).strip(),
            'base_smiles': base_smi,
            'additive': str(row['additive']).strip() if pd.notna(row.get('additive')) else '',
            'additive_smiles': additive_smi if additive_smi != 'nan' else '',
        })
        labels_list.append(float(row['yield']))

    labels = np.array(labels_list)
    n = len(labels)

    # Additive-based groups for GroupKFold (leave-one-additive-out)
    unique_additives = sorted(set(c['additive'] for c in components))
    add_to_id = {a: i for i, a in enumerate(unique_additives)}
    groups = np.array([add_to_id[c['additive']] for c in components])

    # Random 70/30 splits kept for backward compatibility / comparison
    random_splits = []
    for i in range(n_splits):
        rng = np.random.RandomState(42 + i)
        perm = rng.permutation(n)
        n_train = int(0.7 * n)
        random_splits.append((perm[:n_train], perm[n_train:]))

    print(f"  BH Doyle plates: {n} reactions, {len(unique_additives)} additive "
          f"groups, yield range [{labels.min():.1f}, {labels.max():.1f}]")

    return {
        'rxn_smiles': rxn_smiles,
        'labels': labels,
        'groups': groups,
        'splits': random_splits,  # deprecated: random 70/30
        'components': components,
        'task_type': 'regression',
        'metadata': {
            'name': 'bh_doyle_plates',
            'n_reactions': n,
            'n_groups': len(unique_additives),
            'group_key': 'additive',
            'description': (
                'Buchwald-Hartwig C-N coupling yields from Doyle lab HTE plates '
                '(doylelab/rxnpredict). '
                'DIFFERENT from Ahneman et al. Science 2018 — different plates, '
                '~4599 reactions, 23 additives.'
            ),
        },
    }


class ELNDataNotAvailableError(Exception):
    """Raised when ELN data is requested but not publicly available."""


def load_eln_buchwald_hartwig(data_dir=None) -> Dict:
    """Load ELN Buchwald-Hartwig dataset — STUB, data not publicly available.

    The ELN (Electronic Lab Notebook) Buchwald-Hartwig dataset from
    Saebi et al. (JCIM 2023) "When Are Reaction Yield Prediction Models
    Useful in Drug Discovery?" uses proprietary AstraZeneca / pharmaceutical
    company ELN data that is NOT publicly available.

    If you have obtained the data through a data access agreement:
      - Place the CSV file at: <data_dir>/eln_buchwald_hartwig/eln_bh.csv
      - Required columns: rxn_smiles (or reactants/products), yield,
        and optionally substrate_class or aryl_halide for GroupKFold.
      - Then implement load_eln_buchwald_hartwig() following the pattern
        of load_buchwald_hartwig() in this file.

    References:
        Saebi et al. (2023) "When Are Reaction Yield Prediction Models
        Useful in Drug Discovery?" JCIM.
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422

    Raises:
        ELNDataNotAvailableError: Always. This is a stub.
    """
    raise ELNDataNotAvailableError(
        "The ELN Buchwald-Hartwig dataset (Saebi et al. JCIM 2023) uses "
        "proprietary pharmaceutical ELN data that is not publicly available. "
        "\n\n"
        "To use this dataset, you must obtain the data through an institutional "
        "data access agreement with AstraZeneca or the relevant pharmaceutical "
        "company.\n\n"
        "Once obtained, place your CSV at:\n"
        "  <data_dir>/eln_buchwald_hartwig/eln_bh.csv\n\n"
        "The CSV should have columns: rxn_smiles (reaction SMILES in >> format), "
        "yield (numeric, 0-100), and ideally a substrate_class or aryl_halide "
        "column for GroupKFold splitting.\n\n"
        "Reference: https://pubs.acs.org/doi/10.1021/acs.jcim.2c01422"
    )
