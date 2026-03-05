"""Reaction yield prediction datasets: Buchwald-Hartwig and Suzuki-Miyaura.

Standard benchmarks for evaluating reaction-aware representations.
Data from: rxn4chemistry/rxn_yields (Schwaller et al. 2021).

Buchwald-Hartwig: ~3955 C-N coupling reactions (Ahneman et al. Science 2018)
Suzuki-Miyaura:  ~5760 cross-coupling reactions (Perera et al. Science 2018)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def _cache_dir() -> Path:
    seagate = Path('/Volumes/seagate/kirby/reaction_yields')
    if seagate.parent.parent.exists():
        return seagate
    return Path.home() / '.cache' / 'kirby' / 'reaction_yields'


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
        splits: List[Tuple[np.ndarray, np.ndarray]] — (train_idx, test_idx)
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

    # Random 70/30 splits (matches original protocol)
    splits = []
    for i in range(n_splits):
        rng = np.random.RandomState(42 + i)
        perm = rng.permutation(n)
        n_train = int(0.7 * n)
        splits.append((perm[:n_train], perm[n_train:]))

    print(f"  Buchwald-Hartwig: {n} reactions, yield range "
          f"[{labels.min():.1f}, {labels.max():.1f}]")

    return {
        'rxn_smiles': rxn_smiles,
        'labels': labels,
        'splits': splits,
        'components': components,
        'task_type': 'regression',
        'metadata': {
            'name': 'buchwald_hartwig',
            'n_reactions': n,
            'description': 'Buchwald-Hartwig C-N coupling yields '
                           '(Ahneman et al. Science 2018)',
        },
    }


def load_suzuki(n_splits: int = 10, data_dir=None) -> Dict:
    """Load Suzuki-Miyaura coupling yield dataset (~5760 reactions).

    Returns dict with:
        rxn_smiles: List[str] — full reaction SMILES
        labels: np.ndarray — yields (0-100, scaled from 0-1 in source)
        splits: List[Tuple[np.ndarray, np.ndarray]] — (train_idx, test_idx)
        components: List[Dict] — parsed from reaction SMILES
        metadata: dict
    """
    cache = Path(data_dir) if data_dir else _cache_dir()
    suzuki_dir = cache / 'suzuki'

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
            # Fallback: random split if SMILES don't match exactly
            rng = np.random.RandomState(42 + i)
            perm = rng.permutation(n)
            splits.append((perm[:n_train], perm[n_train:]))
        else:
            splits.append((shuffled_idx[:n_train], shuffled_idx[n_train:]))

    # Parse components from reaction SMILES
    components = _parse_rxn_components(rxn_smiles)

    print(f"  Suzuki-Miyaura: {n} reactions, yield range "
          f"[{labels.min():.1f}, {labels.max():.1f}]")

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
