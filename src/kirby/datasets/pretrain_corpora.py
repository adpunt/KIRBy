"""Named pretraining corpora for cross-species transfer learning (T-PT.1).

Assembles the six corpora defined in `docs/cross_species_pretraining_design.md`
§3 from existing per-species ChEMBL and Polaris microsomes loaders. Output
schema matches `chembl_bioactivity.download_chembl_bioactivity`:

    columns: smiles, pActivity, target_chembl_id, inchikey, organism

so that `pretrain_chembl_dnn.py` can swap in a corpus name without touching
its training loop.

Corpora
-------
human_chembl         : chembl_bioactivity(organism='human')                    [PT-α]
rodent_chembl        : union of organism ∈ {rat, mouse}                        [PT-γ]
mammalian_chembl     : union of organism ∈ {human, rat, mouse, dog, cow, horse}
multispecies_chembl  : alias of mammalian_chembl; downstream model is expected
                        to consume the `organism` column as a species token   [PT-β, PT-ζ]
human_microsomes     : Polaris ADME-Fang HLM only                              [PT-δ baseline]
microsome_all        : Polaris HLM + RLM (DLM/FLM gated on X-Data.5/6)         [PT-δ]

Leakage prevention is applied at the union level: every corpus accepts an
`exclude_inchikeys` set and rows with matching InChIKeys are dropped.

CLI
---
    python -m kirby.datasets.pretrain_corpora --list
    python -m kirby.datasets.pretrain_corpora --corpus rodent_chembl --smoke
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from kirby.datasets.chembl_bioactivity import download_chembl_bioactivity
from kirby.utils.species import SPECIES_REGISTRY


def _default_chembl_dir() -> str:
    """ChEMBL corpus dir, preferring Mac seagate then $KIRBY_CACHE_DIR/chembl."""
    seagate = Path('/Volumes/seagate/kirby/chembl')
    if seagate.parent.parent.exists():
        return str(seagate)
    root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return str(root / 'chembl')


CHEMBL_SCHEMA = ['smiles', 'pActivity', 'target_chembl_id', 'inchikey', 'organism']

CHEMBL_SPECIES_HUMAN = ['human']
CHEMBL_SPECIES_RODENT = ['rat', 'mouse']
CHEMBL_SPECIES_MAMMALIAN = ['human', 'rat', 'mouse', 'dog', 'cow', 'horse']

CORPUS_DEFINITIONS: Dict[str, dict] = {
    'human_chembl':        {'kind': 'chembl', 'species': CHEMBL_SPECIES_HUMAN},
    'rodent_chembl':       {'kind': 'chembl', 'species': CHEMBL_SPECIES_RODENT},
    'mammalian_chembl':    {'kind': 'chembl', 'species': CHEMBL_SPECIES_MAMMALIAN},
    'multispecies_chembl': {'kind': 'chembl', 'species': CHEMBL_SPECIES_MAMMALIAN},
    'human_microsomes':    {'kind': 'microsomes', 'sources': ['adme_hlm']},
    'microsome_all':       {'kind': 'microsomes', 'sources': ['adme_hlm', 'adme_rlm']},
}

POLARIS_SOURCE_ORGANISM = {
    'adme_hlm': 'human',
    'adme_rlm': 'rat',
}

POLARIS_SYNTHETIC_TARGET = {
    'adme_hlm': 'POLARIS_HLM',
    'adme_rlm': 'POLARIS_RLM',
}


def list_corpora() -> List[str]:
    return list(CORPUS_DEFINITIONS.keys())


def _compose_chembl(
    species: List[str],
    max_targets: int,
    max_compounds_per_species: int,
    data_dir: str,
    exclude_inchikeys: Optional[Set[str]],
    skip_missing: bool,
) -> pd.DataFrame:
    frames = []
    for sp in species:
        if sp not in SPECIES_REGISTRY:
            raise ValueError(
                f"Species '{sp}' not in SPECIES_REGISTRY. "
                f"Add it to src/kirby/utils/species.py first."
            )
        try:
            df = download_chembl_bioactivity(
                data_dir=data_dir,
                max_targets=max_targets,
                max_compounds=max_compounds_per_species,
                exclude_inchikeys=exclude_inchikeys,
                organism=sp,
            )
        except RuntimeError as e:
            if skip_missing:
                print(f"  [pretrain_corpora] skipping {sp}: {e}")
                continue
            raise
        if len(df) == 0:
            print(f"  [pretrain_corpora] {sp}: 0 rows after filtering")
            continue
        frames.append(df[CHEMBL_SCHEMA])

    if not frames:
        return pd.DataFrame(columns=CHEMBL_SCHEMA)
    return pd.concat(frames, axis=0, ignore_index=True)


def _compose_microsomes(
    sources: List[str],
    exclude_inchikeys: Optional[Set[str]],
) -> pd.DataFrame:
    """Wrap Polaris ADME-Fang HLM/RLM endpoints into the ChEMBL schema."""
    import sys
    repo_tests = Path(__file__).resolve().parents[3] / 'tests'
    if str(repo_tests) not in sys.path:
        sys.path.insert(0, str(repo_tests))
    from find_best_hybrid_molecular import _load_polaris_adme

    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    frames = []
    for src in sources:
        if src not in POLARIS_SOURCE_ORGANISM:
            raise ValueError(
                f"Unsupported microsomes source '{src}'. "
                f"Known: {sorted(POLARIS_SOURCE_ORGANISM)}"
            )
        smiles, labels = _load_polaris_adme(src)
        ikeys = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            k = None
            if mol is not None:
                try:
                    inchi = MolToInchi(mol)
                    k = InchiToInchiKey(inchi) if inchi else None
                except Exception:
                    k = None
            ikeys.append(k)
        df = pd.DataFrame({
            'smiles': smiles,
            'pActivity': np.asarray(labels, dtype=float),
            'target_chembl_id': POLARIS_SYNTHETIC_TARGET[src],
            'inchikey': ikeys,
            'organism': POLARIS_SOURCE_ORGANISM[src],
        })
        df = df.dropna(subset=['inchikey'])
        if exclude_inchikeys:
            n_before = len(df)
            df = df[~df['inchikey'].isin(exclude_inchikeys)]
            n_removed = n_before - len(df)
            if n_removed:
                print(f"  [pretrain_corpora] {src}: excluded {n_removed} "
                      f"rows by InChIKey leakage filter")
        frames.append(df[CHEMBL_SCHEMA])

    if not frames:
        return pd.DataFrame(columns=CHEMBL_SCHEMA)
    return pd.concat(frames, axis=0, ignore_index=True)


def compose(
    name: str,
    exclude_inchikeys: Optional[Set[str]] = None,
    max_compounds_per_species: int = 500_000,
    max_targets: int = 200,
    data_dir: Optional[str] = None,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Assemble a named pretraining corpus.

    Args:
        name: One of `list_corpora()`.
        exclude_inchikeys: Optional set of InChIKeys to drop (leakage prevention
            against downstream vet eval sets). Forwarded to each per-species
            ChEMBL fetch and applied again to Polaris sources.
        max_compounds_per_species: Per-species subsample cap; passed straight
            to `download_chembl_bioactivity.max_compounds`. Total corpus size
            is roughly len(species) × this cap minus duplicate compounds
            across species.
        max_targets: Top-N targets per species (per `download_chembl_bioactivity`).
        data_dir: ChEMBL cache directory (per-species CSV files live here).
        skip_missing: If True (default), skip species whose ChEMBL query returns
            zero targets rather than raising. The cat row count (n=59) makes
            this safety net real.

    Returns:
        DataFrame with columns: smiles, pActivity, target_chembl_id,
        inchikey, organism. Empty DataFrame if all sources failed/were skipped.
    """
    if data_dir is None:
        data_dir = _default_chembl_dir()
    if name not in CORPUS_DEFINITIONS:
        raise ValueError(
            f"Unknown corpus '{name}'. Known: {sorted(CORPUS_DEFINITIONS)}"
        )
    spec = CORPUS_DEFINITIONS[name]
    kind = spec['kind']

    if kind == 'chembl':
        return _compose_chembl(
            species=spec['species'],
            max_targets=max_targets,
            max_compounds_per_species=max_compounds_per_species,
            data_dir=data_dir,
            exclude_inchikeys=exclude_inchikeys,
            skip_missing=skip_missing,
        )
    if kind == 'microsomes':
        return _compose_microsomes(
            sources=spec['sources'],
            exclude_inchikeys=exclude_inchikeys,
        )
    raise AssertionError(f"Unhandled corpus kind '{kind}' for '{name}'")


def collect_vet_inchikeys(
    species: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    max_targets: int = 200,
) -> Set[str]:
    """Collect InChIKeys from any cached per-species ChEMBL files for vet species.

    Best-effort: reads only files already on disk; does NOT trigger network
    fetches. Use the return value as `exclude_inchikeys` to keep vet evaluation
    compounds out of pretraining corpora.

    Returns a set of InChIKey strings (empty if no caches exist).
    """
    if species is None:
        species = ['dog', 'cat', 'cow', 'horse']
    if data_dir is None:
        data_dir = _default_chembl_dir()
    keys: Set[str] = set()
    for sp in species:
        cache_file = os.path.join(
            data_dir, f'chembl_bioactivity_top{max_targets}_{sp}.csv'
        )
        if not os.path.exists(cache_file):
            print(f"  [collect_vet_inchikeys] no cache for {sp}: {cache_file}")
            continue
        try:
            df = pd.read_csv(cache_file, usecols=['inchikey'])
            n_before = len(keys)
            keys.update(k for k in df['inchikey'].dropna().unique())
            print(f"  [collect_vet_inchikeys] {sp}: +{len(keys) - n_before} keys")
        except Exception as e:
            print(f"  [collect_vet_inchikeys] failed to read {cache_file}: {e}")
    print(f"  [collect_vet_inchikeys] total: {len(keys)} unique vet InChIKeys")
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', action='store_true',
                    help='List known corpus names and exit.')
    ap.add_argument('--corpus', type=str, default=None,
                    help='Corpus name to assemble.')
    ap.add_argument('--smoke', action='store_true',
                    help='Smoke test: cap per-species to 5,000 compounds, '
                         'top 10 targets, no network on cat/horse if missing.')
    ap.add_argument('--max-compounds-per-species', type=int, default=500_000)
    ap.add_argument('--max-targets', type=int, default=200)
    ap.add_argument('--data-dir', type=str,
                    default='/Volumes/seagate/kirby/chembl')
    ap.add_argument('--vet-leakage', action='store_true',
                    help='Filter via collect_vet_inchikeys() before assembly.')
    args = ap.parse_args()

    if args.list:
        print("Available corpora:")
        for name in list_corpora():
            spec = CORPUS_DEFINITIONS[name]
            detail = (', '.join(spec.get('species', spec.get('sources', []))))
            print(f"  {name:22s} [{spec['kind']:10s}] {detail}")
        return

    if args.corpus is None:
        ap.error("Pass --corpus <name> or --list.")

    if args.smoke:
        args.max_compounds_per_species = 5_000
        args.max_targets = 10

    exclude = collect_vet_inchikeys(data_dir=args.data_dir,
                                     max_targets=args.max_targets) \
        if args.vet_leakage else None

    df = compose(
        name=args.corpus,
        exclude_inchikeys=exclude,
        max_compounds_per_species=args.max_compounds_per_species,
        max_targets=args.max_targets,
        data_dir=args.data_dir,
    )

    print(f"\nCorpus '{args.corpus}': {len(df)} rows")
    if len(df) == 0:
        return
    print(f"  Unique compounds: {df['inchikey'].nunique()}")
    print(f"  Targets:          {df['target_chembl_id'].nunique()}")
    print(f"  Species breakdown:")
    for sp, n in df.groupby('organism').size().items():
        print(f"    {sp:8s}: {n} rows")
    print(f"  pActivity:        "
          f"mean {df['pActivity'].mean():.2f} ± {df['pActivity'].std():.2f}")


if __name__ == '__main__':
    main()
