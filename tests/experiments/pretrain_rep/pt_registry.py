"""PT-Rep Phase 0 — pre-training corpus registry.

Single entry point `get_pt_dataset(domain, source, task=None, ...)` returns a
`PTDataset` carrying the *raw* entities (SMILES / antibody sequences), labels,
groups and dedup keys for one named pre-training corpus. Representation
computation is deliberately NOT done here — that is the experiment harness's
job (Phase 1+). This module only resolves "which data" and hands back a clean,
uniform container.

Design notes
------------
* Network-dependent loaders are imported lazily inside each loader function so
  this module imports (and its `synthetic` source runs) with no network and no
  heavy deps.
* Every PT source declares its `domain`, `kind` (labelled / unlabelled), and a
  `task` axis where a source exposes more than one label column.
* `synthetic` sources exist for both domains so the Phase 1 harness can be
  bug-tested offline before fanning out to real corpora (roadmap §6.5).

The roadmap signature is `get_pt_dataset(domain, source, task)` returning
`(X_pt, y_pt, groups_pt)`; here `X_pt` is the raw entity list rather than a
feature matrix, because base-rep computation is a separate, heavy step that the
experiment owns. `PTDataset.as_tuple()` gives the `(entities, labels, groups)`
triple for callers that want the positional form.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow `from experiments...` style imports when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# Container
# =============================================================================

@dataclass
class PTDataset:
    """Raw pre-training corpus, uniform across domains.

    entities     : SMILES (molecular) or heavy-chain sequences (antibody). This
                   is the canonical per-sample string used for dedup + reps.
    heavy_seqs / light_seqs : antibody only; `entities` aliases `heavy_seqs`.
    smiles       : molecular only; aliases `entities`.
    labels       : (n,) float/int array, or None for unlabelled corpora.
    groups       : (n,) array for group-aware splitting, or None.
    dedup_keys   : (n,) list of str — InChIKey (molecular) or exact sequence
                   (antibody). Used by pt_leakage + pt_quality.
    task_type    : 'regression' | 'classification' | 'unlabelled'.
    """
    domain: str
    source: str
    task: Optional[str]
    entities: List[str]
    dedup_keys: List[str]
    labels: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    task_type: str = 'unlabelled'
    heavy_seqs: Optional[List[str]] = None
    light_seqs: Optional[List[str]] = None
    smiles: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.entities)
        if len(self.dedup_keys) != n:
            raise ValueError(
                f"dedup_keys length {len(self.dedup_keys)} != entities {n}")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError(
                f"labels length {len(self.labels)} != entities {n}")
        if self.groups is not None and len(self.groups) != n:
            raise ValueError(
                f"groups length {len(self.groups)} != entities {n}")
        if self.labels is None and self.task_type != 'unlabelled':
            raise ValueError(
                f"task_type={self.task_type!r} but labels is None")

    def __len__(self) -> int:
        return len(self.entities)

    def as_tuple(self) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """The roadmap's `(X_pt, y_pt, groups_pt)` positional form."""
        return self.entities, self.labels, self.groups

    def subsample(self, max_samples: int, seed: int = 42) -> 'PTDataset':
        """Random-shuffled cap. Shuffle-before-anything is a roadmap requirement
        (§6.2): rows get shuffled here so any later group split is order-free.
        The split itself stays group-aware/non-random — this only touches the
        size cap."""
        n = len(self)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        if max_samples is not None and max_samples < n:
            idx = idx[:max_samples]
        return PTDataset(
            domain=self.domain, source=self.source, task=self.task,
            entities=[self.entities[i] for i in idx],
            dedup_keys=[self.dedup_keys[i] for i in idx],
            labels=None if self.labels is None else self.labels[idx],
            groups=None if self.groups is None else self.groups[idx],
            task_type=self.task_type,
            heavy_seqs=None if self.heavy_seqs is None
            else [self.heavy_seqs[i] for i in idx],
            light_seqs=None if self.light_seqs is None
            else [self.light_seqs[i] for i in idx],
            smiles=None if self.smiles is None
            else [self.smiles[i] for i in idx],
            meta={**self.meta, 'subsampled_from': n, 'subsample_seed': seed},
        )


# =============================================================================
# Source registry
# =============================================================================

@dataclass
class PTSource:
    domain: str               # 'molecular' | 'antibody'
    kind: str                 # 'labelled' | 'unlabelled' | 'synthetic'
    loader: Callable          # (task, max_samples, exclude_keys) -> PTDataset
    tasks: Optional[List[str]] = None   # None => single implicit task
    note: str = ''


def _synthetic_molecular(task, max_samples, exclude_keys) -> PTDataset:
    """Tiny offline molecular corpus for pipeline bug-catching. No network."""
    n = max_samples or 64
    rng = np.random.RandomState(0)
    # A small fixed SMILES alphabet; not chemically meaningful, just parseable
    # by RDKit-free string handling. Real reps are computed downstream.
    frags = ['C', 'CC', 'CCO', 'c1ccccc1', 'CN', 'CCN', 'CO', 'CCC']
    smiles = [''.join(rng.choice(frags, rng.randint(1, 4))) for _ in range(n)]
    keys = [f'SYNTH-MOL-{i:06d}-{hash(s) & 0xfffff:05x}' for i, s in enumerate(smiles)]
    labels = rng.normal(6.0, 1.5, size=n).astype(np.float64)
    groups = rng.randint(0, max(2, n // 8), size=n)
    ds = PTDataset(
        domain='molecular', source='synthetic_molecular', task=task,
        entities=list(smiles), dedup_keys=keys, labels=labels, groups=groups,
        task_type='regression', smiles=list(smiles),
        meta={'synthetic': True},
    )
    return _apply_exclude(ds, exclude_keys)


def _synthetic_antibody(task, max_samples, exclude_keys) -> PTDataset:
    """Tiny offline antibody corpus for pipeline bug-catching. No network."""
    n = max_samples or 64
    rng = np.random.RandomState(1)
    aa = list('ACDEFGHIKLMNPQRSTVWY')
    heavy = [''.join(rng.choice(aa, rng.randint(110, 125))) for _ in range(n)]
    light = [''.join(rng.choice(aa, rng.randint(105, 115))) for _ in range(n)]
    keys = list(heavy)  # exact sequence is the dedup key for antibody
    labels = rng.randint(0, 2, size=n).astype(np.int64)
    ds = PTDataset(
        domain='antibody', source='synthetic_antibody', task=task,
        entities=list(heavy), dedup_keys=keys, labels=labels, groups=None,
        task_type='classification', heavy_seqs=list(heavy), light_seqs=light,
        meta={'synthetic': True},
    )
    return _apply_exclude(ds, exclude_keys)


def _load_molecular_corpus(corpus_name):
    """Factory: wrap a `pretrain_corpora` ChEMBL/microsomes corpus."""
    def _loader(task, max_samples, exclude_keys) -> PTDataset:
        from kirby.datasets import pretrain_corpora
        cap = max_samples if max_samples is not None else 500_000
        df = pretrain_corpora.compose(
            name=corpus_name,
            exclude_inchikeys=set(exclude_keys) if exclude_keys else None,
            max_compounds_per_species=cap,
        )
        if len(df) == 0:
            raise RuntimeError(
                f"pretrain corpus '{corpus_name}' returned 0 rows — check the "
                f"ChEMBL cache directory.")
        smiles = df['smiles'].tolist()
        keys = df['inchikey'].astype(str).tolist()
        labels = df['pActivity'].to_numpy(dtype=np.float64)
        groups = df['target_chembl_id'].astype(str).to_numpy()
        ds = PTDataset(
            domain='molecular', source=corpus_name, task=task,
            entities=smiles, dedup_keys=keys, labels=labels, groups=groups,
            task_type='regression', smiles=smiles,
            meta={'corpus': corpus_name,
                  'n_targets': df['target_chembl_id'].nunique()},
        )
        return ds.subsample(max_samples) if max_samples else ds
    return _loader


def _load_oas_species(task, max_samples, exclude_keys) -> PTDataset:
    """OAS-derived species classification (human vs non-human VH)."""
    from kirby.datasets.humanness import load_humanness_dataset
    cap = max_samples if max_samples is not None else 20_000
    d = load_humanness_dataset(preset='species', max_samples=cap)
    heavy = list(d['heavy_seqs']) if 'heavy_seqs' in d else list(d['sequences'])
    labels = np.asarray(d['labels'])
    ds = PTDataset(
        domain='antibody', source='oas_species', task=task,
        entities=heavy, dedup_keys=list(heavy), labels=labels, groups=None,
        task_type='classification', heavy_seqs=heavy,
        meta={'preset': 'species'},
    )
    return _apply_exclude(ds, exclude_keys)


def _load_ginkgo_gdpa1(task, max_samples, exclude_keys) -> PTDataset:
    """Full GDPa1 developability — one regression property per `task`."""
    from kirby.datasets.ginkgo_dev import load_ginkgo_dataset, GINKGO_PROPERTIES
    prop = task or 'ginkgo_hic'
    if prop not in GINKGO_PROPERTIES:
        raise ValueError(
            f"ginkgo_gdpa1 task '{prop}' not in {sorted(GINKGO_PROPERTIES)}")
    d = load_ginkgo_dataset(name=prop)
    heavy = list(d['heavy_seqs'])
    labels = np.asarray(d['labels'], dtype=np.float64)
    groups = np.asarray(d['fold_assignments'])
    ds = PTDataset(
        domain='antibody', source='ginkgo_gdpa1', task=prop,
        entities=heavy, dedup_keys=list(heavy), labels=labels, groups=groups,
        task_type='regression', heavy_seqs=heavy,
        light_seqs=list(d['light_seqs']),
        meta={'property': prop, 'pre_split': True},
    )
    ds = _apply_exclude(ds, exclude_keys)
    return ds.subsample(max_samples) if max_samples else ds


def _load_sabdab(task, max_samples, exclude_keys) -> PTDataset:
    """SAbDab structural antibody sequences — unlabelled PT corpus (for P2c
    contrastive PT, or stand-in labels via pt_quality helpers)."""
    from kirby.datasets.sabdab import load_sabdab_sequences
    cap = max_samples if max_samples is not None else 2_000
    d = load_sabdab_sequences(task='structure', n_samples=cap, paired_only=True)
    heavy = list(d['heavy_seqs'])
    ds = PTDataset(
        domain='antibody', source='sabdab', task=task,
        entities=heavy, dedup_keys=list(heavy), labels=None, groups=None,
        task_type='unlabelled', heavy_seqs=heavy,
        light_seqs=list(d['light_seqs']),
        meta={'task': 'structure', 'pdb_ids': d.get('pdb_ids')},
    )
    return _apply_exclude(ds, exclude_keys)


def _apply_exclude(ds: PTDataset, exclude_keys) -> PTDataset:
    """Drop rows whose dedup_key is in `exclude_keys` (leakage pre-filter).

    pt_leakage is the authoritative leakage module; this is a cheap exact-key
    pre-filter so registry callers can pass FT keys directly. For antibody
    identity-threshold leakage, use pt_leakage.sequence_leakage_filter."""
    if not exclude_keys:
        return ds
    excl = set(exclude_keys)
    keep = [i for i, k in enumerate(ds.dedup_keys) if k not in excl]
    if len(keep) == len(ds):
        return ds
    return PTDataset(
        domain=ds.domain, source=ds.source, task=ds.task,
        entities=[ds.entities[i] for i in keep],
        dedup_keys=[ds.dedup_keys[i] for i in keep],
        labels=None if ds.labels is None else ds.labels[keep],
        groups=None if ds.groups is None else ds.groups[keep],
        task_type=ds.task_type,
        heavy_seqs=None if ds.heavy_seqs is None
        else [ds.heavy_seqs[i] for i in keep],
        light_seqs=None if ds.light_seqs is None
        else [ds.light_seqs[i] for i in keep],
        smiles=None if ds.smiles is None else [ds.smiles[i] for i in keep],
        meta={**ds.meta, 'exclude_removed': len(ds) - len(keep)},
    )


PT_SOURCES: Dict[str, PTSource] = {
    # --- offline pipeline-test sources --------------------------------------
    'synthetic_molecular': PTSource(
        domain='molecular', kind='synthetic', loader=_synthetic_molecular,
        note='Offline tiny molecular corpus for bug-catching.'),
    'synthetic_antibody': PTSource(
        domain='antibody', kind='synthetic', loader=_synthetic_antibody,
        note='Offline tiny antibody corpus for bug-catching.'),
    # --- molecular ----------------------------------------------------------
    'human_chembl': PTSource(
        domain='molecular', kind='labelled',
        loader=_load_molecular_corpus('human_chembl'),
        note='ChEMBL IC50, human targets.'),
    'rodent_chembl': PTSource(
        domain='molecular', kind='labelled',
        loader=_load_molecular_corpus('rodent_chembl'),
        note='ChEMBL IC50, rat+mouse targets.'),
    'mammalian_chembl': PTSource(
        domain='molecular', kind='labelled',
        loader=_load_molecular_corpus('mammalian_chembl'),
        note='ChEMBL IC50, 6-species union.'),
    'human_microsomes': PTSource(
        domain='molecular', kind='labelled',
        loader=_load_molecular_corpus('human_microsomes'),
        note='Polaris ADME-Fang HLM.'),
    'microsome_all': PTSource(
        domain='molecular', kind='labelled',
        loader=_load_molecular_corpus('microsome_all'),
        note='Polaris ADME-Fang HLM+RLM.'),
    # --- antibody -----------------------------------------------------------
    'oas_species': PTSource(
        domain='antibody', kind='labelled', loader=_load_oas_species,
        note='OAS-derived human-vs-non-human VH classification.'),
    'ginkgo_gdpa1': PTSource(
        domain='antibody', kind='labelled', loader=_load_ginkgo_gdpa1,
        tasks=None,  # filled lazily from GINKGO_PROPERTIES; see get_pt_dataset
        note='Full GDPa1 developability; task selects the property column.'),
    'sabdab': PTSource(
        domain='antibody', kind='unlabelled', loader=_load_sabdab,
        note='SAbDab structural sequences; unlabelled (P2c / stand-in labels).'),
}


def list_pt_sources(domain: Optional[str] = None) -> List[str]:
    """Names of registered PT sources, optionally filtered by domain."""
    if domain is None:
        return sorted(PT_SOURCES)
    return sorted(n for n, s in PT_SOURCES.items() if s.domain == domain)


def get_pt_dataset(
    domain: str,
    source: str,
    task: Optional[str] = None,
    max_samples: Optional[int] = None,
    exclude_keys: Optional[Sequence[str]] = None,
) -> PTDataset:
    """Resolve a named pre-training corpus to a `PTDataset`.

    Args:
        domain: 'molecular' or 'antibody'. Must match the source's domain.
        source: A key of `PT_SOURCES` (see `list_pt_sources()`).
        task: Sub-task selector for multi-label sources (e.g. a GDPa1 property
            name for 'ginkgo_gdpa1'). None for single-task sources.
        max_samples: Optional size cap. Applied with a shuffled draw (roadmap
            §6.2 — test several cap sizes). None = full corpus.
        exclude_keys: Optional iterable of dedup keys (InChIKeys / sequences)
            to drop up front — a cheap exact-match leakage pre-filter. For
            antibody identity-threshold leakage use `pt_leakage`.

    Returns:
        PTDataset with raw entities, labels, groups and dedup keys.
    """
    if source not in PT_SOURCES:
        raise KeyError(
            f"Unknown PT source '{source}'. "
            f"Known for domain={domain!r}: {list_pt_sources(domain)}")
    spec = PT_SOURCES[source]
    if spec.domain != domain:
        raise ValueError(
            f"PT source '{source}' is domain={spec.domain!r}, "
            f"but get_pt_dataset was called with domain={domain!r}.")
    ds = spec.loader(task, max_samples, exclude_keys)
    # Defensive: loaders must honour the requested domain.
    if ds.domain != domain:
        raise AssertionError(
            f"loader for '{source}' returned domain={ds.domain!r}")
    return ds


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list', action='store_true', help='List PT sources.')
    ap.add_argument('--domain', default=None)
    ap.add_argument('--source', default=None)
    ap.add_argument('--task', default=None)
    ap.add_argument('--max-samples', type=int, default=None)
    args = ap.parse_args()

    if args.list or not args.source:
        for name in list_pt_sources(args.domain):
            s = PT_SOURCES[name]
            print(f"  {name:22s} [{s.domain:10s} {s.kind:10s}] {s.note}")
        return
    ds = get_pt_dataset(args.domain or PT_SOURCES[args.source].domain,
                        args.source, task=args.task,
                        max_samples=args.max_samples)
    print(f"\n{ds.source}: n={len(ds)}  task_type={ds.task_type}  "
          f"task={ds.task}")
    print(f"  labels: {'none' if ds.labels is None else ds.labels.shape}")
    print(f"  groups: {'none' if ds.groups is None else len(set(ds.groups))} "
          f"unique")
    print(f"  meta: {ds.meta}")


if __name__ == '__main__':
    _cli()
