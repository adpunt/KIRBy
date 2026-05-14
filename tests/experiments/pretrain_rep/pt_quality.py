"""PT-Rep Phase 0 — two-layer PT corpus quality pipeline.

Roadmap §6.3: both layers are mandatory and the criteria must be written down
per corpus (publication-facing). This module is that written-down record plus
the code that enforces it.

Layer 1 — basic dedup hygiene
    Remove exact-duplicate entities and obvious junk (empty/invalid strings,
    non-standard alphabet). Cheap, always safe.

Layer 2 — explicit quality filter
    Domain-specific sanity checks:
      * molecular  : label (pActivity) within a plausible range; drop
                     contradictory measurements — the same compound/target
                     pair carrying labels that disagree beyond a tolerance.
      * antibody   : sequence length within a plausible chain-length window;
                     reject ambiguous residues (X, *, gaps).

`apply_quality_pipeline(ds)` runs both layers, returns the filtered `PTDataset`
and a report dict. `write_quality_report()` serialises that report to JSON so a
publication can cite exactly what was filtered and why.

The per-corpus criteria live in `QUALITY_CRITERIA`; a corpus with no explicit
entry falls back to the domain default. To change what a corpus does, edit the
dict here — that keeps the audit trail in one reviewable place.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from experiments.pretrain_rep.pt_registry import PTDataset


STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


# =============================================================================
# Criteria — the written-down, publication-facing record
# =============================================================================

@dataclass
class QualityCriteria:
    """One corpus's quality rules. Every field is a documented decision."""
    # Layer 1
    # 'entity'     : exact-duplicate entity strings (antibody — one sequence
    #                is one sample).
    # 'row'        : exact-duplicate (dedup_key, group, label) tuples
    #                (molecular — a sample is a compound/target measurement;
    #                the same compound legitimately recurs across targets, and
    #                divergent measurements must survive to Layer 2's
    #                contradiction check rather than being silently deduped).
    dedup_on: str = 'row'               # 'entity' or 'row'
    drop_empty: bool = True
    require_alphabet: Optional[str] = None  # 'aa' for antibody; None = skip
    # Layer 2 — molecular
    label_min: Optional[float] = None
    label_max: Optional[float] = None
    contradiction_tol: Optional[float] = None  # max label spread within a
    #                                            (dedup_key, group) cluster
    # Layer 2 — antibody
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    reject_ambiguous: bool = False      # drop seqs with non-standard residues
    note: str = ''


# Domain defaults. pActivity range: pIC50/pChEMBL realistically sits in ~2-12;
# 0-14 is a generous sanity window that only catches unit/parse errors.
# Antibody VH/VL chains are ~100-135 residues; 90-150 is a generous window.
DOMAIN_DEFAULTS: Dict[str, QualityCriteria] = {
    'molecular': QualityCriteria(
        dedup_on='row', drop_empty=True, require_alphabet=None,
        label_min=0.0, label_max=14.0, contradiction_tol=1.0,
        note='pActivity sanity window 0-14; contradictory (compound,target) '
             'measurements with spread > 1.0 log unit are dropped.'),
    'antibody': QualityCriteria(
        dedup_on='entity', drop_empty=True, require_alphabet='aa',
        min_length=90, max_length=150, reject_ambiguous=True,
        note='Standard 20-AA alphabet only; chain length window 90-150; '
             'ambiguous residues (X/*/gap) rejected.'),
}

# Per-corpus overrides. A corpus absent here uses its domain default.
QUALITY_CRITERIA: Dict[str, QualityCriteria] = {
    # Synthetic corpora: keep hygiene, skip range/contradiction checks (the
    # generators are already clean and the synthetic labels are arbitrary).
    'synthetic_molecular': QualityCriteria(
        dedup_on='row', drop_empty=True,
        note='Synthetic offline corpus — hygiene only, no label-range filter.'),
    'synthetic_antibody': QualityCriteria(
        dedup_on='entity', drop_empty=True, require_alphabet='aa',
        min_length=90, max_length=150, reject_ambiguous=True,
        note='Synthetic offline corpus — alphabet+length enforced, no label '
             'filter (labels are arbitrary).'),
    # ChEMBL corpora are already (inchikey,target) geo-mean deduped upstream
    # in chembl_bioactivity; the contradiction check here is a second pass
    # that also catches cross-fetch duplicates from the species union.
    'human_chembl': DOMAIN_DEFAULTS['molecular'],
    'rodent_chembl': DOMAIN_DEFAULTS['molecular'],
    'mammalian_chembl': DOMAIN_DEFAULTS['molecular'],
    # Polaris microsomes report log-scaled clearance; the 0-14 window is wrong
    # for them, so widen and skip the contradiction check (single curated value
    # per compound already).
    'human_microsomes': QualityCriteria(
        dedup_on='row', drop_empty=True,
        label_min=-6.0, label_max=8.0, contradiction_tol=None,
        note='Polaris HLM log-clearance; wide sanity window, no contradiction '
             'check (one curated value per compound).'),
    'microsome_all': QualityCriteria(
        dedup_on='row', drop_empty=True,
        label_min=-6.0, label_max=8.0, contradiction_tol=None,
        note='Polaris HLM+RLM log-clearance; wide sanity window, no '
             'contradiction check.'),
    'oas_species': DOMAIN_DEFAULTS['antibody'],
    'ginkgo_gdpa1': QualityCriteria(
        dedup_on='entity', drop_empty=True, require_alphabet='aa',
        min_length=90, max_length=150, reject_ambiguous=True,
        note='GDPa1 VH sequences; alphabet+length enforced. Label is a '
             'continuous developability property — no range filter (ranges '
             'differ per property and are already curated).'),
    'sabdab': DOMAIN_DEFAULTS['antibody'],
}


def get_criteria(source: str, domain: str) -> QualityCriteria:
    """Resolve the criteria for a corpus: explicit override, else domain default."""
    if source in QUALITY_CRITERIA:
        return QUALITY_CRITERIA[source]
    if domain not in DOMAIN_DEFAULTS:
        raise KeyError(f"No quality default for domain '{domain}'")
    return DOMAIN_DEFAULTS[domain]


# =============================================================================
# Report
# =============================================================================

@dataclass
class QualityReport:
    source: str
    domain: str
    criteria: dict
    n_in: int
    n_out: int
    stages: List[dict] = field(default_factory=list)

    def _stage(self, name: str, n_before: int, n_after: int, detail: str = ''):
        self.stages.append({
            'stage': name, 'n_before': n_before, 'n_after': n_after,
            'n_removed': n_before - n_after, 'detail': detail,
        })

    def to_dict(self) -> dict:
        return asdict(self)


def write_quality_report(report: QualityReport, path: str) -> str:
    """Serialise a QualityReport to JSON (publication-facing audit trail)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


# =============================================================================
# Pipeline
# =============================================================================

def _select(ds: PTDataset, keep_idx: List[int]) -> PTDataset:
    return PTDataset(
        domain=ds.domain, source=ds.source, task=ds.task,
        entities=[ds.entities[i] for i in keep_idx],
        dedup_keys=[ds.dedup_keys[i] for i in keep_idx],
        labels=None if ds.labels is None else ds.labels[keep_idx],
        groups=None if ds.groups is None else ds.groups[keep_idx],
        task_type=ds.task_type,
        heavy_seqs=None if ds.heavy_seqs is None
        else [ds.heavy_seqs[i] for i in keep_idx],
        light_seqs=None if ds.light_seqs is None
        else [ds.light_seqs[i] for i in keep_idx],
        smiles=None if ds.smiles is None else [ds.smiles[i] for i in keep_idx],
        meta=dict(ds.meta),
    )


def _layer1_dedup_hygiene(ds: PTDataset, crit: QualityCriteria,
                          report: QualityReport) -> PTDataset:
    """Exact-duplicate removal + empty/invalid-string + alphabet check."""
    n0 = len(ds)

    # 1a. drop empty / whitespace entities
    if crit.drop_empty:
        keep = [i for i, e in enumerate(ds.entities)
                if isinstance(e, str) and e.strip()]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('drop_empty', n_before, len(ds))

    # 1b. alphabet check (antibody: standard 20 AA only)
    if crit.require_alphabet == 'aa':
        keep = [i for i, e in enumerate(ds.entities)
                if set(e.upper()) <= STANDARD_AA]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('alphabet_check', n_before, len(ds),
                      detail='standard 20-AA alphabet')

    # 1c. exact-duplicate removal
    #   'entity' : one sequence == one sample (antibody).
    #   'row'    : a row is unique on (dedup_key, group, label) — preserves the
    #              same compound across targets, and keeps divergent
    #              measurements for Layer 2's contradiction check.
    if crit.dedup_on == 'entity':
        row_keys = list(ds.entities)
    elif crit.dedup_on == 'row':
        groups = ds.groups if ds.groups is not None else [None] * len(ds)
        labels = ds.labels if ds.labels is not None else [None] * len(ds)
        row_keys = [(ds.dedup_keys[i], groups[i], labels[i])
                    for i in range(len(ds))]
    else:
        raise ValueError(
            f"dedup_on must be 'entity' or 'row', got {crit.dedup_on!r}")
    seen = set()
    keep = []
    for i, k in enumerate(row_keys):
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    n_before = len(ds)
    ds = _select(ds, keep)
    report._stage('exact_dedup', n_before, len(ds),
                  detail=f'dedup_on={crit.dedup_on}')

    report._stage('layer1_total', n0, len(ds), detail='basic dedup hygiene')
    return ds


def _layer2_quality_filter(ds: PTDataset, crit: QualityCriteria,
                           report: QualityReport) -> PTDataset:
    """Domain-specific sanity checks (label range, contradictions, length)."""
    n0 = len(ds)

    # 2a. label range
    if (crit.label_min is not None or crit.label_max is not None) \
            and ds.labels is not None:
        lo = -np.inf if crit.label_min is None else crit.label_min
        hi = np.inf if crit.label_max is None else crit.label_max
        lab = np.asarray(ds.labels, dtype=float)
        keep = [i for i in range(len(ds))
                if lo <= lab[i] <= hi and np.isfinite(lab[i])]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('label_range', n_before, len(ds),
                      detail=f'[{lo}, {hi}]')

    # 2b. contradictory measurements within a (dedup_key, group) cluster
    if crit.contradiction_tol is not None and ds.labels is not None:
        groups = ds.groups if ds.groups is not None else np.zeros(len(ds))
        clusters: Dict[tuple, List[int]] = {}
        for i in range(len(ds)):
            clusters.setdefault((ds.dedup_keys[i], groups[i]), []).append(i)
        drop = set()
        n_contradictory = 0
        for idxs in clusters.values():
            if len(idxs) < 2:
                continue
            vals = np.asarray([ds.labels[i] for i in idxs], dtype=float)
            if float(vals.max() - vals.min()) > crit.contradiction_tol:
                drop.update(idxs)
                n_contradictory += 1
        keep = [i for i in range(len(ds)) if i not in drop]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('contradiction_filter', n_before, len(ds),
                      detail=f'{n_contradictory} clusters with spread > '
                             f'{crit.contradiction_tol}')

    # 2c. sequence-length window (antibody)
    if crit.min_length is not None or crit.max_length is not None:
        lo = 0 if crit.min_length is None else crit.min_length
        hi = 10 ** 9 if crit.max_length is None else crit.max_length
        keep = [i for i, e in enumerate(ds.entities) if lo <= len(e) <= hi]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('length_window', n_before, len(ds),
                      detail=f'[{lo}, {hi}]')

    # 2d. ambiguous-residue rejection (redundant with 1b alphabet check when
    # that runs, but kept explicit for corpora that skip the alphabet check).
    if crit.reject_ambiguous:
        keep = [i for i, e in enumerate(ds.entities)
                if set(e.upper()) <= STANDARD_AA]
        n_before = len(ds)
        ds = _select(ds, keep)
        report._stage('reject_ambiguous', n_before, len(ds))

    report._stage('layer2_total', n0, len(ds), detail='explicit quality filter')
    return ds


def apply_quality_pipeline(
    ds: PTDataset,
    criteria: Optional[QualityCriteria] = None,
    report_path: Optional[str] = None,
) -> tuple[PTDataset, QualityReport]:
    """Run both quality layers on a PT corpus.

    Args:
        ds: Raw `PTDataset` from `pt_registry.get_pt_dataset`.
        criteria: Override the resolved per-corpus criteria. Default: looks up
            `QUALITY_CRITERIA[ds.source]`, falling back to the domain default.
        report_path: If given, write the JSON audit report there.

    Returns:
        (filtered_ds, report). `report` records criteria + per-stage counts.
    """
    crit = criteria or get_criteria(ds.source, ds.domain)
    report = QualityReport(
        source=ds.source, domain=ds.domain, criteria=asdict(crit),
        n_in=len(ds), n_out=len(ds))

    ds = _layer1_dedup_hygiene(ds, crit, report)
    ds = _layer2_quality_filter(ds, crit, report)

    report.n_out = len(ds)
    if report_path:
        write_quality_report(report, report_path)
    return ds, report
