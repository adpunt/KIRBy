"""PT-Rep Phase 0 — PT/FT leakage removal.

If the same compound or antibody appears in both the pre-training corpus and a
fine-tuning target, any PT→FT improvement is partly memorisation, not transfer.
This module strips those overlaps out of the PT corpus before it is used.

Molecular (roadmap §6.3, RESOLVED)
    InChIKey exact match. A PT compound whose InChIKey is in the FT set is
    dropped. Deterministic, no tuning.

Antibody (roadmap §6.3, TBD — see §9 research-prompt Q6)
    The correct sequence-identity threshold (full-VH vs CDR-only, 95% / 50% /
    30%) is an open research question. Until that lands, the default here is
    `identity_threshold=1.0` — exact-sequence match only, the safe floor that
    can never over-remove. A sub-1.0 threshold is supported but must be passed
    explicitly, and a warning is emitted so an unvalidated threshold is never
    applied silently.

`remove_pt_ft_leakage()` is the domain-dispatching entry point and returns the
filtered `PTDataset` plus a small report dict.
"""
from __future__ import annotations

import difflib
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from experiments.pretrain_rep.pt_registry import PTDataset


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


# =============================================================================
# Molecular — InChIKey exact match
# =============================================================================

def inchikey_leakage_filter(
    pt_ds: PTDataset,
    ft_inchikeys: Iterable[str],
) -> Tuple[PTDataset, dict]:
    """Drop PT compounds whose InChIKey appears in the FT set.

    Args:
        pt_ds: Molecular PT corpus. Its `dedup_keys` must be InChIKeys.
        ft_inchikeys: InChIKeys of every fine-tuning-target compound.

    Returns:
        (filtered_ds, report).
    """
    if pt_ds.domain != 'molecular':
        raise ValueError(
            f"inchikey_leakage_filter expects a molecular corpus, "
            f"got domain={pt_ds.domain!r}")
    ft = set(k for k in ft_inchikeys if k)
    keep = [i for i, k in enumerate(pt_ds.dedup_keys) if k not in ft]
    out = _select(pt_ds, keep)
    report = {
        'mode': 'inchikey_exact',
        'n_in': len(pt_ds),
        'n_out': len(out),
        'n_removed': len(pt_ds) - len(out),
        'n_ft_keys': len(ft),
    }
    return out, report


# =============================================================================
# Antibody — sequence identity
# =============================================================================

def _seq_identity(a: str, b: str) -> float:
    """Approximate sequence identity in [0, 1].

    Uses difflib's ratio (stdlib, no Biopython/ANARCI dependency). This is an
    alignment-free similarity, not a true global-alignment identity — adequate
    for a leakage *screen*, not for reporting identities. The proper metric
    (and the threshold to use) is deferred to §9 research-prompt Q6.
    """
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def sequence_leakage_filter(
    pt_ds: PTDataset,
    ft_sequences: Sequence[str],
    identity_threshold: float = 1.0,
) -> Tuple[PTDataset, dict]:
    """Drop PT antibodies that overlap the FT set by sequence identity.

    Args:
        pt_ds: Antibody PT corpus. Leakage is screened on `entities`
            (heavy-chain sequences).
        ft_sequences: Heavy-chain sequences of every FT-target antibody.
        identity_threshold: A PT sequence is dropped if its identity to ANY FT
            sequence is >= this value. Default 1.0 = exact match only (the safe
            floor). Values < 1.0 trigger a warning — the correct threshold for
            antibody PT/FT dedup is an open research question (§9 Q6) and must
            not be applied silently.

    Returns:
        (filtered_ds, report).
    """
    if pt_ds.domain != 'antibody':
        raise ValueError(
            f"sequence_leakage_filter expects an antibody corpus, "
            f"got domain={pt_ds.domain!r}")
    if not 0.0 < identity_threshold <= 1.0:
        raise ValueError(
            f"identity_threshold must be in (0, 1], got {identity_threshold}")

    ft = list(dict.fromkeys(s for s in ft_sequences if s))

    if identity_threshold >= 1.0:
        ft_set = set(ft)
        keep = [i for i, s in enumerate(pt_ds.entities) if s not in ft_set]
        mode = 'sequence_exact'
    else:
        warnings.warn(
            f"sequence_leakage_filter called with identity_threshold="
            f"{identity_threshold} < 1.0. The antibody PT/FT identity cutoff "
            f"is an OPEN research question (PRETRAIN_REP_ROADMAP §9 Q6); this "
            f"value is not yet validated. Using approximate difflib identity.",
            stacklevel=2)
        keep = []
        for i, s in enumerate(pt_ds.entities):
            leaked = any(_seq_identity(s, f) >= identity_threshold for f in ft)
            if not leaked:
                keep.append(i)
        mode = f'sequence_identity>={identity_threshold}'

    out = _select(pt_ds, keep)
    report = {
        'mode': mode,
        'identity_threshold': identity_threshold,
        'n_in': len(pt_ds),
        'n_out': len(out),
        'n_removed': len(pt_ds) - len(out),
        'n_ft_sequences': len(ft),
        'threshold_validated': identity_threshold >= 1.0,
    }
    return out, report


# =============================================================================
# Dispatcher
# =============================================================================

def remove_pt_ft_leakage(
    pt_ds: PTDataset,
    ft_keys: Optional[Iterable[str]] = None,
    ft_sequences: Optional[Sequence[str]] = None,
    identity_threshold: float = 1.0,
) -> Tuple[PTDataset, dict]:
    """Domain-dispatching PT/FT leakage removal.

    Molecular corpora are filtered by InChIKey (`ft_keys`); antibody corpora by
    sequence identity (`ft_sequences`, `identity_threshold`). Pass the FT input
    matching the corpus domain.
    """
    if pt_ds.domain == 'molecular':
        if ft_keys is None:
            raise ValueError("molecular corpus needs ft_keys (InChIKeys)")
        return inchikey_leakage_filter(pt_ds, ft_keys)
    if pt_ds.domain == 'antibody':
        if ft_sequences is None:
            raise ValueError("antibody corpus needs ft_sequences")
        return sequence_leakage_filter(pt_ds, ft_sequences, identity_threshold)
    raise ValueError(f"no leakage filter for domain '{pt_ds.domain}'")
