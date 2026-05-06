"""Standard result schema for ANNITIA experiments (Phase X.3).

Every search/run runner emits one record per condition:

    {
      'schema_version':  int,                     # bump when shape changes
      'endpoint':        'hepatic' | 'death',
      'layer1':          {'views': [...],
                          'n_features': int,
                          'n_features_after_prune': int | None,
                          'feature_set_id': str},  # hash of the (subset, prune)
      'layer2':          {'framing': str},          # cox | pseudo_obs | landmark | deephit | ...
      'layer3':          {'algorithm': str},        # cox | coxnet | rsf | gbsa | xgbcox | ...
      'layer4':          {'tricks': [...]},         # ordered list of trick names
      'fold_metrics':    {'fold_cindex':  [floats per fold],
                          'fold_cindex_other': [...] | None,  # for joint models
                          'mean_cindex':  float,
                          'std_cindex':   float},
      'oof_path':        str | None,                # OOF risks .npy (X.4)
      'model_path':      str | None,                # refit-on-full pickle path
      'config_id':       str,                       # sha key for this record
      'elapsed_sec':     float,
      # ---- legacy fields kept for backwards compat with run.py outputs ----
      'views':           [list of view names],
      'model':           str,
      'prune':           int | None,
      'fold_cindex':     [floats],
      'mean_cindex':     float,
      'std_cindex':      float,
      'n_features':      int,
      'n_features_after_prune': int,
    }

The legacy fields duplicate the layer-1/3 fields. They exist because:
    - Older `tests/results/annitia_*_search.json` files are read by
      build_submission_zip / submit / combine_branches, which key off
      `r['views']` and `r['model']`.
    - The legacy fields are produced by the same writer so a single
      downstream consumer never has to branch on schema_version.

When reading old JSONs (no 'schema_version' key), `upgrade_record` lifts the
flat fields into the layer-1..4 block.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


@dataclass
class ConditionRecord:
    endpoint: str
    layer1_views: tuple = ()
    layer1_n_features: int = 0
    layer1_prune: Optional[int] = None
    layer2_framing: str = 'cox'
    layer3_algorithm: str = ''
    layer4_tricks: List[str] = field(default_factory=list)
    fold_cindex: List[float] = field(default_factory=list)
    fold_cindex_other: Optional[List[float]] = None
    mean_cindex: float = float('nan')
    std_cindex: float = float('nan')
    oof_path: Optional[str] = None
    model_path: Optional[str] = None
    elapsed_sec: float = 0.0


def feature_set_id(views: tuple, prune: Optional[int]) -> str:
    """Stable id for a (view-subset, prune) pair. Used by the cache layer
    and OOF filenames."""
    payload = '+'.join(sorted(views)) + f'__p{prune if prune is not None else "all"}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def config_id(layer1_views: tuple,
              layer1_prune: Optional[int],
              layer2_framing: str,
              layer3_algorithm: str,
              layer4_tricks: Optional[List[str]] = None) -> str:
    """Stable id for a full Layer 1-4 configuration."""
    tricks = sorted(layer4_tricks or [])
    payload = (f'l1={"+".join(sorted(layer1_views))}|'
               f'p={layer1_prune}|l2={layer2_framing}|'
               f'l3={layer3_algorithm}|l4={",".join(tricks)}')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def make_record(*,
                endpoint: str,
                views: tuple,
                model: str,
                prune: Optional[int],
                fold_cindex: List[float],
                mean_cindex: float,
                std_cindex: float,
                n_features: int,
                # NEW fields with defaults so the existing search loop can
                # emit a record without touching every call site.
                framing: str = 'cox',
                tricks: Optional[List[str]] = None,
                fold_cindex_other: Optional[List[float]] = None,
                oof_path: Optional[str] = None,
                model_path: Optional[str] = None,
                elapsed_sec: float = 0.0) -> Dict[str, Any]:
    """Build one condition record with both the new layered fields and the
    legacy flat fields populated. Returns a plain dict (JSON-serialisable
    after _convert)."""
    tricks = list(tricks or [])
    n_after = n_features if prune is None else min(prune, n_features)
    fsid = feature_set_id(tuple(views), prune)
    cid = config_id(tuple(views), prune, framing, model, tricks)
    return {
        'schema_version': SCHEMA_VERSION,
        'endpoint': endpoint,
        'layer1': {
            'views': list(views),
            'n_features': int(n_features),
            'n_features_after_prune': int(n_after),
            'prune': prune,
            'feature_set_id': fsid,
        },
        'layer2': {'framing': framing},
        'layer3': {'algorithm': model},
        'layer4': {'tricks': tricks},
        'fold_metrics': {
            'fold_cindex': list(fold_cindex),
            'fold_cindex_other': list(fold_cindex_other) if fold_cindex_other is not None else None,
            'mean_cindex': float(mean_cindex),
            'std_cindex': float(std_cindex),
        },
        'oof_path': oof_path,
        'model_path': model_path,
        'config_id': cid,
        'elapsed_sec': float(elapsed_sec),
        # ---- legacy duplicates (don't remove) ----
        'views': list(views),
        'model': model,
        'prune': prune,
        'fold_cindex': list(fold_cindex),
        'mean_cindex': float(mean_cindex),
        'std_cindex': float(std_cindex),
        'n_features': int(n_features),
        'n_features_after_prune': int(n_after),
    }


def upgrade_record(record: Dict[str, Any], endpoint: str = '') -> Dict[str, Any]:
    """Lift an old flat-field record (or a partial one) into the new schema.

    Tolerant to missing fields; fills sensible defaults. Used when a JSON file
    pre-dates the schema or was produced by an older runner.
    """
    if record.get('schema_version') == SCHEMA_VERSION and 'layer1' in record:
        return record
    views = tuple(record.get('views', ()) or ())
    prune = record.get('prune')
    model = record.get('model', '')
    n_features = int(record.get('n_features', 0) or 0)
    fold_cindex = list(record.get('fold_cindex', []) or [])
    mean_c = float(record.get('mean_cindex', float('nan')))
    std_c = float(record.get('std_cindex', float('nan')))
    framing = record.get('framing', 'cox')
    tricks = list(record.get('tricks', []) or [])
    return make_record(
        endpoint=record.get('endpoint', endpoint),
        views=views,
        model=model,
        prune=prune,
        fold_cindex=fold_cindex,
        mean_cindex=mean_c,
        std_cindex=std_c,
        n_features=n_features,
        framing=framing,
        tricks=tricks,
        fold_cindex_other=record.get('fold_cindex_other'),
        oof_path=record.get('oof_path'),
        model_path=record.get('model_path'),
        elapsed_sec=float(record.get('elapsed_sec', 0.0) or 0.0),
    )


def condition_key(record: Dict[str, Any]):
    """Resume key — must be stable across runs. Used by search to skip
    already-evaluated tuples. Tolerates both legacy and v1 records."""
    rec = upgrade_record(record) if record.get('schema_version') != SCHEMA_VERSION else record
    return (
        tuple(rec['layer1']['views']),
        rec['layer3']['algorithm'],
        rec['layer1'].get('prune'),
        rec['layer2']['framing'],
        tuple(sorted(rec['layer4'].get('tricks', []))),
    )
