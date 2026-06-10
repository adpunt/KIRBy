"""Per-dataset SQLite result store for hybrid search.

Design (REPRODUCIBILITY_LEDGER.md §X8.5 D5, 2026-06-10d):
- ONE SQLite DB file per dataset, under  <cache>/result_db/<dataset>.db
- Each ROW is one config evaluation — the same unit the engine already lists in
  `SearchResult.all_results`.
- Reuse = SELECT the rows that already exist for a search IDENTITY; the engine
  then computes only the configs not present, and INSERTs them.
- Single-writer (the user runs no concurrent jobs against one dataset), so plain
  `sqlite3` with the default rollback journal is safe on NFS — the NFS locking
  problem only bites with *simultaneous* writers.
- `sqlite3` is in the Python standard library, so this adds NO dependency and a
  DB is a single self-contained, portable file (publishing-safe).

Why this replaces the old `searches/*.pkl` bundle:
- No combined-summary file → nothing for two runs to clobber (ledger O2/O3).
- Each tuning knob is an explicit COLUMN, not part of a filename → the old
  hybrid_eval filename-collision bug (X8.7 TRAP-2) cannot occur.
- Asking for a different model/ranker/budget list never invalidates a file — it
  is just a different `SELECT` (the months-long all-or-nothing rebuild, X1).

This module is intentionally generic: it stores and returns ROW DICTS. The
caller (find_best_hybrid) owns the translation between a row dict and an
`all_results` entry (including the pickled blueprint blob), so this file has no
dependency on the engine's internals.
"""

from __future__ import annotations

import os
import json
import sqlite3
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Bump only on an INCOMPATIBLE schema change. A bump makes old DBs ignored
# (treated as empty) rather than mis-read — never silently mixed.
SCHEMA_VERSION = 1

# Columns that scope ONE search invocation (the "spine" + the conditional knobs
# that genuinely change a computed score — ledger X8.6 / X8.7).
#
# WHY rep_names / rep_shapes_hash are here (determinism-critical): the config
# key's rep component is just `rep_set_name` ('all' or 'no_<rep>'), which does
# NOT encode WHICH reps are in the pool. Two runs over different base-rep pools
# both label the full set 'all', so without the pool in the identity their rows
# would collide and reuse would be WRONG (allocation re-ranks over the pool —
# ledger X3). rep_names pins the pool by name; rep_shapes_hash additionally
# catches a rep whose GENERATING CODE changed its width without changing its
# name (the X9.4 rep-code-drift gap, e.g. graph_kernel n_iter / a PCA
# n_components). Mirrors the old `_make_search_meta` rep_names/rep_shapes key.
#
# WHY metric_name is here (TRANSITIONAL): until D6 stores the FULL metric set
# per row, `score_mean` is the single ranking metric's value, so a different
# ranking metric must not reuse it. Keying on metric_name keeps that correct.
# Once D6 lands (full metric dict stored), DROP metric_name here and derive
# score_mean = metrics[designated] on read (ledger D2 — metric is NOT a key).
IDENTITY_COLS: Tuple[str, ...] = (
    'tier', 'aug_strategy', 'speed', 'task_type', 'condition_suffix',
    'rep_names', 'rep_shapes_hash', 'metric_name',
    # screen_threshold MUTATES the base-rep pool before the search (drops weak
    # reps), so it changes every computed score → it MUST be in the identity
    # (the old _make_search_meta keyed on it). rep_factories_id pins the factory
    # version: factory reps enter via fold_reps (NOT base_reps), so neither
    # rep_names nor rep_shapes_hash captures a factory bump — without this a new
    # FT-embedding version would silently reuse old scores.
    'screen_threshold', 'rep_factories_id',
    'labels_hash', 'groups_hash', 'n_samples', 'n_folds', 'seed',
    'cv_splitter_id',
    'candidate_ranking', 'greedy_tolerance',
    'aug_threshold', 'aug_budget', 'aug_names',
)

# Columns that identify ONE result within a search invocation. Order MUST match
# `cfg_key()` below. step_size is a CONFIG column (not identity): the main search
# uses a constant step_size, but two-phase refinement varies it per result, so
# refinement rows must NOT collide with their phase-1 siblings on the key.
CONFIG_COLS: Tuple[str, ...] = (
    'rep_subset', 'allocation_method', 'budget',
    'importance_method', 'model', 'internal_evaluator', 'step_size',
)

# All key columns are stored canonicalised to TEXT (see `canon`) so matching is
# a plain `=` with no NULL-vs-NULL pitfalls and no int/float drift across runs.
KEY_COLS: Tuple[str, ...] = IDENTITY_COLS + CONFIG_COLS

# Result payload columns. `metrics_json` carries the FULL metric set (ledger D6);
# `score_mean` is the single designated ranking metric (kept so existing readers
# and the argmax are unchanged). `blueprint` is an opaque pickled blob.
RESULT_COLS: Tuple[str, ...] = (
    'score_mean', 'score_std', 'ranking_metric',
    'fold_scores_json', 'metrics_json',
    'n_features', 'time_seconds', 'had_imputation',
    'blueprint',
)

ALL_COLS: Tuple[str, ...] = KEY_COLS + RESULT_COLS


def canon(v: Any) -> str:
    """Canonical TEXT form of a key value (None -> '' so it matches with `=`)."""
    if v is None:
        return ''
    if isinstance(v, float):
        # Stable, round-trippable text for floats used as keys (e.g. tolerance).
        return repr(v)
    return str(v)


def _cache_base() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def db_path_for(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else _cache_base()
    return base / 'result_db' / f'{dataset}.db'


def cfg_key(rep_subset, allocation_method, budget, importance_method,
            model, internal_evaluator, step_size) -> Tuple[str, ...]:
    """Canonical config key tuple. Order MUST match CONFIG_COLS."""
    return (canon(rep_subset), canon(allocation_method), canon(budget),
            canon(importance_method), canon(model), canon(internal_evaluator),
            canon(step_size))


class ResultCache:
    """A single dataset's result table. Open, fetch existing, insert new, close.

    Usage:
        with ResultCache('caco2') as rc:
            have = rc.fetch_rows(identity)        # list[dict]
            done = {rc.row_cfg_key(r) for r in have}
            ...                                   # compute only missing configs
            rc.upsert_rows(new_rows)              # persist
    """

    def __init__(self, dataset: str, cache_dir: Optional[Path] = None):
        self.dataset = dataset
        self.path = db_path_for(dataset, cache_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> 'ResultCache':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute('PRAGMA user_version')
        ver = cur.fetchone()[0]
        if ver and ver != SCHEMA_VERSION:
            # Incompatible old schema: drop and recreate (we never migrate —
            # ledger D4; the rebuild is the source of truth).
            cur.execute('DROP TABLE IF EXISTS results')
            ver = 0
        key_defs = ',\n  '.join(f'{c} TEXT NOT NULL' for c in KEY_COLS)
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS results (
              {key_defs},
              score_mean REAL,
              score_std REAL,
              ranking_metric TEXT,
              fold_scores_json TEXT,
              metrics_json TEXT,
              n_features INTEGER,
              time_seconds REAL,
              had_imputation INTEGER,
              blueprint BLOB,
              created_at REAL,
              PRIMARY KEY ({', '.join(KEY_COLS)})
            )
        ''')
        # Index the identity prefix so per-identity SELECTs are fast.
        cur.execute('CREATE INDEX IF NOT EXISTS ix_identity ON results '
                    f'({", ".join(IDENTITY_COLS)})')
        cur.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
        self.conn.commit()

    # -- read -------------------------------------------------------------
    def fetch_rows(self, identity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """All stored rows for a search identity, as plain dicts.

        `blueprint` is unpickled; `*_json` columns are parsed; the canonical
        config key is attached as `_cfg_key`.
        """
        where = ' AND '.join(f'{c} = ?' for c in IDENTITY_COLS)
        params = [canon(identity.get(c)) for c in IDENTITY_COLS]
        cur = self.conn.execute(f'SELECT * FROM results WHERE {where}', params)
        out: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            d = dict(row)
            if d.get('blueprint') is not None:
                d['blueprint'] = pickle.loads(d['blueprint'])
            d['fold_scores'] = json.loads(d.pop('fold_scores_json') or 'null')
            d['metrics'] = json.loads(d.pop('metrics_json') or 'null')
            d['_cfg_key'] = self.row_cfg_key(d)
            out.append(d)
        return out

    @staticmethod
    def row_cfg_key(row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(canon(row.get(c)) for c in CONFIG_COLS)

    # -- write ------------------------------------------------------------
    def upsert_rows(self, rows: Sequence[Dict[str, Any]]) -> int:
        """INSERT OR REPLACE a batch of fully-specified row dicts.

        Each row must carry every KEY column. Result columns are optional but
        normally present. `blueprint` (if given) is pickled; `fold_scores` and
        `metrics` are JSON-encoded into their `*_json` columns.
        """
        if not rows:
            return 0
        import time as _time
        cols = list(ALL_COLS) + ['created_at']
        placeholders = ', '.join('?' for _ in cols)
        sql = (f'INSERT OR REPLACE INTO results ({", ".join(cols)}) '
               f'VALUES ({placeholders})')
        now = _time.time()
        payload = []
        for r in rows:
            vals: List[Any] = []
            for c in KEY_COLS:
                vals.append(canon(r.get(c)))
            vals.append(r.get('score_mean'))
            vals.append(r.get('score_std'))
            vals.append(r.get('ranking_metric'))
            vals.append(json.dumps(r.get('fold_scores'))
                        if r.get('fold_scores') is not None else None)
            vals.append(json.dumps(r.get('metrics'))
                        if r.get('metrics') is not None else None)
            vals.append(r.get('n_features'))
            vals.append(r.get('time_seconds'))
            vals.append(int(r['had_imputation'])
                        if r.get('had_imputation') is not None else None)
            bp = r.get('blueprint')
            vals.append(sqlite3.Binary(pickle.dumps(bp, protocol=pickle.HIGHEST_PROTOCOL))
                        if bp is not None else None)
            vals.append(now)
            payload.append(vals)
        self.conn.executemany(sql, payload)
        self.conn.commit()
        return len(payload)
