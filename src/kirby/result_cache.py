"""Per-dataset SQLite result store for hybrid search.

Design:
- ONE SQLite DB file per dataset, under  <cache>/result_db/<dataset>.db
- Each ROW is one config evaluation — the same unit the engine already lists in
  `SearchResult.all_results`.
- Reuse = SELECT the rows that already exist for a search IDENTITY; the engine
  then computes only the configs not present, and INSERTs them.
- Single-writer (no concurrent jobs against one dataset), so plain `sqlite3`
  with the default rollback journal is safe on NFS — the NFS locking problem
  only bites with *simultaneous* writers.
- `sqlite3` is in the Python standard library, so this adds NO dependency and a
  DB is a single self-contained, portable file (publishing-safe).

Why this replaces a combined `searches/*.pkl` bundle:
- No combined-summary file → nothing for two runs to clobber.
- Each tuning knob is an explicit COLUMN, not part of a filename → no filename
  collision between configs.
- Asking for a different model/ranker/budget list never invalidates a file — it
  is just a different `SELECT`, not an all-or-nothing rebuild.

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

# NO schema version / migration logic. If the table layout ever changes, the
# discipline is the same as for any other change: WIPE the DBs and re-run. The
# code never stamps, compares, or migrates a version.

# A search's IDENTITY — the columns that decide "is this the same cached
# search?". The data, the fold split, the metric the stored score is in, and
# tier (which fixes the rep pool and the grouping/fold logic). Change any of
# these and it's a genuinely different search; everything else about a result is
# looked up within it.
IDENTITY_COLS: Tuple[str, ...] = (
    'tier',
    'labels_hash', 'groups_hash', 'n_folds', 'seed', 'cv_splitter_id',
    # task_type picks the fold splitter -- classification gets
    # StratifiedGroupKFold, everything else GroupKFold -- so the same rows and
    # the same seed split differently under it. metric decides what score_mean
    # MEANS; a row's stored ranking_metric was dropped on read and never
    # checked, so a run asking for rmse could be handed a row scored with
    # spearman and rank it as if it were rmse. Both added 2026-09-04, which
    # widens the primary key: every result_db/*.db written before that date has
    # to be deleted (there is no migration path -- see the module note).
    'metric', 'task_type',
    # What the stored score MEASURES: 'predictive' (the downstream model's
    # score in `metric`), 'similarity' (each molecule's activity predicted from
    # its five nearest neighbours), 'struct_fid' (MCS fidelity), or a blend's
    # stable sorted name. It used to reach this identity only by riding on the
    # free-text `tier`, so a run that passed no tier marker could be handed a
    # predictive row for a geometry search. Added 2026-09-10, which widens the
    # primary key: every result_db/*.db written before that date has to be
    # deleted (there is no migration -- see the module note).
    'objective',
    # WHICH REPRESENTATIONS the search ran over, as sorted (name, n_features)
    # pairs. Until 2026-09-10 this was recorded as plain data and never matched
    # on, and the comment here said the pool was "fixed by tier+config, so it
    # never needs to be a key". That was false: tier is caller-chosen free text.
    # Measured -- two searches over the same rows, folds, seed, metric and tier,
    # the first over two representations and the second over four: the second
    # reused the first's row, fitted no model, and reported the two-
    # representation score, width and budget split as its own. Names alone are
    # not enough either, because the same name can arrive at two widths, and a
    # representation's width decides how much of the budget it can absorb.
    'rep_pool',
    # Whether the models were TUNED. A tuned fit and an untuned fit on identical
    # rows are different estimators. Both the per-representation baseline files
    # and the per-fold hybrid files carry a '__tuned' marker in their names, so
    # they separate correctly; this store had nothing, so a tuned run read the
    # untuned score and fitted nothing. Canonical form: '' when off, else
    # 'tuned<n_iter>' with '_k' appended when the tuned neighbour count is kept.
    'tune',
    # The caller-supplied version of any representation built INSIDE each fold
    # from that fold's own training labels. find_best_hybrid refuses to cache
    # without one -- "a stable per-version identifier is the caller's
    # responsibility" -- and then had nowhere to put it, so bumping it, which is
    # the documented way to invalidate a changed builder, invalidated the
    # fine-tuned embeddings and left every stored score reachable. A run with no
    # such builder writes '', which is every ordinary run.
    'rep_factories_id',
)

# A result's CONFIG — the full recipe that defines ONE row within a search.
# Change any of these and you get a NEW row computed on demand, never a rebuild:
# this is where swapping a model / ranker / budget lands (the whole point of the
# redesign), plus the augmentation recipe and the greedy/candidate knobs. They
# are stored per row, so e.g. a 'none' run and an 'uncapped' run over the same
# folds coexist instead of overwriting each other.
CONFIG_COLS: Tuple[str, ...] = (
    'rep_subset', 'allocation_method', 'budget', 'importance_method',
    'model', 'internal_evaluator', 'step_size',
    'aug_strategy', 'aug_names', 'aug_threshold', 'aug_budget',
    'candidate_ranking', 'greedy_tolerance',
    # How much better than keeping NO extras a block of extras must score to be
    # accepted. Added 2026-08-26. A KEY setting, not data: a search must be
    # able to try 0.0 and 0.005 and keep both results. Adding it widened the
    # table's primary key, and _ensure_schema has no migration path, so every
    # result_db/*.db written before this date is stale and must be deleted.
    # Free at the time: the cache had already been cleared.
    'augmentation_margin',
    # Extras sized as a fraction of the budget instead of the fixed ladder.
    # None = ladder. A key setting: the grid sweeps 0.0/0.05/0.10/0.25/0.50 and
    # every value must keep its own result.
    'augmentation_fraction',
)

# Recorded on every row as plain data (NOT part of the key). `rep_pool` moved
# OUT of here and into IDENTITY_COLS on 2026-09-10 — see the note there.
DATA_COLS: Tuple[str, ...] = ()

# All key columns are stored canonicalised to TEXT (see `canon`) so matching is
# a plain `=` with no NULL-vs-NULL pitfalls and no int/float drift across runs.
KEY_COLS: Tuple[str, ...] = IDENTITY_COLS + CONFIG_COLS

# Result payload columns. `metrics_json` carries the FULL metric set;
# `score_mean` is the single designated ranking metric (kept so existing readers
# and the argmax are unchanged). `blueprint` is an opaque pickled blob.
RESULT_COLS: Tuple[str, ...] = (
    'score_mean', 'score_std', 'ranking_metric',
    'fold_scores_json', 'metrics_json',
    'n_features', 'time_seconds', 'had_imputation',
    'blueprint',
)

ALL_COLS: Tuple[str, ...] = KEY_COLS + DATA_COLS + RESULT_COLS


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


def cfg_key(config: Dict[str, Any]) -> Tuple[str, ...]:
    """Canonical per-row config key: the CONFIG_COLS of one result, in order.

    Takes a dict so the key is built by COLUMN NAME (order-safe) — adding a
    CONFIG column can't silently shift the key out from under a caller.
    """
    return tuple(canon(config.get(c)) for c in CONFIG_COLS)


class ResultCache:
    """A single dataset's result table. Open, fetch existing, insert new, close.

    Usage:
        with ResultCache('caco2') as rc:
            have = rc.fetch_rows(identity)        # list[dict]
            done = {rc.row_cfg_key(r) for r in have}
            ...                                   # compute only missing configs
            rc.upsert_rows(new_rows)              # persist
    """

    # How long to wait for another process to release its write lock before
    # giving up. The module is written for a single writer, and the SLURM array
    # scripts break that rule: several tasks of one array can share a dataset,
    # and every arm of an experiment rides on `tier`, so they land in one file.
    # Uncontended concurrency was measured clean -- three processes, 120 rows
    # each with a ~320 KB blueprint, 360 of 360 rows present, integrity ok.
    # Contended was not: with no timeout argument sqlite3 waits the library
    # default of 5 seconds and then raises, and nothing here or in
    # find_best_hybrid catches it, so a whole search dies AFTER paying for its
    # per-representation baselines and its per-fold score precompute (24 s on a
    # 120-row synthetic set; hours on a real dataset). The open is the worse of
    # the two failure points, because CREATE TABLE IF NOT EXISTS takes a write
    # lock even when the table is already there. Waiting costs nothing when
    # uncontended and turns a dead job into a pause.
    BUSY_TIMEOUT_SECONDS: float = 120.0

    def __init__(self, dataset: str, cache_dir: Optional[Path] = None,
                 timeout: Optional[float] = None):
        self.dataset = dataset
        self.path = db_path_for(dataset, cache_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path),
            timeout=(self.BUSY_TIMEOUT_SECONDS if timeout is None
                     else float(timeout)))
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
        # No version pragma, no migration — see the module note. Just create the
        # table if it isn't there. A stale-layout DB must be wiped and rebuilt.
        cur = self.conn.cursor()
        key_defs = ',\n  '.join(f'{c} TEXT NOT NULL' for c in KEY_COLS)
        data_defs = ''.join(f'{c} TEXT,\n              ' for c in DATA_COLS)
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS results (
              {key_defs},
              {data_defs}score_mean REAL,
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
        # A file written under an older column list is not migrated: CREATE
        # TABLE IF NOT EXISTS leaves it alone, and the SELECT below would then
        # fail deep inside a search with "no such column". Say so here instead.
        have = {r[1] for r in cur.execute('PRAGMA table_info(results)')}
        missing = [c for c in ALL_COLS if c not in have]
        if missing:
            raise RuntimeError(
                f"{self.path} was written under an older column list and is "
                f"missing {missing}. There is no migration path by design: "
                f"delete this file and re-run.")
        # Index the identity prefix so per-identity SELECTs are fast.
        cur.execute('CREATE INDEX IF NOT EXISTS ix_identity ON results '
                    f'({", ".join(IDENTITY_COLS)})')
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
        return cfg_key(row)

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
            for c in DATA_COLS:
                vals.append(r.get(c))   # plain data, stored as-is
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
