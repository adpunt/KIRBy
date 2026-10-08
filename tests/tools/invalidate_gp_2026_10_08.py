"""Remove every stored gp result for the datasets whose gp results predate
commit ffcd254 (2026-10-08), so the next run fits gp again.

ffcd254 changed the gp regressor the downstream model uses
(find_best_hybrid._get_gp_regressor, now WidthScaledGPR: the RBF length scale
starts at sqrt(number of features) at fit time instead of 1). Nothing in any
cache key records which gp code produced a stored score, so a stored gp score
is read back unchanged unless it is removed.

    python tests/tools/invalidate_gp_2026_10_08.py                # dry run
    python tests/tools/invalidate_gp_2026_10_08.py --delete       # remove
    python tests/tools/invalidate_gp_2026_10_08.py --datasets ginkgo_hic
    python tests/tools/invalidate_gp_2026_10_08.py --cache-dir /path/to/cache

Dry run is the default: it opens every candidate, prints what it would remove
and what it keeps, and changes nothing.

What counts as a gp entry: the stored model is 'gp' (or 'gp#<fingerprint>',
the spelling a non-registry gp estimator is filed under), or the stored
internal evaluator is 'gp'. --model-only drops the second test. Note: the
internal evaluator 'gp' is a different estimator (kirby/hybrid.py
_evaluate_allocation, ConstantKernel*RBF(length_scale=1.0)) and ffcd254 did
not change it.

Where the dataset is read from, per store (the dataset is never taken from a
file's modification date):
  result_db/<name>.db       no dataset column exists; the file name IS the
                            dataset key (result_cache.db_path_for). <name> is
                            the dataset, or dataset__<namespace> for a per-fold
                            or experiment-local search.
  baselines/*.pkl           _cache_meta['dataset_name'], _cache_meta['model_name']
  hybrid_evals/<dir>/*.pkl  _cache_meta['dataset_name'], ['model_name'], ['evaluator']
  fold_scores/<dir>/alloc__*.pkl
                            _cache_meta['evaluator']; the dataset only in <dir>
  searches/*.ckpt.pkl       _cache_meta['dataset_name'], ['model_names']

A store entry that holds gp and non-gp results together and cannot be split
(a search checkpoint) is listed and never deleted.

Exit status is 1 when any candidate could not be opened; those files are
reported and never deleted.
"""
import argparse
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Checkpoint payloads pickle kirby objects; unpickling them needs kirby.
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

# The five datasets whose decided downstream models include gp
# (tests/experiments/decided_grid.json, checked 2026-10-08).
DEFAULT_DATASETS = ('abagym_Ang2', 'abagym_VEGF', 'ginkgo_acsins',
                    'ginkgo_hic', 'ginkgo_pr_cho')


def cache_root(arg):
    if arg:
        return Path(arg)
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def dataset_of(name, datasets):
    """The dataset a cache key belongs to: the key itself, or the part before
    '__<namespace>'. Longest match wins. None when no named dataset matches."""
    best = None
    for ds in datasets:
        # '<dataset>_<suffix>' too: c05's searches on fewer molecules
        # (abagym_Ang2_n297_s0) and c07's old half names
        # (abagym_Ang2_feature_variance_easy) are this dataset's searches.
        # None of the named datasets is the start of another dataset's name.
        if name == ds or name.startswith(ds + '__') or name.startswith(ds + '_'):
            if best is None or len(ds) > len(best):
                best = ds
    return best


def near_miss(name, datasets):
    """A name that starts with a dataset but is not that dataset or
    dataset__<namespace> (e.g. 'ginkgo_hic_n500'). Not touched, but printed so
    a search filed under a decorated name is not missed in silence."""
    return any(name.startswith(ds) for ds in datasets)


MODEL = 'gp'      # --model; 'dnn' for the adme_hlm/adme_rlm dnn cap (2026-10-08)


def is_gp_model(m):
    m = '' if m is None else str(m)
    return m == MODEL or m.startswith(MODEL + '#')


def load_meta(path):
    """(meta dict, None) or (None, reason)."""
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'
    if not isinstance(wrapper, dict) or '_cache_meta' not in wrapper:
        return None, 'no _cache_meta envelope'
    meta = wrapper.get('_cache_meta')
    if not isinstance(meta, dict):
        return None, '_cache_meta is not a dict'
    return meta, None


class Report:
    def __init__(self):
        # (store, dataset) -> counters
        self.n = defaultdict(lambda: defaultdict(int))
        self.unreadable = []      # (path, reason)
        self.mixed = []           # (path, explanation)
        self.to_delete = []       # paths (pickles)

    def add(self, store, ds, key, k=1):
        self.n[(store, ds)][key] += k


# ── result_db ────────────────────────────────────────────────────────────────

def _gp_where(model_only):
    w = (f"(model = '{MODEL}' OR substr(model, 1, {len(MODEL) + 1}) = '{MODEL}#')")
    if not model_only:
        w += f" OR internal_evaluator = '{MODEL}'"
    return w


def scan_result_db(root, datasets, model_only, delete, rep, out):
    d = root / 'result_db'
    if not d.is_dir():
        out(f'result_db: {d} does not exist, nothing to do.')
        return
    where = _gp_where(model_only)
    for path in sorted(d.glob('*.db')):
        ds = dataset_of(path.stem, datasets)
        if ds is None:
            if near_miss(path.stem, datasets):
                out(f'  result_db {path.name}: name starts with a named dataset '
                    f'but is not <dataset> or <dataset>__<namespace>; NOT '
                    f'checked.')
            continue
        try:
            if delete:
                conn = sqlite3.connect(str(path), timeout=120.0)
            else:
                conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True,
                                       timeout=120.0)
            try:
                cols = {r[1] for r in conn.execute('PRAGMA table_info(results)')}
                missing = {'model', 'internal_evaluator'} - cols
                if missing:
                    raise RuntimeError(f'table results has no column(s) '
                                       f'{sorted(missing)}')
                total = conn.execute('SELECT COUNT(*) FROM results').fetchone()[0]
                n_gp = conn.execute(
                    f'SELECT COUNT(*) FROM results WHERE {where}').fetchone()[0]
                by = conn.execute(
                    f'SELECT model, internal_evaluator, COUNT(*) FROM results '
                    f'WHERE {where} GROUP BY model, internal_evaluator').fetchall()
                removed = 0
                if delete and n_gp:
                    with conn:   # one transaction: all rows or none
                        cur = conn.execute(f'DELETE FROM results WHERE {where}')
                        removed = cur.rowcount
            finally:
                conn.close()
        except Exception as e:
            rep.unreadable.append((path, f'{type(e).__name__}: {e}'))
            continue
        rep.add('result_db', ds, 'files')
        rep.add('result_db', ds, 'gp rows', n_gp)
        rep.add('result_db', ds, 'other rows kept', total - n_gp)
        if delete:
            rep.add('result_db', ds, 'gp rows removed', removed)
        if n_gp:
            detail = ', '.join(f'model={m} evaluator={e}: {c}' for m, e, c in by)
            verb = 'removed' if delete else 'would remove'
            out(f'  result_db {path.name}: {verb} {n_gp} of {total} rows '
                f'({detail}); keeps {total - n_gp}.')


# ── pickle stores ────────────────────────────────────────────────────────────

def _named_gp(fname):
    """Filename narrowing only: the model and evaluator sit between '__'."""
    return f'__{MODEL}__' in fname or f'__{MODEL}#' in fname


def scan_baselines(root, datasets, model_only, rep, out):
    d = root / 'baselines'
    if not d.is_dir():
        out(f'baselines: {d} does not exist, nothing to do.')
        return
    for path in sorted(d.glob('*.pkl')):
        ds = dataset_of(path.name, datasets)
        if ds is None:
            continue
        if not _named_gp(path.name):
            rep.add('baselines', ds, 'other files kept (name has no gp)')
            continue
        meta, why = load_meta(path)
        if meta is None:
            rep.unreadable.append((path, why))
            continue
        mds = dataset_of(str(meta.get('dataset_name')), datasets)
        if meta.get('kind') == 'baseline' and mds == ds \
                and is_gp_model(meta.get('model_name')):
            rep.to_delete.append(path)
            rep.add('baselines', ds, 'gp files')
        else:
            rep.add('baselines', ds, 'other files kept (meta is not gp)')


def scan_hybrid_evals(root, datasets, model_only, rep, out):
    d = root / 'hybrid_evals'
    if not d.is_dir():
        out(f'hybrid_evals: {d} does not exist, nothing to do.')
        return
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        ds = dataset_of(sub.name, datasets)
        if ds is None:
            if near_miss(sub.name, datasets):
                out(f'  {d.name}/{sub.name}/: name starts with a named dataset '
                    f'but is not <dataset> or <dataset>__<namespace>; NOT '
                    f'checked.')
            continue
        for path in sorted(sub.glob('*.pkl')):
            if not _named_gp(path.name):
                rep.add('hybrid_evals', ds, 'other files kept (name has no gp)')
                continue
            meta, why = load_meta(path)
            if meta is None:
                rep.unreadable.append((path, why))
                continue
            mds = dataset_of(str(meta.get('dataset_name')), datasets)
            gp = is_gp_model(meta.get('model_name')) or (
                not model_only and meta.get('evaluator') == MODEL)
            if meta.get('kind') == 'hybrid_eval' and mds == ds and gp:
                rep.to_delete.append(path)
                rep.add('hybrid_evals', ds, 'gp files')
            else:
                rep.add('hybrid_evals', ds, 'other files kept (meta is not gp)')


def scan_fold_scores(root, datasets, model_only, rep, out):
    if model_only:
        out('fold_scores: skipped (--model-only; these files hold only the '
            'internal evaluator\'s per-representation scores).')
        return
    d = root / 'fold_scores'
    if not d.is_dir():
        out(f'fold_scores: {d} does not exist, nothing to do.')
        return
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        ds = dataset_of(sub.name, datasets)
        if ds is None:
            if near_miss(sub.name, datasets):
                out(f'  {d.name}/{sub.name}/: name starts with a named dataset '
                    f'but is not <dataset> or <dataset>__<namespace>; NOT '
                    f'checked.')
            continue
        for path in sorted(sub.glob('*.pkl')):
            if not (path.name.startswith('alloc__') and _named_gp(path.name)):
                rep.add('fold_scores', ds, 'other files kept (name has no gp)')
                continue
            meta, why = load_meta(path)
            if meta is None:
                rep.unreadable.append((path, why))
                continue
            if meta.get('kind') == 'fold_alloc' and meta.get('evaluator') == MODEL:
                rep.to_delete.append(path)
                rep.add('fold_scores', ds, 'gp files')
            else:
                rep.add('fold_scores', ds, 'other files kept (meta is not gp)')


def scan_checkpoints(root, datasets, model_only, rep, out):
    d = root / 'searches'
    if not d.is_dir():
        out(f'searches: {d} does not exist, nothing to do.')
        return
    for path in sorted(d.glob('*.ckpt.pkl')):
        ds = dataset_of(path.name, datasets)
        if ds is None:
            continue
        meta, why = load_meta(path)
        if meta is None:
            rep.unreadable.append((path, why))
            continue
        mds = dataset_of(str(meta.get('dataset_name')), datasets)
        models = list(meta.get('model_names') or ())
        evals = list(meta.get('internal_evaluators') or ())
        has_gp = any(is_gp_model(m) for m in models) or (
            not model_only and (MODEL in evals
                                or ('match' in evals
                                    and any(is_gp_model(m) for m in models))))
        if mds != ds or not has_gp:
            rep.add('checkpoints', ds, 'other files kept (meta is not gp)')
            continue
        if models and all(is_gp_model(m) for m in models):
            rep.to_delete.append(path)
            rep.add('checkpoints', ds, 'gp files')
        else:
            rep.mixed.append((path, f'models {models}'))
            rep.add('checkpoints', ds, 'mixed gp + other, NOT deleted')


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--datasets', nargs='+', default=list(DEFAULT_DATASETS))
    ap.add_argument('--cache-dir', default=None,
                    help='cache root (default: KIRBY_CACHE_DIR, else ~/.cache/kirby)')
    ap.add_argument('--delete', action='store_true',
                    help='remove the entries (default: dry run, change nothing)')
    ap.add_argument('--model-only', action='store_true',
                    help="only entries whose downstream model is gp; keep "
                         "entries where only the internal evaluator is gp")
    ap.add_argument('--model', default='gp',
                    help="downstream model whose stored results go (default gp; "
                         "'dnn' for the adme dnn prediction cap)")
    a = ap.parse_args(argv)
    global MODEL
    MODEL = a.model
    root = cache_root(a.cache_dir)
    out = print
    out(f'cache root: {root}')
    out(f'datasets: {", ".join(a.datasets)}')
    out(f'model: {MODEL}')
    out(f'mode: {"DELETE" if a.delete else "dry run (nothing is changed)"}'
        f'{"; downstream model gp only" if a.model_only else ""}')
    if not root.is_dir():
        out(f'{root} does not exist.')
        return 1

    rep = Report()
    out('\nresult_db (SQLite, rows):')
    scan_result_db(root, a.datasets, a.model_only, a.delete, rep, out)
    scan_baselines(root, a.datasets, a.model_only, rep, out)
    scan_hybrid_evals(root, a.datasets, a.model_only, rep, out)
    scan_fold_scores(root, a.datasets, a.model_only, rep, out)
    scan_checkpoints(root, a.datasets, a.model_only, rep, out)

    deleted, failed = 0, []
    if a.delete:
        for p in rep.to_delete:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                failed.append((p, f'{type(e).__name__}: {e}'))

    out('\nPickle files with a gp result:')
    for p in rep.to_delete:
        out(f'  {"removed" if a.delete else "would remove"} {p.relative_to(root)}')
    if not rep.to_delete:
        out('  none')

    out('\nSummary, per store and dataset:')
    for (store, ds) in sorted(rep.n):
        parts = ', '.join(f'{k} {v}' for k, v in rep.n[(store, ds)].items())
        out(f'  {store:<13} {ds:<15} {parts}')
    if not rep.n:
        out('  nothing for these datasets in any store')

    if rep.mixed:
        out('\nSearch checkpoints holding gp AND other models (NOT deleted):')
        for p, why in rep.mixed:
            out(f'  {p.relative_to(root)}  ({why})')
        out('  A checkpoint holds the fold-0 results and the pruning decisions '
            'of every model in one search. If pruning is on and the checkpoint '
            'says fold 0 is done, the search reuses its stored fold-0 gp '
            'scores. Deleting the file makes that search redo fold 0 for every '
            'model in it, not only gp.')

    if rep.unreadable:
        out('\nCould not open (NOT deleted):')
        for p, why in rep.unreadable:
            out(f'  {p}: {why}')
    if failed:
        out('\nCould not delete:')
        for p, why in failed:
            out(f'  {p}: {why}')
    if a.delete:
        out(f'\nRemoved {deleted} pickle files.')
    return 1 if (rep.unreadable or failed) else 0


if __name__ == '__main__':
    sys.exit(main())
