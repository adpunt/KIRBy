"""tests/tools/invalidate_gp_2026_10_08.py removes only gp entries of the named
datasets, and only when --delete is given.

A temporary cache is built with the engine's own writers: ResultCache.upsert_rows
for the SQLite rows, and the real path + meta + save functions for baseline,
hybrid-eval and per-fold allocation-score pickles.

Run: python -m pytest tests/test_invalidate_gp.py -q
"""
import importlib.util
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from kirby import find_best_hybrid as fbh  # noqa: E402
from kirby.result_cache import ResultCache, KEY_COLS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'invalidate_gp', REPO / 'tests' / 'tools' / 'invalidate_gp_2026_10_08.py')
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

LABELS = np.arange(20, dtype=float)
TARGET = 'ginkgo_hic'
OTHER = 'herg_ki'


def _row(model, evaluator, budget):
    r = {c: '' for c in KEY_COLS}
    r.update(tier='general', metric='spearman', task_type='regression',
             model=model, internal_evaluator=evaluator, budget=budget,
             allocation_method='performance_weighted',
             importance_method='lightgbm_gain', score_mean=0.5)
    return r


def _db_rows(path):
    with sqlite3.connect(str(path)) as c:
        return sorted(c.execute(
            'SELECT model, internal_evaluator, budget FROM results').fetchall())


def _baseline(ds, model):
    p = fbh._baseline_cache_path(ds, 'onehot', model, 'spearman', 5, 42)
    meta = fbh._make_baseline_meta(ds, 'onehot', (20, 4), model, 'spearman',
                                   5, 42, LABELS, None)
    assert fbh._save_cache_with_meta(p, meta, {'score': 0.1})
    return p


def _hybrid_eval(ds, model, evaluator):
    kw = dict(alloc='performance_weighted', imp='lightgbm_gain',
              evaluator=evaluator, budget=25, fold_idx=0, n_folds=5, seed=42,
              model_name=model)
    p = fbh._hybrid_eval_cache_path(ds, 'abc123', **kw)
    meta = fbh._make_hybrid_eval_meta(
        ds, ['onehot'], (('onehot', 0),), train_idx=np.arange(10),
        val_idx=np.arange(10, 20), labels_hash=1, aug_strategy='none',
        aug_names=(), aug_hashes=(), step_size=5, candidate_ranking='x',
        greedy_tolerance=0.0, augmentation_margin=0.0,
        augmentation_fraction=None, augmentation_threshold=None,
        augmentation_budget=0, metric_name='spearman',
        task_type='regression', **kw)
    assert fbh._save_cache_with_meta(p, meta, {'score': 0.2})
    return p


def _fold_alloc(ds, evaluator):
    p = fbh._fold_score_cache_path(ds, 'alloc', 0, evaluator, 'h1')
    assert fbh._save_cache_with_meta(
        p, {'kind': 'fold_alloc', 'evaluator': evaluator, 'fold_idx': 0,
            'input_hash': 'h1'}, {'default_scores': {}, 'metric_scores': {}})
    return p


def _checkpoint(ds, models):
    p = fbh._search_checkpoint_path(ds, 'general', 'none')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps({
        '_cache_meta': {'kind': 'search', 'dataset_name': ds,
                        'model_names': tuple(models),
                        'internal_evaluators': ('match',)},
        'payload': {'fold0_done': True, 'fold0_results': []}}))
    return p


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    files = {}
    for ds in (TARGET, f'{TARGET}__ocv5f0s42', OTHER):
        with ResultCache(ds, cache_dir=tmp_path) as rc:
            rc.upsert_rows([_row('gp', 'gp', 25), _row('gp', 'rf', 25),
                            _row('svm', 'svm', 25), _row('svm', 'rf', 50)])
        files[('db', ds)] = tmp_path / 'result_db' / f'{ds}.db'
        for m in ('gp', 'svm'):
            files[('base', ds, m)] = _baseline(ds, m)
            files[('he', ds, m)] = _hybrid_eval(ds, m, m)
        for ev in ('gp', 'rf'):
            files[('alloc', ds, ev)] = _fold_alloc(ds, ev)
    files['ckpt_mixed'] = _checkpoint(TARGET, ['gp', 'svm'])
    files['ckpt_gp'] = _checkpoint(f'{TARGET}__ocv5f0s42', ['gp'])
    # A file that does not unpickle, named like a gp baseline of the target.
    bad = tmp_path / 'baselines' / f'{TARGET}__x__gp__spearman__5f_42.pkl'
    bad.write_bytes(b'not a pickle')
    files['bad'] = bad
    return tmp_path, files


def _snapshot(root):
    return {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_changes_nothing(cache, capsys):
    root, files = cache
    before = _snapshot(root)
    rc = tool.main(['--datasets', TARGET, '--cache-dir', str(root)])
    assert rc == 1                      # the unreadable file
    assert _snapshot(root) == before
    out = capsys.readouterr().out
    assert 'would remove 2 of 4 rows' in out
    assert 'Could not open' in out and files['bad'].name in out


def test_delete_removes_only_gp_of_named_dataset(cache):
    root, files = cache
    rc = tool.main(['--datasets', TARGET, '--cache-dir', str(root), '--delete'])
    assert rc == 1
    for ds in (TARGET, f'{TARGET}__ocv5f0s42'):
        assert _db_rows(files[('db', ds)]) == [
            ('svm', 'rf', '50'), ('svm', 'svm', '25')]
        assert not files[('base', ds, 'gp')].exists()
        assert files[('base', ds, 'svm')].exists()
        assert not files[('he', ds, 'gp')].exists()
        assert files[('he', ds, 'svm')].exists()
        assert not files[('alloc', ds, 'gp')].exists()
        assert files[('alloc', ds, 'rf')].exists()
    # Another dataset is untouched.
    assert len(_db_rows(files[('db', OTHER)])) == 4
    for k, p in files.items():
        if isinstance(k, tuple) and k[1] == OTHER:
            assert p.exists(), k
    # Mixed checkpoint kept, gp-only checkpoint removed, unreadable kept.
    assert files['ckpt_mixed'].exists()
    assert not files['ckpt_gp'].exists()
    assert files['bad'].exists()


def test_model_only_keeps_evaluator_gp(cache):
    root, files = cache
    tool.main(['--datasets', TARGET, '--cache-dir', str(root), '--delete',
               '--model-only'])
    assert _db_rows(files[('db', TARGET)]) == [
        ('svm', 'rf', '50'), ('svm', 'svm', '25')]
    assert files[('alloc', TARGET, 'gp')].exists()
