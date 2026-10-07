"""A GP is handed per-molecule weights: every path records it as not run.

Added 2026-10-07. sklearn's GaussianProcessRegressor.fit takes no
sample_weight. On the server, n02 and n07 failed on ginkgo_pr_cho because a
weighting cleanup reached a gp fit through the D-versus-D' rows, which catch
nothing. Each test here goes through one of the paths that used to raise or
return an unexplained nan.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'src'), os.path.join(ROOT, 'tests'),
                os.path.join(ROOT, 'tests', 'experiments')]

from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

from experiments.noise import noise_cache as nc  # noqa: E402


def _data(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    w = rng.uniform(0.2, 1.0, size=n)
    return X[:30], y[:30], X[30:], y[30:], w[:30]


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('KIRBY_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('KIRBY_NOISE_CACHE_DIR', str(tmp_path / 'noise_cache'))


def test_fit_takes_sample_weight():
    assert not nc.fit_takes_sample_weight(GaussianProcessRegressor)
    assert not nc.fit_takes_sample_weight(GaussianProcessRegressor())
    assert nc.fit_takes_sample_weight(RandomForestRegressor)


def test_cached_fit_predict_raises_the_named_reason():
    Xtr, ytr, Xva, _yva, w = _data()
    with pytest.raises(nc.WeightsNotAccepted, match='not run'):
        nc.cached_fit_predict('t', 'gp', GaussianProcessRegressor, Xtr, ytr,
                              Xva, sample_weight=w)
    # Unweighted gp still fits.
    pred = nc.cached_fit_predict('t', 'gp', GaussianProcessRegressor, Xtr, ytr,
                                 Xva)
    assert np.all(np.isfinite(pred))


def test_d_vs_dprime_row_records_not_run(monkeypatch):
    from experiments.noise import d_vs_dprime as dvd
    import experiments._shared as sh
    Xtr, ytr, Xva, yva, w = _data()
    monkeypatch.setattr(sh, 'get_model_factories', lambda names, *a, **k: {
        'gp': GaussianProcessRegressor,
        'rf': lambda: RandomForestRegressor(n_estimators=10, random_state=0)})
    cleaned = {'X': Xtr, 'y': ytr, 'sample_weight': w}
    metric = ('rmse', _rmse, True)
    out = dvd.score_d_dprime('t', cleaned, Xtr, ytr, Xva, yva, metric=metric,
                             task_type='regression', y_tr_clean=ytr,
                             models=('gp', 'rf'))
    assert np.isnan(out['gp']['score_Dprime'])
    assert out['gp']['dprime_not_run'] == nc.WEIGHTS_NOT_RUN
    assert np.isfinite(out['gp']['score_D'])
    assert np.isfinite(out['rf']['score_Dprime'])
    assert 'dprime_not_run' not in out['rf']
    ident = {k: 'x' for k in dvd.KEY_FIELDS if k != 'model'}
    rows = {r['model']: r for r in dvd.rows_for(out, metric=metric, **ident)}
    assert rows['gp']['dprime_not_run'] == nc.WEIGHTS_NOT_RUN
    assert 'dprime_not_run' not in rows['rf']


def test_sim_axis_rows_record_not_run():
    from experiments.noise import _sim_axis_n15_n22 as sx
    from experiments.noise import d_vs_dprime as dvd
    Xtr, ytr, Xva, yva, w = _data()

    class Sink:
        def is_done(self, *a):
            return False

        def write(self, rows):
            self.rows = rows
            return len(rows)

    sink = Sink()
    ident = {k: 'x' for k in dvd.KEY_FIELDS if k != 'model'}
    sx.dprime_rows(sink, 't', ident, {'X': Xtr, 'y': ytr, 'sample_weight': w},
                   X_tr=Xtr, y_D=ytr, X_va=Xva, y_va=yva,
                   metric=('rmse', _rmse, True),
                   factories={'gp': GaussianProcessRegressor})
    (r,) = sink.rows
    assert np.isnan(r['score_Dprime'])
    assert r['dprime_not_run'] == nc.WEIGHTS_NOT_RUN


def test_n24_fit_score_returns_the_reason():
    from experiments.noise.n24_delete_or_keep import _fit_score
    Xtr, ytr, Xva, yva, w = _data()
    s, why = _fit_score(GaussianProcessRegressor,
                        {'X': Xtr, 'y': ytr, 'sample_weight': w}, Xva, yva,
                        _rmse, 'rmse')
    assert s is None and why == nc.WEIGHTS_NOT_RUN


def test_n32_direct_fit_raises_the_named_reason():
    from experiments.noise.n32_iterative_and_dual_selection import (
        fit_predict_cached)
    Xtr, ytr, Xva, _yva, w = _data()
    with pytest.raises(nc.WeightsNotAccepted):
        fit_predict_cached(None, 'gp', GaussianProcessRegressor, Xtr, ytr, Xva,
                           'rmse', sample_weight=w)
