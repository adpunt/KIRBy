"""c22 SHAP on every model (2026-10-07).

dnn: DeepExplainer on its torch network, every held-out molecule.
svm and gp: KernelExplainer on 50 fixed held-out molecules.
In each case a representation whose features determine y gets most of the
SHAP mass, and the explained rows are the same on a second call.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('src', 'tests', os.path.join('tests', 'experiments')):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


def _c22():
    import experiments.complexity.c22_explainability as c22
    return c22


def _data(n=260, n_signal=3, n_noise=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, n_signal + n_noise)).astype(np.float32)
    y = X[:, :n_signal] @ np.array([1.0, -1.0, 0.8])[:n_signal]
    fi = {
        'signal_rep': {'selected_indices': np.arange(n_signal),
                       'n_features': n_signal},
        'noise_rep': {'selected_indices': np.arange(n_noise),
                      'n_features': n_noise},
        'allocation': {'signal_rep': n_signal, 'noise_rep': n_noise},
    }
    return X[:180], X[180:], y[:180], y[180:], fi


def test_method_per_model():
    c22 = _c22()
    assert c22.shap_method_for('rf') == 'TreeExplainer'
    assert c22.shap_method_for('dnn') == 'DeepExplainer'
    assert c22.shap_method_for('svm') == 'KernelExplainer'
    assert c22.shap_method_for('gp') == 'KernelExplainer'
    assert not hasattr(c22, 'SHAP_NOT_RUN')


def test_kernel_rows_fixed_per_fold():
    c22 = _c22()
    a = c22.kernel_explained_rows(0, 200)
    assert len(a) == 50 and len(set(a)) == 50
    assert np.array_equal(a, c22.kernel_explained_rows(0, 200))
    assert not np.array_equal(a, c22.kernel_explained_rows(1, 200))
    assert len(c22.kernel_explained_rows(0, 30)) == 30
    assert c22.kernel_nsamples(250) == 2548
    assert c22.kernel_nsamples(2000) == c22.KERNEL_NSAMPLES_CAP


def _shares(c22, fi, out):
    from kirby.importance.hybrid import aggregate_importance_by_rep
    return aggregate_importance_by_rep(fi, out['shap_mean_abs'])


def test_svm_kernel_explainer_50_rows():
    from kirby.hybrid import SVRScaledTarget
    c22 = _c22()
    Xt, Xv, yt, yv, fi = _data()
    m = SVRScaledTarget(kernel='linear').fit(Xt, yt)
    out = c22._shap_mean_abs('svm', m, Xt, Xv, fold=0)
    assert out['shap_error'] is None
    assert out['shap_method'] == 'KernelExplainer'
    assert out['shap_n_explained'] == 50
    assert out['shap_nsamples'] == c22.kernel_nsamples(Xv.shape[1])
    assert out['shap_background_k'] == c22.KERNEL_BG_K
    sh = _shares(c22, fi, out)
    assert sh['signal_rep'] > 0.9
    again = c22._shap_mean_abs('svm', m, Xt, Xv, fold=0)
    assert np.allclose(again['shap_mean_abs'], out['shap_mean_abs'],
                       rtol=0.05, atol=1e-3)


def test_gp_kernel_explainer():
    from kirby.find_best_hybrid import _get_gp_regressor
    c22 = _c22()
    Xt, Xv, yt, yv, fi = _data()
    m = _get_gp_regressor().fit(Xt, yt)
    out = c22._shap_mean_abs('gp', m, Xt, Xv, fold=2)
    assert out['shap_error'] is None
    assert out['shap_n_explained'] == 50
    assert _shares(c22, fi, out)['signal_rep'] > 0.9


@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_dnn_deep_explainer_all_rows(task):
    from kirby.models.dnn import DNNRegressor, DNNClassifier
    c22 = _c22()
    Xt, Xv, yt, yv, fi = _data()
    if task == 'regression':
        m = DNNRegressor(epochs=60, hidden_sizes=(32, 16)).fit(Xt, yt)
    else:
        m = DNNClassifier(epochs=60, hidden_sizes=(32, 16)).fit(
            Xt, (yt > 0).astype(int))
    out = c22._shap_mean_abs('dnn', m, Xt, Xv, fold=0)
    assert out['shap_error'] is None, out['shap_error']
    assert out['shap_method'] == 'DeepExplainer'
    assert out['shap_n_explained'] == len(Xv)
    assert _shares(c22, fi, out)['signal_rep'] > 0.5
