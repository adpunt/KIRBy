"""Guard test: every experiment result JSON must persist its metric DIRECTION.

The recurring sign bug (lower-is-better RMSE/MAE molecular results read as
regressions) is caused by result JSON not recording `lower_is_better`, forcing
analysis code to re-guess it. `save_experiment_results` now stamps
`metric_name` + `lower_is_better` (derived from the dataset registry when not
passed). This test locks that behaviour in.

Run: python tests/test_metric_direction_persisted.py   (also works under pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'experiments'))
from experiments._shared import save_experiment_results  # noqa: E402


def _save(results, dataset, **kw):
    with tempfile.TemporaryDirectory() as d:
        fp = save_experiment_results(results, d, 'c99_test', dataset, **kw)
        return json.loads(Path(fp).read_text())


def test_lower_better_dataset_stamped_from_registry():
    # caco2 is molecular regression -> rmse, lower_is_better=True
    out = _save({'some': 'plain_dict'}, 'caco2')
    assert out['lower_is_better'] is True, out.get('lower_is_better')
    assert out['metric_name'] == 'rmse', out.get('metric_name')


def test_higher_better_dataset_stamped_from_registry():
    # abagym_Ang2 is antibody regression -> spearman, lower_is_better=False
    out = _save({'some': 'plain_dict'}, 'abagym_Ang2')
    assert out['lower_is_better'] is False, out.get('lower_is_better')
    assert out['metric_name'] == 'spearman', out.get('metric_name')


def test_explicit_value_not_clobbered():
    # A results dict that already declares direction must win over the registry.
    out = _save({'lower_is_better': False, 'metric_name': 'custom'}, 'caco2')
    assert out['lower_is_better'] is False
    assert out['metric_name'] == 'custom'


def test_caller_override_used():
    out = _save({'x': 1}, 'caco2', metric_name='mae', lower_is_better=True)
    assert out['lower_is_better'] is True
    assert out['metric_name'] == 'mae'


def test_unregistered_dataset_does_not_crash():
    # No registry entry -> no direction stamped, but must not raise.
    out = _save({'x': 1}, 'not_a_real_dataset_xyz')
    assert 'x' in out  # saved fine
    # direction simply absent (or None); the key thing is no exception


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
