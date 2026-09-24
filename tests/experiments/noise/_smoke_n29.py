#!/usr/bin/env python3
"""Smoke for n29: one offset per ChEMBL document.

What has to be true before any n29 number can be read:

  1. THE GROUPS OF DOCUMENTS ARE THE CONNECTED ONES. Two documents that
     measured a common molecule are in one group; a document that shares no
     molecule is alone, and alone it gets no offset.
  2. THE RIDGE STEP IS RIDGE. The closed form sum / (count + alpha) equals
     sklearn's Ridge on a one-hot document membership, before centring.
  3. OFFSETS SUM TO ZERO WITHIN EVERY GROUP.
  4. KNOWN OFFSETS ARE RECOVERED. On toy data built with one offset per
     document, the fitted offsets correlate with the true ones.
  5. The run writes every arm for every model, in both views, and the counts
     are bounded by what exists. The hybrid path is reachable.
  6. The guards: no document ids, a length mismatch, a class label.
  7. The parser exposes its flags.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process kill it with exit 139 and no traceback.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import subprocess
import tempfile

import numpy as np

import experiments.noise.n29_document_bias as n29
from experiments.noise.n19_measurement_count import measurement_stats
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n29_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n_mol=160, n_doc=24, n_feat=6, bias_sd=0.8, seed=0):
    """Molecules measured 1-3 times, each measurement from one document.

    The label of a measurement is a function of the molecule's features, plus
    its document's offset, plus small noise. Molecules are placed in 40
    scaffold groups so the folds are grouped.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_mol, n_feat))
    w = rng.normal(size=n_feat)
    f = X @ w
    b_true = rng.normal(0, bias_sd, n_doc)
    scaffold = rng.integers(0, 40, n_mol)
    rows_mol, rows_doc = [], []
    for m in range(n_mol):
        k = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        for d in rng.choice(n_doc, size=k, replace=False):
            rows_mol.append(m)
            rows_doc.append(int(d))
    rows_mol = np.array(rows_mol)
    rows_doc = np.array(rows_doc)
    y = f[rows_mol] + b_true[rows_doc] + rng.normal(0, 0.1, len(rows_mol))
    Xr = X[rows_mol]
    return {
        'labels': y,
        'reps': {'ecfp4': Xr, 'second': Xr[:, ::-1] + rng.normal(0, 0.3, Xr.shape)},
        'groups': scaffold[rows_mol],
        'replicate_groups': rows_mol,
        'record_documents': np.array([f'DOC{d}' for d in rows_doc]),
        'task_type': 'regression',
        'metric': 'rmse',
        'domain': 'molecular',
    }, b_true, rows_mol, rows_doc


def fake_recipe(rep_names, budget=4):
    bp = Blueprint(reps=list(rep_names), allocation_method='fixed',
                   importance_method='lightgbm_gain', budget=budget,
                   internal_evaluator='lightgbm', metric='rmse',
                   lower_is_better=True, task_type='regression',
                   augmentation_strategy='none')
    meta = {'best_config': {'blueprint': bp},
            'objective': 'predictive', 'objective_choice': 'smoke_fixed_recipe',
            'pool_subset': 'predictive_and_similarity', 'tier': 'general',
            'aug_condition': 'none', 'by_objective': {}}
    return lambda *a, **k: (None, meta)


def no_recipe(*a, **k):
    return None, None


def run_it(data, name, recipe=no_recipe, **kw):
    real = n29.get_best_search_for_downstream
    n29.get_best_search_for_downstream = recipe
    try:
        return n29.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n29.get_best_search_for_downstream = real


def test_groups():
    print("groups of documents:")
    # molecules 0,1,2; documents 0,1,2,3
    # doc0 & doc1 share mol0; doc1 & doc2 share mol1; doc3 alone on mol2
    rec_mol = np.array([0, 0, 1, 1, 2])
    rec_doc = np.array([0, 1, 1, 2, 3])
    comp, info = n29.document_groups(rec_mol, rec_doc)
    check("documents joined through a shared molecule are one group",
          comp[0] == comp[1] == comp[2])
    check("a document sharing no molecule is alone", comp[3] != comp[0])
    check("group sizes counted in documents, molecules, measurements",
          info[comp[0]] == {'documents': 3, 'molecules': 2, 'measurements': 4}
          and info[comp[3]] == {'documents': 1, 'molecules': 1,
                                'measurements': 1},
          str(info))
    elig = n29.eligible_documents(comp, info)
    check("only documents in a group of 2+ get an offset", elig == [0, 1, 2],
          str(elig))


def test_ridge():
    print("the ridge step:")
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(1)
    rec_doc = rng.integers(0, 6, 80)
    resid = rng.normal(size=80) + rec_doc * 0.3
    comp = {d: 0 if d < 4 else 1 for d in range(6)}
    elig = list(range(6))
    got = n29.ridge_offsets(resid, rec_doc, elig, comp, alpha=2.0)
    Z = np.eye(6)[rec_doc]
    raw = Ridge(alpha=2.0, fit_intercept=False).fit(Z, resid).coef_
    centred = raw.copy()
    centred[:4] -= raw[:4].mean()
    centred[4:] -= raw[4:].mean()
    check("closed form equals sklearn Ridge on a one-hot membership, centred",
          np.allclose([got[d] for d in elig], centred, atol=1e-10))
    check("offsets sum to zero within each group",
          abs(sum(got[d] for d in range(4))) < 1e-10
          and abs(sum(got[d] for d in range(4, 6))) < 1e-10)
    got_w = n29.ridge_offsets(resid, rec_doc, elig, comp, 2.0,
                              weight=np.where(rec_doc == 5, 0, 1))
    check("a document with no weight in a resample comes back NaN",
          np.isnan(got_w[5]))


def test_recovery():
    print("recovering known offsets:")
    data, b_true, rec_mol, rec_doc = make_data()
    comp, info = n29.document_groups(rec_mol, rec_doc)
    elig = n29.eligible_documents(comp, info)
    X_rec = data['reps']['ecfp4']
    off, resid, trace = n29.fit_document_offsets(
        X_rec, data['labels'], rec_doc, data['groups'], elig, comp,
        n_iter=3, alpha=1.0, oof_splits=3)
    by_c = {}
    for d in elig:
        by_c.setdefault(comp[d], []).append(b_true[d])
    truth = np.array([b_true[d] - np.mean(by_c[comp[d]]) for d in elig])
    fitted = np.array([off[d] for d in elig])
    r = float(np.corrcoef(truth, fitted)[0, 1])
    check("fitted offsets correlate with the true ones above 0.8",
          r > 0.8, f"pearson {r:.3f} over {len(elig)} documents")
    check("one trace entry per pass", len(trace) == 3, str(trace))


def test_order():
    print("molecule order:")
    data, _, _, _ = make_data(seed=4)
    stats = measurement_stats(data['labels'], data['replicate_groups'])
    rm = n29.record_to_molecule(data['replicate_groups'])
    first = np.array([np.where(rm == j)[0][0] for j in range(len(stats['row_index']))])
    check("record_to_molecule follows measurement_stats' order",
          np.array_equal(first, stats['row_index'])
          and np.array_equal(np.bincount(rm), stats['n_measurements']))


def test_run():
    print("a whole run:")
    data, _, _, _ = make_data(seed=2)
    res = run_it(data, 'toy_docs', models=['rf', 'lightgbm'], n_bootstrap=20,
                 oof_splits=3)
    sc = res['scores']['all']['base']
    check("every arm scored for every model in the base space",
          all(len(sc[a][m]) == 3 for a in n29.ARMS for m in ('rf', 'lightgbm')),
          str({a: {m: len(v) for m, v in sc[a].items()} for a in sc}))
    ok = all(fb['held_out_molecules_isolated'] <= fb['held_out_molecules']
             and fb['documents_given_an_offset'] <= fb['training_documents']
             for fb in res['per_fold'].values())
    check("counts bounded by what exists", ok)
    kb = res.get('known_bias_check') or {}
    check("the check with known biases ran and recovers them",
          kb.get('pearson_added_vs_fitted_minus_real', 0) > 0.5,
          f"{kb.get('pearson_added_vs_fitted_minus_real')}")
    s = res['summary']['per_view']['all']['base']['rf']['per_arm']
    check("summary pairs three arms against mean_label",
          set(s) == set(n29.ARMS) - {'mean_label'}
          and all(v['n_folds_paired'] == 3 for v in s.values()))

    res_h = run_it(data, 'toy_docs_h', recipe=fake_recipe(['ecfp4', 'second']),
                   models=['rf'], n_bootstrap=5, oof_splits=3)
    check("the hybrid space is reached",
          'hybrid' in res_h['scores']['all']
          and len(res_h['scores']['all']['hybrid']['per_record_bias']['rf']) == 3,
          str(res_h.get('_errors')))


def test_guards():
    print("guards:")
    data, _, _, _ = make_data(seed=3)
    d2 = dict(data)
    d2.pop('record_documents')
    r = run_it(d2, 'toy_nodoc', models=['rf'])
    check("no document ids -> skipped with a reason",
          r.get('skipped') == 'no_record_documents')
    d3 = dict(data)
    d3['record_documents'] = data['record_documents'][:-1]
    try:
        run_it(d3, 'toy_short', models=['rf'])
        check("a length mismatch raises", False)
    except RuntimeError:
        check("a length mismatch raises", True)
    d4 = dict(data)
    d4['task_type'] = 'classification'
    check("a class label is skipped",
          run_it(d4, 'toy_clf', models=['rf']).get('skipped') == 'not_regression')


def test_help():
    print("parser:")
    out = subprocess.run([sys.executable, n29.__file__, '--help'],
                         capture_output=True, text=True)
    check("--help lists the flags",
          all(f in out.stdout for f in ('--n-iter', '--ridge-alpha',
                                        '--n-bootstrap', '--base-rep', '--tag')),
          out.stderr[-300:])


if __name__ == '__main__':
    test_groups()
    test_ridge()
    test_recovery()
    test_order()
    test_run()
    test_guards()
    test_help()
    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
