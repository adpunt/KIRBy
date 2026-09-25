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
  8. (2026-09-24) THE THREE NEWER ARMS. Method of moments recovers known
     sigma2 and tau2; the random-effect penalty shrinks toward 0 when the
     offsets have no variance; the quantile map leaves small documents alone
     and puts a large one on the pooled quantiles; within_document_rank is
     scored for xgboost only.
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

# 2026-09-25: n29 now caches through noise_cache under KIRBY_CACHE_DIR and
# checkpoints under the output directory, so the toy entries go to a
# temporary directory, as tests/smoke/smoke_noise_similarity_foundation.py
# does.
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n29_smoke_cache_')


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
        kw.setdefault('base_reps', {m: ['ecfp4', 'second']
                                    for m in kw.get('models', ['rf'])})
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


def test_new_arms_parts():
    print("the three newer arms, piece by piece:")
    rng = np.random.default_rng(7)
    n_doc, per = 60, 12
    b = rng.normal(0, 0.5, n_doc)                 # tau2 = 0.25
    rec_doc = np.repeat(np.arange(n_doc), per)
    resid = b[rec_doc] + rng.normal(0, 1.0, len(rec_doc))   # sigma2 = 1
    s2, t2 = n29.moment_variances(resid, rec_doc, list(range(n_doc)))
    check("method of moments recovers sigma2 near 1 and tau2 near 0.25",
          abs(s2 - 1.0) < 0.15 and abs(t2 - 0.25) < 0.12,
          f"sigma2 {s2:.3f}, tau2 {t2:.3f}")
    s2n, t2n = n29.moment_variances(rng.normal(0, 1, len(rec_doc)), rec_doc,
                                    list(range(n_doc)))
    check("no offsets -> tau2 small", t2n < 0.05, f"tau2 {t2n:.4f}")

    y = np.concatenate([rng.normal(0, 1, 30), rng.normal(3, 0.2, 25),
                        rng.normal(0, 1, 5)])
    d = np.array([0] * 30 + [1] * 25 + [2] * 5)
    mapped, nd, nm = n29.quantile_map_labels(y, d, min_measurements=20)
    check("quantile map: 2 documents of 20+ mapped, 55 measurements",
          nd == 2 and nm == 55, f"{nd} documents, {nm} measurements")
    check("quantile map leaves the 5-measurement document untouched",
          np.array_equal(mapped[d == 2], y[d == 2]))
    check("quantile map keeps each document's order",
          np.array_equal(np.argsort(mapped[d == 1]), np.argsort(y[d == 1])))
    check("a mapped document's values sit inside the pooled range",
          mapped[d == 1].min() >= y.min() and mapped[d == 1].max() <= y.max())

    data, b_true, rec_mol, rdoc = make_data(seed=5)
    comp, info = n29.document_groups(rec_mol, rdoc)
    elig = n29.eligible_documents(comp, info)
    vt = []
    off, _r, _t = n29.fit_document_offsets(
        data['reps']['ecfp4'], data['labels'], rdoc, data['groups'], elig,
        comp, n_iter=3, alpha='fitted', oof_splits=3, variance_trace=vt)
    by_c = {}
    for dd in elig:
        by_c.setdefault(comp[dd], []).append(b_true[dd])
    truth = np.array([b_true[dd] - np.mean(by_c[comp[dd]]) for dd in elig])
    r = float(np.corrcoef(truth, [off[dd] for dd in elig])[0, 1])
    check("random-effect offsets correlate with the true ones above 0.8",
          r > 0.8, f"pearson {r:.3f}")
    check("one variance entry per pass, each with sigma2, tau2, penalty",
          len(vt) == 3 and all({'sigma2', 'tau2', 'penalty_in_measurements'}
                               <= set(v) for v in vt), str(vt))


def test_run():
    print("a whole run:")
    data, _, _, _ = make_data(seed=2)
    res = run_it(data, 'toy_docs', models=['rf', 'lightgbm', 'xgboost'],
                 n_bootstrap=20, oof_splits=3)
    check("both single-representation spaces ran, and are named",
          res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {
              m: ['ecfp4', 'second'] for m in ('rf', 'lightgbm', 'xgboost')},
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check("base_2 is scored", 'mean_label' in res['scores']['all'].get(
        'base_2', {}))
    sc = res['scores']['all']['base_1']
    label_arms = [a for a in n29.ARMS if a != 'within_document_rank']
    check("every label arm scored for every model in base_1",
          all(len(sc[a][m]) == 3 for a in label_arms
              for m in ('rf', 'lightgbm', 'xgboost')),
          str({a: {m: len(v) for m, v in sc[a].items()} for a in sc}))
    check("within_document_rank scored for xgboost only",
          set(sc['within_document_rank']) == {'xgboost'}
          and len(sc['within_document_rank']['xgboost']) == 3,
          str(res.get('_errors')))
    fb0 = res['per_fold']['f0']
    check("quantile-map counts and random-effect variances written per fold",
          'document_quantile_map' in fb0
          and 'random_effect' in fb0['per_space']['base_1']
          and 'within_document_rank' in fb0['per_space']['base_1'],
          str(list(fb0['per_space']['base_1'])))
    ok = all(fb['held_out_molecules_isolated'] <= fb['held_out_molecules']
             and fb['documents_given_an_offset'] <= fb['training_documents']
             for fb in res['per_fold'].values())
    check("counts bounded by what exists", ok)
    kb = res.get('known_bias_check') or {}
    check("the check with known biases ran in rf's rank-1 representation",
          kb.get('model') == 'rf' and kb.get('rep') == 'ecfp4',
          f"{kb.get('model')}, {kb.get('rep')}")
    check("the check with known biases ran and recovers them",
          kb.get('pearson_added_vs_fitted_minus_real', 0) > 0.5,
          f"{kb.get('pearson_added_vs_fitted_minus_real')}")
    s = res['summary']['per_view']['all']['base_1']['rf']['per_arm']
    check("summary pairs every label arm against mean_label for rf",
          set(s) == set(n29.ARMS) - {'mean_label', 'within_document_rank'}
          and all(v['n_folds_paired'] == 3 for v in s.values()), str(set(s)))
    sx = res['summary']['per_view']['all']['base_1']['xgboost']['per_arm']
    check("summary pairs within_document_rank against xgboost's mean_label",
          sx.get('within_document_rank', {}).get('n_folds_paired') == 3)

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
                                        '--n-bootstrap', '--tag')),
          out.stderr[-300:])
    check("--base-rep is gone", '--base-rep' not in out.stdout)


# ── the D-versus-D' rows and the checkpoint (2026-09-25) ────────────────────

def test_rows_and_checkpoint():
    """Plan section 2 on n29: the run writes D-versus-D' rows beside its
    result file, each with the metric direction and a gain signed by it, and
    leaves no per-pass checkpoint behind once the result is complete."""
    import json
    print("\nthe D-versus-D' rows and the checkpoint:")
    res = run_it(make_data(seed=2)[0], 'toy_rows', models=['rf'], n_bootstrap=5, oof_splits=3)
    path = res['rows']['rows_file']
    rows = ([json.loads(l) for l in open(path)]
            if path and os.path.exists(path) else [])
    check('rows were written to the rows file', len(rows) > 0, str(path))
    methods = {x['method'] for x in rows}
    check('the rows name the methods that change the training data',
          {'per_record', 'per_record_bias', 'mean_label_bias_corrected'} <= methods, str(sorted(methods)))
    check('every row carries the direction the dataset\'s metric has',
          all(x['lower_is_better'] is bool(res['lower_is_better'])
              for x in rows), str(res['lower_is_better']))
    ok = all((x['gain'] != x['gain']) or abs(
        x['gain'] - ((x['score_D'] - x['score_Dprime'])
                     if x['lower_is_better']
                     else (x['score_Dprime'] - x['score_D']))) < 1e-12
             for x in rows)
    check('gain is score_D against score_Dprime, signed by the direction', ok)
    check('real labels: no clean score exists, so score_clean is nan',
          all(x['score_clean'] != x['score_clean'] for x in rows))
    ck = os.path.join(OUT, 'n29_document_bias', '_checkpoints')
    check('no checkpoint is left once the result is complete',
          not (os.path.isdir(ck) and os.listdir(ck)), ck)


if __name__ == '__main__':
    test_groups()
    test_ridge()
    test_recovery()
    test_order()
    test_new_arms_parts()
    test_run()
    test_guards()
    test_help()
    test_rows_and_checkpoint()
    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
