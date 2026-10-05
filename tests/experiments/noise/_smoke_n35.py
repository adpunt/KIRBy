#!/usr/bin/env python3
"""Smoke for n35: near-duplicate label disagreement, and downweighting it.

What has to be true before any number n35 writes can be read:

  1. TANIMOTO DISTANCE IS THE FORMULA on 0/1 rows, 0 for identical rows, and
     two all-zero rows sit at 0.
  2. THE CLOSE PAIRS ARE THE BOTTOM 1% of the i < j distances, ties at the
     cut-off kept, and no pair is listed twice.
  3. THE ESTIMATE RISES BY THE INJECTED VARIANCE. On toy near-duplicates with a
     known label noise, adding independent noise of variance v to every label
     moves the closest-bin E[dy^2]/2 up by about v, at three values of v.
  4. THE FLAG RULE is |dy| > 3 * sqrt(2) * sigma, and both members of a flagged
     pair get weight 0.25, nobody else.
  5. THE POOLED REPLICATE VARIANCE matches a hand computation.
  6. CLIFF COUNTING on real SMILES: a similar pair with a large label gap is a
     cliff; a dissimilar one is not.
  7. A FULL RUN, single representations only (base_1 'toy', base_2 'second')
     and with a hybrid recipe, fingerprint and continuous representations,
     with repeat measurements: every arm and every level scored, tail scores
     written, same- and different-molecule pairs split, the check block
     filled. Since 2026-10-05 (chat 5) the per-fold searches and the decided
     models are replaced by fixed ones here (fake_fold_configurations,
     SMOKE_MODELS); the real ones need a real dataset.
  8. A classification label is skipped with its reason.
  9. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# The noise cache in a temporary directory, never the real one. Until
# 2026-10-05 this also set KIRBY_SIMILARITY_SETTINGS=model_input, a setting
# the kNN record cut; sweep_settings now raises on it.
import tempfile as _tf
os.environ['KIRBY_CACHE_DIR'] = _tf.mkdtemp(prefix='n35_smoke_cache_')

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import tempfile
import warnings

import numpy as np

warnings.simplefilter('ignore')

import experiments.noise.n35_near_duplicate_disagreement as n35
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n35_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n_proto=70, copies=3, n_feat=6, seed=0, binary=False,
              replicates=False, noise_sd=0.3):
    """Clusters of near-identical molecules; label = the prototype's value + noise.

    Groups are the prototype, so copies stay in one fold. With replicates=True
    every copy is a repeat measurement of ONE molecule (same features exactly).
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_proto, n_feat))
    f = P @ rng.normal(size=n_feat) + 5.0
    X, y, g = [], [], []
    for p in range(n_proto):
        for c in range(copies):
            jitter = 0.0 if replicates else rng.normal(0, 0.02, n_feat)
            X.append(P[p] + jitter)
            y.append(f[p] + rng.normal(0, noise_sd))
            g.append(p)
    X = np.asarray(X)
    if binary:
        # 64 bits from random hyperplanes: near-identical rows share most bits.
        H = rng.normal(size=(n_feat, 64))
        X = (X @ H > 0).astype(float)
    y = np.asarray(y)
    g = np.asarray(g)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    d = {'labels': y, 'reps': {'toy': X, 'second': X2}, 'groups': g,
         'task_type': 'regression', 'domain': 'molecular'}
    if replicates:
        d['replicate_groups'] = g.copy()
    return d


# The two models a toy dataset stands in with: it has no DECISIONS.md row.
SMOKE_MODELS = ['rf', 'lightgbm']


def fake_blueprint(rep_names, budget=4):
    return Blueprint(reps=list(rep_names), allocation_method='fixed',
                     importance_method='lightgbm_gain', budget=budget,
                     internal_evaluator='lightgbm', metric='rmse',
                     lower_is_better=True, task_type='regression',
                     augmentation_strategy='none')


def fake_fold_configurations(with_hybrid):
    """Stands in for noise_helpers.fold_configurations (2026-10-05, chat 5):
    every model gets base ['toy', 'second'] and, with_hybrid, one general
    hybrid of both. No search runs."""
    def _fc(data, ds, **kw):
        out = {}
        for m in SMOKE_MODELS:
            e = {'base': ['toy', 'second'], 'hybrids': [], 'recipes': {},
                 'unavailable': {}}
            if with_hybrid:
                e['hybrids'] = [{'label': 'smoke_fixed_recipe',
                                 'score_mean': None, 'budget': 4,
                                 'allocation_method': 'fixed',
                                 'importance_method': 'lightgbm_gain',
                                 'evaluator': 'lightgbm', 'n_reps': 2,
                                 'aug_condition': 'none'}]
                e['recipes'] = {'hybrid': (fake_blueprint(['toy', 'second']),
                                           'none')}
            else:
                e['unavailable'] = {'hybrid': 'smoke: no hybrid',
                                    'hybrid__general2': 'smoke: no hybrid'}
            out[m] = e
        return out
    return _fc


def run_it(data, name, with_hybrid=False, **kw):
    real_fc, real_dm = n35.fold_configurations, n35.decided_models
    n35.fold_configurations = fake_fold_configurations(with_hybrid)
    n35.decided_models = lambda ds, tier='general': list(SMOKE_MODELS)
    try:
        kw.setdefault('draws', [0])
        return n35.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n35.fold_configurations, n35.decided_models = real_fc, real_dm


def test_pure():
    print("the pure functions:")
    A = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    D = n35.tanimoto_distance_matrix(A)
    check('1. Tanimoto distance: identical 0, half-overlap 0.5, empty pair 0',
          abs(D[0, 2]) < 1e-7 and abs(D[0, 1] - 0.5) < 1e-7
          and abs(D[3, 4]) < 1e-7 and abs(D[0, 3] - 1.0) < 1e-7)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    Dm = n35.euclidean_distance_matrix(X)
    i, j, d, thr, total = n35.close_pairs(Dm)
    iu = np.triu_indices(300, 1)
    full = Dm[iu]
    expect = int(np.sum(full <= thr))
    check('2. close pairs are the bottom 1% of i<j distances, each listed once',
          d.size == expect and np.all(i < j)
          and len(set(zip(i.tolist(), j.tolist()))) == d.size
          and abs(d.size / total - 0.01) < 0.002,
          f'{d.size} of {total} pairs, threshold {thr:.4f}')

    data = make_data(n_proto=150, copies=3, noise_sd=0.3, seed=5)
    Xs = data['reps']['toy']
    y = data['labels']
    Dn = n35.euclidean_distance_matrix(Xs)
    rec0, _ = n35.pair_record(Dn, y)
    rises = []
    for v in (0.1, 0.25, 0.5):
        r_list = []
        for s in range(5):
            e = np.random.default_rng(100 + s).normal(0, np.sqrt(v), y.size)
            r, _ = n35.pair_record(Dn, y + e)
            r_list.append(r['closest_bin_estimate'] - rec0['closest_bin_estimate'])
        rises.append(float(np.mean(r_list)))
    ok = all(abs(r - v) < 0.35 * v for r, v in zip(rises, (0.1, 0.25, 0.5)))
    check('3. the closest-bin estimate rises by the injected variance',
          ok and rises[0] < rises[1] < rises[2],
          f'clean {rec0["closest_bin_estimate"]:.4f} (true noise 0.09); rises '
          f'{", ".join(f"{r:.3f}" for r in rises)} for 0.1, 0.25, 0.5')

    ii = np.array([0, 1, 2]); jj = np.array([3, 4, 5])
    dy = np.array([0.1, 2.0, -3.0])
    fl = n35.flag_pairs(ii, jj, dy, 0.25)   # sigma 0.5 -> cut 2.1213
    w = n35.pair_weights(8, ii, jj, fl)
    check('4. flag rule and weights',
          fl.tolist() == [False, False, True]
          and w.tolist() == [1, 1, 0.25, 1, 1, 0.25, 1, 1])

    yv = np.array([1.0, 3.0, 2.0, 5.0, 5.0, 7.0])
    gv = np.array([0, 0, 1, 2, 2, 2])
    pr = n35.pooled_replicate_variance(yv, gv)
    hand = ((1 - 2) ** 2 + (3 - 2) ** 2 + 2 * (5 - 17 / 3) ** 2
            + (7 - 17 / 3) ** 2) / (1 + 2)
    check('5. pooled replicate variance', abs(
        pr['pooled_within_molecule_variance'] - hand) < 1e-12
        and pr['molecules_measured_twice_or_more'] == 2)

    fps = n35.ecfp4_fingerprints(['Cc1ccc(cc1)C(=O)Nc1ccc(Cl)cc1CCN(C)C',
                                   'Cc1ccc(cc1)C(=O)Nc1ccc(Cl)cc1CCN(C)CC',
                                   'c1ccccc1', 'CCO'])
    if fps is None:
        check('6. cliff counting (RDKit missing, skipped)', True)
    else:
        ysn = np.array([5.0, 9.0, 5.2, 5.1, 5.0, 5.3, 4.9, 5.0])
        c = n35.count_cliff_pairs(np.array([0, 0]), np.array([1, 2]),
                                  np.array([-4.0, -4.0]), ysn, fps_train=fps)
        check('6. similar pair with a big gap is a cliff; dissimilar is not',
              c['flagged_pairs'] == 2 and c['flagged_cliff_pairs'] == 1,
              str(c))


def _per_space(r, level='0'):
    return r['pairs_per_pass'][level][0]['per_space']


def test_runs():
    print("\nfull runs:")
    T = n35.THIS_SPACE_ARM
    data = make_data(binary=False)
    r = run_it(data, 'toy_cont', levels=[0.0, 0.5], skip_cliffs=True)
    lv = r['per_level']
    ok = all(arm in lv[l]['base_1'] for l in ('0', '0.5')
             for arm in ('untouched', 'clean', T))
    n_scores = sum(len(v) for v in lv['0']['base_1']['untouched'].values())
    check('7a. continuous base: untouched, clean and this_space at every '
          'level, 3 folds x 2 models',
          ok and n_scores == 6, f'{n_scores} untouched scores at level 0')
    kept = [a for a in lv['0']['base_1'] if a.startswith(n35.ARM_PREFIX)]
    check('7a-2. the screen kept settings and they were refitted', bool(kept),
          str(kept))
    t = r['tails_per_level']['0']['base_1']['untouched']['rf']
    check('7b. tail scores written per pass',
          len(t) == 3 and all('calibration_slope' in v for v in t.values()))
    ch = r['check'].get(f'{T}|rf|base_1', {}).get('0.5')
    check('7c. the check holds the base_1 this_space rise beside the injected '
          'variance', ch is not None and len(ch['folds']) == 3
          and ch['estimate_rise']['mean'] is not None
          and ch['pair_set_reads_labels'] is False,
          (f"rise {ch['estimate_rise']['mean']:.3f}, injected "
           f"{ch['realised_injected_variance']['mean']:.3f}") if ch else '')
    ps = _per_space(r)
    check('7d. this_space is Euclidean on a continuous representation',
          ps.get(f'{T}|rf|base_1', {}).get('distance')
          == 'euclidean_on_fitted_matrix' and not r.get('_errors'),
          str(r.get('_errors', ''))[:300])
    check('7d-2. base_2 scores every arm at every level',
          all(a in lv[l]['base_2'] for l in ('0', '0.5')
              for a in ('untouched', T)))
    check('7d-3. one forest proximity per fitted configuration',
          any(k.startswith('forest_proximity') and k.endswith('|rf|base_1')
              for k in ps)
          and any(k.startswith('forest_proximity')
                  and k.endswith('|lightgbm|base_2') for k in ps),
          str(sorted(k for k in ps if k.startswith('forest'))))
    check('7d-4. fitted configurations per fold = 2 models x 4',
          r['fitted_configurations_per_fold'] == 8
          and r['per_fold']['0']['models']['rf']['base'] == ['toy', 'second'])

    datab = make_data(binary=True, replicates=True, seed=2)
    rb = run_it(datab, 'toy_bin', with_hybrid=True, levels=[0.0, 0.5])
    psb = _per_space(rb)
    check('7e. a binary base representation uses Tanimoto, the hybrid '
          'Euclidean', psb.get(f'{T}|rf|base_1', {}).get('distance')
          == 'tanimoto' and psb.get(f'{T}|rf|hybrid', {}).get('distance')
          == 'euclidean_on_fitted_matrix',
          str({k: v.get('distance') for k, v in psb.items()
               if k.startswith(T)}))
    check('7f. a hybrid configuration ran with untouched and this_space',
          'hybrid' in rb['per_level']['0']
          and T in rb['per_level']['0']['hybrid'],
          str(list(rb['per_level']['0'])))
    b = psb[f'{T}|rf|base_1']
    rec = rb['pairs_per_pass']['0'][0]
    check('7g. repeat measurements: same- and different-molecule pairs split, '
          'pooled replicate variance written',
          'same_molecule_pairs' in b and 'different_molecule_pairs' in b
          and 'replicate_variance_seen' in rec
          and b['same_molecule_pairs']['n_close_pairs'] > 0,
          f"same {b.get('same_molecule_pairs', {}).get('n_close_pairs')}")
    check('7h. cliffs counted through n25 when there are no SMILES',
          'cliffs' in b, str(b.get('cliffs')))
    check('7i. no errors in the binary + hybrid run',
          not rb.get('_errors'), str(rb.get('_errors', ''))[:400])

    dc = make_data()
    dc['task_type'] = 'classification'
    rc = n35.run(dc, {'n_splits': 3}, OUT, 'toy_clf')
    check('8. classification skipped with its reason',
          rc.get('skipped') == 'not_regression')


def test_parser():
    print("\nthe parser:")
    import subprocess
    out = subprocess.run([sys.executable, n35.__file__, '--help'],
                         capture_output=True, text=True).stdout
    flags = ('--levels', '--draws', '--skip-proximity', '--skip-cliffs',
             '--exp-folds', '--tag')
    missing = [f for f in flags if f not in out]
    check('9. every flag is in --help', not missing, f'missing {missing}')
    check('9b. --base-rep is gone', '--base-rep' not in out)



def _rows_of(res):
    """The D-versus-D' rows a run wrote (2026-09-25)."""
    import json
    p = (res.get('d_vs_dprime') or {}).get('rows_file')
    if not p or not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _with_settings(names, fn):
    """Run fn() under KIRBY_SIMILARITY_SETTINGS=names, then restore it."""
    old = os.environ.get('KIRBY_SIMILARITY_SETTINGS')
    os.environ['KIRBY_SIMILARITY_SETTINGS'] = names
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop('KIRBY_SIMILARITY_SETTINGS', None)
        else:
            os.environ['KIRBY_SIMILARITY_SETTINGS'] = old



def _close(a, b, rtol=1e-12):
    """a == b, floats to rtol. rf's default factory predicts with n_jobs=-1,
    and its threads sum the trees in no fixed order, so two fits of the same
    forest can differ in the last bit (measured 2026-09-25: 1.402832974397575
    against 1.4028329743975747). That is not a change of number."""
    import math
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (a == b or (math.isnan(a) and math.isnan(b))
                or math.isclose(a, b, rel_tol=rtol, abs_tol=0.0))
    return a == b

def test_rows_and_resume():
    """The D-versus-D' rows, and a rerun that resumes every pass."""
    print("\nrows and resume:")
    from experiments.noise.d_vs_dprime import ROW_FIELDS
    kw = dict(levels=[0.0, 0.5], draws=[0], skip_cliffs=True,
              skip_proximity=True)
    res = run_it(make_data(), 'toy_rows', **dict(kw))
    rows = _rows_of(res)
    check('R1. D-vs-D\' rows are written, each with exactly the d_vs_dprime '
          'columns, D and D\' both scored, the model its own column',
          bool(rows) and all(
              set(r) == set(ROW_FIELDS) and r['score_D'] is not None
              for r in rows) and set(SMOKE_MODELS) <= {r['model'] for r in rows}
          and all('/' not in str(r.get('model_input')) for r in rows),
          f'{len(rows)} rows, methods {sorted({r["method"] for r in rows})[:4]}')
    sims = {r.get('similarity_setting') for r in rows
            if r.get('method') == 'downweight_pairs'}
    check('R1-2. a downweight_pairs@<setting> row keeps the whole setting name',
          any(str(x).count('|') >= 2 for x in sims), str(sorted(map(str, sims))))
    again = run_it(make_data(), 'toy_rows', **dict(kw))
    got = again['d_vs_dprime']
    check('R2. a rerun resumes every pass from the checkpoint and returns the '
          'same scores', got['passes_resumed_from_checkpoint'] > 0
          and got['rows_written_this_run'] == 0
          and _close(again['per_level'], res['per_level']),
          f"{got['passes_resumed_from_checkpoint']} passes resumed")


if __name__ == '__main__':
    test_pure()
    test_runs()
    test_rows_and_resume()
    test_parser()
    print("\n" + ("ALL GREEN" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
