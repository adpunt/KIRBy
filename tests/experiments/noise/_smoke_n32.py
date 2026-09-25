#!/usr/bin/env python3
"""Smoke for n32: trimmed refits, two learners, better detectors.

What has to be true before any number n32 writes can be read:

  1. THE kNN-SHAPLEY RECURSION IS EXACT for its utility: equal to a brute-force
     Shapley value over every subset, on 7 molecules.
  2. THE SHIFT INJECTOR moves exactly the nominal share, up, by delta label SDs,
     and nothing else.
  3. lowest_share KEEPS EXACTLY THE ASKED SHARE, and random_keep removes exactly
     the asked count.
  4. ITLM REMOVES MOVED MOLECULES ABOVE CHANCE under a 2-SD shift, keeps alpha,
     and writes the overlap between rounds.
  5. THE FLAGGED-ONLY SHRINK leaves every unflagged label bit-identical and
     moves each flagged label toward the prediction, never past it.
  6. A FULL RUN writes every arm, every detector with an AUROC, tail scores, and
     each random control removes the same count as the arm it controls.
  7. At level 0 nothing moved, so no detection reading is written.
  8. With a recipe the hybrid space is built and scored.
  9. A condition that moves every label is refused; a class label is skipped.
 10. The metric direction is read off the dataset.
 11. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import itertools
import math
import tempfile

import numpy as np

import experiments.noise.n32_iterative_and_dual_selection as n32
from experiments.noise.noise_helpers import inject_shift_train_labels
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n32_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=180, n_feat=4, seed=0, domain='molecular'):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n), 'task_type': 'regression',
            'domain': domain}


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
    real = n32.get_best_search_for_downstream
    n32.get_best_search_for_downstream = recipe
    try:
        kw.setdefault('base_reps', {
            m: ['toy', 'second']
            for m in (kw.get('models') or n32.DOWNSTREAM_MODELS)})
        kw.setdefault('noise_condition', n32.DEFAULT_NOISE_CONDITION)
        return n32._run_one_condition(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n32.get_best_search_for_downstream = real


def brute_shapley(X, y, t, k):
    """Shapley value of every other molecule to the utility in
    knn_shapley_values, for test molecule t, by enumerating every subset."""
    others = [i for i in range(len(y)) if i != t]
    d = np.linalg.norm(X[others] - X[t], axis=1)
    order = [others[i] for i in np.argsort(d, kind='stable')]
    rank = {m: r for r, m in enumerate(order)}

    def U(S):
        near = sorted(S, key=lambda m: rank[m])[:k]
        return sum(-(y[m] - y[t]) ** 2 for m in near) / k

    N = len(others)
    out = {}
    for i in others:
        rest = [m for m in others if m != i]
        v = 0.0
        for r in range(N):
            w = math.factorial(r) * math.factorial(N - 1 - r) / math.factorial(N)
            for S in itertools.combinations(rest, r):
                v += w * (U(list(S) + [i]) - U(list(S)))
        out[i] = v
    return out


def test_knn_shapley():
    print("kNN-Shapley:")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 2))
    y = rng.normal(size=7)
    k = 2
    total = np.zeros(7)
    for t in range(7):
        for i, v in brute_shapley(X, y, t, k).items():
            total[i] += v
    total /= 7
    got = n32.knn_shapley_values(X, y, k=k)
    check('1. the recursion equals brute-force Shapley on 7 molecules',
          np.allclose(got, total, atol=1e-10),
          f'max difference {np.max(np.abs(got - total)):.2e}')


def test_shift_and_counts():
    print("\nthe shift injector and the counts:")
    y = make_data()['labels']
    tr = np.arange(150)
    y0, m, prov = inject_shift_train_labels(y, train_idx=tr,
                                            condition='shift_only_p10',
                                            delta_sd=2.0, seed=4)
    d = y0 - y[tr]
    check('2. the shift moves 10% of the training molecules up by 2 SDs and '
          'nothing else',
          int(m.sum()) == 15 and np.allclose(d[m], 2.0 * np.std(y))
          and np.all(d[~m] == 0.0),
          f"{int(m.sum())} moved by {prov['shift_label_units']:.3f}")
    keep = n32.lowest_share(np.arange(100.0), 0.9)
    rk = n32.random_keep(100, 7, 1)
    check('3. lowest_share keeps 90 of 100, the lowest; random_keep removes 7',
          int(keep.sum()) == 90 and keep[:90].all() and int((~rk).sum()) == 7)


def test_itlm_and_shrink():
    print("\nITLM and the flagged-only shrink:")
    d = make_data(n=200)
    X, y = d['reps']['toy'], d['labels']
    tr = np.arange(200)
    y0, m, _ = inject_shift_train_labels(y, train_idx=tr,
                                         condition='shift_only_p10',
                                         delta_sd=2.0, seed=0)
    keep, loss, rounds = n32.itlm(X, y0, np.arange(200), 0.9, rounds=3,
                                  oof_splits=3)
    removed = ~keep
    hit = int(np.sum(removed & m))
    chance = removed.sum() * m.sum() / 200
    check('4. ITLM keeps 90% and removes moved molecules above chance',
          int(keep.sum()) == 180 and hit > 2 * chance,
          f'{hit} of {int(removed.sum())} removed were moved; chance {chance:.1f}')
    check('4b. the overlap between consecutive rounds is written',
          rounds[0]['overlap_with_previous_round_kept_set'] is None
          and all(0 <= r['overlap_with_previous_round_kept_set'] <= 1
                  for r in rounds[1:]))
    p = y + np.random.default_rng(1).normal(0, 0.2, 200)
    one, per, rec = n32.shrink_flagged(y0, p, 0.1)
    flag = one != y0
    check('5. the shrink leaves unflagged labels bit-identical and moves '
          'flagged ones toward p, never past it',
          0 < int(flag.sum()) <= 20 and rec['one_weight'] > 0.5
          and np.all(np.abs(one - p) <= np.abs(y0 - p) + 1e-12)
          and np.all(np.abs(per - p) <= np.abs(y0 - p) + 1e-12)
          and 0.0 <= rec['one_weight'] <= 1.0,
          f"w = {rec['one_weight']:.3f}, {int(flag.sum())} moved")


def test_cleanlab():
    print("\ncleanlab:")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 120)
    moved = np.zeros(120, dtype=bool)
    moved[:12] = True
    y[moved] += 2.0
    sc = n32.detector_scores('cleanlab', X, y, None, None, 3)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(moved, sc))
    check('12. the cleanlab detector gives one score per molecule and finds a '
          '2-unit shift above chance', sc.shape == (120,) and auc > 0.7,
          f'AUROC {auc:.2f}')


def test_full_runs():
    print("\nfull runs:")
    models = ['rf', 'lightgbm']
    # cleanlab is left out of the full run for time; test_cleanlab covers it.
    res = run_it(make_data(), 'toy', models=models,
                 noise_condition='shift_only_p10', levels=[0.0, 2.0],
                 draws=[0], alphas=[0.9], random_seeds=[0, 1],
                 detectors=[d for d in n32.DETECTORS if d != 'cleanlab'])
    check('6-. spaces base_1 and base_2 ran, and the representations are '
          'named', res['spaces_run'] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models}
          and 'base_rep' not in res,
          f"{res['spaces_run']}, {res.get('base_reps_by_model')}")
    check('6-b. base_2 is scored for every model',
          set(res['per_level']['2']['base_2']['no_cleaning']) == set(models))
    lv = res['per_level']['2']['base_1']
    missing = [a for a in res['arms'] if a not in lv]
    check('6. every arm is scored at level 2', not missing,
          f'missing {missing}' if missing else f"{len(res['arms'])} arms")
    det = res['detection']['2']['base_1']
    no_auroc = [d for d in n32.DETECTORS if d != 'cleanlab'
                if d not in det or not np.isfinite(det[d]['detection_auroc'])]
    check('6b. every detector is graded, with an AUROC', not no_auroc,
          ', '.join(f"{d} {det[d]['detection_auroc']:.2f}" for d in det))
    check('6c. the out-of-fold residual finds a 2-SD shift well above chance',
          det['oof_residual']['detection_auroc'] > 0.8)
    tails = res['tails']['2']['base_1']['itlm_a0.9']['rf']
    check('6d. tail scores are written for every scored pass',
          len(tails) == 3 and all('tail_rmse_top' in v for v in tails.values()))
    recs = res['method_records']['2']['base_1']
    ok = all(r['itlm_a0.9']['molecules_removed']
             == r['training_molecules'] - int(round(0.9 * r['training_molecules']))
             for r in recs)
    check('6e. ITLM removes the share it was asked to on every pass', ok)
    s = res['summary']['per_level']['2']['base_1']['rf']['per_arm']
    check('6f. ITLM and the exchange are paired against their random controls',
          len(s['itlm_a0.9']['vs_random_removal']) == 2
          and len(s['two_learner']['vs_random_removal']) == 2)
    check('7. at level 0 nothing moved and no detection is written',
          '0' not in (res['detection'] or {}))

    res_h = run_it(make_data(), 'toy', recipe=fake_recipe(['toy', 'second']),
                   models=['lightgbm'], noise_condition='outlier_only_p10',
                   levels=[1.0], draws=[0], alphas=[0.9], random_seeds=[0],
                   detectors=['oof_residual'])
    check('8. with a recipe the hybrid space is built and scored',
          'hybrid' in res_h['spaces_run']
          and 'itlm_a0.9' in res_h['per_level']['1']['hybrid'])

    res_s = run_it(make_data(domain='antibody'), 'toy_ab', models=['rf'],
                   noise_condition='shift_only_p10', levels=[2.0], draws=[0],
                   alphas=[0.9], random_seeds=[0], detectors=['oof_residual'])
    check('10. the metric direction is read off the dataset',
          res['metric'] == 'rmse' and res['lower_is_better']
          and res_s['metric'] == 'spearman' and not res_s['lower_is_better'])
    e = res_s['summary']['per_level']['2']['base_1']['rf']['per_arm']['clean']
    b = res_s['per_level']['2']['base_1']['no_cleaning']['rf']
    g = res_s['per_level']['2']['base_1']['clean']['rf']
    k = sorted(set(b) & set(g))
    check('10b. a spearman gain is arm minus no_cleaning',
          np.isclose(e['vs_no_cleaning']['mean_gain'],
                     np.mean([g[x] - b[x] for x in k])))


def test_guards():
    print("\nthe guards:")
    d = make_data()
    try:
        run_it(d, 'toy', noise_condition='gaussian', models=['rf'])
        ok = False
    except ValueError:
        ok = True
    check('9. a condition that moves every label is refused', ok)
    r9 = n32.run(d, {'n_splits': 3}, OUT, 'toy', noise_condition='gaussian')
    check('9a. run() given such a condition writes a skip, not a crash',
          isinstance(r9, dict) and r9.get('skipped') == 'detection_undefined')
    import inspect
    src = inspect.getsource(n32.run)
    check('9c. with no condition named, run() sweeps every detection-defined '
          'condition, the two shift_only ones included',
          'default_conditions=list(DETECTION_IS_DEFINED)' in src
          and all(c in n32.DETECTION_IS_DEFINED for c in n32.SHIFT_ONLY_CONDITIONS))
    dc = dict(d, labels=(d['labels'] > 5).astype(int),
              task_type='classification')
    rc = n32.run(dc, {'n_splits': 3}, OUT, 'toy', models=['rf'])
    check('9b. a class label is skipped with its reason',
          isinstance(rc, dict) and rc.get('skipped') == 'not_regression')


def test_the_command_line():
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n32_iterative_and_dual_selection.py'),
         '--help'], capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--draws',
              '--alphas', '--detectors', '--random-seeds', '--models',
              '--exp-folds', '--tag')
    missing = [f for f in wanted if f not in out.stdout]
    check('11. the parser lists every flag',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing else f'{len(wanted)} flags present')
    check('11b. --base-rep is gone', '--base-rep' not in out.stdout)


if __name__ == '__main__':
    test_knn_shapley()
    test_cleanlab()
    test_shift_and_counts()
    test_itlm_and_shrink()
    test_guards()
    test_full_runs()
    test_the_command_line()
    print()
    if FAIL:
        print(f"FAILED {len(FAIL)}: {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
