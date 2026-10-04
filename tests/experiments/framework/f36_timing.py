#!/usr/bin/env python3
"""How long each piece of one f36 unit takes on the server, so the f36 grid can
be sized to 5 sruns x 4 hours x 3 rounds (60 session-hours).

One unit: fold 0 of the 5-fold split, one model input, one noise setting
(outlier_p10 at 0.5 on regression, uniform at 0.1 on herg). The model is the
SVM; the model input is the SVM's best single representation (--input base)
or its best prediction-built hybrid (--input hybrid), both read from the
stored precompute. No search runs: KIRBY_NO_COLD_SEARCH=1.0,
KIRBY_COLD_SEARCH_MAX_CONFIGS=0.

Each piece is timed once and stopped at --limit minutes (default 30); a piece
that hits the limit is printed as 'over N min'. A line is printed every 60 s
so the terminal never goes silent. Everything computed goes into the shared
noise cache, so the real f36 run does not compute it again.

    python tests/experiments/framework/f36_timing.py logd --input base
"""
import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

RESULTS = []
NOW = {'what': 'starting'}


class Over(Exception):
    pass


def _alarm(_sig, _frm):
    raise Over()


def tick(name, fn, limit_s):
    NOW['what'] = name
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(limit_s))
    t0 = time.time()
    out, note = None, ''
    try:
        out = fn()
    except Over:
        note = f'over {limit_s / 60:.0f} min, stopped'
    except Exception as exc:
        note = f'failed: {type(exc).__name__}: {str(exc)[:120]}'
    finally:
        signal.alarm(0)
    dt = time.time() - t0
    RESULTS.append((name, dt, note))
    print(f'TIMED  {name:<44} {dt:8.1f} s  {note}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--input', choices=('base', 'hybrid'), default='base')
    ap.add_argument('--limit', type=float, default=30.0,
                    help='minutes before one piece is stopped')
    a = ap.parse_args()
    os.environ['KIRBY_NO_COLD_SEARCH'] = '1.0'
    os.environ['KIRBY_COLD_SEARCH_MAX_CONFIGS'] = '0'
    limit = a.limit * 60

    def beat():
        while True:
            time.sleep(60)
            print(f"  [{time.strftime('%H:%M:%S')}] still running: "
                  f"{NOW['what']}", flush=True)
    threading.Thread(target=beat, daemon=True).start()

    import functools
    from experiments._shared import (
        cv_split, get_domain_metric, get_model_factories,
        get_best_search_for_downstream, noise_base_reps_by_model)
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        load_noise_data, noise_spec_for, inject_train_labels,
        pool_recipe_per_downstream_model)
    from experiments.noise import noise_cache as nc
    from experiments.noise import n07_remediation as n07
    from experiments.noise import n23_uncertainty_cleaning as n23
    from experiments.noise import similarity_space as ss
    from experiments.framework import f36_label_smoothing_portfolio as f36

    ds = a.dataset
    data, _ = tick('load data', lambda: load_noise_data(ds), limit)
    is_clf = data['task_type'] == 'classification'
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    reps = data['reps']
    metric = get_domain_metric(data)
    m = 'svm'
    fac = get_model_factories([m], data['task_type'], n_samples=len(labels))

    lookup = functools.partial(
        get_best_search_for_downstream, reps, labels, groups,
        budget=None, augmentations=data.get('augmentations'), dataset_name=ds,
        task_type=data['task_type'], data=data,
        pool_subset=f36.POOL_SUBSET)
    by_model, no_recipe = tick(
        'read stored searches',
        lambda: pool_recipe_per_downstream_model(lookup, [m]), limit)
    recipes, rec_un = f36._recipes_for(by_model, [m])
    base, base_missing = noise_base_reps_by_model(
        {mm: by_model[mm]['search'] for mm in by_model}, [m], reps=reps,
        dataset_name=ds, verbose=False)
    print(f'svm best single representations {base.get(m)}; hybrid recipe '
          f'{"found" if "hybrid" in recipes.get(m, {}) else "MISSING"} '
          f'{rec_un or ""} {no_recipe or ""}', flush=True)

    tr, va = next(iter(cv_split(groups, 5, next(iter(reps.values())),
                                labels)))
    g_tr = None if groups is None else np.asarray(groups)[tr]
    cond, level = ('uniform', 0.1) if is_clf else ('outlier_p10', 0.5)
    spec = noise_spec_for(ds, cond, level, noise_seed=0, is_clf=is_clf)
    y0 = np.asarray(inject_train_labels(
        labels, spec, train_idx=tr, is_clf=is_clf,
        groups=noise_groups_for(data)[0], dataset_name=ds)[0])
    mi = 'base_1' if a.input == 'base' else 'hybrid'
    sp = tick(f'build model input {mi}', lambda: n07.fold_similarity_spaces(
        ds, [ss.SimilaritySetting(mi, 'none', 'manhattan')],
        model_input_tr=None, model_input_va=None, reps=reps, train_idx=tr,
        val_idx=va, y_train=y0, groups_train=g_tr,
        base_reps=base.get(m, []), recipes=recipes.get(m, {}),
        augmentations=data.get('augmentations'), model_name=m,
        model_factory=fac.get(m), metric=metric,
        task_type=data['task_type']), limit)
    if sp is None or sp[0].get(mi) is None:
        print(f'model input {mi} was not built: {sp and sp[2]}')
        return
    X_tr, X_va = sp[0][mi], sp[1][mi]
    print(f'{ds} {mi}: {X_tr.shape[0]} training molecules x {X_tr.shape[1]} '
          f'features, {X_va.shape[0]} held out', flush=True)

    mode = nc.predict_mode_for(metric[0])
    tick('one SVM fit', lambda: nc.cached_fit_predict(
        ds, m, fac[m], X_tr, y0, X_va, predict_mode=mode), limit)
    yy = y0.copy()
    if not is_clf:
        yy = yy + 1e-6  # a different label column: not a cache hit
        tick('a second SVM fit, kernel already computed',
             lambda: nc.cached_fit_predict(ds, m, fac[m], X_tr, yy, X_va,
                                           predict_mode=mode), limit)

    for det in ('cross_val', 'neighbor_residual', 'activity_cliff_aware'):
        tick(f'detector {det}', lambda det=det: nc.cached_detector_scores(
            ds, X_tr, y0, is_clf=is_clf, detector=det, groups=g_tr), limit)

    errors = f36._Errors()
    vals = f36.Values(ds, X_tr, y0, g_tr, errors)
    tick('neighbours (all k, Euclidean)', vals.nbrs, limit)
    tick('forest neighbours', lambda: vals.forest(is_clf=is_clf, y=y0), limit)
    if not is_clf:
        for v in ('lgbm_oof', 'lgbm_huber_oof', 'gp_oof', 'qrf_q50',
                  'pair_mean'):
            tick(f'value {v}', lambda v=v: vals.at(v, y0), limit)

        def clip_forest():
            from kirby.utils.noise_estimation import QuantileForestEstimator
            est = QuantileForestEstimator(n_folds=5, groups=g_tr)
            est.estimate(X_tr, y0)
        tick('quantile_clip forest (10th-90th)', clip_forest, limit)
        for src in ('hetero_gp', 'ngboost', 'heteroscedastic_nn'):
            tick(f'uncertainty model {src}', lambda src=src: n23._uncertainty(
                src, X_tr, y0, g_tr, False, 5), limit)
        tick('uncertainty gate (hetero_gp latent variance)',
             vals.uncertainty, limit)
    print('errors:', errors.d, vals.unavailable)

    print(f'\n== {ds} {mi}: {X_tr.shape[0]} x {X_tr.shape[1]}')
    for name, dt, note in RESULTS:
        print(f'{name:<44} {dt:8.1f} s  {note}')


if __name__ == '__main__':
    main()
