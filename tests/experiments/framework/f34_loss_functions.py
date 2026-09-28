#!/usr/bin/env python3
"""f34 — does the loss a downstream model is trained with change the result?

A local framework experiment: it decides which loss KIRBy's downstream models
should be fitted with. Every model in the suite is fitted with squared error today
(rf, lightgbm, xgboost at their defaults, dnn with loss='mse'); svm is sklearn's
SVR, whose loss is epsilon-insensitive and has no swap here.

NO SEARCH IS RUN. The hybrid for each downstream model is the best row already in
<cache>/result_db/<dataset>.db (tier 'general', objective 'predictive', no
augmentation), read straight from the database and rebuilt inside each fold. The
pool lookup is NOT used: it asks for the full 816-configuration grid, of which a
local database holds a fraction, and it searches the remainder live.

The representations are the ones that stored search was run on, not the dataset's
current DATASET_REPS list; the rows only apply to that list. The top 2 base
representations per model are ranked on the cached 5-fold baseline scores
(<cache>/baselines), and only files fitted on the whole dataset are read.

THE ARMS, per downstream model and per feature space (base_1, base_2, hybrid):
  current         the model exactly as the suite builds it today (squared error, except svm)
  every loss swap n33 defines for that model (n33.build_swaps / make_swap):
    lightgbm  huber at c*s for c in HUBER_C, fair, quantile median, Tukey
              biweight, expectile at 0.4 and 0.6
    xgboost   pseudo-Huber at c*s, absolute error
    rf        absolute-error splits
    dnn       huber at c*s
  s = 1.4826 * median |y - p|, p a group-aware out-of-fold LightGBM on the fold's
  training rows in that space. The same rule n33 uses, so the two files compare.

The hybrid's FEATURES are built once per fold with the squared-error model, and
every arm is fitted on those same features. Only the loss of the final fit
changes. Changing the loss inside the search is a separate experiment.

Scored on the held-out fold in the dataset's own metric. Raw fold scores and
means only.

Usage:
    python tests/experiments/framework/f34_loss_functions.py
    python tests/experiments/framework/f34_loss_functions.py --datasets logd --models rf,lightgbm
"""
import os
# One OpenMP thread. LightGBM and torch in one process load two OpenMP runtimes
# and a DNN fit hangs or segfaults otherwise (53dbfa3, and again 2026-09-28).
for _k, _v in (('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
               ('KMP_DUPLICATE_LIB_OK', 'TRUE'), ('KIRBY_LGBM_THREADS', '1')):
    os.environ.setdefault(_k, _v)

import argparse
import ast
import json
import pickle
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from kirby.paths import cache_root
from kirby.find_best_hybrid import (
    BUILTIN_METRICS, get_model, _hash_bytes, _canonical_bytes)
from kirby.utils.noisy_labels.core import oof_predict
from experiments._shared import load_data, cv_split, save_experiment_results
from experiments.noise.noise_helpers import hybrid_for_fold
from experiments.noise.n33_robust_training import (
    HUBER_C, build_swaps, make_swap, robust_scale)

EXP_ID = 'f34_loss_functions'
DATASETS = ('logd', 'herg_ki', 'caco2', 'bindingdb_kd')
MODELS = ('rf', 'lightgbm', 'xgboost', 'svm', 'dnn')
N_FOLDS = 3
N_BASE = 2
OOF_SPLITS = 3
SEED = 42


# ── What the cache holds ────────────────────────────────────────────────────

def stored_pool(dataset):
    """The stored predictive search with the most rows: its representations,
    metric and rows. None if the database has no such search."""
    path = cache_root() / 'result_db' / f'{dataset}.db'
    if not path.exists():
        return None
    con = sqlite3.connect(str(path))
    rows = con.execute(
        "SELECT rep_pool, metric, model, score_mean, labels_hash, blueprint, "
        "budget, allocation_method, importance_method, internal_evaluator "
        "FROM results WHERE tier='general' AND objective='predictive' "
        "AND aug_strategy='none' AND blueprint IS NOT NULL").fetchall()
    con.close()
    if not rows:
        return None
    by_pool = defaultdict(list)
    for r in rows:
        by_pool[r[0]].append(r)
    pool, prow = max(by_pool.items(), key=lambda kv: len(kv[1]))
    reps = [name for name, _w in ast.literal_eval(pool)]
    metric = prow[0][1]
    return {'reps': reps, 'metric': metric, 'rows': prow,
            'n_rows': len(prow), 'other_pools': len(by_pool) - 1}


def best_recipe_per_model(pool, labels_hash, lower):
    """{model: (blueprint, row summary)} -- the best stored row per model whose
    labels match the loaded data."""
    best = {}
    for (_pool, _metric, model, score, lh, bp_blob, budget, alloc, imp,
         ev) in pool['rows']:
        if str(lh) != str(labels_hash) or score is None:
            continue
        if model not in best or (score < best[model][1]['search_score'] if lower
                                 else score > best[model][1]['search_score']):
            best[model] = (pickle.loads(bp_blob),
                           {'search_score': float(score), 'budget': budget,
                            'allocation_method': alloc,
                            'importance_method': imp,
                            'internal_evaluator': ev})
    return best


def top_base_reps(dataset, reps, model, metric, lower, n_samples, labels_hash):
    """Top N_BASE representations for one model from the cached 5-fold
    baselines. Only files fitted on the whole dataset (same molecule count and
    labels) are read; the rest are listed as missing."""
    scored, missing = [], []
    for rep in reps:
        f = cache_root() / 'baselines' / f'{dataset}__{rep}__{model}__{metric}__5f_{SEED}.pkl'
        try:
            d = pickle.load(open(f, 'rb'))
            meta, payload = d['_cache_meta'], d['payload']
        except Exception:
            missing.append(rep)
            continue
        if (meta.get('n_samples') != n_samples
                or str(meta.get('labels_hash')) != str(labels_hash)):
            missing.append(rep)
            continue
        scored.append((float(payload['score']), rep))
    scored.sort(reverse=not lower)
    return [r for _s, r in scored[:N_BASE]], dict(scored_reps={r: s for s, r in scored},
                                                  missing=missing)


# ── One dataset ─────────────────────────────────────────────────────────────

def run_dataset(dataset, models, output):
    t0 = time.time()
    pool = stored_pool(dataset)
    if pool is None:
        return {'dataset': dataset, 'skipped': 'no stored predictive search'}
    data = load_data(dataset, rep_preset=','.join(pool['reps']))
    labels = np.asarray(data['labels'], dtype=float)
    groups = data['groups']
    reps = data['reps']
    if data.get('task_type') != 'regression':
        return {'dataset': dataset, 'skipped': 'not regression'}
    if sorted(reps) != sorted(pool['reps']):
        return {'dataset': dataset,
                'skipped': f"loaded reps {sorted(reps)} are not the stored "
                           f"search's {sorted(pool['reps'])}"}
    metric = pool['metric']
    metric_fn, lower = BUILTIN_METRICS[metric]
    labels_hash = _hash_bytes(_canonical_bytes(labels))

    recipes = best_recipe_per_model(pool, labels_hash, lower)
    if not recipes:
        return {'dataset': dataset,
                'skipped': 'no stored row matches the loaded labels'}

    swaps = build_swaps(HUBER_C)
    res = {'dataset': dataset, 'metric': metric, 'lower_is_better': lower,
           'n_molecules': int(len(labels)), 'n_folds': N_FOLDS,
           'reps': pool['reps'], 'stored_rows': pool['n_rows'],
           'models': {}, 'errors': {}}
    splits = list(cv_split(groups, N_FOLDS, next(iter(reps.values())), labels,
                           mode='full', task_type='regression'))

    # Which spaces each model is scored in.
    spaces = {}
    for m in models:
        base, base_info = top_base_reps(dataset, pool['reps'], m, metric, lower,
                                        len(labels), labels_hash)
        sp = {f'base_{i}': rep for i, rep in enumerate(base, start=1)}
        if m in recipes:
            sp['hybrid'] = 'hybrid'
        arms = ['current'] + [s for s, parent in swaps.items() if parent == m]
        spaces[m] = sp
        res['models'][m] = {
            'spaces': sp, 'base_ranking': base_info,
            'hybrid_recipe': recipes[m][1] if m in recipes else None,
            'arms': arms,
            'scores': {s: {a: [] for a in arms} for s in sp},
        }
        if not sp:
            res['errors'][m] = 'no base baseline and no stored hybrid'

    for fi, (tr, va) in enumerate(splits):
        y_tr, y_va = labels[tr], labels[va]
        g_tr = np.asarray(groups)[tr] if groups is not None else None
        tr_reps = {k: np.asarray(v[tr], dtype=np.float64) for k, v in reps.items()}
        va_reps = {k: np.asarray(v[va], dtype=np.float64) for k, v in reps.items()}
        for m in models:
            for space, what in spaces[m].items():
                key = f'{m}|{space}|f{fi}'
                if what == 'hybrid':
                    X_tr, X_va = hybrid_for_fold(
                        tr_reps, va_reps, y_tr, recipes[m][0], g_tr,
                        get_model(m, 'regression'), metric_fn, lower)
                    if X_tr is None:
                        res['errors'][key] = 'hybrid could not be built'
                        continue
                else:
                    X_tr, X_va = tr_reps[what], va_reps[what]
                sc = StandardScaler().fit(X_tr)
                X_tr_s = np.nan_to_num(sc.transform(X_tr))
                X_va_s = np.nan_to_num(sc.transform(X_va))
                try:
                    p = oof_predict(X_tr_s, y_tr, g_tr, n_splits=OOF_SPLITS)
                    s = robust_scale(y_tr, p)
                except Exception as exc:
                    res['errors'][f'{key}|scale'] = f'{type(exc).__name__}: {exc}'
                    s = None
                for arm in res['models'][m]['arms']:
                    try:
                        if arm == 'current':
                            est = get_model(m, 'regression')
                        elif s is None:
                            continue
                        else:
                            est = make_swap(arm, s, HUBER_C)
                        t = time.time()
                        est.fit(X_tr_s, y_tr)
                        pred = np.asarray(est.predict(X_va_s), dtype=float)
                        score = float(metric_fn(y_va, pred))
                        res['models'][m]['scores'][space][arm].append(score)
                        print(f"  {dataset} f{fi} {m:8s} {space:7s} {arm:28s} "
                              f"{metric}={score:.4f}  ({time.time() - t:.0f}s)",
                              flush=True)
                    except Exception as exc:
                        res['errors'][f'{key}|{arm}'] = f'{type(exc).__name__}: {exc}'
        save_experiment_results(summarise(res), output, EXP_ID, dataset,
                                metric_name=metric, lower_is_better=lower)
    res['seconds'] = round(time.time() - t0)
    return summarise(res)


def summarise(res):
    """Means per arm, the paired change against 'current' in the same space,
    and hybrid minus the better base rep, per arm. Raw numbers only."""
    lower = res['lower_is_better']
    for m, mr in res['models'].items():
        means, deltas = {}, {}
        for space, arms in mr['scores'].items():
            ref = arms.get('current', [])
            means[space] = {a: (float(np.mean(v)) if v else None) for a, v in arms.items()}
            deltas[space] = {}
            for a, v in arms.items():
                if a == 'current' or not v or len(v) != len(ref):
                    continue
                d = np.asarray(v) - np.asarray(ref)
                deltas[space][a] = {'mean': float(d.mean()),
                                    'per_fold': [float(x) for x in d],
                                    'folds_better': int(((d < 0) if lower else (d > 0)).sum())}
        mr['mean_by_space_and_arm'] = means
        mr['change_vs_current'] = deltas
        hyb = means.get('hybrid', {})
        bases = [means[s] for s in means if s.startswith('base_')]
        mr['hybrid_minus_best_base'] = {}
        for a, hv in hyb.items():
            bv = [b[a] for b in bases if b.get(a) is not None]
            if hv is None or not bv:
                continue
            best = min(bv) if lower else max(bv)
            mr['hybrid_minus_best_base'][a] = {'hybrid': hv, 'best_base': best,
                                               'difference': hv - best}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--output', default='tests/results')
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        print(f"\n=== {EXP_ID}: {ds} ===", flush=True)
        out = run_dataset(ds, models, args.output)
        if 'skipped' in out:
            print(f"  SKIPPED: {out['skipped']}", flush=True)
            save_experiment_results(out, args.output, EXP_ID, ds)
        else:
            print(f"  done in {out['seconds']}s; errors: {len(out['errors'])}",
                  flush=True)


if __name__ == '__main__':
    main()
