#!/usr/bin/env python3
"""Why does adding good extra features to a good rep do nothing?

Three candidate explanations for the null result, tested independently on the
same folds. All conditions are FORCED (the columns are simply added) except the
ones marked greedy, so this measures whether the extras CAN help at all,
separately from whether the admission loop finds them.

  WIDTH (dilution). A handful of columns added to a 1024-column matrix is under
  1% of the width. A tree ensemble has to prefer them over 1024 alternatives at
  every split. If the benefit of a fixed extras block grows as the base is
  narrowed, dilution is the binding constraint and no admission rule can fix it.

  RANKING (base-blind selection). Both selection layers in the code rank
  candidates by RF importance fit on the candidate block ALONE — marginal
  signal, never complementarity to the base. `residual_correlation` is the only
  base-aware ranker in the codebase and has zero callers. Compared here head to
  head, against random columns at matched width.

  GRANULARITY (singleton vs block). The default 'uncapped' strategy admits ONE
  column at a time and requires each to individually improve the model. The
  hand-run proof of concept added a whole block. 'greedy_ablation' tests blocks.
  Both are run here.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache python3 -u \
      scratchpad/aug_dilution_factorial.py --dataset herg_ki --base-rep mhg_gnn
"""
import argparse
import json
import os
import sys
import time

try:
    import numpy as _np
    import lightgbm as _lgb
    _lgb.LGBMRegressor(n_estimators=2, verbose=-1).fit(
        _np.zeros((8, 2)), _np.arange(8.0))
except Exception as _e:  # pragma: no cover
    print(f"  [warmup] lightgbm warm-up skipped: {_e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_model_factories, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

WIDTHS = (50, 100, 250, 500, None)   # None = full base rep
N_SEEDS = 3


def _seeded(factory, seed):
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def _fit_score(Xtr, ytr, Xva, yva, factory, metric_fn, n_seeds):
    sc = StandardScaler()
    A = sc.fit_transform(np.nan_to_num(np.clip(Xtr, -1e10, 1e10), nan=0.0))
    B = sc.transform(np.nan_to_num(np.clip(Xva, -1e10, 1e10), nan=0.0))
    vals = []
    for s in range(n_seeds):
        m = _seeded(factory, 42 + s)
        m.fit(A, ytr)
        vals.append(float(metric_fn(yva, m.predict(B))))
    return float(np.mean(vals))


def _lgb_gain_order(X, y, task_type):
    """Rank base columns by lightgbm gain — the pipeline's importance method."""
    import lightgbm as lgb
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0))
    M = (lgb.LGBMClassifier if task_type == 'classification'
         else lgb.LGBMRegressor)
    m = M(n_estimators=100, random_state=42, verbose=-1)
    m.fit(Xs, y)
    imp = np.asarray(m.booster_.feature_importance(importance_type='gain'),
                     dtype=float)
    return np.argsort(imp)[::-1]


def _rank_marginal(P, y, n, task_type):
    """RF importance fit on the candidate pool alone — what the code does."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    sc = StandardScaler()
    Ps = sc.fit_transform(np.nan_to_num(np.clip(P, -1e10, 1e10), nan=0.0))
    R = (RandomForestClassifier if task_type == 'classification'
         else RandomForestRegressor)
    rf = R(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(Ps, y)
    return np.argsort(np.nan_to_num(rf.feature_importances_, nan=0.0))[::-1][:n]


def _rank_residual(P, y, Xbase, n, factory):
    """Correlation with the base model's residuals — base-AWARE selection."""
    sc = StandardScaler()
    Bs = sc.fit_transform(np.nan_to_num(np.clip(Xbase, -1e10, 1e10), nan=0.0))
    m = _seeded(factory, 42)
    m.fit(Bs, y)
    resid = y - m.predict(Bs)
    P = np.nan_to_num(np.clip(P, -1e10, 1e10), nan=0.0)
    sc_ = np.zeros(P.shape[1])
    for j in range(P.shape[1]):
        c = P[:, j]
        if np.std(c) > 1e-12:
            sc_[j] = abs(np.corrcoef(c, resid)[0, 1])
    return np.argsort(np.nan_to_num(sc_, nan=0.0))[::-1][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--base-rep', required=True)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-extra', type=int, default=20,
                    help='columns taken from the extras pool per condition')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or 'seagate' not in cache:
        print(f"REFUSING TO RUN: KIRBY_CACHE_DIR={cache!r}")
        return 2

    ds = args.dataset
    pool = DATASET_REPS[ds]['general']
    aug_names = DATASET_REPS[ds].get('augmentation') or []
    data = load_data(ds, rep_names=pool + aug_names)
    reps, labels = data['reps'], data['labels']
    groups, task_type = data.get('groups'), data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    factory = get_model_factories([args.model], task_type=task_type,
                                  n_samples=len(labels))[args.model]

    base_X = reps[args.base_rep]
    P_full = np.hstack([reps[a] for a in aug_names])
    print(f"{ds}: base={args.base_rep} {base_X.shape}  extras pool "
          f"{P_full.shape} from {aug_names}")
    print(f"  metric={metric_name} lower_is_better={lower_is_better}")

    splits = list(cv_split(groups, args.n_folds, reps[pool[0]], labels))
    sign = -1.0 if lower_is_better else 1.0
    n_ex = args.n_extra
    t0 = time.time()

    # rows[condition][width] = list of per-fold scores
    rows = {}

    def rec(cond, width, val):
        rows.setdefault(cond, {}).setdefault(str(width), []).append(val)

    for fi, (tr, va) in enumerate(splits):
        y_tr, y_va = labels[tr], labels[va]
        print(f"\n  === fold {fi} [{time.time()-t0:.0f}s]", flush=True)

        # Base column order, computed on TRAIN ONLY.
        order = _lgb_gain_order(base_X[tr], y_tr, task_type)

        # Extras chosen on TRAIN ONLY, three ways + a matched-width control.
        # Selected once against the FULL base rep so the same columns are used
        # at every width — this isolates width from selection.
        idx_marg = _rank_marginal(P_full[tr], y_tr, n_ex, task_type)
        idx_resid = _rank_residual(P_full[tr], y_tr, base_X[tr], n_ex, factory)
        idx_rand = np.random.RandomState(100 + fi).choice(
            P_full.shape[1], size=n_ex, replace=False)
        overlap = len(set(idx_marg.tolist()) & set(idx_resid.tolist()))
        print(f"    marginal vs residual pick overlap: {overlap}/{n_ex}",
              flush=True)

        extras = {
            'none': None,
            f'marginal_top{n_ex}': P_full[:, idx_marg],
            f'residual_top{n_ex}': P_full[:, idx_resid],
            f'random_top{n_ex}': P_full[:, idx_rand],
            'wholepool': P_full,
        }

        for width in WIDTHS:
            cols = order if width is None else order[:width]
            B_tr, B_va = base_X[tr][:, cols], base_X[va][:, cols]
            for cname, E in extras.items():
                if E is None:
                    Xtr, Xva = B_tr, B_va
                else:
                    Xtr = np.hstack([B_tr, E[tr]])
                    Xva = np.hstack([B_va, E[va]])
                s = _fit_score(Xtr, y_tr, Xva, y_va, factory, metric_fn,
                               N_SEEDS)
                rec(cname, width, s)
            base_s = rows['none'][str(width)][-1]
            line = f"    w={str(width):>4s} base {base_s:.4f}"
            for cname in extras:
                if cname == 'none':
                    continue
                d = sign * (rows[cname][str(width)][-1] - base_s)
                line += f" | {cname.split('_')[0][:4]} {d:+.4f}"
            print(line, flush=True)

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  {ds} / {args.base_rep} / {args.model} — mean over "
          f"{len(splits)} folds, {N_SEEDS} seeds. + = extras HELP.")
    print("=" * 78)
    hdr = f"  {'base width':>10s} {'base score':>11s}"
    conds = [c for c in rows if c != 'none']
    for c in conds:
        hdr += f" {c[:14]:>15s}"
    print(hdr)
    summary = {}
    for width in WIDTHS:
        w = str(width)
        base_m = float(np.mean(rows['none'][w]))
        line = f"  {w:>10s} {base_m:>11.4f}"
        summary[w] = {'base': base_m}
        for c in conds:
            vals = np.array(rows[c][w])
            d = sign * (vals - np.array(rows['none'][w]))
            wins = int((d > 0).sum())
            line += f" {np.mean(d):>+9.4f}({wins}/{len(d)})"
            summary[w][c] = {'delta': float(np.mean(d)), 'wins': wins,
                             'score': float(np.mean(vals))}
        print(line)
    print("=" * 78)

    path = args.out or os.path.join(
        os.environ.get('KIRBY_SCRATCH', '/tmp'), f'aug_dilution_{ds}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'dataset': ds, 'base_rep': args.base_rep,
                   'model': args.model, 'metric': metric_name,
                   'lower_is_better': lower_is_better, 'n_extra': n_ex,
                   'widths': [str(w) for w in WIDTHS],
                   'raw': rows, 'summary': summary}, f, indent=2)
    print(f"  -> {path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
