#!/usr/bin/env python3
"""Did the fix work? Same measurement as the autopsy, on the REAL fixed code.

Calls the actual `augment_uncapped` from src/kirby/hybrid.py end to end and asks
the two questions that condemned the old mechanism:

  1. Does the gain it CLAIMS on its internal holdout survive on the untouched
     outer fold? Old code: claimed +0.0214, delivered -0.0041 (sign flip on one
     fold). Anything that still claims a lot and delivers nothing is not fixed.

  2. Does it admit PURE GAUSSIAN NOISE? Old code admitted 1.7 noise columns per
     fold and reported a positive inner gain while doing it. A fixed mechanism
     must keep zero.

Also reports the real-pool admission count and the score on the outer fold, so
the fix can be compared against the pre-fix numbers directly.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache python3 -u \
      scratchpad/aug_fix_verify.py --dataset herg_ki --base-rep mhg_gnn --n-folds 3
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
from sklearn.model_selection import GroupKFold

from experiments._shared import (
    load_data, get_model_factories, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

from kirby.hybrid import augment_uncapped, _evaluate_allocation


def _seeded(factory, seed):
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def _inner_split(groups, labels, n_samples, validation_split=0.2):
    """Replica of create_hybrid's internal holdout (hybrid.py:3642-3665)."""
    if groups is not None and len(np.unique(groups)) >= 2:
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(int(round(1.0 / validation_split)), n_groups))
        return next(GroupKFold(n_splits=n_splits).split(
            np.zeros(n_samples), labels, groups))
    n_val = max(int(n_samples * validation_split), 1)
    idx = np.random.RandomState(42).permutation(n_samples)
    return idx[n_val:], idx[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--base-rep', required=True)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--top-n', type=int, default=5,
                    help='per-source budget (None-able via 0)')
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
    groups = data.get('groups')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    factory = get_model_factories([args.model], task_type=data['task_type'],
                                  n_samples=len(labels))[args.model]
    model = factory()

    base_X = reps[args.base_rep]
    aug_full = {a: reps[a] for a in aug_names}
    psb = args.top_n if args.top_n > 0 else None
    print(f"{ds}: base={args.base_rep} {base_X.shape}  metric={metric_name} "
          f"lower_is_better={lower_is_better}")
    print(f"  aug pool: {[(a, v.shape[1]) for a, v in aug_full.items()]}  "
          f"per_source_budget={psb}")

    splits = list(cv_split(groups, args.n_folds, reps[pool[0]], labels))
    out = []
    t0 = time.time()

    for fi, (otr, ova) in enumerate(splits):
        print(f"\n  === outer fold {fi} [{time.time()-t0:.0f}s]", flush=True)
        g_tr = groups[otr] if groups is not None else None
        itr, ivl = _inner_split(g_tr, labels[otr], len(otr))
        itr_g, ivl_g = otr[itr], otr[ivl]
        ytr, yvl = labels[itr_g], labels[ivl_g]
        Xb_tr, Xb_vl = base_X[itr_g], base_X[ivl_g]

        def _run(aug_tr, aug_vl, tag, per_source):
            r = augment_uncapped(
                Xb_tr, Xb_vl, aug_tr, aug_vl, ytr, yvl,
                model=model, metric_fn=metric_fn,
                lower_is_better=lower_is_better,
                per_source_budget=per_source,
            )
            gain = r['final_score'] - r['baseline_score']
            print(f"      [{tag}] kept {r['n_added']} cols from "
                  f"{list(r['kept_augmentations'])}, inner gain {gain:+.4f}",
                  flush=True)
            return r, float(gain)

        # --- the real augmentation pool ---
        aug_tr = {a: v[itr_g] for a, v in aug_full.items()}
        aug_vl = {a: v[ivl_g] for a, v in aug_full.items()}
        real, inner_gain = _run(aug_tr, aug_vl, 'real', psb)

        # What did those columns actually deliver on the untouched fold?
        base_o = _evaluate_allocation(
            base_X[otr], base_X[ova], labels[otr], labels[ova],
            model=model, metric_fn=metric_fn, lower_is_better=lower_is_better)
        info = real['augmentation_info']
        if info:
            extra = np.hstack([aug_full[a][:, np.asarray(d['selected_indices'],
                                                         dtype=int)]
                               for a, d in info.items()])
            aug_o = _evaluate_allocation(
                np.hstack([base_X[otr], extra[otr]]),
                np.hstack([base_X[ova], extra[ova]]),
                labels[otr], labels[ova], model=model, metric_fn=metric_fn,
                lower_is_better=lower_is_better)
        else:
            aug_o = base_o
        outer_gain = float(aug_o - base_o)
        print(f"    OUTER: base {base_o:+.4f} -> aug {aug_o:+.4f} "
              f"({outer_gain:+.4f})   [inner claimed {inner_gain:+.4f}]",
              flush=True)

        # --- pure gaussian noise: a fixed mechanism must keep ZERO ---
        rs = np.random.RandomState(7 + fi)
        n_noise = sum(v.shape[1] for v in aug_full.values())
        noise = rs.standard_normal((len(labels), n_noise))
        nz, noise_gain = _run({'__gaussian__': noise[itr_g]},
                              {'__gaussian__': noise[ivl_g]},
                              'pure-noise', psb)

        out.append({
            'fold': fi,
            'real_n_added': int(real['n_added']),
            'real_inner_gain': inner_gain,
            'outer_base': float(base_o), 'outer_aug': float(aug_o),
            'outer_gain': outer_gain,
            'noise_n_added': int(nz['n_added']),
            'noise_inner_gain': noise_gain,
        })

    print("\n" + "=" * 72)
    ig = [o['real_inner_gain'] for o in out]
    og = [o['outer_gain'] for o in out]
    na = [o['real_n_added'] for o in out]
    nn = [o['noise_n_added'] for o in out]
    print(f"  FIXED CODE — {ds} / {args.base_rep} / {args.model}, "
          f"{len(out)} folds")
    print(f"  inner gain claimed : {np.mean(ig):+.4f}   {['%+.4f' % v for v in ig]}")
    print(f"  outer gain delivered: {np.mean(og):+.4f}   {['%+.4f' % v for v in og]}")
    print(f"  columns admitted, real pool : {na}  mean {np.mean(na):.1f}")
    print(f"  columns admitted, PURE NOISE: {nn}  mean {np.mean(nn):.1f}"
          f"   (must be 0)")
    print(f"\n  BEFORE THE FIX (same dataset/base/folds): claimed +0.0214, "
          f"delivered -0.0041, real 3.7 cols, noise 1.7 cols")
    print("=" * 72)

    path = args.out or os.path.join(
        os.environ.get('KIRBY_SCRATCH', '/tmp'), f'aug_fixverify_{ds}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'dataset': ds, 'base_rep': args.base_rep,
                   'model': args.model, 'metric': metric_name,
                   'lower_is_better': lower_is_better, 'folds': out}, f,
                  indent=2)
    print(f"  -> {path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
