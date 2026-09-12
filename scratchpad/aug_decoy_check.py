#!/usr/bin/env python3
"""Do the extra-feature rules take bait? Two planted decoys, three arms.

Runs the REAL `augment_uncapped` from src/kirby/hybrid.py against a candidate
pool that has two fakes mixed into it, and measures what actually lands on an
outer fold that neither the ranking nor the choosing ever saw.

The two decoys
--------------
  __gaussian__  pure random numbers. Anything that keeps these is broken in the
                obvious way. The old code kept 1.7 columns per fold.

  __echo__      columns COPIED out of the base representation itself with a
                little noise added. This is the decoy that was missing, and it
                is the one that matters: it carries real signal, so a ranker
                that asks "does this predict the property?" will rate it highly,
                while a ranker that asks "does this add anything the base does
                not already have?" must throw it out. Random noise cannot tell
                those two rankers apart. This can.

Three arms, all scored on the same untouched outer fold
------------------------------------------------------
  none   base representation alone.
  all    base + EVERY real candidate column, no selection at all. This is the
         null hypothesis from the literature (Grinsztajn 2022, Cherepanova
         2023): let the downstream model ignore what is useless. Selection has
         to beat this to be worth its cost.
  rule   base + whatever the current rule picks out of (real blocks + decoys).

Readouts: delivered gain of each arm over `none`, columns taken from each
decoy, and how often the rule correctly added nothing.

Usage (representations live on the external drive, NOT ~/.cache):
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache python3 -u \
      scratchpad/aug_decoy_check.py --dataset logd --base-rep mordred --n-folds 5
"""
import argparse
import json
import os
import sys
import time

# This laptop loads two copies of a threading library and the first model fit
# after the data is read kills the interpreter with no traceback. Fitting a
# two-tree model before anything else is loaded avoids it. Server does not need
# this, which is why it lives here and not in the pipeline.
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

GAUSSIAN = '__gaussian__'
ECHO = '__echo__'


def _inner_split(groups, labels, n_samples, validation_split=0.2):
    """Replica of the holdout create_hybrid carves inside a training fold.

    Group-aware when groups exist, so related molecules cannot straddle it.
    """
    if groups is not None and len(np.unique(groups)) >= 2:
        n_groups = len(np.unique(groups))
        n_splits = max(2, min(int(round(1.0 / validation_split)), n_groups))
        return next(GroupKFold(n_splits=n_splits).split(
            np.zeros(n_samples), labels, groups))
    n_val = max(int(n_samples * validation_split), 1)
    idx = np.random.RandomState(42).permutation(n_samples)
    return idx[n_val:], idx[:n_val]


def _make_echo(base_X, n_cols, jitter, rng):
    """Copy columns out of the base representation and rough them up slightly.

    Columns with no spread are skipped — a constant column is not a duplicate of
    anything, it is just dead, and letting one in would make the decoy easier
    than it should be.
    """
    sd = base_X.std(axis=0)
    live = np.flatnonzero(sd > 0)
    if live.size == 0:
        return None
    take = rng.choice(live, size=min(n_cols, live.size), replace=False)
    echo = base_X[:, take].astype(float, copy=True)
    echo += rng.standard_normal(echo.shape) * (sd[take] * jitter)
    return echo


def _selected_columns(info, pool_full):
    """Rebuild the chosen columns, full length, so they can be sliced by fold."""
    blocks = [pool_full[name][:, np.asarray(d['selected_indices'], dtype=int)]
              for name, d in info.items()
              if len(d.get('selected_indices', ())) > 0]
    return np.hstack(blocks) if blocks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='logd')
    ap.add_argument('--base-rep', required=True)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--top-n', type=int, default=5,
                    help='columns taken from EACH source before pooling; 0=all')
    ap.add_argument('--ranking', default='residual_correlation',
                    choices=['residual_correlation', 'rf_importance'],
                    help='residual_correlation asks "does this add anything the '
                         'base lacks"; rf_importance asks "does this predict the '
                         'property on its own". Both are live in the engine.')
    ap.add_argument('--echo-cols', type=int, default=64)
    ap.add_argument('--echo-jitter', type=float, default=0.05,
                    help='noise added to the copied columns, as a fraction of '
                         'each column\'s own spread')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or not os.path.isdir(cache):
        print(f"REFUSING TO RUN: KIRBY_CACHE_DIR={cache!r} is not a directory.")
        print("The representations are on the external drive. Without this the "
              "run regenerates them from scratch, which takes hours.")
        return 2

    ds = args.dataset
    pool_names = DATASET_REPS[ds]['general']
    aug_names = DATASET_REPS[ds].get('augmentation') or []
    if not aug_names:
        print(f"REFUSING TO RUN: {ds} has no extra feature blocks to test.")
        return 2

    data = load_data(ds, rep_names=pool_names + aug_names)
    reps, labels = data['reps'], data['labels']
    groups = data.get('groups')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    factory = get_model_factories([args.model], task_type=data['task_type'],
                                  n_samples=len(labels))[args.model]
    model = factory()

    if args.base_rep not in reps:
        print(f"REFUSING TO RUN: base rep {args.base_rep!r} not in {ds}. "
              f"Available: {sorted(reps)}")
        return 2

    base_X = reps[args.base_rep]
    real = {a: reps[a] for a in aug_names}
    psb = args.top_n if args.top_n > 0 else None

    print(f"{ds}: base={args.base_rep} {base_X.shape}  metric={metric_name} "
          f"(higher-is-better after sign correction)")
    print(f"  real extra blocks: {[(a, v.shape[1]) for a, v in real.items()]}")
    print(f"  decoys: {GAUSSIAN} and {ECHO} "
          f"({args.echo_cols} cols copied from the base, "
          f"{args.echo_jitter:.0%} jitter)")
    print(f"  columns offered per source: {psb}")

    splits = list(cv_split(groups, args.n_folds, reps[pool_names[0]], labels))
    rows = []
    t0 = time.time()

    for fi, (otr, ova) in enumerate(splits):
        print(f"\n  === outer fold {fi} [{time.time()-t0:.0f}s]", flush=True)
        rng = np.random.RandomState(1000 + fi)

        # Decoys are built full length so the same columns can be scored on the
        # outer fold later. Both are rebuilt per fold with a fresh seed.
        n_noise = sum(v.shape[1] for v in real.values())
        pool_full = dict(real)
        pool_full[GAUSSIAN] = rng.standard_normal((len(labels), n_noise))
        echo = _make_echo(base_X, args.echo_cols, args.echo_jitter, rng)
        if echo is not None:
            pool_full[ECHO] = echo

        g_tr = groups[otr] if groups is not None else None
        itr, ivl = _inner_split(g_tr, labels[otr], len(otr))
        itr_g, ivl_g = otr[itr], otr[ivl]

        result = augment_uncapped(
            base_X[itr_g], base_X[ivl_g],
            {k: v[itr_g] for k, v in pool_full.items()},
            {k: v[ivl_g] for k, v in pool_full.items()},
            labels[itr_g], labels[ivl_g],
            model=model, metric_fn=metric_fn,
            lower_is_better=lower_is_better,
            per_source_budget=psb,
        )
        info = result['augmentation_info']
        n_from = {name: len(d.get('selected_indices', ()))
                  for name, d in info.items()}
        claimed = float(result['final_score'] - result['baseline_score'])

        # --- the three arms, all on the untouched outer fold ---
        def _outer(extra_full):
            if extra_full is None:
                Xtr, Xva = base_X[otr], base_X[ova]
            else:
                Xtr = np.hstack([base_X[otr], extra_full[otr]])
                Xva = np.hstack([base_X[ova], extra_full[ova]])
            return float(_evaluate_allocation(
                Xtr, Xva, labels[otr], labels[ova], model=model,
                metric_fn=metric_fn, lower_is_better=lower_is_better))

        s_none = _outer(None)
        s_all = _outer(np.hstack([real[a] for a in aug_names]))
        s_rule = _outer(_selected_columns(info, pool_full))

        n_gauss = n_from.get(GAUSSIAN, 0)
        n_echo = n_from.get(ECHO, 0)
        print(f"    kept {result['n_added']} cols  {n_from}")
        print(f"      decoys taken: gaussian {n_gauss}, echo {n_echo}")
        print(f"    OUTER  none {s_none:+.4f} | all {s_all:+.4f} "
              f"({s_all-s_none:+.4f}) | rule {s_rule:+.4f} "
              f"({s_rule-s_none:+.4f})   [rule claimed {claimed:+.4f}]",
              flush=True)

        rows.append({
            'fold': fi, 'n_added': int(result['n_added']),
            'n_from': {k: int(v) for k, v in n_from.items()},
            'n_gaussian': int(n_gauss), 'n_echo': int(n_echo),
            'claimed': claimed,
            'outer_none': s_none, 'outer_all': s_all, 'outer_rule': s_rule,
            'delivered_all': s_all - s_none,
            'delivered_rule': s_rule - s_none,
            'added_nothing': bool(result['n_added'] == 0),
        })

    print("\n" + "=" * 74)
    d_all = [r['delivered_all'] for r in rows]
    d_rule = [r['delivered_rule'] for r in rows]
    clm = [r['claimed'] for r in rows]
    print(f"  {ds} / {args.base_rep} / {args.model} — {len(rows)} folds")
    print(f"  rule CLAIMS         : {np.mean(clm):+.4f}   "
          f"{['%+.4f' % v for v in clm]}")
    print(f"  rule DELIVERS       : {np.mean(d_rule):+.4f}   "
          f"{['%+.4f' % v for v in d_rule]}")
    print(f"  no selection at all : {np.mean(d_all):+.4f}   "
          f"{['%+.4f' % v for v in d_all]}")
    print(f"  gaussian cols taken : {[r['n_gaussian'] for r in rows]}  "
          f"(must be 0)")
    print(f"  echo cols taken     : {[r['n_echo'] for r in rows]}  "
          f"(must be 0 — these duplicate the base)")
    print(f"  folds adding nothing: {sum(r['added_nothing'] for r in rows)}"
          f" of {len(rows)}")
    print("=" * 74)

    path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'aug_decoy_{ds}_{args.base_rep}.json')
    with open(path, 'w') as f:
        json.dump({'dataset': ds, 'base_rep': args.base_rep,
                   'model': args.model, 'metric': metric_name,
                   'lower_is_better': lower_is_better,
                   'echo_cols': args.echo_cols,
                   'echo_jitter': args.echo_jitter,
                   'top_n_per_source': psb, 'folds': rows}, f, indent=2)
    print(f"  -> {path}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
