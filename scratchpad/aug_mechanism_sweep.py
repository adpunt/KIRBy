#!/usr/bin/env python3
"""Which specific way of adding extra features works? Full cross-product sweep.

This drives the REAL engine (create_hybrid), not a substitute harness. Every
axis below is a thing the engine can actually be told to do differently.

AXES
----
base            single  = the best representation at its full native width
                hybrid  = the multi-representation hybrid at a budget
source          AUG     = the curated extra blocks for this dataset
                BORROW  = the top-k features of every OTHER pool representation
                BOTH
mechanism       none            = control, no extras
                all             = take top-N per donor, no test at all
                greedy_ablation = keep or drop each donor's block whole
                capped          = fill spare budget, then evict base features
                uncapped        = rank once, then pick a block size from a ladder
ranker          rf_importance        = rank a candidate by how well it predicts
                                       the property on its own (base-blind)
                residual_correlation = rank it by how well it explains what the
                                       base gets WRONG
donor_cap       how many features each donor may offer: 5, 64, or all
max_candidates  the cut-off on the pooled candidate list: 50 or 200
budget          hybrid base only: 100 / 250 / 500 / 1000
admission       honest   = scaffold groups, the real model and the real metric
                           are passed into the admission decision
                pipeline = none of them are passed, which is what every
                           experiment in the tree actually does today

CONTROLS, run per real cell at matched shape
--------------------------------------------
RANDOM   the same number of extra features, drawn at random from the same
         donor pool. If a cell cannot beat this, the gain is model capacity,
         not the choosing.

DEGENERATE COMBINATIONS ARE NOT SILENTLY DROPPED. Where the engine does not
read an axis for a given mechanism, the sweep collapses that axis and PRINTS
why, with the reason. Run with --dry-run to see the full enumeration and the
collapse table before spending anything.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache python3 -u \
      scratchpad/aug_mechanism_sweep.py --dataset herg_ki --dry-run
"""
import argparse
import itertools
import json
import os
import sys
import time

# This laptop loads two copies of a threading library; the first model fit after
# the data is read kills the interpreter with no traceback. Fit something tiny
# before anything else is imported.
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
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_model_factories, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

from kirby.hybrid import create_hybrid, apply_feature_selection, \
    apply_augmentation_selection

MECHANISMS = ['none', 'all', 'greedy_ablation', 'capped', 'uncapped']
RANKERS = ['rf_importance', 'residual_correlation']
DONOR_CAPS = [5, 64, None]
MAX_CANDIDATES = [50, 200]
SOURCES = ['AUG', 'BORROW', 'BOTH']
BASES = ['single', 'hybrid']
HYBRID_BUDGETS = [100, 250, 500, 1000]
ADMISSION = ['honest', 'pipeline']

# Which axes the engine ACTUALLY reads for each mechanism. Anything False here
# is collapsed to a single value and the reason is printed — it is not a
# judgement about what should win, it is a statement about what the code reads.
AXIS_READ = {
    #                ranker  donor_cap  max_candidates
    'none':           (False, False, False),
    'all':            (False, False, False),
    'greedy_ablation': (False, False, False),
    'capped':         (True,  False, True),
    'uncapped':       (True,  True,  True),
}
COLLAPSE_REASON = {
    'none': 'no extras are offered at all, so no selection axis is read',
    'all': 'this path ranks with its own forest and ignores the ranker, the '
           'per-donor cap and the candidate cut-off',
    'greedy_ablation': 'this path ranks each donor block with its own forest '
                       'and ignores the ranker, the per-donor cap and the cut-off',
    'capped': 'the per-donor cap is never forwarded to this path',
}


_RANK_CACHE = {}


def _rank_within(X_tr, y_tr, k, is_clf, cache_key=None):
    """Top-k feature positions of one donor, ranked on TRAINING rows only.

    Memoised on (donor, cap, fold): the ranking depends only on those, but the
    sweep asks for it once per cell. Without this the same forest is refitted
    hundreds of times. This changes no result — it is the free saving the design
    document flagged and nobody took.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    if cache_key is not None and cache_key in _RANK_CACHE:
        return _RANK_CACHE[cache_key]
    if k is None or k >= X_tr.shape[1]:
        idx = np.arange(X_tr.shape[1])
        if cache_key is not None:
            _RANK_CACHE[cache_key] = idx
        return idx
    cls = LGBMClassifier if is_clf else LGBMRegressor
    m = cls(n_estimators=100, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=-1)
    sc = StandardScaler()
    m.fit(sc.fit_transform(np.nan_to_num(X_tr)), y_tr)
    imp = np.asarray(m.booster_.feature_importance(importance_type='gain'),
                     dtype=float)
    idx = np.sort(np.argsort(imp)[::-1][:k])
    if cache_key is not None:
        _RANK_CACHE[cache_key] = idx
    return idx


def _fit_score(X_tr, y_tr, X_va, y_va, factory, metric_fn, seeds):
    """Average the metric over model seeds before any difference is taken."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(np.nan_to_num(X_tr))
    X_va = sc.transform(np.nan_to_num(X_va))
    out = []
    for s in range(seeds):
        m = factory()
        try:
            if 'random_state' in m.get_params():
                m.set_params(random_state=42 + s)
        except Exception:
            pass
        m.fit(X_tr, y_tr)
        out.append(float(metric_fn(y_va, m.predict(X_va))))
    return float(np.mean(out))


def enumerate_grid(base_shapes, sources, mechanisms, admissions):
    """Full cross-product, with unread axes collapsed and the reason recorded."""
    cells, collapses = [], {}
    for base, mech, adm in itertools.product(base_shapes, mechanisms, admissions):
        r_read, c_read, m_read = AXIS_READ[mech]
        rankers = RANKERS if r_read else [RANKERS[1]]
        caps = DONOR_CAPS if c_read else [DONOR_CAPS[0]]
        maxc = MAX_CANDIDATES if m_read else [MAX_CANDIDATES[0]]
        srcs = ['-'] if mech == 'none' else sources
        budgets = HYBRID_BUDGETS if base == 'hybrid' else [None]
        if not (r_read and c_read and m_read) and mech in COLLAPSE_REASON:
            collapses[mech] = COLLAPSE_REASON[mech]
        for src, rk, cap, mc, b in itertools.product(
                srcs, rankers, caps, maxc, budgets):
            cells.append({'base': base, 'source': src, 'mechanism': mech,
                          'ranker': rk, 'donor_cap': cap, 'max_candidates': mc,
                          'budget': b, 'admission': adm})
    # 'none' does not depend on the admission context either
    seen, uniq = set(), []
    for c in cells:
        k = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        if c['mechanism'] == 'none':
            k = ('none', c['base'], c['budget'])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq, collapses


def build_donors(reps, pool, best, aug_names, source, cap, tr, y_tr, is_clf,
                 fold=0):
    """The donor pool for one cell, full length, trimmed on training rows only."""
    donors = {}
    if source in ('AUG', 'BOTH'):
        for a in aug_names:
            idx = _rank_within(reps[a][tr], y_tr, cap, is_clf, (a, cap, fold))
            donors[a] = reps[a][:, idx]
    if source in ('BORROW', 'BOTH'):
        for r in pool:
            if r == best:
                continue
            idx = _rank_within(reps[r][tr], y_tr, cap, is_clf, (r, cap, fold))
            donors[f'borrow__{r}'] = reps[r][:, idx]
    return donors


def run_cell(cell, reps, pool, best, aug_names, labels, groups, tr, va,
             factory, metric_fn, lower, seeds, is_clf, rng, fold=0):
    """One cell on one fold, through the real engine. Returns (score, info)."""
    y_tr, y_va = labels[tr], labels[va]
    mech, src = cell['mechanism'], cell['source']

    if cell['base'] == 'single':
        base_reps = {best: reps[best]}
        budget = reps[best].shape[1]
    else:
        base_reps = {r: reps[r] for r in pool}
        budget = cell['budget']

    tr_base = {k: v[tr] for k, v in base_reps.items()}
    va_base = {k: v[va] for k, v in base_reps.items()}

    if mech == 'none':
        X_tr, info = create_hybrid(
            tr_base, y_tr, allocation_method='top_3', budget=budget,
            augmentations=None, augmentation_strategy='none',
            groups=groups[tr] if groups is not None else None)
        X_va = apply_feature_selection(va_base, info)
        return _fit_score(X_tr, y_tr, X_va, y_va, factory, metric_fn, seeds), \
            {'n_added': 0}

    donors = build_donors(reps, pool, best, aug_names, src,
                          cell['donor_cap'], tr, y_tr, is_clf, fold=fold)
    if not donors:
        return None, {'skipped': 'no donors for this source'}

    honest = cell['admission'] == 'honest'
    kw = dict(groups=groups[tr] if (honest and groups is not None) else None,
              model=factory() if honest else None,
              metric_fn=metric_fn if honest else None,
              lower_is_better=lower if honest else False)

    X_tr, info = create_hybrid(
        tr_base, y_tr, allocation_method='top_3', budget=budget,
        augmentations={k: v[tr] for k, v in donors.items()},
        augmentation_strategy=mech,
        candidate_ranking=cell['ranker'],
        aug_per_source_budget=cell['donor_cap'],
        aug_max_candidates=cell['max_candidates'],
        **kw)
    X_va = apply_feature_selection(va_base, info)
    aug_info = info.get('augmentation_info')
    n_added = 0
    if aug_info:
        extra_va = apply_augmentation_selection(
            {k: v[va] for k, v in donors.items()}, aug_info)
        if extra_va is not None:
            X_va = np.hstack([X_va, extra_va])
            n_added = extra_va.shape[1]
    score = _fit_score(X_tr, y_tr, X_va, y_va, factory, metric_fn, seeds)

    # matched-shape random control: same count, drawn from the same pool
    rand_score = None
    if n_added > 0:
        allc_tr = np.hstack([v[tr] for v in donors.values()])
        allc_va = np.hstack([v[va] for v in donors.values()])
        pick = rng.choice(allc_tr.shape[1],
                          size=min(n_added, allc_tr.shape[1]), replace=False)
        base_only_tr = apply_feature_selection(tr_base, info)
        base_only_va = apply_feature_selection(va_base, info)
        rand_score = _fit_score(
            np.hstack([base_only_tr, allc_tr[:, pick]]), y_tr,
            np.hstack([base_only_va, allc_va[:, pick]]), y_va,
            factory, metric_fn, seeds)

    return score, {'n_added': int(n_added), 'random_score': rand_score,
                   'donors': sorted(aug_info) if aug_info else []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--bases', nargs='+', default=BASES)
    ap.add_argument('--mechanisms', nargs='+', default=MECHANISMS)
    ap.add_argument('--sources', nargs='+', default=SOURCES)
    ap.add_argument('--admissions', nargs='+', default=ADMISSION)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cells, collapses = enumerate_grid(args.bases, args.sources,
                                      args.mechanisms, args.admissions)
    print(f"\nGRID: {len(cells)} cells x {args.n_folds} folds x "
          f"{args.seeds} model seeds")
    print("\nAxes collapsed because the engine does not read them "
          "(NOT a judgement about what wins):")
    for m, why in sorted(collapses.items()):
        print(f"  {m:<16} {why}")
    if not collapses:
        print("  none")

    by_mech = {}
    for c in cells:
        by_mech[c['mechanism']] = by_mech.get(c['mechanism'], 0) + 1
    print("\ncells per mechanism: " +
          ', '.join(f'{k}={v}' for k, v in sorted(by_mech.items())))

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or not os.path.isdir(cache):
        print(f"\nREFUSING TO RUN: KIRBY_CACHE_DIR={cache!r} is not a "
              f"directory. The representations live on the external drive; "
              f"without it this regenerates them from scratch.")
        return 2

    ds = args.dataset
    pool = DATASET_REPS[ds]['general']
    aug_names = DATASET_REPS[ds].get('augmentation') or []
    data = load_data(ds, rep_names=pool + aug_names)
    reps, labels = data['reps'], data['labels']
    groups = data.get('groups')
    metric_name, metric_fn, lower = get_domain_metric(data)
    factory = get_model_factories([args.model], task_type=data['task_type'],
                                  n_samples=len(labels))[args.model]
    is_clf = data['task_type'] != 'regression'

    # The target representation is fixed by solo score on training rows of the
    # first fold only — never by argmax over the folds the differences are read
    # on. Falls back to the first pool member if scoring fails.
    splits = list(cv_split(groups, args.n_folds, reps[pool[0]], labels))
    tr0 = splits[0][0]
    solo = {}
    for r in pool:
        try:
            solo[r] = _fit_score(reps[r][tr0], labels[tr0],
                                 reps[r][splits[0][1]], labels[splits[0][1]],
                                 factory, metric_fn, 1)
        except Exception:
            solo[r] = -np.inf if not lower else np.inf
    best = (min(solo, key=solo.get) if lower else max(solo, key=solo.get))
    print(f"\n{ds}: target={best} ({reps[best].shape[1]} features), "
          f"metric={metric_name}, lower_is_better={lower}")
    print(f"  curated blocks: {[(a, reps[a].shape[1]) for a in aug_names]}")

    rows, t0 = [], time.time()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f'aug_sweep_{ds}.json')

    for ci, cell in enumerate(cells):
        for fi, (tr, va) in enumerate(splits):
            rng = np.random.RandomState(9000 + fi)
            try:
                score, info = run_cell(cell, reps, pool, best, aug_names,
                                       labels, groups, tr, va, factory,
                                       metric_fn, lower, args.seeds, is_clf,
                                       rng, fold=fi)
            except Exception as e:
                rows.append({**cell, 'fold': fi, 'error': repr(e)[:300]})
                continue
            if score is None:
                rows.append({**cell, 'fold': fi, **info})
                continue
            rows.append({**cell, 'fold': fi, 'score': score, **info})
        done = ci + 1
        el = time.time() - t0
        print(f"  [{done}/{len(cells)}] {el:6.0f}s  "
              f"eta {el / done * (len(cells) - done):6.0f}s  {cell}",
              flush=True)
        with open(out, 'w') as f:
            json.dump({'dataset': ds, 'target': best, 'metric': metric_name,
                       'lower_is_better': lower, 'model': args.model,
                       'n_folds': args.n_folds, 'seeds': args.seeds,
                       'collapsed_axes': collapses, 'rows': rows}, f, indent=1)

    print(f"\n  -> {out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
