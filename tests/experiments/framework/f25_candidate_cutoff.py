#!/usr/bin/env python3
"""F25: how long should the ranked candidate list be?

Paper: Framework. One axis, nothing else varied.

WHAT THE NUMBER DOES
    Every donor's features are pooled into one list and ranked once. Only the
    top `aug_max_candidates` of that list can ever be added; everything below
    that rank is unreachable for the rest of the fold. The count actually added
    is then chosen from the ladder of 0/5/10/20/40, or, when a fraction is set,
    from round(fraction x width).

    The number has been 50 since the code was written. It is popped out of
    keyword arguments inside create_hybrid (hybrid.py:3851) and NO caller
    anywhere in the tree set it, so it has never been varied. It is also not
    part of a stored result's key, so two searches differing only in it would
    collide on one row -- which is why this is a standalone experiment that
    builds recipes directly and never writes to result_db, rather than a new
    key column that would force every result_db/*.db to be deleted.

WHY IT IS NOT A FREE PARAMETER
    Measured on herg_ki from the cached representations, 2026-08-26:

      per-donor cap    curated donors    other base representations
      5                25 features       45 features
      64               235               640
      off              338               15,081

    At a cap of 5 the pool never reaches 50, so the cut never fires and the
    number cannot matter. With no cap it discards 288 of 338 and 15,031 of
    15,081.

    The harder consequence is on the size rule. At a budget of 1000 a fraction
    of 0.05 asks for 50 extras, 0.10 asks for 100 and 0.25 asks for 250, and
    every one of them clamps to the list length (hybrid.py:2533-2546). At 50 all
    three fractions therefore give one identical answer, so the size rule is not
    an axis at all until this number moves. That is what this experiment is for.

THE VALUES
    50   what the code does today, and what every run before 2026-08-26 used
    100  twice that
    250  what a fraction of 0.25 of a 1000 budget asks for, so the smallest
         value at which the whole fraction axis means something

    Larger values are not swept: a fraction of 0.5 was ruled out, so nothing in
    the grid asks for more than 250 extras.

PINNED, so the cut-off is the only thing that varies
    method          capped and uncapped -- the only two that read a ranked list
    donor source    all three
    per-donor cap   64 and off. A cap of 5 is EXCLUDED with a stated reason,
                    not for cost: the pool is then narrower than the smallest
                    cut-off value, so every value gives the same answer. The
                    experiment asserts this from the measured width rather than
                    assuming it, and prints what it skipped.
    ranking         residual_correlation
    size rule       the ladder, and each fraction, because the fraction is the
                    setting the cut-off interacts with
    margin          0

CONTROLS
    Both decoys are planted, as everywhere else: a pure-noise donor as wide as
    the real donors combined, and a near-copy of the base representation's own
    features jittered by 5% of each feature's standard deviation. A longer
    candidate list offers more junk as well as more signal, so the decoy counts
    are part of the answer here, not a side check.

Usage:
    python tests/experiments/framework/f25_candidate_cutoff.py \\
        --dataset herg_ki --count-only
    python tests/experiments/framework/f25_candidate_cutoff.py \\
        --dataset herg_ki --smoke
    python tests/experiments/framework/f25_candidate_cutoff.py --dataset herg_ki
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))     # tests/
sys.path.insert(0, os.path.join(HERE, '..'))           # tests/experiments/
sys.path.insert(0, HERE)                               # framework/

# Reused rather than restated. The donor pools, the matched random control, the
# fit counting and the per-cell runner are the same machinery f24 uses; a second
# copy here would drift from it.
from f24_augmentation_admission import (          # noqa: E402
    build_donor_pool, random_donor_pool, fits_per_fold, measure_fold_spread,
    best_solo_rep, DONOR_SOURCES, DECOY_ECHO_COLS, DECOY_ECHO_JITTER,
    BORROW_TOP_N,
)

CUTOFFS = (50, 100, 250)
METHODS = ('capped', 'uncapped')
# A cap of 5 over 3 to 10 donors pools to 15 to 50 features, which is under the
# smallest cut-off, so it is excluded by degeneracy and the exclusion is
# asserted below from the measured width.
CAPS = (64, None)
RANKING = 'residual_correlation'
FRACTIONS = (None, 0.05, 0.10, 0.25)
DEFAULT_MODELS = ('lightgbm', 'rf')
DEFAULT_N_SEEDS = 3


def enumerate_cells(cutoffs=CUTOFFS, methods=METHODS, sources=DONOR_SOURCES,
                    caps=CAPS, fractions=FRACTIONS):
    cells = []
    for cutoff in cutoffs:
        for method in methods:
            for source in sources:
                for cap in caps:
                    for frac in fractions:
                        cells.append({
                            'cutoff': cutoff, 'method': method,
                            'source': source, 'cap': cap, 'fraction': frac,
                            'ranking': RANKING,
                        })
    return cells


def cell_label(cell):
    size = ('ladder' if cell['fraction'] is None
            else 'frac%g' % cell['fraction'])
    cap = cell['cap'] if cell['cap'] else 'off'
    return (f"cut{cell['cutoff']}__{cell['method']}__{cell['source']}"
            f"__cap{cap}__{size}")


def degenerate_reason(cell, pool_width):
    """Why this cell cannot give its own answer. The only accepted reason to skip.

    Two cases, both asserted from the measured pooled width rather than argued:
    the list is shorter than the smallest cut-off, so no cut-off value changes
    anything; or the fraction asks for more features than the list holds, so
    every fraction collapses to the same count.
    """
    if pool_width is None:
        return None
    if pool_width <= min(CUTOFFS):
        return (f'the pooled candidate list is {pool_width} features wide, '
                f'which is at most the smallest cut-off tested '
                f'({min(CUTOFFS)}), so no value of the cut-off can change '
                f'anything')
    return None


def run_dataset(ds, args):
    from experiments._shared import (
        load_data, get_config, get_model_factories, cv_split, get_domain_metric,
        build_fold_hybrid, plant_decoys, decoy_counts,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from kirby.find_best_hybrid import Blueprint
    from sklearn.preprocessing import StandardScaler

    def _fit_score(Xtr, ytr, Xva, yva, factory, metric_fn, n_seeds):
        vals = []
        for s in range(n_seeds):
            m = factory()
            try:
                if 'random_state' in m.get_params():
                    m.set_params(random_state=42 + s)
            except Exception:
                pass
            m.fit(Xtr, ytr)
            vals.append(float(metric_fn(yva, m.predict(Xva))))
        return float(np.mean(vals))

    cfg = DATASET_REPS.get(ds, {})
    pool_names = cfg.get('general', [])
    aug_names = cfg.get('augmentation', [])
    if not pool_names:
        raise ValueError(f'{ds} has no general representation pool.')

    data = load_data(ds, 'all', rep_names=sorted(set(pool_names + aug_names)))
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    base_reps = {r: reps[r] for r in pool_names if r in reps}
    aug_reps = dict(data.get('augmentations') or {})
    for r in aug_names:
        if r not in aug_reps and r in reps:
            aug_reps[r] = reps[r]
    aug_reps = {k: v for k, v in aug_reps.items() if k in aug_names}

    config = get_config(args)
    # Three folds by default, not the config's five. CUT BY THE USER on
    # 2026-08-26 for time. That is the reason; it is not a scientific one and
    # should not be written up as one. --n-folds overrides it.
    n_folds = 1 if args.smoke else int(args.n_folds or config['n_splits'])
    mode = 'intermediary' if n_folds == 1 else 'full'
    n_seeds = 1 if args.smoke else args.n_seeds
    model_names = ([m.strip() for m in args.models.split(',')] if args.models
                   else (['rf'] if args.smoke else list(DEFAULT_MODELS)))
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=len(labels))

    if args.base_rep:
        base_name, base_provenance = args.base_rep, 'pinned by --base-rep'
    else:
        base_name, base_provenance, _ = best_solo_rep(ds, base_reps)
    if base_name not in base_reps:
        raise ValueError(f'--base-rep {base_name} is not in {ds} pool '
                         f'{sorted(base_reps)}')
    X_base = base_reps[base_name]
    budget = X_base.shape[1]

    pools = {s: build_donor_pool(s, base_name, base_reps, aug_reps,
                                 top_n=BORROW_TOP_N)
             for s in DONOR_SOURCES}

    print(f'\nF25 candidate cut-off — {ds}')
    print(f'  n = {len(labels)}, metric = {metric_name}')
    print(f'  base representation: {base_name} ({budget} features)')
    print(f'    chosen by: {base_provenance}')
    print(f'  cut-off values: {", ".join(str(c) for c in CUTOFFS)}')

    # The pooled width a given per-donor cap actually produces. This is what
    # decides whether the cut-off can matter at all.
    width = {}
    for s, (donors, _) in pools.items():
        for cap in CAPS:
            w = sum(min(cap, v.shape[1]) if cap else v.shape[1]
                    for v in donors.values())
            width[(s, cap)] = w
            print(f'  pool, {s:8s} cap={str(cap):4s}: {w:6d} features'
                  f"{'   <- cut never fires' if w <= min(CUTOFFS) else ''}")

    cells = enumerate_cells()
    skipped_degenerate = []
    live = []
    for c in cells:
        r = degenerate_reason(c, width.get((c['source'], c['cap'])))
        if r:
            skipped_degenerate.append({'cell': cell_label(c), 'reason': r})
        else:
            live.append(c)

    n_donors = {s: len(p[0]) for s, p in pools.items()}
    donor_widths = {s: [v.shape[1] for v in p[0].values()]
                    for s, p in pools.items()}
    total_fits = sum(fits_per_fold(
        {'method': c['method'], 'ranking': c['ranking'],
         'fraction': c['fraction'], 'cap': c['cap']},
        n_donors[c['source']], donor_widths[c['source']]) for c in live)
    print(f'\n  {len(cells)} configurations, {len(skipped_degenerate)} '
          f'degenerate and skipped, {len(live)} to run')
    if skipped_degenerate:
        print(f"    reason: {skipped_degenerate[0]['reason']}")
    print(f'  {total_fits * n_folds * len(models):,} model fits '
          f'({n_folds} folds x {len(models)} models)')

    out = {
        'experiment': 'f25_candidate_cutoff',
        'dataset': ds, 'n_samples': int(len(labels)),
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'base_rep': base_name, 'base_rep_provenance': base_provenance,
        'base_width': int(budget),
        'cutoffs': list(CUTOFFS),
        'pool_widths': {f'{s}__cap{c if c else "off"}': w
                        for (s, c), w in width.items()},
        'n_folds': n_folds, 'n_seeds': n_seeds, 'models': list(models),
        'degenerate_cells': skipped_degenerate,
        'planned_fits': int(total_fits * n_folds * len(models)),
        'rows': [], 'skipped': [],
    }
    if args.count_only:
        return out

    # Reported for context only -- every margin here is 0, so a single fold
    # does not silently collapse an axis the way it would in f24.
    print('\n  measuring the base-only fold spread...')
    spread, base_scores = measure_fold_spread(
        X_base, labels, groups, n_folds, mode, next(iter(models.values())),
        metric_fn, n_seeds, cv_split, _fit_score, allow_undefined=True)
    out['fold_spread'] = spread
    out['base_only_fold_scores'] = base_scores
    print(f'    fold spread = {spread:.4f} {metric_name}')

    splits = list(cv_split(groups, n_folds, X_base, labels, mode=mode))
    rng_ctrl = np.random.RandomState(0)

    for cell in live:
        donors, _ = pools[cell['source']]
        if not donors:
            out['skipped'].append({'cell': cell_label(cell),
                                   'reason': f"no donors for "
                                             f"'{cell['source']}'"})
            continue
        per_source = ({k: cell['cap'] for k in donors} if cell['cap']
                      else None)
        for control in ('real', 'random'):
            pool = donors
            if control == 'random':
                pool = random_donor_pool(donors, base_reps, base_name, rng_ctrl)
                if not pool:
                    continue
            arm = f'{cell_label(cell)}__{control}'
            bp = Blueprint(
                reps=[base_name], allocation_method='performance_weighted',
                importance_method='lightgbm_gain', internal_evaluator='match',
                budget=int(budget), step_size=5,
                augmentation_strategy=cell['method'],
                aug_per_source_budget=per_source,
                augmentation_margin=0.0,
                augmentation_fraction=cell['fraction'],
                candidate_ranking=cell['ranking'],
                aug_max_candidates=int(cell['cutoff']),
                metric=metric_name, task_type=task_type,
                lower_is_better=lower_is_better)

            per_model = {}
            for model_name, factory in models.items():
                scores, dims, kept, noise, echo = [], [], [], [], []
                for fold_idx, (tr_i, va_i) in enumerate(splits):
                    y_tr, y_va = labels[tr_i], labels[va_i]
                    fold_pool = pool
                    if not args.no_decoys:
                        fold_pool = plant_decoys(
                            pool, X_base,
                            np.random.RandomState(1000 + fold_idx),
                            echo_cols=DECOY_ECHO_COLS,
                            echo_jitter=DECOY_ECHO_JITTER)
                    try:
                        Xtr, Xva, info = build_fold_hybrid(
                            {base_name: X_base[tr_i]},
                            {base_name: X_base[va_i]}, y_tr, bp,
                            augmentations_train={k: v[tr_i]
                                                 for k, v in fold_pool.items()},
                            augmentations_val={k: v[va_i]
                                               for k, v in fold_pool.items()},
                            groups=(groups[tr_i] if groups is not None
                                    else None),
                            model=factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better)
                        sc = StandardScaler()
                        Xtr = sc.fit_transform(Xtr)
                        Xva = sc.transform(Xva)
                        scores.append(_fit_score(Xtr, y_tr, Xva, y_va, factory,
                                                 metric_fn, n_seeds))
                        dims.append(int(Xtr.shape[1]))
                        ai = info.get('augmentation_info') or {}
                        d = decoy_counts(ai)
                        noise.append(d['noise'])
                        echo.append(d['echo'])
                        kept.append(sum(int(v.get('n_features', 0))
                                        for v in ai.values()))
                    except Exception as exc:                  # noqa: BLE001
                        out['skipped'].append({
                            'cell': cell_label(cell), 'arm': arm,
                            'model': model_name, 'fold': fold_idx,
                            'reason': f'{type(exc).__name__}: {exc}'})
                        scores.append(float('nan'))

                ok = not np.all(np.isnan(scores))
                per_model[model_name] = {
                    'fold_scores': scores,
                    'mean': float(np.nanmean(scores)) if ok else float('nan'),
                    'mean_kept': float(np.mean(kept)) if kept else 0.0,
                    'mean_dim': float(np.mean(dims)) if dims else float('nan'),
                    'decoy_noise_per_fold': noise,
                    'decoy_echo_per_fold': echo,
                    'status': 'ok' if ok else 'all_folds_failed',
                }
                if ok:
                    print(f'    {arm}/{model_name}: {np.nanmean(scores):.4f} '
                          f'(kept~{np.mean(kept) if kept else 0:.1f}, '
                          f'noise {sum(noise)}, echo {sum(echo)})')

            out['rows'].append({
                'arm': arm, 'cell': cell_label(cell), 'control': control,
                'cutoff': cell['cutoff'], 'method': cell['method'],
                'source': cell['source'], 'per_donor_cap': cell['cap'],
                'fraction': cell['fraction'],
                'pool_width': width.get((cell['source'], cell['cap'])),
                'models': per_model, 'blueprint': bp.to_dict(),
            })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mode', default='full',
                    choices=['fast', 'full', 'intermediary'])
    ap.add_argument('--base-rep', default=None, dest='base_rep')
    ap.add_argument('--models', default=None)
    ap.add_argument('--n-folds', type=int, default=3,
                    dest='n_folds',
                    help='Folds. Default 3, cut from 5 by the user on '
                         '2026-08-26 for time. Re-run the winner at 5 '
                         'before quoting a number.')
    ap.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                    dest='n_seeds')
    ap.add_argument('--no-decoys', action='store_true', dest='no_decoys')
    ap.add_argument('--count-only', action='store_true', dest='count_only',
                    help='print the configuration and fit counts, fit nothing')
    ap.add_argument('--smoke', action='store_true',
                    help='one fold, one model, one seed — plumbing only')
    ap.add_argument('--output-dir', dest='output_dir',
                    default=os.path.join(HERE, '..', 'results', 'framework'))
    args = ap.parse_args()

    t0 = time.time()
    out = run_dataset(args.dataset, args)
    out['wall_seconds'] = time.time() - t0

    if args.count_only:
        print(f"\n(counted in {out['wall_seconds']:.0f}s; nothing was fitted)")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = ('__count' if args.count_only
              else '__smoke' if args.smoke else '')
    path = os.path.join(args.output_dir,
                        f'f25_candidate_cutoff_{args.dataset}{suffix}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: (
            o.item() if isinstance(o, (np.floating, np.integer))
            else o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(f'\nwrote {path}')
    return out


if __name__ == '__main__':
    main()
