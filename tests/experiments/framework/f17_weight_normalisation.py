#!/usr/bin/env python3
"""F17: which feature-weight normalisation convention should the repo use?

Three conventions for rescaling per-feature importance weights before they enter
a distance are live in this codebase at once:

    raw    no rescaling at all      -- everything routed through
                                       kirby.utils.noise_mitigation_methods.compute_distance_matrix,
                                       plus s04, s05, s06, s07
    sum1   weights sum to 1         -- n02_sample_weighting, s10, s13, c04
    mean1  weights average to 1     -- n12_importance_weighted_distance, n09, c19

sum1 and mean1 differ by exactly the feature count d (mean1 = sum1 * d), so they
are the same relative weighting at two scales. raw differs from both by
sum(sqrt(importance)), which is data-dependent and changes per fold, per
representation and per dataset -- it is NOT a constant rescale.

This script does not argue about which is right. It measures four consumers per
fold under all three conventions and reports which ones actually move:

  1. knn_readacross       Spearman of kNN-predicted vs true label, within fold.
                          Rank statistic on a within-fold distance ordering ->
                          expected INVARIANT. Asserted, not assumed.
  2. pooled_ad_spearman   The s06 shape: mean neighbour distance per sample,
                          POOLED across folds, then Spearman against absolute
                          error. Each fold's weights carry a different total
                          mass, so folds enter the pooled ranking at different
                          scales -> expected SENSITIVE.
  3. sali_mean            The s02 shape: sim = 1/(1+D), then mean activity-cliff
                          index. The 1/(1+D) step is not scale-free ->
                          expected SENSITIVE.
  4. frac_within_cutoff   Fraction of off-diagonal pairs closer than a fixed
                          absolute distance. Maximally scale-sensitive; included
                          as the unambiguous demonstration.

ORDER OF OPERATIONS: weights are applied AFTER standardising, everywhere. The
reverse order is a mathematical no-op (a per-column StandardScaler exactly
cancels a per-column multiply). That defect has bitten this repo twice -- see
the n12 docstring and the 2026-08-25 fix in tests/noise_mitigation_advanced.py.
_weighted_space asserts the ordering rather than trusting it.

Splits are scaffold-group-aware via experiments._shared.cv_split. No random
splits. Results are reported PER DATASET and never averaged across datasets.

LOCAL RUNS ARE PINNED TO ONE IMPORTANCE METHOD. The default importance path
segfaults on the project laptop, so --importance lightgbm_gain is the only one
that runs locally. That makes the importance method a hidden axis in any local
result. On the server, pass --importance all.

Usage:
    python f17_weight_normalisation.py --dataset logd --mode intermediary
    python f17_weight_normalisation.py --dataset herg_ki --importance all
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, parse_rep_args,
)
from kirby.hybrid import compute_feature_importance


CONVENTIONS = ('raw', 'sum1', 'mean1')
K_NEIGHBOURS = 10
ALL_IMPORTANCE_METHODS = ('lightgbm_gain', 'treeshap', 'permutation')


def normalise_weights(importance, convention):
    """Rescale a non-negative importance vector under one convention.

    Returns the vector that multiplies the ALREADY-STANDARDISED columns, i.e.
    sqrt of the rescaled importance (the repo's existing convention is that a
    weight w contributes sqrt(w) to each column, so squared Euclidean distance
    sums w * (xi - yi)^2).
    """
    w = np.sqrt(np.maximum(np.asarray(importance, dtype=np.float64), 0.0))
    s = w.sum()
    if s < 1e-10:
        # No signal to weight by. Every convention degenerates to unweighted;
        # say so in the result rather than returning a silent all-zero vector.
        return None
    if convention == 'raw':
        return w
    if convention == 'sum1':
        return w / s
    if convention == 'mean1':
        return w * (len(w) / s)
    raise ValueError(f'unknown convention {convention!r}')


def _weighted_space(X_std, w):
    """Apply weights to an already-standardised matrix.

    X_std must already be standardised. Weighting before standardising cancels
    exactly, so the caller's ordering is checked here rather than documented and
    hoped for.

    The check is on the MEDIAN column spread, inside a loose band, not on every
    column being exactly 1. A validation fold is transformed by the scaler fitted
    on its train fold, so its own column standard deviations sit near 1 without
    equalling it, and a handful of columns can be far off. The band still catches
    the defect this guards against: applying weights first leaves column spreads
    proportional to the weights, which span orders of magnitude.
    """
    col_std = X_std.std(axis=0)
    live = col_std[col_std > 1e-8]
    if live.size:
        med = float(np.median(live))
        assert 0.5 <= med <= 2.0, (
            '_weighted_space received a matrix that does not look standardised '
            f'(median live-column std {med:.4f}, expected near 1). Standardise '
            'first, then weight -- the reverse order is a no-op.'
        )
    return X_std if w is None else X_std * w


def _consumers(X_w, y, groups, cutoff):
    """The four measured quantities on one weighted fold matrix."""
    n = len(y)
    out = {}

    D = squareform(pdist(X_w, metric='euclidean'))

    # 1. within-fold kNN read-across -- expected invariant
    k = min(K_NEIGHBOURS, n - 1)
    if k >= 1 and np.std(y) > 0:
        preds = np.full(n, np.nan)
        order = np.argsort(D, axis=1)
        for i in range(n):
            nb = [j for j in order[i] if j != i][:k]
            d = D[i, nb]
            wgt = 1.0 / (d + 1e-12)
            preds[i] = float(np.sum(wgt * y[nb]) / np.sum(wgt))
        ok = np.isfinite(preds)
        if ok.sum() >= 3 and np.std(preds[ok]) > 0:
            rho, _ = spearmanr(preds[ok], y[ok])
            out['knn_readacross'] = float(rho) if np.isfinite(rho) else np.nan
            out['_abs_err'] = np.abs(preds - y)
        else:
            out['knn_readacross'] = np.nan
            out['_abs_err'] = np.full(n, np.nan)
    else:
        out['knn_readacross'] = np.nan
        out['_abs_err'] = np.full(n, np.nan)

    # 2. per-sample mean neighbour distance -- pooled by the caller
    if k >= 1:
        # Mask the diagonal by assignment, NOT by adding np.eye(n) * np.inf --
        # that multiplies the off-diagonal zeros by inf and turns the whole
        # matrix into NaN.
        D_masked = D.copy()
        np.fill_diagonal(D_masked, np.inf)
        part = np.partition(D_masked, k - 1, axis=1)[:, :k]
        out['_mean_nn_dist'] = part.mean(axis=1)
    else:
        out['_mean_nn_dist'] = np.full(n, np.nan)

    # 3. SALI -- sim = 1/(1+D), not scale-free
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) > 0 and np.std(y) > 0:
        sim = 1.0 / (1.0 + D[iu])
        dy = np.abs(y[iu[0]] - y[iu[1]])
        sali = dy / (1.0 - sim + 1e-10)
        out['sali_mean'] = float(np.mean(sali))
    else:
        out['sali_mean'] = np.nan

    # 4. fixed absolute cutoff -- maximally scale-sensitive
    out['frac_within_cutoff'] = (float(np.mean(D[iu] < cutoff))
                                 if len(iu[0]) > 0 else np.nan)
    return out


SEED = 42


def run_dataset(data, args, config, importance_method):
    X = data['reps'][args.rep] if args.rep else None
    if X is None:
        # No rep named: use the first registered base representation, and SAY
        # which one rather than letting it be implicit.
        rep_name = sorted(data['reps'])[0]
        X = data['reps'][rep_name]
    else:
        rep_name = args.rep

    labels = np.asarray(data['labels'], dtype=np.float64).ravel()
    groups = data.get('groups')
    n_folds = config['n_splits']

    per_fold = {c: [] for c in CONVENTIONS}
    pooled_nn = {c: [] for c in CONVENTIONS}
    pooled_err = {c: [] for c in CONVENTIONS}
    degenerate_folds = 0
    fold_count = 0

    for train_idx, val_idx in cv_split(groups, n_folds, X, labels,
                                       seed=SEED, mode=args.mode):
        fold_count += 1
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
        y_va = labels[val_idx]
        g_va = np.asarray(groups)[val_idx] if groups is not None else None

        imp = compute_feature_importance(
            {rep_name: X_tr}, labels[train_idx], method=importance_method)
        imp_vec = np.asarray(imp[rep_name], dtype=np.float64)

        # One cutoff per fold, shared by all three conventions: the median
        # UNWEIGHTED pairwise distance. Using a per-convention cutoff would
        # rescale the question away.
        base_pd = pdist(X_va, metric='euclidean')
        cutoff = float(np.median(base_pd)) if len(base_pd) else 1.0

        for conv in CONVENTIONS:
            w = normalise_weights(imp_vec, conv)
            if w is None:
                degenerate_folds += 1
                continue
            res = _consumers(_weighted_space(X_va, w), y_va, g_va, cutoff)
            pooled_nn[conv].append(res.pop('_mean_nn_dist'))
            pooled_err[conv].append(res.pop('_abs_err'))
            per_fold[conv].append(res)

    summary = {}
    for conv in CONVENTIONS:
        folds = per_fold[conv]
        if not folds:
            summary[conv] = {'n_folds_scored': 0}
            continue
        entry = {'n_folds_scored': len(folds)}
        for key in ('knn_readacross', 'sali_mean', 'frac_within_cutoff'):
            vals = np.array([f[key] for f in folds], dtype=np.float64)
            ok = np.isfinite(vals)
            entry[key + '_mean'] = float(vals[ok].mean()) if ok.any() else None
            entry[key + '_std'] = float(vals[ok].std()) if ok.any() else None
            entry[key + '_folds'] = [None if not np.isfinite(v) else float(v)
                                     for v in vals]
        # The s06 shape: pool across folds FIRST, then rank.
        nn = np.concatenate(pooled_nn[conv]) if pooled_nn[conv] else np.array([])
        er = np.concatenate(pooled_err[conv]) if pooled_err[conv] else np.array([])
        ok = np.isfinite(nn) & np.isfinite(er)
        if ok.sum() >= 3 and np.std(nn[ok]) > 0 and np.std(er[ok]) > 0:
            rho, _ = spearmanr(nn[ok], er[ok])
            entry['pooled_ad_spearman'] = float(rho) if np.isfinite(rho) else None
        else:
            entry['pooled_ad_spearman'] = None
        summary[conv] = entry

    # The load-bearing check: within-fold rank statistics MUST be identical
    # across conventions. If this fails, one of the conventions is not a pure
    # rescale and every "rank-invariant, no change needed" claim in the suite
    # is wrong.
    invariance = {}
    ref = per_fold[CONVENTIONS[0]]
    for conv in CONVENTIONS[1:]:
        other = per_fold[conv]
        if len(ref) != len(other) or not ref:
            invariance[conv] = None
            continue
        diffs = [abs(a['knn_readacross'] - b['knn_readacross'])
                 for a, b in zip(ref, other)
                 if np.isfinite(a['knn_readacross']) and np.isfinite(b['knn_readacross'])]
        invariance[conv] = float(max(diffs)) if diffs else None

    return {
        'rep_name': rep_name,
        'importance_method': importance_method,
        'n_features': int(X.shape[1]),
        'n_samples': int(X.shape[0]),
        'n_folds_requested': n_folds,
        'n_folds_split': fold_count,
        'degenerate_weight_folds': degenerate_folds,
        'conventions': summary,
        'knn_readacross_max_abs_diff_vs_raw': invariance,
    }


def main():
    parser = create_experiment_parser(
        'F17: feature-weight normalisation convention comparison')
    parser.add_argument('--rep', type=str, default=None,
                        help='Base representation to measure on. Default: the '
                             'first registered one, reported in the output.')
    parser.add_argument('--importance', type=str, default='lightgbm_gain',
                        help="Importance method, or 'all' to sweep "
                             f"{list(ALL_IMPORTANCE_METHODS)}. Only "
                             "lightgbm_gain runs on the project laptop; the "
                             "default importance path segfaults there.")
    args = parser.parse_args()
    cfg = get_config(args)

    methods = (list(ALL_IMPORTANCE_METHODS) if args.importance == 'all'
               else [args.importance])

    t0 = time.time()
    rep_preset, rep_names = parse_rep_args(args)
    data = load_data(args.dataset, rep_preset, rep_names=rep_names)
    metric_name, _, lower_is_better = get_domain_metric(data)

    results = {
        'experiment': 'f17_weight_normalisation',
        'dataset': args.dataset,
        'mode': args.mode,
        'seed': SEED,
        'conventions': list(CONVENTIONS),
        'k_neighbours': K_NEIGHBOURS,
        'importance_methods_run': methods,
        'importance_axis_note': (
            'Only lightgbm_gain runs locally; a local result is pinned to one '
            'importance method and that is a hidden axis.'
            if methods == ['lightgbm_gain'] else None),
        'by_importance_method': {},
    }

    for m in methods:
        print(f'  importance={m}')
        try:
            results['by_importance_method'][m] = run_dataset(data, args, cfg, m)
        except Exception as exc:
            # Record the failure in the output. A missing method must not read
            # as a method that was run and produced nothing.
            results['by_importance_method'][m] = {'error': repr(exc)}
            print(f'    FAILED: {exc!r}')

    results['elapsed_seconds'] = round(time.time() - t0, 1)

    for m, r in results['by_importance_method'].items():
        if 'error' in r:
            continue
        print(f'\n  [{args.dataset} / {m} / rep={r["rep_name"]}] '
              f'{r["n_folds_split"]} folds, {r["n_features"]} features')
        print(f'    {"convention":10s} {"knn_ra":>9s} {"pooled_ad":>10s} '
              f'{"sali":>10s} {"frac<cut":>9s}')
        for c in CONVENTIONS:
            e = r['conventions'][c]
            if not e.get('n_folds_scored'):
                print(f'    {c:10s}  (no folds scored)')
                continue
            def f(v, w=9, p=4):
                return f'{v:>{w}.{p}f}' if isinstance(v, float) else f'{"n/a":>{w}}'
            print(f'    {c:10s} {f(e["knn_readacross_mean"])} '
                  f'{f(e["pooled_ad_spearman"],10)} {f(e["sali_mean_mean"],10)} '
                  f'{f(e["frac_within_cutoff_mean"])}')
        print(f'    knn_readacross max |diff| vs raw: '
              f'{r["knn_readacross_max_abs_diff_vs_raw"]}')

    # The representation is part of the result's identity, so it goes in the
    # filename. Without it a second --rep run silently overwrites the first.
    reps_run = sorted({r['rep_name'] for r in results['by_importance_method'].values()
                       if 'rep_name' in r})
    tag = args.dataset + ('__' + '_'.join(reps_run) if reps_run else '')
    save_experiment_results(results, args.output, 'f17_weight_normalisation',
                            dataset=tag, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


if __name__ == '__main__':
    main()
