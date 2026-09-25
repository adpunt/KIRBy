#!/usr/bin/env python3
"""Is n12's hybrid win a width effect? Narrow the base representation and see.

THE FINDING THIS FOLLOWS. In n12's result file of 2026-09-20 on
flab_thermo_tresanco under outlier_p10 at level 0.5, the unweighted hybrid space
beat the unweighted base representation in 6 of 6 detection average precision
readings under lightgbm_gain and 6 of 6 under treeshap -- the largest and most
consistent effect in that file, and larger than the importance weighting, which
won 5 of 6 and 3 of 6.

THE QUESTION. Two things differ between those two spaces at once, not one. The
hybrid is built from a MIX of representations, and it is 50 features wide while
the base representation is thousands. Either could be what the detector is
responding to. Nothing in the file separates them.

This separates them on the side that needs no stored search: keep the base
representation and cut it to the same widths by importance, hard, and score
detection in each. If a plain narrowing of one representation also beats the
full-width base representation, the 6-of-6 is about how many features the
distance is computed over and not about mixing representations.

It also puts n12's importance weighting on the same axis. Weighting multiplies a
feature by its importance raised to a power; hard selection at a budget is the
limit of that as the power grows. Both are measured here at the same widths.

⛔ NO HYBRID IS BUILT AND NONE IS CLAIMED. flab_thermo_tresanco's stored search
is missing the distance_metric column commit 9c6f7d7 added, so n12.run cannot
open the pool at all. What this measures is the base representation at several
widths. It does not measure the hybrid, and it cannot say the hybrid's win is
ONLY width -- only whether width alone reproduces it.

⛔ ROWS ARE NEVER CUT. All 672 antibodies.
"""
import os

for _k, _v in {
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'KIRBY_LGBM_THREADS': '1', 'KIRBY_RF_JOBS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'KIRBY_CACHE_DIR': '/Volumes/seagate/kirby_cache',
}.items():
    os.environ.setdefault(_k, _v)

if os.environ.get('KIRBY_SMOKE_N', '').strip():
    raise SystemExit('KIRBY_SMOKE_N is a ROW CAP inside load_data. Unset it.')

import argparse
import json
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np

# Feature counts the base representation is cut to. 50 is n12's own search
# budget and the width both hybrid recipes in the result file were built at.
WIDTHS = [50, 200, 500]
OUT = '/Volumes/seagate/local_noise_run/loop_n12'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='flab_thermo_tresanco')
    p.add_argument('--condition', default='outlier_p10')
    p.add_argument('--level', type=float, default=0.5)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--seeds', default='0')
    p.add_argument('--out', default=OUT)
    p.add_argument('--base-rep', required=True,
                   help=('the single representation to compare against. '
                         'Required: the fixed preference list that used to '
                         'choose it was deleted 2026-09-25; the noise '
                         'experiments now compare against each downstream '
                         "model's top 2 single representations by its "
                         'stored score'))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    from experiments._shared import load_data, cv_split
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores,
    )
    from experiments.noise import n12_importance_weighted_distance as n12
    from kirby.hybrid import compute_feature_importance
    from kirby.similarity.distance_config import get_distance_metric
    from sklearn.preprocessing import StandardScaler

    rep_names = DATASET_REPS[a.dataset]['general']
    data = load_data(a.dataset, rep_preset='all', rep_names=rep_names)
    for r, X in list(data['reps'].items()):
        if int(np.isnan(X).sum()):
            data['reps'][r] = np.nan_to_num(X, nan=0.0)
    reps, labels, groups = data['reps'], data['labels'], data['groups']
    is_clf = data['task_type'] == 'classification'
    base_rep_name = a.base_rep
    if base_rep_name not in reps:
        raise SystemExit(f'--base-rep {base_rep_name!r} is not among the loaded '
                         f'representations {sorted(reps)}')
    full_width = reps[base_rep_name].shape[1]
    print(f'{len(labels)} molecules, base representation {base_rep_name}, '
          f'{full_width} features, neighbours found under '
          f'{get_distance_metric()}, exponent in the file '
          f'{n12.IMPORTANCE_EXPONENT}', flush=True)

    noise_groups, _ng, _lg = noise_groups_for(data)
    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, a.folds, sample_X, labels, mode='full'))

    rows = {}
    moved, n_train = [], []

    def score(space_name, X, y, mask, imp):
        for est_name, est_cls in n12.ESTIMATOR_FACTORIES.items():
            for k in n12.K_VALUES:
                try:
                    sc = est_cls(k=k, metric=None, is_clf=is_clf).estimate(X, y)
                    det = detection_scores(mask, sc, a.condition)
                except Exception as e:
                    print(f'  {space_name} {est_name}_k{k} failed: {e}')
                    continue
                if det is None:
                    continue
                rows.setdefault((imp, f'{est_name}_k{k}', space_name),
                                []).append(
                    float(det['detection_average_precision']))

    for fold_idx, (tr, va) in enumerate(splits):
        for seed in seeds:
            spec = noise_spec_for(a.dataset, a.condition, a.level,
                                  noise_seed=seed, is_clf=is_clf)
            y_noisy, cmask, _ = inject_train_labels(
                labels, spec, train_idx=tr, is_clf=is_clf,
                groups=noise_groups, dataset_name=a.dataset)
            mask = np.asarray(cmask).ravel().astype(bool)
            if mask.size == len(labels):
                mask = mask[tr]
            y_tr = y_noisy[tr] if len(y_noisy) == len(labels) else y_noisy
            moved.append(int(mask.sum()))
            n_train.append(int(mask.size))

            X_base = reps[base_rep_name][tr]
            X_base_std = StandardScaler().fit_transform(X_base)

            for imp_method in n12.IMPORTANCE_METHODS:
                t = time.time()
                try:
                    imp = compute_feature_importance(
                        {base_rep_name: X_base}, y_tr,
                        method=imp_method)[base_rep_name]
                except Exception as e:
                    print(f'  {imp_method} importance failed: {e}')
                    continue
                nonzero = int((np.asarray(imp) > 0).sum())
                print(f'  fold {fold_idx + 1} draw {seed} {imp_method}: '
                      f'importance in {time.time() - t:.0f}s; '
                      f'{nonzero} of {full_width} features scored above zero',
                      flush=True)

                # n12's two base conditions, at full width.
                score(f'full_{full_width}_unweighted', X_base_std, y_tr,
                      mask, imp_method)
                n12.IMPORTANCE_EXPONENT = 0.5
                score(f'full_{full_width}_weighted_e0.5',
                      n12._weight_features(X_base_std, imp), y_tr, mask,
                      imp_method)

                order = np.argsort(np.asarray(imp))[::-1]
                for w in WIDTHS:
                    if w >= full_width:
                        continue
                    keep = np.sort(order[:w])
                    Xw = X_base_std[:, keep]
                    score(f'top{w}_unweighted', Xw, y_tr, mask, imp_method)
                    score(f'top{w}_weighted_e0.5',
                          n12._weight_features(Xw, np.asarray(imp)[keep]),
                          y_tr, mask, imp_method)
                print(f'    scored {len(WIDTHS) * 2 + 2} spaces in '
                      f'{time.time() - t:.0f}s', flush=True)

    report(a, base_rep_name, full_width, rows, moved, n_train)
    return 0


def report(a, base_rep_name, full_width, rows, moved, n_train):
    spaces = [f'full_{full_width}_unweighted', f'full_{full_width}_weighted_e0.5']
    for w in WIDTHS:
        if w < full_width:
            spaces += [f'top{w}_unweighted', f'top{w}_weighted_e0.5']
    imps = sorted({k[0] for k in rows})
    ests = sorted({k[1] for k in rows})
    print('\n' + '=' * 100)
    print('DETECTION AVERAGE PRECISION against the molecules whose labels '
          'moved. Higher is better.')
    print(f'Dataset {a.dataset}, condition {a.condition}, injected level '
          f'{a.level}, {a.folds} outer folds. The space is the '
          f'{base_rep_name} representation, z-scored, cut to its highest-'
          f'importance features.')
    if moved:
        print(f'{float(np.mean(moved)):.1f} of '
              f'{float(np.mean(n_train)):.1f} training molecules had their '
              f'label moved, averaged over {len(moved)} passes.')
    print('One row is one importance scorer and one neighbour count.')
    print('=' * 100)
    hdr = f'{"scorer":<15}{"neighbours":<22}'
    for s in spaces:
        hdr += f'{s[:19]:>20}'
    print(hdr)
    base_key = f'full_{full_width}_unweighted'
    wins = {s: 0 for s in spaces}
    means = {s: [] for s in spaces}
    n = 0
    for imp in imps:
        for est in ests:
            vals = {s: rows.get((imp, est, s)) for s in spaces}
            if any(v is None for v in vals.values()):
                continue
            n += 1
            m = {s: float(np.mean(v)) for s, v in vals.items()}
            line = f'{imp:<15}{est:<22}'
            for s in spaces:
                line += f'{m[s]:>20.4f}'
                means[s].append(m[s])
                if s != base_key and m[s] > m[base_key]:
                    wins[s] += 1
            print(line)
    if not n:
        print('no complete row')
        return
    print(f'\nRows above: {n}. Each is one importance scorer and one neighbour '
          f'count.')
    for s in spaces:
        if s == base_key:
            print(f'  {s}: mean average precision over those {n} rows '
                  f'{float(np.mean(means[s])):.4f} (this is the reference)')
        else:
            print(f'  {s}: mean average precision {float(np.mean(means[s])):.4f}; '
                  f'beat the full-width unweighted space in {wins[s]} of {n} '
                  f'rows')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out,
                        f'{a.dataset}__{a.condition}__width_vs_weighting.json')
    with open(path, 'w') as f:
        json.dump({'dataset': a.dataset, 'condition': a.condition,
                   'level': a.level, 'folds': a.folds,
                   'base_rep': base_rep_name, 'full_width': full_width,
                   'widths': WIDTHS, 'n_rows': n,
                   'reading': 'detection_average_precision',
                   'labels_moved_per_training_fold': float(np.mean(moved)),
                   'training_molecules_per_fold': float(np.mean(n_train)),
                   'mean_by_space': {s: float(np.mean(means[s]))
                                     for s in spaces},
                   'rows_beating_full_width_unweighted': {
                       s: wins[s] for s in spaces if s != base_key},
                   'per_row': {f'{k[0]}|{k[1]}|{k[2]}': float(np.mean(v))
                               for k, v in rows.items()}}, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    sys.exit(main())
