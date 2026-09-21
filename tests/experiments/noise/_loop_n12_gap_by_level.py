#!/usr/bin/env python3
"""How much the injected noise actually costs n12's downstream fit, level by level.

WHY. n12 reports repair as a recovery percentage: the repaired fit's distance
from the noisy fit, as a share of the clean fit's distance from the noisy fit.
That share is only readable when the denominator -- the clean-to-noisy gap -- is
large enough to divide by. On flab_thermo_tresanco under outlier_p10 at level
0.5, the random forest's gap was 0.008551 spearman against a clean score of
0.458964, and 71 of the 192 recovery readings in that result file came out
outside 0 to 100 percent, the extremes reaching -615 and +578 percent.

n07, n11, n21 and n06 all refuse a recovery figure when the gap is under
GAP_MIN_FRAC = 0.02 of the clean score. n12 has no such floor: its only guard is
abs(gap) > 1e-10. So n12 is the one file in the directory that divides by a gap
the other four would have refused.

⛔ THE GAP IS REPORTED BEFORE ANY RECOVERY NUMBER. This script fits nothing but
the clean and the noisy reference -- no detector, no repair, no recovery. It
prints, per level and per model, the clean score, the noisy score, the gap and
the gap as a percentage of the clean score, so the level at which the dose is
large enough to divide by is chosen from a measurement rather than assumed.

⛔ ROWS ARE NEVER CUT. All 672 antibodies. What is cut is the search space: no
detector, no repair, no hybrid.
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

GAP_MIN_FRAC = 0.02
OUT = '/Volumes/seagate/local_noise_run/loop_n12'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='flab_thermo_tresanco')
    p.add_argument('--condition', default='outlier_p10')
    p.add_argument('--levels', default='0.5,1.0,1.5,2.0,3.0')
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--seeds', default='0')
    p.add_argument('--models', default='rf,lightgbm')
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    levels = [float(x) for x in a.levels.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    models = [m.strip() for m in a.models.split(',')]

    from experiments._shared import (
        load_data, cv_split, pick_noise_base_rep, get_domain_metric,
        get_model_factories,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for,
    )
    from sklearn.preprocessing import StandardScaler

    rep_names = DATASET_REPS[a.dataset]['general']
    data = load_data(a.dataset, rep_preset='all', rep_names=rep_names)
    for r, X in list(data['reps'].items()):
        if int(np.isnan(X).sum()):
            data['reps'][r] = np.nan_to_num(X, nan=0.0)
    reps, labels, groups = data['reps'], data['labels'], data['groups']
    is_clf = data['task_type'] == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    base_rep_name = pick_noise_base_rep(reps, a.dataset)
    mf = get_model_factories(models, task_type=data['task_type'],
                             n_samples=len(labels))
    models = [m for m in models if m in mf]
    print(f'{len(labels)} molecules, base representation {base_rep_name}, '
          f'{reps[base_rep_name].shape[1]} features, metric {metric_name}, '
          f'lower_is_better {lower_is_better}, models {models}', flush=True)

    noise_groups, _ng, _lg = noise_groups_for(data)
    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, a.folds, sample_X, labels, mode='full'))

    clean = {m: [] for m in models}
    noisy = {(m, lv): [] for m in models for lv in levels}
    moved = {lv: [] for lv in levels}
    n_train = []

    for fold_idx, (tr, va) in enumerate(splits):
        Xtr = reps[base_rep_name][tr]
        Xva = reps[base_rep_name][va]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)
        y_va = labels[va]
        n_train.append(len(tr))
        for m in models:
            t = time.time()
            mdl = mf[m]()
            mdl.fit(Xtr_s, labels[tr])
            clean[m].append(float(metric_fn(y_va, mdl.predict(Xva_s))))
            print(f'  fold {fold_idx + 1} {m} clean '
                  f'{clean[m][-1]:.6f} ({time.time() - t:.0f}s)', flush=True)
        for lv in levels:
            for seed in seeds:
                spec = noise_spec_for(a.dataset, a.condition, lv,
                                      noise_seed=seed, is_clf=is_clf)
                y_noisy, cmask, _ = inject_train_labels(
                    labels, spec, train_idx=tr, is_clf=is_clf,
                    groups=noise_groups, dataset_name=a.dataset)
                mask = np.asarray(cmask).ravel().astype(bool)
                if mask.size == len(labels):
                    mask = mask[tr]
                moved[lv].append(int(mask.sum()))
                y_tr_noisy = (y_noisy[tr] if len(y_noisy) == len(labels)
                              else y_noisy)
                for m in models:
                    mdl = mf[m]()
                    mdl.fit(Xtr_s, y_tr_noisy)
                    noisy[(m, lv)].append(
                        float(metric_fn(y_va, mdl.predict(Xva_s))))
            print(f'  fold {fold_idx + 1} level {lv}: '
                  + ', '.join(f'{m} noisy {np.mean(noisy[(m, lv)][-len(seeds):]):.6f}'
                              for m in models), flush=True)

    print('\n' + '=' * 84)
    print(f'THE CLEAN-TO-NOISY GAP, dataset {a.dataset}, condition '
          f'{a.condition}, {a.folds} outer folds, metric {metric_name}.')
    print(f'A positive gap means the injected noise cost the model something. '
          f'The last column is the gap as a percentage of the clean score; '
          f'n07, n11, n21 and n06 refuse a recovery figure below '
          f'{GAP_MIN_FRAC:.0%}.')
    print('=' * 84)
    print(f'{"model":<12}{"level":>8}{"labels moved":>14}{"clean":>12}'
          f'{"noisy":>12}{"gap":>12}{"gap % of clean":>16}')
    out = []
    for m in models:
        c = float(np.mean(clean[m]))
        for lv in levels:
            nz = float(np.mean(noisy[(m, lv)]))
            gap = (nz - c) if lower_is_better else (c - nz)
            frac = 100.0 * gap / abs(c) if abs(c) > 1e-12 else float('nan')
            flag = '' if frac >= 100 * GAP_MIN_FRAC else '   UNDER FLOOR'
            print(f'{m:<12}{lv:>8}{float(np.mean(moved[lv])):>14.1f}'
                  f'{c:>12.6f}{nz:>12.6f}{gap:>12.6f}{frac:>15.2f}%{flag}')
            out.append({'model': m, 'level': lv,
                        'labels_moved_per_training_fold': float(np.mean(moved[lv])),
                        'training_molecules_per_fold': float(np.mean(n_train)),
                        'clean_score': c, 'noisy_score': nz, 'gap': gap,
                        'gap_pct_of_clean': frac,
                        'over_gap_min_frac': bool(frac >= 100 * GAP_MIN_FRAC)})
    print(f'\nOne row is one downstream model at one injected level. '
          f'"labels moved" counts training molecules whose label was changed, '
          f'out of {float(np.mean(n_train)):.0f} training molecules per fold.')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out,
                        f'{a.dataset}__{a.condition}__gap_by_level.json')
    with open(path, 'w') as f:
        json.dump({'dataset': a.dataset, 'condition': a.condition,
                   'folds': a.folds, 'metric': metric_name,
                   'lower_is_better': lower_is_better,
                   'base_rep': base_rep_name,
                   'gap_min_frac': GAP_MIN_FRAC, 'rows': out}, f, indent=1)
    print('wrote', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
