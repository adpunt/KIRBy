#!/usr/bin/env python3
"""n12's importance exponent, swept, on the two conditions that need no stored search.

WHY THIS EXISTS BESIDE _loop_n12_exponent.py. That script calls n12.run, which
asks the pool for a hybrid recipe before it measures anything. The stored search
for flab_thermo_tresanco is 11,692 rows written on 2026-09-20 at 04:08, and
commit 9c6f7d7 (2026-09-20 18:19) added a `distance_metric` column to the key,
so opening that store now raises StaleResultDB and n12.run cannot start. The
script has therefore printed one line and exited 0 on every supervisor pass. It
is still correct and still queued; it runs once the store is rebuilt.

The exponent, though, does not need the pool. n12 builds two conditions from the
base representation alone -- `base_unweighted` and `base_imp_weighted` -- and the
exponent is applied only in the second. This measures those two, at every
exponent on the ladder, using n12's own helpers: n12._weight_features for the
space, n12.ESTIMATOR_FACTORIES for the suspicion score, n12.K_VALUES for the
neighbour counts, and noise_helpers.detection_scores for the reading.

WHAT THE READING IS. Detection average precision: how well a molecule's
suspicion score ranks the molecules whose label actually moved. No downstream
model is fitted to produce it, which is why a seven-point sweep costs what two
arms cost in _loop_n12_exponent.py.

⛔ ROWS ARE NEVER CUT. All 672 antibodies, 448 of them in each training fold at
3 folds. What is cut is the search space: one injected level, and no repair
step. The exponent, the importance scorer, the neighbour count and the distance
are all kept -- they are the question.

⛔ EXPONENT 0.0 IS A CONTROL, NOT A CANDIDATE. At 0.0 every weight is 1 and the
weighted space IS the unweighted one, so its reading must equal
base_unweighted's to floating point. If it does not, the harness is wrong and
nothing else in the table means anything.
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
    raise SystemExit(
        "KIRBY_SMOKE_N is set. It is a ROW CAP inside load_data, not a speed "
        "flag, and a detector graded against a corrupted subset has nothing to "
        "find once the rows are cut. Unset it.")

import argparse
import json
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np

# The ladder in the repo, plus 0.0 which is the control described above.
EXPONENTS = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
OUT = '/Volumes/seagate/local_noise_run/loop_n12'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='flab_thermo_tresanco')
    p.add_argument('--condition', default='gaussian')
    p.add_argument('--level', type=float, default=0.5)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    from experiments._shared import load_data, cv_split, pick_noise_base_rep
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.dataset_policy import noise_groups_for
    from experiments.noise.noise_helpers import (
        inject_train_labels, noise_spec_for, detection_scores,
        detection_is_defined, DETECTION_IS_DEFINED,
    )

    # ⛔ REFUSE A CONDITION THIS SCRIPT CANNOT READ, BEFORE ANYTHING LOADS.
    # The only reading here is detection average precision, and
    # detection_scores returns None under a condition where EVERY label moves
    # -- precision is then 1.0 by construction. `gaussian` is one of those, and
    # it is n12's own default condition, so it is the easy one to pass by
    # mistake. This script was registered with the supervisor under `gaussian`
    # and ran to completion four times between 03:26 and 08:04 on 2026-09-21,
    # each time computing every importance array and every neighbour search and
    # then printing `no complete row` and writing no file -- so the supervisor
    # saw no result and started it again.
    if not detection_is_defined(a.condition):
        raise SystemExit(
            f"condition {a.condition!r} moves every training label, so "
            f"detection average precision is 1.0 by construction and "
            f"noise_helpers.detection_scores returns None for every reading. "
            f"This script has no other reading. Pass one of "
            f"{list(DETECTION_IS_DEFINED)}.")
    from experiments.noise import n12_importance_weighted_distance as n12
    from kirby.hybrid import compute_feature_importance
    from kirby.similarity.distance_config import get_distance_metric
    from sklearn.preprocessing import StandardScaler

    print('DISTANCE THE NEIGHBOURS ARE FOUND UNDER:', get_distance_metric())
    print('EXPONENT THAT DISTANCE MATCHES:',
          n12.EXPONENT_MATCHED_TO_METRIC.get(get_distance_metric()))
    print('EXPONENT IN THE FILE NOW:', n12.IMPORTANCE_EXPONENT)

    rep_names = DATASET_REPS[a.dataset]['general']
    t0 = time.time()
    data = load_data(a.dataset, rep_preset='all', rep_names=rep_names)
    for r, X in list(data['reps'].items()):
        if int(np.isnan(X).sum()):
            data['reps'][r] = np.nan_to_num(X, nan=0.0)
    reps, labels, groups = data['reps'], data['labels'], data['groups']
    is_clf = data['task_type'] == 'classification'
    base_rep_name = pick_noise_base_rep(reps, a.dataset)
    print(f'loaded {len(labels)} molecules in {time.time() - t0:.1f}s; '
          f'base representation {base_rep_name}, '
          f'{reps[base_rep_name].shape[1]} features')

    noise_groups, _ng, _lg = noise_groups_for(data)
    sample_X = next(iter(reps.values()))
    splits = list(cv_split(groups, a.folds, sample_X, labels, mode='full'))

    # rows[(imp_method, est_key, exponent)] -> list of one average precision per pass
    rows = {}
    unweighted = {}
    n_corrupt_seen, n_train_seen = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        for seed in seeds:
            spec = noise_spec_for(a.dataset, a.condition, a.level,
                                  noise_seed=seed, is_clf=is_clf)
            y_noisy, cmask, _prov = inject_train_labels(
                labels, spec, train_idx=train_idx, is_clf=is_clf,
                groups=noise_groups, dataset_name=a.dataset)
            y_train = y_noisy[train_idx] if len(y_noisy) == len(labels) else y_noisy
            mask = np.asarray(cmask).ravel().astype(bool)
            if mask.size == len(labels):
                mask = mask[train_idx]
            n_corrupt_seen.append(int(mask.sum()))
            n_train_seen.append(int(mask.size))

            X_base = reps[base_rep_name][train_idx]
            X_base_std = StandardScaler().fit_transform(X_base)

            # The unweighted space, once per pass: no exponent enters it.
            for est_name, est_cls in n12.ESTIMATOR_FACTORIES.items():
                for k in n12.K_VALUES:
                    est_key = f'{est_name}_k{k}'
                    try:
                        sc = est_cls(k=k, metric=None, is_clf=is_clf).estimate(
                            X_base_std, y_train)
                        det = detection_scores(mask, sc, a.condition)
                    except Exception as e:
                        print(f'  base_unweighted {est_key} failed: {e}')
                        continue
                    if det is None:
                        continue
                    unweighted.setdefault(est_key, []).append(
                        float(det['detection_average_precision']))

            for imp_method in n12.IMPORTANCE_METHODS:
                t = time.time()
                try:
                    imp = compute_feature_importance(
                        {base_rep_name: X_base}, y_train,
                        method=imp_method)[base_rep_name]
                except Exception as e:
                    print(f'  {imp_method} importance failed: {e}')
                    continue
                print(f'  fold {fold_idx + 1} draw {seed} {imp_method}: '
                      f'importance in {time.time() - t:.1f}s', flush=True)
                for exponent in EXPONENTS:
                    n12.IMPORTANCE_EXPONENT = exponent
                    X_w = n12._weight_features(X_base_std, imp)
                    for est_name, est_cls in n12.ESTIMATOR_FACTORIES.items():
                        for k in n12.K_VALUES:
                            est_key = f'{est_name}_k{k}'
                            try:
                                sc = est_cls(k=k, metric=None,
                                             is_clf=is_clf).estimate(X_w, y_train)
                                det = detection_scores(mask, sc, a.condition)
                            except Exception as e:
                                print(f'  {imp_method} e={exponent} {est_key} '
                                      f'failed: {e}')
                                continue
                            if det is None:
                                continue
                            rows.setdefault(
                                (imp_method, est_key, exponent), []).append(
                                float(det['detection_average_precision']))
                n12.IMPORTANCE_EXPONENT = 0.5

            # WRITE AFTER EVERY PASS, NOT ONLY AT THE END. Nine passes take
            # hours and the four runs of 2026-09-21 that were killed part way
            # left nothing on disk at all. The file is rewritten each time and
            # carries `passes_behind_each_number`, so a reader always knows how
            # many fold-and-draw passes are averaged in what they are looking
            # at.
            report(a, base_rep_name, rows, unweighted, n_corrupt_seen,
                   n_train_seen, verbose=False)

    report(a, base_rep_name, rows, unweighted, n_corrupt_seen, n_train_seen)
    return 0


def report(a, base_rep_name, rows, unweighted, n_corrupt_seen, n_train_seen,
           verbose=True):
    from experiments.noise import n12_importance_weighted_distance as n12
    _p = print if verbose else (lambda *a, **k: None)
    _p('\n' + '=' * 78)
    _p('DETECTION AVERAGE PRECISION, averaged over the fold-and-draw passes.')
    _p(f'Dataset {a.dataset}, condition {a.condition}, injected level '
          f'{a.level}, {a.folds} outer folds.')
    if n_corrupt_seen:
        _p(f'Molecules whose label moved, per training fold: '
              f'{float(np.mean(n_corrupt_seen)):.1f} of '
              f'{float(np.mean(n_train_seen)):.1f} training molecules, '
              f'averaged over {len(n_corrupt_seen)} passes.')
    _p(f'The space is the {base_rep_name} representation, z-scored, each '
          f'feature multiplied by its importance raised to the exponent.')
    _p('One row is one importance scorer and one neighbour count. Higher '
          'is better.')
    _p('=' * 78)
    ests = sorted({k[1] for k in rows})
    imps = sorted({k[0] for k in rows})
    hdr = f'{"scorer":<15}{"neighbours":<20}{"unweighted":>12}'
    for e in EXPONENTS:
        hdr += f'{("e=" + str(e)):>10}'
    _p(hdr)
    per_exp_wins = {e: 0 for e in EXPONENTS}
    per_exp_vals = {e: [] for e in EXPONENTS}
    # The unweighted reading of the SAME row, so the weighting can be compared
    # against plain within the row rather than between two column means.
    plain_vals = []
    n_rows = 0
    for imp in imps:
        for est in ests:
            vals = {e: rows.get((imp, est, e)) for e in EXPONENTS}
            if any(v is None for v in vals.values()):
                continue
            n_rows += 1
            m = {e: float(np.mean(v)) for e, v in vals.items()}
            u = unweighted.get(est)
            um = float(np.mean(u)) if u else float('nan')
            line = f'{imp:<15}{est:<20}{um:>12.4f}'
            plain_vals.append(um)
            best = max(m, key=lambda e: m[e])
            for e in EXPONENTS:
                line += f'{m[e]:>10.4f}'
                per_exp_vals[e].append(m[e])
            per_exp_wins[best] += 1
            _p(line)
    if not n_rows:
        _p('no complete row')
        return
    _p()
    _p(f'Rows above: {n_rows}. Each is one importance scorer and one '
          f'neighbour count.')
    for e in EXPONENTS:
        _p(f'  exponent {e}: mean average precision over those {n_rows} '
              f'rows {float(np.mean(per_exp_vals[e])):.4f}; it was the best '
              f'exponent in {per_exp_wins[e]} of {n_rows} rows')
    cur, matched = 0.5, 1.0
    a_cur = float(np.mean(per_exp_vals[cur]))
    a_mat = float(np.mean(per_exp_vals[matched]))
    beat = sum(1 for i in range(n_rows)
               if per_exp_vals[matched][i] > per_exp_vals[cur][i])
    _p(f'\nExponent {matched} against exponent {cur}, the value in the file '
          f'now: {a_mat:.4f} against {a_cur:.4f} mean average precision, and '
          f'{matched} scored higher in {beat} of {n_rows} rows.')
    # THE MARGIN THE WEIGHTING BUYS, WITHIN EACH ROW.
    # Each row is one importance scorer and one neighbour count, and its
    # unweighted reading is the same estimator and the same neighbour count in
    # the plain z-scored representation. Subtracting within the row is what
    # says whether weighting helped; two column means do not, because the
    # estimators differ from each other by more than the weighting moves any
    # of them.
    margin_mean = {}
    margin_rows_won = {}
    for e in EXPONENTS:
        _d = [per_exp_vals[e][i] - plain_vals[i] for i in range(n_rows)
              if plain_vals[i] == plain_vals[i]]
        margin_mean[e] = float(np.mean(_d)) if _d else float('nan')
        margin_rows_won[e] = int(sum(1 for x in _d if x > 0))
    _p('\nWHAT THE WEIGHTING BUYS OVER THE PLAIN SPACE, within each row:')
    for e in EXPONENTS:
        _p(f'  exponent {e}: {margin_mean[e]:+.4f} mean average precision over '
           f'the unweighted reading of the same estimator and neighbour count, '
           f'and it was higher in {margin_rows_won[e]} of {n_rows} rows')
    ctrl = float(np.mean(per_exp_vals[0.0]))
    uall = [float(np.mean(v)) for v in unweighted.values() if v]
    _p(f'CONTROL: exponent 0.0 reads {ctrl:.4f}, the unweighted space reads '
          f'{float(np.mean(uall)):.4f}. These must agree.')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(
        a.out, f'{a.dataset}__{a.condition}__exponent_sweep_base.json')
    with open(path, 'w') as f:
        json.dump({
            'dataset': a.dataset, 'condition': a.condition,
            'level': a.level, 'folds': a.folds,
            'base_rep': base_rep_name,
            'exponent_in_file': n12.IMPORTANCE_EXPONENT,
            'exponents': EXPONENTS,
            'reading': 'detection_average_precision',
            'n_corrupted_per_fold_mean': float(np.mean(n_corrupt_seen)),
            'n_training_per_fold_mean': float(np.mean(n_train_seen)),
            'mean_average_precision_by_exponent': {
                str(e): float(np.mean(per_exp_vals[e])) for e in EXPONENTS},
            'rows_best_exponent_counts': {
                str(e): per_exp_wins[e] for e in EXPONENTS},
            'n_rows': n_rows,
            'passes_behind_each_number': len(n_corrupt_seen),
            'rows_where_exponent_1p0_beat_0p5': beat,
            'mean_margin_over_unweighted_by_exponent': {
                str(e): margin_mean[e] for e in EXPONENTS},
            'rows_where_exponent_beat_unweighted': {
                str(e): margin_rows_won[e] for e in EXPONENTS},
            'exponent_in_the_file_when_this_ran': 0.5,
            'per_row': {f'{k[0]}|{k[1]}|{k[2]}': float(np.mean(v))
                        for k, v in rows.items()},
            'base_unweighted': {k: float(np.mean(v))
                                for k, v in unweighted.items()},
        }, f, indent=1)
    _p('wrote', path)


if __name__ == '__main__':
    sys.exit(main())
