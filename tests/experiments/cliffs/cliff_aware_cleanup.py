#!/usr/bin/env python3
"""Does knowing which molecules are on cliffs make a cleanup work better.

HER OUTLINE, verbatim, which this is written from:

    "Performance in downstream tasks like AC-aware noise mitigation. Look at
    the best outlier detection methods, call one M, and then subsequent best
    label handling methods, call one L. Take dataset D. Run M on D, giving you
    a list of outliers. Now we have to determine if the outliers are noise or
    ACs. Take the AC detection method in question and run it, cross-check with
    the list of outliers. Now first, apply L without AC-awareness to get D'.
    Then apply L with AC-awareness to get D''. Compare the results of training
    on D, D', and D''. Assuming M and L work, for a higher-is-better metric, we
    should see D'' > D' > D. Potentially D > D' if the removal of ACs was
    actually damaging. But the important thing being D'' should be the best.
    And then the difference between D'' and D and D'' and D' should indicate
    the relative performance of different AC detection methods."

ONE ROW IS one AC definition, one M, one L, one setting of each, one
downstream model, one feature space and one fold. It carries four scores of
the same held-out molecules in the dataset's own metric:

  score_D            fitted on the training labels as the loader gives them
  score_Dprime       fitted on what L returned, knowing nothing about cliffs
  score_Dprimeprime  fitted on what L returned with the cliff molecules put
                     back, so L touched none of them
  score_random_put_back  the same with as many NON-cliff molecules put back
                     instead, which is the control: it differs from D'' in
                     which molecules were spared and not in how many

⛔ NO NOISE IS INJECTED. Her sentence is "Take dataset D", so D is the
dataset's own labels. There is no clean reference to score against and none is
reported: the comparison is between the three training sets, not against a
truth. This is what separates this experiment from the noise suite's own
D-against-D', which injects a known corruption and can score the clean labels.

THE CROSS-CHECK IS ON EVERY ROW. Her "cross-check with the list of outliers":
how many of M's outliers are cliff molecules, how many are not, and how many
cliff molecules M did not flag. Without it a row cannot be read, because D''
equals D' whenever M flagged no cliff molecule at all, and then the difference
between them is zero by construction rather than by measurement.

WHY TWO SETTINGS HAVE NO VALUE HERE. M and L are "the best" of the noise
experiments' own results, and those results do not exist yet. So this file is
finished and does not run until `outlier_detection_methods` and
`label_handling_methods` are written in the settings file. It says so and stops
rather than picking one.

Usage, as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments cliff_aware_cleanup
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments._shared import (get_domain_metric, get_model_factories,  # noqa: E402
                                 input_fingerprint,
                                 save_experiment_results)
from experiments.noise import cliff_rules as cr                      # noqa: E402
from experiments.noise import noise_cache as nc                      # noqa: E402
from experiments.noise import n07_remediation as n07                 # noqa: E402
from experiments.noise import n25_cliff_exemption as n25             # noqa: E402
from experiments.noise.n07_remediation import _predict_for_metric    # noqa: E402
from experiments.cliffs import cliff_sets as P                       # noqa: E402
from experiments.cliffs import definitions as D                      # noqa: E402
from experiments.cliffs import settings as S                         # noqa: E402

EXP_ID = 'cliff_aware_cleanup'

#: The feature spaces a model is fitted on, the same three the error-split
#: experiment fits.
SPACES = ('base_1', 'base_2', 'hybrid')

#: ⛔ THERE IS NO FIT CEILING, and there was one until 2026-11-06. It refused
#: above 20,000 model fits, a number Claude chose from a fit it had timed on
#: synthetic data. That is pruning the search space on an estimate, which
#: Adelaide's standing rule forbids: a cut needs the setting to be degenerate,
#: broken or undefined, and "my guess says this is too big" is none of those.
#: The run now prints the number of fits and the measured cost of one, and
#: starts.
FIT_COST_MEASURED = (
    'one svm fit and predict on 1,129 molecules by 900 features took 2.06 s '
    'and one dnn 9.18 s, measured 2026-11-04 on synthetic data of that shape. '
    'A real per-fit time is whatever this run records.')


def known_detectors():
    """The detector names this file can run, read from the registry."""
    from kirby.utils.noise_estimation import ESTIMATOR_REGISTRY
    return sorted(ESTIMATOR_REGISTRY)


def known_cleanups(is_clf):
    """The cleanup names this file can run on this task."""
    return sorted(n25.cleanups_for(is_clf))


def _fit_score(X, y, weight, X_va, y_va, factory, metric_fn, metric_name,
               label):
    try:
        m = factory()
        nc.require_weights_accepted(m, weight)
        if weight is None:
            m.fit(X, y)
        else:
            m.fit(X, y, sample_weight=weight)
        pred = _predict_for_metric(m, X_va, metric_name)
        return float(metric_fn(y_va, pred)), None
    except Exception as exc:
        return float('nan'), f'{label}: {type(exc).__name__}: {exc}'


def _arm(cleaned, X_tr, y_tr, X_va, y_va, factory, metric_fn, metric_name,
         label):
    """One training set fitted and scored."""
    keep = cleaned.get('keep')
    X = cleaned.get('X')
    if X is None:
        X = X_tr if keep is None else X_tr[np.asarray(keep, dtype=bool)]
    y = cleaned.get('y')
    if y is None:
        col = cleaned.get('label_column')
        y = (y_tr if col is None else np.asarray(col))
        if keep is not None:
            y = y[np.asarray(keep, dtype=bool)]
    return _fit_score(X, y, cleaned.get('sample_weight'), X_va, y_va,
                      factory, metric_fn, metric_name, label)


def cross_check(outlier_flags, cliff_flags):
    """Her cross-check, as counts of molecules.

    Every number says what it counts. Without these a row cannot be read: D''
    equals D' whenever M flagged no cliff molecule, and the difference is then
    zero by construction.
    """
    o = np.asarray(outlier_flags, dtype=bool)
    c = np.asarray(cliff_flags, dtype=bool)
    return {
        'n_training_molecules': int(o.size),
        'n_outliers_M_flagged': int(o.sum()),
        'n_cliff_molecules': int(c.sum()),
        'n_outliers_that_are_cliff_molecules': int((o & c).sum()),
        'n_outliers_that_are_not_cliff_molecules': int((o & ~c).sum()),
        'n_cliff_molecules_M_did_not_flag': int((~o & c).sum()),
        'share_of_outliers_that_are_cliff_molecules': (
            float((o & c).sum()) / float(o.sum()) if o.sum() else None),
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    print('\n' + '=' * 70)
    print("CLIFF-AWARE CLEANUP: training on D, D' and D''")
    print('=' * 70)

    from experiments.noise.noise_helpers import (fold_configurations,
                                                 fold_model_inputs)
    from experiments.decided_grid import decided_models

    stored = S.read()
    labels = np.asarray(data['labels']).ravel()
    is_clf = data.get('task_type') == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    groups = data.get('groups')

    # ── the settings that have to be hers ────────────────────────────────
    try:
        detectors = S.methods_for('outlier_detection_methods',
                                  known_detectors(), stored)
        cleanups = S.methods_for('label_handling_methods',
                                 known_cleanups(is_clf), stored)
    except (S.Unset, ValueError) as exc:
        print(f'  NOT RUN: {exc}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'waiting_for': str(exc),
                        'settings': S.record(stored)},
               'rows': [], 'result_status': 'skipped', 'skipped': True}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    shares = S.values('outlier_shares', stored)
    scopes = S.values('definition_scope', stored)
    folds, n_folds = P.folds_of(data, config)
    try:
        models = [m for m in decided_models(dataset_name, 'general')
                  if m != 'knn']
    except Exception as exc:
        why = (f'the decided grid has no entry for {dataset_name}: '
               f'{type(exc).__name__}: {exc}')
        print(f'  REFUSED: {why}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'refused': why}, 'rows': [],
               'result_status': 'failed'}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    # ── how big the grid is, before anything is fitted ───────────────────
    # The definitions are counted on a context that needs no search: the
    # settings decide how many there are, and the count is reported whether
    # the run starts or not.
    n_defs = _count_definitions(data, stored, scopes)
    per_definition = (len(detectors) * len(cleanups) * len(shares)
                      * len(models) * len(SPACES) * len(folds) * 3)
    n_fits = n_defs * per_definition + len(models) * len(SPACES) * len(folds)
    print(f'  {n_defs} definitions x {len(detectors)} detector(s) x '
          f'{len(cleanups)} cleanup(s) x {len(shares)} outlier share(s) x '
          f'{len(models)} model(s) x {len(SPACES)} space(s) x '
          f'{len(folds)} fold(s) x 3 arms = {n_fits:,} model fits',
          flush=True)
    print(f'  one fit, measured: {FIT_COST_MEASURED}', flush=True)

    jsonl = Path(output_dir) / EXP_ID / f'{dataset_name}__rows.jsonl'
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if jsonl.exists():
        jsonl.unlink()

    n_rows, errors, fold_records = 0, {}, []
    t_all = time.time()

    with open(jsonl, 'a') as sink:
        for fold_i, train_idx, val_idx in folds:
            t0 = time.time()
            y_tr, y_va = labels[train_idx], labels[val_idx]
            g_tr = (None if groups is None
                    else np.asarray(groups)[np.asarray(train_idx)])
            print(f'\n  fold {fold_i} of {n_folds}: {len(train_idx)} training, '
                  f'{len(val_idx)} held-out molecules', flush=True)

            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
            spaces = {}
            for model in models:
                factory = get_model_factories(
                    [model], data.get('task_type') or 'regression',
                    len(labels)).get(model)
                inputs, unavailable, what = fold_model_inputs(
                    data, dataset_name, model, entries.get(model),
                    train_idx=train_idx, val_idx=val_idx, y_train=y_tr,
                    factory=factory, metric=metric_name, spaces=SPACES)
                for space, why in (unavailable or {}).items():
                    errors[f'fold{fold_i}|{model}|{space}'] = why
                for space, (X_tr, X_va) in inputs.items():
                    spaces[(model, space)] = (X_tr, X_va, factory,
                                              (what or {}).get(space))

            # D, once per space: it reads no detector and no cleanup.
            score_D = {}
            for key, (X_tr, X_va, factory, _what) in spaces.items():
                score_D[key], why = _fit_score(
                    X_tr, y_tr, None, X_va, y_va, factory, metric_fn,
                    metric_name, f'D {key}')
                if why:
                    errors[f'fold{fold_i}|D|{key}'] = why

            # M's scores, once per space per detector.
            outlier_scores = {}
            for detector in detectors:
                for (model, space), (X_tr, _X_va, _f, _w) in spaces.items():
                    try:
                        sc = nc.cached_detector_scores(
                            dataset_name, X_tr, y_tr, is_clf=is_clf,
                            detector=detector, groups=g_tr)
                    except Exception as exc:
                        errors[f'fold{fold_i}|M={detector}|{space}'] = \
                            f'{type(exc).__name__}: {exc}'
                        sc = None
                    if sc is not None:
                        outlier_scores[(detector, model, space)] = \
                            np.asarray(sc, dtype=float)

            for scope in scopes:
                ctx, _whole, record = D.context_for_fold(
                    data, dataset_name, train_idx=train_idx, fold_i=fold_i,
                    n_folds=n_folds, stored=stored, scope=scope, verbose=False)
                if scope != 'per_fold':
                    print('    note: this scope computes the definition over '
                          'every molecule, so a held-out label reached it',
                          flush=True)
                defs = D.expand(ctx, stored=stored, scope=scope)
                got, errs, _cost = D.masks(ctx, defs)
                for key, why in errs.items():
                    errors[f'{scope}|{fold_i}|{key}'] = why

                for d in defs:
                    if d.key not in got:
                        continue
                    cliff_mask, counts = got[d.key]
                    cliff_train = _on_training_rows(cliff_mask, scope,
                                                    train_idx, len(labels))
                    for detector in detectors:
                        for (model, space), (X_tr, X_va, factory, what) in \
                                spaces.items():
                            sc = outlier_scores.get((detector, model, space))
                            if sc is None:
                                continue
                            for share in shares:
                                flags = _flags_from(sc, float(share))
                                check = cross_check(flags, cliff_train)
                                for cleanup in cleanups:
                                    axis, values = n25.setting_axis(cleanup,
                                                                    is_clf)
                                    for setting in values:
                                        row = _one_row(
                                            dataset_name, fold_i, scope, d,
                                            detector, cleanup, axis, setting,
                                            share,
                                            model, space, what, X_tr, X_va,
                                            y_tr, y_va, factory, metric_fn,
                                            metric_name, lower_is_better,
                                            sc, cliff_train, check, g_tr,
                                            is_clf, score_D.get((model, space)),
                                            errors)
                                        if row is None:
                                            continue
                                        sink.write(json.dumps(row,
                                                              default=float)
                                                   + '\n')
                                        n_rows += 1
                sink.flush()
                fold_records.append({
                    'fold': int(fold_i), 'definition_scope': scope,
                    'n_definitions': len(defs),
                    'n_definitions_that_did_not_run': len(errs),
                    'spaces_fitted': sorted(f'{m}|{s}' for m, s in spaces),
                    'hybrids_refused': record['hybrids_refused'],
                    'seconds': round(time.time() - t0, 1),
                })
                print(f'    scope {scope}: {n_rows} rows so far, '
                      f'{round(time.time() - t0, 1)}s', flush=True)

    meta = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'metric': metric_name, 'lower_is_better': lower_is_better,
        'task_type': data.get('task_type'),
        'models': models, 'feature_spaces': list(SPACES),
        'outlier_detection_methods': list(detectors),
        'label_handling_methods': list(cleanups),
        'outlier_shares': [float(s) for s in shares],
        'n_outer_folds': int(n_folds),
        'definition_scopes_run': list(scopes),
        'no_noise_was_injected': (
            'her sentence is "Take dataset D", so D is the dataset\'s own '
            'labels. There is no clean reference and none is reported.'),
        'per_fold': fold_records,
        'n_rows': n_rows, 'rows_file': str(jsonl),
        'seconds': round(time.time() - t_all, 1),
        'settings': S.record(stored),
    }
    result = {
        'meta': meta, 'errors': errors,
        'what_one_row_is': (
            'one AC definition, one outlier detector, one cleanup and its '
            'setting, one outlier share, one downstream model, one feature '
            "space and one fold: four scores of the same held-out molecules "
            "in the dataset's own metric."),
        'how_to_read_a_row': (
            "D is the training labels as given, D' is the cleanup knowing "
            "nothing about cliffs, D'' is the cleanup with the cliff "
            'molecules put back, and the random arm puts back as many '
            'non-cliff molecules instead. Where M flagged no cliff molecule, '
            "D'' equals D' and the difference between them is zero by "
            'construction, which is what the cross-check counts show.'),
        'where_every_row_is': str(jsonl),
    }
    save_experiment_results(result, output_dir, EXP_ID, dataset_name,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better,
                            fingerprint=input_fingerprint(data, config))
    print(f'\n  {n_rows} rows written to {jsonl}', flush=True)
    return result


def _one_row(dataset_name, fold_i, scope, d, detector, cleanup, axis, setting,
             share,
             model, space, what, X_tr, X_va, y_tr, y_va, factory, metric_fn,
             metric_name, lower_is_better, scores, cliff_train, check, g_tr,
             is_clf, score_D, errors):
    """One row: the three arms and the control, fitted and scored.

    `axis` is which of its own settings this cleanup is swept over
    (n25.setting_axis): the flag rate for a cleanup that thresholds the
    suspicion scores, a strength for one that reads every score, and neither
    for one that reads no score at all.
    """
    kw = dict(noise_fraction=float(share), is_clf=is_clf, groups=g_tr,
              dataset_name=dataset_name)
    if axis == 'flag_rate':
        kw['noise_fraction'] = float(setting)
    elif axis == 'strength':
        kw['strength'] = setting
    try:
        cleaned = n07.cleaned_training_set(cleanup, X_tr, y_tr, scores, **kw)
    except Exception as exc:
        errors[f'{scope}|{fold_i}|L={cleanup}|{setting}'] = \
            f'{type(exc).__name__}: {exc}'
        return None

    exempt = np.asarray(cliff_train, dtype=bool)
    restored = n25.put_back(cleaned, exempt, X_tr, y_tr)
    rand, n_drawn = n25.random_exempt_mask(exempt, int(exempt.sum()),
                                           int(fold_i), 0)
    restored_rand = n25.put_back(cleaned, rand, X_tr, y_tr)

    sd, why_d = _arm(cleaned, X_tr, y_tr, X_va, y_va, factory, metric_fn,
                     metric_name, "D'")
    sdd, why_dd = _arm(restored, X_tr, y_tr, X_va, y_va, factory, metric_fn,
                       metric_name, "D''")
    srand, why_r = _arm(restored_rand, X_tr, y_tr, X_va, y_va, factory,
                        metric_fn, metric_name, 'random put back')
    for why in (why_d, why_dd, why_r):
        if why:
            errors[f'{scope}|{fold_i}|{d.key}|{cleanup}|{setting}'] = why

    def _gain(treatment, reference):
        if not (np.isfinite(treatment) and np.isfinite(reference)):
            return float('nan')
        return ((reference - treatment) if lower_is_better
                else (treatment - reference))

    return {
        'dataset': dataset_name, 'fold': int(fold_i),
        'definition': d.key, 'definition_scope': scope,
        'coefficient': d.record.get('coefficient'),
        'representation': d.record.get('representation'),
        'similarity_used_labels': d.record.get('similarity_used_labels'),
        'outlier_detection_method': detector,
        'label_handling_method': cleanup,
        'label_handling_setting': setting,
        'label_handling_setting_axis': axis,
        'outlier_share': float(share),
        'model': model, 'feature_space': space, 'what_the_space_is': what,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'n_held_out_molecules': int(len(y_va)),
        'score_D': score_D, 'score_Dprime': sd, 'score_Dprimeprime': sdd,
        'score_random_put_back': srand,
        'Dprimeprime_over_Dprime': _gain(sdd, sd),
        'Dprimeprime_over_D': (_gain(sdd, score_D)
                               if score_D is not None else float('nan')),
        'Dprime_over_D': (_gain(sd, score_D)
                          if score_D is not None else float('nan')),
        'Dprimeprime_over_random_put_back': _gain(sdd, srand),
        'n_cliff_molecules_the_cleanup_would_have_touched':
            n25.n_spared(cleaned, restored, exempt, y_tr),
        'n_non_cliff_molecules_put_back_in_the_control': int(n_drawn),
        'control_put_back_fewer_than_the_cliff_arm':
            bool(n_drawn < int(exempt.sum())),
        **check,
    }


def _flags_from(scores, share):
    """The `share` most suspect training molecules, as a flag. n44's rule."""
    s = np.asarray(scores, dtype=float)
    k = int(round(share * len(s)))
    out = np.zeros(len(s), dtype=bool)
    if k <= 0:
        return out
    out[np.argsort(-s)[:k]] = True
    return out


def _on_training_rows(mask, scope, train_idx, n_total):
    """The cliff flag over the fold's training rows.

    Under 'per_fold' the mask is already one per training molecule, because
    the definition was computed on that subset. Under 'whole_dataset' it is
    one per molecule and is taken at the training rows.
    """
    m = np.asarray(mask, dtype=bool)
    if len(m) == len(train_idx):
        return m
    if len(m) == n_total:
        return m[np.asarray(train_idx)]
    raise ValueError(
        f'a cliff flag of {len(m)} molecules fits neither the '
        f'{len(train_idx)} training molecules nor the {n_total} in the '
        f'dataset (scope {scope!r})')


def _count_definitions(data, stored, scopes):
    """How many definitions the settings ask for, without running a search.

    The hybrid names are stood in for so the count includes the definitions
    that read one; whether a hybrid actually builds is the run's business.
    """
    from kirby.similarity.preprocessing import feature_kind
    from experiments.cliffs.binary_hybrid import EXPECTED_KIND, HYBRID_REP_NAME
    labels = np.asarray(data['labels']).ravel()
    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    kinds = {k: feature_kind(v) for k, v in reps.items()}
    stand_in = dict(reps)
    for kind, name in HYBRID_REP_NAME.items():
        key = f'{name}__stand_in'
        kinds[key] = EXPECTED_KIND[kind]
        stand_in[key] = np.zeros((len(labels), 1))
    ctx = cr.CliffContext(labels, smiles=data.get('smiles'), reps=stand_in,
                          is_clf=(data.get('task_type') == 'classification'),
                          dataset_name=data.get('dataset_name'))
    return sum(len(D.expand(ctx, stored=stored, kinds=kinds, scope=scope))
               for scope in scopes)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--output', default='tests/results/cliffs')
    ap.add_argument('--folds', default=None)
    a = ap.parse_args(argv)
    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401
    if a.folds:
        os.environ['KIRBY_NOISE_ONLY_FOLDS'] = a.folds
    data = load_data(a.dataset)
    run(data, {}, a.output, a.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
