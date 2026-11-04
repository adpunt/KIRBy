#!/usr/bin/env python3
"""A model trained to predict MoleculeACE's cliff flag, on scaffold-grouped
folds, against the baselines a trained model has to beat.

HER QUESTION THIS IS WRITTEN FOR. TBD7: "Train my own model on MoleculeACE ...
is this a reasonable task? Is this straightforward?" The file is finished and
does not run until `train_cliff_model_on_moleculeace` has a value in the
settings file, because whether to do it is hers.

WHAT IT PREDICTS, and the one thing that makes it odd. `cliff_mol` is a
property of a MOLECULE that is defined by its PAIRS: a molecule carries the
flag because some other molecule in the same target is similar to it and
differs in potency. So a model given one molecule's features is asked for a
property of its neighbourhood. That is learnable -- a molecule in a dense,
steep region of a target is flagged more often -- but it is not the same task
as deciding whether a PAIR is a cliff, and a number from it should not be read
as the latter.

THE BASELINES, because a correlation on its own says nothing. Each is scored
on the same held-out molecules as the model:

  the majority class            every molecule called not-a-cliff. Its
                                Matthews correlation is 0 by construction and
                                its accuracy is whatever the flag's share is,
                                which runs 8.8 to 54.0 per cent across the 30
                                targets. This is why accuracy is not the
                                number to read.
  a published rule              ECFP4 Tanimoto at 0.55 with twice the median
                                absolute deviation, which is a rule that does
                                NOT read their definition. Pooled over the 30
                                targets its Matthews correlation against their
                                flag is 0.317
                                (tests/results/comparisons/opig_moleculeace_cliff_check.json,
                                read 2026-11-04).
  their own rule re-implemented MoleculeACE's any-of-three at 0.90 with a
                                1-log-unit gap, pooled Matthews correlation
                                0.99875 against their flag, same file. This is
                                the ceiling and it is not a baseline a model
                                should be compared against: it is the same
                                definition computed twice.

⛔ THEIR SPLIT IS NOT USED. The CSVs ship one. It is a published split and
this lab does not use published or random splits, so the folds are
scaffold-grouped on our own Bemis-Murcko groups, which
build_moleculeace.py computes.

Usage:
    python tests/experiments/cliffs/moleculeace_cliff_model.py \\
        --targets CHEMBL4203_Ki
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

from experiments.noise import cliff_rules as cr                      # noqa: E402
from experiments.cliffs import moleculeace_agreement as MA           # noqa: E402
from experiments.cliffs import settings as S                         # noqa: E402

OUT_DEFAULT = Path('tests/results/comparisons')

#: The published rules scored beside the model on the same held-out molecules.
#: The first does not read their definition; the second is their definition.
BASELINE_RULES = ('n01', 'moleculeace')

#: The classifiers fitted. Both are in the suite's own registry; the point of
#: two is that a number from one alone cannot be told from a property of that
#: model.
MODELS = ('rf', 'lightgbm')


def _scores(y_true, y_prob):
    """Matthews correlation, AUROC and accuracy at the 0.5 threshold.

    The Matthews correlation is the number to read: accuracy is whatever the
    flag's share is on a target where the flag is rare, and the share runs 8.8
    to 54.0 per cent across the 30 targets.
    """
    from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                                 roc_auc_score)
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_prob, dtype=float)
    pred = p >= 0.5
    out = {'n_molecules': int(y.size),
           'n_cliff_molecules': int(y.sum()),
           'share_cliff_molecules': (float(y.mean()) if y.size else None)}
    try:
        out['auroc'] = (float(roc_auc_score(y, p))
                        if len(set(y.tolist())) == 2 else None)
    except Exception:
        out['auroc'] = None
    out['matthews_correlation'] = (float(matthews_corrcoef(y, pred))
                                   if y.size else None)
    out['accuracy'] = float(accuracy_score(y, pred)) if y.size else None
    return out


def run_target(name, reps_wanted=None, verbose=True):
    """Every model on every representation, per scaffold-grouped fold."""
    from experiments._shared import get_model_factories

    smiles, y, theirs = MA.read_target(name)
    reps, groups, rep_record = MA.read_reps(name, smiles)
    if not reps:
        return {'target': name, 'refused': rep_record['reason'],
                'representations': rep_record, 'rows': []}
    if reps_wanted:
        missing = [r for r in reps_wanted if r not in reps]
        if missing:
            return {'target': name,
                    'refused': f'the npz has no {missing}; it has '
                               f'{sorted(reps)}',
                    'representations': rep_record, 'rows': []}
        reps = {r: reps[r] for r in reps_wanted}

    folds = MA.scaffold_folds(reps, y, groups)
    ctx = cr.CliffContext(y, smiles=smiles, reps=reps,
                          dataset_name=f'moleculeace_{name}')
    rows, errors = [], {}

    for fold_i, train_idx, val_idx in folds:
        y_tr = theirs[train_idx]
        y_va = theirs[val_idx]
        if len(set(y_tr.tolist())) < 2:
            errors[f'fold{fold_i}'] = (
                f'the training molecules of this fold are all '
                f'{"cliff" if y_tr.all() else "non-cliff"} molecules, so no '
                f'classifier can be fitted')
            continue

        # The baselines, on the same held-out molecules.
        rows.append({'target': name, 'fold': int(fold_i), 'arm': 'majority_class',
                     'what_it_is': 'every held-out molecule called '
                                   'not-a-cliff',
                     'model': None, 'representation': None,
                     **_scores(y_va, np.zeros(len(val_idx)))})
        for rule in BASELINE_RULES:
            try:
                mask, _info = cr.cliff_mask(ctx, rule=rule)
            except Exception as exc:
                errors[f'fold{fold_i}|{rule}'] = f'{type(exc).__name__}: {exc}'
                continue
            rows.append({
                'target': name, 'fold': int(fold_i), 'arm': f'rule:{rule}',
                'what_it_is': ('a published rule that does not read their '
                               'definition' if rule == 'n01'
                               else 'their own definition re-implemented, '
                                    'which is the ceiling and not a baseline'),
                'model': None,
                'representation': 'ecfp4 at 2048 bits, computed here',
                **_scores(y_va, np.asarray(mask, dtype=float)[val_idx])})

        for rep_name, X in reps.items():
            for model in MODELS:
                try:
                    factory = get_model_factories(
                        [model], 'classification', len(y)).get(model)
                    if factory is None:
                        errors[f'{model}'] = 'no factory for this model'
                        continue
                    m = factory()
                    m.fit(X[train_idx], y_tr)
                    if hasattr(m, 'predict_proba'):
                        p = np.asarray(m.predict_proba(X[val_idx]))
                        p = p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p
                    else:
                        p = np.asarray(m.predict(X[val_idx]), dtype=float)
                except Exception as exc:
                    errors[f'fold{fold_i}|{model}|{rep_name}'] = \
                        f'{type(exc).__name__}: {exc}'
                    continue
                rows.append({
                    'target': name, 'fold': int(fold_i),
                    'arm': 'trained_model',
                    'what_it_is': 'a classifier given one molecule\'s '
                                  'features and asked for a property of its '
                                  'neighbourhood',
                    'model': model, 'representation': rep_name,
                    **_scores(y_va, p)})
        if verbose:
            here = [r for r in rows if r['fold'] == fold_i]
            best = max((r for r in here
                        if r['arm'] == 'trained_model'
                        and r['matthews_correlation'] is not None),
                       key=lambda r: r['matthews_correlation'], default=None)
            rule = next((r for r in here if r['arm'] == 'rule:n01'), None)
            line = f'    fold {fold_i}: {len(val_idx)} held-out molecules'
            if best is not None:
                line += (f", best model {best['model']} on "
                         f"{best['representation']} Matthews "
                         f"{best['matthews_correlation']:.4f}")
            if rule is not None and rule['matthews_correlation'] is not None:
                line += (f", the published rule "
                         f"{rule['matthews_correlation']:.4f}")
            print(line, flush=True)

    return {'target': name, 'n_molecules': int(len(y)),
            'n_cliff_molecules_moleculeace': int(theirs.sum()),
            'pct_cliff_molecules_moleculeace': (
                100.0 * float(theirs.sum()) / len(y) if len(y) else 0.0),
            'representations': rep_record,
            'n_folds': len(folds), 'rows': rows, 'errors': errors}


def summarise(results):
    """One row per arm, with its per-target spread. No average across targets:
    the flag's share runs 8.8 to 54.0 per cent and an average over targets of
    different difficulty hides the ones where an arm failed."""
    acc = {}
    for res in results:
        for row in res['rows']:
            key = (row['arm'], row['model'], row['representation'])
            a = acc.setdefault(key, {'arm': row['arm'], 'model': row['model'],
                                     'representation': row['representation'],
                                     'what_it_is': row['what_it_is'],
                                     'mcc': [], 'auroc': [], 'n_folds': 0})
            a['n_folds'] += 1
            if row['matthews_correlation'] is not None:
                a['mcc'].append(row['matthews_correlation'])
            if row.get('auroc') is not None:
                a['auroc'].append(row['auroc'])
    out = []
    for a in acc.values():
        mcc, auc = sorted(a.pop('mcc')), sorted(a.pop('auroc'))
        out.append({**a,
                    'matthews_correlation_median': (float(np.median(mcc))
                                                    if mcc else None),
                    'matthews_correlation_lowest': mcc[0] if mcc else None,
                    'matthews_correlation_highest': mcc[-1] if mcc else None,
                    'auroc_median': float(np.median(auc)) if auc else None,
                    'auroc_lowest': auc[0] if auc else None,
                    'auroc_highest': auc[-1] if auc else None})
    out.sort(key=lambda r: -(r['matthews_correlation_median'] or -2))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', default=None)
    ap.add_argument('--reps', default=None,
                    help='comma-separated representation names from the npz; '
                         'default is every one it holds')
    ap.add_argument('--output', default=str(OUT_DEFAULT))
    ap.add_argument('--run-anyway', action='store_true',
                    help='run even though the setting has no value. For a '
                         'crash check only; the result is marked.')
    a = ap.parse_args(argv)

    stored = S.read()
    try:
        asked = S.one('train_cliff_model_on_moleculeace', stored)
    except (S.Unset, ValueError) as exc:
        if not a.run_anyway:
            print(f'NOT RUN: {exc}')
            return 0
        asked = None
        print(f'  --run-anyway: {exc}', flush=True)
    else:
        if not asked:
            print('NOT RUN: train_cliff_model_on_moleculeace is false in '
                  f'{S.path()}')
            return 0

    names = ([t.strip() for t in a.targets.split(',')] if a.targets
             else MA.targets())
    reps_wanted = ([r.strip() for r in a.reps.split(',')] if a.reps else None)
    t0 = time.time()
    results, failed = [], {}
    for i, name in enumerate(names):
        print(f'\n  [{i + 1}/{len(names)}] {name}', flush=True)
        try:
            results.append(run_target(name, reps_wanted=reps_wanted))
        except Exception as exc:
            failed[name] = f'{type(exc).__name__}: {exc}'
            print(f'    FAILED {failed[name]}', flush=True)

    rows = summarise(results)
    print('\n  one row is one arm. The Matthews correlation is the number to '
          'read; accuracy is whatever the flag\'s share is.\n')
    print(f"    {'arm':16s} {'model':10s} {'representation':18s} "
          f"{'MCC median':>11s} {'lowest':>8s} {'highest':>8s}")
    for r in rows[:25]:
        def _f(v):
            return '     n/a' if v is None else f'{v:8.4f}'
        print(f"    {r['arm'][:16]:16s} {str(r['model'] or '-')[:10]:10s} "
              f"{str(r['representation'] or '-')[:18]:18s} "
              f"{_f(r['matthews_correlation_median'])}"
              f"{_f(r['matthews_correlation_lowest'])}"
              f"{_f(r['matthews_correlation_highest'])}")

    out = {
        'what_it_is': ("a classifier trained to predict MoleculeACE's own "
                       'cliff flag, on scaffold-grouped folds, beside the '
                       'baselines it has to beat.'),
        'what_makes_the_task_odd': (
            'cliff_mol is a property of a molecule that is defined by its '
            'pairs, so a model given one molecule\'s features is asked for a '
            'property of its neighbourhood. A number from it is not a number '
            'about deciding whether a pair is a cliff.'),
        'their_split_was_not_read': (
            'the CSVs ship one. It is a published split and this lab does not '
            'use published or random splits; the folds are scaffold-grouped '
            'on our own Bemis-Murcko groups.'),
        'setting_said': asked,
        'n_targets': len(results), 'targets_that_failed': failed,
        'models': list(MODELS), 'baseline_rules': list(BASELINE_RULES),
        'settings': S.record(stored),
        'seconds': round(time.time() - t0, 1),
        'by_arm': rows, 'per_target': results,
    }
    if a.run_anyway and asked is None:
        out['run_without_the_setting'] = (
            'this was run with --run-anyway while '
            'train_cliff_model_on_moleculeace had no value. It is a crash '
            'check, not a result.')
    p = Path(a.output) / 'moleculeace_cliff_model.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f'\n  {len(results)} target(s), {out["seconds"]}s. written {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
