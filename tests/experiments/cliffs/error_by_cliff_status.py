#!/usr/bin/env python3
"""Held-out prediction error on cliff molecules against non-cliff molecules,
under every definition.

HER SENTENCE THIS ANSWERS. "S04 - do models predict better/worse for ACs
compared to non-ACs? (held-out prediction error on cliff molecules against
non-cliff molecules)". Her outline then says "Make this a separate set of AC
experiments", which is why it is here and not a change to s04.

ONE ROW IS one definition, one downstream model, one feature space and one
fold: the dataset's own metric over the held-out cliff molecules, over the
held-out non-cliff molecules, and over all of them, with how many molecules
are in each.

THE MODELS ARE FITTED ONCE PER FOLD AND SPACE, not once per definition. A
definition changes which held-out molecules are called cliffs; it does not
change the fit or the predictions. So every definition is scored off the same
held-out predictions, and two definitions' numbers differ only by the split.

HOW A HELD-OUT MOLECULE BECOMES A CLIFF MOLECULE, which is her open DECIDE
item (THESIS_OUTLINE.md 2274). Both rules are runnable and the setting
`held_out_cliff_rule` chooses; every row says which it used.

  same_as_training                a held-out molecule is a cliff molecule when
                                  it is similar to a training molecule whose
                                  label differs by more than the gap. This is
                                  the training rule applied across the fold
                                  boundary.
  two_similar_training_molecules  a held-out molecule is a cliff molecule when
                                  two of the training molecules similar to it
                                  differ from each other by more than the gap.
                                  It reads nothing about the held-out label.
                                  This is the rule her s04 and s12 runs used,
                                  and it is looser: her own record has it
                                  reaching about 40 per cent of held-out
                                  molecules where the training rule aims at 15.

⛔ BOTH CUTOFFS COME FROM THE TRAINING MOLECULES. A share-of-pairs criterion
cut on the held-out pairs would be a different criterion, and a gap solved to a
share of the held-out molecules would read their labels. So the definition is
computed on the fold's training molecules, the two numbers it reached are kept,
and those numbers are what the cross-boundary pairs are thresholded at.

Usage, as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments error_by_cliff_status
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
from experiments.noise import cliff_rules as cr                     # noqa: E402
from experiments.noise.n07_remediation import _predict_for_metric    # noqa: E402
from experiments.cliffs import cliff_sets as P                       # noqa: E402
from experiments.cliffs import definitions as D                      # noqa: E402
from experiments.cliffs import settings as S                         # noqa: E402

EXP_ID = 'error_by_cliff_status'

#: The feature spaces a model is fitted on, from the fold's own search: its
#: two best single representations and its first general hybrid. The same three
#: the noise suite fits (noise_helpers.CONFIGURATION_SPACES less the second
#: hybrid, which the noise decision of 2026-10-30 does not fit either).
SPACES = ('base_1', 'base_2', 'hybrid')

#: How few molecules on one side of the split is too few to score. Her thesis
#: records the same number: "only two folds on each hold the 10 needed for a
#: score" (thesis.tex:3314).
MIN_MOLECULES_TO_SCORE = 10


def held_out_cliff_flag(ctx_full, train_idx, val_idx, similar_full, gap_cut,
                        rule):
    """Which held-out molecules are cliff molecules, under one rule.

    `similar_full` is the similar mask over every molecule, thresholded at the
    cut the TRAINING molecules reached. `gap_cut` is the gap the definition
    reached on the training molecules, in the label's own units.
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    y = ctx_full.labels
    block = np.asarray(similar_full)[np.ix_(val_idx, train_idx)]

    if rule == 'same_as_training':
        gaps = np.abs(y[val_idx][:, None] - y[train_idx][None, :])
        if ctx_full.is_clf:
            gaps = (y[val_idx][:, None] != y[train_idx][None, :]).astype(float)
        return (block & (gaps > gap_cut)).any(axis=1)

    if rule == 'two_similar_training_molecules':
        # Among the training molecules similar to this held-out one, do two of
        # them differ by more than the gap. The widest difference in a set is
        # between its largest and smallest member, so the largest and smallest
        # similar training label answer it in one pass.
        y_tr = y[train_idx]
        big = np.where(block, y_tr[None, :], -np.inf).max(axis=1)
        small = np.where(block, y_tr[None, :], np.inf).min(axis=1)
        has_two = block.sum(axis=1) >= 2
        if ctx_full.is_clf:
            # Two similar training molecules of different classes.
            any_one = np.where(block, y_tr[None, :], np.nan)
            differ = np.nanmax(any_one, axis=1) != np.nanmin(any_one, axis=1)
            return has_two & np.nan_to_num(differ, nan=False).astype(bool)
        return has_two & ((big - small) > gap_cut)

    raise ValueError(f'held_out_cliff_rule {rule!r} is not one of '
                     f'{S.HELD_OUT_CLIFF_RULES}')


def _similar_full_for(d, ctx_full, counts):
    """The similar mask over every molecule, cut where the training molecules
    cut it, or None where a definition cannot be carried across the boundary.
    """
    sinfo = counts.get('similarity_info') or {}
    fn = d.similarity
    rep = getattr(fn, 'representation', None)
    coef = getattr(fn, 'coefficient', None)
    if rep is not None and coef is not None:
        S_full = cr.coefficient_matrix(ctx_full, rep, coef)
        cut = sinfo.get('cutoff', sinfo.get('cutoff_reached'))
        if cut is None:
            return None, 'the training cut recorded no similarity value'
        m = np.asarray(S_full) >= float(cut)
        np.fill_diagonal(m, False)
        return m, None
    # A fixed published criterion: its cutoff is the same on any molecules, so
    # running it on the whole set reproduces the training cut.
    if sinfo.get('percentile') is not None:
        return None, ('a share-of-pairs criterion on a matrix this file '
                      'cannot re-threshold at the training cut')
    try:
        m, _ = (cr.SIMILARITY_CRITERIA[fn](ctx_full) if isinstance(fn, str)
                else fn(ctx_full))
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    m = np.asarray(m, dtype=bool).copy()
    np.fill_diagonal(m, False)
    return m, None


def _score_split(y_true, y_pred, flag, metric_fn, metric_name):
    """The metric over the flagged molecules, the rest, and all of them."""
    out = {}
    for label, sel in (('cliff', flag), ('non_cliff', ~flag),
                       ('all', np.ones(len(flag), dtype=bool))):
        n = int(sel.sum())
        out[f'n_{label}_molecules'] = n
        if n < MIN_MOLECULES_TO_SCORE:
            out[f'score_{label}'] = None
            out[f'score_{label}_not_scored'] = (
                f'{n} molecules, under the {MIN_MOLECULES_TO_SCORE} a score '
                f'needs')
            continue
        try:
            out[f'score_{label}'] = float(metric_fn(y_true[sel], y_pred[sel]))
        except Exception as exc:
            out[f'score_{label}'] = None
            out[f'score_{label}_not_scored'] = f'{type(exc).__name__}: {exc}'
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    print('\n' + '=' * 70)
    print('ERROR BY CLIFF STATUS: held-out error on cliff molecules against '
          'non-cliff molecules')
    print('=' * 70)

    from experiments.noise.noise_helpers import (fold_configurations,
                                                 fold_model_inputs)
    from experiments.decided_grid import decided_models

    stored = S.read()
    labels = np.asarray(data['labels']).ravel()
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    is_clf = data.get('task_type') == 'classification'
    rules = S.values('held_out_cliff_rule', stored)
    scopes = S.values('definition_scope', stored)
    folds, n_folds = P.folds_of(data, config)
    try:
        models = [m for m in decided_models(dataset_name, 'general')
                  if m != 'knn']
    except Exception as exc:
        why = (f'the decided grid has no entry for {dataset_name}, so its two '
               f'downstream models are not recorded: '
               f'{type(exc).__name__}: {exc}')
        print(f'  REFUSED: {why}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'refused': why}, 'rows': [],
               'result_status': 'failed'}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    jsonl = Path(output_dir) / EXP_ID / f'{dataset_name}__rows.jsonl'
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if jsonl.exists():
        jsonl.unlink()

    n_rows, errors, fold_records = 0, {}, []
    t_all = time.time()

    with open(jsonl, 'a') as sink:
        for fold_i, train_idx, val_idx in folds:
            t0 = time.time()
            print(f'\n  fold {fold_i} of {n_folds}: {len(train_idx)} training, '
                  f'{len(val_idx)} held-out molecules', flush=True)

            # ── the fits, once per model and space ───────────────────────
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
            predictions, fit_failed = {}, {}
            for model in models:
                factories = get_model_factories(
                    [model], data.get('task_type') or 'regression', len(labels))
                factory = factories.get(model)
                inputs, unavailable, what = fold_model_inputs(
                    data, dataset_name, model, entries.get(model),
                    train_idx=train_idx, val_idx=val_idx,
                    y_train=labels[train_idx], factory=factory,
                    metric=metric_name, spaces=SPACES)
                for space, why in (unavailable or {}).items():
                    fit_failed[f'{model}|{space}'] = why
                for space, (X_tr, X_va) in inputs.items():
                    try:
                        m = factory()
                        m.fit(X_tr, labels[train_idx])
                        pred = _predict_for_metric(m, X_va, metric_name)
                        predictions[(model, space)] = (
                            np.asarray(pred).ravel(),
                            (what or {}).get(space))
                    except Exception as exc:
                        fit_failed[f'{model}|{space}'] = \
                            f'{type(exc).__name__}: {exc}'
            print(f'    {len(predictions)} fits over {len(models)} models and '
                  f'{len(SPACES)} spaces, {len(fit_failed)} did not fit',
                  flush=True)
            if not predictions:
                errors[f'fold{fold_i}'] = ('no model fitted on this fold: '
                                           + json.dumps(fit_failed)[:400])
                continue

            # ── the definitions, on the training molecules ───────────────
            for scope in scopes:
                ctx, whole, record = D.context_for_fold(
                    data, dataset_name, train_idx=train_idx, fold_i=fold_i,
                    n_folds=n_folds, stored=stored, scope=scope, verbose=False)
                defs = D.expand(ctx, stored=stored, scope=scope)
                got, errs, _cost = D.masks(ctx, defs, keep_masks=False)
                for key, why in errs.items():
                    errors[f'{scope}|{fold_i}|{key}'] = why
                carried, not_carried = 0, {}
                # The whole-molecule matrices are freed when the group that
                # reads them is finished, not after each definition: every
                # cutoff of one coefficient on one representation reads the
                # same matrix, and freeing between them rebuilds it each time.
                defs_in_order = sorted(
                    (d for d in defs if d.key in got),
                    key=lambda d: D.similarity_group(d))
                last_group = None
                for d in defs_in_order:
                    group = D.similarity_group(d)
                    if last_group is not None and group != last_group:
                        whole.forget_similarities()
                    last_group = group
                    _m, counts = got[d.key]
                    gap_cut = (counts['gap_info'] or {}).get(
                        'cutoff_label_units',
                        (counts['gap_info'] or {}).get('cutoff'))
                    if gap_cut is None:
                        not_carried[d.key] = 'the gap recorded no cutoff'
                        continue
                    similar_full, why = _similar_full_for(d, whole, counts)
                    if similar_full is None:
                        not_carried[d.key] = why
                        continue
                    carried += 1
                    for rule in rules:
                        flag = held_out_cliff_flag(
                            whole, train_idx, val_idx, similar_full,
                            float(gap_cut), rule)
                        for (model, space), (pred, what) in predictions.items():
                            row = {
                                'dataset': dataset_name, 'fold': int(fold_i),
                                'definition': d.key,
                                'definition_scope': scope,
                                'held_out_cliff_rule': rule,
                                'model': model, 'feature_space': space,
                                'what_the_space_is': what,
                                'metric': metric_name,
                                'lower_is_better': bool(lower_is_better),
                                'gap_cutoff_label_units': float(gap_cut),
                                'n_held_out_molecules': int(len(val_idx)),
                                'similarity_used_labels': d.record.get(
                                    'similarity_used_labels'),
                                'coefficient': d.record.get('coefficient'),
                                'representation': d.record.get(
                                    'representation'),
                                **_score_split(labels[val_idx], pred, flag,
                                               metric_fn, metric_name),
                            }
                            sink.write(json.dumps(row, default=float) + '\n')
                            n_rows += 1
                whole.forget_similarities()
                sink.flush()
                print(f'    scope {scope}: {carried} of {len(defs)} '
                      f'definitions carried across the fold boundary, '
                      f'{len(not_carried)} could not be', flush=True)
                fold_records.append({
                    'fold': int(fold_i), 'definition_scope': scope,
                    'n_definitions': len(defs),
                    'n_definitions_carried_to_held_out_molecules': carried,
                    'definitions_not_carried': dict(
                        list(not_carried.items())[:20]),
                    'n_definitions_not_carried': len(not_carried),
                    'fits': sorted(f'{m}|{s}' for m, s in predictions),
                    'fits_that_failed': fit_failed,
                    'hybrids_refused': record['hybrids_refused'],
                    'seconds': round(time.time() - t0, 1),
                })

    meta = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'metric': metric_name, 'lower_is_better': lower_is_better,
        'task_type': data.get('task_type'),
        'label_is_class_mismatch': is_clf,
        'models': models, 'feature_spaces': list(SPACES),
        'n_outer_folds': int(n_folds),
        'held_out_cliff_rules_run': list(rules),
        'definition_scopes_run': list(scopes),
        'minimum_molecules_to_score': MIN_MOLECULES_TO_SCORE,
        'per_fold': fold_records,
        'n_rows': n_rows, 'rows_file': str(jsonl),
        'seconds': round(time.time() - t_all, 1),
        'settings': S.record(stored),
    }
    result = {
        'meta': meta, 'errors': errors,
        'what_one_row_is': (
            "one definition, one downstream model, one feature space and one "
            "fold: the dataset's own metric over the held-out cliff "
            "molecules, over the held-out non-cliff molecules, and over all "
            "of them."),
        'the_fits_do_not_depend_on_the_definition': (
            'the models are fitted once per fold and space. A definition '
            'changes which held-out molecules are called cliffs and nothing '
            'else, so two definitions differ only by the split.'),
        'where_every_row_is': str(jsonl),
    }
    save_experiment_results(result, output_dir, EXP_ID, dataset_name,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better,
                            fingerprint=input_fingerprint(data, config))
    print(f'\n  {n_rows} rows written to {jsonl}', flush=True)
    return result


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
