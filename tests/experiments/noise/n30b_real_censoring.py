#!/usr/bin/env python3
"""N30b: real censoring in the labels as delivered (ginkgo_pr_cho).

Paper: Noise. Approved by the author 2026-10-09 (HANDOFF_CENSORING_PRCHO.md;
THESIS_OUTLINE.md 7.7).

THE QUESTION. n30 clips labels itself and asks which treatment recovers them.
Here nothing is clipped by us. ginkgo_pr_cho (PR_CHO polyreactivity, 197
antibodies) has 58 labels at exactly 0 and none below. The PROPHET-Ab preprint
normalises the bead MFI "between 0 to 1 based on three reference antibodies";
it does not say values below the low reference are set to 0. So 0 is treated as
a LOWER limit, inferred from the labels. Do the treatments that worked on
injected censoring also work here, with the best single representation and
with the hybrid?

THE LOWER LIMIT. n30's code handles a pile-up at the TOP label only. Every
label move here runs n30's own functions on the NEGATED labels, so the floor
becomes a pile-up at the top, and the moved labels are negated back. n30 is not
edited, so its injected-censoring results cannot change. The residual spread
`sd` is the same either way; the out-of-fold prediction p is negated with the
labels.

THE TREATMENTS. Each is fitted on the training labels as delivered, and each
is scored against `as_delivered` (no treatment) on the same held-out
antibodies.
  as_delivered                 the labels as delivered
  constant_lift                every label at 0 moved down by the same amount
  per_molecule_lift            each label at 0 moved down by its own amount
  per_molecule_lift_shuffled/seed<k>
                               the same amounts dealt to other antibodies at 0
  truncated_normal_mean        each label at 0 set to the mean of a normal
                               truncated at 0, centred on p
  censored_em                  the truncated-normal step repeated with the
                               configuration's own model (n30, 3 rounds)
  winsor_c<c>                  every label pulled toward p beyond c * s
  drop_clipped                 the antibodies at 0 removed from training
  not_at_floor_control/seed<k> NEW. The same number of training labels NOT at 0,
                               chosen at random, each moved down by
                               constant_lift's amount. Tests whether knowing
                               which labels sit at the limit is what helps.
The formulas are n30's (`lift_columns`, `censored_em_column`, `winsorise`);
see n30's docstring for the two lift formulas.

NOT RUN: n30's xgboost_aft. It replaces xgboost, and xgboost is not one of this
dataset's two decided models (author, 2026-10-09).

THE CONFIGURATIONS. As n30: per outer fold and decided model, the 2 best single
representations and the 2 best general hybrids from that fold's search
(noise_helpers.fold_configurations). The hybrids are built from the labels as
delivered, which are the labels the search saw. No treatment reaches the
feature choice.

THE SCORES, per fold. The held-out labels are censored too, so three:
  spearman_all           Spearman on every held-out antibody (the dataset's
                         metric)
  spearman_above_floor   Spearman on the held-out antibodies above 0
  harrell_c              Harrell's concordance with antibodies at 0 treated as
                         censored: the share of pairs ranked in the right
                         order, over the pairs whose order is known. Two
                         antibodies at 0 have no known order and are skipped.
                         With one lower limit shared by every censored label,
                         that is every pair with two different labels.
Per-fold scores and their spread only. No significance tests.

QM9, WHERE THE TRUTH IS KNOWN (--inject-censoring, author 2026-10-09).
ginkgo_pr_cho tests the flag where the true labels are unknown. QM9's labels
are computed and clean, so here the truth is known. The training labels are
clipped at the TOP with n30's own injection (noise_helpers.inject_condition,
'censoring', the given share, draw 0), and the limit is read off the clipped
training labels with n30's limit_from_labels. No negation: n30's functions run
as they are. The same treatments run, plus:
  clean                        fitted on the training labels before the clip
The new control moves labels UP, by constant_lift's amount, on the same number
of training labels NOT at the limit. As in n30, the single representations are
built with the unclipped labels and the hybrids are rebuilt from the clipped
ones. Held-out labels are never clipped, so the score is the dataset's own
metric on every held-out molecule (RMSE on QM9).

Usage:
    python tests/experiments/noise/n30b_real_censoring.py --dataset ginkgo_pr_cho
    python tests/experiments/noise/n30b_real_censoring.py --dataset ginkgo_pr_cho --folds 0
    python tests/experiments/noise/n30b_real_censoring.py --dataset qm9 --inject-censoring 0.25
With --folds the result file name carries the folds, so runs of different folds
in separate shells never write the same file.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import convert_numpy, get_model_factories
from experiments.decided_grid import decided_models
from experiments.noise import noise_cache as nc
from experiments.noise.noise_helpers import (
    load_noise_data, inject_condition, noise_outer_folds, fold_configurations,
    fold_model_inputs, configuration_record, CONFIGURATION_SPACES,
)
from experiments.noise.n30_censored_likelihood import (
    limit_from_labels, rows_at_limit, residual_sd, lift_columns,
    censored_em_column, _own_model_oof, winsorise,
    WINSOR_C, SHUFFLE_SEEDS, OOF_SPLITS,
)

EXPERIMENT = 'n30b_real_censoring'
RESULT_LAYOUT = 'real_lower_floor_2026-10-09'
CONTROL_SEEDS = (0, 1, 2)
SCORES = ('spearman_all', 'spearman_above_floor', 'harrell_c')


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── The lower limit, through n30's upper-limit code on negated labels ───────

def floor_from_labels(y):
    """The lower limit a user would read off the labels, or None. n30's rule
    (at least 1% of the molecules and at least 2 sit on the smallest label),
    applied to the negated labels."""
    top = limit_from_labels(-np.asarray(y, dtype=float))
    return None if top is None else -top


def lower_label_columns(y, p, at, floor, sd, shuffle_seeds, seed_key):
    """n30's lift_columns on -y, negated back. Every label at the floor moves
    DOWN; no other label moves."""
    cols, info = lift_columns(-np.asarray(y, dtype=float),
                              -np.asarray(p, dtype=float), at, -floor, sd,
                              shuffle_seeds, seed_key)
    return {a: -c for a, c in cols.items()}, info


def lower_censored_em(factory, X, y, at, floor, groups, n_splits):
    """n30's censored_em_column on -y. The model is always fitted on the real
    labels: the negated column it is handed is negated back before the fit,
    and its predictions are negated on the way out."""
    col, info = censored_em_column(
        lambda yy: -_own_model_oof(factory, X, -np.asarray(yy), groups,
                                   n_splits),
        -np.asarray(y, dtype=float), at, -floor)
    return (None if col is None else -col), info


def not_at_floor_control(y, at, amount, seed, seed_key, direction=-1.0):
    """The same number of labels as sit at the limit, drawn at random from the
    labels NOT at the limit, each moved by `amount` (constant_lift's mean
    lift, label units): down for a lower limit (direction -1), up for an upper
    one (+1). Returns (column, how many moved)."""
    y = np.asarray(y, dtype=float)
    off = np.flatnonzero(~at)
    k = int(min(at.sum(), off.size))
    rng = np.random.default_rng([int(seed)] + [int(s) for s in seed_key])
    pick = rng.choice(off, size=k, replace=False)
    out = y.copy()
    out[pick] = y[pick] + float(direction) * float(amount)
    return out, k


# ── Scores ──────────────────────────────────────────────────────────────────

def harrell_c_lower(y, pred, floor):
    """Harrell's C with labels at `floor` censored below. A pair is comparable
    when its two labels differ (the larger is then above the floor, so its
    order is known). Concordant when the larger label has the larger
    prediction; a tied prediction counts one half. Returns (C, pairs)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    dy = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    comp = dy > 0
    if floor is not None:
        comp &= (y[:, None] > floor)
    n = int(comp.sum())
    if n == 0:
        return float('nan'), 0
    conc = np.sum((dp > 0) & comp) + 0.5 * np.sum((dp == 0) & comp)
    return float(conc / n), n


def three_scores(y_va, pred, floor):
    y_va = np.asarray(y_va, dtype=float)
    pred = np.asarray(pred, dtype=float).ravel()
    out = {}
    out['spearman_all'] = float(spearmanr(y_va, pred).correlation)
    above = y_va > floor if floor is not None else np.ones(len(y_va), bool)
    out['n_held_out_above_floor'] = int(above.sum())
    out['spearman_above_floor'] = (
        float(spearmanr(y_va[above], pred[above]).correlation)
        if above.sum() >= 3 else float('nan'))
    out['harrell_c'], out['harrell_c_pairs'] = harrell_c_lower(
        y_va, pred, floor)
    return out


def fit_predict(factory, X_tr, y_tr, X_va, keep=None):
    m = factory()
    if keep is None:
        m.fit(X_tr, y_tr)
    else:
        m.fit(X_tr[keep], np.asarray(y_tr)[keep])
    return np.asarray(m.predict(X_va), dtype=float).ravel()


# ── One fold ────────────────────────────────────────────────────────────────

def run_fold(data, dataset_name, fold_i, train_idx, val_idx, n_folds, models,
             factories, metric, winsor_c, shuffle_seeds, oof_splits,
             inject_level=None):
    """One outer fold. inject_level None: the labels as delivered, lower
    limit read off them. A number: n30's censoring injection at that share
    (upper limit), with a `clean` fit as well."""
    labels = np.asarray(data['labels'], dtype=float)
    groups = data.get('groups')
    g_tr = None if groups is None else np.asarray(groups)[train_idx]
    y_clean_tr, y_va = labels[train_idx], labels[val_idx]
    metric_name, metric_fn, _lower = metric
    injected = inject_level is not None

    if injected:
        y_tr, cmask, _prov = inject_condition(
            data, dataset_name, 'censoring', float(inject_level), 0,
            train_idx=train_idx)
        y_tr = np.asarray(y_tr, dtype=float)
        limit = limit_from_labels(y_tr)
        at = rows_at_limit(y_tr, limit)
        side, direction = 'upper', 1.0
    else:
        y_tr = y_clean_tr
        limit = floor_from_labels(y_tr)
        at = rows_at_limit(-y_tr, None if limit is None else -limit)
        side, direction = 'lower', -1.0
    frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
            'side': side, 'limit': limit,
            'limit_read_from': 'the training labels (n30 limit_from_labels'
                               + ('' if injected else ', on the negated labels')
                               + ')',
            'training_rows_at_limit': int(at.sum()),
            'models': {}, 'scores': {}, 'errors': {}}
    if injected:
        frec['injected_share'] = float(inject_level)
        frec['training_rows_the_clip_moved'] = int(np.sum(cmask))
    else:
        frec['held_out_rows_at_limit'] = (
            int(np.sum(y_va <= limit + 1e-12)) if limit is not None else 0)
    _log(f"fold {fold_i}: {len(train_idx)} training, {len(val_idx)} held out; "
         f"{side} limit {limit}, {frec['training_rows_at_limit']} training "
         f"rows at it")
    if limit is None:
        frec['errors']['limit'] = 'no limit in the training labels'
        return frec

    t = time.time()
    entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                  train_idx=train_idx, n_folds=n_folds)
    _log(f"fold {fold_i}: per-fold searches read in {time.time() - t:.0f}s")

    base_sp = tuple(s for s in CONFIGURATION_SPACES if s.startswith('base'))
    hyb_sp = tuple(s for s in CONFIGURATION_SPACES if s not in base_sp)
    for m in models:
        entry = entries.get(m)
        # As n30: single representations with the unclipped labels, hybrids
        # rebuilt from the labels the treatments start from.
        ins, unav, names = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=y_clean_tr, factory=factories[m],
            metric=metric, spaces=base_sp)
        ins_h, unav_h, names_h = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=y_tr, factory=factories[m],
            metric=metric, spaces=hyb_sp)
        ins, unav, names = ({**ins, **ins_h}, {**unav, **unav_h},
                            {**names, **names_h})
        rec = configuration_record(entry, names)
        rec['unavailable'] = dict(unav)
        frec['models'][m] = rec
        _log(f"fold {fold_i} {m}: base {rec.get('base')}, hybrids "
             f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        for space in CONFIGURATION_SPACES:
            pair = ins.get(space)
            key = f'{space}/{m}'
            if pair is None:
                frec['errors'][key] = unav.get(space) or 'not built'
                continue
            X_tr, X_va = pair
            t = time.time()
            p = nc.cached_oof_predict(dataset_name, X_tr, y_tr, g_tr,
                                      n_splits=oof_splits)
            sd = residual_sd(y_tr, p, at)
            seed_key = (fold_i, CONFIGURATION_SPACES.index(space),
                        models.index(m))
            cols = {'as_delivered': (y_tr, None)}
            if injected:
                cols['clean'] = (y_clean_tr, None)
            info = {'residual_sd_rows_not_at_limit': sd}
            if np.isfinite(sd) and at.any():
                if injected:
                    lc, linfo = lift_columns(y_tr, p, at, limit, sd,
                                             shuffle_seeds, seed_key)
                else:
                    lc, linfo = lower_label_columns(y_tr, p, at, limit, sd,
                                                    shuffle_seeds, seed_key)
                info.update(linfo)
                for a, c in lc.items():
                    cols[a] = (c, None)
                cols['drop_clipped'] = (y_tr, ~at)
                if g_tr is not None:
                    try:
                        if injected:
                            _f = factories[m]
                            em, einfo = censored_em_column(
                                lambda yy: _own_model_oof(
                                    _f, X_tr, yy, g_tr, oof_splits),
                                y_tr, at, limit)
                        else:
                            em, einfo = lower_censored_em(
                                factories[m], X_tr, y_tr, at, limit, g_tr,
                                oof_splits)
                        info.update(einfo)
                        if em is not None:
                            cols['censored_em'] = (em, None)
                    except Exception as exc:
                        frec['errors'][f'{key}/censored_em'] = (
                            f'{type(exc).__name__}: {exc}')
                amount = linfo['mean_lift_label_units']
                for s in CONTROL_SEEDS:
                    c, k = not_at_floor_control(y_tr, at, amount, s, seed_key,
                                                direction=direction)
                    cols[f'not_at_floor_control/seed{s}'] = (c, None)
                info['not_at_floor_control_labels_moved'] = k
            for c in winsor_c:
                col, s_, moved = winsorise(y_tr, p, c)
                cols[f'winsor_c{c:g}'] = (col, None)
                info[f'winsor_c{c:g}'] = {'threshold_label_units': c * s_,
                                          'labels_moved': moved}

            sc = {}
            for arm, (col, keep) in cols.items():
                try:
                    pred = fit_predict(factories[m], X_tr, col, X_va, keep)
                    sc[arm] = ({metric_name: float(metric_fn(y_va, pred))}
                               if injected else
                               three_scores(y_va, pred, limit))
                except Exception as exc:
                    frec['errors'][f'{key}/{arm}'] = (
                        f'{type(exc).__name__}: {exc}')
            frec['scores'][key] = {'info': info, 'per_arm': sc}
            first = metric_name if injected else 'spearman_all'
            _log(f"fold {fold_i} {key}: {len(sc)} treatments in "
                 f"{time.time() - t:.0f}s; as delivered {first} "
                 f"{sc.get('as_delivered', {}).get(first, float('nan')):.4f}")
    return frec


# ── Summary: per-fold scores and their spread, no tests ─────────────────────

def summarise(folds, scores, lower_is_better):
    """Per configuration, treatment and score: the per-fold scores, the
    per-fold difference from as_delivered, how many folds the treatment was
    better on (lower is better for the scores in lower_is_better), the mean
    difference and its fold spread."""
    out = {}
    for f, frec in sorted(folds.items(), key=lambda kv: int(kv[0])):
        for key, block in (frec.get('scores') or {}).items():
            base = block['per_arm'].get('as_delivered') or {}
            for arm, s in block['per_arm'].items():
                row = out.setdefault(key, {}).setdefault(arm, {
                    sc: {'per_fold': {}, 'minus_as_delivered_per_fold': {}}
                    for sc in scores})
                for sc in scores:
                    v = s.get(sc)
                    row[sc]['per_fold'][f] = v
                    b = base.get(sc)
                    row[sc]['minus_as_delivered_per_fold'][f] = (
                        None if v is None or b is None else v - b)
    for key, arms in out.items():
        for arm, row in arms.items():
            for sc, r in row.items():
                d = [v for v in r['minus_as_delivered_per_fold'].values()
                     if v is not None and np.isfinite(v)]
                better = ((lambda v: v < 0) if sc in lower_is_better
                          else (lambda v: v > 0))
                r['lower_is_better'] = sc in lower_is_better
                r['folds_better_than_as_delivered'] = int(
                    sum(better(v) for v in d))
                r['folds_compared'] = len(d)
                r['mean_difference'] = float(np.mean(d)) if d else None
                r['fold_sd_of_difference'] = (float(np.std(d, ddof=1))
                                              if len(d) > 1 else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dataset', default='ginkgo_pr_cho')
    ap.add_argument('--folds', default=None,
                    help='comma-separated outer folds to run (default all)')
    ap.add_argument('--output', default='tests/results')
    ap.add_argument('--inject-censoring', type=float, default=None,
                    help="clip this share of training labels at the top with "
                         "n30's injection (QM9, where the truth is known)")
    args = ap.parse_args()
    inject = args.inject_censoring
    ds = args.dataset

    from kirby.find_best_hybrid import (
        DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS,
        DEFAULT_N_FOLDS)
    from experiments._shared import FULL_BUDGETS, decided_config
    from experiments.noise.noise_helpers import INTERNAL_EVALUATORS
    from experiments._shared import get_domain_metric
    # The config run_all builds (noise_helpers.run_all), so the per-fold
    # searches are the same cached searches the noise runs read.
    config = decided_config({
        'n_splits': DEFAULT_N_FOLDS,
        'allocations': DEFAULT_ALLOCATION_METHODS,
        'importance_methods': DEFAULT_IMPORTANCE_METHODS,
        'internal_evaluators': INTERNAL_EVALUATORS,
        'budgets': FULL_BUDGETS,
    }, ds)
    if args.folds:
        config['only_folds'] = [int(x) for x in args.folds.split(',')]

    t0 = time.time()
    data, _ = load_noise_data(ds)
    _log(f"loaded {ds} in {time.time() - t0:.0f}s")
    if data['task_type'] != 'regression':
        raise SystemExit(f'{ds} is {data["task_type"]}; censoring is '
                         f'regression only')
    labels = np.asarray(data['labels'], dtype=float)
    metric = get_domain_metric(data)
    models = [m for m in decided_models(ds, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type='regression',
                                    n_samples=len(labels))
    models = [m for m in models if m in factories]

    out_dir = os.path.join(args.output, EXPERIMENT)
    os.makedirs(out_dir, exist_ok=True)
    name = ds + (f'__injected{inject:g}' if inject is not None else '')
    if args.folds:
        name += '__folds' + args.folds.replace(',', '_')
    out_path = os.path.join(out_dir, f'{name}.json')
    scores = [metric[0]] if inject is not None else list(SCORES)
    lower = [metric[0]] if (inject is not None and metric[2]) else []
    res = None
    if os.path.exists(out_path):
        with open(out_path) as fh:
            res = json.load(fh)
        if res.get('result_layout') != RESULT_LAYOUT:
            res = None
        else:
            _log(f"resuming {out_path}: folds done {sorted(res['folds'])}")
    if res is None:
        floor_all = None if inject is not None else floor_from_labels(labels)
        res = {
            'mode': ('injected censoring at the top, share '
                     f'{inject:g}, draw 0 (n30 injection)'
                     if inject is not None else
                     'labels as delivered, lower limit read off them'),
            'injected_share': inject,
            'experiment': EXPERIMENT, 'dataset': ds,
            'result_layout': RESULT_LAYOUT,
            'metric': metric[0],
            'models': models,
            'configurations_per_model': list(CONFIGURATION_SPACES),
            'floor_whole_dataset': floor_all,
            'rows_at_floor_whole_dataset': int(
                np.sum(labels <= floor_all + 1e-12)) if floor_all is not None
            else 0,
            'n_rows': int(len(labels)),
            'floor_note': (None if inject is not None else
                'Treated as a lower limit by the author (2026-10-09). On '
                'ginkgo_pr_cho the source normalises to reference antibodies '
                'and does not state that values below the low reference are '
                'set to 0.'),
            'not_run': {'xgboost_aft': 'xgboost is not one of the decided '
                                       'models (author, 2026-10-09)'},
            'winsor_c': list(WINSOR_C), 'shuffle_seeds': list(SHUFFLE_SEEDS),
            'control_seeds': list(CONTROL_SEEDS), 'oof_splits': OOF_SPLITS,
            'scores': scores,
            'folds': {},
        }

    folds, n_folds = noise_outer_folds(data, config)
    res['n_folds'] = int(n_folds)
    for fold_i, tr, va in folds:
        if str(fold_i) in res['folds']:
            _log(f"fold {fold_i}: on disk, skipped")
            continue
        t = time.time()
        res['folds'][str(fold_i)] = run_fold(
            data, ds, fold_i, tr, va, n_folds, models, factories, metric,
            list(WINSOR_C), list(SHUFFLE_SEEDS), OOF_SPLITS,
            inject_level=inject)
        res['folds'][str(fold_i)]['seconds'] = time.time() - t
        res['summary'] = summarise(res['folds'], scores, lower)
        with open(out_path, 'w') as fh:
            json.dump(convert_numpy(res), fh, indent=1)
        _log(f"fold {fold_i} done in {time.time() - t:.0f}s; written "
             f"{out_path}")
    _log(f"all folds done in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
