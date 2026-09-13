#!/usr/bin/env python3
# =============================================================================
# ⛔ DO NOT COPY THIS DESIGN. KEPT ON DISK AS THE EXAMPLE OF WHAT NOT TO DO.
#
# Retired 2026-08-27 by the owner. Replaced by
# tests/experiments/framework/f27_objective_and_budget.py.
#
# WHAT IS WRONG WITH IT, and it is the whole file, not a detail:
# Below this banner, the file contains ZERO matches for each of:
#   create_hybrid   allocate_   sim_   nested_cv
# Its only touch on the engine is `from find_best_hybrid_molecular import
# load_dataset, assign_scaffold_groups`, which loads rows and scaffold groups.
# Nothing in it builds a hybrid or splits a budget.
# It scores feature subsets written by hand inside the experiment, over five
# fingerprints. A hybrid the framework builds is a different object: real
# allocation methods, the dataset's own representation pool, the search choosing
# budget and model, and an honest outer fold. Nothing measured here transfers
# to one.
#
# A FRAMEWORK EXPERIMENT DECIDES HOW KIRBy BUILDS THE BEST HYBRID FOR EACH
# DATASET x MODEL x BUDGET. One that never builds a hybrid cannot do that,
# however careful its statistics are.
#
# Its result files were deleted. Any number quoted from it anywhere — including
# in SIMILARITY_METRICS_ROADMAP.md, STRUCTURAL_OBJECTIVE_PROMPT.md and the SAC
# decision block in find_best_hybrid.py — is a number about hand-written feature
# subsets, not about a hybrid.
# =============================================================================
"""F21: what should decide how many features each representation gets?

STANDALONE AND CACHE-FREE. Builds representations from SMILES, does its own
splitting, and never reads or writes a search pickle, a baseline pickle, a
hybrid-eval pickle or a result database.

---------------------------------------------------------------------------
THE QUESTION

Under a geometry objective the engine scored every representation with sac and
nothing else, and those scores set each representation's share of the budget.
Asking for readacross did not change it: sac and readacross were treated
identically, because the allocators were not given scaffold groups and
readacross needs them.

OUT OF DATE AS OF 2026-08-26. sac was cut as an objective and the budget split
now uses readacross, with scaffold groups sliced and passed down. The question
below is still open -- it asks which of six rules should size the share, and
readacross is only one of them -- but the premise "the budget is currently
shared out using the weaker signal" no longer describes the engine.

sac is the smaller number on all five molecular datasets measured on 2026-08-26
(0.0423 adme_rlm to 0.1824 herg_ki). readacross on the same picks ran 0.3594 to
0.6955. So the budget is currently shared out using the weaker signal.

This measures whether that matters, without changing the engine. Six ways of
sharing the budget out, same folds, same representations, same budgets:

  equal        every representation gets budget/n_reps        (neutral reference)
  sac          share proportional to each representation's sac
  readacross   share proportional to each representation's readacross
  predictive   share proportional to a fitted model's held-out Spearman
  rank_50      rank representations under sac and under readacross, average
               the two positions, share proportional to the average
  rank_75ra    same, weighted 0.75 readacross / 0.25 sac

Ranking before blending is the point of rank_50 and rank_75ra. sac and
readacross are both Spearman rho but occupy different parts of the range, so a
weighted sum of the raw numbers gives readacross most of the influence. A rank
runs 0 to 1 for both.

WHAT A NEGATIVE RESULT LOOKS LIKE, stated up front: if every rule lands within
the fold-to-fold standard deviation of `equal`, then how the budget is shared
out does not matter and the engine should not be changed. The summary prints
each rule's gap to `equal` in units of that standard deviation.

---------------------------------------------------------------------------
Usage:
    python f21_budget_split_signal.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    get_config, create_experiment_parser, save_experiment_results,
    cv_split, get_domain_metric,
)
# f18 sits beside this file. The bare name resolves only when python is run
# from inside tests/experiments/framework; through the package it is not on
# the path, so importing this module raised ModuleNotFoundError.
from experiments.framework.f18_distance_metric_sweep import (
    build_representations, DEFAULT_REPS)
from experiments.framework.f19_objective_scoring import (
    y_predictive, y_sac, y_readacross, SEED, PRED_MODEL, PRED_MODELS,
)

BUDGETS = (100, 250, 500)
RULES = ('equal', 'sac', 'readacross', 'predictive', 'rank_50', 'rank_75ra')
YARDSTICKS = ('predictive', 'sac', 'readacross')


def score_each_representation(reps, a_idx, b_idx, labels, groups):
    """Score every representation on its own, three ways, on a held-out slice.

    a_idx fits the scaler and the model; b_idx is scored. Returns
    {rule_signal: {rep_name: score}} for sac, readacross and predictive.
    """
    out = {'sac': {}, 'readacross': {}, 'predictive': {}}
    for name, X in sorted(reps.items()):
        sc = StandardScaler().fit(X[a_idx])
        Xa = np.nan_to_num(sc.transform(X[a_idx]))
        Xb = np.nan_to_num(sc.transform(X[b_idx]))
        out['sac'][name] = y_sac(Xa, labels[a_idx], Xb, labels[b_idx])
        out['readacross'][name] = y_readacross(
            Xa, labels[a_idx], Xb, labels[b_idx], groups[b_idx])
        out['predictive'][name] = y_predictive(
            Xa, labels[a_idx], Xb, labels[b_idx])
    return out


def shares_from_scores(scores, names):
    """Turn per-representation scores into non-negative shares summing to 1.

    Negative and non-finite scores become 0. If every score collapses to 0 the
    split falls back to equal, which is stated in the returned flag so a
    degenerate fold is visible rather than silently looking like a real split.
    """
    v = np.array([scores.get(n, np.nan) for n in names], dtype=np.float64)
    v = np.where(np.isfinite(v), v, 0.0)
    v = np.maximum(v, 0.0)
    if v.sum() <= 0:
        return np.full(len(names), 1.0 / len(names)), True
    return v / v.sum(), False


def shares_from_ranks(scores_a, scores_b, names, w_b):
    """Rank under each signal, blend the two positions, normalise to shares.

    w_b is the weight on scores_b. A rank runs 0 to 1 under both signals, so
    w_b really is the share of the influence -- which is not true of a weighted
    sum of the raw numbers.
    """
    def _r(sc):
        v = np.array([sc.get(n, np.nan) for n in names], dtype=np.float64)
        ok = np.isfinite(v)
        r = np.zeros(len(v))
        if ok.sum() >= 2:
            r[ok] = rankdata(v[ok]) / ok.sum()
        elif ok.sum() == 1:
            r[ok] = 1.0
        return r
    blended = (1.0 - w_b) * _r(scores_a) + w_b * _r(scores_b)
    if blended.sum() <= 0:
        return np.full(len(names), 1.0 / len(names)), True
    return blended / blended.sum(), False


def build_hybrid(reps, names, shares, budget, fit_idx, importance):
    """Take each representation's share of the budget, best features first.

    `importance` is a per-representation vector fitted on fit_idx only. Columns
    are ordered by it and the top n_i are kept, where n_i is that
    representation's share of the budget. Returns the column index list per
    representation so the same columns can be applied to any row slice.
    """
    picked = {}
    for name, s in zip(names, shares):
        n_i = int(round(s * budget))
        w = reps[name].shape[1]
        n_i = max(0, min(n_i, w))
        if n_i == 0:
            continue
        imp = importance[name]
        picked[name] = np.argsort(imp)[::-1][:n_i]
    return picked


def apply_pick(reps, picked, idx):
    parts = [reps[n][np.ix_(idx, c)] for n, c in sorted(picked.items())
             if len(c)]
    if not parts:
        return None
    return np.hstack(parts)


def variance_importance(reps, fit_idx):
    """Per-column ranking signal, fitted on fit_idx only.

    Deliberately label-free (column variance after standardising on fit_idx).
    The experiment is about how the BUDGET is shared between representations,
    so the within-representation ranking is held fixed and identical across all
    six rules. A label-driven ranking would let importance differences leak
    into the comparison.
    """
    out = {}
    for name, X in reps.items():
        sub = X[fit_idx]
        sd = np.nanstd(sub, axis=0)
        out[name] = np.nan_to_num(sd, nan=0.0)
    return out


def run(labels, groups, reps, config, args):
    n_folds = config['n_splits']
    names = sorted(reps)
    X_any = reps[names[0]]
    print(f'  {len(names)} representations: '
          + ', '.join(f'{n}({reps[n].shape[1]}D)' for n in names))
    print(f'  budgets {list(BUDGETS)}; rules {list(RULES)}')

    held = {r: {b: {y: [] for y in YARDSTICKS} for b in BUDGETS} for r in RULES}
    shares_log = {r: {b: [] for b in BUDGETS} for r in RULES}
    degenerate = {r: 0 for r in RULES}
    fold_count = 0

    for tr_idx, te_idx in cv_split(groups, n_folds, X_any, labels,
                                   seed=SEED, mode=args.mode):
        fold_count += 1
        # Split the training rows by scaffold group. The shares are decided on
        # one part and the hybrid is scored on the untouched held-out fold.
        g_tr = groups[tr_idx]
        uniq = np.unique(g_tr)
        rng = np.random.RandomState(SEED + fold_count)
        pick = set(rng.choice(uniq, size=max(1, len(uniq) // 2),
                              replace=False).tolist())
        m = np.array([g in pick for g in g_tr])
        if m.sum() < 10 or (~m).sum() < 10:
            print(f'    fold {fold_count}: skipped, split too small')
            continue
        a_idx, b_idx = tr_idx[m], tr_idx[~m]

        sig = score_each_representation(reps, a_idx, b_idx, labels, groups)
        print(f'    fold {fold_count}: '
              + '; '.join(f'{k}=' + '/'.join(
                  f'{sig[k][n]:.3f}' if np.isfinite(sig[k][n]) else 'nan'
                  for n in names) for k in ('sac', 'readacross')))

        imp = variance_importance(reps, tr_idx)

        for rule in RULES:
            if rule == 'equal':
                sh, deg = np.full(len(names), 1.0 / len(names)), False
            elif rule in ('sac', 'readacross', 'predictive'):
                sh, deg = shares_from_scores(sig[rule], names)
            elif rule == 'rank_50':
                sh, deg = shares_from_ranks(sig['sac'], sig['readacross'],
                                            names, 0.5)
            elif rule == 'rank_75ra':
                sh, deg = shares_from_ranks(sig['sac'], sig['readacross'],
                                            names, 0.75)
            else:
                raise ValueError(rule)
            if deg:
                degenerate[rule] += 1

            for bud in BUDGETS:
                picked = build_hybrid(reps, names, sh, bud, tr_idx, imp)
                Xtr_raw = apply_pick(reps, picked, tr_idx)
                Xte_raw = apply_pick(reps, picked, te_idx)
                if Xtr_raw is None or Xte_raw is None:
                    continue
                sc = StandardScaler().fit(Xtr_raw)
                Xtr = np.nan_to_num(sc.transform(Xtr_raw))
                Xte = np.nan_to_num(sc.transform(Xte_raw))
                held[rule][bud]['predictive'].append(
                    y_predictive(Xtr, labels[tr_idx], Xte, labels[te_idx]))
                held[rule][bud]['sac'].append(
                    y_sac(Xtr, labels[tr_idx], Xte, labels[te_idx]))
                held[rule][bud]['readacross'].append(
                    y_readacross(Xtr, labels[tr_idx], Xte, labels[te_idx],
                                 groups[te_idx]))
                shares_log[rule][bud].append(
                    {n: round(float(s), 4) for n, s in zip(names, sh)})

    summary = {}
    for rule in RULES:
        summary[rule] = {'degenerate_folds': degenerate[rule], 'by_budget': {}}
        for bud in BUDGETS:
            e = {}
            for y in YARDSTICKS:
                arr = np.array(held[rule][bud][y], dtype=np.float64)
                ok = np.isfinite(arr)
                e[y] = float(arr[ok].mean()) if ok.any() else None
                e[y + '_sd'] = float(arr[ok].std(ddof=1)) if ok.sum() > 1 else None
                e[y + '_folds'] = int(ok.sum())
            e['mean_shares'] = shares_log[rule][bud]
            summary[rule]['by_budget'][bud] = e

    return {'rules': summary, 'representations': names,
            'n_folds_scored': fold_count, 'budgets': list(BUDGETS)}


def main():
    parser = create_experiment_parser(
        'F21: what should decide how many features each representation gets')
    parser.add_argument('--pred-model', type=str, default='histgb',
                        choices=sorted(PRED_MODELS),
                        help="Model behind the predictive yardstick. lightgbm "
                             "SEGFAULTS on the project laptop once the dataset "
                             "loader is imported -- server only.")
    args = parser.parse_args()
    PRED_MODEL[0] = args.pred_model
    config = get_config(args)

    if config['n_splits'] < 3:
        raise SystemExit(
            f"F21 needs at least 3 folds to report a standard deviation; "
            f"got n_splits={config['n_splits']}. Pass --mode full. "
            f"(--mode intermediary sets 1 fold and wasted a 2.1 hour run on "
            f"2026-08-26.)")

    t0 = time.time()
    from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
    smiles, labels_raw, task_type = load_dataset(args.dataset)
    labels = np.asarray(labels_raw, dtype=np.float64).ravel()
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    metric_name, _, lower_is_better = get_domain_metric(
        {'domain': 'molecular', 'task_type': task_type})

    wanted = ([r.strip() for r in args.reps.split(',')]
              if args.reps and args.reps not in ('fast', 'all', 'fast+hf')
              else list(DEFAULT_REPS))
    print(f'\nF21 {args.dataset}: {len(labels)} samples, {n_scaffolds} scaffolds, '
          f'{config["n_splits"]} folds (no cache)')
    reps, rep_failures = build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built: {rep_failures}')

    results = {
        'experiment': 'f21_budget_split_signal',
        'dataset': args.dataset, 'mode': args.mode, 'seed': SEED,
        'n_samples': int(len(labels)), 'n_scaffolds': int(n_scaffolds),
        'cache_free': True, 'predictive_model': args.pred_model,
        'representation_build_failures': rep_failures,
    }
    results.update(run(labels, groups, reps, config, args))
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    print(f'\n  Held-out score of the hybrid each rule built:')
    for bud in BUDGETS:
        print(f'\n    budget {bud}')
        print(f'      {"rule":12s} ' + ' '.join(f'{y:>22s}' for y in YARDSTICKS))
        base = results['rules']['equal']['by_budget'][bud]
        for rule in RULES:
            e = results['rules'][rule]['by_budget'][bud]
            cells = []
            for y in YARDSTICKS:
                if e[y] is None:
                    cells.append(f'{"n/a":>22}')
                    continue
                sd = base.get(y + '_sd')
                if rule == 'equal' or not sd:
                    cells.append(f'{e[y]:>10.4f}{"":>12}')
                else:
                    d = (e[y] - base[y]) / sd
                    cells.append(f'{e[y]:>10.4f}  ({d:+5.2f} sd)')
            print(f'      {rule:12s} ' + ' '.join(cells))
    print(f'\n    (sd = fold-to-fold standard deviation of the equal split. '
          f'A gap under 1 sd is not a result.)')
    for rule in RULES:
        d = results['rules'][rule]['degenerate_folds']
        if d:
            print(f'    {rule}: {d} folds where every score was <= 0 and the '
                  f'split fell back to equal')

    save_experiment_results(results, args.output, 'f21_budget_split_signal',
                            dataset=args.dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


if __name__ == '__main__':
    main()
