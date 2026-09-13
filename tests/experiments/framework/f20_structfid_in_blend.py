#!/usr/bin/env python3
"""F20: is maximum-common-substructure fidelity usable INSIDE a fold, and does
including it in a blended objective beat a blend that leaves it out?

STANDALONE AND CACHE-FREE. Reuses F19's four scorers and candidate pool so the
two experiments are directly comparable. Never reads or writes a search pickle,
a baseline pickle, a hybrid-eval pickle or a result database.

---------------------------------------------------------------------------
THE QUESTION

F19 measured that on herg_ki the best objective was a blend of readacross and
struct_fid (0.75 / 0.25). The engine cannot express that blend, because
_structural_fidelity_score is called in exactly one place
(find_best_hybrid.py:6350, the outer-fold scorer) and returns NaN when fewer
than 10 maximum-common-substructure pairs have BOTH molecules inside the fold.

So before changing the engine there are two things to establish:

  A. FEASIBILITY. How many of the N sampled pairs actually survive inside a
     fold? If the answer is "a handful", struct_fid cannot be a selection
     signal at all and the blend has to drop it. Measured here at several pair
     counts and several fold counts, per dataset. Also reported: how stable the
     resulting rho is, as the standard deviation across folds.

  B. WORTH IT. Head to head, on held-out data:
        blends that INCLUDE struct_fid   vs   blends that EXCLUDE it
     If excluding it costs nothing, the engine change is unnecessary and the
     blend should use sac + readacross, both of which are already computable
     wherever the search decides anything.

WHAT A NEGATIVE RESULT LOOKS LIKE, stated up front: if the best
struct_fid-free blend scores within noise of the best struct_fid-containing
blend on every yardstick, then struct_fid is not worth making computable
per fold. The summary prints that gap explicitly so it cannot be talked around.

---------------------------------------------------------------------------
Usage:
    python f20_structfid_in_blend.py --dataset herg_ki --mode intermediary
    python f20_structfid_in_blend.py --dataset logd --pair-counts 2500,10000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import itertools
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
    SCORERS, YARDSTICKS, make_candidates, SEED, PRED_MODEL, PRED_MODELS,
)


def pair_survival(pairs, groups, labels, X, n_folds_list, mode):
    """A. How many sampled pairs land wholly inside one fold?

    Reports, for each fold count, the number of pairs with both molecules in
    the held-out fold. The engine's own floor is 10 (below that
    _structural_fidelity_score returns NaN); a Spearman on 10 points is not
    worth reporting, so the count matters, not just whether it passes.
    """
    I = np.asarray(pairs['I']).astype(int)
    J = np.asarray(pairs['J']).astype(int)
    out = {}
    for nf in n_folds_list:
        counts = []
        for _tr, te in cv_split(groups, nf, X, labels, seed=SEED, mode=mode):
            s = set(int(t) for t in te)
            counts.append(int(np.sum([(int(i) in s) and (int(j) in s)
                                      for i, j in zip(I, J)])))
        out[nf] = {
            'per_fold_pairs': counts,
            'min': int(min(counts)) if counts else 0,
            'median': float(np.median(counts)) if counts else 0.0,
            'frac_of_sampled': (float(np.median(counts)) / max(1, len(I))),
            'folds_below_engine_floor_10': int(sum(c < 10 for c in counts)),
            'folds_below_100': int(sum(c < 100 for c in counts)),
        }
    return out


def blends_with_and_without():
    """Every single objective and blend, tagged by whether it uses struct_fid."""
    out = {}
    for y in YARDSTICKS:
        out[y] = ({y: 1.0}, 'struct_fid' == y)
    for a, b in itertools.combinations(YARDSTICKS, 2):
        for w in (0.25, 0.5, 0.75):
            name = f'{a}*{w:g}+{b}*{1 - w:g}'
            out[name] = ({a: w, b: round(1 - w, 2)},
                         'struct_fid' in (a, b))
    for trio in itertools.combinations(YARDSTICKS, 3):
        out['+'.join(trio) + ':thirds'] = ({t: 1 / 3 for t in trio},
                                           'struct_fid' in trio)
    return out


def run_scoring(labels, groups, reps, pairs, config, args):
    """B. Each objective picks on a selection half; the pick is scored on an
    untouched held-out half under all four yardsticks. Same protocol as F19."""
    n_folds = config['n_splits']
    rng = np.random.RandomState(SEED)
    X_pool, cands, rep_names = make_candidates(reps, rng)
    blends = blends_with_and_without()
    print(f'  {len(cands)} candidates over {X_pool.shape[1]} pooled features; '
          f'{len(blends)} objectives '
          f'({sum(1 for _, u in blends.values() if u)} use struct_fid)')

    picked = {b: {y: [] for y in YARDSTICKS} for b in blends}
    struct_fid_nan_folds = 0
    fold_count = 0

    for tr_idx, te_idx in cv_split(groups, n_folds, X_pool, labels,
                                   seed=SEED, mode=args.mode):
        fold_count += 1
        g_tr = groups[tr_idx]
        inner = np.unique(g_tr)
        rng2 = np.random.RandomState(SEED + fold_count)
        sel_groups = set(rng2.choice(inner, size=max(1, len(inner) // 2),
                                     replace=False).tolist())
        sel_mask = np.array([g in sel_groups for g in g_tr])
        if sel_mask.sum() < 10 or (~sel_mask).sum() < 10:
            print(f'    fold {fold_count}: skipped, selection split too small')
            continue
        a_idx, b_idx = tr_idx[sel_mask], tr_idx[~sel_mask]

        sel_scores = {y: [] for y in YARDSTICKS}
        for c in cands:
            cidx = c['idx']
            sc = StandardScaler().fit(X_pool[np.ix_(a_idx, cidx)])
            Xa = np.nan_to_num(sc.transform(X_pool[np.ix_(a_idx, cidx)]))
            Xb = np.nan_to_num(sc.transform(X_pool[np.ix_(b_idx, cidx)]))
            lg = {int(g): l for l, g in enumerate(b_idx)}
            for y in YARDSTICKS:
                sel_scores[y].append(
                    SCORERS[y](Xa, labels[a_idx], Xb, labels[b_idx],
                               groups[b_idx], lg, pairs))

        # Did struct_fid survive on the SELECTION half at all? If it is NaN for
        # every candidate then it could not have been used to select, which is
        # exactly the failure mode A is measuring.
        if not np.any(np.isfinite(np.array(sel_scores['struct_fid'],
                                           dtype=np.float64))):
            struct_fid_nan_folds += 1

        ranks = {}
        for y in YARDSTICKS:
            arr = np.array(sel_scores[y], dtype=np.float64)
            ok = np.isfinite(arr)
            r = np.full(len(arr), np.nan)
            if ok.sum() >= 2:
                r[ok] = rankdata(arr[ok]) / ok.sum()
            ranks[y] = r

        for bname, (weights, _uses) in blends.items():
            combo = np.zeros(len(cands))
            valid = np.ones(len(cands), dtype=bool)
            for y, w in weights.items():
                valid &= np.isfinite(ranks[y])
                combo += w * np.nan_to_num(ranks[y])
            if not valid.any():
                continue
            combo[~valid] = -np.inf
            pick = int(np.argmax(combo))
            cidx = cands[pick]['idx']
            sc = StandardScaler().fit(X_pool[np.ix_(tr_idx, cidx)])
            Xtr = np.nan_to_num(sc.transform(X_pool[np.ix_(tr_idx, cidx)]))
            Xte = np.nan_to_num(sc.transform(X_pool[np.ix_(te_idx, cidx)]))
            lg = {int(g): l for l, g in enumerate(te_idx)}
            for y in YARDSTICKS:
                picked[bname][y].append(
                    SCORERS[y](Xtr, labels[tr_idx], Xte, labels[te_idx],
                               groups[te_idx], lg, pairs))

    summary = {}
    for bname, (weights, uses) in blends.items():
        e = {'uses_struct_fid': bool(uses), 'weights': weights}
        for y in YARDSTICKS:
            arr = np.array(picked[bname][y], dtype=np.float64)
            ok = np.isfinite(arr)
            e[y] = float(arr[ok].mean()) if ok.any() else None
            e[y + '_sd'] = float(arr[ok].std()) if ok.sum() > 1 else None
            e[y + '_n_folds'] = int(ok.sum())
        summary[bname] = e

    return {'objectives': summary, 'n_candidates': len(cands),
            'n_folds_scored': fold_count,
            'folds_where_struct_fid_was_all_nan': struct_fid_nan_folds,
            'pooled_features': int(X_pool.shape[1]),
            'representations': rep_names}


def head_to_head(summary):
    """The single number the experiment exists to produce: best blend that uses
    struct_fid minus best blend that does not, on each yardstick."""
    out = {}
    for y in YARDSTICKS:
        with_sf = [(e[y], b) for b, e in summary.items()
                   if e['uses_struct_fid'] and isinstance(e[y], float)]
        without = [(e[y], b) for b, e in summary.items()
                   if not e['uses_struct_fid'] and isinstance(e[y], float)]
        if not with_sf or not without:
            continue
        bw, nw = max(with_sf), max(without)
        out[y] = {'best_with_struct_fid': bw[1], 'score_with': bw[0],
                  'best_without_struct_fid': nw[1], 'score_without': nw[0],
                  'gap_with_minus_without': bw[0] - nw[0]}
    return out


def main():
    parser = create_experiment_parser(
        'F20: is struct_fid usable inside a fold, and is it worth blending in')
    parser.add_argument('--pair-counts', type=str, default='2500,10000',
                        help='Comma-separated maximum-common-substructure pair '
                             'counts to sample. Feasibility (part A) is '
                             'reported for each; scoring (part B) uses the '
                             'largest.')
    parser.add_argument('--fold-counts', type=str, default='3,5,10',
                        help='Fold counts to measure pair survival at.')
    parser.add_argument('--pred-model', type=str, default='histgb',
                        choices=sorted(PRED_MODELS),
                        help="Model behind the 'predictive' yardstick. lightgbm "
                             "SEGFAULTS on the project laptop once the dataset "
                             "loader is imported -- server only.")
    args = parser.parse_args()
    PRED_MODEL[0] = args.pred_model
    config = get_config(args)

    t0 = time.time()
    from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
    from kirby.similarity.mcs import compute_mcs_pairs

    smiles, labels_raw, task_type = load_dataset(args.dataset)
    labels = np.asarray(labels_raw, dtype=np.float64).ravel()
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    metric_name, _, lower_is_better = get_domain_metric(
        {'domain': 'molecular', 'task_type': task_type})

    wanted = ([r.strip() for r in args.reps.split(',')]
              if args.reps and args.reps not in ('fast', 'all', 'fast+hf')
              else list(DEFAULT_REPS))
    print(f'\nF20 {args.dataset}: {len(labels)} samples, {n_scaffolds} scaffolds, '
          f'{config["n_splits"]} folds (no cache)')
    reps, rep_failures = build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built: {rep_failures}')

    pair_counts = [int(x) for x in args.pair_counts.split(',')]
    fold_counts = [int(x) for x in args.fold_counts.split(',')]
    X_any = next(iter(reps.values()))

    # ---- A. Feasibility -----------------------------------------------------
    feasibility, pairs_by_n = {}, {}
    for npair in pair_counts:
        print(f'  building {npair} structural pairs (computed, not cached)...')
        p = compute_mcs_pairs(smiles, labels, n_pairs=npair, seed=SEED)
        pairs_by_n[npair] = p
        feasibility[npair] = pair_survival(p, groups, labels, X_any,
                                           fold_counts, args.mode)
        print(f'    {len(p["I"])} pairs sampled')
        for nf, d in feasibility[npair].items():
            print(f'      {nf:2d} folds: median {d["median"]:.0f} pairs inside a '
                  f'fold ({d["frac_of_sampled"]*100:.1f}% of sampled); '
                  f'{d["folds_below_engine_floor_10"]} folds under the engine '
                  f"floor of 10, {d['folds_below_100']} under 100")

    # ---- B. Worth it --------------------------------------------------------
    biggest = max(pair_counts)
    print(f'\n  Scoring with {biggest} pairs.')
    results = {
        'experiment': 'f20_structfid_in_blend',
        'dataset': args.dataset, 'mode': args.mode, 'seed': SEED,
        'n_samples': int(len(labels)), 'n_scaffolds': int(n_scaffolds),
        'cache_free': True, 'predictive_model': args.pred_model,
        'representation_build_failures': rep_failures,
        'pair_counts_tested': pair_counts, 'fold_counts_tested': fold_counts,
        'feasibility': feasibility,
        'scoring_pair_count': biggest,
    }
    results.update(run_scoring(labels, groups, reps, pairs_by_n[biggest],
                               config, args))
    results['head_to_head'] = head_to_head(results['objectives'])
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    obj = results['objectives']
    print(f'\n  Held-out score of each objective\'s pick '
          f'(* = the objective uses struct_fid):')
    print(f'    {"objective":36s} ' + ' '.join(f'{y:>11s}' for y in YARDSTICKS))
    order = sorted(obj.items(),
                   key=lambda kv: -np.nanmean([kv[1][y] for y in YARDSTICKS
                                               if kv[1][y] is not None]
                                              or [np.nan]))
    for bname, e in order:
        def f(v):
            return f'{v:>11.4f}' if isinstance(v, float) else f'{"n/a":>11}'
        star = '*' if e['uses_struct_fid'] else ' '
        print(f'  {star} {bname:36s} ' + ' '.join(f(e[y]) for y in YARDSTICKS))

    print(f'\n  HEAD TO HEAD (positive = including struct_fid helps):')
    for y, d in results['head_to_head'].items():
        print(f'    {y:12s} {d["gap_with_minus_without"]:+.4f}   '
              f'with={d["score_with"]:.4f} ({d["best_with_struct_fid"]})  '
              f'without={d["score_without"]:.4f} '
              f'({d["best_without_struct_fid"]})')
    print(f'    folds where struct_fid was unusable on the selection half: '
          f'{results["folds_where_struct_fid_was_all_nan"]} of '
          f'{results["n_folds_scored"]}')

    # The four yardstick columns are ALL higher-is-better Spearman rho --
    # including the 'predictive' one, which is held-out Spearman of a fitted
    # model, not the dataset's domain metric. Stamping the domain metric
    # (RMSE on these datasets) at the top level mislabelled every column and
    # inverted its direction for any reader that trusted `lower_is_better`.
    # The domain metric is kept under its own key: it is what a downstream
    # KIRBy run would be scored by, it is just not what this file reports.
    results['metric_name'] = 'spearman'
    results['lower_is_better'] = False
    results['yardstick_direction'] = {y: 'higher_is_better' for y in YARDSTICKS}
    results['domain_metric'] = {'name': metric_name,
                                'lower_is_better': bool(lower_is_better)}

    save_experiment_results(results, args.output, 'f20_structfid_in_blend',
                            dataset=args.dataset,
                            metric_name='spearman',
                            lower_is_better=False)
    return results


if __name__ == '__main__':
    main()
