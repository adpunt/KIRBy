#!/usr/bin/env python3
"""F23: which similarity space finds wrong activity values best?

STANDALONE AND CACHE-FREE, like f18, f19 and f22. Reads no hybrid search, no
baseline pickle, no hybrid-eval pickle and no result database, and writes none.

---------------------------------------------------------------------------
PROVENANCE. This is a rebuild of
tests/experiments/similarity_metrics/similarity_noise_cleaning.py, which was
written as a private script, run by hand, and registered nowhere. Four things
changed and each one changes a number.

1. It loaded molecules through the standalone study's own loader and read
   maximum-common-substructure pairs from mcs_pairs_<dataset>.npz. Those pair
   files index the study's row order. The framework loads a different number of
   rows, so reusing them pairs each structural distance with the wrong molecule.
   Pairs are computed here from the framework's own molecule list.

2. It ranked the candidate spaces and kept the best three of each ranking.
   Adelaide, 2026-08-26: "there should not be a top 5 in any scenario." Every
   space is scored.

3. It chose which features to keep using every pair, then reported structural
   agreement on those same pairs. Even the corrected version split the PAIR
   list, so the same molecule appeared on both sides. Here the MOLECULES are
   split by scaffold group: features are chosen on one half and every number is
   reported on the other half, which shares no molecule with it.

4. It hardcoded four block combinations. Here the spaces come from the same
   representations f22 compares, so whatever f22 settles on is what gets tested
   for finding wrong labels.

---------------------------------------------------------------------------
WHAT IT MEASURES. Activity values are corrupted on purpose, then each space is
asked to find the corrupted ones. The battery is simeval.noise_sweep_scores,
already written and already calibrated against ChEMBL: Gaussian noise at three
sizes (0.3, 0.6, 0.9 in label units) crossed with three corruption rates (10%,
20%, 30%), three injections each.

  detection AUROC     does a corrupted value disagree with its neighbours in
                      this space more than an untouched one does
  recovery corr       does averaging over neighbours pull the value back toward
                      the truth

Reported beside them, on the same held-out molecules:

  readacross          Spearman(kNN-predicted activity, true activity)
  struct_fid          Spearman(pairwise distance, maximum common substructure)

The baseline every space is measured against is the ECFP4 count fingerprint
compared with Tanimoto, which is what the July study used and what the field
uses.

Usage:
    python f23_noise_cleaning_metric.py --dataset herg_ki
    python f23_noise_cleaning_metric.py --dataset logd --budgets 100,500
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import warnings
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'similarity_metrics'))
import simeval  # noqa: E402

# f22 owns the representation list, the feature rankers and the gluing, so this
# script imports them rather than keeping a second copy that can drift.
import importlib.util as _ilu
_f22_path = os.path.join(os.path.dirname(__file__),
                         'f22_similarity_feature_sources.py')
_spec = _ilu.spec_from_file_location('f22_similarity_feature_sources', _f22_path)
f22 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(f22)

SEED = 42
BUDGETS = (100, 500)


def split_molecules_by_group(groups, frac_select=0.5, seed=SEED):
    """Two disjoint halves of the molecules, split on whole scaffold groups.

    Splitting on groups rather than on rows means no molecule and no close
    relative of it is on both sides. Feature choice happens on the select half;
    every reported number comes from the score half.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    rs = np.random.RandomState(seed)
    sel_groups = set(rs.permutation(uniq)[:int(round(frac_select * len(uniq)))])
    is_sel = np.array([x in sel_groups for x in g])
    return np.where(is_sel)[0], np.where(~is_sel)[0]


def build_spaces(reps, sel_idx, score_idx, pairs, labels, budgets,
                 candidate_names):
    """Every space to be tested, each as a distance matrix over the score half.

    Features are ranked on the select half only. Standardising uses the select
    half's statistics. Nothing here reads the score half's activity values.
    """
    y_sel = labels[sel_idx]
    in_sel = np.zeros(len(labels), bool)
    in_sel[sel_idx] = True
    I = np.asarray(pairs['I']).astype(int)
    J = np.asarray(pairs['J']).astype(int)
    keep = in_sel[I] & in_sel[J]
    pi, pj = I[keep], J[keep]
    mcs_sel = np.asarray(pairs['mcs'], dtype=np.float64)[keep]
    dy_sel = np.abs(labels[pi] - labels[pj])

    spaces = {}

    # The baseline: ECFP4 counts compared with Tanimoto. No selection, no
    # standardising -- Tanimoto is defined on the raw counts.
    if 'ecfp4_count' in reps:
        spaces['BASELINE ecfp4_count:tanimoto'] = simeval.dist_tanimoto(
            reps['ecfp4_count'][score_idx])

    # Each representation on its own, full width, Euclidean.
    for name, X in reps.items():
        sc = StandardScaler().fit(X[sel_idx])
        Z = np.nan_to_num(sc.transform(X[score_idx]), nan=0.0, posinf=0.0,
                          neginf=0.0)
        spaces[f'single {name}'] = cdist(Z, Z, metric='euclidean')

    rankings = {}
    for name, X in reps.items():
        for kind in f22.RANKERS:
            if kind != 'model' and len(pi) < 20:
                continue
            target = mcs_sel if kind == 'structure' else dy_sel
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rankings[(name, kind)] = f22.rank_features(
                    X, sel_idx, y_sel, pi, pj, target, kind)

    rep_sets = {
        'candidates': [r for r in candidate_names if r in reps],
        'registered': [r for r in reps if r not in candidate_names],
    }
    rep_sets['both'] = rep_sets['candidates'] + rep_sets['registered']

    for set_name, names in rep_sets.items():
        if not names:
            continue
        for ranker in f22.RANKERS:
            if any((n, ranker) not in rankings for n in names):
                continue
            for budget in budgets:
                per = max(1, budget // len(names))
                cut = {}
                for n in names:
                    k = min(per, reps[n].shape[1])
                    order = np.argsort(-rankings[(n, ranker)])[:k]
                    cut[n] = (reps[n][np.ix_(sel_idx, order)],
                              reps[n][np.ix_(score_idx, order)])
                for mode in ('raw', 'zscore', 'zscore_sqrt'):
                    widths = {v[0].shape[1] for v in cut.values()}
                    if mode == 'zscore_sqrt' and len(widths) == 1:
                        continue  # uniform rescale, identical to zscore
                    _, G = f22.glue(cut, mode, set(candidate_names))
                    if G is None:
                        continue
                    spaces[f'{set_name}|{ranker}|{budget}|{mode}'] = cdist(
                        G, G, metric='euclidean')
    return spaces, int(len(pi))


def score_space(D, y, groups, score_idx, pairs):
    """The noise battery plus the two yardsticks, all on the score half."""
    grid = simeval.noise_sweep_scores(D, y)
    dets = [v[0] for v in grid.values() if np.isfinite(v[0])]
    recs = [v[1] for v in grid.values() if np.isfinite(v[1])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            ra, _ = simeval.eval_knn_readacross(D, y, groups)
            ra = float(ra)
        except Exception:
            ra = float('nan')
    local = {int(g): l for l, g in enumerate(score_idx)}
    return {
        'detect_mean': float(np.mean(dets)) if dets else None,
        'recovery_mean': float(np.mean(recs)) if recs else None,
        'readacross': ra if np.isfinite(ra) else None,
        'struct_fid': f22.score_struct_fidelity(D, local, pairs),
        'grid': grid,
    }


def main():
    parser = create_experiment_parser(
        'F23: which similarity space finds wrong activity values best?')
    parser.add_argument('--n-pairs', type=int, default=4000,
                        help='Maximum common substructure pairs to compute.')
    args = parser.parse_args()
    if not args.budgets:
        args.budgets = ','.join(map(str, BUDGETS))
    budgets = [int(b) for b in args.budgets.split(',')]
    get_config(args)

    t0 = time.time()
    from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
    smiles, labels_raw, task_type = load_dataset(args.dataset)
    labels = np.asarray(labels_raw, dtype=np.float64).ravel()
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    data = {'labels': labels, 'groups': groups, 'smiles': smiles,
            'task_type': task_type, 'domain': 'molecular'}
    metric_name, _, lower_is_better = get_domain_metric(data)

    # Read-across here is a k-nearest-neighbour REGRESSION scored by Spearman.
    # On a classification label that is the wrong measurement, and it would run
    # without complaining. Refuse instead of reporting a wrong number.
    if str(task_type).lower().startswith('class'):
        raise ValueError(
            f"{args.dataset} is a classification dataset. Read-across here is a "
            f"nearest-neighbour regression scored by Spearman, which is not "
            f"defined on class labels. Use a ranking metric such as AUROC "
            f"before running this on {args.dataset}.")

    listed, usable, dropped = f22.registered_reps_for(args.dataset)
    wanted = list(f22.CANDIDATE_REPS) + usable
    print(f'\nF23 {args.dataset}: {len(labels)} molecules, {n_scaffolds} '
          f'scaffold groups', flush=True)
    if dropped:
        print(f'  dropped (need torch or a graph library): {dropped}', flush=True)
    print(f'  building {len(wanted)} representations from SMILES '
          f'(no cache)...', flush=True)
    reps, rep_failures = f22.build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built. Failures: {rep_failures}')

    from kirby.similarity.mcs import compute_mcs_pairs
    print(f'  computing {args.n_pairs} maximum common substructure pairs...',
          flush=True)
    tp = time.time()
    pairs = compute_mcs_pairs(smiles, labels, n_pairs=args.n_pairs, seed=SEED)
    print(f'    {len(pairs["I"])} pairs in {time.time() - tp:.1f}s', flush=True)

    sel_idx, score_idx = split_molecules_by_group(groups, seed=SEED)
    print(f'  {len(sel_idx)} molecules choose the features, '
          f'{len(score_idx)} molecules carry every reported number, '
          f'no molecule in both', flush=True)

    spaces, n_sel_pairs = build_spaces(reps, sel_idx, score_idx, pairs, labels,
                                       budgets, list(f22.CANDIDATE_REPS))
    print(f'  {len(spaces)} spaces built, features chosen on '
          f'{n_sel_pairs} pairs', flush=True)

    y_score = labels[score_idx]
    g_score = groups[score_idx]
    rows = {}
    for i, (name, D) in enumerate(sorted(spaces.items())):
        ts = time.time()
        rows[name] = score_space(D, y_score, g_score, score_idx, pairs)
        print(f'    [{i + 1}/{len(spaces)}] {name}  '
              f'detect {rows[name]["detect_mean"]}  '
              f'({time.time() - ts:.1f}s)', flush=True)

    results = {
        'experiment': 'f23_noise_cleaning_metric',
        'dataset': args.dataset,
        'seed': SEED,
        'cache_free': True,
        'n_samples': int(len(labels)),
        'n_scaffold_groups': int(n_scaffolds),
        'n_molecules_choosing_features': int(len(sel_idx)),
        'n_molecules_scored': int(len(score_idx)),
        'n_pairs_for_feature_choice': n_sel_pairs,
        'noise_sigmas': list(simeval.NOISE_SIGMAS),
        'noise_rates': list(simeval.NOISE_RATES),
        'budgets': budgets,
        'registered_representations_dropped_locally': dropped,
        'representation_build_failures': rep_failures,
        'spaces': rows,
        'elapsed_seconds': None,
    }
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    base_key = 'BASELINE ecfp4_count:tanimoto'
    base = rows.get(base_key)

    def f(v, w=11):
        return f'{v:>{w}.4f}' if isinstance(v, float) else f'{"n/a":>{w}}'

    order = sorted(rows.items(),
                   key=lambda kv: -(kv[1]['detect_mean'] or -9))
    print('\n  EVERY SPACE, ranked by how well it finds corrupted values')
    print(f'    {"space":46s} {"detect":>9s} {"recovery":>9s} '
          f'{"readacross":>11s} {"struct_fid":>11s}')
    for name, r in order:
        mark = ' <-' if name == base_key else '   '
        print(f'    {name:46s} {f(r["detect_mean"], 9)} '
              f'{f(r["recovery_mean"], 9)} {f(r["readacross"])} '
              f'{f(r["struct_fid"])}{mark}')

    if base and base['detect_mean'] is not None:
        beat_d = sum(1 for n, r in rows.items()
                     if n != base_key and (r['detect_mean'] or -9) > base['detect_mean'])
        beat_r = sum(1 for n, r in rows.items()
                     if n != base_key and (r['recovery_mean'] or -9) > (base['recovery_mean'] or -9))
        print(f'\n  against the ECFP4-count Tanimoto baseline '
              f'(detect {base["detect_mean"]:.4f}, '
              f'recovery {base["recovery_mean"]:.4f}):')
        print(f'    {beat_d} of {len(rows) - 1} spaces detect better')
        print(f'    {beat_r} of {len(rows) - 1} spaces recover better')

    save_experiment_results(results, args.output, 'f23_noise_cleaning_metric',
                            dataset=args.dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    print(f'\n  total {results["elapsed_seconds"]}s')
    return results


if __name__ == '__main__':
    main()
