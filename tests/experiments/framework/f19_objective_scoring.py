#!/usr/bin/env python3
"""F19: which objective should the search optimise, and do blends beat singles?

STANDALONE AND CACHE-FREE. Builds its own representations from SMILES, runs its
own candidate generation, and never reads or writes a search pickle, a baseline
pickle, a hybrid-eval pickle or a result database.

---------------------------------------------------------------------------
THE QUESTION

The search picks a feature subset by maximising ONE number. Which number it
maximises is a free choice, and today it is one of three:

  predictive   a model is fit on the subset and scored on held-out data
  sac          Spearman(pairwise distance, |activity difference|)
  readacross   Spearman(kNN-predicted activity, true activity)

There is a fourth quantity the code computes but cannot optimise for:

  struct_fid   Spearman(pairwise distance, maximum-common-substructure
               distance) -- how well distance tracks shared substructure

Nobody has measured which of these is the right thing to maximise, or whether a
weighted combination of two or three beats any single one. This experiment
measures exactly that, without touching the engine.

---------------------------------------------------------------------------
HOW

For each scaffold-grouped fold:

  1. Generate a diverse pool of candidate feature subsets from the pooled
     multi-representation matrix -- random subsets at several budgets, plus
     each representation on its own, plus importance-ranked subsets. The pool
     only has to be DIVERSE, not optimal: the experiment is about the selection
     rule, not about finding the best subset.

  2. Score every candidate on the TRAINING half of the fold under every
     objective, single and blended.

  3. Each objective then PICKS its argmax candidate.

  4. Score that picked candidate on the HELD-OUT half under all four
     yardsticks.

So the output answers: if you had optimised objective X, what would you have
ended up with, measured on everything? An objective whose pick scores well
across the board is a good objective. An objective whose pick scores well only
on itself is optimising a private quantity.

Selection happens on train, evaluation on held-out. Scoring a subset with the
same data that chose it is circular and would make every objective look good.

BLENDS. Every score is converted to a within-fold RANK across candidates before
blending, because the four quantities live on different scales and a raw
weighted sum would be dominated by whichever has the widest range. Weights
0.25 / 0.5 / 0.75 on every pair, plus equal thirds on each triple.

WHAT A NEGATIVE RESULT LOOKS LIKE, stated up front so it is not rationalised
later: if every objective's pick scores about the same on every yardstick, then
the objective is not a real knob and the choice does not matter. That is a
useful finding and the summary reports the spread so it is visible.

---------------------------------------------------------------------------
Usage:
    python f19_objective_scoring.py --dataset herg_ki --mode intermediary
    python f19_objective_scoring.py --dataset logd --mode full --struct-pairs
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import itertools
import warnings
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

from experiments._shared import (
    get_config, create_experiment_parser, save_experiment_results,
    cv_split, get_domain_metric,
)
from f18_distance_metric_sweep import build_representations, DEFAULT_REPS

SEED = 42
N_PAIR_SAMPLE = 4000
K_NEIGHBOURS = 5
BUDGETS = (25, 50, 100, 250, 500)
N_RANDOM_PER_BUDGET = 8

YARDSTICKS = ('predictive', 'sac', 'readacross', 'struct_fid')


# ---------------------------------------------------------------------------
# The four yardsticks, all higher-is-better, all on a standardised block.
# ---------------------------------------------------------------------------
def _euclid(X):
    G = X @ X.T
    sq = np.diag(G)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))


# The predictive yardstick's model. Default is pure sklearn on purpose:
# lightgbm SEGFAULTS in this process once the rdkit-based dataset loader has been
# imported (verified 2026-08-25, exit 139, no traceback). It works fine in a
# clean interpreter, so the conflict is between the two native libraries, not
# lightgbm itself. Anything with a native extension is a risk here; sklearn's
# HistGradientBoosting is the same family of model with no separate binary.
PRED_MODELS = {
    'histgb': lambda: __import__(
        'sklearn.ensemble', fromlist=['HistGradientBoostingRegressor']
    ).HistGradientBoostingRegressor(max_iter=100, random_state=SEED),
    'rf': lambda: __import__(
        'sklearn.ensemble', fromlist=['RandomForestRegressor']
    ).RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
    'lightgbm': lambda: __import__(
        'lightgbm', fromlist=['LGBMRegressor']
    ).LGBMRegressor(n_estimators=100, verbosity=-1, random_state=SEED),
}
PRED_MODEL = ['histgb']  # mutable so --pred-model can set it before the run


def y_predictive(X_tr, y_tr, X_te, y_te, *_):
    """Held-out Spearman of a fitted model. Model chosen by --pred-model."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m = PRED_MODELS[PRED_MODEL[0]]()
            m.fit(X_tr, y_tr)
            p = m.predict(X_te)
        except Exception:
            return np.nan
    if np.std(p) == 0 or np.std(y_te) == 0:
        return np.nan
    rho, _ = spearmanr(p, y_te)
    return float(rho) if np.isfinite(rho) else np.nan


def y_sac(_a, _b, X_te, y_te, *_):
    n = len(y_te)
    if n < 3:
        return np.nan
    rng = np.random.RandomState(SEED)
    m = min(N_PAIR_SAMPLE, n * (n - 1) // 2)
    pi, pj = rng.randint(0, n, m * 2), rng.randint(0, n, m * 2)
    keep = pi != pj
    pi, pj = pi[keep][:m], pj[keep][:m]
    if len(pi) < 3:
        return np.nan
    d = np.linalg.norm(X_te[pi] - X_te[pj], axis=1)
    dy = np.abs(y_te[pi] - y_te[pj])
    if np.std(d) == 0 or np.std(dy) == 0:
        return np.nan
    rho, _ = spearmanr(d, dy)
    return float(rho) if np.isfinite(rho) else np.nan


def y_readacross(_a, _b, X_te, y_te, g_te, *_):
    n = len(y_te)
    if n < 10 or g_te is None:
        return np.nan
    g = np.asarray(g_te).ravel()
    ng = len(np.unique(g))
    if ng < 2:
        return np.nan
    D = _euclid(X_te)
    preds = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(np.arange(n), y_te, g):
        kk = min(K_NEIGHBOURS, len(tr))
        if kk < 1:
            continue
        m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                                metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], y_te[tr])
        preds[te] = m.predict(D[np.ix_(te, tr)])
    ok = np.isfinite(preds)
    if ok.sum() < 3 or np.std(y_te[ok]) == 0 or np.std(preds[ok]) == 0:
        return np.nan
    rho, _ = spearmanr(preds[ok], y_te[ok])
    return float(rho) if np.isfinite(rho) else np.nan


def y_struct_fid(_a, _b, X_te, _y, _g, local_of_global, pairs):
    if pairs is None:
        return np.nan
    I = np.asarray(pairs['I']).astype(int)
    J = np.asarray(pairs['J']).astype(int)
    mcs = np.asarray(pairs['mcs'], dtype=np.float64)
    li = np.array([local_of_global.get(int(i), -1) for i in I])
    lj = np.array([local_of_global.get(int(j), -1) for j in J])
    keep = (li >= 0) & (lj >= 0)
    if int(keep.sum()) < 10:
        return np.nan
    d = np.linalg.norm(X_te[li[keep]] - X_te[lj[keep]], axis=1)
    m = mcs[keep]
    if np.std(d) == 0 or np.std(m) == 0:
        return np.nan
    rho, _ = spearmanr(d, m)
    return float(rho) if np.isfinite(rho) else np.nan


SCORERS = {'predictive': y_predictive, 'sac': y_sac,
           'readacross': y_readacross, 'struct_fid': y_struct_fid}


# ---------------------------------------------------------------------------
def make_candidates(reps, rng):
    """A diverse pool of feature subsets over the pooled matrix.

    Diversity is what matters, not optimality -- the experiment is about the
    selection RULE, so the pool only has to contain choices good and bad enough
    that different objectives would pick differently.
    """
    names = sorted(reps)
    offsets, cols, total = {}, [], 0
    for n in names:
        w = reps[n].shape[1]
        offsets[n] = (total, total + w)
        total += w
    X_pool = np.hstack([reps[n] for n in names])

    cands = []
    for n in names:                                   # each representation whole
        a, b = offsets[n]
        cands.append({'name': f'whole:{n}', 'idx': np.arange(a, b)})
    for bud in BUDGETS:                               # random subsets per budget
        if bud > total:
            continue
        for r in range(N_RANDOM_PER_BUDGET):
            cands.append({'name': f'random:{bud}:{r}',
                          'idx': rng.choice(total, size=bud, replace=False)})
        for n in names:                               # per-rep random slice
            a, b = offsets[n]
            if bud <= (b - a):
                cands.append({'name': f'slice:{n}:{bud}',
                              'idx': a + rng.choice(b - a, size=bud, replace=False)})
    return X_pool, cands, names


def blend_definitions():
    """Single objectives plus every weighted pair and equal-weight triple."""
    out = {y: {y: 1.0} for y in YARDSTICKS}
    for a, b in itertools.combinations(YARDSTICKS, 2):
        for w in (0.25, 0.5, 0.75):
            out[f'{a}*{w:g}+{b}*{1 - w:g}'] = {a: w, b: round(1 - w, 2)}
    for trio in itertools.combinations(YARDSTICKS, 3):
        out['+'.join(trio) + ':thirds'] = {t: 1 / 3 for t in trio}
    return out


def run(labels, groups, reps, pairs, config, args):
    n_folds = config['n_splits']
    rng = np.random.RandomState(SEED)
    X_pool, cands, rep_names = make_candidates(reps, rng)
    blends = blend_definitions()
    print(f'  {len(cands)} candidate subsets over {X_pool.shape[1]} pooled '
          f'features; {len(blends)} objectives (single + blended)')

    picked_scores = {b: {y: [] for y in YARDSTICKS} for b in blends}
    self_agreement = {b: [] for b in blends}
    fold_count = 0

    for tr_idx, te_idx in cv_split(groups, n_folds, X_pool, labels,
                                   seed=SEED, mode=args.mode):
        fold_count += 1
        # Split the TRAINING half again: candidates are SELECTED on the inner
        # half and the winner is EVALUATED on the untouched held-out half.
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

        # Score every candidate on the SELECTION half.
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

        # Rank within the fold before blending -- the four quantities are on
        # different scales and a raw weighted sum would be dominated by
        # whichever has the widest range.
        ranks = {}
        for y in YARDSTICKS:
            arr = np.array(sel_scores[y], dtype=np.float64)
            ok = np.isfinite(arr)
            r = np.full(len(arr), np.nan)
            if ok.sum() >= 2:
                r[ok] = rankdata(arr[ok]) / ok.sum()
            ranks[y] = r

        # Each objective picks its argmax, then that pick is scored on the
        # untouched held-out half.
        for bname, weights in blends.items():
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
                picked_scores[bname][y].append(
                    SCORERS[y](Xtr, labels[tr_idx], Xte, labels[te_idx],
                               groups[te_idx], lg, pairs))
            self_agreement[bname].append(cands[pick]['name'])

    summary = {}
    for bname in blends:
        e = {'picks': self_agreement[bname]}
        for y in YARDSTICKS:
            arr = np.array(picked_scores[bname][y], dtype=np.float64)
            ok = np.isfinite(arr)
            e[y] = float(arr[ok].mean()) if ok.any() else None
            e[y + '_n_folds'] = int(ok.sum())
        summary[bname] = e

    return {'objectives': summary, 'n_candidates': len(cands),
            'n_folds_scored': fold_count, 'pooled_features': int(X_pool.shape[1]),
            'representations': rep_names,
            'structural_pairs_available': pairs is not None}


def main():
    parser = create_experiment_parser(
        'F19: which objective should the search optimise')
    parser.add_argument('--struct-pairs', action='store_true',
                        help='Compute maximum-common-substructure pairs so '
                             'struct_fid becomes a usable objective. Without '
                             'this it is reported as n/a everywhere.')
    parser.add_argument('--n-pairs', type=int, default=2500)
    parser.add_argument('--pred-model', type=str, default='histgb',
                        choices=sorted(PRED_MODELS),
                        help="Model behind the 'predictive' yardstick. Default "
                             "histgb (pure sklearn). lightgbm SEGFAULTS on the "
                             "project laptop once the dataset loader is "
                             "imported -- only pass it on the server.")
    args = parser.parse_args()
    PRED_MODEL[0] = args.pred_model
    config = get_config(args)

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
    print(f'\nF19 {args.dataset}: {len(labels)} samples, {n_scaffolds} scaffolds, '
          f'{config["n_splits"]} folds (no cache)')
    reps, rep_failures = build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built: {rep_failures}')

    pairs = None
    if args.struct_pairs:
        from kirby.similarity.mcs import compute_mcs_pairs
        print(f'  building {args.n_pairs} structural pairs (computed, not cached)...')
        pairs = compute_mcs_pairs(smiles, labels, n_pairs=args.n_pairs, seed=SEED)
        print(f'    {len(pairs["I"])} pairs')

    results = {
        'experiment': 'f19_objective_scoring',
        'dataset': args.dataset, 'mode': args.mode, 'seed': SEED,
        'n_samples': int(len(labels)), 'cache_free': True,
        'predictive_model': args.pred_model,
        'representation_build_failures': rep_failures,
    }
    results.update(run(labels, groups, reps, pairs, config, args))
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    obj = results['objectives']
    print(f'\n  What each objective\'s pick scores on the held-out fold:')
    print(f'    {"objective":34s} ' +
          ' '.join(f'{y:>11s}' for y in YARDSTICKS))
    order = sorted(obj.items(),
                   key=lambda kv: -np.nanmean([kv[1][y] for y in YARDSTICKS
                                               if kv[1][y] is not None] or [np.nan]))
    for bname, e in order:
        def f(v):
            return f'{v:>11.4f}' if isinstance(v, float) else f'{"n/a":>11}'
        print(f'    {bname:34s} ' + ' '.join(f(e[y]) for y in YARDSTICKS))

    # If every objective lands in the same place the objective is not a knob.
    # Say so numerically rather than leaving it to be eyeballed.
    for y in YARDSTICKS:
        vals = [e[y] for e in obj.values() if isinstance(e[y], float)]
        if len(vals) >= 2:
            print(f'    spread on {y}: {max(vals) - min(vals):.4f} '
                  f'(min {min(vals):.4f}, max {max(vals):.4f})')

    save_experiment_results(results, args.output, 'f19_objective_scoring',
                            dataset=args.dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


if __name__ == '__main__':
    main()
