#!/usr/bin/env python3
"""F18: which distance coefficient should each scoring site in the engine use?

STANDALONE AND CACHE-FREE. This script never reads a hybrid search, a baseline
pickle, a hybrid-eval pickle or a result database, and never writes one. It
loads a dataset, computes representations, and measures. Nothing here looks for
a cache and nothing here is invalidated by one being deleted.

---------------------------------------------------------------------------
WHY: the engine computes a pairwise distance at six places and every one of
them is hard-coded to Euclidean. Those six use only THREE distinct distance
kernels, and they serve FOUR distinct decisions:

  ROLE 1  BUDGET SPLIT
          Each representation is scored on its own, and the scores are turned
          into each representation's share of the feature budget.
          Engine: _evaluate_allocation (hybrid.py), used by
          allocate_performance_weighted / _weighted_global / _dominant_greedy
          / _greedy_forward.
          Kernel: SAC on 5000 sampled row pairs.

  ROLE 2  WHICH CONFIGURATION WINS THE SEARCH
          The fold score that ranks one candidate configuration against every
          other, so it decides which recipe the search returns.
          Engine: _similarity_objective_score (objective='similarity_sac') or
          _readacross_objective_score (objective='similarity').
          Kernels: SAC on sampled pairs, or read-across on a full square matrix.

  ROLE 3  WHICH SINGLE REPRESENTATION IS THE COMPARATOR
          The per-representation geometry baseline the hybrid's win or loss is
          measured against.
          Engine: the geometry branch of the baseline loop.
          Kernels: same two as role 2.

  ROLE 4  THE REPORTED NUMBER
          The held-out outer-fold score, plus structural fidelity as a stored
          side metric.
          Engine: _outer_fit_score, _structural_fidelity_score.
          Kernels: SAC or read-across, plus the structural-pair kernel.

Roles 2, 3 and 4 all reduce to "score one fold's geometry", so a coefficient
that changes one changes all three. Role 1 is different in kind: it does not
care about the absolute score, only about the ORDER of the representations,
because that order is what becomes the budget. A coefficient can leave every
absolute score alone and still reorder the representations, or the reverse.
This script measures both.

WHAT IT MEASURES, per dataset x representation x coefficient, on scaffold-
grouped held-out folds:

  sac            Spearman(pairwise distance, |activity difference|) over
                 sampled pairs. The role-1 and role-2 kernel.
  readacross     Spearman(kNN-predicted activity, true activity), neighbours
                 taken under the coefficient. The role-2/3/4 kernel.
  struct_fid     Spearman(pairwise distance, maximum-common-substructure
                 distance) over pairs. The role-4 side metric. Computed only
                 when structural pairs are available for the dataset.

Then, across coefficients:

  rank agreement  Spearman between the representation ORDERINGS produced by
                  each pair of coefficients. This is the role-1 question: if
                  two coefficients rank representations identically the budget
                  split is unchanged, whatever their absolute scores did.

---------------------------------------------------------------------------
COEFFICIENTS. Which ones are mathematically defined depends on what the
representation is, not on what it is called:

  BINARY fingerprints  -> the set-overlap family. Tanimoto, Dice, Cosine,
                          Tversky, plus Euclidean and Manhattan as baselines.
                          Soergel is EXCLUDED here because on binary vectors it
                          equals Tanimoto distance exactly (simeval says so and
                          proves it in its selftest) -- running it would be a
                          duplicate cell, not extra coverage.
  COUNT fingerprints   -> the same family in its continuous form, and Soergel
                          becomes genuinely distinct. rdkit's Tanimoto and Dice
                          accept counts; rdkit's Cosine does not, so the array
                          forms in simeval are used throughout.
  CONTINUOUS features  -> Euclidean, Manhattan, fractional Minkowski (p=0.5,
                          0.3), Chebyshev, Cosine, Correlation, Canberra,
                          Bray-Curtis, Mahalanobis. Set-overlap coefficients
                          are UNDEFINED here -- there is no set to intersect
                          once a feature is a signed z-score. They are skipped
                          and the skip is logged, never left as a blank.

Two traps this avoids on purpose:
  - scipy's 'jaccard' does NOT raise on continuous input. It coerces at zero,
    which on standardised features means "above or below the mean". That is a
    wrong number with no error, so it is never used.
  - Bray-Curtis and Canberra need non-negative input. They are run on RAW
    features, not on the standardised ones, and cells where the raw features go
    negative are skipped and logged.

Citations for the coefficient set: Bajusz, Racz & Heberger, J Cheminform 7
(2015) for the fingerprint family; Todeschini & Consonni (2009) and Todeschini
et al., JCIM 52 (2012) for the wider set; Aggarwal, Hinneburg & Keim, ICDT 2001
for fractional Minkowski in high dimension.

---------------------------------------------------------------------------
Usage:
    python f18_distance_metric_sweep.py --dataset herg_ki --reps ecfp4,ecfp4_count,mordred
    python f18_distance_metric_sweep.py --dataset logd --mode full --struct-pairs
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
from sklearn.covariance import EmpiricalCovariance

# load_data is deliberately NOT imported: its path runs every request through
# the representation cache, and this script is cache-free.
from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results, cv_split, get_domain_metric,
)

# simeval holds the fingerprint coefficients in array form, already written and
# already cited. Reused rather than reimplemented.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'similarity_metrics'))
import simeval  # noqa: E402

SEED = 42
N_PAIR_SAMPLE = 5000
K_NEIGHBOURS = 5


# ---------------------------------------------------------------------------
# Representations are built HERE, straight from SMILES. The dataset loader is
# used only for smiles, labels and scaffold groups.
#
# Two reasons this does not go through the loader's representation machinery:
#   1. The count fingerprints are not in the runner's registry at all -- they
#      exist in kirby.representations.molecular but are only reachable through
#      kirby.similarity.blocks. Set-overlap coefficients on COUNT vectors are
#      half the point of this sweep, so they have to be built directly.
#   2. The loader's path consults a representation cache. This script is
#      cache-free by design, so it does not touch it.
#
# Anything requiring torch or a graph library is deliberately absent -- those
# do not import on the project laptop and would turn a sweep into a debugging
# session. Add them for a server run if wanted.
# ---------------------------------------------------------------------------
def _rep_builders():
    from kirby.representations.molecular import (
        create_ecfp4, create_maccs, create_avalon, create_rdkfp,
        create_atom_pair, create_top_torsion,
        create_ecfp4_count, create_atompair_count,
        create_desc208, create_mordred,
    )
    return {
        # BINARY fingerprints -- the set-overlap family is defined here
        'ecfp4':          create_ecfp4,        # 2048 bits, the field standard
        'maccs':          create_maccs,        # 167 bits, structural keys
        'avalon':         create_avalon,       # 2048 bits
        'rdkfp':          create_rdkfp,        # 2048 bits, path-based
        'atom_pair':      create_atom_pair,    # 2048 bits
        'top_torsion':    create_top_torsion,  # 2048 bits
        # COUNT fingerprints -- Soergel and the continuous Tanimoto form become
        # genuinely distinct here, unlike on binary
        'ecfp4_count':    create_ecfp4_count,     # 1024 integer counts
        'atompair_count': create_atompair_count,  # 1024 integer counts
        # CONTINUOUS descriptors -- only the L_p / angular family is defined
        'desc208':        create_desc208,      # 208 raw RDKit descriptors
        'mordred':        create_mordred,      # 1613 descriptors
    }


DEFAULT_REPS = ('ecfp4', 'maccs', 'ecfp4_count', 'atompair_count', 'desc208')


def build_representations(smiles, wanted):
    """Compute each representation from SMILES. No cache, no lookup."""
    builders = _rep_builders()
    unknown = [w for w in wanted if w not in builders]
    if unknown:
        raise ValueError(
            f'unknown representation(s) {unknown}. Available: '
            f'{sorted(builders)}')
    out, failed = {}, {}
    for name in wanted:
        t = time.time()
        try:
            X = np.asarray(builders[name](smiles), dtype=np.float64)
        except Exception as exc:
            failed[name] = repr(exc)
            print(f'    {name}: FAILED {exc!r}')
            continue
        if X.ndim != 2 or X.shape[0] != len(smiles):
            failed[name] = f'shape {X.shape} does not match {len(smiles)} molecules'
            print(f'    {name}: FAILED {failed[name]}')
            continue
        # Drop all-constant columns: they contribute nothing to any distance and
        # break correlation-based coefficients with a zero-variance denominator.
        keep = np.nanstd(X, axis=0) > 0
        X = np.nan_to_num(X[:, keep], nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = X
        print(f'    {name}: {X.shape[1]}D '
              f'({int((~keep).sum())} constant columns dropped) '
              f'in {time.time() - t:.1f}s')
    return out, failed


# ---------------------------------------------------------------------------
# Representation kind. Decided by inspecting the ARRAY, not by name -- a folded
# Morgan bit vector and a hashed Morgan count vector have similar names and
# different mathematics.
# ---------------------------------------------------------------------------
def classify_representation(X):
    """Return 'binary', 'count' or 'continuous' from the array's own values."""
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        return 'continuous'
    if np.array_equal(finite, finite.astype(np.int64)):
        return 'binary' if set(np.unique(finite)).issubset({0.0, 1.0}) else 'count'
    return 'continuous'


# ---------------------------------------------------------------------------
# Continuous coefficients. The fingerprint ones come from simeval.
# Each entry: (callable taking a raw or standardised block, needs_nonnegative)
# ---------------------------------------------------------------------------
def _cdist_metric(name, **kw):
    def f(X):
        return cdist(X, X, metric=name, **kw)
    return f


def _mahalanobis(X):
    """Mahalanobis needs an invertible covariance. On wide blocks it is not
    invertible, so this returns None and the cell is skipped and logged rather
    than silently falling back to Euclidean."""
    n, d = X.shape
    if d >= n:
        return None
    try:
        cov = EmpiricalCovariance().fit(X)
        return cdist(X, X, metric='mahalanobis', VI=cov.precision_)
    except Exception:
        return None


CONTINUOUS_COEFFS = {
    'euclidean':    (_cdist_metric('euclidean'), False),
    'manhattan':    (_cdist_metric('cityblock'), False),
    'minkowski_p0.5': (_cdist_metric('minkowski', p=0.5), False),
    'minkowski_p0.3': (_cdist_metric('minkowski', p=0.3), False),
    'chebyshev':    (_cdist_metric('chebyshev'), False),
    'cosine':       (_cdist_metric('cosine'), False),
    'correlation':  (_cdist_metric('correlation'), False),
    'canberra':     (_cdist_metric('canberra'), True),
    'braycurtis':   (_cdist_metric('braycurtis'), True),
    'mahalanobis':  (_mahalanobis, False),
}

# simeval's fingerprint families. Soergel is deliberately absent from the binary
# set -- see the module docstring.
BINARY_COEFFS = dict(simeval.BIT_COEFFS)
BINARY_COEFFS['manhattan'] = _cdist_metric('cityblock')
COUNT_COEFFS = dict(simeval.COUNT_COEFFS)
COUNT_COEFFS['manhattan'] = _cdist_metric('cityblock')


def coefficients_for(kind):
    """The coefficients mathematically defined on this kind of representation."""
    if kind == 'binary':
        return {k: (f, False) for k, f in BINARY_COEFFS.items()}
    if kind == 'count':
        return {k: (f, False) for k, f in COUNT_COEFFS.items()}
    return dict(CONTINUOUS_COEFFS)


# ---------------------------------------------------------------------------
# The three kernels the engine actually uses.
# ---------------------------------------------------------------------------
def score_sac(D, y, rng):
    """Spearman(pairwise distance, |activity difference|) over sampled pairs.

    This is the role-1 and role-2 kernel. Sampling matches the engine, which
    draws 5000 off-diagonal pairs rather than using the whole square, so the
    number is comparable to what the engine would produce.
    """
    n = len(y)
    if n < 3:
        return np.nan
    m = min(N_PAIR_SAMPLE, n * (n - 1) // 2)
    pi = rng.randint(0, n, size=m * 2)
    pj = rng.randint(0, n, size=m * 2)
    keep = pi != pj
    pi, pj = pi[keep][:m], pj[keep][:m]
    if len(pi) < 3:
        return np.nan
    d = D[pi, pj]
    dy = np.abs(y[pi] - y[pj])
    if np.std(d) == 0 or np.std(dy) == 0:
        return np.nan
    rho, _ = spearmanr(d, dy)
    return float(rho) if np.isfinite(rho) else np.nan


def score_readacross(D, y, groups):
    """Spearman(kNN-predicted activity, true activity), scaffold-grouped.

    Neighbours are taken under whichever coefficient built D, so this measures
    the coefficient's effect on the decision the engine actually makes.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    n = len(y)
    if n < 10 or groups is None:
        return np.nan
    g = np.asarray(groups).ravel()
    n_groups = len(np.unique(g))
    if n_groups < 2:
        return np.nan
    Dw = np.nan_to_num(D, nan=np.nanmax(D[np.isfinite(D)]) if np.isfinite(D).any() else 1.0)
    preds = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for tr, te in gkf.split(np.arange(n), y, g):
        kk = min(K_NEIGHBOURS, len(tr))
        if kk < 1:
            continue
        m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                                metric='precomputed')
        m.fit(Dw[np.ix_(tr, tr)], y[tr])
        preds[te] = m.predict(Dw[np.ix_(te, tr)])
    ok = np.isfinite(preds)
    if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(preds[ok]) == 0:
        return np.nan
    rho, _ = spearmanr(preds[ok], y[ok])
    return float(rho) if np.isfinite(rho) else np.nan


def score_struct_fidelity(D, local_of_global, pairs):
    """Spearman(pairwise distance, maximum-common-substructure distance).

    `pairs` carries arrays I, J indexing the FULL dataset and the matching MCS
    distances. `local_of_global` maps a full-dataset row index to this fold's
    row position. Pairs with either endpoint outside the fold are dropped.
    """
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
    d = D[li[keep], lj[keep]]
    m = mcs[keep]
    if np.std(d) == 0 or np.std(m) == 0:
        return np.nan
    rho, _ = spearmanr(d, m)
    return float(rho) if np.isfinite(rho) else np.nan


# ---------------------------------------------------------------------------
def run(data, reps, pairs, config, args):
    labels = np.asarray(data['labels'], dtype=np.float64).ravel()
    groups = data.get('groups')
    n_folds = config['n_splits']

    per_rep = {}
    skipped = []          # every skipped cell, with its reason. Never a blank.
    rng_master = np.random.RandomState(SEED)

    for rep_name in sorted(reps):
        X_full = np.asarray(reps[rep_name], dtype=np.float64)
        kind = classify_representation(X_full)
        coeffs = coefficients_for(kind)
        print(f'  {rep_name}: {X_full.shape[1]}D, {kind}, '
              f'{len(coeffs)} coefficients defined')

        fold_scores = {c: {'sac': [], 'readacross': [], 'struct_fid': []}
                       for c in coeffs}

        for fold_i, (tr, va) in enumerate(cv_split(groups, n_folds, X_full,
                                                   labels, seed=SEED,
                                                   mode=args.mode)):
            y_va = labels[va]
            g_va = np.asarray(groups)[va] if groups is not None else None
            local_of_global = {int(g): l for l, g in enumerate(va)}

            X_raw_va = X_full[va]
            # Standardised block, scaler fit on the fold's TRAIN rows only.
            sc = StandardScaler().fit(X_full[tr])
            X_std_va = np.nan_to_num(sc.transform(X_raw_va), nan=0.0,
                                     posinf=0.0, neginf=0.0)

            for cname, (fn, needs_nonneg) in coeffs.items():
                # Fingerprint coefficients are defined on the RAW counts/bits.
                # Continuous coefficients run on the standardised block, except
                # the two that require non-negative input.
                if kind in ('binary', 'count'):
                    Xin = X_raw_va
                elif needs_nonneg:
                    Xin = X_raw_va
                    if np.nanmin(Xin) < 0:
                        skipped.append({
                            'rep': rep_name, 'coefficient': cname,
                            'fold': fold_i,
                            'reason': 'undefined: requires non-negative input, '
                                      'this representation has negative values'})
                        continue
                else:
                    Xin = X_std_va

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        D = fn(Xin)
                    except Exception as exc:
                        skipped.append({'rep': rep_name, 'coefficient': cname,
                                        'fold': fold_i,
                                        'reason': f'raised: {exc!r}'})
                        continue
                if D is None:
                    skipped.append({
                        'rep': rep_name, 'coefficient': cname, 'fold': fold_i,
                        'reason': 'undefined: covariance not invertible at '
                                  f'{Xin.shape[1]} features vs {Xin.shape[0]} rows'})
                    continue
                if not np.isfinite(D).any():
                    skipped.append({'rep': rep_name, 'coefficient': cname,
                                    'fold': fold_i,
                                    'reason': 'all-nan distance matrix'})
                    continue

                seed_c = int(rng_master.randint(0, 2 ** 31 - 1))
                fold_scores[cname]['sac'].append(
                    score_sac(D, y_va, np.random.RandomState(seed_c)))
                fold_scores[cname]['readacross'].append(
                    score_readacross(D, y_va, g_va))
                fold_scores[cname]['struct_fid'].append(
                    score_struct_fidelity(D, local_of_global, pairs))

        summary = {}
        for cname, s in fold_scores.items():
            entry = {}
            for kernel, vals in s.items():
                arr = np.array(vals, dtype=np.float64)
                ok = np.isfinite(arr)
                entry[kernel] = (float(arr[ok].mean()) if ok.any() else None)
                entry[kernel + '_std'] = (float(arr[ok].std()) if ok.any() else None)
                entry[kernel + '_n_folds'] = int(ok.sum())
            summary[cname] = entry
        per_rep[rep_name] = {'kind': kind, 'n_features': int(X_full.shape[1]),
                             'coefficients': summary}

    # ROLE 1: does the coefficient change the ORDER of the representations?
    # That order is what becomes the budget split, so a reordering matters even
    # if every absolute score is unchanged.
    rank_agreement = {}
    for kernel in ('sac', 'readacross'):
        names = sorted(per_rep)
        all_coeffs = sorted({c for r in per_rep.values() for c in r['coefficients']})
        vecs = {}
        for c in all_coeffs:
            v = [per_rep[r]['coefficients'].get(c, {}).get(kernel) for r in names]
            if all(x is not None for x in v) and len(v) >= 3:
                vecs[c] = v
        pairwise = {}
        cs = sorted(vecs)
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                rho, _ = spearmanr(vecs[a], vecs[b])
                pairwise[f'{a}|{b}'] = (float(rho) if np.isfinite(rho) else None)
        rank_agreement[kernel] = {
            'n_representations': len(names),
            'coefficients_compared': cs,
            'pairwise_rank_correlation': pairwise,
            'note': ('Spearman between the representation orderings two '
                     'coefficients produce. 1.0 means the budget split would be '
                     'identical. Needs at least 3 representations to mean '
                     'anything.'),
        }

    return {'by_representation': per_rep,
            'representation_rank_agreement': rank_agreement,
            'skipped_cells': skipped,
            'n_skipped_cells': len(skipped),
            'structural_pairs_available': pairs is not None}


def main():
    parser = create_experiment_parser(
        'F18: distance coefficient sweep across the engine scoring sites')
    parser.add_argument('--struct-pairs', action='store_true',
                        help='Also compute maximum-common-substructure pairs so '
                             'structural fidelity can be scored. Costs roughly '
                             '0.1s per pair on first build.')
    parser.add_argument('--n-pairs', type=int, default=4000,
                        help='Number of structural pairs when --struct-pairs is set.')
    args = parser.parse_args()
    config = get_config(args)

    t0 = time.time()
    # Data comes straight from the dataset loader and the scaffold grouper.
    # experiments._shared.load_data is NOT used: its path runs every request
    # through the representation cache, and this script is cache-free.
    from find_best_hybrid_molecular import load_dataset, assign_scaffold_groups
    smiles, labels_raw, task_type = load_dataset(args.dataset)
    labels = np.asarray(labels_raw, dtype=np.float64).ravel()
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    groups = np.asarray(groups)
    data = {'labels': labels, 'groups': groups, 'smiles': smiles,
            'task_type': task_type, 'domain': 'molecular'}
    metric_name, _, lower_is_better = get_domain_metric(data)

    wanted = ([r.strip() for r in args.reps.split(',')]
              if args.reps and args.reps not in ('fast', 'all', 'fast+hf')
              else list(DEFAULT_REPS))
    print(f'\nF18 {args.dataset}: {len(labels)} samples, {n_scaffolds} scaffolds, '
          f'{config["n_splits"]} folds, building {len(wanted)} representations '
          f'from SMILES (no cache)')
    reps, rep_failures = build_representations(smiles, wanted)
    if not reps:
        raise RuntimeError(f'no representations built. Failures: {rep_failures}')

    pairs = None
    if args.struct_pairs:
        # compute_mcs_pairs is the PURE function. get_mcs_pairs is a caching
        # wrapper around it and is deliberately not used here -- it builds a
        # path under KIRBY_CACHE_DIR and stats it, which is a cache lookup.
        from kirby.similarity.mcs import compute_mcs_pairs
        print(f'  building {args.n_pairs} maximum-common-substructure pairs '
              f'(computed here, not cached)...')
        tp = time.time()
        pairs = compute_mcs_pairs(smiles,
                                  np.asarray(data['labels'], dtype=np.float64),
                                  n_pairs=args.n_pairs, seed=SEED)
        print(f'    {len(pairs["I"])} pairs in {time.time() - tp:.1f}s')

    results = {
        'experiment': 'f18_distance_metric_sweep',
        'dataset': args.dataset,
        'mode': args.mode,
        'seed': SEED,
        'n_samples': int(len(data['labels'])),
        'cache_free': True,
        'representations_requested': wanted,
        'representation_build_failures': rep_failures,
        'roles': {
            'role_1_budget_split': 'representation ORDER -> see representation_rank_agreement',
            'role_2_config_wins_search': 'sac / readacross',
            'role_3_representation_comparator': 'sac / readacross',
            'role_4_reported_number': 'sac / readacross, plus struct_fid',
        },
    }
    results.update(run(data, reps, pairs, config, args))
    results['elapsed_seconds'] = round(time.time() - t0, 1)

    for rep, r in results['by_representation'].items():
        print(f'\n  [{rep}] {r["n_features"]}D {r["kind"]}')
        print(f'    {"coefficient":16s} {"sac":>9s} {"readacross":>11s} {"struct_fid":>11s}')
        for c, e in sorted(r['coefficients'].items(),
                           key=lambda kv: (kv[1]['readacross'] is None,
                                           -(kv[1]['readacross'] or 0))):
            def f(v, w=9):
                return f'{v:>{w}.4f}' if isinstance(v, float) else f'{"n/a":>{w}}'
            print(f'    {c:16s} {f(e["sac"])} {f(e["readacross"],11)} '
                  f'{f(e["struct_fid"],11)}')

    if results['n_skipped_cells']:
        reasons = {}
        for s in results['skipped_cells']:
            reasons.setdefault(s['reason'].split(':')[0], set()).add(
                f"{s['rep']}/{s['coefficient']}")
        print(f'\n  skipped {results["n_skipped_cells"]} cells:')
        for r, cells in reasons.items():
            print(f'    {r}: {sorted(cells)}')

    for kernel, ra in results['representation_rank_agreement'].items():
        pw = ra['pairwise_rank_correlation']
        if not pw:
            print(f'\n  representation ranking under {kernel}: '
                  f'needs >=3 representations, had {ra["n_representations"]}')
            continue
        vals = [v for v in pw.values() if v is not None]
        print(f'\n  representation ranking under {kernel} '
              f'({ra["n_representations"]} reps): pairwise rank correlation '
              f'min {min(vals):.3f}, median {float(np.median(vals)):.3f}')
        worst = min(pw.items(), key=lambda kv: (kv[1] is None, kv[1]))
        print(f'    most disagreement: {worst[0]} = {worst[1]:.3f}')

    save_experiment_results(results, args.output, 'f18_distance_metric_sweep',
                            dataset=args.dataset, metric_name=metric_name,
                            lower_is_better=lower_is_better)
    return results


if __name__ == '__main__':
    main()
