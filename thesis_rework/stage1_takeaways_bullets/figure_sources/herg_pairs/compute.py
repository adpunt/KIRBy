"""Paragraph 6 numbers for "Flaws in similarity and other pairwise computations".

Reads ONLY cached feature matrices (rep_cache/<ds>/<rep>.npy) and meta_<ds>.npz.
Builds no representation. One representation at a time, float32.

Per dataset x representation (the general pool, plus ecfp4 as the cliff reference):
  (a) all-pairs Spearman between pair distance and |label difference|
      (fixed random sample of 2,000 molecules, seed 0, when N > 2,000)
  (b) five-nearest-neighbour prediction, neighbours restricted to other
      scaffold groups: Spearman(pred, label), RMSE in label units
  cliffs: ECFP4 Tanimoto >= 0.55 and the activity gap solved so 15% of
      molecules are cliffs (similarity_helpers.calibrated_cliff_rule);
      also the older 2 x MAD gap; and, as an EXTENSION (not the thesis rule),
      each representation's own closest pairs, as many as ECFP4 calls similar,
      with the ECFP4-solved gap.
Saves 5-NN indices per rep for the overlap step, and for herg_ki the pair
distances for the figure.

usage: python compute.py <dataset> [rep ...]
"""
import json, os, sys, time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests/experiments/complexity')
from complexity_helpers import DATASET_REPS  # builds nothing

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = '/Volumes/seagate/kirby_cache/rep_cache'
META = '/Users/apunt/repos/KIRBy/scratchpad/thesis_code_todo'
SAMPLE_N = 2000
CHUNK = 500
TAN_CUT = 0.55
TARGET_PCT = 15.0


def is_binary(X):
    for s in range(0, X.shape[0], 2000):
        A = np.asarray(X[s:s + 2000])
        if not np.all((A == 0) | (A == 1)):
            return False
    return True


def full_distance(X, kind):
    n = X.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    if kind == 'binary':
        B = np.asarray(X, dtype=np.float32)
        cnt = B.sum(1)
        for s in range(0, n, CHUNK):
            A = B[s:s + CHUNK]
            inter = A @ B.T
            union = cnt[s:s + CHUNK, None] + cnt[None, :] - inter
            with np.errstate(invalid='ignore', divide='ignore'):
                sim = np.where(union > 0, inter / union, 1.0)
            D[s:s + CHUNK] = 1.0 - sim
        del B
        info = {'distance': 'Tanimoto distance (1 - T) on 0/1 features'}
    else:
        A = np.asarray(X, dtype=np.float64)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        sd = A.std(0)
        keep = sd > 0
        Z = (A[:, keep] - A[:, keep].mean(0)) / sd[keep]
        del A
        for s in range(0, n, CHUNK):
            D[s:s + CHUNK] = cdist(Z[s:s + CHUNK], Z, 'cityblock')
        info = {'distance': 'Manhattan on standardised features (sd > 0 only)',
                'n_features_used': int(keep.sum())}
        del Z
    np.fill_diagonal(D, 0.0)
    return D, info


def upper(D, idx=None):
    if idx is not None:
        D = D[np.ix_(idx, idx)]
    iu = np.triu_indices(D.shape[0], 1)
    return D[iu]


def knn_other_groups(D, groups, k=5):
    n = D.shape[0]
    nbr = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, CHUNK):
        R = D[s:s + CHUNK].copy()
        same = groups[s:s + CHUNK, None] == groups[None, :]
        R[same] = np.inf
        part = np.argpartition(R, k, axis=1)[:, :k]
        # order the k by distance, ties broken by index
        rows = np.arange(R.shape[0])[:, None]
        o = np.argsort(R[rows, part], axis=1, kind='stable')
        nbr[s:s + CHUNK] = part[rows, o]
        assert np.all(np.isfinite(R[rows, part])), 'fewer than 5 molecules outside own group'
    return nbr


def pair_list_upper(mask_fn, D, n):
    """(i, j) with i < j where mask_fn(D rows) is True, chunked."""
    I, J = [], []
    for s in range(0, n, CHUNK):
        M = mask_fn(D[s:s + CHUNK])
        ii, jj = np.nonzero(M)
        ii = ii + s
        keep = jj > ii
        I.append(ii[keep]); J.append(jj[keep])
    return np.concatenate(I), np.concatenate(J)


def solve_gap(I, J, y, n, target_pct, n_steps=80):
    dy = np.abs(y[I] - y[J])

    def pct(gap):
        m = dy > gap
        return 100.0 * np.unique(np.concatenate([I[m], J[m]])).size / n
    lo, hi = 0.0, float(np.max(np.abs(y[:, None] - y[None, :]))) if n <= 3000 else float(y.max() - y.min())
    best = (lo, pct(lo))
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        p = pct(mid)
        if abs(p - target_pct) < abs(best[1] - target_pct):
            best = (mid, p)
        if p > target_pct:
            lo = mid
        else:
            hi = mid
    return float(best[0]), float(best[1])


def cliff_molecules(I, J, y, gap):
    m = np.abs(y[I] - y[J]) > gap
    return np.unique(np.concatenate([I[m], J[m]]))


def main():
    ds = sys.argv[1]
    pool = list(DATASET_REPS[ds]['general'])
    only = sys.argv[2:]
    meta = np.load(f'{META}/meta_{ds}.npz', allow_pickle=True)
    y = np.asarray(meta['labels'], dtype=np.float64)
    groups = np.asarray(meta['groups'])
    n = len(y)
    rng = np.random.RandomState(0)
    S = np.sort(rng.choice(n, SAMPLE_N, replace=False)) if n > SAMPLE_N else np.arange(n)
    dy_S = upper(np.abs(y[:, None] - y[None, :]).astype(np.float32), S) if n <= 3000 else \
        np.abs(y[S][:, None] - y[S][None, :]).astype(np.float32)[np.triu_indices(len(S), 1)]

    out_path = f'{HERE}/results_{ds}.json'
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res['dataset'] = ds
    res['n_molecules'] = n
    res['n_scaffold_groups'] = int(np.unique(groups).size)
    res['pool'] = pool
    res['allpairs_sample_molecules'] = int(len(S))
    res['allpairs_n_pairs'] = int(len(S) * (len(S) - 1) // 2)
    res['label_sd'] = float(y.std())
    res['mad_y'] = float(np.median(np.abs(y - np.median(y))))
    res.setdefault('reps', {})

    reps = ['ecfp4'] + [r for r in pool if r != 'ecfp4']
    if only:
        reps = [r for r in reps if r in only]
    for rep in reps:
        t0 = time.time()
        X = np.load(f'{CACHE}/{ds}/{rep}.npy', mmap_mode='r')
        assert X.shape[0] == n, (rep, X.shape, n)
        kind = 'binary' if is_binary(X) else 'nonbinary'
        D, info = full_distance(X, kind)
        del X
        r = {'in_pool': rep in pool, 'n_features': int(np.load(f'{CACHE}/{ds}/{rep}.npy', mmap_mode='r').shape[1]),
             'kind': kind, **info}
        # (a)
        dS = upper(D, S)
        r['allpairs_spearman'] = float(spearmanr(dS, dy_S).correlation)
        if ds == 'herg_ki' and rep in ('ecfp4', 'mordred'):
            np.save(f'{HERE}/herg_ki_pairdist_{rep}.npy', dS)
            np.save(f'{HERE}/herg_ki_pair_dy.npy', dy_S)
        del dS
        # (b)
        nbr = knn_other_groups(D, groups)
        np.save(f'{HERE}/knn5_{ds}_{rep}.npy', nbr)
        pred = y[nbr].mean(1)
        r['knn5_spearman'] = float(spearmanr(pred, y).correlation)
        r['knn5_rmse'] = float(np.sqrt(np.mean((pred - y) ** 2)))
        r['knn5_n_molecules'] = int(n)
        # cliffs
        if rep == 'ecfp4':
            I, J = pair_list_upper(lambda R: (1.0 - R) >= TAN_CUT, D, n)
            gap, ach = solve_gap(I, J, y, n, TARGET_PCT)
            cm = cliff_molecules(I, J, y, gap)
            cm2 = cliff_molecules(I, J, y, 2.0 * res['mad_y'])
            res['cliff_rule'] = {
                'similar': 'ECFP4 Tanimoto >= 0.55', 'target_pct_molecules': TARGET_PCT,
                'n_similar_pairs': int(len(I)), 'activity_gap': gap,
                'activity_gap_over_mad': gap / res['mad_y'] if res['mad_y'] > 0 else None,
                'n_cliff_molecules': int(cm.size), 'pct_cliff_molecules': 100.0 * cm.size / n,
                'mad_rule_gap': 2.0 * res['mad_y'],
                'mad_rule_n_cliff_molecules': int(cm2.size),
                'mad_rule_pct_cliff_molecules': 100.0 * cm2.size / n,
                'n_cliff_pairs': int(np.sum(np.abs(y[I] - y[J]) > gap)),
            }
            np.savez(f'{HERE}/ecfp4_similar_pairs_{ds}.npz', I=I, J=J)
        cr = res['cliff_rule']
        P = cr['n_similar_pairs']
        # EXTENSION: this rep's P closest pairs, same activity gap
        if P > 0:
            u = np.empty(n * (n - 1) // 2, dtype=np.float32)
            pos = 0
            for i in range(n - 1):
                u[pos:pos + n - 1 - i] = D[i, i + 1:]
                pos += n - 1 - i
            thr = float(np.partition(u, P - 1)[P - 1])
            del u
            I2, J2 = pair_list_upper(lambda R: R <= thr, D, n)
            cm = cliff_molecules(I2, J2, y, cr['activity_gap'])
            E = np.load(f'{HERE}/ecfp4_similar_pairs_{ds}.npz')
            a = set(zip(E['I'].tolist(), E['J'].tolist()))
            b = set(zip(I2.tolist(), J2.tolist()))
            r['matched_similar_pairs'] = int(len(I2))
            r['matched_distance_cutoff'] = thr
            r['matched_n_cliff_molecules'] = int(cm.size)
            r['matched_pct_cliff_molecules'] = 100.0 * cm.size / n
            r['frac_ecfp4_similar_pairs_also_closest_here'] = len(a & b) / len(a)
        del D
        r['seconds'] = round(time.time() - t0, 1)
        res['reps'][rep] = r
        json.dump(res, open(out_path, 'w'), indent=1)
        print(ds, rep, {k: v for k, v in r.items() if isinstance(v, (int, float))}, flush=True)


if __name__ == '__main__':
    main()
