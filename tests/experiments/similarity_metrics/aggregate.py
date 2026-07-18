"""Experiment 2 -- metric aggregation.

The user is collecting similarity metrics and wants to COMBINE them. Question:
does fusing several COMPLEMENTARY metrics beat the best single metric -- on
activity read-across, on structural (MCS) fidelity, and on the noise battery?

Six aggregators (all implemented + benchmarked, never an averaged BASELINE):
  A1 score fusion        min-max each distance to [0,1], average
  A2 mean-rank fusion    average each metric's per-query neighbour ranks
  A3 RRF                 sum 1/(k+rank) -- Reciprocal Rank Fusion (IR standard)
  A4 Borda count         points by rank position, summed
  A5 consensus / vote    count how many metrics call a pair top-k similar
  A6 DCA redundancy-pruned   drop near-duplicate metrics (rank-corr > thr),
                             score-fuse only the complementary survivors

Fused over a COMPLEMENTARY metric set (DCA rule): a topological count FP, a 2D
atom-pair FP, physicochemical descriptors, substructure keys, torsions, and the
RDKit topological FP (a strong noise detector on logd in the pilot). MCS is the
structural YARDSTICK, not a fusion input (a full n x n MCS matrix is infeasible).

Every aggregate + every single metric is scored on all three yardsticks; the
comparison is aggregate vs the BEST SINGLE metric (reported per yardstick, never
an average-of-metrics baseline).
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simeval                                    # noqa: E402
from _data import load_prepared                   # noqa: E402

warnings.filterwarnings('ignore')

# complementary metric set (rep, coeff). DCA (A6) prunes redundancy inside this.
METRICS = [
    ('ecfp4_count', 'tanimoto'),   # topological, counts
    ('atompair_count', 'cosine'),  # 2D through-bond distances
    ('desc', 'euclidean'),         # physicochemical
    ('maccs', 'tanimoto'),         # substructure keys
    ('torsion', 'tanimoto'),       # torsions
    ('rdkit', 'tanimoto'),         # RDKit topological (pilot: good noise detector)
]


def build_metric_D(p):
    """Distance matrix per metric. desc is z-scored first (continuous)."""
    Ds = {}
    for rep, coeff in METRICS:
        X = p[rep].astype(np.float64)
        if rep == 'desc':
            mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
            X = (X - mu) / sd
        Ds[f'{rep}:{coeff}'] = getattr(simeval, f'dist_{coeff}')(X)
    return Ds


def _row_ranks(D):
    """R[i,j] = rank of j among row i's neighbours (0 = nearest, self forced last)."""
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    order = np.argsort(Dm, axis=1)
    R = np.empty_like(order)
    rows = np.arange(D.shape[0])[:, None]
    R[rows, order] = np.arange(D.shape[1])[None, :]
    return R.astype(np.float64)


def _minmax(D):
    off = ~np.eye(D.shape[0], dtype=bool)
    lo, hi = D[off].min(), D[off].max()
    return (D - lo) / (hi - lo + 1e-12)


# ---------- the six aggregators: dict of D -> fused D (smaller = closer) ----------
def a1_score_fusion(Ds):
    return np.mean([_minmax(D) for D in Ds.values()], axis=0)


def a2_mean_rank(Ds):
    return np.mean([_row_ranks(D) for D in Ds.values()], axis=0)


def a3_rrf(Ds, k=60):
    S = np.sum([1.0 / (k + _row_ranks(D)) for D in Ds.values()], axis=0)
    return -S


def a4_borda(Ds):
    n = next(iter(Ds.values())).shape[0]
    S = np.sum([(n - 1 - _row_ranks(D)) for D in Ds.values()], axis=0)
    return -S


def a5_consensus(Ds, k=20):
    votes = np.sum([(_row_ranks(D) < k).astype(float) for D in Ds.values()], axis=0)
    mr = a2_mean_rank(Ds)                       # tiny tiebreak so kNN is well-defined
    return -votes + 1e-6 * mr


def a6_dca(Ds, pairs, thr=0.9):
    """Drop near-duplicate metrics (Spearman of pair-distances > thr, keeping the
    higher-activity-fidelity one), then score-fuse the survivors."""
    names = list(Ds)
    pv = {nm: Ds[nm][pairs['I'], pairs['J']] for nm in names}
    act = {nm: abs(spearmanr(pv[nm], pairs['dy'])[0]) for nm in names}
    order = sorted(names, key=lambda nm: -act[nm])
    kept = []
    for nm in order:
        if all(abs(spearmanr(pv[nm], pv[k])[0]) <= thr for k in kept):
            kept.append(nm)
    fused = np.mean([_minmax(Ds[nm]) for nm in kept], axis=0)
    return fused, kept


AGGS = {'A1_score': a1_score_fusion, 'A2_meanrank': a2_mean_rank,
        'A3_rrf': a3_rrf, 'A4_borda': a4_borda, 'A5_consensus': a5_consensus}


def score(D, p, pairs, noise):
    # rank-fusion aggregators return a similarity-derived "distance" that can be
    # negative; kNN(metric='precomputed') requires non-negative. Shift to >= 0 --
    # a monotonic transform, so read-across / structural / noise rankings are all
    # unchanged (Spearman & nearest-neighbour order are shift-invariant).
    D = D - D.min()
    rho, negrmse = simeval.eval_knn_readacross(D, p['y'], p['groups'])
    struct = spearmanr(D[pairs['I'], pairs['J']], pairs['mcs'])[0]
    yn, mask, eps = noise
    det = simeval.eval_noise_detect(D, yn, mask)
    rec = simeval.eval_noise_recovery(D, yn, mask, eps)
    return dict(readacross_rho=float(rho), struct_fidelity=float(0 if not np.isfinite(struct) else struct),
                noise_detect=float(det), noise_recovery=float(rec))


def run_dataset(name):
    t0 = time.time()
    p = load_prepared(name)
    d = np.load(os.path.join(HERE, f'mcs_pairs_{name}.npz'))
    pairs = dict(I=d['I'], J=d['J'], mcs=d['mcs'], dy=d['dy'])
    noise = simeval.inject_label_noise(p['y'], seed=0)      # shared realisation
    Ds = build_metric_D(p)

    rows = []
    for nm, D in Ds.items():
        r = score(D, p, pairs, noise); r.update(kind='single', name=nm); rows.append(r)
    for anm, fn in AGGS.items():
        r = score(fn(Ds), p, pairs, noise); r.update(kind='agg', name=anm); rows.append(r)
    fused6, kept = a6_dca(Ds, pairs)
    r = score(fused6, p, pairs, noise); r.update(kind='agg', name='A6_dca', kept=kept); rows.append(r)

    print(f'[{name}] {len(rows)} rows in {time.time()-t0:.0f}s; A6 kept {kept}', flush=True)
    return rows


def summarise(name, rows):
    singles = [r for r in rows if r['kind'] == 'single']
    best = {y: max(singles, key=lambda r: r[y]) for y in
            ('readacross_rho', 'struct_fidelity', 'noise_detect', 'noise_recovery')}
    print(f'\n===== {name}: aggregation vs best single metric =====')
    hdr = f'  {"method":14s} {"readacross":>10} {"struct":>8} {"noise_det":>9} {"noise_rec":>9}'
    print(hdr)
    for r in rows:
        tag = '' if r['kind'] == 'single' else '*'
        print(f'  {tag}{r["name"]:13s} {r["readacross_rho"]:>10.3f} {r["struct_fidelity"]:>8.3f} '
              f'{r["noise_detect"]:>9.3f} {r["noise_recovery"]:>9.3f}')
    for y, b in best.items():
        print(f'  best single [{y}] = {b["name"]} ({b[y]:.3f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        rows = run_dataset(ds)
        out[ds] = rows
        summarise(ds, rows)
    path = os.path.join(HERE, 'aggregate_experiment.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
