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
                             score-fuse only the complementary survivors. The
                             keep order is by structural fidelity against MCS
                             distance, never by the activity gap A6 is scored on

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
from _nanrank import rank_desc, is_defined        # noqa: E402

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
    one with the higher STRUCTURAL fidelity), then score-fuse the survivors.

    The keep order is taken against MCS distance, not against the activity gap.
    Ordering by the activity gap meant the surviving set was chosen using the
    same gaps A6_dca is then scored on by read-across; MCS distance carries no
    label, and it is already loaded here as the structural yardstick. Which
    metric wins a near-duplicate pair can change; nothing else does, and on the
    two datasets run so far nothing is dropped at all -- the highest correlation
    between two metrics is 0.724, well under thr."""
    names = list(Ds)
    pv = {nm: Ds[nm][pairs['I'], pairs['J']] for nm in names}
    struct = {nm: abs(spearmanr(pv[nm], pairs['mcs'])[0]) for nm in names}
    order = sorted(names, key=lambda nm: -struct[nm])
    kept, dropped = [], []
    for nm in order:
        dup = [k for k in kept if abs(spearmanr(pv[nm], pv[k])[0]) > thr]
        if dup:
            dropped.append({'metric': nm, 'duplicate_of': dup[0]})
        else:
            kept.append(nm)
    fused = np.mean([_minmax(Ds[nm]) for nm in kept], axis=0)
    info = dict(order_by='struct_fidelity_vs_mcs', thr=thr,
                keep_order=order, dropped=dropped,
                struct_order_score={nm: float(struct[nm]) for nm in names},
                max_pair_corr=float(max(
                    abs(spearmanr(pv[a], pv[b])[0])
                    for i, a in enumerate(names) for b in names[i + 1:])))
    return fused, kept, info


AGGS = {'A1_score': a1_score_fusion, 'A2_meanrank': a2_mean_rank,
        'A3_rrf': a3_rrf, 'A4_borda': a4_borda, 'A5_consensus': a5_consensus}

# The three injection realisations simeval.noise_sweep_scores runs by default.
NOISE_SEEDS = (0, 1, 2)


def _mean_sd(v):
    """(mean, sd, n_defined) over the finite entries of v. An all-undefined cell
    returns NaN for both, never 0.0."""
    ok = np.isfinite(v)
    if not ok.any():
        return float('nan'), float('nan'), 0
    return float(v[ok].mean()), float(v[ok].std()), int(ok.sum())


def noise_grid(D, y, seeds=NOISE_SEEDS):
    """Detection AUROC and recovery correlation over simeval's full grid: three
    delivered noise levels by three corrupted fractions, each under `seeds`
    injections. One number from one injection at one dose cannot say whether a
    metric detects noise, because the ranking between metrics can invert between
    low and high dose.

    simeval.noise_sweep_scores averages the seeds inside a single call, which
    hides how far a cell moves between injections, so it is called once PER SEED
    and the mean and spread across the three are taken here. Same number of
    injections either way. Returns {'<sigma>|<rate>': {...}}, the key simeval
    already uses, where sigma is the delivered root-mean-square error in the
    label's own units and rate is the corrupted fraction of molecules."""
    per_seed = [simeval.noise_sweep_scores(D, y, seeds=(s,)) for s in seeds]
    grid = {}
    for cell in per_seed[0]:
        det = np.array([g[cell][0] for g in per_seed], float)
        rec = np.array([g[cell][1] for g in per_seed], float)
        dm, dsd, dn = _mean_sd(det)
        rm, rsd, rn = _mean_sd(rec)
        grid[cell] = dict(detect_mean=dm, detect_sd=dsd, detect_n_seeds=dn,
                          recovery_mean=rm, recovery_sd=rsd, recovery_n_seeds=rn,
                          n_seeds=len(seeds))
    return grid


def score(D, p, pairs, noise):
    # rank-fusion aggregators return a similarity-derived "distance" that can be
    # negative; kNN(metric='precomputed') requires non-negative. Shift to >= 0 --
    # a monotonic transform, so read-across / structural / noise rankings are all
    # unchanged (Spearman & nearest-neighbour order are shift-invariant).
    D = D - D.min()
    rho, negrmse = simeval.eval_knn_readacross(D, p['y'], p['groups'])
    struct = spearmanr(D[pairs['I'], pairs['J']], pairs['mcs'])[0]
    # The full dose x corrupted-fraction grid, three injections per cell. The
    # single realisation stays beside it under its own names, because the stored
    # aggregate_experiment.json was written from it; nothing should rank on a
    # one_injection number, so the two never share a name.
    grid = noise_grid(D, p['y'])
    det_cells = np.array([c['detect_mean'] for c in grid.values()], float)
    rec_cells = np.array([c['recovery_mean'] for c in grid.values()], float)
    det_m, det_sd, det_n = _mean_sd(det_cells)
    rec_m, rec_sd, rec_n = _mean_sd(rec_cells)
    yn, mask, eps = noise
    det1 = simeval.eval_noise_detect(D, yn, mask)
    rec1 = simeval.eval_noise_recovery(D, yn, mask, eps)
    # Neither correlation is folded to 0.0 when it is undefined: rho and struct
    # both stay NaN, and readacross_defined / struct_fidelity_defined are the
    # flags that separate an undefined correlation from a measured one of 0.
    # symmetric says whether D[i,j] equals D[j,i]. The four rank-fusion
    # aggregators rank neighbours WITHIN each row, so their fused distance is not
    # symmetric while every single metric's is; without this flag a later reader
    # compares a symmetric geometry against an asymmetric one without knowing it.
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                struct_fidelity=float(struct),
                struct_fidelity_defined=is_defined(struct),
                symmetric=bool(np.allclose(D, D.T)),
                noise_detect_sweep_mean=det_m, noise_detect_sweep_sd=det_sd,
                noise_detect_cells_defined=det_n,
                noise_recovery_sweep_mean=rec_m, noise_recovery_sweep_sd=rec_sd,
                noise_recovery_cells_defined=rec_n,
                noise_grid=grid, noise_grid_cells=len(grid),
                noise_grid_seeds=list(NOISE_SEEDS),
                noise_detect_one_injection=float(det1),
                noise_recovery_one_injection=float(rec1))


def run_dataset(name):
    t0 = time.time()
    p = load_prepared(name)
    from mcs_cache import load_pairs
    # Refuses a pair file drawn from a different molecule list. I and J are
    # positions into that list, so a rebuilt sim_<name>.npz silently re-pairs.
    d = load_pairs(name, p['smiles'])
    pairs = dict(I=d['I'], J=d['J'], mcs=d['mcs'], dy=d['dy'])
    # One injection at one dose, shared by every row. It feeds the
    # *_one_injection columns only; the headline noise numbers come from the
    # dose x corrupted-fraction grid in noise_grid, which each row runs itself.
    noise = simeval.inject_label_noise(p['y'], seed=0)
    Ds = build_metric_D(p)

    rows = []
    for nm, D in Ds.items():
        r = score(D, p, pairs, noise); r.update(kind='single', name=nm); rows.append(r)
    for anm, fn in AGGS.items():
        r = score(fn(Ds), p, pairs, noise); r.update(kind='agg', name=anm); rows.append(r)
    fused6, kept, dca_info = a6_dca(Ds, pairs)
    r = score(fused6, p, pairs, noise)
    r.update(kind='agg', name='A6_dca', kept=kept, dca=dca_info)
    rows.append(r)

    print(f'[{name}] {len(rows)} rows in {time.time()-t0:.0f}s; A6 kept {kept}', flush=True)
    return rows


def summarise(name, rows):
    singles = [r for r in rows if r['kind'] == 'single']
    # max() returns NaN whenever a NaN is the running maximum, so the best-single
    # pick goes through rank_desc, which places non-finite scores last.
    best = {y: max(singles, key=lambda r: rank_desc(r[y])) for y in
            ('readacross_rho', 'struct_fidelity',
             'noise_detect_sweep_mean', 'noise_recovery_sweep_mean')}
    print(f'\n===== {name}: aggregation vs best single metric =====')
    # The two noise columns are means over the 9 dose x corrupted-fraction cells,
    # each averaged over 3 injections; sd is the spread across those 9 cells.
    # sym is whether the distance is symmetric.
    hdr = (f'  {"method":14s} {"readacross":>10} {"struct":>8} {"noise_det":>9} '
           f'{"+-sd":>6} {"noise_rec":>9} {"+-sd":>6} {"sym":>5}')
    print(hdr)
    for r in rows:
        tag = '' if r['kind'] == 'single' else '*'
        print(f'  {tag}{r["name"]:13s} {r["readacross_rho"]:>10.3f} {r["struct_fidelity"]:>8.3f} '
              f'{r["noise_detect_sweep_mean"]:>9.3f} {r["noise_detect_sweep_sd"]:>6.3f} '
              f'{r["noise_recovery_sweep_mean"]:>9.3f} {r["noise_recovery_sweep_sd"]:>6.3f} '
              f'{"yes" if r["symmetric"] else "NO":>5}')
    for y, b in best.items():
        n_def = sum(1 for r in singles if is_defined(r[y]))
        print(f'  best single [{y}] = {b["name"]} ({b[y]:.3f}) '
              f'over {n_def}/{len(singles)} single metrics with a finite {y}')


def _json_safe(o):
    """Every non-finite float becomes null. json.dump writes NaN and Infinity as
    bare tokens, which no strict JSON reader accepts, so a file holding one
    undefined correlation could not be parsed at all. Nothing is lost: the
    *_defined flags beside each score are what separates an undefined
    correlation from a measured one of 0."""
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


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
    # allow_nan=False so a non-finite value that _json_safe missed raises here
    # rather than being written as a token the file's readers cannot parse.
    json.dump(_json_safe(out), open(path, 'w'), indent=1, allow_nan=False)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
