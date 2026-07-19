"""Learned / triplet similarity metric  (compass Stage 3, brief version).

QUESTION: does a LEARNED distance beat the coarse count-fingerprint Tanimoto
baseline on herg_ki, on BOTH yardsticks -- activity read-across (Spearman rho of
distance-weighted kNN predictions vs true activity) AND structural fidelity
(Spearman of pairwise distance vs MCS distance)?

METHOD (no leakage, scaffold-grouped, never random):
  * base fingerprint = ecfp4_count.
  * ONE scaffold split via GroupKFold -> train scaffolds / held-out scaffolds.
  * reuse learned_embeddings.train_triplet (pure-torch MLP-over-fingerprint deep
    metric learner) in BOTH triplet-definition modes, trained on TRAIN scaffolds
    only:
        structural -- positive = same Bemis-Murcko scaffold cluster (label-free)
        activity   -- positive = smaller |dy| (supervised on train labels)
  * embed ALL molecules with the trained encoder, build a COSINE distance in
    embedding space (simeval.dist_cosine).
  * score read-across on the HELD-OUT scaffolds (train molecules = reference
    pool, test molecules = queries) and MCS structural fidelity on the held-out
    pairs, vs the identical held-out evaluation of the count-FP Tanimoto baseline
    (simeval.dist_tanimoto(ecfp4_count)).

Baselines reported individually, never averaged. Reuses simeval / _data /
learned_embeddings unchanged.

  --quick : smoke on the first N molecules, few epochs, small embedding.
  (full)  : all 1415 herg_ki molecules, full epochs.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from _data import load_prepared                     # noqa: E402
import simeval                                       # noqa: E402
from learned_embeddings import train_triplet         # noqa: E402
from sklearn.model_selection import GroupKFold       # noqa: E402
from sklearn.neighbors import KNeighborsRegressor    # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402

BASE_REP = 'ecfp4_count'


def mcs_npz_path(dataset):
    return os.path.join(HERE, f'mcs_pairs_{dataset}.npz')


def heldout_readacross(D, y, train_idx, test_idx, k=5):
    """Distance-weighted kNN read-across: reference pool = train scaffolds,
    queries = held-out scaffolds. Scaffold-grouped by construction (the split
    keeps whole scaffolds on one side). Returns (spearman_rho, rmse)."""
    kk = min(k, len(train_idx))
    m = KNeighborsRegressor(n_neighbors=kk, weights='distance',
                            metric='precomputed')
    m.fit(D[np.ix_(train_idx, train_idx)], y[train_idx])
    pred = m.predict(D[np.ix_(test_idx, train_idx)])
    rho, _ = spearmanr(pred, y[test_idx])
    rmse = float(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))
    return (0.0 if not np.isfinite(rho) else float(rho)), rmse


def struct_fidelity(D, I, J, mcs):
    """Spearman of embedding distance vs MCS distance over held-out pairs.
    Higher = distance geometry better matches true structural (MCS) distance."""
    if len(I) < 5:
        return float('nan'), int(len(I))
    rho, _ = spearmanr(D[I, J], mcs)
    return (float('nan') if not np.isfinite(rho) else float(rho)), int(len(I))


def one_split(groups, n_splits, fold=0, seed=0):
    """Deterministic single scaffold-grouped split (fold-th of GroupKFold)."""
    n = len(groups)
    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(n)
    for i, (tr, te) in enumerate(gkf.split(np.arange(n), dummy_y, groups)):
        if i == fold:
            return tr, te
    raise RuntimeError('fold out of range')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki',
                    help='dataset name (needs mcs_pairs_<dataset>.npz)')
    ap.add_argument('--quick', action='store_true',
                    help='smoke: first N mols, few epochs, small dim')
    ap.add_argument('--n', type=int, default=500,
                    help='(quick only) number of leading molecules')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--dim', type=int, default=None)
    ap.add_argument('--n-trip', type=int, default=None)
    ap.add_argument('--n-splits', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    quick = args.quick
    epochs = args.epochs if args.epochs is not None else (8 if quick else 40)
    dim = args.dim if args.dim is not None else (64 if quick else 128)
    n_trip = args.n_trip if args.n_trip is not None else (8000 if quick else 20000)
    n_splits = args.n_splits if args.n_splits is not None else (4 if quick else 5)

    print(f'[cfg] quick={quick} epochs={epochs} dim={dim} n_trip={n_trip} '
          f'n_splits={n_splits} seed={args.seed}', flush=True)

    p = load_prepared(args.dataset)
    y = np.asarray(p['y'], dtype=np.float64)
    groups = np.asarray(p['groups'])
    X = np.asarray(p[BASE_REP], dtype=np.float32)
    n_full = len(y)

    if quick:
        N = min(args.n, n_full)
        sl = np.arange(N)
        y, groups, X = y[sl], groups[sl], X[sl]
        print(f'[data] SMOKE subset: first {N}/{n_full} {args.dataset} molecules, '
              f'{len(set(groups.tolist()))} scaffolds', flush=True)
    else:
        N = n_full
        print(f'[data] FULL {args.dataset}: {N} molecules, '
              f'{len(set(groups.tolist()))} scaffolds', flush=True)

    # --- scaffold split (single deterministic fold) ---
    train_idx, test_idx = one_split(groups, n_splits, fold=0, seed=args.seed)
    test_set = set(test_idx.tolist())
    print(f'[split] train={len(train_idx)} test(held-out)={len(test_idx)} '
          f'| train scaffolds={len(set(groups[train_idx].tolist()))} '
          f'test scaffolds={len(set(groups[test_idx].tolist()))}', flush=True)

    # --- MCS held-out pairs (indices are global; subset = leading rows so they
    #     coincide with local indices). Keep pairs with BOTH endpoints held-out.
    m = np.load(mcs_npz_path(args.dataset))
    gI, gJ, gmcs = m['I'], m['J'], m['mcs']
    inrange = (gI < N) & (gJ < N)
    gI, gJ, gmcs = gI[inrange], gJ[inrange], gmcs[inrange]
    heldpair = np.array([(i in test_set) and (j in test_set)
                         for i, j in zip(gI, gJ)])
    hI, hJ, hmcs = gI[heldpair], gJ[heldpair], gmcs[heldpair]
    print(f'[mcs] held-out pairs usable: {len(hI)} '
          f'(of {len(gI)} in-range)', flush=True)

    results = {}

    # --- baseline: count-FP Tanimoto ---
    D_base = simeval.dist_tanimoto(X)
    b_rho, b_rmse = heldout_readacross(D_base, y, train_idx, test_idx)
    b_srho, npairs = struct_fidelity(D_base, hI, hJ, hmcs)
    results['baseline_ecfp4_count_tanimoto'] = dict(
        readacross_rho=b_rho, readacross_rmse=b_rmse,
        struct_mcs_rho=b_srho, n_pairs=npairs)
    print(f'[baseline ecfp4_count/tanimoto] readacross_rho={b_rho:.4f} '
          f'rmse={b_rmse:.4f} | struct_mcs_rho={b_srho:.4f} (n={npairs})',
          flush=True)

    # --- learned triplet metrics ---
    for mode in ('structural', 'activity'):
        print(f'[train] triplet mode={mode} ...', flush=True)
        transform = train_triplet(
            X[train_idx], y[train_idx], groups[train_idx],
            mode=mode, dim=dim, epochs=epochs, n_trip=n_trip, seed=args.seed)
        emb = transform(X)
        D = simeval.dist_cosine(emb)
        r_rho, r_rmse = heldout_readacross(D, y, train_idx, test_idx)
        s_rho, _ = struct_fidelity(D, hI, hJ, hmcs)
        results[f'triplet_{mode}'] = dict(
            readacross_rho=r_rho, readacross_rmse=r_rmse,
            struct_mcs_rho=s_rho, n_pairs=npairs)
        print(f'[triplet {mode}] readacross_rho={r_rho:.4f} rmse={r_rmse:.4f} '
              f'| struct_mcs_rho={s_rho:.4f}', flush=True)

    # --- summary table ---
    print('\n==== SUMMARY (held-out scaffolds) ====', flush=True)
    hdr = f'{"method":32s} {"readacross_rho":>15s} {"struct_mcs_rho":>15s}'
    print(hdr)
    print('-' * len(hdr))
    for name, r in results.items():
        print(f'{name:32s} {r["readacross_rho"]:>15.4f} '
              f'{r["struct_mcs_rho"]:>15.4f}')

    base = results['baseline_ecfp4_count_tanimoto']
    print('\n==== DELTA vs count-FP baseline ====')
    for mode in ('structural', 'activity'):
        r = results[f'triplet_{mode}']
        print(f'triplet_{mode:12s}  d_readacross={r["readacross_rho"]-base["readacross_rho"]:+.4f}'
              f'   d_struct_mcs={r["struct_mcs_rho"]-base["struct_mcs_rho"]:+.4f}')
    return results


if __name__ == '__main__':
    main()
