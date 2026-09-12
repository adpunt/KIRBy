"""Learned / triplet similarity metric  (compass Stage 3, brief version).

QUESTION: does a LEARNED distance beat the coarse count-fingerprint Tanimoto
baseline on herg_ki, on BOTH yardsticks -- activity read-across (Spearman rho of
distance-weighted kNN predictions vs true activity) AND structural fidelity
(Spearman of pairwise distance vs MCS distance)?

METHOD (no leakage, scaffold-grouped, never random):
  * base fingerprint = ecfp4_count.
  * ONE scaffold split via GroupKFold -> train scaffolds / held-out scaffolds.
  * train a pure-torch MLP-over-fingerprint deep metric learner in BOTH
    triplet-definition modes, on TRAIN scaffolds only:
        structural -- positive = same Bemis-Murcko scaffold cluster (label-free)
        activity   -- positive = smaller |dy| (supervised on train labels)
    A FRESH set of triplets is drawn for every pass over the data (see
    train_triplet_resampled below), and the number that survive each pass is
    recorded in the results file.
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
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from _data import load_prepared                     # noqa: E402
import simeval                                       # noqa: E402
from _nanrank import nanmean, is_defined, n_defined   # noqa: E402
from sklearn.model_selection import GroupKFold       # noqa: E402
from sklearn.neighbors import KNeighborsRegressor    # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402

BASE_REP = 'ecfp4_count'


def mcs_npz_path(dataset):
    return os.path.join(HERE, f'mcs_pairs_{dataset}.npz')


def train_triplet_resampled(Xtr_base, ytr, groups_tr, mode='structural',
                            dim=128, epochs=40, margin=0.3, seed=0,
                            n_trip=20000, lr=1e-3):
    """Train the MLP metric learner, drawing a FRESH set of triplets per pass.

    Same encoder, loss and optimiser as learned_embeddings.train_triplet: an MLP
    of (number of fingerprint features) -> 512 -> 256 -> dim, TripletMarginLoss
    at `margin`, Adam at `lr`, batches of 2048 triplets. Trains on the training
    scaffolds only; the caller applies the returned encoder to every molecule.

    The one difference, and the reason this does not call
    learned_embeddings.train_triplet: that function draws its triplets once and
    reuses the same ones for all `epochs` passes, so the learned distance is
    fitted to one fixed set of a few thousand triples. Here each pass draws
    `n_trip` fresh candidate triples and filters them, so over `epochs` passes
    the encoder sees a much wider sample of the pair space. learned_embeddings
    is left as it is because run_battery_moleculeace.py also calls it.

    A candidate triple (anchor, p, q) survives when it can be ordered into a
    positive and a negative: in `structural` mode the positive must share the
    anchor's Bemis-Murcko scaffold and the negative must not, and most candidate
    triples fail that; in `activity` mode the positive is whichever of p and q
    has the smaller absolute activity gap to the anchor, so nearly all survive.

    Returns (transform, n_kept_per_pass) where transform maps a fingerprint
    matrix to its embedding and n_kept_per_pass is one surviving-triplet count
    per pass, `epochs` long.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(Xtr_base, dtype=torch.float32, device=dev)
    d = Xtr.shape[1]
    enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 256),
                        nn.ReLU(), nn.Linear(256, dim)).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    rs = np.random.RandomState(seed)
    n = len(ytr)

    def sample_triplets():
        a = rs.randint(0, n, n_trip)
        p = rs.randint(0, n, n_trip)
        q = rs.randint(0, n, n_trip)
        ok = (a != p) & (a != q) & (p != q)
        a, p, q = a[ok], p[ok], q[ok]
        if mode == 'structural':
            same_p = groups_tr[a] == groups_tr[p]
            same_q = groups_tr[a] == groups_tr[q]
            swap = (~same_p) & same_q            # want p2 same-scaffold, q2 not
            p2 = np.where(swap, q, p)
            q2 = np.where(swap, p, q)
            keep = ((groups_tr[a] == groups_tr[p2])
                    & (groups_tr[a] != groups_tr[q2]))
        else:                                     # activity: positive = smaller |dy|
            dyp = np.abs(ytr[a] - ytr[p])
            dyq = np.abs(ytr[a] - ytr[q])
            swap = dyq < dyp
            p2 = np.where(swap, q, p)
            q2 = np.where(swap, p, q)
            keep = np.abs(ytr[a] - ytr[p2]) < np.abs(ytr[a] - ytr[q2])
        return a[keep], p2[keep], q2[keep]

    trip = nn.TripletMarginLoss(margin=margin)
    bs = 2048
    n_kept_per_pass = []
    for _ in range(epochs):
        a_np, p_np, q_np = sample_triplets()
        n_kept_per_pass.append(int(len(a_np)))
        if len(a_np) == 0:
            continue                              # nothing to learn from this draw
        a = torch.tensor(a_np, device=dev)
        p = torch.tensor(p_np, device=dev)
        q = torch.tensor(q_np, device=dev)
        perm = torch.randperm(len(a), device=dev)
        for s in range(0, len(a), bs):
            idx = perm[s:s + bs]
            ea, ep_, eq = enc(Xtr[a[idx]]), enc(Xtr[p[idx]]), enc(Xtr[q[idx]])
            loss = trip(ea, ep_, eq)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def transform(Xb):
        with torch.no_grad():
            return enc(torch.tensor(Xb, dtype=torch.float32,
                                    device=dev)).cpu().numpy()
    return transform, n_kept_per_pass


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
    # NaN when the Spearman is undefined (constant kNN predictions, or fewer than
    # two held-out molecules). Never 0.0 -- that is the value of a distance with
    # no read-across signal at all, and the two must stay distinguishable.
    return (float(rho) if np.isfinite(rho) else float('nan')), rmse


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


def evaluate_fold(fold, y, groups, X, N, gI, gJ, gmcs,
                  dim, epochs, n_trip, n_splits, seed):
    """Run baseline + both triplet modes on a single scaffold fold.
    Returns a dict method -> {readacross_rho, readacross_rmse, struct_mcs_rho,
    n_pairs}. Read-across = held-out scaffolds as queries against train pool;
    structural fidelity = held-out MCS pairs only. No leakage: the encoder sees
    train scaffolds only, and both yardsticks are scored on held-out scaffolds."""
    train_idx, test_idx = one_split(groups, n_splits, fold=fold, seed=seed)
    test_set = set(test_idx.tolist())
    heldpair = np.array([(i in test_set) and (j in test_set)
                         for i, j in zip(gI, gJ)])
    hI, hJ, hmcs = gI[heldpair], gJ[heldpair], gmcs[heldpair]
    print(f'[fold {fold}] train={len(train_idx)} test={len(test_idx)} '
          f'| train scaffolds={len(set(groups[train_idx].tolist()))} '
          f'test scaffolds={len(set(groups[test_idx].tolist()))} '
          f'| held-out MCS pairs={len(hI)}', flush=True)

    res = {}
    D_base = simeval.dist_tanimoto(X)
    b_rho, b_rmse = heldout_readacross(D_base, y, train_idx, test_idx)
    b_srho, npairs = struct_fidelity(D_base, hI, hJ, hmcs)
    res['baseline_ecfp4_count_tanimoto'] = dict(
        readacross_rho=b_rho, readacross_defined=is_defined(b_rho),
        readacross_rmse=b_rmse,
        struct_mcs_rho=b_srho, struct_mcs_defined=is_defined(b_srho),
        n_pairs=npairs)
    print(f'  [fold {fold} baseline] readacross_rho={b_rho:.4f} rmse={b_rmse:.4f} '
          f'| struct_mcs_rho={b_srho:.4f} (n={npairs})', flush=True)

    for mode in ('structural', 'activity'):
        transform, kept = train_triplet_resampled(
            X[train_idx], y[train_idx], groups[train_idx],
            mode=mode, dim=dim, epochs=epochs, n_trip=n_trip, seed=seed)
        # kept = surviving triplets per pass, one entry per pass. Printed before
        # the scores so a reader of the log sees how many examples the encoder
        # actually trained on -- in structural mode that is a tiny fraction of
        # the n_trip candidate triples drawn each pass.
        print(f'  [fold {fold} triplet {mode}] surviving triplets per pass: '
              f'min={min(kept)} median={int(np.median(kept))} max={max(kept)} '
              f'sum={sum(kept)} over {len(kept)} passes, '
              f'out of {n_trip} candidate triples drawn per pass', flush=True)
        emb = transform(X)
        D = simeval.dist_cosine(emb)
        r_rho, r_rmse = heldout_readacross(D, y, train_idx, test_idx)
        s_rho, _ = struct_fidelity(D, hI, hJ, hmcs)
        res[f'triplet_{mode}'] = dict(
            readacross_rho=r_rho, readacross_defined=is_defined(r_rho),
            readacross_rmse=r_rmse,
            struct_mcs_rho=s_rho, struct_mcs_defined=is_defined(s_rho),
            n_pairs=npairs,
            n_triplets_per_pass=kept,
            n_candidate_triples_per_pass=int(n_trip))
        print(f'  [fold {fold} triplet {mode}] readacross_rho={r_rho:.4f} '
              f'rmse={r_rmse:.4f} | struct_mcs_rho={s_rho:.4f}', flush=True)
    return res


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
    ap.add_argument('--all-folds', action='store_true',
                    help='run EVERY scaffold fold and aggregate (full mode).')
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

    # --- MCS pairs (global indices; leading rows coincide with local indices). ---
    m = np.load(mcs_npz_path(args.dataset))
    gI, gJ, gmcs = m['I'], m['J'], m['mcs']
    inrange = (gI < N) & (gJ < N)
    gI, gJ, gmcs = gI[inrange], gJ[inrange], gmcs[inrange]
    print(f'[mcs] in-range pairs: {len(gI)}', flush=True)

    # In full mode run every scaffold fold; else the single deterministic fold 0.
    all_folds = args.all_folds or (not quick)
    folds = list(range(n_splits)) if all_folds else [0]
    print(f'[folds] running {folds}', flush=True)

    methods = ['baseline_ecfp4_count_tanimoto', 'triplet_structural',
               'triplet_activity']
    per_fold = {mth: {'readacross_rho': [], 'readacross_rmse': [],
                      'struct_mcs_rho': [], 'n_pairs': []} for mth in methods}

    # One list of surviving-triplet counts per fold, each as long as `epochs`.
    trip_counts = {'triplet_structural': [], 'triplet_activity': []}

    for fold in folds:
        res = evaluate_fold(fold, y, groups, X, N, gI, gJ, gmcs,
                            dim, epochs, n_trip, n_splits, args.seed)
        for mth in methods:
            for key in per_fold[mth]:
                per_fold[mth][key].append(res[mth][key])
        for mth in trip_counts:
            trip_counts[mth].append(res[mth]['n_triplets_per_pass'])

    def agg(vals):
        """(mean, std) over the folds whose value is finite. NaN for both when no
        fold is -- an undefined correlation is never entered as 0.0."""
        vals = list(vals)
        m = nanmean(vals)
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        return m, (float(np.std(v)) if len(v) else float('nan'))

    results = {}
    for mth in methods:
        ra_m, ra_s = agg(per_fold[mth]['readacross_rho'])
        rm_m, _ = agg(per_fold[mth]['readacross_rmse'])
        sc_m, sc_s = agg(per_fold[mth]['struct_mcs_rho'])
        # *_n_defined is the fold count each mean was actually taken over.
        results[mth] = dict(
            readacross_rho=ra_m, readacross_rho_std=ra_s,
            readacross_defined=is_defined(ra_m),
            readacross_n_defined=n_defined(per_fold[mth]['readacross_rho']),
            readacross_rmse=rm_m,
            struct_mcs_rho=sc_m, struct_mcs_rho_std=sc_s,
            struct_mcs_defined=is_defined(sc_m),
            struct_mcs_n_defined=n_defined(per_fold[mth]['struct_mcs_rho']),
            n_folds=len(folds),
            n_pairs_total=int(np.sum(per_fold[mth]['n_pairs'])),
            per_fold_readacross_rho=per_fold[mth]['readacross_rho'],
            per_fold_readacross_defined=[is_defined(x) for x in
                                         per_fold[mth]['readacross_rho']],
            per_fold_struct_mcs_rho=per_fold[mth]['struct_mcs_rho'],
            per_fold_struct_mcs_defined=[is_defined(x) for x in
                                         per_fold[mth]['struct_mcs_rho']])
        if mth in trip_counts:
            # How many triplets the encoder trained on. One row per fold, one
            # entry per pass over the data; a fresh draw every pass.
            flat = [c for fold_counts in trip_counts[mth] for c in fold_counts]
            results[mth].update(
                n_triplets_per_pass_by_fold=trip_counts[mth],
                n_candidate_triples_per_pass=int(n_trip),
                n_passes=int(epochs),
                n_triplets_per_pass_min=int(min(flat)) if flat else 0,
                n_triplets_per_pass_median=(int(np.median(flat)) if flat else 0),
                n_triplets_per_pass_max=int(max(flat)) if flat else 0,
                n_triplets_total_over_all_passes_and_folds=int(sum(flat)),
                triplets_resampled_every_pass=True)

    # --- summary table (mean +/- std across folds) ---
    print(f'\n==== SUMMARY {args.dataset} (mean over {len(folds)} scaffold '
          f'folds, held-out) ====', flush=True)
    hdr = (f'{"method":32s} {"readacross_rho":>22s} {"struct_mcs_rho":>22s}')
    print(hdr)
    print('-' * len(hdr))
    for name, r in results.items():
        print(f'{name:32s} '
              f'{r["readacross_rho"]:>10.4f}+/-{r["readacross_rho_std"]:<8.4f} '
              f'{r["struct_mcs_rho"]:>10.4f}+/-{r["struct_mcs_rho_std"]:<8.4f}'
              f'  (over {r["readacross_n_defined"]}/{r["n_folds"]} and '
              f'{r["struct_mcs_n_defined"]}/{r["n_folds"]} defined folds)')

    base = results['baseline_ecfp4_count_tanimoto']
    print('\n==== DELTA vs count-FP baseline (mean over folds) ====')
    for mode in ('structural', 'activity'):
        r = results[f'triplet_{mode}']
        print(f'triplet_{mode:12s}  '
              f'd_readacross={r["readacross_rho"]-base["readacross_rho"]:+.4f}'
              f'   d_struct_mcs={r["struct_mcs_rho"]-base["struct_mcs_rho"]:+.4f}')
    print('\n[per-fold readacross_rho]')
    for name, r in results.items():
        pf = ' '.join(f'{x:+.4f}' for x in r['per_fold_readacross_rho'])
        print(f'  {name:32s} {pf}')
    print('[per-fold struct_mcs_rho]')
    for name, r in results.items():
        pf = ' '.join(f'{x:+.4f}' for x in r['per_fold_struct_mcs_rho'])
        print(f'  {name:32s} {pf}')

    print('\n[surviving triplets per pass]')
    for mth in ('triplet_structural', 'triplet_activity'):
        r = results[mth]
        print(f'  {mth:32s} min={r["n_triplets_per_pass_min"]} '
              f'median={r["n_triplets_per_pass_median"]} '
              f'max={r["n_triplets_per_pass_max"]} '
              f'| {r["n_candidate_triples_per_pass"]} candidate triples drawn '
              f'per pass, {r["n_passes"]} passes x {len(folds)} folds')

    payload = dict(
        config=dict(
            dataset=args.dataset, quick=quick, n_molecules=int(N),
            n_molecules_full=int(n_full), base_rep=BASE_REP,
            epochs=int(epochs), dim=int(dim), n_trip=int(n_trip),
            n_splits=int(n_splits), folds=[int(f) for f in folds],
            seed=int(args.seed), margin=0.3, lr=1e-3,
            triplet_sampling='fresh draw every pass over the data',
            struct_fidelity_min_pairs=5),
        results=results,
        deltas={f'triplet_{mode}': dict(
            d_readacross_rho=(results[f'triplet_{mode}']['readacross_rho']
                              - base['readacross_rho']),
            d_struct_mcs_rho=(results[f'triplet_{mode}']['struct_mcs_rho']
                              - base['struct_mcs_rho']))
                for mode in ('structural', 'activity')},
    )
    tag = f'{args.dataset}_quick' if quick else args.dataset
    out_path = os.path.join(HERE, f'triplet_similarity_{tag}.json')
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=1)
    print(f'\nwrote {out_path}', flush=True)
    return payload


if __name__ == '__main__':
    main()
