"""creative_hybrids -- do STACKED hybrid similarity spaces beat a count fingerprint?

Roadmap 'CREATIVE HYBRIDS'. We build a similarity space by z-scoring and
concatenating heterogeneous BLOCKS, then measure the resulting distance geometry
on the two yardsticks this study uses:

  ACTIVITY  read-across  = scaffold-grouped kNN Spearman rho (simeval.eval_knn_readacross)
  STRUCTURE MCS fidelity = spearman( D[I,J], mcs )  over the precomputed MCS pairs

Blocks:
  coarse   = ecfp4_count        (the whole count fingerprint -- the baseline rep)
  fine-2D  = top-K features of [atompair_count | torsion], selected LABEL-FREE by
             |Spearman(|feature diff|, MCS)| over the MCS pairs (rankers._spearman_cols).
             No activity labels touch this selection, so read-across stays an honest test.
  physchem = desc               (RDKit physicochemical descriptors)
  keys     = maccs              (166-bit substructure keys)

A block is z-scored per feature. Two weighting schemes:
  equal_block : each block scaled by 1/sqrt(d_block) -> every block contributes
                equal TOTAL variance (so a 1024-d block does not swamp a 166-d one).
  coarse_down : equal_block, but the coarse count block additionally x0.5.
Distance on the assembled space is Euclidean (simeval.dist_euclidean).

THE BASELINE TO BEAT is ecfp4_count with Tanimoto (simeval.dist_tanimoto). We also
print count-alone as z-scored Euclidean so the block gains are separable from the
mere change of coefficient.

Question: does stacking fine-2D + physchem (+ keys) onto the coarse count block
beat count-alone on ACTIVITY, not just on structure?

  python creative_hybrids.py --quick        # smoke: first ~400 herg_ki mols, 1-3 min
  python creative_hybrids.py                 # full herg_ki (n=1415)
  python creative_hybrids.py --dataset logd  # (needs sim_logd.npz + mcs_pairs_logd.npz)
"""
import os
import sys
import argparse
import time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
import rankers
from _data import load_prepared


# ---------------------------------------------------------------- block algebra
def _zscore(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def make_D(blocks, weights=None, scheme='equal_block'):
    """Assemble a Euclidean distance matrix from a list of raw block matrices.

    scheme='equal_block' scales every block to unit total variance; scheme='flat'
    leaves block contribution proportional to its feature count. `weights` is an
    optional per-block multiplier applied on top (used for coarse-down-weighting).
    """
    parts = []
    for k, X in enumerate(blocks):
        Z = _zscore(X)
        if scheme == 'equal_block':
            Z = Z / np.sqrt(Z.shape[1])
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    M = np.hstack(parts)
    return simeval.dist_euclidean(M)


# --------------------------------------------------- fine-2D structural selection
def select_fine2d(X_fine, sel_I, sel_J, mcs_sel, top_k, seed=0):
    """Rank the columns of X_fine by |Spearman(|feature diff|, MCS)| over the
    supplied structural pairs (label-free) and return the indices of the top_k.
    Uses rankers._spearman_cols exactly as the ranker menu does."""
    DF = np.abs(X_fine[sel_I].astype(np.float64) - X_fine[sel_J].astype(np.float64))
    rho = rankers._spearman_cols(DF, mcs_sel)
    order = np.argsort(-np.abs(rho))
    return order[:top_k]


# ------------------------------------------------------------------------ scoring
def struct_fidelity(D, I, J, mcs):
    """Spearman between this space's pairwise distances and MCS distance."""
    return float(spearmanr(D[I, J], mcs)[0])


def score_space(name, D, y, groups, I, J, mcs):
    rho, negrmse = simeval.eval_knn_readacross(D, y, groups)
    sf = struct_fidelity(D, I, J, mcs)
    return dict(name=name, readacross_rho=float(rho), readacross_rmse=float(-negrmse),
                struct_fidelity=sf)


def run(dataset='herg_ki', quick=False, n_sub=400, fine_budgets=(50, 100, 250),
        default_k=100, seed=0):
    t0 = time.time()
    p = load_prepared(dataset)
    n_full = p['n']

    mcs_npz = np.load(os.path.join(HERE, f'mcs_pairs_{dataset}.npz'))
    I_all, J_all, mcs_all = mcs_npz['I'], mcs_npz['J'], mcs_npz['mcs']

    if quick:
        n_sub = min(n_sub, n_full)
        keep = np.arange(n_sub)
        pm = (I_all < n_sub) & (J_all < n_sub)
        I, J, mcs = I_all[pm], J_all[pm], mcs_all[pm]
    else:
        keep = np.arange(n_full)
        I, J, mcs = I_all, J_all, mcs_all

    y = p['y'][keep]
    groups = p['groups'][keep]
    reps = {r: p[r][keep] for r in ['ecfp4_count', 'atompair_count', 'torsion', 'desc', 'maccs']}
    coarse = reps['ecfp4_count']
    desc = reps['desc']
    maccs = reps['maccs']
    X_fine_all = np.hstack([reps['atompair_count'].astype(np.float64),
                            reps['torsion'].astype(np.float64)])

    n = len(y)
    print(f'[{dataset}] n={n} (quick={quick})  MCS pairs used={len(I)}  '
          f'fine-2D candidate dims={X_fine_all.shape[1]}', flush=True)

    # pre-select fine-2D feature sets for every budget (label-free, MCS pairs)
    fine_idx = {k: select_fine2d(X_fine_all, I, J, mcs, k, seed=seed)
                for k in sorted(set(list(fine_budgets) + [default_k]))}

    def fine(k):
        return X_fine_all[:, fine_idx[k]]

    rows = []
    # --- THE baseline: count fingerprint with Tanimoto ---
    rows.append(score_space('count_tanimoto[BASELINE]',
                            simeval.dist_tanimoto(coarse), y, groups, I, J, mcs))
    # --- count alone, z-scored Euclidean (isolates the coefficient change) ---
    rows.append(score_space('count_only[z-eucl]',
                            make_D([coarse]), y, groups, I, J, mcs))
    # --- progressive stacking (equal_block weighting), fine-2D at default_k ---
    rows.append(score_space('count+physchem',
                            make_D([coarse, desc]), y, groups, I, J, mcs))
    rows.append(score_space(f'count+fine2D(k={default_k})',
                            make_D([coarse, fine(default_k)]), y, groups, I, J, mcs))
    rows.append(score_space(f'count+fine2D(k={default_k})+physchem',
                            make_D([coarse, fine(default_k), desc]), y, groups, I, J, mcs))
    rows.append(score_space('all_blocks[equal_weight]',
                            make_D([coarse, fine(default_k), desc, maccs]),
                            y, groups, I, J, mcs))
    # --- weighting variant: coarse block down-weighted x0.5 ---
    rows.append(score_space('all_blocks[coarse_down x0.5]',
                            make_D([coarse, fine(default_k), desc, maccs],
                                   weights=[0.5, 1.0, 1.0, 1.0]),
                            y, groups, I, J, mcs))
    # --- BUDGET sweep on the fine-2D block (count + fine only) ---
    for k in fine_budgets:
        rows.append(score_space(f'BUDGET count+fine2D(k={k})',
                                make_D([coarse, fine(k)]), y, groups, I, J, mcs))

    base = rows[0]
    print(f'\n=== {dataset}: creative hybrid similarity spaces ===')
    print(f'  {"space":34s} {"readacross_rho":>14} {"rmse":>7} {"struct_MCS":>11}'
          f' {"d_rho_vs_base":>13}')
    for r in rows:
        d_rho = r['readacross_rho'] - base['readacross_rho']
        print(f'  {r["name"]:34s} {r["readacross_rho"]:>14.4f} {r["readacross_rmse"]:>7.3f}'
              f' {r["struct_fidelity"]:>11.4f} {d_rho:>+13.4f}')
    print(f'\n  baseline = {base["name"]}: readacross_rho={base["readacross_rho"]:.4f}, '
          f'struct_MCS={base["struct_fidelity"]:.4f}')
    print(f'  elapsed {time.time()-t0:.1f}s', flush=True)
    return dict(dataset=dataset, quick=quick, n=n, baseline=base, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-sub', type=int, default=400)
    args = ap.parse_args()
    run(dataset=args.dataset, quick=args.quick, n_sub=args.n_sub)


if __name__ == '__main__':
    main()
