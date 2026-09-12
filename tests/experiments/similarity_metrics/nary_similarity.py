"""nary_similarity.py -- extended n-ary similarity (compass brief 1.5).

Three ideas from the Miranda-Quintana / Lopez-Perez line of work, implemented and
then TESTED against the two local yardsticks: predicting each molecule's activity
from its five nearest neighbours, and MCS structural fidelity.

  (1) iSIM  -- "instant similarity" (Lopez-Perez, Miranda-Quintana, Digital
      Discovery 2024). The average pairwise similarity of a whole SET computed in
      O(N*d) from fingerprint column sums instead of the O(N^2) pairwise loop.
      Binary form uses the classical 1-1 / mismatch counters; a continuous form
      (dot-product pooled Tanimoto) handles COUNT fingerprints such as ecfp4_count.

  (2) COMPLEMENTARY similarity -- per molecule, the similarity of the set with
      that molecule REMOVED (Miranda-Quintana et al.). Because removing an outlier
      makes the remaining set more internally similar, a HIGH complementary
      similarity flags an OUTLIER. Computed for every molecule in O(N*d) via a
      leave-one-out update of the column sums (no N^2 recompute).

  (3) iCliff-style local ROUGHNESS -- per molecule, the structural-closeness-
      weighted mean activity gap to its nearest neighbours. High = a locally rough
      structure-activity landscape (activity-cliff character) around that molecule.
      IMPLEMENTED BELOW BUT NO LONGER SCORED -- see the note under TEST.

TEST (the real question):
  Does complementary similarity (computed on ecfp4_count) RANK the molecules
  whose activity disagrees with their neighbours -- target =
  |activity - kNN-median-activity| -- better than naive baselines (activity
  extremeness |y-median(y)|, structural isolation = mean distance)? And does a
  complementary-similarity re-weighting of the count-FP distance improve the
  prediction of each molecule's activity from its five nearest neighbours
  (Spearman rho) vs the plain count-FP baseline, without wrecking structural
  (MCS) fidelity?

  ROUGHNESS IS NOT IN THAT RANKING ANY MORE. local_roughness and
  neighbor_disagreement are both built from the same count-ECFP4 Tanimoto matrix
  D and the same activity vector y, and both are an activity gap to the
  neighbours D picks. Correlating one with the other restates its own input, so
  the number can only come out high. Row dropped 2026-09-11, repository owner's
  decision. The function is kept because it is one of the three ideas this file
  implements, and because other work may score it against an independent target.

Scaffold-grouped CV only (via simeval.eval_knn_readacross). Baseline reported
individually, never averaged. Nothing here is invented; citations inline.

USAGE
  python nary_similarity.py --quick            # smoke: first ~400 herg_ki mols, ~1-3 min
  python nary_similarity.py                     # full: herg_ki + logd
  python nary_similarity.py herg_ki logd cx     # explicit dataset list
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from _data import load_prepared          # noqa: E402
import simeval                            # noqa: E402
from _nanrank import is_defined            # noqa: E402

BASELINE_REP = 'ecfp4_count'              # the count-FP baseline to beat


# ==========================================================================
# (1) iSIM -- set-level n-ary similarity in O(N*d)
# ==========================================================================
def isim_binary(X):
    """n-ary Tanimoto of a SET from column popcounts (Lopez-Perez &
    Miranda-Quintana, Digital Discovery 2024). For column k let c_k be the number
    of molecules with that bit ON. 1-1 coincidences a = sum_k C(c_k,2); mismatches
    dis = sum_k c_k(N-c_k). n-ary Tanimoto = a / (a + dis). Count vectors are
    binarised (bit on iff count>0), matching the paper's binary derivation."""
    B = (np.asarray(X) > 0).astype(np.float64)
    N = B.shape[0]
    c = B.sum(0)
    a = float((c * (c - 1) / 2.0).sum())
    dis = float((c * (N - c)).sum())
    return a / (a + dis) if (a + dis) > 0 else 0.0


def isim_continuous(X):
    """Continuous (count-aware) pooled n-ary Tanimoto -- the ratio-of-sums form of
    the average pairwise continuous Tanimoto, computed from the column-sum vector.
    num = sum_{i<j}<xi,xj> = (||s||^2 - sum_i||xi||^2)/2 with s = sum_i x_i;
    den = (N-1)*sum_i||xi||^2 - num. O(N*d), no pairwise loop."""
    Xf = np.asarray(X, np.float64)
    N = Xf.shape[0]
    s = Xf.sum(0)
    sq = (Xf * Xf).sum(1)
    A = float(sq.sum())
    num = 0.5 * (float(s @ s) - A)
    den = (N - 1) * A - num
    return num / den if den > 0 else 0.0


# ==========================================================================
# (2) COMPLEMENTARY similarity -- per-molecule outlier score in O(N*d)
# ==========================================================================
def complementary_similarity(X):
    """csim_i = continuous n-ary Tanimoto of the set with molecule i REMOVED.
    Leave-one-out update of the column sums: s' = s - x_i, ||s'||^2 = ||s||^2
    - 2 x_i.s + ||x_i||^2, A' = A - ||x_i||^2, N' = N-1. HIGH csim_i => removing i
    makes the rest more similar => i is an OUTLIER. Returns (csim, outlierness)
    where outlierness is csim min-max scaled to [0,1]."""
    Xf = np.asarray(X, np.float64)
    N = Xf.shape[0]
    s = Xf.sum(0)
    ss = float(s @ s)
    sq = (Xf * Xf).sum(1)
    A = float(sq.sum())
    xs = Xf @ s                                   # x_i . s
    s2p = ss - 2.0 * xs + sq                       # ||s - x_i||^2
    Ap = A - sq
    Np = N - 1
    num = 0.5 * (s2p - Ap)
    den = (Np - 1) * Ap - num
    csim = np.where(den > 0, num / den, 0.0)
    rng = csim.max() - csim.min()
    outlierness = (csim - csim.min()) / rng if rng > 0 else np.zeros_like(csim)
    return csim, outlierness


# ==========================================================================
# (3) iCliff-style local roughness per molecule
# ==========================================================================
def local_roughness(D, y, k=5):
    """Per molecule: structural-closeness-weighted mean activity gap to its k
    nearest neighbours in distance matrix D. weight_j = 1/(d_ij+eps); returns
    sum_j w_j |y_i - y_j| / sum_j w_j. High = locally rough SAR landscape (an
    activity-cliff signature around molecule i), in the iCliff spirit of coupling
    structural similarity with activity difference.

    NOT scored against neighbor_disagreement below: given the same D and the
    same y, the two are the same quantity and the correlation restates its own
    input."""
    idx, dist = simeval._knn_idx_dist(D, k)
    w = 1.0 / (dist + 1e-9)
    dyk = np.abs(y[:, None] - y[idx])
    w = w / w.sum(1, keepdims=True)
    return (w * dyk).sum(1)


def neighbor_disagreement(D, y, k=5):
    """TARGET (ground truth outlier/cliff-ness): |y_i - median activity of i's k
    structural nearest neighbours|. A molecule whose measured activity disagrees
    with its structural neighbourhood -- exactly what an outlier/cliff detector
    should rank highly."""
    idx = simeval._knn_idx(D, k)
    return np.abs(y - np.median(y[idx], axis=1))


# ==========================================================================
# yardsticks
# ==========================================================================
def structural_fidelity(D, I, J, mcs):
    """Spearman(D over the MCS-sampled pairs, MCS distance). Higher = distances
    mirror shared substructure better.

    NaN when the Spearman is undefined (a constant distance over the sampled
    pairs, or fewer than two pairs). Never 0.0 -- that is the value of a
    distance whose ordering has nothing to do with shared substructure, and the
    two must stay distinguishable. The struct_fidelity_*_defined flags in the
    results file say which of the two a reader is looking at."""
    r = spearmanr(D[I, J], mcs)[0]
    return float(r) if np.isfinite(r) else float('nan')


def load_mcs(name, keep_lim=None):
    """MCS pair ground truth for `name`, aligned to load_prepared(name)'s array.
    keep_lim (smoke) keeps only pairs with both indices < keep_lim."""
    f = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    if not os.path.exists(f):
        return None
    m = np.load(f)
    I, J, mcs = m['I'].astype(int), m['J'].astype(int), m['mcs'].astype(float)
    if keep_lim is not None:
        k = (I < keep_lim) & (J < keep_lim)
        I, J, mcs = I[k], J[k], mcs[k]
    return I, J, mcs


# ==========================================================================
# experiment
# ==========================================================================
def run(name, quick=False, k=5, lam=1.0):
    t0 = time.time()
    p = load_prepared(name)
    y, groups = p['y'], p['groups']
    X = p[BASELINE_REP]

    keep_lim = None
    if quick:
        n_sub = min(400, len(y))
        sel = np.arange(n_sub)                     # first N mols -> indices unchanged
        X, y, groups = X[sel], y[sel], groups[sel]
        keep_lim = n_sub

    n = len(y)
    # baseline count-FP Tanimoto distance
    D = simeval.dist_tanimoto(X)

    # --- set-level iSIM (sanity: O(N) matches the mean of pairwise) ---
    isim_c = isim_continuous(X)
    isim_b = isim_binary(X)

    # --- per-molecule scores ---
    csim, outlierness = complementary_similarity(X)
    target = neighbor_disagreement(D, y, k=k)
    # local_roughness is deliberately NOT computed here. It and `target` are
    # both an activity gap to the neighbours this same D picks, out of this same
    # y, so scoring one against the other restates its own input.

    # naive baselines for the outlier/cliff-ranking test
    naive_activity = np.abs(y - np.median(y))      # activity extremeness (no structure)
    naive_isolation = D.mean(1)                     # mean structural distance to all

    def sp(a, b):
        # NaN when the Spearman is undefined (a constant input vector). Never
        # 0.0 -- that is the value of a score that ranks the target no better
        # than chance, and the two must stay distinguishable.
        r = spearmanr(a, b)[0]
        return float(r) if np.isfinite(r) else float('nan')

    outlier_corr = {
        'complementary_sim': sp(csim, target),
        'naive_activity_extremeness': sp(naive_activity, target),
        'naive_structural_isolation': sp(naive_isolation, target),
    }
    outlier_corr_defined = {kk: is_defined(v) for kk, v in outlier_corr.items()}

    # --- baseline vs complementary-similarity re-weighting, scored by
    # --- predicting each molecule's activity from its five nearest neighbours
    # inflate the distance of every pair that involves an outlier, so outlier
    # neighbours are down-weighted / dropped.
    #
    # SYMMETRY: the inflation is the MEAN outlierness of the two molecules in
    # the pair, so Dw[i,j] == Dw[j,i]. The earlier form multiplied column j
    # only, which made Dw asymmetric: sklearn's precomputed-metric kNN is fitted
    # on Dw[train, train] and assumes that matrix is symmetric, and
    # structural_fidelity reads one arbitrary triangle of it, so the two
    # yardsticks were reading two different matrices.
    pair_outlierness = 0.5 * (outlierness[:, None] + outlierness[None, :])
    Dw = D * (1.0 + lam * pair_outlierness)
    base_rho, base_negrmse = simeval.eval_knn_readacross(D, y, groups, k=k)
    wt_rho, wt_negrmse = simeval.eval_knn_readacross(Dw, y, groups, k=k)

    # --- structural fidelity yardstick ---
    mcs = load_mcs(name, keep_lim=keep_lim)
    if mcs is not None and len(mcs[0]) >= 20:
        I, J, mvals = mcs
        base_struct = structural_fidelity(D, I, J, mvals)
        wt_struct = structural_fidelity(Dw, I, J, mvals)
        n_pairs = int(len(mvals))
    else:
        base_struct = wt_struct = float('nan')
        n_pairs = 0

    return dict(
        name=name, n=n, quick=quick, seconds=round(time.time() - t0, 1),
        # Every setting this run used, so the numbers below can be read without
        # the command line that produced them.
        params=dict(
            baseline_rep=BASELINE_REP,
            k_neighbours_target=k,
            k_neighbours_prediction=k,
            lam=lam,
            readacross_n_splits=5,          # simeval.eval_knn_readacross default
            weighting='D * (1 + lam * mean outlierness of the pair)',
            outlierness_scaling=('complementary similarity min-max scaled over '
                                 'every molecule in the dataset, including the '
                                 'molecules held out by each scaffold fold'),
            roughness_scored=False,
        ),
        isim_continuous=isim_c, isim_binary=isim_b,
        outlier_corr=outlier_corr,
        outlier_corr_defined=outlier_corr_defined,
        # Both rho values stay NaN when their Spearman is undefined; the
        # *_defined flags separate that from a measured rho of 0, and the
        # weighted-minus-baseline delta printed below is then NaN too.
        readacross_baseline_rho=float(base_rho),
        readacross_baseline_defined=is_defined(base_rho),
        readacross_weighted_rho=float(wt_rho),
        readacross_weighted_defined=is_defined(wt_rho),
        readacross_baseline_negrmse=float(base_negrmse),
        readacross_weighted_negrmse=float(wt_negrmse),
        struct_fidelity_baseline=base_struct,
        struct_fidelity_baseline_defined=is_defined(base_struct),
        struct_fidelity_weighted=wt_struct,
        struct_fidelity_weighted_defined=is_defined(wt_struct),
        n_mcs_pairs=n_pairs,
    )


def _fmt(r):
    L = []
    L.append(f"\n=== {r['name']}  n={r['n']}  ({'QUICK' if r['quick'] else 'FULL'}, {r['seconds']}s) ===")
    L.append(f"  iSIM (set cohesion)      continuous={r['isim_continuous']:.4f}  binary={r['isim_binary']:.4f}")
    L.append("  OUTLIER / CLIFF RANKING  (Spearman vs |activity - kNN-median-activity|; higher=better):")
    L.append("      roughness_iCliff dropped: same distance matrix and same activity as the target.")
    oc = r['outlier_corr']
    ocd = r.get('outlier_corr_defined', {})
    for kk in ['complementary_sim',
               'naive_activity_extremeness', 'naive_structural_isolation']:
        tag = '  <- method' if kk == 'complementary_sim' else '  (naive)'
        und = '' if ocd.get(kk, True) else '   [UNDEFINED]'
        L.append(f"      {kk:30s} {oc[kk]:+.4f}{tag}{und}")
    L.append("  NEIGHBOUR PREDICTION (Spearman rho of the 5-nearest-neighbour activity"
             " estimate, scaffold GroupKFold):")
    L.append(f"      count-FP baseline            rho={r['readacross_baseline_rho']:+.4f}  (-rmse={r['readacross_baseline_negrmse']:+.4f})")
    L.append(f"      complementary-sim weighted   rho={r['readacross_weighted_rho']:+.4f}  (-rmse={r['readacross_weighted_negrmse']:+.4f})")
    d = r['readacross_weighted_rho'] - r['readacross_baseline_rho']
    both = r['readacross_baseline_defined'] and r['readacross_weighted_defined']
    L.append(f"      delta rho (weighted-baseline) = {d:+.4f}"
             f"{'' if both else '   [one or both rho UNDEFINED]'}")
    L.append(f"  STRUCTURAL FIDELITY (Spearman vs MCS over {r['n_mcs_pairs']} pairs; higher=better):")
    bu = '' if r.get('struct_fidelity_baseline_defined', True) else '   [UNDEFINED]'
    wu = '' if r.get('struct_fidelity_weighted_defined', True) else '   [UNDEFINED]'
    L.append(f"      count-FP baseline            {r['struct_fidelity_baseline']:+.4f}{bu}")
    L.append(f"      complementary-sim weighted   {r['struct_fidelity_weighted']:+.4f}{wu}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--quick', action='store_true', help='smoke: first ~400 herg_ki mols')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--lam', type=float, default=1.0, help='outlier down-weight strength')
    args = ap.parse_args()

    datasets = ['herg_ki'] if args.quick else args.datasets
    out = {}
    for ds in datasets:
        try:
            r = run(ds, quick=args.quick, k=args.k, lam=args.lam)
            out[ds] = r
            print(_fmt(r), flush=True)
        except Exception as e:
            import traceback
            print(f"[{ds}] FAILED: {e}", flush=True)
            traceback.print_exc()
            out[ds] = {'error': str(e)}

    tag = 'quick' if args.quick else 'full'
    fp = os.path.join(HERE, f'nary_similarity_{tag}.json')
    json.dump(out, open(fp, 'w'), indent=1)
    print(f"\nwrote {fp}")


if __name__ == '__main__':
    main()
