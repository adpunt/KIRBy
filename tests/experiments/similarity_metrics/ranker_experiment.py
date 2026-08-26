"""Experiment 1 -- the feature-ranker program.

Question: does a SIMILARITY-AWARE feature ranker, used to pick which features go
into a hybrid similarity space, beat the standard count fingerprint?

This is the untested lever behind why the pilot's search-objective swap was flat:
the objective only controls ALLOCATION (how the budget splits across reps), but
the feature RANKER stayed predictive, so the search never actually selected
features *for* similarity. Here we swap the ranker itself.

Design (settled with the user 2026-07-18):
  * Pool = 4 COMPLEMENTARY reps (ecfp4_count, atompair_count, desc, maccs) -- the
    DCA "fuse complementary, not redundant" rule; NOT three flavours of the same
    circular FP. ~2423 features.
  * Feature COUNT comes from the ALLOCATION knob, not an invented sweep: per-rep
    budget n_per_rep (default 100) x top-N reps kept. The ranker only decides
    WHICH features inside each surviving rep (hybrid.py:694 splits total_budget
    across reps; the ranker replaces compute_feature_importance).
  * Rankers, each run (where meaningful) in BOTH target modes:
       activity   target = |dy|   (the normal, supervised target)
       structural target = MCS    (label-free; the genuinely different one)
    -- the activity-vs-structural selector comparison is the novel headline.
  * Three tiers: (A) hand-rolled similarity-aware rankers R1-R7; (B) set
    selectors R8 (Lasso stand-in; pyHSICLasso absent) + R9 greedy-forward;
    (C) an off-the-shelf scikit-learn feature-selection tier, so we can see
    whether the bespoke rankers beat what sklearn gives for free.
  * All pair-based rankers share the SAME cached MCS pairs (mcs_cache.py) so
    activity and structural modes are perfectly paired.
  * Selected-feature distance: z-scored columns -> cosine AND euclidean.
  * Yardsticks: (1) activity read-across rho (simeval.eval_knn_readacross,
    scaffold GroupKFold -- no random splits); (2) structural fidelity = Spearman
    between the space's pairwise distance and MCS over the cached pairs.
  * Baselines (NEVER averaged): ecfp4_count:tanimoto (pilot's cheap winner) and
    ecfp4:tanimoto (field standard), each on both yardsticks.

Does NOT touch hybrid.py / find_best_hybrid.py.
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simeval                                    # noqa: E402
import rankers                                    # noqa: E402
from _data import load_prepared, POOL_REPS        # noqa: E402
from _nanrank import rank_desc, is_defined        # noqa: E402

warnings.filterwarnings('ignore')


# ===========================================================================
# pooled feature matrix + per-rep column slices
# ===========================================================================
def build_pool(p):
    """Concatenate the 4 pool reps; return (Xz z-scored, slices dict, names)."""
    cols, slices, start = [], {}, 0
    for r in POOL_REPS:
        X = p[r].astype(np.float64)
        slices[r] = (start, start + X.shape[1])
        start += X.shape[1]
        cols.append(X)
    X = np.concatenate(cols, axis=1)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    return Xz, slices


# ===========================================================================
# cached MCS pairs -> pair-difference design matrix
# ===========================================================================
def load_pairs(name, Xz):
    d = np.load(os.path.join(HERE, f'mcs_pairs_{name}.npz'))
    I, J, mcs, dy = d['I'], d['J'], d['mcs'], d['dy']
    DF = np.abs(Xz[I] - Xz[J]).astype(np.float32)      # (m, d) pair |feature diff|
    return dict(I=I, J=J, mcs=mcs, dy=dy, DF=DF)


# ===========================================================================
# TIER A/C per-feature SCORERS  (return (d,) score, higher = keep)
# every pair-based scorer takes (DF, target) on the shared cached pairs
# ===========================================================================
def sc_spearman(DF, t, **kw):                       # R1/R2
    return rankers._spearman_cols(DF, t)


def sc_kta(DF, t, **kw):                             # R7 (centered cosine)
    t = t - t.mean(); DFc = DF - DF.mean(0)
    num = DFc.T @ t
    den = np.sqrt((DFc ** 2).sum(0) * (t ** 2).sum())
    out = np.zeros(DF.shape[1]); nz = den > 0
    out[nz] = np.abs(num[nz]) / den[nz]
    return out


def sc_dcor(DF, t, m=1500, seed=0, **kw):            # R3 (nonlinear, O(m^2)/feat)
    idx = np.random.RandomState(seed).choice(len(t), min(m, len(t)), replace=False)
    D, tt = DF[idx], t[idx]
    return np.array([rankers._dcor(D[:, k], tt) for k in range(D.shape[1])])


def sc_mi(DF, t, m=5000, seed=0, **kw):              # R4 / scikit mutual_info
    from sklearn.feature_selection import mutual_info_regression
    idx = np.random.RandomState(seed).choice(len(t), min(m, len(t)), replace=False)
    return mutual_info_regression(DF[idx], t[idx], random_state=seed)


def sc_fscore(DF, t, **kw):                          # scikit F-test (linear screen)
    from sklearn.feature_selection import f_regression
    F, _ = f_regression(DF, t)
    return np.nan_to_num(F)


def sc_lasso(DF, t, m=5000, seed=0, **kw):           # scikit Lasso |coef| (R8 stand-in)
    from sklearn.linear_model import Lasso
    idx = np.random.RandomState(seed).choice(len(t), min(m, len(t)), replace=False)
    mdl = Lasso(alpha=1e-3, max_iter=5000).fit(DF[idx], t[idx])
    return np.abs(mdl.coef_)


def sc_rf(DF, t, m=4000, seed=0, **kw):              # scikit RandomForest importance
    from sklearn.ensemble import RandomForestRegressor
    idx = np.random.RandomState(seed).choice(len(t), min(m, len(t)), replace=False)
    mdl = RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1,
                                random_state=seed).fit(DF[idx], t[idx])
    return mdl.feature_importances_


def sc_rfe(DF, t, m=4000, seed=0, budget=400, **kw):  # scikit RFE (linear estimator)
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import Ridge
    idx = np.random.RandomState(seed).choice(len(t), min(m, len(t)), replace=False)
    n_sel = min(budget, DF.shape[1])
    step = max(1, DF.shape[1] // 12)
    sel = RFE(Ridge(alpha=1.0), n_features_to_select=n_sel, step=step)
    sel.fit(DF[idx], t[idx])
    return -sel.ranking_.astype(float)              # rank 1 (best) -> highest score


# ---- R5 / R6 (do NOT use pair-target) ----
def sc_rrelieff(Xz, y, **kw):                        # R5 (activity only)
    return rankers.r5_rrelieff(Xz, y)


def sc_laplacian(Xz, **kw):                          # R6 (label-free, unsupervised)
    return rankers.r6_laplacian(Xz)


# ===========================================================================
# TIER B  R9 greedy-forward on space fidelity (matching-pursuit style):
# repeatedly add the feature whose pair-difference best explains the RESIDUAL of
# the target (activity gap or MCS) not yet captured -- rewards COMPLEMENTARITY,
# which marginal per-feature scores cannot. Vectorised over a pre-filtered
# candidate set. Returns a SELECTED SET (not per-feature scores).
# ===========================================================================
def greedy_forward(DF, t, budget, prefilter=400, seed=0):
    m, d = DF.shape
    # pre-filter to the strongest marginal candidates (cheap Spearman), then
    # greedily orthogonalise against the residual.
    pre = np.argsort(np.abs(rankers._spearman_cols(DF, t)))[::-1][:min(prefilter, d)]
    C = DF[:, pre].astype(np.float64)
    C = C - C.mean(0)
    cn = np.linalg.norm(C, axis=0); cn[cn == 0] = 1.0
    Cn = C / cn
    r = (t - t.mean()).astype(np.float64)
    chosen = []
    for _ in range(min(budget, len(pre))):
        corr = np.abs(Cn.T @ r)
        corr[chosen] = -1
        k = int(np.argmax(corr))
        chosen.append(k)
        # regress the chosen direction out of the residual
        v = Cn[:, k]
        r = r - (v @ r) * v
    return pre[chosen]


# ===========================================================================
# ALLOCATION: per-feature scores -> selected feature SET
# feature count owned by the allocation knob (per-rep budget x top-N reps)
# ===========================================================================
def allocate(scores, slices, n_per_rep, top_n):
    rep_kept, rep_strength = {}, {}
    for r, (a, b) in slices.items():
        s = scores[a:b]
        k = min(n_per_rep, len(s))
        top = np.argsort(s)[::-1][:k]
        rep_kept[r] = a + top
        rep_strength[r] = float(np.sum(np.maximum(s[top], 0)))
    keep_reps = sorted(rep_strength, key=rep_strength.get, reverse=True)[:top_n]
    return np.concatenate([rep_kept[r] for r in keep_reps])


# ===========================================================================
# SCORING: a selected feature set -> both yardsticks, both coefficients
# ===========================================================================
def score_space(Xz_sel, p, pairs, coeff):
    D = simeval.dist_cosine(Xz_sel) if coeff == 'cosine' else simeval.dist_euclidean(Xz_sel)
    rho, negrmse = simeval.eval_knn_readacross(D, p['y'], p['groups'])
    dpair = D[pairs['I'], pairs['J']]
    struct = spearmanr(dpair, pairs['mcs'])[0]
    # rho stays NaN when the read-across Spearman is undefined -- unlike
    # struct_fidelity on the next line, it is not folded to 0.0.
    # readacross_defined separates an undefined rho from a measured rho of 0.
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                rmse=float(-negrmse),
                struct_fidelity=float(0.0 if not np.isfinite(struct) else struct),
                n_features=int(Xz_sel.shape[1]))


# ===========================================================================
# the selector menu
# ===========================================================================
# each entry: (name, kind, target_modes)
#   kind 'pair'  -> scorer(DF, t) ; run per target mode
#   kind 'x'     -> scorer(Xz[,y]) ; no pair target
#   kind 'set'   -> returns a feature set directly (greedy)
PAIR_SCORERS = {
    'r1r2_spearman': sc_spearman,      # R1 (activity) / R2 (structural)
    'r7_kta':        sc_kta,
    'r3_dcor':       sc_dcor,
    'r4_mi':         sc_mi,            # == scikit mutual_info
    'sk_fscore':     sc_fscore,
    'sk_lasso':      sc_lasso,        # doubles as R8 (HSIC-Lasso stand-in)
    'sk_rf':         sc_rf,
    'sk_rfe':        sc_rfe,
}


def run_dataset(name, n_per_rep=100, top_n=4, concentrate_top_n=2, quick=False):
    t0 = time.time()
    p = load_prepared(name)
    Xz, slices = build_pool(p)
    pairs = load_pairs(name, Xz)
    DF, dy, mcs = pairs['DF'], pairs['dy'], pairs['mcs']
    budget = n_per_rep * top_n
    rows = []

    def emit(selector, mode, sel_idx, alloc):
        Xz_sel = Xz[:, sel_idx]
        for coeff in ('cosine', 'euclidean'):
            s = score_space(Xz_sel, p, pairs, coeff)
            s.update(selector=selector, target=mode, coeff=coeff, alloc=alloc)
            rows.append(s)

    targets = {'activity': dy, 'structural': mcs}
    pair_menu = list(PAIR_SCORERS.items())
    if quick:
        pair_menu = [('r1r2_spearman', sc_spearman), ('sk_fscore', sc_fscore)]

    # ---- tier A + C : pair-based scorers, both target modes ----
    for sname, fn in pair_menu:
        for mode, t in targets.items():
            sc = fn(DF, t, budget=budget)
            sel = allocate(sc, slices, n_per_rep, top_n)
            emit(sname, mode, sel, f'top{top_n}x{n_per_rep}')
            if not quick and sname in ('r1r2_spearman', 'sk_lasso'):
                sel_c = allocate(sc, slices, n_per_rep, concentrate_top_n)
                emit(sname, mode, sel_c, f'top{concentrate_top_n}x{n_per_rep}')
        print(f'  [{name}] {sname:14s} done ({time.time()-t0:.0f}s)', flush=True)

    # ---- R5 RReliefF (activity only) , R6 Laplacian (label-free) ----
    if not quick:
        sc5 = sc_rrelieff(Xz, p['y'])
        emit('r5_rrelieff', 'activity', allocate(sc5, slices, n_per_rep, top_n), f'top{top_n}x{n_per_rep}')
        sc6 = sc_laplacian(Xz)
        emit('r6_laplacian', 'labelfree', allocate(sc6, slices, n_per_rep, top_n), f'top{top_n}x{n_per_rep}')
        print(f'  [{name}] r5/r6 done ({time.time()-t0:.0f}s)', flush=True)

    # ---- combinations via combine_ranks (user's hedge) ----
    if not quick:
        s_act = sc_spearman(DF, dy)
        s_str = sc_spearman(DF, mcs)
        fused = rankers.combine_ranks(s_act, s_str)                 # R1+R2 equal fuse
        emit('combo_r1+r2', 'act+struct', allocate(fused, slices, n_per_rep, top_n), f'top{top_n}x{n_per_rep}')
        # R2 gated by R1: structural rank, activity breaks ties (small weight)
        gated = rankdata(s_str) + 0.01 * rankdata(s_act)
        emit('combo_r2|r1', 'struct|act', allocate(gated, slices, n_per_rep, top_n), f'top{top_n}x{n_per_rep}')

    # ---- tier B : R9 greedy-forward (set selector), both modes ----
    if not quick:
        for mode, t in targets.items():
            sel = greedy_forward(DF, t, budget)
            emit('r9_greedy', mode, sel, f'greedy<= {budget}')
        print(f'  [{name}] r9 greedy done ({time.time()-t0:.0f}s)', flush=True)

    # ---- BASELINES (native rep + coefficient, never averaged) ----
    for rep, coeff in [('ecfp4_count', 'tanimoto'), ('ecfp4', 'tanimoto'),
                       ('ecfp4_count', 'cosine')]:
        D = getattr(simeval, f'dist_{coeff}')(p[rep])
        rho, negrmse = simeval.eval_knn_readacross(D, p['y'], p['groups'])
        struct = spearmanr(D[pairs['I'], pairs['J']], pairs['mcs'])[0]
        rows.append(dict(selector=f'BASELINE:{rep}', target='-', coeff=coeff,
                         alloc='native', readacross_rho=float(rho),
                         readacross_defined=is_defined(rho), rmse=float(-negrmse),
                         struct_fidelity=float(struct), n_features=int(p[rep].shape[1])))

    print(f'[{name}] {len(rows)} rows in {time.time()-t0:.0f}s', flush=True)
    return rows


def summarise(name, rows):
    print(f'\n===== {name}: ranker experiment (cosine coeff shown) =====')
    print(f'  {"selector":16s} {"target":11s} {"alloc":14s} {"readacross":>10} {"struct":>8} {"nfeat":>6}')
    base = {r['selector']: r for r in rows if r['selector'].startswith('BASELINE') and r['coeff'] == 'tanimoto'}
    shown = [r for r in rows if r['coeff'] == 'cosine' or r['selector'].startswith('BASELINE')]
    # a non-finite readacross_rho held its insertion slot under -r[...]; rank_desc
    # sends it to the end of the table instead.
    for r in sorted(shown, key=lambda r: rank_desc(r['readacross_rho']), reverse=True):
        print(f'  {r["selector"]:16s} {r["target"]:11s} {r["alloc"]:14s} '
              f'{r["readacross_rho"]:>10.3f} {r["struct_fidelity"]:>8.3f} {r["n_features"]:>6d}')
    n_def = sum(1 for r in shown if is_defined(r['readacross_rho']))
    print(f'  read-across rho finite for {n_def}/{len(shown)} rows shown')
    for bn, b in base.items():
        print(f'  -- baseline {bn}: read-across {b["readacross_rho"]:.3f}, '
              f'struct {b["struct_fidelity"]:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--n_per_rep', type=int, default=100)
    ap.add_argument('--top_n', type=int, default=4)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        rows = run_dataset(ds, n_per_rep=args.n_per_rep, top_n=args.top_n, quick=args.quick)
        out[ds] = rows
        summarise(ds, rows)
    tag = '_quick' if args.quick else ''
    path = os.path.join(HERE, f'ranker_experiment{tag}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
