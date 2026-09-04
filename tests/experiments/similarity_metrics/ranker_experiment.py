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
  * Yardsticks: (1) activity read-across rho (five scaffold-grouped folds -- no
    random splits); (2) structural fidelity = Spearman between the space's
    pairwise distance and MCS over the cached pairs.
  * Baselines (NEVER averaged): ecfp4_count:tanimoto (pilot's cheap winner) and
    ecfp4:tanimoto (field standard), each on both yardsticks.

NOTHING IS SCORED ON WHAT PICKED IT. Every ranker here fits something, and both
yardsticks used to be taken on the rows or pairs that fitting saw:

  read-across   every activity-target ranker, and r5_rrelieff, scored features
                using labels from rows that later became the test fold. The
                ranker now re-runs INSIDE each of the five folds, seeing only
                training-fold labels and only the cached pairs whose both
                endpoints are training-fold molecules; the held-out fold is
                predicted from its five nearest training neighbours and the
                Spearman is pooled over all held-out rows.
  structural    combo_r1+r2, combo_r2|r1 and r9_greedy structural ranked features
                by correlation to MCS overlap and were then graded by that same
                correlation on the same pairs, while the baselines, which fit
                nothing, were not -- so the two were not comparable. Molecules
                are now cut in half by whole Bemis-Murcko scaffold group
                (_heldout.scaffold_halves); the ranker sees one half, the
                Spearman is taken on pairs inside the other, and the baselines
                are measured on that same half.

readacross_rho_insample and struct_fidelity_insample carry the old
fit-equals-score numbers beside each held-out one, so the size of the optimism
is visible rather than assumed. The per-column z-score mean and sd follow the
same rule: they come from whichever molecules the selection was allowed to see.

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
from _heldout import (SPLIT_SEED, scaffold_halves, pairs_within,  # noqa: E402
                      pairs_within_mask, zscore_fit)

warnings.filterwarnings('ignore')


# ===========================================================================
# pooled feature matrix + per-rep column slices
# ===========================================================================
def build_pool(p, fit_rows=None):
    """Concatenate the 4 pool reps; return (Xz z-scored, slices dict).

    fit_rows, when given, are the rows the per-column mean and sd come from;
    every row is still z-scored and returned. None takes them from all rows."""
    cols, slices, start = [], {}, 0
    for r in POOL_REPS:
        X = p[r].astype(np.float64)
        slices[r] = (start, start + X.shape[1])
        start += X.shape[1]
        cols.append(X)
    X = np.concatenate(cols, axis=1)
    if fit_rows is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Xz = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Xz = zscore_fit(X, fit_rows)
    return Xz, slices


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
# SCORING
# Every number below is measured on molecules the selection did not see.
#
# read-across   five scaffold-grouped folds. Inside each fold the ranker scores
#               and allocates features using ONLY the cached pairs whose both
#               endpoints are training-fold molecules, and only training-fold
#               labels; the held-out fold is then predicted from its five nearest
#               training neighbours. The pooled Spearman over all held-out rows
#               is the reported rho. Previously every activity-target ranker, and
#               r5_rrelieff, scored features using the labels of the rows they
#               were then tested on.
# structural    the molecules are cut in half by whole Bemis-Murcko scaffold
#               group; features are ranked on the pairs inside one half and the
#               Spearman against MCS distance is taken on the pairs inside the
#               other. Previously combo_r1+r2, combo_r2|r1 and r9_greedy
#               structural were graded on the pairs that picked them, while the
#               baselines, which fit nothing, were not -- so the two were not
#               comparable.
# *_insample    the old fit-equals-score numbers, kept beside each held-out one
#               so the size of the optimism is visible rather than assumed.
# ===========================================================================
def _dist(X, coeff):
    return simeval.dist_cosine(X) if coeff == 'cosine' else simeval.dist_euclidean(X)


def make_ctx(p, rows, I, J, mcs, dy, key):
    """One selection context: the molecules a ranker may see, the pooled feature
    matrix z-scored with THEIR mean and sd, and the cached pairs with both
    endpoints among them. Xz covers every row, so the caller can build a full
    distance matrix from columns chosen here and cut it to whichever rows it
    means to score."""
    Xz, slices = build_pool(p, fit_rows=rows)
    keep = pairs_within_mask(I, J, rows, p['n'])
    DF = np.abs(Xz[I[keep]] - Xz[J[keep]]).astype(np.float32)
    return dict(key=key, rows=rows, Xz=Xz, slices=slices, DF=DF,
                dy=dy[keep], mcs=mcs[keep], n_pairs=int(keep.sum()))


def make_selector(kind, fn, mode, y, n_per_rep, top_n, budget):
    """A callable ctx -> selected feature indices. Everything it reads comes from
    ctx, so the same selector re-runs unchanged inside a training fold, on a
    scaffold half, or on every row."""
    # What a ranker returns depends on the context, never on the coefficient the
    # distance is later built with, so each (selector, context) pair is ranked
    # once and reused for cosine and euclidean. Without this every ranker ran
    # twice per fold, and sk_rf and sk_rfe are the expensive ones.
    memo = {}

    def pick(ctx):
        if ctx['key'] in memo:
            return memo[ctx['key']]
        memo[ctx['key']] = sel = _pick(ctx)
        return sel

    def _pick(ctx):
        if kind == 'pair':
            t = ctx['dy'] if mode == 'activity' else ctx['mcs']
            return allocate(fn(ctx['DF'], t, budget=budget), ctx['slices'],
                            n_per_rep, top_n)
        if kind == 'greedy':
            t = ctx['dy'] if mode == 'activity' else ctx['mcs']
            return greedy_forward(ctx['DF'], t, budget)
        if kind == 'rrelieff':
            r = ctx['rows']
            return allocate(rankers.r5_rrelieff(ctx['Xz'][r], y[r]),
                            ctx['slices'], n_per_rep, top_n)
        if kind == 'laplacian':
            r = ctx['rows']
            return allocate(rankers.r6_laplacian(ctx['Xz'][r]), ctx['slices'],
                            n_per_rep, top_n)
        if kind in ('combo_fuse', 'combo_gate'):
            s_act = sc_spearman(ctx['DF'], ctx['dy'])
            s_str = sc_spearman(ctx['DF'], ctx['mcs'])
            sc = (rankers.combine_ranks(s_act, s_str) if kind == 'combo_fuse'
                  else rankdata(s_str) + 0.01 * rankdata(s_act))
            return allocate(sc, ctx['slices'], n_per_rep, top_n)
        raise ValueError(kind)
    return pick


def readacross_foldwise(pick, fold_ctx, folds, y, coeff, k=5):
    """Pooled held-out five-nearest-neighbour Spearman, with the ranker re-run
    inside every fold. Returns (rho, rmse, mean n_features over the folds)."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    n = len(y)
    preds = np.full(n, np.nan)
    widths = []
    for (tr, te), ctx in zip(folds, fold_ctx):
        sel = pick(ctx)
        widths.append(len(sel))
        D = _dist(ctx['Xz'][:, sel], coeff)
        m = KNeighborsRegressor(n_neighbors=min(k, len(tr)), weights='distance',
                                metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], y[tr])
        preds[te] = m.predict(D[np.ix_(te, tr)])
    ok = np.isfinite(preds)
    rho, _ = spearmanr(preds[ok], y[ok])
    rmse = float(np.sqrt(mean_squared_error(y[ok], preds[ok])))
    return (float('nan') if not np.isfinite(rho) else float(rho)), rmse, \
        float(np.mean(widths))


def struct_on_half(pick, ctx_fit, score_idx, score_pairs, coeff):
    """Rank features on ctx_fit's molecules, then take the Spearman against MCS
    distance over the pairs among score_idx only."""
    sel = pick(ctx_fit)
    D = _dist(ctx_fit['Xz'][:, sel], coeff)[np.ix_(score_idx, score_idx)]
    Ic, Jc, Mc = score_pairs
    r = spearmanr(D[Ic, Jc], Mc)[0]
    return (float('nan') if not np.isfinite(r) else float(r)), len(sel)


def score_insample(pick, ctx_all, y, groups, I, J, mcs, coeff):
    """The old arm: select on every molecule, score on every molecule."""
    sel = pick(ctx_all)
    D = _dist(ctx_all['Xz'][:, sel], coeff)
    rho, negrmse = simeval.eval_knn_readacross(D, y, groups)
    st = spearmanr(D[I, J], mcs)[0]
    return dict(readacross_rho_insample=float(rho),
                rmse_insample=float(-negrmse),
                struct_fidelity_insample=(float('nan') if not np.isfinite(st)
                                          else float(st)),
                n_features=int(len(sel)))


# ===========================================================================
# the selector menu
# ===========================================================================
# each entry: (name, scorer). Every one takes (DF, target) on whichever cached
# pairs its context allows -- one training fold's, one scaffold half's, or all.
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
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    p = load_prepared(name)
    n = p['n']
    y, groups = p['y'], p['groups']
    d = np.load(os.path.join(HERE, f'mcs_pairs_{name}.npz'))
    I, J = d['I'], d['J']
    mcs, dy = d['mcs'].astype(np.float64), d['dy'].astype(np.float64)
    budget = n_per_rep * top_n
    rows = []

    # ---- the two held-out units, built once and shared by every selector ----
    folds = list(GroupKFold(n_splits=5).split(np.arange(n), y, groups))
    fit_idx, sco_idx = scaffold_halves(groups, seed=SPLIT_SEED)
    sco_pairs = pairs_within(I, J, mcs, sco_idx, n)

    ctx_all = make_ctx(p, np.arange(n), I, J, mcs, dy, 'all')
    ctx_fit = make_ctx(p, fit_idx, I, J, mcs, dy, 'fit_half')
    fold_ctx = [make_ctx(p, tr, I, J, mcs, dy, f'fold{k}')
                for k, (tr, _te) in enumerate(folds)]
    n_straddle = len(I) - ctx_fit['n_pairs'] - len(sco_pairs[0])
    print(f'  [{name}] n={n}; five scaffold folds with '
          f'{[c["n_pairs"] for c in fold_ctx]} training pairs each; molecule half '
          f'split (seed {SPLIT_SEED}) {len(fit_idx)}/{len(sco_idx)} with '
          f'{ctx_fit["n_pairs"]} fit / {len(sco_pairs[0])} score pairs, '
          f'{n_straddle} straddling dropped ({time.time()-t0:.0f}s)', flush=True)

    def emit(selector, mode, alloc, kind, fn=None):
        pick = make_selector(kind, fn, mode, y, n_per_rep, top_n, budget)
        for coeff in ('cosine', 'euclidean'):
            ra, rmse, width = readacross_foldwise(pick, fold_ctx, folds, y, coeff)
            st, _w = struct_on_half(pick, ctx_fit, sco_idx, sco_pairs, coeff)
            r = dict(selector=selector, target=mode, coeff=coeff, alloc=alloc,
                     readacross_rho=ra, readacross_defined=is_defined(ra),
                     rmse=rmse,
                     struct_fidelity=st, struct_fidelity_defined=is_defined(st),
                     n_features_per_fold=width)
            r.update(score_insample(pick, ctx_all, y, groups, I, J, mcs, coeff))
            rows.append(r)

    pair_menu = list(PAIR_SCORERS.items())
    if quick:
        pair_menu = [('r1r2_spearman', sc_spearman), ('sk_fscore', sc_fscore)]

    # ---- tier A + C : pair-based scorers, both target modes ----
    for sname, fn in pair_menu:
        for mode in ('activity', 'structural'):
            emit(sname, mode, f'top{top_n}x{n_per_rep}', 'pair', fn)
            if not quick and sname in ('r1r2_spearman', 'sk_lasso'):
                pick_c = make_selector('pair', fn, mode, y, n_per_rep,
                                       concentrate_top_n, budget)
                for coeff in ('cosine', 'euclidean'):
                    ra, rmse, width = readacross_foldwise(pick_c, fold_ctx, folds,
                                                          y, coeff)
                    st, _w = struct_on_half(pick_c, ctx_fit, sco_idx, sco_pairs,
                                            coeff)
                    r = dict(selector=sname, target=mode, coeff=coeff,
                             alloc=f'top{concentrate_top_n}x{n_per_rep}',
                             readacross_rho=ra, readacross_defined=is_defined(ra),
                             rmse=rmse, struct_fidelity=st,
                             struct_fidelity_defined=is_defined(st),
                             n_features_per_fold=width)
                    r.update(score_insample(pick_c, ctx_all, y, groups, I, J,
                                            mcs, coeff))
                    rows.append(r)
        print(f'  [{name}] {sname:14s} done ({time.time()-t0:.0f}s)', flush=True)

    # ---- R5 RReliefF (activity only) , R6 Laplacian (label-free) ----
    if not quick:
        emit('r5_rrelieff', 'activity', f'top{top_n}x{n_per_rep}', 'rrelieff')
        emit('r6_laplacian', 'labelfree', f'top{top_n}x{n_per_rep}', 'laplacian')
        print(f'  [{name}] r5/r6 done ({time.time()-t0:.0f}s)', flush=True)

    # ---- combinations via combine_ranks (user's hedge) ----
    if not quick:
        emit('combo_r1+r2', 'act+struct', f'top{top_n}x{n_per_rep}', 'combo_fuse')
        emit('combo_r2|r1', 'struct|act', f'top{top_n}x{n_per_rep}', 'combo_gate')

    # ---- tier B : R9 greedy-forward (set selector), both modes ----
    if not quick:
        for mode in ('activity', 'structural'):
            emit('r9_greedy', mode, f'greedy<= {budget}', 'greedy')
        print(f'  [{name}] r9 greedy done ({time.time()-t0:.0f}s)', flush=True)

    # ---- BASELINES (native rep + coefficient, never averaged) ----
    # These select nothing, so their two numbers are already honest. They are
    # still taken over the SAME five folds and the SAME score half, or a
    # selector row and a baseline row would be measured on different molecules.
    Ic, Jc, Mc = sco_pairs
    for rep, coeff in [('ecfp4_count', 'tanimoto'), ('ecfp4', 'tanimoto'),
                       ('ecfp4_count', 'cosine')]:
        D = getattr(simeval, f'dist_{coeff}')(p[rep])
        rho, negrmse = simeval.eval_knn_readacross(D, y, groups)
        st = spearmanr(D[np.ix_(sco_idx, sco_idx)][Ic, Jc], Mc)[0]
        st_all = spearmanr(D[I, J], mcs)[0]
        rows.append(dict(selector=f'BASELINE:{rep}', target='-', coeff=coeff,
                         alloc='native', readacross_rho=float(rho),
                         readacross_defined=is_defined(rho), rmse=float(-negrmse),
                         struct_fidelity=(float('nan') if not np.isfinite(st)
                                          else float(st)),
                         struct_fidelity_defined=is_defined(st),
                         readacross_rho_insample=float(rho),
                         rmse_insample=float(-negrmse),
                         struct_fidelity_insample=(float('nan')
                                                   if not np.isfinite(st_all)
                                                   else float(st_all)),
                         n_features=int(p[rep].shape[1]),
                         n_features_per_fold=float(p[rep].shape[1])))

    print(f'[{name}] {len(rows)} rows in {time.time()-t0:.0f}s', flush=True)
    return rows, dict(seed=SPLIT_SEED, by='bemis_murcko_scaffold_group',
                      n_fit_molecules=int(len(fit_idx)),
                      n_score_molecules=int(len(sco_idx)),
                      n_pairs_total=int(len(I)),
                      n_pairs_fit=int(ctx_fit['n_pairs']),
                      n_pairs_score=int(len(Ic)),
                      n_pairs_straddling=int(n_straddle),
                      fold_training_pairs=[int(c['n_pairs']) for c in fold_ctx])


def summarise(name, rows, split):
    print(f'\n===== {name}: ranker experiment (cosine coeff shown) =====')
    print(f'  read-across: the ranker re-runs inside each of five scaffold folds '
          f'and scores the held-out fold.')
    print(f'  struct: the ranker runs on {split["n_fit_molecules"]} molecules and '
          f'is scored on the other {split["n_score_molecules"]}, over '
          f'{split["n_pairs_score"]} pairs.')
    print(f'  in_ra / in_st are the same two numbers with selection and scoring '
          f'on every molecule.')
    print(f'  {"selector":16s} {"target":11s} {"alloc":14s} {"readacross":>10} '
          f'{"struct":>8} {"in_ra":>8} {"in_st":>8} {"nfeat":>6}')
    base = {r['selector']: r for r in rows
            if r['selector'].startswith('BASELINE') and r['coeff'] == 'tanimoto'}
    shown = [r for r in rows if r['coeff'] == 'cosine' or r['selector'].startswith('BASELINE')]
    # a non-finite readacross_rho held its insertion slot under -r[...]; rank_desc
    # sends it to the end of the table instead.
    for r in sorted(shown, key=lambda r: rank_desc(r['readacross_rho']), reverse=True):
        print(f'  {r["selector"]:16s} {r["target"]:11s} {r["alloc"]:14s} '
              f'{r["readacross_rho"]:>10.3f} {r["struct_fidelity"]:>8.3f} '
              f'{r["readacross_rho_insample"]:>8.3f} '
              f'{r["struct_fidelity_insample"]:>8.3f} '
              f'{r["n_features_per_fold"]:>6.0f}')
    n_def = sum(1 for r in shown if is_defined(r['readacross_rho']))
    n_def_st = sum(1 for r in shown if is_defined(r['struct_fidelity']))
    print(f'  read-across rho finite for {n_def}/{len(shown)} rows shown')
    print(f'  structural fidelity finite for {n_def_st}/{len(shown)} rows shown')
    for bn, b in base.items():
        print(f'  -- baseline {bn}: read-across {b["readacross_rho"]:.3f}, '
              f'struct {b["struct_fidelity"]:.3f} '
              f'(this row selects nothing, so held-out and in-sample coincide '
              f'up to which molecules each is measured on)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--n_per_rep', type=int, default=100)
    ap.add_argument('--top_n', type=int, default=4)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        rows, split = run_dataset(ds, n_per_rep=args.n_per_rep,
                                  top_n=args.top_n, quick=args.quick)
        out[ds] = {'rows': rows, 'split': split}
        summarise(ds, rows, split)
    tag = '_quick' if args.quick else ''
    path = os.path.join(HERE, f'ranker_experiment{tag}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
