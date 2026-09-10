"""Does a COMPLEMENTARITY-AWARE feature selector beat marginal importance on the
STANDARD PREDICTION task?

Background
----------
KIRBy's hybrid picks features with `compute_feature_importance` (hybrid.py:2382),
then takes the top-k per rep (`np.argsort(importances)[-n_select:]`, hybrid.py:1107).
Both steps are MARGINAL: a feature's score is computed with its rep in isolation and
does not depend on what has already been selected. So the selector cannot prefer a
feature because it supplies the signal the already-chosen ones lack -- the
complementarity gap documented in project_hybrid_marginal_allocation.

Prior knob work (project_knob_sensitivity_2026_07_16) measured the importance knob as
a WEAK lever (0.0001-0.012 honest delta), but only ever compared MARGINAL methods to
each other (lightgbm_gain vs treeshap vs deeplift). Whether a non-marginal selector
behaves differently is untested. The similarity-metrics study found the two bespoke
non-marginal rankers (R9 greedy-forward, R5 RReliefF) topped its yardsticks -- but on
a similarity objective, not on prediction.

The translation (this is NOT a copy of the similarity rankers)
--------------------------------------------------------------
The similarity study scored features on a PAIR-DIFFERENCE regression: design matrix
|f_i - f_j| over sampled pairs, target |dy| or MCS distance. The prediction task is
the direct problem f -> y, so:

  * R9 greedy-forward is RE-DERIVED for prediction as matching pursuit on (X, y):
    repeatedly take the feature most correlated with the RESIDUAL of y, then project
    that direction out. Complementarity-aware by construction.
  * R5 RReliefF needs NO translation -- `rankers.r5_rrelieff` already operates on
    (X, y) directly. Reused verbatim.
  * mRMR is added because the rebuild roadmap (Appendix B) names it as the sanctioned
    redundancy-aware candidate; it is the natural third non-marginal method.

Design (isolates the selector, holds everything else fixed)
-----------------------------------------------------------
  * Pool = the SAME 4 complementary reps as the similarity study (ecfp4_count,
    atompair_count, desc, maccs) via `_data.load_prepared`, so the two studies are
    directly comparable and no server data is needed.
  * Feature COUNT comes from the allocation knob, exactly as in the similarity study
    and the original working hybrid (c4afb0e): per-rep budget n_per_rep x top-N reps.
  * The REP SET is chosen the SAME way for every selector (train-only standalone
    lightgbm score), so the only thing that varies between arms is WHICH features are
    picked inside each rep. Rank-based selectors emit per-rep orderings that are not
    comparable across reps, so letting the selector also pick the rep set would
    confound the comparison.
  * SELECTION HAPPENS INSIDE EACH TRAINING FOLD. Fitting a selector on all the data
    and then cross-validating leaks the labels. Scaffold-grouped GroupKFold, 5 folds,
    no random splits.
  * Downstream model = lightgbm (the rebuild roadmap's locked internal/default choice,
    ridge removed). Metrics = Spearman rho and RMSE, reported PER FOLD and as PAIRED
    per-fold deltas against the marginal baseline -- not just means.
  * Reference arms: the best single rep at FULL native width (the rebuild roadmap's
    real target, best-rep x model) and the full 4-rep concat.

Does NOT touch hybrid.py / find_best_hybrid.py (rebuild-roadmap lock).
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
SIMDIR = os.path.join(os.path.dirname(HERE), 'similarity_metrics')
sys.path.insert(0, SIMDIR)
from _data import load_prepared, POOL_REPS      # noqa: E402
import rankers                                  # noqa: E402

warnings.filterwarnings('ignore')

N_FOLDS = 5
SEED = 0


# ===========================================================================
# pool assembly
# ===========================================================================
def build_pool(p):
    """Concatenate the 4 pool reps (RAW). Returns (X, slices)."""
    cols, slices, start = [], {}, 0
    for r in POOL_REPS:
        X = np.nan_to_num(np.asarray(p[r], dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        slices[r] = (start, start + X.shape[1])
        start += X.shape[1]
        cols.append(X)
    return np.concatenate(cols, axis=1), slices


def zscore_fit(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


# ===========================================================================
# downstream model
# ===========================================================================
def fit_score(Xtr, ytr, Xte, yte):
    import lightgbm as lgb
    mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, verbose=-1)
    mdl.fit(Xtr, ytr)
    pred = mdl.predict(Xte)
    rho = spearmanr(pred, yte)[0]
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    return (0.0 if not np.isfinite(rho) else float(rho)), rmse


# ===========================================================================
# SELECTORS -- every one returns a per-feature score vector over the columns it
# is given (higher = keep), fitted on TRAIN ONLY.
# ===========================================================================
def sel_lightgbm_gain(Xtr, ytr, **kw):
    """MARGINAL BASELINE -- mirrors compute_feature_importance('lightgbm_gain')."""
    import lightgbm as lgb
    mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                            min_child_samples=10, random_state=SEED, n_jobs=-1,
                            verbose=-1, importance_type='gain')
    mdl.fit(Xtr, ytr)
    return np.asarray(mdl.booster_.feature_importance(importance_type='gain'), float)


def sel_fscore(Xtr, ytr, **kw):
    """MARGINAL, linear -- cheap sklearn screen, free reference point."""
    from sklearn.feature_selection import f_regression
    F, _ = f_regression(Xtr, ytr)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def sel_omp_greedy(Xtr, ytr, budget=100, **kw):
    """R9 greedy-forward, RE-DERIVED FOR PREDICTION (matching pursuit on (X, y)).

    Repeatedly add the feature whose direction best explains the part of y not yet
    explained, then project that direction out of the residual. The greedy ordering
    is NESTED (top-k of the ordering == the greedy set at budget k), so it can be
    emitted as a score vector -- which means it is a genuine drop-in for
    compute_feature_importance and needs NO change to the allocation code.

    No marginal pre-filter (the similarity version pre-filtered to 400 candidates for
    speed); at these widths the full greedy is cheap, so the result is not
    contaminated by a marginal screen.
    """
    C = Xtr - Xtr.mean(0)
    cn = np.linalg.norm(C, axis=0)
    cn[cn < 1e-12] = 1.0
    Cn = C / cn
    r = ytr - ytr.mean()
    d = Xtr.shape[1]
    k_max = int(min(budget, d))
    scores = np.full(d, -np.inf)
    chosen = []
    for step in range(k_max):
        corr = np.abs(Cn.T @ r)
        if chosen:
            corr[chosen] = -1.0
        k = int(np.argmax(corr))
        if not np.isfinite(corr[k]) or corr[k] <= 0:
            break
        chosen.append(k)
        scores[k] = -float(step)          # first chosen -> highest score
        v = Cn[:, k]
        r = r - (v @ r) * v
    return scores


def sel_rrelieff(Xtr, ytr, **kw):
    """R5 RReliefF -- reused VERBATIM from the similarity study's rankers.py.
    Local / nearest-neighbour, already defined on (X, y); no translation needed."""
    return np.asarray(rankers.r5_rrelieff(Xtr, ytr, seed=SEED), float)


def sel_mrmr(Xtr, ytr, budget=100, n_cand=400, **kw):
    """mRMR (MID): greedily maximise relevance(f, y) minus mean |corr| to the
    already-selected set. Redundancy-aware, the roadmap's sanctioned candidate.
    Relevance = |Pearson(f, y)|; candidates pre-screened by relevance for O(n_cand)
    correlation work. Nested ordering -> score vector, like the greedy above."""
    d = Xtr.shape[1]
    Xc = Xtr - Xtr.mean(0)
    sd = Xc.std(0)
    sd[sd < 1e-12] = 1.0
    Z = Xc / sd
    yz = (ytr - ytr.mean()) / (ytr.std() if ytr.std() > 1e-12 else 1.0)
    rel = np.abs(Z.T @ yz) / len(ytr)
    rel = np.nan_to_num(rel, nan=0.0)

    cand = np.argsort(rel)[::-1][:min(n_cand, d)]
    Zc = Z[:, cand]
    rel_c = rel[cand]
    k_max = int(min(budget, len(cand)))

    scores = np.full(d, -np.inf)
    chosen, red_sum = [], np.zeros(len(cand))
    for step in range(k_max):
        if step == 0:
            crit = rel_c.copy()
        else:
            crit = rel_c - red_sum / step
        if chosen:
            crit[chosen] = -np.inf
        k = int(np.argmax(crit))
        chosen.append(k)
        scores[cand[k]] = -float(step)
        red_sum += np.abs(Zc.T @ Zc[:, k]) / len(ytr)
    return scores


SELECTORS = {
    'lightgbm_gain (MARGINAL baseline)': sel_lightgbm_gain,
    'f_regression (MARGINAL)':           sel_fscore,
    'r9_greedy_omp (COMPLEMENTARITY)':   sel_omp_greedy,
    'r5_rrelieff (LOCAL/NN)':            sel_rrelieff,
    'mrmr (REDUNDANCY-AWARE)':           sel_mrmr,
}
# selectors that must see one rep at a time (their scores are per-rep orderings)
PER_REP = {'r9_greedy_omp (COMPLEMENTARITY)', 'mrmr (REDUNDANCY-AWARE)',
           'lightgbm_gain (MARGINAL baseline)', 'f_regression (MARGINAL)'}


def score_features(fn, name, Xtr_z, ytr, slices, n_per_rep):
    """Produce a full-width score vector, running per-rep where that is the
    faithful analogue of compute_feature_importance (which loops over reps)."""
    if name in PER_REP:
        out = np.zeros(Xtr_z.shape[1])
        for r, (a, b) in slices.items():
            out[a:b] = fn(Xtr_z[:, a:b], ytr, budget=n_per_rep)
        return out
    return fn(Xtr_z, ytr, budget=n_per_rep)


def allocate(scores, slices, n_per_rep, keep_reps):
    """Top n_per_rep features inside each of the reps in keep_reps."""
    idx = []
    for r in keep_reps:
        a, b = slices[r]
        s = scores[a:b]
        k = min(n_per_rep, len(s))
        idx.append(a + np.argsort(s)[::-1][:k])
    return np.concatenate(idx)


# ===========================================================================
# one dataset
# ===========================================================================
def run_dataset(name, n_per_rep=100, quick=False):
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    p = load_prepared(name)
    X, slices = build_pool(p)
    y, groups = p['y'], p['groups']
    print(f'[{name}] n={len(y)} pool={X.shape[1]} feats '
          f'({", ".join(f"{r}:{slices[r][1]-slices[r][0]}" for r in POOL_REPS)})',
          flush=True)

    menu = dict(SELECTORS)
    if quick:
        menu = {k: v for k, v in menu.items()
                if 'lightgbm_gain' in k or 'r9_greedy' in k}

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ytr, yte = y[tr], y[te]
        mu, sd = zscore_fit(X[tr])
        Xtr_z, Xte_z = (X[tr] - mu) / sd, (X[te] - mu) / sd

        # --- rep ORDER: chosen by TRAIN-ONLY inner CV (no test leakage).
        # The top-2 arm keeps only the 2 strongest reps, so that choice must not
        # see the test fold. Inner 3-fold GroupKFold on the training groups.
        inner = GroupKFold(n_splits=3)
        rep_score = {}
        for r in POOL_REPS:
            a, b = slices[r]
            inner_rho = []
            for itr, iva in inner.split(Xtr_z[:, a:b], ytr, groups[tr]):
                rho_i, _ = fit_score(Xtr_z[np.ix_(itr, np.arange(a, b))], ytr[itr],
                                     Xtr_z[np.ix_(iva, np.arange(a, b))], ytr[iva])
                inner_rho.append(rho_i)
            rep_score[r] = float(np.mean(inner_rho))
        order = sorted(rep_score, key=rep_score.get, reverse=True)
        best_rep = order[0]

        # --- reference arms, scored on the held-out fold (reporting only; these
        # do NOT feed any selection decision) ---
        for r in POOL_REPS:
            a, b = slices[r]
            rho, rmse = fit_score(Xtr_z[:, a:b], ytr, Xte_z[:, a:b], yte)
            fold_rows.append(dict(fold=fold, arm=f'REF single rep: {r}', selector='-',
                                  n_features=b - a, rho=rho, rmse=rmse,
                                  inner_rho=rep_score[r]))
        rho, rmse = fit_score(Xtr_z, ytr, Xte_z, yte)
        fold_rows.append(dict(fold=fold, arm='REF full 4-rep concat', selector='-',
                              n_features=X.shape[1], rho=rho, rmse=rmse))

        # --- selector arms: same rep sets for everyone ---
        arms = {f'top4x{n_per_rep}': order[:4], f'top2x{n_per_rep}': order[:2]}
        for sname, fn in menu.items():
            ts = time.time()
            sc = score_features(fn, sname, Xtr_z, ytr, slices, n_per_rep)
            for arm, keep in arms.items():
                sel = allocate(sc, slices, n_per_rep, keep)
                rho, rmse = fit_score(Xtr_z[:, sel], ytr, Xte_z[:, sel], yte)
                fold_rows.append(dict(fold=fold, arm=arm, selector=sname,
                                      n_features=len(sel), rho=rho, rmse=rmse))
            print(f'  [{name}] fold{fold} {sname:34s} {time.time()-ts:6.1f}s', flush=True)

        print(f'  [{name}] fold{fold} best rep={best_rep} (inner-CV rho='
              f'{rep_score[best_rep]:.3f}) ({time.time()-t0:.0f}s)', flush=True)

    print(f'[{name}] {len(fold_rows)} rows in {time.time()-t0:.0f}s', flush=True)
    return fold_rows


# ===========================================================================
# reporting -- paired per-fold deltas vs the marginal baseline
# ===========================================================================
def summarise(name, rows, n_per_rep=100):
    import collections
    BASE = 'lightgbm_gain (MARGINAL baseline)'
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['arm'], r['selector'])].append(r)

    def mean(key, field):
        return float(np.mean([r[field] for r in agg[key]]))

    print(f'\n===== {name}: selector comparison (5-fold scaffold GroupKFold, lightgbm) =====')
    print(f'  {"arm":16s} {"selector":34s} {"rho":>7} {"rmse":>7} {"nfeat":>6}')
    refs = sorted([k for k in agg if k[1] == '-'], key=lambda k: -mean(k, 'rho'))
    for k in refs:
        print(f'  {k[0]:16s} {"-":34s} {mean(k,"rho"):>7.3f} {mean(k,"rmse"):>7.3f} '
              f'{agg[k][0]["n_features"]:>6d}')
    print()
    for arm in (f'top4x{n_per_rep}', f'top2x{n_per_rep}'):
        keys = sorted([k for k in agg if k[0] == arm], key=lambda k: -mean(k, 'rho'))
        for k in keys:
            print(f'  {k[0]:16s} {k[1]:34s} {mean(k,"rho"):>7.3f} {mean(k,"rmse"):>7.3f} '
                  f'{agg[k][0]["n_features"]:>6d}')
        print()

    # paired per-fold deltas vs the marginal baseline (the honest comparison)
    print(f'  PAIRED per-fold delta vs {BASE}:')
    print(f'  {"arm":16s} {"selector":34s} {"d_rho":>8} {"win/tie/loss":>13}')
    for arm in (f'top4x{n_per_rep}', f'top2x{n_per_rep}'):
        b = {r['fold']: r for r in agg[(arm, BASE)]}
        for k in sorted([k for k in agg if k[0] == arm and k[1] != BASE]):
            d = [r['rho'] - b[r['fold']]['rho'] for r in agg[k]]
            w = sum(1 for x in d if x > 1e-6)
            l = sum(1 for x in d if x < -1e-6)
            print(f'  {k[0]:16s} {k[1]:34s} {np.mean(d):>+8.4f} '
                  f'{f"{w}/{len(d)-w-l}/{l}":>13}')
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--n_per_rep', type=int, default=100)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        rows = run_dataset(ds, n_per_rep=args.n_per_rep, quick=args.quick)
        out[ds] = rows
        summarise(ds, rows, n_per_rep=args.n_per_rep)
    tag = '_quick' if args.quick else ''
    path = os.path.join(HERE, f'selector_prediction{tag}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
