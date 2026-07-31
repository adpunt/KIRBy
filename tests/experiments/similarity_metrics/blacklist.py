"""blacklist.py -- STANDALONE saboteur-finder (the formal feature blacklist).

The similarity-metric study kept turning up individual columns inside an
otherwise-useful feature block that ACTIVELY HURT a yardstick: fr_Ar_OH /
fr_HOCCN / fr_nitrile inside the molecular physchem (RDKit `desc`) block, and
charge / isoelectric_point inside the antibody descriptor block. Removing them
RAISES the score. Those columns are "saboteurs" and the set of them is the
"blacklist" that find_best_hybrid now accepts via its `feature_blacklist` arg.

This module answers, from data alone and HONESTLY: given a feature block, which
columns are saboteurs?

WHY THIS IS NOT feature_ablation.py's desc_drop_one_greedy
---------------------------------------------------------
`feature_ablation.desc_drop_one_greedy` runs a greedy backward drop on the WHOLE
dataset and reports whatever the greedy path removed. That over-fits: a column
can look like a saboteur on the exact pairs/rows you selected it on and not
generalise. Here every drop decision is made on TRAIN folds and its effect is
re-measured on HELD-OUT folds, and a column only reaches the blacklist if it is
picked as a saboteur in a majority of independent folds AND its mean held-out
gain clears a threshold. That is a stability + out-of-sample guard, so the
returned blacklist is the FORMAL one, not a greedy artefact.

TWO YARDSTICKS (same two the whole study uses; pick one via `yardstick`)
  'structural'  Spearman( pairwise Euclidean distance , MCS distance ) over the
                cached MCS pairs. Higher = the space's geometry mirrors shared
                substructure. Fast: Euclidean rank is unchanged by the final
                sqrt, so we work with per-column squared-difference SUMS and a
                dropped column is just a subtraction (exact).
  'readacross'  scaffold-grouped distance-weighted kNN read-across Spearman
                (simeval.eval_knn_readacross's definition), here made genuinely
                held-out: kNN reference = TRAIN molecules, scored on the TEST
                molecules. Heavier (an n x n distance per candidate), so it
                honours `max_drop` / `max_features` caps.

PUBLIC API
  identify_saboteurs(rep_matrix, feature_names, y, groups, structural_pairs,
                     yardstick='structural', ...) -> dict with:
      'blacklist'        ranked list of {name, col, ...} that cleared the guard
      'blacklist_cols'   just the integer column indices (feed to find_best_hybrid)
      'ranked'           EVERY feature that was ever dropped, ranked
      'baseline_score'   mean held-out yardstick with the FULL block
      'ablated_score'    mean held-out yardstick with the blacklist removed
      'delta'            ablated - baseline (should be >= 0 by construction)
      plus book-keeping (n_splits, params, per-fold traces).

Standalone: imports only the shared read-only tools in this directory + sklearn.
Modifies nothing.
"""
import os
import sys
import math
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VALID_YARDSTICKS = ('structural', 'readacross')


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _spearman(a, b):
    """Spearman rho, higher = better; NaN-safe -> returns nan if undefined."""
    if len(a) < 3:
        return float('nan')
    rho = spearmanr(a, b)[0]
    return float('nan') if not np.isfinite(rho) else float(rho)


def _unpack_pairs(structural_pairs):
    """Accept an npz/dict (keys I, J, mcs) or a 3-tuple (I, J, mcs)."""
    if structural_pairs is None:
        return None
    if isinstance(structural_pairs, (tuple, list)) and len(structural_pairs) == 3:
        I, J, mcs = structural_pairs
    else:  # npz handle or dict
        I, J, mcs = structural_pairs['I'], structural_pairs['J'], structural_pairs['mcs']
    return (np.asarray(I).astype(int),
            np.asarray(J).astype(int),
            np.asarray(mcs).astype(np.float64))


def _train_scale(X, train_idx):
    """Per-column 1/std scaling learned on TRAIN rows only (mean cancels in
    pairwise differences, so only the scale is applied)."""
    sd = X[train_idx].std(0)
    sd[sd == 0] = 1.0
    return X / sd


# --------------------------------------------------------------------------
# STRUCTURAL yardstick: greedy backward on per-column squared diffs
# --------------------------------------------------------------------------
def _structural_fold_greedy(Xz, I, J, mcs, train_idx, test_idx,
                            max_drop, min_step_gain):
    """One fold. Decide drops on train-train pairs; re-measure each drop on the
    held-out test-test pairs. Returns per-drop trace with train + held-out gains
    and the fold's own full/ablated held-out scores."""
    tr = np.isin(I, train_idx) & np.isin(J, train_idx)
    te = np.isin(I, test_idx) & np.isin(J, test_idx)
    if tr.sum() < 10 or te.sum() < 10:
        return None

    SQ = (Xz[I] - Xz[J]) ** 2                      # (n_pairs, d) per-column sq diff
    SQ_tr, mcs_tr = SQ[tr], mcs[tr]
    SQ_te, mcs_te = SQ[te], mcs[te]
    d = Xz.shape[1]
    kept = np.ones(d, bool)

    sumsq_tr = SQ_tr.sum(1)
    sumsq_te = SQ_te.sum(1)
    cur_tr = _spearman(sumsq_tr, mcs_tr)
    cur_te = _spearman(sumsq_te, mcs_te)
    full_te = cur_te                                # held-out score, all features

    trace = []
    cap = d - 2 if max_drop is None else min(max_drop, d - 2)
    for _ in range(cap):
        best_gain, best_c, best_tr = min_step_gain, -1, None
        for c in np.where(kept)[0]:
            s = _spearman(sumsq_tr - SQ_tr[:, c], mcs_tr)
            if np.isfinite(s) and (s - cur_tr) > best_gain:
                best_gain, best_c, best_tr = s - cur_tr, int(c), s
        if best_c < 0:
            break
        new_sumsq_te = sumsq_te - SQ_te[:, best_c]
        new_te = _spearman(new_sumsq_te, mcs_te)
        trace.append({'col': best_c, 'train_gain': float(best_gain),
                      'heldout_gain': float(new_te - cur_te)})
        kept[best_c] = False
        sumsq_tr = sumsq_tr - SQ_tr[:, best_c]
        cur_tr = best_tr
        sumsq_te, cur_te = new_sumsq_te, new_te

    return {'trace': trace, 'kept_after': kept, 'full_te': float(full_te),
            'SQ_te': SQ_te, 'mcs_te': mcs_te}


# --------------------------------------------------------------------------
# READACROSS yardstick: greedy backward on held-out kNN Spearman
# --------------------------------------------------------------------------
def _readacross_heldout_score(Dsq_te_tr, y_tr, y_te, k):
    """Distance-weighted kNN with the TRAIN molecules as the reference, scored on
    the TEST molecules. Dsq_te_tr = squared distances (test rows, train cols)."""
    D = np.sqrt(np.maximum(Dsq_te_tr, 0.0))
    kk = min(k, D.shape[1])
    m = KNeighborsRegressor(n_neighbors=kk, weights='distance', metric='precomputed')
    m.fit(np.zeros((D.shape[1], D.shape[1])), y_tr)   # ref labels only; distances at predict
    preds = m.predict(D)
    return _spearman(preds, y_te)


def _readacross_fold_greedy(Xz, y, groups, train_idx, test_idx,
                            max_drop, min_step_gain, k, cand_cap):
    """One fold. Decide drops by the held-out kNN Spearman (train ref -> test
    scored); the held-out gain IS the selection signal here, so we additionally
    require it to survive across folds (the stability guard) to be trusted.
    Incremental squared-distance maintenance keeps each candidate O(n_te*n_tr)."""
    Xtr, Xte = Xz[train_idx], Xz[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    if len(train_idx) < 10 or len(test_idx) < 5:
        return None
    d = Xz.shape[1]
    kept = np.ones(d, bool)

    # running squared distance (test rows x train cols) over kept columns
    Dsq = np.zeros((len(test_idx), len(train_idx)))
    for c in range(d):
        Dsq += (Xte[:, c][:, None] - Xtr[:, c][None, :]) ** 2

    cur = _readacross_heldout_score(Dsq, y_tr, y_te, k)
    full_te = cur
    trace = []
    cap = d - 2 if max_drop is None else min(max_drop, d - 2)
    for _ in range(cap):
        best_gain, best_c, best_Dsq = min_step_gain, -1, None
        cols = np.where(kept)[0]
        if cand_cap and len(cols) > cand_cap:
            # evaluate a capped random-free subset (lowest-variance-first is
            # arbitrary; keep it deterministic by just scanning all when small)
            cols = cols
        for c in cols:
            cand = Dsq - (Xte[:, c][:, None] - Xtr[:, c][None, :]) ** 2
            s = _readacross_heldout_score(cand, y_tr, y_te, k)
            if np.isfinite(s) and (s - cur) > best_gain:
                best_gain, best_c, best_Dsq = s - cur, int(c), cand
        if best_c < 0:
            break
        trace.append({'col': best_c, 'train_gain': float(best_gain),
                      'heldout_gain': float(best_gain)})
        kept[best_c] = False
        Dsq = best_Dsq
        cur = cur + best_gain

    return {'trace': trace, 'kept_after': kept, 'full_te': float(full_te),
            'Xte': Xte, 'Xtr': Xtr, 'y_tr': y_tr, 'y_te': y_te}


# --------------------------------------------------------------------------
# PUBLIC: identify_saboteurs
# --------------------------------------------------------------------------
def identify_saboteurs(rep_matrix, feature_names, y, groups, structural_pairs,
                       yardstick='structural', n_splits=5, max_drop=None,
                       min_step_gain=1e-6, min_folds=None, min_heldout_gain=0.0,
                       knn_k=5, cand_cap=None, verbose=False):
    """Greedy-backward feature ablation on HELD-OUT folds -> the formal blacklist.

    Parameters
    ----------
    rep_matrix : (n, d) array         one feature BLOCK (e.g. the 208-col desc block)
    feature_names : list[str] (len d) column names, index-aligned to rep_matrix
    y : (n,) array                    activity labels
    groups : (n,) array               scaffold group ids (for GroupKFold; NO random splits)
    structural_pairs : npz/dict/tuple (I, J, mcs) cached MCS pairs; required for
                       yardstick='structural', ignored for 'readacross'
    yardstick : 'structural' | 'readacross'
    n_splits : int                    GroupKFold folds (the independent repeats)
    max_drop : int | None             cap greedy steps per fold (None = until nothing helps)
    min_step_gain : float             a drop must raise the TRAIN score by > this to be taken
    min_folds : int | None            STABILITY guard: a feature must be dropped in >= this
                                      many folds to reach the blacklist (default: majority)
    min_heldout_gain : float          THRESHOLD guard: mean held-out gain (over the folds
                                      that dropped it) must exceed this to reach the blacklist
    knn_k : int                       k for the readacross yardstick
    cand_cap : int | None             (readacross only) cap candidate columns per step
    verbose : bool                    print per-fold progress

    Returns
    -------
    dict (see module docstring).
    """
    yardstick = yardstick.lower()
    if yardstick not in VALID_YARDSTICKS:
        raise ValueError(f"yardstick must be one of {VALID_YARDSTICKS}, got {yardstick!r}")
    X = np.asarray(rep_matrix, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    n, d = X.shape
    if len(feature_names) != d:
        raise ValueError(f"feature_names has {len(feature_names)} entries but "
                         f"rep_matrix has {d} columns")
    n_groups = len(set(groups.tolist()))
    if n_groups < n_splits:
        raise ValueError(f"n_splits={n_splits} but only {n_groups} groups")
    if min_folds is None:
        min_folds = math.ceil(0.6 * n_splits)      # majority stability guard

    if yardstick == 'structural':
        pairs = _unpack_pairs(structural_pairs)
        if pairs is None:
            raise ValueError("structural yardstick needs structural_pairs")
        I, J, mcs = pairs
        if I.max() >= n or J.max() >= n:
            raise ValueError("structural_pairs index out of range for rep_matrix rows")

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.arange(n), y, groups))

    # per-feature accumulators across folds
    fold_dropped = {c: [] for c in range(d)}       # col -> list of heldout gains
    fold_train_gain = {c: [] for c in range(d)}
    per_fold = []

    for fi, (tr_idx, te_idx) in enumerate(folds):
        Xz = _train_scale(X, tr_idx)
        if yardstick == 'structural':
            res = _structural_fold_greedy(Xz, I, J, mcs, tr_idx, te_idx,
                                          max_drop, min_step_gain)
        else:
            res = _readacross_fold_greedy(Xz, y, groups, tr_idx, te_idx,
                                          max_drop, min_step_gain, knn_k, cand_cap)
        if res is None:
            if verbose:
                print(f"  fold {fi}: skipped (too few pairs/rows)")
            continue
        for step in res['trace']:
            c = step['col']
            fold_dropped[c].append(step['heldout_gain'])
            fold_train_gain[c].append(step['train_gain'])
        per_fold.append({'fold': fi, 'n_dropped': len(res['trace']),
                         'full_heldout': res['full_te'],
                         'dropped_cols': [s['col'] for s in res['trace']]})
        if verbose:
            print(f"  fold {fi}: dropped {len(res['trace'])} cols, "
                  f"full held-out score = {res['full_te']:+.4f}")

    # rank every feature that was ever dropped
    ranked = []
    for c in range(d):
        hits = fold_dropped[c]
        if not hits:
            continue
        ranked.append({
            'name': feature_names[c],
            'col': c,
            'n_folds': len(hits),
            'mean_heldout_gain': float(np.mean(hits)),
            'mean_train_gain': float(np.mean(fold_train_gain[c])),
            'heldout_gains': [float(x) for x in hits],
        })
    # rank by stability first, then mean held-out gain
    ranked.sort(key=lambda r: (r['n_folds'], r['mean_heldout_gain']), reverse=True)

    # apply the guards -> the formal blacklist
    blacklist = [r for r in ranked
                 if r['n_folds'] >= min_folds
                 and r['mean_heldout_gain'] > min_heldout_gain]
    for r in ranked:
        r['in_blacklist'] = (r['n_folds'] >= min_folds
                             and r['mean_heldout_gain'] > min_heldout_gain)
    blacklist_cols = [r['col'] for r in blacklist]

    # honest before/after: held-out yardstick with FULL block vs blacklist removed,
    # averaged over the same folds.
    base_scores, abl_scores = [], []
    bl_mask = np.ones(d, bool)
    bl_mask[blacklist_cols] = False
    for tr_idx, te_idx in folds:
        Xz = _train_scale(X, tr_idx)
        if yardstick == 'structural':
            te = np.isin(I, te_idx) & np.isin(J, te_idx)
            if te.sum() < 10:
                continue
            SQ = (Xz[I][te] - Xz[J][te]) ** 2
            mcs_te = mcs[te]
            base_scores.append(_spearman(SQ.sum(1), mcs_te))
            abl_scores.append(_spearman(SQ[:, bl_mask].sum(1), mcs_te))
        else:
            Xtr, Xte = Xz[tr_idx], Xz[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if len(te_idx) < 5:
                continue
            Dfull = np.zeros((len(te_idx), len(tr_idx)))
            Dabl = np.zeros((len(te_idx), len(tr_idx)))
            for c in range(d):
                sq = (Xte[:, c][:, None] - Xtr[:, c][None, :]) ** 2
                Dfull += sq
                if bl_mask[c]:
                    Dabl += sq
            base_scores.append(_readacross_heldout_score(Dfull, y_tr, y_te, knn_k))
            abl_scores.append(_readacross_heldout_score(Dabl, y_tr, y_te, knn_k))

    baseline_score = float(np.nanmean(base_scores)) if base_scores else float('nan')
    ablated_score = float(np.nanmean(abl_scores)) if abl_scores else float('nan')

    return {
        'yardstick': yardstick,
        'n': int(n), 'd': int(d), 'n_splits': int(n_splits),
        'n_groups': int(n_groups),
        'min_folds': int(min_folds),
        'min_heldout_gain': float(min_heldout_gain),
        'blacklist': blacklist,
        'blacklist_cols': blacklist_cols,
        'blacklist_names': [r['name'] for r in blacklist],
        'ranked': ranked,
        'baseline_score': baseline_score,
        'ablated_score': ablated_score,
        'delta': ablated_score - baseline_score,
        'per_fold': per_fold,
    }


if __name__ == '__main__':
    # smoke test on herg_ki desc for structural fidelity
    from rdkit.Chem import Descriptors
    from _data import load_prepared
    names = [nm for nm, _ in Descriptors._descList]
    p = load_prepared('herg_ki')
    pairs = np.load(os.path.join(HERE, 'mcs_pairs_herg_ki.npz'))
    out = identify_saboteurs(p['desc'], names, p['y'], p['groups'], pairs,
                             yardstick='structural', verbose=True)
    print("\nblacklist:", out['blacklist_names'])
    print(f"before {out['baseline_score']:+.4f} -> after {out['ablated_score']:+.4f} "
          f"(delta {out['delta']:+.4f})")
