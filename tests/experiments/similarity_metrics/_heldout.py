"""Held-out molecule halves for the similarity-metric scripts.

Several scripts in this directory FIT something on the cached MCS pairs -- the
drop-one descriptor blacklist, the fine-2D column selection, the per-feature
z-score mean/sd -- and then GRADE that fitted thing on the same pairs. The
number that comes out is the maximum of a search, not a measured fidelity.

The repair used throughout is a molecule split, not a pair split. Splitting the
8000 cached pairs 50/50 is not enough: each molecule sits in about eleven pairs,
so the fit half and the score half are drawn from the same molecules and the
selection has still seen every molecule it is graded on. Here the MOLECULES are
split by whole Bemis-Murcko scaffold group, and a pair survives only when BOTH
of its endpoints are on the same side.

  scaffold_halves(groups)      -> (fit_idx, score_idx), disjoint, no scaffold
                                  straddles the two
  pairs_within(I, J, mcs, idx) -> the cached pairs with both endpoints in idx,
                                  re-indexed to address D[np.ix_(idx, idx)]
  zscore_fit(X, fit_idx)       -> z-score every row of X using the mean and sd
                                  of the fit rows only

These three came out of similarity_noise_cleaning.py, which imports them from
here now, so every script that holds molecules out cuts them the same way.

Two positional splits live here as well, for the antibody scripts:

  frac_single_mutation(muts)   -> fraction of variants carrying at most one
                                  mutation. The position grouping and
                                  dist_position_gated both read only muts[i][0],
                                  so both are undefined below ~1.0.
  degenerate_readacross(D, y, groups) -> whether a position-grouped five-nearest-
                                  neighbour score is arithmetic rather than
                                  measured: every train-test distance constant,
                                  or every query drawing the same neighbours.
"""
import numpy as np

SPLIT_SEED = 0        # fixed, so every rerun sees the same fit/score halves


def scaffold_halves(groups, seed=SPLIT_SEED):
    """Split the MOLECULES into a FIT half and a SCORE half by whole
    Bemis-Murcko scaffold group. Same convention as _data._scaffold_subsample:
    sorted unique groups, fixed-seed permutation, groups added whole until the
    molecule count reaches half of n. Returns (fit_idx, score_idx), both sorted
    and disjoint, and no scaffold straddles the two."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rs = np.random.RandomState(seed)
    order = rs.permutation(len(uniq))
    fit_mask = np.zeros(len(groups), bool)
    target = len(groups) // 2
    count = 0
    for gi in order:
        members = np.where(groups == uniq[gi])[0]
        fit_mask[members] = True
        count += len(members)
        if count >= target:
            break
    return np.where(fit_mask)[0], np.where(~fit_mask)[0]


def pairs_within(I, J, mcs, idx, n):
    """The cached MCS pairs whose BOTH endpoints lie in `idx`, re-indexed into
    positions within `idx` so they address the submatrix D[np.ix_(idx, idx)].
    A pair straddling the two halves is dropped -- it has one endpoint the
    selection was fitted on, so it cannot grade the selection."""
    pos = np.full(n, -1, dtype=int)
    pos[idx] = np.arange(len(idx))
    keep = (pos[I] >= 0) & (pos[J] >= 0)
    return pos[I[keep]], pos[J[keep]], mcs[keep]


def pairs_within_mask(I, J, idx, n):
    """The boolean mask `pairs_within` keeps, for callers that have further
    per-pair vectors (the activity gap `dy`, say) to cut the same way."""
    inside = np.zeros(n, bool)
    inside[idx] = True
    return inside[I] & inside[J]


def zscore_fit(X, fit_idx):
    """z-score every row of X using the mean and sd of the FIT rows only. The
    plain zscore used elsewhere in this directory takes them from all rows,
    which lets the score rows set their own scale."""
    X = np.asarray(X, dtype=np.float64)
    mu = X[fit_idx].mean(0)
    sd = X[fit_idx].std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------- positional (antibody) splits
def frac_single_mutation(muts):
    """Fraction of variants carrying at most ONE mutation. The position grouping
    (`muts[i][0][0]`) and SR.dist_position_gated both read only the first
    mutation, so both are defined only on a genuinely single-mutation library."""
    return float(np.mean([len(m) <= 1 for m in muts]))


def degenerate_readacross(D, y, groups, k=5, n_splits=5, tol=1e-9):
    """Is a group-held-out five-nearest-neighbour score arithmetic rather than
    measured? Two ways it can be, both seen on the koenig position-grouped
    folds:

      constant_train_test   every train-test distance in every fold is the same
                            number, so the neighbours are picked by tie-break
                            order alone (onehot: two single mutants at different
                            positions always differ at exactly four entries).
      same_neighbours       every held-out query in a fold draws the SAME five
                            training rows (a positional representation shares no
                            non-zero feature with a held-out position, so the
                            ordering is set by the training rows' own norms).

    Returns a dict of the two flags plus the fold count each was measured over.
    Nothing is dropped here -- the caller records the flags beside the score."""
    from sklearn.model_selection import GroupKFold
    n = len(y)
    n_groups = len(set(np.asarray(groups).tolist()))
    splits = min(n_splits, n_groups)
    const_folds, same_folds, n_folds = 0, 0, 0
    if splits >= 2:
        for tr, te in GroupKFold(n_splits=splits).split(np.arange(n), y, groups):
            n_folds += 1
            block = D[np.ix_(te, tr)]
            if float(block.max() - block.min()) <= tol:
                const_folds += 1
            kk = min(k, len(tr))
            nb = np.argsort(block, axis=1, kind='stable')[:, :kk]
            if len(nb) and np.all(nb == nb[0]):
                same_folds += 1
    return dict(n_folds=n_folds,
                constant_train_test=(n_folds > 0 and const_folds == n_folds),
                same_neighbours=(n_folds > 0 and same_folds == n_folds),
                constant_train_test_folds=const_folds,
                same_neighbours_folds=same_folds)
