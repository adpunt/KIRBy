"""Leakage-safe, label-blind structural feature selection for similarity recipes.

Two column selectors, ported from the standalone study
(`creative_hybrids_v2.desc_blacklist_greedy` and `.select_fine2d`) into the
framework as per-fold rep factories. Both choose columns using the MCS
structural yardstick ONLY — they never read activity labels — and both are fit
on the TRAINING fold's MCS pairs alone, then applied to the validation fold
without refitting. That is the same leakage-safe contract the framework's other
per-fold factories use (`make_sali_cliff_factory`, `make_pca_factory`):

    factory(train_idx, val_idx, train_labels) -> (train_feats, val_feats)

Structure (the fingerprint/descriptor matrices) is precomputed once over all
rows — structure is not the label, so that is safe. The *selection* is what must
stay in-fold: pairs are restricted to train endpoints before any column is
scored, so no held-out row influences which columns survive.

  saboteur-drop : start from all 208 RDKit descriptors, greedily drop the single
                  column whose removal most RAISES Spearman(euclidean-dist, MCS)
                  until nothing helps. Keeps the physchem columns that don't hurt
                  structural fidelity.
  fine2D        : rank [atompair_count | torsion] columns by |Spearman(|feat
                  diff|, MCS)| on the train pairs, keep the top-k.
"""
import numpy as np
from scipy.stats import spearmanr, rankdata

from kirby.similarity.mcs import restrict_pairs_to_rows


# --------------------------------------------------------------- shared helpers
def _zscore_with_stats(X, mu, sd):
    """Z-score X using externally supplied (train-fold) mean/std. sd==0 -> 1."""
    X = np.asarray(X, dtype=np.float64)
    sd = np.array(sd, dtype=np.float64)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def _spearman_cols(DF, t):
    """Spearman of each column of DF (m, d) against vector t (m,) -> (d,).

    Identical to rankers._spearman_cols in the standalone study.
    """
    tr = rankdata(t); tr = tr - tr.mean()
    Rr = np.apply_along_axis(rankdata, 0, DF)
    Rr = Rr - Rr.mean(0)
    num = Rr.T @ tr
    den = np.sqrt((Rr ** 2).sum(0) * (tr ** 2).sum())
    out = np.zeros(DF.shape[1]); nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


# --------------------------------------------- saboteur-drop (desc blacklist)
def select_saboteur_drop(Z, I, J, mcs, max_drop=None, val_frac=0.5, seed=0,
                         patience=15, min_pairs=20):
    """Greedy structural saboteur blacklist, validated on held-out pairs.

    Port of creative_hybrids_v2.desc_blacklist_greedy with one correction. The
    original picks each drop by the improvement it makes on the SAME pairs it
    scores, and stops only when no further drop helps those pairs. That has no
    defence against noise: on random data with no structural signal at all it
    drops 136 of 208 columns and reports Spearman +0.007 -> +0.251, which is
    +0.010 -> +0.012 on pairs it never saw. The whole gain is fitted noise.

    So the pairs are split in two. The greedy still PICKS each drop on the fit
    half, but the column set is CHOSEN by the held-out half: we keep the snapshot
    with the best held-out fidelity, and stop after `patience` drops with no
    held-out improvement. When nothing beats dropping nothing, nothing is
    dropped. This is model selection on held-out data, the same discipline the
    rest of the framework uses; it changes which columns survive, so results
    produced by the original procedure do not carry over.

    Both halves come from the caller's TRAIN-restricted pairs, so no held-out
    MOLECULE is involved either way — this is about not fitting pair noise.

    Args:
        Z: (n, d) z-scored descriptor matrix (stats fit on train rows).
        I, J: train-restricted pair endpoint indices into Z's rows.
        mcs: (n_pairs,) MCS distance for each pair.
        max_drop: cap on columns dropped (default d - 2).
        val_frac: fraction of pairs held out to choose the column set.
        seed: RNG seed for the deterministic pair split.
        patience: stop after this many drops with no held-out improvement.
        min_pairs: if either half is smaller than this, keep every column.

    Returns:
        (kept_mask (d,) bool, blacklist (k,) int, trace dict).
    """
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    n_pairs = len(I)
    keep_all = (np.ones(d, bool), np.array([], dtype=int))

    perm = np.random.RandomState(seed).permutation(n_pairs)
    n_val = int(round(n_pairs * val_frac))
    val_ix, fit_ix = perm[:n_val], perm[n_val:]
    if len(fit_ix) < min_pairs or len(val_ix) < min_pairs:
        k, b = keep_all
        return k, b, {'base_struct': float('nan'), 'final_struct': float('nan'),
                      'n_total': int(d), 'n_dropped': 0, 'n_kept': int(d),
                      'stopped': 'too_few_pairs'}

    SQ = (Z[I] - Z[J]) ** 2                      # (n_pairs, d) per-col sq diff
    SQ_fit, SQ_val = SQ[fit_ix], SQ[val_ix]
    mcs_fit, mcs_val = np.asarray(mcs)[fit_ix], np.asarray(mcs)[val_ix]

    def sf(sumsq, m):
        r = spearmanr(sumsq, m)[0]
        return float('nan') if not np.isfinite(r) else float(r)

    kept = np.ones(d, bool)
    sum_fit, sum_val = SQ_fit.sum(1), SQ_val.sum(1)
    base_fit, base_val = sf(sum_fit, mcs_fit), sf(sum_val, mcs_val)

    # The incumbent is "drop nothing" — a drop must beat it on the held-out half.
    best_val, best_kept, best_n = base_val, kept.copy(), 0
    dropped, since_improve = [], 0
    if max_drop is None:
        max_drop = d - 2

    for _ in range(max_drop):
        cur_fit = sf(sum_fit, mcs_fit)
        best_gain, best_c = 0.0, -1
        for c in np.where(kept)[0]:
            g = sf(sum_fit - SQ_fit[:, c], mcs_fit) - cur_fit
            if g > best_gain + 1e-9:
                best_gain, best_c = g, int(c)
        if best_c < 0:
            break
        kept[best_c] = False
        sum_fit = sum_fit - SQ_fit[:, best_c]
        sum_val = sum_val - SQ_val[:, best_c]
        v = sf(sum_val, mcs_val)
        dropped.append({'col': best_c, 'fit_gain': best_gain, 'val_struct': v})
        if np.isfinite(v) and v > best_val + 1e-9:
            best_val, best_kept, best_n, since_improve = v, kept.copy(), len(dropped), 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    kept = best_kept
    blacklist = np.array([dd['col'] for dd in dropped[:best_n]], dtype=int)
    trace = {'base_struct': base_fit, 'base_struct_heldout': base_val,
             'final_struct_heldout': best_val,
             'n_total': int(d), 'n_dropped': int(best_n),
             'n_kept': int(kept.sum()), 'n_considered': len(dropped)}
    return kept, blacklist, trace


# ------------------------------------------------------ fine-2D column selection
def select_fine2d(X_fine, I, J, mcs, top_k):
    """Rank X_fine columns label-free by |Spearman(|feat diff|, MCS)| on the
    train pairs and return the top_k column indices.

    Port of creative_hybrids_v2.select_fine2d (pairs are train-restricted here).
    """
    DF = np.abs(X_fine[I].astype(np.float64) - X_fine[J].astype(np.float64))
    rho = _spearman_cols(DF, mcs)
    k = int(min(top_k, X_fine.shape[1]))
    return np.argsort(-np.abs(rho))[:k]


# ------------------------------------------------------------ factory wrappers
def make_saboteur_drop_factory(smiles_list, dataset_name, desc_fn=None,
                               n_pairs=8000, seed=0, max_drop=None):
    """Per-fold saboteur-dropped descriptor block (leakage-safe).

    Precomputes the 208-D descriptor matrix once (structure only -> safe) and the
    dataset's MCS-pair cache. Per fold: z-scores with TRAIN-row stats, restricts
    MCS pairs to train endpoints, greedily drops structural saboteurs, and returns
    the surviving RAW descriptor columns for train and val (widths agree within a
    fold; may differ across folds, like make_pca_factory).

    Returns: factory(train_idx, val_idx, train_labels) -> (train, val).
    """
    from kirby.similarity.blocks import get_block
    from kirby.similarity.mcs import get_mcs_pairs

    desc_full = (np.asarray(desc_fn(smiles_list), dtype=np.float64)
                 if desc_fn is not None else get_block('desc', smiles_list))
    pairs = get_mcs_pairs(dataset_name, smiles_list, n_pairs=n_pairs, seed=seed)

    def factory(train_idx, val_idx, train_labels):
        tr = np.asarray(train_idx)
        mu = desc_full[tr].mean(0)
        sd = desc_full[tr].std(0)
        Z = _zscore_with_stats(desc_full, mu, sd)
        tp = restrict_pairs_to_rows(pairs, tr)
        if len(tp['I']) < 2:
            kept = np.ones(desc_full.shape[1], bool)
        else:
            kept, _, _ = select_saboteur_drop(Z, tp['I'], tp['J'], tp['mcs'],
                                              max_drop=max_drop)
        return (desc_full[tr][:, kept].astype(np.float32),
                desc_full[val_idx][:, kept].astype(np.float32))

    return factory


def make_fine2d_factory(smiles_list, dataset_name, top_k=100,
                        n_pairs=8000, seed=0):
    """Per-fold fine-2D block: top-k of [atompair_count | torsion] by MCS fidelity.

    Precomputes the concatenated fine-2D matrix once (structure only -> safe) and
    the dataset's MCS-pair cache. Per fold: restricts MCS pairs to train, ranks
    columns by |Spearman(|feat diff|, MCS)|, keeps the top-k RAW columns for train
    and val (same columns for both).

    Returns: factory(train_idx, val_idx, train_labels) -> (train, val).
    """
    from kirby.similarity.blocks import get_block
    from kirby.similarity.mcs import get_mcs_pairs

    # fine-2D = [atompair_count(1024) | torsion-bits(1024)]; torsion at 1024 bits
    # to match the pinned recipe geometry (create_top_torsion defaults to 2048).
    X_fine = get_block('fine', smiles_list)
    pairs = get_mcs_pairs(dataset_name, smiles_list, n_pairs=n_pairs, seed=seed)

    def factory(train_idx, val_idx, train_labels):
        tr = np.asarray(train_idx)
        tp = restrict_pairs_to_rows(pairs, tr)
        if len(tp['I']) < 2:
            cols = np.arange(min(top_k, X_fine.shape[1]))
        else:
            cols = select_fine2d(X_fine, tp['I'], tp['J'], tp['mcs'], top_k)
        return (X_fine[tr][:, cols].astype(np.float32),
                X_fine[val_idx][:, cols].astype(np.float32))

    return factory
