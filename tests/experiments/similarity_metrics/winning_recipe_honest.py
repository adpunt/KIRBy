"""Honest re-run of the winning recipe — fixes the STRUCTURE circularity.

The prior winning_recipe / creative_hybrids_v2 selected the fine-2D features and the
desc blacklist using the SAME MCS pairs it then scored structural fidelity on
(select-on-the-data-you-score-on -> inflated structure number).

Splitting the 8000 cached PAIRS 50/50, which this file used to do, does not fix
that. Each molecule sits in about eleven pairs, so nearly every molecule appears
in both halves and the selection has still seen every molecule it is graded on.
The z-scoring was worse still: its per-column mean and sd came from all n
molecules, so even a pair split by molecule would have left the scale fitted on
the scored rows.

So the MOLECULES are split, by whole Bemis-Murcko scaffold group
(_heldout.scaffold_halves, the same cut similarity_noise_cleaning.py uses), and a
cached pair survives only when BOTH its endpoints land on one side:

  SELECT half   the desc drop-one blacklist, the fine-2D column ranking and the
                per-column z-score mean/sd are all fitted here. The drop-one
                blacklist is kirby.similarity.selection.select_saboteur_drop,
                which splits these pairs again and keeps the column set that
                scores best on the half it did not fit -- the version of the
                procedure whose gain survives on pairs it never saw.
  SCORE half    struct_reselected is measured here, on the pairs with both endpoints
                here, over the distance submatrix cut to these rows.

struct_select, the number on the select half, is kept beside it so the size of
the optimism is visible. Read-across is scored on the score half's rows under
their own scaffold GroupKFold; read_rho_all, on every row, is kept beside it —
the selection is label-free, so read-across was never the circular half.
"""
import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
import simeval  # noqa: E402
from _data import load_prepared  # noqa: E402
from _nanrank import is_defined  # noqa: E402
from _heldout import (SPLIT_SEED, scaffold_halves, pairs_within,  # noqa: E402
                      zscore_fit)
from creative_hybrids_v2 import load_pharm3d_full, select_fine2d, FINE_K  # noqa: E402
# The descriptor drop comes from src/, not from creative_hybrids_v2. That
# module's desc_blacklist_greedy picks each drop on the same pairs it scores and
# stops on them too, which on data with no structural signal drops 136 of 208
# descriptors for +0.007 -> +0.251 in-sample and +0.010 -> +0.012 on pairs it
# never saw. select_saboteur_drop splits the select half's pairs and chooses the
# surviving set on the half it did not fit, so the column set is picked on pairs
# it was not graded on.
from kirby.similarity.selection import select_saboteur_drop  # noqa: E402


def struct_on(D, I, J, mcs):
    return float(spearmanr(D[I, J], mcs)[0])


def make_D_fit(blocks, fit_idx, weights=None):
    """creative_hybrids_v2.make_D with the z-score mean and sd taken from the
    SELECT rows only. The equal_block scaling (/sqrt(width), width = the number
    of features in the block) and the optional per-block multiplier are
    identical; only the source of the mean and sd differs. Returns the FULL
    (n, n) Euclidean distance matrix — the caller cuts it."""
    parts = []
    for k, X in enumerate(blocks):
        Z = zscore_fit(X, fit_idx)
        Z = Z / np.sqrt(Z.shape[1])
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return simeval.dist_euclidean(np.hstack(parts))


def run(dataset):
    t0 = time.time()
    p = load_prepared(dataset)
    n = p['n']
    from mcs_cache import load_pairs
    m = load_pairs(dataset, p['smiles'])
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    y, groups = p['y'], p['groups']

    # MOLECULES split by whole scaffold group; a pair counts only when both of
    # its endpoints are on one side, so no pair grades a selection that saw
    # either of its molecules.
    sel_idx, sco_idx = scaffold_halves(groups, seed=SPLIT_SEED)
    Is, Js, Ms = pairs_within(I, J, mcs, sel_idx, n)   # selection pairs
    Ic, Jc, Mc = pairs_within(I, J, mcs, sco_idx, n)   # scoring pairs (disjoint)
    sel_cut, sco_cut = np.ix_(sel_idx, sel_idx), np.ix_(sco_idx, sco_idx)
    y_sc, g_sc = y[sco_idx], groups[sco_idx]

    coarse = p['ecfp4_count']; desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    # SELECT features on the SELECT half's molecules only. Both selectors are
    # handed desc / X_fine cut to those rows, and pair indices local to them, so
    # neither ever reads a molecule from the score half.
    # select_saboteur_drop takes an ALREADY z-scored matrix, so the scale is fit
    # on the select half's molecules here, exactly as make_D_fit does below.
    kept, blk, trace = select_saboteur_drop(zscore_fit(desc, sel_idx)[sel_idx],
                                            Is, Js, Ms)
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all[sel_idx], Is, Js, Ms, FINE_K)
    fine = X_fine_all[:, fidx]
    pharm, n_built = load_pharm3d_full(dataset, n)
    shape_ok = pharm is not None

    n_straddle = len(I) - len(Is) - len(Ic)
    print(f'[{dataset}] n={n} molecules split by scaffold group (seed {SPLIT_SEED}): '
          f'{len(sel_idx)} select / {len(sco_idx)} score; of {len(I)} cached pairs '
          f'{len(Is)} lie inside select, {len(Ic)} inside score, {n_straddle} straddle '
          f'and are dropped; desc keep {trace["n_kept"]}/{trace["n_total"]} '
          f'shape={"FULL" if shape_ok else "NONE"}', flush=True)

    def score(name, D):
        # read-across on the SCORE half's rows, under their own scaffold folds.
        rho, _negrmse = simeval.eval_knn_readacross(D[sco_cut], y_sc, g_sc)
        rho_all, _n2 = simeval.eval_knn_readacross(D, y, groups)
        # every rho stays NaN when its Spearman is undefined; the *_defined flags
        # separate that from a measured rho of 0, so a NaN d_read below is never
        # read as a zero difference.
        return dict(name=name,
                    read_rho=float(rho), read_rho_defined=is_defined(rho),
                    read_rho_all=float(rho_all),
                    read_rho_all_defined=is_defined(rho_all),
                    struct_select=struct_on(D[sel_cut], Is, Js, Ms),  # in-sample
                    struct_reselected=struct_on(D[sco_cut], Ic, Jc, Mc))  # HELD-OUT

    rows = [score('count_tanimoto[BASELINE]', simeval.dist_tanimoto(coarse)),
            score('ecfp4:tanimoto[field]', simeval.dist_tanimoto(p['ecfp4']))]
    if shape_ok:
        rows.append(score('WINNING count+physchem_bl+fine2D+shape',
                          make_D_fit([coarse, physchem_bl, pharm, fine], sel_idx,
                                     weights=[0.5, 1, 1, 1])))

    base = rows[0]
    print(f'  {"space":42s} {"read":>8} {"struct_SELECT":>13} {"struct_HELDOUT":>14}')
    for r in rows:
        print(f'  {r["name"]:42s} {r["read_rho"]:>8.4f} {r["struct_select"]:>13.4f} {r["struct_reselected"]:>14.4f}')
    if shape_ok:
        w = rows[-1]
        print(f'  WINNING vs baseline: d_read={w["read_rho"]-base["read_rho"]:+.4f}  '
              f'd_struct(in-sample)={w["struct_select"]-base["struct_select"]:+.4f}  '
              f'd_struct(HELD-OUT,real)={w["struct_reselected"]-base["struct_reselected"]:+.4f}')
    print(f'  elapsed {time.time()-t0:.1f}s', flush=True)
    return dict(dataset=dataset, n=int(n), rows=rows,
                desc_kept=int(trace['n_kept']), desc_total=int(trace['n_total']),
                split=dict(seed=SPLIT_SEED, by='bemis_murcko_scaffold_group',
                           n_select_molecules=int(len(sel_idx)),
                           n_score_molecules=int(len(sco_idx)),
                           n_pairs_total=int(len(I)),
                           n_pairs_select=int(len(Is)),
                           n_pairs_score=int(len(Ic)),
                           n_pairs_straddling=int(n_straddle)))


if __name__ == '__main__':
    out = {}
    for ds in (sys.argv[1:] or ['herg_ki', 'logd', 'adme_solubility']):
        out[ds] = run(ds)
    json.dump(out, open(os.path.join(HERE, 'winning_recipe_reselected.json'), 'w'), indent=1)
    print('\nwrote winning_recipe_reselected.json')
