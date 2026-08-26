"""Honest re-run of the winning recipe — fixes the STRUCTURE circularity.

The prior winning_recipe / creative_hybrids_v2 selected the fine-2D features and the
desc blacklist using the SAME MCS pairs it then scored structural fidelity on
(select-on-the-data-you-score-on -> inflated structure number). Here the MCS pairs
are split 50/50: features are SELECTED on the SELECT half, structural fidelity is
SCORED on the disjoint SCORE half. Read-across is already honest (scaffold GroupKFold,
label-free selection). This gives the REAL "does the recipe beat the baseline on both
yardsticks, or only activity" answer.
"""
import os, sys, json, time
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simeval  # noqa: E402
from _data import load_prepared  # noqa: E402
from _nanrank import is_defined  # noqa: E402
from creative_hybrids_v2 import (  # noqa: E402
    make_D, load_pharm3d_full, select_fine2d, desc_blacklist_greedy, FINE_K)


def struct_on(D, I, J, mcs):
    return float(spearmanr(D[I, J], mcs)[0])


def run(dataset):
    t0 = time.time()
    p = load_prepared(dataset)
    n = p['n']
    m = np.load(os.path.join(HERE, f'mcs_pairs_{dataset}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    # 50/50 split of the MCS pairs: SELECT half (choose features) vs SCORE half (grade structure)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(I)); half = len(I) // 2
    sel, sco = perm[:half], perm[half:]
    Is, Js, Ms = I[sel], J[sel], mcs[sel]        # selection pairs
    Ic, Jc, Mc = I[sco], J[sco], mcs[sco]        # scoring pairs (disjoint)

    y, groups = p['y'], p['groups']
    coarse = p['ecfp4_count']; desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    # SELECT features on the SELECT half only
    blk, trace, kept = desc_blacklist_greedy(desc, Is, Js, Ms)
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all, Is, Js, Ms, FINE_K)
    fine = X_fine_all[:, fidx]
    pharm, n_built = load_pharm3d_full(dataset, n)
    shape_ok = pharm is not None

    print(f'[{dataset}] n={n} pairs={len(I)} (sel={len(sel)}/sco={len(sco)}) '
          f'desc keep {trace["n_kept"]}/{trace["n_total"]} shape={"FULL" if shape_ok else "NONE"}', flush=True)

    def score(name, D):
        rho, negrmse = simeval.eval_knn_readacross(D, y, groups)   # honest (scaffold CV)
        # read_rho stays NaN when the Spearman is undefined; read_rho_defined
        # separates that from a measured rho of 0. The d_read printed below is
        # then NaN as well, never a zero difference.
        return dict(name=name, read_rho=float(rho), read_rho_defined=is_defined(rho),
                    struct_select=struct_on(D, Is, Js, Ms),        # in-sample (for comparison)
                    struct_honest=struct_on(D, Ic, Jc, Mc))        # HELD-OUT (the real number)

    rows = [score('count_tanimoto[BASELINE]', simeval.dist_tanimoto(coarse)),
            score('ecfp4:tanimoto[field]', simeval.dist_tanimoto(p['ecfp4']))]
    if shape_ok:
        rows.append(score('WINNING count+physchem_bl+fine2D+shape',
                          make_D([coarse, physchem_bl, pharm, fine], weights=[0.5, 1, 1, 1])))

    base = rows[0]
    print(f'  {"space":42s} {"read":>8} {"struct_SELECT":>13} {"struct_HELDOUT":>14}')
    for r in rows:
        print(f'  {r["name"]:42s} {r["read_rho"]:>8.4f} {r["struct_select"]:>13.4f} {r["struct_honest"]:>14.4f}')
    if shape_ok:
        w = rows[-1]
        print(f'  WINNING vs baseline: d_read={w["read_rho"]-base["read_rho"]:+.4f}  '
              f'd_struct(in-sample)={w["struct_select"]-base["struct_select"]:+.4f}  '
              f'd_struct(HELD-OUT,real)={w["struct_honest"]-base["struct_honest"]:+.4f}')
    print(f'  elapsed {time.time()-t0:.1f}s', flush=True)
    return dict(dataset=dataset, n=int(n), rows=rows,
                desc_kept=int(trace['n_kept']), desc_total=int(trace['n_total']))


if __name__ == '__main__':
    out = {}
    for ds in (sys.argv[1:] or ['herg_ki', 'logd', 'adme_solubility']):
        out[ds] = run(ds)
    json.dump(out, open(os.path.join(HERE, 'winning_recipe_honest.json'), 'w'), indent=1)
    print('\nwrote winning_recipe_honest.json')
