"""winning_recipe_vs_standard -- THE winning similarity recipe vs the field-standard metrics.

This is the headline comparison of the similarity-metric study, isolated into one
table: does the winning stacked-block recipe beat the two standard fingerprint
similarity metrics that the cheminformatics field uses by default, judged on BOTH
of the study's honest yardsticks?

THE WINNING RECIPE (a stacked, z-scored, equal-block Euclidean space)
  count        = ecfp4_count                (1024-D count Morgan; the base rep)
  physchem_bl  = desc[saboteurs removed]     (RDKit descriptors minus the columns
                 whose PRESENCE lowers MCS structural fidelity -- per-dataset
                 drop-one-greedy blacklist, label-free, on MCS structure only)
  fine2D       = top-100 cols of [atompair_count | torsion], selected LABEL-FREE
                 by |Spearman(|feat diff|, MCS)| (no activity labels touch it)
  + augmentation = pharmacophore3d           (49-D 3D pharmacophore shape block,
                 FULL rows from sim_<ds>_spatial.npz)
This is identical to creative_hybrids_v2's "count+physchem_bl+shape+fine2D [equal]"
combo; the block algebra + selection helpers are imported from that module so the
numbers are guaranteed to match.

THE STANDARD METRICS (the comparators)
  ecfp4:tanimoto        binary Morgan + Tanimoto -- THE field default.
  ecfp4_count:tanimoto  count Morgan + Tanimoto -- the study's canonical BASELINE
                        (deltas are reported against this one).
  ecfp4_count[z-eucl]   count Morgan as a z-scored Euclidean space -- included so
                        a recipe gain is separable from the mere Tanimoto->Euclid
                        coefficient change.

YARDSTICKS (both, exactly as everywhere in this study; scaffold CV, no random)
  ACTIVITY  read-across = scaffold-grouped kNN Spearman rho + RMSE
            (simeval.eval_knn_readacross -- real labels, no model, no circularity)
  STRUCTURE MCS fidelity = Spearman(D[I,J], mcs) over the cached MCS pairs

MOLECULES HELD OUT. The winning recipe picks its descriptor columns and its 100
fine-2D columns by fitting to MCS, and used to be graded on the very pairs that
picked them; the three comparators select no columns, so only the winning row
carried that inflation. The MOLECULES are now split in half by whole
Bemis-Murcko scaffold group (_heldout.scaffold_halves), the two selectors and
the per-column z-score mean/sd are fitted on one half, and EVERY row --
comparators included, so the deltas stay like-for-like -- is scored on the other
half over the cached pairs with both endpoints there. struct_fidelity_insample,
on the select half, sits beside each held-out number so the size of the
optimism is visible.

Datasets: herg_ki, logd (scaffold-capped ~1400 by _data), adme_solubility.

  python winning_recipe_vs_standard.py --all
  python winning_recipe_vs_standard.py --dataset herg_ki
  python winning_recipe_vs_standard.py --all --out winning_recipe_vs_standard.json

Standalone; imports the study's read-only tools only, modifies nothing. ECFP4
Tanimoto is a CANDIDATE metric judged by the yardsticks here, never a judge --
p['tanimoto'] / diag_ecfp_agreement (the diagnostic-only quantities) never enter.
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
from _data import load_prepared
from _nanrank import is_defined
from _heldout import SPLIT_SEED, scaffold_halves, pairs_within, zscore_fit
# reuse the winning recipe's exact block algebra + selection so results are identical
from creative_hybrids_v2 import (load_pharm3d_full, select_fine2d,
                                 desc_blacklist_greedy, struct_fidelity, FINE_K)


def make_D_fit(blocks, fit_idx, weights=None):
    """creative_hybrids_v2.make_D with the z-score mean and sd taken from the
    SELECT rows only. The equal_block scaling (/sqrt(width), width = the number
    of features in the block) and the optional per-block multiplier are
    identical; only the source of the mean and sd differs. Returns the FULL
    (n, n) Euclidean distance matrix -- the caller cuts it."""
    parts = []
    for k, X in enumerate(blocks):
        Z = zscore_fit(X, fit_idx)
        Z = Z / np.sqrt(Z.shape[1])
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return simeval.dist_euclidean(np.hstack(parts))


def score_space_split(name, D, y, groups, sel_idx, sco_idx,
                      sel_pairs, sco_pairs):
    """Both structural numbers for one distance matrix, and read-across on the
    SCORE half's rows so every row in the table is measured over the same
    molecules. struct_fidelity is the HELD-OUT one; struct_fidelity_insample is
    on the half the selection was fitted on."""
    Is, Js, Ms = sel_pairs
    Ic, Jc, Mc = sco_pairs
    D_sel, D_sco = D[np.ix_(sel_idx, sel_idx)], D[np.ix_(sco_idx, sco_idx)]
    rho, negrmse = simeval.eval_knn_readacross(D_sco, y[sco_idx], groups[sco_idx])
    rho_all, _n2 = simeval.eval_knn_readacross(D, y, groups)
    # every rho stays NaN when its Spearman is undefined; the *_defined flags
    # separate that from a measured rho of 0, so a NaN delta below is never read
    # as a zero difference.
    return dict(name=name, readacross_rho=float(rho),
                readacross_defined=is_defined(rho),
                readacross_rmse=float(-negrmse),
                readacross_rho_all=float(rho_all),
                readacross_rho_all_defined=is_defined(rho_all),
                struct_fidelity=struct_fidelity(D_sco, Ic, Jc, Mc),
                struct_fidelity_insample=struct_fidelity(D_sel, Is, Js, Ms))


def run(dataset='herg_ki'):
    t0 = time.time()
    p = load_prepared(dataset)
    n = p['n']
    m = np.load(os.path.join(HERE, f'mcs_pairs_{dataset}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    y, groups = p['y'], p['groups']
    coarse = p['ecfp4_count']
    desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    # MOLECULES split by whole Bemis-Murcko scaffold group. The winning recipe
    # picks descriptor columns and 100 fine-2D columns by fitting to MCS; scoring
    # that fit on the same pairs inflated only the winning row, because the three
    # comparators select no columns at all. So every row -- comparators included
    # -- is now scored on the SCORE half, over the cached pairs whose BOTH
    # endpoints sit there, and the in-sample number is kept beside it.
    sel_idx, sco_idx = scaffold_halves(groups, seed=SPLIT_SEED)
    sel_pairs = pairs_within(I, J, mcs, sel_idx, n)
    sco_pairs = pairs_within(I, J, mcs, sco_idx, n)
    n_straddle = len(I) - len(sel_pairs[0]) - len(sco_pairs[0])
    Is, Js, Ms = sel_pairs

    # ---- build the winning recipe's blocks (label-free selection throughout,
    # and fitted on the SELECT half's molecules only) ----
    blk, trace, kept = desc_blacklist_greedy(desc[sel_idx], Is, Js, Ms)
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all[sel_idx], Is, Js, Ms, FINE_K)
    fine = X_fine_all[:, fidx]
    pharm, n_built = load_pharm3d_full(dataset, n)              # 3D shape (full rows)
    shape_ok = pharm is not None

    print(f'[{dataset}] n={n} scaffolds={p["n_scaffolds"]} MCS_pairs={len(I)}  '
          f'split by scaffold group (seed {SPLIT_SEED}): {len(sel_idx)} select / '
          f'{len(sco_idx)} score molecules, {len(Is)} select / {len(sco_pairs[0])} '
          f'score pairs, {n_straddle} straddling pairs dropped  '
          f'desc {trace["n_total"]}->keep {trace["n_kept"]} '
          f'(saboteurs: {", ".join(trace["dropped_names"]) or "none"})  '
          f'fine2D top-{FINE_K}  '
          f'shape={"FULL n_built=%d" % n_built if shape_ok else "UNAVAILABLE(n_built=%d<%d)" % (n_built, n)}',
          flush=True)

    def sc(name, D):
        return score_space_split(name, D, y, groups, sel_idx, sco_idx,
                                 sel_pairs, sco_pairs)

    rows = []
    # --- STANDARD metrics (the comparators) ---
    rows.append(sc('ecfp4:tanimoto [STANDARD/field-default]',
                   simeval.dist_tanimoto(p['ecfp4'])))
    rows.append(sc('ecfp4_count:tanimoto [STANDARD/BASELINE]',
                   simeval.dist_tanimoto(coarse)))
    rows.append(sc('ecfp4_count [z-eucl]', make_D_fit([coarse], sel_idx)))

    # --- THE WINNING RECIPE ---
    if shape_ok:
        recipe_D = make_D_fit([coarse, physchem_bl, fine, pharm], sel_idx)
        rows.append(sc('WINNING: count+physchem_bl+fine2D+shape[pharmacophore3d]',
                       recipe_D))
    else:
        rows.append(dict(name='WINNING RECIPE [shape UNAVAILABLE -- skipped]',
                         readacross_rho=float('nan'), readacross_defined=False,
                         readacross_rmse=float('nan'),
                         readacross_rho_all=float('nan'),
                         readacross_rho_all_defined=False,
                         struct_fidelity=float('nan'),
                         struct_fidelity_insample=float('nan')))

    # baseline for deltas = the count Tanimoto (canonical study baseline)
    base = rows[1]
    field = rows[0]   # ecfp4:tanimoto, the field default
    winning = rows[-1]

    print(f'\n=== {dataset}: WINNING RECIPE vs STANDARD metrics (n={n}) ===')
    print(f'  {"space":56s} {"read_rho":>9} {"rmse":>7} {"struct_MCS":>11} '
          f'{"struct_insam":>12} {"d_ra":>8} {"d_struct":>9}  {"ra?":>5}')
    for r in rows:
        # A delta is NaN whenever either rho is undefined. It stays NaN -- the
        # 'ra?' column says which of the two terms was finite, so a NaN delta is
        # never read as a zero difference.
        d_ra = r['readacross_rho'] - base['readacross_rho']
        d_sf = r['struct_fidelity'] - base['struct_fidelity']
        ra_tag = ('ok' if is_defined(r['readacross_rho']) else 'UNDEF')
        print(f'  {r["name"]:56s} {r["readacross_rho"]:>9.4f} {r["readacross_rmse"]:>7.3f} '
              f'{r["struct_fidelity"]:>11.4f} {r["struct_fidelity_insample"]:>12.4f} '
              f'{d_ra:>+8.4f} {d_sf:>+9.4f}  {ra_tag}')
    print(f'  (deltas vs BASELINE ecfp4_count:tanimoto  '
          f'read_rho={base["readacross_rho"]:.4f} struct_MCS={base["struct_fidelity"]:.4f}; '
          f'struct_MCS is on the score half, struct_insam on the half the '
          f'column selection was fitted on)')

    if shape_ok:
        print(f'  WINNING vs ecfp4_count:tanimoto  '
              f'd_read={winning["readacross_rho"]-base["readacross_rho"]:+.4f}  '
              f'd_struct={winning["struct_fidelity"]-base["struct_fidelity"]:+.4f}')
        print(f'  WINNING vs ecfp4:tanimoto (field)  '
              f'd_read={winning["readacross_rho"]-field["readacross_rho"]:+.4f}  '
              f'd_struct={winning["struct_fidelity"]-field["struct_fidelity"]:+.4f}')
    print(f'  elapsed {time.time()-t0:.1f}s', flush=True)

    return dict(
        dataset=dataset, n=int(n), scaffolds=int(p['n_scaffolds']),
        mcs_pairs=int(len(I)), shape_available=bool(shape_ok),
        shape_n_built=int(n_built), fine_k=FINE_K, desc_blacklist=trace,
        split=dict(seed=SPLIT_SEED, by='bemis_murcko_scaffold_group',
                   n_select_molecules=int(len(sel_idx)),
                   n_score_molecules=int(len(sco_idx)),
                   n_pairs_total=int(len(I)),
                   n_pairs_select=int(len(Is)),
                   n_pairs_score=int(len(sco_pairs[0])),
                   n_pairs_straddling=int(n_straddle)),
        baseline_ecfp4_count_tanimoto=base, field_default_ecfp4_tanimoto=field,
        winning=winning, rows=rows,
        # read_rho deltas are NaN when either side's rho is undefined; the
        # read_rho_defined flag says whether BOTH terms were finite, so a NaN
        # difference is never mistaken for a zero difference.
        deltas_vs_baseline=(None if not shape_ok else dict(
            read_rho=winning['readacross_rho'] - base['readacross_rho'],
            read_rho_defined=(is_defined(winning['readacross_rho'])
                              and is_defined(base['readacross_rho'])),
            struct_fidelity=winning['struct_fidelity'] - base['struct_fidelity'],
            struct_fidelity_insample=(winning['struct_fidelity_insample']
                                      - base['struct_fidelity_insample']))),
        deltas_vs_field=(None if not shape_ok else dict(
            read_rho=winning['readacross_rho'] - field['readacross_rho'],
            read_rho_defined=(is_defined(winning['readacross_rho'])
                              and is_defined(field['readacross_rho'])),
            struct_fidelity=winning['struct_fidelity'] - field['struct_fidelity'],
            struct_fidelity_insample=(winning['struct_fidelity_insample']
                                      - field['struct_fidelity_insample']))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--all', action='store_true',
                    help='run herg_ki, logd, adme_solubility')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    datasets = ['herg_ki', 'logd', 'adme_solubility'] if args.all else [args.dataset]
    result = {}
    for ds in datasets:
        try:
            result[ds] = run(ds)
        except FileNotFoundError as e:
            print(f'[{ds}] SKIPPED (missing file): {e}', flush=True)
            result[ds] = {'dataset': ds, 'error': str(e)}
    out = args.out or os.path.join(HERE, 'winning_recipe_vs_standard.json')
    json.dump(result, open(out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(out)}')


if __name__ == '__main__':
    main()
