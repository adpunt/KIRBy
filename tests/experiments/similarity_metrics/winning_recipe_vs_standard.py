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
# reuse the winning recipe's exact block algebra + selection so results are identical
from creative_hybrids_v2 import (make_D, load_pharm3d_full, select_fine2d,
                                 desc_blacklist_greedy, score_space, FINE_K)


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

    # ---- build the winning recipe's blocks (label-free selection throughout) ----
    blk, trace, kept = desc_blacklist_greedy(desc, I, J, mcs)   # saboteur blacklist
    physchem_bl = desc[:, kept]
    fidx = select_fine2d(X_fine_all, I, J, mcs, FINE_K)         # fine-2D structural
    fine = X_fine_all[:, fidx]
    pharm, n_built = load_pharm3d_full(dataset, n)              # 3D shape (full rows)
    shape_ok = pharm is not None

    print(f'[{dataset}] n={n} scaffolds={p["n_scaffolds"]} MCS_pairs={len(I)}  '
          f'desc {trace["n_total"]}->keep {trace["n_kept"]} '
          f'(saboteurs: {", ".join(trace["dropped_names"]) or "none"})  '
          f'fine2D top-{FINE_K}  '
          f'shape={"FULL n_built=%d" % n_built if shape_ok else "UNAVAILABLE(n_built=%d<%d)" % (n_built, n)}',
          flush=True)

    rows = []
    # --- STANDARD metrics (the comparators) ---
    rows.append(score_space('ecfp4:tanimoto [STANDARD/field-default]',
                            simeval.dist_tanimoto(p['ecfp4']), y, groups, I, J, mcs))
    rows.append(score_space('ecfp4_count:tanimoto [STANDARD/BASELINE]',
                            simeval.dist_tanimoto(coarse), y, groups, I, J, mcs))
    rows.append(score_space('ecfp4_count [z-eucl]',
                            make_D([coarse]), y, groups, I, J, mcs))

    # --- THE WINNING RECIPE ---
    if shape_ok:
        recipe_D = make_D([coarse, physchem_bl, fine, pharm])
        rows.append(score_space(
            'WINNING: count+physchem_bl+fine2D+shape[pharmacophore3d]',
            recipe_D, y, groups, I, J, mcs))
    else:
        rows.append(dict(name='WINNING RECIPE [shape UNAVAILABLE -- skipped]',
                         readacross_rho=float('nan'), readacross_rmse=float('nan'),
                         struct_fidelity=float('nan')))

    # baseline for deltas = the count Tanimoto (canonical study baseline)
    base = rows[1]
    field = rows[0]   # ecfp4:tanimoto, the field default
    winning = rows[-1]

    print(f'\n=== {dataset}: WINNING RECIPE vs STANDARD metrics (n={n}) ===')
    print(f'  {"space":56s} {"read_rho":>9} {"rmse":>7} {"struct_MCS":>11} '
          f'{"d_ra":>8} {"d_struct":>9}')
    for r in rows:
        d_ra = r['readacross_rho'] - base['readacross_rho']
        d_sf = r['struct_fidelity'] - base['struct_fidelity']
        print(f'  {r["name"]:56s} {r["readacross_rho"]:>9.4f} {r["readacross_rmse"]:>7.3f} '
              f'{r["struct_fidelity"]:>11.4f} {d_ra:>+8.4f} {d_sf:>+9.4f}')
    print(f'  (deltas vs BASELINE ecfp4_count:tanimoto  '
          f'read_rho={base["readacross_rho"]:.4f} struct_MCS={base["struct_fidelity"]:.4f})')

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
        baseline_ecfp4_count_tanimoto=base, field_default_ecfp4_tanimoto=field,
        winning=winning, rows=rows,
        deltas_vs_baseline=(None if not shape_ok else dict(
            read_rho=winning['readacross_rho'] - base['readacross_rho'],
            struct_fidelity=winning['struct_fidelity'] - base['struct_fidelity'])),
        deltas_vs_field=(None if not shape_ok else dict(
            read_rho=winning['readacross_rho'] - field['readacross_rho'],
            struct_fidelity=winning['struct_fidelity'] - field['struct_fidelity'])))


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
