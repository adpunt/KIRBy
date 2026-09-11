"""creative_hybrids_v2 -- creative stacked similarity spaces, TWO fixes applied.

This re-runs the CREATIVE-HYBRIDS idea (do heterogeneous stacked similarity
blocks beat a plain count fingerprint?) with the two fixes the study asked for:

  FIX 1  FULL 3D shape block. The pharm3d (49-D 3D pharmacophore) block is loaded
         from sim_<ds>_spatial.npz for ALL rows (the herg_ki / logd / adme
         spatial caches are now complete: n_built == n). No row subsetting, so
         the SHAPE combos are scored full-scale like every other combo.

  FIX 2  SABOTEUR-BLACKLISTED physchem block. The desc (RDKit physicochemical)
         block contains descriptors whose PRESENCE lowers MCS structural
         fidelity. We recompute, PER DATASET, a drop-one-greedy blacklist on
         STRUCTURAL fidelity (repeatedly remove the single descriptor whose
         removal most raises Spearman(euclidean-dist, MCS) until nothing helps),
         and build physchem_blacklisted = desc with those columns removed.
         (Same procedure feature_ablation.py used; recomputed here so the script
         is self-contained and covers adme_solubility, which the ablation JSON
         never scored.)

YARDSTICKS (both, exactly as everywhere in this study; scaffold CV, no random):
  ACTIVITY  read-across = scaffold-grouped kNN Spearman rho (simeval.eval_knn_readacross)
  STRUCTURE MCS fidelity = Spearman(D[I,J], mcs) over the cached MCS pairs

BLOCKS
  count             = ecfp4_count            (1024 count fp; THE baseline rep)
  physchem_full     = desc                   (208 RDKit descriptors, all)
  physchem_blacklisted = desc minus the per-dataset drop-one saboteur columns
  shape             = pharm3d                (49-D 3D pharmacophore, FULL rows)
  fine2D            = top-100 cols of [atompair_count | torsion], selected
                      LABEL-FREE by |Spearman(|feat diff|, MCS)| (no activity
                      labels touch selection -> read-across stays honest)

Each block is z-scored per feature; scheme='equal_block' scales every block to
unit total variance (/sqrt(d_block)) so a wide block cannot swamp a narrow one.
Distance on the assembled space is Euclidean.

COMBOS (each in equal-weight AND coarse-down-weight x0.5 on the count block):
  count                                          (single block; weighting is moot)
  count + physchem_full
  count + physchem_blacklisted
  count + shape
  count + physchem_blacklisted + shape + fine2D

BASELINE to beat = count alone with Tanimoto (ecfp4_count:tanimoto). Deltas on
BOTH yardsticks are reported against it. count-alone as z-scored Euclidean is
also printed so a block gain is separable from the mere coefficient change.

  python creative_hybrids_v2.py                        # full herg_ki
  python creative_hybrids_v2.py --dataset logd
  python creative_hybrids_v2.py --dataset adme_solubility
  python creative_hybrids_v2.py --all --out creative_hybrids_v2.json

Standalone; imports the shared read-only tools only, modifies nothing.
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
import rankers
from _data import load_prepared
from _nanrank import rank_desc, is_defined

from rdkit.Chem import Descriptors
DESC_NAMES = [n for n, _ in Descriptors._descList]   # 208, matches desc matrix

FINE_K = 100   # fine-2D budget (matches feature_ablation)


# ---------------------------------------------------------------- block algebra
def _zscore(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def make_D(blocks, weights=None, scheme='equal_block'):
    """Assemble a Euclidean distance matrix from raw block matrices. equal_block
    scales each block to unit total variance; weights is an optional per-block
    multiplier on top (used to down-weight the coarse count block)."""
    parts = []
    for k, X in enumerate(blocks):
        Z = _zscore(X)
        if scheme == 'equal_block':
            Z = Z / np.sqrt(Z.shape[1])
        w = 1.0 if weights is None else float(weights[k])
        parts.append(Z * w)
    return simeval.dist_euclidean(np.hstack(parts))


# ----------------------------------------------------------- SHAPE (3D) loader
def load_pharm3d_full(dataset, n):
    """Load the full pharm3d block from sim_<ds>_spatial.npz. Requires the cache
    to cover ALL n rows (n_built >= n); otherwise returns (None, n_built) and the
    caller reports SHAPE unavailable rather than silently subsetting."""
    path = os.path.join(HERE, f'sim_{dataset}_spatial.npz')
    if not os.path.exists(path):
        return None, 0
    z = np.load(path, allow_pickle=True)
    if 'pharm3d' not in z.files:
        return None, 0
    n_built = int(z['n_built']) if 'n_built' in z.files else int(z['pharm3d'].shape[0])
    if n_built < n:
        return None, n_built
    return np.asarray(z['pharm3d'], dtype=np.float64)[:n], n_built


# --------------------------------------------------- fine-2D structural selection
def select_fine2d(X_fine, I, J, mcs, top_k):
    """Rank X_fine columns label-free by |Spearman(|feat diff|, MCS)| on the MCS
    pairs (rankers._spearman_cols) and return the top_k indices."""
    DF = np.abs(X_fine[I].astype(np.float64) - X_fine[J].astype(np.float64))
    rho = rankers._spearman_cols(DF, mcs)
    return np.argsort(-np.abs(rho))[:top_k]


# ------------------------------ FIX 2: per-dataset desc saboteur drop-one-greedy
def desc_blacklist_greedy(desc, I, J, mcs, max_drop=None):
    """Greedy blacklist on STRUCTURAL fidelity. Start from all 208 z-scored
    descriptors; repeatedly drop the single column whose removal most raises
    Spearman(euclidean-dist, MCS), until nothing helps. Spearman is invariant to
    the final sqrt, so work with per-pair squared-difference SUMS (exact, fast).
    Returns (blacklist_idx array, trace dict)."""
    Z = _zscore(desc)                         # (n, 208)
    SQ = (Z[I] - Z[J]) ** 2                    # (n_pairs, 208) per-col sq diff
    kept = np.ones(Z.shape[1], bool)

    def sf(sumsq):
        r = spearmanr(sumsq, mcs)[0]
        return float('nan') if not np.isfinite(r) else float(r)

    sumsq = SQ.sum(1)
    base_sf = sf(sumsq)
    dropped = []
    if max_drop is None:
        max_drop = Z.shape[1] - 2
    for _ in range(max_drop):
        cur_sf = sf(sumsq)
        best_gain, best_c, best_sf = 0.0, -1, None
        for c in np.where(kept)[0]:
            s = sf(sumsq - SQ[:, c])
            g = s - cur_sf
            if g > best_gain + 1e-9:
                best_gain, best_c, best_sf = g, int(c), s
        if best_c < 0:
            break
        kept[best_c] = False
        sumsq = sumsq - SQ[:, best_c]
        dropped.append({'name': DESC_NAMES[best_c], 'col': best_c,
                        'struct_after': best_sf, 'gain': best_gain})
    blacklist = np.array([d['col'] for d in dropped], dtype=int)
    trace = {'base_struct': base_sf,
             'final_struct': (dropped[-1]['struct_after'] if dropped else base_sf),
             'n_total': int(Z.shape[1]), 'n_dropped': len(dropped),
             'n_kept': int(kept.sum()), 'dropped_names': [d['name'] for d in dropped]}
    return blacklist, trace, kept


# ------------------------------------------------------------------------ scoring
def struct_fidelity(D, I, J, mcs):
    return float(spearmanr(D[I, J], mcs)[0])


def score_space(name, D, y, groups, I, J, mcs):
    rho, negrmse = simeval.eval_knn_readacross(D, y, groups)
    # rho stays NaN when the read-across Spearman is undefined. readacross_defined
    # is the flag that separates that from a measured rho of 0. Every caller that
    # ranks these rows must sort through _nanrank.rank_desc.
    return dict(name=name, readacross_rho=float(rho),
                readacross_defined=is_defined(rho),
                readacross_rmse=float(-negrmse),
                struct_fidelity=struct_fidelity(D, I, J, mcs))


# ------------------------------------------------------------------------ driver
def run(dataset='herg_ki'):
    t0 = time.time()
    p = load_prepared(dataset)
    n = p['n']
    from mcs_cache import load_pairs
    m = load_pairs(dataset, p['smiles'])
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    y, groups = p['y'], p['groups']
    coarse = p['ecfp4_count']
    desc = p['desc']
    X_fine_all = np.hstack([p['atompair_count'].astype(np.float64),
                            p['torsion'].astype(np.float64)])

    # FIX 2: recompute the per-dataset saboteur blacklist
    blk, trace, kept = desc_blacklist_greedy(desc, I, J, mcs)
    physchem_full = desc
    physchem_bl = desc[:, kept]     # survivors (columns that don't hurt structure)

    # fine-2D label-free selection
    fidx = select_fine2d(X_fine_all, I, J, mcs, FINE_K)
    fine = X_fine_all[:, fidx]

    # FIX 1: full 3D shape block (must cover all rows)
    pharm, n_built = load_pharm3d_full(dataset, n)
    shape_ok = pharm is not None

    print(f'[{dataset}] n={n} scaffolds={p["n_scaffolds"]} MCS_pairs={len(I)}  '
          f'desc {trace["n_total"]}->keep {trace["n_kept"]} '
          f'(struct {trace["base_struct"]:+.3f}->{trace["final_struct"]:+.3f})  '
          f'shape={"FULL n_built=%d" % n_built if shape_ok else "UNAVAILABLE(n_built=%d<%d)" % (n_built, n)}',
          flush=True)

    rows = []
    # THE baseline: count alone, Tanimoto
    rows.append(score_space('count_tanimoto[BASELINE]',
                            simeval.dist_tanimoto(coarse), y, groups, I, J, mcs))
    # count alone, z-eucl (isolates coefficient change)
    rows.append(score_space('count_only[z-eucl]',
                            make_D([coarse]), y, groups, I, J, mcs))

    # multi-block combos, each in equal + coarse-down weighting
    combos = [
        ('count+physchem_full',            [coarse, physchem_full]),
        ('count+physchem_blacklisted',     [coarse, physchem_bl]),
        ('count+shape',                    ([coarse, pharm] if shape_ok else None)),
        ('count+physchem_bl+shape+fine2D', ([coarse, physchem_bl, pharm, fine] if shape_ok else None)),
    ]
    for cname, blocks in combos:
        if blocks is None:
            continue
        rows.append(score_space(f'{cname} [equal]',
                                make_D(blocks), y, groups, I, J, mcs))
        w = [0.5] + [1.0] * (len(blocks) - 1)
        rows.append(score_space(f'{cname} [coarse_down x0.5]',
                                make_D(blocks, weights=w), y, groups, I, J, mcs))

    base = rows[0]
    print(f'\n=== {dataset}: creative hybrids v2 (n={n}, FULL scale) ===')
    print(f'  {"space":42s} {"read_rho":>9} {"rmse":>7} {"struct_MCS":>11} '
          f'{"d_ra":>8} {"d_struct":>9}')
    for r in rows:
        d_ra = r['readacross_rho'] - base['readacross_rho']
        d_sf = r['struct_fidelity'] - base['struct_fidelity']
        print(f'  {r["name"]:42s} {r["readacross_rho"]:>9.4f} {r["readacross_rmse"]:>7.3f} '
              f'{r["struct_fidelity"]:>11.4f} {d_ra:>+8.4f} {d_sf:>+9.4f}')
    print(f'  baseline count_tanimoto: read_rho={base["readacross_rho"]:.4f} '
          f'struct_MCS={base["struct_fidelity"]:.4f}')

    # verdict: does anything beat count-alone on ACTIVITY?
    # `>` is already False for a NaN candidate, so no undefined row enters
    # winners; rank_desc keeps the ordering non-finite-safe regardless. When the
    # BASELINE rho is undefined the comparison is undefined for every row, so the
    # count of comparable rows is printed rather than an empty winners list alone.
    winners = [r for r in rows[1:]
               if is_defined(r['readacross_rho']) and is_defined(base['readacross_rho'])
               and r['readacross_rho'] > base['readacross_rho'] + 1e-6]
    winners.sort(key=lambda r: rank_desc(r['readacross_rho']), reverse=True)
    n_cmp = (sum(1 for r in rows[1:] if is_defined(r['readacross_rho']))
             if is_defined(base['readacross_rho']) else 0)
    print(f'  ACTIVITY comparison ran over {n_cmp}/{len(rows)-1} rows '
          f'(baseline rho '
          f'{"defined" if is_defined(base["readacross_rho"]) else "UNDEFINED"})')
    if winners:
        print(f'  ACTIVITY winners over count-alone: ' +
              ', '.join(f'{w["name"]}({w["readacross_rho"]-base["readacross_rho"]:+.4f})'
                        for w in winners))
    else:
        print('  ACTIVITY: NOTHING beats count-alone.')
    print(f'  elapsed {time.time()-t0:.1f}s', flush=True)

    return dict(dataset=dataset, n=int(n), scaffolds=int(p['n_scaffolds']),
                mcs_pairs=int(len(I)), shape_available=bool(shape_ok),
                shape_n_built=int(n_built), fine_k=FINE_K,
                desc_blacklist=trace, baseline=base, rows=rows,
                activity_winners=[w['name'] for w in winners],
                activity_n_comparable=int(n_cmp))


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
    out = args.out or os.path.join(HERE, 'creative_hybrids_v2.json')
    json.dump(result, open(out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(out)}')


if __name__ == '__main__':
    main()
