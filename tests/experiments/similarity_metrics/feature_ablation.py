"""feature_ablation.py -- BLOCK subset search + within-desc drop-one-greedy.

The user's key new question: when you build a similarity space by stitching
together heterogeneous feature BLOCKS (count fingerprint, physicochemical
descriptors, 3D shape, fine 2D substructure, substructure keys), which blocks
HELP and which HURT each of the two yardsticks used across this study --

  * ACTIVITY read-across : scaffold-grouped kNN Spearman rho
                           (simeval.eval_knn_readacross ; NO random splits)
  * STRUCTURAL fidelity  : Spearman(D[pair], MCS) on the cached MCS pairs

and -- inspired by the antibody finding that a couple of descriptors (charge,
isoelectric point) were actively SABOTAGING structural fidelity -- are there
individual RDKit descriptors inside the physchem block that consistently HURT?

METHOD
  Five blocks:
    count   = ecfp4_count            (1024 count fp)
    physchem= desc                   (208 RDKit descriptors)
    shape   = pharm3d                (49 3D pharmacophore ; sim_<ds>_spatial.npz)
    fine2D  = top-100 columns of [atompair_count | torsion] ranked by
              |Spearman(|feature diff|, MCS)| on the MCS pairs
    keys    = maccs                  (167 substructure keys)
  Each subset of blocks -> z-score the kept columns, concatenate, EUCLIDEAN
  distance (simeval.dist_euclidean). Score both yardsticks. Exhaustive 2^K.
  Baseline throughout = ecfp4_count:tanimoto (simeval.dist_tanimoto).

  Then, within the desc block only, drop-one-greedy on STRUCTURAL fidelity:
  repeatedly remove the single descriptor whose removal most raises structural
  fidelity, until nothing helps -> the "saboteur" list. (Rank of a euclidean
  distance is unchanged by the final sqrt, so structural fidelity is computed
  from the per-column squared-difference sums over the MCS pairs -- exact and
  fast.)

DATA REALITY (reported honestly, not silently worked around)
  Only sim_herg_ki_spatial.npz exists, and it holds just the first 250
  molecules. There is NO logd spatial cache. The prompt says LOAD, do not
  rebuild, so the shape block cannot enter a full-n search on either dataset.
  Therefore:
    - FULL-SCALE search (headline) = 4 blocks {count,physchem,fine2D,keys} at
      full n on BOTH datasets (herg_ki n=1415, logd n=1427).
    - SHAPE supplement = the full 5-block 2^5 on herg_ki's first 250 molecules
      only, to isolate whether shape helps. logd shape: unavailable.

Standalone; imports the shared read-only tools, modifies nothing.
"""
import os, sys, json, time, itertools, argparse
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
from _data import load_prepared

# desc column names, in the exact order build_moleculeace.py wrote them
from rdkit.Chem import Descriptors
DESC_NAMES = [n for n, _ in Descriptors._descList]   # 208, matches desc matrix


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def zscore(X):
    X = X.astype(np.float64)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def struct_fidelity(D, I, J, mcs):
    """Spearman between a metric's pair distances and MCS-DISTANCE (the cached
    `mcs` = 1 - MCS-Tanimoto, so it is a DISTANCE: larger = less shared
    substructure). D is also a distance, so a GOOD metric gives a POSITIVE rho.
    Raw spearman, exactly the task's definition; HIGHER = better (matches the
    repo's structural_yardstick convention)."""
    rho = spearmanr(D[I, J], mcs)[0]
    return float('nan') if not np.isfinite(rho) else float(rho)


def score_subset(Zblocks, kept, y, groups, I, J, mcs):
    """kept = tuple of block names present. z-scored block matrices already in
    Zblocks. Concatenate, euclidean, score both yardsticks."""
    X = np.concatenate([Zblocks[b] for b in kept], axis=1)
    D = simeval.dist_euclidean(X)
    ra, _ = simeval.eval_knn_readacross(D, y, groups)
    sf = struct_fidelity(D, I, J, mcs)
    return float(ra), sf, X.shape[1]


def select_fine2D(atompair_count, torsion, I, J, mcs, top=100):
    """top-`top` columns of [atompair_count | torsion] by |Spearman(|feat diff|,
    MCS)| on the MCS pairs. Returns the concatenated selected-column matrix and
    the chosen (source, col) index list."""
    F = np.concatenate([atompair_count.astype(np.float64),
                        torsion.astype(np.float64)], axis=1)   # (n, 2048)
    diff = np.abs(F[I] - F[J])                                  # (n_pairs, 2048)
    rmcs = rankdata(mcs)
    scores = np.full(F.shape[1], 0.0)
    # rank once; correlate each column's |diff| ranks against mcs ranks
    for c in range(F.shape[1]):
        col = diff[:, c]
        if col.std() == 0:
            continue
        r = spearmanr(col, mcs)[0]
        scores[c] = 0.0 if not np.isfinite(r) else abs(r)
    order = np.argsort(scores)[::-1][:top]
    return F[:, order], order, scores[order]


# --------------------------------------------------------------------------
# block subset search
# --------------------------------------------------------------------------
def build_blocks(p, idx, I, J, mcs, include_shape, shape_mat=None):
    """Return {block: z-scored matrix} aligned to idx rows."""
    blk_raw = {
        'count':    p['ecfp4_count'][idx],
        'physchem': p['desc'][idx],
        'keys':     p['maccs'][idx],
    }
    fine, fine_cols, _ = select_fine2D(p['atompair_count'][idx], p['torsion'][idx],
                                       I, J, mcs)
    blk_raw['fine2D'] = fine
    if include_shape:
        blk_raw['shape'] = shape_mat
    return {b: zscore(m) for b, m in blk_raw.items()}, fine_cols


def exhaustive_search(blocks, y, groups, I, J, mcs, base):
    names = list(blocks.keys())
    rows = []
    for k in range(1, len(names) + 1):
        for kept in itertools.combinations(names, k):
            ra, sf, dim = score_subset(blocks, kept, y, groups, I, J, mcs)
            rows.append({'blocks': list(kept), 'k': k, 'dim': dim,
                         'readacross_rho': ra, 'struct_fidelity': sf,
                         'd_readacross': ra - base['readacross_rho'],
                         'd_struct': sf - base['struct_fidelity']})
    rows.sort(key=lambda r: r['readacross_rho'], reverse=True)
    return rows


def block_marginals(rows, names):
    """For each block, mean yardstick over subsets WITH it minus subsets
    WITHOUT it -> does the block, on average, help or hurt each yardstick?"""
    out = {}
    for b in names:
        withb = [r for r in rows if b in r['blocks']]
        wob = [r for r in rows if b not in r['blocks']]
        if not withb or not wob:
            continue
        out[b] = {
            'ra_with': float(np.mean([r['readacross_rho'] for r in withb])),
            'ra_without': float(np.mean([r['readacross_rho'] for r in wob])),
            'ra_delta': float(np.mean([r['readacross_rho'] for r in withb])
                              - np.mean([r['readacross_rho'] for r in wob])),
            'sf_with': float(np.mean([r['struct_fidelity'] for r in withb])),
            'sf_without': float(np.mean([r['struct_fidelity'] for r in wob])),
            'sf_delta': float(np.mean([r['struct_fidelity'] for r in withb])
                              - np.mean([r['struct_fidelity'] for r in wob])),
        }
    return out


# --------------------------------------------------------------------------
# within-desc drop-one-greedy on STRUCTURAL fidelity
# --------------------------------------------------------------------------
def desc_drop_one_greedy(desc, I, J, mcs, max_drop=None):
    """Greedy: start from all 208 z-scored descriptors; repeatedly drop the one
    whose removal most raises structural fidelity (Spearman of the euclidean
    distance vs MCS). Because spearman is invariant to the final sqrt, work with
    per-pair squared-difference SUMS (exact). Returns the drop trace + baseline."""
    Z = zscore(desc)                          # (n, 208)
    SQ = (Z[I] - Z[J]) ** 2                    # (n_pairs, 208) per-column sq diff
    kept = np.ones(Z.shape[1], bool)

    def sf_from_sumsq(sumsq):
        # distance rank == sumsq rank (sqrt is monotone); raw rho, higher=better
        r = spearmanr(sumsq, mcs)[0]
        return float('nan') if not np.isfinite(r) else float(r)

    sumsq = SQ.sum(1)
    base_sf = sf_from_sumsq(sumsq)
    trace = []
    if max_drop is None:
        max_drop = Z.shape[1] - 2
    for _ in range(max_drop):
        cur = sumsq  # sumsq over currently-kept columns
        cur_sf = sf_from_sumsq(cur)
        best_gain, best_c, best_sf = 0.0, -1, None
        cols = np.where(kept)[0]
        for c in cols:
            cand = cur - SQ[:, c]              # drop column c
            sf = sf_from_sumsq(cand)
            gain = sf - cur_sf
            if gain > best_gain + 1e-9:
                best_gain, best_c, best_sf = gain, int(c), sf
        if best_c < 0:
            break
        kept[best_c] = False
        sumsq = cur - SQ[:, best_c]
        trace.append({'dropped': DESC_NAMES[best_c], 'col': best_c,
                      'struct_after': best_sf, 'gain': best_gain})
    return {'base_struct': base_sf,
            'final_struct': (trace[-1]['struct_after'] if trace else base_sf),
            'n_dropped': len(trace), 'dropped': trace}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_full(ds):
    p = load_prepared(ds)
    n = p['n']
    idx = np.arange(n)
    m = np.load(os.path.join(HERE, f'mcs_pairs_{ds}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)

    base_D = simeval.dist_tanimoto(p['ecfp4_count'])
    b_ra, _ = simeval.eval_knn_readacross(base_D, p['y'], p['groups'])
    base = {'readacross_rho': float(b_ra),
            'struct_fidelity': struct_fidelity(base_D, I, J, mcs)}

    blocks, fine_cols = build_blocks(p, idx, I, J, mcs, include_shape=False)
    rows = exhaustive_search(blocks, p['y'], p['groups'], I, J, mcs, base)
    marg = block_marginals(rows, list(blocks.keys()))
    greedy = desc_drop_one_greedy(p['desc'], I, J, mcs)
    return {'ds': ds, 'n': int(n), 'scaffolds': int(p['n_scaffolds']),
            'baseline': base, 'rows': rows, 'block_marginals': marg,
            'desc_greedy': greedy}


def run_shape_supp(ds='herg_ki', n_sub=250):
    """5-block 2^5 on the first n_sub molecules (only place shape data exists)."""
    sp_path = os.path.join(HERE, f'sim_{ds}_spatial.npz')
    if not os.path.exists(sp_path):
        return None
    sp = np.load(sp_path, allow_pickle=True)
    have = int(sp['n_built'])
    n_sub = min(n_sub, have)
    shape_mat = np.asarray(sp['pharm3d'])[:n_sub]

    p = load_prepared(ds)
    idx = np.arange(n_sub)
    y, groups = p['y'][idx], p['groups'][idx]
    m = np.load(os.path.join(HERE, f'mcs_pairs_{ds}.npz'))
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)
    keep = (I < n_sub) & (J < n_sub)
    I, J, mcs = I[keep], J[keep], mcs[keep]

    base_D = simeval.dist_tanimoto(p['ecfp4_count'][idx])
    b_ra, _ = simeval.eval_knn_readacross(base_D, y, groups)
    base = {'readacross_rho': float(b_ra),
            'struct_fidelity': struct_fidelity(base_D, I, J, mcs)}

    blocks, _ = build_blocks(p, idx, I, J, mcs, include_shape=True,
                             shape_mat=shape_mat)
    rows = exhaustive_search(blocks, y, groups, I, J, mcs, base)
    marg = block_marginals(rows, list(blocks.keys()))
    return {'ds': ds, 'n': int(n_sub), 'n_mcs_pairs': int(len(I)),
            'baseline': base, 'rows': rows, 'block_marginals': marg}


def _fmt_rows(rows, base, top=12):
    print(f'  baseline ecfp4_count:tanimoto  ra={base["readacross_rho"]:+.3f}  '
          f'struct={base["struct_fidelity"]:+.3f}')
    print(f'  {"blocks":40s} {"dim":>5} {"read_rho":>9} {"struct":>8}  '
          f'{"d_ra":>7} {"d_struct":>8}')
    for r in rows[:top]:
        print(f'  {"+".join(r["blocks"]):40s} {r["dim"]:>5} '
              f'{r["readacross_rho"]:>9.3f} {r["struct_fidelity"]:>8.3f}  '
              f'{r["d_readacross"]:>+7.3f} {r["d_struct"]:>+8.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--out', default=os.path.join(HERE, 'feature_ablation.json'))
    args = ap.parse_args()

    result = {'full': {}, 'shape_supp': {}}
    for ds in args.datasets:
        t0 = time.time()
        r = run_full(ds)
        r['seconds'] = round(time.time() - t0, 1)
        result['full'][ds] = r
        print(f'\n=== {ds}: FULL-scale 4-block search (n={r["n"]}, '
              f'{r["seconds"]}s) ===')
        _fmt_rows(r['rows'], r['baseline'])
        print('  block marginals (mean over subsets with - without):')
        for b, mm in r['block_marginals'].items():
            print(f'    {b:9s} ra_delta={mm["ra_delta"]:+.3f}  '
                  f'sf_delta={mm["sf_delta"]:+.3f}')
        g = r['desc_greedy']
        print(f'  desc drop-one-greedy: struct {g["base_struct"]:+.3f} -> '
              f'{g["final_struct"]:+.3f}  ({g["n_dropped"]} dropped)')
        for d in g['dropped'][:15]:
            print(f'      drop {d["dropped"]:22s} struct={d["struct_after"]:+.3f} '
                  f'(+{d["gain"]:.4f})')

    supp = run_shape_supp('herg_ki', 250)
    if supp is not None:
        result['shape_supp']['herg_ki'] = supp
        print(f'\n=== herg_ki: SHAPE supplement 5-block 2^5 '
              f'(n={supp["n"]}, mcs_pairs={supp["n_mcs_pairs"]}) ===')
        _fmt_rows(supp['rows'], supp['baseline'])
        print('  block marginals:')
        for b, mm in supp['block_marginals'].items():
            print(f'    {b:9s} ra_delta={mm["ra_delta"]:+.3f}  '
                  f'sf_delta={mm["sf_delta"]:+.3f}')

    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(args.out)}')


if __name__ == '__main__':
    main()
