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

MOLECULES HELD OUT. Two steps here are FITTED to the MCS values: the fine-2D
block is the 100 columns whose |feature difference| best tracks MCS distance,
and the drop sequence keeps whichever drop raises the correlation with MCS most.
Grading either on the same pairs reports the maximum of a search -- 143 drops
deep, in the greedy's case -- not a measured fidelity. So the MOLECULES are cut
in half by whole Bemis-Murcko scaffold group (_heldout.scaffold_halves, the same
cut similarity_noise_cleaning.py uses), and a cached pair counts only when BOTH
its endpoints are on one side:

  heldout   the reported arm. The fine-2D ranking, the per-column z-score
            mean/sd and the drop sequence are fitted on the fit half; every
            number is measured on the other half.
  insample  the old fit-equals-score arm, kept so the size of the optimism is
            visible instead of assumed.

The greedy also carries final_struct_fit, the correlation the sequence reached
on the pairs that steered it, beside the held-out final_struct.

DATA REALITY (read off the caches, not assumed)
  Three spatial caches exist and each one holds every molecule of its dataset:
  sim_herg_ki_spatial.npz n_built=1415, sim_logd_spatial.npz n_built=1427,
  sim_adme_solubility_spatial.npz n_built=2173, each with a (n, 49) pharm3d
  matrix. The prompt says LOAD, do not rebuild, and nothing needs rebuilding.
  Therefore:
    - FULL-SCALE search (headline) = 4 blocks {count,physchem,fine2D,keys} at
      full n on BOTH datasets (herg_ki n=1415, logd n=1427).
    - SHAPE supplement = the full 5-block 2^5 at full n on every dataset whose
      spatial cache exists, to isolate whether shape helps. --shape-cap caps the
      molecule count for a smoke run and is not the default.

Standalone; imports the shared read-only tools, modifies nothing.
"""
import os, sys, json, time, itertools, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
from _data import load_prepared
from _nanrank import rank_desc, nanmean, n_defined, is_defined
from _heldout import SPLIT_SEED, scaffold_halves, pairs_within, zscore_fit

MIN_HELDOUT_PAIRS = 300   # below this a held-out structural fidelity is too thin
                          # to report as a bare number; say so instead. Same
                          # threshold as similarity_noise_cleaning.py.

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


def score_subset(Zblocks, kept, y, groups, I, J, mcs, score_idx=None):
    """kept = tuple of block names present. z-scored block matrices already in
    Zblocks, over ALL rows. Concatenate, euclidean, score both yardsticks on the
    rows in score_idx (all of them when it is None), with I/J/mcs already
    re-indexed into positions within score_idx."""
    X = np.concatenate([Zblocks[b] for b in kept], axis=1)
    D = simeval.dist_euclidean(X)
    if score_idx is not None:
        D = D[np.ix_(score_idx, score_idx)]
    ra, _ = simeval.eval_knn_readacross(D, y, groups)
    sf = struct_fidelity(D, I, J, mcs)
    # ra stays NaN when the read-across Spearman is undefined; the caller stores
    # ra_defined beside it so an undefined rho reads differently from rho = 0.
    return float(ra), sf, X.shape[1]


def select_fine2D(atompair_count, torsion, I, J, mcs, top=100, fit_rows=None):
    """top-`top` columns of [atompair_count | torsion] by |Spearman(|feat diff|,
    MCS)| on the MCS pairs. Returns the concatenated selected-column matrix (over
    ALL rows) and the chosen column index list.

    fit_rows, when given, are row positions the columns are RANKED on; I/J index
    into fit_rows and the ranking never sees a molecule outside it. Those same
    columns are then cut out of every row, so the caller can score them on the
    molecules the ranking did not see. fit_rows=None keeps the old behaviour, in
    which the ranking sees every molecule the score is later taken over."""
    F = np.concatenate([atompair_count.astype(np.float64),
                        torsion.astype(np.float64)], axis=1)   # (n, 2048)
    Ffit = F if fit_rows is None else F[fit_rows]
    diff = np.abs(Ffit[I] - Ffit[J])                            # (n_pairs, 2048)
    scores = np.full(F.shape[1], 0.0)
    # correlate each column's |diff| against the MCS distance over those pairs
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
def build_blocks(p, idx, I, J, mcs, include_shape, shape_mat=None,
                 fit_rows=None):
    """Return {block: z-scored matrix} aligned to idx rows.

    fit_rows are positions within idx that the FITTED steps -- the fine-2D column
    ranking and the per-column z-score mean/sd -- are allowed to see; I/J/mcs are
    the pairs among those rows. Every block is still returned over all of idx, so
    the caller cuts the distance matrix to whichever rows it means to score.
    fit_rows=None fits on every row, which is the in-sample arm."""
    blk_raw = {
        'count':    p['ecfp4_count'][idx],
        'physchem': p['desc'][idx],
        'keys':     p['maccs'][idx],
    }
    fine, fine_cols, _ = select_fine2D(p['atompair_count'][idx], p['torsion'][idx],
                                       I, J, mcs, fit_rows=fit_rows)
    blk_raw['fine2D'] = fine
    if include_shape:
        blk_raw['shape'] = shape_mat
    if fit_rows is None:
        Z = {b: zscore(m) for b, m in blk_raw.items()}
    else:
        Z = {b: zscore_fit(m, fit_rows) for b, m in blk_raw.items()}
    return Z, fine_cols


def exhaustive_search(blocks, y, groups, I, J, mcs, base, score_idx=None):
    names = list(blocks.keys())
    rows = []
    for k in range(1, len(names) + 1):
        for kept in itertools.combinations(names, k):
            ra, sf, dim = score_subset(blocks, kept, y, groups, I, J, mcs,
                                       score_idx=score_idx)
            rows.append({'blocks': list(kept), 'k': k, 'dim': dim,
                         'readacross_rho': ra, 'struct_fidelity': sf,
                         'readacross_defined': is_defined(ra),
                         'struct_fidelity_defined': is_defined(sf),
                         'd_readacross': ra - base['readacross_rho'],
                         'd_struct': sf - base['struct_fidelity']})
    # non-finite readacross_rho sorts last instead of holding its insertion slot.
    rows.sort(key=lambda r: rank_desc(r['readacross_rho']), reverse=True)
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
        # Averaged over the subsets whose yardstick is FINITE, via nanmean. Under
        # np.mean a single undefined subset made every marginal NaN. The
        # *_n_defined counts record how many subsets each mean was taken over,
        # out of *_n_total; a mean over fewer subsets is not the same quantity.
        ra_w = [r['readacross_rho'] for r in withb]
        ra_o = [r['readacross_rho'] for r in wob]
        sf_w = [r['struct_fidelity'] for r in withb]
        sf_o = [r['struct_fidelity'] for r in wob]
        out[b] = {
            'ra_with': nanmean(ra_w),
            'ra_without': nanmean(ra_o),
            'ra_delta': nanmean(ra_w) - nanmean(ra_o),
            'ra_with_n_defined': n_defined(ra_w), 'ra_with_n_total': len(ra_w),
            'ra_without_n_defined': n_defined(ra_o), 'ra_without_n_total': len(ra_o),
            'sf_with': nanmean(sf_w),
            'sf_without': nanmean(sf_o),
            'sf_delta': nanmean(sf_w) - nanmean(sf_o),
            'sf_with_n_defined': n_defined(sf_w), 'sf_with_n_total': len(sf_w),
            'sf_without_n_defined': n_defined(sf_o), 'sf_without_n_total': len(sf_o),
        }
    return out


# --------------------------------------------------------------------------
# within-desc drop-one-greedy on STRUCTURAL fidelity
# --------------------------------------------------------------------------
def desc_drop_one_greedy(desc, I, J, mcs, max_drop=None,
                         fit_rows=None, score_rows=None,
                         score_pairs=None):
    """Greedy: start from all 208 z-scored descriptors; repeatedly drop the one
    whose removal most raises structural fidelity (Spearman of the euclidean
    distance vs MCS). Because spearman is invariant to the final sqrt, work with
    per-pair squared-difference SUMS (exact).

    The drop sequence is chosen on the FIT rows' pairs (I, J, mcs indexed into
    fit_rows). After every drop the same descriptor set is re-scored on
    score_pairs = (I, J, mcs) among score_rows, molecules the sequence has never
    seen and whose z-score mean/sd it did not use. `final_struct` is that
    held-out number; `final_struct_fit` is the search maximum the sequence was
    steered by, kept beside it so the gap between the two is visible.

    fit_rows=None reproduces the old behaviour exactly: the sequence is chosen
    and reported on the same pairs, and final_struct is a search maximum."""
    held = fit_rows is not None and score_rows is not None and score_pairs is not None
    if held:
        Z = zscore_fit(desc, fit_rows)
        Zf, Zs = Z[fit_rows], Z[score_rows]
        Isc, Jsc, Msc = score_pairs
        SQs = (Zs[Isc] - Zs[Jsc]) ** 2
    else:
        Zf = zscore(desc)
        SQs = None
    SQ = (Zf[I] - Zf[J]) ** 2                  # (n_fit_pairs, 208) per-col sq diff
    kept = np.ones(Zf.shape[1], bool)

    def sf_from_sumsq(sumsq, target):
        # distance rank == sumsq rank (sqrt is monotone); raw rho, higher=better
        r = spearmanr(sumsq, target)[0]
        return float('nan') if not np.isfinite(r) else float(r)

    sumsq = SQ.sum(1)
    base_sf_fit = sf_from_sumsq(sumsq, mcs)
    if held:
        sumsq_s = SQs.sum(1)
        base_sf = sf_from_sumsq(sumsq_s, Msc)
    else:
        sumsq_s, base_sf = None, base_sf_fit
    trace = []
    if max_drop is None:
        max_drop = Zf.shape[1] - 2
    for _ in range(max_drop):
        cur = sumsq  # sumsq over currently-kept columns, on the FIT pairs
        cur_sf = sf_from_sumsq(cur, mcs)
        best_gain, best_c, best_sf = 0.0, -1, None
        cols = np.where(kept)[0]
        for c in cols:
            cand = cur - SQ[:, c]              # drop column c
            sf = sf_from_sumsq(cand, mcs)
            gain = sf - cur_sf
            if gain > best_gain + 1e-9:
                best_gain, best_c, best_sf = gain, int(c), sf
        if best_c < 0:
            break
        kept[best_c] = False
        sumsq = cur - SQ[:, best_c]
        step = {'dropped': DESC_NAMES[best_c], 'col': best_c,
                'struct_after_fit': best_sf, 'gain_fit': best_gain}
        if held:
            sumsq_s = sumsq_s - SQs[:, best_c]
            step['struct_after'] = sf_from_sumsq(sumsq_s, Msc)
        else:
            step['struct_after'] = best_sf
        trace.append(step)
    return {'held_out': bool(held),
            'base_struct': base_sf,
            'base_struct_fit': base_sf_fit,
            'final_struct': (trace[-1]['struct_after'] if trace else base_sf),
            'final_struct_fit': (trace[-1]['struct_after_fit'] if trace
                                 else base_sf_fit),
            'n_dropped': len(trace), 'dropped': trace}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _arm(p, idx, I, J, mcs, y, groups, include_shape, shape_mat,
         fit_rows, score_rows, base_X):
    """One arm of the search. fit_rows / score_rows are positions within idx.
    Both None = the in-sample arm: everything is fitted on, and scored on, every
    molecule. Otherwise the fine-2D ranking, the z-score mean/sd and the greedy
    drop sequence see fit_rows only, and every reported number is taken on
    score_rows over the pairs with both endpoints there."""
    n_idx = len(idx)
    if fit_rows is None:
        Ift, Jft, Mft = I, J, mcs
        Isc, Jsc, Msc = I, J, mcs
        score_cut = None
        y_sc, g_sc = y, groups
    else:
        Ift, Jft, Mft = pairs_within(I, J, mcs, fit_rows, n_idx)
        Isc, Jsc, Msc = pairs_within(I, J, mcs, score_rows, n_idx)
        score_cut = score_rows
        y_sc, g_sc = y[score_rows], groups[score_rows]

    base_D = simeval.dist_tanimoto(base_X)
    if score_cut is not None:
        base_D = base_D[np.ix_(score_cut, score_cut)]
    b_ra, _ = simeval.eval_knn_readacross(base_D, y_sc, g_sc)
    base = {'readacross_rho': float(b_ra),
            'readacross_defined': is_defined(b_ra),
            'struct_fidelity': struct_fidelity(base_D, Isc, Jsc, Msc)}

    blocks, fine_cols = build_blocks(p, idx, Ift, Jft, Mft,
                                     include_shape=include_shape,
                                     shape_mat=shape_mat, fit_rows=fit_rows)
    rows = exhaustive_search(blocks, y_sc, g_sc, Isc, Jsc, Msc, base,
                             score_idx=score_cut)
    marg = block_marginals(rows, list(blocks.keys()))
    return {'baseline': base, 'rows': rows, 'block_marginals': marg,
            'n_fit_pairs': int(len(Ift)), 'n_score_pairs': int(len(Isc)),
            'score_pairs_thin': bool(len(Isc) < MIN_HELDOUT_PAIRS),
            'min_score_pairs': MIN_HELDOUT_PAIRS,
            'n_fit_molecules': int(n_idx if fit_rows is None else len(fit_rows)),
            'n_score_molecules': int(n_idx if score_rows is None else len(score_rows)),
            'fine2d_cols': [int(c) for c in fine_cols]}


def run_full(ds):
    """Two arms on the same molecules.

    heldout   the reported one. Molecules are cut in half by whole Bemis-Murcko
              scaffold group; the fine-2D column ranking, the z-score mean/sd and
              the descriptor drop sequence are fitted on one half, and every
              number is measured on the other half over the cached MCS pairs whose
              BOTH endpoints sit there.
    insample  the old fit-equals-score arm, kept so the size of the optimism is
              visible rather than assumed. Its numbers are search maxima, and
              `struct_fidelity` in it is graded on the same pairs the fine-2D
              block and the drop sequence were picked to match."""
    p = load_prepared(ds)
    n = p['n']
    idx = np.arange(n)
    from mcs_cache import load_pairs
    m = load_pairs(ds, p['smiles'])
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)
    y, groups = p['y'], p['groups']

    fit_rows, score_rows = scaffold_halves(groups)

    insample = _arm(p, idx, I, J, mcs, y, groups, False, None,
                    None, None, p['ecfp4_count'])
    insample['desc_greedy'] = desc_drop_one_greedy(p['desc'], I, J, mcs)

    Ift, Jft, Mft = pairs_within(I, J, mcs, fit_rows, n)
    heldout = _arm(p, idx, I, J, mcs, y, groups, False, None,
                   fit_rows, score_rows, p['ecfp4_count'])
    heldout['desc_greedy'] = desc_drop_one_greedy(
        p['desc'], Ift, Jft, Mft, fit_rows=fit_rows, score_rows=score_rows,
        score_pairs=pairs_within(I, J, mcs, score_rows, n))

    return {'ds': ds, 'n': int(n), 'scaffolds': int(p['n_scaffolds']),
            'split': {'seed': SPLIT_SEED, 'by': 'bemis_murcko_scaffold_group',
                      'n_fit_molecules': int(len(fit_rows)),
                      'n_score_molecules': int(len(score_rows)),
                      'n_pairs_total': int(len(I)),
                      'n_pairs_fit': int(heldout['n_fit_pairs']),
                      'n_pairs_score': int(heldout['n_score_pairs']),
                      'n_pairs_straddling': int(len(I) - heldout['n_fit_pairs']
                                                - heldout['n_score_pairs'])},
            'heldout': heldout, 'insample': insample}


def run_shape_supp(ds='herg_ki', n_sub=None):
    """5-block 2^5 on every molecule the spatial cache holds. Same two arms as
    run_full. n_sub caps the molecule count for a smoke run; None takes every row
    the cache and the prepared arrays share, which for all three cached datasets
    is the whole dataset."""
    sp_path = os.path.join(HERE, f'sim_{ds}_spatial.npz')
    if not os.path.exists(sp_path):
        return None
    sp = np.load(sp_path, allow_pickle=True)
    have = int(sp['n_built'])

    p = load_prepared(ds)
    n_sub = min(have, int(p['n'])) if n_sub is None else min(n_sub, have, int(p['n']))
    shape_mat = np.asarray(sp['pharm3d'])[:n_sub]
    idx = np.arange(n_sub)
    y, groups = p['y'][idx], p['groups'][idx]
    from mcs_cache import load_pairs
    m = load_pairs(ds, p['smiles'])
    I, J, mcs = m['I'], m['J'], m['mcs'].astype(np.float64)
    keep = (I < n_sub) & (J < n_sub)
    I, J, mcs = I[keep], J[keep], mcs[keep]

    fit_rows, score_rows = scaffold_halves(groups)
    base_X = p['ecfp4_count'][idx]

    insample = _arm(p, idx, I, J, mcs, y, groups, True, shape_mat,
                    None, None, base_X)
    heldout = _arm(p, idx, I, J, mcs, y, groups, True, shape_mat,
                   fit_rows, score_rows, base_X)
    return {'ds': ds, 'n': int(n_sub), 'n_in_spatial_cache': have,
            'n_in_dataset': int(p['n']), 'n_mcs_pairs': int(len(I)),
            'split': {'seed': SPLIT_SEED, 'by': 'bemis_murcko_scaffold_group',
                      'n_fit_molecules': int(len(fit_rows)),
                      'n_score_molecules': int(len(score_rows)),
                      'n_pairs_fit': int(heldout['n_fit_pairs']),
                      'n_pairs_score': int(heldout['n_score_pairs'])},
            'heldout': heldout, 'insample': insample}


def _fmt_rows(rows, base, top=12):
    print(f'  baseline ecfp4_count:tanimoto  ra={base["readacross_rho"]:+.3f}  '
          f'struct={base["struct_fidelity"]:+.3f}')
    print(f'  {"blocks":40s} {"dim":>5} {"read_rho":>9} {"struct":>8}  '
          f'{"d_ra":>7} {"d_struct":>8}')
    for r in rows[:top]:
        print(f'  {"+".join(r["blocks"]):40s} {r["dim"]:>5} '
              f'{r["readacross_rho"]:>9.3f} {r["struct_fidelity"]:>8.3f}  '
              f'{r["d_readacross"]:>+7.3f} {r["d_struct"]:>+8.3f}')


def _fmt_arm(tag, arm, show_greedy=True):
    print(f'  --- {tag} arm  '
          f'(fit on {arm["n_fit_molecules"]} molecules / {arm["n_fit_pairs"]} pairs, '
          f'scored on {arm["n_score_molecules"]} molecules / {arm["n_score_pairs"]} pairs) ---')
    if arm.get('score_pairs_thin'):
        print(f'  every struct number below rests on {arm["n_score_pairs"]} pairs, under '
              f'the {arm["min_score_pairs"]} this study treats as enough to quote; read the '
              f'ORDER of the subsets, not the values')
    _fmt_rows(arm['rows'], arm['baseline'])
    print('  block marginals (mean over subsets with - without):')
    for b, mm in arm['block_marginals'].items():
        print(f'    {b:9s} ra_delta={mm["ra_delta"]:+.3f}  '
              f'sf_delta={mm["sf_delta"]:+.3f}  '
              f'(ra over {mm["ra_with_n_defined"]}/{mm["ra_with_n_total"]} with, '
              f'{mm["ra_without_n_defined"]}/{mm["ra_without_n_total"]} without)')
    g = arm.get('desc_greedy')
    if show_greedy and g is not None:
        print(f'  desc drop-one-greedy: struct {g["base_struct"]:+.3f} -> '
              f'{g["final_struct"]:+.3f}  ({g["n_dropped"]} dropped)'
              + ('' if not g['held_out'] else
                 f'   [the sequence reached {g["final_struct_fit"]:+.3f} on the '
                 f'pairs it was chosen on]'))
        for d in g['dropped'][:15]:
            print(f'      drop {d["dropped"]:22s} struct={d["struct_after"]:+.3f} '
                  f'(fit +{d["gain_fit"]:.4f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--out', default=os.path.join(HERE, 'feature_ablation.json'))
    ap.add_argument('--shape-datasets', nargs='+',
                    default=['herg_ki', 'logd', 'adme_solubility'],
                    help='datasets to run the 5-block shape supplement on; one '
                         'with no sim_<ds>_spatial.npz is reported as missing')
    ap.add_argument('--shape-cap', type=int, default=None,
                    help='most molecules the shape supplement reads; the default '
                         'takes every molecule the spatial cache holds')
    args = ap.parse_args()

    result = {'full': {}, 'shape_supp': {}}
    for ds in args.datasets:
        t0 = time.time()
        r = run_full(ds)
        r['seconds'] = round(time.time() - t0, 1)
        result['full'][ds] = r
        sp = r['split']
        print(f'\n=== {ds}: FULL-scale 4-block search (n={r["n"]}, '
              f'{r["seconds"]}s) ===')
        print(f'  molecules split by scaffold group (seed {sp["seed"]}): '
              f'{sp["n_fit_molecules"]} fit / {sp["n_score_molecules"]} score; '
              f'of {sp["n_pairs_total"]} cached pairs {sp["n_pairs_fit"]} lie '
              f'inside the fit half, {sp["n_pairs_score"]} inside the score half, '
              f'{sp["n_pairs_straddling"]} straddle and are dropped')
        _fmt_arm('HELD-OUT (reported)', r['heldout'])
        _fmt_arm('in-sample (search maximum, for comparison)', r['insample'])

    for ds in args.shape_datasets:
        supp = run_shape_supp(ds, args.shape_cap)
        if supp is None:
            print(f'\n=== {ds}: SHAPE supplement skipped, no '
                  f'sim_{ds}_spatial.npz ===')
            continue
        result['shape_supp'][ds] = supp
        print(f'\n=== {ds}: SHAPE supplement 5-block 2^5 '
              f'(n={supp["n"]} of {supp["n_in_dataset"]} molecules, '
              f'{supp["n_in_spatial_cache"]} in the spatial cache, '
              f'mcs_pairs={supp["n_mcs_pairs"]}) ===')
        _fmt_arm('HELD-OUT (reported)', supp['heldout'], show_greedy=False)
        _fmt_arm('in-sample (search maximum, for comparison)', supp['insample'],
                 show_greedy=False)

    json.dump(result, open(args.out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(args.out)}')


if __name__ == '__main__':
    main()
