"""3D shape / pharmacophore similarity (compass brief 1.3).

Question: does a 3D shape / pharmacophore view of a molecule capture activity
and/or 2D structure that the coarse count-fingerprint baseline misses?

We build three 3D descriptors on ETKDG conformers (seed=42, via the shared
kirby.representations.molecular functions):
  USR           (12-D) ultrafast shape recognition -- pure shape moments
  USRCAT        (60-D) USR + pharmacophoric atom-type channels
  pharmacophore (49-D) feature counts + pair-distance mean/std

Each descriptor is turned into cosine and (z-scored) euclidean distance spaces
and judged on the SAME two yardsticks used everywhere in this study:
  * ACTIVITY read-across: scaffold-grouped kNN Spearman rho (simeval.eval_knn_readacross)
  * STRUCTURAL fidelity : Spearman(D[pair], MCS-distance) on the cached MCS pairs,
    reported twice -- plain, and holding the pair's summed heavy-atom count
    constant, since size alone moves both a 3D distance and an MCS distance

The competitor to beat is the count-fingerprint baseline:
    ecfp4_count with Tanimoto  ==  simeval.dist_tanimoto(p['ecfp4_count'])

Conformers are the cost, so the 3D reps are cached to sim_<ds>_spatial.npz keyed
by how many leading molecules were embedded (the loader always takes the first
N in _data's fixed order, so a larger cache is a superset and is simply sliced)
AND by a hash of that exact molecule list, so a cache built from other molecules
is rebuilt instead of silently sliced. A file written before the hash existed
cannot be checked, so it is rebuilt too.

Molecules whose conformer failed come back as an all-zero row and stay in the
matrix; the count per representation is in the result payload.

Usage:
  python shape_similarity.py --quick            # smoke: first 250 herg_ki mols
  python shape_similarity.py                     # full: all of herg_ki
  python shape_similarity.py logd herg_ki        # full on several datasets

NOT for import into other scripts; standalone. Does not modify any shared file.
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
from _data import load_prepared
from _nanrank import is_defined
from mcs_cache import smiles_id
from scaffold_experiment import partial_spearman
from rdkit import Chem
from kirby.representations.molecular import (
    create_usr, create_usrcat, create_pharmacophore3d)

SPATIAL_REPS = ('usr', 'usrcat', 'pharm3d')
_BUILDERS = {'usr': create_usr, 'usrcat': create_usrcat,
             'pharm3d': create_pharmacophore3d}


# --------------------------------------------------------------------------
# 3D rep construction + cache (conformers are the expensive part)
# --------------------------------------------------------------------------
def _cache_path(name):
    return os.path.join(HERE, f'sim_{name}_spatial.npz')


def _zero_rows(M):
    """Molecules whose descriptor came back all zero -- a failed conformer.
    They stay in the matrix, so the count belongs in the payload."""
    return int((np.abs(np.asarray(M)).sum(1) == 0).sum())


def build_spatial_reps(name, smiles, need_n):
    """Return ({rep: (need_n, d) matrix}, {rep: all-zero row count}) for the
    first `need_n` molecules.

    Cache stores the largest set ever computed for this dataset. Because the
    dataset order is fixed (_data uses idx=arange for herg_ki), the first
    `need_n` rows of a bigger cache ARE the reps for our subset -> slice. The
    count of molecules whose conformer failed is returned, not just printed."""
    path = _cache_path(name)
    cached = None
    if os.path.exists(path):
        try:
            cached = np.load(path, allow_pickle=True)
        except Exception:
            cached = None

    have_n = int(cached['n_built']) if cached is not None and 'n_built' in cached else 0
    have_all = cached is not None and all(r in cached for r in SPATIAL_REPS)

    # n_built says how MANY molecules the cached rows were built from, never
    # WHICH. Rebuild sim_<ds>.npz with one molecule added, removed or reordered
    # and every cached row is a different molecule's conformer, with nothing to
    # notice it. smiles_id is a hash of the exact list the rows came from. A
    # file written before that field existed is rebuilt rather than trusted:
    # "cannot be checked" and "matches" are not the same thing.
    stored_id = (str(cached['smiles_id'])
                 if cached is not None and 'smiles_id' in cached else None)
    want_id = (smiles_id(list(smiles[:have_n]))
               if have_n and len(smiles) >= have_n else None)
    id_ok = stored_id is not None and want_id is not None and stored_id == want_id

    if have_all and have_n >= need_n and id_ok:
        print(f'[{name}] spatial cache hit (built {have_n} >= need {need_n}, '
              f'molecule list {stored_id})', flush=True)
        reps = {r: np.asarray(cached[r])[:need_n] for r in SPATIAL_REPS}
        return reps, {r: _zero_rows(reps[r]) for r in SPATIAL_REPS}
    if have_all and have_n >= need_n:
        why = ('records no molecule list' if stored_id is None
               else f'was built from molecule list {stored_id}, holding {want_id}')
        print(f'[{name}] spatial cache has {have_n} rows but {why} -- rebuilding',
              flush=True)

    # (Re)build for max(need_n, whatever we already had) so the cache only grows.
    build_n = max(need_n, have_n)
    sub = list(smiles[:build_n])
    out, zeros = {}, {}
    for r in SPATIAL_REPS:
        t0 = time.time()
        out[r] = _BUILDERS[r](sub).astype(np.float32)
        z = _zero_rows(out[r])
        print(f'[{name}] built {r:8s} {out[r].shape} in {time.time()-t0:.0f}s '
              f'({z} all-zero / failed conformers)', flush=True)

    np.savez_compressed(path, n_built=build_n,
                        smiles_id=np.array(smiles_id(sub)), **out)
    print(f'[{name}] wrote {os.path.basename(_cache_path(name))} '
          f'(n_built={build_n}, molecule list {smiles_id(sub)})', flush=True)
    reps = {r: out[r][:need_n] for r in SPATIAL_REPS}
    return reps, {r: _zero_rows(reps[r]) for r in SPATIAL_REPS}


# --------------------------------------------------------------------------
# distance spaces
# --------------------------------------------------------------------------
def _zscore(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd


def distance_spaces(X):
    """cosine (raw) and euclidean (z-scored, so no single moment dominates)."""
    return {
        'cosine': simeval.dist_cosine(X.astype(np.float64)),
        'euclidean': simeval.dist_euclidean(_zscore(X.astype(np.float64))),
    }


# --------------------------------------------------------------------------
# structural (MCS) fidelity on the cached pairs, restricted to our subset
# --------------------------------------------------------------------------
def heavy_atom_counts(smiles):
    """Heavy atoms per molecule -- the quantity MCS divides by (shared atoms
    over atoms_A + atoms_B - shared). 0 for a SMILES RDKit cannot parse."""
    out = np.zeros(len(smiles), dtype=np.float64)
    for k, sm in enumerate(smiles):
        m = Chem.MolFromSmiles(str(sm))
        out[k] = 0.0 if m is None else float(m.GetNumHeavyAtoms())
    return out


def load_mcs_pairs(name, n_sub, sizes):
    """(I, J, MCS distance, summed heavy-atom count) for the stored pairs whose
    both molecules sit inside the first n_sub."""
    path = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    I, J, mcs = d['I'], d['J'], d['mcs']
    keep = (I < n_sub) & (J < n_sub)
    I, J, mcs = I[keep], J[keep], mcs[keep]
    return I, J, mcs, sizes[I] + sizes[J]


def structural_fidelity(D, pairs):
    """Spearman of the space's pairwise distance against MCS distance, and the
    same with the pair's molecule size held constant. Big molecules share
    proportionally fewer atoms AND sit far apart in a z-scored 3D space, so the
    plain number is part size; the controlled one is what survives removing it."""
    if pairs is None:
        return float('nan'), float('nan'), 0
    I, J, mcs, pair_size = pairs
    if len(I) < 10:
        return float('nan'), float('nan'), len(I)
    rho = spearmanr(D[I, J], mcs)[0]
    part = partial_spearman(D[I, J], mcs, pair_size)
    return ((float('nan') if not np.isfinite(rho) else float(rho)),
            (float('nan') if not np.isfinite(part) else float(part)), len(I))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def score_metric(D, y, groups, pairs):
    rho, _ = simeval.eval_knn_readacross(D, y, groups)
    # rho stays NaN when the read-across Spearman is undefined; readacross_defined
    # separates that from a measured rho of 0. The vs-baseline delta printed
    # below is then NaN as well, never a zero difference.
    struct, struct_sz, npairs = structural_fidelity(D, pairs)
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                struct_fidelity=struct,
                struct_fidelity_size_controlled=struct_sz,
                mcs_pairs=npairs)


def run(name, quick, quick_n):
    p = load_prepared(name)
    n = p['n']
    if quick:
        n_sub = min(quick_n, n)
        idx = np.arange(n_sub)   # first n_sub in fixed order (aligns with MCS pairs)
        y, groups, smiles = p['y'][idx], p['groups'][idx], p['smiles'][idx]
    else:
        n_sub = n
        y, groups, smiles = p['y'], p['groups'], p['smiles']
    sizes = heavy_atom_counts(smiles)
    print(f'[{name}] n={n_sub} (quick={quick})  scaffolds='
          f'{len(set(groups.tolist()))}  heavy atoms per molecule '
          f'{sizes.min():.0f}-{sizes.max():.0f} (mean {sizes.mean():.1f})', flush=True)

    pairs = load_mcs_pairs(name, n_sub, sizes)
    if pairs is not None:
        print(f'[{name}] MCS pairs usable within first {n_sub}: {len(pairs[0])}', flush=True)

    # baseline: count fingerprint with Tanimoto
    base_D = simeval.dist_tanimoto(p['ecfp4_count'][idx] if quick else p['ecfp4_count'])
    baseline = score_metric(base_D, y, groups, pairs)
    baseline['metric'] = 'ecfp4_count:tanimoto (BASELINE)'

    reps, failed_rows = build_spatial_reps(name, p['smiles'], n_sub)
    print(f'[{name}] all-zero (failed-conformer) rows: '
          + ', '.join(f'{r} {failed_rows[r]}/{n_sub}' for r in SPATIAL_REPS),
          flush=True)

    rows = [baseline]
    for r in SPATIAL_REPS:
        for cn, D in distance_spaces(reps[r]).items():
            row = score_metric(D, y, groups, pairs)
            row['metric'] = f'{r}:{cn}'
            row['failed_rows'] = failed_rows[r]
            rows.append(row)

    # failed_rows: molecules whose conformer failed, so their descriptor is an
    # all-zero vector sitting in every distance this row was scored on.
    return dict(n=n_sub, quick=quick, baseline=baseline, rows=rows,
                failed_rows=failed_rows)


def _fmt(rows, b):
    print(f'\n  {"metric":26s} {"readacross_rho":>14} {"struct_MCS":>11} '
          f'{"struct|size":>12}   vs baseline')
    print(f'  {"-"*26} {"-"*14} {"-"*11} {"-"*12}')
    for r in rows:
        dr = r['readacross_rho'] - b['readacross_rho']
        ds = r['struct_fidelity'] - b['struct_fidelity']
        # struct|size holds the pair's summed heavy-atom count constant
        dz = r['struct_fidelity_size_controlled'] - b['struct_fidelity_size_controlled']
        tag = ('' if r['metric'].endswith('(BASELINE)')
               else f'  d_ra={dr:+.3f} d_struct={ds:+.3f} d_struct|size={dz:+.3f}')
        print(f'  {r["metric"]:26s} {r["readacross_rho"]:>14.3f} '
              f'{r["struct_fidelity"]:>11.3f} '
              f'{r["struct_fidelity_size_controlled"]:>12.3f}{tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki'])
    ap.add_argument('--quick', action='store_true', help='smoke: first N mols, small')
    ap.add_argument('--quick-n', type=int, default=250)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    result = {}
    for ds in args.datasets:
        t0 = time.time()
        res = run(ds, args.quick, args.quick_n)
        res['seconds'] = round(time.time() - t0, 1)
        result[ds] = res
        b = res['baseline']
        print(f'\n=== {ds}: 3D shape/pharmacophore vs count-FP baseline '
              f'(n={res["n"]}, {res["seconds"]}s) ===')
        _fmt(res['rows'], b)

    out = args.out or os.path.join(
        HERE, 'shape_similarity_quick.json' if args.quick else 'shape_similarity.json')
    json.dump(result, open(out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(out)}')


if __name__ == '__main__':
    main()
