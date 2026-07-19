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
  * STRUCTURAL fidelity : Spearman(D[pair], MCS-distance) on the cached MCS pairs

The competitor to beat is the count-fingerprint baseline:
    ecfp4_count with Tanimoto  ==  simeval.dist_tanimoto(p['ecfp4_count'])

Conformers are the cost, so the 3D reps are cached to sim_<ds>_spatial.npz keyed
by how many leading molecules were embedded (the loader always takes the first
N in _data's fixed order, so a larger cache is a superset and is simply sliced).

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


def build_spatial_reps(name, smiles, need_n):
    """Return {rep: (need_n, d) matrix} for the first `need_n` molecules.

    Cache stores the largest set ever computed for this dataset. Because the
    dataset order is fixed (_data uses idx=arange for herg_ki), the first
    `need_n` rows of a bigger cache ARE the reps for our subset -> slice."""
    path = _cache_path(name)
    cached = None
    if os.path.exists(path):
        try:
            cached = np.load(path, allow_pickle=True)
        except Exception:
            cached = None

    have_n = int(cached['n_built']) if cached is not None and 'n_built' in cached else 0
    have_all = cached is not None and all(r in cached for r in SPATIAL_REPS)

    if have_all and have_n >= need_n:
        print(f'[{name}] spatial cache hit (built {have_n} >= need {need_n})', flush=True)
        return {r: np.asarray(cached[r])[:need_n] for r in SPATIAL_REPS}

    # (Re)build for max(need_n, whatever we already had) so the cache only grows.
    build_n = max(need_n, have_n)
    sub = list(smiles[:build_n])
    out = {}
    for r in SPATIAL_REPS:
        t0 = time.time()
        out[r] = _BUILDERS[r](sub).astype(np.float32)
        z = int((np.abs(out[r]).sum(1) == 0).sum())
        print(f'[{name}] built {r:8s} {out[r].shape} in {time.time()-t0:.0f}s '
              f'({z} all-zero / failed conformers)', flush=True)

    np.savez_compressed(path, n_built=build_n, **out)
    print(f'[{name}] wrote {os.path.basename(_cache_path(name))} (n_built={build_n})', flush=True)
    return {r: out[r][:need_n] for r in SPATIAL_REPS}


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
def load_mcs_pairs(name, n_sub):
    path = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    I, J, mcs = d['I'], d['J'], d['mcs']
    keep = (I < n_sub) & (J < n_sub)
    return I[keep], J[keep], mcs[keep]


def structural_fidelity(D, pairs):
    if pairs is None:
        return float('nan'), 0
    I, J, mcs = pairs
    if len(I) < 10:
        return float('nan'), len(I)
    rho = spearmanr(D[I, J], mcs)[0]
    return (float('nan') if not np.isfinite(rho) else float(rho)), len(I)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def score_metric(D, y, groups, pairs):
    rho, _ = simeval.eval_knn_readacross(D, y, groups)
    struct, npairs = structural_fidelity(D, pairs)
    return dict(readacross_rho=float(rho), struct_fidelity=struct, mcs_pairs=npairs)


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
    print(f'[{name}] n={n_sub} (quick={quick})  scaffolds='
          f'{len(set(groups.tolist()))}', flush=True)

    pairs = load_mcs_pairs(name, n_sub)
    if pairs is not None:
        print(f'[{name}] MCS pairs usable within first {n_sub}: {len(pairs[0])}', flush=True)

    # baseline: count fingerprint with Tanimoto
    base_D = simeval.dist_tanimoto(p['ecfp4_count'][idx] if quick else p['ecfp4_count'])
    baseline = score_metric(base_D, y, groups, pairs)
    baseline['metric'] = 'ecfp4_count:tanimoto (BASELINE)'

    reps = build_spatial_reps(name, p['smiles'], n_sub)

    rows = [baseline]
    for r in SPATIAL_REPS:
        for cn, D in distance_spaces(reps[r]).items():
            row = score_metric(D, y, groups, pairs)
            row['metric'] = f'{r}:{cn}'
            rows.append(row)

    return dict(n=n_sub, quick=quick, baseline=baseline, rows=rows)


def _fmt(rows, b):
    print(f'\n  {"metric":26s} {"readacross_rho":>14} {"struct_MCS":>11}   vs baseline')
    print(f'  {"-"*26} {"-"*14} {"-"*11}')
    for r in rows:
        dr = r['readacross_rho'] - b['readacross_rho']
        ds = r['struct_fidelity'] - b['struct_fidelity']
        tag = '' if r['metric'].endswith('(BASELINE)') else f'  d_ra={dr:+.3f} d_struct={ds:+.3f}'
        print(f'  {r["metric"]:26s} {r["readacross_rho"]:>14.3f} '
              f'{r["struct_fidelity"]:>11.3f}{tag}')


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
