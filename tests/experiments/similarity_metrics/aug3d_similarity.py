"""3D AUGMENTED representations vs the count-fingerprint baseline (compass 1.3).

Question: do genuinely 3D augmented representations -- built on ETKDG conformers
-- capture activity and/or 2D structure that the cheap count-fingerprint
baseline misses?

Two augmented 3D reps from the shared kirby builders (both embed an ETKDG
conformer, seed=42, via kirby.representations.molecular._generate_conformer):

  e3fp     (1024-bit)  E3FP -- the 3D analogue of ECFP; shell-based hashing of
                       local 3D atom environments on the conformer. BINARY, so a
                       Tanimoto space is meaningful (as well as cosine/euclidean).
  coulomb  (50-D)      Coulomb-matrix eigenvalue spectrum -- rotation-invariant
                       encoding of nuclear charge + 3D geometry. CONTINUOUS, so
                       cosine + (z-scored) euclidean only.

Distance spaces built per rep:
  e3fp     -> tanimoto, cosine, euclidean
  coulomb  -> cosine, euclidean (z-scored)

Each distance space is judged on the SAME two yardsticks used everywhere in this
study:
  * ACTIVITY read-across: scaffold-grouped kNN Spearman rho + -RMSE
                          (simeval.eval_knn_readacross, GroupKFold, NO random splits)
  * STRUCTURAL fidelity : Spearman(D[pair], MCS-distance) on the cached MCS pairs
                          (mcs_pairs_<ds>.npz, restricted to the scored subset)

Competitor to beat (BASELINE):
    ecfp4_count with Tanimoto  ==  simeval.dist_tanimoto(p['ecfp4_count'])

Conformers are the cost, so the built rep MATRICES are cached to
sim_<ds>_3daug.npz keyed by how many leading molecules were embedded (the loader
always takes the first N in _data's fixed order, so a larger cache is a superset
and is simply sliced). Nothing here modifies any shared file.

Usage:
  python aug3d_similarity.py --quick                    # smoke: first 120 herg_ki
  python aug3d_similarity.py                            # full: all of herg_ki
  python aug3d_similarity.py herg_ki logd adme_solubility   # full, several datasets
"""
import os, sys, json, time, argparse, logging
import numpy as np
from scipy.stats import spearmanr

# quieten e3fp's very chatty per-molecule INFO logging
logging.disable(logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

import simeval
from _data import load_prepared
from _nanrank import is_defined
from kirby.representations.molecular import create_e3fp, create_coulomb_matrix

# rep name -> (builder, is_binary_fingerprint)
_BUILDERS = {'e3fp': (create_e3fp, True), 'coulomb': (create_coulomb_matrix, False)}
AUG_REPS = ('e3fp', 'coulomb')


# --------------------------------------------------------------------------
# 3D rep construction + cache (conformers are the expensive part)
# --------------------------------------------------------------------------
def _cache_path(name):
    return os.path.join(HERE, f'sim_{name}_3daug.npz')


def build_aug_reps(name, smiles, need_n):
    """Return {rep: (need_n, d) matrix} for the first `need_n` molecules.

    Cache stores the largest set ever computed for this dataset. Dataset order is
    fixed (_data: herg_ki/adme use idx=arange, logd is a fixed scaffold
    subsample), so the first `need_n` rows of a bigger cache ARE our subset."""
    path = _cache_path(name)
    cached = None
    if os.path.exists(path):
        try:
            cached = np.load(path, allow_pickle=True)
        except Exception:
            cached = None

    have_n = int(cached['n_built']) if cached is not None and 'n_built' in cached else 0
    have_all = cached is not None and all(r in cached for r in AUG_REPS)

    if have_all and have_n >= need_n:
        print(f'[{name}] 3d-aug cache hit (built {have_n} >= need {need_n})', flush=True)
        return {r: np.asarray(cached[r])[:need_n] for r in AUG_REPS}

    build_n = max(need_n, have_n)
    sub = list(smiles[:build_n])
    out = {}
    # carry forward any rep already at full size, only (re)build what is short
    for r in AUG_REPS:
        if cached is not None and r in cached and have_n >= build_n:
            out[r] = np.asarray(cached[r])[:build_n].astype(np.float32)
            print(f'[{name}] reuse {r:8s} {out[r].shape} from cache', flush=True)
            continue
        t0 = time.time()
        out[r] = _BUILDERS[r][0](sub).astype(np.float32)
        z = int((np.abs(out[r]).sum(1) == 0).sum())
        print(f'[{name}] built {r:8s} {out[r].shape} in {time.time()-t0:.0f}s '
              f'({z} all-zero / failed conformers)', flush=True)

    np.savez_compressed(path, n_built=build_n, **out)
    print(f'[{name}] wrote {os.path.basename(path)} (n_built={build_n})', flush=True)
    return {r: out[r][:need_n] for r in AUG_REPS}


# --------------------------------------------------------------------------
# distance spaces
# --------------------------------------------------------------------------
def _zscore(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd


def distance_spaces(X, is_fp):
    Xf = X.astype(np.float64)
    spaces = {
        'cosine': simeval.dist_cosine(Xf),
        'euclidean': simeval.dist_euclidean(_zscore(Xf)),
    }
    if is_fp:
        # binary fingerprint -> Tanimoto is the natural coefficient
        spaces['tanimoto'] = simeval.dist_tanimoto(Xf)
    return spaces


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
    rho, neg_rmse = simeval.eval_knn_readacross(D, y, groups)
    struct, npairs = structural_fidelity(D, pairs)
    # rho stays NaN when the read-across Spearman is undefined; readacross_defined
    # separates that from a measured rho of 0. The d_ra printed against the
    # baseline is then NaN as well, never a zero difference.
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                readacross_neg_rmse=float(neg_rmse),
                struct_fidelity=struct, mcs_pairs=npairs)


def run(name, quick, quick_n):
    p = load_prepared(name)
    n = p['n']
    if quick:
        n_sub = min(quick_n, n)
        idx = np.arange(n_sub)
        y, groups = p['y'][idx], p['groups'][idx]
        base_X = p['ecfp4_count'][idx]
    else:
        n_sub = n
        idx = np.arange(n)
        y, groups = p['y'], p['groups']
        base_X = p['ecfp4_count']
    print(f'[{name}] n={n_sub} (quick={quick})  scaffolds='
          f'{len(set(groups.tolist()))}', flush=True)

    pairs = load_mcs_pairs(name, n_sub)
    if pairs is not None:
        print(f'[{name}] MCS pairs usable within first {n_sub}: {len(pairs[0])}', flush=True)

    baseline = score_metric(simeval.dist_tanimoto(base_X), y, groups, pairs)
    baseline['metric'] = 'ecfp4_count:tanimoto (BASELINE)'

    reps = build_aug_reps(name, p['smiles'], n_sub)

    rows = [baseline]
    for r in AUG_REPS:
        is_fp = _BUILDERS[r][1]
        X = reps[r][idx] if quick else reps[r]
        for cn, D in distance_spaces(X, is_fp).items():
            row = score_metric(D, y, groups, pairs)
            row['metric'] = f'{r}:{cn}'
            rows.append(row)

    return dict(n=n_sub, quick=quick, baseline=baseline, rows=rows)


def _fmt(rows, b):
    print(f'\n  {"metric":24s} {"read_rho":>9} {"read_-rmse":>11} {"struct_MCS":>11}   vs baseline')
    print(f'  {"-"*24} {"-"*9} {"-"*11} {"-"*11}')
    for r in rows:
        dr = r['readacross_rho'] - b['readacross_rho']
        ds = r['struct_fidelity'] - b['struct_fidelity']
        tag = '' if r['metric'].endswith('(BASELINE)') else f'  d_ra={dr:+.3f} d_struct={ds:+.3f}'
        print(f'  {r["metric"]:24s} {r["readacross_rho"]:>9.3f} '
              f'{r["readacross_neg_rmse"]:>11.3f} {r["struct_fidelity"]:>11.3f}{tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--quick-n', type=int, default=120)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    result = {}
    for ds in args.datasets:
        t0 = time.time()
        res = run(ds, args.quick, args.quick_n)
        res['seconds'] = round(time.time() - t0, 1)
        result[ds] = res
        print(f'\n=== {ds}: 3D-augmented reps vs count-FP baseline '
              f'(n={res["n"]}, {res["seconds"]}s) ===')
        _fmt(res['rows'], res['baseline'])

    out = args.out or os.path.join(
        HERE, 'aug3d_similarity_quick.json' if args.quick else 'aug3d_similarity.json')
    json.dump(result, open(out, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(out)}')


if __name__ == '__main__':
    main()
