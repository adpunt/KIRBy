"""augmented_descriptor_reps -- score AUGMENTED molecular descriptor reps as
SIMILARITY spaces (compass brief 1.x, "descriptor-vector" family), the reps the
fingerprint grid (extra_fingerprints.py) deliberately left out.

Three augmented descriptor builders from src/kirby/representations/molecular.py:

  create_pdv     -- RDKit Physical Descriptor Vector (~200 physicochemical
                    descriptors: MolWt, LogP, TPSA, ring counts, ...). Continuous.
  create_mordred -- Mordred 2D descriptor block (~1600 descriptors; a superset of
                    the PDV). Continuous. Needs the `mordred` package.
  create_padel   -- PaDEL 1D/2D descriptors (~1444). Needs the `padelpy` package
                    AND a Java runtime. Probed once; reported INFEASIBLE (never
                    faked) if the package is missing.

Each descriptor block is a CONTINUOUS vector, so the fingerprint coefficients
(Tanimoto / Dice / Soergel) do not apply. The natural continuous-embedding
metrics are cosine and euclidean -- but only after a per-feature z-score, because
raw descriptors span wildly different scales (MolWt ~ 100s, fraction descriptors
~ 0-1) and euclidean/cosine would otherwise be dominated by the largest-magnitude
column. z-score = zero mean, unit variance per feature; constant columns -> 0.

Judged on the SAME two yardsticks as the rest of the study, NEVER on ECFP4
Tanimoto:

  (1) ACTIVITY READ-ACROSS  -- simeval.eval_knn_readacross: scaffold-grouped
      GroupKFold distance-weighted kNN, Spearman rho of predicted vs measured
      activity. Higher = better. (NO random splits.)
  (2) STRUCTURAL MCS FIDELITY -- Spearman between the metric's pairwise distance
      and the cached MCS (maximum-common-substructure) distance over the pair set
      in mcs_pairs_<ds>.npz. Higher = distances better mirror shared substructure.

Baseline to beat (reported individually, never averaged) = ecfp4_count with
tanimoto, i.e. simeval.dist_tanimoto(p['ecfp4_count']).

Built rep matrices are cached to sim_<ds>_augmented.npz (raw builder output, one
array per rep) so re-runs skip the expensive Mordred/PaDEL recompute. The z-score
is a scoring-time transform, applied fresh each run.

Usage:
  python augmented_descriptor_reps.py --quick   # smoke: first ~300 mols
  python augmented_descriptor_reps.py           # full herg_ki + logd + adme_solubility
  python augmented_descriptor_reps.py --datasets herg_ki
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

from _data import load_prepared          # noqa: E402
import simeval                            # noqa: E402
from _nanrank import is_defined            # noqa: E402

from rdkit import RDLogger                # noqa: E402
RDLogger.DisableLog('rdApp.*')

from kirby.representations.molecular import create_pdv, create_mordred, create_padel  # noqa: E402

# ---- optional / possibly-missing descriptor backends --------------------
_MORDRED_OK = True
try:
    import mordred as _mordred_pkg        # noqa: F401
except Exception as _e:                   # pragma: no cover
    _MORDRED_OK = False
    _MORDRED_ERR = str(_e)

# PaDEL needs the external padelpy wrapper AND a Java runtime; neither ships with
# rdkit. Probe the python package once so we can report it honestly.
_PADEL_OK = True
try:
    import padelpy as _padel_pkg          # noqa: F401
except Exception as _e:
    _PADEL_OK = False
    _PADEL_ERR = str(_e)

# rep name -> (builder, feasible flag, err-if-infeasible)
BUILDERS = {
    'pdv':     (create_pdv,     True,        ''),
    'mordred': (create_mordred, _MORDRED_OK, '' if _MORDRED_OK else _MORDRED_ERR),
    'padel':   (create_padel,   _PADEL_OK,   '' if _PADEL_OK else _PADEL_ERR),
}
# continuous descriptor blocks -> cosine + euclidean on the z-scored matrix
CONT_COEFFS = (('cosine', simeval.dist_cosine), ('euclidean', simeval.dist_euclidean))


# =========================================================================
# helpers
# =========================================================================
def zscore(X):
    """Per-feature z-score; constant columns -> 0 (unit-variance guard)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def structural_fidelity(D, I, J, mcs):
    """Spearman(D over the MCS pair set, MCS distance). Higher = better.
    NaN when the correlation is undefined (a constant pair-distance vector, or
    fewer than two pairs) -- never 0.0, which would read as a measured zero."""
    rho = spearmanr(D[I, J], mcs)[0]
    return float(rho) if np.isfinite(rho) else float('nan')


def score_metric(D, y, groups, I, J, mcs):
    rho = simeval.eval_knn_readacross(D, y, groups)[0]
    struct = structural_fidelity(D, I, J, mcs)
    # Neither correlation is folded to 0.0 when it is undefined; the two
    # _defined flags separate an undefined correlation from a measured 0.
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                struct_fidelity=struct,
                struct_fidelity_defined=is_defined(struct))


def load_mcs_pairs(name, keep_n):
    path = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    I, J, mcs = d['I'].astype(int), d['J'].astype(int), d['mcs'].astype(float)
    if keep_n is not None:
        m = (I < keep_n) & (J < keep_n)
        I, J, mcs = I[m], J[m], mcs[m]
    return I, J, mcs


def cache_path(name):
    return os.path.join(HERE, f'sim_{name}_augmented.npz')


def load_cached_reps(name):
    p = cache_path(name)
    if not os.path.exists(p):
        return {}
    d = np.load(p, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def save_cached_reps(name, reps):
    if reps:
        np.savez_compressed(cache_path(name), **reps)


# =========================================================================
# driver
# =========================================================================
def run(name, quick=False):
    p = load_prepared(name)
    n_full = p['n']
    keep_n = min(300, n_full) if quick else None
    sl = slice(0, keep_n) if keep_n is not None else slice(0, n_full)

    y = np.asarray(p['y'])[sl]
    groups = np.asarray(p['groups'])[sl]
    smiles = [str(s) for s in np.asarray(p['smiles'])[sl]]
    ecfp4_count = np.asarray(p['ecfp4_count'])[sl]
    n = len(y)

    mcs_pack = load_mcs_pairs(name, keep_n)
    if mcs_pack is None:
        raise RuntimeError(f'no mcs_pairs_{name}.npz -- cannot score structural fidelity')
    I, J, mcs = mcs_pack

    print(f'[{name}] n={n} (quick={quick})  mcs_pairs={len(I)}  '
          f'scaffolds={len(set(groups.tolist()))}', flush=True)

    # ---- baseline: ecfp4_count tanimoto ----
    t0 = time.time()
    D_base = simeval.dist_tanimoto(ecfp4_count)
    base = score_metric(D_base, y, groups, I, J, mcs)
    print(f'[{name}] baseline ecfp4_count:tanimoto  '
          f'readacross_rho={base["readacross_rho"]:.4f}  '
          f'struct={base["struct_fidelity"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)

    results = {'ecfp4_count:tanimoto (BASELINE)': base}

    # cache of raw builder matrices; only rebuild if the FULL matrix is absent.
    # (quick mode never writes cache -- it would be a truncated prefix.)
    cached = load_cached_reps(name) if not quick else {}
    to_persist = dict(cached)
    feas = {}

    for rep, (builder, ok, err) in BUILDERS.items():
        if not ok:
            feas[rep] = {'feasible': False, 'error': err}
            print(f'[{name}] {rep} INFEASIBLE: {err}', flush=True)
            continue
        # reuse cached full matrix if present and long enough; else build.
        Xraw = None
        if rep in cached and cached[rep].shape[0] >= n:
            Xraw = np.asarray(cached[rep])[:n]
            print(f'[{name}] {rep} loaded from cache dim={Xraw.shape}', flush=True)
        else:
            t0 = time.time()
            try:
                Xraw = np.asarray(builder(smiles), dtype=np.float32)
            except Exception as e:
                feas[rep] = {'feasible': False, 'error': f'build failed: {e}'}
                print(f'[{name}] {rep} BUILD FAILED: {e}', flush=True)
                continue
            print(f'[{name}] {rep} built dim={Xraw.shape} ({time.time()-t0:.1f}s)', flush=True)
            if not quick:
                to_persist[rep] = Xraw  # cache the full-scale matrix

        feas[rep] = {'feasible': True, 'dim': int(Xraw.shape[1])}
        Xz = zscore(Xraw)
        for cn, cf in CONT_COEFFS:
            results[f'{rep}:{cn}'] = score_metric(cf(Xz), y, groups, I, J, mcs)
            r = results[f'{rep}:{cn}']
            print(f'[{name}]   {rep}:{cn:10s} RA={r["readacross_rho"]:+.4f} '
                  f'MCS={r["struct_fidelity"]:+.4f}', flush=True)

    if not quick and to_persist:
        save_cached_reps(name, to_persist)
        print(f'[{name}] cached reps -> {os.path.basename(cache_path(name))} '
              f'({list(to_persist.keys())})', flush=True)

    return {'baseline': base, 'results': results, 'feasibility': feas,
            'meta': {'n': n, 'quick': quick, 'mcs_pairs': int(len(I))}}


def print_table(name, out):
    base = out['baseline']
    br, bs = base['readacross_rho'], base['struct_fidelity']
    print(f'\n=== {name}: augmented descriptor reps vs count-FP baseline (full-scale) ===')
    print(f'  {"metric":26s} {"readacross":>11} {"dRA":>8}   {"struct":>8} {"dStruct":>8}   flags')
    for metric, sc in out['results'].items():
        is_base = 'BASELINE' in metric
        dra = sc['readacross_rho'] - br
        dst = sc['struct_fidelity'] - bs
        flags = ''
        if not is_base:
            flags = ('RA+' if dra > 0 else '   ') + ' ' + ('MCS+' if dst > 0 else '    ')
        print(f'  {metric:26s} {sc["readacross_rho"]:>11.4f} '
              f'{("" if is_base else f"{dra:+.4f}"):>8}   '
              f'{sc["struct_fidelity"]:>8.4f} '
              f'{("" if is_base else f"{dst:+.4f}"):>8}   {flags}')
    # `>` is False for a NaN candidate, so no undefined rho becomes a winner.
    # A NaN baseline br makes every comparison False; n_cmp records how many
    # metrics the read-across comparison could actually be made on.
    winners_ra = [m for m, s in out['results'].items()
                  if 'BASELINE' not in m and is_defined(br)
                  and is_defined(s['readacross_rho']) and s['readacross_rho'] > br]
    n_cmp = (sum(1 for m, s in out['results'].items()
                 if 'BASELINE' not in m and is_defined(s['readacross_rho']))
             if is_defined(br) else 0)
    winners_st = [m for m, s in out['results'].items()
                  if 'BASELINE' not in m and is_defined(bs)
                  and is_defined(s['struct_fidelity']) and s['struct_fidelity'] > bs]
    n_cmp_st = (sum(1 for m, s in out['results'].items()
                    if 'BASELINE' not in m and is_defined(s['struct_fidelity']))
                if is_defined(bs) else 0)
    print(f'  beats baseline on READ-ACROSS: {winners_ra or "none"}  '
          f'(compared on {n_cmp} metrics; baseline rho '
          f'{"defined" if is_defined(br) else "UNDEFINED"})')
    print(f'  beats baseline on STRUCTURAL : {winners_st or "none"}  '
          f'(compared on {n_cmp_st} metrics; baseline structural fidelity '
          f'{"defined" if is_defined(bs) else "UNDEFINED"})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*',
                    default=['herg_ki', 'logd', 'adme_solubility'])
    ap.add_argument('--quick', action='store_true', help='smoke: first ~300 mols')
    args = ap.parse_args()

    if not _PADEL_OK:
        print(f'[ISSUES] PaDEL INFEASIBLE locally: {_PADEL_ERR} '
              f'(needs the padelpy package + a Java runtime; padelpy not installed).',
              flush=True)
    if not _MORDRED_OK:
        print(f'[ISSUES] Mordred INFEASIBLE locally: {_MORDRED_ERR}', flush=True)

    allout = {}
    for ds in args.datasets:
        out = run(ds, quick=args.quick)
        print_table(ds, out)
        allout[ds] = out

    suffix = '_quick' if args.quick else ''
    dst = os.path.join(HERE, f'augmented_descriptor_reps{suffix}.json')
    json.dump({'mordred_feasible': _MORDRED_OK,
               'padel_feasible': _PADEL_OK,
               'pdv_feasible': True,
               'datasets': allout}, open(dst, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(dst)}')


if __name__ == '__main__':
    main()
