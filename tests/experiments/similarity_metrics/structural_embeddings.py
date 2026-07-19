"""structural_embeddings -- score three KIRBy molecular representation builders
that the pilot fingerprint grid never used, on the SAME two activity-/structure-
grounded yardsticks as the rest of the similarity study, NEVER on ECFP4 Tanimoto:

  MAP4     (create_map4)     -- MinHashed Atom-Pair fingerprint, folded to a
                               1024-bit binary vector. BIT form -> tanimoto/cosine.
  TDA      (create_tda)      -- 3D persistent-homology (Vietoris-Rips) topological
                               summary, 28 continuous features. CONT -> cosine/euclid.
  mol2vec  (create_mol2vec)  -- pretrained word2vec over Morgan substructures,
                               300-d continuous embedding. CONT -> cosine/euclid.

Yardsticks (both higher = better):
  (1) ACTIVITY READ-ACROSS  -- simeval.eval_knn_readacross: scaffold-grouped
      GroupKFold distance-weighted kNN, Spearman rho of predicted vs measured
      activity. NO random splits.
  (2) STRUCTURAL MCS FIDELITY -- Spearman between the metric's pairwise distance
      and the cached maximum-common-substructure (MCS) distance on the fixed
      pair set (mcs_pairs_<ds>.npz).

Baseline to beat (reported individually, never averaged) =
  ecfp4_count with tanimoto  ==  simeval.dist_tanimoto(p['ecfp4_count']).

Continuous embeddings (TDA, mol2vec) are z-scored per column (StandardScaler)
before cosine/euclidean, so no single large-magnitude column dominates the
distance -- this preprocessing is part of defining the continuous distance space
and is reported explicitly. MAP4 bits are used raw (tanimoto/cosine on bits).

FEASIBILITY NOTES (honest):
  * KIRBy create_map4 as-shipped does `from map4 import MAP4Calculator`, but the
    installable map4 (1.1.3) renamed that class to `map4.map4.MAP4`, so the
    wrapper raises ImportError. MAP4 fingerprints ARE feasible -- built here
    directly via map4.MAP4 (same algorithm, folded 1024-bit output, no tmap).
  * create_tda needs giotto-tda (gtda); create_mol2vec needs the mol2vec pkg +
    the pretrained model_300dim.pkl. Both present locally.

Usage:
  python structural_embeddings.py --quick                 # first ~400 mols
  python structural_embeddings.py                         # herg_ki full
  python structural_embeddings.py --datasets herg_ki logd adme_solubility
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))

from _data import load_prepared          # noqa: E402
import simeval                            # noqa: E402

from rdkit import Chem, RDLogger          # noqa: E402
RDLogger.DisableLog('rdApp.*')

MOL2VEC_MODEL = '/Users/apunt/repos/KIRBy/model_300dim.pkl'

# ---- probe the three backends once, report honestly ---------------------
# MAP4: KIRBy wrapper import (stale) + the real installable package.
_MAP4_WRAPPER_OK, _MAP4_WRAPPER_ERR = True, ''
try:
    from kirby.representations.molecular import create_map4 as _kirby_create_map4
    # the wrapper only fails at call time (import inside), so probe a call:
    _kirby_create_map4(['CCO'])
except Exception as _e:
    _MAP4_WRAPPER_OK = False
    _MAP4_WRAPPER_ERR = f'{type(_e).__name__}: {_e}'

_MAP4_OK, _MAP4_ERR = True, ''
try:
    from map4.map4 import MAP4 as _MAP4CLS
except Exception as _e:
    _MAP4_OK = False
    _MAP4_ERR = f'{type(_e).__name__}: {_e}'

_TDA_OK, _TDA_ERR = True, ''
try:
    from kirby.representations.molecular import create_tda as _create_tda
    import gtda  # noqa: F401
except Exception as _e:
    _TDA_OK = False
    _TDA_ERR = f'{type(_e).__name__}: {_e}'

_MOL2VEC_OK, _MOL2VEC_ERR = True, ''
try:
    from kirby.representations.molecular import create_mol2vec as _create_mol2vec
    if not os.path.exists(MOL2VEC_MODEL):
        raise FileNotFoundError(f'missing pretrained model {MOL2VEC_MODEL}')
except Exception as _e:
    _MOL2VEC_OK = False
    _MOL2VEC_ERR = f'{type(_e).__name__}: {_e}'

MAP4_NBITS = 1024


# =========================================================================
# builders (SMILES -> matrix)
# =========================================================================
def build_map4(smiles, nbits=MAP4_NBITS):
    """MAP4 folded binary fingerprint -> (n, nbits) float32 bit matrix.
    Built via the installable map4.MAP4 (KIRBy wrapper import name is stale)."""
    calc = _MAP4CLS(dimensions=nbits)
    rows, failed = [], 0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append(np.zeros(nbits, dtype=np.float32)); failed += 1; continue
        rows.append(np.asarray(calc.calculate(m), dtype=np.float32))
    return np.vstack(rows), failed


def _zscore(X):
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X.astype(np.float64))
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================================================================
# yardsticks
# =========================================================================
def structural_fidelity(D, I, J, mcs):
    rho = spearmanr(D[I, J], mcs)[0]
    return 0.0 if not np.isfinite(rho) else float(rho)


def score_metric(D, y, groups, I, J, mcs):
    rho = simeval.eval_knn_readacross(D, y, groups)[0]
    return dict(readacross_rho=float(rho),
                struct_fidelity=structural_fidelity(D, I, J, mcs))


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


# =========================================================================
# driver
# =========================================================================
def run(name, quick=False):
    p = load_prepared(name)
    n_full = p['n']
    keep_n = min(400, n_full) if quick else None
    sl = slice(0, keep_n) if keep_n is not None else slice(0, n_full)

    y = np.asarray(p['y'])[sl]
    groups = np.asarray(p['groups'])[sl]
    smiles = list(np.asarray(p['smiles'])[sl])
    ecfp4_count = np.asarray(p['ecfp4_count'])[sl]
    n = len(y)

    mcs_pack = load_mcs_pairs(name, keep_n)
    if mcs_pack is None:
        raise RuntimeError(f'no mcs_pairs_{name}.npz -- cannot score structural fidelity')
    I, J, mcs = mcs_pack

    print(f'[{name}] n={n} (quick={quick}) mcs_pairs={len(I)} '
          f'scaffolds={len(set(groups.tolist()))}', flush=True)

    # ---- baseline: ecfp4_count tanimoto ----
    t0 = time.time()
    base = score_metric(simeval.dist_tanimoto(ecfp4_count), y, groups, I, J, mcs)
    print(f'[{name}] baseline ecfp4_count:tanimoto '
          f'RA={base["readacross_rho"]:.4f} MCS={base["struct_fidelity"]:.4f} '
          f'({time.time()-t0:.1f}s)', flush=True)
    results = {'ecfp4_count:tanimoto (BASELINE)': base}

    # ---- MAP4 (bit): tanimoto, cosine ----
    if _MAP4_OK:
        t0 = time.time()
        Xm, nfail = build_map4(smiles)
        for cn, cf in (('tanimoto', simeval.dist_tanimoto), ('cosine', simeval.dist_cosine)):
            results[f'map4:{cn}'] = score_metric(cf(Xm), y, groups, I, J, mcs)
        print(f'[{name}] map4 done ({time.time()-t0:.1f}s dim={Xm.shape[1]} '
              f'mean_onbits={Xm.sum(1).mean():.1f} failed={nfail})', flush=True)
    else:
        print(f'[{name}] map4 UNAVAILABLE: {_MAP4_ERR}', flush=True)

    # ---- TDA (continuous, z-scored): cosine, euclidean ----
    if _TDA_OK:
        t0 = time.time()
        Xt = _create_tda(smiles)
        nz = int((Xt.sum(1) != 0).sum())
        Xt = _zscore(Xt)
        for cn, cf in (('cosine', simeval.dist_cosine), ('euclidean', simeval.dist_euclidean)):
            results[f'tda:{cn}'] = score_metric(cf(Xt), y, groups, I, J, mcs)
        print(f'[{name}] tda done ({time.time()-t0:.1f}s dim={Xt.shape[1]} '
              f'nonzero_rows={nz}/{n})', flush=True)
    else:
        print(f'[{name}] tda UNAVAILABLE: {_TDA_ERR}', flush=True)

    # ---- mol2vec (continuous, z-scored): cosine, euclidean ----
    if _MOL2VEC_OK:
        t0 = time.time()
        Xv = _create_mol2vec(smiles, model_path=MOL2VEC_MODEL)
        nz = int((Xv.sum(1) != 0).sum())
        Xv = _zscore(Xv)
        for cn, cf in (('cosine', simeval.dist_cosine), ('euclidean', simeval.dist_euclidean)):
            results[f'mol2vec:{cn}'] = score_metric(cf(Xv), y, groups, I, J, mcs)
        print(f'[{name}] mol2vec done ({time.time()-t0:.1f}s dim={Xv.shape[1]} '
              f'nonzero_rows={nz}/{n})', flush=True)
    else:
        print(f'[{name}] mol2vec UNAVAILABLE: {_MOL2VEC_ERR}', flush=True)

    return {'baseline': base, 'results': results,
            'meta': {'n': n, 'quick': quick, 'mcs_pairs': int(len(I))}}


def print_table(name, out):
    base = out['baseline']
    br, bs = base['readacross_rho'], base['struct_fidelity']
    print(f'\n=== {name}: MAP4 / TDA / mol2vec vs count-FP baseline (FULL n={out["meta"]["n"]}) ===')
    print(f'  {"metric":24s} {"readacross":>11} {"dRA":>8}   {"struct":>8} {"dMCS":>8}   flags')
    for metric, sc in out['results'].items():
        is_base = 'BASELINE' in metric
        dra = sc['readacross_rho'] - br
        dst = sc['struct_fidelity'] - bs
        flags = '' if is_base else (('RA+' if dra > 0 else '   ') + ' ' +
                                    ('MCS+' if dst > 0 else '    '))
        print(f'  {metric:24s} {sc["readacross_rho"]:>11.4f} '
              f'{("" if is_base else f"{dra:+.4f}"):>8}   '
              f'{sc["struct_fidelity"]:>8.4f} '
              f'{("" if is_base else f"{dst:+.4f}"):>8}   {flags}')
    wr = [m for m, s in out['results'].items()
          if 'BASELINE' not in m and s['readacross_rho'] > br]
    ws = [m for m, s in out['results'].items()
          if 'BASELINE' not in m and s['struct_fidelity'] > bs]
    print(f'  beats baseline on READ-ACROSS: {wr or "none"}')
    print(f'  beats baseline on STRUCTURAL : {ws or "none"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=['herg_ki'])
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    print('[FEASIBILITY]')
    print(f'  create_map4 KIRBy wrapper: {"OK" if _MAP4_WRAPPER_OK else "FAIL -> " + _MAP4_WRAPPER_ERR}')
    print(f'  map4 (direct build)      : {"OK" if _MAP4_OK else "INFEASIBLE -> " + _MAP4_ERR}')
    print(f'  create_tda               : {"OK" if _TDA_OK else "INFEASIBLE -> " + _TDA_ERR}')
    print(f'  create_mol2vec           : {"OK" if _MOL2VEC_OK else "INFEASIBLE -> " + _MOL2VEC_ERR}')

    allout = {}
    for ds in args.datasets:
        out = run(ds, quick=args.quick)
        print_table(ds, out)
        allout[ds] = out

    suffix = '_quick' if args.quick else ''
    dst = os.path.join(HERE, f'structural_embeddings{suffix}.json')
    json.dump({'map4_wrapper_feasible': _MAP4_WRAPPER_OK,
               'map4_wrapper_error': _MAP4_WRAPPER_ERR,
               'map4_direct_feasible': _MAP4_OK,
               'tda_feasible': _TDA_OK, 'tda_error': _TDA_ERR,
               'mol2vec_feasible': _MOL2VEC_OK, 'mol2vec_error': _MOL2VEC_ERR,
               'datasets': allout}, open(dst, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(dst)}')


if __name__ == '__main__':
    main()
