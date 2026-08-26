"""extra_fingerprints -- fill out the fingerprint grid (compass brief 1.1-1.2)
with fingerprints the pilot's 10-rep set OMITTED.

The pilot surveyed ecfp4/ecfp6/fcfp4/(+count)/atompair/(+count)/torsion/rdkit/
maccs/desc. Three well-known families were never tried:

  Avalon   -- rdkit.Avalon.pyAvalonTools.GetAvalonFP (a substructure-key style
              bit fingerprint from the Avalon toolkit; different hashing scheme
              than Morgan, so a genuinely new column in the grid).
  MHFP6    -- rdkit.Chem.rdMHFPFingerprint.MHFPEncoder, radius 3 (== MHFP6). A
              MinHash fingerprint whose NATIVE distance is the MinHash-estimated
              Jaccard distance (mean mismatched hash slots). We also fold it to a
              binary SECFP6 fingerprint so it can be scored with the same
              tanimoto/cosine coefficients as every other bit fingerprint.
  MAP4     -- would need the external `map4`+`tmap` packages. NOT installed here,
              so it is reported INFEASIBLE (see ISSUES), not silently skipped.

Each fingerprint is turned into distance spaces (tanimoto / cosine for the bit
forms; native MinHash-Jaccard for MHFP6) and judged on the SAME two yardsticks
the rest of the study uses, NEVER on ECFP4 Tanimoto:

  (1) ACTIVITY READ-ACROSS  -- simeval.eval_knn_readacross: scaffold-grouped
      GroupKFold distance-weighted kNN, Spearman rho of predicted vs measured
      activity. Higher = better. (NO random splits.)
  (2) STRUCTURAL MCS FIDELITY -- Spearman between the metric's pairwise distance
      and the precomputed MCS (maximum-common-substructure) distance on the
      cached herg_ki pair set. Higher = distances better mirror shared
      substructure.

Baseline to beat (reported individually, never averaged) = ecfp4_count with
tanimoto, i.e. simeval.dist_tanimoto(p['ecfp4_count']).

Usage:
  python extra_fingerprints.py --quick   # smoke: first ~400 herg_ki mols, 1-3 min
  python extra_fingerprints.py           # full herg_ki (n=1415)
  python extra_fingerprints.py --datasets herg_ki logd
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

from rdkit import Chem, RDLogger          # noqa: E402
RDLogger.DisableLog('rdApp.*')

# ---- optional / possibly-missing fingerprint backends -------------------
_AVALON_OK = True
try:
    from rdkit.Avalon import pyAvalonTools
except Exception as _e:                   # pragma: no cover
    _AVALON_OK = False
    _AVALON_ERR = str(_e)

_MHFP_OK = True
try:
    from rdkit.Chem import rdMHFPFingerprint
except Exception as _e:                   # pragma: no cover
    _MHFP_OK = False
    _MHFP_ERR = str(_e)

# MAP4 needs the external `map4` package (which itself needs `tmap`); neither is
# a part of rdkit. Probe once so ISSUES can report it honestly.
_MAP4_OK = True
try:
    import map4 as _map4_pkg              # noqa: F401
except Exception as _e:
    _MAP4_OK = False
    _MAP4_ERR = str(_e)

AVALON_NBITS = 2048
MHFP_NBITS = 2048
MHFP_RADIUS = 3          # radius 3 == MHFP6
MHFP_SEED = 42


# =========================================================================
# fingerprint builders  (SMILES -> matrix)
# =========================================================================
def _bitvect_to_row(bv, nbits):
    arr = np.zeros(nbits, dtype=np.uint8)
    for b in bv.GetOnBits():
        arr[b] = 1
    return arr


def build_avalon(mols, nbits=AVALON_NBITS):
    """Avalon bit fingerprint -> (n, nbits) uint8 bit matrix."""
    rows = [_bitvect_to_row(pyAvalonTools.GetAvalonFP(m, nBits=nbits), nbits)
            for m in mols]
    return np.vstack(rows)


def build_secfp6(mols, enc, nbits=MHFP_NBITS, radius=MHFP_RADIUS):
    """MHFP6 folded to a binary SECFP fingerprint -> (n, nbits) uint8 bit
    matrix. This is the bit form of MHFP6 so tanimoto/cosine apply."""
    rows = [_bitvect_to_row(enc.EncodeSECFPMol(m, radius=radius, length=nbits), nbits)
            for m in mols]
    return np.vstack(rows)


def build_mhfp6_minhash(mols, enc, nbits=MHFP_NBITS, radius=MHFP_RADIUS):
    """MHFP6 MinHash vector -> (n, nbits) int64 hash matrix. The NATIVE MHFP
    distance is the fraction of mismatched hash slots (MinHash Jaccard
    estimate), so this matrix feeds mhfp_minhash_distance, not tanimoto."""
    rows = [np.asarray(enc.EncodeMol(m, radius=radius), dtype=np.int64) for m in mols]
    return np.vstack(rows)


def mhfp_minhash_distance(H, chunk=64):
    """(n,n) MinHash-Jaccard distance = mean over slots of (H[i]!=H[j]). The
    canonical MHFP distance (Probst & Reymond, J Cheminform 2018, 10:66)."""
    n, m = H.shape
    D = np.zeros((n, n), dtype=np.float64)
    for s in range(0, n, chunk):
        blk = H[s:s + chunk][:, None, :]          # (c,1,m)
        D[s:s + chunk] = (blk != H[None, :, :]).mean(2)
    np.fill_diagonal(D, 0.0)
    return D


# =========================================================================
# yardsticks
# =========================================================================
def structural_fidelity(D, I, J, mcs):
    """Spearman(D over the MCS pair set, MCS distance). Higher = better."""
    rho = spearmanr(D[I, J], mcs)[0]
    return 0.0 if not np.isfinite(rho) else float(rho)


def score_metric(D, y, groups, I, J, mcs):
    rho = simeval.eval_knn_readacross(D, y, groups)[0]
    # rho stays NaN when the read-across Spearman is undefined -- unlike
    # structural fidelity, it is not folded to 0.0. readacross_defined
    # separates an undefined rho from a measured rho of 0.
    struct = structural_fidelity(D, I, J, mcs)
    return dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
                struct_fidelity=struct)


# =========================================================================
# driver
# =========================================================================
def load_mcs_pairs(name, keep_n):
    """Load cached MCS pair set; in quick mode keep only pairs whose BOTH ends
    fall inside the first keep_n molecules (indices are already 0..n-1 so no
    remap is needed for a leading-prefix subset)."""
    path = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    I, J, mcs = d['I'].astype(int), d['J'].astype(int), d['mcs'].astype(float)
    if keep_n is not None:
        m = (I < keep_n) & (J < keep_n)
        I, J, mcs = I[m], J[m], mcs[m]
    return I, J, mcs


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

    print(f'[{name}] n={n} (quick={quick})  mcs_pairs={len(I)}  '
          f'scaffolds={len(set(groups.tolist()))}', flush=True)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bad = [i for i, m in enumerate(mols) if m is None]
    if bad:
        raise RuntimeError(f'{len(bad)} unparseable SMILES at {bad[:5]}...')

    # ---- baseline: ecfp4_count tanimoto ----
    t0 = time.time()
    D_base = simeval.dist_tanimoto(ecfp4_count)
    base = score_metric(D_base, y, groups, I, J, mcs)
    print(f'[{name}] baseline ecfp4_count:tanimoto  '
          f'readacross_rho={base["readacross_rho"]:.4f}  '
          f'struct={base["struct_fidelity"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)

    results = {'ecfp4_count:tanimoto (BASELINE)': base}
    enc = rdMHFPFingerprint.MHFPEncoder(MHFP_NBITS, MHFP_SEED) if _MHFP_OK else None

    # ---- Avalon (bit): tanimoto, cosine ----
    if _AVALON_OK:
        t0 = time.time()
        Xav = build_avalon(mols)
        for cn, cf in (('tanimoto', simeval.dist_tanimoto), ('cosine', simeval.dist_cosine)):
            results[f'avalon:{cn}'] = score_metric(cf(Xav), y, groups, I, J, mcs)
        print(f'[{name}] avalon done ({time.time()-t0:.1f}s, dim={Xav.shape[1]}, '
              f'mean_onbits={Xav.sum(1).mean():.1f})', flush=True)
    else:
        print(f'[{name}] avalon UNAVAILABLE: {_AVALON_ERR}', flush=True)

    # ---- MHFP6: folded SECFP6 (bit) tanimoto/cosine + native minhash Jaccard ----
    if _MHFP_OK:
        t0 = time.time()
        Xsec = build_secfp6(mols, enc)
        for cn, cf in (('tanimoto', simeval.dist_tanimoto), ('cosine', simeval.dist_cosine)):
            results[f'secfp6:{cn}'] = score_metric(cf(Xsec), y, groups, I, J, mcs)
        H = build_mhfp6_minhash(mols, enc)
        D_mh = mhfp_minhash_distance(H)
        results['mhfp6:minhash_jaccard'] = score_metric(D_mh, y, groups, I, J, mcs)
        print(f'[{name}] mhfp6 done ({time.time()-t0:.1f}s, secfp_onbits='
              f'{Xsec.sum(1).mean():.1f})', flush=True)
    else:
        print(f'[{name}] mhfp6 UNAVAILABLE: {_MHFP_ERR}', flush=True)

    return {'baseline': base, 'results': results,
            'meta': {'n': n, 'quick': quick, 'mcs_pairs': int(len(I))}}


def print_table(name, out):
    base = out['baseline']
    br, bs = base['readacross_rho'], base['struct_fidelity']
    print(f'\n=== {name}: extra fingerprints vs count-FP baseline ===')
    print(f'  {"metric":28s} {"readacross":>11} {"dRA":>8}   {"struct":>8} {"dStruct":>8}   flags')
    for metric, sc in out['results'].items():
        is_base = 'BASELINE' in metric
        dra = sc['readacross_rho'] - br
        dst = sc['struct_fidelity'] - bs
        flags = ''
        if not is_base:
            flags = ('RA+' if dra > 0 else '   ') + ' ' + ('MCS+' if dst > 0 else '    ')
        print(f'  {metric:28s} {sc["readacross_rho"]:>11.4f} '
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
                  if 'BASELINE' not in m and s['struct_fidelity'] > bs]
    print(f'  beats baseline on READ-ACROSS: {winners_ra or "none"}  '
          f'(compared on {n_cmp} metrics; baseline rho '
          f'{"defined" if is_defined(br) else "UNDEFINED"})')
    print(f'  beats baseline on STRUCTURAL : {winners_st or "none"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=['herg_ki'])
    ap.add_argument('--quick', action='store_true', help='smoke: first ~400 mols')
    args = ap.parse_args()

    if not _MAP4_OK:
        print(f'[ISSUES] MAP4 INFEASIBLE locally: {_MAP4_ERR} '
              f'(needs external map4+tmap packages; not installed).', flush=True)

    allout = {}
    for ds in args.datasets:
        out = run(ds, quick=args.quick)
        print_table(ds, out)
        allout[ds] = out

    suffix = '_quick' if args.quick else ''
    dst = os.path.join(HERE, f'extra_fingerprints{suffix}.json')
    json.dump({'map4_feasible': _MAP4_OK,
               'avalon_feasible': _AVALON_OK,
               'mhfp6_feasible': _MHFP_OK,
               'datasets': allout}, open(dst, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(dst)}')


if __name__ == '__main__':
    main()
