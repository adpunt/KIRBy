"""Precompute MCS (maximum common substructure) distances for a FIXED pool of
sampled molecule pairs, per dataset. This is the slow step (MCS is NP-hard
subgraph iso; ~0.09s/pair) so it is cached once and reused for BOTH:

  1. the structural TARGET of the similarity-aware rankers (R2 and the
     structural-mode variants) -- they need MCS distance for the pairs they rank
     over, and
  2. the structural YARDSTICK when scoring a hybrid similarity space (Spearman
     between the space's pairwise distance and MCS over these same pairs).

Pairs are drawn on the PREPARED dataset indexing (_data.load_prepared), so the
logd cap is respected and every downstream script sees identical pairs.

MCS-distance = 1 - |shared atoms| / (atoms_A + atoms_B - |shared atoms|),
ring-aware, timeout-guarded (same definition as structural_yardstick.py).

Output: mcs_pairs_<name>.npz with I, J (pair indices into the prepared arrays),
mcs (distance), dy (|activity gap|, free by-product).
"""
import os
import sys
import time
import argparse
import numpy as np
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _data import load_prepared                      # noqa: E402
from structural_yardstick import mcs_distance         # noqa: E402

RDLogger.DisableLog('rdApp.*')


def build(name, n_pairs=8000, seed=0):
    p = load_prepared(name)
    y = p['y']
    smiles = p['smiles']
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = np.array([m is not None for m in mols])
    n = len(y)

    rs = np.random.RandomState(seed)
    # oversample then filter self-pairs / unparseable mols, take the first n_pairs
    I = rs.randint(0, n, n_pairs * 3)
    J = rs.randint(0, n, n_pairs * 3)
    m = (I != J) & ok[I] & ok[J]
    I, J = I[m][:n_pairs], J[m][:n_pairs]

    print(f'[{name}] MCS for {len(I)} pairs (n={n})...', flush=True)
    t0 = time.time()
    mcs = np.empty(len(I))
    for k, (i, j) in enumerate(zip(I, J)):
        mcs[k] = mcs_distance(mols[i], mols[j])
        if (k + 1) % 1000 == 0:
            print(f'  {k+1}/{len(I)}  ({time.time()-t0:.0f}s)', flush=True)
    dy = np.abs(y[I] - y[J])
    out = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    np.savez_compressed(out, I=I, J=J, mcs=mcs, dy=dy)
    print(f'[{name}] done in {time.time()-t0:.0f}s -> {out}  '
          f'(mcs mean={mcs.mean():.3f} std={mcs.std():.3f})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--pairs', type=int, default=8000)
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, n_pairs=args.pairs)
