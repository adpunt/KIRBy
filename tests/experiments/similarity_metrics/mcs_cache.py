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

ORDERING WARNING. I and J are row numbers into _data.load_prepared's arrays and
NOTHING ELSE. That ordering is the July study's: sim_<name>.npz, with logd cut to
a 1427-molecule scaffold subsample of its 5039. It is not the framework store's
ordering — tests/find_best_hybrid_molecular.load_dataset deduplicates by InChIKey
and returns 1412 herg_ki molecules against these arrays' 1415, and 5028 logd
molecules against these arrays' 5039. Row 900 is a different molecule in each of
the three orderings, and none of the three converts into another by dropping
rows off the end. These files also predate
`smiles_sha`, so nothing in them records which list they belong to; feeding them
to a framework molecule list is refused by
kirby.similarity.mcs.assert_pairs_match_smiles. Framework-side pairs come from
kirby.similarity.mcs.get_mcs_pairs instead; run
`python -m kirby.similarity.mcs_audit <name>` to see the lists side by side.
"""
import hashlib
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



def smiles_id(smiles) -> str:
    """Hash of the exact molecule list a set of pair positions was drawn from."""
    return hashlib.sha1('\n'.join(map(str, smiles)).encode()).hexdigest()[:16]


def load_pairs(name, smiles=None):
    """Load one dataset's stored pairs, refusing a file drawn from other molecules.

    Pass the molecule list you are about to use. A file with no `smiles_id` was
    written before 2026-09-11 and cannot be checked; it is refused too, because
    "cannot be checked" and "matches" are not the same thing and the whole point
    of the field is that the two used to be indistinguishable.
    """
    path = os.path.join(HERE, f'mcs_pairs_{name}.npz')
    d = np.load(path, allow_pickle=False)
    if smiles is not None:
        stored = str(d['smiles_id']) if 'smiles_id' in d.files else None
        if stored is None:
            raise ValueError(
                f"{path} records no molecule list. It was written before the "
                f"list was stored, so nothing can confirm its pair positions "
                f"belong to the {len(smiles)} molecules you are holding. "
                f"Rebuild it: python tests/experiments/similarity_metrics/"
                f"mcs_cache.py {name}")
        want = smiles_id(smiles)
        if stored != want:
            raise ValueError(
                f"{path} was built from a different molecule list "
                f"(stored {stored}, holding {want}, {len(smiles)} molecules). "
                f"Its pair positions point at other molecules. Rebuild it: "
                f"python tests/experiments/similarity_metrics/mcs_cache.py "
                f"{name}")
    return d

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
    # I and J are POSITIONS into this molecule list, and the file recorded
    # nothing about which list. Rebuild sim_<name>.npz with one molecule added,
    # removed or reordered and every stored pair points at two different
    # molecules, while `mcs` and `dy` still describe the old ones. Nothing
    # raised and nothing looked wrong. `smiles_id` is a hash of the exact list
    # these positions were drawn from; `assert_pairs_match` below refuses a
    # file whose hash disagrees with the molecules a reader is holding.
    np.savez_compressed(out, I=I, J=J, mcs=mcs, dy=dy,
                        smiles_id=np.array(smiles_id(smiles)),
                        n_molecules=np.array(n))
    print(f'[{name}] done in {time.time()-t0:.0f}s -> {out}  '
          f'(mcs mean={mcs.mean():.3f} std={mcs.std():.3f})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--pairs', type=int, default=8000)
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, n_pairs=args.pairs)
