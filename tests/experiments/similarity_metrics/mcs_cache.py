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
mcs (distance), dy (|activity gap|, free by-product). The directory is HERE
unless SIM_DATA_DIR is set or --out is passed, so a rerun with a smaller
--pairs can be written somewhere else instead of overwriting the 8000-pair
cache the local suite reads.

Two molecule lists, two caches. _data.load_prepared cuts logd to a
1427-molecule scaffold subsample by default and returns all 5039 under
cap=None; a pair position means a different molecule in each list. `--whole`
loads with cap=None and writes mcs_pairs_<name>_whole.npz, so the uncapped
cache cannot overwrite the capped one. load_pairs(..., whole=True) reads that
file, falling back to the plain one when it is absent -- for a dataset the cap
does not cut the two lists are the same, and `smiles_id` is what says so.

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
from _data import load_prepared, CAP_N                # noqa: E402
from structural_yardstick import mcs_distance         # noqa: E402

RDLogger.DisableLog('rdApp.*')

# where the pair caches are read from and written to. HERE unless SIM_DATA_DIR
# says otherwise; build(out_dir=...) / --out overrides it for one run.
PAIR_DIR = os.environ.get('SIM_DATA_DIR', HERE)


def smiles_id(smiles) -> str:
    """Hash of the exact molecule list a set of pair positions was drawn from."""
    return hashlib.sha1('\n'.join(map(str, smiles)).encode()).hexdigest()[:16]


def pairs_path(name, out_dir=None, whole=False):
    """Where one dataset's pair cache lives. `whole` names the cache drawn from
    an uncapped load_prepared -- a separate file, because a pair position drawn
    over logd's 1427-molecule subsample addresses a different molecule in the
    5039-molecule list."""
    stem = f'mcs_pairs_{name}_whole' if whole else f'mcs_pairs_{name}'
    return os.path.join(out_dir or PAIR_DIR, f'{stem}.npz')


def load_pairs(name, smiles=None, out_dir=None, whole=False):
    """Load one dataset's stored pairs, refusing a file drawn from other molecules.

    Pass the molecule list you are about to use. A file with no `smiles_id` was
    written before 2026-09-11 and cannot be checked; it is refused too, because
    "cannot be checked" and "matches" are not the same thing and the whole point
    of the field is that the two used to be indistinguishable.

    `whole=True` asks for the cache drawn from an uncapped load; when that file
    is absent the plain one is read instead, since for a dataset the cap does
    not cut the two molecule lists are identical -- and `smiles_id` is what
    decides, not the file name.
    """
    path = pairs_path(name, out_dir, whole)
    fell_back = whole and not os.path.exists(path)
    if fell_back:
        path = pairs_path(name, out_dir, whole=False)
    d = np.load(path, allow_pickle=False)
    cmd = 'python tests/experiments/similarity_metrics/mcs_cache.py'
    rebuild = f"{cmd} {name}{' --whole' if whole and not fell_back else ''}"
    if fell_back:
        # the plain file is standing in for a _whole one. Either the two lists
        # are the same and restamping this file is enough, or they are not and
        # the uncapped cache has to be built.
        rebuild = (f"{cmd} {name}  (or, if this dataset's uncapped list is "
                   f"larger than its capped one, {cmd} {name} --whole)")
    if smiles is not None:
        stored = str(d['smiles_id']) if 'smiles_id' in d.files else None
        if stored is None:
            raise ValueError(
                f"{path} records no molecule list. It was written before the "
                f"list was stored, so nothing can confirm its pair positions "
                f"belong to the {len(smiles)} molecules you are holding. "
                f"Rebuild it: {rebuild}")
        want = smiles_id(smiles)
        if stored != want:
            raise ValueError(
                f"{path} was built from a different molecule list "
                f"(stored {stored}, holding {want}, {len(smiles)} molecules). "
                f"Its pair positions point at other molecules. Rebuild it: "
                f"{rebuild}")
    return d

def build(name, n_pairs=8000, seed=0, whole=False, out_dir=None):
    """Draw n_pairs molecule pairs and cache their MCS distance. `whole` loads
    every molecule (cap=None) instead of the default subsample and writes the
    _whole cache; `out_dir` overrides SIM_DATA_DIR for this one write."""
    p = load_prepared(name, cap=None if whole else CAP_N)
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

    print(f'[{name}] MCS for {len(I)} pairs (n={n} of {p["n_total"]} molecules)...',
          flush=True)
    t0 = time.time()
    mcs = np.empty(len(I))
    for k, (i, j) in enumerate(zip(I, J)):
        mcs[k] = mcs_distance(mols[i], mols[j])
        if (k + 1) % 1000 == 0:
            print(f'  {k+1}/{len(I)}  ({time.time()-t0:.0f}s)', flush=True)
    dy = np.abs(y[I] - y[J])
    out = pairs_path(name, out_dir, whole)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # I and J are POSITIONS into this molecule list, and the file recorded
    # nothing about which list. Rebuild sim_<name>.npz with one molecule added,
    # removed or reordered and every stored pair points at two different
    # molecules, while `mcs` and `dy` still describe the old ones. Nothing
    # raised and nothing looked wrong. `smiles_id` is a hash of the exact list
    # these positions were drawn from; `load_pairs` above refuses a file whose
    # hash disagrees with the molecules a reader is holding.
    np.savez_compressed(out, I=I, J=J, mcs=mcs, dy=dy,
                        smiles_id=np.array(smiles_id(smiles)),
                        n_molecules=np.array(n))
    print(f'[{name}] done in {time.time()-t0:.0f}s -> {out}  '
          f'(mcs mean={mcs.mean():.3f} std={mcs.std():.3f})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--pairs', type=int, default=8000)
    ap.add_argument('--whole', action='store_true',
                    help='load every molecule (cap=None) and write the _whole '
                         'cache, leaving the capped one alone')
    ap.add_argument('--out', default=None,
                    help=f'directory to write into (default {PAIR_DIR}); a '
                         f'rerun with fewer --pairs belongs somewhere else')
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, n_pairs=args.pairs, whole=args.whole, out_dir=args.out)
