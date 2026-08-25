"""Maximum-common-substructure (MCS) structural-overlap engine + per-dataset cache.

MCS = the largest subgraph two molecules share. MCS-distance =
    1 - |shared atoms| / (atoms_A + atoms_B - |shared atoms|)
ring-aware, timeout-guarded (2D graph only, no 3D). This is the LABEL-BLIND
structural ground truth the similarity recipe is tuned and ranked against — it
never touches activity, so using it to judge activity-label noise is not circular.

Ported from the standalone study (structural_yardstick.mcs_distance,
mcs_cache.build) into the framework so ANY molecular dataset can build/read an
MCS-pair cache once under $KIRBY_CACHE_DIR. MCS is the slow step (NP-hard subgraph
iso, ~0.09s/pair), hence compute-once / read-many.

The cache stores {I, J, mcs, dy} over a FIXED sampled pool of molecule pairs:
  I, J  pair endpoints, indices into the dataset's `smiles` array
  mcs   MCS distance in [0, 1] per pair
  dy    |activity gap| per pair (free by-product; zeros if no labels supplied)
"""
import os
import hashlib
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog('rdApp.*')


def mcs_distance(mol_a, mol_b, timeout=2):
    """Ring-aware MCS distance in [0, 1] (0 = identical scaffold, 1 = disjoint).

    Identical definition to structural_yardstick.mcs_distance (the study origin):
    ringMatchesRingOnly + completeRingsOnly, timeout-guarded.
    """
    r = rdFMCS.FindMCS([mol_a, mol_b], timeout=timeout,
                       ringMatchesRingOnly=True, completeRingsOnly=True)
    n = r.numAtoms
    den = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - n
    sim = n / den if den > 0 else 0.0
    return 1.0 - sim


def compute_mcs_pairs(smiles, y=None, n_pairs=8000, seed=0, timeout=2):
    """Sample molecule pairs and compute their MCS distances.

    Oversamples x3 then filters self-pairs and unparseable molecules, taking the
    first n_pairs valid pairs (mirrors mcs_cache.build). Deterministic in `seed`.

    Args:
        smiles: sequence of SMILES strings (defines the row space that I/J index)
        y: optional per-row activity labels; if given, dy = |y_i - y_j|
        n_pairs: number of pairs to keep
        seed: RNG seed for reproducible pair sampling
        timeout: per-pair MCS timeout (seconds)

    Returns:
        dict {I, J (int64), mcs, dy (float64)}. I/J index into `smiles`.
    """
    smiles = list(smiles)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = np.array([m is not None for m in mols])
    n = len(smiles)
    if n < 2:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return {'I': empty_i, 'J': empty_i.copy(), 'mcs': empty_f, 'dy': empty_f.copy()}

    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 3)
    J = rs.randint(0, n, n_pairs * 3)
    m = (I != J) & ok[I] & ok[J]
    I, J = I[m][:n_pairs], J[m][:n_pairs]

    mcs = np.empty(len(I), dtype=np.float64)
    for k, (i, j) in enumerate(zip(I, J)):
        mcs[k] = mcs_distance(mols[i], mols[j], timeout=timeout)

    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        dy = np.abs(y[I] - y[J])
    else:
        dy = np.zeros(len(I), dtype=np.float64)

    return {'I': I.astype(np.int64), 'J': J.astype(np.int64),
            'mcs': mcs, 'dy': dy}


def _mcs_cache_dir():
    root = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    d = root / 'mcs_pairs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _smiles_hash(smiles, n_pairs, seed):
    """Content hash over the exact SMILES ordering plus sampling params.

    Because I/J index into `smiles`, the cache MUST invalidate whenever the row
    space (or its order) changes — this hash guarantees that.
    """
    h = hashlib.sha256()
    h.update(f'{n_pairs}|{seed}|'.encode())
    for s in smiles:
        h.update(str(s).encode())
        h.update(b'\x00')
    return h.hexdigest()[:16]


def get_mcs_pairs(dataset_name, smiles, y=None, n_pairs=8000, seed=0, timeout=2):
    """Compute-once / read-many per-dataset MCS-pair cache.

    Keyed by (dataset_name, content-hash of smiles, n_pairs, seed) under
    $KIRBY_CACHE_DIR/mcs_pairs/. Returns {I, J, mcs, dy}.
    """
    key = _smiles_hash(smiles, n_pairs, seed)
    path = _mcs_cache_dir() / f'{dataset_name}_{key}.npz'
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in ('I', 'J', 'mcs', 'dy')}
    pairs = compute_mcs_pairs(smiles, y=y, n_pairs=n_pairs, seed=seed,
                              timeout=timeout)
    np.savez_compressed(path, **pairs)
    return pairs


def restrict_pairs_to_rows(pairs, row_idx):
    """Keep only pairs whose BOTH endpoints lie in row_idx (e.g. a train fold).

    Used by the leakage-safe per-fold selection factories (Phase C): structural
    selection must never see held-out rows. I/J in the returned dict stay in the
    ORIGINAL global row space (callers index the full feature matrix by them).

    Args:
        pairs: dict {I, J, mcs, dy} from get_mcs_pairs / compute_mcs_pairs
        row_idx: iterable of allowed global row indices (the train rows)

    Returns:
        dict {I, J, mcs, dy} restricted to in-fold pairs.
    """
    allowed = set(int(r) for r in np.asarray(row_idx).ravel())
    I, J = pairs['I'], pairs['J']
    keep = np.array([(int(i) in allowed) and (int(j) in allowed)
                     for i, j in zip(I, J)], dtype=bool)
    return {'I': I[keep], 'J': J[keep],
            'mcs': pairs['mcs'][keep], 'dy': pairs['dy'][keep]}
