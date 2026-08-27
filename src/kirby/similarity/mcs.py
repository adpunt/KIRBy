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
plus `smiles_sha`, a hash of the exact SMILES ordering I/J were numbered against.

I and J are ROW NUMBERS, so a pair set only means anything beside the molecule
list it was built from, and one dataset name can carry several such lists:
herg_ki is 1415 molecules before the loader's InChIKey deduplication and 1412
after, and the July study caps logd to a 1427-molecule scaffold subsample of the
full 5039. `smiles_sha` says which list a pair set belongs to. It is checked on
every cache read, and assert_pairs_match_smiles refuses a pair set handed to a
different list. A cache file written before `smiles_sha` existed carries none
and is rebuilt, never trusted.
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
        dict {I, J (int64), mcs, dy (float64), smiles_sha}. I/J index into
        `smiles`; smiles_sha identifies that exact ordering.
    """
    smiles = list(smiles)
    sha = _ordering_hash(smiles)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = np.array([m is not None for m in mols])
    n = len(smiles)
    # Heavy-atom count per molecule, stored so the size-controlled structural
    # fidelity can be computed without re-parsing SMILES. MCS distance is
    # 1 - shared/(a + b - shared), so it carries the pair's combined size
    # directly: measured 2026-08-26, Spearman(MCS distance, atoms_i + atoms_j)
    # is 0.53 on herg_ki and 0.44 on logd. NaN for an unparseable molecule.
    n_atoms = np.array([float(m.GetNumAtoms()) if m is not None else np.nan
                        for m in mols], dtype=np.float64)
    if n < 2:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return {'I': empty_i, 'J': empty_i.copy(), 'mcs': empty_f,
                'dy': empty_f.copy(), 'smiles_sha': sha, 'n_atoms': n_atoms}

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
            'mcs': mcs, 'dy': dy, 'smiles_sha': sha, 'n_atoms': n_atoms}


def _mcs_cache_dir():
    root = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    d = root / 'mcs_pairs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ordering_hash(smiles):
    """Content hash over the exact SMILES ordering ALONE (no sampling params).

    This is the identity of the row space that I/J are numbered against. It is
    stored in the .npz and re-checked on read, and it is what
    assert_pairs_match_smiles compares, so a pair set can be checked against a
    molecule list without knowing which n_pairs/seed built it.
    """
    h = hashlib.sha256()
    for s in smiles:
        h.update(str(s).encode())
        h.update(b'\x00')
    return h.hexdigest()[:16]


def _smiles_hash(smiles, n_pairs, seed):
    """Cache key: the SMILES ordering plus the sampling params.

    Because I/J index into `smiles`, the cache MUST invalidate whenever the row
    space (or its order) changes — this hash guarantees that.
    """
    h = hashlib.sha256()
    h.update(f'{n_pairs}|{seed}|{_ordering_hash(smiles)}'.encode())
    return h.hexdigest()[:16]


def pairs_ordering_hash(smiles):
    """Public name for the ordering hash of a molecule list (16 hex chars)."""
    return _ordering_hash(smiles)


def assert_pairs_match_smiles(pairs, smiles, context='structural pairs'):
    """Refuse a pair set whose I/J were numbered against a different molecule list.

    The failure this catches is silent: herg_ki's study pairs reach index 1414,
    which is a valid row in the 1415-molecule study arrays and a DIFFERENT
    molecule in the 1412-molecule deduplicated loader output. Pairs carrying no
    `smiles_sha` (written before the hash existed) are refused too — their
    provenance cannot be proven, so they must be rebuilt.

    Args:
        pairs: dict or NpzFile with I, J and (required here) smiles_sha
        smiles: the molecule list the pairs are about to be used with
        context: caller name, quoted back in the error message

    Returns:
        The ordering hash (16 hex chars) both sides agree on.

    Raises:
        ValueError: on a missing hash, a differing hash, or an index past the
            end of `smiles`.
    """
    smiles = list(smiles)
    order = _ordering_hash(smiles)

    if len(pairs['I']):
        max_ij = max(int(np.max(pairs['I'])), int(np.max(pairs['J'])))
        if max_ij >= len(smiles):
            raise ValueError(
                f'{context}: pair index {max_ij} is past the end of a '
                f'{len(smiles)}-molecule list — these pairs were built against '
                f'a different (longer) molecule list.')

    if 'smiles_sha' not in pairs:
        raise ValueError(
            f'{context}: no smiles_sha stored, so the molecule list these pairs '
            f'were numbered against is unknown. Rebuild them with '
            f'compute_mcs_pairs/get_mcs_pairs against the list in hand '
            f'(ordering {order}).')

    # str in a dict, 0-d '<U16' array once round-tripped through the .npz
    stored = str(np.asarray(pairs['smiles_sha']).item())
    if stored != order:
        raise ValueError(
            f'{context}: pairs were built against SMILES ordering {stored}, but '
            f'the supplied list of {len(smiles)} molecules is ordering {order}. '
            f'Row numbers are not interchangeable between the two — rebuild.')
    return order


def get_mcs_pairs(dataset_name, smiles, y=None, n_pairs=8000, seed=0, timeout=2):
    """Compute-once / read-many per-dataset MCS-pair cache.

    Keyed by (dataset_name, content-hash of smiles, n_pairs, seed) under
    $KIRBY_CACHE_DIR/mcs_pairs/. Returns {I, J, mcs, dy, smiles_sha}.

    A cached file is used only when the SMILES ordering hash stored inside it
    matches `smiles`. A file with no stored hash predates the check and is
    rebuilt over, not trusted; a file whose stored hash contradicts its own
    filename is a corrupted/copied cache entry and raises.
    """
    order = _ordering_hash(smiles)
    key = _smiles_hash(smiles, n_pairs, seed)
    path = _mcs_cache_dir() / f'{dataset_name}_{key}.npz'
    if path.exists():
        z = np.load(path)
        if 'smiles_sha' in z.files:
            stored = str(np.asarray(z['smiles_sha']).item())
            if stored != order:
                raise ValueError(
                    f'{path} stores SMILES ordering {stored}, but its name is '
                    f'the key for ordering {order} — copied or corrupted cache '
                    f'entry, delete it.')
            pairs = {k: z[k] for k in ('I', 'J', 'mcs', 'dy')}
            pairs['smiles_sha'] = stored
            # n_atoms arrived on 2026-08-27. A cache file written before that
            # has the right pairs but no heavy-atom counts, so the
            # size-controlled structural fidelity is unavailable rather than
            # wrong. Recomputing every such file would throw away hours of MCS
            # work for one cheap array, so fill it from the SMILES in hand —
            # the ordering hash above already proved they are the same list.
            if 'n_atoms' in z.files:
                pairs['n_atoms'] = z['n_atoms']
            else:
                pairs['n_atoms'] = np.array(
                    [float(m.GetNumAtoms()) if m is not None else np.nan
                     for m in (Chem.MolFromSmiles(x) for x in smiles)],
                    dtype=np.float64)
                np.savez_compressed(path, **pairs)
            return pairs
        # pre-hash file: the molecule list behind I/J is unknown, so recompute.
        print(f'[mcs] {path.name} has no stored SMILES ordering — rebuilding')
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
        dict {I, J, mcs, dy} restricted to in-fold pairs, carrying the input's
        smiles_sha through when it has one (the row space is unchanged, so the
        restricted pairs belong to the same molecule list).
    """
    allowed = set(int(r) for r in np.asarray(row_idx).ravel())
    I, J = pairs['I'], pairs['J']
    keep = np.array([(int(i) in allowed) and (int(j) in allowed)
                     for i, j in zip(I, J)], dtype=bool)
    out = {'I': I[keep], 'J': J[keep],
           'mcs': pairs['mcs'][keep], 'dy': pairs['dy'][keep]}
    if 'smiles_sha' in pairs:
        out['smiles_sha'] = str(np.asarray(pairs['smiles_sha']).item())
    # n_atoms is per MOLECULE in the global row space, not per pair, so it is
    # carried through whole rather than filtered alongside I/J.
    if 'n_atoms' in pairs:
        out['n_atoms'] = pairs['n_atoms']
    return out
