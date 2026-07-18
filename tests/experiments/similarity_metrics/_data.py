"""Shared dataset loader for the similarity-metric local experiments.

One source of truth so the MCS cache and the ranker/aggregation experiments
index the SAME molecules. logd (n=5039) is capped to a ~1400-molecule
scaffold-grouped subsample (whole Bemis-Murcko scaffolds kept together -> no
leakage), exactly as the pilot did, so the local n x n distance matrices are
tractable. herg_ki (n=1415) is used whole.

The cap is deterministic (fixed seed over the sorted unique scaffold groups),
so every script that imports this sees identical arrays.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# the 4 complementary reps the ranker experiment pools over (DCA rule: a count
# fingerprint, a 2D atom-pair fingerprint, physicochemical descriptors, and
# substructure keys -- deliberately NOT three flavours of the same circular FP).
POOL_REPS = ['ecfp4_count', 'atompair_count', 'desc', 'maccs']
# every rep stored in the npz (for the metric grid / structural yardstick)
ALL_REPS = ['ecfp4', 'ecfp6', 'fcfp4', 'ecfp4_count', 'atompair',
            'atompair_count', 'torsion', 'rdkit', 'maccs', 'desc']

CAP_N = 1400          # target molecule count for the logd subsample
CAP_SEED = 0


def _scaffold_subsample(groups, target_n, seed=CAP_SEED):
    """Indices of a scaffold-grouped subsample of ~target_n molecules. Whole
    groups are kept together; groups are shuffled with a fixed seed and added
    until the molecule count reaches target_n. Returns a SORTED index array."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rs = np.random.RandomState(seed)
    order = rs.permutation(len(uniq))
    keep_mask = np.zeros(len(groups), bool)
    count = 0
    for gi in order:
        members = np.where(groups == uniq[gi])[0]
        keep_mask[members] = True
        count += len(members)
        if count >= target_n:
            break
    return np.where(keep_mask)[0]


def load_prepared(name, cap=CAP_N):
    """Load a prepared dataset dict. logd is capped to a scaffold subsample.

    Returns dict with: y, groups, smiles, tanimoto (n x n ECFP4-Tanimoto
    reference, DIAGNOSTIC only), and every rep matrix in ALL_REPS, plus 'idx'
    (the global indices kept, so external per-molecule data can be aligned).
    """
    d = np.load(os.path.join(HERE, f'sim_{name}.npz'), allow_pickle=True)
    y = np.asarray(d['y'], float)
    groups = np.asarray(d['groups'])
    n = len(y)

    if name == 'logd' and cap and n > cap:
        idx = _scaffold_subsample(groups, cap)
    else:
        idx = np.arange(n)

    out = {
        'name': name,
        'idx': idx,
        'y': y[idx],
        'groups': groups[idx],
        'smiles': np.array([str(s) for s in d['smiles']])[idx],
        'tanimoto': np.asarray(d['tanimoto'])[np.ix_(idx, idx)],
    }
    for r in ALL_REPS:
        if r in d:
            out[r] = np.asarray(d[r])[idx]
    out['n'] = len(idx)
    out['n_scaffolds'] = len(set(out['groups'].tolist()))
    return out


if __name__ == '__main__':
    # quick summary + logd cap sanity
    for nm in ('herg_ki', 'logd'):
        p = load_prepared(nm)
        print(f'{nm:8s} n={p["n"]:5d}  scaffolds={p["n_scaffolds"]:5d}  '
              f'pool_dims={sum(p[r].shape[1] for r in POOL_REPS)}')
