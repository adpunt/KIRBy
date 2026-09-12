"""Shared dataset loader for the similarity-metric local experiments.

One source of truth so the MCS cache and the ranker/aggregation experiments
index the SAME molecules. logd (n=5039) is capped to a ~1400-molecule
scaffold-grouped subsample (whole Bemis-Murcko scaffolds kept together -> no
leakage), exactly as the pilot did, so the local n x n distance matrices are
tractable. herg_ki (n=1415) is used whole.

The cap is deterministic (fixed seed over the sorted unique scaffold groups),
so every script that imports this sees identical arrays.

THE THREE DATASETS ARE DIFFERENT SIZES AND STAY THAT WAY (user decision,
2026-09-11: "leave them and stop comparing across them"). herg_ki returns 1415
molecules, logd returns 1427 of its 5039, adme_solubility returns 2173. Changing
any cap would invalidate every stored hybrid keyed on those molecules, so no cap
moves. What travels instead is the size: load_prepared returns n, n_total,
capped, cap_n and n_scaffolds, and size_label / size_note turn those into text.
Any script that puts a number from one dataset next to a number from another
must print size_note(...) beside it, so the reader can see that the two numbers
rest on different molecule counts.
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
    # Size travels with the data. n is what every downstream array is indexed
    # over; n_total is what the npz held before the cut. They differ only for
    # logd. A script comparing two datasets writes both counts next to the
    # number it reports -- see size_note.
    out['n'] = len(idx)
    out['n_total'] = int(n)
    out['capped'] = bool(len(idx) < n)
    out['cap_n'] = int(cap) if out['capped'] else None
    out['n_scaffolds'] = len(set(out['groups'].tolist()))
    out['size_label'] = size_label(out)
    return out


def size_label(prepared):
    """One dataset's size as text: molecule count, scaffold-group count, and
    the count it was cut from when it was cut."""
    if prepared['capped']:
        return (f"{prepared['name']} {prepared['n']} molecules "
                f"(kept from {prepared['n_total']}, whole scaffold groups, "
                f"target {prepared['cap_n']}) over "
                f"{prepared['n_scaffolds']} scaffold groups")
    return (f"{prepared['name']} {prepared['n']} molecules (whole dataset, "
            f"nothing cut) over {prepared['n_scaffolds']} scaffold groups")


def size_note(*prepared):
    """The sizes of every dataset a number is being compared across, as one
    block of text. Print this beside any number that puts one dataset's result
    next to another's: the two results rest on different molecule counts, and
    the counts are not being equalised."""
    items = prepared[0] if len(prepared) == 1 and isinstance(
        prepared[0], (list, tuple)) else prepared
    lines = ['  molecule counts behind these numbers differ and are not equalised:']
    lines += [f'    {size_label(p)}' for p in items]
    return '\n'.join(lines)


def size_record(*prepared):
    """The same sizes as a plain dict, for writing into a results file next to
    a cross-dataset number."""
    items = prepared[0] if len(prepared) == 1 and isinstance(
        prepared[0], (list, tuple)) else prepared
    return {p['name']: {'n': p['n'], 'n_total': p['n_total'],
                        'capped': p['capped'], 'cap_n': p['cap_n'],
                        'n_scaffolds': p['n_scaffolds']} for p in items}


if __name__ == '__main__':
    # quick summary + logd cap sanity
    loaded = []
    for nm in ('herg_ki', 'logd', 'adme_solubility'):
        try:
            p = load_prepared(nm)
        except FileNotFoundError:
            print(f'{nm:16s} sim_{nm}.npz not present, skipped')
            continue
        loaded.append(p)
        print(f'{nm:16s} n={p["n"]:5d}  of {p["n_total"]:5d}  '
              f'scaffolds={p["n_scaffolds"]:5d}  '
              f'pool_dims={sum(p[r].shape[1] for r in POOL_REPS)}')
    if len(loaded) > 1:
        print(size_note(loaded))
