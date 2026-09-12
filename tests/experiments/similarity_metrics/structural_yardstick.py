"""Structural yardstick: re-rank the metrics by how well each reflects a
CONFORMATION-FREE structural ground truth (MCS), instead of by activity.

Answers: do count fingerprints still win when the judge is STRUCTURE, not
activity? MCS = maximum common substructure; MCS-Tanimoto similarity =
|shared atoms| / (atoms_A + atoms_B - |shared atoms|). 2D graph only, no 3D.

THE PAIRS AND THE MOLECULES ARE THE SUITE'S. The dataset comes from
_data.load_prepared, so the molecule ordering is the one every other script here
uses (logd is its 1427-molecule scaffold subsample, not the raw 5039), and the
pairs come from mcs_pairs_<name>.npz via mcs_cache.load_pairs. No MCS is
computed in this file any more -- it used to run rdFMCS over 6000 freshly drawn
pairs on every invocation, minutes per dataset, for pairs that were already on
disk. mcs_distance stays here because mcs_cache imports it to BUILD that file.

For each metric (representation x coefficient) we correlate its pairwise
distance against MCS-distance over those pairs, and -- for comparison -- against
|dactivity|. Higher structural correlation = the metric's distances better
mirror shared substructure. The structural correlation is reported twice: plain,
and holding the pair's summed heavy-atom count constant, because big molecules
share proportionally fewer atoms AND sit far apart in most distance spaces, so
size alone buys a correlation.
"""
import os, sys, json, argparse, numpy as np
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _nanrank import rank_desc, is_defined        # noqa: E402
from _data import load_prepared                   # noqa: E402

BIT_REPS = ['ecfp4', 'ecfp6', 'fcfp4', 'atompair', 'torsion', 'rdkit', 'maccs']
COUNT_REPS = ['ecfp4_count', 'atompair_count']
CONT_REPS = ['desc']


# ---- pairwise coefficient distances (operate on matched row-pairs) ----
def _pair(Xi, Xj):
    return Xi.astype(np.float64), Xj.astype(np.float64)

def d_tanimoto(Xi, Xj):
    a, b = _pair(Xi, Xj); c = (a * b).sum(1)
    den = (a * a).sum(1) + (b * b).sum(1) - c
    return 1.0 - np.where(den > 0, c / den, 0.0)

def d_dice(Xi, Xj):
    a, b = _pair(Xi, Xj); c = (a * b).sum(1)
    den = (a * a).sum(1) + (b * b).sum(1)
    return 1.0 - np.where(den > 0, 2 * c / den, 0.0)

def d_cosine(Xi, Xj):
    a, b = _pair(Xi, Xj)
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    den = na * nb
    return 1.0 - np.where(den > 0, (a * b).sum(1) / den, 0.0)

def d_euclidean(Xi, Xj):
    a, b = _pair(Xi, Xj); return np.linalg.norm(a - b, axis=1)

def d_soergel(Xi, Xj):
    a, b = _pair(Xi, Xj)
    num = np.abs(a - b).sum(1); den = np.maximum(a, b).sum(1)
    return np.where(den > 0, num / den, 0.0)

def d_tversky(Xi, Xj, al=0.7, be=0.3):
    a, b = _pair(Xi, Xj); c = (a * b).sum(1)
    den = c + al * ((a * a).sum(1) - c) + be * ((b * b).sum(1) - c)
    return 1.0 - np.where(den > 0, c / den, 0.0)

BIT_COEFFS = {'tanimoto': d_tanimoto, 'dice': d_dice, 'cosine': d_cosine,
              'tversky': d_tversky, 'euclidean': d_euclidean}
COUNT_COEFFS = {'tanimoto': d_tanimoto, 'dice': d_dice, 'cosine': d_cosine,
                'soergel': d_soergel, 'euclidean': d_euclidean}
CONT_COEFFS = {'cosine': d_cosine, 'euclidean': d_euclidean}


def mcs_distance(mol_a, mol_b, timeout=2):
    r = rdFMCS.FindMCS([mol_a, mol_b], timeout=timeout,
                       ringMatchesRingOnly=True, completeRingsOnly=True)
    n = r.numAtoms
    den = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - n
    sim = n / den if den > 0 else 0.0
    return 1.0 - sim


def heavy_atom_counts(smiles):
    """Heavy atoms per molecule -- the quantity MCS divides by (shared atoms
    over atoms_A + atoms_B - shared). 0 for a SMILES RDKit cannot parse."""
    out = np.zeros(len(smiles), dtype=np.float64)
    for k, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        out[k] = 0.0 if m is None else float(m.GetNumHeavyAtoms())
    return out


def run(name, n_pairs=None):
    """One row per representation-and-coefficient pair, over the stored MCS
    pairs. n_pairs caps how many of the stored pairs are used; None uses all of
    them."""
    # Both imported inside the function. mcs_cache imports mcs_distance from
    # this module, so importing it at the top is a cycle; and scaffold_experiment
    # pulls the FLAb loader and calls warnings.filterwarnings('ignore') when it
    # loads, which mcs_cache should not inherit just to build a pair file.
    from mcs_cache import load_pairs                 # noqa: E402
    from scaffold_experiment import partial_spearman  # noqa: E402
    p = load_prepared(name)
    y, smiles, n = p['y'], list(p['smiles']), p['n']

    # load_pairs refuses a file whose stored molecule list disagrees with the
    # one in hand. The three mcs_pairs_*.npz on disk record no list at all, so
    # it is called without one and that check cannot run: UNVERIFIED that these
    # pair positions were drawn from these molecules. Rebuilding with
    # mcs_cache.py writes the smiles_id that makes it checkable.
    d = load_pairs(name)
    I, J = d['I'].astype(int), d['J'].astype(int)
    mcs = d['mcs'].astype(float)
    keep = (I < n) & (J < n)
    I, J, mcs = I[keep], J[keep], mcs[keep]
    if n_pairs is not None:
        I, J, mcs = I[:n_pairs], J[:n_pairs], mcs[:n_pairs]
    dy = np.abs(y[I] - y[J])

    # the control: heavy atoms of the two molecules in a pair, added together.
    sizes = heavy_atom_counts(smiles)
    pair_size = sizes[I] + sizes[J]
    print(f'[{name}] {n} molecules, {len(I)} stored MCS pairs, heavy atoms per '
          f'molecule {sizes.min():.0f}-{sizes.max():.0f} (mean {sizes.mean():.1f})',
          flush=True)

    plan = ([(r, BIT_COEFFS) for r in BIT_REPS if r in p]
            + [(r, COUNT_COEFFS) for r in COUNT_REPS if r in p]
            + [(r, CONT_COEFFS) for r in CONT_REPS if r in p])
    rows = []
    for rep, coeffs in plan:
        X = np.asarray(p[rep])
        Xi, Xj = X[I], X[J]
        if rep in CONT_REPS:
            mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
            Xi = (Xi - mu) / sd; Xj = (Xj - mu) / sd
        for cn, cf in coeffs.items():
            dist = cf(Xi, Xj)
            # Every correlation is NaN when it cannot be computed -- a constant
            # pair distance, or fewer than two pairs. Stored as NaN, with a
            # _defined boolean beside it, so it never reads as a measured zero.
            struct_rho = spearmanr(dist, mcs)[0]
            struct_sz = partial_spearman(dist, mcs, pair_size)
            act_rho = spearmanr(dist, dy)[0]
            rows.append(dict(metric=f'{rep}:{cn}',
                             struct_fidelity=float(struct_rho),
                             struct_fidelity_defined=is_defined(struct_rho),
                             struct_fidelity_size_controlled=float(struct_sz),
                             struct_fidelity_size_controlled_defined=is_defined(struct_sz),
                             activity_fidelity=float(act_rho),
                             activity_fidelity_defined=is_defined(act_rho),
                             n_pairs=int(len(I)), n_molecules=int(n)))
    # max()/sort() never reject a NaN, so an uncomputable metric could sit at the
    # top of this table. rank_desc sends it to the bottom instead.
    rows.sort(key=lambda r: rank_desc(r['struct_fidelity']), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--pairs', type=int, default=None,
                    help='use only the first N stored pairs (default: all)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        out[ds] = run(ds, n_pairs=args.pairs)

    # Written BEFORE anything is printed, and to a filename carrying the
    # datasets that were run: a two-dataset table no longer lands on a
    # three-dataset one, and a metric missing from the summary below can no
    # longer cost the whole run.
    dst = args.out or os.path.join(
        HERE, f'structural_rerank_{"_".join(args.datasets)}.json')
    json.dump(out, open(dst, 'w'), indent=1)
    print(f'\nwrote {os.path.basename(dst)}')

    for ds, rows in out.items():
        n_def = sum(1 for r in rows if is_defined(r['struct_fidelity']))
        print(f'\n=== {ds}: metrics ranked by STRUCTURAL fidelity (MCS) ===')
        print(f'  structural fidelity computed for {n_def}/{len(rows)} metrics '
              f'(nan = it could not be computed, not a measured zero)')
        print(f'  {"metric":22s} {"struct":>8} {"struct|size":>12} {"activity":>9}')
        for r in rows[:8]:
            print(f'  {r["metric"]:22s} {r["struct_fidelity"]:>8.3f} '
                  f'{r["struct_fidelity_size_controlled"]:>12.3f} '
                  f'{r["activity_fidelity"]:>9.3f}')
        rk = [r['metric'] for r in rows]
        by_metric = {r['metric']: r for r in rows}
        for want in ('ecfp4:tanimoto', 'ecfp4_count:tanimoto'):
            r = by_metric.get(want)
            if r is None:
                print(f'  ... {want} is not among the {len(rows)} metrics run')
                continue
            print(f'  ... {want} struct={r["struct_fidelity"]:.3f} '
                  f'(molecule size held constant '
                  f'{r["struct_fidelity_size_controlled"]:.3f}, '
                  f'rank {rk.index(want)+1}/{len(rows)})')


if __name__ == '__main__':
    main()
