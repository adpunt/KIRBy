"""Structural yardstick: re-rank the metrics by how well each reflects a
CONFORMATION-FREE structural ground truth (MCS), instead of by activity.

Answers: do count fingerprints still win when the judge is STRUCTURE, not
activity? MCS = maximum common substructure; MCS-Tanimoto similarity =
|shared atoms| / (atoms_A + atoms_B - |shared atoms|). 2D graph only, no 3D.

We sample pairs (full MCS matrix is O(n^2) NP-hard subgraph iso). For each metric
(representation x coefficient) we correlate its pairwise distance against
MCS-distance over the SAME pairs, and — for comparison — against |dactivity|.
Higher structural correlation = the metric's distances better mirror shared
substructure.
"""
import os, sys, json, time, argparse, numpy as np
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))

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


def run(name, n_pairs=6000, seed=0):
    d = np.load(os.path.join(HERE, f'sim_{name}.npz'), allow_pickle=True)
    y = d['y']; smiles = [str(s) for s in d['smiles']]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n = len(y)
    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    m = (I != J) & np.array([mols[i] is not None for i in I]) & np.array([mols[j] is not None for j in J])
    I, J = I[m][:n_pairs], J[m][:n_pairs]

    print(f'[{name}] computing MCS for {len(I)} pairs...', flush=True)
    t0 = time.time()
    mcs = np.array([mcs_distance(mols[i], mols[j]) for i, j in zip(I, J)])
    dy = np.abs(y[I] - y[J])
    print(f'[{name}] MCS done in {time.time()-t0:.0f}s', flush=True)

    plan = ([(r, BIT_COEFFS) for r in BIT_REPS if r in d]
            + [(r, COUNT_COEFFS) for r in COUNT_REPS if r in d]
            + [(r, CONT_COEFFS) for r in CONT_REPS if r in d])
    rows = []
    for rep, coeffs in plan:
        X = d[rep]
        Xi, Xj = X[I], X[J]
        if rep in CONT_REPS:
            mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
            Xi = (Xi - mu) / sd; Xj = (Xj - mu) / sd
        for cn, cf in coeffs.items():
            dist = cf(Xi, Xj)
            struct_rho = spearmanr(dist, mcs)[0]
            act_rho = spearmanr(dist, dy)[0]
            rows.append(dict(metric=f'{rep}:{cn}', struct_fidelity=float(struct_rho),
                             activity_fidelity=float(act_rho)))
    rows.sort(key=lambda r: -r['struct_fidelity'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--pairs', type=int, default=6000)
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        rows = run(ds, n_pairs=args.pairs)
        out[ds] = rows
        print(f'\n=== {ds}: metrics ranked by STRUCTURAL fidelity (MCS) ===')
        print(f'  {"metric":22s} {"struct":>8} {"activity":>9}')
        for r in rows[:8]:
            print(f'  {r["metric"]:22s} {r["struct_fidelity"]:>8.3f} {r["activity_fidelity"]:>9.3f}')
        ec = next(r for r in rows if r['metric'] == 'ecfp4:tanimoto')
        ecc = next(r for r in rows if r['metric'] == 'ecfp4_count:tanimoto')
        rk = [r['metric'] for r in rows]
        print(f'  ... ecfp4:tanimoto struct={ec["struct_fidelity"]:.3f} '
              f'(rank {rk.index("ecfp4:tanimoto")+1}/{len(rows)})')
        print(f'  ... ecfp4_count:tanimoto struct={ecc["struct_fidelity"]:.3f} '
              f'(rank {rk.index("ecfp4_count:tanimoto")+1}/{len(rows)})')
    json.dump(out, open(os.path.join(HERE, 'structural_rerank.json'), 'w'), indent=1)
    print('\nwrote structural_rerank.json')


if __name__ == '__main__':
    main()
