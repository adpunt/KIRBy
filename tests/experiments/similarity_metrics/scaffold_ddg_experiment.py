"""Scaffold-mutation representations on protein ddG data (option (c)).

Why this dataset: `graphinity` is SINGLE-POINT ONLY (verified: every row carries
exactly one mutation), so it reproduces the koenig setup -- one reference sequence
plus point mutations -- WITHOUT the mutation-count confound that ruins the adams
and phillips antibody libraries (see SIMILARITY_METRICS_ROADMAP.md, 2026-07-19).

Key structural difference from koenig: graphinity is MULTI-COMPLEX (~1388 PDB
structures). Two variants of DIFFERENT complexes have wholesale different
sequences, so a pair drawn across complexes is graded on complex identity rather
than on the mutation. We therefore only ever compare variants WITHIN the same
complex, then POOL those pairs across complexes into a single correlation. We do
not average per-complex correlations (project rule: no averaged scores).

Within one complex the wildtype is shared and every variant carries exactly one
mutation, so the grader-weighted sequence distance has a closed form:
    same mutated position      -> M[mut_i, mut_j]
    different positions        -> M[wt_i, mut_i] + M[wt_j, mut_j]
which is computed directly from the mutation records (graphinity does not apply
the mutation to its stored sequences, so encode_sequences cannot be used).

NOTE on the activity axis: graphinity ddG labels are FoldX/Rosetta COMPUTED
values, not experimental measurements. This script therefore reports the
STRUCTURAL axis only; a read-across number here would be a correlation against a
simulator, not against measured biology.

`position_gated` is deliberately ABSENT: it was proven identical (bit-for-bit) to
the Grantham grader on a single-mutation library, so scoring it would grade a
ground truth against itself.
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
import scaffold_reps as SR                          # noqa: E402
from _nanrank import rank_desc, is_defined             # noqa: E402
from kirby.datasets.graphinity import load_graphinity_dataset   # noqa: E402

warnings.filterwarnings('ignore')

GRADERS = ['grantham', 'blosum', 'identity']
REP_BASIS = {
    'MutID': {'identity'},
    'dPhysChem': {'physchem'},
    'dPhysChem_pos': {'physchem'},
    'SubScore': {'grantham', 'blosum'},
    'BaseDelta': {'identity'},
}


def conflicts(grader, basis):
    if grader == 'grantham':
        return 'physchem' in basis or 'grantham' in basis
    if grader == 'blosum':
        return 'blosum' in basis
    if grader == 'identity':
        return 'identity' in basis
    return False


def build_complex_reps(pos, wta, muta, gmat, bmat):
    """Reps for ONE complex. pos/wta/muta are per-variant arrays (single mutation
    each). Positions are remapped to a local 0..L-1 index so the position-indexed
    reps are dense."""
    uniq = np.unique(pos)
    loc = {int(p): i for i, p in enumerate(uniq)}
    L = len(uniq)
    n = len(pos)
    k = len(SR.PROP_SCALES)
    dprop = np.array([SR.PROPZ[SR.AA[m]] - SR.PROPZ[SR.AA[w]]
                      for w, m in zip(wta, muta)], np.float32)   # (n, k)

    mutid = np.zeros((n, L * 20), np.float32)
    basedelta = np.zeros((n, L * 20), np.float32)
    dphys_pos = np.zeros((n, L * k), np.float32)
    subscore = np.zeros((n, 4), np.float32)
    for i in range(n):
        p = loc[int(pos[i])]
        mutid[i, p * 20 + muta[i]] = 1.0
        basedelta[i, p * 20 + wta[i]] -= 1.0
        basedelta[i, p * 20 + muta[i]] += 1.0
        dphys_pos[i, p * k:(p + 1) * k] = dprop[i]
        subscore[i] = [gmat[wta[i], muta[i]], bmat[wta[i], muta[i]], 1, p]
    return {'MutID': mutid, 'dPhysChem': dprop, 'dPhysChem_pos': dphys_pos,
            'SubScore': subscore, 'BaseDelta': basedelta}


def grader_pairs(pos, wta, muta, I, J, M):
    """Closed-form grader distance for within-complex pairs of single mutants."""
    same = pos[I] == pos[J]
    d_same = M[muta[I], muta[J]]
    d_diff = M[wta[I], muta[I]] + M[wta[J], muta[J]]
    return np.where(same, d_same, d_diff)


def run(preset='foldx', max_samples=60000, min_complex=25, pairs_per_complex=400,
        seed=0):
    t0 = time.time()
    d = load_graphinity_dataset(preset=preset, max_samples=max_samples)
    muts = d['mutations']
    pdb = np.asarray(d['pdb_ids'])
    assert all(len(m) == 1 for m in muts), 'graphinity must be single-point'
    pos = np.array([m[0]['pos'] for m in muts])
    wta = np.array([SR.AAI.get(m[0]['wt_aa'], -1) for m in muts])
    muta = np.array([SR.AAI.get(m[0]['mut_aa'], -1) for m in muts])
    ok = (wta >= 0) & (muta >= 0)
    pos, wta, muta, pdb = pos[ok], wta[ok], muta[ok], pdb[ok]
    print(f'loaded {ok.sum()} single-point mutations over {len(set(pdb))} complexes '
          f'({time.time()-t0:.0f}s)', flush=True)

    gmat, bmat, imat = SR.grantham_matrix(), SR.blosum_matrix(), SR.identity_matrix()
    gm = {'grantham': gmat, 'blosum': bmat, 'identity': imat}

    rs = np.random.RandomState(seed)
    acc_d = {}                      # (rep, metric) -> list of pair-distance arrays
    acc_t = {g: [] for g in GRADERS}
    acc_same = []
    n_complexes = 0
    for c in sorted(set(pdb.tolist())):
        sel = np.where(pdb == c)[0]
        if len(sel) < min_complex:
            continue
        p, w, m = pos[sel], wta[sel], muta[sel]
        nn = len(sel)
        reps = build_complex_reps(p, w, m, gmat, bmat)
        I = rs.randint(0, nn, pairs_per_complex * 2)
        J = rs.randint(0, nn, pairs_per_complex * 2)
        keep = I != J
        I, J = I[keep][:pairs_per_complex], J[keep][:pairs_per_complex]
        if len(I) < 10:
            continue
        for g in GRADERS:
            acc_t[g].append(grader_pairs(p, w, m, I, J, gm[g]))
        acc_same.append((p[I] == p[J]).astype(float))
        for name, X in reps.items():
            for coeff in ('cosine', 'euclidean'):
                if coeff == 'euclidean':
                    dv = np.linalg.norm(X[I] - X[J], axis=1)
                else:
                    a, b = X[I], X[J]
                    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
                    dv = 1.0 - np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)
                acc_d.setdefault((name, coeff), []).append(dv)
        n_complexes += 1
    print(f'pooled {n_complexes} complexes with >={min_complex} mutations '
          f'({time.time()-t0:.0f}s)', flush=True)

    truth = {g: np.concatenate(acc_t[g]) for g in GRADERS}
    same = np.concatenate(acc_same)
    print(f'pooled pairs: {len(same)}  (same-position pairs: {int(same.sum())})')
    # Hamming here is 1 for same-position pairs and 2 otherwise -- the residual
    # count-like nuisance. Reported so the confound cannot hide.
    for g in ('grantham', 'blosum'):
        print(f'  confound check: rho(identity-grader, {g}-grader) = '
              f'{spearmanr(truth["identity"], truth[g])[0]:.3f}')

    rows = []
    for (name, coeff), chunks in acc_d.items():
        dv = np.concatenate(chunks)
        basis = REP_BASIS[name]
        out = {'rep': name, 'metric': coeff, 'basis': sorted(basis)}
        # An undefined grading stays NaN and carries a *_defined boolean; 0.0
        # would read later as a measured correlation of zero.
        for g in GRADERS:
            r = spearmanr(dv, truth[g])[0]
            out[f'struct_{g}'] = float(r) if np.isfinite(r) else float('nan')
            out[f'struct_{g}_defined'] = is_defined(r)
            out[f'clean_{g}'] = (not conflicts(g, basis))
        # max() never rejects a NaN, so filter to the finite clean graders first.
        clean = [out[f'struct_{g}'] for g in GRADERS
                 if out[f'clean_{g}'] and g != 'identity'
                 and is_defined(out[f'struct_{g}'])]
        out['struct_clean_best'] = float(max(clean)) if clean else float('nan')
        out['struct_clean_best_defined'] = bool(clean)
        out['struct_clean_best_n'] = len(clean)
        rows.append(out)
    return rows


def summarise(rows, preset):
    print(f'\n===== graphinity ({preset}): scaffold-mutation reps, '
          f'within-complex pairs pooled =====')
    print(f'  {"rep":14s} {"metric":10s} {"G":>7} {"B":>7} {"I":>7}  {"clean-best":>10}')
    # rank_desc places a row with no defined clean grader last instead of letting
    # it hold its insertion slot.
    for r in sorted(rows, key=lambda r: rank_desc(r['struct_clean_best']),
                    reverse=True):
        def mark(g):
            return f"{r['struct_'+g]:.3f}{'' if r['clean_'+g] else '*'}"
        print(f'  {r["rep"]:14s} {r["metric"]:10s} {mark("grantham"):>7} '
              f'{mark("blosum"):>7} {mark("identity"):>7}  '
              f'{r["struct_clean_best"]:>10.3f}')
    n_def = sum(1 for r in rows if is_defined(r['struct_clean_best']))
    print(f'  clean-best defined for {n_def}/{len(rows)} rows '
          f'(nan = no non-circular grader could be correlated, not a zero)')
    print('  (* = grader shares the rep\'s basis -> circular, excluded)')
    print('  STRUCTURAL AXIS ONLY: graphinity ddG is FoldX/Rosetta-computed, '
          'not measured.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='foldx', choices=['foldx', 'rosetta'])
    ap.add_argument('--max-samples', type=int, default=60000)
    ap.add_argument('--min-complex', type=int, default=25)
    args = ap.parse_args()
    rows = run(preset=args.preset, max_samples=args.max_samples,
               min_complex=args.min_complex)
    summarise(rows, args.preset)
    path = os.path.join(HERE, f'scaffold_ddg_{args.preset}.json')
    json.dump(rows, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
