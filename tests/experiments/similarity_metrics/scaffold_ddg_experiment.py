"""Scaffold-mutation representations on protein ddG data (option (c)).

Why this dataset: `graphinity` is SINGLE-POINT ONLY (verified: every row carries
exactly one mutation), so it reproduces the koenig setup -- one reference sequence
plus point mutations -- WITHOUT the mutation-count confound that ruins the adams
and phillips antibody libraries (see SIMILARITY_METRICS_ROADMAP.md, 2026-07-19).

Key structural difference from koenig: graphinity is MULTI-COMPLEX (~1388 PDB
structures). Two variants of DIFFERENT complexes have wholesale different
sequences, so a pair drawn across complexes is graded on complex identity rather
than on the mutation. We therefore only ever compare variants WITHIN the same
complex, then POOL those pairs across complexes into a single correlation.

A complex here is the key the loader stores its sequences under, (pdb, ab_chain)
when the loader hands the antibody chain back per row and the PDB identifier
alone when it does not: one PDB entry can carry more than one antibody chain
pair, and those are different parent sequences. A position is the chain letter
and the residue number together, because heavy and light chain both number a
residue 102 and `same position` is what the closed-form grader below turns on.

SAMPLE SIZE (user decision, 2026-09-11: "keep flat 400 and report each complex
separately and averaged, no scaling"). Every complex contributes the same 400
sampled pairs whatever it holds, so a complex of 20 variants is sampled close to
exhaustively and a complex of 2000 barely at all. The sample is NOT scaled with
the complex's size -- scaling only moves the domination to the largest complex.
Instead all three numbers are written out: the pooled correlation over every
complex's pairs, one correlation per complex on its own, and the mean over
complexes, each carrying the number of variants the complex holds and the number
of pairs drawn from it. The mean over complexes and the pooled correlation
answer different questions and can disagree; both are in the results file and
neither replaces the other.

Within one complex the wildtype is shared and every variant carries exactly one
mutation, so the grader-weighted sequence distance has a closed form:
    same mutated position      -> M[mut_i, mut_j]
    different positions        -> M[wt_i, mut_i] + M[wt_j, mut_j]
which is computed directly from the mutation records (graphinity does not apply
the mutation to its stored sequences, so encode_sequences cannot be used).

NOTE on the activity axis: graphinity ddG labels are FoldX/Rosetta COMPUTED
values, not experimental measurements. This script therefore reports the
STRUCTURAL axis only; predicting a variant's ddG from its five nearest
neighbours here would correlate against a simulator, not against measured
biology.

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
    each). A position is the chain letter and the residue number together, as
    built in run(). Positions are remapped to a local 0..L-1 index so the
    position-indexed reps are dense."""
    uniq = np.unique(pos)
    loc = {p: i for i, p in enumerate(uniq.tolist())}
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
        p = loc[pos[i]]
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


def score_rep(name, coeff, dv, truth):
    """One representation's Spearman against each grader, over the pair distances
    handed in. `truth` maps grader name -> that grader's distance for the SAME
    pairs. Used twice with the same code: once on one complex's pairs, once on
    every complex's pairs concatenated."""
    basis = REP_BASIS[name]
    out = {'rep': name, 'metric': coeff, 'basis': sorted(basis),
           'n_pairs': int(len(dv))}
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
    return out


def pair_distances(X, I, J, coeff):
    """Distance for each sampled pair of variants under one coefficient."""
    if coeff == 'euclidean':
        return np.linalg.norm(X[I] - X[J], axis=1)
    a, b = X[I], X[J]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return 1.0 - np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)


def average_over_complexes(per_complex, n_complexes_kept):
    """One row per (rep, coeff): the mean over complexes of the per-complex
    clean-best Spearman, with the spread and the variant counts that produced
    it. A complex whose clean-best is undefined is counted, not averaged, so
    n_complexes_scored is always printed next to the mean."""
    by = {}
    for r in per_complex:
        by.setdefault((r['rep'], r['metric']), []).append(r)
    rows = []
    for (rep, coeff), rs in by.items():
        vals = [r['struct_clean_best'] for r in rs
                if is_defined(r['struct_clean_best'])]
        nv = [r['n_variants'] for r in rs]
        rows.append({
            'rep': rep, 'metric': coeff,
            'mean_over_complexes': float(np.mean(vals)) if vals else float('nan'),
            'mean_over_complexes_defined': bool(vals),
            'median_over_complexes': float(np.median(vals)) if vals else float('nan'),
            'sd_over_complexes': float(np.std(vals)) if len(vals) > 1 else float('nan'),
            'min_over_complexes': float(np.min(vals)) if vals else float('nan'),
            'max_over_complexes': float(np.max(vals)) if vals else float('nan'),
            'n_complexes_scored': len(vals),
            'n_complexes_kept': int(n_complexes_kept),
            'n_pairs_total': int(sum(r['n_pairs'] for r in rs)),
            'variants_min': int(min(nv)) if nv else 0,
            'variants_median': float(np.median(nv)) if nv else float('nan'),
            'variants_max': int(max(nv)) if nv else 0,
        })
    return rows


def run(preset='foldx', max_samples=60000, min_complex=25, pairs_per_complex=400,
        seed=0):
    t0 = time.time()
    d = load_graphinity_dataset(preset=preset, max_samples=max_samples)
    muts = d['mutations']
    pdb = np.asarray(d['pdb_ids'])
    assert all(len(m) == 1 for m in muts), 'graphinity must be single-point'
    # A position is the chain letter AND the residue number. Heavy and light
    # chain both number a residue 102, so the residue number on its own merges
    # two residues that are not the same residue -- and `same position` is what
    # the closed-form grader and the position-indexed reps both turn on.
    pos = np.array([f"{m[0]['chain']}{m[0]['pos']}" for m in muts])
    wta = np.array([SR.AAI.get(m[0]['wt_aa'], -1) for m in muts])
    muta = np.array([SR.AAI.get(m[0]['mut_aa'], -1) for m in muts])
    # The loader keys its sequences by (pdb, ab_chain): one PDB entry can carry
    # more than one antibody chain pair, and those are different parent
    # sequences, so grouping on the PDB identifier alone pools variants that do
    # not share a wildtype. Group on the same pair when the loader hands the
    # antibody chain back per row, and on the PDB identifier alone when it does
    # not (see the ab_chains note in run()'s caller).
    ab_chains = d.get('ab_chains')
    if ab_chains is None:
        complex_key = pdb
        complex_key_name = 'pdb'
    else:
        complex_key = np.array([f'{p}_{c}' for p, c in zip(d['pdb_ids'], ab_chains)])
        complex_key_name = 'pdb_abchain'
    ok = (wta >= 0) & (muta >= 0)
    pos, wta, muta = pos[ok], wta[ok], muta[ok]
    pdb, complex_key = pdb[ok], complex_key[ok]
    print(f'loaded {ok.sum()} single-point mutations over '
          f'{len(set(complex_key.tolist()))} complexes, grouped by '
          f'{complex_key_name} ({time.time()-t0:.0f}s)', flush=True)

    gmat, bmat, imat = SR.grantham_matrix(), SR.blosum_matrix(), SR.identity_matrix()
    gm = {'grantham': gmat, 'blosum': bmat, 'identity': imat}

    rs = np.random.RandomState(seed)
    acc_d = {}                      # (rep, metric) -> list of pair-distance arrays
    acc_t = {g: [] for g in GRADERS}
    acc_same = []
    per_complex = []                # one row per (complex, rep, metric)
    complex_sizes = []              # one row per kept complex
    n_complexes = 0
    n_complexes_seen = len(set(complex_key.tolist()))
    for c in sorted(set(complex_key.tolist())):
        sel = np.where(complex_key == c)[0]
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
        # This complex's graders, for this complex's pairs. Used twice: pooled
        # into the long list, and scored here on their own.
        tg = {g: grader_pairs(p, w, m, I, J, gm[g]) for g in GRADERS}
        for g in GRADERS:
            acc_t[g].append(tg[g])
        same_c = (p[I] == p[J]).astype(float)
        acc_same.append(same_c)
        # How exhaustively this complex was sampled: the pairs drawn, against
        # the nn*(nn-1)/2 distinct pairs its nn variants can form. The two
        # variants of a pair are drawn independently, so pairs repeat and the
        # ratio can pass 1 on a complex with few variants.
        n_possible = max(nn * (nn - 1) // 2, 1)
        complex_sizes.append({
            'complex': str(c), 'pdb': str(pdb[sel[0]]),
            'n_variants': int(nn), 'n_pairs': int(len(I)),
            'n_possible_pairs': int(n_possible),
            'pairs_drawn_over_pairs_possible': float(len(I)) / float(n_possible),
            'n_mutated_positions': int(len(np.unique(p))),
            'n_same_position_pairs': int(same_c.sum()),
        })
        for name, X in reps.items():
            for coeff in ('cosine', 'euclidean'):
                dv = pair_distances(X, I, J, coeff)
                acc_d.setdefault((name, coeff), []).append(dv)
                row = score_rep(name, coeff, dv, tg)
                row['complex'] = str(c)
                row['pdb'] = str(pdb[sel[0]])
                row['n_variants'] = int(nn)
                row['n_possible_pairs'] = int(n_possible)
                row['n_same_position_pairs'] = int(same_c.sum())
                per_complex.append(row)
        n_complexes += 1
    print(f'pooled {n_complexes} complexes with >={min_complex} mutations '
          f'({time.time()-t0:.0f}s)', flush=True)

    if n_complexes == 0:
        raise ValueError(
            f'no complex cleared the variant floor, so there are no pairs to '
            f'correlate: a complex is kept when it holds at least {min_complex} '
            f'single-point variants, the data holds {n_complexes_seen} complexes '
            f'grouped by {complex_key_name}, and 0 were kept. Lower '
            f'--min-complex or raise --max-samples.')

    truth = {g: np.concatenate(acc_t[g]) for g in GRADERS}
    same = np.concatenate(acc_same)
    print(f'pooled pairs: {len(same)}  (same-position pairs: {int(same.sum())})')
    # Hamming here is 1 for same-position pairs and 2 otherwise -- the residual
    # count-like nuisance. Reported so the confound cannot hide.
    confound = {}
    for g in ('grantham', 'blosum'):
        confound[g] = float(spearmanr(truth['identity'], truth[g])[0])
        print(f'  confound check: rho(identity-grader, {g}-grader) = '
              f'{confound[g]:.3f}')

    pooled = []
    for (name, coeff), chunks in acc_d.items():
        pooled.append(score_rep(name, coeff, np.concatenate(chunks), truth))

    averaged = average_over_complexes(per_complex, n_complexes)
    nv = [cs['n_variants'] for cs in complex_sizes]
    fr = [cs['pairs_drawn_over_pairs_possible'] for cs in complex_sizes]
    meta = {
        'preset': preset,
        'max_samples': int(max_samples),
        'min_complex_variants': int(min_complex),
        'pairs_per_complex': int(pairs_per_complex),
        'pairs_scaled_with_complex_size': False,
        'seed': int(seed),
        'n_single_point_mutations': int(ok.sum()),
        'complex_grouped_by': complex_key_name,
        'position_key': 'chain_letter_and_residue_number',
        'n_complexes_in_data': int(n_complexes_seen),
        'n_complexes_kept': int(n_complexes),
        'n_pooled_pairs': int(len(same)),
        'n_same_position_pairs': int(same.sum()),
        'variants_per_complex_min': int(min(nv)) if nv else 0,
        'variants_per_complex_median': float(np.median(nv)) if nv else float('nan'),
        'variants_per_complex_max': int(max(nv)) if nv else 0,
        'pairs_drawn_over_possible_min': float(min(fr)) if fr else float('nan'),
        'pairs_drawn_over_possible_max': float(max(fr)) if fr else float('nan'),
        'confound_identity_vs_grantham': confound['grantham'],
        'confound_identity_vs_blosum': confound['blosum'],
    }
    return {'meta': meta, 'pooled': pooled, 'averaged': averaged,
            'per_complex': per_complex, 'complex_sizes': complex_sizes}


def summarise(res, preset):
    meta, rows, avg = res['meta'], res['pooled'], res['averaged']
    cs = res['complex_sizes']
    print(f'\n===== graphinity ({preset}): scaffold-mutation reps, '
          f'within-complex pairs =====')
    print(f'  {meta["n_complexes_kept"]} complexes kept of '
          f'{meta["n_complexes_in_data"]} (a complex is kept when it holds at '
          f'least {meta["min_complex_variants"]} variants; a complex is one '
          f'{meta["complex_grouped_by"]})')
    print(f'  {meta["pairs_per_complex"]} pairs drawn from every complex, the '
          f'same count whatever the complex holds (not scaled)')
    print(f'  variants per complex: {meta["variants_per_complex_min"]} smallest, '
          f'{meta["variants_per_complex_median"]:.0f} median, '
          f'{meta["variants_per_complex_max"]} largest')
    print(f'  pairs drawn divided by pairs the complex could form: '
          f'{meta["pairs_drawn_over_possible_max"]:.3f} at its highest, '
          f'{meta["pairs_drawn_over_possible_min"]:.5f} at its lowest '
          f'(the two variants of a pair are drawn independently, so a ratio '
          f'above 1 means pairs repeat)')
    print(f'  {meta["n_pooled_pairs"]} pairs in total, of which '
          f'{meta["n_same_position_pairs"]} mutate the same residue number in '
          f'both variants')

    print(f'\n  POOLED over every complex\'s pairs (one correlation, all pairs '
          f'weighed equally):')
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

    print(f'\n  AVERAGED over complexes (each complex\'s own clean-best Spearman, '
          f'then the mean; every complex weighs the same):')
    print(f'  {"rep":14s} {"metric":10s} {"mean":>7} {"median":>7} {"sd":>7} '
          f'{"min":>7} {"max":>7} {"complexes":>10}')
    for r in sorted(avg, key=lambda r: rank_desc(r['mean_over_complexes']),
                    reverse=True):
        print(f'  {r["rep"]:14s} {r["metric"]:10s} '
              f'{r["mean_over_complexes"]:>7.3f} '
              f'{r["median_over_complexes"]:>7.3f} '
              f'{r["sd_over_complexes"]:>7.3f} '
              f'{r["min_over_complexes"]:>7.3f} '
              f'{r["max_over_complexes"]:>7.3f} '
              f'{str(r["n_complexes_scored"]) + "/" + str(r["n_complexes_kept"]):>10}')
    print('  complexes column = how many complexes gave a defined correlation '
          'out of how many were kept')

    # The widest and the narrowest complex, so the spread behind both tables is
    # visible without opening the results file.
    big = max(cs, key=lambda c: c['n_variants'])
    small = min(cs, key=lambda c: c['n_variants'])
    print(f'\n  largest complex {big["complex"]}: {big["n_variants"]} variants, '
          f'{big["n_pairs"]} pairs drawn of {big["n_possible_pairs"]} possible')
    print(f'  smallest complex {small["complex"]}: {small["n_variants"]} variants, '
          f'{small["n_pairs"]} pairs drawn of {small["n_possible_pairs"]} possible')
    print(f'  every complex\'s own row is in the results file under '
          f'"per_complex" ({len(res["per_complex"])} rows)')
    print('  (* = grader shares the rep\'s basis -> circular, excluded)')
    print('  STRUCTURAL AXIS ONLY: graphinity ddG is FoldX/Rosetta-computed, '
          'not measured.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='foldx', choices=['foldx', 'rosetta'])
    ap.add_argument('--max-samples', type=int, default=60000)
    ap.add_argument('--min-complex', type=int, default=25)
    ap.add_argument('--pairs-per-complex', type=int, default=400)
    args = ap.parse_args()
    res = run(preset=args.preset, max_samples=args.max_samples,
              min_complex=args.min_complex,
              pairs_per_complex=args.pairs_per_complex)
    summarise(res, args.preset)
    path = os.path.join(HERE, f'scaffold_ddg_{args.preset}.json')
    json.dump(res, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
