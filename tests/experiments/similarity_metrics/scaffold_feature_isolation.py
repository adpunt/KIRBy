"""Which FEATURES inside the scaffold-diff reps carry the structural signal?

The scaffold study established THAT dphyschem_pos works. This isolates WHY, by
scoring individual features and feature groups instead of the whole rep.

METRIC (stated explicitly, because it is easy to mistake for a model score):
every number is a SPEARMAN RANK CORRELATION in [-1, 1], higher is better. It is
computed over sampled PAIRS of variants: correlate "how far apart this feature
subset says the pair is" against "how far apart an independent substitution
matrix says the pair is". NO model is trained and this is NOT R^2 or accuracy.

Two gradings are reported for every row:
  raw     — plain Spearman.
  ctrl    — the same with the differing-position-count component removed
            (partial correlation against the identity/Hamming grader). On
            libraries where variants carry different numbers of mutations the
            ground truth largely just counts mutations, so `raw` rewards any
            feature set that recovers the count. `ctrl` is the honest column.

Anti-circularity: the physicochemical reps are graded by BLOSUM (evolutionary),
never by Grantham, whose composition/polarity/volume basis overlaps theirs.

NOTE: this script analyses the ORIGINAL five property scales, because it is the
search tool that decides which to keep. The registered pipeline reps
(`kirby.representations.mutation.dphyschem` / `dphyschem_pos`) now ship only the
three that survived that search. Re-run this to revisit the choice.

Three analyses:
  1. PROPERTY SOLO      — each of the 5 property scales alone.
  2. PROPERTY DROP-ONE  — all five minus one, to expose redundancy.
  3. POSITION SPREAD    — for dphyschem_pos, drop each mutated position in turn;
                          is the signal carried by a few sites or spread out?
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
from scaffold_experiment import partial_spearman    # noqa: E402
from kirby.datasets.flab import load_flab_dataset   # noqa: E402

warnings.filterwarnings('ignore')

PROP_NAMES = ['hydrophobicity', 'charge', 'isoelectric_point',
              'molecular_weight', 'aromaticity']


def pair_dist(X, I, J):
    """Euclidean distance between paired rows (the coefficient that won)."""
    return np.linalg.norm(X[I] - X[J], axis=1)


def score(X, I, J, truth_blosum, ham):
    d = pair_dist(X, I, J)
    if np.std(d) == 0:
        return 0.0, 0.0                     # constant distance carries nothing
    raw = spearmanr(d, truth_blosum)[0]
    ctrl = partial_spearman(d, truth_blosum, ham)
    return (float(0 if not np.isfinite(raw) else raw),
            float(0 if not np.isfinite(ctrl) else ctrl))


def run(dataset, n_pairs=8000, seed=0):
    t0 = time.time()
    d = load_flab_dataset(dataset, max_samples=5000)
    S, wt, muts = SR.encode_sequences(d)
    n = len(S)
    bmat, imat = SR.blosum_matrix(), SR.identity_matrix()
    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    m = I != J; I, J = I[m][:n_pairs], J[m][:n_pairs]
    truth = SR.grader_pair_distance(S, I, J, bmat)      # BLOSUM = independent
    ham = SR.grader_pair_distance(S, I, J, imat)
    nmut = np.array([len(x) for x in muts])
    print(f'{dataset}: n={n}, mean mutations/variant={nmut.mean():.2f}, '
          f'pairs={len(I)} ({time.time()-t0:.0f}s)', flush=True)

    Xa = SR.rep_dphys(S, wt, muts)                      # (n, 5)
    Xp = SR.rep_dphys_positional(S, wt, muts)           # (n, L*5)
    k = len(PROP_NAMES)
    L = Xp.shape[1] // k
    rows = []

    def add(analysis, rep, feature, X):
        raw, ctrl = score(X, I, J, truth, ham)
        rows.append(dict(dataset=dataset, analysis=analysis, rep=rep,
                         feature=feature, raw=raw, ctrl=ctrl))

    add('whole', 'dphyschem', 'ALL_5', Xa)
    add('whole', 'dphyschem_pos', 'ALL_5', Xp)

    for j, name in enumerate(PROP_NAMES):
        add('property_solo', 'dphyschem', name, Xa[:, [j]])
        add('property_solo', 'dphyschem_pos', name,
            Xp[:, [p * k + j for p in range(L)]])
    for j, name in enumerate(PROP_NAMES):
        keep = [x for x in range(k) if x != j]
        add('property_drop_one', 'dphyschem', f'without_{name}', Xa[:, keep])
        add('property_drop_one', 'dphyschem_pos', f'without_{name}',
            Xp[:, [p * k + x for p in range(L) for x in keep]])

    # POSITION SPREAD: drop one mutated position at a time from dphyschem_pos.
    mutated = sorted({p for ms in muts for p, _w, _m in ms})
    full_raw, full_ctrl = score(Xp, I, J, truth, ham)
    drops = []
    for p in mutated:
        keep = [c for c in range(Xp.shape[1]) if not (p * k <= c < (p + 1) * k)]
        raw, ctrl = score(Xp[:, keep], I, J, truth, ham)
        drops.append((p, full_ctrl - ctrl))             # damage done by removing p
    drops.sort(key=lambda x: -x[1])
    print(f'  position drop-one over {len(mutated)} mutated positions '
          f'({time.time()-t0:.0f}s)', flush=True)
    for p, dmg in drops[:5]:
        rows.append(dict(dataset=dataset, analysis='position_drop_one',
                         rep='dphyschem_pos', feature=f'pos_{p}',
                         raw=float('nan'), ctrl=float(full_ctrl - dmg),
                         damage=float(dmg)))
    dmgs = np.array([d for _p, d in drops])
    rows.append(dict(dataset=dataset, analysis='position_spread',
                     rep='dphyschem_pos', feature='summary',
                     raw=float(full_raw), ctrl=float(full_ctrl),
                     n_positions=len(mutated),
                     max_single_position_damage=float(dmgs.max()),
                     mean_damage=float(dmgs.mean()),
                     top5_damage_share=float(
                         np.sort(dmgs)[-5:].sum() / max(dmgs.sum(), 1e-12))))
    return rows


def summarise(rows):
    ds = rows[0]['dataset']
    print(f'\n===== {ds}: which features carry the structural signal =====')
    print('  (Spearman rank correlation, -1..1, higher better; graded by BLOSUM.')
    print('   ctrl = mutation-count component removed -- the honest column.)')
    for analysis in ('whole', 'property_solo', 'property_drop_one'):
        print(f'\n  -- {analysis} --')
        print(f'  {"rep":16s} {"feature":22s} {"raw":>7} {"ctrl":>7}')
        for r in [x for x in rows if x['analysis'] == analysis]:
            print(f'  {r["rep"]:16s} {r["feature"]:22s} '
                  f'{r["raw"]:>7.3f} {r["ctrl"]:>7.3f}')
    s = [x for x in rows if x['analysis'] == 'position_spread'][0]
    print(f'\n  -- position spread (dphyschem_pos) --')
    print(f'  mutated positions            : {s["n_positions"]}')
    print(f'  full rep ctrl score          : {s["ctrl"]:.3f}')
    print(f'  worst single-position damage : {s["max_single_position_damage"]:.4f}')
    print(f'  mean per-position damage     : {s["mean_damage"]:.4f}')
    print(f'  share of damage from top 5   : {s["top5_damage_share"]:.1%}')
    print('  (small damage everywhere = signal is spread across sites, not a few)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['flab_expression_koenig', 'flab_expression_adams',
                             'flab_binding_phillips_h3'])
    args = ap.parse_args()
    allrows = []
    for ds in args.datasets:
        rows = run(ds)
        summarise(rows)
        allrows += rows
    path = os.path.join(HERE, 'scaffold_feature_isolation.json')
    json.dump(allrows, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
