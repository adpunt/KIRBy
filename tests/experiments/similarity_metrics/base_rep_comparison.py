"""BARE-MINIMUM HONEST COMPARISON: does 'describe the edit' beat 'describe the
whole sequence' at guessing a variant's measured value from its neighbours?

THE TASK (the one the user called useful): hide a variant's label, guess it by
averaging the labels of its 5 nearest neighbours under a given distance, and see
how well the guesses rank against the real labels. Scaffold-grouped folds (by
mutated position; no random splits). Score = Spearman rank correlation between
guessed and true labels, in [-1, 1]. 0 = the distance is useless for guessing;
1 = perfect. NO model is trained. This uses the REAL measured labels, so there is
no circularity and no independent-grader argument to make.

Two families compared on identical folds:
  DESCRIBE THE EDIT (scaffold-diff): dphyschem, dphyschem_pos
  DESCRIBE THE WHOLE SEQUENCE (base): aac (composition), onehot, blosum_mean
Plus two sanity controls (random distance, constant distance) that MUST score ~0.

IMPORTANT LIMITATION: the big pretrained protein-language-model reps (esm2,
prot_t5, ablang2, ...) are NOT here — they need GPU inference and cannot run
locally. This compares only against the cheap sequence base reps. So the result
answers 'does the edit beat simple whole-sequence descriptors', NOT 'does it beat
the language models'. That second comparison needs a server run.

Every number carries a 95% bootstrap confidence interval, so 'wins' means the
intervals actually separate, not a third-decimal difference.

WHAT THE FOLD SPLIT IS, WRITTEN DOWN. Variants are grouped by their FIRST
mutation's position, so a fold holds out whole mutated positions. That grouping
means one thing only on a single-mutation library, since it reads muts[i][0]
and nothing else. Every output row now carries frac_single, the position-group
count and the variant count, so a later reader can see how the split was cut
without re-deriving it, and the whole read-across is refused below
FRAC_SINGLE_MIN -- the same threshold scaffold_experiment.py already gated
dist_position_gated on.

Two further flags per row say whether a rho is arithmetic rather than measured:
whether every train-test distance in every fold is the same number (onehot does
this on a single-mutation library, since two single mutants at different
positions always differ at exactly four entries), and whether every held-out
variant draws the same five training neighbours (a positional representation
shares no non-zero feature with a held-out position). Both are recorded beside
the score; neither drops a row.

None of the six representations here is in the signed-off representation pool
for these datasets -- they are the cheap local stand-ins the limitation note
above describes.
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
import simeval                                       # noqa: E402
from _nanrank import rank_desc, is_defined           # noqa: E402
from _heldout import (frac_single_mutation,          # noqa: E402
                      degenerate_readacross)
import scaffold_reps as SR                           # noqa: E402
from kirby.datasets.flab import load_flab_dataset    # noqa: E402

warnings.filterwarnings('ignore')

DATASETS = ['flab_expression_koenig', 'flab_binding_koenig',
            'flab_expression_adams', 'flab_binding_adams']

# Folds group variants by their FIRST mutation's position, so a fold holds out
# whole mutated positions. That grouping means one thing only on a
# single-mutation library; below this fraction the score is refused rather than
# written, the same threshold scaffold_experiment.py gates position_gated on.
FRAC_SINGLE_MIN = 0.99


def rep_aac(S):
    n, L = S.shape
    X = np.zeros((n, 20), np.float32)
    for i in range(n):
        X[i] = np.bincount(S[i], minlength=20) / L
    return X


def rep_onehot(S):
    n, L = S.shape
    X = np.zeros((n, L * 20), np.float32)
    for i in range(n):
        X[i, np.arange(L) * 20 + S[i]] = 1.0
    return X


def rep_blosum_mean(S):
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")
    Bm = np.array([[float(B[SR.AA[i], SR.AA[j]]) for j in range(20)]
                   for i in range(20)])
    return Bm[S].mean(1).astype(np.float32)


def knn_readacross_predictions(D, y, groups, k=5, n_splits=5):
    """Return (pred, true) over all held-out variants, so a CI can be bootstrapped
    on the final correlation. Same scaffold-grouped kNN as simeval, exposed."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    n = len(y)
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(np.arange(n), y, groups):
        m = KNeighborsRegressor(n_neighbors=min(k, len(tr)),
                                weights='distance', metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], y[tr])
        pred[te] = m.predict(D[np.ix_(te, tr)])
    ok = np.isfinite(pred)
    return pred[ok], y[ok]


def boot_ci(pred, true, n_boot=1000, seed=0):
    """(rho, ci_lo, ci_hi, n_boot_defined). An undefined Spearman -- constant
    predictions, or fewer than two points -- is NaN, never 0.0: 0.0 is exactly
    the value of a useless distance, so the two must stay distinguishable.
    Resamples whose own correlation is undefined are DROPPED from the percentile
    rather than entered as 0.0, and n_boot_defined reports how many were left."""
    rho = spearmanr(pred, true)[0]
    rho = float(rho) if np.isfinite(rho) else float('nan')
    rs = np.random.RandomState(seed)
    n = len(true)
    boots = []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        r = spearmanr(pred[idx], true[idx])[0]
        if np.isfinite(r):
            boots.append(float(r))
    if not boots:
        return rho, float('nan'), float('nan'), 0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi), len(boots)


def run_dataset(name, seed=0):
    d = load_flab_dataset(name, max_samples=5000)
    S, wt, muts = SR.encode_sequences(d)
    y = np.asarray(d['labels'], float)
    n = len(S)
    groups = np.array([ms[0][0] if ms else -1 for ms in muts])

    reps = {
        # describe the edit
        ('dphyschem', 'euclidean'):     SR.rep_dphys(S, wt, muts),
        ('dphyschem_pos', 'euclidean'): SR.rep_dphys_positional(S, wt, muts),
        # describe the whole sequence
        ('aac', 'euclidean'):           rep_aac(S),
        ('aac', 'cosine'):              rep_aac(S),
        ('onehot', 'euclidean'):        rep_onehot(S),
        ('blosum_mean', 'euclidean'):   rep_blosum_mean(S),
    }
    family = {'dphyschem': 'edit', 'dphyschem_pos': 'edit',
              'aac': 'whole-seq', 'onehot': 'whole-seq',
              'blosum_mean': 'whole-seq'}

    frac_single = frac_single_mutation(muts)
    n_groups = int(len(set(groups.tolist())))
    gate_ok = frac_single >= FRAC_SINGLE_MIN
    split = dict(grouped_by='first_mutated_position', frac_single=frac_single,
                 frac_single_min=FRAC_SINGLE_MIN, n_position_groups=n_groups,
                 n_variants=int(n), readacross_gated=bool(gate_ok))
    if not gate_ok:
        print(f'  {name}: READ-ACROSS REFUSED on every row -- folds hold out whole '
              f'mutated positions, taken from each variant\'s FIRST mutation, but '
              f'only {frac_single:.0%} of variants carry one mutation '
              f'(needs {FRAC_SINGLE_MIN:.0%}).', flush=True)

    def scored(D, fam, rep, coeff):
        """One row. rho is refused when the position grouping is undefined for
        this library. The two degeneracy flags say whether a rho that WAS
        computed is arithmetic rather than measured -- every train-test distance
        the same number, or every variant drawing the same five neighbours.
        Recorded, never used to drop a row."""
        row = dict(dataset=name, family=fam, rep=rep, metric=coeff, **split)
        if not gate_ok:
            row.update(rho=float('nan'), rho_defined=False,
                       ci_lo=float('nan'), ci_hi=float('nan'),
                       ci_defined=False, n_boot_defined=0)
            return row
        pred, true = knn_readacross_predictions(D, y, groups)
        rho, lo, hi, nb = boot_ci(pred, true, seed=seed)
        row.update(rho=rho, rho_defined=is_defined(rho), ci_lo=lo, ci_hi=hi,
                   ci_defined=is_defined(lo) and is_defined(hi),
                   n_boot_defined=nb)
        row.update({f'degenerate_{k}': v for k, v in
                    degenerate_readacross(D, y, groups).items()})
        return row

    rows = []
    for (rep, coeff), X in reps.items():
        D = getattr(simeval, f'dist_{coeff}')(X)
        rows.append(scored(D, family[rep], rep, coeff))

    # controls
    rs = np.random.RandomState(seed)
    Drand = rs.rand(n, n); Drand = (Drand + Drand.T) / 2; np.fill_diagonal(Drand, 0)
    for label, D in (('RANDOM', Drand), ('CONSTANT', np.ones((n, n)) - np.eye(n))):
        rows.append(scored(D, 'control', label, '-'))

    meanmut = float(np.mean([len(m) for m in muts]))
    return rows, n, meanmut


def summarise(all_rows, meta):
    print('\n' + '=' * 82)
    print('CAN THE DISTANCE GUESS A VARIANT\'S MEASURED VALUE FROM ITS NEIGHBOURS?')
    print('Spearman of guessed-vs-true labels, [-1,1], higher better. 95% CI in [].')
    print('No model trained. Big language-model reps NOT included (need a server).')
    print('=' * 82)
    # iterate the datasets actually run, not the full roster: `--datasets` with a
    # subset used to raise KeyError here before printing anything.
    for ds in [d for d in DATASETS if d in meta] + [d for d in meta if d not in DATASETS]:
        n, mm = meta[ds]
        rows = [r for r in all_rows if r['dataset'] == ds]
        if not rows:
            continue
        sp = rows[0]
        print(f'\n{ds}   (n={n}, {mm:.2f} mutations/variant; folds grouped by '
              f'{sp["grouped_by"]}: {sp["n_position_groups"]} groups, '
              f'{sp["frac_single"]:.0%} of variants carry one mutation)')
        if not sp['readacross_gated']:
            print(f'  every rho REFUSED: the grouping reads only each variant\'s '
                  f'first mutation and needs {sp["frac_single_min"]:.0%} '
                  f'single-mutation variants')
        edit = [r for r in rows if r['family'] == 'edit']
        whole = [r for r in rows if r['family'] == 'whole-seq']
        # max() never rejects a NaN and sorted() leaves it where it started, so
        # both the pick and the table order go through rank_desc. A family with
        # no defined rho at all has no best rep to name.
        best_edit = (max(edit, key=lambda r: rank_desc(r['rho']))
                     if any(is_defined(r['rho']) for r in edit) else None)
        best_whole = (max(whole, key=lambda r: rank_desc(r['rho']))
                      if any(is_defined(r['rho']) for r in whole) else None)
        for r in sorted(rows, key=lambda r: rank_desc(r['rho']), reverse=True):
            tag = ''
            if r is best_edit:
                tag = '  <- best EDIT rep'
            elif r is best_whole:
                tag = '  <- best WHOLE-SEQ rep'
            deg = []
            if r.get('degenerate_constant_train_test'):
                deg.append('every train-test distance identical')
            if r.get('degenerate_same_neighbours'):
                deg.append('same five neighbours for every variant')
            dtag = ('   [' + '; '.join(deg) + ']') if deg else ''
            print(f'  {r["family"]:9s} {r["rep"]:14s} {r["metric"]:9s} '
                  f'{r["rho"]:6.3f}  [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]{tag}{dtag}')
        n_def = sum(1 for r in rows if is_defined(r['rho']))
        if n_def < len(rows):
            print(f'  ({len(rows) - n_def} of {len(rows)} rows have an UNDEFINED '
                  f'rho -- nan, not a measured zero)')
        # verdict: do the CIs separate?
        e, w = best_edit, best_whole
        if e is None or w is None:
            fam = 'edit' if e is None else 'whole-seq'
            print(f'  => NO VERDICT (every {fam} rep has an undefined rho)')
            continue
        if not (is_defined(e['ci_lo']) and is_defined(w['ci_hi'])
                and is_defined(w['ci_lo']) and is_defined(e['ci_hi'])):
            print('  => NO VERDICT (a confidence interval could not be formed)')
            continue
        if e['ci_lo'] > w['ci_hi']:
            v = f'EDIT WINS (CIs separate: edit >= {e["ci_lo"]:.3f} > {w["ci_hi"]:.3f})'
        elif w['ci_lo'] > e['ci_hi']:
            v = f'WHOLE-SEQ WINS (CIs separate: whole >= {w["ci_lo"]:.3f} > {e["ci_hi"]:.3f})'
        else:
            v = 'TIE (confidence intervals overlap — no real difference)'
        print(f'  => {v}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    args = ap.parse_args()
    all_rows, meta, t0 = [], {}, time.time()
    for ds in args.datasets:
        rows, n, mm = run_dataset(ds)
        all_rows += rows
        meta[ds] = (n, mm)
        print(f'  {ds:28s} done ({time.time()-t0:.0f}s)', flush=True)
    summarise(all_rows, meta)
    out = os.path.join(HERE, 'base_rep_comparison.json')
    json.dump(all_rows, open(out, 'w'), indent=1)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
