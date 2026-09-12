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
intervals actually separate, not a third-decimal difference. The edit-versus-
whole-sequence verdict is NOT read off those two marginal intervals. The best
edit representation and the best whole-sequence one are resampled on the SAME
1000 index draws, and the DIFFERENCE between their two Spearman values on each
draw is what carries the interval; the verdict turns on whether that interval
crosses zero.

WHAT THE FOLD SPLIT IS, WRITTEN DOWN. Variants are grouped by the SET of their
mutated positions: two variants whose mutations sit at the same positions go in
one group whatever the substitutions are, so a fold holds out whole combinations
of mutated positions. That is defined however many mutations a variant carries,
which the earlier first-mutation grouping was not, so the two adams libraries at
1.75 mutations per variant are no longer refused. Every output row carries
frac_single, the position-group count and the variant count, so a later reader
can see how the split was cut without re-deriving it. A library with fewer
position-sets than folds is still refused: GroupKFold has nothing to hold out.

WHICH SCORER EACH DATASET GETS. load_flab_dataset binarises the censored binding
assays and returns task_type='classification' for them. Those rows are scored by
scaffold_experiment.readacross_classification -- the same scaffold-grouped
distance-weighted kNN, fitted as a classifier and scored by AUROC re-centred so
0 is a random ranking -- and the number is written as auroc_centred, never as
rho, so a later reader cannot read an AUROC as a Spearman. task_type is on every
row. No paired interval is formed on a centred AUROC here, so those libraries
get no edit-versus-whole-sequence verdict.

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
from scaffold_experiment import readacross_classification  # noqa: E402
from kirby.datasets.flab import load_flab_dataset    # noqa: E402

warnings.filterwarnings('ignore')

DATASETS = ['flab_expression_koenig', 'flab_binding_koenig',
            'flab_expression_adams', 'flab_binding_adams']

# Folds hold out whole SETS of mutated positions, so the grouping is defined
# however many mutations a variant carries and nothing is refused on mutation
# count. A library with fewer position-sets than folds is refused, because
# GroupKFold then has nothing to hold out. frac_single stays on every row: other
# files (scaffold_reps.dist_position_gated, scaffold_experiment's
# 'position_modulo' strategy) read only each variant's FIRST mutation and still
# need it, so the 0.99 threshold lives on there, not here.
N_SPLITS = 5


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


def position_set_groups(muts):
    """Fold group = the SET of a variant's mutated positions. Two variants whose
    mutations sit at the same positions share a group whatever the substitutions
    are; a variant carrying none gets the empty set, which is its own group.
    Returns an integer group id per variant, numbered off the sorted unique
    keys so the numbering does not depend on row order."""
    keys = [tuple(sorted(pos for pos, _w, _m in ms)) for ms in muts]
    ids = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([ids[k] for k in keys], dtype=np.int64)


def knn_readacross_full(D, y, groups, k=5, n_splits=N_SPLITS):
    """Out-of-fold guess for EVERY variant, NaN where no fold produced one. Full
    length, so two representations' guesses stay row-aligned and a paired
    bootstrap can resample the same variants under both."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    n = len(y)
    pred = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(np.arange(n), y, groups):
        m = KNeighborsRegressor(n_neighbors=min(k, len(tr)),
                                weights='distance', metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], y[tr])
        pred[te] = m.predict(D[np.ix_(te, tr)])
    return pred


def knn_readacross_predictions(D, y, groups, k=5, n_splits=N_SPLITS):
    """Return (pred, true) over all held-out variants, so a CI can be bootstrapped
    on the final correlation. Same scaffold-grouped kNN as simeval, exposed."""
    pred = knn_readacross_full(D, y, groups, k=k, n_splits=n_splits)
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


def paired_boot_diff(pred_a, pred_b, true, n_boot=1000, seed=0):
    """(d_rho, ci_lo, ci_hi, n_boot_defined, n_paired). The DIFFERENCE between
    two representations' Spearman scores, resampled on the SAME variants: one
    index array per round, both correlations computed on that same draw, the
    difference kept. Only variants both representations produced a guess for
    enter, so the two correlations are over one set of rows.

    Comparing the two MARGINAL intervals instead declares a tie whenever they
    overlap, which they can do while every single resample puts one ahead. A
    round whose own difference is undefined -- either correlation constant -- is
    DROPPED from the percentile rather than entered as 0.0, the same way boot_ci
    drops one, and n_boot_defined reports how many were left."""
    ok = np.isfinite(pred_a) & np.isfinite(pred_b) & np.isfinite(true)
    a, b, t = pred_a[ok], pred_b[ok], true[ok]
    n = int(ok.sum())
    if n < 2:
        return float('nan'), float('nan'), float('nan'), 0, n
    ra, rb = spearmanr(a, t)[0], spearmanr(b, t)[0]
    d = float(ra - rb) if np.isfinite(ra) and np.isfinite(rb) else float('nan')
    rs = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rs.randint(0, n, n)
        ba, bb = spearmanr(a[idx], t[idx])[0], spearmanr(b[idx], t[idx])[0]
        if np.isfinite(ba) and np.isfinite(bb):
            boots.append(float(ba) - float(bb))
    if not boots:
        return d, float('nan'), float('nan'), 0, n
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d, float(lo), float(hi), len(boots), n


def best_in_family(rows, fam, key='rho'):
    """Highest `key` among the rows of one family, or None when no row in it has
    a defined one. max() never rejects a NaN and sorted() leaves it where it
    started, so the pick goes through rank_desc."""
    cand = [r for r in rows if r['family'] == fam]
    if not any(is_defined(r[key]) for r in cand):
        return None
    return max(cand, key=lambda r: rank_desc(r[key]))


def run_dataset(name, seed=0):
    d = load_flab_dataset(name, max_samples=5000)
    S, wt, muts = SR.encode_sequences(d)
    y = np.asarray(d['labels'], float)
    n = len(S)
    # load_flab_dataset binarises the censored binding assays and says so here.
    # A 0/1 label must not go through the regression neighbour guess and be
    # ranked by Spearman, so this decides which scorer every row gets.
    task_type = d.get('task_type', 'regression')
    groups = position_set_groups(muts)

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
    split_ok = n_groups >= N_SPLITS
    split = dict(grouped_by='set_of_mutated_positions', frac_single=frac_single,
                 n_position_groups=n_groups, n_variants=int(n),
                 n_splits=N_SPLITS, task_type=task_type,
                 readacross_gated=bool(split_ok))
    if not split_ok:
        print(f'  {name}: SCORE REFUSED on every row -- folds hold out whole sets '
              f'of mutated positions, and this library has {n_groups} such sets, '
              f'fewer than the {N_SPLITS} folds.', flush=True)

    # Out-of-fold guesses kept per representation, keyed by (rep, metric), so the
    # paired bootstrap below can resample two of them on one set of index draws.
    preds = {}

    def scored(D, fam, rep, coeff):
        """One row. A regression library gets rho with its own bootstrap
        interval; a classification library gets auroc_centred and no rho, so the
        two can never be read as the same number. Both are refused when the
        library has fewer position-sets than folds. The two degeneracy flags say
        whether a score that WAS computed is arithmetic rather than measured --
        every train-test distance the same number, or every variant drawing the
        same five neighbours. Recorded, never used to drop a row."""
        row = dict(dataset=name, family=fam, rep=rep, metric=coeff, **split)
        row.update(rho=float('nan'), rho_defined=False,
                   ci_lo=float('nan'), ci_hi=float('nan'),
                   ci_defined=False, n_boot_defined=0,
                   auroc_centred=float('nan'), auroc_centred_defined=False)
        if not split_ok:
            return row
        if task_type == 'classification':
            auc = readacross_classification(D, y, groups, n_splits=N_SPLITS)
            row.update(auroc_centred=float(auc),
                       auroc_centred_defined=is_defined(auc))
        else:
            pred = knn_readacross_full(D, y, groups, n_splits=N_SPLITS)
            preds[(rep, coeff)] = pred
            ok = np.isfinite(pred)
            rho, lo, hi, nb = boot_ci(pred[ok], y[ok], seed=seed)
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

    # The edit-versus-whole-sequence verdict comes from ONE distribution: the
    # best edit representation and the best whole-sequence one resampled on the
    # same index draws, the difference between their two Spearman values taken on
    # each draw. Two marginal intervals cannot be subtracted.
    paired = dict(paired_edit_rep=None, paired_whole_rep=None,
                  d_rho=float('nan'), d_ci_lo=float('nan'), d_ci_hi=float('nan'),
                  d_ci_defined=False, d_n_boot_defined=0, d_n_paired_variants=0)
    be, bw = best_in_family(rows, 'edit'), best_in_family(rows, 'whole-seq')
    if be is not None and bw is not None:
        ka, kb = (be['rep'], be['metric']), (bw['rep'], bw['metric'])
        if ka in preds and kb in preds:
            dr, lo, hi, nb, npair = paired_boot_diff(preds[ka], preds[kb], y,
                                                     seed=seed)
            paired.update(paired_edit_rep=f'{ka[0]}:{ka[1]}',
                          paired_whole_rep=f'{kb[0]}:{kb[1]}',
                          d_rho=dr, d_ci_lo=lo, d_ci_hi=hi,
                          d_ci_defined=is_defined(lo) and is_defined(hi),
                          d_n_boot_defined=nb, d_n_paired_variants=int(npair))
    for r in rows:
        r.update(paired)

    meanmut = float(np.mean([len(m) for m in muts]))
    return rows, n, meanmut


def summarise(all_rows, meta):
    print('\n' + '=' * 82)
    print('CAN THE DISTANCE GUESS A VARIANT\'S MEASURED VALUE FROM ITS NEIGHBOURS?')
    print('Regression libraries: Spearman of guessed-vs-true labels, [-1,1], 95% CI in [].')
    print('Binarised binding libraries: AUROC re-centred so 0 is a random ranking.')
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
        is_cls = sp.get('task_type') == 'classification'
        # One column name per library, never both: a centred AUROC and a Spearman
        # are not the same number and must not share a heading.
        key = 'auroc_centred' if is_cls else 'rho'
        print(f'\n{ds}   (n={n}, {mm:.2f} mutations/variant; folds grouped by '
              f'{sp["grouped_by"]}: {sp["n_position_groups"]} groups, '
              f'{sp["frac_single"]:.0%} of variants carry one mutation; scored as '
              f'{"centred AUROC on a binarised label" if is_cls else "Spearman"})')
        if not sp['readacross_gated']:
            print(f'  every score REFUSED: {sp["n_position_groups"]} sets of mutated '
                  f'positions is fewer than the {sp["n_splits"]} folds')
        # A family with no defined score at all has no best rep to name.
        best_edit = best_in_family(rows, 'edit', key)
        best_whole = best_in_family(rows, 'whole-seq', key)
        for r in sorted(rows, key=lambda r: rank_desc(r[key]), reverse=True):
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
            ci = ('  [no interval -- a centred AUROC is not bootstrapped here]'
                  if is_cls else f'  [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]')
            print(f'  {r["family"]:9s} {r["rep"]:14s} {r["metric"]:9s} '
                  f'{r[key]:6.3f}{ci}{tag}{dtag}')
        n_def = sum(1 for r in rows if is_defined(r[key]))
        if n_def < len(rows):
            print(f'  ({len(rows) - n_def} of {len(rows)} rows have an UNDEFINED '
                  f'{key} -- nan, not a measured zero)')
        # verdict: does the PAIRED difference interval clear zero?
        e, w = best_edit, best_whole
        if e is None or w is None:
            fam = 'edit' if e is None else 'whole-seq'
            print(f'  => NO VERDICT (every {fam} rep has an undefined {key})')
            continue
        if is_cls:
            print(f'  => NO VERDICT: {e["rep"]} scores {e[key]:.3f} and {w["rep"]} '
                  f'scores {w[key]:.3f}, and no interval is formed on a centred '
                  f'AUROC here, so nothing says whether the two separate')
            continue
        if not sp['d_ci_defined']:
            print('  => NO VERDICT (the paired difference interval could not be formed)')
            continue
        lo, hi = sp['d_ci_lo'], sp['d_ci_hi']
        where = (f'{sp["paired_edit_rep"]} minus {sp["paired_whole_rep"]} = '
                 f'{sp["d_rho"]:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}] over '
                 f'{sp["d_n_boot_defined"]} resamples of the same '
                 f'{sp["d_n_paired_variants"]} variants')
        if lo > 0:
            v = f'EDIT WINS ({where}; the interval is entirely above zero)'
        elif hi < 0:
            v = f'WHOLE-SEQ WINS ({where}; the interval is entirely below zero)'
        else:
            v = f'TIE ({where}; the interval crosses zero)'
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
