"""Scaffold-mutation representation investigation (koenig).

Builds the 5 scaffold-mutation reps + representative base reps (aac, blosum_mean),
pairs them with standard (cosine/euclidean) and NEW scaffold-relative metrics
(position-gated, perturbation-cosine), and grades each resulting similarity space
on:
  * ACTIVITY  — read-across ρ against the fitness label (scaffold-grouped by
    mutated position; NO random splits);
  * STRUCTURE — Spearman of the space's pairwise distance against three
    grader-weighted sequence distances: Grantham (physicochemical), BLOSUM
    (evolutionary), identity (=Hamming, the degenerate baseline).

Anti-circularity: for each rep we mark which graders are INDEPENDENT of the rep's
construction basis and headline only those (e.g. a physicochemical rep is graded
by BLOSUM/identity, never by Grantham). All three are computed for cross-check.

THE FOLD SPLIT, AND WHEN IT MEANS ANYTHING. Variants are grouped by their
FIRST mutation's position, so a fold holds out whole mutated positions. That
grouping, and SR.dist_position_gated, both read only muts[i][0] -- neither is
defined once variants carry more than one mutation. position_gated was already
refused below 99% single-mutation variants; the five-nearest-neighbour
read-across ran with no such check. It is now refused on the same threshold, and
frac_single, the position-group count and the variant count go into EVERY output
row, so a later reader can see how uneven the split was without re-deriving it.

Two further flags per row say whether a read-across number is arithmetic rather
than measured: whether every train-test distance in every fold is the same
number, and whether every held-out variant draws the same five training
neighbours. Both are recorded, never used to drop a row.

koenig only for now. Does NOT touch hybrid.py / find_best_hybrid.py.
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
import simeval                                   # noqa: E402
from _nanrank import rank_desc, is_defined        # noqa: E402
from _heldout import frac_single_mutation, degenerate_readacross  # noqa: E402
import scaffold_reps as SR                       # noqa: E402
from kirby.datasets.flab import load_flab_dataset  # noqa: E402

warnings.filterwarnings('ignore')

GRADERS = ['grantham', 'blosum', 'identity']

# Both the position grouping and SR.dist_position_gated read only the FIRST
# mutation of each variant, so both are defined only on a single-mutation
# library. One threshold, used by both.
FRAC_SINGLE_MIN = 0.99


def conflicts(grader, basis):
    if grader == 'grantham':
        return 'physchem' in basis or 'grantham' in basis
    if grader == 'blosum':
        return 'blosum' in basis
    if grader == 'identity':
        return 'identity' in basis
    return False


# ---- representative BASE reps (whole-sequence; expected near-constant on a
# single-mutation DMS -- that is the point of the comparison) ----
def rep_aac(S):
    n, L = S.shape
    X = np.zeros((n, 20), np.float32)
    for i in range(n):
        c = np.bincount(S[i], minlength=20)
        X[i] = c / L
    return X


def rep_blosum_mean(S):
    from Bio.Align import substitution_matrices
    B = substitution_matrices.load("BLOSUM62")
    Bm = np.array([[float(B[SR.AA[i], SR.AA[j]]) for j in range(20)] for i in range(20)])
    return Bm[S].mean(1).astype(np.float32)     # mean BLOSUM row over residues -> (n,20)

BASE_BASIS = {'aac': {'identity'}, 'blosum_mean': {'blosum'}}


def build_all(dataset='flab_expression_koenig', cap=None, seed=0, n_pairs=8000):
    d = load_flab_dataset(dataset, max_samples=5000)
    S, wt, muts = SR.encode_sequences(d)
    y = np.asarray(d['labels'], float)
    task_type = d.get('task_type', 'regression')
    n = len(S)
    # scaffold groups = mutated position (whole positions kept together; no leakage)
    groups = np.array([ms[0][0] if ms else -1 for ms in muts])

    if cap and n > cap:
        rs = np.random.RandomState(seed)
        idx = rs.choice(n, cap, replace=False)
        S, y, groups = S[idx], y[idx], groups[idx]
        muts = [muts[i] for i in idx]
        n = cap

    gmat, bmat, imat = SR.grantham_matrix(), SR.blosum_matrix(), SR.identity_matrix()

    # representations
    reps = {
        'MutID':          (SR.rep_mutid(S, wt, muts),               SR.REP_BASIS['MutID']),
        'dPhysChem':      (SR.rep_dphys(S, wt, muts),               SR.REP_BASIS['dPhysChem']),
        'dPhysChem_pos':  (SR.rep_dphys_positional(S, wt, muts),    SR.REP_BASIS['dPhysChem_pos']),
        'SubScore':       (SR.rep_subscore(S, wt, muts, gmat, bmat), SR.REP_BASIS['SubScore']),
        'BaseDelta':      (SR.rep_basedelta(S, wt, muts),           SR.REP_BASIS['BaseDelta']),
        'aac':            (rep_aac(S),                              BASE_BASIS['aac']),
        'blosum_mean':    (rep_blosum_mean(S),                      BASE_BASIS['blosum_mean']),
    }

    # sampled pairs + the three structural ground truths on them
    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    m = I != J; I, J = I[m][:n_pairs], J[m][:n_pairs]
    truth = {'grantham': SR.grader_pair_distance(S, I, J, gmat),
             'blosum':   SR.grader_pair_distance(S, I, J, bmat),
             'identity': SR.grader_pair_distance(S, I, J, imat)}
    # Fraction of variants carrying at most ONE mutation. dist_position_gated (and
    # the `groups` definition above) read only muts[i][0], so both are only valid
    # on a genuinely single-mutation library. Recorded so a multi-mutation set
    # cannot silently produce a meaningless number.
    frac_single = frac_single_mutation(muts)
    # Number of mutations carried by each variant, and whether the two members of
    # each sampled pair carry the SAME number. This is the confound: when the count
    # varies, the Grantham/BLOSUM ground truths (which SUM over positions) are
    # largely just counting mutations, so a representation that recovers the count
    # scores well without capturing any substitution chemistry.
    nmut = np.array([len(m) for m in muts])
    eqcount = nmut[I] == nmut[J]
    return dict(S=S, wt=wt, muts=muts, y=y, groups=groups, reps=reps,
                pairs=(I, J), truth=truth, gmat=gmat, n=n, nmut=nmut,
                eqcount=eqcount, n_groups=int(len(set(groups.tolist()))),
                task_type=task_type, frac_single=frac_single, dataset=dataset)


def partial_spearman(x, y, z):
    """Spearman correlation between x and y with the influence of z removed.
    Rank-transform, then apply the standard first-order partial-correlation
    formula. Used with z = Hamming distance (the mutation-count proxy) so that a
    representation which merely counts mutations scores ~0."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    if np.std(rz) == 0:
        return float(spearmanr(x, y)[0])       # z constant: nothing to remove
    c = np.corrcoef(np.vstack([rx, ry, rz]))
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    denom = np.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    if denom < 1e-12:
        return float('nan')
    return float((rxy - rxz * ryz) / denom)


def readacross_classification(D, y, groups, k=5, n_splits=5):
    """Binary analogue of simeval.eval_knn_readacross: SAME scaffold-grouped CV,
    same distance-weighted kNN on the precomputed distance matrix, but scored with
    AUROC instead of Spearman. Used for the censored binding sets, whose labels the
    FLAb loader has already binarised (binder / non-binder). Kept structurally
    identical to the regression path so the two are directly comparable."""
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    n = len(y)
    cls = (np.asarray(y) > np.min(y)).astype(int)
    proba = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(np.arange(n), cls, groups):
        if len(np.unique(cls[tr])) < 2:
            continue                     # single-class train fold: cannot fit
        m = KNeighborsClassifier(n_neighbors=min(k, len(tr)), weights='distance',
                                 metric='precomputed')
        m.fit(D[np.ix_(tr, tr)], cls[tr])
        proba[te] = m.predict_proba(D[np.ix_(te, tr)])[:, 1]
    ok = np.isfinite(proba)
    if ok.sum() == 0 or len(np.unique(cls[ok])) < 2:
        # No AUROC can be formed. NaN, not 0.0 -- 0.0 is the value of a genuinely
        # random ranking on the re-centred scale below, so writing it here would
        # be indistinguishable from a measured result.
        return float('nan')
    # AUROC re-centred to 0 (0 = random, like a null Spearman) so the classification
    # and regression read-across columns share an origin.
    return float(2 * roc_auc_score(cls[ok], proba[ok]) - 1)


def score_space(D, ctx, basis):
    I, J = ctx['pairs']
    # The folds hold out whole MUTATED POSITIONS: `groups` is muts[i][0][0], the
    # first mutation's position. That grouping means the same thing only on a
    # genuinely single-mutation library, which is the same condition
    # dist_position_gated is gated on below. Under 0.99 the five-nearest-
    # neighbour score is refused rather than written as a number, exactly as the
    # position_gated metric is refused, and frac_single plus the group count go
    # into the row either way so a later reader can see how the split was cut.
    gate_ok = ctx['frac_single'] >= FRAC_SINGLE_MIN
    if not gate_ok:
        rho = float('nan')
    elif ctx['task_type'] == 'classification':
        # readacross_classification returns NaN when the AUROC cannot be formed
        # (no finite held-out probability, or one class only), matching the
        # regression path below. readacross_defined records which it was.
        rho = readacross_classification(D, ctx['y'], ctx['groups'])
    else:
        # eval_knn_readacross returns NaN when the Spearman is undefined. That
        # NaN is kept below instead of being written as 0.0, so an undefined
        # read-across is distinguishable from a measured zero correlation.
        rho, _negrmse = simeval.eval_knn_readacross(D, ctx['y'], ctx['groups'])
    dpair = D[I, J]
    ham = ctx['truth']['identity']
    eq = ctx['eqcount']
    # No correlation below is folded to 0.0 when it is undefined: each stays NaN
    # and carries a *_defined boolean, so an undefined grading never reads as a
    # measured correlation of zero.
    out = dict(readacross_rho=float(rho), readacross_defined=is_defined(rho),
               readacross_gated=bool(gate_ok),
               frac_single=float(ctx['frac_single']),
               frac_single_min=FRAC_SINGLE_MIN,
               n_position_groups=int(ctx['n_groups']),
               n_variants=int(ctx['n']),
               grouped_by='first_mutated_position')
    # Whether the score is arithmetic rather than measured: every train-test
    # distance the same number, or every held-out variant drawing the same five
    # training neighbours. Recorded, never used to drop a row.
    if gate_ok:
        out.update({f'degenerate_{k}': v for k, v in
                    degenerate_readacross(D, ctx['y'], ctx['groups']).items()})
    for g in GRADERS:
        r = spearmanr(dpair, ctx['truth'][g])[0]
        out[f'struct_{g}'] = float(r) if np.isfinite(r) else float('nan')
        out[f'struct_{g}_defined'] = is_defined(r)
        out[f'clean_{g}'] = (not conflicts(g, basis))
        # (a) count-controlled: remove the mutation-count (Hamming) component.
        if g != 'identity':
            p = partial_spearman(dpair, ctx['truth'][g], ham)
            out[f'partial_{g}'] = float(p) if np.isfinite(p) else float('nan')
            out[f'partial_{g}_defined'] = is_defined(p)
            # (b) equal-count pairs only: the confound cannot act at all here.
            if eq.sum() >= 200:
                e = spearmanr(dpair[eq], ctx['truth'][g][eq])[0]
                out[f'eqcount_{g}'] = float(e) if np.isfinite(e) else float('nan')
            else:
                out[f'eqcount_{g}'] = float('nan')
            out[f'eqcount_{g}_defined'] = is_defined(out[f'eqcount_{g}'])
    # best CLEAN structural fidelity (the honest headline number). The identity
    # (Hamming) grader is DEGENERATE on a single-mutation DMS — 99% of pairs sit at
    # Hamming 2 (different positions), std ~0.07 — so it cannot grade and is
    # excluded from clean-best. A rep whose ONLY non-circular grader is identity
    # (e.g. SubScore, built from both Grantham AND BLOSUM) therefore has NO clean
    # rich grader -> clean-best = nan (an honest limitation, not a zero).
    # max() never rejects a NaN, so the clean graders are filtered to the finite
    # ones BEFORE the max. No finite clean grader -> nan, and *_n says how many
    # graders the headline was taken over.
    clean = [out[f'struct_{g}'] for g in GRADERS
             if out[f'clean_{g}'] and g != 'identity' and is_defined(out[f'struct_{g}'])]
    out['struct_clean_best'] = float(max(clean)) if clean else float('nan')
    out['struct_clean_best_defined'] = bool(clean)
    out['struct_clean_best_n'] = len(clean)
    # Same headline number under each confound-corrected grading.
    for tag in ('partial', 'eqcount'):
        v = [out[f'{tag}_{g}'] for g in GRADERS
             if g != 'identity' and out[f'clean_{g}'] and is_defined(out[f'{tag}_{g}'])]
        out[f'{tag}_clean_best'] = float(max(v)) if v else float('nan')
        out[f'{tag}_clean_best_defined'] = bool(v)
        out[f'{tag}_clean_best_n'] = len(v)
    return out


def run(dataset='flab_expression_koenig', cap=None):
    t0 = time.time()
    ctx = build_all(dataset=dataset, cap=cap)
    print(f'dataset={dataset} task={ctx["task_type"]} '
          f'frac_single_mutation={ctx["frac_single"]:.2f} '
          f'n_position_groups={ctx["n_groups"]}')
    if ctx['frac_single'] < FRAC_SINGLE_MIN:
        print(f'  READ-ACROSS REFUSED on every row: folds hold out whole mutated '
              f'positions, taken from each variant\'s FIRST mutation, but only '
              f'{ctx["frac_single"]:.0%} of variants carry one mutation '
              f'(needs {FRAC_SINGLE_MIN:.0%}). frac_single and the group count '
              f'are still written into every row.', flush=True)
    print(f'built: n={ctx["n"]}, reps={list(ctx["reps"])}, '
          f'truth spreads grantham/blosum/identity std = '
          f'{ctx["truth"]["grantham"].std():.2f}/{ctx["truth"]["blosum"].std():.2f}/'
          f'{ctx["truth"]["identity"].std():.2f}  ({time.time()-t0:.0f}s)', flush=True)
    rows = []
    for name, (X, basis) in ctx['reps'].items():
        for coeff in ('cosine', 'euclidean'):
            D = getattr(simeval, f'dist_{coeff}')(X)
            r = score_space(D, ctx, basis)
            r.update(rep=name, metric=coeff, basis=sorted(basis))
            rows.append(r)
        del D
        print(f'  {name:14s} done ({time.time()-t0:.0f}s)', flush=True)

    # NEW metric: position-gated (scaffold-aware; built from Grantham -> grade by
    # blosum/identity only). Its implementation reads only muts[i][0], so it is
    # DEFINED only for a single-mutation library. On a multi-mutation set it would
    # silently score a metric that ignores most of each variant, so refuse instead.
    if ctx['frac_single'] >= FRAC_SINGLE_MIN:
        Dpg = SR.dist_position_gated(ctx['S'], ctx['wt'], ctx['muts'], ctx['gmat'])
        r = score_space(Dpg, ctx, {'grantham', 'physchem'})
        r.update(rep='(metric-only)', metric='position_gated', basis=['grantham'])
        rows.append(r)
        print(f'  position_gated done ({time.time()-t0:.0f}s)', flush=True)
    else:
        print(f'  position_gated SKIPPED: only {ctx["frac_single"]:.0%} of variants '
              f'are single-mutation; the metric is undefined for multi-mutation '
              f'variants and would ignore all but the first mutation.', flush=True)
    return rows


def summarise(rows, dataset='flab_expression_koenig'):
    print(f'\n===== {dataset}: scaffold-mutation reps x metrics =====')
    print(f'  {"rep":14s} {"metric":14s} {"read-across":>11} '
          f'{"G":>6} {"B":>6} {"I":>6}  {"raw":>7} {"partial":>8} {"eqcount":>8}')
    # rank_desc places a row with no defined clean grader last instead of letting
    # it hold its insertion slot.
    for r in sorted(rows, key=lambda r: rank_desc(r['struct_clean_best']),
                    reverse=True):
        def mark(g):
            return f"{r['struct_'+g]:.3f}{'' if r['clean_'+g] else '*'}"
        print(f'  {r["rep"]:14s} {r["metric"]:14s} {r["readacross_rho"]:>11.3f} '
              f'{mark("grantham"):>6} {mark("blosum"):>6} {mark("identity"):>6}  '
              f'{r["struct_clean_best"]:>7.3f} {r["partial_clean_best"]:>8.3f} '
              f'{r["eqcount_clean_best"]:>8.3f}')
    n_def = sum(1 for r in rows if is_defined(r['struct_clean_best']))
    print(f'  clean-best defined for {n_def}/{len(rows)} rows '
          f'(nan = no non-circular grader could be correlated, not a zero)')
    print('  (* = grader shares the rep\'s basis -> circular, excluded from clean-best)')
    print('  raw = original grading; partial = mutation-count effect removed; '
          'eqcount = equal-mutation-count pairs only')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap', type=int, default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--dataset', default='flab_expression_koenig')
    args = ap.parse_args()
    rows = run(dataset=args.dataset, cap=1200 if args.quick else args.cap)
    for r in rows:
        r['dataset'] = args.dataset
    summarise(rows, args.dataset)
    tag = '_quick' if args.quick else ''
    # koenig expression keeps its original filename so the committed result stands
    suffix = '' if args.dataset == 'flab_expression_koenig' else f'_{args.dataset}'
    path = os.path.join(HERE, f'scaffold_experiment{suffix}{tag}.json')
    json.dump(rows, open(path, 'w'), indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
