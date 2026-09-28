#!/usr/bin/env python3
"""
Scaffold-Diff Similarity Experiment (OPIG / curiosity suite)

Small perturbations of a reference — a wildtype antibody with one point mutation,
a protein mutation with a ΔΔG, a matched molecular pair — break ordinary
representations: every variant looks ~99% identical, so a whole-object encoding is
nearly constant. This experiment studies SCAFFOLD-DIFF representations (describe a
variant by WHAT CHANGED relative to its reference) as SIMILARITY SPACES, graded
against INDEPENDENT structural ground truths.

Scientific questions (one per mode):
  1. REPRESENTATIONS — which scaffold-diff representation × distance metric
     produces a similarity space whose distances track true structural
     difference?  (--mode representations)
  2. FEATURES — what KIND of feature earns its place in a distance-based
     similarity space? Is it signal strength, or non-redundancy?
     (--mode features). Each dataset's variants are split in two halves; a
     pair is kept only when both its variants are in the same half. The
     feature subset is chosen on one half and its score reported from the
     other, so greedy's number is not measured on the pairs that shaped it.
     --n-pairs defaults to 32000, which leaves about 16000 pairs in each
     half — twice what the un-split 8000-pair sample carried.
  3. CONTROLS — are the numbers honest? Three failure modes that silently
     inflate scores, each tested explicitly.  (--mode controls)

METRIC — stated because it is easily mistaken for a model score. Every number is
a SPEARMAN RANK CORRELATION in [-1, 1], higher better, computed over sampled PAIRS
of variants: correlate "how far apart the representation says the pair is" against
"how far apart an independent substitution matrix says they are". NO model is
trained. This is NOT R², NOT accuracy, NOT predictive performance.

THREE HONESTY CONTROLS (all mandatory; `--mode controls` reports them)
  C1 BASIS INDEPENDENCE — never grade a representation with a ground truth built
     on its own basis (a physicochemical rep is graded by BLOSUM, never Grantham).
  C2 COUNT CONTROL — the Grantham/BLOSUM ground truths SUM over positions, so on a
     library where variants differ in how many positions changed they largely just
     count differences. Any representation that recovers that count then scores
     high with no chemistry. Every score is reported with the count component
     partialled out.
  C3 METRIC ≠ GROUND TRUTH — a proposed distance must not be algebraically
     identical to a grader. This caught `position_gated`, which is bit-for-bit the
     Grantham ground truth on a single-mutation library and whose apparent 0.519
     was grader-vs-grader agreement, not a result.

DATASETS — every scaffold-compatible dataset in the repo, both antibody and
protein, single-scaffold and multi-scaffold. Multi-scaffold sets (graphinity)
compare variants only WITHIN a reference and pool the pairs, because a
cross-reference pair is graded on wholesale sequence difference, not on the
mutation.

Usage:
    # smoke test (~1 min)
    python tests/experiments/opig_scaffold_similarity.py --mode all --smoke

    # one mode, all datasets
    python tests/experiments/opig_scaffold_similarity.py --mode features

    # everything (the publication run)
    python tests/experiments/opig_scaffold_similarity.py --mode all

Results doc: SCAFFOLD_SIMILARITY_RESULTS.md
Roadmap:     SIMILARITY_METRICS_ROADMAP.md
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(HERE, 'similarity_metrics')
sys.path.insert(0, _SIM)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

# --distance-metric, taken straight from the config module: this script does not
# put tests/ on sys.path, so experiments._shared is not importable here.
from kirby.similarity.distance_config import (          # noqa: E402
    DISTANCE_METRICS as _DISTANCE_METRICS,
    DEFAULT_DISTANCE_METRIC as _DEFAULT_DISTANCE_METRIC,
    set_distance_metric as _set_distance_metric,
)


def add_distance_metric_arg(parser):
    parser.add_argument('--distance-metric', type=str,
                        default=_DEFAULT_DISTANCE_METRIC,
                        choices=list(_DISTANCE_METRICS),
                        help=('Pairwise distance for every neighbour, cliff and '
                              'retrieval quantity this script computes. Default '
                              'manhattan. KIRBY_DISTANCE_METRIC in the '
                              'environment does the same thing.'))
    return parser


def apply_distance_metric_arg(args):
    return _set_distance_metric(
        getattr(args, 'distance_metric', None) or _DEFAULT_DISTANCE_METRIC)


import scaffold_reps as SR                                  # noqa: E402
from feature_type_study import (SCALES, NAMES,              # noqa: E402
                                property_table, score_subset, greedy_forward)
from scaffold_experiment import partial_spearman            # noqa: E402

warnings.filterwarnings('ignore')

# ── Dataset registry ─────────────────────────────────────────────────────────
# kind: 'single_scaffold' = one wildtype, pairs drawn globally.
#       'multi_scaffold'  = many references, pairs drawn WITHIN a reference.
SCAFFOLD_DATASETS = {
    'flab_expression_koenig':   dict(kind='single_scaffold', domain='antibody'),
    'flab_binding_koenig':      dict(kind='single_scaffold', domain='antibody'),
    'flab_expression_adams':    dict(kind='single_scaffold', domain='antibody'),
    'flab_binding_adams':       dict(kind='single_scaffold', domain='antibody'),
}
# Excluded, with the reason recorded so it is not silently dropped:
EXCLUDED_DATASETS = {
    'flab_thermo_hie':     'n=10 — far too small',
    'flab_binding_rosace': 'n=14 — far too small',
    'ddg_combined':        'multi-point mutations + multi-PDB; needs the '
                           'multi-mutation generalisation, not yet built',
    # Moved out of SCAFFOLD_DATASETS 2026-09-12. Both were already cut from the
    # roster; this file was still listing them, so it was the only place either
    # still ran.
    'flab_binding_phillips_h3': 'CUT 2026-07-18 (DATASET_STATUS.md) — single-WT '
                                'combinatorial library, no clean grouped split, '
                                'saturated (aac AUROC 0.98 while leaking)',
    'graphinity_foldx':         'CUT 2026-07-09 — labels are FoldX simulation '
                                'output, not measurements. Its sibling '
                                'graphinity_rosetta went with it, and '
                                'opig_graphinity_ddg dropped both conditions '
                                'on 2026-09-04',
}

GRADERS = ['grantham', 'blosum', 'identity']
# which ground truths share a basis with which representation
REP_BASIS = {'MutID': {'identity'}, 'dPhysChem': {'physchem'},
             'dPhysChem_pos': {'physchem'}, 'SubScore': {'grantham', 'blosum'},
             'BaseDelta': {'identity'}, 'aac': {'identity'},
             'blosum_mean': {'blosum'}}


def conflicts(grader, basis):
    """C1: does this ground truth share the representation's basis?"""
    return ((grader == 'grantham' and ('physchem' in basis or 'grantham' in basis))
            or (grader == 'blosum' and 'blosum' in basis)
            or (grader == 'identity' and 'identity' in basis))


# ── Data access ──────────────────────────────────────────────────────────────
def load_single_scaffold(name, n_pairs, seed, max_samples):
    from kirby.datasets.flab import load_flab_dataset
    d = load_flab_dataset(name, max_samples=max_samples)
    S, wt, muts = SR.encode_sequences(d)
    # No label is used below, so a row is a SEQUENCE. The Adams sets carry the
    # same sequence on many rows (5000 rows, 2400 distinct sequences after the
    # loader's subsample, measured 2026-09-28). Kept as rows, two copies of one
    # sequence formed pairs at distance 0 on every representation and grader,
    # and could sit in both variant halves, so --mode features chose and
    # reported on the same variant. Each distinct sequence is kept once.
    n_rows = len(S)
    _u, first = np.unique(S, axis=0, return_index=True)
    first = np.sort(first)
    S = S[first]
    muts = [muts[i] for i in first]
    n = len(S)
    rs = np.random.RandomState(seed)
    # Each VARIANT goes to half 0 or half 1, and a pair is kept only when both
    # of its variants are in the same half. `--mode features` then chooses its
    # feature subset on half 0 and reports that subset's score from half 1, so
    # no variant contributes to both the choosing and the reporting. About half
    # the drawn candidates cross the two halves and are dropped, which is why
    # --n-pairs defaults to 32000. Measured on flab_expression_koenig: 15254
    # pairs in the choosing half and 16548 in the reporting half, so each half
    # carries about twice the 8000 pairs the whole sample used to carry, and
    # the whole load takes 25 s.
    variant_half = rs.randint(0, 2, n)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    m = (I != J) & (variant_half[I] == variant_half[J])
    I, J = I[m][:n_pairs], J[m][:n_pairs]
    truth = {'grantham': SR.grader_pair_distance(S, I, J, SR.grantham_matrix()),
             'blosum': SR.grader_pair_distance(S, I, J, SR.blosum_matrix()),
             'identity': SR.grader_pair_distance(S, I, J, SR.identity_matrix())}
    return dict(name=name, kind='single_scaffold', S=S, wt=wt, muts=muts, n=n,
                n_rows_loaded=n_rows,
                pairs=(I, J), pair_half=variant_half[I], truth=truth,
                mean_mutations=float(np.mean([len(x) for x in muts])),
                max_mutations=int(max((len(x) for x in muts), default=0)),
                task_type=d.get('task_type', 'regression'))


def load_multi_scaffold(name, n_pairs, seed, max_samples):
    """graphinity: single-point, ~1388 complexes. Pairs WITHIN a complex only."""
    from kirby.datasets.graphinity import load_graphinity_dataset
    preset = name.split('_', 1)[1] if '_' in name else 'foldx'
    d = load_graphinity_dataset(preset=preset, max_samples=max_samples)
    muts_raw, pdb = d['mutations'], np.asarray(d['pdb_ids'])
    pos = np.array([m[0]['pos'] for m in muts_raw])
    wta = np.array([SR.AAI.get(m[0]['wt_aa'], -1) for m in muts_raw])
    mta = np.array([SR.AAI.get(m[0]['mut_aa'], -1) for m in muts_raw])
    ok = (wta >= 0) & (mta >= 0)
    pos, wta, mta, pdb = pos[ok], wta[ok], mta[ok], pdb[ok]
    return dict(name=name, kind='multi_scaffold', pos=pos, wta=wta, mta=mta,
                pdb=pdb, n=int(ok.sum()), mean_mutations=1.0,
                n_pairs_per_complex=max(50, n_pairs // 200), seed=seed)


def load_dataset(name, n_pairs=32000, seed=0, max_samples=5000):
    kind = SCAFFOLD_DATASETS[name]['kind']
    if kind == 'single_scaffold':
        return load_single_scaffold(name, n_pairs, seed, max_samples)
    return load_multi_scaffold(name, n_pairs, seed, max_samples if
                               max_samples > 5000 else 60000)


# ── Feature-space construction (shared by all modes) ─────────────────────────
def feature_sqdiff(ctx, zscore=True, positional=True):
    """Per-feature squared pair-differences (n_features, n_pairs), plus the
    BLOSUM ground truth, the count proxy, and each pair's variant half.

    Euclidean distance over a feature subset is sqrt of the summed rows, so any
    subset can be scored without rebuilding the space — this is what makes the
    exhaustive and greedy searches in --mode features affordable.

    The fourth return is a 0/1 array, one entry per pair, saying which variant
    half both of that pair's variants came from. Only `--mode features` uses
    it: it chooses on half 0 and reports from half 1.
    """
    T = property_table(zscore)
    k = len(NAMES)
    if ctx['kind'] == 'single_scaffold':
        S, muts = ctx['S'], ctx['muts']
        I, J = ctx['pairs']
        sites = sorted({p for ms in muts for p, _w, _m in ms})
        sidx = {p: i for i, p in enumerate(sites)}
        width = len(sites) if positional else 1
        X = np.zeros((len(S), width, k), np.float32)
        for i, ms in enumerate(muts):
            for p, w, m in ms:
                if positional:
                    X[i, sidx[p]] = T[m] - T[w]
                else:
                    X[i, 0] += T[m] - T[w]
        SD = np.empty((k, len(I)), np.float64)
        for j in range(k):
            dd = X[I, :, j] - X[J, :, j]
            SD[j] = np.sum(dd * dd, axis=1)
        return (SD, ctx['truth']['blosum'], ctx['truth']['identity'],
                np.asarray(ctx['pair_half']))

    # multi-scaffold: build per complex, pool pairs
    pos, wta, mta, pdb = ctx['pos'], ctx['wta'], ctx['mta'], ctx['pdb']
    bmat, imat = SR.blosum_matrix(), SR.identity_matrix()
    rs = np.random.RandomState(ctx['seed'])
    per = ctx['n_pairs_per_complex']
    sd_chunks, t_chunks, h_chunks, half_chunks = [], [], [], []
    for c in sorted(set(pdb.tolist())):
        sel = np.where(pdb == c)[0]
        if len(sel) < 25:
            continue
        p, w, m = pos[sel], wta[sel], mta[sel]
        nn = len(sel)
        uniq = np.unique(p); loc = {int(v): i for i, v in enumerate(uniq)}
        width = len(uniq) if positional else 1
        X = np.zeros((nn, width, k), np.float32)
        for i in range(nn):
            slot = loc[int(p[i])] if positional else 0
            X[i, slot] += T[m[i]] - T[w[i]]
        # Same variant-half rule as the single-scaffold loader, applied inside
        # each complex so both halves cover every complex.
        variant_half = rs.randint(0, 2, nn)
        I = rs.randint(0, nn, per * 2); J = rs.randint(0, nn, per * 2)
        keep = (I != J) & (variant_half[I] == variant_half[J])
        I, J = I[keep][:per], J[keep][:per]
        if len(I) < 10:
            continue
        chunk = np.empty((k, len(I)), np.float64)
        for j in range(k):
            dd = X[I, :, j] - X[J, :, j]
            chunk[j] = np.sum(dd * dd, axis=1)
        sd_chunks.append(chunk)
        half_chunks.append(variant_half[I])
        same = p[I] == p[J]
        for M, out in ((bmat, t_chunks), (imat, h_chunks)):
            out.append(np.where(same, M[m[I], m[J]],
                                M[w[I], m[I]] + M[w[J], m[J]]))
    return (np.concatenate(sd_chunks, axis=1),
            np.concatenate(t_chunks), np.concatenate(h_chunks),
            np.concatenate(half_chunks))


# ── MODE 1: representations ──────────────────────────────────────────────────
def mode_representations(ctx):
    """Which scaffold-diff representation × distance metric wins, per C1+C2."""
    if ctx['kind'] != 'single_scaffold':
        SD, truth, ham, _half = feature_sqdiff(ctx)
        rows = []
        for label, positional in (('dPhysChem_pos', True), ('dPhysChem', False)):
            SDx, tx, hx, _hx = feature_sqdiff(ctx, positional=positional)
            rows.append(dict(rep=label, metric='euclidean',
                             struct_blosum_ctrl=score_subset(
                                 SDx, range(SDx.shape[0]), tx, hx)))
        return rows
    S, wt, muts = ctx['S'], ctx['wt'], ctx['muts']
    I, J = ctx['pairs']
    gmat, bmat = SR.grantham_matrix(), SR.blosum_matrix()
    reps = {
        'MutID': SR.rep_mutid(S, wt, muts),
        'dPhysChem': SR.rep_dphys(S, wt, muts),
        'dPhysChem_pos': SR.rep_dphys_positional(S, wt, muts),
        'SubScore': SR.rep_subscore(S, wt, muts, gmat, bmat),
        'BaseDelta': SR.rep_basedelta(S, wt, muts),
    }
    ham = ctx['truth']['identity']
    rows = []
    for name, X in reps.items():
        for metric in ('cosine', 'euclidean'):
            if metric == 'euclidean':
                d = np.linalg.norm(X[I] - X[J], axis=1)
            else:
                a, b = X[I], X[J]
                na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
                d = 1.0 - np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)
            r = dict(rep=name, metric=metric)
            basis = REP_BASIS[name]
            cleans = []
            for g in GRADERS:
                raw = spearmanr(d, ctx['truth'][g])[0]
                r[f'{g}_raw'] = float(0 if not np.isfinite(raw) else raw)
                clean = not conflicts(g, basis)
                r[f'{g}_clean'] = clean
                if g != 'identity':
                    c = partial_spearman(d, ctx['truth'][g], ham)
                    r[f'{g}_ctrl'] = float(0 if not np.isfinite(c) else c)
                    if clean:
                        cleans.append(r[f'{g}_ctrl'])
            r['struct_ctrl_best'] = float(max(cleans)) if cleans else float('nan')
            rows.append(r)
    return rows


# ── MODE 2: features ─────────────────────────────────────────────────────────
def mode_features(ctx):
    """What kind of feature improves a distance-based similarity space?

    Every subset here is CHOSEN on half 0 of the pairs and SCORED on half 1.
    That covers greedy_forward's subset, the top-k ordering (which comes from
    the per-feature solo scores), and the truncation curve's peak — all three
    are choices made by looking at pair scores. all_features_score and
    raw_units_score choose nothing, and are reported on half 1 too so the four
    numbers sit on the same pairs.

    Before 2026-09-04 all of them were computed on one 8000-pair sample, so
    greedy's score was tuned to the pairs it was reported on while top-k and
    all-14 were not — the greedy-vs-top-k comparison was between a fitted
    number and two unfitted ones.
    """
    SD, truth, ham, half = feature_sqdiff(ctx)
    k = SD.shape[0]
    fit, rep = (half == 0), (half == 1)
    n_fit, n_rep = int(fit.sum()), int(rep.sum())
    SDf, tf, hf = SD[:, fit], truth[fit], ham[fit]
    SDr_, tr_, hr_ = SD[:, rep], truth[rep], ham[rep]

    # Ordering and the greedy subset come from the choosing half.
    solo_fit = np.array([score_subset(SDf, [j], tf, hf) for j in range(k)])
    order = np.argsort(-solo_fit)
    R = np.corrcoef(SDf)                      # redundancy in DISTANCE terms
    chosen, greedy_fit_v, _ = greedy_forward(SDf, tf, hf)

    # Every reported score comes from the other half.
    solo = np.array([score_subset(SDr_, [j], tr_, hr_) for j in range(k)])
    greedy_v = score_subset(SDr_, chosen, tr_, hr_)
    topk = [int(x) for x in order[:max(len(chosen), 1)]]
    topk_v = score_subset(SDr_, topk, tr_, hr_)
    all_v = score_subset(SDr_, list(range(k)), tr_, hr_)
    curve_fit = [(m, score_subset(SDf, [int(x) for x in order[:m]], tf, hf))
                 for m in range(1, k + 1)]
    curve = [(m, score_subset(SDr_, [int(x) for x in order[:m]], tr_, hr_))
             for m in range(1, k + 1)]
    peak_m = max(curve_fit, key=lambda t: t[1])[0]
    peak_v = dict(curve)[peak_m]
    SDraw, traw, hraw = feature_sqdiff(ctx, zscore=False)[:3]
    raw_v = score_subset(SDraw[:, rep], list(range(k)), traw[rep], hraw[rep])

    # Diagnostics about the greedy search itself, so they stay on its own half.
    excluded = []
    for j in range(k):
        if j in chosen:
            continue
        excluded.append(dict(
            feature=NAMES[j], solo=float(solo[j]),
            max_corr_with_chosen=float(max((R[j, c] for c in chosen), default=0)),
            delta_if_added=float(score_subset(SDf, chosen + [j], tf, hf)
                                 - greedy_fit_v)))
    return dict(n_pairs_choosing_half=n_fit, n_pairs_reporting_half=n_rep,
                solo={NAMES[j]: float(solo[j]) for j in range(k)},
                greedy=[NAMES[j] for j in chosen], greedy_score=greedy_v,
                greedy_score_on_choosing_half=greedy_fit_v,
                topk=[NAMES[j] for j in topk], topk_score=topk_v,
                all_features_score=all_v, raw_units_score=raw_v,
                peak_n=peak_m, peak_score=peak_v, curve=curve,
                excluded=sorted(excluded, key=lambda e: -e['solo']))


# ── MODE 3: controls ─────────────────────────────────────────────────────────
def mode_controls(ctx):
    """C1/C2/C3 diagnostics — the numbers that say whether scores are honest."""
    out = {}
    if ctx['kind'] == 'single_scaffold':
        t = ctx['truth']
        out['C2_count_contamination'] = {
            g: float(spearmanr(t['identity'], t[g])[0]) for g in ('grantham', 'blosum')}
        out['C2_note'] = ('correlation of the plain differing-position count with '
                          'each rich ground truth; high = the ground truth is '
                          'mostly counting, so raw scores are inflated')
        # C3: is the bespoke metric identical to a ground truth?
        # dist_position_gated reads only each variant's first mutation, so it
        # is defined only when no variant carries more than one. A mean of 1.0
        # does not guarantee that; the maximum does.
        if ctx.get('max_mutations', 99) <= 1:
            D = SR.dist_position_gated(ctx['S'], ctx['wt'], ctx['muts'],
                                       SR.grantham_matrix())
            I, J = ctx['pairs']
            max_diff = float(np.abs(D[I, J] - t['grantham']).max())
            out['C3_position_gated_vs_grantham'] = dict(
                spearman=float(spearmanr(D[I, J], t['grantham'])[0]),
                max_abs_difference=max_diff,
                verdict=('IDENTICAL — not a metric result' if max_diff < 1e-9
                         else 'DIFFERS from the Grantham grader'))
        out['mean_mutations_per_variant'] = ctx['mean_mutations']
    else:
        _SD, truth, ham, _half = feature_sqdiff(ctx)
        out['C2_count_contamination'] = {
            'blosum': float(spearmanr(ham, truth)[0])}
        out['mean_mutations_per_variant'] = 1.0
    return out


# ── Reporting ────────────────────────────────────────────────────────────────
def print_analysis(results):
    print('\n' + '=' * 78)
    print('SCAFFOLD-DIFF SIMILARITY — SUMMARY')
    print('Spearman rank correlation over variant pairs, -1..1, higher better.')
    print('Count-controlled and basis-independent throughout. No model trained.')
    print('=' * 78)

    if any('representations' in r for r in results.values()):
        print('\n## 1. REPRESENTATIONS (best clean, count-controlled score)')
        print(f'  {"dataset":26s} {"muts/var":>8}  {"best rep x metric":34s} {"score":>7}')
        for ds, r in results.items():
            rows = r.get('representations')
            if not rows:
                continue
            key = 'struct_ctrl_best' if 'struct_ctrl_best' in rows[0] \
                else 'struct_blosum_ctrl'
            best = max((x for x in rows if np.isfinite(x.get(key, np.nan))),
                       key=lambda x: x[key], default=None)
            if best:
                print(f'  {ds:26s} {r["mean_mutations"]:8.2f}  '
                      f'{best["rep"]+" x "+best["metric"]:34s} {best[key]:7.3f}')

    if any('features' in r for r in results.values()):
        print('\n## 2. FEATURES — does signal or non-redundancy decide?')
        print('  Subsets chosen on one half of the variants, scored on the '
              'other. "fit" is greedy on its own half, for contrast.')
        print(f'  {"dataset":26s} {"all 14":>7} {"top-k solo":>11} {"greedy":>7} '
              f'{"greedy fit":>10} {"raw units":>10}  selected')
        for ds, r in results.items():
            f = r.get('features')
            if not f:
                continue
            print(f'  {ds:26s} {f["all_features_score"]:7.3f} '
                  f'{f["topk_score"]:11.3f} {f["greedy_score"]:7.3f} '
                  f'{f["greedy_score_on_choosing_half"]:10.3f} '
                  f'{f["raw_units_score"]:10.3f}  {", ".join(f["greedy"])}')
        print('\n  High-signal features REJECTED for redundancy '
              '(solo score good, but duplicates a chosen axis):')
        for ds, r in results.items():
            f = r.get('features')
            if not f:
                continue
            for e in f['excluded'][:3]:
                if e['solo'] > 0.25:
                    print(f'    {ds:26s} {e["feature"]:14s} solo={e["solo"]:.3f} '
                          f'corr={e["max_corr_with_chosen"]:.2f} '
                          f'delta={e["delta_if_added"]:+.3f}')

    if any('controls' in r for r in results.values()):
        print('\n## 3. CONTROLS')
        print(f'  {"dataset":26s} {"muts/var":>8} {"count contamination":>20}  verdict')
        for ds, r in results.items():
            c = r.get('controls')
            if not c:
                continue
            cc = c['C2_count_contamination'].get('blosum', float('nan'))
            verdict = 'CLEAN' if cc < 0.4 else 'CONTAMINATED — raw scores inflated'
            print(f'  {ds:26s} {c["mean_mutations_per_variant"]:8.2f} '
                  f'{cc:20.3f}  {verdict}')
            if 'C3_position_gated_vs_grantham' in c:
                g = c['C3_position_gated_vs_grantham']
                print(f'    C3: position_gated vs Grantham ground truth — '
                      f'spearman {g["spearman"]:.6f}, max diff '
                      f'{g["max_abs_difference"]:.1e} => {g["verdict"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='all',
                    choices=['representations', 'features', 'controls', 'all'])
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='default: every entry in SCAFFOLD_DATASETS')
    ap.add_argument('--smoke', action='store_true',
                    help='2 small datasets, fewer pairs')
    ap.add_argument('--n-pairs', type=int, default=32000,
                    help='Pairs kept per dataset, after dropping the ones '
                         'whose two variants are in different halves. Each '
                         'half ends with about half of this number.')
    ap.add_argument('--output', default=None)
    ap.add_argument('--force-rerun', action='store_true',
                    help='Discard the (dataset, mode) results an earlier '
                         'attempt finished (<output>.partial.jsonl).')
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()

    apply_distance_metric_arg(args)
    datasets = args.datasets or list(SCAFFOLD_DATASETS)
    n_pairs = args.n_pairs
    if args.smoke:
        datasets = ['flab_expression_koenig', 'flab_expression_adams']
        n_pairs = 8000
    modes = (['representations', 'features', 'controls']
             if args.mode == 'all' else [args.mode])

    print(f'Scaffold-diff similarity experiment | modes={modes} | '
          f'datasets={len(datasets)} | pairs={n_pairs}')
    print(f'Excluded datasets: '
          + '; '.join(f'{k} ({v})' for k, v in EXCLUDED_DATASETS.items()))

    out = args.output or os.path.join(HERE, 'results',
                                      'opig_scaffold_similarity.json')

    # Each (dataset, mode) result is written to <output>.partial.jsonl as soon
    # as it finishes and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). The pairs come from a fixed seed, so a
    # result read back is what a fresh computation gives. When every mode of a
    # dataset is stored the dataset is not loaded again. A mode that raised, or
    # a dataset that failed to load, is not stored and makes the script exit 1
    # after the JSON is written. tests/experiments is appended to sys.path, not
    # prepended: its package names (similarity, noise, ...) must not shadow
    # the similarity_metrics modules imported above.
    if HERE not in sys.path:
        sys.path.append(HERE)
    from _checkpoint import script_checkpoint, exit_if_failed
    _ck = script_checkpoint('opig_scaffold_similarity', out, args,
                            exclude=('datasets', 'mode', 'output'),
                            n_pairs_used=n_pairs)
    _failed = []

    results, t0 = {}, time.time()
    for ds in datasets:
        if ds not in SCAFFOLD_DATASETS:
            print(f'  SKIP {ds}: not a scaffold-compatible dataset')
            continue
        if modes and all(_ck.done(dataset=ds, mode=m) for m in modes):
            _saved = [_ck.get(dataset=ds, mode=m) for m in modes]
            r = dict(_saved[0]['base'])
            for m, sv in zip(modes, _saved):
                r[m] = sv['result']
            results[ds] = r
            print(f'  {ds:26s} finished in an earlier attempt, read back')
            continue
        try:
            ctx = load_dataset(ds, n_pairs=n_pairs)
        except Exception as e:
            print(f'  FAILED to load {ds}: {type(e).__name__}: {e}')
            _failed.append(ds)
            continue
        r = {'mean_mutations': ctx['mean_mutations'], 'n': ctx['n'],
             'n_rows_loaded': ctx.get('n_rows_loaded', ctx['n']),
             'kind': ctx['kind'], 'domain': SCAFFOLD_DATASETS[ds]['domain']}
        _base = dict(r)
        for m in modes:
            try:
                if _ck.done(dataset=ds, mode=m):
                    r[m] = _ck.get(dataset=ds, mode=m)['result']
                    continue
                r[m] = {'representations': mode_representations,
                        'features': mode_features,
                        'controls': mode_controls}[m](ctx)
                _ck.put({'base': _base, 'result': r[m]}, dataset=ds, mode=m)
            except Exception as e:
                print(f'  {ds}/{m} FAILED: {type(e).__name__}: {e}')
                _failed.append(f'{ds}/{m}')
        results[ds] = r
        print(f'  {ds:26s} done ({time.time()-t0:.0f}s)', flush=True)

    print_analysis(results)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=conv)
    print(f'\nwrote {out}')
    exit_if_failed(_failed, 'dataset or mode')


if __name__ == '__main__':
    main()
