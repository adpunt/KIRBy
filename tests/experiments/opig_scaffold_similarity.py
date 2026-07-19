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
     (--mode features)
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
    'flab_binding_phillips_h3': dict(kind='single_scaffold', domain='antibody'),
    'graphinity_foldx':         dict(kind='multi_scaffold', domain='protein_ddg'),
}
# Excluded, with the reason recorded so it is not silently dropped:
EXCLUDED_DATASETS = {
    'flab_thermo_hie':     'n=10 — far too small',
    'flab_binding_rosace': 'n=14 — far too small',
    'ddg_combined':        'multi-point mutations + multi-PDB; needs the '
                           'multi-mutation generalisation, not yet built',
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
    n = len(S)
    rs = np.random.RandomState(seed)
    I = rs.randint(0, n, n_pairs * 2); J = rs.randint(0, n, n_pairs * 2)
    m = I != J; I, J = I[m][:n_pairs], J[m][:n_pairs]
    truth = {'grantham': SR.grader_pair_distance(S, I, J, SR.grantham_matrix()),
             'blosum': SR.grader_pair_distance(S, I, J, SR.blosum_matrix()),
             'identity': SR.grader_pair_distance(S, I, J, SR.identity_matrix())}
    return dict(name=name, kind='single_scaffold', S=S, wt=wt, muts=muts, n=n,
                pairs=(I, J), truth=truth,
                mean_mutations=float(np.mean([len(x) for x in muts])),
                labels=np.asarray(d['labels'], float),
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


def load_dataset(name, n_pairs=8000, seed=0, max_samples=5000):
    kind = SCAFFOLD_DATASETS[name]['kind']
    if kind == 'single_scaffold':
        return load_single_scaffold(name, n_pairs, seed, max_samples)
    return load_multi_scaffold(name, n_pairs, seed, max_samples if
                               max_samples > 5000 else 60000)


# ── Feature-space construction (shared by all modes) ─────────────────────────
def feature_sqdiff(ctx, zscore=True, positional=True):
    """Per-feature squared pair-differences (n_features, n_pairs), plus the
    BLOSUM ground truth and the count proxy.

    Euclidean distance over a feature subset is sqrt of the summed rows, so any
    subset can be scored without rebuilding the space — this is what makes the
    exhaustive and greedy searches in --mode features affordable.
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
        return SD, ctx['truth']['blosum'], ctx['truth']['identity']

    # multi-scaffold: build per complex, pool pairs
    pos, wta, mta, pdb = ctx['pos'], ctx['wta'], ctx['mta'], ctx['pdb']
    bmat, imat = SR.blosum_matrix(), SR.identity_matrix()
    rs = np.random.RandomState(ctx['seed'])
    per = ctx['n_pairs_per_complex']
    sd_chunks, t_chunks, h_chunks = [], [], []
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
        I = rs.randint(0, nn, per * 2); J = rs.randint(0, nn, per * 2)
        keep = I != J; I, J = I[keep][:per], J[keep][:per]
        if len(I) < 10:
            continue
        chunk = np.empty((k, len(I)), np.float64)
        for j in range(k):
            dd = X[I, :, j] - X[J, :, j]
            chunk[j] = np.sum(dd * dd, axis=1)
        sd_chunks.append(chunk)
        same = p[I] == p[J]
        for M, out in ((bmat, t_chunks), (imat, h_chunks)):
            out.append(np.where(same, M[m[I], m[J]],
                                M[w[I], m[I]] + M[w[J], m[J]]))
    return (np.concatenate(sd_chunks, axis=1),
            np.concatenate(t_chunks), np.concatenate(h_chunks))


# ── MODE 1: representations ──────────────────────────────────────────────────
def mode_representations(ctx):
    """Which scaffold-diff representation × distance metric wins, per C1+C2."""
    if ctx['kind'] != 'single_scaffold':
        SD, truth, ham = feature_sqdiff(ctx)
        rows = []
        for label, positional in (('dPhysChem_pos', True), ('dPhysChem', False)):
            SDx, tx, hx = feature_sqdiff(ctx, positional=positional)
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
    """What kind of feature improves a distance-based similarity space?"""
    SD, truth, ham = feature_sqdiff(ctx)
    k = SD.shape[0]
    solo = np.array([score_subset(SD, [j], truth, ham) for j in range(k)])
    order = np.argsort(-solo)
    R = np.corrcoef(SD)                       # redundancy in DISTANCE terms
    chosen, greedy_v, _ = greedy_forward(SD, truth, ham)
    topk = [int(x) for x in order[:max(len(chosen), 1)]]
    topk_v = score_subset(SD, topk, truth, ham)
    all_v = score_subset(SD, list(range(k)), truth, ham)
    curve = [(m, score_subset(SD, [int(x) for x in order[:m]], truth, ham))
             for m in range(1, k + 1)]
    peak_m, peak_v = max(curve, key=lambda t: t[1])
    SDr, tr, hr = feature_sqdiff(ctx, zscore=False)
    raw_v = score_subset(SDr, list(range(k)), tr, hr)
    excluded = []
    for j in range(k):
        if j in chosen:
            continue
        excluded.append(dict(
            feature=NAMES[j], solo=float(solo[j]),
            max_corr_with_chosen=float(max((R[j, c] for c in chosen), default=0)),
            delta_if_added=float(score_subset(SD, chosen + [j], truth, ham)
                                 - greedy_v)))
    return dict(solo={NAMES[j]: float(solo[j]) for j in range(k)},
                greedy=[NAMES[j] for j in chosen], greedy_score=greedy_v,
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
        if ctx['mean_mutations'] <= 1.001:
            D = SR.dist_position_gated(ctx['S'], ctx['wt'], ctx['muts'],
                                       SR.grantham_matrix())
            I, J = ctx['pairs']
            out['C3_position_gated_vs_grantham'] = dict(
                spearman=float(spearmanr(D[I, J], t['grantham'])[0]),
                max_abs_difference=float(np.abs(D[I, J] - t['grantham']).max()),
                verdict='IDENTICAL — not a metric result')
        out['mean_mutations_per_variant'] = ctx['mean_mutations']
    else:
        _SD, truth, ham = feature_sqdiff(ctx)
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
        print(f'  {"dataset":26s} {"all 14":>7} {"top-k solo":>11} {"greedy":>7} '
              f'{"raw units":>10}  selected')
        for ds, r in results.items():
            f = r.get('features')
            if not f:
                continue
            print(f'  {ds:26s} {f["all_features_score"]:7.3f} '
                  f'{f["topk_score"]:11.3f} {f["greedy_score"]:7.3f} '
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
    ap.add_argument('--n-pairs', type=int, default=8000)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    datasets = args.datasets or list(SCAFFOLD_DATASETS)
    n_pairs = args.n_pairs
    if args.smoke:
        datasets = ['flab_expression_koenig', 'flab_expression_adams']
        n_pairs = 2000
    modes = (['representations', 'features', 'controls']
             if args.mode == 'all' else [args.mode])

    print(f'Scaffold-diff similarity experiment | modes={modes} | '
          f'datasets={len(datasets)} | pairs={n_pairs}')
    print(f'Excluded datasets: '
          + '; '.join(f'{k} ({v})' for k, v in EXCLUDED_DATASETS.items()))

    results, t0 = {}, time.time()
    for ds in datasets:
        if ds not in SCAFFOLD_DATASETS:
            print(f'  SKIP {ds}: not a scaffold-compatible dataset')
            continue
        try:
            ctx = load_dataset(ds, n_pairs=n_pairs)
        except Exception as e:
            print(f'  FAILED to load {ds}: {type(e).__name__}: {e}')
            continue
        r = {'mean_mutations': ctx['mean_mutations'], 'n': ctx['n'],
             'kind': ctx['kind'], 'domain': SCAFFOLD_DATASETS[ds]['domain']}
        for m in modes:
            try:
                r[m] = {'representations': mode_representations,
                        'features': mode_features,
                        'controls': mode_controls}[m](ctx)
            except Exception as e:
                print(f'  {ds}/{m} FAILED: {type(e).__name__}: {e}')
        results[ds] = r
        print(f'  {ds:26s} done ({time.time()-t0:.0f}s)', flush=True)

    print_analysis(results)
    out = args.output or os.path.join(HERE, 'results',
                                      'opig_scaffold_similarity.json')
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


if __name__ == '__main__':
    main()
