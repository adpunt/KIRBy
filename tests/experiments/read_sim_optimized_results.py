#!/usr/bin/env python3
"""Compact reader for similarity_optimized_hybrid.py streaming JSON output.

Usage:
    python tests/experiments/read_sim_optimized_results.py [results_dir]

Default results_dir = tests/results/sim_optimized_hybrid. Handles PARTIAL files
(jobs still running or killed early) — anything not yet computed shows 'nan'.
"""
import glob
import json
import os
import sys


def m(x):
    return x.get('mean') if isinstance(x, dict) else None


def f(v):
    return '   nan' if v is None else '{:+.3f}'.format(v)


def cmean(node, keys, sp, key):
    vals = []
    for k in keys:
        cell = node.get(k, {}).get(sp, {})
        v = m(cell.get(key)) if isinstance(cell, dict) else None
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _divergence(d):
    """Did the similarity search actually pick a DIFFERENT blueprint than the
    predictive search? If the two agree on every seed, every downstream payoff
    is comparing a space against itself and no effect can exist by construction.
    This is the first thing to check before interpreting any payoff below."""
    picks = d.get('picks') or []
    if not picks:
        print('  blueprint picks: (none recorded)')
        return
    same = 0
    for p in picks:
        sim = (p.get('sim_alloc'), p.get('sim_budget'))
        pred = (p.get('pred_alloc'), p.get('pred_budget'))
        if sim == pred:
            same += 1
        print('  seed {:<3} sim=({}, b={})  pred=({}, b={})  best_rep={}{}'.format(
            p.get('seed'), p.get('sim_alloc'), p.get('sim_budget'),
            p.get('pred_alloc'), p.get('pred_budget'), p.get('best_rep'),
            '   <-- IDENTICAL' if sim == pred else ''))
    if same == len(picks):
        print('  ** WARNING: sim and pred blueprints IDENTICAL on all {} seed(s) — '
              'the A/B is degenerate; payoffs below cannot show an effect. **'
              .format(len(picks)))
    elif same:
        print('  ** NOTE: identical blueprint on {}/{} seeds. **'.format(same, len(picks)))


def summarize(fp):
    d = json.load(open(fp))
    print('=' * 74)
    status = ('PARTIAL @' + str(d.get('progress'))) if d.get('partial') else 'COMPLETE'
    print('{}  n={}  {}  seeds {}/{}  {}'.format(
        d.get('dataset'), d.get('n_samples'), d.get('task_type'),
        d.get('seeds_completed'), d.get('seeds_requested'), status))
    _divergence(d)

    p1 = d.get('payoff1_heldout_similarity', {})
    sac = p1.get('sac', {})
    vio = p1.get('violation', {})
    # SAC = distance-vs-property correlation (higher=better; 0=meaningless map).
    print('  P1 distance-property corr   sim={}  pred={}  rep={}'.format(
        f(m(sac.get('sim', {}))), f(m(sac.get('pred', {}))), f(m(sac.get('rep', {})))))
    # Violation = fraction of a molecule's nearest neighbours whose property is
    # very different ("neighbours that lie"). LOWER is better.
    print('  P1 neighbour-lie rate (lo=better)  sim={}  pred={}  rep={}'.format(
        f(m(vio.get('sim', {}))), f(m(vio.get('pred', {}))), f(m(vio.get('rep', {})))))

    p2 = d.get('payoff2_noise', {})
    if 'skipped' in p2:
        print('  P2 noise: skipped (classification)')
        return
    for nt in p2:
        methods = [k for k in p2[nt] if k != '_global_smoothing']
        g = p2[nt].get('_global_smoothing', {})
        print('  [{}] global-smooth  sim={} pred={} rep={}'.format(
            nt, f(m(g.get('sim', {}))), f(m(g.get('pred', {}))), f(m(g.get('rep', {})))))
        for cls in ('similarity', 'geometry', 'control'):
            ks = [k for k in methods if p2[nt][k].get('class') == cls]
            if not ks:
                continue
            print('     {:10s} remove sim={} pred={} | smooth sim={} pred={} | detF1 sim={} pred={}'.format(
                cls,
                f(cmean(p2[nt], ks, 'sim', 'recovery_remove')),
                f(cmean(p2[nt], ks, 'pred', 'recovery_remove')),
                f(cmean(p2[nt], ks, 'sim', 'recovery_smooth')),
                f(cmean(p2[nt], ks, 'pred', 'recovery_smooth')),
                f(cmean(p2[nt], ks, 'sim', 'detection_f1')),
                f(cmean(p2[nt], ks, 'pred', 'detection_f1'))))

    pe = d.get('payoff2_estimation', {})
    if isinstance(pe, dict) and 'skipped' not in pe:
        for nt in pe:
            es = list(pe[nt].keys())
            # AUROC = how well noise-scores rank the truly-corrupted samples.
            print('  [{}] noise-detect AUROC   sim={} pred={} rep={}'.format(
                nt, f(cmean(pe[nt], es, 'sim', 'auroc')),
                f(cmean(pe[nt], es, 'pred', 'auroc')),
                f(cmean(pe[nt], es, 'rep', 'auroc'))))
            # precision@k = of the k most-suspicious labels (k = #injected), the
            # fraction that are ACTUALLY corrupted. Intuitive "if we flag the top
            # 25%, how many are real errors".
            print('       precision@k       sim={} pred={} rep={}'.format(
                f(cmean(pe[nt], es, 'sim', 'precision_at_k')),
                f(cmean(pe[nt], es, 'pred', 'precision_at_k')),
                f(cmean(pe[nt], es, 'rep', 'precision_at_k'))))

    p3 = d.get('payoff3_importance_weighted', {})
    if isinstance(p3, dict) and 'skipped' not in p3:
        fashions = p3.get('fashions', [])
        for mth in [k for k in p3 if k not in ('noise_type', 'fashions')]:
            cells = []
            for fa in fashions:
                cell = p3[mth].get(fa, {}).get('sim', {})
                cells.append('{}={}'.format(fa, f(m(cell.get('detection_f1')))))
            print('  wgt {:20s} sim detF1: {}'.format(mth, ' '.join(cells)))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'tests/results/sim_optimized_hybrid'
    files = sorted(fp for fp in glob.glob(os.path.join(d, '*.json'))
                   if os.path.basename(fp) != 'summary.json')
    if not files:
        print('No result JSON files in {}'.format(d))
        return
    for fp in files:
        try:
            summarize(fp)
        except Exception as e:
            print('{}: could not read ({})'.format(os.path.basename(fp), e))


if __name__ == '__main__':
    main()
