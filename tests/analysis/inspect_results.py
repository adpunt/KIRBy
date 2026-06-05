"""Lightweight inspector for the 13 completed antibody result JSONs.

Walks tests/results/{complexity,similarity}/<exp>/<dataset>.json for the
13-dataset analysis batch (2026-05-13) and dumps a structured per-experiment
summary. Includes per-experiment summarizers for the headline experiments;
falls back to a generic schema dump for the rest.

Usage:
    python tests/analysis/inspect_results.py c01_rep_characterization
    python tests/analysis/inspect_results.py --list
    python tests/analysis/inspect_results.py --all-coverage
    python tests/analysis/inspect_results.py --all      # run every summarizer
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / 'tests' / 'results'

DATASETS = [
    'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2', 'abagym_VEGF', 'abcov_kd',
    'flab_aggregation_jain', 'flab_immunogenicity_marks', 'flab_pk_jain_heprt',
    'flab_polyreact_jain_bvp', 'flab_polyreact_jain_cic',
    'ginkgo_heparin', 'humanness_therapeutic', 'tz_her2_binary',
]
REGRESSION = {
    'abagym_Ang2', 'abagym_EGFR', 'abagym_HER2', 'abagym_VEGF',
    'abcov_kd', 'flab_aggregation_jain', 'flab_pk_jain_heprt', 'ginkgo_heparin',
}
CLASSIFICATION = {
    'flab_immunogenicity_marks', 'flab_polyreact_jain_bvp',
    'flab_polyreact_jain_cic', 'humanness_therapeutic', 'tz_her2_binary',
}


def _load(cat, exp, ds):
    p = RESULTS / cat / exp / f'{ds}.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {'__load_error__': str(e)}


def _cat(exp):
    return 'complexity' if exp.startswith('c') else 'similarity'


# ---------------------------------------------------------------------------
# Summarizers
# ---------------------------------------------------------------------------

def summarize_c01(data, ds):
    """rep_characterization: top rep + its best baseline."""
    rows = []
    for rep, entry in data.items():
        if rep == 'timestamp' or not isinstance(entry, dict):
            continue
        baselines = entry.get('baselines', {})
        best = None
        for model, m in baselines.items():
            if isinstance(m, dict) and 'score_mean' in m:
                if best is None or m['score_mean'] > best[1]:
                    best = (model, m['score_mean'])
        if best:
            rows.append((rep, best[0], best[1]))
    rows.sort(key=lambda r: r[2], reverse=True)
    return {
        'n_reps': len(rows),
        'best': (rows[0][0], rows[0][1], round(rows[0][2], 3)) if rows else None,
        'top3': [(r[0], r[1], round(r[2], 3)) for r in rows[:3]],
        'range': (round(min(r[2] for r in rows), 3), round(max(r[2] for r in rows), 3))
                 if rows else None,
    }


def summarize_c02(data, ds):
    """Matching grid — headline KIRBy result. Returns best baseline + best
    hybrid per condition, with delta."""
    # Baseline: best single rep across all models
    best_baseline = max(
        (m.get('best_single_score', -9) for m in data.get('baselines', {}).values()),
        default=None)
    # Hybrid per condition: best across all models
    cond_best = {}
    for cond, models in (data.get('conditions') or {}).items():
        scores = [m.get('score_mean') for m in models.values()
                  if isinstance(m, dict) and m.get('score_mean') is not None]
        if scores:
            cond_best[cond] = round(max(scores), 3)
    # PCA baseline best
    pca_best = None
    for dim, models in (data.get('pca_baselines') or {}).items():
        sc = max(models.values()) if isinstance(models, dict) else None
        if sc is not None and (pca_best is None or sc > pca_best):
            pca_best = sc
    return {
        'baseline': round(best_baseline, 3) if best_baseline is not None else None,
        'pca_best': round(pca_best, 3) if pca_best is not None else None,
        'hybrid_by_cond': cond_best,
        'hybrid_best_overall': round(max(cond_best.values()), 3) if cond_best else None,
        'delta_vs_baseline': (round(max(cond_best.values()) - best_baseline, 3)
                              if cond_best and best_baseline is not None else None),
        'has_aug': data.get('has_augmentation'),
    }


def summarize_c04(data, ds):
    """importance_as_metric — how well does importance score predict rep utility?"""
    if not isinstance(data, dict):
        return {'type': type(data).__name__}
    # Likely structure: per-importance-method correlations with downstream score
    keys = list(data.keys())
    correlations = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, (int, float)) and 'corr' in subk.lower():
                    correlations[f'{k}.{subk}'] = round(subv, 3)
                elif isinstance(subv, (int, float)) and ('spearman' in subk.lower()
                                                          or 'pearson' in subk.lower()):
                    correlations[f'{k}.{subk}'] = round(subv, 3)
    return {'top_keys': keys[:10], 'correlations_found': correlations}


def summarize_c11(data, ds):
    """Hybrid vs best_single vs pca_matched."""
    results = data.get('results', [])
    methods = {}
    for r in results:
        method = r.get('method')
        score = r.get('score_mean')
        if method and score is not None:
            if method not in methods or score > methods[method][0]:
                methods[method] = (score, r.get('model'), r.get('budget') or r.get('total_dim'))
    out = {m: (round(v[0], 3), v[1], v[2]) for m, v in methods.items()}
    if 'hybrid' in methods and 'best_single' in methods:
        out['hybrid_beats_best_single'] = methods['hybrid'][0] > methods['best_single'][0]
        out['delta'] = round(methods['hybrid'][0] - methods['best_single'][0], 3)
    return out


def summarize_c05(data, ds):
    """learning_curves: report n_points × n_reps and ranking-stability metric if present."""
    keys = list(data.keys()) if isinstance(data, dict) else []
    # Look for ranking-stability or kendall tau across sizes
    return {'top_keys': keys[:15]}


def summarize_c10(data, ds):
    """noise_robustness: how does hybrid vs baseline change as noise increases?"""
    keys = list(data.keys()) if isinstance(data, dict) else []
    return {'top_keys': keys[:15]}


def summarize_s01(data, ds):
    """similarity_baseline: best similarity rep + its score."""
    if not isinstance(data, dict):
        return {'type': type(data).__name__}
    # Try common shapes
    results = data.get('results') or data.get('similarity_baseline') or data
    rows = []
    if isinstance(results, dict):
        for rep, m in results.items():
            if rep in ('timestamp', 'meta'):
                continue
            if isinstance(m, dict):
                # find a spearman/pearson-like number
                for k, v in m.items():
                    if isinstance(v, (int, float)) and ('sac_rho' in k or 'spearman' in k
                                                        or 'rho' in k):
                        rows.append((rep, k, v))
                        break
    rows.sort(key=lambda r: r[2], reverse=True)
    return {'n_reps_with_signal': len(rows),
            'top3': [(r[0], r[1], round(r[2], 3)) for r in rows[:3]],
            'top_keys': list(data.keys())[:8]}


def summarize_s02(data, ds):
    """hybrid_similarity: does hybrid beat best single similarity rep?"""
    return summarize_generic(data, ds)


def summarize_generic(data, ds):
    if not isinstance(data, dict):
        return {'type': type(data).__name__,
                'len': len(data) if hasattr(data, '__len__') else None}
    keys = list(data.keys())
    return {'top_keys': keys[:12], 'n_keys': len(keys)}


SUMMARIZERS = {
    'c01_rep_characterization': summarize_c01,
    'c02_complexity_matching_grid': summarize_c02,
    'c04_importance_as_metric': summarize_c04,
    'c05_learning_curves': summarize_c05,
    'c10_noise_robustness': summarize_c10,
    'c11_ensemble_vs_hybrid': summarize_c11,
    's01_similarity_baseline': summarize_s01,
    's02_hybrid_similarity': summarize_s02,
}


def coverage_table():
    for cat in ('complexity', 'similarity'):
        print(f'=== {cat} ===')
        for d in sorted((RESULTS / cat).iterdir()):
            if not d.is_dir():
                continue
            present = [ds for ds in DATASETS if (d / f'{ds}.json').exists()]
            missing = [ds for ds in DATASETS if ds not in present]
            print(f'  {d.name:<48} {len(present):2d}/13  missing={missing}')


def inspect_experiment(exp, quiet=False):
    cat = _cat(exp)
    summarizer = SUMMARIZERS.get(exp, summarize_generic)
    if not quiet:
        print(f'\n=== {cat}/{exp} ===')
    findings = {}
    for ds in DATASETS:
        data = _load(cat, exp, ds)
        if data is None:
            continue
        if isinstance(data, dict) and '__load_error__' in data:
            if not quiet:
                print(f'  {ds:<35} LOAD ERR: {data["__load_error__"]}')
            findings[ds] = {'error': data['__load_error__']}
            continue
        try:
            summary = summarizer(data, ds)
            findings[ds] = summary
            if not quiet:
                tag = 'reg' if ds in REGRESSION else 'clf'
                print(f'  {ds:<35} [{tag}] {summary}')
        except Exception as e:
            if not quiet:
                print(f'  {ds:<35} SUMMARY ERR: {e}')
            findings[ds] = {'error': str(e)}
    return findings


def all_experiments():
    exps = []
    for cat in ('complexity', 'similarity'):
        for d in sorted((RESULTS / cat).iterdir()):
            if d.is_dir():
                exps.append(d.name)
    return exps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', nargs='?')
    p.add_argument('--list', action='store_true')
    p.add_argument('--all-coverage', action='store_true')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    if args.all_coverage:
        coverage_table()
        return
    if args.list:
        for e in all_experiments():
            print(e)
        return
    if args.all:
        for e in all_experiments():
            inspect_experiment(e)
        return
    if not args.experiment:
        coverage_table()
        return
    inspect_experiment(args.experiment)


if __name__ == '__main__':
    main()
