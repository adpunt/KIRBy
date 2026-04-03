#!/usr/bin/env python3
"""Generate supervisor meeting presentation from hybrid search results.

Run ON THE SERVER:
    cd /data/stat-cadd/scat9264/KIRBy
    python tests/supervisor_presentation.py | tee presentation_output.txt

Data sources (in priority order):
  1. Pickle caches: ~/.cache/kirby/hybrid_search/*.pkl (SearchResult objects)
  2. Experiment JSONs: tests/results/complexity/c02_*/*.json (hybrid configs)
     + tests/results/phase5_baselines/*/*.json (per-rep baseline scores)

For each dataset shows:
  - Dataset info (domain, metric, splitting, general/top reps)
  - All base rep scores sorted by RF (no +/- values)
  - Hybrid heatmap (model x budget)
  - Top hybrid config per model (allocation, importance, budget)
  - Number of configs tested and number exceeding best baseline
"""
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# ── Path setup ──
ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / 'results'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

CACHE_DIR = Path.home() / '.cache' / 'kirby' / 'hybrid_search'

try:
    from experiments.complexity.complexity_helpers import DATASET_REPS
except ImportError:
    DATASET_REPS = {}

SPLITTING = {
    'molecular': 'Scaffold GroupKFold',
    'antibody': 'Identity/Position-modulo GroupKFold',
    'tcga': 'TSS-grouped StratifiedGroupKFold',
    'reaction': 'GroupKFold',
    'dta': 'Cold-target GroupKFold',
    'nucleotide': 'KFold',
}


def infer_domain(ds_name, metric_name=None):
    if ds_name:
        if ds_name.startswith('tcga_'):
            return 'tcga'
        if ds_name.startswith(('flab_', 'abagym_', 'abcov_', 'ddg_', 'ginkgo_',
                               'tz_her2', 'humanness')):
            return 'antibody'
        if ds_name in ('buchwald_hartwig', 'bh_doyle_plates', 'suzuki'):
            return 'reaction'
        if ds_name in ('davis', 'kiba', 'bindingdb_kd', 'pdbbind_refined'):
            return 'dta'
        if ds_name.startswith(('nt_', 'gb_')):
            return 'nucleotide'
    if metric_name == 'spearman':
        return 'antibody'
    if metric_name == 'ci':
        return 'dta'
    return 'molecular'


def sep(char='=', width=100):
    print(char * width)


# =========================================================================
# SOURCE 1: PICKLE CACHES
# =========================================================================

def load_pickle_caches():
    """Load all cached SearchResult objects and match to datasets."""
    matched = defaultdict(list)
    if not CACHE_DIR.exists():
        return matched

    pkl_files = sorted(CACHE_DIR.glob('*.pkl'))
    if not pkl_files:
        return matched

    print(f"  Found {len(pkl_files)} pickle caches in {CACHE_DIR}")
    for pf in pkl_files:
        try:
            result = pickle.loads(pf.read_bytes())
            if not hasattr(result, 'baselines') or not result.baselines:
                continue
            ds_name, aug_cond = _match_to_dataset(result)
            if ds_name:
                matched[ds_name].append((aug_cond, result, 'pkl'))
        except Exception as e:
            print(f"    SKIP {pf.name}: {e}")
    return matched


def _match_to_dataset(result):
    cache_reps = set(result.baselines.keys())
    best_match = None
    best_overlap = 0
    for ds_name, ds_info in DATASET_REPS.items():
        general = set(ds_info.get('general', []))
        if not general:
            continue
        overlap = len(cache_reps & general)
        if overlap > best_overlap and overlap >= len(cache_reps) * 0.5:
            best_overlap = overlap
            best_match = ds_name

    aug_condition = 'none'
    if best_match:
        aug_reps = set(DATASET_REPS.get(best_match, {}).get('augmentation', []))
        if aug_reps and len(cache_reps & aug_reps) > 0:
            aug_condition = 'augmented'
        top_tier = set(DATASET_REPS.get(best_match, {}).get('top_tier', []))
        if top_tier and cache_reps.issubset(top_tier):
            aug_condition = 'top_tier'
    return best_match, aug_condition


# =========================================================================
# SOURCE 2: EXPERIMENT JSONs
# =========================================================================

def load_json_results():
    """Load from tests/results/ — phase5_baselines + C02 + other experiments."""
    datasets = {}  # {ds_name: {baselines: {...}, hybrid: {...}, meta: {...}}}

    # ── Phase 5 baselines (per-rep scores) ──
    bl_dir = RESULTS_DIR / 'phase5_baselines'
    if bl_dir.exists():
        for domain_dir in bl_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            for jf in domain_dir.glob('*.json'):
                ds = jf.stem
                try:
                    data = json.loads(jf.read_text())
                    datasets.setdefault(ds, {})['baseline_json'] = data
                    datasets[ds]['domain_dir'] = domain_dir.name
                except Exception:
                    pass

    # ── Experiment results with hybrid configs ──
    # Scan complexity, similarity, noise for any JSON with 'configs' key
    for cat_name in ('complexity', 'similarity', 'noise', 'comparisons',
                     'forward_compat', 'phase5_hybrids'):
        cat_dir = RESULTS_DIR / cat_name
        if not cat_dir.exists():
            continue
        for exp_dir in cat_dir.iterdir():
            if exp_dir.is_dir():
                for jf in exp_dir.glob('*.json'):
                    _try_load_hybrid_json(jf, datasets, cat_name, exp_dir.name)
            elif exp_dir.suffix == '.json':
                _try_load_hybrid_json(exp_dir, datasets, cat_name, '')

    return datasets


def _try_load_hybrid_json(jf, datasets, cat_name, exp_name):
    ds = jf.stem
    try:
        data = json.loads(jf.read_text())
    except Exception:
        return
    # Only interested if it has hybrid search results (configs list)
    if 'configs' not in data and 'results' not in data:
        return
    configs = data.get('configs', [])
    if not configs:
        # Try nested results.configs (framework format)
        configs = (data.get('results', {}) or {}).get('configs', [])
    if not configs:
        return
    key = f"{cat_name}/{exp_name}"
    datasets.setdefault(ds, {}).setdefault('hybrid_jsons', []).append({
        'source': key,
        'configs': configs,
        'best_by_model': data.get('best_by_model', {}),
        'baselines_summary': data.get('baselines', {}),
        'heatmap': data.get('heatmap', {}),
        'conditions': data.get('conditions', {}),
    })


def extract_baseline_scores(bl_data):
    """Extract per-rep scores from phase5_baselines JSON."""
    meta = bl_data.get('metadata', {})
    baselines = bl_data.get('baselines', {})
    primary_metric = meta.get('primary_metric', 'r2')

    rep_scores = {}  # {rep: {model: score}}
    rep_dims = {}
    for rep_name, rep_data in baselines.items():
        rep_dims[rep_name] = rep_data.get('dimension', '?')
        scores_raw = rep_data.get('scores', {})
        rep_scores[rep_name] = {}
        for model, val in scores_raw.items():
            if isinstance(val, dict):
                for key_try in [f'{primary_metric}_mean', 'spearman_mean', 'r2_mean']:
                    if key_try in val:
                        rep_scores[rep_name][model] = val[key_try]
                        break
                else:
                    for k, v in val.items():
                        if k.endswith('_mean') and isinstance(v, (int, float)):
                            rep_scores[rep_name][model] = v
                            break
            else:
                rep_scores[rep_name][model] = val
    return rep_scores, rep_dims, meta


def extract_baseline_scores_r2(bl_data):
    """Extract per-rep R² scores from phase5_baselines JSON.

    Used when hybrid configs are in R² (C02 format) so baselines must match.
    """
    meta = bl_data.get('metadata', {})
    baselines = bl_data.get('baselines', {})

    rep_scores = {}
    rep_dims = {}
    for rep_name, rep_data in baselines.items():
        rep_dims[rep_name] = rep_data.get('dimension', '?')
        scores_raw = rep_data.get('scores', {})
        rep_scores[rep_name] = {}
        for model, val in scores_raw.items():
            if isinstance(val, dict):
                # Always use r2_mean
                if 'r2_mean' in val:
                    rep_scores[rep_name][model] = val['r2_mean']
                elif 'r2' in val and isinstance(val['r2'], dict):
                    rep_scores[rep_name][model] = val['r2'].get('mean', 0)
            elif isinstance(val, (int, float)):
                # Molecular format: scores.model = rmse_value
                # Need to get R² from metrics instead
                metrics = rep_data.get('metrics', {}).get(model, {})
                r2_info = metrics.get('r2', {})
                if isinstance(r2_info, dict) and 'mean' in r2_info:
                    rep_scores[rep_name][model] = r2_info['mean']
                # else: skip — can't get R² from this format
    return rep_scores, rep_dims, meta


# =========================================================================
# DISPLAY
# =========================================================================

def print_from_pickle(ds_name, domain, searches):
    """Print results from cached SearchResult objects."""
    primary = searches[0][1]
    _print_dataset(ds_name, domain, primary.metric_name, primary.lower_is_better,
                   baselines=primary.baselines,
                   all_results=primary.all_results,
                   best_by_model=primary.best_by_model,
                   n_searches=len(searches),
                   search_labels=[c for c, _, _ in searches])


def print_from_json(ds_name, ds_data):
    """Print results from JSON files."""
    bl_json = ds_data.get('baseline_json')
    hybrid_jsons = ds_data.get('hybrid_jsons', [])
    domain = ds_data.get('domain_dir', infer_domain(ds_name))

    # Baselines
    rep_scores = {}
    rep_dims = {}
    meta = {}
    if bl_json:
        rep_scores, rep_dims, meta = extract_baseline_scores(bl_json)

    # Use the baseline JSON's metric — it's always domain-correct.
    # C02 configs use 'r2_mean' as the key name regardless of actual metric,
    # but the VALUES are whatever metric the search used.
    metric = meta.get('primary_metric', '?')
    higher = meta.get('higher_is_better', True)
    lower = not higher

    # Convert to baselines format matching SearchResult: {rep: {model: score, dimension: dim}}
    baselines = {}
    for rn, scores in rep_scores.items():
        baselines[rn] = dict(scores)
        baselines[rn]['dimension'] = rep_dims.get(rn, '?')

    # Pick best hybrid json (prefer C02, largest configs list)
    all_results = []
    best_by_model = {}
    source_label = ''
    if hybrid_jsons:
        # Prefer c02
        best_hj = max(hybrid_jsons, key=lambda h: len(h['configs']))
        source_label = best_hj['source']
        all_results = best_hj['configs']
        best_by_model = best_hj.get('best_by_model', {})

    _print_dataset(ds_name, domain, metric, lower,
                   baselines=baselines,
                   all_results=all_results,
                   best_by_model=best_by_model,
                   n_searches=len(hybrid_jsons),
                   search_labels=[source_label] if source_label else [],
                   meta=meta,
                   is_json=True)


def _print_dataset(ds_name, domain, metric, lower_is_better,
                   baselines=None, all_results=None, best_by_model=None,
                   n_searches=0, search_labels=None, meta=None, is_json=False):
    """Unified display for one dataset."""
    higher = not lower_is_better
    direction = '↑' if higher else '↓'
    baselines = baselines or {}
    all_results = all_results or []
    best_by_model = best_by_model or {}
    meta = meta or {}

    ds_info = DATASET_REPS.get(ds_name, {})
    general = ds_info.get('general', [])
    top_tier = ds_info.get('top_tier', [])
    aug_reps = ds_info.get('augmentation', [])

    # ── Header ──
    sep()
    print(f"DATASET: {ds_name}")
    sep('-')
    print(f"  Domain:       {domain}")
    print(f"  Metric:       {metric} ({direction})")
    n_samples = meta.get('n_samples', '?')
    if n_samples != '?':
        print(f"  Samples:      {n_samples}")
    print(f"  Splitting:    {SPLITTING.get(domain, '?')}")
    if general:
        print(f"  General ({len(general)}):  {', '.join(general)}")
    if top_tier:
        print(f"  Top-tier ({len(top_tier)}): {', '.join(top_tier)}")
    if aug_reps:
        print(f"  Augmentation ({len(aug_reps)}): {', '.join(aug_reps)}")
    if search_labels:
        print(f"  Sources: {', '.join(search_labels)}")

    # ── Baselines ──
    if not baselines:
        print("\n  (no baseline scores)")
        return

    all_models = set()
    for rep_data in baselines.values():
        all_models.update(k for k in rep_data
                          if k not in ('dimension', '_secondary')
                          and not k.startswith('_'))
    all_models = sorted(all_models)

    sort_model = 'rf' if 'rf' in all_models else (all_models[0] if all_models else None)
    if not sort_model:
        print("  (no model scores)")
        return

    def _sk(rn):
        s = baselines[rn].get(sort_model)
        if not isinstance(s, (int, float)):
            return float('-inf') if higher else float('inf')
        return s if higher else -s

    sorted_reps = sorted(baselines.keys(), key=_sk, reverse=True)

    show_models = [sort_model]
    for m in ['lightgbm', 'xgboost', 'catboost', 'mlp', 'dnn', 'knn', 'gp']:
        if m in all_models and m != sort_model:
            show_models.append(m)
    show_models = show_models[:5]

    print(f"\n  BASE REP SCORES (sorted by {sort_model}, metric={metric}):")
    hdr = f"  {'rep':<25} {'dim':>6}"
    for m in show_models:
        hdr += f"{m:>12}"
    print(hdr)
    print("  " + "-" * (31 + 12 * len(show_models)))

    for rn in sorted_reps:
        dim = baselines[rn].get('dimension', '?')
        row = f"  {rn:<25} {str(dim):>6}"
        for m in show_models:
            s = baselines[rn].get(m)
            if isinstance(s, (int, float)):
                row += f"{s:>12.4f}"
            else:
                row += f"{'--':>12}"
        print(row)

    best_rep = sorted_reps[0]
    best_val = baselines[best_rep].get(sort_model, 0)
    if isinstance(best_val, (int, float)):
        print(f"\n  >>> Best baseline ({sort_model}): {best_rep} = {best_val:.4f}")

    # ── Hybrid results ──
    if not all_results:
        print("\n  (no hybrid search results)")
        print()
        return

    # Detect score key — C02 JSONs use 'r2_mean', SearchResult uses 'score_mean'
    score_key = 'score_mean'
    if all_results[0].get('r2_mean') is not None and all_results[0].get('score_mean') is None:
        score_key = 'r2_mean'

    n_total = len(all_results)
    print(f"\n  HYBRID RESULTS ({n_total} total configs):")

    # Build heatmap from all_results
    heatmap = {}
    for r in all_results:
        mn = r.get('model')
        b = r.get('budget')
        s = r.get(score_key, 0)
        if mn and b is not None:
            heatmap.setdefault(mn, {})
            prev = heatmap[mn].get(b)
            if prev is None or (higher and s > prev) or (not higher and s < prev):
                heatmap[mn][b] = s

    if heatmap:
        all_budgets = sorted(set(b for m_s in heatmap.values() for b in m_s))
        print(f"\n    Heatmap (best {metric} per model x budget):")
        hdr = f"    {'model':<12}" + "".join(f"{'b='+str(b):>10}" for b in all_budgets)
        print(hdr)
        print("    " + "-" * (12 + 10 * len(all_budgets)))
        model_order = [sort_model] + [m for m in sorted(heatmap) if m != sort_model]
        for mn in model_order:
            if mn not in heatmap:
                continue
            row = f"    {mn:<12}"
            for b in all_budgets:
                val = heatmap[mn].get(b)
                row += f"{val:>10.4f}" if val is not None else f"{'--':>10}"
            print(row)

    # Per-model detail
    models_to_show = [sort_model] + [m for m in sorted(heatmap) if m != sort_model]
    for mn in models_to_show:
        model_cfgs = [r for r in all_results if r.get('model') == mn]
        if not model_cfgs:
            continue

        # Find best config for this model
        if is_json and mn in best_by_model:
            best = best_by_model[mn]
            best_score = best.get(score_key, best.get('r2_mean', 0))
        else:
            if higher:
                best = max(model_cfgs, key=lambda c: c.get(score_key, 0))
            else:
                best = min(model_cfgs, key=lambda c: c.get(score_key, float('inf')))
            best_score = best.get(score_key, 0)

        # Baseline for this model
        bl_score = None
        bl_rep = None
        for rn in sorted_reps:
            s = baselines[rn].get(mn)
            if isinstance(s, (int, float)):
                if bl_score is None:
                    bl_score = s
                    bl_rep = rn
                elif (higher and s > bl_score) or (not higher and s < bl_score):
                    bl_score = s
                    bl_rep = rn

        # Count exceeding
        n_exceed = 0
        if bl_score is not None:
            for c in model_cfgs:
                cs = c.get(score_key, 0)
                if (higher and cs > bl_score) or (not higher and cs < bl_score):
                    n_exceed += 1

        delta = (best_score - bl_score) if bl_score is not None else None

        print(f"\n    {mn}:")
        if bl_rep:
            print(f"      Best baseline:    {bl_rep} = {bl_score:.4f}")
        print(f"      Best hybrid:      {best_score:.4f}"
              + (f" (delta={delta:+.4f})" if delta is not None else ""))
        print(f"      Configs tested:   {len(model_cfgs)}")
        if bl_score is not None:
            print(f"      Beat baseline:    {n_exceed}/{len(model_cfgs)} "
                  f"({100*n_exceed/max(len(model_cfgs),1):.0f}%)")
        # Config details — field names differ between JSON and pickle
        alloc = best.get('allocation_method', best.get('allocation', '?'))
        imp = best.get('importance_method', best.get('importance', '?'))
        ie = best.get('internal_evaluator', '?')
        bgt = best.get('budget', '?')
        nf = best.get('n_features', '?')
        print(f"      Allocation:       {alloc}")
        print(f"      Importance:       {imp}")
        if ie != '?':
            print(f"      Int. evaluator:   {ie}")
        print(f"      Budget:           {bgt}")
        print(f"      N features:       {nf}")

    print()
    return {
        'sort_model': sort_model, 'higher': higher,
        'baselines': baselines, 'all_results': all_results,
        'sorted_reps': sorted_reps, 'score_key': score_key,
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("\n" + "=" * 100)
    print("  KIRBY — SUPERVISOR PRESENTATION")
    print("=" * 100)

    # Try pickle caches first
    pkl_matched = load_pickle_caches()

    # Also load JSON results
    json_datasets = load_json_results()
    n_bl = sum(1 for d in json_datasets.values() if 'baseline_json' in d)
    n_hyb = sum(1 for d in json_datasets.values() if d.get('hybrid_jsons'))
    print(f"  JSON results: {n_bl} baseline datasets, {n_hyb} with hybrid configs")

    # Merge: pickle wins over JSON for same dataset
    all_ds = sorted(set(list(pkl_matched.keys()) + list(json_datasets.keys())))
    print(f"  Total datasets: {len(all_ds)}")
    sep()

    summary_rows = []

    for ds_name in all_ds:
        domain = infer_domain(ds_name)
        info = None

        if ds_name in pkl_matched:
            searches = pkl_matched[ds_name]
            order = {'none': 0, 'augmented': 1, 'top_tier': 2}
            searches.sort(key=lambda x: order.get(x[0], 99))
            domain = infer_domain(ds_name, searches[0][1].metric_name)
            info = print_from_pickle(ds_name, domain, searches)
        elif ds_name in json_datasets:
            ds_data = json_datasets[ds_name]
            if ds_data.get('domain_dir'):
                domain = ds_data['domain_dir']
            info = print_from_json(ds_name, ds_data)

        # Collect summary
        if info and info.get('all_results'):
            higher = info['higher']
            bl = info['baselines']
            sort_model = info['sort_model']
            score_key = info['score_key']

            rf_scores = {}
            for rn, rs in bl.items():
                s = rs.get(sort_model)
                if isinstance(s, (int, float)):
                    rf_scores[rn] = s

            best_rep = '?'
            best_val = None
            if rf_scores:
                best_rep = (max(rf_scores, key=rf_scores.get) if higher
                            else min(rf_scores, key=rf_scores.get))
                best_val = rf_scores[best_rep]

            model_cfgs = [r for r in info['all_results']
                          if r.get('model') == sort_model]
            n_tested = len(model_cfgs)

            # Best hybrid
            hybrid_score = None
            if model_cfgs:
                if higher:
                    hybrid_score = max(c.get(score_key, 0) for c in model_cfgs)
                else:
                    hybrid_score = min(c.get(score_key, float('inf'))
                                       for c in model_cfgs)

            delta = None
            n_exceed = 0
            if best_val is not None and hybrid_score is not None:
                delta = hybrid_score - best_val
                for c in model_cfgs:
                    cs = c.get(score_key, 0)
                    if (higher and cs > best_val) or (not higher and cs < best_val):
                        n_exceed += 1

            summary_rows.append({
                'ds': ds_name, 'domain': domain,
                'metric': info.get('metric', '?'),
                'model': sort_model,
                'best_rep': best_rep, 'best_val': best_val,
                'hybrid': hybrid_score, 'delta': delta,
                'n_tested': n_tested, 'n_exceed': n_exceed,
            })

    # ── Grand summary ──
    if summary_rows:
        sep('=')
        print("GRAND SUMMARY")
        sep('-')
        sm = summary_rows[0].get('model', 'rf')
        print(f"  {'dataset':<35} {'domain':<10} {'metric':<8} "
              f"{'best_rep':<18} {'base':>8} {'hybrid':>8} {'delta':>8} "
              f"{'tested':>7} {'beat':>7}")
        print("  " + "-" * 115)

        for s in summary_rows:
            bv = f"{s['best_val']:.4f}" if s['best_val'] is not None else '--'
            hv = f"{s['hybrid']:.4f}" if s['hybrid'] is not None else '--'
            dv = f"{s['delta']:+.4f}" if s['delta'] is not None else '--'
            print(f"  {s['ds']:<35} {s['domain']:<10} "
                  f"{s.get('metric','?'):<8} "
                  f"{s['best_rep']:<18} {bv:>8} {hv:>8} {dv:>8} "
                  f"{s['n_tested']:>7} {s['n_exceed']:>7}")

        # ── Evidence ──
        sep('=')
        print("EVIDENCE: HYBRID IMPROVEMENTS ARE STRUCTURAL, NOT ONE-OFF")
        sep('-')
        with_hybrids = [s for s in summary_rows if s['n_tested'] > 0]
        if with_hybrids:
            total_tested = sum(s['n_tested'] for s in with_hybrids)
            total_beat = sum(s['n_exceed'] for s in with_hybrids)
            n_improved = sum(1 for s in with_hybrids
                             if s['delta'] is not None and s['delta'] > 0)
            print(f"  Datasets with hybrid results:    {len(with_hybrids)}")
            print(f"  Datasets where hybrid > base:    {n_improved}/{len(with_hybrids)}")
            print(f"  Total configs tested:            {total_tested}")
            print(f"  Total configs beating baseline:  {total_beat}/{total_tested} "
                  f"({100*total_beat/max(total_tested,1):.1f}%)")
            print()
            print(f"  {'dataset':<35} {'beat/tested':>12} {'pct':>6}")
            print("  " + "-" * 55)
            for s in with_hybrids:
                pct = f"{100*s['n_exceed']/max(s['n_tested'],1):.0f}%"
                print(f"  {s['ds']:<35} "
                      f"{s['n_exceed']:>5}/{s['n_tested']:<6} {pct:>6}")
    print()


if __name__ == '__main__':
    main()
