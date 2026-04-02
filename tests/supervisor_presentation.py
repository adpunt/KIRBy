#!/usr/bin/env python3
"""Generate supervisor meeting presentation from cached hybrid search results.

Run ON THE SERVER:
    cd /data/stat-cadd/scat9264/KIRBy
    python tests/supervisor_presentation.py | tee presentation_output.txt

Loads SearchResult pickle caches from ~/.cache/kirby/hybrid_search/.
Matches each cache to a dataset via rep names in DATASET_REPS.
For each dataset shows:
  - Dataset info (domain, metric, n, splitting, general/top reps)
  - All base rep scores sorted by RF (no +/- values)
  - Hybrid heatmap (model x budget)
  - Top hybrid config per model (allocation, importance, budget)
  - Number of configs tested and number exceeding best baseline
"""
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
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

# ── Infer domain from metric ──
METRIC_DOMAIN = {
    'rmse': 'molecular',
    'spearman': 'antibody',
    'auc': 'classification',
    'f1_weighted': 'classification',
    'ci': 'dta',
    'mcc': 'nucleotide',
    'r2': 'molecular',
}


def load_all_caches():
    """Load all cached SearchResult objects."""
    caches = []
    if not CACHE_DIR.exists():
        print(f"  Cache dir not found: {CACHE_DIR}")
        return caches
    pkl_files = sorted(CACHE_DIR.glob('*.pkl'))
    print(f"  Found {len(pkl_files)} cache files in {CACHE_DIR}")
    for pf in pkl_files:
        try:
            result = pickle.loads(pf.read_bytes())
            caches.append({'path': pf, 'result': result})
        except Exception as e:
            print(f"  SKIP {pf.name}: {e}")
    return caches


def match_cache_to_dataset(result):
    """Match a SearchResult to a dataset name using rep names."""
    if not hasattr(result, 'baselines') or not result.baselines:
        return None, None

    cache_reps = set(result.baselines.keys())

    best_match = None
    best_overlap = 0
    for ds_name, ds_info in DATASET_REPS.items():
        general = set(ds_info.get('general', []))
        if not general:
            continue
        overlap = len(cache_reps & general)
        # Match if most of the cache reps appear in general list
        if overlap > best_overlap and overlap >= len(cache_reps) * 0.5:
            best_overlap = overlap
            best_match = ds_name

    # Infer aug condition from presence of augmentation reps
    aug_condition = 'none'
    if best_match:
        aug_reps = set(DATASET_REPS.get(best_match, {}).get('augmentation', []))
        if aug_reps and len(cache_reps & aug_reps) > 0:
            aug_condition = 'augmented'
        # Check if it's top-tier only
        top_tier = set(DATASET_REPS.get(best_match, {}).get('top_tier', []))
        if top_tier and cache_reps == top_tier:
            aug_condition = 'top_tier'
        elif top_tier and cache_reps.issubset(top_tier):
            aug_condition = 'top_tier'

    return best_match, aug_condition


def infer_domain(ds_name, metric_name):
    """Infer domain from dataset name or metric."""
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
    return METRIC_DOMAIN.get(metric_name, 'molecular')


def get_n_samples(result):
    """Infer n_samples from baselines — check first rep's per_fold if available."""
    # all_results might have fold info
    if result.all_results:
        fs = result.all_results[0].get('fold_scores', [])
        # Can't get n_samples from fold scores alone
    return '?'


def sep(char='=', width=100):
    print(char * width)


def print_dataset_results(ds_name, domain, searches):
    """Print full results for one dataset across all its cached searches.

    searches: list of (aug_condition, SearchResult)
    """
    ds_info = DATASET_REPS.get(ds_name, {})
    general = ds_info.get('general', [])
    top_tier = ds_info.get('top_tier', [])
    aug_reps = ds_info.get('augmentation', [])

    # Use the first search (usually 'none' / general reps) for baselines
    primary = searches[0][1]
    metric = primary.metric_name
    lower = primary.lower_is_better
    higher = not lower
    direction = '↑' if higher else '↓'

    sep()
    print(f"DATASET: {ds_name}")
    sep('-')
    print(f"  Domain:       {domain}")
    print(f"  Metric:       {metric} ({direction})")
    print(f"  Splitting:    {SPLITTING.get(domain, '?')}")
    if general:
        print(f"  General ({len(general)}):  {', '.join(general)}")
    if top_tier:
        print(f"  Top-tier ({len(top_tier)}): {', '.join(top_tier)}")
    if aug_reps:
        print(f"  Augmentation ({len(aug_reps)}): {', '.join(aug_reps)}")
    print(f"  Cached searches: {len(searches)} "
          f"({', '.join(cond for cond, _ in searches)})")

    # ── Baselines: all rep scores ──
    baselines = primary.baselines  # {rep_name: {model: score}}
    if not baselines:
        print("\n  (no baselines in cache)")
        return

    all_models = set()
    for rep_scores in baselines.values():
        all_models.update(k for k in rep_scores
                          if not k.startswith('_') and k != 'dimension')
    all_models = sorted(all_models)

    sort_model = 'rf' if 'rf' in all_models else (all_models[0] if all_models else None)
    if not sort_model:
        print("  (no model scores)")
        return

    # Sort reps by sort_model
    def _sk(rn):
        s = baselines[rn].get(sort_model, float('-inf') if higher else float('inf'))
        if not isinstance(s, (int, float)):
            return float('-inf') if higher else float('inf')
        return s if higher else -s

    sorted_reps = sorted(baselines.keys(), key=_sk, reverse=True)

    # Pick models to show
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

    # Best baseline
    best_rep = sorted_reps[0]
    best_val = baselines[best_rep].get(sort_model, 0)
    if isinstance(best_val, (int, float)):
        print(f"\n  >>> Best baseline ({sort_model}): {best_rep} = {best_val:.4f}")

    # ── Hybrid results per search ──
    for aug_cond, search in searches:
        all_results = search.all_results
        if not all_results:
            continue

        n_total = len(all_results)
        print(f"\n  HYBRID RESULTS [{aug_cond}] ({n_total} total configs):")

        # Best by model x budget — build from all_results since older caches
        # may not have best_by_model_budget
        heatmap = {}  # {model: {budget: best_score}}
        for r in all_results:
            mn = r.get('model')
            b = r.get('budget')
            s = r.get('score_mean', 0)
            if mn and b is not None:
                if mn not in heatmap:
                    heatmap[mn] = {}
                prev = heatmap[mn].get(b)
                if prev is None or (higher and s > prev) or (lower and s < prev):
                    heatmap[mn][b] = s

        if heatmap:
            all_budgets = set()
            for m_scores in heatmap.values():
                all_budgets.update(m_scores.keys())
            budgets = sorted(all_budgets)

            print(f"\n    Heatmap (best {metric} per model x budget):")
            hdr = f"    {'model':<12}" + "".join(f"{'b='+str(b):>10}" for b in budgets)
            print(hdr)
            print("    " + "-" * (12 + 10 * len(budgets)))

            model_order = ([sort_model] +
                           [m for m in heatmap if m != sort_model])
            for mn in model_order:
                if mn not in heatmap:
                    continue
                row = f"    {mn:<12}"
                for b in budgets:
                    val = heatmap[mn].get(b)
                    if val is not None:
                        row += f"{val:>10.4f}"
                    else:
                        row += f"{'--':>10}"
                print(row)

        # Per-model detail
        for mn in ([sort_model] +
                   [m for m in (search.best_by_model or {}) if m != sort_model]):
            if mn not in (search.best_by_model or {}):
                continue
            best = search.best_by_model[mn]
            best_score = best.get('score_mean', 0)

            # Baseline for this model
            bl_score = None
            bl_rep = None
            for rn in sorted_reps:
                s = baselines[rn].get(mn)
                if isinstance(s, (int, float)):
                    if bl_score is None:
                        bl_score = s
                        bl_rep = rn
                    elif (higher and s > bl_score) or (lower and s < bl_score):
                        bl_score = s
                        bl_rep = rn

            # Count configs beating baseline
            model_cfgs = [r for r in all_results if r.get('model') == mn]
            n_exceed = 0
            if bl_score is not None:
                for c in model_cfgs:
                    cs = c.get('score_mean', 0)
                    if (higher and cs > bl_score) or (lower and cs < bl_score):
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
            print(f"      Allocation:       {best.get('allocation_method', '?')}")
            print(f"      Importance:       {best.get('importance_method', '?')}")
            print(f"      Int. evaluator:   {best.get('internal_evaluator', '?')}")
            print(f"      Budget:           {best.get('budget', '?')}")
            print(f"      N features:       {best.get('n_features', '?')}")

    print()


def main():
    print("\n" + "=" * 100)
    print("  KIRBY — SUPERVISOR PRESENTATION (from cached hybrid searches)")
    print("=" * 100)

    caches = load_all_caches()
    if not caches:
        print("  No cached results found. Run experiments first.")
        return

    # Match each cache to a dataset
    matched = defaultdict(list)  # {ds_name: [(aug_condition, SearchResult), ...]}
    unmatched = []

    for entry in caches:
        result = entry['result']
        ds_name, aug_cond = match_cache_to_dataset(result)
        if ds_name:
            matched[ds_name].append((aug_cond, result))
        else:
            unmatched.append(entry)

    print(f"\n  Matched: {len(matched)} datasets, {sum(len(v) for v in matched.values())} searches")
    print(f"  Unmatched: {len(unmatched)} cache files")
    if unmatched:
        for u in unmatched[:5]:
            r = u['result']
            reps = list(r.baselines.keys())[:5] if hasattr(r, 'baselines') and r.baselines else []
            print(f"    {u['path'].name}: reps={reps}...")
    sep()

    # ── Per-dataset output ──
    summary_rows = []

    for ds_name in sorted(matched.keys()):
        searches = matched[ds_name]
        # Sort: 'none' first, then 'augmented', then 'top_tier'
        order = {'none': 0, 'augmented': 1, 'top_tier': 2}
        searches.sort(key=lambda x: order.get(x[0], 99))

        domain = infer_domain(ds_name, searches[0][1].metric_name)
        print_dataset_results(ds_name, domain, searches)

        # Summary row
        primary = searches[0][1]
        higher = not primary.lower_is_better
        bl = primary.baselines or {}
        rf_scores = {}
        for rn, rs in bl.items():
            s = rs.get('rf')
            if isinstance(s, (int, float)):
                rf_scores[rn] = s

        best_rep = '?'
        best_val = None
        if rf_scores:
            best_rep = (max(rf_scores, key=rf_scores.get) if higher
                        else min(rf_scores, key=rf_scores.get))
            best_val = rf_scores[best_rep]

        hybrid_score = None
        delta = None
        n_tested = 0
        n_exceed = 0
        if primary.best_by_model and 'rf' in primary.best_by_model:
            hybrid_score = primary.best_by_model['rf'].get('score_mean')
            if hybrid_score is not None and best_val is not None:
                delta = hybrid_score - best_val
        model_cfgs = [r for r in (primary.all_results or []) if r.get('model') == 'rf']
        n_tested = len(model_cfgs)
        if best_val is not None:
            for c in model_cfgs:
                cs = c.get('score_mean', 0)
                if (higher and cs > best_val) or (not higher and cs < best_val):
                    n_exceed += 1

        summary_rows.append({
            'ds': ds_name, 'domain': domain, 'metric': primary.metric_name,
            'best_rep': best_rep, 'best_val': best_val,
            'hybrid': hybrid_score, 'delta': delta,
            'n_tested': n_tested, 'n_exceed': n_exceed,
            'n_searches': len(searches),
        })

    # ── Grand summary ──
    sep('=')
    print("GRAND SUMMARY (RF downstream)")
    sep('-')
    print(f"  {'dataset':<35} {'domain':<10} {'metric':<8} "
          f"{'best_rep':<18} {'base':>8} {'hybrid':>8} {'delta':>8} "
          f"{'tested':>7} {'beat':>7}")
    print("  " + "-" * 115)

    for s in summary_rows:
        bv = f"{s['best_val']:.4f}" if s['best_val'] is not None else '--'
        hv = f"{s['hybrid']:.4f}" if s['hybrid'] is not None else '--'
        dv = f"{s['delta']:+.4f}" if s['delta'] is not None else '--'
        print(f"  {s['ds']:<35} {s['domain']:<10} {s['metric']:<8} "
              f"{s['best_rep']:<18} {bv:>8} {hv:>8} {dv:>8} "
              f"{s['n_tested']:>7} {s['n_exceed']:>7}")

    # ── Evidence section ──
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
        print(f"  Total RF configs tested:         {total_tested}")
        print(f"  Total RF configs beating base:   {total_beat}/{total_tested} "
              f"({100*total_beat/max(total_tested,1):.1f}%)")
        print()
        print(f"  {'dataset':<35} {'beat/tested':>12} {'pct':>6}")
        print("  " + "-" * 55)
        for s in with_hybrids:
            pct = f"{100*s['n_exceed']/max(s['n_tested'],1):.0f}%"
            print(f"  {s['ds']:<35} {s['n_exceed']:>5}/{s['n_tested']:<6} {pct:>6}")
    print()


if __name__ == '__main__':
    main()
