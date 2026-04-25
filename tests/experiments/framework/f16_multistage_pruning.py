#!/usr/bin/env python3
"""F16: per-model pruning threshold sweep — offline simulation on f15 data.

For each (model M, schedule S) on each f15 dataset D, simulate applying
schedule S ONLY to model M's configs (other models stay at baseline).
Report two numbers:

    rel_loss(M, S, D)
        = (pruned_best_for_M - baseline_best_for_M) / |baseline_best_for_M|
          sign-corrected for lower-is-better metrics
        Interpretation: how much worse is the best fold-mean score for
        model M after applying schedule S, expressed as a % of M's
        unpruned best. Zero means the same config was preserved.

    saved(M, S, D)
        = (M's fold-evals avoided by S) / (M's total fold-evals)
        Interpretation: fraction of compute spent on model M that would
        not have been done under schedule S. Higher = more saved.

A schedule "passes" for (M, D) if rel_loss <= 2%. A schedule is
universally good for model M if it passes on every dataset where M has
data. A schedule is universally good overall if it passes for every
(M, D) pair.

Why per-model: pruning is most useful for expensive models (svm, mlp)
and harmless on cheap ones (rf, ridge). We want to know whether the
SAME schedule can be applied uniformly OR whether different models need
different schedules.

Why this criterion: experiments consume best_by_model[m]. If pruning
preserves that score within 2%, the experiment result is unchanged in
practice. Top-K rank-stability across models is irrelevant — different
models are read independently.

Caveat: if every model's best config sits at budget <= max(checkpoints)
on a dataset, that (model, dataset) is flagged "trivial preservation"
because the schedule wasn't given the chance to fail (the winner's data
is in the cut-decision set). Trivial cases are reported but don't count
toward "passes" — they're inconclusive, not pass.

f15 source: tests/results/framework/f15_adaptive_pruning/
  - {dataset}_{model}.json: per-model split file (preferred, complete)
  - {dataset}.json: combined file (often incomplete; only used as fallback
    for models lacking a per-model file)

Output: tests/results/framework/f16_multistage_pruning/per_model_sweep.json
        + a printed table per dataset and a final cross-dataset summary.

Usage:
    python tests/experiments/framework/f16_multistage_pruning.py
    python tests/experiments/framework/f16_multistage_pruning.py --dataset caco2
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
F15_DIR = REPO_ROOT / 'tests' / 'results' / 'framework' / 'f15_adaptive_pruning'
F16_DIR = REPO_ROOT / 'tests' / 'results' / 'framework' / 'f16_multistage_pruning'

F16_DATASETS = [
    'flab_pk_jain_fcrnrt',
    'abagym_EGFR',
    'caco2',
    'herg_ki',
    'logd',
    'buchwald_hartwig',
]

# Candidate schedules. Each defines: at which budgets do we look at fold-0
# scores within a model's group, and what fraction of the bottom (alloc, imp)
# pairs do we drop. Pruned pairs lose all their higher-budget configs too.
SCHEDULES: Dict[str, Dict[str, Any]] = {
    'baseline': {
        'checkpoints': [],
        'tail_drop_per_checkpoint': 0.0,
    },
    'light': {
        'checkpoints': [50],
        'tail_drop_per_checkpoint': 0.25,
    },
    'medium': {
        'checkpoints': [25, 50],
        'tail_drop_per_checkpoint': 0.50,
    },
    'aggressive': {
        'checkpoints': [25, 50, 100],
        'tail_drop_per_checkpoint': 0.50,
    },
    'very_aggressive': {
        'checkpoints': [10, 25, 50],
        'tail_drop_per_checkpoint': 0.75,
    },
}

# Acceptance threshold for rel_loss. 2% means the pruned best is within 2% of
# baseline best for that model. Configurable but default is the same threshold
# F15 used when validating the legacy single-stage adaptive_pruning=0.15.
PASS_THRESHOLD = 0.02

F16_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']


# ---------------------------------------------------------------------------
# F15 loading: merge per-model split files
# ---------------------------------------------------------------------------

def _load_f15(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Merge per-model f15 files into a single results list.

    Prefers per-model files (complete per model). Falls back to the combined
    file's slice ONLY when the combined file is marked complete. Returns None
    if no usable data exists.
    """
    combined_path = F15_DIR / f'{dataset_name}.json'
    combined_data: Optional[Dict[str, Any]] = None
    combined_complete = False
    if combined_path.exists():
        with open(combined_path) as f:
            combined_data = json.load(f)
        combined_complete = bool(combined_data.get('complete'))

    combined_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in F16_MODELS}
    if combined_data is not None:
        for r in combined_data.get('results', []):
            m = r.get('model')
            if m in combined_by_model:
                combined_by_model[m].append(r)

    merged: List[Dict[str, Any]] = []
    sources: Dict[str, str] = {}
    meta_seed: Optional[Dict[str, Any]] = None
    for model in F16_MODELS:
        per_path = F15_DIR / f'{dataset_name}_{model}.json'
        if per_path.exists():
            with open(per_path) as f:
                pm = json.load(f)
            if pm.get('complete'):
                merged.extend(pm.get('results', []))
                sources[model] = 'per-model'
                meta_seed = meta_seed or pm
                continue
        if combined_complete and combined_by_model[model]:
            merged.extend(combined_by_model[model])
            sources[model] = 'combined'
            meta_seed = meta_seed or combined_data
        else:
            sources[model] = 'skipped'

    if not merged:
        return None
    base = meta_seed or combined_data or {}
    return {
        'dataset': dataset_name,
        'metric': base.get('metric'),
        'lower_is_better': bool(base.get('lower_is_better')),
        'n_samples': base.get('n_samples'),
        'n_folds': base.get('n_folds'),
        'results': merged,
        'merge_sources': sources,
    }


# ---------------------------------------------------------------------------
# Pruning simulation, scoped to one model
# ---------------------------------------------------------------------------

def _f0_score(result: Dict[str, Any], lower_is_better: bool) -> float:
    fs = result.get('fold_scores') or []
    if not fs:
        return float('inf') if lower_is_better else float('-inf')
    v = fs[0]
    if v != v:  # NaN
        return float('inf') if lower_is_better else float('-inf')
    return v


def _apply_per_model_pruning(
    results: List[Dict[str, Any]],
    model: str,
    schedule: Dict[str, Any],
    lower_is_better: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Apply a pruning schedule ONLY to one model's configs.

    Returns: (kept_configs_for_this_model, n_configs_dropped, n_evals_avoided).

    Within the model's results, group by `evaluator`. At each checkpoint
    budget b in schedule['checkpoints'] (in ascending order):
      1. Take all surviving configs with budget <= b.
      2. Sort by fold-0 score; keep top (1 - tail_drop) fraction by count.
      3. Identify dropped (alloc, importance) pairs (those NOT in keep set).
      4. Mark all surviving configs (at any budget) with those (alloc, imp)
         pairs as dropped — they would never have run.
    """
    checkpoints = sorted(int(b) for b in schedule.get('checkpoints', []))
    tail_drop = float(schedule.get('tail_drop_per_checkpoint', 0.0))

    model_results = [r for r in results if r.get('model') == model]
    if not model_results or tail_drop <= 0.0 or not checkpoints:
        return model_results, 0, 0

    # Group by evaluator
    by_eval: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in model_results:
        by_eval[r['evaluator']].append(r)

    kept_results: List[Dict[str, Any]] = []
    n_dropped = 0
    n_evals_avoided = 0

    for ev, group_results in by_eval.items():
        alive = list(group_results)
        for ckpt in checkpoints:
            up_to = [r for r in alive if r['budget'] <= ckpt]
            if not up_to:
                continue
            up_to_sorted = sorted(up_to,
                                  key=lambda r: _f0_score(r, lower_is_better),
                                  reverse=not lower_is_better)
            n = len(up_to_sorted)
            n_keep = max(1, int(round(n * (1 - tail_drop))))
            kept_pairs = {(r['strategy'], r['importance'])
                          for r in up_to_sorted[:n_keep]}
            new_alive = []
            for r in alive:
                pair = (r['strategy'], r['importance'])
                if pair in kept_pairs:
                    new_alive.append(r)
                else:
                    n_dropped += 1
                    fs = r.get('fold_scores') or []
                    n_folds_total = len(fs)
                    if r['budget'] > ckpt:
                        # Never reached even fold 0
                        n_evals_avoided += n_folds_total
                    else:
                        # Did fold 0; saved folds 1..N-1
                        n_evals_avoided += max(0, n_folds_total - 1)
            alive = new_alive
        kept_results.extend(alive)

    return kept_results, n_dropped, n_evals_avoided


def _best_for_model(results: List[Dict[str, Any]], model: str,
                    lower_is_better: bool) -> Optional[Dict[str, Any]]:
    rows = [r for r in results
            if r.get('model') == model
            and r.get('score_mean') is not None
            and r['score_mean'] == r['score_mean']]
    if not rows:
        return None
    return min(rows, key=lambda r: r['score_mean']) if lower_is_better \
        else max(rows, key=lambda r: r['score_mean'])


# ---------------------------------------------------------------------------
# Per-(dataset, schedule, model) sweep
# ---------------------------------------------------------------------------

def _sweep_dataset(f15: Dict[str, Any]) -> Dict[str, Any]:
    results = f15['results']
    lower = f15['lower_is_better']
    metric = f15['metric']
    dataset = f15['dataset']

    print(f"\n{dataset} (metric={metric}, lower_is_better={lower}):", flush=True)
    print(f"  merge sources: " + ', '.join(f"{m}={s}"
                                           for m, s in sorted(f15['merge_sources'].items())),
          flush=True)

    # Baseline best-per-model (used as the reference for rel_loss)
    baseline_best: Dict[str, Optional[Dict[str, Any]]] = {
        m: _best_for_model(results, m, lower) for m in F16_MODELS
    }

    # Header row
    models_present = [m for m in F16_MODELS if baseline_best[m] is not None]
    if not models_present:
        print("  no usable models — skipping", flush=True)
        return {'dataset': dataset, 'skipped': True}

    print(f"  baseline best (model: score@budget):", flush=True)
    for m in models_present:
        b = baseline_best[m]
        print(f"    {m:10s}  {b['score_mean']:+.4f} @ b={b['budget']}", flush=True)

    out_schedules: Dict[str, Dict[str, Any]] = {}
    for sname, sched in SCHEDULES.items():
        max_ckpt = max(sched['checkpoints']) if sched['checkpoints'] else 0
        per_model: Dict[str, Dict[str, Any]] = {}
        for m in models_present:
            kept_for_m, n_dropped, n_evals_avoided = _apply_per_model_pruning(
                results, m, sched, lower)
            best_after = _best_for_model(kept_for_m, m, lower)
            base_best = baseline_best[m]
            if best_after is None:
                rel_loss = float('inf')
            elif abs(base_best['score_mean']) < 1e-12:
                # Avoid divide-by-zero on near-zero baseline; report absolute delta
                rel_loss = best_after['score_mean'] - base_best['score_mean']
                if lower:
                    rel_loss = -rel_loss
            else:
                delta = best_after['score_mean'] - base_best['score_mean']
                rel_loss = delta / abs(base_best['score_mean'])
                if lower:
                    rel_loss = -rel_loss

            # Total fold-evals for this model (baseline)
            m_total_evals = sum(len(r.get('fold_scores') or [])
                                for r in results if r.get('model') == m)
            pct_saved = (n_evals_avoided / m_total_evals * 100) \
                if m_total_evals else 0.0

            # Trivial preservation: best config sits at or below all checkpoints
            trivial = (max_ckpt > 0 and base_best['budget'] <= max_ckpt)
            passed = (not trivial) and (rel_loss <= PASS_THRESHOLD) \
                if max_ckpt > 0 else True
            per_model[m] = {
                'baseline_best_budget': base_best['budget'],
                'baseline_best_score': base_best['score_mean'],
                'pruned_best_score': best_after['score_mean'] if best_after else None,
                'rel_loss': rel_loss,
                'pct_evals_saved_for_this_model': pct_saved,
                'n_configs_dropped': n_dropped,
                'trivial_preservation': trivial,
                'passes': bool(passed),
            }
        out_schedules[sname] = {
            'schedule': sched,
            'per_model': per_model,
        }

    # Print table: rows = schedules, columns = models, cells = "loss% / saved%"
    # plus a trivial flag.
    print(f"\n  per-model (rel_loss / pct_evals_saved) — T = trivial preservation",
          flush=True)
    header = f"  {'schedule':16s}  " + "  ".join(f"{m:>15s}" for m in models_present)
    print(header, flush=True)
    for sname in SCHEDULES.keys():
        cells = []
        for m in models_present:
            entry = out_schedules[sname]['per_model'][m]
            triv = 'T' if entry['trivial_preservation'] else ' '
            if entry['rel_loss'] == float('inf'):
                cells.append(f"  {triv} INF / -    ")
            else:
                cells.append(
                    f"  {triv} {entry['rel_loss']*100:+5.2f}% / {entry['pct_evals_saved_for_this_model']:4.1f}%")
        print(f"  {sname:16s}  " + "".join(cells), flush=True)

    return {
        'dataset': dataset,
        'metric': metric,
        'lower_is_better': lower,
        'n_samples': f15.get('n_samples'),
        'merge_sources': f15.get('merge_sources'),
        'baseline_best_by_model': {
            m: {'score_mean': baseline_best[m]['score_mean'],
                'budget': baseline_best[m]['budget']}
            for m in models_present
        },
        'schedules': out_schedules,
    }


# ---------------------------------------------------------------------------
# Cross-dataset aggregation
# ---------------------------------------------------------------------------

def _cross_dataset_summary(per_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each (schedule, model), report:
        - n_total: number of (dataset) cells where this model has data
        - n_trivial: cells where the best budget <= max(checkpoint), so the
          schedule wasn't given a chance to fail. Inconclusive, not pass.
        - n_pass: cells where rel_loss <= PASS_THRESHOLD AND not trivial.
        - n_fail: cells where rel_loss > PASS_THRESHOLD.
        - max_rel_loss: worst non-trivial rel_loss for this (schedule, model).
        - avg_pct_saved_nontrivial: average compute saved on non-trivial cells.
    """
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sname in SCHEDULES.keys():
        summary[sname] = {}
        for m in F16_MODELS:
            cells = []
            for ds in per_dataset:
                if ds.get('skipped'):
                    continue
                entry = ds['schedules'].get(sname, {}).get('per_model', {}).get(m)
                if entry is None:
                    continue
                cells.append(entry)
            if not cells:
                continue
            n_total = len(cells)
            n_trivial = sum(1 for c in cells if c['trivial_preservation'])
            n_nontrivial = n_total - n_trivial
            nontrivial_cells = [c for c in cells if not c['trivial_preservation']]
            n_pass = sum(1 for c in nontrivial_cells
                         if c['rel_loss'] <= PASS_THRESHOLD)
            n_fail = n_nontrivial - n_pass
            max_loss = max((c['rel_loss'] for c in nontrivial_cells),
                           default=0.0)
            avg_save_nt = (
                sum(c['pct_evals_saved_for_this_model'] for c in nontrivial_cells)
                / n_nontrivial) if n_nontrivial else 0.0
            avg_save_all = (
                sum(c['pct_evals_saved_for_this_model'] for c in cells) / n_total)
            summary[sname][m] = {
                'n_total': n_total,
                'n_trivial': n_trivial,
                'n_nontrivial': n_nontrivial,
                'n_pass': n_pass,
                'n_fail': n_fail,
                'max_rel_loss_nontrivial': max_loss,
                'avg_pct_evals_saved_nontrivial': avg_save_nt,
                'avg_pct_evals_saved_all': avg_save_all,
            }
    return summary


def _print_summary(summary: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    print(f"\n{'='*78}", flush=True)
    print(f"Cross-dataset summary — pass = (rel_loss <= {PASS_THRESHOLD*100:.0f}%) "
          f"on non-trivial cells", flush=True)
    print(f"{'='*78}", flush=True)
    for m in F16_MODELS:
        # Show only models that have data in at least one schedule
        has_data = any(m in summary[s] for s in summary)
        if not has_data:
            continue
        print(f"\nModel: {m}", flush=True)
        print(f"  {'schedule':16s}  total  trivial  nontriv  pass  fail   max_loss  avg_saved_nontriv",
              flush=True)
        for sname in SCHEDULES.keys():
            s = summary[sname].get(m)
            if s is None:
                continue
            print(f"  {sname:16s}  "
                  f"{s['n_total']:5d}  "
                  f"{s['n_trivial']:7d}  "
                  f"{s['n_nontrivial']:7d}  "
                  f"{s['n_pass']:4d}  "
                  f"{s['n_fail']:4d}   "
                  f"{s['max_rel_loss_nontrivial']*100:+7.2f}%  "
                  f"{s['avg_pct_evals_saved_nontrivial']:14.1f}%",
                  flush=True)


def _universal_winners(summary: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Identify schedules that pass for every (model × dataset) non-trivial cell.

    Returns: {schedule_name: {'universal': bool, 'fails_for': [...], ...}}
    """
    out: Dict[str, Any] = {}
    for sname in SCHEDULES.keys():
        fails = []
        n_nontriv_total = 0
        n_pass_total = 0
        for m, s in summary.get(sname, {}).items():
            n_nontriv_total += s['n_nontrivial']
            n_pass_total += s['n_pass']
            if s['n_fail'] > 0:
                fails.append(f"{m}({s['n_fail']}/{s['n_nontrivial']})")
        out[sname] = {
            'universal_pass': len(fails) == 0 and n_nontriv_total > 0,
            'fails_for_models': fails,
            'n_pass_total': n_pass_total,
            'n_nontrivial_total': n_nontriv_total,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='F16: per-model pruning threshold sweep on f15 data.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (default: all f16 datasets).')
    parser.add_argument('-o', '--output', type=str,
                        default=str(F16_DIR / 'per_model_sweep.json'),
                        help='Output JSON path.')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else F16_DATASETS
    F16_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] F16 per-model pruning sweep",
          flush=True)
    print(f"  schedules: {list(SCHEDULES.keys())}", flush=True)
    print(f"  models:    {F16_MODELS}", flush=True)
    print(f"  pass threshold (rel_loss): {PASS_THRESHOLD*100:.1f}%", flush=True)

    per_dataset = []
    for ds in datasets:
        f15 = _load_f15(ds)
        if f15 is None:
            print(f"\n  {ds}: no f15 data, skip", flush=True)
            continue
        per_dataset.append(_sweep_dataset(f15))

    summary = _cross_dataset_summary(per_dataset)
    _print_summary(summary)
    universal = _universal_winners(summary)

    print(f"\n{'='*78}", flush=True)
    print("Universal-pass check (a schedule passes for EVERY model × dataset "
          "non-trivial cell)", flush=True)
    print(f"{'='*78}", flush=True)
    for sname, info in universal.items():
        if info['universal_pass']:
            print(f"  {sname:16s}  UNIVERSAL PASS  "
                  f"({info['n_pass_total']}/{info['n_nontrivial_total']} cells)",
                  flush=True)
        else:
            print(f"  {sname:16s}  fails: {info['fails_for_models']}  "
                  f"({info['n_pass_total']}/{info['n_nontrivial_total']} cells)",
                  flush=True)

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'experiment': 'f16_multistage_pruning',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'pass_threshold_rel_loss': PASS_THRESHOLD,
            'schedules': SCHEDULES,
            'per_dataset': per_dataset,
            'cross_dataset_summary': summary,
            'universal_pass': universal,
        }, f, indent=2, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] Wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
