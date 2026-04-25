#!/usr/bin/env python3
"""F16: Multi-stage adaptive pruning calibration — offline simulation.

Reuses the per-(config, fold) score data captured by F15 to simulate
various multi-stage pruning schedules WITHOUT running new searches.

A "schedule" is a dict:
    {
        # Budgets at which to checkpoint and drop tail configs.
        # Empty = single end-of-fold-0 prune only (= current behaviour).
        'checkpoints': [25, 50, 100],

        # Within each (model, evaluator) group at a checkpoint, drop
        # bottom X% by fold-0 score. Dropped configs do no further
        # fold-0 work AND are excluded from full-eval.
        'tail_drop_per_checkpoint': 0.5,

        # End-of-fold-0 strategy pruning threshold (current top-3 logic).
        'final_strategy_threshold': 0.15,

        # Floor: never prune a model below this many surviving configs.
        # Guarantees best_by_model[m] exists for every m.
        'min_configs_per_model': 1,
    }

Per-model overrides (optional) merge on top of the global schedule for the
named models only.

For each (dataset, schedule), this script reports:
- Wall-time saving estimate (fraction of fold-evals avoided across folds 1-N)
- Result quality vs unpruned baseline (best_by_model score deltas per model)
- Top-K stability (fraction of unpruned top-K configs that survive pruning)
- Per-model survivor count at end of fold-0

Why offline: F15 ran the full unpruned search on these datasets and saved
fold_scores[0..n-1] for every config. From that we can replay any pruning
schedule and ask "would the surviving configs have produced a different
best_by_model?" without re-evaluating anything. Saves days of compute.

Why SVM and not GP: F15 didn't include GP. SVM is the closest cost-shape
analogue (kernel-based, expensive at high budgets). If multi-stage pruning
preserves SVM rankings, the methodology is sound; we then apply it to GP
with confidence that the algorithm — not a GP-specific NaN pathology — is
what's being tested.

Output: tests/results/framework/f16_multistage_pruning/calibration_summary.json

Usage:
    python tests/experiments/framework/f16_multistage_pruning.py
    python tests/experiments/framework/f16_multistage_pruning.py --dataset caco2
    python tests/experiments/framework/f16_multistage_pruning.py --schedule aggressive
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
F15_DIR = REPO_ROOT / 'tests' / 'results' / 'framework' / 'f15_adaptive_pruning'
F16_DIR = REPO_ROOT / 'tests' / 'results' / 'framework' / 'f16_multistage_pruning'

# Datasets to evaluate: span small antibody and mid-size molecular,
# chosen to cover the regimes where multi-stage pruning would actually
# bite (i.e. SVM/MLP at high budgets is expensive).
F16_DATASETS = [
    'flab_pk_jain_fcrnrt',  # small antibody, n≈241
    'abagym_EGFR',          # small antibody (no aug)
    'caco2',                # mid molecular, n≈2158
    'herg_ki',              # mid molecular
    'logd',                 # mid molecular
    'buchwald_hartwig',     # reaction, n≈3955
]

# Candidate schedules to compare. baseline = current single-stage pruning.
# Each entry will be applied to every dataset; per-model overrides applied
# only on top of the schedule named in the override block (see below).
SCHEDULES: Dict[str, Dict[str, Any]] = {
    'baseline': {
        'checkpoints': [],
        'tail_drop_per_checkpoint': 0.0,
        'final_strategy_threshold': 0.15,
        'min_configs_per_model': 1,
    },
    'light': {
        'checkpoints': [50],
        'tail_drop_per_checkpoint': 0.25,
        'final_strategy_threshold': 0.15,
        'min_configs_per_model': 1,
    },
    'medium': {
        'checkpoints': [25, 50],
        'tail_drop_per_checkpoint': 0.50,
        'final_strategy_threshold': 0.15,
        'min_configs_per_model': 1,
    },
    'aggressive': {
        'checkpoints': [25, 50, 100],
        'tail_drop_per_checkpoint': 0.50,
        'final_strategy_threshold': 0.15,
        'min_configs_per_model': 1,
    },
    'very_aggressive': {
        'checkpoints': [10, 25, 50],
        'tail_drop_per_checkpoint': 0.75,
        'final_strategy_threshold': 0.15,
        'min_configs_per_model': 1,
    },
}

# Per-model overrides (applied to the named global schedule). Tested separately
# from the global schedule levels above.
MODEL_OVERRIDES_TO_TEST: Dict[str, Dict[str, Dict[str, Any]]] = {
    'svm_aggressive_only': {
        'svm': {'checkpoints': [25, 50, 100], 'tail_drop_per_checkpoint': 0.75},
    },
    'mlp_aggressive_only': {
        'mlp': {'checkpoints': [25, 50, 100], 'tail_drop_per_checkpoint': 0.75},
    },
    'svm_mlp_aggressive': {
        'svm': {'checkpoints': [25, 50, 100], 'tail_drop_per_checkpoint': 0.75},
        'mlp': {'checkpoints': [25, 50, 100], 'tail_drop_per_checkpoint': 0.75},
    },
}


# ---------------------------------------------------------------------------
# Schedule application — pure simulation against f15 data
# ---------------------------------------------------------------------------

def _resolve_model_schedule(
    model_name: str,
    global_sched: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Merge per-model override on top of global schedule for one model."""
    sched = dict(global_sched)
    if overrides and model_name in overrides:
        for k, v in overrides[model_name].items():
            sched[k] = v
    return sched


def _apply_pruning(
    results: List[Dict[str, Any]],
    global_sched: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]],
    lower_is_better: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    """Simulate a pruning schedule against f15 result rows.

    Args:
        results: list of f15 result dicts. Each must have fields
            'strategy', 'importance', 'evaluator', 'model', 'budget',
            'fold_scores' (list of per-fold scores, NaN allowed).
        global_sched: schedule dict.
        overrides: optional per-model override dict.
        lower_is_better: metric direction.

    Returns:
        survivors: list of result dicts that pass the schedule (i.e. would
            have been evaluated on all folds).
        per_model_dropped: {model_name: n_dropped} for accounting.
        n_fold0_evals_avoided: total fold-evals avoided beyond fold 0
            by the mid-flight pruning steps (used as a wall-time proxy).
    """
    # Group results by (model, evaluator). Within each group, ascending budget
    # is the order in which configs would be encountered if we run the search
    # budget-first.
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_group[(r['model'], r['evaluator'])].append(r)
    for k in by_group:
        by_group[k].sort(key=lambda r: (r['budget'], r['strategy'], r['importance']))

    per_model_dropped: Dict[str, int] = defaultdict(int)
    n_fold0_evals_avoided = 0

    surviving_keys = set()  # set of (strategy, importance, evaluator, model, budget)

    for (model, evaluator), group_results in by_group.items():
        sched = _resolve_model_schedule(model, global_sched, overrides)
        checkpoints = sorted(sched.get('checkpoints', []))
        tail_drop = float(sched.get('tail_drop_per_checkpoint', 0.0))
        floor = int(sched.get('min_configs_per_model', 1))

        # Track which configs are still "alive" at each point in the schedule
        alive = list(group_results)

        # Simulate checkpoint pruning
        # At each checkpoint b: among configs with budget <= b that are still
        # alive, rank by fold-0 score, drop bottom (tail_drop * n_alive_so_far).
        # All higher-budget configs whose (strategy, importance) pair was in the
        # dropped set are also pruned (since we'd never run them).
        budgets_seen_so_far = []
        for ckpt in checkpoints:
            # Configs at or below this checkpoint that are alive
            up_to_ckpt = [r for r in alive if r['budget'] <= ckpt]
            if not up_to_ckpt:
                continue

            # Rank by fold-0 score (NaN sinks to bottom)
            def f0_score(r):
                fs = r.get('fold_scores') or []
                if not fs:
                    return float('inf') if lower_is_better else float('-inf')
                v = fs[0]
                if v != v:  # NaN
                    return float('inf') if lower_is_better else float('-inf')
                return v

            up_to_ckpt_sorted = sorted(up_to_ckpt, key=f0_score,
                                       reverse=not lower_is_better)
            n = len(up_to_ckpt_sorted)
            n_keep = max(floor, int(round(n * (1 - tail_drop))))
            kept_strats_imps = {(r['strategy'], r['importance'])
                                for r in up_to_ckpt_sorted[:n_keep]}

            new_alive = []
            for r in alive:
                pair = (r['strategy'], r['importance'])
                if pair in kept_strats_imps:
                    new_alive.append(r)
                else:
                    per_model_dropped[model] += 1
                    # Wall-time savings proxy: any fold-eval beyond fold 0
                    # that would have been done is now avoided. f15 reports
                    # fold_scores so we count len(fold_scores)-1 avoided.
                    fs = r.get('fold_scores') or []
                    n_folds_total = len(fs)
                    if r['budget'] > ckpt:
                        # Also avoid the fold-0 eval (config never reached).
                        n_fold0_evals_avoided += n_folds_total
                    else:
                        # Already did fold 0; saved folds 1..N-1.
                        n_fold0_evals_avoided += max(0, n_folds_total - 1)
            alive = new_alive

        # End-of-fold-0 strategy pruning (top-3 in <X% of groups).
        # We apply this as a SECOND stage on whatever survives mid-flight.
        # NOTE: this matches the existing find_best_hybrid logic; we just
        # apply it after the mid-flight pruning instead of as the only step.
        for r in alive:
            surviving_keys.add((r['strategy'], r['importance'], r['evaluator'],
                                r['model'], r['budget']))

    # Final strategy pruning (top-3 within each (imp, eval, model, bgt) group)
    f0_groups: Dict[Tuple[str, str, str, int], List[Tuple[str, float]]] = \
        defaultdict(list)
    for r in results:
        key = (r['strategy'], r['importance'], r['evaluator'],
               r['model'], r['budget'])
        if key not in surviving_keys:
            continue
        fs = r.get('fold_scores') or []
        f0 = fs[0] if fs and fs[0] == fs[0] else (
            float('inf') if lower_is_better else float('-inf'))
        gk = (r['importance'], r['evaluator'], r['model'], r['budget'])
        f0_groups[gk].append((r['strategy'], f0))

    threshold_frac = float(global_sched.get('final_strategy_threshold', 0.15))
    n_groups = len(f0_groups)
    threshold = n_groups * threshold_frac
    top3_counts: Dict[str, int] = defaultdict(int)
    for gk, entries in f0_groups.items():
        ranked = sorted(entries, key=lambda x: x[1], reverse=not lower_is_better)
        for s, _ in ranked[:3]:
            top3_counts[s] += 1
    pruned_strategies = {s for s in top3_counts
                         if top3_counts[s] < threshold}

    # Apply strategy pruning on top of mid-flight pruning
    final_keys = {k for k in surviving_keys if k[0] not in pruned_strategies}

    survivors = [r for r in results
                 if (r['strategy'], r['importance'], r['evaluator'],
                     r['model'], r['budget']) in final_keys]

    return survivors, dict(per_model_dropped), n_fold0_evals_avoided


# ---------------------------------------------------------------------------
# Quality metrics — what we lose / keep
# ---------------------------------------------------------------------------

def _best_by_model(
    results: List[Dict[str, Any]],
    lower_is_better: bool,
) -> Dict[str, Dict[str, Any]]:
    """For each model, find the single best config (by 5-fold mean score)."""
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_model[r['model']].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    cmp = min if lower_is_better else max
    for m, rows in by_model.items():
        valid = [r for r in rows
                 if r.get('score_mean') is not None
                 and r['score_mean'] == r['score_mean']]
        if not valid:
            continue
        out[m] = cmp(valid, key=lambda r: r['score_mean'])
    return out


def _topk_overlap(
    baseline: List[Dict[str, Any]],
    pruned: List[Dict[str, Any]],
    k: int,
    lower_is_better: bool,
) -> float:
    """Fraction of baseline top-K configs that survive pruning."""
    cmp_key = (lambda r: r['score_mean']) if lower_is_better else \
              (lambda r: -r['score_mean'])
    valid_b = [r for r in baseline
               if r.get('score_mean') == r.get('score_mean')]
    if not valid_b:
        return 0.0
    top_b = sorted(valid_b, key=cmp_key)[:k]
    top_b_keys = {(r['strategy'], r['importance'], r['evaluator'],
                   r['model'], r['budget']) for r in top_b}
    pruned_keys = {(r['strategy'], r['importance'], r['evaluator'],
                    r['model'], r['budget']) for r in pruned}
    if not top_b_keys:
        return 0.0
    return len(top_b_keys & pruned_keys) / len(top_b_keys)


def _evaluate_schedule(
    results: List[Dict[str, Any]],
    sched: Dict[str, Any],
    overrides: Optional[Dict[str, Dict[str, Any]]],
    lower_is_better: bool,
) -> Dict[str, Any]:
    """Run one schedule, compute summary stats vs baseline."""
    survivors, per_model_dropped, n_evals_avoided = _apply_pruning(
        results, sched, overrides, lower_is_better)

    baseline_best = _best_by_model(results, lower_is_better)
    pruned_best = _best_by_model(survivors, lower_is_better)

    score_deltas: Dict[str, Optional[float]] = {}
    rel_deltas: Dict[str, Optional[float]] = {}
    for m in baseline_best:
        b_score = baseline_best[m]['score_mean']
        if m not in pruned_best:
            score_deltas[m] = None  # model entirely lost — should never happen with floor=1
            rel_deltas[m] = None
            continue
        p_score = pruned_best[m]['score_mean']
        delta = p_score - b_score
        score_deltas[m] = float(delta)
        if abs(b_score) > 1e-9:
            rel = (delta / abs(b_score))
            if lower_is_better:
                rel = -rel  # larger relative delta = worse
            rel_deltas[m] = float(rel)
        else:
            rel_deltas[m] = None

    n_total_configs = len(results)
    n_survived_configs = len(survivors)
    pct_dropped_configs = (1 - n_survived_configs / n_total_configs) * 100 \
                          if n_total_configs else 0.0

    n_total_evals = sum(len(r.get('fold_scores') or []) for r in results)
    pct_evals_avoided = (n_evals_avoided / n_total_evals * 100) if n_total_evals else 0.0

    top10_overlap = _topk_overlap(results, survivors, 10, lower_is_better)
    top3_overlap = _topk_overlap(results, survivors, 3, lower_is_better)

    return {
        'n_configs_total': n_total_configs,
        'n_configs_survived': n_survived_configs,
        'pct_configs_dropped': float(pct_dropped_configs),
        'n_evals_total': n_total_evals,
        'n_evals_avoided': n_evals_avoided,
        'pct_evals_avoided': float(pct_evals_avoided),
        'top3_overlap_with_baseline': float(top3_overlap),
        'top10_overlap_with_baseline': float(top10_overlap),
        'per_model_score_delta': score_deltas,
        'per_model_relative_delta': rel_deltas,
        'per_model_dropped': per_model_dropped,
        'baseline_best_by_model': {
            m: {'score_mean': r['score_mean'], 'budget': r['budget']}
            for m, r in baseline_best.items()
        },
        'pruned_best_by_model': {
            m: {'score_mean': r['score_mean'], 'budget': r['budget']}
            for m, r in pruned_best.items()
        },
    }


# ---------------------------------------------------------------------------
# IO + main
# ---------------------------------------------------------------------------

_F15_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'mlp', 'lightgbm']


def _load_f15(dataset_name: str, model_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load f15 results for a dataset, merging per-model split files when present.

    F15 was originally run as one all-6-models job per dataset, writing
    `{dataset}.json`. Many of those timed out before completion, and were
    re-run as per-model jobs writing `{dataset}_{model}.json`. This function:

      1. If `model_filter` is set, reads ONLY that single per-model file.
      2. Otherwise, prefers per-model files (complete per model) over the
         combined file (often incomplete). For each of the 6 standard
         F15 models:
             - if `{dataset}_{model}.json` exists AND is complete → use it
             - else if combined `{dataset}.json` has results for this model
               AND combined is marked complete → use the combined slice
             - else SKIP the model (log a warning)
      3. If the dataset has zero usable models, returns None.

    Returns the standard F15-shaped dict (`{'results': [...], 'metric': ...,
    'lower_is_better': ..., 'n_folds': ..., ...}`) constructed by merging.
    """
    if model_filter:
        path = F15_DIR / f'{dataset_name}_{model_filter}.json'
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        if not data.get('complete'):
            print(f"    WARNING: {dataset_name}_{model_filter} incomplete: "
                  f"{data.get('progress')}", flush=True)
        return data

    combined_path = F15_DIR / f'{dataset_name}.json'
    combined_data: Optional[Dict[str, Any]] = None
    combined_complete = False
    if combined_path.exists():
        with open(combined_path) as f:
            combined_data = json.load(f)
        combined_complete = bool(combined_data.get('complete'))

    # Bucket combined results by model for fallback
    combined_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in _F15_MODELS}
    if combined_data is not None:
        for r in combined_data.get('results', []):
            m = r.get('model')
            if m in combined_by_model:
                combined_by_model[m].append(r)

    merged_results: List[Dict[str, Any]] = []
    sources: Dict[str, str] = {}    # model -> 'per-model' | 'combined' | 'skipped'
    meta_seed: Optional[Dict[str, Any]] = None

    for model in _F15_MODELS:
        per_model_path = F15_DIR / f'{dataset_name}_{model}.json'
        if per_model_path.exists():
            with open(per_model_path) as f:
                pm_data = json.load(f)
            if pm_data.get('complete'):
                merged_results.extend(pm_data.get('results', []))
                sources[model] = 'per-model'
                meta_seed = meta_seed or pm_data
                continue
            else:
                # per-model file exists but incomplete — fall through to combined
                pass
        # Fallback to combined slice (only when combined IS complete)
        if combined_complete and combined_by_model[model]:
            merged_results.extend(combined_by_model[model])
            sources[model] = 'combined'
            meta_seed = meta_seed or combined_data
        else:
            sources[model] = 'skipped'

    if not merged_results:
        return None

    # Borrow metric/lower_is_better/n_folds/n_samples from whichever source we used
    base = meta_seed or combined_data or {}
    out = {
        'experiment': 'f15_adaptive_pruning_validation',
        'dataset': dataset_name,
        'domain': base.get('domain'),
        'metric': base.get('metric'),
        'lower_is_better': base.get('lower_is_better'),
        'n_samples': base.get('n_samples'),
        'n_reps': base.get('n_reps'),
        'n_folds': base.get('n_folds'),
        'config': base.get('config'),
        'results': merged_results,
        'complete': True,
        'merge_sources': sources,
    }
    n_per_model = {m: sum(1 for r in merged_results if r.get('model') == m)
                   for m in _F15_MODELS}
    print(f"    {dataset_name}: merged from {sources} "
          f"(configs/model: {n_per_model})", flush=True)
    skipped = [m for m, s in sources.items() if s == 'skipped']
    if skipped:
        print(f"    WARNING: missing models for {dataset_name}: {skipped} — "
              f"these models will not appear in best_by_model and any schedule "
              f"that depends on them is invalid for this dataset.", flush=True)
    return out


def _evaluate_dataset(
    dataset_name: str,
    f15_data: Dict[str, Any],
    schedules: Dict[str, Dict[str, Any]],
    model_overrides: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run all schedules + override variants on one dataset's f15 data."""
    results = f15_data['results']
    lower_is_better = bool(f15_data.get('lower_is_better', False))
    metric = f15_data.get('metric', '?')
    print(f"\n  {dataset_name}: {len(results)} configs, "
          f"metric={metric} (lower={lower_is_better})", flush=True)

    out: Dict[str, Any] = {
        'dataset': dataset_name,
        'metric': metric,
        'lower_is_better': lower_is_better,
        'n_samples': f15_data.get('n_samples'),
        'n_reps': f15_data.get('n_reps'),
        'n_folds': f15_data.get('n_folds'),
        'schedules': {},
        'overrides': {},
    }

    # Plain schedules (no per-model override)
    for sched_name, sched in schedules.items():
        summary = _evaluate_schedule(results, sched, None, lower_is_better)
        out['schedules'][sched_name] = {'schedule': sched, **summary}
        worst_rel = max((d for d in summary['per_model_relative_delta'].values()
                         if d is not None), default=0.0)
        print(f"    {sched_name:18s}  "
              f"dropped={summary['pct_configs_dropped']:5.1f}% "
              f"evals_avoided={summary['pct_evals_avoided']:5.1f}% "
              f"top3_keep={summary['top3_overlap_with_baseline']:.2f} "
              f"worst_rel_loss={worst_rel*100:+.2f}%",
              flush=True)

    # Per-model overrides (each applied on top of global 'baseline' schedule
    # so we isolate the contribution of the override)
    for ov_name, ov in model_overrides.items():
        summary = _evaluate_schedule(
            results, schedules['baseline'], ov, lower_is_better)
        out['overrides'][ov_name] = {
            'global_schedule': 'baseline',
            'overrides': ov,
            **summary,
        }
        worst_rel = max((d for d in summary['per_model_relative_delta'].values()
                         if d is not None), default=0.0)
        print(f"    [override] {ov_name:22s}  "
              f"dropped={summary['pct_configs_dropped']:5.1f}% "
              f"evals_avoided={summary['pct_evals_avoided']:5.1f}% "
              f"top3_keep={summary['top3_overlap_with_baseline']:.2f} "
              f"worst_rel_loss={worst_rel*100:+.2f}%",
              flush=True)

    return out


def _recommend_schedule(per_dataset_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the most aggressive schedule that meets quality criteria
    on every dataset.

    Acceptance: per-model relative loss <= 2% AND top-3 overlap >= 0.66
    on every dataset.
    """
    sched_names = list(SCHEDULES.keys())

    # For each schedule, check pass/fail on every dataset
    scoreboard: Dict[str, Dict[str, Any]] = {}
    for sname in sched_names:
        passes_all = True
        worst_loss = 0.0
        avg_savings = 0.0
        for ds_summary in per_dataset_summary:
            entry = ds_summary['schedules'].get(sname)
            if not entry:
                passes_all = False
                continue
            rels = [v for v in entry['per_model_relative_delta'].values()
                    if v is not None]
            top3 = entry['top3_overlap_with_baseline']
            wl = max(rels) if rels else 0.0
            worst_loss = max(worst_loss, wl)
            if wl > 0.02 or top3 < 0.66:
                passes_all = False
            avg_savings += entry['pct_evals_avoided']
        avg_savings /= max(1, len(per_dataset_summary))
        scoreboard[sname] = {
            'passes': passes_all,
            'worst_relative_loss': worst_loss,
            'avg_pct_evals_avoided': avg_savings,
        }

    # Aggressiveness order, most → least
    order = ['very_aggressive', 'aggressive', 'medium', 'light', 'baseline']
    chosen = next((s for s in order if scoreboard.get(s, {}).get('passes')),
                  'baseline')

    return {
        'recommended_global_schedule': chosen,
        'scoreboard': scoreboard,
        'acceptance_criteria': {
            'max_per_model_relative_loss': 0.02,
            'min_top3_overlap': 0.66,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='F16: Multi-stage adaptive pruning calibration (offline).')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (default: all six F16 datasets).')
    parser.add_argument('--model-filter', type=str, default=None,
                        help='Use F15 single-model run if available '
                             '(e.g. "svm" → reads f15/{dataset}_svm.json).')
    parser.add_argument('--schedule', type=str, default=None,
                        help='Single schedule by name (default: all).')
    parser.add_argument('-o', '--output', type=str,
                        default=str(F16_DIR / 'calibration_summary.json'),
                        help='Output JSON path.')
    args = parser.parse_args()

    if args.schedule:
        if args.schedule not in SCHEDULES:
            print(f"Unknown schedule '{args.schedule}'. "
                  f"Choose from: {list(SCHEDULES.keys())}")
            sys.exit(1)
        scheds = {args.schedule: SCHEDULES[args.schedule]}
    else:
        scheds = SCHEDULES

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = F16_DATASETS

    F16_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] F16 multi-stage pruning calibration",
          flush=True)
    print(f"  F15 source: {F15_DIR}", flush=True)
    print(f"  Datasets: {datasets}", flush=True)
    print(f"  Schedules: {list(scheds.keys())}", flush=True)
    print(f"  Per-model overrides: {list(MODEL_OVERRIDES_TO_TEST.keys())}",
          flush=True)

    per_dataset = []
    missing = []
    for ds in datasets:
        f15_data = _load_f15(ds, model_filter=args.model_filter)
        if f15_data is None:
            print(f"  {ds}: f15 data missing — skip", flush=True)
            missing.append(ds)
            continue
        per_dataset.append(
            _evaluate_dataset(ds, f15_data, scheds, MODEL_OVERRIDES_TO_TEST))

    if not per_dataset:
        print("\nNo f15 data found. Run f15 first or check F15_DIR path.",
              flush=True)
        sys.exit(1)

    recommendation = _recommend_schedule(per_dataset) \
        if not args.schedule else None

    output = {
        'experiment': 'f16_multistage_pruning',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'f15_source': str(F15_DIR),
        'datasets_evaluated': [d['dataset'] for d in per_dataset],
        'datasets_missing': missing,
        'schedules_tested': list(scheds.keys()),
        'overrides_tested': list(MODEL_OVERRIDES_TO_TEST.keys()),
        'per_dataset': per_dataset,
        'recommendation': recommendation,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. Wrote {out_path}", flush=True)
    if recommendation:
        chosen = recommendation['recommended_global_schedule']
        sb = recommendation['scoreboard'][chosen]
        print(f"  Recommended: {chosen}", flush=True)
        print(f"    Worst per-model relative loss: "
              f"{sb['worst_relative_loss']*100:+.2f}%", flush=True)
        print(f"    Avg fold-evals avoided across datasets: "
              f"{sb['avg_pct_evals_avoided']:.1f}%", flush=True)


if __name__ == '__main__':
    main()
