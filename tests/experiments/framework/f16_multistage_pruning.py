#!/usr/bin/env python3
"""F16: per-model threshold sweep for multi-stage adaptive pruning.

The pruning algorithm has several tunable parameters. This experiment
holds the f15-validated ones fixed and sweeps the new ones as a grid.

Fixed (NOT swept here):
    final_strategy_threshold = 0.15
        End-of-fold-0 strategy prune. Already validated by F15.
    min_configs_per_model = 1
        Floor: never prune a model below 1 surviving config so
        best_by_model[m] is always populated. Safety, not a threshold to
        tune.

Swept (the actual f16 question):
    checkpoint_config:
        At which budgets do we look at fold-0 scores and prune the bottom
        of each (model, evaluator) group?
        Candidates:
            [25]              - single, early
            [50]              - single, late-low
            [100]             - single, mid
            [25, 50]          - two-stage early
            [25, 50, 100]     - three-stage progressive
            [10, 25, 50]      - three-stage very early

    tail_drop:
        At each checkpoint, drop the bottom X fraction of (alloc, imp)
        pairs by fold-0 score. Same value applied at every checkpoint.
        Candidates:
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

Per (model, dataset, checkpoint_config, tail_drop) we compute:
    rel_loss = (pruned_best_for_model - baseline_best_for_model)
               / |baseline_best_for_model|, sign-corrected for
               lower-is-better. % worse the model's best fold-mean score
               is after applying these thresholds.
    pct_saved = pruned-out fold-evals / total fold-evals for this model.

A cell PASSES if rel_loss <= PASS_THRESHOLD (default 2%) AND it's not
trivial-preservation (best budget > max checkpoint — i.e. the schedule
was actually given a chance to fail).

Output (printed + JSON):
  1. Per-dataset, per-checkpoint-config tables: rows = tail_drop, cols
     = models, cells = `loss% / saved%` plus T marker for trivial cells.
  2. Per-model summary: for each model, what's the highest tail_drop
     value that passes on every dataset at each checkpoint_config?
  3. Universal-pass check: are there any (ckpt, tail_drop) combos where
     ALL models pass on ALL datasets?
"""

import argparse
import json
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

F16_MODELS = ['rf', 'xgboost', 'ridge', 'svm', 'dnn', 'lightgbm']

# The two swept dimensions.
CHECKPOINT_CONFIGS: Dict[str, List[int]] = {
    'ckpt_25':              [25],
    'ckpt_50':              [50],
    'ckpt_100':             [100],
    'ckpt_25_50':           [25, 50],
    'ckpt_10_25_50':        [10, 25, 50],
    'ckpt_25_50_100':       [25, 50, 100],
}

TAIL_DROPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

PASS_THRESHOLD = 0.02   # rel_loss must be <= 2% to pass


# ---------------------------------------------------------------------------
# Loading f15 data: prefer per-model files, fall back to combined slice
# ---------------------------------------------------------------------------

def _load_f15(dataset_name: str) -> Optional[Dict[str, Any]]:
    combined_path = F15_DIR / f'{dataset_name}.json'
    combined: Optional[Dict[str, Any]] = None
    combined_complete = False
    if combined_path.exists():
        with open(combined_path) as f:
            combined = json.load(f)
        combined_complete = bool(combined.get('complete'))
    by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in F16_MODELS}
    if combined is not None:
        for r in combined.get('results', []):
            m = r.get('model')
            if m in by_model:
                by_model[m].append(r)
    merged: List[Dict[str, Any]] = []
    sources: Dict[str, str] = {}
    seed: Optional[Dict[str, Any]] = None
    for m in F16_MODELS:
        per = F15_DIR / f'{dataset_name}_{m}.json'
        if per.exists():
            with open(per) as f:
                pm = json.load(f)
            if pm.get('complete'):
                merged.extend(pm.get('results', []))
                sources[m] = 'per-model'
                seed = seed or pm
                continue
        if combined_complete and by_model[m]:
            merged.extend(by_model[m])
            sources[m] = 'combined'
            seed = seed or combined
        else:
            sources[m] = 'skipped'
    if not merged:
        return None
    base = seed or combined or {}
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

def _f0_score(r: Dict[str, Any], lower: bool) -> float:
    fs = r.get('fold_scores') or []
    if not fs:
        return float('inf') if lower else float('-inf')
    v = fs[0]
    if v != v:
        return float('inf') if lower else float('-inf')
    return v


def _apply_for_model(
    results: List[Dict[str, Any]],
    model: str,
    checkpoints: List[int],
    tail_drop: float,
    lower: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply pruning to model-M configs only. Return (kept, n_evals_avoided)."""
    model_rows = [r for r in results if r.get('model') == model]
    if not model_rows or tail_drop <= 0 or not checkpoints:
        return model_rows, 0
    by_eval: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in model_rows:
        by_eval[r['evaluator']].append(r)
    kept: List[Dict[str, Any]] = []
    n_evals_avoided = 0
    for ev, group in by_eval.items():
        alive = list(group)
        for ckpt in sorted(checkpoints):
            up_to = [r for r in alive if r['budget'] <= ckpt]
            if not up_to:
                continue
            up_sorted = sorted(up_to, key=lambda r: _f0_score(r, lower),
                               reverse=not lower)
            n = len(up_sorted)
            n_keep = max(1, int(round(n * (1 - tail_drop))))
            keep_pairs = {(r['strategy'], r['importance'])
                          for r in up_sorted[:n_keep]}
            new_alive = []
            for r in alive:
                pair = (r['strategy'], r['importance'])
                if pair in keep_pairs:
                    new_alive.append(r)
                else:
                    fs = r.get('fold_scores') or []
                    n_folds = len(fs)
                    if r['budget'] > ckpt:
                        n_evals_avoided += n_folds
                    else:
                        n_evals_avoided += max(0, n_folds - 1)
            alive = new_alive
        kept.extend(alive)
    return kept, n_evals_avoided


def _best_for_model(rows: List[Dict[str, Any]], model: str,
                    lower: bool) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows
                  if r.get('model') == model
                  and r.get('score_mean') is not None
                  and r['score_mean'] == r['score_mean']]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r['score_mean']) if lower \
        else max(candidates, key=lambda r: r['score_mean'])


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _evaluate_cell(
    results: List[Dict[str, Any]],
    model: str,
    checkpoints: List[int],
    tail_drop: float,
    lower: bool,
    baseline_best: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute rel_loss + pct_saved for one (model, ckpt, tail_drop) cell.

    Convention: rel_loss POSITIVE means pruned is WORSE than baseline (a
    loss). Negative would mean pruned is somehow better (numerical noise
    from non-monotonic search, generally near zero).

    For lower-is-better (rmse): worse = larger value, so
        rel_loss = (pruned - baseline) / |baseline|   (positive when worse)
    For higher-is-better (spearman): worse = smaller value, so
        rel_loss = (baseline - pruned) / |baseline|   (positive when worse)

    The pass criterion `rel_loss <= PASS_THRESHOLD` then reads as
    "loss is at most PASS_THRESHOLD".
    """
    kept, n_evals_avoided = _apply_for_model(
        results, model, checkpoints, tail_drop, lower)
    after = _best_for_model(kept, model, lower)
    base = baseline_best
    if after is None:
        rel_loss = float('inf')
    elif abs(base['score_mean']) < 1e-12:
        # Degenerate baseline; report absolute delta with worse→positive sign
        delta_signed = after['score_mean'] - base['score_mean']
        rel_loss = delta_signed if lower else -delta_signed
    else:
        delta_signed = after['score_mean'] - base['score_mean']
        # Worse: delta_signed > 0 for lower-better, delta_signed < 0 for higher-better.
        # Flip for higher-better so positive = worse uniformly.
        worse_magnitude = delta_signed if lower else -delta_signed
        rel_loss = worse_magnitude / abs(base['score_mean'])
    total_evals = sum(len(r.get('fold_scores') or [])
                      for r in results if r.get('model') == model)
    pct_saved = (n_evals_avoided / total_evals * 100) if total_evals else 0.0
    max_ckpt = max(checkpoints) if checkpoints else 0
    trivial = (max_ckpt > 0 and base['budget'] <= max_ckpt)
    return {
        'rel_loss': rel_loss,
        'pct_saved': pct_saved,
        'trivial_preservation': trivial,
        'pruned_best_score': after['score_mean'] if after else None,
        'pruned_best_budget': after['budget'] if after else None,
    }


def _sweep_dataset(f15: Dict[str, Any]) -> Dict[str, Any]:
    results = f15['results']
    lower = f15['lower_is_better']
    metric = f15['metric']
    dataset = f15['dataset']

    print(f"\n{dataset}  (metric={metric}, lower_is_better={lower})", flush=True)
    print(f"  merge sources: " + ', '.join(f"{m}={s}"
                                           for m, s in sorted(f15['merge_sources'].items())),
          flush=True)

    baseline = {m: _best_for_model(results, m, lower) for m in F16_MODELS}
    models_present = [m for m in F16_MODELS if baseline[m] is not None]
    if not models_present:
        return {'dataset': dataset, 'skipped': True}

    print(f"  baseline best (score@budget):  " +
          ', '.join(f"{m}={baseline[m]['score_mean']:+.4f}@{baseline[m]['budget']}"
                    for m in models_present), flush=True)

    out: Dict[str, Any] = {
        'dataset': dataset,
        'metric': metric,
        'lower_is_better': lower,
        'merge_sources': f15['merge_sources'],
        'baseline_best_by_model': {
            m: {'score_mean': baseline[m]['score_mean'],
                'budget': baseline[m]['budget']}
            for m in models_present
        },
        'cells': {},  # ckpt_name -> tail_drop -> model -> {rel_loss, pct_saved, ...}
    }

    for ckpt_name, ckpts in CHECKPOINT_CONFIGS.items():
        out['cells'][ckpt_name] = {}
        # Print header
        print(f"\n  checkpoint = {ckpts}  (T = trivial preservation, best≤max_ckpt)",
              flush=True)
        header = f"  {'tail_drop':>10s}  " + "  ".join(f"{m:>14s}" for m in models_present)
        print(header, flush=True)
        for td in TAIL_DROPS:
            row_cells = {}
            for m in models_present:
                cell = _evaluate_cell(results, m, ckpts, td, lower, baseline[m])
                row_cells[m] = cell
            out['cells'][ckpt_name][f"{td:.1f}"] = row_cells

            # Print row
            chunks = []
            for m in models_present:
                c = row_cells[m]
                T = 'T' if c['trivial_preservation'] else ' '
                if c['rel_loss'] == float('inf'):
                    chunks.append(f"  {T} INF/-       ")
                else:
                    chunks.append(
                        f"  {T} {c['rel_loss']*100:+5.2f}%/{c['pct_saved']:4.1f}%")
            print(f"  {td:>10.1f}  " + "".join(chunks), flush=True)

    return out


# ---------------------------------------------------------------------------
# Cross-dataset analysis: per-model max safe tail_drop
# ---------------------------------------------------------------------------

def _max_safe_tail_drop_per_model(
    per_dataset: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """For each (model, ckpt_config), find the largest tail_drop where
    rel_loss <= PASS_THRESHOLD on every NON-TRIVIAL (dataset) cell.

    Returns {ckpt_name: {model: {'max_safe_tail_drop': float|None,
                                  'datasets_used': [...],
                                  'avg_pct_saved_at_max': float}}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for ckpt_name in CHECKPOINT_CONFIGS:
        out[ckpt_name] = {}
        for m in F16_MODELS:
            # Walk tail_drop values from highest to lowest; first one that
            # passes on every non-trivial dataset wins.
            max_safe: Optional[float] = None
            datasets_used: List[str] = []
            avg_saved = 0.0
            for td in sorted(TAIL_DROPS, reverse=True):
                td_key = f"{td:.1f}"
                non_trivial_cells = []
                trivial_cells = []
                for ds in per_dataset:
                    if ds.get('skipped'):
                        continue
                    cell = ds['cells'].get(ckpt_name, {}).get(td_key, {}).get(m)
                    if cell is None:
                        continue
                    if cell['trivial_preservation']:
                        trivial_cells.append((ds['dataset'], cell))
                    else:
                        non_trivial_cells.append((ds['dataset'], cell))
                if not non_trivial_cells:
                    # No non-trivial evidence at this tail_drop — can't conclude
                    continue
                all_pass = all(c['rel_loss'] <= PASS_THRESHOLD
                               for _, c in non_trivial_cells)
                if all_pass:
                    max_safe = td
                    datasets_used = [n for n, _ in non_trivial_cells]
                    saved_vals = [c['pct_saved']
                                  for _, c in non_trivial_cells + trivial_cells]
                    avg_saved = sum(saved_vals) / len(saved_vals) if saved_vals else 0.0
                    break
            out[ckpt_name][m] = {
                'max_safe_tail_drop': max_safe,
                'nontrivial_datasets_used': datasets_used,
                'avg_pct_saved_at_max': avg_saved,
            }
    return out


def _print_per_model_summary(
    max_safe: Dict[str, Dict[str, Any]],
) -> None:
    print(f"\n{'='*78}", flush=True)
    print(f"Max safe tail_drop per (model, checkpoint_config)", flush=True)
    print(f"  pass = rel_loss <= {PASS_THRESHOLD*100:.0f}% on every NON-TRIVIAL dataset cell",
          flush=True)
    print(f"  '—' = no non-trivial evidence at any tail_drop (winners all at low budget)",
          flush=True)
    print(f"{'='*78}", flush=True)
    header = f"  {'model':10s}  " + "  ".join(f"{c:>16s}" for c in CHECKPOINT_CONFIGS)
    print(header, flush=True)
    for m in F16_MODELS:
        cells = []
        for ckpt_name in CHECKPOINT_CONFIGS:
            entry = max_safe[ckpt_name].get(m, {})
            tval = entry.get('max_safe_tail_drop')
            n_ds = len(entry.get('nontrivial_datasets_used', []))
            saved = entry.get('avg_pct_saved_at_max', 0.0)
            if tval is None:
                cells.append(f"  {'—':>14s}  ")
            else:
                cells.append(f"  td={tval:.1f} ({n_ds}ds, saves {saved:.0f}%)")
        print(f"  {m:10s}  " + "".join(cells), flush=True)


def _universal_combos(
    max_safe: Dict[str, Dict[str, Any]],
    per_dataset: List[Dict[str, Any]],
) -> List[Tuple[str, float, float, int]]:
    """Identify (ckpt_config, tail_drop) combos where every model with
    non-trivial evidence passes (rel_loss <= PASS_THRESHOLD on every
    non-trivial dataset cell).

    Returns list of (ckpt_name, tail_drop, avg_pct_saved, n_models_with_data),
    sorted by descending tail_drop, then by descending avg_pct_saved.

    A checkpoint_config that has ZERO models with non-trivial evidence
    (e.g. ckpt_25 when no dataset has best > 25) is excluded — we can't
    claim a schedule is universally safe when nothing actually tested it.
    """
    out = []
    for ckpt_name in CHECKPOINT_CONFIGS:
        ms = max_safe[ckpt_name]
        models_with_data = [m for m, e in ms.items()
                            if e.get('max_safe_tail_drop') is not None]
        if not models_with_data:
            continue
        caps = [ms[m]['max_safe_tail_drop'] for m in models_with_data]
        cap = min(caps)
        # Compute actual avg_pct_saved at the universal cap by looking up
        # each model's per-dataset cells AT THIS tail_drop, not at each
        # model's individual max.
        td_key = f"{cap:.1f}"
        saved_vals = []
        for m in models_with_data:
            for ds in per_dataset:
                if ds.get('skipped'):
                    continue
                cell = ds.get('cells', {}).get(ckpt_name, {}).get(
                    td_key, {}).get(m)
                if cell is None:
                    continue
                # Include both trivial and non-trivial cells in the saving
                # average — saving is real either way; only the loss claim
                # is conditional on non-trivial.
                saved_vals.append(cell['pct_saved'])
        avg_saved = (sum(saved_vals) / len(saved_vals)) if saved_vals else 0.0
        out.append((ckpt_name, cap, avg_saved, len(models_with_data)))
    out.sort(key=lambda x: (-x[1], -x[2]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PASS_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (default: all)')
    parser.add_argument('-o', '--output', type=str,
                        default=str(F16_DIR / 'per_model_sweep.json'))
    parser.add_argument('--pass-threshold', type=float, default=PASS_THRESHOLD,
                        help=f'rel_loss threshold for pass (default {PASS_THRESHOLD})')
    args = parser.parse_args()
    PASS_THRESHOLD = args.pass_threshold

    datasets = [args.dataset] if args.dataset else F16_DATASETS
    F16_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] F16 threshold sweep", flush=True)
    print(f"  checkpoint configs: {list(CHECKPOINT_CONFIGS.keys())}", flush=True)
    print(f"  tail_drops:         {TAIL_DROPS}", flush=True)
    print(f"  models:             {F16_MODELS}", flush=True)
    print(f"  pass threshold (rel_loss): {PASS_THRESHOLD*100:.1f}%", flush=True)

    per_dataset: List[Dict[str, Any]] = []
    for ds in datasets:
        f15 = _load_f15(ds)
        if f15 is None:
            print(f"\n  {ds}: no f15 data, skip", flush=True)
            continue
        per_dataset.append(_sweep_dataset(f15))

    if not per_dataset:
        print("\nNo datasets loaded — aborting.", flush=True)
        return

    max_safe = _max_safe_tail_drop_per_model(per_dataset)
    _print_per_model_summary(max_safe)

    universal = _universal_combos(max_safe, per_dataset)
    print(f"\n{'='*78}", flush=True)
    print(f"Universal combos (tail_drop ≤ this passes for ALL models with non-trivial data)",
          flush=True)
    print(f"{'='*78}", flush=True)
    if not universal:
        print("  None: every checkpoint_config has at least one model with no "
              "non-trivial evidence (winners at low budgets only).", flush=True)
    else:
        print(f"  {'ckpt_config':16s}  {'max universal td':>17s}  {'n_models':>10s}  {'avg saved':>10s}",
              flush=True)
        for ckpt_name, td, saved, n_models in universal:
            print(f"  {ckpt_name:16s}  {td:>17.1f}  {n_models:>10d}  {saved:>9.1f}%",
                  flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'experiment': 'f16_threshold_sweep',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'pass_threshold_rel_loss': PASS_THRESHOLD,
            'checkpoint_configs': CHECKPOINT_CONFIGS,
            'tail_drops': TAIL_DROPS,
            'per_dataset': per_dataset,
            'max_safe_tail_drop_per_model': max_safe,
            'universal_combos': [
                {'checkpoint_config': c, 'max_universal_tail_drop': td,
                 'avg_pct_saved': saved, 'n_models_with_data': n}
                for c, td, saved, n in universal
            ],
        }, f, indent=2, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] Wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()
