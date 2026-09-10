#!/usr/bin/env python3
"""Honest nested-CV reporting for the find_best_hybrid_* CLI drivers.

Shared by find_best_hybrid_{molecular,dta,nucleotide,tcga}.py so all four print
and persist the SAME honest numbers.

WHY THIS EXISTS
---------------
The drivers used to call find_best_hybrid on the WHOLE dataset and print its
cross-validated score as "the hybrid's performance". That number is
selection-optimistic: the recipe (reps x allocation x importance x budget x
model) was chosen by looking at the very folds being reported. The gap between
that number and reality is exactly `bias` below.

The drivers now call kirby.find_best_hybrid.nested_cv_evaluate. Per outer fold
the search sees ONLY outer-train; the winning hybrid AND every individual base
representation are then scored on the untouched outer fold. Both arms pay the
identical held-out tax, so `delta_mean_of_folds` (hybrid minus best single rep,
already direction-corrected: + = hybrid better) is a number you can report.

Baselines are ALWAYS individual representations, reported individually. There is
no mean-of-reps baseline and no concat baseline here.
"""

from datetime import datetime
from pathlib import Path
import json

import numpy as np

METRIC_DISPLAY = {
    'r2': 'R2', 'rmse': 'RMSE', 'mae': 'MAE',
    'accuracy': 'Accuracy', 'auc': 'AUC', 'mcc': 'MCC',
    'spearman': 'Spearman rho', 'pearson': 'Pearson r',
    'ci': 'CI', 'f1_weighted': 'F1w',
}


# =============================================================================
# GUARDS
# =============================================================================

class NestedCVUnsupported(RuntimeError):
    """The requested feature cannot be evaluated honestly under nested CV."""


def check_rep_factories(rep_factories):
    """No-op. Kept so callers don't have to change.

    HISTORY (2026-07-13): rep factories closed over the FULL SMILES/sequence
    list and were called as factory(train_idx, val_idx, y_train), indexing that
    full array. nested_cv_evaluate hands the inner search only OUTER-TRAIN rows,
    so the indices it passed were subset-local (0..len(outer_train)-1) and
    addressed the WRONG molecules — silently, because the shapes still lined up.

    That is now fixed IN THE ENGINE: nested_cv_evaluate wraps every factory so
    subset-local indices are lifted back to global dataset positions before the
    factory sees them (see the `rep_factories` remap in nested_cv_evaluate). The
    factories are also re-fit per outer fold on outer-train rows only, and their
    embedding cache is namespaced per fold.

    So --finetune and the *_pca50 / sns / pdv_rank / pdv_binary_mean / sali_cliff
    reps ARE supported on the honest path. This guard no longer refuses.
    """
    return None


# =============================================================================
# HELPERS
# =============================================================================

def _fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x, width=10, prec=4):
    return f"{float(x):>{width}.{prec}f}" if _fin(x) else f"{'--':>{width}}"


def _sgn(x, width=10, prec=4):
    return f"{float(x):>+{width}.{prec}f}" if _fin(x) else f"{'--':>{width}}"


def _bp_field(bp, name):
    if bp is None:
        return None
    if isinstance(bp, dict):
        return bp.get(name)
    return getattr(bp, name, None)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, 'to_dict'):
        return o.to_dict()
    return str(o)


# =============================================================================
# REPORTING
# =============================================================================

def print_nested_cv_summary(result, task_type, title="NESTED-CV RESULTS"):
    """Print the honest per-model comparison + the per-rep baselines.

    `result` is a kirby.find_best_hybrid.NestedCVResult.
    """
    metric = METRIC_DISPLAY.get(result.metric_name, result.metric_name)
    direction = "lower is better" if result.lower_is_better else "higher is better"

    print("\n" + "=" * 88)
    print(f"{title}  —  {metric} ({direction}), {result.n_outer_folds} outer folds")
    print("=" * 88)
    print("Every score below is measured on an outer fold the search NEVER saw.")
    print("Both arms pay the same tax: the hybrid recipe AND the best single rep")
    print("are each picked on outer-train only, then scored on the held-out fold.")

    by_model = result.by_model or {}
    if not by_model:
        print("\nNo results (every outer fold's inner search failed).")
        return

    print(f"\n=== HYBRID vs BEST SINGLE REP (honest, paired per fold) ===")
    print(f"{'Model':<10}{'hybrid':>10}{'best_single':>13}{'delta':>10}"
          f"{'verdict':>9}{'folds':>7}{'search':>10}{'bias':>10}")
    print("-" * 88)
    for m in sorted(by_model):
        e = by_model[m]
        dl = e.get('delta_mean_of_folds')
        verdict = "--"
        if _fin(dl):
            verdict = "HYBRID" if dl > 0 else ("SINGLE" if dl < 0 else "tie")
        print(f"{m:<10}{_f(e.get('hybrid_mean'))}{_f(e.get('best_single_mean'), 13)}"
              f"{_sgn(dl)}{verdict:>9}{str(e.get('n_paired_folds', '--')):>7}"
              f"{_f(e.get('search_mean'))}{_sgn(e.get('bias'))}")
    print("-" * 88)
    print("delta = hybrid - best_single, direction-corrected (+ = hybrid better).")
    print("search = the inner search's own (selection-leaked) score; bias = search - honest.")
    print("bias is what the OLD whole-dataset driver used to print as performance.")

    # ---- per-rep baselines, individually (never averaged) --------------------
    bbm = result.baselines_by_model or {}
    if bbm:
        models = sorted(bbm)
        reps = sorted({r for mm in bbm.values() for r in mm})
        print(f"\n=== BASELINES — every rep scored INDIVIDUALLY on the held-out folds ===")
        header = f"{'rep':<24}" + "".join(f"{m:>11}" for m in models)
        print(header)
        print("-" * len(header))
        for rep in reps:
            row = f"{rep[:23]:<24}"
            for m in models:
                e = bbm[m].get(rep) or {}
                row += _f(e.get('mean'), 11)
            print(row)

    # ---- blueprints, per outer fold -----------------------------------------
    print(f"\n=== WINNING BLUEPRINT PER OUTER FOLD ===")
    print("Nested CV legitimately picks a different recipe per fold. If the reps")
    print("change fold to fold, there is no single 'the blueprint' to publish.")
    for m in sorted(by_model):
        e = by_model[m]
        bps = e.get('blueprints_per_fold') or []
        singles = e.get('best_single_reps_per_fold') or []
        print(f"  [{m}]")
        for i, bp in enumerate(bps):
            if bp is None:
                print(f"    fold {i}: (search failed)")
                continue
            reps = _bp_field(bp, 'reps')
            print(f"    fold {i}: budget={_bp_field(bp, 'budget')} "
                  f"alloc={_bp_field(bp, 'allocation_method')} "
                  f"imp={_bp_field(bp, 'importance_method')} "
                  f"reps={list(reps) if reps is not None else '?'}")
        if singles:
            print(f"    best single rep per fold: {list(singles)}")
    print("=" * 88)


def save_nested_cv_results(result, dataset_name, search_mode, reps_tested,
                           output_path, extra_metadata=None):
    """Persist the honest block. Top-level key 'nested_cv' — the analysis
    readers (tests/analysis/firstpass_notready.py) look for exactly that."""
    output = {
        'metadata': {
            'dataset': dataset_name,
            'search_mode': search_mode,
            'reps_available': list(reps_tested),
            'evaluation': 'nested_cv',
            'timestamp': datetime.now().isoformat(),
            **(extra_metadata or {}),
        },
        'nested_cv': result.to_dict(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\nResults saved to: {output_path}")
    return output_path


# =============================================================================
# THE SELECTION-OPTIMISTIC PATH (blueprint publication only)
# =============================================================================

SELECTION_OPTIMISTIC_WARNING = """
################################################################################
#  WARNING — SELECTION-OPTIMISTIC SCORES
#
#  This run used a WHOLE-DATASET find_best_hybrid search. The winning recipe was
#  chosen by looking at the same folds whose scores are printed below, so those
#  scores are an UPPER BOUND, not a performance estimate. Do NOT report them, do
#  NOT compare them against a baseline, do NOT put them in a paper.
#
#  This mode exists only to produce ONE blueprint to publish/reuse. For an
#  honest hybrid-vs-baseline number, re-run WITHOUT --publish-blueprint (the
#  default nested-CV path).
################################################################################
"""
