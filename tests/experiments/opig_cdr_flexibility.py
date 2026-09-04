#!/usr/bin/env python3
"""
CDR Flexibility Prediction Experiment (OPIG — Spoendlin et al. 2025)

Replicates Spoendlin et al. 2025 "Predicting the conformational flexibility
of antibody and T-cell receptor CDRs" — tests whether sequence-based antibody
reps can predict CDR flexibility, and whether hybrid reps can match the
structure-based ITsFlexible GNN.

Scientific questions:
  1. Can sequence-based reps predict CDR3 flexibility from VH/VL?
  2. Does hybrid augmentation help for flexibility classification?
  3. How far are sequence-based approaches from the structure-based GNN?
  4. Which CDR types are easiest/hardest to classify?

CDR types:
  - CDRH3: heavy chain CDR3 (most variable, hardest to predict)
  - CDRL3: light chain CDR3
  - CDRA3: TCR alpha chain CDR3
  - CDRB3: TCR beta chain CDR3

Paper baselines (Spoendlin et al. 2025, Nature Machine Intelligence):
  ITsFlexible GNN:    ROC AUC = 0.82, PR AUC = 0.60 (overall crystal)
  Outperforms AlphaFold2 pLDDT and ABB2 RMSPE baselines.
  "Only method that consistently achieves high accuracy across all test sets"

Usage:
    # Smoke test (~2 min, CDRH3 only, fast reps)
    python tests/experiments/opig_cdr_flexibility.py \\
        --cdr-types CDRH3 --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation (all antibody CDR types)
    python tests/experiments/opig_cdr_flexibility.py --cdr-types CDRH3,CDRL3

Reference:
    Spoendlin et al. (2025). Nature Machine Intelligence.
    https://github.com/oxpig/ITsFlexible
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, average_precision_score,
    matthews_corrcoef, accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore, build_honest_rescore_input,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CDR_TYPES = ['CDRH3', 'CDRL3']

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical
    'ablang2', 'esm2', 'antiberty',      # LMs
    'igt5', 'prot_t5',                    # additional LMs
    # 'paratope' (8D Tier 1 structural-sequence rep) intentionally omitted as a
    # base rep — too low-dimensional to be a meaningful baseline here; it's
    # consumed indirectly through the hybrid search if relevant.
]

MODELS_LIST = DOWNSTREAM_MODELS


def _smoke_search_kwargs():
    """Smoke-only hybrid-search shrink, gated on KIRBY_SMOKE_N.

    The hybrid search (find_best_hybrid via get_best_search_for_downstream)
    defaults to FULL_BUDGETS (budgets up to 2000) across a
    pool of tier x aug conditions — minutes of compute even on tiny data.
    For local smoke runs this caps it to one small budget + fast tier so the
    experiment finishes in seconds. No-op (empty dict) when KIRBY_SMOKE_N is
    unset, so server/full runs are unaffected.
    """
    if not os.environ.get('KIRBY_SMOKE_N', '').strip():
        return {}
    # speed is now accepted but ignored (the former tier dict was removed; defaults
    # live in the DEFAULT_* module constants). Smoke shrink is via tiny budgets
    # only. internal_evaluators=['rf'] cuts the search ~5x (DEFAULT_INTERNAL_EVALUATORS
    # is match/rf/svm/lightgbm/mlp); rf alone still exercises the hybrid pipeline.
    # Also drop importance to lightgbm_gain (cheap): DEFAULT_IMPORTANCE_METHODS runs
    # deeplift+treeshap, which compute SHAP over the (here 5600-dim) features
    # per fold and dominate the smoke wall time. lightgbm_gain still produces a
    # real per-rep importance ranking so the hybrid pipeline is exercised.
    return {'speed': 'full', 'budgets': [25], 'internal_evaluators': ['rf'],
            'importance_methods': ['lightgbm_gain']}


# Paper baselines
PAPER_BASELINES = {
    'ITsFlexible GNN (crystal)': {'roc_auc': 0.82, 'pr_auc': 0.60},
}


# =============================================================================
# EVALUATION
# =============================================================================

def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


def evaluate_rep_clf(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for flexibility classification.

    Returns dict with roc_auc, pr_auc, balanced_accuracy, mcc, accuracy.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    # Groups are V-gene families (see run()), so they are genuinely larger than
    # one sample and StratifiedGroupKFold holds out whole germline lineages.
    # Previously groups were unique CDR3 strings, which -- because the loader
    # dedups on CDR3 -- made every group size-1 and silently degraded this to a
    # stratified per-sample KFold, i.e. a random split.
    _, group_sizes = np.unique(groups, return_counts=True)
    if group_sizes.max() == 1:
        raise ValueError(
            "every CV group has one sample: this is a random split, not a "
            "group split. Expected V-gene family groups.")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    roc_scores, pr_scores, bacc_scores, mcc_scores, acc_scores = [], [], [], [], []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if not is_binary:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, y_train)
        except Exception as e:
            # Log fold-fit failures (OPIG-18) instead of silently dropping.
            print(f"      [fold fit failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

        preds = m.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))
        bacc_scores.append(balanced_accuracy_score(y_val, preds))
        mcc_scores.append(matthews_corrcoef(y_val, preds))

        # Scores for AUROC and PR-AUC
        try:
            if hasattr(m, 'decision_function'):
                scores = m.decision_function(X_val)
            elif hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_val)[:, 1]
            else:
                scores = preds.astype(float)
            roc_scores.append(roc_auc_score(y_val, scores))
            pr_scores.append(average_precision_score(y_val, scores))
        except Exception:
            pass

    if not acc_scores:
        return None

    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'balanced_accuracy': {'mean': np.mean(bacc_scores), 'std': np.std(bacc_scores)},
        'mcc': {'mean': np.mean(mcc_scores), 'std': np.std(mcc_scores)},
    }
    if roc_scores:
        result['roc_auc'] = {'mean': np.mean(roc_scores), 'std': np.std(roc_scores)}
    if pr_scores:
        result['pr_auc'] = {'mean': np.mean(pr_scores), 'std': np.std(pr_scores)}
    return result


def evaluate_cdr_type(cdr_types, rep_names, n_folds, pruning_mode='default_pruning'):
    """Load CDR data, generate reps, evaluate all.

    Returns dict with per-rep results.
    """
    from kirby.datasets.itsflexible import load_itsflexible

    print(f"\n{'=' * 60}")
    print(f"CDR types: {cdr_types}")
    print(f"{'=' * 60}")

    data = load_itsflexible(mode='cdr', cdr_types=cdr_types)
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_flex = int((labels == 1).sum())

    print(f"  {n} sequences, {n_flex} flexible ({100*n_flex/n:.1f}%)")

    # Groups = IGHV germline family, from ANARCI on the heavy chain.
    #
    # The loader dedups on CDR3, so grouping by CDR3 (the old behaviour) gave
    # 216 groups for 216 samples -- a random split with a group-CV label. Fv
    # sequence-identity clustering does not rescue it either: antibody
    # frameworks are ~98% conserved, so at 0.95 you get 210 groups of 216 (a
    # no-op) and at 0.70 a single cluster swallows 150 of them. The germline
    # family is the coarse, biologically meaningful key: 11 families here,
    # sizes 2-76. heavy_seqs is what _smoke_subsample subsamples, so groups
    # always align with X/labels.
    from kirby.datasets.cross_species_ness import _assign_vgene_groups
    _cache_root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    fams = _assign_vgene_groups(heavy_seqs, 'imgt', _cache_root)
    _fam_to_id = {f: i for i, f in enumerate(sorted(set(fams)))}
    groups = np.array([_fam_to_id[f] for f in fams], dtype=int)
    print(f"  V-gene groups: {len(_fam_to_id)} families "
          f"({', '.join(sorted(_fam_to_id))})")

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=f"cdr_{'_'.join(cdr_types)}",
    )

    if not reps:
        print("  No reps available, skipping")
        return None

    # Run the hybrid search. No `kirby_hybrid` matrix is materialised over the
    # full data and re-evaluated under our CV: that path lets every validation
    # fold see its own labels through the full-data fit of `selected_indices`.
    #
    # The hybrid's reported AUC is the leave-one-fold-out re-score
    # (honest_comparison below). Until 2026-09-04 this file reported
    # kirby_hybrid_search_score_mean instead — the winning recipe's own mean
    # over the folds that chose it, a maximum over hundreds of configurations,
    # printed beside per-representation AUCs that were means over held-out
    # folds. 0.6309 in the stored result file is that number.
    # No augmentation reps for CDR flexibility classification.
    hybrid_summary = None
    honest_block = None
    honest_result = None
    try:
        from experiments._shared import (
            resolve_pruning_kwargs, get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label=f"cdr_flex/{'_'.join(sorted(cdr_types))}")
        _smoke = _smoke_search_kwargs()
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            # augmentations=None -> pool only iterates aug_condition='none';
            # helper sets augmentation_strategy per iteration, no collision.
            augmentations=None,
            task_type='classification',
            dataset_name='cdr_flexibility',
            return_fitted=True,
            # _smoke may override the evaluator list (rf-only) for local runs;
            # falls back to the full INTERNAL_EVALUATORS on server.
            internal_evaluators=_smoke.get('internal_evaluators',
                                           INTERNAL_EVALUATORS),
            condition_suffix='_'.join(sorted(cdr_types)),
            adaptive_pruning=0.25,
            **{k: v for k, v in _smoke.items()
               if k not in ('speed', 'internal_evaluators')},
            **_pkw,
        )
        top3 = extract_top3_base_reps(search) if search is not None else None
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
        if search is not None:
            honest_block = build_honest_rescore_input(search)
            honest_result = honest_rescore(
                honest_block['candidates_by_model'],
                honest_block['baselines_by_model'],
                honest_block['lower_is_better'],
                metric_name=honest_block.get('metric_name'),
            )
            for mn, r in sorted(honest_result.items()):
                flag = ' [single-fold: NOT honest]' if r['single_fold'] else ''
                print(f"  honest × {mn:8s}  hybrid={r['hybrid_honest']:.4f}  "
                      f"best_single={r['best_single_honest']:.4f}  "
                      f"delta={r['delta']:+.4f}{flag}")
        if _meta is not None:
            hybrid_summary = {
                # Search diagnostic only — the maximum over the searched
                # recipes, taken on the folds that scored them. Renamed from
                # kirby_hybrid_search_score_mean so an old reader fails loudly
                # instead of quietly plotting it as a result (2026-09-04).
                'search_best_score_mean': float(_meta.get('score_mean')),
                'search_model': _meta.get('model'),
                'search_budget': _meta.get('budget'),
                'search_tier': _meta.get('tier'),
                'search_aug_condition': _meta.get('aug_condition'),
                'search_metric': 'auc',
                'note': ('search_best_score_mean is NOT a held-out score. The '
                         'hybrid number for this experiment is in '
                         'honest_comparison.'),
            }
        else:
            hybrid_summary = {'kirby_hybrid_failed': True,
                              'error': 'no candidate returned by pool'}
            print("  Hybrid search returned no candidate")
    except Exception as e:
        hybrid_summary = {'kirby_hybrid_failed': True, 'error': str(e)}
        print(f"  Hybrid build failed: {e}")

    # Evaluate
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × {n_folds}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_clf(X, labels, groups, factory(), n_folds)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                roc = res.get('roc_auc', {}).get('mean', float('nan'))
                pr = res.get('pr_auc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"ROC={roc:.4f}  PR={pr:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'n_flexible': n_flex,
        'n_rigid': n - n_flex,
        'cdr_types': cdr_types,
        'results': results,
        'hybrid_summary': hybrid_summary,
        # Held-out hybrid-vs-best-single comparison per downstream model, plus
        # the per-fold scores it was computed from.
        'honest_rescore': honest_block,
        'honest_comparison': honest_result,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Print comparison table with paper baselines."""
    print(f"\n{'=' * 100}")
    print(f"CDR FLEXIBILITY EXPERIMENT RESULTS (OPIG — Spoendlin et al. 2025)")
    print(f"{'=' * 100}")

    # Paper baselines
    print(f"\n  Paper baselines:")
    for method, metrics in PAPER_BASELINES.items():
        roc = metrics.get('roc_auc', float('nan'))
        pr = metrics.get('pr_auc', float('nan'))
        print(f"    {method}: ROC AUC = {roc:.2f}, PR AUC = {pr:.2f}")

    for key, ds in all_results.items():
        results = ds['results']
        cdr_types = ds['cdr_types']

        print(f"\n  --- CDR types: {cdr_types} ({ds['n_samples']} samples, "
              f"{ds['n_flexible']} flexible) ---")

        print(f"  {'Rep':25s} {'Model':8s} {'ROC AUC':>8s} {'PR AUC':>8s} "
              f"{'MCC':>8s} {'BalAcc':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        rows = []
        for rep_name, rep_data in results.items():
            for model_name, metrics in rep_data['models'].items():
                roc = metrics.get('roc_auc', {}).get('mean', float('nan'))
                pr = metrics.get('pr_auc', {}).get('mean', float('nan'))
                mcc = metrics.get('mcc', {}).get('mean', float('nan'))
                bacc = metrics.get('balanced_accuracy', {}).get('mean', float('nan'))
                rows.append((rep_name, model_name, roc, pr, mcc, bacc))

        rows.sort(key=lambda r: (r[2] if not np.isnan(r[2]) else 0), reverse=True)

        for rep_name, model_name, roc, pr, mcc, bacc in rows:
            roc_s = f"{roc:.4f}" if not np.isnan(roc) else "—"
            pr_s = f"{pr:.4f}" if not np.isnan(pr) else "—"
            print(f"  {rep_name:25s} {model_name:8s} {roc_s:>8s} {pr_s:>8s} "
                  f"{mcc:8.4f} {bacc:8.4f}")

        # Hybrid: leave-one-fold-out re-score, the same held-out treatment the
        # per-representation AUCs above get. The search's own winning mean is
        # kept in hybrid_summary as a diagnostic and is not printed here.
        hs = ds.get('hybrid_summary') or {}
        honest = ds.get('honest_comparison') or {}
        if honest:
            print(f"\n  kirby_hybrid vs best single representation "
                  f"(leave-one-fold-out, AUC):")
            print(f"    {'Model':10s} {'Hybrid':>10s} {'BestSingle':>12s} "
                  f"{'Delta':>10s} {'Folds':>6s}")
            print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")
            for model_name in sorted(honest):
                r = honest[model_name]
                h, b, d = (r['hybrid_honest'], r['best_single_honest'],
                           r['delta'])
                h_s = f"{h:.4f}" if not np.isnan(h) else "—"
                b_s = f"{b:.4f}" if not np.isnan(b) else "—"
                d_s = f"{d:+.4f}" if not np.isnan(d) else "—"
                flag = ' (single-fold: NOT honest)' if r['single_fold'] else ''
                print(f"    {model_name:10s} {h_s:>10s} {b_s:>12s} "
                      f"{d_s:>10s} {r['n_folds_used']:>6d}{flag}")
        elif hs.get('kirby_hybrid_failed'):
            print(f"  kirby_hybrid: FAILED ({hs.get('error')})")

        # Comparison to paper baseline.
        # WARNING: paper used held-out test sets; this experiment uses
        # CDR3-deduped stratified KFold CV. Numbers are NOT directly
        # comparable — treat the gap as a rough sanity check only.
        best_roc = max(
            (m.get('roc_auc', {}).get('mean', 0)
             for rd in results.values() for m in rd['models'].values()),
            default=0
        )
        gnn_roc = PAPER_BASELINES['ITsFlexible GNN (crystal)']['roc_auc']
        print(f"\n  Best KIRBy ROC AUC: {best_roc:.4f}")
        print(f"  ITsFlexible GNN:    {gnn_roc:.2f} "
              f"(paper held-out test set — not directly comparable)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CDR flexibility experiment (Spoendlin 2025). '
                    'Tests antibody reps for CDR3 flexibility classification.')
    parser.add_argument(
        '--cdr-types', type=str, default=None,
        help='Comma-separated CDR types (default: CDRH3,CDRL3)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 9)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_cdr_flexibility.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    args = parser.parse_args()

    cdr_types = ([c.strip() for c in args.cdr_types.split(',') if c.strip()]
                 if args.cdr_types else DEFAULT_CDR_TYPES)
    rep_names = ([r.strip() for r in args.reps.split(',') if r.strip()]
                 if args.reps else DEFAULT_REPS)

    print("=" * 70)
    print("CDR FLEXIBILITY EXPERIMENT (Spoendlin et al. 2025)")
    print(f"  Paper: Nature Machine Intelligence")
    print(f"  CDR types: {cdr_types}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print("=" * 70)

    all_results = {}

    # Evaluate each CDR type separately + combined
    for cdr_type in cdr_types:
        try:
            res = evaluate_cdr_type([cdr_type], rep_names, args.n_folds,
                                     pruning_mode=args.pruning_mode)
            if res is not None:
                all_results[cdr_type] = res
        except Exception as e:
            print(f"  FAILED for {cdr_type}: {e}")
            import traceback
            traceback.print_exc()

    # Also evaluate combined
    if len(cdr_types) > 1:
        try:
            res = evaluate_cdr_type(cdr_types, rep_names, args.n_folds,
                                     pruning_mode=args.pruning_mode)
            if res is not None:
                all_results['combined'] = res
        except Exception as e:
            print(f"  FAILED for combined: {e}")

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_cdr_flexibility.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
