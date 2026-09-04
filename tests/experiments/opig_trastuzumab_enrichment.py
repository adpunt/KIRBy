#!/usr/bin/env python3
"""
Trastuzumab-HER2 Binder Enrichment Experiment (OPIG B1)

Extends Chinery et al. 2024 "Baselining the Buzz: Trastuzumab-HER2
Affinity, and Beyond" — the paper evaluates zero-shot LIBRARY DESIGN
methods (BLOSUM, AbLang, ESM, ProteinMPNN as GENERATORS). KIRBy instead
evaluates REPRESENTATION QUALITY for supervised classification. These are
different questions: the paper asks "which generator designs more binders?"
while KIRBy asks "which embedding best separates binders from non-binders?"

Two evaluation modes:
  1. Zero-shot ranking: LLR reps used directly as scores (closer to paper)
  2. Supervised classification: train classifier on reps, rank by score
     (KIRBy's extension — paper did not train classifiers)

Paper baselines (Chinery et al. 2024, Table 2):
  These are % of a designed library predicted as binders by a CNN, NOT
  top-K enrichment on a held-out test set. Numbers are not directly
  comparable to KIRBy's enrichment, but reported for context:
  Random baseline:   ~13% predicted binder rate
  ProteinMPNN:       ~19%
  BLOSUM62:          ~26%
  AbLang:            ~27-29%
  ESM:               ~30%
  DMS (oracle):      ~33%
  Experimental SPR (v2, 720 designs): BLOSUM outperformed all 4 DL methods.

Usage:
    # Smoke test (~5 min)
    python tests/experiments/opig_trastuzumab_enrichment.py \\
        --n-folds 2 --reps blosum,onehot,aac

    # Full evaluation
    python tests/experiments/opig_trastuzumab_enrichment.py

Reference:
    Chinery et al. (2024). bioRxiv/PNAS.
    https://github.com/lewis-chinery/Tz_her2_affinity_and_beyond
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
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from experiments._shared import (
    convert_numpy, get_model_factories, DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    extract_top3_base_reps, FULL_BUDGETS,
    honest_rescore, build_honest_rescore_input,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_REPS = [
    'blosum', 'onehot', 'aac',           # classical (paper used BLOSUM)
    'ablang2', 'esm2', 'antiberty',      # LMs (paper used AbLang, ESM)
    'igt5', 'prot_t5',                    # additional LMs
    'esmc_llr', 'antiberty_llr',          # LLR (paper used ProteinMPNN LLR)
    'abroberta',                           # Ab-RoBERTa
]

ENRICHMENT_FRACTIONS = [0.01, 0.05, 0.10]  # top-1%, top-5%, top-10%
# Note: at n_val ~= 1000 (5-fold on ~5K), top-1% = ~10 samples — high
# variance. Reported but interpret with care; E@5% and E@10% are more stable.

MODELS_LIST = DOWNSTREAM_MODELS


def _smoke_search_kwargs():
    """Smoke-only hybrid-search shrink, gated on KIRBY_SMOKE_N.

    Caps the find_best_hybrid pool to one small budget + fast tier for local
    smoke runs (default is minutes of compute).
    No-op when KIRBY_SMOKE_N is unset.
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


# Paper baselines: CNN-predicted binder enrichment (Chinery et al. 2024)
# % sequences with CNN-HER2-max binding probability >90%
# Note: these are predicted enrichment rates, not raw top-K enrichment.
# KIRBy measures raw top-K enrichment (different metric but comparable in spirit).
PAPER_BASELINES = {
    'Random (paper)':          {'predicted_binder_pct': 0.13},
    'ProteinMPNN (paper)':     {'predicted_binder_pct': 0.19},
    'BLOSUM62 (paper)':        {'predicted_binder_pct': 0.26},
    'AbLang mask-all (paper)': {'predicted_binder_pct': 0.27},
    'AbLang mask-one (paper)': {'predicted_binder_pct': 0.29},
    'ESM mask-one (paper)':    {'predicted_binder_pct': 0.30},
    'DMS oracle (paper)':      {'predicted_binder_pct': 0.33},
}
# Experimental SPR validation (v2, 720 designs):
# BLOSUM outperformed all 4 DL methods, 9 sub-nanomolar binders found


# =============================================================================
# ENRICHMENT METRIC
# =============================================================================

def compute_enrichment(y_true, y_score, fractions=None):
    """Compute binder enrichment at various top-N fractions.

    Enrichment = (% binders in top-N) / (% binders overall).
    Raw enrichment = % binders in top-N (what the paper reports).

    Args:
        y_true: binary labels (1=binder, 0=non-binder)
        y_score: predicted scores (higher = more likely binder)
        fractions: list of fractions to evaluate (e.g., [0.01, 0.05, 0.10])

    Returns:
        dict: {f'enrich_{frac}': raw_enrichment, f'ef_{frac}': enrichment_factor}
    """
    if fractions is None:
        fractions = ENRICHMENT_FRACTIONS

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    base_rate = y_true.mean()

    # Rank by score descending. Use kind='stable' so that ties are broken
    # deterministically by input order (not by quicksort's data-dependent
    # tie-breaking, which can leak label information when scores are coarse
    # — e.g. binary reps or LLR sums collapsing to a handful of values).
    order = np.argsort(-y_score, kind='stable')
    y_sorted = y_true[order]

    result = {'base_rate': float(base_rate)}
    for frac in fractions:
        k = max(1, int(n * frac))
        top_k = y_sorted[:k]
        raw_enrichment = top_k.mean()
        enrichment_factor = raw_enrichment / base_rate if base_rate > 0 else 0.0
        result[f'enrich_{frac}'] = float(raw_enrichment)
        result[f'ef_{frac}'] = float(enrichment_factor)

    return result


def _is_binary_rep(X):
    if X.dtype == np.uint8:
        return True
    uniq = np.unique(X)
    return len(uniq) <= 2 and set(uniq).issubset({0, 0.0, 1, 1.0})


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_enrichment(X, labels, groups, model, n_folds):
    """Evaluate one (rep, model) for enrichment via stratified CV.

    For each fold: train → predict scores on hold-out → compute enrichment.
    Returns dict with enrichment metrics (mean/std across folds) + AUROC.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if groups is None or len(np.unique(groups)) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds}-fold group split: "
            f"groups={'None' if groups is None else len(np.unique(groups))} unique")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, labels, groups=groups))

    is_binary = _is_binary_rep(X)
    fold_enrichments = []
    auroc_scores = []
    acc_scores = []
    n_folds_attempted = len(splits)
    n_folds_completed = 0

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
        except Exception as _fit_err:
            print(
                f"[WARN] Fold fit failed (skipping fold): {_fit_err!r}",
                flush=True,
            )
            continue
        n_folds_completed += 1

        # Get prediction scores for ranking
        if hasattr(m, 'decision_function'):
            scores = m.decision_function(X_val)
        elif hasattr(m, 'predict_proba'):
            scores = m.predict_proba(X_val)[:, 1]
        else:
            scores = m.predict(X_val).astype(float)

        preds = m.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))

        try:
            auroc_scores.append(roc_auc_score(y_val, scores))
        except Exception:
            pass

        enrichment = compute_enrichment(y_val, scores)
        fold_enrichments.append(enrichment)

    if not fold_enrichments:
        return None

    if n_folds_completed < n_folds_attempted:
        print(
            f"[WARN] Only {n_folds_completed}/{n_folds_attempted} folds completed "
            f"(fit failures dropped silently before P1-fix); results may be biased.",
            flush=True,
        )

    # Aggregate across folds
    result = {
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
        'n_folds_completed': int(n_folds_completed),
        'n_folds_attempted': int(n_folds_attempted),
    }
    if auroc_scores:
        result['auroc'] = {'mean': np.mean(auroc_scores), 'std': np.std(auroc_scores)}

    # Enrichment metrics
    for frac in ENRICHMENT_FRACTIONS:
        key = f'enrich_{frac}'
        vals = [fe[key] for fe in fold_enrichments if key in fe]
        if vals:
            result[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

        ef_key = f'ef_{frac}'
        ef_vals = [fe[ef_key] for fe in fold_enrichments if ef_key in fe]
        if ef_vals:
            result[ef_key] = {'mean': np.mean(ef_vals), 'std': np.std(ef_vals)}

    result['base_rate'] = np.mean([fe['base_rate'] for fe in fold_enrichments])
    return result


def evaluate_dataset(rep_names, n_folds, pruning_mode='default_pruning'):
    """Evaluate all reps × models on tz_her2_binary for enrichment."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: tz_her2_binary (Trastuzumab-HER2 binding)")
    print(f"{'=' * 60}")

    data = load_dataset('tz_her2_binary')
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_binders = int((labels == 1).sum())
    print(f"  {n} samples, {n_binders} binders ({100*n_binders/n:.1f}%), "
          f"{n - n_binders} non-binders ({100*(n-n_binders)/n:.1f}%)")

    # Use registry-provided groups (edit-distance buckets per
    # DATASET_REGISTRY['tz_her2_binary']). Earlier code rebuilt groups by
    # sequence identity, which collapsed Trastuzumab CDRH3 variants into
    # ~N singletons and degenerated StratifiedGroupKFold to stratified
    # random (P0-2 fix).
    groups = data.get('groups')
    if groups is None:
        # Fall back to identity groups only when registry didn't supply them.
        unique_seqs = {}
        group_ids = np.zeros(n, dtype=int)
        for i, s in enumerate(heavy_seqs):
            if s not in unique_seqs:
                unique_seqs[s] = len(unique_seqs)
            group_ids[i] = unique_seqs[s]
        groups = group_ids

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag='tz_her2_binary',
        mutations=data.get('mutations'),
    )

    if not reps:
        print("  No reps, aborting")
        return None

    # Hybrid search. No `kirby_hybrid` matrix is materialised for the
    # supervised loop: the earlier flow ran the search on the full label
    # vector and re-evaluated it under the same StratifiedGroupKFold, which
    # carries the per-representation `selected_indices` across folds (P0 fix).
    #
    # The number reported for the hybrid is the leave-one-fold-out re-score
    # (honest_block below), not the search's own best mean AUC. That best mean
    # was the maximum over every allocation method, importance method,
    # evaluator and budget, taken on the same folds it was measured on, while
    # every single representation beside it was a mean over held-out folds.
    # Changed 2026-09-04, split audit.
    # Augmentation config matches tz_her2_binary in DATASET_REPS:
    # augmentation = [] (no augmentation reps for antibody classification)
    hybrid_search_meta = None
    honest_block = None
    honest_result = None
    try:
        from experiments._shared import (resolve_pruning_kwargs,
                                          get_best_search_for_downstream)
        _pkw = resolve_pruning_kwargs(pruning_mode, len(labels),
                                       dataset_label='tz_her2_binary')
        _smoke = _smoke_search_kwargs()
        search, _meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model='rf', budget=None,
            augmentations=None,
            metric='auc',
            task_type='classification',
            dataset_name='tz_her2_binary',
            return_fitted=True,
            adaptive_pruning=0.25,
            **{k: v for k, v in _smoke.items() if k != 'speed'},
            **_pkw,
        )
        if search is not None:
            top3 = extract_top3_base_reps(search)
            if top3:
                print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            # Per-fold scores for every searched recipe and every baseline
            # representation, then the leave-one-fold-out re-score: on each
            # held-out fold both arms pick their winner from the other folds
            # only, so the hybrid AUROC is measured the same way as the
            # per-representation AUROCs it sits beside.
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
        if _meta is not None and _meta.get('score_mean') is not None:
            hybrid_search_meta = {
                # The search's own winning mean, kept ONLY as a search
                # diagnostic. It is a maximum over recipes taken on the folds
                # that scored them; do not print it beside a held-out AUROC.
                'search_best_cv_auc': float(_meta['score_mean']),
                'tier': _meta.get('tier'),
                'aug_condition': _meta.get('aug_condition'),
                'note': (
                    'search_best_cv_auc is the best recipe mean over the '
                    'search folds, not a held-out score — it is the maximum '
                    'over every allocation method, importance method, '
                    'evaluator and budget, measured on the folds that chose '
                    'it. The hybrid number to report is in honest_comparison, '
                    'a leave-one-fold-out re-score against the same baseline '
                    'representations (2026-09-04, split audit). The key was '
                    'renamed from rf_cv_auc, so a reader that asks for the old '
                    'name gets nothing rather than this number.'
                ),
            }
    except Exception as e:
        print(f"  Hybrid build failed: {e}")

    # Zero-shot evaluation: LLR reps used directly as ranking scores
    # (matches paper's protocol — no classifier training, just rank by score)
    LLR_REPS = {'esmc_llr', 'antiberty_llr'}
    print(f"\n  Zero-shot ranking (paper's protocol)...")
    zero_shot_results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        if rep_name in LLR_REPS and X.shape[1] <= 3:
            # LLR reps return [sum_llr, mean_llr, min_llr] per sequence
            # (see src/kirby/representations/mutation.py, around lines
            # 551-553 / 581 for ESM-C / AntiBERTy respectively). Averaging
            # across the 3 columns was a P0 bug: sum_llr scales with the
            # number of mutations, dominates the mean, and effectively
            # ranks sequences by mutation count rather than per-residue
            # binder plausibility. Use column 1 (mean_llr) — length-
            # invariant and matches the paper's per-residue protocol.
            scores = X[:, 1]
            enrichment = compute_enrichment(labels, scores)
            try:
                auroc_val = roc_auc_score(labels, scores)
            except Exception:
                auroc_val = float('nan')
            zero_shot_results[rep_name] = {
                'enrichment': enrichment,
                'auroc': float(auroc_val),
            }
            e1 = enrichment.get('enrich_0.01', float('nan'))
            print(f"    {rep_name:25s} zero-shot  "
                  f"E@1%={e1:.4f}  AUROC={auroc_val:.4f}")

    # Supervised evaluation (KIRBy's extension beyond paper).
    # NOTE: this is a second pass over `reps` after the zero-shot loop
    # above. The two passes are kept separate because the zero-shot pass
    # has no model factory and the supervised pass runs |models| inner
    # iterations per rep; merging them is mechanical but not minimal-diff.
    models = get_model_factories(MODELS_LIST, 'classification', n_samples=n)
    print(f"\n  Supervised evaluation ({len(reps)} reps × {len(models)} models × {n_folds}-fold CV)...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_enrichment(X, labels, groups, factory(), n_folds)
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                e1 = res.get('enrich_0.01', {}).get('mean', float('nan'))
                auroc = res.get('auroc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"E@1%={e1:.4f}  AUROC={auroc:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    return {
        'n_samples': n,
        'n_binders': n_binders,
        'base_rate': n_binders / n,
        'zero_shot': zero_shot_results,  # paper's protocol (no training)
        'supervised': results,            # KIRBy's extension (trained classifiers)
        'hybrid_search': hybrid_search_meta,  # search diagnostic; NOT held-out
        # The held-out hybrid-vs-best-single comparison, per downstream model,
        # plus the raw per-fold scores it was computed from.
        'honest_rescore': honest_block,
        'honest_comparison': honest_result,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(ds_results):
    """Print enrichment comparison table with paper baselines."""
    if ds_results is None:
        return

    results = ds_results['supervised']
    base_rate = ds_results['base_rate']

    print(f"\n{'=' * 100}")
    print(f"TRASTUZUMAB ENRICHMENT COMPARISON (OPIG B1)")
    print(f"  {ds_results['n_samples']} samples, base binder rate = {base_rate:.3f}")
    print(f"{'=' * 100}")

    # Header
    header = f"  {'Method':30s} {'E@1%':>8s} {'E@5%':>8s} {'E@10%':>8s} {'AUROC':>8s} {'Acc':>8s}"
    print(header)
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Random baseline
    print(f"  {'Random baseline':30s} {base_rate:8.4f} {base_rate:8.4f} "
          f"{base_rate:8.4f} {'0.5000':>8s} {'—':>8s}")

    # Paper baselines (predicted binder %, different metric but reference point)
    print(f"  {'— Paper (pred. binder %) —':30s}")
    for method, metrics in PAPER_BASELINES.items():
        pct = metrics.get('predicted_binder_pct', float('nan'))
        pct_s = f"{pct:.2%}" if not np.isnan(pct) else "—"
        print(f"  {method:30s} {pct_s:>8s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s}")

    print(f"  {'. . .':30s}")

    # KIRBy results: best model per rep, sort by E@1%
    rows = []
    for rep_name, rep_data in results.items():
        for model_name, metrics in rep_data['models'].items():
            e1 = metrics.get('enrich_0.01', {}).get('mean', float('nan'))
            e5 = metrics.get('enrich_0.05', {}).get('mean', float('nan'))
            # Stored key is f'enrich_{0.10}' -> 'enrich_0.1' (Python drops the
            # trailing zero), so look up 'enrich_0.1' not 'enrich_0.10'.
            e10 = metrics.get('enrich_0.1', {}).get('mean', float('nan'))
            auroc = metrics.get('auroc', {}).get('mean', float('nan'))
            acc = metrics.get('accuracy', {}).get('mean', float('nan'))
            rows.append((rep_name, model_name, e1, e5, e10, auroc, acc))

    rows.sort(key=lambda r: (r[2] if not np.isnan(r[2]) else 0), reverse=True)

    for rep_name, model_name, e1, e5, e10, auroc, acc in rows:
        label = f"{rep_name} ({model_name})"
        tag = ' ***' if rep_name == 'kirby_hybrid' else ''
        e1s = f"{e1:.4f}" if not np.isnan(e1) else "—"
        e5s = f"{e5:.4f}" if not np.isnan(e5) else "—"
        e10s = f"{e10:.4f}" if not np.isnan(e10) else "—"
        aurocs = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        accs = f"{acc:.4f}" if not np.isnan(acc) else "—"
        print(f"  {label:30s} {e1s:>8s} {e5s:>8s} {e10s:>8s} {aurocs:>8s} {accs:>8s}{tag}")

    # Summary
    best_indiv = [r for r in rows if r[0] != 'kirby_hybrid']

    print(f"\n  Summary (Enrichment @ top-1%):")
    if best_indiv:
        bi = best_indiv[0]
        print(f"    Best individual: {bi[0]} ({bi[1]}) E@1% = {bi[2]:.4f}")

    # Hybrid number: the leave-one-fold-out re-score. On each held-out fold the
    # hybrid recipe and the best single representation are both picked from the
    # other folds, so the two arms pay the same tax. AUROC, so it is not
    # rankable against the E@1% column above; printed separately.
    honest = ds_results.get('honest_comparison') or {}
    if honest:
        print(f"\n  Hybrid vs best single representation "
              f"(leave-one-fold-out, AUROC):")
        print(f"    {'Model':10s} {'Hybrid':>10s} {'BestSingle':>12s} "
              f"{'Delta':>10s} {'Folds':>6s}")
        print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*6}")
        for model_name in sorted(honest):
            r = honest[model_name]
            h, b, d = r['hybrid_honest'], r['best_single_honest'], r['delta']
            h_s = f"{h:.4f}" if not np.isnan(h) else "—"
            b_s = f"{b:.4f}" if not np.isnan(b) else "—"
            d_s = f"{d:+.4f}" if not np.isnan(d) else "—"
            flag = ' (single-fold: NOT honest)' if r['single_fold'] else ''
            print(f"    {model_name:10s} {h_s:>10s} {b_s:>12s} {d_s:>10s} "
                  f"{r['n_folds_used']:>6d}{flag}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Trastuzumab binder enrichment experiment (OPIG B1). '
                    'Replicates Chinery et al. 2024 enrichment analysis.')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 11)')
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of stratified CV folds (default: 5)')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_trastuzumab_enrichment.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help="Adaptive pruning mode (default 'default_pruning'). "
             "'aggressive_pruning' = F16-validated multi-stage pruning "
             "(checkpoints=[25,50,100], td=0.5) applied globally to all models.")

    args = parser.parse_args()
    rep_names = args.reps.split(',') if args.reps else DEFAULT_REPS
    # Validate rep names against the registries so a typo doesn't silently
    # drop a rep (P1-8 fix).
    from kirby.representations.antibody import ANTIBODY_REPS
    from kirby.representations.mutation import MUTATION_REPS
    _known = set(ANTIBODY_REPS) | set(MUTATION_REPS)
    _unknown = [r for r in rep_names if r not in _known]
    if _unknown:
        print(f"  WARNING: unknown rep name(s) {sorted(_unknown)} — "
              f"will be silently skipped by compute_or_load_reps")

    print("=" * 70)
    print("TRASTUZUMAB BINDER ENRICHMENT EXPERIMENT (OPIG B1)")
    print(f"  Paper: Chinery et al. 2024")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {MODELS_LIST}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Enrichment fractions: {ENRICHMENT_FRACTIONS}")
    print("=" * 70)

    ds_results = evaluate_dataset(rep_names, args.n_folds,
                                    pruning_mode=args.pruning_mode)
    if ds_results is not None:
        print_analysis(ds_results)

    # Save results
    output_path = (args.output or
                   'tests/results/comparisons/opig_trastuzumab_enrichment.json')
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, 'w') as f:
        json.dump(convert_numpy(ds_results or {}), f, indent=2)
    print(f"\nResults saved: {p}")


if __name__ == '__main__':
    main()
