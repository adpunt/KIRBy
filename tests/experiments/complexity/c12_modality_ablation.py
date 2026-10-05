#!/usr/bin/env python3
"""C12: TCGA modality ablation — which omics modality combinations help vs hurt?

Paper: Complexity
Conclusion: Determines whether adding a second omics modality to expression
improves prediction for each TCGA task. Identifies beneficial, neutral, and
harmful modality pairings.

Design:
  For each TCGA task:
    1. Score each modality group individually (best rep within group)
    2. Score each pairwise modality combination (concatenate best reps)
    3. Score all modalities combined

  2026-09-12, decided: a combination is two whole representations side by side
  at full width — no hybrid, no feature selection, no budget, no search. The
  leave-one-fold-out delta re-picks BOTH sides: for each held-out fold, the best
  representation of each modality is decided on the other folds and that
  combination is scored on this fold, against the single representation picked
  the same way. Representations are already scored on every fold, so the only
  new model fits are the combinations where the folds disagree about which
  representation wins.

Modality groups:
  expression   — gene_hvg2000, gene_pca100, gene_hallmark, gene_progeny,
                 gene_kegg, gene_dorothea, gene_reactome, gene_oncogenic,
                 gene_combined_pathway, gene_nmf50, gene_ica50,
                 gene_sparse_pca50, gene_autoencoder50, gene_corr_modules,
                 gene_quantile_rank, gene_immune, gene_stats,
                 gene_chromosome_arm
  methylation  — methyl_pca50, methyl_hvg
  mutation     — mutation_matrix, mut_tmb, mut_driver, mut_pathway,
                 mut_pca50, sig_sbs, sig_dbs, sig_id
  cnv          — cnv_gistic, cnv_pca50
  mirna        — mirna_top200
  rppa         — rppa
  immune       — immune_cibersort, immune_xcell
  clinical     — clinical, purity_consensus, stemness_mrnasi,
                 heterogeneity_math, msi_status, hrd_score, cimp_status
  wsi          — wsi_ctranspath, wsi_phikon, wsi_hoptimus0, wsi_gigapath,
                 wsi_moco, wsi_titan, wsi_uni2h

Notes:
  - Factory reps (those that require per-fold fitting) are NOT used here
    to keep this experiment self-contained and comparable.
  - We use only the precomputed base reps (non-factory) from each group.
  - For each group, we evaluate all available reps and pick the best.
  - Pairwise combinations concatenate the best rep from each group.
  - Metric: the dataset's registry metric and direction (get_domain_metric):
    auc on tcga_luad_vs_lusc, f1_weighted on tcga_brca_pam50. c12 scores only
    auc, f1_weighted and r2 itself, and raises on any other.
  - Groups from TSS (tissue source site) for site-corrected CV.

  2026-10-05, aligned (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4; selection rule 2.3
  "Still open" (B), decided 2026-10-05: one representation per modality side by
  side, no hybrids, on the decided models, per fold):
  - Models: decided_grid.decided_models(dataset, 'general') only, built by
    find_best_hybrid.get_model (the classifier the rest of the suite fits). c12
    builds no hybrid and runs no search, so it has no distance search and no knn.
  - Representations: DATASET_REPS[dataset]['general'] only. Step 8 cut
    augmentation on both TCGA datasets and c12 has no augmentation comparison.
  - Outer folds: kirby.find_best_hybrid.nested_cv_outer_folds with
    config['n_splits'], the same folds the per-fold searches use.
  - Scaling: kirby.scaling.FeatureScaler (0/1 features left as they are).
  - Result: one record per model per outer fold under 'per_fold'.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score, f1_score

from kirby.scaling import FeatureScaler
from kirby.find_best_hybrid import nested_cv_outer_folds
from experiments._shared import (
    get_config, create_experiment_parser,
    fold_reselection, get_domain_metric,
)
from experiments.decided_grid import decided_models, decided_aug_conditions

# 2026-10-05: stamped into the result. Results written before the alignment
# fitted five models on cv_split folds with StandardScaler; this tells them
# apart. run_all's resume compares only input_fingerprint, which does not see
# a code change (requested as a shared change).
RESULT_VERSION = 'c12_aligned_2026-10-05'

# 2026-10-05: what chose each representation, written on every per-fold
# record. Fold f's representation of a modality is the one with the best mean
# held-out score over the OTHER folds (the 2026-09-12 design, kept). Those
# held-out rows are all inside fold f's training rows, but the models that
# scored them were fitted on rows that include fold f's held-out rows.
SELECTION_SCOPE = 'other_folds_held_out'
# 2026-10-05: with one outer fold there is no other fold. The single
# representation and every combination are then picked on that fold's
# held-out rows, the same rows they are scored on; records say so.
SELECTION_SCOPE_ONE_FOLD = 'same_fold_held_out_rows'


def _selection_scope(n_folds):
    """What chose the representations, given the number of outer folds."""
    return SELECTION_SCOPE if n_folds >= 2 else SELECTION_SCOPE_ONE_FOLD


def _reselected_pair_delta(pair_model_scores, baseline_rep_names, all_rep_scores,
                       lower_is_better, metric_name=None):
    """Leave-one-fold-out delta for one modality combination.

    On each held-out fold the single representation it is compared against is
    re-picked using the other folds only, from `baseline_rep_names`.

    2026-09-12 — DECIDED: the same re-pick is applied to the combination side.
    `pair_model_scores` is now {model: {'fold_scores': [...]}} where the score
    on fold f came from the pair chosen on the folds other than f
    (_lofo_combination_fold_scores below). Both sides now pay the same tax for
    being a maximum over representations, on both arms.

    This function itself still costs no fits — it is arithmetic on per-fold
    scores its caller has already computed.

    Returns {'delta_by_model', 'delta', 'wins', 'n_cells', 'per_model'}:
      delta_by_model  {model: mean over folds of (combination - baseline)}
      delta           mean of delta_by_model over the models that produced one
      wins / n_cells  (model, fold) cells the combination won, and cells compared
    """
    baselines_by_model = {}
    for rep_name in baseline_rep_names:
        for m, ms in (all_rep_scores.get(rep_name) or {}).items():
            folds = list(ms.get('fold_scores', []) or [])
            if folds:
                baselines_by_model.setdefault(m, {})[rep_name] = folds
    candidates_by_model = {}
    for m, ms in (pair_model_scores or {}).items():
        folds = list(ms.get('fold_scores', []) or [])
        if folds:
            candidates_by_model[m] = [{'name': 'combination',
                                       'fold_scores': folds}]
    if not candidates_by_model or not baselines_by_model:
        return {'delta_by_model': {}, 'delta': float('nan'),
                'wins': 0, 'n_cells': 0, 'per_model': {}}

    # 2026-10-05: metric_name lets fold_reselection refuse a direction that
    # contradicts the metric.
    hr = fold_reselection(candidates_by_model, baselines_by_model,
                        lower_is_better, metric_name=metric_name)
    delta_by_model, wins, n_cells = {}, 0, 0
    for m, r in hr.items():
        d = r.get('delta_mean_of_folds')
        if d is not None and not np.isnan(d):
            delta_by_model[m] = float(d)
        for pfd in (r.get('per_fold_delta') or []):
            if pfd is None:
                continue
            n_cells += 1
            if pfd > 0:
                wins += 1
    vals = list(delta_by_model.values())
    return {'delta_by_model': delta_by_model,
            'delta': float(np.mean(vals)) if vals else float('nan'),
            'wins': wins, 'n_cells': n_cells,
            'per_model': {m: {k: v for k, v in r.items()
                              if k in ('delta_mean_of_folds', 'per_fold_delta',
                                       'baseline_winners_per_fold',
                                       'n_paired_folds')}
                          for m, r in hr.items()}}


# =============================================================================
# LEAVE-ONE-FOLD-OUT RE-PICK ON THE COMBINATION SIDE
# =============================================================================
# 2026-09-12 — DECIDED (EXPERIMENTS_FINALIZED.md, c12). Two parts.
#
# First: a combination stays two full-width matrices side by side. c12 asks
# whether a second omics assay carries anything the first does not, which is a
# question about the assay, not about whether a selector can find it, and c12
# builds no hybrid anywhere else in the file. Nothing below selects features,
# spends a budget, or calls a search — np.hstack at full width, as before.
#
# Second: the leave-one-fold-out re-pick is applied to the combination side as
# well as the single side. For each held-out fold, take the best representation
# of modality A and the best of modality B as decided on the OTHER folds, and
# score that pair on this fold. Every representation is already scored on every
# fold, so the only new model fits are the pair matrices where the folds
# disagree about which representation wins; under the pruned ten-representation
# list most modalities carry exactly one representation, so on those nothing
# changes and no new fit happens.
#
# The re-pick is per model, which is what the single side already does
# (fold_reselection picks the baseline representation per model, per held-out
# fold). Doing it cross-model on the combination side would make the two arms
# differ in a second way.

def _is_bad_score(x) -> bool:
    """True when a fold score is missing, NaN, or not a number."""
    if x is None:
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return True
    return xf != xf  # NaN != NaN


def _lofo_best_rep_per_fold(rep_names, all_rep_scores, model_name, n_folds,
                            lower_is_better):
    """Which representation of one modality wins on every fold but one.

    Args:
        rep_names: the representations belonging to a single modality.
        all_rep_scores: {rep: {model: {'fold_scores': [...]}}} from phase 1.
        model_name: the fold scores are read for this model only.
        n_folds: number of folds.

    Returns a list of length n_folds. Entry f is the representation whose mean
    score over the folds OTHER than f is best, or None when no representation
    has a usable score on any other fold.

    Same arithmetic as _hr_lofo_score in _shared.py: mean over the other folds,
    strictly-better comparison, so a tie keeps the first representation in
    `rep_names` order.
    """
    items = []
    for rep_name in rep_names:
        scores = ((all_rep_scores.get(rep_name) or {}).get(model_name) or {})
        folds = list(scores.get('fold_scores') or [])
        if folds:
            items.append((rep_name, folds))

    out = [None] * n_folds
    if not items:
        return out

    better = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)
    for f in range(n_folds):
        best_name, best_mean = None, None
        for name, folds in items:
            others = [float(folds[k]) for k in range(len(folds))
                      if k != f and not _is_bad_score(folds[k])]
            if not others:
                continue
            mean_other = sum(others) / len(others)
            if best_mean is None or better(mean_other, best_mean):
                best_name, best_mean = name, mean_other
        out[f] = best_name
    return out


def _score_combination_on_fold(rep_names, reps, labels, fold_indices, fold_i,
                               model_factory, task_type, n_classes, metric,
                               model_name=None):
    """Fit one model on one fold's training rows and score the held-out rows.

    The matrix is the named representations placed side by side at full width —
    the same np.hstack eval_matrix scores, scaled the same way, on one fold.

    A fit or a prediction that raises returns NaN for this fold and names the
    representations and the model that raised. It used to propagate, so one
    representation the model could not fit ended the whole run.
    """
    X = np.hstack([reps[r] for r in rep_names])
    train_idx, val_idx = fold_indices[fold_i]
    return _fit_score_fold(X, labels, train_idx, val_idx, model_factory,
                           task_type, n_classes, metric,
                           label=f"fold {fold_i}: {'+'.join(rep_names)} with "
                                 f"{model_name or 'model'}")


def _fit_score_fold(X, labels, train_idx, val_idx, model_factory, task_type,
                    n_classes, metric, label=''):
    """Scale on the training rows, fit one model, score the held-out rows.

    2026-10-05: the one fit-and-score every arm of c12 goes through. Scaling is
    kirby.scaling.FeatureScaler (0/1 features left as they are, 2.3 item 17),
    fitted on the training rows only. It keeps the matrix's dtype, so float32
    representations stay float32. A fit or prediction that raises scores NaN
    and prints `label`; the run continues.
    """
    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = labels[train_idx], labels[val_idx]
    try:
        model = model_factory()
        model.fit(X_train, y_train)
        predict_fn = get_predict_fn(model, task_type, n_classes)
        return float(score_predictions(y_val, predict_fn(X_val), metric))
    except Exception as e:
        print(f"    FAILED {label} raised {type(e).__name__}: {e}")
        return float('nan')


def _lofo_combination_fold_scores(modalities, available_by_modality,
                                  all_rep_scores, reps, labels, fold_indices,
                                  model_factories, task_type, n_classes,
                                  metric, fit_cache, lower_is_better,
                                  fixed_reps=None,
                                  fixed_fold_scores=None):
    """Score one modality combination with its representations re-picked per fold.

    For each model and each held-out fold, every modality in `modalities`
    contributes the representation that is best on the other folds
    (_lofo_best_rep_per_fold). Those representations are placed side by side and
    scored on the held-out fold.

    `fixed_reps` / `fixed_fold_scores` are the combination already evaluated on
    every fold (one representation per modality, chosen over all folds). They
    seed `fit_cache`, so a fold whose re-pick lands on the same representations
    reuses the score instead of fitting again. `fit_cache` is shared across
    combinations: {(reps_tuple, model): {fold: score}}.

    Returns:
        {'fold_scores_by_model':   {model: [score per fold, NaN where none]},
         'reps_per_fold_by_model': {model: [[rep per modality] or None]},
         'n_new_fits':             model fits this call performed,
         'n_reused':               (model, fold) scores taken from the cache,
         'repick_applied':         False when there is only one fold, so no
                                   other fold exists to re-pick on}
    """
    n_folds = len(fold_indices)
    fixed_reps = list(fixed_reps or [])
    out = {'fold_scores_by_model': {}, 'reps_per_fold_by_model': {},
           'n_new_fits': 0, 'n_reused': 0, 'repick_applied': True}

    # Seed the cache with the already-computed combination.
    for model_name, folds in (fixed_fold_scores or {}).items():
        if not fixed_reps:
            break
        slot = fit_cache.setdefault((tuple(fixed_reps), model_name), {})
        for f, s in enumerate(list(folds or [])):
            if f < n_folds and not _is_bad_score(s):
                slot.setdefault(f, float(s))

    if n_folds < 2:
        # One fold: there is no other fold to re-pick on. The combination stays
        # the one chosen on that fold, which is what it was before; the honest
        # block already prints SINGLE-FOLD (optimistic) for this case.
        out['repick_applied'] = False
        for model_name, folds in (fixed_fold_scores or {}).items():
            out['fold_scores_by_model'][model_name] = [
                float(s) if not _is_bad_score(s) else float('nan')
                for s in list(folds or [])]
            out['reps_per_fold_by_model'][model_name] = [
                list(fixed_reps) for _ in list(folds or [])]
        return out

    for model_name, model_factory in model_factories.items():
        per_modality = {
            mod: _lofo_best_rep_per_fold(available_by_modality.get(mod, []),
                                         all_rep_scores, model_name, n_folds,
                                         lower_is_better)
            for mod in modalities
        }
        fold_scores, reps_per_fold = [], []
        for f in range(n_folds):
            chosen = [per_modality[mod][f] for mod in modalities]
            if any(c is None or c not in reps for c in chosen):
                fold_scores.append(float('nan'))
                reps_per_fold.append(None)
                continue
            slot = fit_cache.setdefault((tuple(chosen), model_name), {})
            if f in slot:
                out['n_reused'] += 1
            else:
                slot[f] = _score_combination_on_fold(
                    chosen, reps, labels, fold_indices, f, model_factory,
                    task_type, n_classes, metric, model_name=model_name)
                out['n_new_fits'] += 1
            fold_scores.append(slot[f])
            reps_per_fold.append(list(chosen))
        out['fold_scores_by_model'][model_name] = fold_scores
        out['reps_per_fold_by_model'][model_name] = reps_per_fold
    return out


# =============================================================================
# MODALITY GROUPS
# =============================================================================

# Map: modality name -> list of rep names (non-factory only)
# Only reps available via _REP_DISPATCH with rep_type != 'genomic'/'genomic_expr'
# Factory reps are excluded (they need per-fold fitting and are not cacheable here).
MODALITY_GROUPS = {
    'expression': [
        # Non-factory expression reps
        'gene_stats',
        'gene_chromosome_arm',
        # Foundation model embeddings
        'gene_geneformer',
        'gene_geneformer_v2',  # added 2026-05-25 — was silently absent from
                                # the ablation despite being the top expression
                                # rep on tcga_luad_vs_lusc / tcga_brca_pam50.
        'gene_bulkrnabert',
        'gene_scfoundation',
        'gene_scgpt',
        'gene_uce',
    ],
    'methylation': [
        'methyl_pca50',
        'methyl_hvg',
        'promoter_meth',
    ],
    'mutation': [
        'mutation_matrix',
        'mut_tmb',
        'mut_driver',
        'mut_pathway',
        'mut_pca50',
        'sig_sbs',
        'sig_dbs',
        'sig_id',
    ],
    'cnv': [
        'cnv_gistic',
        'cnv_pca50',
    ],
    'mirna': [
        'mirna_top200',
    ],
    'rppa': [
        'rppa',
    ],
    'immune': [
        'immune_cibersort',
        'immune_xcell',
    ],
    'clinical': [
        'clinical',
        'purity_consensus',
        'stemness_mrnasi',
        'heterogeneity_math',
        'msi_status',
        'hrd_score',
        'cimp_status',
    ],
    'wsi': [
        'wsi_ctranspath',
        'wsi_phikon',
        'wsi_hoptimus0',
        'wsi_gigapath',
        'wsi_moco',
        'wsi_titan',
        'wsi_uni2h',
    ],
}

# Factory-based expression reps — evaluated separately but included in group best
FACTORY_EXPRESSION_REPS = [
    'gene_hvg2000',
    'gene_pca100',
    'gene_hallmark',
    'gene_progeny',
    'gene_kegg',
    'gene_dorothea',
    'gene_reactome',
    'gene_oncogenic',
    'gene_combined_pathway',
    'gene_nmf50',
    'gene_ica50',
    'gene_sparse_pca50',
    'gene_autoencoder50',
    'gene_corr_modules',
    'gene_quantile_rank',
    'gene_immune',
    # Tier 2 expression-derived factory reps
    'lncrna_pca50',
    'splicing_psi50',
    'gsva_hallmark',
]


# =============================================================================
# METRIC HELPERS
# =============================================================================

# 2026-10-05: the metric name and direction come from get_domain_metric (the
# dataset registry), never from the task type. These are the metrics c12's
# scorer below computes; any other raises before a model is fitted.
C12_METRICS = ('auc', 'f1_weighted', 'r2')


def score_predictions(y_true, y_pred, metric: str) -> float:
    """Score held-out predictions under the dataset's registry metric.

    2026-10-05: keyed on the metric name from get_domain_metric, not the task
    type. 'auc' takes positive-class probabilities (or a probability matrix);
    'f1_weighted' takes class labels, or the argmax of a probability matrix (as
    before); 'r2' takes values.
    """
    y_pred = np.asarray(y_pred)
    if metric == 'r2':
        return float(r2_score(y_true, y_pred))
    if metric == 'auc':
        if y_pred.ndim == 2:
            return float(roc_auc_score(y_true, y_pred[:, 1]))
        return float(roc_auc_score(y_true, y_pred))
    if metric == 'f1_weighted':
        y_class = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
        return float(f1_score(y_true, y_class, average='weighted',
                              zero_division=0))
    raise ValueError(f"c12 scores only {C12_METRICS}; got metric '{metric}'")


# `higher_is_better(metric)` used to live here and returned True for ANY metric
# name handed to it, ignoring the argument. It had no caller. Deleted 2026-09-10
# (metric-direction audit) so nothing can pick it up and get a wrong answer for
# rmse or mae. 2026-10-05: the direction now comes from get_domain_metric's
# lower_is_better and is passed to every comparison below; no literal remains.


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_c12_model_factories(model_names, task_type, n_samples=None):
    """One factory per decided model, from find_best_hybrid.get_model.

    2026-10-05: replaces c12's own classifier catalog (rf, knn, mlp, xgboost,
    lightgbm) and the regression path's get_model_factories. Every model is now
    the one the engine fits, so lightgbm gets lgbm_n_jobs() (KIRBY_LGBM_THREADS)
    instead of n_jobs=-1, and xgboost no longer adds eval_metric='logloss'.
    get_model's `_max_samples` gate still skips a model above its row cap.
    ridge is removed everywhere and never asked for.
    """
    from kirby.find_best_hybrid import DEFAULT_MODELS, get_model

    factories = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            print(f"  WARNING: model '{name}' is not in find_best_hybrid's "
                  f"models, skipping")
            continue
        if get_model(name, task_type, n_samples=n_samples) is None:
            continue  # get_model printed the row-cap reason
        factories[name] = (lambda _n=name: get_model(_n, task_type))
    return factories


def get_predict_fn(model, task_type, n_classes):
    """Return a callable that produces scores for metric computation."""
    if task_type == 'regression':
        return model.predict
    # Classification — prefer predict_proba for AUC
    if hasattr(model, 'predict_proba'):
        if n_classes == 2:
            # Return 1D positive-class probabilities
            return lambda X: model.predict_proba(X)[:, 1]
        else:
            return model.predict_proba
    # Fallback: use decision_function or predict
    if hasattr(model, 'decision_function'):
        return model.decision_function
    return model.predict


# =============================================================================
# SINGLE-REP EVALUATION
# =============================================================================

def _fold_summary(fold_scores, dim):
    """Mean and spread over folds, keeping every fold's score in fold order.

    2026-10-05: one outer fold has no spread; 'score_std' is then None and
    'spread' says so, instead of a 0 that reads as no variation.
    """
    finite = [s for s in fold_scores if np.isfinite(s)]
    out = {
        'score_mean': float(np.mean(finite)) if finite else float('nan'),
        'dim': int(dim),
        # NaN folds keep their position: the re-score pairs arms by fold index.
        'fold_scores': [float(s) for s in fold_scores],
    }
    if len(fold_scores) < 2:
        out['score_std'] = None
        out['spread'] = 'one fold, no spread'
    else:
        out['score_std'] = (float(np.std(finite)) if finite
                            else float('nan'))
    return out


def eval_matrix(X, labels, fold_indices, model_factories, task_type,
                n_classes, metric, rep_label=None):
    """Score one matrix on every outer fold, for every model.

    2026-10-05: replaces eval_single_rep and eval_concat_reps, which were the
    same loop over cv_split folds. The folds are now the outer folds handed in
    (nested_cv_outer_folds), the same ones every combination is re-scored on.

    Returns {model_name: {'score_mean', 'score_std', 'dim', 'fold_scores'}}.
    A fold whose fit or prediction raises scores NaN and prints `rep_label` and
    the model; the run continues.
    """
    results = {}
    for model_name, model_factory in model_factories.items():
        fold_scores = [
            _fit_score_fold(X, labels, tr, va, model_factory, task_type,
                            n_classes, metric,
                            label=f"fold {f}: {rep_label or 'matrix'} with "
                                  f"{model_name}")
            for f, (tr, va) in enumerate(fold_indices)]
        results[model_name] = _fold_summary(fold_scores, X.shape[1])
    return results


def _better(a, b, lower_is_better):
    """True when score a beats score b under the metric's direction."""
    return a < b if lower_is_better else a > b


def best_score_across_models(model_results: dict, lower_is_better) -> float:
    """Return the best score_mean across all model results."""
    scores = [v['score_mean'] for v in model_results.values()
              if not np.isnan(v.get('score_mean', float('nan')))]
    if not scores:
        return float('nan')
    return min(scores) if lower_is_better else max(scores)


def best_model_for_result(model_results: dict, lower_is_better) -> str:
    """Return name of model with best score_mean."""
    best_name, best_score = None, None
    for name, vals in model_results.items():
        s = vals.get('score_mean', float('nan'))
        if not np.isnan(s) and (best_score is None
                                or _better(s, best_score, lower_is_better)):
            best_score = s
            best_name = name
    return best_name


def _signed_gain(new, old, lower_is_better):
    """new minus old, signed so a positive number means `new` is better."""
    return (old - new) if lower_is_better else (new - old)


def _mean_other_folds(fold_scores, f):
    """Mean of the finite scores on every fold but f; None when there is none."""
    others = [float(fold_scores[k]) for k in range(len(fold_scores))
              if k != f and not _is_bad_score(fold_scores[k])]
    return (sum(others) / len(others)) if others else None


def _lofo_winner_on_fold(named_fold_lists, f, lower_is_better):
    """Name picked for fold f by the fold_reselection rule, or None (2026-10-05).

    Same rule as _shared._hr_lofo_score: best mean over the other folds,
    strictly better replaces, so ties keep the first in order; with one fold,
    best on that fold. Computed per fold index here because the helper's
    winners list skips folds with no usable item and so is not aligned by fold.
    """
    clean = [(n, list(fs)) for n, fs in named_fold_lists if fs]
    if not clean:
        return None
    F = max(len(fs) for _, fs in clean)
    better = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)
    best_name, best_val = None, None
    for n, fs in clean:
        if F == 1:
            v = None if (not fs or _is_bad_score(fs[0])) else float(fs[0])
        else:
            others = [float(fs[k]) for k in range(len(fs))
                      if k != f and not _is_bad_score(fs[k])]
            v = (sum(others) / len(others)) if others else None
        if v is None:
            continue
        if best_val is None or better(v, best_val):
            best_name, best_val = n, v
    return best_name


def _per_fold_records(model_keys, fold_indices, available_by_modality,
                      all_rep_scores, pairwise_results, all_combined_result,
                      reselection_result, reps, metric, lower_is_better,
                      candidates_by_model=None, baselines_by_model=None):
    """One record per model per outer fold (2026-10-05).

    What each record holds: the single representation the combinations are
    compared against on this fold (picked on the other folds, as
    fold_reselection picks it) with its held-out score; each modality
    combination's representations on this fold, its width in features and its
    held-out score. c12 builds no hybrid, so 'hybrids' is empty. Every
    'selection_score_other_folds' is the mean held-out score over the other
    folds, the number the pick was made on; c12 computes no score on a fold's
    own training rows.
    """
    def _clean(x):
        return None if _is_bad_score(x) else float(x)

    records = []
    scope = _selection_scope(len(fold_indices))
    for m in model_keys:
        rr = (reselection_result or {}).get(m) or {}
        # 2026-10-05: baseline_winners_per_fold skips folds where no
        # representation scored on the other folds, so entry f need not be
        # fold f. The winner is recomputed per fold index from the same lists
        # fold_reselection was given; baseline_fold_scores is already aligned.
        base_lists = list(((baselines_by_model or {}).get(m) or {}).items())
        cand_lists = [(c['name'], c.get('fold_scores') or [])
                      for c in ((candidates_by_model or {}).get(m) or [])]
        base_scores = list(rr.get('baseline_fold_scores') or [])
        comb_scores = list(rr.get('hybrid_fold_scores') or [])
        for f, (tr, va) in enumerate(fold_indices):
            b_rep = _lofo_winner_on_fold(base_lists, f, lower_is_better)
            c_pick = _lofo_winner_on_fold(cand_lists, f, lower_is_better)
            b_folds = (((all_rep_scores.get(b_rep) or {}).get(m) or {})
                       .get('fold_scores') or []) if b_rep else []
            combos = {}
            sources = [(k, pr) for k, pr in pairwise_results.items()]
            if all_combined_result:
                sources.append(('all_combined', all_combined_result))
            for key, pr in sources:
                reps_f = ((pr.get('lofo_pair_reps_per_fold') or {})
                          .get(m) or [None] * len(fold_indices))
                sc_f = ((pr.get('lofo_pair_fold_scores') or {})
                        .get(m) or [float('nan')] * len(fold_indices))
                chosen = reps_f[f] if f < len(reps_f) else None
                combos[key] = {
                    'modalities': list(pr.get('modalities') or []),
                    'reps': list(chosen) if chosen else None,
                    'width_features': (int(sum(reps[r].shape[1]
                                               for r in chosen))
                                       if chosen else None),
                    'held_out_score': _clean(sc_f[f]) if f < len(sc_f)
                                      else None,
                    'selection_score_other_folds': (
                        {r: _mean_other_folds(
                            ((all_rep_scores.get(r) or {}).get(m) or {})
                            .get('fold_scores') or [], f) for r in chosen}
                        if chosen else None),
                }
            records.append({
                'model': m,
                'fold': f,
                'n_train_rows': int(len(tr)),
                'n_held_out_rows': int(len(va)),
                'metric_name': metric,
                'lower_is_better': bool(lower_is_better),
                'selection_scope': scope,
                'aug_condition': 'none',
                # 2026-10-05: the combination the per-model table picked for
                # this fold (by its re-picked scores on the other folds) and
                # its held-out score, next to the single representation below.
                'combination_picked': c_pick,
                'combination_picked_held_out_score': (
                    _clean(comb_scores[f]) if f < len(comb_scores) else None),
                'base_reps': ([{
                    'rep': b_rep,
                    'width_features': int(reps[b_rep].shape[1])
                                      if b_rep in reps else None,
                    'selection_score_other_folds':
                        _mean_other_folds(b_folds, f),
                    'held_out_score': _clean(base_scores[f])
                                      if f < len(base_scores) else None,
                }] if b_rep else []),
                'combinations': combos,
                'hybrids': [],
            })
    return records


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Run TCGA modality ablation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C12: TCGA MODALITY ABLATION")
    print("=" * 70)

    if data.get('domain') != 'tcga':
        raise ValueError(
            f"C12 is TCGA-specific. Got domain='{data.get('domain')}'. "
            "Use a tcga_* dataset name."
        )

    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    reps = data['reps']

    n_classes = int(len(np.unique(labels)))
    # 2026-10-05: metric name AND direction from the dataset registry, never
    # assumed from the task type (handoff 2.4 step 11).
    data.setdefault('dataset_name', dataset_name)
    metric, _metric_fn, lower_is_better = get_domain_metric(data)
    if metric not in C12_METRICS:
        raise ValueError(
            f"{dataset_name}'s registry metric is '{metric}'; c12 scores only "
            f"{C12_METRICS}.")
    if metric == 'auc' and n_classes != 2:
        raise ValueError(f"{dataset_name}: metric auc needs 2 classes, "
                         f"got {n_classes}.")
    # 2026-10-05: the outer fold count comes from config['n_splits'] only
    # (handoff 2.4 step 14); 1 means one held-out split.
    n_splits = int(config['n_splits'])
    seed = int(config.get('seed', 42))

    print(f"  Dataset:   {dataset_name}")
    print(f"  Task:      {task_type} ({n_classes} classes)")
    print(f"  Metric:    {metric} "
          f"({'lower' if lower_is_better else 'higher'} is better)")
    print(f"  n_samples: {len(labels)}, outer folds asked for: {n_splits}")

    # Encode classification labels to int if needed
    if task_type == 'classification':
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    # 2026-10-05: the dataset's two decided models only (DECISIONS.md Step 1,
    # rule R1). config['models'] is set from the same record by decided_config;
    # a --models narrowing on the command line is kept, anything outside the
    # decided pair is dropped with a printed line.
    decided = list(decided_models(dataset_name, 'general'))
    asked = list(config.get('models') or decided)
    model_names = [m for m in asked if m in decided] or decided
    dropped = [m for m in asked if m not in decided]
    if dropped:
        print(f"  Models outside {dataset_name}'s decided pair {decided} "
              f"dropped: {dropped}")
    model_factories = get_c12_model_factories(model_names, task_type,
                                              n_samples=len(labels))
    if not model_factories:
        raise RuntimeError("No valid model factories available.")
    print(f"  Models:    {list(model_factories)} (decided: {decided})")

    # 2026-10-05: representations from DATASET_REPS[dataset]['general'] only
    # (handoff 2.4 step 9). The filter is on the per-representation loops below,
    # not on data['reps']. Augmentation: Step 8 cut it on both TCGA datasets
    # and c12 has no augmentation comparison, so nothing is added.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    _entry = DATASET_REPS.get(dataset_name) or {}
    if not _entry.get('general'):
        raise ValueError(f"{dataset_name} has no DATASET_REPS['general'] list.")
    allowed_reps = [r for r in _entry['general'] if r in reps]
    _missing = [r for r in _entry['general'] if r not in reps]
    if _missing:
        print(f"  NOTE: approved representations not loaded: {_missing}")
    _ignored = sorted(r for r in reps if r not in _entry['general'])
    if _ignored:
        print(f"  NOTE: loaded but not in {dataset_name}'s approved list, "
              f"not scored: {_ignored}")
    _aug = list(decided_aug_conditions(dataset_name, 'general'))
    if _aug != ['none']:
        print(f"  NOTE: the decided grid keeps augmentation {_aug} on this "
              f"dataset; c12 compares no augmentation, so it is not used.")

    # 2026-10-05: the outer folds are nested_cv_outer_folds, the single source
    # the per-fold searches use (ocv{n}f{i}s{seed}). It replaces cv_split; for a
    # classification task both build StratifiedGroupKFold(shuffle, seed), so
    # the folds are the same ones. Every arm below is scored on this list.
    fold_indices = [(np.asarray(tr), np.asarray(va)) for tr, va in
                    nested_cv_outer_folds(
                        {r: reps[r] for r in allowed_reps[:1]}, labels, groups,
                        n_splits, seed, task_type)]
    n_folds = len(fold_indices)
    if n_folds != n_splits:
        print(f"  NOTE: {n_splits} outer folds asked for, {n_folds} built "
              f"(too few groups).")
    print(f"  Outer folds: {n_folds}"
          + (" (one held-out split: no re-pick on other folds, no spread)"
             if n_folds < 2 else ""))

    # Scores already fitted, keyed by (representations, model) -> {fold: score}.
    # Shared by every combination so a re-pick that lands on representations
    # already fitted on that fold costs nothing.
    combination_fit_cache = {}

    # =========================================================================
    # 1. Identify which reps are available per modality group
    # =========================================================================
    available_by_modality = {}
    for mod_name, mod_reps in MODALITY_GROUPS.items():
        avail = [r for r in mod_reps if r in allowed_reps]
        if avail:
            available_by_modality[mod_name] = avail
        else:
            print(f"  NOTE: modality '{mod_name}' — no reps available in loaded data")

    # Also check for any expression reps (factory or otherwise) in loaded data
    expr_avail = [r for r in FACTORY_EXPRESSION_REPS if r in allowed_reps]
    # Plus non-factory expression reps already in MODALITY_GROUPS['expression']
    expr_avail += [r for r in MODALITY_GROUPS.get('expression', [])
                   if r in allowed_reps and r not in expr_avail]
    if expr_avail:
        available_by_modality['expression'] = expr_avail

    # 2026-10-05: an approved representation in no modality group is named,
    # so a representation dropped from the comparison shows in the log.
    _grouped = {r for v in available_by_modality.values() for r in v}
    _ungrouped = [r for r in allowed_reps if r not in _grouped]
    if _ungrouped:
        print(f"  NOTE: approved representations in no modality group, "
              f"not scored: {_ungrouped}")

    active_modalities = sorted(available_by_modality.keys())
    print(f"\n  Active modalities ({len(active_modalities)}): {', '.join(active_modalities)}")
    for mod in active_modalities:
        print(f"    {mod}: {available_by_modality[mod]}")

    # =========================================================================
    # 2. Score each rep individually
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 1: Individual rep scores")
    print("-" * 60)

    all_rep_scores = {}  # rep_name -> {model_name -> {score_mean, score_std, dim}}

    all_candidate_reps = set()
    for mod_reps in available_by_modality.values():
        all_candidate_reps.update(mod_reps)

    for rep_name in sorted(all_candidate_reps):
        X = reps[rep_name]
        result = eval_matrix(X, labels, fold_indices, model_factories,
                             task_type, n_classes, metric, rep_label=rep_name)
        all_rep_scores[rep_name] = result
        best_s = best_score_across_models(result, lower_is_better)
        best_m = best_model_for_result(result, lower_is_better)
        print(f"    {rep_name:<30} {metric}={best_s:+.4f}  dim={X.shape[1]}  "
              f"best_model={best_m}")

    # =========================================================================
    # 3. Find best rep per modality (for use in pairwise combinations)
    # =========================================================================
    best_rep_per_modality = {}   # modality -> rep_name
    best_score_per_modality = {}  # modality -> float

    # Chosen over all folds and all models: the 'same_fold' numbers. The
    # per-fold re-pick below is what the comparison reads.
    _worst = np.inf if lower_is_better else -np.inf
    for mod_name, mod_reps in available_by_modality.items():
        best_r, best_s = None, _worst
        for rep_name in mod_reps:
            if rep_name not in all_rep_scores:
                continue
            s = best_score_across_models(all_rep_scores[rep_name],
                                         lower_is_better)
            if not np.isnan(s) and _better(s, best_s, lower_is_better):
                best_s = s
                best_r = rep_name
        best_rep_per_modality[mod_name] = best_r
        best_score_per_modality[mod_name] = best_s

    print(f"\n  Best rep per modality:")
    for mod in active_modalities:
        rep = best_rep_per_modality.get(mod)
        score = best_score_per_modality.get(mod, float('nan'))
        dim = reps[rep].shape[1] if rep and rep in reps else '?'
        print(f"    {mod:<15} {rep:<30} {metric}={score:+.4f}  dim={dim}")

    # =========================================================================
    # 4. Pairwise modality combinations
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 2: Pairwise modality combinations")
    print("-" * 60)
    # One line per pair of modalities. 'lofo Δ' re-picks both representations
    # of the pair AND the single representation it is compared against on the
    # other folds; 'fits' counts the extra model fits that re-pick needed,
    # which is zero for a pair whose folds all agree on the representations.

    # Reference: best expression score (or best single modality score overall)
    if 'expression' in best_rep_per_modality and best_rep_per_modality['expression']:
        ref_modality = 'expression'
    else:
        # Fall back to best-scoring modality
        _pick = min if lower_is_better else max
        ref_modality = _pick(active_modalities,
                             key=lambda m: best_score_per_modality.get(m, _worst))

    ref_score = best_score_per_modality.get(ref_modality, float('nan'))
    print(f"  Reference modality: {ref_modality} ({metric}={ref_score:+.4f})")

    pairwise_results = {}  # (mod_a, mod_b) -> {score_mean, score_std, dim, model_scores}

    modalities_with_reps = [m for m in active_modalities
                             if best_rep_per_modality.get(m) is not None]

    for mod_a, mod_b in itertools.combinations(modalities_with_reps, 2):
        rep_a = best_rep_per_modality[mod_a]
        rep_b = best_rep_per_modality[mod_b]

        if rep_a is None or rep_b is None:
            continue
        if rep_a not in reps or rep_b not in reps:
            continue

        pair_key = f"{mod_a}+{mod_b}"
        X_a, X_b = reps[rep_a], reps[rep_b]
        result = eval_matrix(np.hstack([X_a, X_b]), labels, fold_indices,
                             model_factories, task_type, n_classes, metric,
                             rep_label=f'{rep_a}+{rep_b}')

        best_s = best_score_across_models(result, lower_is_better)
        best_m = best_model_for_result(result, lower_is_better)
        total_dim = X_a.shape[1] + X_b.shape[1]

        pairwise_results[pair_key] = {
            'modalities': [mod_a, mod_b],
            'reps_used': [rep_a, rep_b],
            'score_mean': best_s,
            'score_std': result.get(best_m, {}).get('score_std', float('nan'))
                         if best_m else float('nan'),
            'total_dim': total_dim,
            'best_model': best_m,
            # 2026-10-05: the three deltas below are signed so a positive
            # number means the pair scores better, whatever the direction.
            'model_scores': {k: {'score_mean': v['score_mean'],
                                  'score_std': v['score_std'],
                                  'dim': v['dim'],
                                  'fold_scores': v.get('fold_scores', [])}
                             for k, v in result.items()},
            'delta_vs_a': _signed_gain(
                best_s, best_score_per_modality.get(mod_a, float('nan')),
                lower_is_better),
            'delta_vs_b': _signed_gain(
                best_s, best_score_per_modality.get(mod_b, float('nan')),
                lower_is_better),
            'delta_vs_best_single': _signed_gain(
                best_s,
                (min if lower_is_better else max)(
                    best_score_per_modality.get(mod_a, _worst),
                    best_score_per_modality.get(mod_b, _worst)),
                lower_is_better),
        }

        # 2026-09-12 — DECIDED: the leave-one-fold-out re-pick now applies to
        # the pair side too. On each held-out fold the pair is the best
        # representation of modality A and the best of modality B as decided on
        # the other folds, scored on this fold. The single representation it is
        # compared against is re-picked the same way, as before. Neither side is
        # a maximum taken on the fold it is read on any more.
        # 'score_mean', 'delta_vs_a/b' and 'delta_vs_best_single' above are
        # untouched: they are still the pair chosen over all folds.
        _lofo = _lofo_combination_fold_scores(
            [mod_a, mod_b], available_by_modality, all_rep_scores, reps, labels,
            fold_indices, model_factories, task_type, n_classes, metric,
            combination_fit_cache, lower_is_better,
            fixed_reps=[rep_a, rep_b],
            fixed_fold_scores={m: ms.get('fold_scores', [])
                               for m, ms in result.items()})
        _lofo_model_scores = {m: {'fold_scores': fs}
                              for m, fs in _lofo['fold_scores_by_model'].items()}

        _hp = _reselected_pair_delta(
            _lofo_model_scores,
            list(available_by_modality.get(mod_a, []))
            + list(available_by_modality.get(mod_b, [])),
            all_rep_scores, lower_is_better, metric_name=metric)
        pairwise_results[pair_key]['reselected_delta'] = _hp['delta']
        pairwise_results[pair_key]['reselected_delta_by_model'] = _hp['delta_by_model']
        pairwise_results[pair_key]['reselected_per_model'] = _hp['per_model']
        pairwise_results[pair_key]['reselected_wins'] = _hp['wins']
        pairwise_results[pair_key]['reselected_n_cells'] = _hp['n_cells']
        # What each side's representations were chosen on.
        pairwise_results[pair_key]['same_fold_reps_selected_on'] = 'all folds'
        pairwise_results[pair_key]['reps_selected_on'] = (
            'other folds, per held-out fold and per model'
            if _lofo['repick_applied'] else 'all folds (single fold: no re-pick)')
        # The pair actually scored on each held-out fold, per model, and the
        # scores it got there. Both feed reselected_delta.
        pairwise_results[pair_key]['lofo_pair_fold_scores'] = \
            _lofo['fold_scores_by_model']
        pairwise_results[pair_key]['lofo_pair_reps_per_fold'] = \
            _lofo['reps_per_fold_by_model']
        pairwise_results[pair_key]['lofo_pair_new_fits'] = _lofo['n_new_fits']

        delta = pairwise_results[pair_key]['delta_vs_best_single']
        sign = '+' if delta >= 0 else ''
        print(f"    {pair_key:<35} {metric}={best_s:+.4f}  "
              f"Δbest={sign}{delta:.4f}  lofo Δ={_hp['delta']:+.4f} "
              f"({_hp['wins']}/{_hp['n_cells']})  dim={total_dim}  model={best_m}"
              f"  +{_lofo['n_new_fits']} fits")

    # =========================================================================
    # 5. All-modalities combined
    # =========================================================================
    print("\n" + "-" * 60)
    print("  PHASE 3: All modalities combined")
    print("-" * 60)

    all_arrays = []
    all_mods_used = []
    all_reps_used = []

    for mod in modalities_with_reps:
        rep = best_rep_per_modality[mod]
        if rep and rep in reps:
            all_arrays.append(reps[rep])
            all_mods_used.append(mod)
            all_reps_used.append(rep)

    all_combined_result = None
    if len(all_arrays) >= 2:
        all_result = eval_matrix(np.hstack(all_arrays), labels, fold_indices,
                                 model_factories, task_type, n_classes, metric,
                                 rep_label='+'.join(all_reps_used))
        total_dim = sum(a.shape[1] for a in all_arrays)
        best_all_s = best_score_across_models(all_result, lower_is_better)
        best_all_m = best_model_for_result(all_result, lower_is_better)
        best_single = (min if lower_is_better else max)(
            best_score_per_modality.get(m, _worst)
            for m in modalities_with_reps)
        # Signed so a positive number means the combination scores better.
        delta_all = _signed_gain(best_all_s, best_single, lower_is_better)

        all_combined_result = {
            'modalities': all_mods_used,
            'reps_used': all_reps_used,
            'score_mean': best_all_s,
            'score_std': all_result.get(best_all_m, {}).get('score_std', float('nan'))
                         if best_all_m else float('nan'),
            'total_dim': total_dim,
            'best_model': best_all_m,
            'model_scores': {k: {'score_mean': v['score_mean'],
                                  'score_std': v['score_std'],
                                  'dim': v['dim'],
                                  'fold_scores': v.get('fold_scores', [])}
                             for k, v in all_result.items()},
            'delta_vs_best_single': delta_all,
        }

        # 2026-09-12 — the same per-fold re-pick, on the arm built the same way.
        # The decision was written about the pair; the all-modalities matrix is
        # the identical construction over every modality (one best
        # representation each, side by side at full width), so it takes the
        # re-pick too. Leaving it on the all-folds pick would have made this arm
        # differ from the pairs in a second way.
        _la = _lofo_combination_fold_scores(
            all_mods_used, available_by_modality, all_rep_scores, reps, labels,
            fold_indices, model_factories, task_type, n_classes, metric,
            combination_fit_cache, lower_is_better,
            fixed_reps=all_reps_used,
            fixed_fold_scores={m: ms.get('fold_scores', [])
                               for m, ms in all_result.items()})
        _la_model_scores = {m: {'fold_scores': fs}
                            for m, fs in _la['fold_scores_by_model'].items()}

        _ha = _reselected_pair_delta(
            _la_model_scores,
            [r for m in modalities_with_reps
             for r in available_by_modality.get(m, [])],
            all_rep_scores, lower_is_better, metric_name=metric)
        all_combined_result['reselected_delta'] = _ha['delta']
        all_combined_result['reselected_delta_by_model'] = _ha['delta_by_model']
        all_combined_result['reselected_per_model'] = _ha['per_model']
        all_combined_result['reselected_wins'] = _ha['wins']
        all_combined_result['reselected_n_cells'] = _ha['n_cells']
        all_combined_result['same_fold_reps_selected_on'] = 'all folds'
        all_combined_result['reps_selected_on'] = (
            'other folds, per held-out fold and per model'
            if _la['repick_applied'] else 'all folds (single fold: no re-pick)')
        all_combined_result['lofo_pair_fold_scores'] = _la['fold_scores_by_model']
        all_combined_result['lofo_pair_reps_per_fold'] = _la['reps_per_fold_by_model']
        all_combined_result['lofo_pair_new_fits'] = _la['n_new_fits']

        sign = '+' if delta_all >= 0 else ''
        print(f"  All combined ({len(all_mods_used)} modalities): "
              f"{metric}={best_all_s:+.4f}  Δ={sign}{delta_all:.4f}  "
              f"lofo Δ={_ha['delta']:+.4f} ({_ha['wins']}/{_ha['n_cells']})  "
              f"dim={total_dim}  model={best_all_m}  "
              f"+{_la['n_new_fits']} fits")
        print(f"  Modalities: {', '.join(all_mods_used)}")
    else:
        print("  Not enough modalities for all-combined evaluation.")

    # =========================================================================
    # 6. Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("  MODALITY ABLATION SUMMARY")
    print("=" * 70)
    print(f"\n  Dataset: {dataset_name} | Metric: {metric} | "
          f"Task: {task_type} | n_folds: {n_folds}")

    print(f"\n  Individual modality scores (best rep within group):")
    print(f"    {'Modality':<15} {'Best rep':<30} {metric:>8}  {'Dim':>6}")
    print("    " + "-" * 64)
    for mod in sorted(active_modalities,
                       key=lambda m: _signed_gain(
                           0.0, best_score_per_modality.get(m, _worst),
                           lower_is_better)):
        rep = best_rep_per_modality.get(mod, 'N/A')
        score = best_score_per_modality.get(mod, float('nan'))
        dim = reps[rep].shape[1] if rep and rep in reps else '?'
        print(f"    {mod:<15} {rep:<30} {score:>8.4f}  {dim:>6}")

    # HELPS / HURTS is read off the leave-one-fold-out delta. Since 2026-09-12
    # both of its sides are re-picked on the other folds: the pair's own two
    # representations and the single representation it is compared against.
    # 'Delta' beside it is the same-fold difference: both of its sides are a
    # maximum taken on the folds they are then read on, so it has no honest
    # direction.
    print(f"\n  Pairwise combinations (HELPS/HURTS from the leave-one-fold-out delta):")
    print(f"    {'Pair':<35} {metric:>8}  {'Delta':>8}  {'lofo Δ':>8}  "
          f"{'wins':>7}  {'Dim':>6}")
    print("    " + "-" * 80)
    for pair_key, pr in sorted(pairwise_results.items(),
                                key=lambda x: -(x[1].get('reselected_delta')
                                                if x[1].get('reselected_delta') is not None
                                                and not np.isnan(x[1]['reselected_delta'])
                                                else -np.inf)):
        score = pr['score_mean']
        delta = pr['delta_vs_best_single']
        hdelta = pr.get('reselected_delta', float('nan'))
        dim = pr['total_dim']
        wins = f"{pr.get('reselected_wins', 0)}/{pr.get('reselected_n_cells', 0)}"
        flag = ''
        if not np.isnan(hdelta):
            flag = (' <-- HELPS' if hdelta > 0.01
                    else (' <-- HURTS' if hdelta < -0.01 else ''))
        print(f"    {pair_key:<35} {score:>8.4f}  {delta:>+8.4f}  "
              f"{hdelta:>+8.4f}  {wins:>7}  {dim:>6}{flag}")
        # 2026-10-05: the HELPS/HURTS delta is the mean over the decided
        # models; each model's own delta is printed under it.
        _bym = pr.get('reselected_delta_by_model') or {}
        if _bym:
            print("      per model: " + ", ".join(
                f"{m} {d:+.4f}" for m, d in sorted(_bym.items())))

    if all_combined_result:
        print(f"\n  All modalities combined:")
        print(f"    {metric}={all_combined_result['score_mean']:+.4f}  "
              f"Δ={all_combined_result['delta_vs_best_single']:+.4f}  "
              f"lofo Δ={all_combined_result.get('reselected_delta', float('nan')):+.4f}  "
              f"dim={all_combined_result['total_dim']}")

    # Rank modalities by contribution when paired with expression
    if 'expression' in modalities_with_reps:
        print(f"\n  Effect of adding each modality to expression "
              f"(HELPS/HURTS from the leave-one-fold-out delta):")
        print(f"    {'Added modality':<20} {'Δ'+metric:>10}  {'lofo Δ':>10}  "
              f"{'Combined '+metric:>16}  {'expression only':>15}")
        print("    " + "-" * 82)
        expr_score = best_score_per_modality.get('expression', float('nan'))
        partner_deltas = []
        for mod in modalities_with_reps:
            if mod == 'expression':
                continue
            pair_key = '+'.join(sorted(['expression', mod]))
            # Try both orderings
            if pair_key not in pairwise_results:
                pair_key = '+'.join([mod, 'expression'])
            if pair_key not in pairwise_results:
                continue
            pr = pairwise_results[pair_key]
            delta = _signed_gain(pr['score_mean'], expr_score, lower_is_better)
            # Against expression alone, re-picking which expression
            # representation on each held-out fold's other folds.
            # 2026-09-12 — the pair side is re-picked on the other folds too,
            # so this reads the per-fold pair scores, not the all-folds pair.
            _he = _reselected_pair_delta(
                {m: {'fold_scores': fs} for m, fs in
                 pr.get('lofo_pair_fold_scores', {}).items()},
                available_by_modality.get('expression', []),
                all_rep_scores, lower_is_better, metric_name=metric)
            pr['reselected_delta_vs_expression'] = _he['delta']
            pr['reselected_delta_vs_expression_by_model'] = _he['delta_by_model']
            partner_deltas.append((mod, delta, pr['score_mean'], _he))

        for mod, delta, combined_score, _he in sorted(
                partner_deltas,
                key=lambda x: -(x[3]['delta'] if not np.isnan(x[3]['delta'])
                                else -np.inf)):
            hd = _he['delta']
            flag = ''
            if not np.isnan(hd):
                flag = (' <-- HELPS' if hd > 0.01
                        else (' <-- HURTS' if hd < -0.01 else ''))
            print(f"    {mod:<20} {delta:>+10.4f}  {hd:>+10.4f}  "
                  f"{combined_score:>16.4f}  {expr_score:>15.4f}{flag}")

    # =========================================================================
    # 7. Honest per-model re-score (leave-one-fold-out; NO cross-model argmax)
    # =========================================================================
    # The prints above pick the best score across models AND across cells
    # (double-sided cross-model max), which is optimistically biased. Here we
    # instead persist the raw per-fold scores per model for BOTH arms and let
    # the shared fold_reselection helper re-pick the winner on each held-out fold,
    # keeping every model's result separately (no collapse to one max cell).
    #
    # Mapping onto the helper's contract:
    #   hybrid candidates = every modality combination (pairwise + all-combined),
    #                       each one re-picked per held-out fold since 2026-09-12
    #   baselines         = every individual rep
    # 2026-10-05: lower_is_better is get_domain_metric's, set at the top of
    # run(); the literal False that stood here is gone.

    model_keys = list(model_factories.keys())

    # 2026-09-12 — DECIDED: each combination enters as its per-fold re-picked
    # scores, not the all-folds pick. Fold f's score is the combination whose
    # representations won on the other folds, under this model.
    candidates_by_model = {m: [] for m in model_keys}
    for pair_key, pr in pairwise_results.items():
        for m, fold_scores in pr.get('lofo_pair_fold_scores', {}).items():
            if m not in candidates_by_model:
                candidates_by_model[m] = []
            candidates_by_model[m].append({
                'name': pair_key,
                'fold_scores': list(fold_scores),
            })
    if all_combined_result:
        for m, fold_scores in all_combined_result.get(
                'lofo_pair_fold_scores', {}).items():
            if m not in candidates_by_model:
                candidates_by_model[m] = []
            candidates_by_model[m].append({
                'name': 'all_combined',
                'fold_scores': list(fold_scores),
            })

    baselines_by_model = {m: {} for m in model_keys}
    for rep_name, model_scores in all_rep_scores.items():
        for m, ms in model_scores.items():
            if m not in baselines_by_model:
                baselines_by_model[m] = {}
            baselines_by_model[m][rep_name] = list(ms.get('fold_scores', []))

    # Drop empty per-model entries so the helper only sees models with data.
    candidates_by_model = {m: c for m, c in candidates_by_model.items() if c}
    baselines_by_model = {m: b for m, b in baselines_by_model.items() if b}

    reselection_block = {
        'lower_is_better': lower_is_better,
        # Every fold counts. The candidates here are the arms this file built itself,
        # not a search's recipe list, so no fold filtered them.
        'exclude_folds': [],
        'candidates_by_model': candidates_by_model,
        'baselines_by_model': baselines_by_model,
    }

    # Compute the honest per-model deltas now for the log; readers can re-derive
    # from the persisted block via fold_reselection(...).
    reselection_result = fold_reselection(candidates_by_model, baselines_by_model,
                                   lower_is_better, metric_name=metric)

    print("\n" + "-" * 60)
    print("  HONEST PER-MODEL RE-SCORE (leave-one-fold-out, per model)")
    print("-" * 60)
    # 2026-10-05: 'combination' is the modality combination picked on the
    # other folds (c12 builds no hybrid); the delta printed is
    # delta_mean_of_folds, the mean of the per-fold differences over folds
    # where both sides scored, not the difference of the two means.
    if reselection_result:
        print(f"    {'Model':<12} {'combination':>12} {'best_single':>12} "
              f"{'paired Δ':>10} {'folds':>6}  note")
        for m in sorted(reselection_result.keys()):
            r = reselection_result[m]
            note = ('one fold: no re-pick on other folds, no spread'
                    if r.get('single_fold') else '')
            print(f"    {m:<12} {r['hybrid_reselected']:>12.4f} "
                  f"{r['best_single_reselected']:>12.4f} "
                  f"{r['delta_mean_of_folds']:>+10.4f} "
                  f"{r.get('n_paired_folds', 0):>6}  {note}")
    else:
        print("    (no models with usable per-fold data)")

    # =========================================================================
    # Per-fold records (2026-10-05, handoff 2.4 step 11)
    # =========================================================================
    per_fold = _per_fold_records(
        model_keys, fold_indices, available_by_modality, all_rep_scores,
        pairwise_results, all_combined_result, reselection_result, reps,
        metric, lower_is_better,
        candidates_by_model=candidates_by_model,
        baselines_by_model=baselines_by_model)

    # =========================================================================
    # Build final results dict
    # =========================================================================
    output = {
        'dataset': dataset_name,
        'result_version': RESULT_VERSION,
        'task_type': task_type,
        'n_classes': n_classes,
        'metric': metric,
        'metric_name': metric,
        'lower_is_better': bool(lower_is_better),
        'models': list(model_keys),
        'decided_models': decided,
        'representations_approved': allowed_reps,
        'aug_condition': 'none',
        'selection_scope': _selection_scope(n_folds),
        'selection_design': (
            "handoff 2.3 (B), decided 2026-10-05: one representation per "
            "modality side by side at full width, no hybrids, on the decided "
            "models, per outer fold"),
        'hybrids': [],
        'n_splits': n_splits,
        'n_folds': n_folds,
        'outer_fold_seed': seed,
        'spread': ('one fold, no spread' if n_folds < 2 else 'over folds'),
        'per_fold': per_fold,
        'n_samples': int(len(labels)),
        'active_modalities': active_modalities,
        'available_reps_per_modality': available_by_modality,
        'individual': {
            'best_rep_per_modality': best_rep_per_modality,
            'best_score_per_modality': {k: float(v)
                                         for k, v in best_score_per_modality.items()},
            'all_rep_scores': {
                rep: {
                    model: {
                        sk: (sv if isinstance(sv, (str, list)) or sv is None
                             else float(sv))
                        for sk, sv in scores.items()
                    }
                    for model, scores in model_scores.items()
                }
                for rep, model_scores in all_rep_scores.items()
            },
        },
        'pairwise': pairwise_results,
        'all_combined': all_combined_result,
        # Honest, per-model, leave-one-fold-out re-score block. Matches the
        # shared contract exactly: a reader loads this and calls
        # fold_reselection(block['candidates_by_model'], block['baselines_by_model'],
        #                block['lower_is_better']).
        'fold_reselection': reselection_block,
        # Pre-computed per-model honest deltas (convenience; re-derivable from
        # 'fold_reselection' above). NOT a cross-model max — one entry per model.
        'fold_reselection_result': reselection_result,
    }

    return output


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = create_experiment_parser(
        'C12: TCGA Modality Ablation',
        has_delta=False,
        has_augment=False,
    )
    # Add n_splits override
    parser.add_argument('--n-splits', type=int, default=None,
                        help='Override number of outer CV folds from mode config')
    args = parser.parse_args()

    if not args.dataset.startswith('tcga_'):
        parser.error(f"C12 requires a tcga_* dataset. Got: '{args.dataset}'")

    config = get_config(args)
    if args.n_splits is not None:
        config = dict(config)
        config['n_splits'] = args.n_splits

    # 2026-10-05: the standalone path goes through complexity run_all, so it
    # loads exactly what the suite loads (DATASET_REPS[dataset]['general'] via
    # load_data, float32, TSS groups), applies the decided grid, runs no
    # whole-dataset search, and saves with the same metric stamp and input
    # fingerprint. It used to load the representations named by --reps through
    # find_best_hybrid_tcga, a different list from the suite's (the audit's
    # "two entry points load different representations"); --reps and
    # --genomic-only are no longer read. A hand run always recomputes.
    if getattr(args, 'reps', None) not in (None, 'fast'):
        print(f"  --reps {args.reps} ignored: c12 scores "
              f"DATASET_REPS['{args.dataset}']['general'] only")
    from experiments.complexity.complexity_helpers import run_all
    run_all(args.dataset, args.output, experiments=['c12'], config=config,
            force_rerun=True)
