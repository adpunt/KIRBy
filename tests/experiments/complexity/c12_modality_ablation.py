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
  - Metric: AUROC for binary classification, R² for regression,
    F1-weighted for multi-class classification.
  - Groups from TSS (tissue source site) for site-corrected CV.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score, f1_score

from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results,
    get_model_factories, cv_split,
    fold_reselection,
)


def _reselected_pair_delta(pair_model_scores, baseline_rep_names, all_rep_scores,
                       lower_is_better=False):
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

    hr = fold_reselection(candidates_by_model, baselines_by_model,
                        lower_is_better)
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
                            lower_is_better=False):
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
                               model_factory, task_type, n_classes,
                               model_name=None):
    """Fit one model on one fold's training rows and score the held-out rows.

    The matrix is the named representations placed side by side at full width —
    the same np.hstack eval_concat_reps uses, scaled the same way, on one fold.

    A fit or a prediction that raises returns NaN for this fold and names the
    representations and the model that raised. It used to propagate, so one
    representation the model could not fit ended the whole run.
    """
    X = np.hstack([reps[r] for r in rep_names])
    train_idx, val_idx = fold_indices[fold_i]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = labels[train_idx], labels[val_idx]

    try:
        model = model_factory()
        model.fit(X_train, y_train)
        predict_fn = get_predict_fn(model, task_type, n_classes)
        return float(score_predictions(y_val, predict_fn(X_val), task_type,
                                       n_classes))
    except Exception as e:
        print(f"    FAILED fold {fold_i}: {'+'.join(rep_names)} with "
              f"{model_name or 'model'} raised {type(e).__name__}: {e}")
        return float('nan')


def _lofo_combination_fold_scores(modalities, available_by_modality,
                                  all_rep_scores, reps, labels, fold_indices,
                                  model_factories, task_type, n_classes,
                                  fit_cache, fixed_reps=None,
                                  fixed_fold_scores=None,
                                  lower_is_better=False):
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
                    task_type, n_classes, model_name=model_name)
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

def score_predictions(y_true, y_pred, task_type: str, n_classes=None) -> float:
    """Compute the primary metric for the given task type.

    `n_classes` is the number of classes in the whole dataset. It decides
    AUROC against weighted F1. It used to be counted on the held-out fold, so
    a four-class fold that held only two classes was scored as AUROC on one
    probability column and averaged with F1 folds (fixed 2026-09-28).
    """
    if task_type == 'regression':
        return float(r2_score(y_true, y_pred))
    elif task_type == 'classification':
        # y_pred may be probabilities or class labels
        if n_classes is None:
            n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # For binary: use probabilities if available (shape (n, 2))
            if y_pred.ndim == 2:
                return float(roc_auc_score(y_true, y_pred[:, 1]))
            else:
                return float(roc_auc_score(y_true, y_pred))
        else:
            if y_pred.ndim == 2:
                y_class = np.argmax(y_pred, axis=1)
            else:
                y_class = y_pred
            return float(f1_score(y_true, y_class, average='weighted', zero_division=0))
    return float('nan')


def metric_name(task_type: str, n_classes: int) -> str:
    if task_type == 'regression':
        return 'r2'
    elif n_classes == 2:
        return 'auroc'
    else:
        return 'f1_weighted'


# `higher_is_better(metric)` used to live here and returned True for ANY metric
# name handed to it, ignoring the argument. It had no caller. Deleted 2026-09-10
# (metric-direction audit) so nothing can pick it up and get a wrong answer for
# rmse or mae. c12's own scores come from score_predictions above, which is r2 /
# auroc / f1_weighted only — all higher-is-better — and that is why the
# lower_is_better=False constants further down are correct.


# =============================================================================
# MODEL FACTORIES (classification-aware)
# =============================================================================

def get_clf_model_factories(model_names, n_samples=None):
    """Return classification model factories.

    Gated by n_samples the same way the regression path is: a requested model
    that also appears in find_best_hybrid.DEFAULT_MODELS with a `_max_samples`
    cap is skipped when n_samples exceeds that cap. Previously this path was
    ungated while the reg path (get_model_factories) auto-skipped large-n models
    (audit: c12 clf path UNGATED). Models absent from DEFAULT_MODELS keep their
    prior always-on behaviour.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    # Thread counts come from KIRBY_RF_JOBS and KIRBY_LGBM_THREADS, the
    # repository's switches for a shared machine; both default to -1, which is
    # what was hard-coded here, so a cluster run is unchanged. n_jobs changes how
    # the same forest is built, not the forest, so no score moves (2026-09-28).
    try:
        _rf_jobs = int(os.environ.get('KIRBY_RF_JOBS', '-1') or -1)
    except ValueError:
        _rf_jobs = -1
    try:
        from kirby.hybrid import lgbm_n_jobs as _lgbm_n_jobs
        _lgbm_jobs = _lgbm_n_jobs()
    except Exception:
        _lgbm_jobs = -1

    catalog = {
        'rf':    lambda: RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                                random_state=42, n_jobs=_rf_jobs),
        'knn':   lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'mlp':   lambda: MLPClassifier(hidden_layer_sizes=(100, 50),
                                       random_state=42, max_iter=500),
    }
    try:
        from xgboost import XGBClassifier
        # `use_label_encoder` was removed in xgboost >= 1.6; passing it raises
        # TypeError which the ImportError except below would not catch.
        catalog['xgboost'] = lambda: XGBClassifier(n_estimators=100, colsample_bytree=0.5,
                                                    random_state=42, n_jobs=-1, verbosity=0,
                                                    eval_metric='logloss')
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        catalog['lightgbm'] = lambda: LGBMClassifier(n_estimators=100, random_state=42,
                                                      n_jobs=_lgbm_jobs, verbose=-1)
    except ImportError:
        pass

    # Same _max_samples gate the reg path applies via get_model_factories.
    try:
        from kirby.find_best_hybrid import (DEFAULT_MODELS as _DEFAULT_MODELS,
                                            get_model as _get_model)
    except Exception:
        _DEFAULT_MODELS, _get_model = {}, None

    def _kirby_factory(_name):
        return lambda: _get_model(_name, 'classification')

    factories = {}
    for name in model_names:
        factory = catalog.get(name)
        if factory is None:
            # The default downstream set is rf, xgboost, svm, dnn, lightgbm.
            # svm and dnn have no entry in the catalog above, so both were
            # warned and skipped on every run — two of the five models asked
            # for. A name the catalog does not define is built by
            # find_best_hybrid.get_model instead, the same classifier the rest
            # of the suite fits.
            if name == 'ridge':
                # Deleted from this file deliberately; the fallback must not
                # put it back.
                print("  Skipping clf ridge: not a downstream evaluator here")
                continue
            if _get_model is None or name not in _DEFAULT_MODELS:
                print(f"  WARNING: clf model '{name}' not available, skipping")
                continue
            factory = _kirby_factory(name)
        spec = _DEFAULT_MODELS.get(name, {})
        max_s = spec.get('_max_samples')
        if max_s is not None and n_samples is not None and n_samples > max_s:
            print(f"  Skipping clf {name}: n_samples={n_samples} > _max_samples={max_s}")
            continue
        factories[name] = factory
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
            # Weighted F1 needs class labels. The argmax is a COLUMN index;
            # model.classes_ turns it into the label. The bare argmax equalled
            # the label only when every class was in the training fold
            # (fixed 2026-09-28).
            return lambda X: np.asarray(model.classes_)[
                np.argmax(model.predict_proba(X), axis=1)]
    # Fallback: use decision_function or predict
    if hasattr(model, 'decision_function'):
        return model.decision_function
    return model.predict


# =============================================================================
# SINGLE-REP EVALUATION
# =============================================================================

def eval_single_rep(X, labels, groups, model_factories, n_folds, task_type,
                    n_classes, mode, rep_label=None):
    """Evaluate a single representation across folds and models.

    Returns dict: {model_name: {'score_mean': float, 'score_std': float}}

    A fold whose fit or prediction raises scores NaN and prints the
    representation and the model that raised; it no longer ends the run.
    `rep_label` is only used in that message.
    """
    sample_X = X
    results = {}

    for model_name, model_factory in model_factories.items():
        fold_scores = []
        for f, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, sample_X, labels, mode=mode)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train, y_val = labels[train_idx], labels[val_idx]

            try:
                model = model_factory()
                model.fit(X_train, y_train)
                predict_fn = get_predict_fn(model, task_type, n_classes)
                y_pred = predict_fn(X_val)
                score = score_predictions(y_val, y_pred, task_type, n_classes)
            except Exception as e:
                print(f"    FAILED fold {f}: {rep_label or 'rep'} with "
                      f"{model_name} raised {type(e).__name__}: {e}")
                score = float('nan')
            fold_scores.append(score)

        # Persist the raw per-fold scores (not just mean/std) so the honest
        # leave-one-fold-out re-score can re-pick the winner per held-out fold.
        # NaN folds keep their position: the re-score pairs arms by fold index.
        if any(np.isfinite(s) for s in fold_scores):
            results[model_name] = {
                'score_mean': float(np.nanmean(fold_scores)),
                'score_std': float(np.nanstd(fold_scores)),
                'dim': X.shape[1],
                'fold_scores': [float(s) for s in fold_scores],
            }
        else:
            results[model_name] = {
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'dim': X.shape[1],
                # Every fold raised. The NaNs are kept in fold order so the
                # per-fold reselection still lines this model up with the others.
                'fold_scores': [float(s) for s in fold_scores],
            }

    return results


def eval_concat_reps(rep_arrays, labels, groups, model_factories, n_folds,
                     task_type, n_classes, mode, rep_label=None):
    """Evaluate concatenation of multiple representation arrays.

    Returns dict: {model_name: {'score_mean': float, 'score_std': float, 'dim': int}}

    Same crash rule as eval_single_rep: a fold that raises scores NaN, names
    the representations and the model, and the run continues.
    """
    X = np.hstack(rep_arrays)
    dim = X.shape[1]
    results = {}

    for model_name, model_factory in model_factories.items():
        fold_scores = []
        for f, (train_idx, val_idx) in enumerate(
                cv_split(groups, n_folds, X, labels, mode=mode)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train, y_val = labels[train_idx], labels[val_idx]

            try:
                model = model_factory()
                model.fit(X_train, y_train)
                predict_fn = get_predict_fn(model, task_type, n_classes)
                y_pred = predict_fn(X_val)
                score = score_predictions(y_val, y_pred, task_type, n_classes)
            except Exception as e:
                print(f"    FAILED fold {f}: {rep_label or 'combination'} with "
                      f"{model_name} raised {type(e).__name__}: {e}")
                score = float('nan')
            fold_scores.append(score)

        # Persist the raw per-fold scores for honest re-scoring (see eval_single_rep).
        if any(np.isfinite(s) for s in fold_scores):
            results[model_name] = {
                'score_mean': float(np.nanmean(fold_scores)),
                'score_std': float(np.nanstd(fold_scores)),
                'dim': dim,
                'fold_scores': [float(s) for s in fold_scores],
            }
        else:
            results[model_name] = {
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'dim': dim,
                # Every fold raised; NaNs kept in fold order (see above).
                'fold_scores': [float(s) for s in fold_scores],
            }

    return results


def best_score_across_models(model_results: dict) -> float:
    """Return the best score_mean across all model results."""
    scores = [v['score_mean'] for v in model_results.values()
              if not np.isnan(v.get('score_mean', float('nan')))]
    return max(scores) if scores else float('nan')


def best_model_for_result(model_results: dict) -> str:
    """Return name of model with best score_mean."""
    best_name, best_score = None, -np.inf
    for name, vals in model_results.items():
        s = vals.get('score_mean', float('nan'))
        if not np.isnan(s) and s > best_score:
            best_score = s
            best_name = name
    return best_name


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
    metric = metric_name(task_type, n_classes)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    print(f"  Dataset:   {dataset_name}")
    print(f"  Task:      {task_type} ({n_classes} classes)")
    print(f"  Metric:    {metric}")
    print(f"  n_samples: {len(labels)}, n_folds: {n_folds}")
    print(f"  n_reps available: {len(reps)}")

    # Encode classification labels to int if needed
    if task_type == 'classification':
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    # Get model factories
    model_names = config['models']
    if task_type == 'classification':
        model_factories = get_clf_model_factories(model_names, n_samples=len(labels))
    else:
        model_factories = get_model_factories(model_names,
                                                task_type=task_type,
                                                n_samples=len(labels))

    if not model_factories:
        raise RuntimeError("No valid model factories available.")

    # 2026-09-12 — the per-fold re-pick on the combination side needs the fold
    # indices themselves, not just per-fold scores. cv_split is deterministic
    # given groups, labels, n_folds and the seed, and it reads only the number
    # of rows from the matrix it is handed, so this is the same fold sequence
    # every eval_single_rep / eval_concat_reps call below walks.
    fold_indices = [(np.asarray(tr), np.asarray(va)) for tr, va in
                    cv_split(groups, n_folds, np.zeros((len(labels), 1)),
                             labels, mode=mode)]

    # Scores already fitted, keyed by (representations, model) -> {fold: score}.
    # Shared by every combination so a re-pick that lands on representations
    # already fitted on that fold costs nothing.
    combination_fit_cache = {}

    # =========================================================================
    # 1. Identify which reps are available per modality group
    # =========================================================================
    available_by_modality = {}
    for mod_name, mod_reps in MODALITY_GROUPS.items():
        avail = [r for r in mod_reps if r in reps]
        if avail:
            available_by_modality[mod_name] = avail
        else:
            print(f"  NOTE: modality '{mod_name}' — no reps available in loaded data")

    # Also check for any expression reps (factory or otherwise) in loaded data
    expr_avail = [r for r in FACTORY_EXPRESSION_REPS if r in reps]
    # Plus non-factory expression reps already in MODALITY_GROUPS['expression']
    expr_avail += [r for r in MODALITY_GROUPS.get('expression', []) if r in reps
                   and r not in expr_avail]
    if expr_avail:
        available_by_modality['expression'] = expr_avail

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
        result = eval_single_rep(X, labels, groups, model_factories, n_folds,
                                 task_type, n_classes, mode, rep_label=rep_name)
        all_rep_scores[rep_name] = result
        best_s = best_score_across_models(result)
        best_m = best_model_for_result(result)
        print(f"    {rep_name:<30} {metric}={best_s:+.4f}  dim={X.shape[1]}  "
              f"best_model={best_m}")

    # =========================================================================
    # 3. Find best rep per modality (for use in pairwise combinations)
    # =========================================================================
    best_rep_per_modality = {}   # modality -> rep_name
    best_score_per_modality = {}  # modality -> float

    for mod_name, mod_reps in available_by_modality.items():
        best_r, best_s = None, -np.inf
        for rep_name in mod_reps:
            if rep_name not in all_rep_scores:
                continue
            s = best_score_across_models(all_rep_scores[rep_name])
            if not np.isnan(s) and s > best_s:
                best_s = s
                best_r = rep_name
        best_rep_per_modality[mod_name] = best_r
        best_score_per_modality[mod_name] = best_s

    print(f"\n  Best rep per modality:")
    for mod in active_modalities:
        # None when every fit of every representation in the modality failed;
        # formatting None with a width raised TypeError and ended the run.
        rep = best_rep_per_modality.get(mod) or 'none'
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
        ref_modality = max(active_modalities,
                           key=lambda m: best_score_per_modality.get(m, -np.inf))

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
        result = eval_concat_reps([X_a, X_b], labels, groups, model_factories,
                                  n_folds, task_type, n_classes, mode,
                                  rep_label=f'{rep_a}+{rep_b}')

        best_s = best_score_across_models(result)
        best_m = best_model_for_result(result)
        total_dim = X_a.shape[1] + X_b.shape[1]

        pairwise_results[pair_key] = {
            'modalities': [mod_a, mod_b],
            'reps_used': [rep_a, rep_b],
            'score_mean': best_s,
            'score_std': result.get(best_m, {}).get('score_std', float('nan'))
                         if best_m else float('nan'),
            'total_dim': total_dim,
            'best_model': best_m,
            'model_scores': {k: {'score_mean': v['score_mean'],
                                  'score_std': v['score_std'],
                                  'dim': v['dim'],
                                  'fold_scores': v.get('fold_scores', [])}
                             for k, v in result.items()},
            'delta_vs_a': best_s - best_score_per_modality.get(mod_a, float('nan')),
            'delta_vs_b': best_s - best_score_per_modality.get(mod_b, float('nan')),
            'delta_vs_best_single': (
                best_s - max(best_score_per_modality.get(mod_a, -np.inf),
                             best_score_per_modality.get(mod_b, -np.inf))
            ),
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
            fold_indices, model_factories, task_type, n_classes,
            combination_fit_cache,
            fixed_reps=[rep_a, rep_b],
            fixed_fold_scores={m: ms.get('fold_scores', [])
                               for m, ms in result.items()},
            lower_is_better=False)
        _lofo_model_scores = {m: {'fold_scores': fs}
                              for m, fs in _lofo['fold_scores_by_model'].items()}

        _hp = _reselected_pair_delta(
            _lofo_model_scores,
            list(available_by_modality.get(mod_a, []))
            + list(available_by_modality.get(mod_b, [])),
            all_rep_scores, lower_is_better=False)
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
        all_result = eval_concat_reps(all_arrays, labels, groups, model_factories,
                                      n_folds, task_type, n_classes, mode,
                                      rep_label='+'.join(all_reps_used))
        total_dim = sum(a.shape[1] for a in all_arrays)
        best_all_s = best_score_across_models(all_result)
        best_all_m = best_model_for_result(all_result)
        best_single = max(best_score_per_modality.get(m, -np.inf)
                          for m in modalities_with_reps)
        delta_all = best_all_s - best_single

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
            fold_indices, model_factories, task_type, n_classes,
            combination_fit_cache,
            fixed_reps=all_reps_used,
            fixed_fold_scores={m: ms.get('fold_scores', [])
                               for m, ms in all_result.items()},
            lower_is_better=False)
        _la_model_scores = {m: {'fold_scores': fs}
                            for m, fs in _la['fold_scores_by_model'].items()}

        _ha = _reselected_pair_delta(
            _la_model_scores,
            [r for m in modalities_with_reps
             for r in available_by_modality.get(m, [])],
            all_rep_scores, lower_is_better=False)
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
                       key=lambda m: -best_score_per_modality.get(m, -np.inf)):
        rep = best_rep_per_modality.get(mod) or 'none'
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
            delta = pr['score_mean'] - expr_score
            # Against expression alone, re-picking which expression
            # representation on each held-out fold's other folds.
            # 2026-09-12 — the pair side is re-picked on the other folds too,
            # so this reads the per-fold pair scores, not the all-folds pair.
            _he = _reselected_pair_delta(
                {m: {'fold_scores': fs} for m, fs in
                 pr.get('lofo_pair_fold_scores', {}).items()},
                available_by_modality.get('expression', []),
                all_rep_scores, lower_is_better=False)
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
    # All metrics in this experiment are higher-is-better, so lower_is_better=False.
    lower_is_better = False  # r2 / auroc / f1_weighted all higher = better

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
                                   lower_is_better)

    print("\n" + "-" * 60)
    print("  HONEST PER-MODEL RE-SCORE (leave-one-fold-out, per model)")
    print("-" * 60)
    if reselection_result:
        print(f"    {'Model':<12} {'hybrid':>10} {'best_single':>12} {'delta':>10}  note")
        for m in sorted(reselection_result.keys()):
            r = reselection_result[m]
            note = 'SINGLE-FOLD (optimistic)' if r.get('single_fold') else ''
            print(f"    {m:<12} {r['hybrid_reselected']:>10.4f} "
                  f"{r['best_single_reselected']:>12.4f} {r['delta']:>+10.4f}  {note}")
    else:
        print("    (no models with usable per-fold data)")

    # =========================================================================
    # Build final results dict
    # =========================================================================
    output = {
        'dataset': dataset_name,
        'task_type': task_type,
        'n_classes': n_classes,
        'metric': metric,
        'n_folds': n_folds,
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
                        sk: (sv if isinstance(sv, (str, list))
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
                        help='Override number of CV folds from mode config')
    # Allow explicit rep list override (for loading)
    parser.add_argument('--genomic-only', action='store_true',
                        help='Skip WSI reps when loading data')
    args = parser.parse_args()

    if not args.dataset.startswith('tcga_'):
        parser.error(f"C12 requires a tcga_* dataset. Got: '{args.dataset}'")

    config = get_config(args)
    if args.n_splits is not None:
        config = dict(config)
        config['n_splits'] = args.n_splits

    print(f"Mode:    {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps:    {args.reps}")
    print(f"Models:  {config['models']}")
    print(f"Folds:   {config['n_splits']}")

    # Load TCGA data directly (not via generic load_data to keep reps flexible)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

    from find_best_hybrid_tcga import (
        load_dataset as load_tcga_dataset,
        get_representations as get_tcga_representations,
        assign_tss_groups,
    )

    print(f"\nLoading dataset: {args.dataset}")
    expression, labels, task_type, metadata = load_tcga_dataset(args.dataset)
    sample_ids = metadata['sample_ids']
    gene_names = metadata['gene_names']
    wsi_dir = metadata.get('wsi_dir')
    cohort = metadata.get('cohort', '')

    print(f"  {len(sample_ids)} samples, task: {task_type}")

    groups, n_groups = assign_tss_groups(sample_ids)
    print(f"  {n_groups} TSS sites")

    # Use the preset from args; for ablation, 'all' or 'genomic_all' gives most coverage.
    # Factory representations (gene_pca100 and the rest of FACTORY_REPS) do not
    # arrive here at all: get_representations drops every name in FACTORY_REPS
    # before it loads anything (tests/find_best_hybrid_tcga.py:376), whatever
    # the preset asks for and whatever is already cached. c12 therefore scores
    # the non-factory representations only.
    print(f"\nLoading representations (preset={args.reps})...")

    reps = get_tcga_representations(
        expression, gene_names, sample_ids, wsi_dir,
        args.reps,
        cohort=cohort,
        cache=True,
        genomic_only=args.genomic_only,
    )

    print(f"\n  Loaded {len(reps)} representations")

    data = {
        'expression': expression,
        'labels': labels,
        'reps': reps,
        'groups': groups,
        'n_groups': n_groups,
        'task_type': task_type,
        'domain': 'tcga',
        'metadata': metadata,
    }

    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c12_modality_ablation', args.dataset)
