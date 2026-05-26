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
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split,
)


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

def score_predictions(y_true, y_pred, task_type: str) -> float:
    """Compute the primary metric for the given task type."""
    if task_type == 'regression':
        return float(r2_score(y_true, y_pred))
    elif task_type == 'classification':
        # y_pred may be probabilities or class labels
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


def higher_is_better(metric: str) -> bool:
    return True  # all our metrics: higher = better


# =============================================================================
# MODEL FACTORIES (classification-aware)
# =============================================================================

def get_clf_model_factories(model_names):
    """Return classification model factories."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    catalog = {
        'rf':    lambda: RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                                random_state=42, n_jobs=-1),
        'knn':   lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'ridge': lambda: RidgeClassifierCV(cv=5),
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
                                                      n_jobs=-1, verbose=-1)
    except ImportError:
        pass

    factories = {}
    for name in model_names:
        if name in catalog:
            factories[name] = catalog[name]
        else:
            print(f"  WARNING: clf model '{name}' not available, skipping")
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

def eval_single_rep(X, labels, groups, model_factories, n_folds, task_type,
                    n_classes, mode):
    """Evaluate a single representation across folds and models.

    Returns dict: {model_name: {'score_mean': float, 'score_std': float}}
    """
    sample_X = X
    results = {}

    for model_name, model_factory in model_factories.items():
        fold_scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, sample_X, labels, mode=mode):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            predict_fn = get_predict_fn(model, task_type, n_classes)
            y_pred = predict_fn(X_val)
            score = score_predictions(y_val, y_pred, task_type)
            fold_scores.append(score)

        if fold_scores:
            results[model_name] = {
                'score_mean': float(np.nanmean(fold_scores)),
                'score_std': float(np.nanstd(fold_scores)),
                'dim': X.shape[1],
            }
        else:
            results[model_name] = {
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'dim': X.shape[1],
            }

    return results


def eval_concat_reps(rep_arrays, labels, groups, model_factories, n_folds,
                     task_type, n_classes, mode):
    """Evaluate concatenation of multiple representation arrays.

    Returns dict: {model_name: {'score_mean': float, 'score_std': float, 'dim': int}}
    """
    X = np.hstack(rep_arrays)
    dim = X.shape[1]
    results = {}

    for model_name, model_factory in model_factories.items():
        fold_scores = []
        for train_idx, val_idx in cv_split(groups, n_folds, X, labels, mode=mode):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train, y_val = labels[train_idx], labels[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            predict_fn = get_predict_fn(model, task_type, n_classes)
            y_pred = predict_fn(X_val)
            score = score_predictions(y_val, y_pred, task_type)
            fold_scores.append(score)

        if fold_scores:
            results[model_name] = {
                'score_mean': float(np.nanmean(fold_scores)),
                'score_std': float(np.nanstd(fold_scores)),
                'dim': dim,
            }
        else:
            results[model_name] = {
                'score_mean': float('nan'),
                'score_std': float('nan'),
                'dim': dim,
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
        model_factories = get_clf_model_factories(model_names)
    else:
        model_factories = get_model_factories(model_names,
                                                task_type=task_type,
                                                n_samples=len(labels))

    if not model_factories:
        raise RuntimeError("No valid model factories available.")

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
                                 task_type, n_classes, mode)
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
                                  n_folds, task_type, n_classes, mode)

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
                                  'dim': v['dim']}
                             for k, v in result.items()},
            'delta_vs_a': best_s - best_score_per_modality.get(mod_a, float('nan')),
            'delta_vs_b': best_s - best_score_per_modality.get(mod_b, float('nan')),
            'delta_vs_best_single': (
                best_s - max(best_score_per_modality.get(mod_a, -np.inf),
                             best_score_per_modality.get(mod_b, -np.inf))
            ),
        }

        delta = pairwise_results[pair_key]['delta_vs_best_single']
        sign = '+' if delta >= 0 else ''
        print(f"    {pair_key:<35} {metric}={best_s:+.4f}  "
              f"Δbest={sign}{delta:.4f}  dim={total_dim}  model={best_m}")

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
                                      n_folds, task_type, n_classes, mode)
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
                                  'dim': v['dim']}
                             for k, v in all_result.items()},
            'delta_vs_best_single': delta_all,
        }

        sign = '+' if delta_all >= 0 else ''
        print(f"  All combined ({len(all_mods_used)} modalities): "
              f"{metric}={best_all_s:+.4f}  Δ={sign}{delta_all:.4f}  "
              f"dim={total_dim}  model={best_all_m}")
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
        rep = best_rep_per_modality.get(mod, 'N/A')
        score = best_score_per_modality.get(mod, float('nan'))
        dim = reps[rep].shape[1] if rep and rep in reps else '?'
        print(f"    {mod:<15} {rep:<30} {score:>8.4f}  {dim:>6}")

    print(f"\n  Pairwise combinations (best score, delta vs best single):")
    print(f"    {'Pair':<35} {metric:>8}  {'Delta':>8}  {'Dim':>6}")
    print("    " + "-" * 64)
    for pair_key, pr in sorted(pairwise_results.items(),
                                key=lambda x: -x[1]['score_mean']):
        score = pr['score_mean']
        delta = pr['delta_vs_best_single']
        dim = pr['total_dim']
        flag = ' <-- HELPS' if delta > 0.01 else (' <-- HURTS' if delta < -0.01 else '')
        print(f"    {pair_key:<35} {score:>8.4f}  {delta:>+8.4f}  {dim:>6}{flag}")

    if all_combined_result:
        print(f"\n  All modalities combined:")
        print(f"    {metric}={all_combined_result['score_mean']:+.4f}  "
              f"Δ={all_combined_result['delta_vs_best_single']:+.4f}  "
              f"dim={all_combined_result['total_dim']}")

    # Rank modalities by contribution when paired with expression
    if 'expression' in modalities_with_reps:
        print(f"\n  Effect of adding each modality to expression:")
        print(f"    {'Added modality':<20} {'Δ'+metric:>10}  {'Combined '+metric:>16}  "
              f"{'expression only':>15}")
        print("    " + "-" * 70)
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
            partner_deltas.append((mod, delta, pr['score_mean']))

        for mod, delta, combined_score in sorted(partner_deltas, key=lambda x: -x[1]):
            flag = ' <-- HELPS' if delta > 0.01 else (' <-- HURTS' if delta < -0.01 else '')
            print(f"    {mod:<20} {delta:>+10.4f}  {combined_score:>16.4f}  "
                  f"{expr_score:>15.4f}{flag}")

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
                    model: {sk: float(sv) if not isinstance(sv, str) else sv
                            for sk, sv in scores.items()}
                    for model, scores in model_scores.items()
                }
                for rep, model_scores in all_rep_scores.items()
            },
        },
        'pairwise': pairwise_results,
        'all_combined': all_combined_result,
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

    # Use the preset from args; for ablation, 'all' or 'genomic_all' gives most coverage
    # Note: factory reps (PCA, NMF, etc.) are included in get_representations()
    # because find_best_hybrid_tcga handles them — but here we pass the preset
    # directly and only use the reps that end up in base_reps (non-factory).
    # Factory reps (gene_pca100 etc.) are also pre-computed in base_reps when
    # the preset is expanded — so they will be available if requested.
    print(f"\nLoading representations (preset={args.reps})...")

    # get_representations() excludes factory reps by default (they go to get_rep_factories)
    # To include them pre-computed for ablation, we call get_tcga_representations with cache.
    # The factory reps that were previously computed and cached will be loaded automatically.
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
