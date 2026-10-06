#!/usr/bin/env python3
"""C22: Hybrid Explainability.

Paper: Complexity
Conclusion: The best hybrid is interpretable — we can trace every
    feature back to its source representation, provide per-rep
    importance breakdowns, and identify the most influential features.

Dependencies: none stored. Every search is run inside the experiment, one per
    outer fold, on that fold's training rows (handoff R2, R6; 2026-10-05).

Design (aligned 2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4):
    1. Outer folds from kirby.find_best_hybrid.nested_cv_outer_folds, count
       from config['n_splits'] (1 = one held-out split).
    2. Per outer fold, fold_standard_selection: one search on the fold's
       training rows on the dataset's decided grid (pruning off, augmentation
       only as decided_aug_conditions keeps it), and for each of the dataset's
       two decided models the 2 best base representations and the 2 best
       hybrids, ranked on those training rows. A second hybrid whose selected
       features equal the first's is skipped for the next one.
    3. Each base representation is fitted on the training rows and scored on
       the held-out rows.
    4. Each hybrid is rebuilt on the fold's training rows with
       build_fold_hybrid, using the per-representation scores that same
       per-fold search measured (dirty-run bug 1), with that model as the
       keep/drop model and the dataset's own metric. The model is fitted on
       it and scored on the held-out rows.
    5. SHAP on every hybrid of every model on every fold (owner, 2026-10-05,
       tracker C22a: every model explained on every fold, shares per model),
       except svm, dnn and gp: decided 2026-10-06, their SHAP is not run
       (KernelExplainer, about 175 hours per model per dataset) and the
       reason is stored as shap_not_run.
       Each fold's SHAP vector is read against that fold's own feature map,
       aggregated to a share per representation inside the fold, and only the
       per-fold shares are averaged.
    6. explain_hybrid maps each hybrid's top features back to their source
       representation, per fold.

Usage:
    python c22_explainability.py --dataset herg_ki --reps all --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    fold_standard_selection, top_base_reps,
)
from kirby.find_best_hybrid import _predict_for_metric, nested_cv_outer_folds
from kirby.scaling import FeatureScaler
from kirby.importance.hybrid import (
    explain_hybrid, aggregate_importance_by_rep,
)

EXPERIMENT_DIR = 'c22_explainability'
SEED = 42
# The result identity (2026-10-05). Per-fold searches on the decided grid and
# the dataset's two decided models replaced the whole-dataset stored search,
# rf and xgboost, and the 2 general + 1 distance hybrids. A partial file
# written under any other format is moved aside, never read back.
# 2026-10-06: SHAP is no longer computed for svm, dnn and gp (see
# SHAP_NOT_RUN), so the identity changed.
RESULT_FORMAT = 'per_fold_standard_selection_2026_10_06'

# The recipe fields written for every hybrid (2026-10-05, handoff 2.4 step 11).
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size')

# Models shap.TreeExplainer reads directly. Every other model goes through
# shap.KernelExplainer, as before 2026-10-05.
TREE_MODELS = ('rf', 'xgboost', 'lightgbm', 'catboost')

# DECIDED 2026-10-06 (Adelaide delegated): SHAP is not run for svm, dnn and
# gp. They would go through shap.KernelExplainer, measured at 210 s per
# explained molecule for svm at 250 features, about 175 hours per model per
# dataset at 5 folds. Their hybrids are still built, fitted and scored on the
# held-out molecules; only the SHAP shares are left empty, with this reason.
SHAP_NOT_RUN = {
    m: (f'SHAP not run for {m}: it would need shap.KernelExplainer, measured '
        f'at 210 s per explained molecule (svm, 250 features), about 175 hours '
        f'per model per dataset at 5 folds (decided 2026-10-06)')
    for m in ('svm', 'dnn', 'gp')}

ONE_FOLD_NOTE = 'one fold, no spread'


def _reduce_shap_to_per_feature(shap_vals, n_features):
    """Collapse a SHAP output to a 1-D mean-abs-importance per feature.

    OTHER-29: downstream `explain_hybrid` / `aggregate_importance_by_rep`
    index importances as a 1-D array of length n_features. But SHAP's shape
    is task-dependent:
      - regression / single-output: (n_samples, n_features)
      - multiclass (shap>=0.40): a 3-D ndarray (n_samples, n_features, n_classes)
      - older shap multiclass: a Python list of (n_samples, n_features) arrays
    Without normalisation, binary/multiclass classification produces a
    2-D (n_features, n_classes) importance array → explain_hybrid crashes on
    float(importances[i]) (length-2 array) → SHAP silently marked 'fail' and
    the per-rep SHAP breakdown is empty. Reduce all cases to 1-D here.
    Returns None if the result can't be aligned to n_features.
    """
    arr = shap_vals
    if isinstance(arr, list):
        # list of per-class (n_samples, n_features) -> mean abs across classes
        arr = np.mean([np.abs(sv) for sv in arr], axis=0)
    arr = np.abs(np.asarray(arr))
    # Mean over the sample axis (axis 0) first.
    if arr.ndim >= 1:
        arr = arr.mean(axis=0)
    # arr is now either (n_features,) or (n_features, n_classes); collapse any
    # trailing class axis by averaging.
    while arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if arr.shape[0] != n_features:
        return None
    return arr


def _shap_mean_abs(model_name, model, X_train_s, X_val_s):
    """Mean |SHAP| per feature of a fitted model on the held-out rows.

    Returns (mean_abs_shap or None, explainer name, error string or None).
    """
    _lifted = {}
    try:
        import shap
        # ⛔ LIFT transformers OUT OF sys.modules FOR THE SHAP CALL.
        #
        # caco2's shap_share_by_rep came back EMPTY on all five folds while
        # flab_thermo_tresanco and herg_ki filled it, and it is not a caco2
        # defect. caco2's pool holds molformer, grover and smited, whose
        # builders import transformers, so sys.modules['transformers'] is a
        # lazy module by the time this loop runs. shap's safe_isinstance
        # SKIPS a class whose module is absent from sys.modules but calls
        # getattr(module, class_name) when it is present, and shap asks for
        # transformers.TFPreTrainedModel. The lazy module answers that by
        # importing transformers.modeling_tf_utils, which imports
        # TensorFlow, which dies here: this machine's protobuf has dropped
        # FieldDescriptor.is_repeated and the installed TensorFlow still
        # calls it. `import tensorflow` alone raises the same error in a
        # bare process, so TensorFlow is unusable here whatever shap does.
        #
        # Removing the entries makes safe_isinstance take its skip path.
        # They go back in the finally below, so nothing downstream sees a
        # different sys.modules.
        _lifted = {k: v for k, v in list(sys.modules.items())
                   if k == 'transformers' or k.startswith('transformers.')}
        for k in _lifted:
            del sys.modules[k]
        # The same fitted instance the held-out score was computed from.
        if model_name in TREE_MODELS:
            explainer_name = 'TreeExplainer'
            shap_vals = shap.TreeExplainer(model).shap_values(X_val_s)
        else:
            # 100 training rows as background, every held-out row explained.
            # Not reached by svm, dnn or gp (SHAP_NOT_RUN, decided
            # 2026-10-06); every decided model is one of those or a tree
            # model, so this branch runs only for a model added later.
            explainer_name = 'KernelExplainer'
            bg = shap.sample(X_train_s, min(100, len(X_train_s)))
            shap_vals = shap.KernelExplainer(model.predict, bg).shap_values(
                X_val_s)
        n_feat = X_val_s.shape[1]
        mean_abs = _reduce_shap_to_per_feature(shap_vals, n_feat)
        if mean_abs is None:
            return None, explainer_name, (
                f'SHAP shape could not be aligned to {n_feat} features')
        return mean_abs, explainer_name, None
    except Exception as e:
        return None, None, f'{type(e).__name__}: {e}'
    finally:
        sys.modules.update(_lifted)


def _fit_score(X_tr, X_va, y_tr, y_va, factory, metric_fn):
    """Fit one model on a fold's training rows, score it on its held-out rows.

    2026-10-05: kirby.scaling.FeatureScaler fitted on the training rows (0/1
    features left as they are; handoff 2.3 item 17; this file used
    StandardScaler on every feature). Held-out features are NOT capped at the
    training range: c22 never capped, and whether it should is an open
    question for the owner (2026-10-05). No float64 conversion (handoff 2.3
    item 18).

    Returns (score, fitted model, scaled training rows, scaled held-out rows).
    """
    X_tr = np.nan_to_num(np.asarray(X_tr), nan=0.0)
    X_va = np.nan_to_num(np.asarray(X_va), nan=0.0)
    scaler = FeatureScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    m = factory()
    m.fit(X_tr, y_tr)
    # OTHER-29: _predict_for_metric so a binary-AUROC metric is scored on
    # class-1 probabilities, not hard labels.
    score = float(metric_fn(y_va, _predict_for_metric(m, X_va, metric_fn)))
    return score, m, X_tr, X_va


def _over_folds(values):
    """Mean and raw spread over outer folds. No interval, no test."""
    vals = [float(v) for v in values
            if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': None, 'min': None, 'max': None, 'n_folds': 0}
    out = {'mean': float(np.mean(vals)), 'n_folds': len(vals)}
    if len(vals) == 1:
        out.update(min=None, max=None, spread=ONE_FOLD_NOTE)
    else:
        out.update(min=float(np.min(vals)), max=float(np.max(vals)))
    return out


def _run_fold(k, tr, va, *, reps, labels, groups, augmentations,
              dataset_name, n_outer, models, factories, general_names,
              task_type, metric_name, metric_fn, lower_is_better, smiles):
    """One outer fold: standard selection, held-out scores, SHAP per hybrid."""
    y_tr, y_va = labels[tr], labels[va]
    g_tr = np.asarray(groups)[tr] if groups is not None else None
    rec = {'fold': int(k), 'n_train': int(len(tr)), 'n_val': int(len(va)),
           'selection_scope': 'fold_train_rows',
           'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'search_failed': [], 'shortfall': [], 'models': {}}

    sel = fold_standard_selection(
        reps, labels, groups, train_idx=tr, fold_i=k,
        n_outer_folds=n_outer, dataset_name=dataset_name,
        models=list(models), augmentations=augmentations,
        task_type=task_type, metric=metric_name, seed=SEED)

    tr_reps = {r: X[tr] for r, X in reps.items()}
    va_reps = {r: X[va] for r, X in reps.items()}
    tr_augs = ({a: X[tr] for a, X in augmentations.items()}
               if augmentations else None)
    va_augs = ({a: X[va] for a, X in augmentations.items()}
               if augmentations else None)

    for m in models:
        s = sel.get(m)
        if s is None:
            rec['search_failed'].append(m)
            print(f"    fold {k} {m:8s}: the search on this fold's training "
                  f"rows failed; nothing scored for this model")
            continue
        factory = factories[m]
        # Base representations: re-ranked from the same fold search, limited
        # to the dataset's general list (handoff 2.4 step 9, 2026-10-05).
        # fold_standard_selection ranks every representation in reps, which
        # run on its own with --reps all can include ones outside that list.
        bd = top_base_reps(s['search'], m, n=2, only=general_names,
                           require_full=False,
                           context=f'{dataset_name} fold {k} model {m}')
        if len(bd.get('selected') or []) < 2 or len(s['hybrids']) < 2:
            rec['shortfall'].append({
                'model': m, 'n_base': len(bd.get('selected') or []),
                'n_hybrids': len(s['hybrids']),
                'reason': bd.get('shortfall_reason')})

        # ── Base representations: fitted and scored, not explained ──
        base_rows = []
        for i, e in enumerate(bd.get('selected') or [], 1):
            rep = e['rep']
            row = {'slot': f'base{i}', 'rep': rep,
                   'in_general_list': rep in general_names,
                   # The search's mean over its inner folds, inside this outer
                   # fold's training rows.
                   'training_rows_score': e.get('score'),
                   'n_features': int(np.asarray(reps[rep]).shape[1])}
            try:
                sc, _, _, _ = _fit_score(tr_reps[rep], va_reps[rep], y_tr,
                                         y_va, factory, metric_fn)
                row['heldout_score'] = sc
                print(f"    fold {k} {m:8s} base{i}   {rep:28s} "
                      f"{metric_name}={sc:.4f}")
            except Exception as ex:
                row['heldout_score'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"    fold {k} {m:8s} base{i} {rep} failed: "
                      f"{row['error']}")
            base_rows.append(row)

        # ── Hybrids: rebuilt on the fold's training rows, scored, explained ──
        hyb_rows = []
        for i, e in enumerate(s['hybrids'], 1):
            bp = e['blueprint']
            row = {'slot': f'hybrid{i}',
                   **{f: e.get(f) for f in HYBRID_FIELDS},
                   'aug_condition': e.get('aug_condition'),
                   'objective': getattr(bp, 'objective', None),
                   # Mean over the search's inner folds, inside this outer
                   # fold's training rows.
                   'training_rows_score': e.get('score_mean')}
            try:
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp,
                    augmentations_train=tr_augs, augmentations_val=va_augs,
                    groups=g_tr,
                    # This fold's search measured these on this fold's
                    # training rows (dirty-run bug 1, 2026-10-05).
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                sc, fitted, X_tr_s, X_va_s = _fit_score(
                    X_tr_h, X_va_h, y_tr, y_va, factory, metric_fn)
            except Exception as ex:
                row['heldout_score'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
                print(f"    fold {k} {m:8s} hybrid{i} failed: {row['error']}")
                hyb_rows.append(row)
                continue
            row['heldout_score'] = sc
            row['n_features'] = int(X_tr_s.shape[1])
            # The budget split, both meanings. 'allocation' is the quota the
            # allocator asked for under the per-representation methods and
            # the built count under wglobal_*; 'allocation_built' is always
            # the built count.
            row['allocation'] = {r: int(v) for r, v in
                                 (fi.get('allocation') or {}).items()}
            row['allocation_built'] = {
                r: int(v.get('n_features', 0)) for r, v in fi.items()
                if isinstance(v, dict) and 'n_features' in v}
            if fi.get('rep_count_applied') is not None:
                row['rep_count_applied'] = int(fi['rep_count_applied'])
            if fi.get('augmentation_info'):
                row['augmentation_info'] = {
                    a: {'n_features': v.get('n_features', 0)}
                    for a, v in fi['augmentation_info'].items()}
                row['shap_note'] = (
                    'augmentation features are in this hybrid; the '
                    'per-representation SHAP shares leave them out')

            if m in SHAP_NOT_RUN:
                row['shap_explainer'] = None
                row['shap_not_run'] = SHAP_NOT_RUN[m]
                print(f"    fold {k} {m:8s} hybrid{i} {row['n_features']:5d} "
                      f"features {metric_name}={sc:.4f}  SHAP not run "
                      f"(KernelExplainer cost)")
                hyb_rows.append(row)
                continue
            mean_abs, explainer, err = _shap_mean_abs(m, fitted, X_tr_s, X_va_s)
            row['shap_explainer'] = explainer
            if mean_abs is None:
                row['shap_error'] = err
                print(f"    fold {k} {m:8s} hybrid{i} {row['n_features']:5d} "
                      f"features {metric_name}={sc:.4f}  SHAP failed: {err}")
            else:
                # Aggregated to representations INSIDE the fold, against this
                # fold's own feature map: position 37 names a different
                # feature in each fold. Shares sum to 1 within the hybrid.
                row['shap_share_by_rep'] = aggregate_importance_by_rep(
                    fi, mean_abs)
                row['shap_n_val_rows'] = int(len(X_va_s))
                try:
                    ex_ = explain_hybrid(fi, smiles_list=smiles,
                                         importances=mean_abs, top_k=30)
                    row['top_features'] = (ex_.get('top_features') or [])[:30]
                except Exception as ex:
                    row['top_features_error'] = f'{type(ex).__name__}: {ex}'
                _top = sorted(row['shap_share_by_rep'].items(),
                              key=lambda kv: -kv[1])[:3]
                print(f"    fold {k} {m:8s} hybrid{i} {row['n_features']:5d} "
                      f"features ({row['allocation_method']}, "
                      f"{row['importance_method']}, budget {row['budget']}, "
                      f"eval {row['internal_evaluator']}, rep_count "
                      f"{row['rep_count']}) {metric_name}={sc:.4f}  SHAP "
                      f"{', '.join(f'{r} {100 * v:.0f}%' for r, v in _top)}")
            hyb_rows.append(row)

        rec['models'][m] = {
            'base': base_rows, 'hybrids': hyb_rows,
            'hybrids_skipped_identical_features': [
                x.get('label') for x in
                (s.get('hybrids_skipped_identical_features') or [])]}
    return rec


def _summarise(per_fold, models):
    """Per model and slot: held-out score over folds, and each
    representation's SHAP share over folds (per-fold shares, then averaged).

    A slot ('hybrid1') is the fold's rank-1 hybrid for that model, chosen on
    that fold's training rows, so its recipe can differ between folds; the
    recipe per fold is beside each number ('label_by_fold').
    """
    out = {}
    for m in models:
        slots = {}
        for rec in per_fold:
            mr = (rec.get('models') or {}).get(m)
            if not mr:
                continue
            for row in mr['base'] + mr['hybrids']:
                d = slots.setdefault(row['slot'], {
                    'heldout_by_fold': [], 'name_by_fold': [],
                    'shares_by_fold': []})
                d['heldout_by_fold'].append(
                    {'fold': rec['fold'], 'score': row.get('heldout_score')})
                d['name_by_fold'].append(
                    {'fold': rec['fold'],
                     'name': row.get('rep') or row.get('label')})
                if row.get('shap_share_by_rep'):
                    d['shares_by_fold'].append(
                        {'fold': rec['fold'],
                         'share': row['shap_share_by_rep']})
        for slot, d in slots.items():
            d['heldout'] = _over_folds(
                [x['score'] for x in d['heldout_by_fold']])
            if d['shares_by_fold']:
                reps_seen = sorted({r for x in d['shares_by_fold']
                                    for r in x['share']})
                d['shap_share_by_rep'] = {
                    r: _over_folds([x['share'].get(r, 0.0)
                                    for x in d['shares_by_fold']])
                    for r in reps_seen}
                d['shap_folds_scored'] = [x['fold'] for x in d['shares_by_fold']]
        out[m] = slots
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """C22: Hybrid Explainability, per outer fold."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C22: HYBRID EXPLAINABILITY (per fold)")
    print("=" * 70)

    from experiments.decided_grid import decided_models
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data['task_type']
    augmentations = data.get('augmentations')
    smiles = data.get('smiles')

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    # Models: the dataset's two decided models (DECISIONS.md Step 1). knn is
    # never a general model here. Was rf and xgboost on every dataset.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_without_factory = [m for m in models if m not in factories]
    models = [m for m in models if m in factories]
    if not models:
        return {'error': 'no_decided_model_has_a_factory',
                'models_without_factory': models_without_factory}

    # Representations: the per-representation loop (the base representations
    # scored alone) reads the dataset's general list only (handoff 2.4 step
    # 9). data['reps'] goes to the search and the hybrid builder unfiltered.
    general_names = [r for r in ((DATASET_REPS.get(dataset_name) or {})
                                 .get('general') or sorted(reps))
                     if r in reps]
    extra = sorted(set(reps) - set(general_names))
    if extra:
        print(f"  representations outside DATASET_REPS['{dataset_name}']"
              f"['general']: {extra}")

    n_requested = int(config['n_splits'])
    folds = nested_cv_outer_folds(reps, labels, groups, n_requested, SEED,
                                  task_type)
    n_outer = len(folds)
    print(f"  {len(reps)} representations, models {models}, {n_outer} outer "
          f"fold(s) (asked {n_requested})")

    # One partial file per dataset; each finished fold is stored at once and
    # read back on the next attempt (dirty-run bug 12).
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, config,
                     result_format=RESULT_FORMAT))

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        if ck.done(fold=k):
            rec = ck.get(fold=k)
            print(f"  fold {k}: finished in an earlier attempt, read back")
        else:
            print(f"\n  --- outer fold {k + 1}/{n_outer}: {len(tr)} training "
                  f"rows, {len(va)} held out ---")
            rec = convert_numpy(_run_fold(
                k, tr, va, reps=reps, labels=labels, groups=groups,
                augmentations=augmentations, dataset_name=dataset_name,
                n_outer=n_outer, models=models, factories=factories,
                general_names=general_names, task_type=task_type,
                metric_name=metric_name, metric_fn=metric_fn,
                lower_is_better=lower_is_better, smiles=smiles))
            # A fold whose search failed for a model is in the result file but
            # not stored, so the next attempt searches it again.
            if not rec.get('search_failed'):
                ck.put(rec, fold=k)
        per_fold.append(rec)

    summary = _summarise(per_fold, models)

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print("EXPLAINABILITY SUMMARY (held-out rows; SHAP shares per fold, "
          "then averaged)")
    print(f"{'=' * 70}")
    for m in models:
        for slot in sorted(summary.get(m, {})):
            d = summary[m][slot]
            h = d['heldout']
            if h['mean'] is None:
                print(f"  {m:8s} {slot:8s}: no fold scored")
                continue
            spread = (h.get('spread') if h['n_folds'] == 1
                      else f"min {h['min']:.4f}, max {h['max']:.4f}")
            print(f"  {m:8s} {slot:8s}: {metric_name} {h['mean']:.4f} over "
                  f"{h['n_folds']} fold(s) ({spread})")
            for r, v in sorted((d.get('shap_share_by_rep') or {}).items(),
                               key=lambda kv: -(kv[1]['mean'] or 0))[:5]:
                print(f"      {r:30s} SHAP share {100 * v['mean']:.1f}%")

    return convert_numpy({
        'result_format': RESULT_FORMAT,
        'selection_scope': 'fold_train_rows',
        'selection_rule': ('per outer fold, per decided model: 2 best base '
                           'representations and 2 best hybrids, ranked on '
                           "that fold's training rows"),
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'models': models,
        'models_without_factory': models_without_factory,
        'representations': sorted(reps),
        'representations_general': general_names,
        'representations_outside_general': extra,
        'n_folds_requested': n_requested,
        'n_outer_folds': n_outer,
        'n_samples': int(len(labels)),
        'seed': SEED,
        # Per fold, per model: base and hybrid rows with recipe, training-row
        # score, held-out score, budget split and SHAP share by representation.
        'per_fold': per_fold,
        # Per model, per slot: held-out score and SHAP share over folds.
        # Every number here is one model's; nothing mixes two models.
        'summary_by_model': summary,
        'shap_share_basis': ("each fold's SHAP vector read against that "
                             "fold's own feature map, aggregated to a share "
                             'per representation inside the fold, then '
                             'averaged over folds'),
        'shap_explainer_by_model': {
            m: (None if m in SHAP_NOT_RUN
                else 'TreeExplainer' if m in TREE_MODELS
                else 'KernelExplainer')
            for m in models},
        # Models whose hybrids are scored but not explained, with the reason.
        'shap_not_run': {m: SHAP_NOT_RUN[m] for m in models
                         if m in SHAP_NOT_RUN},
        'base_representations_explained': False,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C22: Hybrid Explainability',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c22_explainability',
                            args.dataset)
