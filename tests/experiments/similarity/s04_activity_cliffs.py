#!/usr/bin/env python3
"""S04: Activity Cliffs -- prediction error on cliff molecules.

Paper: Similarity
Question: is a molecule that sits on an activity cliff predicted better from
its nearest neighbours in a hybrid's features than in a single
representation's, and better by the dataset's two models on a hybrid than on
a single representation?

Which molecules are cliffs comes from similarity_helpers.calibrated_cliff_rule,
which s12 also calls, so the two files no longer partition a dataset two
different ways. ECFP4 Tanimoto at or above 0.55 on the structural side, and an
activity gap solved per dataset so a stated share of the molecules are cliffs.

2026-10-05 (handoff Part 2, chat 4): every selection is per outer fold, from
searches on that fold's TRAINING rows (similarity_helpers.
fold_similarity_selection, neighbour spaces on). Per outer fold:

  kNN arms (each molecule's label predicted from its nearest training
  molecules), scored at every decided kNN setting (knn_axes: the dataset
  group's distances, 5/10/20 neighbours, rank and inverse_square counting,
  scaffold group not excluded), unweighted and importance-weighted:
    knn_base1, knn_base2     knn's 2 best single representations in the
                             fold's distance search. These replace the loop
                             over every representation (Part 1 step 5).
    general|<model>|hybrid1, general|<model>|hybrid2
                             each decided model's 2 general hybrids, used as
                             feature spaces for the neighbours
    distance|knn|hybrid1     knn's top hybrid of the fold's distance search
    similarity|hybrid1       the top hybrid of the fold's similarity-objective
                             search
  Decided-model arms: each of the dataset's two models fitted on its 2 base
  representations and its 2 general hybrids (8 fits for two models). These
  replace the rf baseline, rf being decided on none of s04's datasets.

Before this date the arms came from get_rule_hybrids, a ranking over the
other folds of searches that had run over every row (so the fold's rows were
inside every search), at a fixed 100-feature budget; the per-representation
arms covered every representation; the kNN arms used sklearn's default
euclidean distance at 5 neighbours with equal votes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.metrics import mean_squared_error

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models
from experiments.similarity.similarity_helpers import (
    calibrated_cliff_rule, cliff_structural_similarity, CLIFF_TARGET_PCT,
    sim_outer_folds, fold_spread_note, fold_similarity_selection,
    build_fold_entry, entry_record, knn_axes, scale_fold, knn_lists,
    knn_from_lists, score_prediction, DATASET_REPS)
# Scores for the metric (2026-10-05), as s10 does. On a binary dataset scored
# by auc this returns the predicted probability of class 1, which is how the
# searches score auc (find_best_hybrid.py); every other metric (mcc, f1,
# rmse, spearman) gets model.predict. The kNN arms take auc from each class's
# share of the neighbours' weight (similarity_helpers.score_prediction).
from kirby.find_best_hybrid import _predict_for_metric

from kirby.hybrid import compute_feature_importance
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)

# 2026-09-20: the ECFP4 build and the cliff rule that read it moved to
# similarity_helpers (cliff_structural_similarity, calibrated_cliff_rule), so
# s12 decides which molecules are cliffs with the same two calls instead of a
# rule of its own. What that fixed is written at the top of that block.

# Checkpoint identity. A sidecar written under the stored-search arms (before
# 2026-10-05) holds numbers from whole-dataset searches and one kNN setting;
# this value keeps it from being read back.
_CHECKPOINT_VERSION = 'per_fold_selection_knn_axes_2026_10_05'

# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Prediction performance on activity cliff molecules, per outer fold.

    The cliff rule is shared with s12: ECFP4 Tanimoto at or above
    tanimoto_threshold, and an activity gap solved so cliff_target_pct per cent
    of the molecules are cliffs. Pass activity_k to use a fixed multiple of the
    labels' median absolute deviation instead.

    Reports, per outer fold, the dataset's metric over all held-out molecules,
    over the held-out cliff molecules and over the held-out non-cliff
    molecules (and rmse on regression) for every kNN arm at every decided kNN
    setting, and for each decided model on its 2 base representations and 2
    general hybrids. The arms are listed in the module docstring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT S04: PREDICTION PERFORMANCE ON ACTIVITY CLIFFS")
    print("=" * 70)

    # 2026-09-20: which molecules count as cliffs. The structural side is the
    # ECFP4 Tanimoto 0.55 of Stumpfe & Bajorath 2019, unchanged. The activity
    # gap is no longer a fixed 2 x the median absolute deviation of the labels:
    # that flagged 1625 of 2158 caco2 molecules and 125 of 2641 adme_mdr1
    # molecules, so the stratum meant something different on every dataset. It
    # is now solved per dataset to flag cliff_target_pct per cent of the
    # molecules. Passing activity_k restores the old fixed-multiple rule, and
    # either way the file records both counts. See
    # similarity_helpers.calibrated_cliff_rule.
    tanimoto_threshold = kwargs.get('tanimoto_threshold', 0.55)
    activity_k = kwargs.get('activity_k')       # None = calibrate
    cliff_target_pct = kwargs.get('cliff_target_pct', CLIFF_TARGET_PCT)
    # The exponent the importance weights are raised to before they multiply
    # the scaled features in the weighted kNN arms. 0.5 (a square root) is the
    # value this file has always used. Importance weighting inside a
    # similarity experiment's own distance, and its power, are OPEN (handoff
    # 2.10), so the weighted arms and this power are left as they were.
    weight_exponent = float(kwargs.get('weight_exponent', 0.5))
    # The neighbour count and the budget are no longer arguments (2026-10-05):
    # the neighbour counts are the kNN record's 5, 10 and 20, and the hybrids
    # come from per-fold searches over the dataset's decided budgets. Before
    # this date the budget was a fixed 100 features, below the decided general
    # range on 7 of s04's 9 datasets.
    for _gone in ('knn_k', 'budget'):
        if _gone in kwargs:
            print(f"  {_gone}={kwargs[_gone]!r} ignored: set by the kNN record "
                  f"and the decided grid")

    reps = data['reps']
    labels = np.asarray(data['labels'])

    # --- Identify cliff compounds using ECFP4 Tanimoto ---
    # S04 is molecular-only (Bajorath ECFP4 standard). Non-molecular domains
    # should use S12 (cross-domain Euclidean cliffs) instead.
    domain = data.get('domain', 'molecular')
    if domain != 'molecular':
        print(f"  SKIP: S04 requires molecular ECFP4 (domain={domain}). Use S12 for cross-domain cliffs.")
        # F21: the S12 pointer is a CLAIM about another experiment. Nothing here
        # loads, runs or checks an S12 result, so the stub records the pointer as
        # unverified — "handled elsewhere" must stay distinguishable from
        # "answered nowhere" for a reader who only has this file's output.
        return {
            'skipped': True,
            'reason': f'molecular-only experiment, domain={domain}',
            'replacement_experiment': 's12',
            'replacement_verified': False,
            'replacement_note': (
                'S12 (cross-domain Euclidean cliffs) is named as the replacement; '
                'this stub does not read or check for any S12 cliff result for '
                'this dataset. Verify one exists before treating cliffs as covered.'),
        }
    # Derive task_type BEFORE cliff identification so classification cliffs
    # use the class-mismatch indicator, not the regression |delta_y| rule.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    # The same two calls s12 makes, so the two files partition a dataset into
    # cliff and non-cliff molecules identically instead of two different ways.
    cliff_sim, cliff_cutoff, cliff_fp_note, cliff_fp_source = \
        cliff_structural_similarity(data, reps, is_molecular=True)
    if tanimoto_threshold is not None and \
            cliff_fp_source == 'smiles_ecfp4_tanimoto':
        cliff_cutoff = float(tanimoto_threshold)
    print(f"  Cliff structural side: {cliff_fp_source} -- {cliff_fp_note}")

    cliff_mask, cliff_info = calibrated_cliff_rule(
        cliff_sim, labels,
        structural_cutoff=cliff_cutoff,
        structural_source=cliff_fp_note,
        target_pct=cliff_target_pct,
        is_clf=is_clf,
        activity_k=activity_k,
    )
    cliff_indices = set(np.where(cliff_mask)[0].tolist())

    if is_clf:
        print(f"  Cliff rule: similarity>={cliff_cutoff:.4f}, different class")
    elif cliff_info['rule'] == 'calibrated_to_target_share':
        print(f"  Cliff rule: similarity>={cliff_cutoff:.4f} and an activity "
              f"gap above {cliff_info['activity_threshold']:.4f} label units, "
              f"solved so that {cliff_target_pct:.0f} per cent of the "
              f"molecules are cliffs. That gap is "
              f"{cliff_info['activity_k']:.2f} x the median absolute deviation "
              f"of the labels, against the fixed 2.0 this file used before.")
    else:
        print(f"  Cliff rule: similarity>={cliff_cutoff:.4f} and an activity "
              f"gap above {cliff_info['activity_k']:.2f} x the median absolute "
              f"deviation of the labels = "
              f"{cliff_info['activity_threshold']:.4f} label units")
    print(f"  Cliff molecules: {cliff_info['n_cliff_compounds']} of "
          f"{len(labels)} molecules "
          f"({cliff_info['pct_cliff_compounds']:.1f} per cent); "
          f"{cliff_info['n_cliff_pairs']} cliff pairs")
    print(f"  Under the fixed 2 x median-absolute-deviation gap this file used "
          f"before: {cliff_info['uncalibrated_n_cliff_compounds']} of "
          f"{len(labels)} molecules "
          f"({cliff_info['uncalibrated_pct_cliff_compounds']:.1f} per cent), "
          f"{cliff_info['uncalibrated_n_cliff_pairs']} cliff pairs")

    # The decided kNN settings for this dataset (KNN_SEARCH_SPACE_RECORD.md
    # axes 2, 4, 5, 6): distances by dataset group, 5/10/20 neighbours, rank
    # and inverse_square counting, scaffold group not excluded. Every kNN
    # number below carries its distance, neighbour count and counting scheme.
    axes = knn_axes(dataset_name, len(labels))
    print(f"  kNN settings ({axes['group']}): distances {axes['metrics']}, "
          f"neighbours {axes['ks']}, counting {axes['schemes']}, scaffold "
          f"group not excluded")

    # The dataset's two decided models (DECISIONS.md Step 1). knn enters only
    # through the distance search (Step 9), inside fold_similarity_selection.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    # knn as the model a distance hybrid is rebuilt with, when its recipe's
    # internal evaluator is 'match' (fold_hybrid_scoring_kwargs).
    knn_factory = get_model_factories(['knn'], task_type=task_type,
                                      n_samples=len(labels)).get('knn')

    fold_splits = sim_outer_folds(data, config)
    n_folds = len(fold_splits)

    results = {
        # Which selection rule produced the numbers below. s09 reads this
        # and refuses to run its signed-rank test across a mixed set — see
        # _summary_keys.SELECTION_RULE. Added 2026-09-18.
        'selection_rule': _SELECTION_RULE,
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'n_outer_folds': n_folds,
        'models': list(models),
        'models_without_factory': [m for m in models if m not in factories],
        # WHICH RULE DECIDED THE CLIFF MOLECULES. s04 and s12 disagreed about
        # this until 2026-09-20 and neither file said which rule it had used, so
        # every key the shared rule returns is written out here and s12 writes
        # the same block. 'rule' is 'calibrated_to_target_share' or
        # 'fixed_mad_multiple'; 'uncalibrated_*' is what the old fixed 2 x
        # median-absolute-deviation gap would have flagged on these same rows.
        'cliff_definition': {
            'fingerprint_source': cliff_fp_source,
            'fingerprint_note': cliff_fp_note,
            'tanimoto_threshold': tanimoto_threshold,
            'shared_with_s12': True,
            **{k: v for k, v in cliff_info.items() if k != 'cliff_pair_counts'},
        },
        'knn_settings': {
            'group': axes['group'],
            'distances': list(axes['metrics']),
            'neighbour_counts': list(axes['ks']),
            'counting_schemes': list(axes['schemes']),
            'scaffold_group_excluded': False,
            'source': axes['source'],
            'importance_weight_exponent': weight_exponent,
            'note': ('every kNN arm is scored at every (distance, neighbour '
                     'count, counting scheme); features are scaled on the '
                     "fold's training rows (0/1 features unscaled, the rest "
                     'standardised, held-out values cut to the training '
                     'range); the weighted arms then multiply each scaled '
                     'feature by its importance raised to this exponent'),
        },
        'folds': {},
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's04_activity_cliffs', dataset_name),
        run_identity('s04_activity_cliffs', dataset_name, data, config,
                     kwargs=kwargs, weight_exponent=weight_exponent,
                     knn_axes={k: v for k, v in axes.items()
                               if k != 'source'},
                     version=_CHECKPOINT_VERSION))

    def _subset_scores(y_val, pred, proba, classes, rows):
        """The dataset's metric on the held-out rows `rows` (positions in the
        held-out fold); NaN with 1 or fewer rows or where the metric is
        undefined on that subset (one class only, constant labels)."""
        if rows is not None and len(rows) <= 1:
            return float('nan')
        sl = slice(None) if rows is None else rows
        try:
            return float(score_prediction(
                y_val[sl], pred[sl], None if proba is None else proba[sl],
                classes, metric_name, metric_fn))
        except Exception:
            return float('nan')

    def _rmse(y_val, pred, rows):
        if is_clf or (rows is not None and len(rows) <= 1):
            return None
        sl = slice(None) if rows is None else rows
        return float(np.sqrt(mean_squared_error(y_val[sl], pred[sl])))

    def _knn_settings_scores(X_tr_s, X_va_s, y_tr, y_va, val_cliff,
                             val_noncliff):
        """One row per decided (distance, neighbour count, counting scheme):
        the metric over all, cliff and non-cliff held-out molecules. One
        neighbour list per distance serves every count and scheme."""
        rows = []
        for dist_name in axes['metrics']:
            idx, dist = knn_lists(X_tr_s, X_va_s, metric=dist_name,
                                  k_max=axes['k_max'])
            for k in axes['ks']:
                for scheme in axes['schemes']:
                    row = {'distance': dist_name, 'k': int(k),
                           'scheme': scheme}
                    if k > idx.shape[1]:
                        row.update(score_all=None, score_cliff=None,
                                   score_noncliff=None,
                                   note=f'only {idx.shape[1]} training rows')
                        rows.append(row)
                        continue
                    pred, proba, classes = knn_from_lists(
                        y_tr, idx, dist, k=k, scheme=scheme, is_clf=is_clf)
                    row['score_all'] = _subset_scores(y_va, pred, proba,
                                                      classes, None)
                    row['score_cliff'] = _subset_scores(y_va, pred, proba,
                                                        classes, val_cliff)
                    row['score_noncliff'] = _subset_scores(
                        y_va, pred, proba, classes, val_noncliff)
                    if not is_clf:
                        row['rmse_all'] = _rmse(y_va, pred, None)
                        row['rmse_cliff'] = _rmse(y_va, pred, val_cliff)
                        row['rmse_noncliff'] = _rmse(y_va, pred, val_noncliff)
                    rows.append(row)
        return rows

    def _model_scores(factory, X_tr, X_va, y_tr, y_va, val_cliff,
                      val_noncliff):
        """One decided model fitted on the fold's training rows, scored on the
        held-out rows: all, cliff, non-cliff."""
        X_tr_s, X_va_s = scale_fold(X_tr, X_va)
        model = factory()
        model.fit(X_tr_s, y_tr)
        pred = np.asarray(_predict_for_metric(model, X_va_s, metric_fn))

        def _m(rows):
            if rows is not None and len(rows) <= 1:
                return float('nan')
            sl = slice(None) if rows is None else rows
            try:
                return float(metric_fn(y_va[sl], pred[sl]))
            except Exception:
                return float('nan')
        out = {'score_all': _m(None), 'score_cliff': _m(val_cliff),
               'score_noncliff': _m(val_noncliff)}
        if not is_clf:
            out.update(rmse_all=_rmse(y_va, pred, None),
                       rmse_cliff=_rmse(y_va, pred, val_cliff),
                       rmse_noncliff=_rmse(y_va, pred, val_noncliff))
        return out

    def _weights_for_rep(X_tr, y_tr):
        """Importance weights of one representation's features, lightgbm_gain
        on the fold's training rows, raised to weight_exponent (unchanged
        from before 2026-10-05; weighting is OPEN, handoff 2.10)."""
        imp = compute_feature_importance({'_r': X_tr}, y_tr,
                                         method='lightgbm_gain')['_r']
        return np.maximum(np.asarray(imp, dtype=np.float64), 0) ** weight_exponent

    def _weights_for_hybrid(bp, feat_info, tr, y_tr, width):
        """Importance weights of a hybrid's selected features, from the same
        importance method the search built the hybrid with, on the fold's
        training rows, raised to weight_exponent. Unchanged from before
        2026-10-05 (weighting is OPEN, handoff 2.10), except that a weight
        vector whose length differs from the built hybrid's width now raises
        instead of broadcasting wrongly."""
        train_reps_h = {name: np.asarray(rep)[tr] for name, rep in reps.items()}
        imp = compute_feature_importance(
            train_reps_h, y_tr,
            method=getattr(bp, 'importance_method', 'lightgbm_gain'))
        w_parts = []
        for rname in feat_info:
            if not isinstance(feat_info[rname], dict):
                continue
            if 'selected_indices' not in feat_info[rname]:
                continue
            idx = feat_info[rname]['selected_indices']
            if rname in imp:
                w_parts.append(np.asarray(imp[rname])[idx])
            else:
                w_parts.append(np.ones(len(idx)))
        w = np.maximum(np.concatenate(w_parts), 0) ** weight_exponent
        if len(w) != width:
            raise ValueError(f'{len(w)} importance weights for a hybrid '
                             f'{width} features wide')
        return w

    def _run_fold(fold_i, tr, va):
        y_tr, y_va = labels[tr], labels[va]
        val_cliff = np.array([i for i, vi in enumerate(va)
                              if int(vi) in cliff_indices], dtype=int)
        val_noncliff = np.array([i for i, vi in enumerate(va)
                                 if int(vi) not in cliff_indices], dtype=int)
        rec = {'fold': int(fold_i), 'selection_scope': 'fold_train_rows',
               'n_train_rows': int(len(tr)), 'n_heldout_rows': int(len(va)),
               'n_cliff_rows': int(len(val_cliff)),
               'n_noncliff_rows': int(len(val_noncliff)),
               'knn_arms': {}, 'model_arms': {}, 'hybrid_width_built': {},
               'errors': []}
        print(f"\n  Fold {fold_i}: {len(tr)} training rows, {len(va)} held "
              f"out ({len(val_cliff)} cliff, {len(val_noncliff)} non-cliff)")

        # Everything this fold selects, from searches on its training rows
        # only: each decided model's 2 base representations and 2 general
        # hybrids, knn's 2 best single representations and top hybrid in the
        # distance search, and the top hybrid of the similarity search.
        sel = fold_similarity_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
        rec['unavailable'] = dict(sel.get('unavailable') or {})
        rec['selection'] = {
            'models': {
                m: (None if s is None else {
                    'base': list(s.get('base') or []),
                    'base_train_rows_scores': [
                        b.get('score') for b in
                        ((s.get('base_detail') or {}).get('selected') or [])],
                    'base_shortfall': (s.get('base_detail') or {}).get(
                        'shortfall_reason'),
                    'hybrids': [entry_record(e)
                                for e in (s.get('hybrids') or [])],
                    'hybrids_skipped_identical_features': list(
                        s.get('hybrids_skipped_identical_features') or []),
                })
                for m, s in sel['models'].items()},
            'knn_base': list(sel.get('knn_base') or []),
            'knn_base_train_rows_scores': [
                b.get('score') for b in
                ((sel.get('knn_base_detail') or {}).get('selected') or [])],
            'distance_hybrid': entry_record(sel.get('distance_hybrid')),
            'similarity_hybrid': entry_record(sel.get('similarity_hybrid')),
        }

        # The hybrid spaces, each rebuilt once on the fold's training rows
        # with the fold search's own representation scores (build_fold_entry;
        # dirty-run bug 1) and shared by the kNN arms and the model arms.
        hybrid_spaces = []        # (arm, entry, model factory for the rebuild)
        for m in models:
            s = sel['models'].get(m)
            for j, e in enumerate((s or {}).get('hybrids') or [], 1):
                hybrid_spaces.append((f'general|{m}|hybrid{j}', e,
                                      factories.get(m)))
        if sel.get('distance_hybrid') is not None:
            hybrid_spaces.append(('distance|knn|hybrid1',
                                  sel['distance_hybrid'], knn_factory))
        if sel.get('similarity_hybrid') is not None:
            # A similarity-objective recipe is rebuilt with no model.
            hybrid_spaces.append(('similarity|hybrid1',
                                  sel['similarity_hybrid'], None))
        built = {}
        for arm, e, fac in hybrid_spaces:
            try:
                X_tr_h, X_va_h, feat_info = build_fold_entry(
                    data, e, train_idx=tr, val_idx=va, model_factory=fac,
                    metric_fn=metric_fn, lower_is_better=lower_is_better)
                X_tr_h, X_va_h = np.asarray(X_tr_h), np.asarray(X_va_h)
                built[arm] = (X_tr_h, X_va_h, feat_info, e)
                rec['hybrid_width_built'][arm] = {
                    'budget': getattr(e.get('blueprint'), 'budget', None),
                    'width_built': int(X_tr_h.shape[1])}
            except Exception as exc:
                rec['errors'].append({'arm': arm, 'stage': 'rebuild',
                                      'error': f'{type(exc).__name__}: {exc}'})
                print(f"    {arm}: rebuild failed: {type(exc).__name__}: {exc}")

        # --- kNN arms ---
        knn_spaces = []           # (arm, kind, name, X_tr, X_va, weight_fn)
        for j, rname in enumerate(sel.get('knn_base') or [], 1):
            if rname not in reps:
                rec['errors'].append({'arm': f'knn_base{j}', 'stage': 'load',
                                      'error': f'{rname} not loaded'})
                continue
            X = np.asarray(reps[rname])
            knn_spaces.append((
                f'knn_base{j}', 'representation', rname, X[tr], X[va],
                (lambda _Xt=X[tr]: _weights_for_rep(_Xt, y_tr))))
        for arm, (X_tr_h, X_va_h, feat_info, e) in built.items():
            knn_spaces.append((
                arm, 'hybrid', e.get('label'), X_tr_h, X_va_h,
                (lambda _bp=e.get('blueprint'), _fi=feat_info,
                 _w=X_tr_h.shape[1]: _weights_for_hybrid(_bp, _fi, tr, y_tr,
                                                         _w))))
        for arm, kind, name, X_tr, X_va, weight_fn in knn_spaces:
            entry = {'kind': kind, 'name': name}
            X_tr_s, X_va_s = scale_fold(X_tr, X_va)
            try:
                entry['unweighted'] = _knn_settings_scores(
                    X_tr_s, X_va_s, y_tr, y_va, val_cliff, val_noncliff)
            except Exception as exc:
                entry['unweighted'] = None
                rec['errors'].append({'arm': arm, 'stage': 'knn unweighted',
                                      'error': f'{type(exc).__name__}: {exc}'})
            try:
                w = weight_fn()
                entry['weighted'] = _knn_settings_scores(
                    X_tr_s * w, X_va_s * w, y_tr, y_va, val_cliff,
                    val_noncliff)
            except Exception as exc:
                entry['weighted'] = None
                rec['errors'].append({'arm': arm, 'stage': 'knn weighted',
                                      'error': f'{type(exc).__name__}: {exc}'})
            rec['knn_arms'][arm] = entry
            _best = [r for r in (entry['unweighted'] or [])
                     if r.get('score_cliff') is not None
                     and np.isfinite(r['score_cliff'])]
            if _best:
                _b = (min if lower_is_better else max)(
                    _best, key=lambda r: r['score_cliff'])
                print(f"    kNN {arm:<28} ({name}) best {metric_name}_cliff "
                      f"over {len(_best)} settings = {_b['score_cliff']:.4f} "
                      f"[{_b['distance']}, k={_b['k']}, {_b['scheme']}] "
                      f"(not a selection; every setting is stored)")

        # --- Decided-model arms: 2 base representations + 2 general hybrids
        for m in models:
            fac = factories.get(m)
            s = sel['models'].get(m)
            if fac is None or s is None:
                rec['model_arms'][m] = None
                continue
            arms = {}
            for j, rname in enumerate(s.get('base') or [], 1):
                try:
                    X = np.asarray(reps[rname])
                    arms[f'base{j}'] = dict(
                        rep=rname,
                        **_model_scores(fac, X[tr], X[va], y_tr, y_va,
                                        val_cliff, val_noncliff))
                except Exception as exc:
                    rec['errors'].append({'arm': f'{m}|base{j}',
                                          'stage': 'fit',
                                          'error': f'{type(exc).__name__}: {exc}'})
            for j in (1, 2):
                arm = f'general|{m}|hybrid{j}'
                if arm not in built:
                    continue
                X_tr_h, X_va_h, _fi, e = built[arm]
                try:
                    arms[f'hybrid{j}'] = dict(
                        hybrid=entry_record(e),
                        **_model_scores(fac, X_tr_h, X_va_h, y_tr, y_va,
                                        val_cliff, val_noncliff))
                except Exception as exc:
                    rec['errors'].append({'arm': f'{m}|hybrid{j}',
                                          'stage': 'fit',
                                          'error': f'{type(exc).__name__}: {exc}'})
            rec['model_arms'][m] = arms
            _line = ', '.join(f"{k}={v['score_cliff']:.4f}"
                              for k, v in arms.items())
            print(f"    {m} held-out {metric_name}_cliff: {_line or '-'}")
        return rec

    # --- One record per outer fold, written to the sidecar as it finishes ---
    for fold_i, (tr, va) in enumerate(fold_splits):
        if _ck.done(fold=fold_i):
            rec = _ck.get(fold=fold_i)
            print(f"  Fold {fold_i}: read back from the checkpoint")
        else:
            rec = _run_fold(fold_i, np.asarray(tr), np.asarray(va))
            # A fold with an error is not stored, so a rerun retries it.
            if not rec['errors']:
                _ck.put(rec, fold=fold_i)
        results['folds'][str(fold_i)] = rec

    _summarise(results, n_folds, lower_is_better, is_clf, metric_name)
    return results


# =============================================================================
# SUMMARY
# =============================================================================

def _setting_key(row):
    return f"{row['distance']}|k{row['k']}|{row['scheme']}"


def _knn_series(results, arm, weighting, stratum):
    """{setting key: [score per fold, NaN where absent]} for one kNN arm."""
    folds = sorted(results['folds'], key=int)
    out = {}
    for pos, f in enumerate(folds):
        entry = (results['folds'][f].get('knn_arms') or {}).get(arm) or {}
        for row in entry.get(weighting) or []:
            v = row.get(stratum)
            out.setdefault(_setting_key(row), [float('nan')] * len(folds))[
                pos] = float('nan') if v is None else float(v)
    return out


def _headline(series, lower_is_better, select_series=None):
    """Per fold, the score at the setting with the best mean over the OTHER
    folds of select_series (the overall held-out metric), so no fold's own
    score picks its setting. Returns (per-fold scores, per-fold setting);
    with one fold there are no other folds and both are None."""
    select_series = select_series or series
    keys = sorted(series)
    if not keys:
        return None, None
    n = len(next(iter(series.values())))
    if n < 2:
        return None, None
    scores, picks = [], []
    for i in range(n):
        best_k, best_v = None, None
        for k in keys:
            other = [v for j, v in enumerate(select_series.get(k, []))
                     if j != i and np.isfinite(v)]
            if not other:
                continue
            m = float(np.mean(other))
            if best_v is None or (m < best_v if lower_is_better else m > best_v):
                best_k, best_v = k, m
        picks.append(best_k)
        scores.append(float('nan') if best_k is None else series[best_k][i])
    return scores, picks


def _nanmean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def _summarise(results, n_folds, lower_is_better, is_clf, metric_name):
    """Per arm: per-fold scores and their mean over folds, and the
    fold_reselection block. No significance test, no interval: per-fold
    scores and their spread only."""
    folds = sorted(results['folds'], key=int)
    knn_arms = []
    for f in folds:
        for a in (results['folds'][f].get('knn_arms') or {}):
            if a not in knn_arms:
                knn_arms.append(a)

    # kNN arms. 'headline' is each fold's score at the setting chosen on the
    # other folds' overall held-out scores; every setting is under 'by_setting'.
    summary = {}
    for arm in knn_arms:
        summary[arm] = {}
        for weighting in ('unweighted', 'weighted'):
            sel_series = _knn_series(results, arm, weighting, 'score_all')
            block = {'by_setting': {}}
            for stratum in ('score_all', 'score_cliff', 'score_noncliff') + (
                    () if is_clf else ('rmse_all', 'rmse_cliff',
                                       'rmse_noncliff')):
                ser = _knn_series(results, arm, weighting, stratum)
                for k, v in ser.items():
                    block['by_setting'].setdefault(k, {})[stratum] = {
                        'fold_scores': v, 'mean_over_folds': _nanmean(v),
                        'spread_note': fold_spread_note(v)}
                # The setting is always chosen on the overall metric in its
                # own direction; the stratum only says which score is read.
                hl, picks = _headline(ser, lower_is_better, sel_series)
                block.setdefault('headline', {})[stratum] = (
                    None if hl is None else
                    {'fold_scores': hl, 'mean_over_folds': _nanmean(hl),
                     'settings_by_fold': picks})
            if n_folds < 2:
                block['headline_note'] = (
                    'one fold: no other fold to choose the kNN setting on, so '
                    'no headline; every setting is under by_setting')
            summary[arm][weighting] = block
    results['knn_summary'] = summary

    # Decided-model arms.
    msum = {}
    for f in folds:
        for m, arms in (results['folds'][f].get('model_arms') or {}).items():
            for slot, v in (arms or {}).items():
                d = msum.setdefault(m, {}).setdefault(slot, {
                    'fold_scores': {}, 'fold_scores_cliff': {},
                    'fold_scores_noncliff': {}})
                d['fold_scores'][f] = v.get('score_all')
                d['fold_scores_cliff'][f] = v.get('score_cliff')
                d['fold_scores_noncliff'][f] = v.get('score_noncliff')
    for m in msum:
        for slot, d in msum[m].items():
            for k in ('fold_scores', 'fold_scores_cliff', 'fold_scores_noncliff'):
                d[k] = [d[k].get(f, float('nan')) for f in folds]
                d[k] = [float('nan') if x is None else float(x) for x in d[k]]
            d['mean_all'] = _nanmean(d['fold_scores'])
            d['mean_cliff'] = _nanmean(d['fold_scores_cliff'])
            d['mean_noncliff'] = _nanmean(d['fold_scores_noncliff'])
            d['spread_note'] = fold_spread_note(d['fold_scores_cliff'])
    results['model_summary'] = msum

    # Kept under the keys s09 reads: per_rep[*] and hybrid['weighted'/
    # 'unweighted'], at each fold's headline setting. per_rep now holds the
    # two knn_base slots (knn's 2 best single representations per fold); the
    # representation in a slot can differ between folds. The hybrid lead arm
    # is knn's top hybrid of the distance search, the arm the old lead
    # (predictive distance1) became.
    def _hl(arm, weighting, stratum):
        # NaN, not None, where there is no headline: s09 runs np.isnan on
        # these values.
        b = (((summary.get(arm) or {}).get(weighting) or {})
             .get('headline') or {}).get(stratum)
        return float('nan') if b is None else b['mean_over_folds']

    def _hl_settings(arm, weighting, stratum):
        # The kNN setting ('distance|k<count>|scheme') behind each fold's
        # headline score; None where there is no headline (one fold).
        b = (((summary.get(arm) or {}).get(weighting) or {})
             .get('headline') or {}).get(stratum)
        return None if b is None else {
            f: s for f, s in zip(folds, b['settings_by_fold'])}

    def _settings_block(arm, strata):
        # Every number in per_rep/hybrid is a mean over folds of a per-fold
        # headline, and the headline setting can differ between folds, so
        # the settings are stored next to the numbers.
        return {
            'knn_setting_format': 'distance|k<neighbour count>|counting scheme',
            'knn_settings_source': f"knn_summary['{arm}'][weighting]"
                                   "['headline'][stratum]['settings_by_fold']",
            'knn_settings_by_fold': {
                w: {st: _hl_settings(arm, w, st) for st in strata}
                for w in ('unweighted', 'weighted')}}
    _strata = ('score_all', 'score_cliff', 'score_noncliff') + (
        () if is_clf else ('rmse_cliff',))
    results['per_rep'] = {}
    for arm in ('knn_base1', 'knn_base2'):
        if arm not in summary:
            continue
        pr = {'reps_by_fold': {
            f: ((results['folds'][f].get('knn_arms') or {}).get(arm) or {})
            .get('name') for f in folds}}
        for st in ('score_all', 'score_cliff', 'score_noncliff'):
            pr[st] = _hl(arm, 'unweighted', st)
            pr[f'{st}_weighted'] = _hl(arm, 'weighted', st)
        if not is_clf:
            pr['rmse_cliff'] = _hl(arm, 'unweighted', 'rmse_cliff')
            pr['rmse_cliff_weighted'] = _hl(arm, 'weighted', 'rmse_cliff')
        pr['metric'] = metric_name
        pr.update(_settings_block(arm, _strata))
        results['per_rep'][arm] = pr
    lead = next((a for a in ('distance|knn|hybrid1', 'similarity|hybrid1')
                 if a in summary),
                next((a for a in knn_arms if not a.startswith('knn_base')),
                     None))
    results['hybrid'] = {'lead_arm': lead}
    if lead is not None:
        for weighting in ('unweighted', 'weighted'):
            h = {st: _hl(lead, weighting, st)
                 for st in ('score_all', 'score_cliff', 'score_noncliff')}
            if not is_clf:
                h['rmse_cliff'] = _hl(lead, weighting, 'rmse_cliff')
            h['knn_setting_format'] = (
                'distance|k<neighbour count>|counting scheme')
            h['knn_settings_by_fold'] = {
                st: _hl_settings(lead, weighting, st) for st in _strata}
            h['knn_settings_source'] = (
                f"knn_summary['{lead}']['{weighting}']['headline'][stratum]"
                "['settings_by_fold']")
            results['hybrid'][weighting] = h

    # --- Per-fold reselection block (CONTRACT json_convention) ---
    # Reader: fold_reselection(block['candidates_by_model'],
    #                        block['baselines_by_model'], block['lower_is_better']).
    # fold_scores: the metric over the fold's CLIFF rows alone, so the
    # leave-one-fold-out re-pick answers the cliff question. A fold whose
    # held-out rows held 1 or fewer cliff molecules is NaN in every arm.
    # knn: each (hybrid arm, weighting, kNN setting) is a candidate and each
    # (knn_base slot, weighting, kNN setting) a baseline, so the re-pick
    # chooses the kNN setting on the other folds as well. Each decided model:
    # its 2 general hybrids against its 2 base representations. Pruning is
    # off in every per-fold search, so no fold is excluded.
    cands, bases = {}, {}
    for arm in knn_arms:
        for weighting in ('unweighted', 'weighted'):
            ser = _knn_series(results, arm, weighting, 'score_cliff')
            for k, v in ser.items():
                rec = {'name': f'{arm}|{weighting}|{k}', 'fold_scores': v}
                if arm.startswith('knn_base'):
                    bases.setdefault('knn', {})[rec['name']] = v
                else:
                    cands.setdefault('knn', []).append(rec)
    for m, slots in msum.items():
        for slot, d in slots.items():
            if slot.startswith('hybrid'):
                cands.setdefault(m, []).append(
                    {'name': f'{m}|{slot}', 'fold_scores': d['fold_scores_cliff']})
            else:
                bases.setdefault(m, {})[f'{m}|{slot}'] = d['fold_scores_cliff']
    results['fold_reselection'] = {
        'lower_is_better': bool(lower_is_better),
        'fold_score_stratum': 'cliff_rows_only',
        'exclude_folds': [],
        'fold0_pruned': False,
        'recipe_selection': 'fold_train_rows',
        'recipe_selection_note': (
            "each fold's hybrids and representations were chosen by searches "
            "on that fold's training rows only"),
        'candidates_by_model': cands,
        'baselines_by_model': bases,
    }

    # Printed summary: each arm's mean over folds of the cliff metric at its
    # headline setting.
    print(f"\n  --- Summary ({metric_name} on cliff molecules, mean over "
          f"{n_folds} fold(s)) ---")
    for arm in knn_arms:
        for weighting in ('unweighted', 'weighted'):
            v = _hl(arm, weighting, 'score_cliff')
            print(f"    kNN {arm:<28} {weighting:<10} "
                  f"{'-' if not np.isfinite(v) else f'{v:.4f}'}")
    for m, slots in msum.items():
        for slot, d in slots.items():
            print(f"    {m:<8} {slot:<8} {d['mean_cliff']:.4f}"
                  + (f"  ({d['spread_note']})" if d['spread_note'] else ''))


if __name__ == '__main__':
    parser = create_experiment_parser('S04: Activity Cliffs',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    # Representations: the dataset's pruned general list only (handoff 2.4
    # step 9), loaded the way similarity_helpers.run_all loads them. --reps is
    # not read (2026-10-05); it named a preset, not the decided list.
    _ds_reps = DATASET_REPS.get(args.dataset)
    if _ds_reps is None:
        raise SystemExit(f"{args.dataset} has no DATASET_REPS entry")
    print(f"Reps: {_ds_reps['general']} (DATASET_REPS general; --reps "
          f"{args.reps!r} not read)")
    data = load_data(args.dataset, rep_preset='all',
                     rep_names=_ds_reps['general'])
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's04_activity_cliffs', args.dataset)
