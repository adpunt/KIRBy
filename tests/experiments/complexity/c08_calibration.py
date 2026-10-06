#!/usr/bin/env python3
"""C08: Uncertainty Calibration at Optimal vs Suboptimal Complexity.

Paper: Complexity
Conclusion: Complexity matching improves calibration.

2026-10-05, aligned with HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4 (chat 3):
per outer fold, one search on that fold's training rows on the dataset's
decided grid (fold_standard_selection). For each of the dataset's two decided
models it takes the 2 best single representations and the 2 best hybrids, both
ranked on the fold's training rows. Each single representation is fitted at its
own width; each hybrid is rebuilt on the fold's training rows at its searched
budget ('optimal'), at a smaller budget ('underfit') and at the summed width of
the representations ('overfit'). Every one is scored on the held-out rows with
split-conformal 90% intervals (regression) or prediction sets (classification).

Until 2026-10-05 the recipe came from a search over every row of the dataset
(c02's result file, or the stored whole-dataset search) and one 'mismatched'
condition fitted Ridge or LogisticRegression. The whole-dataset recipe is gone:
it leaked the held-out fold into the recipe. c08 no longer reads c02. Ridge and
LogisticRegression are gone too: neither is one of the dataset's two models.

DECIDED 2026-10-06 (Adelaide delegated): the 'mismatched' condition fits the
dataset's other decided model on each hybrid as built at the overfit width
(the summed width of the representations). On a dataset with one decided model
it is not run, and the reason is stored.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_model_factories, make_cv_splitter,
    get_domain_metric, fold_standard_selection, fold_searches, top_hybrids,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from experiments.decided_grid import decided_models, decided_search
from kirby.scaling import FeatureScaler
from kirby.find_best_hybrid import _predict_for_metric, nested_cv_outer_folds

# Seed of the outer split and of every per-fold search's namespace
# (ocv{n}f{i}s42), the same one the other aligned experiments use so they
# share the stored per-fold searches.
SEED = 42

# What this file's results are. A file written before the per-fold alignment
# carries no such key. Bumped 2026-10-06 when 'mismatched' started fitting the
# dataset's other decided model on the overfit-width hybrid.
C08_RESULT_IDENTITY = 'c08_per_fold_mismatched_other_model_2026-10-06'

ONE_FOLD_NOTE = 'one fold, no spread'

ALL_CONDITIONS = ('optimal', 'underfit', 'overfit', 'mismatched')

# DECIDED 2026-10-06 (Adelaide delegated): what 'mismatched' fits. Stored in
# the result file.
MISMATCHED_RULE = ("the dataset's other decided model, fitted on the hybrid "
                   "chosen for this model as built at the overfit width (the "
                   "summed width of the representations)")

UNDERFIT_BUDGET_RULE = ('25 features if below the optimal budget, else the '
                        'largest budget in the budget list below it, else not '
                        'run')

# The fields that name a hybrid recipe in a top_hybrids entry, written to the
# result file beside every rebuilt hybrid.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')


def _cal_split(train_groups, n_train, n_cal, fold_i):
    """Cut the conformal calibration rows out of a fold's training rows.

    Group-aware: a scaffold that appears in the proper-training half never
    appears in the calibration half, so the calibrated interval width comes
    from residuals as hard as the held-out fold's. Falls back to a permutation
    only when the fold has fewer than two distinct groups, where no group-aware
    cut exists.
    """
    if train_groups is not None and len(np.unique(train_groups)) >= 2:
        splitter, _ = make_cv_splitter(
            1, groups=train_groups, seed=42 + fold_i)
        dummy = np.zeros((n_train, 1))
        ptrain_idx, cal_idx = next(
            splitter.split(dummy, groups=train_groups))
        return ptrain_idx, cal_idx
    perm = np.random.RandomState(42 + fold_i).permutation(n_train)
    return perm[n_cal:], perm[:n_cal]


def _underfit_for(opt_budget, total_dim, budget_list):
    """(underfit budget in features or None, reason when None).

    2026-10-05, owner delegated, 2026-10-05: the underfit condition must ask
    for fewer features than the optimal condition. It was
    min(25, max(5, total_dim // 4)), which on cdr asked for more features
    than the optimal recipe's budget. Now: 25 features if 25 is below the
    optimal budget; otherwise the largest budget in this experiment's budget
    list (config['budgets'], the dataset's decided budgets) below the optimal
    budget; if there is none, the underfit condition is not run and the reason
    is stored.
    """
    _opt = total_dim if opt_budget is None else int(opt_budget)
    if 25 < _opt:
        return 25, None
    _below = [b for b in budget_list if b < _opt]
    if _below:
        return _below[-1], None
    return None, (f'no budget below the optimal budget of {_opt} features '
                  f'in the budget list {budget_list}; underfit '
                  f'condition not run')


def _train_row_score_at(search, model, entry, budget):
    """The per-fold search's own score for this recipe at another budget.

    The mean over the inner folds of the outer fold's training rows, for the
    configuration that equals `entry` in every field but the budget. None when
    the search did not run that budget (25 features and the summed width are
    usually not searched), with the reason.
    """
    if search is None:
        return None, 'the search for this augmentation condition is missing'
    try:
        th = top_hybrids(search, model, n=100000, budget=budget,
                         require_full=False)
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'
    keys = ('allocation_method', 'importance_method', 'internal_evaluator',
            'step_size', 'rep_count', 'rep_subset', 'tune')
    for cand in th['selected']:
        if all(cand.get(k) == entry.get(k) for k in keys):
            return cand['score_mean'], None
    return None, f'the per-fold search did not run this recipe at {budget} features'


def _calibrate(X_train, X_val, y_train, y_val, train_groups, factory, fold_i,
               metric_fn, is_classification):
    """Fit one model on a fold's training rows and measure its conformal
    calibration on the held-out rows.

    Returns (fold_result dict, None) or (None, reason).
    """
    # 2026-10-05 (handoff 2.3 item 17): FeatureScaler, so 0/1 features pass
    # through unscaled and every other feature is standardised. Fitted on the
    # training rows only. Matrices stay in the dtype they were loaded in
    # (float32, item 18).
    scaler = FeatureScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    # 2026-10-05, owner delegated, 2026-10-05: each validation
    # feature is capped at the range it takes on the scaled
    # training rows. On adme_hlm one validation molecule sat 83
    # million training standard deviations out and the dnn's rmse
    # ran to thousands.
    X_val_s = np.clip(X_val_s, X_train_s.min(0), X_train_s.max(0))

    # 2026-10-05: the model fitted on ALL training rows that stood here was
    # never used for any number (every score below comes from model_conf, the
    # proper-training model), so it is no longer fitted.

    # The calibration rows are cut on scaffold groups, not at random. A random
    # 20% puts molecules from the same scaffold on both sides, so the residuals
    # the interval is calibrated from are easier than the held-out fold the
    # coverage is reported on.
    n_train = len(y_train)
    n_cal = max(20, int(n_train * 0.2))
    ptrain_idx, cal_idx = _cal_split(train_groups, n_train, n_cal, fold_i)
    X_ptrain, y_ptrain = X_train_s[ptrain_idx], y_train[ptrain_idx]
    X_cal, y_cal = X_train_s[cal_idx], y_train[cal_idx]

    model_conf = factory()
    model_conf.fit(X_ptrain, y_ptrain)

    alpha = 0.1
    if is_classification:
        # val_score from model_conf so the performance number and the
        # calibration metrics come from the SAME model. For AUC metrics,
        # predict_proba (hard labels would give a meaningless AUROC).
        _from_kfh = _predict_for_metric(model_conf, X_val_s, metric_fn)
        val_score = float(metric_fn(y_val, _from_kfh))

        if not hasattr(model_conf, 'predict_proba'):
            # No predict_proba: report the score only.
            return {
                'val_score': val_score, 'picp': float('nan'),
                'avg_set_size': float('nan'),
                'sharpness_ratio': float('nan'), 'cal_error': float('nan'),
                'conformal_q': float('nan'), 'n_val': len(y_val),
                'ue_corr': float('nan'),
            }, None

        # Conformal classification: nonconformity = 1 - P(true class).
        # Calibration points whose true label was never seen in proper-train
        # are reported and assigned worst-case nonconformity (1.0) rather than
        # silently dropped (which would bias coverage).
        cal_probs = model_conf.predict_proba(X_cal)
        class_to_col = {c: j for j, c in enumerate(model_conf.classes_)}
        cal_scores = np.empty(len(y_cal), dtype=float)
        n_unseen = 0
        for i in range(len(y_cal)):
            col = class_to_col.get(int(y_cal[i]))
            if col is None:
                cal_scores[i] = 1.0
                n_unseen += 1
            else:
                cal_scores[i] = 1 - cal_probs[i, col]
        if n_unseen:
            print(f"    fold {fold_i}: {n_unseen}/{len(y_cal)} "
                  f"calibration points have classes unseen in proper-train "
                  f"(assigned worst-case nonconformity 1.0)")
        if len(cal_scores) == 0:
            return None, 'no valid calibration scores'
        residuals = cal_scores
    else:
        y_pred_val = model_conf.predict(X_val_s)
        val_score = float(metric_fn(y_val, y_pred_val))
        y_pred_cal = model_conf.predict(X_cal)
        residuals = np.abs(y_cal - y_pred_cal)

    # ⛔ A CALIBRATION HALF THIS SMALL CANNOT CARRY THE LEVEL ASKED FOR. n_cal
    # above is computed and then IGNORED by the group-aware branch of
    # _cal_split, which takes make_cv_splitter(1, ...) -- GroupShuffleSplit at
    # test_size 0.2 -- so the real size is whatever the groups give. Split
    # conformal needs at least (1-alpha)/alpha rows, 9 at alpha=0.1: below
    # that ceil((n+1)(1-alpha))/n exceeds 1, the min() clamps it, and q becomes
    # the LARGEST calibration residual. The interval is then the widest the
    # calibration set can produce, not a 90% one, and the PICP printed beside
    # it is not coverage at the stated level.
    # Conformal quantile with finite-sample correction (Romano et al. 2019).
    n_cal_used = len(residuals)
    _raw_level = np.ceil((n_cal_used + 1) * (1 - alpha)) / n_cal_used
    conformal_level = min(1.0, _raw_level)
    level_clamped = bool(_raw_level > 1.0)
    if level_clamped:
        print(f"      fold {fold_i}: calibration half has {n_cal_used} "
              f"rows, under the {int(np.ceil((1 - alpha) / alpha))} that "
              f"alpha={alpha} needs; q is the largest residual and the "
              f"interval is NOT at the {100 * (1 - alpha):.0f}% level")
    q = np.quantile(residuals, conformal_level)

    if is_classification:
        val_probs = model_conf.predict_proba(X_val_s)
        classes = model_conf.classes_
        pred_sets = []
        for i in range(len(y_val)):
            pset = [classes[j] for j in range(len(classes))
                    if 1 - val_probs[i, j] <= q]
            pred_sets.append(pset)
        covered = np.array([y_val[i] in pred_sets[i] for i in range(len(y_val))])
        picp = float(np.mean(covered))
        avg_set_size = float(np.mean([len(ps) for ps in pred_sets]))
        n_classes = len(classes)
        # Sharpness: avg set size / n_classes (≈ 1/n_classes ⇒ sharp/decisive;
        # 1 ⇒ trivial full-prediction-set). LOWER is better.
        sharpness_ratio = (avg_set_size / n_classes if n_classes > 0
                           else float('nan'))
        cal_error = abs(picp - (1 - alpha))
        # Uncertainty-error: entropy of predicted probabilities vs correctness.
        pred_labels = model_conf.predict(X_val_s)
        correct = (pred_labels == y_val).astype(float)
        entropy = -np.sum(val_probs * np.log(val_probs + 1e-12), axis=1)
        ue_corr = float('nan')
        if len(entropy) >= 3:
            ue_corr = float(spearmanr(entropy, 1 - correct)[0])
        return {
            'val_score': val_score, 'picp': picp,
            'avg_set_size': avg_set_size,
            'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
            'conformal_q': float(q), 'n_val': len(y_val),
            'n_cal': int(n_cal_used),
            'conformal_level_clamped': level_clamped,
            'n_classes': n_classes,
            'ue_corr': ue_corr,
        }, None

    interval_lo = y_pred_val - q
    interval_hi = y_pred_val + q
    covered = (y_val >= interval_lo) & (y_val <= interval_hi)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(interval_hi - interval_lo))
    target_range = float(np.ptp(y_val))
    sharpness_ratio = mpiw / target_range if target_range > 0 else float('nan')
    cal_error = abs(picp - (1 - alpha))
    errors = np.abs(y_val - y_pred_val)
    # Tracker C08d, owner delegated 2026-10-05: the uncertainty-error
    # correlation needs a model with per-tree predictions; on a model without
    # them it stays NaN (no new model is fitted for it).
    ue_corr = float('nan')
    if hasattr(model_conf, 'estimators_') and len(errors) >= 3:
        try:
            tree_preds = np.array([t.predict(X_val_s)
                                   for t in model_conf.estimators_])
            ue_corr = float(spearmanr(tree_preds.std(axis=0), errors)[0])
        except Exception:
            ue_corr = float('nan')
    # 'val_score' was 'val_r2' until 2026-10-05: it holds the dataset's
    # metric, which is rmse or spearman, not R2.
    return {
        'val_score': val_score, 'picp': picp, 'mpiw': mpiw,
        'sharpness_ratio': sharpness_ratio, 'cal_error': cal_error,
        'conformal_q': float(q), 'n_val': len(y_val),
        'n_cal': int(n_cal_used),
        'conformal_level_clamped': level_clamped,
        'ue_corr': ue_corr,
    }, None


def _print_fold_line(fold_i, model, slot, cond, metric_name, res, n_feat):
    extra = (f"set_size={res['avg_set_size']:.2f}" if 'avg_set_size' in res
             else f"MPIW={res.get('mpiw', float('nan')):.3f}")
    print(f"    fold {fold_i} {model:8s} {slot:8s} {cond:9s} {n_feat:6d} "
          f"features  {metric_name}={res['val_score']:.4f}, "
          f"PICP={res['picp']:.3f}, {extra}, "
          f"cal_error={res['cal_error']:.3f}")


def _summarise(per_fold):
    """{model: {slot: {condition: {'avg', 'n_folds_in_avg', 'n_folds_scored',
    'per_fold'}}}} over the held-out folds.

    A slot is a rank, not a fixed recipe: 'hybrid1' on fold 0 and on fold 1 are
    each fold's top hybrid, chosen on that fold's training rows. With one
    scored fold there is no spread and 'spread_note' says so.
    """
    out = {}
    for fold_rec in per_fold:
        for m, mr in (fold_rec.get('models') or {}).items():
            if not mr:
                continue
            for row in mr.get('base', []) + mr.get('hybrids', []):
                for cond, c in (row.get('conditions') or {}).items():
                    h = c.get('heldout')
                    if h is None:
                        continue
                    d = (out.setdefault(m, {}).setdefault(row['slot'], {})
                         .setdefault(cond, {'per_fold': []}))
                    d['per_fold'].append(dict(
                        h, fold=fold_rec['fold'],
                        name=row.get('rep') or row.get('label'),
                        n_features_built=c.get('n_features_built'),
                        train_row_score=c.get('train_row_score')))
    for m in out:
        for slot in out[m]:
            for cond, d in out[m][slot].items():
                fms = d['per_fold']
                keys = set()
                for fm in fms:
                    keys.update(k for k, v in fm.items()
                                if isinstance(v, (int, float))
                                and not isinstance(v, bool))
                keys -= {'fold'}
                avg, n_avg, spread = {}, {}, {}
                for key in sorted(keys):
                    vals = [float(fm[key]) for fm in fms
                            if fm.get(key) is not None
                            and np.isfinite(float(fm[key]))]
                    avg[key] = float(np.mean(vals)) if vals else float('nan')
                    n_avg[key] = len(vals)
                    spread[key] = (float(np.std(vals)) if len(vals) > 1
                                   else None)
                d['avg'] = avg
                d['std_over_folds'] = spread
                d['n_folds_in_avg'] = n_avg
                d['n_folds_scored'] = len(fms)
                if len(fms) == 1:
                    d['spread_note'] = ONE_FOLD_NOTE
    return out


def run(data, config, output_dir, dataset_name, conditions=None,
        c02_results_dir=None):
    """Exp 8: Complexity matching produces better-calibrated uncertainty estimates.

    Per outer fold and per decided model: the 2 best single representations
    and the 2 best hybrids from a search on that fold's training rows. Each
    hybrid is rebuilt at its searched budget ('optimal'), at a smaller budget
    ('underfit') and at the summed width of the representations ('overfit').
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: UNCERTAINTY CALIBRATION AT OPTIMAL VS SUBOPTIMAL COMPLEXITY")
    print("=" * 70)

    # Domain-aware metric; the direction comes from here, never assumed.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    task_type = data['task_type']
    is_classification = (task_type == 'classification')

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data['groups']
    groups_arr = np.asarray(groups) if groups is not None else None
    augs = data.get('augmentations') or None
    domain = data.get('domain')
    # 2026-10-05 (handoff 2.4 step 14): the outer fold count comes from the
    # config only. 1 means one held-out split.
    n_folds = int(config['n_splits'])
    total_dim = sum(v.shape[1] for v in reps.values())

    all_conditions = list(conditions or ALL_CONDITIONS)

    # 2026-10-05: c02's result file is not read. Its recipe was chosen by a
    # search over every row (rule R2), and the per-fold search below chooses
    # the recipes instead (tracker C08c: "alignment pass replaces where
    # recipes come from").
    c02_note = None
    if c02_results_dir:
        c02_note = ('c02 results are not read since 2026-10-05: every recipe '
                    "comes from a search on each outer fold's training rows")
        print(f"  {c02_note}")

    # Models: the dataset's two decided models only (DECISIONS.md Step 1).
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_without_factory = {
        m: 'no model factory on this dataset (size guard or unknown name)'
        for m in models if m not in factories}
    for m, why in models_without_factory.items():
        print(f"  {m}: {why}; not fitted")

    # The budget list the underfit rule reads: the dataset's decided budgets.
    budget_list = sorted(int(b) for b in (
        config.get('budgets')
        or decided_search(dataset_name, 'general')['budgets']))

    # The model each model's 'mismatched' condition fits, or why it is not
    # run.
    other_model, other_model_missing = {}, {}
    for m in models:
        _others = [o for o in models if o != m]
        if not _others:
            other_model_missing[m] = (
                f"not run: {dataset_name} has one decided model ({m}), so "
                f"there is no other model to fit")
        elif _others[0] not in factories:
            other_model_missing[m] = (
                f"not run: the other decided model {_others[0]} has no model "
                f"factory on this dataset (size guard or unknown name)")
        else:
            other_model[m] = _others[0]
    if 'mismatched' in all_conditions:
        print(f"  mismatched: {MISMATCHED_RULE}")
        for m, why in other_model_missing.items():
            print(f"    {m}: {why}")

    folds = nested_cv_outer_folds(reps, labels, groups_arr, n_folds, SEED,
                                  task_type)
    print(f"  {len(folds)} outer fold(s) (n_splits={n_folds}); models {models}; "
          f"{len(reps)} representations, {total_dim} features in all")

    # The arguments fold_standard_selection hands fold_searches. Repeated in
    # the fold_searches call below so it reads the same in-process memo entry
    # instead of searching again.
    search_kw = dict(dataset_name=dataset_name, augmentations=augs,
                     task_type=task_type, metric=metric_name, domain=domain,
                     seed=SEED)

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        tr = np.asarray(tr)
        va = np.asarray(va)
        fold_rec = {'fold': int(k), 'n_train': int(len(tr)),
                    'n_val': int(len(va)),
                    'selection_scope': 'fold_train_rows', 'models': {}}
        per_fold.append(fold_rec)
        try:
            sel = fold_standard_selection(
                reps, labels, groups_arr, train_idx=tr, fold_i=k,
                n_outer_folds=n_folds, models=models, **search_kw)
            searches = fold_searches(
                reps, labels, groups_arr, train_idx=tr, fold_i=k,
                n_outer_folds=n_folds, tier='general', **search_kw)
        except Exception as e:
            fold_rec['error'] = f'{type(e).__name__}: {e}'
            print(f"  fold {k}: per-fold search failed - {fold_rec['error']}")
            continue

        tr_reps = {r: X[tr] for r, X in reps.items()}
        va_reps = {r: X[va] for r, X in reps.items()}
        y_tr, y_va = labels[tr], labels[va]
        g_tr = groups_arr[tr] if groups_arr is not None else None

        for m in models:
            s = sel.get(m)
            if s is None:
                fold_rec['models'][m] = None
                fold_rec.setdefault('model_errors', {})[m] = (
                    "the search on this fold's training rows failed")
                continue
            factory = factories.get(m)
            if factory is None:
                fold_rec['models'][m] = None
                fold_rec.setdefault('model_errors', {})[m] = (
                    models_without_factory[m])
                continue

            # ── the 2 best single representations, each at its own width ──
            base_rows = []
            for i, e in enumerate(s['base_detail']['selected'], 1):
                rep = e['rep']
                row = {'slot': f'base{i}', 'rep': rep,
                       'conditions': {}}
                c = {'budget_asked': None,
                     'n_features_built': int(reps[rep].shape[1]),
                     'train_row_score': e.get('score')}
                try:
                    res, why = _calibrate(
                        tr_reps[rep], va_reps[rep], y_tr, y_va, g_tr,
                        factory, k, metric_fn, is_classification)
                except Exception as ex:
                    res, why = None, f'{type(ex).__name__}: {ex}'
                c['heldout'] = res
                if why:
                    c['error'] = why
                    print(f"    fold {k} {m:8s} base{i}: {why}")
                else:
                    _print_fold_line(k, m, f'base{i}', 'single', metric_name,
                                     res, c['n_features_built'])
                row['conditions']['single'] = c
                base_rows.append(row)
            if not s['base_detail'].get('complete', True):
                fold_rec.setdefault('base_shortfall', {})[m] = (
                    s['base_detail'].get('shortfall_reason'))

            # ── the 2 best hybrids, each under optimal / underfit / overfit ──
            hyb_rows = []
            for i, e in enumerate(s['hybrids'], 1):
                bp = e['blueprint']
                aug_cond = e.get('aug_condition') or 'none'
                row = {'slot': f'hybrid{i}',
                       **{f: e.get(f) for f in HYBRID_FIELDS},
                       'aug_condition': aug_cond,
                       'conditions': {}}
                tr_augs = va_augs = None
                if aug_cond != 'none' and augs:
                    tr_augs = {a: X[tr] for a, X in augs.items()}
                    va_augs = {a: X[va] for a, X in augs.items()}
                uf_budget, uf_reason = _underfit_for(bp.budget, total_dim,
                                                     budget_list)
                cond_budget = {'optimal': bp.budget, 'underfit': uf_budget,
                               'overfit': total_dim, 'mismatched': total_dim}
                # The overfit-width build, kept so 'mismatched' fits the other
                # model on the same features.
                overfit_built = None
                for cond in all_conditions:
                    if cond == 'mismatched' and m not in other_model:
                        row['conditions'][cond] = {
                            'heldout': None,
                            'skipped_reason': other_model_missing[m]}
                        continue
                    fit_model = (other_model[m] if cond == 'mismatched'
                                 else m)
                    fit_factory = factories[fit_model]
                    if cond == 'underfit' and uf_budget is None:
                        row['conditions'][cond] = {
                            'heldout': None, 'skipped_reason': uf_reason}
                        print(f"    fold {k} {m:8s} hybrid{i} underfit "
                              f"skipped: {uf_reason}")
                        continue
                    budget = cond_budget[cond]
                    c = {'budget_asked': int(budget), 'fitted_model': fit_model}
                    if cond == 'optimal':
                        c['train_row_score'] = e.get('score_mean')
                    else:
                        c['train_row_score'], _why = _train_row_score_at(
                            searches.get(aug_cond), fit_model, e, budget)
                        if _why:
                            c['train_row_score_note'] = _why
                    try:
                        if cond == 'mismatched' and overfit_built is not None:
                            built = overfit_built
                        else:
                            # The hybrid is built for model m (its own
                            # feature scoring) under every condition,
                            # 'mismatched' included; only the fitted model
                            # changes there.
                            built = build_fold_hybrid(
                                tr_reps, va_reps, y_tr, bp,
                                augmentations_train=tr_augs,
                                augmentations_val=va_augs,
                                budget_override=(None if cond == 'optimal'
                                                 else int(budget)),
                                groups=g_tr,
                                # Dirty-run bug 1: the scores of THIS fold's
                                # search (measured on its training rows),
                                # never whole-dataset ones.
                                precomputed_alloc_scores=getattr(
                                    bp, 'rep_scores', None),
                                is_clf=is_classification,
                                **fold_hybrid_scoring_kwargs(
                                    bp, model=factory(), metric_fn=metric_fn,
                                    lower_is_better=lower_is_better))
                        if cond in ('overfit', 'mismatched'):
                            overfit_built = built
                        X_tr_h, X_va_h, fi = built
                    except Exception as ex:
                        c['heldout'] = None
                        c['error'] = f'{type(ex).__name__}: {ex}'
                        row['conditions'][cond] = c
                        print(f"    fold {k} {m:8s} hybrid{i} {cond}: build "
                              f"failed - {c['error']}")
                        continue
                    # ⛔ THE WIDTH BUILT, NOT THE BUDGET ASKED FOR. create_hybrid
                    # stops each representation at its last feature scoring
                    # above zero, so 'overfit' can be built on a fraction of
                    # the summed width. budget_asked is the number asked for;
                    # this is the number built.
                    c['n_features_built'] = int(np.asarray(X_tr_h).shape[1])
                    c['features_by_rep'] = {
                        r: int(len(info['selected_indices']))
                        for r, info in fi.items()
                        if isinstance(info, dict)
                        and 'selected_indices' in info}
                    if fi.get('rep_count_kept') is not None:
                        c['rep_count_kept'] = fi['rep_count_kept']
                    try:
                        res, why = _calibrate(
                            X_tr_h, X_va_h, y_tr, y_va, g_tr, fit_factory, k,
                            metric_fn, is_classification)
                    except Exception as ex:
                        res, why = None, f'{type(ex).__name__}: {ex}'
                    c['heldout'] = res
                    if why:
                        c['error'] = why
                        print(f"    fold {k} {m:8s} hybrid{i} {cond}: {why}")
                    else:
                        _print_fold_line(k, m, f'hybrid{i}', cond, metric_name,
                                         res, c['n_features_built'])
                    row['conditions'][cond] = c
                hyb_rows.append(row)
            if len(s['hybrids']) < 2:
                fold_rec.setdefault('hybrid_shortfall', {})[m] = (
                    f"the search on this fold's training rows gave "
                    f"{len(s['hybrids'])} hybrid(s) with distinct features")

            fold_rec['models'][m] = {
                'base': base_rows,
                'hybrids': hyb_rows,
                'hybrids_skipped_identical_features': [
                    h.get('label') for h in
                    s.get('hybrids_skipped_identical_features') or []],
            }

    summary = _summarise(per_fold)

    results = {
        'result_identity': C08_RESULT_IDENTITY,
        'selection_scope': 'fold_train_rows',
        'selection_rule': ('per outer fold, per decided model: 2 best single '
                           'representations + 2 best hybrids, ranked on the '
                           "fold's training rows (fold_standard_selection)"),
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'task_type': task_type,
        'models': models,
        'models_without_factory': models_without_factory,
        'n_outer_folds': n_folds,
        'n_outer_folds_built': len(folds),
        'representations': sorted(reps),
        'total_dim': total_dim,
        'budget_list': budget_list,
        'conditions_requested': all_conditions,
        'condition_budget_rules': {
            'single': "the representation's own width",
            'optimal': "the hybrid's searched budget",
            'underfit': UNDERFIT_BUDGET_RULE,
            'overfit': f'the summed width of the representations, {total_dim} '
                       f'features',
            'mismatched': (f'{MISMATCHED_RULE}; asks for {total_dim} '
                           f'features'),
        },
        'mismatched_rule': MISMATCHED_RULE,
        'mismatched_fitted_model': other_model,
        'mismatched_not_run_reason': other_model_missing,
        'c02_not_read_reason': c02_note,
        'per_fold': per_fold,
        'summary': summary,
    }

    # Summary table: mean over folds of the held-out numbers.
    print("\n  CALIBRATION SUMMARY (mean over held-out folds):")
    third = 'SetSize' if is_classification else 'MPIW'
    third_key = 'avg_set_size' if is_classification else 'mpiw'
    print(f"  {'model':<9} {'slot':<8} {'condition':<10} {metric_name:<9} "
          f"{'PICP':<7} {third:<8} {'cal_err':<8} {'features':<9} folds")
    for m, slots in summary.items():
        for slot, conds in slots.items():
            for cond, d in conds.items():
                a = d['avg']
                print(f"  {m:<9} {slot:<8} {cond:<10} "
                      f"{a.get('val_score', float('nan')):<9.4f} "
                      f"{a.get('picp', float('nan')):<7.3f} "
                      f"{a.get(third_key, float('nan')):<8.3f} "
                      f"{a.get('cal_error', float('nan')):<8.4f} "
                      f"{a.get('n_features_built', float('nan')):<9.0f} "
                      f"{d['n_folds_scored']}"
                      + (f" ({ONE_FOLD_NOTE})" if d.get('spread_note') else ''))

    return results


if __name__ == '__main__':
    from experiments.complexity.complexity_helpers import DATASET_REPS
    parser = create_experiment_parser('C08: Uncertainty Calibration',
                                      has_delta=True, has_augment=True)
    parser.add_argument('--c02-results', type=str, default=None,
                        help='Not read since 2026-10-05; kept so old command '
                             'lines still parse')
    parser.add_argument('--conditions', type=str, nargs='+', default=None,
                        choices=list(ALL_CONDITIONS),
                        help='Which conditions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    # 2026-10-05 (handoff 2.4 step 9): the dataset's pruned representations
    # (DATASET_REPS[ds]['general']), as run_all loads them, not a --reps preset.
    _general = (DATASET_REPS.get(args.dataset) or {}).get('general')
    if _general:
        data = load_data(args.dataset, rep_preset='all', rep_names=_general,
                         delta=getattr(args, 'delta', False),
                         augment=getattr(args, 'augment', False))
    else:
        data = load_data(args.dataset, args.reps,
                         delta=getattr(args, 'delta', False),
                         augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  conditions=args.conditions,
                  c02_results_dir=args.c02_results)
    save_experiment_results(results, args.output, 'c08_calibration', args.dataset)
