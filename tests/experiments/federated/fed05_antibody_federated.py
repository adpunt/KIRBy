#!/usr/bin/env python3
"""Fed05: Antibody Federated Extension.

Adapts FLuID's federated framework from small molecules to antibody sequences.
Uses ESM2 embedding clustering instead of Morgan FP clustering. Tests the same
core questions as fed01/02 in the antibody domain: does KIRBy's feature-level
fusion outperform prediction-level ensemble across heterogeneous antibody silos?

3 natural silos (row counts measured 2026-09-12, see SILO_DATASETS):
    A: Experimental DDG (ddg_combined, 518 mutations, kcal/mol)
    B: DMS binding (abagym_EGFR, 659 mutations, normalized score)
    C: Developability (flab_thermo_tresanco, 672 variants, Tm °C)

8 conditions, every one run per outer fold of each silo (2026-10-05):
    C1: Independent baselines per silo: the 2 best base representations and,
        at each of the silo's decided budgets, the 2 best hybrids, all chosen
        by a search on that fold's training rows of that silo.
    C2: Cross-silo importance transfer
    C3: Within-silo ESM2 clustering → importance stability across clusters
    C4: Noise heterogeneity — NOT RUN, recorded as such (decided 2026-10-06):
        the three silos share no representation.
    C5: Pooled binding (DDG + DMS, z-score normalized)
    C6: Rep diversity per silo (different rep subsets)
    C7: SKIPPED — FLuID quality-weighted ensemble is incompatible with the
        antibody silo structure here. Each silo has a different feature space
        (different shared-rep dimensionalities + a per-silo PCA basis via
        cap_dimensions), and labels are in incompatible ranges (kcal/mol vs
        normalized DMS score vs °C). Calling a teacher.predict on another
        silo's features either crashes or silently returns meaningless
        predictions, so we report an explicit skip rather than fake numbers.
    C8: SKIPPED — same reason as C7. The KNN distance still uses the target
        silo's ESM2 embedding, but the regression head was fit on a
        different feature space, so predictions are not comparable.

Alignment of 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.1, 2.4):
    - Models: svm and dnn (FED05_MODELS), the two decided models of all three
      silo datasets. run_all reads FED05_MODELS. rf and xgb are gone.
    - Every recipe comes from a search on the rows the condition trains on:
      per silo, per outer fold of that silo, on the silo's own decided grid
      under its own dataset name; the pooled condition (C5) searches the pooled
      training rows on the grid the pooled silos share; C6 searches the lead
      silo's training rows over its diversity representations only. Pruning
      off. No whole-dataset search, no fallback recipe.
    - Each fold is saved as it finishes (<dataset>__folds.partial.jsonl).

Usage:
    python fed05_antibody_federated.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hashlib
import time
import traceback
import numpy as np

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    top_base_reps, top_hybrids, fold_standard_selection, search_on_fold,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
)
from kirby.find_best_hybrid import BUILTIN_METRICS
from experiments.federated.federated_helpers import (
    compute_reg_metrics, importance_profile_transfer,
    extract_importance_profile, compare_importance_profiles,
    cluster_antibody_silos, federated_budgets,
    _get_federated_config,
    make_fed_regressor, fit_fed_regressor, fed_to_kirby_model,
    fed_outer_folds, fed_fold_selection, smoke_n_folds,
    FED_FOLD_SEED,
)
from kirby.hybrid import apply_feature_selection
from kirby.scaling import FeatureScaler
from kirby.result_cache import StaleResultDB


# ── Constants ────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = 'fed05_antibody_federated'

# Bumped whenever what a saved fold means changes, so a partial file written
# under the old design is moved aside instead of read back. 2026-10-05: per
# outer fold, svm + dnn, recipes from searches on each condition's own
# training rows (was one 80/20 split per seed, rf/xgb/dnn, whole-pool slots).
# '_scaled' (2026-10-05, later): every fit now scales features with
# FeatureScaler, base arms carry their rank slot, and C6's ensemble weights
# are cross-validated by group, so folds saved before that are not read back.
# 2026-10-06: C4 recorded as not run on every fold; abagym_EGFR grouped by
# mutated position (shared loader change), so its folds are different rows.
ALIGNMENT_ID = 'fed05_fold_train_rows_scaled_2026-10-06'

# The models fed05 fits, in the fed_* names (svm and dnn are the same in
# KIRBy's names). DECISIONS.md Step 1: ddg_combined svm+dnn, abagym_EGFR
# svm+dnn, flab_thermo_tresanco dnn+svm. federated_helpers.run_all reads this
# constant for fed05 instead of fed_models_for(<handed dataset>), because
# fed05 ignores the dataset it is handed (2026-10-05).
FED05_MODELS = ['svm', 'dnn']

# The metric fed05 reports first: spearman, higher is better. It is the
# decided metric of all three silo datasets (_resolve_dataset_metric, checked
# 2026-10-05), and the metric every per-fold search ranks by. rmse is
# recorded beside it. run_all stamped fed05 'rmse, lower is better' until
# 2026-10-05; the stamp now comes from this run's return value.
REPORTED_METRIC = 'spearman'
REPORTED_LOWER_IS_BETTER = False
_HELD_OUT_KEY = 'spearman_rho'   # compute_reg_metrics' name for it

# The three antibody sets fed05 loads, whatever dataset name it is handed.
#
# The deep-mutational-scanning silo was abagym_HER2 until 2026-09-12. That
# name is not in the antibody dataset table in tests/antibody_baseline_survey.py
# — it was cut on 2026-07-09 for holding only 146 mutations — so every fed05
# run raised on it, loaded two silos instead of three, and the pooled-binding
# condition reported 'need >= 2 binding silos, got 1' every single time.
#
# abagym_EGFR is the replacement: same deep-mutational-scanning shape, in the
# table, and 659 mutations against ddg_combined's 518 and
# flab_thermo_tresanco's 672, so the three silos are within a quarter of each
# other in size rather than one being a fifth of the others. The two other
# live AbAgym sets are abagym_VEGF at 418 and abagym_Ang2 at 397; EGFR is the
# closest in size to the other two silos. Row counts measured 2026-09-12.
SILO_DATASETS = {
    'ddg_exp': 'ddg_combined',
    'dms': 'abagym_EGFR',
    'develop': 'flab_thermo_tresanco',
}

# Rep subsets for Condition 6 (rep diversity per silo). Each group is cut to
# the lead silo's own representations (DATASET_REPS[ds]['general']); a name
# not in that list is not loaded and drops out of its group.
REP_DIVERSITY_SILOS = {
    'structural': ['aac', 'saprot', 'esm2'],
    'sequence_lm': ['ablang2', 'antiberty', 'prot_t5'],
    'cdr_focused': ['prot_t5_cdr', 'esm2_midlayers', 'igt5'],
}

K_INTRA = 3   # sub-clusters per silo for Condition 3

# C5 pools the two binding silos.
C5_BINDING_SILOS = ('ddg_exp', 'dms')

# C3's per-cluster feature budget. Adelaide's choice (open, 2026-10-05):
#   'recipe'        — each hybrid recipe's own searched budget (the silo's
#                     decided budgets: ddg_combined 100-500, abagym_EGFR
#                     250-2000, flab_thermo_tresanco 50-2000 features). R3:
#                     every hybrid build uses the decided budgets.
#   'small_cluster' — the rule before 2026-10-05, min(30, cluster rows // 3)
#                     features, below every silo's decided range. A cluster
#                     of about 150 training rows got 30 features.
C3_CLUSTER_BUDGET_RULE = 'recipe'

# The metric the pooled search (C5) ranks by. The pooled labels are
# z-scored per silo, so a rank correlation is the one number that does not
# depend on which silo's scale a row came from. Same choice the old C4 made
# for its silo scores.
POOLED_METRIC = 'spearman'

# Per-fold selections, memoised in this process. run_all calls run() six
# times (2 models x 3 seeds); the selection depends on neither, so each
# fold's searches and identical-feature checks are run once per process.
_SELECTION_MEMO = {}


# ── Small helpers ────────────────────────────────────────────────────────────

def _rows_id(idx):
    """A short hash of a row-number array: two different row sets never share
    a memo entry."""
    a = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def _metrics(y_true, y_pred):
    """compute_reg_metrics without its p-values (no significance tests,
    Adelaide). spearman_rho is None when the prediction is constant."""
    m = compute_reg_metrics(y_true, y_pred)
    return {k: v for k, v in m.items() if not k.endswith('_p')}


def _fit_score(fed_model, X_tr, y_tr, X_te, y_te, seed):
    """Fit the run's model on X_tr and score it on X_te.

    The features are scaled on the training rows with FeatureScaler first
    (answer 17: continuous features standardised, 0/1 features passed
    through). The per-fold search that chose every base representation and
    hybrid scored each model after that same scaler (find_best_hybrid's
    _evaluate_single), and svm (SVRScaledTarget) and dnn (DNNRegressor)
    standardise only the labels, not the features. Until 2026-10-05 fed05
    fitted them on raw features, so a held-out score came from a different
    model setup than the training-row score recorded beside it.

    Returns (metrics, model). The model is a Pipeline of the fitted scaler and
    the fitted regressor, so it takes RAW features, and clone() of it (C6's
    quality ensemble cross-validates each teacher) refits both steps on the
    rows it is given."""
    from sklearn.pipeline import Pipeline
    sc = FeatureScaler().fit(X_tr)
    reg = fit_fed_regressor(fed_model, sc.transform(X_tr), y_tr, seed=seed)
    model = Pipeline([('scale', sc), ('model', reg)])
    return _metrics(y_te, model.predict(X_te)), model


def _quality_ensemble_grouped(models, train_sets, test_sets, groups_tr):
    """FLuID's quality-weighted regression ensemble, with the weights
    cross-validated by sequence-identity group.

    Each teacher's weight is 1 / its cross-validated rmse on the lead silo's
    fold training rows, normalised to sum to 1; the prediction is the weighted
    mean of the teachers' held-out predictions. Same rule as
    federated_helpers.predict_fluid_quality_ensemble_regression, which also
    cross-validates by group since 2026-10-06. Here the folds are GroupKFold
    over the training rows' groups, up to 5 (2026-10-05).

    Returns (predictions, weights, n_cv_folds), or raises ValueError when the
    training rows hold fewer than 2 groups (no grouped split exists)."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    g = np.asarray(groups_tr)
    n_groups = len(np.unique(g))
    if n_groups < 2:
        raise ValueError(f'{n_groups} sequence-identity group(s) in the '
                         f'training rows; a grouped cross-validation needs 2')
    n_cv = min(5, n_groups)
    preds, inv = {}, {}
    for name, model in models.items():
        X_tr, y_tr = train_sets[name]
        preds[name] = np.asarray(model.predict(test_sets[name]), dtype=float)
        y_cv = cross_val_predict(model, X_tr, y_tr, groups=g,
                                 cv=GroupKFold(n_splits=n_cv))
        rmse = float(np.sqrt(np.mean((np.asarray(y_tr) - y_cv) ** 2)))
        inv[name] = 1.0 / max(rmse, 1e-6)
    total = sum(inv.values())
    weights = {n: v / total for n, v in inv.items()}
    y_ens = sum(preds[n] * weights[n] for n in preds)
    return y_ens, weights, n_cv


def _fail(failures, *, condition, dataset, fold, model, error, **extra):
    """One failure record: which condition, dataset, fold and model, and why.
    The arm it belonged to is skipped, never filled with another recipe."""
    rec = {'condition': condition, 'dataset': dataset, 'fold': int(fold),
           'model': model, 'error': str(error)}
    rec.update(extra)
    failures.append(rec)
    print(f"    FAILED {condition} [{dataset}, fold {fold}, {model}]: {error}")
    return rec


def _recipe_record(entry):
    """What one selected hybrid is, for the result file: allocation method,
    importance method, budget in features, internal evaluator, representation
    count, augmentation condition, and its training-row score (the search's
    mean over the inner folds of this outer fold's training rows)."""
    bp = entry.get('blueprint')
    return {
        'label': entry.get('label'),
        'rank': entry.get('rank'),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'budget_features': (int(bp.budget) if bp is not None
                             and getattr(bp, 'budget', None) is not None
                             else None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
        'rep_count': getattr(bp, 'rep_count', None),
        'representations_searched': list(getattr(bp, 'reps', None) or []),
        'aug_condition': entry.get('aug_condition', 'none'),
        'train_score': entry.get('score_mean'),
        'train_score_metric': getattr(bp, 'metric', None),
        'feature_signature_error': entry.get('signature_error'),
    }


def _bp_metric(bp):
    """(metric_fn, lower_is_better) of the metric the recipe was searched
    under, so create_hybrid's accept/reject runs on the search's own metric."""
    fn, lib = BUILTIN_METRICS[getattr(bp, 'metric', REPORTED_METRIC)]
    return fn, lib


def _hybrid_arm(fed_model, seed, reps, labels, groups, tr, te, entry):
    """Rebuild one selected hybrid on the training rows tr, fit the run's
    model on it, score on te. Returns (record, feat_info).

    build_fold_hybrid rebuilds the recipe in full: its budget, allocation
    method, importance method, internal evaluator ('match' resolved by the
    run's own model instance), and its representation count, ranked by the
    blueprint's rep_scores — which come from the per-fold search on tr, never
    from a whole-dataset search (dirty-run bug 1)."""
    bp = entry['blueprint']
    tr_reps = {r: np.asarray(a)[tr] for r, a in reps.items()}
    te_reps = {r: np.asarray(a)[te] for r, a in reps.items()}
    y = np.asarray(labels)
    metric_fn, lib = _bp_metric(bp)
    X_tr, X_te, fi = build_fold_hybrid(
        tr_reps, te_reps, y[tr], bp,
        groups=(np.asarray(groups)[tr] if groups is not None else None),
        **fold_hybrid_scoring_kwargs(
            bp, model=make_fed_regressor(fed_model, seed=seed),
            metric_fn=metric_fn, lower_is_better=lib))
    m, _ = _fit_score(fed_model, X_tr, y[tr], X_te, y[te], seed)
    rec = dict(_recipe_record(entry), held_out=m,
               n_features=int(X_tr.shape[1]))
    return rec, fi


def _base_arm(fed_model, seed, reps, labels, tr, te, rep, train_score, slot):
    """slot is 'base<rank>' by the representation's rank in the per-fold
    selection, not by where it falls among the arms that fitted: when the
    rank-1 fit fails, the rank-2 score stays under base2 (2026-10-05)."""
    y = np.asarray(labels)
    X = np.asarray(reps[rep])
    m, _ = _fit_score(fed_model, X[tr], y[tr], X[te], y[te], seed)
    return {'representation': rep, 'slot': slot, 'train_score': train_score,
            'held_out': m, 'n_features': int(X.shape[1])}


def _spread(vals):
    """Mean and spread over folds. One fold has no spread, and says so
    (handoff 2.4 step 14) instead of reporting a standard deviation of 0."""
    vals = [float(v) for v in vals
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not np.isnan(v)]
    if not vals:
        return {'mean': None, 'std': None, 'n_folds': 0}
    return {'mean': float(np.mean(vals)),
            'std': float(np.std(vals)) if len(vals) > 1 else None,
            'n_folds': len(vals),
            **({'note': 'one fold, no spread'} if len(vals) == 1 else {})}


def _fmt(agg):
    if agg.get('mean') is None:
        return 'n/a'
    if agg.get('std') is None:
        return f"{agg['mean']:.3f} (one fold, no spread)"
    return f"{agg['mean']:.3f} ± {agg['std']:.3f}"


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_silo(silo_name, ds_name, failures=None):
    """Load one antibody silo dataset. Returns data dict or None.

    A silo that fails is recorded in `failures` so the saved result can say
    which of the three actually contributed.
    """
    try:
        # Test-only: under the smoke flags load cheap sequence reps so the silo
        # loads locally without server-only antibody foundation models (ESM2,
        # ProtT5, AntiBERTy, SaProt, AbLang2).
        #
        # Two of these four are POSITION-ENCODED (onehot, blosum) and a deep
        # mutational scan breaks them. MEASURED on abagym_EGFR, 2026-09-18,
        # under the grouping of that date (abagym is grouped by mutated
        # position since 2026-10-06; not re-measured):
        # 659 single mutants of one antibody give 5 sequence-identity groups
        # that are 5 disjoint sets of mutated positions, so under the
        # group-aware split onehot has 580 columns varying in the training
        # rows, 115 in the held-out rows and ZERO varying in both — the model
        # predicts one constant value and the rank correlation is undefined.
        # 'aac' (amino-acid composition, 40D) and 'paratope' (8D,
        # sequence-derived, no dependencies) are not position-indexed and
        # both survive that split, so the silo has two usable
        # representations rather than none. rep_names must be passed
        # EXPLICITLY — leaving it None makes load_data auto-resolve to the full
        # 18-rep DATASET_REPS 'general' list (which includes the foundation
        # models). The rep-diversity condition (C6) self-skips when those
        # foundation reps are absent. On the server rep_preset='all' with no
        # rep_names loads the dataset's own pool, DATASET_REPS[ds]['general']
        # (_shared._load_data_impl), not FRAMEWORK_TASKS. So on tresanco the
        # sequence_lm group of C6 (ablang2, antiberty, prot_t5) is absent:
        # those were cut from its pool on 2026-08-20 (checked 2026-10-05).
        from experiments.federated.federated_helpers import _smoke_active
        if _smoke_active():
            sd = load_data(
                ds_name,
                rep_names=['aac', 'paratope', 'onehot', 'blosum'])
        else:
            sd = load_data(ds_name, rep_preset='all')
        # ⛔ A REPRESENTATION MATRIX WITH THE WRONG ROW COUNT IS REFUSED.
        # The dirty run found ddg_combined's local cache holding 429 of 534
        # molecules (DIRTY_RUN_RESULTS.md bug 9). load_data already raises on
        # a matrix whose rows differ from the label count; this repeats the
        # check after any local width cap so a mismatched silo is recorded
        # with the representation named and never trained on (2026-10-05).
        n = len(sd['labels'])
        bad = {r: int(np.asarray(a).shape[0]) for r, a in sd['reps'].items()
               if np.asarray(a).shape[0] != n}
        if bad:
            raise ValueError(
                f'representation row counts {bad} differ from the {n} labels '
                f'(stale representation cache?); silo refused')
        print(f"    {silo_name} ({ds_name}): {n} samples, "
              f"{len(sd['reps'])} reps")
        return sd
    except Exception as e:
        print(f"    {silo_name} ({ds_name}): FAILED — {e}")
        if failures is not None:
            failures[silo_name] = f'{ds_name}: {e}'
        return None


# ── Selections (all on training rows of one outer fold) ─────────────────────

def _silo_selection(sname, sd, fold_i, n_folds, train_idx):
    """fed_fold_selection for one silo and fold: per decided model, the 2 best
    base representations and, at each of the silo's decided budgets, the 2
    best hybrids with distinct selected features. ONE general search on this
    fold's training rows, on the silo's own decided grid and under its own
    dataset name (R3); pruning off; augmentation 'none' (Step 8 cut it on all
    three silos, so the silo's added-feature sets are not passed)."""
    ds = SILO_DATASETS[sname]
    key = ('silo', ds, int(fold_i), int(n_folds), _rows_id(train_idx))
    if key not in _SELECTION_MEMO:
        try:
            sel = fed_fold_selection(
                sd['reps'], sd['labels'], sd.get('groups'),
                train_idx=train_idx, fold_i=fold_i, n_folds=n_folds,
                dataset_name=ds, budgets=federated_budgets(ds),
                models=[fed_to_kirby_model(m) for m in FED05_MODELS],
                task_type=sd['task_type'], metric='auto')
        except StaleResultDB:
            # Not recorded and carried past: a result store written under an
            # older column list stops the run (result_cache.StaleResultDB;
            # 2026-10-05). Caught as one more failure, the fold was saved as
            # finished with its arms missing.
            raise
        except Exception as e:
            traceback.print_exc()
            sel = {fed_to_kirby_model(m): None for m in FED05_MODELS}
            sel['_errors'] = [{'error': f'{type(e).__name__}: {e}',
                               'traceback': traceback.format_exc()}]
        _SELECTION_MEMO[key] = sel
    return _SELECTION_MEMO[key]


def _grid_intersection(ds_names, budgets):
    """The search axes every one of ds_names keeps (decided_grid), with the
    given budgets. For a search on rows pooled from several silo datasets: a
    method or evaluator one silo's record cut is not searched on rows that
    include that silo. Raises when an axis has no shared member."""
    from experiments.decided_grid import decided_search, AXES
    grids = [decided_search(d, 'general') for d in ds_names]
    out = {'tier': 'general'}
    for axis in AXES:
        if axis == 'budgets':
            continue
        kept = [x for x in grids[0][axis] if all(x in g[axis] for g in grids)]
        if not kept:
            raise ValueError(f'{ds_names}: no {axis} is decided on all of '
                             f'them ({[g[axis] for g in grids]})')
        out[axis] = kept
    out['budgets'] = [int(b) for b in budgets]
    speeds = {g['speed'] for g in grids}
    out['speed'] = speeds.pop() if len(speeds) == 1 else 'full'
    return out


def _pooled_selection(tag, reps, labels, groups, train_idx, fold_i, n_folds,
                      ds_names, budgets, models):
    """Per model: 2 best base representations and, per budget, 2 best hybrids
    with distinct selected features, from ONE search on the pooled training
    rows (train_idx into the pooled arrays).

    The pooled rows are no single dataset, so there is no decided grid of
    their own. The search runs on the axes every pooled silo keeps
    (_grid_intersection), under the first silo's real dataset name with the
    pool named in condition_suffix (R3: decorations go in the suffix), so it
    is filed apart from that silo's own per-fold search. Pruning off.
    """
    from experiments._shared import _distinct_feature_hybrids
    key = ('pooled', tag, int(fold_i), int(n_folds), _rows_id(train_idx),
           hashlib.sha1(np.ascontiguousarray(
               np.asarray(labels, dtype=np.float64)).tobytes()).hexdigest()[:16])
    if key in _SELECTION_MEMO:
        return _SELECTION_MEMO[key]
    out = {'_errors': [], '_grid': None}
    try:
        grid = _grid_intersection(ds_names, budgets)
        out['_grid'] = {k: v for k, v in grid.items()}
        searches = search_on_fold(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, dataset_name=ds_names[0],
            task_type='regression', metric=POOLED_METRIC,
            seed=FED_FOLD_SEED, condition_suffix=f'fed05_{tag}',
            fixed_grid=grid)
        out['_errors'] = list(getattr(searches, 'failures', None) or [])
        srch = searches.get('none')
    except StaleResultDB:
        raise   # stops the run; see _silo_selection (2026-10-05)
    except Exception as e:
        traceback.print_exc()
        out['_errors'].append({'error': f'{type(e).__name__}: {e}'})
        srch = None
    for m in models:
        if srch is None or (out['_grid'] and m not in out['_grid']['models']):
            out[m] = None
            continue
        ctx = f'fed05 {tag} fold {fold_i} model {m}'
        base = top_base_reps(srch, m, n=STANDARD_N_BASE, context=ctx,
                             require_full=False)
        entry = {'base': [e['rep'] for e in base['selected']],
                 'base_detail': base, 'by_budget': {},
                 'search_lower_is_better': bool(srch.lower_is_better),
                 'search_metric': getattr(srch, 'metric_name', None)}
        for b in budgets:
            th = top_hybrids(srch, m, n=STANDARD_N_HYBRIDS + 4, budget=int(b),
                             context=ctx, require_full=False)
            pooled = [dict(e, aug_condition='none') for e in th['selected']]
            if len(pooled) > 1:
                chosen, skipped = _distinct_feature_hybrids(
                    pooled, STANDARD_N_HYBRIDS, reps=reps, labels=labels,
                    groups=groups, train_idx=train_idx, augmentations=None,
                    model=m, dataset_name=ds_names[0],
                    task_type='regression', metric=POOLED_METRIC,
                    domain=None, lower_is_better=bool(srch.lower_is_better))
            else:
                chosen, skipped = pooled[:STANDARD_N_HYBRIDS], []
            entry['by_budget'][int(b)] = {
                'hybrids': chosen,
                'hybrids_skipped_identical_features': skipped}
        out[m] = entry
    _SELECTION_MEMO[key] = out
    return out


def _diversity_selection(lead_ds, div_reps, labels, groups, train_idx,
                         fold_i, n_folds, budgets, kmodel):
    """C6's recipe: a search on the lead silo's training rows over the
    diversity representations only, on the lead silo's decided grid and real
    dataset name. Filed under condition_suffix 'fed05_c6_<reps>' because the
    representation list is not part of the stored-result key. Returns
    {budget: fold_standard_selection entry for kmodel or None}."""
    tag = 'fed05_c6_' + '+'.join(sorted(div_reps))
    out = {}
    for b in budgets:
        key = ('c6', lead_ds, tag, int(fold_i), int(n_folds),
               _rows_id(train_idx), int(b))
        if key not in _SELECTION_MEMO:
            try:
                _SELECTION_MEMO[key] = fold_standard_selection(
                    div_reps, labels, groups, train_idx=train_idx,
                    fold_i=fold_i, n_outer_folds=n_folds,
                    dataset_name=lead_ds,
                    models=[fed_to_kirby_model(m) for m in FED05_MODELS],
                    task_type='regression', metric='auto',
                    seed=FED_FOLD_SEED, condition_suffix=tag, budget=int(b))
            except StaleResultDB:
                raise   # stops the run; see _silo_selection (2026-10-05)
            except Exception as e:
                traceback.print_exc()
                _SELECTION_MEMO[key] = {'_error': f'{type(e).__name__}: {e}'}
        sel = _SELECTION_MEMO[key]
        out[int(b)] = (sel.get(kmodel) if '_error' not in sel
                       else {'_error': sel['_error']})
    return out


def _zscore_with(train_y, y):
    """y z-scored with train_y's mean and spread; zeros when train_y is
    constant. Held-out rows are scaled with the TRAINING rows' numbers."""
    mu, sd = float(np.mean(train_y)), float(np.std(train_y))
    if sd <= 1e-12:
        return np.zeros_like(np.asarray(y, dtype=float))
    return (np.asarray(y, dtype=float) - mu) / sd


def _pool_rows(silo_data, silo_folds, names, fold_i, rep_names,
               include_test=True):
    """Stack fold fold_i's rows of several silos into one set of arrays.

    Labels are z-scored per silo with that silo's training-row mean and
    spread (different endpoints). Group ids are factorised per silo and
    shifted by an accumulating offset so two silos' ids never collide; a silo
    without groups gives every row its own group. Returns (reps, labels,
    groups, train_idx, test_idx, source) with train/test row numbers into the
    pooled arrays."""
    reps = {r: [] for r in rep_names}
    ys, gs, src, tr_all, te_all = [], [], [], [], []
    offset_rows, offset_groups = 0, 0
    for s in names:
        sd = silo_data[s]
        tr, te = silo_folds[s][fold_i]
        rows = np.concatenate([tr, te]) if include_test else np.asarray(tr)
        y_all = np.asarray(sd['labels'], dtype=float)
        ys.append(_zscore_with(y_all[tr], y_all[rows]))
        for r in rep_names:
            reps[r].append(np.asarray(sd['reps'][r])[rows])
        g = sd.get('groups')
        if g is not None:
            codes = np.unique(np.asarray(g)[rows], return_inverse=True)[1]
            gs.append(codes.astype(np.int64) + offset_groups)
            offset_groups = int(gs[-1].max()) + 1 if len(rows) else offset_groups
        else:
            gs.append(np.arange(len(rows)) + offset_groups)
            offset_groups += len(rows)
        tr_all.append(np.arange(offset_rows, offset_rows + len(tr)))
        if include_test:
            te_all.append(np.arange(offset_rows + len(tr),
                                    offset_rows + len(rows)))
        src.extend([s] * len(rows))
        offset_rows += len(rows)
    return ({r: np.vstack(v) for r, v in reps.items()},
            np.concatenate(ys), np.concatenate(gs),
            np.concatenate(tr_all),
            (np.concatenate(te_all) if te_all else np.array([], dtype=int)),
            src)


# ── One fold ─────────────────────────────────────────────────────────────────

def _run_fold(*, fold_i, n_folds, fed_model, fed_seed, silo_data, silo_folds,
              silo_budgets, cross_budgets):
    """Every condition on outer fold fold_i of every silo, for one model.
    Returns the fold record (JSON-plain, saved as soon as it returns)."""
    kmodel = fed_to_kirby_model(fed_model)
    failures = []
    # Every per-fold search that raised or came back empty, by condition and
    # dataset. A fold with any entry here is written to the result file but
    # NOT saved to the fold checkpoint, so the next submission searches again
    # instead of reading back a fold with its arms missing (2026-10-05).
    search_failures = []
    silo_names = sorted(silo_data)
    rec = {'fold': int(fold_i), 'model': kmodel, 'fed_model': fed_model,
           'fed_seed': int(fed_seed), 'selection_scope': 'fold_train_rows',
           'metric_name': REPORTED_METRIC,
           'lower_is_better': REPORTED_LOWER_IS_BETTER,
           'silos': {}, 'failures': failures,
           'search_failures': search_failures}

    # ── Selections: one search per silo on this fold's training rows ──────
    sel_by_silo = {}
    for s in silo_names:
        sd = silo_data[s]
        ds = SILO_DATASETS[s]
        tr, te = silo_folds[s][fold_i]
        print(f"\n  [{s} / {ds}] fold {fold_i}: {len(tr)} train, "
              f"{len(te)} held out — per-fold search")
        sel = _silo_selection(s, sd, fold_i, n_folds, tr)
        msel = sel.get(kmodel)
        srec = {'dataset': ds, 'n_train': int(len(tr)), 'n_test': int(len(te)),
                'budgets': list(silo_budgets[s]),
                'search_errors': convert_numpy(sel.get('_errors') or [])}
        if sel.get('_errors') or msel is None:
            search_failures.append({'condition': 'selection', 'dataset': ds})
        if msel is None:
            _fail(failures, condition='selection', dataset=ds, fold=fold_i,
                  model=kmodel, error='the per-fold search returned nothing '
                  'for this model; every arm of this silo is skipped',
                  search_errors=srec['search_errors'])
            srec['selection'] = None
        else:
            sel_by_silo[s] = msel
            srec['selection'] = {
                'search_metric': msel.get('search_metric'),
                'search_lower_is_better': msel.get('search_lower_is_better'),
                'base': list(msel.get('base') or []),
                'base_shortfall': (msel.get('base_detail') or {}).get(
                    'shortfall_reason'),
                'hybrids': {
                    int(b): [_recipe_record(e) for e in bb['hybrids']]
                    for b, bb in msel['by_budget'].items()},
                'hybrids_skipped_identical_features': {
                    int(b): convert_numpy(
                        bb.get('hybrids_skipped_identical_features') or [])
                    for b, bb in msel['by_budget'].items()},
            }
        rec['silos'][s] = srec

    # ── C1: independent baselines per silo ────────────────────────────────
    print(f"\n--- C1 — independent baselines (fold {fold_i}) ---")
    c1 = {}
    built_fi = {}      # (silo, budget, hybrid index) -> feat_info, for C2
    for s in silo_names:
        msel = sel_by_silo.get(s)
        if msel is None:
            continue
        sd, ds = silo_data[s], SILO_DATASETS[s]
        tr, te = silo_folds[s][fold_i]
        out = {'base': [], 'hybrids': {}}
        scores = {e['rep']: e.get('score')
                  for e in (msel.get('base_detail') or {}).get('selected', [])}
        for i, r in enumerate(msel.get('base') or []):
            try:
                arm = _base_arm(fed_model, fed_seed, sd['reps'], sd['labels'],
                                tr, te, r, scores.get(r), f'base{i + 1}')
                out['base'].append(arm)
                print(f"    {s} base {r}: {REPORTED_METRIC} "
                      f"{arm['held_out'].get(_HELD_OUT_KEY)}")
            except Exception as e:
                _fail(failures, condition='c1_base', dataset=ds, fold=fold_i,
                      model=kmodel, error=f'{type(e).__name__}: {e}',
                      representation=r, slot=f'base{i + 1}')
        for b, bb in sorted(msel['by_budget'].items()):
            arms = []
            for j, entry in enumerate(bb['hybrids']):
                try:
                    arm, fi = _hybrid_arm(fed_model, fed_seed, sd['reps'],
                                          sd['labels'], sd.get('groups'),
                                          tr, te, entry)
                    arm['slot'] = f'hybrid{j + 1}'
                    arms.append(arm)
                    built_fi[(s, int(b), j)] = fi
                    print(f"    {s} hybrid{j + 1} @ {b} features: "
                          f"{REPORTED_METRIC} "
                          f"{arm['held_out'].get(_HELD_OUT_KEY)}")
                except Exception as e:
                    _fail(failures, condition='c1_hybrid', dataset=ds,
                          fold=fold_i, model=kmodel,
                          error=f'{type(e).__name__}: {e}', budget=int(b),
                          slot=f'hybrid{j + 1}',
                          traceback=traceback.format_exc())
            out['hybrids'][int(b)] = arms
        c1[s] = out
    rec['c1_independent'] = c1

    # ── C2: cross-silo importance transfer ────────────────────────────────
    # The source's hybrid, built on the source silo's fold training rows at
    # each cross-silo budget, hands its selected feature numbers to the
    # target, which fits on its own fold training rows and scores on its own
    # held-out rows. Only representations both silos have can carry
    # features (importance_profile_transfer skips the rest). The target's own
    # hybrid at the same budget and slot is recorded beside it, raw.
    print(f"\n--- C2 — cross-silo importance transfer (fold {fold_i}) ---")
    c2 = {}
    for src in silo_names:
        for tgt in silo_names:
            if src == tgt or tgt not in sel_by_silo:
                continue
            sd_t, ds_t = silo_data[tgt], SILO_DATASETS[tgt]
            tr, te = silo_folds[tgt][fold_i]
            y_t = np.asarray(sd_t['labels'])
            rep_tr = {r: np.asarray(a)[tr] for r, a in sd_t['reps'].items()}
            rep_te = {r: np.asarray(a)[te] for r, a in sd_t['reps'].items()}
            for b in cross_budgets:
                for j in range(STANDARD_N_HYBRIDS):
                    key = f"{src}→{tgt}@b{b}#hybrid{j + 1}"
                    src_fi = built_fi.get((src, int(b), j))
                    if src_fi is None:
                        continue   # the source arm failed or was not chosen;
                        # its failure is already recorded under C1
                    try:
                        X_tr, tfi, stats = importance_profile_transfer(
                            src_fi, rep_tr, y_t[tr], budget=int(b))
                        if X_tr is None:
                            c2[key] = {'not_run': 'the source selected no '
                                       'feature in a representation the '
                                       'target has', 'transfer_stats': stats}
                            continue
                        X_te = apply_feature_selection(rep_te, tfi)
                        m, _ = _fit_score(fed_model, X_tr, y_t[tr], X_te,
                                          y_t[te], fed_seed)
                        own = [a for a in c1.get(tgt, {}).get(
                            'hybrids', {}).get(int(b), [])
                               if a.get('slot') == f'hybrid{j + 1}']
                        c2[key] = {
                            'source': src, 'target': tgt,
                            'budget_features': int(b),
                            'slot': f'hybrid{j + 1}',
                            'held_out': m, 'transfer_stats': stats,
                            'n_features': int(X_tr.shape[1]),
                            'target_own_hybrid_held_out': (
                                own[0]['held_out'] if own else None)}
                        print(f"    {key}: {REPORTED_METRIC} "
                              f"{m.get(_HELD_OUT_KEY)}")
                    except Exception as e:
                        _fail(failures, condition='c2_transfer', dataset=ds_t,
                              fold=fold_i, model=kmodel,
                              error=f'{type(e).__name__}: {e}', pair=key)
    rec['c2_importance_transfer'] = c2

    # ── C3: within-silo clustering ────────────────────────────────────────
    # The silo's fold training rows are clustered (k-means on one embedding,
    # train rows only); each hybrid recipe of that silo is rebuilt on every
    # cluster and the per-representation importance profiles are compared
    # across clusters. A cluster whose build fails is recorded (it was
    # dropped silently by `except Exception: pass` until 2026-10-05).
    print(f"\n--- C3 — within-silo clustering (fold {fold_i}) ---")
    c3 = {}
    for s in silo_names:
        msel = sel_by_silo.get(s)
        if msel is None:
            continue
        sd, ds = silo_data[s], SILO_DATASETS[s]
        tr, _ = silo_folds[s][fold_i]
        y_tr = np.asarray(sd['labels'])[tr]
        g = sd.get('groups')
        g_tr = np.asarray(g)[tr] if g is not None else None
        rep_tr = {r: np.asarray(a)[tr] for r, a in sd['reps'].items()}
        try:
            clusters = cluster_antibody_silos(rep_tr, k=K_INTRA, seed=fed_seed)
        except Exception as e:
            _fail(failures, condition='c3_clustering', dataset=ds,
                  fold=fold_i, model=kmodel, error=f'{type(e).__name__}: {e}')
            continue
        viable = {c: idx for c, idx in clusters.items() if len(idx) >= 10}
        out = {'n_sub_clusters': len(viable),
               'cluster_sizes': {int(c): int(len(i))
                                 for c, i in clusters.items()},
               'budget_rule': C3_CLUSTER_BUDGET_RULE, 'by_recipe': {}}
        if len(viable) < 2:
            out['not_run'] = (f'{len(viable)} sub-cluster(s) of at least 10 '
                              f'training rows; 2 are needed')
            c3[s] = out
            continue
        for b, bb in sorted(msel['by_budget'].items()):
            for j, entry in enumerate(bb['hybrids']):
                bp = entry['blueprint']
                metric_fn, lib = _bp_metric(bp)
                profiles, cluster_fail = {}, {}
                for c, idx in viable.items():
                    c_reps = {r: a[idx] for r, a in rep_tr.items()}
                    budget_override = (None if C3_CLUSTER_BUDGET_RULE == 'recipe'
                                       else min(30, len(idx) // 3))
                    if budget_override is not None and budget_override < 5:
                        cluster_fail[int(c)] = (f'{len(idx)} rows give a '
                                                f'budget under 5 features')
                        continue
                    try:
                        _, _, fi_c = build_fold_hybrid(
                            c_reps, {r: a[:1] for r, a in c_reps.items()},
                            y_tr[idx], bp,
                            budget_override=budget_override,
                            groups=(g_tr[idx] if g_tr is not None else None),
                            **fold_hybrid_scoring_kwargs(
                                bp, model=make_fed_regressor(fed_model,
                                                             seed=fed_seed),
                                metric_fn=metric_fn, lower_is_better=lib))
                        profiles[int(c)] = extract_importance_profile(fi_c)
                    except Exception as e:
                        cluster_fail[int(c)] = f'{type(e).__name__}: {e}'
                        _fail(failures, condition='c3_cluster', dataset=ds,
                              fold=fold_i, model=kmodel,
                              error=f'{type(e).__name__}: {e}',
                              cluster=int(c), budget=int(b),
                              slot=f'hybrid{j + 1}')
                mean_sp, pairwise = compare_importance_profiles(profiles)
                out['by_recipe'][f'b{int(b)}#hybrid{j + 1}'] = {
                    'recipe': _recipe_record(entry),
                    # None when undefined: fewer than two clusters built, or
                    # fewer than three representations to correlate over.
                    'profile_stability': mean_sp,
                    'pairwise': convert_numpy(pairwise),
                    'profiles': convert_numpy(profiles),
                    'clusters_failed': cluster_fail,
                }
                _sp = 'undefined' if mean_sp is None else f'{mean_sp:.3f}'
                print(f"    {s} hybrid{j + 1} @ {b} features: "
                      f"{len(profiles)}/{len(viable)} clusters built, "
                      f"stability {_sp}")
        c3[s] = out
    rec['c3_within_silo_clustering'] = c3

    # ── C4: noise heterogeneity ───────────────────────────────────────────
    # DECIDED 2026-10-06 (Adelaide delegated): C4 is recorded as not run.
    # federated_feature_selection builds ONE feature set over the pooled rows
    # of every silo, so it can only use representations every silo has, and
    # the three silos' pools share no representation by name (DATASET_REPS,
    # checked 2026-10-05: ddg_combined 8, abagym_EGFR 10,
    # flab_thermo_tresanco 10, three-way intersection empty). The shared
    # representations are still counted and written beside the reason.
    print(f"\n--- C4 — noise heterogeneity (fold {fold_i}) ---")
    shared_all = sorted(set.intersection(
        *[set(silo_data[s]['reps']) for s in silo_names]))
    shared_pairs = {
        f'{a}+{b}': sorted(set(silo_data[a]['reps']) & set(silo_data[b]['reps']))
        for i, a in enumerate(silo_names) for b in silo_names[i + 1:]}
    c4 = {'shared_representations': shared_all,
          'shared_by_pair': shared_pairs,
          'not_run': ('the antibody silos share no representation; C4 '
                      'builds one feature set over every silo\'s rows and '
                      'needs representations all silos have (decided '
                      '2026-10-06)')}
    print(f"  SKIP: {c4['not_run']}")
    rec['c4_noise_heterogeneity'] = convert_numpy(c4)

    # ── C5: pooled binding ────────────────────────────────────────────────
    # The two binding silos' fold training rows are pooled (labels z-scored
    # per silo with the training rows' mean and spread) and the model is
    # scored on the two silos' held-out rows of the same fold. Until
    # 2026-10-05 C5 drew a second 80/20 split inside the pooled training rows
    # and scored on its 20%, with a recipe searched on all of one silo's
    # training rows, so the recipe had seen C5's test labels; now the recipe
    # comes from a search on the pooled training rows only. Representations:
    # the ones these two silos share (it filtered on all three silos' common
    # set before, which is empty, so C5 never ran).
    print(f"\n--- C5 — pooled binding (fold {fold_i}) ---")
    bind = [s for s in C5_BINDING_SILOS if s in silo_data]
    c5 = {'silos_pooled': bind}
    if len(bind) < 2:
        c5['not_run'] = f'need 2 binding silos, {len(bind)} loaded'
    else:
        c5_reps = sorted(set.intersection(
            *[set(silo_data[s]['reps']) for s in bind]))
        c5['representations'] = c5_reps
        c5_ds = [SILO_DATASETS[s] for s in bind]
        if len(c5_reps) < 2:
            c5['not_run'] = (f'{bind} share {len(c5_reps)} representation(s); '
                             f'a hybrid needs 2')
        else:
            try:
                p_reps, p_y, p_g, p_tr, p_te, _ = _pool_rows(
                    silo_data, silo_folds, bind, fold_i, c5_reps)
                c5['n_train'], c5['n_test'] = int(len(p_tr)), int(len(p_te))
                psel = _pooled_selection(
                    'c5_' + '+'.join(bind), p_reps, p_y, p_g, p_tr, fold_i,
                    n_folds, c5_ds, cross_budgets, [kmodel])
                c5['search_errors'] = convert_numpy(psel.get('_errors') or [])
                c5['search_grid'] = convert_numpy(psel.get('_grid'))
                msel = psel.get(kmodel)
                if psel.get('_errors'):
                    search_failures.append({'condition': 'c5_selection',
                                            'dataset': '+'.join(c5_ds)})
                if msel is None:
                    _fail(failures, condition='c5_selection',
                          dataset='+'.join(c5_ds), fold=fold_i, model=kmodel,
                          error='the pooled search returned nothing')
                else:
                    scores = {e['rep']: e.get('score') for e in
                              (msel.get('base_detail') or {}).get('selected', [])}
                    c5['base'], c5['hybrids'] = [], {}
                    for i, r in enumerate(msel['base']):
                        try:
                            c5['base'].append(_base_arm(
                                fed_model, fed_seed, p_reps, p_y, p_tr, p_te,
                                r, scores.get(r), f'base{i + 1}'))
                        except Exception as e:
                            _fail(failures, condition='c5_base',
                                  dataset='+'.join(c5_ds), fold=fold_i,
                                  model=kmodel, representation=r,
                                  slot=f'base{i + 1}',
                                  error=f'{type(e).__name__}: {e}')
                    for b, bb in sorted(msel['by_budget'].items()):
                        arms = []
                        for j, entry in enumerate(bb['hybrids']):
                            try:
                                arm, _ = _hybrid_arm(fed_model, fed_seed,
                                                     p_reps, p_y, p_g, p_tr,
                                                     p_te, entry)
                                arm['slot'] = f'hybrid{j + 1}'
                                arms.append(arm)
                                print(f"    pooled hybrid{j + 1} @ {b} "
                                      f"features: {REPORTED_METRIC} "
                                      f"{arm['held_out'].get(_HELD_OUT_KEY)}")
                            except Exception as e:
                                _fail(failures, condition='c5_hybrid',
                                      dataset='+'.join(c5_ds), fold=fold_i,
                                      model=kmodel, budget=int(b),
                                      slot=f'hybrid{j + 1}',
                                      error=f'{type(e).__name__}: {e}')
                        c5['hybrids'][int(b)] = arms
            except Exception as e:
                _fail(failures, condition='c5', dataset='+'.join(c5_ds),
                      fold=fold_i, model=kmodel,
                      error=f'{type(e).__name__}: {e}',
                      traceback=traceback.format_exc())
    if 'not_run' in c5:
        print(f"  SKIP: {c5['not_run']}")
    rec['c5_pooled_binding'] = convert_numpy(c5)

    # ── C6: representation diversity within one silo ──────────────────────
    # The silo with the most rows is split by representation groups; each
    # group's concatenation is fitted alone, the FLuID quality-weighted
    # ensemble combines them, and a KIRBy hybrid over all the groups'
    # representations is built from a search on this silo's fold training
    # rows over those representations only. Until 2026-10-05 that hybrid used
    # the recipe of whichever silo came first in dict order (ddg_combined),
    # searched on other rows.
    print(f"\n--- C6 — representation diversity (fold {fold_i}) ---")
    lead = max(silo_names, key=lambda s: (len(silo_data[s]['labels']), s))
    sd_l, lead_ds = silo_data[lead], SILO_DATASETS[lead]
    div = {}
    for name, rl in REP_DIVERSITY_SILOS.items():
        avail = [r for r in rl if r in sd_l['reps']]
        if avail:
            div[name] = avail
    c6 = {'lead_silo': lead, 'dataset': lead_ds, 'diversity_silos': div}
    if len(div) < 2:
        c6['not_run'] = f'{len(div)} representation group(s) available; 2 needed'
        print(f"  SKIP: {c6['not_run']}")
    else:
        tr, te = silo_folds[lead][fold_i]
        y = np.asarray(sd_l['labels'], dtype=float)
        y_tr, y_te = y[tr], y[te]
        names = sorted(div)
        teachers, tq, teq = {}, {}, {}
        c6['baselines'] = {}
        try:
            for dname in names:
                X_tr = np.hstack([np.asarray(sd_l['reps'][r])[tr]
                                  for r in div[dname]])
                X_te = np.hstack([np.asarray(sd_l['reps'][r])[te]
                                  for r in div[dname]])
                m, reg = _fit_score(fed_model, X_tr, y_tr, X_te, y_te, fed_seed)
                c6['baselines'][dname] = dict(m, reps=div[dname])
                teachers[dname] = reg
                tq[dname] = (X_tr, y_tr)
                teq[dname] = X_te
        except Exception as e:
            _fail(failures, condition='c6_baselines', dataset=lead_ds,
                  fold=fold_i, model=kmodel, error=f'{type(e).__name__}: {e}')
        # The ensemble's weights are cross-validated by sequence-identity
        # group on the lead silo's fold training rows (_quality_ensemble_
        # grouped). Until 2026-10-05 they came from
        # predict_fluid_quality_ensemble_regression's shuffled KFold. Run only
        # when every group's baseline fitted, so the ensemble always combines
        # the same groups.
        if teachers and len(teachers) == len(names):
            try:
                y_ens, w, n_cv = _quality_ensemble_grouped(
                    teachers, tq, teq, np.asarray(sd_l['groups'])[tr])
                c6['quality_ensemble'] = dict(
                    _metrics(y_te, y_ens), rmse_weights=w,
                    weight_cv='GroupKFold by sequence-identity group',
                    weight_cv_folds=int(n_cv))
            except Exception as e:
                c6['quality_ensemble_not_run'] = f'{type(e).__name__}: {e}'
                _fail(failures, condition='c6_quality_ensemble',
                      dataset=lead_ds, fold=fold_i, model=kmodel,
                      error=f'{type(e).__name__}: {e}')
        div_reps = {r: sd_l['reps'][r] for d in names for r in div[d]}
        c6['kirby_hybrid'] = {}
        if len(div_reps) < 2:
            c6['kirby_hybrid_not_run'] = (f'{len(div_reps)} representation; '
                                          f'a hybrid needs 2')
        else:
            dsel = _diversity_selection(lead_ds, div_reps, y, sd_l.get('groups'),
                                        tr, fold_i, n_folds,
                                        silo_budgets[lead], kmodel)
            for b, msel in sorted(dsel.items()):
                if msel is None or '_error' in msel:
                    search_failures.append({'condition': 'c6_selection',
                                            'dataset': lead_ds,
                                            'budget': int(b)})
                    _fail(failures, condition='c6_selection', dataset=lead_ds,
                          fold=fold_i, model=kmodel, budget=int(b),
                          error=(msel or {}).get(
                              '_error', 'the per-fold search over the '
                              'diversity representations returned nothing'))
                    continue
                arms = []
                for j, entry in enumerate(msel['hybrids']):
                    try:
                        arm, fi_h = _hybrid_arm(fed_model, fed_seed, div_reps,
                                                y, sd_l.get('groups'), tr, te,
                                                entry)
                        arm['slot'] = f'hybrid{j + 1}'
                        arm['allocation'] = convert_numpy(
                            fi_h.get('allocation', {})
                            if isinstance(fi_h, dict) else {})
                        arms.append(arm)
                        print(f"    KIRBy hybrid{j + 1} @ {b} features: "
                              f"{REPORTED_METRIC} "
                              f"{arm['held_out'].get(_HELD_OUT_KEY)}")
                    except Exception as e:
                        _fail(failures, condition='c6_hybrid', dataset=lead_ds,
                              fold=fold_i, model=kmodel, budget=int(b),
                              slot=f'hybrid{j + 1}',
                              error=f'{type(e).__name__}: {e}')
                c6['kirby_hybrid'][int(b)] = arms
    rec['c6_rep_diversity'] = convert_numpy(c6)

    return convert_numpy(rec)


# ── Summary over folds ───────────────────────────────────────────────────────

def _summarise(fold_recs):
    """Held-out spearman per arm, mean and spread over folds (or 'one fold,
    no spread'). Arms are named by slot, not by representation, because the 2
    best base representations can differ from fold to fold. Per dataset;
    nothing is averaged across datasets."""
    acc = {}

    def _add(key, held):
        if isinstance(held, dict):
            acc.setdefault(key, []).append(held.get(_HELD_OUT_KEY))

    for fr in fold_recs:
        for s, out in (fr.get('c1_independent') or {}).items():
            for i, arm in enumerate(out.get('base') or []):
                _add(f"c1/{s}/{arm.get('slot') or f'base{i + 1}'}",
                     arm.get('held_out'))
            for b, arms in (out.get('hybrids') or {}).items():
                for arm in arms:
                    _add(f"c1/{s}/{arm.get('slot')}@{b}", arm.get('held_out'))
        for k, v in (fr.get('c2_importance_transfer') or {}).items():
            _add(f'c2/{k}', v.get('held_out'))
        c5 = fr.get('c5_pooled_binding') or {}
        for i, arm in enumerate(c5.get('base') or []):
            _add(f"c5/{arm.get('slot') or f'base{i + 1}'}",
                 arm.get('held_out'))
        for b, arms in (c5.get('hybrids') or {}).items():
            for arm in arms:
                _add(f"c5/{arm.get('slot')}@{b}", arm.get('held_out'))
        c6 = fr.get('c6_rep_diversity') or {}
        for d, m in (c6.get('baselines') or {}).items():
            _add(f'c6/baseline_{d}', m)
        _add('c6/quality_ensemble', c6.get('quality_ensemble'))
        for b, arms in (c6.get('kirby_hybrid') or {}).items():
            for arm in arms:
                _add(f"c6/{arm.get('slot')}@{b}", arm.get('held_out'))
    return {k: dict(_spread(v), per_fold=v) for k, v in sorted(acc.items())}


# ── The run ──────────────────────────────────────────────────────────────────

def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed05: Antibody federated extension, one (model, seed) call."""
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity, fingerprint_of)
    print("\n" + "=" * 70)
    print("EXPERIMENT FED05: ANTIBODY FEDERATED EXTENSION")
    print("=" * 70)

    # One of FED05_MODELS (run_all); no 'rf' default (R1, 2026-10-05).
    fed_model = config.get('fed_model') or FED05_MODELS[0]
    if fed_model not in FED05_MODELS:
        raise ValueError(f'fed05: model {fed_model!r} is not one of the silo '
                         f'datasets\' decided models {FED05_MODELS}')
    fed_seed = int(config.get('fed_seed', 0))
    kmodel = fed_to_kirby_model(fed_model)
    # The outer fold count is ONE setting, config['n_splits'] (handoff 2.4
    # step 14), defaulting to the federated config's; 1 gives a single
    # grouped held-out split.
    n_folds = smoke_n_folds(int(config.get('n_splits')
                                or _get_federated_config()['n_splits']))

    # ── Load silos ────────────────────────────────────────────────────────
    print("\n--- Phase 1: Load silos ---")
    silo_data, silo_load_failures = {}, {}
    for silo_name, ds_name in SILO_DATASETS.items():
        sd = _load_silo(silo_name, ds_name, silo_load_failures)
        if sd is not None:
            silo_data[silo_name] = sd
    if silo_load_failures:
        # The saved result names three silos. Without this a run on two of them
        # reads afterwards as a three-silo result.
        print(f"  WARNING: {len(silo_load_failures)} silo(s) did not load: "
              f"{', '.join(sorted(silo_load_failures))}")

    # ── Outer folds per silo ──────────────────────────────────────────────
    # fed_outer_folds: nested_cv_outer_folds with seed 42, the same rows as
    # each silo's per-fold search store. They do not depend on fed_seed
    # (until 2026-10-05 each seed drew its own 80/20 split, so the spread over
    # seeds mixed the model's seed with which antibodies were held out).
    # Silo i's fold k and silo j's fold k are unrelated rows; conditions that
    # pool silos pool fold k's training rows of each.
    silo_folds = {}
    for s, sd in list(silo_data.items()):
        try:
            if sd.get('groups') is None:
                raise ValueError('no sequence-identity groups; a random split '
                                 'is not allowed')
            silo_folds[s] = list(fed_outer_folds(
                sd['reps'], sd['labels'], sd['groups'], n_folds,
                sd['task_type']))
        except Exception as e:
            print(f"  {s}: cannot draw {n_folds} outer folds — {e}")
            silo_load_failures[s] = f'outer folds failed: {e}'
            silo_data.pop(s)
    if len(silo_data) < 2:
        return {'error': f'Only {len(silo_data)} silos usable, need >= 2',
                'silo_load_failures': silo_load_failures,
                'metric_name': REPORTED_METRIC,
                'lower_is_better': REPORTED_LOWER_IS_BETTER}
    n_run = min(len(f) for f in silo_folds.values())

    # Budgets. Per-silo conditions use that silo's decided general budgets
    # (C1, C3, C6). Conditions that cross silos (C2, C5) use the budgets
    # decided on all three silo datasets: ddg_combined 100-500,
    # abagym_EGFR 250-2000 and flab_thermo_tresanco 50-2000 features share
    # 250 and 500 (decided_grid.json, read 2026-10-05).
    silo_budgets = {s: [int(b) for b in federated_budgets(SILO_DATASETS[s])]
                    for s in silo_data}
    cross_budgets = sorted(set.intersection(
        *[set(int(b) for b in federated_budgets(d))
          for d in SILO_DATASETS.values()]))
    print(f"  model {kmodel}, seed {fed_seed}, {n_run} outer fold(s)")
    print(f"  budgets per silo (features): {silo_budgets}; "
          f"cross-silo: {cross_budgets}")

    # Written after every fold (dirty-run bug 12): one unit per (model, seed,
    # fold) in <dataset>__folds.partial.jsonl. The identity carries each
    # silo's fingerprint (rows, label and group hashes, representations), so
    # a change in any silo's data moves the old file aside; the stand-in
    # `data` run_all hands fed05 has no rows and could not.
    _cfg_ident = {k: v for k, v in config.items()
                  if k not in ('fed_model', 'fed_seed')}
    fold_ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name,
                           suffix='folds'),
        run_identity(EXPERIMENT_DIR, dataset_name, None, _cfg_ident,
                     alignment=ALIGNMENT_ID, n_folds=n_folds,
                     c3_rule=C3_CLUSTER_BUDGET_RULE,
                     silos={s: fingerprint_of(sd)
                            for s, sd in sorted(silo_data.items())}))

    fold_recs = []
    for fold_i in range(n_run):
        print(f"\n========== fold {fold_i + 1}/{n_run}, model {kmodel}, "
              f"seed {fed_seed} ==========")
        key = dict(fed_model=str(fed_model), fed_seed=fed_seed,
                   fold=int(fold_i))
        if fold_ck.done(**key):
            fr = fold_ck.get(**key)
            print(f"  [partial] fold {fold_i}: read back from {fold_ck.path}")
        else:
            t0 = time.time()
            fr = _run_fold(fold_i=fold_i, n_folds=n_folds, fed_model=fed_model,
                           fed_seed=fed_seed, silo_data=silo_data,
                           silo_folds=silo_folds, silo_budgets=silo_budgets,
                           cross_budgets=cross_budgets)
            fr['seconds'] = round(time.time() - t0, 1)
            if fr.get('search_failures'):
                # In the result file, not in the fold checkpoint: the next
                # submission searches this fold again (Checkpoint contract: a
                # failed unit is retried, not read back as finished).
                print(f"  fold {fold_i}: {len(fr['search_failures'])} per-fold "
                      f"search(es) failed; fold not saved to {fold_ck.path}, "
                      f"it is retried on the next submission")
            else:
                fold_ck.put(fr, **key)
        fold_recs.append(fr)

    summary = _summarise(fold_recs)
    _skip_msg = (
        "NOT IMPLEMENTED: cross-silo predict() requires a shared feature "
        "space and shared label scale. Antibody silos here have neither "
        "(per-silo PCA basis via cap_dimensions; labels in kcal/mol vs "
        "normalized DMS vs °C). See module docstring."
    )
    final = {
        'experiment': EXPERIMENT_DIR,
        'dataset': dataset_name,
        'metric_name': REPORTED_METRIC,
        'lower_is_better': REPORTED_LOWER_IS_BETTER,
        'selection_scope': 'fold_train_rows',
        'alignment': ALIGNMENT_ID,
        'config': {
            'silos': dict(SILO_DATASETS),
            'silos_loaded': sorted(silo_data),
            'silo_load_failures': silo_load_failures,
            'silo_rows': {s: int(len(sd['labels']))
                          for s, sd in silo_data.items()},
            'silo_representations': {s: sorted(sd['reps'])
                                     for s, sd in silo_data.items()},
            'model': kmodel, 'fed_model': fed_model, 'fed_seed': fed_seed,
            'n_outer_folds': n_folds, 'fold_seed': FED_FOLD_SEED,
            'silo_budgets_features': silo_budgets,
            'cross_silo_budgets_features': cross_budgets,
            'k_intra': K_INTRA,
            'c3_cluster_budget_rule': C3_CLUSTER_BUDGET_RULE,
            'pooled_search_metric': POOLED_METRIC,
        },
        'folds': fold_recs,
        'summary_held_out_spearman': summary,
        'failures': [f for fr in fold_recs for f in (fr.get('failures') or [])],
        'c7_fluid_quality': {'error': _skip_msg, 'skipped': True},
        'c8_fluid_similarity': {'error': _skip_msg, 'skipped': True},
    }

    print(f"\n{'─' * 70}")
    print(f"  held-out {REPORTED_METRIC}, {kmodel}, seed {fed_seed}:")
    for k, v in summary.items():
        print(f"    {k:<48} {_fmt(v)}")
    if final['failures']:
        print(f"  {len(final['failures'])} failure(s) recorded under 'failures'")
    return final


# Each fold is saved as it finishes (_run_one's fold checkpoint). The
# whole-call resumable_run used until 2026-10-05 is gone: its identity
# fingerprinted the stand-in data run_all passes fed05 (no rows), so a change
# in the silo data never triggered a re-run. A finished call now reloads the
# silos and reads every fold back. --force-rerun / KIRBY_FORCE_RERUN=1
# discards the fold file.
def run(data, config, output_dir, dataset_name, **kwargs):
    return _run_one(data, config, output_dir, dataset_name, **kwargs)


if __name__ == '__main__':
    # Fed05 loads its own silo data — the dataset arg is for naming only
    import argparse
    parser = argparse.ArgumentParser(
        description='Fed05: Antibody Federated Extension')
    parser.add_argument('--output', type=str,
                        default='tests/results/federated')
    args = parser.parse_args()

    # Pass placeholder data — run() loads silos internally. One model of
    # FED05_MODELS; run_all is the path that runs both.
    data = {'reps': {}, 'labels': np.array([]), 'domain': 'antibody'}
    cfg = dict(_get_federated_config(), fed_model=FED05_MODELS[0])
    results = run(data, cfg, args.output, 'antibody_federated')
    save_experiment_results(
        results, args.output, EXPERIMENT_DIR, 'antibody',
        metric_name=REPORTED_METRIC, lower_is_better=REPORTED_LOWER_IS_BETTER)
