#!/usr/bin/env python3
"""Fed02: Representation Diversity as Federated Silos.

Tests whether KIRBy's feature-level hybrid outperforms FLuID's prediction-level
ensemble when silos contribute different molecular representations (not different
data). Each silo sees the same molecules but through a different computational
lens (e.g., ECFP4, Mordred, MolFormer, PDV, ChemBERTa).

6 conditions:
    1. Best single silo — best individual rep baseline
    2. FLuID similarity ensemble — 8-NN weighted mean of regressor predictions
    3. FLuID quality ensemble — inverse-RMSE weighted mean
    4. KIRBy hybrid — feature-level fusion across all 5 silo reps
    5. KIRBy top-tier hybrid — feature-level fusion on top 3 reps only
    6. FLuID→KIRBy pipeline — intra-silo clustering + cross-silo fusion (novel)

Since 2026-10-05 (handoff 2.5 group B, Adelaide "yes"): the comparison is
kept, run on the dataset's two decided models (one run() per model, from
federated_helpers.run_all), and every hybrid recipe comes from a search on
that outer fold's training rows (federated_helpers.fed_fold_selection). Per
fold and per budget, each model's 2 best hybrids (hybrid1, hybrid2) are each
built the three ways (C4 all silos, C5 top three, C6 clustered). The fold
search's 2 best base representations are fitted as their own arm beside C1.

Datasets: lhasa_herg (classification), logd, caco2 (regression) -- the
launchers' FEDERATED_DATASETS.

Usage:
    python fed02_rep_diversity_silos.py --dataset lhasa_herg --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hashlib
import traceback
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy, cv_split, get_domain_metric,
)
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments.federated.federated_helpers import (
    compute_reg_metrics,
    compute_clf_metrics,
    predict_fluid_similarity_ensemble_regression,
    predict_fluid_quality_ensemble_regression,
    predict_fluid_similarity_ensemble_classification,
    predict_fluid_quality_ensemble_classification,
    fluid_within_kirby_across,
    smoke_n_folds, federated_budgets,
    fit_fed_model,
    fed_to_kirby_model,
    fed_models_for,
    fed_outer_folds,
    fed_fold_selection,
    FED_FOLD_SEED,
    make_fed_classifier,
    make_fed_regressor,
)


EXPERIMENT_DIR = 'fed02_rep_diversity_silos'

# ── Constants ────────────────────────────────────────────────────────────────

# 5 silo representations, preferred. Each is taken only when it is one of the
# dataset's pruned representations (DATASET_REPS[ds]['general'], handoff 2.1);
# otherwise a substitute from _SILO_FALLBACKS, from the same list.
SILO_REPS = {
    'A': 'ecfp4',        # structural fingerprints, 2048D
    'B': 'mordred',       # physicochemical descriptors, ~1600D
    'C': 'molformer',     # foundation model embeddings, 768D
    'D': 'pdv',           # RDKit descriptors, 200D
    'E': 'chemberta',     # learned SMILES embeddings, 768D
}

# Fallback for datasets where chemberta is unavailable
SILO_REPS_FALLBACK = {
    'caco2': {**SILO_REPS, 'E': 'avalon'},
}

K_INTRA = 3   # sub-clusters per silo for Condition 6
# Folds of the inner split C5's three representations are chosen on. Three, not
# five: this runs inside every outer fold, so the cost is
# 5 representations x INNER_RANK_FOLDS fits per outer fold. An inner split of
# the training rows only; the outer folds come from fed_outer_folds.
INNER_RANK_FOLDS = 3

# ── Per-fold hybrids (2026-10-05, handoff 2.1 R2/R4) ─────────────────────────
# Per model, per outer fold, per budget: the 2 best hybrids of the search run
# on that fold's training rows (fed_fold_selection; a second hybrid whose
# selected features equal the first's is already skipped there). Replaces the
# 2026-10-02 set of general1, general2 and distance1, which was read from the
# whole-dataset searches. No distance hybrid: fed02 measures no neighbours
# (handoff 2.3 item 1). hybrid1 writes the keys this file always wrote.
HYBRID_SLOTS = ('hybrid1', 'hybrid2')

# Whether the recipe's representation count (2, 3, 5 or all) is applied to
# the silo representations each condition is built on. DECIDED 2026-10-06
# (Adelaide delegated): False. 'All silos' (C4) is built on all 5 silo
# representations, C5 on its 3 and C6 on all 5, whatever representation
# count the recipe won with. With True, a recipe that won on 2 of the
# dataset's 10 pruned representations was rebuilt on 2 of the 5 silo
# representations, so 'all silos' could mean 2.
APPLY_REP_COUNT_TO_SILOS = False

# Continuous features are standardised before every model fit, 0/1 features
# are passed through (kirby.scaling.FeatureScaler; Adelaide's answer 17,
# 2026-10-05). The per-fold search scores every recipe and every base
# representation this way, so the fits here now match what was ranked. Until
# 2026-10-05 every fit in this file took the raw features, which leaves svm
# and dnn -- the decided models on caco2 and logd -- on unscaled descriptors.
SCALE_FEATURES_BEFORE_FIT = True

# Bumped whenever what one fold or one run computes changes, so a partial file
# written under the old layout is not read back as this one (2026-10-05).
# 2026-10-06: every condition built on all its silo representations
# (APPLY_REP_COUNT_TO_SILOS False); C3's weights cross-validated by scaffold
# group (was KFold(shuffle=True)).
RESULT_LAYOUT = 'fold_train_rows_2026_10_06'

# compute_reg_metrics also returns these two; they are dropped from every
# metrics dict this file writes (2026-10-05: no p-values in any result).
_P_VALUE_KEYS = ('spearman_p', 'pearson_p')

# ── The rest of each recipe (2026-10-02) ────────────────────────────────────
# Until 2026-10-02 this file rebuilt each hybrid from four of its recipe's
# settings: allocation method, importance method, internal evaluator and
# augmentation condition. Everything else the search had chosen took
# create_hybrid's defaults, so general1 and general2 could build identical
# features. The owner's decision: build with the whole recipe. The names and
# defaults below are the ones _shared._create_hybrid_from_blueprint uses, which
# is how the search's own rebuild (build_hybrid_from_entry) calls create_hybrid.
#
# Privacy check, done for this file: every one of these settings is a number
# or a name chosen by the same stored search the four settings already came
# from. None carries a row, a label or a feature, so passing them on shares
# nothing across silos that the four settings did not already share.
def _recipe_kwargs(bp, objective=None):
    """create_hybrid keyword arguments for every recipe setting beyond the
    allocation method, importance method, internal evaluator, augmentation
    condition and budget. {} when there is no stored recipe."""
    if bp is None:
        return {}
    kw = {
        'step_size': int(getattr(bp, 'step_size', 5) or 5),
        'augmentation_margin': float(
            getattr(bp, 'augmentation_margin', 0.0) or 0.0),
        'augmentation_fraction': getattr(bp, 'augmentation_fraction', None),
        'candidate_ranking': getattr(bp, 'candidate_ranking',
                                     'residual_correlation'),
        'augmentation_threshold': getattr(bp, 'augmentation_threshold', None),
        'augmentation_budget': int(
            getattr(bp, 'augmentation_budget', 20) or 20),
        'aug_max_candidates': int(getattr(bp, 'aug_max_candidates', 50) or 50),
        'aug_per_source_budget': getattr(bp, 'aug_per_source_budget', None),
    }
    if objective is not None:
        kw['objective'] = objective
    return kw

def _fit(fed_model, X_tr, y_tr, *, seed, task_type):
    """fit_fed_model on standardised features (SCALE_FEATURES_BEFORE_FIT).

    Returns a model that takes the RAW features: a Pipeline of the scaler
    fitted on X_tr and the fitted model. predict / predict_proba scale first,
    and clone() of it (FLuID's quality ensemble cross-validates each teacher
    with cross_val_predict) refits both steps on the rows it is given.
    """
    if not SCALE_FEATURES_BEFORE_FIT:
        return fit_fed_model(fed_model, X_tr, y_tr, seed=seed,
                             task_type=task_type)
    from sklearn.pipeline import Pipeline
    from kirby.scaling import FeatureScaler
    sc = FeatureScaler().fit(X_tr)
    mdl = fit_fed_model(fed_model, sc.transform(X_tr), y_tr, seed=seed,
                        task_type=task_type)
    return Pipeline([('scale', sc), ('model', mdl)])


def _resolve_evaluator(internal_evaluator, kirby_model):
    """'match' is the run's own model (KIRBy name), never rf (2026-10-05)."""
    return kirby_model if internal_evaluator == 'match' else internal_evaluator


def _restrict_rep_count(rep_dict, bp, context):
    """The silo representations one recipe is rebuilt on, cut to the
    recipe's representation count by the per-fold search's own scores
    (APPLY_REP_COUNT_TO_SILOS). Returns (rep_dict kept, note or None)."""
    if not APPLY_REP_COUNT_TO_SILOS:
        return rep_dict, None
    from kirby.find_best_hybrid import restrict_to_rep_count
    return restrict_to_rep_count(rep_dict, bp, context=context)


def _recipe_summary(entry, kirby_model, search_metric=None,
                    search_lower_is_better=None):
    """What the result file records about one hybrid of the per-fold search:
    its recipe and its score on this fold's training rows (the mean over the
    search's inner folds, on the dataset's pruned representations), with the
    search's metric name and direction. That metric can differ from the one
    the held-out scores are read in: on lhasa_herg the search scores auc and
    the held-out rows are scored in mcc (2026-10-05)."""
    bp = entry.get('blueprint')
    return {
        'label': entry.get('label'),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'budget': (int(bp.budget) if bp is not None else entry.get('budget')),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
        'internal_evaluator_resolved': _resolve_evaluator(
            getattr(bp, 'internal_evaluator', None), kirby_model),
        'rep_count': getattr(bp, 'rep_count', None),
        'aug_condition': entry.get('aug_condition', 'none'),
        'step_size': getattr(bp, 'step_size', None),
        'train_rows_score': entry.get('score_mean'),
        'train_rows_metric': search_metric,
        'train_rows_lower_is_better': search_lower_is_better,
        'train_rows_score_basis': ('mean over the per-fold search\'s inner '
                                   'folds of this outer fold\'s training rows, '
                                   'on the dataset\'s pruned representations'),
    }


# One fed_fold_selection per (dataset, outer fold, budgets, rows, labels) per
# process. run_all calls run() once per model and seed; the search itself is
# memoised in _shared.fold_searches and stored on disk, but the
# distinct-features check builds candidate hybrids, and that is not. Both
# models' selections come from the one call.
_FOLD_SELECTION_MEMO = {}


def _fold_selection(reps, labels, groups, *, train_idx, fold_i, n_folds,
                    dataset_name, budgets, task_type, augmentations):
    key = (dataset_name, int(n_folds), int(fold_i),
           tuple(int(b) for b in budgets),
           hashlib.sha1(np.asarray(train_idx).tobytes()).hexdigest(),
           hashlib.sha1(np.ascontiguousarray(labels).tobytes()).hexdigest(),
           tuple(sorted((k, int(v.shape[1])) for k, v in reps.items())),
           tuple(sorted(augmentations)) if augmentations else None)
    if key not in _FOLD_SELECTION_MEMO:
        _FOLD_SELECTION_MEMO[key] = fed_fold_selection(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_folds=n_folds, dataset_name=dataset_name, budgets=budgets,
            models=None, task_type=task_type, metric='auto',
            augmentations=augmentations)
    return _FOLD_SELECTION_MEMO[key]


def _mean_spread(vals):
    """(mean, std, note) over folds. One fold has no spread: std is None and
    the note says so, rather than a 0.0 that reads as a measurement."""
    vals = [float(v) for v in vals]
    if not vals:
        return float('nan'), float('nan'), None
    if len(vals) == 1:
        return vals[0], None, 'one fold, no spread'
    return float(np.mean(vals)), float(np.std(vals)), None


def _rank_reps_train_only(silo_letters, silo_reps, train_reps, y_train,
                          groups, fed_model, fed_seed, task_type, is_clf,
                          compute_metrics, n_inner_folds=INNER_RANK_FOLDS):
    """Score each silo representation inside the training rows only.

    Returns {silo_letter: mean inner score} — MCC for classification, RMSE for
    regression, the same two numbers C5 was previously ranked on, but measured
    on rows the outer fold's test set never touches.

    The inner split is group-aware; with too few distinct scaffold groups in
    the training rows it raises rather than falling back to a random split
    (feedback_no_random_splits.md).
    """
    if groups is None:
        raise ValueError(
            "_rank_reps_train_only: no groups for the training rows — "
            "ranking representations on a random inner split is forbidden.")
    n_unique = len(np.unique(groups))
    if n_unique < n_inner_folds:
        raise ValueError(
            f"_rank_reps_train_only: {n_unique} distinct scaffold groups in "
            f"the training rows but n_inner_folds={n_inner_folds}.")

    per_rep = {s: [] for s in silo_letters}
    inner = list(cv_split(groups, n_inner_folds,
                          np.zeros(len(y_train)), y_train, seed=fed_seed,
                          task_type=task_type))
    for in_tr, in_va in inner:
        for s in silo_letters:
            X = train_reps[silo_reps[s]]
            mdl = _fit(fed_model, X[in_tr], y_train[in_tr],
                       seed=fed_seed, task_type=task_type)
            y_hat = mdl.predict(X[in_va])
            if is_clf:
                y_prob = (mdl.predict_proba(X[in_va])[:, 1]
                          if hasattr(mdl, 'predict_proba') else None)
                m = compute_metrics(y_train[in_va], y_hat, y_prob=y_prob)
                per_rep[s].append(m['mcc'])
            else:
                m = compute_metrics(y_train[in_va], y_hat)
                per_rep[s].append(m['rmse'])
    return {s: float(np.mean(v)) for s, v in per_rep.items()}


# Substitutes tried, in order, when a silo's own representation is not in the
# dataset's pool. A substitute already taken by another silo is skipped: the
# whole experiment is five silos holding five DIFFERENT representations, and
# two silos on the same one is not a weaker version of that question, it is a
# different question.
_SILO_FALLBACKS = ['chembert', 'avalon', 'graph_kernel', 'mordred', 'pdv',
                   'atom_pair', 'top_torsion', 'mhg_gnn', 'grover', 'smited']


def _get_silo_reps(dataset_name, available_reps):
    """Resolve silo reps for a dataset, using fallbacks if needed.

    No representation is ever given to two silos. Until 2026-09-12 the
    duplicate check ran only on the fallback path, so a silo whose OWN
    representation was in the pool took it even when an earlier silo had
    already been given the same one as a substitute. On caco2 that put avalon
    in silo B (standing in for mordred, which caco2's pool lacks) and again in
    silo E (avalon is E's own representation there), so two of the five silos
    held identical features and the run answered a four-silo question while
    reporting five.
    """
    import warnings
    base = SILO_REPS_FALLBACK.get(dataset_name, SILO_REPS)
    resolved = {}
    # Own representation first, so a silo that has its own is never displaced
    # by another silo's substitute.
    deferred = []
    for silo_letter, rep_name in base.items():
        if rep_name in available_reps:
            resolved[silo_letter] = rep_name
        else:
            deferred.append((silo_letter, rep_name))
    for silo_letter, rep_name in deferred:
        for fallback in _SILO_FALLBACKS:
            if fallback in available_reps and fallback not in resolved.values():
                resolved[silo_letter] = fallback
                warnings.warn(
                    f"Silo {silo_letter}: '{rep_name}' unavailable, "
                    f"using fallback '{fallback}'")
                break
        else:
            warnings.warn(
                f"Silo {silo_letter}: '{rep_name}' unavailable and "
                f"no unused fallback found — silo dropped")
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError(
            f"_get_silo_reps({dataset_name}): two silos resolved to the same "
            f"representation — {resolved}. Every silo must hold a different "
            f"one or the experiment is not asking its own question.")
    return resolved

def _silo_pool(dataset_name, loaded):
    """The representations a silo may hold: the dataset's pruned list
    (DATASET_REPS[ds]['general']) that was loaded. Handoff 2.1, 2026-10-05:
    no representation outside that list. Before this the silos were drawn
    from whatever data['reps'] held, which for lhasa_herg was the 'all'
    preset of about 33."""
    general = (DATASET_REPS.get(dataset_name) or {}).get('general')
    if not general:
        raise ValueError(
            f"fed02: {dataset_name} has no DATASET_REPS[...]['general'] list; "
            f"the silo representations are drawn only from it.")
    return [r for r in general if r in loaded]


def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed02: Representation diversity as federated silos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED02: REPRESENTATION DIVERSITY AS FEDERATED SILOS")
    print("=" * 70)

    # The run's model is one of the dataset's two decided models (R1,
    # 2026-10-05). run_all sets config['fed_model'] from fed_models_for; a
    # direct call without one takes the first. Until 2026-10-05 the default
    # was 'rf', which no federated dataset decided.
    _decided_fed = fed_models_for(dataset_name)
    fed_model = config.get('fed_model') or _decided_fed[0]
    if fed_model not in _decided_fed:
        raise ValueError(
            f"fed02 {dataset_name}: model {fed_model!r} is not one of the "
            f"dataset's decided models {_decided_fed} (DECISIONS.md Step 1).")
    kirby_model = fed_to_kirby_model(fed_model)
    fed_seed = config.get('fed_seed', 0)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    _primary_key = 'mcc' if is_clf else 'rmse'

    # The metric create_hybrid is told to optimise (as before) -- not the
    # metric this file reports, which is _primary_key.
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # Augmentation only where DECISIONS.md Step 8 keeps a condition other
    # than 'none' (R3). It is cut on lhasa_herg, caco2 and logd, so on every
    # dataset fed02 runs on no added features enter any search or arm. Before
    # 2026-10-05 a budget with no stored recipe forced 'uncapped'
    # augmentation onto C4 and C5 (the hard-defaults path, now removed).
    from experiments.decided_grid import decided_aug_conditions
    _aug_kept = [c for c in decided_aug_conditions(dataset_name, 'general')
                 if c != 'none']
    augs = data.get('augmentations') if _aug_kept else None
    if data.get('augmentations') and not _aug_kept:
        print(f"  augmentation cut for {dataset_name} (DECISIONS.md Step 8): "
              f"no added features in any search or arm")

    # The dataset's decided general budgets (R3), named here and not at
    # import: lhasa_herg 100-2000 features, caco2 and logd 500-2000. Before
    # 2026-10-05 every dataset swept 50, 100, 250, 500 and 1000.
    BUDGETS = [int(b) for b in federated_budgets(dataset_name)]
    # The width the per-fold summary rows are taken at: 100 features where
    # the dataset's budgets include it (lhasa_herg), else its smallest
    # budget (500 on caco2 and logd). Every width is in budget_sweep_per_fold.
    HEADLINE_BUDGET = 100 if 100 in BUDGETS else BUDGETS[0]

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Resolve silo reps, from the dataset's pruned representations only.
    _pool = _silo_pool(dataset_name, set(reps.keys()))
    silo_reps = _get_silo_reps(dataset_name, set(_pool))
    silo_letters = sorted(silo_reps.keys())
    K = len(silo_reps)

    if K < 3:
        raise ValueError(
            f"Need at least 3 silo reps, got {K}. "
            f"Available among the pruned representations: {_pool}")

    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{K} silos assigned:")
    for s in silo_letters:
        rep_name = silo_reps[s]
        dim = reps[rep_name].shape[1]
        print(f"    Silo {s}: {rep_name} ({dim}D)")
    print(f"  model: {kirby_model} (decided: {_decided_fed}); budgets "
          f"{BUDGETS}, headline {HEADLINE_BUDGET} features")

    # Outer folds: ONE setting, config['n_splits'] (handoff 2.4 step 14), and
    # fed_outer_folds -- seed 42, the same rows as the per-fold search store.
    # Until 2026-10-05 this was smoke_n_folds(5) with seed=fed_seed, so fold i
    # was different molecules for each of the three seeds and none of them
    # was fold i of the stored search. 1 is a single grouped held-out split.
    N_FOLDS = int(config['n_splits'])
    if N_FOLDS > 1:
        N_FOLDS = smoke_n_folds(N_FOLDS)   # KIRBY_FED_FOLDS / KIRBY_SMOKE_N
    _need = max(2, N_FOLDS)
    if groups is None or len(np.unique(groups)) < _need:
        n_groups = 0 if groups is None else len(np.unique(groups))
        raise ValueError(
            f"fed02 requires scaffold groups for {N_FOLDS}-fold CV. "
            f"Got n_groups={n_groups} (need >= {_need}). "
            f"Per project rule (no random splits), refusing to fall back "
            f"to KFold.")

    fold_splits = list(fed_outer_folds(reps, labels, groups, N_FOLDS,
                                       task_type))
    print(f"  Using {N_FOLDS} outer fold(s), grouped, seed {FED_FOLD_SEED}")

    def _make_model():
        # Unfitted template of the run's model. Used only as create_hybrid's
        # model=; the internal evaluator is resolved by name before every
        # build (_resolve_evaluator), so 'match' never depends on it.
        if is_clf:
            return make_fed_classifier(fed_model, seed=fed_seed)
        return make_fed_regressor(fed_model, seed=fed_seed)

    _compute_metrics_raw = (compute_clf_metrics if is_clf
                            else compute_reg_metrics)

    def _compute_metrics(*a, **kw):
        """The held-out metrics, without compute_reg_metrics' spearman_p and
        pearson_p (2026-10-05: no p-values in any result file). Until then
        both were written per fold and averaged over folds."""
        m = _compute_metrics_raw(*a, **kw)
        for _k in _P_VALUE_KEYS:
            m.pop(_k, None)
        return m

    def _score(mdl, X_te, y_te):
        y_pred = mdl.predict(X_te)
        if is_clf:
            y_prob = (mdl.predict_proba(X_te)[:, 1]
                      if hasattr(mdl, 'predict_proba') else None)
            return _compute_metrics(y_te, y_pred, y_prob=y_prob)
        return _compute_metrics(y_te, y_pred)

    def _fmt(m):
        return (f"MCC={m['mcc']:.4f}" if is_clf else f"RMSE={m['rmse']:.4f}")

    from kirby.hybrid import (
        create_hybrid, apply_feature_selection, apply_augmentation_selection)
    from experiments._shared import (
        hybrid_feature_signature, fold_searches, top_hybrids, hybrid_deltas)

    # One record per outer fold, written to <dataset>__folds.partial.jsonl
    # the moment the fold finishes and read back on the next attempt
    # (dirty-run bug 12; 2026-10-05). Keyed on the model, the seed and the
    # fold, under the input fingerprint, the plain config without those two,
    # the KIRBY_* settings and RESULT_LAYOUT.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _fold_ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name,
                           suffix='folds'),
        run_identity(EXPERIMENT_DIR, dataset_name, data,
                     {k: v for k, v in config.items()
                      if k not in ('fed_model', 'fed_seed')},
                     layout=RESULT_LAYOUT))

    def _one_fold(fold_i, train_idx, test_idx):
        print(f"\n{'=' * 70}\nFOLD {fold_i + 1}/{N_FOLDS}\n{'=' * 70}")
        rec = {'fold': int(fold_i), 'model': kirby_model,
               'fed_model': fed_model,
               'selection_scope': 'fold_train_rows',
               'n_train': int(len(train_idx)), 'n_test': int(len(test_idx))}

        y_train = labels[train_idx]
        y_test = labels[test_idx]
        g_train = groups[train_idx] if groups is not None else None
        print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

        # Pre-split reps (per fold)
        train_reps = {r: arr[train_idx] for r, arr in reps.items()}
        test_reps = {r: arr[test_idx] for r, arr in reps.items()}
        train_augs = ({k: v[train_idx] for k, v in augs.items()}
                      if augs else None)
        test_augs = ({k: v[test_idx] for k, v in augs.items()}
                     if augs else None)

        def _append_test_augs(X_test, feature_info):
            if not test_augs:
                return X_test
            aug_info = (feature_info.get('augmentation_info')
                        if isinstance(feature_info, dict) else None)
            if not aug_info:
                return X_test
            X_aug = apply_augmentation_selection(test_augs, aug_info)
            return np.hstack([X_test, X_aug]) if X_aug is not None else X_test

        _match_model = _make_model()

        # ── Phase 0: the per-fold search and its selection ───────────────
        # ONE search on this fold's training rows, on the dataset's decided
        # grid with pruning off, shared by every budget and both models
        # (fed_fold_selection, R2/R6). It replaces the whole-dataset reads
        # (get_best_search_for_downstream, get_rule_hybrids) every recipe of
        # this file came from until 2026-10-05.
        # An exception from inside the selection (a candidate build, the
        # ranking, the search itself) is recorded with the dataset, fold and
        # model, and only the arms that need the search are skipped; C1-C3
        # still run on this fold (2026-10-05). Before, it ended run() for
        # this model and seed with no per-fold record.
        try:
            sel = _fold_selection(
                reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
                n_folds=N_FOLDS, dataset_name=dataset_name, budgets=BUDGETS,
                task_type=task_type, augmentations=augs)
        except Exception as _e:
            print(f"  per-fold selection RAISED: {type(_e).__name__}: {_e}")
            traceback.print_exc()
            sel = {'_errors': [{
                'dataset': dataset_name, 'fold': int(fold_i),
                'model': kirby_model,
                'error': f'{type(_e).__name__}: {_e}',
                'traceback': traceback.format_exc()}]}
        rec['search_errors'] = convert_numpy(list(sel.get('_errors') or []))
        my_sel = sel.get(kirby_model)
        if my_sel is None:
            rec['search_failed'] = {
                'dataset': dataset_name, 'fold': int(fold_i),
                'model': kirby_model,
                'error': ('the per-fold search returned nothing for this '
                          'model; every arm that needs a recipe or the '
                          'search\'s base representations is skipped'),
                'search_errors': rec['search_errors']}
            print(f"  per-fold search FAILED for {kirby_model}: "
                  f"{rec['search_errors']}")

        # Report rows, per fold, for both decided models (2026-10-05). They
        # replace top2_base_reps / top2_hybrids / top2_pairs_by_model /
        # rule_set_by_model, which ranked the whole-dataset search with no
        # row held out. They fit nothing.
        _report = {}
        for _m, _s in sel.items():
            if _m == '_errors':
                continue
            if _s is None:
                _report[_m] = {'error': 'the per-fold search failed'}
                continue
            _bd = _s.get('base_detail') or {}
            _report[_m] = {
                'base': list(_s.get('base') or []),
                'base_train_rows_scores': [
                    {'rep': e.get('rep'), 'score': e.get('score'),
                     'dimension': e.get('dimension')}
                    for e in (_bd.get('selected') or [])],
                'base_shortfall': _bd.get('shortfall_reason'),
                'by_budget': {
                    str(_b): {
                        'hybrids': [_recipe_summary(
                                        e, _m, _s.get('search_metric'),
                                        _s.get('search_lower_is_better'))
                                    for e in (_v.get('hybrids') or [])],
                        'skipped_identical_features': convert_numpy(
                            _v.get('hybrids_skipped_identical_features')
                            or []),
                    } for _b, _v in (_s.get('by_budget') or {}).items()},
                'search_metric': _s.get('search_metric'),
                'search_lower_is_better': _s.get('search_lower_is_better'),
                'selection_scope': 'fold_train_rows',
            }
        rec['selection_by_model'] = _report

        # The two hybrid deltas per model, from the same per-fold search
        # (training-row scores), replacing the whole-dataset ones. Same
        # arguments as fed_fold_selection's search, so the memo serves it.
        _deltas = {}
        try:
            _srch = fold_searches(
                reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
                n_outer_folds=N_FOLDS, dataset_name=dataset_name,
                augmentations=augs, task_type=task_type, metric='auto',
                seed=FED_FOLD_SEED, verbose=False).get('none')
        except Exception as _e:
            _srch = None
            _deltas['_error'] = f'{type(_e).__name__}: {_e}'
        if _srch is not None:
            for _m in _report:
                try:
                    _top1 = top_hybrids(
                        _srch, _m, 1, require_full=False,
                        context=f'fed02 {dataset_name} fold {fold_i} '
                                f'deltas/{_m}')
                    if not _top1['selected']:
                        _deltas[_m] = {'error': 'no hybrid configuration for '
                                                f"this model: "
                                                f"{_top1['shortfall_reason']}"}
                        continue
                    _deltas[_m] = hybrid_deltas(
                        _srch, _m, hybrid=_top1['selected'][0],
                        context=f'fed02 {dataset_name} fold {fold_i} '
                                f'hybrid deltas')
                except Exception as _e:
                    _deltas[_m] = {'error': f'{type(_e).__name__}: {_e}'}
        rec['hybrid_deltas_by_model'] = convert_numpy(_deltas)

        # ── Phase 1b: Rank the five representations inside the training
        # rows only ──────────────────────────────────────────────────
        # This ranking feeds two conditions. C5 takes its three from it. C1
        # takes its one from it. Cost is 5 representations x
        # INNER_RANK_FOLDS fits per fold.
        try:
            inner_scores = _rank_reps_train_only(
                silo_letters, silo_reps, train_reps, y_train,
                groups=g_train, fed_model=fed_model, fed_seed=fed_seed,
                task_type=task_type, is_clf=is_clf,
                compute_metrics=_compute_metrics,
                n_inner_folds=INNER_RANK_FOLDS)
            silo_ranked = sorted(inner_scores.items(), key=lambda x: x[1],
                                 reverse=bool(is_clf))
        except Exception as e:
            print(f"  Inner rep ranking FAILED: {e}")
            traceback.print_exc()
            inner_scores, silo_ranked = None, None
        rec['inner_scores'] = (convert_numpy(inner_scores)
                               if inner_scores else None)
        # How far the inner ranking's #1 and #2 silos sit apart, in the
        # metric's own units, and which silo was #2. A margin near zero means
        # the pick between them is close to a coin flip -- MEASURED on caco2
        # fold 0: the #1 pick flipped between two silos in 6 of 20 redraws of
        # the same training rows, and their scores sat 0.0010 apart against a
        # run-to-run wobble of 0.0006-0.0010 each (2026-09-18 check).
        if silo_ranked and len(silo_ranked) >= 2:
            rec['rank0_margin'] = float(abs(silo_ranked[1][1]
                                            - silo_ranked[0][1]))
            rec['rank0_runnerup'] = silo_ranked[1][0]
        else:
            rec['rank0_margin'] = None
            rec['rank0_runnerup'] = None

        # ── Phase 2: Single-silo baselines (C1) ──────────────────────
        print(f"\n--- Phase 2: Single-silo baselines (C1) [task={task_type}] ---")
        single_silo = {}
        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            mdl = _fit(fed_model, X_tr, y_train, seed=fed_seed,
                       task_type=task_type)
            metrics = _score(mdl, test_reps[rep_name], y_test)
            single_silo[s] = {'rep_name': rep_name,
                              'n_features': int(X_tr.shape[1]), **metrics}
            if is_clf:
                print(f"  Silo {s} ({rep_name}): "
                      f"MCC={metrics['mcc']:.4f}  "
                      f"BAC={metrics['bac']:.4f}  "
                      f"AUROC={metrics.get('auroc', float('nan')):.4f}")
            else:
                _rho = metrics.get('spearman_rho')
                _rho_s = ('undefined' if _rho is None else f"{_rho:.4f}")
                print(f"  Silo {s} ({rep_name}): "
                      f"RMSE={metrics['rmse']:.4f}  "
                      f"rho={_rho_s}  "
                      f"R2={metrics['r2']:.4f}")

        # Best silo (per fold), picked two ways. Both are reported; they
        # answer different questions and must never be merged into one row.
        #
        #  test-picked — the representation with the best score on THIS fold's
        #    held-out molecules. Nobody could have chosen it in advance, so it
        #    is a ceiling.
        #  train-picked — the representation that won the inner ranking on the
        #    training rows, scored afterwards on the held-out molecules. This
        #    is the honest single-representation baseline, and it is the one
        #    C4's gain has to be measured against.
        if is_clf:
            best_silo_f = max(single_silo, key=lambda s: single_silo[s]['mcc'])
        else:
            best_silo_f = min(single_silo, key=lambda s: single_silo[s]['rmse'])
        best_silo_train_f = silo_ranked[0][0] if silo_ranked else None
        print(f"  Best on the held-out fold (ceiling): Silo {best_silo_f} "
              f"({silo_reps[best_silo_f]})")
        if best_silo_train_f is not None:
            print(f"  Best on the training rows (honest):  Silo "
                  f"{best_silo_train_f} ({silo_reps[best_silo_train_f]})")
        else:
            print("  Best on the training rows (honest):  unavailable "
                  "— the inner ranking failed on this fold")
        rec['single_silo'] = convert_numpy(single_silo)
        rec['best_silo_test'] = best_silo_f
        rec['best_silo_train'] = best_silo_train_f

        # ── Phase 2b: the per-fold search's 2 best base representations
        # (R4, 2026-10-05) ──────────────────────────────────────────────
        # Ranked by the per-fold search over the dataset's pruned
        # representations, not only the five silo ones, and fitted whole,
        # like C1. Beside C1, not instead of it: C1 is the silo question.
        base_arms = {}
        if my_sel is not None:
            _bscores = {e.get('rep'): e.get('score') for e in
                        ((my_sel.get('base_detail') or {}).get('selected')
                         or [])}
            for _i, _rep in enumerate(my_sel.get('base') or []):
                _slot = f'base{_i + 1}'
                try:
                    mdl = _fit(fed_model, train_reps[_rep], y_train,
                               seed=fed_seed, task_type=task_type)
                    m = _score(mdl, test_reps[_rep], y_test)
                    base_arms[_slot] = {
                        'rep_name': _rep,
                        'n_features': int(train_reps[_rep].shape[1]),
                        'train_rows_score': _bscores.get(_rep),
                        'train_rows_metric': my_sel.get('search_metric'),
                        'train_rows_lower_is_better': my_sel.get(
                            'search_lower_is_better'),
                        'held_out_metric': _primary_key,
                        **m}
                    print(f"  {_slot} ({_rep}): {_fmt(m)}")
                except Exception as e:
                    base_arms[_slot] = {'rep_name': _rep, 'error': str(e),
                                        'traceback': traceback.format_exc()}
                    print(f"  {_slot} ({_rep}) FAILED: {e}")
            _short = (my_sel.get('base_detail') or {}).get('shortfall_reason')
            if len(my_sel.get('base') or []) < 2:
                base_arms['_shortfall'] = _short or 'fewer than 2 scored'
        rec['base_arms'] = convert_numpy(base_arms)

        # ── Phase 3: FLuID ensemble (C2, C3) ─────────────────────────
        print("\n--- Phase 3: FLuID ensemble conditions ---")
        teacher_models = [None]
        silo_train_features_sim = [None]
        silo_test_features_sim = [None]
        silo_train_features_qual = [None]
        silo_test_features_qual = [None]

        for s in silo_letters:
            rep_name = silo_reps[s]
            X_tr = train_reps[rep_name]
            X_te = test_reps[rep_name]
            mdl = _fit(fed_model, X_tr, y_train, seed=fed_seed,
                       task_type=task_type)
            teacher_models.append((mdl, len(X_tr), X_tr))
            silo_train_features_sim.append(X_tr)
            silo_test_features_sim.append(X_te)
            silo_train_features_qual.append((X_tr, y_train))
            silo_test_features_qual.append(X_te)

        print("  C2: FLuID similarity...", end=' ')
        try:
            if is_clf:
                y_pred_c2, y_prob_c2, tw_c2 = predict_fluid_similarity_ensemble_classification(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2, y_prob=y_prob_c2)
            else:
                y_pred_c2, tw_c2 = predict_fluid_similarity_ensemble_regression(
                    teacher_models, silo_train_features_sim,
                    silo_test_features_sim, K)
                c2 = _compute_metrics(y_test, y_pred_c2)
            print(_fmt(c2))
            c2['teacher_weights_mean'] = convert_numpy(
                {silo_letters[i]: float(tw_c2[:, i].mean())
                 for i in range(K)})
        except Exception as e:
            c2 = {'error': str(e)}
            print(f"FAILED: {e}")
        rec['c2'] = convert_numpy(c2)

        # Each teacher's weight is cross-validated by scaffold group on the
        # fold's training rows (every silo holds the same rows, so the same
        # groups), 2026-10-06; was KFold(shuffle=True).
        silo_train_groups_qual = [None] + [g_train] * K
        print("  C3: FLuID quality...", end=' ')
        try:
            if is_clf:
                y_pred_c3, y_prob_c3, mcc_w_c3 = predict_fluid_quality_ensemble_classification(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K,
                    silo_train_groups=silo_train_groups_qual)
                c3 = _compute_metrics(y_test, y_pred_c3, y_prob=y_prob_c3)
                c3['mcc_weights'] = convert_numpy(mcc_w_c3)
            else:
                y_pred_c3, rmse_w_c3 = predict_fluid_quality_ensemble_regression(
                    teacher_models, silo_train_features_qual,
                    silo_test_features_qual, K,
                    silo_train_groups=silo_train_groups_qual)
                c3 = _compute_metrics(y_test, y_pred_c3)
                c3['rmse_weights'] = convert_numpy(rmse_w_c3)
            print(_fmt(c3))
        except Exception as e:
            c3 = {'error': str(e)}
            print(f"FAILED: {e}")
        c3['weight_cv'] = 'GroupKFold on scaffold groups'
        rec['c3'] = convert_numpy(c3)

        # ── Phases 4-5: C4, C5, C6 per hybrid per budget ─────────────
        # C4, C5 and C6 all pick `budget` features and are measured against
        # C2 and C3, which average five models over WHOLE representations, so
        # all three sweep the width together.
        kirby_train = {silo_reps[s]: train_reps[silo_reps[s]]
                       for s in silo_letters}
        kirby_test = {silo_reps[s]: test_reps[silo_reps[s]]
                      for s in silo_letters}
        top_3_silos = ([s for s, _ in silo_ranked[:3]]
                       if silo_ranked is not None else None)
        top_3_names = ([silo_reps[s] for s in top_3_silos]
                       if top_3_silos else None)
        rec['top_3_silos'] = top_3_silos
        rec['top_3_reps'] = top_3_names

        def _build_selected(rep_tr, rep_te, bp, aug_cond, budget):
            """create_hybrid with the whole recipe on the given silo
            representations (all of them: APPLY_REP_COUNT_TO_SILOS is False).
            Returns (X_train, X_test, feature_info, representations used)."""
            rep_tr, _rc = _restrict_rep_count(
                rep_tr, bp, context=f'fed02 {dataset_name} fold {fold_i}')
            rep_te = {k: rep_te[k] for k in rep_tr}
            _use_aug = aug_cond != 'none' and train_augs is not None
            X_tr, fi = create_hybrid(
                rep_tr, y_train, groups=g_train,
                model=_match_model,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                allocation_method=bp.allocation_method,
                importance_method=bp.importance_method,
                budget=int(budget),
                internal_evaluator=_resolve_evaluator(
                    bp.internal_evaluator, kirby_model),
                augmentations=train_augs if _use_aug else None,
                augmentation_strategy=aug_cond if _use_aug else 'none',
                **_recipe_kwargs(bp))
            X_te = apply_feature_selection(rep_te, fi)
            if _use_aug:
                X_te = _append_test_augs(X_te, fi)
            return X_tr, X_te, fi, sorted(rep_tr)

        by_budget = {}
        for BUDGET in BUDGETS:
            print(f"\n--- Phases 4-5: KIRBy hybrids at {BUDGET} features ---")
            _bsel = ((my_sel or {}).get('by_budget') or {}).get(int(BUDGET))
            _hybrids = list((_bsel or {}).get('hybrids') or [])
            _skipped_sel = convert_numpy(
                (_bsel or {}).get('hybrids_skipped_identical_features') or [])
            by_budget[str(BUDGET)] = {'_skipped_by_search_identical_features':
                                      _skipped_sel}
            # Features of hybrid1's three arms, to catch hybrid2 building the
            # same ones on the silo representations (answer 8: a second
            # hybrid whose selected features equal the first's is skipped;
            # the search checked this on the pruned representations, not on
            # the silo subset).
            _sig_first = {}
            for _hi, _slot in enumerate(HYBRID_SLOTS):
                if _hi >= len(_hybrids):
                    _why = ('the per-fold search failed for this model'
                            if my_sel is None else
                            f'the per-fold search has {len(_hybrids)} '
                            f'distinct hybrid(s) for {kirby_model} at '
                            f'{BUDGET} features')
                    by_budget[str(BUDGET)][_slot] = {
                        'skipped': _why, 'dataset': dataset_name,
                        'fold': int(fold_i), 'model': kirby_model}
                    print(f"  {_slot}: skipped, {_why}")
                    continue
                entry = _hybrids[_hi]
                bp = entry['blueprint']
                aug_cond = entry.get('aug_condition', 'none')
                recipe = _recipe_summary(
                    entry, kirby_model, my_sel.get('search_metric'),
                    my_sel.get('search_lower_is_better'))
                print(f"  {_slot}: allocation={recipe['allocation_method']} "
                      f"importance={recipe['importance_method']} "
                      f"evaluator={recipe['internal_evaluator_resolved']} "
                      f"rep_count={recipe['rep_count']} "
                      f"augmentation={aug_cond} "
                      f"train-rows score={recipe['train_rows_score']}")
                # The metric the arms' held-out scores below are read in,
                # beside the search's training-row metric in the recipe.
                recipe['held_out_metric'] = _primary_key
                recipe['held_out_lower_is_better'] = bool(not is_clf)
                arms = {'recipe': recipe}

                def _finish(cond, X_tr, X_te, sig, extra):
                    """Skip a hybrid2 arm whose features equal hybrid1's;
                    else fit and score it on the held-out rows."""
                    if _hi == 0:
                        _sig_first[cond] = sig
                    elif sig is not None and _sig_first.get(cond) == sig:
                        print(f"  {cond} [{_slot}]: same selected features "
                              f"as hybrid1, skipped")
                        return {'skipped': 'identical selected features to '
                                           'hybrid1 on these silo '
                                           'representations',
                                'identical_features_to': 'hybrid1',
                                'n_features': int(X_tr.shape[1]), **extra}
                    mdl = _fit(fed_model, X_tr, y_train, seed=fed_seed,
                               task_type=task_type)
                    m = _score(mdl, X_te, y_test)
                    print(f"  {cond} [{_slot}]: {_fmt(m)}  "
                          f"({X_tr.shape[1]}D)")
                    return {**m, 'n_features': int(X_tr.shape[1]), **extra}

                # Condition 4: KIRBy hybrid (all K silo representations)
                try:
                    X_tr, X_te, fi, used = _build_selected(
                        kirby_train, kirby_test, bp, aug_cond, BUDGET)
                    arms['c4_kirby_hybrid'] = _finish(
                        'c4_kirby_hybrid', X_tr, X_te,
                        hybrid_feature_signature(fi),
                        {'reps_used': used,
                         'allocation': convert_numpy(
                             fi.get('allocation', {}))})
                except Exception as e:
                    arms['c4_kirby_hybrid'] = {
                        'error': str(e), 'traceback': traceback.format_exc()}
                    print(f"  C4 [{_slot}] FAILED: {e}")
                    traceback.print_exc()

                # Condition 5: the top three silo representations of the
                # inner ranking on the training rows (Phase 1b).
                try:
                    if top_3_silos is None:
                        raise RuntimeError(
                            "C5 needs the Phase 1b train-only ranking and it "
                            "failed on this fold.")
                    X_tr, X_te, fi, used = _build_selected(
                        {r: kirby_train[r] for r in top_3_names},
                        {r: kirby_test[r] for r in top_3_names},
                        bp, aug_cond, BUDGET)
                    arms['c5_kirby_top_tier'] = _finish(
                        'c5_kirby_top_tier', X_tr, X_te,
                        hybrid_feature_signature(fi),
                        {'reps_used': used,
                         'top_3_silos': top_3_silos,
                         'top_3_reps': top_3_names,
                         'inner_rank_scores': convert_numpy(inner_scores),
                         'inner_rank_metric': _primary_key,
                         'inner_rank_folds': INNER_RANK_FOLDS,
                         'allocation': convert_numpy(
                             fi.get('allocation', {}))})
                except Exception as e:
                    arms['c5_kirby_top_tier'] = {
                        'error': str(e), 'traceback': traceback.format_exc()}
                    print(f"  C5 [{_slot}] FAILED: {e}")
                    traceback.print_exc()

                # ── Phase 5: FLuID→KIRBy pipeline (C6) ───────────────────
                # The silo representations (all 5: APPLY_REP_COUNT_TO_SILOS
                # is False), and the recipe's allocation method,
                # importance method, internal evaluator and budget. The rest
                # of the recipe (step size, augmentation settings) has no way
                # into fluid_within_kirby_across without a change to
                # federated_helpers; two hybrids that differ only there build
                # the same C6 features, and _finish catches and records it.
                try:
                    c6_tr, _rc6 = _restrict_rep_count(
                        kirby_train, bp,
                        context=f'fed02 {dataset_name} fold {fold_i} C6')
                    c6_te = {k: kirby_test[k] for k in c6_tr}
                    X_tr, X_te, budgets_c6, scores_c6, meta_c6 = \
                        fluid_within_kirby_across(
                            c6_tr, c6_te, y_train,
                            k_intra=K_INTRA,
                            budget=int(BUDGET),
                            allocation_method=bp.allocation_method,
                            importance_method=bp.importance_method,
                            internal_evaluator=_resolve_evaluator(
                                bp.internal_evaluator, kirby_model),
                            downstream_model=kirby_model,
                            groups=g_train,
                            seed=fed_seed,
                            task_type=task_type,
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better)
                    # Failed or too-small virtual silos get 0 features and a
                    # None score (federated_helpers, 2026-10-05).
                    _failed = {str(k): v.get('error') or v.get('score_source')
                               for k, v in meta_c6.items()
                               if scores_c6.get(k) is None}
                    _c6sig = (tuple(X_tr.shape),
                              hashlib.sha1(np.ascontiguousarray(X_tr)
                                           .tobytes()).hexdigest())
                    arms['c6_fluid_within_kirby_across'] = _finish(
                        'c6_fluid_within_kirby_across', X_tr, X_te, _c6sig,
                        {'reps_used': sorted(c6_tr),
                         'silo_budgets': convert_numpy(budgets_c6),
                         'silo_scores': convert_numpy(scores_c6),
                         'failed_virtual_silos': _failed,
                         'n_virtual_silos': len(meta_c6),
                         'virtual_silo_metadata': convert_numpy(meta_c6)})
                except Exception as e:
                    arms['c6_fluid_within_kirby_across'] = {
                        'error': str(e), 'traceback': traceback.format_exc()}
                    print(f"  C6 [{_slot}] FAILED: {e}")
                    traceback.print_exc()
                by_budget[str(BUDGET)][_slot] = convert_numpy(arms)
        rec['by_budget'] = by_budget
        return convert_numpy(rec)

    fold_records = []
    for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
        _k = dict(fed_model=fed_model, fed_seed=int(fed_seed), fold=fold_i)
        if _fold_ck.done(**_k):
            print(f"  [partial] fold {fold_i}: read back from {_fold_ck.path}")
            rec = _fold_ck.get(**_k)
        else:
            rec = _one_fold(fold_i, train_idx, test_idx)
            # A fold whose per-fold search failed is recorded in this run's
            # result but not saved as finished, so the next attempt runs the
            # search again instead of reading back a fold with every recipe
            # arm skipped (2026-10-05). The searches that did finish are on
            # disk in the per-fold search store and are not redone.
            if rec.get('search_failed'):
                print(f"  [partial] fold {fold_i}: not saved as finished, "
                      f"its per-fold search failed; retried next attempt")
            else:
                _fold_ck.put(rec, **_k)
        fold_records.append(rec)

    # ── Phase 6: Aggregate per-fold metrics → mean ± std ─────────────────
    print(f"\n{'=' * 70}\nAGGREGATING ACROSS {N_FOLDS} FOLD(S)\n{'=' * 70}")

    # No spearman_p / pearson_p (2026-10-05): no p-values are reported.
    _scalar_keys_reg = ['rmse', 'spearman_rho', 'pearson_r', 'r2']
    _scalar_keys_clf = ['mcc', 'bac', 'f1', 'auroc', 'accuracy', 'f1_weighted']
    _scalar_keys = _scalar_keys_clf if is_clf else _scalar_keys_reg

    def _is_num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and not (isinstance(v, float) and np.isnan(v)))

    def _aggregate_folds(fold_list):
        """Mean±std of scalar metrics across folds; preserve last-fold extras.
        With one fold the std is None and 'spread' says 'one fold, no
        spread' (2026-10-05)."""
        out = {}
        for k in _scalar_keys:
            vals = [f.get(k) for f in fold_list
                    if isinstance(f, dict) and _is_num(f.get(k))]
            mu, sd, note = _mean_spread(vals)
            out[k] = mu
            out[f'{k}_std'] = sd
            out[f'{k}_per_fold'] = [float(v) for v in vals]
            if note:
                out['spread'] = note
        # Preserve non-scalar extras from the last successful fold (e.g.
        # allocation, n_features, teacher_weights_mean, silo_budgets).
        for f in reversed(fold_list):
            if isinstance(f, dict) and 'error' not in f and 'skipped' not in f:
                for k, v in f.items():
                    if k in _scalar_keys:
                        continue
                    if k.endswith('_std') or k.endswith('_per_fold'):
                        continue
                    out.setdefault(k, v)
                break
        errors = [f.get('error') for f in fold_list
                  if isinstance(f, dict) and 'error' in f]
        if errors:
            out['errors_per_fold'] = errors
        skipped = [f.get('skipped') for f in fold_list
                   if isinstance(f, dict) and 'skipped' in f]
        if skipped:
            out['skipped_per_fold'] = skipped
        return out

    def _arm(rec, budget, slot, cond):
        a = ((rec.get('by_budget') or {}).get(str(budget)) or {}).get(slot)
        if not isinstance(a, dict):
            return None
        if 'skipped' in a and cond not in a:
            return {'skipped': a['skipped']}
        return a.get(cond)

    _conds = ('c4_kirby_hybrid', 'c5_kirby_top_tier',
              'c6_fluid_within_kirby_across')
    c2 = _aggregate_folds([r.get('c2') for r in fold_records])
    c3 = _aggregate_folds([r.get('c3') for r in fold_records])
    # hybrid1 at the headline width: the keys this file always wrote.
    c4, c5, c6 = (_aggregate_folds([_arm(r, HEADLINE_BUDGET, 'hybrid1', c)
                                    for r in fold_records]) for c in _conds)
    # hybrid2 at the headline width, aggregated the same way. Every width of
    # both is in budget_sweep_per_fold.
    hybrid2_arms = {c: _aggregate_folds([_arm(r, HEADLINE_BUDGET, 'hybrid2', c)
                                         for r in fold_records])
                    for c in _conds}
    base_arms_agg = {
        _slot: _aggregate_folds([(r.get('base_arms') or {}).get(_slot)
                                 for r in fold_records])
        for _slot in ('base1', 'base2')}

    # Aggregate single-silo per-letter, then pick overall best by mean primary
    silo_agg = {}
    for s in silo_letters:
        vals_by_metric = {k: [] for k in _scalar_keys}
        n_feat = None
        for r in fold_records:
            entry = (r.get('single_silo') or {}).get(s)
            if entry is None:
                continue
            for k in _scalar_keys:
                if _is_num(entry.get(k)):
                    vals_by_metric[k].append(float(entry.get(k)))
            if n_feat is None:
                n_feat = entry.get('n_features')
        agg = {'rep_name': silo_reps[s], 'n_features': n_feat}
        for k, vals in vals_by_metric.items():
            mu, sd, note = _mean_spread(vals)
            agg[k] = mu
            agg[f'{k}_std'] = sd
            if note:
                agg['spread'] = note
        silo_agg[s] = agg

    if is_clf:
        best_silo = max(silo_agg, key=lambda s: (silo_agg[s].get('mcc')
                                                 if _is_num(silo_agg[s].get('mcc'))
                                                 else -np.inf))
    else:
        best_silo = min(silo_agg, key=lambda s: (silo_agg[s].get('rmse')
                                                 if _is_num(silo_agg[s].get('rmse'))
                                                 else np.inf))
    print(f"  Best silo on the held-out folds (ceiling): "
          f"{best_silo} ({silo_reps[best_silo]})")

    fold_best_silo = [r.get('best_silo_test') for r in fold_records]
    fold_best_silo_train = [r.get('best_silo_train') for r in fold_records]
    fold_rank0_margin = [r.get('rank0_margin') for r in fold_records]
    fold_rank0_runnerup = [r.get('rank0_runnerup') for r in fold_records]

    def _picked_agg(picks):
        """The held-out scores of the silo each fold picked on its training
        rows; no representation is ever scored on rows that chose it."""
        vals = {}
        for r, s in zip(fold_records, picks):
            entry = (r.get('single_silo') or {}).get(s) if s else None
            if entry is None:
                continue
            for k, v in entry.items():
                if _is_num(v):
                    vals.setdefault(k, []).append(float(v))
        out = {'picked_per_fold': list(picks),
               'picked_reps_per_fold': [silo_reps.get(s) if s else None
                                        for s in picks],
               'n_folds_scored': len(vals.get(_primary_key, []))}
        for k, v in vals.items():
            mu, sd, note = _mean_spread(v)
            out[k] = mu
            out[f'{k}_std'] = sd
            if note:
                out['spread'] = note
        return out

    c1_trainpick = _picked_agg(fold_best_silo_train)
    # How close the #1 pick was to the runner-up, per fold, in
    # inner_rank_metric's own units.
    c1_trainpick['rank0_margin_per_fold'] = list(fold_rank0_margin)
    c1_trainpick['rank0_runnerup_per_fold'] = list(fold_rank0_runnerup)
    c1_trainpick['rank0_runnerup_reps_per_fold'] = [
        silo_reps.get(s) if s else None for s in fold_rank0_runnerup]
    # The inner ranking's runner-up on each fold, scored the same way.
    c1_trainpick_second = _picked_agg(fold_rank0_runnerup)
    if c1_trainpick['n_folds_scored'] == 0:
        print("  Best silo on the training rows (honest): none — the inner "
              "ranking failed on every fold")
    else:
        _picked = [s for s in fold_best_silo_train if s is not None]
        print(f"  Best silo on the training rows (honest): "
              f"{sorted(set(_picked))} across {len(_picked)} folds, "
              f"{_primary_key}={c1_trainpick[_primary_key]:.4f}")

    # Per-fold failures of the per-fold search (dataset, fold, model, error),
    # so a skipped arm is never read as a measured one (R6).
    search_failures = [r['search_failed'] for r in fold_records
                       if r.get('search_failed')]
    search_errors = [{'fold': r['fold'], 'errors': r.get('search_errors')}
                     for r in fold_records if r.get('search_errors')]

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': 'fed02_rep_diversity_silos',
        'dataset': dataset_name,
        'task_type': task_type,
        # The metric every held-out row below is read in (run_all stamps the
        # result file with these two). mcc on lhasa_herg's two classes, rmse
        # on caco2 and logd.
        'metric_name': _primary_key,
        'lower_is_better': (not is_clf),
        'model': kirby_model,
        'fed_model': fed_model,
        'selection_scope': 'fold_train_rows',
        'result_layout': RESULT_LAYOUT,
        'config': {
            'silo_reps': {s: silo_reps[s] for s in silo_letters},
            'silo_pool': _pool,
            'n_silos': K,
            'n_samples': len(labels),
            'n_folds': N_FOLDS,
            'outer_fold_seed': FED_FOLD_SEED,
            'n_train_per_fold': [r.get('n_train') for r in fold_records],
            'n_test_per_fold': [r.get('n_test') for r in fold_records],
            'k_intra_c6': K_INTRA,
            'budgets_swept': BUDGETS,
            'budget': HEADLINE_BUDGET,
            'hybrid_slots': list(HYBRID_SLOTS),
            'apply_rep_count_to_silos': APPLY_REP_COUNT_TO_SILOS,
            'scale_features_before_fit': SCALE_FEATURES_BEFORE_FIT,
            'augmentation': (_aug_kept or ['none']),
            'best_silo_per_fold': fold_best_silo,
            'best_silo_train_picked_per_fold': fold_best_silo_train,
            'inner_rank_scores_per_fold': [r.get('inner_scores')
                                           for r in fold_records],
            'inner_rank_metric': _primary_key,
            'inner_rank_folds': INNER_RANK_FOLDS,
            'top_3_silos_per_fold': [r.get('top_3_silos')
                                     for r in fold_records],
        },
        # C1 is reported two ways and there is no single 'C1 number'. Read
        # 'train_picked' against C4; read 'test_picked_ceiling' only as a
        # bound, and never put it in a table beside the other conditions
        # without saying it is one.
        'c1_best_single_silo': convert_numpy({
            'train_picked': c1_trainpick,
            'train_picked_second': c1_trainpick_second,
            'test_picked_ceiling': {
                'best_silo': best_silo,
                'best_rep': silo_reps[best_silo],
                f'best_{_primary_key}': silo_agg[best_silo][_primary_key],
                f'best_{_primary_key}_std': silo_agg[best_silo].get(
                    f'{_primary_key}_std'),
                'picked_per_fold': fold_best_silo,
            },
            'per_silo': silo_agg,
        }),
        # The per-fold search's 2 best base representations for this model,
        # fitted whole and scored on the held-out rows (R4, 2026-10-05).
        'base_reps_fold_search': convert_numpy(base_arms_agg),
        'c2_fluid_similarity': convert_numpy(c2),
        'c3_fluid_quality': convert_numpy(c3),
        # hybrid1 at the headline width.
        'c4_kirby_hybrid': convert_numpy(c4),
        'c5_kirby_top_tier': convert_numpy(c5),
        'c6_fluid_within_kirby_across': convert_numpy(c6),
        # hybrid2 at the headline width, same three conditions.
        'hybrid2_arms': convert_numpy(hybrid2_arms),
        # Every width, per fold, for both hybrids and the three conditions,
        # each with its recipe (allocation method, importance method, budget
        # in features, internal evaluator, representation count, augmentation
        # condition) and its training-row score from the per-fold search.
        'budget_sweep_per_fold': convert_numpy(
            [{'fold': r.get('fold'), **(r.get('by_budget') or {})}
             for r in fold_records]),
        # Per fold, both decided models: the per-fold search's 2 base
        # representations and 2 hybrids per budget, and the two hybrid deltas.
        # Report rows; nothing here is fitted.
        'selection_by_model_per_fold': convert_numpy(
            [{'fold': r.get('fold'),
              'selection': r.get('selection_by_model'),
              'hybrid_deltas': r.get('hybrid_deltas_by_model')}
             for r in fold_records]),
        'search_failures': convert_numpy(search_failures),
        'search_errors_per_fold': convert_numpy(search_errors),
        'privacy_notes': {
            'c1': 'each silo trains independently, no sharing',
            'c2': 'silos share predictions only',
            'c3': f'silos share predictions + one {"MCC" if is_clf else "RMSE"} scalar',
            'c4': 'centralized: all reps must be co-located',
            'c5': 'centralized: all reps must be co-located',
            'c6': 'intra-silo clustering local, cross-silo via federated FS',
        },
    }

    # Summary table
    print(f"\n{'─' * 70}")
    def _summary_row(key):
        """The metrics dict one summary line prints."""
        c1 = results['c1_best_single_silo']
        if key == 'c1_train_picked':
            return c1.get('train_picked', {})
        if key == 'c1_test_picked_ceiling':
            return c1.get('per_silo', {}).get(best_silo, {})
        if key.startswith('base'):
            return results['base_reps_fold_search'].get(key, {})
        return results[key]

    summary_items = [
        ('c1_train_picked',
         'C1: Best silo, picked on train'),
        ('c1_test_picked_ceiling',
         f'C1 CEILING: best on test ({silo_reps[best_silo]})'),
        ('base1', 'Fold search base rep 1'),
        ('base2', 'Fold search base rep 2'),
        ('c2_fluid_similarity', 'C2: FLuID similarity'),
        ('c3_fluid_quality', 'C3: FLuID quality'),
        ('c4_kirby_hybrid', 'C4: KIRBy hybrid1 (all)'),
        ('c5_kirby_top_tier', 'C5: KIRBy hybrid1 top-tier (3)'),
        ('c6_fluid_within_kirby_across', 'C6: FLuID→KIRBy hybrid1'),
    ]

    def _f(v, w):
        return f"{v:>{w}.4f}" if _is_num(v) else f"{'N/A':>{w}}"

    if is_clf:
        print(f"  {'Condition':<35} {'MCC':>8} {'BAC':>8} {'AUROC':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = _summary_row(key)
            print(f"  {name:<35} {_f(r.get('mcc'), 8)} {_f(r.get('bac'), 8)} "
                  f"{_f(r.get('auroc'), 8)}")
    else:
        print(f"  {'Condition':<35} {'RMSE':>8} {'Spearman':>10} {'R2':>8}")
        print(f"  {'─' * 61}")
        for key, name in summary_items:
            r = _summary_row(key)
            print(f"  {name:<35} {_f(r.get('rmse'), 8)} "
                  f"{_f(r.get('spearman_rho'), 10)} {_f(r.get('r2'), 8)}")

    return results


# ── One (model, seed) call saved as it finishes (2026-09-25) ─────────────────
# federated_helpers.run_all calls run() once per (decided model, fed_seed) and
# writes fed02_rep_diversity_silos/<dataset>.json only after the last. Each
# call's return value is written to fed02_rep_diversity_silos/<dataset>.partial.jsonl
# the moment it returns and read back on the next attempt
# (_checkpoint.resumable_run), keyed on the model, the seed and any keyword
# argument, under an identity that includes the input fingerprint, the plain
# config, the KIRBY_* settings and RESULT_LAYOUT (added 2026-10-05, so a call
# saved under the whole-dataset recipes is not read back). Inside a call each
# outer fold is saved too (<dataset>__folds.partial.jsonl). --force-rerun /
# KIRBY_FORCE_RERUN=1 discards both.
#
# The same sidecar and identity as _checkpoint.resumable_run, written out here
# so that one thing differs (2026-10-05): a call with any failed per-fold
# search ('search_failures' not empty) is returned and written by run_all but
# not saved as finished, so the next attempt retries those folds instead of
# reading the skipped arms back for good. resumable_run saves every call
# without a top-level 'error' key.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _key = {'fed_model': str(config.get('fed_model')
                             or fed_models_for(dataset_name)[0]),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    _cfg = ({k: v for k, v in config.items() if k not in _key}
            if isinstance(config, dict) else config)
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, _cfg,
                     layout=RESULT_LAYOUT))
    if ck.done(**_key):
        print(f'  [partial] {EXPERIMENT_DIR} {_key}: finished in an earlier '
              f'attempt, read back from {ck.path}')
        return ck.get(**_key)
    val = _run_one(data, config, output_dir, dataset_name, **kwargs)
    if isinstance(val, dict) and val.get('search_failures'):
        print(f'  [partial] {EXPERIMENT_DIR} {_key}: not saved as finished, '
              f'{len(val["search_failures"])} fold(s) had a failed per-fold '
              f'search; retried next attempt')
    else:
        ck.put(val, **_key)
    return val



if __name__ == '__main__':
    parser = create_experiment_parser('Fed02: Rep Diversity Silos')
    args = parser.parse_args()
    config = get_config(args)
    # The dataset's pruned representations only, as run_all loads them
    # (handoff 2.1, 2026-10-05). Until then a direct run loaded whatever
    # preset --reps named, and the per-fold search and the base arms ranked
    # over all of it; --reps is now ignored here.
    _general = (DATASET_REPS.get(args.dataset) or {}).get('general')
    if not _general:
        raise SystemExit(
            f"fed02: {args.dataset} has no DATASET_REPS[...]['general'] list")
    if args.dataset in ('herg', 'lhasa_herg'):
        from experiments.federated.federated_helpers import (
            load_lhasa_herg_data)
        data = load_lhasa_herg_data(rep_names=list(_general))
    else:
        data = load_data(args.dataset, rep_preset=','.join(_general),
                         rep_names=list(_general))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(
        results, args.output, 'fed02_rep_diversity_silos', args.dataset)
