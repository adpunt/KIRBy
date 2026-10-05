#!/usr/bin/env python3
"""Fed01: Noise Heterogeneity Across Silos + Federated Noise Detection.

Tests whether KIRBy's feature-level fusion handles heterogeneous noise across
federated silos better than FLuID's prediction-level ensemble. Two-class
labels (lhasa_herg as delivered; caco2 and logd cut at CLASSIFICATION_CUTOFFS
in federated_helpers). Reported metric: mcc, higher is better.

Conditions compared at each flip level, per outer fold:
    C0. Centralized KIRBy hybrid trained on NOISY labels (privacy-violating;
        matched to the federated conditions, NOT a clean-label oracle). One
        per hybrid: hybrid1, hybrid2.
    base1, base2. The model's 2 best single representations, trained
        centrally on the noisy labels (privacy-violating, like C0). Added
        2026-10-05 (handoff 2.1 R4).
    C1. Naive pool on Morgan fingerprints (privacy-violating baseline)
    C2. FLuID similarity-weighted ensemble (privacy-preserving)
    C3. FLuID quality-weighted ensemble (privacy-preserving)
    C4. FLuID published student pipeline (privacy-preserving). Its teachers
        are the run's model; its student is a random forest whatever the
        run's model is (see C4_STUDENT_MODEL).
    C5. KIRBy federated feature selection (privacy-preserving), per hybrid.
    C6. C5 with each silo's flagged rows removed first (privacy-preserving),
        per hybrid. Identical call to C5 otherwise, so C6 minus C5 is
        per-silo detection alone.

C1-C4 are the external comparator: FLuID's own ensembles on its own Morgan
fingerprints. C1-C3 and C4's teachers use the run's model; C4's student does
not (C4_STUDENT_MODEL). They see no search.

Since 2026-10-05 (handoff 2.1 R1-R4, 2.4) every recipe comes from ONE search
per outer fold on that fold's training rows (federated_helpers.
fed_fold_selection, the dataset's decided grid, pruning off), run on the
clean two-class labels (SEARCH_LABELS below) and shared by every flip level,
seed and budget. C0, C5 and C6 are built from its 2 best hybrids at each
decided budget; base1 and base2 are its 2 best single representations.

Plus: noise detection analysis, importance-noise correlation, smooth factor
sweep.

Usage:
    python fed01_noise_heterogeneity.py --dataset lhasa_herg
(runs federated_helpers.run_all for fed01 alone: the cluster's loader,
representations, config, models and seeds.)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import numpy as np
from scipy.stats import spearmanr
from experiments._shared import convert_numpy
from experiments.federated.federated_helpers import (
    smiles_to_fluid_df, cluster_training_only, inject_noise_per_silo,
    predict_fluid_similarity_ensemble,
    predict_fluid_quality_ensemble, federate_labels_and_train_student,
    federated_feature_selection, federated_noise_detection,
    stack_reps_for_distance,
    compute_clf_metrics, _resolve_match,
    smoke_grid, smoke_n_folds, smoke_k_silos,
    federated_budgets, fed_outer_folds, fed_fold_selection, recipe_reps,
    fed_models_for,
    fit_fed_classifier, make_fed_classifier, fed_to_kirby_model,
    binarise_labels,
)
from kirby.hybrid import apply_feature_selection
from kirby.scaling import FeatureScaler


# ── Constants ────────────────────────────────────────────────────────────────
# smoke_grid/smoke_n_folds/smoke_k_silos are no-ops unless KIRBY_SMOKE_N is set.

EXPERIMENT_DIR = 'fed01_noise_heterogeneity'

K = smoke_k_silos(5)               # number of silos/clusters
SIGMA_FRACS = smoke_grid([0, 0.05, 0.1, 0.15, 0.2, 0.3], [0, 0.2])
SMOOTH_FACTORS = smoke_grid([0, 0.01, 0.05, 0.1, 0.2], [0, 0.05])

DETECTORS = smoke_grid([
    'knn_k5', 'knn_k10', 'lof', 'distance_weighted',
    'cv_disagreement', 'bootstrap_ensemble',
    'zscore', 'prediction_error',
], ['knn_k5', 'bootstrap_ensemble'])

# What the experiment reports: mcc on the clean two-class labels of the
# held-out rows, higher is better. run() returns these two keys and run_all
# stamps the result file with them (dirty run bug 13: fed01 was stamped
# 'rmse, lower is better' on caco2 and logd).
REPORT_METRIC = 'mcc'
REPORT_LOWER_IS_BETTER = False

# What the per-fold search ranks recipes by, on the two-class labels. Named
# explicitly, with task_type='classification': metric='auto' resolves through
# the dataset registry, which has caco2 and logd as regression under rmse, and
# would run a regression search on 0/1 labels. lhasa_herg's registry entry is
# auc already, so its search is the same one any other experiment builds on
# that fold. The decided grid (DECISIONS.md) was measured on caco2's and
# logd's continuous labels; applying it to their two-class labels is accepted
# (handoff 2.0 chat 2, 2026-10-05).
SEARCH_METRIC = 'auc'

# Which labels the per-fold search runs on (2026-10-05, Claude's choice, for
# Adelaide to confirm). 'clean': the fold's training rows with their clean
# two-class labels; ONE search per fold, shared by every flip level, every
# seed and every budget, so the recipe source stays the same along the flip
# axis. Until 2026-10-05 flip level 0 read a fixed fallback recipe and every
# other level read a whole-dataset search on its own noisy labels, so the
# curve from 0 to 0.05 changed the noise and the recipe source at once.
# The alternative ('noisy': one search per flip level on that level's flipped
# training labels, flip levels x seeds x folds searches) is not implemented:
# fed_fold_selection takes no condition_suffix, and the in-process search
# memo (_shared.fold_searches) is not keyed on the labels, so two label
# vectors on one fold would be handed the same search.
SEARCH_LABELS = 'clean'

# Bumped whenever the shape or meaning of a stored unit changes, so the
# per-call and per-fold checkpoints written by older code are moved aside
# instead of read back (handoff 2.4 step 11).
# 2026-10-05b: outer folds drawn with the dataset's own task type (caco2 and
# logd changed rows), 'federated_hybrid' left out of noise_detection, C4
# stamped with its student model.
RESULT_FORMAT = 'per_fold_selection_2026-10-05b'

# Which task type the outer folds are drawn with (2026-10-05, Claude's
# choice, for Adelaide to confirm). 'dataset': the dataset's own task type
# and labels, as fed02/fed03 and the per-fold search store draw them, so
# fold i is the same molecules here and in every other federated file on
# that dataset (caco2 and logd: GroupKFold on scaffold groups; lhasa_herg:
# StratifiedGroupKFold, unchanged). The alternative, 'two_class', draws them
# with task_type 'classification' on the two-class labels
# (StratifiedGroupKFold): each fold keeps the active fraction of the whole
# dataset, but on caco2 and logd fold i is then different molecules from
# fold i of fed02/fed03. fed04 currently draws 'two_class'.
FOLD_TASK_TYPE = 'dataset'

# C4's student. federated_helpers.federate_labels_and_train_student takes an
# `algorithm` argument and ignores it: the student is always
# RandomForestClassifier(n_estimators=150, max_depth=15), FLuID's published
# student. rf is not a decided model on lhasa_herg, caco2 or logd (R1). Until
# the helper fits the student with fit_fed_classifier(algorithm), C4 is
# stamped with what it fits: teachers = the run's model, student = rf
# (2026-10-05; the result file used to file C4 under the run's model only).
C4_STUDENT_MODEL = 'rf'

# The hybrid arms: the 2 best hybrids of this fold's search at each budget.
_HYBRID_SLOTS = ('hybrid1', 'hybrid2')


def _fit_score(fed_model, X_tr, y_tr, X_te, y_te, seed):
    """Fit the run's model on (X_tr, y_tr) and score it on (X_te, y_te).

    Features are scaled on the training rows with FeatureScaler: continuous
    features standardised, 0/1 features passed through (answer 17,
    2026-10-05). The search scores every model on scaled features
    (find_best_hybrid._outer_fit_score), and svm and dnn are not scale-free,
    so an unscaled fit here would score a different model from the one the
    recipe was chosen for. Trees (lightgbm, xgb) are unaffected.
    """
    sc = FeatureScaler().fit(X_tr)
    clf = fit_fed_classifier(fed_model, sc.transform(X_tr), y_tr, seed=seed)
    X_te_s = sc.transform(X_te)
    prob = clf.predict_proba(X_te_s)
    prob_pos = prob[:, 1] if prob.shape[1] == 2 else None
    m = compute_clf_metrics(y_te, clf.predict(X_te_s), prob_pos)
    m['n_features'] = int(X_tr.shape[1])
    return m


def _inject_flips(labels_train, silo_indices, noise_levels, seed):
    """Per-silo label flips, including silos that hold one class only.

    inject_noise_per_silo hands each silo to noiseInject's uniform flip, which
    draws the replacement class from the classes present in THAT silo's
    labels. A silo holding one class has no other class to draw, and numpy
    raises "'a' cannot be empty unless no samples are taken". That is why
    smoothing factor 0 failed on every fold of the dirty run (2026-09-28,
    traceback in the herg_ki result file): with no smoothing the k-means
    clusters are pure, and some hold inactives only.

    The label is two-class everywhere in fed01, so the other class is known.
    A one-class silo is flipped here instead: exactly round(level x rows)
    rows, chosen by a seeded draw, set to 1 - label. Every other silo goes
    through inject_noise_per_silo unchanged. Returns (labels_noisy,
    noise_mask_global, noise_mask_per_silo, notes), notes listing every silo
    flipped here so the result file says which.
    """
    labels_train = np.asarray(labels_train)
    two_class, one_class = {}, {}
    for s, lv in noise_levels.items():
        idx = silo_indices[s]
        if lv <= 0 or len(np.unique(labels_train[idx])) >= 2:
            two_class[s] = lv
        else:
            one_class[s] = lv
    y_noisy, mask, per_silo = inject_noise_per_silo(
        labels_train, silo_indices, two_class, noise_type='label_flip',
        seed=seed)
    notes = []
    for s, lv in one_class.items():
        idx = np.asarray(silo_indices[s], dtype=int)
        m = np.zeros(len(labels_train), dtype=bool)
        if len(idx) == 0:
            per_silo[s] = m
            notes.append({'silo': int(s), 'n_rows': 0, 'level': float(lv),
                          'n_flipped': 0, 'note': 'empty silo'})
            continue
        n_flip = int(round(float(lv) * len(idx)))
        rng = np.random.RandomState(
            (int(seed) * 1000003 + int(s) * 7919
             + int(round(float(lv) * 1000))) & 0x7FFFFFFF)
        pick = (idx[rng.choice(len(idx), size=n_flip, replace=False)]
                if n_flip > 0 else idx[:0])
        y_noisy[pick] = 1 - y_noisy[pick]
        m[pick] = True
        mask |= m
        per_silo[s] = m
        notes.append({'silo': int(s), 'n_rows': int(len(idx)),
                      'class_held': int(labels_train[idx][0]),
                      'level': float(lv), 'n_flipped': n_flip,
                      'note': 'one class only; flipped to the other class '
                              'here, exact count'})
    return y_noisy, mask, per_silo, notes


def _hybrid_record(h):
    """What the result file keeps about one selected hybrid (handoff 2.4 step
    11): its recipe and its score on this fold's training rows."""
    bp = h.get('blueprint')
    return {
        'label': h.get('label'),
        'allocation_method': h.get('allocation_method',
                                   getattr(bp, 'allocation_method', None)),
        'importance_method': h.get('importance_method',
                                   getattr(bp, 'importance_method', None)),
        'budget': h.get('budget', getattr(bp, 'budget', None)),
        'internal_evaluator': h.get('internal_evaluator',
                                    getattr(bp, 'internal_evaluator', None)),
        'rep_count': h.get('rep_count', getattr(bp, 'rep_count', None)),
        'aug_condition': h.get('aug_condition', 'none'),
        'n_features': h.get('n_features'),
        'score_train_rows': h.get('score_mean'),
        'fold_scores_train_rows': h.get('fold_scores'),
    }


def _condition_naive_pool(train_fps, train_labels_noisy, test_fps,
                          test_labels_clean, fed_model, fed_seed=0):
    """Condition 1: Merge all silo data, train single model.

    Privacy: VIOLATES (all data pooled). Baseline only.
    """
    clf = fit_fed_classifier(
        fed_model, train_fps, train_labels_noisy, seed=fed_seed)
    pred_probs = clf.predict_proba(test_fps)
    pred_classes = clf.predict(test_fps)
    prob_positive = pred_probs[:, 1] if pred_probs.shape[1] == 2 else None
    return compute_clf_metrics(test_labels_clean, pred_classes, prob_positive)


def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed01: Noise heterogeneity across silos, for one (model, seed)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED01: NOISE HETEROGENEITY ACROSS SILOS")
    print("=" * 70)

    if SEARCH_LABELS != 'clean':
        raise NotImplementedError(
            f"fed01: SEARCH_LABELS={SEARCH_LABELS!r}; only 'clean' is built "
            f"(see the comment at the constant)")

    # The run's model is one of the dataset's two decided models (R1,
    # 2026-10-05), handed in by run_all as config['fed_model'] in FLuID's
    # names. No 'rf' default: rf is not decided on any of fed01's datasets.
    _decided_fed = fed_models_for(dataset_name)
    fed_model = config.get('fed_model') or _decided_fed[0]
    if fed_model not in _decided_fed:
        raise ValueError(
            f"fed01 {dataset_name}: model {fed_model!r} is not one of the "
            f"dataset's decided models {_decided_fed} (DECISIONS.md Step 1)")
    kirby_model = fed_to_kirby_model(fed_model)
    fed_seed = config.get('fed_seed', 0)

    reps = data['reps']
    labels = data['labels']
    smiles = data['smiles']

    # ── Phase 1: Setup ───────────────────────────────────────────────────
    print("\n--- Phase 1: Setup ---")

    # Binarize labels (skip if already binary). The Lhasa loader sets
    # data['is_binary'], but the regular load_data('herg', ...) path returns
    # 0/1 labels WITHOUT that flag — so also detect already-binary labels by
    # value (unique values ⊆ {0,1}). Continuous endpoints take their own
    # cutoff from CLASSIFICATION_CUTOFFS in federated_helpers.
    labels_binary, cutoff_note = binarise_labels(
        dataset_name, labels, is_binary=bool(data.get('is_binary')))
    n_active = int(labels_binary.sum())
    print(f"  {dataset_name}: {len(labels)} samples, "
          f"{n_active} active, {len(labels_binary) - n_active} inactive "
          f"({cutoff_note})")

    # Compute Morgan FPs for all molecules (to get valid_mask)
    fluid_df_all, valid_mask = smiles_to_fluid_df(
        smiles, labels_binary, pki_threshold=0.5)

    # Filter to valid molecules only
    groups = data.get('groups')
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        print(f"  WARNING: {n_invalid} molecules failed RDKit parsing, "
              f"excluded from experiment")
        valid_idx = np.where(valid_mask)[0]
        reps = {r: arr[valid_idx] for r, arr in reps.items()}
        labels_binary = labels_binary[valid_idx]
        labels = labels[valid_idx]
        if groups is not None:
            groups = groups[valid_idx]
        smiles = [smiles[i] for i in valid_idx]

    n_samples = len(labels_binary)

    # Augmentation is cut on lhasa_herg, caco2 and logd (DECISIONS.md Step 8)
    # and run_all no longer loads it there. No arm here builds added features.
    if data.get('augmentations'):
        print(f"  {dataset_name}: augmentation is cut (DECISIONS.md Step 8); "
              f"the {len(data['augmentations'])} added-feature sets handed in "
              f"are not used")

    from kirby.find_best_hybrid import BUILTIN_METRICS
    search_metric_fn, search_lower_is_better = BUILTIN_METRICS[SEARCH_METRIC]

    def _noise_seed(fold_i):
        """Which labels get flipped, for this run and this fold.

        run_all calls fed01 once per (decided model, seed): two models at
        three seeds. The spread across the three seeds is a spread over flip
        draws only if each seed flips different labels, so fed_seed is mixed
        in (inject_noise_per_silo's default of 42 used to give all three the
        same flips). The fold index is mixed in so one draw is not reused
        across folds. Since 2026-10-05 the folds themselves do not depend on
        fed_seed (fed_outer_folds, seed 42).
        """
        return (int(fed_seed) * 7919 + int(fold_i) * 104729 + 42) & 0x7FFFFFFF

    # The widths C0, C5 and C6 are built at: the dataset's decided general
    # budgets (lhasa_herg 100-2000, caco2 and logd 500-2000 features; R3,
    # 2026-10-05). Until 2026-10-05 the sweep was 50-1000 on every dataset and
    # every summary table read 100 only.
    BUDGETS = [int(b) for b in federated_budgets(dataset_name)]

    # Outer folds from ONE setting, config['n_splits'] (handoff 2.4 step 14),
    # through smoke_n_folds so KIRBY_FED_FOLDS / KIRBY_SMOKE_N still narrow a
    # local run. 1 gives a single grouped 80/20 split.
    if groups is None:
        # Hard-fail per project rule (feedback_no_random_splits.md).
        raise ValueError(
            "fed01: groups is None — refusing to run without scaffold groups "
            "(no random splits).")
    # n_splits=1 stays 1: smoke_n_folds returns at least 2 under
    # KIRBY_SMOKE_N / KIRBY_FED_FOLDS, which would turn the configured single
    # split into 2 folds (2026-10-05).
    _n_splits = int(config['n_splits'])
    N_FOLDS = _n_splits if _n_splits <= 1 else smoke_n_folds(_n_splits)
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < max(2, N_FOLDS):
        raise ValueError(
            f"fed01: only {n_unique_groups} unique scaffold groups for "
            f"{N_FOLDS} outer fold(s) — refusing to split.")
    # Seed 42 (nested_cv_outer_folds). Drawn with seed=fed_seed until
    # 2026-10-05, so the spread over seeds mixed the model's seed with which
    # molecules were held out. FOLD_TASK_TYPE says which task type and labels
    # the splitter sees. Until 2026-10-05b caco2 and logd were drawn as
    # 'classification' on the two-class labels (StratifiedGroupKFold), which
    # is not the rows fed02/fed03 and the per-fold search store hold out
    # (GroupKFold); the comment here claimed they were the same.
    if FOLD_TASK_TYPE == 'dataset':
        _fold_task = data.get('task_type') or 'classification'
        _fold_labels = (labels if _fold_task != 'classification'
                        else labels_binary)
    elif FOLD_TASK_TYPE == 'two_class':
        _fold_task, _fold_labels = 'classification', labels_binary
    else:
        raise ValueError(f"fed01: FOLD_TASK_TYPE={FOLD_TASK_TYPE!r}")
    _folds = fed_outer_folds(reps, _fold_labels, groups, N_FOLDS, _fold_task)

    # ── Per-fold checkpoint (dirty-run bug 12, 2026-10-05) ──
    # run() saves each (model, seed) call when it returns; a call is up to
    # N_FOLDS x flip levels x budgets of work, so each fold is now written the
    # moment it finishes and read back on the next attempt. Same identity
    # rules as _checkpoint.resumable_run, plus RESULT_FORMAT.
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    _cfg_ident = {k: v for k, v in config.items()
                  if k not in ('fed_model', 'fed_seed')}
    fold_ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name,
                           suffix='folds'),
        run_identity(EXPERIMENT_DIR, dataset_name, data, _cfg_ident,
                     result_format=RESULT_FORMAT,
                     search_labels=SEARCH_LABELS,
                     search_metric=SEARCH_METRIC))

    # Morgan FPs for FLuID conditions
    all_fps = np.asarray(list(fluid_df_all.FP))

    per_fold = []
    for fold_idx, (train_idx, test_idx) in enumerate(_folds):
        _ck_key = dict(fed_model=str(fed_model), fed_seed=int(fed_seed),
                       fold=int(fold_idx))
        if fold_ck.done(**_ck_key):
            print(f"\n========== FOLD {fold_idx + 1}/{N_FOLDS}: read back "
                  f"from {fold_ck.path} ==========")
            per_fold.append(fold_ck.get(**_ck_key))
            continue
        rec = _run_fold(
            fold_idx=fold_idx, n_folds=N_FOLDS, train_idx=train_idx,
            test_idx=test_idx, reps=reps, labels_binary=labels_binary,
            groups=groups, fluid_df_all=fluid_df_all, all_fps=all_fps,
            dataset_name=dataset_name, fed_model=fed_model,
            kirby_model=kirby_model, fed_seed=fed_seed, budgets=BUDGETS,
            noise_seed=_noise_seed(fold_idx),
            search_metric_fn=search_metric_fn,
            search_lower_is_better=search_lower_is_better)
        rec = convert_numpy(rec)
        # A fold whose per-fold search failed (its KIRBy arms skipped) is not
        # stored: the next attempt builds the search again instead of reading
        # 'arms skipped' back for good (a Seagate drop or a killed worker is
        # not a result; 2026-10-05).
        if rec.get('search_ok'):
            fold_ck.put(rec, **_ck_key)
        else:
            print(f"  fold {fold_idx}: per-fold search failed; fold not "
                  f"checkpointed, retried on the next attempt")
        per_fold.append(rec)

    # ── Aggregate across folds (mean ± std on numeric leaves) ─────────────
    def _aggregate(fold_dicts):
        """Recursively mean/std numeric leaves across a list of nested dicts.

        Non-numeric leaves: take the first fold's value as representative.
        Lists of numerics: aggregate element-wise. Mixed/string lists: pass
        through from fold 0. Missing keys in some folds are skipped. A leaf
        measured on one fold only gets std None and 'one fold, no spread'
        (2026-10-05: n_splits may be 1).
        """
        if not fold_dicts:
            return {}
        first = fold_dicts[0]
        if isinstance(first, dict):
            keys = []
            for fd in fold_dicts:
                if isinstance(fd, dict):
                    keys += [k for k in fd.keys() if k not in keys]
            out = {}
            for k in keys:
                sub = [fd[k] for fd in fold_dicts
                       if isinstance(fd, dict) and k in fd]
                out[k] = _aggregate(sub)
            return out
        numeric = []
        for v in fold_dicts:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and np.isnan(v)
            ):
                numeric.append(float(v))
        non_bool = [v for v in fold_dicts if not isinstance(v, bool)]
        if numeric and non_bool and all(
            isinstance(v, (int, float)) for v in non_bool
        ):
            arr = np.asarray(numeric, dtype=float)
            if len(arr) == 1:
                return {'mean': float(arr[0]), 'std': None,
                        'spread_note': 'one fold, no spread',
                        'n_folds': 1, 'per_fold': numeric}
            return {'mean': float(np.mean(arr)),
                    'std': float(np.std(arr, ddof=0)),
                    'n_folds': int(len(arr)),
                    'per_fold': numeric}
        if isinstance(first, list) and all(
            isinstance(v, list) and len(v) == len(first) for v in fold_dicts
        ) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for v in fold_dicts for x in v
        ):
            arr = np.asarray(fold_dicts, dtype=float)
            return {'mean': arr.mean(axis=0).tolist(),
                    'std': (arr.std(axis=0, ddof=0).tolist()
                            if arr.shape[0] > 1 else None),
                    'n_folds': int(arr.shape[0])}
        return first

    # Per-fold descriptions are left out of the aggregate: each fold has its
    # own recipe, and averaging a budget or keeping fold 0's allocation name
    # would read as one recipe. They stay in 'per_fold' and
    # 'selection_by_fold'.
    _NOT_AGGREGATED = ('recipe', 'traceback', 'detection_results',
                       'silo_failures', 'representations', 'rep_count_note',
                       'same_silo_build_as', 'rep')

    def _strip(o):
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items()
                    if k not in _NOT_AGGREGATED}
        return o

    primary_agg = _aggregate([_strip(f['primary_sweep']) for f in per_fold])
    detection_agg = _aggregate([f['noise_detection'] for f in per_fold])
    correlation_agg = _aggregate(
        [_strip(f['importance_noise_correlation']) for f in per_fold])
    smooth_agg = _aggregate([_strip(f['smooth_factor_sweep'])
                             for f in per_fold])

    # ── Phase 6: Assemble results ─────────────────────────────────────────
    results = {
        'experiment': EXPERIMENT_DIR,
        'dataset': dataset_name,
        # The reported metric (run_all stamps the file with these two).
        'metric_name': REPORT_METRIC,
        'lower_is_better': REPORT_LOWER_IS_BETTER,
        'config': {
            'k': K,
            'classification_cutoff': cutoff_note,
            'sigma_fracs': SIGMA_FRACS,
            'smooth_factors': SMOOTH_FACTORS,
            'n_samples': n_samples,
            'n_folds': N_FOLDS,
            'fold_seed': 42,
            'fold_task_type': FOLD_TASK_TYPE,
            'fold_sizes': [{'fold': f['fold'], 'n_train': f['n_train'],
                            'n_test': f['n_test']} for f in per_fold],
            'detectors': DETECTORS,
            'budgets_swept': BUDGETS,
            'fed_model': fed_model,
            'kirby_model': kirby_model,
            'c4_teacher_model': fed_model,
            'c4_student_model': C4_STUDENT_MODEL,
            'search_labels': SEARCH_LABELS,
            'search_metric': SEARCH_METRIC,
            'search_lower_is_better': search_lower_is_better,
            'result_format': RESULT_FORMAT,
        },
        # Per fold: which 2 base representations and which 2 hybrids at each
        # budget this fold's search chose, with their scores on the fold's
        # training rows (selection_scope 'fold_train_rows').
        'selection_by_fold': [f['selection'] for f in per_fold],
        # Every per-fold search that failed (dataset, fold, model, error);
        # its KIRBy arms were skipped on that fold, never filled.
        'selection_failures': [x for f in per_fold
                               for x in f['selection_failures']],
        # Every silo that produced no features, per arm (federated_helpers
        # gives it 0 features and score None since 2026-10-05).
        'silo_failures': [x for f in per_fold for x in f['silo_failures']],
        # Silos that held one class and were flipped by _inject_flips.
        'single_class_silo_flips': [x for f in per_fold
                                    for x in f['single_class_silo_flips']],
        'primary_sweep': primary_agg,
        'noise_detection': detection_agg,
        'importance_noise_correlation': correlation_agg,
        'smooth_factor_sweep': smooth_agg,
        'per_fold': per_fold,
        # False when any fold's per-fold search failed; run() then does not
        # store this (model, seed) call, so it is computed again next time.
        'all_searches_ok': all(f.get('search_ok') for f in per_fold),
    }

    _print_summary(primary_agg, BUDGETS, N_FOLDS)
    return results


def _fmt(agg):
    """'mean±std' for an aggregated mcc leaf; one fold says so."""
    if not isinstance(agg, dict) or 'mean' not in agg:
        return f"{'N/A':>13}"
    if agg.get('std') is None:
        return f"{agg['mean']:>6.3f} (1 fold)"
    return f"{agg['mean']:>6.3f}±{agg['std']:.3f}"


def _mcc(cond):
    return cond.get('mcc') if isinstance(cond, dict) else None


def _print_summary(primary_agg, budgets, n_folds):
    """MCC per flip level: the arms that do not depend on the budget once,
    then the hybrid arms at every decided budget (2026-10-05; the summary
    used to read 100 features only)."""
    spread = ('mean ± std across folds' if n_folds > 1
              else 'one fold, no spread')
    print(f"\n--- Summary: mcc on held-out rows, higher is better "
          f"({spread}) ---")
    cols = [('c1_naive_pool', 'C1 Pool'), ('c2_fluid_similarity', 'C2 Sim'),
            ('c3_fluid_quality', 'C3 Qual'), ('c4_fluid_student', 'C4 Stud'),
            ('base1', 'base1'), ('base2', 'base2')]
    print(f"  {'Flip':<6} " + ' '.join(f"{c[1]:<15}" for c in cols))
    for sigma in SIGMA_FRACS:
        sr = primary_agg.get(str(sigma), {})
        vals = []
        for key, _ in cols:
            if key.startswith('base'):
                vals.append(_fmt(_mcc((sr.get('base') or {}).get(key))))
            else:
                vals.append(_fmt(_mcc(sr.get(key))))
        print(f"  {sigma:<6} " + ' '.join(f"{v:<15}" for v in vals))
    hcols = [(f'{c}__{h}', f'{c[:2].upper()} {h}')
             for c in ('c0_centralized_kirby', 'c5_kirby_federated',
                       'c6_kirby_noise_detect')
             for h in _HYBRID_SLOTS]
    for b in budgets:
        print(f"\n  budget {b} features")
        print(f"  {'Flip':<6} " + ' '.join(f"{c[1]:<15}" for c in hcols))
        for sigma in SIGMA_FRACS:
            arms = ((primary_agg.get(str(sigma), {}).get('by_budget') or {})
                    .get(str(b)) or {})
            print(f"  {sigma:<6} " + ' '.join(
                f"{_fmt(_mcc(arms.get(k))):<15}" for k, _ in hcols))


def _run_fold(*, fold_idx, n_folds, train_idx, test_idx, reps, labels_binary,
              groups, fluid_df_all, all_fps, dataset_name, fed_model,
              kirby_model, fed_seed, budgets, noise_seed, search_metric_fn,
              search_lower_is_better):
    """Everything fed01 measures on one outer fold, as one plain dict."""
    print(f"\n========== FOLD {fold_idx + 1}/{n_folds} ==========")
    test_labels_clean = labels_binary[test_idx]
    print(f"  Split: {len(train_idx)} train / {len(test_idx)} test")

    rec = {'fold': int(fold_idx), 'n_train': int(len(train_idx)),
           'n_test': int(len(test_idx)), 'selection_failures': [],
           'silo_failures': [], 'single_class_silo_flips': []}

    # ── The per-fold search and its selection (R2, R4; 2026-10-05) ──
    # ONE search on this fold's training rows, clean two-class labels, the
    # dataset's decided grid, pruning off. It replaces the whole-dataset
    # searches this file read through get_best_search_for_downstream and
    # get_rule_hybrids (one per flip level and seed, every row of the
    # dataset), the 'single best lookup' and the fixed fallback recipe
    # (performance_weighted, treeshap, rf). A failed search is recorded and
    # the KIRBy arms are skipped on this fold; nothing is filled in.
    sel_by_model = {}
    try:
        sel_by_model = fed_fold_selection(
            reps, labels_binary, groups, train_idx=train_idx,
            fold_i=fold_idx, n_folds=n_folds, dataset_name=dataset_name,
            budgets=budgets, models=[kirby_model],
            task_type='classification', metric=SEARCH_METRIC)
    except Exception as e:
        from kirby.result_cache import StaleResultDB as _Stale
        if isinstance(e, _Stale):
            raise
        rec['selection_failures'].append({
            'dataset': dataset_name, 'fold': int(fold_idx),
            'model': kirby_model, 'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc()})
    sel = sel_by_model.get(kirby_model)
    for err in sel_by_model.get('_errors') or []:
        rec['selection_failures'].append({
            'dataset': dataset_name, 'fold': int(fold_idx),
            'model': kirby_model,
            'error': (err.get('error') or err.get('error_type')
                      if isinstance(err, dict) else str(err)),
            'detail': err})
    if sel is None and not rec['selection_failures']:
        rec['selection_failures'].append({
            'dataset': dataset_name, 'fold': int(fold_idx),
            'model': kirby_model,
            'error': 'the per-fold search returned no selection'})
    # Whether this fold may be checkpointed: the KIRBy arms were built. Errors
    # of single configurations inside a search that still returned a
    # selection are recorded above and do not block the checkpoint.
    rec['search_ok'] = sel is not None

    # base: [(slot, rep name, training-row score)]
    base_slots = []
    # hybrids_by_budget: {budget: [(slot, top_hybrids entry)]}
    hybrids_by_budget = {int(b): [] for b in budgets}
    if sel is not None:
        _bd = {e.get('rep'): e for e in
               ((sel.get('base_detail') or {}).get('selected') or [])}
        for i, r in enumerate(sel.get('base') or []):
            base_slots.append((f'base{i + 1}', r,
                               (_bd.get(r) or {}).get('score')))
        for b in budgets:
            ents = ((sel.get('by_budget') or {}).get(int(b)) or {}).get(
                'hybrids') or []
            hybrids_by_budget[int(b)] = [
                (_HYBRID_SLOTS[i], h) for i, h in
                enumerate(ents[:len(_HYBRID_SLOTS)])]
    rec['selection'] = {
        'fold': int(fold_idx),
        'model': kirby_model,
        'selection_scope': 'fold_train_rows' if sel is not None else None,
        'search_labels': SEARCH_LABELS,
        'search_metric': (sel or {}).get('search_metric') or SEARCH_METRIC,
        'search_lower_is_better': (sel or {}).get(
            'search_lower_is_better', search_lower_is_better),
        'base': [{'slot': s, 'rep': r, 'score_train_rows': sc}
                 for s, r, sc in base_slots],
        'base_shortfall': ((sel or {}).get('base_detail') or {}).get(
            'shortfall_reason'),
        'hybrids_by_budget': {
            str(b): {
                'hybrids': [dict(_hybrid_record(h), slot=s)
                            for s, h in hybrids_by_budget[int(b)]],
                'hybrids_skipped_identical_features': [
                    x.get('label') if isinstance(x, dict) else str(x)
                    for x in ((((sel or {}).get('by_budget') or {})
                               .get(int(b)) or {})
                              .get('hybrids_skipped_identical_features')
                              or [])],
                'missing': [
                    f'{s}: the per-fold search gave fewer than '
                    f'{j + 1} hybrids with distinct features at {b}'
                    for j, s in enumerate(_HYBRID_SLOTS)
                    if j >= len(hybrids_by_budget[int(b)])],
            } for b in budgets},
    }

    # Two hybrids can differ in the search and still build the same features
    # per silo: federated_feature_selection takes only the allocation method,
    # importance method, internal evaluator, budget and the representations.
    # Recorded, not dropped.
    def _silo_build_key(bp):
        return (bp.allocation_method, bp.importance_method,
                _resolve_match(bp.internal_evaluator, kirby_model))

    # CLUSTER TRAINING DATA ONLY — matching FLuID's
    # cluster_data_space(trainingData). Test molecules never influence
    # K-Means centroids.
    t0 = time.time()
    train_fluid_df = fluid_df_all.iloc[train_idx]
    train_silo_indices, cluster_data, _ = cluster_training_only(
        train_fluid_df, k=K, smooth_factor=0.05, seed=fed_seed)
    print(f"  Clustering (train only): K={K}, {time.time() - t0:.1f}s")
    train_labels_binary = labels_binary[train_idx]
    for s in range(1, K + 1):
        idx = train_silo_indices[s]
        n_act = int(train_labels_binary[idx].sum())
        print(f"    Silo {s}: {len(idx)} samples "
              f"({n_act} active, {len(idx) - n_act} inactive)")

    train_fps = all_fps[train_idx]
    test_fps = all_fps[test_idx]
    train_reps = {r: arr[train_idx] for r, arr in reps.items()}
    test_reps = {r: arr[test_idx] for r, arr in reps.items()}
    train_groups = groups[train_idx]

    def _silo_arm(bp, y_tr, silo_idx, rows, budget, ctx, failures):
        """C5 / C6 for one hybrid: per-silo selection on the recipe's own
        representations, fitted centrally on the selected features, scored on
        the held-out rows. `rows` is None for all training rows, else the
        positions kept (C6)."""
        tr = (train_reps if rows is None
              else {r: a[rows] for r, a in train_reps.items()})
        g = train_groups if rows is None else train_groups[rows]
        # The recipe's representations, cut to its representation count by
        # this fold's own search scores (federated_helpers.recipe_reps).
        kept, rc_note = recipe_reps(tr, bp, context=ctx)
        X_tr, sb, ss, sfis = federated_feature_selection(
            kept, y_tr, silo_idx, K, budget=int(budget),
            allocation_method=bp.allocation_method,
            importance_method=bp.importance_method,
            # 'match' resolves to the run's own model, never rf.
            internal_evaluator=_resolve_match(bp.internal_evaluator,
                                              kirby_model),
            downstream_model=kirby_model, groups=g,
            task_type='classification',
            metric_fn=search_metric_fn,
            lower_is_better=search_lower_is_better,
            failures=failures)
        te = {r: test_reps[r] for r in kept}
        # Every silo that produced training features must produce the same
        # features on the held-out rows; a failure here raises instead of
        # being dropped (it used to `continue`, leaving the test matrix
        # narrower than the training one).
        parts = [apply_feature_selection(te, sfis[s])
                 for s in range(1, K + 1) if sfis.get(s) is not None]
        m = _fit_score(fed_model, X_tr, y_tr, np.hstack(parts),
                       test_labels_clean, fed_seed)
        m['silo_budgets'] = convert_numpy(sb)
        # None for a silo that produced no features (not 0.0).
        m['silo_scores'] = convert_numpy(ss)
        m['silo_failures'] = {str(s): why for s, why in failures.items()}
        m['representations'] = sorted(kept)
        if rc_note:
            m['rep_count_note'] = rc_note
        return m

    # ── Phase 2: Primary noise sweep ─────────────────────────────────────
    print("\n  --- Phase 2: Primary noise sweep ---")
    primary_sweep = {}

    for sigma_frac in SIGMA_FRACS:
        print(f"\n  === Sigma={sigma_frac} (noise linspace 0..{sigma_frac} "
              f"across K={K} silos) ===")

        # Heterogeneous linspace: silo 1 = 0.0, silo K = sigma_frac, others
        # interpolated. Phase 4 reads this sigma's levels for its correlation.
        silo_noise_arr = np.linspace(0.0, sigma_frac, K).tolist()
        noise_levels = {s: float(silo_noise_arr[s - 1])
                        for s in range(1, K + 1)}

        train_labels_clean = labels_binary[train_idx]
        train_labels_noisy, noise_mask_global, _, _flip_notes = \
            _inject_flips(train_labels_clean, train_silo_indices,
                          noise_levels, seed=noise_seed)
        for _n in _flip_notes:
            rec['single_class_silo_flips'].append(
                dict(_n, phase='primary', sigma=float(sigma_frac)))

        n_flipped = int(noise_mask_global.sum())
        print(f"    {n_flipped}/{len(train_idx)} labels flipped in training")

        sigma_results = {'n_flipped': n_flipped, 'base': {},
                         'by_budget': {str(b): {} for b in budgets}}

        # Rebuild training cluster_data with noisy labels.
        # FLuID's cluster_data[1..K] are smoothed views (k-NN bleed across
        # clusters when smooth_factor>0). Re-slicing by CLUSTER would drop
        # those smoothed rows — instead, copy each existing view and
        # overwrite CLASS/ACTIVITY for the rows whose index appears in our
        # noise table (P1-8 fix).
        train_df_noisy = cluster_data[0].copy()
        train_df_noisy['CLASS'] = train_labels_noisy
        train_df_noisy['ACTIVITY'] = np.where(
            train_labels_noisy == 1, 'Active', 'Inactive')
        cluster_data_noisy = [train_df_noisy]
        noisy_lookup = train_df_noisy[['CLASS', 'ACTIVITY']]
        for s in range(1, K + 1):
            smoothed_view = cluster_data[s].copy()
            shared = smoothed_view.index.intersection(noisy_lookup.index)
            if len(shared) > 0:
                smoothed_view.loc[shared, 'CLASS'] = noisy_lookup.loc[shared, 'CLASS']
                smoothed_view.loc[shared, 'ACTIVITY'] = noisy_lookup.loc[shared, 'ACTIVITY']
            cluster_data_noisy.append(smoothed_view)

        # Train FLuID teachers on training data per cluster, with the run's
        # model (FLuID's own for rf/xgb, KIRBy's estimator otherwise).
        teacher_models = []
        cluster_data_train = [None]  # Index 0 placeholder (Teacher 0 unused)
        for t in range(0, K + 1):
            cd = cluster_data_noisy[t]
            # A silo needs two rows AND both classes. fit_fed_classifier
            # puts every silo through RandomUnderSampler, which raises on a
            # single class. The ensembles skip a None teacher.
            if len(cd) < 2 or len(np.unique(np.asarray(cd.CLASS))) < 2:
                teacher_models.append((None, 0, cd))
                if t > 0:
                    cluster_data_train.append(cd)
                continue
            X_t = np.asarray(list(cd.FP))
            y_t = np.asarray(list(cd.CLASS))
            clf = fit_fed_classifier(fed_model, X_t, y_t, seed=fed_seed)
            teacher_models.append((clf, len(cd), cd))
            if t > 0:
                cluster_data_train.append(cd)

        # -- base1, base2: the 2 best single representations (2026-10-05) --
        # Centralised on the noisy labels, like C0. Do not depend on budget.
        for _slot, _rep, _sc in base_slots:
            print(f"    {_slot} [{_rep}]: single representation...", end=' ')
            try:
                m = _fit_score(fed_model, train_reps[_rep], train_labels_noisy,
                               test_reps[_rep], test_labels_clean, fed_seed)
                m['rep'] = _rep
                m['score_train_rows'] = _sc
                sigma_results['base'][_slot] = m
                print(f"MCC={m['mcc']:.3f}")
            except Exception as e:
                sigma_results['base'][_slot] = {
                    'rep': _rep, 'error': str(e),
                    'traceback': traceback.format_exc()}
                print(f"FAILED: {e}")

        # -- Condition 0: Centralized KIRBy hybrid trained on NOISY labels --
        # NOT a clean-label oracle / upper bound: C0 sees the SAME noisy labels
        # the federated conditions see, with centralized access to all silo
        # features. One C0 per hybrid at each budget, rebuilt on this fold's
        # training rows with the whole recipe (build_fold_hybrid: allocation,
        # importance, evaluator, budget, representation count ranked by this
        # fold's search scores) and the run's model.
        from experiments._shared import (build_fold_hybrid,
                                         fold_hybrid_scoring_kwargs)
        for b in budgets:
            for _slot, h in hybrids_by_budget[int(b)]:
                bp = h['blueprint']
                print(f"    C0 [{_slot} @ {b}]: Centralized KIRBy (noisy "
                      f"labels)...", end=' ')
                try:
                    X_h_tr, X_h_te, _fi = build_fold_hybrid(
                        train_reps, test_reps, train_labels_noisy, bp,
                        groups=train_groups,
                        **fold_hybrid_scoring_kwargs(
                            bp, model=make_fed_classifier(fed_model,
                                                          seed=fed_seed),
                            metric_fn=search_metric_fn,
                            lower_is_better=search_lower_is_better))
                    c0 = _fit_score(fed_model, X_h_tr, train_labels_noisy,
                                    X_h_te, test_labels_clean, fed_seed)
                    c0['recipe'] = _hybrid_record(h)
                    sigma_results['by_budget'][str(b)][
                        f'c0_centralized_kirby__{_slot}'] = c0
                    print(f"MCC={c0['mcc']:.3f}")
                except Exception as e:
                    sigma_results['by_budget'][str(b)][
                        f'c0_centralized_kirby__{_slot}'] = {
                        'error': str(e), 'traceback': traceback.format_exc(),
                        'recipe': _hybrid_record(h)}
                    print(f"FAILED: {e}")
                    traceback.print_exc()

        # -- Condition 1: Naive pool --
        print("    C1: Naive pool...", end=' ')
        try:
            c1 = _condition_naive_pool(
                train_fps, train_labels_noisy, test_fps, test_labels_clean,
                fed_model=fed_model, fed_seed=fed_seed)
            sigma_results['c1_naive_pool'] = c1
            print(f"MCC={c1['mcc']:.3f}")
        except Exception as e:
            sigma_results['c1_naive_pool'] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")

        # -- Condition 2: FLuID similarity ensemble --
        # Pass train-only cluster data to avoid test leakage in KNN
        print("    C2: FLuID similarity...", end=' ')
        try:
            pred_cls, pred_prob, tw = predict_fluid_similarity_ensemble(
                teacher_models, cluster_data_train, test_fps, K)
            c2 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c2['teacher_weights_mean'] = convert_numpy(
                {str(s): float(tw[:, s - 1].mean()) for s in range(1, K + 1)}
                if hasattr(tw, 'shape') else {})
            sigma_results['c2_fluid_similarity'] = c2
            print(f"MCC={c2['mcc']:.3f}")
        except Exception as e:
            sigma_results['c2_fluid_similarity'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 3: FLuID quality ensemble --
        # Pass train-only cluster data for cross-val MCC (no test leakage)
        print("    C3: FLuID quality...", end=' ')
        try:
            pred_cls, pred_prob, mcc_weights = \
                predict_fluid_quality_ensemble(
                    teacher_models, cluster_data_train, test_fps, K)
            c3 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c3['mcc_weights'] = convert_numpy(mcc_weights)
            sigma_results['c3_fluid_quality'] = c3
            print(f"MCC={c3['mcc']:.3f}")
        except Exception as e:
            sigma_results['c3_fluid_quality'] = {'error': str(e)}
            print(f"FAILED: {e}")

        # -- Condition 4: FLuID published student --
        # cluster_data_noisy[0] is the full training DataFrame with noisy
        # labels (already training-only).
        print("    C4: FLuID student...", end=' ')
        try:
            pred_cls, pred_prob, n_fed = \
                federate_labels_and_train_student(
                    teacher_models, cluster_data_noisy,
                    cluster_data_noisy[0], K, test_fps,
                    algorithm=fed_model, seed=fed_seed)
            c4 = compute_clf_metrics(test_labels_clean, pred_cls, pred_prob)
            c4['n_federated_samples'] = n_fed
            # What C4 fitted: the helper ignores algorithm= for the student.
            c4['teacher_model'] = fed_model
            c4['student_model'] = C4_STUDENT_MODEL
            sigma_results['c4_fluid_student'] = c4
            print(f"MCC={c4['mcc']:.3f} (n_fed={n_fed}, student "
                  f"{C4_STUDENT_MODEL})")
        except Exception as e:
            sigma_results['c4_fluid_student'] = {
                'error': str(e), 'teacher_model': fed_model,
                'student_model': C4_STUDENT_MODEL}
            print(f"FAILED: {e}")

        # -- Condition 5: KIRBy federated feature selection, per hybrid --
        for b in budgets:
            _keys_seen = {}
            for _slot, h in hybrids_by_budget[int(b)]:
                bp = h['blueprint']
                print(f"    C5 [{_slot} @ {b}]: KIRBy federated FS...",
                      end=' ')
                _fail = {}
                try:
                    c5 = _silo_arm(
                        bp, train_labels_noisy, train_silo_indices, None, b,
                        f'{dataset_name} fold {fold_idx} {kirby_model} C5 '
                        f'{_slot} budget {b}', _fail)
                    print(f"MCC={c5['mcc']:.3f} budgets={c5['silo_budgets']}")
                except Exception as e:
                    c5 = {'error': str(e), 'traceback': traceback.format_exc(),
                          'silo_failures': {str(s): w
                                            for s, w in _fail.items()}}
                    print(f"FAILED: {e}")
                    traceback.print_exc()
                c5['recipe'] = _hybrid_record(h)
                _bk = _silo_build_key(bp)
                if _bk in _keys_seen:
                    c5['same_silo_build_as'] = _keys_seen[_bk]
                _keys_seen.setdefault(_bk, _slot)
                for s, why in _fail.items():
                    rec['silo_failures'].append({
                        'arm': 'c5', 'hybrid': _slot, 'budget': int(b),
                        'sigma': float(sigma_frac), 'silo': int(s),
                        'reason': why})
                sigma_results['by_budget'][str(b)][
                    f'c5_kirby_federated__{_slot}'] = c5

        # -- Condition 6: KIRBy + per-silo noise detection --
        # Each silo runs detectors locally, flags noisy samples, retrain.
        print("    C6: per-silo noise detection...")
        from kirby.utils.noise_mitigation_methods import get_mitigation_method
        # Three detectors vote and a row needs two of them. 'zscore' used to
        # be the third and it cannot fire on a two-class label (|z| exceeds 3
        # only when one class holds under a tenth of the silo), so C6 removed
        # zero rows and came out bit-identical to C5. MEASURED 2026-09-18,
        # 200 rows with 20 labels flipped: the old trio flags 12 rows and 8
        # are really flipped; this one flags 20 and 15 are.
        C6_DETECTORS = ['knn_k5', 'cv_disagreement', 'bootstrap_ensemble']
        cleaned_train_mask = np.ones(len(train_idx), dtype=bool)
        detection_results = {}
        for s in range(1, K + 1):
            s_idx = train_silo_indices[s]
            if len(s_idx) < 20:
                detection_results[f'silo_{s}'] = {
                    'n_flagged': 0, 'n_total': int(len(s_idx)),
                    'note': 'fewer than 20 rows; no detection'}
                continue
            # Each representation z-scored on this silo's own rows and divided
            # by sqrt(its width), so a wide representation does not pick the
            # neighbours on its own. Already scaled — do not scale it again.
            X_silo_s = stack_reps_for_distance(train_reps, rows=s_idx)
            vote_count = np.zeros(len(s_idx), dtype=int)
            det_errors = {}
            for det_name in C6_DETECTORS:
                try:
                    det = get_mitigation_method(det_name)
                    flagged = det.identify_noise(
                        X_silo_s, train_labels_noisy[s_idx])
                    vote_count[flagged] += 1
                except Exception as e:
                    # Recorded: a silent detector changes the vote.
                    det_errors[det_name] = f'{type(e).__name__}: {e}'
            flagged_local = np.where(vote_count >= 2)[0]
            cleaned_train_mask[s_idx[flagged_local]] = False
            detection_results[f'silo_{s}'] = {
                'n_flagged': int(len(flagged_local)),
                'n_total': int(len(s_idx)),
                'detectors_used': C6_DETECTORS,
                'detector_errors': det_errors,
            }

        # C6 IS C5 WITH THE FLAGGED ROWS REMOVED — same call, same per-silo
        # feature selection, same budget split across silos, same model. So
        # C6 minus C5 is per-silo detection and nothing else.
        n_excluded = int((~cleaned_train_mask).sum())
        clean_idx = np.where(cleaned_train_mask)[0]
        # Silo membership, re-expressed as positions in the cleaned rows. A
        # silo that loses every row drops out.
        _pos = np.full(len(train_idx), -1, dtype=int)
        _pos[clean_idx] = np.arange(len(clean_idx))
        clean_silo_indices = {}
        for s in range(1, K + 1):
            mapped = _pos[train_silo_indices[s]]
            clean_silo_indices[s] = mapped[mapped >= 0]
        for b in budgets:
            _keys_seen = {}
            for _slot, h in hybrids_by_budget[int(b)]:
                bp = h['blueprint']
                print(f"      C6 [{_slot} @ {b}]", end=' ')
                _fail = {}
                if len(clean_idx) > 20:
                    try:
                        c6 = _silo_arm(
                            bp, train_labels_noisy[clean_idx],
                            clean_silo_indices, clean_idx, b,
                            f'{dataset_name} fold {fold_idx} {kirby_model} '
                            f'C6 {_slot} budget {b}', _fail)
                    except Exception as e:
                        # No Morgan-fingerprint fallback: a fold where the
                        # hybrid never built must not read as a hybrid result.
                        c6 = {'error': str(e),
                              'traceback': traceback.format_exc(),
                              'silo_failures': {str(s): w
                                                for s, w in _fail.items()}}
                        traceback.print_exc()
                else:
                    c6 = {'error': 'too few samples after cleaning'}
                c6['recipe'] = _hybrid_record(h)
                c6['n_excluded'] = n_excluded
                c6['detection_results'] = convert_numpy(detection_results)
                _bk = _silo_build_key(bp)
                if _bk in _keys_seen:
                    c6['same_silo_build_as'] = _keys_seen[_bk]
                _keys_seen.setdefault(_bk, _slot)
                for s, why in _fail.items():
                    rec['silo_failures'].append({
                        'arm': 'c6', 'hybrid': _slot, 'budget': int(b),
                        'sigma': float(sigma_frac), 'silo': int(s),
                        'reason': why})
                sigma_results['by_budget'][str(b)][
                    f'c6_kirby_noise_detect__{_slot}'] = c6
                mcc_str = f"MCC={c6['mcc']:.3f}" if 'mcc' in c6 else 'N/A'
                print(f"{mcc_str} (excluded {n_excluded})")

        primary_sweep[str(sigma_frac)] = convert_numpy(sigma_results)
    rec['primary_sweep'] = primary_sweep

    # ── Phase 3: Noise detection analysis ─────────────────────────────────
    print("\n--- Phase 3: Noise detection analysis ---")
    # Spread noise across silos so the detectors see heterogeneous levels.
    sigma_det = 0.2
    silo_noise_levels = np.linspace(0.0, sigma_det, K).tolist()
    noise_levels_det = {s: float(silo_noise_levels[s - 1])
                        for s in range(1, K + 1)}
    train_labels_noisy_det, noise_mask_det, _, _flip_notes = _inject_flips(
        labels_binary[train_idx], train_silo_indices, noise_levels_det,
        seed=noise_seed)
    for _n in _flip_notes:
        rec['single_class_silo_flips'].append(
            dict(_n, phase='detection', sigma=sigma_det))
    try:
        # Pass NOISY labels so detectors can find the corrupted samples.
        det_results = federated_noise_detection(
            train_reps, train_labels_noisy_det, train_silo_indices,
            noise_mask_det, K, detector_names=DETECTORS,
            groups=train_groups, task_type='classification')
        # Its 'federated_hybrid' setting is left out of the result file
        # (2026-10-05). The helper builds those features with a fixed recipe:
        # performance_weighted, treeshap, an rf internal evaluator, every
        # representation, 100 features. No per-fold search chose it, rf is
        # not decided here, 100 is below caco2's and logd's decided 500-2000,
        # and treeshap is cut on logd. The helper still computes it until it
        # takes the fold's recipe (shared code, open).
        det_results.pop('federated_hybrid', None)
        detection_analysis = convert_numpy(det_results)
        detection_analysis['federated_hybrid_omitted'] = (
            'fixed recipe in federated_noise_detection (performance_weighted, '
            'treeshap, rf, 100 features), not from the per-fold search')
        print(f"  Centralized detectors: "
              f"{len(det_results.get('centralized', {}))} methods")
        print(f"  Per-silo detectors: "
              f"{len(det_results.get('per_silo', {}))} silos")
    except Exception as e:
        detection_analysis = {'error': str(e)}
        print(f"  FAILED: {e}")
    rec['noise_detection'] = detection_analysis

    # ── Phase 4: Importance-noise correlation ─────────────────────────────
    # Spearman between each silo's flip fraction and the features C5 gave it
    # (and its reported score), at flip level 0.2, per budget and hybrid.
    # Only silos that produced features enter: a failed silo has score None
    # and 0 features (2026-10-05). In the dirty run failed silos entered as
    # score 0.0 with 1 feature, so the correlation measured which silos
    # failed (+1.000 on one fold, -1.000 on another). No p-values.
    print("\n--- Phase 4: Importance-noise correlation ---")
    _corr_sigma = 0.2 if 0.2 in SIGMA_FRACS else max(SIGMA_FRACS)
    _corr_levels = np.linspace(0.0, _corr_sigma, K).tolist()
    importance_correlation = {}
    _sweep_c = primary_sweep.get(str(_corr_sigma), {})
    for b in budgets:
        importance_correlation[str(b)] = {}
        for _slot in _HYBRID_SLOTS:
            c5_data = ((_sweep_c.get('by_budget') or {}).get(str(b)) or {}).get(
                f'c5_kirby_federated__{_slot}')
            if not isinstance(c5_data, dict) or 'error' in c5_data \
                    or 'silo_budgets' not in c5_data:
                importance_correlation[str(b)][_slot] = {
                    'note': f'C5 not available at flip level {_corr_sigma}'}
                continue
            ok = [s for s in range(1, K + 1)
                  if (c5_data.get('silo_scores') or {}).get(str(s))
                  is not None]
            noise_v = [float(_corr_levels[s - 1]) for s in ok]
            bud_v = [c5_data['silo_budgets'].get(str(s), 0) for s in ok]
            sc_v = [c5_data['silo_scores'][str(s)] for s in ok]
            entry = {'flip_level': _corr_sigma, 'silos_used': ok,
                     'silos_failed': [s for s in range(1, K + 1)
                                      if s not in ok],
                     'noise_levels': noise_v, 'budgets': bud_v,
                     'scores': sc_v}
            if len(ok) < 3 or len(set(noise_v)) < 2:
                entry['note'] = (f'{len(ok)} silo(s) produced features; '
                                 f'too few to rank')
            else:
                r_b = spearmanr(noise_v, bud_v)[0]
                r_s = spearmanr(noise_v, sc_v)[0]
                entry['spearman_budget_r'] = (float(r_b)
                                              if np.isfinite(r_b) else None)
                entry['spearman_score_r'] = (float(r_s)
                                             if np.isfinite(r_s) else None)
                print(f"  {_slot} @ {b}: budget-noise r={entry['spearman_budget_r']}, "
                      f"score-noise r={entry['spearman_score_r']} "
                      f"({len(ok)} of {K} silos)")
            importance_correlation[str(b)][_slot] = entry
    rec['importance_noise_correlation'] = convert_numpy(importance_correlation)

    # ── Phase 5: Smooth factor sweep ──────────────────────────────────────
    print("\n--- Phase 5: Smooth factor sweep (sigma=0.2) ---")
    smooth_sweep = {}
    for sf in SMOOTH_FACTORS:
        print(f"  smooth_factor={sf}...")
        try:
            # Re-cluster training data with this smooth factor.
            train_silo_sf, cd_sf, _ = cluster_training_only(
                train_fluid_df, k=K, smooth_factor=sf, seed=fed_seed)
            # Same linspace(0.0, 0.2, K) pattern as Phases 2-4. _inject_flips
            # handles the one-class clusters smoothing factor 0 produces
            # (the dirty run's failure on every fold).
            silo_noise_sf = np.linspace(0.0, 0.2, K)
            noise_levels_sf = {s: float(silo_noise_sf[s - 1])
                               for s in range(1, K + 1)}
            tl_noisy, _, _, _flip_notes = _inject_flips(
                labels_binary[train_idx], train_silo_sf, noise_levels_sf,
                seed=noise_seed)
            for _n in _flip_notes:
                rec['single_class_silo_flips'].append(
                    dict(_n, phase='smooth_factor', smooth_factor=float(sf),
                         sigma=0.2))

            # Rebuild training cluster_data with noisy labels, preserving the
            # smoothed views as in the P1-8 fix above.
            train_df_sf = cd_sf[0].copy()
            train_df_sf['CLASS'] = tl_noisy
            train_df_sf['ACTIVITY'] = np.where(
                tl_noisy == 1, 'Active', 'Inactive')
            cd_sf_noisy = [train_df_sf]
            noisy_lookup_sf = train_df_sf[['CLASS', 'ACTIVITY']]
            for s in range(1, K + 1):
                smoothed_view_sf = cd_sf[s].copy()
                shared_sf = smoothed_view_sf.index.intersection(
                    noisy_lookup_sf.index)
                if len(shared_sf) > 0:
                    smoothed_view_sf.loc[shared_sf, 'CLASS'] = (
                        noisy_lookup_sf.loc[shared_sf, 'CLASS'])
                    smoothed_view_sf.loc[shared_sf, 'ACTIVITY'] = (
                        noisy_lookup_sf.loc[shared_sf, 'ACTIVITY'])
                cd_sf_noisy.append(smoothed_view_sf)

            teacher_models_sf = []
            cd_train_sf = [None]  # index 0 placeholder
            for t in range(0, K + 1):
                cd_t = cd_sf_noisy[t]
                # Same guard as the primary sweep: two rows and both classes.
                if len(cd_t) < 2 or len(np.unique(np.asarray(cd_t.CLASS))) < 2:
                    teacher_models_sf.append((None, 0, cd_t))
                    if t > 0:
                        cd_train_sf.append(cd_t)
                    continue
                X_t = np.asarray(list(cd_t.FP))
                y_t = np.asarray(list(cd_t.CLASS))
                clf_t = fit_fed_classifier(fed_model, X_t, y_t, seed=fed_seed)
                teacher_models_sf.append((clf_t, len(cd_t), cd_t))
                if t > 0:
                    cd_train_sf.append(cd_t)

            # C2: FLuID similarity (train-only cluster data)
            pred_cls_sf, pred_prob_sf, _ = predict_fluid_similarity_ensemble(
                teacher_models_sf, cd_train_sf, test_fps, K)
            c2_sf = compute_clf_metrics(
                test_labels_clean, pred_cls_sf, pred_prob_sf)
            _sf_entry = {'c2_fluid_similarity': c2_sf,
                         'by_budget': {str(b): {} for b in budgets}}

            # C5 per hybrid at every budget.
            for b in budgets:
                for _slot, h in hybrids_by_budget[int(b)]:
                    _fail = {}
                    try:
                        c5_sf = _silo_arm(
                            h['blueprint'], tl_noisy, train_silo_sf, None, b,
                            f'{dataset_name} fold {fold_idx} {kirby_model} '
                            f'smooth {sf} C5 {_slot} budget {b}', _fail)
                    except Exception as _e_slot:
                        c5_sf = {'error': str(_e_slot),
                                 'traceback': traceback.format_exc(),
                                 'silo_failures': {str(s): w
                                                   for s, w in _fail.items()}}
                    c5_sf['recipe'] = _hybrid_record(h)
                    for s, why in _fail.items():
                        rec['silo_failures'].append({
                            'arm': 'c5_smooth', 'smooth_factor': float(sf),
                            'hybrid': _slot, 'budget': int(b), 'sigma': 0.2,
                            'silo': int(s), 'reason': why})
                    _sf_entry['by_budget'][str(b)][
                        f'c5_kirby_federated__{_slot}'] = c5_sf
                    print(f"    C2 MCC={c2_sf.get('mcc', float('nan')):.3f}, "
                          f"C5 {_slot} @ {b} MCC="
                          f"{c5_sf.get('mcc', float('nan')):.3f}")
            smooth_sweep[str(sf)] = convert_numpy(_sf_entry)

        except Exception as e:
            smooth_sweep[str(sf)] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()
    rec['smooth_factor_sweep'] = smooth_sweep
    return rec


# ── One (model, seed) call saved as it finishes (2026-09-25) ────────────────
# federated_helpers.run_all calls run() once per (decided model, fed_seed) --
# six calls -- and writes fed01_noise_heterogeneity/<dataset>.json after the
# last. Each call's return value is written to
# fed01_noise_heterogeneity/<dataset>.partial.jsonl the moment it returns, and
# each fold inside a call to <dataset>__folds.partial.jsonl (2026-10-05), and
# both are read back on the next attempt, except a fold or call whose per-fold
# search failed, which is computed again. The identity carries RESULT_FORMAT,
# so files written before the per-fold alignment are moved aside.
# --force-rerun / KIRBY_FORCE_RERUN=1 discards them.
def run(data, config, output_dir, dataset_name, **kwargs):
    # _checkpoint.resumable_run, written out here so a call in which any
    # fold's per-fold search failed is NOT stored (2026-10-05): resumable_run
    # stores every return value, so one failed search used to become a
    # permanent 'arms skipped' result read back on every later attempt.
    from experiments._checkpoint import (Checkpoint, experiment_sidecar,
                                         run_identity)
    _key = {'fed_model': str(config.get('fed_model')
                             or fed_models_for(dataset_name)[0]),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    _cfg = {k: v for k, v in config.items() if k not in _key}
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, _cfg,
                     result_format=RESULT_FORMAT,
                     search_labels=SEARCH_LABELS))
    if ck.done(**_key):
        print(f'  [partial] {EXPERIMENT_DIR} {_key}: finished in an earlier '
              f'attempt, read back from {ck.path}')
        return ck.get(**_key)
    val = _run_one(data, config, output_dir, dataset_name, **kwargs)
    if val.get('all_searches_ok'):
        ck.put(val, **_key)
    else:
        print(f'  [partial] {EXPERIMENT_DIR} {_key}: a per-fold search '
              f'failed; this call is not stored and is computed again on '
              f'the next attempt')
    return val


if __name__ == '__main__':
    # The standalone entry point runs exactly what the cluster runs
    # (2026-10-05): federated_helpers.run_all for fed01 alone, which loads
    # DATASET_REPS[ds]['general'] (load_lhasa_herg_data for lhasa_herg),
    # applies decided_config, and loops over the dataset's decided models and
    # seeds. It used to call load_data(args.dataset, args.reps), whose --reps
    # default is the 'fast' preset, so a direct run searched a different set
    # of representations from the cluster run.
    import argparse
    from experiments.federated.federated_helpers import run_all
    _p = argparse.ArgumentParser(description='Fed01: Noise Heterogeneity')
    _p.add_argument('--dataset', type=str, default='lhasa_herg',
                    help='lhasa_herg, caco2 or logd')
    _p.add_argument('--output', type=str, default='tests/results',
                    help='Output directory for results JSON')
    _p.add_argument('--force-rerun', action='store_true',
                    help='Recompute instead of reading back finished work')
    args = _p.parse_args()
    run_all(args.dataset, args.output, experiments=['fed01'],
            force_rerun=args.force_rerun)
