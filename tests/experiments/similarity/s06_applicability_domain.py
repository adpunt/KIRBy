#!/usr/bin/env python3
"""S06: Applicability Domain -- does distance to the nearest training molecules
track prediction error?

Paper: Similarity
Question: for each held-out molecule, does its distance to its k nearest
          training molecules, measured in a representation or a hybrid,
          correlate with how wrong a model's prediction of it is (i.e. is that
          distance a usable applicability domain)?
Dependencies: none. Every search it reads is a per-fold search, built by the
          experiment itself when missing (handoff R6).

WHAT IS COMPUTED, PER OUTER FOLD (aligned 2026-10-05, HANDOFF_EXPERIMENT_
ALIGNMENT.md Part 2 chat 4; everything below this paragraph that names
get_rule_hybrids, the distance pool, rf or a fixed 100-feature budget describes
the design before that date):

  Folds       sim_outer_folds(data, config): the count is config['n_splits'].
              A count of 1 is one held-out split; spreads then say "one fold,
              no spread".
  Selection   fold_similarity_selection(..., neighbour_spaces=True), from
              searches on the fold's TRAINING rows on the dataset's decided
              grid. Nothing is read from a whole-dataset search.
  Spaces      (each one an "arm"; each also has an importance-weighted twin,
              '<arm>_weighted', unchanged from before -- weighting inside a
              similarity experiment's own distance is still open, handoff 2.10)
                knn_base1, knn_base2   knn's 2 best single representations,
                                       ranked by the per-fold distance search
                                       on the training rows. knn_base1 is the
                                       comparator ("the kNN pick").
                hybrid                 knn's top hybrid of the per-fold
                                       distance search
                hybrid_similarity      the top hybrid of the per-fold search
                                       run under objective='similarity'
                hybrid_<model>_general1, hybrid_<model>_general2
                                       each decided model's 2 best general
                                       hybrids (identical-feature duplicates
                                       already skipped)
              Every representation used to be an arm; that loop is cut to
              knn_base (Part 1 step 5).
  Distance    knn_axes(dataset, n): every distance of the dataset's group
              (full_knn: manhattan, cosine, minkowski_p0.5; fast_knn:
              manhattan, cosine), neighbour counts 5, 10, 20, scaffold group
              never excluded. Features scaled with FeatureScaler on the
              training rows (0/1 features untouched), held-out values cut to
              the training range. The number taken off the k distances is
              KIRBY_S06_AD_STATISTIC (mean by default). A distance statistic
              predicts nothing, so the counting scheme (rank,
              inverse_square) does not apply to it; it does apply to the
              held-out kNN scores stored per arm.
  Errors      per decided model, two error vectors per fold:
                <model>_own_hybrid          the model fitted on its own
                                            general1 hybrid. Same feature
                                            space as hybrid_<model>_general1,
                                            so that pair is tautological
                                            (rows carry same_space=True).
                <model>_all_representations the model fitted on every general
                                            representation glued together; it
                                            never sees a hybrid space.
              error = |residual| (regression) or 1 - P(true class)
              (classification).
  Correlation Spearman(distance statistic, error), per fold, for every
              distance x neighbour count x error vector x arm. Also pooled over
              folds (one Spearman over the concatenated arrays) and a
              reliability table (distance decile -> mean error), both as
              before.
  Conformal   regression only: per decided model and arm, split conformal on
              the fold's training rows (80% fit, 20% calibration, group-aware),
              nonconformity = |residual|; PICP and MPIW on the held-out rows.
              DECIDED 2026-10-06 (Adelaide delegated): the interval stays the
              plain absolute residual, so its width is the same for every
              held-out molecule of a fold.

Result file: results['folds'][i] holds the fold's selection (entry_record per
hybrid), the held-out scores, and every per-fold Spearman row; results['rows']
holds every Spearman (per fold and pooled), each with its distance, neighbour
count, statistic, error vector and arm. Per-fold scores and their spread only;
no significance test.

-- History, kept for the result files written before 2026-10-05 --

Per-fold CV with per-fold hybrid construction:
  - kNN distances in the best single base representation, the hybrid, and the
    weighted hybrid
  - error = |residual| (regression) or 1 - P(true class) (classification;
    graded, NOT a degenerate 0/1 indicator)
  - Spearman correlation between distance and error, reported two ways that
    differ in how folds are combined:
      * POOLED -- `distance_error_correlation['k<K>'][<comparator rep> |
        'hybrid' | '<comparator rep>_weighted' | 'hybrid_weighted']`: the
        comparator representation's name is recorded at `results['comparator_rep']`
        (it was the fixed string 'concat' until 2026-09-04; a concatenation is
        not a baseline in this project). Distances and errors from
        every fold are concatenated and ONE Spearman is taken over the pooled
        arrays. Every fold refits its own StandardScaler, so fold-to-fold
        differences in distance scale enter this single rank ordering.
      * PER-FOLD -- `distance_error_correlation['k<K>']['per_fold_mean']`:
        one Spearman inside each fold, then the mean over folds. Ranks never
        cross a fold boundary.
    Both blocks are split the same way by which model produced the errors:
      * top level -- errors from the RF fitted on the per-fold HYBRID
        features, i.e. the same feature space the `hybrid` and
        `hybrid_weighted` distances are measured in (tautological pair).
      * `['neutral_model']` -- errors from an RF fitted on every representation
        glued together, which never sees the hybrid feature space
        (de-tautologised pair). That model is an error generator, not an arm;
        nothing is ranked against it.
  - Reliability diagram (distance decile -> mean error)
  - Conformal calibration comparison: comparator rep vs hybrid (PICP, MPIW, AND
    undercoverage + `hybrid_genuinely_tighter` so a tighter-but-undercovered
    interval cannot be misread as an improvement)

Empirically (2026-06-03) the regression distance-error correlation is near
zero and sign-inconsistent across reps -> for these representations the
applicability-domain signal does NOT hold. This is a genuine negative finding,
not a measurement bug (error is the absolute residual; per-fold standardization
is train-only and consistent). That entry was recorded against the pooled
correlation, which was the only one stored at the time; the per-fold mean
added on 2026-08-26 is a separate number and was not part of it.

2026-09-20: the paragraph above has not held on the three datasets run since.
In tests/results_local_sim/s06_applicability_domain, every one of the 18 pooled
arms is a POSITIVE Spearman on all three: caco2 0.1403 to 0.2540, herg_ki 0.0768
to 0.2977, adme_mdr1 0.0432 to 0.3494. The de-tautologised block, where the
errors come from a forest that never saw the hybrid features, is positive in all
18 arms on all three as well: 0.1456 to 0.2408, 0.0925 to 0.3040, 0.0719 to
0.3840. None of those three datasets is one the 2026-06-03 entry was written
from, and its own run is not on disk here, so this is a disagreement between two
sets of datasets and not a correction of that entry.

2026-09-12 (owner's decision): the hybrid this experiment measures distance in
may now also come from a search run under objective='similarity'. Her reason:
s06's headline is a negative, that distance does not track error, and a
negative measured in a space never built for distance is not yet that negative.
Since 2026-10-05 that hybrid is 'hybrid_similarity', from the per-fold
similarity search.

2026-09-12 (owner's decision): every representation and its importance-weighted
twin were arms in every block. Cut to knn_base on 2026-10-05 (Part 1 step 5).

2026-10-02, the owner's set (top 2 general + top 1 distance per model per
family) read with get_rule_hybrids for knn from whole-dataset searches.
Replaced on 2026-10-05 by the per-fold selection above.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, flatten_fold_weights,
    compute_feature_importance, get_domain_metric, get_model_factories,
    importance_feature_weights, fold_reselection,
    # One sentence saying what a search family chose its features for.
    describe_pool_family,
)
from experiments.similarity.similarity_helpers import (
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold, knn_lists,
    knn_from_lists, score_prediction, fold_similarity_selection,
    build_fold_entry, entry_record, DATASET_REPS,
)
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)

# What the per-fold checkpoint units were computed under. Bumped 2026-10-05
# when the arms became the per-fold selection and every distance of
# knn_axes, so a partial file written under the whole-dataset owner's set is
# never read back.
CHECKPOINT_VERSION = 's06_per_fold_selection_2026_10_05'

# The comparator arm: knn's best single representation of the fold, ranked on
# the fold's training rows by the per-fold distance search ("the kNN pick").
COMPARATOR_ARM = 'knn_base1'

# Importance method for the weighted twins of the single-representation arms
# when the fold's distance hybrid carries none. The hybrid arms' twins use
# the importance scores the rebuild itself returns.
_FALLBACK_IMPORTANCE = 'lightgbm_gain'

# WHICH NUMBER IS TAKEN OFF THE k NEAREST TRAINING MOLECULES. Until
# 2026-09-20 this file computed one: the mean of the k distances, written
# inline as `.mean(axis=1)` at eight places. The applicability-domain
# literature uses three others as routinely, and a run can now pick one with
# KIRBY_S06_AD_STATISTIC. Unset gives 'mean', which is what every number
# already on disk was computed under.
AD_STATISTICS = ('mean', 'nn1', 'harmonic', 'kth')
ENV_AD_STATISTIC = 'KIRBY_S06_AD_STATISTIC'


def ad_statistic_name():
    """The statistic in force for this run. 'mean' when nothing is set."""
    name = os.environ.get(ENV_AD_STATISTIC) or 'mean'
    name = name.strip().lower()
    if name not in AD_STATISTICS:
        raise ValueError(
            f"{ENV_AD_STATISTIC}={name!r}: this file computes one of "
            f"{list(AD_STATISTICS)}.")
    return name


def _ad_statistic(dists, k, name=None):
    """One distance-to-training number per held-out molecule.

    `dists` is (n held-out molecules, k), ascending, straight out of
    kneighbors. Each name below reduces it to one number a molecule:

      mean       the mean of the k distances. What this file computed at every
                 one of its eight neighbour searches before 2026-09-20.
      nn1        the distance to the single nearest training molecule. k plays
                 no part, so every k reports the same number.
      harmonic   k / sum(1/d), the distance-weighted mean: a training molecule
                 close to the held-out one counts for more than a far one. A
                 held-out molecule sitting exactly on a training molecule gives
                 d=0 and a harmonic mean of 0, which is the right answer for
                 how far it is from the training set.
      kth        the distance to the k-th neighbour, i.e. the radius of the
                 ball around the held-out molecule holding k training
                 molecules.
    """
    name = ad_statistic_name() if name is None else name
    d = np.asarray(dists, dtype=float)
    if name == 'nn1':
        return d[:, 0].copy()
    if name == 'kth':
        return d[:, -1].copy()
    if name == 'harmonic':
        width = d.shape[1]
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.sum(1.0 / np.where(d <= 0, np.inf, d), axis=1)
        # A held-out molecule sitting exactly on ANY training molecule has one
        # infinite term in the sum, so the harmonic mean is 0 -- the classical
        # definition, and the right answer for how far it is from the training
        # set. Dropping the zero instead would report it as further out than a
        # molecule with no exact match at all.
        out = np.where(inv > 0, width / inv, 0.0)
        return np.where(np.any(d <= 0, axis=1), 0.0, out)
    return d.mean(axis=1)


def _importance_exponent_in_force():
    """The power a feature's importance is raised to before it weights it.

    Every `_weighted` arm in this file multiplies its z-scored features by the
    feature's lightgbm_gain importance raised to this power. It was written out
    as `np.sqrt(...)` at four places, so it was 0.5 and nothing recorded it.
    KIRBY_IMPORTANCE_EXPONENT now sets it; unset still gives 0.5.
    """
    from kirby.similarity.preprocessing import importance_exponent_override
    value = importance_exponent_override()
    return 0.5 if value is None else value


def _graded_clf_error(clf, X, y_true):
    """Continuous classification error = 1 - P(true class).

    Robust to classes present in y_true but unseen in training (those get
    error 1.0). Far higher power than a 0/1 misclassification indicator for
    Spearman(distance, error) applicability-domain correlations.
    """
    proba = clf.predict_proba(X)
    col = {c: i for i, c in enumerate(clf.classes_)}
    return np.array([
        1.0 - (proba[i, col[c]] if c in col else 0.0)
        for i, c in enumerate(y_true)
    ], dtype=float)


def _fold_spearman(dists, errors):
    """Spearman(distance, error) inside a single fold.

    Returns NaN when the fold has fewer than 3 points or either array is
    constant -- spearmanr is undefined there, and a NaN is dropped by
    `_mean_over_folds` rather than propagating into the fold mean.
    """
    if len(dists) < 3:
        return float('nan')
    rho, _ = spearmanr(dists, errors)
    return float(rho)


def _mean_over_folds(rhos):
    """Mean of the finite per-fold rhos plus how many folds contributed.

    Returns (None, 0) when no fold produced a finite rho.
    """
    finite = [r for r in rhos if np.isfinite(r)]
    if not finite:
        return None, 0
    return float(np.mean(finite)), len(finite)



def _winners_by_fold(fold_lists, winners):
    """fold_reselection's winners list, one entry per held-out fold.

    Added 2026-10-05. fold_reselection lists the winner of each held-out fold
    but leaves out a fold where no arm has a usable score on the other folds,
    so position j of its list is not always fold j. This puts None at those
    folds, using the same test it applies, so the list lines up with the
    per-fold scores.
    """
    def _ok(v):
        return v is not None and np.isfinite(v)
    F = max((len(v) for v in fold_lists.values()), default=0)
    it = iter(winners)
    out = []
    for f in range(F):
        usable = any(_ok(v[k]) for v in fold_lists.values()
                     for k in range(len(v)) if k != f)
        out.append(next(it, None) if usable else None)
    return out


def _fmt_rho(value):
    """4-dp rho for the printed lines; 'n/a' when the fold mean is None."""
    return 'n/a' if value is None else f'{value:.4f}'


def _fold_arms(sel, models):
    """The arms of one fold, in a fixed order: [(arm, kind, source, payload)].

    kind 'rep' carries a representation name; kind 'hybrid' carries a
    selection entry (blueprint + aug_condition). An arm the fold's selection
    could not fill is absent here and named in the fold's 'unavailable'.
    """
    arms = []
    for j, rname in enumerate(sel.get('knn_base') or []):
        arms.append((f'knn_base{j + 1}', 'rep', 'distance_search/knn/base',
                     rname))
    if sel.get('distance_hybrid') is not None:
        arms.append(('hybrid', 'hybrid', 'distance_search/knn',
                     sel['distance_hybrid']))
    if sel.get('similarity_hybrid') is not None:
        arms.append(('hybrid_similarity', 'hybrid', 'similarity_search',
                     sel['similarity_hybrid']))
    for m in models:
        ms = (sel.get('models') or {}).get(m)
        if not ms:
            continue
        for j, entry in enumerate(ms.get('hybrids') or []):
            arms.append((f'hybrid_{m}_general{j + 1}', 'hybrid',
                         f'general_search/{m}', entry))
    return arms


def _predict_errors(est, X_va, y_va, is_clf):
    """Held-out predictions of a fitted model, and the per-molecule error.

    Returns (pred, proba, classes, errors, error_kind). Classification error
    is 1 - P(true class) when the model gives probabilities; a model without
    predict_proba falls back to the 0/1 miss, and error_kind says so.
    """
    pred = est.predict(X_va)
    if not is_clf:
        return pred, None, None, np.abs(np.asarray(y_va, float) - pred), \
            'absolute_residual'
    if hasattr(est, 'predict_proba'):
        try:
            proba = est.predict_proba(X_va)
            classes = np.asarray(est.classes_)
            return pred, proba, classes, _graded_clf_error(est, X_va, y_va), \
                '1_minus_p_true_class'
        except Exception:
            pass
    return pred, None, None, (np.asarray(pred) != np.asarray(y_va)).astype(
        float), 'misclassified_0_1'


def _run_fold(data, dataset_name, fold_i, train_idx, val_idx, *, n_outer,
              models, factories, knn_factory, axes, metric_name, metric_fn,
              lower_is_better, is_clf, neutral_reps, ad_statistic):
    """Everything s06 computes on one outer fold. Returns a checkpointable
    payload: the fold's record plus the distance and error arrays the pooled
    blocks need."""
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    y_tr, y_va = labels[train_idx], labels[val_idx]
    unavailable = {}

    sel = fold_similarity_selection(
        data, dataset_name, train_idx=train_idx, fold_i=fold_i,
        n_outer_folds=n_outer, neighbour_spaces=True)
    unavailable.update(sel.get('unavailable') or {})

    selection_record = {
        'selection_scope': sel.get('selection_scope'),
        'models': {
            m: (None if not (sel['models'] or {}).get(m) else {
                'base': list(sel['models'][m].get('base') or []),
                'hybrids': [entry_record(e) for e in
                            sel['models'][m].get('hybrids') or []],
                'hybrids_skipped_identical_features': [
                    (s.get('label') if isinstance(s, dict) else s)
                    for s in sel['models'][m].get(
                        'hybrids_skipped_identical_features') or []]})
            for m in models},
        'knn_base': list(sel.get('knn_base') or []),
        'knn_base_scores': [
            (e.get('rep'), e.get('score'))
            for e in ((sel.get('knn_base_detail') or {}).get('selected')
                      or [])],
        'distance_hybrid': entry_record(sel.get('distance_hybrid')),
        'similarity_hybrid': entry_record(sel.get('similarity_hybrid')),
    }

    # ── Build every arm on the training rows, scaled on the training rows ──
    # spaces[arm] = (X_train scaled, X_val scaled, importance weights or None)
    spaces = {}
    arm_meta = {}
    arms = _fold_arms(sel, models)
    rep_arms = [(a, r) for a, k, _s, r in arms if k == 'rep']
    imp = {}
    if rep_arms:
        # Weighted twins of the single representations: importance on the
        # fold's training rows, with the method the fold's distance hybrid was
        # built with (the old code used its hybrid's method the same way).
        _dh = sel.get('distance_hybrid') or {}
        _imp_method = (getattr(_dh.get('blueprint'), 'importance_method', None)
                       or _FALLBACK_IMPORTANCE)
        try:
            imp = compute_feature_importance(
                {r: np.asarray(reps[r])[train_idx] for _a, r in rep_arms},
                y_tr, method=_imp_method)
        except Exception as exc:
            unavailable['knn_base_importance'] = (
                f'{_imp_method} raised {type(exc).__name__}: {exc}')
            imp = {}
    for arm, kind, source, payload in arms:
        try:
            if kind == 'rep':
                X = np.asarray(reps[payload])
                A, B = scale_fold(X[train_idx], X[val_idx])
                w = (importance_feature_weights(imp[payload])
                     if payload in imp else None)
                arm_meta[arm] = {'kind': 'single_representation',
                                 'space': payload, 'source': source,
                                 'width_built': int(A.shape[1])}
            else:
                if source.startswith('general_search/'):
                    _fac = factories.get(source.split('/', 1)[1])
                elif source == 'distance_search/knn':
                    _fac = knn_factory
                else:
                    _fac = None   # similarity recipe: no model admits features
                Xtr_h, Xva_h, finfo = build_fold_entry(
                    data, payload, train_idx=train_idx, val_idx=val_idx,
                    model_factory=_fac, metric_fn=metric_fn,
                    lower_is_better=lower_is_better)
                A, B = scale_fold(Xtr_h, Xva_h)
                w = importance_feature_weights(flatten_fold_weights(finfo))
                if w.shape[0] != A.shape[1]:
                    unavailable[f'{arm}_weighted'] = (
                        f'{w.shape[0]} importance weights for '
                        f'{A.shape[1]} built features')
                    w = None
                bp = payload.get('blueprint')
                arm_meta[arm] = {
                    'kind': 'hybrid', 'space': payload.get('label'),
                    'source': source, 'record': entry_record(payload),
                    'features_chosen_by': describe_pool_family(
                        'similarity' if source == 'similarity_search'
                        else 'predictive'),
                    'budget': getattr(bp, 'budget', None),
                    'width_built': int(A.shape[1])}
            spaces[arm] = (A, B, w)
        except Exception as exc:
            unavailable[arm] = (f'build on the fold failed: '
                                f'{type(exc).__name__}: {exc}')
            print(f"    fold {fold_i}: {arm} not built: "
                  f"{type(exc).__name__}: {exc}")

    # ── The error vectors: each decided model, two inputs ────────────────────
    errors = {}
    error_meta = {}
    model_heldout = []
    neutral_X = None
    if neutral_reps:
        neutral_X = scale_fold(
            np.hstack([np.asarray(reps[r])[train_idx] for r in neutral_reps]),
            np.hstack([np.asarray(reps[r])[val_idx] for r in neutral_reps]))
    for m in models:
        fac = factories.get(m)
        if fac is None:
            unavailable[f'errors/{m}'] = ('no model factory (the size guard of '
                                          'DEFAULT_MODELS skipped it)')
            continue
        fits = []
        for j in (1, 2):
            arm = f'hybrid_{m}_general{j}'
            if arm in spaces:
                fits.append((arm, spaces[arm][0], spaces[arm][1]))
        if neutral_X is not None:
            fits.append(('all_representations', neutral_X[0], neutral_X[1]))
        for fit_on, A, B in fits:
            try:
                est = fac()
                est.fit(A, y_tr)
                pred, proba, classes, err, ekind = _predict_errors(
                    est, B, y_va, is_clf)
                score = score_prediction(y_va, pred, proba, classes,
                                         metric_name, metric_fn)
            except Exception as exc:
                unavailable[f'fit/{m}/{fit_on}'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            model_heldout.append({'model': m, 'fitted_on': fit_on,
                                  'held_out_score': score})
            src = None
            if fit_on == f'hybrid_{m}_general1':
                src = f'{m}_own_hybrid'
            elif fit_on == 'all_representations':
                src = f'{m}_all_representations'
            if src is not None:
                errors[src] = np.asarray(err, dtype=np.float64)
                error_meta[src] = {'model': m, 'fitted_on': fit_on,
                                   'error': ekind,
                                   'held_out_score': score}

    # ── Distances: every arm, every decided distance, every neighbour count ──
    stats = {}           # 'distance|k|arm' -> per held-out molecule statistic
    knn_heldout = []
    for arm, (A, B, w) in spaces.items():
        variants = [(arm, A, B)]
        if w is not None:
            variants.append((f'{arm}_weighted', A * w, B * w))
        for name, XA, XB in variants:
            for metric in axes['metrics']:
                try:
                    idx, dist = knn_lists(XA, XB, metric=metric,
                                          k_max=axes['k_max'])
                except Exception as exc:
                    unavailable[f'{name}/{metric}'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                for k in axes['ks']:
                    if k > dist.shape[1]:
                        unavailable[f'{name}/{metric}/k{k}'] = (
                            f'only {dist.shape[1]} training molecules')
                        continue
                    stats[f'{metric}|{k}|{name}'] = _ad_statistic(
                        dist[:, :k], k, ad_statistic).astype(np.float64)
                    if name != arm:
                        continue
                    # Held-out kNN score in this arm, at every decided
                    # setting, from the same neighbour list.
                    for scheme in axes['schemes']:
                        try:
                            p, pr, cl = knn_from_lists(
                                y_tr, idx, dist, k=k, scheme=scheme,
                                is_clf=is_clf)
                            s = score_prediction(y_va, p, pr, cl,
                                                 metric_name, metric_fn)
                        except Exception as exc:
                            s = None
                            unavailable[f'knn/{arm}/{metric}/k{k}/{scheme}'] = (
                                f'{type(exc).__name__}: {exc}')
                        knn_heldout.append({'arm': arm, 'distance': metric,
                                            'k': int(k), 'scheme': scheme,
                                            'held_out_score': s})

    # ── Per-fold Spearman rows ───────────────────────────────────────────────
    rho_rows = []
    for key, d in stats.items():
        metric, k, name = key.split('|', 2)
        base_arm = name[:-len('_weighted')] if name.endswith('_weighted') \
            else name
        for src, err in errors.items():
            m = error_meta[src]['model']
            rho_rows.append({
                'fold': int(fold_i), 'distance': metric, 'k': int(k),
                'statistic': ad_statistic, 'counting_scheme': None,
                'error_vector': src, 'arm': name,
                'weighted': name.endswith('_weighted'),
                'same_space': (base_arm == f'hybrid_{m}_general1'
                               and src == f'{m}_own_hybrid'),
                'rho': _fold_spearman(d, err), 'n_molecules': int(len(d))})

    # ── Conformal intervals (regression only) ───────────────────────────────
    conformal = None
    if not is_clf:
        alpha = 0.1
        train_groups = None if groups is None else np.asarray(groups)[train_idx]
        if train_groups is not None and len(np.unique(train_groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_local, cal_local = next(gss.split(
                np.arange(len(y_tr)), y_tr, groups=train_groups))
        else:
            n_cal = max(int(len(y_tr) * 0.2), 10)
            perm = np.random.RandomState(42).permutation(len(y_tr))
            cal_local, tr_local = perm[:n_cal], perm[n_cal:]
        conformal = {}
        for m in models:
            fac = factories.get(m)
            if fac is None:
                continue
            conformal[m] = {}
            for arm, (A, B, _w) in spaces.items():
                try:
                    est = fac()
                    est.fit(A[tr_local], y_tr[tr_local])
                    q = float(np.quantile(
                        np.abs(y_tr[cal_local] - est.predict(A[cal_local])),
                        1 - alpha))
                    pred = est.predict(B)
                    conformal[m][arm] = {
                        'picp': float(np.mean(np.abs(y_va - pred) <= q)),
                        'mpiw': float(2 * q)}
                except Exception as exc:
                    unavailable[f'conformal/{m}/{arm}'] = (
                        f'{type(exc).__name__}: {exc}')

    record = {
        'fold': int(fold_i), 'n_train': int(len(train_idx)),
        'n_held_out': int(len(val_idx)),
        'selection_scope': 'fold_train_rows',
        'selection': selection_record,
        'arms': arm_meta,
        'error_vectors': error_meta,
        'model_held_out': model_heldout,
        'knn_held_out': knn_heldout,
        'conformal': conformal,
        'unavailable': unavailable,
    }
    return {'record': record, 'rho_rows': rho_rows,
            'errors': {s: e.astype(np.float32) for s, e in errors.items()},
            'stats': {s: v.astype(np.float32) for s, v in stats.items()}}


def _reliability(dists, errs):
    """Mean error in each distance decile (the last bin includes its upper
    edge), as before 2026-10-05."""
    dists = np.asarray(dists, float)
    errs = np.asarray(errs, float)
    ok = np.isfinite(dists)
    dists, errs = dists[ok], errs[ok]
    if len(dists) == 0:
        return None
    edges = np.percentile(dists, np.arange(0, 110, 10))
    out_e, out_n = [], []
    for di in range(10):
        hi = (dists <= edges[di + 1]) if di == 9 else (dists < edges[di + 1])
        mask = (dists >= edges[di]) & hi
        out_n.append(int(mask.sum()))
        out_e.append(float(np.mean(errs[mask])) if mask.sum() else float('nan'))
    return {'decile_errors': out_e, 'decile_counts': out_n,
            'n_molecules_dropped_infinite_distance': int((~ok).sum())}


def run(data, config, output_dir, dataset_name, **kwargs):
    """Does distance to the nearest training molecules track prediction
    error? Per outer fold, on the fold's own selection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S06: APPLICABILITY DOMAIN")
    print("=" * 70)

    # Settings that used to come in from the caller and are now decided
    # elsewhere: neighbour counts by knn_axes, the budget and the models by
    # the per-fold selection on the decided grid.
    for _k in ('k_values', 'budget', 'downstream_models'):
        if kwargs.get(_k) is not None:
            print(f"  [s06] {_k}={kwargs[_k]!r} is ignored: since 2026-10-05 "
                  f"it is set by the decided grid and knn_axes")

    reps = data['reps']
    labels = np.asarray(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    axes = knn_axes(dataset_name, len(labels))
    ad_statistic = ad_statistic_name()
    n_outer = int(config['n_splits'])
    folds = sim_outer_folds(data, config)

    from experiments.decided_grid import decided_models
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    knn_factory = get_model_factories(['knn'], task_type=task_type,
                                      n_samples=len(labels)).get('knn')

    # The error model that never sees a hybrid: every general representation
    # of the dataset glued together (DATASET_REPS[ds]['general']).
    # A run whose loaded representations hold none of that list (a standalone
    # run with --reps) uses every loaded one, and the result file names them.
    _gen = (DATASET_REPS.get(dataset_name) or {}).get('general')
    neutral_reps = ([r for r in sorted(reps) if _gen and r in _gen]
                    or sorted(reps))

    print(f"  {len(folds)} outer folds; models {models}; distances "
          f"{axes['metrics']} ({axes['group']}); k {axes['ks']}; statistic "
          f"{ad_statistic!r}; importance exponent "
          f"{_importance_exponent_in_force()}")

    results = {
        'selection_rule': _SELECTION_RULE,
        'selection_scope': 'fold_train_rows',
        'checkpoint_version': CHECKPOINT_VERSION,
        'metric': metric_name, 'lower_is_better': bool(lower_is_better),
        'ad_number': ('Spearman between distance to the k nearest training '
                      'molecules and prediction error; higher = distance '
                      'tracks error more closely'),
        'knn_axes': axes,
        'distance_statistic': ad_statistic,
        'counting_scheme': ('not applicable to the distance statistic; the '
                            'held-out kNN scores carry their scheme'),
        'importance_exponent': _importance_exponent_in_force(),
        'n_outer_folds': n_outer,
        'models': list(models),
        'neutral_model_representations': list(neutral_reps),
        'comparator_rep': COMPARATOR_ARM,
        'comparator_note': (
            "knn_base1 is knn's best single representation of each fold, "
            'ranked by the per-fold distance search on the training rows; the '
            "representation it names per fold is in folds[i]['arms']"),
        'folds': [], 'rows': [], 'unavailable_folds': {},
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's06_applicability_domain', dataset_name),
        run_identity('s06_applicability_domain', dataset_name, data, config,
                     version=CHECKPOINT_VERSION, knn_axes=axes,
                     ad_statistic=ad_statistic, models=list(models),
                     importance_exponent=_importance_exponent_in_force()))

    fold_payloads = {}
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        if _ck.done(fold=fold_i):
            fold_payloads[fold_i] = _ck.get(fold=fold_i)
            print(f"  fold {fold_i}: read back from the partial file")
            continue
        print(f"\n  fold {fold_i} ({len(train_idx)} training, "
              f"{len(val_idx)} held-out molecules)")
        try:
            pl = _run_fold(
                data, dataset_name, fold_i, train_idx, val_idx,
                n_outer=n_outer, models=models, factories=factories,
                knn_factory=knn_factory, axes=axes, metric_name=metric_name,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
                is_clf=is_clf, neutral_reps=neutral_reps,
                ad_statistic=ad_statistic)
        except Exception as exc:
            print(f"    fold {fold_i} failed: {type(exc).__name__}: {exc}")
            results['unavailable_folds'][str(fold_i)] = (
                f'{type(exc).__name__}: {exc}')
            continue
        fold_payloads[fold_i] = pl
        _ck.put(pl, fold=fold_i)
        _r = pl['record']
        print("    arms: " + ', '.join(
            f"{a}={m.get('space')}" for a, m in _r['arms'].items()))
        for _u, _why in sorted(_r['unavailable'].items()):
            print(f"    [unavailable] {_u}: {_why}")

    if not fold_payloads:
        return {'error': 'no outer fold produced a result: '
                         + '; '.join(results['unavailable_folds'].values())}

    fold_ids = sorted(fold_payloads)
    for f in fold_ids:
        results['folds'].append(fold_payloads[f]['record'])
        results['rows'].extend(fold_payloads[f]['rho_rows'])

    # ── Per-fold and pooled summaries ────────────────────────────────────────
    # Keyed distance -> k<K> -> error vector -> arm. Error vectors and arms
    # are the union over folds; a fold lacking one contributes None.
    keys = set()
    for f in fold_ids:
        for row in fold_payloads[f]['rho_rows']:
            keys.add((row['distance'], row['k'], row['error_vector'],
                      row['arm']))
    rho_by = {(r['fold'], r['distance'], r['k'], r['error_vector'], r['arm']):
              r['rho'] for f in fold_ids for r in fold_payloads[f]['rho_rows']}

    dec = {}
    reliability = {}
    for (metric, k, src, arm) in sorted(keys):
        blk = dec.setdefault(metric, {}).setdefault(f'k{k}', {}).setdefault(
            src, {'distance': metric, 'k': int(k), 'error_vector': src,
                  'statistic': ad_statistic, 'counting_scheme': None,
                  'fold_ids': list(fold_ids), 'per_fold': {},
                  'per_fold_mean': {}, 'per_fold_n': {},
                  'per_fold_spread': {}, 'pooled': {}})
        vals = [rho_by.get((f, metric, k, src, arm)) for f in fold_ids]
        blk['per_fold'][arm] = vals
        mean, n = _mean_over_folds([v for v in vals if v is not None])
        blk['per_fold_mean'][arm] = mean
        blk['per_fold_n'][arm] = n
        _fin = [v for v in vals if v is not None and np.isfinite(v)]
        blk['per_fold_spread'][arm] = (
            fold_spread_note(_fin)
            or {'min': float(min(_fin)), 'max': float(max(_fin)),
                'sd': float(np.std(_fin, ddof=1))})
        # Pooled: concatenate the folds where both the arm's distances and
        # this error vector exist, so the two stay row-aligned.
        d_all, e_all = [], []
        for f in fold_ids:
            d = fold_payloads[f]['stats'].get(f'{metric}|{k}|{arm}')
            e = fold_payloads[f]['errors'].get(src)
            if d is not None and e is not None and len(d) == len(e):
                d_all.append(np.asarray(d, float))
                e_all.append(np.asarray(e, float))
        if d_all:
            D, E = np.concatenate(d_all), np.concatenate(e_all)
            rho = _fold_spearman(D, E)
            blk['pooled'][arm] = rho
            results['rows'].append({
                'fold': 'pooled', 'distance': metric, 'k': int(k),
                'statistic': ad_statistic, 'counting_scheme': None,
                'error_vector': src, 'arm': arm,
                'weighted': arm.endswith('_weighted'),
                'rho': rho, 'n_molecules': int(len(D)),
                'n_folds_pooled': len(d_all)})
            reliability.setdefault(metric, {}).setdefault(
                f'k{k}', {}).setdefault(src, {})[arm] = _reliability(D, E)
    results['distance_error_correlation'] = dec
    results['reliability'] = reliability

    # ── The comparator picked per held-out fold on the other folds ──────────
    # (e7d9ae2, kept.) On each held-out fold, among the single-representation
    # arms (knn_base1, knn_base2 and their weighted twins), the one with the
    # highest mean per-fold Spearman over the OTHER folds; 'hybrid' (knn's
    # distance hybrid) is the one fixed candidate, as before. Per distance,
    # neighbour count and error vector. Needs 2 or more folds.
    resel = {}
    for metric, by_k in dec.items():
        for kk, by_src in by_k.items():
            for src, blk in by_src.items():
                singles = {a: [None if v is None or not np.isfinite(v)
                               else float(v) for v in vals]
                           for a, vals in blk['per_fold'].items()
                           if a.startswith('knn_base')}
                hyb = blk['per_fold'].get('hybrid')
                if not singles or hyb is None or len(fold_ids) < 2:
                    continue
                try:
                    _rs = fold_reselection(
                        {'ad': [{'name': 'hybrid', 'fold_scores': [
                            None if v is None or not np.isfinite(v)
                            else float(v) for v in hyb]}]},
                        {'ad': singles}, lower_is_better=False)['ad']
                except Exception as exc:
                    resel.setdefault(metric, {}).setdefault(kk, {})[src] = {
                        'not_computed': f'{type(exc).__name__}: {exc}'}
                    continue
                resel.setdefault(metric, {}).setdefault(kk, {})[src] = {
                    'fold_ids': list(fold_ids),
                    'picked_per_fold': _winners_by_fold(
                        singles, _rs['baseline_winners_per_fold']),
                    'comparator_fold_scores': list(_rs['baseline_fold_scores']),
                    'hybrid_fold_scores': list(_rs['hybrid_fold_scores']),
                    'per_fold_delta': list(_rs['per_fold_delta']),
                    'delta_mean_of_folds': _rs['delta_mean_of_folds'],
                    'n_paired_folds': _rs['n_paired_folds']}
    results['comparator_reselected'] = resel or {
        'not_computed': ('needs 2 or more outer folds with a hybrid and a '
                         'single-representation arm')}

    # ── Conformal summary: per model and arm, per fold and mean ─────────────
    if is_clf:
        results['conformal'] = {'skipped': True, 'reason': 'classification'}
    else:
        conf = {'nominal_coverage': 0.9,
                'nonconformity': 'absolute residual (split conformal)',
                'by_model': {}}
        for m in models:
            per_arm = {}
            for f in fold_ids:
                c = (fold_payloads[f]['record'].get('conformal') or {}).get(m)
                for arm, v in (c or {}).items():
                    per_arm.setdefault(arm, {})[f] = v
            out = {}
            for arm, by_f in per_arm.items():
                picp = [by_f[f]['picp'] for f in sorted(by_f)]
                mpiw = [by_f[f]['mpiw'] for f in sorted(by_f)]
                out[arm] = {'fold_ids': sorted(by_f), 'picp_per_fold': picp,
                            'mpiw_per_fold': mpiw,
                            'picp': float(np.mean(picp)),
                            'mpiw': float(np.mean(mpiw)),
                            'undercoverage': float(0.9 - np.mean(picp))}
            # A hybrid's interval counts as tighter than a single
            # representation's only if it keeps coverage (tolerance 0.02), on
            # the folds both have. Against knn_base1 (the comparator) and
            # knn_base2.
            tighter = {}
            for h in [a for a in out if a.startswith('hybrid')]:
                for b in [a for a in out if a.startswith('knn_base')]:
                    common = sorted(set(per_arm[h]) & set(per_arm[b]))
                    if not common:
                        continue
                    hp = np.mean([per_arm[h][f]['picp'] for f in common])
                    hw = np.mean([per_arm[h][f]['mpiw'] for f in common])
                    bp_ = np.mean([per_arm[b][f]['picp'] for f in common])
                    bw = np.mean([per_arm[b][f]['mpiw'] for f in common])
                    tighter[f'{h}_vs_{b}'] = {
                        'tighter_and_covered': bool(hw < bw and hp >= bp_ - 0.02),
                        'n_folds': len(common)}
            conf['by_model'][m] = {'arms': out, 'hybrid_genuinely_tighter':
                                   tighter}
        results['conformal'] = conf

    # ── Printed summary: knn_base1 against each hybrid, per fold mean ───────
    for metric, by_k in dec.items():
        for kk, by_src in by_k.items():
            for src, blk in by_src.items():
                pm = blk['per_fold_mean']
                line = ', '.join(f"{a}={_fmt_rho(pm.get(a))}"
                                 for a in sorted(pm) if not a.endswith(
                                     '_weighted'))
                print(f"  {metric} {kk} errors {src}: per-fold mean "
                      f"Spearman {line}")
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S06: Applicability Domain',
                                     has_delta=True, has_augment=True)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 's06_applicability_domain', args.dataset)
