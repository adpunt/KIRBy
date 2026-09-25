#!/usr/bin/env python3
"""S06: Applicability Domain -- does hybrid distance track prediction error?

Paper: Similarity
Question: does distance-to-training in the hybrid space correlate with the
          model's prediction error (i.e. is it a usable applicability domain)?
Dependencies: s02 (reads JSON, optional -- falls back to fresh search).

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
may now also come from a search run under objective='similarity' -- a search
whose features were chosen to make the distance geometry predict each
molecule's activity from its five nearest neighbours. Her reason: s06's
headline is a negative, that distance does not track error, and a negative
measured in a space never built for distance is not yet that negative. The pool
subset here is therefore 'predictive_and_similarity' instead of the default
'predictive_only'. Which family the winning hybrid came from is written to
`results['hybrid_family']`, and every eligible family's own best to
`results['pool_by_objective']`, so the two are never read as one number.

2026-09-19: the geometry family is BUILT now, not only recorded. The pool names
the two families in preference order and returns the first that has an entry; a
predictive recipe always exists, so between those two dates every arm here was
measured in the prediction-built hybrid and the other family appeared only in
`pool_by_objective`. Two arms are added -- `hybrid_similarity` and
`hybrid_similarity_weighted`, built from the similarity family's own recipe
taken by name. `hybrid` and `hybrid_weighted` are unchanged. Both new arms are
scored against the SAME two error vectors as every other arm, so nothing extra
is fitted in the correlation block: one build and two neighbour searches per
fold. The conformal block adds one build and one random forest per fold. When
the pool holds no similarity search the arms are absent and
`results['hybrid_similarity_unavailable_reason']` says why.

2026-09-12 (owner's decision): every representation and its importance-weighted
twin are arms in every block, not only in the pooled correlation. The per-fold
means, the neutral-model correlations (pooled and per-fold), the reliability
deciles and the prediction intervals all carry them now, and
`conformal['hybrid_genuinely_tighter_by_rep']` holds one true/false per
representation beside the comparator's own flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from sklearn.model_selection import GroupShuffleSplit
from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results, convert_numpy,
    get_fitted_hybrid, get_fitted_pool_hybrid, flatten_weights, flatten_fold_weights, make_cv_splitter,
    apply_feature_selection, compute_feature_importance,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    importance_feature_weights,
    fold_hybrid_scoring_kwargs,
    get_best_for_downstream, extract_top3_base_reps,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS, DISTANCE_MODELS,
    # One sentence saying what a search family chose its features for. Added
    # 2026-09-19 with the second hybrid arm.
    describe_pool_family,
)
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)
# Default kNN budget for AD hybrid search (kept modest; AD is a diagnostic, not
# a tuned downstream model).
_DEFAULT_AD_BUDGET = 100

# The two arms built from the similarity search's own recipe, added 2026-09-19.
# 'hybrid' and 'hybrid_weighted' are the predictive ones and are unchanged.
SIM_ARM = 'hybrid_similarity'
SIM_ARM_W = 'hybrid_similarity_weighted'


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


def resolved_k_values(kwargs):
    """How many nearest training molecules the statistic is taken over.

    The caller's own `k_values` wins. Otherwise KIRBY_S06_K_VALUES, a comma
    separated list of neighbour counts; unset gives [5], which is the single
    value this file used from 2026-06 to 2026-09-20 and the one every number
    already on disk was computed at. The similarity wave passes nothing, so
    before this the environment could not reach it at all.
    """
    if kwargs.get('k_values'):
        return [int(v) for v in kwargs['k_values']]
    raw = os.environ.get('KIRBY_S06_K_VALUES')
    if not raw or not raw.strip():
        return [5]
    return [int(v) for v in raw.split(',') if v.strip()]


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


def _fmt_rho(value):
    """4-dp rho for the printed lines; 'n/a' when the fold mean is None."""
    return 'n/a' if value is None else f'{value:.4f}'


def run(data, config, output_dir, dataset_name, **kwargs):
    """Hybrid distance tracks prediction uncertainty better than standard distance."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S06: APPLICABILITY DOMAIN")
    print("=" * 70)

    n_folds = config['n_splits']
    k_values = resolved_k_values(kwargs)
    ad_statistic = ad_statistic_name()

    print(f"  distance statistic {ad_statistic!r} over k={k_values} nearest "
          f"training molecules; importance exponent "
          f"{_importance_exponent_in_force()}")

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

    task_type = data['task_type']
    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    _rf_factory = get_model_factories(['rf'], task_type=task_type,
                                       n_samples=len(labels))['rf']

    results = {'distance_error_correlation': {}, 'reliability': {},
               'conformal': {}, 'selection_rule': _SELECTION_RULE,
               # Which number was taken off each molecule's k nearest training
               # molecules, and over how many. Both were fixed before
               # 2026-09-20 -- the mean, over 5 -- and neither was recorded, so
               # a result file could not say what its distances were.
               'distance_statistic': ad_statistic,
               'k_values': list(k_values),
               'importance_exponent': _importance_exponent_in_force()}

    augmentations = data.get('augmentations')
    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])

    # Get hybrid + blueprint for per-fold CV
    try:
        budget = kwargs.get('budget', _DEFAULT_AD_BUDGET)
        augmentations_search = data.get('augmentations')
        # 2026-09-12 (owner's decision): the searches this hybrid may be drawn
        # from now include the ones run under objective='similarity', whose
        # features were chosen so the distance geometry predicts each
        # molecule's activity from its five nearest neighbours -- not only the
        # ones chosen to make a downstream model predict. distance=True below
        # is a separate, older setting (it picks the evaluator and budget
        # lineage) and is left exactly as it was.
        X_hybrid, fi, search, _meta = get_fitted_pool_hybrid(
            reps, labels, groups,
            downstream_model='knn', budget=budget,
            augmentations=augmentations_search,
            metric=metric_name, task_type=task_type, domain=domain,
            dataset_name=dataset_name, distance=True,
            pool_subset='predictive_and_similarity')
        # Which family the winning recipe came from, so a reader can tell a
        # hybrid built to predict from one built to make the distance work.
        # The two families are scored in different units -- a predictive search
        # stores the dataset's own metric, a similarity search stores the
        # Spearman correlation between each held-out molecule's activity and
        # what its five nearest neighbours predict -- so every eligible
        # family's own best is recorded too, never merged into one number.
        if _meta:
            results['hybrid_family'] = _meta.get('objective')
            results['hybrid_family_choice'] = _meta.get('objective_choice')
            results['hybrid_metric'] = _meta.get('metric_name')
            results['pool_subset'] = _meta.get('pool_subset')
            results['pool_by_objective'] = {
                o: {'tier': d.get('tier'),
                    'aug_condition': d.get('aug_condition'),
                    'score_mean': d.get('score_mean'),
                    'metric_name': d.get('metric_name'),
                    'lower_is_better': d.get('lower_is_better')}
                for o, d in (_meta.get('by_objective') or {}).items()
            }
            print(f"  Hybrid built under objective="
                  f"{results['hybrid_family']!r} (pool subset "
                  f"{results['pool_subset']!r}, chosen by "
                  f"{results['hybrid_family_choice']!r}); its search score is "
                  f"in {results['hybrid_metric']!r}")
        # The comparator representation below is ranked out of this same
        # search, so it comes from whichever family won.
        best_config = get_best_for_downstream(search, 'knn')
        bp = best_config['blueprint'] if best_config and 'blueprint' in best_config else None
        # Pool helper overrides blueprint.augmentation_strategy to dodge the
        # 2026-04-21 cache-collision bug; mirror that override before per-fold
        # rebuilds so they match the pooled meta.
        if bp is not None and _meta and 'aug_condition' in _meta:
            bp.augmentation_strategy = _meta['aug_condition']

        # ⛔ THE GEOMETRY FAMILY'S OWN RECIPE, TAKEN BY NAME. Added 2026-09-19.
        #
        # POOL_SUBSETS names the two families in PREFERENCE order and the pool
        # returns the first that has an entry. A predictive recipe always
        # exists, so from 2026-09-12 to 2026-09-19 this file asked for both
        # families, built the predictive one and wrote the other into
        # 'pool_by_objective' as a score nothing had measured in.
        #
        # It matters here because the distance-error correlation and the
        # reliability diagram are both built on one quantity: a molecule's mean
        # distance to its k nearest TRAINING neighbours, in the space it is
        # read in. The similarity search chooses features so that exactly that
        # distance predicts activity from the five nearest neighbours. The
        # conformal block is not a distance -- it is a prediction interval from
        # a forest fitted on each arm's features -- so there the space changes
        # what the forest is given rather than which neighbours are found.
        #
        # 'hybrid' and 'hybrid_weighted' are unchanged and still the predictive
        # recipe. The two arms added are 'hybrid_similarity' and
        # 'hybrid_similarity_weighted'.
        _sim_entry = ((_meta or {}).get('by_objective') or {}).get('similarity')
        bp_sim = ((_sim_entry or {}).get('best_config') or {}).get('blueprint')
        if bp_sim is not None:
            _sim_aug = _sim_entry.get('aug_condition')
            if _sim_aug:
                bp_sim.augmentation_strategy = _sim_aug
            results['hybrid_similarity_available'] = True
            results['hybrid_similarity_budget'] = getattr(bp_sim, 'budget', None)
            results['hybrid_similarity_reps'] = sorted(
                getattr(bp_sim, 'reps', []) or [])
            results['hybrid_similarity_search_score_mean'] = _sim_entry.get(
                'score_mean')
            results['hybrid_similarity_search_metric_name'] = _sim_entry.get(
                'metric_name')
            results['hybrid_similarity_features_chosen_by'] = (
                describe_pool_family('similarity'))
            print(f"  Second hybrid arm: the similarity family's own recipe, "
                  f"{results['hybrid_similarity_budget']} features, search "
                  f"score {_sim_entry.get('score_mean')} in "
                  f"{_sim_entry.get('metric_name')!r}")
        else:
            results['hybrid_similarity_available'] = False
            results['hybrid_similarity_unavailable_reason'] = (
                'the pool held no search under objective=similarity for this '
                'dataset, so the arm whose features were chosen to make the '
                'neighbour distance predictive could not be built. Every '
                'other arm ran unchanged.'
                if 'similarity' not in ((_meta or {}).get('by_objective') or {})
                else 'the similarity family was in the pool but its winning '
                     'configuration carried no blueprint.')
            print(f"  no similarity-objective arm: "
                  f"{results['hybrid_similarity_unavailable_reason']}")
        top3 = extract_top3_base_reps(search)
        if top3:
            print(f"  Top-3 base reps (by RF): {', '.join(f'{n}={s:.4f}' for n, s in top3)}")
            results['top3_base_reps'] = [(n, s) for n, s in top3]
    except Exception as e:
        print(f"  Hybrid search failed: {e}")
        return {'error': str(e)}

    # The arm the hybrid is compared against is ONE base representation, never
    # every representation glued together. Until 2026-09-04 'concat' was the
    # only comparator in the reliability deciles and in the conformal block, and
    # it was one of the four printed arms of the distance-error correlation. A
    # concatenation is not a baseline in this project.
    #
    # Ranked by knn, which is what this experiment searched on; rf is the
    # fallback because extract_top3_base_reps defaults to it and some searches
    # carry no knn baseline.
    _cmp_ranked = (extract_top3_base_reps(search, model='knn')
                   or extract_top3_base_reps(search, model='rf'))
    if not _cmp_ranked:
        return {'error': "s06 has no ranked base representation to compare the "
                         "hybrid against; the search carried no knn or rf "
                         "baselines."}
    cmp_rep = _cmp_ranked[0][0]
    results['comparator_rep'] = cmp_rep
    print(f"  Comparator: {cmp_rep} (best single rep by "
          f"{'knn' if extract_top3_base_reps(search, model='knn') else 'rf'})")

    # ── Top-2 rows and the two hybrid deltas (owner's instruction, 2026-09-18) ──
    # Added beside what this file already reports. Nothing above or below reads
    # these four keys and no number already stored here changes. Every call
    # reads rows the search has already stored, so this costs 0 new model fits.
    #
    # What was absent without them. This file names ONE comparator
    # representation (`comparator_rep`, printed above) and measures distance in
    # ONE hybrid. So a reader could not tell whether the runner-up
    # representation was 0.001 or 0.2 behind it in the search's own metric, and
    # could not tell whether the hybrid beat the representations that fed it,
    # the best single representation, or neither.
    #
    # `top2_pairs_by_model` is asked for DISTANCE_MODELS -- the five downstream
    # models plus knn, 6 models -- and not select_model_rep_pairs' 5-model
    # default, because knn is the model every distance in this file is measured
    # for and the default list leaves it out (_shared.py:539, :542).
    #
    # require_full=False on every call: a downstream model this search never
    # scored comes back as a short list carrying its own 'shortfall_reason'
    # instead of raising and taking the run with it.
    try:
        results['top2_base_reps'] = top_base_reps(
            search, 'knn', 2, require_full=False,
            context='s06 top-2 base representations for knn')
        results['top2_hybrids'] = top_hybrids(
            search, 'knn', 2, require_full=False,
            context='s06 top-2 hybrid configurations for knn')
        results['top2_pairs_by_model'] = select_model_rep_pairs(
            search, models=DISTANCE_MODELS, n_base=2, n_hybrids=2,
            require_full=False, context='s06 top-2 grid')

        # The two deltas, one entry per downstream model.
        #   delta_vs_source_mean -- the hybrid against the MEAN of the base
        #     representations the budget split actually gave features to, read
        #     off Blueprint.allocation_per_fold, so a representation given 0
        #     features on every fold is not in that mean.
        #   delta_vs_best_rep -- the hybrid against the single best base
        #     representation for that same downstream model.
        # Positive means the hybrid is better, on either metric direction.
        #
        # The hybrid is handed over explicitly. Called as hybrid_deltas(search,
        # model) it reads _picked['selected'][0]['row'] (_shared.py:7424) and
        # top_hybrids' selected entries carry no 'row' key -- run in this
        # session against a stub search, that call raises KeyError: 'row'. The
        # entry passed instead carries 'score_mean' and 'blueprint', which are
        # the only two keys hybrid_deltas reads off it (_shared.py:7435, :7444).
        _deltas = {}
        for _m in DISTANCE_MODELS:
            try:
                _top1 = top_hybrids(search, _m, 1, require_full=False,
                                    context=f's06 hybrid deltas ({_m})')
                if not _top1['selected']:
                    _deltas[_m] = {'error': _top1['shortfall_reason']}
                    continue
                _deltas[_m] = hybrid_deltas(
                    search, _m, hybrid=_top1['selected'][0],
                    context=f's06 hybrid deltas ({_m})')
            except Exception as _de:
                _deltas[_m] = {'error': f'{type(_de).__name__}: {_de}'}
        results['hybrid_deltas_by_model'] = _deltas

        # Printed for knn only, the model this file's distances are measured
        # for. Both numbers are in the metric the search stored, which is the
        # dataset's own metric for a predictive search and a Spearman
        # correlation between each held-out molecule's activity and what its
        # five nearest neighbours predict for a similarity one -- so the metric
        # is named rather than assumed (results['hybrid_family'] above says
        # which family won).
        _sm = getattr(search, 'metric_name', None) or 'the search metric'
        _kd = _deltas.get('knn') or {}
        if _kd.get('delta_vs_source_mean') is not None:
            # n_source_reps counts the representations that BOTH got features
            # from the budget split AND carry a knn baseline score, which is
            # what the mean is actually over (_shared.py:7470-7480). A
            # representation given features but never scored for knn is
            # dropped from it and named in _kd['notes'].
            print(f"  knn hybrid minus the mean of the {_kd['n_source_reps']} "
                  f"representations that got features from the budget split "
                  f"and carry a knn score: "
                  f"{_kd['delta_vs_source_mean']:+.4f} {_sm}")
        if _kd.get('delta_vs_best_rep') is not None:
            print(f"  knn hybrid minus the best single representation "
                  f"({_kd.get('best_rep')}): "
                  f"{_kd['delta_vs_best_rep']:+.4f} {_sm} "
                  f"(positive = the hybrid is better)")
    except Exception as _e:
        print(f"  [warn] the top-2 rows and the hybrid deltas were not added: "
              f"{type(_e).__name__}: {_e}")

    splitter, use_groups = make_cv_splitter(n_folds, groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))

    for k in k_values:
        print(f"\n  k={k}:")
        all_d_cmp = []
        all_d_cmp_w = []
        all_d_hybrid = []
        all_d_weighted = []
        all_d_per_rep = {rname: [] for rname in reps}
        all_d_per_rep_w = {rname: [] for rname in reps}
        all_errors = []
        all_errors_neutral = []  # all-features RF (removes tautology)
        # F05: per-fold Spearman, one value per fold, kept separately from the
        # pooled lists above. Every fold refits its own StandardScaler, so a
        # correlation taken inside a fold never ranks one fold's distances
        # against another's.
        fold_rhos = {cmp_rep: [], f'{cmp_rep}_weighted': [],
                     'hybrid': [], 'hybrid_weighted': []}
        fold_rhos_neutral = {cmp_rep: [], f'{cmp_rep}_weighted': [],
                             'hybrid': [], 'hybrid_weighted': []}
        # The geometry family's own arms, when the pool held that family.
        all_d_hybrid_sim = []
        all_d_weighted_sim = []
        if bp_sim is not None:
            for _n in (SIM_ARM, SIM_ARM_W):
                fold_rhos[_n] = []
                fold_rhos_neutral[_n] = []
        # 2026-09-12 (owner's decision): every representation and its
        # importance-weighted twin are arms here too, not only in the pooled
        # block. Until now the per-fold block held four names and the
        # neutral-model block three, so a single representation could be read
        # pooled but not within a fold, and never against the errors of a model
        # that had not seen the hybrid feature space.
        for _rname in reps:
            fold_rhos.setdefault(_rname, [])
            fold_rhos.setdefault(f'{_rname}_weighted', [])
            fold_rhos_neutral.setdefault(_rname, [])
            fold_rhos_neutral.setdefault(f'{_rname}_weighted', [])

        for train_idx, val_idx in splitter.split(sample_X, labels, groups=groups) if use_groups else splitter.split(sample_X, labels):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Per-fold hybrid using search blueprint.
            # NOTE: use a distinct comprehension variable (aug_k) — NOT `k` —
            # so the outer `for k in k_values:` loop variable is never at risk of
            # being shadowed (Python-3 comprehensions are scoped, but a bare
            # `for k,...` would leak; keep this unambiguous so every downstream
            # min(k, len(y_train)-1) uses the integer neighbour count, never a key).
            train_reps_h = {name: rep[train_idx] for name, rep in reps.items()}
            val_reps_h = {name: rep[val_idx] for name, rep in reps.items()}
            aug_tr = {aug_k: v[train_idx] for aug_k, v in augmentations.items()} if augmentations else None
            aug_va = {aug_k: v[val_idx] for aug_k, v in augmentations.items()} if augmentations else None
            try:
                X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                    train_reps_h, val_reps_h, y_train, bp,
                    augmentations_train=aug_tr, augmentations_val=aug_va,
                    groups=(groups[train_idx] if groups is not None else None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=_rf_factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))

                # Importance weights for this fold (also weight the comparator)
                # Same importance method the search used to build this
                # hybrid; the comparator is weighted with it too.
                imp = compute_feature_importance(
                    train_reps_h, y_train,
                    method=getattr(bp, 'importance_method', 'lightgbm_gain'))
                # Use the shared helper: handles the augmentation tail so the
                # weight vector matches X_tr_h/X_va_h column count exactly.
                # NaN-guard: importance helpers can emit NaN for constant
                # features; np.maximum(NaN, 0) is NaN which would corrupt the
                # weighted kNN distance.
                w = importance_feature_weights(
                    flatten_fold_weights(feat_info))
            except Exception as exc:
                # N9: log the dropped fold so a thrown hybrid/importance build
                # is distinguishable from a genuinely skipped fold.
                print(f"      AD hybrid weight fold failed: "
                      f"{type(exc).__name__}: {exc}")
                continue

            # Train RF on hybrid features (existing — may have tautological AD advantage)
            scaler_h = StandardScaler()
            X_tr_s = scaler_h.fit_transform(X_tr_h)
            X_va_s = scaler_h.transform(X_va_h)
            rf = _rf_factory()
            rf.fit(X_tr_s, y_train)
            y_pred = rf.predict(X_va_s)
            # Classification: graded error = 1 - P(true class). A 0/1
            # misclassification indicator makes Spearman(dist, error) a
            # degenerate, low-power statistic; the predicted-probability of the
            # true label is the continuous AD analogue. Regression: |residual|.
            if task_type == 'classification':
                errors = _graded_clf_error(rf, X_va_s, y_val)
            else:
                errors = np.abs(y_val - y_pred)

            # Neutral error model: an RF fitted on every representation glued
            # together. This is NOT an arm and nothing is ranked against it — it
            # exists only to produce prediction errors that neither the hybrid
            # nor the comparator representation supplied the features for, so
            # neither one is being scored against its own model. Fitting it on a
            # single representation would just move the tautology onto that arm.
            scaler_c = StandardScaler()
            Xtr_c = scaler_c.fit_transform(X_concat[train_idx])
            Xva_c = scaler_c.transform(X_concat[val_idx])
            rf_neutral = _rf_factory()
            rf_neutral.fit(Xtr_c, y_train)
            y_pred_neutral = rf_neutral.predict(Xva_c)
            if task_type == 'classification':
                errors_neutral = _graded_clf_error(rf_neutral, Xva_c, y_val)
            else:
                errors_neutral = np.abs(y_val - y_pred_neutral)

            # Comparator arm: the best single base representation, scaled on
            # this fold's training rows. Its importance weights fall back to
            # ones when the fold produced none, which matches what the old
            # concat arm did for a representation the search never scored.
            scaler_cmp = StandardScaler()
            Xtr_cmp = scaler_cmp.fit_transform(reps[cmp_rep][train_idx])
            Xva_cmp = scaler_cmp.transform(reps[cmp_rep][val_idx])
            w_cmp = (imp[cmp_rep] if cmp_rep in imp
                     else np.ones(reps[cmp_rep].shape[1]))
            w_cmp = importance_feature_weights(w_cmp)

            # Distance to k-NN in training set for each val molecule
            # Comparator representation (unweighted)
            nn_c = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_c.fit(Xtr_cmp)
            dists_c, _ = nn_c.kneighbors(Xva_cmp)
            d_cmp = _ad_statistic(dists_c, k)

            # Comparator representation (weighted)
            nn_cw = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_cw.fit(Xtr_cmp * w_cmp)
            dists_cw, _ = nn_cw.kneighbors(Xva_cmp * w_cmp)
            d_cmp_w = _ad_statistic(dists_cw, k)

            # Per-rep distances (unweighted + weighted)
            for rname, X_rep in reps.items():
                scaler_r = StandardScaler()
                Xtr_r = scaler_r.fit_transform(X_rep[train_idx])
                Xva_r = scaler_r.transform(X_rep[val_idx])
                nn_r = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
                nn_r.fit(Xtr_r)
                dr, _ = nn_r.kneighbors(Xva_r)
                d_rep = _ad_statistic(dr, k)
                all_d_per_rep[rname].extend(d_rep.tolist())
                # The comparator representation is appended by the block below,
                # which computes the same distance; appending it twice here
                # would put two entries per fold in one arm's list.
                if rname != cmp_rep:
                    fold_rhos[rname].append(_fold_spearman(d_rep, errors))
                    fold_rhos_neutral[rname].append(
                        _fold_spearman(d_rep, errors_neutral))

                if rname in imp:
                    w_r = importance_feature_weights(imp[rname])
                    nn_rw = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
                    nn_rw.fit(Xtr_r * w_r)
                    drw, _ = nn_rw.kneighbors(Xva_r * w_r)
                    d_rep_w = _ad_statistic(drw, k)
                    all_d_per_rep_w[rname].extend(d_rep_w.tolist())
                    if rname != cmp_rep:
                        fold_rhos[f'{rname}_weighted'].append(
                            _fold_spearman(d_rep_w, errors))
                        fold_rhos_neutral[f'{rname}_weighted'].append(
                            _fold_spearman(d_rep_w, errors_neutral))
                elif rname != cmp_rep:
                    # No importance for this representation on this fold, so
                    # the weighted arm has no value here. NaN keeps the list one
                    # entry per fold; _mean_over_folds drops it.
                    fold_rhos[f'{rname}_weighted'].append(float('nan'))
                    fold_rhos_neutral[f'{rname}_weighted'].append(float('nan'))

            # Hybrid space (unweighted)
            nn_h = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_h.fit(X_tr_s)
            dists_h, _ = nn_h.kneighbors(X_va_s)
            d_hybrid = _ad_statistic(dists_h, k)

            # Weighted hybrid space
            nn_w = make_nearest_neighbors(n_neighbors=min(k, len(y_train) - 1))
            nn_w.fit(X_tr_s * w)
            dists_w, _ = nn_w.kneighbors(X_va_s * w)
            d_weighted = _ad_statistic(dists_w, k)

            # The geometry family's own hybrid, same fold, same rows, same two
            # distances. Its distances are scored against the SAME two error
            # vectors every other arm is scored against, so nothing extra is
            # fitted: one build and two neighbour searches per fold.
            # fold_hybrid_scoring_kwargs drops the model, the metric and its
            # direction under a geometry recipe, where build_fold_hybrid
            # refuses the pair.
            d_hybrid_sim = d_weighted_sim = None
            if bp_sim is not None:
                try:
                    X_tr_h2, X_va_h2, feat_info2 = build_fold_hybrid(
                        train_reps_h, val_reps_h, y_train, bp_sim,
                        augmentations_train=aug_tr, augmentations_val=aug_va,
                        groups=(groups[train_idx] if groups is not None
                                else None),
                        **fold_hybrid_scoring_kwargs(
                            bp_sim, model=_rf_factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                    sc_h2 = StandardScaler()
                    X_tr_s2 = sc_h2.fit_transform(X_tr_h2)
                    X_va_s2 = sc_h2.transform(X_va_h2)
                    w2 = importance_feature_weights(
                        flatten_fold_weights(feat_info2))
                    nn_h2 = make_nearest_neighbors(
                        n_neighbors=min(k, len(y_train) - 1))
                    nn_h2.fit(X_tr_s2)
                    d_hybrid_sim = _ad_statistic(
                        nn_h2.kneighbors(X_va_s2)[0], k)
                    nn_w2 = make_nearest_neighbors(
                        n_neighbors=min(k, len(y_train) - 1))
                    nn_w2.fit(X_tr_s2 * w2)
                    d_weighted_sim = _ad_statistic(
                        nn_w2.kneighbors(X_va_s2 * w2)[0], k)
                except Exception as exc:
                    print(f"      {SIM_ARM} fold build failed: "
                          f"{type(exc).__name__}: {exc}")
                    d_hybrid_sim = d_weighted_sim = None
                if d_hybrid_sim is not None:
                    all_d_hybrid_sim.extend(d_hybrid_sim.tolist())
                    all_d_weighted_sim.extend(d_weighted_sim.tolist())
                else:
                    # One entry per fold in every arm's list, so a fold that
                    # failed here does not shift the other arms' folds.
                    fold_rhos[SIM_ARM].append(float('nan'))
                    fold_rhos[SIM_ARM_W].append(float('nan'))
                    fold_rhos_neutral[SIM_ARM].append(float('nan'))
                    fold_rhos_neutral[SIM_ARM_W].append(float('nan'))

            all_d_cmp.extend(d_cmp.tolist())
            all_d_cmp_w.extend(d_cmp_w.tolist())
            all_d_hybrid.extend(d_hybrid.tolist())
            all_d_weighted.extend(d_weighted.tolist())
            all_errors.extend(errors.tolist())
            all_errors_neutral.extend(errors_neutral.tolist())

            # Per-fold correlations, computed before the fold's distances are
            # pooled. Top block scores against `errors` (RF fitted on this
            # fold's hybrid features); neutral block scores the same distances
            # against `errors_neutral` (RF fitted on concat features).
            # One loop for both blocks: the neutral-model block now carries the
            # weighted comparator too, so the two name the same arms.
            _fold_arms = [(cmp_rep, d_cmp),
                          (f'{cmp_rep}_weighted', d_cmp_w),
                          ('hybrid', d_hybrid),
                          ('hybrid_weighted', d_weighted)]
            if d_hybrid_sim is not None:
                _fold_arms += [(SIM_ARM, d_hybrid_sim),
                               (SIM_ARM_W, d_weighted_sim)]
            for _name, _d in _fold_arms:
                fold_rhos[_name].append(_fold_spearman(_d, errors))
                fold_rhos_neutral[_name].append(
                    _fold_spearman(_d, errors_neutral))

        if not all_errors:
            continue

        all_d_cmp = np.array(all_d_cmp)
        all_d_cmp_w = np.array(all_d_cmp_w)
        all_d_hybrid = np.array(all_d_hybrid)
        all_d_weighted = np.array(all_d_weighted)
        all_errors = np.array(all_errors)

        # Spearman correlations: distance vs |error| (hybrid-model errors),
        # over the arrays pooled across all folds.
        rho_c, _ = spearmanr(all_d_cmp, all_errors)
        rho_cw, _ = spearmanr(all_d_cmp_w, all_errors)
        rho_h, _ = spearmanr(all_d_hybrid, all_errors)
        rho_w, _ = spearmanr(all_d_weighted, all_errors)

        corr_result = {
            # Unchanged keys: pooled-over-folds Spearman against the errors of
            # the RF fitted on the per-fold hybrid features.
            cmp_rep: float(rho_c),
            f'{cmp_rep}_weighted': float(rho_cw),
            'hybrid': float(rho_h),
            'hybrid_weighted': float(rho_w),
            'error_source': 'rf_fitted_on_hybrid_features',
            'hybrid_note': (
                "'hybrid' is built from the predictive search's recipe. "
                "'hybrid_similarity', when present, is built from the "
                "similarity search's own recipe: features chosen so that each "
                "molecule's activity is predicted by its five nearest "
                'neighbours in the distance they define. Both are scored here '
                'against the same two error vectors, so they are comparable.'),
            'error_source_note': (
                "the 'hybrid' and 'hybrid_weighted' distances here are measured "
                "in the same feature space this error model was fitted on "
                "(tautological pair); ['neutral_model'] holds the "
                "de-tautologised pair"),
            'fold_handling': 'pooled_across_folds',
            'fold_handling_note': (
                "distances and errors from every fold concatenated, one Spearman "
                "over the pooled arrays; each fold refit its own StandardScaler, "
                "so cross-fold scale differences enter this rank ordering; "
                "['per_fold_mean'] holds the within-fold version"),
        }

        # The geometry family's pooled numbers, against the same errors.
        if all_d_hybrid_sim and len(all_d_hybrid_sim) == len(all_errors):
            _sim_d = np.array(all_d_hybrid_sim)
            _sim_dw = np.array(all_d_weighted_sim)
            corr_result[SIM_ARM] = float(spearmanr(_sim_d, all_errors)[0])
            corr_result[SIM_ARM_W] = float(spearmanr(_sim_dw, all_errors)[0])

        # F05: per-fold means alongside the pooled values above.
        per_fold_mean = {
            'error_source': 'rf_fitted_on_hybrid_features',
            'fold_handling': 'spearman_within_fold_then_mean_over_folds',
        }
        for _name, _vals in fold_rhos.items():
            _m, _n = _mean_over_folds(_vals)
            per_fold_mean[_name] = _m
            per_fold_mean[f'{_name}_n_folds'] = _n
            per_fold_mean[f'{_name}_per_fold'] = [float(v) for v in _vals]
        corr_result['per_fold_mean'] = per_fold_mean

        # Neutral-model correlations (errors from RF on concat — removes tautology)
        all_errors_neutral = np.array(all_errors_neutral)
        if len(all_errors_neutral) == len(all_d_cmp):
            rho_c_n, _ = spearmanr(all_d_cmp, all_errors_neutral)
            rho_h_n, _ = spearmanr(all_d_hybrid, all_errors_neutral)
            rho_w_n, _ = spearmanr(all_d_weighted, all_errors_neutral)
            neutral_per_fold = {
                'error_source': 'rf_fitted_on_concat_features',
                'fold_handling': 'spearman_within_fold_then_mean_over_folds',
            }
            for _name, _vals in fold_rhos_neutral.items():
                _m, _n = _mean_over_folds(_vals)
                neutral_per_fold[_name] = _m
                neutral_per_fold[f'{_name}_n_folds'] = _n
                neutral_per_fold[f'{_name}_per_fold'] = [float(v) for v in _vals]
            _sim_neutral = {}
            if (all_d_hybrid_sim
                    and len(all_d_hybrid_sim) == len(all_errors_neutral)):
                _sim_neutral = {
                    SIM_ARM: float(spearmanr(np.array(all_d_hybrid_sim),
                                             all_errors_neutral)[0]),
                    SIM_ARM_W: float(spearmanr(np.array(all_d_weighted_sim),
                                               all_errors_neutral)[0]),
                }
            corr_result['neutral_model'] = {
                cmp_rep: float(rho_c_n),
                'hybrid': float(rho_h_n),
                'hybrid_weighted': float(rho_w_n),
                **_sim_neutral,
                'error_source': 'rf_fitted_on_concat_features',
                'error_source_note': (
                    'errors come from a model that never sees the hybrid feature '
                    'space, so these distances and these errors do not share a '
                    'feature space (de-tautologised pair)'),
                'fold_handling': 'pooled_across_folds',
                'per_fold_mean': neutral_per_fold,
            }
            print(f"    Neutral-model AD [errors from RF on concat features; "
                  f"de-tautologised pair], pooled over folds: "
                  f"{cmp_rep}={rho_c_n:.4f}, hybrid={rho_h_n:.4f}, "
                  f"hybrid_w={rho_w_n:.4f}")
            print(f"      same, per-fold mean ({neutral_per_fold['hybrid_n_folds']} "
                  f"folds): {cmp_rep}={_fmt_rho(neutral_per_fold[cmp_rep])}, "
                  f"hybrid={_fmt_rho(neutral_per_fold['hybrid'])}, "
                  f"hybrid_w={_fmt_rho(neutral_per_fold['hybrid_weighted'])}")

        # Per-rep correlations
        _neutral = corr_result.get('neutral_model')
        for rname in reps:
            if all_d_per_rep[rname]:
                rho_r, _ = spearmanr(np.array(all_d_per_rep[rname]), all_errors)
                corr_result[rname] = float(rho_r)
                # The same distances against the errors of the model that never
                # saw the hybrid feature space, so every representation has a
                # de-tautologised number too, not only the comparator.
                if (_neutral is not None
                        and len(all_d_per_rep[rname]) == len(all_errors_neutral)):
                    _rr, _ = spearmanr(np.array(all_d_per_rep[rname]),
                                       all_errors_neutral)
                    _neutral[rname] = float(_rr)
            if all_d_per_rep_w[rname]:
                rho_rw, _ = spearmanr(np.array(all_d_per_rep_w[rname]), all_errors)
                corr_result[f'{rname}_weighted'] = float(rho_rw)
                if (_neutral is not None
                        and len(all_d_per_rep_w[rname]) == len(all_errors_neutral)):
                    _rrw, _ = spearmanr(np.array(all_d_per_rep_w[rname]),
                                        all_errors_neutral)
                    _neutral[f'{rname}_weighted'] = float(_rrw)

        results['distance_error_correlation'][f'k{k}'] = corr_result
        print(f"    Same-space AD [errors from RF fitted on the hybrid features, "
              f"the space hybrid/hybrid_w distances are measured in; "
              f"tautological pair], pooled over folds: "
              f"{cmp_rep}={rho_c:.4f}, {cmp_rep}_w={rho_cw:.4f}, "
              f"hybrid={rho_h:.4f}, hybrid_w={rho_w:.4f}")
        print(f"      same, per-fold mean ({per_fold_mean['hybrid_n_folds']} folds): "
              f"{cmp_rep}={_fmt_rho(per_fold_mean[cmp_rep])}, "
              f"{cmp_rep}_w={_fmt_rho(per_fold_mean[f'{cmp_rep}_weighted'])}, "
              f"hybrid={_fmt_rho(per_fold_mean['hybrid'])}, "
              f"hybrid_w={_fmt_rho(per_fold_mean['hybrid_weighted'])}")
        print(f"      keys: {cmp_rep!r}/'hybrid'/... = pooled; "
              f"['per_fold_mean'] = within-fold mean; ['neutral_model'] = "
              f"errors from the all-features model")

        # Reliability diagram: bin by distance decile.
        # 2026-09-12: every representation and its weighted twin are arms here
        # too. The comparator's own arrays are kept where a name appears twice.
        _reliability_arms = {cmp_rep: all_d_cmp,
                             f'{cmp_rep}_weighted': all_d_cmp_w,
                             'hybrid': all_d_hybrid,
                             'hybrid_weighted': all_d_weighted}
        if all_d_hybrid_sim:
            _reliability_arms[SIM_ARM] = np.array(all_d_hybrid_sim)
            _reliability_arms[SIM_ARM_W] = np.array(all_d_weighted_sim)
        for rname in reps:
            if all_d_per_rep[rname]:
                _reliability_arms.setdefault(rname,
                                             np.array(all_d_per_rep[rname]))
            if all_d_per_rep_w[rname]:
                _reliability_arms.setdefault(f'{rname}_weighted',
                                             np.array(all_d_per_rep_w[rname]))
        for dist_name, dists in _reliability_arms.items():
            dists = np.asarray(dists)
            if len(dists) != len(all_errors):
                # An arm that lost a fold cannot be binned against errors it is
                # not row-aligned with.
                continue
            deciles = np.percentile(dists, np.arange(0, 110, 10))
            decile_errors = []
            decile_counts = []
            for di in range(10):
                mask = (dists >= deciles[di]) & (dists < deciles[di + 1])
                if di == 9:  # include upper bound in last bin
                    mask = (dists >= deciles[di]) & (dists <= deciles[di + 1])
                if mask.sum() > 0:
                    decile_errors.append(float(np.mean(all_errors[mask])))
                    decile_counts.append(int(mask.sum()))
                else:
                    decile_errors.append(float('nan'))
                    decile_counts.append(0)

            results['reliability'].setdefault(f'k{k}', {})[dist_name] = {
                'decile_errors': decile_errors,
                'decile_counts': decile_counts,
            }

    # Conformal calibration comparison (regression only — residual-based intervals)
    if task_type == 'classification':
        print("\n  Conformal calibration: SKIP (regression-only, not applicable to classification)")
        results['conformal'] = {'skipped': True, 'reason': 'classification'}
    else:
        print("\n  Conformal calibration (CV):")
        try:
            splitter_conf, use_groups_conf = make_cv_splitter(
                n_folds, groups, task_type=task_type)
            # 2026-09-12 (owner's decision): every representation gets an
            # interval, not only the comparator. One random forest per
            # representation per fold, plus the hybrid's.
            _conf_arms = list(reps) + ['hybrid']
            if bp_sim is not None:
                _conf_arms.append(SIM_ARM)
            conf_picp = {name: [] for name in _conf_arms}
            conf_mpiw = {name: [] for name in _conf_arms}
            alpha = 0.1

            for train_idx, val_idx in (splitter_conf.split(sample_X, labels, groups=groups) if use_groups_conf else splitter_conf.split(sample_X, labels)):
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Per-fold hybrid using search blueprint (avoids data leakage)
                train_reps_fold = {k: v[train_idx] for k, v in reps.items()}
                val_reps_fold = {k: v[val_idx] for k, v in reps.items()}
                aug_tr_c = {k: v[train_idx] for k, v in augmentations.items()} if augmentations else None
                aug_va_c = {k: v[val_idx] for k, v in augmentations.items()} if augmentations else None
                try:
                    X_hybrid_tr, X_hybrid_va, fi = build_fold_hybrid(
                        train_reps_fold, val_reps_fold, y_train, bp,
                        augmentations_train=aug_tr_c, augmentations_val=aug_va_c,
                        groups=(groups[train_idx] if groups is not None else None),
                        **fold_hybrid_scoring_kwargs(
                            bp, model=_rf_factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as exc:
                    # N9: log the dropped fold so a thrown hybrid build is
                    # distinguishable from a genuinely skipped fold.
                    print(f"      AD calibration fold hybrid build failed: "
                          f"{type(exc).__name__}: {exc}")
                    continue

                # The geometry family's own recipe, same fold, same split.
                X_hyb_sim_tr = X_hyb_sim_va = None
                if bp_sim is not None:
                    try:
                        X_hyb_sim_tr, X_hyb_sim_va, _fi2 = build_fold_hybrid(
                            train_reps_fold, val_reps_fold, y_train, bp_sim,
                            augmentations_train=aug_tr_c,
                            augmentations_val=aug_va_c,
                            groups=(groups[train_idx] if groups is not None
                                    else None),
                            **fold_hybrid_scoring_kwargs(
                                bp_sim, model=_rf_factory(),
                                metric_fn=metric_fn,
                                lower_is_better=lower_is_better))
                    except Exception as exc:
                        print(f"      AD calibration fold {SIM_ARM} build "
                              f"failed: {type(exc).__name__}: {exc}")
                        X_hyb_sim_tr = X_hyb_sim_va = None

                # Split train into proper-train and calibration (group-aware)
                train_groups = groups[train_idx] if groups is not None else None
                if train_groups is not None and len(np.unique(train_groups)) >= 2:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    tr_local, cal_local = next(gss.split(
                        np.arange(len(y_train)), y_train, groups=train_groups))
                else:
                    n_cal = max(int(len(y_train) * 0.2), 10)
                    rng = np.random.RandomState(42)
                    perm = rng.permutation(len(y_train))
                    cal_local, tr_local = perm[:n_cal], perm[n_cal:]

                for feat_name, X_tr_all, X_va in (
                    [(rname, reps[rname][train_idx], reps[rname][val_idx])
                     for rname in reps]
                    + [('hybrid', X_hybrid_tr, X_hybrid_va)]
                    + ([(SIM_ARM, X_hyb_sim_tr, X_hyb_sim_va)]
                       if X_hyb_sim_tr is not None else [])
                ):
                    scaler = StandardScaler()
                    X_tr_all_s = scaler.fit_transform(X_tr_all)
                    X_va_s = scaler.transform(X_va)

                    X_tr = X_tr_all_s[tr_local]
                    X_cal = X_tr_all_s[cal_local]
                    y_tr = y_train[tr_local]
                    y_cal = y_train[cal_local]

                    # Train RF, compute conformal quantile on calibration set
                    rf = _rf_factory()
                    rf.fit(X_tr, y_tr)

                    # Calibration: residuals normalized by tree variance
                    y_pred_cal = rf.predict(X_cal)
                    tree_preds_cal = np.array([t.predict(X_cal) for t in rf.estimators_])
                    cal_std = tree_preds_cal.std(axis=0) + 1e-10
                    conformity = np.abs(y_cal - y_pred_cal) / cal_std
                    q = np.quantile(conformity, 1 - alpha)

                    # Val: predict + interval
                    y_pred_va = rf.predict(X_va_s)
                    tree_preds_va = np.array([t.predict(X_va_s) for t in rf.estimators_])
                    va_std = tree_preds_va.std(axis=0)
                    interval_width = 2 * q * va_std

                    picp = float(np.mean(np.abs(y_val - y_pred_va) <= interval_width / 2))
                    mpiw = float(np.mean(interval_width))
                    conf_picp[feat_name].append(picp)
                    conf_mpiw[feat_name].append(mpiw)

            results['conformal'] = {'nominal_coverage': 1 - alpha}
            for feat_name in _conf_arms:
                if conf_picp[feat_name]:
                    picp_mean = float(np.mean(conf_picp[feat_name]))
                    mpiw_mean = float(np.mean(conf_mpiw[feat_name]))
                    # undercoverage = how far PICP falls below the 90% target.
                    # A smaller MPIW (tighter interval) is ONLY an improvement
                    # if undercoverage stays ~0; tighter-but-undercovered is not.
                    undercov = float((1 - alpha) - picp_mean)
                    results['conformal'][feat_name] = {
                        'picp': picp_mean, 'mpiw': mpiw_mean,
                        'undercoverage': undercov,
                    }
                    print(f"    {feat_name.capitalize()}:  PICP={picp_mean:.3f}, "
                          f"MPIW={mpiw_mean:.3f}, undercoverage={undercov:+.3f}")
            # Honest comparison: hybrid's tighter interval only counts as better
            # if it does not lose coverage relative to the representation it is
            # put against (tolerance 0.02).
            #
            # 2026-09-12: taken against every representation, not the
            # comparator alone. 'hybrid_genuinely_tighter' keeps its meaning —
            # the comparator — for the readers that already read it, and
            # 'hybrid_genuinely_tighter_vs_all' is true only when it holds
            # against every representation that got an interval.
            if 'hybrid' in results['conformal']:
                h = results['conformal']['hybrid']
                _tighter = {}
                for feat_name in _conf_arms:
                    if (feat_name in ('hybrid', SIM_ARM)
                            or feat_name not in results['conformal']):
                        continue
                    c = results['conformal'][feat_name]
                    _tighter[feat_name] = bool(
                        h['mpiw'] < c['mpiw'] and h['picp'] >= c['picp'] - 0.02)
                if _tighter:
                    results['conformal']['hybrid_genuinely_tighter_by_rep'] = _tighter
                    results['conformal']['hybrid_genuinely_tighter_vs_all'] = bool(
                        all(_tighter.values()))
                    results['conformal']['n_reps_hybrid_tighter_than'] = int(
                        sum(_tighter.values()))
                    results['conformal']['n_reps_with_intervals'] = len(_tighter)
                if cmp_rep in _tighter:
                    results['conformal']['hybrid_genuinely_tighter'] = _tighter[cmp_rep]
                    print(f"    hybrid genuinely tighter than {cmp_rep}: "
                          f"{_tighter[cmp_rep]}; than "
                          f"{sum(_tighter.values())} of {len(_tighter)} "
                          f"representations")
        except Exception as e:
            print(f"    Conformal failed: {e}")
            results['conformal'] = {'error': str(e)}

    # Name the comparator by the quantity THIS FILE measures.
    #
    # `comparator_rep` above is extract_top3_base_reps(search, model='knn')[0]:
    # the representation a kNN predicts the LABEL best from, by RMSE. What s06
    # reports is the Spearman correlation between a held-out molecule's distance
    # to its neighbours and a model's absolute residual. Different quantities.
    #
    # Measured 2026-09-24 by ranking all 14 single-representation arms already
    # stored here by the reported correlation: the named comparator came 13th of
    # 14 on caco2, 8th on herg_ki and 10th on adme_mdr1 — bottom half on every
    # dataset, while the printed headline and the conformal
    # `hybrid_genuinely_tighter` flag compared the hybrid against it.
    #
    # Every representation is scored either way, so NO stored number changes.
    # What changes is which arm the headline names. The kNN-RMSE choice is kept
    # at `comparator_rep_by_knn_rmse` so any earlier result stays readable, and
    # KIRBY_S06_COMPARATOR=knn_rmse restores the old behaviour.
    try:
        _mode = os.environ.get('KIRBY_S06_COMPARATOR', '').strip().lower()
        _dec = results.get('distance_error_correlation') or {}
        _kk = sorted(_dec, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        if _mode != 'knn_rmse' and _kk:
            _arms = _dec[_kk[0]]
            _singles = {n: v for n, v in _arms.items()
                        if isinstance(v, (int, float)) and not n.startswith('hybrid')}
            if _singles:
                _best = max(_singles, key=_singles.get)
                _old = results.get('comparator_rep')
                _rank = (sorted(_singles, key=_singles.get, reverse=True).index(_old) + 1
                         if _old in _singles else None)
                results['comparator_rep_by_knn_rmse'] = _old
                results['comparator_rep'] = _best
                results['comparator_basis'] = {
                    'ranked_by': 'distance-error correlation, the quantity s06 reports',
                    'previous': _old,
                    'previous_rank_by_reported_quantity': _rank,
                    'n_single_rep_arms': len(_singles),
                    'note': ('Every representation is scored either way; this '
                             'only decides which arm the headline and the '
                             'conformal hybrid_genuinely_tighter flag name. '
                             'KIRBY_S06_COMPARATOR=knn_rmse restores the old '
                             'choice.')}
                if _best != _old:
                    print(f"  Comparator re-pointed: {_old} (rank "
                          f"{_rank} of {len(_singles)} by the reported "
                          f"quantity) -> {_best}")
    except Exception as _e:
        results['comparator_basis'] = {'error': str(_e)}

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S06: Applicability Domain',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5],
                        help='k values for kNN distance (default: [5])')
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")
    print(f"k values: {args.k_values}")

    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset,
                  k_values=args.k_values)
    save_experiment_results(results, args.output, 's06_applicability_domain', args.dataset)
