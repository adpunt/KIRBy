#!/usr/bin/env python3
"""S12: Cross-Domain Activity Cliff Analysis — generic cliff detection for all domains.

Paper: Similarity
Companion to S04 (which uses ECFP4 Tanimoto, molecular-only). S12 measures
similarity on the standardised concatenation of every representation, making it
applicable to antibody, TCGA, and molecular datasets alike. Same approach as
N01's non-molecular branch.

Cliff definition (2026-09-20): the same one s04 uses, in
similarity_helpers.calibrated_cliff_rule. On a molecular dataset with SMILES the
structural side is ECFP4 Tanimoto at or above 0.55, so this file and s04 flag
the same molecules; otherwise it is 1/(1+d) on the standardised concatenation of
every representation, at a fixed percentile of pairs. The activity gap is solved
per dataset so a stated share of molecules are cliffs, instead of being a fixed
multiple of the labels' median absolute deviation. What that fixed, and the
counts before and after, are written above calibrated_cliff_rule.

Hybrid recipes (2026-09-12, owner's decision): read from the pool, ranked for
each held-out fold on the other folds' scores. This file runs no search of its
own. The subset is regular and augmented only — no hybrids built under
objective='similarity' — because the cliffs are defined on the standardised
concatenation of every representation, a space outside every hybrid, so an arm
whose features were picked by distance could not be attributed.

Usage:
    python s12_cross_domain_cliffs.py --dataset flab_thermo_tresanco --mode intermediary
    python s12_cross_domain_cliffs.py --dataset tcga_luad_vs_lusc --mode intermediary
    python s12_cross_domain_cliffs.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, get_precomputed_search,
    get_fold_pool_recipes,
    build_fold_hybrid, DOWNSTREAM_MODELS,
    fold_reselection,
    hr_exclude_folds,
)
from experiments.similarity.similarity_helpers import (
    calibrated_cliff_rule, cliff_structural_similarity, CLIFF_TARGET_PCT)
from kirby.find_best_hybrid import nested_cv_outer_folds

# 2026-09-12, owner's decision: s12 stops calling get_fold_best_search and reads
# the pool instead, chosen per fold from the other folds' scores. The pool's
# per-fold scores come from a search run with 5 folds at seed 42, so this file
# must split the same way for its fold i to be the pool's fold i.
_FOLD_SEARCH_SEED = 42

# 2026-09-12, owner's decision: the subset is regular and augmented ONLY — no
# hybrids built under objective='similarity'. s12's cliffs are defined on the
# standardised concatenation of every representation, a space outside every
# hybrid, so an arm whose features were picked by distance could not be
# attributed. 'predictive_only' is also the default; it is written out to say so.
_POOL_SUBSET = 'predictive_only'

# How many hybrid configurations the pool returns per fold. Each one is built
# and scored on the fold; none is dropped.
_N_POOL_HYBRIDS = 3

from kirby.hybrid import create_hybrid
from kirby.hybrid import apply_feature_selection
from kirby.utils.noise_mitigation_methods import compute_distance_matrix
# The process-wide distance coefficient compute_distance_matrix falls back to
# when no metric is named. Read, never asserted: it has been manhattan since
# 2026-09-04 and the result file used to record 'euclidean' regardless.
from kirby.similarity.distance_config import get_distance_metric


def _finite_mean(fold_scores):
    """Mean over finite fold scores (NaN = fold that had no usable cliff/non-cliff
    subset); NaN if none are finite. Local helper: the cliff/non-cliff CV scores are
    a redesigned segmented statistic that never flows through a SearchResult, so the
    shared arithmetic in _shared.py does not fit here. Kept fold-aligned (NaN
    placeholders) so fold_reselection can index the same fold across reps/hybrid."""
    finite = [s for s in fold_scores if s is not None and not np.isnan(s)]
    return float(np.mean(finite)) if finite else float('nan')


# =========================================================================
# CLIFF DETECTION (domain-generic, mirrors N01's identify_cliff_pairs)
# =========================================================================

def identify_cliff_pairs(X_concat, y, sim_threshold=0.55, activity_k=2.0,
                         max_compounds=5000, is_clf=False,
                         similar_percentile=None):
    """Identify activity cliff pairs on the stacked representation.

    ⛔ 2026-09-20: run() no longer calls this. It is the rule that disagreed
    with s04 — on herg_ki it called 295 of 1412 molecules cliffs where s04
    called 451 — and its percentile auto-select never reached its own stated 10
    per cent target on caco2, herg_ki or adme_mdr1, because even its smallest
    candidate overshot. Kept because smoke_tests_broken_fixes.py exercises it
    through identify_cliff_pairs_euclidean. New callers want
    similarity_helpers.calibrated_cliff_rule.

    sim = 1/(1+d) on standardised features, where d comes from
    compute_distance_matrix with no metric named — so it is whatever
    kirby.similarity.distance_config holds, manhattan since 2026-09-04, not
    Euclidean. The coefficient actually used is written into info.

    Cliff = high structural similarity AND high activity difference.

    Regression / binary: |delta_y| > k * MAD(y).
    Classification (is_clf): different class (high sim + different label).

    Args:
        sim_threshold: absolute similarity cutoff (used only when
            ``similar_percentile`` is None).
        similar_percentile: if set (e.g. 5.0), define "structurally similar" as
            the top ``similar_percentile``% most-similar off-diagonal pairs
            instead of the absolute ``sim_threshold``. On standardised high-D
            embeddings the pairwise Euclidean distance is large so sim = 1/(1+d)
            never reaches an absolute 0.55 (s12 found 0 cliffs in 13/16
            datasets). A percentile makes the threshold distribution-relative.
            See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            If a list/tuple of candidate percentiles is passed, the expensive
            similarity matrix is built ONCE and every candidate is swept; the
            one whose cliff-compound prevalence is closest to
            ``TARGET_CLIFF_PCT`` (10%) is auto-selected (preferring candidates
            with prevalence > 0). The returned ``info`` then also carries
            ``'percentile_sweep'`` and ``'selected_percentile'``.

    Returns (cliff_indices, scanned_indices, info). scanned_indices holds every
    row the pair scan actually looked at: above max_compounds the scan runs on
    a subsample, and a row outside it was never tested for being a cliff. A
    caller must not score such a row as a non-cliff row.
    """
    from experiments.similarity.similarity_helpers import _pairwise_delta_matrix

    TARGET_CLIFF_PCT = 10.0
    sweep_candidates = None
    if isinstance(similar_percentile, (list, tuple)):
        sweep_candidates = list(similar_percentile)

    n = len(y)
    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        X_sub = X_concat[idx]
        y_sub = y[idx]
    else:
        X_sub = X_concat
        y_sub = y
        idx = np.arange(n)

    D = compute_distance_matrix(X_sub, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)

    if is_clf:
        activity_threshold = 0.5  # indicator is 0 or 1
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y_sub - np.median(y_sub))))
        activity_threshold = activity_k * mad_y

    # "Structurally similar" pairs: absolute cutoff, or distribution-relative
    # (top similar_percentile% most-similar off-diagonal pairs). The sim matrix
    # and delta_y are the expensive part; given any sim_cutoff the cliff stats
    # are cheap boolean ops, so we factor them into a closure that can be swept
    # over several candidate percentiles essentially for free.
    delta_y = _pairwise_delta_matrix(y_sub, is_clf=is_clf)
    _off = sim[~np.eye(len(sim), dtype=bool)]

    def _percentile_cutoff(pct):
        return float(np.percentile(_off, 100.0 - pct))

    def _cliff_stats(sim_cutoff):
        cliff_pair_mask = (sim >= sim_cutoff) & (delta_y > activity_threshold)
        np.fill_diagonal(cliff_pair_mask, False)
        cliff_mask_sub = cliff_pair_mask.any(axis=1)
        n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
        n_cliff_compounds = int(cliff_mask_sub.sum())
        pct_cliff = float(100.0 * n_cliff_compounds / len(y_sub))
        return cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff

    percentile_sweep = None
    selected_percentile = None
    if sweep_candidates is not None:
        # Build the expensive matrices once (already done), then sweep cheaply.
        percentile_sweep = {}
        for cand in sweep_candidates:
            _, _, _, pct_cliff = _cliff_stats(_percentile_cutoff(cand))
            percentile_sweep[cand] = pct_cliff
        # Auto-select: prevalence closest to the 10% target, preferring
        # candidates with prevalence > 0 (only fall back to including 0%
        # candidates if every candidate is 0%).
        positive = [c for c in sweep_candidates if percentile_sweep[c] > 0]
        pool = positive if positive else list(sweep_candidates)
        selected_percentile = float(
            min(pool, key=lambda c: abs(percentile_sweep[c] - TARGET_CLIFF_PCT)))
        sim_cutoff = _percentile_cutoff(selected_percentile)
    elif similar_percentile is not None:
        sim_cutoff = _percentile_cutoff(similar_percentile)
    else:
        sim_cutoff = sim_threshold

    cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff_compounds = \
        _cliff_stats(sim_cutoff)

    # Map back to full indices
    cliff_indices = set(idx[cliff_mask_sub].tolist())
    # Every row the scan covered. Rows outside it were never tested.
    scanned_indices = set(int(i) for i in idx.tolist())

    info = {
        'distance_metric': str(get_distance_metric()),
        'sim_threshold': sim_threshold,
        'similar_percentile': similar_percentile,
        'sim_cutoff': float(sim_cutoff),
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': pct_cliff_compounds,
        'n_analyzed': len(y_sub),
        'n_unanalyzed': max(0, n - max_compounds),
        'max_compounds_cap': max_compounds,
        # The coefficient the distance behind sim = 1/(1+d) was taken with,
        # read from the process-wide setting rather than asserted. It said
        # 'euclidean' on every file written since the default moved to
        # manhattan on 2026-09-04.
        'similarity_type': str(get_distance_metric()),
    }
    if percentile_sweep is not None:
        info['percentile_sweep'] = percentile_sweep
        info['selected_percentile'] = selected_percentile

    return cliff_indices, scanned_indices, info


def identify_cliff_pairs_euclidean(*args, **kwargs):
    """Two-value wrapper kept for callers written before the scanned set existed.

    Returns (cliff_indices, info). The name is historical: the distance has not
    been Euclidean since 2026-09-04 — info['similarity_type'] says which
    coefficient was used. New callers want identify_cliff_pairs, whose third
    value is the set of rows the scan covered.
    """
    cliff_indices, _scanned, info = identify_cliff_pairs(*args, **kwargs)
    return cliff_indices, info


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S12: Cross-domain activity cliff analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S12: CROSS-DOMAIN ACTIVITY CLIFFS")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    augmentations = data.get('augmentations')
    domain = data.get('domain', 'molecular')
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    model_factories = get_model_factories(DOWNSTREAM_MODELS, task_type=task_type,
                                           n_samples=len(labels))
    print(f"  Models: {list(model_factories.keys())}")

    # 2026-09-20: WHICH MOLECULES ARE CLIFFS. This file and s04 used to decide
    # that two different ways. s04 read ECFP4 Tanimoto at or above 0.55 and an
    # activity gap of 2 x the median absolute deviation of the labels. This file
    # read the top half per cent of pairwise similarity on the standardised
    # concatenation of every representation, with the same activity gap. On
    # herg_ki s04 called 451 of 1412 molecules cliffs and this file called 295
    # of 1412 — same dataset, same molecules, two answers.
    #
    # There was a second disagreement inside this file alone. The percentile was
    # auto-selected from a sweep of 0.5, 1, 2, 5 and 10 per cent, picking the
    # one whose share of cliff molecules came nearest a stated 10 per cent
    # target. On all three of caco2, herg_ki and adme_mdr1 even the smallest
    # candidate overshot — 44.9, 20.9 and 30.7 per cent of molecules at 0.5 per
    # cent of pairs — so the sweep always chose its own floor and the 10 per
    # cent target was never reached on any of them.
    #
    # Both files now call similarity_helpers.calibrated_cliff_rule. On a dataset
    # with SMILES the structural side is the same ECFP4 Tanimoto s04 builds, so
    # the two files flag the same molecules; on a dataset without SMILES it is
    # the concatenation similarity at a FIXED percentile, and the activity gap
    # is solved to the target share either way.
    activity_k = kwargs.get('activity_k')       # None = calibrate
    cliff_target_pct = kwargs.get('cliff_target_pct', CLIFF_TARGET_PCT)
    # A dataset outside the molecular domain has no ECFP4, so it takes the
    # concatenation branch. Passing force_concat_similarity=True takes it on a
    # molecular dataset too, which is what makes the two branches comparable on
    # one dataset.
    force_concat = bool(kwargs.get('force_concat_similarity', False))

    X_concat = np.hstack([reps[k] for k in sorted(reps.keys())])
    print(f"  Domain: {domain}, {len(labels)} molecules, "
          f"{X_concat.shape[1]} features in the concatenation")

    # NOTE: cliff identity uses full-dataset labels — accepted leakage for cliff
    # partitioning, per N01 precedent.
    cliff_sim, cliff_cutoff, cliff_note, cliff_kind = \
        cliff_structural_similarity(
            data, reps, is_molecular=(domain == 'molecular' and not force_concat))
    print(f"  Cliff structural side: {cliff_kind} -- {cliff_note}")
    cliff_mask, cliff_info = calibrated_cliff_rule(
        cliff_sim, labels,
        structural_cutoff=cliff_cutoff,
        structural_source=cliff_note,
        target_pct=cliff_target_pct,
        is_clf=is_clf,
        activity_k=activity_k,
    )
    cliff_info = {k: v for k, v in cliff_info.items() if k != 'cliff_pair_counts'}
    cliff_info['shared_with_s04'] = (cliff_kind == 'smiles_ecfp4_tanimoto')
    cliff_indices = set(np.where(cliff_mask)[0].tolist())
    # This rule scans every pair of every row, so nothing is left untested. The
    # 5000-row subsample the old rule took above that size is gone with it.
    scanned_indices = set(range(len(labels)))
    cliff_info['n_analyzed'] = len(labels)
    cliff_info['n_unanalyzed'] = 0
    print(f"  Cliff molecules: {cliff_info['n_cliff_compounds']} of "
          f"{len(labels)} molecules "
          f"({cliff_info['pct_cliff_compounds']:.1f} per cent); "
          f"{cliff_info['n_cliff_pairs']} cliff pairs")
    print(f"  Under the fixed 2 x median-absolute-deviation gap this file used "
          f"before: {cliff_info['uncalibrated_n_cliff_compounds']} of "
          f"{len(labels)} molecules "
          f"({cliff_info['uncalibrated_pct_cliff_compounds']:.1f} per cent)")

    if not cliff_indices:
        # No activity cliffs is a LEGITIMATE methodological outcome, not a
        # failure: when the representation captures the structure-activity
        # relationship well, the structurally-most-similar pairs share
        # activity, so no cliff exists even after widening the structural
        # percentile to 30%. This is common for antibody datasets on
        # foundation-model embeddings (abagym/abcov/flab_polyreact in the
        # production wave). Returning a key literally named 'error' made the
        # wave audit ("if 'error' in result") count s12 as a failed
        # experiment, leaving similarity stuck at 11/12. Signal the outcome
        # with the canonical 'skipped'/'reason' convention (cf. s04) — no
        # 'error' key — so the experiment is counted complete.
        #
        # 2026-09-20: the shared rule solves the activity gap down towards zero
        # to reach its target share, so it reaches this only when the structural
        # side alone flags no pair at all.
        print("  No activity cliffs: no pair of molecules is structurally "
              "similar enough under this rule at any activity gap. Reporting "
              "as a valid no-cliff result, not an error.")
        return {
            'skipped': True,
            'reason': 'no_cliffs_found',
            'domain': domain,
            'metric': metric_name,
            'lower_is_better': lower_is_better,
            'cliff_info': convert_numpy(cliff_info),
        }

    # CV evaluation: per-rep baselines + hybrid on cliff vs non-cliff
    sample_X = next(iter(reps.values()))

    # ── Per-fold hybrid recipes, READ FROM THE POOL ──
    # 2026-09-12, owner's decision: s12 stops calling get_fold_best_search and
    # reads the pool instead, chosen per fold from the other folds' scores. No
    # search runs in this file any more. For each held-out fold the pool ranks
    # its stored configurations by their mean over the OTHER folds and hands
    # back the top few; that fold's own score never enters the ranking.
    #
    # What this holds out changed, and the results record it. The searches in
    # the pool ran over every row, so a recipe is chosen without fold i's
    # SCORES, not by a search that never saw fold i's rows. The previous code
    # searched each fold's training rows, which held out the rows themselves.
    budget = kwargs.get('budget', 100)
    top3 = None  # ensure defined even if the pool read exits early
    folds = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                  _FOLD_SEARCH_SEED, task_type)

    # How many held-out rows per fold the pair scan never looked at. Those rows
    # are scored in neither arm; before 2026-09-12 every one of them counted as
    # a non-cliff row. Zero on any dataset at or below the 5000-row cap.
    unscanned_heldout_by_fold = {
        str(fold_i): int(sum(1 for vi in val_idx if vi not in scanned_indices))
        for fold_i, (_tr, val_idx) in enumerate(folds)}
    _n_unscanned_total = sum(unscanned_heldout_by_fold.values())
    if _n_unscanned_total:
        print(f"  {_n_unscanned_total} held-out rows across {len(folds)} folds "
              f"were never covered by the pair scan; they are scored as "
              f"neither cliff nor non-cliff.")

    _pool_cache = {}
    pool_warnings = {}
    pool_n_folds_seen = set()
    print(f"  Hybrid recipes: read from the pool, subset={_POOL_SUBSET} "
          f"(regular and augmented only, no similarity-built hybrids), "
          f"budget={budget} features, top {_N_POOL_HYBRIDS} per fold")

    def _fold_pool(mname, fold_i):
        """The pool's ranking for one model on one held-out fold, or None."""
        ck = (mname, fold_i)
        if ck in _pool_cache:
            return _pool_cache[ck]
        try:
            pool = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(folds),
                downstream_model=mname, budget=budget,
                dataset_name=dataset_name,
                augmentations=data.get('augmentations'),
                data=data, metric=metric_name, task_type=task_type,
                domain=domain,
                pool_subset=_POOL_SUBSET,
                n_hybrids=_N_POOL_HYBRIDS)
        except Exception as e:
            print(f"  [warn] pool read ({mname}, fold {fold_i}): "
                  f"{type(e).__name__}: {str(e)[:100]}")
            pool = None
        if pool is not None:
            for w in pool.get('warnings') or []:
                print(f"  [pool {mname} fold {fold_i}] {w}")
            pool_warnings[f"{mname}|fold{fold_i}"] = list(
                pool.get('warnings') or [])
            if pool.get('n_folds_in_pool') is not None:
                pool_n_folds_seen.add(int(pool['n_folds_in_pool']))
        _pool_cache[ck] = pool
        return pool

    # Fold 0's pool, reported for the base-representation ranking only. The
    # scores are each representation's mean over folds 1..n-1 in the pool
    # search, in the dataset's own metric — not the cliff-row metric below.
    _pool0 = _fold_pool('rf', 0) if 'rf' in model_factories else None
    if _pool0 is not None:
        _ranked0 = ((_pool0.get('by_objective') or {}).get('predictive')
                    or {}).get('base_reps_ranked') or []
        top3 = [(b['rep'], float(b['score_other_folds'])) for b in _ranked0[:3]]
        if top3:
            print(f"  Top-3 base reps for fold 0 (by RF, pool mean over the "
                  f"other folds): "
                  f"{', '.join(f'{n}={s:.4f}' for n, s in top3)}")

    # Per-rep baselines per model
    per_rep_results = {}

    # Each (representation, model) baseline and each model's hybrid block are
    # written to s12_cross_domain_cliffs/<dataset>.partial.jsonl when they
    # finish and read back on the next attempt (2026-09-25; _checkpoint.py).
    # The folds, the cliff rows and the pool reads are recomputed identically
    # and every model is seeded, so a unit read back holds what a fresh
    # computation gives. A unit in which a fit or a hybrid build raised (its
    # fold is NaN) is not stored, so it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's12_cross_domain_cliffs', dataset_name),
        run_identity('s12_cross_domain_cliffs', dataset_name, data, config,
                     kwargs=kwargs, models=sorted(model_factories),
                     budget=budget, n_pool_hybrids=_N_POOL_HYBRIDS))
    _hyb_threw = []

    for rep_name, X_rep in reps.items():
        rep_model_results = {}
        for mname, mfactory in model_factories.items():
            if _ck.done(rep=rep_name, model=mname):
                rep_model_results[mname] = _ck.get(rep=rep_name, model=mname)
                continue
            _threw = False
            # Fold-aligned per-fold cliff scores: append NaN for any fold that has
            # no usable cliff subset (or a failed fit) so fold f means the same fold
            # across every rep + the hybrid — required for held-out selection in
            # fold_reselection (the scalar means below are NaN-tolerant, so identical
            # to the old skip-based values).
            cliff_fold_scores = []
            noncliff_fold_scores = []
            for train_idx, val_idx in folds:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_rep[train_idx])
                X_va = scaler.transform(X_rep[val_idx])

                try:
                    m = mfactory()
                    m.fit(X_tr, labels[train_idx])
                    y_pred = m.predict(X_va)
                except Exception as e:
                    print(f"  [warn] per-rep fit/predict ({rep_name}/{mname}): {type(e).__name__}: {str(e)[:100]}")
                    _threw = True
                    cliff_fold_scores.append(float('nan'))
                    noncliff_fold_scores.append(float('nan'))
                    continue

                val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
                # Only rows the pair scan covered can be called non-cliff. A
                # row outside the scan was never tested, so it lands in
                # neither arm instead of padding the non-cliff one.
                val_noncliff = [i for i, vi in enumerate(val_idx)
                                if vi in scanned_indices and vi not in cliff_indices]

                cliff_fold_scores.append(
                    float(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
                    if len(val_cliff) >= 2 else float('nan'))
                noncliff_fold_scores.append(
                    float(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))
                    if len(val_noncliff) >= 2 else float('nan'))

            rep_model_results[mname] = {
                'cliff_score': _finite_mean(cliff_fold_scores),
                'noncliff_score': _finite_mean(noncliff_fold_scores),
                # NEW (additive): per-fold scores feed honest held-out selection.
                'cliff_fold_scores': cliff_fold_scores,
                'noncliff_fold_scores': noncliff_fold_scores,
            }
            if not _threw:
                _ck.put(rep_model_results[mname], rep=rep_name, model=mname)
        per_rep_results[rep_name] = rep_model_results

    def _score_hybrid_fold(mname, mfactory, blueprint, aug_cond,
                           train_idx, val_idx):
        """(cliff score, non-cliff score) for one recipe on one fold; NaN on failure."""
        _use_aug = augmentations if aug_cond != 'none' else None
        train_reps_h = {k: v[train_idx] for k, v in reps.items()}
        val_reps_h = {k: v[val_idx] for k, v in reps.items()}
        aug_tr = {k: v[train_idx] for k, v in _use_aug.items()} if _use_aug else None
        aug_va = {k: v[val_idx] for k, v in _use_aug.items()} if _use_aug else None

        try:
            X_tr_h, X_va_h, fi = build_fold_hybrid(
                train_reps_h, val_reps_h, labels[train_idx], blueprint,
                augmentations_train=aug_tr, augmentations_val=aug_va,
                groups=(groups[train_idx] if groups is not None else None),
                model=mfactory(),
                metric_fn=metric_fn,
                lower_is_better=lower_is_better)
        except Exception as e:
            print(f"  [warn] hybrid build ({mname}): {type(e).__name__}: {str(e)[:100]}")
            _hyb_threw.append(mname)
            return float('nan'), float('nan')

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_h)
        X_va_s = scaler.transform(X_va_h)

        try:
            m = mfactory()
            m.fit(X_tr_s, labels[train_idx])
            y_pred = m.predict(X_va_s)
        except Exception as e:
            print(f"  [warn] hybrid fit/predict ({mname}): {type(e).__name__}: {str(e)[:100]}")
            _hyb_threw.append(mname)
            return float('nan'), float('nan')

        val_cliff = [i for i, vi in enumerate(val_idx) if vi in cliff_indices]
        # Same rule as the per-rep arm: an unscanned row is in neither.
        val_noncliff = [i for i, vi in enumerate(val_idx)
                        if vi in scanned_indices and vi not in cliff_indices]
        c = (float(metric_fn(labels[val_idx[val_cliff]], y_pred[val_cliff]))
             if len(val_cliff) >= 2 else float('nan'))
        nc = (float(metric_fn(labels[val_idx[val_noncliff]], y_pred[val_noncliff]))
              if len(val_noncliff) >= 2 else float('nan'))
        return c, nc

    # Hybrid evaluation per model.
    # 2026-09-12, owner's decision: the recipes come from the pool, ranked per
    # fold on the other folds' scores. The pool hands back _N_POOL_HYBRIDS of
    # them for each fold and every one is built and scored — none is dropped.
    # Rank 1 stays the headline arm under results['hybrid'], which is the same
    # shape the per-fold search wrote before; ranks 2 and 3 are stored beside it
    # under results['pool_hybrid_ranks'].
    hybrid_results_by_model = {}
    pool_rank_results_by_model = {}
    pool_selection_by_model = {}
    fold0_pruned_any = None
    for mname, mfactory in model_factories.items():
        if _ck.done(hybrid_model=mname):
            _saved = _ck.get(hybrid_model=mname)
            pool_selection_by_model[mname] = _saved['selection']
            pool_rank_results_by_model[mname] = _saved['ranks']
            hybrid_results_by_model[mname] = _saved['hybrid']
            _p = _saved['fold0_pruned']
            if _p is not None:
                fold0_pruned_any = (_p if fold0_pruned_any is None
                                    else (fold0_pruned_any or _p))
            continue
        _n_threw_before = len(_hyb_threw)
        _fold0_pruned_this = None
        # Fold-aligned per-fold hybrid scores (NaN placeholder on skipped/failed
        # folds) so fold_reselection can index the same fold as the per-rep baselines.
        rank_cliff = [[] for _ in range(_N_POOL_HYBRIDS)]
        rank_noncliff = [[] for _ in range(_N_POOL_HYBRIDS)]
        rank_recipes = [{} for _ in range(_N_POOL_HYBRIDS)]
        sel_by_fold = {}
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            pool = _fold_pool(mname, fold_i)
            hybrids = (pool or {}).get('hybrids') or []
            if pool is not None:
                _p = pool.get('fold0_pruned')
                _p = True if _p is None else bool(_p)
                fold0_pruned_any = (_p if fold0_pruned_any is None
                                    else (fold0_pruned_any or _p))
                _fold0_pruned_this = (_p if _fold0_pruned_this is None
                                      else (_fold0_pruned_this or _p))
                _base = pool.get('base_rep') or {}
                sel_by_fold[fold_i] = {
                    'pool_subset': pool.get('pool_subset'),
                    'objectives': pool.get('objectives'),
                    'n_folds_in_pool': pool.get('n_folds_in_pool'),
                    'fold0_pruned': pool.get('fold0_pruned'),
                    'selection_scope': pool.get('selection_scope'),
                    'base_rep': _base.get('rep'),
                    'base_rep_score_other_folds': _base.get('score_other_folds'),
                    'hybrids': [
                        {'rank': h.get('rank'), 'name': h.get('name'),
                         'objective': h.get('objective'),
                         'aug_condition': h.get('aug_condition'),
                         'metric_name': h.get('metric_name'),
                         'score_other_folds': h.get('score_other_folds'),
                         'n_other_folds': h.get('n_other_folds')}
                        for h in hybrids],
                }
            if not hybrids:
                print(f"  [warn] the pool held no hybrid configuration for "
                      f"{mname} on fold {fold_i}")
            for r in range(_N_POOL_HYBRIDS):
                if r >= len(hybrids):
                    rank_cliff[r].append(float('nan'))
                    rank_noncliff[r].append(float('nan'))
                    continue
                h = hybrids[r]
                bp = h.get('blueprint')
                if bp is None:
                    rank_cliff[r].append(float('nan'))
                    rank_noncliff[r].append(float('nan'))
                    continue
                aug_cond = h.get('aug_condition')
                rank_recipes[r][fold_i] = {
                    'allocation': getattr(bp, 'allocation_method', None),
                    'importance': getattr(bp, 'importance_method', None),
                    'evaluator': getattr(bp, 'internal_evaluator', None),
                    'budget': getattr(bp, 'budget', None),
                    'augmentation_strategy': getattr(
                        bp, 'augmentation_strategy', aug_cond),
                    # Where this recipe came from, so a stored result says which
                    # search family and which pool member it was read out of.
                    'pool_name': h.get('name'),
                    'pool_objective': h.get('objective'),
                    'pool_tier': h.get('tier'),
                    'pool_score_other_folds': h.get('score_other_folds'),
                    'pool_metric_name': h.get('metric_name'),
                }
                c, nc = _score_hybrid_fold(mname, mfactory, bp, aug_cond,
                                           train_idx, val_idx)
                rank_cliff[r].append(c)
                rank_noncliff[r].append(nc)

        pool_selection_by_model[mname] = {int(fi): s
                                          for fi, s in sorted(sel_by_fold.items())}
        pool_rank_results_by_model[mname] = {
            f'pool_rank{r + 1}': {
                'cliff_score': _finite_mean(rank_cliff[r]),
                'noncliff_score': _finite_mean(rank_noncliff[r]),
                'cliff_fold_scores': rank_cliff[r],
                'noncliff_fold_scores': rank_noncliff[r],
                'fold_recipes': {int(fi): dict(rec)
                                 for fi, rec in sorted(rank_recipes[r].items())},
            }
            for r in range(_N_POOL_HYBRIDS)
        }

        hybrid_results_by_model[mname] = {
            'cliff_score': _finite_mean(rank_cliff[0]),
            'noncliff_score': _finite_mean(rank_noncliff[0]),
            # NEW (additive): per-fold scores feed honest held-out selection.
            'cliff_fold_scores': rank_cliff[0],
            'noncliff_fold_scores': rank_noncliff[0],
            # The recipe the pool ranked first for each fold, on the other
            # folds' scores. They can differ from fold to fold; one recipe for
            # the model is not a fact about this run.
            'fold_recipes': {int(fi): dict(rec)
                             for fi, rec in sorted(rank_recipes[0].items())},
            'pool_rank': 1,
        }
        if not rank_recipes[0]:
            hybrid_results_by_model[mname]['skipped'] = 'no_pool_recipe_on_any_fold'
        if len(_hyb_threw) == _n_threw_before:
            _ck.put({'selection': pool_selection_by_model[mname],
                     'ranks': pool_rank_results_by_model[mname],
                     'hybrid': hybrid_results_by_model[mname],
                     'fold0_pruned': _fold0_pruned_this},
                    hybrid_model=mname)

    results = {
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'cliff_info': convert_numpy(cliff_info),
        # Held-out rows per fold the pair scan never covered. They are in
        # neither the cliff arm nor the non-cliff one.
        'unscanned_heldout_by_fold': unscanned_heldout_by_fold,
        'n_unscanned_heldout': _n_unscanned_total,
        'per_rep': convert_numpy(per_rep_results),
        'hybrid': convert_numpy(hybrid_results_by_model),
        # 2026-09-12, owner's decision: the recipes are read from the pool, so
        # every configuration the pool returned for a fold is scored and kept.
        # 'hybrid' above is pool_rank1.
        'pool_hybrid_ranks': convert_numpy(pool_rank_results_by_model),
        'pool_selection': convert_numpy(pool_selection_by_model),
        'pool_read': {
            'pool_subset': _POOL_SUBSET,
            'n_hybrids_per_fold': _N_POOL_HYBRIDS,
            'budget': budget,
            'searches_run_by_this_file': 0,
            'note': ('recipes read from the stored pool searches and ranked per '
                     'held-out fold on the other folds mean; the pool search '
                     'itself ran over every row, so what is held out is the '
                     "fold's scores, not its rows"),
            'warnings': pool_warnings,
        },
    }
    if top3:
        # Now the pool's mean over the other folds for fold 0, in the dataset's
        # own metric — not the fold-0-training-rows search baselines it was.
        results['top3_base_reps'] = [(n, s) for n, s in top3]
        results['top3_base_reps_source'] = (
            'pool, rf, fold 0, mean over the other folds')

    # Honest-comparison block (CONTRACT json_convention). Built on the CLIFF score
    # (the headline cliff comparison the same-fold argmax below optimises). Fold
    # scores are fold-aligned NaN-padded lists so fold_reselection can re-pick the
    # winning rep / hybrid per HELD-OUT fold instead of the same-fold argmax. This
    # experiment computes its own segmented CV baselines (no SearchResult), so the
    # two per-model dicts are assembled by hand rather than via
    # build_fold_reselection_input. Only models actually evaluated appear.
    #
    # 2026-09-12, owner's decision (s12 reads the pool): the hybrid candidates
    # are now one per pool rank — 'pool_rank1' is "take the configuration the
    # pool ranks first for this fold", 'pool_rank2' the second, and so on. Each
    # is scored on every fold, so all three cover the same folds and the
    # leave-one-fold-out selection below has no gaps to fall through.
    candidates_by_model = {}
    baselines_by_model = {}
    for mname in model_factories:
        cands = []
        for arm_name, arm in (pool_rank_results_by_model.get(mname) or {}).items():
            fs = arm.get('cliff_fold_scores')
            if fs and any(s is not None and not np.isnan(s) for s in fs):
                cands.append({'name': arm_name, 'fold_scores': fs})
        if cands:
            candidates_by_model[mname] = cands
        base = {}
        for rep_name, rep_model_results in per_rep_results.items():
            rf = rep_model_results.get(mname, {}).get('cliff_fold_scores')
            if rf is not None:
                base[rep_name] = rf
        if base:
            baselines_by_model[mname] = base
    # 2026-09-12: this block used to set exclude_folds to [] and say "no fold
    # filtered them", which was true while this file searched each fold itself.
    # The recipe list now comes from the stored pool searches, and a search that
    # pruned mid-flight filtered its own candidate list on fold 0's scores. The
    # pool reports that as 'fold0_pruned'; when it is set, fold 0 drops from
    # BOTH arms, which is what hr_exclude_folds asks for.
    #
    # Fold 0 is only droppable when this run's folds ARE the pool's folds: same
    # count, and nested_cv_outer_folds at seed 42 on the same groups. With
    # --mode intermediary this file makes one 80/20 split instead of 5 folds,
    # so its fold 0 is not the pool's fold 0 and there is nothing to drop.
    _fold0_pruned = True if fold0_pruned_any is None else bool(fold0_pruned_any)
    _folds_match_pool = (pool_n_folds_seen == {len(folds)})
    results['fold_reselection'] = {
        'lower_is_better': lower_is_better,
        'exclude_folds': ([0] if (_fold0_pruned and _folds_match_pool) else []),
        'fold0_pruned': _fold0_pruned,
        'fold_numbering_matches_pool': bool(_folds_match_pool),
        'n_folds_in_pool': sorted(pool_n_folds_seen),
        # The recipe was chosen without fold i's SCORES. The pool search ran
        # over every row, so fold i's rows are not held out of the recipe.
        'recipe_selection': 'other_folds_of_pool_search',
        'candidates_by_model': convert_numpy(candidates_by_model),
        'baselines_by_model': convert_numpy(baselines_by_model),
    }

    # Sign convention, recorded in the output so a stored result is readable
    # without knowing which key it came from. 'hybrid_cliff_delta' is the
    # ORIGINAL key: a raw subtraction, NOT direction-corrected, so on a
    # lower-is-better metric its positive values mean the hybrid's cliff error
    # rose. 'hybrid_cliff_delta_directed' is the same pair of scores with the
    # metric direction applied, matching 'hybrid_cliff_delta_reselected'. Both are
    # kept so results written under either convention still load.
    results['delta_conventions'] = {
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'hybrid_cliff_delta': {
            'formula': 'hybrid_cliff_score - best_rep_cliff_score',
            'direction_corrected': False,
            'positive_means': ('hybrid cliff score is numerically higher '
                               '(worse when lower_is_better)'),
            'selection': 'same-fold argmax over reps (selection-optimistic)',
            # 2026-09-12: the hybrid side is the configuration the pool ranked
            # first for each fold (results['hybrid'] = pool_rank1).
            'hybrid_arm': 'pool_rank1',
        },
        'hybrid_cliff_delta_directed': {
            'formula': ('best_rep_cliff_score - hybrid_cliff_score '
                        'if lower_is_better else '
                        'hybrid_cliff_score - best_rep_cliff_score'),
            'direction_corrected': True,
            'positive_means': 'hybrid did better',
            'selection': 'same-fold argmax over reps (selection-optimistic)',
            'hybrid_arm': 'pool_rank1',
        },
        'hybrid_cliff_delta_reselected': {
            'formula': 'see fold_reselection in _shared.py',
            'direction_corrected': True,
            'positive_means': 'hybrid did better',
            # 2026-09-12, owner's decision (s12 reads the pool): rewritten,
            # because the old sentence said the recipe was searched inside each
            # fold on that fold's training rows and that is no longer what
            # happens.
            'selection': ('leave-one-fold-out re-selection (held out) on both '
                          'arms; the hybrid recipes are read from the stored '
                          'pool searches and ranked for each fold on the other '
                          "folds' scores, so the recipe never sees the fold's "
                          'score — but the pool search ran over every row, so '
                          'it did see the rows'),
        },
    }

    # Summary per model
    print(f"\n  Hybrid cliff {metric_name} per model "
          f"(hybrid arm = the recipe the pool ranked first for each fold; "
          f"lower_is_better={bool(lower_is_better)}; 'directed' and the honest "
          f"block below are direction-corrected so + means the hybrid did "
          f"better; 'raw' is the plain subtraction hybrid - best rep; both are "
          f"same-fold argmax over reps, i.e. selection-optimistic):")
    results['hybrid_cliff_delta'] = {}
    results['hybrid_cliff_delta_directed'] = {}
    for mname in model_factories:
        h_cliff = hybrid_results_by_model.get(mname, {}).get('cliff_score', float('nan'))
        # Best single rep cliff score for this model (same-fold argmax — optimistic;
        # retained for back-compat, superseded by fold_reselection above).
        rep_cliffs = [r.get(mname, {}).get('cliff_score', float('nan'))
                      for r in per_rep_results.values()]
        rep_cliffs = [s for s in rep_cliffs if not np.isnan(s)]
        best_rep_cliff = max(rep_cliffs, default=float('nan')) if not lower_is_better \
            else min(rep_cliffs, default=float('nan'))
        if not np.isnan(h_cliff) and not np.isnan(best_rep_cliff):
            delta = h_cliff - best_rep_cliff
            delta_directed = -delta if lower_is_better else delta
            results['hybrid_cliff_delta'][mname] = float(delta)
            results['hybrid_cliff_delta_directed'][mname] = float(delta_directed)
            print(f"    {mname}: {h_cliff:.4f} (best rep: {best_rep_cliff:.4f}, "
                  f"directed delta: {delta_directed:+.4f}, "
                  f"raw delta: {delta:+.4f})")

    # Honest held-out cliff delta (replaces the same-fold argmax): pick the winning
    # rep / hybrid on the OTHER folds, score it on the held-out fold, average.
    honest = fold_reselection(
        results['fold_reselection']['candidates_by_model'],
        results['fold_reselection']['baselines_by_model'],
        lower_is_better,
        exclude_folds=hr_exclude_folds(results['fold_reselection']),
        # 2026-09-12: copied onto every model entry, so a stored result says
        # what chose the recipe without opening this file.
        recipe_selection=results['fold_reselection']['recipe_selection'])
    results['hybrid_cliff_delta_reselected'] = {
        m: convert_numpy(r) for m, r in honest.items()}
    print(f"\n  Honest held-out cliff {metric_name} per model "
          f"(direction-corrected, + means the hybrid did better; rep chosen on "
          f"the other folds, i.e. held out):")
    for mname, r in honest.items():
        if not np.isnan(r.get('delta', float('nan'))):
            print(f"    {mname}: hybrid={r['hybrid_reselected']:.4f} "
                  f"best_single={r['best_single_reselected']:.4f} "
                  f"delta={r['delta']:+.4f}"
                  f"{' [single_fold]' if r.get('single_fold') else ''}")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S12: Cross-Domain Activity Cliffs')
    # 2026-09-20: --sim-threshold and --similar-percentile are gone with the
    # rule they configured. The structural side is fixed (ECFP4 Tanimoto 0.55
    # where there are SMILES, otherwise a fixed percentile of the concatenation
    # similarity) and the activity gap is solved to --cliff-target-pct.
    parser.add_argument('--cliff-target-pct', type=float,
                        default=CLIFF_TARGET_PCT,
                        help='the share of this dataset\'s molecules the cliff '
                             'stratum is calibrated to, in per cent')
    parser.add_argument('--activity-k', type=float, default=None,
                        help='skip the calibration and use an activity gap of '
                             'this many times the median absolute deviation of '
                             'the labels, which is what this file did before '
                             '2026-09-20 with 2.0')
    parser.add_argument('--force-concat-similarity', action='store_true',
                        help='take the concatenation-similarity branch even on '
                             'a molecular dataset, instead of the ECFP4 '
                             'Tanimoto s04 also uses')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    _run_kwargs = {'cliff_target_pct': args.cliff_target_pct,
                   'activity_k': args.activity_k,
                   'force_concat_similarity': args.force_concat_similarity}
    results = run(data, config, args.output, args.dataset, **_run_kwargs)
    save_experiment_results(results, args.output, 's12_cross_domain_cliffs', args.dataset)
