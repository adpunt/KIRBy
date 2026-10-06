#!/usr/bin/env python3
"""S02: does distance in the hybrid follow activity better than distance in the
best single representation?

Paper: Similarity
Dependencies: None.

The statistic is the Spearman correlation between the distance of two
molecules and the difference in their activity (on a classification dataset,
whether the two are in different classes), over sampled pairs of held-out
molecules. Higher is better. Beside it, per pair of a distance space and a
distance: the mean SALI, and the share of each molecule's nearest held-out
neighbours whose activity differs by more than the median pair difference
(on classification: that are in another class).

ALIGNED 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2, chat 4). What every
outer fold does now:
  - folds: sim_outer_folds, count from config['n_splits'] (handoff 2.4 step
    14); a count of 1 is one held-out split.
  - selection: fold_similarity_selection(..., neighbour_spaces=True) on the
    fold's TRAINING rows. Distance spaces scored:
      single representations: knn's 2 best single representations from the
        per-fold distance search, and each decided model's 2 best base
        representations from the per-fold general search (a representation
        named by both is scored once and lists both sources);
      hybrids: each decided model's 2 general hybrids, knn's top hybrid of the
        per-fold distance search, and the top hybrid of the per-fold
        similarity-objective search.
    Every hybrid is rebuilt on the fold's training rows with the fold search's
    own representation scores (build_fold_entry; dirty-run bug 1).
  - budgets: the dataset's decided budgets (decided_grid), swept per fold by
    re-reading the same per-fold searches at each budget:
    fold_standard_selection(budget=b) for the decided models' hybrids and for
    knn's distance hybrid, top_hybrids_across_models(budget=b) on the
    similarity search. No extra search runs for the sweep.
  - distances: every distance of the dataset's kNN group (knn_axes); neighbour
    counts 5, 10, 20; counting schemes rank and inverse_square; the scaffold
    group is not excluded; a molecule is never its own neighbour (excluded by
    index).
  - scaling: FeatureScaler fitted on the fold's training rows (0/1 features
    unscaled, the rest standardised), held-out values cut to the training
    range (scale_fold; dirty-run bug 3, handoff 2.3 item 17).
  - no importance weighting inside the distance. DECIDED 2026-10-06
    (Adelaide delegated): neighbour lists are weighted by importance only
    where weighting is the experiment's question, and s02's question is hybrid
    against single representation. Until 2026-10-06 s02 also scored each space
    weighted by train-row lightgbm_gain (singles) and by each of
    `weight_methods` (hybrids). Every row still carries 'weighting':
    'unweighted' so s09 reads the same keys.
  - no concatenation-of-all-representations arm (removed 2026-10-05; kept
    removed, DECIDED 2026-10-06: baselines are individual representations).
  - no whole-dataset number, no p-value, no fold re-selection: every choice
    is made on the fold's training rows, and every reported number is
    held out.

History kept from before 2026-10-05: the file swept all eight budgets over
whole-dataset searches (get_rule_hybrids, get_best_search_for_downstream),
picked hybrids for knn only, and scored the whole dataset beside a per-fold
block. All of that is gone; see the handoff for the reasons (R1-R6).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments.similarity.similarity_helpers import _CODE_FAULTS

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results,
    get_domain_metric, get_model_factories,
)
from experiments.similarity.similarity_helpers import (
    similarity_activity_correlation, compute_sali,
    run_antibody_cliff_analysis,
    sim_outer_folds, fold_spread_note, knn_axes, scale_fold,
    fold_similarity_selection, build_fold_entry, entry_record,
    SIM_FOLD_SEED,
)

from kirby.utils.noise_mitigation_methods import compute_distance_matrix

# What s09 compares before it reads this file (similarity/_summary_keys.py).
# s02's own value since 2026-10-05: per-fold selection on training rows, so a
# file written under the old whole-dataset rule never matches it.
from experiments.similarity._summary_keys import SELECTION_RULE as _SELECTION_RULE

# How many held-out pairs the median activity difference is taken over, for
# the neighbour-violation share on regression. Same as violation_rate.
_VIOLATION_MEDIAN_PAIRS = 50000


# =============================================================================
# ONE DISTANCE SPACE, SCORED ON ONE FOLD'S HELD-OUT MOLECULES
# =============================================================================

def _held_out_distance(X_scaled, metric):
    """Distances among the held-out rows under one decided distance, every
    feature weighing the same.

    X_scaled is already scaled on the training rows (scale_fold), so
    compute_distance_matrix is told not to standardise again. A pair
    the distance leaves undefined (cosine on a row with no spread) is set to
    +inf, so it ranks as the farthest pair and is nobody's neighbour.
    """
    D = np.asarray(compute_distance_matrix(
        X_scaled, metric=metric, standardize=False), dtype=np.float64)
    D[np.isnan(D)] = np.inf
    np.fill_diagonal(D, 0.0)
    return D


def _violation_rows(D, y, *, ks, schemes, is_clf):
    """Share of each held-out molecule's nearest held-out neighbours whose
    activity differs by more than the median pair difference (classification:
    that are in another class), averaged over molecules. One row per
    (neighbour count, counting scheme).

    A molecule is excluded from its own list by index (its diagonal set to
    +inf), never by dropping the first column (dirty-run bug 7). No scaffold
    group is excluded (kNN record, axis 6). Each neighbour counts by the
    scheme's weight (rank: 1, 1/2, 1/3, ...; inverse_square: 1/d^2), so the
    share is the weighted share; until 2026-10-05 every neighbour counted
    equally and only k = 5 was read.
    """
    from experiments.noise import similarity_space as ss
    y = np.asarray(y)
    n = len(y)
    Dn = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(Dn, np.inf)
    k_max = int(min(max(ks), n - 1))
    if k_max < 1:
        return [{'k': int(k), 'scheme': s, 'violation_share': None,
                 'note': f'{n} held-out molecules leave no neighbour'}
                for k in ks for s in schemes]
    idx = np.argsort(Dn, axis=1, kind='stable')[:, :k_max]
    dist = np.take_along_axis(Dn, idx, axis=1)
    if is_clf:
        viol = (y[idx] != y[:, None])
    else:
        ti, tj = np.triu_indices(n, k=1)
        if len(ti) > _VIOLATION_MEDIAN_PAIRS:
            pick = np.random.RandomState(42).choice(
                len(ti), _VIOLATION_MEDIAN_PAIRS, replace=False)
            ti, tj = ti[pick], tj[pick]
        med = float(np.median(np.abs(y[ti] - y[tj])))
        viol = np.abs(y[idx] - y[:, None]) > med
    rows = []
    for k in ks:
        if k > k_max:
            rows.extend({'k': int(k), 'scheme': s, 'violation_share': None,
                         'note': f'only {k_max} other held-out molecules'}
                        for s in schemes)
            continue
        for s in schemes:
            w = ss.neighbour_weights(dist[:, :k], s)
            tot = w.sum(axis=1)
            share = np.where(
                tot > 0,
                (w * viol[:, :k]).sum(axis=1) / np.where(tot > 0, tot, 1.0),
                np.nan)
            ok = np.isfinite(share)
            rows.append({'k': int(k), 'scheme': s,
                         'violation_share': (float(share[ok].mean())
                                             if ok.any() else None),
                         'n_molecules': int(ok.sum())})
    return rows


def _score_space(X_train, X_test, y_test, *, axes, n_sample, is_clf):
    """Every held-out number for one distance space on one fold.

    Returns one row per distance, each marked 'weighting': 'unweighted';
    its violation rows carry their neighbour count and counting scheme.
    """
    A, B = scale_fold(X_train, X_test)
    del A
    y_test = np.asarray(y_test)
    rows = []
    for metric in axes['metrics']:
        row = {'distance': metric, 'weighting': 'unweighted'}
        try:
            D = _held_out_distance(B, metric)
            rho, _ = similarity_activity_correlation(
                D, y_test, n_sample, is_clf=is_clf)
            sali = compute_sali(D, y_test, is_clf=is_clf)
            row.update({
                'sac_rho': (float(rho) if np.isfinite(rho) else None),
                'sali_mean': sali['mean'],
                'sali_n_identical_pairs': sali['n_identical_pairs'],
                'neighbour_violation': _violation_rows(
                    D, y_test, ks=axes['ks'], schemes=axes['schemes'],
                    is_clf=is_clf),
            })
        except Exception as e:
            row['not_computed'] = f'{type(e).__name__}: {e}'
        rows.append(row)
    return rows


def _rho(rows, distance, weighting):
    for r in rows or []:
        if r.get('distance') == distance and r.get('weighting') == weighting:
            v = r.get('sac_rho')
            return v if isinstance(v, (int, float)) and np.isfinite(v) else None
    return None


def _plain(obj):
    """JSON-plain copy: numbers, strings, lists and dicts only."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()
                if k not in ('blueprint', 'search', 'feature_signature')}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


# =============================================================================
# ONE OUTER FOLD
# =============================================================================

def _score_hybrid(data, entry, *, model, train_idx, test_idx, factories,
                  metric_fn, dataset_lower, axes, n_sample, is_clf):
    """Rebuild one selected hybrid on the fold's training rows and score its
    distance on the held-out rows."""
    out = {'record': _plain(entry_record(entry)), 'rebuild_model': model}
    t0 = time.time()
    try:
        Xtr, Xte, fi = build_fold_entry(
            data, entry, train_idx=train_idx, val_idx=test_idx,
            model_factory=factories.get(model) if model else None,
            metric_fn=metric_fn, lower_is_better=dataset_lower)
    except Exception as e:
        out['build_failed'] = f'{type(e).__name__}: {e}'
        return out
    out['width_built'] = int(np.asarray(Xte).shape[1])
    out['build_seconds'] = round(time.time() - t0, 1)
    y = np.asarray(data['labels'])
    out['scores'] = _score_space(Xtr, Xte, y[np.asarray(test_idx)], axes=axes,
                                 n_sample=n_sample, is_clf=is_clf)
    return out


def _comparisons(singles, hybrids, pairs, *, axes):
    """Hybrid minus single representation, per held-out fold, per distance,
    both unweighted. Positive = the hybrid's distance follows activity better.
    """
    out = {}
    for name, h_arm, s_rep in pairs:
        h = (hybrids.get(h_arm) or {}).get('scores')
        s = (singles.get(s_rep) or {}).get('scores') if s_rep else None
        rows = []
        for metric in axes['metrics']:
            hv = _rho(h, metric, 'unweighted')
            sv = _rho(s, metric, 'unweighted')
            rows.append({
                'distance': metric, 'hybrid_weighting': 'unweighted',
                'single_weighting': 'unweighted', 'hybrid_sac_rho': hv,
                'single_sac_rho': sv,
                'delta': (hv - sv if hv is not None and sv is not None
                          else None)})
        out[name] = {'hybrid_arm': h_arm, 'single_rep': s_rep, 'rows': rows}
    return out


def _model_pairs(sel_models, knn_first, *, distance_arm, similarity_arm,
                 prefix=''):
    """(comparison name, hybrid arm, single representation) for one fold.

    Each decided model's hybrids are set against that model's best base
    representation (rank 1 in the same per-fold search); knn's distance hybrid
    and the similarity hybrid against knn's best single representation from
    the per-fold distance search.
    """
    pairs = []
    for m, ms in (sel_models or {}).items():
        if not ms:
            continue
        b1 = ms['base'][0] if ms.get('base') else None
        for h_i in range(len(ms.get('hybrids') or [])):
            arm = f'{prefix}{m}/hybrid{h_i + 1}'
            pairs.append((f'{m} hybrid {h_i + 1} - {m} base rank 1', arm, b1))
    if distance_arm:
        pairs.append(('knn distance hybrid - knn base rank 1', distance_arm,
                      knn_first))
    if similarity_arm:
        pairs.append(('similarity hybrid - knn base rank 1', similarity_arm,
                      knn_first))
    return pairs


def _similarity_top(data, dataset_name, *, train_idx, fold_i, n_folds):
    """(entry, None) or (None, reason): the top hybrid, over every model
    scored, of this fold's similarity-objective search on its training rows.
    Same search and arguments as fold_similarity_selection."""
    from experiments._shared import fold_searches, top_hybrids_across_models
    from experiments.dataset_policy import similarity_objective_undefined
    labels = np.asarray(data['labels'])
    task_type = data.get('task_type', 'auto')
    why = similarity_objective_undefined(dataset_name, labels, task_type)
    if why:
        return None, why
    try:
        srch = fold_searches(
            data['reps'], labels, data.get('groups'), tier='general',
            objective='similarity', train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_folds, dataset_name=dataset_name,
            augmentations=data.get('augmentations'), task_type=task_type,
            seed=SIM_FOLD_SEED).get('none')
        if srch is None:
            return None, 'the per-fold similarity search failed'
        th = top_hybrids_across_models(
            srch, n=1, context=f'{dataset_name} similarity fold {fold_i}')
    except Exception as exc:
        return None, (f'the per-fold similarity search raised '
                      f'{type(exc).__name__}: {exc}')
    if not th['selected']:
        return None, 'the per-fold similarity search scored no hybrid'
    return dict(th['selected'][0], aug_condition='none'), None


def _budget_selection(data, dataset_name, *, train_idx, fold_i, n_folds,
                      budget, in_general, in_distance):
    """The hybrids at one decided budget, re-read from this fold's searches.

    The searches are the ones fold_similarity_selection already ran (the
    in-process memo in _shared.fold_searches is keyed on the same arguments),
    so this adds no search; it adds the identical-features builds that
    fold_standard_selection makes for the hybrids it ranks.
    """
    from experiments._shared import (
        fold_standard_selection, fold_searches, top_hybrids_across_models)
    from experiments.dataset_policy import similarity_objective_undefined
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data.get('task_type', 'auto')
    common = dict(train_idx=train_idx, fold_i=fold_i, n_outer_folds=n_folds,
                  dataset_name=dataset_name,
                  augmentations=data.get('augmentations'),
                  task_type=task_type, seed=SIM_FOLD_SEED)
    got = {'models': {}, 'distance_hybrid': None, 'similarity_hybrid': None,
           'unavailable': {}}
    if in_general:
        try:
            sel = fold_standard_selection(reps, labels, groups, budget=budget,
                                          **common)
            for m, s in sel.items():
                got['models'][m] = (None if s is None else
                                    {'base': s.get('base'),
                                     'hybrids': s.get('hybrids'),
                                     'hybrids_skipped_identical_features':
                                         s.get('hybrids_skipped_identical_features')})
                if s is not None and not s.get('hybrids'):
                    got['unavailable'][f'general/{m}'] = (
                        f'the per-fold general search scored no hybrid at '
                        f'{budget} features for {m}')
        except Exception as e:
            got['unavailable']['general'] = f'{type(e).__name__}: {e}'
        why = similarity_objective_undefined(dataset_name, labels, task_type)
        if why:
            got['unavailable']['similarity_hybrid'] = why
        else:
            try:
                srch = fold_searches(reps, labels, groups, tier='general',
                                     objective='similarity',
                                     **common).get('none')
                if srch is None:
                    got['unavailable']['similarity_hybrid'] = (
                        'the per-fold similarity search failed')
                else:
                    # top_hybrids_across_models already passes
                    # require_full=False to top_hybrids; passing it here too
                    # raised a TypeError at every budget.
                    th = top_hybrids_across_models(
                        srch, n=1, budget=budget,
                        context=f'{dataset_name} similarity fold {fold_i} '
                                f'budget {budget}')
                    if th['selected']:
                        got['similarity_hybrid'] = dict(th['selected'][0],
                                                        aug_condition='none')
                    else:
                        got['unavailable']['similarity_hybrid'] = (
                            f'the per-fold similarity search scored no hybrid '
                            f'at {budget} features')
            except Exception as e:
                got['unavailable']['similarity_hybrid'] = (
                    f'{type(e).__name__}: {e}')
    if in_distance:
        try:
            sd = fold_standard_selection(reps, labels, groups, distance=True,
                                         models=['knn'], n_hybrids=1,
                                         budget=budget, **common)
            k_sel = sd.get('knn')
            if k_sel and k_sel.get('hybrids'):
                got['distance_hybrid'] = k_sel['hybrids'][0]
            else:
                got['unavailable']['distance_hybrid'] = (
                    f'the per-fold distance search scored no knn hybrid at '
                    f'{budget} features')
        except Exception as e:
            got['unavailable']['distance_hybrid'] = f'{type(e).__name__}: {e}'
    return got


def _fold_unit(data, dataset_name, *, fold_i, n_folds, train_idx, test_idx,
               axes, n_sample, is_clf, metric_fn,
               dataset_lower, budgets_general, budgets_distance, models):
    """Everything s02 measures on one outer fold. JSON-plain."""
    t_fold = time.time()
    reps = data['reps']
    y = np.asarray(data['labels'])
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    task_type = data.get('task_type', 'regression')

    sel = fold_similarity_selection(
        data, dataset_name, train_idx=tr, fold_i=fold_i, n_outer_folds=n_folds,
        neighbour_spaces=True, verbose=False)
    # similarity_helpers.fold_similarity_selection passes require_full=False
    # to top_hybrids_across_models, which passes it again to top_hybrids; the
    # TypeError is caught there and the similarity hybrid comes back None in
    # every fold. Until that is fixed in the shared file, s02 re-reads the top
    # hybrid of the same per-fold similarity search (memoised in
    # _shared.fold_searches, so no new search) and records that it did.
    sim_why = str((sel.get('unavailable') or {}).get('similarity_hybrid', ''))
    if sel.get('similarity_hybrid') is None and 'require_full' in sim_why:
        sel = dict(sel, unavailable=dict(sel.get('unavailable') or {}))
        sel['unavailable'].pop('similarity_hybrid', None)
        sim_entry, sim_err = _similarity_top(
            data, dataset_name, train_idx=tr, fold_i=fold_i, n_folds=n_folds)
        if sim_entry is not None:
            sel['similarity_hybrid'] = dict(
                sim_entry, read_by='s02 (shared selection raised: '
                                   + sim_why + ')')
        else:
            sel['unavailable']['similarity_hybrid'] = sim_err

    factories = get_model_factories(list(models) + ['knn'],
                                    task_type=task_type, n_samples=len(tr))

    # Single-representation distance spaces: knn's 2 from the distance search
    # and each decided model's 2 base representations (Part 1 step 5; the
    # other representations of the pool are not neighbour arms).
    sources = {}
    for r_i, rep in enumerate(sel.get('knn_base') or []):
        sources.setdefault(rep, []).append(
            f'knn base rank {r_i + 1} (per-fold distance search)')
    for m, ms in (sel.get('models') or {}).items():
        for r_i, rep in enumerate((ms or {}).get('base') or []):
            sources.setdefault(rep, []).append(
                f'{m} base rank {r_i + 1} (per-fold general search)')
    singles = {}
    for rep, src in sources.items():
        if rep not in reps:
            singles[rep] = {'selected_as': src,
                            'not_computed': 'not in the loaded representations'}
            continue
        X = np.asarray(reps[rep], dtype=np.float32)
        singles[rep] = {
            'selected_as': src, 'dim': int(X.shape[1]),
            'scores': _score_space(X[tr], X[te], y[te], axes=axes,
                                   n_sample=n_sample, is_clf=is_clf)}

    def _hyb(entry, model):
        return _score_hybrid(
            data, entry, model=model, train_idx=tr, test_idx=te,
            factories=factories, metric_fn=metric_fn,
            dataset_lower=dataset_lower, axes=axes, n_sample=n_sample,
            is_clf=is_clf)

    hybrids = {}
    for m, ms in (sel.get('models') or {}).items():
        for h_i, e in enumerate((ms or {}).get('hybrids') or []):
            hybrids[f'{m}/hybrid{h_i + 1}'] = _hyb(e, m)
    if sel.get('distance_hybrid') is not None:
        hybrids['knn/distance_hybrid'] = _hyb(sel['distance_hybrid'], 'knn')
    if sel.get('similarity_hybrid') is not None:
        # A similarity-objective recipe fits no model when it is rebuilt.
        hybrids['similarity_hybrid'] = _hyb(sel['similarity_hybrid'], None)

    knn_first = (sel.get('knn_base') or [None])[0]
    pairs = _model_pairs(
        sel.get('models'), knn_first,
        distance_arm=('knn/distance_hybrid' if 'knn/distance_hybrid' in hybrids
                      else None),
        similarity_arm=('similarity_hybrid' if 'similarity_hybrid' in hybrids
                        else None))
    comparisons = _comparisons(singles, hybrids, pairs, axes=axes)

    # The budget sweep: the same comparisons at each decided budget.
    by_budget = {}
    for b in sorted(set(budgets_general) | set(budgets_distance)):
        bsel = _budget_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i, n_folds=n_folds,
            budget=b, in_general=b in budgets_general,
            in_distance=b in budgets_distance)
        b_h = {}
        for m, ms in bsel['models'].items():
            for h_i, e in enumerate((ms or {}).get('hybrids') or []):
                b_h[f'{m}/hybrid{h_i + 1}'] = _hyb(e, m)
        if bsel['distance_hybrid'] is not None:
            b_h['knn/distance_hybrid'] = _hyb(bsel['distance_hybrid'], 'knn')
        if bsel['similarity_hybrid'] is not None:
            b_h['similarity_hybrid'] = _hyb(bsel['similarity_hybrid'], None)
        # The single representation each hybrid is set against is the one the
        # fold's main selection ranked first; a budget does not change it.
        b_pairs = _model_pairs(
            {m: {'base': ((sel.get('models') or {}).get(m) or {}).get('base'),
                 'hybrids': (ms or {}).get('hybrids')}
             for m, ms in bsel['models'].items()},
            knn_first,
            distance_arm=('knn/distance_hybrid' if 'knn/distance_hybrid' in b_h
                          else None),
            similarity_arm=('similarity_hybrid' if 'similarity_hybrid' in b_h
                            else None))
        by_budget[str(b)] = {
            'budget_features': int(b),
            'searches_read': (['general', 'similarity'] if b in budgets_general
                              else []) + (['general_distance']
                                          if b in budgets_distance else []),
            'hybrids': b_h,
            'hybrids_skipped_identical_features': {
                m: _plain((ms or {}).get('hybrids_skipped_identical_features'))
                for m, ms in bsel['models'].items()},
            'comparisons': _comparisons(singles, b_h, b_pairs, axes=axes),
            'unavailable': bsel['unavailable'],
        }

    sel_models = {}
    for m, ms in (sel.get('models') or {}).items():
        if ms is None:
            sel_models[m] = None
            continue
        sel_models[m] = {
            'base': list(ms.get('base') or []),
            'base_train_rows_scores': _plain(
                (ms.get('base_detail') or {}).get('selected')),
            'hybrids': [_plain(entry_record(e)) for e in ms.get('hybrids') or []],
            'hybrids_skipped_identical_features': _plain(
                ms.get('hybrids_skipped_identical_features')),
        }
    return {
        'fold': int(fold_i),
        'n_outer_folds': int(n_folds),
        'n_train': int(len(tr)), 'n_test': int(len(te)),
        'selection_scope': 'fold_train_rows',
        'selection': {
            'models': sel_models,
            'knn_base': list(sel.get('knn_base') or []),
            'knn_base_train_rows_scores': _plain(
                (sel.get('knn_base_detail') or {}).get('selected')),
            'distance_hybrid': _plain(entry_record(sel.get('distance_hybrid'))),
            'similarity_hybrid': _plain(
                entry_record(sel.get('similarity_hybrid'))),
            'unavailable': _plain(sel.get('unavailable')),
        },
        'singles': singles,
        'hybrids': hybrids,
        'comparisons': comparisons,
        'by_budget': by_budget,
        'fold_seconds': round(time.time() - t_fold, 1),
    }


# =============================================================================
# ACROSS FOLDS: per-fold numbers and their spread, nothing else
# =============================================================================

def _summarise(fold_payloads, n_folds, get_comparisons):
    """{comparison: [row per (distance, weighting)]}, each row holding the
    per-fold deltas in fold order (None where that fold has none), their mean
    over the folds that have one, and the smallest and largest. No test, no
    interval (handoff 2.4 rule 11)."""
    acc = {}
    for f_i in range(n_folds):
        p = fold_payloads.get(f_i)
        comps = get_comparisons(p) if p else None
        for name, c in (comps or {}).items():
            for r in c.get('rows', []):
                key = (name, r['distance'], r['hybrid_weighting'],
                       r['single_weighting'])
                acc.setdefault(key, [None] * n_folds)[f_i] = r.get('delta')
    out = {}
    for (name, dist, hw, sw), vals in sorted(acc.items()):
        ok = [v for v in vals if v is not None]
        out.setdefault(name, []).append({
            'distance': dist, 'hybrid_weighting': hw, 'single_weighting': sw,
            'delta_per_fold': vals,
            'n_folds_with_delta': len(ok),
            'delta_mean_of_folds': float(np.mean(ok)) if ok else None,
            'delta_min': float(min(ok)) if ok else None,
            'delta_max': float(max(ok)) if ok else None,
            'spread_note': fold_spread_note(ok),
        })
    return out


# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Per outer fold: is the hybrid's distance closer to activity than the
    best single representation's, on the held-out molecules?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT S02: HYBRID PROPERTY-AWARE SIMILARITY")
    print("=" * 70)

    from experiments.decided_grid import decided_search, decided_models, is_decided

    # DECIDED 2026-10-06 (Adelaide delegated): no importance weighting in
    # s02's distances. A 'weight_methods' argument is recorded and not used.
    ignored_weight_methods = kwargs.get('weight_methods')
    if ignored_weight_methods is not None:
        print(f"  s02: ignoring weight_methods={ignored_weight_methods!r}; "
              f"s02 scores unweighted distances only")
    n_sample = kwargs.get('n_sample', 5000)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    labels = np.asarray(data['labels'])

    metric_name, metric_fn, dataset_lower = get_domain_metric(data)
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    budgets_general = [int(b) for b in
                       decided_search(dataset_name, 'general')['budgets']]
    budgets_distance = ([int(b) for b in decided_search(
        dataset_name, 'general_distance')['budgets']]
        if is_decided(dataset_name, 'general_distance') else [])
    axes = knn_axes(dataset_name, len(labels))
    n_folds = int(config['n_splits'])

    results = {
        'selection_rule': _SELECTION_RULE,
        'selection_scope': 'fold_train_rows',
        # Every sac_rho and delta here is the Spearman between the distance of
        # a held-out pair and their activity difference; higher is better. The
        # dataset's own metric only ranked the searches' configurations.
        'metric_name': 'spearman',
        'lower_is_better': False,
        'metric_definition': (
            'Spearman correlation between pairwise distance and pairwise '
            'activity difference (1 if the two are in different classes on a '
            'classification dataset), over sampled pairs of held-out '
            'molecules; higher is better'),
        'dataset_metric': metric_name,
        'dataset_metric_lower_is_better': bool(dataset_lower),
        'models': models,
        'n_outer_folds': n_folds,
        'n_sample_pairs': int(n_sample),
        'knn_axes': axes,
        'weighting': 'unweighted only (decided 2026-10-06)',
        'ignored_weight_methods': ignored_weight_methods,
        'budgets': {
            'general': budgets_general, 'distance': budgets_distance,
            'sweep_source': (
                "re-read from each fold's own searches: "
                "fold_standard_selection(budget=b) for the decided models and "
                "for knn on the distance search, top_hybrids_across_models("
                "budget=b) on the similarity search; no search runs for the "
                "sweep"),
        },
    }
    print(f"  {n_folds} outer fold(s); models {models}; distances "
          f"{axes['metrics']}; neighbour counts {axes['ks']}; schemes "
          f"{axes['schemes']}; budgets general {budgets_general}, distance "
          f"{budgets_distance}")

    # Each fold is written to s02_hybrid_similarity/<dataset>.partial.jsonl
    # when it finishes and read back on the next attempt. The identity names
    # the per-fold design, so a sidecar written under the whole-dataset design
    # (before 2026-10-05) is never read back.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's02_hybrid_similarity', dataset_name),
        run_identity('s02_hybrid_similarity', dataset_name, data, config,
                     kwargs=kwargs, n_sample=n_sample, models=models,
                     budgets_general=budgets_general,
                     budgets_distance=budgets_distance, knn_axes=axes,
                     design='per_fold_selection_unweighted_2026_10_06'))

    folds = list(sim_outer_folds(data, config))
    payloads, fold_errors = {}, {}
    for f_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  -- fold {f_i + 1} of {len(folds)}: {len(train_idx)} "
              f"training, {len(test_idx)} held-out molecules --")
        if _ck.done(fold=f_i):
            payloads[f_i] = _ck.get(fold=f_i)
            continue
        try:
            p = _fold_unit(
                data, dataset_name, fold_i=f_i, n_folds=len(folds),
                train_idx=train_idx, test_idx=test_idx, axes=axes,
                n_sample=n_sample, is_clf=is_clf, metric_fn=metric_fn,
                dataset_lower=dataset_lower, budgets_general=budgets_general,
                budgets_distance=budgets_distance, models=models)
        except _CODE_FAULTS:
            raise  # a fault in the code, not a failed search
        except Exception as e:
            import traceback
            fold_errors[str(f_i)] = f'{type(e).__name__}: {e}'
            print(f"  fold {f_i} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        payloads[f_i] = p
        _ck.put(p, fold=f_i)
        for name, c in p['comparisons'].items():
            for r in c['rows']:
                if r['hybrid_weighting'] == 'unweighted' and r['delta'] is not None:
                    print(f"    {name} [{r['distance']}]: hybrid "
                          f"{r['hybrid_sac_rho']:.4f}, single "
                          f"{c['single_rep']} {r['single_sac_rho']:.4f}, "
                          f"delta {r['delta']:+.4f}")

    results['n_outer_folds'] = len(folds)
    results['folds'] = [payloads.get(f_i, {'fold': f_i, 'failed':
                                           fold_errors.get(str(f_i))})
                        for f_i in range(len(folds))]
    results['fold_errors'] = fold_errors
    results['summary'] = _summarise(payloads, len(folds),
                                    lambda p: p.get('comparisons'))
    results['summary_by_budget'] = {
        str(b): _summarise(
            payloads, len(folds),
            lambda p, _b=str(b): ((p.get('by_budget') or {}).get(_b) or {})
            .get('comparisons'))
        for b in sorted(set(budgets_general) | set(budgets_distance))}

    print("\n  Summary (mean of held-out folds, smallest, largest):")
    for name, rows in results['summary'].items():
        for r in rows:
            if r['hybrid_weighting'] != 'unweighted' or r['delta_mean_of_folds'] is None:
                continue
            print(f"    {name} [{r['distance']}]: {r['delta_mean_of_folds']:+.4f} "
                  f"({r['delta_min']:+.4f} to {r['delta_max']:+.4f}, "
                  f"{r['n_folds_with_delta']} fold(s))"
                  + (f" {r['spread_note']}" if r['spread_note'] else ''))

    # Sequence-based cliff sweep (antibody datasets + nucleotide) -- independent
    # of any representation choice and of any selection. Same sweep as s01.
    cliff_results = run_antibody_cliff_analysis(
        data, is_clf=is_clf, dataset_name=dataset_name)
    if cliff_results is not None:
        results['_sequence_cliffs'] = cliff_results

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S02: Hybrid Property-Aware Similarity',
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
    save_experiment_results(results, args.output, 's02_hybrid_similarity', args.dataset)
