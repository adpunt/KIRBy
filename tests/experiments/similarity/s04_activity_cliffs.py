#!/usr/bin/env python3
"""S04: Activity Cliffs -- RMSE_cliff improvement.

Paper: Similarity
Conclusion: Hybrid reduces RMSE on activity cliffs.
Dependencies: S02 (reads JSON, optional).

Which molecules are cliffs comes from similarity_helpers.calibrated_cliff_rule,
which s12 also calls, so the two files no longer partition a dataset two
different ways. ECFP4 Tanimoto at or above 0.55 on the structural side, and an
activity gap solved per dataset so a stated share of the molecules are cliffs.
CV evaluation of kNN on cliff vs non-cliff molecules for: per-rep kNN baselines,
the pool's own single representation, the pool's hybrid recipes (unweighted and
importance-weighted), RF baseline.

2026-09-12, owner's decision: this file runs NO hybrid recipe search. For each
held-out fold it reads the pool -- the best single representation plus the top
three hybrid configurations for its downstream model (kNN), ranked on the other
folds' scores. Its subset admits both families of stored search: the predictive
ones and the ones built under objective='similarity', where a configuration is
scored by predicting each held-out molecule's activity from its five nearest
neighbours. Three per family, so six hybrid arms per fold, not three.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, load_experiment_results,
    flatten_fold_weights, get_fold_pool_recipes,
    get_domain_metric, get_model_factories, build_fold_hybrid,
    fold_hybrid_scoring_kwargs,
    DISTANCE_BUDGETS, DISTANCE_EVALUATORS,
)
from experiments.similarity.similarity_helpers import (
    calibrated_cliff_rule, cliff_structural_similarity, CLIFF_TARGET_PCT)
from kirby.find_best_hybrid import nested_cv_outer_folds

# 2026-09-12, owner's decision (EXPERIMENTS_FINALIZED.md, s04): s04 stops
# calling get_fold_best_search and reads the pool instead. So this seed no
# longer names a per-fold search cache -- it is the seed the folds are built
# with, and it has to be 42 because that is the seed the stored pool searches
# split on. Fold i here is fold i there only while both are 5 folds, seed 42,
# same groups and task type.
_FOLD_SEARCH_SEED = 42

# The subset of search families the pool may serve from: the hybrids built
# under the predictive objective, and the ones built under
# objective='similarity' (each held-out molecule's activity predicted from its
# five nearest neighbours). The owner put similarity hybrids in s04's subset.
_POOL_SUBSET = 'predictive_and_similarity'

# How many hybrid configurations the pool returns per family, per fold.
_N_POOL_HYBRIDS = 3

from kirby.hybrid import apply_feature_selection, compute_feature_importance
from experiments.similarity._summary_keys import (
    SELECTION_RULE as _SELECTION_RULE)

# 2026-09-20: the ECFP4 build and the cliff rule that read it moved to
# similarity_helpers (cliff_structural_similarity, calibrated_cliff_rule), so
# s12 decides which molecules are cliffs with the same two calls instead of a
# rule of its own. What that fixed is written at the top of that block.

# =============================================================================
# EXPERIMENT
# =============================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """Prediction performance on activity cliff molecules.

    The cliff rule is shared with s12: ECFP4 Tanimoto at or above
    tanimoto_threshold, and an activity gap solved so cliff_target_pct per cent
    of the molecules are cliffs. Pass activity_k to use a fixed multiple of the
    labels' median absolute deviation instead.

    Reports RMSE_cliff, RMSE_noncliff for:
      - Per-rep kNN baselines
      - Hybrid kNN (unweighted and weighted)
      - RF baseline
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
    # How many nearest neighbours every kNN arm below predicts from, and the
    # exponent the importance weights are raised to before they multiply the
    # standardised features. Both were written into the code as 5 and a square
    # root; they are arguments so the same file can be run at other settings and
    # the arms compared under the held-out re-pick.
    knn_k = int(kwargs.get('knn_k', 5))
    weight_exponent = float(kwargs.get('weight_exponent', 0.5))
    n_folds = config['n_splits']
    mode = 'intermediary' if config['n_splits'] == 1 else 'full'

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']

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

    results = {
        # Which selection rule produced the numbers below. s09 reads this
        # and refuses to run its signed-rank test across a mixed set — see
        # _summary_keys.SELECTION_RULE. Added 2026-09-18.
        'selection_rule': _SELECTION_RULE,
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
        # The two settings every kNN arm below ran at. Before 2026-09-20 they
        # were 5 and a square root written into the code with no way to change
        # them.
        'knn_settings': {
            'k_neighbours': knn_k,
            'importance_weight_exponent': weight_exponent,
            'note': ('every molecule is predicted from its k nearest '
                     'neighbours; the weighted arms multiply each standardised '
                     'feature by its importance raised to this exponent, 0.5 '
                     'being the square root that was hardcoded'),
        },
        'per_rep': {},
        'hybrid': {},
    }

    # --- Per-fold reselection accumulators (this file's own cliff-CV folds) ---
    # The honest hybrid-vs-best-single comparison is done by the reader via the
    # shared fold_reselection() helper. We only persist the per-fold inputs here,
    # keyed by downstream MODEL, using the CONTRACT json_convention names. This
    # file runs its OWN CV (kNN + RF on the shared splitter), so we build the
    # block MANUALLY rather than build_fold_reselection_input(search) — the search's
    # internal baselines are on different folds/metric. Fold scores are POSITIONAL
    # (one slot per split, NaN where a fold was skipped) so both arms stay aligned
    # across the identical split order; the helper skips NaN slots.
    reselection_candidates = {}  # model -> [ {name, fold_scores}, ... ]
    reselected_baselines = {}   # model -> { rep_name: [fold_scores], ... }

    # --- CV evaluation of prediction on cliff vs non-cliff ---
    # The folds come from nested_cv_outer_folds, the helper each fold's recipe
    # search caches against, so fold i's recipe was searched on exactly fold i's
    # training rows. Same group-aware split as before: GroupKFold, or
    # StratifiedGroupKFold on a classification dataset.
    sample_X = next(iter(reps.values()))
    fold_splits = nested_cv_outer_folds(reps, labels, groups, n_folds,
                                        _FOLD_SEARCH_SEED, task_type)

    # --- Held-out cliff molecules whose cliff partner is in the training rows ---
    # A molecule is a cliff when ANY molecule in the dataset is its cliff
    # partner. Under a scaffold split that partner is often held out in the same
    # fold, so no model ever sees it. Measured 2026-09-28 at 3 folds: 74 of 211
    # held-out cliff molecules on herg_ki and 106 of 324 on caco2 had no partner
    # in their fold's training rows. Those molecules are scored as cliffs but
    # nothing about the cliff can reach the prediction. This records how many
    # there are and scores the other subset separately. It changes no
    # existing number.
    from experiments.similarity.similarity_helpers import (
        _pairwise_delta_matrix as _pdm)
    _cliff_pairs = ((np.asarray(cliff_sim) >= cliff_cutoff)
                    & (_pdm(np.asarray(labels), is_clf=is_clf)
                       > cliff_info['activity_threshold']))
    np.fill_diagonal(_cliff_pairs, False)
    # {fold_i: positions WITHIN val_idx of cliff molecules with a partner in train}
    _cliff_seen_pos = {}
    _split_summary = []
    for _fi_cs, (_tr_cs, _va_cs) in enumerate(fold_splits):
        _vc = np.array([i for i, vi in enumerate(_va_cs) if vi in cliff_indices],
                       dtype=int)
        if len(_vc):
            _seen = _cliff_pairs[np.ix_(_va_cs[_vc], _tr_cs)].any(axis=1)
        else:
            _seen = np.zeros(0, dtype=bool)
        _cliff_seen_pos[_fi_cs] = _vc[_seen]
        _split_summary.append({
            'fold': _fi_cs,
            'n_cliff_held_out': int(len(_vc)),
            'n_cliff_held_out_partner_in_train': int(_seen.sum()),
            'n_cliff_held_out_partner_only_held_out': int((~_seen).sum()),
        })
    _fold_of = np.full(len(labels), -1, dtype=int)
    for _fi_cs, (_tr_cs, _va_cs) in enumerate(fold_splits):
        _fold_of[_va_cs] = _fi_cs
    _I, _J = np.where(np.triu(_cliff_pairs, 1))
    results['cliff_partner_split'] = {
        'n_cliff_pairs': int(len(_I)),
        'n_cliff_pairs_both_in_one_held_out_fold': int(
            (_fold_of[_I] == _fold_of[_J]).sum()),
        'by_fold': _split_summary,
        'note': ('a cliff molecule is one with a cliff partner anywhere in '
                 'the dataset; partner_in_train counts the held-out cliff '
                 'molecules whose partner sits in that fold\'s training rows, '
                 'the only ones whose cliff a model can meet. Every arm also '
                 'carries score_cliff_partner_in_train over that subset.'),
    }
    del _cliff_pairs

    # --- Per-fold pool read (2026-09-12, owner's decision) -------------------
    # DECIDED: s04 stops calling get_fold_best_search. It reads the pool
    # instead -- the best single representation plus the top three hybrids for
    # its downstream model, chosen per fold from the other folds' scores.
    # Similarity hybrids are in its subset.
    #
    # What changed. Nothing here runs a hybrid recipe search any more. For each
    # held-out fold, get_fold_pool_recipes reads the stored searches' per-fold
    # scores and ranks both the single representations and the hybrid
    # configurations by their mean over the OTHER folds. Fold i's own score
    # never enters its own ranking. The searches themselves ran over every row,
    # so this holds out the fold's SCORES, not its rows -- and where a pool
    # search pruned mid-flight, fold 0's scores chose which configurations
    # survived at all ('fold0_pruned').
    budget = kwargs.get('budget', 100)
    augmentations = data.get('augmentations')
    _pool_cache = {}

    def _fold_pool(fold_i, model):
        """The pool read for held-out fold `fold_i`, ranked for `model`."""
        key = (int(fold_i), model)
        if key not in _pool_cache:
            _pool_cache[key] = get_fold_pool_recipes(
                reps, labels, groups,
                fold_i=fold_i, n_outer_folds=len(fold_splits),
                downstream_model=model, budget=budget,
                dataset_name=dataset_name, augmentations=augmentations,
                distance=True, data=data,
                metric=metric_name, task_type=task_type, domain=domain,
                pool_subset=_POOL_SUBSET, n_hybrids=_N_POOL_HYBRIDS)
        return _pool_cache[key]

    knn_pools = {}
    hybrid_arms = []          # [(objective, rank), ...] in subset order
    fold0_pruned = None
    try:
        for _fi in range(len(fold_splits)):
            knn_pools[_fi] = _fold_pool(_fi, 'knn')
        results['hybrid_search_status'] = {
            'ok': bool(any(p['hybrids'] for p in knn_pools.values())),
            'error': (None if any(p['hybrids'] for p in knn_pools.values())
                      else 'the pool held no kNN hybrid configuration for any fold'),
        }
    except Exception as e:
        print(f"  Pool read failed: {type(e).__name__}: {e}")
        # F17: persist the throw. Without this the 'hybrid' block stays the empty
        # dict it was initialised to while 'per_rep' fills normally, so the file
        # reads as a complete run with no hybrid arm.
        results['hybrid_search_status'] = {
            'ok': False,
            'error_type': type(e).__name__,
            'error': str(e),
        }

    if knn_pools:
        _p0 = knn_pools[0]
        # One arm per (family, rank) slot. The RECIPE inside a slot is re-picked
        # for every fold from that fold's other-folds ranking, so the arm is the
        # slot, not one fixed recipe. Under 'predictive_and_similarity' that is
        # three predictive slots plus three similarity slots, not three in total.
        _seen = []
        for _fi in sorted(knn_pools):
            for h in knn_pools[_fi]['hybrids']:
                if (h['objective'], h['rank']) not in _seen:
                    _seen.append((h['objective'], h['rank']))
        _obj_order = {o: i for i, o in enumerate(_p0['objectives'])}
        hybrid_arms = sorted(_seen, key=lambda k: (_obj_order.get(k[0], 99), k[1]))

        for _fi in sorted(knn_pools):
            for w in knn_pools[_fi]['warnings']:
                print(f"  [pool fold {_fi}] {w}")
        fold0_pruned = _p0.get('fold0_pruned')

        # What the pool chose, per fold, so the file records which single
        # representation and which recipes each fold's ranking picked. It is a
        # per-fold choice, not a whole-dataset ranking.
        results['pool_selection'] = {
            str(_fi): {
                'base_rep': (knn_pools[_fi]['base_rep'] or {}).get('rep'),
                'base_rep_score_other_folds':
                    (knn_pools[_fi]['base_rep'] or {}).get('score_other_folds'),
                'base_reps_ranked': [
                    (b['rep'], b['score_other_folds'])
                    for o in knn_pools[_fi]['by_objective']
                    for b in knn_pools[_fi]['by_objective'][o]['base_reps_ranked'][:3]
                ],
                'hybrids': [
                    {'name': h['name'], 'objective': h['objective'],
                     'rank': h['rank'], 'recipe': h['recipe'],
                     'metric_name': h['metric_name'],
                     'score_other_folds': h['score_other_folds']}
                    for h in knn_pools[_fi]['hybrids']],
                'selection_scope': knn_pools[_fi]['selection_scope'],
                'fold0_pruned': knn_pools[_fi]['fold0_pruned'],
                'n_folds_in_pool': knn_pools[_fi]['n_folds_in_pool'],
                'warnings': knn_pools[_fi]['warnings'],
            }
            for _fi in sorted(knn_pools)
        }
        results['pool_subset'] = _p0['pool_subset']
        results['pool_objectives'] = list(_p0['objectives'])
        results['pool_selection_note'] = _p0['selection_note']

        _b0 = _p0['base_rep']
        if _b0 is not None:
            print(f"  Pool base representation, fold 0 (by knn, other folds' "
                  f"{_b0['metric_name']}): {_b0['rep']}="
                  f"{_b0['score_other_folds']:.4f}")
        for _obj in _p0['by_objective']:
            _ranked = _p0['by_objective'][_obj]['base_reps_ranked'][:3]
            if _ranked:
                print(f"  Top-3 base reps, fold 0 ({_obj}, by knn): "
                      + ', '.join(f"{b['rep']}={b['score_other_folds']:.4f}"
                                  for b in _ranked))
        # Kept under its old key so the readers of s04's JSON still find it. It
        # is now the pool's kNN ranking on fold 0's other folds, not an RF
        # ranking out of a fold-0 search.
        results['top3_base_reps_first_fold'] = [
            (b['rep'], b['score_other_folds'])
            for o in _p0['by_objective']
            for b in _p0['by_objective'][o]['base_reps_ranked'][:3]]
        if hybrid_arms:
            print(f"  Pool hybrid arms: {len(hybrid_arms)} "
                  f"({', '.join(f'{o} rank {r}' for o, r in hybrid_arms)})")

    hybrid_blueprint_available = bool(hybrid_arms)

    def _make_knn(n_train):
        k = min(knn_k, n_train - 1)
        if is_clf:
            return KNeighborsClassifier(n_neighbors=k)
        return KNeighborsRegressor(n_neighbors=k)

    # Every kNN condition below (each representation unweighted and weighted,
    # the pool's per-fold single representation, each pool hybrid arm) is one
    # call of _evaluate_condition. Its result is written to
    # s04_activity_cliffs/<dataset>.partial.jsonl when the call returns and
    # read back on the next attempt (2026-09-25; _checkpoint.py). A condition
    # is a pure function of the folds, the labels and the pool read: no random
    # state is shared between conditions. A condition in which any fold's
    # features raised while building, or that scored no fold (None), is not
    # stored, so it is retried.
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's04_activity_cliffs', dataset_name),
        run_identity('s04_activity_cliffs', dataset_name, data, config,
                     kwargs=kwargs, budget=budget, knn_k=knn_k,
                     weight_exponent=weight_exponent))

    def _evaluate_condition(get_Xy_fn, pre_scaled=False, ck_name=None):
        """Run CV for a condition, return cliff/noncliff metrics."""
        if ck_name is not None and _ck.done(condition=ck_name):
            return _ck.get(condition=ck_name)
        _threw = []
        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_all, fold_rmse_cliff, fold_rmse_noncliff = [], [], []
        # Positional per-fold metrics for fold_reselection, one slot per split so
        # every arm stays index-aligned. NaN where the fold was dropped, and on
        # the cliff/non-cliff lists also where the held-out fold carried 1 or
        # fewer rows of that stratum.
        fold_scores_pos = []
        fold_cliff_pos, fold_noncliff_pos = [], []
        # How many held-out rows of each stratum the fold carried. score_cliff
        # and score_noncliff can rest on different numbers of folds, and
        # neither count was written out before.
        n_cliff_rows_pos, n_noncliff_rows_pos = [], []
        fold_cliff_seen, fold_cliff_seen_pos = [], []

        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]

            # Stratify by cliff membership. Depends only on the split, so it is
            # counted even for a fold whose features failed to build.
            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices], dtype=int)
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices], dtype=int)
            n_cliff_rows_pos.append(int(len(val_cliff)))
            n_noncliff_rows_pos.append(int(len(val_noncliff)))

            def _drop_fold():
                fold_scores_pos.append(float('nan'))
                fold_cliff_pos.append(float('nan'))
                fold_noncliff_pos.append(float('nan'))
                fold_cliff_seen_pos.append(float('nan'))

            try:
                X_tr, X_va = get_Xy_fn(train_idx, val_idx, y_train, fold_i)
            except Exception as exc:
                # N9: log the dropped fold so a thrown build is distinguishable
                # from a genuinely empty fold (X_tr is None below).
                print(f"      cliff CV fold build failed: "
                      f"{type(exc).__name__}: {exc}")
                _threw.append(fold_i)
                _drop_fold()
                continue
            if X_tr is None:
                _drop_fold()
                continue

            if pre_scaled:
                X_tr_s, X_va_s = X_tr, X_va
            else:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

            model = _make_knn(len(y_train))
            model.fit(X_tr_s, y_train)
            y_pred = model.predict(X_va_s)

            # Overall
            _score_all = float(metric_fn(y_val, y_pred))
            fold_all.append(_score_all)
            fold_scores_pos.append(_score_all)
            if not is_clf:
                fold_rmse_all.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

            if len(val_cliff) > 1:
                _score_cliff = float(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                fold_cliff.append(_score_cliff)
                fold_cliff_pos.append(_score_cliff)
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            else:
                fold_cliff_pos.append(float('nan'))
            if len(val_noncliff) > 1:
                _score_noncliff = float(metric_fn(y_val[val_noncliff],
                                                  y_pred[val_noncliff]))
                fold_noncliff.append(_score_noncliff)
                fold_noncliff_pos.append(_score_noncliff)
                if not is_clf:
                    fold_rmse_noncliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_noncliff], y_pred[val_noncliff]))))
            else:
                fold_noncliff_pos.append(float('nan'))
            _vcs = _cliff_seen_pos.get(fold_i, np.zeros(0, dtype=int))
            if len(_vcs) > 1:
                _score_seen = float(metric_fn(y_val[_vcs], y_pred[_vcs]))
                fold_cliff_seen.append(_score_seen)
                fold_cliff_seen_pos.append(_score_seen)
            else:
                fold_cliff_seen_pos.append(float('nan'))

        if not fold_all:
            return None
        out = {
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
            # Positional overall per-fold scores for the fold_reselection block.
            'fold_scores': [float(s) for s in fold_scores_pos],
            # The same folds, scored on the cliff rows alone and on the
            # non-cliff rows alone. The cliff list is what the fold_reselection
            # block below is fed, so its held-out re-pick answers the cliff
            # question rather than the overall one.
            'fold_scores_cliff': [float(s) for s in fold_cliff_pos],
            'fold_scores_noncliff': [float(s) for s in fold_noncliff_pos],
            'n_cliff_rows_by_fold': n_cliff_rows_pos,
            'n_noncliff_rows_by_fold': n_noncliff_rows_pos,
            'n_folds_scored_cliff': int(len(fold_cliff)),
            'n_folds_scored_noncliff': int(len(fold_noncliff)),
            # The cliff rows whose cliff partner is in the fold's training
            # rows; see results['cliff_partner_split'].
            'score_cliff_partner_in_train': (float(np.mean(fold_cliff_seen))
                                             if fold_cliff_seen
                                             else float('nan')),
            'fold_scores_cliff_partner_in_train': [
                float(v) for v in fold_cliff_seen_pos],
        }
        if not is_clf:
            out['rmse_all'] = float(np.mean(fold_rmse_all))
            out['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
            out['rmse_noncliff'] = float(np.mean(fold_rmse_noncliff)) if fold_rmse_noncliff else float('nan')
        if ck_name is not None and not _threw:
            _ck.put(out, condition=ck_name)
        return out

    # --- Precompute per-fold feature importance per rep (hoisted from lambda) ---
    # Keyed by fold number. It was keyed by (first training row, last training
    # row, number of training rows) until 2026-09-28. Two folds can share all
    # three: row 0 and the last row both held out by a third fold, and equal
    # training sizes. The later fold's importance, fitted on rows the earlier
    # fold holds out, then weighted the earlier fold's distance. Not hit on
    # herg_ki or caco2 at 3 folds; nothing prevented it.
    fold_importance = {}  # {rep_name: {fold_i: imp_array}}
    for rep_name, X_rep in reps.items():
        fold_importance[rep_name] = {}
        for _fi_imp, (tr, _va) in enumerate(fold_splits):
            try:
                imp = compute_feature_importance(
                    {rep_name: X_rep[tr]}, labels[tr], method='lightgbm_gain')
                fold_importance[rep_name][_fi_imp] = imp[rep_name]
            except Exception as exc:
                # N9: log so a None-importance fold reads as "compute failed"
                # rather than silently weakening the weighted kNN.
                print(f"      importance fold failed (rep={rep_name}): "
                      f"{type(exc).__name__}: {exc}")
                fold_importance[rep_name][_fi_imp] = None

    # --- Per-rep kNN baselines (unweighted + weighted) ---
    print(f"\n  Per-rep kNN baselines (k={knn_k}, metric={metric_name}):")
    for rep_name, X_rep in reps.items():
        # Unweighted
        def _get_rep(tr, va, yt, fold_i, _X=X_rep):
            return _X[tr], _X[va]
        res = _evaluate_condition(_get_rep, ck_name=f'per_rep|{rep_name}|uw')
        if res:
            results['per_rep'][rep_name] = res
            # Plain single-rep kNN is a baseline arm for fold_reselection (model=knn).
            # The cliff-row metric, not the overall one: this block's held-out
            # re-pick is the only place s04's cliff question is answered.
            reselected_baselines.setdefault('knn', {})[rep_name] = res['fold_scores_cliff']

            # Weighted: importance-weighted distance on this rep (precomputed per fold)
            def _get_rep_w(tr, va, yt, fold_i, _X=X_rep, _rname=rep_name):
                imp_arr = fold_importance[_rname].get(fold_i)
                if imp_arr is None:
                    return None, None
                w = np.maximum(imp_arr, 0) ** weight_exponent
                scaler = StandardScaler()
                return scaler.fit_transform(_X[tr]) * w, scaler.transform(_X[va]) * w
            res_w = _evaluate_condition(_get_rep_w, pre_scaled=True,
                                        ck_name=f'per_rep|{rep_name}|w')
            if res_w:
                results['per_rep'][rep_name]['score_cliff_weighted'] = res_w['score_cliff']
                results['per_rep'][rep_name]['score_all_weighted'] = res_w['score_all']
                results['per_rep'][rep_name]['fold_scores_cliff_weighted'] = \
                    res_w['fold_scores_cliff']
                results['per_rep'][rep_name]['fold_scores_noncliff_weighted'] = \
                    res_w['fold_scores_noncliff']
                if not is_clf:
                    results['per_rep'][rep_name]['rmse_cliff_weighted'] = res_w.get('rmse_cliff', float('nan'))
                # Importance-weighted single-rep kNN is still a single-rep baseline
                # (no cross-rep mixing) — record as a distinct baseline entry.
                reselected_baselines.setdefault('knn', {})[f'{rep_name}|weighted'] = res_w['fold_scores_cliff']

            rmse_part = f"  RMSE_cliff={res.get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
            w_part = ""
            if res_w and not is_clf:
                w_part = f"  | weighted: RMSE_cliff={res_w.get('rmse_cliff', float('nan')):.4f}"
            print(f"    {rep_name:<15} {metric_name}_all={res['score_all']:.4f}  "
                  f"{metric_name}_cliff={res['score_cliff']:.4f}{rmse_part}  "
                  f"{metric_name}_noncliff={res['score_noncliff']:.4f}"
                  + w_part)

    # --- The pool's best single representation, re-picked per fold -----------
    # 2026-09-12 decision: the pool supplies the best single representation as
    # well as the hybrids, chosen per fold from the other folds' scores. Which
    # representation that is can differ from fold to fold, so this is one arm
    # whose identity moves, not one of the fixed per-rep arms above.
    if knn_pools:
        _pool_base_by_fold = {
            _fi: (knn_pools[_fi]['base_rep'] or {}).get('rep')
            for _fi in knn_pools}

        def _get_pool_base(tr, va, yt, fold_i):
            _rname = _pool_base_by_fold.get(fold_i)
            if _rname is None or _rname not in reps:
                return None, None
            return reps[_rname][tr], reps[_rname][va]

        res_pb = _evaluate_condition(_get_pool_base, ck_name='pool_base|uw')
        res_pb_w = None
        if res_pb:
            def _get_pool_base_w(tr, va, yt, fold_i):
                _rname = _pool_base_by_fold.get(fold_i)
                if _rname is None or _rname not in reps:
                    return None, None
                imp_arr = fold_importance[_rname].get(fold_i)
                if imp_arr is None:
                    return None, None
                w = np.maximum(imp_arr, 0) ** weight_exponent
                _X = reps[_rname]
                scaler = StandardScaler()
                return (scaler.fit_transform(_X[tr]) * w,
                        scaler.transform(_X[va]) * w)

            res_pb_w = _evaluate_condition(_get_pool_base_w, pre_scaled=True,
                                           ck_name='pool_base|w')
            results['pool_base_rep'] = {
                'reps_by_fold': {str(_fi): _pool_base_by_fold.get(_fi)
                                 for _fi in sorted(_pool_base_by_fold)},
                'unweighted': res_pb,
                'weighted': res_pb_w,
                'selection_scope': 'other_folds_of_pool_search',
            }
            # A single representation, no cross-rep mixing, so it is a baseline
            # arm — one whose name says its identity was re-picked per fold.
            reselected_baselines.setdefault('knn', {})['pool_base_rep|per_fold'] = \
                res_pb['fold_scores_cliff']
            if res_pb_w:
                reselected_baselines['knn']['pool_base_rep|per_fold|weighted'] = \
                    res_pb_w['fold_scores_cliff']
            _names = sorted({v for v in _pool_base_by_fold.values() if v})
            rmse_part = (f"  RMSE_cliff={res_pb.get('rmse_cliff', float('nan')):.4f}"
                         if not is_clf else "")
            print(f"    pool_base({'/'.join(_names) if _names else 'none'}) "
                  f"{metric_name}_all={res_pb['score_all']:.4f}  "
                  f"{metric_name}_cliff={res_pb['score_cliff']:.4f}{rmse_part}")
        else:
            results['pool_base_rep'] = {
                'reps_by_fold': {str(_fi): _pool_base_by_fold.get(_fi)
                                 for _fi in sorted(_pool_base_by_fold)},
                'computed': False,
                'reason': ('the pool named no single representation for any '
                           'fold, or every cliff-CV fold was dropped'),
            }

    # --- Hybrid kNN, one arm per (family, rank) slot the pool returned ---
    if hybrid_blueprint_available:
        print(f"\n  Hybrid kNN (k={knn_k}), {len(hybrid_arms)} pool arms "
              f"(subset {_POOL_SUBSET}, top {_N_POOL_HYBRIDS} per family):")

        def _arm_entry(fold_i, objective, rank):
            """The pool's hybrid for this fold in the (family, rank) slot."""
            pool = knn_pools.get(fold_i)
            if pool is None:
                return None
            for h in pool['hybrids']:
                if h['objective'] == objective and h['rank'] == rank:
                    return h
            return None

        # Which representations a fold's hybrid keeps when the recipe uses
        # fewer than the job loads. Added 2026-09-28, same fix as c13, c15 and
        # c22. Without scores passed in, build_fold_hybrid keeps the best
        # rep_count by blueprint.rep_scores, which the search measured over
        # every row (find_best_hybrid.py:9492-9522), this fold's held-out rows
        # included. Under the local run logd loads 6 representations and
        # rep_count is 5, so the held-out labels chose which one each fold's
        # hybrid dropped. This scores each representation the way the search
        # and create_hybrid do: the recipe's internal evaluator, fitted on the
        # internal carve of the fold's training rows, read off with the
        # dataset's metric (read-across for the similarity family). None when
        # nothing is cut, so herg_ki and caco2 build exactly as before.
        _alloc_cache = {}

        def _fold_alloc_scores(fold_i, train_reps_h, yt, groups_tr, bp):
            rep_count = getattr(bp, 'rep_count', None)
            if rep_count is None or int(rep_count) >= len(train_reps_h):
                return None
            evaluator = getattr(bp, 'internal_evaluator', None)
            objective = getattr(bp, 'objective', None)
            from kirby.find_best_hybrid import (OBJECTIVE_PREDICTIVE,
                                                OBJECTIVE_SIMILARITY,
                                                _is_geometry_objective)
            # The similarity family is scored the way create_hybrid scores it:
            # read-across on the internal carve, which needs that carve's
            # scaffold groups. A blend or any other geometry objective is left
            # on blueprint.rep_scores and says so.
            _sim = (objective == OBJECTIVE_SIMILARITY)
            if (not isinstance(evaluator, str) or evaluator == 'match'
                    or (_sim and groups_tr is None)
                    or (objective is not None and not _sim
                        and (isinstance(objective, dict)
                             or _is_geometry_objective(objective)))):
                print(f"      [s04] fold-local representation ranking not "
                      f"computed (evaluator={evaluator!r}, "
                      f"objective={objective!r}); build_fold_hybrid ranks on "
                      f"the whole-dataset rep_scores")
                return None
            key = (int(fold_i), evaluator, str(objective))
            if key in _alloc_cache:
                return _alloc_cache[key]
            from kirby.hybrid import _score_rep_for_budget, internal_holdout_split
            # The direction of metric_fn, which is what build_fold_hybrid hands
            # create_hybrid for this recipe, so the scores match the ones
            # create_hybrid would compute on the kept representations.
            lib = bool(lower_is_better)
            inner_tr, inner_va = internal_holdout_split(len(yt), yt, groups_tr)
            out = {}
            for name in sorted(train_reps_h):
                X = train_reps_h[name]
                Xtr = np.nan_to_num(np.clip(X[inner_tr], -1e10, 1e10),
                                    nan=0.0, posinf=1e10, neginf=-1e10)
                Xva = np.nan_to_num(np.clip(X[inner_va], -1e10, 1e10),
                                    nan=0.0, posinf=1e10, neginf=-1e10)
                if _sim:
                    out[name] = float(_score_rep_for_budget(
                        Xtr, Xva, yt[inner_tr], yt[inner_va],
                        internal_evaluator=evaluator,
                        objective=objective, is_clf=is_clf,
                        groups_val=np.asarray(groups_tr)[inner_va]))
                    continue
                out[name] = float(_score_rep_for_budget(
                    Xtr, Xva, yt[inner_tr], yt[inner_va],
                    internal_evaluator=evaluator, metric_fn=metric_fn,
                    lower_is_better=lib,
                    objective=objective or OBJECTIVE_PREDICTIVE,
                    is_clf=is_clf))
            _alloc_cache[key] = out
            return out

        def _build_arm(tr, va, yt, fold_i, objective, rank):
            """(X_train, X_val, feature_info, blueprint) for one arm, one fold."""
            h = _arm_entry(fold_i, objective, rank)
            if h is None or h.get('blueprint') is None:
                return None, None, None, None
            bp_fold = h['blueprint']   # augmentation_strategy already set by the pool
            aug_cond = h['aug_condition']
            train_reps_h = {name: rep[tr] for name, rep in reps.items()}
            val_reps_h = {name: rep[va] for name, rep in reps.items()}
            _use_aug = augmentations if aug_cond != 'none' else None
            aug_tr = {k: v[tr] for k, v in _use_aug.items()} if _use_aug else None
            aug_va = {k: v[va] for k, v in _use_aug.items()} if _use_aug else None
            _g_tr = groups[tr] if groups is not None else None
            X_tr_h, X_va_h, feat_info = build_fold_hybrid(
                train_reps_h, val_reps_h, yt, bp_fold,
                augmentations_train=aug_tr, augmentations_val=aug_va,
                groups=_g_tr,
                precomputed_alloc_scores=_fold_alloc_scores(
                    fold_i, train_reps_h, yt, _g_tr, bp_fold),
                **fold_hybrid_scoring_kwargs(
                    bp_fold, model=_make_knn(len(yt)),
                    metric_fn=metric_fn, lower_is_better=lower_is_better))
            return X_tr_h, X_va_h, feat_info, bp_fold

        results['pool_hybrid_arms'] = {}
        for _obj, _rank in hybrid_arms:
            _arm = f'{_obj}|rank{_rank}'

            def _get_arm_uw(tr, va, yt, fold_i, _o=_obj, _r=_rank):
                X_tr_h, X_va_h, _fi, _bp = _build_arm(tr, va, yt, fold_i, _o, _r)
                if X_tr_h is None:
                    return None, None
                return X_tr_h, X_va_h

            def _get_arm_w(tr, va, yt, fold_i, _o=_obj, _r=_rank):
                X_tr_h, X_va_h, feat_info, bp_fold = _build_arm(
                    tr, va, yt, fold_i, _o, _r)
                if X_tr_h is None:
                    return None, None
                train_reps_h = {name: rep[tr] for name, rep in reps.items()}
                # Same importance method the search used to build this hybrid.
                imp = compute_feature_importance(
                    train_reps_h, yt,
                    method=getattr(bp_fold, 'importance_method', 'lightgbm_gain'))
                w_parts = []
                for rname in feat_info:
                    if not isinstance(feat_info[rname], dict):
                        continue
                    if 'selected_indices' not in feat_info[rname]:
                        continue
                    idx = feat_info[rname]['selected_indices']
                    if rname in imp:
                        w_parts.append(imp[rname][idx])
                    else:
                        w_parts.append(np.ones(len(idx)))
                w = np.maximum(np.concatenate(w_parts), 0) ** weight_exponent
                # Scale-then-weight (matches Exp 5 and compute_distance_matrix
                # convention)
                scaler_w = StandardScaler()
                return (scaler_w.fit_transform(X_tr_h) * w,
                        scaler_w.transform(X_va_h) * w)

            res = _evaluate_condition(_get_arm_uw, ck_name=f'arm|{_arm}|uw')
            res_w = _evaluate_condition(_get_arm_w, pre_scaled=True,
                                        ck_name=f'arm|{_arm}|w')

            _entry = {
                'objective': _obj,
                'rank': _rank,
                # The recipe in this slot is re-picked per fold, so store the
                # one each fold used rather than one name for the arm.
                'recipe_by_fold': {
                    str(_fi): ((_arm_entry(_fi, _obj, _rank) or {}).get('name'))
                    for _fi in sorted(knn_pools)},
                'score_other_folds_by_fold': {
                    str(_fi): ((_arm_entry(_fi, _obj, _rank) or {})
                               .get('score_other_folds'))
                    for _fi in sorted(knn_pools)},
                'pool_metric_name': next(
                    ((_arm_entry(_fi, _obj, _rank) or {}).get('metric_name')
                     for _fi in sorted(knn_pools)
                     if _arm_entry(_fi, _obj, _rank) is not None), None),
                'unweighted': res,
                'weighted': res_w,
            }
            if res is None:
                _entry['unweighted_status'] = {
                    'computed': False,
                    'reason': ('no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'),
                }
            if res_w is None:
                _entry['weighted_status'] = {
                    'computed': False,
                    'reason': ('no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'),
                }
            results['pool_hybrid_arms'][_arm] = _entry

            # Each arm is its own candidate for fold_reselection (model=knn).
            if res:
                reselection_candidates.setdefault('knn', []).append(
                    {'name': f'hybrid_{_obj}_rank{_rank}_unweighted',
                     'fold_scores': res['fold_scores_cliff']})
            if res_w:
                reselection_candidates.setdefault('knn', []).append(
                    {'name': f'hybrid_{_obj}_rank{_rank}_weighted',
                     'fold_scores': res_w['fold_scores_cliff']})

            for _label, _r in (('uw', res), ('w', res_w)):
                if not _r:
                    print(f"    {_arm}_{_label:<2} not computed: all cliff-CV "
                          f"folds dropped")
                    continue
                rmse_part = (f"  RMSE_cliff={_r.get('rmse_cliff', float('nan')):.4f}"
                             if not is_clf else "")
                print(f"    {_arm}_{_label:<2} {metric_name}_all={_r['score_all']:.4f}  "
                      f"{metric_name}_cliff={_r['score_cliff']:.4f}{rmse_part}  "
                      f"{metric_name}_noncliff={_r['score_noncliff']:.4f}")

        # results['hybrid']['unweighted'] / ['weighted'] are what s09 and the
        # first-pass reader read. They now hold the FIRST arm in subset order —
        # the predictive family's rank-1 hybrid where that family is present —
        # so those readers keep getting one hybrid number. Every arm is in
        # results['pool_hybrid_arms'].
        _lead = f'{hybrid_arms[0][0]}|rank{hybrid_arms[0][1]}' if hybrid_arms else None
        _lead_entry = results['pool_hybrid_arms'].get(_lead, {})
        results['hybrid']['lead_arm'] = _lead
        for _cond in ('unweighted', 'weighted'):
            if _lead_entry.get(_cond):
                results['hybrid'][_cond] = _lead_entry[_cond]
            else:
                # F17: every fold of this condition was dropped, so the key is
                # absent. Name that instead of leaving it simply missing.
                results['hybrid'][f'{_cond}_status'] = _lead_entry.get(
                    f'{_cond}_status',
                    {'computed': False,
                     'reason': 'no cliff-CV fold produced a score (all folds '
                               'skipped or threw)'})
    else:
        # F17: no recipe means NEITHER hybrid cliff score is attempted. Record
        # both absences and which of the two causes applies — the pool read
        # threw, or it returned no kNN hybrid configuration for any fold.
        _st = results.get('hybrid_search_status', {})
        if _st.get('error_type'):
            _reason = (f"pool read raised "
                       f"{_st.get('error_type')}: {_st.get('error', '')}")
        else:
            _reason = (_st.get('error')
                       or 'the pool held no kNN hybrid configuration')
        for _cond in ('unweighted', 'weighted'):
            results['hybrid'][f'{_cond}_status'] = {
                'computed': False,
                'reason': _reason,
            }
        print(f"\n  Hybrid kNN: not computed ({_reason})")

    # --- RF baseline (best single representation for RF, per fold) ----------
    # 2026-09-12: the representation came from fold 0's own search, which no
    # longer runs. It now comes from the same pool, read for downstream_model
    # 'rf' and ranked on each fold's OTHER folds, so it can differ per fold.
    # Where the pool names none, the fallback is unchanged: the best-scoring
    # representation under the kNN baselines above.
    print("\n  RF baseline (per-rep, best rep for RF):")
    fallback_rep_name = None
    if results['per_rep']:
        fallback_rep_name = (min if lower_is_better else max)(
            results['per_rep'], key=lambda r: results['per_rep'][r]['score_all']
        )
    # The fallback is chosen per fold on the OTHER folds' kNN scores. It was
    # the representation with the best kNN score_all, a mean that includes the
    # fold being predicted, so where the pool named no representation the
    # held-out fold's own labels chose the RF's input (fixed 2026-09-28).
    # fallback_rep_name above is kept only for the printed name.
    def _fallback_for_fold(fi):
        best, best_m = None, None
        for r, rv in results['per_rep'].items():
            fs = np.asarray(rv.get('fold_scores') or [], dtype=float)
            other = np.array([s for j, s in enumerate(fs) if j != fi
                              and np.isfinite(s)], dtype=float)
            if other.size == 0:
                continue
            m = float(other.mean())
            if best is None or (m < best_m if lower_is_better else m > best_m):
                best, best_m = r, m
        return best
    rf_rep_by_fold = {}
    rf_rep_source_by_fold = {}
    for _fi in range(len(fold_splits)):
        _rep = None
        try:
            _rep = (_fold_pool(_fi, 'rf')['base_rep'] or {}).get('rep')
        except Exception as exc:
            print(f"      rf pool read failed on fold {_fi}: "
                  f"{type(exc).__name__}: {exc}")
        _src = 'pool_other_folds'
        if _rep is None or _rep not in reps:
            _rep = _fallback_for_fold(_fi)
            _src = 'knn_other_folds_fallback'
        rf_rep_by_fold[_fi] = _rep
        rf_rep_source_by_fold[_fi] = _src
    _rf_reps = sorted({r for r in rf_rep_by_fold.values() if r})
    best_rep_name = (rf_rep_by_fold.get(0) or fallback_rep_name)
    if best_rep_name:
        rf_factory = get_model_factories(['rf'], task_type=task_type,
                                          n_samples=len(labels))['rf']

        fold_all, fold_cliff, fold_noncliff = [], [], []
        fold_rmse_cliff = []
        # Same positional convention as the kNN arms: one slot per split, NaN
        # where the held-out fold carried 1 or fewer rows of the stratum.
        fold_cliff_pos, fold_noncliff_pos = [], []
        n_cliff_rows_pos, n_noncliff_rows_pos = [], []
        fold_cliff_seen, fold_cliff_seen_pos = [], []
        for fold_i, (train_idx, val_idx) in enumerate(fold_splits):
            y_train, y_val = labels[train_idx], labels[val_idx]
            _rep = rf_rep_by_fold.get(fold_i) or best_rep_name
            X_best = reps[_rep]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_best[train_idx])
            X_va = scaler.transform(X_best[val_idx])
            rf = rf_factory()
            rf.fit(X_tr, y_train)
            y_pred = rf.predict(X_va)

            fold_all.append(metric_fn(y_val, y_pred))
            val_cliff = np.array([i for i, vi in enumerate(val_idx)
                                  if vi in cliff_indices], dtype=int)
            val_noncliff = np.array([i for i, vi in enumerate(val_idx)
                                     if vi not in cliff_indices], dtype=int)
            n_cliff_rows_pos.append(int(len(val_cliff)))
            n_noncliff_rows_pos.append(int(len(val_noncliff)))
            if len(val_cliff) > 1:
                _score_cliff = float(metric_fn(y_val[val_cliff], y_pred[val_cliff]))
                fold_cliff.append(_score_cliff)
                fold_cliff_pos.append(_score_cliff)
                if not is_clf:
                    fold_rmse_cliff.append(float(np.sqrt(mean_squared_error(
                        y_val[val_cliff], y_pred[val_cliff]))))
            else:
                fold_cliff_pos.append(float('nan'))
            if len(val_noncliff) > 1:
                _score_noncliff = float(metric_fn(y_val[val_noncliff],
                                                  y_pred[val_noncliff]))
                fold_noncliff.append(_score_noncliff)
                fold_noncliff_pos.append(_score_noncliff)
            else:
                fold_noncliff_pos.append(float('nan'))
            _vcs = _cliff_seen_pos.get(fold_i, np.zeros(0, dtype=int))
            if len(_vcs) > 1:
                _score_seen = float(metric_fn(y_val[_vcs], y_pred[_vcs]))
                fold_cliff_seen.append(_score_seen)
                fold_cliff_seen_pos.append(_score_seen)
            else:
                fold_cliff_seen_pos.append(float('nan'))

        results['rf_baseline'] = {
            'rep': best_rep_name,
            # Which representation each fold actually fitted. 'rep' above is
            # fold 0's, kept as a plain name for the readers that print it.
            'reps_by_fold': {str(_fi): rf_rep_by_fold.get(_fi)
                             for _fi in sorted(rf_rep_by_fold)},
            'rep_selection': 'other_folds_of_pool_search',
            # Per fold: 'pool_other_folds', or 'knn_other_folds_fallback' where
            # the pool named no representation for rf.
            'rep_source_by_fold': {str(_fi): rf_rep_source_by_fold.get(_fi)
                                   for _fi in sorted(rf_rep_source_by_fold)},
            'score_all': float(np.mean(fold_all)),
            'score_cliff': float(np.mean(fold_cliff)) if fold_cliff else float('nan'),
            'score_noncliff': float(np.mean(fold_noncliff)) if fold_noncliff else float('nan'),
            'metric': metric_name,
            'fold_scores': [float(s) for s in fold_all],
            'fold_scores_cliff': [float(s) for s in fold_cliff_pos],
            'fold_scores_noncliff': [float(s) for s in fold_noncliff_pos],
            'n_cliff_rows_by_fold': n_cliff_rows_pos,
            'n_noncliff_rows_by_fold': n_noncliff_rows_pos,
            'n_folds_scored_cliff': int(len(fold_cliff)),
            'n_folds_scored_noncliff': int(len(fold_noncliff)),
            'score_cliff_partner_in_train': (float(np.mean(fold_cliff_seen))
                                             if fold_cliff_seen
                                             else float('nan')),
            'fold_scores_cliff_partner_in_train': [
                float(v) for v in fold_cliff_seen_pos],
        }
        # RF is a separate downstream model — record its per-fold single-rep
        # baseline under model=rf. No RF hybrid candidate exists here, so the
        # honest delta for 'rf' is NaN (hybrid arm empty); the helper tolerates it.
        # The name says the representation was re-picked per fold when the folds
        # did not all land on one. The cliff-row metric, same as the kNN arms.
        _rf_key = (best_rep_name if len(_rf_reps) <= 1
                   else 'pool_base_rep|per_fold')
        reselected_baselines.setdefault('rf', {})[_rf_key] = [
            float(s) for s in fold_cliff_pos]
        if not is_clf:
            results['rf_baseline']['rmse_cliff'] = float(np.mean(fold_rmse_cliff)) if fold_rmse_cliff else float('nan')
        rmse_part = f"  RMSE_cliff={results['rf_baseline'].get('rmse_cliff', float('nan')):.4f}" if not is_clf else ""
        print(f"    RF({'/'.join(_rf_reps) if _rf_reps else best_rep_name}) "
              f"{metric_name}_all={results['rf_baseline']['score_all']:.4f}  "
              f"{metric_name}_cliff={results['rf_baseline']['score_cliff']:.4f}"
              f"{rmse_part}")

    # --- Summary ---
    print(f"\n  --- Summary ({metric_name}_cliff comparison) ---")
    all_methods = []
    for rname, rv in results['per_rep'].items():
        all_methods.append((f"kNN({rname})", rv.get('score_cliff', float('nan'))))
        if 'score_cliff_weighted' in rv:
            all_methods.append((f"kNN({rname})_w", rv.get('score_cliff_weighted', float('nan'))))
    # The pool's own single representation, re-picked per fold.
    _pb = results.get('pool_base_rep') or {}
    if _pb.get('unweighted'):
        all_methods.append(("kNN(pool_base_rep)",
                            _pb['unweighted'].get('score_cliff', float('nan'))))
    if _pb.get('weighted'):
        all_methods.append(("kNN(pool_base_rep)_w",
                            _pb['weighted'].get('score_cliff', float('nan'))))
    # One line per pool arm, both weightings. The lead arm also appears under
    # results['hybrid'] for the readers that expect one hybrid there.
    for _arm, _entry in (results.get('pool_hybrid_arms') or {}).items():
        if _entry.get('unweighted'):
            all_methods.append((f"hybrid_kNN_uw[{_arm}]",
                                _entry['unweighted'].get('score_cliff', float('nan'))))
        if _entry.get('weighted'):
            all_methods.append((f"hybrid_kNN_w[{_arm}]",
                                _entry['weighted'].get('score_cliff', float('nan'))))
    if 'rf_baseline' in results:
        rv = results['rf_baseline']
        all_methods.append((f"RF({rv['rep']})", rv.get('score_cliff', float('nan'))))

    _nan_sentinel = np.inf if lower_is_better else -np.inf
    all_methods.sort(key=lambda x: x[1] if not np.isnan(x[1]) else _nan_sentinel,
                     reverse=not lower_is_better)
    print(f"  {'Method':<25} {metric_name + '_cliff':<12}")
    print("  " + "-" * 37)
    for name, score in all_methods:
        print(f"  {name:<25} {score:<12.4f}")

    # --- Per-fold reselection block (CONTRACT json_convention) ---
    # Reader: fold_reselection(block['candidates_by_model'],
    #                        block['baselines_by_model'], block['lower_is_better']).
    # 2026-09-12: the recipes are the pool's now, so both of these change.
    # exclude_folds: where a pool search pruned mid-flight it filtered its own
    # candidate list using fold 0's scores, so fold 0 contributes no held-out
    # term — the same reading hr_exclude_folds takes, and [] only when the pool
    # says outright that it did not prune.
    # recipe_selection: each fold's recipe was ranked on the other folds'
    # scores of a search that itself ran over every row. That is neither
    # 'whole_dataset' nor 'fold_train_rows'.
    # fold_scores: the metric over the fold's CLIFF rows alone, not over the
    # whole held-out fold, so the leave-one-fold-out re-pick answers the cliff
    # question. A fold whose held-out rows held 1 or fewer cliff molecules is
    # NaN in every arm and contributes nothing; the counts are on each arm as
    # 'n_cliff_rows_by_fold'.
    results['fold_reselection'] = {
        'lower_is_better': bool(lower_is_better),
        # The metric's name as well as its direction, so hr_direction can check
        # one against the other instead of trusting the flag alone.
        'metric_name': metric_name,
        'fold_score_stratum': 'cliff_rows_only',
        'exclude_folds': ([] if fold0_pruned is False else [0]),
        'fold0_pruned': (None if fold0_pruned is None else bool(fold0_pruned)),
        'recipe_selection': 'other_folds_of_pool_search',
        'recipe_selection_note': (
            "recipe and single representation ranked by their mean over the "
            "pool search's other folds; the pool search ran over every row, so "
            "this holds out the fold's SCORES, not its rows"),
        'candidates_by_model': reselection_candidates,
        'baselines_by_model': reselected_baselines,
    }

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('S04: Activity Cliffs',
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
    save_experiment_results(results, args.output, 's04_activity_cliffs', args.dataset)
