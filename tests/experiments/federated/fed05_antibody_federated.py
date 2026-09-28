#!/usr/bin/env python3
"""Fed05: Antibody Federated Extension.

Adapts FLuID's federated framework from small molecules to antibody sequences.
Uses ESM2 embedding clustering instead of Morgan FP clustering. Tests the same
core questions as fed01/02 in the antibody domain: does KIRBy's feature-level
fusion outperform prediction-level ensemble across heterogeneous antibody silos?

3 natural silos:
    A: Experimental DDG (ddg_combined, ~4500 mutations, kcal/mol)
    B: DMS binding (abagym_EGFR, 659 mutations, normalized score)
    C: Developability (flab_thermo_tresanco, Tm °C)

8 conditions:
    C1: Independent baselines per silo
    C2: Cross-silo importance transfer
    C3: Within-silo ESM2 clustering → federated feature selection
    C4: Noise heterogeneity (Gaussian noise on one silo)
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

Usage:
    python fed05_antibody_federated.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import traceback
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import zscore

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy, cv_split,
    cap_dimensions, get_domain_metric, FULL_BUDGETS,
)
from kirby.find_best_hybrid import BUILTIN_METRICS
from experiments.federated.federated_helpers import (
    compute_reg_metrics, compute_clf_metrics, inject_noise_per_silo,
    federated_feature_selection, importance_profile_transfer,
    extract_importance_profile, compare_importance_profiles,
    cluster_antibody_silos, federated_budgets,
    predict_fluid_quality_ensemble_regression,
    _resolve_match,
    fit_fed_classifier, fit_fed_regressor, fit_fed_model, fed_to_kirby_model,
    smoke_search_kwargs,
)
from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
from kirby.find_best_hybrid import find_best_hybrid


# ── Constants ────────────────────────────────────────────────────────────────

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

# Rep subsets for Condition 6 (rep diversity per silo)
REP_DIVERSITY_SILOS = {
    'structural': ['aac', 'saprot', 'esm2'],
    'sequence_lm': ['ablang2', 'antiberty', 'prot_t5'],
    'cdr_focused': ['prot_t5_cdr', 'esm2_midlayers', 'igt5'],
}

K_INTRA = 3   # sub-clusters per silo for Condition 3
NOISE_SIGMA = 0.2  # noise level for Condition 4


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
        # mutational scan breaks them. MEASURED on abagym_EGFR, 2026-09-18:
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
        # foundation reps are absent. No-op ('all') on the server, where the
        # foundation reps are the whole point of this study.
        from experiments.federated.federated_helpers import _smoke_active
        if _smoke_active():
            sd = load_data(
                ds_name,
                rep_names=['aac', 'paratope', 'onehot', 'blosum'])
        else:
            sd = load_data(ds_name, rep_preset='all')
        n = len(sd['labels'])
        print(f"    {silo_name} ({ds_name}): {n} samples, "
              f"{len(sd['reps'])} reps")
        return sd
    except Exception as e:
        print(f"    {silo_name} ({ds_name}): FAILED — {e}")
        if failures is not None:
            failures[silo_name] = f'{ds_name}: {e}'
        return None


def _split_silo(sd, seed=42):
    """80/20 group-aware split for one silo. Returns (train_idx, test_idx)."""
    groups = sd.get('groups')
    n = len(sd['labels'])
    return next(cv_split(groups, 1, np.zeros(n), sd['labels'], seed=seed))


def _silo_baseline(sd, shared_reps, seed=42,
                   alloc='performance_weighted', imp='treeshap',
                   int_eval='match', fed_model='rf', fed_seed=0,
                   split=None, budget=100):
    """Train create_hybrid on one silo and evaluate. Returns metrics + fi.

    Routes the downstream model + metric on the silo's task_type — P1-4
    fix: prior hardcoded RandomForestRegressor silently mis-handled clf
    silos.
    """
    # Defensive: resolve 'match' to a concrete evaluator ('rf') so the no-model
    # create_hybrid call below never falls through to 'ridge'
    # (hybrid.py:3354-3355, forbidden by feedback_no_ridge.md). run() already
    # passes a resolved int_eval; this guards future callers.
    int_eval = _resolve_match(int_eval, 'rf')

    reps_s = {r: sd['reps'][r] for r in shared_reps if r in sd['reps']}
    if len(reps_s) < 2:
        reps_s = sd['reps']

    # `split` is the one the caller drew before the recipe search. Drawing a
    # fresh one here would score the hybrid on rows the recipe search was
    # allowed to see.
    tr, te = split if split is not None else _split_silo(sd, seed)
    rep_tr = {r: arr[tr] for r, arr in reps_s.items()}
    rep_te = {r: arr[te] for r, arr in reps_s.items()}
    y_tr, y_te = sd['labels'][tr], sd['labels'][te]

    s_augs = sd.get('augmentations')
    s_aug_cond = 'uncapped' if s_augs else 'none'
    tr_augs = ({k: v[tr] for k, v in s_augs.items()} if s_augs else None)
    te_augs = ({k: v[te] for k, v in s_augs.items()} if s_augs else None)

    # ⛔ THE SILO'S SEQUENCE-IDENTITY GROUPS, WHICH THIS CALL DID NOT PASS.
    # Split audit kind B, fed04_cluster_aware_importance.py:89. create_hybrid
    # carves an internal holdout to decide which features are admitted;
    # internal_holdout_split (src/kirby/hybrid.py:5435) uses GroupKFold when it
    # is HANDED groups and a fixed-seed random carve when it is not. fed05
    # honours identity clusters in its 80/20 splits and dropped them here, so
    # features were admitted on a carve that could put two antibodies of one
    # cluster on opposite sides. No score already reported is contaminated:
    # this carve is inside the training rows and the test rows were cut
    # group-aware first. Fixed 2026-09-18.
    _sb_groups = sd.get('groups')
    # Each representation's share of the budget is sized on the silo's own
    # metric (spearman on all three antibody silos), the metric the recipe
    # search scored the same allocation with. With no metric_fn create_hybrid
    # sizes the shares on the internal evaluator's R^2 instead
    # (_score_rep_for_budget, src/kirby/hybrid.py), so the recipe the search
    # chose was rebuilt on a different quantity. C4 already passed spearman.
    _, _sb_mfn, _sb_lib = get_domain_metric(sd)
    X_h, fi = create_hybrid(
        rep_tr, y_tr,
        groups=(np.asarray(_sb_groups)[tr] if _sb_groups is not None else None),
        allocation_method=alloc,
        importance_method=imp,
        internal_evaluator=int_eval,
        metric_fn=_sb_mfn, lower_is_better=_sb_lib,
        budget=budget,
        augmentations=tr_augs,
        augmentation_strategy=s_aug_cond)
    X_h_te = apply_feature_selection(rep_te, fi)
    _aug_info_b = fi.get('augmentation_info') if isinstance(fi, dict) else None
    if _aug_info_b and te_augs:
        _X_te_aug_b = apply_augmentation_selection(te_augs, _aug_info_b)
        if _X_te_aug_b is not None:
            X_h_te = np.hstack([X_h_te, _X_te_aug_b])

    if sd.get('task_type') == 'classification':
        clf = fit_fed_classifier(fed_model, X_h, y_tr, seed=fed_seed)
        y_pred = clf.predict(X_h_te)
        y_prob = (clf.predict_proba(X_h_te)
                  if hasattr(clf, 'predict_proba') else None)
        metrics = compute_clf_metrics(y_te, y_pred, y_prob)
    else:
        reg = fit_fed_regressor(fed_model, X_h, y_tr, seed=fed_seed)
        y_pred = reg.predict(X_h_te)
        metrics = compute_reg_metrics(y_te, y_pred)
    metrics['n_features'] = int(X_h.shape[1])
    return metrics, fi, (tr, te)


def _run_one(data, config, output_dir, dataset_name, **kwargs):
    """Fed05: Antibody federated extension."""
    print("\n" + "=" * 70)
    print("EXPERIMENT FED05: ANTIBODY FEDERATED EXTENSION")
    print("=" * 70)

    fed_model = config.get('fed_model', 'rf')
    fed_seed = config.get('fed_seed', 0)

    # ── Phase 1: Load silos ──────────────────────────────────────────────
    print("\n--- Phase 1: Load silos ---")
    silo_data = {}
    silo_load_failures = {}
    for silo_name, ds_name in SILO_DATASETS.items():
        sd = _load_silo(silo_name, ds_name, silo_load_failures)
        if sd is not None:
            silo_data[silo_name] = sd

    if silo_load_failures:
        # The saved result names three silos. Without this a run on two of them
        # reads afterwards as a three-silo result.
        print(f"  WARNING: {len(silo_load_failures)} silo(s) did not load: "
              f"{', '.join(sorted(silo_load_failures))}")

    if len(silo_data) < 2:
        return {'error': f'Only {len(silo_data)} silos loaded, need >= 2',
                'silo_load_failures': silo_load_failures}

    # Every silo's 80/20 split is drawn ONCE, here, before anything is fitted
    # or searched. Everything downstream reuses it: the recipe search below,
    # C1, C4's pooled rows, C5's pooled rows and C6. Until 2026-09-04 the
    # search ran on all of a silo's rows and the split was drawn afterwards
    # inside _silo_baseline, so the allocation method, importance method and
    # evaluator every condition uses were picked with the test rows in view —
    # while the quality-weighted ensemble C6 is compared against was tuned on
    # nothing at all.
    silo_splits = {}
    for sname, sd in list(silo_data.items()):
        try:
            silo_splits[sname] = _split_silo(sd, fed_seed)
        except Exception as e:
            print(f"  {sname}: cannot split — {e}")
            silo_load_failures[sname] = f'split failed: {e}'
            silo_data.pop(sname)

    if len(silo_data) < 2:
        return {'error': f'Only {len(silo_data)} silos usable, need >= 2',
                'silo_load_failures': silo_load_failures}

    # Discover optimal hybrid config per silo via find_best_hybrid, on that
    # silo's TRAINING rows only.
    silo_blueprints = {}
    for sname, sd in silo_data.items():
        if len(sd['reps']) >= 2:
            try:
                _metric = get_domain_metric(sd)[0]
                _silo_ds_name = SILO_DATASETS.get(sname, sname)
                s_tr, _ = silo_splits[sname]
                s_groups = sd.get('groups')
                s_search = find_best_hybrid(
                    {r: arr[s_tr] for r, arr in sd['reps'].items()},
                    sd['labels'][s_tr],
                    groups=(np.asarray(s_groups)[s_tr]
                            if s_groups is not None else None),
                    budgets=FULL_BUDGETS,
                    dataset_name=_silo_ds_name, tier='general',
                    metric=_metric, task_type=sd['task_type'],
                    adaptive_pruning=0.25,
                    augmentation_strategy='none',
                    # Each search here covers up to 3,000 recipes on one
                    # silo's training rows. Without this every re-run pays
                    # the whole bill again.
                    cache=True,
                    **smoke_search_kwargs(config))
                s_bp_entry = s_search.best_by_model.get(
                    fed_to_kirby_model(fed_model))
                if s_bp_entry:
                    silo_blueprints[sname] = s_bp_entry['blueprint']
            except Exception:
                traceback.print_exc()
    # Every silo searched its own recipe above. Conditions that build a
    # feature set for ONE silo use that silo's own recipe (see _silo_recipe).
    # The three names below are only for the conditions that build ACROSS
    # silos — pooled rows, or one lead silo's rows carrying several silos'
    # representations — where no single silo's recipe is the right one. That
    # fallback is whichever silo happens to come first in dict order, which is
    # arbitrary, so it is recorded in the result file rather than left
    # implicit.
    _fallback_silo = next(iter(silo_blueprints), None)
    _default_bp = silo_blueprints.get(_fallback_silo) if _fallback_silo else None
    _alloc = _default_bp.allocation_method if _default_bp else 'performance_weighted'
    _imp = _default_bp.importance_method if _default_bp else 'treeshap'
    _int_eval = _default_bp.internal_evaluator if _default_bp else 'match'
    _int_eval = _resolve_match(_int_eval, 'rf')
    print(f"  Cross-silo recipe (from silo {_fallback_silo}): "
          f"allocation={_alloc}, importance={_imp}, evaluator={_int_eval}")

    def _silo_recipe(sname):
        """The searched recipe for one silo, falling back to the cross-silo one.

        Returns (allocation_method, importance_method, internal_evaluator).
        """
        sbp = silo_blueprints.get(sname)
        if sbp is None:
            return _alloc, _imp, _int_eval
        return (sbp.allocation_method, sbp.importance_method,
                _resolve_match(sbp.internal_evaluator, 'rf'))

    # Find shared reps
    all_rep_sets = [set(sd['reps'].keys()) for sd in silo_data.values()]
    shared_reps = sorted(set.intersection(*all_rep_sets))
    print(f"\n  Shared reps ({len(shared_reps)}): {', '.join(shared_reps)}")

    if len(shared_reps) < 3:
        warnings.warn("Few shared reps — using per-silo available reps")

    results = {}
    silo_names = sorted(silo_data.keys())

    # ── Phase 2: C1 — Independent baselines ──────────────────────────────
    print("\n--- Phase 2: C1 — Independent baselines ---")
    independent = {}
    silo_fis = {}

    for sname in silo_names:
        sd = silo_data[sname]
        print(f"  {sname}...", end=' ')
        try:
            s_alloc, s_imp, s_int_eval = _silo_recipe(sname)
            metrics, fi, split = _silo_baseline(
                sd, shared_reps, seed=fed_seed,
                alloc=s_alloc, imp=s_imp, int_eval=s_int_eval,
                fed_model=fed_model, fed_seed=fed_seed,
                split=silo_splits[sname])
            independent[sname] = metrics
            silo_fis[sname] = fi
            if 'rmse' in metrics:
                # rho is None (not NaN) when the model predicted one constant
                # value for every held-out row — see compute_reg_metrics.
                _rho = metrics.get('spearman_rho')
                _rho_s = ('undefined (constant prediction)' if _rho is None
                          else f"{_rho:.4f}")
                print(f"RMSE={metrics['rmse']:.4f}  rho={_rho_s}")
            else:
                print(f"MCC={metrics.get('mcc', float('nan')):.4f}  "
                      f"BAC={metrics.get('bac', float('nan')):.4f}")
        except Exception as e:
            # Surface the traceback (not just str(e)) so a shape/import/
            # NameError in the create_hybrid path is debuggable from the
            # expensive run's logs (FED-21).
            independent[sname] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()

    results['c1_independent'] = convert_numpy(independent)

    # ── Phase 3: C2 — Cross-silo importance transfer ─────────────────────
    print("\n--- Phase 3: C2 — Cross-silo importance transfer ---")
    transfer_results = {}
    # The transfer was measured at 100 features and nowhere else, so a
    # negative result could not be told apart from a result about 100.
    # KIRBY_FED_BUDGETS narrows this for a local run.
    TRANSFER_BUDGETS = federated_budgets()
    HEADLINE_BUDGET = (100 if 100 in TRANSFER_BUDGETS
                       else TRANSFER_BUDGETS[0])
    print(f"  budgets swept: {TRANSFER_BUDGETS}, "
          f"headline {HEADLINE_BUDGET} features")

    # ONE HYBRID PER SILO PER WIDTH. Until 2026-09-28 every transfer took the
    # source's C1 selection, which create_hybrid builds at its default 100
    # features, and truncated it to the transfer width. A width above 100
    # therefore carried the same 100 features as 100 itself: the sweep above
    # 100 was the headline repeated. The comparison was also lopsided: a
    # transfer at width b was scored against the target's own hybrid at 100.
    # Now each silo builds its own hybrid at every width, on its own training
    # rows with its own recipe. That hybrid is the source of a transfer at
    # width b AND the target's own-selection comparison at width b.
    # C1 is create_hybrid at 100 with the same arguments, so it is reused at
    # width 100 rather than built twice.
    width_fis = {}
    width_metrics = {}
    for sname in silo_names:
        width_fis[sname] = {}
        width_metrics[sname] = {}
        s_alloc, s_imp, s_int_eval = _silo_recipe(sname)
        for b in TRANSFER_BUDGETS:
            if b == 100 and sname in silo_fis:
                width_fis[sname][b] = silo_fis[sname]
                width_metrics[sname][b] = independent[sname]
                continue
            try:
                m_b, fi_b, _ = _silo_baseline(
                    silo_data[sname], shared_reps, seed=fed_seed,
                    alloc=s_alloc, imp=s_imp, int_eval=s_int_eval,
                    fed_model=fed_model, fed_seed=fed_seed,
                    split=silo_splits[sname], budget=b)
                width_fis[sname][b] = fi_b
                width_metrics[sname][b] = m_b
            except Exception as e:
                width_metrics[sname][b] = {'error': str(e)}
                print(f"  {sname} own hybrid at {b} features FAILED: {e}")
    results['c2_own_selection_by_width'] = convert_numpy(
        {s: {f'b{b}': m for b, m in d.items()}
         for s, d in width_metrics.items()})

    for source_name in silo_names:
        if not width_fis.get(source_name):
            continue
        for target_name in silo_names:
            if source_name == target_name:
                continue
            for transfer_budget in TRANSFER_BUDGETS:
                source_fi = width_fis[source_name].get(transfer_budget)
                if source_fi is None:
                    continue
                key = f"{source_name}→{target_name}@b{transfer_budget}"
                print(f"  {key}...", end=' ')
                try:
                    target_sd = silo_data[target_name]
                    target_reps = {r: target_sd['reps'][r]
                                   for r in shared_reps if r in target_sd['reps']}
                    if len(target_reps) < 2:
                        target_reps = target_sd['reps']

                    tr, te = silo_splits[target_name]
                    rep_tr = {r: arr[tr] for r, arr in target_reps.items()}
                    rep_te = {r: arr[te] for r, arr in target_reps.items()}
                    y_tr = target_sd['labels'][tr]
                    y_te = target_sd['labels'][te]

                    X_target, transferred_fi, stats = importance_profile_transfer(
                        source_fi, rep_tr, y_tr, budget=transfer_budget)

                    if X_target is None:
                        transfer_results[key] = {'error': 'no features transferred'}
                        print("no features")
                        continue

                    X_te = apply_feature_selection(rep_te, transferred_fi)
                    if target_sd.get('task_type') == 'classification':
                        mdl = fit_fed_classifier(
                            fed_model, X_target, y_tr, seed=fed_seed)
                        y_pred = mdl.predict(X_te)
                        y_prob = (mdl.predict_proba(X_te)
                                  if hasattr(mdl, 'predict_proba') else None)
                        metrics = compute_clf_metrics(y_te, y_pred, y_prob)
                    else:
                        mdl = fit_fed_regressor(
                            fed_model, X_target, y_tr, seed=fed_seed)
                        y_pred = mdl.predict(X_te)
                        metrics = compute_reg_metrics(y_te, y_pred)
                    metrics['transfer_stats'] = stats

                    # The comparison is the target's OWN hybrid at the SAME
                    # width, on the same rows. Positive means the transfer
                    # did better in both: RMSE is lower-better, so own minus
                    # transfer; spearman is higher-better, so transfer minus
                    # own. Spearman is the metric the three silos resolve to.
                    _own = width_metrics.get(target_name, {}).get(
                        transfer_budget, {})
                    ind_rmse = _own.get('rmse', float('nan'))
                    if ind_rmse is not None and not np.isnan(ind_rmse):
                        metrics['rmse_improvement'] = float(ind_rmse - metrics['rmse'])
                    _own_rho = _own.get('spearman_rho')
                    _tr_rho = metrics.get('spearman_rho')
                    metrics['spearman_improvement'] = (
                        float(_tr_rho - _own_rho)
                        if _own_rho is not None and _tr_rho is not None
                        else None)
                    metrics['own_selection_rmse'] = ind_rmse
                    metrics['own_selection_spearman'] = _own_rho

                    transfer_results[key] = metrics
                    if 'rmse' in metrics:
                        print(f"RMSE={metrics['rmse']:.4f}  "
                              f"(vs ind {ind_rmse:.4f})")
                    else:
                        print(f"MCC={metrics.get('mcc', float('nan')):.4f}")
                except Exception as e:
                    transfer_results[key] = {'error': str(e)}
                    print(f"FAILED: {e}")

    results['c2_importance_transfer'] = convert_numpy(transfer_results)

    # ── Phase 4: C3 — Within-silo ESM2 clustering ────────────────────────
    print("\n--- Phase 4: C3 — Within-silo ESM2 clustering ---")
    clustering_results = {}

    for sname in silo_names:
        sd = silo_data[sname]
        silo_reps = {r: sd['reps'][r] for r in shared_reps if r in sd['reps']}
        if len(silo_reps) < 2:
            silo_reps = sd['reps']

        print(f"  {sname}: clustering...", end=' ')
        try:
            tr, te = silo_splits[sname]
            y_tr = sd['labels'][tr]
            rep_tr = {r: arr[tr] for r, arr in silo_reps.items()}
            rep_te = {r: arr[te] for r, arr in silo_reps.items()}
            s_augs = sd.get('augmentations')
            s_aug_cond = 'uncapped' if s_augs else 'none'
            aug_tr = ({k: v[tr] for k, v in s_augs.items()} if s_augs else None)

            # Cluster on TRAIN data only (no test leakage into clusters)
            sub_silo_indices = cluster_antibody_silos(
                rep_tr, k=K_INTRA, seed=fed_seed)

            # sub_silo_indices are already train-relative (0..len(tr)-1)
            train_sub_indices = {}
            for c_id, idx in sub_silo_indices.items():
                if len(idx) >= 10:
                    train_sub_indices[c_id] = idx

            if len(train_sub_indices) < 2:
                clustering_results[sname] = {
                    'error': 'too few viable sub-clusters'}
                print("too few clusters")
                continue

            # Per-cluster importance profiles
            # Use lower budget for small clusters to avoid overfitting
            profiles = {}
            # A cluster whose hybrid raised, or whose budget came out below 5
            # features, used to vanish with nothing in the result saying so,
            # and the stability was then computed over whichever clusters
            # were left. Each one is now named with its reason.
            profile_failures = {}
            # Every cluster here is a subset of ONE silo's rows, so the recipe
            # is that silo's own searched one, not the cross-silo fallback.
            c3_alloc, c3_imp, c3_int_eval = _silo_recipe(sname)
            for c_id, c_idx in train_sub_indices.items():
                c_reps = {r: arr[c_idx] for r, arr in rep_tr.items()}
                c_labels = y_tr[c_idx]
                c_augs = ({k: v[c_idx] for k, v in aug_tr.items()}
                          if aug_tr else None)
                c_budget = min(30, len(c_idx) // 3)  # conservative for small clusters
                if c_budget < 5:
                    profile_failures[c_id] = (
                        f'budget {c_budget} features from {len(c_idx)} rows, '
                        f'below the 5-feature minimum')
                    continue
                try:
                    # ⛔ THE SILO'S SEQUENCE-IDENTITY GROUPS, WHICH THIS CALL DID NOT PASS.
                    # Split audit kind B, fed04_cluster_aware_importance.py:89. create_hybrid
                    # carves an internal holdout to decide which features are admitted;
                    # internal_holdout_split (src/kirby/hybrid.py:5435) uses GroupKFold when it
                    # is HANDED groups and a fixed-seed random carve when it is not. fed05
                    # honours identity clusters in its 80/20 splits and dropped them here, so
                    # features were admitted on a carve that could put two antibodies of one
                    # cluster on opposite sides. No score already reported is contaminated:
                    # this carve is inside the training rows and the test rows were cut
                    # group-aware first. Fixed 2026-09-18.
                    # THIS silo's groups on THIS silo's training rows. Until
                    # 2026-09-28 this read s_groups and s_tr, which are left
                    # over from the recipe-search loop above and belong to
                    # whichever silo was searched LAST (develop, 672 rows).
                    # ddg_exp (429 rows) and dms (659 rows) therefore got
                    # another dataset's cluster ids, or an IndexError that the
                    # bare except below turned into a silently missing profile.
                    _c3_groups = sd.get('groups')
                    _, fi_c = create_hybrid(
                        c_reps, c_labels,
                        groups=(np.asarray(_c3_groups)[tr][c_idx]
                                if _c3_groups is not None else None),
                        allocation_method=c3_alloc,
                        importance_method=c3_imp,
                        internal_evaluator=c3_int_eval,
                        metric_fn=get_domain_metric(sd)[1],
                        lower_is_better=get_domain_metric(sd)[2],
                        budget=c_budget,
                        augmentations=c_augs,
                        augmentation_strategy=s_aug_cond)
                    profiles[c_id] = extract_importance_profile(fi_c)
                except Exception as e:
                    profile_failures[c_id] = f'{type(e).__name__}: {e}'
                    print(f"\n    cluster {c_id} hybrid FAILED: {e}", end=' ')

            mean_sp, pairwise = compare_importance_profiles(profiles)
            clustering_results[sname] = {
                'n_sub_clusters': len(train_sub_indices),
                'n_profiles_compared': len(profiles),
                'profile_failures': profile_failures,
                'profile_stability': mean_sp,
                'profiles': profiles,
                'cluster_sizes': {
                    c: len(idx) for c, idx in train_sub_indices.items()},
            }
            # compare_importance_profiles returns None, not a number, when
            # stability is undefined — fewer than three representations to
            # correlate over, or every cluster's importance vector constant.
            # Formatting None with :.3f raises, and the raise was caught by
            # the except below, which then REPLACED the stability that had
            # just been computed and stored with an error dict. A print was
            # discarding a real result. fed04 already guards this; fed05 did
            # not. Fixed 2026-09-12.
            _sp_str = 'undefined' if mean_sp is None else f"{mean_sp:.3f}"
            print(f"{len(train_sub_indices)} clusters, stability={_sp_str}")

        except Exception as e:
            clustering_results[sname] = {
                'error': str(e), 'traceback': traceback.format_exc()}
            print(f"FAILED: {e}")
            traceback.print_exc()

    results['c3_within_silo_clustering'] = convert_numpy(clustering_results)

    # ── Phase 5: C4 — Noise heterogeneity ─────────────────────────────────
    print("\n--- Phase 5: C4 — Noise heterogeneity ---")
    noise_results = {}

    # Inject noise into the largest silo (ddg_exp), keep others clean
    noisy_silo = 'ddg_exp' if 'ddg_exp' in silo_data else silo_names[0]
    print(f"  Injecting Gaussian noise (sigma={NOISE_SIGMA}) into {noisy_silo}")

    # Build pooled data across silos for federated feature selection
    all_labels = []
    all_reps_pooled = {r: [] for r in shared_reps}
    silo_idx_map = {}  # silo_name → indices into pooled array
    # Sequence-identity groups, concatenated in the same row order as the
    # pooled features and shifted by an ACCUMULATING offset so two silos'
    # cluster ids never collide. federated_feature_selection needs these or it
    # refuses to score a silo on a held-out slice.
    pooled_groups_parts = []
    offset = 0
    group_offset = 0

    for sname in silo_names:
        sd = silo_data[sname]
        tr, _ = silo_splits[sname]
        n_tr = len(tr)
        # Z-score normalize per silo before pooling (different endpoints)
        silo_y = sd['labels'][tr]
        silo_std = np.std(silo_y)
        silo_y_z = (zscore(silo_y) if silo_std > 1e-12
                     else np.zeros_like(silo_y))
        all_labels.append(silo_y_z)
        for r in shared_reps:
            if r in sd['reps']:
                all_reps_pooled[r].append(sd['reps'][r][tr])
        g_s = sd.get('groups')
        if g_s is not None and n_tr:
            # Cluster ids are integers on the antibody loaders, but factorise
            # anyway so a string-valued group vector does not raise here.
            codes = np.unique(np.asarray(g_s)[tr], return_inverse=True)[1]
            shifted = codes.astype(np.int64) + group_offset
            pooled_groups_parts.append(shifted)
            group_offset = int(shifted.max()) + 1
        else:
            # No groups for this silo: every row is its own cluster, which is
            # the conservative reading — nothing is allowed to share a split
            # side on the strength of an identity we do not have.
            pooled_groups_parts.append(np.arange(n_tr) + group_offset)
            group_offset += n_tr
        silo_idx_map[sname] = np.arange(offset, offset + n_tr)
        offset += n_tr

    pooled_groups = (np.concatenate(pooled_groups_parts)
                     if pooled_groups_parts else None)

    # Check all reps have consistent sizes
    valid_reps = {}
    for r in shared_reps:
        parts = all_reps_pooled[r]
        if len(parts) == len(silo_names):
            valid_reps[r] = np.vstack(parts)
    pooled_labels = np.concatenate(all_labels)

    if len(valid_reps) >= 2:
        # Inject noise
        noise_silo_idx = {
            i + 1: silo_idx_map[sname]
            for i, sname in enumerate(silo_names)
        }
        noise_levels = {i + 1: 0.0 for i in range(len(silo_names))}
        noisy_silo_id = silo_names.index(noisy_silo) + 1
        noise_levels[noisy_silo_id] = NOISE_SIGMA

        labels_noisy, noise_mask, _ = inject_noise_per_silo(
            pooled_labels, noise_silo_idx, noise_levels,
            noise_type='gaussian')

        # Federated feature selection with noisy labels
        try:
            X_h_noisy, silo_budgets, silo_scores, _ = \
                federated_feature_selection(
                    valid_reps, labels_noisy, noise_silo_idx,
                    len(silo_names),
                    allocation_method=_alloc, importance_method=_imp,
                    internal_evaluator=_int_eval,
                    groups=pooled_groups, task_type='regression',
                    downstream_model=fed_to_kirby_model(fed_model),
                    # A downstream model with no metric leaves create_hybrid
                    # scoring each representation by the internal evaluator's
                    # R² instead of by the metric the three antibody silos
                    # resolve to, so the share of the budget each one gets is
                    # decided on a different quantity from the one this
                    # experiment reports. The pooled labels here are z-scored
                    # per silo before pooling, so spearman — which is
                    # scale-free — is the one that survives that pooling.
                    metric_fn=BUILTIN_METRICS['spearman'][0],
                    lower_is_better=BUILTIN_METRICS['spearman'][1])

            # Check: does noisy silo get less budget?
            noise_results = {
                'noisy_silo': noisy_silo,
                'sigma': NOISE_SIGMA,
                'silo_budgets': silo_budgets,
                'silo_scores': silo_scores,
                'noisy_silo_budget': silo_budgets.get(noisy_silo_id, 0),
                'clean_silo_budgets': {
                    silo_names[i]: silo_budgets.get(i + 1, 0)
                    for i in range(len(silo_names))
                    if silo_names[i] != noisy_silo},
            }

            noisy_budget = silo_budgets.get(noisy_silo_id, 0)
            other_budgets = [silo_budgets.get(i + 1, 0)
                             for i in range(len(silo_names))
                             if i + 1 != noisy_silo_id]
            avg_clean = np.mean(other_budgets) if other_budgets else 0
            print(f"  Noisy silo budget: {noisy_budget}, "
                  f"avg clean: {avg_clean:.0f}")
            if noisy_budget < avg_clean:
                print("  --> KIRBy downweights noisy silo")
            else:
                print("  --> No implicit noise detection")

        except Exception as e:
            noise_results = {'error': str(e)}
            print(f"  FAILED: {e}")
    else:
        noise_results = {'error': f'only {len(valid_reps)} valid pooled reps'}
        print(f"  SKIP: insufficient pooled reps")

    results['c4_noise_heterogeneity'] = convert_numpy(noise_results)

    # ── Phase 6: C5 — Pooled binding ──────────────────────────────────────
    print("\n--- Phase 6: C5 — Pooled binding ---")
    pooled_binding = {}

    binding_silos = ['ddg_exp', 'dms']
    available_binding = [s for s in binding_silos if s in silo_data]

    if len(available_binding) >= 2:
        print(f"  Pooling: {available_binding} (z-score normalized)")
        try:
            # Z-score normalize each silo's labels before pooling
            # The representations BOTH binding silos have. This used to be
            # shared_reps, the three-way intersection that also demands the
            # developability silo's representations, which this condition
            # never touches. With the DATASET_REPS 'general' lists of
            # 2026-09-28 that three-way set is empty while ddg_combined and
            # abagym_EGFR share mutation_features and esm2_concat, so the
            # pool was skipped for a silo that is not in it.
            c5_reps = sorted(set.intersection(
                *[set(silo_data[s]['reps']) for s in available_binding]))
            print(f"  Representations both binding silos have "
                  f"({len(c5_reps)}): {', '.join(c5_reps)}")
            pool_labels = []
            pool_reps = {r: [] for r in c5_reps}
            pool_source = []

            for sname in available_binding:
                sd = silo_data[sname]
                tr, _ = silo_splits[sname]
                y_tr = sd['labels'][tr]
                y_z = (zscore(y_tr) if len(y_tr) > 1 and np.std(y_tr) > 1e-12
                       else np.zeros_like(y_tr))
                pool_labels.append(y_z)
                pool_source.extend([sname] * len(tr))
                for r in c5_reps:
                    if r in sd['reps']:
                        pool_reps[r].append(sd['reps'][r][tr])

            pool_labels = np.concatenate(pool_labels)
            pool_valid_reps = {}
            for r in c5_reps:
                parts = pool_reps[r]
                if len(parts) == len(available_binding):
                    pool_valid_reps[r] = np.vstack(parts)

            if len(pool_valid_reps) >= 2:
                # Concatenate groups with offset to avoid cross-silo collisions.
                # offset must ACCUMULATE across silos — if we reassigned it to
                # the current silo's max+1, silos ≥ 3 would collide because the
                # third silo's offset would be based only on silo 2's range and
                # could overlap silo 1.
                pool_groups_list = []
                offset = 0
                for sname in available_binding:
                    sd = silo_data[sname]
                    tr_s, _ = silo_splits[sname]
                    if len(tr_s) == 0:
                        continue
                    g = sd.get('groups')
                    if g is not None:
                        shifted = g[tr_s] + offset
                        pool_groups_list.append(shifted)
                        offset = int(shifted.max()) + 1
                    else:
                        pool_groups_list.append(np.arange(len(tr_s)) + offset)
                        offset += len(tr_s)
                pool_groups = np.concatenate(pool_groups_list)

                # 80/20 split (group-aware)
                p_tr, p_te = next(cv_split(
                    pool_groups, 1, np.zeros(len(pool_labels)),
                    pool_labels, seed=fed_seed))

                rep_tr = {r: arr[p_tr] for r, arr in pool_valid_reps.items()}
                rep_te = {r: arr[p_te] for r, arr in pool_valid_reps.items()}

                # Augmentations would require cross-silo concatenation —
                # same shape contract as labels. Skip with explicit 'none'.
                # ⛔ THE SILO'S SEQUENCE-IDENTITY GROUPS, WHICH THIS CALL DID NOT PASS.
                # Split audit kind B, fed04_cluster_aware_importance.py:89. create_hybrid
                # carves an internal holdout to decide which features are admitted;
                # internal_holdout_split (src/kirby/hybrid.py:5435) uses GroupKFold when it
                # is HANDED groups and a fixed-seed random carve when it is not. fed05
                # honours identity clusters in its 80/20 splits and dropped them here, so
                # features were admitted on a carve that could put two antibodies of one
                # cluster on opposite sides. No score already reported is contaminated:
                # this carve is inside the training rows and the test rows were cut
                # group-aware first. Fixed 2026-09-18.
                X_h, fi = create_hybrid(
                    rep_tr, pool_labels[p_tr],
                    groups=(np.asarray(pool_groups)[p_tr]
                            if pool_groups is not None else None),
                    allocation_method=_alloc,
                    importance_method=_imp,
                    internal_evaluator=_int_eval,
                    # Labels are z-scored per silo before pooling; spearman
                    # is the scale-free metric both silos resolve to (as C4).
                    metric_fn=BUILTIN_METRICS['spearman'][0],
                    lower_is_better=BUILTIN_METRICS['spearman'][1],
                    augmentation_strategy='none')
                X_h_te = apply_feature_selection(rep_te, fi)

                reg = fit_fed_regressor(
                    fed_model, X_h, pool_labels[p_tr], seed=fed_seed)
                y_pred = reg.predict(X_h_te)
                pooled_binding = compute_reg_metrics(pool_labels[p_te], y_pred)
                pooled_binding['n_pooled'] = len(pool_labels)
                pooled_binding['silos_pooled'] = available_binding
                _prho = pooled_binding.get('spearman_rho')
                _prho_s = ('undefined (constant prediction)' if _prho is None
                           else f"{_prho:.4f}")
                print(f"  Pooled RMSE={pooled_binding['rmse']:.4f}  "
                      f"rho={_prho_s}")
            else:
                pooled_binding = {'error': 'insufficient shared reps for pooling'}
                print("  SKIP: insufficient reps")

        except Exception as e:
            pooled_binding = {'error': str(e)}
            print(f"  FAILED: {e}")
    else:
        pooled_binding = {'error': f'need >= 2 binding silos, got {len(available_binding)}'}
        print(f"  SKIP: only {len(available_binding)} binding silos")

    results['c5_pooled_binding'] = convert_numpy(pooled_binding)

    # ── Phase 7: C6 — Rep diversity per silo ──────────────────────────────
    print("\n--- Phase 7: C6 — Rep diversity per silo ---")
    rep_diversity = {}

    # Use one silo's data (largest) for this condition
    lead_silo = max(silo_data, key=lambda s: len(silo_data[s]['labels']))
    sd_lead = silo_data[lead_silo]
    print(f"  Using {lead_silo} data ({len(sd_lead['labels'])} samples)")

    # Resolve available reps per diversity silo
    div_silos = {}
    for div_name, rep_list in REP_DIVERSITY_SILOS.items():
        available = [r for r in rep_list if r in sd_lead['reps']]
        if available:
            div_silos[div_name] = available

    K_div = len(div_silos)
    if K_div < 2:
        rep_diversity = {'error': f'only {K_div} rep diversity silos available'}
        print(f"  SKIP: need >= 2 diversity silos")
    else:
        print(f"  {K_div} diversity silos: "
              f"{', '.join(f'{k}={v}' for k, v in div_silos.items())}")

        tr, te = silo_splits[lead_silo]
        y_tr = sd_lead['labels'][tr]
        y_te = sd_lead['labels'][te]

        # Per-silo baselines
        div_baselines = {}
        teacher_models = [None]  # index 0 placeholder
        silo_train_sim = [None]
        silo_test_sim = [None]
        silo_train_qual = [None]
        silo_test_qual = [None]

        div_names = sorted(div_silos.keys())
        for dname in div_names:
            rlist = div_silos[dname]
            X_tr = np.hstack([sd_lead['reps'][r][tr] for r in rlist])
            X_te = np.hstack([sd_lead['reps'][r][te] for r in rlist])

            reg = fit_fed_regressor(fed_model, X_tr, y_tr, seed=fed_seed)
            y_pred = reg.predict(X_te)
            div_baselines[dname] = compute_reg_metrics(y_te, y_pred)
            div_baselines[dname]['reps'] = rlist

            teacher_models.append((reg, len(X_tr), X_tr))
            silo_train_sim.append(X_tr)
            silo_test_sim.append(X_te)
            silo_train_qual.append((X_tr, y_tr))
            silo_test_qual.append(X_te)

            print(f"    {dname}: RMSE={div_baselines[dname]['rmse']:.4f}")

        # FLuID-style quality ensemble
        try:
            # Each teacher's weight is 1/RMSE from cross-validation on the
            # lead silo's training rows. With no groups that cross-validation
            # is a random KFold, which puts two antibodies of one
            # sequence-identity cluster on both sides and rewards memorising
            # them, while every other number in this file is scored across
            # clusters. flab_thermo_tresanco, the lead silo, has 247 clusters
            # in its training rows (measured 2026-09-28), so GroupKFold's five
            # folds are always available there.
            _lead_g = sd_lead.get('groups')
            y_pred_ens, rmse_w = predict_fluid_quality_ensemble_regression(
                teacher_models, silo_train_qual, silo_test_qual, K_div,
                groups=(np.asarray(_lead_g)[tr]
                        if _lead_g is not None else None))
            ensemble_metrics = compute_reg_metrics(y_te, y_pred_ens)
            ensemble_metrics['rmse_weights'] = rmse_w
            print(f"    Quality ensemble: RMSE={ensemble_metrics['rmse']:.4f}")
        except Exception as e:
            ensemble_metrics = {'error': str(e)}
            print(f"    Ensemble FAILED: {e}")

        # KIRBy hybrid across all diversity silo reps
        try:
            all_div_reps_tr = {}
            all_div_reps_te = {}
            for dname in div_names:
                for r in div_silos[dname]:
                    all_div_reps_tr[r] = sd_lead['reps'][r][tr]
                    all_div_reps_te[r] = sd_lead['reps'][r][te]
            lead_augs = sd_lead.get('augmentations')
            lead_aug_cond = 'uncapped' if lead_augs else 'none'
            lead_tr_augs = ({k: v[tr] for k, v in lead_augs.items()}
                            if lead_augs else None)
            lead_te_augs = ({k: v[te] for k, v in lead_augs.items()}
                            if lead_augs else None)

            # ⛔ THE SILO'S SEQUENCE-IDENTITY GROUPS, WHICH THIS CALL DID NOT PASS.
            # Split audit kind B, fed04_cluster_aware_importance.py:89. create_hybrid
            # carves an internal holdout to decide which features are admitted;
            # internal_holdout_split (src/kirby/hybrid.py:5435) uses GroupKFold when it
            # is HANDED groups and a fixed-seed random carve when it is not. fed05
            # honours identity clusters in its 80/20 splits and dropped them here, so
            # features were admitted on a carve that could put two antibodies of one
            # cluster on opposite sides. No score already reported is contaminated:
            # this carve is inside the training rows and the test rows were cut
            # group-aware first. Fixed 2026-09-18.
            _lead_groups = sd_lead.get('groups')
            X_h, fi_h = create_hybrid(
                all_div_reps_tr, y_tr,
                groups=(np.asarray(_lead_groups)[tr]
                        if _lead_groups is not None else None),
                allocation_method=_alloc,
                importance_method=_imp,
                internal_evaluator=_int_eval,
                metric_fn=get_domain_metric(sd_lead)[1],
                lower_is_better=get_domain_metric(sd_lead)[2],
                augmentations=lead_tr_augs,
                augmentation_strategy=lead_aug_cond)
            X_h_te = apply_feature_selection(all_div_reps_te, fi_h)
            _aug_info_h = fi_h.get('augmentation_info') if isinstance(fi_h, dict) else None
            if _aug_info_h and lead_te_augs:
                _X_te_aug_h = apply_augmentation_selection(lead_te_augs, _aug_info_h)
                if _X_te_aug_h is not None:
                    X_h_te = np.hstack([X_h_te, _X_te_aug_h])

            reg_h = fit_fed_regressor(fed_model, X_h, y_tr, seed=fed_seed)
            y_pred_h = reg_h.predict(X_h_te)
            hybrid_metrics = compute_reg_metrics(y_te, y_pred_h)
            hybrid_metrics['allocation'] = convert_numpy(
                fi_h.get('allocation', {}))
            print(f"    KIRBy hybrid: RMSE={hybrid_metrics['rmse']:.4f}")
        except Exception as e:
            hybrid_metrics = {'error': str(e), 'traceback': traceback.format_exc()}
            print(f"    Hybrid FAILED: {e}")
            traceback.print_exc()

        rep_diversity = {
            'lead_silo': lead_silo,
            'diversity_silos': dict(div_silos),
            'baselines': div_baselines,
            'quality_ensemble': ensemble_metrics,
            'kirby_hybrid': hybrid_metrics,
        }

    results['c6_rep_diversity'] = convert_numpy(rep_diversity)

    # ── Phase 8: C7/C8 — FLuID ensemble baselines (SKIPPED) ──────────────
    # Original intent: train per-silo teachers, then predict-and-ensemble
    # across silos to compare KIRBy's feature-level fusion against FLuID's
    # prediction-level ensemble.
    #
    # NOT IMPLEMENTED here: the antibody silos do not share a common feature
    # space. Each silo's teacher is trained on `cap_dimensions(X_tr, X_te)`
    # where X_tr/X_te come from its OWN shared-rep concatenation. The
    # cap_dimensions PCA basis is fit per silo, so a teacher's `predict()`
    # called on another silo's features either crashes (dim mismatch) or
    # silently returns meaningless numbers (different PCA basis, different
    # label scale: kcal/mol vs normalized DMS score vs °C). Either failure
    # mode produces invalid results.
    #
    # Safer to report an explicit skip than fabricate numbers. A correct
    # implementation would require a shared feature space (e.g. a fixed
    # cross-silo PCA basis) AND per-silo label z-scoring before teacher
    # training. Left for future work.
    print("\n--- Phase 8: C7/C8 — FLuID ensemble baselines ---")
    _skip_msg = (
        "NOT IMPLEMENTED: cross-silo predict() requires a shared feature "
        "space and shared label scale. Antibody silos here have neither "
        "(per-silo PCA basis via cap_dimensions; labels in kcal/mol vs "
        "normalized DMS vs °C). See module docstring."
    )
    results['c7_fluid_quality'] = {'error': _skip_msg, 'skipped': True}
    results['c8_fluid_similarity'] = {'error': _skip_msg, 'skipped': True}
    print(f"  SKIP — {_skip_msg}")

    # ── Phase 9: Save results ─────────────────────────────────────────────
    final = {
        'experiment': 'fed05_antibody_federated',
        'dataset': dataset_name,
        'config': {
            'silos': dict(SILO_DATASETS),
            'per_silo_recipe': {
                _sn: {'allocation_method': _bp.allocation_method,
                      'importance_method': _bp.importance_method,
                      'internal_evaluator': _bp.internal_evaluator}
                for _sn, _bp in silo_blueprints.items()},
            'cross_silo_recipe_taken_from': _fallback_silo,
            # Which of those silos actually contributed, and why any did not.
            'silos_loaded': sorted(silo_data),
            'silo_load_failures': silo_load_failures,
            'shared_reps': shared_reps,
            'k_intra': K_INTRA,
            'noise_sigma': NOISE_SIGMA,
        },
        **results,
    }

    # Summary table
    print(f"\n{'─' * 70}")
    # Both columns are RMSE at the headline width, so the two are the same
    # number of features. The first column used to be C1, which is always
    # 100 features, whatever width the second column was read at.
    print(f"  RMSE at {HEADLINE_BUDGET} features, lower is better")
    print(f"  {'Silo':<12} {'Own':>10} {'Best C2':>10}")
    for sname in silo_names:
        ind = width_metrics.get(sname, {}).get(HEADLINE_BUDGET, {})
        ind_rmse = ind.get('rmse', float('nan'))
        best_tr = float('nan')
        for k, v in transfer_results.items():
            if k.endswith(f'→{sname}@b{HEADLINE_BUDGET}') and 'rmse' in v:
                if np.isnan(best_tr) or v['rmse'] < best_tr:
                    best_tr = v['rmse']
        print(f"  {sname:<12} {ind_rmse:>10.4f} {best_tr:>10.4f}")

    return final


# ── One (FLuID model, seed) call saved as it finishes (2026-09-25) ──────────
# federated_helpers.run_all calls run() once per (fed_model, fed_seed) -- nine
# calls -- and wrote fed05_antibody_federated/<dataset>.json only after the ninth. A kill or
# an exception in any call lost every call before it. Each call's return value
# is now written to fed05_antibody_federated/<dataset>.partial.jsonl the moment it returns
# and read back on the next attempt (_checkpoint.resumable_run), keyed on the
# model, the seed and any keyword argument, under an identity that includes the
# input fingerprint, the plain config and the KIRBY_* settings. The returned
# dict is decoded to the same objects, so aggregate_seed_runs and the final
# JSON are unchanged. --force-rerun / KIRBY_FORCE_RERUN=1 discards the file.
def run(data, config, output_dir, dataset_name, **kwargs):
    from experiments._checkpoint import resumable_run
    _key = {'fed_model': str(config.get('fed_model', 'rf')),
            'fed_seed': int(config.get('fed_seed', 0)),
            **{k: v for k, v in kwargs.items()}}
    return resumable_run(
        'fed05_antibody_federated',
        lambda: _run_one(data, config, output_dir, dataset_name, **kwargs),
        data, config, output_dir, dataset_name, key=_key)



if __name__ == '__main__':
    # Fed05 loads its own silo data — the dataset arg is for naming only
    import argparse
    parser = argparse.ArgumentParser(
        description='Fed05: Antibody Federated Extension')
    parser.add_argument('--output', type=str,
                        default='tests/results/federated')
    args = parser.parse_args()

    # Pass placeholder data — run() loads silos internally
    data = {'reps': {}, 'labels': np.array([]), 'domain': 'antibody'}
    results = run(data, {}, args.output, 'antibody_federated')
    save_experiment_results(
        results, args.output, 'fed05_antibody_federated', 'antibody')
