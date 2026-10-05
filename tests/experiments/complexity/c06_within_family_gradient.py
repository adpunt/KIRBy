#!/usr/bin/env python3
"""C06: Within-Family Complexity Gradient — vary capacity within model families.

Paper: Complexity
Conclusion: Within-family capacity gradient confirms complexity matching.
Dependencies: None (standalone).

For each variant, runs find_best_hybrid() with that variant as the sole
downstream model. The library searches importance methods, allocation methods,
and budgets to find the variant-optimal hybrid recipe. Optimal budget per
variant is the budget where val score peaks. Spearman correlation between
capacity index and optimal budget tests whether capacity and budget are coupled.

Train scores / bias-variance gap are not collected here (find_best_hybrid does
not return train scores). Moved to a dedicated experiment if/when needed.

2026-09-12, decided and built: each cell of families[<family>]['heatmap'] now
carries the per-fold scores behind its mean, and the internal evaluator under
its resolved name — an SVM downstream with the matched evaluator reads
'SVM/SVM'. c06 passes that name in its own evaluator list, and the searches
run on the fast speed tier.
"""
import sys
import os
import time as _time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import spearmanr

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_family_variants,
    find_best_hybrid, get_domain_metric,
    INTERNAL_EVALUATORS,
)
from kirby.find_best_hybrid import (
    _resolve_models, FAST_INTERNAL_EVALUATORS, KNN_TUNED_K_ATTR)


# Map c06 family labels to canonical names in DEFAULT_MODELS. List-form
# _resolve_models is the only way to trigger _max_samples + _DOMAIN_MODEL_SKIPS
# + _DATASET_STRATEGY_SKIPS; the dict form is a passthrough.
# ⛔ 'Ridge' WAS A DEAD KEY. get_model_family_variants has returned Lasso, not
# Ridge, since 2026-07-18. The stale key did two things: `--families Ridge`
# matched nothing, so the filter below produced {}, the family loop never ran,
# and c06 SAVED `{'families': {}}` as a result with no error. And Lasso, having
# no entry, fell through the `canonical is None` branch below and so never went
# through _resolve_models -- the only thing that applies _max_samples,
# _DOMAIN_MODEL_SKIPS and _DATASET_STRATEGY_SKIPS.
#
# Lasso maps to None deliberately: DEFAULT_MODELS has no 'lasso' (it holds
# ridge and linear, which are different estimators), so there is no canonical
# name to gate it by. That is now stated rather than silent -- the loop prints
# it.
_FAMILY_TO_CANONICAL = {
    'RF': 'rf', 'kNN': 'knn', 'DNN': 'dnn',
    'XGBoost': 'xgboost', 'SVM': 'svm', 'GP': 'gp',
    'Lasso': None,
}

# DECIDED 2026-09-12: c06 passes the family's evaluator name in its own list.
# What changed and why it had to change here: find_best_hybrid resolves the
# word 'match' through a table of plain model names — rf, lightgbm, xgboost,
# mlp, svm, gp, knn, catboost, dnn, cnn, bnn, ngboost
# (find_best_hybrid.py:5622-5628). c06 names its downstream model
# 'SVM:C=1.0' so that lookup returned nothing and 'match' added no evaluator
# to any variant's search. Stripping the family prefix inside
# find_best_hybrid would change every caller that passes a prefixed model
# name, so the decision was to resolve the name in c06 alone. This touches
# c06 and nothing else.
#
# The linear family has no entry. The internal evaluator is fitted by
# hybrid.py:_evaluate_allocation, whose branches are rf, xgboost, lightgbm,
# mlp, svm, gp, knn, catboost, dnn, cnn, ngboost, bnn — no linear model — so
# there is no name to pass for Lasso. Its searches run on the rest of the
# list and a line below says so.
_FAMILY_TO_EVALUATOR = {
    'RF': 'rf', 'kNN': 'knn', 'DNN': 'dnn',
    'XGBoost': 'xgboost', 'SVM': 'svm', 'GP': 'gp',
}

# DECIDED 2026-09-12: c06 moves to the fast speed tier, under the rule that an
# experiment calling find_best_hybrid more than once per dataset runs fast —
# 28 searches per dataset, seven families at four capacity settings each.
# Fast cuts the allocation methods to three (performance_weighted, power_2,
# rank_exp_d0.3) and the importance methods to ONE (lightgbm_gain -- this
# comment said two and was wrong), and leaves the budget axis alone, which is
# the axis c06 reports. FAST_INTERNAL_EVALUATORS is
# ['rf', 'match', 'dnn', 'lightgbm'], so the SVM family searches
# ['rf', 'svm', 'dnn', 'lightgbm'] and the RF family ['rf', 'dnn', 'lightgbm'];
# an earlier note here said ['rf', 'match'] and halved the cost estimate. Those two
# lists are part of the search cache key (find_best_hybrid.py:2643), so a fast
# search is stored as a separate file from a full one and reuses none of it.
_C06_SPEED = 'fast'


def _family_evaluators(fam_name):
    """Internal evaluator names searched for one family, 'match' resolved here.

    Returns the speed tier's own evaluator list with 'match' replaced by the
    family's plain model name. Fast is ['rf', 'match'], so the SVM family
    searches ['rf', 'svm'] and the RF family searches ['rf'] alone.
    """
    base = (FAST_INTERNAL_EVALUATORS if _C06_SPEED == 'fast'
            else INTERNAL_EVALUATORS)
    matched = _FAMILY_TO_EVALUATOR.get(fam_name)
    names = []
    for ev in base:
        name = matched if ev == 'match' else ev
        if name is None:
            continue  # no plain model name for this family; nothing to pass
        if name not in names:
            names.append(name)
    if not names:
        # MEASURED 2026-09-25, local debug grid only. _local_grid.apply() sets
        # FAST_INTERNAL_EVALUATORS to ['match'] alone. Lasso has no plain model
        # name for 'match' to resolve to, so the loop above dropped its only
        # entry and this returned []. find_best_hybrid was then handed an EMPTY
        # internal_evaluators list for all four Lasso variants. Under the
        # shipped list ['rf', 'match', 'dnn', 'lightgbm'] Lasso still gets
        # ['rf', 'dnn', 'lightgbm'], so this branch never fires on the server.
        # 'rf' is the evaluator every other family's list already contains.
        names = ['rf']
        print(f"    {fam_name}: the evaluator list resolved to nothing "
              f"(base={list(base)}, no matched name), falling back to ['rf']")
    return names


def _apply_model_njobs(est, fam_name, var_name):
    """Set the estimator's worker-process count, when it has one.

    MEASURED 2026-09-25, cProfile over one c06 search on flab_thermo_tresanco
    (672 antibodies, 5 representations at 1000 features each, 2 budgets, 3
    allocation methods, 1 importance method, 1 internal evaluator, 3 folds =
    18 downstream fits). The search took 175.4s. 157.9s of that was the 18
    RandomForestRegressor.fit calls, at 8.8s each. The trees are not the cost:

      102.1s  time.sleep inside joblib's _retrieve — the parent process
              polling for worker processes
       53.5s  sklearn's _make_estimator / clone / get_params /
              inspect.signature — 1800 estimator clones, 100 trees x 18 fits

    get_model_family_variants builds RF, kNN and XGBoost with n_jobs=-1, so one
    fit asks for one worker per core. The local env file already pins every
    other thread pool to one (OMP_NUM_THREADS=1, KIRBY_LGBM_THREADS=1) because
    this laptop runs several experiments at once; n_jobs=-1 was the one place
    left oversubscribing it, and the workers then spent their time waiting on
    each other.

    This changes no condition and no row: same estimator, same
    hyperparameters, same molecules, same folds, identical predictions. It
    changes only how many processes one fit spawns.

    KIRBY_C06_MODEL_NJOBS sets the value. Unset, the estimator is left exactly
    as get_model_family_variants built it EXCEPT under the local debug grid,
    where it defaults to 1. The server is unchanged.
    """
    _default = '1' if os.environ.get('KIRBY_LOCAL_GRID') else ''
    _njobs = os.environ.get('KIRBY_C06_MODEL_NJOBS', _default).strip()
    if not _njobs:
        return est
    try:
        if 'n_jobs' in est.get_params():
            est.set_params(n_jobs=int(_njobs))
    except (AttributeError, ValueError, TypeError) as _e:
        print(f"      {fam_name}/{var_name}: n_jobs not set ({_e})")
    return est


def _hold_variant_setting(est, fam_name, var_name):
    """Make the search fit the setting this variant is named after.

    Added 2026-10-05. Two families did not keep the setting they vary:

    kNN: find_best_hybrid._set_knn_k resets n_neighbors to min(5, sqrt of the
    training rows) before every fit unless the estimator carries the attribute
    named by KNN_TUNED_K_ATTR, so k=20, k=10, k=5 and k=1 were all fitted at
    k=5. Stamping it keeps the variant's own k (clamped to the training rows
    minus 1 by _set_knn_k).

    GP: the four variants differed only in the RBF length scale they started
    from, and sklearn re-fits the length scale by maximising the marginal
    likelihood, so all four could end at the same length scale. Setting
    length_scale_bounds='fixed' on the RBF keeps it at the variant's value. The
    WhiteKernel noise level is still fitted.

    The variant definitions themselves live in _shared.get_model_family_variants
    and are not changed; this is applied to the estimator c06 hands the search.
    """
    if fam_name == 'kNN' and hasattr(est, 'n_neighbors'):
        setattr(est, KNN_TUNED_K_ATTR, True)
    if fam_name == 'GP' and hasattr(est, 'kernel') and est.kernel is not None:
        from sklearn.gaussian_process.kernels import RBF as _RBF

        def _fix(k):
            if isinstance(k, _RBF):
                return _RBF(length_scale=k.length_scale,
                            length_scale_bounds='fixed')
            if hasattr(k, 'k1') and hasattr(k, 'k2'):
                # Sum / Product of two kernels, e.g. RBF + WhiteKernel.
                return type(k)(_fix(k.k1), _fix(k.k2))
            return k
        est.set_params(kernel=_fix(est.kernel))
    return est


def _no_score_reason(search, model_key):
    """Why a variant has no scored budget, from the search's own rows.

    Added 2026-10-05. find_best_hybrid leaves a model out of
    best_by_model_budget when none of its configurations has a finite score,
    and c06 saved budget None with nothing saying why (the Lasso alpha=100,
    10 and 1 variants on the server run). This counts the configurations the
    search scored for this model and how many have a finite score. A Lasso
    whose coefficients are all zero predicts one constant, and a rank
    correlation of a constant is undefined; that is the likely cause for the
    large alphas but is NOT checked here.
    """
    rows = [r for r in (getattr(search, 'all_results', None) or [])
            if r.get('model') == model_key]
    finite = [r for r in rows if r.get('score_mean') is not None
              and np.isfinite(r['score_mean'])]
    if not rows:
        reason = 'the search returned no configuration for this variant'
    else:
        reason = (f'{len(rows)} configurations scored, {len(finite)} with a '
                  f'finite score')
    return {'reason': reason, 'n_configurations': len(rows),
            'n_configurations_finite': len(finite)}


def _evaluator_label(fam_name, evaluator):
    """DECIDED 2026-09-12: file the matched evaluator under its resolved name.

    An SVM downstream with the matched internal evaluator reads 'SVM/SVM',
    never 'match'. An SVM downstream scored by the rf evaluator reads
    'SVM/rf'.
    """
    if evaluator is None:
        return None
    matched = _FAMILY_TO_EVALUATOR.get(fam_name)
    return f"{fam_name}/{fam_name if evaluator == matched else evaluator}"


def run(data, config, output_dir, dataset_name, families=None):
    """Exp 6: vary complexity within each model family."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: WITHIN-FAMILY MODEL COMPLEXITY GRADIENT")
    print("=" * 70)

    task_type = data.get('task_type')
    if task_type != 'regression':
        # Reason text corrected 2026-10-05. get_model_family_variants has had
        # classifier variants since the CXA-6 fix, but this file still runs
        # regression only: it calls get_model_family_variants() with no
        # task_type, hands 'regression' to _resolve_models, the precompute
        # lookup and every find_best_hybrid call, and has no entry for the
        # classification-only 'Logit' family in _FAMILY_TO_CANONICAL.
        print(f"  Skipping: c06 is not wired for classification "
              f"(task_type={task_type}).")
        return {'skipped': True,
                'reason': ('c06 passes task_type=regression to every search '
                           'and has no entry for the Logit family; the '
                           'classifier variants exist but are not wired here'),
                'task_type': task_type}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    budgets = config['budgets']
    n_folds = config['n_splits']
    n_samples = len(labels)

    # This used to drop every budget above the molecule count, calling itself
    # "the canonical budget filter (matches find_best_hybrid.py:3934)". That
    # engine filter is gone: allow_budget_exceeds_n has defaulted True since
    # 2026-09-10, so nothing in find_best_hybrid removes a budget for exceeding
    # n. The mirror here outlived what it mirrored and was the only thing still
    # cutting c06's budget axis at the molecule count.
    eff_budgets = sorted(budgets)

    # The capacity<->budget correlation is undefined unless several DISTINCT
    # budgets are viable. On small datasets a coarse grid (e.g. [50,500,...])
    # collapses to a single value <= n_samples, pinning every variant's optimal
    # budget and forcing a NaN correlation. Guarantee a data-relative spread of
    # at least 4 distinct budgets below n_samples (and below the feature dim).
    # See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
    max_dim = max((v.shape[1] for v in reps.values()), default=n_samples)
    viable_cap = min(n_samples, max_dim)
    #
    # KIRBY_C06_BUDGET_WIDEN: '1' widens as described above, '0' uses exactly
    # the budgets handed in. The default is '1' — the shipped behaviour, so the
    # server is unchanged — EXCEPT under the local debug grid, where
    # KIRBY_LOCAL_BUDGETS is already the whole budget axis and widening it
    # undoes the cut the owner made in env.sh.
    #
    # MEASURED 2026-09-25 on flab_thermo_tresanco (672 antibodies, 5
    # representations capped at 1000 features each): with KIRBY_LOCAL_BUDGETS
    # =100,250 this printed [33, 67, 100, 134, 235, 250, 336] — 7 budgets where
    # the grid asked for 2. Each search is allocations x importance methods x
    # internal evaluators x budgets, so that is 3.5x the configurations, and
    # c06 runs 28 searches per dataset (7 families x 4 capacity variants).
    #
    # What the cut costs: each variant's peak budget is located to one of 2
    # levels instead of 7, so the capacity-vs-budget rho below is coarser. It
    # is still defined — the test below is n_distinct_budgets >= 2.
    _widen_default = '0' if os.environ.get('KIRBY_LOCAL_GRID') else '1'
    _widen = os.environ.get('KIRBY_C06_BUDGET_WIDEN', _widen_default) == '1'
    if _widen and len(eff_budgets) < 4:
        data_rel = sorted({max(5, int(viable_cap * f))
                           for f in (0.05, 0.1, 0.2, 0.35, 0.5)})
        eff_budgets = sorted({b for b in (eff_budgets + data_rel)
                              if 5 <= b <= viable_cap})
        print(f"  Small-data budget grid widened to {eff_budgets}")
    elif not _widen:
        print(f"  Budget grid left at {eff_budgets} "
              f"(KIRBY_C06_BUDGET_WIDEN=0; widening is off under the local "
              f"debug grid)")

    all_families = get_model_family_variants()

    # KIRBY_C06_FAMILIES: comma-separated family names, same effect as
    # --families. Unset, every family runs, so the server is unchanged. An
    # explicit --families on the command line wins over it.
    _fams_env = os.environ.get('KIRBY_C06_FAMILIES', '').strip()
    if _fams_env and not families:
        families = [f.strip() for f in _fams_env.split(',') if f.strip()]
        print(f"  KIRBY_C06_FAMILIES={_fams_env}: running {families} only")

    if families:
        _unknown = [f for f in families if f not in all_families]
        if _unknown:
            raise ValueError(
                f"--families named {_unknown}, which "
                f"get_model_family_variants does not return. It returns "
                f"{sorted(all_families)}. c06 used to filter these out and "
                f"save an empty result with no error.")
        all_families = {k: v for k, v in all_families.items() if k in families}

    # Compute aug_strategy first so we can pass it to the family gate
    # (matches the strategy each per-variant find_best_hybrid call will use).
    augs = data.get('augmentations')

    # c06 took data['augmentations'] straight and searched 'uncapped' whenever
    # the dataset had any, so KIRBY_LOCAL_AUG_CONDITIONS never reached it. c02
    # and c05 both call _local_aug_conditions for exactly this reason
    # (c02_complexity_grid.py:73, c05_learning_curves.py:157) and c06 was the
    # one left out. _shared._local_aug_conditions states the cost of the
    # augmented conditions: an 'uncapped' or 'capped' configuration-fold is
    # about eight model fits where 'none' is one, because the extra feature
    # block is offered at several widths and each one is fitted and scored on
    # the internal split.
    #
    # MEASURED 2026-09-25 on flab_thermo_tresanco, all 672 antibodies: one
    # search under 'uncapped' — 2 budgets, 3 allocation methods, 1 importance
    # method, 1 internal evaluator, 3 folds — took 572.3s, and c06 runs one per
    # capacity variant.
    #
    # WHAT IT COSTS: with KIRBY_LOCAL_AUG_CONDITIONS set, the augmentation
    # question has no answer in this run. Every family's heatmap, optimal
    # budget and capacity correlation is then measured on the base
    # representations alone. Every antibody is still used. It also changes
    # which families run: _DATASET_STRATEGY_SKIPS skips gp for
    # flab_thermo_tresanco under 'uncapped' and not under 'none', so the GP
    # family comes BACK when the augmentations are dropped. Unset — the
    # cluster — this block does nothing and no server run changes.
    from experiments._shared import _local_aug_conditions
    _aug_kept = _local_aug_conditions(['none', 'uncapped', 'capped'])
    if augs and 'uncapped' not in _aug_kept:
        print(f"  [local] KIRBY_LOCAL_AUG_CONDITIONS keeps {_aug_kept} and c06 "
              f"asked for 'uncapped', so the augmentation representations are "
              f"dropped from every search here. All {n_samples} antibodies are "
              f"still used; the augmentation question has no answer in this "
              f"run.")
        augs = None
    results_aug_conditions_kept = list(_aug_kept)

    aug_strategy = 'uncapped' if augs else 'none'

    # Family-level canonical gate via _resolve_models (LIST form).
    # Replaces the old hardcoded `_FAMILY_MAX_SAMPLES = {'GP': 2000, 'SVM': 10000}`
    # which diverged from DEFAULT_MODELS (gp: 1000, svm: 10000) and didn't
    # inherit domain/dataset skips.
    for fam_name in list(all_families.keys()):
        if fam_name not in _FAMILY_TO_CANONICAL:
            raise ValueError(
                f"family {fam_name!r} has no entry in _FAMILY_TO_CANONICAL. "
                f"Add one, or map it to None if no DEFAULT_MODELS name "
                f"matches it. Falling through silently is how Ridge went "
                f"stale for two months.")
        canonical = _FAMILY_TO_CANONICAL[fam_name]
        if canonical is None:
            print(f"  {fam_name}: no canonical DEFAULT_MODELS name, so "
                  f"_max_samples / _DOMAIN_MODEL_SKIPS / "
                  f"_DATASET_STRATEGY_SKIPS are NOT applied to it")
            continue
        # Pass strategy so _DATASET_STRATEGY_SKIPS / _DOMAIN_STRATEGY_SKIPS
        # are applied (e.g., 'gp' skipped for flab_thermo_tresanco/uncapped).
        # This gate stays at speed='full' (the default) even though the
        # searches below run fast, 2026-09-12: _resolve_models under 'fast'
        # keeps only the models labelled tier 'fast' — rf and lightgbm — so a
        # fast gate would delete six of the seven families. The searches are
        # unaffected because c06 hands find_best_hybrid a models DICT, which
        # never goes through that tier filter.
        resolved = _resolve_models(
            [canonical], task_type='regression', 
            n_samples=n_samples, dataset_name=dataset_name,
            strategy=aug_strategy)
        if not resolved:
            print(f"  Skipping {fam_name}: gated by _resolve_models "
                  f"(canonical={canonical}, n_samples={n_samples}, "
                  f"strategy={aug_strategy})")
            del all_families[fam_name]

    # Fetch precompute's importance_cache once for the dataset and reuse
    # across all per-variant find_best_hybrid calls. Skips the per-fold
    # importance rebuild (lightgbm_gain × n_reps × n_folds) inside each
    # variant search. Splits align because c06 passes seed=42 + same
    # n_folds + same task_type as precompute. If precompute SearchResult
    # is missing/empty, falls back to building importance from scratch
    # (find_best_hybrid handles `importance_cache=None` as the no-op default).
    _precompute_importance = None
    try:
        from experiments._shared import get_precomputed_search
        _pc_search = get_precomputed_search(
            reps, labels, groups,
            # Fetch the cache that matches the actual augmentation condition
            # we'll be using (otherwise we may fall through to a from-scratch
            # find_best_hybrid run when the 'none' cache doesn't exist).
            aug_condition=aug_strategy, augmentations=augs,
            task_type='regression', metric=metric_name,
            dataset_name=dataset_name, tier='general',
            # The importance scores are keyed by fold index. Ask for them at
            # the fold count c06 itself runs at (1 with no --mode flag, 5 under
            # --mode full), so fold 0's scores were fitted on fold 0's training
            # rows and not on some other partition's.
            n_folds=n_folds, seed=42,
        )
        _pc_imp = getattr(_pc_search, 'importance_cache', None) or {}
        # A cache built at a different fold count puts rows this run validates
        # on inside the training rows behind (fold 0, method). Its fold indices
        # must cover exactly the folds this run will ask for.
        _cache_folds = set()
        for _k in _pc_imp:
            if isinstance(_k, tuple) and _k and isinstance(_k[0], int):
                _cache_folds.add(_k[0])
        if _pc_imp and _cache_folds != set(range(n_folds)):
            print(f"  Discarding precompute importance_cache: it holds folds "
                  f"{sorted(_cache_folds)}, this run uses {n_folds}. Importance "
                  f"will be rebuilt per fold.")
            _pc_imp = {}
        if _pc_imp:
            _precompute_importance = _pc_imp
            print(f"  Reusing precompute importance_cache: "
                  f"{len(_precompute_importance)} (fold, method) entries "
                  f"at n_folds={n_folds}")
    except Exception as _e:
        print(f"  Could not load precompute importance_cache: {_e}")

    results = {'families': {},
               'augmentation_conditions_kept': results_aug_conditions_kept}

    # KIRBY_C06_VARIANTS_PER_FAMILY: how many capacity settings of each family
    # are searched. Unset (the shipped value) every family keeps all four, so
    # the server is unchanged. Set to k, each family keeps k settings evenly
    # spaced along its capacity index INCLUDING both ends, so the gradient
    # still spans the family's full capacity range — the lowest-capacity and
    # highest-capacity variant are always kept.
    #
    # What it costs: the capacity-vs-optimal-budget rho below is computed over
    # k points instead of 4, and the code refuses to compute it at all below 3
    # points ('insufficient_variants'). So 3 is the floor. It is a CONDITION
    # cut: every variant still trains on every molecule.
    _vpf = os.environ.get('KIRBY_C06_VARIANTS_PER_FAMILY', '').strip()
    if _vpf:
        _k = int(_vpf)
        if _k < 3:
            raise ValueError(
                f"KIRBY_C06_VARIANTS_PER_FAMILY={_k}: spearmanr over capacity "
                f"and optimal budget needs at least 3 variants, and c06 records "
                f"'insufficient_variants' below that. Use 3 or more.")
        for _fam in list(all_families):
            _vs = all_families[_fam]
            if len(_vs) > _k:
                _idx = sorted({round(i * (len(_vs) - 1) / (_k - 1))
                               for i in range(_k)})
                _kept = [_vs[i] for i in _idx]
                print(f"  {_fam}: {len(_vs)} capacity settings cut to {_k} "
                      f"({[v[0] for v in _kept]}); "
                      f"dropped {[v[0] for v in _vs if v not in _kept]}")
                all_families[_fam] = _kept

    for fam_name, variants in all_families.items():
        # DECIDED 2026-09-12: the family's own evaluator name goes in the list
        # c06 passes. Printed per family so the run says which names it asked
        # for, including the families that have no plain model name to pass.
        fam_evaluators = _family_evaluators(fam_name)
        print(f"\n  Family: {fam_name} ({len(variants)} variants, "
              f"speed={_C06_SPEED}, internal evaluators={fam_evaluators})")
        if fam_name not in _FAMILY_TO_EVALUATOR:
            print(f"    {fam_name} has no plain model name among the internal "
                  f"evaluators hybrid.py can fit, so no matched evaluator is "
                  f"searched for it.")
        fam_result = {'heatmap': {}, 'optimal_budget': {},
                      'internal_evaluators': list(fam_evaluators),
                      'speed': _C06_SPEED}

        for var_name, var_factory, capacity_idx in variants:
            print(f"\n    Variant: {var_name} (capacity={capacity_idx})")
            _t_var = _time.time()
            fam_result['heatmap'][var_name] = {}

            # Family-unique downstream-model name. The result cache keys on the
            # model name, and variant names like 'C=0.01' repeat across families
            # (Logit vs SVM on clf data) — without the family prefix two families'
            # rows would collide on one key and overwrite each other.
            model_key = f"{fam_name}:{var_name}"

            # Per-variant search: this variant is the SOLE downstream model.
            # find_best_hybrid searches its full grid (alloc × imp × budget)
            # against this specific capacity. seed=42 matches precompute so
            # CV splits align with any reused cache state.
            try:
                search = find_best_hybrid(
                    base_reps=reps,
                    labels=labels,
                    groups=groups,
                    models={model_key: _hold_variant_setting(
                        _apply_model_njobs(var_factory(), fam_name, var_name),
                        fam_name, var_name)},
                    budgets=eff_budgets,
                    task_type='regression',
                    metric=metric_name,
                    seed=42,
                    n_folds=n_folds,
                    cache=True,
                    # Use real dataset_name + condition_suffix so the
                    # _DATASET_STRATEGY_SKIPS lookup still matches (the
                    # mangled dataset_name would never hit those entries).
                    dataset_name=dataset_name,
                    condition_suffix=f"c06_{fam_name}_{var_name}",
                    tier='c06_variant',
                    augmentations=augs,
                    augmentation_strategy=aug_strategy,
                    # DECIDED 2026-09-12: the family's evaluator name, in the
                    # list c06 passes, in place of 'match'. Was
                    # INTERNAL_EVALUATORS, whose 'match' resolved to nothing
                    # against the family-prefixed downstream model name.
                    internal_evaluators=fam_evaluators,
                    # DECIDED 2026-09-12: fast tier. It picks the allocation
                    # and importance lists; the downstream model comes in as a
                    # dict here, so fast's model-tier filter never touches it.
                    speed=_C06_SPEED,
                    verbose=False,
                    importance_cache=_precompute_importance,
                )
            except SystemExit as e:
                # SystemExit is NOT an Exception, so the handler below does not
                # see it. A library that calls sys.exit() inside a search --
                # AntiFold's loader does -- would otherwise walk straight past
                # this loop and kill the process with no traceback and no line
                # saying which variant it died on, taking the other six
                # families with it. Same remedy as complexity_helpers.py:1240.
                print(f"      search failed: a library called sys.exit({e.code}) "
                      f"instead of raising. Recorded; the remaining variants "
                      f"still run.")
                import traceback
                traceback.print_exc()
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'score': float('nan'),
                    'capacity_index': capacity_idx,
                    'error': f'SystemExit({e.code})',
                    'error_kind': 'SystemExit',
                    'evaluator': None, 'evaluator_label': None,
                    'fold_scores': None,
                    'n_folds_scored': None, 'n_folds_asked': None,
                }
                continue
            except Exception as e:
                print(f"      search failed: {e}")
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'score': float('nan'),
                    'capacity_index': capacity_idx,
                    # The exception text, 2026-10-05; it was printed only.
                    'error': f'{type(e).__name__}: {e}',
                    'error_kind': type(e).__name__,
                    # Same keys as a scored variant, 2026-09-12, so a reader
                    # never has to tell an absent key from an absent number.
                    'evaluator': None, 'evaluator_label': None,
                    'fold_scores': None,
                    'n_folds_scored': None, 'n_folds_asked': None,
                }
                continue

            print(f"      search done in {_time.time() - _t_var:.1f}s")
            per_budget = search.best_by_model_budget.get(model_key, {})
            per_budget_val = {}
            # Iterate keys actually present (find_best_hybrid may append n_samples
            # as an at-n boundary point even when it wasn't in eff_budgets).
            for budget in sorted(per_budget.keys()):
                best = per_budget.get(budget)
                if best is None:
                    continue
                bp = best['blueprint']
                val_score = best['score_mean']
                val_std = best.get('score_std', float('nan'))
                n_feat = best.get('n_features', 0)
                evaluator = best.get('internal_evaluator',
                                     getattr(bp, 'internal_evaluator', None))
                # DECIDED 2026-09-12: write the per-fold scores out. Each cell
                # held a mean and a standard deviation over folds and not the
                # fold values themselves. The values already sit on the
                # search's own result row, keyed 'fold_scores'
                # (find_best_hybrid.py:4808), so storing them costs no extra
                # model fits. A row rebuilt from the result database can carry
                # a mean with no folds behind it; that stores as None here
                # rather than as an empty list of folds.
                fold_scores = best.get('fold_scores')
                # 'val_score' was 'val_r2' until 2026-10-05. It holds the
                # dataset's metric (spearman, rmse, ...), never R-squared. No
                # reader of c06 output reads the old name (c09's c06 branch,
                # the only one, is updated to the new one).
                cell = {
                    'val_score': val_score,
                    'val_std': val_std,
                    'n_features': n_feat,
                    'importance': bp.importance_method,
                    'allocation': bp.allocation_method,
                    # DECIDED 2026-09-12: the matched evaluator is filed under
                    # its resolved name, so SVM downstream with the matched
                    # evaluator reads 'SVM/SVM'.
                    'evaluator': evaluator,
                    'evaluator_label': _evaluator_label(fam_name, evaluator),
                    'fold_scores': ([float(s) for s in fold_scores]
                                    if fold_scores else None),
                    'n_folds_scored': best.get('n_folds_scored'),
                    'n_folds_asked': best.get('n_folds_asked'),
                }
                fam_result['heatmap'][var_name][budget] = cell
                per_budget_val[budget] = (val_score, bp, cell)
                print(f"      b={budget}: {metric_name}={val_score:.4f}±{val_std:.3f} "
                      f"[{bp.importance_method}/{bp.allocation_method}/"
                      f"{cell['evaluator_label']}]")

            if per_budget_val:
                cmp_fn = min if lower_is_better else max
                opt_budget, (opt_val, opt_bp, opt_cell) = cmp_fn(
                    per_budget_val.items(), key=lambda kv: kv[1][0])
                fam_result['optimal_budget'][var_name] = {
                    'budget': opt_budget,
                    # 'score' was 'r2' until 2026-10-05; the dataset's metric.
                    'score': opt_val,
                    'capacity_index': capacity_idx,
                    'importance': opt_bp.importance_method,
                    'allocation': opt_bp.allocation_method,
                    # DECIDED 2026-09-12: the evaluator under its resolved name
                    # and the per-fold scores, on the peak budget's cell too.
                    # The peak budget is an argmax over per-cell means; the
                    # folds behind the winning mean are these.
                    'evaluator': opt_cell['evaluator'],
                    'evaluator_label': opt_cell['evaluator_label'],
                    'fold_scores': opt_cell['fold_scores'],
                    'n_folds_scored': opt_cell['n_folds_scored'],
                    'n_folds_asked': opt_cell['n_folds_asked'],
                }
                print(f"      >> optimal: b={opt_budget}, {metric_name}={opt_val:.4f} "
                      f"[{opt_bp.importance_method}/{opt_bp.allocation_method}/"
                      f"{opt_cell['evaluator_label']}]")
            else:
                _why = _no_score_reason(search, model_key)
                print(f"      no budget scored: {_why['reason']}")
                fam_result['optimal_budget'][var_name] = {
                    'budget': None, 'score': float('nan'),
                    'capacity_index': capacity_idx,
                    # Same keys as a scored variant, 2026-09-12, so a reader
                    # never has to tell an absent key from an absent number.
                    'evaluator': None, 'evaluator_label': None,
                    'fold_scores': None,
                    'n_folds_scored': None, 'n_folds_asked': None,
                    # Why no budget has a score, 2026-10-05.
                    'error': _why['reason'],
                    'n_configurations': _why['n_configurations'],
                    'n_configurations_finite': _why['n_configurations_finite'],
                }

        # Correlation: capacity_index vs optimal_budget within family.
        capacities = []
        opt_budgets = []
        for var_name, info in fam_result['optimal_budget'].items():
            if info.get('budget') is not None:
                capacities.append(info['capacity_index'])
                opt_budgets.append(info['budget'])

        # spearmanr is NaN unless BOTH variables vary. On small/low-signal data
        # every capacity variant can optimise at the same budget -> no variance
        # -> NaN. Distinguish that genuine "no budget variance" case from a real
        # measured correlation so the result is interpretable (not a bare NaN).
        n_distinct_budgets = len(set(opt_budgets))
        # rho only; the p-value spearmanr also returns is not stored or
        # printed (removed 2026-10-05, owner's rule: no significance tests).
        if len(capacities) >= 3 and n_distinct_budgets >= 2:
            rho = spearmanr(capacities, opt_budgets)[0]
            fam_result['capacity_budget_correlation'] = {
                'rho': float(rho),
                'n_distinct_budgets': n_distinct_budgets,
                'n_variants': len(capacities),
                'status': 'measured',
            }
            print(f"\n  >> {fam_name}: capacity vs optimal_budget: "
                  f"rho={rho:.3f} over {len(capacities)} variants")
        else:
            status = ('insufficient_variants' if len(capacities) < 3
                      else 'no_budget_variance')
            fam_result['capacity_budget_correlation'] = {
                'rho': float('nan'),
                'n_distinct_budgets': n_distinct_budgets,
                'n_variants': len(capacities),
                'status': status,
            }
            print(f"\n  >> {fam_name}: capacity vs optimal_budget UNDEFINED "
                  f"({status}: {n_distinct_budgets} distinct budget(s) over "
                  f"{len(capacities)} variants)")

        results['families'][fam_name] = fam_result

    # Summary across all families.
    print("\n  SUMMARY: optimal budget by variant")
    for fam_name, fam_result in results['families'].items():
        for var_name, info in fam_result['optimal_budget'].items():
            b = info.get('budget', '?')
            r2 = info.get('score', float('nan'))
            imp = info.get('importance', '?')
            ev = info.get('evaluator_label') or '?'
            n_f = len(info.get('fold_scores') or [])
            print(f"    {fam_name}/{var_name}: optimal_budget={b}, "
                  f"{metric_name}={r2:.4f} [{imp}/{ev}] "
                  f"over {n_f} fold score(s)")

    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C06: Within-Family Complexity Gradient',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--families', type=str, nargs='+', default=None,
                        help='Model families to test (RF, Lasso, kNN, DNN, '
                             'XGBoost, SVM, GP). Ridge was replaced by Lasso '
                             'on 2026-07-18.')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, families=args.families)
    save_experiment_results(results, args.output, 'c06_within_family_gradient', args.dataset)
