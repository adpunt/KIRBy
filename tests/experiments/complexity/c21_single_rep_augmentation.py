#!/usr/bin/env python3
"""C21: Single-Rep Augmentation Test.

Paper: Complexity
Question: a top-tier standalone rep already does well and hybridisation adds
    little. Can a handful of INDIVIDUAL extra features nudge it further?

Design: for each base rep in DATASET_REPS[dataset]['general'] (the 10-rep pool):
    A_baseline      base rep alone
    B_aug           base + the dataset's augmentation-category reps, uncapped.
                    These are the small engineered feature sets the surveys
                    assigned to 'augmentation' (PIPELINE_READINESS_TODO.md).
    C_topreps       base + the top-N features of every OTHER pool rep,
                    each capped to N by per-fold RF importance.
    D_both          B and C together (aug reps uncapped, pool reps capped to N).

    E_hybrid          the c02 pool hybrid (reference; reuses precompute)
    E_hybrid_aug      that hybrid + the augmentation features (reuses precompute)

NO SEARCH. Every B/C/D condition constructs the blueprint directly — one base
rep gets 100% of the budget at its full native dimension, and augment_uncapped
adds candidate features ONE AT A TIME, keeping each only if the downstream score
improves. Allocation, importance and internal-evaluator never enter: with a
single base rep they cannot change the selection. The E_* reference arms do not
search either — they read the existing tier='top' precompute (the same result
c02 uses) and re-score it on these folds.

Conditions are skipped, not faked, when their pool is empty: on datasets whose
survey found no augmentation rep helps, `augmentation` is legitimately [] and
B/D do not run. C still runs, and is the condition that answers "can other top
reps donate a few good features".

Usage:
    python c21_single_rep_augmentation.py --dataset herg --mode intermediary
    python c21_single_rep_augmentation.py --dataset abcov_kd --top-n 5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, cv_split, get_domain_metric,
    build_fold_hybrid)
from kirby.find_best_hybrid import Blueprint
from experiments.complexity.complexity_helpers import DATASET_REPS

# Floor on the denominator of relative_improvement (= delta / |base_score|).
# On antibody tasks the base Spearman rho routinely lands near zero, and an
# unfloored |base_score| manufactures absurd relative improvements (e.g. a
# base rho of 0.004 turned a modest +0.12 delta into a bogus +27.8). Below
# this magnitude the base score is not a meaningful signal, so we clamp the
# denominator to keep the ratio finite and interpretable. Base scores at or
# above the floor are unaffected.
REL_IMPROVEMENT_DENOM_FLOOR = 0.1

# Features taken from each OTHER top_tier rep, chosen per fold by RF importance.
DEFAULT_TOP_N = 5

# Downstream-model seeds averaged within each fold before any delta is taken.
# Adding columns changes which features a stochastic tree model sees, which
# moves the score by itself: on tresanco, permuting the column order of an
# otherwise IDENTICAL 48-column block moved a fold Spearman across a 0.032
# range — larger than any effect this experiment is trying to detect. One seed
# is therefore not a measurement.
DEFAULT_N_SEEDS = 3

# Fixed recipe for the multi-rep reference arms (H0/H1). NOT searched: the
# previous E_* arms picked a blueprint by argmax over the same folds they were
# then scored on, which gave the hybrid reference a winner's curse the
# fixed-config single-rep arms do not have, and made the two incomparable.
# 'weighted_global' pools features across reps and ranks once globally, so a
# large near-tied pool does not shatter the budget into per-rep dust.
HYBRID_REF_ALLOCATION = 'weighted_global'
HYBRID_REF_IMPORTANCE = 'lightgbm_gain'
HYBRID_REF_BUDGETS = (500, 1000, 2000)
# Pre-registered headline budget, fixed BEFORE seeing any result so that
# quoting it is not a selection.
HYBRID_REF_HEADLINE_BUDGET = 1000


def _seeded(factory, seed):
    """Return a fresh model with its random_state pinned to `seed` when it has one."""
    m = factory()
    try:
        if 'random_state' in m.get_params():
            m.set_params(random_state=seed)
    except Exception:
        pass
    return m


def _fit_score_seeds(Xtr, ytr, Xva, yva, factory, metric_fn, n_seeds):
    """Mean score over n_seeds model seeds on ONE fold.

    The averaging happens before any delta is taken, so model stochasticity
    does not masquerade as an effect of the added columns.
    """
    vals = []
    for s in range(n_seeds):
        m = _seeded(factory, 42 + s)
        m.fit(Xtr, ytr)
        vals.append(float(metric_fn(yva, m.predict(Xva))))
    return float(np.mean(vals)), vals


def _build_conditions(base_name, top_tier_reps, aug_reps, top_n, rng=None,
                      with_controls=True):
    """Return {condition_name: (aug_pool, per_source_budget)}.

    per_source_budget is a dict so the wide top-tier reps get capped to top_n
    while the small engineered augmentation reps stay uncapped — a source
    absent from the dict contributes all of its columns.

    Two matched-width CONTROLS are added when with_controls is set. Adding any
    columns changes model capacity, so without them a win in B/C/D cannot be
    told apart from "the model simply got more features":

      R_rand_n{top_n}  offers top_n columns picked UNIFORMLY AT RANDOM from each
                       other base rep, instead of its top_n by importance. Same
                       width as C, same donors, same admission path — so C minus
                       R isolates the SELECTION, not the capacity.
      R_randaug        offers as many randomly-chosen columns (drawn from the
                       other base reps) as the curated augmentation pool has.
                       Same width as B, so B minus R_randaug isolates whether
                       the curated blocks carry something the base pool does not.

    Random columns carry no label information, so drawing them once outside the
    fold loop leaks nothing.
    """
    other_top = {k: v for k, v in top_tier_reps.items() if k != base_name}
    aug_only = {k: v for k, v in aug_reps.items() if k != base_name}

    conditions = {'A_baseline': (None, None)}
    if aug_only:
        conditions['B_aug'] = (aug_only, None)
    if other_top:
        conditions[f'C_topreps_n{top_n}'] = (
            other_top, {k: top_n for k in other_top})
    if aug_only and other_top:
        both = {**aug_only, **other_top}
        conditions[f'D_both_n{top_n}'] = (
            both, {k: top_n for k in other_top})

    if not with_controls or not other_top:
        return conditions
    rng = rng if rng is not None else np.random.RandomState(0)

    # Control for C: same donors, same 5-per-donor width, random columns.
    rand_pool = {}
    for k, v in other_top.items():
        n_take = min(top_n, v.shape[1])
        cols = rng.choice(v.shape[1], size=n_take, replace=False)
        rand_pool[f'{k}__rand'] = v[:, np.sort(cols)]
    conditions[f'R_rand_n{top_n}'] = (rand_pool, None)

    # Control for B: as many random columns as the curated pool is wide,
    # spread evenly over the other base reps.
    if aug_only:
        aug_width = sum(v.shape[1] for v in aug_only.values())
        per_donor = max(1, int(np.ceil(aug_width / len(other_top))))
        randaug, taken = {}, 0
        for k, v in other_top.items():
            if taken >= aug_width:
                break
            n_take = min(per_donor, v.shape[1], aug_width - taken)
            cols = rng.choice(v.shape[1], size=n_take, replace=False)
            randaug[f'{k}__randaug'] = v[:, np.sort(cols)]
            taken += n_take
        if randaug:
            conditions['R_randaug'] = (randaug, None)
    return conditions


def _score_hybrid_condition(cond_name, top_tier_reps, aug_dict, aug_condition,
                            labels, groups, models, metric_fn, lower_is_better,
                            n_folds, mode, data, dataset_name, task_type,
                            metric_name, budgets=HYBRID_REF_BUDGETS,
                            n_seeds=DEFAULT_N_SEEDS):
    """Score a MULTI-rep hybrid on the same folds as the single-rep arms.

    FIXED CONFIGURATION, NO SEARCH. The blueprint is constructed, not fetched.

    The previous implementation pulled a blueprint from get_precomputed_search
    and took `search.best_by_model` — an argmax over the same folds the arm was
    then scored on. That gave the hybrid reference a winner's curse the
    fixed-config single-rep arms (A/B/C/D) do not carry, so the two could not be
    compared, which is the whole point of having the reference. It also required
    a precomputed search that does not exist locally; without one it silently
    fell back to a full find_best_hybrid over every budget.

    Here every budget in `budgets` is a separate fixed arm and all of them are
    reported. Nothing is selected, so each fold mean is unbiased.
    """
    rows = []
    # cv_split forwards X to sklearn's split(X, y, groups) — it only needs the
    # row count, but it must be an array. Any base rep will do; all share the
    # sample axis, so the folds match the single-rep arms exactly.
    X_for_split = next(iter(top_tier_reps.values()))

    for budget in budgets:
        bp = Blueprint(
            reps=sorted(top_tier_reps),
            allocation_method=HYBRID_REF_ALLOCATION,
            importance_method=HYBRID_REF_IMPORTANCE,
            internal_evaluator='lightgbm',
            budget=budget,
            step_size=5,
            augmentation_strategy=aug_condition,
            aug_per_source_budget=None,
            metric=metric_name,
            task_type=task_type,
            lower_is_better=lower_is_better)
        arm = f'{cond_name}_b{budget}'

        for model_name, model_factory in models.items():
            scores, dims, kept = [], [], set()
            for train_idx, val_idx in cv_split(groups, n_folds, X_for_split,
                                               labels, mode=mode):
                y_train, y_val = labels[train_idx], labels[val_idx]
                tr = {k: v[train_idx] for k, v in top_tier_reps.items()}
                va = {k: v[val_idx] for k, v in top_tier_reps.items()}
                tr_aug = ({k: v[train_idx] for k, v in aug_dict.items()}
                          if aug_dict else None)
                va_aug = ({k: v[val_idx] for k, v in aug_dict.items()}
                          if aug_dict else None)
                try:
                    Xtr, Xva, finfo = build_fold_hybrid(
                        tr, va, y_train, bp,
                        augmentations_train=tr_aug, augmentations_val=va_aug)
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(Xtr)
                    Xva = sc.transform(Xva)
                    mean_s, _ = _fit_score_seeds(Xtr, y_train, Xva, y_val,
                                                 model_factory, metric_fn, n_seeds)
                    scores.append(mean_s)
                    dims.append(int(Xtr.shape[1]))
                    kept.update((finfo.get('augmentation_info') or {}).keys())
                except Exception as e:
                    print(f"    {arm}/{model_name}: fold failed - {e}")
                    scores.append(float('nan'))

            ok = not np.all(np.isnan(scores))
            rows.append({
                'condition': arm,
                'condition_family': cond_name,
                'budget': budget,
                'headline': budget == HYBRID_REF_HEADLINE_BUDGET,
                'model': model_name,
                'base_rep': f'HYBRID({len(top_tier_reps)} reps)',
                'base_tier': 'multi_rep_hybrid_fixed',
                'hybrid_mean': float(np.nanmean(scores)) if ok else float('nan'),
                'hybrid_std': float(np.nanstd(scores)) if ok else float('nan'),
                'hybrid_scores': scores,
                'mean_dim': float(np.mean(dims)) if dims else float('nan'),
                'kept_augmentations': sorted(kept),
                'aug_status': 'ok' if ok else 'all_folds_failed',
                'hybrid_reps': sorted(top_tier_reps),
                'aug_pool_reps': sorted(aug_dict) if aug_dict else [],
                'n_seeds': n_seeds,
                'blueprint': bp.to_dict(),
            })
            if ok:
                star = ' *headline' if budget == HYBRID_REF_HEADLINE_BUDGET else ''
                print(f"    {arm}/{model_name}: {np.nanmean(scores):.4f} "
                      f"(dim~{np.mean(dims):.0f}, kept={sorted(kept)}){star}")
    return rows


def run(data, config, output_dir, dataset_name, top_n=DEFAULT_TOP_N,
        base_reps=None, model_names=None, n_seeds=DEFAULT_N_SEEDS,
        hybrid_budgets=HYBRID_REF_BUDGETS, with_controls=True,
        skip_single_rep=False, **kwargs):
    """C21: Single-Rep Augmentation Test.

    base_reps: restrict the base-rep loop to these names. Default (None) is
        every rep in the pool, which is a survey: 10 bases x 3 models x 5
        conditions x folds, each non-baseline cell paying up to ~50 sequential
        model fits inside the greedy augmentation loop. Pin it to the dataset's
        best rep to get the best-vs-best comparison at ~1/10th the cost.
    model_names: restrict the downstream models. Default is the historical
        ['rf', 'knn', 'xgboost'].
    n_seeds: downstream-model seeds averaged within each fold (see
        DEFAULT_N_SEEDS for why one is not enough).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C21: SINGLE-REP AUGMENTATION")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")

    model_names = list(model_names) if model_names else ['rf', 'knn', 'xgboost']
    models = get_model_factories(model_names, task_type=task_type, n_samples=n_samples)
    print(f"  Models: {list(models.keys())}  (x{n_seeds} seeds averaged per fold)")

    ds_config = DATASET_REPS.get(dataset_name, {})
    # The 10-rep pool (the retired 'top_tier' == 'general'); var name kept.
    top_tier_names = ds_config.get('general', [])
    aug_rep_names = ds_config.get('augmentation', [])

    # Augmentation reps: prefer the ones run_all already loaded. complexity
    # run_all pops aug-only reps out of data['reps'] (so find_best_hybrid never
    # sees a rep in both the base and augmentation sets) and parks them in
    # data['augmentations']. Re-deriving them from data['reps'] would silently
    # find nothing there.
    aug_reps = dict(data.get('augmentations') or {})

    # Base reps are TOP TIER. c21 asks "can the BEST rep be nudged further" —
    # the general pool is the hybrid search pool, not the answer to that.
    missing = [r for r in top_tier_names if r not in reps]
    missing += [r for r in aug_rep_names if r not in reps and r not in aug_reps]
    if missing:
        print(f"  Loading {len(missing)} reps not already in memory: {missing}")
        extra = load_data(dataset_name, rep_preset='all', rep_names=missing)
        for rname, arr in extra['reps'].items():
            if rname in aug_rep_names:
                aug_reps[rname] = arr
            else:
                reps[rname] = arr

    top_tier_reps = {r: reps[r] for r in top_tier_names if r in reps}
    aug_reps = {r: v for r, v in aug_reps.items() if r in aug_rep_names}

    if not top_tier_reps:
        print(f"  No top_tier reps available for {dataset_name} — skipping C21")
        return {'error': 'no_top_tier_reps', 'dataset': dataset_name}

    print(f"  Base reps (top_tier, {len(top_tier_reps)}): "
          f"{', '.join(f'{k}={v.shape[1]}D' for k, v in top_tier_reps.items())}")
    if aug_reps:
        print(f"  Augmentation reps ({len(aug_reps)}): "
              f"{', '.join(f'{k}={v.shape[1]}D' for k, v in aug_reps.items())}")
    else:
        # Not a defect: the survey tested augmentation on this dataset and none
        # helped, so PIPELINE_READINESS_TODO records "Augmentation | none".
        print(f"  Augmentation reps: NONE configured for {dataset_name} "
              f"(survey found none helped) — B/D conditions will be skipped")
    print(f"  Top-N features taken from each other top_tier rep: {top_n}")

    # Pin the base-rep loop when asked. Unpinned this is a 10-rep survey; the
    # goal is best-rep x best-model, so normally exactly one name comes in here.
    if base_reps:
        wanted = [r for r in base_reps if r in top_tier_reps]
        missing_bases = [r for r in base_reps if r not in top_tier_reps]
        if missing_bases:
            raise ValueError(
                f"--base-reps named {missing_bases} which are not in "
                f"{dataset_name}'s pool ({sorted(top_tier_reps)}).")
        base_loop = {r: top_tier_reps[r] for r in wanted}
        print(f"  Base reps PINNED to {wanted} "
              f"(pool has {len(top_tier_reps)}; donors still come from all of it)")
    else:
        base_loop = top_tier_reps

    rng = np.random.RandomState(0)
    rep_results = []

    if skip_single_rep:
        print("  Single-rep arms SKIPPED (skip_single_rep) — "
              "reference arms only")
        base_loop = {}

    for base_name, X_base in base_loop.items():
        base_dim = X_base.shape[1]
        print(f"\n  {base_name} ({base_dim}D)")

        conditions = _build_conditions(base_name, top_tier_reps, aug_reps, top_n,
                                       rng=rng, with_controls=with_controls)
        if len(conditions) == 1:
            print(f"    only baseline available (no aug reps, no other top reps)")

        for cond_name, (aug_dict, per_source) in conditions.items():
            is_baseline = aug_dict is None

            # No search. With one base rep the recipe is fully determined:
            # 100% of the budget goes to that rep, budget = its full native
            # dimension (so every base feature is kept), and the only real work
            # is augment_uncapped adding candidate features one at a time. So we
            # CONSTRUCT the blueprint rather than searching for it. Allocation /
            # importance / internal_evaluator cannot change a single-rep
            # selection, so their values here are inert placeholders.
            bp = None
            if not is_baseline:
                bp = Blueprint(
                    reps=[base_name],
                    allocation_method='performance_weighted',  # 1 rep -> 100%
                    importance_method='lightgbm_gain',         # inert at full budget
                    internal_evaluator='match',                # inert (no search)
                    budget=base_dim,                           # keep all base features
                    step_size=5,
                    augmentation_strategy='uncapped',
                    aug_per_source_budget=per_source,
                    metric=metric_name,
                    task_type=task_type,
                    lower_is_better=lower_is_better)

            for model_name, model_factory in models.items():
                # Feature SELECTION is model-independent here (build_fold_hybrid's
                # augment_uncapped runs with create_hybrid's internal default),
                # so every downstream model shares one blueprint. The downstream
                # model only enters at SCORING below.
                bp_model = bp

                scores_base = []
                scores_aug = []
                fold_aug_info = []

                for fold_idx, (train_idx, val_idx) in enumerate(
                    cv_split(groups, n_folds, X_base, labels, mode=mode)
                ):
                    y_train, y_val = labels[train_idx], labels[val_idx]

                    scaler_b = StandardScaler()
                    Xtr_b = scaler_b.fit_transform(X_base[train_idx])
                    Xva_b = scaler_b.transform(X_base[val_idx])
                    mean_b, _ = _fit_score_seeds(Xtr_b, y_train, Xva_b, y_val,
                                                 model_factory, metric_fn, n_seeds)
                    scores_base.append(mean_b)

                    if is_baseline or bp is None:
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim})
                        continue

                    train_base = {base_name: X_base[train_idx]}
                    val_base = {base_name: X_base[val_idx]}
                    train_augs = {rn: aug_dict[rn][train_idx] for rn in aug_dict}
                    val_augs = {rn: aug_dict[rn][val_idx] for rn in aug_dict}

                    try:
                        # build_fold_hybrid -> create_hybrid -> augment_uncapped.
                        # per_source rides on the blueprint; passed explicitly
                        # too for clarity. No search anywhere in this path.
                        Xtr_h, Xva_h, feature_info = build_fold_hybrid(
                            train_base, val_base, y_train, bp_model,
                            augmentations_train=train_augs,
                            augmentations_val=val_augs,
                            aug_per_source_budget=per_source)

                        scaler_h = StandardScaler()
                        Xtr_h = scaler_h.fit_transform(Xtr_h)
                        Xva_h = scaler_h.transform(Xva_h)
                        mean_h, _ = _fit_score_seeds(Xtr_h, y_train, Xva_h, y_val,
                                                     model_factory, metric_fn, n_seeds)
                        scores_aug.append(mean_h)

                        aug_info = feature_info.get('augmentation_info', {})
                        # Per-source detail: which rep donated how many columns.
                        kept_detail = {
                            k: int(v.get('n_features', 0))
                            for k, v in (aug_info or {}).items()
                        }
                        n_added = Xtr_h.shape[1] - base_dim
                        fold_aug_info.append({
                            'kept': sorted(kept_detail),
                            'kept_detail': kept_detail,
                            'n_added': max(0, n_added),
                            'final_dim': Xtr_h.shape[1],
                        })
                    except Exception as e:
                        print(f"    {model_name} fold {fold_idx}: augmentation failed - {e}")
                        scores_aug.append(float('nan'))
                        fold_aug_info.append({'kept': [], 'n_added': 0,
                                              'final_dim': base_dim})

                base_mean = float(np.nanmean(scores_base))
                # A search that found no winning config makes every augmented
                # fold NaN. Flag it explicitly rather than emitting a bare NaN
                # aug_mean, which is what nan-saturated the old c21 results.
                if is_baseline:
                    aug_status = 'baseline'
                elif bp is None:
                    aug_status = 'search_no_config'
                elif np.all(np.isnan(scores_aug)):
                    aug_status = 'all_folds_failed'
                else:
                    aug_status = 'ok'

                if aug_status in ('baseline', 'search_no_config', 'all_folds_failed'):
                    aug_mean = float('nan')
                    delta = 0.0 if is_baseline else float('nan')
                else:
                    aug_mean = float(np.nanmean(scores_aug))
                    delta = aug_mean - base_mean

                all_kept = set()
                total_added = []
                for fi in fold_aug_info:
                    all_kept.update(fi['kept'])
                    total_added.append(fi['n_added'])

                result = {
                    'base_rep': base_name,
                    'base_tier': 'top_tier',
                    'condition': cond_name,
                    'model': model_name,
                    'base_dim': base_dim,
                    'base_mean': base_mean,
                    'base_std': float(np.nanstd(scores_base)),
                    'base_scores': scores_base,
                    'aug_mean': aug_mean,
                    'aug_std': float(np.nanstd(scores_aug)) if not is_baseline else 0.0,
                    'aug_scores': scores_aug,
                    'delta': delta,
                    'aug_status': aug_status,
                    # Sign-normalise: positive == improvement regardless of
                    # metric direction. Without this, RMSE/MAE wins are stored
                    # as negative relative_improvement and any downstream filter
                    # on positive sign rejects them.
                    'relative_improvement': (
                        (-1 if lower_is_better else 1) * delta
                        / max(abs(base_mean), REL_IMPROVEMENT_DENOM_FLOOR)
                        if not np.isnan(delta) else float('nan')),
                    'kept_augmentations': sorted(all_kept),
                    'mean_features_added': float(np.mean(total_added)),
                    'fold_aug_info': fold_aug_info,
                    'aug_pool_reps': sorted(aug_dict) if aug_dict else [],
                    'per_source_budget': per_source,
                    'top_n': top_n,
                    'blueprint': bp.to_dict() if bp else None,
                }
                rep_results.append(result)

                sign = -1 if lower_is_better else 1
                if aug_status not in ('ok', 'baseline'):
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  "
                          f"SKIPPED ({aug_status})")
                elif is_baseline:
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}")
                else:
                    status = "+" if sign * delta > 0 else "-" if sign * delta < 0 else "="
                    print(f"    {cond_name}/{model_name}: base={base_mean:.4f}  "
                          f"aug={aug_mean:.4f}  delta={sign * delta:+.4f} [{status}]  "
                          f"kept={sorted(all_kept)}  added={np.mean(total_added):.1f}")

    # ── Multi-rep reference arms: the c02 top-tier hybrid, and that hybrid
    #    plus augmentation features. Scored on the SAME folds and the SAME
    #    models as the single-rep arms, so the numbers compare directly.
    hybrid_results = []
    if len(top_tier_reps) >= 2:
        print(f"\n  HYBRID reference arms ({len(top_tier_reps)} top-tier reps)")
        hybrid_results += _score_hybrid_condition(
            'H0_hybrid', top_tier_reps, None, 'none',
            labels, groups, models, metric_fn, lower_is_better,
            n_folds, mode, data, dataset_name, task_type, metric_name, budgets=hybrid_budgets, n_seeds=n_seeds)
        if aug_reps:
            # USE CASE A: small augmented blocks added to the MULTI-REP hybrid.
            # This is the arm that has never been run anywhere in the tree.
            hybrid_results += _score_hybrid_condition(
                'H1_hybrid_aug', top_tier_reps, aug_reps, 'uncapped',
                labels, groups, models, metric_fn, lower_is_better,
                n_folds, mode, data, dataset_name, task_type, metric_name, budgets=hybrid_budgets, n_seeds=n_seeds)
        else:
            print("    H1_hybrid_aug: skipped (no augmentation reps configured)")
    else:
        print(f"\n  HYBRID reference arms: skipped "
              f"({len(top_tier_reps)} top-tier rep — a hybrid needs >= 2)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Head-to-head: best single base rep vs the c02 hybrid vs every augmented
    # arm, per model. This is the comparison the experiment exists to make.
    sign = -1 if lower_is_better else 1
    print("\n  HEAD-TO-HEAD (higher = better after sign-normalisation)")
    for model_name in models:
        print(f"\n  [{model_name}]")
        base_rows = [r for r in rep_results
                     if r['condition'] == 'A_baseline' and r['model'] == model_name
                     and not np.isnan(r['base_mean'])]
        if base_rows:
            best_base = max(base_rows, key=lambda r: sign * r['base_mean'])
            print(f"    best single base rep : {best_base['base_rep']:<22s} "
                  f"{best_base['base_mean']:+.4f}")
        aug_rows = [r for r in rep_results
                    if r['condition'] != 'A_baseline' and r['model'] == model_name
                    and r.get('aug_status') == 'ok']
        for cond in sorted(set(r['condition'] for r in aug_rows)):
            cr = [r for r in aug_rows if r['condition'] == cond]
            best = max(cr, key=lambda r: sign * r['aug_mean'])
            print(f"    {cond:<21s}: {best['base_rep']:<22s} "
                  f"{best['aug_mean']:+.4f}  (delta {sign * best['delta']:+.4f} "
                  f"vs its own base)")
        for r in hybrid_results:
            if r['model'] == model_name and r.get('aug_status') == 'ok':
                print(f"    {r['condition']:<21s}: {'(hybrid)':<22s} "
                      f"{r['hybrid_mean']:+.4f}")

    # Only aggregate over results that were actually evaluable. Skipped ones
    # carry a NaN delta and would poison the mean / argmax; count them
    # separately so a NaN-saturated run reads as "N skipped" rather than a
    # meaningless NaN mean_delta.
    aug_results_all = [r for r in rep_results if r['condition'] != 'A_baseline']
    aug_results = [r for r in aug_results_all if r.get('aug_status') == 'ok']
    n_skipped = len(aug_results_all) - len(aug_results)
    n_tests = len(aug_results)
    improved = hurt = 0
    mean_delta = 0.0
    if n_skipped:
        skip_reasons = {}
        for r in aug_results_all:
            if r.get('aug_status') != 'ok':
                skip_reasons[r['aug_status']] = skip_reasons.get(r['aug_status'], 0) + 1
        print(f"  Skipped (not evaluable): {n_skipped} ({skip_reasons})")
    if n_tests > 0:
        sign = -1 if lower_is_better else 1
        improved = sum(1 for r in aug_results if sign * r['delta'] > 0)
        hurt = sum(1 for r in aug_results if sign * r['delta'] < 0)
        mean_delta = float(np.mean([sign * r['delta'] for r in aug_results]))
        all_conditions = sorted(set(r['condition'] for r in aug_results))
        print(f"  Total evaluable tests: {n_tests} "
              f"({len(top_tier_reps)} top-tier reps x {len(models)} models x "
              f"{len(all_conditions)} conditions)")
        print(f"  Augmentation improved: {improved}/{n_tests}")
        print(f"  Augmentation hurt:     {hurt}/{n_tests}")
        print(f"  Mean delta:            {mean_delta:+.4f}")

        for cond in all_conditions:
            cond_results = [r for r in aug_results if r['condition'] == cond]
            cond_improved = sum(1 for r in cond_results if sign * r['delta'] > 0)
            cond_mean = float(np.mean([sign * r['delta'] for r in cond_results]))
            best_c = max(cond_results, key=lambda r: sign * r['delta'])
            print(f"  {cond}: {cond_improved}/{len(cond_results)} improved, "
                  f"mean={cond_mean:+.4f}, "
                  f"best={best_c['base_rep']}/{best_c['model']} "
                  f"delta={sign * best_c['delta']:+.4f}")
    else:
        print(f"  No evaluable augmentation results "
              f"(all {len(aug_results_all)} non-baseline tests were skipped).")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'top_n': top_n,
        'base_reps': sorted(top_tier_reps),
        'base_dims': {k: int(v.shape[1]) for k, v in top_tier_reps.items()},
        # The c02 top-tier hybrid and that hybrid + augmentation features,
        # scored on the same folds/models as the single-rep arms.
        'hybrid_results': hybrid_results,
        'aug_reps': sorted(aug_reps),
        'aug_dims': {k: int(v.shape[1]) for k, v in aug_reps.items()},
        'rep_results': rep_results,
        'summary': {
            'n_tests': n_tests,          # evaluable (aug_status == 'ok') only
            'n_skipped': n_skipped,      # search_no_config / all_folds_failed
            'improved': improved if n_tests > 0 else 0,
            'hurt': hurt if n_tests > 0 else 0,
            'mean_delta': mean_delta if n_tests > 0 else 0.0,
            # When every non-baseline test was skipped the experiment produced
            # no augmentation signal — flag it so a NaN-saturated run is not
            # mistaken for a "no improvement" result.
            'all_skipped': (n_tests == 0 and n_skipped > 0),
        },
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C21: Single-Rep Augmentation',
                                      has_delta=False, has_augment=False)
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                        help='Features taken from each other top_tier rep '
                             f'(default {DEFAULT_TOP_N})')
    parser.add_argument('--base-reps', type=str, default=None,
                        help='Comma-separated base reps to use (default: the '
                             'whole pool, which is a 10x more expensive survey). '
                             'Pin this to the dataset best rep.')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated downstream models '
                             '(default: rf,knn,xgboost)')
    parser.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                        help=f'Model seeds averaged per fold (default {DEFAULT_N_SEEDS})')
    parser.add_argument('--hybrid-budgets', type=str,
                        default=','.join(str(b) for b in HYBRID_REF_BUDGETS),
                        help='Comma-separated budgets for the fixed multi-rep '
                             'reference arms H0/H1')
    parser.add_argument('--no-controls', action='store_true',
                        help='Skip the matched-width random-column control arms')
    args = parser.parse_args()
    config = get_config(args)

    _split = lambda s: [x.strip() for x in s.split(',') if x.strip()]
    base_reps = _split(args.base_reps) if args.base_reps else None
    model_names = _split(args.models) if args.models else None
    hybrid_budgets = tuple(int(b) for b in _split(args.hybrid_budgets))

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset, top_n=args.top_n,
                  base_reps=base_reps, model_names=model_names,
                  n_seeds=args.n_seeds, hybrid_budgets=hybrid_budgets,
                  with_controls=not args.no_controls)
    save_experiment_results(results, args.output, 'c21_single_rep_augmentation',
                            args.dataset)
