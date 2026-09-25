#!/usr/bin/env python3
"""N41: flag suspect labels with a controlled false-flag rate, then act on the flags.

Paper: Noise.

WHERE THIS CAME FROM. The third research report (2026-09-25, "Noise Mitigation
... Organised by Noise Type: Dense Gaussian Noise vs Sparse Gross Errors"),
sections B1 and D3 #11-#12:

  - conformal outlier p-values (Bates, Candes, Lei, Romano & Sesia 2023, Annals
    of Statistics 51:149-178) with Benjamini-Hochberg control of the share of
    flags that are false, calibrated on a TRUSTED subset of molecules;
  - isolation forest or LOF on the features joined with the out-of-fold residual;
  - a large residual with LOW disagreement between refits as the sign of a wrong
    label rather than a hard molecule;
  - cleanlab's regression label-quality score;
  - act by winsorising only the flagged molecules, and always beside a
    random-flag control of the same size.

What already existed, read 2026-09-25: n32's `conformal` detector ranks by
|y - pred| / spread and never turns the ranking into p-values, so no share of
false flags is controlled; `LOFFilter` in src/kirby/utils/noise_mitigation_methods.py
runs on the features only and is unused; n09's `ConfidentLearning` combines a
large residual with low ensemble spread but predicts IN-SAMPLE.

THE CONFORMAL P-VALUES. Per training fold:
  r        y - p, p the group-aware out-of-fold LightGBM prediction (oof_predict).
  sigma    a second group-aware out-of-fold LightGBM, fitted to |r|: the model's
           own estimate of how large this molecule's residual usually is. Floored
           at SIGMA_FLOOR_OF_MEDIAN x the median |r|. Chosen over NGBoost because
           it is one more oof_predict call, where NGBoost is 500 boosting stages
           per fold; both give a per-molecule spread, and n32 already grades
           NGBoost's.
  score    |r| / sigma.
  p-value  for a molecule outside the calibration set,
           (1 + number of calibration molecules scoring at least as high)
           / (calibration size + 1).
Benjamini-Hochberg at each q in BH_QS flags molecules outside the calibration
set. The p-values are valid when the calibration molecules are clean and
exchangeable with the clean molecules tested; under that, the expected share of
flags that are false is at most q. A corrupted calibration molecule makes the
p-values LARGER, so the flags fewer, not more.

THE FLOOR ON HOW FEW IT CAN FLAG, found by the smoke 2026-09-25. No p-value is
below 1 / (calibration size + 1), so Benjamini-Hochberg flags either nothing or
at least n_tested / (q (calibration size + 1)) molecules -- about (1 - s) / (s q)
for a calibration share s, whatever the fold size: 40 molecules at s = 0.2 and
q = 0.1. A condition that moves fewer molecules than that, or moves them by too
little to outscore every calibration molecule, gets no flags at all. Every BH
record writes this floor as `fewest_flags_bh_can_make`, so a zero is readable.

THE TWO CALIBRATION SETS.
  random_s<pct>  a random share of the training fold, one set per share in
            CALIBRATION_SHARES (0.2 and 0.5). What any dataset has.
  trusted   herg_ic50 and bindingdb_kd only (any dataset with replicate groups):
            training molecules measured 3+ times whose spread across their own
            measurements -- the corrupted labels, what a user sees -- is in the
            lowest half among those. Refused, with the reason written, when
            fewer than MIN_CALIBRATION molecules qualify.

THE OTHER DETECTORS, graded against the mask with detection_scores beside the
conformal score. None acts on the labels.
  rescaled_residual          the conformal score |r| / sigma itself.
  isolation_forest           on [the fold's features, r]. The residual is
                             z-scored and multiplied by sqrt(number of features),
                             so it carries as much variance as all the
                             standardised features together; unscaled it would be
                             one feature in thousands.
  lof                        the same joined matrix, local outlier factor.
  residual_low_disagreement  |mean of 5 seeds' out-of-fold predictions - y|
                             divided by (their SD across seeds + a floor). High
                             when the refits agree on a prediction far from the
                             label.
  cleanlab                   1 - cleanlab's label quality
                             (noise_helpers.cleanlab_suspicion).

THE ARMS. Every downstream model, every feature space, scored on held-out
molecules against their UNTOUCHED labels.
  clean, no_cleaning
  winsorise_<cal>_q<q>        the flagged molecules' labels moved to
                              p + clip(r, +-c*s), s = 1.4826 * median |r| over
                              the whole training fold (n30's winsorise), c =
                              WINSOR_C; every other label untouched.
  remove_<cal>_q<q>           the flagged molecules dropped.
  ..._random/seed<k>          the same number of molecules, drawn at random from
                              the same candidates (outside the calibration set).
When BH flags nothing, the arm's training set IS no_cleaning's, and its score is
written as no_cleaning's for that pass: a tie, not a missing pass.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. The false-flag share is measured against the injector's mask.

Usage:
    python n41_conformal_fdr_detection.py --dataset logd --mode full
    python n41_conformal_fdr_detection.py --dataset herg_ic50 --mode full \
        --noise-condition sparse_dprime_p05
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import (
    noise_levels, noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    noise_spec_for, inject_train_labels, hybrid_for_fold,
    pool_recipe_per_downstream_model, group_models_by_recipe,
    detection_scores, detection_is_defined, sweep_conditions,
    DETECTION_IS_DEFINED, SHIFT_ONLY_CONDITIONS, SHIFT_ONLY_DELTAS_SD,
    inject_shift_train_labels, RELATIVE_CONDITIONS, relative_levels,
    inject_relative_train_labels, oof_residual_sd, cleanlab_suspicion,
)
from experiments.noise.n32_iterative_and_dual_selection import (
    _fit_score, _paired, recovery_share, _mean_on, _tail_row,
    _mean_detection, replicate_seed,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noisy_labels import oof_predict

EXP_ID = 'n41_conformal_fdr_detection'
DEFAULT_NOISE_CONDITION = 'sparse_dprime_p05'

NOISE_DRAWS = (0, 1, 2)
BH_QS = (0.05, 0.1, 0.2)
# The random calibration set's share of the training fold, swept. Owner's
# decision 2026-09-25: at 0.2 alone Benjamini-Hochberg can flag nothing or at
# least ~(1 - s) / (s q) molecules, 40 at q = 0.1, more than the 2-10% the
# sparse conditions move on many folds. 0.5 brings that to 10 at q = 0.1 and
# tests half the fold. Each share is its own calibration set, named
# random_s<percent>.
CALIBRATION_SHARES = (0.2, 0.5)
MIN_CALIBRATION = 20
TRUSTED_MIN_MEASUREMENTS = 3
SIGMA_FLOOR_OF_MEDIAN = 0.1
WINSOR_C = 1.0
RANDOM_SEEDS = (0, 1, 2)
OOF_SPLITS = 5
DISAGREEMENT_SEEDS = (0, 1, 2, 3, 4)
LOF_NEIGHBOURS = 20
IFOREST_TREES = 200
GAP_MIN_FRAC = 0.02
DETECTORS = ('rescaled_residual', 'isolation_forest', 'lof',
             'residual_low_disagreement', 'cleanlab')
ACTIONS = ('winsorise', 'remove')

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}


# ── Conformal p-values and Benjamini-Hochberg ────────────────────────────────

def conformal_pvalues(cal_scores, test_scores):
    """Split-conformal outlier p-values (Bates et al. 2023, eq. 2).

    p_j = (1 + #{calibration scores >= test score j}) / (n_cal + 1).
    """
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    t = np.asarray(test_scores, dtype=float)
    n_ge = cal.size - np.searchsorted(cal, t, side='left')
    return (1.0 + n_ge) / (cal.size + 1.0)


def benjamini_hochberg(p, q):
    """Boolean mask of the p-values Benjamini-Hochberg rejects at level q."""
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind='stable')
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.max(np.flatnonzero(below)))
        out[order[:k + 1]] = True
    return out


def rescaled_residual(X, y, groups, p, oof_splits, floor_of_median=SIGMA_FLOOR_OF_MEDIAN):
    """(score |r|/sigma, sigma): sigma an out-of-fold LightGBM fit to |r|."""
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    a = np.abs(r)
    sigma = oof_predict(X, a, groups, n_splits=oof_splits)
    floor = max(floor_of_median * float(np.median(a)), 1e-9)
    sigma = np.clip(sigma, floor, None)
    return a / sigma, sigma


def random_cal_name(share):
    """'random_s20' for a share of 0.2."""
    return f'random_s{int(round(100 * float(share)))}'


def random_calibration(n, share, seed):
    cal = np.zeros(n, dtype=bool)
    k = int(round(share * n))
    if k > 0:
        rng = np.random.default_rng(int(seed))
        cal[rng.choice(n, k, replace=False)] = True
    return cal


def trusted_calibration(y_seen, rep_groups_tr,
                        min_measurements=TRUSTED_MIN_MEASUREMENTS):
    """Training rows of molecules measured min_measurements+ times in this fold
    whose spread (SD, ddof=1, of the labels as seen) is in the lowest half.

    Returns (mask, record). The mask is None when there is no replicate
    structure.
    """
    if rep_groups_tr is None:
        return None, {'why_none': 'no replicate groups on this dataset'}
    y_seen = np.asarray(y_seen, dtype=float)
    g = np.asarray(rep_groups_tr)
    rows_by = {}
    for i, gi in enumerate(g):
        rows_by.setdefault(gi, []).append(i)
    sd = {gi: float(np.std(y_seen[rows], ddof=1))
          for gi, rows in rows_by.items() if len(rows) >= min_measurements}
    if not sd:
        return np.zeros(len(g), dtype=bool), {
            'molecules_measured_enough': 0}
    cut = float(np.median(list(sd.values())))
    keep_g = {gi for gi, s in sd.items() if s <= cut}
    mask = np.array([gi in keep_g for gi in g])
    return mask, {'molecules_measured_enough': len(sd),
                  'molecules_trusted': len(keep_g),
                  'rows_trusted': int(mask.sum()),
                  'spread_cut_label_units': cut}


def min_bh_flags(n_tested, n_cal, q):
    """The fewest molecules Benjamini-Hochberg can flag at once, other than none.

    No conformal p-value is below 1 / (n_cal + 1), and BH flags k molecules only
    if k of them have p <= q k / n_tested. So k >= n_tested / (q (n_cal + 1)):
    about (1 - s) / (s q) with a calibration share s, whatever the fold size.
    At s = 0.2 that is 40 molecules at q = 0.1, 80 at q = 0.05, 20 at q = 0.2,
    and every one of them has to outscore the WHOLE calibration set.
    """
    return int(np.ceil(n_tested / (q * (n_cal + 1.0)))) if n_cal > 0 else None


def bh_record(flags, cmask, cal, q=None):
    """Counts for one (calibration set, q): flags, false flags, precision, recall."""
    flags = np.asarray(flags, dtype=bool)
    cmask = np.asarray(cmask, dtype=bool)
    n_flag = int(flags.sum())
    n_false = int(np.sum(flags & ~cmask))
    moved_tested = int(np.sum(cmask & ~cal))
    return {
        'molecules_flagged': n_flag,
        'false_flags': n_false,
        'false_flag_share': (n_false / n_flag) if n_flag else 0.0,
        'precision': ((n_flag - n_false) / n_flag) if n_flag else None,
        'recall_among_tested': (((n_flag - n_false) / moved_tested)
                                if moved_tested else None),
        'molecules_tested': int((~cal).sum()),
        'calibration_molecules': int(cal.sum()),
        'moved_molecules_in_calibration': int(np.sum(cmask & cal)),
        'moved_molecules_tested': moved_tested,
        'fewest_flags_bh_can_make': (None if q is None else min_bh_flags(
            int((~cal).sum()), int(cal.sum()), q)),
    }


# ── The other detectors ──────────────────────────────────────────────────────

def joined_matrix(X, r):
    """[X, r'] with r z-scored and multiplied by sqrt(number of features)."""
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=float)
    sd = float(np.std(r))
    rz = (r - float(np.mean(r))) / (sd if sd > 0 else 1.0)
    return np.hstack([X, (rz * np.sqrt(X.shape[1]))[:, None]])


def isolation_forest_suspicion(X, r, seed=0):
    from sklearn.ensemble import IsolationForest
    Z = joined_matrix(X, r)
    m = IsolationForest(n_estimators=IFOREST_TREES, random_state=int(seed))
    m.fit(Z)
    return -m.score_samples(Z)


def lof_suspicion(X, r, n_neighbors=LOF_NEIGHBOURS):
    from sklearn.neighbors import LocalOutlierFactor
    Z = joined_matrix(X, r)
    k = int(min(n_neighbors, len(Z) - 1))
    m = LocalOutlierFactor(n_neighbors=k)
    m.fit(Z)
    return -m.negative_outlier_factor_


def residual_low_disagreement(X, y, groups, oof_splits, seeds=DISAGREEMENT_SEEDS):
    """|mean seed prediction - y| / (SD across seeds + floor), all out of fold."""
    y = np.asarray(y, dtype=float)
    P = np.vstack([oof_predict(X, y, groups, n_splits=oof_splits,
                               params={'random_state': int(s)}) for s in seeds])
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    floor = max(0.1 * float(np.median(sd)), 1e-9)
    return np.abs(y - mu) / (sd + floor)


def detector_scores(name, X, y, groups, p, score, oof_splits):
    if name == 'rescaled_residual':
        return score
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    if name == 'isolation_forest':
        return isolation_forest_suspicion(X, r)
    if name == 'lof':
        return lof_suspicion(X, r)
    if name == 'residual_low_disagreement':
        return residual_low_disagreement(X, y, groups, oof_splits)
    if name == 'cleanlab':
        return cleanlab_suspicion(X, y)
    raise ValueError(f'n41: unknown detector {name!r}')


# ── Actions ─────────────────────────────────────────────────────────────────

def winsorise_flagged(y, p, flags, c=WINSOR_C):
    """Flagged labels moved to p + clip(y - p, +-c*s); every other label as is.

    s = 1.4826 * median |y - p| over the whole training fold, n30's scale.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    r = y - p
    s = 1.4826 * float(np.median(np.abs(r)))
    out = y.copy()
    if not np.isfinite(s) or s <= 0:
        return out
    f = np.asarray(flags, dtype=bool)
    out[f] = p[f] + np.clip(r[f], -c * s, c * s)
    return out


def random_flags(candidates, n_flag, seed):
    """n_flag molecules drawn at random from the candidate mask."""
    cand = np.flatnonzero(np.asarray(candidates, dtype=bool))
    out = np.zeros(len(candidates), dtype=bool)
    n_flag = int(min(n_flag, cand.size))
    if n_flag > 0:
        rng = np.random.default_rng(int(seed))
        out[rng.choice(cand, n_flag, replace=False)] = True
    return out


def build_arm_names(cals, qs, seeds, actions=ACTIONS):
    out = ['clean', 'no_cleaning']
    for a in actions:
        for c in cals:
            for q in qs:
                base = f'{a}_{c}_q{q:g}'
                out.append(base)
                out += [f'{base}_random/seed{s}' for s in seeds]
    return out


def _levels_and_draws(dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs. At level 0 nothing moves, so one draw is kept."""
    if levels is None:
        if condition in SHIFT_ONLY_CONDITIONS:
            levels = (0.0,) + tuple(SHIFT_ONLY_DELTAS_SD)
        elif condition in RELATIVE_CONDITIONS:
            levels = relative_levels(condition)
        else:
            levels = noise_levels(dataset_name, strategy=condition,
                                  detect_grid=True, is_clf=False)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        for d in (draws if float(lvl) > 0 else draws[:1]):
            out.append((float(lvl), int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N41 across every condition where detection is defined, one file each.

    The shift_only_* and sparse_dprime_* conditions are not in the NoiseInject
    registry, so no per-condition job names them; they run here, in the job
    that names no condition.
    """
    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    named = kwargs.get('noise_condition')
    if named and not detection_is_defined(named):
        print(f"  SKIPPED: {named} moves every training label, so there is no "
              f"damaged subset to find.")
        return {'skipped': 'detection_undefined', 'noise_condition': named}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(DETECTION_IS_DEFINED), kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N41: conformal p-values with BH, four other detectors, flag-only actions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N41: FLAGS WITH A CONTROLLED FALSE-FLAG RATE")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. The conformal score is "
              f"a residual, so n41 is regression only.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    if not detection_is_defined(condition):
        raise ValueError(
            f"n41: condition {condition!r} moves every training label, so there "
            f"is no damaged subset to find. Run it under one of: "
            f"{', '.join(DETECTION_IS_DEFINED)}.")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    qs = [float(q) for q in (kwargs.get('bh_qs') or BH_QS)]
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))

    rep_groups = build_replicate_groups(data)
    cals = [random_cal_name(s) for s in CALIBRATION_SHARES] + (
        ['trusted'] if rep_groups is not None else [])
    arm_names = build_arm_names(cals, qs, seeds)

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n41: no downstream model could be built.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n41: condition {condition!r} needs scaffold groups and "
            f"{dataset_name} has none.")

    if condition in SHIFT_ONLY_CONDITIONS:
        dose_units = 'label SDs of upward shift'
    elif condition in RELATIVE_CONDITIONS:
        dose_units = ('residual SDs: lightgbm out-of-fold residual on the '
                      'fold\'s clean training labels, base space')
    else:
        dose_units = noise_dose_units(dataset_name)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': dose_units,
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'models': downstream_models,
        'base_rep': base_rep_name,
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'detectors': detectors,
        'bh_qs': qs,
        'calibration_sets': cals,
        'calibration_shares': list(CALIBRATION_SHARES),
        'min_calibration': MIN_CALIBRATION,
        'trusted_rule': (f'molecules measured {TRUSTED_MIN_MEASUREMENTS}+ times '
                         f'in the training fold whose spread across their own '
                         f'measurements, as seen, is in the lowest half'),
        'conformal_score': ('|y - p| / sigma, p the out-of-fold LightGBM '
                            'prediction, sigma an out-of-fold LightGBM fitted '
                            f'to |y - p|, floored at {SIGMA_FLOOR_OF_MEDIAN} x '
                            'the median |y - p|'),
        'winsor_c': WINSOR_C,
        'random_seeds': seeds,
        'oof_splits': oof_splits,
        'joined_matrix_note': ('isolation_forest and lof run on [features, r]; r '
                               'z-scored and multiplied by sqrt(number of '
                               'features)'),
        'no_flag_note': ('When BH flags nothing the arm IS no_cleaning, and '
                         'no_cleaning\'s score is written for it on that pass.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset. Held-out molecules are graded against '
            'untouched labels in every arm.'),
        'per_level': {},
        'tails': {},
        'detection': {},
        'bh': {},
        'fold_records': {},
    }

    sample_X = reps[base_rep_name]
    splits = list(cv_split(groups, n_folds, sample_X, labels, mode=mode))

    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type=task_type, data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values()
               if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}
    recipes_by_model, recipe_for_space_by_model = {}, {}
    hybrid_unavailable = {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY,
            eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
        recipes_by_model[_m] = _m_recipes
        recipe_for_space_by_model[_m] = {r['space']: r['blueprint']
                                         for r in _m_recipes}
        for _sp, _why in _m_unavail.items():
            hybrid_unavailable.setdefault(_sp, {})[_m] = _why
    results['hybrid_spaces_unavailable'] = hybrid_unavailable
    _ordered = []
    for _m in downstream_models:
        for r in recipes_by_model.get(_m, ()):
            if r['space'] not in _ordered:
                _ordered.append(r['space'])
    spaces = ('base',) + tuple(_ordered)
    results['hybrid_available'] = bool(_ordered)
    if _ordered:
        results['hybrid_recipe_by_model'] = convert_numpy({
            m: {r['space']: {
                'built_under_objective': r['objective'],
                'features_chosen_by': describe_pool_family(r['objective']),
                'budget': r['budget'],
                'allocation_method': r['allocation_method'],
                'importance_method': r['importance_method'],
                'reps': sorted(getattr(r['blueprint'], 'reps', []) or []),
            } for r in rs}
            for m, rs in recipes_by_model.items() if rs})
    else:
        results['hybrid_unavailable_reason'] = (
            (f"the pool lookup RAISED {results['hybrid_lookup_error']}; the "
             f"pool was never read. Every arm ran in the base representation.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the base representation.'))
        print("  no hybrid recipe; base representation only")
    results['spaces_run'] = list(spaces)

    passes = _levels_and_draws(dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes, {len(arm_names)} arms, "
          f"calibration sets {cals}, detectors {detectors}")

    acc, tail_acc, det_acc, bh_acc, rec_acc, errors = {}, {}, {}, {}, {}, {}
    t0 = time.time()

    for level, draw in passes:
        lvl_key = f'{level:g}'
        spec = (None if condition in SHIFT_ONLY_CONDITIONS
                or condition in RELATIVE_CONDITIONS
                else noise_spec_for(dataset_name, condition, level,
                                    noise_seed=draw, is_clf=False))
        print(f"\n  --- {condition} / level {lvl_key} / draw {draw} ---")
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = (np.asarray(groups)[train_idx] if groups is not None
                    else np.arange(len(train_idx)))
            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}
            sc = StandardScaler()
            X_base_tr = sc.fit_transform(train_reps[base_rep_name])
            X_base_va = sc.transform(val_reps[base_rep_name])
            fold_rec = {'pass': pass_key}
            try:
                if condition in SHIFT_ONLY_CONDITIONS:
                    y0, cmask, prov = inject_shift_train_labels(
                        labels, train_idx=train_idx, condition=condition,
                        delta_sd=level, seed=draw * 1000 + fold_i)
                elif condition in RELATIVE_CONDITIONS:
                    s_r, _ = oof_residual_sd(X_base_tr, y_clean_tr, groups=g_tr,
                                             n_splits=oof_splits)
                    fold_rec['resid_sd_label_units'] = s_r
                    y0, cmask, prov = inject_relative_train_labels(
                        labels, train_idx=train_idx, condition=condition,
                        level=level, seed=draw * 1000 + fold_i, resid_sd=s_r)
                else:
                    y0, cmask, prov = inject_train_labels(
                        labels, spec, train_idx=train_idx, is_clf=False,
                        groups=noise_groups, dataset_name=dataset_name)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y0 = np.asarray(y0, dtype=float)
            cmask = np.asarray(cmask, dtype=bool)
            n_tr = int(len(y0))
            fold_rec['molecules_moved'] = int(cmask.sum())
            fold_rec['training_molecules'] = n_tr

            # ── Calibration sets, shared by every space ──
            cal_masks = {random_cal_name(sh): random_calibration(
                n_tr, sh,
                replicate_seed('cal', lvl_key, draw, fold_i, n_tr, sh))
                for sh in CALIBRATION_SHARES}
            if 'trusted' in cals:
                tmask, trec = trusted_calibration(
                    y0, np.asarray(rep_groups)[train_idx])
                fold_rec['trusted_calibration'] = trec
                if tmask is not None and int(tmask.sum()) >= MIN_CALIBRATION:
                    cal_masks['trusted'] = tmask
                else:
                    fold_rec['trusted_calibration']['refused_because'] = (
                        f'{int(0 if tmask is None else tmask.sum())} rows '
                        f'qualify, under {MIN_CALIBRATION}')
            rec_acc.setdefault(lvl_key, []).append(fold_rec)

            fold_spaces = {'base': [(X_base_tr, X_base_va,
                                     list(downstream_models))]}
            for _sp in spaces:
                if _sp == 'base':
                    continue
                _cells = []
                for _bp_m, _mnames, _shareable in group_models_by_recipe(
                        {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                         for _m in downstream_models}, downstream_models):
                    X_tr_h, X_va_h = hybrid_for_fold(
                        train_reps, val_reps, y0, _bp_m, g_tr,
                        None if _shareable
                        else model_factories[_mnames[0]](),
                        metric_fn, lower_is_better)
                    if X_tr_h is None:
                        errors[f'{lvl_key}|{_sp}/{_mnames[0]}|f{fold_i}|'
                               f'build'] = 'the hybrid could not be built'
                        continue
                    sh = StandardScaler()
                    _cells.append((sh.fit_transform(X_tr_h),
                                   sh.transform(X_va_h), list(_mnames)))
                if _cells:
                    fold_spaces[_sp] = _cells

            for space in spaces:
                for X_tr_s, X_va_s, cell_models in fold_spaces.get(space, ()):
                    _ekey = (space if cell_models == list(downstream_models)
                             else f'{space}/{cell_models[0]}')
                    try:
                        p = oof_predict(X_tr_s, y0, g_tr, n_splits=oof_splits)
                        score, _sigma = rescaled_residual(X_tr_s, y0, g_tr, p,
                                                          oof_splits)
                    except Exception as exc:
                        errors[f'{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue

                    for det in detectors:
                        try:
                            sus = detector_scores(det, X_tr_s, y0, g_tr, p,
                                                  score, oof_splits)
                        except Exception as exc:
                            errors[f'{lvl_key}|{_ekey}|{det}|f{fold_i}'] = (
                                f'{type(exc).__name__}: {exc}')
                            continue
                        d = detection_scores(cmask, sus, condition)
                        if d is not None:
                            (det_acc.setdefault(lvl_key, {})
                             .setdefault(_ekey, {})
                             .setdefault(det, []).append(d))

                    cols = {'clean': (y_clean_tr, None),
                            'no_cleaning': (y0, None)}
                    no_flag_arms = set()
                    for cname, cal in cal_masks.items():
                        pv = conformal_pvalues(score[cal], score[~cal])
                        for q in qs:
                            flags = np.zeros(n_tr, dtype=bool)
                            flags[np.flatnonzero(~cal)[
                                benjamini_hochberg(pv, q)]] = True
                            brec = bh_record(flags, cmask, cal, q)
                            (bh_acc.setdefault(lvl_key, {})
                             .setdefault(_ekey, {})
                             .setdefault(f'{cname}_q{q:g}', []).append(brec))
                            n_flag = int(flags.sum())
                            for s in seeds:
                                rf = random_flags(~cal, n_flag, replicate_seed(
                                    'rf', cname, q, s, lvl_key, draw, fold_i))
                                for act in ACTIONS:
                                    base = f'{act}_{cname}_q{q:g}'
                                    arm = f'{base}_random/seed{s}'
                                    if n_flag == 0:
                                        no_flag_arms.add(arm)
                                    elif act == 'winsorise':
                                        cols[arm] = (winsorise_flagged(y0, p, rf),
                                                     None)
                                    else:
                                        cols[arm] = (y0, ~rf)
                            for act in ACTIONS:
                                base = f'{act}_{cname}_q{q:g}'
                                if n_flag == 0:
                                    no_flag_arms.add(base)
                                elif act == 'winsorise':
                                    cols[base] = (winsorise_flagged(y0, p, flags),
                                                  None)
                                else:
                                    cols[base] = (y0, ~flags)

                    space_cell = acc.setdefault(lvl_key, {}).setdefault(
                        space, {})
                    tail_cell = tail_acc.setdefault(lvl_key, {}).setdefault(
                        space, {})
                    for arm in arm_names:
                        if arm not in cols:
                            continue
                        y_use, keep = cols[arm]
                        for mname in cell_models:
                            s_, tl, err = _fit_score(
                                model_factories[mname], X_tr_s, y_use, X_va_s,
                                y_val, metric_fn, metric_name, keep=keep)
                            if s_ is None:
                                errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                       f'f{fold_i}'] = err
                                continue
                            (space_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = s_
                            if tl is not None:
                                (tail_cell.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = tl
                    # An arm with no flags is no_cleaning on this pass.
                    for arm in no_flag_arms:
                        for mname in cell_models:
                            b = (space_cell.get('no_cleaning') or {}).get(
                                mname, {})
                            if pass_key in b:
                                (space_cell.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = b[pass_key]
                                bt = (tail_cell.get('no_cleaning') or {}).get(
                                    mname, {})
                                if pass_key in bt:
                                    (tail_cell.setdefault(arm, {})
                                     .setdefault(mname, {})[pass_key]) = \
                                        bt[pass_key]
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{int(cmask.sum())} of {n_tr} training molecules moved "
                  f"[{time.time() - t0:.0f}s]")

    results['per_level'] = convert_numpy(acc)
    results['tails'] = convert_numpy(tail_acc)
    results['detection'] = convert_numpy({
        lvl: {sp: {d: _mean_detection(rows) for d, rows in by_det.items()}
              for sp, by_det in by_sp.items()}
        for lvl, by_sp in det_acc.items()})
    results['bh'] = convert_numpy({
        lvl: {sp: {k: {'passes': rows, 'mean': _mean_bh(rows)}
                   for k, rows in by_k.items()}
              for sp, by_k in by_sp.items()}
        for lvl, by_sp in bh_acc.items()})
    results['fold_records'] = convert_numpy(rec_acc)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, tail_acc, arm_names, downstream_models, cals, qs, seeds,
        metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _mean_bh(rows):
    """Means over passes; precision and recall over the passes where defined."""
    out = {'n_passes': len(rows)}
    for k in ('molecules_flagged', 'false_flags', 'false_flag_share',
              'precision', 'recall_among_tested', 'calibration_molecules',
              'moved_molecules_in_calibration', 'fewest_flags_bh_can_make'):
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
        if k in ('precision', 'recall_among_tested'):
            out[f'{k}_n_passes_defined'] = len(vals)
    return out


def _summarise(acc, tail_acc, arm_names, models, cals, qs, seeds, metric_name,
               lower_is_better):
    """Every arm against no_cleaning; each flag arm against its random twins."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against no_cleaning, paired per pass; a '
                        'pass is one (draw, fold)'),
           'per_level': {}}
    twins = {}
    for a in ACTIONS:
        for c in cals:
            for q in qs:
                base = f'{a}_{c}_q{q:g}'
                twins[base] = [f'{base}_random/seed{s}' for s in seeds]
    for lvl, by_space in acc.items():
        lrow = {}
        for space, by_arm in by_space.items():
            base = by_arm.get('no_cleaning') or {}
            clean = by_arm.get('clean') or {}
            tails = (tail_acc.get(lvl) or {}).get(space) or {}
            srow = {}
            for m in models:
                b = base.get(m) or {}
                if not b:
                    continue
                row = {'no_cleaning_mean': _mean_on(b, list(b)), 'per_arm': {}}
                for arm in arm_names:
                    if arm == 'no_cleaning':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_no_cleaning': _paired(b, g, lower_is_better)}
                    if arm != 'clean':
                        keys = sorted(set(clean.get(m) or {}) & set(b) & set(g))
                        sh, why = (recovery_share(
                            _mean_on(clean[m], keys), _mean_on(b, keys),
                            _mean_on(g, keys), lower_is_better,
                            gap_min_frac=GAP_MIN_FRAC)
                            if keys else (None, 'no pass scored all three'))
                        e['recovery_share'] = sh
                        e['recovery_share_refused_because'] = why
                    if arm in twins:
                        vs = {}
                        for r_arm in twins[arm]:
                            rg = (by_arm.get(r_arm) or {}).get(m) or {}
                            if rg:
                                vs[r_arm] = _paired(rg, g, lower_is_better)
                        e['vs_random_flags'] = vs
                        e['beat_every_random_seed'] = bool(vs and all(
                            v['passes_won'] * 2 > v['n_passes_paired']
                            for v in vs.values()))
                    ta = (tails.get(arm) or {}).get(m) or {}
                    tb = (tails.get('no_cleaning') or {}).get(m) or {}
                    if ta and tb:
                        e['tails'] = _tail_row(ta, tb)
                    row['per_arm'][arm] = e
                srow[m] = row
            lrow[space] = srow
        out['per_level'][lvl] = lrow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against no_cleaning)")
    for lvl, by_sp in (results.get('bh') or {}).items():
        for sp, by_k in by_sp.items():
            for k, d in by_k.items():
                m = d['mean']
                prec = ('undefined' if m['precision'] is None
                        else f"{m['precision']:.3f}")
                print(f"  level {lvl} {sp} BH {k}: {m['molecules_flagged']:.1f} "
                      f"molecules flagged per pass, false-flag share "
                      f"{m['false_flag_share']:.3f}, precision {prec}, over "
                      f"{m['n_passes']} passes")
    for lvl, by_sp in (results.get('detection') or {}).items():
        for sp, by_det in by_sp.items():
            for det, row in by_det.items():
                if row:
                    print(f"  level {lvl} {sp}/{det}: AUROC "
                          f"{row['detection_auroc']:.3f}, average precision "
                          f"{row['detection_average_precision']:.3f}, over "
                          f"{row['detection_n_corrupted']:.0f} moved molecules "
                          f"of {row['detection_n_training']:.0f}")
    for lvl, lrow in s['per_level'].items():
        print(f"\n  level {lvl}:")
        for space, srow in lrow.items():
            for m, row in srow.items():
                print(f"    {space}/{m}: no_cleaning "
                      f"{row['no_cleaning_mean']:.4f}")
                for arm, e in row['per_arm'].items():
                    if '_random/' in arm:
                        continue
                    v = e['vs_no_cleaning']
                    tail = ''
                    if 'beat_every_random_seed' in e:
                        tail = (f"; beat every random seed: "
                                f"{e['beat_every_random_seed']}")
                    print(f"      {arm:28s} won {v['passes_won']} of "
                          f"{v['n_passes_paired']} passes, mean "
                          f"{v['mean_gain']:+.4f} (spread "
                          f"{v['spread_over_pass_gains']:.4f}){tail}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N41: flags with a controlled false-flag rate')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'Injected condition (default: every one where '
                             f'detection is defined). One of: '
                             f'{", ".join(DETECTION_IS_DEFINED)}')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels. Default: the condition\'s '
                             'own grid')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates')
    parser.add_argument('--bh-qs', type=str, default=None,
                        help='Comma-separated Benjamini-Hochberg levels')
    parser.add_argument('--detectors', type=str, default=None,
                        help=f'Comma-separated detectors (default '
                             f'{",".join(DETECTORS)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-flag replicates')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides the 1 or 5 --mode gives')
    parser.add_argument('--base-rep', type=str, default=None,
                        help="Base representation. Default: pick_noise_base_rep's")
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.bh_qs:
        kw['bh_qs'] = [float(x) for x in args.bh_qs.split(',')]
    if args.detectors:
        kw['detectors'] = [x.strip() for x in args.detectors.split(',')]
    if args.random_seeds:
        kw['random_seeds'] = [int(x) for x in args.random_seeds.split(',')]
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    if args.tag:
        kw.setdefault('noise_condition', DEFAULT_NOISE_CONDITION)
        res = _run_one_condition(data, config, args.output, args.dataset, **kw)
        save_experiment_results(res, args.output, EXP_ID, args.dataset,
                                filename_suffix=f"{kw['noise_condition']}_{args.tag}")
        return
    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output, EXP_ID, args.dataset)


if __name__ == '__main__':
    main()
