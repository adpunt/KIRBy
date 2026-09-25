#!/usr/bin/env python3
"""N34: an offset per scaffold group -- subtract the group mean, or fit a random intercept?

Paper: Noise.

THE QUESTION. Under `grouped_shifted` every scaffold group gets one shared offset
on top of a per-molecule error. A method that knows the groups can estimate each
group's offset from the model's residuals and take it off the training labels.
The obvious estimate, the group's mean out-of-fold residual, LOST on all four
models on herg_ki (`/Volumes/seagate/local_noise_run/loop_headroom/FINDINGS.md`
section 5, `grouped_offset_probe.py`, 2 folds, level 0.5). The reason given there:
herg_ki's injector groups hold a MEDIAN OF 1 MOLECULE in the training block, and
a group of one cannot separate its offset from that molecule's own residual, so
subtracting its mean sets the molecule's training residual to zero.

The second research report (2026-09-24, its A3 and B) proposes a random
intercept instead, as GPBoost and MERF fit: each group's offset is shrunk toward
zero by how much evidence the group carries, offset = residual sum /
(count + sigma^2 / tau^2). A group of one with a within-group spread sigma^2 as
large as the between-group spread tau^2 keeps half its mean residual, not all of
it. This file puts the two side by side.

WHERE THE GROUPS COME FROM. `dataset_policy.noise_groups_for(data)`, the
scaffold groups the injector draws its offsets over, sliced to this fold's
training rows. They are NOT the outer fold grouping (`data['groups']`), which
`cv_split` uses. So a held-out molecule can share a noise group with training
molecules; the count of such held-out molecules is written per pass
(`held_out_molecules_in_a_training_group`). It does not matter for scoring: every
arm predicts held-out molecules with the fitted model alone, no offset added,
and grades them against their CLEAN labels. An offset only changes what the
model is trained on.

THE CONDITIONS.
  grouped_shifted   a shared offset per group plus a per-molecule error, with
                    0.62 of the variance at group level; the offsets are not
                    centred (`noiseInject/core.py:_draw_grouped_shifted`)
  grouped_wider     20% of groups get a 3x wider error; zero mean, so there is
                    no offset to find. The arms should do nothing here.
  none              level 0.0: nothing injected. Tests whether the REAL labels
                    carry group offsets. Run once per fold, not once per
                    condition, because level 0.0 is the same column under both.

THE ARMS. All five downstream models, each in the base representation and both
hybrid spaces. Paired against `none`.

  clean                 fitted on the clean training labels. The reference the
                        recovery share divides by. Not a method.
  none                  fitted on the noisy training labels as they are.
  group_center          every training label minus its group's mean
                        out-of-fold residual.
  group_center_3plus    the same, only for groups holding 3 or more training
                        molecules; the others are left alone.
  random_intercept      every training label minus a shrunk offset, fitted
                        MERF-style (below).

WHICH MODEL THE OFFSETS COME FROM. The residuals every offset arm reads are from
ONE model, a group-aware out-of-fold LightGBM (`kirby.utils.noisy_labels.
oof_predict`), fitted in the space the arm is scored in, on this fold's noisy
training labels. The five downstream models then each train on the corrected
column. That is n30's arrangement for its lifts, and it keeps the cost to one
out-of-fold fit per (space, fold) per EM round rather than one per model.
Out-of-fold, not in-fold: an in-fold fit has memorised the offset it is being
asked to reveal, and its residual would be near zero on every row.

THE RANDOM INTERCEPT, MERF-STYLE. Hajjem, Bellavance & Larocque (2014) fit a
forest inside an EM loop. Here, written by hand so it runs with any model and
needs neither gpboost nor merf (neither is installed):

  b = 0
  repeat up to EM_MAX_ROUNDS:
      p   = out-of-fold LightGBM on (y - b[g])
      r   = y - p                       residual including the offset
      sigma^2 = mean of (r - b[g])^2     within-group spread
      tau^2   = var over groups of mean_g(r) - sigma^2 * mean over groups of
                1/n_g, floored at TAU2_FLOOR_FRAC * var(y)   (method of moments)
      b_g = sum_g(r) / (n_g + sigma^2 / tau^2)
  stop when no offset moved more than EM_TOL_FRAC * sd(y)

The fitted sigma^2, tau^2, the rounds used and whether it converged are written
per (pass, space). So is the fraction a group of ONE keeps of its mean residual,
tau^2 / (tau^2 + sigma^2). `group_center` keeps all of it.

HOW WELL THE OFFSETS WERE RECOVERED. The injector's provenance row does not carry
the per-group offsets it drew (checked in this session: `noise_provenance_row`
writes scalars only). So under `grouped_shifted` the injected offset of a group
is RECOMPUTED here as the mean of (noisy - clean) over that group's training
molecules. For a group of one that mean still contains the molecule's own
per-molecule error, so it is the offset plus noise, not the offset. The
correlation between recovered and injected offsets is reported over groups of
3 or more training molecules (`corr_groups_of_3_plus`), with the count of groups
used and the count left out beside it, for `group_center` and
`random_intercept`. The all-groups number is kept as
`corr_all_groups_contaminated`: it is not the result.

WHERE THE FEATURES COME FROM. The recipe per downstream model is read from the
precomputed pool, searched on clean labels. Each fold rebuilds each hybrid once
from that fold's NOISY training labels, and every arm shares it, clean included.

Usage:
    python n34_group_random_effects.py --dataset herg_ki --mode full
    python n34_group_random_effects.py --dataset logd --noise-condition grouped_shifted
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
    pool_recipe_per_downstream_model, group_models_by_recipe, tail_scores,
)
from experiments.noise.n07_remediation import _predict_for_metric
from kirby.utils.noisy_labels import oof_predict

GROUPED_CONDITIONS = ('grouped_shifted', 'grouped_wider')
# The level-0.0 pass is stored under this name: nothing is injected, so it is
# the same column whichever grouped condition asked for it.
NO_INJECTION = 'none'

# Three draws, as n27. Unlike censoring, the grouped draw depends on the seed.
NOISE_DRAWS = (0, 1, 2)

OOF_SPLITS = 5
EM_MAX_ROUNDS = 5
# Stop when no offset moves by more than this share of the training label SD.
EM_TOL_FRAC = 1e-3
# tau^2 floor, as a share of the training label variance. At the floor the
# random intercept keeps almost nothing of any group's mean residual.
TAU2_FLOOR_FRAC = 1e-6
MIN_GROUP_SIZE_FOR_CENTER = 3

# The recovery share is refused below this share of the clean score. n07:294,
# n11:272, n30.
GAP_MIN_FRAC = 0.02

ARMS = ('clean', 'none', 'group_center', 'group_center_3plus',
        'random_intercept')
ARM_TO_BEAT = 'none'

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}


# ── Offsets ────────────────────────────────────────────────────────────────

def _group_index(g):
    """(unique groups, inverse index, count per group)."""
    uniq, inv, counts = np.unique(np.asarray(g), return_inverse=True,
                                  return_counts=True)
    return uniq, inv, counts.astype(float)


def group_center_offsets(r, g, min_size=1):
    """Each group's mean residual, as a per-row offset. Groups below min_size get 0."""
    _, inv, counts = _group_index(g)
    sums = np.bincount(inv, weights=np.asarray(r, dtype=float))
    means = sums / counts
    means[counts < min_size] = 0.0
    return means[inv]


def moments_tau2(r, inv, counts, sigma2, floor):
    """Between-group variance by the method of moments, floored.

    var over groups of mean_g(r) = tau^2 + sigma^2 * mean(1 / n_g), so
    tau^2 = var(group means) - sigma^2 * mean(1 / n_g).
    """
    means = np.bincount(inv, weights=r) / counts
    if means.size < 2:
        return float(floor)
    t = float(np.var(means) - sigma2 * np.mean(1.0 / counts))
    return float(max(t, floor))


def random_intercept_offsets(X, y, g, oof_fn, max_rounds=EM_MAX_ROUNDS,
                             tol_frac=EM_TOL_FRAC,
                             tau2_floor_frac=TAU2_FLOOR_FRAC):
    """MERF-style EM for one random intercept per group. See the docstring.

    oof_fn(X, y) must return an out-of-fold prediction for every row.
    Returns (per-row offset, info dict).
    """
    y = np.asarray(y, dtype=float)
    _, inv, counts = _group_index(g)
    var_y = float(np.var(y))
    floor = tau2_floor_frac * max(var_y, 1e-12)
    tol = tol_frac * max(np.sqrt(var_y), 1e-12)
    b_group = np.zeros(counts.size)
    sigma2 = tau2 = float('nan')
    converged = False
    rounds = 0
    history = []
    for rounds in range(1, max_rounds + 1):
        p = np.asarray(oof_fn(X, y - b_group[inv]), dtype=float)
        r = y - p
        # The EM step for sigma^2 adds each offset's posterior variance,
        # 1 / (n_g / sigma^2 + 1 / tau^2), to the squared residual, as MERF
        # does. Without it sigma^2 is biased low: a shrunk offset has already
        # absorbed part of its own molecules' residuals. MEASURED on the smoke's
        # toy (true sigma^2 1.0): 0.73 without the term.
        if np.isfinite(sigma2) and np.isfinite(tau2):
            post_var = 1.0 / (counts / sigma2 + 1.0 / tau2)
        else:
            post_var = np.zeros(counts.size)
        sigma2 = float(max(np.mean((r - b_group[inv]) ** 2 + post_var[inv]),
                           1e-12))
        tau2 = moments_tau2(r, inv, counts, sigma2, floor)
        sums = np.bincount(inv, weights=r)
        new_b = sums / (counts + sigma2 / tau2)
        step = float(np.max(np.abs(new_b - b_group))) if new_b.size else 0.0
        history.append({'round': rounds, 'sigma2': sigma2, 'tau2': tau2,
                        'largest_offset_move': step})
        b_group = new_b
        if step <= tol:
            converged = True
            break
    info = {
        'sigma2_within_group': sigma2,
        'tau2_between_group': tau2,
        'tau2_at_floor': bool(np.isfinite(tau2) and tau2 <= floor * (1 + 1e-9)),
        'em_rounds': int(rounds),
        'em_converged': bool(converged),
        'em_history': history,
        'share_a_group_of_one_keeps': (
            float(tau2 / (tau2 + sigma2)) if np.isfinite(tau2) else float('nan')),
        'mean_abs_offset_groups_of_one': (
            float(np.mean(np.abs(b_group[counts == 1])))
            if np.any(counts == 1) else float('nan')),
    }
    return b_group[inv], info


def injected_offsets(y_noisy, y_clean, g):
    """Per-group mean of (noisy - clean), per group. See the docstring's caveat."""
    uniq, inv, counts = _group_index(g)
    d = np.asarray(y_noisy, dtype=float) - np.asarray(y_clean, dtype=float)
    return uniq, np.bincount(inv, weights=d) / counts, counts


def offset_recovery(offset_rows, y_noisy, y_clean, g):
    """Correlation of recovered per-group offsets with the recomputed injected ones.

    The headline is `corr_groups_of_3_plus`: only groups with at least
    MIN_GROUP_SIZE_FOR_CENTER training molecules. The injected offset is
    recomputed as the group mean of (noisy - clean), so for a group of one or
    two it still holds most of those molecules' own per-molecule error. Those
    groups are counted in `n_groups_excluded_under_3` and left out.
    """
    uniq, inj, counts = injected_offsets(y_noisy, y_clean, g)
    _, inv, _ = _group_index(g)
    rec = np.bincount(inv, weights=np.asarray(offset_rows, dtype=float)) / counts

    def _corr(mask):
        if mask.sum() < 3 or np.std(inj[mask]) == 0 or np.std(rec[mask]) == 0:
            return None
        return float(np.corrcoef(inj[mask], rec[mask])[0, 1])
    big = counts >= MIN_GROUP_SIZE_FOR_CENTER
    return {
        'corr_groups_of_3_plus': _corr(big),
        'n_groups_of_3_plus': int(big.sum()),
        'n_groups_excluded_under_3': int((~big).sum()),
        'min_group_size': MIN_GROUP_SIZE_FOR_CENTER,
        # Every group, including groups of one and two. For those the
        # "injected" offset is the offset plus the molecule's own error, so
        # this number is contaminated and is not the headline.
        'corr_all_groups_contaminated': _corr(np.ones(counts.size, dtype=bool)),
        'n_groups_all': int(counts.size),
    }


def recovery_share(clean, noisy, arm, lower_is_better):
    """(share, None) or (None, why). n30's rule."""
    vals = (clean, noisy, arm)
    if not all(np.isfinite(v) for v in vals):
        return None, 'a mean is not finite'
    gap = (noisy - clean) if lower_is_better else (clean - noisy)
    if abs(gap) < GAP_MIN_FRAC * max(abs(clean), 1e-12):
        return None, (f'the clean-to-noisy gap {gap:+.4g} is under '
                      f'{GAP_MIN_FRAC:.0%} of the clean score')
    got = (noisy - arm) if lower_is_better else (arm - noisy)
    return float(got / gap), None


# ── Runner ────────────────────────────────────────────────────────────────

def _fit_predict(factory, X_tr, y_tr, X_va, metric_name):
    m = factory()
    m.fit(X_tr, y_tr)
    return _predict_for_metric(m, X_va, metric_name)


def _passes(dataset_name, conditions, levels=None, draws=None):
    """(condition, level, draw). Level 0.0 appears once, under NO_INJECTION."""
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    zero_done = False
    for cond in conditions:
        lv = (noise_levels(dataset_name, strategy=cond, is_clf=False)
              if levels is None else levels)
        for l in lv:
            l = float(l)
            if l == 0.0:
                if zero_done:
                    continue
                zero_done = True
                # Nothing is drawn at level 0, so one draw.
                out.append((NO_INJECTION, 0.0, 0))
                continue
            for d in draws:
                out.append((cond, l, int(d)))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N34: group mean subtraction against a shrunk random intercept."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N34: SCAFFOLD-GROUP OFFSETS -- MEAN OR RANDOM INTERCEPT")
    print("=" * 70)

    condition = kwargs.get('noise_condition')
    if condition is not None and condition not in GROUPED_CONDITIONS:
        print(f"  SKIPPED: condition {condition!r} has no group structure. n34 "
              f"runs under {GROUPED_CONDITIONS} only.")
        return {'skipped': 'condition_not_grouped', 'noise_condition': condition}
    conditions = [condition] if condition else list(GROUPED_CONDITIONS)

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A class label has no "
              f"offset to subtract.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    noise_groups = noise_groups_for(data)[0]
    if noise_groups is None:
        print(f"  SKIPPED: {dataset_name} has no SMILES, so no scaffold groups. "
              f"The grouped conditions are undefined here.")
        return {'skipped': 'no_scaffold_groups'}
    noise_groups = np.asarray(noise_groups)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    em_max_rounds = int(kwargs.get('em_max_rounds', EM_MAX_ROUNDS))

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n34: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS)}, got "
            f"{sorted(model_factories)}.")

    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_conditions': conditions,
        'dose_units': noise_dose_units(dataset_name),
        'models': downstream_models,
        'base_rep': base_rep_name,
        'arms': list(ARMS),
        'arm_to_beat': ARM_TO_BEAT,
        'oof_splits': oof_splits,
        'em_max_rounds': em_max_rounds,
        'em_tol_frac_of_label_sd': EM_TOL_FRAC,
        'tau2_floor_frac_of_label_var': TAU2_FLOOR_FRAC,
        'min_group_size_for_center_3plus': MIN_GROUP_SIZE_FOR_CENTER,
        'gap_min_frac': GAP_MIN_FRAC,
        'offsets_from_note': (
            'Every offset arm reads the residuals of ONE group-aware '
            'out-of-fold LightGBM fitted in the scored space on the noisy '
            'training labels. The downstream models then train on the '
            'corrected column. Held-out molecules are predicted with no offset '
            'and graded against clean labels.'),
        'groups_note': (
            'Offsets are per scaffold group from noise_groups_for, the groups '
            'the injector draws over. They are not the outer fold grouping.'),
        'injected_offsets_note': (
            'The injector does not return its per-group offsets. The injected '
            'offset of a group is recomputed as the mean of (noisy - clean) over '
            'its training molecules, which for a group of one still holds that '
            "molecule's own per-molecule error. The recovery correlation is "
            'therefore read on groups of 3 or more training molecules '
            '(corr_groups_of_3_plus); corr_all_groups_contaminated includes '
            'groups of one and two and is not the result.'),
        'features_are_fixed_note': (
            'Each hybrid is built once per pass and fold from the NOISY '
            'training labels and shared by every arm, clean included.'),
        'claim_scope': (
            'The grouped noise is injected; the none pass is the real labels. '
            'Every number is confined to this dataset.'),
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
                'search_score_mean': r['search_score_mean'],
                'search_metric_name': r['search_metric_name'],
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

    passes = _passes(dataset_name, conditions, levels=kwargs.get('levels'),
                     draws=kwargs.get('draws'))
    results['passes'] = [f'{c}|{l:g}|d{d}' for c, l, d in passes]
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}")
    print(f"  {len(passes)} (condition, level, draw) passes; base "
          f"representation {base_rep_name}")

    # acc[condition][level][space][arm][model][pass] = score
    acc, tails, group_rec = {}, {}, {}
    errors = {}
    t0 = time.time()

    for cond, level, draw in passes:
        lvl_key = f'{level:g}'
        cell = acc.setdefault(cond, {}).setdefault(lvl_key, {})
        tcell = tails.setdefault(cond, {}).setdefault(lvl_key, {})
        spec = (None if cond == NO_INJECTION else
                noise_spec_for(dataset_name, cond, level, noise_seed=draw,
                               is_clf=False))
        print(f"\n  --- {cond} / level {lvl_key} / draw {draw} ---")

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pass_key = f'd{draw}f{fold_i}'
            y_val = labels[val_idx]
            y_clean_tr = labels[train_idx]
            g_tr = groups[train_idx] if groups is not None else None
            ng_tr = noise_groups[train_idx]
            if spec is None:
                y0 = y_clean_tr.copy()
            else:
                try:
                    y0, _mask, _prov = inject_train_labels(
                        labels, spec, train_idx=train_idx, is_clf=False,
                        groups=noise_groups, dataset_name=dataset_name)
                except Exception as exc:
                    errors[f'{cond}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                y0 = np.asarray(y0, dtype=float)

            _, _inv_tr, counts = _group_index(ng_tr)
            train_groups = set(np.unique(ng_tr).tolist())
            rec = {
                'pass': pass_key,
                'training_molecules': int(len(y0)),
                'training_groups': int(counts.size),
                'groups_of_one': int((counts == 1).sum()),
                'median_group_size': float(np.median(counts)),
                'groups_of_3_plus': int((counts >= MIN_GROUP_SIZE_FOR_CENTER).sum()),
                'held_out_molecules_in_a_training_group': int(sum(
                    1 for x in noise_groups[val_idx] if x in train_groups)),
                'held_out_molecules': int(len(val_idx)),
                'per_space': {},
            }

            train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                          for k, v in reps.items()}
            val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}
            fold_spaces = {}
            sc = StandardScaler()
            fold_spaces['base'] = [(
                sc.fit_transform(train_reps[base_rep_name]),
                sc.transform(val_reps[base_rep_name]),
                list(downstream_models))]
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
                        errors[f'{cond}|{lvl_key}|{_sp}/{_mnames[0]}|'
                               f'd{draw}f{fold_i}|build'] = (
                            'the hybrid could not be built')
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
                    space_cell = cell.setdefault(space, {})
                    tspace = tcell.setdefault(space, {})
                    oof_fn = (lambda X, y, _g=g_tr: oof_predict(
                        X, y, _g if _g is not None else np.arange(len(y)),
                        n_splits=oof_splits))
                    cols = {'clean': y_clean_tr, 'none': y0}
                    sp_rec = {}
                    try:
                        p = oof_fn(X_tr_s, y0)
                        r = y0 - p
                        off_c = group_center_offsets(r, ng_tr)
                        off_c3 = group_center_offsets(
                            r, ng_tr, min_size=MIN_GROUP_SIZE_FOR_CENTER)
                        cols['group_center'] = y0 - off_c
                        cols['group_center_3plus'] = y0 - off_c3
                        off_ri, info = random_intercept_offsets(
                            X_tr_s, y0, ng_tr, oof_fn,
                            max_rounds=em_max_rounds)
                        cols['random_intercept'] = y0 - off_ri
                        sp_rec['random_intercept'] = info
                        _one = counts[_inv_tr] == 1
                        sp_rec['mean_abs_offset_groups_of_one_group_center'] = (
                            float(np.mean(np.abs(off_c[_one])))
                            if _one.any() else float('nan'))
                        if cond == 'grouped_shifted':
                            sp_rec['offset_recovery'] = {
                                'group_center': offset_recovery(
                                    off_c, y0, y_clean_tr, ng_tr),
                                'random_intercept': offset_recovery(
                                    off_ri, y0, y_clean_tr, ng_tr),
                            }
                    except Exception as exc:
                        errors[f'{cond}|{lvl_key}|{_ekey}|d{draw}f{fold_i}|'
                               f'offsets'] = f'{type(exc).__name__}: {exc}'
                    rec['per_space'][_ekey] = sp_rec

                    for arm in ARMS:
                        if arm not in cols:
                            continue
                        for mname in cell_models:
                            try:
                                pred = _fit_predict(
                                    model_factories[mname], X_tr_s, cols[arm],
                                    X_va_s, metric_name)
                                s = float(metric_fn(y_val, pred))
                            except Exception as exc:
                                errors[f'{cond}|{lvl_key}|{_ekey}|{arm}|{mname}|'
                                       f'd{draw}f{fold_i}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            (space_cell.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = s
                            (tspace.setdefault(arm, {})
                             .setdefault(mname, {})[pass_key]) = tail_scores(
                                y_val, pred)

            group_rec.setdefault(cond, {}).setdefault(lvl_key, []).append(rec)
            print(f"    {cond} {lvl_key} d{draw} fold {fold_i}: "
                  f"{rec['training_groups']} groups, {rec['groups_of_one']} of "
                  f"one, median size {rec['median_group_size']:g} "
                  f"[{time.time() - t0:.0f}s]")

    results['per_condition'] = convert_numpy(acc)
    results['tail_per_condition'] = convert_numpy(tails)
    results['groups_per_pass'] = convert_numpy(group_rec)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, downstream_models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _paired(base, got, lower_is_better):
    """One arm against another, on the passes both scored. Positive means better."""
    shared = sorted(set(base) & set(got))
    gains = np.array([(base[k] - got[k]) if lower_is_better
                      else (got[k] - base[k]) for k in shared], dtype=float)
    gains = gains[np.isfinite(gains)]
    scale = max(abs(float(np.mean(list(base.values())))) if base else 1.0, 1.0)
    tol = 1e-9 * scale
    return {
        'n_passes_paired': int(gains.size),
        'passes_won': int((gains > tol).sum()),
        'passes_lost': int((gains < -tol).sum()),
        'passes_tied': int((np.abs(gains) <= tol).sum()),
        'tie_tolerance': float(tol),
        'mean_gain': float(gains.mean()) if gains.size else float('nan'),
        'spread_over_pass_gains': (float(gains.std()) if gains.size
                                   else float('nan')),
        'pass_gains': [float(x) for x in gains],
    }


def _mean_on(d, keys):
    vals = [d[k] for k in keys if k in d and np.isfinite(d[k])]
    return float(np.mean(vals)) if vals else float('nan')


def _summarise(acc, models, metric_name, lower_is_better):
    """Every arm against `none`, paired per pass. A pass is one (draw, fold)."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against the noisy labels left as they are, '
                        'paired per pass; a pass is one (draw, fold)'),
           'per_condition': {}}
    for cond, by_level in acc.items():
        crow = {}
        for lvl, by_space in by_level.items():
            lrow = {}
            for space, by_arm in by_space.items():
                none = by_arm.get(ARM_TO_BEAT) or {}
                clean = by_arm.get('clean') or {}
                srow = {}
                for m in models:
                    b = none.get(m) or {}
                    if not b:
                        continue
                    row = {'none_mean': _mean_on(b, list(b)), 'per_arm': {}}
                    for arm in ARMS:
                        if arm == ARM_TO_BEAT:
                            continue
                        g = (by_arm.get(arm) or {}).get(m) or {}
                        if not g:
                            continue
                        entry = {'vs_none': _paired(b, g, lower_is_better)}
                        if arm != 'clean':
                            c = clean.get(m) or {}
                            keys = sorted(set(c) & set(b) & set(g))
                            if keys:
                                sh, why = recovery_share(
                                    _mean_on(c, keys), _mean_on(b, keys),
                                    _mean_on(g, keys), lower_is_better)
                            else:
                                sh, why = None, 'no pass scored all three'
                            entry['recovery_share'] = sh
                            entry['recovery_share_refused_because'] = why
                        row['per_arm'][arm] = entry
                    srow[m] = row
                lrow[space] = srow
            crow[lvl] = lrow
        out['per_condition'][cond] = crow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against the noisy labels "
          f"left as they are)")
    for cond, crow in s['per_condition'].items():
        for lvl, lrow in crow.items():
            print(f"\n  {cond} level {lvl}:")
            for space, srow in lrow.items():
                for m, row in srow.items():
                    print(f"    {space}/{m}: none {row['none_mean']:.4f}")
                    for arm, e in row['per_arm'].items():
                        v = e['vs_none']
                        sh = e.get('recovery_share')
                        tail = (f", recovers {sh:.2f} of the clean-to-noisy gap"
                                if sh is not None else '')
                        print(f"      {arm:20s} won {v['passes_won']} of "
                              f"{v['n_passes_paired']} passes, mean "
                              f"{v['mean_gain']:+.4f} (spread "
                              f"{v['spread_over_pass_gains']:.4f}){tail}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N34: scaffold-group offsets -- group mean or random intercept')
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'One of {GROUPED_CONDITIONS}. Default: both')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels. Default: the shared grid')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates (default '
                             f'{",".join(str(d) for d in NOISE_DRAWS)})')
    parser.add_argument('--em-max-rounds', type=int, default=None,
                        help=f'EM rounds for the random intercept (default '
                             f'{EM_MAX_ROUNDS})')
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
    if args.em_max_rounds:
        kw['em_max_rounds'] = args.em_max_rounds
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = kw.get('noise_condition') or 'grouped'
    if args.tag:
        suffix = f'{suffix}_{args.tag}'
    save_experiment_results(res, args.output, 'n34_group_random_effects',
                            args.dataset, filename_suffix=suffix)


if __name__ == '__main__':
    main()
