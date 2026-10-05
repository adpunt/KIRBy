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
the outer folds (noise_helpers.noise_outer_folds) use. So a held-out molecule can share a noise group with training
molecules; the count of such held-out molecules is written per pass
(`held_out_molecules_in_a_training_group`). It does not matter for scoring: every
arm predicts held-out molecules with the fitted model alone, no offset added,
and grades them against their CLEAN labels. An offset only changes what the
model is trained on.

THE CONDITIONS.
  grouped_shifted   a shared offset per group plus a per-molecule error, with
                    0.62 of the variance at group level; the offsets are not
                    centred (`noiseInject/core.py:_draw_grouped_shifted`).
                    One of the six decided kinds (KNN_SEARCH_SPACE_RECORD.md,
                    "The noise grid", Adelaide 2026-10-05).
  grouped_wider     20% of groups get a 3x wider error; zero mean, so there is
                    no offset to find. The arms should do nothing here.
                    OPEN (chat 5, 2026-10-05): the noise-grid decision cut
                    this kind as gaussian repeated. n34 keeps it as its
                    no-offset control until Adelaide says otherwise.
  none              level 0.0: nothing injected. Tests whether the REAL labels
                    carry group offsets. Run once per fold, not once per
                    condition, because level 0.0 is the same column under both.
Levels are the decided grid for the dataset's group
(noise_helpers.decided_levels): 0, 0.5, 1.0, 1.5 on full_knn datasets and 0,
1.0 on fast_knn ones, each as a fraction of the clean training-label SD.
Three draws per non-zero level; level 0 once.

THE ARMS. Each of the dataset's two decided models (DECISIONS.md Step 1), each
in its own 4 configurations per outer fold: its 2 best single representations
(base_1, base_2) and its 2 best general hybrids (hybrid, hybrid__general2),
all ranked by searches on that fold's training rows. 2 models x 4 = 8 fitted
configurations (chat 5, 2026-10-05). Paired against `none`.

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
oof_predict`), fitted in the configuration the arm is scored in, on this
fold's noisy training labels. The configuration's model then trains on the
corrected column. That is n30's arrangement for its lifts. Out-of-fold, not
in-fold: an in-fold fit has memorised the offset it is being asked to reveal,
and its residual would be near zero on every row.
OPEN (chat 5, 2026-10-05): this LightGBM is part of the method, not a
downstream model, so it is kept on every dataset, including the six whose two
decided models are svm and dnn.

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

WHERE THE FEATURES COME FROM (chat 5, 2026-10-05). Per outer fold,
noise_helpers.fold_configurations runs the fold's searches on its training
rows (clean labels, the dataset's decided grid, pruning off) and names each
decided model's 2 best single representations and 2 best general hybrids.
Nothing is read from a whole-dataset search. The base matrices are built
once per fold; each hybrid is rebuilt once per pass from that pass's NOISY
training labels (noise_helpers.fold_model_inputs, with the fold search's own
representation scores) and every arm shares it, clean included. Which
representations and recipes each model got on each fold is written to
`per_fold[fold]['models']` (configuration_record).
n34 measures nothing with neighbours or distances, so the distance hybrid
and the similarity hybrid are not used (handoff 2.3 item 1).
Until 2026-10-05 the recipes and the base representations came from
whole-dataset stored searches, for five models, with up to six hybrid spaces
per model (two general and one distance hybrid in each of two families).

Usage:
    python n34_group_random_effects.py --dataset herg_ki --mode full
    python n34_group_random_effects.py --dataset logd --noise-condition grouped_shifted
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import (
    noise_groups_for, noise_dose_units,
)
from experiments.noise.noise_helpers import (
    tail_scores,
    # CHAT 5 ALIGNMENT (2026-10-05): folds, the 8 configurations, the decided
    # noise grid.
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_levels,
    dataset_knn_group, level_unit, CONFIGURATION_SPACES,
)
from experiments.noise.n07_remediation import _predict_for_metric
# The D-versus-D' rows, per-pass checkpoint, cached fits and controls shared
# by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    fit_predict_cached, ledger_signature, PassLedger, restart_flag, arm_rows,
    arm_result, control_twins, twin_scores, controls_enabled, replicate_seed,
    _nc,
)

EXP_ID = 'n34_group_random_effects'

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

# Bumped whenever what a pass writes changes, so a checkpoint or rows file of
# an older layout is set aside, never read back (PassLedger's signature reads
# the result header, which carries this). 2026-10-05: per-fold searches, the
# two decided models, model_input is the configuration name alone.
RESULT_LAYOUT = 'chat5_per_fold_2026_10_05'

# Each decided model's 4 fitted configurations per outer fold
# (noise_helpers.CONFIGURATION_SPACES). The base matrices do not depend on the
# labels, so they are built once per fold; the hybrids choose features from
# the labels, so they are rebuilt per pass from the noisy labels.
BASE_SPACES = ('base_1', 'base_2')
HYBRID_SPACES = tuple(sp for sp in CONFIGURATION_SPACES
                      if sp not in BASE_SPACES)


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
    """(share, None) or (None, why). n30's rule (n30:341-352).

    The gap is signed by the metric: positive when the noise made the score
    worse. Until 2026-09-25 this refused only on abs(gap), so a level where
    the noise IMPROVED the score passed the floor with a negative gap, and the
    share came out with its sign flipped -- a method that moved the score
    toward clean read as recovering a negative share. It now refuses whenever
    the gap is under the floor, negative gaps included, as n30 (n30:348) and
    n32 do. n39 imports this function and gets the same rule."""
    vals = (clean, noisy, arm)
    if not all(np.isfinite(v) for v in vals):
        return None, 'a mean is not finite'
    gap = (noisy - clean) if lower_is_better else (clean - noisy)
    floor = GAP_MIN_FRAC * max(abs(clean), 1e-12)
    if gap < floor:
        if gap < 0:
            return None, (f'the noise improved the score: the clean-to-noisy '
                          f'gap is {gap:+.4g}, so there is nothing to recover')
        return None, (f'the clean-to-noisy gap {gap:+.4g} is under '
                      f'{GAP_MIN_FRAC:.0%} of the clean score')
    got = (noisy - arm) if lower_is_better else (arm - noisy)
    return float(got / gap), None


# ── Runner ────────────────────────────────────────────────────────────────

def _fit_predict(factory, X_tr, y_tr, X_va, metric_name):
    m = factory()
    m.fit(X_tr, y_tr)
    return _predict_for_metric(m, X_va, metric_name)


def _passes(data, dataset_name, conditions, levels=None, draws=None):
    """(condition, level, draw). Level 0.0 appears once, under NO_INJECTION.

    Levels are the decided grid for this dataset's group
    (noise_helpers.decided_levels, Adelaide 2026-10-05): 0, 0.5, 1.0, 1.5 on
    full_knn and 0, 1.0 on fast_knn. Until 2026-10-05 they came from
    dataset_policy.noise_levels. A `levels` kwarg (a smoke) overrides."""
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    zero_done = False
    for cond in conditions:
        lv = (decided_levels(data, dataset_name, cond)
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


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N34: group mean subtraction against a shrunk random intercept.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    What each outer fold does, in order:
      1. Folds are noise_helpers.noise_outer_folds: the splits the per-fold
         searches are filed under, the count from config['n_splits'] or
         KIRBY_NOISE_N_FOLDS. Until 2026-10-05 they were _shared.cv_split.
      2. noise_helpers.fold_configurations runs this fold's searches on its
         training rows and gives each of the dataset's two decided models its
         2 best base representations and its 2 best general hybrids.
      3. base_1 and base_2 are built once per fold. For every (condition,
         level, draw) pass the two hybrids are rebuilt from that pass's noisy
         training labels, with the fold search's own representation scores
         (dirty-run bug 1).
      4. In each of the 8 configurations, the out-of-fold LightGBM gives the
         residuals, the offset arms correct the labels, and the
         configuration's own model is fitted once per arm.
    Each (condition, level, draw, fold) pass is checkpointed as it finishes,
    and its rows go to disk first.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass."""
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
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
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
    metric = (metric_name, metric_fn, lower_is_better)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    em_max_rounds = int(kwargs.get('em_max_rounds', EM_MAX_ROUNDS))

    # R1 (chat 5, 2026-10-05): the dataset's two decided models
    # (DECISIONS.md Step 1), nothing else; knn runs only inside the distance
    # search. Replaces downstream_model_list, which took config['models']
    # (five models from get_config) and added rf and svm. A `models` kwarg
    # (a smoke, a direct caller) can only narrow the decided pair; a name
    # outside it is dropped and written down.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        models = [m for m in asked if m in decided]
        models_not_decided = [m for m in asked if m not in decided]
    else:
        models = list(decided)
        models_not_decided = []
    if config.get('models'):
        print(f"  config['models'] {list(config['models'])} is not read: "
              f"n34 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n34: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    knn_grp = dataset_knn_group(data, dataset_name)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'selection_scope': 'fold_train_rows',
        'noise_conditions': conditions,
        'level_unit': level_unit('grouped_shifted', False),
        'knn_group': knn_grp,
        'dose_units': noise_dose_units(dataset_name),
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
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
            'out-of-fold LightGBM fitted in the scored configuration on the '
            'noisy training labels. The configuration\'s model then trains on '
            'the corrected column. Held-out molecules are predicted with no '
            'offset and graded against clean labels.'),
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
            "Each decided model has four configurations per outer fold: its "
            "2 best single representations and its 2 best general hybrids, "
            "all ranked by searches on that fold's training rows with the "
            "clean labels (fold_configurations). The hybrids are rebuilt on "
            "the fold's training molecules from the NOISY labels of each "
            "pass, once, and shared by every arm, clean included."),
        'robustness_note': (
            'damaged_minus_clean in the summary is the score on the noisy '
            'labels left as they are minus the score on the clean labels, per '
            "model and configuration, in the dataset's own metric (read with "
            'lower_is_better), as a mean over draws per fold and then over '
            'folds. No auc_norm is computed here; on fast_knn datasets it is '
            'absent by decision.'),
        'claim_scope': (
            'The grouped noise is injected; the none pass is the real labels. '
            'Every number is confined to this dataset.'),
        'similarity_note': ('n34 finds no neighbours, so it has no similarity '
                            'axis; its rows carry model_input|none|process.'),
        'controls_run': controls_enabled(),
    }

    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]

    passes = _passes(data, dataset_name, conditions,
                     levels=kwargs.get('levels'), draws=kwargs.get('draws'))
    results['passes'] = [f'{c}|{l:g}|d{d}' for c, l, d in passes]
    n_cfg = len(models) * len(CONFIGURATION_SPACES)
    results['fitted_configurations_per_fold'] = n_cfg
    print(f"  {len(labels)} molecules, {n_folds} outer folds (running "
          f"{results['folds_run']}), decided models {models}, group {knn_grp}")
    print(f"  {len(passes)} (condition, level, draw) passes; {n_cfg} fitted "
          f"configurations per fold ({len(models)} models x "
          f"{len(CONFIGURATION_SPACES)}: {list(CONFIGURATION_SPACES)})")

    # acc[condition][level][space][arm][model][pass] = score. space is the
    # configuration name of the model it is keyed under, so base_1 of one
    # model and base_1 of the other can be different representations.
    acc, tails, group_rec = {}, {}, {}
    errors = {}
    # Per outer fold: what each model was fitted on (handoff 2.4 item 11).
    fold_records = {}
    # Configurations that could not be built, per (pass, model, space).
    config_unavailable = {}
    t0 = time.time()
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). The
    # file stem matches main()'s result suffix: the named condition, or
    # 'grouped' for both. The signature reads `results` (RESULT_LAYOUT,
    # models, n_folds) and the folds, so a file of the old layout is set
    # aside, never read back.
    ledger = PassLedger(
        output_dir, EXP_ID, dataset_name, condition or 'grouped',
        signature=ledger_signature(
            data, config, results, [(tr, va) for _, tr, va in folds],
            extra={'layout': RESULT_LAYOUT, 'passes': results['passes']}),
        restart=bool(force_rerun) or restart_flag(kwargs))
    if ledger.restored():
        st = ledger.restored()
        acc, tails, group_rec, errors = (st['acc'], st['tails'],
                                         st['group_rec'], st['errors'])
        fold_records = st.get('fold_records', {})
        config_unavailable = st.get('config_unavailable', {})
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'acc': acc, 'tails': tails, 'group_rec': group_rec,
                'errors': errors, 'fold_records': fold_records,
                'config_unavailable': config_unavailable}

    for fold_i, train_idx, val_idx in folds:
        _keys = [f'{c}|{l:g}|d{d}f{fold_i}' for c, l, d in passes]
        if (str(fold_i) in fold_records
                and all(ledger.pass_done(k) for k in _keys)):
            print(f"\n  Fold {fold_i}: every pass is on disk in the checkpoint")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            why = (f'the per-fold searches failed: {type(exc).__name__}: '
                   f'{exc}')
            print(f"    {why}")
            errors[f'f{fold_i}|fold_configurations'] = why
            fold_records[str(fold_i)] = {'error': why}
            continue
        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        ng_tr = noise_groups[train_idx]

        # base_1 and base_2 of each model: no features are chosen from the
        # labels, so one matrix per fold serves every pass. FeatureScaler
        # inside (0/1 features unscaled), float32 kept: until 2026-10-05 the
        # fold loop cast every representation to float64 and standardised
        # every feature (handoff 2.3 items 17 and 18).
        base_inputs = {}
        frec = {'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_clean_tr,
                factory=model_factories[m], metric=metric,
                spaces=BASE_SPACES)
            base_inputs[m] = (ins, unav)
            rec = configuration_record(entry, names)
            rec['base_unavailable'] = {sp: why for sp, why in unav.items()
                                       if sp in BASE_SPACES}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['training_row_score_note'] = (
                "hybrids[*].score_mean is the fold search's score of that "
                "hybrid on this fold's training rows (its own inner folds), "
                "in the search metric. The base representations' search "
                "scores are not handed through fold_configurations. The "
                "held-out scores are in held_out_score below, keyed by "
                "condition|level|draw, then configuration, then arm.")
            rec['held_out_score'] = {}
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        # A resumed fold keeps the held-out scores its finished passes wrote.
        _prev = (fold_records.get(str(fold_i)) or {}).get('models') or {}
        for m, rec in frec['models'].items():
            rec['held_out_score'].update(
                (_prev.get(m) or {}).get('held_out_score') or {})
        fold_records[str(fold_i)] = frec

        for cond, level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'd{draw}f{fold_i}'
            ledger_key = f'{cond}|{lvl_key}|{pass_key}'
            if ledger.pass_done(ledger_key):
                continue
            cell = acc.setdefault(cond, {}).setdefault(lvl_key, {})
            tcell = tails.setdefault(cond, {}).setdefault(lvl_key, {})
            print(f"\n    --- fold {fold_i}, {cond} / level {lvl_key} / "
                  f"draw {draw} ---")
            pass_rows = []
            if cond == NO_INJECTION:
                y0 = y_clean_tr.copy()
            else:
                # The decided injector (noise_helpers.inject_condition), over
                # the scaffold groups noise_groups_for gives, as before.
                try:
                    y0, _mask, _prov = inject_condition(
                        data, dataset_name, cond, float(level), int(draw),
                        train_idx=train_idx, groups=noise_groups)
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

            # Each model's 4 configurations: base_1 and base_2 from the fold,
            # the two hybrids rebuilt from this pass's noisy labels.
            configs = []
            for m in models:
                entry = entries.get(m)
                hyb, unav_h, _names = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=train_idx,
                    val_idx=val_idx, y_train=y0,
                    factory=model_factories[m], metric=metric,
                    spaces=HYBRID_SPACES)
                b_ins, b_unav = base_inputs[m]
                for sp in CONFIGURATION_SPACES:
                    src_ = hyb if sp in HYBRID_SPACES else b_ins
                    unav_ = unav_h if sp in HYBRID_SPACES else b_unav
                    pair = src_.get(sp)
                    if pair is None:
                        (config_unavailable.setdefault(ledger_key, {})
                         .setdefault(m, {})[sp]) = (unav_.get(sp)
                                                    or 'not built')
                        continue
                    configs.append((m, sp, pair[0], pair[1]))

            for mname, space, X_tr_s, X_va_s in configs:
                _ekey = f'{space}/{mname}'
                space_cell = cell.setdefault(space, {})
                tspace = tcell.setdefault(space, {})
                # Cached on the label hash (2026-09-25): the first call
                # is the plain out-of-fold p n30-n33 compute for the
                # draw; each EM round's labels have their own entry.
                oof_fn = (lambda X, y, _g=g_tr: _nc.cached_oof_predict(
                    dataset_name, X, y,
                    _g if _g is not None else np.arange(len(y)),
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

                pass_scores = {}
                for arm in ARMS:
                    if arm not in cols:
                        continue
                    try:
                        # Through noise_cache for a seeded model
                        # (2026-09-25); the same prediction.
                        pred = fit_predict_cached(
                            dataset_name, mname, model_factories[mname],
                            X_tr_s, cols[arm], X_va_s, metric_name)
                        s = float(metric_fn(y_val, pred))
                    except Exception as exc:
                        errors[f'{cond}|{lvl_key}|{_ekey}|{arm}|'
                               f'd{draw}f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    (space_cell.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = s
                    pass_scores.setdefault(arm, {})[mname] = s
                    (tspace.setdefault(arm, {})
                     .setdefault(mname, {})[pass_key]) = tail_scores(
                        y_val, pred)
                (frec['models'][mname]['held_out_score']
                 .setdefault(f'{cond}|{lvl_key}|d{draw}', {}))[space] = {
                    arm: (pass_scores.get(arm) or {}).get(mname)
                    for arm in ARMS}

                # ── Controls (KIRBY_NOISE_CONTROLS=1): every offset arm
                # is a label move, twinned by controls.py. ──
                for arm in ('group_center', 'group_center_3plus',
                            'random_intercept'):
                    if arm not in cols:
                        continue
                    twins = control_twins(
                        arm_result(y0, y=cols[arm]), X_tr_s, y0,
                        (replicate_seed(arm, cond), draw, fold_i,
                         int(round(level * 1000))))
                    for tw, by_m in twin_scores(
                            dataset_name, twins, model_factories,
                            [mname], X_va_s, y_val, metric).items():
                        pass_scores[f'{arm}|{tw}'] = by_m
                # ── D against D': `none` is D, each offset arm a D'. ──
                # model_input is the configuration name alone and the model
                # is its own column (dirty-run bug 13: until 2026-10-05 n32-n35
                # wrote 'base_1/rf', n30 and n31 'base_1').
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric, condition=cond, level=level, draw=draw,
                    fold=fold_i, model_input=space,
                    scores_by_arm=pass_scores, reference=ARM_TO_BEAT)

            group_rec.setdefault(cond, {}).setdefault(lvl_key, []).append(rec)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(ledger_key, _state())
            print(f"    {cond} {lvl_key} d{draw} fold {fold_i}: "
                  f"{rec['training_groups']} groups, {rec['groups_of_one']} of "
                  f"one, median size {rec['median_group_size']:g} "
                  f"[{time.time() - t0:.0f}s]")

    results['per_fold'] = convert_numpy(fold_records)
    results['configurations_unavailable'] = convert_numpy(config_unavailable)
    results['per_condition'] = convert_numpy(acc)
    results['tail_per_condition'] = convert_numpy(tails)
    results['groups_per_pass'] = convert_numpy(group_rec)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, models, metric_name, lower_is_better))
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
                    # Robustness in the dataset's own metric (chat 5,
                    # 2026-10-05): noisy-labels score minus clean-labels
                    # score, averaged over draws within a fold, then
                    # described over folds (fold_spread: mean and sd, or
                    # "one fold, no spread"). Read with lower_is_better.
                    c_m = clean.get(m) or {}
                    by_fold = {}
                    for k in sorted(set(b) & set(c_m)):
                        if np.isfinite(b[k]) and np.isfinite(c_m[k]):
                            by_fold.setdefault(k.split('f')[-1], []).append(
                                float(b[k]) - float(c_m[k]))
                    row['damaged_minus_clean'] = fold_spread(
                        [float(np.mean(v)) for _f, v in sorted(by_fold.items())])
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

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = kw.get('noise_condition') or 'grouped'
    if args.tag:
        suffix = f'{suffix}_{args.tag}'
    save_experiment_results(res, args.output, EXP_ID,
                            args.dataset, filename_suffix=suffix)


if __name__ == '__main__':
    main()
