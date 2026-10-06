#!/usr/bin/env python3
"""N38: noise sized by the model's own error, not by the label SD.

Paper: Noise.

WHERE THIS CAME FROM. The third research report (2026-09-25, "Noise Mitigation
... Organised by Noise Type: Dense Gaussian Noise vs Sparse Gross Errors") says
a gross error is findable only when it is several times the out-of-fold
residual SD s_r, and that every other injected condition should be sized
against the model's error too:

  B2  the detection limit is d' = error size / s_r. With 5% of molecules moved
      and a flag at |r| > 2.5 s_r, power is about 31% at d' = 2 and 7% at d' = 1.
  D4  dense noise at sigma^2 = 0.1, 0.3, 1.0 x OOF MSE; sparse shifts of d' = 1,
      2, 4 residual SDs on 2, 5 or 10% of molecules, random sign or one
      direction; noise whose size follows a feature the model sees, and one
      whose size follows something it cannot see; start from a clean
      semi-synthetic label.

What already existed, read 2026-09-25: every NoiseInject condition is dosed in
label SDs (src/kirby/noise_spec.py); n31's shift_only_* moves 5 or 10% UP by
0.5, 1 or 2 LABEL SDs. Nothing was sized by the model's error.

THE CONDITIONS are four of noiseInject.RELATIVE_CONDITIONS (N38_CONDITIONS),
drawn by noiseInject's RelativeNoiseInjector through
noise_helpers.inject_relative_train_labels. Their unit, resid_sd, is the SD of
lightgbm's out-of-fold residual (noise_helpers.oof_residual_sd) on THIS fold's
clean training labels, in ONE representation, `resid_sd_rep` (see THE UNIT'S
REPRESENTATION), scaffold-grouped. It is written per fold and per label base,
beside every result.

  dense_oofmse          every training label + N(0, level * resid_sd^2).
  sparse_dprime_up_p10  10% of training molecules moved up by
                        level * resid_sd.
  hetero_seen           every label, SD proportional to heavy-atom count (the
                        representation sees molecular size), mean variance
                        level * resid_sd^2.
  hetero_hidden         every label, SD from a per-molecule lognormal draw
                        nothing in the features carries, same mean variance.

The molecules and signs a draw picks do not depend on the level, so the level
sweep nests: level 4 moves the same molecules as level 1, four times as far.

LEVELS (2026-10-05, chat 5). sparse_dprime_up_p10 is one of the six kinds the
noise grid keeps (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05), so it runs at
the decided levels (noise_helpers.decided_levels): 0, 0.5, 1.0, 1.5 residual
SDs on a dataset of 5000 molecules or fewer, 0 and 1.0 above. dense_oofmse,
hetero_seen and hetero_hidden keep n38's own grid (noise_helpers.
relative_levels): 0.1, 0.3, 1.0 x OOF MSE, with 0. Level 0 is the labels as
delivered and runs once; every other level runs 3 draws
(noise_helpers.draws_for).

CUT 2026-10-06 (Adelaide delegated): sparse_dprime_p02, sparse_dprime_p05,
sparse_dprime_p10, sparse_dprime_up_p02 and sparse_dprime_up_p05, the five
sparse conditions the noise grid decision (2026-10-05) cut. The comparison
across shares (2, 5, 10% of molecules) and across sign (random versus up) is
not run.

THE LABEL BASE is a second axis.

  real            the dataset's labels. They already carry real error of
                  unknown size, so "clean" is only as clean as the assay.
  semi_synthetic  built per fold from the TRAINING labels only:
                    p_tr   lightgbm's out-of-fold prediction of every training
                           molecule (the same fit that gives resid_sd);
                    r      the pool of its out-of-fold residuals y_tr - p_tr;
                    p_va   the prediction for each held-out molecule of one
                           lightgbm fitted on the whole training fold;
                  clean training label = p_tr + a residual drawn from r with
                  replacement; held-out truth = p_va + a residual drawn from r.
                  No held-out label is ever read. The signal is known here --
                  it is p -- and every residual is independent of the
                  features by construction, so NONE of the clean label's
                  scatter is structure a better model could have learned. Each
                  arm is also scored against p_va alone ('vs_signal'), which a
                  real dataset cannot offer.
                  Caveat: the signal is a lightgbm fitted in resid_sd_rep,
                  so lightgbm on that representation fits it more easily than
                  any other model or space does. Compare arms within one model
                  and space, not models against each other.
                  resid_sd is re-measured on the semi-synthetic training
                  labels, so the conditions are sized against the error the
                  model makes on THIS base.

DETECTION (sparse conditions only; under the others every label moves).
Graded by detection_scores against the molecules the injector moved:

  AUROC and average precision of |y - p|, p the group-aware out-of-fold lightgbm
  prediction (kirby.utils.noisy_labels.oof_predict) in each space.

  power and precision at |y - p| > 2.5 s, at two choices of s:
    s = resid_sd              the unit the condition was sized in (clean labels,
                              resid_sd_rep). Known only because we injected.
    s = 1.4826 * MAD(y - p)   what a user without the clean labels can compute.

  The report's normal theory, beside them. A moved molecule's residual is
  N(+-d' s_r, s_r^2) and a clean one's N(0, s_r^2), so for |r| > t s_r:
    power     = Phi(-t - d') + 1 - Phi(t - d')
    false pos = 2 Phi(-t)
    precision = pi power / (pi power + (1 - pi) false pos),   pi = the share
    AUROC     = P(|Z + d'| > |Z'|) = integral (2 Phi(|x|) - 1) phi(x - d') dx
  The sign does not enter: |Z + d'| and |Z - d'| have one distribution, so the
  theory holds for the one-direction condition run here.
  Checked: d' = 2, pi = 0.05 gives power 0.309, false positives 0.0124,
  precision 0.568 -- the report's 31%, 1.2%, 58%.
  The theory is evaluated twice: at the nominal d' (in resid_sd_rep's resid_sd),
  and at d'_space = shift / SD of the unmoved molecules' residuals in that
  space, the d' the space's own model actually faced. The second uses the mask
  and is written as a description, never as a detector.

THE ARMS. Each of the dataset's two decided models is fitted on what each arm
leaves, in each of its four feature spaces, and scored on held-out molecules
against their UNTOUCHED labels.

  clean                        the training labels before injection.
  no_cleaning                  the corrupted labels. The arm to beat.
  winsorise_c<c>               n30's winsorise: y -> p + clip(y - p, +-c s),
                               s = 1.4826 MAD(y - p), p the out-of-fold
                               prediction.
  winsorise_c<c>_random/seed<k>  the SAME label changes winsorising made,
                               applied to the same NUMBER of molecules picked
                               at random. Whether winsorising gains by where it
                               acts or merely by acting.
  robust_loss                  n33's loss swap for each downstream model at
                               c = ROBUST_C: lightgbm_huber, xgboost
                               pseudohuber, dnn huber, rf_l1. svm and gp have
                               no swap and are not scored in this arm (written
                               as a note, not an error).
  itlm_a<alpha>                n32's ITLM, 4 rounds, out-of-fold loss.
  itlm_a<alpha>_random/seed<k> the same number removed at random.
  invvar_ngboost               n33's weights: 1 / sigma^2 from an out-of-fold
                               NGBoost (Normal), floored, averaging one. Not
                               run for gp: GaussianProcessRegressor.fit takes
                               no sample_weight.

WHERE THE FEATURES COME FROM (2026-10-05, chat 5, HANDOFF_EXPERIMENT_ALIGNMENT
2.4). The models are the dataset's two decided models (DECISIONS.md Step 1,
decided_grid.decided_models). Per outer fold (noise_helpers.noise_outer_folds)
and per model, four feature spaces, all chosen by searches on that fold's
training rows with the clean labels (noise_helpers.fold_configurations):
`base_1` and `base_2`, the model's 2 best single representations, and `hybrid`
and `hybrid__general2`, its 2 best general hybrids (a second hybrid whose
selected features equal the first's is skipped for the next). 2 models x 4 =
8 fitted configurations. Each hybrid is rebuilt on the fold's training rows
from the CORRUPTED training labels of the pass (noise_helpers.
fold_model_inputs, with the fold search's own representation scores), and
every arm of that model shares it. n38 finds no neighbours, so the distance
hybrid and the similarity hybrid are not used. Which configurations each fold
used is under `folds` in the result file.
Until 2026-10-05 the recipes and base representations were read from the
whole-dataset stored searches of five models, up to 6 hybrids per model
(2 general + 1 distance, under the predictive and the similarity objective).

THE UNIT'S REPRESENTATION. resid_sd and the semi-synthetic signal must be one
per fold for every space and model, so they are computed in ONE
representation, `resid_sd_rep`: the dataset's first pruned representation as
loaded (the first key of data['reps']), FeatureScaler fitted on the fold's
training rows. This is the rule noise_helpers.inject_condition uses, so on
real labels n38's sparse_dprime_up_p10 damage equals the other noise
experiments' (DECIDED 2026-10-06, Adelaide delegated). Written per fold under
fold_records. Until 2026-10-06 it was, per fold, the rank-1 single
representation of the first decided model from that fold's search; until
2026-10-05 the rank-1 representation of rf in the whole-dataset search.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. Nothing here measures real label error.

Usage:
    python n38_noise_relative_to_model_error.py --dataset logd --mode full
    python n38_noise_relative_to_model_error.py --dataset herg_ki --mode full \
        --noise-condition sparse_dprime_up_p10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import functools
import time

import numpy as np
from scipy import integrate
from scipy.stats import norm

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, as_rep_dtype,
)
from experiments.decided_grid import decided_models
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds, the
# 8 fitted configurations per fold, and the decided levels come from
# noise_helpers' shared block. Removed: cv_split, DOWNSTREAM_MODELS (through
# n32.downstream_model_list), get_best_search_for_downstream,
# noise_base_reps_by_model, pool_recipe_per_downstream_model,
# pool_recipes_by_family, the per-cell hybrid build and its StandardScaler
# (whole-dataset searches, five models, up to 6 hybrids per model).
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, tail_scores, sweep_conditions,
    RELATIVE_CONDITIONS, RELATIVE_SPARSE_SHARES, relative_levels,
    inject_relative_train_labels, oof_residual_sd, heavy_atom_counts,
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, decided_conditions, decided_levels,
    dataset_knn_group, CONFIGURATION_SPACES, draws_for,
)
from experiments.noise.n07_remediation import _predict_for_metric
from experiments.noise.n30_censored_likelihood import winsorise
from experiments.noise.n32_iterative_and_dual_selection import (
    itlm, random_keep, replicate_seed, _paired, recovery_share, _mean_on,
    _tail_row, _mean_detection, _per_fold_means,
    # The D-versus-D' rows, per-pass checkpoint, cached fits and controls
    # shared by n32-n41 (n32's shared block, 2026-09-25).
    method_setting, fit_predict_cached, ledger_signature, PassLedger,
    restart_flag, arm_rows, arm_result, control_twins, twin_scores,
    controls_enabled, _nc, _ss,
)
from experiments.noise.n33_robust_training import (
    make_swap, robust_scale, oof_ngboost_sigma, invvar_weights,
)
from kirby.find_best_hybrid import get_model
from kirby.utils.noisy_labels import oof_predict

EXP_ID = 'n38_noise_relative_to_model_error'
# DECIDED 2026-10-06 (Adelaide delegated): n38 runs four of the nine
# conditions sized by the model's error: sparse_dprime_up_p10 (the one the
# noise grid keeps, at the decided levels), dense_oofmse, hetero_seen and
# hetero_hidden (at n38's own grid). The five sparse conditions the noise grid
# cut are not run: sparse_dprime_p02, sparse_dprime_p05, sparse_dprime_p10,
# sparse_dprime_up_p02, sparse_dprime_up_p05.
N38_CONDITIONS = ('sparse_dprime_up_p10', 'dense_oofmse', 'hetero_seen',
                  'hetero_hidden')
CUT_CONDITIONS = ('sparse_dprime_p02', 'sparse_dprime_p05',
                  'sparse_dprime_p10', 'sparse_dprime_up_p02',
                  'sparse_dprime_up_p05')
DEFAULT_NOISE_CONDITION = 'sparse_dprime_up_p10'

# Identity of this version's checkpoints and condition files. Files written
# before the alignment carry another signature and are set aside, not read
# back (handoff 2.4 item 11).
ALIGNMENT_VERSION = 'chat5_2026-10-06'

# DECIDED 2026-10-06 (Adelaide delegated): resid_sd and the semi-synthetic
# signal are measured in the dataset's first pruned representation as loaded,
# the rule noise_helpers.inject_condition uses, so n38's damage under
# sparse_dprime_up_p10 on real labels matches the other noise experiments'.
RESID_SD_REP_RULE = ("the dataset's first pruned representation as loaded "
                     "(first key of data['reps']), FeatureScaler fitted on the "
                     "fold's training rows; the rule noise_helpers."
                     "inject_condition uses")

# Arms a model's fit cannot take. invvar_ngboost passes sample_weight, and
# sklearn's GaussianProcessRegressor.fit has none (n33 checked 2026-10-05). gp
# is decided on abagym_Ang2, abagym_VEGF, ginkgo_acsins, ginkgo_hic and
# ginkgo_pr_cho.
ARMS_UNDEFINED_FOR = {'gp': {'invvar_ngboost': (
    'GaussianProcessRegressor.fit takes no sample_weight')}}

LABEL_BASES = ('real', 'semi_synthetic')
NOISE_DRAWS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2)
OOF_SPLITS = 5
WINSOR_C = 1.5
ROBUST_C = 1.5
ITLM_ALPHA = 0.9
ITLM_ROUNDS = 4
DETECT_T = 2.5          # the report's flag threshold, in residual SDs

# n33's swap for each downstream model; svm and gp have none.
ROBUST_SWAP = {
    'lightgbm': f'lightgbm_huber_c{ROBUST_C:g}',
    'xgboost': f'xgboost_pseudohuber_c{ROBUST_C:g}',
    'dnn': f'dnn_huber_c{ROBUST_C:g}',
    'rf': 'rf_l1',
}


# ── The report's normal theory ───────────────────────────────────────────────

def theory_threshold(dprime, share, t=DETECT_T):
    """Power, false-positive rate and precision of |r| > t s_r. See docstring."""
    d = float(dprime)
    power = float(norm.cdf(-t - d) + 1.0 - norm.cdf(t - d))
    fpr = float(2.0 * norm.cdf(-t))
    den = share * power + (1.0 - share) * fpr
    return {'theory_power': power, 'theory_false_positive_rate': fpr,
            'theory_precision': float(share * power / den) if den > 0 else float('nan')}


def theory_auroc_abs(dprime):
    """P(|Z + d'| > |Z'|): the AUROC of |r| when moved residuals sit at +-d'."""
    d = float(dprime)
    if not np.isfinite(d):
        return float('nan')
    f = lambda x: (2.0 * norm.cdf(abs(x)) - 1.0) * norm.pdf(x - d)
    val, _ = integrate.quad(f, d - 12.0, d + 12.0, limit=200)
    return float(val)


def measured_threshold(cmask, abs_r, s, t=DETECT_T):
    """Power and precision of abs_r > t s against the moved molecules."""
    cmask = np.asarray(cmask, dtype=bool)
    if not np.isfinite(s) or s <= 0 or cmask.sum() == 0 or (~cmask).sum() == 0:
        return None
    flag = np.asarray(abs_r) > t * s
    tp = int(np.sum(flag & cmask))
    return {'scale': float(s), 'n_flagged': int(flag.sum()),
            'power': tp / int(cmask.sum()),
            'false_positive_rate': int(np.sum(flag & ~cmask)) / int((~cmask).sum()),
            'precision': tp / int(flag.sum()) if flag.sum() else float('nan')}


# ── Label bases ──────────────────────────────────────────────────────────────

def label_base_for_fold(base, Xb_tr, Xb_va, y_tr, g_tr, seed,
                        oof_splits=OOF_SPLITS):
    """(clean training labels, held-out truth, held-out signal or None,
    resid_sd, provenance) for one label base on one fold. Reads no held-out
    label under semi_synthetic."""
    y_tr = np.asarray(y_tr, dtype=float)
    sd_real, p_tr = oof_residual_sd(Xb_tr, y_tr, groups=g_tr, n_splits=oof_splits)
    if base == 'real':
        return y_tr, None, None, sd_real, {'resid_sd_real_labels': sd_real}
    if base != 'semi_synthetic':
        raise ValueError(f'n38: unknown label base {base!r}')
    rng = np.random.RandomState(int(seed))
    pool = y_tr - p_tr
    m = get_model('lightgbm', 'regression')
    m.fit(Xb_tr, y_tr)
    p_va = np.asarray(m.predict(Xb_va), dtype=float)
    y_tr_syn = p_tr + rng.choice(pool, size=len(y_tr), replace=True)
    y_va_syn = p_va + rng.choice(pool, size=len(p_va), replace=True)
    sd_syn, _ = oof_residual_sd(Xb_tr, y_tr_syn, groups=g_tr, n_splits=oof_splits)
    return y_tr_syn, y_va_syn, p_va, sd_syn, {
        'resid_sd_real_labels': sd_real, 'resid_sd_semi_synthetic': sd_syn,
        'signal_sd_train': float(np.std(p_tr)),
        'residual_pool_sd': float(np.std(pool))}


# ── Arms ─────────────────────────────────────────────────────────────────────

def random_twin_of_changes(y, changed, delta, seed):
    """The same label changes, applied to the same number of random molecules."""
    y = np.asarray(y, dtype=float)
    n_ch = int(np.sum(changed))
    out = y.copy()
    if n_ch == 0:
        return out
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(len(y), n_ch, replace=False)
    out[pick] = y[pick] + rng.permutation(np.asarray(delta)[changed])
    return out


def build_arm_names(seeds, winsor_c=WINSOR_C, alpha=ITLM_ALPHA):
    out = ['clean', 'no_cleaning', f'winsorise_c{winsor_c:g}']
    out += [f'winsorise_c{winsor_c:g}_random/seed{s}' for s in seeds]
    out.append('robust_loss')
    out.append(f'itlm_a{alpha:g}')
    out += [f'itlm_a{alpha:g}_random/seed{s}' for s in seeds]
    out.append('invvar_ngboost')
    return out


def _fit_score(factory, X_tr, y_tr, X_va, truths, metric_fn, metric_name,
               keep=None, sample_weight=None, dataset_name=None,
               model_name=None):
    """One fit scored against each truth; ({truth: score}, tail dict, None) or
    (None, None, why). Tails are against the first truth.

    dataset_name / model_name, added 2026-09-25, route a seeded downstream
    model through noise_cache (n32.fit_predict_cached); the prediction is the
    same. A loss swap is passed with model_name None and fitted here."""
    try:
        pred = np.asarray(fit_predict_cached(
            dataset_name, model_name, factory, X_tr,
            np.asarray(y_tr, dtype=float), X_va, metric_name, keep=keep,
            sample_weight=sample_weight), dtype=float)
        scores = {k: float(metric_fn(v, pred)) for k, v in truths.items()}
        first = next(iter(truths.values()))
        return scores, tail_scores(first, pred), None
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'


def _levels_for(data, dataset_name, condition, levels=None):
    """(levels, where they came from). A kind the noise grid keeps runs at the
    decided levels (noise_helpers.decided_levels, 2026-10-05); every other
    relative condition at n38's own grid (relative_levels). A caller's list
    (a smoke) is taken as given."""
    if levels is not None:
        return tuple(float(l) for l in levels), 'given by the caller'
    if condition in decided_conditions(data):
        return (tuple(decided_levels(data, dataset_name, condition)),
                'decided noise grid (KNN_SEARCH_SPACE_RECORD, Adelaide '
                '2026-10-05), by dataset group')
    return (tuple(relative_levels(condition)),
            "n38's own grid (noise_helpers.relative_levels); this condition is "
            'not one of the six the noise grid keeps')


def _passes(levels, condition, draws=None):
    """(level, draw) pairs. Level 0 is the labels as delivered, so one draw.
    The draws per condition come from noise_helpers.draws_for."""
    draws = draws_for(condition, NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in levels:
        for d in (draws if float(lvl) > 0 else draws[:1]):
            out.append((float(lvl), int(d)))
    return out


# ── Entry points ─────────────────────────────────────────────────────────────

def run(data, config, output_dir, dataset_name, **kwargs):
    """N38 across every relative condition, one file each."""
    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    named = kwargs.get('noise_condition')
    if named and named in CUT_CONDITIONS:
        print(f"  SKIPPED: {named} was cut (2026-10-06); n38 runs only "
              f"{', '.join(N38_CONDITIONS)}.")
        return {'skipped': 'condition_cut', 'noise_condition': named}
    if named and named not in N38_CONDITIONS:
        print(f"  SKIPPED: {named} is not sized by the model's error; n38 runs "
              f"only {', '.join(N38_CONDITIONS)}.")
        return {'skipped': 'not_a_relative_condition', 'noise_condition': named}
    # sweep_conditions pops force_rerun; PassLedger reads restart_passes.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    # This version goes into the input fingerprint every condition file is
    # stamped with (2026-10-05, chat 5), so a condition file written before
    # the alignment (whole-dataset recipes, five models) no longer matches and
    # is recomputed instead of being kept.
    config = dict(config)
    config['noise_seeds'] = {'draws': list(kwargs.get('draws') or NOISE_DRAWS),
                             'alignment': ALIGNMENT_VERSION}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(N38_CONDITIONS), kwargs=kwargs)


def _resid_sd_rep(data):
    """The representation resid_sd and the semi-synthetic signal are measured
    in (RESID_SD_REP_RULE): the dataset's first pruned representation as
    loaded, as noise_helpers.inject_condition takes it."""
    return next(iter(data['reps']))


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N38 under one relative condition, both label bases."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N38: NOISE SIZED BY THE MODEL'S OWN ERROR")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    groups = data.get('groups')
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}; every condition here "
              f"is sized by a regression residual.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    if condition not in N38_CONDITIONS:
        raise ValueError(f"n38: {condition!r} is not one of "
                         f"{', '.join(N38_CONDITIONS)}")
    heavy_all = None
    if condition == 'hetero_seen':
        if data.get('smiles') is None:
            print(f"  SKIPPED: {dataset_name} loads no SMILES, so there is no "
                  f"heavy-atom count for hetero_seen to follow.")
            return {'skipped': 'no_smiles', 'noise_condition': condition}
        heavy_all = heavy_atom_counts(data['smiles'])
    sparse = detection_is_defined(condition)
    share = (RELATIVE_SPARSE_SHARES[condition.rsplit('_', 1)[1]]
             if sparse else None)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    bases = list(kwargs.get('label_bases') or LABEL_BASES)
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    itlm_rounds = int(kwargs.get('itlm_rounds', ITLM_ROUNDS))
    arm_names = list(kwargs.get('arms') or build_arm_names(seeds))
    w_arm = f'winsorise_c{WINSOR_C:g}'
    i_arm = f'itlm_a{ITLM_ALPHA:g}'

    # ── R1: the dataset's two decided models (DECISIONS.md Step 1),
    # 2026-10-05 ──
    # Replaces n32.downstream_model_list (config['models'] or
    # DOWNSTREAM_MODELS, five models, with rf and svm appended). A caller's
    # `models` can only narrow the decided pair (a smoke); a name outside it
    # is dropped and printed. lightgbm and NGBoost are still fitted inside the
    # method (the unit, the semi-synthetic signal, p, ITLM, the weights); they
    # are not downstream models.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        dropped = [m for m in asked if m not in decided]
        if dropped:
            print(f"  n38: models {dropped} are not this dataset's decided "
                  f"models {decided}; dropped")
        downstream_models = [m for m in decided if m in asked]
    else:
        downstream_models = list(decided)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    no_factory = [m for m in downstream_models if m not in model_factories]
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n38: no downstream model could be built. Decided models "
            f"{decided}, asked {asked}, built {sorted(model_factories)}.")
    arms_not_run = {}
    for _m in downstream_models:
        _undef = {a: why for a, why in ARMS_UNDEFINED_FOR.get(_m, {}).items()
                  if a in arm_names}
        if _undef:
            arms_not_run[_m] = _undef

    # ── Folds: noise_helpers.noise_outer_folds (handoff 2.4 item 14) ──
    # The count is config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides); the
    # splits are the ones the per-fold searches are filed under. Was
    # cv_split(groups, n_folds, ...), which is not those splits.
    folds, n_folds = noise_outer_folds(data, config)
    knn_grp = dataset_knn_group(data, dataset_name)
    levels, level_source = _levels_for(data, dataset_name, condition,
                                       kwargs.get('levels'))

    results = {
        'alignment': ALIGNMENT_VERSION,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'condition_in_noise_grid': condition in decided_conditions(data),
        'knn_group': knn_grp,
        'level_source': level_source,
        'dose_units': ('residual SDs (resid_sd) of shift' if sparse
                       else 'injected variance as a share of OOF MSE (resid_sd^2)'),
        'resid_sd_is': ("SD of lightgbm's out-of-fold residual on this fold's "
                        "clean training labels, in resid_sd_rep, per label "
                        "base; see 'fold_records'"),
        'resid_sd_rep_rule': RESID_SD_REP_RULE,
        'resid_sd_rep': _resid_sd_rep(data),
        'conditions_run': list(N38_CONDITIONS),
        'conditions_cut': list(CUT_CONDITIONS),
        'detection_defined': bool(sparse),
        'share_moved': share,
        'label_bases': bases,
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'models': downstream_models,
        'models_decided': decided,
        'models_without_factory': no_factory,
        'arms_not_run_by_model': arms_not_run,
        'n_folds': int(n_folds),
        'folds_run': [int(i) for i, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'configuration_spaces': list(CONFIGURATION_SPACES),
        'configurations_note': (
            'Per outer fold and decided model: base_1 and base_2 are its 2 '
            'best single representations and hybrid and hybrid__general2 its 2 '
            "best general hybrids, all ranked by searches on that fold's "
            'training rows with clean labels (noise_helpers.'
            'fold_configurations). 2 models x 4 = 8 fitted configurations. '
            'Which ones, per fold, is under folds. Each hybrid is rebuilt from '
            "the pass's corrupted training labels. n38 finds no neighbours, "
            'so no distance or similarity hybrid is used.'),
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'winsor_c': WINSOR_C, 'robust_c': ROBUST_C,
        'robust_swap_by_model': {m: ROBUST_SWAP.get(m) for m in downstream_models},
        'itlm_alpha': ITLM_ALPHA, 'itlm_rounds': itlm_rounds,
        'random_seeds': seeds, 'oof_splits': oof_splits,
        'detect_threshold_in_residual_sds': DETECT_T,
        'semi_synthetic_note': (
            'clean training label = lightgbm OOF prediction + a residual '
            'resampled from the training OOF residuals; held-out truth = the '
            'whole-training-fold lightgbm prediction + a resampled residual. '
            "No held-out label is read. 'vs_signal' scores against the "
            'prediction alone. The signal is a lightgbm on resid_sd_rep (the '
            "dataset's first pruned representation), which favours lightgbm "
            'on that representation. The configurations were '
            "chosen by the fold's search on the REAL training labels, under "
            'both label bases.'),
        'robustness_note': (
            'no_cleaning_minus_clean is the plain difference, the no_cleaning '
            'arm minus the clean arm, per outer fold (mean over draws) in the '
            "dataset's own metric; read its sign with lower_is_better. On "
            'fast_knn datasets it is the robustness number (auc_norm is '
            'absent there by decision, 2026-10-05); it is written on every '
            'dataset.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset.'),
        'similarity_note': ('n38 finds no neighbours, so it has no similarity '
                            'axis; its rows carry model_input|none|process.'),
        'controls_run': controls_enabled(),
        'rows_note': ('One row per (label base, arm, model, configuration, '
                      'pass): the label base is in method_setting, the '
                      'configuration (base_1, hybrid, ...) in model_input and '
                      'the model in its own field; scores are against the '
                      'held-out truth, not the signal.'),
        'per_level': {}, 'per_level_vs_signal': {}, 'tails': {},
        'detection': {}, 'fold_records': {}, 'method_records': {},
    }
    if sparse:
        results['theory_at_nominal_dprime'] = {
            f'{d:g}': dict(theory_threshold(d, share),
                           theory_auroc=theory_auroc_abs(d))
            for d in levels if d > 0}

    passes = _passes(levels, condition, draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(folds)} of {n_folds} folds, models "
          f"{downstream_models} ({knn_grp}), label bases {bases}")
    print(f"  condition {condition}, levels {results['noise_levels']} "
          f"({level_source}), {len(passes)} (level, draw) passes, "
          f"{len(arm_names)} arms")

    acc, acc_sig, tail_acc, det_acc, rec_acc, errors = {}, {}, {}, {}, {}, {}
    fold_recs = {}
    # {fold: {model: configuration_record}}: what each fold fitted (handoff
    # 2.4 item 11).
    fold_records = {}
    t0 = time.time()
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). A pass
    # is one (label base, level, draw, fold). The signature carries the header
    # above (ALIGNMENT_VERSION included) and the fold indices, so a checkpoint
    # from before 2026-10-05 is set aside.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _i, tr, va in folds]),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        (acc, acc_sig, tail_acc, det_acc, rec_acc, errors) = (
            st_['acc'], st_['acc_sig'], st_['tail_acc'], st_['det_acc'],
            st_['rec_acc'], st_['errors'])
        fold_records = st_.get('fold_records') or {}
        fold_recs = st_.get('fold_recs') or {}
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    # Per-fold configurations, once per fold for both label bases and every
    # (level, draw): the searches run on the fold's training rows with the
    # clean labels and are stored on disk by fold_searches, so a resumed run
    # reads them back.
    fold_entries = {}

    def _entries_for(fold_i, train_idx):
        if fold_i not in fold_entries:
            try:
                fold_entries[fold_i] = fold_configurations(
                    data, dataset_name, fold_i=fold_i, train_idx=train_idx,
                    n_folds=n_folds)
            except Exception as exc:
                errors[f'f{fold_i}|fold_configurations'] = (
                    f'{type(exc).__name__}: {exc}')
                print(f"  fold {fold_i}: fold_configurations raised "
                      f"{type(exc).__name__}: {exc}")
                fold_entries[fold_i] = None
        return fold_entries[fold_i]

    for base in bases:
        # ── The clean labels and the unit, once per fold ──
        fold_clean = {}
        for fold_i, train_idx, val_idx in folds:
            # None without groups, as inject_condition passes it, so the
            # out-of-fold split (and resid_sd on real labels) is the same.
            g_unit = (np.asarray(groups)[train_idx] if groups is not None
                      else None)
            entries = _entries_for(fold_i, train_idx)
            if entries is None:
                continue
            try:
                rep = _resid_sd_rep(data)
                # Scaled once per fold through noise_cache with
                # kirby.scaling.FeatureScaler (0/1 features unscaled), in
                # float32 (as_rep_dtype), as inject_condition scales it.
                Xb_tr, Xb_va = _nc.cached_scaled(
                    dataset_name, as_rep_dtype(reps[rep]), train_idx, val_idx)
                y_tr_c, y_va_c, sig_va, rsd, prov = label_base_for_fold(
                    base, Xb_tr, Xb_va, labels[train_idx], g_unit,
                    seed=replicate_seed('semi', fold_i), oof_splits=oof_splits)
            except Exception as exc:
                errors[f'{base}|f{fold_i}|label_base'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            if y_va_c is None:
                y_va_c = labels[val_idx]
            fold_clean[fold_i] = (y_tr_c, y_va_c, sig_va, rsd)
            prov['resid_sd'] = rsd
            prov['resid_sd_rep'] = rep
            fold_recs.setdefault(base, {})[f'f{fold_i}'] = prov

        for level, draw in passes:
            lvl_key = f'{level:g}'
            print(f"\n  --- {base} / {condition} / level {lvl_key} / draw {draw} ---")
            for fold_i, train_idx, val_idx in folds:
                if fold_i not in fold_clean:
                    continue
                entries = fold_entries.get(fold_i)
                y_clean_tr, y_val, sig_va, rsd = fold_clean[fold_i]
                pass_key = f'l{lvl_key}d{draw}f{fold_i}'
                if ledger.pass_done(f'{base}|{pass_key}'):
                    continue
                pass_rows = []
                g_tr = (np.asarray(groups)[train_idx] if groups is not None
                        else np.arange(len(train_idx)))
                # the base's clean column stands in for the labels: the
                # injector reads y_all[train_idx], so hand it a column whose
                # training rows are the base's clean labels.
                y_col = np.array(labels, dtype=float)
                y_col[train_idx] = y_clean_tr
                try:
                    y0, cmask, prov = inject_relative_train_labels(
                        y_col, train_idx=train_idx, condition=condition,
                        level=level, resid_sd=rsd,
                        # One seed rule for every experiment (noise_helpers.
                        # noise_draw_seed), so 'real' and 'semi_synthetic'
                        # get the same noise draw. Whole-column heavy atoms.
                        draw=draw, heavy_atoms=heavy_all)
                except Exception as exc:
                    errors[f'{base}|{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                y0 = np.asarray(y0, dtype=float)
                cmask = np.asarray(cmask, dtype=bool)
                n_tr = int(len(y0))
                truths = {'truth': y_val}
                if sig_va is not None:
                    truths['signal'] = sig_va
                fold_rec = fold_records.setdefault(str(fold_i), {})

                # What every arm reads for ONE training matrix of this pass:
                # p, detection, and the arms' training columns. Two models
                # whose rank-1 single representation is the same share one
                # matrix, so this is memoised on the matrix's content.
                ctx_memo = {}

                def _matrix_context(X_tr_s, ekey):
                    rec = {'pass': pass_key, 'training_molecules': n_tr,
                           'molecules_moved': int(prov.get('n_moved', 0)),
                           'resid_sd': rsd,
                           'injection': {k: v for k, v in prov.items()
                                         if k not in ('condition',)}}
                    try:
                        # Cached on the label hash (2026-09-25).
                        p = _nc.cached_oof_predict(dataset_name, X_tr_s, y0,
                                                   g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{base}|{lvl_key}|{ekey}|f{fold_i}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        p = None

                    # ── Detection (sparse only) ──
                    drow = None
                    if p is not None and sparse and cmask.any():
                        r = y0 - p
                        ar = np.abs(r)
                        d = detection_scores(cmask, ar, condition)
                        unmoved_sd = float(np.std(r[~cmask]))
                        dp_space = (prov['shift_label_units'] / unmoved_sd
                                    if unmoved_sd > 0 else float('nan'))
                        drow = {
                            'scores': d,
                            'at_resid_sd': measured_threshold(cmask, ar, rsd),
                            'at_mad_scale': measured_threshold(
                                cmask, ar, robust_scale(y0, p)),
                            'dprime_nominal': level,
                            'dprime_in_this_space': dp_space,
                            'theory_at_dprime_in_this_space': dict(
                                theory_threshold(dp_space, share),
                                theory_auroc=theory_auroc_abs(dp_space)),
                        }

                    # ── The training columns ──
                    cols = {'clean': dict(y=y_clean_tr),
                            'no_cleaning': dict(y=y0)}
                    if p is not None:
                        yw, s_w, n_moved_w = winsorise(y0, p, WINSOR_C)
                        cols[w_arm] = dict(y=yw)
                        delta = yw - y0
                        changed = np.abs(delta) > 0
                        rec[w_arm] = {'labels_moved': int(changed.sum()),
                                      'moved_labels_that_were_injected':
                                          int(np.sum(changed & cmask)),
                                      'robust_scale': s_w}
                        for s in seeds:
                            cols[f'{w_arm}_random/seed{s}'] = dict(
                                y=random_twin_of_changes(
                                    y0, changed, delta, replicate_seed(
                                        'w', s, base, lvl_key, draw, fold_i)))
                        cols['robust_loss'] = dict(
                            y=y0, swap_scale=robust_scale(y0, p))
                    try:
                        # Round 1 read off p (n32.itlm first_loss,
                        # 2026-09-25): the same loss, not refitted.
                        keep, last_loss, rounds = itlm(
                            X_tr_s, y0, g_tr, ITLM_ALPHA, rounds=itlm_rounds,
                            oof_splits=oof_splits,
                            first_loss=(None if p is None
                                        else (y0 - p) ** 2))
                        cols[i_arm] = dict(y=y0, keep=keep)
                        n_rm = int(n_tr - keep.sum())
                        rec[i_arm] = {
                            'rounds': rounds, 'molecules_removed': n_rm,
                            'moved_molecules_among_the_removed':
                                int(np.sum(cmask & ~keep))}
                        for s in seeds:
                            cols[f'{i_arm}_random/seed{s}'] = dict(
                                y=y0, keep=random_keep(n_tr, n_rm, replicate_seed(
                                    'i', s, base, lvl_key, draw, fold_i, n_rm)))
                    except Exception as exc:
                        errors[f'{base}|{lvl_key}|{ekey}|itlm|f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
                    if 'invvar_ngboost' in arm_names:
                        try:
                            sig = oof_ngboost_sigma(X_tr_s, y0, g_tr, oof_splits)
                            s_r = (robust_scale(y0, p) if p is not None
                                   else float(np.std(y0)))
                            cols['invvar_ngboost'] = dict(
                                y=y0, weight=invvar_weights(sig, s_r))
                        except Exception as exc:
                            errors[f'{base}|{lvl_key}|{ekey}|ngboost|'
                                   f'f{fold_i}'] = f'{type(exc).__name__}: {exc}'
                    return cols, rec, drow

                for mname in downstream_models:
                    entry = (entries or {}).get(mname)
                    # The model's 4 configurations on this fold, hybrids
                    # rebuilt from this pass's corrupted labels with the fold
                    # search's own representation scores (dirty-run bug 1).
                    try:
                        inputs, unavail, names = fold_model_inputs(
                            data, dataset_name, mname, entry,
                            train_idx=train_idx, val_idx=val_idx, y_train=y0,
                            factory=model_factories[mname], metric=metric)
                    except Exception as exc:
                        errors[f'{base}|{lvl_key}|{mname}|d{draw}|f{fold_i}|'
                               f'build'] = f'{type(exc).__name__}: {exc}'
                        continue
                    if mname not in fold_rec:
                        fold_rec[mname] = convert_numpy(
                            configuration_record(entry, names))
                    for _sp, _why in (unavail or {}).items():
                        errors.setdefault(
                            f'{base}|{lvl_key}|{_sp}/{mname}|f{fold_i}|'
                            f'unavailable', str(_why))

                    for space in CONFIGURATION_SPACES:
                        if space not in inputs:
                            continue
                        X_tr_s, X_va_s = inputs[space]
                        _ekey = f'{space}/{mname}'
                        xkey = _nc.array_hash(X_tr_s)
                        if xkey not in ctx_memo:
                            ctx_memo[xkey] = _matrix_context(X_tr_s, _ekey)
                        cols, rec, drow = ctx_memo[xkey]
                        if drow is not None:
                            (det_acc.setdefault(base, {}).setdefault(lvl_key, {})
                             .setdefault(_ekey, []).append(drow))
                        rec_acc.setdefault(base, {}).setdefault(
                            lvl_key, {}).setdefault(_ekey, []).append(rec)

                        cell = (acc.setdefault(base, {}).setdefault(lvl_key, {})
                                .setdefault(space, {}))
                        cell_sig = (acc_sig.setdefault(base, {})
                                    .setdefault(lvl_key, {}).setdefault(space, {}))
                        tcell = (tail_acc.setdefault(base, {})
                                 .setdefault(lvl_key, {}).setdefault(space, {}))
                        pass_scores = {}
                        for arm in arm_names:
                            if arm not in cols:
                                continue
                            if arm in arms_not_run.get(mname, {}):
                                continue
                            c = cols[arm]
                            factory = model_factories[mname]
                            cache_as = mname
                            if arm == 'robust_loss':
                                swap = ROBUST_SWAP.get(mname)
                                if swap is None:
                                    continue
                                factory = functools.partial(
                                    make_swap, swap, c['swap_scale'], (ROBUST_C,))
                                cache_as = None
                            sc_, tl, err = _fit_score(
                                factory, X_tr_s, c['y'], X_va_s, truths,
                                metric_fn, metric_name, keep=c.get('keep'),
                                sample_weight=c.get('weight'),
                                dataset_name=dataset_name,
                                model_name=cache_as)
                            if sc_ is None:
                                errors[f'{base}|{lvl_key}|{_ekey}|{arm}|'
                                       f'f{fold_i}'] = err
                                continue
                            (cell.setdefault(arm, {}).setdefault(mname, {})
                             [pass_key]) = sc_['truth']
                            pass_scores.setdefault(arm, {})[mname] = \
                                sc_['truth']
                            if 'signal' in sc_:
                                (cell_sig.setdefault(arm, {})
                                 .setdefault(mname, {})[pass_key]) = sc_['signal']
                            if tl is not None:
                                (tcell.setdefault(arm, {}).setdefault(mname, {})
                                 [pass_key]) = tl

                        # ── Controls (KIRBY_NOISE_CONTROLS=1): winsorising
                        # (a label move), ITLM (a deletion) and the NGBoost
                        # weights, twinned by controls.py beside n38's own
                        # random twins, which are kept: they draw differently
                        # (random_twin_of_changes does not sort its pick). ──
                        for arm, src in ((w_arm, dict(y=cols.get(w_arm, {})
                                                      .get('y'))),
                                         (i_arm, dict(keep=cols.get(i_arm, {})
                                                      .get('keep'))),
                                         ('invvar_ngboost', dict(
                                             weight=cols.get('invvar_ngboost',
                                                             {}).get('weight')))):
                            if arm not in cols:
                                continue
                            if arm in arms_not_run.get(mname, {}):
                                continue
                            twins = control_twins(
                                arm_result(y0, **src), X_tr_s, y0,
                                (replicate_seed(arm, base), draw, fold_i,
                                 int(round(level * 1000))))
                            for tw, by_m in twin_scores(
                                    dataset_name, twins, model_factories,
                                    [mname], X_va_s, y_val,
                                    metric).items():
                                pass_scores[f'{arm}|{tw}'] = by_m
                        # model_input is the configuration name alone (base_1,
                        # hybrid, ...) and the model is its own field,
                        # 2026-10-05: n38 used to write 'base_1' on rows two
                        # models shared and 'base_1/rf' on the others, so a
                        # join across experiments missed rows (dirty-run bug
                        # 13, two row formats).
                        pass_rows += arm_rows(
                            dataset_name=dataset_name, exp_id=EXP_ID,
                            metric=metric, condition=condition, level=level,
                            draw=draw, fold=fold_i, model_input=space,
                            scores_by_arm=pass_scores,
                            setting_of=lambda a, _b=base: (
                                a, method_setting(label_base=_b),
                                _ss.MODEL_INPUT.name))
                ledger.add_rows(pass_rows)
                ledger.finish_pass(f'{base}|{pass_key}', {
                    'acc': acc, 'acc_sig': acc_sig, 'tail_acc': tail_acc,
                    'det_acc': det_acc, 'rec_acc': rec_acc, 'errors': errors,
                    'fold_records': fold_records, 'fold_recs': fold_recs})
                print(f"    {base} level {lvl_key} draw {draw} fold {fold_i}: "
                      f"{int(prov.get('n_moved', 0))} of {n_tr} training "
                      f"molecules moved, resid_sd {rsd:.3f} "
                      f"[{time.time() - t0:.0f}s]")

    removal = {w_arm: [f'{w_arm}_random/seed{s}' for s in seeds],
               i_arm: [f'{i_arm}_random/seed{s}' for s in seeds]}
    results['folds'] = convert_numpy(fold_records)
    results['per_level'] = convert_numpy(acc)
    results['per_level_vs_signal'] = convert_numpy(
        {b: v for b, v in acc_sig.items() if any(
            any(sp for sp in lv.values()) for lv in v.values())})
    results['tails'] = convert_numpy(tail_acc)
    results['detection'] = convert_numpy(_summarise_detection(det_acc))
    results['fold_records'] = convert_numpy(fold_recs)
    results['method_records'] = convert_numpy(rec_acc)
    results['d_vs_dprime'] = ledger.describe()
    results['robust_loss_not_scored_for'] = [
        m for m in downstream_models if m not in ROBUST_SWAP]
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy({
        b: _summarise(acc.get(b, {}), tail_acc.get(b, {}), arm_names,
                      downstream_models, removal, metric_name, lower_is_better)
        for b in bases})
    if results['per_level_vs_signal']:
        results['summary_vs_signal'] = convert_numpy({
            b: _summarise(acc_sig.get(b, {}), {}, arm_names, downstream_models,
                          removal, metric_name, lower_is_better)
            for b in results['per_level_vs_signal']})
    _print_summary(results, metric_name)
    return results


def _summarise_detection(det_acc):
    """Means over passes of every detection reading, per base, level, space."""
    out = {}
    for base, by_lvl in det_acc.items():
        for lvl, by_sp in by_lvl.items():
            for sp, rows in by_sp.items():
                e = {'scores': _mean_detection([r['scores'] for r in rows
                                                if r['scores']])}
                for k in ('at_resid_sd', 'at_mad_scale'):
                    got = [r[k] for r in rows if r[k]]
                    e[k] = ({f: float(np.nanmean([g[f] for g in got]))
                             for f in ('power', 'false_positive_rate',
                                       'precision', 'n_flagged', 'scale')}
                            if got else None)
                e['dprime_in_this_space'] = float(np.nanmean(
                    [r['dprime_in_this_space'] for r in rows]))
                th = [r['theory_at_dprime_in_this_space'] for r in rows]
                e['theory_at_dprime_in_this_space'] = {
                    f: float(np.nanmean([t[f] for t in th])) for f in th[0]}
                e['n_passes'] = len(rows)
                out.setdefault(base, {}).setdefault(lvl, {})[sp] = e
    return out


def _summarise(acc, tail_acc, arm_names, models, removal, metric_name,
               lower_is_better):
    """Every arm against no_cleaning; each flagging arm against its randoms.

    2026-10-05 (chat 5): each row also carries its spread over outer folds
    (noise_helpers.fold_spread over the per-fold means, 'one fold, no spread'
    on one fold) and no_cleaning_minus_clean, the no_cleaning score minus the
    clean score on the passes both scored, in the dataset's own metric (the
    robustness number on fast_knn datasets, where auc_norm is absent by
    decision)."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against no_cleaning, paired per pass; a '
                        'pass is one (draw, fold)'),
           'per_level': {}}
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
                row = {'no_cleaning_mean': _mean_on(b, list(b)),
                       'no_cleaning_over_folds': fold_spread(
                           _per_fold_means(b)),
                       'per_arm': {}}
                c_m = clean.get(m) or {}
                both = sorted(set(c_m) & set(b))
                diffs = {k: float(b[k]) - float(c_m[k]) for k in both}
                row['no_cleaning_minus_clean'] = (
                    _mean_on(diffs, both) if both else float('nan'))
                row['no_cleaning_minus_clean_over_folds'] = fold_spread(
                    _per_fold_means(diffs))
                for arm in arm_names:
                    if arm == 'no_cleaning':
                        continue
                    g = (by_arm.get(arm) or {}).get(m) or {}
                    if not g:
                        continue
                    e = {'vs_no_cleaning': _paired(b, g, lower_is_better),
                         'over_folds': fold_spread(_per_fold_means(g))}
                    if arm != 'clean':
                        keys = sorted(set(clean.get(m) or {}) & set(b) & set(g))
                        sh, why = (recovery_share(
                            _mean_on(clean[m], keys), _mean_on(b, keys),
                            _mean_on(g, keys), lower_is_better)
                            if keys else (None, 'no pass scored all three'))
                        e['recovery_share'] = sh
                        e['recovery_share_refused_because'] = why
                    if arm in removal:
                        vs = {}
                        for r_arm in removal[arm]:
                            rg = (by_arm.get(r_arm) or {}).get(m) or {}
                            if rg:
                                vs[r_arm] = _paired(rg, g, lower_is_better)
                        e['vs_random_twin'] = vs
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
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against no_cleaning)")
    for base, by_lvl in (results.get('detection') or {}).items():
        for lvl, by_sp in by_lvl.items():
            th = (results.get('theory_at_nominal_dprime') or {}).get(lvl) or {}
            for sp, e in by_sp.items():
                sc = e['scores'] or {}
                a = e['at_resid_sd'] or {}
                print(f"  {base} d'={lvl} {sp}: AUROC "
                      f"{sc.get('detection_auroc', float('nan')):.3f} (theory "
                      f"{th.get('theory_auroc', float('nan')):.3f}); at 2.5 "
                      f"resid_sd power {a.get('power', float('nan')):.2f} "
                      f"precision {a.get('precision', float('nan')):.2f} (theory "
                      f"{th.get('theory_power', float('nan')):.2f}, "
                      f"{th.get('theory_precision', float('nan')):.2f}); d' in "
                      f"this space {e['dprime_in_this_space']:.2f}")
    for base, s in results['summary'].items():
        for lvl, lrow in s['per_level'].items():
            print(f"\n  {base}, level {lvl}:")
            for space, srow in lrow.items():
                for m, row in srow.items():
                    fs = row.get('no_cleaning_minus_clean_over_folds') or {}
                    spread = (f"sd {fs['sd']:.4f} over {fs['n_folds']} folds"
                              if fs.get('sd') is not None
                              else fs.get('note', 'no spread'))
                    print(f"    {space}/{m}: no_cleaning "
                          f"{row['no_cleaning_mean']:.4f}; no_cleaning minus "
                          f"clean {row['no_cleaning_minus_clean']:+.4f} "
                          f"({spread})")
                    for arm, e in row['per_arm'].items():
                        if '_random/' in arm:
                            continue
                        v = e['vs_no_cleaning']
                        tail = ''
                        if 'beat_every_random_seed' in e:
                            tail = (f"; beat every random twin: "
                                    f"{e['beat_every_random_seed']}")
                        print(f"      {arm:22s} won {v['passes_won']} of "
                              f"{v['n_passes_paired']} passes, mean "
                              f"{v['mean_gain']:+.4f} (spread "
                              f"{v['spread_over_pass_gains']:.4f}){tail}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        "N38: noise sized by the model's own error")
    parser.add_argument('--noise-condition', type=str, default=None,
                        help=f'One condition (default: sweep all four). One of: '
                             f'{", ".join(N38_CONDITIONS)}')
    parser.add_argument('--levels', type=str, default=None,
                        help="Comma-separated levels. Default: the decided "
                             "levels for sparse_dprime_up_p10; otherwise share "
                             "of OOF MSE (0.1,0.3,1), with 0")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates')
    parser.add_argument('--label-bases', type=str, default=None,
                        help=f'Comma-separated (default {",".join(LABEL_BASES)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-twin replicates')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds (config n_splits). '
                             'KIRBY_NOISE_N_FOLDS overrides both')
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
    if args.label_bases:
        kw['label_bases'] = [x.strip() for x in args.label_bases.split(',')]
    if args.random_seeds:
        kw['random_seeds'] = [int(x) for x in args.random_seeds.split(',')]

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
