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

THE CONDITIONS are noise_helpers.RELATIVE_CONDITIONS, injected by
inject_relative_train_labels. Their unit, resid_sd, is the SD of lightgbm's
out-of-fold residual (noise_helpers.oof_residual_sd) on THIS fold's clean
training labels, in ONE representation, `resid_sd_rep` (see THE FEATURE
SPACES), scaffold-grouped. It is written per fold and per label base, beside
every result.

  dense_oofmse          every training label + N(0, level * resid_sd^2).
  sparse_dprime_pXX     XX% of training molecules moved by level * resid_sd,
                        sign random per molecule.
  sparse_dprime_up_pXX  the same, all up.
  hetero_seen           every label, SD proportional to heavy-atom count (the
                        representation sees molecular size), mean variance
                        level * resid_sd^2.
  hetero_hidden         every label, SD from a per-molecule lognormal draw
                        nothing in the features carries, same mean variance.

The molecules and signs a draw picks do not depend on the level, so the level
sweep nests: level 4 moves the same molecules as level 1, four times as far.

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
  theory is the same for the random-sign and the one-direction conditions.
  Checked: d' = 2, pi = 0.05 gives power 0.309, false positives 0.0124,
  precision 0.568 -- the report's 31%, 1.2%, 58%.
  The theory is evaluated twice: at the nominal d' (in resid_sd_rep's resid_sd),
  and at d'_space = shift / SD of the unmoved molecules' residuals in that
  space, the d' the space's own model actually faced. The second uses the mask
  and is written as a description, never as a detector.

THE ARMS. Every downstream model is fitted on what each arm leaves, in every
feature space, and scored on held-out molecules against their UNTOUCHED labels.

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
                               pseudohuber, dnn huber, rf_l1. svm has no swap
                               and is not scored in this arm (written as a
                               note, not an error).
  itlm_a<alpha>                n32's ITLM, 4 rounds, out-of-fold loss.
  itlm_a<alpha>_random/seed<k> the same number removed at random.
  invvar_ngboost               n33's weights: 1 / sigma^2 from an out-of-fold
                               NGBoost (Normal), floored, averaging one.

WHERE THE FEATURES COME FROM. n32's block: the recipe per downstream model is
read from the precomputed pool; each fold rebuilds the hybrid from that fold's
CORRUPTED training labels once, and every arm shares it.

THE FEATURE SPACES. base_1 and base_2 are each downstream model's top 2 single
representations, ranked by that model's own score in the stored search
(_shared.noise_base_reps_by_model, owner's decision 2026-09-25);
`base_reps_by_model` names them. Then one hybrid per family of stored search. A
model the pool holds no search for has neither.
resid_sd and the semi-synthetic signal must be one per fold for every space and
model, so they are computed in ONE representation, `resid_sd_rep`: the rank-1
single representation of the first downstream model that has one. That choice
was made on 2026-09-25 when the fixed preference list was deleted; the owner has
not signed it off.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. Nothing here measures real label error.

Usage:
    python n38_noise_relative_to_model_error.py --dataset logd --mode full
    python n38_noise_relative_to_model_error.py --dataset herg_ki --mode full \
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
from scipy import integrate
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream,
    noise_base_reps_by_model, base_rank_groups, NOISE_BASE_REP_COUNT,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
    detection_scores, detection_is_defined, tail_scores, sweep_conditions,
    RELATIVE_CONDITIONS, RELATIVE_SPARSE_SHARES, relative_levels,
    inject_relative_train_labels, oof_residual_sd, heavy_atom_counts,
)
from experiments.noise.n07_remediation import _predict_for_metric
from experiments.noise.n30_censored_likelihood import winsorise
from experiments.noise.n32_iterative_and_dual_selection import (
    itlm, random_keep, replicate_seed, _paired, recovery_share, _mean_on,
    _tail_row, _mean_detection,
)
from experiments.noise.n33_robust_training import (
    make_swap, robust_scale, oof_ngboost_sigma, invvar_weights,
)
from kirby.find_best_hybrid import get_model
from kirby.utils.noisy_labels import oof_predict

EXP_ID = 'n38_noise_relative_to_model_error'
DEFAULT_NOISE_CONDITION = 'sparse_dprime_p05'

LABEL_BASES = ('real', 'semi_synthetic')
NOISE_DRAWS = (0, 1, 2)
RANDOM_SEEDS = (0, 1, 2)
OOF_SPLITS = 5
WINSOR_C = 1.5
ROBUST_C = 1.5
ITLM_ALPHA = 0.9
ITLM_ROUNDS = 4
DETECT_T = 2.5          # the report's flag threshold, in residual SDs

# n33's swap for each downstream model; svm has none.
ROBUST_SWAP = {
    'lightgbm': f'lightgbm_huber_c{ROBUST_C:g}',
    'xgboost': f'xgboost_pseudohuber_c{ROBUST_C:g}',
    'dnn': f'dnn_huber_c{ROBUST_C:g}',
    'rf': 'rf_l1',
}

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
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
               keep=None, sample_weight=None):
    """One fit scored against each truth; ({truth: score}, tail dict, None) or
    (None, None, why). Tails are against the first truth."""
    try:
        m = factory()
        X_f, y_f, w_f = X_tr, np.asarray(y_tr, dtype=float), sample_weight
        if keep is not None:
            X_f, y_f = X_f[keep], y_f[keep]
            w_f = None if w_f is None else np.asarray(w_f)[keep]
        if w_f is None:
            m.fit(X_f, y_f)
        else:
            m.fit(X_f, y_f, sample_weight=w_f)
        pred = np.asarray(_predict_for_metric(m, X_va, metric_name), dtype=float)
        scores = {k: float(metric_fn(v, pred)) for k, v in truths.items()}
        first = next(iter(truths.values()))
        return scores, tail_scores(first, pred), None
    except Exception as exc:
        return None, None, f'{type(exc).__name__}: {exc}'


def _passes(condition, levels=None, draws=None):
    levels = relative_levels(condition) if levels is None else levels
    draws = tuple(NOISE_DRAWS if draws is None else draws)
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
    if named and named not in RELATIVE_CONDITIONS:
        print(f"  SKIPPED: {named} is not sized by the model's error; n38 runs "
              f"only {', '.join(RELATIVE_CONDITIONS)}.")
        return {'skipped': 'not_a_relative_condition', 'noise_condition': named}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=list(RELATIVE_CONDITIONS), kwargs=kwargs)


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
    if condition not in RELATIVE_CONDITIONS:
        raise ValueError(f"n38: {condition!r} is not one of "
                         f"{', '.join(RELATIVE_CONDITIONS)}")
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
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    bases = list(kwargs.get('label_bases') or LABEL_BASES)
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    itlm_rounds = int(kwargs.get('itlm_rounds', ITLM_ROUNDS))
    arm_names = list(kwargs.get('arms') or build_arm_names(seeds))
    w_arm = f'winsorise_c{WINSOR_C:g}'
    i_arm = f'itlm_a{ITLM_ALPHA:g}'

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n38: no downstream model could be built.")

    # {model: [rank-1 rep, rank-2 rep]} given directly (a smoke or a direct
    # caller). None means: rank them from the stored search, below.
    base_reps_override = kwargs.get('base_reps')

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': ('residual SDs (resid_sd) of shift' if sparse
                       else 'injected variance as a share of OOF MSE (resid_sd^2)'),
        'resid_sd_is': ("SD of lightgbm's out-of-fold residual on this fold's "
                        "clean training labels, in resid_sd_rep, per label "
                        "base; see 'fold_records'"),
        'detection_defined': bool(sparse),
        'share_moved': share,
        'label_bases': bases,
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'models': downstream_models,
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
            'prediction alone. The signal is a lightgbm on resid_sd_rep, which '
            'favours lightgbm on that representation.'),
        'claim_scope': (
            'The noise is injected, so every number is confined to this '
            'condition on this dataset.'),
        'per_level': {}, 'per_level_vs_signal': {}, 'tails': {},
        'detection': {}, 'fold_records': {}, 'method_records': {},
    }
    if sparse:
        results['theory_at_nominal_dprime'] = {
            f'{d:g}': dict(theory_threshold(d, share),
                           theory_auroc=theory_auroc_abs(d))
            for d in relative_levels(condition) if d > 0}

    splits = list(cv_split(groups, n_folds, next(iter(reps.values())), labels,
                           mode=mode))

    # ── The recipe per downstream model, n32's block ──
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

    # ── The single-representation comparison, per downstream model ──
    if base_reps_override is not None:
        base_by_model = {m: list(base_reps_override[m])
                         for m in downstream_models if base_reps_override.get(m)}
        base_missing = {m: 'not given by the caller' for m in downstream_models
                        if m not in base_by_model}
    else:
        base_by_model, base_missing = noise_base_reps_by_model(
            {m: d['search'] for m, d in by_model.items()}, downstream_models,
            reps=reps, dataset_name=dataset_name)
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces = tuple(
        f'base_{_k}' for _k in range(1, NOISE_BASE_REP_COUNT + 1)
        if base_rank_groups(base_by_model, _k, downstream_models))
    # The one representation resid_sd and the semi-synthetic signal are
    # computed in (module docstring, THE FEATURE SPACES).
    _ref_model = next((m for m in downstream_models if base_by_model.get(m)),
                      None)
    resid_sd_rep = base_by_model[_ref_model][0] if _ref_model else None
    results['resid_sd_rep'] = resid_sd_rep
    results['resid_sd_rep_chosen_as'] = (
        f'rank-1 single representation of {_ref_model}, the first downstream '
        f'model that has one' if _ref_model else
        'none: no downstream model has a single representation, so no '
        'residual SD can be measured and nothing is injected')
    spaces = base_spaces + tuple(_ordered)
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
             f"pool was never read. Every arm ran in the single "
             f"representations only.")
            if results['hybrid_lookup_error'] else
            (f'no recipe came back from pool subset {POOL_SUBSET!r} for any '
             f'downstream model. Every arm ran in the single representations '
             f'only.'))
        print("  no hybrid recipe; single representations only")
    results['spaces_run'] = list(spaces)

    passes = _passes(condition, levels=kwargs.get('levels'),
                     draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(splits)} folds, models "
          f"{downstream_models}, label bases {bases}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes, {len(arm_names)} arms")

    acc, acc_sig, tail_acc, det_acc, rec_acc, errors = {}, {}, {}, {}, {}, {}
    fold_recs = {}
    t0 = time.time()

    for base in bases:
        # ── The clean labels and the unit, once per fold ──
        fold_clean = {}
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            g_tr = (np.asarray(groups)[train_idx] if groups is not None
                    else np.arange(len(train_idx)))
            try:
                if resid_sd_rep is None:
                    raise RuntimeError(results['resid_sd_rep_chosen_as'])
                sc = StandardScaler()
                Xb_tr = sc.fit_transform(np.asarray(reps[resid_sd_rep][train_idx],
                                                    dtype=float))
                Xb_va = sc.transform(np.asarray(reps[resid_sd_rep][val_idx],
                                                dtype=float))
                y_tr_c, y_va_c, sig_va, rsd, prov = label_base_for_fold(
                    base, Xb_tr, Xb_va, labels[train_idx], g_tr,
                    seed=replicate_seed('semi', fold_i), oof_splits=oof_splits)
            except Exception as exc:
                errors[f'{base}|f{fold_i}|label_base'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            if y_va_c is None:
                y_va_c = labels[val_idx]
            fold_clean[fold_i] = (y_tr_c, y_va_c, sig_va, rsd)
            prov['resid_sd'] = rsd
            fold_recs.setdefault(base, {})[f'f{fold_i}'] = prov

        for level, draw in passes:
            lvl_key = f'{level:g}'
            print(f"\n  --- {base} / {condition} / level {lvl_key} / draw {draw} ---")
            for fold_i, (train_idx, val_idx) in enumerate(splits):
                if fold_i not in fold_clean:
                    continue
                y_clean_tr, y_val, sig_va, rsd = fold_clean[fold_i]
                pass_key = f'l{lvl_key}d{draw}f{fold_i}'
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
                        seed=replicate_seed(condition, base, draw, fold_i),
                        heavy_atoms=(None if heavy_all is None
                                     else heavy_all[train_idx]))
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

                train_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                              for k, v in reps.items()}
                val_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                            for k, v in reps.items()}
                # Each single-representation space holds one matrix per
                # distinct representation at that rank, shared by the models
                # that rank it there.
                fold_spaces = {}
                for _sp in base_spaces:
                    _bcells = []
                    for _rn, _ms in base_rank_groups(
                            base_by_model, int(_sp.split('_')[1]),
                            downstream_models).items():
                        sc = StandardScaler()
                        _bcells.append((sc.fit_transform(train_reps[_rn]),
                                        sc.transform(val_reps[_rn]), list(_ms)))
                    fold_spaces[_sp] = _bcells
                for _sp in spaces:
                    if _sp in base_spaces:
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
                            errors[f'{base}|{lvl_key}|{_sp}/{_mnames[0]}|'
                                   f'f{fold_i}|build'] = 'the hybrid could not be built'
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
                        rec = {'pass': pass_key, 'training_molecules': n_tr,
                               'molecules_moved': int(prov.get('n_moved', 0)),
                               'resid_sd': rsd,
                               'injection': {k: v for k, v in prov.items()
                                             if k not in ('condition',)}}
                        try:
                            p = oof_predict(X_tr_s, y0, g_tr, n_splits=oof_splits)
                        except Exception as exc:
                            errors[f'{base}|{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                                f'{type(exc).__name__}: {exc}')
                            p = None

                        # ── Detection (sparse only) ──
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
                            (det_acc.setdefault(base, {}).setdefault(lvl_key, {})
                             .setdefault(_ekey, []).append(drow))

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
                            keep, last_loss, rounds = itlm(
                                X_tr_s, y0, g_tr, ITLM_ALPHA, rounds=itlm_rounds,
                                oof_splits=oof_splits)
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
                            errors[f'{base}|{lvl_key}|{_ekey}|itlm|f{fold_i}'] = (
                                f'{type(exc).__name__}: {exc}')
                        if 'invvar_ngboost' in arm_names:
                            try:
                                sig = oof_ngboost_sigma(X_tr_s, y0, g_tr, oof_splits)
                                s_r = (robust_scale(y0, p) if p is not None
                                       else float(np.std(y0)))
                                cols['invvar_ngboost'] = dict(
                                    y=y0, weight=invvar_weights(sig, s_r))
                            except Exception as exc:
                                errors[f'{base}|{lvl_key}|{_ekey}|ngboost|'
                                       f'f{fold_i}'] = f'{type(exc).__name__}: {exc}'

                        rec_acc.setdefault(base, {}).setdefault(
                            lvl_key, {}).setdefault(_ekey, []).append(rec)

                        cell = (acc.setdefault(base, {}).setdefault(lvl_key, {})
                                .setdefault(space, {}))
                        cell_sig = (acc_sig.setdefault(base, {})
                                    .setdefault(lvl_key, {}).setdefault(space, {}))
                        tcell = (tail_acc.setdefault(base, {})
                                 .setdefault(lvl_key, {}).setdefault(space, {}))
                        for arm in arm_names:
                            if arm not in cols:
                                continue
                            c = cols[arm]
                            for mname in cell_models:
                                factory = model_factories[mname]
                                if arm == 'robust_loss':
                                    swap = ROBUST_SWAP.get(mname)
                                    if swap is None:
                                        continue
                                    factory = functools.partial(
                                        make_swap, swap, c['swap_scale'], (ROBUST_C,))
                                sc_, tl, err = _fit_score(
                                    factory, X_tr_s, c['y'], X_va_s, truths,
                                    metric_fn, metric_name, keep=c.get('keep'),
                                    sample_weight=c.get('weight'))
                                if sc_ is None:
                                    errors[f'{base}|{lvl_key}|{_ekey}|{arm}|'
                                           f'{mname}|f{fold_i}'] = err
                                    continue
                                (cell.setdefault(arm, {}).setdefault(mname, {})
                                 [pass_key]) = sc_['truth']
                                if 'signal' in sc_:
                                    (cell_sig.setdefault(arm, {})
                                     .setdefault(mname, {})[pass_key]) = sc_['signal']
                                if tl is not None:
                                    (tcell.setdefault(arm, {}).setdefault(mname, {})
                                     [pass_key]) = tl
                print(f"    {base} level {lvl_key} draw {draw} fold {fold_i}: "
                      f"{int(prov.get('n_moved', 0))} of {n_tr} training "
                      f"molecules moved, resid_sd {rsd:.3f} "
                      f"[{time.time() - t0:.0f}s]")

    removal = {w_arm: [f'{w_arm}_random/seed{s}' for s in seeds],
               i_arm: [f'{i_arm}_random/seed{s}' for s in seeds]}
    results['per_level'] = convert_numpy(acc)
    results['per_level_vs_signal'] = convert_numpy(
        {b: v for b, v in acc_sig.items() if any(
            any(sp for sp in lv.values()) for lv in v.values())})
    results['tails'] = convert_numpy(tail_acc)
    results['detection'] = convert_numpy(_summarise_detection(det_acc))
    results['fold_records'] = convert_numpy(fold_recs)
    results['method_records'] = convert_numpy(rec_acc)
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
    """Every arm against no_cleaning; each flagging arm against its randoms."""
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
                    print(f"    {space}/{m}: no_cleaning "
                          f"{row['no_cleaning_mean']:.4f}")
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
                        help=f'One condition (default: sweep all). One of: '
                             f'{", ".join(RELATIVE_CONDITIONS)}')
    parser.add_argument('--levels', type=str, default=None,
                        help="Comma-separated levels. Default: share of OOF MSE "
                             "(0.1,0.3,1) or d' (1,2,4), with 0")
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates')
    parser.add_argument('--label-bases', type=str, default=None,
                        help=f'Comma-separated (default {",".join(LABEL_BASES)})')
    parser.add_argument('--random-seeds', type=str, default=None,
                        help='Comma-separated random-twin replicates')
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
