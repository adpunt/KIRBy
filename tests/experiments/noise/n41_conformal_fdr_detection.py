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

THE CALIBRATION SETS.
  random_s<pct>  a random share of the training fold, one set per share in
            CALIBRATION_SHARES (0.2 and 0.5). What any dataset has. It holds
            wrong labels in the same share as the rest of the fold, and a wrong
            label is flagged only if it outscores nearly every calibration
            molecule, the wrong ones included -- so on the smoke's data it
            flagged nothing at either share.
  oracle_clean   ORACLE, no real user has it. ORACLE_CALIBRATION_SHARE of the
            training fold drawn only from molecules the injection did not move.
            Owner's decision 2026-09-25: the report's method assumes a clean
            calibration set, and without this one it is tested fairly only on
            the two datasets with repeat measurements. It measures how well the
            flagging could work if a clean set existed: a ceiling, not an arm to
            recommend.
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
  lof                        local outlier factor on [a neighbour space's
                             features, r], under every decided neighbour
                             setting (THE NEIGHBOUR SPACES below).
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

THE FEATURE SPACES (2026-10-05, chat 5, HANDOFF_EXPERIMENT_ALIGNMENT 2.4). The
models are the dataset's two decided models (DECISIONS.md Step 1,
decided_grid.decided_models). For each outer fold
(noise_helpers.noise_outer_folds), searches on that fold's training rows with
the clean training labels (noise_helpers.fold_configurations) give each model
its 2 best base representations (`base_1`, `base_2`) and its 2 best general
hybrids (`hybrid`, `hybrid__general2`): 2 models x 4 = 8 fitted
configurations. Each pass rebuilds the hybrids on the fold's training rows from
that pass's CORRUPTED training labels (noise_helpers.fold_model_inputs); every
detector, calibration set and arm shares them. Until 2026-10-05 the base
representations and up to six hybrids per model were read once from the
whole-dataset stored searches, for five models.
A relative condition (sparse_dprime_up_p10) sizes its shift by lightgbm's
out-of-fold residual SD on the fold's clean training labels, measured in the
first of the dataset's pruned representations as loaded
(noise_helpers.inject_condition), one choice for every model and
configuration. Until 2026-10-05 it was the first model's rank-1
representation from the whole-dataset ranking.

THE NEIGHBOUR SPACES. lof is the one detector here that finds neighbours. It is
scored under every decided neighbour setting (KNN_SEARCH_SPACE_RECORD: the
dataset group's spaces and distances, 5, 10 and 20 neighbours, scaffold-mates
not excluded). Its spaces include the top hybrid of the per-fold distance
search (`hybrid__distance1`) and the top hybrid of the per-fold similarity
search (`hybrid_similarity`); those two are neighbour spaces only and no model
is fitted on them. No model is refitted per setting, so every setting is scored
and no screen runs. How much each neighbour counts (rank, inverse_square) does
not reach LOF: LOF compares a molecule's neighbour distances with its
neighbours' own, and no neighbour label is averaged.

THE NOISE GRID (KNN_SEARCH_SPACE_RECORD, "The noise grid", Adelaide
2026-10-05). The decided kinds where detection is defined (some molecules
move, some do not): outlier_p05, censoring, shift_only_p10 and
sparse_dprime_up_p10. Levels by dataset group (noise_helpers.decided_levels),
3 draws per non-zero level, level 0 once and run last in each fold. Until
2026-10-05 n41 swept every condition in DETECTION_IS_DEFINED, sixteen of them
cut since.

WHAT THIS CANNOT SAY. The noise is injected, so every number is confined to the
named condition. The false-flag share is measured against the injector's mask.

Usage:
    python n41_conformal_fdr_detection.py --dataset logd --mode full
    python n41_conformal_fdr_detection.py --dataset herg_ki --mode full \
        --noise-condition sparse_dprime_up_p10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time

import numpy as np

from experiments._shared import (
    get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories, REP_DTYPE,
)
from experiments.dataset_policy import noise_groups_for
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import (
    detection_scores, detection_is_defined, sweep_conditions,
    DETECTION_IS_DEFINED, cleanlab_suspicion,
)
# CHAT 5 ALIGNMENT (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT 2.4): folds, the
# per-fold configurations, the decided noise grid and one injection call. They
# replace cv_split, get_best_search_for_downstream,
# pool_recipe_per_downstream_model, pool_recipes_by_family,
# noise_base_reps_by_model, group_models_by_recipe, noise_levels and the three
# injectors this file called itself.
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, inject_condition, decided_conditions,
    decided_levels, dataset_knn_group, level_unit, CONFIGURATION_SPACES,
)
from experiments.noise.n32_iterative_and_dual_selection import (
    _fit_score, _paired, recovery_share, _mean_on, _tail_row,
    _mean_detection, replicate_seed, _per_fold_means,
    # The D-versus-D' rows, per-pass checkpoint, controls and the similarity
    # axis shared by n32-n41 (n32's shared block, 2026-09-25).
    ledger_signature, PassLedger, restart_flag, arm_rows, arm_result,
    control_twins, twin_scores, controls_enabled, sweep_settings,
    is_model_input, fold_similarity, setting_models, similarity_key, _nc,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from kirby.noise_spec import GROUPED_STRATEGIES
from kirby.utils.noisy_labels import oof_predict

EXP_ID = 'n41_conformal_fdr_detection'
# sparse_dprime_p05 until 2026-10-05; that kind was cut by the noise grid
# decision. sparse_dprime_up_p10 is the decided sparse kind.
DEFAULT_NOISE_CONDITION = 'sparse_dprime_up_p10'

# Bumped 2026-10-05 (chat 5): the configurations are chosen per outer fold by
# searches on that fold's training rows, for the dataset's two decided models.
# It is written into the result header, which ledger_signature hashes, and
# into the condition-file fingerprint (run), so a rows file, checkpoint or
# condition file written by the whole-dataset design is set aside and never
# read back as done.
RESULT_LAYOUT = 'fold_aligned_2026-10-05'

# 3 draws per non-zero level, the noise grid rule.
# OPEN (chat 5): under censoring the three draws give the same training labels
# (n32 checked inject_condition('censoring', 0.25): draws bit-identical, the
# clip reads no seed). Only the calibration sets and the random flags differ
# by draw. Left at three, as n32 left it.
NOISE_DRAWS = (0, 1, 2)
BH_QS = (0.05, 0.1, 0.2)
# The random calibration set's share of the training fold, swept. Owner's
# decision 2026-09-25: at 0.2 alone Benjamini-Hochberg can flag nothing or at
# least ~(1 - s) / (s q) molecules, 40 at q = 0.1, more than the 2-10% the
# sparse conditions move on many folds. 0.5 brings that to 10 at q = 0.1 and
# tests half the fold. Each share is its own calibration set, named
# random_s<percent>.
CALIBRATION_SHARES = (0.2, 0.5)
# The oracle set's share of the training fold, capped at the molecules left unmoved.
ORACLE_CALIBRATION_SHARE = 0.5
ORACLE_CAL_NAME = 'oracle_clean'
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

LOF_SCHEME_NOTE = (
    'lof reads no neighbour label, so how much each neighbour counts (rank, '
    'inverse_square; KNN_SEARCH_SPACE_RECORD axis 5) does not apply to it')


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


def rescaled_residual(X, y, groups, p, oof_splits, floor_of_median=SIGMA_FLOOR_OF_MEDIAN,
                      dataset_name=None):
    """(score |r|/sigma, sigma): sigma an out-of-fold LightGBM fit to |r|.

    dataset_name, added 2026-09-25, caches the sigma fit through noise_cache
    (keyed on |r|'s content); the same values."""
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    a = np.abs(r)
    sigma = (oof_predict(X, a, groups, n_splits=oof_splits)
             if dataset_name is None else
             _nc.cached_oof_predict(dataset_name, X, a, groups,
                                    n_splits=oof_splits))
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


def oracle_calibration(moved, share, seed):
    """A random `share` of the rows, drawn only from rows NOT in `moved`.

    ORACLE: it reads the injection's own mask. Capped at the number unmoved.
    """
    moved = np.asarray(moved, dtype=bool)
    n = moved.size
    free = np.flatnonzero(~moved)
    k = min(int(round(share * n)), free.size)
    cal = np.zeros(n, dtype=bool)
    if k > 0:
        rng = np.random.default_rng(int(seed))
        cal[rng.choice(free, k, replace=False)] = True
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
    """[X, r'] with r z-scored and multiplied by sqrt(number of features).

    2026-10-05 (chat 5, handoff 2.3 item 18): X keeps its own float dtype
    (float32 on load) and r' is cast to it. Until then X was cast to float64."""
    X = np.asarray(X)
    if X.dtype.kind != 'f':
        X = X.astype(REP_DTYPE)
    r = np.asarray(r, dtype=float)
    sd = float(np.std(r))
    rz = (r - float(np.mean(r))) / (sd if sd > 0 else 1.0)
    return np.hstack([X, (rz * np.sqrt(X.shape[1])).astype(X.dtype)[:, None]])


def isolation_forest_suspicion(X, r, seed=0):
    from sklearn.ensemble import IsolationForest
    Z = joined_matrix(X, r)
    m = IsolationForest(n_estimators=IFOREST_TREES, random_state=int(seed))
    m.fit(Z)
    return -m.score_samples(Z)


def lof_metric_kwargs(metric):
    """sklearn LocalOutlierFactor arguments for a similarity setting's distance.

    2026-10-05 (chat 5): the decided distances are manhattan, cosine and
    minkowski p=0.5 (KNN_SEARCH_SPACE_RECORD axis 2). 'minkowski_p0.5' is the
    record's name, not a sklearn metric, so it is passed as metric='minkowski',
    p=0.5. sklearn computes p below 1 only by brute force, so brute is asked
    for here; it still warns that p below 1 is not a true distance, and
    lof_suspicion silences that one warning. Until 2026-10-05 the setting's
    name was handed to sklearn as it stood, and every minkowski_p0.5 setting
    raised. None is sklearn's default (Euclidean), kept for an explicit
    KIRBY_SIMILARITY_SETTINGS=model_input."""
    if metric is None:
        return {}
    if metric == 'minkowski_p0.5':
        return {'metric': 'minkowski', 'p': 0.5, 'algorithm': 'brute'}
    if metric in ('manhattan', 'cosine', 'euclidean'):
        return {'metric': metric}
    raise ValueError(f'n41: no LOF distance for setting metric {metric!r}')


def lof_suspicion(X, r, n_neighbors=LOF_NEIGHBOURS, metric=None):
    """Local outlier factor on [X, r']. metric None is sklearn's default
    (Minkowski p=2, Euclidean). metric, added 2026-09-25, is a similarity
    setting's distance when X is that setting's feature matrix
    (n32.FoldSimilarity.space_matrix), mapped by lof_metric_kwargs."""
    from sklearn.neighbors import LocalOutlierFactor
    Z = joined_matrix(X, r)
    k = int(min(n_neighbors, len(Z) - 1))
    import warnings
    m = LocalOutlierFactor(n_neighbors=k, **lof_metric_kwargs(metric))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mind that for 0 < p < 1')
        m.fit(Z)
    return -m.negative_outlier_factor_


def residual_low_disagreement(X, y, groups, oof_splits, seeds=DISAGREEMENT_SEEDS,
                              dataset_name=None):
    """|mean seed prediction - y| / (SD across seeds + floor), all out of fold.

    dataset_name, added 2026-09-25, caches each seed's fit through noise_cache;
    the same values."""
    y = np.asarray(y, dtype=float)
    if dataset_name is None:
        P = np.vstack([oof_predict(X, y, groups, n_splits=oof_splits,
                                   params={'random_state': int(s)})
                       for s in seeds])
    else:
        P = np.vstack([_nc.cached_oof_predict(
            dataset_name, X, y, groups, n_splits=oof_splits,
            params={'random_state': int(s)}) for s in seeds])
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    floor = max(0.1 * float(np.median(sd)), 1e-9)
    return np.abs(y - mu) / (sd + floor)


def detector_scores(name, X, y, groups, p, score, oof_splits, dataset_name=None):
    if name == 'rescaled_residual':
        return score
    r = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    if name == 'isolation_forest':
        return isolation_forest_suspicion(X, r)
    if name == 'lof':
        return lof_suspicion(X, r)
    if name == 'residual_low_disagreement':
        return residual_low_disagreement(X, y, groups, oof_splits,
                                         dataset_name=dataset_name)
    if name == 'cleanlab':
        return cleanlab_suspicion(X, y)
    raise ValueError(f'n41: unknown detector {name!r}')


# ── Actions ─────────────────────────────────────────────────────────────────

def _lof_sweep(fsims, settings, ks, X_tr_s, X_va_s, y0, p, cmask, condition,
               cell_models, det_acc, errors, lvl_key, ekey, fold_i,
               unavailable):
    """LOF under every (setting, k), graded against the mask.
    Added 2026-09-25 (plan section 1).

    LOF runs on [features, r'], so a setting needs coordinates: each feature
    space's standardised matrix (FoldSimilarity.space_matrix) takes the place
    of the cell's matrix and LOF reads the setting's distance
    (lof_metric_kwargs). r is this configuration's out-of-fold residual. A
    model_input setting (only by an explicit KIRBY_SIMILARITY_SETTINGS) keeps
    the cell's matrix and sklearn's default distance. Tanimoto and forest
    proximity are distances with no coordinates to join r to, so they are
    recorded as unavailable.
    TODO(similarity-sweep 2026-09-25): whether LOF under Tanimoto or forest
    proximity should use metric='precomputed' on the setting's distance with r
    left out (a different detector), or be dropped, is the owner's call.

    2026-10-05 (chat 5): fsims is {model: FoldSimilarity}, one per decided
    model, built from that model's per-fold recipes; a per-model space is read
    from the model's own. The neighbours are found among every other training
    molecule: no scaffold group is left out (axis 6). LOF reads no neighbour
    label, so the counting schemes do not apply (LOF_SCHEME_NOTE). No model is
    refitted per setting, so every setting is scored.
    """
    r = np.asarray(y0, dtype=float) - np.asarray(p, dtype=float)
    per_model = len(cell_models) > 1
    for setting in settings:
        for model in setting_models(setting, cell_models):
            fsim = fsims.get(model)
            if is_model_input(setting):
                Z, metric = X_tr_s, None
            elif fsim is None:
                unavailable[f'{setting.name}|{model}'] = (
                    f'no per-fold neighbour spaces for {model}')
                continue
            else:
                Z, _zv, why = fsim.space_matrix(setting, model, X_tr_s)
                if Z is None:
                    unavailable[f'{setting.name}|{model}'] = why
                    continue
                metric = setting.metric
            for k in ks:
                sfx = similarity_key(setting, k, model, today_k=LOF_NEIGHBOURS,
                                     per_model=per_model)
                if not sfx:
                    continue       # model_input at k=20: detector_scores' lof
                try:
                    sus = lof_suspicion(Z, r, n_neighbors=k, metric=metric)
                except Exception as exc:
                    errors[f'{lvl_key}|{ekey}|lof{sfx}|f{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                d = detection_scores(cmask, sus, condition)
                if d is not None:
                    d = dict(d, similarity_setting=setting.name, k=int(k),
                             similarity_model=model)
                    (det_acc.setdefault(lvl_key, {}).setdefault(ekey, {})
                     .setdefault(f'lof{sfx}', []).append(d))


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


def _levels_and_draws(data, dataset_name, condition, levels=None, draws=None):
    """(level, draw) pairs for one fold, level 0 LAST.

    Levels: noise_helpers.decided_levels (the noise grid decision, Adelaide
    2026-10-05) for this dataset's group, each kind in its own unit
    (level_unit). A `levels` kwarg (a smoke, or --levels) overrides. 3 draws
    per non-zero level; level 0 is the labels as delivered and runs once.
    Level 0 runs last in each fold since 2026-10-05 (chat 5), as in n32: every
    pass fits the `clean` arm itself, so no pass needs level 0 first, and a
    time-limited job reaches the damaged levels. Until 2026-10-05 the levels
    came from dataset_policy.noise_levels, shift_only from SHIFT_ONLY_DELTAS_SD
    and the sparse_dprime kinds from relative_levels, level 0 first.
    """
    if levels is None:
        levels = decided_levels(data, dataset_name, condition)
    draws = tuple(NOISE_DRAWS if draws is None else draws)
    out = []
    for lvl in sorted({float(l) for l in levels if float(l) > 0}):
        for d in draws:
            out.append((lvl, int(d)))
    if any(float(l) == 0.0 for l in levels):
        out.append((0.0, int(draws[0])))
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N41 across every decided condition where detection is defined, one
    file each.

    Since 2026-10-05 (chat 5) the default is the decided noise grid's kinds
    that move a subset of the training labels (noise_helpers
    .decided_conditions, filtered by detection_is_defined): outlier_p05,
    censoring, shift_only_p10, sparse_dprime_up_p10. A named condition that
    the grid cut, or one where every label moves, gets a skip carrying the
    reason. Until then the default was every condition in
    DETECTION_IS_DEFINED, sixteen of them cut since.
    """
    if data['task_type'] != 'regression':
        return {'skipped': 'not_regression', 'task_type': data['task_type']}
    decided = list(decided_conditions(data))
    default = [c for c in decided if detection_is_defined(c)]
    named = kwargs.get('noise_condition')
    if named and not detection_is_defined(named):
        print(f"  SKIPPED: {named} moves every training label, so there is no "
              f"damaged subset to find.")
        return {'skipped': 'detection_undefined', 'noise_condition': named}
    if named and named not in decided:
        print(f"  SKIPPED: {named} is not one of the decided kinds of damage "
              f"(KNN_SEARCH_SPACE_RECORD, noise grid, 2026-10-05): "
              f"{', '.join(default)}.")
        return {'skipped': 'condition_cut_by_noise_grid',
                'noise_condition': named, 'decided_conditions': default}
    if kwargs.get('noise_conditions'):
        asked = [str(c) for c in kwargs['noise_conditions']]
        cut = [c for c in asked if c not in default]
        if cut:
            print(f"  dropped {cut}: cut by the noise grid, or every label "
                  f"moves. Decided here: {default}")
        kwargs['noise_conditions'] = [c for c in asked if c in default]
    # sweep_conditions pops force_rerun; PassLedger reads restart_passes.
    kwargs['restart_passes'] = bool(kwargs.get('force_rerun'))
    # The condition files' fingerprint carries RESULT_LAYOUT (through the
    # noise-grid field input_fingerprint reads), so a condition file written
    # by the whole-dataset design is recomputed instead of being kept
    # (2026-10-05, chat 5; n33 and n38 do the same).
    config = dict(config)
    config['noise_seeds'] = {'draws': list(kwargs.get('draws') or NOISE_DRAWS),
                             'alignment': RESULT_LAYOUT}
    return sweep_conditions(
        _run_one_condition, EXP_ID, data, config, output_dir, dataset_name,
        default_conditions=default, kwargs=kwargs,
        groups_needed=GROUPED_STRATEGIES)


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N41: conformal p-values with BH, four other detectors, flag-only actions.

    Per outer fold (noise_helpers.noise_outer_folds):
      1. noise_helpers.fold_configurations runs this fold's searches on its
         training rows (clean labels) and gives each decided model 2 base
         representations, 2 general hybrids and the neighbour-space recipes.
      2. Per (level, draw), noise_helpers.inject_condition damages the
         training labels, and fold_model_inputs builds each model's 4
         configurations, the hybrids rebuilt from the damaged labels.
      3. Configurations that two models share (the same matrix) are one cell,
         so the out-of-fold fits, detectors and flags run once on it.
      4. lof is scored under every neighbour setting of each model.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N41: FLAGS WITH A CONTROLLED FALSE-FLAG RATE")
    print("=" * 70)

    condition = kwargs.get('noise_condition') or DEFAULT_NOISE_CONDITION
    labels = np.asarray(data['labels'], dtype=float)
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
    # The --tag entry point calls this function directly, not run(), so the
    # noise grid's refusal of a cut kind is repeated here (2026-10-05, chat 5).
    decided = list(decided_conditions(data))
    if condition not in decided:
        print(f"  SKIPPED: {condition} is not one of the decided kinds of "
              f"damage (KNN_SEARCH_SPACE_RECORD, noise grid, 2026-10-05): "
              f"{', '.join(c for c in decided if detection_is_defined(c))}.")
        return {'skipped': 'condition_cut_by_noise_grid',
                'noise_condition': condition}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    # Outer folds from config['n_splits'] (KIRBY_NOISE_N_FOLDS overrides), the
    # splits the per-fold searches are filed under. Until 2026-10-05:
    # cv_split with mode 'intermediary' at 1 fold and 'full' otherwise.
    folds, n_folds = noise_outer_folds(data, config)

    qs = [float(q) for q in (kwargs.get('bh_qs') or BH_QS)]
    seeds = [int(s) for s in (kwargs.get('random_seeds') or RANDOM_SEEDS)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))

    rep_groups = build_replicate_groups(data)
    cals = [random_cal_name(s) for s in CALIBRATION_SHARES] + [ORACLE_CAL_NAME] + (
        ['trusted'] if rep_groups is not None else [])
    arm_names = build_arm_names(cals, qs, seeds)

    # The dataset's two decided models (DECISIONS.md Step 1, handoff R1).
    # Until 2026-10-05: the config's five plus rf and svm
    # (downstream_model_list). A `models` kwarg (a smoke) can only narrow them.
    decided = list(decided_models(dataset_name, 'general'))
    asked = kwargs.get('models')
    downstream_models = ([m for m in decided if m in asked] if asked
                         else list(decided))
    if asked and [m for m in asked if m not in decided]:
        print(f"  models {[m for m in asked if m not in decided]} dropped: "
              f"not decided for {dataset_name} ({decided})")
    if kwargs.get('base_reps') is not None:
        # Until 2026-10-05 a caller could hand over {model: [rep, rep]}; the
        # base representations now come from each fold's own search only.
        print("  base_reps given by the caller: ignored, the per-fold search "
              "picks them")
    # OPEN (chat 5): on herg_ic50 (one row per measurement, 12,008 rows)
    # get_model_factories leaves svm out above 10,000 rows (dirty-run bug 11),
    # so herg_ki's decided svm is not fitted there. n41 keeps every measurement
    # because the trusted calibration set reads repeat measurements. herg_ic50
    # is not in the noise submit roster; a direct run would hit this.
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    models_unavailable = {
        m: 'get_model_factories gave no factory (size limit or unknown name)'
        for m in downstream_models if m not in model_factories}
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError("n41: no downstream model could be built.")

    # The neighbour settings (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05):
    # the dataset group's spaces and distances, no weighting, ks 5, 10, 20.
    # Until 2026-10-05 sweep_settings() was called with no dataset, so every
    # dataset got full_knn's list, and the ks were (5, 9, 10, 20).
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    knn_group = dataset_knn_group(data, dataset_name)
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = sweep_settings(dataset_name, n_molecules,
                                  has_smiles=has_smiles)
    from experiments.noise.n07_remediation import sweep_ks as _n07_sweep_ks
    sim_ks = tuple(_n07_sweep_ks())

    noise_groups = noise_groups_for(data)[0]
    if condition in GROUPED_STRATEGIES and noise_groups is None:
        raise RuntimeError(
            f"n41: condition {condition!r} needs scaffold groups and "
            f"{dataset_name} has none.")

    results = {
        'result_layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_condition': condition,
        'dose_units': level_unit(condition),
        'knn_group': knn_group,
        'noise_draws': list(NOISE_DRAWS if kwargs.get('draws') is None
                            else kwargs['draws']),
        'models': downstream_models,
        'models_decided': decided,
        'models_unavailable': models_unavailable,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'per outer fold and decided model: the 2 best base '
            'representations and the 2 best general hybrids from searches on '
            'that fold\'s training rows with the clean training labels '
            '(noise_helpers.fold_configurations); which ones, per fold, is '
            'under folds. Each hybrid\'s training-row score is its score_mean '
            'there; held_out_score is the configuration fitted on the clean '
            'training labels at level 0.'),
        'n_folds': n_folds,
        'folds_run': [int(i) for i, _, _ in folds],
        'arms': arm_names,
        'arm_to_beat': 'no_cleaning',
        'detectors': detectors,
        'bh_qs': qs,
        'calibration_sets': cals,
        'calibration_shares': list(CALIBRATION_SHARES),
        'oracle_calibration_share': ORACLE_CALIBRATION_SHARE,
        'oracle_calibration_note': (
            f'{ORACLE_CAL_NAME} is drawn only from molecules the injection did '
            'not move. No real user has it; it is the ceiling of the flagging.'),
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
        'similarity_settings': [st.name for st in sim_settings],
        'similarity_ks': list(sim_ks),
        'similarity_note': (
            'lof is the one n41 detector that finds neighbours, so it is the '
            'one swept: lof@<setting>|k<k>[|<model>] is LOF on [that '
            'setting\'s features, r\'] under its distance, scaffold-mates kept '
            'in the neighbour list. hybrid__distance1 and hybrid_similarity are '
            'neighbour spaces only; no model is fitted on them. Tanimoto and '
            'forest proximity have no feature matrix and are recorded as '
            'unavailable. The bare lof key (the configuration\'s own matrix, '
            'Euclidean, k=20) is not computed: that space and that distance '
            'are cut.'),
        'lof_neighbour_weighting': LOF_SCHEME_NOTE,
        'neighbour_screen': (
            'not run: no model is refitted per neighbour setting, so every '
            'setting is scored'),
        'detection_at_level_0': (
            'not computed: no molecule moved, so there is nothing to find'),
        'robustness_note': (
            'damaged_minus_clean in each summary row is the no_cleaning score '
            'minus the clean score in the dataset\'s own metric (direction: '
            'lower_is_better). auc_norm is not computed here.'),
        'controls_run': controls_enabled(),
        'per_level': {},
        'tails': {},
        'detection': {},
        'bh': {},
        'fold_records': {},
        'folds': {},
    }

    passes = _levels_and_draws(data, dataset_name, condition,
                               levels=kwargs.get('levels'),
                               draws=kwargs.get('draws'))
    results['noise_levels'] = sorted({l for l, _ in passes})
    print(f"  {len(labels)} molecules, {len(folds)} of {n_folds} folds, "
          f"models {downstream_models}, {knn_group}")
    print(f"  condition {condition}, levels {results['noise_levels']}, "
          f"{len(passes)} (level, draw) passes per fold, {len(arm_names)} "
          f"arms, calibration sets {cals}, detectors {detectors}, "
          f"{len(sim_settings)} neighbour settings x {len(sim_ks)} ks")

    acc, tail_acc, det_acc, bh_acc, rec_acc, errors = {}, {}, {}, {}, {}, {}
    fold_records, prov_acc = {}, {}
    t0 = time.time()
    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). The
    # signature hashes the header above, which carries RESULT_LAYOUT.
    ledger = PassLedger(output_dir, EXP_ID, dataset_name, condition,
                        signature=ledger_signature(
                            data, config, results,
                            [(tr, va) for _, tr, va in folds]),
                        restart=restart_flag(kwargs))
    if ledger.restored():
        st_ = ledger.restored()
        acc, tail_acc, det_acc, bh_acc, rec_acc, errors = (
            st_['acc'], st_['tail_acc'], st_['det_acc'], st_['bh_acc'],
            st_['rec_acc'], st_['errors'])
        fold_records = st_.get('fold_records') or {}
        prov_acc = st_.get('prov_acc') or {}
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'acc': acc, 'tail_acc': tail_acc, 'det_acc': det_acc,
                'bh_acc': bh_acc, 'rec_acc': rec_acc, 'errors': errors,
                'fold_records': fold_records, 'prov_acc': prov_acc}

    for fold_i, train_idx, val_idx in folds:
        fold_keys = [f'l{l:g}d{d}f{fold_i}' for l, d in passes]
        if all(ledger.pass_done(k) for k in fold_keys):
            print(f"\n  fold {fold_i}: every pass on disk in the checkpoint")
            continue
        print(f"\n  === fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows ===")
        # The per-fold searches, on this fold's training rows. Stored per fold
        # (ocv{n}f{i}s{seed}) and shared with every experiment on the dataset.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            errors[f'f{fold_i}|fold_configurations'] = (
                f'{type(exc).__name__}: {exc}')
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue
        frec = fold_records.get(str(fold_i))
        if not frec or 'models' not in frec:
            frec = fold_records[str(fold_i)] = {
                'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in downstream_models:
            mrec = configuration_record(entries.get(m))
            mrec['metric'] = metric_name
            mrec['lower_is_better'] = bool(lower_is_better)
            frec['models'].setdefault(m, mrec)
            print(f"    {m}: base {mrec.get('base')}, hybrids "
                  f"{[h.get('label') for h in mrec.get('hybrids') or []]}")

        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        g_tr = (np.asarray(groups)[train_idx] if groups is not None
                else np.arange(len(train_idx)))

        for level, draw in passes:
            lvl_key = f'{level:g}'
            pass_key = f'l{lvl_key}d{draw}f{fold_i}'
            if ledger.pass_done(pass_key):
                continue
            print(f"\n    --- {condition} / level {lvl_key} / draw {draw} / "
                  f"fold {fold_i} ---")
            pass_rows = []
            fold_rec = {'pass': pass_key}
            try:
                y0, cmask, prov = inject_condition(
                    data, dataset_name, condition, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'{lvl_key}|d{draw}|f{fold_i}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            prov_acc[pass_key] = convert_numpy(prov)
            y0 = np.asarray(y0, dtype=float)
            cmask = np.asarray(cmask, dtype=bool)
            n_tr = int(len(y0))
            moved = int(cmask.sum())
            fold_rec['molecules_moved'] = moved
            fold_rec['training_molecules'] = n_tr

            # ── Calibration sets, shared by every configuration ──
            cal_masks = {random_cal_name(sh): random_calibration(
                n_tr, sh,
                replicate_seed('cal', lvl_key, draw, fold_i, n_tr, sh))
                for sh in CALIBRATION_SHARES}
            cal_masks[ORACLE_CAL_NAME] = oracle_calibration(
                cmask, ORACLE_CALIBRATION_SHARE,
                replicate_seed('oracle_cal', lvl_key, draw, fold_i, n_tr))
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

            # Each model's 4 configurations, the hybrids rebuilt on this
            # fold's training rows from THIS pass's damaged labels with the
            # per-fold recipe (its own representation scores: dirty-run bug
            # 1), 0/1 features unscaled. Until 2026-10-05 the hybrids were
            # rebuilt here with StandardScaler from whole-dataset recipes.
            inputs = {}
            for m in downstream_models:
                ins, unav, names = fold_model_inputs(
                    data, dataset_name, m, entries.get(m),
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    factory=model_factories[m], metric=metric)
                inputs[m] = ins
                mrec = frec['models'][m]
                if names:
                    mrec.setdefault('configuration_names', dict(names))
                mrec.setdefault('n_features', {})
                for sp, (Xa, _Xb) in ins.items():
                    mrec['n_features'].setdefault(sp, int(np.shape(Xa)[1]))
                for sp in CONFIGURATION_SPACES:
                    if sp not in ins:
                        why = (((entries.get(m) or {}).get('unavailable')
                                or {}).get(sp) or unav.get(sp) or 'not built')
                        (mrec.setdefault('configurations_unavailable', {})
                         .setdefault(sp, why))
                        errors[f'{lvl_key}|d{draw}|f{fold_i}|{sp}/{m}|build'] = (
                            why)

            # A configuration two models share (the same matrix) is one cell:
            # the out-of-fold fits, detectors and flags read only the features
            # and the labels, so they run once and both models are fitted on it.
            cells = []
            for space in CONFIGURATION_SPACES:
                by_hash = {}
                for m in downstream_models:
                    if space not in inputs[m]:
                        continue
                    X_tr_s, X_va_s = inputs[m][space]
                    h = (_nc.array_hash(np.asarray(X_tr_s)),
                         _nc.array_hash(np.asarray(X_va_s)))
                    if h in by_hash:
                        by_hash[h][2].append(m)
                    else:
                        by_hash[h] = [X_tr_s, X_va_s, [m]]
                for X_tr_s, X_va_s, ms in by_hash.values():
                    cells.append((space, X_tr_s, X_va_s, ms))

            # The neighbour spaces of this fold, one FoldSimilarity per model,
            # from the damaged labels (plan section 1, 2026-09-25). Not built
            # at level 0, where no molecule moved and lof is not scored.
            fsims = {}
            for m in (downstream_models if moved > 0 else ()):
                entry = entries.get(m)
                if entry is None:
                    continue
                fsim = fold_similarity(
                    dataset_name, data, reps=data['reps'],
                    train_idx=train_idx, val_idx=val_idx, y_train=y0,
                    groups_train=g_tr,
                    base_by_model={m: list(entry.get('base') or [])},
                    recipe_for_space_by_model={
                        m: dict(entry.get('recipes') or {})},
                    model_factories=model_factories, metric=metric,
                    task_type=task_type)
                # The 4 configurations are the same matrices as the base and
                # general-hybrid neighbour spaces: handed over, not rebuilt.
                for sp, (Xa, Xb) in inputs[m].items():
                    fsim._spaces[(m, sp)] = (Xa, Xb, None)
                for sp, why in (entry.get('unavailable') or {}).items():
                    if sp not in inputs[m]:
                        fsim._spaces[(m, sp)] = (None, None, why)
                        fsim.unavailable[f'{m}|{sp}'] = why
                fsims[m] = fsim

            for space, X_tr_s, X_va_s, cell_models in cells:
                _ekey = (space if cell_models == list(downstream_models)
                         else f'{space}/{cell_models[0]}')
                try:
                    # Both out-of-fold fits cached on the label hash
                    # (2026-09-25).
                    p = _nc.cached_oof_predict(dataset_name, X_tr_s, y0,
                                               g_tr, n_splits=oof_splits)
                    score, _sigma = rescaled_residual(
                        X_tr_s, y0, g_tr, p, oof_splits,
                        dataset_name=dataset_name)
                except Exception as exc:
                    errors[f'{lvl_key}|{_ekey}|f{fold_i}|oof'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue

                # ── Detectors, graded against the mask (not at level 0) ──
                for det in (detectors if moved > 0 else ()):
                    if det == 'lof':
                        _lof_sweep(fsims, sim_settings, sim_ks, X_tr_s,
                                   X_va_s, y0, p, cmask, condition,
                                   cell_models, det_acc, errors, lvl_key,
                                   _ekey, fold_i, fold_rec.setdefault(
                                       'similarity_unavailable', {}))
                        if not any(is_model_input(st)
                                   for st in sim_settings):
                            continue
                    try:
                        sus = detector_scores(det, X_tr_s, y0, g_tr, p,
                                              score, oof_splits,
                                              dataset_name=dataset_name)
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
                pass_scores = {}
                for arm in arm_names:
                    if arm not in cols:
                        continue
                    y_use, keep = cols[arm]
                    for mname in cell_models:
                        s_, tl, err = _fit_score(
                            model_factories[mname], X_tr_s, y_use, X_va_s,
                            y_val, metric_fn, metric_name, keep=keep,
                            dataset_name=dataset_name, model_name=mname)
                        if s_ is None:
                            errors[f'{lvl_key}|{_ekey}|{arm}|{mname}|'
                                   f'f{fold_i}'] = err
                            continue
                        (space_cell.setdefault(arm, {})
                         .setdefault(mname, {})[pass_key]) = s_
                        pass_scores.setdefault(arm, {})[mname] = s_
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
                            pass_scores.setdefault(arm, {})[mname] = \
                                b[pass_key]
                if level == 0.0:
                    # The configuration's held-out score: fitted on the clean
                    # training labels, scored on the held-out rows.
                    for mname, s_ in (pass_scores.get('clean') or {}).items():
                        (frec['models'][mname]
                         .setdefault('held_out_score', {})[space]) = s_
                # ── Controls (KIRBY_NOISE_CONTROLS=1): every flag arm
                # (winsorise: a label move; remove: a deletion), twinned
                # by controls.py beside n41's own random-flag twins, which
                # are kept -- they draw only outside the calibration set. ──
                for arm in arm_names:
                    if (arm not in cols or arm in ('clean', 'no_cleaning')
                            or '_random/' in arm):
                        continue
                    y_use, keep = cols[arm]
                    twins = control_twins(
                        arm_result(y0, y=None if keep is not None
                                   else y_use, keep=keep),
                        X_tr_s, y0, (replicate_seed(arm), draw, fold_i,
                                     int(round(level * 1000))))
                    for tw, by_m in twin_scores(
                            dataset_name, twins, model_factories,
                            cell_models, X_va_s, y_val, metric).items():
                        pass_scores[f'{arm}|{tw}'] = by_m
                # ── D against D', one row per (arm, model). model_input is
                # the configuration name ('base_1'), the model its own field:
                # until 2026-10-05 it was 'base_1/<first model of the cell>',
                # even on the other model's rows, so a join with n30-n32's
                # 'base_1' missed rows (dirty-run bug 13). ──
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric, condition=condition, level=level,
                    draw=draw, fold=fold_i, model_input=space,
                    scores_by_arm=pass_scores)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(pass_key, _state())
            print(f"    level {lvl_key} draw {draw} fold {fold_i}: "
                  f"{moved} of {n_tr} training molecules moved, "
                  f"{len(cells)} configuration cells [{time.time() - t0:.0f}s]")

    results['folds'] = convert_numpy(fold_records)
    results['injection_provenance'] = convert_numpy(prov_acc)
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
    results['d_vs_dprime'] = ledger.describe()
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
    """Every arm against no_cleaning; each flag arm against its random twins.

    2026-10-05 (chat 5): each row also carries its spread over folds
    (noise_helpers.fold_spread over the per-fold means, 'one fold, no spread'
    on one fold) and damaged_minus_clean, the no_cleaning score minus the clean
    score on the passes both scored, in the dataset's own metric. auc_norm is
    not computed here (absent on fast_knn datasets by decision)."""
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
                row = {'no_cleaning_mean': _mean_on(b, list(b)),
                       'no_cleaning_over_folds': fold_spread(
                           _per_fold_means(b)),
                       'per_arm': {}}
                c_m = clean.get(m) or {}
                both = sorted(set(c_m) & set(b))
                diffs = {k: float(b[k]) - float(c_m[k]) for k in both}
                row['damaged_minus_clean'] = (
                    _mean_on(diffs, both) if both else float('nan'))
                row['damaged_minus_clean_over_folds'] = fold_spread(
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
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    # 2026-10-05 (chat 5, handoff 2.4 item 9): the representations run_all
    # loads -- the dataset's pruned list (herg_ic50 takes herg_ki's), plus its
    # augmentation list -- not --reps (default 'fast'), so this entry point
    # and run_all search the same representations.
    from experiments.noise.noise_helpers import load_noise_data
    if getattr(args, 'reps', None) not in (None, 'fast'):
        print(f"  --reps {args.reps} is not read: n41 loads the dataset's "
              f"pruned representations")
    data, _ = load_noise_data(args.dataset)

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
