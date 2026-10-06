#!/usr/bin/env python3
"""N39: several sources per molecule, with correlated errors and a bias per source.

Paper: Noise.

THE QUESTION. A third research report (2026-09-25, its A8 and D4) says repeat
measurements give the only label-side correction that is worth anything, and
that it is weaker than it looks. When a molecule is measured by k sources, the
mean's error variance is sigma_e^2 [rho + (1 - rho) / k], not sigma_e^2 / k:
rho is the correlation between two sources' errors on the same molecule, and it
puts a floor of rho * sigma_e^2 under what averaging can remove. A source may
also sit above or below the others by a fixed offset b_s. The report proposes
(a) removing the offsets with a mixed model, and (b) weighting each molecule by
1 / (sigma_delta^2 + sigma_e^2 [rho + (1 - rho) / k]), sigma_delta^2 the
model's own error. This file injects exactly that structure and tests both.

WHAT IS INJECTED. On the clean label column, before any fold is split:

  * N_SOURCES simulated sources.
  * Each molecule gets k sources, drawn without replacement. k is drawn from one
    of COUNT_DISTRIBUTIONS:
      herg_ic50        the real count distribution of herg_ic50's molecules,
                       capped at MAX_SOURCES_PER_MOLECULE. Read live from
                       `load_herg_ic50_measurements`; if that raises, the
                       counts measured on 2026-09-25 (HERG_IC50_COUNT_FALLBACK)
                       are used, and `count_distribution_source` says which.
                       87.6% of molecules there were measured once.
      uniform_1_to_4   1, 2, 3 or 4 with equal chance, mean 2.5: the report's
                       own setting.
  * Source s's measurement of molecule i is
        y_is = y_clean_i + b_s + sigma_e (sqrt(rho) u_i + sqrt(1 - rho) v_is)
    with u_i, v_is, and b_s / tau standard normal. sigma_e = level x the SD of
    the clean label column, the level taken from the suite's decided grid
    (`noise_levels` with the molecule count, 0 dropped: 0.5, 1.0, 1.5 on
    full_knn datasets, 1.0 on fast_knn; 2026-10-05). rho in RHOS, tau = TAU_FRACS x sigma_e.
    The offsets are not centred.
  * The training label of a molecule is the mean of its sources.

The standard normals are drawn from a seed of (dataset, count distribution,
draw) only, so every level, rho and tau at one draw reuse the same molecules,
sources and directions and only their scale changes. The levels nest.

Held-out molecules are graded against their CLEAN label in every arm.

⛔ RHO IS NOT IDENTIFIABLE FROM THE LABELS ALONE. The shared part
sqrt(rho) u_i is the same in every source of molecule i, so it cannot be told
apart from the molecule's true value. What the labels do identify is the
within-molecule variance, (1 - rho) sigma_e^2, and the total a + b/k of the
squared out-of-fold error (a = sigma_delta^2 + rho sigma_e^2, b = (1 - rho)
sigma_e^2; n19's `fit_noise_variance` fits exactly that line). So rho_hat here
needs sigma_e from outside the labels. A practitioner's equivalent is a
published per-measurement assay error; in the simulation it is the injected
sigma_e, and the arm that uses it is marked as such:

    rho_hat = 1 - s_w^2 / sigma_e^2, clipped to [0, 1],

s_w^2 the within-molecule variance after offset removal. And the weight
1 / (sigma_delta^2 + sigma_e^2 [rho + (1-rho)/k]) depends on rho only through
a + b/k, so the fitted n19 weight and the rho-aware weight differ in how a and b
are estimated, not in what they target. The oracle arms show how much a WRONG
rho costs when every variance is known.

THE ARMS. The dataset's two decided models (DECISIONS.md Step 1), each in its
four configurations per outer fold: base_1, base_2, hybrid, hybrid__general2
(2 models x 4 = 8; chat 5, 2026-10-05). Until 2026-10-05 it was all five
downstream models in up to eight spaces each, read from the whole-dataset
stored search. Paired against `plain_mean`.

  clean                          fitted on the clean labels. The reference the
                                 recovery share divides by. Not a method.
  plain_mean                     the mean of each molecule's sources, unweighted.
  offset_removed                 the mean over sources of (y_is - b_hat_s).
  weights_n19_fitted             plain_mean labels, weight 1 / (sigma_u^2 +
                                 sigma_e^2 / k) with both read off n19's
                                 regression of squared out-of-fold error on 1/k.
  weights_within_var             plain_mean labels, weight 1 / (sd2_hat + v_i),
                                 v_i = (s_w^2 + tau_hat^2) / k_i: s_w^2 the
                                 within-molecule variance after offset removal,
                                 tau_hat^2 the variance of the fitted offsets.
                                 sd2_hat is the mean squared out-of-fold error
                                 minus mean(v_i), floored, so it carries the
                                 model error AND the shared part rho sigma_e^2,
                                 which is exactly the sum the weight needs.
                                 Estimated from the labels and the model's
                                 errors only (owner's decision 2026-09-28).
                                 Until then this arm was weights_rho_hat and
                                 read the injected sigma_e.
  offset_removed_within_var_weights
                                 offset_removed labels, the same weight without
                                 the tau term, with sd2_hat from the out-of-fold
                                 error on the offset-removed labels. Was
                                 offset_removed_rho_hat_weights.
  weights_rho0_oracle            plain_mean labels, weight from the TRUE
                                 sigma_e, tau and model error, with rho set to 0.
  weights_true_rho_oracle        the same with the true rho. The ceiling.

Model error for the two oracle arms is the mean squared out-of-fold error of a
LightGBM fitted on the CLEAN training labels in the same space. It uses the
clean labels, which is why those arms are oracles.

HOW THE OFFSETS ARE FITTED. y_is = theta_i + b_s + e_is on the training
molecules' measurements only.
  mixedlm   statsmodels MixedLM, molecule as the random intercept, source as the
            fixed effect, REML. Molecules measured once still inform the source
            means through the random intercept. The residual variance (`scale`)
            is s_w^2. Used up to MIXEDLM_MAX_ROWS measurements.
  als       least squares for the two-way additive model, by alternating
            theta_i = mean_s (y_is - b_s) and b_s = mean_i (y_is - theta_i).
            A molecule measured once is fitted exactly by its own theta_i and
            teaches nothing about b. Used above MIXEDLM_MAX_ROWS, or when MixedLM
            raises, and `offset_method_used` says so per pass.
The fitted offsets are centred: a shift common to every source cannot be told
apart from the molecules' values. b_hat is scored against the centred true b
(correlation and rmse), per pass.

THE REAL-DATA BLOCK. On a dataset whose loader keeps repeat measurements
(herg_ic50, bindingdb_kd; `build_replicate_groups`), per fold and space, on the
training molecules measured twice or more:
  * one pair of measurements per molecule, drawn at random. On herg_ic50 the pair
    is drawn from two DIFFERENT ChEMBL documents where the molecule has them, so
    it is a pair between sources; the counts with and without such a pair are
    written. bindingdb_kd carries no source per measurement, so any two.
  * the correlation of the pair's residuals from a group-aware out-of-fold
    LightGBM prediction of the molecule mean. It equals (sigma_delta^2 + rho
    sigma_e^2) / (sigma_delta^2 + sigma_e^2), so it is an UPPER BOUND on rho.
  * s_w^2, the pooled within-molecule variance (ddof = 1), and n19's a and b.
  * rho at the dataset's published per-measurement error, where one exists
    (`REFERENCE_ERROR_LABEL_UNITS`). Neither herg_ic50 nor bindingdb_kd has one,
    so this is None with a reason on both, never a guess.
The simulation still runs on those datasets, on one clean label per molecule:
the mean of its measurements.

WHERE THE FEATURES COME FROM (rewritten 2026-10-05, chat 5). Per outer fold
(noise_helpers.noise_outer_folds, on one row per molecule), one search on that
fold's training molecules and their clean labels ranks each decided model's 2
base representations and 2 hybrids (_shared.fold_standard_selection; decided
grid, pruning off; a second hybrid with the same selected features as the
first is skipped). The hybrids are rebuilt on the fold's training molecules
through noise_helpers.fold_model_inputs, with the fold search's own
representation scores. What each fold used is under `per_fold_configurations`.
Until 2026-10-05 the recipes and base representations came from the
whole-dataset stored search, which had seen every held-out molecule.
hybrid_labels='clean' (default) builds each hybrid ONCE PER FOLD from the clean
training labels and every arm and pass shares it. That differs from n34, which
rebuilds per pass from the noisy labels: n39 has count distributions x levels x
rho x tau x draws passes (108 per fold on a full_knn dataset, 36 on fast_knn),
and a rebuild per pass would dominate the run. Every arm sees the same features, so the paired arm
comparison is unaffected; what it cannot show is how the sources damage feature
choice (n20's question). hybrid_labels='noisy' rebuilds per pass from the
plain_mean labels instead.

Usage:
    python n39_correlated_sources.py --dataset logd --mode full
    python n39_correlated_sources.py --dataset herg_ic50 --rhos 0,0.7
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import warnings
import zlib

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
    # Per outer fold, the 2 best base representations and 2 best hybrids of
    # each decided model, from a search on that fold's training rows
    # (handoff 2.4 step 5; chat 5, 2026-10-05).
    fold_standard_selection,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_levels, REFERENCE_ERROR_LABEL_UNITS
# Chat 5 (2026-10-05): outer folds, the four fitted configurations of one
# model on one fold, the result-file record of them, and the description of a
# number over folds.
from experiments.noise.noise_helpers import (
    tail_scores, noise_outer_folds, fold_spread, fold_model_inputs,
    configuration_record, CONFIGURATION_SPACES,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n19_measurement_count import (
    measurement_stats, fit_noise_variance, inverse_variance_weights,
    # One row per molecule (herg_ic50) and one model's fold_standard_selection
    # result in the form fold_model_inputs reads (chat 5, 2026-10-05).
    _molecule_rows_data, _entry_from_selection, SEARCH_CONDITION_SUFFIX,
)
from experiments.noise.n34_group_random_effects import (
    recovery_share, _paired, _mean_on,
)
# The D-versus-D' rows, per-pass checkpoint, cached fits and controls shared
# by n32-n41 (n32's shared block, 2026-09-25).
from experiments.noise.n32_iterative_and_dual_selection import (
    fit_predict_cached, ledger_signature, PassLedger, restart_flag, arm_rows,
    arm_result, control_twins, twin_scores, controls_enabled, replicate_seed,
    _nc,
)

EXP_ID = 'n39_correlated_sources'

N_SOURCES = 8
MAX_SOURCES_PER_MOLECULE = 4
RHOS = (0.0, 0.3, 0.7)
# tau as a share of sigma_e.
TAU_FRACS = (0.0, 0.5)
COUNT_DISTRIBUTIONS = ('herg_ic50', 'uniform_1_to_4')
# Molecules of herg_ic50 by measurement count, capped at 4. Counted 2026-09-25
# with load_herg_ic50_measurements on the ChEMBL pull: 9,802 molecules.
HERG_IC50_COUNT_FALLBACK = {1: 8584, 2: 813, 3: 263, 4: 142}
NOISE_DRAWS = (0, 1, 2)

OOF_SPLITS = 5
ALS_MAX_ITER = 200
ALS_TOL = 1e-10
MIXEDLM_MAX_ROWS = 20000
# The model-error estimate by subtraction is floored at this share of the mean
# squared out-of-fold error, so no weight is infinite.
MODEL_ERROR_FLOOR_FRAC = 0.01

ARMS = ('clean', 'plain_mean', 'offset_removed', 'weights_n19_fitted',
        'weights_within_var', 'offset_removed_within_var_weights',
        'weights_rho0_oracle', 'weights_true_rho_oracle')
ARM_TO_BEAT = 'plain_mean'

# The spaces each decided model is fitted in, per outer fold (chat 5,
# 2026-10-05): its rank-1 and rank-2 base representation and its two best
# hybrids, all ranked by a search on that fold's training molecules. Two
# models x 4 = 8 (R5). n39 finds no neighbours, so the distance hybrid and the
# similarity hybrid are not among its spaces (handoff 2.3 item 1). Until
# 2026-10-05 each model had up to six hybrid spaces, two search families
# (predictive, similarity) x (top 2 general + top 1 distance).
BASE_SPACES = ('base_1', 'base_2')
HYBRID_SPACES = ('hybrid', 'hybrid__general2')
SPACES = tuple(CONFIGURATION_SPACES)

# The rows file and the checkpoint carry this so nothing written under the
# whole-dataset selection is read back (handoff 2.4 item 11).
SELECTION_VERSION = 'per_fold_2026_10_05'


# ── Count distributions and the source draw ────────────────────────────────

def count_shares(name):
    """(shares over k = 1..MAX, provenance string)."""
    ks = np.arange(1, MAX_SOURCES_PER_MOLECULE + 1)
    if name == 'uniform_1_to_4':
        return np.full(ks.size, 1.0 / ks.size), 'uniform over 1..4'
    if name != 'herg_ic50':
        raise ValueError(f'unknown count distribution {name!r}')
    try:
        from kirby.datasets.herg import load_herg_ic50_measurements
        d = load_herg_ic50_measurements()
        _, c = np.unique(np.asarray(d['replicate_groups']), return_counts=True)
        c = np.minimum(c, MAX_SOURCES_PER_MOLECULE)
        counts = np.array([(c == k).sum() for k in ks], dtype=float)
        src = f'herg_ic50 live: {int(counts.sum())} molecules'
    except Exception as exc:
        counts = np.array([HERG_IC50_COUNT_FALLBACK[int(k)] for k in ks],
                          dtype=float)
        src = (f'herg_ic50 fallback counts of 2026-09-25 (live load raised '
               f'{type(exc).__name__}: {exc})')
    return counts / counts.sum(), src


def pass_seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7fffffff


def draw_structure(n, shares, seed, n_sources=N_SOURCES):
    """Which sources measured each molecule, and the standard normals.

    Level-, rho- and tau-free: `simulate` only scales what is drawn here.
    """
    rng = np.random.default_rng(int(seed))
    ks = rng.choice(np.arange(1, len(shares) + 1), size=n, p=shares)
    mol, src = [], []
    for i, k in enumerate(ks):
        s = rng.choice(n_sources, size=int(k), replace=False)
        mol.extend([i] * int(k))
        src.extend(s.tolist())
    mol = np.asarray(mol, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    return {'k': ks.astype(np.int64), 'mol': mol, 'src': src,
            'u': rng.standard_normal(n), 'v': rng.standard_normal(mol.size),
            'z': rng.standard_normal(n_sources), 'n_sources': int(n_sources)}


def simulate(y_clean, st, sigma_e, rho, tau):
    """Per-measurement values, per-molecule mean label and the true offsets."""
    y_clean = np.asarray(y_clean, dtype=float)
    b = float(tau) * st['z']
    e = float(sigma_e) * (np.sqrt(rho) * st['u'][st['mol']]
                          + np.sqrt(1.0 - rho) * st['v'])
    val = y_clean[st['mol']] + b[st['src']] + e
    mean = np.bincount(st['mol'], weights=val, minlength=len(y_clean)) / st['k']
    return val, mean, b


def restrict(st, val, idx):
    """The measurements of the molecules in idx, renumbered 0..len(idx)-1."""
    idx = np.asarray(idx)
    pos = np.full(len(st['k']), -1, dtype=np.int64)
    pos[idx] = np.arange(idx.size)
    keep = pos[st['mol']] >= 0
    return pos[st['mol'][keep]], st['src'][keep], val[keep]


# ── Offsets ────────────────────────────────────────────────────────────────

def offsets_als(mol, src, val, n_sources, max_iter=ALS_MAX_ITER, tol=ALS_TOL):
    """Two-way additive least squares. Returns (b_hat centred, s_w^2, info)."""
    n_mol = int(mol.max()) + 1 if mol.size else 0
    k = np.bincount(mol, minlength=n_mol).astype(float)
    ns = np.bincount(src, minlength=n_sources).astype(float)
    b = np.zeros(n_sources)
    it = 0
    for it in range(1, max_iter + 1):
        theta = np.bincount(mol, weights=val - b[src], minlength=n_mol) / k
        new_b = np.zeros(n_sources)
        seen = ns > 0
        new_b[seen] = (np.bincount(src, weights=val - theta[mol],
                                   minlength=n_sources)[seen] / ns[seen])
        new_b[seen] -= new_b[seen].mean()
        step = float(np.max(np.abs(new_b - b)))
        b = new_b
        if step < tol:
            break
    theta = np.bincount(mol, weights=val - b[src], minlength=n_mol) / k
    res = val - theta[mol] - b[src]
    df = val.size - n_mol - max(int((ns > 0).sum()) - 1, 0)
    s_w2 = float(np.sum(res ** 2) / df) if df > 0 else float('nan')
    return b, s_w2, {'method': 'als', 'iterations': int(it),
                     'residual_df': int(df)}


def offsets_mixedlm(mol, src, val, n_sources):
    """statsmodels MixedLM: source fixed effects, molecule random intercept."""
    import statsmodels.api as sm
    used = np.unique(src)
    X = (src[:, None] == used[None, :]).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = sm.MixedLM(val, X, groups=mol).fit(reml=True)
    fe = np.asarray(fit.fe_params, dtype=float)[:used.size]
    b = np.zeros(n_sources)
    b[used] = fe - fe.mean()
    return b, float(fit.scale), {
        'method': 'mixedlm', 'converged': bool(getattr(fit, 'converged', True)),
        'molecule_variance': float(np.asarray(fit.cov_re).ravel()[0])}


def fit_offsets(mol, src, val, n_sources, method='mixedlm'):
    """Centred source offsets and s_w^2, by the method asked for or ALS."""
    reason = None
    if method == 'mixedlm' and val.size <= MIXEDLM_MAX_ROWS:
        try:
            return offsets_mixedlm(mol, src, val, n_sources)
        except Exception as exc:
            reason = f'mixedlm raised {type(exc).__name__}: {exc}'
    elif method == 'mixedlm':
        reason = (f'{val.size} measurements > MIXEDLM_MAX_ROWS '
                  f'{MIXEDLM_MAX_ROWS}')
    b, s_w2, info = offsets_als(mol, src, val, n_sources)
    if reason:
        info['fell_back_from_mixedlm_because'] = reason
    return b, s_w2, info


def rho_hat_from(s_w2, sigma_e):
    """1 - s_w^2 / sigma_e^2, clipped to [0, 1]. sigma_e is from outside the labels."""
    if not np.isfinite(s_w2) or sigma_e <= 0:
        return float('nan')
    return float(np.clip(1.0 - s_w2 / sigma_e ** 2, 0.0, 1.0))


def offset_recovery(b_hat, b_true, used):
    """b_hat against the centred true b, over the sources that measured something."""
    bt = np.asarray(b_true, dtype=float)[used]
    bh = np.asarray(b_hat, dtype=float)[used]
    bt = bt - bt.mean()
    out = {'n_sources': int(used.size),
           'rmse': float(np.sqrt(np.mean((bh - bt) ** 2))),
           'corr': None}
    if used.size >= 3 and np.std(bt) > 0 and np.std(bh) > 0:
        out['corr'] = float(np.corrcoef(bh, bt)[0, 1])
    return out


# ── Weights ────────────────────────────────────────────────────────────────

def weights_from_variance(model_error, label_var):
    """1 / (model_error + label_var), normalised to average one."""
    return inverse_variance_weights(model_error, label_var)


def subtracted_model_error(y, oof, label_var, floor_frac=MODEL_ERROR_FLOOR_FRAC):
    """Mean squared out-of-fold error minus the mean label variance, floored."""
    mse = float(np.mean((np.asarray(y) - np.asarray(oof)) ** 2))
    sd2 = mse - float(np.mean(label_var))
    floor = floor_frac * mse
    return max(sd2, floor), {'mean_squared_oof_error': mse,
                             'model_error_by_subtraction': sd2,
                             'floored': bool(sd2 < floor)}


def rho_aware_variance(sigma_e, rho, k, tau2=0.0):
    """sigma_e^2 [rho + (1 - rho) / k] + tau^2 / k, per molecule."""
    k = np.asarray(k, dtype=float)
    return sigma_e ** 2 * (rho + (1.0 - rho) / k) + tau2 / k


# ── The real-data block ───────────────────────────────────────────────────

def real_pairs(labels, rec_mol, rec_doc, train_mol, seed):
    """One pair of measurements per training molecule measured 2+ times.

    rec_mol maps each measurement row to its molecule index; train_mol are the
    training molecules. With rec_doc the pair is drawn from two different
    documents where the molecule has them. Returns (molecule, value_a, value_b,
    counts).
    """
    rng = np.random.default_rng(int(seed))
    rows_of = {}
    for r, m in enumerate(rec_mol):
        rows_of.setdefault(int(m), []).append(r)
    mols, a, b = [], [], []
    n_multi = n_cross = n_same_only = 0
    for m in train_mol:
        rows = rows_of.get(int(m), [])
        if len(rows) < 2:
            continue
        n_multi += 1
        rows = np.asarray(rows)
        if rec_doc is not None:
            docs = np.asarray(rec_doc)[rows]
            if np.unique(docs).size < 2:
                n_same_only += 1
                continue
            i = rng.integers(rows.size)
            other = rows[docs != docs[i]]
            j_row = other[rng.integers(other.size)]
            pair = (rows[i], j_row)
            n_cross += 1
        else:
            i, j = rng.choice(rows.size, size=2, replace=False)
            pair = (rows[i], rows[j])
        mols.append(int(m))
        a.append(float(labels[pair[0]]))
        b.append(float(labels[pair[1]]))
    return (np.asarray(mols, dtype=np.int64), np.asarray(a), np.asarray(b),
            {'training_molecules_measured_2_plus': n_multi,
             'pairs_from_two_documents': n_cross if rec_doc is not None else None,
             'molecules_with_one_document_only': (n_same_only
                                                  if rec_doc is not None else None),
             'pairs_used': len(mols)})


def pair_residual_correlation(ra, rb):
    """Double-entry Pearson correlation: the pair's order carries no meaning."""
    x = np.concatenate([ra, rb])
    y = np.concatenate([rb, ra])
    if x.size < 6 or np.std(x) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pooled_within_variance(labels, rec_mol, train_mol):
    tm = set(int(m) for m in train_mol)
    vals = {}
    for r, m in enumerate(rec_mol):
        if int(m) in tm:
            vals.setdefault(int(m), []).append(float(labels[r]))
    ss = df = 0.0
    for v in vals.values():
        if len(v) >= 2:
            v = np.asarray(v)
            ss += float(np.sum((v - v.mean()) ** 2))
            df += len(v) - 1
    return (ss / df) if df > 0 else float('nan')


# ── Runner ────────────────────────────────────────────────────────────────

def _fit_predict(factory, X_tr, y_tr, X_va, metric_name, weights=None,
                 dataset_name=None, model_name=None):
    """One fit's held-out prediction. dataset_name / model_name, added
    2026-09-25, route a seeded model through noise_cache
    (n32.fit_predict_cached); the prediction is the same."""
    return fit_predict_cached(dataset_name, model_name, factory, X_tr, y_tr,
                              X_va, metric_name, sample_weight=weights)


def _passes(dataset_name, count_dists, levels, rhos, tau_fracs, draws,
            n_molecules=None):
    # The decided level grid (KNN_SEARCH_SPACE_RECORD "The noise grid",
    # Adelaide 2026-10-05): 0.5, 1.0, 1.5 on full_knn, 1.0 on fast_knn, after
    # 0 is dropped (the clean arm of every pass is level 0). n_molecules sizes
    # the group on a dataset the record's list does not name (herg_ic50,
    # 9,802 molecules; chat 5, 2026-10-05). The source error is n39's own
    # kind, not one of the six decided kinds; its level is in the same unit
    # as gaussian's (a share of the clean label SD).
    # DECIDED 2026-10-06 (Adelaide delegated): the injected error size sigma_e
    # stays on the decided level grid as it is here, all levels kept: 108
    # passes per fold on a full_knn dataset (2 count distributions x 3 levels
    # x 3 rho x 2 tau x 3 draws), 36 on fast_knn.
    lv = [float(l) for l in (noise_levels(dataset_name, is_clf=False,
                                          n_molecules=n_molecules)
                             if levels is None else levels) if float(l) > 0]
    out = []
    for cd in count_dists:
        for l in lv:
            for rho in rhos:
                for tf in tau_fracs:
                    for d in draws:
                        out.append((cd, l, float(rho), float(tf), int(d)))
    return out


def cond_key(cd, rho, tf):
    return f'{cd}|rho{rho:g}|tau{tf:g}'


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N39: simulated sources with correlated errors and offsets.

    force_rerun (run_dataset.py --force-rerun, via run_all) sets aside the
    rows file and the per-pass checkpoint and recomputes every pass.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4):
      1. Models are the dataset's two decided models (DECISIONS.md Step 1).
         Until 2026-10-05 they were rf, xgboost, svm, dnn and lightgbm on
         every dataset.
      2. Folds are noise_helpers.noise_outer_folds on the rows n39 scores (one
         per molecule), the splits the per-fold searches are filed under.
         Until 2026-10-05 they were _shared.cv_split.
      3. Per outer fold, one search on that fold's training molecules ranks
         each model's 2 base representations and 2 hybrids
         (_shared.fold_standard_selection): 2 models x 4 = 8 fitted
         configurations. Until 2026-10-05 the base representations and up to
         six hybrids per model were read once from the whole-dataset stored
         search, which had seen every held-out molecule.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N39: SEVERAL SOURCES PER MOLECULE, CORRELATED ERRORS")
    print("=" * 70)

    if data['task_type'] != 'regression':
        print(f"  SKIPPED: {dataset_name} is {data['task_type']}. A class label "
              f"has no source offset or error variance.")
        return {'skipped': 'not_regression', 'task_type': data['task_type']}

    labels = np.asarray(data['labels'], dtype=float)
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    offset_method = kwargs.get('offset_method') or 'mixedlm'
    hybrid_labels = kwargs.get('hybrid_labels') or 'clean'
    if hybrid_labels not in ('clean', 'noisy'):
        raise ValueError(f"hybrid_labels must be 'clean' or 'noisy', got "
                         f"{hybrid_labels!r}")

    # One clean label per molecule. On a dataset that keeps repeats it is the
    # mean of the molecule's measurements; the real-data block reads the rows.
    # 2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11): the folds
    # and the per-fold searches run on these molecule rows too, so all of a
    # molecule's measurements sit on one side of every split and the search
    # sees the rows n39 scores. The rows and the label are the ones n19 uses
    # (first measurement row, mean of all measurements), so n39 files its
    # searches under n19's SEARCH_CONDITION_SUFFIX and the two share them.
    rep_groups = build_replicate_groups(data)
    if rep_groups is not None:
        stats = measurement_stats(labels, rep_groups)
        y_ent = np.asarray(stats['mean_all'], dtype=float)
        from experiments.noise.n29_document_bias import record_to_molecule
        rec_mol = record_to_molecule(rep_groups)
        if not np.array_equal(np.bincount(rec_mol), stats['n_measurements']):
            raise RuntimeError("n39: molecule order disagrees with "
                               "measurement_stats.")
        rec_doc = data.get('record_documents')
        rec_doc = (None if rec_doc is None
                   else np.unique(np.asarray(rec_doc).astype(str),
                                  return_inverse=True)[1])
        ent_data = _molecule_rows_data(data, stats, y_ent, dataset_name)
        search_suffix = SEARCH_CONDITION_SUFFIX
    else:
        stats = None
        y_ent = labels
        rec_mol = rec_doc = None
        ent_data = data
        search_suffix = None
    ent_groups = (None if ent_data.get('groups') is None
                  else np.asarray(ent_data['groups']))
    n_ent = len(y_ent)
    label_sd = float(np.std(y_ent))

    # R1 (chat 5, 2026-10-05): the dataset's two decided models, nothing
    # else; knn runs only inside the distance search, which n39 does not use.
    # Replaces downstream_model_list (config['models'], five models, plus rf
    # and svm). A `models` kwarg (a smoke, a direct caller) can only narrow
    # the decided pair; a name outside it is dropped and written down.
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
              f"n39 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type='regression', n_samples=n_ent)
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n39: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    count_dists = list(kwargs.get('count_distributions') or COUNT_DISTRIBUTIONS)
    shares_by_cd, shares_src = {}, {}
    for cd in count_dists:
        shares_by_cd[cd], shares_src[cd] = count_shares(cd)
    rhos = tuple(kwargs.get('rhos') or RHOS)
    tau_fracs = tuple(kwargs.get('tau_fracs') if kwargs.get('tau_fracs')
                      is not None else TAU_FRACS)
    draws = tuple(kwargs.get('draws') or NOISE_DRAWS)
    passes = _passes(dataset_name, count_dists, kwargs.get('levels'), rhos,
                     tau_fracs, draws, n_molecules=n_ent)

    folds, n_folds = noise_outer_folds(ent_data, config)
    n_cfg = len(models) * len(SPACES)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(SPACES),
        'fitted_configurations_per_fold': n_cfg,
        'selection_scope': 'fold_train_rows',
        'selection_version': SELECTION_VERSION,
        'selection_note': (
            'Per outer fold and per decided model: the 2 best base '
            'representations and the 2 best hybrids, ranked by one search on '
            "that fold's training molecules with their clean labels (decided "
            'grid, pruning off). Nothing is read from a whole-dataset search.'),
        'search_rows': ('one row per molecule (its first measurement row)'
                        if rep_groups is not None else 'the dataset as loaded'),
        'search_labels': ("each training molecule's mean over all of its "
                          'measurements' if rep_groups is not None
                          else 'the clean labels'),
        'search_condition_suffix': search_suffix,
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'per_fold_configurations': {},
        'arms': list(ARMS),
        'arm_to_beat': ARM_TO_BEAT,
        'n_sources': N_SOURCES,
        'max_sources_per_molecule': MAX_SOURCES_PER_MOLECULE,
        'count_distributions': count_dists,
        'count_shares': {cd: [float(x) for x in s]
                         for cd, s in shares_by_cd.items()},
        'count_distribution_source': shares_src,
        'rhos': list(rhos),
        'tau_fracs_of_sigma_e': list(tau_fracs),
        'sigma_e_units': 'level x the SD of the clean label column',
        'levels_source': (
            'kwargs levels' if kwargs.get('levels') is not None else
            'dataset_policy.noise_levels with the molecule count (the decided '
            'grid, 0 dropped: the clean arm in every pass is level 0)'),
        'clean_label_sd': label_sd,
        'offset_method_asked': offset_method,
        'mixedlm_max_rows': MIXEDLM_MAX_ROWS,
        'oof_splits': oof_splits,
        'hybrid_labels': hybrid_labels,
        'molecules': int(n_ent),
        'measurements_per_row_dataset': (int(len(labels))
                                         if rep_groups is not None else None),
        'rho_identifiability_note': (
            'rho is not identifiable from the labels alone: the shared error '
            'is the same in every source of a molecule. rho_hat = 1 - s_w^2 / '
            'sigma_e^2 uses the injected sigma_e, so it is written as a '
            'diagnostic and no arm uses it (2026-09-28). The weights depend on '
            'rho only through a + b/k: weights_within_var takes b from the '
            'within-molecule variance and a from the model\'s out-of-fold '
            'error.'),
        'oracle_arms_note': (
            'weights_rho0_oracle and weights_true_rho_oracle use the true '
            'sigma_e, tau and rho, and a model error read off an out-of-fold '
            'LightGBM on the CLEAN training labels.'),
        'features_note': (
            "Each decided model has four configurations per outer fold "
            "(configurations_per_model), named per fold under "
            "per_fold_configurations. hybrid_labels=clean: each hybrid is "
            "rebuilt once per fold from the clean training labels and shared "
            "by every arm and pass. hybrid_labels=noisy: rebuilt per pass "
            "from the plain_mean labels. The recipe is chosen on clean labels "
            "either way."),
        'robustness_note': (
            'damaged_minus_clean in the summary is the plain_mean score minus '
            "the clean score, per model and configuration, in the dataset's "
            'own metric (read with lower_is_better), as a mean over draws per '
            'fold and then over folds. No auc_norm is computed here.'),
        'claim_scope': ('Sources, offsets and correlation are simulated. Every '
                        'number is confined to this dataset.'),
        'similarity_note': ('n39 finds no neighbours, so it has no similarity '
                            'axis and no neighbour space; its rows carry '
                            'model_input|none|process.'),
        'controls_run': controls_enabled(),
        'rows_note': ('D is plain_mean, D\' each other arm, score_clean the '
                      'clean arm. noise_condition in a row is '
                      '<count distribution>|rho<rho>|tau<tau>, level the '
                      'sigma_e level. model_input is <configuration>/<model>.'),
    }
    results['passes'] = [f'{cond_key(cd, r, t)}|{l:g}|d{d}'
                         for cd, l, r, t, d in passes]
    print(f"  {n_ent} molecules, {n_folds} outer folds (running "
          f"{results['folds_run']}), decided models {models}")
    print(f"  {len(passes)} passes per fold (count distribution x level x rho "
          f"x tau x draw); {n_cfg} fitted configurations per fold "
          f"({len(models)} models x {list(SPACES)})")

    def oof_fn(X, y, g):
        # Cached on the content hash of X, y and groups (2026-09-25).
        # DECIDED 2026-10-06 (Adelaide delegated): this out-of-fold LightGBM
        # is part of the method (the model-error estimate every weighted arm
        # reads), not a downstream model, so it stays as it is, also on
        # datasets whose two decided models do not include lightgbm.
        return _nc.cached_oof_predict(
            dataset_name, X, y, g if g is not None else np.arange(len(y)),
            n_splits=oof_splits)

    # Rows and the per-pass checkpoint, 2026-09-25 (n32's PassLedger). A pass
    # is one (count distribution, rho, tau, level, draw, fold); the real-data
    # block is checkpointed per fold as 'real|f<fold>'. The file stem is
    # main()'s result suffix. The signature reads `results` (models,
    # selection_version, n_folds) and the folds, so a file written under the
    # whole-dataset selection is set aside, never read back (2026-10-05).
    ledger = PassLedger(
        output_dir, EXP_ID, dataset_name, 'sources',
        signature=ledger_signature(
            data, config, results, [(tr, va) for _, tr, va in folds],
            extra={'selection': SELECTION_VERSION,
                   'passes': results['passes']}),
        restart=bool(force_rerun) or restart_flag(kwargs))
    restored = ledger.restored() or {}

    acc = restored.get('acc') or {}
    tails = restored.get('tails') or {}
    est = restored.get('est') or {}
    errors = dict(restored.get('errors') or {})
    fold_records = dict(restored.get('fold_records') or {})
    real_block = restored.get('real_block')
    if rec_mol is not None and real_block is None:
        real_block = {
            'per_fold': [], 'documents_available': rec_doc is not None,
            'published_error_label_units':
                REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)}
    if restored:
        print(f"  resumed {len(ledger.passes_done)} finished passes")

    def _state():
        return {'acc': acc, 'tails': tails, 'est': est, 'errors': errors,
                'real_block': real_block, 'fold_records': fold_records}

    structures = {}

    def _structure(cd, draw):
        # Level-, rho-, tau- and fold-free: one per (count distribution, draw).
        if (cd, draw) not in structures:
            structures[(cd, draw)] = draw_structure(
                n_ent, shares_by_cd[cd], pass_seed(dataset_name, cd, draw))
        return structures[(cd, draw)]

    t0 = time.time()
    for fold_i, train_idx, val_idx in folds:
        keys = [f'{cond_key(cd, r, t)}|{l:g}|d{d}f{fold_i}'
                for cd, l, r, t, d in passes]
        if rec_mol is not None:
            keys.append(f'real|f{fold_i}')
        if (str(fold_i) in fold_records
                and all(ledger.pass_done(k) for k in keys)):
            print(f"\n  Fold {fold_i}: every pass is on disk in the checkpoint")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out molecules")

        # ── This fold's configurations (R2, R4, R5) ──
        try:
            sel = fold_standard_selection(
                ent_data['reps'], y_ent, ent_groups, train_idx=train_idx,
                fold_i=int(fold_i), n_outer_folds=n_folds,
                dataset_name=dataset_name, models=list(models),
                augmentations=ent_data.get('augmentations'),
                task_type='regression', condition_suffix=search_suffix)
        except Exception as exc:
            why = (f'the per-fold search failed: {type(exc).__name__}: {exc}')
            print(f"    {why}")
            errors[f'f{fold_i}|fold_standard_selection'] = why
            fold_records[str(fold_i)] = {'error': why}
            continue
        entries = {m: _entry_from_selection(sel.get(m)) for m in models}

        y_cl = y_ent[train_idx]
        y_va = y_ent[val_idx]
        g_tr = ent_groups[train_idx] if ent_groups is not None else None

        # base_1 and base_2: no features are chosen from the labels, so one
        # matrix per fold serves every pass. FeatureScaler inside (0/1
        # features unscaled) and float32 kept: until 2026-10-05 every
        # representation was cast to float64 here and every hybrid feature
        # was standardised with StandardScaler (handoff 2.3 items 17, 18).
        base_ins = {}
        for m in models:
            base_ins[m] = fold_model_inputs(
                ent_data, dataset_name, m, entries[m], train_idx=train_idx,
                val_idx=val_idx, y_train=y_cl, factory=model_factories[m],
                metric=metric, spaces=BASE_SPACES)

        def configurations(y_feat):
            """[(model_input key, model, X_tr, X_va)] over every model's four
            configurations, the hybrids rebuilt on this fold's training
            molecules from y_feat with the fold search's own representation
            scores (dirty-run bug 1); and {key: why} for those not built."""
            out, unavailable = [], {}
            for m in models:
                h_ins, h_unav, _ = fold_model_inputs(
                    ent_data, dataset_name, m, entries[m],
                    train_idx=train_idx, val_idx=val_idx, y_train=y_feat,
                    factory=model_factories[m], metric=metric,
                    spaces=HYBRID_SPACES)
                b_ins, b_unav, _ = base_ins[m]
                for sp in SPACES:
                    src, un = ((h_ins, h_unav) if sp in HYBRID_SPACES
                               else (b_ins, b_unav))
                    if sp in src:
                        out.append((f'{sp}/{m}', m, src[sp][0], src[sp][1]))
                    else:
                        unavailable[f'{sp}/{m}'] = un.get(sp) or 'not built'
            return out, unavailable

        clean_cfgs = configurations(y_cl) if hybrid_labels == 'clean' else None
        frec = _fold_record(entries, base_ins, metric_name, lower_is_better)
        frec['n_train'] = int(len(train_idx))
        frec['n_val'] = int(len(val_idx))
        if clean_cfgs is not None:
            frec['configurations_unavailable'] = clean_cfgs[1]
        fold_records[str(fold_i)] = frec
        for m, rec in frec['models'].items():
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")

        # ── The real-data block, this fold ──
        if rec_mol is not None and not ledger.pass_done(f'real|f{fold_i}'):
            real_cfgs = clean_cfgs or configurations(y_cl)
            mols, va, vb, cnt = real_pairs(labels, rec_mol, rec_doc,
                                           train_idx, seed=fold_i)
            s_w2 = pooled_within_variance(labels, rec_mol, train_idx)
            frec_r = {'fold': int(fold_i), 'pairs': cnt,
                      'pooled_within_variance': s_w2, 'per_space': {}}
            pub = REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)
            frec_r['rho_at_published_error'] = (
                rho_hat_from(s_w2, pub) if pub else None)
            frec_r['rho_at_published_error_reason'] = (
                None if pub else f'no published per-measurement error for '
                                 f'{dataset_name} in REFERENCE_ERROR_LABEL_UNITS')
            pos = {int(m): j for j, m in enumerate(train_idx)}
            k_tr_real = stats['n_measurements'][train_idx]
            for key, _m, X_tr_s, _X_va in real_cfgs[0]:
                try:
                    p = oof_fn(X_tr_s, y_cl, g_tr)
                    pm = np.array([p[pos[int(m)]] for m in mols])
                    corr = pair_residual_correlation(va - pm, vb - pm)
                    su2, se2, finfo = fit_noise_variance(y_cl, p, k_tr_real)
                    frec_r['per_space'][key] = {
                        'pair_residual_correlation_upper_bound_on_rho': corr,
                        'n19_intercept_a': su2, 'n19_slope_b': se2,
                        'n19_fit': finfo}
                except Exception as exc:
                    errors[f'real|{key}|f{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
            real_block['per_fold'].append(frec_r)
            ledger.finish_pass(f'real|f{fold_i}', _state())
            print(f"    real pairs fold {fold_i}: {cnt['pairs_used']} pairs, "
                  f"within variance {s_w2:.4f}")

        # ── The simulation, this fold ──
        for cd, level, rho, tf, draw in passes:
            ck, lk = cond_key(cd, rho, tf), f'{level:g}'
            pk = f'd{draw}f{fold_i}'
            ledger_key = f'{ck}|{lk}|{pk}'
            if ledger.pass_done(ledger_key):
                continue
            st = _structure(cd, draw)
            sigma_e = level * label_sd
            tau = tf * sigma_e
            val, y_mean, b_true = simulate(y_ent, st, sigma_e, rho, tau)
            cell = acc.setdefault(ck, {}).setdefault(lk, {})
            tcell = tails.setdefault(ck, {}).setdefault(lk, {})
            pass_rows = []
            y_pl = y_mean[train_idx]
            k_tr = st['k'][train_idx]
            mol_t, src_t, val_t = restrict(st, val, train_idx)
            erec = {'pass': pk, 'sigma_e': sigma_e, 'tau': tau, 'rho_true': rho,
                    'training_molecules_with_2_plus_sources': int((k_tr >= 2).sum()),
                    'true_within_variance': (1.0 - rho) * sigma_e ** 2}
            try:
                b_hat, s_w2, oinfo = fit_offsets(mol_t, src_t, val_t, N_SOURCES,
                                                 method=offset_method)
                # rho_hat reads the injected sigma_e, so it is a DIAGNOSTIC
                # only (how well the sources recover rho); no arm uses it.
                r_hat = rho_hat_from(s_w2, sigma_e)
                used = np.unique(src_t)
                y_off = (np.bincount(mol_t, weights=val_t - b_hat[src_t],
                                     minlength=len(train_idx)) / k_tr)
                tau2_hat = float(np.var(b_hat[used])) if used.size > 1 else 0.0
                erec.update({'offset_method_used': oinfo['method'],
                             'offset_info': oinfo, 'within_variance_hat': s_w2,
                             'rho_hat': r_hat, 'tau2_hat': tau2_hat,
                             'offset_recovery': offset_recovery(b_hat, b_true,
                                                                used)})
            except Exception as exc:
                errors[f'{ck}|{lk}|{pk}|offsets'] = f'{type(exc).__name__}: {exc}'
                y_off = r_hat = s_w2 = None
                tau2_hat = 0.0
            s_w2_ok = s_w2 is not None and np.isfinite(s_w2)
            est.setdefault(ck, {}).setdefault(lk, []).append(erec)

            cfgs = clean_cfgs if clean_cfgs is not None else configurations(y_pl)
            for ekey, mname, X_tr_s, X_va_s in cfgs[0]:
                cell_models = [mname]
                cols = {'clean': (y_cl, None), 'plain_mean': (y_pl, None)}
                try:
                    # The oracle arms' model error: out-of-fold error on the
                    # clean labels in this configuration. Recomputed per pass
                    # (the cache returns it) rather than memoised by name,
                    # because under hybrid_labels=noisy the hybrid's features
                    # change per pass (2026-10-05).
                    pc = oof_fn(X_tr_s, y_cl, g_tr)
                    sd2_clean = float(np.mean((y_cl - pc) ** 2))
                    p_pl = oof_fn(X_tr_s, y_pl, g_tr)
                    su2, se2, _fi = fit_noise_variance(y_pl, p_pl, k_tr)
                    if su2 is not None:
                        cols['weights_n19_fitted'] = (
                            y_pl, inverse_variance_weights(su2, se2 / k_tr))
                    # 2026-09-28: no injected sigma_e. The per-molecule
                    # variance is read off the sources (s_w2, tau2_hat) and
                    # the rest off the model's out-of-fold error; the
                    # shared part rho sigma_e^2 cannot be told apart from
                    # the model error, and does not need to be, because the
                    # weight only needs their sum.
                    if s_w2_ok:
                        v = (s_w2 + tau2_hat) / np.asarray(k_tr, dtype=float)
                        sd2, _ = subtracted_model_error(y_pl, p_pl, v)
                        cols['weights_within_var'] = (
                            y_pl, weights_from_variance(sd2, v))
                    if y_off is not None:
                        cols['offset_removed'] = (y_off, None)
                        if s_w2_ok:
                            p_off = oof_fn(X_tr_s, y_off, g_tr)
                            v0 = s_w2 / np.asarray(k_tr, dtype=float)
                            sd2o, _ = subtracted_model_error(y_off, p_off, v0)
                            cols['offset_removed_within_var_weights'] = (
                                y_off, weights_from_variance(sd2o, v0))
                    cols['weights_rho0_oracle'] = (y_pl, weights_from_variance(
                        sd2_clean, rho_aware_variance(sigma_e, 0.0, k_tr,
                                                      tau ** 2)))
                    cols['weights_true_rho_oracle'] = (
                        y_pl, weights_from_variance(
                            sd2_clean, rho_aware_variance(sigma_e, rho, k_tr,
                                                          tau ** 2)))
                except Exception as exc:
                    errors[f'{ck}|{lk}|{ekey}|{pk}|arms'] = (
                        f'{type(exc).__name__}: {exc}')
                space_cell = cell.setdefault(ekey, {})
                tspace = tcell.setdefault(ekey, {})
                pass_scores = {}
                for arm in ARMS:
                    if arm not in cols:
                        continue
                    y_arm, w_arm = cols[arm]
                    try:
                        pred = _fit_predict(model_factories[mname],
                                            X_tr_s, y_arm, X_va_s,
                                            metric_name, w_arm,
                                            dataset_name=dataset_name,
                                            model_name=mname)
                        s = float(metric_fn(y_va, pred))
                    except Exception as exc:
                        errors[f'{ck}|{lk}|{ekey}|{arm}|{mname}|{pk}'] = (
                            f'{type(exc).__name__}: {exc}')
                        continue
                    space_cell.setdefault(arm, {}).setdefault(
                        mname, {})[pk] = s
                    pass_scores.setdefault(arm, {})[mname] = s
                    tspace.setdefault(arm, {}).setdefault(
                        mname, {})[pk] = tail_scores(y_va, pred)
                # ── Controls (KIRBY_NOISE_CONTROLS=1): each arm that
                # changed the plain_mean labels or their weights. ──
                # TODO(similarity-sweep 2026-09-25):
                # offset_removed_within_var_weights changes the labels AND
                # the weights at once; controls.py twins one change, so
                # it has no twin until the owner says which the control
                # should move (arm_result returns None for it).
                for arm in ARMS:
                    if arm in ('clean', ARM_TO_BEAT) or arm not in cols:
                        continue
                    y_arm, w_arm = cols[arm]
                    twins = control_twins(
                        arm_result(y_pl, y=y_arm, weight=w_arm), X_tr_s,
                        y_pl, (replicate_seed(arm, ck), draw, fold_i,
                               int(round(level * 1000))))
                    for tw, by_m in twin_scores(
                            dataset_name, twins, model_factories,
                            cell_models, X_va_s, y_va, metric).items():
                        pass_scores[f'{arm}|{tw}'] = by_m
                pass_rows += arm_rows(
                    dataset_name=dataset_name, exp_id=EXP_ID,
                    metric=metric, condition=ck, level=level, draw=draw,
                    fold=fold_i, model_input=ekey,
                    scores_by_arm=pass_scores, reference=ARM_TO_BEAT)
            ledger.add_rows(pass_rows)
            ledger.finish_pass(ledger_key, _state())
            rh = erec.get('rho_hat')
            print(f"    fold {fold_i} {ck} level {lk} d{draw}: rho_hat "
                  f"{'n/a' if rh is None else f'{rh:.3f}'} (true {rho:g}) "
                  f"[{time.time() - t0:.0f}s]")

    results['per_fold_configurations'] = convert_numpy(fold_records)
    results['real_data'] = convert_numpy(real_block) if real_block else None
    results['per_condition'] = convert_numpy(acc)
    results['tail_per_condition'] = convert_numpy(tails)
    results['estimates_per_pass'] = convert_numpy(est)
    results['d_vs_dprime'] = ledger.describe()
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, est, models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _fold_record(entries, base_ins, metric_name, lower_is_better):
    """What the result file says about one fold's configurations, per model:
    noise_helpers.configuration_record (base representations; hybrids with
    allocation, importance, budget in features, evaluator, representation
    count, augmentation condition) plus each one's score on the fold's
    training molecules (the search's own inner folds). 2026-10-05."""
    out = {'models': {}, 'metric': metric_name,
           'lower_is_better': bool(lower_is_better),
           'selection_scope': 'fold_train_rows',
           'held_out_score_where': (
               "per_condition[<condition>][<level>][<configuration>/<model>]"
               "[<arm>][<model>]['d<draw>f<fold>']: the held-out score of "
               "that configuration on this fold, graded against clean labels")}
    for m, e in entries.items():
        names = {}
        if e is not None:
            for i, sp in enumerate(BASE_SPACES):
                if len(e['base']) > i:
                    names[sp] = e['base'][i]
            for i, sp in enumerate(HYBRID_SPACES):
                if len(e['hybrids']) > i:
                    names[sp] = e['hybrids'][i].get('label')
        rec = configuration_record(e, names)
        if e is not None:
            # No neighbour space is built here; n39 finds no neighbours.
            rec['neighbour_spaces'] = []
            bd = e.get('base_detail') or {}
            rec['base_training_row_scores'] = {
                s['rep']: s.get('score') for s in (bd.get('selected') or [])}
            rec['hybrid_training_row_scores'] = {
                h.get('label'): h.get('score_mean') for h in e['hybrids']}
            _b_unav = (base_ins.get(m) or ({}, {}, {}))[1]
            rec['base_unavailable'] = {sp: why for sp, why in _b_unav.items()
                                       if sp in BASE_SPACES}
        out['models'][m] = convert_numpy(rec)
    return out


def _summarise(acc, est, models, metric_name, lower_is_better):
    """Every arm against plain_mean, paired per pass; rho_hat and b_hat per cell.

    2026-10-05 (chat 5): every description over folds goes through
    noise_helpers.fold_spread (mean and sd, or "one fold, no spread"), and
    damaged_minus_clean (plain_mean minus clean, in the dataset's own metric)
    is added per model and configuration."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'headline': ('passes won against plain_mean, paired per pass; a pass '
                        'is one (draw, fold)'),
           'per_condition': {}, 'estimates': {}}
    for ck, by_level in est.items():
        row = {}
        for lk, recs in by_level.items():
            rh = [r['rho_hat'] for r in recs
                  if r.get('rho_hat') is not None and np.isfinite(r['rho_hat'])]
            bc = [r['offset_recovery']['corr'] for r in recs
                  if r.get('offset_recovery', {}).get('corr') is not None]
            row[lk] = {
                'rho_true': recs[0]['rho_true'] if recs else None,
                'rho_hat_mean': float(np.mean(rh)) if rh else None,
                'rho_hat_spread': float(np.std(rh)) if rh else None,
                'offset_corr_mean': float(np.mean(bc)) if bc else None,
                'n_passes': len(recs),
            }
        out['estimates'][ck] = row

    def _by_fold(a, b, signed_gain):
        """Per fold, the mean over draws of a[k] - b[k] (or the gain of b over
        a when signed_gain), on the passes both scored; then fold_spread."""
        per = {}
        for k in sorted(set(a) & set(b)):
            if not (np.isfinite(a[k]) and np.isfinite(b[k])):
                continue
            if signed_gain:
                d = (a[k] - b[k]) if lower_is_better else (b[k] - a[k])
            else:
                d = float(a[k]) - float(b[k])
            per.setdefault(k.split('f')[-1], []).append(float(d))
        return fold_spread([float(np.mean(v)) for _f, v in sorted(per.items())])

    for ck, by_level in acc.items():
        crow = {}
        for lk, by_space in by_level.items():
            lrow = {}
            for space, by_arm in by_space.items():
                base = by_arm.get(ARM_TO_BEAT) or {}
                clean = by_arm.get('clean') or {}
                srow = {}
                for m in models:
                    b = base.get(m) or {}
                    if not b:
                        continue
                    r = {'plain_mean_mean': _mean_on(b, list(b)), 'per_arm': {}}
                    r['damaged_minus_clean'] = _by_fold(
                        b, clean.get(m) or {}, signed_gain=False)
                    for arm in ARMS:
                        if arm == ARM_TO_BEAT:
                            continue
                        g = (by_arm.get(arm) or {}).get(m) or {}
                        if not g:
                            continue
                        e = {'vs_plain_mean': _paired(b, g, lower_is_better),
                             'gain_vs_plain_mean_over_folds': _by_fold(
                                 b, g, signed_gain=True)}
                        if arm != 'clean':
                            c = clean.get(m) or {}
                            keys = sorted(set(c) & set(b) & set(g))
                            if keys:
                                sh, why = recovery_share(
                                    _mean_on(c, keys), _mean_on(b, keys),
                                    _mean_on(g, keys), lower_is_better)
                            else:
                                sh, why = None, 'no pass scored all three'
                            e['recovery_share'] = sh
                            e['recovery_share_refused_because'] = why
                        r['per_arm'][arm] = e
                    srow[m] = r
                lrow[space] = srow
            crow[lk] = lrow
        out['per_condition'][ck] = crow
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric_name}; paired per pass against plain_mean)")
    for ck, row in s['estimates'].items():
        for lk, e in row.items():
            rh = e['rho_hat_mean']
            oc = e['offset_corr_mean']
            print(f"  {ck} level {lk}: rho_hat "
                  f"{'n/a' if rh is None else f'{rh:.3f}'} (true "
                  f"{e['rho_true']:g}), offset corr "
                  f"{'n/a' if oc is None else f'{oc:.3f}'}")
    for ck, crow in s['per_condition'].items():
        for lk, lrow in crow.items():
            print(f"\n  {ck} level {lk}:")
            for space, srow in lrow.items():
                for m, r in srow.items():
                    dmc = r.get('damaged_minus_clean') or {}
                    print(f"    {space}: plain_mean {r['plain_mean_mean']:.4f}; "
                          f"plain_mean minus clean over folds "
                          f"{'n/a' if dmc.get('mean') is None else format(dmc['mean'], '+.4f')}"
                          f"{'' if dmc.get('sd') is None else ' (sd ' + format(dmc['sd'], '.4f') + ')'}"
                          f"{' (' + dmc['note'] + ')' if dmc.get('note') else ''}")
                    for arm, e in r['per_arm'].items():
                        v = e['vs_plain_mean']
                        print(f"      {arm:31s} won {v['passes_won']} of "
                              f"{v['n_passes_paired']} passes, mean "
                              f"{v['mean_gain']:+.4f} (spread "
                              f"{v['spread_over_pass_gains']:.4f})")
    if results.get('real_data'):
        print("\n  real repeat measurements:")
        for f in results['real_data']['per_fold']:
            for sp, e in f['per_space'].items():
                c = e['pair_residual_correlation_upper_bound_on_rho']
                print(f"    fold {f['fold']} {sp}: {f['pairs']['pairs_used']} "
                      f"pairs, residual correlation (upper bound on rho) "
                      f"{'n/a' if c is None else f'{c:.3f}'}")
    print("\n  The count of passes won is the headline. A mean gain no larger "
          "than the spread of the per-pass gains behind it is not a gain.")


def main():
    parser = create_experiment_parser(
        'N39: several sources per molecule, with correlated errors and offsets')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated levels (sigma_e in label SDs). '
                             'Default: the shared grid without 0')
    parser.add_argument('--rhos', type=str, default=None,
                        help=f'Comma-separated rho (default {RHOS})')
    parser.add_argument('--tau-fracs', type=str, default=None,
                        help=f'Comma-separated tau / sigma_e (default {TAU_FRACS})')
    parser.add_argument('--count-distributions', type=str, default=None,
                        help=f'Comma-separated, from {COUNT_DISTRIBUTIONS}')
    parser.add_argument('--draws', type=str, default=None,
                        help=f'Comma-separated draws (default {NOISE_DRAWS})')
    parser.add_argument('--offset-method', type=str, default=None,
                        choices=['mixedlm', 'als'],
                        help='How source offsets are fitted (default mixedlm)')
    parser.add_argument('--hybrid-labels', type=str, default=None,
                        choices=['clean', 'noisy'],
                        help='Labels each hybrid is built from (default clean)')
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
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.rhos:
        kw['rhos'] = [float(x) for x in args.rhos.split(',')]
    if args.tau_fracs:
        kw['tau_fracs'] = [float(x) for x in args.tau_fracs.split(',')]
    if args.count_distributions:
        kw['count_distributions'] = args.count_distributions.split(',')
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.offset_method:
        kw['offset_method'] = args.offset_method
    if args.hybrid_labels:
        kw['hybrid_labels'] = args.hybrid_labels

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = 'sources' + (f'_{args.tag}' if args.tag else '')
    save_experiment_results(res, args.output, EXP_ID, args.dataset,
                            filename_suffix=suffix)


if __name__ == '__main__':
    main()
