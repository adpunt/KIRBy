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
    the clean label column, the level taken from the suite's shared grid
    (`noise_levels`, 0 dropped). rho in RHOS, tau = TAU_FRACS x sigma_e.
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

THE ARMS. All five downstream models, in the base representation and each
hybrid space. Paired against `plain_mean`.

  clean                          fitted on the clean labels. The reference the
                                 recovery share divides by. Not a method.
  plain_mean                     the mean of each molecule's sources, unweighted.
  offset_removed                 the mean over sources of (y_is - b_hat_s).
  weights_n19_fitted             plain_mean labels, weight 1 / (sigma_u^2 +
                                 sigma_e^2 / k) with both read off n19's
                                 regression of squared out-of-fold error on 1/k.
  weights_rho_hat                plain_mean labels, weight 1 / (sd2_hat + v_i),
                                 v_i = sigma_e^2 [rho_hat + (1 - rho_hat)/k_i]
                                 + tau_hat^2 / k_i, tau_hat^2 the variance of the
                                 fitted offsets, sd2_hat the mean squared
                                 out-of-fold error minus mean(v_i), floored.
                                 Uses the injected sigma_e.
  offset_removed_rho_hat_weights offset_removed labels, the same weight without
                                 the tau term.
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

WHERE THE FEATURES COME FROM. The recipe per downstream model is read from the
precomputed pool, searched on clean labels, as in n34. hybrid_labels='clean'
(default) builds each hybrid ONCE PER FOLD from the clean training labels and
every arm and pass shares it. That differs from n34, which rebuilds per pass
from the noisy labels: n39 has count distributions x levels x rho x tau x draws
passes (216 at full settings on the default grid), and a rebuild per pass would
dominate the run. Every arm sees the same features, so the paired arm
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

import functools
import time
import warnings
import zlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS,
    get_best_search_for_downstream, pick_noise_base_rep,
    pool_recipes_by_family, describe_pool_family, POOL_SUBSETS,
)
from experiments.dataset_policy import noise_levels, REFERENCE_ERROR_LABEL_UNITS
from experiments.noise.noise_helpers import (
    hybrid_for_fold, pool_recipe_per_downstream_model, group_models_by_recipe,
    tail_scores,
)
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n07_remediation import _predict_for_metric
from experiments.noise.n19_measurement_count import (
    measurement_stats, fit_noise_variance, inverse_variance_weights,
)
from experiments.noise.n34_group_random_effects import (
    recovery_share, _paired, _mean_on,
)
from kirby.utils.noisy_labels import oof_predict

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
        'weights_rho_hat', 'offset_removed_rho_hat_weights',
        'weights_rho0_oracle', 'weights_true_rho_oracle')
ARM_TO_BEAT = 'plain_mean'

POOL_SUBSET = 'predictive_and_similarity'
SPACE_FOR_FAMILY = {
    'predictive': 'hybrid',
    'similarity': 'hybrid_similarity',
}


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

def _fit_predict(factory, X_tr, y_tr, X_va, metric_name, weights=None):
    m = factory()
    if weights is None:
        m.fit(X_tr, y_tr)
    else:
        m.fit(X_tr, y_tr, sample_weight=weights)
    return _predict_for_metric(m, X_va, metric_name)


def _passes(dataset_name, count_dists, levels, rhos, tau_fracs, draws):
    lv = [float(l) for l in (noise_levels(dataset_name, is_clf=False)
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


def run(data, config, output_dir, dataset_name, **kwargs):
    """N39: simulated sources with correlated errors and offsets."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N39: SEVERAL SOURCES PER MOLECULE, CORRELATED ERRORS")
    print("=" * 70)

    if data['task_type'] != 'regression':
        print(f"  SKIPPED: {dataset_name} is {data['task_type']}. A class label "
              f"has no source offset or error variance.")
        return {'skipped': 'not_regression', 'task_type': data['task_type']}

    labels = np.asarray(data['labels'], dtype=float)
    reps = data['reps']
    row_groups = data.get('groups')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'
    oof_splits = int(kwargs.get('oof_splits', OOF_SPLITS))
    offset_method = kwargs.get('offset_method') or 'mixedlm'
    hybrid_labels = kwargs.get('hybrid_labels') or 'clean'
    if hybrid_labels not in ('clean', 'noisy'):
        raise ValueError(f"hybrid_labels must be 'clean' or 'noisy', got "
                         f"{hybrid_labels!r}")

    # One clean label per molecule. On a dataset that keeps repeats it is the
    # mean of the molecule's measurements; the real-data block reads the rows.
    rep_groups = build_replicate_groups(data)
    if rep_groups is not None:
        stats = measurement_stats(labels, rep_groups)
        ent_rows = stats['row_index']
        y_ent = stats['mean_all']
        from experiments.noise.n29_document_bias import record_to_molecule
        rec_mol = record_to_molecule(rep_groups)
        if not np.array_equal(np.bincount(rec_mol), stats['n_measurements']):
            raise RuntimeError("n39: molecule order disagrees with "
                               "measurement_stats.")
        rec_doc = data.get('record_documents')
        rec_doc = (None if rec_doc is None
                   else np.unique(np.asarray(rec_doc).astype(str),
                                  return_inverse=True)[1])
    else:
        ent_rows = np.arange(len(labels))
        y_ent = labels
        rec_mol = rec_doc = None
    ent_reps = {k: np.asarray(v[ent_rows], dtype=np.float64)
                for k, v in reps.items()}
    ent_groups = (None if row_groups is None
                  else np.asarray(row_groups)[ent_rows])
    n_ent = len(y_ent)
    label_sd = float(np.std(y_ent))

    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS)
    model_factories = get_model_factories(
        downstream_models, task_type='regression', n_samples=n_ent)
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(f"n39: no downstream model could be built from "
                           f"{kwargs.get('models') or DOWNSTREAM_MODELS}.")
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))

    count_dists = list(kwargs.get('count_distributions') or COUNT_DISTRIBUTIONS)
    shares_by_cd, shares_src = {}, {}
    for cd in count_dists:
        shares_by_cd[cd], shares_src[cd] = count_shares(cd)
    rhos = tuple(kwargs.get('rhos') or RHOS)
    tau_fracs = tuple(kwargs.get('tau_fracs') if kwargs.get('tau_fracs')
                      is not None else TAU_FRACS)
    draws = tuple(kwargs.get('draws') or NOISE_DRAWS)
    passes = _passes(dataset_name, count_dists, kwargs.get('levels'), rhos,
                     tau_fracs, draws)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'models': downstream_models,
        'base_rep': base_rep_name,
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
            'sigma_e^2 uses the injected sigma_e, the stand-in for a published '
            'per-measurement error. The weights depend on rho only through '
            'a + b/k.'),
        'oracle_arms_note': (
            'weights_rho0_oracle and weights_true_rho_oracle use the true '
            'sigma_e, tau and rho, and a model error read off an out-of-fold '
            'LightGBM on the CLEAN training labels.'),
        'features_note': (
            'hybrid_labels=clean: each hybrid is built once per fold from the '
            'clean training labels and shared by every arm and pass. '
            'hybrid_labels=noisy: rebuilt per pass from the plain_mean labels.'),
        'claim_scope': ('Sources, offsets and correlation are simulated. Every '
                        'number is confined to this dataset.'),
    }

    splits = list(cv_split(ent_groups, n_folds, ent_reps[base_rep_name], y_ent,
                           mode=mode))

    by_model, no_recipe = pool_recipe_per_downstream_model(
        functools.partial(
            get_best_search_for_downstream, reps, labels, row_groups,
            budget=None, augmentations=data.get('augmentations'),
            dataset_name=dataset_name, task_type='regression', data=data,
            pool_subset=POOL_SUBSET),
        downstream_models)
    _raised = [d['lookup_error'] for d in no_recipe.values() if d['lookup_error']]
    results['hybrid_lookup_error'] = _raised[0] if _raised else None
    results['hybrid_unavailable_by_model'] = {
        m: d['why'] for m, d in no_recipe.items()}
    recipes_by_model, recipe_for_space_by_model, hybrid_unavailable = {}, {}, {}
    for _m in downstream_models:
        _m_recipes, _m_unavail = pool_recipes_by_family(
            by_model[_m]['meta'] if _m in by_model else None,
            SPACE_FOR_FAMILY, eligible_families=POOL_SUBSETS.get(POOL_SUBSET))
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
        print("  no hybrid recipe; base representation only")
    results['spaces_run'] = list(spaces)
    results['passes'] = [f'{cond_key(cd, r, t)}|{l:g}|d{d}'
                         for cd, l, r, t, d in passes]
    print(f"  {n_ent} molecules, {len(splits)} folds, models {downstream_models}")
    print(f"  {len(passes)} passes (count distribution x level x rho x tau x "
          f"draw); base representation {base_rep_name}")

    errors = {}

    def build_spaces(fold_i, train_idx, val_idx, y_feat):
        train_reps = {k: v[train_idx] for k, v in ent_reps.items()}
        val_reps = {k: v[val_idx] for k, v in ent_reps.items()}
        g_tr = ent_groups[train_idx] if ent_groups is not None else None
        out = {}
        sc = StandardScaler()
        out['base'] = [(sc.fit_transform(train_reps[base_rep_name]),
                        sc.transform(val_reps[base_rep_name]),
                        list(downstream_models))]
        for _sp in spaces:
            if _sp == 'base':
                continue
            cells = []
            for _bp_m, _mnames, _shareable in group_models_by_recipe(
                    {_m: recipe_for_space_by_model.get(_m, {}).get(_sp)
                     for _m in downstream_models}, downstream_models):
                X_tr_h, X_va_h = hybrid_for_fold(
                    train_reps, val_reps, y_feat, _bp_m, g_tr,
                    None if _shareable else model_factories[_mnames[0]](),
                    metric_fn, lower_is_better)
                if X_tr_h is None:
                    errors[f'{_sp}/{_mnames[0]}|f{fold_i}|build'] = (
                        'the hybrid could not be built')
                    continue
                sh = StandardScaler()
                cells.append((sh.fit_transform(X_tr_h), sh.transform(X_va_h),
                              list(_mnames)))
            if cells:
                out[_sp] = cells
        return out

    def oof_fn(X, y, g):
        return oof_predict(X, y, g if g is not None else np.arange(len(y)),
                           n_splits=oof_splits)

    fold_cache = {}

    def fold_ctx(fold_i, train_idx, val_idx):
        if fold_i not in fold_cache:
            fs = (build_spaces(fold_i, train_idx, val_idx, y_ent[train_idx])
                  if hybrid_labels == 'clean' else None)
            fold_cache[fold_i] = {'spaces': fs, 'clean_mse': {}}
        return fold_cache[fold_i]

    # ── The real-data block ──
    real_block = None
    if rec_mol is not None:
        real_block = {'per_fold': [], 'documents_available': rec_doc is not None,
                      'published_error_label_units':
                          REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)}
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            ctx = fold_ctx(fold_i, train_idx, val_idx)
            spaces_now = ctx['spaces'] or build_spaces(
                fold_i, train_idx, val_idx, y_ent[train_idx])
            g_tr = ent_groups[train_idx] if ent_groups is not None else None
            mols, va, vb, cnt = real_pairs(labels, rec_mol, rec_doc,
                                           train_idx, seed=fold_i)
            s_w2 = pooled_within_variance(labels, rec_mol, train_idx)
            frec = {'fold': fold_i, 'pairs': cnt,
                    'pooled_within_variance': s_w2, 'per_space': {}}
            pub = REFERENCE_ERROR_LABEL_UNITS.get(dataset_name)
            frec['rho_at_published_error'] = (
                rho_hat_from(s_w2, pub) if pub else None)
            frec['rho_at_published_error_reason'] = (
                None if pub else f'no published per-measurement error for '
                                 f'{dataset_name} in REFERENCE_ERROR_LABEL_UNITS')
            pos = {int(m): j for j, m in enumerate(train_idx)}
            for space, cells in spaces_now.items():
                for X_tr_s, _X_va, cell_models in cells:
                    key = (space if cell_models == list(downstream_models)
                           else f'{space}/{cell_models[0]}')
                    try:
                        p = oof_fn(X_tr_s, y_ent[train_idx], g_tr)
                        pm = np.array([p[pos[int(m)]] for m in mols])
                        corr = pair_residual_correlation(va - pm, vb - pm)
                        k_tr = stats['n_measurements'][train_idx]
                        su2, se2, finfo = fit_noise_variance(
                            y_ent[train_idx], p, k_tr)
                        frec['per_space'][key] = {
                            'pair_residual_correlation_upper_bound_on_rho': corr,
                            'n19_intercept_a': su2, 'n19_slope_b': se2,
                            'n19_fit': finfo}
                    except Exception as exc:
                        errors[f'real|{key}|f{fold_i}'] = (
                            f'{type(exc).__name__}: {exc}')
            real_block['per_fold'].append(frec)
            print(f"    real pairs fold {fold_i}: {cnt['pairs_used']} pairs, "
                  f"within variance {s_w2:.4f}")
    results['real_data'] = convert_numpy(real_block) if real_block else None

    # ── The simulation ──
    acc, tails, est = {}, {}, {}
    t0 = time.time()
    structures = {}
    for cd, level, rho, tf, draw in passes:
        skey = (cd, draw)
        if skey not in structures:
            structures[skey] = draw_structure(
                n_ent, shares_by_cd[cd], pass_seed(dataset_name, cd, draw))
        st = structures[skey]
        sigma_e = level * label_sd
        tau = tf * sigma_e
        val, y_mean, b_true = simulate(y_ent, st, sigma_e, rho, tau)
        ck, lk = cond_key(cd, rho, tf), f'{level:g}'
        cell = acc.setdefault(ck, {}).setdefault(lk, {})
        tcell = tails.setdefault(ck, {}).setdefault(lk, {})

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            pk = f'd{draw}f{fold_i}'
            y_va = y_ent[val_idx]
            y_cl = y_ent[train_idx]
            y_pl = y_mean[train_idx]
            k_tr = st['k'][train_idx]
            g_tr = ent_groups[train_idx] if ent_groups is not None else None
            mol_t, src_t, val_t = restrict(st, val, train_idx)
            erec = {'pass': pk, 'sigma_e': sigma_e, 'tau': tau, 'rho_true': rho,
                    'training_molecules_with_2_plus_sources': int((k_tr >= 2).sum()),
                    'true_within_variance': (1.0 - rho) * sigma_e ** 2}
            try:
                b_hat, s_w2, oinfo = fit_offsets(mol_t, src_t, val_t, N_SOURCES,
                                                 method=offset_method)
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
                y_off = r_hat = None
                tau2_hat = 0.0
            est.setdefault(ck, {}).setdefault(lk, []).append(erec)

            ctx = fold_ctx(fold_i, train_idx, val_idx)
            fspaces = ctx['spaces'] or build_spaces(fold_i, train_idx, val_idx,
                                                    y_pl)
            for space, cells in fspaces.items():
                for X_tr_s, X_va_s, cell_models in cells:
                    ekey = (space if cell_models == list(downstream_models)
                            else f'{space}/{cell_models[0]}')
                    cols = {'clean': (y_cl, None), 'plain_mean': (y_pl, None)}
                    try:
                        if ekey not in ctx['clean_mse']:
                            pc = oof_fn(X_tr_s, y_cl, g_tr)
                            ctx['clean_mse'][ekey] = float(np.mean((y_cl - pc) ** 2))
                        sd2_clean = ctx['clean_mse'][ekey]
                        p_pl = oof_fn(X_tr_s, y_pl, g_tr)
                        su2, se2, _fi = fit_noise_variance(y_pl, p_pl, k_tr)
                        if su2 is not None:
                            cols['weights_n19_fitted'] = (
                                y_pl, inverse_variance_weights(su2, se2 / k_tr))
                        if r_hat is not None and np.isfinite(r_hat):
                            v = rho_aware_variance(sigma_e, r_hat, k_tr, tau2_hat)
                            sd2, _ = subtracted_model_error(y_pl, p_pl, v)
                            cols['weights_rho_hat'] = (
                                y_pl, weights_from_variance(sd2, v))
                        if y_off is not None:
                            cols['offset_removed'] = (y_off, None)
                            if r_hat is not None and np.isfinite(r_hat):
                                p_off = oof_fn(X_tr_s, y_off, g_tr)
                                v0 = rho_aware_variance(sigma_e, r_hat, k_tr)
                                sd2o, _ = subtracted_model_error(y_off, p_off, v0)
                                cols['offset_removed_rho_hat_weights'] = (
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
                    for arm in ARMS:
                        if arm not in cols:
                            continue
                        y_arm, w_arm = cols[arm]
                        for mname in cell_models:
                            try:
                                pred = _fit_predict(model_factories[mname],
                                                    X_tr_s, y_arm, X_va_s,
                                                    metric_name, w_arm)
                                s = float(metric_fn(y_va, pred))
                            except Exception as exc:
                                errors[f'{ck}|{lk}|{ekey}|{arm}|{mname}|{pk}'] = (
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            space_cell.setdefault(arm, {}).setdefault(
                                mname, {})[pk] = s
                            tspace.setdefault(arm, {}).setdefault(
                                mname, {})[pk] = tail_scores(y_va, pred)
        rh = [e.get('rho_hat') for e in est[ck][lk] if e.get('rho_hat') is not None]
        print(f"    {ck} level {lk} d{draw}: rho_hat "
              f"{np.nanmean(rh) if rh else float('nan'):.3f} (true {rho:g}) "
              f"[{time.time() - t0:.0f}s]")

    results['per_condition'] = convert_numpy(acc)
    results['tail_per_condition'] = convert_numpy(tails)
    results['estimates_per_pass'] = convert_numpy(est)
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        acc, est, downstream_models, metric_name, lower_is_better))
    _print_summary(results, metric_name)
    return results


def _summarise(acc, est, models, metric_name, lower_is_better):
    """Every arm against plain_mean, paired per pass; rho_hat and b_hat per cell."""
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
                    for arm in ARMS:
                        if arm == ARM_TO_BEAT:
                            continue
                        g = (by_arm.get(arm) or {}).get(m) or {}
                        if not g:
                            continue
                        e = {'vs_plain_mean': _paired(b, g, lower_is_better)}
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
                    print(f"    {space}/{m}: plain_mean {r['plain_mean_mean']:.4f}")
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
    if args.base_rep:
        kw['base_rep'] = args.base_rep

    res = run(data, config, args.output, args.dataset, **kw)
    suffix = 'sources' + (f'_{args.tag}' if args.tag else '')
    save_experiment_results(res, args.output, EXP_ID, args.dataset,
                            filename_suffix=suffix)


if __name__ == '__main__':
    main()
