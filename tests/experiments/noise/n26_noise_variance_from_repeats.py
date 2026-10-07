#!/usr/bin/env python3
"""N26: how large is the label noise, measured from repeat measurements?

Paper: Noise.

THE QUESTION. Every cleaning method in this suite spends model error to remove
label error. That trade only pays when the label error is larger than the model
error. Nothing in the suite has measured the label error on real labels. This
file measures it three ways and puts the model error beside it.

WHERE IT COMES FROM. The research report of 2026-09-24 (items D1, D2 and D3)
proposed these three measurements. It described the data as assets valued by
one to four sources. Here a source is a ChEMBL record, and a molecule's count
is how many records it has.

NOTHING IS INJECTED IN THE MAIN MEASUREMENT. The training label of a molecule is
the mean of ALL its measurements. Its error variance is therefore the
per-measurement noise variance divided by its measurement count.

THE DATASETS. Only a loader that keeps repeat measurements can be used:
`herg_ic50` (one row per ChEMBL record), and `bindingdb_kd` and `kiba` (one
row per measured drug-protein pair; there a "molecule" below is a drug-protein
pair, since build_replicate_groups keys on drug|protein). Any other dataset
writes {'skipped': 'no_replicate_structure'}, the same as n19.

THE EXPERIMENT'S OWN ROWS (2026-10-05, chat 5; handoff 2.3 item 14, dirty-run
bug 11). One row per molecule: its first measurement row's features, and the
mean of all its measurements as the label (molecule_rows_data). The outer
folds, the per-fold searches and every feature matrix are on these rows, so
all of a molecule's measurements sit on one side of every split and the search
sees the rows the experiment scores. The searches are filed under
SEARCH_CONDITION_SUFFIX whenever these rows differ from the loader's. Since
2026-10-06 that string is n19's (MOLECULE_MEAN_SEARCH_SUFFIX), so n19 and n26
share one search per fold.

WHAT IS COMPUTED.

  1. ERROR BY MEASUREMENT COUNT. For each molecule, the out-of-fold prediction
     from each downstream model in each feature space. The residual is the mean
     label minus that prediction, in label units. Molecules are binned by count
     (1, 2, 3, 4, 5 or more). Each bin reports its molecule count, the mean
     residual, the mean absolute residual and the mean squared residual.

  2. NOISE VARIANCE FROM THE COUNT. If a molecule's mean label carries noise of
     variance s2e / k, then its expected squared residual is s2u + s2e / k.
     s2u is the model's own error variance. Regressing the squared residual on
     1/k gives s2e as the slope and s2u as the intercept, both in label units
     squared. Three fits:
       ols      ordinary least squares over every molecule
       binned   one point per distinct count, the mean squared residual of the
                molecules with that count, weighted by how many there are
       gamma    a gamma GLM with an identity link, because a squared residual
                is skewed. Recorded as failed when statsmodels does not converge.
     Beside them, s2e measured DIRECTLY: the pooled within-molecule variance of
     the repeat measurements, over molecules measured two or more times, each
     molecule weighted by its count minus one.

  3. RELIABILITY AND THE THRESHOLD. With sL2 = s2e * mean(1/k), the average
     noise variance of a training label:
       reliability     1 - sL2 / Var(mean label). The share of the label
                       variance that is not noise.
       r2_over_reliability  the model's out-of-fold R2 divided by the
                       reliability, per reliability estimate (added
                       2026-09-25): the R2 corrected for attenuation.
       r2_threshold    1 - sL2 / Var(true label), with
                       Var(true label) = Var(mean label) - sL2. A model whose
                       R2 is below this has an out-of-fold error larger than the
                       average label noise.
     Each is computed from the direct s2e and from each fitted s2e. The model's
     own out-of-fold R2 against the mean label is written beside them.
     Two estimates that need no model at all:
       split_half      on molecules measured two or more times, two different
                       measurements are drawn at random. The Pearson
                       correlation across molecules is the reliability of ONE
                       measurement. It is projected to k measurements with
                       Spearman-Brown, k r / (1 + (k - 1) r). Repeated with
                       different draws, and every repeat is written.
       icc             the one-way random-effects intraclass correlation, over
                       the same molecules, with the unequal-count correction.

A CHECK OF THE ESTIMATOR, NOT A RESULT. The report states it found no paper
using the 1/k regression and asks that it be checked first. So noise of known
variance s2_inj / k is added to each molecule's mean label. The out-of-fold
prediction is refitted for the first decided model in its rank-1 single
representation of each fold (space base_1), and the ols slope is refitted. What is written is the change in
slope against the injected s2_inj. If the estimator works, the two are close.
This arm says nothing about the real noise.

WHAT THIS CANNOT SEPARATE. A bias shared by every measurement of a molecule --
the same paper reporting the same number twice, or two papers copying a third --
does not shrink with the count. It lands in the intercept, not the slope. So s2e
here is the part of the noise that differs between records. The count of
molecules whose repeats are all identical is written, because those repeats
push the direct s2e toward zero.

WHERE THE FEATURES COME FROM (rewritten 2026-10-05, chat 5 of the alignment,
HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). Each of the dataset's two decided models
(DECISIONS.md Step 1) is fitted in four configurations per outer fold: its
rank-1 and rank-2 single representations (spaces base_1 and base_2) and its
two best hybrids with distinct selected features (spaces hybrid and
hybrid__general2). All four are ranked by ONE search on that fold's training
molecules (_shared.fold_standard_selection: decided grid, pruning off). Two
models x 4 = 8 fitted configurations (R5). The representation in base_1 can
differ between folds; which one each fold used is under
per_fold_configurations. Each hybrid is rebuilt on the fold's training
molecules from the fold search's recipe (noise_helpers.fold_model_inputs).
Nothing is read from a whole-dataset search. n26 finds no neighbours, so it
has no distance or similarity hybrid (handoff 2.3 item 1). The model error
s2u depends on the feature space. The noise variance s2e should not. How far
the fitted slopes move between spaces is written as a spread.

THE HEADLINE IS NOT A MEAN. Every (space, model) pair writes its own three
fitted slopes. Nothing is averaged across models or across spaces.

CHECKPOINT AND CACHES (2026-09-25, plan section 3). n26 changes no training
data, so it writes no D-versus-D' rows. Its out-of-fold predictions are
checkpointed per fold, and the scaled representations and hybrid builds go
through noise_cache. The checkpoint key carries SELECTION_VERSION since
2026-10-05, so nothing written under the whole-dataset selection is read back.

Usage:
    python n26_noise_variance_from_repeats.py --dataset herg_ic50 --mode full
    python n26_noise_variance_from_repeats.py --dataset bindingdb_kd --mode full
    python n26_noise_variance_from_repeats.py --dataset kiba --mode full
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
    get_domain_metric, get_model_factories, as_rep_dtype,
    fold_standard_selection,
)
from experiments.decided_grid import decided_models
from experiments.noise._ground_truth_providers import build_replicate_groups
from experiments.noise.n17_replicate_removal import build_entities
from experiments.noise.n19_measurement_count import (
    measurement_stats, MOLECULE_MEAN_SEARCH_SUFFIX,
)
# 2026-10-05 (chat 5): folds, the 8 fitted configurations per fold and the
# result-file record, from noise_helpers' CHAT 5 ALIGNMENT block. The
# whole-dataset pool reads (get_best_search_for_downstream,
# pool_recipe_per_downstream_model, pool_recipes_by_family,
# noise_base_reps_by_model) are gone (R2).
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_model_inputs, configuration_record,
    CONFIGURATION_SPACES, molecule_mean_data, check_molecule_order,
)
# The per-fold checkpoint (2026-09-25, plan section 3). It lives in n23 for
# the reason given at the top of its shared block.
from experiments.noise.n23_uncertainty_cleaning import PassCheckpoint

# The four fitted configurations of each decided model on each outer fold
# (2026-10-05, chat 5): rank-1 and rank-2 base representation and the two
# best hybrids, all ranked on that fold's training molecules. 2 models x 4 = 8
# (R5). The similarity-family hybrid ('hybrid_similarity') is gone: n26 finds
# no neighbours (handoff 2.3 item 1).
SPACES = tuple(CONFIGURATION_SPACES)

# The per-fold searches run on one row per molecule with its mean label, not
# on the loader's one row per measurement, so they are filed apart from the
# dataset's own per-fold searches (handoff 2.3 item 14, dirty-run bug 11).
# Used only when those rows differ from the loader's; otherwise the dataset's
# own per-fold searches are the same search and are read as they are.
# DECIDED 2026-10-06 (Adelaide delegated): n26 files its searches under n19's
# string, imported from n19_measurement_count. Both search the same molecule
# rows (first row's representations, float32, loader's groups on that row),
# the same mean labels, the same outer folds and the same two decided models,
# so the two experiments share one search per fold. Until 2026-10-06 n26 used
# 'n26_one_row_per_molecule_mean_label' and built the same searches again.
# 2026-10-07 (Adelaide): n19's string is now noise_helpers'
# MOLECULE_MEAN_SEARCH_SUFFIX, shared with n17, n29 and the ledger's herg_ic50
# search units, and the rows come from noise_helpers.molecule_mean_data.
SEARCH_CONDITION_SUFFIX = MOLECULE_MEAN_SEARCH_SUFFIX

# In the checkpoint key, so nothing written under the whole-dataset selection
# is read back (handoff 2.4 item 11).
SELECTION_VERSION = 'per_fold_2026_10_05'

# Measurement-count bins for the error-by-count table. The last one is open.
COUNT_BINS = (1, 2, 3, 4, 5)

# The binned fit uses a count only when this many molecules carry it. One
# molecule measured nine times is one squared residual, not a bin mean.
MIN_MOLECULES_PER_BIN = 5

# How many times two measurements are drawn for the split-half correlation.
N_SPLIT_HALF_REPEATS = 20

# The noise variances, in label units squared, the estimator check injects. A
# molecule measured k times gets noise of variance s2_inj / k.
INJECTED_VARIANCES = (0.1, 0.25, 0.5)

# The counts the split-half reliability is projected to.
PROJECT_TO_COUNTS = (1, 2, 3, 4, 5)


def measurements_by_molecule(labels, replicate_groups):
    """Every molecule's own measurements, in `measurement_stats`' order.

    That order is molecules sorted by the row of their first measurement. It is
    rebuilt here the same way, and the caller checks the counts agree.
    """
    labels = np.asarray(labels, dtype=np.float64)
    order = {}
    for i, g in enumerate(np.asarray(replicate_groups)):
        order.setdefault(g, []).append(i)
    gids = sorted(order, key=lambda g: order[g][0])
    return [labels[np.asarray(order[g], dtype=np.int64)] for g in gids]


def direct_noise_variance(values_per_molecule):
    """The pooled within-molecule variance of the repeats, label units squared.

    Over molecules measured two or more times. Each molecule's ddof=1 variance
    is weighted by its count minus one, which is the pooled variance of a
    one-way layout.
    """
    num, den, n_mol, n_identical = 0.0, 0, 0, 0
    for v in values_per_molecule:
        if len(v) < 2:
            continue
        n_mol += 1
        var = float(np.var(v, ddof=1))
        if var == 0.0:
            n_identical += 1
        num += (len(v) - 1) * var
        den += len(v) - 1
    return {
        's2e': (num / den) if den > 0 else float('nan'),
        'molecules_used': n_mol,
        'degrees_of_freedom': int(den),
        'molecules_whose_repeats_are_all_identical': n_identical,
    }


def error_by_count(residual, k):
    """One row per count bin: molecules, mean residual, mean |residual|, mean r2."""
    out = {}
    for i, lo in enumerate(COUNT_BINS):
        last = i == len(COUNT_BINS) - 1
        sel = (k >= lo) if last else (k == lo)
        key = f'{lo}+' if last else str(lo)
        r = residual[sel]
        out[key] = {
            'molecules': int(sel.sum()),
            'mean_residual': float(np.mean(r)) if r.size else float('nan'),
            'mean_abs_residual': (float(np.mean(np.abs(r))) if r.size
                                  else float('nan')),
            'mean_squared_residual': (float(np.mean(r ** 2)) if r.size
                                      else float('nan')),
        }
    return out


def fit_ols(r2, inv_k):
    """Squared residual on 1/k by least squares. (slope, intercept)."""
    if np.unique(inv_k).size < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'molecules': int(r2.size),
                'failed': 'every molecule has the same measurement count'}
    slope, intercept = np.polyfit(inv_k, r2, 1)
    return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
            'molecules': int(r2.size), 'failed': None}


def fit_binned(r2, k, min_per_bin=MIN_MOLECULES_PER_BIN):
    """One point per distinct count, weighted by the molecules carrying it."""
    xs, ys, ws, used = [], [], [], {}
    for c in np.unique(k):
        sel = k == c
        if int(sel.sum()) < min_per_bin:
            continue
        xs.append(1.0 / float(c))
        ys.append(float(np.mean(r2[sel])))
        ws.append(float(sel.sum()))
        used[int(c)] = int(sel.sum())
    if len(xs) < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'molecules_by_count_used': used,
                'failed': (f'fewer than 2 measurement counts carry '
                           f'{min_per_bin} or more molecules')}
    # np.polyfit weights multiply the residual, so the square root of the
    # molecule count gives each bin a squared-error weight of its count.
    slope, intercept = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
            'molecules_by_count_used': used, 'failed': None}


def fit_gamma(r2, inv_k):
    """Gamma GLM with identity link. Recorded as failed rather than raised."""
    if np.unique(inv_k).size < 2:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'failed': 'every molecule has the same measurement count'}
    try:
        import statsmodels.api as sm
        y = np.maximum(r2, 1e-12)
        X = sm.add_constant(inv_k)
        fam = sm.families.Gamma(link=sm.families.links.Identity())
        # The identity link can step outside the positive mean the gamma
        # family needs. Starting from the least-squares line keeps the first
        # step inside it when the data allow.
        start = np.polyfit(inv_k, y, 1)[::-1]
        res = sm.GLM(y, X, family=fam).fit(start_params=start, maxiter=200)
        if not res.converged:
            return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                    'failed': 'statsmodels did not converge'}
        intercept, slope = res.params
        return {'slope_s2e': float(slope), 'intercept_s2u': float(intercept),
                'failed': None}
    except Exception as exc:
        return {'slope_s2e': float('nan'), 'intercept_s2u': float('nan'),
                'failed': f'{type(exc).__name__}: {exc}'}


def _ratio(a, b):
    """a / b, NaN when either is not finite or b is not positive."""
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return float('nan')
    return float(a / b)


def reliability_from(s2e, mean_inv_k, var_y):
    """Reliability and the R2 threshold for one s2e, in the units they need."""
    if not np.isfinite(s2e):
        return {'s2e': float('nan'), 'label_noise_variance_sL2': float('nan'),
                'reliability': float('nan'), 'r2_threshold': float('nan')}
    sL2 = s2e * mean_inv_k
    var_true = var_y - sL2
    return {
        's2e': float(s2e),
        'label_noise_variance_sL2': float(sL2),
        'reliability': float(1.0 - sL2 / var_y) if var_y > 0 else float('nan'),
        'r2_threshold': (float(1.0 - sL2 / var_true) if var_true > 0
                         else float('nan')),
        'variance_of_the_true_label': float(var_true),
    }


def split_half(values_per_molecule, n_repeats, seed=0,
               project_to=PROJECT_TO_COUNTS):
    """Two measurements drawn per molecule, correlated across molecules.

    Over molecules measured two or more times. Returns one entry per repeat.
    The correlation is the reliability of ONE measurement, projected to k
    measurements by Spearman-Brown.
    """
    multi = [v for v in values_per_molecule if len(v) >= 2]
    if len(multi) < 3:
        return {'molecules': len(multi), 'repeats': [],
                'failed': 'fewer than 3 molecules measured twice or more'}
    rng = np.random.default_rng(seed)
    repeats = []
    for _ in range(int(n_repeats)):
        a = np.empty(len(multi))
        b = np.empty(len(multi))
        for j, v in enumerate(multi):
            i1, i2 = rng.choice(len(v), 2, replace=False)
            a[j], b[j] = v[i1], v[i2]
        if np.std(a) == 0 or np.std(b) == 0:
            repeats.append({'r_one_measurement': float('nan')})
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        repeats.append({
            'r_one_measurement': r,
            'spearman_brown_projected': {
                str(c): float(c * r / (1.0 + (c - 1.0) * r))
                for c in project_to},
        })
    return {'molecules': len(multi), 'repeats': repeats, 'failed': None}


def one_way_icc(values_per_molecule):
    """ICC(1) over molecules measured two or more times, unequal-count form."""
    multi = [np.asarray(v, dtype=float) for v in values_per_molecule
             if len(v) >= 2]
    g = len(multi)
    if g < 2:
        return {'icc': float('nan'), 'molecules': g,
                'failed': 'fewer than 2 molecules measured twice or more'}
    ks = np.array([len(v) for v in multi], dtype=float)
    N = float(ks.sum())
    grand = float(np.concatenate(multi).mean())
    ssb = float(sum(len(v) * (v.mean() - grand) ** 2 for v in multi))
    ssw = float(sum(((v - v.mean()) ** 2).sum() for v in multi))
    msb = ssb / (g - 1)
    msw = ssw / (N - g)
    n0 = (N - float((ks ** 2).sum()) / N) / (g - 1)
    denom = msb + (n0 - 1.0) * msw
    return {'icc': float((msb - msw) / denom) if denom > 0 else float('nan'),
            'molecules': g, 'measurements': int(N),
            'mean_square_between': msb, 'mean_square_within': msw,
            'effective_count_n0': float(n0), 'failed': None}


def _fit_predict(factory, X_tr, y_tr, X_va):
    m = factory()
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_va), dtype=np.float64).ravel()


def molecule_rows_data(data, stats, dataset_name):
    """The dataset as n26 fits and scores it: one row per molecule.

    2026-10-05 (chat 5, handoff 2.3 item 14, dirty-run bug 11). The loader
    keeps one row per measurement (herg_ic50: 12,008 rows over 9,802
    molecules). Until 2026-10-05 the hybrid recipes came from a whole-dataset
    search over those measurement rows while n26 fitted on molecules. Now the
    outer folds, the per-fold searches and every feature matrix are on these
    rows.

    Built by noise_helpers.molecule_mean_data since 2026-10-07 (shared with
    n17, n19, n29 and the ledger's herg_ic50 search units): representations
    and augmentation matrices are each molecule's first row (`row_index`),
    float32; groups the loader's groups on that row; the label the mean of
    all the molecule's measurements (`mean_all`), checked against `stats`.
    """
    out = molecule_mean_data(data, dataset_name)
    if out is None:
        raise RuntimeError(f'{dataset_name}: no repeat structure, so there are '
                           f'no molecule rows to build')
    check_molecule_order(out, stats['row_index'], stats['mean_all'],
                         who='n26')
    return out


def _entry_from_selection(s_m):
    """One model's fold_standard_selection result in the form
    noise_helpers.fold_model_inputs reads: 'base', 'hybrids', 'recipes'
    {'hybrid', 'hybrid__general2': (Blueprint, aug_condition)},
    'unavailable'. None stays None (that model's search failed).

    fold_standard_selection is called directly rather than
    noise_helpers.fold_configurations, because fold_configurations also runs
    the per-fold distance search and similarity search for the neighbour
    spaces, and n26 finds no neighbours, so those two searches would be built
    and never read (2026-10-05)."""
    if s_m is None:
        return None
    recipes, unavailable = {}, {}
    for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
        if len(s_m['hybrids']) > i:
            h = s_m['hybrids'][i]
            recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
        else:
            unavailable[space] = (f'the per-fold general search gave this '
                                  f'model fewer than {i + 1} hybrids with '
                                  f'distinct features')
    return {
        'base': list(s_m['base']),
        'base_detail': s_m.get('base_detail'),
        'hybrids': list(s_m['hybrids']),
        'hybrids_skipped_identical_features':
            s_m.get('hybrids_skipped_identical_features', []),
        'fit_configurations': (list(s_m['base'])
                               + [h['label'] for h in s_m['hybrids']]),
        'recipes': recipes,
        'unavailable': unavailable,
        'selection_scope': 'fold_train_rows',
    }


def _fold_record(entries):
    """What the result file says about one fold's configurations, per model:
    noise_helpers.configuration_record (base representations; hybrids with
    allocation, importance, budget in features, evaluator, representation
    count, augmentation condition) plus each one's score on the fold's
    training molecules (the search's own inner folds)."""
    out = {}
    for m, e in entries.items():
        names = {}
        if e is not None:
            for i, sp in enumerate(('base_1', 'base_2')):
                if len(e['base']) > i:
                    names[sp] = e['base'][i]
            for i, sp in enumerate(('hybrid', 'hybrid__general2')):
                if len(e['hybrids']) > i:
                    names[sp] = e['hybrids'][i].get('label')
        rec = configuration_record(e, names)
        if e is not None:
            # No neighbour space is built; n26 finds no neighbours.
            rec['neighbour_spaces'] = []
            bd = e.get('base_detail') or {}
            rec['base_training_row_scores'] = {
                s['rep']: s.get('score') for s in (bd.get('selected') or [])}
            rec['hybrid_training_row_scores'] = {
                h.get('label'): h.get('score_mean') for h in e['hybrids']}
        out[m] = convert_numpy(rec)
    return out


def run(data, config, output_dir, dataset_name, **kwargs):
    """N26: label-noise variance, reliability and the R2 threshold, from repeats."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N26: HOW LARGE IS THE LABEL NOISE, FROM REPEATS")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A repeat of a class "
              f"label has no variance in label units.")
        return {'skipped': 'not_regression', 'task_type': task_type}

    rep_groups = build_replicate_groups(data)
    if rep_groups is None:
        print(f"  SKIPPED: {dataset_name}'s loader keeps no repeat "
              f"measurements, so no molecule has a measurement count.")
        return {'skipped': 'no_replicate_structure', 'dataset': dataset_name}

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_split_half = int(kwargs.get('n_split_half_repeats', N_SPLIT_HALF_REPEATS))
    injected = [float(x) for x in
                (kwargs.get('injected_variances') or INJECTED_VARIANCES)]

    stats = measurement_stats(labels, rep_groups)
    probe = build_entities(labels, rep_groups, seed=0)
    if not np.array_equal(probe['n_measurements'], stats['n_measurements']):
        raise RuntimeError(
            "n26: measurement_stats and build_entities disagree about how many "
            "times each molecule was measured, so they are not in the same "
            "order.")
    if not np.array_equal(probe['row_index'], stats['row_index']):
        raise RuntimeError(
            "n26: measurement_stats and build_entities disagree about which "
            "row represents each molecule.")
    values = measurements_by_molecule(labels, rep_groups)
    if not np.array_equal([len(v) for v in values], stats['n_measurements']):
        raise RuntimeError(
            "n26: the per-molecule measurement lists are not in "
            "measurement_stats' order.")

    k = stats['n_measurements'].astype(np.float64)
    inv_k = 1.0 / k
    y = stats['mean_all'].astype(np.float64)
    n_ent = len(y)
    var_y = float(np.var(y, ddof=1))
    mean_inv_k = float(np.mean(inv_k))

    if data.get('groups') is None:
        print("  SKIPPED: the loader gives no fold groups, and random folds "
              "are not used in this suite.")
        return {'skipped': 'no_fold_groups', 'dataset': dataset_name}

    # ── Models: the dataset's two decided models (R1, DECISIONS.md Step 1) ──
    # 2026-10-05 (chat 5): config['models'] (the five-model DOWNSTREAM_MODELS
    # list) and a `models` kwarg are no longer read. knn is never a
    # downstream model here; it runs only on the distance search.
    models_decided = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
    model_factories = get_model_factories(
        models_decided, task_type=task_type, n_samples=n_ent)
    downstream_models = [m for m in models_decided if m in model_factories]
    models_not_built = {
        m: ('get_model_factories returned no factory (over its row limit at '
            f'{n_ent} molecules, or not defined)')
        for m in models_decided if m not in model_factories}
    if not downstream_models:
        raise RuntimeError(
            f"n26: no decided model could be built. Decided "
            f"{models_decided}, got {sorted(model_factories)}.")

    # ── One row per molecule, and the outer folds on those rows ──
    ent_data = molecule_rows_data(data, stats, dataset_name)
    ent_groups = ent_data['groups']
    same_rows = (n_ent == len(labels)
                 and np.array_equal(stats['row_index'], np.arange(n_ent)))
    search_suffix = None if same_rows else SEARCH_CONDITION_SUFFIX
    folds, n_folds = noise_outer_folds(ent_data, config)
    print(f"  {n_folds} outer folds on {n_ent} molecule rows; running "
          f"{[f for f, _, _ in folds]}")

    direct = direct_noise_variance(values)
    counts = {int(c): int((stats['n_measurements'] >= c).sum())
              for c in (2, 3, 4, 5)}
    print(f"  {len(labels)} measurements over {n_ent} molecules")
    for c, v in counts.items():
        print(f"    molecules measured {c}+ times: {v} of {n_ent}")
    print(f"  molecules whose repeats are all identical: "
          f"{direct['molecules_whose_repeats_are_all_identical']} of "
          f"{direct['molecules_used']} measured twice or more")
    print(f"  variance of the mean label: {var_y:.4f} label units squared; "
          f"mean of 1/count: {mean_inv_k:.4f}")

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'noise_regime': 'real_replicate_disagreement',
        'noise_injection': 'none in the main measurement',
        'training_label': 'the mean of all of a molecule\'s measurements',
        'residual_units': 'label units; squared residuals in label units squared',
        'models': downstream_models,
        'models_decided': models_decided,
        'models_not_built': models_not_built,
        'spaces': list(SPACES),
        'n_folds': n_folds,
        'folds_run': [int(f) for f, _, _ in folds],
        'selection_scope': 'fold_train_rows',
        'selection_note': (
            'Per outer fold and per decided model: the 2 best base '
            'representations and the 2 best hybrids, ranked by one search on '
            "that fold's training molecules (decided grid, pruning off). "
            "Hybrids are rebuilt on the fold's training molecules. Nothing is "
            'read from a whole-dataset search.'),
        'search_rows': 'one row per molecule (its first measurement row)',
        'search_labels': ("each training molecule's mean over all of its "
                          'measurements'),
        'search_condition_suffix': search_suffix,
        'search_shared_with': ('n19_measurement_count (same rows, labels, '
                               'folds and models)'),
        'per_fold_configurations': {},
        'feasibility': {
            'n_measurements': int(len(labels)),
            'n_molecules': n_ent,
            'molecules_by_min_measurements': counts,
        },
        'variance_of_the_mean_label': var_y,
        'mean_of_one_over_count': mean_inv_k,
        'direct': {
            **direct,
            'reliability': reliability_from(direct['s2e'], mean_inv_k, var_y),
            'note': ('s2e measured from the repeats alone, with no model. '
                     'Repeats that are all identical push it toward zero.'),
        },
        'split_half': split_half(values, n_split_half, seed=0),
        'icc': one_way_icc(values),
        'shared_bias_note': (
            'A bias shared by every measurement of a molecule does not shrink '
            'with the count. The 1/count regression puts it in the intercept, '
            'not the slope, so each fitted s2e is only the part of the noise '
            'that differs between records.'),
        'features_note': (
            "Each configuration is chosen per outer fold on that fold's "
            'training molecules. base_1 and base_2 can hold a different '
            'representation on each fold, and the out-of-fold prediction of '
            'a space joins the folds; per_fold_configurations names what each '
            'fold used.'),
        'per_space': {},
    }
    print(f"  spaces {list(SPACES)} for models {downstream_models}")

    # The estimator check (below) refits the first decided model in its
    # base_1 configuration on each fold. Its perturbed labels are drawn once
    # over every molecule, before any fold, in the order the injected
    # variances are listed, as before 2026-10-05.
    # DECIDED 2026-10-06 (Adelaide delegated): the check runs for the first
    # decided model only, with one draw per injected variance.
    check_model = downstream_models[0]
    rng = np.random.default_rng(12345)
    y_pert = {s2: y + rng.normal(0.0, np.sqrt(s2 * inv_k)) for s2 in injected}

    # oof[space][model] = one prediction per molecule
    oof = {s: {m: np.full(n_ent, np.nan) for m in downstream_models}
           for s in SPACES}
    fold_scores = {s: {m: {} for m in downstream_models} for s in SPACES}
    check_oof = {f'{s2:g}': np.full(n_ent, np.nan) for s2 in injected}
    check_rep_by_fold = {}
    errors = {}
    t0 = time.time()

    # ── Per-fold checkpoint (2026-09-25) ──
    # n26 changes no training data, so it writes no D-versus-D' rows; it
    # measures. The out-of-fold predictions are checkpointed after each fold
    # and a rerun skips the folds already done (PassCheckpoint, n23's shared
    # block). SELECTION_VERSION, the fold count, the folds run and the search
    # suffix are in the key since 2026-10-05.
    ckpt = PassCheckpoint(
        output_dir, 'n26_noise_variance_from_repeats', dataset_name,
        condition=kwargs.get('tag'), data=data, config=config,
        source_file=__file__,
        key={'models': list(downstream_models), 'spaces': list(SPACES),
             'n_splits': n_folds, 'folds_run': results['folds_run'],
             'selection': SELECTION_VERSION,
             'search_condition_suffix': search_suffix,
             'injected_variances': list(injected)})
    done_folds = set()
    _state = ckpt.load()
    if _state:
        oof, fold_scores, errors = (_state['oof'], _state['fold_scores'],
                                    _state['errors'])
        check_oof = _state['check_oof']
        check_rep_by_fold = _state['check_rep_by_fold']
        results['per_fold_configurations'].update(
            _state.get('per_fold_configurations') or {})
        done_folds = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_folds)} folds already "
              f"done")
    _metric = (metric_name, metric_fn, lower_is_better)

    for fold_i, tr, va in folds:
        fold_key = f'f{fold_i}'
        if fold_key in done_folds:
            print(f"    fold {fold_i}: on disk in the checkpoint")
            continue

        # The 2 base + 2 hybrids per decided model, from one search on this
        # fold's training molecules (handoff 2.4 step 5).
        sel = fold_standard_selection(
            ent_data['reps'], ent_data['labels'], ent_groups,
            train_idx=tr, fold_i=int(fold_i), n_outer_folds=n_folds,
            dataset_name=dataset_name, models=list(downstream_models),
            augmentations=ent_data['augmentations'], task_type=task_type,
            condition_suffix=search_suffix)
        entries = {m: _entry_from_selection(sel.get(m))
                   for m in downstream_models}
        results['per_fold_configurations'][str(fold_i)] = _fold_record(
            entries)

        for mname in downstream_models:
            factory = model_factories[mname]
            # Standardised inputs, 0/1 features unscaled; hybrids rebuilt on
            # this fold's training molecules with the fold search's own
            # representation scores (dirty-run bug 1).
            ins, unav, names = fold_model_inputs(
                ent_data, dataset_name, mname, entries[mname],
                train_idx=tr, val_idx=va, y_train=y[tr], factory=factory,
                metric=_metric)
            for space in SPACES:
                if space not in ins:
                    errors[f'{space}|{mname}|{fold_key}'] = unav.get(
                        space, 'configuration not built on this fold')
                    continue
                X_tr_s, X_va_s = ins[space]
                try:
                    pred = _fit_predict(factory, X_tr_s, y[tr], X_va_s)
                except Exception as exc:
                    errors[f'{space}|{mname}|{fold_key}'] = (
                        f'{type(exc).__name__}: {exc}')
                    continue
                oof[space][mname][va] = pred
                fold_scores[space][mname][fold_key] = float(
                    metric_fn(y[va], pred))

            # The estimator check: the same model and base_1 inputs, fitted
            # on the perturbed mean labels.
            if mname == check_model:
                check_rep_by_fold[fold_key] = names.get('base_1')
                if 'base_1' in ins:
                    X_tr_s, X_va_s = ins['base_1']
                    for s2 in injected:
                        try:
                            check_oof[f'{s2:g}'][va] = _fit_predict(
                                factory, X_tr_s, y_pert[s2][tr], X_va_s)
                        except Exception as exc:
                            errors[f'estimator_check|{s2:g}|{fold_key}'] = (
                                f'{type(exc).__name__}: {exc}')

        done_folds.add(fold_key)
        ckpt.save({'oof': oof, 'fold_scores': fold_scores, 'errors': errors,
                   'check_oof': check_oof,
                   'check_rep_by_fold': check_rep_by_fold,
                   'per_fold_configurations':
                       results['per_fold_configurations'],
                   'done': sorted(done_folds)})
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    # ── The three measurements, per (space, model) ──
    # Over the molecules that have a prediction: every molecule when every
    # fold ran, the held-out molecules of the folds run otherwise.
    for space in SPACES:
        srow = {}
        for mname in downstream_models:
            p = oof[space][mname]
            have = np.isfinite(p)
            if not have.any():
                continue
            res = y[have] - p[have]
            r2 = res ** 2
            kk, ik = k[have], inv_k[have]
            fits = {'ols': fit_ols(r2, ik),
                    'binned': fit_binned(r2, kk),
                    'gamma': fit_gamma(r2, ik)}
            var_here = float(np.var(y[have]))
            r2_model = (float(1.0 - np.mean(r2) / var_here) if var_here > 0
                        else float('nan'))
            rel_fit = {f: reliability_from(fits[f]['slope_s2e'], mean_inv_k, var_y)
                       for f in fits}
            srow[mname] = {
                'molecules_with_a_prediction': int(have.sum()),
                'fold_scores': fold_scores[space][mname],
                # The held-out score per fold in the dataset's own metric
                # (`metric`, `lower_is_better`), mean and spread over folds,
                # or "one fold, no spread" (2026-10-05).
                'held_out_score_over_folds': fold_spread(
                    list(fold_scores[space][mname].values())),
                'model_r2_against_the_mean_label': r2_model,
                # ADDED 2026-09-25 (third research report, A6): the model's R2
                # divided by the reliability of the label it is scored on, so
                # datasets with noisier labels are not read as worse models.
                # One per reliability estimate; the reliability is on the
                # mean label, which is what the model is scored against.
                'r2_over_reliability': {
                    'direct': _ratio(r2_model, results['direct']['reliability']
                                     ['reliability']),
                    **{f: _ratio(r2_model, rel_fit[f]['reliability'])
                       for f in rel_fit}},
                'mean_squared_residual': float(np.mean(r2)),
                'error_by_count': error_by_count(res, kk),
                'fits': fits,
                'reliability_by_fit': rel_fit,
            }
        results['per_space'][space] = srow

    # How far each fit's slope moves between feature spaces, per model. The
    # noise variance is a property of the labels, so a large spread says the
    # slope is picking up model error.
    slope_spread = {}
    for mname in downstream_models:
        for f in ('ols', 'binned', 'gamma'):
            vals = [results['per_space'][s][mname]['fits'][f]['slope_s2e']
                    for s in SPACES if mname in results['per_space'].get(s, {})]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                slope_spread.setdefault(mname, {})[f] = {
                    'spaces': len(vals), 'min': float(min(vals)),
                    'max': float(max(vals))}
    results['slope_across_spaces'] = slope_spread

    # ── The estimator check ──
    base_oof = oof['base_1'][check_model]
    check = {'model': check_model, 'space': 'base_1',
             'rep_by_fold': check_rep_by_fold,
             'what_it_is': ('a check of the 1/count regression, not a '
                            'measurement of the real noise'),
             'per_injected_variance': {}}
    have = np.isfinite(base_oof)
    if have.any():
        check['molecules'] = int(have.sum())
        s0 = fit_ols((y[have] - base_oof[have]) ** 2, inv_k[have])['slope_s2e']
        check['slope_without_injection'] = s0
        for s2 in injected:
            p = check_oof[f'{s2:g}']
            both = have & np.isfinite(p)
            if not np.array_equal(both, have):
                errors[f'estimator_check|{s2:g}'] = (
                    'the refit has no prediction for some molecules the '
                    'unperturbed fit predicted; not compared')
                continue
            s1 = fit_ols((y_pert[s2][have] - p[have]) ** 2,
                         inv_k[have])['slope_s2e']
            check['per_injected_variance'][f'{s2:g}'] = {
                'injected_s2': s2,
                'slope_with_injection': s1,
                'slope_change': float(s1 - s0),
            }
    else:
        check['failed'] = (f'{check_model} has no out-of-fold prediction in '
                           f'base_1 on any fold run')
    results['estimator_check'] = check

    if errors:
        results['_errors'] = errors
    results = convert_numpy(results)
    _print_summary(results)
    save_experiment_results(results, output_dir,
                            'n26_noise_variance_from_repeats', dataset_name,
                            filename_suffix=kwargs.get('tag'))
    ckpt.clear()
    return results


def _print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY (label units squared unless stated)")
    d = results['direct']
    rel = d['reliability']
    print(f"  s2e measured directly from repeats: {d['s2e']:.4f}, over "
          f"{d['molecules_used']} molecules measured twice or more")
    print(f"    reliability {rel['reliability']:.4f}, R2 threshold "
          f"{rel['r2_threshold']:.4f}")
    sh = results['split_half']
    rs = [r['r_one_measurement'] for r in sh['repeats']
          if np.isfinite(r['r_one_measurement'])]
    if rs:
        print(f"  split-half correlation of one measurement, {len(rs)} draws "
              f"over {sh['molecules']} molecules: {min(rs):.4f} to "
              f"{max(rs):.4f}")
    icc = results['icc']
    print(f"  one-way ICC over {icc['molecules']} molecules: {icc['icc']:.4f}")
    for space, srow in results['per_space'].items():
        for m, row in srow.items():
            f = row['fits']
            print(f"  {space}/{m}: out-of-fold R2 "
                  f"{row['model_r2_against_the_mean_label']:.4f}; slope s2e "
                  f"ols {f['ols']['slope_s2e']:.4f}, binned "
                  f"{f['binned']['slope_s2e']:.4f}, gamma "
                  f"{f['gamma']['slope_s2e']:.4f}; intercept s2u ols "
                  f"{f['ols']['intercept_s2u']:.4f}")
    chk = results['estimator_check']
    for key, row in chk.get('per_injected_variance', {}).items():
        print(f"  estimator check ({chk['model']}, base_1 of each fold): "
              f"injected {key}, "
              f"slope moved {row['slope_change']:+.4f}")


def main():
    parser = create_experiment_parser(
        'N26: label-noise variance, reliability and the R2 threshold, from '
        'repeat measurements')
    parser.add_argument('--n-split-half-repeats', type=int,
                        default=N_SPLIT_HALF_REPEATS,
                        help='How many times two measurements are drawn per '
                             'molecule for the split-half correlation')
    parser.add_argument('--injected-variances', type=str, default=None,
                        help='Comma-separated noise variances, label units '
                             'squared, for the estimator check (default '
                             f'{",".join(str(v) for v in INJECTED_VARIANCES)})')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {'n_split_half_repeats': args.n_split_half_repeats}
    if args.injected_variances:
        kw['injected_variances'] = [
            float(x) for x in args.injected_variances.split(',')]
    if args.tag:
        kw['tag'] = args.tag
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
