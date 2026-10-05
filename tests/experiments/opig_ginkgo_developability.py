#!/usr/bin/env python3
"""
Ginkgo Antibody Developability Experiment (OPIG B5)

Replicates van Niekerk et al. 2026 "2025 Ginkgo Datapoints Antibody
Developability Competition" — evaluates KIRBy reps across the configured
biophysical properties using the competition's pre-computed hierarchical
cluster splits.

Scientific questions:
  1. Which antibody reps best predict each developability property?
  2. Does a hybrid help on these small (n=94-242) datasets?
  3. Per property and per downstream model, how does each of the 2 hybrids
     compare with each of the 2 base representations on held-out folds?
  4. Which properties are predictable and which are noise?

Properties (GDPa1 v1.2, 242 antibodies):
  HIC (retention), Tm2 (thermal stability), Titer (expression),
  PR_CHO (protein recovery), AC-SINS (self-interaction),
  Purity, SEC %Monomer (aggregation), SMAC (colloidal), HAC (heparin)

Paper baselines (van Niekerk et al. 2026):
  Competition had 100+ team submissions. Best reported Spearman rho:
    HIC ~0.65-0.70, Tm2 ~0.30-0.45, Titer ~0.35-0.50,
    AC-SINS ~0.55-0.65, Heparin ~0.60-0.75 (varies by property)
  Pre-computed 5-fold hierarchical cluster splits (IgG-stratified).

What one held-out fold computes (re-aligned 2026-10-05, handoff 2.5 group C):
  The folds are the competition's own cluster folds (the dataset's
  fold_assignments), not nested_cv_outer_folds. For each fold, one general
  search runs on that fold's training rows on the property's decided grid
  (DECISIONS.md), pruning off, no augmentation. For each of the property's two
  decided downstream models it gives the 2 best base representations and the 2
  best hybrids (a second hybrid whose selected features equal the first's is
  skipped), all ranked on the training rows. Each of the four is fitted on the
  training rows and scored on the held-out fold. paratope and structural are
  not in the search pool (handoff 2.3 answer 12) and are scored alone. No
  distance hybrid and no similarity hybrid (answer 1).

Usage:
    # Smoke test (1 property, 2 of the competition folds)
    python tests/experiments/opig_ginkgo_developability.py \\
        --properties ginkgo_hic --n-folds 2

    # Full evaluation (every configured property, every competition fold)
    python tests/experiments/opig_ginkgo_developability.py

Reference:
    van Niekerk et al. (2026). mAbs.
    https://github.com/ginkgobioworks/abdev-benchmark
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from run_baseline_survey import _filter_high_dim_per_fold
from experiments._shared import _resolve_dataset_metric
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories,
    fold_standard_selection, fold_searches,
    STANDARD_N_BASE, STANDARD_N_HYBRIDS,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from experiments.decided_grid import decided_models, decided_search
from kirby.scaling import FeatureScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

# The standard selection (handoff 2.1 R4; 2.5 group C, Adelaide 2026-10-05):
# per decided downstream model, per held-out fold, the 2 best base
# representations and the 2 best hybrids, both ranked on that fold's training
# rows. This replaced, on 2026-10-05, the owner's set of 2026-10-02 (top 2
# general + top 1 distance hybrid per model) and the one best single
# representation re-picked from the other folds' held-out scores.
N_BASE = STANDARD_N_BASE
N_HYBRIDS = STANDARD_N_HYBRIDS

# Arms per model per fold, in the order they are reported.
ARMS = tuple([f'base{i}' for i in range(1, N_BASE + 1)]
             + [f'hybrid{i}' for i in range(1, N_HYBRIDS + 1)])

# Every representation in the pool is ALSO fitted alone on every held-out fold
# with each decided model, as a descriptive table (question 1 above). This is
# how paratope and structural are scored alone (answer 12), and it is the table
# this file always produced. It selects nothing: no number in it chooses an
# arm. Open question for Adelaide (2026-10-05): keep it, or score only
# paratope, structural and the 2 selected base representations. Turning it off
# cuts the single-representation fits per fold from 12 per model to at most 4.
SCORE_EVERY_REPRESENTATION = True

# run_baseline_survey's cut: a matrix wider than 500 features is reduced to 200
# by a LightGBM importance filter fitted on the training rows. Until 2026-10-05
# both arms went through it, so a hybrid the search ranked at 1,000 or 2,000
# features was fitted on 200 of them. Off now, so the held-out fit uses the
# features the search ranked. Open question for Adelaide: True puts the cut
# back on both arms (base representations and hybrids alike).
APPLY_BASELINE_SURVEY_FEATURE_CUT = False

# Resume identity of <output>.partial.jsonl. Bumped 2026-10-05: the units are
# now one competition fold each, in the 2 base + 2 hybrid layout, so a sidecar
# written under the 3-hybrid, one-unit-per-property layout is moved aside.
CHECKPOINT_LAYOUT = 'fold_unit_2base_2hybrid_2026-10-05'

# 5 of the competition's 9 properties. The other 4 were cut, each with a reason
# on record:
#   ginkgo_tm2, ginkgo_aggregation, ginkgo_purity  cut 2026-08-20 as near-dead
#     targets — best single representation reaches spearman +0.092, +0.179 and
#     +0.151, with most representations scoring negative (DATASET_STATUS.md).
#   ginkgo_heparin  cut 2026-07-09 at n=94, too small
#     (DATASET_EXPANSION_2026_07_09.md, via antibody_baseline_survey.py:133).
ALL_PROPERTIES = [
    'ginkgo_hic', 'ginkgo_titer', 'ginkgo_pr_cho',
    'ginkgo_acsins', 'ginkgo_colloidal',
]

# The two augmentation representations of every ginkgo pool in DATASET_REPS.
# Kept out of the search pool and scored alone (answer 12, 2026-10-05). Step 8
# keeps an augmentation condition only on the distance search of
# ginkgo_colloidal and ginkgo_hic (answer 13), and this experiment runs no
# distance search, so no search here runs with augmentation.
AUG_REPS = ('paratope', 'structural')


# Representations come from DATASET_REPS, resolved PER PROPERTY — every ginkgo
# property has its own approved pool and they differ, so one list cannot serve
# all of them: the 10 general representations plus paratope and structural.
#
# A property with no DATASET_REPS entry is skipped and recorded in '_skipped'
# rather than run against a hand-written list.
def _default_reps(prop_name):
    from experiments._shared import approved_rep_pool
    return approved_rep_pool(prop_name)


def _downstream_models(prop_name):
    """The two downstream models decided for this property (DECISIONS.md Step 1).

    This replaced MODELS_LIST, which was five names — rf, xgboost, svm, dnn,
    lightgbm — for every property. Three of the five properties have 'gp' as
    one of their two.

    knn is not a downstream model here: Step 9 runs it on the distance search
    only, and this experiment runs no distance search (answer 1, 2026-10-05).
    """
    return [m for m in decided_models(prop_name, 'general') if m != 'knn']


# Competition baselines (van Niekerk et al. 2026, 113 teams)
# test_top = best team Spearman rho on held-out test set (80 antibodies)
# cv_top = best team cross-validation Spearman rho (246 training antibodies)
# Note: competition only evaluated 5 of 9 properties
PAPER_BASELINES = {
    'ginkgo_hic':         {'test_top': 0.708, 'cv_top': 0.758},
    'ginkgo_tm2':         {'test_top': 0.392, 'cv_top': 0.140},
    'ginkgo_titer':       {'test_top': 0.310, 'cv_top': 0.640},
    'ginkgo_pr_cho':      {'test_top': 0.337, 'cv_top': 0.559},
    'ginkgo_acsins':      {'test_top': 0.356, 'cv_top': 0.653},
    # These 4 were NOT in the competition — no published baselines
    'ginkgo_purity':      {},
    'ginkgo_aggregation': {},
    'ginkgo_colloidal':   {},
    'ginkgo_heparin':     {},
}

# =============================================================================
# FOLDS
# =============================================================================

def _competition_folds(fold_ids, n_folds):
    """The competition's cluster folds as held-out splits.

    fold_ids: one competition fold ID per antibody (the dataset's
        fold_assignments). The number of distinct IDs is the competition's fold
        count; it is the one setting the outer fold count comes from
        (2026-10-05, handoff 2.4 step 14).
    n_folds: how many of those folds to hold out, in ID order. None = every
        one. 1 = a single held-out split: the first competition fold held out,
        trained on all the others. The training rows of a fold are the same
        whichever count is asked for, so its search is the same too.

    Returns ([(fold_id, train_idx, val_idx), ...], n_competition_folds).
    """
    fa = np.asarray(fold_ids)
    unique_folds = sorted(set(fa.tolist()))
    n_comp = len(unique_folds)
    if n_comp < 2:
        raise ValueError(f'the competition split has {n_comp} fold ID(s); '
                         f'a held-out fold needs at least 2')
    n_used = n_comp if n_folds is None else int(n_folds)
    if n_used < 1:
        raise ValueError(f'--n-folds must be at least 1, got {n_folds}')
    if n_used > n_comp:
        print(f"  WARNING: requested {n_used} folds but the competition split "
              f"has {n_comp}; using {n_comp}")
        n_used = n_comp
    splits = []
    for fold_id in unique_folds[:n_used]:
        splits.append((fold_id, np.where(fa != fold_id)[0],
                       np.where(fa == fold_id)[0]))
    return splits, n_comp


def _fold_namespace(n_comp, fold_id):
    """Result-store namespace of one competition fold's search.

    Names the competition split ('ginkgocv') so a search on these training rows
    can never be read as, or overwrite, a nested_cv_outer_folds fold, whose
    namespace is 'ocv{n}f{i}s{seed}'. Asked for 2026-10-05; it replaces
    condition_suffix 'ginkgo_hold_fold{i}', which rode on the tier name.
    """
    return f'ginkgocv{n_comp}f{fold_id}'


# =============================================================================
# EVALUATION
# =============================================================================

def _recall_at_k(y_true, y_pred, frac=0.1):
    """Top-K% recall: overlap between true top-K and predicted top-K.

    Matches ginkgobioworks/abdev-benchmark evaluation_metrics.recall_at_k.
    """
    top_k = max(1, int(len(y_true) * frac))
    true_top = set(np.argsort(y_true)[-top_k:].tolist())
    pred_top = set(np.argsort(y_pred)[-top_k:].tolist())
    return len(true_top & pred_top) / top_k


# Properties where LOWER values are better (sign-flip for top-K recall).
# From abdev-benchmark constants.py ASSAY_HIGHER_IS_BETTER:
#   HIC=False, PR_CHO=False, AC-SINS=False (i.e. lower is better for these)
LOWER_IS_BETTER = {'ginkgo_hic', 'ginkgo_pr_cho', 'ginkgo_acsins'}


def _held_out_score(X_tr, X_va, y_tr, y_va, factory, train_groups,
                    prop_name, metric_fn):
    """Fit one downstream model on the training rows, score the held-out fold.

    The same routine for every arm — base representations, hybrids, and the
    descriptive single-representation table — so the arms differ only in their
    features. Until 2026-10-05 the base side went through
    run_baseline_survey.evaluate_rep and the hybrid side through a copy of it
    here, with StandardScaler behind an all-0/1 check on the whole matrix, and
    evaluate_rep left out a fold whose fit raised, which put the base and
    hybrid per-fold lists out of step.

    Features are scaled with FeatureScaler (answer 17): 0/1 features as they
    are, the rest standardised, fitted on the training rows. Missing values get
    the training median (find_best_hybrid._impute_nan). The model is fitted
    with the fold's training groups, which the neural models use for a
    group-aware early-stopping holdout, as the engine's held-out score does.
    Representations stay float32 (answer 18).

    Returns (scores, None) or (None, reason). scores holds the property's own
    metric under 'primary' and r2, rmse, spearman and top-10% recall beside it.
    """
    from kirby.find_best_hybrid import _impute_nan, _fit_model
    try:
        if APPLY_BASELINE_SURVEY_FEATURE_CUT:
            X_tr, X_va, _ = _filter_high_dim_per_fold(
                X_tr, X_va, y_tr, 'regression')
        scaler = FeatureScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_tr, X_va, _ = _impute_nan(X_tr, X_va)
        m = factory()
        if m is None:
            return None, 'model unavailable at this training-row count'
        _fit_model(m, X_tr, y_tr, groups=train_groups)
        preds = np.asarray(m.predict(X_va)).ravel()
    except Exception as e:
        return None, f'fit raised {type(e).__name__}: {e}'

    def _finite(v):
        return float(v) if v is not None and np.isfinite(v) else float('nan')

    out = {}
    try:
        out['primary'] = (_finite(metric_fn(y_va, preds))
                          if len(y_va) > 2 else float('nan'))
    except Exception:
        out['primary'] = float('nan')
    ss_res = float(np.sum((y_va - preds) ** 2))
    ss_tot = float(np.sum((y_va - y_va.mean()) ** 2))
    out['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    out['rmse'] = float(np.sqrt(np.mean((y_va - preds) ** 2)))
    try:
        from scipy.stats import spearmanr
        out['spearman'] = (_finite(spearmanr(y_va, preds)[0])
                           if len(y_va) > 2 else float('nan'))
    except Exception:
        out['spearman'] = float('nan')
    flip = prop_name in LOWER_IS_BETTER
    out['top10_recall'] = _recall_at_k(-y_va if flip else y_va,
                                       -preds if flip else preds, frac=0.1)
    return out, None


def _search_failures(base_reps, labels, groups, train_idx, fold_pos, n_comp,
                     prop_name, metric_name, namespace):
    """The recorded failures of this fold's general search.

    fold_standard_selection prints a failed search and maps every model to
    None. The failure records sit on the search set itself; this reads them
    back through fold_searches with the same arguments, which the in-process
    memo answers without running the search again.
    """
    try:
        ss = fold_searches(
            base_reps, labels, groups, train_idx=train_idx, fold_i=fold_pos,
            n_outer_folds=n_comp, dataset_name=prop_name, augmentations=None,
            task_type='regression', metric=metric_name, tier='general',
            seed=42, cache_namespace=namespace)
        return [{k: str(v) for k, v in f.items()}
                for f in (getattr(ss, 'failures', None) or [])]
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {e}'}]


def _hybrid_record(entry, rank):
    """The parts of one top_hybrids entry the result file keeps (no blueprint)."""
    return {
        'rank': rank,
        'label': entry.get('label'),
        'allocation_method': entry.get('allocation_method'),
        'importance_method': entry.get('importance_method'),
        'budget': entry.get('budget'),
        'internal_evaluator': entry.get('internal_evaluator'),
        'rep_count': entry.get('rep_count'),
        'rep_subset': entry.get('rep_subset'),
        'aug_condition': entry.get('aug_condition', 'none'),
        'score_train_rows': entry.get('score_mean'),
        'train_rows_fold_scores': entry.get('fold_scores'),
        'signature_error': entry.get('signature_error'),
    }


def _run_fold(prop_name, fold_pos, fold_id, train_idx, val_idx, n_comp,
              base_reps, all_reps, labels, groups, models, metric_name,
              metric_fn, metric_lower):
    """Everything one competition fold computes. Returns a plain-data payload.

    base_reps: the search pool (the general representations; paratope and
    structural are out, answer 12). all_reps: every loaded representation, for
    the descriptive single-representation table.
    """
    namespace = _fold_namespace(n_comp, fold_id)
    y_tr, y_va = labels[train_idx], labels[val_idx]
    g_tr = groups[train_idx]
    payload = {
        'fold': int(fold_pos), 'fold_id': int(fold_id),
        'cache_namespace': namespace,
        'n_train': int(len(train_idx)), 'n_held_out': int(len(val_idx)),
        'metric': metric_name, 'lower_is_better': bool(metric_lower),
        'selection_scope': 'fold_train_rows',
        'models': {}, 'single_representations': {}, 'failures': [],
    }

    # One general search on this fold's training rows, on the property's
    # decided grid (fold_searches -> search_on_fold -> apply_decided_grid),
    # pruning off, filed under this competition fold's own namespace. It is
    # built here when not stored (R6).
    try:
        sel = fold_standard_selection(
            base_reps, labels, groups, train_idx=train_idx, fold_i=fold_pos,
            n_outer_folds=n_comp, dataset_name=prop_name,
            models=list(models), distance=False, augmentations=None,
            task_type='regression', metric=metric_name, seed=42,
            n_base=N_BASE, n_hybrids=N_HYBRIDS, cache_namespace=namespace,
            distinct_features=True)
    except Exception as e:
        sel = {m: None for m in models}
        payload['failures'].append({
            'dataset': prop_name, 'fold_id': int(fold_id), 'model': None,
            'stage': 'search',
            'error': f'{type(e).__name__}: {e}'})

    if any(v is None for v in sel.values()) and not payload['failures']:
        for f in _search_failures(base_reps, labels, groups, train_idx,
                                  fold_pos, n_comp, prop_name, metric_name,
                                  namespace) or [{'error': 'no record'}]:
            for m in models:
                if sel.get(m) is None:
                    payload['failures'].append(dict(
                        f, dataset=prop_name, fold_id=int(fold_id), model=m,
                        stage='search'))

    # Held-out scores of single representations, computed once per
    # (representation, model) on this fold and shared by the base arms and
    # the descriptive table.
    single_cache = {}

    def _single(rep, m, factory):
        if (rep, m) not in single_cache:
            single_cache[(rep, m)] = _held_out_score(
                all_reps[rep][train_idx], all_reps[rep][val_idx], y_tr, y_va,
                factory, g_tr, prop_name, metric_fn)
        return single_cache[(rep, m)]

    train_reps = {r: X[train_idx] for r, X in base_reps.items()}
    val_reps = {r: X[val_idx] for r, X in base_reps.items()}

    for m, factory in models.items():
        s = sel.get(m)
        rec = {'base': [], 'hybrids': [], 'missing': [],
               'hybrids_skipped_identical_features': []}
        payload['models'][m] = rec
        if s is None:
            rec['missing'].append('the general search on this fold failed; '
                                  'see failures')
            continue

        # The 2 best base representations for this model, ranked by their
        # score in this fold's search (training rows only).
        bd = s.get('base_detail') or {}
        for rank, e in enumerate(bd.get('selected', []), 1):
            sc, why = _single(e['rep'], m, factory)
            rec['base'].append({'rank': rank, 'rep': e['rep'],
                                'n_features': int(all_reps[e['rep']].shape[1]),
                                'score_train_rows': e.get('score'),
                                'held_out': sc, 'held_out_missing': why})
        if len(rec['base']) < N_BASE:
            rec['missing'].append(
                f"{len(rec['base'])} of {N_BASE} base representations: "
                f"{bd.get('shortfall_reason')}")

        # The 2 best hybrids for this model, ranked on the training rows, a
        # second one with the same selected features as the first skipped.
        # Each is rebuilt on the training rows with this model and the
        # property's metric, as the search scored it; the blueprint's own
        # representation scores are this fold's training-row scores, so no
        # whole-dataset score enters the rebuild (dirty-run bug 1).
        rec['hybrids_skipped_identical_features'] = [
            {k: v for k, v in d.items()}
            for d in s.get('hybrids_skipped_identical_features', [])]
        for rank, e in enumerate(s.get('hybrids', []), 1):
            hr = _hybrid_record(e, rank)
            bp = e.get('blueprint')
            try:
                trh, vah, _ = build_fold_hybrid(
                    train_reps, val_reps, y_tr, bp, groups=g_tr,
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=metric_lower))
                hr['n_features_built'] = int(trh.shape[1])
                sc, why = _held_out_score(trh, vah, y_tr, y_va, factory, g_tr,
                                          prop_name, metric_fn)
            except Exception as ex:
                sc, why = None, f'build raised {type(ex).__name__}: {ex}'
                payload['failures'].append({
                    'dataset': prop_name, 'fold_id': int(fold_id),
                    'model': m, 'stage': f'hybrid{rank} rebuild',
                    'error': why})
                print(f"    fold {fold_id} {m} hybrid{rank}: {why}")
            hr['held_out'] = sc
            hr['held_out_missing'] = why
            rec['hybrids'].append(hr)
        if len(rec['hybrids']) < N_HYBRIDS:
            rec['missing'].append(
                f"{len(rec['hybrids'])} of {N_HYBRIDS} hybrids with distinct "
                f"selected features in this fold's search")

    if SCORE_EVERY_REPRESENTATION:
        reps_alone = sorted(all_reps)
    else:
        reps_alone = sorted({r for r in all_reps if r in AUG_REPS}
                            | {b['rep'] for rec in payload['models'].values()
                               for b in rec['base']})
    for rep in reps_alone:
        row = {}
        for m, factory in models.items():
            sc, why = _single(rep, m, factory)
            row[m] = {'held_out': sc, 'held_out_missing': why}
        payload['single_representations'][rep] = {
            'n_features': int(all_reps[rep].shape[1]),
            'in_search_pool': rep in base_reps, 'models': row}
    return payload


def _spread(values, n_folds_used):
    """Mean and spread of per-fold values. No interval, no test."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    out = {'mean': float(v.mean()) if len(v) else float('nan'),
           'n_folds_scored': int(len(v))}
    if n_folds_used == 1:
        out['spread'] = 'one fold, no spread'
    elif len(v) < 2:
        out['spread'] = f'{len(v)} fold(s) scored, no spread'
    else:
        out['std'] = float(v.std())
    return out


def _summarise(fold_payloads, models, n_folds_used):
    """Per model and arm, the held-out primary metric per fold, in fold order.

    Every list has one entry per fold run (NaN where the arm was not scored),
    so arm k on fold i is always paired with arm j on fold i. Also the per-fold
    difference hybrid1 minus base1, both chosen on the same fold's training
    rows. Positive = the hybrid scored higher (the primary metric here is
    higher-is-better spearman on every property).
    """
    out = {}
    for m in models:
        per_arm = {a: [] for a in ARMS}
        for p in fold_payloads:
            rec = p['models'].get(m) or {}
            for i in range(N_BASE):
                b = rec.get('base', [])
                v = (b[i]['held_out'] or {}).get('primary') if len(b) > i else None
                per_arm[f'base{i + 1}'].append(float('nan') if v is None else v)
            for i in range(N_HYBRIDS):
                h = rec.get('hybrids', [])
                v = (h[i]['held_out'] or {}).get('primary') if len(h) > i else None
                per_arm[f'hybrid{i + 1}'].append(float('nan') if v is None else v)
        diff = [h - b for h, b in zip(per_arm['hybrid1'], per_arm['base1'])]
        out[m] = {
            'fold_ids': [p['fold_id'] for p in fold_payloads],
            'per_fold': per_arm,
            'by_arm': {a: _spread(per_arm[a], n_folds_used) for a in ARMS},
            'hybrid1_minus_base1_per_fold': diff,
            'hybrid1_minus_base1': _spread(diff, n_folds_used),
        }
    return out


def _single_rep_table(fold_payloads, models, n_folds_used):
    """Descriptive: each representation's held-out primary metric per fold."""
    reps = sorted({r for p in fold_payloads
                   for r in p['single_representations']})
    out = {}
    for r in reps:
        out[r] = {'models': {}}
        for m in models:
            vals = []
            for p in fold_payloads:
                cell = (p['single_representations'].get(r, {})
                        .get('models', {}).get(m) or {})
                v = (cell.get('held_out') or {}).get('primary')
                vals.append(float('nan') if v is None else v)
            out[r]['models'][m] = dict(_spread(vals, n_folds_used),
                                       per_fold=vals)
        first = next(p['single_representations'][r] for p in fold_payloads
                     if r in p['single_representations'])
        out[r]['n_features'] = first['n_features']
        out[r]['in_search_pool'] = first['in_search_pool']
    return out


def evaluate_property(prop_name, rep_names, n_folds, ck=None, on_fold=None):
    """Every competition fold of one property. Returns the property's result.

    ck: checkpoint; each finished fold is one unit (property, rep_names,
    fold_id), so a crash loses at most the fold in progress (dirty-run bug 12).
    on_fold(partial_result): called after every fold so the caller can write
    the JSON then and not only at the end.
    """
    print(f"\n  --- {prop_name} ---")

    data = load_dataset(prop_name)
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    labels = np.asarray(data['labels'])
    n = len(labels)
    if data.get('fold_assignments') is None:
        # The competition split is the design of this experiment. Without it
        # load_dataset falls back to identity groups, which are not the
        # competition's folds, so the property is not run.
        return {'error': 'no competition fold_assignments for this property; '
                         'the experiment uses the competition split only'}
    # For a pre_split dataset the groups ARE the competition fold IDs
    # (antibody_baseline_survey._assign_groups). _smoke_subsample subsets
    # 'groups' but not 'fold_assignments', so the IDs are read off 'groups'.
    groups = np.asarray(data['groups'])
    print(f"  {n} samples, target={data.get('property_name', prop_name)}")

    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=data['heavy_seqs'],
        light_seqs=data['light_seqs'],
        dataset_tag=prop_name,
    )
    if not reps:
        return {'error': 'no representation could be loaded or computed'}

    # The search pool: every loaded representation except paratope and
    # structural (answer 12). Those two are scored alone only.
    base_reps = {k: v for k, v in reps.items() if k not in AUG_REPS}
    if not base_reps:
        return {'error': 'no representation left for the search after '
                         f'removing {list(AUG_REPS)}'}

    fold_splits, n_comp = _competition_folds(groups, n_folds)
    n_used = len(fold_splits)
    print(f"  competition split: {n_comp} cluster folds; holding out "
          f"{n_used} ({[f for f, _, _ in fold_splits]})")

    model_names = _downstream_models(prop_name)
    models = get_model_factories(model_names, 'regression', n_samples=n)
    print(f"  downstream models: {model_names}")
    models_dropped = {m: 'no factory at this sample count'
                      for m in model_names if m not in models}

    from kirby.find_best_hybrid import BUILTIN_METRICS
    metric_name, _task, metric_lower = _resolve_dataset_metric(prop_name)
    metric_fn = BUILTIN_METRICS[metric_name][0]

    def _assemble(fold_payloads):
        return {
            'n_samples': n,
            'property': data.get('property_name', prop_name),
            'metric': metric_name,
            'lower_is_better': bool(metric_lower),
            'fold_source': ('competition hierarchical cluster split '
                            '(fold_assignments)'),
            'n_competition_folds': n_comp,
            'n_folds_used': n_used,
            'fold_ids_used': [int(f) for f, _, _ in fold_splits],
            'n_folds_finished': len(fold_payloads),
            'downstream_models': model_names,
            'downstream_models_dropped': models_dropped,
            'search_pool': sorted(base_reps),
            'scored_alone_only': sorted(k for k in reps if k in AUG_REPS),
            'selection': (f'{N_BASE} base representations + {N_HYBRIDS} '
                          f'hybrids per model per fold, ranked on the fold\'s '
                          f'training rows; no distance hybrid'),
            'selection_scope': 'fold_train_rows',
            # What every search was allowed to use, straight from DECISIONS.md.
            'decided_grid': {'general': decided_search(prop_name, 'general')},
            'by_model': _summarise(fold_payloads, list(models), n_used),
            'single_representations': _single_rep_table(
                fold_payloads, list(models), n_used),
            'folds': fold_payloads,
            'failures': [f for p in fold_payloads for f in p['failures']],
        }

    fold_payloads = []
    for fold_pos, (fold_id, tr, va) in enumerate(fold_splits):
        key = dict(property=prop_name, rep_names=list(rep_names),
                   fold_id=int(fold_id))
        if ck is not None and ck.done(**key):
            p = ck.get(**key)
            print(f"  fold {fold_id}: finished in an earlier attempt, read back")
        else:
            print(f"  fold {fold_id} ({fold_pos + 1}/{n_used}): "
                  f"{len(tr)} training, {len(va)} held out")
            p = _run_fold(prop_name, fold_pos, fold_id, tr, va, n_comp,
                          base_reps, reps, labels, groups, models,
                          metric_name, metric_fn, metric_lower)
            # A fold with a failed search or rebuild is written to the result
            # file but not to the checkpoint, so the next attempt retries it.
            if ck is not None and not p['failures']:
                ck.put(p, **key)
        for m, rec in p['models'].items():
            for b in rec['base']:
                v = (b['held_out'] or {}).get('primary')
                print(f"    {m:8s} base{b['rank']} {b['rep']:22s} "
                      f"train={b['score_train_rows']!s:.6s}  held-out="
                      f"{'-' if v is None else format(v, '.4f')}")
            for h in rec['hybrids']:
                v = (h['held_out'] or {}).get('primary')
                print(f"    {m:8s} hybrid{h['rank']} {h['label']!s:22.22s} "
                      f"train={h['score_train_rows']!s:.6s}  held-out="
                      f"{'-' if v is None else format(v, '.4f')}")
            for why in rec['missing']:
                print(f"    {m:8s} missing: {why}")
        fold_payloads.append(p)
        if on_fold is not None:
            on_fold(_assemble(fold_payloads))

    return _assemble(fold_payloads)


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(all_results):
    """Per property and downstream model: each arm's mean held-out metric.

    Replaced 2026-10-05. The old table took a maximum over models on the hybrid
    side against a maximum over every representation and model on the other,
    both read off the held-out folds, and its second table re-picked one best
    single representation from the other folds' held-out scores. Here every
    arm was chosen on the fold's training rows before the held-out fold was
    scored. No averaging across properties.
    """
    print(f"\n{'=' * 100}")
    print("GINKGO DEVELOPABILITY: 2 BASE + 2 HYBRIDS PER MODEL, HELD-OUT "
          "COMPETITION FOLDS (OPIG B5)")
    print(f"{'=' * 100}")
    print("  One row per property and downstream model. Each cell is the mean "
          "held-out metric over the folds that arm scored; base1/base2 and "
          "hybrid1/hybrid2 were ranked on each fold's training rows. "
          "'h1-b1' is the mean per-fold difference hybrid1 minus base1.")
    print(f"\n  {'Property':18s} {'Comp.test':>9s} {'Comp.CV':>8s} "
          f"{'Model':>8s} " + ' '.join(f'{a:>8s}' for a in ARMS)
          + f" {'h1-b1':>8s} {'spread':>20s}")
    for prop_name, ds in all_results.items():
        if prop_name.startswith('_') or not isinstance(ds, dict) \
                or 'by_model' not in ds:
            continue
        bl = PAPER_BASELINES.get(prop_name, {})
        t = f"{bl['test_top']:.3f}" if 'test_top' in bl else '-'
        c = f"{bl['cv_top']:.3f}" if 'cv_top' in bl else '-'
        for m, s in ds['by_model'].items():
            cells = []
            for a in ARMS:
                v = s['by_arm'][a]['mean']
                cells.append(f'{v:8.4f}' if np.isfinite(v) else f"{'-':>8s}")
            d = s['hybrid1_minus_base1']
            ds_ = f"{d['mean']:+8.4f}" if np.isfinite(d['mean']) else f"{'-':>8s}"
            sp = (d['spread'] if 'spread' in d else f"std {d['std']:.4f}")
            print(f"  {prop_name:18s} {t:>9s} {c:>8s} {m:>8s} "
                  + ' '.join(cells) + f" {ds_} {sp:>20s}")
        print(f"  {'':18s} folds: {ds['n_folds_finished']} of "
              f"{ds['n_folds_used']} held out "
              f"(competition split has {ds['n_competition_folds']})")


# =============================================================================
# MAIN
# =============================================================================

def _write_json(output_path, all_results, skipped):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(all_results)
    if skipped:
        out['_skipped'] = skipped
    tmp = p.with_suffix(p.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(convert_numpy(out), f, indent=2)
    os.replace(tmp, p)
    return p


def main():
    parser = argparse.ArgumentParser(
        description='Ginkgo developability experiment (OPIG B5). '
                    'Evaluates the configured properties with KIRBy antibody '
                    'reps on the competition\'s cluster folds.')
    parser.add_argument(
        '--properties', type=str, default=None,
        help='Comma-separated property names (default: all configured)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: the property\'s approved '
             'pool from DATASET_REPS)')
    # The fold count comes from the competition split (2026-10-05, handoff 2.4
    # step 14): None holds out every competition fold. A number holds out that
    # many, in fold-ID order; 1 is a single held-out split.
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help='How many of the competition\'s cluster folds to hold out '
             '(default: every one). 1 = a single held-out split.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_ginkgo_developability.json)')
    # Accepted so older launchers that still pass it do not fail; ignored.
    # Every per-fold search runs unpruned (Adelaide, 2026-10-05; handoff 2.1).
    parser.add_argument(
        '--pruning-mode', type=str, default=None,
        help='Ignored: per-fold searches run unpruned (2026-10-05).')

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every fold.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()

    apply_distance_metric_arg(args)
    if args.pruning_mode:
        print(f"  --pruning-mode {args.pruning_mode} ignored: per-fold "
              f"searches run unpruned (2026-10-05).")
    properties = (args.properties.split(',') if args.properties
                  else ALL_PROPERTIES)
    rep_override = (args.reps.split(',') if args.reps else None)

    print("=" * 70)
    print("GINKGO DEVELOPABILITY EXPERIMENT (OPIG B5)")
    print(f"  Paper: van Niekerk et al. 2026, mAbs")
    print(f"  Properties: {len(properties)}")
    print(f"  Reps: {rep_override if rep_override else 'per-property approved pool'}")
    print(f"  Models: two per property (DECISIONS.md Step 1), "
          f"printed per property below")
    print(f"  Folds: {'every competition fold' if args.n_folds is None else args.n_folds}")
    print(f"  Per model per fold: {N_BASE} base representations + "
          f"{N_HYBRIDS} hybrids, ranked on the training rows")
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_ginkgo_developability.json')

    # Each finished competition fold is written to <output>.partial.jsonl and
    # read back on the next attempt (experiments/_checkpoint.py), and the JSON
    # below is rewritten after every fold. n_folds and pruning_mode are left
    # out of the identity: a fold's training rows, and so its unit, are the
    # same whatever count is held out, and pruning_mode is ignored.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    _ck = script_checkpoint('opig_ginkgo_developability', output_path, args,
                            exclude=('properties', 'output', 'n_folds',
                                     'pruning_mode'),
                            layout=CHECKPOINT_LAYOUT)
    _failed = []

    all_results = {}
    skipped = {}
    for prop_name in properties:
        if prop_name not in DATASET_REGISTRY:
            print(f"WARNING: Unknown property '{prop_name}', skipping")
            continue
        try:
            from experiments._shared import NoApprovedRepPool
            try:
                rep_names = (rep_override if rep_override
                             else _default_reps(prop_name))
            except NoApprovedRepPool as e:
                print(f"\n  SKIPPED {prop_name}: {e}")
                skipped[prop_name] = str(e)
                continue

            def _on_fold(partial, _p=prop_name):
                all_results[_p] = partial
                _write_json(output_path, all_results, skipped)

            prop_results = evaluate_property(prop_name, rep_names,
                                             args.n_folds, ck=_ck,
                                             on_fold=_on_fold)
            if is_failed_item(prop_results) or prop_results.get('failures'):
                _failed.append(prop_name)
            all_results[prop_name] = prop_results
            _write_json(output_path, all_results, skipped)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(prop_name)

    if all_results:
        print_analysis(all_results)
    if skipped:
        print(f"\n  {len(skipped)} of {len(properties)} properties skipped "
              f"for want of an approved rep pool: {sorted(skipped)}")

    p = _write_json(output_path, all_results, skipped)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'property')


if __name__ == '__main__':
    main()
