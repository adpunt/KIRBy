#!/usr/bin/env python3
"""
CGR vs SMILES Representation Comparison (Phase 3 of CGR_EMBEDDING_RESEARCH.md)

Tests whether CGR (Condensed Graph of Reaction) representations add unique
signal beyond SMILES-based representations for reaction yield prediction.

Scientific questions:
  1. Do CGR reps outperform their SMILES counterparts individually?
  2. Does adding CGR reps to the hybrid pool improve hybrid performance?
  3. Which CGR features (if any) are selected by the hybrid allocator?
  4. How correlated are CGR and SMILES reps (Linear CKA)?

Two comparison families:
  Fingerprint: cgr_morgan vs diff_ecfp4 vs drfp_2048
  Neural:      chytorch_graphormer vs rxnfp vs reactiont5

Datasets:
  - buchwald_hartwig (Ahneman/Dreher/Doyle, Science 2018, ~3955 reactions)
  - bh_doyle_plates (Doyle lab HTE, ~4599 reactions)
  - suzuki (Perera et al., Science 2018, ~5760 reactions)

Everything is per outer fold since 2026-10-04 (owner's rules of 2026-10-03):
  - Downstream models are the dataset's two decided models (DECISIONS.md
    Step 1): rf+dnn on bh_doyle_plates, rf+lightgbm on buchwald_hartwig and
    suzuki. Every representation, every hybrid and the family comparison are
    scored with those two only.
  - The outer folds come from kirby.find_best_hybrid.nested_cv_outer_folds, so
    the single representations and the hybrids are scored on the same folds.
  - For each group (without_cgr, with_cgr; below) and each outer fold, one search
    runs on that fold's training rows on the dataset's decided grid
    (fold_standard_selection). From it, per model: the 2 best single
    representations and the 2 best hybrids. Both are fitted on the fold's
    training rows and scored on its held-out rows.
  - The CGR difference per model is with_cgr minus without_cgr for each of
    those four slots, paired by fold.

Re-aligned 2026-10-05 to Adelaide's answers of that day (handoff 2.3):
  - Two groups of representations (answer 11), not "every generated
    representation with and without CGR":
      without_cgr: the dataset's pruned 10 (DATASET_REPS[ds]['general'])
                   minus cgr_morgan = 9 representations.
      with_cgr:    the pruned 10 plus chytorch_graphormer and cgr_smiles_lm
                   = 12 representations.
    Only these 12 are generated and scored. drfp_2048 and diff_ecfp4, which
    the fingerprint family used to compare against, are not in any pruned 10
    of these datasets (see FAMILY_EXTRA_REPS).
  - Per outer fold, per decided model, per group: 2 best single
    representations + 2 best hybrids from that group's search on the fold's
    training rows. No distance hybrid (answer 1: this experiment measures no
    neighbours or distances).
  - A representation matrix whose features are all constant (all zero
    included), or that holds a non-finite value, is refused and recorded,
    never scored (dirty-run bug 9: the cached chytorch_graphormer was all
    zeros).
  - A per-fold search that fails is recorded with dataset, group, fold, model
    and error, and that model's slots are skipped for that fold.
  - Each finished fold is written to <output>.partial.jsonl at once (dirty-run
    bug 12).
  - The outer fold count comes from --n-folds, whose default is get_config's
    n_splits (5 under --mode full, KIRBY_LOCAL_FOLDS honoured). 1 = a single
    grouped held-out split.

Usage:
    # Smoke test (1 dataset, 2 folds, no hybrid)
    python tests/experiments/opig_cgr_comparison.py \\
        --datasets buchwald_hartwig --n-folds 2 --skip-hybrid

    # Full evaluation
    python tests/experiments/opig_cgr_comparison.py

Reference:
    CGR_EMBEDDING_RESEARCH.md — Phase 3: Comparison Experiment
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load xgboost/lightgbm native OpenMP runtimes BEFORE rdkit (imported during
# rep generation) to avoid a duplicate-libomp segfault on macOS when these
# models and threaded sklearn estimators run in sequence. Optional deps —
# guard so environments without them are unaffected.
try:
    import xgboost as _xgboost  # noqa: F401
except Exception:
    pass
try:
    import lightgbm as _lightgbm  # noqa: F401
except Exception:
    pass

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, _cache_rep,
    fold_standard_selection, fold_searches, build_fold_hybrid,
    fold_hybrid_scoring_kwargs, get_domain_metric,
)
from experiments.decided_grid import decided_models
from kirby.find_best_hybrid import (nested_cv_outer_folds,
                                    nested_cv_cache_namespace)
from kirby.scaling import FeatureScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_DATASETS = ['buchwald_hartwig', 'bh_doyle_plates', 'suzuki']

CGR_REPS = {'cgr_morgan', 'chytorch_graphormer', 'cgr_smiles_lm'}

# The two groups of representations (Adelaide, 2026-10-05, answer 11).
# without_cgr = the pruned 10 minus this one; with_cgr = the pruned 10 plus
# CGR_ADDED. Every one of the three datasets has cgr_morgan in its pruned 10
# (complexity_helpers.DATASET_REPS).
CGR_IN_PRUNED = 'cgr_morgan'
CGR_ADDED = ('chytorch_graphormer', 'cgr_smiles_lm')
GROUPS = ('without_cgr', 'with_cgr')
GROUP_TITLES = {
    'without_cgr': 'pruned 10 without cgr_morgan',
    'with_cgr': 'pruned 10 + chytorch_graphormer + cgr_smiles_lm',
}


def representation_groups(dataset_name):
    """{'without_cgr': [9 names], 'with_cgr': [12 names]} for one dataset.

    The intended groups, before any representation is found missing or
    refused. Read from DATASET_REPS[ds]['general'] (representation pruning);
    the two CGR additions are the one exception answer 11 allows.
    """
    from experiments.complexity.complexity_helpers import DATASET_REPS
    pruned = list(DATASET_REPS[dataset_name]['general'])
    if CGR_IN_PRUNED not in pruned:
        raise ValueError(
            f"{dataset_name}: {CGR_IN_PRUNED} is not in its pruned "
            f"representations {pruned}; the without_cgr group is undefined")
    return {
        'without_cgr': [r for r in pruned if r != CGR_IN_PRUNED],
        'with_cgr': pruned + [r for r in CGR_ADDED if r not in pruned],
    }


# Representations outside the two groups that the family head-to-head may
# still score alone (never in a hybrid). Empty since 2026-10-05: the rule is
# "only the pruned representations", and drfp_2048 and diff_ecfp4 -- the
# fingerprint family's SMILES side -- were cut from these datasets' pruned 10.
# With it empty the fingerprint family has no SMILES side and is reported as
# not compared. Set it to ('diff_ecfp4', 'drfp_2048') to restore the old
# fingerprint comparison (2 more representations generated and scored alone).
FAMILY_EXTRA_REPS = ()

FAMILIES = {
    'fingerprint': {
        'cgr': ['cgr_morgan'],
        'smiles': ['diff_ecfp4', 'drfp_2048'],
    },
    'neural': {
        'cgr': ['chytorch_graphormer', 'cgr_smiles_lm'],
        'smiles': ['rxnfp', 'reactiont5'],
    },
}

# All family reps for CKA analysis
FAMILY_REPS = ['cgr_morgan', 'diff_ecfp4', 'drfp_2048',
               'chytorch_graphormer', 'cgr_smiles_lm', 'rxnfp', 'reactiont5']

# Seed of the outer split. nested_cv_cache_namespace takes it too, so the
# per-fold searches are filed under the split they were run on.
SEED = 42

# Folds inside one outer fold's training rows, used only to pick the best CGR
# and the best SMILES representation of a family for that outer fold.
INNER_FOLDS = 5

# Recipe fields copied off each selected hybrid into the result JSON.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count', 'step_size', 'tune')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cgr_comparison_data(dataset_name, rep_names):
    """Load a reaction dataset and generate only the representations named.

    rep_names: the two groups' 12 representations (+ FAMILY_EXTRA_REPS).
    Since 2026-10-05 nothing else is generated: the old loader built every
    reaction representation (up to 44, with the mordred variants) and searched
    all of them. A representation that cannot be generated is listed in the
    returned 'missing' with its reason.
    """
    from kirby.datasets.reaction_yields import (
        load_buchwald_hartwig, load_bh_doyle_plates, load_suzuki,
    )
    from kirby.representations.reaction import (
        create_drfp, create_concat_ecfp4, create_concat_maccs,
        create_concat_pdv, create_product_ecfp4, create_reactant_ecfp4,
        create_product_maccs, create_reactant_maccs,
        create_product_pdv, create_reactant_pdv,
        create_rxnfp, create_onehot_components,
        create_diff_ecfp4, create_diff_maccs,
        create_component_descriptors, create_ligand_descriptors,
        create_reactiont5, create_cgr_morgan, create_chytorch_graphormer,
        create_cgr_smiles_lm,
        create_reaction_rep_variant, _MOL_REP_VARIANTS,
    )

    if dataset_name == 'buchwald_hartwig':
        data = load_buchwald_hartwig()
    elif dataset_name == 'bh_doyle_plates':
        data = load_bh_doyle_plates()
    elif dataset_name == 'suzuki':
        data = load_suzuki()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    groups = data.get('groups')
    if groups is None:
        print(f"  WARNING: {dataset_name} has no groups, skipping")
        return None

    rxn_smiles = data['rxn_smiles']
    components = data['components']
    labels = list(data['labels'])

    # Test-only row cap BEFORE rep generation (no-op unless KIRBY_SMOKE_N set).
    # Capping here keeps the slow rxnmapper/neural rep generators off the full
    # dataset. Cache key is suffixed so capped reps never pollute the real cache.
    _smoke = os.environ.get('KIRBY_SMOKE_N', '').strip()
    cache_key = dataset_name
    if _smoke.isdigit() and 0 < int(_smoke) < len(labels):
        n_keep = int(_smoke)
        n = len(labels)
        _idx = np.unique(np.linspace(0, n - 1, n_keep).round().astype(int))
        rxn_smiles = [rxn_smiles[i] for i in _idx]
        components = [components[i] for i in _idx]
        labels = [labels[i] for i in _idx]
        groups = np.asarray(groups)[_idx]
        cache_key = f'{dataset_name}_smoke{n_keep}'
        print(f"  [SMOKE] KIRBY_SMOKE_N={n_keep}: capped {n} -> {len(_idx)} reactions before rep gen")
    data = dict(data)
    data['labels'] = np.asarray(labels, dtype=np.float64)

    # Full generator registry — all reaction reps
    _generators = {
        'drfp_2048': lambda: create_drfp(rxn_smiles),
        'concat_ecfp4': lambda: create_concat_ecfp4(components),
        'concat_maccs': lambda: create_concat_maccs(components),
        'concat_pdv': lambda: create_concat_pdv(components),
        'product_ecfp4': lambda: create_product_ecfp4(rxn_smiles),
        'reactant_ecfp4': lambda: create_reactant_ecfp4(rxn_smiles),
        'product_maccs': lambda: create_product_maccs(rxn_smiles),
        'reactant_maccs': lambda: create_reactant_maccs(rxn_smiles),
        'product_pdv': lambda: create_product_pdv(rxn_smiles),
        'reactant_pdv': lambda: create_reactant_pdv(rxn_smiles),
        'rxnfp': lambda: create_rxnfp(rxn_smiles),
        'onehot_components': lambda: create_onehot_components(components),
        'diff_ecfp4': lambda: create_diff_ecfp4(rxn_smiles),
        'diff_maccs': lambda: create_diff_maccs(rxn_smiles),
        'component_descriptors': lambda: create_component_descriptors(components),
        'ligand_descriptors': lambda: create_ligand_descriptors(components),
        'reactiont5': lambda: create_reactiont5(rxn_smiles),
        'cgr_morgan': lambda: create_cgr_morgan(rxn_smiles),
        'chytorch_graphormer': lambda: create_chytorch_graphormer(rxn_smiles),
        'cgr_smiles_lm': lambda: create_cgr_smiles_lm(rxn_smiles),
    }

    # Add molecular variant generators
    for mol_name in _MOL_REP_VARIANTS:
        for role in ('product', 'reactant', 'diff'):
            rep_name = f'{role}_{mol_name}'
            _generators[rep_name] = (
                lambda rr=rxn_smiles, mn=mol_name, r=role:
                    create_reaction_rep_variant(rr, mn, r)
            )

    # Test-only: under KIRBY_SMOKE_N skip the heavy rep generators that dominate
    # (or hang) local wall time — neural reps (transformer/graph-net CPU forward
    # passes) and the mordred reaction variants (~1613-D each; observed to stall).
    # cgr_morgan (a fingerprint-based CGR rep) is kept, so the CGR-vs-SMILES
    # comparison still has a CGR member. No-op when the env var is unset, so
    # server runs generate the full reaction-rep set.
    _smoke_skip = set()
    if os.environ.get('KIRBY_SMOKE_N', '').strip():
        _smoke_skip = {'reactiont5', 'chytorch_graphormer', 'cgr_smiles_lm'}
        _smoke_skip |= {n for n in _generators if 'mordred' in n}

    # Generate the named representations only, with caching.
    reps = {}
    missing = {}
    for name in rep_names:
        gen_fn = _generators.get(name)
        if gen_fn is None:
            missing[name] = 'no generator for this name in this script'
            print(f"  WARNING: {name}: {missing[name]}")
            continue
        if name in _smoke_skip:
            missing[name] = 'skipped under KIRBY_SMOKE_N (local test only)'
            print(f"  [SMOKE] skipping neural rep {name} (KIRBY_SMOKE_N set)")
            continue
        try:
            reps[name] = _cache_rep(cache_key, name, gen_fn)
        except ImportError as e:
            missing[name] = f'missing dependency: {e}'
            print(f"  WARNING: {name} skipped (missing dep): {e}")
        except Exception as e:
            missing[name] = f'{type(e).__name__}: {e}'
            print(f"  WARNING: {name} failed: {e}")

    print(f"  Generated {len(reps)} of {len(rep_names)} representations")

    return {
        'labels': np.array(data['labels'], dtype=np.float64),
        'groups': np.array(groups),
        'reps': reps,
        'missing': missing,
        'components': components,
        'metadata': data.get('metadata', {}),
    }


# =============================================================================
# REFUSING DEGENERATE REPRESENTATIONS (dirty-run bug 9, 2026-10-05)
# =============================================================================

def refusal_reason(X, n_rows):
    """Why a representation matrix cannot be scored, or None if it can.

    Refused: not 2-D, a row count other than the labels', no features, any
    non-finite value, or every feature constant over all rows (an all-zero
    matrix is the case found: the cached chytorch_graphormer of the dirty
    run). Such a matrix carries no information, and scoring it would report a
    model that predicts the mean as if it were the representation's result.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        return f'{X.ndim}-D array, expected 2-D'
    if X.shape[0] != n_rows:
        return f'{X.shape[0]} rows, labels have {n_rows}'
    if X.shape[1] == 0:
        return 'no features'
    if not np.issubdtype(X.dtype, np.number):
        return f'non-numeric dtype {X.dtype}'
    n_bad = int(np.size(X) - np.count_nonzero(np.isfinite(X)))
    if n_bad:
        return f'{n_bad} non-finite values'
    if not np.any(X != 0):
        return f'all {X.shape[1]} features are zero on every row'
    n_varying = int(np.count_nonzero(np.ptp(X, axis=0)))
    if n_varying == 0:
        return f'all {X.shape[1]} features are constant over every row'
    return None


def usable_representations(reps, n_rows):
    """(kept {name: X}, refused {name: reason}). Printed per refusal."""
    kept, refused = {}, {}
    for name in sorted(reps):
        why = refusal_reason(reps[name], n_rows)
        if why is None:
            kept[name] = reps[name]
        else:
            refused[name] = why
            print(f"  REFUSED {name}: {why}. Not scored, not searched.")
    return kept, refused


# =============================================================================
# REP POOL PARTITIONING
# =============================================================================

def partition_rep_pools(reps, intended):
    """The two groups, restricted to the representations that are usable.

    reps: the usable matrices (after usable_representations).
    intended: representation_groups(ds).
    Returns {group: {name: X}}, in the intended order.
    """
    return {g: {r: reps[r] for r in intended[g] if r in reps}
            for g in GROUPS}


# =============================================================================
# EVALUATION
# =============================================================================

def _rmse(y_true, y_pred):
    """The dataset metric, as build_fold_hybrid's metric_fn wants it."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_score(X_train, X_val, y_train, y_val, factory):
    """Fit one model on a fold's training rows, score it on its held-out rows.

    The scaler is fitted on the training rows only. Since 2026-10-05 it is
    FeatureScaler (answer 17): each 0/1 feature passes through unscaled and
    every other feature is standardised, feature by feature. Before, the whole
    matrix was either standardised or left alone depending on whether every
    value was 0/1, so a hybrid mixing fingerprint bits and descriptors had its
    bits standardised. The matrices are no longer converted to float64 here
    (answer 18: float32 for every representation). Returns {'rmse', 'r2',
    'spearman'}, or None when the fit raised (printed, never silent).
    """
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)

    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    try:
        m = factory()
        m.fit(X_train, y_train)
    except Exception as e:
        # Log fold-fit failures (OPIG-18) instead of silently dropping.
        print(f"      [fold fit failed: {type(e).__name__}: {e}]", flush=True)
        return None

    preds = np.asarray(m.predict(X_val), dtype=np.float64).ravel()
    ss_res = np.sum((y_val - preds) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    rho = (float(spearmanr(y_val, preds)[0]) if len(y_val) > 2
           else float('nan'))
    return {
        'rmse': _rmse(y_val, preds),
        'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
        'spearman': rho,
    }


ONE_FOLD_NOTE = 'one fold, no spread'


def _summarise(per_fold):
    """Mean and spread over folds of a list of _fit_score results (None = failed).

    With one completed fold the spread is None and 'spread_note' says so
    (2026-10-05: --n-folds 1 is a single held-out split).
    """
    ok = [p for p in per_fold if p is not None]
    if not ok:
        return None
    out = {}
    for key in ('r2', 'rmse', 'spearman'):
        vals = [p[key] for p in ok]
        out[key] = {'mean': float(np.nanmean(vals)),
                    'std': (float(np.nanstd(vals)) if len(ok) > 1 else None)}
    if len(ok) == 1:
        out['spread_note'] = ONE_FOLD_NOTE
    out['n_folds_completed'] = len(ok)
    out['per_fold'] = per_fold
    return out


def evaluate_rep(X, labels, folds, factory, ck=None, ck_key=None):
    """One representation x one model over the given (train, val) folds.

    With a checkpoint, each fold's score is stored as soon as it is computed
    and read back on the next attempt (2026-10-05, dirty-run bug 12: before,
    a representation's scores were stored only once every fold of both models
    had finished). A failed fit (None) is not stored, so it is retried.
    """
    X = np.asarray(X)
    labels = np.asarray(labels, dtype=np.float64)
    per_fold = []
    for k, (tr, va) in enumerate(folds):
        fk = dict(ck_key or {}, fold=k)
        if ck is not None and ck_key is not None and ck.done(**fk):
            per_fold.append(ck.get(**fk))
            continue
        r = _fit_score(X[tr], X[va], labels[tr], labels[va], factory)
        r = None if r is None else dict(r, fold=k)
        if ck is not None and ck_key is not None and r is not None:
            ck.put(convert_numpy(r), **fk)
        per_fold.append(r)
    return _summarise(per_fold)


# =============================================================================
# HYBRID POOLS, PER OUTER FOLD
# =============================================================================

def _group_namespace(fold_i, n_outer, group):
    """The per-fold search store name for one group of representations.

    Representations are not part of the stored-result key, so each group's
    searches are filed under the fold's nested-CV namespace plus the group
    name; without it the second group would read the first group's rows.
    Renamed 2026-10-05 from '__cgr_smiles_only' / '__cgr_smiles_plus_cgr'
    because the groups changed (answer 11): rows stored under the old names
    came from searches over up to 44 representations and are never read.
    """
    return f'{nested_cv_cache_namespace(fold_i, n_outer, SEED)}__cgr_{group}'


def _search_failures(pool_reps, labels, groups, tr, k, n_outer, dataset_name,
                     metric_name, ns):
    """The recorded failures of the search fold_standard_selection just ran.

    Reads the in-process memo with the same arguments fold_standard_selection
    passed (no models, general tier), so nothing is searched again.
    """
    try:
        srch = fold_searches(
            pool_reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            task_type='regression', metric=metric_name, seed=SEED,
            cache_namespace=ns)
        return [{kk: (str(vv) if not isinstance(vv, (int, float, str,
                                                     type(None))) else vv)
                 for kk, vv in f.items()}
                for f in (getattr(srch, 'failures', None) or [])]
    except Exception as e:  # reading the memo must not stop the fold
        return [{'error': f'could not read the failure record: '
                          f'{type(e).__name__}: {e}'}]


def evaluate_group_per_fold(pool_reps, labels, groups, folds, factories,
                            group, dataset_name, metric, ck=None):
    """The standard selection for one group of representations, per outer fold.

    For each outer fold: one search on that fold's training rows over this
    group, on the dataset's decided grid (fold_standard_selection; pruning
    off, augmentation as decided_aug_conditions keeps -- 'none' on all three
    reaction datasets). Per decided model it names the 2 best single
    representations and the 2 best hybrids (a second hybrid whose selected
    features equal the first's is skipped for the next). Each is fitted on
    the fold's training rows and scored on its held-out rows. A hybrid is
    rebuilt with build_fold_hybrid from its recipe, with the
    per-representation scores that same per-fold search measured.

    metric: (metric_name, metric_fn, lower_is_better) from get_domain_metric.

    A search that fails, or a model short of 2 representations or 2 hybrids,
    is recorded in the fold's 'search_failures' / 'shortfall' with dataset,
    group, fold, model and the error, and that model's missing slots are
    skipped for that fold (2026-10-05). Nothing is filled in instead.

    ck: the script's checkpoint. Each finished fold is stored at once under
    (dataset, group, fold) and read back on the next attempt (dirty-run bug
    12, 2026-10-05).
    """
    metric_name, metric_fn, lower_is_better = metric
    n_outer = len(folds)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    print(f"\n  Per-fold searches ({GROUP_TITLES[group]}: {len(pool_reps)} "
          f"representations, {n_outer} outer folds)...")

    per_fold = []
    for k, (tr, va) in enumerate(folds):
        ck_key = dict(dataset=dataset_name, group=group, fold=int(k))
        if ck is not None and ck.done(**ck_key):
            per_fold.append(ck.get(**ck_key))
            print(f"    fold {k}: finished in an earlier attempt, read back")
            continue

        ns = _group_namespace(k, n_outer, group)
        sel = fold_standard_selection(
            pool_reps, labels, groups, train_idx=tr, fold_i=k,
            n_outer_folds=n_outer, dataset_name=dataset_name,
            models=list(factories), task_type='regression',
            metric=metric_name, seed=SEED, cache_namespace=ns)

        tr_reps = {r: np.asarray(X)[tr] for r, X in pool_reps.items()}
        va_reps = {r: np.asarray(X)[va] for r, X in pool_reps.items()}
        y_tr, y_va = labels[tr], labels[va]

        fold_rec = {'fold': k, 'group': group,
                    'n_train': int(len(tr)), 'n_val': int(len(va)),
                    'cache_namespace': ns,
                    'representations': sorted(pool_reps),
                    'selection_scope': 'fold_train_rows',
                    'metric': metric_name,
                    'lower_is_better': bool(lower_is_better),
                    'search_failures': [], 'shortfall': [], 'models': {}}

        failed = sorted(m for m, s in sel.items() if s is None)
        if failed:
            errs = _search_failures(pool_reps, labels, groups, tr, k, n_outer,
                                    dataset_name, metric_name, ns)
            for m in failed:
                fold_rec['search_failures'].append({
                    'dataset': dataset_name, 'group': group, 'fold': int(k),
                    'model': m, 'errors': errs})
                print(f"    fold {k} {m:8s} SEARCH FAILED, slots skipped: "
                      f"{[e.get('error') or e.get('error_type') for e in errs]}")

        for m, s in sel.items():
            if s is None:
                continue
            factory = factories[m]
            bd = s.get('base_detail') or {}
            if len(bd.get('selected') or []) < 2 or len(s['hybrids']) < 2:
                fold_rec['shortfall'].append({
                    'dataset': dataset_name, 'group': group, 'fold': int(k),
                    'model': m,
                    'n_base': len(bd.get('selected') or []),
                    'n_hybrids': len(s['hybrids']),
                    'reason': bd.get('shortfall_reason')})
            base_rows = []
            for i, e in enumerate(bd.get('selected') or [], 1):
                rep = e['rep']
                res = _fit_score(tr_reps[rep], va_reps[rep], y_tr, y_va,
                                 factory)
                base_rows.append({'slot': f'base{i}', 'rep': rep,
                                  'training_rows_score': e.get('score'),
                                  'heldout': res})
                if res is not None:
                    print(f"    fold {k} {m:8s} base{i}   {rep:28s} "
                          f"RMSE={res['rmse']:.2f}")

            hyb_rows = []
            for i, e in enumerate(s['hybrids'], 1):
                bp = e['blueprint']
                row = {'slot': f'hybrid{i}',
                       **{f: e.get(f) for f in HYBRID_FIELDS},
                       'aug_condition': e.get('aug_condition'),
                       # Mean over the search's inner folds, inside this outer
                       # fold's training rows.
                       'training_rows_score': e.get('score_mean')}
                try:
                    X_tr_h, X_va_h, fi = build_fold_hybrid(
                        tr_reps, va_reps, y_tr, bp, groups=groups[tr],
                        # This fold's search measured these on this fold's
                        # training rows; reused so the budget split is the one
                        # that search settled on (dirty-run bug 1).
                        precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                         None),
                        **fold_hybrid_scoring_kwargs(
                            bp, model=factory(), metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                except Exception as ex:
                    row['error'] = f'{type(ex).__name__}: {ex}'
                    row['heldout'] = None
                    print(f"    fold {k} {m:8s} hybrid{i} build failed: "
                          f"{row['error']}")
                    hyb_rows.append(row)
                    continue
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                row['n_features'] = int(np.asarray(X_tr_h).shape[1])
                row['heldout'] = _fit_score(X_tr_h, X_va_h, y_tr, y_va,
                                            factory)
                hyb_rows.append(row)
                if row['heldout'] is not None:
                    print(f"    fold {k} {m:8s} hybrid{i} "
                          f"{row['n_features']:5d} features "
                          f"({row['allocation_method']}, "
                          f"{row['importance_method']}, "
                          f"budget {row['budget']}, "
                          f"eval {row['internal_evaluator']}, "
                          f"rep_count {row['rep_count']})  "
                          f"RMSE={row['heldout']['rmse']:.2f}")
            fold_rec['models'][m] = {
                'base': base_rows, 'hybrids': hyb_rows,
                'hybrids_skipped_identical_features': [
                    x.get('label') for x in
                    (s.get('hybrids_skipped_identical_features') or [])]}
        per_fold.append(fold_rec)
        # A fold whose search failed is in the result file but not stored, so
        # the next attempt searches it again.
        if ck is not None and not fold_rec['search_failures']:
            ck.put(convert_numpy(fold_rec), **ck_key)

    return {
        'group': group,
        'title': GROUP_TITLES[group],
        'n_reps': len(pool_reps),
        'rep_names': sorted(pool_reps.keys()),
        'selection_scope': 'fold_train_rows',
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'per_fold': per_fold,
        'search_failures': [f for r in per_fold
                            for f in r.get('search_failures', [])],
        'by_model': _slot_summary(per_fold),
    }


def _slot_summary(per_fold):
    """{model: {slot: {'per_fold': [...], 'rmse_mean', 'r2_mean',
    'spearman_mean', 'n_folds'}}} over the held-out folds."""
    out = {}
    for fold_rec in per_fold:
        k = fold_rec['fold']
        for m, mr in fold_rec['models'].items():
            for row in mr['base'] + mr['hybrids']:
                d = out.setdefault(m, {}).setdefault(row['slot'],
                                                     {'per_fold': []})
                h = row.get('heldout')
                d['per_fold'].append({
                    'fold': k,
                    'name': row.get('rep') or row.get('label'),
                    'rmse': None if h is None else h['rmse'],
                    'r2': None if h is None else h['r2'],
                    'spearman': None if h is None else h['spearman']})
    for m in out:
        for d in out[m].values():
            for key in ('rmse', 'r2', 'spearman'):
                vals = [p[key] for p in d['per_fold'] if p[key] is not None]
                d[f'{key}_mean'] = (float(np.nanmean(vals)) if vals
                                    else float('nan'))
            d['n_folds'] = sum(p['rmse'] is not None for p in d['per_fold'])
    return out


def extract_cgr_selection_info(pool_result):
    """How many features each CGR representation got in each hybrid built.

    Read from the hybrids rebuilt on each outer fold's training rows, so the
    answer is per fold and per model, not from one recipe fitted on every row.
    """
    info = {}
    for fold_rec in pool_result.get('per_fold') or []:
        for m, mr in fold_rec['models'].items():
            for h in mr['hybrids']:
                if 'features_by_rep' not in h:
                    continue
                for cgr_rep in sorted(CGR_REPS):
                    d = info.setdefault(m, {}).setdefault(
                        cgr_rep, {'per_fold': [], 'n_hybrids_built': 0,
                                  'n_hybrids_with_features': 0})
                    n_f = int(h['features_by_rep'].get(cgr_rep, 0))
                    d['per_fold'].append({'fold': fold_rec['fold'],
                                          'slot': h['slot'],
                                          'n_features': n_f})
                    d['n_hybrids_built'] += 1
                    d['n_hybrids_with_features'] += int(n_f > 0)
    return info


# =============================================================================
# CORRELATION ANALYSIS (LINEAR CKA)
# =============================================================================

def linear_cka(X, Y):
    """Linear CKA between two representation matrices.

    Kornblith et al. (2019). Similarity of Neural Network Representations
    Revisited. ICML.

    Args:
        X: (n, d1) array.
        Y: (n, d2) array.

    Returns:
        float in [0, 1]. Higher = more similar.
    """
    # float64 here only for the sums of squares inside CKA, a description of
    # two matrices, not a model fit; nothing is fitted or stored from it.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # CKA is only defined for representations of the SAME samples in the SAME
    # order. If a rep silently changed its row count (e.g. a generator dropped
    # rows), the matmuls below would still produce a number — assert instead so
    # a row-count drift fails loudly rather than returning a meaningless score
    # (and 0.0, which is indistinguishable from "no similarity") (OPIG-23).
    assert X.shape[0] == Y.shape[0], (
        f"linear_cka: row-count mismatch X={X.shape[0]} vs Y={Y.shape[0]}; "
        "reps must describe the same samples in the same order")
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX_norm_sq = np.linalg.norm(X.T @ X, 'fro') ** 2
    YtY_norm_sq = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    YtX_norm_sq = np.linalg.norm(Y.T @ X, 'fro') ** 2

    denom = np.sqrt(XtX_norm_sq * YtY_norm_sq)
    if denom < 1e-10:
        return 0.0
    return float(YtX_norm_sq / denom)


def compute_correlation_matrix(reps):
    """Pairwise Linear CKA for family reps.

    Returns:
        (rep_names, cka_matrix) — only includes reps that were generated.
    """
    available = [r for r in FAMILY_REPS if r in reps]
    n = len(available)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            cka = linear_cka(reps[available[i]], reps[available[j]])
            matrix[i, j] = cka
            matrix[j, i] = cka

    return available, matrix


# =============================================================================
# FAMILY ANALYSIS
# =============================================================================

def _pick_on_training_rows(reps, rep_names, labels, groups, train_idx,
                           factory):
    """The representation of rep_names with the lowest RMSE on the training rows.

    Scored by INNER_FOLDS group-aware folds inside the outer fold's training
    rows, so the held-out fold plays no part in the choice. A group with one
    available representation needs no choice and costs no fits.
    """
    avail = [r for r in rep_names if r in reps]
    if len(avail) <= 1:
        return (avail[0] if avail else None), {}
    y_tr = np.asarray(labels, dtype=np.float64)[train_idx]
    g_tr = np.asarray(groups)[train_idx]
    n_inner = min(INNER_FOLDS, len(np.unique(g_tr)))
    if n_inner < 2:
        return None, {}
    inner = nested_cv_outer_folds({}, y_tr, g_tr, n_inner, SEED, 'regression')
    scores = {}
    for r in avail:
        res = evaluate_rep(np.asarray(reps[r])[train_idx], y_tr, inner,
                           factory)
        if res is not None:
            scores[r] = res['rmse']['mean']
    if not scores:
        return None, scores
    # Lowest RMSE first, then name, so a tie resolves the same way every run.
    return min(scores, key=lambda r: (scores[r], r)), scores


def analyze_families(reps, labels, groups, folds, factories,
                     individual_results):
    """Compare CGR vs SMILES within each family, per model, per outer fold.

    For each outer fold the best CGR and the best SMILES representation of a
    family are picked on that fold's training rows (_pick_on_training_rows),
    then their held-out scores on that fold are read from individual_results,
    which scored every representation on these same folds.

    Returns {model: {family: {'per_fold': [...], 'cgr_rmse_mean',
    'smiles_rmse_mean', 'n_folds', 'n_folds_cgr_wins'}}}. A family with no
    usable representation on one side is {'not_compared': reason} (since
    2026-10-05 the fingerprint family's SMILES side, diff_ecfp4 and
    drfp_2048, is not generated unless FAMILY_EXTRA_REPS names it).
    """
    analysis = {}
    for model_name, factory in factories.items():
        for family_name, family_def in FAMILIES.items():
            empty = [side for side in ('cgr', 'smiles')
                     if not any(r in individual_results
                                for r in family_def[side])]
            if empty:
                analysis.setdefault(model_name, {})[family_name] = {
                    'not_compared': (
                        f"no scored representation on the {empty} side; "
                        f"named: {[family_def[s] for s in empty]}")}
                continue
            rows = []
            for k, (tr, _va) in enumerate(folds):
                picked = {}
                for side in ('cgr', 'smiles'):
                    rep, inner_scores = _pick_on_training_rows(
                        reps, family_def[side], labels, groups, tr, factory)
                    held = None
                    if rep is not None:
                        pf = (individual_results.get(rep, {}).get('models', {})
                              .get(model_name) or {}).get('per_fold') or []
                        if k < len(pf) and pf[k] is not None:
                            held = pf[k]['rmse']
                    picked[side] = {'rep': rep, 'heldout_rmse': held,
                                    'training_rows_rmse': inner_scores}
                c, s = (picked['cgr']['heldout_rmse'],
                        picked['smiles']['heldout_rmse'])
                rows.append({
                    'fold': k,
                    'cgr_rep': picked['cgr']['rep'],
                    'cgr_rmse': c,
                    'cgr_training_rows_rmse': picked['cgr']['training_rows_rmse'],
                    'smiles_rep': picked['smiles']['rep'],
                    'smiles_rmse': s,
                    'smiles_training_rows_rmse':
                        picked['smiles']['training_rows_rmse'],
                    'cgr_wins': (None if c is None or s is None
                                 else bool(c < s)),
                })
            paired = [r for r in rows if r['cgr_wins'] is not None]
            analysis.setdefault(model_name, {})[family_name] = {
                'per_fold': rows,
                'cgr_rmse_mean': (float(np.mean([r['cgr_rmse'] for r in paired]))
                                  if paired else float('nan')),
                'smiles_rmse_mean': (float(np.mean([r['smiles_rmse']
                                                    for r in paired]))
                                     if paired else float('nan')),
                'n_folds': len(paired),
                'n_folds_cgr_wins': sum(r['cgr_wins'] for r in paired),
                'selection_scope': 'fold_train_rows',
            }
    return analysis


# =============================================================================
# MAIN PER-DATASET EVALUATION
# =============================================================================

def evaluate_dataset(dataset_name, n_folds, skip_hybrid=False,
                     skip_correlation=False, ck=None):
    """Run full CGR comparison for one dataset.

    n_folds: the outer fold count (--n-folds; default get_config's n_splits).
    Capped by the number of groups. 1 = a single grouped held-out split.
    ck: the script's checkpoint; single-representation scores and each
    group's folds are stored as they finish (dirty-run bug 12).
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # The two groups (answer 11) and nothing else is generated.
    intended = representation_groups(dataset_name)
    wanted = list(dict.fromkeys(intended['with_cgr'] + intended['without_cgr']
                                + list(FAMILY_EXTRA_REPS)))
    data = load_cgr_comparison_data(dataset_name, wanted)
    if data is None:
        return None

    # Test-only row cap for local smoke tests (no-op unless KIRBY_SMOKE_N set).
    # Caps reps/labels/groups consistently; components stays unused downstream.
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)

    labels = data['labels']
    groups = data['groups']
    n = len(labels)
    n_groups = len(np.unique(groups))

    # Dirty-run bug 9 (2026-10-05): a matrix that is all zeros, all constant
    # or holds a non-finite value is refused here, before any scoring or
    # search, and the reason is written to the result file.
    reps, refused = usable_representations(data['reps'], n)
    missing = dict(data.get('missing') or {})
    pools = partition_rep_pools(reps, intended)
    group_record = {
        g: {'title': GROUP_TITLES[g], 'intended': intended[g],
            'used': list(pools[g]),
            'missing': {r: missing[r] for r in intended[g] if r in missing},
            'refused': {r: refused[r] for r in intended[g] if r in refused}}
        for g in GROUPS}
    for g in GROUPS:
        gr = group_record[g]
        print(f"  {g}: {len(gr['used'])} of {len(gr['intended'])} "
              f"representations usable ({gr['title']})"
              + (f"; missing {sorted(gr['missing'])}" if gr['missing'] else '')
              + (f"; refused {sorted(gr['refused'])}" if gr['refused'] else ''))

    actual_folds = min(int(n_folds), n_groups)
    if n_groups < 2 or actual_folds < 1:
        return {'error': f'{n_groups} group(s): no outer split possible',
                'representation_groups': group_record}

    # The dataset's two decided downstream models (DECISIONS.md Step 1).
    # Raises UndecidedSearch for a dataset with no row there.
    model_names = decided_models(dataset_name, 'general')
    models = get_model_factories(model_names, 'regression', n_samples=n)
    # The dataset's metric and its direction (rmse, lower is better, on all
    # three). The tables below are written for rmse, so another metric stops
    # the dataset instead of being printed under the wrong name.
    metric = get_domain_metric({'dataset_name': dataset_name,
                                'task_type': 'regression',
                                'domain': 'reaction', 'labels': labels})
    if metric[0] != 'rmse':
        return {'error': f"{dataset_name}: metric {metric[0]}, this script "
                         f"reports rmse only"}

    # One outer split for everything below: the single representations, the
    # family picks and the per-fold searches.
    folds = nested_cv_outer_folds(reps, labels, groups, actual_folds, SEED,
                                  'regression')

    print(f"  {n} reactions, {n_groups} groups, {actual_folds} outer "
          f"fold(s), models {list(models)}")

    cgr_available = {k for k in CGR_REPS if k in reps}

    # --- Phase A: Individual rep evaluation ---
    # Every usable representation of the two groups, each decided model, on
    # the outer folds. Stored per representation as it finishes.
    print(f"\n  --- Individual Rep Evaluation ---")
    individual_results = {}

    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_type = 'cgr' if rep_name in CGR_REPS else 'smiles'
        family = _get_family(rep_name)

        ck_key = dict(dataset=dataset_name, phase='single', rep=rep_name)
        if ck is not None and ck.done(**ck_key):
            rep_results = ck.get(**ck_key)
        else:
            rep_results = {}
            for model_name, factory in models.items():
                res = evaluate_rep(
                    X, labels, folds, factory, ck=ck,
                    ck_key=dict(dataset=dataset_name, phase='single_fold',
                                rep=rep_name, model=model_name))
                if res is not None:
                    rep_results[model_name] = res
            if ck is not None and rep_results:
                ck.put(convert_numpy(rep_results), **ck_key)

        if rep_results:
            tag = ' [CGR]' if rep_type == 'cgr' else ''
            line = '  '.join(f"{m} RMSE={r['rmse']['mean']:.2f}"
                             for m, r in rep_results.items())
            print(f"    {rep_name:30s} {X.shape[1]:>5d}D  {line}{tag}")

            individual_results[rep_name] = {
                'dim': X.shape[1],
                'family': family,
                'type': rep_type,
                'in_groups': [g for g in GROUPS if rep_name in pools[g]],
                'models': rep_results,
            }

    # --- Phase B: Hybrid comparison, one per-fold search per group ---
    hybrid_comparison = {}
    if not skip_hybrid:
        hybrid_comparison['without_cgr'] = evaluate_group_per_fold(
            pools['without_cgr'], labels, groups, folds, models,
            'without_cgr', dataset_name, metric, ck=ck)

        # with_cgr adds nothing when every CGR representation is missing or
        # refused: the two groups would hold the same representations.
        if set(pools['with_cgr']) == set(pools['without_cgr']):
            hybrid_comparison['with_cgr'] = {
                'not_run': ('no usable CGR representation: the group equals '
                            'without_cgr'),
                'missing': group_record['with_cgr']['missing'],
                'refused': group_record['with_cgr']['refused']}
            print("  WARNING: with_cgr has no usable CGR representation; "
                  "not run")
        else:
            hybrid_comparison['with_cgr'] = evaluate_group_per_fold(
                pools['with_cgr'], labels, groups, folds, models,
                'with_cgr', dataset_name, metric, ck=ck)
            hybrid_comparison['with_cgr']['cgr_selection'] = \
                extract_cgr_selection_info(hybrid_comparison['with_cgr'])

    # --- Phase C: Compute deltas ---
    # Per model and per slot (base1, base2, hybrid1, hybrid2): the with_cgr
    # held-out RMSE minus the without_cgr one, paired by outer fold. Negative
    # = CGR helped. Both groups were searched on each fold's training rows
    # only.
    delta_by_model = {}
    if all('by_model' in hybrid_comparison.get(g, {}) for g in GROUPS):
        so = hybrid_comparison['without_cgr']['by_model']
        sc = hybrid_comparison['with_cgr']['by_model']
        for model_name in so:
            if model_name not in sc:
                continue
            for slot, d_so in so[model_name].items():
                d_sc = sc[model_name].get(slot)
                if d_sc is None:
                    continue
                by_fold_sc = {p['fold']: p for p in d_sc['per_fold']}
                pairs = []
                for p in d_so['per_fold']:
                    q = by_fold_sc.get(p['fold'])
                    if q is None or p['rmse'] is None or q['rmse'] is None:
                        continue
                    pairs.append({'fold': p['fold'],
                                  'without_cgr_name': p['name'],
                                  'with_cgr_name': q['name'],
                                  'without_cgr_rmse': p['rmse'],
                                  'with_cgr_rmse': q['rmse'],
                                  'rmse_delta': q['rmse'] - p['rmse']})
                if not pairs:
                    continue
                a = float(np.mean([x['without_cgr_rmse'] for x in pairs]))
                b = float(np.mean([x['with_cgr_rmse'] for x in pairs]))
                delta_by_model.setdefault(model_name, {})[slot] = {
                    'rmse_delta': b - a,
                    'without_cgr_rmse_mean': a,
                    'with_cgr_rmse_mean': b,
                    'n_folds_used': len(pairs),
                    'per_fold': pairs,
                }
                if len(pairs) == 1:
                    delta_by_model[model_name][slot]['spread_note'] = \
                        ONE_FOLD_NOTE

    # --- Phase D: Family analysis ---
    family_analysis = analyze_families(reps, labels, groups, folds, models,
                                       individual_results)

    # --- Phase E: CKA correlation ---
    correlation = {}
    if not skip_correlation:
        names, matrix = compute_correlation_matrix(reps)
        if len(names) >= 2:
            correlation = {
                'rep_names': names,
                'cka_matrix': matrix.tolist(),
            }

    return {
        'n_reactions': n,
        'n_groups': n_groups,
        'n_folds': actual_folds,
        'models': list(models),
        'metric': metric[0],
        'lower_is_better': bool(metric[2]),
        'selection_scope': 'fold_train_rows',
        'outer_split': {'splitter': 'nested_cv_outer_folds', 'seed': SEED,
                        'n_folds': actual_folds,
                        'n_val_per_fold': [int(len(va)) for _, va in folds]},
        'group_key': data['metadata'].get('group_key', 'unknown'),
        'representation_groups': group_record,
        'refused_representations': refused,
        'missing_representations': missing,
        'cgr_reps_available': sorted(cgr_available),
        'individual_results': individual_results,
        'hybrid_comparison': hybrid_comparison,
        'delta_by_model': delta_by_model,
        'family_analysis': family_analysis,
        'correlation_matrix': correlation,
    }


def _get_family(rep_name):
    """Return family name for a rep, or 'other'."""
    for fam_name, fam_def in FAMILIES.items():
        if rep_name in fam_def['cgr'] or rep_name in fam_def['smiles']:
            return fam_name
    return 'other'



# =============================================================================
# ANALYSIS PRINTING
# =============================================================================

SLOTS = ('base1', 'base2', 'hybrid1', 'hybrid2')


def _fmt(x, spec='.2f'):
    return '---' if x is None or x != x else format(x, spec)


def _search_failures_of(ds):
    """Every recorded per-fold search failure of one dataset's result."""
    hc = (ds or {}).get('hybrid_comparison') or {}
    return [f for g in GROUPS for f in (hc.get(g) or {}).get(
        'search_failures', [])]


def print_analysis(all_results):
    """Print the CGR comparison, one dataset at a time (never averaged across
    datasets)."""
    print(f"\n{'=' * 100}")
    print(f"CGR vs SMILES COMPARISON ANALYSIS")
    print(f"{'=' * 100}")

    for ds_name, ds in all_results.items():
        if ds is None or 'error' in ds:
            continue
        print(f"\n  Dataset: {ds_name} ({ds['n_reactions']} reactions, "
              f"{ds['n_groups']} groups, {ds['n_folds']} outer fold(s))")
        if ds['n_folds'] == 1:
            print(f"  ({ONE_FOLD_NOTE}: every number below is one held-out "
                  f"split)")

        # The two groups as used: intended, missing, refused.
        for g, gr in (ds.get('representation_groups') or {}).items():
            print(f"  {g:12s} {len(gr['used']):2d} of {len(gr['intended']):2d} "
                  f"representations ({gr['title']})")
            for r, why in sorted(gr['missing'].items()):
                print(f"      missing {r}: {why}")
            for r, why in sorted(gr['refused'].items()):
                print(f"      refused {r}: {why}")
        for f in _search_failures_of(ds):
            print(f"  SEARCH FAILED: {f['dataset']} {f['group']} fold "
                  f"{f['fold']} model {f['model']}: "
                  f"{[e.get('error') for e in f['errors']]}")

        # Table 1: Family comparison. One row per model and family. The two
        # representations are picked per fold on that fold's training rows;
        # the RMSEs are the means of their held-out scores.
        print(f"\n  --- Family Head-to-Head (held-out RMSE, lower is better) ---")
        print(f"  {'Model':9s} {'Family':12s} {'CGR RMSE':>9s} "
              f"{'SMILES RMSE':>12s}  folds CGR lower / folds")
        for model_name, fams in ds.get('family_analysis', {}).items():
            for fam_name, fam in fams.items():
                if 'not_compared' in fam:
                    print(f"  {model_name:9s} {fam_name:12s} not compared: "
                          f"{fam['not_compared']}")
                    continue
                print(f"  {model_name:9s} {fam_name:12s} "
                      f"{_fmt(fam['cgr_rmse_mean']):>9s} "
                      f"{_fmt(fam['smiles_rmse_mean']):>12s}  "
                      f"{fam['n_folds_cgr_wins']} / {fam['n_folds']}")

        # Table 2: Group comparison. One row per model and slot: held-out
        # RMSE averaged over the folds both groups scored.
        delta = ds.get('delta_by_model', {})
        if delta:
            print(f"\n  --- Group Comparison (held-out RMSE per fold, "
                  f"averaged over the folds both groups scored) ---")
            print(f"  {'Model':9s} {'Slot':8s} {'without_cgr':>12s} "
                  f"{'with_cgr':>11s} {'Delta':>9s} {'folds':>5s}")
            for model_name in sorted(delta):
                for slot in SLOTS:
                    d = delta[model_name].get(slot)
                    if d is None:
                        continue
                    print(f"  {model_name:9s} {slot:8s} "
                          f"{d['without_cgr_rmse_mean']:>12.2f} "
                          f"{d['with_cgr_rmse_mean']:>11.2f} "
                          f"{d['rmse_delta']:>+9.2f} {d['n_folds_used']:>5d}")
            print("  (Delta = with_cgr minus without_cgr; negative = CGR "
                  "helped)")

        # Table 3: CGR selection, counted over the hybrids built per fold.
        hc = ds.get('hybrid_comparison', {})
        cgr_sel = (hc.get('with_cgr') or {}).get('cgr_selection', {})
        if cgr_sel:
            print(f"\n  --- CGR Feature Selection (with_cgr hybrids, "
                  f"2 per fold per model) ---")
            for model_name, by_rep in sorted(cgr_sel.items()):
                for rep_name, info in sorted(by_rep.items()):
                    feats = [p['n_features'] for p in info['per_fold']
                             if p['n_features'] > 0]
                    print(f"    {model_name:9s} {rep_name:25s} given features "
                          f"in {info['n_hybrids_with_features']} of "
                          f"{info['n_hybrids_built']} hybrids"
                          + (f" ({min(feats)}-{max(feats)} features)"
                             if feats else ''))

        # Table 4: CKA correlation
        corr = ds.get('correlation_matrix', {})
        if corr.get('rep_names'):
            names = corr['rep_names']
            matrix = corr['cka_matrix']
            print(f"\n  --- Linear CKA Correlation ---")
            header = f"  {'':25s}" + ''.join(f"  {n[:10]:>10s}" for n in names)
            print(header)
            for i, n in enumerate(names):
                row = f"  {n:25s}"
                for j in range(len(names)):
                    row += f"  {matrix[i][j]:>10.3f}"
                print(row)

        # Per-model summary: the best hybrid of each group (slot hybrid1).
        for model_name in sorted(delta):
            d = delta[model_name].get('hybrid1')
            if d is None:
                continue
            print(f"\n  {model_name}: best hybrid per fold, without_cgr "
                  f"{d['without_cgr_rmse_mean']:.2f} -> with_cgr "
                  f"{d['with_cgr_rmse_mean']:.2f} RMSE "
                  f"({d['rmse_delta']:+.2f}, {d['n_folds_used']} folds)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CGR vs SMILES representation comparison for reaction '
                    'yield prediction. Tests whether CGR reps add unique '
                    'signal beyond SMILES-based representations.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated datasets (default: all 3)')
    # 2026-10-05 (handoff 2.4 step 14): no literal fold count. The default is
    # get_config's n_splits under --mode full (5, or KIRBY_LOCAL_FOLDS), the
    # one setting every experiment reads. 1 = a single grouped held-out split.
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help="Outer folds (capped by #groups; default: get_config's n_splits). "
             'The single representations and the per-fold searches use the '
             'same folds. 1 = one grouped held-out split.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path (default: tests/results/comparisons/'
             'opig_cgr_comparison.json)')
    parser.add_argument(
        '--skip-hybrid', action='store_true',
        help='Skip hybrid building (individual rep eval only)')
    parser.add_argument(
        '--skip-correlation', action='store_true',
        help='Skip CKA correlation analysis')
    # Accepted only because the cluster launcher still passes it
    # (submit_everything.sh OPIG_JOBS row 'cgr_comparison ... PM'). It has no
    # effect since 2026-10-04: the per-fold searches run on the decided grid
    # with no pruning schedule, as search_on_fold does by default, because a
    # schedule votes decided members out of the importance, evaluator,
    # allocation and representation-count lists during the search.
    parser.add_argument(
        '--pruning-mode', type=str, default=None,
        help='No effect (see the comment above it).')

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
    datasets = (args.datasets.split(',') if args.datasets else ALL_DATASETS)
    if args.n_folds is None:
        from experiments._shared import get_config
        args.n_folds = int(get_config(argparse.Namespace(
            mode='full', distance_metric=args.distance_metric))['n_splits'])
    if args.n_folds < 1:
        parser.error(f'--n-folds {args.n_folds}: must be 1 or more')

    print("=" * 70)
    print("CGR vs SMILES REPRESENTATION COMPARISON")
    print(f"  Phase 3 of CGR_EMBEDDING_RESEARCH.md")
    print(f"  Datasets: {datasets}")
    print(f"  Models: each dataset's two decided models")
    print(f"  Groups: {GROUP_TITLES['without_cgr']} vs "
          f"{GROUP_TITLES['with_cgr']}")
    print(f"  Max outer folds: {args.n_folds}")
    print(f"  Hybrid: {'skip' if args.skip_hybrid else 'yes'}")
    print(f"  CKA: {'skip' if args.skip_correlation else 'yes'}")
    if args.pruning_mode:
        print(f"  --pruning-mode {args.pruning_mode} ignored: per-fold "
              f"searches run unpruned on the decided grid")
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_cgr_comparison.json')

    # Each dataset is written to <output>.partial.jsonl as soon as it finishes
    # and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # dataset, because the one JSON below was written after the last. A dataset
    # that raised or returned {'error'} is not stored, is retried next time,
    # and makes the script exit 1 after the JSON is written.
    # 'selection' in the identity: a partial file written by an earlier
    # version of this script is never read back as this one's. Bumped
    # 2026-10-05: the groups changed (answer 11), so neither the whole-dataset
    # version's nor the 2026-10-04 version's partial files are read. The same
    # file also holds each group's finished folds and each representation's
    # single scores, stored as they finish (dirty-run bug 12).
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    _ck = script_checkpoint('opig_cgr_comparison', output_path, args,
                            exclude=('datasets', 'output', 'pruning_mode'),
                            selection='fold_standard_selection_2026_10_05',
                            groups='pruned10_minus_cgr_morgan_vs_plus_2_cgr',
                            family_extra_reps=list(FAMILY_EXTRA_REPS))
    _failed = []

    all_results = {}
    for ds_name in datasets:
        try:
            if _ck.done(dataset=ds_name):
                result = _ck.get(dataset=ds_name)
                print(f"  {ds_name}: finished in an earlier attempt, read back")
            else:
                result = evaluate_dataset(
                    ds_name, args.n_folds,
                    skip_hybrid=args.skip_hybrid,
                    skip_correlation=args.skip_correlation, ck=_ck)
                # A dataset with a failed per-fold search is written to the
                # JSON with its failures but not stored, so the next attempt
                # retries those folds (the finished ones are read back).
                if not _search_failures_of(result):
                    _ck.put(result, dataset=ds_name)
            if is_failed_item(result) or _search_failures_of(result):
                _failed.append(ds_name)
            all_results[ds_name] = result
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(ds_name)
            # 2026-10-05: the failure is written to the result file too, not
            # only printed (an exception that escaped the per-fold search's
            # own record, e.g. an undecided grid or a stale result store).
            # An {'error'} entry is never stored in the partial file, so the
            # dataset is retried on the next attempt.
            all_results[ds_name] = {
                'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}

    # Analysis
    if all_results:
        print_analysis(all_results)

    # Save
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Replace non-finite floats (NaN/inf) with null before serialising —
    # undefined metrics at tiny n should be JSON null, not NaN (invalid JSON).
    def _nan_to_none(o):
        if isinstance(o, float):
            return None if (o != o or o in (float('inf'), float('-inf'))) else o
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_nan_to_none(v) for v in o]
        return o

    with open(p, 'w') as f:
        json.dump(_nan_to_none(convert_numpy(all_results)), f, indent=2)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'dataset')


if __name__ == '__main__':
    main()
