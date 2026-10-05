#!/usr/bin/env python3
"""
Graphinity DDG Prediction Experiment (OPIG B2)

Replicates Hummer et al. 2025 "Investigating the Volume and Diversity of
Data Needed for Generalizable Antibody-Antigen DDG Prediction" —
tests whether sequence-based reps + simple ML can approach the
structure-based Graphinity EGNN for DDG prediction.

Scientific questions:
  1. How does transfer from synthetic → experimental compare?
  2. Does hybrid augmentation help for DDG prediction?
  3. How far are simple rep→model approaches from the structure-based EGNN?

Conditions:
  - transfer_foldx:  Train on FoldX synthetic (--max-foldx rows, default 5000),
                     evaluate on ALL experimental SKEMPI. Base representations
                     only: no hybrid search runs in this condition.
  - experimental:    Train/eval on experimental SKEMPI only (ddg_combined).
                     Per outer fold, a search on that fold's TRAINING rows
                     picks the 2 best base representations and the 2 best
                     hybrids for each of ddg_combined's two decided downstream
                     models, and all four are scored on the held-out fold.

Both conditions use ddg_combined's two decided downstream models and, where a
search runs, its decided grid (DECISIONS.md 2026-10-03, through
experiments/decided_grid.py). Augmentation is cut on ddg_combined, so nothing
here is an augmentation donor.

Removed 2026-09-04: within_foldx and within_rosetta, which trained and scored
inside graphinity_foldx and graphinity_rosetta. Neither dataset is registered,
so both conditions were skipped on every run since they were written; both
datasets were cut on 2026-07-09 because their labels are FoldX and Rosetta
output rather than measurements. The question they answered — can sequence
representations predict simulated binding free energy change — is dropped with
them. FoldX rows are still loaded for transfer_foldx, which trains on them and
scores on measured SKEMPI.

Paper baselines (Hummer et al. 2025, Nature Computational Science):
  Graphinity EGNN (structure-based):  Pearson r ~0.87 on experimental test
  Key finding: orders of magnitude more experimental data needed for
  generalizable DDG prediction.

Usage:
    # Smoke test (~3 min)
    python tests/experiments/opig_graphinity_ddg.py \\
        --conditions experimental --n-folds 2 --reps onehot,mutation_features

    # Full evaluation
    python tests/experiments/opig_graphinity_ddg.py

Reference:
    Hummer et al. (2025). Nature Computational Science.
    Zenodo: https://doi.org/10.5281/zenodo.15384945
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories,
    fold_standard_selection, STANDARD_N_BASE, STANDARD_N_HYBRIDS,
    build_fold_hybrid, fold_hybrid_scoring_kwargs,
    _resolve_dataset_metric,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
#
# Removed 2026-10-04 (the owner's per-fold rules). Everything hybrid in this
# file used to hang off ONE search over every row of ddg_combined, read back in
# three ways: fold_reselection on build_fold_reselection_input, the
# whole-dataset top-2 block (top_base_reps / top_hybrids /
# select_model_rep_pairs / hybrid_deltas), and a top-2-general-plus-top-1-
# distance set re-ranked fold by fold out of that same stored search
# (get_rule_hybrids). A fold's hybrid was therefore chosen with that fold's own
# rows inside the search. The selection below is run on each outer fold's
# training rows instead, and nothing reads a whole-dataset search.

ALL_CONDITIONS = ['experimental', 'transfer_foldx']

# The measured dataset. Both conditions score on its rows, and it is the
# dataset whose decided grid every search here runs on.
DDG_DATASET = 'ddg_combined'

# Representations come from DATASET_REPS['ddg_combined'] — the pool the
# rep-pruning step approved and precompute builds. The hand-written list this
# replaced asked for blosum, esm2, ablang2 and prot_t5, none of which are in
# that pool and none of which this experiment does anything special with; it
# also skipped 5 of the 8 that are.
#
# 'aac' is out since 2026-10-05. It is ddg_combined's augmentation
# representation. The 2026-10-04 edit took it in through approved_rep_pool,
# which always appends the augmentation list, and scored it as a ninth
# representation. The rule (handoff 2.4 step 9) is the 'general' list, plus
# the 'augmentation' list only where DECISIONS.md Step 8 keeps an augmentation
# condition. Step 8 cuts augmentation on ddg_combined, so aac is neither a
# representation nor a donor here. The check reads the record.
#
# Both conditions score on measured SKEMPI (transfer_foldx trains on FoldX rows
# and evaluates on ddg_combined), so one pool covers both.
def _default_reps():
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments.decided_grid import decided_aug_conditions
    entry = DATASET_REPS[DDG_DATASET]
    general = list(entry['general'])
    aug = list(entry.get('augmentation', []))
    kept = [c for c in decided_aug_conditions(DDG_DATASET, 'general')
            if c != 'none']
    if kept:
        # Not reached on ddg_combined today. The donors would have to go to
        # fold_standard_selection as augmentations, which this file does not do.
        raise ValueError(
            f"{DDG_DATASET}: Step 8 now keeps augmentation {kept}, but this "
            f"script passes no augmentation donors ({aug}).")
    print(f"  [reps] {DDG_DATASET}: {len(general)} general representations; "
          f"{aug} left out (Step 8 cuts augmentation on {DDG_DATASET})")
    return general

# The downstream models are ddg_combined's two decided ones (DECISIONS.md
# Step 1: svm+dnn), read from the record rather than written out here, so this
# file cannot disagree with the search it runs. Both conditions score on
# measured ddg_combined rows, so transfer_foldx uses the same two.
#
# It used to be DOWNSTREAM_MODELS, the five-model list (rf, xgboost, svm, dnn,
# lightgbm), and the one search it ran was asked for rf — not one of the two.
def _decided_models():
    """ddg_combined's two decided downstream models, in the record's order."""
    from experiments.decided_grid import decided_models
    return [m for m in decided_models(DDG_DATASET, 'general') if m != 'knn']


# The number of outer folds (2026-10-05, handoff 2.4 step 14). One setting:
# kirby.find_best_hybrid.DEFAULT_N_FOLDS, the constant get_config's
# config['n_splits'] takes under --mode full, and KIRBY_LOCAL_FOLDS over it the
# same way get_config applies it. --n-folds overrides both. It used to be a
# literal 5 in the --n-folds default. 1 is a single grouped held-out split.
def _default_n_outer_folds():
    from kirby.find_best_hybrid import DEFAULT_N_FOLDS
    lf = os.environ.get('KIRBY_LOCAL_FOLDS', '').strip()
    return int(lf) if lf else int(DEFAULT_N_FOLDS)


def _scale(X_tr, X_va):
    """Scale on the training rows: 0/1 features untouched, the rest
    standardised (answer 17, 2026-10-05; kirby.scaling.FeatureScaler).

    Replaces run_baseline_survey._maybe_scale here, which left a whole matrix
    alone only when EVERY feature was 0/1 and otherwise standardised all of
    them, 0/1 features included.
    """
    from kirby.scaling import FeatureScaler
    sc = FeatureScaler().fit(X_tr)
    X_tr_s = np.nan_to_num(sc.transform(X_tr), nan=0.0, posinf=1e10,
                           neginf=-1e10)
    X_va_s = np.nan_to_num(sc.transform(X_va), nan=0.0, posinf=1e10,
                           neginf=-1e10)
    return X_tr_s, X_va_s


# Whether a matrix wider than 500 features is cut to its 200 highest-ranked
# features (LightGBM gain on the training rows, run_baseline_survey.
# _filter_high_dim_per_fold) before it is fitted and scored. Off since
# 2026-10-05. With it on, a hybrid at ddg_combined's decided budgets (100, 250
# or 500 features) is fitted at its full width, while esm2_layer24 (1280
# features), esm2_concat (2560) and the other wide representations are fitted
# on 200. The rank-1 hybrid against the rank-1 representation would then
# differ in two ways at once: how the features were chosen, and how many there
# are. The per-fold search ranks every representation at its full width, so
# off is also what matches the search. One setting for every fit in this file.
CUT_WIDE_REPRESENTATIONS = False


def _maybe_cut_wide(X_tr, X_va, y_tr):
    """The wide-representation cut above, applied only when it is switched on."""
    if not CUT_WIDE_REPRESENTATIONS:
        return X_tr, X_va
    from run_baseline_survey import _filter_high_dim_per_fold
    X_tr, X_va, _ = _filter_high_dim_per_fold(X_tr, X_va, y_tr, 'regression')
    return X_tr, X_va


# Whether the experimental condition also scores every general representation
# with each decided model on every outer fold (the table under 'results').
# Off since 2026-10-05. Its headline numbers were the maximum over those 16
# held-out scores (8 representations x 2 models), a winner picked on the rows
# it is reported on (the same fault as dirty-run bug s06). The 2 base
# representations per fold that R4 asks for are already in
# 'standard_selection', chosen on each fold's training rows. Switched on, the
# table is written as a description only; nothing reads a maximum from it.
SCORE_EVERY_REPRESENTATION = False


def _missing_reps_error(rep_names, got, what):
    """An error record when any requested representation did not come back,
    else None (2026-10-05, dirty-run bug 9).

    _require_rows cannot see the 429-row ddg_combined cache: compute_or_load_reps
    drops any cached matrix whose row count is not the dataset's, prints
    'cache shape mismatch ... recomputing', and recomputes it. When that
    recompute fails, the representation is simply absent from what it returns.
    Absent used to mean "left out and the run continues on the rest"; it now
    means this condition is not measured, the record goes into the result file,
    the condition is not stored as finished, and the script exits 1.
    """
    missing = sorted(set(rep_names) - set(got))
    if not missing:
        return None
    msg = (f"{what}: {len(missing)} of {len(rep_names)} requested "
           f"representation(s) were not computed: {missing}. A cached matrix "
           f"whose row count differs from the dataset's is dropped and "
           f"recomputed by experiments/rep_cache.py; look above for 'cache "
           f"shape mismatch' and for the recompute's error. Nothing was "
           f"measured for this condition.")
    print(f"  {msg}")
    return {'error': msg, 'representations_not_computed': missing,
            'representations_computed': sorted(got)}


def _require_rows(reps, n_rows, what):
    """Refuse any representation matrix whose row count is not the dataset's.

    Dirty-run bug 9 (2026-10-05): a local ddg_combined cache held 429 rows
    while the dataset loads 534. A matrix with the wrong row count would pair
    features with another molecule's label, so this stops the run and names
    the representation instead of continuing.
    """
    bad = {r: tuple(np.shape(X)) for r, X in reps.items()
           if np.ndim(X) != 2 or np.shape(X)[0] != n_rows}
    if bad:
        raise ValueError(
            f"{what}: {len(bad)} representation matrix(es) do not have one row "
            f"per molecule ({n_rows} rows expected): {bad}. Delete the cached "
            f"matrix and recompute it; do not run on it.")


# The smoke-only search shrink (KIRBY_SMOKE_N: tiny budgets, rf as the only
# internal evaluator, lightgbm_gain as the only importance method) is gone with
# the whole-dataset search. Every axis now comes from the decided grid, and a
# named axis the record cut raises instead of running a different search.
# KIRBY_SMOKE_N still subsamples the rows through _smoke_subsample.

# Condition → dataset name mapping
CONDITION_DATASETS = {
    'experimental':   DDG_DATASET,
    'transfer_foldx': None,  # handled specially: train on synthetic, eval on experimental
}

# Paper baselines (Hummer et al. 2025, Nature Computational Science)
# IMPORTANT: The paper's core finding is that random splits inflate performance.
# Stringent CDR identity cutoffs reveal much lower generalization.
# Numbers below are from the repo's results notebooks (Figure-2b, Figure-2c).
PAPER_BASELINES = {
    'Graphinity EGNN synthetic FoldX (random)':     {'pearson': 0.896},
    'Graphinity EGNN synthetic FoldX (cutoff_100)':  {'pearson': 0.894},
    'Graphinity EGNN synthetic FoldX (cutoff_90)':   {'pearson': 0.893},
    'Graphinity EGNN synthetic FoldX (cutoff_70)':   {'pearson': 0.894},
    # Experimental: -RevMuts -NonBinders (closest to ddg_combined)
    'Graphinity EGNN experimental (random)':         {'pearson': 0.525},
    'Graphinity EGNN experimental (cutoff_100)':     {'pearson': 0.271},
    'Graphinity EGNN experimental (cutoff_90)':      {'pearson': 0.291},
    'Graphinity EGNN experimental (cutoff_70)':      {'pearson': 0.216},
    # Experimental: +NonBinders variant (main Figure-2c plot)
    'Graphinity EGNN exp+nonbinders (random)':       {'pearson': 0.382},
    'Graphinity EGNN exp+nonbinders (cutoff_70)':    {'pearson': 0.040},
    # Reference: FoldX computed vs experimental correlation
    'FoldX vs experimental':                         {'pearson': 0.46},
}
# Note: KIRBy groups by PDB complex (≈cutoff_100). Synthetic FoldX is stable
# across cutoffs (~0.89), but experimental collapses from 0.53 → 0.22 at cutoff_70.
# The "0.87" number previously cited is from the +RevMuts variant with random splits.


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_ddg(X, labels, groups, model, folds):
    """Evaluate one (rep, model) for DDG regression on given outer folds.

    Single-pass CV: runs folds once, collecting per-fold metrics AND the
    held-out predictions, then computes global-concatenation Pearson r
    (paper's method: concatenate all held-out predictions, compute a single
    Pearson r on the full array). This guarantees per-fold-mean Pearson and
    pearson_global use the SAME feature space (incl. _maybe_cut_wide).

    folds: the (train_idx, val_idx) list, built once per condition by
    kirby.find_best_hybrid.nested_cv_outer_folds. It used to build its own
    GroupKFold through run_baseline_survey._make_cv; the per-fold hybrid
    selection splits with nested_cv_outer_folds, so passing one list in is what
    puts the single representations and the hybrids on the same held-out rows.

    Returns dict with pearson, spearman, r2, rmse as {metric: {mean, std}},
    plus pearson_global (single value, comparable to paper's reported numbers).
    """
    from scipy.stats import pearsonr
    from sklearn.base import clone
    from run_baseline_survey import _score

    # The matrix keeps its float32 (answer 18, 2026-10-05); it used to be cast
    # to float64 here.
    X = np.asarray(X)
    labels = np.asarray(labels, dtype=np.float64)

    fold_metrics = []
    all_preds = np.full_like(labels, np.nan)
    for train_idx, val_idx in folds:
        X_tr, X_va = X[train_idx], X[val_idx]
        X_tr, X_va = _maybe_cut_wide(X_tr, X_va, labels[train_idx])
        X_tr, X_va = _scale(X_tr, X_va)
        try:
            m = clone(model)
            m.fit(X_tr, labels[train_idx])
            preds = m.predict(X_va)
            all_preds[val_idx] = preds
            fold_metrics.append(_score(labels[val_idx], preds, 'regression'))
        except Exception as e:
            # Log rather than silently drop the fold (OPIG-18) — a missing fold
            # biases both the per-fold mean and pearson_global.
            print(f"      [fold failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

    if not fold_metrics:
        return None
    all_keys = fold_metrics[0].keys()
    result = {}
    for k in all_keys:
        vals = [f[k] for f in fold_metrics if k in f]
        # One scored fold has no spread; say so instead of writing std 0.0
        # (2026-10-05, handoff 2.4 step 14).
        result[k] = {'mean': float(np.mean(vals)),
                     'std': (float(np.std(vals)) if len(vals) > 1 else None),
                     'n_folds': len(vals)}
        if len(vals) == 1:
            result[k]['spread_note'] = 'one fold, no spread'

    valid = ~np.isnan(all_preds)
    if valid.sum() > 2:
        r_global, _ = pearsonr(labels[valid], all_preds[valid])
        result['pearson_global'] = float(r_global)
    else:
        result['pearson_global'] = float('nan')

    return result


def evaluate_transfer(rep_names, max_foldx=5000):
    """Transfer learning: train on FoldX synthetic, evaluate on experimental.

    Train on (optionally subsampled) FoldX synthetic, test on ALL experimental.
    Groups experimental by PDB complex for reporting variance.

    NO SEARCH RUNS HERE, and none ever did. Every number is one base
    representation fitted once on the synthetic rows and scored once on the
    measured ddg_combined rows: no hybrid, no folds, no recipe. What changed on
    2026-10-04 is the models — ddg_combined's two decided ones instead of the
    five-model DOWNSTREAM_MODELS list, since the rows it scores are
    ddg_combined's.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.base import clone

    print(f"\n{'=' * 60}")
    print(f"Condition: transfer_foldx (train=synthetic, test=experimental)")
    print(f"{'=' * 60}")

    from kirby.datasets.graphinity import load_graphinity_dataset
    train_data = load_graphinity_dataset(preset='foldx', max_samples=max_foldx)
    test_data = load_dataset(DDG_DATASET)

    from experiments._shared import _smoke_subsample
    train_data = _smoke_subsample(train_data)
    test_data = _smoke_subsample(test_data)

    # Guard: unpaired sequences → empty strings
    for d in [train_data, test_data]:
        if d['light_seqs'] is None:
            d['light_seqs'] = [''] * len(d['heavy_seqs'])

    # PDB-level leakage filter (P0-5): FoldX synthetic is built from the same
    # SAbDab PDBs SKEMPI uses. Drop synthetic rows whose PDB also appears
    # in the experimental test set; otherwise sequences leak verbatim into
    # `pearson_global` and overstate transfer.
    test_pdbs = set()
    for key in ('pdb_ids', 'pdb'):
        if key in test_data and test_data[key] is not None:
            test_pdbs.update(str(p).upper() for p in test_data[key])
    if not test_pdbs:
        # Silence here means the filter did nothing and every number below is a
        # transfer score with the same complexes on both sides.
        raise ValueError(
            "transfer_foldx: the experimental test set carries no PDB "
            "identifiers, so the overlap filter cannot run. Without it the "
            "synthetic training rows share complexes with the test rows and "
            "the reported correlation is not a transfer measurement.")
    if test_pdbs:
        train_pdb_col = train_data.get('pdb_ids', train_data.get('pdb'))
        if train_pdb_col is None:
            raise ValueError(
                "transfer_foldx: the synthetic training set carries no PDB "
                "identifiers, so overlapping complexes cannot be dropped.")
        if train_pdb_col is not None:
            train_pdb_arr = np.asarray([str(p).upper() for p in train_pdb_col])
            keep_mask = ~np.isin(train_pdb_arr, list(test_pdbs))
            n_removed = int((~keep_mask).sum())
            if n_removed:
                for fld in ('heavy_seqs', 'light_seqs', 'labels',
                            'mutations', 'pdb_ids', 'pdb', 'groups'):
                    if fld in train_data and train_data[fld] is not None:
                        col = train_data[fld]
                        if hasattr(col, '__len__') and len(col) == len(keep_mask):
                            if isinstance(col, np.ndarray):
                                train_data[fld] = col[keep_mask]
                            else:
                                train_data[fld] = [v for v, k in zip(col, keep_mask) if k]
                print(f"  PDB leakage filter: removed {n_removed} synthetic rows "
                      f"whose PDB is in the experimental test set")

    n_train = len(train_data['labels'])
    n_test = len(test_data['labels'])
    print(f"  Train: {n_train} synthetic FoldX samples")
    print(f"  Test: {n_test} experimental SKEMPI samples")

    # Generate representations for both sets (with disk caching)
    from experiments.rep_cache import compute_or_load_reps

    reps_train = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=train_data['heavy_seqs'],
        light_seqs=train_data['light_seqs'],
        mutations=train_data.get('mutations'),
        dataset_tag=f"graphinity_foldx_{n_train}",
    )
    reps_test = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=test_data['heavy_seqs'],
        light_seqs=test_data['light_seqs'],
        mutations=test_data.get('mutations'),
        dataset_tag=f"ddg_combined_{n_test}",
    )

    _require_rows(reps_train, n_train, 'transfer_foldx synthetic training rows')
    _require_rows(reps_test, n_test, f'transfer_foldx {DDG_DATASET} test rows')

    # Every requested representation on both sides, or this condition is a
    # recorded failure (2026-10-05, dirty-run bug 9). See _missing_reps_error.
    common_reps = sorted(set(reps_train) & set(reps_test))
    err = _missing_reps_error(rep_names, common_reps,
                              'transfer_foldx (synthetic training rows and '
                              f'{DDG_DATASET} test rows)')
    if err is not None:
        return dict(err, condition='transfer_foldx')
    not_computed = []

    y_train = train_data['labels']
    y_test = test_data['labels']
    # ddg_combined's two decided models: the rows scored below are its rows.
    models = get_model_factories(_decided_models(), 'regression',
                                 n_samples=n_train)

    print(f"\n  Evaluating {len(reps_train)} reps × {len(models)} models "
          f"(train-on-synthetic, test-on-experimental)...")
    results = {}
    for rep_name in sorted(reps_train.keys()):
        X_tr = reps_train[rep_name]
        X_te = reps_test[rep_name]
        # The same wide-representation setting as every other fit here
        # (CUT_WIDE_REPRESENTATIONS, off since 2026-10-05).
        X_tr_f, X_te_f = _maybe_cut_wide(X_tr, X_te, y_train)
        # Hoist scaling out of inner model loop — same scaler for every model.
        X_tr_s, X_te_s = _scale(X_tr_f, X_te_f)
        rep_results = {}
        for model_name, factory in models.items():
            try:
                m = factory()
                m.fit(X_tr_s, y_train)
                preds = m.predict(X_te_s)
                r_p, _ = pearsonr(y_test, preds)
                r_s, _ = spearmanr(y_test, preds)
                rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
                # Single train/test pair — no fold variance to report.
                # Use flat values; consumers below handle both shapes via
                # `_metric_value()` (P1-11 fix).
                rep_results[model_name] = {
                    'pearson': float(r_p),
                    'spearman': float(r_s),
                    'rmse': rmse,
                    'pearson_global': float(r_p),
                }
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  rho={r_s:.4f}  rmse={rmse:.4f}")
            except Exception as e:
                print(f"    {rep_name:25s} {model_name:8s}  FAILED ({e})")
        if rep_results:
            results[rep_name] = {'dim': X_tr.shape[1], 'models': rep_results}

    return {
        'n_samples': n_test,
        'n_train': n_train,
        'condition': 'transfer_foldx',
        'dataset': 'graphinity_foldx -> ddg_combined',
        'results': results,
        'models': sorted(models),
        'representations_not_computed': not_computed,
        # Said in the file the reader opens, not only in the report: this
        # condition runs no hybrid search of any kind.
        'search_run': False,
        'search_note': ('one fit per representation per model on the synthetic '
                        'rows, scored once on the measured rows; no hybrid '
                        'search, no folds, no recipe'),
    }


# The seed that names the outer folds and each fold's search cache namespace.
# nested_cv_outer_folds and search_on_fold take the same one, or a fold's
# search is filed under a different fold's name.
_FOLD_SEED = 42


def _fmt(v, signed=False):
    """A score for printing, or '-' for None and NaN. A missing score is None
    in every block here, and a format spec on None raises TypeError."""
    if v is None or not isinstance(v, (int, float)) or v != v:
        return '-'
    return format(float(v), '+.4f' if signed else '.4f')


def _fit_score_on_fold(X_tr, X_va, y_tr, y_va, factory):
    """One (feature matrix, model) on one fold, scored on the held-out rows.

    The same preprocessing every single-representation row gets in
    evaluate_rep_ddg: the wide-representation feature cut and the scaler, both
    fitted on this fold's training rows. Returns the metrics dict plus the
    width it was fitted at, or None when the fit raised.
    """
    from sklearn.base import clone
    from run_baseline_survey import _score

    # No float64 cast (answer 18, 2026-10-05).
    X_tr = np.asarray(X_tr)
    X_va = np.asarray(X_va)
    # The wide-representation cut follows CUT_WIDE_REPRESENTATIONS (off since
    # 2026-10-05), the same for a hybrid and a base representation.
    X_tr, X_va = _maybe_cut_wide(X_tr, X_va, y_tr)
    X_tr, X_va = _scale(X_tr, X_va)
    try:
        m = clone(factory())
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)
    except Exception as e:
        print(f"      [fit failed: {type(e).__name__}: {e}]", flush=True)
        return None
    out = _score(y_va, preds, 'regression')
    out['n_features_fitted'] = int(X_tr.shape[1])
    # The held-out predictions, so a Pearson r over every fold's held-out rows
    # pooled (the paper's way of reporting) can be computed per slot. Popped
    # into the slot record by _take_predictions, not left in the metrics.
    out['_predictions'] = np.asarray(preds, dtype=float).tolist()
    return out


def _take_predictions(rec):
    """Move the held-out predictions out of rec['heldout'] into the slot."""
    hv = rec.get('heldout')
    rec['heldout_predictions'] = (hv.pop('_predictions', None)
                                  if isinstance(hv, dict) else None)


def _hybrid_recipe(entry):
    """One chosen hybrid's recipe, JSON-safe (the Blueprint object left out)."""
    return {k: entry.get(k) for k in (
        'label', 'allocation_method', 'importance_method', 'budget',
        'internal_evaluator', 'rep_count', 'rep_count_source', 'rep_subset',
        'aug_condition', 'score_mean', 'score_std', 'n_features',
        'n_folds_scored', 'n_folds_asked')}


def standard_selection_per_fold(reps, labels, groups, folds, *, ds_name,
                                models, metric_name, task_type,
                                lower_is_better, ck=None, on_fold=None):
    """2 base representations + 2 hybrids per model per outer fold, scored.

    For each outer fold and each decided downstream model,
    fold_standard_selection runs ONE search on that fold's TRAINING rows, on
    ddg_combined's decided grid (budgets, importance methods, allocation
    methods, internal evaluators, representation counts, models), and returns
    the 2 base representations and the 2 hybrid configurations it ranks highest
    for that model. Each of the four is then fitted on those same training rows
    and scored on the fold's held-out rows.

    Augmentation: none. DECISIONS.md Step 8 cuts it on ddg_combined, so no
    representation is passed as a donor and the search runs the 'none'
    condition only.

    Pruning: off (2026-10-05, handoff 2.1). No pruning schedule and no
    per-model aggressiveness is passed; the --pruning-mode it used to forward
    is ignored.

    ck / on_fold (2026-10-05, dirty-run bug 12): each finished fold is stored
    in the script's checkpoint under (condition, fold) and on_fold(block so
    far) is called, so the result file is rewritten after every fold. A fold
    whose search failed or raised is not stored and runs again next attempt.

    A failed per-fold search is recorded in 'missing' (dataset, fold, model,
    error) and that fold's slots are left empty. Nothing fills them.

    Returns the JSON block.
    """
    from kirby.find_best_hybrid import BUILTIN_METRICS

    metric_fn, _ = BUILTIN_METRICS[metric_name]
    factories = get_model_factories(models, task_type, n_samples=len(labels))

    fold_rows, missing = [], []
    n_search_failed = 0
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        if ck is not None and ck.done(condition='experimental', fold=fold_i):
            got_back = ck.get(condition='experimental', fold=fold_i)
            fold_rows += got_back['rows']
            missing += got_back['missing']
            print(f"  fold {fold_i}: finished in an earlier attempt, read back")
            continue
        rows_f, miss_f, failed_f = _one_fold_selection(
            reps, labels, groups, fold_i, train_idx, val_idx,
            n_outer_folds=len(folds), ds_name=ds_name, models=models,
            metric_name=metric_name, metric_fn=metric_fn, task_type=task_type,
            lower_is_better=lower_is_better, factories=factories)
        fold_rows += rows_f
        missing += miss_f
        if failed_f:
            n_search_failed += 1
        elif ck is not None:
            ck.put({'rows': rows_f, 'missing': miss_f},
                   condition='experimental', fold=fold_i)
        if on_fold is not None:
            on_fold({'per_fold': fold_rows, 'missing': missing,
                     'folds_done': fold_i + 1, 'n_outer_folds': len(folds),
                     'in_progress': True})

    return _selection_block(fold_rows, missing, n_search_failed, folds=folds,
                            ds_name=ds_name, models=models,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better)


def _fold_search_failures(reps, labels, groups, *, train_idx, fold_i,
                          n_outer_folds, ds_name, task_type, metric_name):
    """The error records of one fold's general search, read back from the
    in-process memo fold_standard_selection filled (same arguments, so no
    search is re-run). fold_standard_selection prints them but does not
    return them."""
    from experiments._shared import fold_searches
    try:
        s = fold_searches(reps, labels, groups, train_idx=train_idx,
                          fold_i=fold_i, n_outer_folds=n_outer_folds,
                          dataset_name=ds_name, augmentations=None,
                          task_type=task_type, metric=metric_name,
                          tier='general', seed=_FOLD_SEED)
        # The keys PrecomputeResults.record_failure writes (2026-10-05: this
        # used to ask for 'condition', which no record carries, so the failed
        # augmentation condition never reached the result file).
        return [{k: f.get(k) for k in ('dataset', 'tier', 'aug_condition',
                                       'error_type', 'error', 'traceback')}
                for f in (getattr(s, 'failures', None) or [])]
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {e}'}]


def _one_fold_selection(reps, labels, groups, fold_i, train_idx, val_idx, *,
                        n_outer_folds, ds_name, models, metric_name,
                        metric_fn, task_type, lower_is_better, factories):
    """One outer fold: select on its training rows, score on its held-out rows.

    Returns (rows, missing, search_failed).
    """
    rows_out, missing = [], []
    y_tr, y_va = labels[train_idx], labels[val_idx]
    train_reps = {r: X[train_idx] for r, X in reps.items()}
    val_reps = {r: X[val_idx] for r, X in reps.items()}
    train_groups = groups[train_idx] if groups is not None else None
    try:
        sel = fold_standard_selection(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_outer_folds, dataset_name=ds_name,
            models=list(models), augmentations=None,
            task_type=task_type, metric=metric_name, seed=_FOLD_SEED)
    except Exception as e:
        print(f"  fold {fold_i}: the selection raised "
              f"{type(e).__name__}: {e}")
        missing.append({'dataset': ds_name, 'fold': fold_i, 'model': None,
                        'kind': 'search_failed',
                        'reason': f'selection raised {type(e).__name__}: {e}'})
        return rows_out, missing, True

    search_failed = False
    if all(sel.get(m) is None for m in models):
        # fold_standard_selection maps every model to None when the fold's
        # search failed.
        search_failed = True
        errs = _fold_search_failures(
            reps, labels, groups, train_idx=train_idx, fold_i=fold_i,
            n_outer_folds=n_outer_folds, ds_name=ds_name,
            task_type=task_type, metric_name=metric_name)
        for m in models:
            missing.append({'dataset': ds_name, 'fold': fold_i, 'model': m,
                            'kind': 'search_failed',
                            'reason': "this fold's general search failed",
                            'errors': errs})
        return rows_out, missing, True

    for model_name in models:
        got = sel.get(model_name)
        factory = factories.get(model_name)
        if factory is None:
            missing.append({'dataset': ds_name, 'fold': fold_i,
                            'model': model_name, 'kind': 'no_estimator',
                            'reason': 'no estimator built for this model'})
            continue
        if got is None:
            missing.append({'dataset': ds_name, 'fold': fold_i,
                            'model': model_name, 'kind': 'search_failed',
                            'reason': "this fold's search produced "
                                      "nothing for this model"})
            search_failed = True
            continue
        row = {'dataset': ds_name, 'fold': fold_i, 'model': model_name,
               'n_train_rows': int(len(train_idx)),
               'n_heldout_rows': int(len(val_idx)),
               'selection_scope': got.get('selection_scope'),
               # The held-out rows' labels, in the order every slot's
               # 'heldout_predictions' follows (2026-10-05).
               'heldout_labels': np.asarray(y_va, dtype=float).tolist(),
               'base': [], 'hybrids': []}

        for i, b in enumerate(got['base_detail']['selected'], 1):
            rep = b.get('rep')
            # 'score' is what top_base_reps stores (the search's own mean
            # over its inner folds of this fold's training rows), not
            # 'score_mean', which is the hybrid rows' key.
            rec = {'slot': f'base{i}', 'rep': rep,
                   'score_train_rows': b.get('score'),
                   'dimension': b.get('dimension'),
                   'heldout': None}
            if rep in reps:
                rec['heldout'] = _fit_score_on_fold(
                    train_reps[rep], val_reps[rep], y_tr, y_va, factory)
            _take_predictions(rec)
            if rec['heldout'] is None:
                missing.append({'dataset': ds_name, 'fold': fold_i, 'model': model_name,
                                'slot': rec['slot'], 'reason':
                                f'representation {rep} scored nothing'})
            row['base'].append(rec)
        if not got['base_detail'].get('complete'):
            missing.append({
                'dataset': ds_name, 'fold': fold_i, 'model': model_name,
                'slot': 'base',
                'reason': got['base_detail'].get('shortfall_reason')})

        for i, e in enumerate(got['hybrids'], 1):
            rec = _hybrid_recipe(e)
            rec['slot'] = f'hybrid{i}'
            # The search's mean over its inner folds of this fold's training
            # rows, under the same key the base slots use.
            rec['score_train_rows'] = e.get('score_mean')
            rec['heldout'] = None
            bp = e.get('blueprint')
            if bp is None:
                missing.append({'dataset': ds_name, 'fold': fold_i, 'model': model_name,
                                'slot': rec['slot'], 'reason':
                                'the chosen configuration carries no '
                                'recipe to rebuild'})
                row['hybrids'].append(rec)
                continue
            try:
                X_tr_h, X_va_h, _ = build_fold_hybrid(
                    train_reps, val_reps, y_tr, bp,
                    groups=train_groups,
                    # WHICH representations a recipe searched over a
                    # subset of the pool used: the allocation scores from
                    # this fold's own search, carried on the recipe it
                    # returned. There is no whole-dataset search here to
                    # take them from.
                    precomputed_alloc_scores=getattr(bp, 'rep_scores',
                                                     None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=factory(), metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                rec['heldout'] = _fit_score_on_fold(
                    X_tr_h, X_va_h, y_tr, y_va, factory)
                _take_predictions(rec)
            except Exception as ex:
                print(f"  fold {fold_i} {model_name} {rec['slot']}: "
                      f"rebuild failed: {type(ex).__name__}: {ex}")
                missing.append({'dataset': ds_name, 'fold': fold_i, 'model': model_name,
                                'slot': rec['slot'], 'reason':
                                f'rebuild raised {type(ex).__name__}: {ex}'})
            row['hybrids'].append(rec)
        if len(got['hybrids']) < STANDARD_N_HYBRIDS:
            missing.append({
                'dataset': ds_name, 'fold': fold_i, 'model': model_name,
                'slot': 'hybrids',
                'reason': f"this fold's search offered "
                          f"{len(got['hybrids'])} configuration(s) for "
                          f"{model_name}, not {STANDARD_N_HYBRIDS}"})

        _b = ', '.join(
            f"{r['rep']}={_fmt((r['heldout'] or {}).get(metric_name))}"
            for r in row['base'])
        _h = ', '.join(
            f"{h['slot']}={_fmt((h['heldout'] or {}).get(metric_name))}"
            for h in row['hybrids'])
        print(f"  fold {fold_i} {model_name:8s} held-out {metric_name}: "
              f"representations {_b or '-'} | hybrids {_h or '-'}")
        # A second hybrid whose selected features equal the first's was
        # skipped by fold_standard_selection (answer 8); which ones, and the
        # hybrid each matched, are kept with the row.
        row['hybrids_skipped_identical_features'] = list(
            got.get('hybrids_skipped_identical_features') or [])
        rows_out.append(row)

    return rows_out, missing, search_failed


def _pooled_pearson(y, p):
    """Pearson r on pooled held-out labels and predictions; NaN under 3 rows
    or when either side is constant."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float('nan')
    return float(np.corrcoef(y, p)[0, 1])


def _selection_block(fold_rows, missing, n_search_failed, *, folds, ds_name,
                     models, metric_name, lower_is_better):
    """The JSON block from every fold's rows: per model and slot, the
    held-out score on each fold and its mean."""

    # Per model and slot, the held-out score on each fold and its mean. Pearson
    # r rides along because the paper's numbers are Pearson r; the selection
    # itself ranks on the dataset's own metric.
    summary = {}
    for model_name in models:
        slots = {}
        for row in fold_rows:
            if row['model'] != model_name:
                continue
            for item in row['base'] + row['hybrids']:
                d = slots.setdefault(item['slot'], {
                    'per_fold': [], 'what': []})
                hv = item.get('heldout') or {}
                d['per_fold'].append({
                    'fold': row['fold'],
                    metric_name: hv.get(metric_name),
                    'pearson': hv.get('pearson')})
                d['what'].append({'fold': row['fold'],
                                  'rep': item.get('rep'),
                                  'recipe': item.get('label')})
        # Per slot, Pearson r over every fold's held-out rows pooled: the
        # paper's way of reporting (2026-10-05). The slot is "the rank-1
        # representation" (or hybrid) of each fold, so the representation
        # behind it can differ from fold to fold; 'what' says which.
        pooled = {}
        for row in fold_rows:
            if row['model'] != model_name:
                continue
            for item in row['base'] + row['hybrids']:
                pr = item.get('heldout_predictions')
                yl = row.get('heldout_labels')
                if pr is None or yl is None or len(pr) != len(yl):
                    continue
                py = pooled.setdefault(item['slot'], ([], [], set()))
                py[0].extend(yl)
                py[1].extend(pr)
                py[2].add(row['fold'])
        for slot, d in slots.items():
            yl, pr, fset = pooled.get(slot, ([], [], set()))
            d['pearson_pooled'] = _pooled_pearson(yl, pr)
            d['pearson_pooled_n_folds'] = len(fset)
        for d in slots.values():
            vals = [x[metric_name] for x in d['per_fold']
                    if x[metric_name] is not None
                    and not np.isnan(x[metric_name])]
            pear = [x['pearson'] for x in d['per_fold']
                    if x['pearson'] is not None and not np.isnan(x['pearson'])]
            d['n_folds'] = len(vals)
            d[f'mean_{metric_name}'] = (float(np.mean(vals)) if vals
                                        else float('nan'))
            d['mean_pearson'] = float(np.mean(pear)) if pear else float('nan')
        # The rank-1 hybrid against the rank-1 representation, paired on the
        # fold they were both chosen and scored on. Both ranks come from that
        # fold's training rows, so no maximum is taken on held-out rows.
        paired = []
        for row in fold_rows:
            if row['model'] != model_name:
                continue
            hb = next((h for h in row['hybrids'] if h['slot'] == 'hybrid1'),
                      None)
            bs = next((b for b in row['base'] if b['slot'] == 'base1'), None)
            hv = (hb or {}).get('heldout') or {}
            bv = (bs or {}).get('heldout') or {}
            if (hv.get(metric_name) is None or bv.get(metric_name) is None):
                continue
            paired.append({'fold': row['fold'],
                           'hybrid1': hv[metric_name],
                           'base1': bv[metric_name],
                           'difference': hv[metric_name] - bv[metric_name]})
        diffs = [p['difference'] for p in paired if not np.isnan(p['difference'])]
        summary[model_name] = {
            'slots': slots,
            'hybrid1_minus_base1': {
                'per_fold': paired,
                'n_paired_folds': len(diffs),
                'mean': float(np.mean(diffs)) if diffs else float('nan'),
                'metric_name': metric_name,
                'lower_is_better': bool(lower_is_better),
            },
        }

    return {
        'rule': ('per outer fold and per decided downstream model: the 2 best '
                 'base representations and the 2 best hybrids, both ranked on '
                 "that fold's training rows"),
        'selection_scope': 'fold_train_rows',
        'dataset': ds_name,
        'models': list(models),
        'n_base': STANDARD_N_BASE,
        'n_hybrids': STANDARD_N_HYBRIDS,
        'n_outer_folds': len(folds),
        'fold_seed': _FOLD_SEED,
        'metric_name': metric_name,
        'lower_is_better': bool(lower_is_better),
        'augmentation': 'none',
        'augmentation_note': ('DECISIONS.md Step 8 cuts augmentation on '
                              'ddg_combined, so no representation is passed '
                              'as a donor and only the none condition runs'),
        'per_fold': fold_rows,
        'summary': summary,
        'missing': missing,
        # Folds whose per-fold search failed or raised: their slots are empty
        # and listed in 'missing' with the error; nothing was filled in.
        'n_folds_search_failed': int(n_search_failed),
    }


def evaluate_condition(condition, rep_names, n_folds, max_foldx=5000,
                       ck=None, on_fold=None):
    """Evaluate one condition: load data, generate reps, evaluate all.

    ck / on_fold: the script checkpoint and the per-fold result writer,
    handed to standard_selection_per_fold (2026-10-05, dirty-run bug 12).

    Returns dict with per-rep results. It carries an 'error' key when any
    fold's search failed, so the condition is not stored as finished and the
    script exits 1 after writing; the folds that did finish are stored.
    """
    if condition == 'transfer_foldx':
        return evaluate_transfer(rep_names, max_foldx=max_foldx)

    ds_name = CONDITION_DATASETS[condition]

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition} (dataset: {ds_name})")
    print(f"{'=' * 60}")

    data = load_dataset(ds_name)

    # A loader that returns pdb_ids but no groups gets a per-row 'groups' array
    # here, BEFORE smoke subsampling, so _smoke_subsample (which knows 'groups'
    # but not 'pdb_ids') keeps groups aligned with the subsampled labels and
    # sequences. Without this, groups would stay full-length and GroupKFold
    # would see inconsistent sample counts under KIRBY_SMOKE_N.
    if data.get('groups') is None and data.get('pdb_ids') is not None:
        pdb_ids = data['pdb_ids']
        unique_ids = {pid: i for i, pid in enumerate(sorted(set(pdb_ids)))}
        data['groups'] = np.array([unique_ids[pid] for pid in pdb_ids])

    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    if light_seqs is None:
        light_seqs = [''] * len(heavy_seqs)
    labels = data['labels']
    groups = data.get('groups')
    n = len(labels)

    print(f"  {n} samples, DDG range [{labels.min():.2f}, {labels.max():.2f}]")
    if groups is not None:
        print(f"  {len(np.unique(groups))} groups (PDB complexes)")

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        mutations=data.get('mutations'),
        dataset_tag=f"{ds_name}_{n}",
    )

    # One row per molecule in every matrix, or stop (dirty-run bug 9).
    _require_rows(reps, n, f'{ds_name} ({condition})')
    # Every requested representation, or this condition is a recorded failure
    # (2026-10-05). It used to return None when none came back, which the
    # checkpoint stored as finished; and to run on whatever subset came back.
    err = _missing_reps_error(rep_names, reps, f'{ds_name} ({condition})')
    if err is not None:
        return dict(err, condition=condition, dataset=ds_name, n_samples=n)
    not_computed = []

    if groups is None:
        # GroupKFold without real groups silently degenerates to KFold, which
        # violates the prohibition on random splits.
        raise ValueError(
            f"evaluate_condition({condition}): groups is None for "
            f"{ds_name}; refusing to fall back to np.arange(n) "
            "(would degenerate GroupKFold to KFold = random split).")

    from kirby.find_best_hybrid import nested_cv_outer_folds
    metric_name, task_type, lower_is_better = _resolve_dataset_metric(
        dataset_name=ds_name, labels=labels, metric='auto')
    models = _decided_models()

    # ONE fold list for everything this condition reports: the per-fold
    # selection searches on each fold's training rows, and the
    # single-representation table scores on the same held-out rows. Built by
    # nested_cv_outer_folds, which is also what search_on_fold names its
    # per-fold cache namespace from.
    folds = nested_cv_outer_folds(reps, labels, groups, n_folds, _FOLD_SEED,
                                  task_type)
    print(f"  {len(folds)} outer folds, seed {_FOLD_SEED}; "
          f"decided models {models}; search metric {metric_name}")

    # 2 base representations + 2 hybrids per model per fold, every one of them
    # chosen by a search on that fold's training rows. None when the whole
    # block raised, which this file has to keep running through.
    standard_block, selection_error = None, None
    try:
        standard_block = standard_selection_per_fold(
            reps, labels, groups, folds, ds_name=ds_name, models=models,
            metric_name=metric_name, task_type=task_type,
            lower_is_better=lower_is_better, ck=ck, on_fold=on_fold)
    except Exception as e:
        print(f"  per-fold selection failed: {type(e).__name__}: {e}")
        # Recorded in the result file, not only printed (2026-10-05).
        selection_error = f'{type(e).__name__}: {e}'
        standard_block = {'error': selection_error, 'dataset': ds_name,
                          'selection_scope': 'fold_train_rows'}

    # NO 'kirby_hybrid' entry is added to `reps`. It used to be one matrix: a
    # recipe refit on every row, its feature indices applied to every row, then
    # scored by GroupKFold over those same rows, so each fold's own labels had
    # chosen both the recipe and the features it was then measured on. The
    # hybrid numbers are the per-fold ones above and nothing else.

    # The every-representation table, on the decided models, only when
    # SCORE_EVERY_REPRESENTATION is on (off since 2026-10-05; see there).
    model_factories = get_model_factories(models, task_type, n_samples=n)
    results = {}
    if SCORE_EVERY_REPRESENTATION:
        print(f"\n  Every representation (a description; nothing is picked "
              f"from it): {len(reps)} reps x {len(model_factories)} models "
              f"x {len(folds)} folds...")
    for rep_name in (sorted(reps) if SCORE_EVERY_REPRESENTATION else []):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in model_factories.items():
            t0 = time.time()
            res = evaluate_rep_ddg(X, labels, groups, factory(), folds)
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                r_p = res['pearson']['mean']
                r_g = res.get('pearson_global', float('nan'))
                rho = res['spearman']['mean']
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"r={r_p:.4f}  r_g={r_g:.4f}  rho={rho:.4f}  "
                      f"({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {'dim': X.shape[1], 'models': rep_results}

    out = {
        'n_samples': n,
        'condition': condition,
        'dataset': ds_name,
        'results': results,
        'models': list(models),
        'n_outer_folds': len(folds),
        'fold_seed': _FOLD_SEED,
        # The per-fold standard selection: which models, which 2 base
        # representations and which 2 hybrids (allocation method, importance
        # method, budget, internal evaluator, representation count,
        # augmentation condition) were used on each fold, each one's held-out
        # score, and selection_scope 'fold_train_rows'. None when the block
        # raised. Beside 'results', never inside it: print_analysis loops
        # results.items() and reads rep_data['models'] for every value, so an
        # entry there with no 'models' key takes the whole table with it.
        'standard_selection': standard_block,
        'every_representation_scored': bool(SCORE_EVERY_REPRESENTATION),
        'representations': sorted(reps),
        'representations_not_computed': not_computed,
    }
    n_failed = ((standard_block or {}).get('n_folds_search_failed') or 0)
    if selection_error or n_failed:
        out['error'] = (selection_error or
                        f'{n_failed} of {len(folds)} fold search(es) failed; '
                        f"see standard_selection['missing']")
    return out


# =============================================================================
# ANALYSIS
# =============================================================================

def _metric_value(metrics, key):
    """Return the mean of a metric whether it's stored as a flat float or
    as {'mean': float, 'std': float}. Lets transfer_foldx (single train/test,
    no std) coexist with k-fold conditions (mean+std) without lying about
    a `std=0.0` for the former (P1-11).
    """
    v = metrics.get(key)
    if isinstance(v, dict):
        return v.get('mean', float('nan'))
    return float('nan') if v is None else float(v)


def print_analysis(all_results):
    """Paper baselines beside this file's numbers, one condition at a time.

    Rewritten 2026-10-05. It used to report, per condition, the maximum over
    every (representation, model) held-out score, twice: a 'Best rep' row and
    a 'best individual Pearson r_g' gap to the EGNN. Both were a winner picked
    on the rows it was reported on. Now:
      experimental:   per decided model, each slot of the per-fold selection
                      (base1, base2, hybrid1, hybrid2), each chosen on its
                      fold's training rows, with Pearson r over the held-out
                      rows of every fold pooled (the paper's way of reporting)
                      and its gap to the EGNN.
      transfer_foldx: every representation with each decided model, listed;
                      no row is picked, since this condition has no training
                      rows to pick on that are not also the FoldX rows.
    """
    print(f"\n{'=' * 100}")
    print(f"GRAPHINITY DDG COMPARISON (OPIG B2)")
    print(f"{'=' * 100}")

    print(f"\n  Paper baselines:")
    for method, metrics in PAPER_BASELINES.items():
        r_p = metrics.get('pearson', float('nan'))
        print(f"    {method}: Pearson r = {r_p:.2f}")

    # EGNN at cutoff_100 is the closest to KIRBy's grouping by PDB complex.
    egnn_r = PAPER_BASELINES['Graphinity EGNN experimental (cutoff_100)']['pearson']
    print(f"\n  Structure-based EGNN reference (cutoff_100): "
          f"Pearson r = {egnn_r:.3f}")

    for cond in ALL_CONDITIONS:
        cr = all_results.get(cond)
        if not cr:
            continue
        if cr.get('error') and not cr.get('standard_selection') \
                and not cr.get('results'):
            print(f"\n  {cond}: FAILED, nothing measured: {cr['error']}")
            continue
        if cond == 'transfer_foldx':
            print(f"\n  transfer_foldx: every representation, Pearson r on "
                  f"{cr.get('n_samples')} measured rows (one fit each; listed, "
                  f"none picked). Gap = EGNN minus this.")
            print(f"    {'Representation':25s} {'Model':>8s} {'Pearson r':>10s} "
                  f"{'Spearman':>10s} {'Gap':>8s}")
            for rep_name, rd in sorted((cr.get('results') or {}).items()):
                for model_name, m in rd['models'].items():
                    r_p = _metric_value(m, 'pearson')
                    print(f"    {rep_name:25s} {model_name:>8s} "
                          f"{_fmt(r_p):>10s} "
                          f"{_fmt(_metric_value(m, 'spearman')):>10s} "
                          f"{_fmt(egnn_r - r_p, signed=True):>8s}")
            continue
        block = cr.get('standard_selection') or {}
        print(f"\n  {cond}: per-fold selection slots, Pearson r over the "
              f"held-out rows of every fold pooled ({cr.get('n_samples')} "
              f"molecules, {block.get('n_outer_folds')} outer folds). "
              f"Gap = EGNN minus this.")
        print(f"    {'Model':>8s} {'Slot':>8s} {'Pearson r':>10s} "
              f"{'Folds':>6s} {'Gap':>8s}")
        for model in block.get('models', []):
            slots = (((block.get('summary') or {}).get(model) or {})
                     .get('slots') or {})
            for slot in sorted(slots):
                d = slots[slot]
                r = d.get('pearson_pooled')
                print(f"    {model:>8s} {slot:>8s} {_fmt(r):>10s} "
                      f"{d.get('pearson_pooled_n_folds', 0):>6d} "
                      f"{_fmt(None if r is None else egnn_r - r, signed=True):>8s}")


def print_standard_selection(all_results):
    """The per-fold standard selection, per condition and downstream model.

    One block per condition. For each decided downstream model it prints the 2
    base representations and the 2 hybrids as 4 slots, each slot's mean
    held-out score over the outer folds, and how many folds it scored. Every
    one of the four was ranked by a search on that fold's training rows, so no
    re-ranking or selection correction is applied to the scores here.

    Difference is the mean over folds of (rank-1 hybrid minus rank-1
    representation) on the same held-out fold, both ranks taken from that
    fold's training rows.

    It replaced a fold_reselection table on 2026-10-04. That table existed to
    undo the optimism of picking one winner from a search over every row; with
    the selection run per fold there is no such optimism to undo.
    """
    print(f"\n{'=' * 100}")
    print(f"PER-FOLD STANDARD SELECTION: 2 REPRESENTATIONS + 2 HYBRIDS PER MODEL")
    print(f"{'=' * 100}")
    for cond in ALL_CONDITIONS:
        cr = all_results.get(cond)
        if not cr:
            continue
        block = cr.get('standard_selection')
        if not block and cr.get('error'):
            print(f"\n  {cond}: FAILED, nothing measured: {cr['error']}")
            continue
        if not block:
            # transfer_foldx fits base representations only and has no folds;
            # any condition whose selection raised also lands here.
            print(f"\n  {cond}: no per-fold selection "
                  f"(this condition runs no search)")
            continue
        if block.get('error') and not block.get('per_fold'):
            print(f"\n  {cond}: per-fold selection raised: {block['error']}")
            continue
        metric_name = block.get('metric_name')
        metric_label = metric_name or 'the search metric'
        direction = ('lower better' if block.get('lower_is_better')
                     else 'higher better')
        print(f"\n  {cond}: {block.get('n_outer_folds')} outer folds, "
              f"selection on {block.get('selection_scope')}, augmentation "
              f"{block.get('augmentation')}")
        print(f"  scores in {metric_label} ({direction}) on the held-out fold")
        print(f"    {'Model':>8s} {'Slot':>8s} {'Mean':>9s} "
              f"{'MeanPearson':>12s} {'Folds':>6s}")
        for model in block.get('models', []):
            s = (block.get('summary') or {}).get(model) or {}
            for slot in sorted((s.get('slots') or {})):
                d = s['slots'][slot]
                print(f"    {model:>8s} {slot:>8s} "
                      f"{_fmt(d.get(f'mean_{metric_name}')):>9s} "
                      f"{_fmt(d.get('mean_pearson')):>12s} "
                      f"{d.get('n_folds', 0):>6d}")
            diff = s.get('hybrid1_minus_base1') or {}
            print(f"    {model:>8s} rank-1 hybrid minus rank-1 representation: "
                  f"{_fmt(diff.get('mean'), signed=True)} over "
                  f"{diff.get('n_paired_folds', 0)} folds")
        if block.get('n_folds_search_failed'):
            print(f"    {block['n_folds_search_failed']} fold search(es) "
                  f"failed; those folds have no slots")
        miss = block.get('missing') or []
        if miss:
            print(f"    {len(miss)} slot(s) not filled; "
                  f"results['standard_selection']['missing'] says why")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Graphinity DDG experiment (OPIG B2). '
                    'Tests sequence-based reps for DDG prediction.')
    parser.add_argument(
        '--conditions', type=str, default=None,
        help='Comma-separated conditions '
             f'(default: {",".join(ALL_CONDITIONS)})')
    parser.add_argument(
        '--reps', type=str, default=None,
        help="Comma-separated rep names (default: DATASET_REPS['ddg_combined']"
             "['general'], the 8 approved)")
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help='Number of outer folds. Default: find_best_hybrid.DEFAULT_N_FOLDS '
             '(KIRBY_LOCAL_FOLDS overrides), the same setting as '
             "get_config's n_splits. 1 = one grouped held-out split.")
    parser.add_argument(
        '--max-foldx', type=int, default=5000,
        help='Max samples for FoldX synthetic preset (default: 5000). '
             'FoldX raw size is ~943K rows; subsampling is essential.')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_graphinity_ddg.json)')
    # Kept so an old command line still parses; ignored with a printed line.
    # Pruning is off in every per-fold search (2026-10-05, handoff 2.1).
    parser.add_argument(
        '--pruning-mode', type=str, default=None,
        help='Ignored: pruning is off in every per-fold search (2026-10-05).')

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
    if args.pruning_mode not in (None, '', 'no_pruning'):
        print(f"  --pruning-mode {args.pruning_mode} ignored: pruning is off "
              f"in every per-fold search")
    if args.n_folds is None:
        args.n_folds = _default_n_outer_folds()
    conditions = (args.conditions.split(',') if args.conditions
                  else ALL_CONDITIONS)
    if args.reps:
        rep_names = args.reps.split(',')
        print(f"  [reps] --reps given: the {DDG_DATASET} general list is "
              f"bypassed for {rep_names}")
    else:
        rep_names = _default_reps()

    max_foldx = args.max_foldx

    print("=" * 70)
    print("GRAPHINITY DDG EXPERIMENT (OPIG B2)")
    print(f"  Paper: Hummer et al. 2025, Nature Computational Science")
    print(f"  Conditions: {conditions}")
    print(f"  Reps: {rep_names}")
    print(f"  Models: {_decided_models()} "
          f"(ddg_combined's decided pair, DECISIONS.md Step 1)")
    print(f"  Folds: {args.n_folds}")
    print(f"  Max samples (FoldX): {max_foldx}")
    print("=" * 70)

    output_path = (args.output or
                   'tests/results/comparisons/opig_graphinity_ddg.json')

    # Each condition is written to <output>.partial.jsonl as soon as it finishes
    # and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # condition, because the one JSON below was written after the last. A condition
    # that raised or returned {'error'} is not stored, is retried next time,
    # and makes the script exit 1 after the JSON is written.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    # Identity bumped 2026-10-05: a partial file from the 2026-10-04 edit
    # (aac in the pool, pruning on) or older is moved aside, not read back.
    # Each fold of 'experimental' is also stored under (condition, fold).
    _ck = script_checkpoint('opig_graphinity_ddg', output_path, args,
                            exclude=('conditions', 'output', 'pruning_mode'),
                            rep_names=rep_names, max_foldx=max_foldx,
                            alignment='2026-10-05 per-fold 2+2, no aac, '
                                      'pruning off, FeatureScaler, float32, '
                                      'no wide cut, slot predictions kept, '
                                      'every-representation table off',
                            cut_wide=CUT_WIDE_REPRESENTATIONS,
                            score_every_rep=SCORE_EVERY_REPRESENTATION)
    _failed = []

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _write(results):
        # Rewritten after every fold and every condition (dirty-run bug 12),
        # not only at the end.
        tmp = p.with_suffix(p.suffix + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        os.replace(tmp, p)

    all_results = {}
    for cond in conditions:
        if cond not in CONDITION_DATASETS:
            print(f"WARNING: Unknown condition '{cond}', skipping")
            continue
        ds_name = CONDITION_DATASETS[cond]
        # transfer_foldx has ds_name=None (handled specially in evaluate_condition)
        if ds_name is not None and ds_name not in DATASET_REGISTRY:
            print(f"WARNING: Dataset '{ds_name}' not in registry, skipping")
            continue
        try:
            if _ck.done(condition=cond):
                cond_results = _ck.get(condition=cond)
                print(f"  {cond}: finished in an earlier attempt, read back")
            else:
                def _on_fold(partial, _cond=cond):
                    _write(dict(all_results, **{_cond: {
                        'condition': _cond, 'in_progress': True,
                        'standard_selection': partial}}))
                cond_results = evaluate_condition(
                    cond, rep_names, args.n_folds,
                    max_foldx=max_foldx, ck=_ck, on_fold=_on_fold)
                # None is never stored as a finished condition (2026-10-05):
                # Checkpoint.put refuses only an {'error'} dict, so a None
                # would be read back next attempt as done with nothing in it.
                if cond_results is None:
                    cond_results = {'error': f'{cond}: returned no result',
                                    'condition': cond}
                _ck.put(cond_results, condition=cond)
            if cond_results is None or is_failed_item(cond_results):
                _failed.append(cond)
            if cond_results is not None:
                all_results[cond] = cond_results
                _write(all_results)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(cond)

    # Cross-condition analysis
    if all_results:
        print_analysis(all_results)
        # The hybrid numbers: per outer fold, per decided model, the 2 chosen
        # representations and the 2 chosen hybrids on their held-out rows.
        print_standard_selection(all_results)

    # Save results
    _write(all_results)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'condition')


if __name__ == '__main__':
    main()
