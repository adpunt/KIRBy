"""Test whether allocation scoring is deterministic on the same inputs.

Hypothesis: `_evaluate_allocation` has silent failures (caught by bare
except → score=0.0) or hidden non-determinism. Either would cause top-K
prediction to flip between runs and fragment the hybrid_eval cache.

This script runs the per-rep allocation scoring loop TWICE on identical
inputs, for a single fold and single evaluator, and compares the
resulting per-rep scores. Any difference → non-determinism or failure.

Usage:
    python tests/scripts/test_alloc_determinism.py                       # default set
    python tests/scripts/test_alloc_determinism.py ginkgo_heparin
    python tests/scripts/test_alloc_determinism.py ginkgo_heparin flab_thermo_jain
    python tests/scripts/test_alloc_determinism.py all                   # every antibody dataset
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments._shared import load_data
from experiments.complexity.complexity_helpers import DATASET_REPS
from kirby.hybrid import _evaluate_allocation
from kirby.find_best_hybrid import _make_cv_splitter


def _impute_nan(X_tr, X_va):
    """Mirror find_best_hybrid._impute_nan — fill NaN with train mean."""
    if not np.isnan(X_tr).any() and not np.isnan(X_va).any():
        return X_tr, X_va, None
    means = np.nanmean(X_tr, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    X_tr = np.where(np.isnan(X_tr), means, X_tr)
    X_va = np.where(np.isnan(X_va), means, X_va)
    return X_tr, X_va, means


def compute_alloc_scores(base_reps, labels, groups, fold_idx, evaluator,
                         n_folds=5, seed=42, verbose=False):
    """Replicate the inner alloc-scoring loop from find_best_hybrid.py:2632-2689.

    Returns: dict {rep_name: score, ..., '_failed': [rep_names that raised]}
    """
    splitter, use_groups = _make_cv_splitter(
        groups, n_folds, seed, cv_splitter=None)
    sample_X = next(iter(base_reps.values()))
    split_args = (sample_X, labels, groups) if use_groups else (sample_X, labels)
    fold_splits = list(splitter.split(*split_args))
    train_idx, _ = fold_splits[fold_idx]

    fold_train_reps = {k: v[train_idx] for k, v in base_reps.items()}
    fold_train_labels = labels[train_idx]
    n_fold = len(fold_train_labels)
    fold_train_groups = groups[train_idx] if groups is not None else None

    if fold_train_groups is not None and len(np.unique(fold_train_groups)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(gss.split(
            np.zeros(n_fold), fold_train_labels, fold_train_groups))
    else:
        n_val_internal = max(1, int(n_fold * 0.2))
        indices = np.random.RandomState(42).permutation(n_fold)
        tr_idx = indices[n_val_internal:]
        va_idx = indices[:n_val_internal]

    train_labels = fold_train_labels[tr_idx]
    val_labels = fold_train_labels[va_idx]

    scores = {}
    failures = []
    for rep_name, X_full in fold_train_reps.items():
        X_tr = X_full[tr_idx].copy()
        X_va = X_full[va_idx].copy()
        X_tr, X_va, _ = _impute_nan(X_tr, X_va)
        try:
            default_score, _ = _evaluate_allocation(
                X_tr, X_va, train_labels, val_labels,
                internal_evaluator=evaluator, return_predictions=True)
            scores[rep_name] = max(default_score, 0.0)
        except Exception as e:
            scores[rep_name] = 0.0
            failures.append((rep_name, f'{type(e).__name__}: {e}'))
            if verbose:
                print(f'    [FAIL] {rep_name}: {type(e).__name__}: {e}')
    scores['_failed'] = failures
    return scores


def test_one(dataset: str):
    """Returns (n_failures_any_pass, n_score_diffs, n_top5_diffs)."""
    ds_reps = DATASET_REPS.get(dataset)
    if ds_reps is None or 'general' not in ds_reps:
        print(f'[{dataset}] SKIP: no general rep list in DATASET_REPS')
        return None

    print(f'\n{"=" * 70}')
    print(f'DATASET: {dataset}')
    print(f'{"=" * 70}')

    try:
        data = load_data(dataset, rep_preset='all',
                        rep_names=ds_reps['general'])
    except Exception as e:
        print(f'[{dataset}] load_data FAILED: {type(e).__name__}: {e}')
        return None

    labels = data['labels']
    groups = data.get('groups')
    reps = data['reps']
    print(f'Loaded: {len(labels)} samples, {len(reps)} reps')

    FOLD = 1
    EVAL = 'rf'

    t0 = time.time()
    scores_1 = compute_alloc_scores(reps, labels, groups, FOLD, EVAL)
    t1 = time.time() - t0
    t0 = time.time()
    scores_2 = compute_alloc_scores(reps, labels, groups, FOLD, EVAL)
    t2 = time.time() - t0
    print(f'Pass 1 took {t1:.1f}s, pass 2 took {t2:.1f}s')

    failures_1 = scores_1.pop('_failed')
    failures_2 = scores_2.pop('_failed')
    n_fail = len(failures_1) + len(failures_2)
    if failures_1:
        print(f'Pass 1 failures ({len(failures_1)}):')
        for name, err in failures_1:
            print(f'  {name}: {err}')
    if failures_2:
        print(f'Pass 2 failures ({len(failures_2)}):')
        for name, err in failures_2:
            print(f'  {name}: {err}')

    diffs = []
    for name in sorted(set(scores_1) | set(scores_2)):
        s1 = scores_1.get(name)
        s2 = scores_2.get(name)
        if s1 != s2:
            diffs.append((name, s1, s2))

    if diffs:
        print(f'{len(diffs)} reps scored DIFFERENTLY between passes:')
        for name, s1, s2 in diffs:
            print(f'  {name}: pass1={s1!r}  pass2={s2!r}')
    else:
        print(f'All {len(scores_1)} reps scored identically between passes.')

    def topk(scores, k=5):
        items = [(n, s) for n, s in scores.items()
                 if isinstance(s, (int, float))]
        items.sort(key=lambda t: t[1], reverse=True)
        return [n for n, _ in items[:k]]

    top_1 = topk(scores_1, 5)
    top_2 = topk(scores_2, 5)
    top_same = top_1 == top_2
    if not top_same:
        print(f'Top-5 order differs:')
        print(f'  pass 1: {top_1}')
        print(f'  pass 2: {top_2}')
    else:
        print(f'Top-5 is identical: {top_1}')

    return (n_fail, len(diffs), 0 if top_same else 1)


def main(datasets):
    results = {}
    for ds in datasets:
        out = test_one(ds)
        if out is not None:
            results[ds] = out

    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    print(f'{"dataset":<36} {"failures":>10} {"score_diffs":>13} '
          f'{"top5_diff":>11}')
    print('-' * 72)
    any_fail = False
    any_diff = False
    any_top5 = False
    for ds, (nf, nd, nt) in results.items():
        print(f'{ds:<36} {nf:>10} {nd:>13} {nt:>11}')
        if nf:
            any_fail = True
        if nd:
            any_diff = True
        if nt:
            any_top5 = True

    print('\n=== OVERALL VERDICT ===')
    if any_fail:
        print('- Silent failures happened: bare-except is active somewhere.')
    if any_diff:
        print('- Per-rep scores diverged between two identical passes: '
              'non-determinism confirmed.')
    if any_top5:
        print('- Top-5 ordering differed on at least one dataset: '
              'enough to flip the cached subset_hash.')
    if not (any_fail or any_diff or any_top5):
        print('- All datasets scored deterministically in-process. '
              'Fragmentation originates from something that differs '
              'between separate processes/runs (rep set, code version, '
              'BLAS threading, process-level RNG state).')


DEFAULT_DATASETS = [
    # antibody
    'ginkgo_heparin',
    'ginkgo_acsins',
    'flab_thermo_jain',
    'abcov_kd',
    # molecular
    'caco2',
    # reaction
    'buchwald_hartwig',
]


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        datasets = DEFAULT_DATASETS
    elif args == ['all']:
        datasets = [k for k, v in DATASET_REPS.items() if 'general' in v]
    else:
        datasets = args
    main(datasets)
