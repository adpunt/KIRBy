"""Test whether allocation scoring is deterministic on the same inputs.

Hypothesis: `_evaluate_allocation` has silent failures (caught by bare
except → score=0.0) or hidden non-determinism. Either would cause top-K
prediction to flip between runs and fragment the hybrid_eval cache.

This script runs the per-rep allocation scoring loop TWICE on identical
inputs, for a single fold and single evaluator, and compares the
resulting per-rep scores. Any difference → non-determinism or failure.

Usage:
    python tests/scripts/test_alloc_determinism.py [dataset]
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


def main(dataset: str) -> None:
    ds_reps = DATASET_REPS[dataset]
    print(f'Loading {dataset} with {len(ds_reps["general"])} general reps...')
    data = load_data(dataset, rep_preset='all',
                     rep_names=ds_reps['general'])
    labels = data['labels']
    groups = data['groups']
    reps = data['reps']

    # Test fold 1 + evaluator rf — the specific config where we saw
    # onehot/prot_t5 flip between Run A and Run B.
    FOLD = 1
    EVAL = 'rf'

    print(f'\n=== PASS 1: fold={FOLD} evaluator={EVAL} ===')
    t0 = time.time()
    scores_1 = compute_alloc_scores(reps, labels, groups, FOLD, EVAL,
                                     verbose=True)
    print(f'Pass 1 took {time.time() - t0:.1f}s')

    print(f'\n=== PASS 2: fold={FOLD} evaluator={EVAL} (same inputs) ===')
    t0 = time.time()
    scores_2 = compute_alloc_scores(reps, labels, groups, FOLD, EVAL,
                                     verbose=True)
    print(f'Pass 2 took {time.time() - t0:.1f}s')

    print('\n=== COMPARISON ===')
    failures_1 = scores_1.pop('_failed')
    failures_2 = scores_2.pop('_failed')
    print(f'Pass 1 failures: {len(failures_1)}')
    for name, err in failures_1:
        print(f'  {name}: {err}')
    print(f'Pass 2 failures: {len(failures_2)}')
    for name, err in failures_2:
        print(f'  {name}: {err}')

    diffs = []
    for name in sorted(set(scores_1) | set(scores_2)):
        s1 = scores_1.get(name, 'MISSING')
        s2 = scores_2.get(name, 'MISSING')
        if s1 != s2:
            diffs.append((name, s1, s2))

    if diffs:
        print(f'\n{len(diffs)} reps scored DIFFERENTLY between passes:')
        for name, s1, s2 in diffs:
            print(f'  {name}: pass1={s1}  pass2={s2}')
    else:
        print('\nAll reps scored identically between the two passes.')

    # Also show top-5 per pass so we can see if top-k ranking is stable
    def top5(scores):
        items = [(n, s) for n, s in scores.items() if isinstance(s, (int, float))]
        items.sort(key=lambda t: t[1], reverse=True)
        return items[:5]

    print('\nTop-5 by pass 1:')
    for n, s in top5(scores_1):
        print(f'  {n}: {s:.6f}')
    print('Top-5 by pass 2:')
    for n, s in top5(scores_2):
        print(f'  {n}: {s:.6f}')

    print('\n=== VERDICT ===')
    if failures_1 or failures_2:
        print('Silent failures exist. Bare-except path is ACTIVE.')
    if diffs:
        print('Scores are non-deterministic between passes — the '
              'hypothesis is supported.')
    else:
        print('Scores are deterministic within a single process. '
              'Fragmentation must come from elsewhere (rep set differences '
              'between runs, code changes, or cross-process non-determinism).')


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'ginkgo_heparin'
    main(dataset)
