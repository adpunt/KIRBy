#!/usr/bin/env python3
"""Test: does group-aware internal splitting change allocation decisions?

Runs create_hybrid twice on the same data:
1. Current: random 80/20 internal split (RandomState(42).permutation)
2. Proposed: GroupShuffleSplit internal split (respects groups)

Compares:
- Per-rep allocation scores (do reps get ranked differently?)
- Feature allocations (does the budget distribution change?)
- Final hybrid scores on held-out test (does performance differ?)

Usage (on server — needs data):
    python tests/analysis/test_internal_split_impact.py herg_ki
    python tests/analysis/test_internal_split_impact.py flab_pk_jain_heprt
    python tests/analysis/test_internal_split_impact.py flab_binding_koenig  # DMS, 5 groups
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def get_allocation_scores(reps, labels, groups, internal_evaluator='rf',
                          use_group_split=False, seed=42):
    """Score each rep using an internal train/val split.

    Returns dict {rep_name: score} using the same logic as create_hybrid's
    allocation scoring, but with control over the split method.
    """
    from kirby.hybrid import _evaluate_allocation

    n = len(labels)

    if use_group_split and groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(gss.split(np.zeros(n), labels, groups))
    else:
        # Current random split (replicates create_hybrid behavior)
        indices = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * 0.2))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    scores = {}
    for rep_name, X in reps.items():
        X_tr = X[train_idx].copy()
        X_va = X[val_idx].copy()

        # Impute NaN (same as find_best_hybrid)
        col_means = np.nanmean(X_tr, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for col in range(X_tr.shape[1]):
            mask_tr = np.isnan(X_tr[:, col])
            mask_va = np.isnan(X_va[:, col])
            X_tr[mask_tr, col] = col_means[col]
            X_va[mask_va, col] = col_means[col]

        try:
            score, _ = _evaluate_allocation(
                X_tr, X_va, train_labels, val_labels,
                internal_evaluator=internal_evaluator,
                return_predictions=True)
            scores[rep_name] = max(score, 0.0)
        except Exception as e:
            print(f"    {rep_name}: FAILED ({e})")
            scores[rep_name] = 0.0

    return scores, train_idx, val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--evaluators', default='rf,ridge,svm',
                        help='Comma-separated internal evaluators to test')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of random seeds for group split (default 5)')
    args = parser.parse_args()

    from experiments._shared import load_data

    print(f"Loading {args.dataset}...")
    data = load_data(args.dataset)
    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')

    n_groups = len(np.unique(groups)) if groups is not None else 0
    print(f"  {len(labels)} samples, {len(reps)} reps, {n_groups} groups")
    print()

    # Use first outer CV fold's training data (same as precompute would)
    cv = GroupKFold(n_splits=5)
    train_idx, test_idx = next(cv.split(np.zeros(len(labels)), labels, groups))

    fold_reps = {k: v[train_idx] for k, v in reps.items()}
    fold_labels = labels[train_idx]
    fold_groups = groups[train_idx] if groups is not None else None

    n_fold_groups = len(np.unique(fold_groups)) if fold_groups is not None else 0
    print(f"  Outer fold 0: {len(fold_labels)} train samples, "
          f"{len(test_idx)} test samples, {n_fold_groups} groups in train")
    print()

    evaluators = [e.strip() for e in args.evaluators.split(',')]

    for ev in evaluators:
        print(f"{'='*70}")
        print(f"Internal evaluator: {ev}")
        print(f"{'='*70}")

        # 1. Current: random split
        random_scores, r_tr, r_va = get_allocation_scores(
            fold_reps, fold_labels, fold_groups,
            internal_evaluator=ev, use_group_split=False)

        # Check group leakage in random split
        if fold_groups is not None:
            train_groups = set(fold_groups[r_tr])
            val_groups = set(fold_groups[r_va])
            leaked = train_groups & val_groups
            print(f"\n  Random split: {len(r_tr)} train, {len(r_va)} val")
            print(f"  Groups in train: {len(train_groups)}, in val: {len(val_groups)}")
            print(f"  LEAKED groups (in both): {len(leaked)}/{len(val_groups)} "
                  f"({len(leaked)/len(val_groups)*100:.0f}%)")

        # 2. Group-aware split (multiple seeds to check stability)
        group_scores_all = []
        for seed in range(args.n_trials):
            gs, g_tr, g_va = get_allocation_scores(
                fold_reps, fold_labels, fold_groups,
                internal_evaluator=ev, use_group_split=True, seed=seed)
            group_scores_all.append(gs)

            if seed == 0 and fold_groups is not None:
                train_groups_g = set(fold_groups[g_tr])
                val_groups_g = set(fold_groups[g_va])
                leaked_g = train_groups_g & val_groups_g
                print(f"\n  Group split (seed 0): {len(g_tr)} train, {len(g_va)} val")
                print(f"  Groups in train: {len(train_groups_g)}, in val: {len(val_groups_g)}")
                print(f"  LEAKED groups (in both): {len(leaked_g)}/{len(val_groups_g)} "
                      f"({len(leaked_g)/len(val_groups_g)*100:.0f}%)")

        # Average group scores across seeds
        group_scores_avg = {}
        for rep in fold_reps:
            vals = [gs[rep] for gs in group_scores_all]
            group_scores_avg[rep] = np.mean(vals)

        # 3. Compare rankings
        random_ranking = sorted(random_scores.keys(),
                                key=lambda r: -random_scores[r])
        group_ranking = sorted(group_scores_avg.keys(),
                               key=lambda r: -group_scores_avg[r])

        print(f"\n  {'Rep':<25} {'Random':>10} {'Group':>10} {'Diff':>10} "
              f"{'Rank_R':>7} {'Rank_G':>7} {'RankΔ':>7}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")

        for rep in random_ranking:
            rs = random_scores[rep]
            gs = group_scores_avg[rep]
            diff = gs - rs
            rank_r = random_ranking.index(rep) + 1
            rank_g = group_ranking.index(rep) + 1
            rank_diff = rank_r - rank_g
            print(f"  {rep:<25} {rs:>10.4f} {gs:>10.4f} {diff:>+10.4f} "
                  f"{rank_r:>7} {rank_g:>7} {rank_diff:>+7}")

        # Rank correlation
        from scipy.stats import spearmanr, kendalltau
        r_vals = [random_scores[r] for r in random_ranking]
        g_vals = [group_scores_avg[r] for r in random_ranking]
        rho, _ = spearmanr(r_vals, g_vals)
        tau, _ = kendalltau(r_vals, g_vals)
        print(f"\n  Rank correlation: Spearman rho={rho:.4f}, Kendall tau={tau:.4f}")

        # Top-k agreement
        for k in [3, 5]:
            if k > len(random_ranking):
                continue
            top_r = set(random_ranking[:k])
            top_g = set(group_ranking[:k])
            overlap = len(top_r & top_g)
            print(f"  Top-{k} overlap: {overlap}/{k}")

        print()


if __name__ == '__main__':
    main()
