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


def _score_one_split(reps, labels, train_idx, val_idx, internal_evaluator):
    """Score each rep on a single train/val split."""
    from kirby.hybrid import _evaluate_allocation

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    scores = {}
    for rep_name, X in reps.items():
        X_tr = X[train_idx].copy()
        X_va = X[val_idx].copy()

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
            scores[rep_name] = 0.0

    return scores


def get_allocation_scores(reps, labels, groups, internal_evaluator='rf',
                          split_method='random', seed=42):
    """Score each rep using an internal train/val split.

    split_method:
        'random' — current behavior (RandomState permutation, no group awareness)
        'group'  — GroupShuffleSplit (holds out ~20% of groups)
        'logo'   — Leave-One-Group-Out CV (average across all group holdouts)
    """
    n = len(labels)

    if split_method == 'logo' and groups is not None:
        unique_groups = np.unique(groups)
        all_fold_scores = []
        split_info = []
        for held_out_group in unique_groups:
            val_idx = np.where(groups == held_out_group)[0]
            train_idx = np.where(groups != held_out_group)[0]
            fold_scores = _score_one_split(reps, labels, train_idx, val_idx,
                                           internal_evaluator)
            all_fold_scores.append(fold_scores)
            split_info.append((train_idx, val_idx))

        # Average across folds
        avg_scores = {}
        for rep_name in reps:
            vals = [fs[rep_name] for fs in all_fold_scores]
            avg_scores[rep_name] = np.mean(vals)

        # Return last split's indices for leakage reporting (not meaningful for LOGO)
        return avg_scores, split_info[0][0], split_info[0][1]

    elif split_method == 'group' and groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(gss.split(np.zeros(n), labels, groups))
    else:
        indices = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * 0.2))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

    scores = _score_one_split(reps, labels, train_idx, val_idx,
                              internal_evaluator)
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

        methods = {}

        # 1. Current: random split
        random_scores, r_tr, r_va = get_allocation_scores(
            fold_reps, fold_labels, fold_groups,
            internal_evaluator=ev, split_method='random')
        methods['random'] = random_scores

        if fold_groups is not None:
            train_groups = set(fold_groups[r_tr])
            val_groups = set(fold_groups[r_va])
            leaked = train_groups & val_groups
            print(f"\n  Random split: {len(r_tr)} train, {len(r_va)} val")
            print(f"  Groups in train: {len(train_groups)}, in val: {len(val_groups)}")
            print(f"  LEAKED groups (in both): {len(leaked)}/{len(val_groups)} "
                  f"({len(leaked)/len(val_groups)*100:.0f}%)")

        # 2. Group-aware single split (multiple seeds, averaged)
        group_scores_all = []
        for seed in range(args.n_trials):
            gs, g_tr, g_va = get_allocation_scores(
                fold_reps, fold_labels, fold_groups,
                internal_evaluator=ev, split_method='group', seed=seed)
            group_scores_all.append(gs)

            if seed == 0 and fold_groups is not None:
                train_groups_g = set(fold_groups[g_tr])
                val_groups_g = set(fold_groups[g_va])
                leaked_g = train_groups_g & val_groups_g
                print(f"\n  Group split (seed 0): {len(g_tr)} train, {len(g_va)} val")
                print(f"  Groups in train: {len(train_groups_g)}, in val: {len(val_groups_g)}")
                print(f"  LEAKED groups (in both): {len(leaked_g)}/{len(val_groups_g)} "
                      f"({len(leaked_g)/len(val_groups_g)*100:.0f}%)")

        group_scores_avg = {}
        for rep in fold_reps:
            group_scores_avg[rep] = np.mean([gs[rep] for gs in group_scores_all])
        methods['group'] = group_scores_avg

        # 3. Leave-one-group-out CV
        logo_scores, _, _ = get_allocation_scores(
            fold_reps, fold_labels, fold_groups,
            internal_evaluator=ev, split_method='logo')
        methods['logo'] = logo_scores

        n_inner_groups = len(np.unique(fold_groups)) if fold_groups is not None else 0
        print(f"\n  LOGO: {n_inner_groups}-fold internal CV (one group held out per fold)")

        # 4. Compare all three
        random_ranking = sorted(fold_reps.keys(), key=lambda r: -methods['random'][r])
        group_ranking = sorted(fold_reps.keys(), key=lambda r: -methods['group'][r])
        logo_ranking = sorted(fold_reps.keys(), key=lambda r: -methods['logo'][r])

        print(f"\n  {'Rep':<25} {'Random':>8} {'Group':>8} {'LOGO':>8}  "
              f"{'R_rank':>6} {'G_rank':>6} {'L_rank':>6}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  {'-'*6} {'-'*6} {'-'*6}")

        for rep in random_ranking:
            rs = methods['random'][rep]
            gs = methods['group'][rep]
            ls = methods['logo'][rep]
            rank_r = random_ranking.index(rep) + 1
            rank_g = group_ranking.index(rep) + 1
            rank_l = logo_ranking.index(rep) + 1
            print(f"  {rep:<25} {rs:>8.4f} {gs:>8.4f} {ls:>8.4f}  "
                  f"{rank_r:>6} {rank_g:>6} {rank_l:>6}")

        # Pairwise rank correlations
        from scipy.stats import spearmanr
        reps_list = list(fold_reps.keys())
        r_vals = [methods['random'][r] for r in reps_list]
        g_vals = [methods['group'][r] for r in reps_list]
        l_vals = [methods['logo'][r] for r in reps_list]

        rg_rho, _ = spearmanr(r_vals, g_vals)
        rl_rho, _ = spearmanr(r_vals, l_vals)
        gl_rho, _ = spearmanr(g_vals, l_vals)

        print(f"\n  Rank correlations:")
        print(f"    Random vs Group: rho={rg_rho:.4f}")
        print(f"    Random vs LOGO:  rho={rl_rho:.4f}")
        print(f"    Group  vs LOGO:  rho={gl_rho:.4f}")

        # Top-k agreement
        for k in [3, 5]:
            if k > len(reps_list):
                continue
            top_r = set(random_ranking[:k])
            top_g = set(group_ranking[:k])
            top_l = set(logo_ranking[:k])
            rg = len(top_r & top_g)
            rl = len(top_r & top_l)
            gl = len(top_g & top_l)
            print(f"  Top-{k}: R∩G={rg}/{k}  R∩L={rl}/{k}  G∩L={gl}/{k}")

        print()


if __name__ == '__main__':
    main()
