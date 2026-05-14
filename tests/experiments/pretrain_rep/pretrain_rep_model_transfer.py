#!/usr/bin/env python3
"""PT-Rep P5 — model transfer on a pre-trained hybrid representation.

Classical transfer learning, with a KIRBy hybrid as the fixed representation:

  1. Build the KIRBy hybrid on the PT (large) dataset.
  2. Pre-train a DNN on the PT dataset using that hybrid.
  3. Apply the SAME hybrid (same selected features) to the FT (small) dataset.
  4. Fine-tune that same DNN on the FT dataset.
  5. Compare against the same DNN trained from scratch on the same FT hybrid.

The question this answers: is model transfer more effective when the
representation it runs on is a good KIRBy hybrid? Both the transfer DNN and the
scratch DNN run on the *same* representation (the PT-built hybrid applied to
FT), so the only thing that varies is whether the model started from
pre-trained weights.

This is distinct from the A/B/F/G/H harness (`pretrain_rep_experiment.py`),
which carries over the representation choice and never touches model weights.
P5 carries over a trained model.

Constraint: PT and FT must both be regression tasks — `PretrainedDNNRegressor`
is regression-only and the continued training needs a consistent target.

Reuses: `kirby.models.dnn_pretrained.PretrainedDNNRegressor` and the checkpoint
format from `kirby.models.pretrain_chembl_dnn` (same mechanism as the Imrie C4
experiment, but the representation is a KIRBy hybrid, not a fixed fingerprint).
`run()` is a server workload (DNN training + hybrid search).

Usage:
    python pretrain_rep_model_transfer.py \\
        --pt-source human_chembl --ft-dataset herg_ki --mode full
"""
from __future__ import annotations

import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    _cv_split, get_precomputed_search, get_domain_metric, resolve_pruning_kwargs,
)
from experiments.pretrain_rep.pretrain_rep_experiment import (
    prepare_pt_corpus, shared_rep_set, pt_search_metric,
    aggregate_fold_scores, extract_top_n_blueprints, blueprint_key,
)
from kirby.hybrid import create_hybrid, apply_feature_selection

HIDDEN_SIZES = (256, 128, 64)


def _checkpoint_dir() -> Path:
    root = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    d = root / 'pt_model_transfer'
    d.mkdir(parents=True, exist_ok=True)
    return d


def pretrain_dnn_on_matrix(X, y, output_path, hidden_sizes=HIDDEN_SIZES,
                           dropout=0.2, epochs=200, batch_size=1024,
                           lr=0.001, patience=30, seed=42, verbose=True):
    """Pre-train a DNN on a precomputed feature matrix; write a checkpoint in
    the exact format `PretrainedDNNRegressor` expects.

    Mirrors `kirby.models.pretrain_chembl_dnn.pretrain_one_rep` but takes the
    feature matrix directly — here X is the KIRBy hybrid built on the PT
    corpus, not a named fingerprint.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]

    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y_mean = float(y.mean())
    y_std = float(y.std()) if y.std() > 1e-8 else 1.0
    y_norm = (y - y_mean) / y_std

    n = len(y)
    idx = rng.permutation(n)
    split = max(1, int(0.9 * n))
    train_idx, ho_idx = idx[:split], idx[split:]
    if len(ho_idx) == 0:  # tiny corpus — hold out one row
        ho_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y_norm[train_idx])
    X_ho = torch.from_numpy(X[ho_idx])
    y_ho = torch.from_numpy(y_norm[ho_idx])

    layers, prev = [], X.shape[1]
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    net = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_ho, best_state, patience_ctr = float('inf'), None, 0
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            b = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(net(X_train[b]).squeeze(-1), y_train[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            ho_loss = criterion(net(X_ho).squeeze(-1), y_ho).item()
        scheduler.step(ho_loss)
        if ho_loss < best_ho:
            best_ho = ho_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    checkpoint = {
        'state_dict': best_state if best_state is not None else net.state_dict(),
        'input_dim': X.shape[1],
        'hidden_sizes': hidden_sizes,
        'n_compounds': int(n),
        'y_mean': y_mean,
        'y_std': y_std,
        'feat_mean': feat_scaler.mean_.tolist(),
        'feat_scale': feat_scaler.scale_.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(checkpoint, output_path)
    if verbose:
        print(f"    pretrained DNN saved: {output_path} "
              f"(input_dim={X.shape[1]}, n={n})")
    return output_path


def run(ft_data, pt_source, pt_task, config, output_dir,
        pt_max_samples=None, pruning_mode='default_pruning',
        epochs=200, finetune_lr=1e-4):
    """P5 — model transfer on a PT-built KIRBy hybrid.

    Args:
        ft_data: dict from load_data() for the fine-tuning target.
        pt_source: PT source name.
        pt_task: PT sub-task selector, or None.
        config: dict from get_config().
        output_dir: results / quality-report directory.
        pt_max_samples: optional PT corpus size cap.
        pruning_mode: adaptive-pruning mode for the PT hybrid search.
        epochs: max fine-tuning epochs.
        finetune_lr: fine-tuning learning rate (10x below pre-training).
    """
    from kirby.models.dnn import DNNRegressor
    from kirby.models.dnn_pretrained import PretrainedDNNRegressor

    ft_name = ft_data.get('dataset_name', 'ft')
    domain = ft_data['domain']
    task_type = ft_data['task_type']
    if task_type != 'regression':
        raise ValueError(
            f"P5 model transfer is regression-only (PretrainedDNNRegressor); "
            f"FT dataset '{ft_name}' is task_type={task_type!r}.")
    metric_name, metric_fn, lower_is_better = get_domain_metric(ft_data)

    print("\n" + "=" * 72)
    print("PT-Rep P5 — model transfer on a pre-trained hybrid representation")
    print(f"  FT target : {ft_name} ({len(ft_data['labels'])} samples, {domain})")
    print(f"  PT source : {pt_source} (task={pt_task})")
    print(f"  Metric    : {metric_name} ({'lower' if lower_is_better else 'higher'} better)")
    print("=" * 72)

    ft_reps_all = ft_data['reps']
    ft_labels = ft_data['labels']
    ft_groups = ft_data['groups']
    n_folds = config['n_splits']

    # ── PT corpus prep ──
    pt_reps_all, pt_labels, pt_groups, pt_ds, pt_reports = prepare_pt_corpus(
        pt_source, pt_task, ft_data, list(ft_reps_all.keys()),
        max_samples=pt_max_samples, output_dir=output_dir)
    if pt_labels is None:
        raise ValueError(
            f"PT source '{pt_source}' is unlabelled — P5 needs PT labels to "
            f"pre-train the model.")
    if pt_ds.task_type != 'regression':
        raise ValueError(
            f"P5 needs a regression PT corpus; '{pt_source}' is "
            f"task_type={pt_ds.task_type!r}.")

    # ── Shared rep set ──
    shared = shared_rep_set(ft_reps_all, pt_reps_all)
    if len(shared) < 2:
        print("  ERROR: need >=2 shared reps for a hybrid")
        return {'error': 'insufficient_shared_reps', 'shared_reps': shared}
    ft = {k: ft_reps_all[k] for k in shared}
    pt = {k: pt_reps_all[k] for k in shared}
    print(f"\n  Shared reps ({len(shared)}): {', '.join(shared)}")

    # ── Build the KIRBy hybrid on the PT corpus ──
    pt_pkw = resolve_pruning_kwargs(pruning_mode, len(pt_labels),
                                    dataset_label=f'pt_{pt_source}')
    print(f"\n  PT hybrid search ({pt_source})")
    pt_search = get_precomputed_search(
        pt, pt_labels, pt_groups, aug_condition='none',
        speed=config['search_speed'], metric=pt_search_metric(pt_ds.task_type),
        task_type=pt_ds.task_type, domain=domain,
        dataset_name=f'pt__{pt_source}', tier='general', **pt_pkw)

    bps = extract_top_n_blueprints(pt_search, n=1)
    if not bps:
        print("  ERROR: PT hybrid search produced no usable blueprint")
        return {'error': 'no_pt_blueprint'}
    pt_bp = bps[0]
    print(f"  PT hybrid: budget={pt_bp.budget}, "
          f"alloc={pt_bp.allocation_method}, imp={pt_bp.importance_method}")

    # The hybrid's concrete feature selection, built once on the FULL PT corpus.
    _, fi_pt = create_hybrid(
        pt, pt_labels,
        allocation_method=pt_bp.allocation_method,
        budget=pt_bp.budget,
        importance_method=pt_bp.importance_method,
        internal_evaluator=pt_bp.internal_evaluator)
    X_pt_hybrid = apply_feature_selection(pt, fi_pt)
    print(f"  PT hybrid feature matrix: {X_pt_hybrid.shape}")

    # ── Pre-train the DNN on the PT hybrid ──
    bp_hash = hashlib.sha1(
        repr(blueprint_key(pt_bp)).encode()).hexdigest()[:10]
    ckpt = _checkpoint_dir() / f'{pt_source}__{pt_task or "default"}__{bp_hash}.pt'
    print(f"\n  Pre-training DNN on PT hybrid (epochs<= {epochs})")
    pretrain_dnn_on_matrix(X_pt_hybrid, pt_labels, str(ckpt), epochs=epochs)

    # ── Per-fold FT evaluation: transfer vs scratch, same representation ──
    print(f"\n  FT evaluation ({n_folds}-fold CV) — transfer vs scratch")
    folds = list(_cv_split(ft_groups, n_folds,
                           next(iter(ft.values())), ft_labels))
    scores = {'transfer': [], 'scratch': []}

    for fi, (tr_idx, va_idx) in enumerate(folds):
        tr = {k: v[tr_idx] for k, v in ft.items()}
        va = {k: v[va_idx] for k, v in ft.items()}
        y_tr, y_va = ft_labels[tr_idx], ft_labels[va_idx]

        # Same representation for both: the PT-built hybrid applied to FT.
        X_tr = apply_feature_selection(tr, fi_pt)
        X_va = apply_feature_selection(va, fi_pt)

        # transfer: DNN initialised from the PT-pre-trained weights
        try:
            m = PretrainedDNNRegressor(
                hidden_sizes=HIDDEN_SIZES, pretrained_path=str(ckpt),
                finetune_lr=finetune_lr, freeze_layers=0, epochs=epochs)
            m.fit(X_tr, y_tr)
            scores['transfer'].append(float(metric_fn(y_va, m.predict(X_va))))
        except Exception as e:
            print(f"    fold {fi} transfer failed: {e}")
            scores['transfer'].append(float('nan'))

        # scratch: same DNN architecture, random init, same representation
        try:
            m = DNNRegressor(hidden_sizes=HIDDEN_SIZES, epochs=epochs)
            m.fit(X_tr, y_tr)
            scores['scratch'].append(float(metric_fn(y_va, m.predict(X_va))))
        except Exception as e:
            print(f"    fold {fi} scratch failed: {e}")
            scores['scratch'].append(float('nan'))

        print(f"    fold {fi}: transfer={scores['transfer'][-1]:.4f}  "
              f"scratch={scores['scratch'][-1]:.4f}")

    transfer = aggregate_fold_scores(scores['transfer'])
    scratch = aggregate_fold_scores(scores['scratch'])
    delta = ((scratch['mean'] - transfer['mean']) if lower_is_better
             else (transfer['mean'] - scratch['mean']))

    arrow = '↓' if lower_is_better else '↑'
    print(f"\n  {'='*60}")
    print(f"  RESULT — {ft_name} <- PT:{pt_source}   metric={metric_name} ({arrow})")
    print(f"    transfer DNN : {transfer['mean']:.4f} ± {transfer['std']:.4f}")
    print(f"    scratch DNN  : {scratch['mean']:.4f} ± {scratch['std']:.4f}")
    print(f"    delta (transfer benefit): {delta:+.4f}")
    print(f"  {'='*60}")

    return {
        'ft_dataset': ft_name,
        'pt_source': pt_source,
        'pt_task': pt_task,
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'shared_reps': shared,
        'pt_blueprint': pt_bp.to_dict(),
        'pt_reports': pt_reports,
        'transfer': convert_numpy(transfer),
        'scratch': convert_numpy(scratch),
        'transfer_benefit': float(delta),
    }


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='PT-Rep P5 — model transfer on a pre-trained hybrid')
    ap.add_argument('--pt-source', required=True)
    ap.add_argument('--pt-task', default=None)
    ap.add_argument('--ft-dataset', required=True)
    ap.add_argument('--pt-max-samples', type=int, default=None)
    ap.add_argument('--mode', choices=['intermediary', 'full'], default='full')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--finetune-lr', type=float, default=1e-4)
    ap.add_argument('--budgets', default=None)
    ap.add_argument('--models', default=None)
    ap.add_argument('--pruning-mode', default='default_pruning')
    ap.add_argument('--output', default='tests/results')
    args = ap.parse_args()

    config = get_config(args)
    ft_data = load_data(args.ft_dataset)
    results = run(ft_data, args.pt_source, args.pt_task, config, args.output,
                  pt_max_samples=args.pt_max_samples,
                  pruning_mode=args.pruning_mode, epochs=args.epochs,
                  finetune_lr=args.finetune_lr)
    save_experiment_results(
        results, args.output, 'pretrain_rep_model_transfer',
        f'{args.ft_dataset}__pt_{args.pt_source}')
