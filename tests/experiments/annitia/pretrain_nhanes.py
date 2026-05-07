#!/usr/bin/env python3
"""Pretrain an MLP encoder on NHANES death endpoint, save weights for finetuning.

Pipeline:
    1. Load (or download + assemble) NHANES cohort via
       kirby.datasets.nhanes_liver.load_nhanes_liver. Default restrict =
       'fib4_high' so the pretraining cohort sits closer to ANNITIA's
       MASLD-clinic distribution than the raw community sample would.
    2. Standardise features. Save the scaler.
    3. Fit a small MLP with a discrete-time hazard head over a quantile
       grid of follow-up times. The trunk is what we transfer; the head is
       discarded at finetune time. Loss is the cause-pooled cross-entropy
       over (cause, time-bin) tuples — equivalent to a discrete-time hazard
       NN. We use sample weights from NHANES (WTMEC2YR) when available.
    4. Persist (encoder_state_dict, scaler, feature_schema, time_grid) to
       <output>/nhanes_encoder.pt.

Usage:
    python tests/experiments/annitia/pretrain_nhanes.py \
        --restrict fib4_high \
        --output tests/results/annitia_nhanes/encoder.pt
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))

from kirby.datasets.nhanes_liver import load_nhanes_liver, NHANES_FEATURE_SCHEMA


def _build_mlp(d_in: int, hidden=(128, 64), dropout: float = 0.3,
               n_bins: int = 10):
    import torch.nn as nn
    layers = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    trunk = nn.Sequential(*layers)
    head = nn.Linear(prev, n_bins)
    return trunk, head, prev


def _bin_targets(time, event, grid):
    """Discrete-time hazard targets.

    For each patient, returns:
        bin_idx: index of their event/censor bin (0..n_bins-1)
        observed: whether they experienced an event in that bin
    The loss is cross-entropy on the hazard logit at the bin, with
    censored patients excluded from the bin-level negative log-likelihood
    term but still contributing to the survival product term (we approximate
    via simple BCE per bin instead of the full discrete-time NLL — slightly
    looser but simpler and the encoder is what we care about).
    """
    n_bins = len(grid) - 1
    idx = np.searchsorted(grid, time, side='right') - 1
    idx = np.clip(idx, 0, n_bins - 1).astype(np.int64)
    return idx, event.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=str, default=None)
    p.add_argument('--restrict', type=str, default='fib4_high',
                   choices=['none', 'fib4_high', 'metab_loaded'])
    p.add_argument('--target', type=str, default='all_cause',
                   choices=['all_cause', 'liver'])
    p.add_argument('--n-bins', type=int, default=10)
    p.add_argument('--hidden', type=str, default='128x64',
                   help='LxL... hidden layer sizes')
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--output', type=str,
                   default='tests/results/annitia_nhanes/encoder.pt')
    args = p.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 70)
    print('NHANES pretraining for ANNITIA encoder transfer')
    print(f'  restrict: {args.restrict}')
    print(f'  target:   {args.target}')
    print(f'  hidden:   {args.hidden}')
    print('=' * 70)

    cohort = load_nhanes_liver(
        data_dir=args.data_dir,
        restrict=None if args.restrict == 'none' else args.restrict)
    X = cohort['X']
    if args.target == 'liver':
        event = cohort['liver_death']
    else:
        event = cohort['event']
    time = cohort['time']
    sw = cohort['sample_weight']
    print(f'  cohort: n={len(X)}  events={int(event.sum())}  '
          f'feature_dim={X.shape[1]}  '
          f'(features={list(NHANES_FEATURE_SCHEMA)})')

    # Standardise on the cohort
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xs = ((X - mean) / std).astype(np.float32)

    # Quantile grid over event times
    pos = time[event > 0]
    if pos.size < 5:
        pos = time
    qs = np.linspace(0.0, 1.0, args.n_bins + 1)
    grid = np.unique(np.quantile(pos, qs)).astype(np.float32)
    if len(grid) < 3:
        grid = np.linspace(time.min(), time.max(), args.n_bins + 1)
    n_bins = len(grid) - 1
    print(f'  time grid bins={n_bins}')

    bin_idx, observed = _bin_targets(time.astype(np.float32),
                                     event.astype(np.float32), grid)

    hidden = tuple(int(h) for h in args.hidden.split('x'))
    trunk, head, dim_emb = _build_mlp(Xs.shape[1], hidden=hidden,
                                       dropout=args.dropout, n_bins=n_bins)
    net = torch.nn.Sequential(trunk, head).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    Xt = torch.tensor(Xs, dtype=torch.float32, device=args.device)
    bin_t = torch.tensor(bin_idx, dtype=torch.long, device=args.device)
    obs_t = torch.tensor(observed, dtype=torch.float32, device=args.device)
    sw_t = torch.tensor(sw, dtype=torch.float32, device=args.device)

    n = len(Xs)
    idx = np.arange(n)
    best_loss = float('inf')
    best_state = None
    bad = 0

    for ep in range(args.epochs):
        np.random.RandomState(args.seed + ep).shuffle(idx)
        net.train()
        losses = []
        for i in range(0, n, args.batch_size):
            bs = idx[i:i + args.batch_size]
            bt = torch.tensor(bs, dtype=torch.long, device=args.device)
            opt.zero_grad()
            logits = net(Xt[bt])  # (B, n_bins)
            # Build per-bin label: 1 in the event bin only for observed events;
            # otherwise 0. (Censored patients keep all-zero labels — a soft
            # negative target at every bin. This is intentionally simpler
            # than a strict discrete-time NLL but is sufficient for the
            # trunk's representational learning, which is what we transfer.)
            target = torch.zeros_like(logits)
            target[torch.arange(len(bs), device=args.device),
                   bin_t[bt]] = obs_t[bt]
            loss_per_bin = bce(logits, target)  # (B, n_bins)
            loss = (loss_per_bin.mean(dim=1) * sw_t[bt]).sum() / max(sw_t[bt].sum(), 1.0)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f'  early stop at epoch {ep} (loss={mean_loss:.4f})')
                break
        if ep % 5 == 0:
            print(f'  ep={ep:3d} loss={mean_loss:.4f}')

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'trunk_state': trunk.state_dict(),
        'head_state': head.state_dict(),
        'mean': mean,
        'std': std,
        'feature_schema': list(NHANES_FEATURE_SCHEMA),
        'hidden': hidden,
        'dim_emb': dim_emb,
        'time_grid': grid,
        'restrict': args.restrict,
        'target': args.target,
        'meta': cohort.get('meta', {}),
        'best_loss': best_loss,
    }
    torch.save(payload, output_path)
    print(f'\nencoder saved: {output_path}')
    print(f'  meta: {payload["meta"]}')


if __name__ == '__main__':
    main()
