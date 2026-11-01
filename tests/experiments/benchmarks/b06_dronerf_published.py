#!/usr/bin/env python3
"""DroneRF as published by Al-Sa'd et al.: which drone, and which flight mode,
from a 2.4 GHz RF recording.

Source of the protocol: Al-Sa'd, Al-Ali, Mohamed, Khattab, Erbad, "RF-based
drone detection and identification using deep learning approaches: an
initiative towards a large open source drone database", Future Generation
Computer Systems 100:86-97, 2019. Code: github.com/Al-Sad/DroneRF
(Apache-2.0): Matlab/Main_1_Data_aggregation.m, Matlab/Main_2_Data_labeling.m,
Python/Classification.py. Published 10-fold mean accuracies:

    task              classes   accuracy
    detection         2         99.7
    drone type        4         84.5
    drone and mode    10        46.8

This file is a line-for-line port of the three published scripts, with one
addition: --split grouped. Nothing else is changed.

Main_1 (port: `aggregate`): each raw file pair <BUI>L_<n>.csv / <BUI>H_<n>.csv
is cut into chunks of L = 100,000 samples. Per chunk and per band: subtract
the chunk mean, FFT of length M = 2,048 (MATLAB fft(x, M) TRUNCATES x to its
first 2,048 samples, and so does numpy; the other 97,952 samples of the chunk
are not used), magnitude, fftshift, keep the upper 1,024 bins. The high band
is scaled so its first Q = 10 bins continue the low band's last 10. The two
halves are stacked (2,048 bins) and squared.

Main_2 (port: `assemble`): each BUI code's block is divided by its own
maximum over all its chunks (test chunks included; recorded in
NEW_DATASETS.md section 6, not changed here), the ten blocks are
concatenated, and three labels are attached: 2-class (background 0, any
drone 1), 4-class (background 0, Bebop 1, AR 2, Phantom 3), 10-class (the BUI
index 0..9 in the order background, Bebop modes 1-4, AR modes 1-4, Phantom).

Classification.py (port: `run_task`): features are the first 2,047 of the
2,048 bins (`Data[0:2047]` in the original), three hidden layers of 128 ReLU
units (number_inner_neurons = 256, halved), sigmoid output over one-hot
targets, MSE loss, Adam at the Keras default learning rate 1e-3, 200 epochs,
batch 10, accuracy by arg max, mean over 10 folds. The network is written in
PyTorch with Keras's initialisation (Glorot uniform weights, zero biases).
Adam's epsilon differs (Keras 1e-7, PyTorch 1e-8); nothing else does.

Splits:
    --split published   StratifiedKFold(n_splits=10, shuffle=True,
                        random_state=1) over chunks, stratified on the
                        task's own label, exactly as in Classification.py.
                        Chunks of one recording land in both the training
                        and the test fold. Comparison with the published
                        numbers only.
    --split grouped     StratifiedGroupKFold(n_splits=10, shuffle=True,
                        random_state=1), groups = recording (<BUI>_<n>, 227
                        of them), stratified on the task's label. No
                        published number exists under this split. This is
                        the split for every other use (owner, 2026-10-29).

Data: Mendeley Data v1, doi 10.17632/f4c2b4n755.1, CC BY 4.0, 454 CSV files,
over 40 GB. Expected under <KIRBY_DRONERF_DIR>/raw/ (override the directory
with KIRBY_DRONERF_DIR or --raw-dir; default
/Volumes/seagate/benchmarks_protocol/dronerf). The spectra are cached under
<KIRBY_DRONERF_DIR>/spectra/<BUI>.npy after the first run.

Output: /Volumes/seagate/benchmarks_protocol/results/b06_dronerf_<split>.json

    python tests/experiments/benchmarks/b06_dronerf_published.py --split published
    python tests/experiments/benchmarks/b06_dronerf_published.py --split grouped

--epochs and --folds exist for smoke tests only; any value other than 200
and 10 is not the published protocol and is written into the output file.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

DRONERF_DIR = Path(os.environ.get(
    'KIRBY_DRONERF_DIR', '/Volumes/seagate/benchmarks_protocol/dronerf'))
RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                               '/Volumes/seagate/benchmarks_protocol/results'))

# Main_1_Data_aggregation.m, in the order the labelling script concatenates them.
BUI = [
    ('00000', 'background'),
    ('10000', 'bebop_mode1'), ('10001', 'bebop_mode2'),
    ('10010', 'bebop_mode3'), ('10011', 'bebop_mode4'),
    ('10100', 'ar_mode1'), ('10101', 'ar_mode2'),
    ('10110', 'ar_mode3'), ('10111', 'ar_mode4'),
    ('11000', 'phantom_mode1'),
]
# Highest file index n per BUI (the loop is `for n = 0:N`).
N_LAST = {'00000': 40, '10111': 17}
N_LAST_DEFAULT = 20
M = 2048     # FFT length (and bins per chunk after stacking the two bands)
L = 100_000  # samples per chunk
Q = 10       # bins used to join the two bands
N_FEATURES = 2047  # Classification.py: x = Data[0:2047]

# Main_2_Data_labeling.m: label 1 and label 2 by BUI position in the list above.
LABEL_2CLASS = [0] + [1] * 9
LABEL_4CLASS = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3]
TASKS = {2: LABEL_2CLASS, 4: LABEL_4CLASS, 10: list(range(10))}

PUBLISHED = {2: 99.7, 4: 84.5, 10: 46.8}

# Classification.py
HIDDEN_LAYERS = 3
HIDDEN_UNITS = 256 // 2
EPOCHS = 200
BATCH = 10
FOLDS = 10
SPLIT_SEED = 1
LR = 1e-3  # Keras Adam default


def _half_spectrum(x):
    """abs(fftshift(fft(x - mean(x), M))), upper half: bins 1024..2047."""
    xf = np.abs(np.fft.fftshift(np.fft.fft(x - x.mean(), M)))
    return xf[M // 2:]


def chunk_spectra(low, high):
    """Main_1's inner loop for one recording. Returns (n_chunks, 2048)."""
    n_chunks = len(low) // L
    out = np.empty((n_chunks, M), dtype=np.float64)
    for i in range(n_chunks):
        st, fi = i * L, (i + 1) * L
        xf = _half_spectrum(low[st:fi])
        yf = _half_spectrum(high[st:fi])
        yf = yf * xf[-Q:].mean() / yf[:Q].mean()
        out[i] = np.concatenate([xf, yf])
    return out ** 2


def aggregate(raw_dir, spectra_dir, log):
    """Main_1 for every BUI, cached as <spectra_dir>/<BUI>.npy and <BUI>_rec.npy
    (recording index per chunk)."""
    spectra_dir.mkdir(parents=True, exist_ok=True)
    for bui, _ in BUI:
        spec_path = spectra_dir / f'{bui}.npy'
        rec_path = spectra_dir / f'{bui}_rec.npy'
        if spec_path.exists() and rec_path.exists():
            continue
        t0 = time.time()
        blocks, recs = [], []
        for n in range(N_LAST.get(bui, N_LAST_DEFAULT) + 1):
            low = np.loadtxt(raw_dir / f'{bui}L_{n}.csv', delimiter=',', dtype=np.float64)
            high = np.loadtxt(raw_dir / f'{bui}H_{n}.csv', delimiter=',', dtype=np.float64)
            low, high = np.ravel(low), np.ravel(high)
            s = chunk_spectra(low, high)
            blocks.append(s)
            recs.append(np.full(len(s), n, dtype=np.int32))
        np.save(spec_path, np.concatenate(blocks))
        np.save(rec_path, np.concatenate(recs))
        log(f'{bui}: {sum(len(b) for b in blocks)} chunks from '
            f'{len(blocks)} recordings in {time.time() - t0:.0f} s')


def assemble(spectra_dir):
    """Main_2: per-BUI max scaling, concatenation, labels. Also returns the
    recording id of each chunk (BUI index * 1000 + file index), which the
    published scripts do not keep."""
    X, labels, groups = [], {k: [] for k in TASKS}, []
    for b, (bui, _) in enumerate(BUI):
        data = np.load(spectra_dir / f'{bui}.npy')
        rec = np.load(spectra_dir / f'{bui}_rec.npy')
        data = data / data.max()
        X.append(data)
        for k, lab in TASKS.items():
            labels[k].append(np.full(len(data), lab[b], dtype=np.int64))
        groups.append(b * 1000 + rec)
    X = np.concatenate(X)[:, :N_FEATURES]
    labels = {k: np.concatenate(v) for k, v in labels.items()}
    return X, labels, np.concatenate(groups)


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        layers, d = [], n_in
        for _ in range(HIDDEN_LAYERS):
            lin = torch.nn.Linear(d, HIDDEN_UNITS)
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.ReLU()]
            d = HIDDEN_UNITS
        out = torch.nn.Linear(d, n_out)
        torch.nn.init.xavier_uniform_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers += [out, torch.nn.Sigmoid()]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_predict(X_tr, Y_tr, X_te, epochs, seed):
    torch.manual_seed(seed)
    X_tr = torch.as_tensor(X_tr, dtype=torch.float32)
    Y_tr = torch.as_tensor(Y_tr, dtype=torch.float32)
    X_te = torch.as_tensor(X_te, dtype=torch.float32)
    model = Net(X_tr.shape[1], Y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    n = len(X_tr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)  # Keras fit shuffles each epoch
        for st in range(0, n, BATCH):
            idx = perm[st:st + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), Y_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(X_te).numpy()


def make_folds(split, y, groups, folds):
    if split == 'published':
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SPLIT_SEED)
        return list(kf.split(np.zeros(len(y)), y))
    kf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SPLIT_SEED)
    return list(kf.split(np.zeros(len(y)), y, groups))


def run_task(X, y, groups, n_classes, split, epochs, folds, log):
    Y = np.eye(n_classes)[y]
    accs = []
    for f, (tr, te) in enumerate(make_folds(split, y, groups, folds)):
        t0 = time.time()
        pred = fit_predict(X[tr], Y[tr], X[te], epochs, seed=SPLIT_SEED + f)
        acc = 100.0 * (pred.argmax(1) == y[te]).mean()
        accs.append(acc)
        leak = len(np.intersect1d(groups[tr], groups[te]))
        log(f'{n_classes}-class {split} fold {f + 1}/{folds}: accuracy {acc:.2f}, '
            f'{len(te)} test chunks, recordings in both train and test: {leak}; '
            f'{time.time() - t0:.0f} s')
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['published', 'grouped'], default='published')
    ap.add_argument('--tasks', default='2,4,10')
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--folds', type=int, default=FOLDS)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else DRONERF_DIR / 'raw'
    spectra_dir = DRONERF_DIR / 'spectra'
    out_path = Path(args.out) if args.out else RESULTS_DIR / f'b06_dronerf_{args.split}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix('.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    t_all = time.time()
    aggregate(raw_dir, spectra_dir, log)
    X, labels, groups = assemble(spectra_dir)
    log(f'{len(X)} chunks, {X.shape[1]} features, {len(np.unique(groups))} recordings; '
        f'split {args.split}')

    protocol = {'epochs': args.epochs, 'folds': args.folds, 'batch': BATCH,
                'hidden': [HIDDEN_UNITS] * HIDDEN_LAYERS, 'lr': LR,
                'as_published': args.epochs == EPOCHS and args.folds == FOLDS}
    out = {'split': args.split, 'protocol': protocol, 'n_chunks': int(len(X)),
           'n_recordings': int(len(np.unique(groups))), 'tasks': {}}
    for k in [int(t) for t in args.tasks.split(',')]:
        accs = run_task(X, labels[k], groups, k, args.split, args.epochs, args.folds, log)
        out['tasks'][str(k)] = {'fold_accuracy': accs, 'mean_accuracy': float(np.mean(accs)),
                                'published_mean_accuracy': PUBLISHED[k]
                                if args.split == 'published' else None}
        log(f'{k}-class {args.split}: mean accuracy {np.mean(accs):.2f}'
            + (f' (published {PUBLISHED[k]})' if args.split == 'published' else ''))
    out['seconds'] = time.time() - t_all
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    log(f'written {out_path}; total {out["seconds"]:.0f} s')


if __name__ == '__main__':
    main()
