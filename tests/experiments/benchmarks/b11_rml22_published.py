#!/usr/bin/env python3
"""RML22 (Sathyanarayanan et al.), modulation class of one 2 x 128 I/Q frame,
scored on frames the model never saw.

Source: github.com/venkateshsathya/RML22 (no licence file), code/RML22.ipynb
cells 5 (train) and 7 (test), code/CNNArchitecture.py, code/TrainHelper.py,
read 2026-11-09. Paper: IEEE Trans. Wireless Commun. 22(11):7663-7675, 2023.
Data: the pickle `RML22` in the Google Drive folder the README links
(1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J): a dict (modulation, SNR) -> frames.
10 modulations x 21 SNR levels (-20..20 dB, step 2) x 2,000 frames = 420,000.

The fault in the published code (owner told 2026-11-09):
    cell 5 splits all frames once, 80/20, random_state=1 (the 20% is held
    out), then the 80% again 80/20, random_state=1 (validation, used for
    early stopping and for keeping the best weights). The held-out 20% is
    never scored. Cell 7 instead calls train_test_split(test_size=0.2,
    random_state=1) inside every (modulation, SNR) block of 2,000 frames,
    a draw unrelated to cell 5's, so most of its "test" frames were
    training or validation frames. Its confusion matrix uses a third such
    split over -4..20 dB.

Stages:
    released   the authors' trained weights (code/RML22_model.pt), scored on
               (a) cell 5's held-out 20%, the honest number, and (b) cell 7's
               per-block re-split, the published protocol. For each, the
               number of scored frames that were in training and in
               validation is counted and written out.
    train      cell 5's training, ported line for line, on a random split
               given by --test-size and --seed (no defaults), then scored
               on that split's held-out frames.

Port notes (train): Adam lr 1e-3, weight decay 5e-4, batch 32, StepLR step 8
gamma 0.1, up to 200 epochs, early stop after 16 epochs without a lower
validation loss, and, as in TrainHelper.train_model, the best weights are
reloaded every time validation loss does not improve. TrainParams['clip']
is set in the notebook but never used, so no clipping. The notebook seeds
nothing; here numpy and torch are seeded with --seed, the one addition.

Output: <KIRBY_BENCH_RESULTS or <data-dir>/results>/b11_rml22_<stage>[_...].json
with overall accuracy, macro-F1, accuracy per SNR level, the mean over SNR
levels <= 0 dB and >= 0 dB (STF-GCN Table IV's columns), per modulation,
and the confusion matrix.

    export KIRBY_RML22_DIR=/Volumes/seagate/benchmarks_protocol/rml22
    python tests/experiments/benchmarks/b11_rml22_published.py --stage released
    python tests/experiments/benchmarks/b11_rml22_published.py --stage train --test-size 0.2 --seed 1
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Cell 5 / cell 7 constants, verbatim.
LABEL_DICT = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6,
              'GFSK': 7, 'PAM4': 8, 'QPSK': 9}
MODULATION_TYPES = ['QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK']
SNR_RANGE = np.arange(-20, 21, 2)
NUM_FRAMES = 2000
L = 128
VALIDATION_SIZE = 0.2
PUBLISHED_SEED = 1
INV_LABEL = {v: k for k, v in LABEL_DICT.items()}


class CNNArchitecture(nn.Module):
    """code/CNNArchitecture.py, relu branch (the one the notebook uses)."""

    def __init__(self, num_classes):
        super().__init__()
        k, p = 3, 1
        self.conv1 = nn.Conv1d(2, 64, kernel_size=k, padding=p)
        self.conv1_norm = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=k, padding=p)
        self.conv2_norm = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=k, padding=p)
        self.conv3_norm = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        for m in (self.conv1, self.conv2, self.conv3, self.fc1, self.fc2):
            torch_init.xavier_normal_(m.weight)

    def forward(self, batch):
        batch = self.pool1(func.relu(self.conv1_norm(self.conv1(batch))))
        batch = self.pool2(func.relu(self.conv2_norm(self.conv2(batch))))
        batch = self.pool3(func.relu(self.conv3_norm(self.conv3(batch))))
        batch = batch.view(batch.shape[0], -1)
        batch = self.dropout(self.fc1(batch))
        return self.fc2(batch)


def data_dir(args):
    d = args.data_dir or os.environ.get('KIRBY_RML22_DIR')
    if not d:
        raise SystemExit('set KIRBY_RML22_DIR or --data-dir (the folder holding the RML22 pickle)')
    return Path(d)


def load_frames(ddir):
    """Cell 5's assembly: mods and SNRs sorted from the keys, first 2,000
    frames of each block, first L samples of each frame. Returns data
    (N, 2, 128) float32, label (N,), snr (N,), and each frame's position
    inside its block (0..1999), which cell 7's re-split indexes."""
    with open(ddir / 'RML22', 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], dataset.keys())))), [1, 0])
    X, lab, snr_col, pos = [], [], [], []
    for mod in mods:
        if mod in MODULATION_TYPES:
            for snr in snrs:
                if snr in SNR_RANGE:
                    block = dataset[(mod, snr)][0:NUM_FRAMES]
                    X.append(block)
                    lab += [LABEL_DICT[mod]] * NUM_FRAMES
                    snr_col += [snr] * NUM_FRAMES
                    pos += list(range(NUM_FRAMES))
    data = np.vstack(X)[:, :, 0:L].astype(np.float32)
    return data, np.array(lab), np.array(snr_col), np.array(pos)


def cell5_split(n, label, test_size, seed):
    """Indices of cell 5's train / validation / held-out sets. Splitting
    indices with train_test_split gives the same rows as splitting the
    arrays (the shuffle depends only on n, test_size and random_state)."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    tr, va = train_test_split(tr, test_size=VALIDATION_SIZE, random_state=seed)
    return tr, va, te


def cell7_indices(label, snr, pos):
    """Global indices cell 7 scores: inside each (mod, SNR) block, the test
    part of train_test_split(test_size=0.2, random_state=1) of its 2,000."""
    _, local_te = train_test_split(np.arange(NUM_FRAMES), test_size=VALIDATION_SIZE,
                                   random_state=PUBLISHED_SEED)
    local_te = set(local_te.tolist())
    return np.where(np.array([p in local_te for p in pos]))[0]


def predict(model, data, device, batch_size=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            out.append(model(torch.from_numpy(data[i:i + batch_size]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def scores(y, yhat, snr):
    per_snr = {int(s): float(np.mean(yhat[snr == s] == y[snr == s])) for s in SNR_RANGE if np.any(snr == s)}
    le0 = [v for s, v in per_snr.items() if s <= 0]
    ge0 = [v for s, v in per_snr.items() if s >= 0]
    labels = sorted(INV_LABEL)
    return {
        'n_frames': int(len(y)),
        'accuracy': float(np.mean(y == yhat)),
        'n_wrong': int(np.sum(y != yhat)),
        'macro_f1': float(f1_score(y, yhat, average='macro')),
        'accuracy_per_snr': per_snr,
        'mean_over_snr_le_0dB': float(np.mean(le0)) if le0 else None,
        'mean_over_snr_ge_0dB': float(np.mean(ge0)) if ge0 else None,
        'accuracy_per_modulation': {INV_LABEL[c]: float(np.mean(yhat[y == c] == c)) for c in labels},
        'confusion_labels': [INV_LABEL[c] for c in labels],
        'confusion': confusion_matrix(y, yhat, labels=labels).tolist(),
    }


def overlap(idx, tr, va):
    s = set(idx.tolist())
    return {'scored_frames_in_training': len(s & set(tr.tolist())),
            'scored_frames_in_validation': len(s & set(va.tolist()))}


def out_dir(ddir):
    d = Path(os.environ.get('KIRBY_BENCH_RESULTS') or ddir / 'results')
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)
    print('wrote', path, flush=True)


def device_pick():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stage_released(args):
    ddir = data_dir(args)
    data, label, snr, pos = load_frames(ddir)
    tr, va, te = cell5_split(len(label), label, VALIDATION_SIZE, PUBLISHED_SEED)
    c7 = cell7_indices(label, snr, pos)
    dev = device_pick()
    model = CNNArchitecture(len(MODULATION_TYPES))
    model.load_state_dict(torch.load(ddir / 'RML22_model.pt', map_location='cpu'))
    model.to(dev)
    res = {'stage': 'released', 'weights': 'code/RML22_model.pt', 'device': str(dev),
           'n_train': len(tr), 'n_validation': len(va), 'n_held_out': len(te)}
    for name, idx in (('held_out_cell5', te), ('published_cell7', c7)):
        t0 = time.time()
        yhat = predict(model, data[idx], dev)
        r = scores(label[idx], yhat, snr[idx])
        r.update(overlap(idx, tr, va))
        r['seconds'] = round(time.time() - t0, 1)
        res[name] = r
        print(f"{name}: accuracy {r['accuracy']:.4f} on {r['n_frames']} frames, "
              f"{r['scored_frames_in_training']} in training, {r['scored_frames_in_validation']} in validation",
              flush=True)
    res['result_status'] = 'ok'
    write(out_dir(ddir) / 'b11_rml22_released.json', res)


def stage_train(args):
    if args.test_size is None or args.seed is None:
        raise SystemExit('--stage train needs --test-size and --seed (no defaults)')
    ddir = data_dir(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, label, snr, pos = load_frames(ddir)
    tr, va, te = cell5_split(len(label), label, args.test_size, args.seed)
    dev = device_pick()
    model = CNNArchitecture(len(MODULATION_TYPES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.1)
    crit = nn.CrossEntropyLoss()
    Xtr, ytr = torch.from_numpy(data[tr]), torch.from_numpy(label[tr]).long()
    Xva, yva = torch.from_numpy(data[va]), torch.from_numpy(label[va]).long()

    def val_loss():
        model.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(Xva), 1024):
                tot += crit(model(Xva[i:i + 1024].to(dev)), yva[i:i + 1024].to(dev)).item()
                nb += 1
        return tot / nb

    tag = f'ts{args.test_size}_seed{args.seed}' + (f'_ep{args.epochs}' if args.epochs != 200 else '')
    ckpt = out_dir(ddir) / f'b11_rml22_train_{tag}.pt'
    best, stop, log = float('inf'), 0, []
    t_start = time.time()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(model(Xtr[b].to(dev)), ytr[b].to(dev))
            loss.backward()
            opt.step()
        vl = val_loss()
        if vl < best:
            best, stop = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            model.load_state_dict(torch.load(ckpt))   # TrainHelper reloads the best weights
            stop += 1
        log.append({'epoch': epoch, 'val_loss': vl, 'seconds': round(time.time() - t0, 1)})
        print(f'epoch {epoch} val_loss {vl:.4f} best {best:.4f} {time.time() - t0:.0f}s', flush=True)
        sched.step()
        if stop >= 16:
            break
    model.load_state_dict(torch.load(ckpt))
    yhat = predict(model, data[te], dev)
    r = scores(label[te], yhat, snr[te])
    r.update(overlap(te, tr, va))
    res = {'stage': 'train', 'test_size': args.test_size, 'seed': args.seed, 'device': str(dev),
           'epochs_max': args.epochs, 'epochs_run': epoch, 'train_seconds': round(time.time() - t_start, 1),
           'n_train': len(tr), 'n_validation': len(va), 'n_held_out': len(te),
           'epoch_log': log, 'held_out': r,
           'debug': args.epochs != 200,
           'result_status': 'ok'}
    print(f"held-out accuracy {r['accuracy']:.4f} on {r['n_frames']} frames", flush=True)
    write(out_dir(ddir) / f'b11_rml22_train_{tag}.json', res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['released', 'train'])
    ap.add_argument('--data-dir')
    ap.add_argument('--test-size', type=float)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--epochs', type=int, default=200,
                    help='200 is the published maximum; anything else is marked debug')
    args = ap.parse_args()
    {'released': stage_released, 'train': stage_train}[args.stage](args)


if __name__ == '__main__':
    main()
