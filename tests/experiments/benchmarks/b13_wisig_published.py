#!/usr/bin/env python3
"""WiSig: the authors' protocols and network, ported line by line from
github.com/WiSig-dataset/wisig-examples (BSD-3-Clause, read 2026-11-09):
`data_utilities.py` and the notebooks d003, d004, d006, d007 (non-equalised)
and d005, d008, d011 (equalised).

Paper: S. Hanna, S. Karunaratne, D. Cabric, "WiSig: A Large-Scale WiFi Signal
Dataset for Receiver and Channel Agnostic RF Fingerprinting", IEEE Access 10:
22808-22818, 2022, doi 10.1109/ACCESS.2022.3154790. Data: CC BY-NC-SA 4.0,
the four compact subsets from cores.ee.ucla.edu/downloads/datasets/wisig/.

Object: one packet, the first 256 complex samples at 25 Msps after packet
detection, stored (256, 2) as I and Q. Label: which transmitter sent it.

Every protocol the authors ran, with the accuracy their notebooks print
(the committed .ipynb outputs, read 2026-11-09). One protocol setting may
hold several folds (the five receiver orders of the receiver study).

    ntx_<N>             ManyTx; the first N of 150 transmitters, all 18
                        receivers and 4 days pooled; seeded packet-level
                        80/10/10 split (d006, non-equalised only).
                        N = 10, 45, 80, 115, 150; printed test accuracy
                        0.8155, 0.6959, 0.5663, 0.5492, 0.5310.
    nrx_<K>             ManyRx, day 1, 10 transmitters. Five receiver orders
                        (np.random.seed(0), five cumulative shuffles of the
                        32 receivers); test on the last 5 receivers; train
                        on the first K+1 (d003, d011). K = 0, 5, 10, 15, 20,
                        25. Score: accuracy on the 5 unseen receivers.
                        Mean over the five orders, printed: non-equalised
                        0.1465 0.2380 0.3719 0.4339 0.5256 0.7150;
                        equalised 0.3234 0.5425 0.6730 0.7467 0.7871 0.7909.
    ndays_<D>           ManySig, receiver 0 only, 6 transmitters. Train on
                        days 1..D+1, test on day 4 (d007, d008). D = 0, 1, 2.
                        Printed: non-equalised 0.4935 0.7407 0.8292;
                        equalised 0.9585 0.9740 0.9987.
    nsig_<K>            ManySig, all 12 receivers and 4 days, the first K+200
                        packets of each transmitter, receiver and day; 100
                        of every K+200 to validation and 100 to test (d004,
                        d005). K = 5, 10, 25, 50, 100, 400, 800. Printed:
                        non-equalised 0.7347 0.8552 0.9305 0.9542 0.9758
                        0.9895 0.9898; equalised 0.9147 0.9453 0.9644 0.9781
                        0.9852 0.9908 0.9926.

Two facts about the authors' code, recorded and ported as they are:

- d003 and d011 shuffle `compact_dataset['rx_list']` in place (the list
  object is shared), then look receivers up by name in that shuffled list
  while the data stay in the original order. So the receiver named in the
  printed lists is not the receiver whose packets were used. The five
  shuffles are cumulative and all run before any training, so the position
  used for a name is its position after the fifth shuffle. Training and test
  receivers stay disjoint, because the mapping is one to one; the protocol is
  still cross-receiver. `nrx_split` reproduces the mapping.
- The packet-level split (d004-d008 same-day test, d006) puts packets of one
  transmitter-receiver-day capture on both sides; the paper notes this is
  why same-day accuracy is near 1.

The network (`create_net` in every notebook): Conv2D 8, 16, 16, 32, 16
filters, kernels (3,2) (3,2) (3,2) (3,1) (3,1), 'same' padding, ReLU;
max-pool (2,1) (2,1) (2,2) (2,1) after the first four; flatten; dense 100
and 80 ReLU, dropout 0.5, softmax; L2 1e-4 on the three dense kernels; Adam
5e-4; categorical cross-entropy weighted by class (max count / count over
the training packets); Keras default batch 32; up to 100 epochs; early stop
on validation loss, patience 5, best-validation-loss weights restored.
Keras to PyTorch: Glorot-uniform kernels and zero biases kept; Adam epsilon
1e-7 as Keras; the L2 term added to the loss as Keras does. The paper's text
lists filters "8, 16, 32, and 16"; the code's five layers are used.

    python tests/experiments/benchmarks/b13_wisig_published.py --protocol ntx_10
    python tests/experiments/benchmarks/b13_wisig_published.py --protocol nrx_25 --equalized 1 --folds 0

Reads <KIRBY_WISIG_DIR>/compact/<subset>.pkl; writes
<KIRBY_BENCH_RESULTS>/b13_wisig_published/<protocol>__eq<e>.json.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

WISIG_DIR = Path(os.environ.get('KIRBY_WISIG_DIR', '/Volumes/seagate/benchmarks_protocol/wisig'))
COMPACT_DIR = WISIG_DIR / 'compact'
RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                                  '/Volumes/seagate/benchmarks_protocol/results'))

SEED = 0                  # data_utilities.shuffle hard-codes np.random.seed(0)
N_RX_ORDERS = 5           # d003 / d011: n_real
N_TEST_RX = 5             # d003 / d011: n_test_rx
EPOCHS = 100
PATIENCE = 5
BATCH = 32                # Keras default; the notebooks pass none
LR = 5e-4
L2 = 1e-4

PROTOCOL_VALUES = {
    'ntx': [10, 45, 80, 115, 150],
    'nrx': [0, 5, 10, 15, 20, 25],
    'ndays': [0, 1, 2],
    'nsig': [5, 10, 25, 50, 100, 400, 800],
}
SUBSET = {'ntx': 'ManyTx', 'nrx': 'ManyRx', 'ndays': 'ManySig', 'nsig': 'ManySig'}
# Equalised versions the authors ran (d006 has no equalised notebook).
EQ_RUN = {'ntx': (0,), 'nrx': (0, 1), 'ndays': (0, 1), 'nsig': (0, 1)}

# Printed by the committed notebooks. ntx/ndays/nsig: test accuracy;
# nrx: accuracy on the unseen receivers, per receiver order (5 lists of 6).
PUBLISHED = {
    ('ntx', 0): dict(zip(PROTOCOL_VALUES['ntx'], [0.8155456, 0.69589835, 0.566258, 0.549167, 0.53099644])),
    ('ndays', 0): dict(zip(PROTOCOL_VALUES['ndays'], [0.4935, 0.7406667, 0.82916665])),
    ('ndays', 1): dict(zip(PROTOCOL_VALUES['ndays'], [0.9585, 0.974, 0.99866664])),
    ('nsig', 0): dict(zip(PROTOCOL_VALUES['nsig'], [0.7346527777777778, 0.8551736111111111, 0.9304861111111111,
                                                    0.9542013888888888, 0.9757638888888889, 0.9895486111111111,
                                                    0.9898263888888889])),
    ('nsig', 1): dict(zip(PROTOCOL_VALUES['nsig'], [0.9147222222222222, 0.9453472222222222, 0.9644097222222222,
                                                    0.9780902777777778, 0.9852430555555556, 0.9907638888888889,
                                                    0.9926388888888888])),
    ('nrx', 0): [[0.1456, 0.2524, 0.5632, 0.5513, 0.5226, 0.7212],
                 [0.0907, 0.2325, 0.4219, 0.438, 0.6864, 0.7775],
                 [0.15244897959183673, 0.15346938775510205, 0.3662244897959184,
                  0.42989795918367346, 0.4023469387755102, 0.8426530612244898],
                 [0.15448979591836734, 0.3006122448979592, 0.2642857142857143,
                  0.4014285714285714, 0.5624489795918367, 0.6004081632653061],
                 [0.18948979591836734, 0.25112244897959185, 0.2436734693877551,
                  0.3486734693877551, 0.45418367346938776, 0.633061224489796]],
    ('nrx', 1): [[0.3165, 0.5837, 0.767, 0.8163, 0.8418, 0.8252],
                 [0.3539, 0.6334, 0.6731, 0.8027, 0.8417, 0.8553],
                 [0.3097758405977584, 0.5153590701535907, 0.7284142797841427,
                  0.7468866749688667, 0.7787463677874636, 0.8270029057700291],
                 [0.28977980889073535, 0.44869131699210635, 0.5234732031574574,
                  0.6456169505608641, 0.7226838388034899, 0.7449106771915247],
                 [0.3469387755102041, 0.5313265306122449, 0.6729591836734694,
                  0.7221428571428572, 0.7504081632653061, 0.7018367346938775]],
}


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def parse_protocol(name):
    kind, val = name.rsplit('_', 1)
    if kind not in PROTOCOL_VALUES or int(val) not in PROTOCOL_VALUES[kind]:
        raise SystemExit(f'unknown protocol {name!r}; kinds {sorted(PROTOCOL_VALUES)} with values '
                         f'{PROTOCOL_VALUES}')
    return kind, int(val)


def published_score(kind, val, eq, fold=None):
    p = PUBLISHED.get((kind, eq))
    if p is None:
        return None
    if kind == 'nrx':
        i = PROTOCOL_VALUES['nrx'].index(val)
        return p[fold][i] if fold is not None else float(np.mean([r[i] for r in p]))
    return p[val]


# ----------------------------------------------------------------------------
# The compact subset as one packet table. A packet's id is its row in the
# subset's packet array; the authors' splits are reproduced as id arrays.
# ----------------------------------------------------------------------------

_COMPACT = {}


def load_compact(subset):
    if subset not in _COMPACT:
        p = COMPACT_DIR / f'{subset}.pkl'
        if not p.exists():
            raise SystemExit(f'{p} missing: download the {subset} compact subset '
                             '(cores.ee.ucla.edu/downloads/datasets/wisig/) and unzip it there')
        with open(p, 'rb') as f:
            _COMPACT[subset] = pickle.load(f)
    return _COMPACT[subset]


_WHOLE = {}


def _load_whole(path):
    """Read the file into memory in one sequential pass. Picking scattered
    rows out of a memory-mapped file on the exFAT Seagate drive hung for over
    ten minutes on 40,296 rows (2026-11-09)."""
    key = str(path)
    if key not in _WHOLE:
        _WHOLE.clear()
        _WHOLE[key] = np.load(path)
    return _WHOLE[key]


def packet_table(subset, eq):
    """(offsets, X): offsets[(tx_i, rx_i, day_i)] = (start, n) into X, the
    (N, 256, 2) float32 array of every packet of the subset, in tx, rx, day
    order, for one equalisation flag. Cached as .npy beside the pickle."""
    cache = WISIG_DIR / 'packets' / f'{subset}_eq{eq}'
    xf, mf = cache.with_suffix('.npy'), cache.with_suffix('.meta.json')
    if xf.exists() and mf.exists():
        meta = json.loads(mf.read_text())
        offs = {tuple(map(int, k.split(','))): tuple(v) for k, v in meta['offsets'].items()}
        return offs, _load_whole(xf)
    c = load_compact(subset)
    e = c['equalized_list'].index(eq)
    blocks, offs, start = [], {}, 0
    for ti in range(len(c['tx_list'])):
        for ri in range(len(c['rx_list'])):
            for di in range(len(c['capture_date_list'])):
                a = np.asarray(c['data'][ti][ri][di][e], dtype=np.float32).reshape(-1, 256, 2)
                offs[(ti, ri, di)] = (start, len(a))
                blocks.append(a)
                start += len(a)
    X = np.concatenate(blocks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache.with_suffix('.tmp.npy'), X)
    os.replace(cache.with_suffix('.tmp.npy'), xf)
    mf.write_text(json.dumps({'subset': subset, 'eq': eq, 'n': int(len(X)),
                              'tx_list': list(map(str, c['tx_list'])),
                              'rx_list': list(map(str, c['rx_list'])),
                              'capture_date_list': list(map(str, c['capture_date_list'])),
                              'offsets': {f'{k[0]},{k[1]},{k[2]}': list(v) for k, v in offs.items()}}))
    return offs, _load_whole(xf)


def packet_meta(subset, eq):
    """Per packet: transmitter, receiver and day index into the subset's lists."""
    offs, X = packet_table(subset, eq)
    tx = np.empty(len(X), np.int32)
    rx = np.empty(len(X), np.int32)
    day = np.empty(len(X), np.int32)
    for (ti, ri, di), (s, n) in offs.items():
        tx[s:s + n], rx[s:s + n], day[s:s + n] = ti, ri, di
    return tx, rx, day


# ----------------------------------------------------------------------------
# data_utilities.py, on packet ids instead of arrays.
# ----------------------------------------------------------------------------

def merge_ids(subset, eq, date_idx, tx_idx, rx_idx, max_sig=None):
    """merge_compact_dataset: per transmitter (in tx_idx order) the packets
    of each date (outer loop) and receiver (inner loop), first max_sig of
    each cell. Returns one id array per transmitter."""
    offs, _ = packet_table(subset, eq)
    out = [[] for _ in tx_idx]
    for d in date_idx:
        for r in rx_idx:
            for i, t in enumerate(tx_idx):
                s, n = offs[(t, r, d)]
                n = n if max_sig is None else min(n, max_sig)
                out[i].append(np.arange(s, s + n))
    return [np.concatenate(o) if o else np.zeros(0, int) for o in out]


def prepare_ids(per_tx, val_frac, test_frac):
    """prepare_dataset: concatenate per transmitter with label i, shuffle with
    np.random.seed(0), then split3: validation = first int(f1 n), test = next
    int(f2 n), training = the rest. Returns ((ids, y) for train, val, test)."""
    ids = np.concatenate(per_tx)
    y = np.concatenate([np.full(len(p), i) for i, p in enumerate(per_tx)])
    np.random.seed(SEED)
    shfl = np.arange(len(ids))
    np.random.shuffle(shfl)
    ids, y = ids[shfl], y[shfl]
    n1, n2 = int(val_frac * len(ids)), int(test_frac * len(ids))
    val = (ids[:n1], y[:n1])
    test = (ids[n1:n1 + n2], y[n1:n1 + n2])
    train = (ids[n1 + n2:], y[n1 + n2:])
    return train, val, test


def rx_orders(n_rx):
    """d003/d011 cell 4, including the in-place shuffle: returns the five
    printed receiver orders as positions into the printed list, and the map
    from a printed name's position to the data position actually loaded."""
    names = list(range(n_rx))           # stand-ins for the receiver names
    np.random.seed(SEED)
    orders = []
    for _ in range(N_RX_ORDERS):
        np.random.shuffle(names)
        orders.append(list(names))
    final = list(names)                  # compact_dataset['rx_list'] after the shuffles
    # merge_compact_dataset looks a name up in the shuffled list:
    # rx_i = final.index(name), and reads data[tx][rx_i].
    data_pos = {name: final.index(name) for name in range(n_rx)}
    return orders, data_pos


def subset_lists(subset, eq):
    """tx_list, rx_list and capture_date_list, from the packet table's meta
    file (so a unit does not reload the pickle)."""
    packet_table(subset, eq)
    m = json.loads((WISIG_DIR / 'packets' / f'{subset}_eq{eq}.meta.json').read_text())
    return {k: m[k] for k in ('tx_list', 'rx_list', 'capture_date_list')}


def protocol_folds(kind, val, eq):
    """List of folds; each fold is a dict with train/val/test (ids, y),
    the number of classes, and what was held out."""
    subset = SUBSET[kind]
    c = subset_lists(subset, eq)
    n_tx, n_rx, n_day = len(c['tx_list']), len(c['rx_list']), len(c['capture_date_list'])
    folds = []
    if kind == 'ntx':
        per_tx = merge_ids(subset, eq, range(n_day), range(val), range(n_rx))
        tr, va, te = prepare_ids(per_tx, 0.1, 0.1)
        folds.append({'train': tr, 'val': va, 'test': te, 'n_class': val,
                      'held_out': 'random 10% of packets (seed 0)'})
    elif kind == 'nrx':
        orders, data_pos = rx_orders(n_rx)
        for r, order in enumerate(orders):
            test_rx = [data_pos[x] for x in order[-N_TEST_RX:]]
            train_rx = [data_pos[x] for x in order[:val + 1]]
            te_per_tx = merge_ids(subset, eq, [0], range(n_tx), test_rx)
            te, _, _ = prepare_ids(te_per_tx, 0.0, 0.0)
            tr, va, _same = prepare_ids(merge_ids(subset, eq, [0], range(n_tx), train_rx), 0.1, 0.1)
            folds.append({'train': tr, 'val': va, 'test': te, 'same_rx_test': _same,
                          'n_class': n_tx, 'held_out': f'receivers at data positions {sorted(test_rx)}',
                          'train_rx': sorted(train_rx), 'test_rx': sorted(test_rx)})
    elif kind == 'ndays':
        rx = [0]
        te, _, _ = prepare_ids(merge_ids(subset, eq, [n_day - 1], range(n_tx), rx), 0.0, 0.0)
        tr, va, same = prepare_ids(merge_ids(subset, eq, range(val + 1), range(n_tx), rx), 0.1, 0.1)
        folds.append({'train': tr, 'val': va, 'test': te, 'same_day_test': same, 'n_class': n_tx,
                      'held_out': f'day {n_day} ({c["capture_date_list"][-1]}), receiver 0'})
    elif kind == 'nsig':
        frac = 100 / (val + 200)
        tr, va, te = prepare_ids(merge_ids(subset, eq, range(n_day), range(n_tx), range(n_rx),
                                           max_sig=val + 200), frac, frac)
        folds.append({'train': tr, 'val': va, 'test': te, 'n_class': n_tx,
                      'held_out': f'random {frac:.4f} of packets (seed 0)'})
    return folds


def norm(X):
    """data_utilities.norm: each packet to unit average power."""
    X = np.asarray(X, dtype=np.float32)
    pwr = np.sqrt(np.mean(np.sum(X ** 2, axis=-1), axis=-1))
    return X / pwr[:, None, None]


def class_weights(y, n):
    stat = np.bincount(y, minlength=n).astype(float)
    with np.errstate(divide='ignore'):
        w = stat.max() / stat
    w[~np.isfinite(w)] = 0.0
    return w


# ----------------------------------------------------------------------------
# The network, PyTorch port of create_net.
# ----------------------------------------------------------------------------

def build_net(n_class, in_ch=1, width=2):
    import torch.nn as nn

    class WiSigCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Keras Conv2D on (256, width, 1), 'same' padding. Kernel (3,2) on
            # an even axis: Keras pads 0 before and 1 after; torch 'same' does
            # the same for an even kernel (extra padding at the end).
            self.c1 = nn.Conv2d(in_ch, 8, (3, 2), padding='same')
            self.c2 = nn.Conv2d(8, 16, (3, 2), padding='same')
            self.c3 = nn.Conv2d(16, 16, (3, 2), padding='same')
            self.c4 = nn.Conv2d(16, 32, (3, 1), padding='same')
            self.c5 = nn.Conv2d(32, 16, (3, 1), padding='same')
            self.p1, self.p2 = nn.MaxPool2d((2, 1)), nn.MaxPool2d((2, 1))
            self.p3 = nn.MaxPool2d((2, 2)) if width >= 2 else nn.MaxPool2d((2, 1))
            self.p4 = nn.MaxPool2d((2, 1))
            w = width // 2 if width >= 2 else 1
            self.d1 = nn.Linear(16 * 16 * w, 100)
            self.d2 = nn.Linear(100, 80)
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(80, n_class)
            self.relu = nn.ReLU()
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def features(self, x):
            # x: (b, 256, width) -> Keras Reshape((256, width, 1)), channels last
            x = x.unsqueeze(1) if x.dim() == 3 else x
            r = self.relu
            x = self.p1(r(self.c1(x)))
            x = self.p2(r(self.c2(x)))
            x = self.p3(r(self.c3(x)))
            x = self.p4(r(self.c4(x)))
            x = r(self.c5(x))
            # Keras Flatten is channels-last: (h, w, c) order
            x = x.permute(0, 2, 3, 1).flatten(1)
            x = r(self.d1(x))
            return r(self.d2(x))

        def forward(self, x):
            return self.out(self.drop(self.features(x)))

        def l2(self):
            return L2 * sum((m.weight ** 2).sum() for m in (self.d1, self.d2, self.out))

    return WiSigCNN()


def train_net(X_tr, y_tr, X_va, y_va, n_class, seed, epochs=EPOCHS, log_prefix=''):
    """classifier.fit(..., class_weight, validation_data, callbacks=[best
    val_loss checkpoint, EarlyStopping(val_loss, patience 5)], epochs 100),
    then load the best weights. Returns the trained model and a history."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('KIRBY_TORCH_THREADS') or 4)
    torch.set_num_threads(threads)
    width = X_tr.shape[-1]
    net = build_net(n_class, width=width)
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-7)
    cw = torch.tensor(class_weights(y_tr, n_class), dtype=torch.float32)
    Xt = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))
    Xv = torch.from_numpy(np.ascontiguousarray(X_va, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(y_va, dtype=np.int64))
    best, best_state, wait, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(Xt), generator=g)       # Keras shuffles each epoch
        tot = 0.0
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            logits = net(Xt[b])
            # Keras: sum(class_weight * per-sample loss) / batch size, plus L2
            loss = (F.cross_entropy(logits, yt[b], reduction='none') * cw[yt[b]]).mean() + net.l2()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            vl, va_acc = 0.0, 0
            for i in range(0, len(Xv), 4096):
                lo = net(Xv[i:i + 4096])
                vl += float(F.cross_entropy(lo, yv[i:i + 4096], reduction='sum'))
                va_acc += int((lo.argmax(1) == yv[i:i + 4096]).sum())
            vl = vl / max(len(Xv), 1) + float(net.l2())
        hist.append({'epoch': ep + 1, 'loss': tot / len(Xt), 'val_loss': vl,
                     'val_acc': va_acc / max(len(Xv), 1), 'seconds': time.time() - t0})
        log(f'{log_prefix}epoch {ep + 1}: loss {tot / len(Xt):.4f} val_loss {vl:.4f} '
            f'val_acc {va_acc / max(len(Xv), 1):.4f} ({time.time() - t0:.0f} s)')
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def predict(net, X, embed=False):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 4096], dtype=np.float32))
            out.append((net.features(xb) if embed else net(xb)).numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------

def run(protocol, eq, folds_only=None, epochs=EPOCHS):
    kind, val = parse_protocol(protocol)
    subset = SUBSET[kind]
    out = RESULTS_DIR / 'b13_wisig_published' / f'{protocol}__eq{eq}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {'folds': {}}
    _, X = packet_table(subset, eq)
    for f, fold in enumerate(protocol_folds(kind, val, eq)):
        if folds_only is not None and f not in folds_only:
            continue
        if str(f) in res['folds'] and res['folds'][str(f)].get('epochs_cap') == epochs:
            continue
        (tri, ytr), (vai, yva), (tei, yte) = fold['train'], fold['val'], fold['test']
        t0 = time.time()
        log(f'{protocol} eq{eq} fold {f}: {len(tri)} train, {len(vai)} validation, {len(tei)} test packets, '
            f'{fold["n_class"]} transmitters; held out: {fold["held_out"]}')
        net, hist = train_net(norm(X[tri]), ytr,
                              norm(X[vai]), yva, fold['n_class'], SEED + f, epochs,
                              log_prefix=f'{protocol} eq{eq} fold {f} ')
        acc = float((predict(net, norm(X[tei])).argmax(1) == yte).mean())
        row = {'accuracy': acc, 'published': published_score(kind, val, eq, f if kind == 'nrx' else None),
               'n_train': int(len(tri)), 'n_val': int(len(vai)), 'n_test': int(len(tei)),
               'n_class': fold['n_class'], 'held_out': fold['held_out'], 'epochs_run': len(hist),
               'epochs_cap': epochs, 'history': hist, 'seconds': time.time() - t0}
        for extra in ('same_rx_test', 'same_day_test'):
            if extra in fold:
                si, sy = fold[extra]
                row[extra + '_accuracy'] = float((predict(net, norm(X[si])).argmax(1) == sy).mean())
        res['folds'][str(f)] = row
        res.update({'protocol': protocol, 'equalized': eq, 'subset': subset,
                    'debug': epochs != EPOCHS})
        out.write_text(json.dumps(res, indent=1))
        log(f'{protocol} eq{eq} fold {f}: accuracy {acc:.4f}, published {row["published"]} '
            f'({row["seconds"]:.0f} s, {len(hist)} epochs)')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True, help='e.g. ntx_10, nrx_25, ndays_2, nsig_5')
    ap.add_argument('--equalized', type=int, required=True, choices=[0, 1])
    ap.add_argument('--folds', default=None, help='comma list (nrx has 5 receiver orders)')
    ap.add_argument('--epochs', type=int, default=EPOCHS,
                    help=f'cap; {EPOCHS} is the protocol, anything else is marked debug')
    a = ap.parse_args()
    run(a.protocol, a.equalized, [int(x) for x in a.folds.split(',')] if a.folds else None, a.epochs)


if __name__ == '__main__':
    main()
