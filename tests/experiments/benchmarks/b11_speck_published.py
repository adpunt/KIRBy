#!/usr/bin/env python3
"""Speck32/64 real-or-random (Gohr, CRYPTO 2019; github.com/agohr/deep_speck):
a basic check that this code lands near the published accuracies, before any
screen is written.

Object: one ciphertext pair (C, C') of r-round Speck32/64 under one random
key. Label 1: the plaintexts differ by 0x0040/0000. Label 0: the second
plaintext is random. Each pair has its own key, so no two rows share
anything but the cipher. Gohr: 10^7 training pairs, 10^6 test pairs.

Gohr draws every value from os.urandom, so his data has no seed. Here the
same draws come from numpy's default_rng(seed); the generator is otherwise a
line-for-line port of speck.make_train_data, and it is checked against the
Speck32/64 test vector before anything runs.

Stages
    check      the Speck32/64 test vector, and that the port's encryption
               equals deep_speck/speck.py's on the same inputs (when
               --deep-speck points at a clone)
    published  Gohr's four pretrained one-block nets (net5..8_small.h5),
               read from the .h5 weights with h5py and run in numpy (the
               Keras 2.1 / TensorFlow 1.6 the repo was tested on is not
               needed), on --n-test fresh pairs per round count. Gohr's
               eval.py does the same. Published Table 2 is for the ten-block
               nets at 5 and 6 rounds; the repo's readme says the one-block
               nets shipped are "slightly lower" there.
    trees      LightGBM on each candidate view (with --deep-speck, also the
               64 values of the pretrained net's last hidden layer), --n-train training pairs,
               --n-test test pairs, per round count. Benamira et al. 2021
               Table 9: LightGBM on the raw 64 bits, 5 rounds, 0.7634.

Results: <KIRBY_BENCH_RESULTS>/b11_speck/<stage>.json.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                                  '/Volumes/seagate/benchmarks_protocol/results')) / 'b11_speck'
DIFF = (0x0040, 0x0000)
MASK = 0xFFFF

# Gohr 2019 Table 2 (ten-block nets at 5 and 6 rounds, one-block at 7 and 8;
# N7 and N8 trained with key averaging / staged training) and the full DDT.
PUBLISHED_NET = {5: 0.929, 6: 0.788, 7: 0.616, 8: 0.514}
PUBLISHED_DDT = {5: 0.911, 6: 0.758, 7: 0.591, 8: 0.512}
# Bacuieti, Batina, Picek 2022 Table 2: one block with residual, direct training.
PUBLISHED_ONE_BLOCK = {5: 0.927, 6: 0.783, 7: 0.608, 8: 0.500}


# ── Speck32/64, vectorised over uint16 arrays ─────────────────────────────

def _ror(x, k):
    return ((x >> k) | (x << (16 - k))) & MASK


def _rol(x, k):
    return ((x << k) | (x >> (16 - k))) & MASK


def enc_one_round(x, y, k):
    x = _ror(x, 7)
    x = (x + y) & MASK
    x = x ^ k
    y = _rol(y, 2)
    y = y ^ x
    return x, y


def expand_key(k, rounds):
    """k: four uint32 arrays (k[0] is the most significant word), as speck.py."""
    ks = [None] * rounds
    ks[0] = k[3]
    l = [k[2], k[1], k[0]]
    for i in range(rounds - 1):
        l[i % 3], ks[i + 1] = enc_one_round(l[i % 3], ks[i], i)
    return ks


def encrypt(x, y, ks):
    for k in ks:
        x, y = enc_one_round(x, y, k)
    return x, y


def check_testvector():
    key = [np.array([v], dtype=np.uint32) for v in (0x1918, 0x1110, 0x0908, 0x0100)]
    ks = expand_key(key, 22)
    x, y = encrypt(np.array([0x6574], dtype=np.uint32), np.array([0x694c], dtype=np.uint32), ks)
    return int(x[0]) == 0xa868 and int(y[0]) == 0x42f2


def make_pairs(n, rounds, seed):
    """Port of speck.make_train_data. Returns the four ciphertext words
    (L0, R0, L1, R1) as uint32 arrays and the label."""
    rng = np.random.default_rng(seed)
    w = lambda m: rng.integers(0, 1 << 16, size=m, dtype=np.uint32)
    y = rng.integers(0, 2, size=n, dtype=np.uint8)
    keys = [w(n) for _ in range(4)]
    p0l, p0r = w(n), w(n)
    p1l, p1r = p0l ^ DIFF[0], p0r ^ DIFF[1]
    nr = int(np.sum(y == 0))
    p1l[y == 0] = w(nr)
    p1r[y == 0] = w(nr)
    ks = expand_key(keys, rounds)
    c0l, c0r = encrypt(p0l, p0r, ks)
    c1l, c1r = encrypt(p1l, p1r, ks)
    return (c0l, c0r, c1l, c1r), y


def to_bits(words):
    """Most significant bit first per word, words in the given order, as
    speck.convert_to_binary."""
    cols = []
    for a in words:
        a = np.asarray(a, dtype=np.uint32)
        for off in range(15, -1, -1):
            cols.append(((a >> off) & 1).astype(np.uint8))
    return np.stack(cols, axis=1)


# ── Candidate views, all key-free (research file speck.md, Table B) ───────

def views(ct):
    L0, R0, L1, R1 = ct
    dL, dR = L0 ^ L1, R0 ^ R1
    V0, V1 = L0 ^ R0, L1 ^ R1
    dV = V0 ^ V1
    back0, back1 = _ror(V0, 2), _ror(V1, 2)        # previous-round right word
    hw = lambda a: np.unpackbits(a.astype('>u2').view(np.uint8).reshape(-1, 2), axis=1).sum(1)
    return {
        'raw': to_bits([L0, R0, L1, R1]),                         # Gohr 2019
        'inter_xor': to_bits([dL, dR]),                           # Benamira, Bao
        'intra_xor': to_bits([V0, V1]),                           # Bao
        'total_xor': to_bits([dV]),                               # Bao
        'cross_xor': to_bits([L0 ^ R1, R0 ^ L1]),                 # Bao
        'benamira_input': to_bits([dL, dV, V0, V1]),              # Benamira Conj. 4
        'zheng_input': to_bits([dL, V0, L0 ^ R1]),                # Zheng 2024
        'calc_back': to_bits([back0, back1, back0 ^ back1]),      # Gohr-Leander-Neumann
        'modular_diff': to_bits([(L0 - L1) & MASK, (R0 - R1) & MASK]),
        'hamming': np.stack([hw(dL), hw(dR), hw(dV)], axis=1).astype(np.float32),
    }


# ── Gohr's pretrained one-block nets, forward pass in numpy ───────────────

def load_gohr_net(path):
    """Weights keyed by role, not by Keras's layer name: the 7- and 8-round
    files number their layers differently from the 5- and 6-round ones.
    Conv, batch-norm and dense layers are taken in the file's layer order."""
    import h5py
    f = h5py.File(path, 'r')
    g = f['model_weights'] if 'model_weights' in f else f
    order = [n.decode() if isinstance(n, bytes) else n for n in g.attrs['layer_names']]
    roles = {'conv1d': ['conv1d_3', 'conv1d_4', 'conv1d_5'],
             'batch_normalization': [f'batch_normalization_{i}' for i in range(1, 6)],
             'dense': ['dense_4', 'dense_5', 'dense_6']}
    seen = {k: 0 for k in roles}
    W = {}
    for layer in order:
        kind = re.sub(r'_\d+$', '', layer)
        if kind not in roles or not len(g[layer]):
            continue
        role = roles[kind][seen[kind]]
        seen[kind] += 1
        def grab(name, obj):
            if hasattr(obj, 'shape'):
                W[(role, name.split('/')[-1].split(':')[0])] = np.asarray(obj)
        g[layer].visititems(grab)
    return W


def _bn(x, W, name, eps=1e-3):
    return ((x - W[(name, 'moving_mean')]) / np.sqrt(W[(name, 'moving_variance')] + eps)
            * W[(name, 'gamma')] + W[(name, 'beta')])


def _conv_same(x, kernel, bias):
    # x (n, 16, cin); kernel (k, cin, cout); 'same' padding, stride 1
    k = kernel.shape[0]
    if k == 1:
        return x @ kernel[0] + bias
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (0, 0)))
    out = sum(xp[:, j:j + x.shape[1], :] @ kernel[j] for j in range(k))
    return out + bias


def gohr_forward(W, X, batch=100_000, hidden=False):
    relu = lambda a: np.maximum(a, 0)
    out = []
    for s in range(0, len(X), batch):
        x = X[s:s + batch].astype(np.float32).reshape(-1, 4, 16).transpose(0, 2, 1)
        c0 = relu(_bn(_conv_same(x, W[('conv1d_3', 'kernel')], W[('conv1d_3', 'bias')]), W, 'batch_normalization_1'))
        c1 = relu(_bn(_conv_same(c0, W[('conv1d_4', 'kernel')], W[('conv1d_4', 'bias')]), W, 'batch_normalization_2'))
        c2 = relu(_bn(_conv_same(c1, W[('conv1d_5', 'kernel')], W[('conv1d_5', 'bias')]), W, 'batch_normalization_3'))
        h = (c0 + c2).reshape(len(x), -1)
        d1 = relu(_bn(h @ W[('dense_4', 'kernel')] + W[('dense_4', 'bias')], W, 'batch_normalization_4'))
        d2 = relu(_bn(d1 @ W[('dense_5', 'kernel')] + W[('dense_5', 'bias')], W, 'batch_normalization_5'))
        if hidden:                     # the 64 values the output unit reads
            out.append(d2.astype(np.float32))
            continue
        z = d2 @ W[('dense_6', 'kernel')] + W[('dense_6', 'bias')]
        out.append(1 / (1 + np.exp(-z[:, 0])))
    return np.concatenate(out)


# ── stages ────────────────────────────────────────────────────────────────

def stage_check(args):
    res = {'testvector': check_testvector()}
    if args.deep_speck:
        sys.path.insert(0, args.deep_speck)
        import speck as sp                                   # Gohr's module
        rng = np.random.default_rng(0)
        n = 10_000
        keys = rng.integers(0, 1 << 16, size=(4, n), dtype=np.uint16)
        pl, pr = (rng.integers(0, 1 << 16, size=n, dtype=np.uint16) for _ in range(2))
        theirs = sp.encrypt((pl, pr), sp.expand_key(keys, 7))
        ours = encrypt(pl.astype(np.uint32), pr.astype(np.uint32),
                       expand_key([k.astype(np.uint32) for k in keys], 7))
        res['equal_to_deep_speck_7_rounds'] = bool(all(np.array_equal(a.astype(np.uint32), b)
                                                       for a, b in zip(theirs, ours)))
    return res


def stage_published(args):
    res = {}
    for r in args.rounds:
        t = time.time()
        ct, y = make_pairs(args.n_test, r, seed=1000 + r)
        W = load_gohr_net(Path(args.deep_speck) / f'net{r}_small.h5')
        z = gohr_forward(W, to_bits(ct))
        acc = float(np.mean((z > 0.5) == y))
        res[r] = {'accuracy': acc, 'n_test': args.n_test,
                  'published_table2': PUBLISHED_NET[r], 'published_ddt': PUBLISHED_DDT[r],
                  'one_block_bacuieti': PUBLISHED_ONE_BLOCK[r], 'seconds': round(time.time() - t, 1)}
        print(f'net{r}_small: accuracy {acc:.4f} on {args.n_test:,} pairs '
              f'(Table 2 {PUBLISHED_NET[r]}, one block re-trained {PUBLISHED_ONE_BLOCK[r]})', flush=True)
    return res


def stage_trees(args):
    import lightgbm as lgb
    res = {}
    for r in args.rounds:
        ct_tr, y_tr = make_pairs(args.n_train, r, seed=r)
        ct_te, y_te = make_pairs(args.n_test, r, seed=1000 + r)
        V_tr, V_te = views(ct_tr), views(ct_te)
        if args.deep_speck:            # Gohr's pretrained net for this round count
            W = load_gohr_net(Path(args.deep_speck) / f'net{r}_small.h5')
            V_tr['gohr_net_emb'] = gohr_forward(W, V_tr['raw'], hidden=True)
            V_te['gohr_net_emb'] = gohr_forward(W, V_te['raw'], hidden=True)
        names = args.views or list(V_tr)
        res[r] = {}
        for v in names:
            t = time.time()
            m = lgb.LGBMClassifier(n_estimators=args.n_trees, learning_rate=0.1, num_leaves=63,
                                   n_jobs=args.threads, verbose=-1)
            m.fit(V_tr[v], y_tr)
            acc = float(np.mean(m.predict(V_te[v]) == y_te))
            res[r][v] = {'accuracy': acc, 'features': int(V_tr[v].shape[1]),
                         'seconds': round(time.time() - t, 1)}
            print(f'{r} rounds  {v:15s} {V_tr[v].shape[1]:3d} features  '
                  f'lightgbm {acc:.4f}  ({time.time() - t:.0f} s)', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['check', 'published', 'trees'])
    ap.add_argument('--deep-speck', help='path to a clone of github.com/agohr/deep_speck')
    ap.add_argument('--rounds', type=int, nargs='+', default=[5, 6, 7, 8])
    ap.add_argument('--n-train', type=int, default=10**6)
    ap.add_argument('--n-test', type=int, default=10**6)
    ap.add_argument('--n-trees', type=int, default=500)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--views', nargs='+')
    ap.add_argument('--tag', help='suffix for the result file name')
    args = ap.parse_args()
    out = {'check': stage_check, 'published': stage_published, 'trees': stage_trees}[args.stage](args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    path = RESULTS_DIR / f'{args.stage}{tag}.json'
    path.write_text(json.dumps({'args': vars(args), 'result': out}, indent=1))
    print(json.dumps(out, indent=1) if args.stage == 'check' else f'wrote {path}')


if __name__ == '__main__':
    main()
