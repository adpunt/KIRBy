#!/usr/bin/env python3
"""Kepler DR24 TCEs as published by AstroNet: planet vs not-planet from the
global and local views of a phase-folded light curve.

Source of the protocol: Shallue & Vanderburg, "Identifying Exoplanets with
Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth
Planet around Kepler-90", AJ 155:94, 2018, sections 3-5, Tables 1 and 2.
Code: google-research/exoplanet-ml (Apache-2.0, archived 2023), directory
exoplanet-ml/astronet. Published test-set numbers (Tables 1 and 2):

    model            inputs           accuracy   AUC
    linear           global           0.869      0.922
    linear           local            0.879      0.933
    linear           global + local   0.917      0.963
    fully connected  global + local   0.941      0.977
    convolutional    global + local   0.960      0.988

What this file reproduces: the three LINEAR rows. The paper defines the
linear model as the network "with zero hidden layers (which is equivalent to
a linear logistic regression model)", trained to maximise validation AUC
with hyperparameters chosen by Google Vizier (values not published). Here it
is scikit-learn LogisticRegression on the shipped views, with the inverse
regularisation strength C chosen on the validation set from a small grid
(see GRID: 1e-5 to 100, 15 values). Everything else is as shipped.

What it does NOT reproduce: the fully connected and convolutional rows. Those
are TensorFlow 1 estimators trained for 50 epochs with Adam and averaged
over 10 random initialisations (paper section 4.2). Run them with the
published code on the server; see NEW_DATASETS.md section 4.

Data: the pre-computed training set the AstroNet README links (10 TFRecord
files, Google Drive folder 1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE), downloaded
2026-10-29 to /Volumes/seagate/benchmarks_protocol/kepler_astronet/tfrecord/
(override the parent directory with KIRBY_KEPLER_DIR). The files are read
with the small TFRecord reader below, so TensorFlow is not needed. Each
record holds global_view (2,001 bins), local_view (201 bins), the label
av_training_set (PC / AFP / NTP) and seven TCE catalogue values.

Split: fixed by the published code (astronet/data/generate_input_records.py):
the labelled rows of the DR24 TCE table in archive order, shuffled with
numpy seed 123, first 80% train, next 10% validation, last 10% test. With
--tce-csv pointing at the DR24 table downloaded from the archive's TAP
service, the split is replayed and checked against the shipped files.

With --exonet, the Exonet input bundle (Ansdell et al. 2018, ApJL 869:L7;
GitLab frontierdevelopmentlab/exoplanets/exonet-pytorch; Google Drive file
1N6bA2rahvV5kcOGmJnTA5gl_Thno7hTh, unzipped and untarred under
<KIRBY_KEPLER_DIR>/exonet/unzipped/exonet_inputs/) is loaded, its TCE ids
are matched to the AstroNet split, and the same logistic regression is run
on all five Exonet inputs concatenated, on the TCEs both bundles have.
Measured 2026-10-29: the bundle holds 15,084 of AstroNet's 15,737 TCEs
(train 11,937 of 12,589; val 1,574 of 1,574; test 1,573 of 1,574), every one
in the same split as AstroNet, with the same label. Its info file carries 10
standardised stellar values per TCE, not the 6 the paper names; the bundle
does not say which column is which. Exonet publishes no linear number, so
that row has nothing to be compared with; it only shows the loader works.
Exonet's own model (accuracy 0.975 ensembled) is a PyTorch CNN with .cuda()
hard-coded: server only.

Output: /Volumes/seagate/benchmarks_protocol/results/b05_kepler_astronet.json

    python tests/experiments/benchmarks/b05_kepler_astronet_published.py \
        --tce-csv /Volumes/seagate/benchmarks_protocol/kepler_astronet/dr24_tce.csv \
        --exonet
"""
import argparse
import csv
import glob
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

KEPLER_DIR = Path(os.environ.get(
    'KIRBY_KEPLER_DIR', '/Volumes/seagate/benchmarks_protocol/kepler_astronet'))
OUT_DIR = Path('/Volumes/seagate/benchmarks_protocol/results')

PUBLISHED = {  # (accuracy, AUC) on the test set, Tables 1 and 2
    'linear global': (0.869, 0.922),
    'linear local': (0.879, 0.933),
    'linear global+local': (0.917, 0.963),
    'fully connected global+local': (0.941, 0.977),
    'convolutional global+local': (0.960, 0.988),
}
PUBLISHED_TEST_N = 1523          # paper section 3.1; the code's split gives 1,574
LABEL_MAP = {'PC': 1, 'AFP': 0, 'NTP': 0}   # astro_model/configurations.py
GRID = [10.0 ** k for k in range(-5, 3)] + [3 * 10.0 ** k for k in range(-5, 2)]  # C, by validation AUC
SPLIT_SEED = 123                           # generate_input_records.py
SPLIT_FRACTIONS = (0.80, 0.90)             # train < 0.80 <= val < 0.90 <= test
TCE_SCALARS = ['tce_period', 'tce_duration', 'tce_depth', 'tce_impact',
               'tce_max_mult_ev', 'tce_model_snr', 'tce_prad']
EXONET_STELLAR_N = 10   # columns 6..15 of each *_info.npy; names not in the bundle


# ----------------------------------------------------------------------------
# TFRecord + tf.train.Example reader. TFRecord framing is
# uint64 length, uint32 crc, bytes, uint32 crc; the bytes are a protobuf
# Example{Features{map<string, Feature{oneof bytes_list|float_list|int64_list}}}.
# ----------------------------------------------------------------------------
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7f) << s
        s += 7
        if not c & 0x80:
            return r, i


def _fields(b):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError(f'unknown protobuf wire type {w}')
        yield f, w, v


def parse_example(b):
    out = {}
    for _, _, feats in _fields(b):
        for _, _, entry in _fields(feats):
            name = val = None
            for f3, _, v in _fields(entry):
                if f3 == 1:
                    name = v.decode()
                elif f3 == 2:
                    val = v
            for f4, _, lst in _fields(val):
                items = []
                for _, w5, x in _fields(lst):
                    if f4 == 1:
                        items.append(x)
                    elif f4 == 2:
                        items.extend(np.frombuffer(x, '<f4') if w5 == 2
                                     else [struct.unpack('<f', x)[0]])
                    elif f4 == 3:
                        if w5 == 2:
                            j = 0
                            while j < len(x):
                                n, j = _varint(x, j)
                                items.append(n)
                        else:
                            items.append(x)
                out[name] = items if f4 == 1 else np.asarray(items)
    return out


def read_tfrecord(path):
    with open(path, 'rb') as fh:
        while True:
            hdr = fh.read(8)
            if not hdr:
                return
            n, = struct.unpack('<Q', hdr)
            fh.read(4)
            data = fh.read(n)
            fh.read(4)
            yield parse_example(data)


def load_astronet():
    """{split: dict of arrays} from the 10 shipped TFRecord files."""
    out = {}
    for split, pattern in [('train', 'train-0000?-of-00008'),
                           ('val', 'val-00000-of-00001'),
                           ('test', 'test-00000-of-00001')]:
        files = sorted(glob.glob(str(KEPLER_DIR / 'tfrecord' / pattern)))
        assert files, f'no {pattern} under {KEPLER_DIR / "tfrecord"}'
        rows = [ex for f in files for ex in read_tfrecord(f)]
        d = {
            'kepid': np.array([int(r['kepid'][0]) for r in rows]),
            'tce_plnt_num': np.array([int(r['tce_plnt_num'][0]) for r in rows]),
            'label_text': np.array([r['av_training_set'][0].decode() for r in rows]),
            'global_view': np.stack([r['global_view'] for r in rows]).astype(np.float32),
            'local_view': np.stack([r['local_view'] for r in rows]).astype(np.float32),
            'tce_scalars': np.stack([[float(r[k][0]) for k in TCE_SCALARS]
                                     for r in rows]).astype(np.float32),
        }
        d['y'] = np.array([LABEL_MAP[t] for t in d['label_text']])
        assert d['global_view'].shape[1] == 2001 and d['local_view'].shape[1] == 201
        out[split] = d
    assert len(out['train']['y']) == 12589 and len(out['val']['y']) == 1574 \
        and len(out['test']['y']) == 1574, {k: len(v['y']) for k, v in out.items()}
    return out


def check_split(data, tce_csv):
    """Replay generate_input_records.py on the DR24 table and compare ids."""
    rows = [r for r in csv.DictReader(open(tce_csv))
            if r['av_training_set'] in LABEL_MAP]
    n = len(rows)
    np.random.seed(SPLIT_SEED)
    perm = np.random.permutation(n)
    cut = (int(SPLIT_FRACTIONS[0] * n), int(SPLIT_FRACTIONS[1] * n))
    replay = {'train': perm[:cut[0]], 'val': perm[cut[0]:cut[1]], 'test': perm[cut[1]:]}
    report = {'tce_csv_labelled_rows': n}
    for split, idx in replay.items():
        ids = set((int(rows[i]['kepid']), int(rows[i]['tce_plnt_num'])) for i in idx)
        shipped = set(zip(data[split]['kepid'].tolist(), data[split]['tce_plnt_num'].tolist()))
        report[f'{split}_replayed'] = len(ids)
        report[f'{split}_shipped'] = len(shipped)
        report[f'{split}_in_both'] = len(ids & shipped)
    return report


def load_exonet(data):
    """Exonet's five inputs per TCE, in AstroNet split order, for the TCEs
    both bundles hold. Returns {split: {...arrays..., 'y', 'n_dropped'}}."""
    base = KEPLER_DIR / 'exonet' / 'unzipped' / 'exonet_inputs'
    out = {}
    for split in ('train', 'val', 'test'):
        def files(suffix):
            fs = [f for f in glob.glob(str(base / f'{split}_*' / '**' / f'*{suffix}.npy'),
                                       recursive=True) if '/._' not in f]
            return {os.path.basename(f).replace(f'_{suffix}.npy', ''): f for f in fs}
        info_f = files('info')
        parts = {k: files(k) for k in ('global', 'local', 'global_cen_w', 'local_cen_w')}
        assert info_f, f'no *_info.npy under {base}/{split}_*'
        recs = {}
        for stem, f in info_f.items():
            info = np.load(f, allow_pickle=False)
            assert info.shape == (6 + EXONET_STELLAR_N,), (f, info.shape)
            recs[(int(info[0]), int(info[1]))] = (stem, info)
        want = list(zip(data[split]['kepid'].tolist(), data[split]['tce_plnt_num'].tolist()))
        keep = [i for i, w in enumerate(want) if w in recs]
        extra = set(recs) - set(want)
        assert not extra, f'{split}: {len(extra)} Exonet TCEs are not in this AstroNet split'
        d = {k: [] for k in parts}
        d['stellar'], d['label'] = [], []
        for i in keep:
            stem, info = recs[want[i]]
            for k in parts:
                d[k].append(np.load(parts[k][stem], allow_pickle=False))
            d['stellar'].append(info[6:])
            d['label'].append(info[5])
        d = {k: np.stack(v).astype(np.float32) for k, v in d.items()}
        y = data[split]['y'][keep]
        assert np.array_equal(d['label'].astype(int), y), f'{split}: Exonet labels differ'
        out[split] = {'n_exonet_files': len(info_f), 'n_dropped': len(want) - len(keep),
                      'y': y, **d}
    return out


def fit_linear(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Logistic regression; C chosen on validation AUC, scored once on test."""
    best = None
    for C in GRID:
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    auc_va, C, m = best
    p = m.predict_proba(X_te)[:, 1]
    return {'C': C, 'val_auc': float(auc_va),
            'test_accuracy': float(accuracy_score(y_te, p >= 0.5)),
            'test_auc': float(roc_auc_score(y_te, p)),
            'test_n': int(len(y_te)), 'n_features': int(X_tr.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tce-csv', default=None,
                    help='DR24 TCE table (TAP download) to replay the split against')
    ap.add_argument('--exonet', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    data = load_astronet()
    print(f"loaded {len(data['train']['y'])} train / {len(data['val']['y'])} val / "
          f"{len(data['test']['y'])} test TCEs in {time.time() - t0:.0f} s; "
          f"planets in test: {int(data['test']['y'].sum())}", flush=True)
    out = {'benchmark': 'kepler_astronet', 'published': PUBLISHED,
           'published_test_n': PUBLISHED_TEST_N, 'grid_C': GRID,
           'n': {k: int(len(v['y'])) for k, v in data.items()},
           'labels': {k: {t: int((v['label_text'] == t).sum()) for t in LABEL_MAP}
                      for k, v in data.items()}}

    if args.tce_csv:
        out['split_check'] = check_split(data, args.tce_csv)
        print('split replay:', out['split_check'], flush=True)

    views = {'global': ['global_view'], 'local': ['local_view'],
             'global+local': ['global_view', 'local_view']}
    out['linear'] = {}
    for name, keys in views.items():
        X = {s: np.hstack([data[s][k] for k in keys]) for s in data}
        t1 = time.time()
        r = fit_linear(X['train'], data['train']['y'], X['val'], data['val']['y'],
                       X['test'], data['test']['y'])
        r['seconds'] = round(time.time() - t1)
        out['linear'][name] = r
        pa, pu = PUBLISHED[f'linear {name}']
        print(f"linear {name}: C={r['C']}, test accuracy {r['test_accuracy']:.4f} "
              f"(published {pa}), test AUC {r['test_auc']:.4f} (published {pu}); "
              f"{r['seconds']} s", flush=True)

    if args.exonet:
        t1 = time.time()
        ex = load_exonet(data)
        print(f"exonet: loaded and aligned in {time.time() - t1:.0f} s; "
              f"TCEs per split {[len(ex[s]['y']) for s in ('train', 'val', 'test')]}, "
              f"AstroNet TCEs absent from the bundle {[ex[s]['n_dropped'] for s in ('train', 'val', 'test')]}; "
              f"widths {[(k, ex['test'][k].shape[1]) for k in ('global', 'local', 'global_cen_w', 'local_cen_w', 'stellar')]}",
              flush=True)
        keys = ['global', 'local', 'global_cen_w', 'local_cen_w', 'stellar']
        X = {s: np.hstack([ex[s][k] for k in keys]) for s in ex}
        t1 = time.time()
        r = fit_linear(X['train'], ex['train']['y'], X['val'], ex['val']['y'],
                       X['test'], ex['test']['y'])
        r['seconds'] = round(time.time() - t1)
        out['linear']['exonet all five inputs (no published number)'] = r
        out['exonet'] = {'widths': {k: int(ex['test'][k].shape[1]) for k in keys},
                         'n_files': {s: ex[s]['n_exonet_files'] for s in ex},
                         'n_dropped': {s: ex[s]['n_dropped'] for s in ex}}
        print(f"linear exonet five inputs: C={r['C']}, test accuracy {r['test_accuracy']:.4f}, "
              f"test AUC {r['test_auc']:.4f}; {r['seconds']} s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'b05_kepler_astronet.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f"written {OUT_DIR / 'b05_kepler_astronet.json'}; total {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
