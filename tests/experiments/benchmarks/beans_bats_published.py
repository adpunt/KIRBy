#!/usr/bin/env python3
"""BEANS "bats" (Hagiwara et al. 2022, arXiv 2210.12300): which of 10 Egyptian
fruit bats made a call. The published classical protocol, ported from
github.com/earthspecies/beans (commit b278b1d9, MIT): scripts/bats.py (the
split), beans/datasets.py (the MFCC features), scripts/evaluate.py and
run_benchmark.py (models, grids, selection on the validation calls).

Object: one call, a float32 WAV at 250 kHz, cut from Prat et al. 2017
(Scientific Data 4:170143). The BEANS file `N.wav` is row N (0-based) of
Prat's Annotations.csv: same emitter, same length to one sample (checked on
46690.wav, 2026-11-09). 10,000 calls, 1,000 per bat.

Split (scripts/bats.py): train_test_split(test_size=0.4, random_state=42,
stratify=label), then the 40% halved the same way: 6,000 train, 2,000 valid,
2,000 test. `beans_split` rebuilds it and checks the three CSVs against the md5
BEANS ships in data/file_hashes; it refuses to go on if one differs.

Features (beans/datasets.py `_get_spectrogram(return_mfcc=True)` and
evaluate.py `spec2feats`): torchaudio MFCC, 20 coefficients, 128 mel bands,
n_fft = 0.05 s x 250,000 = 12,500 samples, hop 0.01 s = 2,500 samples, power
2; frames past 5 s cut, shorter calls padded with zero frames to 500; mean,
std, min and max of each coefficient over the 500 frames: 80 features.
The zero frames enter the four statistics, as in the published code.

Models (run_benchmark.py MODELS, evaluate.py train_sklearn_model):
StandardScaler fitted on the training calls; one fit per grid value on the
training calls; the value with the best validation accuracy (first one on a
tie) is scored once on the test calls.
    lr            LogisticRegression(max_iter=1000), C in 0.1, 1, 10
    svm           SVC(), C in 0.1, 1, 10
    decisiontree  DecisionTreeClassifier(), max_depth in None, 5, 10, 20, 30
    gbdt          GradientBoostingClassifier(), n_estimators in 10, 50, 100, 200
    xgboost       XGBClassifier(n_jobs=4), n_estimators in 10, 50, 100, 200
The published code sets no seed on these. --seed sets random_state on the
three that take one; left out, they run unseeded, as published.

Published test accuracy (BEANS Table 2, bat column, read 2026-11-09):
lr 0.661, svm 0.720, decisiontree 0.474, gbdt 0.674, xgboost 0.692.

Audio is read from, in order: <KIRBY_BEANS_DIR>/audio/, the zip
<KIRBY_BEANS_DIR>/egyptian_fruit_bats.zip, or, with --stream, the zip on
Google Cloud Storage by HTTP range requests (nothing written but features).

    python tests/experiments/benchmarks/beans_bats_published.py --stage fetch
    python tests/experiments/benchmarks/beans_bats_published.py --stage features
    python tests/experiments/benchmarks/beans_bats_published.py --stage evaluate

Writes <KIRBY_BENCH_RESULTS>/beans_bats_published.json.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ZIP_URL = 'https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip'
ZIP_BYTES = 5246087171
# data/file_hashes in earthspecies/beans at b278b1d9
SPLIT_MD5 = {'train': 'f123a13ff5bb335562b84debc710ba1a',
             'valid': '4a8c734357782defa4d4986e5f2c12e8',
             'test': '937b559089ae6ef2c219e41f9943392b'}
LABELS = [111, 210, 211, 215, 216, 220, 226, 228, 230, 231]   # datasets.yml
SAMPLE_RATE = 250_000
MAX_DURATION = 5
FFT_SIZE_IN_SECS = 0.05
HOP_LENGTH_IN_SECS = 0.01

PUBLISHED = {'lr': 0.661, 'svm': 0.720, 'decisiontree': 0.474, 'gbdt': 0.674, 'xgboost': 0.692}
GRIDS = {'lr': ('C', [0.1, 1.0, 10.0]),
         'svm': ('C', [0.1, 1.0, 10.0]),
         'decisiontree': ('max_depth', [None, 5, 10, 20, 30]),
         'gbdt': ('n_estimators', [10, 50, 100, 200]),
         'xgboost': ('n_estimators', [10, 50, 100, 200])}

DATA_DIR = Path(os.environ.get('KIRBY_BEANS_DIR', '/Volumes/seagate/benchmarks_protocol/beans_bats'))
RESULTS = Path(os.environ.get('KIRBY_BENCH_RESULTS', '/Volumes/seagate/benchmarks_protocol/results'))


def log(msg):
    print('%s %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


# ── audio ─────────────────────────────────────────────────────────────────

class _RangeFile(io.RawIOBase):
    """A remote file read by HTTP range requests, so zipfile can seek in it."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={'Range': 'bytes=%d-%d' % (self.pos, end)})
                d = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


class AudioSource:
    """Returns one call as a float32 mono array at 250 kHz."""

    def __init__(self, stream=False):
        self.dir = DATA_DIR / 'audio'
        self.zip = None
        if self.dir.is_dir() and any(self.dir.iterdir()):
            return
        zp = DATA_DIR / 'egyptian_fruit_bats.zip'
        if zp.exists():
            self.zip = zipfile.ZipFile(zp)
        elif stream:
            self.zip = zipfile.ZipFile(io.BufferedReader(_RangeFile(ZIP_URL, ZIP_BYTES), buffer_size=1 << 22))
        else:
            raise SystemExit('no audio under %s (audio/ or egyptian_fruit_bats.zip); run --stage fetch, '
                             'or pass --stream' % DATA_DIR)

    def annotations_csv(self):
        if self.zip is None:
            return (DATA_DIR / 'annotations.csv').read_text()
        return self.zip.read('annotations.csv').decode()

    def read(self, name):
        from scipy.io import wavfile
        src = io.BytesIO(self.zip.read('audio/' + name)) if self.zip is not None else str(self.dir / name)
        sr, w = wavfile.read(src)
        if sr != SAMPLE_RATE:
            raise ValueError('%s: %d Hz, expected %d' % (name, sr, SAMPLE_RATE))
        w = np.asarray(w, np.float32)
        if w.ndim == 2:                       # torch.mean(waveform, dim=0) in the published loader
            w = w.mean(axis=1)
        return w


# ── split ─────────────────────────────────────────────────────────────────

def beans_split(annotations_text):
    """scripts/bats.py, line for line. Returns (table, split per row) where
    the table has the BEANS columns path and label in the annotations' row
    order. Raises if a CSV it writes differs from the md5 BEANS ships."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(io.StringIO(annotations_text))
    df = df.apply(lambda r: pd.Series({'path': f"data/egyptian_fruit_bats/audio/{r['File Name']}",
                                       'label': r['Emitter']}), axis=1)
    df_train, df_valid_test = train_test_split(df, test_size=0.4, random_state=42, shuffle=True,
                                               stratify=df['label'])
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42, shuffle=True,
                                         stratify=df_valid_test['label'])
    split = np.empty(len(df), dtype=object)
    for name, d in (('train', df_train), ('valid', df_valid), ('test', df_test)):
        got = hashlib.md5(d.sort_index().to_csv().encode()).hexdigest()
        if got != SPLIT_MD5[name]:
            raise SystemExit('BEANS %s split differs from the published one: md5 %s, expected %s'
                             % (name, got, SPLIT_MD5[name]))
        split[d.index.values] = name
    return df, split


# ── features ──────────────────────────────────────────────────────────────

_MFCC = None


def mfcc_stats(w):
    """_get_spectrogram(return_mfcc=True) then spec2feats: 80 features."""
    global _MFCC
    import torch
    import torch.nn.functional as F
    import torchaudio
    n_fft = int(FFT_SIZE_IN_SECS * SAMPLE_RATE)
    hop = int(HOP_LENGTH_IN_SECS * SAMPLE_RATE)
    if _MFCC is None:
        _MFCC = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=20,
                                           melkwargs={'n_mels': 128, 'n_fft': n_fft,
                                                      'hop_length': hop, 'power': 2.0})
    x = torch.from_numpy(w).unsqueeze(0)
    if x.shape[1] < n_fft:
        x = F.pad(x, (0, n_fft - x.shape[1]))
    spec = _MFCC(x)
    max_frames = int(MAX_DURATION * int(1 / HOP_LENGTH_IN_SECS))
    spec = spec[0, :, :max_frames]
    if spec.shape[1] < max_frames:
        spec = F.pad(spec, (0, max_frames - spec.shape[1]))
    feats = torch.cat([spec.mean(dim=1), spec.std(dim=1), spec.min(dim=1)[0], spec.max(dim=1)[0]])
    return feats.numpy().reshape(-1)


def calls_table(src):
    """The 10,000 calls with label and BEANS split, written once to
    <KIRBY_BEANS_DIR>/calls.csv (row order = annotations.csv)."""
    import pandas as pd
    p = DATA_DIR / 'calls.csv'
    if p.exists():
        return pd.read_csv(p)
    text = src.annotations_csv()
    df, split = beans_split(text)
    raw = pd.read_csv(io.StringIO(text))
    out = pd.DataFrame({'file': raw['File Name'], 'label': df['label'].astype(int), 'split': split})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out


def stage_features(stream):
    out = DATA_DIR / 'views' / 'mfcc_stats.npy'
    if out.exists():
        log('mfcc_stats already at %s' % out)
        return
    src = AudioSource(stream=stream)
    calls = calls_table(src)
    part = out.with_name('mfcc_stats.partial.npy')
    X = np.load(part) if part.exists() else np.full((len(calls), 80), np.nan, np.float32)
    todo = np.where(np.isnan(X[:, 0]))[0]
    log('mfcc_stats: %d of %d calls to do' % (len(todo), len(calls)))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, i in enumerate(todo):
        X[i] = mfcc_stats(src.read(calls.file[i]))
        if (n + 1) % 250 == 0:
            np.save(part, X)
            log('  %d / %d calls, %.0f s' % (n + 1, len(todo), time.time() - t0))
    np.save(part, X)
    os.replace(part, out)
    log('mfcc_stats written: %s' % out)


# ── models ────────────────────────────────────────────────────────────────

def make_model(name, value, seed=None):
    key, _ = GRIDS[name]
    kw = {key: value}
    if name == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1_000, **kw)
    if name == 'svm':
        from sklearn.svm import SVC
        return SVC(**kw)
    if name == 'decisiontree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, **kw)
    if name == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=seed, **kw)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_jobs=4, random_state=seed if seed is not None else 0, **kw)
    raise ValueError(name)


def published_protocol(name, X_tr, y_tr, X_va, y_va, X_te, seed=None):
    """train_sklearn_model + eval on test. y are class ids 0..9. Returns
    (test predictions, chosen grid value, validation accuracy per value)."""
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_tr)
    Xtr, Xva, Xte = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    best, best_acc, best_v, valid = None, 0.0, None, []
    for v in GRIDS[name][1]:
        m = make_model(name, v, seed).fit(Xtr, y_tr)
        acc = float(np.mean(m.predict(Xva) == y_va))
        valid.append({'value': v, 'valid_accuracy': acc})
        if acc > best_acc:
            best, best_acc, best_v = m, acc, v
    return best.predict(Xte), best_v, valid


def stage_evaluate(models, seed):
    import pandas as pd
    calls = pd.read_csv(DATA_DIR / 'calls.csv')
    X = np.load(DATA_DIR / 'views' / 'mfcc_stats.npy').astype(np.float64)
    y = calls.label.map({l: i for i, l in enumerate(LABELS)}).values
    idx = {s: np.where(calls.split.values == s)[0] for s in ('train', 'valid', 'test')}
    out = RESULTS / 'beans_bats_published.json'
    res = json.loads(out.read_text()) if out.exists() else {'rows': {}}
    for name in models:
        t0 = time.time()
        pred, v, valid = published_protocol(name, X[idx['train']], y[idx['train']], X[idx['valid']],
                                            y[idx['valid']], X[idx['test']], seed)
        acc = float(np.mean(pred == y[idx['test']]))
        res['rows'][name] = {'test_accuracy': acc, 'published': PUBLISHED[name],
                             'test_wrong': int(np.sum(pred != y[idx['test']])),
                             'chosen': {GRIDS[name][0]: v}, 'valid': valid,
                             'seconds': time.time() - t0, 'seed': seed}
        log('%-12s test accuracy %.4f, published %.3f (%s=%s, %.0f s)'
            % (name, acc, PUBLISHED[name], GRIDS[name][0], v, time.time() - t0))
        RESULTS.mkdir(parents=True, exist_ok=True)
        res.update({'n_train': len(idx['train']), 'n_valid': len(idx['valid']), 'n_test': len(idx['test']),
                    'features': 'mfcc_stats, 80', 'result_status': 'ok'})
        out.write_text(json.dumps(res, indent=1))


def stage_fetch():
    """Download the BEANS zip into KIRBY_BEANS_DIR and unzip it there."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zp = DATA_DIR / 'egyptian_fruit_bats.zip'
    if not zp.exists() or zp.stat().st_size != ZIP_BYTES:
        log('downloading %s (%.1f GB) to %s' % (ZIP_URL, ZIP_BYTES / 1e9, zp))
        tmp = zp.with_suffix('.part')
        urllib.request.urlretrieve(ZIP_URL, tmp)
        os.replace(tmp, zp)
    if not (DATA_DIR / 'audio').is_dir():
        log('unzipping (18.8 GB of WAV)')
        zipfile.ZipFile(zp).extractall(DATA_DIR)
    log('audio at %s' % (DATA_DIR / 'audio'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['fetch', 'features', 'evaluate'], required=True)
    ap.add_argument('--models', default=','.join(GRIDS))
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--stream', action='store_true', help='read the zip remotely; store only features')
    a = ap.parse_args()
    if a.stage == 'fetch':
        stage_fetch()
    elif a.stage == 'features':
        stage_features(a.stream)
    else:
        stage_evaluate(a.models.split(','), a.seed)


if __name__ == '__main__':
    main()
