#!/usr/bin/env python3
"""DroneRF view and model screen: every candidate representation of a
100,000-sample chunk, under the six models the DroneRF papers used, with
per-fold accuracy and macro-F1, out-of-fold predictions, and correlations
between views.

Object: one chunk of L = 100,000 samples per band (2.5 ms at 40 MS/s), the
object of b06_dronerf_published.py. 22,700 chunks, 227 recordings.
Labels and splits: as in b06 (published = StratifiedKFold seed 1 over chunks;
grouped = StratifiedGroupKFold seed 1 over the 227 recordings).

Views (one fixed-length vector per chunk). "Code" means ported from the
named repository; "paper" means written from the paper's text and so not a
reproduction of anybody's code.

    name                 size   source
    spectrum             2,047  Al-Sa'd 2019, code (b06: Main_1 + Main_2 scaling,
                                first 2,047 of 2,048 bins as Classification.py)
    spectrum_low         1,024  the low-band half of `spectrum` (Medaiyese 2020
                                reports the low band alone; no code)
    spectrum_high        1,023  the high-band half of `spectrum`
    welch_psd            2,052  Almasri 2021, paper: per band Welch PSD, Hamming
                                window, 2,050-sample segments, 50% overlap,
                                one-sided (1,026 bins), both bands concatenated,
                                min-max per chunk. The paper says its input has
                                2,050 values; the two-bin difference is this
                                file's reading of "2050", not the paper's code.
                                No low-power chunk is removed (owner, 2026-10-30).
    spectral_scalars        16  Shulman 2026, code (spectrahawk/features.py
                                `feature_matrix_windows`, Welch nperseg 1024):
                                total power, centroid, flatness, 99 % bandwidth
                                and four quarter-band powers, per band. The
                                paper describes four; the code computes eight.
    stats                   28  no DroneRF paper; the UCI HAR measure list
                                (NEW_DATASETS.md section 1) per band: mean, std,
                                median absolute deviation, max, min, energy,
                                interquartile range, entropy of a 256-bin
                                histogram, four Yule-Walker AR coefficients,
                                skewness, kurtosis.
    psd_resnet50         2,048  Swinney and Woods 2021, paper: the Welch PSD
                                (1,024-point FFT, 120-sample overlap, Hann)
                                drawn as a line plot, 224x224, ImageNet
                                ResNet50, global-average-pooled features.
                                The paper's 100,352-value flattened map is not
                                used (owner's choice pending). Plot style is
                                this file's, the paper does not give one.
    spectrogram_resnet50 2,048  same paper: STFT spectrogram (1,024-point FFT)
                                as an image, both bands stacked.
    histogram_resnet50   2,048  same paper: 500-bin amplitude histogram as an
                                image.
    gaf_resnet50         2,048  Fu and He 2024, paper: STFT 1,024-point Hann,
                                128-sample overlap, time-averaged to a 513-bin
                                spectrum, Gramian angular field 513x513, the
                                two bands' fields averaged, resized to 224.
                                They train a 2-D CNN on it; here it goes
                                through the frozen ResNet50 like the others.
    rfuavnet_emb           320  Huynh-The 2022, code (RF_UAVNet.m,
                                main_trainingDL.m): the 1-D CNN trained on
                                10,000-sample frames (10 per chunk, global
                                per-band min-max to [0, 1]); the 320-value
                                concatenation of its five global-average-pool
                                outputs, averaged over a chunk's 10 frames.
                                Trained inside each fold on the training
                                chunks only, so this view exists per split,
                                task and fold. The code trains 5 epochs, SGD
                                momentum 0.9, lr 1e-3 dropped x0.1 after epoch
                                4, batch 128; the paper says 90 epochs. The
                                code is ported.
    alemadi_emb            256  Al-Emadi 2020, code (Classification_CNN_
                                DroneModeIdentification.py.py): the 1-D CNN
                                on the 2,047-bin spectrum; the 256-unit dense
                                layer before the softmax. Fold-trained like
                                rfuavnet_emb. 350 epochs, batch 50, Adam 1e-3,
                                categorical cross-entropy.

Models (every vector model any DroneRF paper used), applied to every view:

    dense     Al-Sa'd 2019: 3 x 128 ReLU, sigmoid, MSE, Adam, 200 epochs,
              batch 10 (b06 `fit_predict`).
    xgboost   Medaiyese 2020: settings not stated; xgboost defaults. Fitted in a
              torch-free subprocess (_xgb_worker.py).
    svm       Kilic 2022: Gaussian kernel, C and gamma over 2^-15..2^15; the
              step is not stated, 2^2 is used here (16 x 16 grid), chosen by
              3-fold cross-validation inside the training fold. Standardised.
    logreg    Swinney and Woods 2021: L2, LBFGS. Standardised.
    rf        Shulman 2026 code: 300 trees, class_weight balanced.
    knn       Nemer 2021: k not stated; k = 5.

--debug shrinks epochs, folds and the SVM grid for a local run. Its output
is marked debug and is not a result (owner's rule). Defaults are the
protocols above.

Stages (run in order; each is resumable, finished units are skipped):
    extract      one pass over the 454 CSVs: static views, int16 frames
    images       ResNet50 features of the four image views
    evaluate     static views x models x tasks x splits
    embed        fold-trained CNN views, then evaluated the same way
    summarise    tables: accuracy / macro-F1 per view and model, CKA between
                 views, error overlap between views per model

    python tests/experiments/benchmarks/b06_dronerf_views.py --stage extract
    python tests/experiments/benchmarks/b06_dronerf_views.py --stage all --debug

Output: /Volumes/seagate/benchmarks_protocol/dronerf/views/ (cached views) and
/Volumes/seagate/benchmarks_protocol/results/b06_views/ (one JSON per
view x model x task x split, plus summary.md and summary.json).
"""
import argparse
import socket
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy import stats as sst
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('b06', HERE / 'b06_dronerf_published.py')
b06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b06)

DRONERF_DIR = b06.DRONERF_DIR
VIEWS_DIR = DRONERF_DIR / 'views'
RESULTS_DIR = Path(os.environ.get('KIRBY_BENCH_RESULTS',
                               '/Volumes/seagate/benchmarks_protocol/results')) / 'b06_views'
FS = 40e6          # samples per second per band (Almasri 2021: "Fs = 40MHz")
L = b06.L          # 100,000 samples per chunk
FRAME = 10_000     # Huynh-The 2022 frame length
FRAMES_PER_CHUNK = L // FRAME
N_CHUNKS_EXPECTED = 22_700
SEED = b06.SPLIT_SEED

STATIC_VIEWS = ['spectrum', 'spectrum_low', 'spectrum_high', 'welch_psd',
                'spectral_scalars', 'stats']
IMAGE_VIEWS = ['psd_resnet50', 'spectrogram_resnet50', 'histogram_resnet50', 'gaf_resnet50']
EMBED_VIEWS = ['rfuavnet_emb', 'alemadi_emb']
MODELS = ['knn', 'rf', 'logreg', 'xgboost', 'dense', 'svm']
TASKS = [2, 4, 10]
SPLITS = ['published', 'grouped']


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'b06_views.log', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}\n')


# ----------------------------------------------------------------------------
# Stage: extract. One pass over the raw files.
# ----------------------------------------------------------------------------

def read_csv(path):
    """One real-valued row of 10 million integers, read as b06 does (loadtxt;
    3.6 s per file against 17 s for np.fromfile and 263 s for pandas)."""
    return np.loadtxt(path, delimiter=',', dtype=np.float64).ravel()


def welch_psd_view(X):
    """Almasri 2021, per band: Hamming, 2,050-sample segments, 50 % overlap,
    one-sided. X: (n, L) one band. Returns (n, 1026)."""
    _, p = sps.welch(X, fs=FS, window='hamming', nperseg=2050, noverlap=1025, axis=-1)
    return p


def spectrahawk_feature_matrix_windows(X, fs, n_fft=1024):
    """Verbatim port of spectrahawk/features.py `feature_matrix_windows`
    (github.com/shulm/spectrahawk, MIT)."""
    X = np.asarray(X)
    f, Pxx = sps.welch(X, fs=fs, nperseg=n_fft, return_onesided=False, axis=-1)
    sort_idx = np.argsort(f)
    f = f[sort_idx]
    Pxx = Pxx[:, sort_idx]
    total_power = np.sum(Pxx, axis=1) + 1e-12
    centroid = np.sum(Pxx * f[None, :], axis=1) / total_power
    gmean = np.exp(np.mean(np.log(Pxx + 1e-12), axis=1))
    amean = np.mean(Pxx, axis=1)
    flatness = gmean / (amean + 1e-12)
    cum_power = np.cumsum(Pxx, axis=1) / total_power[:, None]
    idx_low = np.argmax(cum_power >= 0.005, axis=1)
    has_high = cum_power[:, -1] >= 0.995
    idx_high = np.argmax(cum_power >= 0.995, axis=1)
    bw_99 = np.full(len(Pxx), fs, dtype=float)
    bw_99[has_high] = f[idx_high[has_high]] - f[idx_low[has_high]]
    band_powers = [np.sum(q, axis=1) / total_power for q in np.array_split(Pxx, 4, axis=1)]
    return np.column_stack([total_power, centroid, flatness, bw_99, *band_powers])


def ar_coefficients(X, order=4):
    """Yule-Walker AR coefficients per row. X: (n, L)."""
    Xc = X - X.mean(1, keepdims=True)
    n = Xc.shape[1]
    r = np.stack([(Xc[:, :n - k] * Xc[:, k:]).sum(1) / n for k in range(order + 1)], 1)
    from scipy.linalg import toeplitz
    out = np.empty((len(X), order))
    for i in range(len(X)):
        R = toeplitz(r[i, :order])
        try:
            out[i] = np.linalg.solve(R, r[i, 1:order + 1])
        except np.linalg.LinAlgError:
            out[i] = 0.0
    return out


def stats_view(X):
    """HAR measure list per band. X: (n, L). Returns (n, 14)."""
    mean = X.mean(1)
    std = X.std(1)
    med = np.median(X, 1, keepdims=True)
    mad = np.median(np.abs(X - med), 1)
    mx, mn = X.max(1), X.min(1)
    energy = (X ** 2).mean(1)
    q75, q25 = np.percentile(X, [75, 25], axis=1)
    iqr = q75 - q25
    ent = np.empty(len(X))
    for i in range(len(X)):
        h, _ = np.histogram(X[i], bins=256)
        ent[i] = sst.entropy(h + 1e-12)
    ar = ar_coefficients(X, 4)
    skew = sst.skew(X, axis=1)
    kurt = sst.kurtosis(X, axis=1)
    return np.column_stack([mean, std, mad, mx, mn, energy, iqr, ent, ar, skew, kurt])


def stage_extract(raw_dir):
    """Static views for all chunks, in b06's chunk order (BUI order, file
    index, chunk index), plus int16 frames for RF-UAVNet. Also runs b06's
    spectrum pipeline (cached under spectra/)."""
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    done_flag = VIEWS_DIR / 'extract_done.json'
    if done_flag.exists():
        log('extract: already done')
        return
    spectra_dir = DRONERF_DIR / 'spectra'
    b06.aggregate(raw_dir, spectra_dir, log)
    X, labels, groups = b06.assemble(spectra_dir)
    n = len(X)
    assert n == N_CHUNKS_EXPECTED, n
    np.save(VIEWS_DIR / 'spectrum.npy', X.astype(np.float32))
    np.save(VIEWS_DIR / 'spectrum_low.npy', X[:, :1024].astype(np.float32))
    np.save(VIEWS_DIR / 'spectrum_high.npy', X[:, 1024:].astype(np.float32))
    np.save(VIEWS_DIR / 'groups.npy', groups)
    for k, y in labels.items():
        np.save(VIEWS_DIR / f'labels_{k}.npy', y)

    frames = np.lib.format.open_memmap(VIEWS_DIR / 'frames_int16.npy', mode='w+',
                                       dtype=np.int16, shape=(n * FRAMES_PER_CHUNK, 2, FRAME))
    welch = np.empty((n, 2 * 1026), np.float32)
    scal = np.empty((n, 16), np.float32)
    stat = np.empty((n, 28), np.float32)
    band_min = np.array([np.inf, np.inf])
    band_max = np.array([-np.inf, -np.inf])
    i0 = 0
    for bui, _ in b06.BUI:
        t0 = time.time()
        for nfile in range(b06.N_LAST.get(bui, b06.N_LAST_DEFAULT) + 1):
            low = read_csv(raw_dir / f'{bui}L_{nfile}.csv')
            high = read_csv(raw_dir / f'{bui}H_{nfile}.csv')
            nc = len(low) // L
            lo = low[:nc * L].reshape(nc, L)
            hi = high[:nc * L].reshape(nc, L)
            sl = slice(i0, i0 + nc)
            wl, wh = welch_psd_view(lo), welch_psd_view(hi)
            w = np.concatenate([wl, wh], 1)
            w = (w - w.min(1, keepdims=True)) / (w.max(1, keepdims=True) - w.min(1, keepdims=True) + 1e-12)
            welch[sl] = w
            scal[sl] = np.concatenate([spectrahawk_feature_matrix_windows(lo, FS),
                                       spectrahawk_feature_matrix_windows(hi, FS)], 1)
            stat[sl] = np.concatenate([stats_view(lo), stats_view(hi)], 1)
            fr = np.stack([lo.reshape(nc * FRAMES_PER_CHUNK, FRAME),
                           hi.reshape(nc * FRAMES_PER_CHUNK, FRAME)], 1)
            assert np.all(fr == np.round(fr)) and np.abs(fr).max() < 32768
            frames[i0 * FRAMES_PER_CHUNK:(i0 + nc) * FRAMES_PER_CHUNK] = fr.astype(np.int16)
            band_min = np.minimum(band_min, [lo.min(), hi.min()])
            band_max = np.maximum(band_max, [lo.max(), hi.max()])
            i0 += nc
        log(f'extract {bui}: through chunk {i0} in {time.time() - t0:.0f} s')
    assert i0 == n, (i0, n)
    frames.flush()
    np.save(VIEWS_DIR / 'welch_psd.npy', welch)
    np.save(VIEWS_DIR / 'spectral_scalars.npy', scal)
    np.save(VIEWS_DIR / 'stats.npy', stat)
    with open(done_flag, 'w') as f:
        json.dump({'n_chunks': n, 'band_min': band_min.tolist(), 'band_max': band_max.tolist()}, f)
    log('extract done')


# ----------------------------------------------------------------------------
# Stage: images. Four image renderings per chunk through a frozen ResNet50.
# ----------------------------------------------------------------------------

def _fig_to_array(fig):
    import matplotlib
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    from PIL import Image
    im = np.asarray(Image.open(buf).convert('RGB').resize((224, 224)))
    matplotlib.pyplot.close(fig)
    return im


def render_psd(lo, hi):
    """Swinney and Woods 2021: Welch PSD, 1,024-point FFT, 120 overlap, Hann,
    as a line plot. Both bands, one after the other."""
    import matplotlib.pyplot as plt
    _, pl = sps.welch(lo, fs=FS, window='hann', nperseg=1024, noverlap=120)
    _, ph = sps.welch(hi, fs=FS, window='hann', nperseg=1024, noverlap=120)
    fig = plt.figure(figsize=(2.24, 2.24))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(10 * np.log10(np.concatenate([pl, ph]) + 1e-20), lw=0.5, color='k')
    ax.axis('off')
    return _fig_to_array(fig)


def render_spectrogram(lo, hi):
    """Swinney and Woods 2021: STFT spectrogram, 1,024-point FFT (overlap not
    stated; scipy default), log magnitude, low band above high band."""
    import matplotlib.pyplot as plt
    _, _, sl = sps.spectrogram(lo, fs=FS, nperseg=1024)
    _, _, sh = sps.spectrogram(hi, fs=FS, nperseg=1024)
    s = np.concatenate([sl, sh], 0)
    fig = plt.figure(figsize=(2.24, 2.24))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(10 * np.log10(s + 1e-20), aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')
    return _fig_to_array(fig)


def render_histogram(lo, hi):
    """Swinney and Woods 2021: 500-bin amplitude histogram, both bands."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2.24, 2.24))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.hist(lo, bins=500, histtype='step', color='b')
    ax.hist(hi, bins=500, histtype='step', color='r')
    ax.axis('off')
    return _fig_to_array(fig)


def render_gaf(lo, hi):
    """Fu and He 2024: STFT 1,024-point Hann, 128-sample overlap, magnitude
    averaged over time to 513 bins, scaled to [-1, 1], Gramian angular
    summation field cos(phi_i + phi_j); the two bands' fields averaged;
    resized to 224 and coloured with viridis for the ImageNet network."""
    import matplotlib.pyplot as plt
    fields = []
    for x in (lo, hi):
        _, _, z = sps.stft(x, fs=FS, window='hann', nperseg=1024, noverlap=128)
        s = np.abs(z).mean(1)  # 513 bins
        s = 2 * (s - s.min()) / (s.max() - s.min() + 1e-20) - 1
        phi = np.arccos(np.clip(s, -1, 1))
        fields.append(np.cos(phi[:, None] + phi[None, :]))
    g = (fields[0] + fields[1]) / 2
    fig = plt.figure(figsize=(2.24, 2.24))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(g, cmap='viridis', vmin=-1, vmax=1)
    ax.axis('off')
    return _fig_to_array(fig)


RENDERERS = {'psd_resnet50': render_psd, 'spectrogram_resnet50': render_spectrogram,
             'histogram_resnet50': render_histogram, 'gaf_resnet50': render_gaf}


def resnet50_pooled():
    import torch
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed(imgs):  # imgs: (b, 224, 224, 3) uint8
        x = torch.as_tensor(imgs).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            return m(x).numpy().astype(np.float32)
    return embed


def stage_images(raw_dir, debug, views=None):
    import matplotlib
    matplotlib.use('Agg')
    views = views or IMAGE_VIEWS
    unknown = [v for v in views if v not in IMAGE_VIEWS]
    if unknown:
        raise SystemExit(f'images: not image views: {unknown}')
    todo = [v for v in views if not (VIEWS_DIR / f'{v}.npy').exists()]
    if not todo:
        log('images: already done')
        return
    frames = np.load(VIEWS_DIR / 'frames_int16.npy', mmap_mode='r')  # written by extract
    n = frames.shape[0] // FRAMES_PER_CHUNK
    assert n == N_CHUNKS_EXPECTED, n
    embed = resnet50_pooled()
    outs, done = {}, {}
    for v in todo:
        p = VIEWS_DIR / f'{v}.partial.npy'
        if p.exists():
            outs[v] = np.load(p, mmap_mode='r+')
            done[v] = json.load(open(p.with_suffix('.json')))['chunks_done'] if p.with_suffix('.json').exists() else 0
        else:
            outs[v] = np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(n, 2048))
            done[v] = 0
    step = 100  # one file's worth of chunks
    t0 = time.time()
    for c0 in range(0, n, step):
        c1 = min(n, c0 + step)
        for v in todo:
            if c0 < done[v]:
                continue
            imgs = []
            for c in range(c0, c1):
                fr = np.asarray(frames[c * FRAMES_PER_CHUNK:(c + 1) * FRAMES_PER_CHUNK], np.float64)
                imgs.append(RENDERERS[v](fr[:, 0].ravel(), fr[:, 1].ravel()))
            imgs = np.stack(imgs)
            for s in range(0, len(imgs), 32):
                outs[v][c0 + s:c0 + s + len(imgs[s:s + 32])] = embed(imgs[s:s + 32])
            outs[v].flush()
            done[v] = c1
            with open((VIEWS_DIR / f'{v}.partial.npy').with_suffix('.json'), 'w') as fh:
                json.dump({'chunks_done': c1}, fh)
        if (c0 // step) % 10 == 0:
            log(f'images: through chunk {c1}/{n} in {time.time() - t0:.0f} s')
        if debug and c1 >= 300:
            log('images: debug stop after 300 chunks; partial files kept, nothing final written')
            return
    for v in todo:
        os.rename(VIEWS_DIR / f'{v}.partial.npy', VIEWS_DIR / f'{v}.npy')
    log('images done')


# ----------------------------------------------------------------------------
# Models.
# ----------------------------------------------------------------------------

def run_worker(kind, X_tr, y_tr, X_te, seed, debug):
    """Every non-torch model runs in a torch-free subprocess: torch's OpenMP
    runtime and the one xgboost and scikit-learn's thread pools use clash on
    this Mac and the parent dies silently (see _xgb_worker.py)."""
    import json
    import subprocess
    # one folder per process: two jobs on the same dataset must not share it
    tmp = DRONERF_DIR / 'worker_tmp' / f'{socket.gethostname()}_{os.getpid()}'
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / 'X_tr.npy', X_tr)
    np.save(tmp / 'y_tr.npy', y_tr)
    np.save(tmp / 'X_te.npy', X_te)
    for f in ('pred.npy', 'score.npy', 'info.json'):
        (tmp / f).unlink(missing_ok=True)
    cmd = [sys.executable, str(HERE / '_xgb_worker.py'), kind, str(tmp), str(seed)] + (['debug'] if debug else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not (tmp / 'pred.npy').exists():
        raise RuntimeError(f'{kind} worker failed (exit {r.returncode}): {r.stderr[-800:]}')
    info = json.load(open(tmp / 'info.json')) if (tmp / 'info.json').exists() else {}
    return np.load(tmp / 'pred.npy'), info


def fit_predict_model(name, X_tr, y_tr, X_te, n_classes, debug, seed):
    """Model settings (the worker holds the non-torch ones):
    knn k=5; rf 300 trees, class_weight balanced (Shulman code); logreg L2 LBFGS,
    standardised (Swinney); xgboost defaults (Medaiyese, settings not stated);
    svm Gaussian, C and gamma over 2^-15..2^15 step 2^2, inner 3-fold (Kilic;
    debug: 4 x 4 grid chosen on 3,000 chunks); dense = Al-Sa'd's network (b06)."""
    if name == 'dense':
        epochs = 20 if debug else b06.EPOCHS
        Y = np.eye(n_classes)[y_tr]
        return b06.fit_predict(X_tr, Y, X_te, epochs, seed).argmax(1), {'epochs': epochs}
    if name in ('knn', 'rf', 'logreg', 'xgboost', 'svm'):
        return run_worker(name, X_tr, y_tr, X_te, seed, debug)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Stage: evaluate.
# ----------------------------------------------------------------------------

def load_labels():
    labels = {k: np.load(VIEWS_DIR / f'labels_{k}.npy') for k in TASKS}
    groups = np.load(VIEWS_DIR / 'groups.npy')
    return labels, groups


def folds_for(split, y, groups, debug):
    return b06.make_folds(split, y, groups, 3 if debug else b06.FOLDS)


def evaluate_one(view, X_by_fold, model, task, split, y, groups, debug):
    """X_by_fold: callable fold_index -> X (n, d); static views ignore the index."""
    out_path = result_path(view, model, task, split)
    if out_path.exists():
        try:
            if bool(json.load(open(out_path)).get('debug')) == bool(debug):
                return
        except Exception:
            pass
        log(f'{out_path} is from a {"debug" if not debug else "full"} run; redoing it')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds = folds_for(split, y, groups, debug)
    oof = np.full(len(y), -1, dtype=np.int64)
    rows, infos = [], []
    t_all = time.time()
    # A unit killed part way (wall time) resumes at its next fold.
    part = out_path.with_suffix('.partial.json')
    if part.exists():
        try:
            pj = json.load(open(part))
            if bool(pj.get('debug')) == bool(debug):
                rows, infos = pj['rows'], pj['infos']
                oof = np.asarray(pj['oof'], dtype=np.int64)
                log(f'{view} {model} {split} {task}-class: resuming after {len(rows)} fold(s)')
        except Exception:
            rows, infos = [], []
    for f, (tr, te) in enumerate(folds):
        if f < len(rows):
            continue
        X = X_by_fold(f)
        t0 = time.time()
        pred, info = fit_predict_model(model, X[tr], y[tr], X[te], task, debug, SEED + f)
        oof[te] = pred
        rows.append({'fold': f, 'accuracy': float(100 * (pred == y[te]).mean()),
                     'macro_f1': float(100 * f1_score(y[te], pred, average='macro')),
                     'n_test': int(len(te)), 'seconds': time.time() - t0})
        infos.append(info)
        log(f'{split} {task}-class {view} {model} fold {f + 1}/{len(folds)}: '
            f'acc {rows[-1]["accuracy"]:.2f} f1 {rows[-1]["macro_f1"]:.2f} ({rows[-1]["seconds"]:.0f} s)')
        with open(part, 'w') as fh:
            json.dump({'debug': debug, 'rows': rows, 'infos': infos, 'oof': oof.tolist()}, fh)
    cm = confusion_matrix(y, oof, labels=list(range(task)))
    res = {'view': view, 'model': model, 'task': task, 'split': split, 'debug': debug,
           'n_features': int(X_by_fold(0).shape[1]), 'folds': rows,
           'mean_accuracy': float(np.mean([r['accuracy'] for r in rows])),
           'mean_macro_f1': float(np.mean([r['macro_f1'] for r in rows])),
           'confusion_oof': cm.tolist(), 'fit_info': infos, 'seconds': time.time() - t_all}
    res['result_status'] = 'ok'
    np.save(out_path.with_suffix('.oof.npy'), oof)
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=1)
    part.unlink(missing_ok=True)


def result_path(view, model, task, split):
    """The file the ledger reads for one (view, model, task, split) unit."""
    return RESULTS_DIR / split / str(task) / f'{view}__{model}.json'


def fold_view_path(view, task, split, fold):
    return VIEWS_DIR / 'fold_views' / split / str(task) / f'{view}_fold{fold}.npy'


def stage_evaluate(views, models, tasks, splits, debug):
    labels, groups = load_labels()
    n_folds = 3 if debug else b06.FOLDS
    for view in views:
        if view in EMBED_VIEWS:
            for split in splits:
                for task in tasks:
                    missing = [f for f in range(n_folds) if not fold_view_path(view, task, split, f).exists()]
                    if missing:
                        raise SystemExit(f'evaluate: {view} {split} {task}-class has no fold view for '
                                         f'fold(s) {missing}; run --stage embed first')
                    cache = {}

                    def X_by_fold(f, view=view, task=task, split=split, cache=cache):
                        if f not in cache:
                            cache.clear()
                            cache[f] = np.load(fold_view_path(view, task, split, f))
                        return cache[f]
                    for model in models:
                        evaluate_one(view, X_by_fold, model, task, split, labels[task], groups, debug)
            continue
        path = VIEWS_DIR / f'{view}.npy'
        if not path.exists():
            raise SystemExit(f'evaluate: {view} not built ({path}); run --stage extract or images first')
        X = np.load(path)
        for split in splits:
            for task in tasks:
                for model in models:
                    evaluate_one(view, lambda f: X, model, task, split, labels[task], groups, debug)


# ----------------------------------------------------------------------------
# Stage: embed. Fold-trained CNN views (RF-UAVNet, Al-Emadi).
# ----------------------------------------------------------------------------

def build_rfuavnet(n_class):
    """Port of RF_UAVNet.m (github.com/ThienHuynhThe/RF-based-Drone-Surveillance-with-DL).
    Input (b, 2, 10000). Returns logits and the 320-value depthcat embedding."""
    import torch
    import torch.nn as nn

    class GConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(64, 64, 3, stride=2, padding=1, groups=8)  # 8 filters x 8 groups
            self.elu = nn.ELU(1.0)

        def forward(self, x):
            return self.elu(self.conv(x))

    class RFUAVNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(2, 64, 5, stride=5, padding=2),
                                      nn.BatchNorm1d(64), nn.ELU(1.0))
            self.pool = nn.MaxPool1d(2, 2)
            self.g = nn.ModuleList([GConv() for _ in range(4)])
            self.fc = nn.Linear(5 * 64, n_class)

        def forward(self, x):
            x = self.stem(x)                       # elu_1
            gaps = []
            for g in self.g:
                skip = self.pool(x)                # maxpool_k
                x = g(x)                           # elu_(k+1)
                gaps.append(x.mean(2))             # gapool of elu_2 .. elu_5
                x = x + skip                       # addition_k
            gaps.append(x.mean(2))                 # gapool_1 of addition_4
            # depthcat order in the .m file: gapool_1, gapool_3(elu_5), gapool_4(elu_4),
            # gapool_5(elu_3), gapool_2(elu_2)
            emb = torch.cat([gaps[4], gaps[3], gaps[2], gaps[1], gaps[0]], 1)
            return self.fc(emb), emb
    return RFUAVNet()


def build_alemadi(n_class, n_in=2047):
    """Port of Classification_CNN_DroneModeIdentification.py.py (Keras).
    Input (b, 1, 2047). Returns logits and the 256-unit dense embedding."""
    import torch.nn as nn

    class AlEmadi(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.AvgPool1d(3),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AvgPool1d(3),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.AvgPool1d(3),
                nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.2), nn.Flatten())
            flat = 256 * (n_in // 3 // 3 // 3)
            self.dense = nn.Sequential(nn.Linear(flat, 256), nn.ReLU())
            self.out = nn.Linear(256, n_class)

        def forward(self, x):
            emb = self.dense(self.conv(x))
            return self.out(emb), emb
    return AlEmadi()


def _train_torch(model, X, y, epochs, batch, opt, sched, seed, log_every, Xdtype_fn):
    import torch
    g = torch.Generator().manual_seed(seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.as_tensor(y, dtype=torch.long)
    n = len(y)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        t0, tot = time.time(), 0.0
        for st in range(0, n, batch):
            idx = perm[st:st + batch].numpy()
            xb = Xdtype_fn(idx)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if sched is not None:
            sched.step()
        log(f'{log_every} epoch {ep + 1}/{epochs}: loss {tot / n:.4f} ({time.time() - t0:.0f} s)')


def _embed_torch(model, n, batch, Xdtype_fn):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for st in range(0, n, batch):
            idx = np.arange(st, min(n, st + batch))
            outs.append(model(Xdtype_fn(idx))[1].numpy())
    return np.concatenate(outs)


def embed_rfuavnet(tr_chunks, y_chunks, task, debug, seed, tag):
    """Train on the training chunks' frames; return (n_chunks, 320), each chunk
    the mean of its 10 frames' embeddings."""
    import torch
    torch.manual_seed(seed)
    frames = np.load(VIEWS_DIR / 'frames_int16.npy', mmap_mode='r')
    meta = json.load(open(VIEWS_DIR / 'extract_done.json'))
    bmin = np.array(meta['band_min'], np.float32).reshape(1, 2, 1)
    bmax = np.array(meta['band_max'], np.float32).reshape(1, 2, 1)

    def get(idx_frames):
        x = np.asarray(frames[np.sort(idx_frames)], np.float32)
        return torch.as_tensor((x - bmin) / (bmax - bmin))

    fpc = 1 if debug else FRAMES_PER_CHUNK  # debug: the first frame of each chunk only
    tr_frames = (tr_chunks[:, None] * FRAMES_PER_CHUNK + np.arange(fpc)[None]).ravel()
    y_frames = np.repeat(y_chunks, fpc)
    order = np.argsort(tr_frames)
    tr_frames, y_frames = tr_frames[order], y_frames[order]
    model = build_rfuavnet(task)
    epochs = 1 if debug else 5
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.1)
    _train_torch(model, None, y_frames, epochs, 128, opt, sched, seed, f'{tag} rfuavnet',
                 lambda idx: get(tr_frames[idx]))
    n_frames = frames.shape[0]
    emb = _embed_torch(model, n_frames, 512, lambda idx: get(idx))
    return emb.reshape(-1, FRAMES_PER_CHUNK, 320).mean(1)


def embed_alemadi(X, tr, y_tr, task, debug, seed, tag):
    import torch
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    model = build_alemadi(task, X.shape[1])
    epochs = 5 if debug else 350
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _train_torch(model, None, y_tr, epochs, 50, opt, None, seed, f'{tag} alemadi',
                 lambda idx: Xt[tr[idx]])
    return _embed_torch(model, len(X), 512, lambda idx: Xt[idx])


def stage_embed(views, tasks, splits, debug, only_folds=None):
    """Train each fold-trained CNN on one fold's training chunks and store the
    embedding of every chunk. Builds only; --stage evaluate scores them."""
    labels, groups = load_labels()
    spectrum = np.load(VIEWS_DIR / 'spectrum.npy')
    for view in views:
        for split in splits:
            for task in tasks:
                y = labels[task]
                folds = folds_for(split, y, groups, debug)
                for f, (tr, te) in enumerate(folds):
                    if only_folds is not None and f not in only_folds:
                        continue
                    p = fold_view_path(view, task, split, f)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if p.exists():
                        continue
                    tag = f'{split} {task}-class fold {f + 1}/{len(folds)}'
                    t0 = time.time()
                    if view == 'rfuavnet_emb':
                        E = embed_rfuavnet(tr, y[tr], task, debug, SEED + f, tag)
                    else:
                        E = embed_alemadi(spectrum, tr, y[tr], task, debug, SEED + f, tag)
                    tmp = p.with_suffix('.tmp.npy')
                    np.save(tmp, E.astype(np.float32))
                    os.replace(tmp, p)
                    log(f'{tag} {view}: embedded {E.shape} in {time.time() - t0:.0f} s')


# ----------------------------------------------------------------------------
# Stage: summarise.
# ----------------------------------------------------------------------------

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B, 'fro') ** 2
    return float(hsic / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def stage_summarise(debug):
    labels, groups = load_labels()
    results = []
    for p in RESULTS_DIR.rglob('*__*.json'):
        if p.name.endswith('.partial.json'):
            continue
        with open(p) as fh:
            results.append(json.load(fh))
    lines = ['# DroneRF view and model screen', '',
             f'{len(results)} (view, model, task, split) runs. '
             + ('DEBUG settings: not results.' if any(r['debug'] for r in results) else ''), '']
    views = [v for v in STATIC_VIEWS + IMAGE_VIEWS + EMBED_VIEWS
             if any(r['view'] == v for r in results)]
    models = [m for m in MODELS if any(r['model'] == m for r in results)]
    for split in SPLITS:
        for task in TASKS:
            sub = [r for r in results if r['split'] == split and r['task'] == task]
            if not sub:
                continue
            lines += [f'## {split} split, {task}-class', '',
                      'One row is one view; a cell is the mean over folds of accuracy / macro-F1, in %.', '',
                      '| view | features | ' + ' | '.join(models) + ' |',
                      '|---|---|' + '---|' * len(models)]
            for v in views:
                cells = []
                for m in models:
                    r = [x for x in sub if x['view'] == v and x['model'] == m]
                    cells.append(f'{r[0]["mean_accuracy"]:.1f} / {r[0]["mean_macro_f1"]:.1f}' if r else '')
                nf = next((x['n_features'] for x in sub if x['view'] == v), '')
                lines.append(f'| {v} | {nf} | ' + ' | '.join(cells) + ' |')
            lines.append('')
            # error overlap between views, per model
            for m in models:
                oofs = {}
                for v in views:
                    p = RESULTS_DIR / split / str(task) / f'{v}__{m}.oof.npy'
                    if p.exists():
                        oofs[v] = np.load(p)
                if len(oofs) < 2:
                    continue
                y = labels[task]
                vs = list(oofs)
                lines += [f'### error overlap, {m}, {split} {task}-class', '',
                          'Cell: of the chunks that either view gets wrong, the fraction both get wrong '
                          '(Jaccard of the two error sets), in %. Diagonal: error rate in %.', '',
                          '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
                err = {v: (oofs[v] != y) & (oofs[v] >= 0) for v in vs}
                for a in vs:
                    cells = []
                    for b in vs:
                        if a == b:
                            cells.append(f'{100 * err[a].mean():.1f}')
                        else:
                            u = (err[a] | err[b]).sum()
                            cells.append(f'{100 * (err[a] & err[b]).sum() / max(u, 1):.0f}')
                    lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
                lines.append('')
    # CKA between static and image views on a stratified subsample
    stat_views = [v for v in STATIC_VIEWS + IMAGE_VIEWS if (VIEWS_DIR / f'{v}.npy').exists()]
    if len(stat_views) >= 2:
        y10 = labels[10]
        rng = np.random.RandomState(SEED)
        idx = np.concatenate([rng.choice(np.where(y10 == c)[0], min(300, (y10 == c).sum()), replace=False)
                              for c in range(10)])
        mats = {}
        for v in stat_views:
            X = np.load(VIEWS_DIR / f'{v}.npy', mmap_mode='r')[np.sort(idx)].astype(np.float64)
            X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            mats[v] = X
        lines += ['## linear CKA between views', '',
                  f'Linear centred kernel alignment between standardised view matrices on {len(idx)} chunks '
                  '(up to 300 per 10-class label). 1 = the same information up to a linear map, 0 = unrelated.', '',
                  '| | ' + ' | '.join(stat_views) + ' |', '|---|' + '---|' * len(stat_views)]
        cka = {}
        for a in stat_views:
            cells = []
            for b in stat_views:
                key = tuple(sorted((a, b)))
                if key not in cka:
                    cka[key] = linear_cka(mats[a], mats[b])
                cells.append(f'{cka[key]:.2f}')
            lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    with open(RESULTS_DIR / 'summary.md', 'w') as fh:
        fh.write('\n'.join(lines))
    with open(RESULTS_DIR / 'summary.json', 'w') as fh:
        json.dump(results, fh, indent=1)
    log(f'summary written: {RESULTS_DIR / "summary.md"}')


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['extract', 'images', 'evaluate', 'embed', 'summarise', 'all'],
                    default='all')
    ap.add_argument('--views', default=None, help='comma list; default all of the stage')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--tasks', default='2,4,10')
    ap.add_argument('--splits', default='published,grouped')
    ap.add_argument('--folds', default=None, help='embed only: comma list of fold indices to build')
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else DRONERF_DIR / 'raw'
    models = args.models.split(',')
    tasks = [int(t) for t in args.tasks.split(',')]
    splits = args.splits.split(',')
    st = args.stage
    if st in ('extract', 'all'):
        stage_extract(raw_dir)
    if st in ('images', 'all'):
        views = [v for v in args.views.split(',') if v in IMAGE_VIEWS] if args.views else None
        stage_images(raw_dir, args.debug, views)
    if st in ('embed', 'all'):
        views = [v for v in (args.views.split(',') if args.views else EMBED_VIEWS) if v in EMBED_VIEWS]
        only = [int(f) for f in args.folds.split(',')] if args.folds else None
        stage_embed(views, tasks, splits, args.debug, only)
    if st in ('evaluate', 'all'):
        views = args.views.split(',') if args.views else STATIC_VIEWS + IMAGE_VIEWS + EMBED_VIEWS
        stage_evaluate(views, models, tasks, splits, args.debug)
    if st in ('summarise', 'all'):
        stage_summarise(args.debug)


if __name__ == '__main__':
    main()
