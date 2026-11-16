#!/usr/bin/env python3
"""WiSig view and model screen: candidate representations of one packet,
scored under the authors' protocols (b13_wisig_published.py), with per-fold
accuracy and macro-F1, out-of-fold predictions, error overlap and linear CKA
between views.

Object: one packet, 256 complex samples at 25 Msps (10.24 us: the 8 us
L-STF and the first 2.24 us of the L-LTF), normalised to unit average power
as the authors' `norm` does. Label: the transmitter.

OPEN SETTINGS, no default (the screen refuses to start without them):
    --protocols   which of the authors' protocol settings to score, each with
                  its equalisation flag, e.g. ntx_150:0,nrx_25:0,nrx_25:1.
                  The values and the authors' printed accuracies are in
                  b13_wisig_published.py.
    --models      which models; candidates below.

Views (one fixed-length vector per packet). "code" = ported from a named
repository; "paper" = written from the paper's text; "this file" = a
setting the source does not give, chosen here and open.

    name                      size  source
    iq                         512  WiSig code: the network's input, (256, 2)
                                    flattened as I0 Q0 I1 Q1 ...
    polar                      512  amplitude |x| then phase atan2(Q, I).
                                    Elmaghbub & Hamdaoui 2021 (abstract only)
    phase_increment            255  wrapped phase difference phi[n]-phi[n-1].
                                    arXiv 2605.30364 (preprint, no code): the
                                    third input channel of its Variant E
    cfo_compensated_iq         512  iq after removing the carrier frequency
                                    offset estimated from the L-STF (below)
    impairments                  6  CFO (kHz), DC offset I and Q, I/Q gain
                                    ratio, I/Q phase skew (rad), EVM against
                                    the ideal preamble. Sankhe 2019 (ORACLE)
                                    shows I/Q imbalance and DC offset
                                    separate radios; estimators this file's
    fft_magnitude              256  |FFT(x)|, fftshifted. Shen 2021 (LoRa)
    welch_psd                   64  log10 Welch PSD, 64-sample Hann
                                    segments, 50% overlap, two-sided (this
                                    file's settings; research note "e.g.")
    spectral_stats               6  from welch_psd: centroid, spread,
                                    flatness, 95% roll-off, skewness and
                                    kurtosis of the PSD. No measured RF
                                    fingerprinting result found
    stft_spectrogram           448  log |STFT|^2, Hann 64, hop 32, two-sided:
                                    64 bins x 7 frames (this file's window;
                                    Shen 2021 used 256/128 on 8,192 samples)
    channel_independent_spectrogram
                               384  log(|X[k,m+1]|^2 / |X[k,m]|^2) on the
                                    STFT above: 64 x 6. Shen et al., TIFS 2022
    stf_response                24  the L-STF's 12 occupied subcarriers:
                                    magnitude (dB) and phase with its best
                                    linear fit removed. 25 -> 20 Msps
                                    (resample_poly 4/5, as the authors'
                                    equaliser does), CFO removed, the nine
                                    whole 16-sample periods after the first
                                    averaged, 16-point FFT, divided by the
                                    standard's L-STF values. This file's
                                    construction from IEEE 802.11-2016
                                    17.3.3; WiSig Fig. 12 motivates it. The
                                    L-LTF is not used: only 56 of its samples
                                    are in the packet, less than one symbol
    preamble_error             512  r - g * s: the packet minus the ideal
                                    L-STF + L-LTF (standard, resampled to
                                    25 Msps) at the best timing offset, after
                                    CFO removal, with the least-squares
                                    complex gain g. Merchant et al. (error
                                    signal); this file's alignment
    dctf                     4,096  differential constellation trace figure:
                                    D[n] = x[n+1] conj(x[n]); 64 x 64
                                    histogram of (Re D, Im D) over [-3, 3]^2.
                                    Peng et al., IEEE TVT 2020; lag 1 and
                                    range are this file's (paper not read)
    hos_cumulants                8  |C20| |C21| |C40| |C41| |C42| |C60| |C61|
                                    |C63| of the packet (Swami & Sadler 2000
                                    formulas). Research note: no numbers found
    cnn_emb                     80  the authors' network trained on the fold's
                                    training packets (b13 train_net, its
                                    early stopping on the fold's validation
                                    packets); the 80-unit layer. One build per
                                    protocol setting, flag and fold

Left out, with the reason: wavelet scalogram, phase noise and turn-on
transient (no recipe in any source the research file cites); bispectrum
image (no size or reduction given; the cumulants are kept); DoLoS (needs a
whole L-LTF symbol); Transformer embeddings (arXiv 2605.30364 has no code).

Models (candidates; which run is open):
    wisig_cnn  the authors' network (b13 train_net), on the views that keep
               the (256, 2) shape: iq, cfo_compensated_iq, polar
    knn, rf, logreg, xgboost
               the DroneRF screen's settings (_xgb_worker.py): k = 5;
               300 trees, class_weight balanced; L2 LBFGS standardised;
               xgboost defaults. No WiSig paper uses a vector model.
Vector models fit on the fold's training packets; the validation packets
are used only by the networks' early stopping. Every fold is scored on the
authors' test packets.

Stages (each resumable; finished units skipped):
    packets    one subset and flag as one (n, 256, 2) array, from the pickle
    views      one view on every packet of one subset and flag
    embed      cnn_emb for one protocol setting, flag and fold
    evaluate   one view x model x protocol setting (every fold)
    summarise  summary.md

Output: <KIRBY_WISIG_DIR>/views/<subset>_eq<e>/<view>.npy,
<KIRBY_WISIG_DIR>/views/fold_views/<protocol>__eq<e>/cnn_emb_fold<f>.npy,
<KIRBY_BENCH_RESULTS>/b13_wisig_views/<protocol>__eq<e>/<view>__<model>.json
"""
import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal as sps
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('b13', HERE / 'b13_wisig_published.py')
b13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b13)

WISIG_DIR = b13.WISIG_DIR
VIEWS_DIR = WISIG_DIR / 'views'
RESULTS_DIR = b13.RESULTS_DIR / 'b13_wisig_views'
FS = 25e6
N = 256
CHUNK = 20_000

PACKET_VIEWS = {   # name -> width
    'iq': 512, 'polar': 512, 'phase_increment': 255, 'cfo_compensated_iq': 512,
    'impairments': 6, 'fft_magnitude': 256, 'welch_psd': 64, 'spectral_stats': 6,
    'stft_spectrogram': 448, 'channel_independent_spectrogram': 384,
    'stf_response': 24, 'preamble_error': 512, 'dctf': 4096, 'hos_cumulants': 8,
}
FOLD_VIEWS = {'cnn_emb': 80}
SEQUENCE_VIEWS = ('iq', 'cfo_compensated_iq', 'polar')   # keep the (256, 2) shape
MODELS = ['wisig_cnn', 'knn', 'rf', 'logreg', 'xgboost']
VECTOR_MODELS = ('knn', 'rf', 'logreg', 'xgboost')


def all_views():
    return list(PACKET_VIEWS) + list(FOLD_VIEWS)


def model_applies(view, model):
    return model != 'wisig_cnn' or view in SEQUENCE_VIEWS


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


# ----------------------------------------------------------------------------
# IEEE 802.11-2016 17.3.3: L-STF and L-LTF, 20 Msps, then 25 Msps.
# ----------------------------------------------------------------------------

_STF = np.sqrt(13 / 6) * np.array(
    [0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
     1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0,
     0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0])                                   # k = -26..26
_LTF = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
                 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1,
                 1, 1, 1], dtype=complex)


def _ifft64(spec53):
    X = np.zeros(64, complex)
    k = np.arange(-26, 27)
    X[k % 64] = spec53
    return np.fft.ifft(X) * 64 / np.sqrt(52)


def ideal_preamble_25():
    """L-STF (160 samples) + L-LTF (GI2 32 + 2 x 64) at 20 Msps, resampled to
    25 Msps (400 samples), unit average power."""
    s = _ifft64(_STF)
    stf = np.tile(s, 3)[:160]
    l = _ifft64(_LTF)
    ltf = np.concatenate([l[-32:], l, l])
    p = sps.resample_poly(np.concatenate([stf, ltf]), 5, 4)
    return p / np.sqrt(np.mean(np.abs(p) ** 2))


IDEAL = ideal_preamble_25()
# STF occupied subcarriers k = 4m, m = +-1..+-6, as 16-point FFT bins.
_STF_K = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
_STF_BINS = (_STF_K // 4) % 16
_STF_VALS = _STF[_STF_K + 26]


# ----------------------------------------------------------------------------
# Per-packet views. x: (n, 256) complex, unit average power.
# ----------------------------------------------------------------------------

def to_complex(X):
    X = b13.norm(X)
    return X[..., 0].astype(np.float64) + 1j * X[..., 1].astype(np.float64)


def cfo_estimate(x):
    """L-STF autocorrelation at one period (0.8 us = 20 samples at 25 Msps)
    over the first 200 samples. Returns Hz; range +-625 kHz."""
    L = 20
    c = np.sum(x[:, L:200] * np.conj(x[:, :200 - L]), axis=1)
    return np.angle(c) / (2 * np.pi * L / FS)


def derotate(x, f_hz):
    n = np.arange(x.shape[1])
    return x * np.exp(-2j * np.pi * f_hz[:, None] * n[None, :] / FS)


def v_iq(x):
    return np.stack([x.real, x.imag], -1).reshape(len(x), -1)


def v_polar(x):
    return np.concatenate([np.abs(x), np.angle(x)], 1)


def v_phase_increment(x):
    return np.angle(x[:, 1:] * np.conj(x[:, :-1]))


def v_fft_magnitude(x):
    return np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1))


def welch(x):
    _, P = sps.welch(x, fs=FS, window='hann', nperseg=64, noverlap=32, return_onesided=False,
                     axis=1, detrend=False)
    return np.fft.fftshift(P, axes=1)


def v_welch_psd(x):
    return np.log10(welch(x) + 1e-12)


def v_spectral_stats(x):
    P = welch(x)
    f = np.arange(P.shape[1]) - P.shape[1] / 2
    p = P / P.sum(1, keepdims=True)
    cen = (p * f).sum(1)
    spread = np.sqrt((p * (f - cen[:, None]) ** 2).sum(1))
    flat = np.exp(np.mean(np.log(P + 1e-12), 1)) / (P.mean(1) + 1e-12)
    roll = np.argmax(np.cumsum(p, 1) >= 0.95, 1).astype(float)
    z = (f[None, :] - cen[:, None]) / (spread[:, None] + 1e-12)
    skew = (p * z ** 3).sum(1)
    kurt = (p * z ** 4).sum(1)
    return np.stack([cen, spread, flat, roll, skew, kurt], 1)


def stft(x):
    _, _, Z = sps.stft(x, fs=FS, window='hann', nperseg=64, noverlap=32, return_onesided=False,
                       boundary=None, padded=False, axis=1)
    return np.fft.fftshift(Z, axes=1)                      # (n, 64, 7)


def v_stft_spectrogram(x):
    return np.log10(np.abs(stft(x)) ** 2 + 1e-12).reshape(len(x), -1)


def v_channel_independent_spectrogram(x):
    S = np.abs(stft(x)) ** 2 + 1e-12
    return np.log(S[:, :, 1:] / S[:, :, :-1]).reshape(len(x), -1)


def v_stf_response(x, cfo):
    y = derotate(x, cfo)
    y20 = sps.resample_poly(y, 4, 5, axis=1)               # 204 samples at 20 Msps
    per = y20[:, 16:160].reshape(len(y), 9, 16).mean(1)
    H = np.fft.fft(per, axis=1)[:, _STF_BINS] / _STF_VALS[None, :]
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    ph = np.unwrap(np.angle(H), axis=1)
    A = np.vstack([_STF_K, np.ones(12)]).T
    coef, *_ = np.linalg.lstsq(A, ph.T, rcond=None)
    return np.concatenate([mag, ph - (A @ coef).T], 1)


def align_preamble(y, max_offset=len(IDEAL) - N):
    """Offset into the ideal preamble that best matches the packet (the
    packet starts somewhere inside the L-STF), and the LS complex gain."""
    best_o = np.zeros(len(y), int)
    best_c = np.full(len(y), -1.0)
    for o in range(0, max_offset + 1):
        s = IDEAL[o:o + N]
        c = np.abs(y @ np.conj(s)) / np.sqrt((np.abs(s) ** 2).sum())
        better = c > best_c
        best_c[better], best_o[better] = c[better], o
    S = np.stack([IDEAL[o:o + N] for o in best_o])
    g = np.sum(y * np.conj(S), 1) / np.sum(np.abs(S) ** 2, 1)
    return S, g, best_o


def preamble_residual(x, cfo):
    y = derotate(x, cfo)
    S, g, _ = align_preamble(y)
    return y - g[:, None] * S, g[:, None] * S


def v_preamble_error(x, cfo):
    e, _ = preamble_residual(x, cfo)
    return np.stack([e.real, e.imag], -1).reshape(len(x), -1)


def v_impairments(x, cfo):
    dc = x.mean(1)
    z = x - dc[:, None]
    ii, qq, iq = (z.real ** 2).mean(1), (z.imag ** 2).mean(1), (z.real * z.imag).mean(1)
    gain = np.sqrt(qq / ii)
    skew = np.arcsin(np.clip(iq / np.sqrt(ii * qq), -1, 1))
    e, s = preamble_residual(x, cfo)
    evm = np.sqrt((np.abs(e) ** 2).sum(1) / (np.abs(s) ** 2).sum(1))
    return np.stack([cfo / 1e3, dc.real, dc.imag, gain, skew, evm], 1)


def v_dctf(x, lag=1, bins=64, lim=3.0):
    D = x[:, lag:] * np.conj(x[:, :-lag])
    edges = np.linspace(-lim, lim, bins + 1)
    ix = np.clip(np.digitize(D.real, edges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(D.imag, edges) - 1, 0, bins - 1)
    out = np.zeros((len(x), bins * bins), np.uint8)
    flat = ix * bins + iy
    for i in range(len(x)):
        out[i] = np.bincount(flat[i], minlength=bins * bins).astype(np.uint8)
    return out


def v_hos_cumulants(x):
    z = x - x.mean(1, keepdims=True)
    M = lambda p, q: np.mean(z ** (p - q) * np.conj(z) ** q, 1)   # noqa: E731
    m20, m21, m40, m41, m42 = M(2, 0), M(2, 1), M(4, 0), M(4, 1), M(4, 2)
    m60, m61, m63 = M(6, 0), M(6, 1), M(6, 3)
    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
    c60 = m60 - 15 * m20 * m40 + 30 * m20 ** 3
    c61 = m61 - 5 * m21 * m40 - 10 * m20 * m41 + 30 * m20 ** 2 * m21
    c63 = m63 - 9 * m21 * m42 + 12 * m21 ** 3 - 3 * m20 * np.conj(M(4, 3)) \
        - 3 * np.conj(m20) * m41 + 18 * m20 * np.conj(m20) * m21
    return np.abs(np.stack([m20, m21, c40, c41, c42, c60, c61, c63], 1))


def compute_view(view, X):
    x = to_complex(X)
    if view in ('cfo_compensated_iq', 'impairments', 'stf_response', 'preamble_error'):
        cfo = cfo_estimate(x)
        if view == 'cfo_compensated_iq':
            return v_iq(derotate(x, cfo))
        return {'impairments': v_impairments, 'stf_response': v_stf_response,
                'preamble_error': v_preamble_error}[view](x, cfo)
    return globals()[f'v_{view}'](x)


def view_path(subset, eq, view):
    return VIEWS_DIR / f'{subset}_eq{eq}' / f'{view}.npy'


def stage_views(subset, eq, views):
    _, X = b13.packet_table(subset, eq)
    for view in views:
        out = view_path(subset, eq, view)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp.npy')
        dtype = np.uint8 if view == 'dctf' else np.float32
        M = np.lib.format.open_memmap(tmp, mode='w+', dtype=dtype, shape=(len(X), PACKET_VIEWS[view]))
        t0 = time.time()
        for s in range(0, len(X), CHUNK):
            V = compute_view(view, np.asarray(X[s:s + CHUNK]))
            if not np.isfinite(V.astype(np.float64)).all():
                raise RuntimeError(f'{view}: non-finite values in packets {s}..{s + len(V)}')
            M[s:s + len(V)] = V
            log(f'{subset} eq{eq} {view}: {s + len(V)} / {len(X)} packets ({time.time() - t0:.0f} s)')
        M.flush()
        del M
        os.replace(tmp, out)
        log(f'{subset} eq{eq} {view}: {len(X)} x {PACKET_VIEWS[view]} written ({time.time() - t0:.0f} s)')


# ----------------------------------------------------------------------------
# Folds, fold views, models.
# ----------------------------------------------------------------------------

def parse_setting(s):
    """'nrx_25:1' -> ('nrx_25', 1)"""
    p, eq = s.split(':')
    b13.parse_protocol(p)
    return p, int(eq)


def folds_for(protocol, eq):
    kind, val = b13.parse_protocol(protocol)
    return b13.protocol_folds(kind, val, eq), b13.SUBSET[kind]


def fold_view_path(protocol, eq, view, fold, debug=False):
    """A fold view trained with a shortened epoch cap goes to its own folder,
    so a real run never reads it."""
    return VIEWS_DIR / ('fold_views_debug' if debug else 'fold_views') / f'{protocol}__eq{eq}' / \
        f'{view}_fold{fold}.npy'


def stage_embed(protocol, eq, folds_only=None, epochs=b13.EPOCHS):
    """cnn_emb: train the authors' network on the fold's training packets
    (early stopping on its validation packets), keep the 80-unit layer for
    every packet of the subset."""
    folds, subset = folds_for(protocol, eq)
    _, X = b13.packet_table(subset, eq)
    for f, fold in enumerate(folds):
        if folds_only is not None and f not in folds_only:
            continue
        out = fold_view_path(protocol, eq, 'cnn_emb', f, epochs != b13.EPOCHS)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        (tri, ytr), (vai, yva) = fold['train'], fold['val']
        net, hist = b13.train_net(b13.norm(X[tri]), ytr, b13.norm(X[vai]), yva, fold['n_class'],
                                  b13.SEED + f, epochs, log_prefix=f'cnn_emb {protocol} eq{eq} fold {f} ')
        E = np.concatenate([b13.predict(net, b13.norm(np.asarray(X[s:s + CHUNK])), embed=True)
                            for s in range(0, len(X), CHUNK)])
        tmp = out.with_suffix('.tmp.npy')
        np.save(tmp, E.astype(np.float32))
        os.replace(tmp, out)
        log(f'cnn_emb {protocol} eq{eq} fold {f}: {E.shape} after {len(hist)} epochs')


def run_worker(kind, X_tr, y_tr, X_te, seed):
    """Non-torch models in a torch-free subprocess (see _xgb_worker.py)."""
    import subprocess
    tmp = WISIG_DIR / 'worker_tmp' / f'{socket.gethostname()}_{os.getpid()}'
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / 'X_tr.npy', X_tr)
    np.save(tmp / 'y_tr.npy', y_tr)
    np.save(tmp / 'X_te.npy', X_te)
    for f in ('pred.npy', 'score.npy', 'info.json'):
        (tmp / f).unlink(missing_ok=True)
    r = subprocess.run([sys.executable, str(HERE / '_xgb_worker.py'), kind, str(tmp), str(seed)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not (tmp / 'pred.npy').exists():
        raise RuntimeError(f'{kind} worker failed (exit {r.returncode}): {r.stderr[-800:]}')
    return np.load(tmp / 'pred.npy')


def fit_predict(model, view, X_tr, y_tr, X_va, y_va, X_te, n_class, seed, epochs):
    if model == 'wisig_cnn':
        shp = lambda A: np.asarray(A, np.float32).reshape(len(A), 256, 2)   # noqa: E731
        net, hist = b13.train_net(shp(X_tr), y_tr, shp(X_va), y_va, n_class, seed, epochs,
                                  log_prefix=f'wisig_cnn {view} ')
        return b13.predict(net, shp(X_te)).argmax(1), {'epochs_run': len(hist)}
    if model in VECTOR_MODELS:
        return run_worker(model, np.asarray(X_tr, np.float32), y_tr, np.asarray(X_te, np.float32), seed), {}
    raise ValueError(model)


def result_path(protocol, eq, view, model):
    return RESULTS_DIR / f'{protocol}__eq{eq}' / f'{view}__{model}.json'


def evaluate_one(protocol, eq, view, model, epochs=b13.EPOCHS):
    if not model_applies(view, model):
        raise SystemExit(f'{model} takes (256, 2) inputs; {view} is not one of {SEQUENCE_VIEWS}')
    out = result_path(protocol, eq, view, model)
    debug = epochs != b13.EPOCHS
    if out.exists() and bool(json.loads(out.read_text()).get('debug')) == debug:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    folds, subset = folds_for(protocol, eq)
    part = out.with_suffix('.partial.json')
    rows = []
    if part.exists():
        pj = json.loads(part.read_text())
        if bool(pj.get('debug')) == debug:
            rows = pj['rows']
    oof = {}
    t_all = time.time()
    for f, fold in enumerate(folds):
        if f < len(rows):
            continue
        if view in FOLD_VIEWS:
            p = fold_view_path(protocol, eq, view, f, debug)
            if not p.exists():
                raise SystemExit(f'{p} missing: run --stage embed first')
            V = np.load(p)  # whole file: scattered reads from a memory map hang on exFAT
        else:
            p = view_path(subset, eq, view)
            if not p.exists():
                raise SystemExit(f'{p} missing: run --stage views first')
            V = np.load(p)  # whole file: scattered reads from a memory map hang on exFAT
        (tri, ytr), (vai, yva), (tei, yte) = fold['train'], fold['val'], fold['test']
        t0 = time.time()
        pred, info = fit_predict(model, view, V[tri], ytr, V[vai], yva, V[tei], fold['n_class'],
                                 b13.SEED + f, epochs)
        rows.append({'fold': f, 'accuracy': float(100 * (pred == yte).mean()),
                     'macro_f1': float(100 * f1_score(yte, pred, average='macro')),
                     'n_train': int(len(tri)), 'n_test': int(len(tei)), 'n_class': fold['n_class'],
                     'held_out': fold['held_out'], 'seconds': time.time() - t0, 'info': info})
        np.save(out.parent / f'{view}__{model}.fold{f}.pred.npy', np.stack([tei, yte, pred]))
        log(f'{protocol} eq{eq} {view} {model} fold {f + 1}/{len(folds)}: acc {rows[-1]["accuracy"]:.2f} '
            f'f1 {rows[-1]["macro_f1"]:.2f} ({rows[-1]["seconds"]:.0f} s)')
        part.write_text(json.dumps({'debug': debug, 'rows': rows}))
    kind, val = b13.parse_protocol(protocol)
    res = {'protocol': protocol, 'equalized': eq, 'view': view, 'model': model, 'debug': debug,
           'n_features': int(FOLD_VIEWS.get(view) or PACKET_VIEWS[view]), 'folds': rows,
           'mean_accuracy': float(np.mean([r['accuracy'] for r in rows])),
           'mean_macro_f1': float(np.mean([r['macro_f1'] for r in rows])),
           'published_accuracy': (None if b13.published_score(kind, val, eq) is None
                                  else 100 * b13.published_score(kind, val, eq)),
           'seconds': time.time() - t_all, 'result_status': 'ok'}
    out.write_text(json.dumps(res, indent=1))
    part.unlink(missing_ok=True)
    del oof


# ----------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------

def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B, 'fro') ** 2
    return float(hsic / (np.linalg.norm(A.T @ A, 'fro') * np.linalg.norm(B.T @ B, 'fro') + 1e-30))


def stage_summarise(settings):
    lines = ['# WiSig view and model screen', '']
    for protocol, eq in settings:
        d = RESULTS_DIR / f'{protocol}__eq{eq}'
        res = [json.loads(p.read_text()) for p in sorted(d.glob('*__*.json'))
               if not p.name.endswith('.partial.json')] if d.exists() else []
        if not res:
            continue
        kind, val = b13.parse_protocol(protocol)
        pub = b13.published_score(kind, val, eq)
        f0 = res[0]['folds'][0]
        lines += [f'## {protocol}, equalised {eq}', '',
                  f'{f0["n_class"]} transmitters; {len(res[0]["folds"])} fold(s); held out: {f0["held_out"]}; '
                  f'{f0["n_test"]} test packets in fold 0. Authors\' network, printed: '
                  + (f'{100 * pub:.2f} % accuracy.' if pub is not None else 'none.'), '',
                  'One row is one view; a cell is the mean over folds of accuracy / macro-F1, in %.'
                  + (' DEBUG settings: not results.' if any(r['debug'] for r in res) else ''), '']
        models = [m for m in MODELS if any(r['model'] == m for r in res)]
        views = [v for v in all_views() if any(r['view'] == v for r in res)]
        lines += ['| view | features | ' + ' | '.join(models) + ' |', '|---|---|' + '---|' * len(models)]
        for v in views:
            cells = []
            for m in models:
                r = [x for x in res if x['view'] == v and x['model'] == m]
                cells.append(f'{r[0]["mean_accuracy"]:.1f} / {r[0]["mean_macro_f1"]:.1f}' if r else '')
            lines.append(f'| {v} | {FOLD_VIEWS.get(v) or PACKET_VIEWS[v]} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        # error overlap between views, per model, on the test packets of every fold
        for m in models:
            errs = {}
            for v in views:
                ps = sorted(d.glob(f'{v}__{m}.fold*.pred.npy'))
                if ps:
                    errs[v] = np.concatenate([(np.load(p)[1] != np.load(p)[2]) for p in ps])
            vs = [v for v in errs if len(errs[v]) == len(next(iter(errs.values())))]
            if len(vs) < 2:
                continue
            lines += [f'### error overlap, {m}', '',
                      'Cell: of the test packets either view gets wrong, the fraction both get wrong, in %. '
                      'Diagonal: error rate in %.', '',
                      '| | ' + ' | '.join(vs) + ' |', '|---|' + '---|' * len(vs)]
            for a in vs:
                cells = []
                for b in vs:
                    if a == b:
                        cells.append(f'{100 * errs[a].mean():.1f}')
                    else:
                        u = (errs[a] | errs[b]).sum()
                        cells.append(f'{100 * (errs[a] & errs[b]).sum() / max(u, 1):.0f}')
                lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
            lines.append('')
        # CKA between the per-packet views on up to 3,000 test packets of fold 0
        folds, subset = folds_for(protocol, eq)
        tei = folds[0]['test'][0]
        idx = np.sort(np.random.RandomState(b13.SEED).choice(tei, min(3000, len(tei)), replace=False))
        pv = [v for v in views if v in PACKET_VIEWS and view_path(subset, eq, v).exists()]
        if len(pv) >= 2:
            mats = {}
            for v in pv:
                A = np.load(view_path(subset, eq, v))[idx].astype(np.float64)
                mats[v] = (A - A.mean(0)) / (A.std(0) + 1e-12)
            lines += ['### linear CKA between views', '',
                      f'{len(idx)} test packets of fold 0, standardised views. 1 = the same information up to '
                      'a linear map, 0 = unrelated.', '',
                      '| | ' + ' | '.join(pv) + ' |', '|---|' + '---|' * len(pv)]
            for a in pv:
                lines.append(f'| {a} | ' + ' | '.join(f'{linear_cka(mats[a], mats[b]):.2f}' for b in pv) + ' |')
            lines.append('')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'summary.md').write_text('\n'.join(lines))
    log(f'summary written: {RESULTS_DIR / "summary.md"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['packets', 'views', 'embed', 'evaluate', 'summarise'])
    ap.add_argument('--subset', help='views: ManyTx, ManyRx or ManySig')
    ap.add_argument('--equalized', type=int, choices=[0, 1], help='views')
    ap.add_argument('--views', help='comma list')
    ap.add_argument('--protocols', help='comma list of PROTOCOL:EQ, e.g. ntx_150:0,nrx_25:1 (no default)')
    ap.add_argument('--models', help='comma list (no default)')
    ap.add_argument('--folds', help='embed: comma list of fold indices')
    ap.add_argument('--epochs', type=int, default=b13.EPOCHS,
                    help=f'network epoch cap; {b13.EPOCHS} is the protocol, anything else is marked debug')
    a = ap.parse_args()
    if a.stage == 'packets':
        if a.subset is None or a.equalized is None:
            raise SystemExit('--stage packets needs --subset and --equalized')
        b13.packet_table(a.subset, a.equalized)
        log(f'{a.subset} eq{a.equalized}: packet table at {b13.WISIG_DIR / "packets"}')
        return
    if a.stage == 'views':
        if a.subset is None or a.equalized is None:
            raise SystemExit('--stage views needs --subset and --equalized')
        stage_views(a.subset, a.equalized, a.views.split(',') if a.views else list(PACKET_VIEWS))
        return
    if not a.protocols:
        raise SystemExit('--protocols is an open setting with no default, e.g. ntx_150:0,nrx_25:1')
    settings = [parse_setting(s) for s in a.protocols.split(',')]
    if a.stage == 'embed':
        for p, eq in settings:
            stage_embed(p, eq, [int(f) for f in a.folds.split(',')] if a.folds else None, a.epochs)
    elif a.stage == 'evaluate':
        if not a.models or not a.views:
            raise SystemExit('--stage evaluate needs --views and --models (no default)')
        for p, eq in settings:
            for v in a.views.split(','):
                for m in a.models.split(','):
                    if model_applies(v, m):
                        evaluate_one(p, eq, v, m, a.epochs)
    else:
        stage_summarise(settings)


if __name__ == '__main__':
    main()
