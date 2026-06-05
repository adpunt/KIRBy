"""Local generalization-method tests: D.25 (conformal), E.30 (pseudo-label), E.29 (mixup).

Each method produces a val_test submission CSV under tests/results/annitia/full/
and appends a row to method_test_log.csv (method, hep_C, death_C, weighted_LB,
n_param, runtime).

These are *cheap* sanity tests built on top of cached OOF / val artefacts and a
simple Cox PH baseline so they run in minutes. They are *not* full pipeline
re-runs.

Usage:
    python tests/experiments/annitia/method_local_tests.py --method conformal
    python tests/experiments/annitia/method_local_tests.py --method pseudolabel
    python tests/experiments/annitia/method_local_tests.py --method mixup
    python tests/experiments/annitia/method_local_tests.py --method all
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia
from experiments.annitia.survival import concordance
from experiments.annitia.views import _snapshot_view, _static_matrix

FULL_DIR = REPO / 'tests/results/annitia/full'
ANNITIA_RES = REPO / 'tests/results/annitia'
ANCHOR_PATH = FULL_DIR / 'sub_master_meta_avg_aware_weighted_v2.csv'
LOG_PATH = FULL_DIR / 'method_test_log.csv'
VAL_CSV = '/Volumes/seagate/kirby/annitia/val_test.csv'

RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _list_oof_pairs(endpoint: str):
    """Return list of (oof_path, val_path, label) under cd/be/branches/external/hep_boost."""
    pairs = []
    for sub in ('cd', 'be', 'branches', 'external', 'hep_boost'):
        d = ANNITIA_RES / sub
        if not d.is_dir():
            continue
        for oof_p in sorted(d.glob(f'oof_{endpoint}_*.npy')):
            label = oof_p.stem[len(f'oof_{endpoint}_'):]
            val_p = d / f'val_{endpoint}_{label}.npy'
            if val_p.exists():
                pairs.append((oof_p, val_p, f'{sub}/{label}'))
    return pairs


def _rank01(x: np.ndarray) -> np.ndarray:
    """NaN-safe rank to (0,1] then offset to (0,1)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    mask = np.isfinite(x)
    if not mask.any():
        return out
    r = pd.Series(x[mask]).rank(method='average').to_numpy()
    out[mask] = r / (mask.sum() + 1.0)
    return out


def _pseudo_residual(oof_rank: np.ndarray, event: np.ndarray,
                     time: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Residual signal for split-conformal: per-patient |rank(risk) - target|.

    target = 1 for events (high-risk should rank high), 0 for censored.
    Censored patients are noisy targets — that's fine for relative widths.
    Returns NaN where invalid or oof is NaN.
    """
    res = np.full_like(oof_rank, np.nan)
    target = np.where(event == 1, 1.0, 0.0)
    m = valid & np.isfinite(oof_rank)
    res[m] = np.abs(oof_rank[m] - target[m])
    return res


def _save_submission(name: str, hep: np.ndarray, death: np.ndarray):
    anchor = pd.read_csv(ANCHOR_PATH)
    assert len(hep) == len(anchor) == len(death) == 423, \
        f'expected 423 rows, got hep={len(hep)} death={len(death)}'
    out = pd.DataFrame({
        'trustii_id': anchor['trustii_id'].values,
        'risk_hepatic_event': hep,
        'risk_death': death,
    })
    p = FULL_DIR / f'sub_master_method_{name}.csv'
    out.to_csv(p, index=False)
    print(f'  wrote {p}')
    return p


def _append_log(method: str, hep_c: float, death_c: float, lb: float,
                n_param: int, runtime: float):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with LOG_PATH.open('a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['method', 'hep_C', 'death_C', 'weighted_LB',
                        'n_param', 'runtime_sec'])
        w.writerow([method, f'{hep_c:.4f}', f'{death_c:.4f}',
                    f'{lb:.4f}', n_param, f'{runtime:.1f}'])
    print(f'  appended {LOG_PATH}')


def _oof_concordance(oof: np.ndarray, y_struct, valid: np.ndarray) -> float:
    """C-index across patients with finite OOF and valid outcome."""
    m = valid & np.isfinite(oof)
    if m.sum() < 5 or not y_struct['event'][m].any():
        return float('nan')
    return concordance(y_struct[m], oof[m])


def _run_pre_check(p: Path):
    import subprocess
    r = subprocess.run(
        ['python', str(REPO / 'tests/experiments/annitia/pre_submit_check.py'),
         str(p)], capture_output=True, text=True)
    print(r.stdout, end='')
    if r.stderr:
        print('STDERR:', r.stderr, end='')


# ----------------------------------------------------------------------
# Method 1: Conformal wrapper
# ----------------------------------------------------------------------
def method_conformal(train_data, alpha: float = 0.1, top_k: int = 12):
    """Split-conformal calibration: per-base-model bound widths -> inverse-width
    weights for ensembling on val_test.

    Steps:
      1. For each base model OOF: rank-transform.
      2. Hold out 20% of train (calibration set). Compute residuals
         |rank(oof) - target| on that calibration set.
      3. Per-patient (val_test): bound width is the (1-alpha)-quantile of
         residuals — actually a *global* width per model since features are
         not used for adaptive intervals; equivalent to symmetric split-conformal.
      4. Per-model weight = 1 / (width + eps). Final risk = weighted mean of
         val rank predictions.

    For a per-patient adaptive width we use a kNN-on-rank version: for each
    val patient, find the 50 calibration patients with closest predicted rank,
    take the (1-alpha)-quantile of their residuals as the local width.
    """
    print('\n=== METHOD 1: split-conformal wrapper ===')
    n_train = len(train_data['patient_ids'])
    n_val = 423
    rng = np.random.default_rng(7)

    out_preds = {}
    diag = {}
    for endpoint in ('hepatic', 'death'):
        ydata = train_data[endpoint]
        if ydata is None:
            continue
        valid = ydata['valid_mask']
        y_struct = ydata['y_struct']
        event = ydata['event']

        pairs = _list_oof_pairs(endpoint)
        # Filter to top_k by OOF C-index for stability
        scored = []
        for op, vp, lab in pairs:
            try:
                oof = np.load(op)
                if oof.shape[0] != n_train:
                    continue
                c = _oof_concordance(oof, y_struct, valid)
                if not np.isfinite(c):
                    continue
                scored.append((c, op, vp, lab))
            except Exception:
                continue
        scored.sort(key=lambda x: -x[0])
        scored = scored[:top_k]
        print(f'  [{endpoint}] using top-{len(scored)} OOFs')

        # Calibration split: hold out 20% of *valid* patients
        valid_idx = np.where(valid)[0]
        rng.shuffle(valid_idx)
        n_calib = max(50, int(0.2 * valid_idx.size))
        calib_idx = valid_idx[:n_calib]
        calib_mask = np.zeros(n_train, dtype=bool)
        calib_mask[calib_idx] = True

        widths = []
        val_ranks = []
        oof_ranks_sum = np.zeros(n_train)
        oof_w_sum = np.zeros(n_train)
        for c, op, vp, lab in scored:
            oof = np.load(op)
            val = np.load(vp)
            if val.shape[0] != n_val:
                continue
            oof_r = _rank01(oof)
            val_r = _rank01(val)
            res = _pseudo_residual(oof_r, event, ydata['time'], valid)
            cal_res = res[calib_mask]
            cal_res = cal_res[np.isfinite(cal_res)]
            if cal_res.size < 10:
                continue
            w = float(np.quantile(cal_res, 1 - alpha))
            widths.append(w)
            val_ranks.append(val_r)
            inv_w = 1.0 / (w + 1e-3)
            mfin = np.isfinite(oof_r)
            oof_ranks_sum[mfin] += oof_r[mfin] * inv_w
            oof_w_sum[mfin] += inv_w

        if not widths:
            print(f'  [{endpoint}] no usable models')
            out_preds[endpoint] = np.full(n_val, 0.5)
            continue

        widths = np.asarray(widths)
        inv = 1.0 / (widths + 1e-3)
        weights = inv / inv.sum()
        val_pred = np.zeros(n_val)
        for w_i, vr in zip(weights, val_ranks):
            val_pred += w_i * vr
        out_preds[endpoint] = val_pred

        # OOF C of the conformal-weighted ensemble (training side)
        oof_pred = np.where(oof_w_sum > 0, oof_ranks_sum / np.maximum(oof_w_sum, 1e-9),
                            np.nan)
        oof_c = _oof_concordance(oof_pred, y_struct, valid)
        diag[endpoint] = {
            'n_models': len(widths),
            'mean_width': float(widths.mean()),
            'oof_C': oof_c,
        }
        print(f'  [{endpoint}] n_models={len(widths)} mean_width={widths.mean():.3f} '
              f'OOF C={oof_c:.4f}')

    return out_preds, diag


# ----------------------------------------------------------------------
# Method 2: Pseudo-labelling
# ----------------------------------------------------------------------
def method_pseudolabel(train_data, weight: float = 0.1):
    """Use anchor val_test predictions as soft labels; refit a small Cox PH on
    train + pseudo-labelled val_test (downweighted).

    "Soft labels" for survival are tricky — true (event, time) targets are not
    available. We synthesise pseudo-outcomes: for each val_test patient i,
    treat the anchor's risk_hepatic_event[i] (in [0,1]) as the *probability*
    they had a hepatic event; sample event ~ Bernoulli(p) and time = mean
    train follow-up. Same for death. We then fit a single Cox PH on the
    weighted union and predict val_test risk as the linear predictor.
    """
    print('\n=== METHOD 2: pseudo-labelling ===')
    from lifelines import CoxPHFitter

    anchor = pd.read_csv(ANCHOR_PATH)
    val_data = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                            filename=Path(VAL_CSV).name, has_outcomes=False)

    Xt_full, _ = _snapshot_view(train_data)
    Xv_full, _ = _snapshot_view(val_data)
    # Standardise
    mu = np.nanmean(Xt_full, axis=0); sd = np.nanstd(Xt_full, axis=0); sd[sd == 0] = 1
    Xt_full = (np.where(np.isfinite(Xt_full), Xt_full, mu) - mu) / sd
    Xv_full = (np.where(np.isfinite(Xv_full), Xv_full, mu) - mu) / sd
    # Drop near-zero variance and missingness indicator columns to avoid singularity
    var = Xt_full.var(axis=0)
    keep_cols = (var > 0.05) & (var < 50)
    Xt_full = Xt_full[:, keep_cols]
    Xv_full = Xv_full[:, keep_cols]
    # PCA to a small set of orthogonal features (fast, robust)
    from numpy.linalg import svd
    n_pcs_target = 8
    U, S, Vt = svd(Xt_full, full_matrices=False)
    n_pcs = min(n_pcs_target, S.size)
    PC = Vt[:n_pcs].T
    Xt = Xt_full @ PC
    Xv = Xv_full @ PC
    # Re-standardise PCs (zero mean already)
    sd2 = Xt.std(axis=0); sd2[sd2 == 0] = 1
    Xt = Xt / sd2
    Xv = Xv / sd2

    out_preds = {}
    diag = {}
    rng = np.random.default_rng(11)
    for endpoint, anchor_col in (('hepatic', 'risk_hepatic_event'),
                                 ('death', 'risk_death')):
        ydata = train_data[endpoint]
        valid = ydata['valid_mask']
        event = ydata['event'][valid]
        time_t = ydata['time'][valid]
        Xtv = Xt[valid]

        # Pseudo-outcome for val_test: Bernoulli(p) on anchor risk; censored time = median train follow-up
        p_event = np.clip(anchor[anchor_col].to_numpy(), 0.001, 0.999)
        # Calibrate p_event to roughly match training event rate
        p_event = p_event * (event.mean() / p_event.mean())
        p_event = np.clip(p_event, 0.001, 0.999)
        pseudo_event = (rng.uniform(0, 1, size=p_event.size) < p_event).astype(int)
        pseudo_time = np.full(p_event.size, np.median(time_t))

        Xtv_s = Xtv
        Xv_s = Xv
        cols = [f'pc{i}' for i in range(n_pcs)]

        df_t = pd.DataFrame(Xtv_s, columns=cols)
        df_t['T'] = time_t
        df_t['E'] = event
        df_t['w'] = 1.0
        df_p = pd.DataFrame(Xv_s, columns=cols)
        df_p['T'] = pseudo_time
        df_p['E'] = pseudo_event
        df_p['w'] = weight
        df = pd.concat([df_t, df_p], ignore_index=True)

        cph = CoxPHFitter(penalizer=0.05)
        try:
            cph.fit(df, duration_col='T', event_col='E', weights_col='w',
                    robust=False, show_progress=False)
            val_pred = cph.predict_partial_hazard(pd.DataFrame(Xv_s, columns=cols)).to_numpy()
            oof_pred_valid = cph.predict_partial_hazard(pd.DataFrame(Xtv_s, columns=cols)).to_numpy()
        except Exception as e:
            print(f'  [{endpoint}] cox failed: {e} — falling back to anchor')
            val_pred = anchor[anchor_col].to_numpy()
            oof_pred_valid = np.full(valid.sum(), np.nan)

        # Build OOF buffer (in-sample, NOT out-of-fold — caveat below)
        oof_buf = np.full(len(train_data['patient_ids']), np.nan)
        oof_buf[valid] = oof_pred_valid
        oof_c = _oof_concordance(oof_buf, ydata['y_struct'], valid)
        out_preds[endpoint] = val_pred
        diag[endpoint] = {
            'n_features': n_pcs,
            'pseudo_event_rate': float(pseudo_event.mean()),
            'in_sample_C': oof_c,  # NOT proper OOF — see caveat
        }
        print(f'  [{endpoint}] in-sample C={oof_c:.4f} '
              f'pseudo_event_rate={pseudo_event.mean():.3f}')

    return out_preds, diag


# ----------------------------------------------------------------------
# Method 3: Mixup
# ----------------------------------------------------------------------
def method_mixup(train_data, n_mix: int = 200, beta_a: float = 0.4,
                 n_folds: int = 5):
    """Add 200 mixup synthetic patients, refit Cox PH, predict val_test.

    For survival mixup, we pair patients (i, j) and mix features X with
    lambda~Beta(0.4,0.4). Outcome rule: event = 1 if either had event; time
    = lambda*time_i + (1-lambda)*time_j. (Naive but cheap.)
    """
    print('\n=== METHOD 3: mixup augmentation ===')
    from lifelines import CoxPHFitter
    from sklearn.model_selection import KFold

    Xt_full, _ = _snapshot_view(train_data)
    val_data = load_annitia(data_dir=str(Path(VAL_CSV).parent),
                            filename=Path(VAL_CSV).name, has_outcomes=False)
    Xv_full, _ = _snapshot_view(val_data)
    mu = np.nanmean(Xt_full, axis=0); sd = np.nanstd(Xt_full, axis=0); sd[sd == 0] = 1
    Xt_full = (np.where(np.isfinite(Xt_full), Xt_full, mu) - mu) / sd
    Xv_full = (np.where(np.isfinite(Xv_full), Xv_full, mu) - mu) / sd
    var = Xt_full.var(axis=0); keep = (var > 0.05) & (var < 50)
    Xt_full = Xt_full[:, keep]; Xv_full = Xv_full[:, keep]
    from numpy.linalg import svd
    U, S, Vt = svd(Xt_full, full_matrices=False)
    n_pcs = min(8, S.size)
    PC = Vt[:n_pcs].T
    Xt = Xt_full @ PC; Xv = Xv_full @ PC
    sd2 = Xt.std(axis=0); sd2[sd2 == 0] = 1
    Xt = Xt / sd2; Xv = Xv / sd2
    cols = [f'pc{i}' for i in range(n_pcs)]

    rng = np.random.default_rng(23)
    out_preds = {}
    diag = {}
    for endpoint in ('hepatic', 'death'):
        ydata = train_data[endpoint]
        valid = ydata['valid_mask']
        event = ydata['event'][valid].astype(int)
        time_t = ydata['time'][valid]
        Xtv = Xt[valid]
        idx_valid = np.where(valid)[0]

        # OOF via 5-fold (so we honestly score the mixup-augmented model)
        oof_buf = np.full(len(train_data['patient_ids']), np.nan)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold_i, (tr, va) in enumerate(kf.split(Xtv)):
            X_tr, X_va = Xtv[tr], Xtv[va]
            e_tr, t_tr = event[tr], time_t[tr]
            # generate mixup
            n_tr = X_tr.shape[0]
            i_idx = rng.integers(0, n_tr, size=n_mix)
            j_idx = rng.integers(0, n_tr, size=n_mix)
            lam = rng.beta(beta_a, beta_a, size=n_mix)
            X_mix = lam[:, None] * X_tr[i_idx] + (1 - lam[:, None]) * X_tr[j_idx]
            t_mix = lam * t_tr[i_idx] + (1 - lam) * t_tr[j_idx]
            e_mix = ((e_tr[i_idx] == 1) | (e_tr[j_idx] == 1)).astype(int)

            X_aug = np.vstack([X_tr, X_mix])
            t_aug = np.concatenate([t_tr, t_mix])
            e_aug = np.concatenate([e_tr, e_mix])
            df_aug = pd.DataFrame(X_aug, columns=cols)
            df_aug['T'] = t_aug; df_aug['E'] = e_aug
            cph = CoxPHFitter(penalizer=0.05)
            try:
                cph.fit(df_aug, duration_col='T', event_col='E',
                        robust=False, show_progress=False)
                pred = cph.predict_partial_hazard(
                    pd.DataFrame(X_va, columns=cols)).to_numpy()
            except Exception as e:
                print(f'  [{endpoint}] fold{fold_i} fit failed: {e}')
                pred = np.zeros(X_va.shape[0])
            oof_buf[idx_valid[va]] = pred

        # Refit on full + mixup, predict val_test
        n_tr = Xtv.shape[0]
        i_idx = rng.integers(0, n_tr, size=n_mix)
        j_idx = rng.integers(0, n_tr, size=n_mix)
        lam = rng.beta(beta_a, beta_a, size=n_mix)
        X_mix = lam[:, None] * Xtv[i_idx] + (1 - lam[:, None]) * Xtv[j_idx]
        t_mix = lam * time_t[i_idx] + (1 - lam) * time_t[j_idx]
        e_mix = ((event[i_idx] == 1) | (event[j_idx] == 1)).astype(int)
        X_aug = np.vstack([Xtv, X_mix])
        t_aug = np.concatenate([time_t, t_mix])
        e_aug = np.concatenate([event, e_mix])
        df_aug = pd.DataFrame(X_aug, columns=cols)
        df_aug['T'] = t_aug; df_aug['E'] = e_aug
        cph = CoxPHFitter(penalizer=0.05)
        try:
            cph.fit(df_aug, duration_col='T', event_col='E', show_progress=False)
            val_pred = cph.predict_partial_hazard(
                pd.DataFrame(Xv, columns=cols)).to_numpy()
        except Exception as e:
            print(f'  [{endpoint}] full fit failed: {e}')
            val_pred = np.zeros(423)

        oof_c = _oof_concordance(oof_buf, ydata['y_struct'], valid)
        out_preds[endpoint] = val_pred
        diag[endpoint] = {
            'n_mix': n_mix,
            'beta_a': beta_a,
            'oof_C': oof_c,
        }
        print(f'  [{endpoint}] OOF C={oof_c:.4f} (5-fold w/ mixup)')

    return out_preds, diag


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run_method(name: str, train_data):
    t0 = time.time()
    if name == 'conformal':
        preds, diag = method_conformal(train_data)
        n_param = sum(d.get('n_models', 0) for d in diag.values())
    elif name == 'pseudolabel':
        preds, diag = method_pseudolabel(train_data)
        n_param = sum(d.get('n_features', 0) for d in diag.values())
    elif name == 'mixup':
        preds, diag = method_mixup(train_data)
        n_param = 200
    else:
        raise ValueError(f'unknown method {name}')
    runtime = time.time() - t0

    hep = preds.get('hepatic', np.full(423, 0.5))
    death = preds.get('death', np.full(423, 0.5))
    p = _save_submission(name, hep, death)

    hep_c = diag.get('hepatic', {}).get('oof_C',
                diag.get('hepatic', {}).get('in_sample_C', float('nan')))
    death_c = diag.get('death', {}).get('oof_C',
                  diag.get('death', {}).get('in_sample_C', float('nan')))
    if np.isfinite(hep_c) and np.isfinite(death_c):
        lb = 0.7 * hep_c + 0.3 * death_c
    else:
        lb = float('nan')
    _append_log(name, hep_c, death_c, lb, n_param, runtime)
    _run_pre_check(p)
    return diag, hep_c, death_c, lb, runtime


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['conformal', 'pseudolabel', 'mixup', 'all'],
                    default='all')
    args = ap.parse_args()

    train_data = load_annitia()
    methods = ['conformal', 'pseudolabel', 'mixup'] if args.method == 'all' else [args.method]
    for m in methods:
        run_method(m, train_data)
