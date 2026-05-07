#!/usr/bin/env python3
"""ANNITIA feat-branch local runner — DSM, NHANES-pretrained MLP, TSFM views,
TabPFN landmark.

Estimated runtime LOCALLY (from user's table, with aggressive trimming):
    DSM + DSM-CR              ~20-40 min
    NHANES-pretrained MLP     ~30-50 min  (first-time NHANES download)
    TSFM views (FFT fallback) ~10 min
    TabPFN landmark           ~30-60 min  (small grid, CPU-painful)
    Total                     ~1.5-3 h

What runs (5-fold CV, both endpoints, RESUME-friendly):
    1. DSM + DSM-CR on top BE view-sets (auto-detected from BE cv_summary).
    2. NHANES-pretrained MLP on `nhanes_proj` view (calls pretrain_nhanes.py
       upfront if the encoder.pt isn't on disk yet).
    3. The B/E algos (cox, coxnet, rsf, gbsa, lgbm_aft, ngboost) on the
       `tsfm` view (FFT fallback unless TimesFM/Chronos installed).
    4. TabPFN-landmark with horizon=median-event-time on the primary view-set
       and on the `tsfm` view, hepatic + death.

Outputs:
    tests/results/annitia/branches/
        cv_summary.json
        oof_<endpoint>_<config_id>.npy
        val_<endpoint>_<config_id>.npy
        nhanes_encoder.pt           (if pretrain ran here)

Usage:
    python tests/experiments/annitia/experiment_branches.py
    python tests/experiments/annitia/experiment_branches.py --skip-tabpfn
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds
from experiments.annitia.views import build_views
from experiments.annitia.survival import (
    SurvivalCoxPH, SurvivalCoxnet, SurvivalRSF, SurvivalGBSA,
    SurvivalLightGBMAft, SurvivalNGBoost, SURVIVAL_MODELS, concordance)
from experiments.annitia.tabpfn import TabPFNSurvivalLandmark
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold


def _convert(o):
    if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


class _AutoHorizonLandmark:
    def __init__(self, base_cls, **kwargs):
        self.base_cls = base_cls; self.kwargs = kwargs; self._inner = None
    def fit(self, X, y):
        time = np.asarray(y['time'], dtype=np.float64)
        event = np.asarray(y['event'], dtype=bool)
        ev = time[event & np.isfinite(time)]
        if ev.size == 0: ev = time[np.isfinite(time)]
        h = float(np.median(ev))
        self._inner = self.base_cls(horizon=h, **self.kwargs)
        self._inner.fit(X, y)
        return self
    def predict_risk(self, X): return self._inner.predict_risk(X)


def _read_be_top_view_sets(be_dir: Path, endpoint: str, k: int = 3):
    cv_path = be_dir / 'cv_summary.json'
    if not cv_path.exists():
        return [['snapshot','clinical_pro','fpca','gp_smooth'],
                ['snapshot','clinical_pro','fpca'],
                ['snapshot','clinical_pro','meld_na','visit_density']]
    sm = json.loads(cv_path.read_text())
    recs = [r for r in sm.get('per_condition', {}).get(endpoint, [])
            if np.isfinite(r.get('mean_cindex', float('nan')))]
    recs.sort(key=lambda r: -r['mean_cindex'])
    seen, out = set(), []
    for r in recs:
        vs = tuple(r['views'])
        if vs not in seen:
            seen.add(vs); out.append(list(vs))
        if len(out) >= k: break
    return out


def _ensure_nhanes_encoder(out_dir: Path):
    """Train the NHANES encoder if it's not already on disk. ~30 min CPU."""
    encoder_path = out_dir / 'nhanes_encoder.pt'
    if encoder_path.exists():
        print(f'  NHANES encoder cached: {encoder_path}')
        return str(encoder_path)
    print(f'  NHANES encoder MISSING — pretraining now (~30 min)')
    cmd = ['python', '-u',
           str(_REPO / 'tests' / 'experiments' / 'annitia' / 'pretrain_nhanes.py'),
           '--restrict', 'fib4_high',
           '--output', str(encoder_path)]
    print(f'  >>> {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0 or not encoder_path.exists():
        print(f'  WARNING: pretrain_nhanes failed (rc={r.returncode}); pretrained_mlp will be skipped')
        return None
    os.environ['ANNITIA_NHANES_ENCODER'] = str(encoder_path)
    return str(encoder_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--be-dir', default='tests/results/annitia/be')
    ap.add_argument('--output-dir', default='tests/results/annitia/branches')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-dsm', action='store_true')
    ap.add_argument('--skip-nhanes', action='store_true')
    ap.add_argument('--skip-tsfm', action='store_true')
    ap.add_argument('--skip-tabpfn', action='store_true')
    ap.add_argument('--tsfm-backend', default='auto',
                    help='auto|timesfm|chronos|fft (auto tries timesfm, falls back).')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('ANNITIA_CACHE_DIR', '/tmp/annitia_branches_cache')
    t_start = time.time()

    print('Loading train.csv + val_test.csv...')
    train = load_annitia()
    n_train = len(train['patient_ids'])
    val_path = Path(args.validation_csv)
    if not val_path.exists():
        sys.exit(f'val_test missing: {val_path}')
    val = load_annitia(data_dir=str(val_path.parent), filename=val_path.name,
                       has_outcomes=False)
    n_val = len(val['patient_ids'])

    summary_path = out / 'cv_summary.json'
    summary = {'per_condition': {'hepatic': [], 'death': []}, 'n_folds': args.n_folds}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            for ep in ('hepatic', 'death'):
                summary['per_condition'][ep] = existing.get('per_condition', {}).get(ep, [])
            n_resume = sum(len(summary['per_condition'][e]) for e in ('hepatic','death'))
            if n_resume:
                print(f'  RESUME: loaded {n_resume} prior conditions')
        except Exception as e:
            print(f'  RESUME: failed ({e}); starting fresh')

    done = {ep: {r['config_id']: r for r in summary['per_condition'][ep]
                 if 'config_id' in r}
            for ep in ('hepatic', 'death')}

    def _run(label, ctor, X_tr, X_va, endpoint, view_set, needs_unlab=False):
        cid = f'{label}__{"+".join(view_set)}'
        oof_path = out / f'oof_{endpoint}_{cid}.npy'
        val_path_npy = out / f'val_{endpoint}_{cid}.npy'
        existing = done[endpoint].get(cid)
        if (existing is not None and oof_path.exists() and val_path_npy.exists()
                and np.isfinite(existing.get('mean_cindex', float('nan')))):
            print(f'  {label:30s} {endpoint:7s}  CACHED  C={existing["mean_cindex"]:.4f}')
            return existing
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        folds = get_cv_folds(train, endpoint=endpoint,
                             n_folds=args.n_folds, seed=args.seed)
        t0 = time.time()
        oof = init_oof_buffer(n_train); fold_c = []
        try:
            for k, (tr, va) in enumerate(folds):
                est = ctor()
                if needs_unlab:
                    est.fit(X_tr[tr], y[tr], X_unlab=X_tr[va])
                else:
                    est.fit(X_tr[tr], y[tr])
                risk = est.predict_risk(X_tr[va])
                fold_c.append(concordance(y[va], risk))
                fill_oof_fold(oof, va, risk)
            mean_c = float(np.nanmean(fold_c))
            std_c  = float(np.nanstd(fold_c))
            valid_idx = np.where(valid)[0]
            refit = ctor()
            if needs_unlab:
                refit.fit(X_tr[valid_idx], y[valid_idx], X_unlab=X_va)
            else:
                refit.fit(X_tr[valid_idx], y[valid_idx])
            val_risk = np.asarray(refit.predict_risk(X_va), dtype=np.float64)
        except ImportError as e:
            print(f'  {label:30s} {endpoint:7s}  SKIP: {str(e)[:60]}')
            return None
        except Exception as e:
            print(f'  {label:30s} {endpoint:7s}  ERR: {type(e).__name__}: {str(e)[:120]}')
            return None
        np.save(oof_path, oof)
        np.save(val_path_npy, val_risk)
        rec = {'config_id': cid, 'label': label, 'view_set': view_set,
               'fold_cindex': fold_c, 'mean_cindex': mean_c, 'std_cindex': std_c,
               'elapsed_sec': time.time() - t0}
        summary['per_condition'][endpoint] = [
            r for r in summary['per_condition'][endpoint]
            if r.get('config_id') != cid
        ]
        summary['per_condition'][endpoint].append(rec)
        done[endpoint][cid] = rec
        summary_path.write_text(json.dumps(_convert(summary), indent=2))
        print(f'  {label:30s} {endpoint:7s} {"+".join(view_set)[:32]:32s} '
              f'C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')
        return rec

    # ---------- 1. DSM + DSM-CR on top hep/death view-sets ----------
    if not args.skip_dsm:
        print('\n=== DSM + DSM-CR ===')
        if 'dsm' not in SURVIVAL_MODELS:
            print('  dsm not registered (torch missing?); skipping')
        else:
            from experiments.annitia.survival import SurvivalDSM
            for endpoint in ('hepatic', 'death'):
                top_sets = _read_be_top_view_sets(Path(args.be_dir), endpoint, k=3)
                for vs in top_sets:
                    views_train = build_views(train, view_names=vs)
                    views_val = build_views(val,   view_names=vs)
                    X_tr = np.concatenate([views_train[v][0] for v in vs], axis=1)
                    X_va = np.concatenate([views_val[v][0]   for v in vs], axis=1)
                    _run('dsm-K4', lambda: SurvivalDSM(n_components=4, epochs=80),
                         X_tr, X_va, endpoint, vs)

    # ---------- 2. NHANES-pretrained MLP on nhanes_proj view ----------
    if not args.skip_nhanes:
        print('\n=== NHANES pretrained MLP ===')
        encoder_path = _ensure_nhanes_encoder(out)
        if encoder_path is None:
            print('  encoder missing; skipping pretrained_mlp')
        else:
            os.environ['ANNITIA_NHANES_ENCODER'] = encoder_path
            try:
                from experiments.annitia.survival import SurvivalPretrainedMLP
            except ImportError:
                SurvivalPretrainedMLP = None
            if SurvivalPretrainedMLP is None:
                print('  pretrained_mlp not registered; skipping')
            else:
                vs = ['nhanes_proj']
                views_train = build_views(train, view_names=vs)
                views_val = build_views(val,   view_names=vs)
                X_tr = views_train['nhanes_proj'][0]
                X_va = views_val['nhanes_proj'][0]
                for endpoint in ('hepatic', 'death'):
                    _run('pretrained-mlp', lambda: SurvivalPretrainedMLP(
                            encoder_path=encoder_path),
                         X_tr, X_va, endpoint, vs)

    # ---------- 3. B/E algos × tsfm view (FFT fallback by default) ----------
    if not args.skip_tsfm:
        print(f'\n=== TSFM view (backend={args.tsfm_backend}) × algos ===')
        os.environ.setdefault('ANNITIA_TSFM_BACKEND', args.tsfm_backend)
        try:
            views_train = build_views(train, view_names=['tsfm'])
            views_val = build_views(val,   view_names=['tsfm'])
            X_tr_tsfm = views_train['tsfm'][0]
            X_va_tsfm = views_val['tsfm'][0]
            print(f'  tsfm view: dim={X_tr_tsfm.shape[1]}')
            for mname in ('cox', 'coxnet', 'rsf', 'gbsa', 'lgbm_aft', 'ngboost'):
                cls = SURVIVAL_MODELS[mname]
                for endpoint in ('hepatic', 'death'):
                    _run(f'tsfm-{mname}', cls, X_tr_tsfm, X_va_tsfm, endpoint, ['tsfm'])
        except Exception as e:
            print(f'  tsfm view build failed: {type(e).__name__}: {e}')

    # ---------- 4. TabPFN landmark (small grid, capped) ----------
    if not args.skip_tabpfn:
        print('\n=== TabPFN landmark (small grid) ===')
        # Single primary view-set (top hep) — keep grid tight.
        primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
        views_train = build_views(train, view_names=primary)
        views_val = build_views(val,   view_names=primary)
        X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
        X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
        for endpoint in ('hepatic', 'death'):
            _run('tabpfn-landmark',
                 lambda: _AutoHorizonLandmark(TabPFNSurvivalLandmark),
                 X_tr, X_va, endpoint, primary)

    summary['elapsed_sec'] = time.time() - t_start
    summary_path.write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min')
    print(f'  cv_summary: {summary_path}')


if __name__ == '__main__':
    main()
