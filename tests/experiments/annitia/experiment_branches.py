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
from experiments.annitia.tabpfn import TabPFNSurvivalLandmark, TabPFN25Classifier
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold
from experiments.annitia.discretised_survival import DiscretisedSurvival
from experiments.annitia.synthetic_aug import SyntheticAugmentedSurvival


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
    # Roadmap T1.x / T2.x additions:
    ap.add_argument('--skip-mitra', action='store_true',
                    help='Skip MITRA discretised-classifier survival (T1.1).')
    ap.add_argument('--skip-tabpfn25-disc', action='store_true',
                    help='Skip TabPFN-2.5 discretised-classifier survival (T1.1).')
    ap.add_argument('--skip-tabsurv', action='store_true',
                    help='Skip TabSurv FT-Transformer survival head (T1.3).')
    ap.add_argument('--skip-synthetic-aug', action='store_true',
                    help='Skip synthetic-positive augmentation conditions (T1.4).')
    ap.add_argument('--skip-survtrace', action='store_true',
                    help='Skip SurvTRACE competing-risk branch (T1.5).')
    ap.add_argument('--skip-nhanes-transfer', action='store_true',
                    help='Skip NHANES->ANNITIA transfer recipes a-f (T2.2). '
                         'NOTE: T2.x lives in experiment_external.py; this '
                         'flag is kept for backward-compat but the section '
                         'is no-op here.')
    ap.add_argument('--skip-tabtune', action='store_true',
                    help='Skip TabTune PEFT fine-tune (T1.2).')
    ap.add_argument('--K-disc', type=int, default=10,
                    help='Number of bins for discretised-classifier reductions.')
    ap.add_argument('--n-synth', type=str, default='100',
                    help='Synthetic positives per fold for T1.4. Comma-list '
                         'allowed (e.g. "50,100,200,500" sweeps all four).')
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
                    _run('dsm-K4', lambda: SurvivalDSM(k=4, epochs=80),
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

    # ---------- 5. T1.1 — MITRA discretised classifier (per-bin) ----------
    if not args.skip_mitra:
        print(f'\n=== T1.1 MITRA discretised-survival (K={args.K_disc}) ===')
        try:
            from experiments.annitia.mitra_survival import MitraClassifier
        except ImportError as e:
            print(f'  mitra import failed: {e}; skipping')
            MitraClassifier = None
        if MitraClassifier is not None:
            primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
            views_train = build_views(train, view_names=primary)
            views_val = build_views(val,   view_names=primary)
            X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
            X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
            for endpoint in ('hepatic', 'death'):
                K = args.K_disc if endpoint == 'hepatic' else max(args.K_disc, 20)
                _run(f'mitra-disc-K{K}',
                     lambda K=K: DiscretisedSurvival(
                         classifier_factory=lambda: MitraClassifier(),
                         K=K, strategy='equal_event'),
                     X_tr, X_va, endpoint, primary)

    # ---------- 6. T1.1 — TabPFN-2.5 discretised classifier ----------
    if not args.skip_tabpfn25_disc:
        print(f'\n=== T1.1 TabPFN-2.5 discretised-survival (K={args.K_disc}) ===')
        primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
        views_train = build_views(train, view_names=primary)
        views_val = build_views(val,   view_names=primary)
        X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
        X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
        for endpoint in ('hepatic', 'death'):
            K = args.K_disc if endpoint == 'hepatic' else max(args.K_disc, 20)
            _run(f'tabpfn25-disc-K{K}',
                 lambda K=K: DiscretisedSurvival(
                     classifier_factory=lambda: TabPFN25Classifier(max_features=100),
                     K=K, strategy='equal_event'),
                 X_tr, X_va, endpoint, primary)

    # ---------- 7. T1.3 — TabSurv FT-Transformer ----------
    if not args.skip_tabsurv:
        print('\n=== T1.3 TabSurv (FT-Transformer survival head) ===')
        try:
            from experiments.annitia.tabsurv import TabSurv
            primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
            views_train = build_views(train, view_names=primary)
            views_val = build_views(val,   view_names=primary)
            X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
            X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
            for endpoint in ('hepatic', 'death'):
                K = args.K_disc if endpoint == 'hepatic' else max(args.K_disc, 15)
                _run(f'tabsurv-K{K}',
                     lambda K=K: TabSurv(K=K, epochs=60, batch_size=64,
                                          n_layers=2, d_emb=64),
                     X_tr, X_va, endpoint, primary)
        except ImportError as e:
            print(f'  tabsurv unavailable (torch missing?): {e}')

    # ---------- 8. T1.4 — Synthetic-positive augmentation (hep-focused) ----------
    if not args.skip_synthetic_aug:
        n_synth_list = [int(x) for x in str(args.n_synth).split(',') if x.strip()]
        print(f'\n=== T1.4 Synthetic-positive augmentation '
              f'(hep, n_synth sweep={n_synth_list}) ===')
        primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
        views_train = build_views(train, view_names=primary)
        views_val = build_views(val,   view_names=primary)
        X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
        X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
        # base_ctor: zero-arg callable that returns a fresh survival model.
        base_ctors = (
            ('cox',     lambda: SurvivalCoxPH()),
            ('rsf-500', lambda: SurvivalRSF(n_estimators=500)),
        )
        # Run the no-aug baseline once per base ctor so calibration regression
        # against augmented variants is visible. Tagged as synaug-none so it
        # still lands in cv_summary alongside the augmented runs.
        for base_label, base_ctor in base_ctors:
            _run(f'synaug-none-{base_label}-n0',
                 lambda c=base_ctor: c(),
                 X_tr, X_va, 'hepatic', primary)
        for backend in ('gmm', 'tabddpm', 'tabsyn', 'tabebm'):
            for n_synth in n_synth_list:
                for base_label, base_ctor in base_ctors:
                    _run(f'synaug-{backend}-{base_label}-n{n_synth}',
                         lambda b=backend, c=base_ctor, n=n_synth:
                             SyntheticAugmentedSurvival(
                                 base_ctor=c, backend=b, n_synth=n,
                                 safe_fallback=True),
                         X_tr, X_va, 'hepatic', primary)

        # ---- T1.4 post-pass: Brier OOF + calibration regression vs no-aug ----
        from experiments.annitia.synthetic_aug import brier_score_at_horizon
        y_hep = train['hepatic']['y_struct']
        valid_hep = train['hepatic']['valid_mask']
        # Median event time as the calibration horizon
        ev_t = y_hep['time'][y_hep['event'] & valid_hep & np.isfinite(y_hep['time'])]
        horizon = float(np.median(ev_t)) if ev_t.size else float('nan')
        baselines = {}  # base_label -> brier
        for r in summary['per_condition']['hepatic']:
            cid = r.get('config_id', '')
            if not cid.startswith('synaug-'):
                continue
            oof_p = out / f'oof_hepatic_{cid}.npy'
            if not oof_p.exists():
                continue
            oof = np.load(oof_p)
            r['brier_oof'] = brier_score_at_horizon(
                y_hep, y_hep, oof, horizon)
            r['brier_horizon'] = horizon
            # Tag base_label for regression check
            for bl in ('cox', 'rsf-500'):
                if f'-{bl}-' in cid:
                    r['base_label'] = bl
                    if cid.startswith('synaug-none-'):
                        baselines[bl] = r['brier_oof']
                    break
        # Calibration regression flag: aug Brier > baseline + 10%
        for r in summary['per_condition']['hepatic']:
            if not r.get('config_id', '').startswith('synaug-'):
                continue
            bl = r.get('base_label')
            if bl in baselines and np.isfinite(baselines[bl]) and np.isfinite(r.get('brier_oof', float('nan'))):
                r['brier_regression_pct'] = float(
                    100.0 * (r['brier_oof'] - baselines[bl]) / max(baselines[bl], 1e-9))
                r['calibration_abort'] = bool(r['brier_regression_pct'] > 10.0)
        summary_path.write_text(json.dumps(_convert(summary), indent=2))
        print(f'\n  T1.4 calibration: horizon={horizon:.2f}; '
              f'baselines={baselines}; '
              f'aborted={[r["config_id"] for r in summary["per_condition"]["hepatic"] if r.get("calibration_abort")]}')

    # ---------- 9. T1.5 — SurvTRACE competing risks (own fold loop) ----------
    if not args.skip_survtrace:
        print('\n=== T1.5 SurvTRACE competing-risk joint hep+death ===')
        try:
            from experiments.annitia.survtrace_branch import SurvTRACEBranch
        except ImportError as e:
            print(f'  survtrace unavailable: {e}')
            SurvTRACEBranch = None
        if SurvTRACEBranch is not None:
            primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
            views_train = build_views(train, view_names=primary)
            views_val = build_views(val,   view_names=primary)
            X_tr_full = np.concatenate([views_train[v][0] for v in primary], axis=1)
            X_va_full = np.concatenate([views_val[v][0]   for v in primary], axis=1)
            y_hep_full = train['hepatic']['y_struct']
            y_dea_full = train['death']['y_struct']
            for endpoint in ('hepatic', 'death'):
                cid = f'survtrace-{endpoint}__{"+".join(primary)}'
                oof_path = out / f'oof_{endpoint}_{cid}.npy'
                val_path_npy = out / f'val_{endpoint}_{cid}.npy'
                existing = done[endpoint].get(cid)
                if (existing is not None and oof_path.exists() and val_path_npy.exists()
                        and np.isfinite(existing.get('mean_cindex', float('nan')))):
                    print(f'  survtrace-{endpoint:7s}  CACHED  C={existing["mean_cindex"]:.4f}')
                    continue
                y_self_full = train[endpoint]['y_struct']
                valid_self = train[endpoint]['valid_mask']
                folds = get_cv_folds(train, endpoint=endpoint,
                                     n_folds=args.n_folds, seed=args.seed)
                t0 = time.time()
                oof = init_oof_buffer(n_train); fold_c = []
                try:
                    for k, (tr, va) in enumerate(folds):
                        m = SurvTRACEBranch(cause_of_interest=endpoint, epochs=30)
                        if endpoint == 'hepatic':
                            m.fit(X_tr_full[tr], y_hep_full[tr], y_dea_full[tr])
                        else:
                            m.fit(X_tr_full[tr], y_hep_full[tr], y_dea_full[tr])
                        risk = m.predict_risk(X_tr_full[va])
                        fold_c.append(concordance(y_self_full[va], risk))
                        fill_oof_fold(oof, va, risk)
                    mean_c = float(np.nanmean(fold_c))
                    std_c  = float(np.nanstd(fold_c))
                    vidx = np.where(valid_self)[0]
                    refit = SurvTRACEBranch(cause_of_interest=endpoint, epochs=30)
                    refit.fit(X_tr_full[vidx], y_hep_full[vidx], y_dea_full[vidx])
                    val_risk = np.asarray(refit.predict_risk(X_va_full), dtype=np.float64)
                except ImportError as e:
                    print(f'  survtrace-{endpoint}  SKIP: {str(e)[:80]}')
                    continue
                except Exception as e:
                    print(f'  survtrace-{endpoint}  ERR: {type(e).__name__}: {str(e)[:120]}')
                    continue
                np.save(oof_path, oof); np.save(val_path_npy, val_risk)
                rec = {'config_id': cid, 'label': f'survtrace-{endpoint}',
                       'view_set': primary,
                       'fold_cindex': fold_c, 'mean_cindex': mean_c, 'std_cindex': std_c,
                       'elapsed_sec': time.time() - t0}
                summary['per_condition'][endpoint] = [
                    r for r in summary['per_condition'][endpoint]
                    if r.get('config_id') != cid]
                summary['per_condition'][endpoint].append(rec)
                done[endpoint][cid] = rec
                summary_path.write_text(json.dumps(_convert(summary), indent=2))
                print(f'  survtrace-{endpoint:7s}  C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')

    # ---------- 10. T1.2 — TabTune PEFT fine-tune ----------
    if not args.skip_tabtune:
        print('\n=== T1.2 TabTune PEFT fine-tune (TabPFN-v2.6 + MITRA) ===')
        try:
            from experiments.annitia.tabtune_survival import (
                TabTuneTabPFN, TabTuneMITRA)
            from experiments.annitia.discretised_survival import DiscretisedSurvival
            primary = _read_be_top_view_sets(Path(args.be_dir), 'hepatic', k=1)[0]
            views_train = build_views(train, view_names=primary)
            views_val = build_views(val,   view_names=primary)
            X_tr = np.concatenate([views_train[v][0] for v in primary], axis=1)
            X_va = np.concatenate([views_val[v][0]   for v in primary], axis=1)
            tabtune_models = [
                ('tabtune-tabpfn26-peft', TabTuneTabPFN, 'peft'),
                ('tabtune-mitra-peft',    TabTuneMITRA,  'peft'),
            ]
            for label, cls, strategy in tabtune_models:
                for endpoint in ('hepatic', 'death'):
                    K = args.K_disc if endpoint == 'hepatic' else max(args.K_disc, 20)
                    _run(f'{label}-K{K}',
                         lambda C=cls, s=strategy, K=K: DiscretisedSurvival(
                             classifier_factory=lambda C=C, s=s: C(
                                 strategy=s, epochs=8),
                             K=K, strategy='equal_event'),
                         X_tr, X_va, endpoint, primary)
        except ImportError as e:
            print(f'  tabtune unavailable: {e}')

    # ---------- (T2.x lives in experiment_external.py; nothing to do here) ----------
    if not args.skip_nhanes_transfer:
        print('\n=== T2.x has moved to experiment_external.py — '
              'run that for the external-data layer. ===')

    summary['elapsed_sec'] = time.time() - t_start
    summary_path.write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min')
    print(f'  cv_summary: {summary_path}')


if __name__ == '__main__':
    main()
