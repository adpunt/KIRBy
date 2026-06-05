#!/usr/bin/env python3
"""ANNITIA synthetic-augmentation grid runner.

For each (mechanism × protocol × base_model × endpoint) combination:
    1. Build/load synthetic cohort.
    2. Project synthetic + real to the chosen view (default: snapshot).
    3. Run 5-fold CV on real, with synthetic injected per protocol at fit time.
    4. Refit on full real (+ synth) and predict val_test.
    5. Save:
         tests/results/annitia/branches/oof_<endpoint>_synth_<cid>.npy
         tests/results/annitia/branches/val_<endpoint>_synth_<cid>.npy

Resumable: skips conditions whose OOF file already exists.

Reports per-condition mean OOF C-index to stdout.
Writes summary JSON to tests/results/annitia/synth/grid_summary.json.

Usage:
    python tests/experiments/annitia/synth/run_grid.py
    python tests/experiments/annitia/synth/run_grid.py --quick   # smaller grid
    python tests/experiments/annitia/synth/run_grid.py --models rsf,mlp_dh --protocols P1,P2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds  # noqa: E402
from experiments.annitia.views import build_views  # noqa: E402
from experiments.annitia.survival import (  # noqa: E402
    SurvivalCoxPH, SurvivalRSF, SurvivalGBSA, concordance,
)
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold  # noqa: E402
from experiments.annitia.synth.masld_simulator import simulate_cohort  # noqa: E402
from experiments.annitia.synth.outcome_conditioned import make_outcome_conditional_cohort  # noqa: E402
from experiments.annitia.synth.counterfactual import make_counterfactual_cohort  # noqa: E402
from experiments.annitia.synth.mlp_hazard import MLPDiscreteHazard  # noqa: E402
from experiments.annitia.synth.protocols import (  # noqa: E402
    P1_oversample, P2_pretrain_finetune, P3_mixed_batch, P4_self_distill,
)


SYNTH_DIR = REPO / 'tests/results/annitia/synth'
BRANCHES_DIR = REPO / 'tests/results/annitia/branches'

VIEW_NAMES = ('snapshot', 'summary', 'slope')

PROTOCOL_FNS = {
    'P1': P1_oversample,
    'P2': P2_pretrain_finetune,
    'P3': P3_mixed_batch,
    'P4': P4_self_distill,
}


def _make_base_ctor(name: str) -> Callable[[], object]:
    if name == 'rsf':
        return lambda: SurvivalRSF(n_estimators=120, min_samples_leaf=20,
                                   n_jobs=-1)
    if name == 'rsf300':
        return lambda: SurvivalRSF(n_estimators=300, min_samples_leaf=20,
                                   n_jobs=-1)
    if name == 'gbsa':
        return lambda: SurvivalGBSA(n_estimators=150, learning_rate=0.05)
    if name == 'cox':
        return lambda: SurvivalCoxPH(alpha=1.0)
    if name == 'mlp_dh':
        return lambda: MLPDiscreteHazard(hidden=(128, 64), epochs=60,
                                         finetune_epochs=40, n_bins=10,
                                         dropout=0.3, batch_size=128,
                                         verbose=False)
    raise ValueError(f'unknown base model: {name}')


def _build_or_load_synth(mech: str, real_data, n: int, seed: int) -> Optional[pd.DataFrame]:
    """Cache synth cohorts on disk so we don't regenerate per condition."""
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    if mech == 'M1':
        path = SYNTH_DIR / f'M1_n{n}_s{seed}.csv'
        if path.exists():
            return pd.read_csv(path)
        df = simulate_cohort(n=n, seed=seed)
        df.to_csv(path, index=False); return df
    if mech == 'M2':
        path = SYNTH_DIR / f'M2_n{n}_s{seed}.csv'
        if path.exists():
            return pd.read_csv(path)
        df = make_outcome_conditional_cohort(real_data,
                                              n_event=max(200, n // 4),
                                              n_censored=max(500, 3 * n // 4),
                                              seed=seed)
        df.to_csv(path, index=False); return df
    if mech == 'M5':
        path = SYNTH_DIR / f'M5_k4_s{seed}.csv'
        if path.exists():
            return pd.read_csv(path)
        df = make_counterfactual_cohort(real_data, k_per_patient=4, seed=seed)
        df.to_csv(path, index=False); return df
    if mech == 'M1+M5':
        df1 = _build_or_load_synth('M1', real_data, n, seed)
        df5 = _build_or_load_synth('M5', real_data, n, seed)
        df = pd.concat([df1, df5], ignore_index=True, sort=False)
        return df
    raise ValueError(f'unknown mechanism: {mech}')


def _project_synth_to_view(synth_df: pd.DataFrame, view_names: List[str]
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_synth, time_synth, event_synth, valid_synth)."""
    # Save to a temp CSV and re-load via load_annitia + build_views
    tmp_path = SYNTH_DIR / f'_tmp_synth_{os.getpid()}.csv'
    synth_df.to_csv(tmp_path, index=False)
    try:
        sd = load_annitia(data_dir=str(SYNTH_DIR), filename=tmp_path.name)
    finally:
        try: tmp_path.unlink()
        except Exception: pass
    views = build_views(sd, view_names=list(view_names))
    X = np.concatenate([views[v][0] for v in view_names], axis=1)
    return X, sd['hepatic']['time'], sd['hepatic']['event'], sd['hepatic']['valid_mask']


def _project_synth_with_endpoint(synth_df: pd.DataFrame, view_names: List[str],
                                  endpoint: str
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tmp_path = SYNTH_DIR / f'_tmp_synth_{os.getpid()}.csv'
    synth_df.to_csv(tmp_path, index=False)
    try:
        sd = load_annitia(data_dir=str(SYNTH_DIR), filename=tmp_path.name)
    finally:
        try: tmp_path.unlink()
        except Exception: pass
    views = build_views(sd, view_names=list(view_names))
    X = np.concatenate([views[v][0] for v in view_names], axis=1)
    ep = sd[endpoint]
    if ep is None:
        return X, np.zeros(X.shape[0]), np.zeros(X.shape[0], bool), np.zeros(X.shape[0], bool)
    return X, ep['time'], ep['event'], ep['valid_mask']


def _build_y_struct(time, event, valid):
    out = np.zeros(len(event), dtype=[('event', '?'), ('time', '<f8')])
    out['event'] = np.asarray(event).astype(bool)
    out['time'] = np.where(valid, time, np.nan)
    return out


def run_condition(real_data, real_val, synth_df, mechanism: str, model_name: str,
                  protocol: str, endpoint: str, view_names: List[str],
                  out_branches: Path, out_synth_dir: Path,
                  seed: int = 42, n_folds: int = 5,
                  positives_only_p1: bool = True) -> Dict:
    cid = f'synth_{mechanism}_{model_name}_{protocol}__{"+".join(view_names)}'
    oof_path = out_branches / f'oof_{endpoint}_{cid}.npy'
    val_path = out_branches / f'val_{endpoint}_{cid}.npy'
    rec = {'cid': cid, 'mechanism': mechanism, 'model': model_name,
           'protocol': protocol, 'endpoint': endpoint,
           'view_names': view_names}
    if oof_path.exists() and val_path.exists():
        # Compute c-index on stored OOF for reporting
        oof = np.load(oof_path)
        ep = real_data[endpoint]
        valid_mask = ep['valid_mask']
        finite_oof = np.isfinite(oof) & valid_mask
        if finite_oof.sum() > 10:
            mean_c = concordance(ep['y_struct'][finite_oof], oof[finite_oof])
        else:
            mean_c = float('nan')
        rec.update({'cached': True, 'mean_cindex': float(mean_c)})
        return rec

    # Build real X
    real_views = build_views(real_data, view_names=list(view_names))
    val_views = build_views(real_val, view_names=list(view_names))
    X_real = np.concatenate([real_views[v][0] for v in view_names], axis=1)
    X_val = np.concatenate([val_views[v][0] for v in view_names], axis=1)
    n_real = X_real.shape[0]
    ep = real_data[endpoint]
    y_real = ep['y_struct']
    valid_mask = ep['valid_mask']

    # Build synth X with endpoint-aware y
    X_syn, t_syn, e_syn, v_syn = _project_synth_with_endpoint(
        synth_df, view_names, endpoint)
    y_syn = _build_y_struct(t_syn, e_syn, v_syn)

    folds = get_cv_folds(real_data, endpoint=endpoint, n_folds=n_folds, seed=seed)

    base_ctor = _make_base_ctor(model_name)
    proto_fn = PROTOCOL_FNS[protocol]

    oof = init_oof_buffer(n_real)
    fold_c = []
    t0 = time.time()
    try:
        for k, (tr, va) in enumerate(folds):
            try:
                est = proto_fn(base_ctor, X_real[tr], y_real[tr], X_syn, y_syn)
                risk = est.predict_risk(X_real[va])
                if not np.all(np.isfinite(risk)):
                    risk = np.nan_to_num(risk, nan=np.nanmedian(risk))
                fill_oof_fold(oof, va, risk)
                fold_c.append(concordance(y_real[va], risk))
            except Exception as e:
                # mark fold as nan; continue
                fold_c.append(float('nan'))
                rec.setdefault('fold_errors', []).append(f'fold{k}: {type(e).__name__}: {str(e)[:120]}')
        mean_c = float(np.nanmean(fold_c)) if fold_c else float('nan')
        # Refit on all valid real
        valid_idx = np.where(valid_mask)[0]
        refit = proto_fn(base_ctor, X_real[valid_idx], y_real[valid_idx], X_syn, y_syn)
        val_risk = np.asarray(refit.predict_risk(X_val), dtype=np.float64)
        if not np.all(np.isfinite(val_risk)):
            val_risk = np.nan_to_num(val_risk, nan=np.nanmedian(val_risk))
    except Exception as e:
        rec.update({'error': f'{type(e).__name__}: {str(e)[:200]}',
                    'mean_cindex': float('nan'),
                    'elapsed_sec': time.time() - t0})
        return rec

    out_branches.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)
    np.save(val_path, val_risk)
    rec.update({'cached': False, 'fold_cindex': fold_c, 'mean_cindex': mean_c,
                'elapsed_sec': time.time() - t0})
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--val-test', type=str, default=None,
                    help='Path to val_test.csv (parent dir if data-dir not set).')
    ap.add_argument('--mechanisms', type=str,
                    default='M1,M2,M5,M1+M5',
                    help='Comma-list of mechanisms to run')
    ap.add_argument('--models', type=str,
                    default='rsf,gbsa,mlp_dh',
                    help='Comma-list of base models to test')
    ap.add_argument('--protocols', type=str, default='P1,P2,P4',
                    help='Comma-list of training protocols')
    ap.add_argument('--endpoints', type=str, default='hepatic,death')
    ap.add_argument('--n-synth', type=int, default=10000,
                    help='M1 simulator cohort size')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--views', type=str, default='snapshot,summary',
                    help='Comma-list of view names to concatenate')
    ap.add_argument('--out-branches', type=str, default=str(BRANCHES_DIR))
    ap.add_argument('--out-synth-dir', type=str, default=str(SYNTH_DIR))
    ap.add_argument('--quick', action='store_true',
                    help='Smoke test: 2-fold, M1 only, RSF only, P1 only')
    ap.add_argument('--mlp-only', action='store_true',
                    help='Restrict to MLP_DH (P2 viable)')
    args = ap.parse_args()

    # Load real
    train = load_annitia(data_dir=args.data_dir)
    val_dir = args.val_test or args.data_dir
    val_kwargs = {'has_outcomes': False}
    val_test_filename = 'val_test.csv'
    if val_dir:
        val = load_annitia(data_dir=val_dir, filename=val_test_filename, **val_kwargs)
    else:
        # Try same dir as train
        val = load_annitia(data_dir=args.data_dir, filename=val_test_filename, **val_kwargs)

    if args.quick:
        args.mechanisms = 'M1'
        args.models = 'rsf'
        args.protocols = 'P1'
        args.n_folds = 2
        args.n_synth = 500

    if args.mlp_only:
        args.models = 'mlp_dh'

    mechanisms = [m for m in args.mechanisms.split(',') if m]
    models = [m for m in args.models.split(',') if m]
    protocols = [p for p in args.protocols.split(',') if p]
    endpoints = [e for e in args.endpoints.split(',') if e]
    views = [v for v in args.views.split(',') if v]

    out_branches = Path(args.out_branches)
    out_synth_dir = Path(args.out_synth_dir)
    out_synth_dir.mkdir(parents=True, exist_ok=True)
    out_branches.mkdir(parents=True, exist_ok=True)

    # Pre-build all synth cohorts
    synth_dfs = {}
    for mech in mechanisms:
        print(f'[synth-build] {mech}', flush=True)
        synth_dfs[mech] = _build_or_load_synth(mech, train, args.n_synth, args.seed)
        print(f'  cohort size={len(synth_dfs[mech])}', flush=True)

    summary_path = out_synth_dir / 'grid_summary.json'
    summary = {'conditions': []}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    total = len(mechanisms) * len(models) * len(protocols) * len(endpoints)
    done = 0
    for ep in endpoints:
        for mech in mechanisms:
            for model_name in models:
                # P2 only valid for MLP
                allowed_protos = list(protocols)
                if 'P2' in allowed_protos and model_name != 'mlp_dh':
                    allowed_protos = [p for p in allowed_protos if p != 'P2']
                for proto in allowed_protos:
                    done += 1
                    print(f'[{done}/{total}] {ep:7s} {mech:6s} {model_name:7s} {proto} ...',
                          flush=True)
                    rec = run_condition(
                        train, val, synth_dfs[mech],
                        mechanism=mech, model_name=model_name, protocol=proto,
                        endpoint=ep, view_names=views,
                        out_branches=out_branches, out_synth_dir=out_synth_dir,
                        seed=args.seed, n_folds=args.n_folds,
                    )
                    print(f'    C={rec.get("mean_cindex", float("nan")):.4f} '
                          f'{("[CACHED]" if rec.get("cached") else "")} '
                          f'{rec.get("error", "")}', flush=True)
                    summary['conditions'] = [c for c in summary['conditions']
                                             if not (c.get('cid') == rec['cid']
                                                     and c.get('endpoint') == rec['endpoint'])]
                    summary['conditions'].append(rec)
                    summary_path.write_text(json.dumps(summary, indent=2,
                                                       default=str))
    print('\n=== TOP-10 by mean C-index per endpoint ===')
    for ep in endpoints:
        def _is_finite(x):
            try: return np.isfinite(float(x))
            except Exception: return False
        recs = [c for c in summary['conditions']
                if c.get('endpoint') == ep
                and _is_finite(c.get('mean_cindex'))]
        recs.sort(key=lambda c: -c['mean_cindex'])
        print(f'\n{ep}:')
        for r in recs[:10]:
            print(f'  C={r["mean_cindex"]:.4f}  {r["mechanism"]:6s} '
                  f'{r["model"]:7s} {r["protocol"]}  {r["cid"]}')


if __name__ == '__main__':
    main()
