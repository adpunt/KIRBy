#!/usr/bin/env python3
"""ANNITIA external-data layer — NHANES pretrain/transfer + future T3 cohorts.

5th layer in the ANNITIA pipeline:

    BE  → CD  → HB  → BRANCHES  → EXTERNAL  → FUSION

What this layer owns:
    - Phase T2.0: NHANES encoder pretraining (calls pretrain_nhanes.py if
      the encoder.pt isn't on disk).
    - Phase T2.1: NHANES-only zero-shot evaluation against ANNITIA. Decision
      gate: if hep zero-shot > 0.55 OR death zero-shot > 0.65, T2.2 recipes
      are worth running.
    - Phase T2.2: Six transfer recipes (a-f) — Bayes-prior Cox, RSF tree
      warm-start, GBSA continued, PEFT MLP, joint+cohort-indicator Cox,
      NHANES-risk-as-feature stacking.
    - Phase T2.3: Subset ablation. By default runs all five subsets:
          fib4_high (default), fib4_high+2007-2018, fib4_high+2017-2018,
          t2dm_only, no_viral. Each produces its own subset-tagged config_id
          so cache + fusion handle them as independent conditions.

Outputs:
    tests/results/annitia/external/
        cv_summary.json
        oof_<endpoint>_<config_id>.npy
        val_<endpoint>_<config_id>.npy
        encoder_<subset>.pt      (one per T2.3 subset)
        zeroshot.json            (T2.1 zero-shot decision gate)

Usage:
    python tests/experiments/annitia/experiment_external.py
    python tests/experiments/annitia/experiment_external.py --skip-t23
    python tests/experiments/annitia/experiment_external.py --t23-subsets fib4_high,t2dm_only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia, get_cv_folds
from experiments.annitia.views import build_views
from experiments.annitia.survival import concordance
from experiments.annitia._oof import init_oof_buffer, fill_oof_fold


# T2.3 subset matrix — env-var triplet per subset
_T23_SUBSETS = {
    'fib4_high':       {'ANNITIA_NHANES_RESTRICT': 'fib4_high'},
    'fib4_high_2007':  {'ANNITIA_NHANES_RESTRICT': 'fib4_high',
                         'ANNITIA_NHANES_CYCLE_RANGE': '2007,2018'},
    'fib4_high_2017':  {'ANNITIA_NHANES_RESTRICT': 'fib4_high',
                         'ANNITIA_NHANES_CYCLE_RANGE': '2017,2018'},
    't2dm_only':       {'ANNITIA_NHANES_RESTRICT': 'fib4_high',
                         'ANNITIA_NHANES_T2DM': '1'},
    'no_viral':        {'ANNITIA_NHANES_RESTRICT': 'fib4_high',
                         'ANNITIA_NHANES_NOVIRAL': '1'},
}


def _convert(o):
    if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_convert(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o


def _set_env(env: dict):
    """Apply (or clear) a subset env dict. Returns the previous values."""
    prev = {}
    for k in ('ANNITIA_NHANES_RESTRICT', 'ANNITIA_NHANES_CYCLE_RANGE',
              'ANNITIA_NHANES_T2DM', 'ANNITIA_NHANES_NOVIRAL'):
        prev[k] = os.environ.get(k)
    for k, v in env.items():
        os.environ[k] = v
    # Clear keys absent in env but present before, to avoid leaking subsets
    for k in prev:
        if k not in env and k in os.environ:
            del os.environ[k]
    return prev


def _restore_env(prev: dict):
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _ensure_nhanes_encoder(out_dir: Path, subset_tag: str):
    """Train the NHANES encoder (subset-tagged) if not on disk."""
    encoder_path = out_dir / f'encoder_{subset_tag}.pt'
    if encoder_path.exists():
        print(f'  NHANES encoder cached: {encoder_path}')
        return str(encoder_path)
    print(f'  Pretraining NHANES encoder for subset={subset_tag} ...')
    cmd = ['python', '-u',
           str(_REPO / 'tests' / 'experiments' / 'annitia' / 'pretrain_nhanes.py'),
           '--restrict',
           os.environ.get('ANNITIA_NHANES_RESTRICT', 'fib4_high'),
           '--output', str(encoder_path)]
    print(f'  >>> {" ".join(cmd)}')
    r = subprocess.run(cmd)
    if r.returncode != 0 or not encoder_path.exists():
        print(f'  WARN: pretrain failed (rc={r.returncode}); subset skipped')
        return None
    return str(encoder_path)


def _zeroshot_eval(train, val, encoder_path: str):
    """T2.1 — apply NHANES-only-trained MLP zero-shot to ANNITIA.

    Returns dict with hep_c, death_c, decisions for the gate.
    """
    try:
        from experiments.annitia.pretrained_mlp import SurvivalPretrainedMLP
    except ImportError as e:
        return {'error': f'pretrained_mlp unavailable: {e}'}
    try:
        views = build_views(train, view_names=['nhanes_proj'])
    except Exception as e:
        return {'error': f'nhanes_proj view build failed: {e}'}
    X = views['nhanes_proj'][0]
    out = {}
    for endpoint in ('hepatic', 'death'):
        try:
            m = SurvivalPretrainedMLP(encoder_path=encoder_path,
                                       freeze_trunk=True, epochs=0)
            # epochs=0 → use the pretrained head directly. We approximate
            # zero-shot by skipping ANNITIA training entirely.
            m._mean = None  # forces _load_encoder
            m._load_encoder(X.shape[1])
            # Predict using pretrained trunk + a constant head (zero-shot)
            import torch
            with torch.no_grad():
                Xs = ((X - m._mean) / m._std).astype(np.float32)
                Xt = torch.tensor(Xs, dtype=torch.float32, device=m.device)
                rep = m._trunk(Xt)
                # No fine-tuned head — use sum-of-rep as a generic risk proxy
                risk = rep.sum(dim=1).cpu().numpy()
            y = train[endpoint]['y_struct']
            valid = train[endpoint]['valid_mask']
            c = concordance(y[valid], risk[valid])
            out[f'{endpoint}_c'] = float(c)
        except Exception as e:
            out[f'{endpoint}_err'] = str(e)[:120]
    out['hep_passes_gate']   = out.get('hepatic_c', 0.0) > 0.55
    out['death_passes_gate'] = out.get('death_c', 0.0)   > 0.65
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--output-dir', default='tests/results/annitia/external')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-t21', action='store_true',
                    help='Skip T2.1 zero-shot eval gate.')
    ap.add_argument('--skip-t22', action='store_true',
                    help='Skip T2.2 transfer recipes.')
    ap.add_argument('--skip-t23', action='store_true',
                    help='Skip T2.3 subset ablation (default subset only).')
    ap.add_argument('--t23-subsets', default=','.join(_T23_SUBSETS),
                    help='Comma-list of T2.3 subset names to run. '
                         f'Available: {",".join(_T23_SUBSETS)}')
    ap.add_argument('--ignore-t21-gate', action='store_true',
                    help='Run T2.2 recipes even if T2.1 zero-shot fails.')
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print('Loading train + val_test ...')
    train = load_annitia()
    n_train = len(train['patient_ids'])
    val_path = Path(args.validation_csv)
    if not val_path.exists():
        sys.exit(f'val_test missing: {val_path}')
    val = load_annitia(data_dir=str(val_path.parent), filename=val_path.name,
                       has_outcomes=False)
    n_val = len(val['patient_ids'])

    # Persistent cv_summary
    summary_path = out / 'cv_summary.json'
    summary = {'per_condition': {'hepatic': [], 'death': []},
               'n_folds': args.n_folds, 't23_subsets_run': []}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            for ep in ('hepatic', 'death'):
                summary['per_condition'][ep] = existing.get(
                    'per_condition', {}).get(ep, [])
            summary['t23_subsets_run'] = existing.get('t23_subsets_run', [])
            n_resume = sum(len(summary['per_condition'][e]) for e in ('hepatic', 'death'))
            if n_resume:
                print(f'  RESUME: loaded {n_resume} prior conditions')
        except Exception as e:
            print(f'  RESUME: failed ({e}); starting fresh')

    done = {ep: {r['config_id']: r for r in summary['per_condition'][ep]
                 if 'config_id' in r}
            for ep in ('hepatic', 'death')}

    # --- Subset list ---
    subsets = [s.strip() for s in args.t23_subsets.split(',') if s.strip()]
    if args.skip_t23:
        subsets = ['fib4_high']  # only default
    for s in subsets:
        if s not in _T23_SUBSETS:
            sys.exit(f'unknown T2.3 subset: {s} (available: {",".join(_T23_SUBSETS)})')

    # ---- Per-subset loop ----
    zeroshot_record = {}
    for subset in subsets:
        print(f'\n{"="*60}\nT2 subset = {subset}\n{"="*60}')
        prev_env = _set_env(_T23_SUBSETS[subset])
        try:
            # Step A: pretrain encoder (subset-tagged)
            encoder_path = _ensure_nhanes_encoder(out, subset)
            if encoder_path is None:
                print(f'  encoder unavailable for {subset}; skipping')
                continue
            os.environ['ANNITIA_NHANES_ENCODER'] = encoder_path

            # Step B: T2.1 zero-shot gate
            if not args.skip_t21:
                zs = _zeroshot_eval(train, val, encoder_path)
                zeroshot_record[subset] = zs
                print(f'  T2.1 zero-shot[{subset}]: hep_c={zs.get("hepatic_c", "n/a")} '
                      f'death_c={zs.get("death_c", "n/a")} '
                      f'hep_gate={zs.get("hep_passes_gate")} '
                      f'death_gate={zs.get("death_passes_gate")}')
                if (not args.ignore_t21_gate
                        and not zs.get('hep_passes_gate')
                        and not zs.get('death_passes_gate')):
                    print(f'  → both gates failed; skipping T2.2 for {subset}')
                    continue

            if args.skip_t22:
                continue

            # Step C: T2.2 recipes (a-f)
            try:
                from experiments.annitia.nhanes_transfer import (
                    BayesianPriorCox, RSFWarmStart, GBSAContinued,
                    PEFTPretrainedMLP, JointCohortCox, NHANESRiskAsFeature)
            except ImportError as e:
                print(f'  nhanes_transfer unavailable: {e}; skipping T2.2')
                continue
            recipes = [
                ('t2-a-bayesprior-cox',   lambda: BayesianPriorCox()),
                ('t2-b-rsf-warmstart',    lambda: RSFWarmStart(n_estimators=200)),
                ('t2-c-gbsa-continued',   lambda: GBSAContinued()),
                ('t2-d-peft-mlp',         lambda: PEFTPretrainedMLP()),
                ('t2-e-joint-cohort',     lambda: JointCohortCox()),
                ('t2-f-nhanes-risk-feat', lambda: NHANESRiskAsFeature()),
            ]
            try:
                views_train = build_views(train, view_names=['nhanes_proj'])
                views_val   = build_views(val,   view_names=['nhanes_proj'])
            except Exception as e:
                print(f'  nhanes_proj view build failed: {e}; skipping {subset}')
                continue
            X_tr = views_train['nhanes_proj'][0]
            X_va = views_val['nhanes_proj'][0]

            for label, ctor in recipes:
                for endpoint in ('hepatic', 'death'):
                    cid = f'{label}__subset={subset}'
                    oof_path = out / f'oof_{endpoint}_{cid}.npy'
                    val_path_npy = out / f'val_{endpoint}_{cid}.npy'
                    existing = done[endpoint].get(cid)
                    if (existing is not None and oof_path.exists() and val_path_npy.exists()
                            and np.isfinite(existing.get('mean_cindex', float('nan')))):
                        print(f'  {label:25s} {endpoint:7s} subset={subset:18s} CACHED  '
                              f'C={existing["mean_cindex"]:.4f}')
                        continue
                    y = train[endpoint]['y_struct']
                    valid = train[endpoint]['valid_mask']
                    folds = get_cv_folds(train, endpoint=endpoint,
                                         n_folds=args.n_folds, seed=args.seed)
                    t0 = time.time()
                    oof = init_oof_buffer(n_train); fold_c = []
                    try:
                        for k, (tr, va) in enumerate(folds):
                            est = ctor()
                            est.fit(X_tr[tr], y[tr])
                            risk = est.predict_risk(X_tr[va])
                            fold_c.append(concordance(y[va], risk))
                            fill_oof_fold(oof, va, risk)
                        mean_c = float(np.nanmean(fold_c))
                        std_c  = float(np.nanstd(fold_c))
                        valid_idx = np.where(valid)[0]
                        refit = ctor()
                        refit.fit(X_tr[valid_idx], y[valid_idx])
                        val_risk = np.asarray(refit.predict_risk(X_va), dtype=np.float64)
                    except ImportError as e:
                        print(f'  {label:25s} {endpoint:7s} subset={subset:18s} '
                              f'SKIP: {str(e)[:60]}')
                        continue
                    except Exception as e:
                        print(f'  {label:25s} {endpoint:7s} subset={subset:18s} '
                              f'ERR: {type(e).__name__}: {str(e)[:120]}')
                        continue
                    np.save(oof_path, oof); np.save(val_path_npy, val_risk)
                    rec = {'config_id': cid, 'label': label, 'subset': subset,
                           'fold_cindex': fold_c, 'mean_cindex': mean_c,
                           'std_cindex': std_c, 'elapsed_sec': time.time() - t0}
                    summary['per_condition'][endpoint] = [
                        r for r in summary['per_condition'][endpoint]
                        if r.get('config_id') != cid]
                    summary['per_condition'][endpoint].append(rec)
                    done[endpoint][cid] = rec
                    summary_path.write_text(json.dumps(_convert(summary), indent=2))
                    print(f'  {label:25s} {endpoint:7s} subset={subset:18s} '
                          f'C={mean_c:.4f}±{std_c:.4f}  ({time.time()-t0:.1f}s)')

            if subset not in summary['t23_subsets_run']:
                summary['t23_subsets_run'].append(subset)
        finally:
            _restore_env(prev_env)

    # Persist zeroshot record
    if zeroshot_record:
        (out / 'zeroshot.json').write_text(json.dumps(_convert(zeroshot_record), indent=2))

    summary['elapsed_sec'] = time.time() - t_start
    summary_path.write_text(json.dumps(_convert(summary), indent=2))
    print(f'\nDONE in {(time.time()-t_start)/60:.1f} min')
    print(f'  cv_summary: {summary_path}')
    print(f'  zeroshot:   {out / "zeroshot.json"}')


if __name__ == '__main__':
    main()
