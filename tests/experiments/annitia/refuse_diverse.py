"""Diversity-aware honest re-fusion.

The CV-best top-K filter picks correlated models (e.g. for hep, all 3
top-CV are bagged-rsf variants on the same view set). Per-patient
residual analysis showed that on the patients top1 gets wrong, very
different model families (tsfm-gbsa, snapshot-coxnet) get them right.
Greedy selection by ensemble OOF concordance — adding the model that
most improves the *current ensemble's* score, not just the highest-CV
candidate — picks a much more diverse set.

Smoke test on the existing pool:

    hepatic  CV-top-3 rank-mean = 0.8700
             diversity-5         = 0.8856   (+1.6%)
    death    CV-top-3 rank-mean = 0.9653
             diversity-5         = 0.9730   (+0.8%)

Output: tests/results/annitia/full/sub_master_diverse.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / 'src'))
sys.path.insert(0, str(_REPO / 'tests'))

from kirby.datasets.annitia_maslg import load_annitia                      # noqa: E402
from experiments.annitia.survival import concordance                       # noqa: E402


def _rank01(x):
    finite = np.isfinite(x); out = np.full_like(x, np.nan, dtype=float)
    if finite.sum():
        out[finite] = rankdata(x[finite]) / finite.sum()
    return out


def _c(y, valid, scores):
    m = valid & np.isfinite(scores) & np.isfinite(y['time'])
    if m.sum() < 5:
        return float('nan')
    return float(concordance(y[m], scores[m]))


def _collect(pool_dirs, endpoint):
    out = []
    for d in pool_dirs:
        cv = d / 'cv_summary.json'
        if not cv.exists():
            continue
        sm = json.loads(cv.read_text())
        for r in sm.get('per_condition', {}).get(endpoint, []):
            cid = r.get('config_id', '')
            mc = r.get('mean_cindex', float('nan'))
            if not np.isfinite(mc):
                continue
            oof_p = d / f'oof_{endpoint}_{cid}.npy'
            val_p = d / f'val_{endpoint}_{cid}.npy'
            if not (oof_p.exists() and val_p.exists()):
                continue
            out.append({
                'label': f'{d.name}__{cid}',
                'cv': float(mc),
                'oof': np.load(oof_p), 'val': np.load(val_p),
            })
    out.sort(key=lambda r: -r['cv'])
    return out


def _greedy_diverse(recs, y, valid, max_size=10, min_gain=1e-4):
    """Greedy add-the-model-that-most-improves-ensemble-OOF-C."""
    if not recs:
        return [], np.zeros_like(y, dtype=float), float('nan')
    chosen = [recs[0]]
    chosen_rks = [_rank01(recs[0]['oof'])]
    chosen_val_rks = [_rank01(recs[0]['val'])]
    best_c = _c(y, valid, np.mean(np.stack(chosen_rks, axis=1), axis=1))
    pool = list(recs[1:])
    history = [(recs[0]['label'], best_c)]
    while len(chosen) < max_size and pool:
        best_score, best_i = -np.inf, None
        for i, cand in enumerate(pool):
            cand_rk = _rank01(cand['oof'])
            tmp = np.mean(np.stack(chosen_rks + [cand_rk], axis=1), axis=1)
            c = _c(y, valid, tmp)
            if c > best_score:
                best_score, best_i = c, i
        if best_i is None or best_score - best_c < min_gain:
            break
        chosen.append(pool[best_i])
        chosen_rks.append(_rank01(pool[best_i]['oof']))
        chosen_val_rks.append(_rank01(pool[best_i]['val']))
        best_c = best_score
        history.append((pool[best_i]['label'], best_c))
        pool.pop(best_i)
    final_oof = np.mean(np.stack(chosen_rks, axis=1), axis=1)
    final_val = np.mean(np.stack(chosen_val_rks, axis=1), axis=1)
    return chosen, history, final_val, best_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-csv',
                    default='/Volumes/seagate/kirby/annitia/val_test.csv')
    ap.add_argument('--max-size', type=int, default=10)
    ap.add_argument('--output', default='tests/results/annitia/full/sub_master_diverse.csv')
    args = ap.parse_args()

    res_root = _REPO / 'tests' / 'results' / 'annitia'
    pool_dirs = [res_root / p for p in
                 ('be', 'cd', 'hep_boost', 'branches', 'external', 'hparam', 'research', 'research_v2')
                 if (res_root / p).exists()]
    print(f'Pooling from: {[d.name for d in pool_dirs]}')

    train = load_annitia()
    val = load_annitia(data_dir=str(Path(args.validation_csv).parent),
                       filename=Path(args.validation_csv).name,
                       has_outcomes=False)

    master = pd.DataFrame({'trustii_id': val['trustii_id']})
    summary = {}
    for endpoint, col in (('hepatic', 'risk_hepatic_event'),
                          ('death', 'risk_death')):
        recs = _collect(pool_dirs, endpoint)
        y = train[endpoint]['y_struct']
        valid = train[endpoint]['valid_mask']
        print(f'\n=== {endpoint} (pool of {len(recs)}) ===')
        chosen, history, val_pred, oof_c = _greedy_diverse(
            recs, y, valid, max_size=args.max_size)
        for rank, (lbl, c) in enumerate(history, 1):
            print(f'  +{rank} {lbl[:60]:60s}  ensemble OOF-C={c:.4f}')
        master[col] = val_pred
        summary[endpoint] = {
            'final_oof_c': oof_c,
            'ensemble_size': len(chosen),
            'members': [r['label'] for r in chosen],
            'history': [{'rank': i + 1, 'model': h[0], 'oof_c': h[1]}
                        for i, h in enumerate(history)],
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)
    out_path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nDONE')
    print(f'  CSV:     {out_path}')
    print(f'  hep OOF C={summary["hepatic"]["final_oof_c"]:.4f}, '
          f'death OOF C={summary["death"]["final_oof_c"]:.4f}')


if __name__ == '__main__':
    main()
