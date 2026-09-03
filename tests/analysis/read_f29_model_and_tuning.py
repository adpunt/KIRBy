#!/usr/bin/env python3
"""Merge f29's per-task JSON files into one table per dataset.

Two tables per dataset:

  MODELS      best score each model reaches over the 10 representations, and
              the representation it reached it on, under the default
              parameters. Answers "which models show signal".
  TUNING      default vs tuned_per_rep vs tuned_per_model, per model, averaged
              over the 10 representations. Answers "does tuning pay, and per
              what".

Seconds per fit and per tune sit beside every number.

Usage:
    python tests/analysis/read_f29_model_and_tuning.py
    python tests/analysis/read_f29_model_and_tuning.py --dataset logd --full
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _ROOT / 'tests' / 'results' / 'framework'
ARMS = ['default', 'tuned_per_rep', 'tuned_per_model']


def load(results_dir, dataset=None):
    """Collect every f29 row, keyed (dataset, model, rep, arm) -> fold scores."""
    cells = defaultdict(lambda: {'scores': [], 'fit': [], 'tune': [],
                                 'dim': None, 'metric': None, 'lower': None,
                                 'params': [], 'errors': []})
    skipped = defaultdict(set)
    for path in sorted(Path(results_dir).glob('f29_*_full*.json')):
        blob = json.loads(path.read_text())
        ds = blob['dataset']
        if dataset and ds != dataset:
            continue
        for r in blob['rows']:
            if r.get('skipped'):
                skipped[ds].add((r['model'], r['skipped']))
                continue
            k = (ds, r['model'], r['rep'], r['arm'])
            c = cells[k]
            c['scores'].extend(r['fold_scores'])
            c['fit'].append(r['sec_per_fit'])
            c['tune'].append(r['sec_per_tune'])
            c['dim'] = r['rep_dim']
            c['metric'] = r['metric']
            c['lower'] = r['lower_is_better']
            c['params'].extend(p for p in r.get('best_params', []) if p)
            c['errors'].extend(r.get('errors', []))
    return cells, skipped


def _agg(c):
    s = np.asarray(c['scores'], dtype=float)
    return (float(np.nanmean(s)) if s.size else float('nan'),
            float(np.nanstd(s)) if s.size else float('nan'),
            float(np.mean(c['fit'])), float(np.mean(c['tune'])),
            int(np.sum(np.isnan(s))))


def report(cells, skipped, full=False):
    datasets = sorted({k[0] for k in cells})
    for ds in datasets:
        rows = {k: v for k, v in cells.items() if k[0] == ds}
        any_c = next(iter(rows.values()))
        metric, lower = any_c['metric'], any_c['lower']
        better = min if lower else max
        arrow = 'lower is better' if lower else 'higher is better'
        print(f"\n{'='*94}\n{ds}   metric={metric} ({arrow})\n{'='*94}")

        models = sorted({k[1] for k in rows})
        reps = sorted({k[2] for k in rows})

        print(f"\nMODELS — best of the 10 representations, default parameters")
        print(f"{'model':<10}{'best':>10}{'+/-':>9}  {'on representation':<22}"
              f"{'s/fit':>9}{'folds failed':>14}")
        ranked = []
        for m in models:
            cand = [(k, _agg(v)) for k, v in rows.items()
                    if k[1] == m and k[3] == 'default' and
                    not np.isnan(_agg(v)[0])]
            if not cand:
                continue
            k, (mu, sd, fit, tune, nf) = better(cand, key=lambda t: t[1][0])
            ranked.append((mu, m, k[2], sd, fit, nf))
        for mu, m, rep, sd, fit, nf in sorted(ranked, reverse=not lower):
            print(f"{m:<10}{mu:>10.4f}{sd:>9.4f}  {rep:<22}{fit:>9.2f}"
                  f"{nf:>14}")

        print(f"\nTUNING — mean over the 10 representations, per model")
        print(f"{'model':<10}" + ''.join(f"{a:>18}" for a in ARMS)
              + f"{'s/tune per fit':>17}")
        for m in models:
            line, tunes = f"{m:<10}", []
            for arm in ARMS:
                vals = [_agg(v)[0] for k, v in rows.items()
                        if k[1] == m and k[3] == arm]
                vals = [v for v in vals if not np.isnan(v)]
                line += f"{(np.mean(vals) if vals else float('nan')):>18.4f}"
                if arm != 'default':
                    tunes += [_agg(v)[3] for k, v in rows.items()
                              if k[1] == m and k[3] == arm]
            line += f"{(np.mean(tunes) if tunes else 0.0):>17.1f}"
            print(line)

        if full:
            print(f"\nEVERY CELL")
            print(f"{'model':<10}{'representation':<22}{'dim':>6}"
                  + ''.join(f"{a:>18}" for a in ARMS))
            for m in models:
                for rep in reps:
                    cs = [rows.get((ds, m, rep, a)) for a in ARMS]
                    if not any(cs):
                        continue
                    dim = next(c['dim'] for c in cs if c)
                    line = f"{m:<10}{rep:<22}{dim:>6}"
                    for c in cs:
                        line += f"{(_agg(c)[0] if c else float('nan')):>18.4f}"
                    print(line)

        for model, why in sorted(skipped.get(ds, ())):
            print(f"\n  {model}: no result — {why}")
        errs = {e for v in rows.values() for e in v['errors']}
        for e in sorted(errs)[:5]:
            print(f"  error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(DEFAULT_DIR))
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--full', action='store_true',
                    help='also print every (representation, model, arm) cell')
    a = ap.parse_args()
    cells, skipped = load(a.results, a.dataset)
    if not cells:
        raise SystemExit(f"no f29_*_full*.json under {a.results}")
    report(cells, skipped, full=a.full)


if __name__ == '__main__':
    main()
