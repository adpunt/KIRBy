"""A first look at the results of a run, dataset by dataset, in each
dataset's own metric: are the hybrids on a par with or ahead of the best
single representation, and are the numbers in a plausible range?

    python tests/experiments/orchestrate/sanity.py                 # the live run
    python tests/experiments/orchestrate/sanity.py --run tests/results/run_2026_10_06
    python tests/experiments/orchestrate/sanity.py --folds         # every fold, not the mean
    python tests/experiments/orchestrate/sanity.py --dataset caco2

Reads two result files per dataset and nothing else:
  complexity/c02_complexity_matching_grid/<ds>.json
      standard_selection_summary[model][slot].per_fold[].heldout_score for
      the slots base1, base2 (the 2 best single representations) and
      hybrid1, hybrid2 (the 2 best hybrids), all picked on each fold's
      training rows and scored on its held-out rows.
  complexity/c01_rep_characterization/<ds>.json
      <rep>.baselines[model].fold_scores: every pruned representation
      alone, per model, per fold.

Flags (printed after the table, one line each):
  LOSES     hybrid1 is behind base1 on the fold mean by more than base1's
            own fold-to-fold spread (max minus min). Smaller gaps are
            inside the noise and not flagged.
  RANGE     a score outside what the metric can give (auc, f1 outside 0..1;
            spearman, mcc outside -1..1; r2 above 1), or an rmse above 100
            (the c01 dnn adme_hlm case of 2026-10-03 was 72,237).
  MISSING   a slot with no held-out score on some fold (an error row).
  ONE FOLD  only one fold scored, so there is no spread to judge by.
  OLD FORMAT  a c02 file from before the per-fold alignment; not checked.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent

BOUNDS = {'auc': (0, 1), 'auroc': (0, 1), 'f1': (0, 1), 'f1_weighted': (0, 1),
          'accuracy': (0, 1), 'spearman': (-1, 1), 'pearson': (-1, 1),
          'mcc': (-1, 1), 'ci': (0, 1), 'r2': (None, 1)}
RMSE_CEILING = 100.0


def resolve_run(run):
    if run:
        p = Path(run)
        return p if p.is_absolute() else REPO / p
    ptr = TESTS / 'results' / 'CURRENT_RUN'
    if ptr.exists():
        return Path(ptr.read_text().strip())
    sys.exit('no run: pass --run or init one')


def _num(x):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return None


def _mean(v):
    v = [x for x in v if x is not None]
    return (sum(v) / len(v)) if v else None


def _spread(v):
    v = [x for x in v if x is not None]
    return (max(v) - min(v)) if len(v) > 1 else None


def range_flag(metric, x):
    if x is None:
        return None
    m = (metric or '').lower()
    if m in ('rmse', 'mae'):
        return f'{m} {x:.4g} above {RMSE_CEILING:g}' if x > RMSE_CEILING else None
    lo, hi = BOUNDS.get(m, (None, None))
    if lo is not None and x < lo:
        return f'{m} {x:.4g} below {lo}'
    if hi is not None and x > hi:
        return f'{m} {x:.4g} above {hi}'
    return None


def read_c02(root, ds):
    f = root / 'complexity' / 'c02_complexity_matching_grid' / f'{ds}.json'
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if d.get('result_status') not in (None, 'ok') and 'standard_selection_summary' not in d:
        return None
    return d


def read_c01(root, ds):
    f = root / 'complexity' / 'c01_rep_characterization' / f'{ds}.json'
    return json.loads(f.read_text()) if f.exists() else None


def fmt(x):
    return '   --  ' if x is None else f'{x:7.4f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default=None)
    ap.add_argument('--dataset', nargs='*', default=None)
    ap.add_argument('--folds', action='store_true', help='one line per fold instead of the mean')
    a = ap.parse_args(argv)
    root = resolve_run(a.run)
    c02_dir = root / 'complexity' / 'c02_complexity_matching_grid'
    datasets = sorted(p.stem for p in c02_dir.glob('*.json')) if c02_dir.exists() else []
    if a.dataset:
        datasets = [d for d in datasets if d in a.dataset]
    if not datasets:
        print(f'no c02 result file yet under {c02_dir}')
        return
    print(f'run {root.name}: {len(datasets)} dataset(s) with a c02 file. One line is one model on '
          f'one dataset; scores are held-out, in the dataset\'s metric, mean over the folds scored '
          f'(spread = max minus min over folds). base1/base2 = the 2 best single representations, '
          f'hybrid1/hybrid2 = the 2 best hybrids, all chosen on training rows.\n')
    hdr = f"{'dataset':24s} {'model':9s} {'metric':8s} {'dir':6s} {'folds':5s} {'base1':>8s} {'base2':>8s} {'hybrid1':>8s} {'hybrid2':>8s} {'h1-b1':>8s} {'spread':>8s}  base1 rep / hybrid1"
    print(hdr)
    print('-' * len(hdr))
    flags = []
    for ds in datasets:
        d = read_c02(root, ds)
        if d is None:
            flags.append(f'{ds}: c02 file present but not ok (status {json.loads((c02_dir / (ds + ".json")).read_text()).get("result_status")})')
            continue
        metric = d.get('metric_name') or '?'
        lower = bool(d.get('lower_is_better'))
        direction = 'lower' if lower else 'higher'
        summ = d.get('standard_selection_summary') or {}
        if not summ:
            flags.append(f"{ds}: OLD FORMAT c02 file has no standard_selection_summary "
                         f"(result_format {d.get('result_format')}); written before the "
                         f"per-fold alignment of 2026-10-05, nothing to check")
            continue
        for model, slots in summ.items():
            per = {}
            names = {}
            for slot in ('base1', 'base2', 'hybrid1', 'hybrid2'):
                rows = (slots.get(slot) or {}).get('per_fold') or []
                per[slot] = {int(r['fold']): _num(r.get('heldout_score')) for r in rows}
                names[slot] = next((r.get('name') for r in rows if r.get('name')), '')
            folds = sorted(set().union(*[set(v) for v in per.values()])) if any(per.values()) else []
            means = {s: _mean(list(per[s].values())) for s in per}
            spread_b1 = _spread(list(per['base1'].values()))
            delta = None
            if means['hybrid1'] is not None and means['base1'] is not None:
                delta = (means['base1'] - means['hybrid1']) if lower else (means['hybrid1'] - means['base1'])
            tag = f"{(names['base1'] or '')[:18]} / {(names['hybrid1'] or '')[:40]}"
            if a.folds:
                for k in folds:
                    vals = [per[s].get(k) for s in ('base1', 'base2', 'hybrid1', 'hybrid2')]
                    print(f"{ds:24s} {model:9s} {metric:8s} {direction:6s} f{k:<4d} "
                          + ' '.join(fmt(v) for v in vals) + f" {'':>8s} {'':>8s}  {tag}")
            print(f"{ds:24s} {model:9s} {metric:8s} {direction:6s} {len(folds):<5d} "
                  f"{fmt(means['base1'])} {fmt(means['base2'])} {fmt(means['hybrid1'])} {fmt(means['hybrid2'])} "
                  f"{('   --  ' if delta is None else f'{delta:+7.4f}')} {fmt(spread_b1)}  {tag}")
            # flags
            if len(folds) == 1:
                flags.append(f'{ds} {model}: ONE FOLD scored, no spread to judge by')
            if delta is not None and spread_b1 is not None and delta < -spread_b1:
                flags.append(f'{ds} {model}: LOSES hybrid1 behind base1 by {-delta:.4f} {metric}, '
                             f'more than base1 fold spread {spread_b1:.4f}')
            for s in per:
                for k, v in per[s].items():
                    if v is None:
                        flags.append(f'{ds} {model}: MISSING {s} fold {k} has no held-out score')
                    rf = range_flag(metric, v)
                    if rf:
                        flags.append(f'{ds} {model}: RANGE {s} fold {k}: {rf}')
        c01 = read_c01(root, ds)
        if c01:
            meta = c01.get('_meta') or {}
            m01 = meta.get('metric') or metric
            for rep, rr in c01.items():
                if rep.startswith('_') or not isinstance(rr, dict):
                    continue
                for model, b in (rr.get('baselines') or {}).items():
                    for k, v in enumerate(b.get('fold_scores') or []):
                        rf = range_flag(m01, _num(v))
                        if rf:
                            flags.append(f'{ds} {model}: RANGE c01 {rep} fold {k}: {rf}')
    print()
    if flags:
        print(f'{len(flags)} flag(s):')
        for f in flags:
            print('  ' + f)
    else:
        print('no flags: every hybrid within its base fold spread, every score in range, every slot scored')


if __name__ == '__main__':
    main()
