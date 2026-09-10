#!/usr/bin/env python3
"""Which distance won, per experiment, per dataset.

Reads the four result trees the distance sweep writes and says, for each
experiment and each dataset, which of manhattan / euclidean / cosine /
minkowski p=0.5 produced the better number.

    tests/results/dist_sweep/<metric>/<experiment_id>/<dataset>.json

run_dataset.py adds a category level and that is read too:

    tests/results/dist_sweep/<metric>/<category>/<experiment_id>/<dataset>.json

WHY IT DOES NOT KNOW EACH EXPERIMENT'S HEADLINE. The suite is not finished. A
table mapping every experiment to its one headline number would be written
against files that are still being rewritten, and would go stale silently. So
this walks every numeric value in each JSON, keeps the ones that appear at the
same position in all four runs, and compares those. Name a headline in
HEADLINE_PATHS when an experiment settles and it gets promoted to its own line.

DIRECTION. Higher is better for a correlation, precision or coherence; lower is
better for an error, a violation rate or a retrieval cost. DIRECTION_RULES below
decides from the key name, and anything it does not recognise is reported with
its four values and NO winner rather than being guessed at. Each results file
also carries `metric_name` and `lower_is_better` from
save_experiment_results, and a value whose key contains that metric name takes
its direction from the file.

Stdlib only. It must run on a laptop with no experiment stack importable.

Usage:
    python tests/analysis/compare_distance_metrics.py
    python tests/analysis/compare_distance_metrics.py --experiment s01_similarity_baseline
    python tests/analysis/compare_distance_metrics.py --csv /tmp/distance_sweep.csv
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = 'tests/results/dist_sweep'
DEFAULT_METRICS = ('manhattan', 'euclidean', 'cosine', 'minkowski_p0.5')
DEFAULT_DATASETS = ('buchwald_hartwig', 'caco2', 'flab_expression_koenig',
                    'tcga_brca_pam50')

# Key-name fragments that fix a value's direction. Checked against the LAST
# segment of the JSON path first, then the whole path.
LOWER_IS_BETTER = (
    'rmse', 'mae', 'mse', '_error', 'error_', 'abs_error', 'loss',
    'violation', 'retrieval_cost', 'false_positive', '_fp_rate', 'fp_rate',
    'regret', 'noise_score', 'concentration', 'condition_number',
)
HIGHER_IS_BETTER = (
    'spearman', 'pearson', 'rho', '_r2', 'r2_', 'auroc', 'auprc', 'f1',
    'accuracy', 'precision', 'recall', 'coherence', 'correlation',
    'improvement', 'delta_r2', 'sac', 'alignment', 'purity', 'win_rate',
)

# Fill in as an experiment settles: experiment_id -> tuple of JSON paths that
# ARE its headline. Empty means every comparable path counts equally.
HEADLINE_PATHS = {}

# Keys that carry configuration or provenance rather than a measurement. They
# differ between runs for reasons that are not results.
IGNORE_PATH_FRAGMENTS = (
    'timestamp', 'elapsed', 'seconds', 'runtime', 'wall', 'n_samples',
    'n_features', 'seed', 'budget', 'n_folds', 'n_splits', 'version',
    'input_fingerprint', 'distance_metric', 'lower_is_better',
)

_NUM = (int, float)


def flatten(obj, prefix=''):
    """Every numeric leaf as {dotted.path: value}. Lists index by position."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, _NUM) and math.isfinite(obj):
        out[prefix] = float(obj)
    return out


def is_ignorable(path):
    low = path.lower()
    return any(f in low for f in IGNORE_PATH_FRAGMENTS)


def direction(path, file_metric_name=None, file_lower_is_better=None):
    """+1 if higher wins, -1 if lower wins, 0 if this script will not guess."""
    low = path.lower()
    last = low.rsplit('.', 1)[-1]
    if file_metric_name and file_metric_name.lower() in low:
        if file_lower_is_better is not None:
            return -1 if file_lower_is_better else 1
    for probe in (last, low):
        for frag in LOWER_IS_BETTER:
            if frag in probe:
                return -1
        for frag in HIGHER_IS_BETTER:
            if frag in probe:
                return 1
    return 0


def load_tree(root, metrics, datasets, experiment_filter=None):
    """{(experiment, dataset): {metric: (flattened, meta)}} plus a warning list."""
    found = defaultdict(dict)
    warnings = []
    for metric in metrics:
        base = Path(root) / metric
        if not base.is_dir():
            warnings.append(f'no directory for {metric}: {base}')
            continue
        # Two layouts reach here. A single experiment writes
        # <metric>/<experiment_id>/<dataset>.json; run_dataset.py adds a
        # category level, <metric>/<category>/<experiment_id>/<dataset>.json.
        # The experiment id is the directory path below <metric> either way.
        exp_dirs = sorted({jf.parent for jf in base.rglob('*.json')})
        for exp_dir in exp_dirs:
            exp = exp_dir.relative_to(base).as_posix()
            if experiment_filter and not re.search(experiment_filter, exp):
                continue
            for jf in sorted(exp_dir.glob('*.json')):
                ds = jf.stem
                if datasets and ds not in datasets:
                    continue
                try:
                    payload = json.loads(jf.read_text())
                except Exception as exc:
                    warnings.append(f'unreadable {jf}: {exc}')
                    continue
                stamped = payload.get('distance_metric')
                if stamped is None:
                    warnings.append(
                        f'{jf} carries no distance_metric stamp. It was written '
                        f'before the flag existed, or by a script that does not '
                        f'call save_experiment_results. Its distance is unknown '
                        f'-- it is NOT evidence for {metric}.')
                elif stamped != metric:
                    warnings.append(
                        f'{jf} sits under {metric}/ but stamps {stamped!r}. '
                        f'The run did not receive --distance-metric {metric}.')
                    continue
                if payload.get('result_status') == 'empty':
                    warnings.append(f'{jf} finished with no numbers in it')
                    continue
                meta = {
                    'metric_name': payload.get('metric_name'),
                    'lower_is_better': payload.get('lower_is_better'),
                }
                found[(exp, ds)][metric] = (flatten(payload), meta)
    return found, warnings


def compare(found, metrics, min_spread):
    """One row per comparable path."""
    rows = []
    for (exp, ds), by_metric in sorted(found.items()):
        present = [m for m in metrics if m in by_metric]
        if len(present) < 2:
            continue
        common = None
        for m in present:
            keys = set(by_metric[m][0])
            common = keys if common is None else (common & keys)
        meta = by_metric[present[0]][1]
        headline = HEADLINE_PATHS.get(exp)
        for path in sorted(common):
            if is_ignorable(path):
                continue
            if headline and path not in headline:
                continue
            vals = {m: by_metric[m][0][path] for m in present}
            spread = max(vals.values()) - min(vals.values())
            scale = max(abs(v) for v in vals.values()) or 1.0
            if spread <= min_spread * scale:
                continue
            d = direction(path, meta.get('metric_name'),
                          meta.get('lower_is_better'))
            if d == 0:
                winner, best = None, None
            else:
                winner = (max(vals, key=lambda m: vals[m]) if d > 0
                          else min(vals, key=lambda m: vals[m]))
                best = vals[winner]
            rows.append({
                'experiment': exp, 'dataset': ds, 'path': path,
                'direction': {1: 'higher', -1: 'lower', 0: 'unknown'}[d],
                'winner': winner, 'best': best, 'spread': spread,
                'is_headline': bool(headline),
                **{f'value_{m}': vals.get(m) for m in metrics},
            })
    return rows


def report(rows, metrics, warnings, top):
    if warnings:
        print(f'\n{len(warnings)} thing(s) to know before reading this:')
        for w in warnings[:40]:
            print(f'  - {w}')
        if len(warnings) > 40:
            print(f'  ... and {len(warnings) - 40} more')

    scored = [r for r in rows if r['winner']]
    unknown = [r for r in rows if not r['winner']]
    if not rows:
        print('\nNo comparable numbers. Nothing has been run yet, or the runs '
              'sit somewhere other than --root.')
        return

    print(f'\n{len(scored)} comparable quantities with a known direction, '
          f'{len(unknown)} whose direction this script will not guess.')

    by_ds = defaultdict(lambda: defaultdict(int))
    by_exp = defaultdict(lambda: defaultdict(int))
    for r in scored:
        by_ds[r['dataset']][r['winner']] += 1
        by_exp[(r['experiment'], r['dataset'])][r['winner']] += 1

    print('\n=== Which distance won, per dataset (count of quantities) ===')
    head = f'{"dataset":26s}' + ''.join(f'{m:>17s}' for m in metrics)
    print(head); print('-' * len(head))
    for ds in sorted(by_ds):
        line = f'{ds:26s}' + ''.join(f'{by_ds[ds].get(m, 0):>17d}' for m in metrics)
        print(line)

    print('\n=== Per experiment, per dataset ===')
    head = f'{"experiment":34s}{"dataset":26s}' + ''.join(f'{m:>17s}' for m in metrics) + '   winner'
    print(head); print('-' * len(head))
    for (exp, ds) in sorted(by_exp):
        counts = by_exp[(exp, ds)]
        top_m = max(counts, key=lambda m: counts[m])
        tied = [m for m in counts if counts[m] == counts[top_m]]
        verdict = top_m if len(tied) == 1 else 'tied: ' + '/'.join(sorted(tied))
        print(f'{exp:34s}{ds:26s}'
              + ''.join(f'{counts.get(m, 0):>17d}' for m in metrics)
              + f'   {verdict}')

    if top:
        print(f'\n=== The {top} widest gaps ===')
        for r in sorted(scored, key=lambda r: -r['spread'])[:top]:
            vals = '  '.join(f'{m}={r[f"value_{m}"]:.4f}'
                             for m in metrics if r.get(f'value_{m}') is not None)
            print(f'  {r["experiment"]}/{r["dataset"]}  {r["path"]}\n'
                  f'      {vals}\n'
                  f'      {r["winner"]} wins ({r["direction"]} is better), '
                  f'spread {r["spread"]:.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--metrics', default=','.join(DEFAULT_METRICS))
    ap.add_argument('--datasets', default=','.join(DEFAULT_DATASETS),
                    help="comma-separated, or 'all'")
    ap.add_argument('--experiment', default=None,
                    help='regex; only experiments whose id matches')
    ap.add_argument('--min-spread', type=float, default=1e-9,
                    help='ignore quantities whose four values agree to within '
                         'this fraction of their own scale')
    ap.add_argument('--top', type=int, default=15,
                    help='how many widest gaps to print in full')
    ap.add_argument('--csv', default=None, help='write every row here')
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    datasets = (None if args.datasets.strip().lower() == 'all'
                else [d.strip() for d in args.datasets.split(',') if d.strip()])

    found, warnings = load_tree(args.root, metrics, datasets, args.experiment)
    rows = compare(found, metrics, args.min_spread)
    report(rows, metrics, warnings, args.top)

    if args.csv:
        cols = ['experiment', 'dataset', 'path', 'direction', 'winner', 'best',
                'spread', 'is_headline'] + [f'value_{m}' for m in metrics]
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print(f'\n{len(rows)} rows written to {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
