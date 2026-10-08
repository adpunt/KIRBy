"""Loader for thesis figure F1: the hybrid against the best single
representation, per dataset and downstream model (outline section 5.4,
file pointer c02, c22).

Source: c02 only (c02_complexity_matching_grid). For each dataset and each of
its two decided models, c02 picks on each outer fold's TRAINING rows the top
single representation ('base1') and the top hybrid ('hybrid1'), fits each on
those rows and scores it on the fold's held-out rows. The experiment's own
summary is the mean over outer folds of those held-out scores:

    standard_selection_summary.<model>.base1.heldout.mean     (best single)
    standard_selection_summary.<model>.hybrid1.heldout.mean   (hybrid)

The best single representation and the hybrid recipe are chosen per fold, so
they can differ between folds; both are returned per fold beside the mean.
best_single_rep is the representation chosen on most folds.

c22 runs the same per-fold selection with independent model fits (LogD dnn
base1 0.4513 in c22 against 0.4497 in c02), so it is a second draw of the same
numbers, not extra coverage. It is not read here.

Datasets: the roster in tests/experiments/orchestrate/plan.py
(DATASET_REPS minus EXCLUDED_DATASETS and CUT_DATASETS). A roster dataset with
no c02 result is reported through missing(). A finished .json is preferred;
without it, a .partial.jsonl is decoded and its fold count reported through
missing() (it stores no metric name, so no mark is built from it).

Read-only. stdlib only.
"""
import collections
import glob
import json
import math
import os
import pathlib
import sys

C02_DIR = 'complexity/c02_complexity_matching_grid'
INPUTS = [C02_DIR + '/*.json', C02_DIR + '/*.partial.jsonl']

# The final script lives at <repo>/tests/analysis/; overridden in __main__
# for the scratch test.
REPO = pathlib.Path(__file__).resolve().parents[2]

METRIC_NAMES = {'auc': 'auroc'}   # c02 writes 'auc' for AUROC


def _repo_imports():
    tests = str(REPO / 'tests')
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from experiments.orchestrate import plan
    from experiments._checkpoint import decode
    return plan, decode


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _modal(names):
    names = [n for n in names if n]
    if not names:
        return None
    c = collections.Counter(names)
    top = max(c.values())
    return sorted(n for n, k in c.items() if k == top)[0]


def _rows_from_json(path, rel, ds, missing):
    d = json.load(open(path))
    metric = d.get('metric_name')
    if not metric:
        missing(f'{ds}: no metric_name in {rel}')
        return []
    metric = METRIC_NAMES.get(metric.lower(), metric.lower())
    summ = d.get('standard_selection_summary') or {}
    out = []
    for m in d.get('models') or []:
        ms = summ.get(m) or {}
        b, h = ms.get('base1') or {}, ms.get('hybrid1') or {}
        bh, hh = b.get('heldout') or {}, h.get('heldout') or {}
        what = []
        if not _finite(bh.get('mean')):
            what.append('base1 held-out mean')
        if not _finite(hh.get('mean')):
            what.append('hybrid1 held-out mean')
        if what:
            missing(f'{ds} / {m}: no {" and no ".join(what)} in {rel}')
            continue
        bpf, hpf = b.get('per_fold') or [], h.get('per_fold') or []
        out.append({
            'dataset': ds, 'model': m, 'metric': metric,
            'lower_is_better': bool(d.get('lower_is_better')),
            'best_single_rep': _modal([p.get('name') for p in bpf]),
            'best_single_rep_by_fold': [p.get('name') for p in bpf],
            'best_single_score': float(bh['mean']),
            'hybrid_label': 'hybrid1',
            'hybrid_label_by_fold': [p.get('name') for p in hpf],
            'hybrid_score': float(hh['mean']),
            'n_folds': int(min(bh.get('n_folds') or 0, hh.get('n_folds') or 0)),
            'n_folds_best_single': int(bh.get('n_folds') or 0),
            'n_folds_hybrid': int(hh.get('n_folds') or 0),
            'n_outer_folds_asked': d.get('n_outer_folds_asked'),
            'partial': False,
            'source': (f'{rel} :: standard_selection_summary.{m}'
                       '.{base1,hybrid1}.heldout.mean'),
        })
        if out[-1]['n_folds_best_single'] != out[-1]['n_folds_hybrid']:
            missing(f'{ds} / {m}: best single scored on '
                    f'{out[-1]["n_folds_best_single"]} folds, hybrid on '
                    f'{out[-1]["n_folds_hybrid"]} folds ({rel})')
    for m in d.get('models_dropped') or []:
        missing(f'{ds} / {m}: model dropped by c02 ({rel} models_dropped)')
    return out


def _rows_from_partial(path, rel, ds, decode, missing):
    """A dataset with only a checkpoint: the checkpoint holds per-fold
    scores but not the metric name, so no mark is built; the folds it holds
    are counted and reported through missing()."""
    folds = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = decode(json.loads(line))
        p = r.get('payload') if isinstance(r, dict) else None
        if isinstance(p, dict) and 'standard_selection' in p:
            folds.append(p)
    if not folds:
        missing(f'{ds}: c02 has no finished .json and its partial '
                f'checkpoint holds no scored fold ({rel})')
        return []
    missing(f'{ds}: c02 has only a partial checkpoint with '
            f'{len(folds)} fold(s) ({rel}); the metric name is not stored '
            'there, so no marks were built')
    return []


def load(results_root: pathlib.Path, missing) -> list:
    results_root = pathlib.Path(results_root)
    plan, decode = _repo_imports()
    roster = plan.dataset_roster()
    dropped = set(plan.CUT_DATASETS) | set(plan.EXCLUDED_DATASETS)
    d = results_root / C02_DIR
    rows = []
    for ds in roster:
        j = d / f'{ds}.json'
        p = d / f'{ds}.partial.jsonl'
        if j.exists():
            rel = str(j.relative_to(results_root))
            rows.extend(_rows_from_json(j, rel, ds, missing))
        elif p.exists():
            rel = str(p.relative_to(results_root))
            rows.extend(_rows_from_partial(p, rel, ds, decode, missing))
        else:
            from experiments.decided_grid import decided_models
            for m in decided_models(ds):
                missing(f'{ds} / {m}: no c02 result '
                        f'({C02_DIR}/{ds}.json absent)')
    # Result files for datasets outside the roster are ignored on purpose
    # when cut/excluded; anything else is unexpected and reported.
    for f in sorted(glob.glob(str(d / '*.json'))):
        ds = os.path.basename(f)[:-len('.json')]
        if ds not in roster and ds not in dropped:
            missing(f'{ds}: c02 result exists but the dataset is not in '
                    'the plan.py roster; not drawn')
    return rows


if __name__ == '__main__':
    REPO = pathlib.Path.home() / 'repos/KIRBy'
    root = pathlib.Path('/Volumes/seagate/kirby_results_2026-10-08')
    msgs = []
    rows = load(root, msgs.append)
    print(f'{len(rows)} rows')
    for r in rows:
        print(f"{r['dataset']:24s} {r['model']:9s} {r['metric']:12s} "
              f"single {r['best_single_rep']:14s} {r['best_single_score']:.4f}"
              f"  hybrid {r['hybrid_score']:.4f}  folds {r['n_folds']}")
    print(f'\n{len(msgs)} missing')
    for m in msgs:
        print('  MISSING', m)
    print('\nmetrics:', sorted(collections.Counter(r['metric'] for r in rows).items()))
