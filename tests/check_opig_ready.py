#!/usr/bin/env python
"""Read-only readiness check for the OPIG experiments. Run it on the server.

Answers, per experiment, without loading data or fitting anything:

  1. Which datasets does it run on?
  2. Does each dataset have an approved representation pool, and is that pool
     measured or borrowed from a neighbour?
  3. Which representations will it actually ask for?
  4. Are those representation matrices already on disk, or will the run rebuild
     them? (rep_cache/<dataset>/<rep>.npy)
  5. Is the dataset in the metric registry, and what metric and direction does
     it declare?

What it does NOT answer: whether the cached SEARCHES exist. That is
`python tests/check_precompute_status.py --family opig`. Run both.

Usage:
    python tests/check_opig_ready.py                  # every experiment
    python tests/check_opig_ready.py --experiment humanness
    python tests/check_opig_ready.py --missing-reps   # only the gaps
    python tests/check_opig_ready.py --names-only     # datasets not ready
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments import dataset_policy  # noqa: F401  (populates the registry)
from kirby.registry import DATASET_REGISTRY

# MIRROR of what each experiment file resolves. Keep in sync with the
# REP_EXCEPTIONS constants in the opig_*.py files; the reason for each exception
# is written at its call site, not here.
#
# 'extra' = representations the experiment adds to the approved pool because it
# needs them. 'own_reps' = the experiment scores a representation set of its own
# design and does not read the pool at all.
EXPERIMENTS = {
    'activity_cliff_qsar':      dict(datasets=['chembl_qsar'], extra=['sns']),
    'blueprint_transfer':       dict(datasets=['logd', 'herg_ki'], extra=[]),
    'cdr_flexibility':          dict(datasets=['cdr_flexibility'], extra=[]),
    'cgr_comparison':           dict(datasets=['buchwald_hartwig', 'bh_doyle_plates',
                                               'suzuki'], extra=[], own_reps=True),
    'ginkgo_developability':    dict(datasets=['ginkgo_hic', 'ginkgo_titer',
                                               'ginkgo_pr_cho', 'ginkgo_acsins',
                                               'ginkgo_colloidal'], extra=[]),
    'graphinity_ddg':           dict(datasets=['ddg_combined'], extra=[]),
    'humanness':                dict(datasets=['humanness_therapeutic'],
                                     extra=['onehot']),
    'kingsmith_component_splits': dict(datasets=['buchwald_hartwig',
                                                 'bh_doyle_plates'], extra=[]),
    'scaffold_hybrid':          dict(datasets=['flab_expression_koenig',
                                               'flab_binding_koenig',
                                               'flab_expression_adams',
                                               'flab_binding_adams'], extra=[]),
    'scaffold_similarity':      dict(datasets=['flab_expression_koenig',
                                               'flab_binding_koenig',
                                               'flab_expression_adams',
                                               'flab_binding_adams'], extra=[]),
    'trastuzumab_enrichment':   dict(datasets=['tz_her2_binary'],
                                     extra=['esmc_llr', 'antiberty_llr', 'esm2']),
}

# Pools copied from a pruned neighbour rather than measured on the dataset
# itself (see the comment on each entry in complexity_helpers.py).
BORROWED_POOLS = {'chembl_qsar', 'suzuki'}


def _cache_root() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _cached_reps(dataset):
    """Representation names with a matrix on disk, and the row count it was
    built from (None when no identity file is beside the arrays)."""
    d = _cache_root() / 'rep_cache' / dataset
    if not d.is_dir():
        return None, None
    names = {f.name[:-4] for f in d.iterdir() if f.name.endswith('.npy')}
    n_rows = None
    ident = d / '_identity.json'
    if ident.exists():
        try:
            obj = json.loads(ident.read_text())
            rows = obj.get('identity') if isinstance(obj, dict) else obj
            if isinstance(rows, list):
                n_rows = len(rows)
        except Exception:
            pass
    return names, n_rows


def resolve(dataset, extra):
    entry = DATASET_REPS.get(dataset)
    if not entry or not entry.get('general'):
        return None
    pool = list(entry['general']) + list(entry.get('augmentation', []))
    out = list(pool)
    for r in extra:
        if r not in out:
            out.append(r)
    return out


def report(name, cfg, show_only_gaps=False):
    lines = []
    ready = True
    for ds in cfg['datasets']:
        want = resolve(ds, cfg.get('extra', []))
        info = DATASET_REGISTRY.get(ds)
        metric = (f"{info.metric} ({'lower' if info.lower_is_better else 'higher'} is better)"
                  if info else "NOT IN REGISTRY — the run cannot resolve a metric")
        if want is None:
            lines.append(f"    {ds}")
            lines.append(f"      NO APPROVED POOL — this dataset will refuse to run")
            lines.append(f"      metric: {metric}")
            ready = False
            continue
        have, n_rows = _cached_reps(ds)
        if have is None:
            missing = want
            cached_note = "no rep_cache directory — every representation rebuilds"
        else:
            missing = [r for r in want if r not in have]
            rows = f", built from {n_rows} rows" if n_rows else ""
            cached_note = f"{len(want) - len(missing)} of {len(want)} cached{rows}"
        if missing:
            ready = False
        if show_only_gaps and not missing and info:
            continue
        tag = "  [BORROWED POOL, not measured]" if ds in BORROWED_POOLS else ""
        lines.append(f"    {ds}{tag}")
        lines.append(f"      metric: {metric}")
        lines.append(f"      wants {len(want)} representations: {cached_note}")
        if cfg.get('own_reps'):
            lines.append(f"      NOTE: this experiment scores its own representation "
                         f"set, not the pool — the pool is listed for cache checking only")
        if missing:
            lines.append(f"      MISSING: {', '.join(missing)}")
        if info is None:
            ready = False
    return ready, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', help='one experiment (short name, e.g. humanness)')
    ap.add_argument('--missing-reps', action='store_true',
                    help='show only datasets with an uncached representation')
    ap.add_argument('--names-only', action='store_true',
                    help='print dataset names that are not ready, one per line')
    args = ap.parse_args()

    names = ([args.experiment] if args.experiment else sorted(EXPERIMENTS))
    bad = [n for n in names if n not in EXPERIMENTS]
    if bad:
        print(f"ERROR: unknown experiment(s): {', '.join(bad)}")
        print(f"known: {', '.join(sorted(EXPERIMENTS))}")
        return 1

    if args.names_only:
        out = []
        for n in names:
            cfg = EXPERIMENTS[n]
            for ds in cfg['datasets']:
                want = resolve(ds, cfg.get('extra', []))
                have, _ = _cached_reps(ds)
                if want is None or have is None or [r for r in want if r not in have]:
                    out.append(ds)
        for ds in sorted(set(out)):
            print(ds)
        return 0

    print(f"cache root: {_cache_root()}")
    print(f"{len(names)} experiment(s)\n")
    n_ready = 0
    for n in names:
        ready, lines = report(n, EXPERIMENTS[n], show_only_gaps=args.missing_reps)
        n_ready += bool(ready)
        if lines or not args.missing_reps:
            print(f"  {n}  {'READY' if ready else 'NOT READY'}")
            for line in lines:
                print(line)
            print()
    print("=" * 70)
    print(f"{n_ready} of {len(names)} experiments have every representation cached")
    print("Searches are a separate question: "
          "python tests/check_precompute_status.py --family opig")
    return 0


if __name__ == '__main__':
    sys.exit(main())
