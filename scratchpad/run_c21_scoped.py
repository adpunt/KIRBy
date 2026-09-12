#!/usr/bin/env python3
"""Run c21 pinned to one base rep and one model, locally.

Why a wrapper rather than calling c21 directly:

1. LOCAL macOS SEGFAULT. load_data's import chain grabs an OpenMP runtime that
   conflicts with lightgbm's, and the first lightgbm fit after load_data kills
   the interpreter (exit 139, no traceback). Warming lightgbm up first fixes it.
   That is a local environment problem, not a server one, so it does not belong
   in the repo module.

2. c21's augmentation plumbing has a trap. run() does
       aug_reps = dict(data.get('augmentations') or {})
   and only falls back to loading by name for reps that are in NEITHER
   data['reps'] NOR data['augmentations']. So if the augmentation reps are
   sitting in data['reps'] (which is what a plain
   load_data(rep_names=base+aug) gives you), aug_reps ends up EMPTY, and the
   B_aug / D_both / H1 arms are silently skipped. That is the same class of
   silent no-op that made the previous attempt's use-case-A arm measure nothing.
   complexity run_all avoids it by POPPING aug-only reps out of data['reps']
   and parking them in data['augmentations']; this wrapper does the same, then
   asserts it worked.

Usage:
    KIRBY_CACHE_DIR=/Volumes/seagate/kirby_cache \
    python scratchpad/run_c21_scoped.py --dataset herg_ki \
        --base-rep mhg_gnn --model lightgbm --n-folds 5
"""
import argparse
import os
import sys

# Must precede the experiments._shared import — see note 1 above.
try:
    import numpy as _np
    import lightgbm as _lgb
    _lgb.LGBMRegressor(n_estimators=2, verbose=-1).fit(
        _np.zeros((8, 2)), _np.arange(8.0))
except Exception as _e:  # pragma: no cover
    print(f"  [warmup] lightgbm warm-up skipped: {_e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'experiments'))

from experiments._shared import load_data, save_experiment_results
from experiments.complexity.complexity_helpers import DATASET_REPS
from experiments.complexity import c21_single_rep_augmentation as c21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--base-rep', required=True,
                    help='THE target rep, taken from pass 1 — not chosen here')
    ap.add_argument('--model', default='lightgbm')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--top-n', type=int, default=5)
    ap.add_argument('--hybrid-budgets', default='500,1000,2000')
    ap.add_argument('--no-controls', action='store_true')
    ap.add_argument('--skip-hybrid-arms', action='store_true',
                    help='Single-rep arms only (A/B/C/D + controls)')
    ap.add_argument('--hybrid-only', action='store_true',
                    help='Reference arms H0/H1 only — skips the single-rep arms')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    cache = os.environ.get('KIRBY_CACHE_DIR')
    if not cache or 'seagate' not in cache:
        print(f"REFUSING TO RUN: KIRBY_CACHE_DIR={cache!r}. The default "
              f"~/.cache/kirby is missing pool reps and would start regenerating "
              f"heavy embeddings. Point it at the seagate cache.")
        return 2

    ds = args.dataset
    pool = DATASET_REPS[ds]['general']
    aug_names = DATASET_REPS[ds].get('augmentation') or []
    if args.base_rep not in pool:
        print(f"ERROR: --base-rep {args.base_rep!r} is not in {ds}'s pool {pool}")
        return 2

    data = load_data(ds, rep_names=pool + aug_names)

    # Partition: base reps stay in data['reps'], augmentation reps move to
    # data['augmentations']. See note 2 above.
    augs = {}
    for name in aug_names:
        if name in data['reps']:
            augs[name] = data['reps'].pop(name)
    data['augmentations'] = augs

    missing_base = [r for r in pool if r not in data['reps']]
    missing_aug = [r for r in aug_names if r not in augs]
    if missing_base or missing_aug:
        print(f"ERROR: missing base={missing_base} aug={missing_aug}")
        return 2
    assert not (set(data['reps']) & set(augs)), "base/aug overlap after partition"
    for n, a in augs.items():
        assert a.shape[0] == len(data['labels']) and a.shape[1] > 0, f"bad aug {n}"
    print(f"  base reps in play: {len(data['reps'])}")
    print(f"  augmentation reps in play: "
          f"{[(n, a.shape[1]) for n, a in augs.items()] or 'NONE'}")
    if not augs:
        print("  WARNING: augmentation pool is EMPTY — B/D/H1 arms cannot run.")

    config = {'n_splits': args.n_folds, 'search_speed': 'full'}
    budgets = tuple(int(b) for b in args.hybrid_budgets.split(',') if b.strip())
    if args.skip_hybrid_arms:
        budgets = ()

    out_dir = args.output or os.path.join(
        os.environ.get('KIRBY_SCRATCH', '/tmp'), 'c21_scoped')
    os.makedirs(out_dir, exist_ok=True)

    results = c21.run(
        data, config, out_dir, ds,
        top_n=args.top_n,
        base_reps=[args.base_rep],
        model_names=[args.model],
        n_seeds=args.n_seeds,
        hybrid_budgets=budgets,
        with_controls=not args.no_controls,
        skip_single_rep=args.hybrid_only,
    )
    save_experiment_results(results, out_dir,
                            f'c21_scoped_{args.base_rep}_{args.model}', ds)
    print(f"\n  -> {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
