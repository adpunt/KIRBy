#!/usr/bin/env python3
"""Run ONE noise experiment on ONE dataset, cut to one fold, one condition, one level.

A SAMPLE for the prelim check (plan section 7, item 2). Not a results run.

WHY THIS FILE EXISTS. run_dataset.py takes --noise-condition, and nothing else
that narrows a noise run. It has no flag for one fold or one level, and this
file must not add one to it. The two env vars that look like they do are wrong
for this check:
  KIRBY_LOCAL_FOLDS  sets find_best_hybrid.DEFAULT_N_FOLDS too (_local_grid.py),
                     so the hybrid search is asked for a 3-fold search that the
                     precompute never stored; with KIRBY_NO_COLD_SEARCH=1 that
                     refuses, without it it searches from nothing.
  KIRBY_SMOKE_N      caps the number of MOLECULES (_shared.load_data). Rows are
                     never cut.

WHAT THIS DOES INSTEAD, all in this process, before any experiment module is
imported:
  fold   _shared.cv_split and _shared.make_cv_splitter are wrapped. A split
         over EVERY molecule of the dataset with 2 or more folds yields only
         fold --fold (default 0) of the same 5-fold split the full run makes.
         A split over fewer molecules -- a split inside a training fold --
         passes through unchanged. Each cut is printed with the file and line
         that asked for it, so the log shows exactly what was cut.
  level  dataset_policy's level grids are replaced by the one level asked for:
         _DEFAULT_NOISE_LEVELS, _NOISE_LEVELS, CENSORING_LEVELS,
         CENSORING_LEVELS_DETECT, CLASSIFICATION_NOISE_LEVELS, and
         noise_helpers.NOISY_SEARCH_LEVELS. A level an experiment writes in its
         own file is not reached, and the relative grids (RELATIVE_*_LEVELS)
         are left alone.
  condition  forwarded to run_all exactly as run_dataset.py forwards it. An
         experiment that names its own condition is run without one.

The hybrid search is untouched: it reads the precompute through the same calls
as a full run, and the engine makes its own folds, not through cv_split.

⛔ The result JSON still says n_splits=5 in its input_fingerprint, because the
config is the full run's. So ALWAYS write to a separate --output directory,
never tests/results/noise, or a later full run resumes on a one-fold file.

Usage:
    python tests/slurm_scripts/noise_prelim/run_one.py caco2 n07 \
        --condition outlier_p10 --output /path/prelim/run1
    python tests/slurm_scripts/noise_prelim/run_one.py caco2 n18 --output ...
    python tests/slurm_scripts/noise_prelim/run_one.py caco2 n07 --all-folds --all-levels ...

Exit code: 0 when the experiment returned a result or a deliberate skip, 1 when
run_all caught an exception from it (run_all itself catches and carries on, so
without this check a crashed experiment exits 0).
"""
import argparse
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))


def _caller():
    """file:line of the first frame outside this file and _shared.py."""
    for fr in inspect.stack()[2:]:
        base = os.path.basename(fr.filename)
        if base not in ('run_one.py', '_shared.py', 'contextlib.py'):
            return f"{base}:{fr.lineno}"
    return '?'


def install_fold_cut(S, fold, state):
    """Wrap _shared.cv_split and make_cv_splitter so a whole-dataset split
    yields fold `fold` only. `state['n']` is the dataset's molecule count, set
    when run_all loads the data; until then nothing is cut."""
    orig_cv_split = S.cv_split
    orig_make = S.make_cv_splitter

    def _should_cut(n_rows, n_splits):
        return (state['n'] is not None and n_rows == state['n']
                and n_splits is not None and int(n_splits) >= 2)

    def cv_split(groups, n_splits, X, labels, seed=42, mode=None, *,
                 task_type=None):
        it = orig_cv_split(groups, n_splits, X, labels, seed=seed, mode=mode,
                           task_type=task_type)
        if not _should_cut(len(X), n_splits):
            yield from it
            return
        where = _caller()
        for i, (tr, va) in enumerate(it):
            if i == fold:
                state['cuts'].append({'function': 'cv_split', 'caller': where,
                                      'n_splits': int(n_splits), 'kept_fold': i,
                                      'n_train': int(len(tr)),
                                      'n_val': int(len(va))})
                print(f"  [run_one] cv_split from {where}: {n_splits} folds over "
                      f"{len(X)} molecules -> fold {i} only "
                      f"({len(tr)} training, {len(va)} held out)")
                yield tr, va
                return
        raise RuntimeError(f"run_one: fold {fold} does not exist in a "
                           f"{n_splits}-fold split (called from {where})")

    class _OneFold:
        def __init__(self, inner, n_splits, where):
            self._inner, self._k, self._where = inner, n_splits, where

        def get_n_splits(self, *a, **k):
            return 1

        def split(self, X, y=None, groups=None):
            args = (X, y, groups) if groups is not None else (X, y)
            for i, (tr, va) in enumerate(self._inner.split(*args)):
                if i == fold:
                    state['cuts'].append({'function': 'make_cv_splitter',
                                          'caller': self._where,
                                          'n_splits': int(self._k),
                                          'kept_fold': i,
                                          'n_train': int(len(tr)),
                                          'n_val': int(len(va))})
                    print(f"  [run_one] make_cv_splitter from {self._where}: "
                          f"{self._k} folds -> fold {i} only")
                    yield tr, va
                    return

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def make_cv_splitter(n_folds, groups=None, seed=42, *, task_type=None):
        splitter, use_groups = orig_make(n_folds, groups, seed,
                                         task_type=task_type)
        n_rows = None if groups is None else len(groups)
        if not _should_cut(n_rows, n_folds):
            return splitter, use_groups
        return _OneFold(splitter, n_folds, _caller()), use_groups

    S.cv_split = cv_split
    S.make_cv_splitter = make_cv_splitter


def default_level(DP, dataset, condition, is_clf):
    if is_clf:
        return 0.2
    if condition == 'censoring':
        return 0.25
    return float(DP.reference_noise_level(dataset))


def install_level(DP, level):
    one = (float(level),)
    patched = {}
    for name in ('_DEFAULT_NOISE_LEVELS', 'CENSORING_LEVELS',
                 'CENSORING_LEVELS_DETECT', 'CLASSIFICATION_NOISE_LEVELS'):
        patched[name] = [getattr(DP, name), one]
        setattr(DP, name, one)
    DP._NOISE_LEVELS = {k: one for k in DP._NOISE_LEVELS}
    return patched


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dataset')
    ap.add_argument('experiment', help="e.g. n07")
    ap.add_argument('--condition', default=None,
                    help='noise condition; ignored with a note for an '
                         'experiment that names its own')
    ap.add_argument('--level', type=float, default=None,
                    help='the one level; default 0.5 of the label SD '
                         '(reference_noise_level), 0.25 for censoring, 0.2 '
                         'flip rate for classification')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--all-folds', action='store_true')
    ap.add_argument('--all-levels', action='store_true')
    ap.add_argument('--output', required=True,
                    help='result directory; never tests/results/noise')
    ap.add_argument('--pruning-mode', default='calibrated')
    ap.add_argument('--manifest', default=None,
                    help='where to write what this run cut; default '
                         '<output>/_run_one_manifest.json')
    args = ap.parse_args()

    out = os.path.abspath(args.output)
    if os.path.abspath(os.path.join(REPO, 'tests', 'results', 'noise')) in (
            out, os.path.dirname(out)):
        sys.exit("run_one: --output must not be tests/results/noise; a one-fold "
                 "file there would be resumed by a full run.")

    from kirby.similarity.distance_config import (
        DEFAULT_DISTANCE_METRIC, set_distance_metric)
    set_distance_metric(os.environ.get('KIRBY_DISTANCE_METRIC')
                        or DEFAULT_DISTANCE_METRIC)

    import experiments._shared as S
    import experiments.dataset_policy as DP
    from experiments._shared import registry_lookup

    is_clf = getattr(registry_lookup(args.dataset), 'task_type', None) == 'classification'
    state = {'n': None, 'cuts': []}

    level = None
    level_patch = {}
    if not args.all_levels:
        level = (args.level if args.level is not None
                 else default_level(DP, args.dataset, args.condition, is_clf))
        level_patch = install_level(DP, level)
    if not args.all_folds:
        install_fold_cut(S, args.fold, state)

    import experiments.noise.noise_helpers as NH
    if not args.all_levels:
        NH.NOISY_SEARCH_LEVELS = (float(level),)

    # The molecule count, from the first load run_all makes.
    _orig_load = NH.load_data

    def _load(*a, **k):
        d = _orig_load(*a, **k)
        if state['n'] is None and isinstance(d, dict) and d.get('labels') is not None:
            state['n'] = int(len(d['labels']))
            print(f"  [run_one] {args.dataset}: {state['n']} molecules; a split "
                  f"over all of them is cut to fold {args.fold}"
                  if not args.all_folds else
                  f"  [run_one] {args.dataset}: {state['n']} molecules; all folds")
        return d
    NH.load_data = _load

    takes = NH.HONOURS_NOISE_CONDITION | NH.SWEEPS_ALL_CONDITIONS
    cond = args.condition
    if cond and args.experiment not in takes:
        print(f"  [run_one] {args.experiment} names its own condition in its "
              f"file; running it without --condition {cond}")
        cond = None

    print(f"[run_one] dataset={args.dataset} experiment={args.experiment} "
          f"condition={cond or 'experiment default'} "
          f"level={'all' if args.all_levels else level} "
          f"fold={'all' if args.all_folds else args.fold} "
          f"pruning={args.pruning_mode} output={out}")
    print(f"  KIRBY_NO_COLD_SEARCH={os.environ.get('KIRBY_NO_COLD_SEARCH', 'unset')} "
          f"KIRBY_LGBM_THREADS={os.environ.get('KIRBY_LGBM_THREADS', 'unset')} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} "
          f"KIRBY_NOISE_CACHE_REPORT={os.environ.get('KIRBY_NOISE_CACHE_REPORT', 'unset')}")

    t0 = time.time()
    results = NH.run_all(dataset_name=args.dataset, output_dir=out,
                         experiments=[args.experiment],
                         pruning_mode=args.pruning_mode, force_rerun=True,
                         noise_condition=cond)
    r = (results or {}).get(args.experiment)
    status = ('error' if isinstance(r, dict) and 'error' in r
              else 'skipped' if isinstance(r, dict) and r.get('skipped')
              else 'missing' if r is None else 'ok')

    manifest = {
        'dataset': args.dataset, 'experiment': args.experiment,
        'experiment_dir': NH._experiment_full_name(args.experiment),
        'condition': cond, 'condition_asked': args.condition,
        'level': level, 'levels_patched': {k: list(v[1]) for k, v in level_patch.items()},
        'fold': None if args.all_folds else args.fold,
        'n_molecules': state['n'], 'fold_cuts': state['cuts'],
        'pruning_mode': args.pruning_mode,
        'status': status,
        'error': r.get('error') if isinstance(r, dict) else None,
        'seconds': round(time.time() - t0, 1),
        'env': {k: os.environ.get(k) for k in (
            'KIRBY_NO_COLD_SEARCH', 'KIRBY_LGBM_THREADS', 'OMP_NUM_THREADS',
            'KIRBY_NOISE_CACHE', 'KIRBY_NOISE_CACHE_REPORT', 'KIRBY_CACHE_DIR',
            'KIRBY_DISTANCE_METRIC', 'KIRBY_NOISE_CONTROLS')},
    }
    mpath = args.manifest or os.path.join(out, '_run_one_manifest.json')
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, 'w') as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"[run_one] status={status}  fold cuts={len(state['cuts'])}  "
          f"manifest={mpath}")
    if not args.all_folds and not state['cuts'] and status == 'ok':
        print("[run_one] NOTE: no whole-dataset split was cut. This experiment "
              "does not take its folds from _shared.cv_split/make_cv_splitter, "
              "so it ran every fold.")
    sys.exit(1 if status in ('error', 'missing') else 0)


if __name__ == '__main__':
    main()
