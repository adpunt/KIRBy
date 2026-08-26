"""Audit which molecule list each MCS pair cache is numbered against.

Three different molecule lists exist per dataset name and their row numbers are
NOT interchangeable:

  loader     tests/find_best_hybrid_molecular.load_dataset — what every
             framework experiment trains on, InChIKey-deduplicated
  sim_npz    tests/experiments/similarity_metrics/sim_<ds>.npz — the July study
             arrays, uncapped, what structural_yardstick.py samples over
  prepared   _data.load_prepared — the same arrays with logd capped to a
             scaffold subsample, what mcs_cache.py numbered its pairs against

For a named dataset this prints the size of each list, which pair files exist
(the study's mcs_pairs_<ds>.npz and the framework's $KIRBY_CACHE_DIR/mcs_pairs/
entries), and for every pair file whether its stored SMILES ordering hash proves
it belongs to one of the three. A file with no stored hash cannot prove anything
and must be rebuilt.

Usage:
    python -m kirby.similarity.mcs_audit herg_ki logd
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

from kirby.similarity.mcs import _mcs_cache_dir, _ordering_hash


def _repo_tests_dir():
    """tests/ of the checkout this file lives in (src/kirby/similarity/...)."""
    return Path(__file__).resolve().parents[3] / 'tests'


def _loader_smiles(dataset, tests_dir):
    """SMILES the framework loader returns (InChIKey-deduplicated)."""
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from find_best_hybrid_molecular import load_dataset
    smiles, _labels, _task = load_dataset(dataset)
    return [str(s) for s in smiles]


def _study_smiles(dataset, tests_dir):
    """(sim_npz ordering, prepared ordering) from the July study arrays.

    The prepared ordering applies _data's own scaffold subsample under _data's
    own cap rule (logd only), without materialising load_prepared's n x n
    Tanimoto matrix.
    """
    sim_dir = tests_dir / 'experiments' / 'similarity_metrics'
    npz = sim_dir / f'sim_{dataset}.npz'
    if not npz.exists():
        return None, None
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    from _data import _scaffold_subsample, CAP_N

    d = np.load(npz, allow_pickle=True)
    smiles = [str(s) for s in d['smiles']]
    groups = np.asarray(d['groups'])
    if dataset == 'logd' and CAP_N and len(smiles) > CAP_N:
        idx = _scaffold_subsample(groups, CAP_N)
    else:
        idx = np.arange(len(smiles))
    return smiles, [smiles[i] for i in idx]


def _report_pair_file(path, orderings):
    """Print one pair file's size, stored hash and which ordering it matches."""
    z = np.load(path)
    I, J = z['I'], z['J']
    max_ij = max(int(np.max(I)), int(np.max(J))) if len(I) else -1
    print(f'  {path}')
    print(f'    {len(I)} pairs, highest row number {max_ij} '
          f'(needs a list of at least {max_ij + 1} molecules)')
    stored = (str(np.asarray(z['smiles_sha']).item())
              if 'smiles_sha' in z.files else None)
    if stored is None:
        print('    stored SMILES ordering: NONE — predates the check, '
              'must be REBUILT, not trusted')
    else:
        print(f'    stored SMILES ordering: {stored}')
    for label, smiles in orderings.items():
        if smiles is None:
            print(f'      vs {label:9s} : list unavailable')
            continue
        order = _ordering_hash(smiles)
        if stored is None:
            fits = 'fits by size only' if max_ij < len(smiles) else 'IMPOSSIBLE'
            verdict = f'UNPROVEN ({fits})'
        elif stored == order:
            verdict = 'MATCH'
        else:
            verdict = 'MISMATCH'
        print(f'      vs {label:9s} ({len(smiles):5d} molecules, {order}): {verdict}')


def audit(dataset, tests_dir):
    print(f'\n=== {dataset} ===')

    try:
        loader = _loader_smiles(dataset, tests_dir)
    except Exception as e:                      # network / missing loader dep
        loader = None
        print(f'  framework loader unavailable: {type(e).__name__}: {e}')

    sim_npz, prepared = _study_smiles(dataset, tests_dir)

    orderings = {'loader': loader, 'sim_npz': sim_npz, 'prepared': prepared}
    print('  molecule lists:')
    for label, smiles in orderings.items():
        if smiles is None:
            print(f'    {label:9s} : (unavailable)')
        else:
            print(f'    {label:9s} : {len(smiles):5d} molecules  '
                  f'ordering {_ordering_hash(smiles)}')

    print('  pair files:')
    found = False
    study_file = (tests_dir / 'experiments' / 'similarity_metrics'
                  / f'mcs_pairs_{dataset}.npz')
    if study_file.exists():
        found = True
        _report_pair_file(study_file, orderings)
    cache_dir = _mcs_cache_dir()
    for path in sorted(cache_dir.glob(f'{dataset}_*.npz')):
        found = True
        _report_pair_file(path, orderings)
    if not found:
        print(f'    none — no {study_file.name} and nothing under {cache_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('datasets', nargs='*', default=['herg_ki', 'logd'])
    ap.add_argument('--tests-dir', default=None,
                    help='checkout tests/ directory (default: alongside src/)')
    args = ap.parse_args()

    tests_dir = Path(args.tests_dir) if args.tests_dir else _repo_tests_dir()
    print(f'tests dir      : {tests_dir}')
    print(f'mcs pair cache : {_mcs_cache_dir()}  '
          f'(KIRBY_CACHE_DIR={os.environ.get("KIRBY_CACHE_DIR", "unset")})')
    for ds in args.datasets:
        audit(ds, tests_dir)


if __name__ == '__main__':
    main()
