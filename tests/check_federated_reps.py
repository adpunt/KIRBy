#!/usr/bin/env python3
"""Say whether every federated experiment can find the representations it needs.

Reads caches and dataset lists. Loads no molecules, computes no representation,
fits no model — so it costs seconds and can be run on a login node.

Exit code 0 only when every dataset has every representation every federated
experiment would ask it for. A non-zero exit means submitting would spend
cluster time and record a failure, or quietly run a smaller experiment than the
one that was designed.

Why it exists: on 2026-09-12 fed03's Part B stopped using seven representation
names written into the file and started reading each dataset's own list. That is
the right thing for the experiment and it means the set of representations the
run needs is no longer knowable by eye. fed02 has the same property — it asks
for five named representations and substitutes from the dataset's pool wherever
one is absent.

Usage, on the server:

    KIRBY_CACHE_DIR=/path/to/cache python tests/check_federated_reps.py
    KIRBY_CACHE_DIR=/path/to/cache python tests/check_federated_reps.py --datasets caco2 logd
    python tests/check_federated_reps.py --json      # machine-readable

To gate a submission:

    python tests/check_federated_reps.py && sbatch tests/slurm_scripts/<script>.sh
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


# The three datasets the federated drivers run on. These are the names in
# tests/experiments/federated/{caco2,logd,herg}.py.
FEDERATED_DATASETS = ['caco2', 'logd', 'lhasa_herg']

# fed05 ignores the dataset it is handed and loads these three itself.
FED05_SILOS = ['ddg_combined', 'abagym_EGFR', 'flab_thermo_tresanco']

# run_all skips the pool lookup for these two names by design — the Lhasa route
# loads its own molecules and searches its own representations from scratch.
LHASA_DATASETS = {'herg', 'lhasa_herg'}


def _cache_root() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _rep_cached(dataset_name: str, rep_name: str) -> bool:
    """Is this representation already on disk for this dataset?

    Molecular representations land at
    $KIRBY_CACHE_DIR/rep_cache/<dataset>/<rep>.npy — the layout _cache_rep in
    tests/experiments/_shared.py writes and reads.
    """
    return (_cache_root() / 'rep_cache' / dataset_name /
            f'{rep_name}.npy').exists()


def _pool_for(dataset_name: str):
    """The representations this dataset's pruned pool holds, or None."""
    from experiments.complexity.complexity_helpers import DATASET_REPS
    entry = DATASET_REPS.get(dataset_name)
    if entry is None:
        return None
    return list(entry.get('general') or []) if isinstance(entry, dict) else list(entry)


def _fed02_wants(dataset_name, pool):
    """The five representations fed02 would put one per silo."""
    from experiments.federated.fed02_rep_diversity_silos import _get_silo_reps
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return sorted(_get_silo_reps(dataset_name, pool).values()), None
        except Exception as e:
            return [], f'{type(e).__name__}: {e}'


def _fed03b_wants(pool):
    """The representations fed03's Part B would sort into its three families."""
    from experiments.federated.fed03_synthetic_private import (
        assign_rep_families)
    families, unclassified = assign_rep_families(pool)
    wanted = sorted({r for v in families.values() for r in v})
    return wanted, families, unclassified


def check(datasets, as_json=False):
    rows = []
    problems = 0

    for ds in datasets:
        row = {'dataset': ds, 'problems': []}
        pool = _pool_for(ds)

        if pool is None:
            if ds in LHASA_DATASETS:
                # Exempt by design. run_all skips the pool lookup for these
                # two names (federated_helpers.py, "Resolve rep list"): the
                # Lhasa route loads its own molecules and searches its own
                # representations from scratch, so having no entry in
                # DATASET_REPS is correct, not a gap.
                row['note'] = ('Lhasa route — loads its own molecules and '
                               'searches from scratch; no pruned pool wanted')
                rows.append(row)
                continue
            row['problems'].append(
                'not in DATASET_REPS — no pruned pool, so fed02 and fed03 '
                'Part B have nothing to read')
            rows.append(row)
            problems += 1
            continue

        row['pool'] = pool
        row['pool_cached'] = sorted(r for r in pool if _rep_cached(ds, r))
        row['pool_absent'] = sorted(r for r in pool if not _rep_cached(ds, r))

        # fed01 and fed04 take the whole pool.
        if row['pool_absent']:
            row['problems'].append(
                f"fed01/fed04 read the whole pool; "
                f"{len(row['pool_absent'])} of {len(pool)} are not cached: "
                f"{row['pool_absent']}")

        # fed02: five silos, one representation each, all distinct.
        f2, f2_err = _fed02_wants(ds, pool)
        row['fed02_silo_reps'] = f2
        if f2_err:
            row['problems'].append(f'fed02 silo assignment raises — {f2_err}')
        else:
            f2_absent = [r for r in f2 if not _rep_cached(ds, r)]
            if f2_absent:
                row['problems'].append(
                    f'fed02 silos need {f2_absent}, not cached')
            if len(f2) < 5:
                row['problems'].append(
                    f'fed02 resolved only {len(f2)} of 5 silos — it would run '
                    f'a smaller experiment than the one designed')

        # fed03 Part B: every representation sorted into a family.
        f3, families, unclassified = _fed03b_wants(pool)
        row['fed03b_families'] = {k: v for k, v in families.items()}
        row['fed03b_unclassified'] = unclassified
        f3_absent = [r for r in f3 if not _rep_cached(ds, r)]
        if f3_absent:
            row['problems'].append(
                f'fed03 Part B needs {f3_absent}, not cached')
        missing_fam = [f for f in ('structural', 'physicochemical', 'learned')
                       if f not in families]
        if missing_fam:
            row['problems'].append(
                f'fed03 Part B has no {missing_fam} representation in this '
                f"dataset's pool — that silo would not exist")
        if unclassified:
            row['problems'].append(
                f'fed03 Part B cannot sort {unclassified} into any family — '
                f'they would sit in no silo. Add them to REP_FAMILIES in '
                f'fed03_synthetic_private.py')

        problems += len(row['problems'])
        rows.append(row)

    # fed05 loads its own three antibody sets, by name, from the registry.
    fed05 = {'silos': {}, 'problems': []}
    try:
        from antibody_baseline_survey import DATASET_REGISTRY
        for s in FED05_SILOS:
            ok = s in DATASET_REGISTRY
            fed05['silos'][s] = 'in the registry' if ok else 'NOT IN REGISTRY'
            if not ok:
                fed05['problems'].append(
                    f"fed05 silo '{s}' is not a key in DATASET_REGISTRY — it "
                    f"would raise and fed05 would run on fewer silos")
    except Exception as e:
        fed05['problems'].append(
            f'could not read the antibody registry — {type(e).__name__}: {e}')
    problems += len(fed05['problems'])

    if as_json:
        print(json.dumps({'datasets': rows, 'fed05': fed05,
                          'cache_root': str(_cache_root()),
                          'total_problems': problems}, indent=2))
        return 1 if problems else 0

    print(f"Cache root: {_cache_root()}")
    print(f"Checking {len(datasets)} dataset(s) for fed01-fed04, "
          f"plus fed05's own three silos\n")

    for row in rows:
        mark = 'OK     ' if not row['problems'] else 'PROBLEM'
        ds = row['dataset']
        if 'pool' not in row:
            print(f"{mark} {ds}")
            if row.get('note'):
                print(f"         {row['note']}")
            for p in row['problems']:
                print(f"         {p}")
            print()
            continue
        print(f"{mark} {ds}  "
              f"{len(row['pool_cached'])}/{len(row['pool'])} pool "
              f"representations cached")
        print(f"         fed02 silos: {row['fed02_silo_reps']}")
        for fam in ('structural', 'physicochemical', 'learned'):
            got = row['fed03b_families'].get(fam, [])
            print(f"         fed03B {fam:16s} {got if got else 'EMPTY'}")
        for p in row['problems']:
            print(f"         -> {p}")
        print()

    print("fed05 silos (loaded by name, whatever dataset it is handed):")
    for s, state in fed05['silos'].items():
        print(f"         {s:26s} {state}")
    for p in fed05['problems']:
        print(f"         -> {p}")
    print()

    if problems:
        print(f"{problems} problem(s). Do NOT submit — the run would either "
              f"fail or quietly answer a smaller question than the one "
              f"designed.")
    else:
        print("No problems. Every federated experiment can find what it needs.")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(
        description='Gate a federated submission on representation availability')
    ap.add_argument('--datasets', nargs='+', default=FEDERATED_DATASETS,
                    help=f'default: {" ".join(FEDERATED_DATASETS)}')
    ap.add_argument('--json', action='store_true',
                    help='machine-readable output')
    args = ap.parse_args()
    sys.exit(check(args.datasets, as_json=args.json))


if __name__ == '__main__':
    main()
