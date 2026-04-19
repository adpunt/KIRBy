#!/usr/bin/env python3
"""Plan precompute submissions from DATASET_REGISTRY.

Reads src/kirby/registry.py to get each dataset's `n_samples`, then picks a
partition + memory + time budget and prints `sbatch` commands. By default,
only PRINTS commands — nothing is submitted until you pipe to a shell.

This is a planner, not a runner. The actual submission is yours.

Usage (local or server, no compute):
    # Dry-run, print all commands
    python tests/scripts/plan_precompute.py

    # Filter to one domain
    python tests/scripts/plan_precompute.py --domain molecular

    # Filter to one size band
    python tests/scripts/plan_precompute.py --band M

    # Exclude nucleotides (out-of-scope per CACHE_FIX_PLAN)
    python tests/scripts/plan_precompute.py --exclude-nucleotide

    # Include only specific datasets
    python tests/scripts/plan_precompute.py caco2 logd herg_ki

    # Generate a SLURM driver script
    python tests/scripts/plan_precompute.py --exclude-nucleotide \
        > precompute_batch.sh
    # Review, then:
    #   bash precompute_batch.sh

The heuristic (partition + --mem + --time) is a STARTING POINT. It picks
up `n_samples` from the registry when set, and falls back to a sensible
conservative guess when the dataset has no size recorded (most antibody
entries). If a dataset times out, re-submit it in a larger band.

Heuristic rules:

  Band  | n_samples      | reps hint | partition | --mem | --time
  ------|----------------|-----------|-----------|-------|---------
  XS    | <200           | any       | short     | 32G   | 12:00:00
  S     | 200 – 1000     | any       | short     | 64G   | 12:00:00
  M     | 1001 – 3000    | any       | medium    | 128G  | 2-00:00:00
  L     | 3001 – 10000   | any       | medium    | 128G  | 2-00:00:00
  XL    | >10000 OR TCGA | any       | long      | 256G  | 7-00:00:00
  NT    | nucleotide     | long reps | long      | 128G  | 7-00:00:00

Override per-dataset with --force (comma-separated band=, e.g.
`--force adme_hlm=XL`) when you have evidence a dataset needs more.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Make kirby importable
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent / 'src'
sys.path.insert(0, str(_SRC))

from kirby.registry import DATASET_REGISTRY, DatasetInfo


# =============================================================================
# SIZE → PARTITION HEURISTIC
# =============================================================================

@dataclass(frozen=True)
class ResourcePlan:
    band: str
    partition: str
    mem: str
    time: str
    rationale: str


# Datasets where empirical sacct runtimes have exceeded the band the
# size heuristic would pick. Source: scat9264 history, 2026-04-03 → -17.
# Additions here should cite the run ID and observed Elapsed.
_KNOWN_LONG_RUNNERS: Dict[str, str] = {
    # flab_thermo_tresanco general_uncapped took 8h 15min (11707716)
    # — well inside short 12h but ~4h shy of it. Upshift to L as a
    # buffer against `capped` conditions going longer.
    'flab_thermo_tresanco': 'L',
    # humanness_therapeutic n=564 (would be S band by size) but has
    # 26 general reps + 21 top_tier reps + GP enabled (n<6000 cap).
    # Consolidated _all_standard job hit 12h limit on `short` before
    # completing top+borderline_uncapped (job 11784359, 2026-04-19).
    # Upshift to M so it gets medium/2d/128G.
    'humanness_therapeutic': 'M',
    # qm9 general_none 2h 10min (11707691) — size hits L naturally,
    # no override needed.
}


# Band → partition/mem/time. Calibrated against scat9264's sacct data from
# 2026-04-03 to 2026-04-17: completed jobs ran 10 min – 8h 15min, well
# inside `short`'s 12h limit. Memory peaks unverified (MaxRSS truncated)
# but 128G fits the largest molecular/antibody baselines by a 2× margin
# on prior 256G runs with no reported OOMs.
_BANDS = {
    'XS': ResourcePlan('XS', 'short',  '64G',  '12:00:00',   '<200 samples'),
    'S':  ResourcePlan('S',  'short',  '128G', '12:00:00',   '200-1k samples'),
    # M promoted to medium 2d (2026-04-18): in consolidated mode a single
    # job runs 6 standard conditions sequentially. Per-condition sacct
    # shows capped conditions alone take 4-6h on M-band datasets (caco2,
    # logd, herg_ki). Summed across 6 conditions, 12h `short` is too
    # tight for reliable completion.
    'M':  ResourcePlan('M',  'medium', '128G', '2-00:00:00', '1k-3k samples (safe for consolidated 6-condition runs)'),
    'L':  ResourcePlan('L',  'medium', '128G', '2-00:00:00', '3k-10k samples (safety buffer)'),
    'XL': ResourcePlan('XL', 'medium', '256G', '2-00:00:00', '>10k samples OR TCGA'),
    'NT': ResourcePlan('NT', 'long',   '128G', '2-00:00:00', 'nucleotide (long reps)'),
    'FT': ResourcePlan('FT', 'long',   '128G', '2-00:00:00', 'fine-tuning (DNN training)'),
}


def _pick_band(info: DatasetInfo) -> str:
    """Decide a band from registry metadata.

    If n_samples is unknown, treat as 'S' (medium-small) — the most
    conservative guess that still fits a 12-hour job for most antibody
    inter-family datasets.
    """
    if info.domain == 'nucleotide':
        return 'NT'
    if info.domain == 'tcga':
        return 'XL'
    # Known-long-runner override from sacct data 2026-04-03 to -17.
    if info.name in _KNOWN_LONG_RUNNERS:
        return _KNOWN_LONG_RUNNERS[info.name]
    n = info.n_samples
    if n is None:
        # Unknown size defaults to M (short 12h 128G). If the sacct
        # history shows a specific dataset needing more, add it to
        # _KNOWN_LONG_RUNNERS rather than blanket-upshifting.
        return 'M'
    if n < 200:
        return 'XS'
    if n <= 1000:
        return 'S'
    if n <= 3000:
        return 'M'
    if n <= 10000:
        return 'L'
    return 'XL'


# =============================================================================
# SBATCH COMMAND GENERATION
# =============================================================================

# Conditions precompute_hybrids.py writes. Standard = 6, distance = 4.
_STANDARD_CONDITIONS = [
    'general_none', 'general_uncapped', 'general_capped',
    'top_none', 'top_uncapped', 'top_capped',
]
_DISTANCE_CONDITIONS = [
    'general_distance_none', 'general_distance_uncapped',
    'top_distance_none', 'top_distance_uncapped',
]


def _sbatch_line(dataset: str, condition: str, plan: ResourcePlan,
                 distance: bool, cluster: Optional[str] = None) -> str:
    """Build one sbatch line for a single (dataset, condition).

    `cluster`, when set (e.g. 'htc' or 'arc' or 'all'), adds
    `--clusters=` so the job routes to that SLURM cluster. $HOME and
    $DATA are shared between Oxford ARC and HTC, so the cache and repo
    paths work from either.
    """
    distance_flag = ' --distance' if distance else ''
    job_name = f'pc_{dataset}_{condition}'
    wrap = (
        f'cd /data/stat-cadd/scat9264/KIRBy && '
        f'export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && '
        f'export PYTHONWARNINGS=ignore && '
        f'export TRANSFORMERS_VERBOSITY=error && '
        f'python tests/precompute_hybrids.py {dataset} '
        f'--condition {condition} --speed full{distance_flag}'
    )
    cluster_flag = f' --clusters={cluster}' if cluster else ''
    return (
        f'sbatch{cluster_flag} --job-name={job_name} '
        f'--partition={plan.partition} '
        f'--mem={plan.mem} --time={plan.time} --cpus-per-task=4 '
        f'--wrap="{wrap}"'
    )


def _sbatch_consolidated(dataset: str, plan: ResourcePlan,
                         distance: bool,
                         cluster: Optional[str] = None) -> str:
    """Build one sbatch line that runs ALL conditions (standard OR distance)
    for a dataset sequentially in one SLURM job.

    precompute_hybrids.py without --condition runs all applicable tier ×
    strategy combinations inside one python process. For small and
    medium datasets this fits inside the 12h `short` window easily and
    collapses 6 sbatch calls (or 4 for distance) to 1.

    Uses a separate job per mode (standard vs distance) because the
    precompute script treats --distance as a global switch: the two
    modes can't be interleaved in one invocation without a code change.
    """
    group = 'distance' if distance else 'standard'
    distance_flag = ' --distance' if distance else ''
    job_name = f'pc_{dataset}_all_{group}'
    wrap = (
        f'cd /data/stat-cadd/scat9264/KIRBy && '
        f'export KIRBY_SKIP_INSTALLS=1 && . ./setup.sh && '
        f'export PYTHONWARNINGS=ignore && '
        f'export TRANSFORMERS_VERBOSITY=error && '
        f'python tests/precompute_hybrids.py {dataset} '
        f'--speed full{distance_flag}'
    )
    cluster_flag = f' --clusters={cluster}' if cluster else ''
    return (
        f'sbatch{cluster_flag} --job-name={job_name} '
        f'--partition={plan.partition} '
        f'--mem={plan.mem} --time={plan.time} --cpus-per-task=4 '
        f'--wrap="{wrap}"'
    )


def plan_dataset(dataset: str, info: DatasetInfo,
                 forced: Optional[str] = None,
                 include_distance: bool = True,
                 conditions: Optional[List[str]] = None,
                 cluster: Optional[str] = None,
                 consolidated: bool = False) -> List[str]:
    """Return the list of sbatch lines for the selected conditions.

    When `conditions` is None: all 6 standard + 4 distance (unless
    include_distance=False). When a list is given, only those conditions
    are emitted — used for 1-per-dataset smoke tests.

    When `consolidated=True`, emit 2 sbatch calls per dataset (one for
    all standard conditions, one for all distance) instead of 10.
    Ignores `conditions` — consolidation means "everything in each
    mode". Cuts total job count by 5x.
    """
    band = forced or _pick_band(info)
    if band not in _BANDS:
        raise ValueError(f"Unknown band: {band}. Choices: {list(_BANDS)}")
    plan = _BANDS[band]

    header = [
        f'# {dataset} ({info.domain}, '
        f'task={info.task_type}, metric={info.metric}, '
        f'n={info.n_samples}, band={band} '
        f'→ {plan.partition} / {plan.mem} / {plan.time})',
    ]

    if consolidated:
        lines = header[:]
        lines.append(_sbatch_consolidated(dataset, plan, distance=False,
                                           cluster=cluster))
        if include_distance:
            lines.append(_sbatch_consolidated(dataset, plan, distance=True,
                                               cluster=cluster))
        lines.append('')
        return lines

    # Resolve which conditions to emit (per-condition mode).
    if conditions is None:
        standard = list(_STANDARD_CONDITIONS)
        distance = list(_DISTANCE_CONDITIONS) if include_distance else []
    else:
        standard = [c for c in conditions if c in _STANDARD_CONDITIONS]
        distance = [c for c in conditions if c in _DISTANCE_CONDITIONS]

    lines = header[:]
    for cond in standard:
        lines.append(_sbatch_line(dataset, cond, plan, distance=False,
                                  cluster=cluster))
    for cond in distance:
        lines.append(_sbatch_line(dataset, cond, plan, distance=True,
                                  cluster=cluster))
    lines.append('')  # blank line between datasets
    return lines


# =============================================================================
# CLI
# =============================================================================

def _parse_forces(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    out = {}
    for item in raw.split(','):
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='*',
                        help='Specific datasets to plan. Empty = all registered.')
    parser.add_argument('--domain',
                        choices=['molecular', 'antibody', 'dta', 'tcga',
                                 'reaction', 'nucleotide'],
                        help='Filter to one domain.')
    parser.add_argument('--band',
                        choices=list(_BANDS.keys()),
                        help='Filter to one size band.')
    parser.add_argument('--exclude-nucleotide', action='store_true',
                        help='Skip nucleotide datasets (per CACHE_FIX_PLAN).')
    parser.add_argument('--skip-distance', action='store_true',
                        help='Skip the 4 distance conditions per dataset.')
    parser.add_argument('--conditions', default='',
                        help='Comma-separated subset of conditions to emit. '
                             'E.g. --conditions general_none for a true '
                             '1-job-per-dataset smoke test. Default: all.')
    parser.add_argument('--force', default='',
                        help='Per-dataset band override, e.g. '
                             '"adme_hlm=XL,flab_thermo_tresanco=M".')
    parser.add_argument('--cluster', default=None,
                        choices=['arc', 'htc', 'all'],
                        help='Pass --clusters= to sbatch so jobs route to '
                             'the given SLURM cluster. $HOME/$DATA are '
                             'shared between arc and htc so caches work '
                             'from either.')
    parser.add_argument('--consolidated', action='store_true',
                        help='Emit 2 sbatch calls per dataset instead of '
                             '10: one that runs ALL standard conditions '
                             'sequentially, one that runs ALL distance '
                             'conditions. 5x fewer jobs. Good for small/'
                             'medium datasets where the combined runtime '
                             'still fits the partition time limit.')
    parser.add_argument('--summary', action='store_true',
                        help='Print a band summary instead of sbatch commands.')
    args = parser.parse_args()

    forces = _parse_forces(args.force)

    # Validate --conditions against known condition names.
    conditions_filter: Optional[List[str]] = None
    if args.conditions:
        valid = set(_STANDARD_CONDITIONS) | set(_DISTANCE_CONDITIONS)
        requested = [c.strip() for c in args.conditions.split(',') if c.strip()]
        bad = [c for c in requested if c not in valid]
        if bad:
            print(f'ERROR: unknown conditions: {bad}', file=sys.stderr)
            print(f'Valid: {sorted(valid)}', file=sys.stderr)
            return 1
        conditions_filter = requested

    # Pick dataset list
    if args.datasets:
        unknown = [d for d in args.datasets if d not in DATASET_REGISTRY]
        if unknown:
            print(f'ERROR: unknown datasets: {unknown}', file=sys.stderr)
            print(f'Available: {sorted(DATASET_REGISTRY.keys())}', file=sys.stderr)
            return 1
        names = list(args.datasets)
    else:
        names = sorted(DATASET_REGISTRY.keys())

    # Apply filters
    def _keep(name: str) -> bool:
        info = DATASET_REGISTRY[name]
        if args.exclude_nucleotide and info.domain == 'nucleotide':
            return False
        if args.domain and info.domain != args.domain:
            return False
        if args.band:
            band = forces.get(name) or _pick_band(info)
            if band != args.band:
                return False
        # Skip the two "parent" registry entries that are only there for
        # condition_suffix lookup — not actual loaders.
        if name in ('chembl_qsar', 'cdr_flexibility'):
            return False
        return True

    names = [n for n in names if _keep(n)]

    if args.summary:
        print(f'Planning {len(names)} datasets')
        print()
        print(f'{"dataset":<35} {"band":<4} {"partition":<8} '
              f'{"mem":<5} {"time":<11} n_samples  domain')
        print('-' * 100)
        band_counts: Dict[str, int] = {}
        for name in names:
            info = DATASET_REGISTRY[name]
            band = forces.get(name) or _pick_band(info)
            plan = _BANDS[band]
            band_counts[band] = band_counts.get(band, 0) + 1
            n_str = str(info.n_samples) if info.n_samples else '?'
            print(f'{name:<35} {band:<4} {plan.partition:<8} '
                  f'{plan.mem:<5} {plan.time:<11} {n_str:>9}  {info.domain}')
        print()
        print(f'Band counts: {dict(sorted(band_counts.items()))}')
        # Per-band sbatch count
        per_band_jobs = {
            b: c * (len(_STANDARD_CONDITIONS) +
                    (0 if args.skip_distance else len(_DISTANCE_CONDITIONS)))
            for b, c in band_counts.items()
        }
        print(f'Total SLURM jobs: {sum(per_band_jobs.values())} '
              f'(per band: {per_band_jobs})')
        return 0

    # Print commands
    print(f'#!/bin/bash')
    print(f'# Precompute plan — {len(names)} datasets')
    print(f'# Generated by plan_precompute.py')
    print(f'# Review before running. Conditions per dataset: '
          f'{len(_STANDARD_CONDITIONS)} standard'
          f'{f" + {len(_DISTANCE_CONDITIONS)} distance" if not args.skip_distance else ""}')
    print(f'set -euo pipefail')
    print()
    total_jobs = 0
    for name in names:
        info = DATASET_REGISTRY[name]
        forced = forces.get(name)
        for line in plan_dataset(name, info, forced=forced,
                                 include_distance=not args.skip_distance,
                                 conditions=conditions_filter,
                                 cluster=args.cluster,
                                 consolidated=args.consolidated):
            print(line)
            if line.startswith('sbatch'):
                total_jobs += 1
    print(f'# Total sbatch calls: {total_jobs}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
