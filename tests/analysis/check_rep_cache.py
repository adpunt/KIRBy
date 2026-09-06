#!/usr/bin/env python
"""Inventory the cached antibody representations.

    python tests/analysis/check_rep_cache.py

Reads $KIRBY_CACHE_DIR/reps (or ~/.cache/kirby/reps). Each subdirectory is one
sequence list, named by a hash of it, holding one .npy per representation.

The geometry survey does not compute representations. It loads aac, blosum,
onehot and paratope itself and takes everything else from this cache, skipping
anything absent with only a warning. So when a survey reports 2 of 9 base
representations, this says whether the other seven are missing everywhere or
are cached under a sequence list that no longer matches.

Stdlib only — runs on a login node.
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

SELF_COMPUTED = {'aac', 'blosum', 'onehot', 'paratope'}


def main():
    env_dir = os.environ.get('KIRBY_CACHE_DIR')
    root = Path(env_dir or (Path.home() / '.cache' / 'kirby')) / 'reps'

    if not env_dir:
        print("!" * 62)
        print("KIRBY_CACHE_DIR is NOT SET, so this is reading")
        print(f"  {root}")
        print("Jobs on the server do not use that. setup.sh sets")
        print("  KIRBY_CACHE_DIR=/data/stat-ecr/scat9264/.cache/kirby")
        print("and every job reads and writes there. Run `. ./setup.sh` first,")
        print("or the inventory below is of a different cache than the one the")
        print("survey used.")
        print("!" * 62)
        print()

    if not root.exists():
        print(f"No representation cache at {root}")
        return 1

    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"{len(dirs)} cached sequence lists under {root}\n")

    counts = Counter()
    per_dir = {}
    for p in dirs:
        names = sorted(f.stem for f in p.glob('*.npy'))
        per_dir[p.name] = names
        counts.update(names)

    print("Representation                 in how many sequence lists")
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        tag = '  (survey computes this itself)' if name in SELF_COMPUTED else ''
        print(f"  {name:<28} {n:5d}{tag}")

    by_size = defaultdict(list)
    for d, names in per_dir.items():
        by_size[len(names)].append(d)
    print("\nHow many representations each sequence list holds:")
    for k in sorted(by_size, reverse=True):
        print(f"  {k:3d} representations: {len(by_size[k])} sequence lists")

    only_cheap = [d for d, n in per_dir.items()
                  if n and set(n) <= SELF_COMPUTED]
    if only_cheap:
        print(f"\n{len(only_cheap)} sequence lists hold ONLY representations the "
              f"survey computes itself. Those are the ones that report 2 of 9.")
    empty = [d for d, n in per_dir.items() if not n]
    if empty:
        print(f"{len(empty)} sequence lists are empty directories.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
