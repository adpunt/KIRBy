#!/usr/bin/env python3
"""How far along f34_importance_sweep is: tasks finished out of the task list.

A task is finished when every (coefficient, importance variant) it owns has a
line in its output file: 7 variants per coefficient, and a hybrid task ('all')
owns 3 or 4 coefficients. Reads files only; safe to run on the login node.

Usage:
    python tests/experiments/framework/f34_importance_progress.py [task list]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'tests' / 'results' / 'framework' / 'f34_importance_sweep'
N_VARIANTS = 7
N_COEFFS = {'binary': 3, 'count': 3, 'continuous': 4}

tasks = Path(sys.argv[1] if len(sys.argv) > 1 else
             ROOT / 'f34_imp_tasks.txt').read_text().split('\n')
tasks = [t.split() for t in tasks if t.strip()]
done, started, per_ds, todo_ds = 0, 0, Counter(), Counter()
for ds, rep, fi, c in tasks:
    f = OUT / ds / f'{rep}__fold{fi}__{c}.jsonl'
    keys, kind = set(), None
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            keys.add((r['coefficient'], r['importance']))
            kind = r['kind']
    need = N_VARIANTS * (N_COEFFS.get(kind, 4) if c == 'all' else 1)
    if keys and len(keys) >= need:
        done += 1
        per_ds[ds] += 1
    else:
        todo_ds[ds] += 1
        started += bool(keys)
print(f'{done} of {len(tasks)} tasks finished ({100 * done / len(tasks):.0f}%), '
      f'{started} more partly written')
for ds in sorted(set(per_ds) | set(todo_ds)):
    print(f'  {ds:10s} {per_ds[ds]} finished, {todo_ds[ds]} to go')
