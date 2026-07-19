"""Aggregate step-1 validity results into pruning tables.

Reads $KIRBY_CACHE_DIR/validity/<ds>.json (written by tests/validity_gate.py)
and prints:
  1. a dataset keep/cut table (best rep + score + verdict), FAIL-first so the
     cut candidates are obvious;
  2. (with --reps) per-dataset rep rankings — the per-rep scores that feed the
     step-3 rep selection.

Read-only. Makes NO cut decisions — the verdict is advisory (validity_gate.py's
0.10/0.05 bands); the USER decides the cut from the numbers.

Cache location comes from KIRBY_CACHE_DIR (no hardcoded path). Run locally after
rsyncing the JSONs, or on a login node:
    python tests/process_validity_results.py
    python tests/process_validity_results.py --reps --md validity_summary.md
"""
import os
import sys
import json
import glob
import argparse
from collections import Counter

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
VDIR = os.path.join(CACHE, 'validity')


def _f(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "nan"


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(VDIR, '*.json'))):
        try:
            rows.append(json.load(open(p)))
        except Exception as e:
            print(f"  WARN: could not read {p}: {e}", file=sys.stderr)
    return rows


def dataset_table(rows):
    order = {'FAIL': 0, 'BORDERLINE': 1, 'PASS': 2}
    rows = sorted(rows, key=lambda r: (order.get(r.get('verdict'), 3),
                                       r.get('best_score') if isinstance(r.get('best_score'), (int, float)) else -9))
    out = ["| dataset | metric | best rep | best score | #reps | verdict |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['dataset']} | {r.get('metric', '?')} | {r.get('best_rep', '-')} | "
                   f"{_f(r.get('best_score'))} | {len(r.get('reps', {}))} | {r.get('verdict', '?')} |")
    return "\n".join(out)


def rep_rankings(rows):
    out = []
    for r in sorted(rows, key=lambda r: r['dataset']):
        reps = r.get('reps', {})
        ranked = sorted(reps.items(),
                        key=lambda kv: -(kv[1].get('score') if isinstance(kv[1].get('score'), (int, float)) else -9))
        out.append(f"\n### {r['dataset']}  ({r.get('metric', '?')}, verdict={r.get('verdict', '?')})")
        for rep, d in ranked:
            out.append(f"  {rep:24s} {_f(d.get('score'))}  ({d.get('model', '?')}, dim={d.get('dim', '?')})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', help='also write the tables to this markdown file')
    ap.add_argument('--reps', action='store_true', help='also print per-dataset rep rankings')
    a = ap.parse_args()

    rows = load()
    if not rows:
        print(f"No validity JSONs found in {VDIR}\n"
              f"(rsync them from the server first, or set KIRBY_CACHE_DIR).", file=sys.stderr)
        sys.exit(1)

    counts = Counter(r.get('verdict') for r in rows)
    header = (f"# Step-1 validity — {len(rows)} datasets  (cache: {VDIR})\n\n"
              f"PASS={counts.get('PASS', 0)}  BORDERLINE={counts.get('BORDERLINE', 0)}  "
              f"FAIL={counts.get('FAIL', 0)}   (verdict is advisory — you decide the cut)\n")
    table = dataset_table(rows)
    print(header)
    print(table)
    body = header + "\n" + table + "\n"
    if a.reps:
        rr = rep_rankings(rows)
        print(rr)
        body += "\n## Per-rep rankings\n" + rr + "\n"
    if a.md:
        open(a.md, 'w').write(body)
        print(f"\nwrote {a.md}", file=sys.stderr)


if __name__ == '__main__':
    main()
