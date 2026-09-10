#!/usr/bin/env python
"""Read the antibody geometry survey result files.

    python tests/analysis/read_geometry_survey.py tests/results/geometry_survey

Reports, per dataset: whether the run scored anything, which of the nine base
representations were actually loaded, which geometry feature groups were scored
and which were dropped, and — when the file carries base-alone scores — what
adding each geometry group did to each base representation.

Files written before the base-alone scores were added hold concat scores with
nothing to subtract from, so no delta can be computed for them. This says so
rather than inventing a comparison.

Deltas are reported per dataset and counted as wins and losses. They are never
averaged across datasets.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

EXPECTED_BASE_REPS = ['aac', 'ablang2', 'antiberty', 'blosum', 'esm2', 'esm3',
                      'mint', 'prot_t5', 'saprot']


def main(results_dir):
    d = Path(results_dir)
    files = sorted(d.glob('geometry_*.json'))
    if not files:
        print(f"No geometry_*.json under {d}")
        return 1

    print(f"{len(files)} result files under {d}\n")

    ok, failed, no_delta = [], [], []
    wins = defaultdict(lambda: [0, 0])       # geometry group -> [helped, hurt]
    per_dataset_best = {}

    for f in files:
        try:
            r = json.loads(f.read_text())
        except Exception as e:
            print(f"{f.name}: unreadable — {e}")
            continue

        ds = r.get('metadata', {}).get('dataset', f.stem)
        md = r.get('metadata', {})
        status = r.get('status', '?')
        is_clf = md.get('task_type') == 'classification'
        primary = 'accuracy_mean' if is_clf else 'spearman_mean'
        label = 'acc' if is_clf else 'rho'

        if status == 'FAILED':
            failed.append(ds)
            reasons = r.get('geometry_failures', {})
            first = next(iter(reasons.values()), r.get('error', 'no reason recorded'))
            print(f"{ds}: FAILED — {first}")
            continue
        ok.append(ds)

        # Compare against what this run actually asked for. Comparing against a
        # fixed list of nine printed "10/9  MISSING: [aac, antiberty, ...]" for
        # a run that used --base-reps pruned and never requested those.
        loaded = md.get('base_reps', [])
        requested = md.get('base_reps_requested') or EXPECTED_BASE_REPS
        missing = md.get('base_reps_missing')
        if missing is None:
            missing = [b for b in requested if b not in loaded]

        scored = sorted(r.get('geometry_baselines', {}).keys())
        dropped = r.get('geometry_failures', {})
        undefined = {k: v for k, v in dropped.items() if 'undefined' in str(v)}
        broken = {k: v for k, v in dropped.items() if k not in undefined}

        print(f"{ds}  n={md.get('n_samples')}  groups={md.get('n_groups')}  "
              f"{md.get('task_type')}  {md.get('elapsed_seconds')}s")
        print(f"  base reps loaded: {len(loaded)}/{len(requested)}"
              + (f"   MISSING: {missing}" if missing else ""))
        print(f"  geometry scored: {len(scored)}  {scored}")
        if undefined:
            print(f"  undefined here: {sorted(undefined)}")
        if broken:
            for k, v in sorted(broken.items()):
                print(f"  DROPPED {k}: {v}")

        base_baselines = r.get('base_baselines', {})
        if not base_baselines:
            no_delta.append(ds)
            print("  no base-alone scores in this file — "
                  "concat scores only, no delta computable\n")
            continue

        best = None
        for key, entry in r.get('augmentations', {}).items():
            for model, dv in entry.get('delta', {}).items():
                geo = entry['geometry']
                wins[geo][0 if dv > 0 else 1] += 1
                if best is None or dv > best[0]:
                    best = (dv, key, model,
                            entry['scores'][model].get(primary))
        if best:
            per_dataset_best[ds] = best
            print(f"  best augmentation: {best[1]} ({best[2]}) "
                  f"{label}={best[3]:.4f}  delta {best[0]:+.4f}")
        print()

    print("=" * 62)
    print(f"{len(ok)} scored, {len(failed)} failed"
          + (f", {len(no_delta)} without base-alone scores" if no_delta else ""))
    if failed:
        print(f"  failed: {failed}")
    if no_delta:
        print(f"  no delta computable: {no_delta}")
        print("  Those files predate the base-alone scoring. Re-running writes "
              "it; predicted structures are cached, so the rerun pays only for "
              "the fits.")

    if wins:
        print("\nHow often each geometry group helped, counted per "
              "dataset x base rep x model. Never averaged.")
        for geo in sorted(wins, key=lambda g: -wins[g][0]):
            helped, hurt = wins[geo]
            print(f"  {geo:<26} helped {helped:4d}   hurt {hurt:4d}")

    if per_dataset_best:
        print("\nBest augmentation per dataset:")
        for ds, (dv, key, model, score) in sorted(per_dataset_best.items()):
            print(f"  {ds:<30} {key:<34} {model:<5} delta {dv:+.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else 'tests/results/geometry_survey'))
