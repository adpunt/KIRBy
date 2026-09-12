"""Read aug_stage_sweep JSONs and report. Non-invasive — safe while a sweep runs.

    python scratchpad/aug_stage_report.py                # all datasets found
    python scratchpad/aug_stage_report.py flab_thermo_tresanco
"""
import os
import sys
import json
import glob

import numpy as np

ROOT = os.path.join(os.environ.get('KIRBY_CACHE_DIR',
                                   '/Volumes/seagate/kirby_cache'), 'aug_stage')


def report(path):
    out = json.load(open(path))
    rows = out['rows']
    prim = out['primary_metric']
    lower_better = prim in ('rmse', 'mae')
    W = out['base_width']

    tgt = {r['fold']: r for r in rows if r['arm'] == 'TARGET'}
    if not tgt:
        print(f"  {os.path.basename(path)}: no TARGET rows yet")
        return

    def delta(r, metric=None):
        m = metric or prim
        d = r[m] - tgt[r['fold']][m]
        return -d if (m in ('rmse', 'mae')) else d

    print(f"\n{'='*78}\n{out['dataset']}  (model={out['model']}, "
          f"{out['n_folds']} folds x {len({r['fold'] for r in rows})} seen, "
          f"pretrim={out.get('pretrim_per_source')})\n{'='*78}")
    tm = {k: float(np.nanmean([tgt[f][k] for f in tgt]))
          for k in ('spearman', 'rmse', 'r2')}
    print(f"TARGET: {out['best_rep']} ({W} cols)  "
          f"rho={tm['spearman']:+.4f}  rmse={tm['rmse']:.4f}  r2={tm['r2']:+.4f}")
    print(f"augmentation pool: "
          + (', '.join(f'{k}={v}' for k, v in out['aug_reps'].items()) or 'EMPTY'))

    # ---- gates -----------------------------------------------------------
    dg = [delta(r) for r in rows if r['arm'] == 'DIRGATE']
    if dg:
        v = float(np.mean(dg))
        print(f"\nDIRECTION GATE (worst rep vs best): {v:+.4f}  "
              f"{'PASS' if v < 0 else '*** FAIL ***'}")
    nz = [delta(r) for r in rows if r['arm'] == 'NOISE']
    if nz:
        seed_sd = float(np.nanmean([tgt[f].get('seed_sd', np.nan) for f in tgt]))
        print(f"NOISE FLOOR (meaningless columns): {float(np.mean(nz)):+.4f} "
              f"(sd {float(np.std(nz)):.4f}) | model-seed sd on target {seed_sd:.4f}")

    # ---- arms ------------------------------------------------------------
    groups = {}
    for r in rows:
        if r['arm'] in ('TARGET', 'DIRGATE'):
            continue
        groups.setdefault((r['source'], r['arm'], r['percent']), []).append(r)

    print(f"\n{'source':<8} {'arm':<9} {'%':>5} {'added':>6} {'mean d':>9} "
          f"{'sd':>7} {'won':>6}")
    print('-' * 56)
    for (src, arm, pct), rs in sorted(groups.items()):
        ds_ = [delta(r) for r in rs]
        won = sum(1 for d in ds_ if d > 0)
        lbl = 'greedy' if pct < 0 else f'{pct}%'
        print(f"{src:<8} {arm:<9} {lbl:>5} {rs[0]['n_added']:>6} "
              f"{np.mean(ds_):>+9.4f} {np.std(ds_):>7.4f} {won:>3}/{len(ds_)}")

    # ---- best RANKONCE setting per source, vs its RANDOM control ---------
    print("\nbest RANKONCE per source (and its matched RANDOM control):")
    for src in sorted({s for s, a, p in groups if a == 'RANKONCE'}):
        cands = [(p, float(np.mean([delta(r) for r in groups[(src, 'RANKONCE', p)]])))
                 for (s, a, p) in groups if s == src and a == 'RANKONCE']
        if not cands:
            continue
        bp, bd = max(cands, key=lambda t: t[1])
        ctrl = groups.get((src, 'RANDOM', bp))
        cd = float(np.mean([delta(r) for r in ctrl])) if ctrl else float('nan')
        print(f"  {src:<8} best={bp}% delta={bd:+.4f}   "
              f"random@{bp}%={cd:+.4f}   selection_effect={bd-cd:+.4f}")

    # ---- donor provenance -------------------------------------------------
    print("\nwho actually donated (RANKONCE, summed over folds):")
    for (src, arm, pct), rs in sorted(groups.items()):
        if arm != 'RANKONCE' or not rs[0].get('donors'):
            continue
        tot = {}
        for r in rs:
            for d, n in (r.get('donors') or {}).items():
                tot[d] = tot.get(d, 0) + n
        s = sum(tot.values()) or 1
        top = sorted(tot.items(), key=lambda kv: -kv[1])[:6]
        print(f"  {src:<8} {pct:>3}% ({len(tot)} donors): "
              + ', '.join(f'{d} {100*n/s:.0f}%' for d, n in top))


if __name__ == '__main__':
    want = sys.argv[1:]
    paths = sorted(glob.glob(os.path.join(ROOT, '*.json')))
    if want:
        paths = [p for p in paths
                 if any(w in os.path.basename(p) for w in want)]
    if not paths:
        print(f"no result JSONs in {ROOT}")
    for p in paths:
        try:
            report(p)
        except Exception as e:
            print(f"{p}: {type(e).__name__}: {e}")
