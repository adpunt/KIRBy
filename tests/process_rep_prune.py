"""Display the rep-pruning EVIDENCE — per dataset, one at a time.

Reads $KIRBY_CACHE_DIR/rep_prune/<ds>/greedy.json and prints, per dataset:
  1. solo scores (best-first),
  2. the greedy-elimination result at each correlation threshold — how many reps
     survive and which — so you can see the pool grow/shrink around 0.90,
  3. the drop trace at a chosen threshold (which rep was dropped, redundant with
     what, at what correlation),
  4. the bounded concat check — for each rep, did adding it to its nearest
     higher-scoring partner gain anything (was dropping it free?).

SELECTS NOTHING. It shows what greedy elimination WOULD keep at each threshold; the
final <=10 per dataset is your call. Sorting is for readability only.

Cache from KIRBY_CACHE_DIR. Run after rsyncing rep_prune/ from the server:
    python tests/process_rep_prune.py                    # all datasets found
    python tests/process_rep_prune.py herg_ki logd       # just these
    python tests/process_rep_prune.py --trace 0.90       # show drops at T=0.90
"""
import os, sys, json, glob, argparse

CACHE = os.environ.get('KIRBY_CACHE_DIR', os.path.expanduser('~/.cache/kirby'))
ROOT = os.path.join(CACHE, 'rep_prune')


def show(ds, trace_T):
    d = os.path.join(ROOT, ds)
    try:
        g = json.load(open(os.path.join(d, 'greedy.json')))
    except FileNotFoundError:
        print(f"\n### {ds}: no greedy.json (task not finished?)"); return
    metric = g.get('primary', '?'); solo = g.get('solo', {})
    ths = g.get('thresholds', {}); nearest = g.get('nearest', [])
    print(f"\n{'='*72}\n{ds}   ({metric})   {g.get('n_reps','?')} reps\n{'='*72}")

    print("  solo (best-first):")
    for r, s in sorted(solo.items(), key=lambda kv: -kv[1]):
        print(f"    {r:24s} {s:+.3f}")

    print(f"\n  greedy elimination — reps kept per correlation threshold:")
    for T in sorted(ths, key=float):
        b = ths[T]
        flag = '  <= 10' if b['n_kept'] <= 10 else ''
        print(f"    T={T}: keep {b['n_kept']:2d}{flag}   {', '.join(b['kept'])}")

    tkey = f"{float(trace_T):.2f}"
    if tkey in ths:
        dropped = ths[tkey]['dropped']
        print(f"\n  drops at T={tkey} ({len(dropped)} reps):")
        print(f"    {'dropped rep':24s} {'redundant with':24s} corr  solo  anchor")
        for r in sorted(dropped, key=lambda r: -r['corr']):
            print(f"    {r['rep']:24s} {r['redundant_with']:24s} {r['corr']:+.2f} "
                  f"{r['solo']:+.3f} {r['anchor_solo']:+.3f}")

    if nearest:
        print(f"\n  bounded concat check — rep + nearest higher-solo partner:")
        print(f"    {'rep':24s} {'partner':24s} corr concat   gain")
        for r in sorted(nearest, key=lambda r: -r['corr']):
            print(f"    {r['rep']:24s} {r['partner']:24s} {r['corr']:+.2f} "
                  f"{r['concat']:+.3f} {r['gain']:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', help='datasets to show (default: all found)')
    ap.add_argument('--trace', type=float, default=0.90,
                    help='threshold whose drop trace to print (default 0.90)')
    a = ap.parse_args()
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, '*'))
                   if os.path.isdir(p))
    if not found:
        print(f"No rep_prune output in {ROOT}\n"
              f"(rsync it from the server, or set KIRBY_CACHE_DIR).", file=sys.stderr)
        sys.exit(1)
    targets = a.datasets or found
    print(f"# rep-pruning EVIDENCE — {len(targets)} dataset(s)  (cache: {ROOT})")
    print("# SELECTS NOTHING — greedy elimination shown per threshold; you make the cut.")
    for ds in targets:
        show(ds, a.trace)


if __name__ == '__main__':
    main()
