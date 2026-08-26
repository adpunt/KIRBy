#!/usr/bin/env python3
"""Read the f24 augmentation results and print what actually happened.

PER-CELL WIN COUNTS, PER DATASET. Never a mean across datasets and never a mean
rank: a method losing on 20 datasets of 30 is not a reason to drop it if it is
the best method on the other 10, and an average is exactly what hides that.

Four things are printed, in this order, because each one can invalidate the
next.

1. DECOYS. How many features each method took from the planted pure-noise donor
   and from the planted near-copy of the base representation. A working rule
   takes zero of both. A method that took decoy features has not "won"
   anything, whatever its score says, so this comes first.

2. AGAINST THE MATCHED RANDOM DONOR. Every configuration was run twice: once
   with the real donors and once with randomly drawn features of the same
   shape. Adding features changes model capacity by itself, so the real-minus-
   random difference is the part that is about the donors. A method that beats
   no-extras but not its own random control has bought capacity, not signal.

3. AGAINST NO EXTRAS. The plain delta, sign-normalised so positive always means
   better whatever the dataset's metric direction.

4. CLAIMED AGAINST DELIVERED. What the method read off its own internal slice,
   beside what it got on the untouched fold. A method that claims much more
   than it delivers is selecting on noise.

Every number is compared to that dataset's own fold-to-fold spread, which is
printed at the top, because a difference smaller than the spread is not a
difference.

Usage:
    python tests/analysis/read_f24_augmentation.py
    python tests/analysis/read_f24_augmentation.py --dir <results dir> --model rf
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, '..', 'experiments', 'results', 'framework')


def load(results_dir):
    """Every real result file, one per (dataset, placement, budget).

    The grid submits four array tasks per dataset and each writes its own file,
    so the scope is in the name: <dataset>__single.json,
    <dataset>__hybrid__b500.json and so on. A file whose name contains __count
    or __timing fitted nothing, and __smoke measured nothing — all three are
    skipped rather than read as results.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(
            results_dir, 'f24_augmentation_admission_*.json'))):
        name = os.path.basename(path)
        if any(k in name for k in ('__count', '__timing', '__smoke')):
            continue
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f'  could not read {name}: {exc}')
            continue
        if not (doc.get('rows') or []):
            print(f'  {name}: no completed rows — check its log for skips')
            continue
        out.append((path, doc))
    return out


def _rows_by_cell(doc, model):
    """{configuration: {control: row}} for one downstream model.

    The margin multiple is part of the key. It is NOT in `cell`, because the
    ladder is enumerated once and the margins are read off its rung scores --
    so two rows that actually RAN under different margins share a `cell` and
    would overwrite each other here. Measured on flab_thermo_tresanco: one
    configuration produced three rows, keeping 8.3, 1.7 and 0 features. Keying
    on `cell` alone would have kept one of the three and shown nothing missing.
    """
    by = defaultdict(dict)
    for r in doc.get('rows') or []:
        m = (r.get('models') or {}).get(model)
        if not m or m.get('status') != 'ok':
            continue
        key = (r['cell'], r.get('margin_multiple'), r['placement'], r['budget'])
        by[key][r['control']] = (r, m)
    return by


def report(path, doc, model):
    ds = doc.get('dataset', '?')
    metric = doc.get('metric', '?')
    lower = bool(doc.get('lower_is_better'))
    spread = doc.get('fold_spread')
    sign = -1 if lower else 1

    print('\n' + '=' * 78)
    print(f'{ds}   metric {metric}   n={doc.get("n_samples")}   model {model}')
    print(f'  base representation {doc.get("base_rep")} '
          f'({doc.get("base_width")} features), chosen by '
          f'{doc.get("base_rep_provenance", "?")}')
    gap = doc.get('base_rep_gap_to_runner_up')
    if gap is not None and spread:
        verdict = ('DOMINANT' if gap > spread
                   else 'flat-topped — no representation dominates here')
        print(f'  it beats the runner-up by {gap:.4f}, against a fold spread of '
              f'{spread:.4f} -> {verdict}')
    if spread:
        print(f'  FOLD SPREAD {spread:.4f} {metric}. A difference smaller than '
              f'this is not a difference.')
    print(f'  folds {doc.get("n_folds")}   seeds {doc.get("n_seeds")}   '
          f'candidate list cut to {doc.get("max_candidates")}')
    if int(doc.get('n_folds') or 0) < 5:
        print(f'  ** {doc.get("n_folds")} folds. Enough to rank methods against '
              f'each other; NOT enough to quote a delta. **')

    by = _rows_by_cell(doc, model)
    if not by:
        print(f'  no completed rows for model {model}')
        return

    # the no-extras control, per placement and budget
    base = {}
    for (cell, mg, pl, bud), got in by.items():
        if cell == 'none' and 'real' in got:
            base[(pl, bud)] = got['real'][1]['mean']

    print(f'\n  {"configuration":52s}{"vs none":>9}{"vs random":>11}'
          f'{"noise":>7}{"echo":>6}')
    lines = []
    for (cell, mg, pl, bud), got in sorted(
            by.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], kv[0][3])):
        if cell == 'none' or 'real' not in got:
            continue
        row, m = got['real']
        b = base.get((pl, bud))
        d_none = sign * (m['mean'] - b) if b is not None else None
        d_rand = (sign * (m['mean'] - got['random'][1]['mean'])
                  if 'random' in got else None)
        noise = sum(m.get('decoy_noise_per_fold') or [])
        echo = sum(m.get('decoy_echo_per_fold') or [])
        name = cell + (f'__m{mg:g}' if mg is not None else '')
        lines.append((d_rand if d_rand is not None else -9e9,
                      name, pl, bud, d_none, d_rand, noise, echo, row, m))

    lines.sort(reverse=True)
    for _, cell, pl, bud, d_none, d_rand, noise, echo, row, m in lines:
        tag = f'{cell}  [{pl} b{bud}]'
        f1 = f'{d_none:+.4f}' if d_none is not None else '    —'
        f2 = f'{d_rand:+.4f}' if d_rand is not None else '     —'
        flag = ''
        if noise or echo:
            flag = '  <- took decoy features'
        print(f'  {tag:52s}{f1:>9}{f2:>11}{noise:7d}{echo:6d}{flag}')

    # win counts, which is the actual readout
    print(f'\n  WIN COUNTS on {ds} (configurations beating each reference)')
    per_method = defaultdict(lambda: [0, 0, 0, 0])   # none, random, clean, total
    for _, cell, pl, bud, d_none, d_rand, noise, echo, row, m in lines:
        meth = row['method']
        c = per_method[meth]
        c[3] += 1
        if d_none is not None and d_none > 0:
            c[0] += 1
        if d_rand is not None and d_rand > 0:
            c[1] += 1
        if not noise and not echo:
            c[2] += 1
    print(f'    {"method":18s}{"beats none":>12}{"beats random":>14}'
          f'{"took no decoys":>16}')
    for meth in sorted(per_method):
        w = per_method[meth]
        print(f'    {meth:18s}{w[0]:>7}/{w[3]:<4}{w[1]:>9}/{w[3]:<4}'
              f'{w[2]:>11}/{w[3]:<4}')

    # claimed against delivered
    print(f'\n  CLAIMED against DELIVERED, worst overstatements first')
    over = []
    for _, cell, pl, bud, d_none, d_rand, noise, echo, row, m in lines:
        cl = [c for c in (m.get('claimed_per_fold') or [])
              if c is not None and c == c]
        if not cl or d_none is None:
            continue
        claimed = sum(cl) / len(cl)
        over.append((claimed - d_none, cell, claimed, d_none))
    for gapv, cell, claimed, delivered in sorted(over, reverse=True)[:6]:
        print(f'    {cell:52s} claimed {claimed:+.4f}  delivered {delivered:+.4f}')

    skipped = doc.get('skipped') or []
    if skipped:
        print(f'\n  {len(skipped)} configurations did not complete. First three:')
        for s in skipped[:3]:
            print(f'    {s.get("cell","?")}: {str(s.get("reason"))[:90]}')
    deg = doc.get('degenerate_cells') or {}
    if deg:
        print(f'  {len(deg)} configurations skipped as degenerate, with the '
              f'arithmetic in the run log.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--model', default=None,
                    help='downstream model to report (default: every one found)')
    args = ap.parse_args()

    docs = load(args.dir)
    if not docs:
        print(f'no f24 result files in {args.dir}')
        return 1
    for path, doc in docs:
        models = ([args.model] if args.model
                  else list(doc.get('models') or []))
        for m in models:
            report(path, doc, m)
    print('\n' + '=' * 78)
    print('Per dataset, never averaged across them. A configuration that took '
          'decoy\nfeatures has not won anything, whatever its score says.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
