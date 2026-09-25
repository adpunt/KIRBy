#!/usr/bin/env python3
"""Read the noise suite's D-versus-D' rows and print four tables.

Plan section 9. Written 2026-09-25. Reads every `*.rows.jsonl` under the
directories given: the files noise experiments write through
d_vs_dprime.make_row, one JSON object per line with the fields
d_vs_dprime.ROW_FIELDS:
    dataset, experiment, method, method_setting, similarity_setting,
    model_input, model, noise_condition, level, draw, fold, metric,
    lower_is_better, score_D, score_Dprime, score_clean, gain
Checkpoint files beside them (`*.units.jsonl`, `*.ckpt.jsonl`, `_checkpoint*/`)
are not rows and are not read.

Every table is per dataset and per downstream model. Nothing is averaged across
datasets. Within one dataset, a mean is over folds and noise draws only.

  A  clean vs D        per experiment, model input, condition and level: the
                       fit on the clean training labels against the fit on D
                       (the injected labels), raw pair first.
  B  D vs D'           per experiment, method, method setting, similarity
                       setting, model input, condition and level: how many of
                       the (fold, draw) pairs D' won, out of how many ran, and
                       the raw pair of means.
  C  D' per similarity per experiment, method, method setting without k,
     setting           model input, condition and level: every similarity
                       setting and k side by side, best first. The order is
                       read from the rows' lower_is_better.
  D  model input x     per experiment, condition and level: for the methods
     similarity space  that read neighbours, how many (fold, draw, method)
                       pairs D' won, per model input kind (base_1, base_2,
                       hybrid) and per similarity space.

A "win" is D' better than D in the dataset's own metric, direction from the
row's lower_is_better, never assumed. A tie (equal scores) is counted apart.
A pair with a non-finite score is not counted as run.

A group whose rows disagree on lower_is_better or on the metric is not ranked;
it is printed as a DIRECTION CONFLICT.

Usage:
    python tests/analysis/analyse_noise_prelim.py DIR [DIR ...]
        [--dataset caco2] [--experiment n42_similarity_noise_sweep]
        [--model rf] [--condition outlier_p10] [--tables ABCD]
        [--csv-dir OUT] [--top 20]

--csv-dir writes each table as a CSV per dataset and model:
    <OUT>/<dataset>__<model>__<table>.csv
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit('analyse_noise_prelim needs pandas')


KEY_FIELDS = ('dataset', 'experiment', 'method', 'method_setting',
              'similarity_setting', 'model_input', 'model', 'noise_condition',
              'level', 'draw', 'fold')
NO_NEIGHBOURS = 'none'


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def find_row_files(dirs):
    out = []
    for d in dirs:
        if os.path.isfile(d) and d.endswith('.rows.jsonl'):
            out.append(d)
            continue
        out += glob.glob(os.path.join(d, '**', '*.rows.jsonl'), recursive=True)
    return sorted(set(out))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return f


def read_rows(files):
    """Every complete row, de-duplicated on the row key (the first copy wins,
    as d_vs_dprime.done_keys skips a key already on disk). A line cut off by a
    kill is skipped."""
    seen = set()
    rows = []
    n_bad = n_dup = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    n_bad += 1
                    continue
                if not isinstance(r, dict) or 'dataset' not in r \
                        or 'score_D' not in r:
                    continue
                key = tuple(json.dumps(r.get(f), sort_keys=True)
                            for f in KEY_FIELDS)
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                r['_file'] = path
                rows.append(r)
    return rows, n_bad, n_dup


def split_k(method_setting):
    """(k or None, method setting without the k part). The sweep writes k as
    a leading 'k=<k>' segment (n07.similarity_sweep, n42)."""
    if method_setting is None:
        return None, ''
    parts = str(method_setting).split('|')
    k = None
    rest = []
    for p in parts:
        if p.startswith('k=') and k is None:
            try:
                k = int(p[2:])
                continue
            except ValueError:
                pass
        rest.append(p)
    return k, '|'.join(rest)


def to_frame(rows):
    df = pd.DataFrame(rows)
    for c in ('score_D', 'score_Dprime', 'score_clean', 'level'):
        df[c] = df[c].map(_num)
    df['lower_is_better'] = df['lower_is_better'].map(
        lambda v: None if v is None else bool(v))
    ks = df['method_setting'].map(split_k)
    df['k'] = ks.map(lambda t: t[0])
    df['setting_wo_k'] = ks.map(lambda t: t[1])
    df['similarity_space'] = df['similarity_setting'].map(
        lambda s: str(s).split('|')[0] if s is not None else None)
    df['model_input_kind'] = df['model_input'].map(
        lambda s: str(s).split(':', 1)[-1] if s is not None else None)
    for c in ('draw', 'fold'):
        df[c] = df[c].map(lambda v: -1 if v is None else int(v))
    return df


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def direction(g):
    """(lower_is_better, metric) of a group, or (None, why) when its rows
    disagree or carry none."""
    libs = set(g['lower_is_better'].dropna().unique().tolist())
    mets = set(g['metric'].dropna().unique().tolist())
    if len(libs) != 1:
        return None, f'lower_is_better {sorted(libs)}'
    if len(mets) > 1:
        return None, f'metrics {sorted(mets)}'
    return bool(next(iter(libs))), (next(iter(mets)) if mets else '?')


def better(a, b, lib):
    """+1 when a beats b, -1 when b beats a, 0 on a tie, None when either is
    not finite."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if a == b:
        return 0
    if lib:
        return 1 if a < b else -1
    return 1 if a > b else -1


def pair_counts(a, b, lib):
    won = lost = tied = 0
    for x, y in zip(a, b):
        s = better(x, y, lib)
        if s is None:
            continue
        if s > 0:
            won += 1
        elif s < 0:
            lost += 1
        else:
            tied += 1
    return won, lost, tied


def _mean(v):
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def _fmt(x, nd=4):
    return 'nan' if not math.isfinite(x) else f'{x:.{nd}f}'


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def table_a(df):
    """clean vs D. One D fit per (experiment, model input, condition, level,
    draw, fold): every method row of that pass carries the same score_D, so
    the first is taken."""
    cols = ['experiment', 'model_input', 'noise_condition', 'level']
    passes = df.drop_duplicates(cols + ['draw', 'fold'])
    out = []
    for key, g in passes.groupby(cols, dropna=False, sort=True):
        lib, met = direction(g)
        rec = dict(zip(cols, key))
        rec['metric'] = met if lib is not None else f'CONFLICT {met}'
        rec['lower_is_better'] = lib
        rec['n_pairs'] = int(np.sum(np.isfinite(g['score_clean'])
                                    & np.isfinite(g['score_D'])))
        rec['mean_clean'] = _mean(g['score_clean'].tolist())
        rec['mean_D'] = _mean(g['score_D'].tolist())
        if lib is not None:
            # "D won" here means the noise HELPED; counted, not hidden.
            w, l_, t = pair_counts(g['score_D'].tolist(),
                                   g['score_clean'].tolist(), lib)
            rec['clean_better'] = l_
            rec['D_better'] = w
            rec['ties'] = t
            rec['damage'] = ((rec['mean_D'] - rec['mean_clean']) if lib
                             else (rec['mean_clean'] - rec['mean_D']))
        out.append(rec)
    return pd.DataFrame(out)


def _dprime_groups(df, cols):
    out = []
    for key, g in df.groupby(cols, dropna=False, sort=True):
        lib, met = direction(g)
        rec = dict(zip(cols, key if isinstance(key, tuple) else (key,)))
        rec['metric'] = met if lib is not None else f'CONFLICT {met}'
        rec['lower_is_better'] = lib
        a = g['score_Dprime'].tolist()
        b = g['score_D'].tolist()
        if lib is None:
            rec.update(won=None, lost=None, tied=None, ran=None)
        else:
            w, l_, t = pair_counts(a, b, lib)
            rec.update(won=w, lost=l_, tied=t, ran=w + l_ + t)
        rec['mean_D'] = _mean(b)
        rec['mean_Dprime'] = _mean(a)
        rec['mean_clean'] = _mean(g['score_clean'].tolist())
        rec['mean_gain'] = (float('nan') if lib is None else
                            ((rec['mean_D'] - rec['mean_Dprime']) if lib
                             else (rec['mean_Dprime'] - rec['mean_D'])))
        out.append(rec)
    return pd.DataFrame(out)


def table_b(df):
    cols = ['experiment', 'method', 'method_setting', 'similarity_setting',
            'model_input', 'noise_condition', 'level']
    return _dprime_groups(df, cols)


def table_c(df):
    """Each method's D' across similarity settings and k, ranked within its
    group by mean D' in the direction the rows carry."""
    sw = df[df['similarity_setting'].astype(str) != NO_NEIGHBOURS]
    cols = ['experiment', 'method', 'setting_wo_k', 'model_input',
            'noise_condition', 'level', 'similarity_setting', 'k']
    t = _dprime_groups(sw, cols)
    if t.empty:
        return t
    grp = ['experiment', 'method', 'setting_wo_k', 'model_input',
           'noise_condition', 'level']
    rank = {}
    for _, g in t.groupby(grp, dropna=False, sort=False):
        libs = set(g['lower_is_better'].dropna().tolist())
        if len(libs) != 1:
            # Direction missing or in conflict: no order is defined.
            continue
        lib = bool(next(iter(libs)))
        fin = [(i, v) for i, v in zip(g.index, g['mean_Dprime'])
               if math.isfinite(v)]
        fin.sort(key=lambda iv: iv[1] if lib else -iv[1])
        for pos, (i, _v) in enumerate(fin):
            rank[i] = pos + 1
    t['rank_in_group'] = [rank.get(i) for i in t.index]
    return t.sort_values(grp + ['rank_in_group'], na_position='last')


def table_d(df):
    """Methods that read neighbours only (similarity_setting not 'none')."""
    sw = df[df['similarity_setting'].astype(str) != NO_NEIGHBOURS]
    cols = ['experiment', 'noise_condition', 'level', 'model_input_kind',
            'similarity_space']
    return _dprime_groups(sw, cols)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_a(t, top):
    if t.empty:
        print('    (no rows)')
        return
    print('    one line = one model input under one condition and level; '
          'means over (fold, draw) pairs')
    for _, r in t.iterrows():
        tail = ('' if r['lower_is_better'] is None else
                f"  clean better in {r['clean_better']} of {r['n_pairs']} "
                f"pairs, D better in {r['D_better']}, tied {r['ties']}; "
                f"damage {_fmt(r['damage'])}")
        print(f"    {str(r['experiment']):<36} {str(r['model_input']):<22} "
              f"{str(r['noise_condition']):<18} level {str(r['level']):<5} "
              f"{r['metric']}: clean {_fmt(r['mean_clean'])} -> D "
              f"{_fmt(r['mean_D'])}{tail}")


def _print_dprime(t, label_cols, top):
    if t.empty:
        print('    (no rows)')
        return
    shown = t.head(top) if top else t
    for _, r in shown.iterrows():
        lab = ' '.join(f"{r[c]}" for c in label_cols)
        if r['lower_is_better'] is None:
            print(f"    {lab}: DIRECTION CONFLICT ({r['metric']}), not ranked")
            continue
        print(f"    {lab}: {r['metric']} D {_fmt(r['mean_D'])} -> D' "
              f"{_fmt(r['mean_Dprime'])} (clean {_fmt(r['mean_clean'])}); "
              f"D' won {r['won']} of {r['ran']} pairs, lost {r['lost']}, "
              f"tied {r['tied']}; gain {_fmt(r['mean_gain'])}")
    if top and len(t) > top:
        print(f"    ... {len(t) - top} more lines (--top 0 prints all, "
              f"--csv-dir writes all)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dirs', nargs='+',
                    help='result directories (searched recursively) or '
                         '.rows.jsonl files')
    ap.add_argument('--dataset', action='append')
    ap.add_argument('--experiment', action='append')
    ap.add_argument('--model', action='append')
    ap.add_argument('--condition', action='append')
    ap.add_argument('--tables', default='ABCD')
    ap.add_argument('--top', type=int, default=40,
                    help='lines printed per table and (dataset, model); 0 = all')
    ap.add_argument('--csv-dir', default=None)
    args = ap.parse_args(argv)

    files = find_row_files(args.dirs)
    if not files:
        sys.exit(f'no *.rows.jsonl under {args.dirs}')
    rows, n_bad, n_dup = read_rows(files)
    print(f'{len(files)} row files, {len(rows)} rows '
          f'({n_dup} duplicate keys skipped, {n_bad} unreadable lines)')
    if not rows:
        return 0
    df = to_frame(rows)
    for col, vals in (('dataset', args.dataset), ('experiment', args.experiment),
                      ('model', args.model), ('noise_condition', args.condition)):
        if vals:
            df = df[df[col].isin(vals)]
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)

    builders = {'A': ('clean vs D', table_a),
                'B': ("D vs D'", table_b),
                'C': ("D' per similarity setting", table_c),
                'D': ('model input x similarity space', table_d)}
    for (ds, model), g in df.groupby(['dataset', 'model'], sort=True):
        print(f"\n{'=' * 78}\n{ds} / {model}   ({len(g)} rows)\n{'=' * 78}")
        for letter in args.tables.upper():
            if letter not in builders:
                continue
            title, fn = builders[letter]
            print(f"\n  {letter}. {title}")
            t = fn(g)
            if letter == 'A':
                _print_a(t, args.top)
            elif letter == 'B':
                if not t.empty:
                    t = t.sort_values(['experiment', 'noise_condition', 'level',
                                       'model_input', 'method',
                                       'similarity_setting', 'method_setting'])
                _print_dprime(t, ['experiment', 'method', 'method_setting',
                                  'similarity_setting', 'model_input',
                                  'noise_condition', 'level'], args.top)
            elif letter == 'C':
                if not t.empty:
                    print('    rank 1 = best mean D\' in the group, read from '
                          'the rows\' lower_is_better')
                _print_dprime(t, ['experiment', 'method', 'setting_wo_k',
                                  'model_input', 'noise_condition', 'level',
                                  'rank_in_group', 'similarity_setting', 'k'],
                              args.top)
            else:
                _print_dprime(t, ['experiment', 'noise_condition', 'level',
                                  'model_input_kind', 'similarity_space'],
                              args.top)
            if args.csv_dir and not t.empty:
                t.to_csv(os.path.join(args.csv_dir,
                                      f'{ds}__{model}__{letter}.csv'),
                         index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
