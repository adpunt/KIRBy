#!/usr/bin/env python3
"""What each experiment has produced so far, per dataset, in the live run.

Reads only. Nothing is fitted, nothing in the run is changed. It reads
`ledger.json` for the status of every unit, the final result files
`<run>/<category>/<experiment dir>/<dataset>[__<noise condition>].json`, and
the partial files next to them (`<same stem>.partial.jsonl`, one line per
finished fold / model / seed of a unit that has not finished).

    python tests/tools/results_state.py overview
    python tests/tools/results_state.py searches
    python tests/tools/results_state.py exp c07
    python tests/tools/results_state.py exp n05 --datasets herg_ki logd
    python tests/tools/results_state.py exp s05 --all-numbers
    python tests/tools/results_state.py exp c18 --tree
    python tests/tools/results_state.py exp opig
    python tests/tools/results_state.py all

`overview` is one table per stage: one row per dataset, one cell per
experiment. `exp` is one section per dataset: the ledger status and hours,
whether a final or a partial file exists, how many units the partial file
holds, and then every number whose key looks like a score (spearman, rmse,
auroc, score, ...), one line per number, with per-fold lists printed as lists.
`--all-numbers` prints every number instead; `--tree` prints the shape of the
file. `all` writes the overview and every experiment to `<run>/analysis/`.

The run is the one named in tests/results/CURRENT_RUN, or `--run PATH`.
Every output is also written to `<run>/analysis/<name>.md`, so the folder
can be copied off the server on its own.
"""
import argparse
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CURRENT_RUN_FILE = REPO / 'tests' / 'results' / 'CURRENT_RUN'
CATEGORY_DIRS = ['complexity', 'federated', 'similarity', 'noise', 'noise_full']
NON_DATASET_DIRS = ['opig', 'cross']

SCORE_KEY = re.compile(
    r'(score|spearman|pearson|kendall|rmse|mse|mae|r2|auroc|auc|aupr|mcc|'
    r'accuracy|balanced|ci$|concordance|precision|recall|f1|ndcg|hit|'
    r'enrichment|recovery|delta|gap|improvement|win|ece|nll|coverage|'
    r'correlation|rho|mi$|mutual)', re.I)
MAX_LINES_PER_FILE = 400


# ── reading ────────────────────────────────────────────────────────────────

def resolve_run(run):
    if run:
        return Path(run).resolve()
    if not CURRENT_RUN_FILE.exists():
        sys.exit(f'no --run given and {CURRENT_RUN_FILE} does not exist')
    p = Path(CURRENT_RUN_FILE.read_text().strip())
    return p if p.is_absolute() else (REPO / p).resolve()


def load_ledger(root):
    f = root / 'ledger.json'
    if not f.exists():
        return None
    return json.loads(f.read_text())


_TAG = '__ckpt__'


def decode(obj):
    """_checkpoint.decode without numpy: arrays become lists, numpy scalars
    become Python numbers, tuple/set become lists, non-string dict keys
    become their text."""
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    tag = obj.get(_TAG)
    if tag is None:
        return {k: decode(v) for k, v in obj.items()}
    if tag == 'ndarray':
        return decode(obj['data'])
    if tag == 'npscalar':
        return obj['value']
    if tag in ('tuple', 'set'):
        return [decode(x) for x in obj['items']]
    if tag == 'dict':
        return {_keytext(decode(k)): decode(v) for k, v in obj['items']}
    return obj


def _keytext(k):
    if isinstance(k, list):
        return '(' + ','.join(_keytext(x) for x in k) + ')'
    return str(k)


def read_partial(path):
    """(header or None, [(key dict, payload)]). A last line cut off by a
    kill is skipped."""
    header, units = None, []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None, []
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and 'key' in d and 'payload' in d:
            units.append((d['key'], decode(d['payload'])))
        elif header is None:
            header = d
    return header, units


def read_json(path):
    try:
        return json.loads(path.read_text()), None
    except Exception as e:  # noqa: BLE001 -- report and carry on
        return None, f'{type(e).__name__}: {e}'


# ── where things are ───────────────────────────────────────────────────────

def exp_code(dirname):
    return dirname.split('_', 1)[0]


def experiment_dirs(root):
    """{code: [(category, dir)]} for every experiment folder in the run."""
    out = defaultdict(list)
    for cat in CATEGORY_DIRS:
        d = root / cat
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and not sub.name.startswith('_'):
                out[exp_code(sub.name)].append((cat, sub))
    return out


def files_in(expdir):
    """{(dataset, condition or None): {'final': path, 'partial': path}}."""
    out = defaultdict(dict)
    for p in sorted(expdir.iterdir()):
        name = p.name
        if name.endswith('.partial.jsonl'):
            stem, kind = name[:-len('.partial.jsonl')], 'partial'
        elif name.endswith('.json'):
            stem, kind = name[:-len('.json')], 'final'
        else:
            continue
        ds, _, cond = stem.partition('__')
        out[(ds, cond or None)][kind] = p
    return out


def ledger_units(ledger):
    """{(dataset, unit key): (stage, unit)} for dataset rows, and the rows."""
    units, rows = {}, OrderedDict()
    if not ledger:
        return units, rows
    for name, row in ledger['rows'].items():
        rows[name] = row
        for st in row.get('stages', []):
            for k, u in st.get('units', {}).items():
                units[(name, k)] = (st['name'], u)
    return units, rows


# ── flattening numbers ─────────────────────────────────────────────────────

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return 'nan'
        if x == int(x) and abs(x) < 1e6:
            return str(int(x))
        return f'{x:.4g}' if abs(x) >= 1e4 or abs(x) < 1e-3 else f'{x:.4f}'
    return str(x)


def flatten(obj, path=()):
    """Yield (path tuple, value) for every number. A list of plain numbers is
    yielded once, as a list, under its own path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(obj, list):
        if obj and all(_is_num(x) or x is None for x in obj):
            yield path, obj
        else:
            for i, v in enumerate(obj):
                yield from flatten(v, path + (f'[{i}]',))
    elif _is_num(obj):
        yield path, obj


def _scoreish(path):
    for part in reversed(path):
        if part.startswith('['):
            continue
        return bool(SCORE_KEY.search(part))
    return False


SKIP_TOP = {'timestamp', 'seconds', 'result_status', 'has_computed_results'}


def number_lines(obj, all_numbers=False, prefix=''):
    lines = []
    for path, v in flatten(obj):
        if path and path[0] in SKIP_TOP:
            continue
        if not all_numbers and not _scoreish(path):
            continue
        key = prefix + '/'.join(path)
        if isinstance(v, list):
            val = '[' + ' '.join('-' if x is None else _fmt(x) for x in v) + ']'
        else:
            val = _fmt(v)
        lines.append(f'{key} = {val}')
    return lines


def tree_lines(obj, depth=0, maxdepth=6, width=10):
    pad = '  ' * depth
    out = []
    if isinstance(obj, dict):
        keys = list(obj)
        for k in keys[:width]:
            v = obj[k]
            if isinstance(v, (dict, list)):
                out.append(f'{pad}{k}: {type(v).__name__}[{len(v)}]')
                if depth < maxdepth:
                    out += tree_lines(v, depth + 1, maxdepth, width)
            else:
                out.append(f'{pad}{k}: {str(v)[:70]}')
        if len(keys) > width:
            out.append(f'{pad}... {len(keys) - width} more keys: {", ".join(map(str, keys[width:width + 12]))}')
    elif isinstance(obj, list) and obj:
        if all(_is_num(x) or x is None for x in obj):
            out.append(f'{pad}{[_fmt(x) if x is not None else None for x in obj[:8]]}'
                       + (' ...' if len(obj) > 8 else ''))
        else:
            out.append(f'{pad}[0] of {len(obj)}:')
            out += tree_lines(obj[0], depth + 1, maxdepth, width)
    return out


# ── reports ────────────────────────────────────────────────────────────────

def _hours(s):
    return '-' if s is None else f'{s / 3600:.1f} h'


def unit_summary(u):
    if u is None:
        return 'not in the ledger'
    bits = [u.get('status', '?')]
    if u.get('attempts'):
        bits.append(f"attempts {u['attempts']}")
    if u.get('seconds') is not None:
        bits.append(_hours(u['seconds']))
    if u.get('status') == 'running' and u.get('started'):
        bits.append(f"started {u['started'][:16]}")
    if u.get('error'):
        bits.append('error: ' + str(u['error']).splitlines()[0][:160])
    return ', '.join(bits)


def partial_summary(units):
    """How many units, and the values each key field takes."""
    fields = OrderedDict()
    for key, _ in units:
        for k, v in (key or {}).items():
            fields.setdefault(k, [])
            t = _keytext(v) if isinstance(v, list) else str(v)
            if t not in fields[k]:
                fields[k].append(t)
    parts = [f'{k} {",".join(vs[:12])}{"..." if len(vs) > 12 else ""}'
             for k, vs in fields.items()]
    return f'{len(units)} units finished' + (' (' + '; '.join(parts) + ')' if parts else '')


def report_experiment(root, code, ledger, datasets=None, all_numbers=False,
                      tree=False):
    dirs = experiment_dirs(root).get(code, [])
    units, rows = ledger_units(ledger)
    out = [f'# {code}', '']
    if not dirs:
        out.append(f'No folder for {code} under {root}.')
    # every (dataset, unit) the ledger has for this experiment, even with no file
    want = OrderedDict()
    for (ds, k), (stage, u) in units.items():
        if k.split('__', 1)[0] != code:
            continue
        cond = k.split('__', 1)[1] if '__' in k else None
        want[(ds, cond, stage)] = u
    found = {}
    for cat, d in dirs:
        out.append(f'Folder: {d.relative_to(root)}')
        for (ds, cond), kinds in files_in(d).items():
            found[(ds, cond, cat)] = kinds
    out.append('')
    keys = sorted(set(want) | set(found),
                  key=lambda t: (t[0], t[2], t[1] or ''))
    if datasets:
        keys = [t for t in keys if t[0] in datasets]
    n_final = sum(1 for t in keys if found.get(t, {}).get('final'))
    n_part = sum(1 for t in keys if found.get(t, {}).get('partial')
                 and not found.get(t, {}).get('final'))
    status = defaultdict(int)
    for t in keys:
        status[(want.get(t) or {}).get('status', 'no ledger unit')] += 1
    out.append(f'{len(keys)} (dataset, condition) units: {n_final} with a final file, '
               f'{n_part} with only a partial file. Ledger: '
               + ', '.join(f'{v} {k}' for k, v in sorted(status.items())) + '.')
    out.append('')
    for ds, cond, cat in keys:
        row = rows.get(ds, {})
        meta = ', '.join(str(x) for x in (row.get('domain'), row.get('task_type'),
                                          f"{row.get('n_samples')} rows" if row.get('n_samples') else None) if x)
        title = f'## {ds}' + (f' / {cond}' if cond else '') + (f' ({cat})' if cat != 'complexity' else '')
        out.append(title + (f'  [{meta}]' if meta else ''))
        out.append(f'Ledger: {unit_summary(want.get((ds, cond, cat)))}')
        kinds = found.get((ds, cond, cat), {})
        body = []
        if kinds.get('final'):
            d, err = read_json(kinds['final'])
            if err:
                out.append(f'Final file unreadable: {err}')
            else:
                out.append(f"Final file: {kinds['final'].name}, written {str(d.get('timestamp', '?'))[:16]}"
                           + (f", result_status {d['result_status']}" if 'result_status' in d else '')
                           + (f", reason: {d['reason']}" if d.get('reason') else ''))
                body = tree_lines(d) if tree else number_lines(d, all_numbers)
        elif kinds.get('partial'):
            header, pu = read_partial(kinds['partial'])
            out.append('Partial file only: ' + partial_summary(pu))
            if tree and pu:
                body = tree_lines(pu[0][1])
            else:
                for key, payload in pu:
                    pre = ','.join(f'{k}={_keytext(v) if isinstance(v, list) else v}'
                                   for k, v in (key or {}).items()) + ' | '
                    body += number_lines(payload, all_numbers, prefix=pre)
        else:
            out.append('No result file.')
        if body:
            if len(body) > MAX_LINES_PER_FILE:
                cut = len(body) - MAX_LINES_PER_FILE
                body = body[:MAX_LINES_PER_FILE] + [f'... {cut} more lines (use --datasets {ds})']
            out.append('```')
            out += body
            out.append('```')
        elif (kinds.get('final') or kinds.get('partial')) and not tree:
            out.append('No score-like numbers; try --all-numbers or --tree.')
        out.append('')
    return out


STATUS_MARK = {'ok': 'ok', 'running': 'RUN', 'failed': 'FAIL', 'pending': '.',
               'skipped': 'skip'}


def report_overview(root, ledger):
    units, rows = ledger_units(ledger)
    dirs = experiment_dirs(root)
    out = [f'# Run {root.name}', '']
    if ledger:
        out.append(f"Ledger created {ledger.get('created', '?')[:16]}. "
                   f"Read {datetime.now().isoformat(timespec='minutes')}.")
    else:
        out.append('No ledger.json: cells below come from files only.')
    out += ['', 'A cell is the ledger status of that unit: ok, RUN (running), FAIL, '
            '. (pending), skip. A "+p" means a partial file exists, "+f" a final file. '
            'For noise, a cell counts the noise conditions: ok/total.', '']
    # stage -> experiment codes in ledger order
    stage_exps = OrderedDict()
    for (ds, k), (stage, u) in units.items():
        stage_exps.setdefault(stage, [])
        e = k.split('__', 1)[0]
        if e not in stage_exps[stage]:
            stage_exps[stage].append(e)
    file_index = {}
    for code, lst in dirs.items():
        for cat, d in lst:
            for (ds, cond), kinds in files_in(d).items():
                file_index[(cat, code, ds, cond)] = kinds
    for stage, exps in stage_exps.items():
        dsets = [n for n, r in rows.items() if r.get('kind') == 'dataset'
                 and any(s['name'] == stage and s.get('units') for s in r.get('stages', []))]
        if not dsets:
            continue
        out.append(f'## {stage}')
        out.append('')
        head = ['dataset', 'rows', 'task'] + exps
        out.append('| ' + ' | '.join(head) + ' |')
        out.append('|' + '---|' * len(head))
        for ds in dsets:
            r = rows[ds]
            cells = [ds, str(r.get('n_samples') or ''), str(r.get('task_type') or '')[:5]]
            for e in exps:
                us = [(k, u) for (d, k), (s, u) in units.items()
                      if d == ds and s == stage and k.split('__', 1)[0] == e]
                if not us:
                    cells.append('')
                    continue
                if len(us) == 1 and '__' not in us[0][0]:
                    u = us[0][1]
                    c = STATUS_MARK.get(u.get('status'), u.get('status'))
                    kinds = file_index.get((stage, e, ds, None), {})
                else:
                    n_ok = sum(1 for _, u in us if u.get('status') == 'ok')
                    n_run = sum(1 for _, u in us if u.get('status') == 'running')
                    n_fail = sum(1 for _, u in us if u.get('status') == 'failed')
                    c = f'{n_ok}/{len(us)}' + (f' {n_run}RUN' if n_run else '') + (f' {n_fail}FAIL' if n_fail else '')
                    kinds = {}
                    for k, _ in us:
                        cond = k.split('__', 1)[1] if '__' in k else None
                        kinds.update(file_index.get((stage, e, ds, cond), {}))
                if kinds.get('final') and 'ok' not in c:
                    c += ' +f'
                elif kinds.get('partial') and not kinds.get('final'):
                    c += ' +p'
                cells.append(c)
            out.append('| ' + ' | '.join(cells) + ' |')
        out.append('')
    others = [n for n, r in rows.items() if r.get('kind') != 'dataset']
    if others:
        out += ['## OPIG and cross rows', '', '| row | units |', '|---|---|']
        for n in others:
            us = [(k, u) for (d, k), (s, u) in units.items() if d == n]
            out.append(f'| {n} | ' + ', '.join(f"{k} {STATUS_MARK.get(u.get('status'), u.get('status'))}"
                                               for k, u in us) + ' |')
        out.append('')
    failed = [(d, s, k, u) for (d, k), (s, u) in units.items() if u.get('status') == 'failed']
    if failed:
        out += ['## Failed units', '', '| dataset | stage | unit | attempts | first error line |',
                '|---|---|---|---|---|']
        for d, s, k, u in failed:
            err = str(u.get('error') or '').splitlines()[0][:140] if u.get('error') else ''
            out.append(f"| {d} | {s} | {k} | {u.get('attempts')} | {err} |")
        out.append('')
    return out


SEARCH_LOG_RE = re.compile(
    r'per-fold searches, fold|searches stored in|unavailable|molecule rows|'
    r'Traceback|Error|Killed|CANCELLED|DUE TO TIME LIMIT|oom', re.I)


def search_log_lines(log, max_lines=30):
    """Per attempt (the runner writes '=== <time> <command>' before each):
    when it started, the lines that say what the search did, and its last line."""
    try:
        text = Path(log).read_text(errors='replace').splitlines()
    except (OSError, TypeError):
        return ['(no log file)']
    attempts, cur = [], None
    for ln in text:
        if ln.startswith('=== '):
            cur = {'start': ln[4:23], 'hits': [], 'last': ''}
            attempts.append(cur)
            continue
        if cur is None:
            cur = {'start': '?', 'hits': [], 'last': ''}
            attempts.append(cur)
        if ln.strip():
            cur['last'] = ln.strip()
        if SEARCH_LOG_RE.search(ln) and len(cur['hits']) < 12:
            cur['hits'].append(ln.strip()[:200])
    out = []
    for i, at in enumerate(attempts, 1):
        out.append(f'attempt {i}, started {at["start"]}')
        out += ['    ' + h for h in at['hits']]
        out.append('    last line: ' + at['last'][:200])
    return out[-max_lines:]


def report_searches(root, ledger):
    """Every dataset's per-fold search units: status, attempts, hours, job,
    and what each attempt's log says."""
    units, rows = ledger_units(ledger)
    out = ['# Per-fold searches', '',
           'One section per dataset. A search unit is one outer fold. "attempts" counts '
           'how many jobs have started it; a job that dies at the wall time or runs out '
           'of memory makes the next job start it again, at most 4 times.', '']
    for ds, row in rows.items():
        if row.get('kind') != 'dataset':
            continue
        su = [(k, u) for (d, k), (s, u) in units.items()
              if d == ds and s == 'complexity' and re.match(r'^search_f\d+$', k)]
        if not su:
            continue
        cut = f" -- CUT: {row['cut']}" if row.get('cut') else ''
        out.append(f'## {ds}{cut}')
        c = row.get('claim') or {}
        if c:
            out.append(f"Held by job {c.get('job')} since {str(c.get('since'))[:16]}, "
                       f"last sign of life {str(c.get('heartbeat'))[:16]}")
        out.append('')
        out.append('| fold | status | attempts | hours | job |')
        out.append('|---|---|---|---|---|')
        for k, u in su:
            out.append(f"| {k[len('search_f'):]} | {u.get('status')} | {u.get('attempts')} | "
                       f"{_hours(u.get('seconds'))} | {u.get('job') or ''} |")
        out.append('')
        if row.get('cut'):
            continue
        for k, u in su:
            out.append(f'{k}:')
            out.append('```')
            out += search_log_lines(u.get('log'))
            out.append('```')
        out.append('')
    return out


def report_non_dataset(root, all_numbers=False, tree=False):
    out = ['# OPIG and cross-dataset results', '']
    for sub in NON_DATASET_DIRS:
        base = root / sub
        if not base.is_dir():
            out.append(f'No {sub}/ folder.')
            continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or not (p.name.endswith('.json') or p.name.endswith('.partial.jsonl')):
                continue
            out.append(f'## {p.relative_to(root)}')
            if p.name.endswith('.json'):
                d, err = read_json(p)
                if err:
                    out.append(f'Unreadable: {err}')
                    continue
                body = tree_lines(d) if tree else number_lines(d, all_numbers)
            else:
                _, pu = read_partial(p)
                out.append(partial_summary(pu))
                body = []
                for key, payload in pu:
                    pre = ','.join(f'{k}={v}' for k, v in (key or {}).items()) + ' | '
                    body += number_lines(payload, all_numbers, prefix=pre)
            if len(body) > MAX_LINES_PER_FILE:
                body = body[:MAX_LINES_PER_FILE] + [f'... {len(body) - MAX_LINES_PER_FILE} more lines']
            if body:
                out += ['```'] + body + ['```']
            out.append('')
    return out


def write(root, name, lines, quiet=False):
    d = root / 'analysis'
    d.mkdir(exist_ok=True)
    text = '\n'.join(lines) + '\n'
    (d / f'{name}.md').write_text(text)
    if not quiet:
        sys.stdout.write(text)
    return d / f'{name}.md'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('what', choices=['overview', 'searches', 'exp', 'all'])
    ap.add_argument('experiment', nargs='?', help='c07, n05, s05, fed02, opig, ...')
    ap.add_argument('--run', default=None)
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--all-numbers', action='store_true')
    ap.add_argument('--tree', action='store_true')
    a = ap.parse_args(argv)
    root = resolve_run(a.run)
    ledger = load_ledger(root)

    if a.what == 'overview':
        write(root, 'OVERVIEW', report_overview(root, ledger))
    elif a.what == 'searches':
        write(root, 'SEARCHES', report_searches(root, ledger))
    elif a.what == 'exp':
        if not a.experiment:
            ap.error('exp needs an experiment code')
        if a.experiment in ('opig', 'cross'):
            write(root, a.experiment, report_non_dataset(root, a.all_numbers, a.tree))
        else:
            name = a.experiment + ('_tree' if a.tree else '') + ('_all' if a.all_numbers else '')
            write(root, name, report_experiment(root, a.experiment, ledger, a.datasets,
                                                a.all_numbers, a.tree))
    else:
        written = [write(root, 'OVERVIEW', report_overview(root, ledger), quiet=True),
                   write(root, 'SEARCHES', report_searches(root, ledger), quiet=True)]
        codes = set(experiment_dirs(root))
        if ledger:
            for (ds, k) in ledger_units(ledger)[0]:
                e = k.split('__', 1)[0]
                if not e.startswith('search'):
                    codes.add(e)
        for code in sorted(codes):
            written.append(write(root, code, report_experiment(root, code, ledger), quiet=True))
            written.append(write(root, code + '_tree', report_experiment(root, code, ledger, tree=True), quiet=True))
        written.append(write(root, 'opig', report_non_dataset(root), quiet=True))
        print(f'Wrote {len(written)} files to {root / "analysis"}')


if __name__ == '__main__':
    main()
