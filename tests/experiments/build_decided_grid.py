"""Write tests/experiments/decided_grid.json from the owner's DECISIONS.md.

    python tests/experiments/build_decided_grid.py

WHY A GENERATED FILE. DECISIONS.md lives in scratchpad/precompute_cuts/ and is
not committed, so the server never sees it. Every experiment reads the
committed JSON through decided_grid.decided_search(); this script is the only
thing that reads DECISIONS.md. Re-run it after any DECISION cell changes and
commit the JSON.

WHAT IT READS. Only the sections whose DECISION cells are filled: Step 1,
Step 2 (v3), Step 3 (v2), the fast-tier table, Step 4, Step 5 (v3), Step 6,
Step 7, Step 8, Step 9. The earlier proposal tables are ignored.

"KEEP ALL STORED". Several decisions say "keep all stored methods" or "keep all
stored rep counts". Which members were stored is read from stored_sets.json
beside DECISIONS.md: the distinct allocation methods, importance methods,
representation counts and internal evaluators present in the slim copies of
result_db (/Volumes/seagate/slim_precompute, pulled 2026-10-03), augmentation
'none', per dataset and per search ('general', 'general_distance').
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CUTS_DIR = REPO / 'scratchpad' / 'precompute_cuts'
DECISIONS = CUTS_DIR / 'DECISIONS.md'
STORED = CUTS_DIR / 'stored_sets.json'
OUT = Path(__file__).resolve().parent / 'decided_grid.json'

GENERAL_BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000]
DISTANCE_BUDGETS = [10, 25, 50, 100, 250, 500]
GENERAL_EVALUATORS = ['match', 'rf', 'lightgbm', 'dnn']
ALL_ALLOC = ['performance_weighted', 'power_2', 'softmax_T1', 'rank_exp_d0.3',
             'wglobal_power2']
ALL_IMP = ['deeplift', 'lightgbm_gain', 'treeshap']


def _section(text, heading_prefix):
    """Rows of the first markdown table after the heading starting with prefix."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith('## ') and l[3:].startswith(heading_prefix))
    rows, in_table = [], False
    for l in lines[start + 1:]:
        if l.startswith('## '):
            break
        if l.startswith('|'):
            cells = [c.strip() for c in l.strip().strip('|').split('|')]
            if not in_table:
                in_table = True          # header row
                continue
            if set(''.join(cells)) <= set('-: '):
                continue                 # separator row
            rows.append(cells)
        elif in_table:
            break
    if not rows:
        raise ValueError(f'no table under "## {heading_prefix}"')
    return rows


def _bold(cell):
    m = re.search(r'\*\*(.+?)\*\*', cell)
    if not m:
        raise ValueError(f'no bold DECISION in cell {cell!r}')
    return m.group(1).strip()


def _range(lo_hi, grid):
    lo, hi = (int(x) for x in lo_hi.split('-'))
    return [b for b in grid if lo <= b <= hi]


def _split(s):
    return [x for x in s.split('/') if x and x != 'none']


def _rc_value(s):
    return None if s in ('', 'all') else int(s)


def _rc_list(stored, cut):
    cut = {_rc_value(c) for c in cut}
    order = [2, 3, 5, None]
    have = {_rc_value(s) for s in stored}
    return [r for r in order if r in have and r not in cut]


def _ordered(values, order):
    return [v for v in order if v in values]


def build():
    text = DECISIONS.read_text()
    stored = json.loads(STORED.read_text())
    out = {}

    for r in _section(text, 'Step 1:'):
        out[r[0]] = {'models': _bold(r[-1]).split('+')}

    budgets = {r[0]: _range(_bold(r[-1]), GENERAL_BUDGETS)
               for r in _section(text, 'Step 2 (v3')}

    fast = {r[0] for r in _section(text, 'Fast-tier datasets')}

    imp = {}
    for r in _section(text, 'Step 3 (v2'):
        d = _bold(r[-1])
        st = stored[r[0]]['general']['importance_method']
        if d.startswith('cut '):
            imp[r[0]] = _ordered(set(st) - set(_split(d[4:])), ALL_IMP)
        elif d == 'keep all stored methods' or d.startswith('lightgbm_gain only'):
            imp[r[0]] = _ordered(set(st), ALL_IMP)
        else:
            raise ValueError(f'Step 3 {r[0]}: unread decision {d!r}')

    alloc = {}
    for r in _section(text, 'Step 4:'):
        ds = r[0].split(' (')[0]
        d = _bold(r[-1])
        st = stored[ds]['general']['allocation_method']
        if d.startswith('cut '):
            alloc[ds] = _ordered(set(st) - set(_split(d[4:])), ALL_ALLOC)
        elif (d == 'keep all stored methods'
              or d.startswith('stored three methods only')
              or d.startswith('HELD')):
            # HELD (suzuki): the cut is not taken, so every stored method runs.
            alloc[ds] = _ordered(set(st), ALL_ALLOC)
        else:
            raise ValueError(f'Step 4 {ds}: unread decision {d!r}')

    evals = {}
    for r in _section(text, 'Step 5 (v3'):
        d = _bold(r[-1])
        if d.startswith('cut '):
            evals[r[0]] = [e for e in GENERAL_EVALUATORS if e not in _split(d[4:])]
        elif d == 'keep all four evaluators':
            evals[r[0]] = list(GENERAL_EVALUATORS)
        else:
            raise ValueError(f'Step 5 {r[0]}: unread decision {d!r}')

    rcs = {}
    for r in _section(text, 'Step 6:'):
        d = _bold(r[-1])
        st = stored[r[0]]['general']['rep_count']
        if d.startswith('cut '):
            rcs[r[0]] = _rc_list(st, _split(d[4:]))
        elif d == 'keep all stored rep counts':
            rcs[r[0]] = _rc_list(st, [])
        else:
            raise ValueError(f'Step 6 {r[0]}: unread decision {d!r}')

    dist = {}
    for r in _section(text, 'Step 7:'):
        ds, brange, icut, acut, ecut, rcut, dec = r
        d = _bold(dec)
        if not d.startswith('as proposed'):
            raise ValueError(f'Step 7 {ds}: unread decision {d!r}')
        if 'allocation cut HELD' in d:
            acut = 'none'
        st = stored[ds]['general_distance']
        dist[ds] = {
            'budgets': _range(brange, DISTANCE_BUDGETS),
            'importance_methods': _ordered(
                set(st['importance_method']) - set(_split(icut)), ALL_IMP),
            'allocation_methods': _ordered(
                set(st['allocation_method']) - set(_split(acut)), ALL_ALLOC),
            'internal_evaluators': [
                e for e in ['knn', 'rf', 'dnn', 'lightgbm', 'svm']
                if e in st['internal_evaluator'] and e not in _split(ecut)],
            'rep_counts': _rc_list(st['rep_count'], _split(rcut)),
        }

    aug_keep = {}
    for r in _section(text, 'Step 8:'):
        d = _bold(r[1])
        if d.startswith('cut'):
            aug_keep[r[0]] = []
        elif d.startswith('keep uncapped'):
            aug_keep[r[0]] = ['uncapped']
        else:
            raise ValueError(f'Step 8 {r[0]}: unread decision {d!r}')

    knn = {}
    for r in _section(text, 'Step 9:'):
        d = _bold(r[1])
        knn[r[0]] = d == 'run'

    for ds, entry in out.items():
        g = {
            'models': list(entry['models']),
            'budgets': budgets[ds],
            'importance_methods': imp[ds],
            'allocation_methods': alloc[ds],
            'internal_evaluators': evals[ds],
            'rep_counts': rcs[ds],
            # Step 8 keeps augmentation on the distance search only; whether
            # the general search keeps it too is OPEN in DECISIONS.md, so the
            # general search runs 'none' alone until she decides.
            'aug_conditions': ['none'],
            'speed': 'fast' if ds in fast else 'full',
        }
        searches = {'general': g}
        if ds in dist:
            searches['general_distance'] = dict(
                dist[ds],
                models=list(entry['models']) + (['knn'] if knn[ds] else []),
                aug_conditions=['none'] + aug_keep[ds],
                speed='fast' if ds in fast else 'full')
        out[ds] = searches

    # suzuki (Adelaide, 2026-10-05): power_2 and performance_weighted score
    # identically on 132 of 144 stored general configurations per model and
    # 147 of 180 distance ones, so they build (nearly) the same hybrid. One
    # is dropped: power_2. The rest of the HELD allocation cut stays uncut.
    for kind in ('general', 'general_distance'):
        if kind in out['suzuki']:
            out['suzuki'][kind]['allocation_methods'] = [
                a for a in out['suzuki'][kind]['allocation_methods'] if a != 'power_2']

    # herg_ic50 (Adelaide, 2026-10-05, option a): not one of the 41, no row
    # in DECISIONS.md. Same target and same 10 representations as herg_ki
    # (noise_helpers.REP_LIST_FROM), so it takes herg_ki's decided models and
    # settings for both searches.
    out['herg_ic50'] = json.loads(json.dumps(out['herg_ki']))

    # chembl_qsar (opig_activity_cliff_qsar's three ChEMBL targets): nothing
    # stored, no DECISIONS.md row. Adelaide, 2026-10-05: settings from a
    # similar dataset (herg_ki: small-molecule binding affinity), two models
    # from simple local fits (default settings, chembl_qsar's pruned
    # representations, 5 scaffold folds, mae). Best mae per model: lightgbm
    # first on all three targets; second dnn on fxa and d2, svm on mpro.
    # Fits on fxa and d2 covered 4 and 2 representations when this was set.
    # Script and logs: session scratchpad qsar_fits/.
    out['chembl_qsar'] = json.loads(json.dumps(out['herg_ki']))
    for kind in out['chembl_qsar']:
        knn_tail = ['knn'] if 'knn' in out['chembl_qsar'][kind]['models'] else []
        out['chembl_qsar'][kind]['models'] = ['lightgbm', 'dnn'] + knn_tail

    # No distance search was ever stored for these three, so Step 7 has no row.
    # Adelaide, 2026-10-05: give them a distance search with their general
    # search's settings and budgets of 100 to 500 features. Models are the
    # two decided ones plus knn where Step 9 runs it.
    for ds in ('bindingdb_kd', 'kiba', 'lhasa_herg'):
        g = out[ds]['general']
        assert 'general_distance' not in out[ds], ds
        out[ds]['general_distance'] = dict(
            g, budgets=[b for b in DISTANCE_BUDGETS if 100 <= b <= 500],
            models=list(g['models']) + (['knn'] if knn[ds] else []),
            aug_conditions=['none'] + aug_keep[ds])

    # Multi-view classification benchmarks (kirby.datasets.multiview): no
    # DECISIONS.md row. Adelaide, 2026-10-23: the search's normal settings --
    # find_best_hybrid's five default downstream models and its default
    # importance methods, allocation methods, internal evaluators and
    # representation counts -- with budgets of 10 to 2000 features. General
    # search only; nothing here measures neighbours.
    for ds in ('mfeat', 'caltech101_7', 'caltech101_20', 'nuswide_object',
               'reuters'):
        assert ds not in out, ds
        out[ds] = {'general': {
            'models': ['dnn', 'lightgbm', 'rf', 'svm', 'xgboost'],
            'budgets': list(GENERAL_BUDGETS),
            'importance_methods': list(ALL_IMP),
            'allocation_methods': list(ALL_ALLOC),
            'internal_evaluators': list(GENERAL_EVALUATORS),
            'rep_counts': [2, 3, 5, None],
            'aug_conditions': ['none'],
            'speed': 'full',
        }}

    # QM9 properties other than the gap (Adelaide, 2026-10-30, for the
    # blueprint transfer): the same molecules and pool as qm9, so qm9's
    # decided models and settings for both searches.
    from kirby.registry import QM9_EXTRA_PROPERTIES
    for p in QM9_EXTRA_PROPERTIES:
        assert f'qm9_{p}' not in out, p
        out[f'qm9_{p}'] = json.loads(json.dumps(out['qm9']))
    return out


def main():
    table = build()
    payload = {
        '_source': ('generated by tests/experiments/build_decided_grid.py from '
                    'scratchpad/precompute_cuts/DECISIONS.md and stored_sets.json; '
                    'do not edit by hand'),
        'datasets': table,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    n_dist = sum('general_distance' in v for v in table.values())
    print(f'wrote {OUT.relative_to(REPO)}: {len(table)} datasets, '
          f'{n_dist} with a distance search decided')


if __name__ == '__main__':
    sys.exit(main())
