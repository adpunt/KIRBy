#!/usr/bin/env python3
"""
Every cliff rule in this suite, scored against MoleculeACE's own cliff labels
(OPIG - van Tilborg, Alenicheva & Grisoni 2022).

WHY THIS EXPERIMENT EXISTS. A rule that decides which molecules are activity
cliffs cannot be scored against anything this repo computes, because every such
quantity is another rule. MoleculeACE ships 30 ChEMBL targets with a
`cliff_mol` column its own authors produced. That is the only per-molecule cliff
label in reach that no rule here wrote, so it is the only answer key available.

WHAT THIS IS AND IS NOT. It is a measurement of how far each rule sits from
MoleculeACE's rule, on 48,714 molecules over 30 targets (the files as fetched; 48,707 is the figure quoted for the paper). It is NOT a measurement
of which rule is correct: `cliff_mol` is the output of one published rule
(similarity at or above 0.90 on any of ECFP Tanimoto, scaffold-ECFP Tanimoto or
scaled Levenshtein on the SMILES, with a 1-log-unit potency difference), not a
reading of nature. Two consequences, and both belong in any write-up:

  Our `moleculeace` rule should recover `cliff_mol` almost exactly, because it
  is an independent implementation of the same published definition. How close
  it comes is the check that `cliff_rules.py` is right. A Matthews correlation
  well below 1.0 there is a defect in our code, not a finding about cliffs.

  Every OTHER rule's distance from `cliff_mol` is a disagreement between two
  definitions. It says which rules are near MoleculeACE's and which are far,
  and by how many molecules in each direction. It does not rank them.

WHAT IT PRODUCES, per target and pooled over the 30:
  for each similarity criterion crossed with each label-gap criterion in
  cliff_rules, the molecules both call cliffs, the molecules only ours does,
  the molecules only MoleculeACE does, and the Matthews correlation. Counts in
  both directions are reported beside the correlation because the two sets can
  be very different sizes and a correlation alone hides which way a
  disagreement runs.

WHAT IT DOES NOT DO. No model is fitted. MoleculeACE's own `split` column is
read and discarded: this lab does not use random or published splits, and no
split is needed to compare two labellings of the same molecules.

THE DATA. The 30 CSVs come from molML/MoleculeACE on GitHub, from
MoleculeACE/Data/benchmark_data/ and not from its `old/` or `raw/`
subdirectories. They are downloaded once into --data-dir (default on
/Volumes/seagate) and read from there afterwards, so a rerun needs no network.
An installed MoleculeACE package is used instead when --data-dir is empty and
the package is importable. Columns, verified 2026-11-02 against the file
served: smiles, exp_mean [nM], y, cliff_mol, split, y [pEC50/pKi]. The
pActivity column `y [pEC50/pKi]` is the label, which is the column
build_moleculeace.py also reads; `cliff_mol` is the column
build_moleculeace.py:10 lists as verified and then does not read.

Usage:
    # Fetch the 30 CSVs and score every rule on all of them
    python tests/experiments/opig_moleculeace_cliff_check.py

    # Two targets, for a crash check
    python tests/experiments/opig_moleculeace_cliff_check.py \\
        --targets CHEMBL214_Ki,CHEMBL204_Ki

    # Only the named rules, not the full grid
    python tests/experiments/opig_moleculeace_cliff_check.py --named-only

Reference:
    van Tilborg, Alenicheva & Grisoni (2022). J. Chem. Inf. Model. 62:5938.
    https://github.com/molML/MoleculeACE
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DEFAULT = '/Volumes/seagate/moleculeace_benchmark'
OUT_DEFAULT = 'tests/results/comparisons/opig_moleculeace_cliff_check.json'

sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

from experiments.noise import cliff_rules as cr  # noqa: E402

GITHUB_TREE = ('https://api.github.com/repos/molML/MoleculeACE/git/trees/'
               'main?recursive=1')
GITHUB_RAW = 'https://raw.githubusercontent.com/molML/MoleculeACE/main/'
BENCHMARK_DIR = 'MoleculeACE/Data/benchmark_data/'

SMILES_COL = 'smiles'
# The pActivity column, the one build_moleculeace.py also reads.
Y_COL = 'y [pEC50/pKi]'
CLIFF_COL = 'cliff_mol'

# Criteria that read a loaded representation or a sequence. None applies here:
# these CSVs carry SMILES and a label and nothing else, so the criteria that
# need a representation matrix, a sequence or a shipped pair list are out by
# the data, not by a choice. cliff_rules.available_similarity_criteria says so
# per target and the reasons are stored in the result file.


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

def list_benchmark_files():
    """The 30 CSV names directly inside benchmark_data, from the GitHub tree.

    `old/` and `raw/` hold 30 more each and are deliberately skipped: the
    paper's benchmark is the top-level set.
    """
    with urllib.request.urlopen(GITHUB_TREE, timeout=60) as fh:
        tree = json.load(fh)
    if 'tree' not in tree:
        raise RuntimeError(f'GitHub returned no tree: {str(tree)[:300]}')
    names = []
    for entry in tree['tree']:
        path = entry['path']
        if not path.startswith(BENCHMARK_DIR) or not path.endswith('.csv'):
            continue
        rest = path[len(BENCHMARK_DIR):]
        if '/' in rest or rest == 'datasets.csv':
            continue
        names.append(rest)
    return sorted(names)


def fetch(data_dir, names=None, verbose=True):
    """Download any missing CSV into data_dir. Returns the local paths."""
    os.makedirs(data_dir, exist_ok=True)
    if names is None:
        have = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
        names = have if len(have) >= 30 else list_benchmark_files()
    paths = []
    for name in names:
        dest = os.path.join(data_dir, name)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            url = GITHUB_RAW + BENCHMARK_DIR + name
            try:
                with urllib.request.urlopen(url, timeout=120) as fh:
                    body = fh.read()
            except urllib.error.URLError as exc:
                raise RuntimeError(f'could not fetch {name}: {exc}') from exc
            # Written whole, then moved, so an interrupted download does not
            # leave a short file that the next run reads as complete.
            tmp = dest + '.part'
            with open(tmp, 'wb') as out:
                out.write(body)
            os.replace(tmp, dest)
            if verbose:
                print(f'  fetched {name} ({len(body)} bytes)', flush=True)
        paths.append(dest)
    return paths


def installed_package_files():
    """The same 30 CSVs from an installed MoleculeACE, or an empty list."""
    import glob
    try:
        import MoleculeACE
    except Exception:
        return []
    base = os.path.dirname(MoleculeACE.__file__)
    hits = glob.glob(os.path.join(base, 'Data', 'benchmark_data', '*.csv'))
    return sorted(p for p in hits if not p.endswith('datasets.csv'))


def read_target(path):
    """(smiles, pActivity, cliff_mol) from one benchmark CSV.

    A row whose label will not parse is dropped and counted, rather than given
    a stand-in value.
    """
    smiles, y, cliff, dropped = [], [], [], 0
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (SMILES_COL, Y_COL, CLIFF_COL)
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{os.path.basename(path)} has no column(s) '
                             f'{missing}; found {reader.fieldnames}')
        for row in reader:
            try:
                value = float(row[Y_COL])
            except (TypeError, ValueError):
                dropped += 1
                continue
            smiles.append(row[SMILES_COL])
            y.append(value)
            flag = str(row.get(CLIFF_COL, '')).strip().lower()
            cliff.append(flag in ('1', '1.0', 'true', 'yes'))
    return (smiles, np.asarray(y, dtype=float), np.asarray(cliff, dtype=bool),
            dropped)


# ---------------------------------------------------------------------------
# Scoring one target
# ---------------------------------------------------------------------------

def score_target(path, named_only=False, verbose=True):
    smiles, y, their_cliff, dropped = read_target(path)
    name = os.path.basename(path)[:-4]
    ctx = cr.CliffContext(y, smiles=smiles, dataset_name=name)
    available, excluded = cr.available_similarity_criteria(ctx)

    if named_only:
        wanted = {(s, g) for s, g in cr.RULES.values()}
    else:
        wanted = {(s, g) for s in available for g in cr.GAP_CRITERIA}

    rows, errors = [], {}
    for s, g in sorted(wanted):
        if s not in available:
            errors[f'{s}__{g}'] = excluded.get(s, 'not available')
            continue
        try:
            got = cr.cliff_sets(ctx, s, g)
        except Exception as exc:
            errors[f'{s}__{g}'] = f'{type(exc).__name__}: {exc}'
            continue
        score = cr.score_against_labels(got['molecule_mask'], their_cliff)
        rows.append({
            'similarity': s, 'gap': g,
            'n_cliff_molecules_ours': got['n_cliff_molecules'],
            'pct_cliff_molecules_ours': got['pct_cliff_molecules'],
            'n_cliff_pairs_ours': got['n_cliff_pairs'],
            'n_similar_pairs': got['n_similar_pairs'],
            'gap_cutoff_label_units': got['gap_info'].get('cutoff_label_units'),
            **score})

    # The named rules, pulled out by name.
    named = {}
    for rule, (s, g) in cr.RULES.items():
        hit = next((r for r in rows
                    if r['similarity'] == s and r['gap'] == g), None)
        named[rule] = hit or {'unavailable': excluded.get(s)
                              or errors.get(f'{s}__{g}') or 'not computed'}

    result = {
        'target': name,
        'n_molecules': int(len(y)),
        'n_rows_dropped_unparseable_label': dropped,
        'n_unique_smiles': len(set(smiles)),
        'n_cliff_molecules_moleculeace': int(their_cliff.sum()),
        'pct_cliff_molecules_moleculeace': (
            100.0 * float(their_cliff.sum()) / len(y) if len(y) else 0.0),
        'label_median_absolute_deviation': float(
            np.median(np.abs(y - np.median(y)))),
        'similarity_criteria_available': available,
        'similarity_criteria_excluded': excluded,
        'rows': rows, 'named_rules': named, 'errors': errors,
    }
    if verbose:
        theirs = result['n_cliff_molecules_moleculeace']
        print(f"  {name}: {len(y)} molecules, MoleculeACE calls {theirs} "
              f"({result['pct_cliff_molecules_moleculeace']:.1f}%) cliffs",
              flush=True)
        ours = named.get('moleculeace')
        if ours and 'unavailable' not in ours:
            print(f"    our moleculeace rule: {ours['n_cliff_molecules_ours']} "
                  f"molecules, both {ours['both']}, ours only "
                  f"{ours['ours_only']}, theirs only {ours['theirs_only']}, "
                  f"Matthews "
                  f"{(ours['matthews_correlation'] if ours['matthews_correlation'] is not None else float('nan')):.4f}",
                  flush=True)
    return result


# ---------------------------------------------------------------------------
# Pooling over targets
# ---------------------------------------------------------------------------

def pool(results):
    """One row per rule, summed over targets, plus the per-target spread.

    The Matthews correlation is NOT averaged across targets: it is recomputed
    from the pooled counts, and the per-target values are reported as a median
    with their range so a mean cannot hide a target where a rule failed.
    """
    acc = {}
    for res in results:
        for row in res['rows']:
            key = (row['similarity'], row['gap'])
            a = acc.setdefault(key, {
                'similarity': row['similarity'], 'gap': row['gap'],
                'n_targets': 0, 'n_molecules': 0, 'both': 0, 'ours_only': 0,
                'theirs_only': 0, 'neither': 0, 'per_target_mcc': []})
            a['n_targets'] += 1
            for field in ('n_molecules', 'both', 'ours_only', 'theirs_only',
                          'neither'):
                a[field] += int(row[field])
            if row['matthews_correlation'] is not None:
                a['per_target_mcc'].append(float(row['matthews_correlation']))

    out = []
    for a in acc.values():
        tp, fp, fn, tn = (a['both'], a['ours_only'], a['theirs_only'],
                          a['neither'])
        den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn - fp * fn) / den) if den > 0 else None
        per = sorted(a.pop('per_target_mcc'))
        out.append({**a,
                    'n_cliff_molecules_ours': tp + fp,
                    'n_cliff_molecules_moleculeace': tp + fn,
                    'recall_of_their_cliffs': (tp / (tp + fn)) if (tp + fn) else None,
                    'share_of_ours_in_theirs': (tp / (tp + fp)) if (tp + fp) else None,
                    'matthews_correlation_pooled': mcc,
                    'matthews_correlation_per_target_median': (
                        float(np.median(per)) if per else None),
                    'matthews_correlation_per_target_min': (
                        per[0] if per else None),
                    'matthews_correlation_per_target_max': (
                        per[-1] if per else None),
                    'n_targets_with_a_correlation': len(per)})
    out.sort(key=lambda r: (-(r['matthews_correlation_pooled'] or -2)))
    return out


def print_pooled(pooled, limit=20):
    print('\n  pooled over the targets, best agreement with MoleculeACE\'s '
          'own labels first.')
    print('  "ours only" and "theirs only" are molecules. The correlation is '
          'recomputed from the pooled counts,')
    print('  and the median is over targets, so a rule that fails on one '
          'target cannot hide in the pool.\n')
    header = (f"    {'similarity':30s} {'gap':32s} {'ours':>7s} "
              f"{'theirs':>7s} {'both':>7s} {'oursOnly':>9s} "
              f"{'theirsOnly':>11s} {'MCC':>7s} {'medMCC':>7s} {'minMCC':>7s}")
    print(header)
    for r in pooled[:limit]:
        def _f(v):
            return 'n/a' if v is None else f'{v:7.4f}'
        print(f"    {r['similarity']:30s} {r['gap']:32s} "
              f"{r['n_cliff_molecules_ours']:>7d} "
              f"{r['n_cliff_molecules_moleculeace']:>7d} {r['both']:>7d} "
              f"{r['ours_only']:>9d} {r['theirs_only']:>11d} "
              f"{_f(r['matthews_correlation_pooled'])} "
              f"{_f(r['matthews_correlation_per_target_median'])} "
              f"{_f(r['matthews_correlation_per_target_min'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default=None,
                    help='comma-separated CSV stems, for example '
                         'CHEMBL214_Ki. Default: all 30.')
    ap.add_argument('--data-dir', default=DATA_DEFAULT,
                    help=f'where the CSVs live. Default {DATA_DEFAULT}')
    ap.add_argument('--output', default=OUT_DEFAULT)
    ap.add_argument('--named-only', action='store_true',
                    help='score only the named rules in cliff_rules.RULES, '
                         'not every similarity-by-gap pairing')
    ap.add_argument('--force-rerun', action='store_true',
                    help='discard targets finished by an earlier attempt '
                         '(<output>.partial.jsonl) and recompute every one')
    ap.add_argument('--no-fetch', action='store_true',
                    help='never reach the network; use --data-dir or an '
                         'installed MoleculeACE package only')
    args = ap.parse_args()

    names = ([t.strip() + '.csv' if not t.strip().endswith('.csv')
              else t.strip() for t in args.targets.split(',')]
             if args.targets else None)

    paths = []
    if args.no_fetch:
        if os.path.isdir(args.data_dir):
            paths = sorted(os.path.join(args.data_dir, f)
                           for f in os.listdir(args.data_dir)
                           if f.endswith('.csv') and f != 'datasets.csv')
        if not paths:
            paths = installed_package_files()
        if names:
            keep = set(names)
            paths = [p for p in paths if os.path.basename(p) in keep]
        if not paths:
            print(f'no CSVs in {args.data_dir} and MoleculeACE is not '
                  f'importable. Drop --no-fetch to download them.')
            return 2
    else:
        paths = fetch(args.data_dir, names)

    print('=' * 78)
    print('CLIFF RULES AGAINST MOLECULEACE\'S OWN LABELS')
    print('  van Tilborg, Alenicheva & Grisoni 2022, JCIM 62:5938')
    print(f'  targets: {len(paths)}')
    which = ('the named ones' if args.named_only
             else 'every similarity criterion by every gap criterion')
    print(f'  rules: {which}')
    print('  cliff_mol is MoleculeACE\'s rule, not nature. Our moleculeace '
          'rule recovering it is')
    print('  a check on this code; any other rule\'s distance from it is a '
          'disagreement between')
    print('  two definitions, not a ranking.')
    print('=' * 78)

    # Resume. Each target is appended to <output>.partial.jsonl as soon as it
    # finishes and read back on the next attempt, so an interrupted run of 30
    # targets does not start over. The identity carries whether the full grid
    # or only the named rules was asked for, because the two produce different
    # rows for the same target. --force-rerun discards the file.
    partial = args.output + '.partial.jsonl'
    done = {}
    if os.path.exists(partial) and not args.force_rerun:
        with open(partial) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('named_only') == bool(args.named_only):
                    done[rec['target']] = rec
        if done:
            print(f'  resuming: {len(done)} target(s) already finished in '
                  f'{partial}')
    elif args.force_rerun and os.path.exists(partial):
        os.remove(partial)

    t0 = time.time()
    results, failed = [], []
    for path in paths:
        target = os.path.basename(path)[:-4]
        if target in done:
            results.append(done[target])
            print(f'  {target}: already done, read back', flush=True)
            continue
        try:
            res = score_target(path, named_only=args.named_only)
        except Exception as exc:
            print(f'  FAILED {os.path.basename(path)}: '
                  f'{type(exc).__name__}: {exc}')
            failed.append(os.path.basename(path))
            continue
        results.append(res)
        os.makedirs(os.path.dirname(partial) or '.', exist_ok=True)
        with open(partial, 'a') as fh:
            fh.write(json.dumps({**res, 'named_only': bool(args.named_only)},
                                default=float) + '\n')

    pooled = pool(results) if results else []
    print_pooled(pooled)

    payload = {
        'experiment': 'opig_moleculeace_cliff_check',
        'source': 'van Tilborg, Alenicheva & Grisoni 2022, JCIM 62:5938; '
                  'CSVs from molML/MoleculeACE, benchmark_data (not old/ or '
                  'raw/)',
        'what_cliff_mol_is': (
            'MoleculeACE\'s own per-molecule cliff flag: similarity at or '
            'above 0.90 on any of ECFP Tanimoto, scaffold-ECFP Tanimoto or '
            'scaled Levenshtein on the SMILES, with a 1-log-unit potency '
            'difference. It is one published rule, not a reading of nature.'),
        'how_to_read_it': (
            'Our moleculeace rule is an independent implementation of the '
            'same definition, so its agreement is a check on cliff_rules.py. '
            'Every other rule\'s distance from cliff_mol measures how far two '
            'definitions sit apart and in which direction; it does not rank '
            'them.'),
        'split_column_ignored': (
            'MoleculeACE ships a train/test split column. It is read and '
            'discarded: no split is needed to compare two labellings of the '
            'same molecules, and this lab does not use published or random '
            'splits.'),
        'named_only': bool(args.named_only),
        'n_targets': len(results),
        'targets_failed': failed,
        'seconds': round(time.time() - t0, 1),
        'pooled': pooled,
        'per_target': results,
    }
    out = args.output
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f'\n  written {out} ({payload["seconds"]}s, {len(results)} targets'
          + (f', {len(failed)} failed' if failed else '') + ')')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
