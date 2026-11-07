#!/usr/bin/env python3
"""Every cliff definition scored against MoleculeACE's own cliff flag.

HER SENTENCE THIS ANSWERS. "My methods on MoleculeACE with identified ACs -
see if it actually identifies them."

WHAT IS NEW HERE against tests/experiments/opig_moleculeace_cliff_check.py,
which already scores the NAMED criteria against that flag on all 30 targets.
Two things that file's own record says it does not do:

  it iterates the named criteria rather than the swept cutoffs, so no cutoff
  of TBD3, TBD4 or TBD5 has ever been scored against anything external. Here
  every definition the settings ask for is scored, cutoff by cutoff, which is
  what turns a cutoff from a judgement into a measurement.

  it builds a context from SMILES and a label and loads no representations, so
  13 of the 44 criteria reached its table and no hybrid criterion did. Here the
  representations and the Bemis-Murcko scaffold groups come from the npz files
  build_moleculeace.py writes, so the hybrid criteria run.

ONE ROW IS one definition on one target: the molecules both call cliff
molecules, the molecules only one of them does, and the Matthews correlation.
Counts in both directions sit beside the correlation because the two sets can
be very different sizes and a correlation alone hides which way a disagreement
runs.

WHAT THE FLAG IS AND IS NOT. `cliff_mol` is the output of one published rule:
similarity at or above 0.90 on any of ECFP Tanimoto, scaffold-ECFP Tanimoto or
scaled Levenshtein on the SMILES, with a 1-log-unit potency difference. It is
not a reading of nature. So this measures how far each definition sits from
THEIR rule. Our own re-implementation of their rule should recover the flag
almost exactly, and how close it comes is a check on cliff_rules and not a
finding about cliffs.

⛔ THEIR SPLIT IS NOT USED. The CSVs ship a `split` column. It is a published
split and this lab does not use published or random splits, so the folds here
are scaffold-grouped on our own Bemis-Murcko groups. Under
definition_scope 'per_fold' a definition is computed on a fold's training
molecules and scored against the flag of those same molecules; under
'whole_dataset' it is computed over all of them, which is what has to happen
for a definition whose similarity side read the labels to be comparable with
the existing file's rows.

Usage:
    # one target, as a crash check
    python tests/experiments/cliffs/moleculeace_agreement.py \\
        --targets CHEMBL4203_Ki

    # every target
    python tests/experiments/cliffs/moleculeace_agreement.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.noise import cliff_rules as cr                      # noqa: E402
from experiments.cliffs import definitions as D                      # noqa: E402
from experiments.cliffs import settings as S                         # noqa: E402
from experiments.cliffs.binary_hybrid import (                       # noqa: E402
    EXPECTED_KIND, HYBRID_REP_NAME, binary_hybrid_inputs, require_kind)

#: The 30 CSVs, fetched 2026-11-02 by opig_moleculeace_cliff_check.py.
CSV_DIR = Path(os.environ.get('MOLECULEACE_DATA',
                              '/Volumes/seagate/moleculeace_benchmark'))
#: The npz per target that build_moleculeace.py writes: the representations,
#: the pActivity column and our own scaffold groups.
REP_DIR = Path(os.environ.get('KIRBY_MOLECULEACE_REPS',
                              '/Volumes/seagate/moleculeace_reps'))
OUT_DEFAULT = Path('tests/results/comparisons')

SMILES_COL = 'smiles'
Y_COL = 'y [pEC50/pKi]'
CLIFF_COL = 'cliff_mol'

#: How many folds a per-fold run uses. The same count the rest of the suite
#: runs (orchestrate.plan.n_outer_folds reads it from the run's settings; this
#: script is not a ledger unit, so it takes the suite's own default).
N_FOLDS = 5


def targets():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(str(CSV_DIR / 'CHEMBL*.csv')))


def read_target(name):
    """(smiles, labels, their cliff flag) for one target, from the CSV."""
    path = CSV_DIR / f'{name}.csv'
    smiles, y, cliff = [], [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (SMILES_COL, Y_COL, CLIFF_COL)
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{path} has no column(s) {missing}; found '
                             f'{reader.fieldnames}')
        for rowd in reader:
            try:
                value = float(rowd[Y_COL])
            except (TypeError, ValueError):
                continue
            smiles.append(rowd[SMILES_COL])
            y.append(value)
            flag = str(rowd.get(CLIFF_COL, '')).strip().lower()
            cliff.append(flag in ('1', '1.0', 'true', 'yes'))
    return (smiles, np.asarray(y, dtype=float),
            np.asarray(cliff, dtype=bool))


def read_reps(name, smiles):
    """({representation: matrix}, groups, record) from the npz, or empty.

    The rows are matched by SMILES, not by order: the npz drops molecules
    whose SMILES will not parse, so its row count can be lower than the CSV's.
    A molecule the npz does not carry has no representation, and the
    representation criteria are left out for the whole target rather than run
    on a subset, because a subset is a different set of molecules.
    """
    p = REP_DIR / f'sim_{name}.npz'
    if not p.exists():
        return {}, None, {'available': False, 'looked_for': str(p),
                          'reason': 'no npz: build_moleculeace.py has not run '
                                    'on this target'}
    with np.load(p, allow_pickle=False) as z:
        stored = [str(s) for s in z['smiles']]
        groups = np.asarray(z['groups'])
        names = [k for k in z.files
                 if k not in ('y', 'groups', 'tanimoto', 'smiles')]
        reps = {k: np.asarray(z[k], dtype=np.float64) for k in names}
    where = {s: i for i, s in enumerate(stored)}
    order = [where.get(str(s)) for s in smiles]
    if any(o is None for o in order):
        n_missing = sum(1 for o in order if o is None)
        return {}, None, {
            'available': False, 'file': str(p),
            'reason': f'the npz holds {len(stored)} molecules and {n_missing} '
                      f'of the CSV\'s {len(smiles)} are not in it, so the '
                      f'representations cover a different set of molecules'}
    idx = np.asarray(order, dtype=int)
    return ({k: v[idx] for k, v in reps.items()}, groups[idx],
            {'available': True, 'file': str(p),
             'representations': sorted(reps),
             'n_molecules': len(smiles),
             'matched_on': 'the SMILES string, not the row order',
             'scaffold_groups': int(len(set(groups[idx].tolist())))})


def hybrids_for(reps, labels, groups, name, train_idx, fold_i, kinds_wanted,
                verbose=True):
    """The restricted hybrids for one target, searched on `train_idx`.

    The same door the per-dataset units use, through _shared.fold_searches on
    this target's own representations and scaffold groups.
    """
    from experiments._shared import (build_fold_hybrid, fold_searches,
                                     get_best_for_downstream,
                                     get_model_factories)
    from kirby.find_best_hybrid import BUILTIN_METRICS

    metric_fn, _ = BUILTIN_METRICS['rmse']
    built, refused = {}, {}
    for kind in kinds_wanted:
        try:
            inputs, expected = binary_hybrid_inputs(reps, kind, where=name)
        except ValueError as exc:
            refused[kind] = str(exc)
            continue
        tr = {k: v[train_idx] for k, v in inputs.items()}
        try:
            searches = fold_searches(
                inputs, labels, groups, train_idx=train_idx, fold_i=int(fold_i),
                n_outer_folds=N_FOLDS, dataset_name=f'moleculeace_{name}',
                task_type='regression', metric='rmse',
                cache_namespace=f'cliff_{kind.replace("+", "_")}',
                verbose=False)
        except Exception as exc:
            refused[kind] = f'the search failed: {type(exc).__name__}: {exc}'
            continue
        search = searches.get('none') or next(
            (s for s in searches.values() if s is not None), None)
        if search is None:
            refused[kind] = 'the search returned nothing'
            continue
        for model in ('rf', 'svm'):
            best = get_best_for_downstream(search, model)
            bp = (best or {}).get('blueprint')
            if bp is None:
                refused[f'{kind}/{model}'] = 'no winning recipe for this model'
                continue
            try:
                factory = get_model_factories([model], 'regression',
                                              len(labels)).get(model)
                _X_tr, X_all, _info = build_fold_hybrid(
                    tr, dict(inputs), labels[train_idx], bp,
                    groups=np.asarray(groups)[train_idx],
                    model=(factory() if factory else None),
                    metric_fn=metric_fn)
                X_all = np.asarray(X_all, dtype=np.float64)
                require_kind(X_all, expected, where=f'{name} {kind}')
            except Exception as exc:
                refused[f'{kind}/{model}'] = f'{type(exc).__name__}: {exc}'
                continue
            built[f'{HYBRID_REP_NAME[kind]}__{model}'] = X_all
            if verbose:
                print(f'      {HYBRID_REP_NAME[kind]}__{model}: '
                      f'{X_all.shape[1]} features', flush=True)
    return built, refused


#: Every setting that changes which definitions `D.expand` produces, read off
#: the calls in definitions.py rather than listed from memory: lines 82, 101-105,
#: 144, 157 and 367 of that file.
SETTINGS_THAT_CHANGE_THE_ROWS = (
    'coefficients',
    'representations',
    'similarity_cutoffs',
    'pair_shares',
    'moleculeace_cutoffs',
    'hybrid_overlap_cutoffs',
    'target_cliff_shares',
    'hybrid_feature_kinds',
)


def run_identity(stored, scopes, with_reps):
    """What a stored row was computed under, so a resume cannot serve a row
    from different settings.

    ⛔ WHY THIS EXISTS. Until 2026-11-07 the partial file was keyed on the
    target and the scopes alone. The 2026-11-06 run swept target_cliff_shares
    at 5 to 30 per cent and agreement was still climbing at 30, so the sweep
    has to be re-run with higher shares -- and a plain re-run would have found
    all 30 targets marked done and handed back the 5-to-30 rows under the new
    settings' name.

    The scopes and whether representations were read are in here too: both
    change which rows a target produces.

    ⛔ THE VALUES ARE RESOLVED, NOT COPIED. Nine of the axes ship as the string
    "all", which means every candidate in the same settings file. Storing the
    raw "all" would make a run that edits the `_candidates` list -- which is
    exactly how the share grid grows past 30 per cent -- look identical to the
    run before it. `S.values` turns "all" into the list it stands for.
    `representations` is the one that cannot be resolved here, because "all"
    is each dataset's own pool, so it is stored as written.
    """
    out = {}
    for name in SETTINGS_THAT_CHANGE_THE_ROWS:
        if name == 'representations':
            out[name] = stored.get(name)
            continue
        try:
            out[name] = S.values(name, stored)
        except S.Unset:
            out[name] = None
    return {
        'scopes': list(scopes),
        'with_reps': bool(with_reps),
        'settings': out,
    }


def score_target(name, stored, with_reps=True, scopes=('whole_dataset',),
                 verbose=True):
    """Every definition this target can run, scored against its own flag."""
    smiles, y, theirs = read_target(name)
    reps, groups, rep_record = read_reps(name, smiles) if with_reps \
        else ({}, None, {'available': False, 'reason': 'asked for without '
                                                       'representations'})
    rows, errors, fold_records = [], {}, []

    for scope in scopes:
        folds = ([(0, np.arange(len(y)), np.zeros(0, dtype=int))]
                 if scope == 'whole_dataset'
                 else scaffold_folds(reps, y, groups))
        for fold_i, train_idx, _val_idx in folds:
            all_reps = dict(reps)
            refused = {}
            if reps and groups is not None:
                hyb, refused = hybrids_for(
                    reps, y, groups, name, train_idx, fold_i,
                    S.values('hybrid_feature_kinds', stored), verbose=verbose)
                all_reps.update(hyb)
            whole = cr.CliffContext(y, smiles=smiles, reps=all_reps,
                                    dataset_name=f'moleculeace_{name}')
            ctx = (whole if scope == 'whole_dataset'
                   else cr.subset_context(whole, train_idx))
            flag_here = theirs if scope == 'whole_dataset' else theirs[train_idx]
            defs = D.expand(ctx, stored=stored, scope=scope)
            got, errs, cost = D.masks(ctx, defs)
            for key, why in errs.items():
                errors[f'{scope}|{fold_i}|{key}'] = why
            for d in defs:
                if d.key not in got:
                    continue
                mask, counts = got[d.key]
                scored = cr.score_against_labels(mask, flag_here)
                rows.append({
                    'target': name, 'fold': int(fold_i),
                    'definition': d.key, 'definition_scope': scope,
                    'coefficient': d.record.get('coefficient'),
                    'representation': d.record.get('representation'),
                    'similarity_setting': d.record.get('similarity_setting'),
                    'gap_form': d.record.get('gap_form'),
                    'gap_setting': d.record.get('gap_setting'),
                    'similarity_used_labels': d.record.get(
                        'similarity_used_labels'),
                    'is_reference': d.record.get('is_reference'),
                    'her_label': d.record.get('her_label'),
                    'n_similar_pairs': counts['n_similar_pairs'],
                    'n_cliff_pairs': counts['n_cliff_pairs'],
                    'gap_cutoff_label_units': (counts['gap_info'] or {}).get(
                        'cutoff_label_units'),
                    'n_cliff_molecules_ours': counts['n_cliff_molecules'],
                    **scored})
            fold_records.append({
                'fold': int(fold_i), 'definition_scope': scope,
                'n_molecules': int(ctx.n),
                'n_definitions': len(defs),
                'n_definitions_that_did_not_run': len(errs),
                'hybrids_refused': refused,
                'representations': sorted(all_reps),
                **cost})
            if verbose:
                ours = [r for r in rows
                        if r['fold'] == fold_i and r['definition_scope'] == scope]
                best = max((r for r in ours
                            if r['matthews_correlation'] is not None),
                           key=lambda r: r['matthews_correlation'],
                           default=None)
                print(f"    {name} fold {fold_i} {scope}: {len(defs)} "
                      f"definitions", flush=True)
                if best is not None:
                    print(f"      closest to their flag: "
                          f"{best['definition']}  Matthews "
                          f"{best['matthews_correlation']:.4f}  ours "
                          f"{best['n_ours']} molecules, theirs "
                          f"{best['n_theirs']}", flush=True)

    return {
        'target': name,
        'n_molecules': int(len(y)),
        'n_unique_smiles': len(set(smiles)),
        'n_cliff_molecules_moleculeace': int(theirs.sum()),
        'pct_cliff_molecules_moleculeace': (100.0 * float(theirs.sum())
                                            / len(y) if len(y) else 0.0),
        'label_median_absolute_deviation': float(
            np.median(np.abs(y - np.median(y)))),
        'representations': rep_record,
        'per_fold': fold_records,
        'rows': rows, 'errors': errors,
    }


def scaffold_folds(reps, y, groups):
    """Scaffold-grouped outer folds on this target, the suite's own splitter.

    Their `split` column is a published split and is not read.
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds
    if not reps or groups is None:
        raise ValueError(
            'a per-fold run needs the representations and the scaffold '
            'groups, and this target has none: run build_moleculeace.py.')
    folds = nested_cv_outer_folds(reps, np.asarray(y), np.asarray(groups),
                                  N_FOLDS, 42, 'regression')
    return [(i, np.asarray(tr), np.asarray(va))
            for i, (tr, va) in enumerate(folds)]


def pool(results):
    """One row per definition, summed over targets.

    The Matthews correlation is NOT averaged across targets: it is recomputed
    from the pooled counts, and the per-target values are reported with their
    range so a mean cannot hide a target where a definition failed. The same
    rule opig_moleculeace_cliff_check.pool applies, so the two tables can be
    read against each other.
    """
    acc = {}
    for res in results:
        for row in res['rows']:
            key = (row['definition_scope'], row['definition'])
            a = acc.setdefault(key, {
                'definition': row['definition'],
                'definition_scope': row['definition_scope'],
                'coefficient': row['coefficient'],
                'representation': row['representation'],
                'similarity_setting': row['similarity_setting'],
                'gap_form': row['gap_form'],
                'gap_setting': row['gap_setting'],
                'similarity_used_labels': row['similarity_used_labels'],
                'is_reference': row['is_reference'],
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
                    'matthews_correlation_per_target_lowest': (
                        per[0] if per else None),
                    'matthews_correlation_per_target_highest': (
                        per[-1] if per else None),
                    'n_targets_with_a_correlation': len(per)})
    out.sort(key=lambda r: -(r['matthews_correlation_pooled'] or -2))
    return out


def print_pooled(pooled, limit=25):
    print('\n  pooled over the targets, closest to MoleculeACE\'s own flag '
          'first.')
    print('  "ours only" and "theirs only" are molecules. The correlation is '
          'recomputed from the pooled counts,')
    print('  and the lowest is over targets, so a definition that fails on '
          'one target cannot hide in the pool.\n')
    print(f"    {'definition':54s} {'ours':>7s} {'theirs':>7s} {'both':>7s} "
          f"{'oursOnly':>9s} {'theirsOnly':>11s} {'MCC':>8s} {'lowest':>8s}")
    for r in pooled[:limit]:
        def _f(v):
            return '     n/a' if v is None else f'{v:8.4f}'
        print(f"    {str(r['definition'])[:54]:54s} "
              f"{r['n_cliff_molecules_ours']:>7d} "
              f"{r['n_cliff_molecules_moleculeace']:>7d} {r['both']:>7d} "
              f"{r['ours_only']:>9d} {r['theirs_only']:>11d} "
              f"{_f(r['matthews_correlation_pooled'])} "
              f"{_f(r['matthews_correlation_per_target_lowest'])}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', default=None,
                    help='comma-separated target names; default is all 30')
    ap.add_argument('--scopes', default='whole_dataset',
                    help="comma-separated: whole_dataset, per_fold. A "
                         "per-fold run needs the npz files "
                         "build_moleculeace.py writes.")
    ap.add_argument('--no-reps', action='store_true',
                    help='score only the criteria that need no representation '
                         '(what opig_moleculeace_cliff_check.py does)')
    ap.add_argument('--output', default=str(OUT_DEFAULT))
    ap.add_argument('--data-dir', default=None,
                    help='the folder holding the 30 CHEMBL*.csv files. '
                         'Default: $MOLECULEACE_DATA, then the Seagate path.')
    ap.add_argument('--force-rerun', action='store_true',
                    help='discard the partial file and score every target '
                         'again')
    ap.add_argument('--rep-dir', default=None,
                    help='the folder holding the per-target npz that '
                         'build_moleculeace.py writes. Default: '
                         '$KIRBY_MOLECULEACE_REPS, then the Seagate path.')
    a = ap.parse_args(argv)

    # Named on the command line rather than only in the environment: the
    # Seagate defaults are this laptop's and are wrong on the server, and a
    # run that cannot find the CSVs says so after the fetch and the
    # representation build have already succeeded (2026-11-05).
    global CSV_DIR, REP_DIR
    if a.data_dir:
        CSV_DIR = Path(a.data_dir)
    if a.rep_dir:
        REP_DIR = Path(a.rep_dir)

    names = ([t.strip() for t in a.targets.split(',')] if a.targets
             else targets())
    if not names:
        raise SystemExit(f'no CHEMBL*.csv under {CSV_DIR}')
    scopes = [s.strip() for s in a.scopes.split(',')]
    stored = S.read()
    print(f'\n{len(names)} target(s) from {CSV_DIR}')
    print(f'representations from {REP_DIR}'
          f"{' (not read: --no-reps)' if a.no_reps else ''}")
    print(f'settings {S.path()}')
    for name in S.unset(stored):
        print(f'  NOT SET: {name}')

    # ⛔ RESUME, added 2026-11-05. Each target is appended to
    # <output>/moleculeace_agreement.partial.jsonl as soon as it finishes, and
    # read back on the next attempt. A run of all 30 is 8.8 hours on one core
    # -- the cost goes as the square of the molecule count and the targets run
    # from 615 to 3,657 molecules, so the largest is 67 minutes by itself --
    # and the first attempt was killed by a 4-hour wall after 14 of them. The
    # The identity carries the scopes AND every setting that changes which
    # definitions a target produces (run_identity above), because a row
    # computed under other settings is not this run's row.
    partial = Path(a.output) / 'moleculeace_agreement.partial.jsonl'
    identity = run_identity(stored, scopes, not a.no_reps)
    done, stale = {}, 0
    if partial.exists() and not a.force_rerun:
        for line in partial.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('_identity') == identity:
                done[rec['target']] = rec
            else:
                stale += 1
        if done:
            print(f'  resuming: {len(done)} target(s) already in {partial}',
                  flush=True)
        if stale:
            # Never silent. A row written under other settings is not this
            # run's row, and the count is what says the file was not empty.
            print(f'  {stale} row(s) in {partial} were computed under '
                  f'different settings or scopes and are being recomputed, '
                  f'not reused', flush=True)
            for name in SETTINGS_THAT_CHANGE_THE_ROWS:
                print(f'      {name} = {stored.get(name)!r}', flush=True)

    t0 = time.time()
    partial.parent.mkdir(parents=True, exist_ok=True)
    results, failed = [], {}
    for i, name in enumerate(names):
        if name in done:
            print(f'  [{i + 1}/{len(names)}] {name}: already done', flush=True)
            results.append(done[name])
            continue
        print(f'\n  [{i + 1}/{len(names)}] {name}', flush=True)
        try:
            res = score_target(name, stored, with_reps=not a.no_reps,
                               scopes=scopes)
        except Exception as exc:
            failed[name] = f'{type(exc).__name__}: {exc}'
            print(f'    FAILED {failed[name]}', flush=True)
            continue
        results.append(res)
        with open(partial, 'a') as fh:
            fh.write(json.dumps({**res, '_scopes': list(scopes),
                                 '_identity': identity},
                                default=float) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    pooled = pool(results)
    print_pooled(pooled)
    out = {
        'what_it_is': (
            "every cliff definition the settings ask for, scored against "
            "MoleculeACE's own per-molecule cliff flag, cutoff by cutoff and "
            "with the hybrid criteria included."),
        'what_the_flag_is': (
            'similarity at or above 0.90 on any of ECFP Tanimoto, '
            'scaffold-ECFP Tanimoto or scaled Levenshtein on the SMILES, with '
            'a 1-log-unit potency difference. It is one published rule, not a '
            'reading of nature.'),
        'their_split_was_not_read': (
            'the CSVs ship a `split` column. It is a published split and this '
            'lab does not use published or random splits; the folds here are '
            'scaffold-grouped on our own Bemis-Murcko groups.'),
        'n_targets': len(results), 'targets_that_failed': failed,
        'definition_scopes_run': scopes,
        'csv_directory': str(CSV_DIR),
        'representation_directory': str(REP_DIR),
        'settings': S.record(stored),
        'seconds': round(time.time() - t0, 1),
        'pooled': pooled,
        'per_target': results,
    }
    p = Path(a.output) / 'moleculeace_agreement.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f'\n  {len(results)} target(s), {len(pooled)} definitions, '
          f'{out["seconds"]}s. written {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
