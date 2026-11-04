#!/usr/bin/env python3
"""How far each cliff definition sits from a cliff set it did not produce.

HER SENTENCE THIS ANSWERS. "Compare the results of my AC detection methods on
my datasets vs the published AC prediction models on those same datasets (so
how much do my methods agree with GraphCliff, SemiMol, and the published
method) as well as the standard ECFP4 Tanimoto with standard cutoffs (I should
already be producing those given my setup, but I need to highlight them)."

ONE ROW IS one definition against one comparison set, on one fold: the
molecules both call cliff molecules, the molecules only one of them does, and
the Matthews correlation between the two flags. Counts in both directions are
beside the correlation because the two sets can be very different sizes and a
correlation alone hides which way a disagreement runs.

THE COMPARISON SETS, in two kinds.

  The standard cutoffs, computed here. ECFP4 Tanimoto at 0.55 with a
  2-log-unit gap (Hu 2015), the same similarity with twice the median absolute
  deviation, the same with the gap moved to 15 per cent of molecules, and
  MoleculeACE's published any-of-three rule. These are the ones her sentence
  says she is already producing and wants highlighted.

  A cliff flag from outside this repository, read from a file. This is where
  GraphCliff's, SemiMol's and the 2025 foundation-model comparison's
  per-molecule predictions go. NONE OF THE THREE IS ON DISK: no file for any
  of them exists, so every row against them is absent rather than estimated,
  and the result records which were looked for and not found. The format one
  file must have is in `read_external_flags`.

WHAT A ROW DOES NOT SAY. Which of the two sets is right. A standard cutoff is
one published rule, not a reading of nature, and MoleculeACE's own paper's
flag is the output of theirs. A high correlation with a standard means a
definition reproduces that standard; whether that standard finds real cliffs
is the question her three confirmation tests ask and this is one of them.

Usage, as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments definition_agreement
"""
from __future__ import annotations

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

from experiments._shared import (get_domain_metric,                 # noqa: E402
                                 input_fingerprint,
                                 save_experiment_results)
from experiments.noise import cliff_rules as cr                     # noqa: E402
from experiments.cliffs import cliff_sets as P                      # noqa: E402
from experiments.cliffs import definitions as D                     # noqa: E402
from experiments.cliffs import settings as S                        # noqa: E402

EXP_ID = 'definition_agreement'

#: Where a per-molecule cliff flag produced outside this repository goes. One
#: file per (source, dataset). Nothing writes here: a file arrives when a
#: published model has been run or its predictions obtained.
EXTERNAL_DIR = Path(os.environ.get(
    'KIRBY_EXTERNAL_CLIFF_FLAGS',
    '/Volumes/seagate/external_cliff_flags'))

#: The three published cliff-prediction models her outline names, by the file
#: stem each one's flags would arrive under. Looked for every run; recorded as
#: missing when absent, never approximated.
EXTERNAL_SOURCES = ('graphcliff', 'semimol', 'foundation_model_2025')

#: How many rows the json carries per comparison set, best agreement first.
#: Every row is in the jsonl beside it.
TOP_IN_JSON = 50


def read_external_flags(dataset_name, source, smiles):
    """(flag per molecule, record) for one outside source on one dataset.

    THE FORMAT. `<EXTERNAL_DIR>/<source>__<dataset>.json`, holding

        {"source": "graphcliff",
         "what_it_is": "one sentence: what produced this flag",
         "reference": "the paper or the commit it came from",
         "smiles": ["...", "..."],
         "cliff_molecule": [true, false, ...]}

    The flags are matched BY SMILES and not by row order, because an outside
    model's molecule order is its own. A molecule this dataset has and the
    file does not is left out of the comparison and counted, rather than
    treated as a non-cliff: a missing prediction is not a prediction of
    "no".
    """
    p = EXTERNAL_DIR / f'{source}__{dataset_name}.json'
    if not p.exists():
        return None, {'source': source, 'available': False,
                      'looked_for': str(p),
                      'reason': 'no file: this model has not been run on this '
                                'dataset and its predictions are not on disk'}
    blob = json.loads(p.read_text())
    theirs = {str(s): bool(f) for s, f in zip(blob['smiles'],
                                              blob['cliff_molecule'])}
    flag = np.zeros(len(smiles), dtype=bool)
    have = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        if str(s) in theirs:
            have[i] = True
            flag[i] = theirs[str(s)]
    return (flag, have), {
        'source': source, 'available': True, 'file': str(p),
        'what_it_is': blob.get('what_it_is'),
        'reference': blob.get('reference'),
        'n_molecules_in_the_file': len(blob['smiles']),
        'n_molecules_matched_by_smiles': int(have.sum()),
        'n_molecules_with_no_prediction': int((~have).sum()),
        'matched_on': 'the SMILES string, not the row order',
    }


def run(data, config, output_dir, dataset_name, **kwargs):
    """Every definition against every comparison set, per fold."""
    print('\n' + '=' * 70)
    print('DEFINITION AGREEMENT: every definition against a set it did not '
          'produce')
    print('=' * 70)

    stored = S.read()
    labels = np.asarray(data['labels']).ravel()
    smiles = data.get('smiles')
    metric_name, _fn, lower_is_better = get_domain_metric(data)
    scopes = S.values('definition_scope', stored)
    folds, n_folds = P.folds_of(data, config)

    stored_masks = P.load_masks(output_dir, dataset_name)
    if stored_masks is None:
        why = (f'{EXP_ID} reads the per-molecule flags {P.EXP_ID} writes, and '
               f'{P.mask_file(output_dir, dataset_name)} is not there. Run '
               f'{P.EXP_ID} on this dataset first.')
        print(f'  REFUSED: {why}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'refused': why},
               'rows': [], 'result_status': 'failed'}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    # The outside flags, looked for once per dataset.
    external, external_records = {}, []
    for source in EXTERNAL_SOURCES:
        got, record = read_external_flags(
            dataset_name, source,
            smiles if smiles is not None else [None] * len(labels))
        external_records.append(record)
        if got is not None:
            external[source] = got
        print(f"  {source}: "
              f"{'matched ' + str(record['n_molecules_matched_by_smiles']) + ' molecules by SMILES' if record['available'] else record['reason']}",
              flush=True)

    jsonl = Path(output_dir) / EXP_ID / f'{dataset_name}__rows.jsonl'
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if jsonl.exists():
        jsonl.unlink()
    rows_written, by_comparison = 0, {}
    t0 = time.time()

    with open(jsonl, 'a') as sink:
        for scope in scopes:
            for fold_i, train_idx, _val_idx in folds:
                prefix = f'{int(fold_i)}|{scope}|'
                here = {k[len(prefix):]: v
                        for k, v in stored_masks.items()
                        if k.startswith(prefix)}
                if not here:
                    continue
                # (mask, which molecules the set has a prediction for)
                refs = {}
                for rule in D.REFERENCE_RULES:
                    k = f'reference:{rule}'
                    if k in here:
                        refs[f'standard:{rule}'] = (here[k], None)
                for source, (flag, have) in external.items():
                    refs[f'published_model:{source}'] = (
                        flag[np.asarray(train_idx)]
                        if scope == 'per_fold' else flag,
                        have[np.asarray(train_idx)]
                        if scope == 'per_fold' else have)
                for key, ours in here.items():
                    if key.startswith('reference:'):
                        continue
                    for cname, (theirs, have) in refs.items():
                        row = _agreement_row(ours, theirs, have)
                        row.update({'dataset': dataset_name, 'fold': int(fold_i),
                                    'definition_scope': scope,
                                    'definition': key, 'comparison_set': cname})
                        sink.write(json.dumps(row, default=float) + '\n')
                        rows_written += 1
                        bucket = by_comparison.setdefault(cname, [])
                        bucket.append(row)
                # The standards against each other, which is the baseline a
                # definition's agreement is read against.
                names = sorted(refs)
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        row = _agreement_row(refs[a][0], refs[b][0],
                                             refs[b][1])
                        row.update({'dataset': dataset_name,
                                    'fold': int(fold_i),
                                    'definition_scope': scope,
                                    'definition': a, 'comparison_set': b,
                                    'both_are_comparison_sets': True})
                        sink.write(json.dumps(row, default=float) + '\n')
                        rows_written += 1
                print(f'  fold {fold_i}, scope {scope}: '
                      f'{len(here)} definitions against {len(refs)} '
                      f'comparison sets', flush=True)
                sink.flush()

    best = {}
    for cname, bucket in by_comparison.items():
        pooled = {}
        for r in bucket:
            pooled.setdefault((r['definition_scope'], r['definition']),
                              []).append(r)
        summary = []
        for (scope, defn), group in pooled.items():
            mccs = [g['matthews_correlation'] for g in group
                    if g['matthews_correlation'] is not None]
            summary.append({
                'definition': defn, 'definition_scope': scope,
                'n_folds': len(group),
                'matthews_correlation_per_fold': mccs,
                'matthews_correlation_lowest': min(mccs) if mccs else None,
                'matthews_correlation_highest': max(mccs) if mccs else None,
                'n_cliff_molecules_ours_per_fold': [g['n_ours'] for g in group],
                'n_cliff_molecules_theirs_per_fold': [g['n_theirs']
                                                      for g in group],
            })
        summary.sort(key=lambda r: -(r['matthews_correlation_lowest']
                                     if r['matthews_correlation_lowest']
                                     is not None else -2))
        best[cname] = summary[:TOP_IN_JSON]
        if summary:
            top = summary[0]
            print(f"\n  against {cname}: the definition whose worst fold "
                  f"agrees best is\n    {top['definition']}\n    Matthews "
                  f"correlation per fold "
                  f"{[round(v, 4) for v in top['matthews_correlation_per_fold']]}",
                  flush=True)

    meta = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'n_outer_folds': int(n_folds),
        'definition_scopes_run': list(scopes),
        'comparison_sets': sorted(by_comparison),
        'external_sources': external_records,
        'external_directory': str(EXTERNAL_DIR),
        'n_rows': rows_written,
        'rows_file': str(jsonl),
        'rows_in_this_json': TOP_IN_JSON,
        'seconds': round(time.time() - t0, 1),
        'metric': metric_name, 'lower_is_better': lower_is_better,
        'settings': S.record(stored),
    }
    result = {
        'meta': meta,
        'best_per_comparison_set': best,
        'what_one_row_is': (
            'one definition against one comparison set on one fold: the '
            'molecules both call cliff molecules, the molecules only one of '
            'them does, and the Matthews correlation between the two flags.'),
        'what_no_row_claims': (
            'which of the two sets is right. A standard cutoff is one '
            'published rule. Agreeing with it means reproducing it.'),
        'where_every_row_is': str(jsonl),
    }
    save_experiment_results(result, output_dir, EXP_ID, dataset_name,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better,
                            fingerprint=input_fingerprint(data, config))
    print(f'\n  {rows_written} rows written to {jsonl}', flush=True)
    return result


def _agreement_row(ours, theirs, have=None):
    """The counts and the correlation for two per-molecule flags.

    `have` is which molecules the comparison set actually has a prediction
    for. Where it is given, the molecules it does not cover are left out of
    both flags and counted, because a missing prediction is not a prediction
    of "not a cliff".
    """
    a = np.asarray(ours, dtype=bool)
    b = np.asarray(theirs, dtype=bool)
    n_no_prediction = 0
    if have is not None:
        have = np.asarray(have, dtype=bool)
        n_no_prediction = int((~have).sum())
        a, b = a[have], b[have]
    scored = cr.score_against_labels(a, b)
    both = cr.agreement(a, b)
    return {
        'n_molecules_compared': scored['n_molecules'],
        'n_molecules_with_no_prediction': n_no_prediction,
        'n_ours': scored['n_ours'], 'n_theirs': scored['n_theirs'],
        'both': scored['both'], 'ours_only': scored['ours_only'],
        'theirs_only': scored['theirs_only'], 'neither': scored['neither'],
        'recall_of_their_cliffs': scored['recall_of_their_cliffs'],
        'share_of_ours_in_theirs': scored['share_of_ours_in_theirs'],
        'matthews_correlation': scored['matthews_correlation'],
        'jaccard': both['jaccard'],
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--output', default='tests/results/cliffs')
    ap.add_argument('--folds', default=None)
    a = ap.parse_args(argv)
    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401
    if a.folds:
        os.environ['KIRBY_NOISE_ONLY_FOLDS'] = a.folds
    data = load_data(a.dataset)
    run(data, {}, a.output, a.dataset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
