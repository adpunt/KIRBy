#!/usr/bin/env python3
"""Which molecules are in a cliff pair, under every definition, per fold.

THIS IS THE PRODUCER. The other three experiments in this folder read its
stored per-molecule flags rather than recomputing them, so a cliff set that two
of them disagree about cannot exist.

ONE ROW IS one definition on one fold: a coefficient on a representation, cut
at a fixed value or at a share of pairs, crossed with one way of saying how far
apart the labels have to be. Its numbers are the pairs the similarity side
passed, the pairs the gap side also passed, the molecules in at least one of
them, and the gap that produced them in the dataset's own label units.

WHAT THE COUNTS ARE FOR, and what they are not. Her outline says of the
per-molecule cliff counts scattered through the old experiments: "I want to get
rid of these numbers". They are here because a cutoff cannot be chosen without
them -- TBD6 is stated as a share of molecules, so the share IS the setting --
and they are not a result. Nothing downstream reports them as a finding.

WHAT NO ROW CLAIMS. Not which definition is right. Every count is a count of a
stated rule, and a rule that calls more molecules cliffs is not thereby better.
The two experiments that score a definition against something outside it are
the agreement ones.

A HYBRID ROW'S SIMILARITY SIDE READ THE LABELS. A hybrid's features are chosen
using them. Under definition_scope 'per_fold' the labels it read are the fold's
training labels only, so no held-out label reaches the definition; under
'whole_dataset' every label did. Every row carries both
`similarity_used_labels` and `definition_scope`.

Usage, as a unit of the papers ledger:
    python tests/experiments/run_dataset.py cliffs herg_ki \\
        --experiments cliff_sets
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
from experiments.noise.cliff_census import _spread, own_repeat_spread  # noqa: E402
from experiments.cliffs import definitions as D                     # noqa: E402
from experiments.cliffs import settings as S                        # noqa: E402

EXP_ID = 'cliff_sets'

#: The per-molecule flags go in their own file beside the json, because the
#: json would otherwise carry a boolean vector per row.
MASK_SUFFIX = '__molecule_flags.npz'

#: ⛔ ONE BLOCK PER (fold, scope), NOT ONE ARRAY PER DEFINITION. A dataset's
#: settings can ask for 12,740 definitions, and two definition scopes over five
#: folds is 127,400 flags: as one npz member each that is 127,400 zip members,
#: which np.savez_compressed writes one at a time. They go in instead as one
#: array of shape (definitions, molecules), bit-packed along the molecules, with
#: the definition keys beside it. At 1,412 molecules one block is 1.8 MB and the
#: whole file is about 18 MB.
MASK_BLOCK_FIELDS = ('flags', 'keys', 'n_molecules')


def mask_file(output_dir, dataset_name):
    return Path(output_dir) / EXP_ID / f'{dataset_name}{MASK_SUFFIX}'


def pack_masks(masks):
    """{block name: array} for np.savez_compressed, from
    {(fold, scope): {definition key: flag}}."""
    out = {}
    for (fold, scope), here in masks.items():
        if not here:
            continue
        keys = sorted(here)
        n = len(here[keys[0]])
        flags = np.zeros((len(keys), n), dtype=bool)
        for i, k in enumerate(keys):
            flags[i] = here[k]
        stem = f'{int(fold)}|{scope}'
        out[f'{stem}|flags'] = np.packbits(flags, axis=1)
        out[f'{stem}|keys'] = np.asarray(keys)
        out[f'{stem}|n_molecules'] = np.asarray([n])
    return out


def load_masks(output_dir, dataset_name):
    """{f'{fold}|{scope}|{definition key}': per-molecule flag} written by a
    past run, or None where the file is not there. The other experiments read
    this, so they never recompute a cliff set this one already produced."""
    p = mask_file(output_dir, dataset_name)
    if not p.exists():
        return None
    out = {}
    with np.load(p, allow_pickle=False) as z:
        stems = sorted({k.rsplit('|', 1)[0] for k in z.files
                        if k.endswith('|flags')})
        for stem in stems:
            n = int(z[f'{stem}|n_molecules'][0])
            flags = np.unpackbits(z[f'{stem}|flags'], axis=1, count=n)
            for i, key in enumerate(z[f'{stem}|keys']):
                out[f'{stem}|{str(key)}'] = flags[i].astype(bool)
    return out


def mask_key(fold, definition_key):
    return f'{int(fold)}|{definition_key}'


def folds_of(data, config):
    from experiments.noise.noise_helpers import noise_outer_folds
    return noise_outer_folds(data, config)


def run(data, config, output_dir, dataset_name, **kwargs):
    """Every definition on every outer fold, counts and per-molecule flags."""
    print('\n' + '=' * 70)
    print('CLIFF SETS: every definition, per fold')
    print('=' * 70)

    stored = S.read()
    labels = np.asarray(data['labels']).ravel()
    is_clf = data.get('task_type') == 'classification'
    metric_name, _metric_fn, lower_is_better = get_domain_metric(data)
    scopes = S.values('definition_scope', stored)
    folds, n_folds = folds_of(data, config)

    limit = _max_pair_rows()
    if len(labels) > limit:
        gb = len(labels) ** 2 * 4 / 1e9
        why = (f'{len(labels)} molecules is over the suite\'s own limit on '
               f'building a pairwise matrix ({limit}, n35.MAX_PAIR_ROWS). One '
               f'similarity matrix at this size is {gb:.1f} GB of 32-bit '
               f'numbers and a definition needs two live at once.')
        print(f'  REFUSED: {why}', flush=True)
        out = {'meta': {'dataset': dataset_name, 'refused': why,
                        'n_molecules': int(len(labels))},
               'rows': [], 'result_status': 'skipped', 'skipped': True}
        save_experiment_results(out, output_dir, EXP_ID, dataset_name,
                                metric_name=metric_name,
                                lower_is_better=lower_is_better,
                                fingerprint=input_fingerprint(data, config))
        return out

    spread, spread_info = own_repeat_spread(dataset_name, labels, data=data)
    rows, errors, masks = [], {}, {}
    fold_records, cost_records = [], []

    for scope in scopes:
        for fold_i, train_idx, val_idx in folds:
            t0 = time.time()
            print(f'\n  fold {fold_i} of {n_folds}, definition scope {scope}: '
                  f'{len(train_idx)} training and {len(val_idx)} held-out '
                  f'molecules', flush=True)
            ctx, whole, record = D.context_for_fold(
                data, dataset_name, train_idx=train_idx, fold_i=fold_i,
                n_folds=n_folds, stored=stored, scope=scope)
            for name, why in (record['hybrids_refused'] or {}).items():
                print(f'    no hybrid for {name}: {why}', flush=True)
            defs = D.expand(ctx, stored=stored, scope=scope)
            print(f'    {len(defs)} definitions', flush=True)
            got, errs, cost = D.masks(ctx, defs, verbose=False)
            for key, why in errs.items():
                errors[f'{scope}|{fold_i}|{key}'] = why
            for d in defs:
                if d.key not in got:
                    continue
                mask, counts = got[d.key]
                rows.append(D.row(d, counts, record, extra={
                    'definition_scope': scope,
                    'gap_over_own_repeat_spread': (
                        float(counts['gap_info'].get('cutoff_label_units')
                              / spread)
                        if (spread and counts['gap_info'].get(
                            'cutoff_label_units')) else None),
                    'own_repeat_spread_label_units': spread,
                }))
                if mask is not None:
                    masks.setdefault((int(fold_i), scope), {})[d.key] = mask
            record['seconds'] = round(time.time() - t0, 1)
            record['n_definitions'] = len(defs)
            record['n_definitions_that_did_not_run'] = len(errs)
            fold_records.append(record)
            cost_records.append({'fold': int(fold_i), 'scope': scope,
                                 'seconds': record['seconds'], **cost})
            print(f"    {len(defs) - len(errs)} definitions in "
                  f"{record['seconds']}s, "
                  f"{cost['n_similarity_criteria']} similarity criteria, "
                  f"{cost['megabytes_freed']:.0f} MB of matrices freed",
                  flush=True)

    meta = {
        'dataset': dataset_name,
        'n_molecules': int(len(labels)),
        'n_unique_smiles': (len(set(map(str, data['smiles'])))
                            if data.get('smiles') is not None else None),
        'task_type': data.get('task_type'),
        'label_is_class_mismatch': is_clf,
        'label_spread': _spread(labels),
        'label_median_absolute_deviation': float(
            np.median(np.abs(labels - np.median(labels)))),
        'own_repeat_spread': spread_info,
        'metric': metric_name, 'lower_is_better': lower_is_better,
        'n_outer_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'definition_scopes_run': list(scopes),
        'per_fold': fold_records,
        'cost': cost_records,
        'settings': S.record(stored),
        'kramer_noise_floor_pki': cr.KRAMER_NOISE_FLOOR_PKI,
    }
    result = {
        'meta': meta, 'rows': rows, 'errors': errors,
        'what_one_row_is': (
            'one cliff definition on one fold: a coefficient on a '
            'representation, cut at a fixed value or at a share of pairs, '
            'crossed with one way of saying how far apart the labels have to '
            'be. The gap is in the label\'s own units.'),
        'what_no_row_claims': (
            'which definition is right. A rule that calls more molecules '
            'cliffs is not thereby better; the agreement experiments are the '
            'ones that score a definition against something outside it.'),
        'the_counts_are_a_setting_not_a_result': (
            'the share of molecules in a cliff pair is how TBD6 is stated, so '
            'it is here to choose a cutoff with. It is not reported as a '
            'finding.'),
    }
    if masks:
        p = mask_file(output_dir, dataset_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **pack_masks(masks))
        n_flags = sum(len(v) for v in masks.values())
        meta['molecule_flag_file'] = str(p)
        meta['n_molecule_flags_stored'] = n_flags
        meta['n_molecule_flag_blocks'] = len(masks)
        meta['how_the_flags_are_stored'] = (
            'one array per fold and definition scope, of shape (definitions, '
            'molecules), bit-packed along the molecules, with the definition '
            'keys beside it. cliff_sets.load_masks unpacks it.')
        print(f'\n  {n_flags} per-molecule flags in {len(masks)} block(s) '
              f'written to {p}', flush=True)
    save_experiment_results(result, output_dir, EXP_ID, dataset_name,
                            metric_name=metric_name,
                            lower_is_better=lower_is_better,
                            fingerprint=input_fingerprint(data, config))
    print(f'\n  {len(rows)} rows, {len(errors)} definitions did not run',
          flush=True)
    for k, v in list(errors.items())[:5]:
        print(f'    {k}: {v}')
    return result


def _max_pair_rows():
    """The row count above which the suite does not build a pairwise matrix.

    n35.MAX_PAIR_ROWS, which carries its own reason, so this file does not set
    a second limit."""
    from experiments.noise.n35_near_duplicate_disagreement import MAX_PAIR_ROWS
    return int(MAX_PAIR_ROWS)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset')
    ap.add_argument('--output', default='tests/results/cliffs')
    ap.add_argument('--folds', default=None,
                    help='comma-separated fold numbers, for a crash check')
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
