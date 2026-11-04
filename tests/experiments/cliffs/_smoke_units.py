#!/usr/bin/env python3
"""cliff_sets and definition_agreement end to end on a made-up dataset.

WHAT IT CHECKS, which the per-dataset crash check on a real dataset cannot
check quickly, because that one spends its time in the fold's hybrid search:

  the two units' file wiring. cliff_sets writes the counts and the
  per-molecule flags; definition_agreement reads those flags back and refuses
  with a message naming cliff_sets when they are not there.

  that a dataset with no 0/1 representation records why no hybrid was built
  and runs the rest of the definitions anyway, rather than failing.

  that the flags survive the round trip through the bit-packed block, keyed
  the way the reader expects.

  that definition_agreement writes a row for every definition against every
  comparison set, and that the standards are compared with each other too.

WHAT IT DOES NOT CHECK. error_by_cliff_status and cliff_aware_cleanup fit
models through the fold's own search and the decided grid, so neither runs on
a made-up dataset. They are checked on a real one.

    python tests/experiments/cliffs/_smoke_units.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: Large files, logs and temp go on the drive, never /tmp.
OUT_DEFAULT = '/Volumes/seagate/cliff_unit_smoke'

SETTINGS = {
    'coefficients': ['tanimoto', 'manhattan'],
    'representations': 'all',
    'similarity_cutoffs': [0.4, 0.7],
    'pair_shares': [2.0],
    'cutoff_scope': 'global',
    'moleculeace_cutoffs': [0.9],
    'hybrid_overlap_cutoffs': [0.8],
    'target_cliff_shares': [15.0],
    'train_cliff_model_on_moleculeace': False,
    'outlier_detection_methods': None,
    'label_handling_methods': None,
    'outlier_shares': [0.1],
    'definition_scope': ['per_fold'],
    'held_out_cliff_rule': ['same_as_training'],
    'hybrid_feature_kinds': ['bits'],
}


def made_up_dataset(n=90, seed=2):
    """SMILES that parse, a continuous label, and two continuous
    representations, so no 0/1 representation exists and no hybrid can be
    built. That is the path this smoke wants: it must be recorded, not fatal.
    """
    rng = np.random.default_rng(seed)
    smiles = []
    for i in range(n):
        smiles.append('C' * (1 + i % 9) + ('O' if i % 3 else 'N')
                      + ('C' * (i % 4)))
    return {
        'dataset_name': 'made_up',
        'smiles': smiles,
        'labels': rng.normal(size=n) * 1.3,
        'reps': {'cont_a': rng.normal(size=(n, 24)),
                 'cont_b': rng.normal(size=(n, 16))},
        'groups': np.repeat(np.arange(n // 5 + 1), 5)[:n],
        'task_type': 'regression',
        'domain': 'molecular',
    }


def _settings_file(tmp):
    p = Path(tmp) / 'CLIFF_SETTINGS.json'
    p.write_text(json.dumps(SETTINGS, indent=1))
    os.environ['KIRBY_CLIFF_SETTINGS'] = str(p)
    return p


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', default=OUT_DEFAULT)
    a = ap.parse_args(argv)

    drive = Path(a.output).parent
    if str(drive).startswith('/Volumes/') and not drive.exists():
        print(f'NOT RUN: {drive} is not mounted. Pass --output somewhere that '
              f'is; large files, logs and temp go on the drive.')
        return 2

    out = Path(a.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _settings_file(out)

    from experiments.cliffs import cliff_sets as P
    from experiments.cliffs import definition_agreement as A
    from experiments.cliffs import settings as S

    data = made_up_dataset()
    n = len(data['labels'])
    config = {'n_splits': 3}
    os.environ['KIRBY_NOISE_ONLY_FOLDS'] = '0,1'

    print(f'\nsettings {S.path()}')
    print('cliff_sets on a made-up dataset with no 0/1 representation')
    res = P.run(data, config, str(out), 'made_up')
    rows = res['rows']
    assert rows, 'cliff_sets wrote no row'
    folds = sorted({r['fold'] for r in rows})
    assert folds == [0, 1], f'folds in the rows are {folds}, wanted [0, 1]'
    refused = res['meta']['per_fold'][0]['hybrids_refused']
    assert refused, ('a dataset with no 0/1 representation should record why '
                     'no hybrid was built, and it recorded nothing')
    assert all(r['n_molecules'] < n for r in rows), (
        'under the per-fold scope every count should be over the fold\'s '
        'training molecules, which is fewer than the dataset\'s '
        f'{n}')
    refs = [r for r in rows if r['is_reference']]
    assert refs, 'the standard cutoffs she wants highlighted are not in the rows'
    print(f"  {len(rows)} rows over {len(folds)} folds, "
          f"{len(refs)} of them a standard cutoff; no hybrid, recorded as "
          f"{list(refused)[:1]}")

    back = P.load_masks(str(out), 'made_up')
    assert back, 'the per-molecule flags were not written or not read back'
    sample = sorted(back)[0]
    assert sample.startswith('0|per_fold|'), f'unexpected flag key {sample}'
    n_train = res['meta']['per_fold'][0]['n_molecules_in_the_context']
    assert len(back[sample]) == n_train, (
        f'a flag holds {len(back[sample])} molecules and the fold has '
        f'{n_train} training molecules')
    print(f'  {len(back)} flags read back, each over the fold\'s '
          f'{n_train} training molecules')

    print('\ndefinition_agreement reading those flags')
    res2 = A.run(data, config, str(out), 'made_up')
    assert 'refused' not in res2['meta'], res2['meta'].get('refused')
    jsonl = Path(res2['meta']['rows_file'])
    lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert lines, 'definition_agreement wrote no row'
    sets = sorted({r['comparison_set'] for r in lines})
    assert any(s.startswith('standard:') for s in sets), sets
    pairs = [r for r in lines if r.get('both_are_comparison_sets')]
    assert pairs, 'the standards were not compared with each other'
    missing = [r for r in lines if r['n_molecules_with_no_prediction']]
    print(f'  {len(lines)} rows against {len(sets)} comparison sets '
          f'{sets}; {len(pairs)} of them one standard against another; '
          f'{len(missing)} rows had a molecule with no prediction')
    for r in lines[:1]:
        print(f"    one row: {r['definition'][:44]} against "
              f"{r['comparison_set']}: both {r['both']}, ours only "
              f"{r['ours_only']}, theirs only {r['theirs_only']}, Matthews "
              f"{r['matthews_correlation']}")

    print('\ndefinition_agreement with no flags on disk')
    (out / P.EXP_ID / f'made_up{P.MASK_SUFFIX}').unlink()
    res3 = A.run(data, config, str(out), 'made_up')
    assert 'refused' in res3['meta'], 'it ran without the flags it reads'
    assert P.EXP_ID in res3['meta']['refused']
    print(f"  it refuses and names {P.EXP_ID}")

    print(f'\nOK. written under {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
