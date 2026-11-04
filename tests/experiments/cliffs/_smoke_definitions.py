#!/usr/bin/env python3
"""The fast sweep against cliff_rules.cliff_sets, definition by definition.

WHY THIS EXISTS. definitions.sweep_one_similarity reads every count off two
sorted vectors instead of building four n-by-n arrays per gap point, because
the per-point version costs 37 minutes a fold on 1,412 molecules. That is only
allowed if it returns the same numbers, so this checks every count of every
definition against cliff_sets on a dataset small enough to run both, including
a dataset with repeat rows of one molecule and a classification label.

It also checks that the exact gap solver puts the number of molecules in a
cliff pair that it says it does, counted from the pair mask and not from its
own arithmetic.

    python tests/experiments/cliffs/_smoke_definitions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.noise import cliff_rules as cr              # noqa: E402
from experiments.cliffs import definitions as D              # noqa: E402
from experiments.cliffs import settings as S                 # noqa: E402

CHECKED = ('n_molecules', 'n_similar_pairs', 'n_cliff_pairs',
           'n_cliff_molecules', 'pct_cliff_molecules',
           'pct_cliff_pairs_of_similar', 'n_cliff_pairs_inside_noise_floor',
           'pct_cliff_pairs_inside_noise_floor',
           'n_similar_pairs_dropped_as_own_repeats')


def _context(n=70, is_clf=False, repeats=False, seed=0):
    rng = np.random.default_rng(seed)
    reps = {
        'bits_a': (rng.random((n, 48)) < 0.25).astype(float),
        'bits_b': (rng.random((n, 32)) < 0.40).astype(float),
        'counts_a': rng.integers(0, 5, (n, 24)).astype(float),
        'hybrid_bits__rf': (rng.random((n, 20)) < 0.35).astype(float),
        'hybrid_counts__svm': rng.integers(0, 3, (n, 18)).astype(float),
    }
    smiles = [f'C{"C" * (i % 11)}O' for i in range(n)]
    if repeats:
        # Three rows of one molecule, which is the herg_ic50 shape: a repeat
        # pair sits at similarity 1.0 and must never be a cliff pair.
        for i in (3, 4, 5):
            smiles[i] = smiles[3]
            reps['bits_a'][i] = reps['bits_a'][3]
    y = (rng.integers(0, 2, n).astype(float) if is_clf
         else rng.normal(size=n) * 1.3)
    return cr.CliffContext(y, smiles=smiles, reps=reps, is_clf=is_clf,
                           dataset_name='smoke')


def _settings():
    """Every axis small enough that both paths run, and still one of each
    form: a bounded coefficient, an unbounded one, a fixed cutoff, a share of
    pairs, the median-absolute-deviation gap and two moved gaps."""
    return {
        'coefficients': ['tanimoto', 'tversky', 'soergel', 'cosine',
                         'manhattan', 'canberra'],
        'representations': 'all',
        'similarity_cutoffs': [0.3, 0.5, 0.7],
        'pair_shares': [1.0, 5.0],
        'cutoff_scope': 'global',
        'moleculeace_cutoffs': [0.9],
        'hybrid_overlap_cutoffs': [0.4, 0.8],
        'target_cliff_shares': [10.0, 15.0, 30.0],
        'train_cliff_model_on_moleculeace': False,
        'definition_scope': 'per_fold',
        'held_out_cliff_rule': 'same_as_training',
        'hybrid_feature_kinds': 'all',
    }


def check_counts(ctx, label):
    """Every definition both ways. Returns how many were compared."""
    defs = D.expand(ctx, stored=_settings())
    got, errors, cost = D.masks(ctx, defs, free_between_groups=False)
    assert not errors, f'{label}: {len(errors)} definitions did not run: ' \
                       f'{list(errors.items())[:3]}'
    assert got, f'{label}: no definition ran at all'
    bad = []
    for d in defs:
        mask_fast, counts_fast = got[d.key]
        slow = cr.cliff_sets(ctx, d.similarity, d.gap)
        for field in CHECKED:
            a, b = counts_fast[field], slow[field]
            same = (a is None and b is None) or (
                a is not None and b is not None
                and abs(float(a) - float(b)) < 1e-9)
            if not same:
                bad.append(f'{d.key} {field}: fast {a!r} slow {b!r}')
        if not np.array_equal(mask_fast, slow['molecule_mask']):
            n_diff = int((mask_fast != slow['molecule_mask']).sum())
            bad.append(f'{d.key} molecule_mask: {n_diff} molecules differ')
    if bad:
        for line in bad[:12]:
            print(f'  MISMATCH {line}')
        raise AssertionError(f'{label}: {len(bad)} mismatches over '
                             f'{len(defs)} definitions')
    print(f'  {label}: {len(defs)} definitions agree on '
          f'{len(CHECKED)} counts and the per-molecule flag '
          f"({cost['n_similarity_criteria']} similarity criteria)")
    return len(defs)


def check_solved_gap(ctx):
    """The exact gap solver's achieved share, counted from the pair mask."""
    fn = cr.coefficient_criterion('bits_a', 'tanimoto', pct=5.0)
    similar, _ = fn(ctx)
    similar = np.asarray(similar, dtype=bool).copy()
    np.fill_diagonal(similar, False)
    for target in (5.0, 15.0, 30.0):
        _m, info = cr.gap_solved_to_target(ctx, target_pct=target,
                                           similar=similar)
        cut = info['cutoff_label_units']
        counted = int((similar & (ctx.gap_matrix > cut)).any(axis=1).sum())
        claimed = round(info['achieved_pct'] * ctx.n / 100.0)
        assert counted == claimed, (
            f'solved gap at target {target}: it claims {claimed} cliff '
            f'molecules and the pair mask holds {counted}')
    curve = cr.gap_curve(ctx, similar)
    assert curve[0]['n_cliff_molecules'] == 0, \
        'the largest gap on the curve should put no molecule in a cliff pair'
    assert curve[-1]['gap_cutoff_label_units'] == 0.0
    mono = [r['n_cliff_molecules'] for r in curve]
    assert mono == sorted(mono), 'the curve must not fall as the gap falls'
    print(f'  the exact gap solver agrees with the pair mask at three '
          f'targets; the curve has {len(curve)} reachable gaps')


def check_settings_refuse():
    """An unset setting stops the run and names itself."""
    try:
        S.values('coefficients', {'coefficients': None})
    except S.Unset as exc:
        assert 'coefficients' in str(exc) and 'TBD1' in str(exc), str(exc)
        print('  an unset setting refuses and names itself and her TBD number')
    else:
        raise AssertionError('an unset setting did not refuse')
    try:
        S.values('coefficients', {'coefficients': ['not_a_coefficient']})
    except ValueError as exc:
        assert 'not_a_coefficient' in str(exc)
        print('  a coefficient the code cannot run is an error, not a skip')
    else:
        raise AssertionError('an unknown coefficient did not refuse')


def main():
    print('\nthe fast sweep against cliff_sets')
    total = 0
    total += check_counts(_context(), 'a regression label')
    total += check_counts(_context(repeats=True), 'with three rows of one '
                                                  'molecule')
    total += check_counts(_context(is_clf=True), 'a class label')
    print('\nthe exact gap solver')
    check_solved_gap(_context())
    print('\nthe settings')
    check_settings_refuse()
    print(f'\nOK: {total} definitions checked both ways')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
