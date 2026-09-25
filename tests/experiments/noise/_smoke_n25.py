#!/usr/bin/env python3
"""Smoke for n25: exempting the cliff molecules from cleaning.

The thing that has to be true: the two arms of a pair differ in EXACTLY the
molecules the exemption names, and in nothing else. If a smoke cannot show that,
the paired difference the experiment reports cannot be read.

  1. PUT BACK NOTHING AND THE ARMS ARE THE SAME TRAINING SET. An empty exempt
     mask reproduces the cleanup's own output, molecule for molecule.
  2. PUT BACK EVERYTHING AND THE CLEANING IS UNDONE. A full exempt mask
     reproduces the labels, the weights and the rows the fold started with.
  3. THE THREE SHAPES ARE ALL RESTORED. A cleanup that deletes a row puts the
     row back, one that lowers a weight sets it to 1, one that rewrites a label
     restores the label -- and the molecules NOT exempted keep exactly what the
     cleanup gave them.
  4. THE COUNT OF MOLECULES PUT BACK COUNTS ONLY EXEMPT MOLECULES THE CLEANUP
     WOULD HAVE TOUCHED. A setting that touches no cliff molecule reports zero,
     and zero is what marks a degenerate pair.
  5. THE DIFFERENCE IS SIGNED BY THE METRIC'S DIRECTION. The same pair of
     scores gives opposite signs under rmse and under spearman.
  6. THE PREMISE IS COUNTED, NOT ASSUMED. Under outlier_only the unselected
     labels are bit-identical, so a cliff molecule the draw missed carries the
     label the loader produced -- and the run records how many such molecules
     the detector ranks most suspicious.
  7. A CONDITION WHERE EVERY LABEL MOVES IS REFUSED, because then no molecule
     has a correct label and the question does not exist in the data.
  8. The counts are bounded by the passes that ran, and both hybrid spaces are
     reachable.

It runs on a toy dataset with no SMILES, so the cliff reference is the
Euclidean-on-every-representation fallback and the run records that the exempt
set and the suspect set are not independent. That is the branch a TCGA run
takes, and it is the one worth smoking because it is the one that can quietly
mean something else.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process kill it with exit 139 and no traceback --
# project_fold_hook_and_lgbm_segfault_2026_09_02. Both switches, before numpy.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
# This file IS the smoke environment, so the flag the noise harness gates its
# grid shrinking on is set here rather than left to the caller.
os.environ.setdefault('KIRBY_SMOKE_N', '1')

import tempfile

import numpy as np

import experiments.noise.n25_cliff_exemption as n25
from experiments.noise.n07_remediation import cleaned_training_set
from experiments.noise.noise_helpers import noise_spec_for, inject_train_labels
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n25_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=240, n_feat=4, seed=0):
    """Two representations and a label built from the first one.

    Some molecules are given a label from a different molecule, which puts
    structurally similar pairs at very different activities -- what the cliff
    reference is built to find, whether the cause is a cliff or a bad label.
    Groups are singletons: the fold split is not what this file tests.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    w = rng.normal(size=n_feat)
    y = X @ w + rng.normal(0, 0.1, n)
    hit = rng.choice(n, size=n // 10, replace=False)
    y[hit] = y[rng.choice(n, size=n // 10, replace=False)]
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {
        'labels': y,
        'reps': {'toy': X, 'second': X2},
        'groups': np.arange(n),
        'task_type': 'regression',
        'metric': 'rmse',
    }


def fake_recipe(rep_names, budget=4):
    """A fixed recipe standing in for the precomputed pool, as _smoke_n18 does."""
    bp = Blueprint(reps=list(rep_names), allocation_method='fixed',
                   importance_method='lightgbm_gain', budget=budget,
                   internal_evaluator='lightgbm', metric='rmse',
                   lower_is_better=True, task_type='regression',
                   augmentation_strategy='none')
    meta = {'best_config': {'blueprint': bp},
            'objective': 'predictive', 'objective_choice': 'smoke_fixed_recipe',
            'pool_subset': 'predictive_and_similarity', 'tier': 'general',
            'aug_condition': 'none', 'by_objective': {}}
    return lambda *a, **k: (None, meta)


def no_recipe(*a, **k):
    """What the pool returns on a dataset it was never built for."""
    return None, None


def run_it(data, name, recipe=no_recipe, **kw):
    real = n25.get_best_search_for_downstream
    n25.get_best_search_for_downstream = recipe
    try:
        return n25.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n25.get_best_search_for_downstream = real


# ── 1-4: what put_back and n_spared do, with no model fitted ───────────────

def test_put_back():
    print("what the exemption restores:")
    rng = np.random.default_rng(3)
    n = 120
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    scores = rng.random(n)
    exempt = np.zeros(n, dtype=bool)
    exempt[:20] = True

    for cleanup in ('remove', 'downweight', 'knn_smooth', 'label_smooth'):
        plain = cleaned_training_set(cleanup, X, y, scores, 0.2)
        none_back = n25.put_back(plain, np.zeros(n, dtype=bool), X, y)
        all_back = n25.put_back(plain, np.ones(n, dtype=bool), X, y)
        some = n25.put_back(plain, exempt, X, y)

        # 1. nothing put back == the cleanup's own output
        same_rows = len(none_back['y']) == len(plain['y'])
        same_y = same_rows and np.allclose(none_back['y'], plain['y'])
        same_w = ((plain['sample_weight'] is None
                   and none_back['sample_weight'] is None)
                  or (plain['sample_weight'] is not None
                      and np.allclose(none_back['sample_weight'],
                                      plain['sample_weight'])))
        check(f'{cleanup}: putting back nothing reproduces the cleanup',
              same_y and same_w,
              f'{len(none_back["y"])} molecules against {len(plain["y"])}')

        # 2. everything put back == the training set the fold started with
        check(f'{cleanup}: putting back every molecule undoes the cleaning',
              len(all_back['y']) == n and np.allclose(all_back['y'], y)
              and (all_back['sample_weight'] is None
                   or np.allclose(all_back['sample_weight'], 1.0)),
              f'{len(all_back["y"])} of {n} molecules back')

        # 3. the molecules NOT exempted keep what the cleanup gave them
        if plain.get('label_column') is not None:
            col = np.asarray(plain['label_column'], dtype=float)
            kept = ~exempt
            check(f'{cleanup}: an unexempted label is the cleanup\'s, an '
                  f'exempt one is the original',
                  np.allclose(some['label_column'][kept], col[kept])
                  and np.allclose(some['label_column'][exempt], y[exempt]))
        if plain.get('sample_weight') is not None:
            w0 = np.asarray(plain['sample_weight'], dtype=float)
            w1 = np.asarray(some['sample_weight'], dtype=float)
            kept = ~exempt
            # Renormalising multiplies every weight by one constant, so the
            # ratios among the molecules that were not put back must survive.
            ratio = w1[kept] / w0[kept]
            check(f'{cleanup}: the weights of the unexempted molecules keep '
                  f'their ratios',
                  float(ratio.max() - ratio.min()) < 1e-9,
                  f'ratio spread {float(ratio.max() - ratio.min()):.2e}')
            check(f'{cleanup}: an exempt molecule\'s weight is the same as '
                  f'every other exempt molecule\'s',
                  float(w1[exempt].max() - w1[exempt].min()) < 1e-12)
        if plain.get('keep') is not None:
            check(f'{cleanup}: an exempt molecule is never deleted',
                  bool(np.all(some['keep'][exempt])),
                  f'{int(some["keep"].sum())} molecules kept against '
                  f'{int(np.asarray(plain["keep"]).sum())} before')

        # 4. the count of molecules put back
        spared_some = n25.n_spared(plain, some, exempt, y)
        spared_none = n25.n_spared(plain, none_back,
                                   np.zeros(n, dtype=bool), y)
        check(f'{cleanup}: putting back nothing spares 0 of {n} molecules',
              spared_none == 0, str(spared_none))
        check(f'{cleanup}: the molecules spared are exempt molecules the '
              f'cleanup would have touched',
              0 <= spared_some <= int(exempt.sum()),
              f'{spared_some} of {int(exempt.sum())} exempt molecules')


def test_direction():
    print("\nthe metric's direction:")
    # 0.40 rmse against 0.50 rmse is an improvement; 0.40 spearman against
    # 0.50 spearman is not.
    lower = n25._gain(0.40, 0.50, True)
    higher = n25._gain(0.40, 0.50, False)
    check('lower-is-better makes the smaller score the gain',
          lower > 0 and abs(lower - 0.10) < 1e-12, f'{lower:+.4f}')
    check('higher-is-better makes the same pair a loss',
          higher < 0 and abs(higher + 0.10) < 1e-12, f'{higher:+.4f}')


def test_outlier_only_leaves_the_rest_alone():
    print("\nwhat the condition does to the labels:")
    data = make_data(n=200, seed=5)
    y = np.asarray(data['labels'])
    spec = noise_spec_for('toy_n25', 'outlier_only_p05', 0.5, noise_seed=0)
    train_idx = np.arange(len(y))
    y_noisy, cmask, _ = inject_train_labels(y, spec, train_idx=train_idx,
                                            dataset_name=None)
    moved = ~np.isclose(y_noisy, y)
    check('some molecules are corrupted and some are not',
          0 < int(cmask.sum()) < len(y),
          f'{int(cmask.sum())} of {len(y)} molecules corrupted')
    check('every label the draw did not select is bit-identical',
          int((moved & ~cmask).sum()) == 0,
          f'{int((moved & ~cmask).sum())} of {int((~cmask).sum())} '
          f'unselected labels moved')


# ── 5-8: the run ───────────────────────────────────────────────────────────

def test_run_base_only():
    print("\nthe run, single representations only:")
    data = make_data(n=240, seed=1)
    r = run_it(data, 'toy_n25', models=['rf'],
               detectors=['neighbor_residual'],
               cleanups=['remove', 'downweight', 'knn_smooth'],
               flag_rates=[0.10], base_reps={'rf': ['toy', 'second']})

    check('the run is not dropped', 'summary' in r)
    check('it records that no hybrid was available, and why',
          r['hybrid_available'] is False
          and 'precomputed pool' in r['hybrid_unavailable_reason'])
    check('both single-representation spaces ran',
          r['spaces_run'] == ['base_1', 'base_2'], str(r['spaces_run']))
    check('which representations they are is recorded',
          r['base_reps_by_model'] == {'rf': ['toy', 'second']},
          str(r.get('base_reps_by_model')))
    check('one model means one matrix per space, named by the space',
          r['feature_matrices'] == {'base_1': {'rf': 'base_1'},
                                    'base_2': {'rf': 'base_2'}},
          str(r.get('feature_matrices')))
    check('the three arms are named', list(r['arms']) ==
          ['plain', 'cliff_exempt', 'random_exempt'], str(r['arms']))
    check('the condition and the draw are recorded',
          r['noise_condition'] == 'outlier_only_p05'
          and r['noise_seeds'] == [0],
          f"{r['noise_condition']}, draws {r['noise_seeds']}")

    ci = r['cliff_reference']
    check('the cliff count is out of the molecule count',
          0 < ci['n_cliff_molecules'] < ci['n_molecules'],
          f"{ci['n_cliff_molecules']} of {ci['n_molecules']} molecules")
    check('a dataset with no structure reference says so',
          r['cliff_reference_independent'] is False
          and 'contamination_note' in ci)
    check("the cliff definition on the result is n01's",
          ci['definition_source'].startswith('n01.identify_cliff_pairs'),
          ci['definition_source'])

    # THE NUMBER THIS SMOKE IS ASKED FOR.
    counts = r['per_pass_counts']
    print("\n  molecules the exemption holds out, per pass:")
    for c in counts:
        print(f"    {c['pass']}: {c['n_cliff_molecules_in_training']} cliff "
              f"molecules of {c['n_training_molecules']} training molecules; "
              f"{c['n_corrupted_molecules']} of those training molecules were "
              f"corrupted, {c['n_cliff_molecules_corrupted']} of them cliff "
              f"molecules")
    check('the control puts back as many molecules as the cliff arm',
          all(c['n_random_control_molecules']
              == c['n_cliff_molecules_in_training'] for c in counts),
          str([(c['n_cliff_molecules_in_training'],
                c['n_random_control_molecules']) for c in counts]))

    blk = (r['per_space']['base_1']['per_detector']['neighbor_residual']
           ['per_setting_slot']['0.1'])
    row = blk['per_model']['rf']
    check('every cleanup carries a pair count',
          all(c in row for c in r['cleanups']), str(sorted(row)))
    ok_bound = True
    for c, cell in row.items():
        for arm in ('cliff_exempt', 'random_exempt'):
            if cell[arm]['passes_won'] > cell[arm]['passes_scored']:
                ok_bound = False
            if len(cell[arm]['pass_gains']) != cell[arm]['passes_scored']:
                ok_bound = False
    check('passes won never exceeds passes scored, and the gains line up',
          ok_bound)
    check('every cleanup says what it does to a molecule',
          {cell['what_the_cleanup_does'] for cell in row.values()}
          <= {'deletes_the_molecule', 'lowers_the_weight',
              'rewrites_the_label'},
          str({c: cell['what_the_cleanup_does'] for c, cell in row.items()}))

    sp = blk['molecules_spared_by_matrix']['base_1']
    print("\n  molecules each cleanup actually spared, per pass:")
    for c in sorted(sp):
        for pk in sorted(sp[c]):
            e = sp[c][pk]
            print(f"    {c} {pk}: {e['n_cliff_molecules_spared']} cliff "
                  f"molecules of {e['n_cliff_molecules_in_training']} in "
                  f"training, out of {e['n_training_molecules']} training "
                  f"molecules")
    check('no cleanup spares more molecules than the training fold holds cliffs',
          all(e['n_cliff_molecules_spared'] <= e['n_cliff_molecules_in_training']
              for c in sp for e in sp[c].values()))

    prem = (r['per_space']['base_1']['per_detector']['neighbor_residual']
            ['cliff_molecules_among_the_suspicious_by_matrix']['base_1'])
    check('the premise is counted on every pass', len(prem) == 3,
          f'{len(prem)} passes')
    print("\n  of the molecules the detector calls most suspicious:")
    for pk in sorted(prem):
        p = prem[pk]
        print(f"    {pk}: {p['n_most_suspicious_molecules_taken']} taken, "
              f"{p['n_corrupted_molecules_among_the_most_suspicious']} of them "
              f"really corrupted, "
              f"{p['n_uncorrupted_cliff_molecules_among_the_most_suspicious']} "
              f"of them cliff molecules whose label never moved")
    check('no count exceeds the number of molecules taken',
          all(p['n_uncorrupted_cliff_molecules_among_the_most_suspicious']
              <= p['n_most_suspicious_molecules_taken'] for p in prem.values()))

    s = r['summary']['per_space']['base_1']
    check('the summary counts settings, and a degenerate pair is counted apart',
          all('settings_where_nothing_was_spared' in v
              for v in s['per_cleanup'].values() if v.get('settings_scored')))
    check('the weighting and rewriting families are counted separately',
          set(s['by_what_the_cleanup_does']) ==
          {'deletes_the_molecule', 'lowers_the_weight', 'rewrites_the_label'},
          str(sorted(s['by_what_the_cleanup_does'])))
    check('the planned fit count is recorded before the run',
          r['planned_model_fits']['total_model_fits'] > 0,
          str(r['planned_model_fits']['total_model_fits']))
    return r


def test_hybrid_space():
    print("\nthe run, with a recipe:")
    data = make_data(n=200, seed=2)
    r = run_it(data, 'toy_n25_hybrid', recipe=fake_recipe(sorted(data['reps'])),
               models=['rf'], detectors=['neighbor_residual'],
               cleanups=['knn_smooth'], flag_rates=[0.10],
               base_reps={'rf': ['toy', 'second']})
    check('every space ran', r['spaces_run'] == ['base_1', 'base_2', 'hybrid'],
          str(r['spaces_run']))
    got = {}
    for space in r['spaces_run']:
        got[space] = (r['per_space'][space]
                      ['per_model_no_cleaning_mean']['rf'])
    check('the two spaces are genuinely different feature spaces',
          abs(got['base_1'] - got['hybrid']) > 1e-9,
          str({k: round(v, 6) for k, v in got.items()}))
    check('the clean-label ceiling is recorded beside the corrupted score',
          all(np.isfinite(r['per_space'][s]
                          ['per_model_clean_label_ceiling_mean']['rf'])
              for s in r['spaces_run']))


def test_per_model_spaces():
    """Each model in its own spaces. Owner's decision 2026-09-25."""
    print("\neach model in its own spaces:")
    asked = []

    def lookup(*a, downstream_model=None, **k):
        asked.append(downstream_model)
        reps_, budget = {'rf': (['toy', 'second'], 4),
                         'lightgbm': (['toy'], 3)}[downstream_model]
        return fake_recipe(reps_, budget)()

    data = make_data(n=200, seed=4)
    r = run_it(data, 'toy_n25_per_model', recipe=lookup,
               models=['rf', 'lightgbm'], detectors=['neighbor_residual'],
               cleanups=['knn_smooth'], flag_rates=[0.10],
               base_reps={'rf': ['toy', 'second'],
                          'lightgbm': ['second', 'toy']})
    check('the pool was asked once per downstream model',
          sorted(asked) == ['lightgbm', 'rf'], str(asked))
    check('every space ran', r['spaces_run'] == ['base_1', 'base_2', 'hybrid'],
          str(r['spaces_run']))
    check('each model carries its own recipe',
          r['hybrid_recipe_by_model']['rf']['hybrid']['budget'] == 4
          and r['hybrid_recipe_by_model']['lightgbm']['hybrid']['budget'] == 3,
          str(r.get('hybrid_recipe_by_model')))
    want = {sp: {m: f'{sp}/{m}' for m in ('rf', 'lightgbm')}
            for sp in r['spaces_run']}
    check('each model was scored in its own matrix in every space',
          r['feature_matrices'] == want, str(r.get('feature_matrices')))
    det = (r['per_space']['base_1']['per_detector']['neighbor_residual']
           ['detection_by_matrix'])
    check('the detector ran once per matrix',
          set(det) == {'base_1/rf', 'base_1/lightgbm'}
          and all(len(v) == 3 for v in det.values()),
          str({k: len(v) for k, v in det.items()}))
    ok = all(np.isfinite(r['per_space'][sp]['per_model_no_cleaning_mean'][m])
             for sp in r['spaces_run'] for m in ('rf', 'lightgbm'))
    check('every model has a score in every space', ok)
    row = (r['per_space']['hybrid']['per_detector']['neighbor_residual']
           ['per_setting_slot']['0.1']['per_model'])
    check('both models carry the paired arms in the hybrid',
          all(row.get(m, {}).get('knn_smooth', {}).get('cliff_exempt', {})
              .get('passes_scored', 0) > 0 for m in ('rf', 'lightgbm')),
          str({m: row.get(m, {}).get('knn_smooth', {}).get('cliff_exempt', {})
               .get('passes_scored') for m in ('rf', 'lightgbm')}))


def test_no_stored_search_means_no_space():
    print("\nno stored search and no representations given:")
    data = make_data(n=120, seed=5)
    r = run_it(data, 'toy_n25_nothing', models=['rf'],
               detectors=['neighbor_residual'], cleanups=['knn_smooth'],
               flag_rates=[0.10])
    check('no space ran, and the reason is recorded',
          r['spaces_run'] == [] and 'rf' in r['base_reps_missing'],
          f"{r['spaces_run']}, {r.get('base_reps_missing')}")
    import experiments._shared as S
    check('the fixed preference list is gone',
          not hasattr(S, 'NOISE_BASE_REP_PREFERENCE')
          and not hasattr(S, 'pick_noise_base_rep'))


def test_refuses_a_condition_with_no_correct_labels():
    print("\na condition where every label moves:")
    data = make_data(n=120, seed=6)
    try:
        run_it(data, 'toy_n25_bad', models=['rf'],
               detectors=['neighbor_residual'], cleanups=['remove'],
               flag_rates=[0.10], base_reps={'rf': ['toy']},
               noise_condition='gaussian')
        check('gaussian is refused', False, 'it ran')
    except ValueError as exc:
        check('gaussian is refused, and the reason names the premise',
              'no molecule has one' in str(exc), str(exc)[:80])


def test_the_exempt_set_is_built_from_what_the_method_can_see():
    """The exempt set must come from the observed labels, over training rows only.

    ⛔ THIS IS THE EXPERIMENT'S PREMISE. n25 asks what it costs to exempt
    molecules whose label is RIGHT but looks wrong. Until 2026-09-23 the exempt
    set was built once, before the fold loop, from the CLEAN label column and
    across every molecule in the dataset. Two faults in one number: it used
    labels no practitioner holds, and a training molecule's cliff status was
    decided partly by molecules in the held-out fold.

    Measured on 200 synthetic molecules, 160 training: the old mask held 26
    cliff molecules, the same clean labels over training rows alone held 21 --
    five came from held-out molecules. Under observed labels at level 0.75 it
    is 40, and 26 of the 160 are classed differently from what the old code
    exempted. The error grew with the noise, so the conclusion would have been
    most wrong where the noise was worst.
    """
    from experiments.noise.n25_cliff_exemption import cliff_mask_for_dataset
    print("\nthe exempt set, from what the method can see:")
    rng = np.random.RandomState(3)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    clean = X[:, 0] * 3 + rng.normal(scale=0.2, size=n)
    data = {'domain': 'other', 'reps': {'a': X}, 'labels': clean}
    train = np.sort(rng.choice(n, size=160, replace=False))

    whole, info = cliff_mask_for_dataset(data, clean, False)
    sim = info.pop('similarity_matrix')
    check('the similarity matrix comes back for reuse and is popped from info',
          sim is not None and 'similarity_matrix' not in info)

    old_slice = whole[train]
    train_only, _ = cliff_mask_for_dataset(data, clean[train], False,
                                           subset=train, structure=sim)
    check('held-out molecules no longer decide a training molecule is a cliff',
          int(train_only.sum()) != int(old_slice.sum()),
          f'{int(old_slice.sum())} over the whole dataset, '
          f'{int(train_only.sum())} over the training molecules alone')

    y = clean.copy()
    hit = rng.random_sample(len(train)) < 0.10
    y[train[hit]] += rng.normal(scale=0.75 * clean.std() * 3, size=hit.sum())
    observed, _ = cliff_mask_for_dataset(data, y[train], False, subset=train,
                                         structure=sim)
    differ = int((observed != old_slice).sum())
    check('corrupting the labels moves the exempt set, as it must',
          differ > 0,
          f'{differ} of {len(train)} training molecules classed differently '
          f'from the old mask')
    check('the mask is one boolean per TRAINING molecule',
          observed.shape == (len(train),), str(observed.shape))


if __name__ == '__main__':
    test_put_back()
    test_direction()
    test_outlier_only_leaves_the_rest_alone()
    test_run_base_only()
    test_hybrid_space()
    test_per_model_spaces()
    test_no_stored_search_means_no_space()
    test_refuses_a_condition_with_no_correct_labels()
    test_the_exempt_set_is_built_from_what_the_method_can_see()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
