#!/usr/bin/env python3
"""Smoke for n22: the curator flag as ground truth.

The thing that has to be true: when the flagged molecules are the ones whose
labels are wrong, a detector finds them; when the flag was stuck on at random,
the same detector scores what a coin flip scores. If a smoke cannot tell those
two apart, neither can the experiment, and a high number out of the real run
would mean nothing.

  1. FLAG ON THE WRONG ONES. A known subset carries a flag AND a label taken
     from somewhere else. Average precision must be high.
  2. FLAG AT RANDOM. Every label is correct and the same number of molecules
     carry a flag, chosen at random. Average precision must land near the
     flagged share, which is what a detector ranking at random scores.
  3. The two must be far apart. A file that passed 1 and 2 by scoring 0.9 and
     0.85 would have shown nothing.

And four more that are about the plumbing rather than the separation:

  4. `grade` reports the two counts its fractions are shares of, and the
     flagged share a random ranking would score.
  5. A molecule with no score is dropped, not filled in with a number nobody
     computed.
  6. Collapsing records to molecules flags a molecule whose ANY record is
     flagged, and gives it its largest record score.
  7. A dataset that is not herg_ic50 is skipped with a reason, and a pull
     with no flag column is skipped with a different one.
  8. `run` end to end: every record scored in the fold that held it out, both
     gradings written, and a run where no hybrid recipe comes back still
     produces every base-space number rather than being dropped. The base
     spaces are rf's top 2 single representations, base_1 and base_2, by rf's
     score in the stored search (owner's decision 2026-09-25).
  9. With no stored search there is no rf ranking: base_1 and base_2 are
     skipped, base_reps_missing says why, and no fallback list is used.
"""
import sys
import os

# The similarity sweep and the noise cache, both added 2026-09-25. A smoke runs
# today's behaviour -- neighbours in the model-input matrix only -- unless the
# caller names settings in KIRBY_SIMILARITY_SETTINGS, and it never reads or
# writes $KIRBY_CACHE_DIR/noise_cache: a cached build would hide the spy.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
os.environ.setdefault('KIRBY_NOISE_CACHE', '0')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import n22_curator_flag as n22  # noqa: E402
from kirby.utils.noise_estimation import get_noise_estimator  # noqa: E402

OUT = tempfile.mkdtemp(prefix='n22_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


# How many of the 300 molecules carry a flag in both sets. Same number in
# both, so the chance level is the same number in both and the two average
# precisions are comparable without a correction.
N_MOL = 300
N_FLAGGED = 30
CHANCE = N_FLAGGED / float(N_MOL)


def make_set(kind, seed=0):
    """Features, labels and a flag, for one of the two sets.

    kind='wrong'   the flagged molecules are the ones whose label was replaced
                   by a value drawn from the far end of the label range. That
                   is what a record a curator would mark looks like.
    kind='random'  every label is the correct one and the flag is stuck on
                   N_FLAGGED molecules chosen at random, so the flag carries
                   no information about the labels at all.
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N_MOL, 8))
    w = rng.normal(size=8)
    y = X @ w + rng.normal(scale=0.1, size=N_MOL)

    flag = np.zeros(N_MOL, dtype=bool)
    idx = rng.permutation(N_MOL)[:N_FLAGGED]
    flag[idx] = True
    if kind == 'wrong':
        # A label 1.5 standard deviations off its own value. Nothing about
        # the features changes, so the only way to find these is through the
        # label and its neighbours.
        #
        # The size is chosen so the detector scores WELL BELOW 1.0. At 6
        # standard deviations it scored exactly 1.0 on all three seeds, which
        # is a smoke that cannot drop: the grading could lose most of its
        # ordering and still pass. At 1.5 there is room underneath.
        y = y.copy()
        y[idx] = y[idx] + 1.5 * np.std(y) * rng.choice([-1.0, 1.0],
                                                       size=N_FLAGGED)
    return X, y, flag


def average_precision(kind, seed=0):
    """What the neighbour-residual detector scores against the flag.

    The detector is the one n11 runs first and the plainest of the set: a
    molecule's score is how far its label sits from its nearest neighbours'
    labels. Graded through n22.grade, which is the same
    noise_helpers.detection_scores the injected runs go through.
    """
    X, y, flag = make_set(kind, seed=seed)
    est = get_noise_estimator('neighbor_residual')
    sus = np.asarray(est.estimate(X, y), dtype=float)
    out = n22.grade(flag, sus, label='molecules')
    return out, flag


print("SMOKE n22 — the curator flag as ground truth")
print("=" * 70)
print(f"  {N_MOL} molecules, {N_FLAGGED} flagged in each set, so a detector "
      f"ranking at random scores {CHANCE:.4f}")

# ── 1, 2, 3: the separation ──
aps_wrong, aps_random = [], []
for seed in (0, 1, 2):
    w, _ = average_precision('wrong', seed=seed)
    r, _ = average_precision('random', seed=seed)
    aps_wrong.append(w['detection_average_precision'])
    aps_random.append(r['detection_average_precision'])
    print(f"    seed {seed}: flag-on-the-wrong-ones "
          f"{w['detection_average_precision']:.4f}, flag-at-random "
          f"{r['detection_average_precision']:.4f}")

check('flag on the wrong molecules: average precision is high',
      min(aps_wrong) >= 0.70,
      f"lowest of three seeds {min(aps_wrong):.4f}, needed 0.70")

# A random flag will not sit exactly on the chance level on 300 molecules, so
# the band is chance times three rather than chance exactly. Three times 0.10
# is still a quarter of what the wrong-label set has to reach, so a detector
# that found the random flag would still fail this.
check('flag at random: average precision is near the flagged share',
      max(aps_random) <= 3.0 * CHANCE,
      f"highest of three seeds {max(aps_random):.4f}, the flagged share is "
      f"{CHANCE:.4f} and the band is {3.0 * CHANCE:.4f}")

check('the two sets are far apart',
      min(aps_wrong) - max(aps_random) >= 0.40,
      f"lowest wrong-label {min(aps_wrong):.4f} minus highest random "
      f"{max(aps_random):.4f} = {min(aps_wrong) - max(aps_random):.4f}")

# ── 4: no bare fraction ──
out, _ = average_precision('wrong', seed=0)
check('the counts the fractions are shares of are reported',
      out.get('n_flagged') == N_FLAGGED and out.get('n_scored') == N_MOL,
      f"n_flagged {out.get('n_flagged')}, n_scored {out.get('n_scored')}")
check('what a random ranking would score is reported',
      abs(out.get('chance_average_precision', -1) - CHANCE) < 1e-12,
      f"{out.get('chance_average_precision')}")
check('what one entry counts is named',
      out.get('graded_over') == 'molecules', str(out.get('graded_over')))

# ── 5: a molecule with no score is dropped, not filled in ──
X, y, flag = make_set('wrong', seed=0)
est = get_noise_estimator('neighbor_residual')
sus = np.asarray(est.estimate(X, y), dtype=float)
sus_holes = sus.copy()
sus_holes[:20] = np.nan
held = n22.grade(flag, sus_holes, label='molecules')
check('a molecule with no score is dropped rather than filled in',
      held['n_scored'] == N_MOL - 20 and held['n_with_no_score_dropped'] == 20,
      f"scored {held['n_scored']}, dropped "
      f"{held['n_with_no_score_dropped']}")
check('the flagged count follows the drop',
      held['n_flagged'] == int(flag[20:].sum()),
      f"{held['n_flagged']} of the {int(flag.sum())} flagged molecules "
      f"survived the drop")

# ── 6: records collapse to molecules the stated way ──
mol_of_row = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
rec_flag = np.array([False, True, False, False, False, True])
rec_sus = np.array([0.1, 0.9, 0.3, np.nan, 0.5, 0.2])
got_flag = n22._by_molecule(rec_flag, mol_of_row, 3, 'any')
got_sus = n22._by_molecule(rec_sus, mol_of_row, 3, 'max')
check('a molecule is flagged when any of its records is',
      list(got_flag) == [True, False, True], str(list(got_flag)))
check("a molecule's score is its largest record score, skipping the missing",
      np.allclose(got_sus, [0.9, 0.5, 0.2]), str(list(got_sus)))

mol_of_row2 = np.array([0, 0, 1], dtype=np.int64)
all_nan = n22._by_molecule(np.array([np.nan, np.nan, 0.4]), mol_of_row2, 2,
                           'max')
check('a molecule none of whose records kept a score comes back missing',
      not np.isfinite(all_nan[0]) and np.isclose(all_nan[1], 0.4),
      str(list(all_nan)))

# ── 7: the two refusals ──
flag_, text_, why = n22._flag_for_rows({'labels': np.zeros(5)}, 'logd')
check('a dataset with no curator flag is skipped with a reason',
      flag_ is None and 'data_validity_comment' in why, str(why)[:90])

flag2, _t2, why2 = n22._flag_for_rows({'labels': np.zeros(5)}, 'herg_ic50')
check('a row-count mismatch refuses rather than grading the wrong records',
      flag2 is None and ('rows' in why2 or 'column' in why2), str(why2)[:110])

# ── 8: run() end to end ──
# The pool is stubbed out. Asking it for a recipe on 240 made-up rows starts a
# live hybrid search, which ran past ten minutes here and is not what this
# file is checking; a run with no recipe is also case 8's second half.
# No recipe, but a stored search whose rf scores rank repA above repB, so
# base_1 is repA and base_2 is repB. repA carries the labels' signal and repB
# is unrelated noise, so the detector check below reads base_1 = repA; a
# detector in repB has no informative neighbours and scores near chance.
_SEARCH = SimpleNamespace(
    metric_name='spearman', lower_is_better=False,
    baselines={'repA': {'rf': 0.55}, 'repB': {'rf': 0.40}},
    baseline_fold_scores={}, all_results=[])
n22._blueprint_for = lambda *a, **k: (
    None, None, 'the pool is not consulted in this smoke', _SEARCH)

N_REC = 240
_rng = np.random.RandomState(7)
_mol = np.repeat(np.arange(N_REC // 2), 2)   # every molecule measured twice
_X = _rng.normal(size=(N_REC, 6))
_y = _X @ _rng.normal(size=6) + _rng.normal(scale=0.1, size=N_REC)
RUN_FLAG = np.zeros(N_REC, dtype=bool)
_idx = _rng.permutation(N_REC)[:24]
RUN_FLAG[_idx] = True
_y[_idx] += 2.0 * np.std(_y) * _rng.choice([-1.0, 1.0], size=24)

n22._flag_for_rows = lambda d, name: (
    RUN_FLAG, ['Outside typical range' if f else '' for f in RUN_FLAG],
    'synthetic, built in this smoke')

os.environ['KIRBY_NOISE_LIGHT_ESTIMATORS'] = '1'   # the four kNN detectors
_RUN_DATA = {'reps': {'repA': _X, 'repB': _rng.normal(size=(N_REC, 5))},
             'labels': _y, 'task_type': 'regression', 'groups': _mol,
             'replicate_groups': _mol, 'augmentations': None}
res = n22.run(_RUN_DATA, {}, OUT, 'herg_ic50')

check('a run with no hybrid recipe still produces the base-space numbers',
      not res.get('skipped')
      and list(res.get('per_space', {})) == ['base_1', 'base_2'],
      f"spaces {list(res.get('per_space', {}))}, skipped "
      f"{res.get('skipped')}")
check("the base spaces are rf's top 2, in rf's order",
      res.get('base_reps_by_model') == {'rf': ['repA', 'repB']}
      and res['space_provenance']['base_1']['representation'] == 'repA'
      and res['space_provenance']['base_2']['representation'] == 'repB',
      f"{res.get('base_reps_by_model')}")

_sp = 'base_1'
_est = list(res['per_space'][_sp])[0]
check('every record kept a score, in the fold that held it out',
      res['per_space'][_sp][_est]['records']['n_scored'] == N_REC,
      f"{res['per_space'][_sp][_est]['records']['n_scored']} of {N_REC}")
check('both gradings are written, over records and over molecules',
      (res['per_space'][_sp][_est]['records']['graded_over'] == 'records'
       and res['per_space'][_sp][_est]['molecules']['graded_over']
       == 'molecules'
       and res['per_space'][_sp][_est]['molecules']['n_scored']
       == N_REC // 2),
      f"molecules n_scored "
      f"{res['per_space'][_sp][_est]['molecules']['n_scored']}, "
      f"expected {N_REC // 2}")
check('the detectors find labels the flag really sits on',
      min(ap for _n, ap in res['average_precision_ranking'][_sp]) >= 0.70,
      ', '.join(f'{n}={ap:.3f}'
                for n, ap in res['average_precision_ranking'][_sp]))
check('the flag counts and the flag texts are on the result',
      res['n_records_flagged'] == 24 and res['n_records'] == N_REC,
      f"{res['n_records_flagged']} flagged of {res['n_records']}")

# ── 9: no stored search, so no rf ranking and no base space ──
n22._blueprint_for = lambda *a, **k: (
    None, None, 'the pool is not consulted in this smoke', None)
res9 = n22.run(_RUN_DATA, {}, OUT, 'herg_ic50')
check('with no stored search, base_1 and base_2 are skipped with a reason',
      res9.get('base_reps_by_model') == {}
      and 'rf' in res9.get('base_reps_missing', {})
      and res9.get('comparator_spaces') == [],
      f"missing {res9.get('base_reps_missing')}")
check('and with no hybrid either, the run is a skip with a reason, not a crash',
      res9.get('skipped') is True
      and res9.get('policy') == 'no_feature_space',
      f"policy {res9.get('policy')}")

# ── 10: the shared similarity axis, added 2026-09-25 ──
# Three settings: today's (neighbours in the space's own features), rf's rank-1
# single representation weighted by lightgbm_gain, and forest proximity. The
# model_input row at k=10 must be the default path's number to the last bit:
# the same estimator, the same exclude_groups, handed the list it would have
# found itself.
n22._blueprint_for = lambda *a, **k: (
    None, None, 'the pool is not consulted in this smoke', _SEARCH)
_prev_sim = os.environ.get('KIRBY_SIMILARITY_SETTINGS')
os.environ['KIRBY_SIMILARITY_SETTINGS'] = (
    'model_input,base_1|lightgbm_gain|euclidean,forest_proximity|none|forest')
try:
    res10 = n22.run(_RUN_DATA, {}, OUT, 'herg_ic50')
finally:
    os.environ['KIRBY_SIMILARITY_SETTINGS'] = _prev_sim or 'model_input'
_sa = res10.get('similarity_axis') or {}
_same, _n = 0, 0
for _sp in _sa:
    for _e, _v in _sa[_sp].get('model_input|none|process|k=10', {}).items():
        _n += 1
        _same += int(_v['records']['detection_average_precision']
                     == res10['per_space'][_sp][_e]['records'][
                         'detection_average_precision'])
check('the similarity axis writes a grading per space, setting and k',
      set(_sa) == {'base_1', 'base_2'}
      and any('|k=5' in k for k in _sa['base_1'])
      and any(k.startswith('forest_proximity') for k in _sa['base_1']),
      f"{ {sp: len(v) for sp, v in _sa.items()} }")
check('model_input at k=10 is the default path to the last bit',
      _n > 0 and _same == _n, f'{_same} of {_n} detectors identical')
_b2 = _sa.get('base_2', {})
_ap_own = res10['per_space']['base_2']['neighbor_residual']['records'][
    'detection_average_precision']
_ap_b1 = (_b2.get('base_1|lightgbm_gain|euclidean|k=10', {})
          .get('neighbor_residual', {}).get('records') or {}).get(
    'detection_average_precision', float('nan'))
check("in base_2 (noise), neighbours found in base_1 find the flag better",
      _ap_b1 > _ap_own + 0.2, f'{_ap_b1:.3f} against {_ap_own:.3f}')

print("=" * 70)
print(f"  {len(FAIL)} failed" if FAIL else "  all checks passed")
if FAIL:
    for f in FAIL:
        print(f"    FAILED: {f}")
sys.exit(1 if FAIL else 0)
