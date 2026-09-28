#!/usr/bin/env python3
"""Smoke for n24: delete the flagged molecules, or keep them.

The thing that has to be true: the control is real. An experiment that deletes
molecules a detector chose has to be measured against deleting the same number
of molecules at random, and if a smoke cannot show that the two arms deleted the
same count on the same passes, the experiment cannot be read at all.

  1. THE CONTROL DELETES THE SAME NUMBER AS THE DETECTOR DID. Counted out of the
     result file, per flag rate, summed over the passes.

  2. THE CONTROL IS REPORTED PER SEED AND NEVER AVERAGED. Every seed has its own
     row, and the best and the worst are both on the file.

  3. EVERY ARM IS SCORED ON THE SAME PASSES AND THE COMPARISONS ARE PAIRED.
     Passes won never exceeds passes paired, and passes paired never exceeds the
     (draw, fold) passes that ran.

  4. THE DIRECTION IS READ OFF THE DATASET. Run once where lower wins and once
     where higher wins. Two checks, and neither is "the same arm wins under both
     metrics", which would be a claim about the arms: the gain under one
     direction must be the gain under the other negated, and the passes the
     summary says an arm won must be the passes where its raw score really is on
     the better side of no cleaning.

  4b. A RATE-BLIND ARM IS BUILT ONCE, NOT ONCE PER RATE. Counted by wrapping
     `cleaned_training_set`, and the repeated rows must carry the same numbers.

  4c. THE PARSER BUILDS AND CARRIES EVERY FLAG, checked by running --help.

  5. THE FOUR RATE-BLIND ARMS REALLY ARE RATE-BLIND. Checked against
     `cleaned_training_set` itself rather than asserted from the table: the
     cleaned label column at two flag rates is compared, and the two arms that
     read the rate must differ while the four that do not must be identical.

  6. THE EXACT-COUNT RULE HOLDS. `random_deletion_scores` must make `remove`
     delete exactly the number asked for, over many (training count, deleted
     count) pairs.

  7. WHAT THE CONDITION LEFT BEHIND IS MEASURED. Under outlier_p10 the widened
     molecules carry only part of the squared label error, so deleting exactly
     them cannot remove all of the damage; under outlier_only_p10 they carry all
     of it. Both are run and the two shares must come out different.

RUN COST. KIRBY_SMOKE_N sets how many molecules the toy dataset carries, so the
whole file stays under a couple of minutes. Unset it defaults to 180.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Two OpenMP runtimes in one process: torch links libiomp5 and LightGBM links
# libomp, and a multi-threaded fit of one after the other kills the process with
# exit 139 and no traceback. project_fold_hook_and_lgbm_segfault_2026_09_02.
os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import tempfile

import numpy as np

import experiments.noise.n24_delete_or_keep as n24
from experiments.noise.n07_remediation import cleaned_training_set

OUT = tempfile.mkdtemp(prefix='n24_smoke_')
FAIL = []

# 2026-09-25: n24 now caches through noise_cache under KIRBY_CACHE_DIR, so the
# toy entries go to a temporary directory, as
# tests/smoke/smoke_noise_similarity_foundation.py does. Every test but
# test_similarity_sweep runs today's path only; that one names its settings.
os.environ['KIRBY_CACHE_DIR'] = tempfile.mkdtemp(prefix='n24_smoke_cache_')
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')

# How many molecules the toy dataset carries. KIRBY_SMOKE_N is the repository's
# own row cap and it is read here for the same purpose.
try:
    SMOKE_N = int(os.environ.get('KIRBY_SMOKE_N', '').strip() or 180)
except ValueError:
    SMOKE_N = 180


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def make_data(n=None, seed=0, domain='molecular'):
    """Two representations. The label is a function of the first one.

    Groups are singletons: the fold split is not what this file is testing.

    `domain` sets the direction, because that is where the direction actually
    comes from: an unregistered dataset falls through to
    `_shared._heuristic_metric_name`, which gives a molecular regression set
    rmse (lower wins) and an antibody one spearman (higher wins). Passing a
    'metric' key here would change nothing — get_domain_metric never reads one.
    """
    n = SMOKE_N if n is None else n
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 6))
    y = a @ rng.normal(size=6) + rng.normal(0, 0.25, n)
    b = a[:, ::-1] + rng.normal(0, 0.6, size=a.shape)
    return {
        'labels': y,
        'reps': {'aaa': a, 'bbb': b},
        'groups': np.arange(n),
        'task_type': 'regression',
        'domain': domain,
    }


def no_recipe(*a, **k):
    """What the pool returns on a dataset it was never built for.

    The hybrid spaces need a precomputed pool and there is none here, so the
    lookup is replaced and every arm runs in the single representations the
    caller names.
    """
    return None, None


def run_it(data, name, recipe=no_recipe, **kw):
    """One tiny n24 run. By default one model and one single representation
    (space base_1 only), so the fit counts below stay per one matrix."""
    real = n24.get_best_search_for_downstream
    n24.get_best_search_for_downstream = recipe
    try:
        return n24.run(
            data, {'n_splits': 3}, OUT, name,
            models=kw.pop('models', ['rf']),
            base_reps=kw.pop('base_reps', {'rf': ['aaa']}),
            detectors=kw.pop('detectors', ['neighbor_residual']),
            flag_rates=kw.pop('flag_rates', [0.05, 0.20]),
            random_seeds=kw.pop('random_seeds', [0, 1, 2]),
            levels=kw.pop('levels', [0.5]),
            draws=kw.pop('draws', [0]),
            **kw)
    finally:
        n24.get_best_search_for_downstream = real


# ── 1, 2, 3. the control ────────────────────────────────────────────────────

def test_the_control():
    print("the control — deleting the same number at random:")
    r = run_it(make_data(seed=1), 'toy')

    # dirty-local 2026-09-28: ensure_dprime_models adds svm (2026-09-25), and
    # svm has no single representation here, so rf's matrix is not shared by
    # every model and its record is keyed 'base_1/rf' (n24: ckey).
    _dl = r['molecules_deleted']['0.5']
    deleted = _dl['base_1'] if 'base_1' in _dl else _dl['base_1/rf']
    who = sorted(deleted)
    # dirty-local 2026-09-28: since a9a933b the random seeds' counts are kept
    # per detector, under '<detector>/random_seed<k>'.
    _rand = 'neighbor_residual/random_seed'
    check('the detector and every random seed have their own deletion counts',
          'neighbor_residual' in who
          and len([w for w in who if w.startswith(_rand)]) == 3,
          f'{who}')

    same = True
    for rate in sorted(deleted['neighbor_residual']):
        by_det = deleted['neighbor_residual'][rate]
        line = [f"the detector deleted "
                f"{by_det['molecules_deleted_summed_over_the_passes']} of "
                f"{by_det['training_molecules_summed_over_the_passes']} "
                f"training molecules summed over "
                f"{by_det['passes']} passes"]
        for w in who:
            if not w.startswith(_rand):
                continue
            row = deleted[w].get(rate)
            if row is None:
                same = False
                continue
            if (row['molecules_deleted_summed_over_the_passes']
                    != by_det['molecules_deleted_summed_over_the_passes']):
                same = False
            line.append(f"{w} deleted "
                        f"{row['molecules_deleted_summed_over_the_passes']}")
        print(f"        rate {rate}: " + "; ".join(line))
    check('every random seed deleted the same number of molecules as the '
          'detector did', same)

    # How many of the deleted molecules were ones the injector widened, for the
    # detector and for chance. Printed rather than asserted: what it should be
    # is the experiment's finding, not the smoke's.
    for rate in sorted(deleted['neighbor_residual']):
        d = deleted['neighbor_residual'][rate]
        print(f"        rate {rate}: of the "
              f"{d['molecules_deleted_summed_over_the_passes']} molecules the "
              f"detector deleted, "
              f"{d['widened_molecules_among_the_deleted_summed_over_the_passes']}"
              f" were among the "
              f"{d['widened_molecules_summed_over_the_passes']} the injector "
              f"widened")

    lvl = r['summary']['per_level']['0.5']['per_space']['base_1']
    rrow = lvl['per_detector']['neighbor_residual']['per_flag_rate']
    ok_per_seed = True
    for rate, block in rrow.items():
        for m, csp in block['random_deletion_against_no_cleaning'].items():
            if len(csp['per_seed']) != 3:
                ok_per_seed = False
            gains = [f"{k.split('/')[-1]} {v['passes_won']} of "
                     f"{v['n_passes_paired']} passes at {v['mean_gain']:+.4f}"
                     for k, v in csp['per_seed'].items()]
            print(f"        rate {rate}, {m}: " + "; ".join(gains))
    check('every seed has its own row and none of them is averaged away',
          ok_per_seed and all(
              'best_seed_mean_gain' in csp and 'worst_seed_mean_gain' in csp
              for block in rrow.values()
              for csp in block['random_deletion_against_no_cleaning'].values()))
    check('no mean across seeds is written anywhere in the summary',
          'mean_across_seeds' not in str(r['summary']))

    # Every arm that is not the control carries its comparison against every
    # seed, and the verdict is the conjunction over them.
    checked, ok_bound, ok_vs = 0, True, True
    n_passes_run = 3          # one draw, three folds
    for rate, block in rrow.items():
        for arm, arow in block['per_arm'].items():
            for m, entry in arow['per_model'].items():
                nc = entry['vs_no_cleaning']
                checked += 1
                if nc['passes_won'] > nc['n_passes_paired']:
                    ok_bound = False
                if nc['n_passes_paired'] > n_passes_run:
                    ok_bound = False
                if len(nc['pass_gains']) != nc['n_passes_paired']:
                    ok_bound = False
                if r['arm_families'][arm] != 'delete_at_random':
                    if entry.get('random_seeds_compared') != 3:
                        ok_vs = False
                    want = all(v['passes_won'] * 2 > v['n_passes_paired']
                               for v in entry['vs_random_deletion'].values())
                    if entry['beat_every_random_seed'] != want:
                        ok_vs = False
    check('passes won never exceeds passes paired, and the gains line up',
          ok_bound, f'{checked} readings')
    check('every detector-driven arm is compared with all three seeds and the '
          'verdict is the conjunction', ok_vs)

    c = r['summary']['counts']
    print(f"        beat every random seed AND no cleaning: "
          f"{c['settings_beating_every_random_seed_AND_no_cleaning']}; "
          f"beat no cleaning only: "
          f"{c['settings_beating_no_cleaning_but_not_every_random_seed']}; "
          f"beat the seeds but LOST to no cleaning: "
          f"{c['settings_beating_every_random_seed_but_LOSING_to_no_cleaning']}")
    check('a setting is defined in words on the file',
          'downstream model' in c['what_a_setting_is'])

    # ⛔ THE HEADLINE COUNT MUST NOT INCLUDE SETTINGS THAT LOST TO NOT
    # CLEANING. Until 2026-09-23 it did: the bucket was filled by
    # beat_every_random_seed alone and the no-cleaning comparison was only
    # reached in the elif, so an arm that cleared all three random deletions
    # and was WORSE than leaving the labels alone counted as a win. Rebuilt
    # here from the per-setting entries, which are the same numbers the count
    # is made of, so the two cannot drift apart silently.
    won_both, won_nc_only, won_seeds_only = [], [], []
    for lvl, lrow in r['summary']['per_level'].items():
        for space, srow in lrow['per_space'].items():
            for det, drow in srow['per_detector'].items():
                for rate, rrow in drow['per_flag_rate'].items():
                    for arm, arow in rrow['per_arm'].items():
                        for m, e in arow['per_model'].items():
                            if 'beat_every_random_seed' not in e:
                                continue
                            key = f'{lvl}|{space}|{det}|{rate}|{arm}|{m}'
                            nc = e['vs_no_cleaning']
                            won_nc = nc['passes_won'] * 2 > nc['n_passes_paired']
                            if e['beat_every_random_seed'] and won_nc:
                                won_both.append(key)
                            elif won_nc:
                                won_nc_only.append(key)
                            elif e['beat_every_random_seed']:
                                won_seeds_only.append(key)
    check('the headline counts only settings that ALSO beat not cleaning',
          c['settings_beating_every_random_seed_AND_no_cleaning'] == len(won_both),
          f"{c['settings_beating_every_random_seed_AND_no_cleaning']} on the "
          f"file, {len(won_both)} rebuilt from the per-setting entries")
    check('and a setting that beat the seeds but lost to no cleaning is in '
          'its own bucket, not the headline',
          c['settings_beating_every_random_seed_but_LOSING_to_no_cleaning']
          == len(won_seeds_only),
          f'{len(won_seeds_only)} such settings')
    check('the three buckets are disjoint and cover every judged setting',
          len(set(won_both) & set(won_nc_only)) == 0
          and len(set(won_both) & set(won_seeds_only)) == 0
          and len(set(won_nc_only) & set(won_seeds_only)) == 0,
          f'{len(won_both)} + {len(won_nc_only)} + {len(won_seeds_only)}')
    return r


# ── the rate-blind arms are fitted once, not three times ────────────────────

def test_rate_blind_arms_are_fitted_once():
    print("\nhow many times each cleanup was built, over two flag rates:")
    calls = {}
    real = n24.cleaned_training_set

    def counted(strategy, *a, **k):
        calls[strategy] = calls.get(strategy, 0) + 1
        return real(strategy, *a, **k)

    n24.cleaned_training_set = counted
    try:
        r = run_it(make_data(seed=11), 'toy_counts',
                   flag_rates=[0.05, 0.20], random_seeds=[0])
    finally:
        n24.cleaned_training_set = real

    # 3 passes, 1 space, 1 detector, 2 flag rates, 1 random-deletion seed.
    want = {'remove': 3 * 2 + 3 * 2,      # the detector's, plus the control's
            'knn_smooth': 3 * 2,          # reads the rate, so once per rate
            'label_smooth': 3, 'oof_smooth': 3,
            'downweight': 3, 'huber_weight': 3}
    for cleanup, n in sorted(want.items()):
        print(f"        {cleanup:14s} built {calls.get(cleanup, 0)} times over "
              f"3 passes and 2 flag rates, expected {n}")
    check('a rate-reading arm is built once per flag rate and a rate-blind one '
          'once in total',
          all(calls.get(c, 0) == n for c, n in want.items()),
          f'{calls}')

    # And the numbers written under the second rate are the SAME numbers, not a
    # second fit that happened to land nearby.
    by_rate = r['per_level']['0.5']['base_1']['neighbor_residual']
    same = all(
        by_rate['0.05'][arm]['rf'] == by_rate['0.2'][arm]['rf']
        for arm in ('keep_rewrite/label_smooth', 'keep_rewrite/oof_smooth',
                    'keep_reweight/downweight', 'keep_reweight/huber_weight'))
    differs = (by_rate['0.05']['delete/remove']['rf']
               != by_rate['0.2']['delete/remove']['rf'])
    check('the rate-blind rows repeat the same per-pass scores and the '
          'deletion rows do not', same and differs)
    check('the file says which flag rate each arm was fitted at',
          r['fitted_at_flag_rate_per_arm']['keep_reweight/downweight'] == 0.05
          and r['fitted_at_flag_rate_per_arm']['delete/remove'] == 'every rate',
          f"{r['fitted_at_flag_rate_per_arm']}")


# ── 4. the direction ────────────────────────────────────────────────────────

def test_direction():
    print("\nthe metric direction:")
    lo = run_it(make_data(seed=4, domain='molecular'), 'toy_rmse')
    hi = run_it(make_data(seed=4, domain='antibody'), 'toy_spearman')
    check('the two runs recorded opposite directions',
          lo['lower_is_better'] is True and hi['lower_is_better'] is False,
          f"{lo['metric']} lower_is_better={lo['lower_is_better']}, "
          f"{hi['metric']} lower_is_better={hi['lower_is_better']}")

    # ⛔ NOT "the same arm wins under both metrics". rmse and spearman measure
    # different things and an arm can improve the rank order while widening the
    # error, so that comparison would be a claim about the arms rather than a
    # check on the sign. What is checked is the sign itself, twice.

    # Once on the arithmetic alone: the same two score columns, read under each
    # direction, must give gains that are exact negatives of each other.
    base = {'p': 1.0, 'q': 2.0, 'r': 3.0}
    got = {'p': 0.5, 'q': 2.5, 'r': 2.0}
    low = n24._paired(base, got, lower_is_better=True)
    high = n24._paired(base, got, lower_is_better=False)
    check('a gain under one direction is the same number negated under the '
          'other',
          np.allclose(low['pass_gains'], [-g for g in high['pass_gains']])
          and low['passes_won'] == high['passes_lost'] == 2,
          f"lower-wins gains {low['pass_gains']}, higher-wins gains "
          f"{high['pass_gains']}")

    # And once against the raw per-pass scores the same file carries: the count
    # of passes the summary says an arm won must equal the count of passes
    # where its score really is on the better side of no cleaning, and "better"
    # is the opposite side in the two runs.
    def _recount(res, wants_lower):
        rows, bad = 0, []
        for lvl, by_space in res['per_level'].items():
            for space, by_det in by_space.items():
                for det, by_rate in by_det.items():
                    for rate, by_arm in by_rate.items():
                        nc = by_arm['no_cleaning']
                        summ = (res['summary']['per_level'][lvl]['per_space']
                                [space]['per_detector'][det]['per_flag_rate']
                                [rate]['per_arm'])
                        for arm, by_model in by_arm.items():
                            if arm == 'no_cleaning':
                                continue
                            for m, per_pass in by_model.items():
                                b = nc[m]
                                won = sum(
                                    1 for k, v in per_pass.items()
                                    if k in b and ((v < b[k]) if wants_lower
                                                   else (v > b[k])))
                                said = (summ[arm]['per_model'][m]
                                        ['vs_no_cleaning']['passes_won'])
                                rows += 1
                                if won != said:
                                    bad.append((arm, m, rate, won, said))
        return rows, bad

    n_lo, bad_lo = _recount(lo, wants_lower=True)
    n_hi, bad_hi = _recount(hi, wants_lower=False)
    print(f"        rmse: {n_lo} arm readings recounted from the raw per-pass "
          f"scores, {len(bad_lo)} disagreed")
    print(f"        spearman: {n_hi} arm readings recounted, "
          f"{len(bad_hi)} disagreed")
    check('the passes the summary says an arm won are the passes where its '
          'score is on the better side, under both directions',
          not bad_lo and not bad_hi and n_lo > 0 and n_hi > 0,
          f'{bad_lo + bad_hi}')


# ── 5. the rate-blind arms ──────────────────────────────────────────────────

def test_rate_blindness():
    print("\nwhich arms read the flag rate, checked against the cleanup itself:")
    rng = np.random.default_rng(3)
    n = 120
    X = rng.normal(size=(n, 5))
    y = X @ rng.normal(size=5) + rng.normal(0, 0.3, n)
    scores = np.abs(rng.normal(size=n))
    targets = {'oof': y + rng.normal(0, 0.2, n)}

    def _column(cleanup, rate):
        got = cleaned_training_set(cleanup, X, y, scores, rate, is_clf=False,
                                   groups=None, strength=None, targets=targets)
        if got is None:
            return None
        if got['keep'] is not None:
            # A deletion has no label column; the mask is what changes.
            return np.asarray(got['keep'], dtype=float)
        if got['sample_weight'] is not None:
            return np.asarray(got['sample_weight'], dtype=float)
        return np.asarray(got['y'], dtype=float)

    ok = True
    for cleanup in ('remove', 'knn_smooth', 'label_smooth', 'oof_smooth',
                    'downweight', 'huber_weight'):
        a, b = _column(cleanup, 0.05), _column(cleanup, 0.20)
        differs = not np.allclose(a, b)
        claimed = n24.reads_flag_rate(cleanup, is_clf=False)
        print(f"        {cleanup:14s} the cleaned training set differs "
              f"between flag rate 0.05 and 0.20: {differs}; the file says it "
              f"reads the rate: {claimed}")
        if differs != claimed:
            ok = False
    check('every arm is on the right side of reads_flag_rate', ok)


# ── 6. the exact-count rule ─────────────────────────────────────────────────

def test_exact_counts():
    print("\nthe random deletion removes exactly the number asked for:")
    bad = []
    for n in (33, 100, 251, 448):
        X = np.zeros((n, 2))
        yv = np.zeros(n)
        for k in (1, 2, 5, 22, 45):
            if k >= n:
                continue
            s, rate = n24.random_deletion_scores(n, k, seed=k * 7 + n)
            got = cleaned_training_set('remove', X, yv, s, rate, is_clf=False,
                                       groups=None, strength=None, targets=None)
            if got is None:
                bad.append((n, k, 'declined'))
                continue
            removed = int(n - int(np.asarray(got['keep']).sum()))
            if removed != k:
                bad.append((n, k, removed))
    check('every (training count, deleted count) pair deleted exactly that '
          'many molecules', not bad, f'mismatches: {bad}' if bad else
          'checked 33, 100, 251 and 448 training molecules')


# ── 7. what the condition left behind ───────────────────────────────────────

def test_injury_share():
    print("\nhow much of the damage sits on the molecules the injector widened:")
    both = {}
    for cond in ('outlier_p10', 'outlier_only_p10'):
        r = run_it(make_data(seed=8), f'toy_{cond}', noise_condition=cond,
                   flag_rates=[0.10], random_seeds=[0])
        row = r['injury_carried_by_the_flagged_molecules']['0.5']
        both[cond] = row['share_of_the_squared_error_on_the_widened_molecules']
        print(f"        {cond}: the "
              f"{row['widened_molecules_summed_over_the_passes']} widened "
              f"molecules of "
              f"{row['training_molecules_summed_over_the_passes']} training "
              f"molecules carry {both[cond]:.3f} of the squared label error")
        check(f'{cond}: the run recorded whether every other label also moved',
              r['condition_moves_every_other_label'] == (
                  cond == 'outlier_p10'))
    check('the widened molecules carry all of the error under '
          'outlier_only_p10 and only part of it under outlier_p10',
          both['outlier_only_p10'] > both['outlier_p10'] + 0.1
          and both['outlier_only_p10'] > 0.99,
          f"{both['outlier_p10']:.3f} against "
          f"{both['outlier_only_p10']:.3f}")


# ── a condition with no damaged subset is refused ───────────────────────────

def test_condition_guard():
    print("\na condition where every label moves is refused:")
    try:
        run_it(make_data(n=80, seed=2), 'toy_guard', noise_condition='gaussian')
        check('gaussian is refused because there is no flagged set to delete',
              False, 'the run returned instead of raising')
    except ValueError as exc:
        check('gaussian is refused because there is no flagged set to delete',
              'no damaged subset' in str(exc), str(exc)[:90])


def test_the_command_line():
    """The flags are checked on the real parser, in a real process.

    Building a second parser here and testing that would test the copy.
    """
    print("\nthe command line:")
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      'n24_delete_or_keep.py'), '--help'],
        capture_output=True, text=True)
    wanted = ('--dataset', '--noise-condition', '--levels', '--detectors',
              '--flag-rates', '--models', '--exp-folds', '--budgets', '--tag',
              '--random-seeds', '--draws')
    missing = [f for f in wanted if f not in out.stdout]
    check('the parser builds and lists every flag the runner needs',
          out.returncode == 0 and not missing,
          f'missing {missing}' if missing
          else f'{len(wanted)} flags present')
    check('--base-rep is gone', '--base-rep' not in out.stdout)


# ── each model its own spaces (owner's decision 2026-09-25) ────────────────

def recipe_per_model(recipes):
    """A pool stand-in that answers each downstream model with its own
    recipe, the way pool_recipe_per_downstream_model asks it."""
    from kirby.find_best_hybrid import Blueprint
    asked = []

    def lookup(*a, downstream_model=None, **k):
        asked.append(downstream_model)
        reps_, budget = recipes[downstream_model]
        bp = Blueprint(reps=list(reps_), allocation_method='fixed',
                       importance_method='lightgbm_gain', budget=budget,
                       internal_evaluator='lightgbm', metric='rmse',
                       lower_is_better=True, task_type='regression',
                       augmentation_strategy='none')
        meta = {'best_config': {'blueprint': bp},
                'objective': 'predictive',
                'objective_choice': 'smoke_fixed_recipe',
                'pool_subset': 'predictive_and_similarity', 'tier': 'general',
                'aug_condition': 'none', 'by_objective': {}}
        return None, meta
    lookup.asked = asked
    return lookup


def test_per_model_spaces():
    print("\neach model in its own spaces:")
    lookup = recipe_per_model({'rf': (['aaa', 'bbb'], 4),
                               'lightgbm': (['aaa'], 3)})
    r = run_it(make_data(seed=21), 'toy_per_model', recipe=lookup,
               models=['rf', 'lightgbm'],
               base_reps={'rf': ['aaa', 'bbb'], 'lightgbm': ['bbb', 'aaa']},
               flag_rates=[0.10], random_seeds=[0])
    check('the pool was asked once per downstream model',
          sorted(lookup.asked) == ['lightgbm', 'rf'], str(lookup.asked))
    check('both single-representation spaces and the hybrid ran',
          r['spaces_run'] == ['base_1', 'base_2', 'hybrid'],
          str(r['spaces_run']))
    check('each model has its own recipe on the file',
          sorted(r['hybrid_recipe_by_model']) == ['lightgbm', 'rf']
          and r['hybrid_recipe_by_model']['lightgbm']['hybrid']['budget'] == 3,
          str({m: v['hybrid']['budget']
               for m, v in r['hybrid_recipe_by_model'].items()}))
    lv = r['per_level']['0.5']
    ok = all(m in lv[sp]['neighbor_residual']['0.1']['no_cleaning']
             for sp in r['spaces_run'] for m in ('rf', 'lightgbm'))
    check('every model is scored in every space', ok)
    det = r['detection']['0.5']
    want = {f'{sp}/{m}' for sp in r['spaces_run'] for m in ('rf', 'lightgbm')}
    check('the detector ran once per distinct matrix, named by space and model',
          want <= set(det), str(sorted(det)))
    dl = r['molecules_deleted']['0.5']
    check('the deletion counts are kept per matrix',
          want <= set(dl), str(sorted(dl)))
    srow = (r['summary']['per_level']['0.5']['per_space']['base_1']
            ['per_detector']['neighbor_residual'])
    check('the summary carries one detection reading per matrix',
          set(srow['detection_by_matrix']) == {'base_1/rf', 'base_1/lightgbm'},
          str(sorted(srow['detection_by_matrix'])))
    check('no errors recorded', not r.get('_errors'),
          str(r.get('_errors'))[:300])


# ── the similarity sweep, the D-versus-D' rows, the resume (2026-09-25) ─────

# Two settings, not the 66: a smoke checks the wiring, not the science.
SWEEP_SETTINGS = ('base_1|none|manhattan', 'tanimoto_ecfp4|none|tanimoto')


def _toy_smiles(n):
    """Distinct parseable SMILES, one per molecule, for tanimoto_ecfp4."""
    out = []
    for i in range(n):
        out.append('C' * (1 + i % 9) + 'O' * (i % 3) + 'N' * ((i // 27) % 3)
                   + 'C(=O)' * ((i // 81) % 3))
    return out


def test_similarity_sweep():
    """Plan sections 1 and 2 on n24.

    Under a named list of settings the sweep writes one row per (setting, k)
    for the neighbour detector's arms, and for the neighbour cleanups only
    under cross_val; today's path keeps its own rows; a row already on disk
    is not written again; and every row carries the metric direction.
    """
    import json
    from experiments.noise import similarity_space as ss
    from experiments.noise.n23_uncertainty_cleaning import (
        TODAY_SIMILARITY, NO_NEIGHBOURS)
    print("\nthe similarity sweep and the D-versus-D' rows:")
    data = make_data(seed=3)
    data['smiles'] = _toy_smiles(len(data['labels']))
    prev = os.environ.get('KIRBY_SIMILARITY_SETTINGS')
    os.environ['KIRBY_SIMILARITY_SETTINGS'] = ','.join(SWEEP_SETTINGS)
    kw = dict(models=['rf'], base_reps={'rf': ['aaa', 'bbb']},
              detectors=['neighbor_residual', 'cross_val'], flag_rates=[0.10],
              random_seeds=[0], levels=[0.5], draws=[0])
    try:
        r = run_it(data, 'toy_sweep', **kw)
        path = os.path.join(OUT, 'n24_delete_or_keep', 'toy_sweep.rows.jsonl')
        rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    finally:
        if prev is None:
            os.environ.pop('KIRBY_SIMILARITY_SETTINGS', None)
        else:
            os.environ['KIRBY_SIMILARITY_SETTINGS'] = prev
    check('rf and svm are both in the run (ensure_dprime_models)',
          {'rf', 'svm'} <= set(r['models']), str(r['models']))
    check('rows were written to the rows file', len(rows) > 0, path)
    labels = {f'{s}|k={k}' for s in SWEEP_SETTINGS for k in ss.SWEPT_KS}
    nr = {x['similarity_setting'] for x in rows
          if x['method_setting'].startswith('neighbor_residual')}
    check('neighbor_residual has rows under every setting x k, and today\'s',
          labels | {TODAY_SIMILARITY} <= nr,
          f'missing {sorted((labels | {TODAY_SIMILARITY}) - nr)}')
    cv_sweep = {x['method'] for x in rows
                if x['method_setting'].startswith('cross_val')
                and x['similarity_setting'] in labels}
    check('under cross_val the sweep ran only the cleanups that read '
          'neighbours', cv_sweep == {'label_smooth', 'knn_smooth'},
          str(sorted(cv_sweep)))
    cv_none = {x['method'] for x in rows
               if x['method_setting'].startswith('cross_val')
               and x['similarity_setting'] == NO_NEIGHBOURS}
    check('cross_val with a cleanup that reads no neighbours is labelled '
          '"none"', 'remove' in cv_none and 'label_smooth' not in cv_none,
          str(sorted(cv_none)))
    check('every row carries the direction and a gain signed by it',
          all(x['lower_is_better'] is True for x in rows)
          and all((x['gain'] != x['gain']) or abs(
              x['gain'] - (x['score_D'] - x['score_Dprime'])) < 1e-12
                  for x in rows), 'rmse: lower wins')
    today = {(x['method'], x['method_setting'], x['fold']): x['score_Dprime']
             for x in rows if x['similarity_setting'] == TODAY_SIMILARITY
             and x['method'] == 'remove'}
    moved = [x for x in rows if x['method'] == 'remove'
             and x['similarity_setting'] == 'base_1|none|manhattan|k=5'
             and today.get((x['method'], x['method_setting'], x['fold']))
             != x['score_Dprime']]
    check('a sweep setting changes what remove deletes, so its score moves',
          bool(moved), f'{len(moved)} rows differ from today\'s')
    check('the unavailable spaces are named with a reason',
          'hybrid' in (r['similarity_sweep'].get('spaces_unavailable') or {}),
          str(sorted(r['similarity_sweep'].get('spaces_unavailable') or {})))
    # The resume rule, without a second run: a sink opened on the same file
    # holds every key, and adding a row that is already there writes nothing.
    from experiments.noise.n23_uncertainty_cleaning import RowSink
    from experiments._shared import get_domain_metric
    sink = RowSink(OUT, 'n24_delete_or_keep', 'toy_sweep', experiment='n24',
                   metric=get_domain_metric(data), data=data,
                   config={'n_splits': 3})
    x = rows[0]
    sink.add(**{k: x[k] for k in ('method', 'method_setting',
                                   'similarity_setting', 'model_input',
                                   'model', 'noise_condition', 'level', 'draw',
                                   'fold', 'score_D', 'score_Dprime',
                                   'score_clean')})
    check('a row already on disk is not written again',
          sink.flush() == 0 and len(sink.done) == len(rows),
          f'{len(sink.done)} keys on disk, {len(rows)} rows')


if __name__ == '__main__':
    print(f"toy dataset: {SMOKE_N} molecules (KIRBY_SMOKE_N)")
    test_the_control()
    test_rate_blind_arms_are_fitted_once()
    test_the_command_line()
    test_direction()
    test_rate_blindness()
    test_exact_counts()
    test_injury_share()
    test_condition_guard()
    test_per_model_spaces()
    test_similarity_sweep()
    print("\n" + "=" * 60)
    if FAIL:
        print(f"SMOKE FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("ALL GREEN")
