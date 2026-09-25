#!/usr/bin/env python3
"""Smoke for n34: scaffold-group offsets -- group mean or random intercept.

What has to be true before any number n34 writes can be read:

  1. KNOWN OFFSETS ARE RECOVERED. On toy labels built as a group offset plus
     noise, the random intercept's per-group offsets correlate with the true
     ones on the groups large enough to carry evidence.
  2. A GROUP OF ONE IS SHRUNK TOWARD ZERO. Its fitted offset is smaller in size
     than its mean residual, which is what group_center subtracts in full.
  3. WITH NO GROUP EFFECT, tau^2 falls toward its floor and a group of one keeps
     almost nothing.
  4. group_center_3plus LEAVES SMALL GROUPS ALONE.
  4b. OFFSET RECOVERY IS READ ON GROUPS OF 3+ ONLY. Small groups built to
     anti-correlate cannot move corr_groups_of_3_plus; they are counted as
     excluded and appear only in corr_all_groups_contaminated.
  5. THE RECOVERY SHARE IS SIGNED BY THE METRIC AND REFUSED UNDER THE 2% FLOOR.
  6. A NON-GROUPED CONDITION, A CLASSIFICATION LABEL AND A DATASET WITHOUT
     SCAFFOLD GROUPS ARE SKIPPED WITH A REASON.
  7. A FULL RUN, single representations only and with a hybrid recipe, writes every arm for every
     model, a tail record per score, the `none` pass once, and offset recovery
     under grouped_shifted only. Under spearman the direction flips.
  8. The parser lists every flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# 2026-09-25: today's similarity (model_input) unless a check sets its own
# list, and the noise cache in a temporary directory, never the real one.
os.environ.setdefault('KIRBY_SIMILARITY_SETTINGS', 'model_input')
import tempfile as _tf
os.environ['KIRBY_CACHE_DIR'] = _tf.mkdtemp(prefix='n34_smoke_cache_')

os.environ.setdefault('KIRBY_LGBM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import tempfile

import numpy as np

import experiments.noise.n34_group_random_effects as n34
from kirby.find_best_hybrid import Blueprint

OUT = tempfile.mkdtemp(prefix='n34_smoke_')
FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


def toy_groups(n, seed=0):
    """Mixed sizes: many groups of one, some of 2-3, a few of 10+."""
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(rng.choice([1, 1, 1, 2, 3, 12])))
    g = np.repeat(np.arange(len(sizes)), sizes)[:n]
    return g[rng.permutation(n)]


def make_data(n=180, n_feat=4, seed=0, domain='molecular'):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = X @ rng.normal(size=n_feat) + 5.0 + rng.normal(0, 0.3, n)
    X2 = X[:, ::-1] + rng.normal(0, 0.5, size=X.shape)
    return {'labels': y, 'reps': {'toy': X, 'second': X2},
            'groups': np.arange(n), 'task_type': 'regression',
            'domain': domain, 'smiles': ['C'] * n}


def fake_recipe(rep_names, budget=4):
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
    return None, None


def run_it(data, name, recipe=no_recipe, groups=None, **kw):
    real_r = n34.get_best_search_for_downstream
    real_g = n34.noise_groups_for
    n34.get_best_search_for_downstream = recipe
    g = toy_groups(len(data['labels'])) if groups is None else groups
    n34.noise_groups_for = lambda d: (g, int(len(np.unique(g))), 0.1)
    try:
        kw.setdefault('base_reps', {
            m: ['toy', 'second']
            for m in (kw.get('models') or n34.DOWNSTREAM_MODELS)})
        return n34.run(data, {'n_splits': 3}, OUT, name, **kw)
    finally:
        n34.get_best_search_for_downstream = real_r
        n34.noise_groups_for = real_g


def test_offsets():
    print("the offset estimators, no model fitted:")
    rng = np.random.default_rng(3)
    n = 900
    g = toy_groups(n, seed=3)
    uniq, inv, counts = n34._group_index(g)
    b_true = rng.normal(0, 1.0, uniq.size)
    y = b_true[inv] + rng.normal(0, 1.0, n)
    # The model part is zero here, so the out-of-fold prediction is the mean.
    oof = lambda X, yy: np.full(len(yy), float(np.mean(yy)))
    off, info = n34.random_intercept_offsets(None, y, g, oof)
    per_group = np.bincount(inv, weights=off) / counts
    big = counts >= 3
    corr = float(np.corrcoef(per_group[big], b_true[big])[0, 1])
    check('1. known offsets recovered on groups of 3+', corr > 0.8,
          f'corr {corr:.3f} over {int(big.sum())} groups; tau2 '
          f'{info["tau2_between_group"]:.3f} (true 1.0), sigma2 '
          f'{info["sigma2_within_group"]:.3f} (true 1.0)')
    r = y - np.mean(y)
    gc = n34.group_center_offsets(r, g)
    one = counts[inv] == 1
    keep = info['share_a_group_of_one_keeps']
    ratio = float(np.mean(np.abs(off[one])) / np.mean(np.abs(gc[one])))
    check('2. a group of one keeps less than its full mean residual',
          0.2 < keep < 0.8 and ratio < 0.8,
          f'keeps {keep:.3f} by formula; mean |offset| is {ratio:.3f} of '
          f"group_center's on {int(one.sum())} molecules in groups of one")

    y_flat = rng.normal(0, 1.0, n)
    _, info0 = n34.random_intercept_offsets(None, y_flat, g, oof)
    check('3. no group effect: a group of one keeps almost nothing',
          info0['share_a_group_of_one_keeps'] < 0.15,
          f'keeps {info0["share_a_group_of_one_keeps"]:.3f}, tau2 '
          f'{info0["tau2_between_group"]:.4f}')

    c3 = n34.group_center_offsets(r, g, min_size=3)
    small = counts[inv] < 3
    check('4. group_center_3plus leaves groups under 3 at zero offset',
          np.all(c3[small] == 0) and np.allclose(c3[~small], gc[~small]))


def test_recovery_masks_small_groups():
    print("\noffset recovery reads groups of 3+ only:")
    # 6 groups of 4 molecules where recovered tracks injected, then 10 groups
    # of 1 and 5 groups of 2 where recovered is the negative of injected.
    # If any small group leaked in, the 3+ correlation would drop below 1.
    rng = np.random.default_rng(3)
    g, y_clean, y_noisy, off = [], [], [], []
    gid = 0
    for size, n_groups, sign in ((4, 6, 1.0), (1, 10, -1.0), (2, 5, -1.0)):
        for _ in range(n_groups):
            b = rng.normal(0, 1.0)
            for _ in range(size):
                g.append(gid)
                yc = rng.normal()
                y_clean.append(yc)
                y_noisy.append(yc + b)
                off.append(sign * b)
            gid += 1
    g = np.array(g)
    out = n34.offset_recovery(np.array(off), np.array(y_noisy),
                              np.array(y_clean), g)
    keys = {'corr_groups_of_3_plus', 'n_groups_of_3_plus',
            'n_groups_excluded_under_3', 'min_group_size',
            'corr_all_groups_contaminated', 'n_groups_all'}
    check('4b. offset_recovery writes the new keys and not the old all-groups key',
          set(out) == keys, str(sorted(out)))
    check('4b. the 3+ correlation uses the 6 groups of 4 and excludes 15',
          out['n_groups_of_3_plus'] == 6 and out['n_groups_excluded_under_3'] == 15
          and out['n_groups_all'] == 21
          and abs(out['corr_groups_of_3_plus'] - 1.0) < 1e-9,
          f"3+ corr {out['corr_groups_of_3_plus']}, used "
          f"{out['n_groups_of_3_plus']}, excluded {out['n_groups_excluded_under_3']}")
    check('4b. the all-groups number is pulled down by the small groups',
          out['corr_all_groups_contaminated'] < 0.5,
          f"all-groups corr {out['corr_all_groups_contaminated']:.3f}")


def test_share():
    print("\nthe recovery share:")
    s, _ = n34.recovery_share(0.5, 0.6, 0.55, True)
    s2, _ = n34.recovery_share(0.8, 0.7, 0.75, False)
    s3, why = n34.recovery_share(0.5, 0.501, 0.5, True)
    check('5. signed by the metric, refused under the floor',
          abs(s - 0.5) < 1e-9 and abs(s2 - 0.5) < 1e-9 and s3 is None and why,
          f'{s:.2f}, {s2:.2f}, refused: {why}')


def test_skips():
    print("\nthe skips:")
    d = make_data()
    r1 = run_it(d, 'toy', noise_condition='gaussian')
    dc = make_data()
    dc['task_type'] = 'classification'
    r2 = run_it(dc, 'toy')
    real = n34.noise_groups_for
    n34.noise_groups_for = lambda d: (None, 0, 0.0)
    try:
        r3 = n34.run(make_data(), {'n_splits': 3}, OUT, 'toy',
                     base_reps={'rf': ['toy', 'second']})
    finally:
        n34.noise_groups_for = real
    check('6. gaussian, a class label and no groups each skip with a reason',
          r1.get('skipped') == 'condition_not_grouped'
          and r2.get('skipped') == 'not_regression'
          and r3.get('skipped') == 'no_scaffold_groups')


def test_full(recipe, label, domain='molecular'):
    print(f"\na full run, {label}:")
    d = make_data(domain=domain)
    models = ['rf', 'lightgbm', 'svm']
    res = run_it(d, 'toy', recipe=recipe, models=models,
                 levels=[0.0, 0.5], draws=[0, 1])
    pc = res.get('per_condition', {})
    errs = res.get('_errors', {})
    check(f'7. [{label}] no errors', not errs, str(list(errs.items())[:3]))
    check(f'7. [{label}] level 0 ran once, under none',
          set(pc) == {'none', 'grouped_shifted', 'grouped_wider'}
          and list(pc['none']) == ['0']
          and '0' not in pc['grouped_shifted'],
          f'{ {c: list(v) for c, v in pc.items()} }')
    spaces = res['spaces_run']
    check(f'7. [{label}] base_1 and base_2 ran first, representations named',
          spaces[:2] == ['base_1', 'base_2']
          and res['base_reps_by_model'] == {m: ['toy', 'second']
                                            for m in models}
          and 'base_rep' not in res,
          f"{spaces}, {res.get('base_reps_by_model')}")
    ok = True
    for cond, lv in (('grouped_shifted', '0.5'), ('grouped_wider', '0.5')):
        for sp in spaces:
            for arm in n34.ARMS:
                for m in models:
                    got = pc.get(cond, {}).get(lv, {}).get(sp, {}).get(arm, {}).get(m, {})
                    if len(got) != 6:
                        ok = False
    check(f'7. [{label}] every arm x model x space has 6 passes '
          f'(2 draws x 3 folds)', ok, f'spaces {spaces}')
    tails = res['tail_per_condition']['grouped_shifted']['0.5']['base_1']['none']['rf']
    check(f'7. [{label}] a tail record per score',
          len(tails) == 6 and all('tail_rmse_top' in t for t in tails.values()))
    gp = res['groups_per_pass']
    has_rec = all('offset_recovery' in sp
                  for r in gp['grouped_shifted']['0.5'] for sp in r['per_space'].values())
    no_rec = all('offset_recovery' not in sp
                 for c in ('grouped_wider', 'none') for lv in gp[c].values()
                 for r in lv for sp in r['per_space'].values())
    check(f'7. [{label}] offset recovery under grouped_shifted only',
          has_rec and no_rec)
    rec_ok = True
    for r in gp['grouped_shifted']['0.5']:
        for sp in r['per_space'].values():
            for arm_rec in sp['offset_recovery'].values():
                if ('corr_all_groups' in arm_rec
                        or 'corr_all_groups_contaminated' not in arm_rec
                        or 'corr_groups_of_3_plus' not in arm_rec
                        or arm_rec['n_groups_of_3_plus']
                        + arm_rec['n_groups_excluded_under_3']
                        != arm_rec['n_groups_all']):
                    rec_ok = False
    check(f'7. [{label}] recovery writes the 3+ correlation with used and '
          f'excluded group counts', rec_ok)
    sh_rec = gp['grouped_shifted']['0.5'][0]['per_space']['base_1']
    print(f"      grouped_shifted fold 0 base: random intercept "
          f"{sh_rec['random_intercept']['em_rounds']} rounds, group of one keeps "
          f"{sh_rec['random_intercept']['share_a_group_of_one_keeps']:.3f}; "
          f"recovery corr (groups of 3+, "
          f"{sh_rec['offset_recovery']['group_center']['n_groups_of_3_plus']} used, "
          f"{sh_rec['offset_recovery']['group_center']['n_groups_excluded_under_3']} "
          f"excluded) group_center "
          f"{sh_rec['offset_recovery']['group_center']['corr_groups_of_3_plus']}, "
          f"random_intercept "
          f"{sh_rec['offset_recovery']['random_intercept']['corr_groups_of_3_plus']}")
    if recipe is not no_recipe:
        check(f'7. [{label}] a hybrid space ran', 'hybrid' in spaces, str(spaces))
    return res


def test_direction():
    print("\nthe direction under spearman:")
    d = make_data(domain='antibody')
    res = run_it(d, 'toy', models=['rf'], levels=[0.5], draws=[0],
                 noise_condition='grouped_shifted')
    s = res['summary']
    e = s['per_condition']['grouped_shifted']['0.5']['base_1']['rf']['per_arm']['clean']['vs_none']
    raw = res['per_condition']['grouped_shifted']['0.5']['base_1']
    gains = [raw['clean']['rf'][k] - raw['none']['rf'][k] for k in raw['none']['rf']]
    check('7. under spearman a higher score counts as a gain',
          res['lower_is_better'] is False
          and np.allclose(sorted(e['pass_gains']), sorted(gains)),
          f"metric {res['metric']}")


def test_parser():
    print("\nthe parser:")
    import subprocess
    out = subprocess.run([sys.executable, n34.__file__, '--help'],
                         capture_output=True, text=True).stdout
    flags = ['--noise-condition', '--levels', '--draws', '--em-max-rounds',
             '--exp-folds', '--tag']
    missing = [f for f in flags if f not in out]
    check('8. the parser lists every flag', not missing, f'missing {missing}')
    check('8b. --base-rep is gone', '--base-rep' not in out)



def _rows_of(res):
    """The D-versus-D' rows a run wrote (2026-09-25)."""
    import json
    p = (res.get('d_vs_dprime') or {}).get('rows_file')
    if not p or not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _with_settings(names, fn):
    """Run fn() under KIRBY_SIMILARITY_SETTINGS=names, then restore it."""
    old = os.environ.get('KIRBY_SIMILARITY_SETTINGS')
    os.environ['KIRBY_SIMILARITY_SETTINGS'] = names
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop('KIRBY_SIMILARITY_SETTINGS', None)
        else:
            os.environ['KIRBY_SIMILARITY_SETTINGS'] = old


def test_rows_and_resume():
    """2026-09-25: the D-versus-D' rows (or, for a measurement-only experiment,
    the per-pass checkpoint), a rerun that resumes every pass, and -- where
    the experiment finds neighbours -- the similarity sweep."""
    print("\nrows, resume and the similarity sweep:")
    from experiments.noise.d_vs_dprime import ROW_FIELDS
    kw = dict(models=['rf', 'svm'], levels=[0.5], draws=[0], noise_condition='grouped_shifted', em_max_rounds=2)
    res = run_it(make_data(), 'toy_rows', **dict(kw))
    rows = _rows_of(res)
    check('R1. D-vs-D\' rows are written, each with exactly the d_vs_dprime '
          'columns, D and D\' both scored', bool(rows) and all(
              set(r) == set(ROW_FIELDS) and r['score_D'] is not None
              for r in rows) and {'rf', 'svm'} <= {r['model'] for r in rows},
          f'{len(rows)} rows, methods {sorted({r["method"] for r in rows})[:4]}')
    again = run_it(make_data(), 'toy_rows', **dict(kw))
    got = again['d_vs_dprime']
    check('R2. a rerun resumes every pass from the checkpoint and returns the '
          'same scores', got['passes_resumed_from_checkpoint'] > 0
          and got['rows_written_this_run'] == 0
          and again['per_condition'] == res['per_condition'],
          f"{got['passes_resumed_from_checkpoint']} passes resumed")


if __name__ == '__main__':
    test_offsets()
    test_recovery_masks_small_groups()
    test_share()
    test_skips()
    test_full(no_recipe, 'single representations only')
    test_full(fake_recipe(['toy', 'second']), 'with a hybrid recipe')
    test_direction()
    test_rows_and_resume()
    test_parser()
    print("\n" + ("ALL GREEN" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
