#!/usr/bin/env python3
"""Local verification for the NOISY-LABEL hybrid search precompute.

Run:  PYTHONPATH=src:tests python tests/experiments/noise/_smoke_noisy_search.py

WHY THIS EXISTS. Noise experiments used to build their hybrid from a blueprint
searched on the CLEAN labels, while the base representations got no such look — so
"the hybrid copes better with noise" could not be distinguished from "the hybrid was
built knowing the answers". `find_best_hybrid(label_noise=...)` fixes that: the search
TRAINS on a deterministic noisy copy of the labels and SCORES on the clean ones.

The failure mode this guards against is SILENT, and it is the whole reason for the
file. Label noise is injected on TRAINING rows only, so the global label array is
byte-identical to the clean run — which means `labels_hash` (in the SQLite result
identity) CANNOT tell a noisy run from a clean one. Get the cache namespacing wrong and
a "noisy" search happily reads the CLEAN rows back out of the store, returns the clean
blueprint, and nothing crashes. Test 4 is the one that catches that.

Everything here is tiny synthetic data — no dataset, no server, seconds to run.
"""
import os
import sys
import shutil
import sqlite3
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

_CACHE = tempfile.mkdtemp(prefix='kirby_noisy_smoke_')
os.environ['KIRBY_CACHE_DIR'] = _CACHE

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from kirby.find_best_hybrid import find_best_hybrid            # noqa: E402
from kirby.noise_spec import (                                  # noqa: E402
    NoiseSpec, make_noisy_labels, calibrate, noisy_labels_hash, default_spec,
)

DS = 'noisysmoke'
FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 8))
    B = rng.normal(size=(n, 6))
    y = 2.0 * A[:, 0] - 1.5 * A[:, 1] + 0.5 * B[:, 0] + rng.normal(0, 0.3, n)
    groups = np.repeat(np.arange(n // 20), 20)
    return {'repA': A, 'repB': B}, y, groups


def search(reps, y, groups, spec=None, tier='general'):
    return find_best_hybrid(
        base_reps=reps, labels=y, groups=groups,
        budgets=[6], models=['rf'], metric='r2', task_type='regression',
        dataset_name=DS, tier=tier,
        allocation_methods=['rank_linear'], importance_methods=['lightgbm_gain'],
        internal_evaluators=['rf'], label_noise=spec,
    )


def main():
    reps, y, groups = make_data()
    spec = default_spec(is_clf=False, target_effective_noise=0.30)

    print("\n1. REPRODUCIBILITY (the cache is void without it)")
    a = make_noisy_labels(y, spec, is_clf=False)
    b = make_noisy_labels(y, spec, is_clf=False)
    check("same spec -> byte-identical labels", np.array_equal(a, b),
          f"hash {noisy_labels_hash(a)}")
    c = make_noisy_labels(y, NoiseSpec('gaussian', 0.30, 1), is_clf=False)
    check("different noise_seed -> different draw", not np.array_equal(a, c))
    # The landmine: NoiseInject's RNG advances on every inject(), so a REUSED
    # injector yields different noise. kirby.noise_spec builds a fresh one per call.
    from noiseInject import NoiseInjectorRegression
    inj = NoiseInjectorRegression.from_condition('gaussian', random_state=0)
    check("reused raw injector DOES drift (why we build a fresh one)",
          not np.array_equal(inj.inject(y, 0.5), inj.inject(y, 0.5)))

    print("\n2. CALIBRATION (levels comparable across datasets)")
    # The level is the amount DELIVERED, measured as a root-mean-square in units
    # of the label spread. It used to be mean |dy| / SD -- the FIRST moment --
    # found by binary search; at identical root-mean-square noise mean|e|/rms is
    # 0.797 for a Gaussian but 0.642 for Student-t at nu = 3, so that definition
    # handed the heavy-tailed conditions up to 24% more actual noise at one
    # nominal level. Second moment, closed form, no search.
    sig = calibrate(y, spec, is_clf=False)
    eff = float(np.sqrt(np.mean((a - y) ** 2)) / np.std(y))
    check("delivered noise lands on target 0.30", abs(eff - 0.30) < 0.03,
          f"measured {eff:.4f} (magnitude={sig:.3f})")
    # And it lands there for EVERY condition, which is the whole point.
    off = []
    for cond in ('student_t_nu3', 'laplace', 'outlier_p10'):
        d = make_noisy_labels(y, NoiseSpec(cond, 0.30, 0), is_clf=False)
        e = float(np.sqrt(np.mean((d - y) ** 2)) / np.std(y))
        if abs(e - 0.30) > 0.05:
            off.append(f"{cond} {e:.4f}")
    check("and every condition delivers the same amount at one level",
          not off, '; '.join(off) or "the six it replaced spanned 0.49x to 2.00x")

    print("\n3. CLEAN RUN IS UNCHANGED (no regression for existing work)")
    clean = search(reps, y, groups)
    check("clean search produces a blueprint", clean.best_by_model.get('rf') is not None,
          f"rf r2={clean.best_by_model['rf']['score_mean']:.4f}")

    print("\n4. THE SILENT-FAILURE TEST — noisy must NOT serve clean numbers")
    noisy = search(reps, y, groups, spec=spec)
    same_base = [r for r in reps
                 if abs(clean.baselines[r]['rf'] - noisy.baselines[r]['rf']) < 1e-12]
    check("baselines DIFFER between clean and noisy runs", not same_base,
          "identical baselines would mean the clean cache was served"
          if same_base else
          ', '.join(f"{r}: {clean.baselines[r]['rf']:.4f}->{noisy.baselines[r]['rf']:.4f}"
                    for r in reps))

    print("\n5. BASELINES ARE NOISED TOO (or the bias just flips to the base reps)")
    worse = sum(noisy.baselines[r]['rf'] < clean.baselines[r]['rf'] for r in reps)
    check("noise degrades the base representations", worse >= 1,
          f"{worse}/{len(reps)} base reps got worse")

    print("\n6. CACHE SEPARATION (SQLite identity)")
    con = sqlite3.connect(os.path.join(_CACHE, 'result_db', f'{DS}.db'))
    # A store written under an older column list answers this SELECT perfectly
    # well, so the tier check would pass on a file the next search refuses to
    # open. Check the columns first — same guard as check_precompute_status.py.
    from kirby.result_cache import ALL_COLS as _ALL_COLS
    _have = {r[1] for r in con.execute('PRAGMA table_info(results)')}
    _missing = [c for c in _ALL_COLS if c not in _have]
    check("the result store has the current column list", not _missing,
          f"missing {_missing} — delete the file and re-run" if _missing else "ok")
    tiers = sorted(r[0] for r in con.execute("SELECT DISTINCT tier FROM results"))
    con.close()
    check("clean and noisy occupy distinct tiers", len(tiers) == 2, str(tiers))
    check("noisy tier is derived from the spec",
          spec.tier_name('general') in tiers, spec.tier_name('general'))

    print("\n7. SIDE-CACHE SEPARATION (baselines/ has no labels hash in its path)")
    bl = os.listdir(os.path.join(_CACHE, 'baselines'))
    check("noisy baseline files are namespaced by the noise id",
          sum(spec.spec_id in f for f in bl) == len(reps),
          f"{sum(spec.spec_id in f for f in bl)}/{len(reps)} namespaced")
    check("clean baseline filenames are UNCHANGED (old caches still hit)",
          sum(spec.spec_id not in f for f in bl) == len(reps))

    print("\n8. MONOTONE DEGRADATION (the plumbing actually bites)")
    scores = []
    for lvl in (0.1, 0.6, 1.0):
        s = search(reps, y, groups, spec=NoiseSpec('gaussian', lvl, 0))
        scores.append(s.baselines['repA']['rf'])
    check("more noise -> worse baseline, monotonically",
          scores[0] > scores[1] > scores[2],
          ' > '.join(f'{s:.4f}' for s in scores))

    print("\n9. REP FACTORIES ARE REFUSED (they would fine-tune on clean labels)")
    try:
        find_best_hybrid(
            base_reps=reps, labels=y, groups=groups, budgets=[6], models=['rf'],
            metric='r2', task_type='regression', n_folds=3, verbose=False,
            cache=False, rep_factories={'x': lambda *a: None}, label_noise=spec)
        check("label_noise + rep_factories raises", False, "it did NOT raise")
    except ValueError as e:
        check("label_noise + rep_factories raises", 'rep_factories' in str(e))

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
    else:
        print("ALL CHECKS PASSED")
    print("=" * 62)
    shutil.rmtree(_CACHE, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
