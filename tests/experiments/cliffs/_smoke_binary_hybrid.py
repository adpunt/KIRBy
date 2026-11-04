#!/usr/bin/env python3
"""Smoke: a hybrid assembled from 0/1 representations stays 0/1, and the guard
raises when it does not.

Fast. Loads one dataset's representations from the cache, assembles a hybrid the
way the search's own apply_feature_selection does, and checks the kind. No search
runs here: _run_binary_hybrid_search.py does that.

    python3 experiments/cliffs/_smoke_binary_hybrid.py --dataset herg_ki
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from kirby.hybrid import apply_feature_selection
from kirby.scaling import FeatureScaler
from experiments.cliffs.binary_hybrid import (
    EXPECTED_KIND, RESTRICTIONS, binary_hybrid_inputs, coefficients_for_matrix,
    representation_kinds, require_kind, restricted_pool,
)

FAILS = []


def check(name, condition, detail=''):
    mark = 'ok  ' if condition else 'FAIL'
    print(f"  [{mark}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def fake_selection(reps, names, per_rep=12, seed=0):
    """A feature_info of the shape create_hybrid produces, spanning every name."""
    rng = np.random.RandomState(seed)
    info = {}
    for n in names:
        width = np.asarray(reps[n]).shape[1]
        k = min(per_rep, width)
        info[n] = {'selected_indices': sorted(
            rng.choice(width, k, replace=False).tolist())}
    return info


def main(argv=None):
    import argparse

    from experiments._shared import load_data

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='herg_ki')
    args = ap.parse_args(argv)

    data = load_data(args.dataset, rep_preset='fast')
    reps = {k: np.asarray(v, dtype=np.float64) for k, v in data['reps'].items()}
    kinds = representation_kinds(reps)
    print(f"\n{args.dataset}: {len(reps)} representations, "
          f"{len(data['labels'])} molecules")

    for restriction in RESTRICTIONS:
        expected = EXPECTED_KIND[restriction]
        print(f"\n{restriction} (the assembled hybrid must be {expected})")
        inputs, exp = binary_hybrid_inputs(reps, restriction,
                                           where=args.dataset)
        check(f"{restriction}: restriction keeps something",
              len(inputs) > 0, f"{sorted(inputs)}")
        check(f"{restriction}: expected kind is {expected}", exp == expected)

        info = fake_selection(reps, list(inputs))
        X = apply_feature_selection(inputs, info)
        check(f"{restriction}: assembled width is the sum of the selections",
              X.shape[1] == sum(len(v['selected_indices'])
                                for v in info.values()),
              f"{X.shape[1]} features")

        read_kind, coeffs = coefficients_for_matrix(X)
        check(f"{restriction}: kind read from the assembled matrix is "
              f"{expected}", read_kind == expected, f"read {read_kind}")
        try:
            require_kind(X, expected, where=restriction)
            check(f"{restriction}: the guard passes the assembled hybrid", True)
        except ValueError as e:
            check(f"{restriction}: the guard passes the assembled hybrid",
                  False, str(e)[:120])
        print(f"         coefficients: {coeffs}")

    print("\nthe failure 6.1c prevents")
    bits = restricted_pool(reps, 'bits')
    inputs = {n: reps[n] for n in bits}
    info = fake_selection(reps, bits)
    X = apply_feature_selection(inputs, info)
    Xz = FeatureScaler().fit_transform(
        np.hstack([X, reps[[k for k in reps if kinds[k] == 'continuous'][0]][:, :8]]))
    try:
        require_kind(Xz, 'binary', where='a hybrid that mixed in a continuous '
                                         'representation')
        check("the guard raises on a hybrid that is not 0/1", False,
              "it passed, which it must not")
    except ValueError as e:
        check("the guard raises on a hybrid that is not 0/1", True,
              str(e)[:90])

    # The reason the guard exists, from require_binary's own docstring: scipy
    # thresholds at nonzero instead of raising, so every z-scored row becomes
    # all-True and every pair comes back identical.
    from scipy.spatial.distance import pdist
    cont = [k for k in reps if kinds[k] == 'continuous'][0]
    Z = FeatureScaler().fit_transform(reps[cont][:60, :64])
    d = pdist(Z, metric='jaccard')
    check("scipy's jaccard on z-scored features returns all zeros",
          bool(np.all(d == 0.0)),
          f"{cont}, 60 molecules, {len(d)} pairs, largest {float(d.max()):.3g}")

    # And FeatureScaler leaves 0/1 features alone, which is what makes the
    # binary hybrid possible at all.
    sc = FeatureScaler().fit(X)
    check("FeatureScaler leaves every 0/1 feature untouched",
          bool(np.allclose(sc.transform(X), X)),
          f"{int(np.asarray(sc.binary_mask_).sum())} of {X.shape[1]} features "
          f"marked 0/1")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
