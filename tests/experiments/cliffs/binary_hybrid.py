#!/usr/bin/env python3
"""The binary hybrid, and the coefficients that become available on it.

HANDOFF_AC_STUDY.md 6.1b and 6.1c. Four overlap-style coefficients -- tanimoto,
dice, tversky and soergel -- are defined only on features that are 0 or 1, or on
counts. A searched hybrid mixes binary fingerprints with embeddings and
descriptors, so it comes out continuous and those four cannot run on it. That is
why no hybrid has ever defined an activity cliff.

The fix is not a new search. It is the same search handed a restricted list of
representations: give it only the dataset's 0/1 representations and the hybrid it
returns is 0/1, because building a hybrid is column selection and concatenation
(kirby.hybrid._build_from_selection) and kirby.scaling.FeatureScaler leaves a 0/1
feature alone.

Two lists per dataset, because both are defined and neither is degenerate:

  bits        only the representations whose features are 0 or 1. The assembled
              hybrid is 0/1, and the binary coefficient list applies, tversky
              included.
  bits+counts the 0/1 representations and the count ones. The assembled hybrid
              is counts, and the count coefficient list applies, soergel
              included. Soergel is deliberately absent from the binary list:
              on 0/1 features it equals tanimoto exactly, proven in simeval's
              selftest.

The order in 6.1c matters and is enforced here: the kind is read from the
ASSEMBLED matrix, never assumed from the inputs, and a hybrid built from binary
inputs that does not come back binary raises instead of being worked around.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from kirby.similarity.preprocessing import feature_kind, require_binary

# The three lists, keyed on the kind of the features, read from one place so two
# experiments cannot disagree about which coefficients a matrix admits.
from experiments.framework.f18_distance_metric_sweep import coefficients_for

#: The lists a dataset's binary representations may be collected under.
RESTRICTIONS = ('bits', 'bits+counts')

#: The kind the assembled hybrid must come back as, per restriction.
EXPECTED_KIND = {'bits': 'binary', 'bits+counts': 'count'}


def representation_kinds(reps):
    """{representation name: 'binary' | 'count' | 'continuous'}.

    Decided from the values, never from the name: ecfp4 is binary, ecfp4_count
    is counts, and a z-scored stack of both is continuous.
    """
    return {name: feature_kind(X) for name, X in reps.items()}


def restricted_pool(reps, restriction='bits'):
    """The representation names this restriction keeps, in the pool's own order.

    Raises on an unknown restriction rather than silently returning everything.
    """
    if restriction not in RESTRICTIONS:
        raise ValueError(
            f"unknown restriction {restriction!r}; the lists are {RESTRICTIONS}. "
            f"'bits' keeps the 0/1 representations, 'bits+counts' also keeps the "
            f"count ones.")
    keep = {'bits': ('binary',), 'bits+counts': ('binary', 'count')}[restriction]
    kinds = representation_kinds(reps)
    return [name for name in reps if kinds[name] in keep]


def require_count(X, where='this matrix'):
    """Raise unless every entry is a non-negative whole number. Never coerce.

    The count coefficients divide by a sum of entries, so a negative entry makes
    the denominator smaller than the numerator and the distance comes back
    negative. Those pairs then sort to the FRONT of a nearest-neighbour list,
    which is the same silent failure require_binary exists to prevent.
    """
    A = np.asarray(X)
    finite = A[np.isfinite(A)]
    bad = finite[(finite < 0) | (finite != np.floor(finite))]
    if bad.size:
        raise ValueError(
            f"{where} needs counts, meaning every entry a whole number at or "
            f"above zero, and {bad.size} entries are not (for example "
            f"{bad.flat[0]!r}).")
    return A


def require_kind(X, expected, where='this matrix'):
    """Raise unless the assembled matrix is the kind the restriction promised.

    HANDOFF_AC_STUDY.md 6.1c: a hybrid built from binary inputs comes back
    binary, and if it does not that is a fault to raise and not to work around.
    The exact check is require_binary / require_count, not feature_kind, because
    feature_kind samples 2000 rows to bound its cost and a matrix that is 0/1 on
    those rows and not on the rest is exactly the case this guard is for.
    """
    if expected == 'binary':
        require_binary(X, coefficient=f'the binary hybrid ({where})')
    elif expected == 'count':
        require_count(X, where=f'the count hybrid ({where})')
    else:
        raise ValueError(
            f"require_kind checks 'binary' or 'count' and got {expected!r}; a "
            f"continuous matrix needs no guard because every coefficient in the "
            f"continuous list is defined on it.")
    return X


def coefficients_for_matrix(X, where='this matrix'):
    """(kind, [coefficient names]) for an assembled matrix.

    The kind is read from the matrix and returned alongside the names, so a
    stored result says which of the three lists its number came from.
    """
    kind = feature_kind(X)
    return kind, sorted(coefficients_for(kind))


def binary_hybrid_inputs(reps, restriction='bits', where='this dataset'):
    """The sub-dictionary of `reps` to hand the hybrid search, and its kind.

    Returns (inputs, expected_kind). The search is unchanged: it is the same
    search asked for a restricted list of representations. Raises when the
    restriction keeps nothing, because a hybrid of no representations is not a
    narrower answer, it is a missing one.
    """
    names = restricted_pool(reps, restriction)
    if not names:
        kinds = representation_kinds(reps)
        raise ValueError(
            f"the {restriction!r} list keeps none of {where}'s "
            f"{len(reps)} representations. Their kinds are "
            f"{ {n: kinds[n] for n in reps} }.")
    inputs = {name: reps[name] for name in names}
    return inputs, EXPECTED_KIND[restriction]


def main(argv=None):
    """Print one dataset's representation kinds and its two restricted lists.

    This is what fills in the per-dataset binary representation list that
    HANDOFF_AC_STUDY.md 6.1b says the AC experiments read from
    scratchpad/precompute_cuts/DECISIONS.md. It loads the representations and
    reads their values; it runs no search and fits no model.
    """
    import argparse

    from experiments._shared import load_data

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--reps', default=None,
                    help='comma-separated list; default is the dataset pool')
    args = ap.parse_args(argv)

    data = load_data(args.dataset, rep_preset=args.reps or 'fast')
    reps = data['reps']
    kinds = representation_kinds(reps)

    print(f"\n{args.dataset}: {len(reps)} representations, "
          f"{len(data['labels'])} molecules")
    print(f"{'representation':>20} {'kind':>11} {'features':>9}  "
          f"{'lowest value':>13}  {'entries below zero':>18}")
    for name, X in reps.items():
        A = np.asarray(X, dtype=np.float64)
        finite = A[np.isfinite(A)]
        low = float(finite.min()) if finite.size else float('nan')
        below = (100.0 * float((finite < 0).sum()) / finite.size
                 if finite.size else 0.0)
        print(f"{name:>20} {kinds[name]:>11} {A.shape[1]:>9}  "
              f"{low:>13.4g}  {below:>17.1f}%")

    for restriction in RESTRICTIONS:
        names = restricted_pool(reps, restriction)
        print(f"\n{restriction}: {len(names)} of {len(reps)} representations "
              f"-> {names or 'NONE'}")
        if not names:
            continue
        stacked = np.hstack([np.asarray(reps[n], dtype=np.float64)
                             for n in names])
        kind, coeffs = coefficients_for_matrix(stacked, where=restriction)
        require_kind(stacked, EXPECTED_KIND[restriction], where=restriction)
        print(f"  concatenated: {stacked.shape[1]} features, kind read from the "
              f"matrix = {kind}")
        print(f"  coefficients defined on it: {coeffs}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
