#!/usr/bin/env python3
"""Score SMILES for REINVENT 4's ExternalProcess component.

REINVENT hands us one batch of SMILES on stdin, one per line, and reads a JSON
object from stdout. Its contract is set by
reinvent_plugins/components/comp_external_process.py at commit
660d2c9cec9ea1ced1b452394fd38452e478fc8b:

    {"version": 1, "payload": {"<property>": [score, ...]}}

Nothing in REINVENT is edited. The arm's TOML names this file as `executable`'s
argument and names the property key below.

⛔ THE OUTPUT LIST MUST BE THE SAME LENGTH AS THE INPUT, IN THE SAME ORDER.
REINVENT zips the scores against its own batch. A molecule RDKit cannot parse,
or one a representation builder drops, still needs a number, so it gets
FAILED_SCORE and the count of those goes back to REINVENT as metadata.

The bundle is written by opig_generator_scoring_swap.py and holds the fitted
predictor, the columns the training fit chose, and the scaler fitted on the
training molecules. Nothing here fits anything: a fit inside the generator's
loop would make every step's score mean something different.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))          # tests/
sys.path.insert(0, os.path.join(_HERE, '..'))                # tests/experiments/

# The number handed back for a molecule we could not featurise. It is NOT zero:
# zero is a real pKi-ish value and would read as a scored molecule. REINVENT's
# transform maps it to the bottom of the range, and the count is reported.
FAILED_SCORE = -999.0


def _load_bundle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def _score(smiles, bundle):
    """Return one float per input SMILES, in the input's order."""
    from find_best_hybrid_molecular import get_representations
    from kirby.hybrid import apply_feature_selection

    n = len(smiles)
    out = np.full(n, FAILED_SCORE, dtype=float)
    if n == 0:
        return out, 0

    # cache=False: get_representations keys its cache on a sha256 of the
    # newline-joined SMILES (_rep_cache_path, find_best_hybrid_molecular.py).
    # Every batch of generated molecules is a new key, so caching them would
    # write one directory per generator step and never read one back.
    reps = get_representations(list(smiles), bundle['rep_spec'], cache=False)

    missing = [r for r in bundle['rep_names'] if r not in reps]
    if missing:
        # A builder that raises is caught and warned about inside
        # get_representations, so a representation the fit needs can vanish
        # silently. Refuse the batch rather than score it on a narrower matrix.
        raise RuntimeError(
            f"representations the fit used are absent for this batch: "
            f"{sorted(missing)}; got {sorted(reps)}")

    # Rows RDKit could not parse come back as all-zero or non-finite. Find them
    # before selection so a bad molecule cannot move a scaler statistic.
    stacked = np.hstack([np.asarray(reps[r], dtype=float)
                         for r in bundle['rep_names']])
    ok = np.isfinite(stacked).all(axis=1) & (np.abs(stacked).sum(axis=1) > 0)
    n_failed = int((~ok).sum())
    if not ok.any():
        return out, n_failed

    sub = {r: np.asarray(reps[r], dtype=float)[ok] for r in bundle['rep_names']}
    X = apply_feature_selection(sub, bundle['feature_info'], base_only=True)
    X = bundle['scaler'].transform(X)
    out[ok] = np.asarray(bundle['model'].predict(X), dtype=float).ravel()
    return out, n_failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True,
                    help='pickle written by opig_generator_scoring_swap.py')
    ap.add_argument('--property', default='kirby_score',
                    help="payload key; must equal `property` in the arm's TOML")
    args = ap.parse_args()

    smiles = [line.strip() for line in sys.stdin.read().splitlines()]
    smiles = [s for s in smiles if s]

    bundle = _load_bundle(args.bundle)
    scores, n_failed = _score(smiles, bundle)

    payload = {
        args.property: [float(v) for v in scores],
        # Metadata: REINVENT keeps every payload key that is not the property
        # and logs it, so the count of molecules we could not featurise is
        # visible in its own log rather than only here.
        'n_smiles_in': len(smiles),
        'n_smiles_not_featurised': n_failed,
    }
    json.dump({'version': 1, 'payload': payload}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
