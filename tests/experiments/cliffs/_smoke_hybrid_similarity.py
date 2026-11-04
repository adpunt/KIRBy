#!/usr/bin/env python3
"""Smoke: every overlap score runs on a hybrid, and the counts come out.

The step this checks: tanimoto, dice, tversky and soergel could only ever be
computed on a fingerprint, because they need features that are 0 or 1 (or whole
numbers at or above zero) and a searched hybrid's features are standardised. A
hybrid built from only the dataset's 0/1 representations is itself 0/1, so it is
handed to cliff_rules as a named representation and those four scores reach it.

The hybrid here is assembled by taking columns, which is exactly what the
search's own hybrid.apply_feature_selection does, so what runs is the real path.
No search is started: which columns a search would pick does not change whether
the scores are defined.

    python3 experiments/cliffs/_smoke_hybrid_similarity.py --dataset herg_ki
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from kirby.hybrid import apply_feature_selection
from experiments.cliffs.binary_hybrid import (
    EXPECTED_KIND, HYBRID_REP_NAME, RESTRICTIONS, binary_hybrid_inputs,
    require_kind,
)
from experiments.noise import cliff_rules as cr

FAILS = []


def check(name, condition, detail=''):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}"
          f"{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILS.append(name)


def selected_hybrid(inputs, budget, seed=0):
    """A hybrid of `budget` features, taken from `inputs` the way a search
    result is turned into a matrix. The allocation is even and the columns are
    drawn at random: a search picks them by score, and which columns are picked
    does not change whether an overlap score is defined on the result."""
    rng = np.random.RandomState(seed)
    per = max(1, budget // len(inputs))
    info = {}
    for name, X in inputs.items():
        width = np.asarray(X).shape[1]
        info[name] = {'selected_indices': sorted(
            rng.choice(width, min(per, width), replace=False).tolist())}
    return apply_feature_selection(inputs, info), info


def main(argv=None):
    import argparse

    from experiments._shared import load_data

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='herg_ki')
    ap.add_argument('--budget', type=int, default=250,
                    help="hERG Ki's lowest decided budget")
    ap.add_argument('--gap', default='mad_2x',
                    help='which gap criterion to pair the similarity with')
    args = ap.parse_args(argv)

    data = load_data(args.dataset, rep_preset='fast')
    reps = {k: np.asarray(v, dtype=np.float64) for k, v in data['reps'].items()}

    hybrids = {}
    for restriction in RESTRICTIONS:
        inputs, expected = binary_hybrid_inputs(reps, restriction,
                                                where=args.dataset)
        X, info = selected_hybrid(inputs, args.budget)
        require_kind(X, expected, where=restriction)
        hybrids[HYBRID_REP_NAME[restriction]] = X
        print(f"{HYBRID_REP_NAME[restriction]}: {X.shape[1]} features from "
              f"{len(inputs)} representations {sorted(inputs)}, "
              f"kind {expected}")

    ctx = cr.CliffContext(
        labels=data['labels'],
        smiles=data.get('smiles'),
        reps={**reps, **hybrids},
        is_clf=(data.get('task_type') == 'classification'),
        dataset_name=args.dataset,
    )

    print(f"\n{args.dataset}: {ctx.n} molecules. "
          f"gap criterion = {args.gap}")
    gap_names = sorted(cr.GAP_CRITERIA)
    check("the gap criterion exists", args.gap in cr.GAP_CRITERIA,
          f"available: {gap_names}")
    if args.gap not in cr.GAP_CRITERIA:
        print("\nFAILED")
        return 1

    ok, out = cr.available_similarity_criteria(ctx)
    hybrid_names = [n for n in cr.SIMILARITY_CRITERIA if 'hybrid' in n]
    check("every hybrid criterion is available once the hybrids are loaded",
          all(n in ok for n in hybrid_names),
          f"{[n for n in hybrid_names if n not in ok]} missing"
          if any(n not in ok for n in hybrid_names) else
          f"{len(hybrid_names)} of them")

    print(f"\nOne row is one way of calling a pair similar, paired with "
          f"{args.gap}.")
    print(f"{'criterion':>38} {'similar pairs':>14} {'cliff pairs':>12} "
          f"{'cliff molecules':>16} {'of molecules':>13} {'cutoff':>8}")
    for name in hybrid_names:
        try:
            r = cr.cliff_sets(ctx, name, args.gap)
        except Exception as e:
            check(f"{name} runs", False, f"{type(e).__name__}: {str(e)[:110]}")
            continue
        sinfo = r['similarity_info']
        cut = sinfo.get('cutoff', sinfo.get('cutoff_reached'))
        print(f"{name:>38} {r['n_similar_pairs']:>14,} "
              f"{r['n_cliff_pairs']:>12,} {r['n_cliff_molecules']:>16,} "
              f"{r['pct_cliff_molecules']:>12.1f}% "
              f"{(f'{cut:.4f}' if cut is not None else '-'):>8}")
        check(f"{name}: cliff pairs do not exceed similar pairs",
              r['n_cliff_pairs'] <= r['n_similar_pairs'])
        check(f"{name}: cliff molecules do not exceed the dataset",
              r['n_cliff_molecules'] <= ctx.n)

    print("\ntwo claims the new formulas make in their docstrings")
    bits = hybrids['hybrid_bits']
    check("the count form of tanimoto equals the 0/1 form on 0/1 features",
          bool(np.allclose(cr.tanimoto_count(bits), cr.tanimoto(bits),
                           atol=1e-6)),
          f"largest difference "
          f"{float(np.abs(cr.tanimoto_count(bits) - cr.tanimoto(bits)).max()):.3g}")
    check("the count form of dice equals the 0/1 form on 0/1 features",
          bool(np.allclose(cr.dice_count(bits), cr.dice(bits), atol=1e-6)),
          f"largest difference "
          f"{float(np.abs(cr.dice_count(bits) - cr.dice(bits)).max()):.3g}")
    check("soergel equals tanimoto on 0/1 features, which is why it is off "
          "the 0/1 list",
          bool(np.allclose(cr.soergel_sim(bits), cr.tanimoto(bits), atol=1e-5)),
          f"largest difference "
          f"{float(np.abs(cr.soergel_sim(bits) - cr.tanimoto(bits)).max()):.3g}")
    tv, di = cr.tversky(bits), cr.dice(bits)
    iu = np.triu_indices(bits.shape[0], k=1)
    print(f"  [note] tversky at alpha {cr.TVERSKY_ALPHA} and beta "
          f"{cr.TVERSKY_BETA} against dice on the same features: largest "
          f"difference {float(np.abs(tv - di)[iu].max()):.4f}, median "
          f"difference {float(np.median(np.abs(tv - di)[iu])):.4f}. "
          f"At alpha and beta both 0.5 they are the same formula.")

    print("\nthe criteria that already existed still run")
    for name in ('ecfp4_tanimoto_055', 'moleculeace_any_090',
                 'ecfp4_top_1pct'):
        try:
            r = cr.cliff_sets(ctx, name, args.gap)
            check(f"{name} runs", True,
                  f"{r['n_cliff_molecules']:,} cliff molecules of {ctx.n:,}")
        except Exception as e:
            check(f"{name} runs", False, f"{type(e).__name__}: {str(e)[:100]}")

    print("\nthe guard, on a hybrid that is not 0/1")
    cont = next(k for k, X in reps.items()
                if cr_feature_kind(X) == 'continuous')
    ctx_bad = cr.CliffContext(labels=data['labels'], smiles=data.get('smiles'),
                              reps={'hybrid_bits': reps[cont]},
                              dataset_name=args.dataset)
    try:
        cr.cliff_sets(ctx_bad, 'hybrid_bits_tanimoto_090', args.gap)
        check("an overlap score refuses a continuous hybrid", False,
              "it returned a number, which it must not")
    except ValueError as e:
        check("an overlap score refuses a continuous hybrid", True,
              f"{cont}: {str(e)[:80]}")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


def cr_feature_kind(X):
    from kirby.similarity.preprocessing import feature_kind
    return feature_kind(X)


if __name__ == '__main__':
    raise SystemExit(main())
