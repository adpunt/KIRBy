#!/usr/bin/env python3
"""AC1: every definition, on every small-molecule dataset, with the gap in the
label's own units. HANDOFF_AC_STUDY.md 6.1 table, row AC1.

ONE ROW IS one dataset crossed with one similarity criterion, one gap criterion
and one feature space the similarity was computed on. Every row carries:

  n_molecules                      how many molecules the loader returned
  n_similar_pairs                  pairs the similarity side called similar
  n_cliff_pairs                    of those, how many the gap side also passed
  n_cliff_molecules                molecules in at least one cliff pair
  pct_cliff_molecules              those as a share of the dataset's molecules
  gap_cutoff_label_units           the gap this row used, in the label's units
  gap_over_own_repeat_spread       that gap divided by this dataset's own
                                   measured repeat spread, where it has one
  gap_over_kramer_floor            that gap divided by 0.54 pKi (Kramer et al.
                                   2012), which is BORROWED on any label that
                                   is not a public Ki
  feature_space                    which matrix the similarity ran on

WHAT IS NEW HERE, against cliff_census.py in the noise suite, which this lifts
its census from:

  the two binary hybrids are built and handed in as named representations, so
  the twelve hybrid criteria in cliff_rules run. Those are the entries that let
  a hybrid define a cliff at all, and no cliff set in this repository has ever
  been built from one.

  the gap in label units is on every row, not only on the named rules.

  the dataset list is Paper 3's small molecules, taken from papers.plan and
  filtered to the molecular domain, so it is not typed at the command line.

THE CIRCULARITY IN THE HYBRID ROWS, which no other row has. A hybrid's features
are chosen using the labels. The gap criterion is a statement about those same
labels, over the same molecules, with no fold held out. So a hybrid row's
similarity side is not label-free, and an ECFP4 row's is. The counts are still
counts of a stated definition, and this is written into every hybrid row as
`similarity_used_labels`, but it is the reason AC4 and AC5 hold a fold out.
Whether AC1's hybrid should instead be built per fold is Adelaide's.

A LOCAL RUN OF THIS IS NOT A RESULT (MEMORY.md: local runs are debug). It
settles which definitions go forward.

Usage:
    python tests/experiments/cliffs/ac1_definition_counts.py --list
    python tests/experiments/cliffs/ac1_definition_counts.py herg_ki \\
        --hybrid-source budget:250
    python tests/experiments/cliffs/ac1_definition_counts.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DEFAULT = '/Volumes/seagate/ac1_definition_counts'

# KIRBY_CACHE_DIR is deliberately NOT set here. The representation cache the
# framework used is the one the environment points at, and pinning it to a
# local path would make this file read a different cache from every other
# experiment. On this machine ~/.cache/kirby is a link to
# /Volumes/seagate/kirby_cache_local.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

from experiments.noise import cliff_rules as cr              # noqa: E402
from experiments.noise.cliff_census import (                 # noqa: E402
    _spread, own_repeat_spread)
from experiments.cliffs.binary_hybrid import (               # noqa: E402
    HYBRID_REP_NAME, RESTRICTIONS, binary_hybrid_inputs,
    coefficients_for_matrix, representation_kinds, require_kind)


def datasets():
    """Paper 3's small molecules, derived and not typed.

    papers.plan.PAPER3_DATASETS is the outline's list; the registry says which
    of them are molecular. Handoff section 7 names the same fourteen: the three
    QM9 targets, hERG Ki, hERG IC50, Caco-2, LogD, rat and human microsomal
    clearance, MDR1-MDCK, hERG blocking, and Dablander's three.
    """
    from experiments.papers.plan import PAPER3_DATASETS
    from kirby.registry import DATASET_REGISTRY
    import experiments.dataset_policy  # noqa: F401  registers the registry
    out = []
    for name in PAPER3_DATASETS:
        info = DATASET_REGISTRY.get(name)
        if info is not None and getattr(info, 'domain', None) == 'molecular':
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# The two binary hybrids, per dataset
# ---------------------------------------------------------------------------

def hybrid_by_search(reps, data, dataset_name, model):
    """The winning hybrid of the restricted search, on every row.

    `_shared.get_fitted_hybrid` is the whole-dataset door: it resolves the
    search through the decided grid (DECISIONS.md) and rebuilds the winning
    recipe's features on all the rows given. One hybrid per downstream model,
    because a dataset has two decided models and neither is a cut.

    No augmentation donors are passed. That is the one way a hybrid of 0/1
    representations stops being 0/1: `hybrid.apply_augmentation_selection`
    appends donor features, and everything else in the assembly is column
    indexing (HANDOFF_AC_STUDY.md 6.1b).
    """
    from experiments._shared import get_best_for_downstream, get_fitted_hybrid
    X, info, search = get_fitted_hybrid(
        reps, data['labels'], data.get('groups'),
        downstream_model=model, aug_condition='none', augmentations=None,
        task_type=data.get('task_type'), domain=data.get('domain'),
        dataset_name=dataset_name)
    # The same configuration get_fitted_hybrid rebuilt from, asked for the same
    # way, so the recipe recorded beside a count is the recipe that produced it.
    best = get_best_for_downstream(search, model) or {}
    bp = best.get('blueprint')
    return X, {
        'how': 'search',
        'downstream_model': model,
        'budget_features': (int(bp.budget) if getattr(bp, 'budget', None)
                            else None),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
        'search_score_mean': best.get('score_mean'),
        'search_metric': getattr(search, 'metric', None),
        'n_features_assembled': int(np.asarray(X).shape[1]),
    }


def hybrid_at_budget(reps, data, dataset_name, budget):
    """One hybrid at one named budget, through the fold-aware cached door with
    every row as the training rows.

    This is the debug path. It names a budget rather than taking the search's
    winner, so a row built this way says so and is not comparable with a row
    built by the search.
    """
    from experiments._shared import create_hybrid_cached
    n = len(np.asarray(data['labels']).ravel())
    X, info = create_hybrid_cached(
        reps, data['labels'], np.arange(n), budget=int(budget),
        groups=data.get('groups'), dataset_name=dataset_name)
    return X, {
        'how': f'budget:{int(budget)}',
        'downstream_model': None,
        'budget_features': int(budget),
        'allocation_method': 'performance_weighted',
        'importance_method': 'lightgbm_gain',
        'internal_evaluator': 'lightgbm',
        'n_features_assembled': int(np.asarray(X).shape[1]),
    }


def binary_hybrids(data, dataset_name, source='search'):
    """{feature_space label: (rep name, matrix, provenance)} for one dataset.

    One entry per restriction per hybrid the source produces. The kind of the
    ASSEMBLED matrix is checked, never assumed from the inputs: handoff 6.1c
    says a hybrid built from binary inputs that does not come back binary is a
    fault to raise.
    """
    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    built, refused = {}, {}

    if source.startswith('budget:'):
        sources = [('budget', source.split(':', 1)[1])]
    else:
        from experiments.decided_grid import decided_models
        try:
            models = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
        except Exception as exc:
            return {}, {'*': f'the decided grid has no entry for this dataset '
                             f'and search: {type(exc).__name__}: {exc}'}
        sources = [('search', m) for m in models]

    for restriction in RESTRICTIONS:
        try:
            inputs, expected = binary_hybrid_inputs(
                reps, restriction, where=dataset_name)
        except ValueError as exc:
            refused[restriction] = str(exc)
            continue
        for how, arg in sources:
            try:
                if how == 'budget':
                    X, prov = hybrid_at_budget(inputs, data, dataset_name, arg)
                else:
                    X, prov = hybrid_by_search(inputs, data, dataset_name, arg)
                require_kind(X, expected, where=f'{dataset_name} {restriction}')
            except Exception as exc:
                refused[f'{restriction}/{arg}'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            label = f'{HYBRID_REP_NAME[restriction]}@{prov["how"]}'
            read_kind, coeffs = coefficients_for_matrix(X, where=label)
            prov.update({'restriction': restriction,
                         'representations': sorted(inputs),
                         'kind_expected': expected,
                         'kind_read_from_the_assembled_matrix': read_kind,
                         'coefficients_defined_on_it': coeffs})
            built[label] = (HYBRID_REP_NAME[restriction], X, prov)
    return built, refused


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------

def _row(r, feature_space, repeat_spread, provenance=None):
    """One census row, with the gap in label units and what it is against."""
    cut = r['gap_info'].get('cutoff_label_units')
    sinfo = r['similarity_info']
    out = {
        'similarity': r['similarity'], 'gap': r['gap'],
        'feature_space': feature_space,
        'n_molecules': r['n_molecules'],
        'n_similar_pairs': r['n_similar_pairs'],
        'n_cliff_pairs': r['n_cliff_pairs'],
        'n_cliff_molecules': r['n_cliff_molecules'],
        'pct_cliff_molecules': r['pct_cliff_molecules'],
        'pct_cliff_pairs_of_similar': r['pct_cliff_pairs_of_similar'],
        'gap_cutoff_label_units': cut,
        'gap_over_own_repeat_spread': (float(cut / repeat_spread)
                                       if (cut and repeat_spread) else None),
        'gap_over_kramer_floor': (float(cut / cr.KRAMER_NOISE_FLOOR_PKI)
                                  if cut else None),
        'own_repeat_spread_label_units': repeat_spread,
        'n_cliff_pairs_inside_noise_floor':
            r['n_cliff_pairs_inside_noise_floor'],
        'pct_cliff_pairs_inside_noise_floor':
            r['pct_cliff_pairs_inside_noise_floor'],
        'similarity_cutoff': sinfo.get('cutoff', sinfo.get('cutoff_reached')),
        'n_similar_pairs_dropped_as_own_repeats':
            r['n_similar_pairs_dropped_as_own_repeats'],
        'achieved_pct_of_molecules': r['gap_info'].get('achieved_pct'),
        'gap_target_pct': r['gap_info'].get('target_pct'),
        'similarity_used_labels': provenance is not None,
    }
    if provenance is not None:
        out['hybrid'] = provenance
    return out


def _print_header():
    print(f"\n  one row is one similarity criterion crossed with one gap "
          f"criterion")
    print(f"  {'similarity':34s} {'gap':26s} {'similar pairs':>13s} "
          f"{'cliff pairs':>11s} {'cliff molecules':>15s} {'of molecules':>12s} "
          f"{'gap in label units':>18s}")


def _print_row(r):
    cut = r['gap_cutoff_label_units']
    print(f"  {str(r['similarity'])[:34]:34s} {str(r['gap'])[:26]:26s} "
          f"{r['n_similar_pairs']:>13,d} {r['n_cliff_pairs']:>11,d} "
          f"{r['n_cliff_molecules']:>15,d} "
          f"{r['pct_cliff_molecules']:>11.1f}% "
          f"{(f'{cut:.4f}' if cut is not None else '-'):>18s}", flush=True)


def census(ctx, hybrids, repeat_spread, verbose=True, sims=None, gaps=None):
    """Every available similarity criterion against every gap criterion.

    The non-hybrid criteria run once on the dataset itself. The hybrid criteria
    run once per hybrid, in a context of their own, because `cliff_rules`
    reaches a hybrid by the name `hybrid_bits` or `hybrid_counts` and two
    hybrids cannot hold one name.

    sims, gaps: names to restrict to, for the smoke. None runs everything the
        dataset can run, which is what a real run does.
    """
    rows, masks, errors = [], {}, {}
    hybrid_names = {n for n in cr.SIMILARITY_CRITERIA if 'hybrid' in n}
    gap_names = list(gaps) if gaps else list(cr.GAP_CRITERIA)

    available, excluded = cr.available_similarity_criteria(ctx)
    if sims is not None:
        available = [s for s in available if s in set(sims)]
    plain = [s for s in available if s not in hybrid_names]
    if verbose:
        _print_header()
    for s in plain:
        for g in gap_names:
            try:
                r = cr.cliff_sets(ctx, s, g)
            except Exception as exc:
                errors[f'{s}__{g}'] = f'{type(exc).__name__}: {exc}'
                continue
            masks[f'{s}__{g}'] = r.pop('molecule_mask')
            r.pop('pair_mask')
            row = _row(r, 'the dataset itself', repeat_spread)
            rows.append(row)
            if verbose:
                _print_row(row)

    for label, (rep_name, X, prov) in hybrids.items():
        hctx = cr.CliffContext(
            ctx.labels, smiles=ctx.smiles, sequences=ctx.sequences,
            reps={rep_name: X}, mmp_pairs=ctx.mmp_pairs, is_clf=ctx.is_clf,
            dataset_name=ctx.dataset_name)
        hsims, _ = cr.available_similarity_criteria(hctx)
        if sims is not None:
            hsims = [n for n in hsims if n in set(sims)]
        if verbose:
            print(f"\n  on {label}: {prov['n_features_assembled']} features "
                  f"from {len(prov['representations'])} representations "
                  f"{prov['representations']}, kind {prov['kind_expected']}")
            _print_header()
        for s in [n for n in hsims if n in hybrid_names]:
            for g in gap_names:
                try:
                    r = cr.cliff_sets(hctx, s, g)
                except Exception as exc:
                    errors[f'{label}__{s}__{g}'] = f'{type(exc).__name__}: {exc}'
                    continue
                masks[f'{label}__{s}__{g}'] = r.pop('molecule_mask')
                r.pop('pair_mask')
                row = _row(r, label, repeat_spread, provenance=prov)
                rows.append(row)
                if verbose:
                    _print_row(row)

    return rows, masks, errors, excluded


def max_pair_rows():
    """The row count above which the suite does not build a pairwise matrix.

    Read from `n35.MAX_PAIR_ROWS`, which is 12,000 and carries its own reason:
    a float32 square matrix at 12,000 rows is 576 MB. Every criterion here is
    O(n^2) and several matrices are live at once, so AC1 refuses above the same
    number rather than setting a second one.
    """
    from experiments.noise.n35_near_duplicate_disagreement import MAX_PAIR_ROWS
    return int(MAX_PAIR_ROWS)


def run_dataset(name, source='search', output=OUT_DEFAULT, verbose=True,
                rep_names=None):
    """One dataset: load it, build both hybrids, count, write the json.

    rep_names: which representations are candidates for the binary hybrid, and
        which the all-features criteria glue together. None takes the dataset's
        own pool from DATASET_REPS, which is what the framework searched. A
        named list is recorded in the result, because it changes what the
        all-features rows mean as well as which hybrid can be built.
    """
    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401

    t0 = time.time()
    print(f'\n=== {name} ===', flush=True)
    data = load_data(name, rep_names=rep_names)
    labels = np.asarray(data['labels'], dtype=float).ravel()
    reps = data.get('reps') or {}
    kinds = representation_kinds(
        {k: np.asarray(v, dtype=np.float64) for k, v in reps.items()})
    is_clf = data.get('task_type') == 'classification'

    limit = max_pair_rows()
    if len(labels) > limit:
        gb = len(labels) ** 2 * 8 / 1e9
        print(f'  REFUSED: {len(labels)} rows is over {limit}, the suite\'s '
              f'own limit on building a pairwise matrix '
              f'(n35.MAX_PAIR_ROWS). One n by n matrix of 64-bit numbers at '
              f'this size is {gb:.1f} GB and several are live at once. '
              f'Nothing was computed.', flush=True)
        return {'meta': {'dataset': name, 'n_rows_loaded': int(len(labels)),
                         'refused': f'over n35.MAX_PAIR_ROWS={limit}',
                         'one_matrix_gigabytes': round(gb, 1)},
                'rows': [], 'errors': {}}

    spread, spread_info = own_repeat_spread(name, labels, data=data)
    print(f"  {len(labels)} molecules, "
          f"{len(set(map(str, data['smiles']))) if data.get('smiles') is not None else 'no'}"
          f" unique SMILES, {len(reps)} representations", flush=True)
    for kind in ('binary', 'count', 'continuous'):
        have = [k for k, v in kinds.items() if v == kind]
        print(f"    {kind:11s} {len(have)} of {len(reps)}: {have}")
    if spread is None:
        print(f"  this dataset's own repeat spread: NONE "
              f"({spread_info.get('reason')}). Every gap is reported against "
              f"Kramer's 0.54 pKi as well, which is borrowed.", flush=True)
    else:
        print(f"  this dataset's own repeat spread: median {spread:.4f} label "
              f"units over "
              f"{spread_info['n_groups_with_disagreeing_repeats']} groups "
              f"whose repeats disagree", flush=True)
    if is_clf:
        print('  this label is a class, so the pairwise gap is a class '
              'mismatch and a gap in label units is 0 or 1, not a potency '
              'difference.', flush=True)

    hybrids, refused = binary_hybrids(data, name, source=source)
    for k, why in refused.items():
        print(f'  no hybrid for {k}: {why}', flush=True)

    ctx = cr.CliffContext(
        labels, smiles=data.get('smiles'),
        sequences=data.get('heavy_seqs') or data.get('sequences'),
        reps={k: np.asarray(v, dtype=np.float64) for k, v in reps.items()},
        mmp_pairs=data.get('mmp_pairs'), is_clf=is_clf, dataset_name=name)

    rows, masks, errors, excluded = census(ctx, hybrids, spread,
                                           verbose=verbose)

    iu = np.triu_indices(ctx.n, k=1)
    meta = {
        'dataset': name,
        'n_rows_loaded': int(len(labels)),
        'n_unique_smiles': (len(set(map(str, data['smiles'])))
                            if data.get('smiles') is not None else None),
        'task_type': data.get('task_type'),
        'label_is_class_mismatch': is_clf,
        'label_spread': _spread(labels),
        'label_median_absolute_deviation': float(
            np.median(np.abs(labels - np.median(labels)))),
        'pairwise_label_difference_spread': _spread(
            cr.pairwise_gap(labels, is_clf)[iu]),
        'representation_kinds': kinds,
        'own_repeat_spread': spread_info,
        'hybrid_source': source,
        'representations_loaded': sorted((data.get('reps') or {})),
        'representations_asked_for': (list(rep_names) if rep_names else
                                      "the dataset's own pool"),
        'hybrids_built': {k: v[2] for k, v in hybrids.items()},
        'hybrids_refused': refused,
        'similarity_criteria_excluded': excluded,
        'seconds': None,
    }
    result = {
        'meta': meta, 'rows': rows, 'errors': errors,
        'what_one_row_is': (
            'one similarity criterion crossed with one gap criterion on one '
            'feature space. Counts are pairs and molecules; the gap is in the '
            "label's own units."),
        'kramer_noise_floor_pki': cr.KRAMER_NOISE_FLOOR_PKI,
        'not_a_result_locally': (
            'a local run of this is debug. It settles which definitions go '
            'forward.'),
    }
    meta['seconds'] = round(time.time() - t0, 1)

    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, f'{name}.json')
    with open(path, 'w') as fh:
        json.dump(result, fh, indent=2, default=float)
    if masks:
        np.savez_compressed(
            os.path.join(output, f'{name}__molecule_masks.npz'), **masks)
    print(f"\n  {len(rows)} rows, {len(errors)} did not compute. "
          f"written {path} ({meta['seconds']}s)", flush=True)
    for k, v in list(errors.items())[:5]:
        print(f'    {k}: {v}')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--all', action='store_true',
                    help="Paper 3's small molecules, all of them")
    ap.add_argument('--hybrid-source', default='search',
                    help="'search' takes the winning recipe of the restricted "
                         "search through the decided grid, one per decided "
                         "model. 'budget:N' names one budget instead and is "
                         "the debug path; every row records which was used.")
    ap.add_argument('--reps', default=None,
                    help="comma-separated representations to consider. "
                         "Default is the dataset's own pool.")
    ap.add_argument('--output', default=OUT_DEFAULT)
    ap.add_argument('--list', action='store_true',
                    help='print the datasets and the criteria, load nothing')
    args = ap.parse_args(argv)

    if args.list:
        ds = datasets()
        print(f"Paper 3's small molecules, {len(ds)} datasets:")
        for d in ds:
            print(f'  {d}')
        hyb = [n for n in cr.SIMILARITY_CRITERIA if 'hybrid' in n]
        print(f'\n{len(cr.SIMILARITY_CRITERIA)} similarity criteria, of which '
              f'{len(hyb)} run on a hybrid:')
        for n in hyb:
            print(f'  {n}')
        print(f'\n{len(cr.GAP_CRITERIA)} gap criteria:')
        for n in cr.GAP_CRITERIA:
            print(f'  {n}')
        return 0

    names = datasets() if args.all else list(args.datasets)
    if not names:
        print('name at least one dataset, or pass --all, or --list')
        return 2

    failed = {}
    for name in names:
        try:
            run_dataset(name, source=args.hybrid_source, output=args.output,
                        rep_names=([r.strip() for r in args.reps.split(',')]
                                   if args.reps else None))
        except Exception as exc:
            failed[name] = f'{type(exc).__name__}: {exc}'
            print(f'  {name} FAILED {failed[name]}', flush=True)
    if failed:
        print(f'\n{len(failed)} of {len(names)} datasets failed:')
        for k, v in failed.items():
            print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
