#!/usr/bin/env python3
"""One place that turns the settings into a list of cliff definitions, and
builds the matrices each one reads.

WHAT A DEFINITION IS, in her outline's two halves:

  which pairs count as similar   a coefficient (TBD1) on a representation
                                 (TBD2), cut either at a fixed value (TBD3) or
                                 at a share of pairs, which is her second
                                 option. MoleculeACE's any-of-three rule at
                                 TBD4, the same shape with one hybrid overlap
                                 at TBD5, and the loader's matched pairs where
                                 it ships them.
  how far apart labels have to   twice the median absolute deviation of that
  be                             dataset's labels, or the gap moved until a
                                 share of molecules are cliffs (TBD6).

Every experiment in this folder reads its definitions from here, so two of
them cannot disagree about what a definition is or about what it was called.

THE REFERENCE DEFINITIONS ARE ALWAYS IN THE LIST. Her outline: "as well as the
standard ECFP4 Tanimoto with standard cutoffs (I should already be producing
those given my setup, but I need to highlight them)". So ECFP4 Tanimoto at
0.55 with a 2-log-unit gap (Hu 2015), the same similarity with twice the
median absolute deviation (her n01), and MoleculeACE's published rule are in
every list whatever the settings say, and each row says it is a reference.

WHERE THE HYBRID COMES FROM, and why it is per fold. A hybrid's features are
chosen using the labels. Computed over the whole dataset, the similarity side
of a hybrid definition has read every molecule's label, including the held-out
ones, and so has the gap side. Both are computed here on one fold's training
molecules, through cliff_rules.subset_context and the fold's own search, so a
held-out label never reaches the definition. `definition_scope` is the setting
that chooses; 'whole_dataset' is what her thesis runs did and is still
runnable, with `similarity_used_labels` and `scope` on every row.

A hybrid is named `hybrid_<kind>__<model>` so a dataset's two decided models
each contribute one and they can live in the same context. Nothing in
cliff_rules knows those are hybrids rather than fingerprints.
"""
from __future__ import annotations

import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent.parent
REPO = TESTS.parent
for _p in (REPO / 'src', TESTS, TESTS / 'experiments'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.noise import cliff_rules as cr                   # noqa: E402
from experiments.cliffs import settings as S                      # noqa: E402
from experiments.cliffs.binary_hybrid import (                    # noqa: E402
    EXPECTED_KIND, HYBRID_REP_NAME, binary_hybrid_inputs, require_kind)

#: One definition: what to call it, the two criteria, and the record that says
#: where it came from. `similarity` and `gap` are what cliff_rules.cliff_sets
#: takes -- a name from its registries or a callable carrying a `label`.
Definition = namedtuple('Definition', 'key similarity gap record')

#: The definitions every list carries, whatever the settings say. Her words:
#: "I should already be producing those given my setup, but I need to highlight
#: them". Each is a (similarity, gap) pair already named in cliff_rules.RULES.
REFERENCE_RULES = ('hu2015', 'n01', 'calibrated', 'moleculeace')


def _gap_points(stored):
    """Her two ways of setting how far apart the labels have to be.

    One entry is (gap criterion for cliff_sets, record). The first is twice the
    median absolute deviation; the rest move the gap until a share of
    molecules are cliffs, one per share in TBD6.
    """
    out = [('mad_2x', {'gap_form': 'median_absolute_deviation_multiple',
                       'gap_setting': 2.0, 'her_label': 'her first gap'})]
    for target in S.values('target_cliff_shares', stored):
        fn = (lambda ctx, similar=None, t=float(target):
              cr.gap_solved_to_target(ctx, target_pct=t, similar=similar))
        fn.label = f'solved@{float(target):g}pct'
        out.append((fn, {'gap_form': 'moved_to_a_share_of_molecules',
                         'gap_setting': float(target),
                         'her_label': 'TBD6'}))
    return out


def _coefficient_points(stored, reps_available, kinds, dataset_name=''):
    """Her first and second ways of calling a pair similar.

    One entry is (similarity criterion for cliff_sets, record). The cross
    product of the coefficients she named, the representations she named, and
    the cutoffs: a fixed value where the coefficient's scale is bounded, and a
    share of pairs in every case, which is her option 2 and the only form of
    an unbounded coefficient that carries across representations.
    """
    coefficients = S.values('coefficients', stored)
    reps = S.representations_for(dataset_name, reps_available, stored)
    fixed = S.values('similarity_cutoffs', stored)
    shares = S.values('pair_shares', stored)
    hybrid_cutoffs = S.values('hybrid_overlap_cutoffs', stored)

    out = []
    for rep in reps:
        kind = kinds.get(rep)
        if kind is None:
            continue
        is_hybrid = rep.startswith('hybrid_')
        for coef in coefficients:
            route = cr.COEFFICIENT_ROUTES[coef]
            if kind not in route['kinds']:
                continue
            cuts = (hybrid_cutoffs if (is_hybrid and route['bounded'])
                    else fixed)
            points = ([{'cutoff': float(c)} for c in cuts]
                      if route['bounded'] else [])
            points += [{'pct': float(p)} for p in shares]
            for kw in points:
                fn = cr.coefficient_criterion(rep, coef, **kw)
                out.append((fn, {
                    'similarity_form': ('fixed_cutoff' if 'cutoff' in kw
                                        else 'share_of_pairs'),
                    'coefficient': coef,
                    'representation': rep,
                    'representation_kind': kind,
                    'coefficient_route': route['route'],
                    'cutoff_is_bounded': route['bounded'],
                    'similarity_setting': kw.get('cutoff', kw.get('pct')),
                    'similarity_used_labels': is_hybrid,
                    'her_label': ('TBD5' if (is_hybrid and route['bounded'])
                                  else 'TBD1 x TBD2 x TBD3'),
                }))
    return out


def _moleculeace_points(stored, reps_available):
    """MoleculeACE's any-of-three rule at each cutoff in TBD4, and their shape
    with one hybrid overlap in place of the three at each cutoff in TBD5."""
    out = []
    for cut in S.values('moleculeace_cutoffs', stored):
        fn = lambda ctx, c=float(cut): cr.sim_moleculeace(ctx, cutoff=c)
        fn.label = f'moleculeace_any@{float(cut):g}'
        out.append((fn, {
            'similarity_form': 'any_of_three_published_scores',
            'coefficient': 'tanimoto, scaffold tanimoto or SMILES edit',
            'representation': 'ecfp4 at 1024 bits, its scaffold, and the SMILES',
            'cutoff_is_bounded': True, 'similarity_setting': float(cut),
            'similarity_used_labels': False, 'her_label': 'TBD4',
            'source': 'van Tilborg, Alenicheva & Grisoni 2022, '
                      'J. Chem. Inf. Model. 62:5938'}))
    hybrids = [r for r in reps_available if r.startswith('hybrid_')]
    for rep in hybrids:
        for cut in S.values('hybrid_overlap_cutoffs', stored):
            fn = (lambda ctx, r=rep, c=float(cut):
                  cr.sim_moleculeace_one_score(ctx, r, 'tanimoto', cutoff=c))
            fn.label = f'moleculeace_one_score|{rep}|tanimoto@{float(cut):g}'
            out.append((fn, {
                'similarity_form': 'moleculeace_shape_with_one_overlap',
                'coefficient': 'tanimoto', 'representation': rep,
                'cutoff_is_bounded': True, 'similarity_setting': float(cut),
                'similarity_used_labels': True, 'her_label': 'TBD5',
                'source': 'their shape; the cutoff is not theirs'}))
    return out


def _matched_pair_points(ctx):
    """Her "mmpdb (one named chemical change)", where the loader ships a pair
    list. Nothing is generated: cliff_rules.mmp_state records that mmpdb is not
    installed and that the literature's restriction on how much of the
    molecule may change is a parameter of its generation step, so it cannot be
    recovered from a shipped list."""
    if not ctx.mmp_pairs:
        return []
    fn = cr.sim_mmp
    fn.label = 'matched_molecular_pairs'
    return [(fn, {'similarity_form': 'one_named_chemical_change',
                  'coefficient': None,
                  'representation': "the loader's matched pair list",
                  'cutoff_is_bounded': None, 'similarity_setting': None,
                  'similarity_used_labels': False,
                  'her_label': 'her mmpdb entry',
                  'source': 'the loader; mmpdb is not installed, '
                            'cliff_rules.mmp_state()'})]


def expand(ctx, stored=None, kinds=None, scope='per_fold',
           include_references=True):
    """The list of Definitions this context can run under these settings.

    `kinds` is {representation name: 'binary'|'count'|'continuous'}, read from
    the matrices and never from the names. None reads them here.
    """
    stored = S.read() if stored is None else stored
    if kinds is None:
        from kirby.similarity.preprocessing import feature_kind
        kinds = {k: feature_kind(np.asarray(v, dtype=np.float64))
                 for k, v in ctx.reps.items()}
    reps_available = sorted(ctx.reps)

    sims = (_coefficient_points(stored, reps_available, kinds,
                                dataset_name=ctx.dataset_name or '')
            + _moleculeace_points(stored, reps_available)
            + _matched_pair_points(ctx))
    gaps = _gap_points(stored)

    out = []
    for sfn, srec in sims:
        for gfn, grec in gaps:
            gname = gfn if isinstance(gfn, str) else gfn.label
            key = f'{sfn.label}__{gname}'
            out.append(Definition(key, sfn, gfn, {
                **srec, **grec, 'key': key, 'is_reference': False,
                'definition_scope': scope,
                'similarity': sfn.label, 'gap': gname}))

    if include_references:
        have, _ = cr.available_similarity_criteria(ctx)
        for rule in REFERENCE_RULES:
            sim, gap = cr.RULES[rule]
            if sim not in have:
                continue
            key = f'reference:{rule}'
            out.append(Definition(key, sim, gap, {
                'key': key, 'is_reference': True, 'reference_rule': rule,
                'similarity': sim, 'gap': gap,
                'similarity_form': 'published', 'gap_form': 'published',
                'similarity_used_labels': False,
                'definition_scope': scope,
                'her_label': 'the standard cutoffs she wants highlighted'}))
    return out


# ---------------------------------------------------------------------------
# The restricted hybrids, built on one fold's training molecules
# ---------------------------------------------------------------------------

def restricted_hybrids(data, dataset_name, *, train_idx, fold_i, n_folds,
                       kinds_wanted=('bits', 'bits+counts'), models=None,
                       verbose=True):
    """{representation name: (matrix over every row, provenance)}.

    One entry per restricted feature kind per decided model: the search is run
    on this fold's training molecules with only the dataset's 0/1
    representations (or those and the count ones), and the winning recipe is
    rebuilt on every row so a pair between a training and a held-out molecule
    can still be scored.

    The kind of the ASSEMBLED matrix is checked, never assumed from the
    inputs: a hybrid built from 0/1 representations that does not come back
    0/1 raises rather than being worked around.
    """
    from experiments._shared import (build_fold_hybrid, fold_searches,
                                     get_best_for_downstream,
                                     get_domain_metric, get_model_factories)
    from experiments.decided_grid import decided_models

    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    labels = np.asarray(data['labels']).ravel()
    groups = data.get('groups')
    train_idx = np.asarray(train_idx)
    all_idx = np.arange(len(labels))
    metric_name, metric_fn, _lower = get_domain_metric(data)

    if models is None:
        try:
            models = [m for m in decided_models(dataset_name, 'general')
                      if m != 'knn']
        except Exception as exc:
            return {}, {'*': f'the decided grid has no entry for this '
                             f'dataset: {type(exc).__name__}: {exc}'}

    built, refused = {}, {}
    for kind in kinds_wanted:
        try:
            inputs, expected = binary_hybrid_inputs(reps, kind,
                                                    where=dataset_name)
        except ValueError as exc:
            refused[kind] = str(exc)
            continue
        tr = {k: v[train_idx] for k, v in inputs.items()}
        full = dict(inputs)
        try:
            searches = fold_searches(
                inputs, labels, groups, train_idx=train_idx, fold_i=int(fold_i),
                n_outer_folds=int(n_folds), dataset_name=dataset_name,
                task_type=data.get('task_type') or 'auto', domain=data.get('domain'),
                cache_namespace=f'cliff_{kind.replace("+", "_")}',
                verbose=False)
        except Exception as exc:
            refused[kind] = (f'the search on the {kind!r} representations '
                             f'failed: {type(exc).__name__}: {exc}')
            continue
        search = searches.get('none') or next(
            (s for s in searches.values() if s is not None), None)
        if search is None:
            refused[kind] = 'the search on this fold returned nothing'
            continue
        for model in models:
            best = get_best_for_downstream(search, model)
            bp = (best or {}).get('blueprint')
            if bp is None:
                refused[f'{kind}/{model}'] = (
                    'the search has no winning recipe for this model')
                continue
            try:
                factory = get_model_factories([model],
                                              data.get('task_type') or 'regression',
                                              len(labels)).get(model)
                _X_tr, X_all, info = build_fold_hybrid(
                    tr, full, labels[train_idx], bp,
                    groups=(None if groups is None
                            else np.asarray(groups)[train_idx]),
                    model=(factory() if factory else None),
                    metric_fn=metric_fn)
                X_all = np.asarray(X_all, dtype=np.float64)
                require_kind(X_all, expected,
                             where=f'{dataset_name} {kind} fold {fold_i}')
            except Exception as exc:
                refused[f'{kind}/{model}'] = f'{type(exc).__name__}: {exc}'
                continue
            name = f'{HYBRID_REP_NAME[kind]}__{model}'
            built[name] = (X_all, {
                'restriction': kind,
                'representations': sorted(inputs),
                'downstream_model': model,
                'fold': int(fold_i),
                'built_on': 'this fold\'s training molecules',
                'n_training_molecules': int(len(train_idx)),
                'budget_features': (int(bp.budget)
                                    if getattr(bp, 'budget', None) else None),
                'allocation_method': getattr(bp, 'allocation_method', None),
                'importance_method': getattr(bp, 'importance_method', None),
                'internal_evaluator': getattr(bp, 'internal_evaluator', None),
                'search_score_mean': (best or {}).get('score_mean'),
                'search_metric': metric_name,
                'kind_expected': expected,
                'n_features_assembled': int(X_all.shape[1]),
            })
            if verbose:
                print(f"    {name}: {X_all.shape[1]} features from "
                      f"{len(inputs)} representations, kind {expected}",
                      flush=True)
    return built, refused


def context_for_fold(data, dataset_name, *, train_idx, fold_i, n_folds,
                     stored=None, scope='per_fold', verbose=True):
    """The CliffContext a fold's definitions are computed in, and its record.

    'per_fold' restricts every matrix to the fold's TRAINING molecules, which
    is the only scope under which no held-out label reaches the definition.
    'whole_dataset' keeps every molecule and is what her thesis runs did.

    The restricted hybrids are rebuilt on every row either way, so the
    'per_fold' context holds the training rows of a hybrid that was searched
    on exactly those rows.
    """
    stored = S.read() if stored is None else stored
    labels = np.asarray(data['labels']).ravel()
    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    kinds_wanted = S.values('hybrid_feature_kinds', stored)

    hybrids, refused = restricted_hybrids(
        data, dataset_name, train_idx=train_idx, fold_i=fold_i,
        n_folds=n_folds, kinds_wanted=kinds_wanted, verbose=verbose)
    for name, (X, _prov) in hybrids.items():
        reps[name] = X

    whole = cr.CliffContext(
        labels, smiles=data.get('smiles'),
        sequences=data.get('heavy_seqs') or data.get('sequences'),
        reps=reps, mmp_pairs=data.get('mmp_pairs'),
        is_clf=(data.get('task_type') == 'classification'),
        dataset_name=dataset_name)

    if scope == 'whole_dataset':
        ctx = whole
    elif scope == 'per_fold':
        ctx = cr.subset_context(whole, np.asarray(train_idx))
    else:
        raise ValueError(
            f"definition_scope {scope!r} is not one of {S.DEFINITION_SCOPES}")

    record = {
        'definition_scope': scope,
        'fold': int(fold_i),
        'n_folds': int(n_folds),
        'n_molecules_in_the_context': int(ctx.n),
        'n_molecules_in_the_dataset': int(len(labels)),
        'hybrids_built': {k: v[1] for k, v in hybrids.items()},
        'hybrids_refused': refused,
        'representations_in_the_context': sorted(ctx.reps),
    }
    return ctx, whole, record


def _gap_cutoffs_for(ctx, similar, gap_points):
    """The gap cutoff each gap point asks for on this similar set, in the
    label's own units, plus what each one recorded.

    Every point is answered off ONE pass over the pairs. `mad_2x` is a number
    the labels give and reads nothing about the similar set; the moved gaps all
    read the same curve of reachable gaps, which `cliff_rules._reachable_gaps`
    builds once.
    """
    out = []
    cuts, counts = cr._reachable_gaps(ctx, similar)
    shares = (100.0 * counts / ctx.n) if ctx.n else np.zeros(len(counts))
    for gfn, grec in gap_points:
        # On a class label there is no gap to move: every gap criterion
        # degenerates to the class-mismatch indicator at 0.5, which is what
        # cliff_rules returns, so the curve above is not read at all.
        if (ctx.is_clf or isinstance(gfn, str)
                or grec['gap_form'] != 'moved_to_a_share_of_molecules'):
            # Let the criterion itself say what its cutoff is; it reads the
            # labels and not the pairs, so this costs nothing per point.
            _far, ginfo = (cr.GAP_CRITERIA[gfn](ctx, similar=similar)
                           if isinstance(gfn, str) else gfn(ctx, similar=similar))
            # On a class label every gap criterion returns the mismatch
            # indicator at 0.5 and records it as 'cutoff', not in label units,
            # because 0 or 1 is not a potency difference.
            cut = ginfo.get('cutoff_label_units', ginfo.get('cutoff'))
            out.append((gfn, grec, cut, ginfo))
            continue
        target = float(grec['gap_setting'])
        pick = int(np.argmin(np.abs(shares - target)))
        mad = float(np.median(np.abs(ctx.labels - np.median(ctx.labels))))
        out.append((gfn, grec, float(cuts[pick]), {
            'gap': 'solved_to_target_share', 'target_pct': target,
            'achieved_pct': float(shares[pick]),
            'cutoff_label_units': float(cuts[pick]),
            'cutoff_in_median_absolute_deviations': (
                float(cuts[pick] / mad) if mad > 0 else None),
            'reached_target': bool(abs(shares[pick] - target) < 1.0),
            'how_the_gap_was_found': 'exact, off the sorted per-molecule '
                                     'largest gap to a similar partner',
            'source': 'similarity_helpers.solve_activity_gap'}))
    return out


def sweep_one_similarity(ctx, sim_fn, gap_points):
    """Every gap point against one similarity criterion, in one pass.

    ⛔ WHY THIS DOES NOT CALL cliff_sets PER GAP POINT. cliff_sets builds four
    n-by-n arrays per call, and the gap axis asks for seven of them on one
    similar set: measured 2026-11-04 at 175 ms per definition on 1,412
    molecules, which is 37 minutes for one fold's 12,740 definitions and, since
    the cost is in n squared, about 31 hours on QM9's 9,978. Here the pair
    gaps of the similar set and the largest gap per molecule are extracted
    once, and every count at every cutoff is then read off two sorted vectors:

      cliff pairs at cutoff c          how many similar-pair gaps exceed c
      cliff pairs inside the floor     how many of those are at or under 0.54
      cliff molecules at cutoff c      how many molecules whose largest gap to
                                       a similar partner exceeds c

    The numbers are the ones cliff_sets returns, which _smoke_definitions.py
    checks pair by pair rather than taking on trust.
    """
    similar, sinfo = sim_fn(ctx) if not isinstance(sim_fn, str) else \
        cr.SIMILARITY_CRITERIA[sim_fn](ctx)
    similar = np.asarray(similar, dtype=bool).copy()
    np.fill_diagonal(similar, False)
    dropped_repeats = 0
    same = ctx.same_molecule
    if same.any():
        dropped_repeats = int((similar & same).sum()) // 2
        similar &= ~same

    gap = ctx.gap_matrix
    iu = np.triu_indices(ctx.n, k=1)
    pair_gaps = np.sort(gap[iu][similar[iu]])          # ascending
    n_similar = int(pair_gaps.size)
    best = np.sort(cr.largest_gap_to_a_similar_partner(ctx, similar))
    floor = cr.KRAMER_NOISE_FLOOR_PKI
    n_at_or_under_floor = int(np.searchsorted(pair_gaps, floor, side='right'))

    out = []
    for gfn, grec, cut, ginfo in _gap_cutoffs_for(ctx, similar, gap_points):
        if cut is None:
            out.append((gfn, grec, None, None))
            continue
        c = float(cut)
        above = n_similar - int(np.searchsorted(pair_gaps, c, side='right'))
        inside = max(n_at_or_under_floor
                     - int(np.searchsorted(pair_gaps, c, side='right')), 0)
        n_cliff_mol = len(best) - int(np.searchsorted(best, c, side='right'))
        counts = {
            'similarity': getattr(sim_fn, 'label', str(sim_fn)),
            'gap': gfn if isinstance(gfn, str) else gfn.label,
            'n_molecules': int(ctx.n),
            'n_similar_pairs': n_similar,
            'n_cliff_pairs': int(above),
            'n_cliff_molecules': int(n_cliff_mol),
            'pct_cliff_molecules': (100.0 * n_cliff_mol / ctx.n) if ctx.n else 0.0,
            'pct_cliff_pairs_of_similar': (100.0 * above / n_similar
                                           if n_similar else None),
            'n_cliff_pairs_inside_noise_floor': int(inside),
            'pct_cliff_pairs_inside_noise_floor': (100.0 * inside / above
                                                   if above else None),
            'n_similar_pairs_dropped_as_own_repeats': dropped_repeats,
            'similarity_info': sinfo, 'gap_info': ginfo,
            'subset_of': getattr(ctx, 'subset_of', None),
        }
        out.append((gfn, grec, c, counts))
    # The per-molecule flag, for the one cutoff a caller asks to keep. Built
    # from the unsorted array so the mask lines up with the molecules.
    unsorted_best = cr.largest_gap_to_a_similar_partner(ctx, similar)
    return out, unsorted_best, similar


def similarity_group(d):
    """The memoised similarity matrix a definition reads, as a name to group
    on. Definitions sharing a group build one matrix between them."""
    fn = d.similarity
    if isinstance(fn, str):
        return f'named:{fn}'
    return getattr(fn, 'similarity_group', None) or f'label:{fn.label}'


def masks(ctx, defs, verbose=False, free_between_groups=True,
          keep_masks=True):
    """{definition key: (molecule mask, counts)}, {key: why it did not run},
    and what the run cost.

    ⛔ THE ORDER IS NOT COSMETIC, two reasons.

    Definitions are run grouped by the n-by-n matrix they read, and the
    matrices of a finished group are dropped before the next group starts. One
    matrix is 8 MB at herg_ki's 1,412 molecules and 398 MB at QM9's 9,978; a
    sweep over 13 coefficients on 14 representations asks for 182 of them, so
    holding them all is 1.5 GB on one dataset and 72 GB on the other.

    Inside a group the definitions that share one similarity criterion are run
    together, because every gap point then reads the same two sorted vectors
    instead of its own four n-by-n arrays (sweep_one_similarity).

    keep_masks=False returns the counts and no per-molecule flag, for a caller
    that wants the table and not the sets.
    """
    out, errors = {}, {}
    groups = {}
    for d in defs:
        groups.setdefault(similarity_group(d), {}).setdefault(
            _similarity_label(d), []).append(d)
    cost = {'n_similarity_groups': len(groups), 'n_similarity_criteria': 0,
            'n_matrices_dropped': 0, 'megabytes_freed': 0.0}
    for gi, (gname, by_sim) in enumerate(groups.items()):
        for sim_label, group in by_sim.items():
            cost['n_similarity_criteria'] += 1
            gap_points = [(d.gap, d.record) for d in group]
            try:
                swept, best, _similar = sweep_one_similarity(
                    ctx, group[0].similarity, gap_points)
            except Exception as exc:
                why = f'{type(exc).__name__}: {exc}'
                for d in group:
                    errors[d.key] = why
                continue
            for d, (_gfn, _grec, cut, counts) in zip(group, swept):
                if counts is None:
                    errors[d.key] = ('the gap criterion returned no cutoff in '
                                     'the label\'s own units')
                    continue
                mask = (best > float(cut)) if keep_masks else None
                out[d.key] = (mask, counts)
                if verbose:
                    print(f"    {d.key[:66]:66s} "
                          f"{counts['n_similar_pairs']:>9,d} similar pairs  "
                          f"{counts['n_cliff_molecules']:>7,d} cliff molecules",
                          flush=True)
        if free_between_groups:
            freed = ctx.forget_similarities()
            cost['n_matrices_dropped'] += freed['n_matrices_dropped']
            cost['megabytes_freed'] += freed['megabytes_freed']
        if verbose:
            print(f"  [{gi + 1}/{len(groups)}] {gname}: "
                  f"{sum(len(v) for v in by_sim.values())} definitions",
                  flush=True)
    return out, errors, cost


def _similarity_label(d):
    fn = d.similarity
    return fn if isinstance(fn, str) else fn.label


def row(d, counts, record, extra=None):
    """One output row: the definition, what it came from, and its counts.

    Every number carries what it counts. `gap_cutoff_label_units` is in the
    dataset's own label units, which is why it is never compared across
    datasets without the unit beside it.
    """
    ginfo = counts.get('gap_info') or {}
    sinfo = counts.get('similarity_info') or {}
    cut = ginfo.get('cutoff_label_units')
    out = {
        **{k: v for k, v in d.record.items()},
        'n_molecules': counts['n_molecules'],
        'n_similar_pairs': counts['n_similar_pairs'],
        'n_cliff_pairs': counts['n_cliff_pairs'],
        'n_cliff_molecules': counts['n_cliff_molecules'],
        'pct_cliff_molecules': counts['pct_cliff_molecules'],
        'pct_cliff_pairs_of_similar': counts['pct_cliff_pairs_of_similar'],
        'gap_cutoff_label_units': cut,
        'gap_over_kramer_floor': (float(cut / cr.KRAMER_NOISE_FLOOR_PKI)
                                  if cut else None),
        'n_cliff_pairs_inside_noise_floor':
            counts['n_cliff_pairs_inside_noise_floor'],
        'pct_cliff_pairs_inside_noise_floor':
            counts['pct_cliff_pairs_inside_noise_floor'],
        'n_similar_pairs_dropped_as_own_repeats':
            counts['n_similar_pairs_dropped_as_own_repeats'],
        'similarity_cutoff_reached': sinfo.get('cutoff_reached'),
        'achieved_pct_of_molecules': ginfo.get('achieved_pct'),
        'gap_target_pct': ginfo.get('target_pct'),
        'fold': record.get('fold'),
    }
    if extra:
        out.update(extra)
    return out


def main(argv=None):
    """Print how many definitions one dataset's settings produce, and the cost
    of one fold, without running a search. Loads the representations only to
    read their kinds."""
    import argparse
    import time

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument('dataset')
    ap.add_argument('--no-hybrids', action='store_true',
                    help='count the definitions a dataset with no hybrid '
                         'built would run')
    a = ap.parse_args(argv)

    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401
    from kirby.similarity.preprocessing import feature_kind

    t0 = time.time()
    data = load_data(a.dataset)
    labels = np.asarray(data['labels']).ravel()
    reps = {k: np.asarray(v, dtype=np.float64)
            for k, v in (data.get('reps') or {}).items()}
    kinds = {k: feature_kind(v) for k, v in reps.items()}
    if not a.no_hybrids:
        for kind, name in HYBRID_REP_NAME.items():
            kinds[f'{name}__model'] = EXPECTED_KIND[kind]
            reps[f'{name}__model'] = np.zeros((len(labels), 1))
    ctx = cr.CliffContext(labels, smiles=data.get('smiles'), reps=reps,
                          is_clf=(data.get('task_type') == 'classification'),
                          dataset_name=a.dataset)
    defs = expand(ctx, kinds=kinds)
    print(f'\n{a.dataset}: {len(labels)} molecules, {len(reps)} '
          f'representations, loaded in {time.time() - t0:.1f}s')
    for k in ('binary', 'count', 'continuous'):
        have = sorted(n for n, v in kinds.items() if v == k)
        print(f'  {k:11s} {len(have):2d}: {have}')
    print(f'\n{len(defs)} definitions, of which '
          f'{sum(1 for d in defs if d.record["is_reference"])} are references '
          f'and {sum(1 for d in defs if d.record.get("similarity_used_labels"))}'
          f' have a similarity side that read the labels')
    by_form = {}
    for d in defs:
        by_form.setdefault((d.record.get('similarity_form'),
                            d.record.get('gap_form')), []).append(d)
    print(f"\n  {'similarity form':38s} {'gap form':38s} {'definitions':>11s}")
    for (sf, gf), group in sorted(by_form.items(), key=lambda kv: str(kv[0])):
        print(f'  {str(sf)[:38]:38s} {str(gf)[:38]:38s} {len(group):>11,d}')
    print(f'\n{S.path()}')
    for name in S.unset():
        print(f'  NOT SET: {name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
