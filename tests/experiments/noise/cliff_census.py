#!/usr/bin/env python3
"""Counts under every cliff rule, per dataset. Measurements 1, 2, 3 and 6 of
HANDOFF_PAPER3_CLIFFS.md section 1f.

WHAT IT PRODUCES, per dataset:

  every pairing of a similarity criterion and a gap criterion from
  cliff_rules: how many pairs the similarity side passed, how many of those
  the gap side also passed, how many molecules end up flagged, and that as a
  share of the dataset. The spread of pairwise similarity and of pairwise
  label difference is printed above the table, so a count can be read against
  what the dataset had to offer.                              (measurement 1)

  for the solved gap, the ACHIEVED share beside the 15 per cent it aims at,
  and the number of similar pairs available to reach it.      (measurement 2)

  every pair of rules, how many molecules both call cliffs and how many only
  one does.                                                   (measurement 3)

  the share of each rule's cliff pairs whose label gap is under 0.54 pKi,
  Kramer's standard deviation. A rule that mostly calls pairs inside the
  measurement floor is calling noise.                          (measurement 6)

WHAT IT DOES NOT DO. No model is fitted and no fold is held out: every count
is over all the molecules the loader returns. That is on purpose for a census
of definitions, and it means a count here is NOT the count a noise experiment
sees -- n25 computes its mask on the fold's training rows at their damaged
labels, for the two reasons its docstring gives. The per-fold version belongs
in n47.

A LOCAL RUN OF THIS IS NOT A RESULT. It settles which definitions to carry
forward (MEMORY.md, local runs are debug).

Usage:
    python tests/experiments/noise/cliff_census.py herg_ki
    python tests/experiments/noise/cliff_census.py herg_ki caco2 --reps
    python tests/experiments/noise/cliff_census.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DEFAULT = '/Volumes/seagate/cliff_census'

os.environ.setdefault('KIRBY_CACHE_DIR', '/Volumes/seagate/kirby_cache')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'tests'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'experiments'))

from experiments.noise import cliff_rules as cr  # noqa: E402

# Representations the embedding criteria want, where the dataset has them.
EMBEDDING_REPS = ('chemberta', 'molformer', 'chemeleon')


def _spread(v):
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {'min': float(v.min()), 'p05': float(np.percentile(v, 5)),
            'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max()), 'mean': float(v.mean())}


def build_context(dataset_name, with_reps=False):
    """A CliffContext for one dataset, and the loader's own metadata."""
    from experiments._shared import load_data
    import experiments.dataset_policy  # noqa: F401  registers the registry

    rep_names = list(EMBEDDING_REPS) if with_reps else []
    data = load_data(dataset_name, rep_names=(rep_names or None))
    labels = np.asarray(data['labels'], dtype=float).ravel()
    reps = data.get('reps') or {}
    if not with_reps:
        reps = {}
    ctx = cr.CliffContext(
        labels,
        smiles=data.get('smiles'),
        sequences=data.get('heavy_seqs') or data.get('sequences'),
        reps=reps,
        mmp_pairs=data.get('mmp_pairs'),
        is_clf=(data.get('task_type') == 'classification'),
        dataset_name=dataset_name)
    meta = {
        'dataset': dataset_name,
        'n_rows': int(len(labels)),
        'n_unique_smiles': (len(set(map(str, data['smiles'])))
                            if data.get('smiles') is not None else None),
        'domain': data.get('domain', 'molecular'),
        'task_type': data.get('task_type'),
        'label_spread': _spread(labels),
        'label_median_absolute_deviation': float(
            np.median(np.abs(labels - np.median(labels)))),
        'representations_loaded': sorted(reps),
    }
    return ctx, meta


def census(ctx, meta, verbose=True):
    """Every available similarity criterion against every gap criterion."""
    sims, excluded = cr.available_similarity_criteria(ctx)
    rows, masks, errors = [], {}, {}

    iu = np.triu_indices(ctx.n, k=1)
    meta['pairwise_label_difference_spread'] = _spread(
        cr.pairwise_gap(ctx.labels, ctx.is_clf)[iu])
    meta['similarity_criteria_available'] = sims
    meta['similarity_criteria_excluded'] = excluded

    for s in sims:
        t0 = time.time()
        for g in cr.GAP_CRITERIA:
            key = f'{s}__{g}'
            try:
                r = cr.cliff_sets(ctx, s, g)
            except Exception as exc:
                errors[key] = f'{type(exc).__name__}: {exc}'
                continue
            masks[key] = r.pop('molecule_mask')
            r.pop('pair_mask')
            rows.append(r)
            if verbose:
                pct = r['pct_cliff_molecules']
                print(f"  {s:30s} {g:32s} similar={r['n_similar_pairs']:>8d} "
                      f"cliff pairs={r['n_cliff_pairs']:>8d} "
                      f"cliff molecules={r['n_cliff_molecules']:>6d} of "
                      f"{r['n_molecules']} ({pct:5.1f}%)", flush=True)
        # The similarity matrix is built once per criterion and reused across
        # every gap; this is the cost of that criterion, not of the pairing.
        if verbose:
            print(f"  -- {s}: {time.time() - t0:.1f}s for "
                  f"{len(cr.GAP_CRITERIA)} gap criteria", flush=True)

    # Measurement 2: where the solved gap cannot reach its target.
    solved = [{'similarity': r['similarity'],
               'target_pct': r['gap_info'].get('target_pct'),
               'achieved_pct': r['gap_info'].get('achieved_pct'),
               'reached_target': r['gap_info'].get('reached_target'),
               'n_similar_pairs_available': r['n_similar_pairs'],
               'cutoff_in_median_absolute_deviations': r['gap_info'].get(
                   'cutoff_in_median_absolute_deviations')}
              for r in rows if r['gap'] == 'solved_target_15pct']

    # Measurement 3: molecule-level agreement between every pair of rules.
    keys = sorted(masks)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ag = cr.agreement(masks[a], masks[b])
            ag['first'], ag['second'] = a, b
            pairs.append(ag)

    # The named rules on their own, so a reader does not have to find them in
    # the full grid.
    named, named_masks = {}, {}
    for name, (s, g) in cr.RULES.items():
        key = f'{s}__{g}'
        hit = next((r for r in rows if r['similarity'] == s and r['gap'] == g),
                   None)
        if hit is None:
            named[name] = {'unavailable': excluded.get(s) or errors.get(key)
                           or 'not computed'}
            continue
        named_masks[name] = masks[key]
        named[name] = {
            'similarity': s, 'gap': g,
            'n_cliff_molecules': hit['n_cliff_molecules'],
            'pct_cliff_molecules': hit['pct_cliff_molecules'],
            'n_cliff_pairs': hit['n_cliff_pairs'],
            'n_similar_pairs': hit['n_similar_pairs'],
            'gap_cutoff_label_units': hit['gap_info'].get('cutoff_label_units'),
            # How the gap this rule uses sits against Kramer's 0.54 pKi. Under
            # 1.0 means the rule calls pairs cliffs whose label difference is
            # inside the spread of two independent measurements of one pair.
            'gap_cutoff_over_noise_floor': (
                float(hit['gap_info']['cutoff_label_units']
                      / cr.KRAMER_NOISE_FLOOR_PKI)
                if hit['gap_info'].get('cutoff_label_units') else None),
            'pct_cliff_pairs_inside_noise_floor': hit[
                'pct_cliff_pairs_inside_noise_floor']}

    # Measurement 3, restricted to the named rules: the full grid's least
    # agreeing pairs are dominated by rules that flag almost nothing, where a
    # Jaccard near zero says only that one set is tiny.
    nkeys = sorted(named_masks)
    named_pairs = []
    for i, a in enumerate(nkeys):
        for b in nkeys[i + 1:]:
            ag = cr.agreement(named_masks[a], named_masks[b])
            ag['first'], ag['second'] = a, b
            named_pairs.append(ag)

    return {'meta': meta, 'rows': rows, 'named_rules': named,
            'solved_gap': solved, 'agreement': pairs,
            'named_rule_agreement': named_pairs, 'errors': errors}, masks


def continuous_scores(ctx, verbose=True):
    """Table 1c of the handoff: a number per molecule or per dataset instead of
    a yes-or-no flag.

    The per-molecule scores are summarised by their spread, not stored in full;
    the full arrays go in the npz beside the json. Everything here is computed
    on the ECFP4 Tanimoto matrix and its 1 - Tanimoto distance, so the scores
    are comparable with each other and with the ECFP4 rules above. Which
    similarity a continuous score should run on is the same open question as
    for the flags.
    """
    out, arrays = {}, {}
    if ctx.smiles is None:
        return {'available': False, 'reason': 'no SMILES'}, arrays

    bits, bad = ctx.bits('ecfp4')
    S = ctx.sim('ecfp4_tanimoto', lambda: cr.tanimoto(bits, bad))
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    y = ctx.labels

    for name, how in (('sali_max', 'max'), ('sali_mean', 'mean')):
        v = cr.sali_per_molecule(S, y, how=how)
        arrays[name] = v
        out[name] = _spread(v)
    out['sali_clip'] = cr.sali_clipped_pairs(S)

    v = cr.local_roughness(D, y, k=5)
    arrays['local_roughness'] = v
    out['local_roughness'] = _spread(v)

    # Averaged across the three fingerprint similarities the suite can build
    # without loading a representation, which is the research session's answer
    # to a score depending on one representation.
    sims = {'ecfp4_tanimoto': S}
    for kind, key in (('maccs', 'maccs_tanimoto'), ('rdkit', 'rdkit_dice')):
        b, bd = ctx.bits(kind)
        sims[key] = (cr.tanimoto(b, bd) if kind == 'maccs'
                     else cr.dice(b, bd))
    avg, info = cr.sali_averaged_over_similarities(sims, y, how='max')
    arrays['sali_max_averaged'] = avg
    out['sali_max_averaged'] = _spread(avg)
    out['sali_max_averaged_spread_across_similarities'] = _spread(
        info['spread_across_similarities'])
    out['sali_max_per_similarity'] = {
        k: _spread(v) for k, v in info['per_similarity'].items()}

    riv = cr.rivality_and_modi(D, y, is_clf=ctx.is_clf)
    arrays['rivality_index'] = riv.pop('rivality_index')
    out['rivality'] = riv

    out['sari'] = cr.sari(S, y)
    out['sas_map'] = cr.sas_map(S, y)
    try:
        out['rogi'] = cr.rogi(D, y)
    except Exception as exc:
        out['rogi'] = {'available': False,
                       'reason': f'{type(exc).__name__}: {exc}'}
    if isinstance(out.get('rogi'), dict):
        out['rogi'].pop('sigma_at_thresholds', None)

    if verbose:
        for k in ('sali_max', 'sali_mean', 'local_roughness',
                  'sali_max_averaged'):
            sp = out.get(k) or {}
            if sp:
                print(f"  {k:28s} median {sp['median']:.3f}, 95th "
                      f"{sp['p95']:.3f}, max {sp['max']:.3f}")
        sa = out['sari']
        if sa.get('available'):
            print(f"  sari  raw continuity {sa['raw_continuity']:.4f}, raw "
                  f"discontinuity "
                  f"{(sa['raw_discontinuity'] if sa['raw_discontinuity'] is not None else float('nan')):.4f} "
                  f"over {sa['n_pairs_above_discontinuity_cut']} pairs above "
                  f"similarity {sa['discontinuity_similarity_cut']} "
                  f"(combination unverified)")
        rg = out['rogi']
        if rg.get('available'):
            print(f"  rogi  {rg['rogi']:.4f} (dispersion step unverified, "
                  f"rogi package not installed)")
        sm = out['sas_map']
        print(f"  sas map  cliff quadrant {sm['cliff_quadrant_high_similarity_high_gap']} "
              f"of {sm['n_pairs']} pairs, at similarity "
              f"{sm['similarity_threshold']:.3f} and gap "
              f"{sm['label_gap_threshold']:.3f} (both medians)")
        cl = out['sali_clip']
        print(f"  sali clip  {cl['n_pairs_at_the_clip']} pairs at similarity "
              f"1.0, covering {cl['n_molecules_in_a_clipped_pair']} molecules; "
              f"their SALI is the clip, not the landscape")
        rv = out['rivality']
        print(f"  rivality  share below zero "
              f"{rv['share_rivality_below_zero']:.3f}, share whose nearest "
              f"neighbour is the other class "
              f"{rv['share_nearest_neighbour_other_class']:.3f} "
              f"(classes from {rv['classes_from']})")
    return out, arrays


# ---------------------------------------------------------------------------
# The threshold sweep. HANDOFF_PAPER3_CLIFFS.md section D.
#
# WHY IT EXISTS. Every cutoff in cliff_rules.py was copied from this repo or
# from a paper -- 0.55, 0.90, 0.85, 0.70 on the similarity side, 1, 2 and 3
# median absolute deviations and 0.5, 1 and 2 label units on the gap side. Three
# or four points on an axis is not a search and no curve can be read from it.
#
# WHY IT IS NEARLY FREE. A similarity matrix is built once per fingerprint and
# then thresholded as many times as asked, and CliffContext memoises it. The
# pairwise label gap is built once per dataset (cliff_rules.CliffContext.
# gap_matrix). So the sweep is thresholding, not computing.
# ---------------------------------------------------------------------------

def own_repeat_spread(dataset_name, labels, data=None):
    """This dataset's own repeatability, in its label's own units, or None.

    data: the dict load_data already returned, when the caller has it. Without
        it this loads the dataset again, and a second load_data with no
        rep_names rebuilds the whole representation pool (added 2026-11-04 for
        the AC experiments, which load a restricted pool on purpose and had it
        silently widened here).

    Row 3 of HANDOFF_PAPER3_CLIFFS.md, and the number section C says settles
    whether a gap may be tuned per dataset. A cliff should be a label
    difference larger than the assay can repeat, and that number legitimately
    differs between datasets, which a fixed share of molecules does not.

    Built from the loader's own repeat structure through
    noise/_ground_truth_providers: `build_replicate_groups` makes one group per
    (drug, protein) and `compute_replicate_stds` returns each group's spread.
    Returns None where the loader exposes no repeat structure, and the caller
    must then report that the dataset has none rather than borrowing Kramer's
    0.54 pKi, which is a figure for public Ki data.
    """
    try:
        from experiments._shared import load_data
        from experiments.noise._ground_truth_providers import (
            build_replicate_groups, compute_replicate_stds)
    except Exception as exc:
        return None, {'available': False,
                      'reason': f'{type(exc).__name__}: {exc}'}
    try:
        if data is None:
            data = load_data(dataset_name)
        groups = build_replicate_groups(data)
    except Exception as exc:
        return None, {'available': False,
                      'reason': f'{type(exc).__name__}: {exc}'}
    if groups is None:
        return None, {'available': False,
                      'reason': 'the loader exposes no repeat structure: no '
                                'replicate_groups, and no drug_ids with '
                                'protein_ids in its metadata'}
    y = np.asarray(data['labels'], dtype=float).ravel()
    stds, has_rep = compute_replicate_stds(y, groups)
    if not has_rep.any():
        return None, {
            'available': False,
            'n_groups': int(len(set(np.asarray(groups).tolist()))),
            'reason': 'repeat groups exist but no group\'s repeats disagree, '
                      'so there is no measured spread'}
    v = stds[has_rep]
    # One number per GROUP, not per row: broadcasting a group's spread to every
    # member would count one measured disagreement once per repeat.
    per_group = {}
    g = np.asarray(groups)
    for gid in np.unique(g[has_rep]):
        per_group[str(gid)] = float(stds[g == gid][0])
    vals = np.asarray(sorted(per_group.values()), dtype=float)
    return float(np.median(vals)), {
        'available': True,
        'median_spread_label_units': float(np.median(vals)),
        'mean_spread_label_units': float(vals.mean()),
        'p95_spread_label_units': float(np.percentile(vals, 95)),
        'max_spread_label_units': float(vals.max()),
        'n_groups_with_disagreeing_repeats': int(len(per_group)),
        'n_rows_in_such_a_group': int(has_rep.sum()),
        'n_rows_total': int(len(y)),
        'note': 'the median over GROUPS, one number a group. Kramer et al. '
                '2012 give 0.54 pKi for two independent public Ki '
                'measurements of one pair; this is the same quantity '
                'measured on this dataset, in this label\'s units.'}


def _labelled(fn, label):
    fn.label = label
    return fn


def _steps(lo, hi, step):
    """lo to hi inclusive, in units of `step`, without floating-point drift."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


#: The similarity axis. One entry is one cutoff of one criterion.
def similarity_grid():
    pts = []
    for cut in _steps(0.30, 0.95, 0.05):
        pts.append(_labelled(lambda ctx, c=cut: cr.sim_ecfp4_tanimoto(ctx, cutoff=c),
                             f'ecfp4_tanimoto@{cut:.2f}'))
        pts.append(_labelled(
            lambda ctx, c=cut: cr.sim_maccs_tanimoto(ctx, cutoff=c),
            f'maccs_tanimoto@{cut:.2f}'))
        pts.append(_labelled(lambda ctx, c=cut: cr.sim_rdkit_dice(ctx, cutoff=c),
                             f'rdkit_dice@{cut:.2f}'))
        pts.append(_labelled(lambda ctx, c=cut: cr.sim_ecfp4_dice(ctx, cutoff=c),
                             f'ecfp4_dice@{cut:.2f}'))
    for cut in _steps(0.70, 0.95, 0.05):
        pts.append(_labelled(lambda ctx, c=cut: cr.sim_moleculeace(ctx, cutoff=c),
                             f'moleculeace_any@{cut:.2f}'))
    for pct in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        pts.append(_labelled(lambda ctx, p=pct: cr.sim_ecfp4_top_pairs(ctx, pct=p),
                             f'ecfp4_top@{pct}pct'))
    for k in (1, 3, 5, 10, 20, 50):
        pts.append(_labelled(lambda ctx, kk=k: cr.sim_ecfp4_knn(ctx, k=kk),
                             f'ecfp4_knn@k{k}'))
    return pts


def similarity_grid_reps():
    """The axes that need representations loaded. Left out without --reps."""
    pts = []
    for metric in cr.CONCAT_METRICS:
        for pct in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
            pts.append(_labelled(
                lambda ctx, m=metric, p=pct: cr.sim_top_pairs_all_features(
                    ctx, pct=p, metric=m),
                f'all_features_{metric}_top@{pct}pct'))
    return pts


def similarity_grid_sequences():
    """The sequence axes. s01 and s02 sweep 1, 2, 3 and 5 edits already; this
    widens the edit count and adds the percentile form."""
    pts = []
    for e in (1, 2, 3, 4, 5, 8, 12):
        pts.append(_labelled(
            lambda ctx, ee=e: cr.sim_sequence_edit(ctx, max_edits=ee),
            f'sequence_edit@{e}'))
        pts.append(_labelled(
            lambda ctx, ee=e: cr.sim_sequence_edit(ctx, max_edits=ee, scope='cdr'),
            f'sequence_cdr_edit@{e}'))
    for pct in (0.5, 1.0, 2.0, 5.0, 10.0):
        pts.append(_labelled(
            lambda ctx, p=pct: cr.sim_sequence_edit_percentile(ctx, pct=p),
            f'sequence_edit_top@{pct}pct'))
    return pts


def gap_grid():
    """The gap axis. One entry is one cutoff of one gap criterion."""
    pts = []
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        pts.append(_labelled(
            lambda ctx, similar=None, kk=k: cr.gap_mad_multiple(
                ctx, k=kk, similar=similar),
            f'mad@{k}x'))
    for u in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        pts.append(_labelled(
            lambda ctx, similar=None, uu=u: cr.gap_absolute(
                ctx, cutoff=uu, similar=similar),
            f'absolute@{u}'))
    for t in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        pts.append(_labelled(
            lambda ctx, similar=None, tt=t: cr.gap_solved_to_target(
                ctx, target_pct=tt, similar=similar),
            f'solved@{t}pct'))
    for q in (75.0, 80.0, 90.0, 95.0, 99.0):
        pts.append(_labelled(
            lambda ctx, similar=None, qq=q: cr.gap_percentile_of_differences(
                ctx, pct=qq, similar=similar),
            f'percentile@{q}'))
    for n in (1.0, 2.0, 3.0):
        pts.append(_labelled(
            lambda ctx, similar=None, nn=n: cr.gap_mean_plus_sd(
                ctx, n_sd=nn, similar=similar),
            f'mean_plus@{n}sd'))
    pts.append(_labelled(
        lambda ctx, similar=None: cr.gap_noise_floor(ctx, similar=similar),
        'kramer_noise_floor'))
    return pts


def sweep(ctx, meta, repeat_spread=None, with_reps=False, verbose=True):
    """Every similarity point against every gap point.

    `repeat_spread` is that dataset's own measured repeatability in label
    units, where it has repeat measurements (row 3 of the handoff). Where it is
    None the gap is reported against Kramer's 0.54 pKi only, and the row says
    so rather than borrowing it silently.
    """
    sims = similarity_grid()
    if ctx.smiles is None:
        sims = []
    if with_reps and ctx.reps:
        sims += similarity_grid_reps()
    if ctx.sequences is not None:
        sims += similarity_grid_sequences()
    gaps = gap_grid()

    rows, errors = [], {}
    floor = cr.KRAMER_NOISE_FLOOR_PKI
    if verbose:
        print(f'  sweep: {len(sims)} similarity points x {len(gaps)} gap '
              f'points = {len(sims) * len(gaps)} rows', flush=True)

    for si, s in enumerate(sims):
        t0 = time.time()
        for g in gaps:
            try:
                r = cr.cliff_sets(ctx, s, g)
            except Exception as exc:
                errors[f'{s.label}__{g.label}'] = f'{type(exc).__name__}: {exc}'
                continue
            r.pop('pair_mask')
            r.pop('molecule_mask')
            cut = r['gap_info'].get('cutoff_label_units')
            r['gap_cutoff_label_units'] = cut
            r['gap_over_kramer_floor'] = (float(cut / floor) if cut else None)
            r['gap_over_own_repeat_spread'] = (
                float(cut / repeat_spread)
                if (cut and repeat_spread) else None)
            r['own_repeat_spread_label_units'] = repeat_spread
            r['similarity_cutoff_reached'] = r['similarity_info'].get(
                'cutoff_reached')
            r.pop('similarity_info')
            r.pop('gap_info')
            rows.append(r)
        if verbose:
            print(f'    [{si + 1}/{len(sims)}] {s.label:34s} '
                  f'{time.time() - t0:5.1f}s for {len(gaps)} gap points',
                  flush=True)

    return {'meta': meta, 'rows': rows, 'errors': errors,
            'n_similarity_points': len(sims), 'n_gap_points': len(gaps),
            'own_repeat_spread_label_units': repeat_spread,
            'kramer_noise_floor_pki': floor,
            'what_it_is': (
                'every similarity cutoff against every gap cutoff, per '
                'dataset. HANDOFF_PAPER3_CLIFFS.md section D. A local run of '
                'this is not a result; it picks which definitions to carry.')}


def print_sweep(result):
    """The three readings the sweep is for, per similarity criterion."""
    rows = result['rows']
    meta = result['meta']
    print(f"\n  sweep over {result['n_similarity_points']} similarity points "
          f"and {result['n_gap_points']} gap points, "
          f"{len(rows)} rows computed")

    # Where a criterion saturates and where it empties, read along the
    # similarity axis at one fixed gap so the two axes are not confounded.
    fixed = 'mad@2.0x'
    by_crit = {}
    for r in rows:
        if r['gap'] != fixed:
            continue
        crit = str(r['similarity']).split('@')[0]
        by_crit.setdefault(crit, []).append(r)
    print(f'\n  along the similarity axis at a fixed gap of {fixed}, '
          f'cliff molecules of {meta["n_rows"]} rows:')
    def _cut_value(r):
        """The cutoff as a number, so 10 per cent does not sort between 0.25
        and 2 per cent the way its text does."""
        tail = str(r['similarity']).split('@')[1]
        digits = ''.join(c for c in tail if c.isdigit() or c == '.')
        try:
            return float(digits)
        except ValueError:
            return 0.0

    for crit, rs in sorted(by_crit.items()):
        rs = sorted(rs, key=_cut_value)
        pts = ' '.join(f"{str(r['similarity']).split('@')[1]}:"
                       f"{r['pct_cliff_molecules']:.0f}%" for r in rs)
        print(f'    {crit:28s} {pts}')

    # Where the gap crosses the dataset's own repeatability, or Kramer's.
    ref = result.get('own_repeat_spread_label_units')
    which = ('this dataset\'s own repeat spread'
             if ref else f"Kramer's {result['kramer_noise_floor_pki']} pKi "
                         '(BORROWED: the right number for public Ki data, not '
                         'necessarily for this assay)')
    below = [r for r in rows
             if (r['gap_over_own_repeat_spread'] if ref
                 else r['gap_over_kramer_floor']) is not None
             and (r['gap_over_own_repeat_spread'] if ref
                  else r['gap_over_kramer_floor']) < 1.0]
    print(f'\n  rows whose gap is below {which}: {len(below)} of {len(rows)}')
    worst = sorted(below, key=lambda r: -(r['n_cliff_molecules']))[:5]
    for r in worst:
        ratio = (r['gap_over_own_repeat_spread'] if ref
                 else r['gap_over_kramer_floor'])
        print(f"    {str(r['similarity']):30s} {str(r['gap']):18s} "
              f"gap {r['gap_cutoff_label_units']:.3f} = {ratio:.2f}x, "
              f"{r['n_cliff_molecules']} cliff molecules "
              f"({r['pct_cliff_molecules']:.1f}%)")

    if result['errors']:
        print(f"\n  did not compute: {len(result['errors'])} rows")
        for k, v in list(result['errors'].items())[:5]:
            print(f'    {k}: {v}')


def environment_state():
    """What of table 1a and 1e is reachable in this environment at all."""
    return {'matched_molecular_pairs': cr.mmp_state(),
            'moleculeace_labels': cr.moleculeace_state(),
            'rogi_package': _rogi_installed(),
            'acnet': {'downloaded': False,
                      'note': 'Zhang et al. 2023, more than 400,000 matched '
                              'pairs over 190 targets; not in this repo'}}


def _rogi_installed():
    try:
        import rogi  # noqa: F401
        return {'installed': True}
    except Exception:
        return {'installed': False,
                'note': 'coleygroup/rogi packages ROGI, RMODI and MODI; '
                        'cliff_rules.rogi is an independent implementation '
                        'whose dispersion step is unchecked against it'}


def print_summary(result):
    meta = result['meta']
    print(f"\n{meta['dataset']}: {meta['n_rows']} rows, "
          f"{meta['n_unique_smiles']} unique SMILES, "
          f"label median absolute deviation "
          f"{meta['label_median_absolute_deviation']:.4f}")
    sp = meta.get('pairwise_label_difference_spread') or {}
    if sp:
        print(f"  pairwise label difference: median {sp['median']:.3f}, "
              f"95th percentile {sp['p95']:.3f}, max {sp['max']:.3f} "
              f"(label units)")

    print('\n  named rules, cliff molecules of '
          f"{meta['n_rows']} rows:")
    for name, r in result['named_rules'].items():
        if 'unavailable' in r:
            print(f"    {name:24s} not run: {r['unavailable']}")
            continue
        floor = r['pct_cliff_pairs_inside_noise_floor']
        floor_s = 'n/a' if floor is None else f'{floor:.1f}%'
        ratio = r.get('gap_cutoff_over_noise_floor')
        ratio_s = 'n/a' if ratio is None else f'{ratio:.2f}x'
        print(f"    {name:24s} {r['n_cliff_molecules']:>6d} molecules "
              f"({r['pct_cliff_molecules']:5.1f}%), "
              f"{r['n_cliff_pairs']:>8d} cliff pairs of "
              f"{r['n_similar_pairs']:>8d} similar; gap {ratio_s} the 0.54 "
              f"noise floor, cliff pairs inside it {floor_s}")

    nagree = result.get('named_rule_agreement') or []
    nagree = sorted((p for p in nagree if p['jaccard'] is not None),
                    key=lambda p: p['jaccard'])
    if nagree:
        print('\n  named rules against each other, molecules, worst agreement '
              'first:')
        for p in nagree:
            print(f"    {p['jaccard']:.3f}  both {p['both']:>6d}, only "
                  f"{p['first']} {p['only_first']:>6d}, only {p['second']} "
                  f"{p['only_second']:>6d}")

    if result['solved_gap']:
        print('\n  the solved gap, achieved share against the 15 per cent it '
              'aims at:')
        for s in result['solved_gap']:
            got = s['achieved_pct']
            got_s = 'n/a' if got is None else f'{got:5.1f}%'
            print(f"    {s['similarity']:30s} achieved {got_s} "
                  f"with {s['n_similar_pairs_available']:>8d} similar pairs "
                  f"available, reached={s['reached_target']}")

    worst = sorted((p for p in result['agreement'] if p['jaccard'] is not None),
                   key=lambda p: p['jaccard'])[:10]
    if worst:
        print('\n  the ten rule pairs that agree least, as molecules:')
        for p in worst:
            print(f"    {p['jaccard']:.3f}  both {p['both']:>6d}, only first "
                  f"{p['only_first']:>6d}, only second {p['only_second']:>6d}"
                  f"   {p['first']} | {p['second']}")

    if result['errors']:
        print('\n  did not compute:')
        for k, v in result['errors'].items():
            print(f'    {k}: {v}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*')
    ap.add_argument('--reps', action='store_true',
                    help='also load chemberta, molformer and chemeleon, so the '
                         'embedding-cosine and all-features criteria can run')
    ap.add_argument('--output', default=OUT_DEFAULT)
    ap.add_argument('--list', action='store_true',
                    help='print the criteria and named rules, load nothing')
    ap.add_argument('--sweep', action='store_true',
                    help='the threshold sweep of handoff section D instead of '
                         'the census: every similarity cutoff against every '
                         'gap cutoff')
    ap.add_argument('--census', action='store_true',
                    help='run the census as well when --sweep is given')
    args = ap.parse_args()

    if args.list:
        print('similarity criteria:')
        for k, v in cr.SIMILARITY_NEEDS.items():
            print(f'  {k:30s} needs {v}')
        print('\ngap criteria:')
        for k in cr.GAP_CRITERIA:
            print(f'  {k}')
        print('\nnamed rules:')
        for k, (s, g) in cr.RULES.items():
            print(f'  {k:24s} {s} + {g}')
        return 0

    if not args.datasets:
        print('name at least one dataset, or pass --list')
        return 2

    os.makedirs(args.output, exist_ok=True)
    for name in args.datasets:
        t0 = time.time()
        print(f'\n=== {name} ===', flush=True)
        ctx, meta = build_context(name, with_reps=args.reps)

        if args.sweep:
            spread, spread_info = own_repeat_spread(name, ctx.labels)
            meta['own_repeat_spread'] = spread_info
            if spread is None:
                print(f"  this dataset's own repeat spread: NONE "
                      f"({spread_info.get('reason')}). Every gap is reported "
                      f"against Kramer's 0.54 pKi, which is borrowed.",
                      flush=True)
            else:
                print(f"  this dataset's own repeat spread: median "
                      f"{spread:.4f} label units over "
                      f"{spread_info['n_groups_with_disagreeing_repeats']} "
                      f"groups whose repeats disagree, "
                      f"{spread_info['n_rows_in_such_a_group']} of "
                      f"{spread_info['n_rows_total']} rows", flush=True)
            sw = sweep(ctx, meta, repeat_spread=spread,
                       with_reps=args.reps)
            sw['environment'] = environment_state()
            sw['meta']['seconds'] = round(time.time() - t0, 1)
            print_sweep(sw)
            path = os.path.join(args.output, f'{name}__sweep.json')
            with open(path, 'w') as fh:
                json.dump(sw, fh, indent=2, default=float)
            print(f"\n  written {path} ({sw['meta']['seconds']}s, "
                  f"{len(sw['rows'])} rows)", flush=True)
            if not args.census:
                continue
            t0 = time.time()

        result, masks = census(ctx, meta)
        print('\n  continuous scores (table 1c), on ECFP4 Tanimoto:',
              flush=True)
        cont, arrays = continuous_scores(ctx)
        result['continuous_scores'] = cont
        result['environment'] = environment_state()
        masks = dict(masks)
        masks.update({f'score__{k}': v for k, v in arrays.items()})
        result['meta']['seconds'] = round(time.time() - t0, 1)
        print_summary(result)
        path = os.path.join(args.output, f'{name}.json')
        with open(path, 'w') as fh:
            json.dump(result, fh, indent=2, default=float)
        np.savez_compressed(
            os.path.join(args.output, f'{name}__molecule_masks.npz'),
            **{k: v for k, v in masks.items()})
        print(f'\n  written {path} ({result["meta"]["seconds"]}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
