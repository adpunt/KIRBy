#!/usr/bin/env python3
"""n16 read-out — which features change rank when the labels get noisier.

Reads every n16_importance_under_noise/<dataset>__<condition>.json and prints four
things. It does not open the npz files beside them.

1. THE CONTROLS. Two of them.
   (a) Level 0.0 must have delivered no noise at all: the delivered amount the injector
       recorded at level 0.0 must be exactly 0.0, in every fold of every noise draw.
       This one can fail, and if it does the reference row is not the clean labels
       and nothing below can be read.
   (b) Level 0.0 against itself must give Spearman 1.0 and keep every top-k feature.
       n16 compares the level-0.0 vector with the very same array here
       (_summarise_replicate reads one key twice), so this only proves the comparison
       code runs. It cannot detect a wrong reference row. A cell whose scorer ranked
       fewer than 3 features has no Spearman at all and is counted apart, not failed.

2. HOW MANY FEATURES CARRY ANY IMPORTANCE, per scoring method per level, beside each
   representation's own width and its pre-screen ceiling. A representation wider than
   2000 features has a block that is structurally zero at every level
   (hybrid.py:4359, :4379-4383), so the count is read against the ceiling, not the
   width.

3. HOW FAR THE RANKING MOVED from level 0.0, per scoring method, per representation,
   per level: the median, minimum and maximum across noise draws of the Spearman over
   the features either level scored non-zero, and how many of the clean top set
   survive. One row is one representation at one level; representations are never
   pooled. The all-features Spearman is printed beside it, and (in the experiment's
   own JSON, not here) the top-k retained fraction over the uncapped set. Both are
   inflated, because lightgbm_gain leaves most features at exactly 0.0 and np.argsort
   orders tied zeros by index. Measured on buchwald_hartwig, lightgbm_gain scores 39
   of diff_maccs's 167 features, so an uncapped top-50 overlap is 1.00 at every level
   while the scored ranking moves from Spearman 0.980 to 0.920. The 'top set' column
   says how many features the reported set holds.

   The Spearman over the features either level scored is pulled the other way. A
   feature scored at one level and not at the other enters with a 0.0 on one side
   (f30_analyse.py:70), so this Spearman falls, and can go below zero, when many
   features switch between scored and unscored, even while the features both levels
   scored keep their order. It measures two things at once: whether the same
   features get any score, and how the shared ones are ordered. Measured 2026-09-28
   on herg_ki, gaussian, fold 0, from the saved vectors: grover at level 0.15 reads
   -0.132 here, and +0.557 over the 255 features both levels scored; pdv reads 0.793
   and 0.774. So a sparse representation and a dense one are not on one scale in
   this column. The Spearman over the features both levels scored is not computed
   by n16 or here.

4. WHETHER THE AMOUNT OF MOVEMENT DEPENDS ON THE NOISE DRAW. Each level is compared with
   level 0.0 on the same training rows with the same scorer seed, so the labels are the
   only thing that differs and any drift above zero is caused by the noise. With two or
   more draws, the range of the drift across draws says how much that amount depends on
   which draw was made. It is not a test of whether the ranking moved, and it is not a
   reference for how much a ranking moves for reasons other than noise.

Datasets are never pooled. A Spearman measured on rmse-scored logd and one measured on
spearman-scored flab_thermo_tresanco are not the same quantity, and each table below is
one dataset on its own.

Usage:
    python tests/experiments/noise/n16_analyse.py
    python tests/experiments/noise/n16_analyse.py --dataset logd
    python tests/experiments/noise/n16_analyse.py --input tests/results/noise
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'tests'))

import numpy as np

DEFAULT_INPUT = os.path.join(
    os.environ.get('KIRBY_RESULTS_DIR', 'tests/results'), 'noise')
EXP_DIR = 'n16_importance_under_noise'

# A cell whose non-zero features number fewer than this on BOTH sides cannot carry a
# rank correlation: _pair_stats itself returns nan below 3 (f30_analyse.py:70-73).
# Cells below it are counted and NAMED in the output, never silently dropped.
MIN_SCORED_FOR_RANK = 3

# The control tolerance. Level 0.0 against itself is the same array, so the Spearman
# is 1.0 exactly; anything below this means the reference row was not the same rows.
CONTROL_RTOL = 1e-9


def load(input_dir, dataset=None):
    """Every n16 result JSON under input_dir, newest schema only."""
    d = Path(input_dir) / EXP_DIR
    if not d.is_dir():
        print(f"  (no {d})")
        return []
    # Two filename shapes, both real. A standalone run writes
    # <dataset>__<condition>.json; noise_helpers.run_all writes <dataset>.json and
    # keeps the condition inside the file (noise_helpers.py:494-495).
    if dataset:
        paths = sorted(set(d.glob(f'{dataset}__*.json')) | set(d.glob(f'{dataset}.json')))
    else:
        paths = sorted(d.glob('*.json'))
    out = []
    for f in paths:
        try:
            p = json.loads(f.read_text())
        except Exception as e:
            print(f'  unreadable: {f.name} ({type(e).__name__}: {e})')
            continue
        if p.get('skipped'):
            print(f"  {f.stem}: SKIPPED — {p.get('reason')} "
                  f"(policy={p.get('policy')})")
            continue
        if 'swept_conditions' in p and 'per_replicate' not in p:
            # The sweep's index (noise_helpers.sweep_conditions), not a result.
            print(f"  {f.stem}: sweep index, names "
                  f"{len(p.get('conditions_written') or {})} condition files; "
                  f"skipped conditions: "
                  f"{sorted((p.get('conditions_skipped_and_why') or {}))}")
            continue
        if 'per_replicate' not in p:
            print(f"  {f.stem}: no per_replicate block — older schema, not read")
            continue
        out.append((f, p))
    return out


def _levels_of(p):
    return [float(x) for x in p['levels']]


def _finite(x):
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def control_clean_reference(p):
    """Level 0.0 delivered no noise: the recorded delivered amount is exactly 0.0.

    This is the control that can fail. n16 compares every level with level 0.0, so
    a level 0.0 that moved any label would make every drift below a comparison of
    two noisy rankings. The delivered amount is the root mean square of the error
    added to that fold's training labels (kirby/noise_spec.py:588), so exactly 0.0
    means not one training label moved.

    Every fold is checked. `dose_check` holds the FIRST fold only
    (n16_importance_under_noise.py:239-240); `noise_provenance` holds one row per
    fold per noise draw (:234), and is read when present. Returns
    (n_rows_checked, failures, n_rows_unrecorded), one row per fold per draw.
    """
    bad, checked, unrecorded = [], 0, 0
    prov = p.get('noise_provenance') or {}
    rows = []
    for spec_id, per_fold in prov.items():
        for fold, r in enumerate(per_fold or []):
            if isinstance(r, dict) and _finite(r.get('noise_level')) \
                    and float(r['noise_level']) == 0.0:
                rows.append((f'{spec_id} fold {fold}', r))
    if not rows:
        rows = [(f"replicate {r.get('replicate')} fold 0", r)
                for r in (p.get('dose_check') or [])
                if float(r.get('level', -1)) == 0.0]
    for where, r in rows:
        got = r.get('realised_dose_label_units')
        if not _finite(got):
            unrecorded += 1
            continue
        checked += 1
        if float(got) != 0.0:
            bad.append((where, f'delivered {float(got):.9g}'))
    return checked, bad, unrecorded


def control(p):
    """Level 0.0 against itself: Spearman 1.0 and the whole top-k retained.

    n16 builds this row from ONE stored array compared with itself
    (_summarise_replicate reads the level-0.0 key as both sides), so it proves the
    comparison code runs and nothing more. A cell whose scorer ranked fewer than
    MIN_SCORED_FOR_RANK features has no Spearman (_pair_stats returns nan) and is
    counted as thin, not as a failure.

    Returns (n_cells_checked, failures, thin).
    """
    bad, thin, checked = [], [], 0
    for rkey, block in p['per_replicate'].items():
        for method, reps in block.get('base', {}).items():
            for rname, cells in reps.items():
                c = cells.get('0')
                if c is None:
                    checked += 1
                    bad.append((rkey, method, rname, 'no level-0.0 row'))
                    continue
                rho = c['mean_spearman_both_scored']
                kept = c['mean_topk_retained_fraction']
                if (c.get('mean_n_scored_features', 0) < MIN_SCORED_FOR_RANK
                        or not _finite(rho)):
                    thin.append((rkey, method, rname))
                    continue
                checked += 1
                if not (abs(rho - 1.0) <= CONTROL_RTOL
                        and abs(kept - 1.0) <= CONTROL_RTOL):
                    bad.append((rkey, method, rname,
                                f'spearman={rho:.9f} retained={kept:.9f}'))
    return checked, bad, thin


def report_counts(p):
    """How many features carry any importance, per method per level."""
    levels = _levels_of(p)
    widths = p.get('rep_widths', {})
    ceilings = p.get('rep_prescreen_width', {})
    print("\n  FEATURES CARRYING ANY IMPORTANCE")
    print("  one row is one representation; the count is of features with a "
          "non-zero score, out of the pre-screen ceiling")
    for method in p['importance_methods']:
        got = {}
        for block in p['per_replicate'].values():
            for rname, cells in block.get('base', {}).get(method, {}).items():
                for lv in levels:
                    c = cells.get(f'{lv:g}')
                    if c is not None:
                        got.setdefault(rname, {}).setdefault(lv, []).append(
                            c['mean_n_scored_features'])
        if not got:
            continue
        print(f"\n  [{method}]")
        head = "    " + f"{'representation':<22}{'width':>7}{'ceiling':>9}"
        head += "".join(f"{f'lv {lv:g}':>10}" for lv in levels)
        print(head)
        for rname in sorted(got):
            row = f"    {rname:<22}{widths.get(rname, 0):>7}" \
                  f"{ceilings.get(rname, 0):>9}"
            for lv in levels:
                vals = got[rname].get(lv)
                row += f"{(np.mean(vals) if vals else float('nan')):>10.0f}"
            print(row)


def _rep_names(p, method):
    names = set()
    for block in p['per_replicate'].values():
        names |= set(block.get('base', {}).get(method, {}))
    order = p.get('rep_order') or []
    return [r for r in order if r in names] + sorted(names - set(order))


def report_drift(p):
    """How far the ranking moved from level 0.0, per method, per representation,
    per level. The spread is across noise draws only; representations are never
    pooled, because a 167-feature fingerprint and a 2048-feature one ranked by the
    same scorer are not the same quantity."""
    levels = [lv for lv in _levels_of(p) if lv > 0.0]
    print("\n  RANK AGREEMENT WITH LEVEL 0.0")
    print("  one row is one representation at one level. rho = Spearman over the "
          "features either")
    print("  level scored non-zero, as median, min and max across noise draws. "
          "kept = the fraction")
    print("  of the level-0.0 top set still in the top set at that level. "
          "'top set' is how many")
    print(f"  features that set holds: {p.get('top_k')} was asked for, capped at "
          f"how many features")
    print("  the level-0.0 scorer gave a non-zero score, so the level-0.0 set never "
          "contains a tied zero.")
    print("  A feature scored at only one of the two levels enters rho with a zero "
          "on the other side,")
    print("  so rho falls, and can go below zero, when many features switch between "
          "scored and")
    print("  unscored, even if the features both levels scored keep their order.")
    print(f"  'too thin' counts draws whose scorer ranked fewer than "
          f"{MIN_SCORED_FOR_RANK} features, or")
    print("  whose Spearman is undefined; they are left out of the row.")
    for method in p['importance_methods']:
        rows = []
        for rname in _rep_names(p, method):
            for lv in levels:
                rho, rho_all, kept, keff, thin = [], [], [], [], 0
                for block in p['per_replicate'].values():
                    c = (block.get('base', {}).get(method, {}).get(rname, {})
                         .get(f'{lv:g}'))
                    if c is None:
                        continue
                    if (c['mean_n_scored_features'] < MIN_SCORED_FOR_RANK
                            or not _finite(c['mean_spearman_both_scored'])):
                        thin += 1
                        continue
                    rho.append(c['mean_spearman_both_scored'])
                    rho_all.append(c['mean_spearman_all_features'])
                    # The EFFECTIVE top set: capped at how many features the
                    # scorer actually ranked. The plain top-k includes tied zeros
                    # ordered by index, identically on both sides, and comes back
                    # 1.00 whenever the scorer left more than (width - k) features
                    # at zero.
                    kept.append(c['mean_topk_effective_retained_fraction'])
                    keff.append(c['mean_topk_effective_size'])
                if rho:
                    rows.append((rname, lv, len(rho), float(np.median(rho)),
                                 float(np.min(rho)), float(np.max(rho)),
                                 float(np.nanmedian(rho_all)),
                                 float(np.nanmedian(kept)),
                                 float(np.nanmedian(keff)), thin))
                elif thin:
                    rows.append((rname, lv, 0, float('nan'), float('nan'),
                                 float('nan'), float('nan'), float('nan'),
                                 float('nan'), thin))
        if not rows:
            continue
        print(f"\n  [{method}]")
        print(f"    {'representation':<22}{'level':>7}{'draws':>7}{'rho med':>10}"
              f"{'rho min':>10}{'rho max':>10}{'rho all':>10}{'kept':>8}"
              f"{'top set':>9}{'too thin':>10}")
        for rname, lv, ndraw, med, lo, hi, med_all, kept, keff, thin in rows:
            print(f"    {rname:<22}{lv:>7.2f}{ndraw:>7}{med:>10.4f}{lo:>10.4f}"
                  f"{hi:>10.4f}{med_all:>10.4f}{kept:>8.3f}{keff:>9.0f}{thin:>10}")


def report_replicate_spread(p):
    """How much the amount of movement depends on which noise draw was made."""
    reps = p['per_replicate']
    if len(reps) < 2:
        print("\n  DRAW-TO-DRAW RANGE: one noise draw only, so there is no range to "
              "print. Re-run with --replicates 2 or more to see whether the drift "
              "above depends on the draw.")
        return
    levels = [lv for lv in _levels_of(p) if lv > 0.0]
    print("\n  DRIFT AND ITS RANGE ACROSS NOISE DRAWS")
    print("  one row is one representation at one level. drift = 1 - rho against "
          "level 0.0,")
    print("  median across draws. range = max minus min of that drift across "
          f"the {len(reps)} draws.")
    print("  Each draw is compared on the same rows with the same scorer seed, so "
          "any drift above")
    print("  zero is caused by the noise. The range says whether the AMOUNT depends "
          "on the draw;")
    print("  it is not a test of whether the ranking moved.")
    for method in p['importance_methods']:
        rows = []
        for rname in _rep_names(p, method):
            for lv in levels:
                per_draw = []
                for block in reps.values():
                    c = (block.get('base', {}).get(method, {}).get(rname, {})
                         .get(f'{lv:g}'))
                    if c is None or not _finite(c['mean_spearman_both_scored']):
                        continue
                    if c['mean_n_scored_features'] < MIN_SCORED_FOR_RANK:
                        continue
                    per_draw.append(1.0 - float(c['mean_spearman_both_scored']))
                if len(per_draw) >= 2:
                    rows.append((rname, lv, float(np.median(per_draw)),
                                 float(np.max(per_draw) - np.min(per_draw)),
                                 len(per_draw)))
        if not rows:
            continue
        print(f"\n  [{method}]")
        print(f"    {'representation':<22}{'level':>7}{'drift':>10}{'range':>10}"
              f"{'draws':>7}  reading")
        for rname, lv, drift, spread, ndraw in rows:
            verdict = ('similar in every draw' if drift > spread
                       else 'depends on the draw')
            print(f"    {rname:<22}{lv:>7.2f}{drift:>10.4f}{spread:>10.4f}"
                  f"{ndraw:>7}  {verdict}")


def report_hybrid_ranking(p):
    """How the hybrid's OWN features rank, level by level — the base table's twin.

    The correlation is over the features BOTH levels selected, because two levels
    select different features and the two vectors cannot be lined up elementwise.
    'shared' says how many features that is; a high correlation over a handful of
    shared features says more about which handful survived than about the ranking, so
    the two are always printed together.
    """
    blocks = p.get('summary', {}).get('hybrid_importance') or {}
    if not blocks:
        return
    print("\n  THE HYBRID'S OWN FEATURE RANKING, AGAINST LEVEL 0.0")
    print("  rho over the features selected at BOTH levels; shared = how many that "
          "is; scored =")
    print("  how many of the hybrid's features carry any importance at that level")
    for method, per_level in blocks.items():
        print(f"\n  [{method}]")
        print(f"    {'level':>7}{'cells':>7}{'rho med':>10}{'rho min':>10}"
              f"{'rho max':>10}{'shared':>9}{'scored':>9}")
        for lv in _levels_of(p):
            c = per_level.get(f'{lv:g}')
            if c is None:
                continue
            if 'median_spearman_vs_level_zero' in c:
                print(f"    {lv:>7.2f}{c['n_cells']:>7}"
                      f"{c['median_spearman_vs_level_zero']:>10.4f}"
                      f"{c['min_spearman_vs_level_zero']:>10.4f}"
                      f"{c['max_spearman_vs_level_zero']:>10.4f}"
                      f"{c['median_n_features_selected_at_both_levels']:>9.0f}"
                      f"{c['median_n_scored_features']:>9.0f}")
            else:
                print(f"    {lv:>7.2f}{c['n_cells']:>7}"
                      f"{'too few shared':>30}"
                      f"{c['median_n_features_selected_at_both_levels']:>9.0f}"
                      f"{c['median_n_scored_features']:>9.0f}")


def report_hybrid(p):
    """Which source features the recipe takes, level by level."""
    hyb = p.get('summary', {}).get('hybrid') or {}
    if not hyb:
        return
    print("\n  THE FEATURES THE RECIPE TAKES")
    print(f"  recipe held fixed at every level: {p.get('recipe_lineage')}")
    bp = p.get('blueprint') or {}
    print(f"  budget={bp.get('budget')} allocation={bp.get('allocation_method')} "
          f"importance={bp.get('importance_method')} "
          f"evaluator={bp.get('internal_evaluator')}")
    # One table per representation: n16's summary is keyed representation
    # first since 2026-09-28, and never pools across representations.
    for rname, per_level in hyb.items():
        if not isinstance(per_level, dict):
            continue
        print(f"\n  [{rname}]")
        print(f"    {'level':>7}{'replicates':>11}{'features taken':>17}"
              f"{'kept from lv 0.0':>19}")
        for lv in _levels_of(p):
            c = per_level.get(f'{lv:g}')
            if not isinstance(c, dict):
                continue
            print(f"    {lv:>7.2f}{c.get('n_replicates', c.get('n_cells')):>11}"
                  f"{c['median_n_selected_features']:>17.0f}"
                  f"{c['median_retained_fraction_of_level_zero']:>19.3f}")


def report_dose(p):
    """What the injector actually delivered against what was asked for."""
    rows = p.get('dose_check') or []
    if not rows:
        return
    print("\n  DELIVERED AMOUNT AGAINST THE REQUESTED AMOUNT")
    print(f"  both in {p.get('dose_units')} label units; noiseInject's own "
          f"DoseWarning cannot reach")
    print("  a log inside this process (find_best_hybrid.py:74), so it is read here")
    print(f"    {'level':>7}{'replicate':>11}{'asked':>12}{'delivered':>12}"
          f"{'off by':>10}")
    for r in rows:
        asked, got = r.get('target_dose_label_units'), r.get(
            'realised_dose_label_units')
        if asked in (None, 0) or got is None:
            off = float('nan')
        else:
            off = 100.0 * (got / asked - 1.0)
        print(f"    {r['level']:>7.2f}{r['replicate']:>11}"
              f"{(asked if asked is not None else float('nan')):>12.6f}"
              f"{(got if got is not None else float('nan')):>12.6f}"
              f"{off:>9.2f}%")


def report(path, p):
    print("\n" + "=" * 78)
    print(f"  {p['dataset']}  —  condition {p['noise_condition']}  "
          f"({p.get('domain')}, n={p.get('n_samples')})")
    print(f"  metric {p.get('metric')} (lower_is_better={p.get('lower_is_better')}), "
          f"{p.get('n_folds')} folds, {p.get('n_replicates')} noise draws, "
          f"arm {p.get('arm')}")
    print("=" * 78)

    if p.get('pool_missing') or p.get('pool_extra'):
        print(f"  POOL DIFFERS from the signed-off list: "
              f"missing={p['pool_missing']} extra={p['pool_extra']}")

    n0, bad0, un0 = control_clean_reference(p)
    if bad0:
        print(f"\n  CONTROL FAILED: level 0.0 delivered noise in {len(bad0)} of "
              f"{n0} folds across the noise draws. The reference row is not the "
              f"clean labels, so no drift below can be read.")
        for where, msg in bad0[:10]:
            print(f"    {where}: {msg}")
    elif n0:
        print(f"\n  CONTROL: level 0.0 delivered exactly 0.0 label units in all "
              f"{n0} folds across the noise draws, so the reference row is the "
              f"clean labels.")
    else:
        print("\n  CONTROL: this file records no delivered amount at level 0.0, so "
              "whether the reference row is clean is NOT checked.")
    if un0:
        print(f"    {un0} level-0.0 folds carry no delivered amount.")

    checked, bad, thin = control(p)
    if not checked and not thin:
        print("  SELF-COMPARISON: no base-arm cells, so nothing to check. The "
              "hybrid tables below stand on their own counts.")
    elif bad:
        print(f"  SELF-COMPARISON FAILED on {len(bad)} cells. Level 0.0 against "
              f"itself must give Spearman 1.0 and retain every top-k feature, so "
              f"the comparison code is broken.")
        for row in bad[:10]:
            print(f"    replicate {row[0]} {row[1]}/{row[2]}: {row[3]}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")
    else:
        print(f"  SELF-COMPARISON: level 0.0 against itself gives Spearman 1.0 on "
              f"all {checked} cells. It compares one array with itself, so this "
              f"shows only that the comparison code runs.")
    if thin:
        print(f"    {len(thin)} cells ranked fewer than {MIN_SCORED_FOR_RANK} "
              f"features at level 0.0 and have no Spearman: "
              f"{sorted({f'{m}/{r}' for _, m, r in thin})}")

    report_dose(p)
    report_counts(p)
    report_drift(p)
    report_replicate_spread(p)
    report_hybrid_ranking(p)
    report_hybrid(p)

    errs = p.get('errors') or []
    if errs:
        print(f"\n  {len(errs)} cells failed and are absent from every table above:")
        seen = defaultdict(int)
        for e in errs:
            seen[(e.get('arm'), e.get('method'), e.get('error', '')[:70])] += 1
        for (a, m, msg), k in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    {a}/{m} x{k}: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', default=DEFAULT_INPUT,
                    help=f'results root holding {EXP_DIR}/ (default: {DEFAULT_INPUT})')
    ap.add_argument('--dataset', default=None,
                    help='one dataset only (default: every file found)')
    args = ap.parse_args()

    found = load(args.input, args.dataset)
    if not found:
        print(f"  nothing to read under {Path(args.input) / EXP_DIR}")
        # Non-zero, so a queue job pointed at the wrong directory is recorded
        # as failed. It exited 0 here, and the five queued n16_analyse jobs
        # read results/n16_analyse/<dataset>, where n16 writes nothing.
        return 1
    for path, p in found:
        report(path, p)
    print("\n" + "=" * 78)
    print(f"  {len(found)} result files read. The tables are never pooled across "
          f"datasets:")
    print("  a Spearman on an rmse-scored dataset and one on a spearman-scored "
          "dataset are")
    print("  not the same quantity.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
