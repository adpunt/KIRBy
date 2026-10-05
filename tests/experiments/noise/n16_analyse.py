#!/usr/bin/env python3
"""n16 read-out — which features change rank when the labels get noisier.

Reads every n16_importance_under_noise/<dataset>__<condition>.json and the npz files
beside them, and prints four things.

1. THE CONTROL. Level 0.0 against itself must give Spearman 1.0 and keep every one of
   the top-k features. If it does not, the run compared two different things and
   nothing below it can be read.

2. HOW MANY FEATURES CARRY ANY IMPORTANCE, per scoring method per level, beside each
   representation's own width and its pre-screen ceiling. A representation wider than
   2000 features has a block that is structurally zero at every level
   (hybrid.py:4359, :4379-4383), so the count is read against the ceiling, not the
   width.

3. HOW FAR THE RANKING MOVED from level 0.0, per scoring method per level: the median,
   minimum and maximum Spearman over the features either level scored non-zero, and
   how many of the clean top set survive. Two things are printed beside the honest
   figures and both are the less honest of their pair, because lightgbm_gain leaves
   most features at exactly 0.0 and np.argsort orders tied zeros by index: the
   all-features Spearman, and (in the experiment's own JSON, not here) the top-k
   retained fraction over the uncapped set. Measured on buchwald_hartwig,
   lightgbm_gain scores 39 of diff_maccs's 167 features, so an uncapped top-50
   overlap is 1.00 at every level while the scored ranking moves from Spearman 0.980
   to 0.920. The 'top set' column says how many features the reported set holds.

3b. THE FLOOR (added 2026-10-05): level 0.0 scored again with another scorer seed,
   printed as the 'floor' row of each table. A level whose drift is no larger than
   the floor has not moved beyond the scorer's own randomness.

4. WHETHER THE MOVEMENT CLEARS THE NOISE DRAW. With two or more replicates, the spread
   between replicates at one level is what any level-to-level movement has to beat. A
   level whose drift from 0.0 sits inside the replicate spread has not moved.

PARTIAL RUNS ARE NAMED (2026-10-05, dirty-run bug "n16_analyse read partial runs
without saying so"). Every file prints how many of its folds and draws ran, and a
dataset missing any of the six decided damage kinds is listed with the missing ones.

Reads only files written by the 2026-10-05 alignment (result_schema
n16_chat5_2026_10_05): per fold, each decided model's 2 base representations and 2
hybrids. Older files held every representation and one rf recipe and are skipped
with a note.

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

# The schema the experiment writes since the 2026-10-05 alignment
# (n16_importance_under_noise.RESULT_SCHEMA), and the six decided regression
# damage kinds (noise_helpers.DECIDED_REGRESSION_CONDITIONS). Copied, not
# imported, so this reader pulls in no torch.
RESULT_SCHEMA = 'n16_chat5_2026_10_05'
DECIDED_CONDITIONS = ('gaussian', 'outlier_p05', 'grouped_shifted', 'censoring',
                      'shift_only_p10', 'sparse_dprime_up_p10')

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
        if 'swept_conditions' in p:
            # noise_helpers.sweep_conditions' index of the condition files, not
            # a result.
            continue
        if p.get('skipped'):
            print(f"  {f.stem}: SKIPPED — {p.get('reason')} "
                  f"(policy={p.get('policy')})")
            continue
        if p.get('result_schema') != RESULT_SCHEMA:
            print(f"  {f.stem}: written before the 2026-10-05 alignment "
                  f"(schema {p.get('result_schema')!r}) — not read")
            continue
        out.append((f, p))
    return out


def _levels_of(p):
    return [float(x) for x in p['levels']]


def _m(spread):
    """The mean out of a fold_spread dict, or nan."""
    if isinstance(spread, dict) and spread.get('mean') is not None:
        return float(spread['mean'])
    return float('nan')


def completeness(p):
    """How much of the run this file holds: folds, draws, and the flag."""
    folds = p.get('folds_run') or []
    n_folds = p.get('n_folds')
    draws = len(p.get('per_replicate') or {})
    n_draws = p.get('n_replicates')
    whole = bool(p.get('complete')) and len(folds) == n_folds and draws == n_draws
    return whole, (f"folds run {folds} of {n_folds}, draws {draws} of "
                   f"{n_draws}")


def missing_conditions(found):
    """{dataset: [decided conditions with no readable file]}."""
    have = defaultdict(set)
    for _f, p in found:
        have[p['dataset']].add(p.get('noise_condition'))
    return {ds: [c for c in DECIDED_CONDITIONS if c not in conds]
            for ds, conds in have.items()
            if any(c not in conds for c in DECIDED_CONDITIONS)}


def control(p):
    """Level 0.0 against itself: Spearman 1.0 and the whole top-k retained.

    Returns (n_cells_checked, failures). The count is returned so a run with no
    base-arm cells reports "nothing to check" rather than passing vacuously — a
    control that cannot fail is not a control.
    """
    bad, checked = [], 0
    for rkey, block in p['per_replicate'].items():
        for method, reps in block.get('base', {}).items():
            for rname, cells in reps.items():
                checked += 1
                c = cells.get('0')
                if c is None:
                    bad.append((rkey, method, rname, 'no level-0.0 row'))
                    continue
                rho = _m(c['spearman_either_scored'])
                kept = _m(c['topk_retained_fraction'])
                if not (abs(rho - 1.0) <= CONTROL_RTOL
                        and abs(kept - 1.0) <= CONTROL_RTOL):
                    bad.append((rkey, method, rname,
                                f'spearman={rho:.9f} retained={kept:.9f}'))
    return checked, bad


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
                            _m(c['n_scored_features']))
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


def report_drift(p):
    """How far the ranking moved from level 0.0, per method per level."""
    levels = [lv for lv in _levels_of(p) if lv > 0.0] + ['floor_seed']
    print("\n  RANK AGREEMENT WITH LEVEL 0.0")
    print("  rho = Spearman over the features either level scored non-zero; "
          "kept = the fraction")
    print(f"  of the level-0.0 top set still in the top set at that level. "
          f"'top set' is how many")
    print(f"  features that set holds: {p.get('top_k')} was asked for, capped at "
          f"how many features")
    print("  the scorer gave a non-zero score, so the set never contains a tied "
          "zero.")
    for method in p['importance_methods']:
        rows = []
        for lv in levels:
            rho, rho_all, kept, keff, thin = [], [], [], [], 0
            key = lv if isinstance(lv, str) else f'{lv:g}'
            for block in p['per_replicate'].values():
                for rname, cells in block.get('base', {}).get(method, {}).items():
                    c = cells.get(key)
                    if c is None:
                        continue
                    if _m(c['n_scored_features']) < MIN_SCORED_FOR_RANK:
                        thin += 1
                        continue
                    rho.append(_m(c['spearman_either_scored']))
                    rho_all.append(_m(c['spearman_all_features']))
                    # The EFFECTIVE top set: capped at how many features the scorer
                    # actually ranked. The plain top-k includes tied zeros ordered by
                    # index, identically on both sides, and comes back 1.00 whenever
                    # the scorer left more than (width - k) features at zero.
                    kept.append(_m(c['topk_effective_retained_fraction']))
                    keff.append(_m(c['topk_effective_size']))
            if rho:
                rows.append((lv, len(rho), float(np.median(rho)),
                             float(np.min(rho)), float(np.max(rho)),
                             float(np.median(rho_all)),
                             float(np.median(kept)),
                             float(np.median(keff)), thin))
        if not rows:
            continue
        print(f"\n  [{method}]")
        print(f"    {'level':>7}{'cells':>7}{'rho med':>10}{'rho min':>10}"
              f"{'rho max':>10}{'rho all':>10}{'kept':>8}{'top set':>9}"
              f"{'too thin':>10}")
        for lv, ncell, med, lo, hi, med_all, kept, keff, thin in rows:
            lab = 'floor' if isinstance(lv, str) else f'{lv:.2f}'
            print(f"    {lab:>7}{ncell:>7}{med:>10.4f}{lo:>10.4f}{hi:>10.4f}"
                  f"{med_all:>10.4f}{kept:>8.3f}{keff:>9.0f}{thin:>10}")


def report_replicate_spread(p):
    """The drift has to clear the spread between independent noise draws."""
    reps = p['per_replicate']
    if len(reps) < 2:
        print("\n  REPLICATE SPREAD: one noise draw only, so there is nothing for "
              "the drift above to clear. Re-run with --replicates 2 or more before "
              "reading any level-to-level movement as real.")
        return
    levels = [lv for lv in _levels_of(p) if lv > 0.0]
    print("\n  DRIFT AGAINST THE SPREAD BETWEEN NOISE DRAWS")
    print(f"  drift = 1 - rho against level 0.0, median over cells. spread = the "
          f"range of that")
    print(f"  same figure across the {len(reps)} independent draws. A level whose "
          f"drift does not")
    print("  exceed its own spread has not moved.")
    for method in p['importance_methods']:
        rows = []
        for lv in levels:
            per_draw = []
            for block in reps.values():
                vals = [_m(c['spearman_either_scored'])
                        for cells in block.get('base', {}).get(method, {}).values()
                        if (c := cells.get(f'{lv:g}')) is not None]
                if vals:
                    per_draw.append(1.0 - float(np.median(vals)))
            if len(per_draw) >= 2:
                rows.append((lv, float(np.median(per_draw)),
                             float(np.max(per_draw) - np.min(per_draw)),
                             len(per_draw)))
        if not rows:
            continue
        print(f"\n  [{method}]")
        print(f"    {'level':>7}{'drift':>10}{'spread':>10}{'draws':>7}  verdict")
        for lv, drift, spread, ndraw in rows:
            verdict = 'clears the draw' if drift > spread else 'inside the draw'
            print(f"    {lv:>7.2f}{drift:>10.4f}{spread:>10.4f}{ndraw:>7}  {verdict}")


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
        for lv in _levels_of(p) + ['floor_seed']:
            key = lv if isinstance(lv, str) else f'{lv:g}'
            lab = 'floor' if isinstance(lv, str) else f'{lv:.2f}'
            c = per_level.get(key)
            if c is None:
                continue
            shared = c.get('median_n_features_selected_at_both_levels',
                           float('nan'))
            scored = c.get('median_n_scored_features', float('nan'))
            if 'median_spearman_either_scored_vs_level_zero' in c:
                print(f"    {lab:>7}{c['n_cells']:>7}"
                      f"{c['median_spearman_either_scored_vs_level_zero']:>10.4f}"
                      f"{c['min_spearman_either_scored_vs_level_zero']:>10.4f}"
                      f"{c['max_spearman_either_scored_vs_level_zero']:>10.4f}"
                      f"{shared:>9.0f}{scored:>9.0f}")
            else:
                print(f"    {lab:>7}{c['n_cells']:>7}"
                      f"{'too few shared':>30}"
                      f"{shared:>9.0f}{scored:>9.0f}")


def report_hybrid(p):
    """Which source features the recipe takes, level by level."""
    hyb = p.get('summary', {}).get('hybrid') or {}
    if not hyb:
        return
    print("\n  THE FEATURES THE RECIPES TAKE")
    print(f"  each fold's recipes held fixed at every level: {p.get('recipe_lineage')}")
    for fold, by_model in sorted((p.get('fold_configurations') or {}).items()):
        for model, rec in by_model.items():
            recs = [(h.get('budget'), h.get('allocation_method'),
                    h.get('importance_method'), h.get('evaluator'))
                   for h in rec.get('hybrids') or []]
            print(f"  fold {fold} {model}: base {rec.get('base')}; hybrids "
                  f"(budget in features, allocation, importance, evaluator) {recs}")
    print(f"    {'level':>7}{'cells':>7}{'features taken':>17}"
          f"{'kept from lv 0.0':>19}")
    for lv in _levels_of(p):
        c = hyb.get(f'{lv:g}')
        if c is None:
            continue
        print(f"    {lv:>7.2f}{c['n_cells']:>7}"
              f"{(c.get('median_n_selected_features') or float('nan')):>17.0f}"
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
          f"models {p.get('models')}, arm {p.get('arm')}")
    whole, how_much = completeness(p)
    print(f"  {'COMPLETE' if whole else 'PARTIAL RUN'}: {how_much}")
    print("=" * 78)

    if p.get('pool_missing') or p.get('pool_extra'):
        print(f"  POOL DIFFERS from the signed-off list: "
              f"missing={p['pool_missing']} extra={p['pool_extra']}")

    checked, bad = control(p)
    if not checked:
        print("\n  CONTROL: not run — this file has no base-arm cells, so there is "
              "no level-0.0-against-itself row to check. The hybrid tables below "
              "stand on their own counts.")
    elif bad:
        print(f"\n  CONTROL FAILED on {len(bad)} cells. Level 0.0 against itself "
              f"must give Spearman 1.0 and retain every top-k feature. Nothing "
              f"below can be read.")
        for row in bad[:10]:
            print(f"    replicate {row[0]} {row[1]}/{row[2]}: {row[3]}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")
    else:
        print(f"\n  CONTROL: level 0.0 against itself gives Spearman 1.0 and keeps "
              f"every top-k feature, on all {checked} cells.")

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
        return
    for path, p in found:
        report(path, p)
    gaps = missing_conditions(found)
    if gaps:
        print("\n  DATASETS MISSING DECIDED DAMAGE KINDS (no readable file):")
        for ds, miss in sorted(gaps.items()):
            print(f"    {ds}: {miss}")
    print("\n" + "=" * 78)
    print(f"  {len(found)} result files read. The tables are never pooled across "
          f"datasets:")
    print("  a Spearman on an rmse-scored dataset and one on a spearman-scored "
          "dataset are")
    print("  not the same quantity.")


if __name__ == '__main__':
    main()
