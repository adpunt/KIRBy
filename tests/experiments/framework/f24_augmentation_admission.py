#!/usr/bin/env python3
"""F24: which way of admitting extra features works, across the roster.

Paper: Framework. Picking one admission method to wire into the hybrid pipeline
is a build decision, so this is a framework experiment, not a noise or
similarity one.

THE QUESTION
    Not "does augmenting work" -- assume it does. Which specific way of doing it
    works, per dataset. If exactly one way works that is a complete answer. A
    method losing on 20 datasets of 30 is not a reason to drop it if it is the
    best method on the other 10, so nothing here is averaged across datasets and
    the report is per-cell win counts.

    c21 answers the sibling question -- whether extras do more on one strong
    representation at full width or on the multi-representation hybrid at a
    budget. This file holds the method axes fixed on ONE placement per run and
    varies the rules instead.

THE FOUR METHODS (greedy_ablation was cut 2026-08-26)
    none            no extras at all. The control every other method must beat.
    all             every donor's top `augmentation_budget` features, no test
                    at all. The do-nothing null. On logd it BEAT real selection
                    (+0.0131 against +0.0058) before its ranking was corrected
                    on 2026-08-26, so that comparison has to be re-run and this
                    is where.
    capped          extras must DISPLACE base features; the finished width never
                    exceeds the budget. The only method where a supplementary
                    feature has to be worth more than a base one, and the design
                    the 21 August budget-split run supports (48 of 48 cells
                    positive on logd).
    uncapped        extras are appended OUTSIDE the budget, so an augmentation-
                    off arm and an augmentation-on arm at the same nominal
                    budget are NOT the same width.

THE AXES, AND WHICH METHOD READS WHICH
    donor source        curated `augmentation` list / top-N features of each
                        other base representation / both        [all but none]
    features per donor  5, 64, off                              [capped, uncapped]
    candidate ranking   residual_correlation, rf_importance     [capped, uncapped]
    size rule           the ladder of 0/5/10/20/40, or a fraction of the width
                                                                [capped, uncapped]
    margin              how much better than keeping NOTHING a block must score,
                        as a multiple of this dataset's own fold spread
                                                                [capped, uncapped]
    features per donor for the untested methods
                        `augmentation_budget`, 5/20/64          [all, greedy_ablation]
    block threshold     `augmentation_threshold`, as fold-spread multiples
                                                                [greedy_ablation]

    An axis a method does not read is COLLAPSED to one value and the reason is
    printed, so a cell that never ran is visible in the log rather than silently
    missing.

THE MARGIN IS IN FOLD-SPREAD UNITS
    `augmentation_margin` reaches the engine as a raw number in the units of the
    dataset's metric. 0.02 is two thirds of logd's fold-to-fold RMSE spread and
    about an eighth of flab_thermo_tresanco's Spearman spread of 0.1546, so one
    list of absolute numbers means very different strictness on different
    datasets. This file measures the base-only arm's fold-to-fold standard
    deviation first, multiplies it by each requested multiplier, and passes the
    PRODUCT in. Both the multiplier and the measured spread are written into
    every row, so a stored number can be read back in either unit.

THREE CONTROLS, NOT OPTIONAL
    matched random  the real donors replaced by randomly drawn features of the
                    same shape. Its own arm, because adding features changes
                    model capacity by itself and without this a win cannot be
                    told apart from "the model got more features".
    pure noise      a planted donor of Gaussian noise as wide as the real donors
                    combined. Planted INTO the pool, so it costs no extra cells.
                    Anything that keeps these features is broken.
    near-copy       a planted donor of features copied out of the base
                    representation and roughed up by 5% of each feature's own
                    standard deviation. Only this decoy catches a rule that
                    rewards duplicating what the base already has; no amount of
                    pure noise reveals it.

WHAT IS REPORTED
    Per (dataset, cell, fold): the outer score, the claimed improvement the
    method read off its own internal slice, the delivered improvement against
    the no-extras control on the untouched fold, how many features were kept
    from each donor, and how many came from each decoy. Never a mean across
    datasets, never a mean rank.

WALL TIME
    Structural, not measured: per fold a `capped` cell on the ladder costs one
    forest fit per donor wider than the cap, plus three forest fits for the
    out-of-fold residual ranking (one for rf_importance), plus one forest fit to
    rank the base features, plus five fits of the real downstream model. With 5
    to 9 donors that is 14 to 18 fits. A fraction replaces the five scoring fits
    with two. `uncapped` is one fit fewer than `capped`.

    Run --count-only first. It prints the cell count and the fit count per
    dataset without fitting anything, and --time-only measures seconds per fit
    on one fold so the two can be multiplied. This is a SERVER job.

Usage:
    # what it would cost, no fitting at all
    python tests/experiments/framework/f24_augmentation_admission.py \\
        --dataset herg_ki --count-only

    # measure seconds per fit on one fold of the most expensive cell
    python tests/experiments/framework/f24_augmentation_admission.py \\
        --dataset herg_ki --time-only

    # plumbing check: every method dispatches, one fold, one model
    python tests/experiments/framework/f24_augmentation_admission.py \\
        --dataset herg_ki --smoke

    # server, one dataset (what the SLURM array task calls)
    python tests/experiments/framework/f24_augmentation_admission.py \\
        --dataset herg_ki --placement single
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import traceback
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))     # tests/
sys.path.insert(0, os.path.join(HERE, '..'))           # tests/experiments/

# ---------------------------------------------------------------- the axes ---

# FOUR methods, not five. 'greedy_ablation' was CUT on 2026-08-26 by the owner
# -- "No greedy at all" -- and it sits inside the standing greedy decision,
# which covers greedy accept/reject as a feature-admission rule. It stays
# dispatchable in the engine so old recipes do not crash; nothing here selects
# it. See the block above greedy_augmentation_ablation in src/kirby/hybrid.py.
# Passing --methods greedy_ablation still works, deliberately, so the cut can
# be checked rather than taken on trust.
METHODS = ('none', 'all', 'capped', 'uncapped')
CUT_METHODS = ('greedy_ablation',)
DONOR_SOURCES = ('curated', 'borrow', 'both')
PER_DONOR_CAPS = (5, 64, None)          # None = every feature of every donor
RANKINGS = ('residual_correlation', 'rf_importance')

# Multiples of the base-only arm's fold-to-fold standard deviation. 0.0
# reproduces the rule before 2026-08-26: any improvement wins, however small,
# so a near-tie is settled by luck. For `capped` a lost near-tie also throws
# away real base features, which is strictly worse than a few useless extras.
MARGIN_MULTIPLES = (0.0, 0.1, 0.25, 0.5, 1.0)

# Extras sized as a fraction of the width instead of the fixed ladder. Stops at
# 0.25 to match the candidate cut-off ceiling f25 sweeps: a quarter of a 1000
# budget is 250 extras, which a cut-off of 250 exactly allows. None = ladder.
FRACTIONS = (None, 0.05, 0.10, 0.25)

# Features taken from EACH donor by `all` and `greedy_ablation`. A different
# setting from the per-donor cap, which only reaches `capped` and `uncapped`.
AUG_BUDGETS = (5, 20, 64)

# Threshold a whole donor block must clear for `greedy_ablation` to keep it,
# in the same fold-spread units as the margin. None = keep any block that
# improves at all.
THRESHOLD_MULTIPLES = (0.0, 0.1, 0.25, 0.5, 1.0)

# The pooled ranked candidate list is cut to this many features before a block
# size is chosen. FIXED here, swept by f25: at 50 every fraction of a 1000
# budget clamps to the same 50 extras, so this file runs at the value f25
# recommends and does not vary it itself.
MAX_CANDIDATES = 250

# Features taken from each OTHER base representation for the `borrow` source.
BORROW_TOP_N = 5

PLACEMENTS = ('single', 'hybrid')
HYBRID_BUDGETS = (500, 1000, 2000)
# Which allocation method builds the base hybrid. A real axis as of 2026-08-27,
# because "which admission method wins" is only useful if the answer survives
# the allocation methods the search actually uses. Default keeps the single
# 'weighted_global' this file used before, so an unflagged run is unchanged.
#
# It applies to the HYBRID placement only. On the single-representation
# placement one representation takes 100% of the budget whatever the allocation
# says, so the axis is inert there and is collapsed with that reason printed.
#
# The nine are find_best_hybrid.DEFAULT_ALLOCATION_METHODS -- what the search
# sweeps -- and NOT the 39 names in NAMED_ALLOCATIONS. The greedy family is cut.
ALLOCATIONS_SEARCHED = ('top_2', 'top_3', 'top_5', 'performance_weighted',
                        'entropy_1.0', 'power_2', 'rank_linear', 'entropy_2.0',
                        'rank_exp_d0.5')
HYBRID_ALLOCATION = 'weighted_global'
HYBRID_IMPORTANCE = 'lightgbm_gain'

# Two downstream models, not three, and both cheap. lightgbm is this
# workstream's recorded default (AUGMENTATION.md section 2: "lightgbm is the
# default downstream model for augmentation runs with a second model as a
# robustness check"). rf is that second check and is a genuinely different
# learner -- bagged rather than boosted -- so the two agreeing is worth
# something. xgboost was dropped on 2026-08-26: it is a third boosted-tree
# model, so it adds cost without adding a different kind of learner.
# GP, MLP and SVR are not here and should not be added; they are the expensive
# ones.
DEFAULT_MODELS = ('lightgbm', 'rf')

# Three folds, not five. CUT BY THE USER on 2026-08-26 for time, not for a
# scientific reason -- say so, and do not present it as anything else. Two
# consequences worth knowing. The fold-to-fold spread is measured on three
# numbers rather than five, and it is the unit every margin and threshold is
# expressed in. And flab_thermo_tresanco's spread is 0.1546, which is large
# next to every effect being looked for. The winner per dataset should be
# re-run at five folds before anything is quoted.
DEFAULT_N_FOLDS = 3
DEFAULT_N_SEEDS = 3

DECOY_ECHO_COLS = 64
DECOY_ECHO_JITTER = 0.05

# Why an axis was collapsed for a method. Printed, never silent -- a cut needs a
# stated reason and "it probably will not matter" is not one.
COLLAPSE_REASON = {
    ('none', 'source'): 'no donors are offered at all',
    ('none', 'cap'): 'no donors are offered at all',
    ('none', 'ranking'): 'no donors are offered at all',
    ('none', 'size'): 'no donors are offered at all',
    ('all', 'cap'): 'aug_per_source_budget never reaches this method; it uses '
                    'augmentation_budget instead',
    ('all', 'ranking'): 'no ranking runs -- every donor block is taken whole',
    ('all', 'size'): 'the size IS augmentation_budget; there is no ladder and '
                     'no fraction',
    ('greedy_ablation', 'cap'): 'aug_per_source_budget never reaches this '
                                'method; it uses augmentation_budget instead',
    ('greedy_ablation', 'ranking'): 'blocks are kept or dropped whole; no '
                                    'pooled candidate ranking runs',
    ('greedy_ablation', 'size'): 'the size is the whole block; the accept test '
                                 'is augmentation_threshold',
}


# ------------------------------------------------------------- enumeration ---

def margins_for(rung_scores, spread, multiples=MARGIN_MULTIPLES):
    """Which block size each margin would have chosen, from one ladder scoring.

    `rung_scores` is {k: score} for every rung the ladder scored, reported by
    augment_capped / augment_uncapped as of 2026-08-26. The engine's rule is:
    beat the no-extras score by more than the margin, and be the best rung that
    does (hybrid.py, the two accept tests). Both parts are applied here.

    The margin never changes what a rung SCORES -- only which rung wins.
    Measured 2026-08-26: six margin values returned the identical no-extras
    score and produced two distinct outcomes. So running five margins as five
    conditions re-fits the same five models five times. Reading them off one
    scoring is exact, not an approximation.

    Returns {multiple: chosen k}.
    """
    if not rung_scores:
        return {m: 0 for m in multiples}
    scores = {int(k): float(v) for k, v in rung_scores.items()}
    base = scores.get(0, float('-inf'))
    out = {}
    for m in multiples:
        bar = base + m * spread
        best_k, best = 0, base
        for k in sorted(scores):
            if k == 0:
                continue
            if scores[k] > bar and scores[k] > best:
                best_k, best = k, scores[k]
        out[m] = best_k
    return out


def _size_settings():
    """(label, fraction, margin_multiple) for capped and uncapped.

    THE MARGIN IS NOT ENUMERATED HERE. The ladder runs ONCE, at margin 0, and
    every margin multiple is then read off the rung scores it reports (see
    margins_for). Running five margins as five conditions re-fits the same five
    models five times: measured 2026-08-26, six margin values returned the
    identical no-extras score and gave two distinct outcomes. Where a margin
    does pick a different block size, that one gets its own run -- so the axis
    is fully covered, at the cost of the outcomes that actually differ rather
    than of every value.

    A fraction and a margin remain alternatives, never a stack: with a fraction
    set the size is already fixed and the engine does not consult the margin
    (hybrid.py, the two accept tests).
    """
    out = [('ladder', None, 0.0)]
    out += [(f'frac{f:g}', f, 0.0) for f in FRACTIONS if f is not None]
    return out


def enumerate_cells(methods=METHODS, sources=DONOR_SOURCES,
                    caps=PER_DONOR_CAPS, rankings=RANKINGS,
                    aug_budgets=AUG_BUDGETS,
                    thresholds=THRESHOLD_MULTIPLES):
    """Every configuration, with the axes each method does not read collapsed.

    Returns a list of dicts. `collapsed` on each cell names the axes that were
    folded to one value and why, so the log says what did not run.
    """
    cells = []
    for method in methods:
        if method == 'none':
            cells.append({
                'method': 'none', 'source': None, 'cap': None,
                'ranking': None, 'size': 'n/a', 'fraction': None,
                'margin_mult': None, 'aug_budget': None, 'threshold_mult': None,
                'collapsed': ['source', 'cap', 'ranking', 'size'],
            })
            continue

        for source in sources:
            if method == 'all':
                for ab in aug_budgets:
                    cells.append({
                        'method': method, 'source': source, 'cap': None,
                        'ranking': None, 'size': f'top{ab}', 'fraction': None,
                        'margin_mult': None, 'aug_budget': ab,
                        'threshold_mult': None,
                        'collapsed': ['cap', 'ranking', 'size'],
                    })
            elif method == 'greedy_ablation':
                for ab, th in product(aug_budgets, thresholds):
                    cells.append({
                        'method': method, 'source': source, 'cap': None,
                        'ranking': None, 'size': f'block{ab}', 'fraction': None,
                        'margin_mult': None, 'aug_budget': ab,
                        'threshold_mult': th,
                        'collapsed': ['cap', 'ranking', 'size'],
                    })
            else:                                    # capped, uncapped
                for cap, ranking, (slabel, frac, mmult) in product(
                        caps, rankings, _size_settings()):
                    cells.append({
                        'method': method, 'source': source, 'cap': cap,
                        'ranking': ranking, 'size': slabel, 'fraction': frac,
                        'margin_mult': mmult, 'aug_budget': None,
                        'threshold_mult': None, 'collapsed': [],
                    })
    return cells


def effective_k(fraction, width_ref, pool_width, cutoff):
    """How many extras a fraction cell can actually take.

    Three ceilings apply at once and the smallest wins: the fraction of the
    reference width, the number of candidates the pool holds after the
    per-donor cap, and the length of the ranked list (hybrid.py:2533-2546,
    :2759-2764). This is why a fraction of 0.05 and one of 0.25 can be the same
    cell: at a budget of 1000 they ask for 50 and 250, and a list cut to 50
    hands both of them 50.
    """
    return min(int(round(fraction * width_ref)), pool_width, cutoff)


def mark_degenerate(cells, pool_width_by_source, max_donor_by_source,
                    width_ref, cutoff):
    """Label every cell that cannot give its own answer, with the arithmetic.

    Degeneracy is the only reason this project accepts for not running a cell,
    so each one is worked out from the measured widths rather than argued, and
    the count it collapses onto is named.

    Returns {cell label: reason}.
    """
    out = {}
    # Fraction cells: two are the same cell when they end up taking the same
    # number of features. Keep the smallest fraction of each distinct count.
    seen_k = {}
    for c in cells:
        if c['fraction'] is None:
            continue
        pw = pool_width_by_source.get(c['source'])
        if pw is None:
            continue
        k = effective_k(c['fraction'], width_ref, pw, cutoff)
        key = (c['method'], c['source'], c['cap'], c['ranking'], k)
        if key in seen_k:
            out[cell_label(c)] = (
                f"takes {k} features, the same as {seen_k[key]}: the fraction "
                f"asks for {int(round(c['fraction'] * width_ref))} but the "
                f"pool holds {pw} and the ranked list is cut to {cutoff}")
        else:
            seen_k[key] = f"frac{c['fraction']:g}"

    # `all` takes augmentation_budget features from EACH donor, so it saturates
    # against the WIDEST single donor, not against the pooled width.
    seen_ab = {}
    for c in cells:
        if c['method'] != 'all' or c['aug_budget'] is None:
            continue
        widest = max_donor_by_source.get(c['source'])
        if widest is None:
            continue
        eff = min(c['aug_budget'], widest)
        key = (c['source'], eff)
        if key in seen_ab:
            out[cell_label(c)] = (
                f"takes {eff} features per donor, the same as "
                f"{seen_ab[key]}: the widest donor here is {widest} features")
        else:
            seen_ab[key] = f"top{c['aug_budget']}"
    return out


def cell_label(cell, with_margin=False):
    """Name for one configuration.

    with_margin adds the margin multiple. It is NOT in the enumeration name,
    because the ladder is enumerated once and the margins are read off it --
    but it MUST be in the name of a row that ran, or two runs of the same
    configuration under different margins print identically and cannot be told
    apart afterwards. Seen on real data: the same label appeared twice on
    flab_thermo_tresanco, once keeping 8.3 features and once 1.7.
    """
    bits = [cell['method']]
    for key in ('source', 'cap', 'ranking', 'size'):
        v = cell.get(key)
        if v is not None and v != 'n/a':
            bits.append(str(v))
    if cell.get('threshold_mult') is not None:
        bits.append(f"th{cell['threshold_mult']:g}")
    if with_margin and cell.get('margin_mult') is not None:
        bits.append(f"m{cell['margin_mult']:g}")
    return '__'.join(bits)


def fits_per_fold(cell, n_donors, donor_widths=None):
    """Model fits one fold of this cell costs. Counted from the engine, not timed.

    Read off src/kirby/hybrid.py. `_select_augmentation_features` fits one
    forest per donor it has to trim (:2180-2184) and NONE at all when the donor
    is already narrower than the cap, because it returns every feature without
    fitting (:2171-2173) -- so a cap of `off` costs no trimming fits whatever.
    `_pool_and_rank_aug_features` fits one forest for rf_importance and three
    for residual_correlation's out-of-fold residuals (:2274-2331).
    `augment_capped` fits one more forest to rank the base features
    (:2503-2512). Each rung of the ladder is one fit of the real downstream
    model (:2581-2599); a fraction leaves only two rungs, k=0 and k.

    donor_widths: the donors' feature counts, so a donor already narrower than
    the cap is not counted. Omitting it assumes every donor needs trimming,
    which OVERSTATES the cost.
    """
    if cell['method'] == 'none':
        return 0

    if cell['method'] in ('all', 'greedy_ablation'):
        # These two use augmentation_budget, not the per-donor cap.
        cut = cell.get('aug_budget') or 20
        trim = (sum(1 for w in donor_widths if w > cut)
                if donor_widths else n_donors)
        score = n_donors + 1 if cell['method'] == 'greedy_ablation' else 0
        return trim + score

    cap = cell.get('cap')
    if cap is None:
        trim = 0                       # every feature pooled; nothing is fitted
    elif donor_widths:
        trim = sum(1 for w in donor_widths if w > cap)
    else:
        trim = n_donors
    rank = 1 if cell['ranking'] == 'rf_importance' else 3
    base_rank = 1 if cell['method'] == 'capped' else 0
    score = 2 if cell['fraction'] is not None else 5
    return trim + rank + base_rank + score


# ------------------------------------------------------------------- donors --

def best_solo_rep(dataset, candidates):
    """The dataset's strongest single representation, and where that came from.

    Read off <cache>/rep_prune/<dataset>/scores.json, which holds every
    representation's out-of-fold solo score under the dataset's own metric.
    That file is what DATASET_STATUS.md's "best solo" column is built from, so
    this is the same yardstick and not a second opinion.

    Returns (name, provenance, gap to the runner-up). The gap is reported
    because it says whether this dataset has a DOMINANT representation to nudge
    at all: on logd the top two are 0.0018 apart against fold spreads around
    0.03, so there is nothing there to be dominant.

    Falls back to the widest candidate when the file is absent, and says so
    rather than presenting a guess as a measurement.
    """
    base = os.environ.get('KIRBY_CACHE_DIR',
                          os.path.join(os.path.expanduser('~'), '.cache',
                                       'kirby'))
    path = os.path.join(base, 'rep_prune', dataset, 'scores.json')
    try:
        with open(path) as fh:
            reps = (json.load(fh) or {}).get('reps', {})
        ranked = sorted(
            ((k, v.get('solo')) for k, v in reps.items()
             if v.get('solo') is not None and k in candidates),
            key=lambda t: -t[1])
        if ranked:
            gap = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else None
            return ranked[0][0], f'rep_prune solo score ({path})', gap
    except (OSError, ValueError, TypeError):
        pass
    widest = max(candidates, key=lambda r: candidates[r].shape[1])
    return widest, 'NOT MEASURED — widest representation in the pool', None


def build_donor_pool(source, base_name, base_reps, aug_reps, top_n=BORROW_TOP_N):
    """{donor name: features} and the per-donor cap dict for one donor source.

    curated  the dataset's own `augmentation` list, whole
    borrow   every OTHER base representation, capped to top_n by importance
    both     the two together, with only the borrowed ones capped
    """
    others = {k: v for k, v in base_reps.items() if k != base_name}
    curated = {k: v for k, v in aug_reps.items() if k != base_name}
    if source == 'curated':
        return dict(curated), None
    if source == 'borrow':
        return dict(others), {k: top_n for k in others}
    return {**curated, **others}, {k: top_n for k in others}


def random_donor_pool(donors, base_reps, base_name, rng):
    """The matched-shape control: same widths, features drawn at random.

    Drawn from the other base representations. Random features carry no label
    information, so drawing them outside the fold loop leaks nothing.
    """
    others = [v for k, v in base_reps.items() if k != base_name]
    if not others:
        return None
    out = {}
    for i, (name, block) in enumerate(donors.items()):
        src = others[i % len(others)]
        n_take = min(block.shape[1], src.shape[1])
        cols = np.sort(rng.choice(src.shape[1], size=n_take, replace=False))
        out[f'{name}__rand'] = src[:, cols]
    return out


# --------------------------------------------------------------- reporting ---

def print_plan(cells, n_donors_by_source, n_folds, n_models, placements,
               hybrid_budgets, widths_by_source=None, n_allocations=1):
    print('\n' + '=' * 74)
    print('PLAN — configurations and model fits, counted from the engine')
    print('=' * 74)
    by_method = {}
    for c in cells:
        by_method.setdefault(c['method'], []).append(c)
    total_cells = total_fits = 0
    print(f"\n  {'method':17s}{'cells':>7}{'fits/fold':>12}{'fits total':>13}")
    for method in METHODS:
        ms = by_method.get(method, [])
        if not ms:
            continue
        fits = sum(fits_per_fold(c, n_donors_by_source.get(c['source'], 0),
                                 (widths_by_source or {}).get(c['source']))
                   for c in ms)
        # Every cell is run twice: once against the real donors and once
        # against the matched-shape random donors of the same width.
        n_arms = 2 if method != 'none' else 1
        tot = fits * n_folds * n_models * n_arms
        print(f"  {method:17s}{len(ms) * n_arms:7d}{fits:12d}{tot:13,d}")
        total_cells += len(ms) * n_arms
        total_fits += tot

    # Allocation multiplies the HYBRID placement only. On the single-
    # representation placement one representation takes the whole budget
    # whatever the allocation says, so it is one condition there.
    n_place = sum(len(hybrid_budgets) * n_allocations if p == 'hybrid' else 1
                  for p in placements)
    print(f"\n  per placement-and-budget : {total_cells:,} configurations, "
          f"{total_fits:,} fits")
    print(f"  placement, budget and allocation combinations: {n_place}"
          f"  ({n_allocations} allocation method"
          f"{'s' if n_allocations != 1 else ''})")
    print(f"  DATASET TOTAL            : {total_cells * n_place:,} "
          f"configurations, {total_fits * n_place:,} model fits")
    print(f"\n  Multiply the fit count by the seconds-per-fit that --time-only "
          f"measures.\n  Nothing here is a timing; it is a count.")
    print(f"\n  THIS COUNT IS A FLOOR, not the final number. The ladder runs")
    print(f"  once and every margin multiple is read off its rung scores. A")
    print(f"  multiple that picks a DIFFERENT block size then gets its own run,")
    print(f"  and how many do that is not knowable until the scores exist.")
    print(f"  Measured on synthetic data: six margin multiples gave two")
    print(f"  distinct outcomes. So expect up to about twice the capped and")
    print(f"  uncapped rows above, and no change to the other three methods.")


def print_degeneracies(degenerate):
    print('\n' + '-' * 74)
    if not degenerate:
        print('DEGENERATE CONFIGURATIONS: none. Every one can give its own '
              'answer.')
        return
    print(f'DEGENERATE CONFIGURATIONS: {len(degenerate)}, skipped for a stated '
          f'reason and not for cost.')
    by_reason = {}
    for label, reason in degenerate.items():
        by_reason.setdefault(reason, []).append(label)
    for reason, labels in sorted(by_reason.items(),
                                 key=lambda kv: -len(kv[1]))[:8]:
        print(f'  {len(labels):4d}  {reason}')
        print(f'        e.g. {sorted(labels)[0]}')


# ------------------------------------------------------------------- driver ---

def measure_fold_spread(X_base, labels, groups, n_folds, mode, model_factory,
                        metric_fn, n_seeds, cv_split, fit_score,
                        allow_undefined=False):
    """Fold-to-fold standard deviation of the base-only arm.

    This is the unit every margin and every threshold is expressed in, so it is
    measured BEFORE any cell runs and reported alongside them. Measured on the
    base representation alone, with no extras of any kind.

    RAISES on fewer than two folds. The standard deviation of one number is
    0.0, which would make every margin multiple collapse to a margin of 0.0 --
    the accept-any-improvement rule -- while the output still named five
    different multipliers. A whole axis would read as swept and not be. Pass
    allow_undefined=True only where the spread is not used, such as a timing
    run.
    """
    from sklearn.preprocessing import StandardScaler
    scores = []
    for train_idx, val_idx in cv_split(groups, n_folds, X_base, labels,
                                       mode=mode):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_base[train_idx])
        Xva = sc.transform(X_base[val_idx])
        scores.append(fit_score(Xtr, labels[train_idx], Xva, labels[val_idx],
                                model_factory, metric_fn, n_seeds))
    if len(scores) < 2:
        if not allow_undefined:
            raise ValueError(
                f'the fold spread is undefined on {len(scores)} fold(s), and '
                f'every margin and threshold in this experiment is a multiple '
                f'of it. One fold would silently make all five multipliers '
                f'mean a margin of 0.0. Run with at least 2 folds, or pass '
                f'--time-only, where the spread is not used.')
        return float('nan'), [float(s) for s in scores]
    return float(np.std(scores)), [float(s) for s in scores]


def run_dataset(ds, args):
    from experiments._shared import (
        load_data, get_config, get_model_factories, cv_split, get_domain_metric,
        build_fold_hybrid, plant_decoys, decoy_counts,
    )
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from kirby.find_best_hybrid import Blueprint
    from sklearn.preprocessing import StandardScaler

    def _fit_score(Xtr, ytr, Xva, yva, factory, metric_fn, n_seeds):
        vals = []
        for s in range(n_seeds):
            m = factory()
            try:
                if 'random_state' in m.get_params():
                    m.set_params(random_state=42 + s)
            except Exception:
                pass
            m.fit(Xtr, ytr)
            vals.append(float(metric_fn(yva, m.predict(Xva))))
        return float(np.mean(vals))

    cfg = DATASET_REPS.get(ds, {})
    pool_names = cfg.get('general', [])
    aug_names = cfg.get('augmentation', [])
    if not pool_names:
        raise ValueError(f'{ds} has no general representation pool in '
                         f'DATASET_REPS; nothing to build a base from.')

    data = load_data(ds, 'all', rep_names=sorted(set(pool_names + aug_names)))
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    base_reps = {r: reps[r] for r in pool_names if r in reps}
    aug_reps = dict(data.get('augmentations') or {})
    for r in aug_names:
        if r not in aug_reps and r in reps:
            aug_reps[r] = reps[r]
    aug_reps = {k: v for k, v in aug_reps.items() if k in aug_names}

    config = get_config(args)
    # Three folds by default, not the config's five. CUT BY THE USER on
    # 2026-08-26 for time. That is the reason; it is not a scientific one and
    # should not be written up as one. --n-folds overrides it.
    n_folds = 1 if args.smoke else int(args.n_folds or config['n_splits'])
    mode = 'intermediary' if n_folds == 1 else 'full'
    n_seeds = 1 if args.smoke else args.n_seeds

    model_names = ([m.strip() for m in args.models.split(',')] if args.models
                   else (['rf'] if args.smoke else list(DEFAULT_MODELS)))
    models = get_model_factories(model_names, task_type=task_type,
                                 n_samples=len(labels))

    # The base representation: the dataset's strongest solo one unless pinned.
    if args.base_rep:
        base_name, base_provenance, solo_gap = args.base_rep, 'pinned by --base-rep', None
    else:
        base_name, base_provenance, solo_gap = best_solo_rep(ds, base_reps)
    if base_name not in base_reps:
        raise ValueError(f'--base-rep {base_name} is not in {ds} pool '
                         f'{sorted(base_reps)}')
    X_base = base_reps[base_name]

    # Donor pools and their widths, needed for the degeneracy assertions.
    pools = {}
    for source in DONOR_SOURCES:
        donors, cap = build_donor_pool(source, base_name, base_reps, aug_reps)
        pools[source] = (donors, cap)
    n_donors_by_source = {s: len(p[0]) for s, p in pools.items()}
    width_by_source = {s: sum(v.shape[1] for v in p[0].values())
                       for s, p in pools.items()}

    methods = ([m.strip() for m in args.methods.split(',')] if args.methods
               else list(METHODS))
    sources = ([s.strip() for s in args.sources.split(',')] if args.sources
               else (['curated'] if args.smoke else list(DONOR_SOURCES)))
    caps = ([None if c.strip() in ('off', 'none', '') else int(c)
             for c in args.caps.split(',')] if args.caps
            else ([5] if args.smoke else list(PER_DONOR_CAPS)))
    if args.smoke:
        # --smoke has to fit the devel partition's 10-minute limit, so it is
        # one fold, one model, one seed, one donor source and one per-donor
        # cap. It proves the methods dispatch and complete on REAL data. It
        # measures nothing: with one fold the fold spread is undefined, so
        # every margin is 0 and the margin axis means nothing here.
        print('  SMOKE: 1 fold, 1 model, 1 seed, curated donors, 5 per donor.')
        print('  Plumbing only — no margin is swept and no number is a result.')
    cells = enumerate_cells(methods=methods, sources=sources, caps=caps)
    placements = ([args.placement] if args.placement != 'both'
                  else list(PLACEMENTS))
    allocations = ([a.strip() for a in args.allocations.split(',')]
                   if args.allocations else [HYBRID_ALLOCATION])
    hybrid_budgets = ([int(b) for b in args.hybrid_budgets.split(',')]
                      if args.hybrid_budgets else list(HYBRID_BUDGETS))

    print(f'\nF24 augmentation admission — {ds}')
    print(f'  n = {len(labels)}, metric = {metric_name} '
          f'(lower_is_better={lower_is_better})')
    print(f'  base representation: {base_name} ({X_base.shape[1]} features)')
    print(f'    chosen by: {base_provenance}')
    if solo_gap is not None:
        print(f'    it beats the runner-up by {solo_gap:.4f}. A gap smaller '
              f'than the fold spread means this dataset has no dominant '
              f'representation to nudge.')
    for s in DONOR_SOURCES:
        print(f'  donors, {s:8s}: {n_donors_by_source[s]} donors, '
              f'{width_by_source[s]} features before any cap')
    print(f'  models: {", ".join(models)}  x{n_seeds} seeds  '
          f'over {n_folds} folds')

    # What each ceiling actually is on this dataset. A per-donor cap of 5 over
    # N donors pools to at most 5N features, and that -- not the raw width --
    # is what a fraction and the cut-off see.
    capped_width = {s: min(5 * n_donors_by_source[s], width_by_source[s])
                    for s in DONOR_SOURCES}
    max_donor = {s: max((v.shape[1] for v in pools[s][0].values()), default=0)
                 for s in DONOR_SOURCES}
    # The fraction's reference width differs by method: `capped` keeps the
    # total inside the budget so it is a fraction of the finished width, while
    # `uncapped` appends outside it so it is a fraction of the base's own width
    # (hybrid.py:2540-2546, :2759-2764). On the single-representation placement
    # the two coincide, because the budget IS the base's width.
    width_ref = X_base.shape[1]
    degenerate = mark_degenerate(cells, capped_width, max_donor, width_ref,
                                 args.max_candidates)
    print_degeneracies(degenerate)
    donor_widths = {s: [v.shape[1] for v in pools[s][0].values()]
                    for s in DONOR_SOURCES}
    print_plan([c for c in cells if cell_label(c) not in degenerate],
               n_donors_by_source, n_folds, len(models), placements,
               hybrid_budgets, widths_by_source=donor_widths,
               n_allocations=len(allocations))

    out = {
        'experiment': 'f24_augmentation_admission',
        'dataset': ds,
        'n_samples': int(len(labels)),
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'base_rep': base_name,
        'base_rep_provenance': base_provenance,
        'base_rep_gap_to_runner_up': solo_gap,
        'base_width': int(X_base.shape[1]),
        'donors': {s: {'n': n_donors_by_source[s],
                       'width': width_by_source[s],
                       'width_at_cap_5': capped_width[s],
                       'widest_donor': max_donor[s]}
                   for s in DONOR_SOURCES},
        'n_folds': n_folds,
        'n_seeds': n_seeds,
        'models': list(models),
        'max_candidates': args.max_candidates,
        'degenerate_cells': degenerate,
        'collapse_reasons': {f'{k[0]}/{k[1]}': v
                             for k, v in COLLAPSE_REASON.items()},
        'rows': [],
        'skipped': [],
    }

    if args.count_only:
        return out

    # The unit every margin and threshold is measured in. Measured first,
    # reported alongside, never assumed.
    print('\n  measuring the base-only fold spread (the margin unit)...')
    t0 = time.time()
    first_model = next(iter(models.values()))
    spread, base_scores = measure_fold_spread(
        X_base, labels, groups, n_folds, mode, first_model, metric_fn,
        n_seeds, cv_split, _fit_score,
        # One fold makes the spread undefined, and it is the unit every margin
        # is expressed in. Allowed only where no margin is being swept: a
        # timing run, and --smoke, which is a plumbing check and says so.
        allow_undefined=args.time_only or args.smoke)
    out['fold_spread'] = spread
    out['base_only_fold_scores'] = base_scores
    if np.isfinite(spread):
        print(f'    fold spread = {spread:.4f} {metric_name} '
              f'({time.time() - t0:.0f}s). A margin multiple of 0.25 therefore '
              f'means {0.25 * spread:.4f} {metric_name}.')
    else:
        print(f'    fold spread not measured — one fold. Every margin is 0.0 '
              f'here, which is why this is only allowed under --time-only.')
        spread = 0.0

    if args.time_only:
        # The single most expensive configuration: capped (it ranks the base
        # features as well as the candidates), both donor sources, no per-donor
        # cap so the widest possible pool is ranked, the out-of-fold residual
        # ranking (3 forest fits rather than 1), and the full 5-rung ladder.
        cells = [c for c in cells
                 if c['method'] == 'capped' and c['source'] == 'both'
                 and c['cap'] is None
                 and c['ranking'] == 'residual_correlation'
                 and c['size'] == 'ladder_m0']
        placements, hybrid_budgets = ['single'], []
        models = {model_names[0]: first_model}
        n_arms = 2      # the real donors, and the matched random control
        out['timed_cell'] = cell_label(cells[0])
        out['timed_fits'] = n_arms * fits_per_fold(
            cells[0], n_donors_by_source['both'],
            [v.shape[1] for v in pools['both'][0].values()])
        print(f"\n  timing {out['timed_cell']}, one fold, "
              f"{out['timed_fits']} model fits over {n_arms} arms")

    splits = list(cv_split(groups, n_folds, X_base, labels, mode=mode))

    # Computed ONCE per fold and reused by every configuration.
    #
    # create_hybrid recomputes both on every call. Measured 2026-08-27 on
    # flab_thermo_tresanco: building the base hybrid for one fold with
    # augmentation OFF took 261.9s -- 22.5s of feature importance and the rest
    # scoring the ten representations one model fit each. The grid calls
    # build_fold_hybrid once per (configuration, model, fold), about 8,800
    # times for one dataset at one budget, and paid that every time.
    #
    # Neither quantity depends on the augmentation settings or on the
    # allocation method. Importance does not depend on the downstream model
    # either. So: importance per fold, scores per (fold, model).
    precomp = {'importance': {}, 'scores': {}}
    # Timed from HERE, so representation loading (minutes off a USB drive) and
    # the fold-spread measurement are not counted as fitting time.
    out['fit_start'] = time.time()
    # The matched-random donors are drawn from a FRESH RandomState(0) at every
    # use, not from one advancing generator. A shared generator was measured
    # handing a different random draw to every re-run: on
    # flab_thermo_tresanco the random control of one configuration returned
    # 0.5885, 0.5534, 0.5836 and 0.5534 across four margin re-runs, because the
    # donors changed underneath it each time.
    #
    # That breaks the comparison the control exists for. Real-minus-random is
    # only the effect of the donors if the random side is held fixed; with a
    # moving draw it also carries the luck of that draw, and two configurations
    # are then compared against two different controls.
    rng_ctrl = None

    for placement in placements:
        budgets = hybrid_budgets if placement == 'hybrid' else [X_base.shape[1]]
        # One representation takes 100% of the budget whatever the allocation
        # says, so the axis cannot change anything on the single-representation
        # placement. Collapsed with the reason printed, not silently dropped.
        allocs = allocations if placement == 'hybrid' else [HYBRID_ALLOCATION]
        if placement == 'single' and len(allocations) > 1:
            print(f'  allocation axis collapsed on the single-representation '
                  f'placement: one representation takes the whole budget '
                  f'whatever the allocation says, so all '
                  f'{len(allocations)} give the same hybrid.')
        for budget, alloc_name in product(budgets, allocs):
            for cell in cells:
                if cell_label(cell) in degenerate:
                    continue
                donors, cap_dict = (pools[cell['source']]
                                    if cell['source'] else ({}, None))
                if cell['method'] != 'none' and not donors:
                    out['skipped'].append({
                        'cell': cell_label(cell), 'placement': placement,
                        'budget': budget,
                        'reason': f"{ds} has no donors for source "
                                  f"'{cell['source']}'"})
                    continue

                # The per-donor cap axis overrides the source's own default cap.
                per_source = None
                if cell['cap'] is not None:
                    per_source = {k: cell['cap'] for k in donors}
                elif cell['method'] in ('capped', 'uncapped'):
                    per_source = None       # 'off' -- every feature pooled
                else:
                    per_source = cap_dict

                margin = ((cell['margin_mult'] or 0.0) * spread
                          if cell['margin_mult'] is not None else 0.0)
                threshold = (cell['threshold_mult'] * spread
                             if cell['threshold_mult'] is not None else None)

                for control in ('real', 'random'):
                    if cell['method'] == 'none' and control == 'random':
                        continue

                    def _go(mult, mgn, _a=alloc_name):
                        c = dict(cell, margin_mult=mult, allocation=_a)
                        return _run_cell(
                            c, control, placement, budget, base_name, X_base,
                            base_reps, donors, per_source, mgn, threshold,
                            labels, groups, splits, models, metric_fn,
                            metric_name, lower_is_better, task_type, n_seeds,
                            args, Blueprint, build_fold_hybrid, plant_decoys,
                            decoy_counts, StandardScaler, _fit_score, rng_ctrl,
                            out['skipped'], precomp)

                    row = _go(cell['margin_mult'], margin)
                    if row is not None:
                        out['rows'].append(row)

                    # The margin axis, covered without re-scoring the ladder.
                    # The run above reported every rung's score. Each margin
                    # multiple is applied to those numbers to see which block
                    # size it would pick. A multiple that picks the SAME size
                    # produces the same hybrid and the same outer score, so it
                    # is recorded from the run already done. Only a multiple
                    # that picks a DIFFERENT size gets its own run. Exact, not
                    # an approximation -- the margin never changes what a rung
                    # scores, only which rung wins.
                    if (row is None or cell['fraction'] is not None
                            or cell['method'] not in ('capped', 'uncapped')):
                        continue
                    chosen = _margin_outcomes(row, spread)
                    row['margin_multiples_sharing_this_run'] = sorted(
                        m for m, k in chosen.items() if k == chosen.get(0.0))
                    done = {chosen.get(0.0)}
                    for mult in MARGIN_MULTIPLES:
                        k = chosen.get(mult)
                        if mult == 0.0 or k in done:
                            continue
                        done.add(k)
                        extra = _go(mult, mult * spread)
                        if extra is not None:
                            extra['margin_multiples_sharing_this_run'] = sorted(
                                m for m, kk in chosen.items() if kk == k)
                            out['rows'].append(extra)

    out['fit_seconds'] = time.time() - out.pop('fit_start')
    return out


def _margin_outcomes(row, spread):
    """Which block size each margin multiple would pick, per this row's folds.

    The rung scores differ between folds, so a margin can pick 10 extras on one
    fold and 0 on another. Taking the fold-wise majority would hide that, so
    the whole per-fold list is compared: two margin multiples share a run only
    when they agree on EVERY fold. Anything else gets its own run.
    """
    per_model = row.get('models') or {}
    sig = {}
    for mult in MARGIN_MULTIPLES:
        picks = []
        for m in sorted(per_model):
            for rungs in (per_model[m].get('rung_scores_per_fold') or []):
                picks.append(margins_for(rungs, spread, (mult,))[mult])
        sig[mult] = tuple(picks)
    return sig


def _run_cell(cell, control, placement, budget, base_name, X_base, base_reps,
              donors, per_source, margin, threshold, labels, groups, splits,
              models, metric_fn, metric_name, lower_is_better, task_type,
              n_seeds, args, Blueprint, build_fold_hybrid, plant_decoys,
              decoy_counts, StandardScaler, fit_score, rng_ctrl, skipped,
              precomp=None):
    """One configuration, every fold, every model. Returns one result row."""
    label = cell_label(cell)
    # with_margin: two runs of one configuration under different margins
    # must not share a name, or they cannot be told apart afterwards.
    _al = cell.get('allocation')
    _alt = f'__{_al}' if (_al and placement == 'hybrid') else ''
    arm = (f'{cell_label(cell, with_margin=True)}__{control}'
           f'__{placement}_b{budget}{_alt}')

    pool = donors
    if control == 'random' and donors:
        # Fresh RandomState(0) per call: the same donor set always yields the
        # same random features, so every configuration built on that donor set
        # is compared against one identical control.
        pool = random_donor_pool(donors, base_reps, base_name,
                                 np.random.RandomState(0))
        if not pool:
            skipped.append({'cell': label, 'reason':
                            'no other base representation to draw a matched '
                            'random donor from'})
            return None

    if placement == 'single':
        bp_reps, rep_dict, alloc, importance = (
            [base_name], {base_name: X_base}, 'performance_weighted',
            'lightgbm_gain')
    else:
        bp_reps, rep_dict, alloc, importance = (
            sorted(base_reps), base_reps,
            cell.get('allocation') or HYBRID_ALLOCATION, HYBRID_IMPORTANCE)

    bp = Blueprint(
        reps=bp_reps, allocation_method=alloc, importance_method=importance,
        internal_evaluator='match', budget=int(budget), step_size=5,
        augmentation_strategy=cell['method'],
        aug_per_source_budget=per_source,
        augmentation_margin=float(margin),
        augmentation_fraction=cell['fraction'],
        candidate_ranking=(cell['ranking'] or 'residual_correlation'),
        augmentation_threshold=threshold,
        augmentation_budget=int(cell['aug_budget'] or 20),
        aug_max_candidates=int(args.max_candidates),
        metric=metric_name, task_type=task_type,
        lower_is_better=lower_is_better)

    per_model = {}
    for model_name, factory in models.items():
        scores, dims, claimed = [], [], []
        noise_taken, echo_taken, kept_counts, rungs = [], [], [], []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            y_tr, y_va = labels[train_idx], labels[val_idx]
            tr = {k: v[train_idx] for k, v in rep_dict.items()}
            va = {k: v[val_idx] for k, v in rep_dict.items()}

            fold_pool = pool
            if pool and not args.no_decoys:
                fold_pool = plant_decoys(
                    pool, X_base, np.random.RandomState(1000 + fold_idx),
                    echo_cols=DECOY_ECHO_COLS, echo_jitter=DECOY_ECHO_JITTER)
            tr_aug = ({k: v[train_idx] for k, v in fold_pool.items()}
                      if fold_pool else None)
            va_aug = ({k: v[val_idx] for k, v in fold_pool.items()}
                      if fold_pool else None)

            # Fill the per-fold cache the first time this fold is seen, then
            # hand it to every configuration. Both quantities are computed by
            # the ENGINE's own functions -- nothing is reimplemented here.
            pre_imp = pre_sc = None
            if precomp is not None and placement == 'hybrid':
                from kirby.hybrid import (compute_feature_importance,
                                          _evaluate_allocation)
                if fold_idx not in precomp['importance']:
                    precomp['importance'][fold_idx] = compute_feature_importance(
                        tr, y_tr, method=importance)
                pre_imp = precomp['importance'][fold_idx]
                key = (fold_idx, model_name)
                if key not in precomp['scores']:
                    precomp['scores'][key] = {
                        rn: float(_evaluate_allocation(
                            tr[rn], va[rn], y_tr, y_va, model=factory(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better))
                        for rn in tr}
                pre_sc = precomp['scores'][key]

            try:
                Xtr, Xva, info = build_fold_hybrid(
                    tr, va, y_tr, bp,
                    augmentations_train=tr_aug, augmentations_val=va_aug,
                    groups=(groups[train_idx] if groups is not None else None),
                    model=factory(), metric_fn=metric_fn,
                    lower_is_better=lower_is_better,
                    precomputed_importance=pre_imp,
                    precomputed_alloc_scores=pre_sc)
                sc = StandardScaler()
                Xtr = sc.fit_transform(Xtr)
                Xva = sc.transform(Xva)
                scores.append(fit_score(Xtr, y_tr, Xva, y_va, factory,
                                        metric_fn, n_seeds))
                dims.append(int(Xtr.shape[1]))
                ai = info.get('augmentation_info') or {}
                rungs.append(info.get('augmentation_rung_scores'))
                d = decoy_counts(ai)
                noise_taken.append(d['noise'])
                echo_taken.append(d['echo'])
                kept_counts.append(sum(int(v.get('n_features', 0))
                                       for v in ai.values()))
                # What the method read off its OWN internal slice. Stored
                # beside the outer score so claimed and delivered can be
                # compared per fold rather than argued about.
                b = info.get('augmentation_baseline')
                f = info.get('augmentation_final')
                claimed.append(float(f - b)
                               if b is not None and f is not None
                               and np.isfinite(b) and np.isfinite(f)
                               else float('nan'))
            except Exception as exc:                          # noqa: BLE001
                skipped.append({'cell': label, 'arm': arm,
                                'model': model_name, 'fold': fold_idx,
                                'reason': f'{type(exc).__name__}: {exc}',
                                'traceback': traceback.format_exc()[-600:]})
                scores.append(float('nan'))

        ok = not np.all(np.isnan(scores))
        per_model[model_name] = {
            'fold_scores': scores,
            'mean': float(np.nanmean(scores)) if ok else float('nan'),
            'mean_dim': float(np.mean(dims)) if dims else float('nan'),
            'mean_kept': float(np.mean(kept_counts)) if kept_counts else 0.0,
            'claimed_per_fold': claimed,
            # Every rung the ladder scored, per fold. This is what
            # lets the margin be read off instead of re-fitted.
            'rung_scores_per_fold': rungs,
            'decoy_noise_per_fold': noise_taken,
            'decoy_echo_per_fold': echo_taken,
            'status': 'ok' if ok else 'all_folds_failed',
        }
        if ok:
            dec = ('' if args.no_decoys or not pool else
                   f"  decoys=noise {sum(noise_taken)}/echo {sum(echo_taken)}")
            print(f'    {arm}/{model_name}: {np.nanmean(scores):.4f} '
                  f'(dim~{np.mean(dims) if dims else 0:.0f}, '
                  f'kept~{np.mean(kept_counts) if kept_counts else 0:.1f}){dec}')

    return {
        'arm': arm, 'cell': label, 'control': control,
        'placement': placement, 'budget': int(budget),
        'allocation_method': (cell.get('allocation') or HYBRID_ALLOCATION)
                             if placement == 'hybrid' else None,
        'method': cell['method'], 'source': cell['source'],
        'per_donor_cap': cell['cap'], 'candidate_ranking': cell['ranking'],
        'size_rule': cell['size'], 'fraction': cell['fraction'],
        'margin_multiple': cell['margin_mult'],
        'margin_absolute': float(margin),
        'threshold_multiple': cell['threshold_mult'],
        'threshold_absolute': (float(threshold) if threshold is not None
                               else None),
        'aug_budget': cell['aug_budget'],
        'collapsed_axes': cell['collapsed'],
        'decoys_planted': not args.no_decoys and bool(pool),
        'models': per_model,
        'blueprint': bp.to_dict(),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mode', default='full',
                    choices=['fast', 'full', 'intermediary'])
    ap.add_argument('--placement', default='both',
                    choices=['single', 'hybrid', 'both'],
                    help='extras onto the best single representation at its '
                         'full width, onto the multi-representation hybrid at '
                         'a budget, or both')
    ap.add_argument('--base-rep', default=None, dest='base_rep',
                    help='pin the base representation instead of taking the '
                         'widest one in the pool')
    ap.add_argument('--methods', default=None,
                    help=f'comma-separated, from {",".join(METHODS)}')
    ap.add_argument('--sources', default=None,
                    help=f'comma-separated donor sources, from '
                         f'{",".join(DONOR_SOURCES)}. Use it to split one '
                         f'dataset further if a task overruns the partition, '
                         f'never to drop a source from the study.')
    ap.add_argument('--caps', default=None,
                    help='comma-separated features per donor, from 5,64,off. '
                         'Same purpose as --sources.')
    ap.add_argument('--models', default=None,
                    help=f'comma-separated (default {",".join(DEFAULT_MODELS)})')
    ap.add_argument('--hybrid-budgets', default=None, dest='hybrid_budgets')
    ap.add_argument('--allocations', default=None,
                    help='comma-separated allocation methods for the HYBRID '
                         f'placement (default {HYBRID_ALLOCATION}). The nine '
                         f'the search sweeps: {",".join(ALLOCATIONS_SEARCHED)}. '
                         f'Inert on the single-representation placement.')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS,
                    dest='n_folds',
                    help='Folds. Default 3, cut from 5 by the user on '
                         '2026-08-26 for time. Re-run the winner at 5 '
                         'before quoting a number.')
    ap.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                    dest='n_seeds')
    ap.add_argument('--max-candidates', type=int, default=MAX_CANDIDATES,
                    dest='max_candidates',
                    help='length of the pooled ranked candidate list. FIXED '
                         'here; f25 is the experiment that sweeps it.')
    ap.add_argument('--no-decoys', action='store_true', dest='no_decoys',
                    help='do not plant the pure-noise and near-copy donors')
    ap.add_argument('--count-only', action='store_true', dest='count_only',
                    help='print the configuration and model-fit counts and '
                         'stop. Fits nothing.')
    ap.add_argument('--time-only', action='store_true', dest='time_only',
                    help='run one fold of the most expensive cell and report '
                         'seconds per fit')
    ap.add_argument('--smoke', action='store_true',
                    help='one fold, one model, one seed — plumbing only')
    ap.add_argument('--output-dir', dest='output_dir',
                    default=os.path.join(HERE, '..', 'results', 'framework'))
    args = ap.parse_args()

    t0 = time.time()
    out = run_dataset(args.dataset, args)
    out['wall_seconds'] = time.time() - t0

    if args.count_only:
        print(f"\n(counted in {out['wall_seconds']:.0f}s; nothing was fitted)")
        return out

    if args.time_only:
        n_fits = max(1, int(out.get('timed_fits', 1)))
        fit_s = out.get('fit_seconds', out['wall_seconds'])
        per_fit = fit_s / n_fits
        out['seconds_per_fit'] = per_fit
        print(f"\n  TIMING — {args.dataset}")
        print(f"    {fit_s:.1f}s of fitting for {n_fits} model fits "
              f"({out.get('timed_cell')})")
        print(f"    {per_fit:.2f} seconds per fit")
        print(f"    {out['wall_seconds'] - fit_s:.0f}s of that run was loading "
              f"representations and is NOT counted")
        print(f"\n  Multiply by the fit count --count-only prints for this "
              f"dataset to get its share of the grid.")

    os.makedirs(args.output_dir, exist_ok=True)
    # The scope has to be in the FILENAME. The grid submits four array tasks
    # per dataset -- one per placement and budget -- and every one of them used
    # to write f24_augmentation_admission_<dataset>.json, so three of the four
    # results were silently overwritten by whichever finished last. Any flag
    # that narrows what ran is included, so a split job cannot clobber its
    # siblings however it was split.
    scope = ''
    if args.placement != 'both':
        scope += f'__{args.placement}'
    if args.hybrid_budgets:
        scope += '__b' + args.hybrid_budgets.replace(',', '-')
    for flag in (args.methods, args.sources, args.caps):
        if flag:
            scope += '__' + flag.replace(',', '-')
    suffix = ('__count' if args.count_only else
              '__timing' if args.time_only else
              '__smoke' if args.smoke else '') + scope
    path = os.path.join(
        args.output_dir,
        f'f24_augmentation_admission_{args.dataset}{suffix}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: (
            o.item() if isinstance(o, (np.floating, np.integer))
            else o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(f'\nwrote {path}')
    return out


if __name__ == '__main__':
    main()
