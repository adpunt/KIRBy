#!/usr/bin/env python3
"""S09: Cross-Dataset Summary.

Paper: Similarity
Dependencies: s02, s04, s05, s06, s08 (reads their JSON outputs; fits nothing).

What it computes (rewritten 2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT.md Part 2,
chat 4; group (A) of section 2.3: an aggregator, its own selection kept).

For each dataset and each input, s09 takes "one hybrid minus one
representation" per held-out fold, in the input's own quantity:
  s02  Spearman between pair distance and pair activity difference over
       held-out pairs (SAC), per distance: each hybrid minus the
       representation s02 pairs it with (a decided model's hybrid against that
       model's rank-1 base representation; knn's distance hybrid and the
       similarity hybrid against knn's rank-1 single representation). Once
       at the budget each hybrid's own per-fold search chose, and once at
       each decided budget (s02 folds[i]['by_budget']).
  s04  the dataset's metric of predicting each held-out molecule from its
       nearest training molecules, on all, cliff and non-cliff molecules, per
       (distance, neighbour count, counting scheme): each hybrid arm minus
       knn_base1 at the same setting and the same weighting. Also each decided
       model fitted on its hybrid 1 and 2 minus the same model on its base 1.
  s05  the same kNN prediction on all held-out molecules, per (condition,
       distance, neighbour count, counting scheme): each hybrid arm minus
       single_knn1 at the same setting and condition.
  s06  Spearman between each held-out molecule's distance to its k nearest
       training molecules and a model's error on it, per (distance, neighbour
       count, error vector): each hybrid arm minus knn_base1 at the same
       weighting. Also the dataset's metric of kNN prediction on held-out
       molecules in each unweighted arm, per (distance, neighbour count,
       counting scheme): each hybrid arm minus knn_base1 at the same setting
       (s06 folds[i]['knn_held_out']).
  s08  each decided model's held-out score on its rank-1 and rank-2 hybrid at
       each decided budget minus the same model on its rank-1 base
       representation; held-out SAC and kNN prediction of knn's distance
       hybrid and the similarity hybrid at each budget minus knn's rank-1
       single representation.
Every hybrid and representation behind these numbers was chosen by searches
on that fold's training rows (the inputs record selection_scope
'fold_train_rows'); an input file without that record is not read.

Per row: the per-fold hybrid score, per-fold representation score, per-fold
difference (positive = hybrid better, in the direction the input records),
their mean over folds, smallest and largest, or "one fold, no spread". Every
kNN number carries its distance, neighbour count and counting scheme.
Rows are never averaged across datasets or metrics, and no significance test
is run (both excluded by Adelaide). Each column of the cross-dataset table
lists the dataset values side by side with each dataset's metric.

The inputs' own held-out re-picks (fold_reselection: the hybrid cell and the
representation cell re-picked for each held-out fold on the other folds) and
s05's pre-registered cell are read where the input stores them, and s08's
budget optima (picked on the numbers they report) are read with that label.

Before reading, s09 checks that the five files of one dataset were computed on
the same rows and folds. Per file: the outer fold count, the kNN settings,
the dataset metric and the decided models, and the label and group hashes
where the file carries input_fingerprint. Per fold, fold by fold: the
training and held-out row counts, and what the per-fold selection returned
(each model's base representations and hybrids with their training-row
scores, knn's 2 representations, the distance and similarity hybrids). Every
input runs the same per-fold searches on the same fold, so two splits with
equal row counts but different molecules give different training-row
scores there. No input stores the row indices, so the molecules themselves
are not compared. A fold one file lost (it failed there) is left out of that
file's comparison and named, not counted as a disagreement. What could not be
checked is named. Until 2026-10-05 this check read 'input_fingerprint' only,
which no input file in the dirty run carried, so it passed without testing
anything (DIRTY_RUN_RESULTS.md, s09).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import numpy as np

from experiments._shared import (
    save_experiment_results, load_experiment_results, fold_reselection,
    hr_exclude_folds,
)
from experiments.similarity._summary_keys import (
    ACCEPTED_INPUT_KEYS, selection_rule_of,
    accepted_names, get_input, print_refusal, refusal,
)

# Stamped into the result file. Bumped whenever what a row holds changes, so a
# reader can tell a table written from the per-fold inputs from the
# whole-dataset one written before 2026-10-05.
RESULT_VERSION = 's09_per_fold_inputs_2026_10_05b'

# Map from experiment IDs used internally to the result directory names.
# The 'expN_*' spellings written by tests/similarity_experiments.py are read too
# (see _summary_keys.ACCEPTED_INPUT_KEYS).
EXPERIMENT_IDS = list(ACCEPTED_INPUT_KEYS)

_S02 = 's02_hybrid_similarity'
_S04 = 's04_activity_cliffs'
_S05 = 's05_knn_prediction'
_S06 = 's06_applicability_domain'
_S08 = 's08_optimal_dimension'

# (name, per-dataset summary field, input). The field holds the number of
# table rows read from that input. Used by _summary_keys.refusal to name what
# each dataset lacked when nothing at all was read.
METRIC_FIELDS = [
    ('SAC_delta', 'sac_delta', _S02),
    ('cliff_delta', 'cliff_delta', _S04),
    ('kNN_delta', 'knn_delta', _S05),
    ('AD_corr_delta', 'ad_corr_delta', _S06),
    ('s08_delta', 's08_delta', _S08),
]
HELDOUT_METRIC_FIELDS = [
    ('cliff_delta_heldout', 'cliff_delta_heldout', _S04),
    ('kNN_delta_heldout', 'knn_delta_heldout', _S05),
    ('kNN_delta_fixed_ref', 'knn_delta_fixed_ref', _S05),
    ('AD_corr_delta_heldout', 'ad_corr_delta_heldout', _S06),
    ('SAC_delta_heldout_s08', 'sac_delta_heldout_s08', _S08),
]

# ── How the number in a row was selected ─────────────────────────────────────
# Every input now chooses its hybrids and representations on each fold's
# training rows. What still differs is whether anything was then picked on
# the held-out numbers themselves.
_ARM_FOLD_TRAIN = 'chosen on training rows, scored held out'
_ARM_HELD_OUT = 'held-out re-pick'
_ARM_PREREGISTERED = 'pre-registered cell, scored held out'
_ARM_OPTIMISTIC = 'selection-optimistic'

_ARM_LEGEND = {
    _ARM_FOLD_TRAIN: (
        "the hybrid and the representation were chosen by searches on the "
        "fold's training rows; the row is read at a stated kNN setting, "
        "budget or distance, and nothing was picked on the held-out scores"),
    _ARM_HELD_OUT: (
        'a cell (kNN setting, budget, arm) is re-picked for each held-out fold '
        "on the other folds' held-out scores, on both sides "
        '(experiments._shared.fold_reselection)'),
    _ARM_PREREGISTERED: (
        'a cell fixed before the run (s05: distance hybrid against '
        "knn's best single representation, both unweighted, the group's first "
        'distance, 5 neighbours, rank counting); scored on held-out rows'),
    _ARM_OPTIMISTIC: (
        'picked as the best over a grid of the same held-out numbers it '
        'reports'),
}


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _finite(v):
    """True for a real, finite number."""
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _num(v):
    return float(v) if _finite(v) else None


def _paired(hyb, sing, lower_is_better):
    """Per-fold difference, positive = hybrid better, and its spread.

    hyb, sing: per-fold scores, fold-aligned, None or NaN where absent. A fold
    counts only where both sides are finite. No interval, no test.
    """
    n = max(len(hyb), len(sing))
    hyb = list(hyb) + [None] * (n - len(hyb))
    sing = list(sing) + [None] * (n - len(sing))
    deltas = []
    for h, s in zip(hyb, sing):
        if _finite(h) and _finite(s):
            deltas.append(float(s) - float(h) if lower_is_better
                          else float(h) - float(s))
        else:
            deltas.append(None)
    ok = [d for d in deltas if d is not None]
    return {
        'hybrid_per_fold': [_num(v) for v in hyb],
        'single_per_fold': [_num(v) for v in sing],
        'delta_per_fold': deltas,
        'n_folds_paired': len(ok),
        'delta_mean_of_folds': float(np.mean(ok)) if ok else None,
        'delta_min': float(min(ok)) if ok else None,
        'delta_max': float(max(ok)) if ok else None,
        'spread_note': 'one fold, no spread' if len(ok) == 1 else (
            'no fold has both scores' if not ok else None),
    }


def _row(*, input_id, quantity, metric, lower_is_better, hybrid_arm,
         single_arm, fold_ids, hyb, sing, source, selection=_ARM_FOLD_TRAIN,
         distance=None, k=None, scheme=None, weighting=None, stratum=None,
         error_vector=None, budget=None, hybrid_per_fold_record=None,
         single_per_fold_name=None, note=None):
    """One table row: one hybrid minus one representation, per fold."""
    r = {
        'input': input_id, 'quantity': quantity, 'metric': metric,
        'lower_is_better': bool(lower_is_better),
        'hybrid_arm': hybrid_arm, 'single_arm': single_arm,
        'distance': distance, 'k': None if k is None else int(k),
        'scheme': scheme, 'weighting': weighting, 'stratum': stratum,
        'error_vector': error_vector,
        'budget_features': None if budget is None else int(budget),
        'selection': selection, 'source': source,
        'fold_ids': list(fold_ids),
    }
    r.update(_paired(hyb, sing, lower_is_better))
    if hybrid_per_fold_record is not None:
        r['hybrid_per_fold_record'] = hybrid_per_fold_record
    if single_per_fold_name is not None:
        r['single_per_fold_name'] = single_per_fold_name
    if note:
        r['note'] = note
    r['key'] = _row_key(r)
    return r


def _row_key(r):
    """The column a row belongs to in the cross-dataset table."""
    parts = [r['input'], r['quantity'],
             f"{r['hybrid_arm']} - {r['single_arm']}"]
    for name, v in (('budget', r.get('budget_features')),
                    ('distance', r.get('distance')), ('k', r.get('k')),
                    ('scheme', r.get('scheme')),
                    ('weighting', r.get('weighting')),
                    ('stratum', r.get('stratum')),
                    ('errors', r.get('error_vector'))):
        if v is not None:
            parts.append(f'{name}={v}')
    return ' | '.join(parts)


def _heldout_entry(entry):
    """A stored fold_reselection entry (one model's) as plain per-fold numbers,
    or (None, reason)."""
    if not isinstance(entry, dict):
        return None, 'key absent'
    if 'error' in entry:
        return None, f"stored an error: {entry['error']}"
    v = entry.get('delta_mean_of_folds')
    if not _finite(v):
        return None, (f"delta_mean_of_folds={v!r}, n_paired_folds="
                      f"{entry.get('n_paired_folds')}")
    arm = _ARM_HELD_OUT
    if entry.get('single_fold'):
        arm += ' (one fold: plain best, no held-out re-pick)'
    if entry.get('recipe_selection') not in (None, 'fold_train_rows'):
        arm += f" (recipe_selection={entry.get('recipe_selection')!r})"
    if entry.get('excluded_folds'):
        arm += f" (folds {sorted(entry['excluded_folds'])} dropped)"
    return {
        'selection': arm,
        'delta_mean_of_folds': float(v),
        'per_fold_delta': [_num(d) for d in entry.get('per_fold_delta') or []],
        'hybrid_fold_scores': [_num(d) for d in
                               entry.get('hybrid_fold_scores') or []],
        'baseline_fold_scores': [_num(d) for d in
                                 entry.get('baseline_fold_scores') or []],
        'hybrid_winners_per_fold': entry.get('hybrid_winners_per_fold'),
        'baseline_winners_per_fold': entry.get('baseline_winners_per_fold'),
        'n_paired_folds': entry.get('n_paired_folds'),
    }, None


def _per_fold_records(r, input_id):
    """[(fold index, record)] in fold order, for each input's own layout."""
    if input_id == _S04:
        folds = r.get('folds') or {}
        return [(int(f), folds[f]) for f in sorted(folds, key=int)]
    if input_id == _S05:
        folds = r.get('per_fold') or {}
        return [(int(f), folds[f]) for f in sorted(folds, key=int)]
    out = []
    for i, rec in enumerate(r.get('folds') or []):
        if isinstance(rec, dict):
            out.append((int(rec.get('fold', i)), rec))
    return out


# =============================================================================
# READERS, ONE PER INPUT. Each returns (rows, extras, absent).
# =============================================================================

def _read_s02(r):
    rows, absent = [], {}
    if 'lower_is_better' not in r or 'metric_name' not in r:
        return [], {}, {'sac_delta': ('the file records no lower_is_better '
                                      'or no metric_name')}
    folds = _per_fold_records(r, _S02)
    fold_ids = [f for f, _ in folds]
    lib = bool(r['lower_is_better'])
    metric = r['metric_name']

    def _collect(get_block):
        """{(comparison, distance, hybrid weighting, single weighting):
        per-fold lists}. get_block(rec) -> (comparisons, hybrids) for one
        fold; hybrids[arm]['record'] is the entry_record of the hybrid
        rebuilt there."""
        by_key = {}
        for pos, (f, rec) in enumerate(folds):
            if rec.get('failed') is not None and not rec.get('comparisons'):
                continue
            comps, hybs = get_block(rec)
            for name, c in (comps or {}).items():
                h_rec = ((hybs or {}).get(c.get('hybrid_arm')) or {}).get(
                    'record')
                for row in c.get('rows') or []:
                    key = (name, row['distance'], row['hybrid_weighting'],
                           row['single_weighting'])
                    acc = by_key.setdefault(key, {
                        'hyb': [None] * len(folds),
                        'sing': [None] * len(folds),
                        'rec': [None] * len(folds),
                        'rep': [None] * len(folds)})
                    acc['hyb'][pos] = row.get('hybrid_sac_rho')
                    acc['sing'][pos] = row.get('single_sac_rho')
                    acc['rec'][pos] = h_rec
                    acc['rep'][pos] = c.get('single_rep')
        return by_key

    def _emit(by_key, budget, where):
        for (name, dist, hw, sw), acc in sorted(by_key.items()):
            h_arm, s_arm = name.split(' - ', 1)
            rows.append(_row(
                input_id=_S02, quantity='sac_rho', metric=metric,
                lower_is_better=lib, hybrid_arm=h_arm, single_arm=s_arm,
                fold_ids=fold_ids, hyb=acc['hyb'], sing=acc['sing'],
                distance=dist, budget=budget,
                weighting=(hw if hw == sw else f'{hw} vs {sw}'),
                source=(f"s02 folds[i]{where}.comparisons['{name}'].rows "
                        f"[distance={dist}, hybrid_weighting={hw}]"),
                hybrid_per_fold_record=acc['rec'],
                single_per_fold_name=acc['rep']))

    # Each hybrid at the budget its own per-fold search chose (no budget in
    # the row: the budget in features is in hybrid_per_fold_record).
    _emit(_collect(lambda rec: (rec.get('comparisons'), rec.get('hybrids'))),
          None, '')
    # The same comparison at each decided budget (s02's budget sweep, read
    # from the same per-fold searches).
    budgets = sorted({int(b) for _, rec in folds
                      for b in (rec.get('by_budget') or {})})
    for b in budgets:
        _emit(_collect(lambda rec, _b=str(b): (
            ((rec.get('by_budget') or {}).get(_b) or {}).get('comparisons'),
            ((rec.get('by_budget') or {}).get(_b) or {}).get('hybrids'))),
            b, f".by_budget['{b}']")
    if not rows:
        absent['sac_delta'] = 'no fold stored a comparison'
    return rows, {}, absent


def _s04_arm_record(rec, arm):
    sel = rec.get('selection') or {}
    if arm.startswith('general|'):
        _, m, hj = arm.split('|')
        hy = ((sel.get('models') or {}).get(m) or {}).get('hybrids') or []
        j = int(hj[len('hybrid'):])
        return hy[j - 1] if len(hy) >= j else None
    if arm.startswith('distance|'):
        return sel.get('distance_hybrid')
    if arm.startswith('similarity|'):
        return sel.get('similarity_hybrid')
    return None


def _read_s04(r):
    rows, extras, absent = [], {}, {}
    folds = _per_fold_records(r, _S04)
    fold_ids = [f for f, _ in folds]
    if 'lower_is_better' not in r:
        return [], {}, {'cliff_delta': 'the file records no lower_is_better'}
    lib = bool(r['lower_is_better'])
    metric = r.get('metric')
    ks = r.get('knn_summary') or {}
    base = 'knn_base1'
    for arm in ks:
        if arm.startswith('knn_base'):
            continue
        for weighting in ('unweighted', 'weighted'):
            h_set = ((ks.get(arm) or {}).get(weighting) or {}).get('by_setting') or {}
            s_set = ((ks.get(base) or {}).get(weighting) or {}).get('by_setting') or {}
            for setting in sorted(h_set):
                if setting not in s_set:
                    continue
                dist, kk, scheme = setting.split('|')
                for stratum, h in h_set[setting].items():
                    s = s_set[setting].get(stratum)
                    if s is None:
                        continue
                    st_lib = True if stratum.startswith('rmse') else lib
                    st_metric = 'rmse' if stratum.startswith('rmse') else metric
                    rows.append(_row(
                        input_id=_S04, quantity='knn_prediction',
                        metric=st_metric, lower_is_better=st_lib,
                        hybrid_arm=arm, single_arm=base, fold_ids=fold_ids,
                        hyb=h.get('fold_scores') or [],
                        sing=s.get('fold_scores') or [],
                        distance=dist, k=int(kk[1:]), scheme=scheme,
                        weighting=weighting, stratum=stratum,
                        source=(f"s04 knn_summary['{arm}'|'{base}']"
                                f"['{weighting}']['by_setting']['{setting}']"
                                f"['{stratum}']['fold_scores']"),
                        hybrid_per_fold_record=[_s04_arm_record(rec, arm)
                                                for _, rec in folds],
                        single_per_fold_name=[
                            ((rec.get('knn_arms') or {}).get(base) or {})
                            .get('name') for _, rec in folds]))
    # Each decided model on its own hybrid 1 and 2 against its own base 1.
    ms = r.get('model_summary') or {}
    for m, slots in ms.items():
        b1 = (slots or {}).get('base1')
        if not b1:
            continue
        for slot in ('hybrid1', 'hybrid2'):
            h = (slots or {}).get(slot)
            if not h:
                continue
            for stratum, key in (('score_all', 'fold_scores'),
                                 ('score_cliff', 'fold_scores_cliff'),
                                 ('score_noncliff', 'fold_scores_noncliff')):
                rows.append(_row(
                    input_id=_S04, quantity='model_prediction', metric=metric,
                    lower_is_better=lib, hybrid_arm=f'{m} {slot}',
                    single_arm=f'{m} base1', fold_ids=fold_ids,
                    hyb=h.get(key) or [], sing=b1.get(key) or [],
                    stratum=stratum,
                    source=f"s04 model_summary['{m}']['{slot}'|'base1']['{key}']",
                    hybrid_per_fold_record=[
                        (((rec.get('model_arms') or {}).get(m) or {})
                         .get(slot) or {}).get('hybrid') for _, rec in folds],
                    single_per_fold_name=[
                        (((rec.get('model_arms') or {}).get(m) or {})
                         .get('base1') or {}).get('rep') for _, rec in folds]))
    if not rows:
        absent['cliff_delta'] = ('no knn_summary hybrid arm with a matching '
                                 'knn_base1 setting, and no model_summary')

    # Held-out re-pick on cliff molecules. s04 stores the per-fold scores of
    # every candidate and baseline but no re-scored result, so it is computed
    # here; pure arithmetic over stored numbers.
    block = r.get('fold_reselection')
    if not isinstance(block, dict):
        absent['cliff_delta_heldout'] = 'fold_reselection absent'
    elif not block.get('candidates_by_model') or not block.get('baselines_by_model'):
        absent['cliff_delta_heldout'] = 'no candidates or no baselines stored'
    else:
        try:
            res = fold_reselection(
                block['candidates_by_model'], block['baselines_by_model'],
                bool(block.get('lower_is_better', lib)),
                exclude_folds=hr_exclude_folds(block),
                recipe_selection=block.get('recipe_selection'))
            got = {}
            for m, entry in res.items():
                e, why = _heldout_entry(entry)
                got[m] = e if e is not None else {'not_read': why}
            extras['cliff_delta_heldout'] = {
                'metric': metric, 'lower_is_better': lib,
                'stratum': block.get('fold_score_stratum'),
                'by_model': got,
                'source': 's04 fold_reselection, re-scored by s09'}
        except Exception as e:
            absent['cliff_delta_heldout'] = (
                f'fold_reselection raised {type(e).__name__}: {e}')
    return rows, extras, absent


def _read_s05(r):
    rows, extras, absent = [], {}, {}
    folds = _per_fold_records(r, _S05)
    fold_ids = [f for f, _ in folds]
    if 'lower_is_better' not in r:
        return [], {}, {'knn_delta': 'the file records no lower_is_better'}
    lib = bool(r['lower_is_better'])
    metric = r.get('metric')
    cells = r.get('cells') or {}
    base = 'single_knn1'
    n_folds = int(r.get('n_outer_folds') or len(folds))

    def _aligned(scores):
        # s05 stores fold_scores indexed by fold number; s09 aligns them with
        # the fold records it read.
        scores = list(scores or [])
        return [scores[f] if f < len(scores) else None for f in fold_ids] \
            if fold_ids else scores[:n_folds]

    for name, c in sorted(cells.items()):
        if c.get('kind') != 'hybrid':
            continue
        bk = (f"{base}|{c['condition']}|{c['distance']}|k{c['k']}"
              f"|{c['scheme']}")
        b = cells.get(bk)
        if b is None:
            continue
        rows.append(_row(
            input_id=_S05, quantity='knn_prediction', metric=metric,
            lower_is_better=lib, hybrid_arm=c['arm'], single_arm=base,
            fold_ids=fold_ids, hyb=_aligned(c.get('fold_scores')),
            sing=_aligned(b.get('fold_scores')), distance=c['distance'],
            k=c['k'], scheme=c['scheme'], weighting=c['condition'],
            stratum='score_all',
            source=f"s05 cells['{name}'] and cells['{bk}'] fold_scores",
            hybrid_per_fold_record=[
                ((rec.get('arms') or {}).get(c['arm']) or {}).get('hybrid')
                for _, rec in folds],
            single_per_fold_name=[
                ((rec.get('arms') or {}).get(base) or {}).get('representation')
                for _, rec in folds]))
    if not rows:
        absent['knn_delta'] = ('no hybrid cell with a matching single_knn1 '
                               'cell')

    summ = r.get('summary') or {}
    fr = summ.get('fixed_ref') or {}
    if _finite(fr.get('delta_vs_best_rep')):
        extras['knn_delta_fixed_ref'] = {
            'selection': _ARM_PREREGISTERED, 'metric': metric,
            'lower_is_better': lib,
            'hybrid_cell': fr.get('hybrid_cell'),
            'single_cell': fr.get('single_cell'),
            'distance': fr.get('distance'), 'k': fr.get('k'),
            'scheme': fr.get('scheme'), 'condition': fr.get('condition'),
            'per_fold_delta': [_num(d) for d in fr.get('per_fold_delta') or []],
            'delta_mean_of_folds': float(fr['delta_vs_best_rep']),
            'n_paired_folds': fr.get('n_paired_folds'),
            'source': 's05 summary.fixed_ref'}
    else:
        absent['knn_delta_fixed_ref'] = (
            (summ.get('fixed_ref_status') or {}).get('reason')
            or 'summary.fixed_ref absent')

    ph = summ.get('posthoc_max') or {}
    if _finite(ph.get('delta_vs_best_rep')):
        extras['knn_delta_posthoc_max'] = dict(
            {k: ph.get(k) for k in ('arm', 'condition', 'distance', 'k',
                                    'scheme', 'score_mean', 'single_arm',
                                    'single_distance', 'single_k',
                                    'single_scheme', 'single_score_mean')},
            selection=_ARM_OPTIMISTIC, metric=metric, lower_is_better=lib,
            delta=float(ph['delta_vs_best_rep']),
            source='s05 summary.posthoc_max')

    k_entry = (r.get('fold_reselection_result') or {}).get('knn')
    e, why = _heldout_entry(k_entry)
    if e is None:
        st = r.get('fold_reselection_status') or {}
        if st.get('computed') is False and st.get('reason'):
            why = f"s05 fold_reselection_status: {st['reason']}"
        absent['knn_delta_heldout'] = why
    else:
        extras['knn_delta_heldout'] = dict(
            e, metric=metric, lower_is_better=lib,
            source='s05 fold_reselection_result.knn')
    return rows, extras, absent


def _read_s06(r):
    rows, extras, absent = [], {}, {}
    folds = _per_fold_records(r, _S06)
    dec = r.get('distance_error_correlation') or {}
    base = r.get('comparator_rep') or 'knn_base1'
    arms_by_fold = {f: (rec.get('arms') or {}) for f, rec in folds}
    for dist, by_k in sorted(dec.items()):
        for kk, by_src in sorted(by_k.items()):
            for src, blk in sorted(by_src.items()):
                fold_ids = list(blk.get('fold_ids') or [])
                pf = blk.get('per_fold') or {}
                for arm in sorted(pf):
                    plain = arm[:-len('_weighted')] if arm.endswith(
                        '_weighted') else arm
                    if not plain.startswith('hybrid'):
                        continue
                    weighted = arm.endswith('_weighted')
                    b_arm = f'{base}_weighted' if weighted else base
                    if b_arm not in pf:
                        continue
                    m = (src[:-len('_own_hybrid')]
                         if src.endswith('_own_hybrid') else None)
                    same_space = (m is not None
                                  and plain == f'hybrid_{m}_general1')
                    rows.append(_row(
                        input_id=_S06, quantity='distance_error_spearman',
                        metric='spearman', lower_is_better=False,
                        hybrid_arm=arm, single_arm=b_arm, fold_ids=fold_ids,
                        hyb=pf[arm], sing=pf[b_arm], distance=dist,
                        k=int(kk[1:]), weighting=('importance_weighted'
                                                  if weighted else
                                                  'unweighted'),
                        error_vector=src,
                        source=(f"s06 distance_error_correlation['{dist}']"
                                f"['{kk}']['{src}']['per_fold']"),
                        hybrid_per_fold_record=[
                            (arms_by_fold.get(f, {}).get(plain) or {})
                            .get('record') for f in fold_ids],
                        single_per_fold_name=[
                            (arms_by_fold.get(f, {}).get(base) or {})
                            .get('space') for f in fold_ids],
                        note=('the error vector comes from a model fitted on '
                              'this same hybrid, so distance and error share '
                              'one feature space' if same_space else None)))
    if not rows:
        absent['ad_corr_delta'] = (f'no hybrid arm with a matching {base} arm '
                                   'in distance_error_correlation')

    # Held-out kNN prediction in each unweighted arm, at every distance,
    # neighbour count and counting scheme s06 stored per fold
    # (folds[i]['knn_held_out']): each hybrid arm minus knn_base1 at the same
    # setting, in the dataset's metric.
    n_ad = len(rows)
    if 'lower_is_better' not in r or r.get('metric') is None:
        absent['knn_delta_s06'] = ('the file records no lower_is_better or no '
                                   'metric')
    else:
        lib, metric = bool(r['lower_is_better']), r['metric']
        fold_ids = [f for f, _ in folds]
        by_setting = {}
        for pos, (f, rec) in enumerate(folds):
            for e in rec.get('knn_held_out') or []:
                key = (e.get('arm'), e.get('distance'), e.get('k'),
                       e.get('scheme'))
                by_setting.setdefault(key, [None] * len(folds))[pos] = (
                    e.get('held_out_score'))
        for (arm, dist, kk, scheme), hyb in sorted(
                by_setting.items(), key=lambda t: tuple(map(str, t[0]))):
            if not str(arm).startswith('hybrid'):
                continue
            sing = by_setting.get((base, dist, kk, scheme))
            if sing is None:
                continue
            rows.append(_row(
                input_id=_S06, quantity='knn_prediction', metric=metric,
                lower_is_better=lib, hybrid_arm=arm, single_arm=base,
                fold_ids=fold_ids, hyb=hyb, sing=sing, distance=dist,
                k=kk, scheme=scheme, weighting='unweighted',
                stratum='score_all',
                source=(f"s06 folds[i]['knn_held_out'] [arm={arm} and "
                        f"{base}, distance={dist}, k={kk}, scheme={scheme}]"),
                hybrid_per_fold_record=[
                    (arms_by_fold.get(f, {}).get(arm) or {}).get('record')
                    for f in fold_ids],
                single_per_fold_name=[
                    (arms_by_fold.get(f, {}).get(base) or {}).get('space')
                    for f in fold_ids]))
        if len(rows) == n_ad:
            absent['knn_delta_s06'] = (f"no fold stored knn_held_out for a "
                                       f"hybrid arm and {base} at the same "
                                       f"setting")

    resel = r.get('comparator_reselected') or {}
    if 'not_computed' in resel:
        absent['ad_corr_delta_heldout'] = resel['not_computed']
    else:
        got = {}
        for dist, by_k in resel.items():
            for kk, by_src in (by_k or {}).items():
                for src, e in (by_src or {}).items():
                    if not isinstance(e, dict) or 'not_computed' in e:
                        continue
                    got[f'{dist}|{kk}|{src}'] = {
                        'selection': _ARM_HELD_OUT + (
                            ' (hybrid fixed; the single-representation arm '
                            're-picked)'),
                        'distance': dist, 'k': int(kk[1:]),
                        'error_vector': src,
                        'per_fold_delta': [_num(d) for d in
                                           e.get('per_fold_delta') or []],
                        'delta_mean_of_folds': _num(e.get('delta_mean_of_folds')),
                        'n_paired_folds': e.get('n_paired_folds'),
                        'picked_per_fold': e.get('picked_per_fold')}
        if got:
            extras['ad_corr_delta_heldout'] = {
                'metric': 'spearman', 'lower_is_better': False,
                'by_setting': got, 'source': 's06 comparator_reselected'}
        else:
            absent['ad_corr_delta_heldout'] = 'comparator_reselected is empty'
    return rows, extras, absent


def _s08_pred_record(rec, m, b, rank):
    """entry_record of model m's rank-th hybrid at budget b in one s08 fold
    (folds[i]['prediction'][m][b]['hybrids'])."""
    cells = ((((rec.get('prediction') or {}).get(m) or {}).get(str(b)) or {})
             .get('hybrids') or [])
    c = next((c for c in cells if c.get('rank') == rank), None)
    return None if c is None else c.get('hybrid')


def _s08_base_name(rows):
    """The rank-1 representation in one s08 fold's prediction_base[m] or
    neighbour_base list."""
    c = next((c for c in (rows or []) if c.get('rank') == 1), None)
    return None if c is None else c.get('representation')


def _read_s08(r):
    rows, extras, absent = [], {}, {}
    folds = _per_fold_records(r, _S08)
    fold_ids = [f for f, _ in folds]
    if 'lower_is_better' not in r:
        return [], {}, {'s08_delta': 'the file records no lower_is_better'}
    lib = bool(r['lower_is_better'])
    metric = r.get('metric')
    models = list(r.get('models') or [])

    # Prediction: each decided model's rank-1 and rank-2 hybrid at a budget
    # against the same model on its rank-1 base representation.
    pred = r.get('prediction_by_budget') or {}
    pbase = r.get('prediction_base_by_rank') or {}
    for m in models:
        b1 = ((pbase.get(m) or {}).get('1') or {}).get('fold_values')
        if b1 is None:
            continue
        for b, d in sorted((pred.get(m) or {}).items(), key=lambda t: int(t[0])):
            for rank in ('rank1', 'rank2'):
                fv = (d.get(rank) or {}).get('fold_values')
                if fv is None:
                    continue
                rows.append(_row(
                    input_id=_S08, quantity='model_prediction', metric=metric,
                    lower_is_better=lib, hybrid_arm=f'{m} hybrid {rank}',
                    single_arm=f'{m} base rank1', fold_ids=fold_ids,
                    hyb=fv, sing=b1, budget=int(b),
                    source=(f"s08 prediction_by_budget['{m}']['{b}']['{rank}'] "
                            f"vs prediction_base_by_rank['{m}']['1']"),
                    hybrid_per_fold_record=[
                        _s08_pred_record(rec, m, b, int(rank[-1]))
                        for _, rec in folds],
                    single_per_fold_name=[
                        _s08_base_name(rec.get('prediction_base', {}).get(m))
                        for _, rec in folds]))

    # Neighbour numbers: knn's distance hybrid and the similarity hybrid at
    # each budget against knn's rank-1 single representation.
    nb = r.get('neighbour_by_budget') or {}
    single = (nb.get('single_representations') or {}).get('knn_base_rank1') or {}
    knn_names = [_s08_base_name(rec.get('neighbour_base')) for _, rec in folds]

    def _fam_records(fam, b):
        # folds[i][family][budget]['hybrid'] is the entry_record of the
        # hybrid that fold built at that budget.
        return [((rec.get(fam) or {}).get(str(b)) or {}).get('hybrid')
                if isinstance((rec.get(fam) or {}).get(str(b)), dict) else None
                for _, rec in folds]
    for fam, arm in (('distance_family', 'knn distance hybrid'),
                     ('similarity_family', 'similarity hybrid')):
        for b, blk in sorted((nb.get(fam) or {}).items(), key=lambda t: int(t[0])):
            for dist, d in sorted((blk.get('by_distance') or {}).items()):
                for key in ('sac_unweighted', 'sac_weighted'):
                    h = (d.get(key) or {}).get('fold_values')
                    s = (((single.get('by_distance') or {}).get(dist) or {})
                         .get(key) or {}).get('fold_values')
                    if h is None or s is None or not any(
                            _finite(v) for v in h):
                        continue
                    rows.append(_row(
                        input_id=_S08, quantity='sac_rho', metric='spearman',
                        lower_is_better=False, hybrid_arm=arm,
                        single_arm='knn base rank1', fold_ids=fold_ids,
                        hyb=h, sing=s, budget=int(b), distance=dist,
                        weighting=key[len('sac_'):],
                        source=(f"s08 neighbour_by_budget['{fam}']['{b}']"
                                f"['by_distance']['{dist}']['{key}']"),
                        hybrid_per_fold_record=_fam_records(fam, b),
                        single_per_fold_name=knn_names))
            for setting, d in sorted((blk.get('knn') or {}).items()):
                s = ((single.get('knn') or {}).get(setting) or {}).get('fold_values')
                if s is None or d.get('fold_values') is None:
                    continue
                rows.append(_row(
                    input_id=_S08, quantity='knn_prediction', metric=metric,
                    lower_is_better=lib, hybrid_arm=arm,
                    single_arm='knn base rank1', fold_ids=fold_ids,
                    hyb=d['fold_values'], sing=s, budget=int(b),
                    distance=d.get('distance'), k=d.get('k'),
                    scheme=d.get('scheme'), weighting='unweighted',
                    source=(f"s08 neighbour_by_budget['{fam}']['{b}']['knn']"
                            f"['{setting}']"),
                    hybrid_per_fold_record=_fam_records(fam, b),
                    single_per_fold_name=knn_names))
    if not rows:
        absent['s08_delta'] = 'no prediction or neighbour block with a base side'

    # Budget optima, each picked on the numbers it reports. The ratio is the
    # distance side's best budget over each decided model's; both budgets are
    # given in features first.
    opt = r.get('optimal') or {}
    knn_b = opt.get('best_knn_budget')
    by_model = {}
    for m in models:
        mb = (opt.get('best_model_budgets') or {}).get(m)
        by_model[m] = {
            'prediction_best_budget_features': mb,
            'knn_distance_hybrid_best_budget_features': knn_b,
            'knn_best_setting': opt.get('best_knn_setting'),
            'ratio_knn_over_model': (float(knn_b) / float(mb)
                                     if knn_b and mb else None)}
    extras['optimal_budgets'] = {
        'selection': _ARM_OPTIMISTIC, 'by_model': by_model,
        'sac_best_budget_by_distance': opt.get('distance_family_sac'),
        'note': ("the distance budget with the best fold-mean kNN prediction "
                 "(over every distance, neighbour count and counting scheme) "
                 "and each decided model's best general budget; the two "
                 "budget grids differ (budget_grids in s08)"),
        'budget_grids': r.get('budget_grids'),
        'source': 's08 optimal'}

    resel = r.get('fold_reselection_by_distance') or {}
    got = {}
    for dist, by_key in resel.items():
        for key, v in (by_key or {}).items():
            e, why = _heldout_entry((v or {}).get('block'))
            got[f'{dist}|{key}'] = (dict(e, distance=dist,
                                         hybrid_weighting=key,
                                         baseline_weighting=(v or {}).get(
                                             'baseline_distance'))
                                    if e is not None else {'not_read': why})
    if any('delta_mean_of_folds' in g for g in got.values()):
        extras['sac_delta_heldout_s08'] = {
            'metric': 'spearman', 'lower_is_better': False,
            'by_distance': got,
            'source': 's08 fold_reselection_by_distance[distance][key].block'}
    else:
        absent['sac_delta_heldout_s08'] = (
            'no distance has a readable re-pick: '
            + '; '.join(f"{k}: {v.get('not_read')}" for k, v in got.items())
            if got else 'fold_reselection_by_distance absent')
    return rows, extras, absent


_READERS = {_S02: _read_s02, _S04: _read_s04, _S05: _read_s05,
            _S06: _read_s06, _S08: _read_s08}

# Which per-dataset field counts the rows of each input.
_ROW_FIELD = {_S02: 'sac_delta', _S04: 'cliff_delta', _S05: 'knn_delta',
              _S06: 'ad_corr_delta', _S08: 's08_delta'}


# =============================================================================
# DO THE FIVE FILES OF ONE DATASET DESCRIBE THE SAME ROWS AND FOLDS?
# =============================================================================

def _knn_axes_of(r):
    a = r.get('knn_axes') or r.get('knn_settings')
    if not isinstance(a, dict):
        return None
    return (a.get('group'),
            tuple(a.get('metrics') or a.get('distances') or ()),
            tuple(int(k) for k in (a.get('ks') or a.get('neighbour_counts')
                                   or ())),
            tuple(a.get('schemes') or a.get('counting_schemes') or ()))


def _dataset_metric_of(input_id, r):
    if input_id == _S02:
        m, lib = r.get('dataset_metric'), r.get('dataset_metric_lower_is_better')
    else:
        m, lib = r.get('metric'), r.get('lower_is_better')
    return None if m is None else (m, None if lib is None else bool(lib))


def _fold_rows_of(input_id, r):
    """{fold: (training rows, held-out rows)} for the folds this input
    recorded. A fold the input lost (s02 writes {'fold', 'failed'}, s06 leaves
    it out and names it in unavailable_folds, s08 writes 'error') has no row
    counts and is simply not in the dict."""
    out = {}
    for f, rec in _per_fold_records(r, input_id):
        n_tr = rec.get('n_train', rec.get('n_train_rows'))
        n_va = next((rec.get(k) for k in ('n_held_out', 'n_test', 'n_val',
                                          'n_heldout_rows')
                     if rec.get(k) is not None), None)
        if n_tr is None or n_va is None:
            continue
        out[int(f)] = (int(n_tr), int(n_va))
    return out


# Training-row scores are compared to this many significant digits, so a
# number that went through JSON once more in one input still matches.
_SCORE_DIGITS = 8


def _score_key(v):
    return None if not _finite(v) else float(f'{float(v):.{_SCORE_DIGITS}g}')


def _scores_in_order(stored, names):
    """Training-row scores of a ranked list of representations, in rank
    order, from any of the layouts the inputs use: a list of numbers (s04,
    s08), a list of {'rep', 'score'} (s02), a list of (rep, score) pairs (s06)
    or a {rep: score} dict (s05). None when the input stores none."""
    if stored is None:
        return None
    if isinstance(stored, dict):
        return tuple(_score_key(stored.get(n)) for n in names)
    out = []
    for e in stored:
        if isinstance(e, dict):
            out.append(_score_key(e.get('score')))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            out.append(_score_key(e[1]))
        else:
            out.append(_score_key(e))
    return tuple(out)


def _hybrid_key(rec):
    """A selected hybrid as (recipe label, training-row score), from its
    entry_record."""
    if not isinstance(rec, dict):
        return None
    return (rec.get('label'), _score_key(rec.get('train_rows_score')))


def _fold_selection_of(input_id, r):
    """{fold: {field: value}}: what fold_similarity_selection returned for
    each fold, as each input stored it.

    Every input calls fold_similarity_selection with the same arguments on the
    same fold, so for the same training rows the searches give the same
    representations, the same hybrids and the same training-row scores. Two
    splits with equal row counts but different molecules give different
    training-row scores. This is the only record of which molecules a fold
    held that the inputs store; no input stores the row indices.
    """
    out = {}
    for f, rec in _per_fold_records(r, input_id):
        sel = rec if input_id == _S05 else rec.get('selection')
        if not isinstance(sel, dict):
            continue
        d = {}
        for m, s in (sel.get('models') or {}).items():
            if not isinstance(s, dict):
                continue
            base = tuple(s.get('base') or ())
            d[f'{m} base representations'] = base
            sc = _scores_in_order(
                next((s[k] for k in ('base_train_rows_scores',
                                     'base_train_rows_score',
                                     'base_train_scores') if k in s), None),
                base)
            if sc is not None:
                d[f'{m} base training-row scores'] = sc
            d[f'{m} hybrids (label, training-row score)'] = tuple(
                _hybrid_key(h) for h in (s.get('hybrids') or ()))
        kb = tuple(sel.get('knn_base') or ())
        d['knn base representations'] = kb
        sc = _scores_in_order(
            next((sel[k] for k in ('knn_base_train_rows_scores',
                                   'knn_base_train_rows_score',
                                   'knn_base_train_scores', 'knn_base_scores')
                  if k in sel), None), kb)
        if sc is not None:
            d['knn base training-row scores'] = sc
        for slot in ('distance_hybrid', 'similarity_hybrid'):
            # A slot the selection left empty is a value too: one file with a
            # hybrid there and another without disagree.
            if slot in sel:
                d[f'{slot} (label, training-row score)'] = (
                    _hybrid_key(sel[slot]) or 'none selected')
        out[int(f)] = d
    return out


def _compare(vals, field, checked, disagreements, not_checked, n_present):
    """Record one compared field; vals = {input: value} (None left out)."""
    vals = {i: v for i, v in vals.items() if v is not None}
    if len(vals) < 2:
        not_checked[field] = (
            f"recorded by {len(vals)} of {n_present} input file(s)"
            + (f" ({', '.join(sorted(vals))})" if vals else ''))
        return
    checked[field] = {i: (list(v) if isinstance(v, tuple) else v)
                      for i, v in vals.items()}
    if len({repr(v) for v in vals.values()}) > 1:
        by_val = {}
        for i, v in sorted(vals.items()):
            by_val.setdefault(repr(v), []).append(i)
        disagreements.append(
            f"{field}: " + '; '.join(f"{', '.join(ins)} -> {v}"
                                     for v, ins in by_val.items()))


def inputs_agreement(present):
    """Compare what the input files of one dataset were computed on.

    present: {input id: stored result}. Returns
      {'status': 'agree' | 'disagree' | 'not_checked',
       'checked': {field: {input: value}}, 'disagreements': [str],
       'not_checked': {field: reason},
       'folds_not_compared': {input: [fold, ...]}}.

    Whole-file fields: outer fold count, kNN settings, dataset metric and
    direction, decided models, and input_fingerprint (row count, label hash,
    group hash, group count, split count) where the file carries it.

    Per-fold fields, compared fold by fold: training and held-out row counts,
    and what the per-fold selection returned (base representations, hybrids
    and their training-row scores, knn's 2 representations, the distance and
    similarity hybrids). A field is compared only on the folds that 2 or more
    inputs recorded. A fold one input lost (it failed there) is left out of
    that input's comparison and named under folds_not_compared; it is not a
    disagreement. A field recorded by fewer than 2 inputs is named under
    not_checked, so 'agree' never stands for an untested field.

    The molecules themselves are not compared: no input stores which rows a
    fold held. Equal row counts with equal training-row scores of the same
    searches is what is checked in their place.
    """
    checked, not_checked, disagreements = {}, {}, []
    n = len(present)
    whole = {
        'n_outer_folds': lambda i, r: r.get('n_outer_folds'),
        'kNN settings (group, distances, neighbour counts, schemes)':
            lambda i, r: _knn_axes_of(r),
        'dataset metric and direction': _dataset_metric_of,
        'decided models': lambda i, r: (tuple(r['models'])
                                        if r.get('models') else None),
    }
    for fp_key in ('n_samples', 'labels_hash', 'groups_hash', 'n_groups',
                   'n_splits'):
        whole[f'input_fingerprint.{fp_key}'] = (
            lambda i, r, _k=fp_key: (r.get('input_fingerprint') or {}).get(_k))
    for field, get in whole.items():
        vals = {}
        for i, r in present.items():
            try:
                vals[i] = get(i, r)
            except Exception:
                vals[i] = None
        _compare(vals, field, checked, disagreements, not_checked, n)

    per_fold = {}
    for i, r in present.items():
        try:
            rows = _fold_rows_of(i, r)
        except Exception:
            rows = {}
        try:
            sels = _fold_selection_of(i, r)
        except Exception:
            sels = {}
        per_fold[i] = {'rows': rows, 'sel': sels}
    all_folds = sorted({f for d in per_fold.values()
                        for f in set(d['rows']) | set(d['sel'])})
    folds_not_compared = {}
    for i, d in per_fold.items():
        missing = [f for f in all_folds
                   if f not in d['rows'] and f not in d['sel']]
        if missing and (d['rows'] or d['sel']):
            folds_not_compared[i] = missing
    for f in all_folds:
        _compare({i: d['rows'].get(f) for i, d in per_fold.items()},
                 f'fold {f}: (training rows, held-out rows)', checked,
                 disagreements, not_checked, n)
        sel_fields = sorted({k for d in per_fold.values()
                             for k in (d['sel'].get(f) or {})})
        for k in sel_fields:
            _compare({i: (d['sel'].get(f) or {}).get(k)
                      for i, d in per_fold.items()},
                     f'fold {f}: selection, {k}', checked, disagreements,
                     not_checked, n)
    status = ('disagree' if disagreements
              else 'agree' if checked else 'not_checked')
    return {'status': status, 'checked': checked,
            'disagreements': disagreements, 'not_checked': not_checked,
            'folds_not_compared': folds_not_compared}


# =============================================================================
# RUN
# =============================================================================

def run(all_dataset_results):
    """One table: per dataset, per input, one hybrid minus one representation
    per held-out fold. Nothing averaged across datasets; no test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT S09: CROSS-DATASET SUMMARY")
    print("=" * 70)

    results = {
        'result_version': RESULT_VERSION,
        'per_dataset': {},
        'summary': {},
        'table': [],
        'arm_legend': dict(_ARM_LEGEND),
        'delta_sign': ('every delta is hybrid minus representation in the '
                       "direction its metric records: positive = hybrid "
                       'better'),
        'no_tests': ('no significance test, p-value or interval; per-fold '
                     'differences and their spread only (Adelaide excluded '
                     'them)'),
        'never_averaged': ('no number is averaged across datasets or across '
                           'metrics; each column lists the datasets side by '
                           'side, each in its own metric'),
        # The process-wide distance is not what the inputs were computed
        # under any more: each records its own kNN settings, compared in
        # input_agreement. Set here so save_experiment_results does not stamp
        # the s09 process's own default.
        'distance_metric': ('per input: see input_agreement, field "kNN '
                            'settings"'),
        'selection_rule_by_dataset': {},
        'input_agreement': {},
        'inputs_not_read': {},
    }

    for ds_name, ds_results in all_dataset_results.items():
        print(f"\n  {ds_name}:")
        ds_summary = {}
        present, refused = {}, {}
        for cid in ACCEPTED_INPUT_KEYS:
            r = get_input(ds_results, cid)
            if not r:
                continue
            scope = r.get('selection_scope')
            if scope != 'fold_train_rows':
                # A file written before the per-fold rewrite chose its hybrids
                # with searches over every row (R2); none of its numbers
                # enter the table.
                refused[cid] = (f"selection_scope={scope!r}, not "
                                f"'fold_train_rows': written before the "
                                f"per-fold rewrite; re-run {cid[:3]}")
                continue
            present[cid] = r
        for cid, why in refused.items():
            print(f"    {cid}: not read ({why})")
        results['inputs_not_read'][ds_name] = refused
        results['selection_rule_by_dataset'][ds_name] = {
            cid: selection_rule_of(r) for cid, r in present.items()}

        agree = inputs_agreement(present)
        results['input_agreement'][ds_name] = agree
        if agree['status'] == 'disagree':
            print("    !! the input files disagree about the rows or folds "
                  "they were computed on:")
            for d in agree['disagreements']:
                print(f"       {d}")
        elif agree['status'] == 'not_checked':
            print("    !! rows and folds not checked: fewer than 2 input "
                  "files record any of the compared fields")
        else:
            print(f"    input files agree on {len(agree['checked'])} compared "
                  f"field(s), including each fold's row counts and per-fold "
                  f"selection where 2 or more files record them")
        if agree['not_checked'] and agree['status'] != 'not_checked':
            print(f"    not compared: {', '.join(agree['not_checked'])}")
        for cid, missing in sorted(agree['folds_not_compared'].items()):
            print(f"    {cid}: fold(s) {missing} absent from this file (failed "
                  f"there), left out of its comparison")

        for cid, r in present.items():
            try:
                rows, extras, absent = _READERS[cid](r)
            except Exception as e:
                rows, extras = [], {}
                absent = {_ROW_FIELD[cid]: f'reader raised {type(e).__name__}: {e}'}
            for row in rows:
                row['dataset'] = ds_name
                row['inputs_agree'] = agree['status']
            if rows:
                ds_summary[_ROW_FIELD[cid]] = len(rows)
            ds_summary.setdefault('rows_by_input', {})[cid] = [
                row['key'] for row in rows]
            for field, e in extras.items():
                ds_summary[field] = e
            for field, why in absent.items():
                ds_summary[f'{field}_status'] = {'computed': False,
                                                 'source': cid, 'reason': why}
            results['table'].extend(rows)
            _print_input(cid, rows, extras, absent)
        results['per_dataset'][ds_name] = ds_summary

    # Refuse an aggregate built from nothing (no input of any dataset gave a
    # row or a held-out read).
    _refusal = refusal(all_dataset_results, results['per_dataset'],
                       METRIC_FIELDS + HELDOUT_METRIC_FIELDS)
    if _refusal is not None:
        _refusal['inputs_not_read'] = results['inputs_not_read']
        print_refusal(_refusal)
        return _refusal

    # Columns: one per row key, listing every dataset that has it. Never a
    # mean across datasets; a column whose datasets use different metrics
    # says so.
    for row in results['table']:
        col = results['summary'].setdefault(row['key'], {
            'input': row['input'], 'quantity': row['quantity'],
            'hybrid_arm': row['hybrid_arm'], 'single_arm': row['single_arm'],
            'distance': row['distance'], 'k': row['k'],
            'scheme': row['scheme'], 'weighting': row['weighting'],
            'stratum': row['stratum'], 'error_vector': row['error_vector'],
            'budget_features': row['budget_features'],
            'selection': row['selection'], 'per_dataset': {}})
        col['per_dataset'][row['dataset']] = {
            'metric': row['metric'], 'lower_is_better': row['lower_is_better'],
            'delta_per_fold': row['delta_per_fold'],
            'delta_mean_of_folds': row['delta_mean_of_folds'],
            'delta_min': row['delta_min'], 'delta_max': row['delta_max'],
            'n_folds_paired': row['n_folds_paired'],
            'spread_note': row['spread_note'],
            'inputs_agree': row['inputs_agree']}
    for col in results['summary'].values():
        col['n_datasets'] = len(col['per_dataset'])
        metrics = sorted({str(v['metric']) for v in col['per_dataset'].values()})
        col['metrics_in_column'] = metrics
        if len(metrics) > 1:
            col['metric_note'] = (
                f"the datasets in this column are scored in {len(metrics)} "
                f"different metrics ({', '.join(metrics)}); read each "
                f"dataset in its own metric")
    print(f"\n  Table: {len(results['table'])} rows over "
          f"{len(results['per_dataset'])} dataset(s), "
          f"{len(results['summary'])} columns. Every row is in the result "
          f"file and in table.csv next to it.")
    return results


def _print_input(cid, rows, extras, absent):
    """A few lines per input: how many rows, and the held-out reads."""
    print(f"    {cid}: {len(rows)} rows (hybrid minus representation, per "
          f"fold)")
    for field, e in extras.items():
        if field == 'optimal_budgets':
            for m, d in e['by_model'].items():
                print(f"      best budget [{e['selection']}]: {m} "
                      f"{d['prediction_best_budget_features']} features, knn "
                      f"distance hybrid {d['knn_distance_hybrid_best_budget_features']} "
                      f"features ({d['knn_best_setting']})")
            continue
        if 'delta_mean_of_folds' in e and _finite(e.get('delta_mean_of_folds')):
            print(f"      {field} [{e.get('selection')}]: "
                  f"{e['delta_mean_of_folds']:+.4f} {e.get('metric') or ''} "
                  f"over {e.get('n_paired_folds')} fold(s)")
            continue
        sub = e.get('by_model') or e.get('by_setting') or e.get('by_distance') or {}
        for name, v in sub.items():
            if isinstance(v, dict) and _finite(v.get('delta_mean_of_folds')):
                print(f"      {field} {name} [{v.get('selection')}]: "
                      f"{v['delta_mean_of_folds']:+.4f} "
                      f"{e.get('metric') or ''} over "
                      f"{v.get('n_paired_folds')} fold(s)")
    for field, why in absent.items():
        print(f"      {field}: not read ({why})")


_CSV_COLUMNS = ('dataset', 'input', 'quantity', 'metric', 'lower_is_better',
                'hybrid_arm', 'single_arm', 'budget_features', 'distance',
                'k', 'scheme', 'weighting', 'stratum', 'error_vector',
                'selection', 'n_folds_paired', 'delta_mean_of_folds',
                'delta_min', 'delta_max', 'spread_note', 'delta_per_fold',
                'hybrid_per_fold', 'single_per_fold', 'single_per_fold_name',
                'inputs_agree', 'source')


def write_table_csv(rows, path):
    """The table, one line per row, per-fold lists joined with ';'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        for r in rows:
            out = []
            for c in _CSV_COLUMNS:
                v = r.get(c)
                if isinstance(v, (list, tuple)):
                    v = ';'.join('' if x is None else
                                 (f'{x:.6g}' if isinstance(x, float) else str(x))
                                 for x in v)
                out.append('' if v is None else v)
            w.writerow(out)
    print(f"  [saved {path}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S09: Cross-dataset summary')
    parser.add_argument('--datasets', nargs='+', default=['logd', 'caco2', 'herg_ki', 'qm9'],
                        help='Datasets to aggregate (default: logd caco2 herg_ki qm9)')
    parser.add_argument('--results-dir', type=str, default='tests/results',
                        help='Directory containing experiment result subdirs')
    parser.add_argument('--output', type=str, default='tests/results',
                        help='Output directory for summary JSON')
    args = parser.parse_args()

    print(f"Datasets: {args.datasets}")
    print(f"Results dir: {args.results_dir}")

    # Load results from each experiment for each dataset. A result directory
    # written under either naming scheme is accepted; it is stored under the
    # canonical sNN id.
    all_dataset_results = {}
    for ds in args.datasets:
        ds_results = {}
        for exp_id in EXPERIMENT_IDS:
            for dir_name in accepted_names(exp_id):
                r = load_experiment_results(args.results_dir, dir_name, ds)
                if r:
                    ds_results[exp_id] = r
                    print(f"  Loaded {dir_name}/{ds}.json")
                    break
            else:
                print(f"  Missing {exp_id}/{ds}.json "
                      f"(also tried {', '.join(accepted_names(exp_id)[1:])})")
        if ds_results:
            all_dataset_results[ds] = ds_results
        else:
            print(f"  WARNING: no results found for {ds}, skipping")

    if not all_dataset_results:
        print("No results found for any dataset, exiting.")
        sys.exit(1)

    results = run(all_dataset_results)
    if results.get('error'):
        # Nothing aggregated: write no result file, fail the job.
        print("\nNo result file written.")
        sys.exit(1)
    write_table_csv(results['table'], os.path.join(
        args.output, 's09_cross_dataset_summary', 'table.csv'))
    save_experiment_results(results, args.output, 's09_cross_dataset_summary')
