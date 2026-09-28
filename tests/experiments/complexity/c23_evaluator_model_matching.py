#!/usr/bin/env python3
"""C23: Internal Evaluator–Downstream Model Matching.

Paper: Complexity
Conclusion: Matched internal evaluators (where the evaluator model family
    matches the downstream model) produce better hybrid representations than
    mismatched configurations. The gap is especially pronounced for NNs.

Dependencies: None (standalone).

Design: For each downstream model and each internal evaluator setting, run
    nested_cv_evaluate: the allocation/importance/budget search runs inside
    each outer training fold, and the winner is scored on the fold it never
    saw. Deltas between evaluator settings are then paired fold by fold.

Conditions:
    A_matched:      evaluator = downstream model type (rf->rf, xgb->xgb, etc.)
    B_slight_off:   evaluator from a similar model family
                    (rf->lightgbm, lightgbm->rf, dnn->dnn_small).
    C_strong_off:   evaluator from a very different family
                    (rf->svm, lightgbm->svm, dnn->knn).
    D_nn_size:      NN size mismatch, dnn scored by a wider dnn
                    (dnn->dnn_wide). The loop skips any pair whose downstream
                    model is not in MATCH_TABLE.
    E_quick_set:    evaluator always from {rf, lightgbm, xgboost}

Reports the held-out score delta (matched - mismatched), paired per outer
fold, per model x dataset x condition.

Usage:
    python c23_evaluator_model_matching.py --dataset herg --mode intermediary
    python c23_evaluator_model_matching.py --dataset caco2 --mode full
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_model_factories, get_domain_metric,
    DOWNSTREAM_MODELS, FULL_BUDGETS)
from kirby.find_best_hybrid import nested_cv_evaluate


# Debug only, off unless KIRBY_C23_STACKDUMP is set to a number of seconds.
# c23 runs every arm with verbose=False, so a slow arm prints nothing at all
# between "A_matched (eval=rf)..." and its score. This writes the interpreter
# stack to stderr on that interval so a stalled arm says where it is standing.
# Costs nothing when the variable is unset.
def _arm_stack_sampler():
    _every = os.environ.get('KIRBY_C23_STACKDUMP', '').strip()
    if not _every:
        return
    import faulthandler
    faulthandler.dump_traceback_later(float(_every), repeat=True, exit=False)


# ── Evaluator mismatch configurations ──
# For each downstream model, define which internal evaluator to use per condition.
# Note: dnn uses itself as its A_matched evaluator (_evaluate_allocation in
# hybrid.py has an explicit branch for it).
#
# Cut to the study's five downstream models on 2026-09-10 (owner decision).
# The table used to carry eleven rows — ridge, knn, mlp, gp, catboost and cnn
# as well. Those six are tier 'extra' or 'off' in DEFAULT_MODELS, so none of
# them is a downstream model of this study any more. Re-adding one means
# writing its own row here with a stated near and far family.
#
# ridge is gone from BOTH sides. As a downstream model it is tier 'off'
# (find_best_hybrid.py, 2026-09-09). As the model scoring candidates inside a
# search there is no ridge branch at all: hybrid.py's _evaluate_allocation
# falls to its else and raises "Unknown internal_evaluator: ridge". It named
# the far family for rf, lightgbm and catboost and the near family for mlp and
# svm, so six cells threw, were caught, printed FAILED and stored nothing.
# svm now stands in as the far family for rf and lightgbm, which is the choice
# xgboost's row already made. Nothing else moved.
MATCH_TABLE = {
    # model:     A_matched   B_slight_off  C_strong_off
    'rf':        ('rf',      'lightgbm',   'svm'),
    'lightgbm':  ('lightgbm','rf',         'svm'),
    'xgboost':   ('xgboost', 'lightgbm',   'svm'),
    'svm':       ('svm',     'gp',         'lightgbm'),
    # The near-family evaluator for dnn was sklearn's mlp until 2026-09-18,
    # when mlp was cut from the suite. It is now the same network at (64, 32).
    # That is a closer neighbour than mlp ever was: mlp differed from dnn in the
    # optimiser, the early-stopping rule and the initialisation as well as the
    # width, so the B_slight_off delta for dnn mixed four changes together.
    'dnn':       ('dnn',     'dnn_small',  'knn'),
}

# NN size mismatch pairs — Condition D.
#
# It was ('mlp', 'dnn') — a small sklearn network downstream, scored by this
# repository's larger one. It never ran: mlp left the downstream set on
# 2026-09-04 and the loop below skips any downstream model not in MATCH_TABLE.
# mlp was cut from the suite entirely on 2026-09-18.
#
# The question it asked stands, and dnn can ask it on its own. dnn is downstream
# at its shipped (256, 128, 64) and is scored by a dnn at (512, 256, 128). One
# thing differs between the two networks, the width, where mlp against dnn
# differed in four.
#
# This is NOT the same cell as B_slight_off for dnn. B scores the same
# downstream network with a SMALLER evaluator at (64, 32); D scores it with a
# LARGER one. The two are mirrored, so the neural row now reads capacity in both
# directions rather than one.
NN_PAIRS = [
    ('dnn', 'dnn_wide'),   # shipped NN downstream, wider NN evaluator
]

# Quick default set — Condition E
QUICK_EVALUATORS = ['rf', 'lightgbm', 'xgboost']


# ── Two local-run knobs. Both default to the shipped behaviour. ──────────────
#
# WHY THEY EXIST. c23 has never written a result file on the laptop grid. What
# it asks for, counted in model fits:
#
#   A/B/C arm  1 evaluator  x 2 budgets x 3 allocation methods =  6 configs
#   E arm      3 evaluators x 2 budgets x 3 allocation methods = 18 configs
#
# and every config is fitted on 3 inner folds inside each of 3 outer folds, so
# an A/B/C arm is 54 model fits and an E arm is 162. With the two downstream
# models the local grid leaves (rf, svm) that is 6 A/B/C arms + 2 E arms =
# 648 model fits, before the per-outer-fold baselines and rebuilds.
#
# Measured on flab_thermo_tresanco (672 antibodies, 5 representations capped at
# 1000 features each) on 2026-09-25: one random forest fit at 448 training rows
# x 250 features takes 3.5s on this machine, so the 648 fits alone are 38
# minutes, and the first A/B/C arm had not finished after 24 minutes of wall
# clock. Nothing was broken — the laptop is a 4-core 1.4 GHz i5 and seven to
# eleven other experiment processes were running at the same time.
#
# ⛔ NEITHER KNOB IS A FINDING. Nothing run under them says which evaluator
# family wins. They are wall-clock cuts for a laptop; a cluster run leaves
# KIRBY_LOCAL_GRID unset and gets the shipped conditions and the shipped
# budgets.

def _c23_conditions():
    """Which of A/B/C/D/E to run. Comma list of letters, or 'all'.

    KIRBY_C23_CONDITIONS wins wherever it is set. Unset, every condition runs
    — EXCEPT under KIRBY_LOCAL_GRID, where E is dropped.

    WHAT DROPPING E COSTS. E is the only condition that asks whether the
    shipped three-evaluator default set {rf, lightgbm, xgboost} beats naming
    the matched evaluator, so that question goes unanswered locally. It is the
    condition to drop first because it is the most expensive one by a long way:
    its search carries 3 evaluators where A, B and C each carry 1, so the two E
    arms are 324 of c23's 648 model fits — half the experiment's whole cost for
    one row of five. A, B and C, the matched-vs-mismatched deltas c23 exists
    for, are untouched.
    """
    raw = os.environ.get('KIRBY_C23_CONDITIONS', '').strip()
    if not raw:
        return {'A', 'B', 'C', 'D'} if os.environ.get(
            'KIRBY_LOCAL_GRID') else {'A', 'B', 'C', 'D', 'E'}
    if raw.lower() == 'all':
        return {'A', 'B', 'C', 'D', 'E'}
    return {t.strip().upper()[:1] for t in raw.split(',') if t.strip()}


def _c23_budgets(shipped):
    """The budget list every arm searches.

    KIRBY_C23_BUDGETS wins wherever it is set. Unset, the shipped list is used
    — EXCEPT under KIRBY_LOCAL_GRID, where only the widest budget is kept.

    WHAT KEEPING ONE BUDGET COSTS. Each arm's recipe is then chosen from 3
    configurations instead of 6, so every condition's score comes out of a
    narrower search, and the per-budget rows nested_cv_evaluate stores carry
    one budget instead of two. It does NOT bend any comparison c23 reports:
    the narrowing is identical in A, B, C and E, and c23's published quantity
    is the fold-by-fold delta between two conditions, never a
    budget-against-budget difference. (c09, c16, c17 and c18 DO compare across
    budgets and must keep both levels; c23 is not one of them.)

    250 rather than 100, from the flab_thermo_tresanco precompute of
    2026-09-25: rf scored 0.6443 spearman at budget 250 against 0.6041 at 100,
    and svm 0.6320 at 250 against 0.6351 at 100. So 250 is much the better of
    the two for rf and all but level for svm.

    ⛔ THIS CUTS CONFIGURATIONS, NEVER ROWS. All 672 antibodies are used in
    every fold either way.
    """
    raw = os.environ.get('KIRBY_C23_BUDGETS', '').strip()
    if raw:
        return [int(b.strip()) for b in raw.split(',') if b.strip()]
    if os.environ.get('KIRBY_LOCAL_GRID') and len(shipped) > 1:
        return [max(int(b) for b in shipped)]
    return list(shipped)


# The study's downstream models, one list shared with every other experiment.
TEST_MODELS = list(DOWNSTREAM_MODELS)


def _run_single_config(reps, labels, groups, model_name, model_factory,
                       evaluator, budgets,  metric_name, task_type,
                       lower_is_better, augmentations, aug_strategy,
                       dataset_name=None, n_outer_folds=5):
    """Held-out score for one (model, evaluator) pair, via nested CV.

    Each outer fold runs its own allocation/importance/budget search on that
    fold's training rows under this one internal evaluator, then scores the
    winner on the fold the search never saw. `score_mean` below is the mean of
    those outer-fold scores.

    Before, this returned the winning configuration's own inner-CV mean —
    maximised over the same folds it was then reported on. Every
    matched-minus-mismatched delta was a difference of two such maxima, so it
    was inflated on both sides before the subtraction.
    """
    models = {model_name: model_factory()}
    _t0 = time.time()
    try:
        ncv = nested_cv_evaluate(
            reps, labels, groups,
            n_outer_folds=n_outer_folds,
            # The real dataset name, so the per-dataset budget cap matches.
            # This used to be f"{dataset_name}_{model_name}_{evaluator}", and
            # the cap and the per-dataset model skips are looked up BY NAME, so
            # neither ever matched: on the 22 capped datasets c23 searched
            # budgets the dataset is capped away from. The namespacing those
            # made-up names were doing now rides on condition_suffix, which is
            # appended to `tier` and so reaches every cache layer and the stored
            # row. Belt and braces in any case: `model` and `internal_evaluator`
            # are already per-row columns, so two of these calls could not have
            # shared a row. Dict-form `models` skips the policy's model list, so
            # no arm of c23 can be dropped by this change.
            dataset_name=dataset_name,
            condition_suffix=f"c23_{model_name}_{evaluator}",
            budgets=budgets,
            models=models,
            internal_evaluators=[evaluator],
            metric=metric_name,
            task_type=task_type,
            cache=True,
            verbose=False,
            augmentations=augmentations,
            augmentation_strategy=aug_strategy,
            tier='general')
        print(f"[c23 timing] {model_name}/{evaluator} nested_cv {time.time()-_t0:.1f}s",
              flush=True)
        entry = (ncv.by_model or {}).get(model_name)
        if entry:
            rep_row = ncv.best_by_model.get(model_name, {})
            return {
                # Mean over the outer folds. Named score_mean so existing
                # readers keep working; it is no longer a search-side number.
                'score_mean': entry.get('hybrid_mean'),
                'score_std': entry.get('hybrid_std'),
                'fold_scores': list(entry.get('hybrid_fold_scores') or []),
                'score_basis': 'mean over outer folds (nested CV)',
                # What the winner's own inner mean was, and how far above the
                # outer-fold score it sat: picking on the folds you report on.
                'search_mean': entry.get('search_mean'),
                'selection_bias': entry.get('bias'),
                'n_features': rep_row.get('n_features') or 0,
                'budget': rep_row.get('budget') or 0,
                'allocation_method': rep_row.get('allocation_method') or '',
                'importance_method': rep_row.get('importance_method') or '',
                'recipe_is_from_fold': 'first outer fold with a winner',
            }
    except Exception as e:
        print(f"      FAILED after {time.time()-_t0:.1f}s: {e}")
    return None


def _paired_fold_delta(a_scores, other_scores, sign):
    """Fold-by-fold (matched - other), averaged over folds both scored.

    Both sides come from the same outer folds, so the difference is paired.
    Returns (delta, n_paired_folds) — (nan, 0) when nothing lines up.
    """
    a = (a_scores or {}).get('fold_scores') or []
    b = (other_scores or {}).get('fold_scores') or []
    paired = [sign * (x - y) for x, y in zip(a, b)
              if x is not None and y is not None
              and not np.isnan(x) and not np.isnan(y)]
    if not paired:
        return float('nan'), 0
    return float(np.mean(paired)), len(paired)


def run(data, config, output_dir, dataset_name, **kwargs):
    """C23: Evaluator-Model Matching."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C23: EVALUATOR-MODEL MATCHING")
    print("=" * 70)

    _arm_stack_sampler()

    reps = data['reps']
    labels = data['labels']
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)
    augmentations = data.get('augmentations') or None
    # CXA-15: `data` never carries an 'augmentation_strategy' key (no loader sets
    # it), so the old `data.get('augmentation_strategy', 'uncapped')` was a dead
    # lookup that always returned 'uncapped' — even when there are NO
    # augmentations. Passing augmentation_strategy='uncapped' with
    # augmentations=None writes a DIFFERENT find_best_hybrid cache key than the
    # canonical 'none' used by precompute / the other experiments, so c23 would
    # silently miss the cache and recompute. Derive it from whether
    # augmentations actually exist (mirrors c24:165 and get_precomputed_search).
    aug_strategy = 'uncapped' if augmentations else 'none'

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_outer_folds = config['n_splits']
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Outer folds: {n_outer_folds} — each runs its own recipe search "
          f"on its training rows, so every score below is held out.")

    all_models = get_model_factories(TEST_MODELS, task_type=task_type,
                                     n_samples=n_samples)
    available_models = {k: v for k, v in all_models.items() if k in MATCH_TABLE}
    print(f"  Models: {list(available_models.keys())}")

    budgets = _c23_budgets(FULL_BUDGETS)
    conditions = _c23_conditions()
    if list(budgets) != list(FULL_BUDGETS):
        print(f"  Budgets: {budgets} (shipped list is {list(FULL_BUDGETS)}; "
              f"see _c23_budgets)")
    if conditions != {'A', 'B', 'C', 'D', 'E'}:
        print(f"  Conditions running: {sorted(conditions)} of A,B,C,D,E "
              f"(see _c23_conditions)")
    sign = -1 if lower_is_better else 1

    results = []

    # ── Conditions A, B, C: matched / slight off / strong off ──
    for model_name, model_factory in available_models.items():
        matched_ev, slight_ev, strong_ev = MATCH_TABLE[model_name]
        print(f"\n  {model_name}: matched={matched_ev}, slight={slight_ev}, strong={strong_ev}")

        scores = {}
        for cond_name, evaluator in [('A_matched', matched_ev),
                                      ('B_slight_off', slight_ev),
                                      ('C_strong_off', strong_ev)]:
            if cond_name[0] not in conditions:
                print(f"    {cond_name}: not in KIRBY_C23_CONDITIONS, skipped")
                continue
            print(f"    {cond_name} (eval={evaluator})...", end=' ', flush=True)
            r = _run_single_config(
                reps, labels, groups, model_name, model_factory,
                evaluator, budgets, metric_name, task_type,
                lower_is_better, augmentations, aug_strategy,
                dataset_name=dataset_name, n_outer_folds=n_outer_folds)
            if r:
                scores[cond_name] = r
                print(f"score={r['score_mean']:.4f}")
            else:
                print("no result")

        # Deltas, paired fold by fold. Both conditions ran on the same outer
        # folds, so subtracting per fold and averaging is the paired quantity;
        # the difference of the two means diverges from it as soon as either
        # side drops a fold.
        if 'A_matched' in scores:
            a_score = scores['A_matched']['score_mean']
            for cond in ['B_slight_off', 'C_strong_off']:
                if cond in scores:
                    d, n_pair = _paired_fold_delta(
                        scores['A_matched'], scores[cond], sign)
                    scores[cond]['delta_vs_matched'] = d
                    scores[cond]['n_paired_folds'] = n_pair
                    scores[cond]['delta_of_means'] = (
                        sign * (a_score - scores[cond]['score_mean']))

        results.append({
            'model': model_name,
            'condition_type': 'ABC',
            'evaluators': {
                'A_matched': matched_ev,
                'B_slight_off': slight_ev,
                'C_strong_off': strong_ev,
            },
            'scores': scores,
        })

    # ── Condition D: NN size mismatch ──
    print(f"\n  NN size mismatch pairs:")
    # A pair whose downstream model is not in this run is skipped. It used to
    # be skipped in silence while conditions_run still listed 'D', so a result
    # file with no D row read as "D ran and produced nothing". Each skipped
    # pair is now printed and written to the result.
    nn_pairs_skipped = []
    for downstream, evaluator in (NN_PAIRS if 'D' in conditions else []):
        if downstream not in available_models:
            print(f"    {downstream} -> eval={evaluator}: skipped, {downstream} "
                  f"is not a downstream model in this run "
                  f"({list(available_models.keys())})")
            nn_pairs_skipped.append({
                'downstream': downstream, 'evaluator': evaluator,
                'reason': 'downstream model not in this run'})
            continue
        model_factory = available_models[downstream]
        print(f"    {downstream} -> eval={evaluator}...", end=' ', flush=True)
        r = _run_single_config(
            reps, labels, groups, downstream, model_factory,
            evaluator, budgets, metric_name, task_type,
            lower_is_better, augmentations, aug_strategy,
            dataset_name=dataset_name, n_outer_folds=n_outer_folds)
        if r:
            print(f"score={r['score_mean']:.4f}")
        else:
            print("no result")

        # Find the matched result for comparison
        matched_entry = None
        for prev in results:
            if prev['model'] == downstream and prev['condition_type'] == 'ABC':
                s = prev['scores'].get('A_matched')
                if s:
                    matched_entry = s

        delta = None
        n_pair = 0
        if matched_entry is not None and r is not None:
            delta, n_pair = _paired_fold_delta(matched_entry, r, sign)

        results.append({
            'model': downstream,
            'condition_type': 'D_nn_size',
            'evaluators': {'downstream': downstream, 'evaluator': evaluator},
            'scores': {'D_nn_size': r} if r else {},
            'delta_vs_matched': delta,
            'n_paired_folds': n_pair,
        })

    # ── Condition E: Quick default set ──
    # For rf, lightgbm and xgboost the quick set CONTAINS that model's own
    # matched evaluator, so E searches a superset of A. On the old same-fold
    # numbers that alone forced its delta to be zero or negative. Each outer
    # fold now picks its evaluator on its own training rows and is scored on
    # rows it never saw, so the sign is no longer forced — the containment is
    # still recorded per row as evaluator_set_contains_matched.
    print(f"\n  Quick default set: {QUICK_EVALUATORS}")
    for model_name, model_factory in (available_models.items()
                                      if 'E' in conditions else []):
        # Run with the quick evaluator set (all 3 evaluated, best wins)
        models = {model_name: model_factory()}
        print(f"    {model_name}...", end=' ', flush=True)
        _tq = time.time()
        try:
            ncv = nested_cv_evaluate(
                reps, labels, groups,
                n_outer_folds=n_outer_folds,
                dataset_name=dataset_name,
                condition_suffix=f"c23quick_{model_name}",
                budgets=budgets,
                models=models,
                internal_evaluators=QUICK_EVALUATORS,
                metric=metric_name,
                task_type=task_type,
                cache=True,
                verbose=False,
                augmentations=augmentations,
                augmentation_strategy=aug_strategy,
                tier='general')
            print(f"[c23 timing] E_quick/{model_name} nested_cv "
                  f"{time.time()-_tq:.1f}s", flush=True)
            entry = (ncv.by_model or {}).get(model_name)
            if entry:
                rep_row = ncv.best_by_model.get(model_name, {})
                score_info = {
                    'score_mean': entry.get('hybrid_mean'),
                    'score_std': entry.get('hybrid_std'),
                    'fold_scores': list(entry.get('hybrid_fold_scores') or []),
                    'score_basis': 'mean over outer folds (nested CV)',
                    'search_mean': entry.get('search_mean'),
                    'selection_bias': entry.get('bias'),
                    'n_features': rep_row.get('n_features') or 0,
                    'winning_evaluator': rep_row.get('internal_evaluator') or '',
                    'winning_evaluator_is_from_fold':
                        'first outer fold with a winner',
                    'evaluator_set_contains_matched': bool(
                        MATCH_TABLE[model_name][0] in QUICK_EVALUATORS),
                }
                print(f"score={score_info['score_mean']:.4f} "
                      f"(winner={score_info['winning_evaluator']})")

                # Delta vs matched, paired fold by fold.
                matched_entry = None
                for prev in results:
                    if prev['model'] == model_name and prev['condition_type'] == 'ABC':
                        s = prev['scores'].get('A_matched')
                        if s:
                            matched_entry = s
                if matched_entry is not None:
                    d, n_pair = _paired_fold_delta(matched_entry, score_info, sign)
                    score_info['delta_vs_matched'] = d
                    score_info['n_paired_folds'] = n_pair

                results.append({
                    'model': model_name,
                    'condition_type': 'E_quick_set',
                    'evaluators': QUICK_EVALUATORS,
                    'scores': {'E_quick_set': score_info},
                })
            else:
                print("no result")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    abc_results = [r for r in results if r['condition_type'] == 'ABC']
    if abc_results:
        print(f"\n  {'model':<12s} {'A_matched':>10s} {'B_slight':>10s} {'C_strong':>10s} {'delta_B':>8s} {'delta_C':>8s}")
        print(f"  {'-'*60}")
        for r in abc_results:
            s = r['scores']
            a = s.get('A_matched', {}).get('score_mean', float('nan'))
            b = s.get('B_slight_off', {}).get('score_mean', float('nan'))
            c = s.get('C_strong_off', {}).get('score_mean', float('nan'))
            db = s.get('B_slight_off', {}).get('delta_vs_matched', float('nan'))
            dc = s.get('C_strong_off', {}).get('delta_vs_matched', float('nan'))
            print(f"  {r['model']:<12s} {a:>10.4f} {b:>10.4f} {c:>10.4f} {db:>+8.4f} {dc:>+8.4f}")

        # Aggregate
        all_b_deltas = [r['scores'].get('B_slight_off', {}).get('delta_vs_matched')
                        for r in abc_results]
        all_c_deltas = [r['scores'].get('C_strong_off', {}).get('delta_vs_matched')
                        for r in abc_results]
        # _paired_fold_delta returns NaN, not None, when no fold paired, so a
        # None test let one unpaired model turn the whole printed mean into NaN.
        b_deltas = [d for d in all_b_deltas
                    if d is not None and np.isfinite(d)]
        c_deltas = [d for d in all_c_deltas
                    if d is not None and np.isfinite(d)]
        if b_deltas:
            print(f"\n  Mean delta (matched - slight_off):  {np.mean(b_deltas):+.4f}")
        if c_deltas:
            print(f"  Mean delta (matched - strong_off):  {np.mean(c_deltas):+.4f}")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'models_tested': list(available_models.keys()),
        'budgets': budgets,
        'n_outer_folds': n_outer_folds,
        # What this run actually asked for, so a result file can never be read
        # as the shipped conditions when it is not. See _c23_conditions and
        # _c23_budgets for what each omission costs.
        'conditions_run': sorted(conditions),
        'conditions_shipped': ['A', 'B', 'C', 'D', 'E'],
        # D pairs asked for but not run, because their downstream model is not
        # in this run. Empty when every pair ran or D was not asked for.
        'nn_pairs_skipped': nn_pairs_skipped,
        # The list _c23_budgets was handed, i.e. before its own cut. On a
        # cluster run that IS the shipped list; locally _shared has already
        # replaced it with KIRBY_LOCAL_BUDGETS.
        'budgets_before_c23_cut': list(FULL_BUDGETS),
        # Every score_mean in results is a mean over outer folds whose recipe
        # search never saw them, and every delta_vs_matched is the fold-by-fold
        # difference of two such scores, averaged.
        'score_basis': 'mean over outer folds (nested CV)',
        'delta_basis': 'paired per outer fold, then averaged',
        'results': results,
    })


if __name__ == '__main__':
    parser = create_experiment_parser('C23: Evaluator-Model Matching',
                                     has_delta=False, has_augment=False)
    args = parser.parse_args()
    config = get_config(args)

    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Reps: {args.reps}")

    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'c23_evaluator_model_matching',
                            args.dataset)
