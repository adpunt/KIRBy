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

    Aligned 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4; selection rule
    2.3 "(B)": one hybrid per internal-evaluator condition, per outer fold, on
    the dataset's two models, no exemption, 2.3 item 16):
    - downstream models are the dataset's two decided models
      (decided_grid.decided_models(ds, 'general')), not the five
      DOWNSTREAM_MODELS. A decided model with no MATCH_TABLE row is not run
      and is listed under models_not_run with the reason.
    - every search is on the decided grid (budgets, importance methods,
      allocation methods, representation counts, speed) for 'general'. The
      internal evaluator is the one axis c23 sets itself: it is the question.
    - augmentation: only decided_aug_conditions(ds, 'general'), which is
      'none' on every dataset today.
    - pruning off (nothing passed; nested_cv_evaluate's default is unpruned).
    - outer fold count from config['n_splits'] only; 1 is one held-out split
      and every spread then says "one fold, no spread".
    - one record per outer fold per arm in 'per_fold', and each finished arm
      is written to a partial file as it finishes.

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
    load_data, get_config, create_experiment_parser, parse_rep_args,
    save_experiment_results, convert_numpy, get_domain_metric)
from kirby.find_best_hybrid import nested_cv_evaluate


# The directory save_experiment_results writes under (complexity_helpers
# _experiment_full_name maps 'c23' to the same name).
EXPERIMENT_DIR = 'c23_evaluator_model_matching'

# Part of the partial file's identity (2026-10-05). Bumped whenever what one
# arm computes changes, so units written by older code are moved aside, never
# read back. '2026-10-05_decided_grid' = the dataset's two decided models, the
# decided search grid, augmentation 'none', per-fold records.
RESULT_VERSION = '2026-10-05_decided_grid'

# Seed of the outer splitter; nested_cv_evaluate's default, named so the
# per-fold store namespace (ocv{n}f{i}s42) is visible here.
SEED = 42


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
# 2026-10-05: the downstream models are now each dataset's two decided models
# (DECISIONS.md Step 1). gp is one of the two on abagym_Ang2, abagym_VEGF,
# ginkgo_acsins, ginkgo_hic and ginkgo_pr_cho, and has no row here, so c23
# does not run gp on those five; run() lists it under models_not_run. Its
# near and far evaluator are the owner's to choose (open question, 2026-10-05).
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


# ── Two named overrides. Unset, both run the full decided design. ────────────
#
# Until 2026-10-05 both also cut the run by themselves under KIRBY_LOCAL_GRID
# (condition E dropped, one budget kept). That automatic cut is gone: a local
# "does it run" check uses normal settings and a time box. What is left only
# acts when the variable is set by hand, and the result file records what ran
# (conditions_run, budgets).
#
# ⛔ NEITHER OVERRIDE IS A FINDING. Nothing run under one says which evaluator
# family wins.

def _c23_conditions():
    """Which of A/B/C/D/E to run. Comma list of letters, or 'all'.

    KIRBY_C23_CONDITIONS wins wherever it is set. Unset, every condition runs.
    """
    raw = os.environ.get('KIRBY_C23_CONDITIONS', '').strip()
    if not raw or raw.lower() == 'all':
        return {'A', 'B', 'C', 'D', 'E'}
    return {t.strip().upper()[:1] for t in raw.split(',') if t.strip()}


def _c23_budgets(decided):
    """The budget list (in features) every arm searches.

    Unset, the dataset's decided budgets (DECISIONS.md, decided_grid). A
    KIRBY_C23_BUDGETS list is kept only where it names decided budgets;
    apply_decided_grid drops and prints any other, and raises when none is
    left. ⛔ This cuts configurations, never rows.
    """
    raw = os.environ.get('KIRBY_C23_BUDGETS', '').strip()
    if raw:
        return [int(b.strip()) for b in raw.split(',') if b.strip()]
    return list(decided)


def _search_kwargs(dataset_name, model_name, evaluators, budgets):
    """find_best_hybrid arguments for one c23 arm, on the decided grid.

    Added 2026-10-05. Every axis but the internal evaluator comes from
    decided_grid.apply_decided_grid for the dataset's 'general' search: models
    narrowed to [model_name], the decided budgets, importance methods,
    allocation methods, representation counts and speed class. The internal
    evaluator is then set to this arm's own list. It is not narrowed to the
    decided evaluators: comparing evaluators is what c23 measures, and
    apply_decided_grid would refuse svm, gp, knn, xgboost, dnn_small and
    dnn_wide, none of which DECISIONS.md keeps as an evaluator.
    """
    from experiments.decided_grid import apply_decided_grid
    kw = apply_decided_grid(
        dict(tier='general', models=[model_name], budgets=list(budgets)),
        dataset_name)
    kw['internal_evaluators'] = list(evaluators)
    return kw


def _run_single_config(reps, labels, groups, model_name, evaluators,
                       metric_name, task_type, *, dataset_name, budgets,
                       n_outer_folds, condition_suffix, condition):
    """Held-out score for one (model, evaluator set) arm, via nested CV.

    Each outer fold runs its own allocation/importance/budget search on that
    fold's training rows under this arm's internal evaluator(s), then scores
    the winner on the fold the search never saw. `score_mean` below is the
    mean of those outer-fold scores.

    Before, this returned the winning configuration's own inner-CV mean —
    maximised over the same folds it was then reported on. Every
    matched-minus-mismatched delta was a difference of two such maxima, so it
    was inflated on both sides before the subtraction.

    2026-10-05: n_outer_folds has no default (it was 5); the caller passes
    config['n_splits']. models is a one-name list from the decided grid, not a
    {name: estimator} dict, so the search's model list and its stored key name
    the same model. Augmentation is 'none' (see run()). Returns a plain dict
    (checkpointable) or {'error': ...}.
    """
    kw = _search_kwargs(dataset_name, model_name, evaluators, budgets)
    _t0 = time.time()
    try:
        ncv = nested_cv_evaluate(
            reps, labels, groups,
            n_outer_folds=n_outer_folds,
            seed=SEED,
            # The real dataset name, so the decided grid and the per-dataset
            # model skips are looked up by name. The arm is namespaced by
            # condition_suffix, which is appended to `tier` and so reaches
            # every cache layer and the stored row.
            dataset_name=dataset_name,
            condition_suffix=condition_suffix,
            metric=metric_name,
            task_type=task_type,
            cache=True,
            verbose=False,
            augmentations=None,
            augmentation_strategy='none',
            **kw)
    except Exception as e:
        print(f"      FAILED after {time.time()-_t0:.1f}s: {e}")
        return {'error': f'{type(e).__name__}: {e}'}
    print(f"[c23 timing] {condition} {model_name}/{','.join(evaluators)} "
          f"nested_cv {time.time()-_t0:.1f}s", flush=True)
    entry = (ncv.by_model or {}).get(model_name)
    if not entry:
        return {'error': f'nested CV returned no result for {model_name}'}
    rep_row = ncv.best_by_model.get(model_name, {})
    _per_fold, _first = _recipe_per_fold(entry)
    fold_scores = list(entry.get('hybrid_fold_scores') or [])
    n_scored = sum(1 for x in fold_scores
                   if x is not None and np.isfinite(x))

    # One record per outer fold: the recipe that fold's search picked, its
    # score on the fold's training rows (the winner's mean over the inner
    # folds), and its score on the held-out rows. Added 2026-10-05.
    inner = list(getattr(ncv, 'inner_searches', None) or [])
    per_fold = []
    for f, rec in enumerate(_per_fold):
        srch = inner[f] if f < len(inner) else None
        _tr = None
        if srch is not None:
            _b = (getattr(srch, 'best_by_model', None) or {}).get(model_name)
            if _b and _b.get('score_mean') is not None:
                _tr = float(_b['score_mean'])
        _sf = entry.get('best_single_fold_scores') or []
        _sr = entry.get('best_single_reps_per_fold') or []
        per_fold.append(dict(
            rec,
            model=model_name,
            condition=condition,
            evaluators_searched=list(evaluators),
            train_rows_score=_tr,
            held_out_score=(fold_scores[f] if f < len(fold_scores) else None),
            # Recorded by nested_cv_evaluate anyway: the single representation
            # with the best training-row score for this model, scored on the
            # same held-out rows. Context only; c23 compares evaluators.
            best_single_rep=(_sr[f] if f < len(_sr) else None),
            best_single_held_out_score=(_sf[f] if f < len(_sf) else None),
            selection_scope='fold_train_rows',
        ))

    return convert_numpy({
        # Mean over the outer folds. Named score_mean so existing readers keep
        # working; it is no longer a search-side number.
        'score_mean': entry.get('hybrid_mean'),
        # NaN from nested_cv_evaluate when one fold scored: no spread.
        'score_std': entry.get('hybrid_std'),
        'n_folds_scored': n_scored,
        'spread_note': (None if n_scored > 1 else 'one fold, no spread'),
        'fold_scores': fold_scores,
        'score_basis': 'mean over outer folds (nested CV)',
        # What the winner's own inner mean was, and how far above the
        # outer-fold score it sat: picking on the folds you report on.
        'search_mean': entry.get('search_mean'),
        'selection_bias': entry.get('bias'),
        # 2026-10-05: the MEAN over outer folds of the width built.
        # Per fold it is in recipe_per_fold[i]['n_features'].
        'n_features': rep_row.get('n_features') or 0,
        'n_features_is': 'mean over outer folds',
        # These are ONE fold's winner (recipe_is_from_fold), not a recipe
        # shared by the folds. Each fold's own is in recipe_per_fold.
        'budget': rep_row.get('budget') or 0,
        'allocation_method': rep_row.get('allocation_method') or '',
        'importance_method': rep_row.get('importance_method') or '',
        'winning_evaluator': rep_row.get('internal_evaluator') or '',
        'recipe_is_from_fold': _first,
        'recipe_per_fold': _per_fold,
        'per_fold': per_fold,
        'search_grid': {k: v for k, v in kw.items()
                        if k in ('models', 'budgets', 'importance_methods',
                                 'allocation_methods', 'internal_evaluators',
                                 'rep_counts', 'speed')},
        'aug_condition': 'none',
        'selection_scope': 'fold_train_rows',
    })


def _recipe_per_fold(entry):
    """Each outer fold's own winning recipe and width, from a nested-CV entry.

    Added 2026-10-05. best_by_model (find_best_hybrid's NestedCVResult) reads
    budget / allocation / importance / evaluator off the FIRST outer fold that
    produced a winner, while its n_features is the mean over every fold. Each
    fold picks its own recipe, so those fields describe one fold and the width
    describes all of them. This returns ([{fold, budget, allocation_method,
    importance_method, internal_evaluator, rep_count, aug_condition,
    n_features}, ...], index of the first fold with a winner or None).
    n_features is the width actually built on that fold (None where the fold
    had no winner or did not build).
    """
    bps = list(entry.get('blueprints_per_fold') or [])
    widths = list(entry.get('n_features_per_fold') or [])
    n = max(len(bps), len(widths))
    out, first = [], None
    for f in range(n):
        bp = bps[f] if f < len(bps) else None
        bpd = (bp.to_dict() if hasattr(bp, 'to_dict')
               else (bp if isinstance(bp, dict) else {})) or {}
        if bpd and first is None:
            first = f
        out.append({
            'fold': f,
            'budget': bpd.get('budget'),
            'allocation_method': bpd.get('allocation_method'),
            'importance_method': bpd.get('importance_method'),
            'internal_evaluator': bpd.get('internal_evaluator'),
            # 2026-10-05: how many representations the recipe kept (None =
            # the whole pool) and the augmentation condition it was built in.
            'rep_count': bpd.get('rep_count'),
            'aug_condition': bpd.get('augmentation_strategy'),
            'n_reps': (len(bpd['reps']) if bpd.get('reps') else None),
            'n_features': (widths[f] if f < len(widths) else None),
        })
    return out, first


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


def _paired_width_gap(a_scores, other_scores):
    """Per outer fold: matched arm's built width minus the other arm's.

    Added 2026-10-05 (dirty run: on buchwald_hartwig a change of evaluator
    also changed the width built, 229 against 239 features at budget 250).
    Recorded beside every paired delta so a score difference can be read with
    the width difference on the same fold. None where either fold built
    nothing.
    """
    a = [r.get('n_features') for r in (a_scores or {}).get('recipe_per_fold') or []]
    b = [r.get('n_features') for r in (other_scores or {}).get('recipe_per_fold') or []]
    return [(int(x) - int(y)) if (x is not None and y is not None) else None
            for x, y in zip(a, b)]


def _fmt(x, spec='.4f'):
    return format(x, spec) if (x is not None and np.isfinite(x)) else 'nan'


def run(data, config, output_dir, dataset_name, **kwargs):
    """C23: Evaluator-Model Matching."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C23: EVALUATOR-MODEL MATCHING")
    print("=" * 70)

    _arm_stack_sampler()

    from experiments.decided_grid import (
        decided_models, decided_search, decided_aug_conditions)
    from experiments.complexity.complexity_helpers import DATASET_REPS
    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    task_type = data['task_type']
    n_samples = len(labels)

    # Representations: the search gets data['reps'] whole (a hybrid builder
    # is never handed a filtered pool). run_all loads exactly
    # DATASET_REPS[ds]['general']; anything else loaded is named here.
    general_names = list((DATASET_REPS.get(dataset_name) or {}).get(
        'general') or [])
    extra = sorted(set(reps) - set(general_names)) if general_names else []
    if extra:
        print(f"  ⚠ representations outside DATASET_REPS['{dataset_name}']"
              f"['general'] are loaded and reach the search: {extra}")

    # Augmentation (2026-10-05): only DECISIONS.md Step 8's conditions for the
    # general search. That is 'none' on every dataset today. c23 compares
    # evaluators, not augmentation, so it runs 'none' only; a kept condition
    # other than 'none' is named in the output, never run silently.
    _aug_kept = [c for c in decided_aug_conditions(dataset_name, 'general')
                 if c != 'none']
    aug_not_run = None
    if data.get('augmentations') and _aug_kept:
        aug_not_run = {'conditions': _aug_kept,
                       'reason': 'c23 runs one augmentation condition per '
                                 "arm, 'none'"}
        print(f"  augmentation {_aug_kept} kept by DECISIONS.md for the "
              f"general search; c23 runs 'none' only")
    elif data.get('augmentations'):
        print("  augmentation features not used: DECISIONS.md Step 8 cuts "
              "augmentation on this dataset's general search")

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_outer_folds = int(config['n_splits'])
    print(f"  Metric: {metric_name} (lower_is_better={lower_is_better})")
    print(f"  Outer folds: {n_outer_folds} — each runs its own recipe search "
          f"on its training rows, so every score below is held out."
          + (" One fold: a single held-out split, no spread."
             if n_outer_folds == 1 else ""))

    # Models (2026-10-05): the dataset's two decided models. knn is never a
    # downstream model here. A decided model with no MATCH_TABLE row (gp) has
    # no near or far evaluator written down, so it is not run and is listed.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    run_models = [m for m in decided if m in MATCH_TABLE]
    models_not_run = [
        {'model': m, 'reason': 'no MATCH_TABLE row (near and far evaluator '
                               'not chosen)'}
        for m in decided if m not in MATCH_TABLE]
    print(f"  Models (decided): {decided}; running {run_models}")
    for m in models_not_run:
        print(f"  NOT RUN {m['model']}: {m['reason']}")

    decided_budgets = list(decided_search(dataset_name, 'general')['budgets'])
    budgets = _c23_budgets(decided_budgets)
    conditions = _c23_conditions()
    print(f"  Budgets: {budgets} features"
          + ("" if budgets == decided_budgets
             else f" (decided list is {decided_budgets}; KIRBY_C23_BUDGETS)"))
    if conditions != {'A', 'B', 'C', 'D', 'E'}:
        print(f"  Conditions running: {sorted(conditions)} of A,B,C,D,E "
              f"(KIRBY_C23_CONDITIONS)")
    sign = -1 if lower_is_better else 1

    # One unit per arm (model x condition), written as it finishes.
    ck = Checkpoint(
        experiment_sidecar(output_dir, EXPERIMENT_DIR, dataset_name),
        run_identity(EXPERIMENT_DIR, dataset_name, data, config,
                     result_version=RESULT_VERSION))

    def _arm(model_name, cond_name, evaluators, suffix):
        if ck.done(model=model_name, condition=cond_name):
            print(f"    {cond_name} {model_name}: finished in an earlier "
                  f"attempt, read back")
            r = ck.get(model=model_name, condition=cond_name)
        else:
            print(f"    {cond_name} {model_name} (eval={','.join(evaluators)})"
                  f"...", flush=True)
            r = _run_single_config(
                reps, labels, groups, model_name, evaluators, metric_name,
                task_type, dataset_name=dataset_name, budgets=budgets,
                n_outer_folds=n_outer_folds, condition_suffix=suffix,
                condition=cond_name)
            ck.put(r, model=model_name, condition=cond_name)
        if 'error' in r:
            print(f"    {cond_name} {model_name}: no result ({r['error']})")
            return None
        print(f"    {cond_name} {model_name}: score={_fmt(r['score_mean'])} "
              f"width={r['n_features']} features (mean over folds)")
        return r

    results = []
    failures = []

    # ── Conditions A, B, C: matched / slight off / strong off ──
    for model_name in run_models:
        matched_ev, slight_ev, strong_ev = MATCH_TABLE[model_name]
        print(f"\n  {model_name}: matched={matched_ev}, slight={slight_ev}, strong={strong_ev}")

        scores = {}
        for cond_name, evaluator in [('A_matched', matched_ev),
                                      ('B_slight_off', slight_ev),
                                      ('C_strong_off', strong_ev)]:
            if cond_name[0] not in conditions:
                print(f"    {cond_name}: not in KIRBY_C23_CONDITIONS, skipped")
                continue
            r = _arm(model_name, cond_name, [evaluator],
                     f"c23_{model_name}_{evaluator}")
            if r:
                scores[cond_name] = r
            else:
                failures.append(f'{model_name}/{cond_name}')

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
                    scores[cond]['width_gap_vs_matched_per_fold'] = (
                        _paired_width_gap(scores['A_matched'], scores[cond]))

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

    def _matched_of(model):
        for prev in results:
            if prev['model'] == model and prev['condition_type'] == 'ABC':
                s = prev['scores'].get('A_matched')
                if s:
                    return s
        return None

    # ── Condition D: NN size mismatch ──
    print(f"\n  NN size mismatch pairs:")
    for downstream, evaluator in (NN_PAIRS if 'D' in conditions else []):
        if downstream not in run_models:
            continue
        r = _arm(downstream, 'D_nn_size', [evaluator],
                 f"c23_{downstream}_{evaluator}")
        if r is None:
            failures.append(f'{downstream}/D_nn_size')

        matched_entry = _matched_of(downstream)
        delta = None
        n_pair = 0
        gap = None
        if matched_entry is not None and r is not None:
            delta, n_pair = _paired_fold_delta(matched_entry, r, sign)
            gap = _paired_width_gap(matched_entry, r)

        results.append({
            'model': downstream,
            'condition_type': 'D_nn_size',
            'evaluators': {'downstream': downstream, 'evaluator': evaluator},
            'scores': {'D_nn_size': r} if r else {},
            'delta_vs_matched': delta,
            'n_paired_folds': n_pair,
            'width_gap_vs_matched_per_fold': gap,
        })

    # ── Condition E: Quick default set ──
    # For rf, lightgbm and xgboost the quick set CONTAINS that model's own
    # matched evaluator, so E searches a superset of A. On the old same-fold
    # numbers that alone forced its delta to be zero or negative. Each outer
    # fold now picks its evaluator on its own training rows and is scored on
    # rows it never saw, so the sign is no longer forced — the containment is
    # still recorded per row as evaluator_set_contains_matched.
    print(f"\n  Quick default set: {QUICK_EVALUATORS}")
    for model_name in (run_models if 'E' in conditions else []):
        score_info = _arm(model_name, 'E_quick_set', QUICK_EVALUATORS,
                          f"c23quick_{model_name}")
        if score_info is None:
            failures.append(f'{model_name}/E_quick_set')
            continue
        # Each fold picks its own evaluator from the quick set; those are in
        # winning_evaluator_per_fold (None = fold had no winner).
        score_info['winning_evaluator_is_from_fold'] = score_info.get(
            'recipe_is_from_fold')
        score_info['winning_evaluator_per_fold'] = [
            r['internal_evaluator'] for r in score_info['recipe_per_fold']]
        score_info['evaluator_set_contains_matched'] = bool(
            MATCH_TABLE[model_name][0] in QUICK_EVALUATORS)
        print(f"      winner per fold: "
              f"{score_info['winning_evaluator_per_fold']}")

        # Delta vs matched, paired fold by fold.
        matched_entry = _matched_of(model_name)
        if matched_entry is not None:
            d, n_pair = _paired_fold_delta(matched_entry, score_info, sign)
            score_info['delta_vs_matched'] = d
            score_info['n_paired_folds'] = n_pair
            score_info['width_gap_vs_matched_per_fold'] = _paired_width_gap(
                matched_entry, score_info)

        results.append({
            'model': model_name,
            'condition_type': 'E_quick_set',
            'evaluators': QUICK_EVALUATORS,
            'scores': {'E_quick_set': score_info},
        })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {metric_name}, {'lower' if lower_is_better else 'higher'} is "
          f"better; delta = matched minus other, signed so positive means "
          f"matched is better; width = features built, mean over folds")

    abc_results = [r for r in results if r['condition_type'] == 'ABC']
    if abc_results:
        print(f"\n  {'model':<12s} {'A_matched':>10s} {'B_slight':>10s} {'C_strong':>10s} {'delta_B':>8s} {'delta_C':>8s} {'width A/B/C':>16s}")
        print(f"  {'-'*78}")
        for r in abc_results:
            s = r['scores']
            a = s.get('A_matched', {}).get('score_mean', float('nan'))
            b = s.get('B_slight_off', {}).get('score_mean', float('nan'))
            c = s.get('C_strong_off', {}).get('score_mean', float('nan'))
            db = s.get('B_slight_off', {}).get('delta_vs_matched', float('nan'))
            dc = s.get('C_strong_off', {}).get('delta_vs_matched', float('nan'))
            w = '/'.join(str(s.get(k, {}).get('n_features', '-'))
                         for k in ('A_matched', 'B_slight_off', 'C_strong_off'))
            print(f"  {r['model']:<12s} {_fmt(a):>10s} {_fmt(b):>10s} "
                  f"{_fmt(c):>10s} {_fmt(db, '+.4f'):>8s} "
                  f"{_fmt(dc, '+.4f'):>8s} {w:>16s}")
        if n_outer_folds == 1:
            print("  one fold: each score is a single held-out split, no spread")

    if failures:
        print(f"\n  arms with no result: {failures}")

    return convert_numpy({
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_samples': n_samples,
        'result_version': RESULT_VERSION,
        'selection_scope': 'fold_train_rows',
        'selection_rule': ('one hybrid per internal-evaluator condition per '
                           'outer fold, from a search on that fold\'s '
                           'training rows (handoff 2.3 (B), 2026-10-05)'),
        'models_decided': decided,
        'models_tested': run_models,
        'models_not_run': models_not_run,
        'budgets': budgets,
        'budgets_decided': decided_budgets,
        'aug_condition': 'none',
        'aug_conditions_not_run': aug_not_run,
        'n_outer_folds': n_outer_folds,
        # What this run actually asked for, so a result file can never be read
        # as the full conditions when it is not. See _c23_conditions.
        'conditions_run': sorted(conditions),
        'conditions_shipped': ['A', 'B', 'C', 'D', 'E'],
        'arms_failed': failures,
        # Every score_mean in results is a mean over outer folds whose recipe
        # search never saw them, and every delta_vs_matched is the fold-by-fold
        # difference of two such scores, averaged.
        'score_basis': 'mean over outer folds (nested CV)',
        'delta_basis': 'paired per outer fold, then averaged',
        # One row per (model, condition, outer fold): the recipe, the
        # training-row score and the held-out score.
        'per_fold': [pf for r in results for s in r['scores'].values()
                     if s for pf in (s.get('per_fold') or [])],
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

    # 2026-10-05: the dataset's general representations (as run_all loads
    # them) unless --reps names a comma-separated list.
    from experiments.complexity.complexity_helpers import DATASET_REPS
    rep_preset, rep_names = parse_rep_args(args)
    if rep_names is None and (DATASET_REPS.get(args.dataset) or {}).get(
            'general'):
        rep_preset = 'all'
        rep_names = list(DATASET_REPS[args.dataset]['general'])
    data = load_data(args.dataset, rep_preset, rep_names=rep_names)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, EXPERIMENT_DIR,
                            args.dataset,
                            metric_name=results.get('metric'),
                            lower_is_better=results.get('lower_is_better'))
