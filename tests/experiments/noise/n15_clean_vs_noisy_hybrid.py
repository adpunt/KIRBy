#!/usr/bin/env python3
"""N15: Clean-searched vs noise-searched hybrid blueprints.

Paper: Noise.

THE QUESTION. Every noise experiment in this suite builds its hybrid from a
blueprint that was searched on the CLEAN labels. The base representations get no
such look — they are fixed matrices. So when a noise experiment reports "the hybrid
copes better with corrupted labels", we cannot tell whether the hybrid is genuinely
better or is simply cashing in on having been BUILT with knowledge of the right
answers. The July 2026 audit let n11 and n12 survive only by making them carry an
"ORACLE-BIASED ... best-case, not a fair baseline ... must not be reported as one"
warning.

This experiment MEASURES the size of that advantage instead of assuming it.

DESIGN, AS ALIGNED 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md section 2.4,
chat 5). Per outer fold, two per-fold searches on that fold's training rows,
on the dataset's decided grid, unpruned:

  clean arm  — searched on the clean training labels. This is the same
               per-fold search every other noise experiment selects from
               (no condition_suffix), so it is run once and shared.
  noisy arm  — searched on the DAMAGED training labels of one noise draw,
               with inner folds trained AND scored on those damaged labels,
               i.e. what an honest practitioner could actually have picked.
               One search per (fold, level, draw), filed under its own
               condition_suffix so no two draws share a stored search.

Each arm selects, per decided model, the 2 best base representations and the
2 best hybrids (fold_standard_selection). Both arms' four configurations are
then given the SAME treatment: hybrid rebuilt on the fold's training rows from
the damaged labels, model trained on the damaged training labels, scored on
the CLEAN held-out labels. The only thing that differs between the arms is
which configurations were chosen and which labels they were chosen on.

  oracle_advantage(slot) = score(clean arm, slot) - score(noisy arm, slot)
                           [direction-aware: positive = clean-searched better]

  slots: base_1, base_2, hybrid, hybrid__general2 (the 8 fitted
  configurations of R5 per arm: 2 models x (2 base + 2 hybrids)).

  ~0     -> the head start is worth nothing. The oracle worry is empirically dead,
            and n11/n12 can drop their warnings.
  large  -> the head start is real, and every noise result that used a
            clean-searched configuration is inflated by roughly this much.

LEVEL 0 IS NOT A COMPARISON (dirty-run bug "one configuration gives 0 by
construction", 2026-10-05). At level 0 the damaged labels ARE the clean labels,
so the noisy search is the clean search and the advantage is 0 by construction.
Level 0 is run once per fold as the clean reference: the clean arm's
configurations trained on clean labels. It is reported under 'level_0' and
never enters an advantage.

Also reported: AGREEMENT per slot — did the two arms pick the same thing? Where
they agree, the advantage is exactly 0 for that slot, and that is a finding.

SCOPE / CAVEATS (read first).
  - Synthetic noise only, injected on TRAINING rows only through
    noise_helpers.inject_condition; held-out labels are never damaged.
  - One kind of damage: gaussian on regression, the uniform flip on
    classification, as before. Levels are the decided grid
    (noise_helpers.decided_levels: full_knn 0, 0.5, 1.0, 1.5; fast_knn 0, 1.0;
    classification the flip-rate grid), 3 draws per non-zero level.
  - This is a comparison of SELECTIONS, not of hybrid-vs-base.

Usage:
    python n15_clean_vs_noisy_hybrid.py --dataset logd
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric, get_model_factories,
    fold_standard_selection,
)
from experiments.decided_grid import decided_models
from experiments.noise.noise_helpers import (
    noise_outer_folds, fold_spread, fold_model_inputs, configuration_record,
    decided_levels, inject_condition, dataset_knn_group, CONFIGURATION_SPACES,
)
from kirby.noise_spec import (noisy_labels_hash, DEFAULT_STRATEGY_CLF,
                              DEFAULT_STRATEGY_REG)
# The per-unit checkpoint and the fit-and-score helper (noise_cache under the
# hood for seeded models). n15 finds no neighbours, so it takes no similarity
# setting and no distance or similarity hybrid (handoff 2.3 item 1).
from experiments.noise import _sim_axis_n15_n22 as sx

# The name the checkpoint and the result file are filed under.
EXP_FULL_NAME = 'n15_clean_vs_noisy_hybrid'

# Bumped 2026-10-05 (chat 5): the design changed from whole-dataset searches
# to per-fold searches, so a checkpoint written before must not be resumed.
CHECKPOINT_VERSION = 'per_fold_2026_10_05'

# Draws per non-zero level (owner's decision 2026-09-12, unchanged).
N_DRAWS = 3

# The four fitted configurations of one model, compared clean arm against
# noisy arm slot by slot.
SLOTS = tuple(CONFIGURATION_SPACES)


def _bp_fingerprint(bp):
    """The parts of a blueprint that define what it selected."""
    if bp is None:
        return None
    return {
        'reps': sorted(getattr(bp, 'reps', []) or []),
        'allocation_method': getattr(bp, 'allocation_method', None),
        'importance_method': getattr(bp, 'importance_method', None),
        'budget': getattr(bp, 'budget', None),
        'internal_evaluator': getattr(bp, 'internal_evaluator', None),
    }


def _record(entry, names):
    """configuration_record, with each hybrid's evaluator and rep count filled.

    2026-10-05 (chat 5): configuration_record copies the keys 'evaluator' and
    'n_reps', but the hybrid entries from _shared.top_hybrids name them
    'internal_evaluator' and 'rep_count', so both came out null. Filled here
    from the entry's own hybrids, in the same order.
    """
    rec = configuration_record(entry, names)
    if entry is None:
        return rec
    for out_h, h in zip(rec.get('hybrids') or [], entry.get('hybrids') or []):
        if out_h.get('evaluator') is None:
            out_h['evaluator'] = h.get('internal_evaluator', h.get('evaluator'))
        if out_h.get('n_reps') is None:
            out_h['n_reps'] = h.get('rep_count', h.get('n_reps'))
    return rec


def _agreement(fp_clean, fp_noisy):
    """How much do the two blueprints agree?"""
    if fp_clean is None or fp_noisy is None:
        return None
    a, b = set(fp_clean['reps']), set(fp_noisy['reps'])
    union = a | b
    return {
        'rep_jaccard': (len(a & b) / len(union)) if union else 1.0,
        'reps_identical': a == b,
        'reps_clean_only': sorted(a - b),
        'reps_noisy_only': sorted(b - a),
        'same_allocation': fp_clean['allocation_method'] == fp_noisy['allocation_method'],
        'same_importance': fp_clean['importance_method'] == fp_noisy['importance_method'],
        'same_budget': fp_clean['budget'] == fp_noisy['budget'],
        'identical': fp_clean == fp_noisy,
    }


def _noise_kind(is_clf):
    """The one kind of damage n15 injects: gaussian on regression, the uniform
    flip on classification (kirby.noise_spec defaults), as before.

    # OPEN (chat 5): n15 has always injected one kind. The decided grid has
    # six regression kinds (noise_helpers.DECIDED_REGRESSION_CONDITIONS).
    # Sweeping all six multiplies the noisy per-fold searches by six; nothing
    # decided whether n15 should. Left at gaussian.
    """
    return DEFAULT_STRATEGY_CLF if is_clf else DEFAULT_STRATEGY_REG


def _draw_plan(data, dataset_name, condition):
    """[(draw id, level, draw)] for every non-zero level, N_DRAWS each.
    Level 0 is not here: it is the clean reference, run once per fold."""
    out = []
    for lvl in decided_levels(data, dataset_name, condition):
        if float(lvl) == 0.0:
            continue
        for d in range(N_DRAWS):
            out.append((f'{condition}_L{float(lvl):g}_d{d}', float(lvl), d))
    return out


def _selection(data, labels, dataset_name, *, fold_i, train_idx, n_folds,
               models, condition_suffix=None):
    """{model: entry} for one arm on one fold, in the form
    noise_helpers.fold_model_inputs reads.

    fold_standard_selection runs ONE per-fold general search on this fold's
    training rows (decided grid, pruning off) and ranks, per model, the 2 best
    base representations and the 2 best hybrids with distinct selected
    features. The distance and similarity searches fold_configurations also
    runs are not run here: n15 measures nothing with neighbours (handoff 2.3
    item 1), so those hybrids would never be used.

    `labels` is the full label column with the training rows as this arm
    sees them. The search reads only the training rows, so for the noisy arm
    its inner folds are trained and scored on the damaged labels.
    """
    sel = fold_standard_selection(
        data['reps'], labels, data.get('groups'),
        train_idx=train_idx, fold_i=fold_i, n_outer_folds=n_folds,
        dataset_name=dataset_name, models=list(models),
        augmentations=data.get('augmentations'),
        task_type=data.get('task_type') or 'auto',
        condition_suffix=condition_suffix)
    out = {}
    for m in models:
        s_m = sel.get(m)
        if s_m is None:
            out[m] = None
            continue
        recipes, unavailable = {}, {}
        for space, i in (('hybrid', 0), ('hybrid__general2', 1)):
            if len(s_m['hybrids']) > i:
                h = s_m['hybrids'][i]
                recipes[space] = (h['blueprint'], h.get('aug_condition', 'none'))
            else:
                unavailable[space] = (f'the per-fold general search gave this '
                                      f'model fewer than {i + 1} hybrids with '
                                      f'distinct features')
        base_sel = (s_m.get('base_detail') or {}).get('selected') or []
        for i, space in enumerate(('base_1', 'base_2')):
            if len(s_m['base']) <= i:
                unavailable[space] = (
                    f'the per-fold search scored fewer than {i + 1} '
                    f'representations for this model: '
                    f"{(s_m.get('base_detail') or {}).get('shortfall_reason')}")
        srch = s_m.get('search')
        out[m] = {
            'base': list(s_m['base']),
            'hybrids': list(s_m['hybrids']),
            'hybrids_skipped_identical_features':
                s_m.get('hybrids_skipped_identical_features', []),
            'recipes': recipes,
            'unavailable': unavailable,
            'fit_configurations': (list(s_m['base'])
                                   + [h['label'] for h in s_m['hybrids']]),
            # The inner-CV score on this fold's training rows, per slot, in the
            # dataset's metric: the training-row score of handoff 2.4 item 11.
            'training_row_scores': {
                **{sp: base_sel[i].get('score') for i, sp in
                   enumerate(('base_1', 'base_2')) if len(base_sel) > i},
                **{sp: s_m['hybrids'][i].get('score_mean') for i, sp in
                   enumerate(('hybrid', 'hybrid__general2'))
                   if len(s_m['hybrids']) > i}},
            'n_candidates': len(getattr(srch, 'all_results', None) or []),
            'selection_scope': 'fold_train_rows',
        }
    return out


def _slot_fingerprint(entry, slot):
    """What one slot of one arm is, for the agreement check: the
    representation name for a base slot, the recipe for a hybrid slot."""
    if entry is None:
        return None
    if slot in ('base_1', 'base_2'):
        i = 0 if slot == 'base_1' else 1
        return {'rep': entry['base'][i]} if len(entry['base']) > i else None
    i = 0 if slot == 'hybrid' else 1
    if len(entry['hybrids']) <= i:
        return None
    h = entry['hybrids'][i]
    fp = _bp_fingerprint(h.get('blueprint'))
    if fp is not None:
        fp = dict(fp, aug_condition=h.get('aug_condition', 'none'))
    return fp


def _arm_scores(data, dataset_name, model_name, entry, factory, *, train_idx,
                val_idx, y_train, y_val, metric):
    """Held-out score per slot of one arm for one model on one fold.

    fold_model_inputs rebuilds each hybrid on the fold's training rows from
    the per-fold recipe with `y_train` (the labels this cell trains on), so
    the hybrid's features are chosen from those labels too, never from clean
    ones. Base representations are the scaled matrices. Every model is
    trained on `y_train` and scored on the clean `y_val`.
    """
    inputs, unavail, names = fold_model_inputs(
        data, dataset_name, model_name, entry, train_idx=train_idx,
        val_idx=val_idx, y_train=y_train, factory=factory, metric=metric)
    scores = {}
    for sp in SLOTS:
        if sp not in inputs:
            continue
        X_tr, X_va = inputs[sp]
        sc = sx.fit_score(dataset_name, model_name, factory, X_tr, y_train,
                          X_va, y_val, metric)
        if np.isfinite(sc):
            scores[sp] = sc
    return scores, dict(unavail), dict(names)


def _advantage(c, n, lower_is_better):
    """Direction-aware: positive = the clean-searched configuration did better."""
    return (n - c) if lower_is_better else (c - n)


def run(data, config, output_dir, dataset_name, **kwargs):
    print("\n" + "=" * 70)
    print("EXPERIMENT N15: CLEAN-SEARCHED vs NOISE-SEARCHED CONFIGURATIONS")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric = get_domain_metric(data)
    metric_name, _metric_fn, lower_is_better = metric

    # The dataset's two decided models (DECISIONS.md Step 1). A run that names
    # models keeps only decided ones (a smoke run on one model, say).
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = list(kwargs.get('models') or config.get('models') or decided)
    models = [m for m in decided if m in asked] or decided
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    models_no_factory = {m: 'get_model_factories gave no model for this '
                            'dataset size' for m in models if m not in factories}

    condition = _noise_kind(is_clf)
    plan = _draw_plan(data, dataset_name, condition)
    folds, n_folds = noise_outer_folds(data, config)
    print(f"  models {models}; {n_folds} outer folds "
          f"({len(folds)} run here); damage {condition} at levels "
          f"{sorted({lvl for _, lvl, _ in plan})} x {N_DRAWS} draws, level 0 "
          f"once as the clean reference; "
          f"knn group {dataset_knn_group(data, dataset_name)}")

    results = {
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'noise_regime': 'synthetic',
        'noise_injection': 'noise_helpers.inject_condition, training rows only',
        'noise_condition': condition,
        'claim_scope': (
            'Compares configurations chosen by a per-fold search on clean '
            'training labels against configurations chosen by a per-fold '
            'search on damaged training labels, both trained on the damaged '
            'labels and scored on clean held-out labels. It does NOT establish '
            'a hybrid-vs-base result, and does not transfer to real label '
            'noise of unknown structure.'),
        'noise_levels': sorted({0.0} | {lvl for _, lvl, _ in plan}),
        'noise_draws': {did: {'level': lvl, 'draw': d} for did, lvl, d in plan},
        'n_draws_per_level': N_DRAWS,
        'knn_group': dataset_knn_group(data, dataset_name),
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'models': list(models),
        'models_unavailable': models_no_factory,
        'selection_scope': 'fold_train_rows',
        'slots': list(SLOTS),
        'per_fold': {},
        'summary': {},
    }

    # ── The checkpoint ──
    # One unit per (fold, clean reference) and per (fold, draw). The per-fold
    # searches are on disk anyway; this keeps the fits.
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'version': CHECKPOINT_VERSION, 'condition': condition,
            'plan': [p[0] for p in plan], 'models': models,
            'n_folds': n_folds}))
    if ck.state is not None:
        results = ck.state

    run_models = [m for m in models if m in factories]

    for fold_i, train_idx, val_idx in folds:
        fkey = str(fold_i)
        fold_out = results['per_fold'].setdefault(fkey, {
            'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
            'clean_arm': {}, 'level_0': {}, 'draws': {}})
        y_val = labels[val_idx]
        y_clean_tr = labels[train_idx]
        print(f"\n--- fold {fold_i} of {n_folds} "
              f"({len(train_idx)} training, {len(val_idx)} held out) ---")

        # The clean arm's selection. Same per-fold search as every other
        # noise experiment's (no condition_suffix), memoised in-process, so
        # every draw below reads it without searching again.
        clean_entries = _selection(data, labels, dataset_name, fold_i=fold_i,
                                   train_idx=train_idx, n_folds=n_folds,
                                   models=run_models)

        unit = f'{fold_i}|level_0'
        if not ck.is_done(unit):
            for m in run_models:
                entry = clean_entries.get(m)
                sc, unav, names = _arm_scores(
                    data, dataset_name, m, entry, factories[m],
                    train_idx=train_idx, val_idx=val_idx, y_train=y_clean_tr,
                    y_val=y_val, metric=metric)
                rec = _record(entry, names)
                if entry is not None:
                    rec['training_row_scores'] = entry['training_row_scores']
                    rec['n_candidates'] = entry['n_candidates']
                fold_out['clean_arm'][m] = rec
                fold_out['level_0'][m] = {'held_out_scores': sc,
                                          'unavailable': unav}
                print(f"  level 0 {m}: " + ', '.join(
                    f'{k}={v:.4f}' for k, v in sc.items()))
            ck.save(unit, results)

        for draw_id, lvl, d in plan:
            unit = f'{fold_i}|{draw_id}'
            if ck.is_done(unit):
                print(f"  {draw_id}: already in the checkpoint, skipped")
                continue
            y_noisy_tr, mask, prov = inject_condition(
                data, dataset_name, condition, lvl, d, train_idx=train_idx)
            y_noisy_tr = np.asarray(y_noisy_tr)
            # The draw is deterministic (the noisy search and the fits below
            # must see one column), has the training rows' length, and moves
            # labels above level 0. A level that delivered no damage would
            # make every advantage exactly zero and read as the null this
            # experiment tests for.
            _again = np.asarray(inject_condition(
                data, dataset_name, condition, lvl, d, train_idx=train_idx)[0])
            _h = noisy_labels_hash(y_noisy_tr)
            if noisy_labels_hash(_again) != _h:
                raise RuntimeError(
                    f"n15 {draw_id} fold {fold_i}: inject_condition is not "
                    f"deterministic for this draw; the noisy search and the "
                    f"fits would see different labels.")
            if len(y_noisy_tr) != len(train_idx):
                raise RuntimeError(
                    f"n15 {draw_id} fold {fold_i}: {len(y_noisy_tr)} damaged "
                    f"labels for {len(train_idx)} training rows.")
            if np.array_equal(y_noisy_tr, y_clean_tr):
                raise RuntimeError(
                    f"n15 {draw_id} fold {fold_i}: level {lvl} delivered labels "
                    f"identical to the clean ones; both arms would be searched "
                    f"on the same labels and every advantage would be zero by "
                    f"construction.")

            # The noisy arm: the full label column with this fold's training
            # rows damaged. The search reads only the training rows, so it
            # trains and validates its inner folds on the damaged labels. Its
            # own condition_suffix per draw, so the stored search of one draw
            # is never read for another (or for the clean arm).
            labels_noisy = labels.copy()
            if not is_clf:
                labels_noisy = labels_noisy.astype(float)
            labels_noisy[train_idx] = y_noisy_tr
            noisy_entries = _selection(
                data, labels_noisy, dataset_name, fold_i=fold_i,
                train_idx=train_idx, n_folds=n_folds, models=run_models,
                condition_suffix=f'n15_{draw_id}')

            draw_out = {
                'level': lvl, 'draw': d, 'condition': condition,
                'noisy_labels_hash': _h,
                'n_training_labels_damaged': int(np.count_nonzero(mask)),
                'provenance': {k: (v.item() if isinstance(v, np.generic) else v)
                               for k, v in (prov or {}).items()
                               if isinstance(v, (int, float, str, bool,
                                                 np.generic)) or v is None},
                'models': {}}
            for m in run_models:
                ce, ne = clean_entries.get(m), noisy_entries.get(m)
                sc_c, un_c, nm_c = _arm_scores(
                    data, dataset_name, m, ce, factories[m],
                    train_idx=train_idx, val_idx=val_idx, y_train=y_noisy_tr,
                    y_val=y_val, metric=metric)
                sc_n, un_n, nm_n = _arm_scores(
                    data, dataset_name, m, ne, factories[m],
                    train_idx=train_idx, val_idx=val_idx, y_train=y_noisy_tr,
                    y_val=y_val, metric=metric)
                ref = (fold_out['level_0'].get(m) or {}).get(
                    'held_out_scores') or {}
                by_slot = {}
                for sp in SLOTS:
                    fp_c, fp_n = _slot_fingerprint(ce, sp), _slot_fingerprint(ne, sp)
                    cell = {
                        'clean_arm_choice': fp_c, 'noisy_arm_choice': fp_n,
                        'same_choice': (fp_c is not None and fp_c == fp_n),
                        'clean_arm_score': sc_c.get(sp),
                        'noisy_arm_score': sc_n.get(sp),
                    }
                    if sp.startswith('hybrid'):
                        i = 0 if sp == 'hybrid' else 1
                        cell['agreement'] = _agreement(
                            _bp_fingerprint(ce['hybrids'][i]['blueprint'])
                            if ce and len(ce['hybrids']) > i else None,
                            _bp_fingerprint(ne['hybrids'][i]['blueprint'])
                            if ne and len(ne['hybrids']) > i else None)
                    if sp in sc_c and sp in sc_n:
                        cell['oracle_advantage'] = float(_advantage(
                            sc_c[sp], sc_n[sp], lower_is_better))
                    # The plain difference damaged minus clean, in the
                    # dataset's own metric, against the clean arm's same slot
                    # trained on clean labels (level 0). Read with
                    # lower_is_better; no normalised score (auc_norm is not
                    # computed on fast_knn datasets, by decision).
                    if sp in ref:
                        if sp in sc_c:
                            cell['clean_arm_damaged_minus_clean'] = float(
                                sc_c[sp] - ref[sp])
                        if sp in sc_n:
                            cell['noisy_arm_damaged_minus_clean'] = float(
                                sc_n[sp] - ref[sp])
                    by_slot[sp] = cell
                rec_n = _record(ne, nm_n)
                if ne is not None:
                    rec_n['training_row_scores'] = ne['training_row_scores']
                    rec_n['n_candidates'] = ne['n_candidates']
                n_c = ce['n_candidates'] if ce else None
                n_n = ne['n_candidates'] if ne else None
                draw_out['models'][m] = {
                    'noisy_arm': rec_n,
                    'unavailable': {'clean_arm': un_c, 'noisy_arm': un_n},
                    'by_slot': by_slot,
                    # Pruning is off in both searches, so both arms score the
                    # same candidate list; a mismatch means configurations
                    # failed in one arm only.
                    'n_candidates_clean_arm': n_c,
                    'n_candidates_noisy_arm': n_n,
                    'candidate_counts_match': (n_c == n_n),
                }
                print(f"  {draw_id} {m}: " + ', '.join(
                    f"{sp} {by_slot[sp]['oracle_advantage']:+.4f}"
                    for sp in SLOTS if 'oracle_advantage' in by_slot[sp])
                    + ' (oracle advantage, ' + metric_name + ')')
            fold_out['draws'][draw_id] = draw_out
            ck.save(unit, results)

    results['summary'] = _summarise(results)
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY: oracle advantage per level, model and slot "
          f"({metric_name}; positive = clean-searched better), "
          f"mean over draws within a fold, then over folds")
    for lvl, per_model in s.get('per_level', {}).items():
        for m, per_slot in per_model.items():
            for sp, c in per_slot.items():
                fs = c['oracle_advantage_over_folds']
                sd = ('' if fs.get('sd') is None else f" sd {fs['sd']:.4f}")
                mean = fs.get('mean')
                print(f"  level {lvl} {m} {sp}: "
                      f"{'n/a' if mean is None else f'{mean:+.4f}'}{sd} "
                      f"({fs.get('note') or str(fs.get('n_folds')) + ' folds'}); "
                      f"same choice in {c['n_same_choice']} of "
                      f"{c['n_cells']} (fold, draw) cells")
    print("=" * 70)

    # n15's result file is written by its caller (run_all or main), not here.
    # The snapshot is cleared on return: a kill between this return and that
    # write costs a re-run of the fits, never a stale resume.
    ck.clear()
    return results


def _summarise(results):
    """Per (level, model, slot): the oracle advantage per fold (mean over that
    fold's draws), its spread over folds (fold_spread), every (fold, draw)
    value, and how often the two arms chose the same configuration. No
    significance test."""
    acc = {}
    for fkey, fo in results['per_fold'].items():
        for draw_id, dr in (fo.get('draws') or {}).items():
            lvl = str(dr['level'])
            for m, mc in (dr.get('models') or {}).items():
                for sp, cell in (mc.get('by_slot') or {}).items():
                    a = acc.setdefault(lvl, {}).setdefault(m, {}).setdefault(
                        sp, {'per_fold': {}, 'per_cell': [], 'same': 0,
                             'n': 0, 'clean_dmc': {}, 'noisy_dmc': {}})
                    a['n'] += 1
                    a['same'] += int(bool(cell.get('same_choice')))
                    v = cell.get('oracle_advantage')
                    if v is not None and np.isfinite(v):
                        a['per_fold'].setdefault(fkey, []).append(v)
                        a['per_cell'].append({'fold': int(fkey),
                                              'draw': draw_id, 'value': v})
                    for k_in, k_out in (('clean_arm_damaged_minus_clean',
                                         'clean_dmc'),
                                        ('noisy_arm_damaged_minus_clean',
                                         'noisy_dmc')):
                        w = cell.get(k_in)
                        if w is not None and np.isfinite(w):
                            a[k_out].setdefault(fkey, []).append(w)
    per_level = {}
    for lvl, per_model in acc.items():
        for m, per_slot in per_model.items():
            for sp, a in per_slot.items():
                def _over_folds(d):
                    return fold_spread([float(np.mean(v)) for _, v in
                                        sorted(d.items(), key=lambda kv: int(kv[0]))])
                per_level.setdefault(lvl, {}).setdefault(m, {})[sp] = {
                    'oracle_advantage_over_folds': _over_folds(a['per_fold']),
                    'oracle_advantage_per_fold_and_draw': a['per_cell'],
                    'clean_arm_damaged_minus_clean_over_folds':
                        _over_folds(a['clean_dmc']),
                    'noisy_arm_damaged_minus_clean_over_folds':
                        _over_folds(a['noisy_dmc']),
                    'n_cells': a['n'],
                    'n_same_choice': a['same'],
                }
    cand_mismatch = [
        f"fold {fkey} {draw_id} {m}"
        for fkey, fo in results['per_fold'].items()
        for draw_id, dr in (fo.get('draws') or {}).items()
        for m, mc in (dr.get('models') or {}).items()
        if not mc.get('candidate_counts_match')]
    return {
        'per_level': per_level,
        'metric': results.get('metric'),
        'lower_is_better': results.get('lower_is_better'),
        'sign': 'positive = the clean-searched configuration scored better',
        'cells_where_candidate_counts_disagreed': cand_mismatch,
    }


if __name__ == '__main__':
    parser = create_experiment_parser('N15: Clean vs Noisy Hybrid Blueprints')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    results = run(data, config, args.output, args.dataset)
    save_experiment_results(results, args.output, 'n15_clean_vs_noisy_hybrid',
                            args.dataset)
