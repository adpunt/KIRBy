#!/usr/bin/env python3
"""N27: what removing training molecules costs, and how much label error reaches the test error.

Paper: Noise.

THE QUESTION. Every cleanup in this suite either removes training molecules or
moves their labels. Two numbers decide whether either can pay, and neither has
been measured in the suite:

  1. what it costs to train on fewer molecules when every label is correct;
  2. how much of an error put into the training labels comes out in the
     held-out error.

This file measures both on the same outer folds. It cleans nothing.

THREE PARTS, ALL PAIRED ON THE SAME OUTER FOLDS.

  removal_cost   train on a random share of the training molecules, at shares
                 0.25, 0.5, 0.7, 0.8 and 0.9, 20 random draws of the molecules
                 at each share, every label correct. Each is paired against
                 training on all of them. A removal cleanup that drops a tenth
                 of the training molecules pays at least the 0.9 row, before
                 any benefit from which molecules it dropped. 0.25 and 0.5 were
                 added 2026-09-24 from a second research report (its D3): a
                 steep curve down to a quarter says the dataset is short of
                 molecules or features, and cleaning comes second.

  injected       gaussian noise at the dataset's decided levels
                 (noise_helpers.decided_levels: 0.5, 1.0, 1.5 of the clean
                 training-label SD on a dataset of 5000 molecules or fewer,
                 1.0 above that), 3 draws per level, put into the training
                 labels only. The injected variance is MEASURED on
                 each pass as the mean of (noisy label - correct label) squared
                 over the training molecules, in label units squared. The
                 held-out mean squared error minus the correct-label fit's
                 held-out mean squared error, divided by that variance, is how
                 much of each unit of injected variance reaches the test error.

  relabelled     the correct training labels moved a fraction alpha of the way
                 toward an out-of-fold prediction, alpha 0.25, 0.5 and 1.0. The
                 prediction is a LightGBM fitted on the other inner folds of the
                 training molecules, group-aware (`oof_predict`). The same ratio
                 is computed, dividing by the mean of (moved label - correct
                 label) squared.

The injected and relabelled ratios are the two numbers a relabelling cleanup is
trading between. A cleanup that moves a wrong label toward a prediction removes
some of the first kind of error and adds some of the second. The file reports
both ratios side by side, per downstream model and per feature space, with the
variance each was divided by. It does not say which is larger.

WHAT THE HELD-OUT MOLECULES ARE GRADED AGAINST. `labels[val_idx]`, as the
loader produced them, in every part. No moved label and no injected label is
ever scored against.

CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
What each outer fold does, in order:
  1. Folds are noise_helpers.noise_outer_folds: the splits the per-fold
     searches are filed under, the count from config['n_splits'] or
     KIRBY_NOISE_N_FOLDS. Until 2026-10-05 they were _shared.cv_split.
  2. noise_helpers.fold_configurations runs this fold's searches on its
     training rows (clean labels, decided grid, pruning off) and gives each of
     the dataset's two decided models (DECISIONS.md Step 1) its 2 best base
     representations and its 2 best general hybrids. Nothing is read from a
     whole-dataset search. Until 2026-10-05 the recipes and the base
     representations came from whole-dataset stored searches, ranked on every
     row, for five models plus rf and svm forced in, with up to 6 hybrids per
     model (2 general + 1 distance, prediction and similarity families).
  3. Each model is fitted on its 4 configurations (2 models x 4 = 8, R5):
     base_1, base_2, hybrid, hybrid__general2, through
     noise_helpers.fold_model_inputs, which rebuilds each hybrid on the fold's
     training rows with the fold search's own representation scores (dirty-run
     bug 1). n27 reads no neighbours, so no distance hybrid or similarity
     hybrid is built (handoff 2.3 item 1).

WHERE THE FEATURES COME FROM, PART BY PART. Each part says it on the result
file too.
  removal_cost and relabelled: each feature space is built ONCE per fold, from
    ALL the fold's training molecules at their correct labels. The molecules
    are removed, or the labels moved, after the features were chosen. So a
    removal row prices a smaller training set for the downstream model only,
    not for the feature choice.
  injected: each hybrid space is rebuilt on every pass from the NOISY training
    labels, the same way n24 builds it. The single-representation spaces do
    not depend on the labels.

THE SINGLE-REPRESENTATION COMPARISON is each decided model's top 2 base
representations by that model's score in this fold's search, on the fold's
training rows -- spaces base_1 and base_2. Which representations those are,
per fold and model, is written to `fold_configurations`.

WHAT IS NOT DONE HERE. No random share is taken of the held-out molecules. The
share 1.0 row is fitted once per fold, not 20 times: it is the same training
set every time. The file does not compare the removal cost to any cleanup's
gain; n24's random-deletion arms and n17's floor arms do that at one share.

WHICH DATASETS. Any regression dataset with one label per molecule. The ones
this was written for are herg_ki, caco2, logd and flab_thermo_tresanco. A
classification label is refused: a variance of the label error is not defined
for a class.

THE HEADLINE IS PER DRAW AND PER PASS, NOT A MEAN. The removal part writes one
row per random draw of the molecules, with the best and the worst draw. The
propagation parts write one row per pass, and a pass is one (noise draw, fold)
for injected and one fold for relabelled.

ROWS, CACHES, CHECKPOINT (2026-09-25, plan sections 2-3; identity bumped to
RESULT_LAYOUT on 2026-10-05 so rows and checkpoints of the whole-dataset design
are never read back). removal_cost and
relabelled change the training data, so each writes one D-versus-D' row per
fold (D = every molecule at its correct label, so score_D = score_clean).
injected adds noise and writes none. The accumulators are checkpointed per
fold; the scaled representations, hybrid builds, out-of-fold targets and the
fits of seeded models on correct and injected labels go through noise_cache.
controls.py's twins of relabelled run with KIRBY_NOISE_CONTROLS=1.

Usage:
    python n27_removal_cost_and_propagation.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import time
import zlib

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.dataset_policy import noise_dose_units, noise_groups_for
from experiments.noise.noise_helpers import fold_spread
from experiments.noise.n07_remediation import _predict_for_metric
# The D-versus-D' rows, the per-fold checkpoint, the caches and the controls
# (2026-09-25, plan sections 2-3). They live in n23 for the reason given at
# the top of its shared block.
from experiments.noise import noise_cache as nc
from experiments.noise.controls import controls_enabled
from experiments.noise.n23_uncertainty_cleaning import (
    RowSink, PassCheckpoint, control_twins, as_cleanup_dict, CACHEABLE_MODELS,
    NO_NEIGHBOURS,
)

# Shares of the training molecules kept in the removal part. 1.0 is the
# reference every other share is paired against.
SHARES = (0.25, 0.5, 0.7, 0.8, 0.9, 1.0)

# Random draws of which molecules are kept, per share below 1.0.
REPEATS = 20

# The condition the injected part puts into the training labels. Gaussian moves
# every training label by a draw of mean zero, which is the error the ratio is
# defined for.
# DECIDED 2026-10-06 (Adelaide delegated): gaussian only.
CONDITION = 'gaussian'

# The replicate index of the injection. n07, n11, n13 and n24 use the same three.
NOISE_DRAWS = (0, 1, 2)

# How far each training label is moved toward its out-of-fold prediction.
ALPHAS = (0.25, 0.5, 1.0)

# Inner folds of the out-of-fold prediction. n19's OOF_SPLITS.
OOF_SPLITS = 5

# Each decided model's rank-1 and rank-2 single representation on the fold.
# They choose no features from the labels, so the injected part reuses them.
BASE_SPACES = ('base_1', 'base_2')

# The two of a model's four configurations whose features are chosen from the
# labels; the injected part rebuilds them on every (level, draw) from the
# noisy training labels.
HYBRID_CONFIGURATIONS = ('hybrid', 'hybrid__general2')

# Written into the checkpoint key and the rows file name (2026-10-05, chat 5),
# so passes and rows written by the whole-dataset design, which used the same
# space names, are never read back as done.
RESULT_LAYOUT = 'fold_aligned_2026-10-05'


def removal_seed(share, repeat, fold_i, n_training):
    """The seed for one random draw of kept molecules on one fold.

    zlib.crc32 and not hash(), for n24.replicate_seed's reason: hash() of a
    string changes per process, so the same command would keep different
    molecules on every run. The feature space is not in the key, so the same
    molecules are kept in every space and the space is the only difference
    between two rows.
    """
    key = f'{share:g}|{repeat}|{fold_i}|{n_training}'
    return int(zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF)


def kept_rows(share, repeat, fold_i, n_training):
    """Indices of the training molecules kept at this share, sorted."""
    n_keep = int(round(float(share) * int(n_training)))
    n_keep = min(max(n_keep, 2), int(n_training))
    rng = np.random.default_rng(removal_seed(share, repeat, fold_i, n_training))
    return np.sort(rng.choice(int(n_training), n_keep, replace=False))


def moved_toward(y, oof, alpha):
    """Each label moved a fraction alpha of the way to its out-of-fold prediction."""
    y = np.asarray(y, dtype=float)
    return (1.0 - float(alpha)) * y + float(alpha) * np.asarray(oof, dtype=float)


def _fit_predict(factory, X_tr, y_tr, X_va, metric_name):
    """One fit and its held-out predictions, or (None, why)."""
    try:
        m = factory()
        m.fit(X_tr, y_tr)
        return np.asarray(_predict_for_metric(m, X_va, metric_name),
                          dtype=float), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _fit_predict_cached(dataset_name, model_name, factory, X_tr, y_tr, X_va,
                        metric_name):
    """_fit_predict through noise_cache.cached_fit_predict for a seeded model
    (CACHEABLE_MODELS), and _fit_predict itself otherwise. Used for the fits
    plan section 3 lists as repeated: the correct-label fit and the fit on
    injected labels, which other experiments make for the same draw. The
    prediction rule is n07._predict_for_metric's (noise_cache.predict_mode_for),
    so a stored value equals a refit. Added 2026-09-25."""
    if dataset_name is None or model_name not in CACHEABLE_MODELS:
        return _fit_predict(factory, X_tr, y_tr, X_va, metric_name)
    try:
        return np.asarray(nc.cached_fit_predict(
            dataset_name, model_name, factory, X_tr, y_tr, X_va,
            predict_mode=nc.predict_mode_for(metric_name)),
            dtype=float), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _score(pred, y_va, metric_fn):
    """(the dataset's own metric, mean squared error), both against y_va."""
    y_va = np.asarray(y_va, dtype=float)
    return (float(metric_fn(y_va, pred)),
            float(np.mean((np.asarray(pred, dtype=float) - y_va) ** 2)))


def _levels_and_draws(data, dataset_name, levels=None, draws=None):
    """(level, draw) pairs to inject. Level 0.0 is dropped: the correct-label
    fit already on every fold is that row (level 0 = the labels as
    delivered, run once).

    2026-10-05 (chat 5): the levels are the decided ones for gaussian on this
    dataset's group (noise_helpers.decided_levels; KNN_SEARCH_SPACE_RECORD,
    "The noise grid"). Until then they were dataset_policy.noise_levels."""
    from experiments.noise.noise_helpers import decided_levels
    levels = (decided_levels(data, dataset_name, CONDITION)
              if levels is None else levels)
    # noise_helpers.draws_for (DECIDED 2026-10-06, Adelaide delegated): one
    # draw under censoring, the full list otherwise. n27 injects gaussian only,
    # so all three draws run; the call keeps the rule in one place.
    from experiments.noise.noise_helpers import draws_for
    draws = draws_for(CONDITION,
                      tuple(NOISE_DRAWS if draws is None else draws))
    return [(float(l), int(d)) for l in levels if float(l) > 0 for d in draws]


def run(data, config, output_dir, dataset_name, **kwargs):
    """N27: the cost of fewer training molecules, and how label error propagates."""
    from experiments.decided_grid import decided_models
    from experiments.noise.noise_helpers import (
        noise_outer_folds, fold_configurations, fold_model_inputs,
        configuration_record, inject_condition, dataset_knn_group,
        level_unit, CONFIGURATION_SPACES)

    print("\n" + "=" * 70)
    print("EXPERIMENT N27: REMOVAL COST AND HOW LABEL ERROR REACHES THE TEST ERROR")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    if task_type != 'regression':
        print(f"  SKIPPED: {dataset_name} is {task_type}. A variance of the "
              f"label error is not defined for a class label.")
        return {'skipped': 'not_regression', 'task_type': task_type}
    # Labels may be float64; the feature matrices stay float32 (handoff 2.3
    # item 18).
    labels = labels.astype(float)

    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    shares = sorted({float(s) for s in (kwargs.get('shares') or SHARES)}
                    | {1.0})
    repeats = int(kwargs.get('repeats') or REPEATS)
    alphas = [float(a) for a in (kwargs.get('alphas') or ALPHAS)]
    oof_splits = int(kwargs.get('oof_splits') or OOF_SPLITS)
    parts = list(kwargs.get('parts')
                 or ('removal_cost', 'injected', 'relabelled'))

    # R1 (chat 5, 2026-10-05): the dataset's two decided models
    # (DECISIONS.md Step 1), nothing else; knn runs only inside the distance
    # search. Replaces config['models'] (run_all hands every experiment five)
    # and ensure_dprime_models, which added rf and svm to every list. A
    # `models` kwarg (a smoke, a direct caller) can only narrow the decided
    # pair; a name outside it is dropped and written down.
    decided = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    asked = kwargs.get('models')
    if asked:
        models = [m for m in asked if m in decided]
        models_not_decided = [m for m in asked if m not in decided]
    else:
        models = list(decided)
        models_not_decided = []
    if config.get('models'):
        print(f"  config['models'] {list(config['models'])} is not read: "
              f"n27 fits the decided models {decided}")
    if kwargs.get('base_reps') is not None:
        print("  base_reps given by the caller are not read: each fold's base "
              "representations come from that fold's search")
    model_factories = get_model_factories(
        models, task_type=task_type, n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n27: no decided model could be built for {dataset_name}. "
            f"Decided {decided}, asked {list(asked or [])}, factories "
            f"{sorted(model_factories)}.")

    knn_grp = dataset_knn_group(data, dataset_name)
    passes = (_levels_and_draws(data, dataset_name, levels=kwargs.get('levels'),
                                draws=kwargs.get('draws'))
              if 'injected' in parts else [])
    noise_groups = noise_groups_for(data)[0] if passes else None

    folds, n_folds = noise_outer_folds(data, config)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'result_layout': RESULT_LAYOUT,
        'selection_scope': 'fold_train_rows',
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _, _ in folds],
        'models': models,
        'models_decided': decided,
        'models_asked_but_not_decided': models_not_decided,
        'models_without_factory': models_without_factory,
        'configurations_per_model': list(CONFIGURATION_SPACES),
        'parts': parts,
        'shares': shares,
        'repeats': repeats,
        'noise_condition': CONDITION,
        'knn_group': knn_grp,
        'level_unit': level_unit(CONDITION),
        'dose_units': noise_dose_units(dataset_name),
        'noise_levels': sorted({l for l, _ in passes}),
        'noise_draws': sorted({d for _, d in passes}),
        'alphas': alphas,
        'oof_splits': oof_splits,
        # DECIDED 2026-10-06 (Adelaide delegated): the out-of-fold prediction
        # the relabelled part moves labels toward stays a LightGBM on every
        # dataset (noise_cache.cached_oof_predict -> kirby.utils.noisy_labels
        # .oof_predict), fitted in each configuration's features. A fit inside
        # a method stays as it is; it is not one of the two decided models.
        # The injected part stays gaussian only, and kiba and bindingdb_kd
        # stay in.
        'relabel_target': ('out-of-fold LightGBM prediction on the fold\'s '
                           'training molecules, group-aware, in the '
                           'configuration\'s own features'),
        'held_out_graded_against': (
            'labels[val_idx] as the loader produced them, in every part. No '
            'moved or injected label is ever scored against.'),
        'features_note': (
            "Each decided model has four configurations per outer fold: its "
            "2 best single representations and its 2 best general hybrids, "
            "all ranked by searches on that fold's training rows "
            "(fold_configurations). removal_cost and relabelled: each "
            'configuration is built once per fold from ALL the training '
            'molecules at their correct labels, and the molecules are removed '
            'or the labels moved after the features were chosen. injected: '
            'the two hybrids are rebuilt on every pass from the noisy training '
            'labels, as n24 does. base_1 and base_2 do not depend on the '
            'labels.'),
        'removal_note': (
            'One row per random draw of the kept molecules. A draw is a '
            'replicate of the whole procedure: each fold draws its own '
            'molecules under the same draw index. The loss is paired against '
            'the share 1.0 fit on the same fold and the same features. '
            'Positive loss means the smaller training set scored worse.'),
        'ratio_note': (
            'propagation ratio = (held-out mean squared error of the fit on '
            'changed labels - held-out mean squared error of the fit on correct '
            'labels) / (mean over training molecules of (changed label - '
            'correct label) squared). Numerator and denominator are both in '
            'label units squared, and both are written beside every ratio.'),
        'per_fold': {},
    }

    n_cfg = len(models) * len(CONFIGURATION_SPACES)
    print(f"  {len(labels)} molecules, {n_folds} outer folds (running "
          f"{results['folds_run']}), decided models {models}, group {knn_grp}")
    print(f"  {n_cfg} fitted configurations per fold ({len(models)} models x "
          f"{len(CONFIGURATION_SPACES)})")
    print(f"  parts {parts}; shares {shares} x {repeats} draws; "
          f"{len(passes)} injected (level, draw) passes; alphas {alphas}")

    # ref[space][model][fold key] = {'score', 'mse'}
    ref = {}
    # removal[space][share][model][repeat][fold key] = {'score', 'mse'}
    removal = {}
    # injected[space][level][model][pass key] = {..., 'injected_variance'}
    injected = {}
    # relabelled[space][alpha][model][fold key] = {..., 'moved_variance'}
    relabelled = {}
    errors = {}
    # Per outer fold: what each model was fitted on (handoff 2.4 item 11).
    fold_records = {}
    t0 = time.time()

    # ── Rows and the per-fold checkpoint (2026-09-25) ──
    # The two parts that CHANGE the training data write D-versus-D' rows:
    # removal_cost (D' = a random share of the molecules) and relabelled (D'
    # = the labels moved toward the out-of-fold prediction). D is the fold's
    # training molecules at their correct labels, so score_D and score_clean
    # are both the share-1.0 fit. The injected part adds noise and removes
    # none, so it is a measurement, not a D', and writes no row.
    # Both carry RESULT_LAYOUT (2026-10-05) so the whole-dataset design's
    # rows and passes, under the same space names, are not read back.
    _tag = kwargs.get('tag')
    cond_tag = f'{_tag}__{RESULT_LAYOUT}' if _tag else RESULT_LAYOUT
    rows = RowSink(output_dir, 'n27_removal_cost_and_propagation',
                   dataset_name, experiment='n27', metric=metric,
                   condition=cond_tag, data=data, config=config)
    ckpt = PassCheckpoint(
        output_dir, 'n27_removal_cost_and_propagation', dataset_name,
        condition=cond_tag, data=data, config=config,
        source_file=__file__,
        key={'layout': RESULT_LAYOUT, 'parts': parts, 'shares': shares,
             'repeats': repeats, 'alphas': alphas, 'oof_splits': oof_splits,
             'passes': passes, 'models': list(models),
             'n_splits': int(n_folds), 'folds_run': results['folds_run']})
    done_folds = set()
    _state = ckpt.load()
    if _state:
        ref, removal, injected, relabelled, errors = (
            _state['ref'], _state['removal'], _state['injected'],
            _state['relabelled'], _state['errors'])
        fold_records = _state.get('fold_records', {})
        done_folds = set(_state['done'])
        print(f"  resumed from {ckpt.path}: {len(done_folds)} folds already "
              f"done")

    def _save_state():
        ckpt.save({'ref': ref, 'removal': removal, 'injected': injected,
                   'relabelled': relabelled, 'errors': errors,
                   'fold_records': fold_records,
                   'done': sorted(done_folds)})

    for fold_i, train_idx, val_idx in folds:
        fkey = f'f{fold_i}'
        if fkey in done_folds:
            print(f"    fold {fold_i}: on disk in the checkpoint")
            continue
        print(f"\n  Fold {fold_i} of {n_folds}: {len(train_idx)} training, "
              f"{len(val_idx)} held-out rows")
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx, n_folds=n_folds)
        except Exception as exc:
            why = (f'the per-fold searches failed: {type(exc).__name__}: '
                   f'{exc}')
            print(f"    {why}")
            errors[f'{fkey}|fold_configurations'] = why
            fold_records[str(fold_i)] = {'error': why}
            _save_state()
            continue
        y_tr = labels[train_idx]
        y_va = labels[val_idx]
        g_tr = groups_arr[train_idx] if groups_arr is not None else None
        n_tr = len(train_idx)

        # Each model's 4 configurations at the correct training labels.
        clean_inputs = {}
        frec = {'n_train': int(n_tr), 'n_val': int(len(val_idx)),
                'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_tr, factory=model_factories[m],
                metric=metric)
            clean_inputs[m] = ins
            rec = configuration_record(entry, names)
            rec['configurations_unavailable'] = dict(unav)
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            rec['training_row_score_note'] = (
                "hybrids[*].score_mean is the fold search's score of that "
                "hybrid on this fold's training rows (its own inner folds), "
                'in the search metric. held_out_score_correct_labels is each '
                'configuration fitted on every training molecule at its '
                'correct label and scored on the held-out rows.')
            rec['held_out_score_correct_labels'] = {}
            frec['models'][m] = rec
            for sp, why in unav.items():
                errors[f'{sp}|{m}|{fkey}|build'] = why
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        fold_records[str(fold_i)] = frec

        for mname in models:
            factory = model_factories[mname]
            for space, (X_tr_s, X_va_s) in clean_inputs[mname].items():
                # ── The correct-label fit: share 1.0, alpha 0, level 0 ──
                # Through noise_cache for a seeded model (2026-09-25).
                pred, err = _fit_predict_cached(
                    dataset_name, mname, factory, X_tr_s, y_tr, X_va_s,
                    metric_name)
                if pred is None:
                    errors[f'{space}|{mname}|{fkey}|reference'] = err
                else:
                    s, mse = _score(pred, y_va, metric_fn)
                    ref.setdefault(space, {}).setdefault(mname, {})[fkey] = {
                        'score': s, 'mse': mse}
                    frec['models'][mname]['held_out_score_correct_labels'][
                        space] = s
                _ref = ((ref.get(space) or {}).get(mname) or {}).get(fkey) or {}

                # ── removal_cost ──
                if 'removal_cost' in parts:
                    for share in shares:
                        if share >= 1.0:
                            continue
                        skey = f'{share:g}'
                        for rep_i in range(repeats):
                            keep = kept_rows(share, rep_i, fold_i, n_tr)
                            pred, err = _fit_predict(
                                factory, X_tr_s[keep], y_tr[keep], X_va_s,
                                metric_name)
                            if pred is None:
                                errors[f'{space}|{skey}|r{rep_i}|{mname}|'
                                       f'{fkey}'] = err
                                continue
                            s, mse = _score(pred, y_va, metric_fn)
                            (removal.setdefault(space, {})
                             .setdefault(skey, {}).setdefault(mname, {})
                             .setdefault(f'r{rep_i}', {}))[fkey] = {
                                'score': s, 'mse': mse,
                                'training_molecules_kept': int(len(keep)),
                                'training_molecules': int(n_tr)}
                            # D' = this random share. It is random by
                            # construction, so it has no random twin.
                            rows.add(method='removal_cost',
                                     method_setting=(f'share={share:g}|'
                                                     f'draw=r{rep_i}'),
                                     similarity_setting=NO_NEIGHBOURS,
                                     model_input=space, model=mname,
                                     noise_condition='none', level=None,
                                     draw=0, fold=fold_i,
                                     score_D=_ref.get('score'),
                                     score_Dprime=s,
                                     score_clean=_ref.get('score'))

                # ── relabelled ──
                if 'relabelled' in parts:
                    try:
                        # noise_cache.cached_oof_predict since 2026-09-25:
                        # the same call (seed 0), stored.
                        oof = nc.cached_oof_predict(dataset_name, X_tr_s, y_tr,
                                                    g_tr, n_splits=oof_splits)
                    except Exception as exc:
                        errors[f'{space}|{mname}|{fkey}|oof'] = (
                            f'{type(exc).__name__}: {exc}')
                        oof = None
                    if oof is not None:
                        for alpha in alphas:
                            akey = f'{alpha:g}'
                            y_mv = moved_toward(y_tr, oof, alpha)
                            moved_var = float(np.mean((y_mv - y_tr) ** 2))
                            pred, err = _fit_predict(
                                factory, X_tr_s, y_mv, X_va_s, metric_name)
                            if pred is None:
                                errors[f'{space}|a{akey}|{mname}|'
                                       f'{fkey}'] = err
                            else:
                                s, mse = _score(pred, y_va, metric_fn)
                                (relabelled.setdefault(space, {})
                                 .setdefault(akey, {})
                                 .setdefault(mname, {}))[fkey] = {
                                    'score': s, 'mse': mse,
                                    'moved_variance': moved_var}
                                rows.add(method='move_toward_oof',
                                         method_setting=f'alpha={alpha:g}',
                                         similarity_setting=NO_NEIGHBOURS,
                                         model_input=space, model=mname,
                                         noise_condition='none', level=None,
                                         draw=0, fold=fold_i,
                                         score_D=_ref.get('score'),
                                         score_Dprime=s,
                                         score_clean=_ref.get('score'))
                            # Controls (KIRBY_NOISE_CONTROLS=1 only): the
                            # label changes y' - y dealt to random molecules
                            # and permuted (controls.py).
                            _cd = as_cleanup_dict(X_tr_s, y_tr, labels=y_mv)
                            _seed = (int(fold_i), int(zlib.crc32(
                                f'{space}|{mname}|{akey}'.encode())
                                & 0x7FFFFFFF))
                            for cname, tw, why in control_twins(
                                    _cd, X_tr_s, y_tr, _seed):
                                if tw is None:
                                    errors[f'{space}|a{akey}|{cname}|'
                                           f'{mname}|{fkey}'] = why
                                    continue
                                pred, err = _fit_predict(
                                    factory, X_tr_s, tw['y'], X_va_s,
                                    metric_name)
                                rows.add(
                                    method='move_toward_oof',
                                    method_setting=(f'alpha={alpha:g}|'
                                                    f'control={cname}'),
                                    similarity_setting=NO_NEIGHBOURS,
                                    model_input=space, model=mname,
                                    noise_condition='none', level=None,
                                    draw=0, fold=fold_i,
                                    score_D=_ref.get('score'),
                                    score_Dprime=(None if pred is None else
                                                  _score(pred, y_va,
                                                         metric_fn)[0]),
                                    score_clean=_ref.get('score'))

        # ── injected ──
        for level, draw in passes:
            lkey = f'{level:g}'
            pkey = f'l{lkey}d{draw}{fkey}'
            try:
                # noise_helpers.inject_condition (chat 5): the same
                # noise_spec_for + inject_train_labels call as before, seed
                # base 42, the draw as the noise seed.
                y_noisy, _mask, _prov = inject_condition(
                    data, dataset_name, CONDITION, level, draw,
                    train_idx=train_idx, groups=noise_groups)
            except Exception as exc:
                errors[f'l{lkey}|d{draw}|{fkey}|inject'] = (
                    f'{type(exc).__name__}: {exc}')
                continue
            y_noisy = np.asarray(y_noisy, dtype=float)
            inj_var = float(np.mean((y_noisy - y_tr) ** 2))

            for mname in models:
                # base_1 and base_2 do not depend on the labels, so they are
                # reused. The two hybrids are rebuilt on the noisy labels, as
                # n24 does.
                noisy_inputs = {sp: X for sp, X in clean_inputs[mname].items()
                                if sp in BASE_SPACES}
                hyb, unav_h, _names = fold_model_inputs(
                    data, dataset_name, mname, entries.get(mname),
                    train_idx=train_idx, val_idx=val_idx, y_train=y_noisy,
                    factory=model_factories[mname], metric=metric,
                    spaces=HYBRID_CONFIGURATIONS)
                noisy_inputs.update(hyb)
                for sp, why in unav_h.items():
                    errors[f'{sp}|{pkey}|{mname}|build'] = why
                for space, (X_tr_s, X_va_s) in noisy_inputs.items():
                    # Through noise_cache for a seeded model: the fit on the
                    # injected labels is the no-cleaning fit other
                    # experiments make for the same draw (2026-09-25).
                    pred, err = _fit_predict_cached(
                        dataset_name, mname, model_factories[mname],
                        X_tr_s, y_noisy, X_va_s, metric_name)
                    if pred is None:
                        errors[f'{space}|{pkey}|{mname}|injected'] = err
                        continue
                    s, mse = _score(pred, y_va, metric_fn)
                    (injected.setdefault(space, {}).setdefault(lkey, {})
                     .setdefault(mname, {}))[pkey] = {
                        'score': s, 'mse': mse,
                        'injected_variance': inj_var, 'fold': fkey}

        # The fold is finished: its rows to disk, then the accumulators.
        rows.flush()
        done_folds.add(fkey)
        _save_state()
        print(f"    fold {fold_i} done [{time.time() - t0:.0f}s]")

    results['fold_configurations'] = convert_numpy(fold_records)
    results['per_fold'] = convert_numpy({
        'reference': ref, 'removal_cost': removal,
        'injected': injected, 'relabelled': relabelled})
    if errors:
        results['_errors'] = errors
    results['summary'] = convert_numpy(_summarise(
        ref, removal, injected, relabelled, lower_is_better, metric_name))
    results['rows'] = rows.summary()
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results)
    print(f"  D-versus-D' rows: {rows.n_written} written this run to "
          f"{rows.path or 'memory'}")
    ckpt.clear()
    return results


def _loss(base, got, lower_is_better):
    """How much worse `got` scored than `base`, in the dataset's metric.

    Positive means worse, whichever way the metric runs: (got - base) where
    lower wins and (base - got) where higher wins.
    """
    return (got - base) if lower_is_better else (base - got)


def _summarise(ref, removal, injected, relabelled, lower_is_better,
               metric_name):
    """The three parts, each per (space, model), paired on the fold."""
    out = {'metric': metric_name, 'lower_is_better': bool(lower_is_better),
           'removal_cost': {}, 'propagation': {}}

    # ── removal: one row per draw, best and worst, no mean across draws ──
    for space, by_share in removal.items():
        for share, by_model in by_share.items():
            for m, by_rep in by_model.items():
                base = (ref.get(space) or {}).get(m) or {}
                rows = {}
                for rep, by_fold in by_rep.items():
                    shared = sorted(set(base) & set(by_fold))
                    if not shared:
                        continue
                    losses = [_loss(base[k]['score'], by_fold[k]['score'],
                                    lower_is_better) for k in shared]
                    mse_losses = [by_fold[k]['mse'] - base[k]['mse']
                                  for k in shared]
                    # fold_spread (chat 5, 2026-10-05): one fold gives its
                    # value and 'one fold, no spread', never a division by 0.
                    _ls = fold_spread(losses)
                    _ms = fold_spread(mse_losses)
                    rows[rep] = {
                        'n_folds_paired': len(shared),
                        'folds_worse': int(sum(l > 0 for l in losses)),
                        'loss_per_fold': [float(x) for x in losses],
                        'loss_over_folds': _ls,
                        'mse_loss_over_folds': _ms,
                        'mean_loss_over_folds': _ls['mean'],
                        'mean_mse_loss_over_folds': _ms['mean'],
                    }
                if not rows:
                    continue
                means = [r['mean_loss_over_folds'] for r in rows.values()
                         if r['mean_loss_over_folds'] is not None]
                if not means:
                    continue
                out['removal_cost'].setdefault(space, {}).setdefault(
                    share, {})[m] = {
                    'per_draw': rows,
                    'draws_read': len(rows),
                    'best_draw_mean_loss': float(min(means)),
                    'worst_draw_mean_loss': float(max(means)),
                    'draws_worse_than_every_molecule': int(
                        sum(x > 0 for x in means)),
                    'share_1_score_over_folds': fold_spread(
                        [v['score'] for v in base.values()]),
                    'share_1_mean_score': fold_spread(
                        [v['score'] for v in base.values()])['mean'],
                }

    # ── propagation: injected and relabelled, per pass, side by side ──
    def _ratio_rows(block, var_key):
        per = {}
        for space, by_setting in block.items():
            for setting, by_model in by_setting.items():
                for m, by_pass in by_model.items():
                    base = (ref.get(space) or {}).get(m) or {}
                    rows = []
                    for pkey, v in sorted(by_pass.items()):
                        fk = v.get('fold', pkey)
                        if fk not in base:
                            continue
                        d_mse = v['mse'] - base[fk]['mse']
                        var = v[var_key]
                        rows.append({
                            'pass': pkey,
                            'held_out_mse_change': float(d_mse),
                            'training_label_variance_added': float(var),
                            'ratio': (float(d_mse / var) if var > 0
                                      else float('nan')),
                            'held_out_score_loss': float(_loss(
                                base[fk]['score'], v['score'],
                                lower_is_better)),
                        })
                    if not rows:
                        continue
                    num = float(np.mean([r['held_out_mse_change']
                                         for r in rows]))
                    den = float(np.mean([r['training_label_variance_added']
                                         for r in rows]))
                    # The spread over passes (folds x draws), fold_spread's
                    # rule: none from one pass (chat 5, 2026-10-05). It was
                    # np.std, which printed 0.0 for a single pass.
                    _rs = fold_spread([r['ratio'] for r in rows])
                    per.setdefault(space, {}).setdefault(m, {})[setting] = {
                        'per_pass': rows,
                        'n_passes': len(rows),
                        'mean_held_out_mse_change': num,
                        'mean_training_label_variance_added': den,
                        'ratio_of_the_two_means': (num / den if den > 0
                                                   else float('nan')),
                        'pass_ratios': _rs,
                        'spread_over_pass_ratios': _rs['sd'],
                    }
        return per

    inj = _ratio_rows(injected, 'injected_variance')
    rel = _ratio_rows(relabelled, 'moved_variance')
    for space in sorted(set(inj) | set(rel)):
        for m in sorted(set(inj.get(space, {})) | set(rel.get(space, {}))):
            out['propagation'].setdefault(space, {})[m] = {
                'injected_by_level': inj.get(space, {}).get(m, {}),
                'relabelled_by_alpha': rel.get(space, {}).get(m, {}),
            }
    return out


def _sd(v):
    return 'none, one pass' if v is None else f'{v:.4f}'


def _print_summary(results):
    s = results['summary']
    metric = s['metric']
    print("\n" + "=" * 70)
    print(f"SUMMARY ({metric}; every comparison paired on the fold)")

    print("\n  REMOVAL COST: loss against training on every molecule, "
          "positive = worse")
    for space, by_share in s['removal_cost'].items():
        for share, by_model in sorted(by_share.items()):
            for m, row in by_model.items():
                _s1 = row['share_1_mean_score']
                print(f"    {space}/{m} keeping {share} of the training "
                      f"molecules: over {row['draws_read']} draws, best "
                      f"{row['best_draw_mean_loss']:+.4f}, worst "
                      f"{row['worst_draw_mean_loss']:+.4f} {metric} "
                      f"(every molecule: "
                      f"{'n/a' if _s1 is None else f'{_s1:.4f}'}); "
                      f"{row['draws_worse_than_every_molecule']} of "
                      f"{row['draws_read']} draws scored worse")

    print("\n  PROPAGATION: held-out MSE change, training-label variance "
          "added, ratio")
    for space, by_model in s['propagation'].items():
        for m, blk in by_model.items():
            for lvl, r in sorted(blk['injected_by_level'].items()):
                print(f"    {space}/{m} injected level {lvl}: "
                      f"{r['mean_held_out_mse_change']:+.5f} / "
                      f"{r['mean_training_label_variance_added']:.5f} = "
                      f"{r['ratio_of_the_two_means']:+.4f} over "
                      f"{r['n_passes']} passes (spread of per-pass ratios "
                      f"{_sd(r['spread_over_pass_ratios'])})")
            for a, r in sorted(blk['relabelled_by_alpha'].items()):
                print(f"    {space}/{m} moved toward prediction, alpha {a}: "
                      f"{r['mean_held_out_mse_change']:+.5f} / "
                      f"{r['mean_training_label_variance_added']:.5f} = "
                      f"{r['ratio_of_the_two_means']:+.4f} over "
                      f"{r['n_passes']} folds (spread of per-fold ratios "
                      f"{_sd(r['spread_over_pass_ratios'])})")
    print("\n  Both variances are in label units squared, averaged over "
          "training molecules.")


def main():
    parser = create_experiment_parser(
        'N27: removal cost and how label error reaches the test error')
    parser.add_argument('--shares', type=str, default=None,
                        help='Comma-separated shares of training molecules kept '
                             f'(default {",".join(str(s) for s in SHARES)}); '
                             '1.0 is always added as the reference')
    parser.add_argument('--repeats', type=int, default=None,
                        help=f'Random draws of kept molecules per share '
                             f'(default {REPEATS})')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated gaussian noise levels. Default: '
                             'the shared grid for this dataset, 0.0 dropped')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Comma-separated fractions each label is moved '
                             'toward its out-of-fold prediction '
                             f'(default {",".join(str(a) for a in ALPHAS)})')
    parser.add_argument('--draws', type=str, default=None,
                        help='Comma-separated injection replicates '
                             f'(default {",".join(str(d) for d in NOISE_DRAWS)})')
    parser.add_argument('--parts', type=str, default=None,
                        help='Comma-separated: removal_cost, injected, '
                             'relabelled (default all three)')
    parser.add_argument('--exp-folds', type=int, default=None,
                        help='Outer folds. Overrides config n_splits (the '
                             '--mode default); KIRBY_NOISE_N_FOLDS overrides '
                             'both')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra text on the result filename')
    args = parser.parse_args()
    config = get_config(args)
    if args.exp_folds:
        config['n_splits'] = int(args.exp_folds)
    data = load_data(args.dataset, rep_preset=args.reps)

    kw = {}
    if args.shares:
        kw['shares'] = [float(x) for x in args.shares.split(',')]
    if args.repeats:
        kw['repeats'] = int(args.repeats)
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.alphas:
        kw['alphas'] = [float(x) for x in args.alphas.split(',')]
    if args.draws:
        kw['draws'] = [int(x) for x in args.draws.split(',')]
    if args.parts:
        kw['parts'] = [x.strip() for x in args.parts.split(',')]

    res = run(data, config, args.output, args.dataset, **kw)
    save_experiment_results(res, args.output,
                            'n27_removal_cost_and_propagation', args.dataset,
                            filename_suffix=args.tag)


if __name__ == '__main__':
    main()
