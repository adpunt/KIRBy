#!/usr/bin/env python3
"""
Humanness Classification Experiment (OPIG B3)

Extends Prihoda et al. 2022 "BioPhi" (mAbs) — a score discrimination study
comparing 8 humanness scoring methods on IMGT/mAb-DB therapeutic antibodies.
BioPhi does NOT train classifiers; it computes ROC AUC via threshold sweep
on the full dataset (no train/test split, no CV). KIRBy instead trains
supervised classifiers on high-dimensional reps with proper CV. The two
classifiers are the ones recorded for this dataset (xgboost and rf); the
docstring said Ridge and RF, and ridge was removed from the repo in 2026-07.

NOTE: Previous versions cited Marks et al. 2021 (Hu-mAb) as primary reference.
Marks 2021 is one of the 8 methods BioPhi evaluates, not the data source.
Data source = BioPhi's IMGT/mAb-DB curation (Prihoda et al. 2022).

Scientific questions:
  1. Which antibody rep is best for humanness classification?
  2. Does hybrid augmentation improve over the best individual rep?
  3. How do sequence-only LM reps compare to the classical encoding (BLOSUM)?
     (The question used to name onehot as well. onehot is not one of the
     dataset's approved representations, so on 2026-10-05 it left this
     experiment altogether: the table of single representations and the
     per-fold searches both run over the approved pool. BLOSUM is the one
     classical encoding left.)

Datasets:
  - therapeutic: Human vs non-human real therapeutics (BioPhi IMGT, ~553)
  (species, OAS 10K balanced, is not registered in DATASET_REGISTRY.)

Paper baselines (Prihoda et al. 2022, BioPhi):
  - 8 methods scored on IMGT/mAb-DB therapeutics (VH+VL averaged):
    Hu-mAb:    AUROC=97.7%  Accuracy=93.3%  (best overall)
    OASis:     AUROC=97.2%  Accuracy=94.4%  (relaxed threshold, their method)
    Germline:  AUROC=96.8%  Accuracy=92.8%
  - No train/test split — full-dataset threshold sweep, NOT comparable to CV.
  - KIRBy uses VH-only; BioPhi averages VH+VL scores.

Usage:
    # Full evaluation (the cluster line, submit_everything.sh:133)
    python tests/experiments/opig_humanness.py --datasets humanness_therapeutic

    # Outer folds: --n-folds, default config['n_splits'] from get_config
    # (5, or KIRBY_LOCAL_FOLDS). 1 = one grouped held-out split.
    python tests/experiments/opig_humanness.py --n-folds 1

Reference:
    Prihoda et al. (2022). BioPhi: A platform for antibody design,
    humanization and humanness evaluation. mAbs 14(1):2020203.
    Marks et al. (2021). J Mol Biol 433(11):166928. (Hu-mAb, one of 8 methods)
    Ramon et al. (2024). Bioinformatics 40(6):btae332. (AbNatiV, species data)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

from antibody_baseline_survey import load_dataset, DATASET_REGISTRY, _identity_groups
from experiments._shared import (add_distance_metric_arg,
                                apply_distance_metric_arg)
from experiments._shared import (
    convert_numpy, get_model_factories, get_config, as_rep_dtype,
    fold_standard_selection, build_fold_hybrid, fold_hybrid_scoring_kwargs,
)
from experiments.decided_grid import decided_models
from kirby.result_cache import StaleResultDB
# 0/1 features pass through, continuous features are standardised, in the
# table of single representations as well as inside a hybrid (2026-10-05).
from kirby.scaling import FeatureScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

# humanness_species is not a key of DATASET_REGISTRY (antibody_baseline_survey.py
# registers humanness_therapeutic only), so naming it here made main's validation
# loop print ERROR and return 1 before any work whenever --datasets was left off.
# The cluster submitter passes --datasets humanness_therapeutic and escaped that;
# a bare local run did not.
DEFAULT_DATASETS = ['humanness_therapeutic']

# Representations come from DATASET_REPS for each dataset — the pool the
# rep-pruning step approved and precompute builds. The hand-written list this
# replaced asked for aac, esm2, antiberty and paratope, none of which are in
# humanness_therapeutic's pool and none of which this experiment does anything
# special with; it also skipped 5 of the 10 that are.
#
# 2026-10-05: onehot is gone from this experiment. It was added on top of the
# pool for scientific question 3 (LM reps against the classical encodings),
# measured as a single representation and kept out of the searches. The
# representations an experiment may use are the approved pool and the
# augmentation list, and nothing else; humanness_therapeutic is not one of the
# two recorded exceptions. BLOSUM is in the pool, so question 3 keeps one
# classical encoding instead of two. humanness_therapeutic's augmentation list
# is empty, so the pool is its 10 general representations and a representation
# count of 'all' means those 10.
#
# NOTE: LLR reps (esmc_llr, antiberty_llr) stay out — they require mutation
# lists, but the humanness loader sets mutations=[[] for _ in range(n)]
# (see src/kirby/datasets/humanness.py:244,322), which makes
# src/kirby/representations/mutation.py:490-494 emit all-zero 3D vectors per sample.
REP_EXCEPTIONS = []


def _default_reps(dataset_name):
    from experiments._shared import approved_rep_pool
    return approved_rep_pool(dataset_name, extra=REP_EXCEPTIONS)


def _decided_models(dataset_name):
    """The dataset's two downstream models, from the record (Step 1).

    humanness_therapeutic is xgboost and rf. This file used to use
    DOWNSTREAM_MODELS (rf, xgboost, svm, dnn, lightgbm) for the table of single
    representations, and the hybrid search it read also scored gp, so six
    models were reported where two are decided. knn is scored inside the
    distance search only (Step 9), and no knn row is read here.
    """
    return list(decided_models(dataset_name, 'general'))


# The outer split every number in this file is measured on. One seed, one
# splitter: the folds come from kirby.find_best_hybrid.nested_cv_outer_folds,
# which is also what each fold's search is filed under.
SEED = 42

# The decided grid IS the cut. An adaptive-pruning schedule votes members off
# the representation-count, importance-method and internal-evaluator axes after
# two folds (find_best_hybrid.py:3789-3890) and drops configurations at the
# checkpoint budgets, so it would delete axis members the record keeps.
# 2026-10-05: no pruning argument of any kind is passed to a search here. This
# file used to hand every search a schedule of its own with the voting emptied
# out; search_on_fold now turns pruning off for every per-fold search itself.

# How many hybrids and how many single representations are selected per model
# per outer fold. 2 and 2 is the standard selection (2026-10-03).
#
# 2026-10-05: the one hybrid from the distance search is gone. A distance
# hybrid belongs to an experiment that measures something with neighbours or
# distances; this one classifies human against non-human with xgboost and rf
# and reads no neighbour list, so its arms are the 2 general hybrids. The file
# read a distance hybrid before through get_rule_hybrids' RULE_N_DISTANCE=1.
N_BASE = 2
N_GENERAL_HYBRIDS = 2


# The smoke-only search shrink that used to live here (one tiny budget, one
# internal evaluator, lightgbm_gain alone) is gone: every axis of a search now
# comes from the record, and naming a budget or an evaluator the record does
# not keep raises. KIRBY_SMOKE_N still subsamples the ROWS, which is the part
# of a smoke run that is allowed to be small.


# Published reference baselines
# Prihoda et al. 2022 (BioPhi): threshold-sweep on full therapeutic dataset,
# NO train/test split. Numbers from notebook 03_oasis_humanness.ipynb Cell 49.
# Note: BioPhi uses VH+VL averaged. KIRBy uses VH-only.
PAPER_BASELINES = {
    # No 'species' entry — Marks 2021 reports near-perfect AUROC on OAS
    # species-level but doesn't publish numbers, and BioPhi doesn't either.
    # Comparison block is skipped when this key is missing.
    'therapeutic': {
        # Human vs non-human therapeutics (BioPhi Cell 49, VH+VL averaged)
        'Hu-mAb (Marks)':      {'auroc': 0.977, 'accuracy': 0.933},
        'OASis relaxed':       {'auroc': 0.972, 'accuracy': 0.944},
        'Germline content':    {'auroc': 0.968, 'accuracy': 0.928},
        'IgReconstruct':       {'auroc': 0.967, 'accuracy': 0.926},
        'MG Score':            {'auroc': 0.959, 'accuracy': 0.918},
        'AbLSTM':              {'auroc': 0.937, 'accuracy': 0.878},
        'T20':                 {'auroc': 0.896, 'accuracy': 0.839},
        'Z-score':             {'auroc': 0.837, 'accuracy': 0.765},
    },
}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rep_clf(X, labels, groups, splits, model):
    """Evaluate one (rep, model) pair for classification on the outer folds.

    splits: the (train rows, test rows) pairs to use, required. The caller
    passes the same outer folds the per-fold selection below is built on, so
    one fold number means the same rows everywhere in this file. They are the
    folds kirby.find_best_hybrid.nested_cv_outer_folds returns, which for a
    classification dataset with groups is StratifiedGroupKFold(n_splits,
    shuffle=True, random_state=seed). This function used to build a splitter of
    its own from n_folds when splits was None, which was a second place the
    fold count was set; the caller is now the only one (2026-10-05).

    Returns dict with accuracy, f1_weighted, auroc, balanced_accuracy
    (mean, and spread over the folds; 'spread' is None on a single fold).
    """
    # The representation keeps the dtype it was loaded at (float32). This cast
    # to float64 changed tree scores (2026-10-05).
    X = as_rep_dtype(X)
    labels = np.asarray(labels, dtype=np.int64)

    acc_scores, f1_scores, auroc_scores, bacc_scores = [], [], [], []

    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # FeatureScaler standardises the continuous features and passes the
        # 0/1 features through, per feature. Before 2026-10-05 this skipped
        # scaling only when EVERY value of the whole matrix was 0 or 1, so a
        # representation with both kinds of feature had its 0/1 features
        # standardised.
        scaler = FeatureScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        try:
            m = clone(model)
            m.fit(X_train, y_train)
        except Exception as e:
            # Don't swallow fold-fit failures silently (OPIG-18): a dropped
            # fold biases the mean. Log the model + exception type and skip.
            print(f"      [fold fit failed: {type(model).__name__} "
                  f"{type(e).__name__}: {e}]", flush=True)
            continue

        preds = m.predict(X_val)
        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average='weighted'))
        bacc_scores.append(balanced_accuracy_score(y_val, preds))

        # AUROC (binary only, need decision function or probabilities)
        try:
            if hasattr(m, 'decision_function'):
                scores = m.decision_function(X_val)
            elif hasattr(m, 'predict_proba'):
                scores = m.predict_proba(X_val)[:, 1]
            else:
                scores = preds.astype(float)
            auroc_scores.append(roc_auc_score(y_val, scores))
        except Exception:
            pass

    if not acc_scores:
        return None

    def _agg(vals):
        # One fold has no spread over folds. 'std' of one number is 0.0, which
        # reads as agreement between folds there are none of (2026-10-05).
        return {'mean': float(np.mean(vals)),
                'spread': float(np.std(vals)) if len(vals) > 1 else None,
                'n_folds': len(vals),
                'per_fold': [float(v) for v in vals]}

    result = {
        'accuracy': _agg(acc_scores),
        'f1_weighted': _agg(f1_scores),
        'balanced_accuracy': _agg(bacc_scores),
    }
    if auroc_scores:
        result['auroc'] = _agg(auroc_scores)
    return result


def evaluate_dataset(dataset_name, rep_names, n_folds, fold_ck=None):
    """Evaluate all reps × models on one humanness dataset.

    fold_ck: the run's checkpoint. Each outer fold's selection is written to
    it as soon as that fold is scored, and a fold already in it is read back
    instead of re-searched (2026-10-05, dirty-run bug 12: this script held
    every fold in memory and wrote one JSON after the last one, so a job
    killed at the wall clock lost every fold it had finished).
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    data = load_dataset(dataset_name)
    from experiments._shared import _smoke_subsample
    data = _smoke_subsample(data)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    n = len(labels)
    n_pos = int((labels == 1).sum())
    print(f"  {n} samples, {n_pos} human ({100*n_pos/n:.1f}%), "
          f"{n - n_pos} non-human ({100*(n-n_pos)/n:.1f}%)")

    # Identity groups via NEAR-duplicate clustering (ISS-29/30 leakage fix, §B).
    # Delegates to the SAME helper the main antibody survey uses
    # (antibody_baseline_survey._identity_groups -> sequence_identity_clusters:
    # mandatory Levenshtein, single-linkage at 95% identity, canonical cluster IDs).
    # This REPLACES the old exact-match grouping, which placed single-indel
    # near-duplicate VH sequences in DIFFERENT groups -> they leaked across the
    # StratifiedGroupKFold boundary, the exact leakage §B exists to prevent.
    # Effect by preset: species (~10K near-unique OAS VH) stays ~one-group-per-sample;
    # therapeutic (INN entries sharing near-identical VH) now co-clusters those.
    groups, _n_groups = _identity_groups(heavy_seqs, light_seqs)

    # Generate representations (with disk caching for expensive LM reps)
    from experiments.rep_cache import compute_or_load_reps
    reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=dataset_name,
        mutations=data.get('mutations'),
    )

    if not reps:
        print("  No reps available, skipping dataset")
        return None

    # The per-fold selection. NO whole-dataset search is read anywhere in this
    # file: the readers that did (get_best_search_for_downstream,
    # build_fold_reselection_input, get_rule_hybrids, select_model_rep_pairs,
    # top_hybrids, hybrid_deltas, extract_top3_base_reps) all ranked recipes on
    # per-fold scores taken from ONE search fitted on every antibody, the
    # held-out fold included, and the helper's own metadata said
    # selection_scope='whole_dataset'. Leaving fold k's score out of a mean does
    # not leave fold k's rows out of the models that produced it.
    #
    # Now each outer fold gets its own search on its own training rows, and the
    # 2 single representations and 2 hybrids it picks are scored on the rows
    # that search never saw. No 'kirby_hybrid' representation is added to
    # `reps` (that arm was built on every antibody's label and was deleted
    # 2026-09-04, split audit). humanness_therapeutic has no augmentation
    # features, so every search runs the 'none' condition only.
    models_list = _decided_models(dataset_name)
    fold_block, fold_error = _per_fold_selection(
        reps, labels, groups, dataset_name, models_list, n_folds,
        fold_ck=fold_ck)

    # Evaluate all reps × models on the same outer folds
    models = get_model_factories(models_list, 'classification', n_samples=n)
    splits = fold_block['splits']
    print(f"\n  Evaluating {len(reps)} reps × {len(models)} models × "
          f"{len(splits)}-fold CV...")
    results = {}
    for rep_name in sorted(reps.keys()):
        X = reps[rep_name]
        rep_results = {}
        for model_name, factory in models.items():
            t0 = time.time()
            try:
                res = evaluate_rep_clf(X, labels, groups, splits, factory())
            except ValueError as e:
                print(f"    {rep_name:25s} {model_name:8s}  SKIPPED: {e}")
                continue
            elapsed = time.time() - t0
            if res is not None:
                rep_results[model_name] = res
                acc = res['accuracy']['mean']
                auroc = res.get('auroc', {}).get('mean', float('nan'))
                print(f"    {rep_name:25s} {model_name:8s}  "
                      f"acc={acc:.4f}  auroc={auroc:.4f}  ({elapsed:.1f}s)")
        if rep_results:
            results[rep_name] = {
                'dim': X.shape[1],
                'models': rep_results,
            }

    out = {
        'n_samples': n,
        'n_human': n_pos,
        'n_nonhuman': n - n_pos,
        'metric': 'auc',
        'metric_lower_is_better': False,
        'models': models_list,
        'n_outer_folds': len(splits),
        'seed': SEED,
        # One list: the representations the searches run over and the ones the
        # table of single representations covers are the same approved pool.
        'search_pool': sorted(reps.keys()),
        'table_reps': sorted(reps.keys()),
        'results': results,
    }
    out.update({k: v for k, v in fold_block.items() if k != 'splits'})
    if fold_error:
        # Not stored by the checkpoint and the script exits 1: a run that
        # measured no hybrid at all used to be written out as a finished item
        # (base table only, no 'error' key), read back on every later attempt,
        # and the only sign was one printed warning line.
        out['error'] = fold_error
    return out


def _per_fold_selection(reps, labels, groups, dataset_name, models_list,
                        n_folds, fold_ck=None):
    """Per outer fold, per model: 2 single representations + 2 hybrids.

    All four are the standard selection (2026-10-03), ranked by
    fold_standard_selection on a search over this fold's TRAINING rows only,
    on the dataset's decided grid, and then rebuilt and scored on the held-out
    rows. The hybrid from the distance search that this file also scored until
    2026-10-05 is gone (see N_GENERAL_HYBRIDS).

    fold_ck: a fold already in the checkpoint is read back; a fold that is
    scored is written to it before the next one starts.

    Returns ({'splits': ..., 'per_fold': [...], 'fold_summary': {...},
              'fold_failures': [...], 'selection_scope': 'fold_train_rows'},
             error string or None).
    """
    from kirby.find_best_hybrid import (nested_cv_outer_folds, get_model,
                                        _outer_fit_score, BUILTIN_METRICS)
    metric_fn, lower_is_better = BUILTIN_METRICS['auc']

    labels = np.asarray(labels)
    groups_arr = np.asarray(groups) if groups is not None else None
    # The single source of the outer split: the same function each fold's
    # search is filed under, so a fold's cache namespace names the rows the
    # score is read on.
    splits = nested_cv_outer_folds(reps, labels, groups_arr, n_folds,
                                   SEED, 'classification')
    n_outer = len(splits)

    block = {'splits': splits, 'selection_scope': 'fold_train_rows',
             'selection_rule': (f'{N_BASE} best single representations and '
                                f'{N_GENERAL_HYBRIDS} best hybrids of the '
                                f'general search, per model, per outer fold, '
                                f'ranked on that fold\'s training rows'),
             'metric': 'auc', 'metric_lower_is_better': lower_is_better,
             'per_fold': [], 'fold_failures': [], 'fold_summary': {}}

    def _score(X_tr, X_te, y_tr, y_te, model_name, train_groups, test_idx,
               test_groups):
        """One feature matrix, fitted on the training rows, scored on the rest.

        _outer_fit_score is the engine's own held-out scorer: it fits the
        scaler on the training rows, builds the model by name at this row
        count, and reads the AUROC off predicted probabilities.
        """
        score, metrics = _outer_fit_score(
            X_tr, y_tr, X_te, y_te, model_name, 'classification', metric_fn,
            seed=SEED, train_groups=train_groups, test_idx=test_idx,
            test_groups=test_groups)
        return float(score), metrics

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        # Every failure of this fold, saved with the fold's own row rather than
        # printed and dropped (2026-10-05, results-audit item 2.13). Each
        # record names the dataset, the fold, the model where there is one,
        # and the exception.
        fold_fail = []
        ck_key = dict(part='per_fold_selection', dataset=dataset_name,
                      fold=int(fold_i), n_outer_folds=int(n_outer),
                      models=list(models_list))
        if fold_ck is not None and fold_ck.done(**ck_key):
            stored = fold_ck.get(**ck_key)
            block['per_fold'].append(stored['fold_row'])
            block['fold_failures'].extend(stored.get('fold_failures') or [])
            print(f"\n  Fold {fold_i}: finished in an earlier attempt, "
                  f"read back")
            continue
        y_tr, y_te = labels[train_idx], labels[test_idx]
        g_tr = groups_arr[train_idx] if groups_arr is not None else None
        g_te = groups_arr[test_idx] if groups_arr is not None else None
        tr_reps = {k: v[train_idx] for k, v in reps.items()}
        te_reps = {k: v[test_idx] for k, v in reps.items()}
        print(f"\n  Fold {fold_i}: {len(train_idx)} training rows, "
              f"{len(test_idx)} held out")

        # No pruning argument: search_on_fold runs every per-fold search
        # unpruned, and the decided grid is the only cut (2026-10-05).
        try:
            general = fold_standard_selection(
                reps, labels=labels, groups=groups_arr, train_idx=train_idx,
                fold_i=fold_i, n_outer_folds=n_outer,
                dataset_name=dataset_name, task_type='classification',
                metric='auc', augmentations=None, seed=SEED,
                models=models_list, n_base=N_BASE,
                n_hybrids=N_GENERAL_HYBRIDS)
        except Exception as e:
            fold_fail.append({'dataset': dataset_name, 'fold': int(fold_i),
                              'model': None, 'what': 'general search',
                              'error_type': type(e).__name__,
                              'error': str(e)})
            block['fold_failures'].extend(fold_fail)
            print(f"    general search failed: {type(e).__name__}: {e}")
            continue

        fold_row = {'fold': fold_i, 'n_train': int(len(train_idx)),
                    'n_held_out': int(len(test_idx)),
                    'selection_scope': 'fold_train_rows', 'models': {}}
        for m in models_list:
            gsel = general.get(m)
            if gsel is None:
                fold_fail.append(
                    {'dataset': dataset_name, 'fold': int(fold_i), 'model': m,
                     'what': 'general search',
                     'error_type': 'NoSearchResult',
                     'error': ('the per-fold general search produced nothing '
                               'for this model')})
                continue
            row = {'base': [], 'hybrids': [], 'not_scored': [],
                   'hybrids_skipped_identical_features':
                       gsel.get('hybrids_skipped_identical_features') or []}

            # The 2 single representations this fold's search ranked highest.
            for rank, rep in enumerate(gsel['base']):
                try:
                    s, _ = _score(reps[rep][train_idx], reps[rep][test_idx],
                                  y_tr, y_te, m, g_tr, test_idx, g_te)
                except Exception as e:
                    row['not_scored'].append(
                        f'base {rep}: {type(e).__name__}: {e}')
                    fold_fail.append(
                        {'dataset': dataset_name, 'fold': int(fold_i),
                         'model': m, 'what': f'base representation {rep}',
                         'error_type': type(e).__name__, 'error': str(e)})
                    continue
                row['base'].append({
                    'rank': rank + 1, 'rep': rep,
                    'score_train_rows': _entry_score(gsel['base_detail'],
                                                     rank),
                    'score_held_out': s,
                })

            # The hybrids: the general search's best 2, each rebuilt on this
            # fold's training rows from its own blueprint and applied to the
            # held-out rows.
            slots = [(f'general{i + 1}', 'general', h)
                     for i, h in enumerate(gsel['hybrids'])]
            for slot, tier, h in slots:
                try:
                    rec = _score_hybrid(h, tr_reps, te_reps, y_tr, y_te, m,
                                        g_tr, test_idx, g_te, _score,
                                        get_model, metric_fn, lower_is_better)
                except Exception as e:
                    row['not_scored'].append(
                        f'{slot}: {type(e).__name__}: {e}')
                    fold_fail.append(
                        {'dataset': dataset_name, 'fold': int(fold_i),
                         'model': m, 'what': f'hybrid {slot}',
                         'error_type': type(e).__name__, 'error': str(e)})
                    print(f"    {m} {slot}: not scored -- "
                          f"{type(e).__name__}: {e}")
                    continue
                rec.update({'slot': slot, 'tier': tier})
                row['hybrids'].append(rec)

            # The one number per model per fold: the top-ranked hybrid against
            # the top-ranked single representation, both ranked without the
            # held-out rows, so the difference is held out too.
            best_b = row['base'][0]['score_held_out'] if row['base'] else None
            best_h = next((h['score_held_out'] for h in row['hybrids']
                           if h['slot'] == 'general1'), None)
            row['top_base_held_out'] = best_b
            row['top_hybrid_held_out'] = best_h
            row['delta_held_out'] = (None if best_b is None or best_h is None
                                     else best_h - best_b)
            fold_row['models'][m] = row
            print(f"    {m}: best single {_fmt(best_b)}, "
                  f"top hybrid {_fmt(best_h)}, "
                  f"difference {_fmt(row['delta_held_out'], signed=True)}")
        block['per_fold'].append(fold_row)
        block['fold_failures'].extend(fold_fail)
        # Written before the next fold starts, so a job killed at the wall
        # clock keeps the folds it scored (dirty-run bug 12).
        if fold_ck is not None:
            fold_ck.put({'fold_row': fold_row, 'fold_failures': fold_fail},
                        **ck_key)

    # Mean over the folds that scored both arms, per model. Every per-fold
    # number is in 'per_fold' above, so a reader can re-pair them any way.
    for m in models_list:
        deltas, hyb, base = [], [], []
        for fr in block['per_fold']:
            r = fr['models'].get(m)
            if not r or r.get('delta_held_out') is None:
                continue
            # A fold whose fit returned no number contributes nothing rather
            # than turning the mean into NaN.
            if not np.isfinite(r['delta_held_out']):
                continue
            deltas.append(r['delta_held_out'])
            hyb.append(r['top_hybrid_held_out'])
            base.append(r['top_base_held_out'])
        block['fold_summary'][m] = {
            'n_folds_both_arms': len(deltas),
            'mean_top_hybrid_auc': float(np.mean(hyb)) if hyb else None,
            'mean_top_single_auc': float(np.mean(base)) if base else None,
            'mean_difference_auc': float(np.mean(deltas)) if deltas else None,
            # The spread of the difference over the folds. One fold has none;
            # the number would be 0.0, which reads as folds agreeing
            # (2026-10-05).
            'difference_spread_over_folds': (float(np.std(deltas))
                                             if len(deltas) > 1 else None),
            'per_fold_difference_auc': [float(d) for d in deltas],
        }

    scored = sum(len(fr['models']) for fr in block['per_fold'])
    error = None
    if scored == 0:
        reasons = ['{} fold {} {}: {}: {}'.format(
            f.get('dataset'), f.get('fold'), f.get('model') or f.get('what'),
            f.get('error_type'), f.get('error'))
            for f in block['fold_failures'][:5]]
        error = ('no model was selected and scored on any outer fold: '
                 + '; '.join(reasons or ['no reason recorded']))
    return block, error


def _entry_score(base_detail, rank):
    """The training-rows score of the rank-th single representation."""
    # top_base_reps names it 'score' (_shared.py:7886); top_hybrids names its
    # own 'score_mean'.
    sel = (base_detail or {}).get('selected') or []
    if rank < len(sel):
        return sel[rank].get('score')
    return None


def _fmt(v, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def _score_hybrid(entry, tr_reps, te_reps, y_tr, y_te, model_name, train_groups,
                  test_idx, test_groups, score_fn, get_model, metric_fn,
                  lower_is_better):
    """Rebuild one hybrid on this fold's training rows and score the rest.

    The blueprint comes from this fold's own search, so the allocation scores
    it carries — the ones that say WHICH representations a recipe with a
    representation count uses — were computed on these training rows and not on
    the whole dataset.
    """
    bp = entry['blueprint']
    model = get_model(model_name, 'classification', n_samples=len(y_tr))
    X_tr_h, X_te_h, feat_info = build_fold_hybrid(
        tr_reps, te_reps, y_tr, bp, groups=train_groups,
        precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
        **fold_hybrid_scoring_kwargs(bp, model=model, metric_fn=metric_fn,
                                     lower_is_better=lower_is_better))
    score, _ = score_fn(X_tr_h, X_te_h, y_tr, y_te, model_name, train_groups,
                        test_idx, test_groups)
    # Everything a reader needs to say which configuration this was. The
    # Blueprint object itself is not JSON and every field reported here is
    # already a key of the entry.
    return {
        'allocation_method': entry.get('allocation_method'),
        'importance_method': entry.get('importance_method'),
        'budget': entry.get('budget'),
        'internal_evaluator': entry.get('internal_evaluator'),
        'rep_count': entry.get('rep_count'),
        'rep_count_source': entry.get('rep_count_source'),
        'step_size': entry.get('step_size'),
        'aug_condition': entry.get('aug_condition'),
        'label': entry.get('label'),
        'reps_used': sorted((feat_info.get('allocation') or {}).keys())
                     or sorted(getattr(bp, 'reps', []) or []),
        'n_features': int(X_tr_h.shape[1]),
        'score_train_rows': entry.get('score_mean'),
        'score_held_out': score,
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def print_analysis(dataset_name, ds_results):
    """Print comparison table with paper baselines."""
    if ds_results is None:
        return

    results = ds_results['results']
    baselines = PAPER_BASELINES.get(dataset_name.replace('humanness_', ''), {})

    print(f"\n{'=' * 90}")
    print(f"HUMANNESS EXPERIMENT RESULTS: {dataset_name}")
    print(f"  {ds_results['n_samples']} samples "
          f"({ds_results['n_human']} human, {ds_results['n_nonhuman']} non-human)")
    print(f"{'=' * 90}")

    # Header
    print(f"\n  {'Rep':25s} {'Model':8s} {'Acc':>10s} {'AUROC':>10s} {'F1w':>10s} {'BalAcc':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Paper baselines first
    for method, metrics in baselines.items():
        acc = metrics.get('accuracy', float('nan'))
        auroc = metrics.get('auroc', float('nan'))
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "—"
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        print(f"  {method:25s} {'paper':8s} {acc_s:>10s} {auroc_s:>10s} {'—':>10s} {'—':>10s}")

    if baselines:
        print(f"  {'. . .':25s}")

    # KIRBy results: find best model per rep, sort by AUROC then accuracy
    rows = []
    for rep_name, rep_data in results.items():
        for model_name, metrics in rep_data['models'].items():
            acc = metrics['accuracy']['mean']
            auroc = metrics.get('auroc', {}).get('mean', float('nan'))
            f1w = metrics['f1_weighted']['mean']
            bacc = metrics['balanced_accuracy']['mean']
            rows.append((rep_name, model_name, acc, auroc, f1w, bacc))

    # Sort by AUROC descending (fallback to accuracy)
    rows.sort(key=lambda r: (r[3] if not np.isnan(r[3]) else 0, r[2]), reverse=True)

    for rep_name, model_name, acc, auroc, f1w, bacc in rows:
        auroc_s = f"{auroc:.4f}" if not np.isnan(auroc) else "—"
        print(f"  {rep_name:25s} {model_name:8s} {acc:10.4f} {auroc_s:>10s} "
              f"{f1w:10.4f} {bacc:10.4f}")

    # Summary: best individual rep. There is no hybrid row here — the hybrid's
    # only number in this experiment is the held-out one printed below.
    best_individual = None
    for rep_name, model_name, acc, auroc, f1w, bacc in rows:
        metric_val = auroc if not np.isnan(auroc) else acc
        if best_individual is None or metric_val > best_individual[2]:
            best_individual = (rep_name, model_name, metric_val)

    print(f"\n  Summary:")
    if best_individual:
        print(f"    Best individual rep: {best_individual[0]} ({best_individual[1]}) "
              f"= {best_individual[2]:.4f}")

    for method, metrics in baselines.items():
        ref_val = metrics.get('auroc', metrics.get('accuracy', float('nan')))
        if not np.isnan(ref_val) and best_individual:
            delta = best_individual[2] - ref_val
            print(f"    vs {method}: {delta:+.4f}")

    # -------------------------------------------------------------------------
    # The per-fold selection. Each fold's 2 single representations and its
    # hybrids were ranked on that fold's training rows and scored on the rows
    # the ranking never saw, so both arms carry the same held-out tax. One row
    # per model; the difference is the top hybrid minus the top single
    # representation on each held-out fold, averaged over the folds where both
    # were scored.
    # -------------------------------------------------------------------------
    summary = ds_results.get('fold_summary') or {}
    if summary:
        print(f"\n  Per-fold selection ({ds_results.get('n_outer_folds')} outer "
              f"folds, selection on each fold's training rows; metric = AUROC):")
        print(f"    {'Model':10s} {'TopHybrid':>10s} {'TopSingle':>12s} "
              f"{'Difference':>12s} {'Folds':>6s}")
        print(f"    {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*6}")
        for model_name in sorted(summary.keys()):
            r = summary[model_name]
            print(f"    {model_name:10s} "
                  f"{_fmt(r['mean_top_hybrid_auc']):>10s} "
                  f"{_fmt(r['mean_top_single_auc']):>12s} "
                  f"{_fmt(r['mean_difference_auc'], signed=True):>12s} "
                  f"{r['n_folds_both_arms']:>6d}")
        for f in (ds_results.get('fold_failures') or []):
            print(f"    not scored: {f.get('dataset')} fold {f.get('fold')} "
                  f"{f.get('model') or '(no model)'} {f.get('what')}: "
                  f"{f.get('error_type')}: {f.get('error')}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Humanness classification experiment (OPIG B3). '
                    'Extends Prihoda et al. 2022 (BioPhi) with KIRBy antibody reps.')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset names '
             '(default: humanness_therapeutic)')
    parser.add_argument(
        '--reps', type=str, default=None,
        help='Comma-separated rep names (default: hand-picked set of 10)')
    # The outer fold count comes from ONE setting: config['n_splits'], which
    # get_config sets from --mode and KIRBY_LOCAL_FOLDS. The default was the
    # literal 5 here, a second place the count was written down (2026-10-05).
    # 1 means one grouped held-out split, not one-fold CV.
    parser.add_argument(
        '--n-folds', type=int, default=None,
        help='Number of outer folds (default: config[\'n_splits\'] from '
             '--mode / KIRBY_LOCAL_FOLDS). 1 = one grouped held-out split.')
    parser.add_argument(
        '--mode', type=str, default='full', choices=['fast', 'full'],
        help="'full' takes the standard outer fold count, 'fast' one split.")
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON path '
             '(default: tests/results/comparisons/opig_humanness.json)')
    parser.add_argument(
        '--pruning-mode', type=str, default='default_pruning',
        help='ACCEPTED AND NOT APPLIED. The cluster line passes it '
             '(submit_everything.sh:85, :420). Every pruning schedule votes '
             'members off the representation-count, importance-method and '
             'internal-evaluator axes, and those axes are the ones the record '
             'keeps for this dataset, so every search here runs unpruned.')

    parser.add_argument(
        '--force-rerun', action='store_true',
        help='Discard what an earlier attempt finished '
             '(<output>.partial.jsonl) and recompute every item.')

    parser = add_distance_metric_arg(parser)

    args = parser.parse_args()


    apply_distance_metric_arg(args)
    # One setting for the outer fold count, used by nested_cv_outer_folds, by
    # every per-fold search's cache namespace and by the table of single
    # representations.
    config = get_config(args)
    n_outer_folds = (args.n_folds if args.n_folds is not None
                     else config['n_splits'])
    datasets = (args.datasets.split(',') if args.datasets
                else DEFAULT_DATASETS)
    rep_override = (args.reps.split(',') if args.reps else None)

    # Validate datasets
    for ds in datasets:
        if ds not in DATASET_REGISTRY:
            print(f"ERROR: Unknown dataset '{ds}'")
            return 1

    print("=" * 70)
    print("HUMANNESS CLASSIFICATION EXPERIMENT (OPIG B3)")
    print(f"  Paper: Prihoda et al. 2022, mAbs 14(1):2020203 (BioPhi)")
    print(f"  Datasets: {datasets}")
    print(f"  Reps: {rep_override if rep_override else 'per-dataset approved pool'}")
    for ds in datasets:
        try:
            print(f"  Models for {ds}: {_decided_models(ds)}")
        except Exception as e:
            print(f"  Models for {ds}: {type(e).__name__}: {e}")
    print(f"  Outer folds: {n_outer_folds}"
          + ("" if args.n_folds is not None else " (from --mode "
                                                 f"{args.mode})"))
    print(f"  Hybrids per model per fold: {N_GENERAL_HYBRIDS} from the "
          f"general search; no distance hybrid (2026-10-05).")
    print(f"  Pruning: off. --pruning-mode={args.pruning_mode} is not applied; "
          f"the decided grid is the cut.")
    print("=" * 70)

    # KIRBY_NO_COLD_SEARCH is left as the launcher sets it: it refuses only
    # whole-dataset searches, and every search here is per fold (it carries a
    # cache namespace, which find_best_hybrid never refuses since fb1df54).

    output_path = (args.output or
                   'tests/results/comparisons/opig_humanness.json')

    # Each dataset is written to <output>.partial.jsonl as soon as it finishes
    # and read back on the next attempt (2026-09-25;
    # experiments/_checkpoint.py). Before, a failure or a kill lost every
    # dataset, because the one JSON below was written after the last. A dataset
    # that raised or returned {'error'} is not stored, is retried next time,
    # and makes the script exit 1 after the JSON is written.
    from experiments._checkpoint import (
        script_checkpoint, is_failed_item, exit_if_failed)
    # The alignment stamp is part of the identity, so a partial file written
    # by the pre-2026-10-05 code -- which read a whole-dataset search, scored
    # a distance hybrid and kept onehot in the table -- is moved aside instead
    # of being read back as finished work.
    _ck = script_checkpoint('opig_humanness', output_path, args,
                            exclude=('datasets', 'output'),
                            alignment='2026-10-05 per-fold, 2 base + 2 '
                                      'general hybrids, approved pool only',
                            n_outer_folds=n_outer_folds)
    _failed = []

    all_results = {}
    skipped = {}
    for ds_name in datasets:
        try:
            from experiments._shared import NoApprovedRepPool
            try:
                rep_names = (rep_override if rep_override
                             else _default_reps(ds_name))
            except NoApprovedRepPool as e:
                # No approved representation list for this dataset. Refuse rather
                # than fall back to a hand-written one; --reps overrides.
                print(f"\n  SKIPPED {ds_name}: {e}")
                skipped[ds_name] = str(e)
                continue
            if _ck.done(dataset=ds_name, rep_names=rep_names):
                ds_results = _ck.get(dataset=ds_name, rep_names=rep_names)
                print(f"  {ds_name}: finished in an earlier attempt, read back")
            else:
                ds_results = evaluate_dataset(ds_name, rep_names,
                                              n_outer_folds, fold_ck=_ck)
                _ck.put(ds_results, dataset=ds_name, rep_names=rep_names)
            if is_failed_item(ds_results):
                _failed.append(ds_name)
            if ds_results is not None:
                all_results[ds_name] = ds_results
                print_analysis(ds_name, ds_results)
        except StaleResultDB:
            # The result database on disk no longer matches the code that wrote
            # it. _shared re-raises it on purpose so it is not swallowed; this
            # except used to catch it, print one line and let the job exit 0.
            raise
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            _failed.append(ds_name)

    # Save results
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if skipped:
        all_results['_skipped'] = skipped

    with open(p, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nResults saved: {p}")
    exit_if_failed(_failed, 'dataset')


if __name__ == '__main__':
    main()
