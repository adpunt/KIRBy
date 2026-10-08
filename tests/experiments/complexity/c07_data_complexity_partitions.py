#!/usr/bin/env python3
"""C07: Data Complexity via Scaffold Partitioning — optimal config by subpopulation.

Paper: Complexity
Conclusion: Optimal config depends on data subpopulation complexity.
Dependencies: none.

Since 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4) every hybrid is chosen
per outer fold, on that fold's training rows only. Since 2026-10-06 (DECIDED,
Adelaide delegated) there is one fold layout: the outer folds are built once,
by whole scaffold group, on the rows c07 searches (at most ~2,000). Fold k of
a half is that half's rows in the layout's fold k. On each fold:
  - the half's own arm: one search on the half's training rows of that fold,
    on the dataset's decided grid, gives each decided model its 2 best base
    representations and 2 best hybrids (fold_standard_selection). Each is
    fitted on the half's training rows and scored on the half's held-out rows.
  - the all-rows arm: one search on the layout's fold-k training rows (both
    halves), shared by every partition and half. Its hybrids per model are
    rebuilt on the half's training rows and scored on the same held-out rows.
    A hybrid whose rebuilt features are identical to one already kept is
    replaced by the next distinct candidate.
Searches per dataset at 5 folds: 4 partitions x 2 halves x 5 folds + 5
all-rows = 45 (80 before 2026-10-06).
There is no distance search: c07 measures nothing with neighbours (handoff
2.3 item 1).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from kirby.hybrid import _graded_scores_for_clf
from kirby.find_best_hybrid import (
    nested_cv_outer_folds, nested_cv_cache_namespace,
    get_model, _fit_model, _impute_nan)
from kirby.scaling import FeatureScaler
from sklearn.decomposition import PCA

from experiments._shared import (
    first_outer_folds,
    load_data, get_config, create_experiment_parser,
    save_experiment_results, get_domain_metric,
    estimate_mutual_information_knn, compute_effective_rank,
    compute_intrinsic_dimensionality,
    fold_standard_selection, build_fold_hybrid, fold_hybrid_scoring_kwargs,
    hybrid_feature_signature, STANDARD_N_HYBRIDS,
)
from experiments.decided_grid import decided_models, decided_search

# Bumped 2026-10-05 when c07 moved to per-fold searches, and 2026-10-06 when
# the halves moved to one shared fold layout with one all-rows search per
# fold, and an identical second hybrid became the next distinct one. A result
# file with an older value is not comparable.
RESULT_VERSION = 'c07_shared_folds_2026-10-06'

# Spare candidates the all-rows arm asks for per model, beyond the 2 it
# scores, so a hybrid identical on the half's rows can be replaced.
ALL_ROWS_SPARE_HYBRIDS = 4

# The outer split's seed, the same 42 every other experiment's per-fold
# searches use.
SPLIT_SEED = 42

# The recipe fields written beside each hybrid's scores.
HYBRID_FIELDS = ('label', 'allocation_method', 'importance_method', 'budget',
                 'internal_evaluator', 'rep_count')

# c07 large-dataset cap (signed off 2026-06-05). Kept as it was.
C07_MAX_SAMPLES = 2000


def _heldout_score(X_tr, X_va, y_tr, y_va, model_name, task_type,
                   metric_name, metric_fn, train_groups=None):
    """Fit one decided model on the training rows, score the held-out rows.

    Features: missing values imputed from the training rows, 0/1 features left
    as they are and the rest standardised on the training rows (FeatureScaler,
    handoff 2.3 item 17), and each held-out feature capped at the range it
    takes on the scaled training rows (owner delegated, 2026-10-05; on adme_hlm
    one held-out molecule sat 83 million training standard deviations out and
    the dnn's rmse ran to thousands).

    AUROC is scored from probabilities, or the decision function where a model
    has no probabilities (ccab773, kept). Every other metric gets the
    predicted labels or values: mcc and accuracy need labels, and before
    2026-10-05 this path handed them probabilities too.

    Returns (score, None) or (None, why).
    """
    X_tr, X_va, _ = _impute_nan(np.asarray(X_tr), np.asarray(X_va))
    scaler = FeatureScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    if X_tr_s.shape[1]:
        X_va_s = np.clip(X_va_s, X_tr_s.min(0), X_tr_s.max(0))
    m = get_model(model_name, task_type, n_samples=len(y_tr))
    if m is None:
        return None, (f'{model_name} is not fitted on {len(y_tr)} training '
                      f'rows (model size limit)')
    _fit_model(m, X_tr_s, y_tr, groups=train_groups)
    y_hard = m.predict(X_va_s)
    if task_type == 'classification' and metric_name == 'auc':
        y_pred = _graded_scores_for_clf(m, X_va_s, y_hard)
    else:
        y_pred = y_hard
    return float(metric_fn(y_va, y_pred)), None


def _spread(values):
    """Mean and spread of per-fold scores. One fold has no spread."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {'mean': float('nan'), 'n_folds': 0, 'std_over_folds': None,
                'spread_note': 'no fold scored'}
    out = {'mean': float(np.mean(vals)), 'n_folds': len(vals),
           'min': float(np.min(vals)), 'max': float(np.max(vals))}
    if len(vals) == 1:
        out['std_over_folds'] = None
        out['spread_note'] = 'one fold, no spread'
    else:
        out['std_over_folds'] = float(np.std(vals))
    return out


def _score_selection(sel, *, tr_reps, va_reps, y_tr, y_va, g_tr,
                     aug_tr, aug_va, task_type, metric_name, metric_fn,
                     lower_is_better, with_base=True,
                     n_keep=STANDARD_N_HYBRIDS):
    """Held-out scores of what fold_standard_selection chose, per model.

    tr_reps/va_reps: the representations on the rows each recipe is rebuilt
    and fitted on, and on the held-out rows. For the half's own arm these are
    the rows its search ran on; for the all-rows arm they are the half's
    training rows, not the rows its search ran on (the old second pass did the
    same: only the feature selection is redone on the half).

    Each hybrid is rebuilt from its recipe with the per-representation scores
    measured by the per-fold search that chose it (blueprint.rep_scores of a
    recipe from that search; dirty-run bug 1), then scored on the held-out
    rows.

    n_keep: how many hybrids are scored per model. sel may carry more
    candidates than that, in rank order (the all-rows arm asks for spares).
    A candidate whose selected features, rebuilt on tr_reps, are identical
    to a hybrid already kept is listed under
    'hybrids_identical_on_these_rows' and the next candidate is taken
    (DECIDED 2026-10-06, Adelaide delegated). Candidates are taken until
    n_keep are kept or none are left; a shortfall is recorded with its
    reason.
    """
    out = {}
    for m, s in sel.items():
        if s is None:
            out[m] = {'error': ("the search on this fold's training rows "
                                "failed for this model")}
            continue
        base_rows = []
        for i, e in enumerate(s['base_detail']['selected'], 1):
            rep = e['rep']
            row = {'slot': f'base{i}', 'rep': rep,
                   'train_rows_score': e.get('score')}
            if with_base:
                try:
                    sc, why = _heldout_score(
                        tr_reps[rep], va_reps[rep], y_tr, y_va, m, task_type,
                        metric_name, metric_fn, train_groups=g_tr)
                    row['heldout_score'] = sc
                    if why:
                        row['error'] = why
                except Exception as ex:
                    row['heldout_score'] = None
                    row['error'] = f'{type(ex).__name__}: {ex}'
            base_rows.append(row)

        hyb_rows = []
        identical_rows = []
        # Selected features of each hybrid as rebuilt here. The search's own
        # identical-features check built its candidates on the rows that
        # search ran on; the all-rows arm rebuilds them on the half's
        # training rows, where two recipes can pick the same features.
        # DECIDED 2026-10-06 (Adelaide delegated): a candidate identical to
        # a kept hybrid is not fitted or scored, and the next candidate takes
        # the slot. Before 2026-10-06 the slot was left empty.
        seen_sigs = {}
        for rank, e in enumerate(s['hybrids'], 1):
            if len(hyb_rows) >= n_keep:
                break
            bp = e['blueprint']
            cond = e.get('aug_condition') or 'none'
            row = {'slot': f'hybrid{len(hyb_rows) + 1}',
                   'candidate_rank': rank,
                   **{f: e.get(f) for f in HYBRID_FIELDS},
                   'aug_condition': cond,
                   'train_rows_score': e.get('score_mean')}
            use_augs = cond != 'none' and aug_tr is not None
            try:
                X_tr_h, X_va_h, fi = build_fold_hybrid(
                    tr_reps, va_reps, y_tr, bp,
                    augmentations_train=aug_tr if use_augs else None,
                    augmentations_val=aug_va if use_augs else None,
                    groups=g_tr,
                    precomputed_alloc_scores=getattr(bp, 'rep_scores', None),
                    **fold_hybrid_scoring_kwargs(
                        bp, model=get_model(m, task_type,
                                            n_samples=len(y_tr)),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better))
                row['n_features'] = int(np.asarray(X_tr_h).shape[1])
                row['features_by_rep'] = {
                    r: int(len(info['selected_indices']))
                    for r, info in fi.items()
                    if isinstance(info, dict) and 'selected_indices' in info}
                sig = hybrid_feature_signature(fi)
                if sig and sig in seen_sigs:
                    row.pop('slot')
                    row['identical_features_to'] = seen_sigs[sig]
                    identical_rows.append(row)
                    continue
                if sig:
                    seen_sigs[sig] = row['slot']
                sc, why = _heldout_score(X_tr_h, X_va_h, y_tr, y_va, m,
                                         task_type, metric_name, metric_fn,
                                         train_groups=g_tr)
                row['heldout_score'] = sc
                if why:
                    row['error'] = why
            except Exception as ex:
                row['heldout_score'] = None
                row['error'] = f'{type(ex).__name__}: {ex}'
            hyb_rows.append(row)

        out[m] = {
            'base': base_rows, 'hybrids': hyb_rows,
            'hybrids_identical_on_these_rows': identical_rows,
            'hybrid_shortfall': (
                None if len(hyb_rows) >= n_keep else
                f'{len(hyb_rows)} of {n_keep} hybrids: the search gave '
                f'{len(s["hybrids"])} candidates and {len(identical_rows)} '
                f'had features identical to a kept one on these rows'),
            'base_shortfall': s['base_detail'].get('shortfall_reason'),
            'hybrids_skipped_identical_features':
                [{k: v for k, v in d.items() if k != 'blueprint'}
                 if isinstance(d, dict) else d
                 for d in (s.get('hybrids_skipped_identical_features') or [])],
            'selection_scope': 'fold_train_rows',
        }
    return out


def _fold_winner(sel, lower_is_better):
    """The configuration c07 compares between halves, for one fold.

    Before 2026-10-05 it was the best hybrid over five models in one search
    over every row of the half. Now it is the best top hybrid over the
    dataset's decided models, ranked on the fold's training rows (the score
    the per-fold search gave it). None when no model has a hybrid.
    """
    best, best_model = None, None
    for m, s in sel.items():
        if not s or not s['hybrids']:
            continue
        e = s['hybrids'][0]
        sc = e.get('score_mean')
        if sc is None or not np.isfinite(sc):
            continue
        if (best is None or (lower_is_better and sc < best['score_mean'])
                or (not lower_is_better and sc > best['score_mean'])):
            best, best_model = e, m
    if best is None:
        return None
    return {
        'model': best_model,
        'budget': best.get('budget'),
        'allocation': best.get('allocation_method'),
        'importance': best.get('importance_method'),
        'internal_evaluator': best.get('internal_evaluator'),
        'rep_count': best.get('rep_count'),
        'aug_condition': best.get('aug_condition'),
        'train_rows_score': best.get('score_mean'),
        'n_features': best.get('n_features'),
    }


def _summarise_half(fold_records, arm):
    """{model: {slot: spread of held-out scores over folds}} for one arm."""
    vals = {}
    for fr in fold_records:
        for m, mr in (fr.get(arm) or {}).items():
            for row in (mr.get('base') or []) + (mr.get('hybrids') or []):
                if 'heldout_score' not in row:
                    continue
                vals.setdefault(m, {}).setdefault(row['slot'], []).append(
                    row['heldout_score'])
    return {m: {sl: _spread(v) for sl, v in d.items()}
            for m, d in vals.items()}


def run(data, config, output_dir, dataset_name, partitions=None):
    """Exp 7: Optimal config depends on data subpopulation complexity.

    Per partition, per half, per outer fold: the standard selection (2 base
    representations and 2 hybrids per decided model) from a search on that
    fold's training rows, scored on its held-out rows.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: DATA COMPLEXITY VIA SCAFFOLD PARTITIONING")
    print("=" * 70)

    reps = data['reps']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)

    # 2026-10-05, owner delegated, 2026-10-05: a dataset with fewer than 10
    # scaffold/sequence groups in total (the groups array every partition below
    # splits by) is skipped. abagym_Ang2/EGFR/VEGF and the four ginkgo datasets
    # have 5. Halving 5 groups leaves too few groups per half to
    # cross-validate, and splitting inside a group would break the
    # no-random-split rule. The {'skipped': True} marker is what
    # classify_result_payload (_shared.py) reads as an intended skip.
    _n_groups_total = int(len(np.unique(np.asarray(groups))))
    if _n_groups_total < 10:
        print(f"  [skip] {_n_groups_total} groups < 10: partitions not run")
        return {
            'skipped': True,
            'reason': (f'{_n_groups_total} scaffold/sequence groups; whole-group '
                       'draws/halves cannot be cross-validated and splitting '
                       'inside groups would break the no-random-split rule'),
            'n_groups': _n_groups_total,
            'n_samples': int(len(labels)),
            'result_version': RESULT_VERSION,
        }

    # Models (2026-10-05, handoff 2.4 step 2): the dataset's two decided
    # models only. Raises for a dataset DECISIONS.md does not cover.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']

    # Representations are the run's pool (run_all loads
    # DATASET_REPS[ds]['general']). Anything else is named, not removed: the
    # pool handed to the hybrid builder is not filtered here (handoff 2.4
    # step 9).
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
        _general = set((DATASET_REPS.get(dataset_name) or {}).get('general')
                       or [])
    except Exception:
        _general = set()
    _outside = sorted(r for r in reps if _general and r not in _general)
    if _outside:
        print(f"  WARNING: representations outside DATASET_REPS['general'] "
              f"in this run's pool: {_outside}")

    # Augmentation (2026-10-05, handoff 2.3 item 4): c07 compares nothing with
    # and without added features. The per-fold search adds a condition only
    # where Step 8 keeps one for the general search (decided_aug_conditions),
    # which is no dataset today, so every hybrid here is 'none'. Before
    # 2026-10-05 both of each half's searches ran 'capped' on every dataset
    # with augmentation representations.
    _aug_kept = [c for c in decided_search(dataset_name, 'general')
                 ['aug_conditions'] if c != 'none']
    if data.get('augmentations') and not _aug_kept:
        print("  Augmentation: not searched. DECISIONS.md Step 8 keeps no "
              "augmentation on this dataset's general search.")

    # Outer folds (2026-10-05, handoff 2.4 step 14): the count is
    # config['n_splits'] and nothing else. A count of 1 is one grouped 80/20
    # held-out split. Built once below, after the row cap.
    n_folds_requested = int(config['n_splits'])

    # For molecular datasets, use smiles; for antibody, this partition may not apply
    smiles = data.get('smiles')

    # 'feature_variance' is the label-free counterpart of 'target_variance':
    # it splits scaffold groups by how spread out their rows are in
    # representation space, so nothing about the property being predicted
    # decides which half a row lands in. 'target_variance' is kept, but its
    # easy/hard membership comes from the same labels the two halves are
    # scored on, so its gap is not a generalisation measure — every
    # target_variance block carries partition_uses_labels=True saying so.
    all_partitions = partitions or ['scaffold_frequency', 'molecular_complexity',
                                    'target_variance', 'feature_variance']
    results = {'partitions': {},
               # Why a partition is absent from 'partitions', added 2026-10-05.
               # {partition name: sentence}. 'partitions' came back {} on some
               # datasets with nothing saying which split failed which minimum.
               'partition_skips': {},
               'result_version': RESULT_VERSION,
               'selection_scope': 'fold_train_rows',
               'selection_rule': ('per decided model, per half, per outer '
                                  'fold: 2 best base representations and 2 '
                                  'best hybrids, both ranked by a search on '
                                  "that fold's training rows"),
               'models': models,
               'metric_name': metric_name,
               'lower_is_better': bool(lower_is_better),
               'n_folds_requested': n_folds_requested,
               'split_seed': SPLIT_SEED}

    # c07 large-dataset cap (signed off 2026-06-05): every search below is run
    # on a fixed subset of at most ~2,000 rows. Selected by WHOLE scaffold
    # groups (not random rows) so the scaffold-based partitions stay intact,
    # no group is fragmented, and downstream CV stays group-aware (NOT a random
    # split). Deterministic (seed 42). Since 2026-10-05 the all-rows arm is
    # searched on this subset too; it used to read the whole-dataset search.
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    _c07_augs = data.get('augmentations')
    cap_tag = 'all'
    if len(labels) > C07_MAX_SAMPLES:
        _rng = np.random.RandomState(42)
        _uniq = np.unique(groups)
        _rng.shuffle(_uniq)
        _chosen, _count = set(), 0
        for _g in _uniq:
            _chosen.add(_g)
            _count += int((groups == _g).sum())
            if _count >= C07_MAX_SAMPLES:
                break
        _idx = np.where(np.array([g in _chosen for g in groups]))[0]
        reps = {k: v[_idx] for k, v in reps.items()}
        labels = labels[_idx]
        groups = groups[_idx]
        if smiles is not None:
            smiles = ([smiles[i] for i in _idx] if isinstance(smiles, list)
                      else np.asarray(smiles)[_idx])
        if _c07_augs:
            _c07_augs = {k: v[_idx] for k, v in _c07_augs.items()}
        cap_tag = f'cap{C07_MAX_SAMPLES}s42'
        print(f"  c07: large dataset capped to {len(labels)} samples "
              f"({len(_chosen)} whole scaffold groups, seed 42) for partition searches")
    results['n_rows_searched'] = int(len(labels))
    results['row_cap'] = cap_tag

    # DECIDED 2026-10-06 (Adelaide delegated): one shared fold layout. The
    # outer folds are built once on the rows searched, by whole scaffold
    # group, and every partition and half uses them. Easy fold k and hard
    # fold k then hold out the same groups, and the all-rows arm needs one
    # search per fold instead of one per partition, half and fold.
    try:
        folds = nested_cv_outer_folds(reps, labels, groups, n_folds_requested,
                                      SPLIT_SEED, task_type)
    except ValueError as e:
        print(f"  [skip] outer folds could not be built: {e}")
        results['skipped'] = True
        results['reason'] = f'outer folds could not be built: {e}'
        return results
    n_outer = len(folds)
    results['n_outer_folds'] = n_outer
    results['fold_layout'] = 'shared'
    # n_outer stays the layout's count (search names, namespaces, the fold id
    # of every row below); the folds run may be the first few of it
    # (OUTER_FOLDS_RUN, 2026-10-07).
    n_run = len(first_outer_folds(folds, 'c07', dataset_name))
    results['n_outer_folds_run'] = n_run
    print(f"  {n_outer} outer folds (asked for {n_folds_requested}), shared "
          f"by every partition and half")

    # The all-rows search of each fold, run the first time a half needs it.
    # {fold k: (selection or None, error or None)}.
    all_rows_by_fold = {}

    def _all_rows_selection(k):
        if k in all_rows_by_fold:
            return all_rows_by_fold[k]
        tr_k = folds[k][0]
        ns_all = (f'{nested_cv_cache_namespace(k, n_outer, SPLIT_SEED)}'
                  f'__c07_{cap_tag}_allrows')
        sel_k, err = None, None
        try:
            sel_k = fold_standard_selection(
                reps, labels, groups, train_idx=tr_k,
                fold_i=k, n_outer_folds=n_outer,
                dataset_name=dataset_name, models=models,
                augmentations=_c07_augs, task_type=task_type,
                metric=metric_name, seed=SPLIT_SEED,
                n_hybrids=STANDARD_N_HYBRIDS + ALL_ROWS_SPARE_HYBRIDS,
                cache_namespace=ns_all)
        except Exception as ex:
            err = f'{type(ex).__name__}: {ex}'
            print(f"    fold {k}: all-rows search failed - {err}")
        all_rows_by_fold[k] = (sel_k, err, ns_all, int(len(tr_k)))
        return all_rows_by_fold[k]

    for partition_name in all_partitions:
        print(f"\n  Partition: {partition_name}")

        # Compute partition masks
        if partition_name == 'scaffold_frequency':
            # Common (top 20% scaffolds) vs rare (bottom 80%)
            unique_scaffolds, scaffold_counts = np.unique(groups, return_counts=True)
            count_threshold = np.percentile(scaffold_counts, 80)  # top 20% most frequent
            frequent_scaffolds = set(unique_scaffolds[scaffold_counts >= count_threshold])
            mask_easy = np.array([g in frequent_scaffolds for g in groups])
            mask_hard = ~mask_easy

        elif partition_name == 'molecular_complexity':
            # Heavy atom count: <= median vs > median
            if smiles is None:
                print("    No SMILES available (antibody dataset?), skipping molecular_complexity")
                results['partition_skips'][partition_name] = (
                    'no SMILES on this dataset, so no heavy-atom count')
                continue
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                # Parse once per SMILES, track valid_mask so failed parses
                # don't silently land in "easy" via hac=0 below median.
                hac = np.full(len(smiles), -1, dtype=int)
                valid_mask = np.zeros(len(smiles), dtype=bool)
                for i, s in enumerate(smiles):
                    mol = Chem.MolFromSmiles(s)
                    if mol is not None:
                        hac[i] = Descriptors.HeavyAtomCount(mol)
                        valid_mask[i] = True
                if not valid_mask.any():
                    print("    No valid SMILES, skipping molecular_complexity")
                    results['partition_skips'][partition_name] = (
                        'no SMILES parsed with RDKit')
                    continue
                median_hac = np.median(hac[valid_mask])
                mask_easy = valid_mask & (hac <= median_hac)
                mask_hard = valid_mask & (hac > median_hac)
            except ImportError:
                print("    RDKit not available, skipping molecular_complexity")
                results['partition_skips'][partition_name] = (
                    'RDKit not importable')
                continue

        elif partition_name == 'target_variance':
            # Scaffolds with low target variance vs high. The label decides
            # which half a row goes to, and the label spread is also part of
            # what the two halves are then compared on, so the easy/hard gap
            # here measures the split as much as it measures generalisation.
            # Flagged in the output; see 'feature_variance' for the label-free
            # version of the same question.
            unique_scaffolds = np.unique(groups)
            scaffold_var = {}
            for s in unique_scaffolds:
                s_labels = labels[groups == s]
                if len(s_labels) >= 2:
                    scaffold_var[s] = np.var(s_labels)
            if not scaffold_var:
                print("    Not enough scaffolds with 2+ members, skipping")
                results['partition_skips'][partition_name] = (
                    'no scaffold group has 2 or more molecules, so no group '
                    'has a label variance')
                continue
            median_var = np.median(list(scaffold_var.values()))
            mask_easy = np.array([scaffold_var.get(g, 0) <= median_var for g in groups])
            mask_hard = ~mask_easy

        elif partition_name == 'feature_variance':
            # Scaffolds whose rows sit close together in representation space
            # vs spread out. Uses no label at any step, so membership carries
            # nothing about the property being predicted. Every representation
            # is scaled first so a wide one does not set the spread on its
            # own, then a group's spread is the mean per-feature variance of
            # its rows. Since 2026-10-05 the scaling is FeatureScaler (0/1
            # features left as they are, the rest standardised; handoff 2.3
            # item 17) and the matrix stays float32 (item 18). It was a z-score
            # of every feature in float64.
            unique_scaffolds = np.unique(groups)
            X_all = np.hstack([reps[k] for k in sorted(reps.keys())])
            X_all = FeatureScaler().fit_transform(X_all)
            scaffold_spread = {}
            for s in unique_scaffolds:
                rows = X_all[groups == s]
                if len(rows) >= 2:
                    scaffold_spread[s] = float(np.mean(np.var(rows, axis=0)))
            if not scaffold_spread:
                print("    Not enough scaffolds with 2+ members, skipping")
                results['partition_skips'][partition_name] = (
                    'no scaffold group has 2 or more molecules, so no group '
                    'has a feature spread')
                continue
            median_spread = np.median(list(scaffold_spread.values()))
            mask_easy = np.array(
                [scaffold_spread.get(g, 0.0) <= median_spread for g in groups])
            mask_hard = ~mask_easy
        else:
            print(f"    Unknown partition: {partition_name}, skipping")
            results['partition_skips'][partition_name] = 'unknown partition name'
            continue

        n_easy = mask_easy.sum()
        n_hard = mask_hard.sum()
        print(f"    easy={n_easy}, hard={n_hard}")

        if n_easy < 50 or n_hard < 50:
            print("    Subpopulation too small, skipping")
            results['partition_skips'][partition_name] = (
                f'easy half {int(n_easy)} rows, hard half {int(n_hard)} rows; '
                f'each half needs at least 50')
            continue

        part_result = {'easy': {}, 'hard': {}, 'n_easy': int(n_easy), 'n_hard': int(n_hard),
                       # True when the label decided which half each row went
                       # to. The easy/hard gap is then partly the split itself.
                       'partition_uses_labels': partition_name == 'target_variance'}

        full_augs = _c07_augs  # c07 cap: subsampled augmentations (see C07_MAX_SAMPLES)
        for subpop_name, mask in [('easy', mask_easy), ('hard', mask_hard)]:
            half_idx = np.where(mask)[0]
            sub_reps = {k: v[mask] for k, v in reps.items()}
            sub_labels = labels[mask]
            sub_groups = groups[mask]
            sub_augs = {k: v[mask] for k, v in full_augs.items()} if full_augs else None
            n_unique_groups = len(np.unique(sub_groups))

            if n_unique_groups < 2:
                print(f"    {subpop_name}: not enough groups for CV, skipping")
                # Recorded 2026-10-05; this half was left out with no entry.
                part_result[subpop_name] = {
                    'skipped': (f'{int(n_unique_groups)} scaffold group in '
                                f'{int(mask.sum())} rows; group-aware folds '
                                f'need at least 2'),
                }
                continue

            # Complexity metrics: descriptive, on every row of the half, no
            # model chosen from them. FeatureScaler since 2026-10-05 (0/1
            # features left as they are; was StandardScaler).
            X_concat = np.hstack([sub_reps[k] for k in sorted(sub_reps.keys())])
            metrics = {}
            try:
                n_comp = min(50, X_concat.shape[1], X_concat.shape[0] - 1)
                X_reduced = PCA(n_components=n_comp).fit_transform(
                    FeatureScaler().fit_transform(X_concat))
                metrics['mi'] = float(estimate_mutual_information_knn(X_reduced, sub_labels, k=5))
            except Exception:
                metrics['mi'] = None
            metrics['effective_rank'] = compute_effective_rank(X_concat)
            dim_info = compute_intrinsic_dimensionality(X_concat)
            metrics['pca_95'] = dim_info.get('pca_95')
            metrics['mle_dim'] = dim_info.get('mle')
            metrics['n_samples'] = int(mask.sum())

            print(f"    {subpop_name}: eff_rank={metrics['effective_rank']:.1f}, "
                  f"MI={metrics.get('mi', '?')}, PCA95={metrics.get('pca_95', '?')}")

            # This half's folds: its rows in each fold of the shared layout.
            # -1: a row in no held-out set (a one-split layout); it trains.
            fold_of_row = np.full(len(labels), -1, dtype=int)
            for _k, (_tr, _va) in enumerate(folds):
                fold_of_row[_va] = _k
            half_fold = fold_of_row[half_idx]

            fold_records = []
            for k in range(n_run):
                tr = np.where(half_fold != k)[0]
                va = np.where(half_fold == k)[0]
                ns_root = nested_cv_cache_namespace(k, n_outer, SPLIT_SEED)
                # The rows of these searches are a subset of the dataset's, and
                # rows are not part of the stored-result key, so each search is
                # filed under its own name: partition, half and the row cap.
                # Without it, a half's fold-k search would read the dataset's
                # own fold-k search.
                ns_half = f'{ns_root}__c07_{cap_tag}_{partition_name}_{subpop_name}'
                fr = {'fold': k, 'n_train': int(len(tr)), 'n_heldout': int(len(va)),
                      'cache_namespace': ns_half,
                      'selection_scope': 'fold_train_rows'}
                if len(va) == 0 or len(tr) == 0:
                    fr['skipped'] = (f'this half has {len(tr)} training and '
                                     f'{len(va)} held-out rows in fold {k} of '
                                     f'the shared layout')
                    print(f"    {subpop_name} fold {k}: {fr['skipped']}")
                    fold_records.append(fr)
                    continue

                tr_reps = {r: np.asarray(X)[tr] for r, X in sub_reps.items()}
                va_reps = {r: np.asarray(X)[va] for r, X in sub_reps.items()}
                aug_tr = ({a: np.asarray(X)[tr] for a, X in sub_augs.items()}
                          if sub_augs else None)
                aug_va = ({a: np.asarray(X)[va] for a, X in sub_augs.items()}
                          if sub_augs else None)
                common = dict(tr_reps=tr_reps, va_reps=va_reps,
                              y_tr=sub_labels[tr], y_va=sub_labels[va],
                              g_tr=sub_groups[tr], aug_tr=aug_tr, aug_va=aug_va,
                              task_type=task_type, metric_name=metric_name,
                              metric_fn=metric_fn,
                              lower_is_better=lower_is_better)

                # The half's own arm: a search on the half's training rows.
                sel = None
                try:
                    sel = fold_standard_selection(
                        sub_reps, sub_labels, sub_groups, train_idx=tr,
                        fold_i=k, n_outer_folds=n_outer,
                        dataset_name=dataset_name, models=models,
                        augmentations=sub_augs, task_type=task_type,
                        metric=metric_name, seed=SPLIT_SEED,
                        cache_namespace=ns_half)
                except Exception as e:
                    fr['search_error'] = f'{type(e).__name__}: {e}'
                    print(f"    {subpop_name} fold {k}: search failed - {e}")
                if sel is not None:
                    fr['half'] = _score_selection(sel, **common)
                    fr['best'] = _fold_winner(sel, lower_is_better)

                # The all-rows arm: the shared search on the layout's fold-k
                # training rows (both halves; every held-out group of fold k
                # is left out whole, so the half's held-out rows are never
                # searched). Its hybrids are rebuilt on the half's training
                # rows and scored on the same held-out rows as the half's own
                # arm. Base representations are named with their
                # training-row score and not fitted, as the old all-rows pass
                # had none.
                sel_all, err_all, ns_all, n_tr_all = _all_rows_selection(k)
                fr['cache_namespace_all_rows'] = ns_all
                if err_all:
                    fr['all_rows_search_error'] = err_all
                if sel_all is not None:
                    fr['n_train_all_rows'] = n_tr_all
                    fr['all_rows'] = _score_selection(sel_all, with_base=False,
                                                      **common)
                    fr['best_all_rows'] = _fold_winner(sel_all, lower_is_better)

                for arm in ('half', 'all_rows'):
                    for m, mr in (fr.get(arm) or {}).items():
                        for row in mr.get('hybrids') or []:
                            sc = row.get('heldout_score')
                            _trs = row.get('train_rows_score')
                            print(f"    {subpop_name} fold {k} {arm:8s} {m:8s} "
                                  f"{row['slot']}: "
                                  + (f"{metric_name}={sc:.4f} held out "
                                     f"(train rows "
                                     f"{_trs if _trs is None else round(_trs, 4)}, "
                                     f"budget {row.get('budget')}, "
                                     f"{row.get('n_features')} features)"
                                     if sc is not None else
                                     f"not scored ({row.get('error')})"))
                fold_records.append(fr)

            part_result[subpop_name] = {
                'metrics': metrics,
                'n_outer_folds': n_outer,
                'fold_layout': 'shared',
                'folds': fold_records,
                # Held-out score per model and slot, over folds.
                'heldout_by_model': _summarise_half(fold_records, 'half'),
                'heldout_by_model_all_rows': _summarise_half(fold_records,
                                                             'all_rows'),
            }

        # Compare easy vs hard. Since 2026-10-06 both halves use the shared
        # fold layout, so fold k of each half holds out rows of the same
        # scaffold groups' fold.
        easy_f = part_result['easy'].get('folds') or []
        hard_f = part_result['hard'].get('folds') or []
        by_fold = []
        for fe, fh in zip(easy_f, hard_f):
            eb, hb = fe.get('best'), fh.get('best')
            if not (eb and hb):
                by_fold.append({'fold': fe['fold'], 'config_differs': None,
                                'why': 'a half has no best hybrid on this fold'})
                continue
            differs = (eb['model'] != hb['model']
                       or eb['budget'] != hb['budget']
                       or eb.get('importance') != hb.get('importance'))
            by_fold.append({
                'fold': fe['fold'], 'config_differs': bool(differs),
                'delta_budget': (hb['budget'] - eb['budget']
                                 if eb['budget'] is not None
                                 and hb['budget'] is not None else None),
                'easy': f"{eb['model']}/b={eb['budget']}/{eb.get('importance')}",
                'hard': f"{hb['model']}/b={hb['budget']}/{hb.get('importance')}"})
            print(f"    >> fold {fe['fold']}: configs differ: {differs} "
                  f"(easy={by_fold[-1]['easy']}, hard={by_fold[-1]['hard']})")
        part_result['config_differs_by_fold'] = by_fold
        _cmp = [b['config_differs'] for b in by_fold
                if b['config_differs'] is not None]
        part_result['config_differs_n_folds'] = int(sum(_cmp))
        part_result['config_compared_n_folds'] = len(_cmp)

        results['partitions'][partition_name] = part_result

    # Every partition skipped: nothing was scored. The run counts, the row
    # count and the representation widths in this dict are numbers, so the
    # file was stamped result_status ok with zero partitions (abcov_kd,
    # cdr_flexibility, 2026-10-08). A skip with the reasons instead.
    if not results['partitions']:
        results['skipped'] = True
        results['reason'] = ('no partition could be scored: ' + '; '.join(
            f'{k}: {v}' for k, v in results['partition_skips'].items()))
    return results


if __name__ == '__main__':
    parser = create_experiment_parser('C07: Data Complexity via Scaffold Partitioning',
                                     has_delta=True, has_augment=True)
    parser.add_argument('--partitions', type=str, nargs='+', default=None,
                        choices=['scaffold_frequency', 'molecular_complexity',
                                 'target_variance', 'feature_variance'],
                        help='Which partitions to test (default: all)')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps,
                     delta=getattr(args, 'delta', False),
                     augment=getattr(args, 'augment', False))
    results = run(data, config, args.output, args.dataset, partitions=args.partitions)
    save_experiment_results(results, args.output, 'c07_data_complexity_partitions', args.dataset)
