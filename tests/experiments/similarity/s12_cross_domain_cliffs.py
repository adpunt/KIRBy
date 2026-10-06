#!/usr/bin/env python3
"""S12: Cross-Domain Activity Cliff Analysis — generic cliff detection for all domains.

Paper: Similarity
Companion to S04 (which uses ECFP4 Tanimoto, molecular-only). S12 measures
similarity on the scaled concatenation of the dataset's representations where
there are no SMILES, making it applicable to antibody, TCGA, and molecular
datasets alike. Question: per outer fold, does each decided model predict the
held-out cliff molecules better from a hybrid than from a single
representation, and how does that compare with the non-cliff molecules?

Cliff definition (2026-09-20): the same one s04 uses, in
similarity_helpers.calibrated_cliff_rule. On a molecular dataset with SMILES the
structural side is ECFP4 Tanimoto at or above 0.55, so this file and s04 flag
the same molecules; otherwise it is 1/(1+d) on the scaled concatenation of the
dataset's representations, at a fixed percentile of pairs. The activity gap is
solved per dataset so a stated share of molecules are cliffs. What that fixed,
and the counts before and after, are written above calibrated_cliff_rule.

DECIDED 2026-10-06 (Adelaide delegated): which molecules are cliffs is decided
per outer fold from that fold's training molecules only; no held-out label is
read (fold_cliff_partition below, which s04 also calls). On the training
molecules the shared rule runs as before: the structural cutoff (the fixed
percentile is taken over training pairs on the concatenation branch) and the
activity gap solved so the target share of TRAINING molecules are cliffs. A
held-out molecule is then a cliff molecule when at least two training
molecules structurally similar to it (at or above the fold's cutoff) differ in
label by more than the fold's activity gap (on classification: are of
different classes). Before this date the rule ran once over every row, so the
held-out labels set the activity gap and decided which held-out molecules were
cliffs.

2026-10-05 (handoff Part 2, chat 4): the distance d on the concatenation branch
is taken under each distance of the dataset's kNN group (knn_axes: full_knn
manhattan, cosine, minkowski_p0.5; fast_knn manhattan, cosine), one cliff
partition per distance, every score reported under each. Before this date it
was the one process-wide distance (KIRBY_DISTANCE_METRIC, manhattan) on
standardised features. The ECFP4 branch is one partition (Tanimoto is fixed).

2026-10-05 (handoff Part 2, chat 4): every selection is per outer fold, from
searches on that fold's TRAINING rows (similarity_helpers.
fold_similarity_selection, neighbour spaces on). Per outer fold:

  Decided-model arms: each of the dataset's two models (DECISIONS.md Step 1)
  fitted on its 2 base representations and its 2 general hybrids, all four
  ranked on the fold's training rows (8 fits for two models).
    <model>|base1, base2        the model's 2 best single representations
    <model>|hybrid1, hybrid2    the model's 2 best general hybrids (a second
                                hybrid with the same selected features as the
                                first is skipped by the selection)
  kNN arms (each held-out molecule's label predicted from its nearest training
  molecules), scored at every decided kNN setting (distance, 5/10/20
  neighbours, rank and inverse_square counting, scaffold group not excluded):
    knn_base1, knn_base2        knn's 2 best single representations in the
                                fold's distance search
    distance_hybrid             knn's top hybrid of the fold's distance search
    similarity_hybrid           the top hybrid of the fold's similarity-
                                objective search
  Headline per model and fold: hybrid1 against base1, the two the model's
  training-row search ranked first. Headline per kNN arm and fold: the setting
  with the best leave-one-out score on the fold's training rows.

Before this date: five models (DOWNSTREAM_MODELS) on every representation;
hybrids read with get_rule_hybrids from searches over every row at a fixed
100-feature budget (top 2 general + top 1 distance), each fold's recipe ranked
on the other folds' stored scores; the best single representation re-picked
per fold on the other folds' cliff scores; no kNN arms; StandardScaler.

Usage:
    python s12_cross_domain_cliffs.py --dataset flab_thermo_tresanco --mode intermediary
    python s12_cross_domain_cliffs.py --dataset tcga_luad_vs_lusc --mode intermediary
    python s12_cross_domain_cliffs.py --dataset logd --mode intermediary
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.decided_grid import decided_models, UndecidedSearch
from experiments.similarity.similarity_helpers import (
    calibrated_cliff_rule, cliff_structural_similarity, CLIFF_TARGET_PCT,
    CLIFF_SIMILARITY_PERCENTILE, DATASET_REPS,
    sim_outer_folds, fold_spread_note, fold_similarity_selection,
    build_fold_entry, entry_record, knn_axes, scale_fold, knn_lists,
    knn_from_lists, score_prediction)
# Scores for the metric (2026-10-05), as s10 does. On a binary dataset scored
# by auc this returns the predicted probability of class 1, which is how the
# searches score auc (find_best_hybrid.py); every other metric (mcc, f1,
# rmse, spearman) gets model.predict. Before this date the models here,
# on each single representation and on each hybrid, scored auc from 0/1
# predictions, which gives a different number from the one the searches
# ranked recipes by. The kNN arms take auc from each class's share of the
# neighbours' weight (similarity_helpers.score_prediction).
from kirby.find_best_hybrid import _predict_for_metric

# Checkpoint identity. A sidecar written before 2026-10-05 holds per-
# representation units and hybrids read from whole-dataset searches; this
# value keeps it from being read back.
_CHECKPOINT_VERSION = 'per_fold_cliff_partition_2026_10_06'

# 2026-10-05, owner delegated, 2026-10-05 (delegated default): a fold's cliff
# score, and its non-cliff score, is computed only when the fold has at least
# this many held-out rows of that kind. Fewer gives NaN; the row count is still
# recorded per fold. Until this date the minimum was 2, and a Spearman over 2
# rows is always +1 or -1.
_MIN_ROWS_PER_FOLD_SCORE = 10

# Used only by identify_cliff_pairs below, which run() no longer calls.
from kirby.utils.noise_mitigation_methods import compute_distance_matrix
# The process-wide distance coefficient compute_distance_matrix falls back to
# when no metric is named. Read, never asserted: it has been manhattan since
# 2026-09-04 and the result file used to record 'euclidean' regardless.
from kirby.similarity.distance_config import get_distance_metric


# =========================================================================
# CLIFF DETECTION (domain-generic, mirrors N01's identify_cliff_pairs)
# =========================================================================

def identify_cliff_pairs(X_concat, y, sim_threshold=0.55, activity_k=2.0,
                         max_compounds=5000, is_clf=False,
                         similar_percentile=None):
    """Identify activity cliff pairs on the stacked representation.

    ⛔ 2026-09-20: run() no longer calls this. Its distance is the one
    process-wide KIRBY_DISTANCE_METRIC on StandardScaler features, not the
    decided per-group distances run() uses (2026-10-05). It is the rule that disagreed
    with s04 — on herg_ki it called 295 of 1412 molecules cliffs where s04
    called 451 — and its percentile auto-select never reached its own stated 10
    per cent target on caco2, herg_ki or adme_mdr1, because even its smallest
    candidate overshot. Kept because smoke_tests_broken_fixes.py exercises it
    through identify_cliff_pairs_euclidean. New callers want
    similarity_helpers.calibrated_cliff_rule.

    sim = 1/(1+d) on standardised features, where d comes from
    compute_distance_matrix with no metric named — so it is whatever
    kirby.similarity.distance_config holds, manhattan since 2026-09-04, not
    Euclidean. The coefficient actually used is written into info.

    Cliff = high structural similarity AND high activity difference.

    Regression / binary: |delta_y| > k * MAD(y).
    Classification (is_clf): different class (high sim + different label).

    Args:
        sim_threshold: absolute similarity cutoff (used only when
            ``similar_percentile`` is None).
        similar_percentile: if set (e.g. 5.0), define "structurally similar" as
            the top ``similar_percentile``% most-similar off-diagonal pairs
            instead of the absolute ``sim_threshold``. On standardised high-D
            embeddings the pairwise Euclidean distance is large so sim = 1/(1+d)
            never reaches an absolute 0.55 (s12 found 0 cliffs in 13/16
            datasets). A percentile makes the threshold distribution-relative.
            See COMPLEXITY_SIMILARITY_BROKEN_2026_06_03.md.
            If a list/tuple of candidate percentiles is passed, the expensive
            similarity matrix is built ONCE and every candidate is swept; the
            one whose cliff-compound prevalence is closest to
            ``TARGET_CLIFF_PCT`` (10%) is auto-selected (preferring candidates
            with prevalence > 0). The returned ``info`` then also carries
            ``'percentile_sweep'`` and ``'selected_percentile'``.

    Returns (cliff_indices, scanned_indices, info). scanned_indices holds every
    row the pair scan actually looked at: above max_compounds the scan runs on
    a subsample, and a row outside it was never tested for being a cliff. A
    caller must not score such a row as a non-cliff row.
    """
    from experiments.similarity.similarity_helpers import _pairwise_delta_matrix

    TARGET_CLIFF_PCT = 10.0
    sweep_candidates = None
    if isinstance(similar_percentile, (list, tuple)):
        sweep_candidates = list(similar_percentile)

    n = len(y)
    if n > max_compounds:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_compounds, replace=False)
        X_sub = X_concat[idx]
        y_sub = y[idx]
    else:
        X_sub = X_concat
        y_sub = y
        idx = np.arange(n)

    D = compute_distance_matrix(X_sub, standardize=True)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)

    if is_clf:
        activity_threshold = 0.5  # indicator is 0 or 1
        mad_y = 0.0
    else:
        mad_y = float(np.median(np.abs(y_sub - np.median(y_sub))))
        activity_threshold = activity_k * mad_y

    # "Structurally similar" pairs: absolute cutoff, or distribution-relative
    # (top similar_percentile% most-similar off-diagonal pairs). The sim matrix
    # and delta_y are the expensive part; given any sim_cutoff the cliff stats
    # are cheap boolean ops, so we factor them into a closure that can be swept
    # over several candidate percentiles essentially for free.
    delta_y = _pairwise_delta_matrix(y_sub, is_clf=is_clf)
    _off = sim[~np.eye(len(sim), dtype=bool)]

    def _percentile_cutoff(pct):
        return float(np.percentile(_off, 100.0 - pct))

    def _cliff_stats(sim_cutoff):
        cliff_pair_mask = (sim >= sim_cutoff) & (delta_y > activity_threshold)
        np.fill_diagonal(cliff_pair_mask, False)
        cliff_mask_sub = cliff_pair_mask.any(axis=1)
        n_cliff_pairs = int(cliff_pair_mask.sum()) // 2
        n_cliff_compounds = int(cliff_mask_sub.sum())
        pct_cliff = float(100.0 * n_cliff_compounds / len(y_sub))
        return cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff

    percentile_sweep = None
    selected_percentile = None
    if sweep_candidates is not None:
        # Build the expensive matrices once (already done), then sweep cheaply.
        percentile_sweep = {}
        for cand in sweep_candidates:
            _, _, _, pct_cliff = _cliff_stats(_percentile_cutoff(cand))
            percentile_sweep[cand] = pct_cliff
        # Auto-select: prevalence closest to the 10% target, preferring
        # candidates with prevalence > 0 (only fall back to including 0%
        # candidates if every candidate is 0%).
        positive = [c for c in sweep_candidates if percentile_sweep[c] > 0]
        pool = positive if positive else list(sweep_candidates)
        selected_percentile = float(
            min(pool, key=lambda c: abs(percentile_sweep[c] - TARGET_CLIFF_PCT)))
        sim_cutoff = _percentile_cutoff(selected_percentile)
    elif similar_percentile is not None:
        sim_cutoff = _percentile_cutoff(similar_percentile)
    else:
        sim_cutoff = sim_threshold

    cliff_mask_sub, n_cliff_pairs, n_cliff_compounds, pct_cliff_compounds = \
        _cliff_stats(sim_cutoff)

    # Map back to full indices
    cliff_indices = set(idx[cliff_mask_sub].tolist())
    # Every row the scan covered. Rows outside it were never tested.
    scanned_indices = set(int(i) for i in idx.tolist())

    info = {
        'distance_metric': str(get_distance_metric()),
        'sim_threshold': sim_threshold,
        'similar_percentile': similar_percentile,
        'sim_cutoff': float(sim_cutoff),
        'activity_k': activity_k,
        'activity_threshold': float(activity_threshold),
        'mad_y': float(mad_y),
        'n_cliff_pairs': n_cliff_pairs,
        'n_cliff_compounds': n_cliff_compounds,
        'pct_cliff_compounds': pct_cliff_compounds,
        'n_analyzed': len(y_sub),
        'n_unanalyzed': max(0, n - max_compounds),
        'max_compounds_cap': max_compounds,
        # The coefficient the distance behind sim = 1/(1+d) was taken with,
        # read from the process-wide setting rather than asserted. It said
        # 'euclidean' on every file written since the default moved to
        # manhattan on 2026-09-04.
        'similarity_type': str(get_distance_metric()),
    }
    if percentile_sweep is not None:
        info['percentile_sweep'] = percentile_sweep
        info['selected_percentile'] = selected_percentile

    return cliff_indices, scanned_indices, info


def identify_cliff_pairs_euclidean(*args, **kwargs):
    """Two-value wrapper kept for callers written before the scanned set existed.

    Returns (cliff_indices, info). The name is historical: the distance has not
    been Euclidean since 2026-09-04 — info['similarity_type'] says which
    coefficient was used. New callers want identify_cliff_pairs, whose third
    value is the set of rows the scan covered.
    """
    cliff_indices, _scanned, info = identify_cliff_pairs(*args, **kwargs)
    return cliff_indices, info


# =========================================================================
# CLIFF PARTITIONS
# =========================================================================

def _concat_partitions(data, rep_names, metrics):
    """One structural similarity per decided distance on the concatenation.

    The concatenation is the dataset's general representations (DATASET_REPS
    [ds]['general'] where the dataset has an entry, else every loaded one),
    scaled with kirby.scaling.FeatureScaler over every row (0/1 features
    untouched, the rest standardised). The scaling reads features only, no
    labels. Similarity is 1/(1+d) under each distance in `metrics`. The
    cutoff yielded here is the top CLIFF_SIMILARITY_PERCENTILE per cent of
    every pair; fold_cliff_partition replaces it with the same percentile over
    each fold's training pairs.

    Yields (name, similarity, cutoff, source sentence).
    """
    from kirby.scaling import FeatureScaler
    from experiments.noise.similarity_space import feature_distance
    reps = data['reps']
    X = np.hstack([np.asarray(reps[k], dtype=np.float64) for k in rep_names])
    X = FeatureScaler().fit(X).transform(X)
    off_mask = ~np.eye(len(X), dtype=bool)
    for metric in metrics:
        D = feature_distance(X, X, metric)
        sim = 1.0 / (1.0 + D)          # an inf distance (cosine, no spread) -> 0
        np.fill_diagonal(sim, 0.0)
        cutoff = float(np.percentile(sim[off_mask], 100.0 - CLIFF_SIMILARITY_PERCENTILE))
        del D
        yield (f'concat_{metric}', sim, cutoff,
               (f'1/(1+d) on the concatenation of {len(rep_names)} '
                f'representations ({X.shape[1]} features), 0/1 features '
                f'unscaled and the rest standardised over every row, d taken '
                f'with {metric}; structurally similar means the top '
                f'{CLIFF_SIMILARITY_PERCENTILE} per cent of pairs by that '
                f'similarity'))


# How a held-out molecule is called a cliff molecule without its own label.
# Stored in every result file next to the per-fold rule.
HELDOUT_CLIFF_RULE = (
    'a held-out molecule is a cliff molecule when at least two training '
    'molecules structurally similar to it (similarity at or above the '
    "fold's cutoff) differ in label by more than the activity gap solved on "
    "the fold's training molecules (classification: are of different "
    "classes). The held-out molecule's own label is not read.")


def fold_cliff_partition(structural_sim, labels, tr, va, *, cutoff,
                         cutoff_kind, structural_source, target_pct, is_clf,
                         activity_k=None):
    """Which of one outer fold's held-out molecules are cliff molecules,
    decided from the fold's training molecules only.

    DECIDED 2026-10-06 (Adelaide delegated), shared by s04 and s12.
    structural_sim is the n x n similarity over every row (it reads features
    or SMILES, never labels). On the training rows: the structural cutoff
    (recomputed as the fixed percentile over training pairs when cutoff_kind
    is 'concatenation_similarity_percentile'; ECFP4 Tanimoto keeps its fixed
    cutoff) and similarity_helpers.calibrated_cliff_rule on the training
    labels, which solves the activity gap. Held-out molecules: see
    HELDOUT_CLIFF_RULE.

    Returns (heldout_mask, info): heldout_mask is boolean over the positions
    of `va`; info is calibrated_cliff_rule's record for the training molecules
    (its counts are over training molecules) plus the held-out counts.
    """
    tr = np.asarray(tr)
    va = np.asarray(va)
    y = np.asarray(labels)
    y_tr = y[tr]
    S = np.asarray(structural_sim)
    S_tr = S[np.ix_(tr, tr)]
    if cutoff_kind == 'concatenation_similarity_percentile':
        off = S_tr[~np.eye(len(tr), dtype=bool)]
        cutoff = float(np.percentile(
            off, 100.0 - CLIFF_SIMILARITY_PERCENTILE))
        del off
    _train_mask, info = calibrated_cliff_rule(
        S_tr, y_tr, structural_cutoff=cutoff,
        structural_source=structural_source, target_pct=target_pct,
        is_clf=is_clf, activity_k=activity_k)
    del S_tr
    gap = float(info['activity_threshold'])
    similar = S[np.ix_(va, tr)] >= cutoff
    n_partners = similar.sum(axis=1)
    held = np.zeros(len(va), dtype=bool)
    for i in np.where(n_partners >= 2)[0]:
        P = y_tr[similar[i]]
        if is_clf:
            held[i] = len(np.unique(P)) > 1
        else:
            held[i] = float(np.max(P) - np.min(P)) > gap
    info = {k: v for k, v in info.items() if k != 'cliff_pair_counts'}
    info.update({
        'scope': 'fold_training_molecules',
        'counts_note': ('n_molecules, n_cliff_compounds, n_cliff_pairs, '
                        'pct_cliff_compounds and the uncalibrated_* counts '
                        "are over the fold's training molecules"),
        'heldout_rule': HELDOUT_CLIFF_RULE,
        'n_heldout_molecules': int(len(va)),
        'n_heldout_with_two_similar_training': int((n_partners >= 2).sum()),
        'n_heldout_cliff': int(held.sum()),
        'pct_heldout_cliff': (float(100.0 * held.sum() / len(va))
                              if len(va) else 0.0),
    })
    return held, info


# =========================================================================
# RUN
# =========================================================================

def run(data, config, output_dir, dataset_name, **kwargs):
    """S12: cross-domain activity cliff analysis, per outer fold.

    Reports, per outer fold and per cliff partition, the dataset's metric over
    the held-out cliff molecules and over the held-out non-cliff molecules
    (and over all held-out molecules) for every arm listed in the module
    docstring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT S12: CROSS-DOMAIN ACTIVITY CLIFFS")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'])
    domain = data.get('domain', 'molecular')
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    task_type = data['task_type']
    is_clf = task_type == 'classification'

    # The budget is no longer an argument (2026-10-05): the hybrids come from
    # per-fold searches over the dataset's decided budgets. Before this date it
    # was a fixed 100 features, outside the decided general range on 19
    # datasets.
    if 'budget' in kwargs:
        print(f"  budget={kwargs['budget']!r} ignored: the hybrids come from "
              f"per-fold searches on the decided grid")

    # The dataset's two decided models (DECISIONS.md Step 1). knn enters only
    # through the distance search (Step 9), inside fold_similarity_selection.
    try:
        models = [m for m in decided_models(dataset_name, 'general')
                  if m != 'knn']
    except UndecidedSearch as exc:
        print(f"  SKIP: no decided models for {dataset_name}: {exc}")
        return {'skipped': True,
                'reason': f'DECISIONS.md decides no general search: {exc}',
                'metric': metric_name, 'lower_is_better': bool(lower_is_better)}
    factories = get_model_factories(models, task_type=task_type,
                                    n_samples=len(labels))
    # knn as the model a distance hybrid is rebuilt with, when its recipe's
    # internal evaluator is 'match' (fold_hybrid_scoring_kwargs).
    knn_factory = get_model_factories(['knn'], task_type=task_type,
                                      n_samples=len(labels)).get('knn')
    print(f"  Models: {models}")

    # The decided kNN settings for this dataset (KNN_SEARCH_SPACE_RECORD.md
    # axes 2, 4, 5, 6). They also give the distances the concatenation
    # partition is taken under.
    axes = knn_axes(dataset_name, len(labels))
    print(f"  kNN settings ({axes['group']}): distances {axes['metrics']}, "
          f"neighbours {axes['ks']}, counting {axes['schemes']}, scaffold "
          f"group not excluded")

    # Representations: the dataset's general list only (handoff 2.4 step 9).
    # Used for the concatenation partition; data['reps'] itself is passed
    # whole to the selection.
    general = (DATASET_REPS.get(dataset_name) or {}).get('general')
    rep_names = (sorted(r for r in reps if r in set(general)) if general
                 else sorted(reps))
    if general and len(rep_names) < len(general):
        print(f"  {len(general) - len(rep_names)} general representations not "
              f"loaded: {sorted(set(general) - set(rep_names))}")

    # 2026-09-20: WHICH MOLECULES ARE CLIFFS. This file and s04 used to decide
    # that two different ways (on herg_ki s04 called 451 of 1412 molecules
    # cliffs and this file 295). Both files now call
    # similarity_helpers.calibrated_cliff_rule. On a dataset with SMILES the
    # structural side is the same ECFP4 Tanimoto s04 builds; on a dataset
    # without SMILES it is the concatenation similarity at a FIXED percentile,
    # and the activity gap is solved to the target share either way.
    activity_k = kwargs.get('activity_k')       # None = calibrate
    cliff_target_pct = kwargs.get('cliff_target_pct', CLIFF_TARGET_PCT)
    # Passing force_concat_similarity=True takes the concatenation branch on a
    # molecular dataset too, which makes the two branches comparable there.
    force_concat = bool(kwargs.get('force_concat_similarity', False))

    smiles = data.get('smiles')
    fold_splits = sim_outer_folds(data, config)
    n_folds = len(fold_splits)

    def _structural():
        """(partition name, similarity, cutoff, source, kind), one at a time,
        so only one n x n similarity is held in memory."""
        if (domain == 'molecular' and not force_concat and smiles is not None
                and len(smiles) == len(labels)):
            sim, cutoff, note, kind = cliff_structural_similarity(
                data, {k: reps[k] for k in rep_names}, is_molecular=True)
            if kind == 'smiles_ecfp4_tanimoto':
                yield ('ecfp4_tanimoto', sim, cutoff, note, kind)
                return
            # The ECFP4 build failed and the helper fell back to the
            # concatenation under one decided distance; the partitions under
            # every decided distance below replace it.
            del sim
        for name, sim, cutoff, note in _concat_partitions(
                data, rep_names, axes['metrics']):
            yield (name, sim, cutoff, note,
                   'concatenation_similarity_percentile')

    # DECIDED 2026-10-06 (Adelaide delegated): which molecules are cliffs is
    # decided per outer fold on that fold's training molecules only
    # (fold_cliff_partition). Before this date it was decided once over every
    # row, held-out labels included.
    partitions = {}      # name -> {fold: boolean mask over held-out positions}
    partition_info = {}
    for name, sim, cutoff, note, kind in _structural():
        masks, by_fold = {}, {}
        for fold_i, (tr, va) in enumerate(fold_splits):
            held, info = fold_cliff_partition(
                sim, labels, tr, va, cutoff=cutoff, cutoff_kind=kind,
                structural_source=note, target_pct=cliff_target_pct,
                is_clf=is_clf, activity_k=activity_k)
            masks[fold_i] = held
            by_fold[str(fold_i)] = info
        n_held = int(sum(len(m) for m in masks.values()))
        n_cliff = int(sum(int(m.sum()) for m in masks.values()))
        partitions[name] = masks
        partition_info[name] = {
            'structural_kind': kind,
            'structural_source': note,
            'shared_with_s04': (kind == 'smiles_ecfp4_tanimoto'),
            'scope': 'per outer fold, training molecules only',
            'heldout_rule': HELDOUT_CLIFF_RULE,
            'n_heldout_molecules': n_held,
            'n_cliff_compounds': n_cliff,
            'pct_cliff_compounds': (float(100.0 * n_cliff / n_held)
                                    if n_held else 0.0),
            'pct_note': ('held-out cliff molecules summed over the outer '
                         'folds, over held-out molecules summed over the '
                         'outer folds'),
            'by_fold': by_fold,
        }
        print(f"  Cliff partition {name}: {note}")
        for f, info in by_fold.items():
            print(f"    fold {f}: training cliff molecules "
                  f"{info['n_cliff_compounds']} of {info['n_molecules']} "
                  f"({info['pct_cliff_compounds']:.1f} per cent), activity "
                  f"gap {info['activity_threshold']:.4f} label units; "
                  f"held-out cliff molecules {info['n_heldout_cliff']} of "
                  f"{info['n_heldout_molecules']}")
        del sim

    if not any(m.any() for masks in partitions.values()
               for m in masks.values()):
        # No activity cliffs is a LEGITIMATE methodological outcome, not a
        # failure: the shared rule solves the activity gap down towards zero,
        # so this is reached only when the structural side alone flags no
        # pair at all. Signalled with the 'skipped'/'reason' convention (cf.
        # s04) — no 'error' key — so the experiment is counted complete.
        print("  No held-out cliff molecules in any fold under any "
              "partition. Reporting as a valid no-cliff result, not an "
              "error.")
        first = next(iter(partition_info.values()), {})
        return {
            'skipped': True,
            'reason': 'no_cliffs_found',
            'domain': domain,
            'metric': metric_name,
            'lower_is_better': bool(lower_is_better),
            'cliff_info': convert_numpy(first),
            'cliff_partitions': convert_numpy(partition_info),
        }

    results = {
        'domain': domain,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'selection_scope': 'fold_train_rows',
        'n_outer_folds': n_folds,
        'models': list(models),
        'models_without_factory': [m for m in models if m not in factories],
        # The first partition, under the key older readers use
        # (_local_read_results reads cliff_info.pct_cliff_compounds, here the
        # held-out share over all outer folds). Every partition, with its
        # per-fold rule, is in cliff_partitions.
        'cliff_info': convert_numpy(next(iter(partition_info.values()))),
        'cliff_partitions': convert_numpy(partition_info),
        'min_rows_per_fold_score': _MIN_ROWS_PER_FOLD_SCORE,
        'knn_settings': {
            'group': axes['group'],
            'distances': list(axes['metrics']),
            'neighbour_counts': list(axes['ks']),
            'counting_schemes': list(axes['schemes']),
            'scaffold_group_excluded': False,
            'source': axes['source'],
            'note': ('every kNN arm is scored at every (distance, neighbour '
                     'count, counting scheme); features are scaled on the '
                     "fold's training rows (0/1 features unscaled, the rest "
                     'standardised, held-out values cut to the training '
                     'range); train_rows_loo_score predicts each training '
                     'molecule from the other training molecules, itself '
                     'excluded by row index'),
        },
        'folds': {},
    }

    from experiments._checkpoint import (
        Checkpoint, experiment_sidecar, run_identity)
    _ck = Checkpoint(
        experiment_sidecar(output_dir, 's12_cross_domain_cliffs', dataset_name),
        run_identity('s12_cross_domain_cliffs', dataset_name, data, config,
                     kwargs=kwargs, models=list(models),
                     knn_axes={k: v for k, v in axes.items()
                               if k != 'source'},
                     partitions=sorted(partitions),
                     min_rows_per_fold_score=_MIN_ROWS_PER_FOLD_SCORE,
                     version=_CHECKPOINT_VERSION))

    def _score(y_true, pred, proba, classes, rows, min_rows):
        """The dataset's metric on held-out positions `rows` (None = all);
        NaN below min_rows rows or where the metric is undefined on that
        subset (one class only, constant labels)."""
        if rows is not None and len(rows) < min_rows:
            return float('nan')
        sl = slice(None) if rows is None else rows
        try:
            if proba is not None:
                return float(score_prediction(y_true[sl], pred[sl], proba[sl],
                                              classes, metric_name, metric_fn))
            return float(metric_fn(y_true[sl], pred[sl]))
        except Exception:
            return float('nan')

    def _strata(y_va, pred, proba, classes, strata):
        """score_all plus, per partition, the cliff and non-cliff scores."""
        out = {'score_all': _score(y_va, pred, proba, classes, None, 2),
               'by_partition': {}}
        for pname, (rc, rn) in strata.items():
            out['by_partition'][pname] = {
                'score_cliff': _score(y_va, pred, proba, classes, rc,
                                      _MIN_ROWS_PER_FOLD_SCORE),
                'score_noncliff': _score(y_va, pred, proba, classes, rn,
                                         _MIN_ROWS_PER_FOLD_SCORE)}
        return out

    def _model_arm(factory, X_tr, X_va, y_tr, y_va, strata):
        X_tr_s, X_va_s = scale_fold(X_tr, X_va)
        model = factory()
        model.fit(X_tr_s, y_tr)
        pred = np.asarray(_predict_for_metric(model, X_va_s, metric_fn))
        return _strata(y_va, pred, None, None, strata)

    def _knn_arm(X_tr, X_va, y_tr, y_va, strata):
        """One row per decided (distance, neighbour count, counting scheme)."""
        X_tr_s, X_va_s = scale_fold(X_tr, X_va)
        rows = []
        for dist_name in axes['metrics']:
            idx, dist = knn_lists(X_tr_s, X_va_s, metric=dist_name,
                                  k_max=axes['k_max'])
            # Leave-one-out on the training rows, each row's own index
            # excluded (dirty-run bug 7): the score a setting is chosen by.
            idx_t, dist_t = knn_lists(X_tr_s, None, metric=dist_name,
                                      k_max=axes['k_max'])
            for k in axes['ks']:
                for scheme in axes['schemes']:
                    row = {'distance': dist_name, 'k': int(k),
                           'scheme': scheme}
                    if k > idx.shape[1] or k > idx_t.shape[1]:
                        row['note'] = f'only {idx.shape[1]} training rows'
                        rows.append(row)
                        continue
                    pred, proba, classes = knn_from_lists(
                        y_tr, idx, dist, k=k, scheme=scheme, is_clf=is_clf)
                    row.update(_strata(y_va, pred, proba, classes, strata))
                    p_t, pr_t, c_t = knn_from_lists(
                        y_tr, idx_t, dist_t, k=k, scheme=scheme,
                        is_clf=is_clf)
                    row['train_rows_loo_score'] = _score(
                        y_tr, p_t, pr_t, c_t, None, 2)
                    rows.append(row)
        return rows

    def _run_fold(fold_i, tr, va):
        y_tr, y_va = labels[tr], labels[va]
        strata = {}
        counts = {}
        for pname, masks in partitions.items():
            mask = masks[fold_i]
            rc = np.where(mask)[0]
            rn = np.where(~mask)[0]
            strata[pname] = (rc, rn)
            counts[pname] = {'n_cliff_rows': int(len(rc)),
                             'n_noncliff_rows': int(len(rn))}
        rec = {'fold': int(fold_i), 'selection_scope': 'fold_train_rows',
               'n_train_rows': int(len(tr)), 'n_heldout_rows': int(len(va)),
               'heldout_rows_by_partition': counts,
               'model_arms': {}, 'knn_arms': {}, 'hybrid_width_built': {},
               'errors': []}
        print(f"\n  Fold {fold_i}: {len(tr)} training rows, {len(va)} held "
              f"out; cliff rows per partition "
              f"{ {p: c['n_cliff_rows'] for p, c in counts.items()} }")

        sel = fold_similarity_selection(
            data, dataset_name, train_idx=tr, fold_i=fold_i,
            n_outer_folds=n_folds, neighbour_spaces=True)
        rec['unavailable'] = dict(sel.get('unavailable') or {})
        rec['selection'] = {
            'models': {
                m: (None if s is None else {
                    'base': list(s.get('base') or []),
                    'base_train_rows_scores': [
                        b.get('score') for b in
                        ((s.get('base_detail') or {}).get('selected') or [])],
                    'base_shortfall': (s.get('base_detail') or {}).get(
                        'shortfall_reason'),
                    'hybrids': [entry_record(e)
                                for e in (s.get('hybrids') or [])],
                    'hybrids_skipped_identical_features': list(
                        s.get('hybrids_skipped_identical_features') or []),
                })
                for m, s in sel['models'].items()},
            'knn_base': list(sel.get('knn_base') or []),
            'knn_base_train_rows_scores': [
                b.get('score') for b in
                ((sel.get('knn_base_detail') or {}).get('selected') or [])],
            'distance_hybrid': entry_record(sel.get('distance_hybrid')),
            'similarity_hybrid': entry_record(sel.get('similarity_hybrid')),
        }

        # Each hybrid rebuilt once on the fold's training rows with the fold
        # search's own representation scores (build_fold_entry; dirty-run
        # bug 1).
        hybrid_spaces = []        # (arm, entry, model factory for the rebuild)
        for m in models:
            s = sel['models'].get(m)
            for j, e in enumerate((s or {}).get('hybrids') or [], 1):
                hybrid_spaces.append((f'{m}|hybrid{j}', e, factories.get(m)))
        if sel.get('distance_hybrid') is not None:
            hybrid_spaces.append(('distance_hybrid', sel['distance_hybrid'],
                                  knn_factory))
        if sel.get('similarity_hybrid') is not None:
            # A similarity-objective recipe is rebuilt with no model.
            hybrid_spaces.append(('similarity_hybrid',
                                  sel['similarity_hybrid'], None))
        built = {}
        for arm, e, fac in hybrid_spaces:
            try:
                X_tr_h, X_va_h, _fi = build_fold_entry(
                    data, e, train_idx=tr, val_idx=va, model_factory=fac,
                    metric_fn=metric_fn, lower_is_better=lower_is_better)
                X_tr_h, X_va_h = np.asarray(X_tr_h), np.asarray(X_va_h)
                built[arm] = (X_tr_h, X_va_h, e)
                rec['hybrid_width_built'][arm] = {
                    'budget': getattr(e.get('blueprint'), 'budget', None),
                    'width_built': int(X_tr_h.shape[1])}
            except Exception as exc:
                rec['errors'].append({'arm': arm, 'stage': 'rebuild',
                                      'error': f'{type(exc).__name__}: {exc}'})
                print(f"    {arm}: rebuild failed: {type(exc).__name__}: {exc}")

        # --- Decided-model arms: 2 base representations + 2 general hybrids
        for m in models:
            fac = factories.get(m)
            s = sel['models'].get(m)
            if fac is None or s is None:
                rec['model_arms'][m] = None
                continue
            arms = {}
            base_scores = [b.get('score') for b in
                           ((s.get('base_detail') or {}).get('selected') or [])]
            for j, rname in enumerate(s.get('base') or [], 1):
                try:
                    X = np.asarray(reps[rname])
                    arms[f'base{j}'] = dict(
                        rep=rname,
                        train_rows_score=(base_scores[j - 1]
                                          if j <= len(base_scores) else None),
                        **_model_arm(fac, X[tr], X[va], y_tr, y_va, strata))
                except Exception as exc:
                    rec['errors'].append({'arm': f'{m}|base{j}', 'stage': 'fit',
                                          'error': f'{type(exc).__name__}: {exc}'})
            for j in (1, 2):
                arm = f'{m}|hybrid{j}'
                if arm not in built:
                    continue
                X_tr_h, X_va_h, e = built[arm]
                try:
                    arms[f'hybrid{j}'] = dict(
                        hybrid=entry_record(e),
                        train_rows_score=e.get('score_mean'),
                        **_model_arm(fac, X_tr_h, X_va_h, y_tr, y_va, strata))
                except Exception as exc:
                    rec['errors'].append({'arm': arm, 'stage': 'fit',
                                          'error': f'{type(exc).__name__}: {exc}'})
            rec['model_arms'][m] = arms
            for pname in partitions:
                _line = ', '.join(
                    f"{k}={v['by_partition'][pname]['score_cliff']:.4f}"
                    for k, v in arms.items())
                print(f"    {m} held-out {metric_name} on cliff rows "
                      f"[{pname}]: {_line or '-'}")

        # --- kNN arms: knn's 2 base representations, the distance hybrid and
        # the similarity hybrid, at every decided setting.
        knn_spaces = []           # (arm, kind, name, X_tr, X_va)
        for j, rname in enumerate(sel.get('knn_base') or [], 1):
            if rname not in reps:
                rec['errors'].append({'arm': f'knn_base{j}', 'stage': 'load',
                                      'error': f'{rname} not loaded'})
                continue
            X = np.asarray(reps[rname])
            knn_spaces.append((f'knn_base{j}', 'representation', rname,
                               X[tr], X[va]))
        for arm in ('distance_hybrid', 'similarity_hybrid'):
            if arm in built:
                X_tr_h, X_va_h, e = built[arm]
                knn_spaces.append((arm, 'hybrid', e.get('label'),
                                   X_tr_h, X_va_h))
        for arm, kind, name, X_tr, X_va in knn_spaces:
            try:
                rows = _knn_arm(X_tr, X_va, y_tr, y_va, strata)
            except Exception as exc:
                rec['errors'].append({'arm': arm, 'stage': 'knn',
                                      'error': f'{type(exc).__name__}: {exc}'})
                continue
            rec['knn_arms'][arm] = {'kind': kind, 'name': name,
                                    'settings': rows}
            ok = [r for r in rows if r.get('train_rows_loo_score') is not None
                  and np.isfinite(r['train_rows_loo_score'])]
            if ok:
                b = (min if lower_is_better else max)(
                    ok, key=lambda r: r['train_rows_loo_score'])
                rec['knn_arms'][arm]['chosen_on_train_rows'] = {
                    'distance': b['distance'], 'k': b['k'],
                    'scheme': b['scheme']}
                for pname in partitions:
                    print(f"    kNN {arm:<18} ({name}) setting chosen on "
                          f"training rows [{b['distance']}, k={b['k']}, "
                          f"{b['scheme']}]: held-out {metric_name} on cliff "
                          f"rows [{pname}] = "
                          f"{b['by_partition'][pname]['score_cliff']:.4f}")
        return rec

    # --- One record per outer fold, written to the sidecar as it finishes ---
    for fold_i, (tr, va) in enumerate(fold_splits):
        if _ck.done(fold=fold_i):
            rec = _ck.get(fold=fold_i)
            print(f"  Fold {fold_i}: read back from the checkpoint")
        else:
            rec = convert_numpy(_run_fold(fold_i, np.asarray(tr),
                                          np.asarray(va)))
            # A fold with an error is not stored, so a rerun retries it.
            if not rec['errors']:
                _ck.put(rec, fold=fold_i)
        results['folds'][str(fold_i)] = rec

    results['summary'] = _summarise(results, models, list(partitions),
                                    lower_is_better, metric_name)
    return results


# =========================================================================
# SUMMARY
# =========================================================================

def _spread(vals):
    """Mean and spread (standard deviation, ddof=1) over the finite per-fold
    values, with 'one fold, no spread' when fewer than two are finite."""
    ok = [float(v) for v in vals if v is not None and np.isfinite(v)]
    note = fold_spread_note(ok)
    if note and len(vals) > 1:
        # Several folds, but fewer than two with a score (a fold below
        # _MIN_ROWS_PER_FOLD_SCORE rows of the stratum is NaN).
        note = f'{len(ok)} of {len(vals)} folds scored, no spread'
    return {'per_fold': [None if v is None or not np.isfinite(v) else float(v)
                         for v in vals],
            'mean': float(np.mean(ok)) if ok else None,
            'spread_sd': (None if note or not ok
                          else float(np.std(ok, ddof=1))),
            'spread_note': note}


def _summarise(results, models, partition_names, lower_is_better, metric_name):
    """Per model and partition: hybrid1 against base1 on the cliff rows and on
    the non-cliff rows, per fold, both chosen on the fold's training rows.
    Per kNN arm and partition: the held-out score at the setting chosen on the
    training rows. Direction-corrected deltas: + means the hybrid did better.
    Per-fold values, their mean and their spread only."""
    folds = sorted(results['folds'], key=int)
    sign = -1.0 if lower_is_better else 1.0
    out = {'delta_formula': ('(hybrid1 - base1) if higher is better, '
                             '(base1 - hybrid1) if lower is better; + means '
                             'hybrid1 did better'),
           'models': {}, 'knn_arms': {}}
    print(f"\n  Summary ({metric_name}, lower_is_better={bool(lower_is_better)};"
          f" + means the hybrid did better):")
    for m in models:
        out['models'][m] = {}
        for pname in partition_names:
            block = {}
            for stratum in ('score_cliff', 'score_noncliff'):
                h_vals, b_vals, d_vals = [], [], []
                for f in folds:
                    arms = (results['folds'][f].get('model_arms') or {}).get(m) or {}
                    h = ((arms.get('hybrid1') or {}).get('by_partition') or {}
                         ).get(pname, {}).get(stratum)
                    b = ((arms.get('base1') or {}).get('by_partition') or {}
                         ).get(pname, {}).get(stratum)
                    h = float('nan') if h is None else float(h)
                    b = float('nan') if b is None else float(b)
                    h_vals.append(h)
                    b_vals.append(b)
                    d_vals.append(sign * (h - b))
                block[stratum] = {'hybrid1': _spread(h_vals),
                                  'base1': _spread(b_vals),
                                  'delta_directed': _spread(d_vals)}
            out['models'][m][pname] = block
            c = block['score_cliff']
            if c['delta_directed']['mean'] is not None:
                print(f"    {m} [{pname}] cliff rows: hybrid1 "
                      f"{c['hybrid1']['mean']:.4f} vs base1 "
                      f"{c['base1']['mean']:.4f}, directed delta per fold "
                      f"{[None if v is None else round(v, 4) for v in c['delta_directed']['per_fold']]}"
                      f"{' (' + c['delta_directed']['spread_note'] + ')' if c['delta_directed']['spread_note'] else ''}")
    arm_names = sorted({a for f in folds
                        for a in (results['folds'][f].get('knn_arms') or {})})
    for arm in arm_names:
        out['knn_arms'][arm] = {}
        for pname in partition_names:
            block = {}
            for stratum in ('score_cliff', 'score_noncliff'):
                vals, picks = [], []
                for f in folds:
                    a = (results['folds'][f].get('knn_arms') or {}).get(arm) or {}
                    ch = a.get('chosen_on_train_rows')
                    v = float('nan')
                    if ch:
                        for r in a.get('settings') or []:
                            if (r.get('distance'), r.get('k'), r.get('scheme')) \
                                    == (ch['distance'], ch['k'], ch['scheme']):
                                x = (r.get('by_partition') or {}).get(
                                    pname, {}).get(stratum)
                                v = float('nan') if x is None else float(x)
                    vals.append(v)
                    picks.append(ch)
                block[stratum] = dict(_spread(vals), settings_per_fold=picks)
            out['knn_arms'][arm][pname] = block
    return out


if __name__ == '__main__':
    parser = create_experiment_parser('S12: Cross-Domain Activity Cliffs')
    # 2026-09-20: --sim-threshold and --similar-percentile are gone with the
    # rule they configured. The structural side is fixed (ECFP4 Tanimoto 0.55
    # where there are SMILES, otherwise a fixed percentile of the concatenation
    # similarity under each decided distance) and the activity gap is solved
    # to --cliff-target-pct.
    parser.add_argument('--cliff-target-pct', type=float,
                        default=CLIFF_TARGET_PCT,
                        help='the share of this dataset\'s molecules the cliff '
                             'stratum is calibrated to, in per cent')
    parser.add_argument('--activity-k', type=float, default=None,
                        help='skip the calibration and use an activity gap of '
                             'this many times the median absolute deviation of '
                             'the labels, which is what this file did before '
                             '2026-09-20 with 2.0')
    parser.add_argument('--force-concat-similarity', action='store_true',
                        help='take the concatenation-similarity branch even on '
                             'a molecular dataset, instead of the ECFP4 '
                             'Tanimoto s04 also uses')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, args.reps)
    _run_kwargs = {'cliff_target_pct': args.cliff_target_pct,
                   'activity_k': args.activity_k,
                   'force_concat_similarity': args.force_concat_similarity}
    results = run(data, config, args.output, args.dataset, **_run_kwargs)
    save_experiment_results(results, args.output, 's12_cross_domain_cliffs', args.dataset)
