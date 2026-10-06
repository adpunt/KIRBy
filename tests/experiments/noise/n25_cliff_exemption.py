#!/usr/bin/env python3
"""N25: does exempting activity-cliff molecules from cleaning improve the result?

Paper: Noise.

THE QUESTION. A molecule sitting on an activity cliff carries a CORRECT label
that looks wrong to any detector that judges a label against its structural
neighbours' labels: its neighbours are close in structure and far in activity,
which is what a mislabelled molecule also looks like. Every cleanup in this
suite then rewrites, deletes or downweights it, and what it removes is real
signal. This file measures whether holding those molecules out of the cleaning
improves the held-out score.

THE DESIGN IS PAIRED AND THE PAIR IS THE EXPERIMENT. On one fold, one noise
draw, one feature space, one detector and one setting, `cleaned_training_set`
is called ONCE. What comes back is then fitted three times:

  plain           exactly what the cleanup produced
  cliff_exempt    the same thing with every cliff molecule put back: its label
                  restored, its weight set to 1, its row undeleted
  random_exempt   the same thing with as many NON-cliff molecules put back,
                  drawn at random from that fold's training molecules

So the detector scores, the flag threshold, the out-of-fold and Gaussian-process
targets, the hybrid features and the downstream model are the same object in all
three arms, and the arms differ in one thing: which molecules were put back. The
headline is the per-pass difference within a pair, never two independent means.

WHY THE THIRD ARM IS NOT OPTIONAL. Putting molecules back means cleaning fewer
molecules, and cleaning fewer molecules can help on its own. `random_exempt`
puts back the same NUMBER of molecules and none of them is on a cliff, so it
measures that alone. A cliff exemption that wins no more passes than the random
one has shown that less cleaning helps, not that cliffs are special. It costs
one model fit per (pass, space, detector, setting, model, cleanup), the same as
either other arm.

WHY `downweight` IS IN THE CLEANUP LIST BESIDE THE LABEL-REWRITING ONES.
`downweight` and `huber_weight` change no label at all; they lower a molecule's
weight in the fit. `remove` deletes the row. The other seven overwrite the
label. If the exemption helps the weighting cleanups as much as it helps the
rewriting ones, then what matters is WHICH molecules are touched; if it helps
only the rewriting ones, what matters is WHAT is done to them.
`summary.per_space[space].by_what_the_cleanup_does` puts the three families side
by side for exactly that reading.

WHAT DEFINES A CLIFF HERE, AND THE TWO DEFINITIONS THAT DISAGREE. Three already
exist in this repository and no fourth is written here:

  n01.identify_cliff_pairs     a pair is a cliff when the two molecules are
                               structurally similar -- ECFP4 Tanimoto at or
                               above 0.55, or edit distance on the heavy chains
                               for an antibody set -- AND their activities
                               differ by more than `activity_k` times the median
                               absolute deviation of the label column. A
                               molecule is a cliff molecule when it is in at
                               least one such pair.
  n14._cliff_mask_from_similarity
                               the same structural test, with the activity test
                               replaced by an ABSOLUTE 2 log units, the
                               literature high-confidence criterion.
  ActivityCliffAwareEstimator.is_likely_cliff_
                               no structure reference at all: a molecule whose
                               label sits more than one label standard deviation
                               from the mean label of its ten nearest neighbours
                               IN THE FEATURE SPACE BEING CLEANED, while those
                               ten neighbours disagree among themselves by more
                               than one label standard deviation.

THE FIRST ONE IS WHAT THIS FILE USES, and the reason is the label axis rather
than the structure axis. n14's threshold is 2 log units and is only meaningful
where the label is a log potency: herg_ki is pKi and 2 log units is 100-fold,
flab_thermo_tresanco is a melting temperature in degrees and 2 of them is not a
cliff, it is noise. n01's threshold is in median absolute deviations of that
dataset's own label column, so it is defined on every dataset this suite runs.
n14's is still reachable with --cliff-threshold-mode absolute, and whichever
ran is written to `cliff_reference` on the result.

The estimator's definition is NOT used to choose the exempt set, and that is
deliberate: it finds cliffs in the same feature space and from the same
neighbour labels that the detector uses to decide what is suspect, so exempting
by it would mostly be exempting the molecules the detector was about to flag,
by construction. It appears instead as one of the three DETECTORS, where it is
the alternative remedy this file's own remedy is measured against: sparing
cliffs inside the suspicion score against sparing them after it.

⛔ ON A DATASET WITH NO STRUCTURE REFERENCE THE COMPARISON IS CONTAMINATED.
Where there are no SMILES and no heavy chains -- TCGA, and any molecular set
RDKit refuses -- the reference falls back to Euclidean distance on every
representation glued together, which is n01's own fallback and n01 records the
same thing. The cliff set is then a function of the representations, and those
are what the detector's neighbourhoods are built from too, so `cliff_exempt`
drifts toward "exempt whatever the neighbourhood finds odd".
`cliff_reference_independent` is False on those runs and the comparison must not
be reported as a clean one.

WHY NOISE IS INJECTED, AND WHICH KINDS OF DAMAGE. n18 asks whether cleaning
helps on the labels as they come and needs no injection. This question is not
that one: it turns on a molecule whose label is RIGHT being treated as wrong,
so the run has to know which labels are right, and some labels have to be left
exactly as the loader produced them. Until 2026-10-05 the default was
outlier_only_p05, which the noise grid decision (KNN_SEARCH_SPACE_RECORD, "The
noise grid", Adelaide 2026-10-05) cut. Since then (chat 5) the default sweeps
the decided kinds that leave the unselected labels as delivered: censoring,
shift_only_p10 and sparse_dprime_up_p10 on a regression dataset, the six flip
kinds on a classification one (DEFAULT_CONDITIONS). outlier_p05 is decided
and detection is defined under it, but every training label moves under it,
so it is not in the default; --noise-condition outlier_p05 still runs it.
Levels are the decided ones for the dataset's group (noise_helpers.
decided_levels), level 0 once, three draws at every other level.

WHAT THE RUN KNOWS BECAUSE IT INJECTED. Two counts that need no model fit and
are the premise of the whole file, recorded per (pass, configuration,
detector):
  * of the molecules the detector ranks most suspicious -- as many of them as
    the draw actually corrupted -- how many are cliff molecules whose label
    never moved. Those are the molecules this exemption exists to protect.
  * of this fold's cliff molecules, how many the draw DID corrupt. Exempting
    those is the cost of the exemption: a genuinely wrong label kept.

HOW MANY MOLECULES THE EXEMPTION ACTUALLY SPARED, PER SETTING. A cleanup that
would not have touched a single cliff molecule at this setting produces two
identical arms and a difference of exactly zero. That is recorded as
`n_cliffs_spared` out of `n_cliff_training` out of `n_training`, counted as the
molecules whose label, weight or presence differs between the plain arm and the
exempt arm -- so it is defined for the deleting, reweighting and rewriting
cleanups alike, and `settings_where_nothing_was_spared` counts the settings
where the pair was degenerate.

THE HEADLINE IS A COUNT OF PASSES, NOT A MEAN. One pass is one (fold, noise
draw) at one level. The reported number is how many passes the exempt arm beat
the plain arm on, out of the passes both scored, with the mean difference and
its spread over folds beside it and never instead of it. Every level is
counted on its own.

THREE DRAWS AT EVERY NON-ZERO LEVEL (chat 5, 2026-10-05). Until then n25 ran
one draw, on the argument that both arms of a pair see the same drawn labels
so the draw cancels out of the paired difference. It does not cancel out of
the absolute scores printed beside it, and the decided noise grid runs three
draws, so n25 now does too. --noise-seeds still overrides. Censoring reads no
seed, so it runs one draw (noise_helpers.draws_for, 2026-10-06).

THE FEATURE SPACES (chat 5, 2026-10-05; HANDOFF_EXPERIMENT_ALIGNMENT 2.4).
Folds are noise_helpers.noise_outer_folds, the splits the per-fold searches
are filed under. On each fold, noise_helpers.fold_configurations runs the
dataset's searches on that fold's training rows (decided grid, pruning off)
and gives each of the dataset's two decided models (DECISIONS.md Step 1) its
2 best single representations and its 2 best general hybrids. Each model is
fitted on its own 4 configurations, base_1, base_2, hybrid and
hybrid__general2: 2 models x 4 = 8 configurations (R5). The hybrids are rebuilt
on the fold's training rows at the DAMAGED labels of each draw
(noise_helpers.fold_model_inputs), from recipes the fold's search chose on
the clean training labels. Nothing is read from a whole-dataset search. The
distance search's top hybrid (hybrid__distance1) and the similarity search's
top hybrid (hybrid_similarity) are neighbour spaces only, never fitted
configurations. `fold_records` holds every fold's choice.

The detectors, the out-of-fold and Gaussian-process targets, the suspicion
counts and the molecules-spared counts all live inside one configuration's
feature matrix, so they are computed once per (pass, model, configuration).

WHAT THIS CANNOT SAY. Nothing here shows that the exempted molecules are on
real cliffs -- the structure reference says they are structurally similar to a
molecule of very different activity, which is what a cliff and a pair of
mislabelled neighbours both look like. A win says the model did better when
those molecules were left alone.

THE NEIGHBOUR SETTINGS (KNN_SEARCH_SPACE_RECORD, Adelaide 2026-10-05).
For each configuration, every setting of the dataset's group
(n07.sweep_settings: no model_input, no importance weighting, distances per
group, scaffold-mates kept in every list) is scored with no model fit at
every neighbour count (5, 10, 20) and counting scheme (rank, inverse_square)
on how well its neighbours predict the labels of a held-back fifth of the
training molecules (n07.screen_settings with inner=). The top 3 are refitted
with all three arms.
  the main counts    DECIDED 2026-10-06 (Adelaide delegated): the per_level
                     counts use each (fold, configuration)'s top screened
                     setting, kept[0]. The neighbour detectors
                     (neighbor_residual, activity_cliff_aware) and
                     label_smooth / knn_smooth read its lists, neighbour count
                     and counting scheme. prox_smooth reads them only when the
                     setting is forest_proximity and otherwise grows its own
                     forest. Until 2026-10-06 they found 9 or 10 Euclidean
                     neighbours in the configuration's own features, each
                     counting equally. If the screen keeps nothing, the
                     neighbour detectors, label_smooth and knn_smooth are not
                     run on that pass and `main_neighbour_setting` says why.
  the other two      kept[1] and kept[2] write rows only.
The neighbour detectors are also graded against the corrupted molecules at
every (setting, neighbour count, counting scheme), which fits no model.
The cliff set is not on either path -- see the TODO in cliff_mask_for_dataset.
Every arm writes one D-versus-D' row per pass (n07.RowSink); every finished
(fold, level, draw) is written to a checkpoint file (n07.FoldCheckpoint) as
it finishes, so a killed run redoes only the pass in flight. The shared
controls' twins of the plain cleanup run with KIRBY_NOISE_CONTROLS=1; n25's
random_exempt always runs.

Usage:
    python n25_cliff_exemption.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from experiments._shared import (
    load_data, get_config, create_experiment_parser, convert_numpy,
    get_domain_metric, get_model_factories, input_fingerprint,
)
from experiments.decided_grid import decided_models
from experiments.dataset_policy import noise_groups_for
from kirby.noise_spec import CLASSIFICATION_STRATEGIES, GROUPED_STRATEGIES
from experiments.noise.noise_helpers import (
    detection_scores, DETECTION_IS_DEFINED, DECIDED_REGRESSION_CONDITIONS,
    sweep_conditions, noise_n_folds, noise_outer_folds, fold_spread,
    fold_configurations, fold_model_inputs, configuration_record,
    inject_condition, decided_levels, dataset_knn_group, level_unit,
    CONFIGURATION_SPACES, draws_for,
)
# The cleanups, the detector and the two baselines are n07's, called rather
# than copied. `cleaned_training_set` is the shared dispatcher n17 and n18 also
# call. The neighbour-setting pieces (sweep_settings to FoldCheckpoint) are
# n07's shared ones since chat 5 (2026-10-05); until then n25 took its own
# copies from n23.
from experiments.noise.n07_remediation import (
    cleaned_training_set, correction_targets, targets_needed,
    _evaluate_fold, _predict_for_metric, REPAIR_OOF_SPLITS,
    CLEANUPS as N07_CLEANUPS,
    REGRESSION_ONLY_CLEANUPS as N07_REGRESSION_ONLY,
    sweep_settings, sweep_ks, sweep_schemes, screen_settings, screen_top,
    SCREEN_SCORED_ON, usable_ks, fold_similarity_spaces, fold_neighbours,
    FoldCheckpoint, RowSink, row_identity, dprime_rows, similarity_record,
    NO_NEIGHBOURS,
)
from experiments.noise.d_vs_dprime import rows_for
# The cliff reference, reused rather than redefined. n14 imports the same two
# names from n01 for the same reason.
from experiments.noise.n01_activity_cliff_framework import (
    identify_cliff_pairs, _compute_ecfp4_tanimoto_sim,
    _compute_reaction_tanimoto_sim, _compute_dta_sim, _compute_euclidean_sim,
    SIMILAR_PERCENTILE,
)
from experiments.noise import noise_cache as nc
from experiments.noise import similarity_space as ss
from experiments.noise.controls import controls_enabled


EXP_FULL_NAME = 'n25_cliff_exemption'

# What this file writes since chat 5 (2026-10-05, HANDOFF_EXPERIMENT_ALIGNMENT
# section 2.4). In the condition files' fingerprint, the rows file name and the
# checkpoint's fingerprint, so nothing the whole-dataset design wrote before
# that date is read back as done.
RESULT_LAYOUT = 'fold_aligned_2026-10-06'

# The similarity_setting on a row whose cleanup found its neighbours in
# the configuration's own features at the cleanup's or detector's own count.
TODAY_SIMILARITY = f'{ss.MODEL_INPUT.name}|k=today'

# The cleanups that, given no neighbour list, find neighbours in the configuration's own
# features (n07.cleaned_training_set): label_smooth and knn_smooth through a
# nearest-neighbour fit, prox_smooth through its own forest.
TODAY_NEIGHBOUR_CLEANUPS = ('label_smooth', 'knn_smooth', 'prox_smooth')

# Models whose fit noise_cache may store: seeded and deterministic. dnn is not.
CACHEABLE_MODELS = ('rf', 'svm', 'lightgbm', 'xgboost')


def cleanup_reads_setting(cleanup, setting):
    """Whether n07.cleaned_training_set reads `setting`'s neighbour list for
    this cleanup: label_smooth and knn_smooth read any setting; prox_smooth
    reads only forest_proximity (n07 refuses the rest)."""
    if cleanup in ('label_smooth', 'knn_smooth'):
        return True
    if cleanup == 'prox_smooth':
        return setting.space == ss.FOREST_SPACE
    return False


def smiles_for(data, idx):
    """The rows' SMILES for tanimoto_ecfp4, or None without SMILES."""
    smi = data.get('smiles')
    if smi is None:
        return None
    return [smi[int(i)] for i in np.asarray(idx)]


def correction_targets_cached(dataset_name, X_tr_s, y_tr, groups, want):
    """n07.correction_targets with its halves stored through noise_cache: the
    out-of-fold target through cached_oof_predict (n07's call), the Gaussian
    process and quantile targets under kind 'correction_targets'. Moved here
    from n23 on 2026-10-05 (chat 5), unchanged, so n25 imports nothing from
    another experiment's file."""
    want = set(want or ())
    if not want:
        return {}
    out = {}
    if 'oof' in want:
        try:
            out['oof'] = nc.cached_oof_predict(
                dataset_name, X_tr_s, y_tr, groups,
                n_splits=min(REPAIR_OOF_SPLITS, len(y_tr)))
        except Exception as exc:
            print(f"      oof target failed: {type(exc).__name__}: {exc}")
            out['oof'] = None
    rest = sorted(want - {'oof'})
    if rest:
        key = {'X': nc.array_hash(X_tr_s), 'y': nc.array_hash(y_tr),
               'groups': nc.array_hash(groups), 'want': rest}
        out.update(nc.cached(
            dataset_name, 'correction_targets', key,
            lambda: correction_targets(X_tr_s, y_tr, groups=groups,
                                       want=set(rest)),
            store_if=lambda v: all(v.get(k) is not None for k in rest)))
    return out


# Every cleanup n07 registers, taken from n07 rather than written out again.
# ⛔ THE LIST IS NOT COPIED ON PURPOSE. n18 carried its own four-name copy while
# n07's had grown to seven, so every n18 number for nine months came from a menu
# of four and nothing in the result file said so. A cleanup added to n07's
# CLEANUPS tuple appears here on the next run.
ALL_CLEANUPS = tuple(N07_CLEANUPS)

# Not defined on a nominal class label. n07's tuple, for the same reason.
REGRESSION_ONLY_CLEANUPS = tuple(N07_REGRESSION_ONLY)

# The one cleanup n07 registers that this question is not defined for, and why.
# Every other cleanup asks a detector which labels look wrong, so a cliff
# molecule can be wrongly named and the exemption has something to spare it
# from. `invvar_weight` asks the record how many times the molecule was
# measured and reads no suspicion score and no label at all
# (n07.SCORE_FREE_CLEANUPS, read in this session), so a cliff molecule's weight
# is low because it was measured once and not because it looked wrong. There is
# nothing for an exemption to correct. It is also refused outright on every
# dataset whose loader averaged a molecule's repeats before the experiment saw
# them, which as of 2026-09-23 is all of them but herg_ic50.
#
# NOT CUT, EXCLUDED FROM THE DEFAULT: --cleanups invvar_weight runs it, and
# `cleanups_not_run_because` on the result says this.
UNDEFINED_HERE = {
    'invvar_weight': (
        'it reads no suspicion score and no label -- the weight comes from how '
        'many times the molecule was measured -- so no cliff molecule is ever '
        'named suspect by it and the exemption has nothing to spare it from'),
}

# What each cleanup does to a molecule it finds suspect. This is the axis the
# `downweight` arm exists to read: if the exemption helps the two weighting
# cleanups as much as the seven rewriting ones, the effect is about WHICH
# molecules are touched rather than about WHAT is done to them.
#
# Read out of n07.cleaned_training_set in this session, branch by branch:
#   remove        returns a `keep` mask with the flagged rows dropped
#   downweight    returns a sample_weight from weights_from_scores; no label moves
#   huber_weight  returns a sample_weight from Huber's redescending weight
#   the rest      return a full-length `label_column` with values overwritten
WHAT_THE_CLEANUP_DOES = {
    'remove': 'deletes_the_molecule',
    'downweight': 'lowers_the_weight',
    'huber_weight': 'lowers_the_weight',
    'label_smooth': 'rewrites_the_label',
    'knn_smooth': 'rewrites_the_label',
    'oof_smooth': 'rewrites_the_label',
    'gp_smooth': 'rewrites_the_label',
    'quantile_clip': 'rewrites_the_label',
    'js_shrink': 'rewrites_the_label',
    'prox_smooth': 'rewrites_the_label',
    # Added 2026-10-05 (f36): raises the labels at the ceiling by one constant.
    'constant_lift': 'rewrites_the_label',
    # Reachable only with --cleanups; see UNDEFINED_HERE below for why it is
    # not in the default list. It sets a row weight from the measurement count.
    'invvar_weight': 'lowers_the_weight',
}

# The three arms. 'plain' is the score to beat and is what n07 and n18 already
# measure; the other two put molecules back.
ARMS = ('plain', 'cliff_exempt', 'random_exempt')

# How many training molecules in every hundred are treated as suspect, for the
# cleanups that threshold. n18's grid, unchanged, so the two files' settings
# line up.
FLAG_RATES = (0.01, 0.05, 0.20)

# What the rate-blind cleanups sweep INSTEAD: how hard they lean on the
# suspicion score, in each one's own units, three values each so every cleanup
# is scored at three settings and the counts stay comparable.
#
# ⛔ THESE ARE NOT IMPORTED FROM n18 AND THE REASON IS A DEFECT IN n18's COPY.
# n18.reads_flag_rate answers True for any cleanup it does not name, and three
# of n07's ten are unnamed there: js_shrink and huber_weight read no quantile at
# all (checked in cleaned_training_set this session -- js_shrink scales a
# variance by `strength`, huber_weight sets the tuning constant c), while
# prox_smooth does threshold at the (1 - rate) quantile. n18 runs none of the
# three, so nothing it has reported is wrong; this file runs all ten and needs
# the answer to be right for each.
STRENGTHS = {
    'downweight': (1.0, 3.0, 9.0),      # row weights of 2:1, 4:1 and 10:1
    'label_smooth': (0.25, 0.5, 1.0),   # a quarter, half, all the way
    'oof_smooth': (0.25, 0.5, 1.0),
    'gp_smooth': (0.25, 0.5, 1.0),
    # The per-molecule variance is multiplied by this before the shrinkage
    # fraction sigma2 / (sigma2 + tau2) is formed, so 1.0 is the derived
    # James-Stein value and the other two bracket it.
    'js_shrink': (0.5, 1.0, 2.0),
    # Huber's tuning constant. 1.345 is the value giving 95 percent efficiency
    # under a normal (Huber 1964) and is what the branch uses when the caller
    # passes nothing; a smaller c weights down sooner.
    'huber_weight': (0.5, 1.345, 3.0),
}

# Which cleanups threshold the suspicion score at the (1 - flag rate) quantile.
THRESHOLDING_CLEANUPS = ('remove', 'knn_smooth', 'prox_smooth')


# The cleanup that is another cleanup under a second name on a class label,
# and which one it duplicates. Dirty-run bug 13 (DIRTY_RUN_RESULTS.md, read
# 2026-10-05): on a classification set n07.cleaned_training_set gives
# label_smooth and knn_smooth the same branch -- the flagged share replaced by
# the majority class of the same CLF_NEIGHBOURS neighbours, or the same
# similarity list -- so the two produce the same training set at every flag
# rate. Run once, under label_smooth's name. Regression is unaffected.
# DECIDED 2026-10-06 (Adelaide delegated): on a classification dataset
# label_smooth only, never knn_smooth as well. A run asked for knn_smooth on
# a classification dataset runs label_smooth in its place.
SAME_AS_ON_A_CLASS_LABEL = {'knn_smooth': 'label_smooth'}


def cleanups_for(is_clf, asked=None):
    """The cleanups this task type can actually run."""
    out = (tuple(asked) if asked
           else tuple(c for c in ALL_CLEANUPS if c not in UNDEFINED_HERE))
    if not is_clf:
        return out
    out = tuple(c for c in out if c not in REGRESSION_ONLY_CLEANUPS)
    renamed = []
    for c in out:
        c = SAME_AS_ON_A_CLASS_LABEL.get(c, c)
        if c not in renamed:
            renamed.append(c)
    return tuple(renamed)


def reads_flag_rate(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag rate?"""
    # prox_smooth is asked BEFORE the thresholding tuple, which names it.
    # Until 2026-10-05 (chat 5) the tuple was asked first, so this branch
    # never ran and a regression prox_smooth was "swept" over three flag rates
    # that it does not read: three identical fits.
    if cleanup == 'prox_smooth':
        # Since 2026-10-05 (f36) every label moves a quarter of the way to its
        # forest-neighbours' mean on a regression set, and no score is read;
        # on a classification set the flagged share is replaced.
        return bool(is_clf)
    if cleanup in THRESHOLDING_CLEANUPS:
        return True
    if cleanup in ('downweight', 'huber_weight', 'js_shrink',
                   # Reads the labels and the out-of-fold prediction only.
                   'constant_lift',
                   # Reads no suspicion score at all, so no quantile of one.
                   'invvar_weight'):
        return False
    if cleanup in ('label_smooth', 'oof_smooth', 'gp_smooth'):
        # All three smooth every row in proportion to its score on a regression
        # set and threshold only on a classification one.
        return bool(is_clf)
    if cleanup == 'quantile_clip':
        # Reads neither axis: a label is inside the range its own leaves carry
        # or it is not.
        return False
    # An unknown cleanup is assumed to read it. Scoring it at one setting and
    # calling that the sweep is the worse error.
    return True


def setting_axis(cleanup, is_clf):
    """(axis name, the values) this cleanup is swept over."""
    if reads_flag_rate(cleanup, is_clf):
        return 'flag_rate', tuple(FLAG_RATES)
    vals = STRENGTHS.get(cleanup)
    if not vals:
        return 'none', (None,)
    return 'strength', tuple(vals)


# Which per-molecule suspicion score decides what is suspect.
#
# activity_cliff_aware IS IN THIS TUPLE AND IT IS NOT IN n18's. n18 cut it on
# 2026-09-18 as neighbor_residual in a second name: Spearman 0.9843 to 0.9888
# over every molecule on the three local datasets. That measurement stands and
# is not re-litigated. It is here because in THIS file it is not a spare
# detector slot, it is the rival remedy: it scores a molecule suspect only when
# its ten nearest neighbours agree with each other, which is cliff-sparing done
# inside the suspicion score. Exempting cliffs after the fact is the other way
# of doing it, and the point of running both is that the two can be read
# against each other on the same folds.
#
# neighbor_residual judges a label against its neighbours' labels; cross_val
# judges it against what a model trained on the other molecules predicts, out of
# fold on scaffold-disjoint folds. The second is the one a cliff should NOT fool
# in the same way, so the pair covers two kinds of evidence.
DETECTORS = ('neighbor_residual', 'cross_val', 'activity_cliff_aware')

# The kinds of damage swept when the caller names none (chat 5, 2026-10-05).
# The decided regression kinds (noise_helpers.DECIDED_REGRESSION_CONDITIONS,
# KNN_SEARCH_SPACE_RECORD "The noise grid") under which some training labels
# stay as delivered, so a cliff molecule the draw did not touch carries its
# real label: censoring (only the clipped labels move), shift_only_p10 and
# sparse_dprime_up_p10 (only the drawn tenth moves). gaussian and
# grouped_shifted move every label and are outside DETECTION_IS_DEFINED.
# outlier_p05 is inside it but moves every label too (noise_helpers, note at
# DETECTION_IS_DEFINED), so it is reachable only by --noise-condition.
# DECIDED 2026-10-06 (Adelaide delegated): outlier_p05 stays out of n25's
# default sweep; --noise-condition outlier_p05 still runs it.
# Until 2026-10-05 the default was the single kind outlier_only_p05, which
# the noise grid decision cut.
UNTOUCHED_LABELS_EXIST = ('censoring', 'shift_only_p10', 'sparse_dprime_up_p10')
DEFAULT_REGRESSION_CONDITIONS = tuple(
    c for c in DECIDED_REGRESSION_CONDITIONS
    if c in DETECTION_IS_DEFINED and c in UNTOUCHED_LABELS_EXIST)

# The regression kinds this file runs at all, named or not (2026-10-05):
# detection is defined under them AND the noise grid decision kept them.
# DETECTION_IS_DEFINED alone also holds 12 kinds that decision cut
# (outlier_p01/p10, outlier_only_p01/p05/p10, grouped_wider, shift_only_p05,
# sparse_dprime_p02/p05/p10, sparse_dprime_up_p02/p05). Naming one of them
# used to run it at the decided level grid, which is not in that kind's own
# unit. Now it is refused. That leaves censoring, shift_only_p10,
# sparse_dprime_up_p10 and outlier_p05.
# DECIDED 2026-10-06 (Adelaide delegated): outlier_only_p05, n25's default
# until 2026-10-05, stays unreachable here. The kinds above already leave
# some labels as delivered.
ALLOWED_REGRESSION_CONDITIONS = tuple(
    c for c in DECIDED_REGRESSION_CONDITIONS if c in DETECTION_IS_DEFINED)


def default_conditions(is_clf):
    """The kinds swept when the caller names none. A classification set takes
    its flip kinds (the decided grid keeps them): a flip moves the drawn share
    and leaves every other class label as delivered."""
    return (list(CLASSIFICATION_STRATEGIES) if is_clf
            else list(DEFAULT_REGRESSION_CONDITIONS))


def condition_allowed(condition, is_clf):
    """Whether this file can run `condition` on this task type."""
    if is_clf:
        return condition in CLASSIFICATION_STRATEGIES
    return condition in ALLOWED_REGRESSION_CONDITIONS


# Three draws at every non-zero level, level 0 once (the decided grid).
NOISE_SEEDS = (0, 1, 2)

# The configurations one model is fitted on (noise_helpers.CONFIGURATION_SPACES):
# its rank-1 and rank-2 single representation and its two general hybrids,
# all chosen on the fold's training rows.
SPACES = tuple(CONFIGURATION_SPACES)
BASE_SPACES = ('base_1', 'base_2')
# The two whose features are chosen from the labels, so they are rebuilt at
# each draw's damaged labels.
HYBRID_CONFIGURATIONS = tuple(sp for sp in SPACES if sp not in BASE_SPACES)

# n01's activity cutoff, in median absolute deviations of this dataset's own
# label column, and n01's structural cutoff.
CLIFF_ACTIVITY_K = 2.0
CLIFF_SIM_THRESHOLD = 0.55
# n14's absolute criterion, reachable with --cliff-threshold-mode absolute.
CLIFF_POTENCY_THRESHOLD = 2.0

# The seed the random exemption is drawn with. Fixed, so two runs of the same
# dataset put back the same molecules and the control arm is reproducible.
RANDOM_EXEMPTION_SEED = 20260923


# ---------------------------------------------------------------------------
# The exempt set
# ---------------------------------------------------------------------------

def cliff_mask_for_dataset(data, labels, is_clf, sim_threshold=CLIFF_SIM_THRESHOLD,
                           activity_k=CLIFF_ACTIVITY_K, threshold_mode='mad',
                           potency_threshold=CLIFF_POTENCY_THRESHOLD,
                           subset=None, structure=None, structure_info=None):
    """One boolean per molecule: is it in at least one activity-cliff pair.

    ⛔ WHAT `subset` AND `structure` ARE FOR, added 2026-09-23.

    This used to be called ONCE, before the fold loop, on data['labels'] -- the
    CLEAN label column -- across every molecule in the dataset, and the result
    was sliced per fold. Two faults, and they are different.

      It used labels nobody has. Under injection the practitioner sees
      corrupted labels. Choosing the exempt molecules from the clean ones tells
      the exemption which labels are right, which is the question n25 asks.

      A training molecule's cliff status depended on held-out molecules. The
      mask was computed over all molecules at once, so whether a training
      molecule counted as a cliff was decided partly by its similarity and its
      activity difference to molecules in the test fold. Slicing afterwards does
      not undo that.

    `subset` is the molecule indices to compute over -- the fold's training rows
    -- and `labels` is then what those molecules' labels LOOK LIKE, corrupted
    and all. Both faults go together: the mask is built from what the method can
    actually see.

    `structure` is the pairwise structural similarity from a previous call, so
    the expensive half is not redone. It depends on the molecules alone and
    never on the labels, so one matrix serves every fold, level and draw; pass
    the whole-dataset matrix and the subset is taken from it here. Get it from
    cliff_structure_for_dataset.

    `structure_info` is the reference the matrix was built with, from the
    call that built it: {'structure_reference', 'similar_percentile',
    'reference_independent'}. Until 2026-10-06 a reused matrix was always
    cut at the top SIMILAR_PERCENTILE of pairs, so on an ECFP4 dataset the
    whole-dataset call cut at Tanimoto 0.55 and every per-fold call cut at
    the top 2 percent of pairs: two definitions of "similar" in one run.
    With structure_info the reused matrix is cut the way it was first cut.
    Without it (callers outside n25) the old rule stands.

    A MOLECULE'S OWN REPEATS ARE NOT CLIFF PARTNERS. DECIDED 2026-10-06
    (Adelaide delegated). On a dataset loaded with one row per measurement
    (herg_ic50: 12,008 rows for 9,802 molecules) two rows of one molecule
    have Tanimoto 1.0, so two repeats more than the activity cutoff apart
    were counted as a cliff pair. Rows with the same SMILES are never a
    pair now; info['n_own_repeat_pairs_dropped'] counts the pairs that
    were removed.

    The reference is built OUTSIDE the representations wherever the dataset
    offers one, which is n01's rule and n01's code: ECFP4 Tanimoto from the
    SMILES for a molecular set, edit distance on the heavy chains for an
    antibody set, Tanimoto on the reactant and product bits for a reaction, and
    Tanimoto on the drug within one protein for a drug-target affinity row.
    A dataset with none of those falls back to Euclidean distance on every
    representation glued together, and the returned info says so.

    threshold_mode:
      'mad'       |delta activity| > activity_k * median absolute deviation of
                  the label column. n01's, and the default -- it is defined
                  whatever units the label is in.
      'absolute'  |delta activity| > potency_threshold, in the label's own
                  units. n14's high-confidence literature criterion, and only
                  meaningful where the label is a log potency.

    Returns (cliff_mask, info).
    """
    labels = np.asarray(labels)
    domain = data.get('domain', 'molecular')
    smiles = data.get('smiles')
    heavy_seqs = data.get('heavy_seqs')
    rxn_smiles = data.get('rxn_smiles')
    drug_smiles = data.get('drug_smiles')

    # Everything below sees the subset and nothing else, so no held-out
    # molecule can decide whether a training molecule is on a cliff.
    if subset is not None:
        subset = np.asarray(subset)
        if len(labels) != len(subset):
            labels = labels[subset]
        _take = lambda seq: (None if seq is None
                             else [seq[i] for i in subset])
        smiles, heavy_seqs = _take(smiles), _take(heavy_seqs)
        rxn_smiles, drug_smiles = _take(rxn_smiles), _take(drug_smiles)
        if structure is not None:
            structure = np.asarray(structure)[np.ix_(subset, subset)]
    n = len(labels)
    # One id per molecule, so rows of the same molecule are never a pair.
    same_molecule = None
    if smiles is not None and len(smiles) == n:
        _ids = {}
        mol_id = np.array([_ids.setdefault(str(x), len(_ids)) for x in smiles])
        if len(_ids) < n:
            same_molecule = mol_id[:, None] == mol_id[None, :]
            np.fill_diagonal(same_molecule, False)

    info = {
        'threshold_mode': threshold_mode,
        'activity_k_in_median_absolute_deviations': (
            activity_k if threshold_mode == 'mad' else None),
        'potency_threshold_label_units': (
            potency_threshold if threshold_mode == 'absolute' else None),
        'sim_threshold': sim_threshold,
        'reference_independent': True,
        'definition_source': (
            'n01.identify_cliff_pairs' if threshold_mode == 'mad'
            else 'n14 absolute potency criterion, structure reference from n01'),
    }

    # ── The structural reference ────────────────────────────────────────────
    # TODO(similarity-sweep 2026-09-25): the cliff set is NOT on the
    # similarity axis. It is a PAIRWISE test -- similarity at or above 0.55
    # Tanimoto, or inside the top SIMILAR_PERCENTILE of pairs, AND an activity
    # gap (n01.identify_cliff_pairs) -- and similarity_space returns k-nearest
    # neighbour lists, not a pairwise matrix with a threshold. Putting it on
    # the axis needs a decision the owner has not made: which threshold means
    # "similar" in a Manhattan or forest distance (a percentile of pairs, as
    # the fallback here uses, or "among the k nearest"), and whether the
    # exempt set may then come from the same features the detector reads,
    # which this file's docstring calls contamination. Until then every
    # setting of the sweep exempts the same ECFP4 cliff set.
    similar_percentile = None
    if heavy_seqs is not None and domain == 'antibody':
        from experiments.similarity.similarity_helpers import (
            identify_cliff_compounds_editdist)
        if threshold_mode == 'absolute':
            raise ValueError(
                "n25: --cliff-threshold-mode absolute asks for a fixed gap in "
                "the label's own units, and the antibody reference "
                "(identify_cliff_compounds_editdist) takes a multiplier of the "
                "median absolute deviation and nothing else. Run this dataset "
                "under the default mad mode.")
        cliff_mask, ci = identify_cliff_compounds_editdist(
            heavy_seqs, labels, activity_k=activity_k, is_clf=is_clf,
            similar_percentile=SIMILAR_PERCENTILE)
        info.update({'structure_reference': 'heavy_chain_edit_distance',
                     'similar_percentile': SIMILAR_PERCENTILE})
        info.update({k: v for k, v in ci.items()
                     if k not in ('per_compound_cliff_pairs',)})
        cliff_mask = np.asarray(cliff_mask, dtype=bool)
        info['n_cliff_molecules'] = int(cliff_mask.sum())
        info['n_molecules'] = int(n)
        return cliff_mask, info

    # DECIDED 2026-10-06 (Adelaide delegated): on a dataset loaded with one
    # row per measurement (herg_ic50) a molecule's own repeats are not cliff
    # partners. n25 still runs on the rows as loaded; the pairs between rows
    # of one molecule are removed below (same_molecule).
    sim = structure
    if sim is not None:
        if structure_info is not None:
            # Cut the way the call that built the matrix cut it.
            info['structure_reference'] = (
                f"{structure_info.get('structure_reference')}, reused from "
                f"the whole-dataset call")
            similar_percentile = structure_info.get('similar_percentile')
            if not structure_info.get('reference_independent', True):
                info['reference_independent'] = False
        else:
            info['structure_reference'] = 'reused from a previous call'
            similar_percentile = SIMILAR_PERCENTILE
        info['similar_percentile'] = similar_percentile
    if sim is None and smiles is not None and domain == 'molecular':
        try:
            sim = _compute_ecfp4_tanimoto_sim(smiles)
            info['structure_reference'] = 'ecfp4_tanimoto'
        except Exception as exc:
            print(f"  ECFP4 Tanimoto failed ({type(exc).__name__}: {exc})")
            sim = None
    elif sim is None and domain == 'reaction' and rxn_smiles is not None:
        try:
            sim = _compute_reaction_tanimoto_sim(rxn_smiles)
            info['structure_reference'] = 'reaction_ecfp4_tanimoto'
            similar_percentile = SIMILAR_PERCENTILE
        except Exception as exc:
            print(f"  reaction Tanimoto failed ({type(exc).__name__}: {exc})")
            sim = None
    elif sim is None and domain == 'dta' and drug_smiles is not None:
        try:
            _meta = data.get('metadata') or {}
            sim = _compute_dta_sim(drug_smiles, _meta.get('protein_ids'))
            info['structure_reference'] = 'drug_ecfp4_tanimoto_within_protein'
            similar_percentile = SIMILAR_PERCENTILE
        except Exception as exc:
            print(f"  drug Tanimoto failed ({type(exc).__name__}: {exc})")
            sim = None

    if sim is None:
        names = sorted(data['reps'].keys())
        # The SUBSET's rows. Without this the fallback built its similarity from
        # every molecule in the dataset while the labels were the subset's, so
        # the two disagreed on size and -- worse, had they matched -- a held-out
        # molecule would have decided a training molecule's cliff status.
        _stack = np.hstack([np.asarray(data['reps'][k])[subset]
                            if subset is not None
                            else np.asarray(data['reps'][k])
                            for k in names])
        sim = _compute_euclidean_sim(_stack)
        similar_percentile = SIMILAR_PERCENTILE
        info.update({
            'structure_reference': 'euclidean_on_every_representation',
            'reference_independent': False,
            'n_features_per_rep': {k: int(np.asarray(data['reps'][k]).shape[1])
                                   for k in names},
            'n_molecules_the_reference_was_built_over': int(len(labels)),
            'contamination_note': (
                'no reference outside the representations exists for this '
                'dataset, so the cliff set is a function of the same feature '
                'matrices the detector finds its neighbours in. The exempt set '
                'and the suspect set are then not independent and the paired '
                'difference below must not be read as a clean one.'),
        })
    info['similar_percentile'] = similar_percentile

    # ── The activity test ───────────────────────────────────────────────────
    if threshold_mode == 'mad':
        pair_mask, ci = identify_cliff_pairs(
            sim, labels, sim_threshold, activity_k, is_clf=is_clf,
            similar_percentile=similar_percentile)
        info.update(ci)
    else:
        off = sim[~np.eye(n, dtype=bool)]
        cutoff = (float(np.percentile(off, 100.0 - similar_percentile))
                  if similar_percentile is not None and off.size
                  else float(sim_threshold))
        delta = np.abs(labels[:, None] - labels[None, :])
        pair_mask = (sim >= cutoff) & (delta > float(potency_threshold))
        np.fill_diagonal(pair_mask, False)
        info['sim_cutoff'] = cutoff
        info['n_cliff_pairs'] = int(pair_mask.sum()) // 2

    # A molecule's own repeats are not cliff partners (2026-10-06).
    info['n_own_repeat_pairs_dropped'] = 0
    if same_molecule is not None:
        dropped = int((pair_mask & same_molecule).sum()) // 2
        pair_mask = pair_mask & ~same_molecule
        info['n_own_repeat_pairs_dropped'] = dropped
        info['n_cliff_pairs'] = int(pair_mask.sum()) // 2
        _nc = int(pair_mask.any(axis=1).sum())
        if 'n_cliff_compounds' in info:
            info['n_cliff_compounds'] = _nc
            info['pct_cliff_compounds'] = (float(100.0 * _nc / n) if n > 0
                                           else 0.0)

    cliff_mask = pair_mask.any(axis=1)
    info['n_cliff_molecules'] = int(cliff_mask.sum())
    info['n_molecules'] = int(n)
    # For a caller that will ask again on a subset of these molecules. It
    # depends on the molecules alone and never on a label, so it is the half
    # that does not need redoing. ⛔ POP IT BEFORE WRITING info TO A RESULT
    # FILE: it is n x n floats.
    info['similarity_matrix'] = sim
    return cliff_mask, info


def random_exempt_mask(cliff_train, n_exempt, fold_i, seed_i):
    """As many NON-cliff training molecules as the cliff exemption holds out.

    Drawn from the molecules the cliff mask did not name, so the two exempt
    sets never overlap and the control differs from the treatment in one thing:
    whether the molecules put back are on cliffs.

    ⛔ WHEN THERE ARE NOT ENOUGH NON-CLIFF MOLECULES (dirty-run bug, n25
    "random comparison size", DIRTY_RUN_RESULTS.md: on buchwald 83% of the
    molecules are cliffs). The control then puts back fewer molecules than the
    cliff arm, so it differs from the treatment in two ways at once: which
    molecules and how many. The returned count is what was drawn; the caller
    records the shortfall per pass and leaves those passes out of the
    cliff-against-random comparison (_level_block), saying how many.
    DECIDED 2026-10-06 (Adelaide delegated): those passes are dropped from the
    cliff-against-random count and the number dropped is recorded
    ('passes_left_out_random_arm_smaller' per cleanup and setting,
    'cliff_over_random_passes_left_out' in the summary). The missing
    molecules are not drawn from the cliff set.
    """
    out = np.zeros(len(cliff_train), dtype=bool)
    pool = np.where(~cliff_train)[0]
    k = int(min(n_exempt, len(pool)))
    if k <= 0:
        return out, 0
    rng = np.random.default_rng(RANDOM_EXEMPTION_SEED + 1009 * fold_i + seed_i)
    out[rng.choice(pool, size=k, replace=False)] = True
    return out, k


def put_back(cleaned, exempt, X_tr_s, y_tr):
    """The cleanup's output with the exempt molecules restored.

    `cleaned` is what n07.cleaned_training_set returned: X, y, sample_weight,
    keep, label_column. Restoring means, for the three shapes a cleanup can
    take:

      a deleted row     put back into `keep`
      a lowered weight  set to 1, and the whole weight column renormalised to
                        mean 1 afterwards, which multiplies every weight by one
                        constant and so leaves the ratios among the molecules
                        that were NOT put back exactly as the cleanup set them
      a rewritten label restored to the value the training column carried

    Nothing else about the cleanup changes: the threshold it used, the scores it
    read and the target it aimed at were all fixed before this is called.
    """
    # The labels keep their own type: a class label is not cast to a float
    # here (2026-10-05, chat 5), so a classifier is fitted on the classes the
    # loader produced.
    y_tr = np.asarray(y_tr)
    exempt = np.asarray(exempt, dtype=bool)

    keep = cleaned.get('keep')
    new_keep = None if keep is None else (np.asarray(keep, dtype=bool) | exempt)

    col = cleaned.get('label_column')
    new_y = y_tr.copy() if col is None else np.array(col, copy=True)
    new_y[exempt] = y_tr[exempt]

    w = cleaned.get('sample_weight')
    new_w = None
    if w is not None:
        new_w = np.asarray(w, dtype=float).copy()
        new_w[exempt] = 1.0
        mean_w = float(new_w.mean())
        if np.isfinite(mean_w) and mean_w > 0:
            new_w = new_w / mean_w

    if new_keep is None:
        return {'X': X_tr_s, 'y': new_y, 'sample_weight': new_w,
                'keep': None, 'label_column': new_y}
    return {'X': X_tr_s[new_keep], 'y': new_y[new_keep],
            'sample_weight': None if new_w is None else new_w[new_keep],
            'keep': new_keep, 'label_column': new_y}


def n_spared(cleaned, restored, exempt, y_tr):
    """How many of the exempt molecules the cleanup would actually have touched.

    Counted as the molecules whose presence, weight or label differs between the
    two arms, so it means the same thing for a cleanup that deletes rows, one
    that lowers weights and one that rewrites labels. Zero means the two arms
    are the same training set and the difference between them is zero by
    construction, not by measurement.
    """
    exempt = np.asarray(exempt, dtype=bool)
    if not exempt.any():
        return 0
    touched = np.zeros(len(exempt), dtype=bool)
    keep = cleaned.get('keep')
    if keep is not None:
        touched |= ~np.asarray(keep, dtype=bool)
    w = cleaned.get('sample_weight')
    if w is not None:
        rw = restored.get('sample_weight')
        touched |= ~np.isclose(np.asarray(w, dtype=float),
                               np.asarray(rw, dtype=float))
    col = cleaned.get('label_column')
    if col is not None:
        try:
            touched |= ~np.isclose(np.asarray(col, dtype=float),
                                   np.asarray(y_tr, dtype=float))
        except (TypeError, ValueError):
            # A class label that is not a number.
            touched |= np.asarray(col) != np.asarray(y_tr)
    return int((touched & exempt).sum())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _fit_score(cleaned, X_va_s, y_val, model_factory, metric_fn, metric_name,
               label=''):
    """Fit one arm's training set and score it on the held-out molecules."""
    try:
        m = model_factory()
        if cleaned['sample_weight'] is None:
            m.fit(cleaned['X'], cleaned['y'])
        else:
            m.fit(cleaned['X'], cleaned['y'],
                  sample_weight=cleaned['sample_weight'])
        return float(metric_fn(y_val, _predict_for_metric(m, X_va_s,
                                                          metric_name)))
    except Exception as exc:
        print(f"      {label} fit failed: {type(exc).__name__}: {exc}")
        return float('nan')


def _gain(treatment, reference, lower_is_better):
    """Signed difference, positive when `treatment` is the better score.

    Every comparison in this file goes through here, so the metric's direction
    is read once per number rather than assumed: herg_ki, caco2 and herg_ic50
    are rmse and lower wins; flab_thermo_tresanco is spearman and higher wins.
    """
    if not (np.isfinite(treatment) and np.isfinite(reference)):
        return float('nan')
    return ((reference - treatment) if lower_is_better
            else (treatment - reference))


def _cliff_suspicion_counts(scores, corrupt_mask, cliff_train):
    """Do the cliff molecules look wrong to this detector? Counts, not ratios.

    Of the molecules the detector ranks most suspicious -- as many of them as
    the draw actually corrupted -- how many are cliff molecules whose label
    never moved. Those are the false positives this exemption exists to stop.
    """
    sus = np.asarray(scores, dtype=float)
    corrupt = np.asarray(corrupt_mask, dtype=bool)
    cliff = np.asarray(cliff_train, dtype=bool)
    if sus.size != corrupt.size or sus.size != cliff.size:
        return None
    k = int(corrupt.sum())
    if k == 0:
        return None
    finite = np.isfinite(sus)
    if not finite.all():
        sus = np.where(finite, sus, np.nanmin(sus[finite]) - 1.0
                       if finite.any() else 0.0)
    top = np.argsort(-sus, kind='stable')[:k]
    in_top = np.zeros(sus.size, dtype=bool)
    in_top[top] = True
    return {
        'n_training_molecules': int(sus.size),
        'n_corrupted_molecules': k,
        'n_most_suspicious_molecules_taken': k,
        'n_cliff_molecules_in_training': int(cliff.sum()),
        'n_cliff_molecules_uncorrupted': int((cliff & ~corrupt).sum()),
        'n_cliff_molecules_corrupted': int((cliff & corrupt).sum()),
        'n_uncorrupted_cliff_molecules_among_the_most_suspicious':
            int((in_top & cliff & ~corrupt).sum()),
        'n_corrupted_molecules_among_the_most_suspicious':
            int((in_top & corrupt).sum()),
    }




def _mean_detection(readings):
    """Average detection readings over passes (n07's rule, written here so
    n25 does not import a private name for it)."""
    if not readings:
        return None
    keys = ('detection_average_precision', 'detection_precision_at_k',
            'detection_recall_at_k', 'detection_n_corrupted',
            'detection_n_training')
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in readings if r.get(k) is not None
                and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['detection_condition'] = readings[0].get('detection_condition')
    out['detection_n_passes'] = len(readings)
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _write_row(sink, identity, model, score_D, score_Dprime, score_clean,
               metric):
    """One D-versus-D' row (d_vs_dprime.make_row through rows_for)."""
    if sink is None or not sink.enabled:
        return
    sink.write(rows_for({model: {'score_D': score_D,
                                 'score_Dprime': score_Dprime,
                                 'score_clean': score_clean}},
                        metric=metric, **identity))


def _pairs_for_detector(detector, scores, nbr, X_tr_s, X_va_s, y_noisy,
                        y_clean_train, y_val, g_tr, model, factory, targets,
                        cliff_train, rand_train, n_rand, n_train, flag_rates,
                        cleanups, is_clf, metric, task_type, dataset_name, *,
                        arm_cell, spared_cell, sink, row_base,
                        similarity_label, score_D, score_clean, only=None):
    """Every (setting slot, cleanup) of one detector on one configuration of
    one model: the plain cleanup, the cliff exemption and the random
    exemption, each fitted once.

    nbr: (lists, k, setting, scheme) for one setting the screen kept; the
        lists carry the scheme, so a neighbour cleanup weights its
        neighbours by it. None only when the screen kept nothing.
    only: restricts the cleanups (cross_val under a kept setting other than
        the top one runs only the cleanups that read the setting; with no
        kept setting the main counts run without label_smooth and
        knn_smooth).
    arm_cell / spared_cell None: the settings after the top one write rows
        only.
    """
    metric_name, metric_fn, _lower = metric
    for slot, rate in enumerate(flag_rates):
        rk = str(rate)
        for cleanup in cleanups:
            if only is not None and cleanup not in only:
                continue
            reads = nbr is not None and cleanup_reads_setting(cleanup, nbr[2])
            axis, vals = setting_axis(cleanup, is_clf)
            if slot >= len(vals):
                # Fewer settings than slots: a cleanup with one setting would
                # be the same training set fitted again.
                continue
            strength = None if axis == 'flag_rate' else vals[slot]
            _sim_kw = ({'similarity_neighbours': nbr[0],
                        'similarity_k': int(nbr[1])} if reads else {})
            try:
                plain = cleaned_training_set(
                    cleanup, X_tr_s, y_noisy, scores, rate,
                    is_clf=is_clf, groups=g_tr, strength=strength,
                    targets=targets, dataset_name=dataset_name, **_sim_kw)
            except Exception as exc:
                print(f"      {cleanup} declined: {type(exc).__name__}: {exc}")
                plain = None
            if plain is None:
                # `remove` left under ten training molecules, or a target
                # could not be built. The whole pair is dropped, so the
                # comparison is never half-scored.
                continue

            cliff_arm = put_back(plain, cliff_train, X_tr_s, y_noisy)
            rand_arm = put_back(plain, rand_train, X_tr_s, y_noisy)
            if spared_cell is not None:
                spared_cell(rk, cleanup).update({
                    'n_training_molecules': n_train,
                    'n_cliff_molecules_in_training': int(cliff_train.sum()),
                    'n_cliff_molecules_spared': n_spared(
                        plain, cliff_arm, cliff_train, y_noisy),
                    'n_random_control_molecules': int(n_rand),
                    'n_random_molecules_spared': n_spared(
                        plain, rand_arm, rand_train, y_noisy),
                    'setting_axis': axis,
                    'setting': rate if axis == 'flag_rate' else strength,
                })
            _setting = rate if axis == 'flag_rate' else strength
            ms = f'detector={detector}|{axis}={_setting}'
            if nbr is not None:
                ms += f'|k={int(nbr[1])}'
                # The neighbour detectors read the counting scheme since
                # 2026-10-06, so every cleanup under one is a different
                # training set per scheme.
                if reads or detector in nc.NEIGHBOUR_DETECTORS:
                    ms += f'|scheme={nbr[3]}'
            arm_scores = {}
            for arm, cleaned in (('plain', plain),
                                 ('cliff_exempt', cliff_arm),
                                 ('random_exempt', rand_arm)):
                s = _fit_score(cleaned, X_va_s, y_val, factory, metric_fn,
                               metric_name, label=f'{cleanup}/{arm}/{model}')
                arm_scores[arm] = s
                _write_row(sink, row_identity(
                    method=cleanup, method_setting=f'{ms}|arm={arm}',
                    similarity_setting=similarity_label(cleanup), **row_base),
                    model, score_D, s, score_clean, metric)
            if arm_cell is not None:
                arm_cell(rk, cleanup).update(arm_scores)

            # The shared controls (KIRBY_NOISE_CONTROLS=1 only): twins of the
            # PLAIN cleanup, written by n07.dprime_rows under
            # '<cleanup>__random_twin' / '__shuffled_twin'. n25's own
            # random_exempt stays as it is and always runs.
            if controls_enabled() and sink is not None and sink.enabled:
                dprime_rows(sink, dataset_name=dataset_name, cleaned=plain,
                            X_tr=X_tr_s, y_D=y_noisy, X_va=X_va_s, y_va=y_val,
                            metric=metric, task_type=task_type,
                            y_clean=y_clean_train, models=(model,),
                            is_clf=is_clf, identity=row_identity(
                                method=cleanup,
                                method_setting=f'{ms}|arm=plain',
                                similarity_setting=similarity_label(cleanup),
                                **row_base))


def run(data, config, output_dir, dataset_name, **kwargs):
    """N25 across every kind of damage this file is defined under.

    Each kind is written to its own condition file as it finishes
    (noise_helpers.sweep_conditions). A caller naming one kind
    (noise_condition=, as run_all does under a job's condition) runs that
    one. The condition files' fingerprint holds the levels, the draws, the
    outer fold count, the folds run and RESULT_LAYOUT, so a file written by
    the whole-dataset design (before 2026-10-05) is recomputed, not skipped.
    """
    is_clf = data.get('task_type') == 'classification'
    conditions = default_conditions(is_clf)
    named = kwargs.get('noise_conditions') or (
        [kwargs['noise_condition']] if kwargs.get('noise_condition') else [])
    # Refuse a cut or undefined kind before any condition runs (2026-10-05),
    # so one named kind cannot run at a level grid outside its own unit.
    refused = [c for c in named if not condition_allowed(c, is_clf)]
    if refused:
        raise ValueError(
            f"n25: {refused} not runnable here. Allowed on a "
            f"{'classification' if is_clf else 'regression'} dataset: "
            f"{sorted(CLASSIFICATION_STRATEGIES) if is_clf else list(ALLOWED_REGRESSION_CONDITIONS)}.")
    fp_config = dict(config)
    fp_config['n_splits'] = noise_n_folds(config)
    fp_config['noise_levels'] = {
        c: list(_condition_levels(data, dataset_name, c, kwargs))
        for c in (named or conditions)}
    fp_config['noise_seeds'] = {
        c: list(draws_for(c, kwargs.get('noise_seeds') or NOISE_SEEDS))
        for c in (named or conditions)}
    fp_config['noise_strategies'] = [
        RESULT_LAYOUT,
        f"only_folds={os.environ.get('KIRBY_NOISE_ONLY_FOLDS') or 'all'}"]
    return sweep_conditions(
        _run_one_condition, EXP_FULL_NAME, data, fp_config, output_dir,
        dataset_name, default_conditions=conditions, kwargs=dict(kwargs),
        groups_needed=GROUPED_STRATEGIES)


def _condition_levels(data, dataset_name, condition, kwargs):
    """The levels one kind runs at: the decided ones for this dataset's group
    (noise_helpers.decided_levels), level 0 first, unless --levels names
    others."""
    if kwargs.get('levels'):
        return tuple(float(x) for x in kwargs['levels'])
    return tuple(float(x) for x in decided_levels(data, dataset_name,
                                                  condition))


def _run_one_condition(data, config, output_dir, dataset_name, **kwargs):
    """N25 under ONE kind of damage at every decided level.

    CHAT 5 ALIGNMENT, 2026-10-05 (HANDOFF_EXPERIMENT_ALIGNMENT section 2.4).
    On each fold, in order:
      1. noise_helpers.fold_configurations runs the fold's searches on its
         training rows and gives each decided model its 2 single
         representations and 2 general hybrids, plus the neighbour spaces'
         recipes (distance hybrid, similarity hybrid).
      2. At every (level, draw): the training labels are damaged
         (noise_helpers.inject_condition), the exempt set is built from the
         damaged labels of the training molecules, the hybrids are rebuilt at
         the damaged labels, and every detector, cleanup and arm runs in each
         model's 4 configurations.
      3. The pass is written to the checkpoint as it finishes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT N25: DOES EXEMPTING CLIFF MOLECULES FROM CLEANING HELP")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    groups_arr = None if groups is None else np.asarray(groups)
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    metric = (metric_name, metric_fn, lower_is_better)

    condition = kwargs.get('noise_condition') or default_conditions(is_clf)[0]
    if not condition_allowed(condition, is_clf):
        raise ValueError(
            f"n25: this file turns on a molecule whose label is RIGHT being "
            f"treated as wrong, so some labels must be left as delivered. "
            f"{condition!r} is not one of the kinds that does that on a "
            f"{'classification' if is_clf else 'regression'} dataset: "
            f"{sorted(CLASSIFICATION_STRATEGIES) if is_clf else list(ALLOWED_REGRESSION_CONDITIONS)} "
            f"(regression: detection defined AND kept by the noise grid "
            f"decision).")
    levels = _condition_levels(data, dataset_name, condition, kwargs)
    # DECIDED 2026-10-06 (Adelaide delegated): one draw under censoring,
    # which reads no seed (noise_helpers.draws_for).
    seeds = [int(s) for s in draws_for(
        condition, kwargs.get('noise_seeds') or NOISE_SEEDS)]
    knn_grp = dataset_knn_group(data, dataset_name)

    # R1: the dataset's two decided models (DECISIONS.md Step 1), nothing
    # else. Replaces config['models'] / DOWNSTREAM_MODELS_NOISE (five models)
    # and the rf and svm ensure_dprime_models added to every row. knn runs
    # only inside the distance search.
    models = [m for m in decided_models(dataset_name, 'general') if m != 'knn']
    model_factories = get_model_factories(models, task_type=task_type,
                                          n_samples=len(labels))
    models_without_factory = [m for m in models if m not in model_factories]
    models = [m for m in models if m in model_factories]
    if not models:
        raise RuntimeError(
            f"n25: neither decided model could be built on {dataset_name} "
            f"({models_without_factory}).")

    flag_rates = [float(x) for x in (kwargs.get('flag_rates') or FLAG_RATES)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    CLEANUPS = cleanups_for(is_clf, kwargs.get('cleanups'))
    cliff_kw = dict(
        sim_threshold=float(kwargs.get('cliff_sim_threshold')
                            or CLIFF_SIM_THRESHOLD),
        activity_k=float(kwargs.get('cliff_activity_k') or CLIFF_ACTIVITY_K),
        threshold_mode=str(kwargs.get('cliff_threshold_mode') or 'mad'),
        potency_threshold=float(kwargs.get('cliff_potency_threshold')
                                or CLIFF_POTENCY_THRESHOLD))

    # ── The whole-dataset cliff set, REPORTED ONLY ────────────────────────
    # It used to be the exempt set: built once from the clean label column
    # across every molecule and sliced per fold, which handed the exemption
    # the answer sheet. The per-pass set below is built from the damaged
    # labels of the fold's training molecules. This call stays to report how
    # far the two differ and to build the structure matrix once.
    cliff_mask, cliff_info = cliff_mask_for_dataset(data, labels, is_clf,
                                                    **cliff_kw)
    print(f"  cliff molecules on the CLEAN labels, whole dataset (reported, "
          f"not used for cleaning): {cliff_info['n_cliff_molecules']} of "
          f"{cliff_info['n_molecules']} molecules, structure reference "
          f"{cliff_info.get('structure_reference')}")
    # The structural half never depends on a label, so one matrix serves every
    # fold, level and draw. Popped: it is n x n floats and cliff_info is
    # written to the result file. The antibody path returns before it is set.
    cliff_structure = cliff_info.pop('similarity_matrix', None)
    # How the matrix was cut, so the per-fold calls cut it the same way.
    cliff_structure_info = {
        'structure_reference': cliff_info.get('structure_reference'),
        'similar_percentile': cliff_info.get('similar_percentile'),
        'reference_independent': cliff_info.get('reference_independent',
                                                True)}
    if not cliff_info['reference_independent']:
        print("  the cliff reference is built from the representations "
              "themselves; the exempt set and the suspect set are not "
              "independent on this dataset")

    noise_groups = noise_groups_for(data)[0]
    folds, n_folds = noise_outer_folds(data, config)
    smiles = data.get('smiles')
    has_smiles = smiles is not None and len(smiles) > 0
    n_molecules = len(set(smiles)) if has_smiles else len(labels)
    sim_settings = sweep_settings(dataset_name, n_molecules, has_smiles)

    print(f"  {len(labels)} rows, {n_folds} outer folds "
          f"({len(folds)} run), group {knn_grp}, models (decided) {models}")
    print(f"  condition {condition}, levels {list(levels)} "
          f"({level_unit(condition, is_clf)}), draws {seeds} (level 0 once)")
    print(f"  detectors {detectors}, flag rates {flag_rates}")
    print(f"  cleanups ({len(CLEANUPS)}): {list(CLEANUPS)}")
    print(f"  neighbour settings: {len(sim_settings)} x neighbour counts "
          f"{list(sweep_ks())} x schemes {list(sweep_schemes())}, top "
          f"{screen_top()} refitted")

    results = {
        'layout': RESULT_LAYOUT,
        'metric': metric_name,
        'lower_is_better': bool(lower_is_better),
        'question': (
            'A molecule on an activity cliff carries a correct label that looks '
            'wrong to a detector comparing it with its structural neighbours. '
            'Does holding those molecules out of the cleaning improve the '
            'held-out score?'),
        'design': (
            'Every cleanup is run twice on the same folds, the same draw, the '
            'same detector scores and the same targets: once as it is, once '
            'with the cliff molecules put back. A third arm puts back as many '
            'non-cliff molecules, so that "less cleaning helps" is separated '
            'from "sparing cliffs helps". The headline is the difference '
            'within a pair, per pass.'),
        'arms': list(ARMS),
        'pass_is': 'one (fold, noise draw) at one level',
        'models': models,
        'models_without_factory': models_without_factory,
        'models_rule': ("the dataset's two decided models "
                        "(decided_grid.decided_models, DECISIONS.md Step 1)"),
        'configurations': list(SPACES),
        'configurations_rule': (
            'per decided model and outer fold: its 2 best single '
            'representations and 2 best general hybrids, ranked by searches '
            "on that fold's training rows (noise_helpers.fold_configurations)."
            ' 2 models x 4 = 8 configurations.'),
        'selection_scope': 'fold_train_rows',
        'n_folds': int(n_folds),
        'folds_run': [int(f) for f, _t, _v in folds],
        'knn_group': knn_grp,
        'cleanups': list(CLEANUPS),
        'cleanups_not_run': [c for c in ALL_CLEANUPS if c not in CLEANUPS],
        'cleanups_not_run_because': {
            c: (UNDEFINED_HERE[c] if c in UNDEFINED_HERE else
                'it moves a label to a numeric value and this dataset\'s label '
                'is a class' if (is_clf and c in REGRESSION_ONLY_CLEANUPS) else
                (f'on a class label it builds the same training set as '
                 f'{SAME_AS_ON_A_CLASS_LABEL[c]} at every flag rate')
                if (is_clf and c in SAME_AS_ON_A_CLASS_LABEL) else
                'the run was asked for a shorter list of cleanups')
            for c in ALL_CLEANUPS if c not in CLEANUPS},
        'what_the_cleanup_does': {c: WHAT_THE_CLEANUP_DOES.get(c, 'unknown')
                                  for c in CLEANUPS},
        'flag_rates': flag_rates,
        'swept_axis_per_cleanup': {
            c: {'axis': setting_axis(c, is_clf)[0],
                'values': list(setting_axis(c, is_clf)[1])}
            for c in CLEANUPS},
        'detectors': detectors,
        'noise_condition': condition,
        'noise_levels': list(levels),
        'noise_level_unit': level_unit(condition, is_clf),
        'noise_seeds': seeds,
        'noise_seed_note': (
            'Three draws at every non-zero level, level 0 (the labels as '
            'delivered) once; one draw under censoring, which reads no seed '
            '(noise_helpers.draws_for, 2026-10-06). Both arms of a pair see the same drawn labels, '
            'the same detector scores and the same targets.'),
        'robustness_reported_as': (
            'damaged_minus_clean: the held-out score with no cleaning at the '
            'damaged labels minus the score at the clean labels, in the '
            "dataset's own metric (lower_is_better says which sign is worse). "
            'auc_norm is not computed here; on fast_knn datasets it is absent '
            'by decision (two levels).'),
        'cliff_reference': convert_numpy(cliff_info),
        'cliff_reference_independent': bool(cliff_info['reference_independent']),
        'random_exemption_note': (
            'The control arm puts back as many molecules as the cliff arm does, '
            'drawn from the training molecules the cliff mask did not name, so '
            'the two exempt sets never overlap. Where there are fewer non-cliff '
            'molecules than cliff molecules it puts back fewer; such passes '
            'are counted and left out of the cliff-against-random comparison.'),
        'weights_reach_feature_selection': False,
        'weights_reach_feature_selection_note': (
            'sample_weight appears nowhere in src/kirby/hybrid.py or '
            'src/kirby/find_best_hybrid.py, so a weighting cleanup weights the '
            'downstream fit only.'),
        'hybrid_sees_cleaned_labels': False,
        'hybrid_sees_cleaned_labels_note': (
            "Each model's two hybrids are rebuilt per (fold, level, draw) from "
            "the recipes the fold's search chose on the clean training labels, "
            "on that fold's training molecules at their DAMAGED labels. No "
            'cleanup and no exemption reaches the feature ranking or the '
            'budget split.'),
        'search_labels_note': (
            'OPEN (noise_helpers, chat 5): the per-fold searches see the clean '
            'training labels, as the whole-dataset precompute did.'),
    }

    # ── Rows and the checkpoint ─────────────────────────────────────────────
    # A new rows file per layout: rows the whole-dataset design wrote carry the
    # same method names and would be skipped as done.
    sink = RowSink(output_dir, EXP_FULL_NAME, dataset_name,
                   f'{condition}__{RESULT_LAYOUT}')
    ckpt = FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name, condition=condition,
        tag='units', fingerprint={
            'layout': RESULT_LAYOUT, 'input': input_fingerprint(data, config),
            'condition': condition, 'levels': list(levels), 'seeds': seeds,
            'n_folds': int(n_folds), 'models': models,
            'detectors': detectors, 'flag_rates': flag_rates,
            'cleanups': list(CLEANUPS), 'cliff': cliff_kw,
            'settings': [s.name for s in sim_settings],
            'screen_top': screen_top()})

    n_passes = len(folds) * sum(1 if float(lv) == 0.0 else len(seeds)
                                for lv in levels)
    n_settings = sum(min(len(flag_rates), len(setting_axis(c, is_clf)[1]))
                     for c in CLEANUPS)
    n_default = (n_passes * len(SPACES) * len(models) * len(detectors)
                 * n_settings * len(ARMS))
    results['planned_model_fits_main_counts'] = {
        'passes': n_passes,
        'configurations_per_pass': len(SPACES) * len(models),
        'cleanup_settings_per_configuration_per_detector': n_settings,
        'cleanup_fits': int(n_default),
        'note': ('One pass is one (fold, level, draw). Every cleanup setting '
                 'is fitted once per arm, and there are three arms. The main '
                 'counts use the top screened neighbour setting; the other '
                 'screen_top() - 1 kept settings add rows on top.'),
    }
    print(f"  main counts: {n_default} cleanup fits over {len(ARMS)} arms "
          f"across {n_passes} passes")

    fold_records = {}
    units = []

    for fold_i, train_idx, val_idx in folds:
        y_clean_train = labels[train_idx]
        y_val = labels[val_idx]
        g_tr = None if groups_arr is None else groups_arr[train_idx]
        n_train = int(len(train_idx))
        smi_tr = smiles_for(data, train_idx)
        smi_va = smiles_for(data, val_idx)
        pass_list = [(float(lv), int(d)) for lv in levels
                     for d in (seeds[:1] if float(lv) == 0.0 else seeds)]
        keys = {p: ['unit', int(fold_i), p[0], p[1]] for p in pass_list}
        have = {p: ckpt.get(k) for p, k in keys.items()}
        frec = ckpt.get(['fold_record', int(fold_i)])
        if frec is not None and all(v is not None for v in have.values()):
            print(f"\n  Fold {fold_i}: every pass read back from {ckpt.path}")
            fold_records[str(fold_i)] = frec
            units += [{'fold': int(fold_i), 'level': p[0], 'draw': p[1],
                       'value': v} for p, v in have.items()]
            continue

        print(f"\n  Fold {fold_i} of {n_folds}: {n_train} training, "
              f"{len(val_idx)} held-out rows")
        # n25 runs on the rows as loaded, so the per-fold searches are the
        # dataset's standard ones and no condition_suffix is needed.
        try:
            entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                          train_idx=train_idx,
                                          n_folds=n_folds)
        except Exception as exc:
            print(f"    per-fold searches failed: {type(exc).__name__}: {exc}")
            fold_records[str(fold_i)] = {
                'error': f'per-fold searches failed: {type(exc).__name__}: '
                         f'{exc}'}
            continue

        # Each model's 4 configurations at the CLEAN labels. The single
        # representations are the same matrices at every level; the hybrids
        # are rebuilt per pass at the damaged labels below.
        clean_inputs = {}
        frec = {'n_train': n_train, 'n_val': int(len(val_idx)), 'models': {}}
        for m in models:
            entry = entries.get(m)
            ins, unav, names = fold_model_inputs(
                data, dataset_name, m, entry, train_idx=train_idx,
                val_idx=val_idx, y_train=y_clean_train,
                factory=model_factories[m], metric=metric)
            clean_inputs[m] = (entry, ins, unav, names)
            rec = configuration_record(entry, names)
            rec['configurations_unavailable'] = {
                sp: why for sp, why in unav.items() if sp in SPACES}
            rec['n_features'] = {sp: int(np.shape(X[0])[1])
                                 for sp, X in ins.items()}
            rec['metric'] = metric_name
            rec['lower_is_better'] = bool(lower_is_better)
            frec['models'][m] = rec
            print(f"    {m}: base {rec.get('base')}, hybrids "
                  f"{[h.get('label') for h in rec.get('hybrids') or []]}")
        ckpt.put(['fold_record', int(fold_i)], frec)
        fold_records[str(fold_i)] = frec

        def _unit(level, draw, _tr=train_idx, _va=val_idx,
                  _ytr=y_clean_train, _yva=y_val, _g=g_tr, _n=n_train,
                  _ci=clean_inputs, _fold=fold_i, _smi_tr=smi_tr,
                  _smi_va=smi_va):
            y_noisy, cmask, prov = inject_condition(
                data, dataset_name, condition, level, draw, train_idx=_tr,
                groups=noise_groups)
            cmask = np.asarray(cmask, dtype=bool)

            # THE EXEMPT SET, from the labels as they LOOK and over the
            # training molecules alone (inject_condition returns the training
            # rows' labels; `subset` says which molecules they belong to).
            cliff_train, _pci = cliff_mask_for_dataset(
                data, np.asarray(y_noisy), is_clf, subset=_tr,
                structure=cliff_structure,
                structure_info=cliff_structure_info, **cliff_kw)
            n_cliff = int(cliff_train.sum())
            rand_train, n_rand = random_exempt_mask(cliff_train, n_cliff,
                                                    int(_fold), int(draw))
            val = {
                'provenance': convert_numpy(prov),
                'cliff_counts': {
                    'n_training_molecules': _n,
                    'n_cliff_molecules_in_training': n_cliff,
                    'n_random_control_molecules': int(n_rand),
                    'random_arm_short_by': int(n_cliff - n_rand),
                    'n_own_repeat_pairs_dropped': int(
                        _pci.get('n_own_repeat_pairs_dropped', 0) or 0),
                    'n_corrupted_molecules': int(cmask.sum()),
                    'n_cliff_molecules_corrupted':
                        int((cliff_train & cmask).sum()),
                    # What the OLD code would have exempted here: the clean
                    # column's mask sliced to this fold.
                    'n_cliff_on_the_clean_labels':
                        int(np.asarray(cliff_mask)[_tr].sum()),
                },
                'per_model': {}, 'screen': [], 'sim_unavailable': {},
            }
            row_level = float(level)
            for m, (entry, ins_c, unav_c, _names) in _ci.items():
                factory = model_factories[m]
                hyb_n, unav_n, _hn = fold_model_inputs(
                    data, dataset_name, m, entry, train_idx=_tr, val_idx=_va,
                    y_train=y_noisy, factory=factory, metric=metric,
                    spaces=HYBRID_CONFIGURATIONS)
                pm = val['per_model'].setdefault(m, {'configurations': {},
                                                     'unavailable': {},
                                                     'sim_detection': {}})
                for sp in SPACES:
                    is_hyb = sp in HYBRID_CONFIGURATIONS
                    pair = hyb_n.get(sp) if is_hyb else ins_c.get(sp)
                    if pair is None:
                        pm['unavailable'][sp] = (
                            (unav_n if is_hyb else unav_c).get(sp)
                            or 'not built')
                        continue
                    X_tr_s, X_va_s = pair
                    cell = {'n_features': int(np.shape(X_tr_s)[1]),
                            'detection': {}, 'premise': {}, 'arms': {},
                            'spared': {}}
                    pm['configurations'][sp] = cell
                    mi = f'{m}:{sp}'
                    row_base = dict(dataset_name=dataset_name,
                                    experiment=EXP_FULL_NAME, model_input=mi,
                                    noise_condition=condition,
                                    level=row_level, draw=int(draw),
                                    fold=int(_fold))

                    # The model-based corrections' targets, once per (pass,
                    # configuration): they depend on the features, the labels
                    # and the groups and nothing else.
                    _want = targets_needed(CLEANUPS)
                    _targets = (correction_targets_cached(
                        dataset_name, X_tr_s, y_noisy, _g, _want)
                        if _want else {})

                    _cd = dataset_name if m in CACHEABLE_MODELS else None
                    # The score to beat: the same damaged labels, no cleaning.
                    sc = _evaluate_fold(
                        X_tr_s, X_va_s, y_noisy, _yva, 'noisy_baseline', 0.0,
                        factory, metric_fn, is_clf=is_clf,
                        metric_name=metric_name, dataset_name=dataset_name,
                        precomputed_y_tr=y_noisy, cache_dataset=_cd,
                        cache_model_name=m)
                    cell['no_cleaning'] = (float(sc) if sc is not None
                                           else float('nan'))
                    # The ceiling: the labels before anything was injected.
                    sc = _evaluate_fold(
                        X_tr_s, X_va_s, _ytr, _yva, 'no_noise', 0.0, factory,
                        metric_fn, is_clf=is_clf, metric_name=metric_name,
                        dataset_name=dataset_name, cache_dataset=_cd,
                        cache_model_name=m)
                    cell['clean_labels'] = (float(sc) if sc is not None
                                            else float('nan'))

                    # ── The neighbour spaces, their lists and the screen ──
                    # DECIDED 2026-10-06 (Adelaide delegated): the main
                    # counts (per_level) use this configuration's top
                    # screened neighbour setting, kept[0]: its neighbour
                    # space and distance, neighbour count and counting
                    # scheme. The neighbour detectors and label_smooth /
                    # knn_smooth read those lists. Until 2026-10-06 the main
                    # counts came from 10 (detectors) or 9 (cleanups)
                    # Euclidean neighbours in the configuration's own
                    # features, each counting equally. The screen predicts a
                    # held-back fifth of the training molecules (inner=).
                    kept, nb, sim_scores = [], {}, {}
                    if sim_settings:
                        sp_tr, sp_va, unav_s = fold_similarity_spaces(
                            dataset_name, sim_settings, model_input_tr=X_tr_s,
                            model_input_va=X_va_s, reps=data['reps'],
                            train_idx=_tr, val_idx=_va, y_train=y_noisy,
                            groups_train=_g,
                            base_reps=(entry or {}).get('base') or [],
                            recipes=(entry or {}).get('recipes') or {},
                            augmentations=data.get('augmentations'),
                            model_name=m, model_factory=factory,
                            metric=metric, task_type=task_type)
                        # One list per setting, scaffold-mates included
                        # (axis 6).
                        nb = fold_neighbours(dataset_name, sim_settings,
                                             sp_tr, sp_va, y_noisy,
                                             smiles_train=_smi_tr,
                                             smiles_val=_smi_va, is_clf=is_clf)
                        for s in sim_settings:
                            L = nb.get(s.name)
                            if L is None or not L.get('available'):
                                val['sim_unavailable'].setdefault(
                                    s.name, {})[mi] = (
                                        unav_s.get(s.space)
                                        or (L or {}).get('reason'))

                    def _nbr_scores(d, s, k, sch, _cache=sim_scores,
                                    _X=X_tr_s, _nb=nb):
                        """A neighbour detector's scores in one (setting,
                        k, counting scheme); the detectors read the scheme
                        since 2026-10-06."""
                        key = (d, s.name, int(k), sch)
                        if key not in _cache:
                            try:
                                _cache[key] = nc.cached_detector_scores(
                                    dataset_name, _X, y_noisy, is_clf=is_clf,
                                    detector=d, groups=_g,
                                    similarity_neighbours=dict(
                                        _nb[s.name], scheme=sch),
                                    similarity_k=int(k))
                            except Exception as exc:
                                print(f"      {d} under {s.name} k={k} "
                                      f"{sch} refused: "
                                      f"{type(exc).__name__}: {exc}")
                                _cache[key] = None
                        return _cache[key]

                    if sim_settings:
                        # Grading a neighbour detector fits no downstream
                        # model, so every (setting, neighbour count, counting
                        # scheme) is graded.
                        if cmask.any():
                            for s in sim_settings:
                                L = nb.get(s.name)
                                for k in usable_ks(L):
                                    for sch in sweep_schemes():
                                        for d in detectors:
                                            if d not in nc.NEIGHBOUR_DETECTORS:
                                                continue
                                            sc_k = _nbr_scores(d, s, k, sch)
                                            if sc_k is None:
                                                continue
                                            det = detection_scores(
                                                cmask, sc_k, condition)
                                            if det is not None:
                                                (pm['sim_detection']
                                                 .setdefault(sp, {})
                                                 .setdefault(d, {}))[
                                                     f'{s.name}|k={k}|{sch}'
                                                 ] = det
                        # The screen: every (setting, k, scheme) scored with
                        # no model fit; the best screen_top() are refitted.
                        kept, table = screen_settings(
                            sim_settings, nb, y_noisy, metric=metric,
                            is_clf=is_clf, y_val=_yva, inner={
                                'dataset_name': dataset_name,
                                'spaces_train': sp_tr,
                                'y_train': y_noisy,
                                'groups_train': _g,
                                'smiles_train': _smi_tr,
                                'is_clf': is_clf})
                        val['screen'].append({
                            'model_input': mi, 'fold': int(_fold),
                            'draw': int(draw), 'level': row_level,
                            'kept': [f'{s_.name}|k={k_}|{sch}'
                                     for s_, k_, sch in kept],
                            'table': table})

                    # ── The main counts: kept[0] ──
                    main = kept[0] if kept else None
                    if main is not None:
                        _s0, _k0, _sch0 = main
                        main_nbr = (dict(nb[_s0.name], scheme=_sch0),
                                    int(_k0), _s0, _sch0)
                        cell['main_neighbour_setting'] = {
                            'setting': _s0.name, 'space': _s0.space,
                            'k': int(_k0), 'scheme': _sch0}
                        main_only = None
                    else:
                        main_nbr = None
                        _why = ('no neighbour settings for this dataset'
                                if not sim_settings else
                                'the screen kept no (setting, k, scheme)')
                        cell['main_neighbour_setting'] = {
                            'setting': None, 'reason': _why,
                            'not_run': sorted(
                                [d for d in detectors
                                 if d in nc.NEIGHBOUR_DETECTORS]
                                + [c for c in CLEANUPS
                                   if c in ('label_smooth', 'knn_smooth')])}
                        main_only = [c for c in CLEANUPS
                                     if c not in ('label_smooth',
                                                  'knn_smooth')]

                    today_scores = {}
                    for detector in detectors:
                        if detector in nc.NEIGHBOUR_DETECTORS:
                            if main is None:
                                continue
                            scores = _nbr_scores(detector, _s0, _k0, _sch0)
                        else:
                            try:
                                scores = nc.cached_detector_scores(
                                    dataset_name, X_tr_s, y_noisy,
                                    is_clf=is_clf, detector=detector,
                                    groups=_g)
                            except Exception as exc:
                                print(f"      {detector} refused: "
                                      f"{type(exc).__name__}: {exc}")
                                scores = None
                            if scores is not None:
                                today_scores[detector] = scores
                        if scores is None:
                            continue
                        scores = np.asarray(scores, dtype=float)
                        det = detection_scores(cmask, scores, condition)
                        if det is not None:
                            cell['detection'][detector] = det
                        pr = _cliff_suspicion_counts(scores, cmask,
                                                     cliff_train)
                        if pr is not None:
                            cell['premise'][detector] = pr

                        if main is not None:
                            _label = (lambda c, _d=detector, _s=_s0: _s.name
                                      if (_d in nc.NEIGHBOUR_DETECTORS
                                          or cleanup_reads_setting(c, _s))
                                      else (TODAY_SIMILARITY
                                            if c in TODAY_NEIGHBOUR_CLEANUPS
                                            else NO_NEIGHBOURS))
                        else:
                            _label = (lambda c: TODAY_SIMILARITY
                                      if c in TODAY_NEIGHBOUR_CLEANUPS
                                      else NO_NEIGHBOURS)
                        _pairs_for_detector(
                            detector, scores, main_nbr, X_tr_s, X_va_s,
                            y_noisy, _ytr, _yva, _g, m, factory, _targets,
                            cliff_train, rand_train, n_rand, _n, flag_rates,
                            CLEANUPS, is_clf, metric, task_type, dataset_name,
                            arm_cell=lambda rk, c, _d=detector: (
                                cell['arms'].setdefault(_d, {})
                                .setdefault(rk, {}).setdefault(c, {})),
                            spared_cell=lambda rk, c, _d=detector: (
                                cell['spared'].setdefault(_d, {})
                                .setdefault(rk, {}).setdefault(c, {})),
                            sink=sink, row_base=row_base,
                            similarity_label=_label,
                            score_D=cell['no_cleaning'],
                            score_clean=cell['clean_labels'], only=main_only)

                    # ── The other kept settings, rows only ──
                    for s, k, scheme in kept[1:]:
                        L = dict(nb[s.name], scheme=scheme)
                        reading = [c for c in CLEANUPS
                                   if cleanup_reads_setting(c, s)]
                        for detector in detectors:
                            if detector in nc.NEIGHBOUR_DETECTORS:
                                # The detector reads the lists and the
                                # scheme, so every cleanup runs.
                                sc_k = _nbr_scores(detector, s, k, scheme)
                                only = None
                            else:
                                sc_k = today_scores.get(detector)
                                only = reading
                            if sc_k is None or (only is not None
                                                and not only):
                                continue
                            _pairs_for_detector(
                                detector, np.asarray(sc_k, dtype=float),
                                (L, k, s, scheme), X_tr_s, X_va_s, y_noisy,
                                _ytr, _yva, _g, m, factory, _targets,
                                cliff_train, rand_train, n_rand, _n,
                                flag_rates, CLEANUPS, is_clf, metric,
                                task_type, dataset_name, arm_cell=None,
                                spared_cell=None, sink=sink,
                                row_base=row_base,
                                similarity_label=lambda c, _s=s.name: _s,
                                score_D=cell['no_cleaning'],
                                score_clean=cell['clean_labels'], only=only)
            return val

        t0 = time.time()
        for p in pass_list:
            v = have.get(p)
            if v is None:
                print(f"    level {p[0]:g}, draw {p[1]}")
                v = convert_numpy(_unit(p[0], p[1]))
                ckpt.put(keys[p], v)
                print(f"    level {p[0]:g}, draw {p[1]} done "
                      f"[{time.time() - t0:.0f}s]")
            else:
                print(f"    level {p[0]:g}, draw {p[1]}: read back from "
                      f"{ckpt.path}")
            units.append({'fold': int(fold_i), 'level': p[0], 'draw': p[1],
                          'value': v})

    results['fold_records'] = fold_records
    results['per_pass_counts'] = [
        dict(u['value'].get('cliff_counts') or {}, fold=u['fold'],
             level=u['level'], draw=u['draw']) for u in units]
    results['noise_provenance'] = [
        dict(u['value'].get('provenance') or {}, fold=u['fold'])
        for u in units]
    results['per_level'] = {}
    for lv in levels:
        lv_units = [u for u in units if u['level'] == float(lv)]
        block = _level_block(lv_units, models, flag_rates, detectors, CLEANUPS,
                             is_clf, lower_is_better)
        block['summary'] = _summarise(block, models, flag_rates, detectors,
                                      metric_name, CLEANUPS)
        results['per_level'][f'{float(lv):g}'] = block
    results['similarity_sweep'] = similarity_record(
        sim_settings, sink,
        screen_top=screen_top(), screen_scored_on=SCREEN_SCORED_ON,
        schemes_swept=list(sweep_schemes()),
        what_is_swept=(
            "Per configuration: every setting of the dataset's group x "
            'neighbour count x counting scheme is screened with no model fit '
            'on a held-back fifth of the training molecules, and the top '
            'screen_top() are refitted with all three arms. The top one '
            '(kept[0]) gives the per_level counts; the others write rows '
            'only. A neighbour detector is recomputed in the setting and '
            'scheme and drives every cleanup; cross_val keeps its own scores '
            'and, after the top setting, drives only the cleanups that read '
            "the neighbours. The cliff set is not swept: it is n01's "
            'structure reference.'),
        screen=[s for u in units for s in (u['value'].get('screen') or [])],
        spaces_unavailable={
            k: v for u in units
            for k, v in (u['value'].get('sim_unavailable') or {}).items()})
    results['controls_run'] = bool(controls_enabled())
    _print_summary(results, metric_name)
    print(f"  D-versus-D' rows: {sink.n_written} written this run to "
          f"{sink.path or 'memory'}")
    return convert_numpy(results)


def _level_block(lv_units, models, flag_rates, detectors, cleanups, is_clf,
                 lower_is_better):
    """Everything one level reports: per configuration, per detector, per
    flag-rate slot, per model, per cleanup, the paired comparisons over the
    passes at this level; and per configuration and model the no-cleaning
    score, the clean-label ceiling and their difference, each as a spread
    over folds (fold_spread: a fold's number is the mean over its draws)."""
    def _fold_of(pk):
        return int(pk.split('|')[0][1:])

    def _by_fold(d):
        out = {}
        for pk, v in d.items():
            if v is None or not np.isfinite(v):
                continue
            out.setdefault(_fold_of(pk), []).append(float(v))
        return fold_spread([float(np.mean(v)) for _f, v in sorted(out.items())])

    short = {f"f{u['fold']}|d{u['draw']}":
             int((u['value'].get('cliff_counts') or {})
                 .get('random_arm_short_by', 0) or 0) for u in lv_units}
    block = {'n_passes': len(lv_units),
             'passes_where_the_random_arm_was_smaller':
                 sorted(pk for pk, n in short.items() if n > 0),
             'per_space': {}}
    for space in SPACES:
        sb = {'per_model': {}, 'per_detector': {}}
        for m in models:
            noisy, clean, diff = {}, {}, {}
            for u in lv_units:
                pk = f"f{u['fold']}|d{u['draw']}"
                c = ((u['value'].get('per_model') or {}).get(m) or {}) \
                    .get('configurations', {}).get(space)
                if not c:
                    continue
                noisy[pk] = c.get('no_cleaning')
                clean[pk] = c.get('clean_labels')
                if (noisy[pk] is not None and clean[pk] is not None):
                    diff[pk] = float(noisy[pk]) - float(clean[pk])
            sb['per_model'][m] = {
                'no_cleaning': _by_fold(noisy),
                'clean_labels': _by_fold(clean),
                'damaged_minus_clean': _by_fold(diff),
            }
        for detector in detectors:
            by_rate = {}
            for rate in flag_rates:
                rk = str(rate)
                blk = {'per_model': {}}
                for m in models:
                    row = {}
                    for cleanup in cleanups:
                        cells, sp_ = {}, {}
                        for u in lv_units:
                            pk = f"f{u['fold']}|d{u['draw']}"
                            c = ((u['value'].get('per_model') or {}).get(m)
                                 or {}).get('configurations', {}).get(space)
                            if not c:
                                continue
                            a = ((c.get('arms') or {}).get(detector) or {}) \
                                .get(rk, {}).get(cleanup)
                            if a:
                                cells[pk] = a
                            s_ = ((c.get('spared') or {}).get(detector) or {}) \
                                .get(rk, {}).get(cleanup)
                            if s_:
                                sp_[pk] = s_
                        keys = sorted(cells)
                        if not keys:
                            continue
                        axis, vals_ = setting_axis(cleanup, is_clf)
                        slot = flag_rates.index(rate)
                        entry = {
                            'swept_axis': axis,
                            'setting': (rate if axis == 'flag_rate'
                                        else (vals_[slot] if slot < len(vals_)
                                              else None)),
                            'what_the_cleanup_does':
                                WHAT_THE_CLEANUP_DOES.get(cleanup, 'unknown'),
                            'plain_mean': float(np.nanmean(
                                [cells[k].get('plain', float('nan'))
                                 for k in keys])),
                            'molecules_spared_per_pass': {k: sp_.get(k)
                                                          for k in keys},
                        }
                        # Two paired comparisons per cleanup, each signed by
                        # _gain, which reads lower_is_better.
                        for arm in ('cliff_exempt', 'random_exempt'):
                            gains = {k: _gain(cells[k].get(arm, float('nan')),
                                              cells[k].get('plain',
                                                           float('nan')),
                                              lower_is_better) for k in keys}
                            g = np.array([x for x in gains.values()
                                          if np.isfinite(x)], dtype=float)
                            spared_key = ('n_cliff_molecules_spared'
                                          if arm == 'cliff_exempt'
                                          else 'n_random_molecules_spared')
                            n_sp = [int((sp_.get(k) or {}).get(spared_key, 0))
                                    for k in keys]
                            entry[arm] = {
                                'passes_won': int((g > 0).sum()),
                                'passes_scored': int(g.size),
                                'mean_gain_over_plain': (
                                    float(g.mean()) if g.size
                                    else float('nan')),
                                'pass_gains': {k: v for k, v in gains.items()},
                                'gain_over_folds': _by_fold(gains),
                                'arm_mean': float(np.nanmean(
                                    [cells[k].get(arm, float('nan'))
                                     for k in keys])),
                                'molecules_put_back_per_pass': n_sp,
                                'nothing_was_spared': bool(
                                    n_sp and max(n_sp) == 0),
                            }
                        # The one number the third arm exists for: the cliff
                        # exemption against the random one, pass by pass, on
                        # the passes where both put back the same number.
                        paired = {k: _gain(cells[k].get('cliff_exempt',
                                                        float('nan')),
                                           cells[k].get('random_exempt',
                                                        float('nan')),
                                           lower_is_better)
                                  for k in keys if short.get(k, 0) == 0}
                        pg = np.array([x for x in paired.values()
                                       if np.isfinite(x)], dtype=float)
                        entry['cliff_exempt_over_random_exempt'] = {
                            'passes_won': int((pg > 0).sum()),
                            'passes_scored': int(pg.size),
                            'passes_left_out_random_arm_smaller': int(sum(
                                1 for k in keys if short.get(k, 0) > 0)),
                            'mean_gain': (float(pg.mean()) if pg.size
                                          else float('nan')),
                            'pass_gains': paired,
                            'gain_over_folds': _by_fold(paired),
                        }
                        row[cleanup] = entry
                    blk['per_model'][m] = row
                by_rate[rk] = blk
            det_by_model, prem_by_model, sim_det = {}, {}, {}
            for m in models:
                dets, prems = [], {}
                for u in lv_units:
                    pmv = (u['value'].get('per_model') or {}).get(m) or {}
                    c = pmv.get('configurations', {}).get(space) or {}
                    d = (c.get('detection') or {}).get(detector)
                    if d:
                        dets.append(d)
                    pr = (c.get('premise') or {}).get(detector)
                    if pr:
                        prems[f"f{u['fold']}|d{u['draw']}"] = pr
                    for lab, dd in ((pmv.get('sim_detection') or {})
                                    .get(space, {}).get(detector, {})).items():
                        sim_det.setdefault(m, {}).setdefault(lab, []).append(dd)
                det_by_model[m] = _mean_detection(dets)
                prem_by_model[m] = prems
            sb['per_detector'][detector] = {
                'per_setting_slot': by_rate,
                'detection_by_model': det_by_model,
                'cliff_molecules_among_the_suspicious_by_model': prem_by_model,
                'detection_by_neighbour_setting': {
                    m: {lab: _mean_detection(rs) for lab, rs in by_l.items()}
                    for m, by_l in sim_det.items()},
            }
        block['per_space'][space] = sb
    return block


def _summarise(block, models, flag_rates, detectors, metric_name, cleanups):
    """How many settings each exemption won a majority of passes in, per
    configuration, at one level.

    One setting is one (detector, setting slot, model). Every count is taken
    inside ONE configuration: a setting that won in a single representation
    and lost in the hybrid is two readings, not an average.
    """
    out = {'metric': metric_name,
           'headline': 'passes won by the exempt arm over the plain arm',
           'pass_is': 'one (fold, noise draw) at this level',
           'spaces': list(SPACES), 'per_space': {}}
    for space in SPACES:
        per_cleanup = {}
        for cleanup in cleanups:
            rows = []
            for d in detectors:
                for r in flag_rates:
                    blk = (block['per_space'].get(space, {})
                           .get('per_detector', {}).get(d, {})
                           .get('per_setting_slot', {}).get(str(r)))
                    if not blk:
                        continue
                    for m in models:
                        cell = blk['per_model'].get(m, {}).get(cleanup)
                        if not cell:
                            continue
                        ce = cell['cliff_exempt']
                        re_ = cell['random_exempt']
                        co = cell['cliff_exempt_over_random_exempt']
                        if ce['passes_scored'] == 0:
                            continue
                        rows.append({
                            'detector': d, 'model': m,
                            'swept_axis': cell['swept_axis'],
                            'setting': cell['setting'],
                            'cliff_passes_won': ce['passes_won'],
                            'passes_scored': ce['passes_scored'],
                            'cliff_mean_gain': ce['mean_gain_over_plain'],
                            'random_passes_won': re_['passes_won'],
                            'random_mean_gain': re_['mean_gain_over_plain'],
                            'nothing_was_spared': ce['nothing_was_spared'],
                            'cliff_over_random_passes_won': co['passes_won'],
                            'cliff_over_random_passes_scored':
                                co['passes_scored'],
                            'cliff_over_random_passes_left_out':
                                co['passes_left_out_random_arm_smaller'],
                            'cliff_over_random_mean_gain': co['mean_gain'],
                        })
            if not rows:
                per_cleanup[cleanup] = {'settings_scored': 0}
                continue
            live = [x for x in rows if not x['nothing_was_spared']]
            majority = [x for x in live
                        if x['cliff_passes_won'] * 2 > x['passes_scored']]
            sweep = [x for x in live
                     if x['cliff_passes_won'] == x['passes_scored']]
            rnd_majority = [x for x in live
                            if x['random_passes_won'] * 2 > x['passes_scored']]
            over_rnd = [x for x in live
                        if x['cliff_over_random_passes_scored']
                        and x['cliff_over_random_passes_won'] * 2
                        > x['cliff_over_random_passes_scored']]
            # The ranking key is a pass count and a mean gain that _gain
            # already signed, so the metric's direction is read where the
            # number was made and not again here.
            best = max(live, key=lambda x: (
                x['cliff_passes_won'] / max(x['passes_scored'], 1),
                x['cliff_mean_gain'])) if live else None
            per_cleanup[cleanup] = {
                'what_the_cleanup_does':
                    WHAT_THE_CLEANUP_DOES.get(cleanup, 'unknown'),
                'settings_scored': len(rows),
                'settings_where_nothing_was_spared': len(rows) - len(live),
                'settings_the_cliff_exemption_won_most_passes_in':
                    len(majority),
                'settings_the_cliff_exemption_won_every_pass_in': len(sweep),
                'settings_the_random_exemption_won_most_passes_in':
                    len(rnd_majority),
                'settings_the_cliff_exemption_beat_the_random_one_in':
                    len(over_rnd),
                # Passes dropped from the cliff-against-random count because
                # the random arm put back fewer molecules, summed over the
                # settings scored (2026-10-06).
                'cliff_over_random_passes_left_out': int(sum(
                    x['cliff_over_random_passes_left_out'] for x in rows)),
                'best_setting': best,
            }
        # The downweight question, in one block: the same counts grouped by
        # what the cleanup does to a molecule it finds suspect.
        by_what = {}
        for cleanup, row in per_cleanup.items():
            fam = WHAT_THE_CLEANUP_DOES.get(cleanup, 'unknown')
            acc = by_what.setdefault(fam, {
                'cleanups': [], 'settings_scored': 0,
                'settings_the_cliff_exemption_won_most_passes_in': 0,
                'settings_the_random_exemption_won_most_passes_in': 0,
                'settings_the_cliff_exemption_beat_the_random_one_in': 0,
                'settings_where_nothing_was_spared': 0})
            acc['cleanups'].append(cleanup)
            for k in ('settings_scored',
                      'settings_the_cliff_exemption_won_most_passes_in',
                      'settings_the_random_exemption_won_most_passes_in',
                      'settings_the_cliff_exemption_beat_the_random_one_in',
                      'settings_where_nothing_was_spared'):
                acc[k] += int(row.get(k, 0) or 0)
        out['per_space'][space] = {
            'per_cleanup': per_cleanup,
            'by_what_the_cleanup_does': by_what,
            'by_what_the_cleanup_does_note': (
                'If the exemption wins about the same share of settings where '
                'the cleanup lowers a weight as where it rewrites a label, the '
                'effect is about WHICH molecules are touched. If it wins only '
                'where labels are rewritten, it is about WHAT is done to them. '
                '⛔ READ IT BESIDE molecules_spared. A cleanup that thresholds '
                '-- remove, knn_smooth -- touches only the flagged share, so an '
                'exemption spares the few cliff molecules that were flagged. A '
                'proportional one -- downweight, huber_weight, label_smooth, '
                'prox_smooth and the other smoothers on a regression label -- '
                'moves or reweights EVERY molecule, so an exemption spares '
                'every cliff molecule in the fold. Measured in the n25 smoke '
                'on 20 exempt molecules: knn_smooth spared 2 of them and '
                'downweight spared all 20. The two families are therefore not '
                'having the same amount undone, and a larger gain for the '
                'proportional family is partly that.'),
        }
    return out


def _print_summary(results, metric_name):
    print("\n" + "=" * 70)
    print(f"SUMMARY — passes the exempt arm won against the same cleanup "
          f"unexempted ({metric_name}), {results['noise_condition']}")
    ci = results['cliff_reference']
    print(f"  {ci['n_cliff_molecules']} of {ci['n_molecules']} molecules are on "
          f"a cliff on the clean labels, by {ci.get('structure_reference')} "
          f"and {ci.get('threshold_mode')}")
    if not results['cliff_reference_independent']:
        print("  ⛔ the cliff reference is built from the representations "
              "themselves on this dataset; the exempt set and the suspect set "
              "are not independent")
    for lv, block in results['per_level'].items():
        s = block['summary']
        print(f"\n  level {lv} ({block['n_passes']} passes)")
        if block['passes_where_the_random_arm_was_smaller']:
            print(f"    random arm smaller than the cliff arm on "
                  f"{len(block['passes_where_the_random_arm_was_smaller'])} "
                  f"passes; those are left out of cliff against random")
        for space, srow in s['per_space'].items():
            print(f"    {space}:")
            for cleanup, row in srow['per_cleanup'].items():
                if not row.get('settings_scored'):
                    print(f"      {cleanup:14s} no setting produced a score")
                    continue
                print(f"      {cleanup:14s} ({row['what_the_cleanup_does']}) "
                      f"cliff exemption won most passes in "
                      f"{row['settings_the_cliff_exemption_won_most_passes_in']}"
                      f" of {row['settings_scored']} settings; the random "
                      f"exemption in "
                      f"{row['settings_the_random_exemption_won_most_passes_in']}"
                      f"; cliff beat random in "
                      f"{row['settings_the_cliff_exemption_beat_the_random_one_in']}")
                if row['settings_where_nothing_was_spared']:
                    print(f"                     "
                          f"{row['settings_where_nothing_was_spared']} of those "
                          f"settings put back no molecule at all, so the two "
                          f"arms were the same training set")
    print("\n  One setting winning on one pass is not a result. Read the counts.")


def main():
    parser = create_experiment_parser(
        'N25: does exempting cliff molecules from cleaning help')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of training molecules '
                             'treated as suspect')
    parser.add_argument('--detectors', type=str, default=None,
                        help='Comma-separated detector names')
    parser.add_argument('--cleanups', type=str, default=None,
                        help="Comma-separated cleanup names. Default: every one "
                             "n07 registers")
    parser.add_argument('--noise-condition', type=str, default=None,
                        choices=sorted(set(ALLOWED_REGRESSION_CONDITIONS)
                                       | set(CLASSIFICATION_STRATEGIES)),
                        help='One kind of damage. Default: every kind in '
                             'default_conditions for the task type')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels. Default: the '
                             "decided levels for the dataset's group")
    parser.add_argument('--noise-seeds', type=str, default=None,
                        help='Comma-separated draw indices at each non-zero '
                             'level. Default 0,1,2')
    parser.add_argument('--cliff-activity-k', type=float,
                        default=CLIFF_ACTIVITY_K,
                        help='How many median absolute deviations of the label '
                             'column two similar molecules must differ by to be '
                             'a cliff pair')
    parser.add_argument('--cliff-sim-threshold', type=float,
                        default=CLIFF_SIM_THRESHOLD,
                        help='ECFP4 Tanimoto at or above which two molecules '
                             'count as structurally similar')
    parser.add_argument('--cliff-threshold-mode', type=str, default='mad',
                        choices=['mad', 'absolute'],
                        help="'mad' is n01's, in median absolute deviations of "
                             "this label column. 'absolute' is n14's fixed gap "
                             "in the label's own units and is only meaningful "
                             "where the label is a log potency")
    parser.add_argument('--cliff-potency-threshold', type=float,
                        default=CLIFF_POTENCY_THRESHOLD,
                        help='The fixed gap, in label units, under '
                             '--cliff-threshold-mode absolute')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    kw = {
        'cliff_activity_k': args.cliff_activity_k,
        'cliff_sim_threshold': args.cliff_sim_threshold,
        'cliff_threshold_mode': args.cliff_threshold_mode,
        'cliff_potency_threshold': args.cliff_potency_threshold,
    }
    if args.noise_condition:
        kw['noise_condition'] = args.noise_condition
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.detectors:
        kw['detectors'] = args.detectors.split(',')
    if args.cleanups:
        kw['cleanups'] = args.cleanups.split(',')
    if args.levels:
        kw['levels'] = [float(x) for x in args.levels.split(',')]
    if args.noise_seeds:
        kw['noise_seeds'] = [int(x) for x in args.noise_seeds.split(',')]
    run(data, config, args.output, args.dataset, **kw)


if __name__ == '__main__':
    main()
