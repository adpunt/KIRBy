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

WHY NOISE IS INJECTED AND WHY THE CONDITION DEFAULTS TO outlier_only_p05.
n18 asks whether cleaning helps on the labels as they come and needs no
injection. This question is not that one: it turns on a molecule whose label is
RIGHT being treated as wrong, so the run has to know which labels are right.
Under the three `outlier_*` conditions every training label moves, so no
molecule has a correct label and the premise of the question does not exist in
the data. Under the three `outlier_only_*` conditions the selected fraction is
widened and every other label is left bit-identical -- a cliff molecule the draw
did not select carries exactly the label the loader produced. That is the
condition this question needs. --noise-condition takes any of
noise_helpers.DETECTION_IS_DEFINED.

WHAT THE RUN KNOWS BECAUSE IT INJECTED. Two counts that need no model fit and
are the premise of the whole file, recorded per (pass, space, detector):
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
draw). The reported number is how many passes the exempt arm beat the plain arm
on, out of the passes both scored, with the mean difference and the spread
beside it and never instead of it.

ONE NOISE DRAW BY DEFAULT, WHERE THE REST OF THE SUITE RUNS THREE. n07, n11,
n12, n13 and n21 average three draws because an unpaired comparison rides on
which molecules the draw hit. Here both arms of every pair see the same drawn
labels, the same detector scores and the same targets, so the draw cancels out
of the difference the headline counts. It does NOT cancel out of the absolute
scores printed beside it, which are one draw and should be read as one.
--noise-seeds runs more and multiplies every model fit by how many.

WHAT THIS CANNOT SAY. Nothing here shows that the exempted molecules are on
real cliffs -- the structure reference says they are structurally similar to a
molecule of very different activity, which is what a cliff and a pair of
mislabelled neighbours both look like. A win says the model did better when
those molecules were left alone.

Usage:
    python n25_cliff_exemption.py --dataset herg_ki --mode full
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    cv_split, get_domain_metric, get_model_factories, DOWNSTREAM_MODELS_NOISE,
    get_best_search_for_downstream, pick_noise_base_rep,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
)
from experiments.dataset_policy import (
    noise_groups_for, reference_noise_level,
)
from experiments.noise.noise_helpers import (
    inject_train_labels, noise_spec_for, detection_scores,
    DETECTION_IS_DEFINED, hybrid_for_fold,
)
# The cleanups, the detector and the two baselines are n07's, called rather
# than copied. `cleaned_training_set` is the shared dispatcher n17 and n18 also
# call; `_compute_noise_scores` is the detector path that RAISES when a flag
# cannot be delivered to an estimator rather than dropping it silently, which
# matters here because activity_cliff_aware names the scaffold groups
# `exclude_groups` and n18's own copy offers only `groups`.
from experiments.noise.n07_remediation import (
    cleaned_training_set, correction_targets, targets_needed,
    _compute_noise_scores, _evaluate_fold, _predict_for_metric,
    CLEANUPS as N07_CLEANUPS,
    REGRESSION_ONLY_CLEANUPS as N07_REGRESSION_ONLY,
)
# The cliff reference, reused rather than redefined. n14 imports the same two
# names from n01 for the same reason.
from experiments.noise.n01_activity_cliff_framework import (
    identify_cliff_pairs, _compute_ecfp4_tanimoto_sim,
    _compute_reaction_tanimoto_sim, _compute_dta_sim, _compute_euclidean_sim,
    SIMILAR_PERCENTILE,
)


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


def cleanups_for(is_clf, asked=None):
    """The cleanups this task type can actually run."""
    out = (tuple(asked) if asked
           else tuple(c for c in ALL_CLEANUPS if c not in UNDEFINED_HERE))
    if not is_clf:
        return out
    return tuple(c for c in out if c not in REGRESSION_ONLY_CLEANUPS)


def reads_flag_rate(cleanup, is_clf):
    """Does this cleanup's cleaned training set depend on the flag rate?"""
    if cleanup in THRESHOLDING_CLEANUPS:
        return True
    if cleanup in ('downweight', 'huber_weight', 'js_shrink',
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

# The condition. See the module docstring: the unselected molecules' labels are
# bit-identical under this one, so a cliff molecule the draw did not select
# carries exactly the label the loader produced.
NOISE_CONDITION = 'outlier_only_p05'

# One draw. See the module docstring for why, and --noise-seeds for more.
NOISE_SEEDS = (0,)

SPACES = ('base', 'hybrid', 'hybrid_similarity')
POOL_SUBSET = 'predictive_and_similarity'

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
                           subset=None, structure=None):
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

    sim = structure
    if sim is not None:
        info['structure_reference'] = 'reused from a previous call'
        info['similar_percentile'] = SIMILAR_PERCENTILE
        similar_percentile = SIMILAR_PERCENTILE
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
    y_tr = np.asarray(y_tr, dtype=float)
    exempt = np.asarray(exempt, dtype=bool)

    keep = cleaned.get('keep')
    new_keep = None if keep is None else (np.asarray(keep, dtype=bool) | exempt)

    col = cleaned.get('label_column')
    new_y = y_tr.copy() if col is None else np.asarray(col, dtype=float).copy()
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
        touched |= ~np.isclose(np.asarray(col, dtype=float),
                               np.asarray(y_tr, dtype=float))
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


def _drop_blueprint(obj):
    """Strip the pickled Blueprint out of the top-2 blocks before they are saved.

    Same reason as n18's: every recipe field on it is already a key of the entry
    it hangs off, and it carries one list of feature indices per fold per donor
    representation.
    """
    if isinstance(obj, dict):
        return {k: _drop_blueprint(v) for k, v in obj.items() if k != 'blueprint'}
    if isinstance(obj, list):
        return [_drop_blueprint(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(data, config, output_dir, dataset_name, **kwargs):
    """N25: exempting the cliff molecules from cleaning."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N25: DOES EXEMPTING CLIFF MOLECULES FROM CLEANING HELP")
    print("=" * 70)

    labels = np.asarray(data['labels'])
    reps = data['reps']
    groups = data.get('groups')
    augmentations = data.get('augmentations')
    task_type = data['task_type']
    is_clf = task_type == 'classification'
    metric_name, metric_fn, lower_is_better = get_domain_metric(data)
    n_folds = config['n_splits']
    mode = 'intermediary' if n_folds == 1 else 'full'

    condition = kwargs.get('noise_condition') or NOISE_CONDITION
    if condition not in DETECTION_IS_DEFINED:
        raise ValueError(
            f"n25: this file turns on a molecule whose label is RIGHT being "
            f"treated as wrong, so it has to know which labels are right. Under "
            f"{condition!r} every training label moves and no molecule has one. "
            f"The conditions where some molecules are corrupted and some are "
            f"not are {list(DETECTION_IS_DEFINED)}.")

    # ⛔ config FIRST, then the kwarg. run_all puts the caller's model list in
    # config['models'] and forwards no `models` kwarg, so a file that reads
    # kwargs alone silently drops the list the runner was asked for. An explicit
    # kwarg still wins, which is what a smoke or a direct caller passes.
    downstream_models = list(
        kwargs.get('models') or config.get('models') or DOWNSTREAM_MODELS_NOISE)
    model_factories = get_model_factories(
        downstream_models, task_type=task_type, n_samples=len(labels))
    downstream_models = [m for m in downstream_models if m in model_factories]
    if not downstream_models:
        raise RuntimeError(
            f"n25: no downstream model could be built. Asked for "
            f"{list(kwargs.get('models') or DOWNSTREAM_MODELS_NOISE)}, got "
            f"{sorted(model_factories)}.")

    flag_rates = [float(x) for x in (kwargs.get('flag_rates') or FLAG_RATES)]
    detectors = list(kwargs.get('detectors') or DETECTORS)
    seeds = [int(s) for s in (kwargs.get('noise_seeds') or NOISE_SEEDS)]
    levels = [float(x) for x in (kwargs.get('levels')
                                 or [reference_noise_level(dataset_name)])]
    CLEANUPS = cleanups_for(is_clf, kwargs.get('cleanups'))
    base_rep_name = (kwargs.get('base_rep')
                     or pick_noise_base_rep(reps, dataset_name))
    X_source = reps[base_rep_name]

    # ── The exempt set is built PER FOLD, from what the method can see ─────
    # It used to be built once here, from the clean label column, across every
    # molecule in the dataset, and sliced per fold. That handed the exemption
    # the answer sheet: under injection nobody holds the clean labels, and a
    # training molecule's cliff status was decided partly by held-out
    # molecules. The per-pass call is in the fold loop below.
    #
    # This call stays for ONE reason: to report the set the old code would have
    # used, so a reader can see how far the honest mask moves from it. Its
    # numbers are recorded and nothing is cleaned by them.
    cliff_mask, cliff_info = cliff_mask_for_dataset(
        data, labels, is_clf,
        sim_threshold=float(kwargs.get('cliff_sim_threshold',
                                       CLIFF_SIM_THRESHOLD)),
        activity_k=float(kwargs.get('cliff_activity_k', CLIFF_ACTIVITY_K)),
        threshold_mode=str(kwargs.get('cliff_threshold_mode', 'mad')),
        potency_threshold=float(kwargs.get('cliff_potency_threshold',
                                           CLIFF_POTENCY_THRESHOLD)))
    print(f"  cliff molecules on the CLEAN labels, whole dataset (reported, "
          f"not used for cleaning): {cliff_info['n_cliff_molecules']} of "
          f"{cliff_info['n_molecules']} molecules, structure reference "
          f"{cliff_info.get('structure_reference')}")

    # The structural half of the definition never depends on a label, so one
    # matrix serves every fold, level and draw. Only the activity test is
    # redone per pass. Without this the fold loop would rebuild an O(n^2)
    # fingerprint or edit-distance matrix once per (fold, level, draw).
    # Popped, not read: it is n x n floats and cliff_info is written to the
    # result file. The antibody path returns before it is set, so that one
    # rebuilds its edit distances per pass and this is None.
    cliff_structure = cliff_info.pop('similarity_matrix', None)
    if not cliff_info['reference_independent']:
        print("  the cliff reference is built from the representations "
              "themselves; the exempt set and the suspect set are not "
              "independent on this dataset")

    noise_groups, _n_groups, _largest = noise_groups_for(data)

    results = {
        'metric': metric_name,
        'lower_is_better': lower_is_better,
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
        'pass_is': 'one (fold, noise draw)',
        'base_rep': base_rep_name,
        'models': downstream_models,
        'cleanups': list(CLEANUPS),
        'cleanups_not_run': [c for c in ALL_CLEANUPS if c not in CLEANUPS],
        'cleanups_not_run_because': {
            c: (UNDEFINED_HERE[c] if c in UNDEFINED_HERE else
                'it moves a label to a numeric value and this dataset\'s label '
                'is a class' if c in REGRESSION_ONLY_CLEANUPS else
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
        'noise_levels': levels,
        'noise_seeds': seeds,
        'noise_seed_note': (
            'Both arms of a pair see the same drawn labels, the same detector '
            'scores and the same targets, so the draw cancels out of the '
            'difference the headline counts. It does NOT cancel out of the '
            'absolute scores beside it.'),
        'cliff_reference': convert_numpy(cliff_info),
        'cliff_reference_independent': bool(cliff_info['reference_independent']),
        'random_exemption_note': (
            'The control arm puts back as many molecules as the cliff arm does, '
            'drawn from the training molecules the cliff mask did not name, so '
            'the two exempt sets never overlap.'),
        'weights_reach_feature_selection': False,
        'weights_reach_feature_selection_note': (
            'sample_weight appears nowhere in src/kirby/hybrid.py or '
            'src/kirby/find_best_hybrid.py, so a weighting cleanup weights the '
            'downstream fit only. The features were chosen without any weight, '
            'in every space.'),
        'hybrid_sees_cleaned_labels': False,
        'hybrid_sees_cleaned_labels_note': (
            "The hybrid is built once per (fold, draw) from that fold's "
            'training molecules at their CORRUPTED labels, which is what a '
            'practitioner holds. No cleanup and no exemption reaches the '
            'feature ranking or the budget split.'),
        'per_space': {},
    }

    splits = list(cv_split(groups, n_folds, X_source, labels, mode=mode))
    print(f"  {len(labels)} molecules, {len(splits)} folds, "
          f"models {downstream_models}")
    print(f"  condition {condition}, levels {levels}, draws {seeds}")
    print(f"  detectors {detectors}, flag rates {flag_rates}")
    print(f"  cleanups ({len(CLEANUPS)}): {list(CLEANUPS)}")
    print(f"  base representation: {base_rep_name}")

    # ── The hybrid recipes, fetched once ────────────────────────────────────
    bp = None
    bp_sim = None
    search = None
    _lookup_error = None
    try:
        search, meta = get_best_search_for_downstream(
            reps, labels, groups,
            downstream_model=downstream_models[0], budget=None,
            augmentations=augmentations, dataset_name=dataset_name,
            task_type=task_type, data=data, pool_subset=POOL_SUBSET)
        bp = meta['best_config']['blueprint'] if meta else None
    except Exception as exc:
        meta = None
        _lookup_error = f"{type(exc).__name__}: {exc}"
        print(f"  hybrid recipe lookup failed: {_lookup_error}")

    if bp is None:
        spaces = ('base',)
        results['hybrid_available'] = False
        results['hybrid_lookup_error'] = _lookup_error
        results['hybrid_unavailable_reason'] = (
            (f'the recipe lookup RAISED -- {_lookup_error} -- so neither hybrid '
             f'space could be built. This is NOT a dataset the precompute never '
             f'covered: something is wrong with the stored search and it can be '
             f'fixed. Every arm ran in the base representation only.')
            if _lookup_error else
            (f'no recipe came back from the precomputed pool for subset '
             f'{POOL_SUBSET!r}. Every arm ran in the base representation only.'))
        print(f"  no hybrid recipe from pool subset {POOL_SUBSET}; "
              f"base representation only")
    else:
        spaces = tuple(SPACES)
        results['hybrid_available'] = True
        # ⛔ ONE RECIPE, SEARCHED FOR ONE DOWNSTREAM MODEL, AND EVERY MODEL IS
        # SCORED IN IT. The pool is asked for downstream_models[0] and the
        # hybrid features that recipe chooses are the features all of the
        # models are fitted on, so a difference between two models here is
        # partly a difference between an estimator and another estimator's
        # feature space. n17, n18 and n23 did the same thing and were changed
        # on 2026-09-23 to ask the pool once per model, through
        # noise_helpers.pool_recipe_per_downstream_model.
        #
        # THIS FILE HAS NOT MOVED, AND THE REASON IS COST, WHICH IS THE OWNER'S
        # CALL AND NOT MINE. Here the feature space does not only choose the
        # features: it is where the detector finds every molecule's neighbours
        # and where the out-of-fold and Gaussian-process targets are fitted. So
        # one recipe per model multiplies the hybrid builds, the detector fits
        # and the correction targets by the number of downstream models -- five
        # in a default run -- on top of a file that already fits three arms.
        # The pool lookups themselves are free; the per-fold work is not.
        results['hybrid_recipe_searched_for_model'] = downstream_models[0]
        results['hybrid_recipe_is_one_model_s'] = True
        results['hybrid_recipe_note'] = (
            f'The hybrid recipe was searched for {downstream_models[0]!r} and '
            f'every downstream model is scored in the features it chose. '
            f'noise_helpers.pool_recipe_per_downstream_model gives each model '
            f'its own recipe; here that would multiply the hybrid builds, the '
            f'detector fits and the correction targets by '
            f'{len(downstream_models)}, so it is a decision rather than a '
            f'default.')
        results['hybrid_family'] = meta['objective']
        results['hybrid_pool_subset'] = meta['pool_subset']
        results['hybrid_budget'] = getattr(bp, 'budget', None)
        results['hybrid_reps'] = sorted(getattr(bp, 'reps', []) or [])
        _by_obj = meta.get('by_objective') or {}
        _sim = _by_obj.get('similarity')
        bp_sim = (_sim or {}).get('best_config', {}).get('blueprint')
        print(f"  hybrid recipe: {len(results['hybrid_reps'])} representations, "
              f"budget {results['hybrid_budget']}, searched under objective "
              f"{meta['objective']}")
        if bp_sim is None:
            spaces = tuple(x for x in spaces if x != 'hybrid_similarity')
            results['hybrid_similarity_available'] = False
            results['hybrid_similarity_unavailable_reason'] = (
                'the pool held no search under objective=similarity for this '
                'dataset, so the space whose features were chosen to make the '
                'neighbour distance predictive could not be built. Every arm '
                'still ran in the other spaces.'
                if 'similarity' not in _by_obj else
                'the similarity family was in the pool but its winning '
                'configuration carried no blueprint.')
            print(f"  {results['hybrid_similarity_unavailable_reason']}")
        else:
            results['hybrid_similarity_available'] = True
            results['hybrid_similarity_budget'] = getattr(bp_sim, 'budget', None)
            results['hybrid_similarity_reps'] = sorted(
                getattr(bp_sim, 'reps', []) or [])
            results['hybrid_similarity_note'] = (
                "Its features were chosen so that the distance they define "
                "predicts each molecule's activity from its five nearest "
                "neighbours -- the same quantity every neighbour-based detector "
                "in this file reads.")
    results['spaces_run'] = list(spaces)

    # What each downstream model would have picked for itself. 0 new model fits:
    # every one of these reads rows the search already stored.
    results['top2_base_reps'] = {}
    results['top2_hybrids'] = {}
    results['top2_pairs_by_model'] = {}
    results['hybrid_deltas_by_model'] = {}
    if search is None:
        results['top2_unavailable_reason'] = (
            'the pool returned no SearchResult, so there are no stored rows to '
            'rank. Every number in this file stands.')
    else:
        for _m in downstream_models:
            try:
                results['top2_base_reps'][_m] = top_base_reps(
                    search, _m, 2, require_full=False)
            except Exception as _e:
                print(f"  top-2 base representations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            _th = None
            try:
                _th = top_hybrids(search, _m, 2, require_full=False)
                results['top2_hybrids'][_m] = _drop_blueprint(_th)
            except Exception as _e:
                print(f"  top-2 hybrid configurations for {_m} unavailable: "
                      f"{type(_e).__name__}: {_e}")
            if _th and _th['selected']:
                try:
                    results['hybrid_deltas_by_model'][_m] = hybrid_deltas(
                        search, _m, hybrid=_th['selected'][0])
                except Exception as _e:
                    print(f"  hybrid deltas for {_m} unavailable: "
                          f"{type(_e).__name__}: {_e}")
        try:
            results['top2_pairs_by_model'] = _drop_blueprint(
                select_model_rep_pairs(search, models=list(downstream_models),
                                       n_base=2, n_hybrids=2,
                                       require_full=False))
        except Exception as _e:
            print(f"  top-2 model-and-representation grid unavailable: "
                  f"{type(_e).__name__}: {_e}")

    # ── What this run will fit, before it starts ────────────────────────────
    n_passes = len(splits) * len(seeds) * len(levels)
    n_settings = sum(min(len(flag_rates), len(setting_axis(c, is_clf)[1]))
                     for c in CLEANUPS)
    n_cleanup_fits = (n_passes * len(spaces) * len(detectors) * n_settings
                      * len(downstream_models) * len(ARMS))
    n_baseline_fits = n_passes * len(spaces) * len(downstream_models) * 2
    results['planned_model_fits'] = {
        'passes': n_passes,
        'cleanup_settings_per_pass_per_space_per_detector': n_settings,
        'cleanup_fits': int(n_cleanup_fits),
        'baseline_fits': int(n_baseline_fits),
        'total_model_fits': int(n_cleanup_fits + n_baseline_fits),
        'note': ('One pass is one (fold, noise draw, level). Every cleanup '
                 'setting is fitted once per arm, and there are three arms.'),
    }
    print(f"  this run fits {n_cleanup_fits + n_baseline_fits} models: "
          f"{n_cleanup_fits} cleanup fits over {len(ARMS)} arms and "
          f"{n_baseline_fits} baseline fits, across {n_passes} passes")

    # ── Accumulators, keyed by pass so the pairing is never positional ──────
    base_noisy = {s: {m: {} for m in downstream_models} for s in spaces}
    base_clean = {s: {m: {} for m in downstream_models} for s in spaces}
    arm_score = {}      # [space][det][rate][model][cleanup][arm][pass] = score
    spared = {}         # [space][det][rate][cleanup][pass] = counts
    det_rows = {s: {d: {} for d in detectors} for s in spaces}
    premise = {s: {d: {} for d in detectors} for s in spaces}
    for s in spaces:
        arm_score[s] = {d: {str(r): {m: {c: {a: {} for a in ARMS}
                                         for c in CLEANUPS}
                                     for m in downstream_models}
                            for r in flag_rates} for d in detectors}
        spared[s] = {d: {str(r): {c: {} for c in CLEANUPS}
                         for r in flag_rates} for d in detectors}
    hybrid_failures = []
    provenance = {}
    fold_cliff_counts = []

    t0 = time.time()
    for level in levels:
        for fold_i, (train_idx, val_idx) in enumerate(splits):
            y_clean_train = labels[train_idx]
            y_val = labels[val_idx]
            g_tr = (np.asarray(groups)[train_idx] if groups is not None
                    else None)
            n_train = int(len(train_idx))

            _tr_reps = {k: np.asarray(v[train_idx], dtype=np.float64)
                        for k, v in reps.items()}
            _va_reps = {k: np.asarray(v[val_idx], dtype=np.float64)
                        for k, v in reps.items()}
            _tr_augs = ({k: np.asarray(v)[train_idx] for k, v in
                         augmentations.items()} if augmentations else None)
            _va_augs = ({k: np.asarray(v)[val_idx] for k, v in
                         augmentations.items()} if augmentations else None)

            for seed in seeds:
                pass_key = f'L{level:g}|f{fold_i}|s{seed}'
                spec = noise_spec_for(dataset_name, condition, level,
                                      noise_seed=seed, is_clf=is_clf)
                y_noisy, corrupt_mask, prov = inject_train_labels(
                    labels, spec, train_idx=train_idx, is_clf=is_clf,
                    groups=noise_groups, dataset_name=dataset_name)
                provenance.setdefault(spec.spec_id, convert_numpy(prov))

                # THE EXEMPT SET, from the labels as they LOOK and over the
                # training molecules alone. inject_train_labels returns
                # y_noisy ALREADY sliced to train_idx (its docstring says so),
                # so it is passed straight through; `subset` tells the mask
                # which molecules those labels belong to.
                cliff_train, pass_cliff_info = cliff_mask_for_dataset(
                    data, np.asarray(y_noisy), is_clf,
                    sim_threshold=float(kwargs.get('cliff_sim_threshold',
                                                   CLIFF_SIM_THRESHOLD)),
                    activity_k=float(kwargs.get('cliff_activity_k',
                                                CLIFF_ACTIVITY_K)),
                    threshold_mode=str(kwargs.get('cliff_threshold_mode',
                                                  'mad')),
                    potency_threshold=float(kwargs.get(
                        'cliff_potency_threshold', CLIFF_POTENCY_THRESHOLD)),
                    subset=train_idx, structure=cliff_structure)

                rand_train, n_rand = random_exempt_mask(
                    cliff_train, int(cliff_train.sum()), fold_i, seed)
                fold_cliff_counts.append({
                    'pass': pass_key,
                    'n_training_molecules': n_train,
                    'n_cliff_molecules_in_training': int(cliff_train.sum()),
                    'n_random_control_molecules': int(n_rand),
                    'n_corrupted_molecules': int(corrupt_mask.sum()),
                    'n_cliff_molecules_corrupted':
                        int((cliff_train & corrupt_mask).sum()),
                    # What the OLD code would have exempted here: the clean
                    # column's mask sliced to this fold. The gap between the
                    # two is how much the answer sheet was worth.
                    'n_cliff_on_the_clean_labels':
                        int(np.asarray(cliff_mask)[train_idx].sum()),
                })

                # ── The feature spaces for this pass ────────────────────────
                fold_spaces = {}
                scaler = StandardScaler()
                fold_spaces['base'] = (
                    scaler.fit_transform(np.asarray(X_source[train_idx],
                                                    dtype=np.float64)),
                    scaler.transform(np.asarray(X_source[val_idx],
                                                dtype=np.float64)))
                for _sp, _bp in (('hybrid', bp), ('hybrid_similarity', bp_sim)):
                    if _sp not in spaces or _bp is None:
                        continue
                    X_tr_h, X_va_h = hybrid_for_fold(
                        _tr_reps, _va_reps, y_noisy, _bp, g_tr,
                        model_factories[downstream_models[0]](),
                        metric_fn, lower_is_better,
                        augmentations_train=_tr_augs,
                        augmentations_val=_va_augs)
                    if X_tr_h is None:
                        hybrid_failures.append({'pass': pass_key, 'space': _sp})
                    else:
                        sc_h = StandardScaler()
                        fold_spaces[_sp] = (sc_h.fit_transform(X_tr_h),
                                            sc_h.transform(X_va_h))

                for space in spaces:
                    if space not in fold_spaces:
                        continue
                    X_tr_s, X_va_s = fold_spaces[space]

                    # The model-based corrections' targets, once per (pass,
                    # space). They depend on the features, the labels and the
                    # groups and on nothing else -- not the suspicion score,
                    # not the strength, not which model is about to be fitted.
                    _targets = correction_targets(
                        X_tr_s, y_noisy, groups=g_tr,
                        want=targets_needed(CLEANUPS))

                    for mname in downstream_models:
                        # The score to beat: the same corrupted labels, no
                        # cleaning. One fit per (pass, space, model), shared by
                        # every cleanup, setting, detector and arm.
                        sc = _evaluate_fold(
                            X_tr_s, X_va_s, y_noisy, y_val, 'noisy_baseline',
                            0.0, model_factories[mname], metric_fn,
                            is_clf=is_clf, metric_name=metric_name,
                            dataset_name=dataset_name, precomputed_y_tr=y_noisy)
                        base_noisy[space][mname][pass_key] = (
                            float(sc) if sc is not None else float('nan'))
                        # The ceiling: the labels before anything was injected.
                        # Nothing can be cleaned back past it.
                        sc = _evaluate_fold(
                            X_tr_s, X_va_s, y_clean_train, y_val, 'no_noise',
                            0.0, model_factories[mname], metric_fn,
                            is_clf=is_clf, metric_name=metric_name,
                            dataset_name=dataset_name)
                        base_clean[space][mname][pass_key] = (
                            float(sc) if sc is not None else float('nan'))

                    for detector in detectors:
                        try:
                            scores = _compute_noise_scores(
                                X_tr_s, y_noisy, is_clf=is_clf,
                                detector=detector, groups=g_tr)
                        except Exception as exc:
                            print(f"      {detector} refused: "
                                  f"{type(exc).__name__}: {exc}")
                            scores = None
                        if scores is None:
                            continue
                        scores = np.asarray(scores, dtype=float)

                        det = detection_scores(corrupt_mask, scores, condition)
                        if det is not None:
                            det_rows[space][detector][pass_key] = det
                        pr = _cliff_suspicion_counts(scores, corrupt_mask,
                                                     cliff_train)
                        if pr is not None:
                            premise[space][detector][pass_key] = pr

                        for slot, rate in enumerate(flag_rates):
                            rk = str(rate)
                            for cleanup in CLEANUPS:
                                axis, vals = setting_axis(cleanup, is_clf)
                                if slot >= len(vals):
                                    # Fewer settings than slots: quantile_clip
                                    # reads neither axis, so a second row here
                                    # would be the same column fitted again.
                                    continue
                                strength = (None if axis == 'flag_rate'
                                            else vals[slot])
                                try:
                                    plain = cleaned_training_set(
                                        cleanup, X_tr_s, y_noisy, scores, rate,
                                        is_clf=is_clf, groups=g_tr,
                                        strength=strength, targets=_targets)
                                except Exception as exc:
                                    print(f"      {cleanup} declined: "
                                          f"{type(exc).__name__}: {exc}")
                                    plain = None
                                if plain is None:
                                    # `remove` left under ten training
                                    # molecules, or a target could not be
                                    # built. The whole pair is dropped, so the
                                    # comparison is never half-scored.
                                    continue

                                cliff_arm = put_back(plain, cliff_train,
                                                     X_tr_s, y_noisy)
                                rand_arm = put_back(plain, rand_train,
                                                    X_tr_s, y_noisy)
                                spared[space][detector][rk][cleanup][pass_key] = {
                                    'n_training_molecules': n_train,
                                    'n_cliff_molecules_in_training':
                                        int(cliff_train.sum()),
                                    'n_cliff_molecules_spared': n_spared(
                                        plain, cliff_arm, cliff_train, y_noisy),
                                    'n_random_control_molecules': int(n_rand),
                                    'n_random_molecules_spared': n_spared(
                                        plain, rand_arm, rand_train, y_noisy),
                                    'setting_axis': axis,
                                    'setting': (rate if axis == 'flag_rate'
                                                else strength),
                                }
                                for mname in downstream_models:
                                    for arm, cleaned in (('plain', plain),
                                                         ('cliff_exempt', cliff_arm),
                                                         ('random_exempt', rand_arm)):
                                        arm_score[space][detector][rk][mname][
                                            cleanup][arm][pass_key] = _fit_score(
                                                cleaned, X_va_s, y_val,
                                                model_factories[mname],
                                                metric_fn, metric_name,
                                                label=f'{cleanup}/{arm}/{mname}')
                print(f"    {pass_key} done [{time.time() - t0:.0f}s]")

    if hybrid_failures:
        results['passes_with_no_hybrid'] = hybrid_failures
        print(f"  a hybrid space could not be built on "
              f"{len(hybrid_failures)} (pass, space) pairs; the spaces that "
              f"built keep their numbers")

    results['noise_provenance'] = provenance
    results['per_pass_counts'] = convert_numpy(fold_cliff_counts)

    # ── The paired differences ──────────────────────────────────────────────
    for space in spaces:
        space_block = {'per_detector': {},
                       'per_model_no_cleaning_mean': {},
                       'per_model_clean_label_ceiling_mean': {}}
        for m in downstream_models:
            vals = np.array(list(base_noisy[space][m].values()), dtype=float)
            space_block['per_model_no_cleaning_mean'][m] = (
                float(np.nanmean(vals)) if vals.size else float('nan'))
            vals = np.array(list(base_clean[space][m].values()), dtype=float)
            space_block['per_model_clean_label_ceiling_mean'][m] = (
                float(np.nanmean(vals)) if vals.size else float('nan'))

        for detector in detectors:
            by_rate = {}
            for rate in flag_rates:
                rk = str(rate)
                block = {'per_model': {}, 'molecules_spared': convert_numpy(
                    spared[space][detector][rk])}
                for m in downstream_models:
                    row = {}
                    for cleanup in CLEANUPS:
                        cells = arm_score[space][detector][rk][m][cleanup]
                        keys = sorted(cells['plain'])
                        if not keys:
                            continue
                        axis, vals_ = setting_axis(cleanup, is_clf)
                        slot = flag_rates.index(rate)
                        entry = {
                            'swept_axis': axis,
                            'setting': (rate if axis == 'flag_rate'
                                        else (vals_[slot]
                                              if slot < len(vals_) else None)),
                            'what_the_cleanup_does':
                                WHAT_THE_CLEANUP_DOES.get(cleanup, 'unknown'),
                            'plain_mean': float(np.nanmean(
                                [cells['plain'][k] for k in keys])),
                        }
                        # Two paired comparisons per cleanup: the treatment
                        # against the plain cleanup, and the control against
                        # the plain cleanup. Every difference is signed by
                        # _gain, which reads lower_is_better.
                        for arm in ('cliff_exempt', 'random_exempt'):
                            gains = [_gain(cells[arm].get(k, float('nan')),
                                           cells['plain'][k], lower_is_better)
                                     for k in keys]
                            gains = np.array([g for g in gains
                                              if np.isfinite(g)], dtype=float)
                            sp = [spared[space][detector][rk][cleanup].get(k, {})
                                  for k in keys]
                            spared_key = ('n_cliff_molecules_spared'
                                          if arm == 'cliff_exempt'
                                          else 'n_random_molecules_spared')
                            n_sp = [int(x.get(spared_key, 0)) for x in sp]
                            entry[arm] = {
                                'passes_won': int((gains > 0).sum()),
                                'passes_scored': int(gains.size),
                                'mean_gain_over_plain': (
                                    float(gains.mean()) if gains.size
                                    else float('nan')),
                                'pass_gains': [float(x) for x in gains],
                                'spread_over_passes': (
                                    float(gains.std()) if gains.size
                                    else float('nan')),
                                'arm_mean': float(np.nanmean(
                                    [cells[arm].get(k, float('nan'))
                                     for k in keys])),
                                'molecules_put_back_per_pass': n_sp,
                                'nothing_was_spared': bool(
                                    n_sp and max(n_sp) == 0),
                            }
                        # The one number the third arm exists for: the cliff
                        # exemption against the random one, pass by pass.
                        paired = [_gain(cells['cliff_exempt'].get(k, float('nan')),
                                        cells['random_exempt'].get(k, float('nan')),
                                        lower_is_better) for k in keys]
                        paired = np.array([g for g in paired if np.isfinite(g)],
                                          dtype=float)
                        entry['cliff_exempt_over_random_exempt'] = {
                            'passes_won': int((paired > 0).sum()),
                            'passes_scored': int(paired.size),
                            'mean_gain': (float(paired.mean()) if paired.size
                                          else float('nan')),
                            'pass_gains': [float(x) for x in paired],
                        }
                        row[cleanup] = entry
                    block['per_model'][m] = row
                by_rate[rk] = convert_numpy(block)
            space_block['per_detector'][detector] = {
                'per_setting_slot': by_rate,
                'detection': convert_numpy(det_rows[space][detector]),
                'cliff_molecules_among_the_suspicious': convert_numpy(
                    premise[space][detector]),
            }
        results['per_space'][space] = space_block

    results['summary'] = _summarise(results, downstream_models, flag_rates,
                                    detectors, spaces, metric_name, CLEANUPS)
    _print_summary(results, metric_name)

    save_experiment_results(results, output_dir, 'n25_cliff_exemption',
                            dataset_name)
    return results


def _summarise(results, models, flag_rates, detectors, spaces, metric_name,
               cleanups):
    """How many settings each exemption won a majority of passes in, per space.

    One setting is one (detector, setting slot, model). Every count is taken
    inside ONE space: a setting that won in the base representation and lost in
    the hybrid is two readings, not an average.
    """
    out = {'metric': metric_name,
           'headline': 'passes won by the exempt arm over the plain arm',
           'pass_is': 'one (fold, noise draw)',
           'spaces': list(spaces), 'per_space': {}}
    for space in spaces:
        per_cleanup = {}
        for cleanup in cleanups:
            rows = []
            for d in detectors:
                for r in flag_rates:
                    blk = (results['per_space'].get(space, {})
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
                            'cliff_over_random_passes_won':
                                cell['cliff_exempt_over_random_exempt']['passes_won'],
                            'cliff_over_random_mean_gain':
                                cell['cliff_exempt_over_random_exempt']['mean_gain'],
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
                        if x['cliff_over_random_passes_won'] * 2
                        > x['passes_scored']]
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
                'settings_where_nothing_was_spared':
                    len(rows) - len(live),
                'settings_the_cliff_exemption_won_most_passes_in':
                    len(majority),
                'settings_the_cliff_exemption_won_every_pass_in': len(sweep),
                'settings_the_random_exemption_won_most_passes_in':
                    len(rnd_majority),
                'settings_the_cliff_exemption_beat_the_random_one_in':
                    len(over_rnd),
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
                '-- remove, knn_smooth, prox_smooth -- touches only the flagged '
                'share, so an exemption spares the few cliff molecules that '
                'were flagged. A proportional one -- downweight, huber_weight, '
                'label_smooth and the other smoothers on a regression label -- '
                'moves or reweights EVERY molecule by its score, so an '
                'exemption spares every cliff molecule in the fold. Measured '
                'in the n25 smoke on 20 exempt molecules: knn_smooth spared 2 '
                'of them and downweight spared all 20. The two families are '
                'therefore not having the same amount undone, and a larger '
                'gain for the proportional family is partly that.'),
        }
    return out


def _print_summary(results, metric_name):
    s = results['summary']
    print("\n" + "=" * 70)
    print(f"SUMMARY — passes the exempt arm won against the same cleanup "
          f"unexempted ({metric_name})")
    ci = results['cliff_reference']
    print(f"  {ci['n_cliff_molecules']} of {ci['n_molecules']} molecules are on "
          f"a cliff, by {ci.get('structure_reference')} and "
          f"{ci.get('threshold_mode')}")
    if not results['cliff_reference_independent']:
        print("  ⛔ the cliff reference is built from the representations "
              "themselves on this dataset; the exempt set and the suspect set "
              "are not independent")
    for space, srow in s['per_space'].items():
        print(f"\n  {space}:")
        for cleanup, row in srow['per_cleanup'].items():
            if not row.get('settings_scored'):
                print(f"    {cleanup:14s} no setting produced a score")
                continue
            print(f"    {cleanup:14s} ({row['what_the_cleanup_does']}) "
                  f"cliff exemption won most passes in "
                  f"{row['settings_the_cliff_exemption_won_most_passes_in']} of "
                  f"{row['settings_scored']} settings; the random exemption in "
                  f"{row['settings_the_random_exemption_won_most_passes_in']}; "
                  f"cliff beat random in "
                  f"{row['settings_the_cliff_exemption_beat_the_random_one_in']}")
            if row['settings_where_nothing_was_spared']:
                print(f"                   "
                      f"{row['settings_where_nothing_was_spared']} of those "
                      f"settings put back no molecule at all, so the two arms "
                      f"were the same training set")
            b = row.get('best_setting')
            if b:
                print(f"                   best: {b['model']}/{b['detector']}"
                      f"/{b['swept_axis']} {b['setting']} — "
                      f"{b['cliff_passes_won']} of {b['passes_scored']} passes, "
                      f"mean gain {b['cliff_mean_gain']:+.4f}")
        print("    by what the cleanup does:")
        for fam, row in srow['by_what_the_cleanup_does'].items():
            print(f"      {fam:22s} cliff exemption won most passes in "
                  f"{row['settings_the_cliff_exemption_won_most_passes_in']} of "
                  f"{row['settings_scored']} settings "
                  f"({', '.join(row['cleanups'])})")
    print("\n  One setting winning on one pass is not a result. Read the counts.")


def main():
    parser = create_experiment_parser(
        'N25: does exempting cliff molecules from cleaning help')
    parser.add_argument('--flag-rates', type=str, default=None,
                        help='Comma-separated shares of training molecules '
                             'treated as suspect')
    parser.add_argument('--base-rep', type=str, default=None,
                        help='Which representation is the base space. Default: '
                             "pick_noise_base_rep's, and it is recorded either way")
    parser.add_argument('--detectors', type=str, default=None,
                        help='Comma-separated detector names')
    parser.add_argument('--cleanups', type=str, default=None,
                        help="Comma-separated cleanup names. Default: every one "
                             "n07 registers")
    parser.add_argument('--noise-condition', type=str, default=NOISE_CONDITION,
                        choices=list(DETECTION_IS_DEFINED),
                        help='Which condition corrupts the training labels')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated noise levels, as fractions of the '
                             'label spread')
    parser.add_argument('--noise-seeds', type=str, default=None,
                        help='Comma-separated draw indices. Each one multiplies '
                             'every model fit')
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
        'noise_condition': args.noise_condition,
        'cliff_activity_k': args.cliff_activity_k,
        'cliff_sim_threshold': args.cliff_sim_threshold,
        'cliff_threshold_mode': args.cliff_threshold_mode,
        'cliff_potency_threshold': args.cliff_potency_threshold,
    }
    if args.flag_rates:
        kw['flag_rates'] = [float(x) for x in args.flag_rates.split(',')]
    if args.base_rep:
        kw['base_rep'] = args.base_rep
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
