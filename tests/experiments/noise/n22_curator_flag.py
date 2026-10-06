#!/usr/bin/env python3
"""N22: do the detectors find the records a human marked wrong?

Paper: Noise.

THE QUESTION. Every detection number in this suite is scored against a list
the same script drew. n11 corrupts labels and asks which detector finds the
corrupted ones; n13 does it across ten noise shapes; n21 across distances. The
list of right answers and the noise are the same act.

ChEMBL writes a field called `data_validity_comment` on records its own
curators found suspect. On CHEMBL240 in nanomolar with relation '=', 301 of
12,008 IC50 records carry one, on 215 of the 9,802 molecules, counted
2026-09-17 under ChEMBL_37. Those 301 are a list of suspect records a person
made. n22 scores the same detectors against that list, through the same
`noise_helpers.detection_scores` that grades the injected runs, so the two
numbers stand beside each other: the same detectors, the same three figures,
one graded against a distribution and one against a person.

WHAT THE FLAG MEANS, AND WHAT THAT COSTS. The 301 are not one kind of thing:

    293  Outside typical range
      8  Potential transcription error

"Outside typical range" is assigned because the VALUE is extreme. A detector
that scores a record by how far its label sits from its neighbours' labels is
therefore being graded partly against a property of the label it was given,
not only against an independent verdict. That is a real limit on how the
headline reads and it is written into the result file rather than left for a
reader to work out. The two texts are graded separately as well as together;
8 records is too few for a number to mean anything on its own, and the file
says so where it prints it.

WHAT IS SCORED, per detector and per feature space:

  detection_average_precision   area under the precision-recall curve over all
                                cut-offs; no threshold to choose.
  detection_precision_at_k      of the k records ranked most suspicious, the
                                share that carry a flag. k is the number that
                                carry one.
  detection_recall_at_k         of the flagged records, the share inside that
                                same top k. Equal to the precision at this k.

Beside every one of them, the two counts they are shares of — how many records
carry a flag and how many records were scored — and `chance_average_precision`,
the flagged share, which is what a detector that ranked at random would get.

GRADED TWICE, OVER DIFFERENT THINGS. One row here is one ChEMBL RECORD, and a
molecule measured four times is four rows with identical features. So the
record-level grading has a molecule's repeats sitting on top of each other.
The same three numbers are therefore also computed over MOLECULES, where a
molecule counts once and is flagged if any of its records is, and the
suspicion score is that molecule's largest. Both are in the file, each saying
what one row of it counts.

THE FEATURE SPACES, chosen per outer fold (2026-10-05, chat 5 alignment,
HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4). The outer folds are
noise_helpers.noise_outer_folds: kirby.find_best_hybrid.nested_cv_outer_folds
on the scaffold groups, the count from the run's config, any count including
1. On each fold, searches on that fold's training records only
(noise_helpers.fold_configurations: decided grid, pruning off) pick, for each
of herg_ic50's two decided models (svm and dnn, herg_ki's, handoff 2.3 item
14), its 2 best single representations and its 2 best general hybrids. That
is 2 models x 4 = 8 spaces per fold, named '<slot>@<model>' (e.g.
'base_1@svm', 'hybrid__general2@dnn'). A slot is a rank, not a fixed
representation: base_1@svm can be a different representation on two folds,
and each fold's record says which. Every detector runs in every space. n22
fits no downstream model, so a "configuration" here is a feature space the
detectors run in.

Until 2026-10-05 the spaces came from whole-dataset stored searches (rf's top
2 single representations and five models' top 2 general + top 1 distance
hybrids per family, up to 40 spaces), so a flagged record's own label helped
choose the space it was scored in.

The searches run on the rows n22 grades: the records as loaded. A molecule's
records share one scaffold group, so the group-aware folds keep all of a
molecule's measurements on one side, in the search and in the grading. At 5
folds a fold's training block is about 9,600 records, under the 10,000 rows
above which the engine skips svm.

THE NEIGHBOUR SETTINGS (KNN_SEARCH_SPACE_RECORD.md, Adelaide 2026-10-05).
Each detector that reads a neighbour list is also run with its neighbours
taken from every setting of the dataset's group (herg_ic50 has 9,802
molecules, so fast_knn), at 5, 10 and 20 neighbours, under both counting
schemes ('rank' and 'inverse_square', similarity_space.SWEPT_NEIGHBOUR_SCHEMES)
for the detectors that weight their neighbours by a scheme. distance_weighted
weights by one over the distance itself and takes no scheme, so it is scored
once per setting and k, keyed 'scheme=own'. The settings' spaces are
the model's own per-fold ones: its best single representation, its two
general hybrids, the top hybrid of the fold's distance search, the top hybrid
of the fold's similarity search, and forest proximity. The distance and
similarity hybrids are neighbour spaces only, never one of the 8 spaces.
Scaffold-mates are not excluded (axis 6); a record's OTHER measurements of
the same molecule are, because they are identical rows that would hand it its
own label back. No setting is screened out: n22 refits no model per setting,
so every setting is scored.

THE DETECTOR LIST IS n11's, imported rather than copied, for the reason
recorded at n06_bindingdb_ground_truth.py:108-112 — two hand-kept lists drift,
and then the two files cannot be joined even where the same code ran.

WHAT THIS DOES NOT ANSWER. Owner's ruling, 2026-09-17. The flag is OUTLIER
noise — a measurement whose value is out of range, or a number typed in wrong.
That is one kind of noise and this experiment tests detectors on that one
kind. A detector that wins here has not been shown to win on any other kind,
and a detector that loses here has not been shown to lose on any other kind.
Nothing in this file may be reported as a result about noise in general.

In particular it has nothing to say about the variation that comes from
different laboratories running the experiment differently, or from the same
laboratory varying between runs. That is a different quantity and it is
measured elsewhere in this directory: n06 correlates the detectors against the
disagreement between laboratories on BindingDB, n17 removes molecules on that
disagreement, and n19 uses the measurement count and the spread. None of those
three reads the curator flag, and this file does not read their disagreement.

WHICH DATASET. `herg_ic50` only, because it is the only dataset here whose
loader keeps one row per ChEMBL record and carries the flag. Anything else
gets a recorded skip with the reason, not a crash, before any search runs.

Usage:
    python tests/experiments/run_dataset.py noise herg_ic50 --experiments n22
    python n22_curator_flag.py --dataset herg_ic50
"""
import sys
import os
import inspect
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

# 2026-10-05 (chat 5 alignment, HANDOFF_EXPERIMENT_ALIGNMENT.md 2.4): the
# whole-dataset readers (get_best_search_for_downstream, get_rule_hybrids,
# noise_base_reps_by_model, extract_top3_base_reps, top_base_reps,
# top_hybrids, select_model_rep_pairs, hybrid_deltas), DOWNSTREAM_MODELS and
# n06's pool helpers are no longer imported. Every space is chosen per outer
# fold on that fold's training rows, for the dataset's two decided models.
from experiments._shared import (  # noqa: E402
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    get_domain_metric, get_model_factories,
)
from experiments.noise.noise_helpers import (  # noqa: E402
    noise_outer_folds, fold_spread, fold_configurations, fold_model_inputs,
    configuration_record, CONFIGURATION_SPACES,
)
from kirby.utils.noise_estimation import get_noise_estimator  # noqa: E402
# The neighbour settings and the neighbour lists (kNN record, 2026-10-05):
# sx.sweep_settings is group-aware when handed the dataset name and its
# molecule count.
from experiments.noise import _sim_axis_n15_n22 as sx  # noqa: E402
from experiments.noise import similarity_space as ss  # noqa: E402

# The name the checkpoint is filed under.
EXP_FULL_NAME = 'n22_curator_flag'

# Identity of the per-fold checkpoint lines and score files. Bump when what a
# fold computes changes, so an older partial run is never read back. The
# whole-dataset design before 2026-10-05 wrote a pickle under
# _checkpoint/; it is not read by this version.
# v2 (2026-10-06): the neighbour settings are also swept over the counting
# scheme, so each setting's key gained '|scheme=<s>'.
CHECKPOINT_IDENTITY = 'n22_per_fold_v2_2026-10-06'

from noise_helpers import detection_scores  # noqa: E402

# n06's estimator list and matrix clean-up, imported so the two files run the
# same detectors on the same matrices and their tables can be read together.
# `resolve_estimators` resolves n11's BASE_ESTIMATORS plus the torch and
# optional ones when they import.
from n06_bindingdb_ground_truth import (  # noqa: E402
    resolve_estimators, _canonical, _clean_matrix,
    GP_SUBSAMPLED_ESTIMATORS, GP_MAX_ROWS,
)

# Spaces the neighbour settings read that are not among the 8 detector
# spaces: the top hybrid of the fold's distance search and the top hybrid of
# the fold's similarity search (handoff 2.3 item 1), and the best single
# representation with augmentation features (unavailable on herg_ic50, which
# loads no augmentation representations). Built per fold, per model, by
# noise_helpers.fold_model_inputs, the same build the 8 spaces get.
# DECIDED 2026-10-06 (Adelaide delegated): the detectors run in the 8 spaces
# only. The distance and similarity hybrids stay neighbour settings and are
# not added as detector spaces (n06 does add them; n22 does not).
NEIGHBOUR_ONLY_SPACES = ('hybrid__distance1', 'hybrid_similarity') + tuple(
    ss.AUG_SPACES)

# The counting schemes swept for the detectors that read one (2026-10-06).
SCHEMES = tuple(ss.SWEPT_NEIGHBOUR_SCHEMES)
# The key used for a detector that weights its neighbours its own way.
OWN_SCHEME = 'own'


def reads_scheme(est_name):
    """Does this detector weight its neighbours by the lists' 'scheme'?
    True for the ones whose class calls noise_estimation._scheme_weights
    (neighbor_residual, local_variance_ratio, activity_cliff_aware as of
    2026-10-06); distance_weighted uses one over the distance and does not."""
    try:
        cls = type(get_noise_estimator(est_name))
        return '_scheme_weights' in inspect.getsource(cls)
    except Exception:
        return False

# The condition name detection_scores is told, so the grading takes the same
# path the injected runs take. It is in
# noise_helpers.DETECTION_DEFINED_WITHOUT_INJECTION and deliberately not in
# DETECTION_IS_DEFINED: no injector can produce it.
CONDITION = 'curator_flag'

# The only dataset whose loader keeps one row per ChEMBL record AND carries
# the flag. Named here rather than tested for, so a run on anything else says
# why it stopped instead of failing on a missing key.
DATASET = 'herg_ic50'


def _flag_for_rows(data, dataset_name):
    """The curator flag for each row, or a reason there is none.

    Returns (flag, flag_text, note) with flag a boolean array one per row, or
    (None, None, reason).

    The flag is re-read from the loader rather than carried through the
    experiment framework, and then checked against the labels the framework
    handed over. The loader is deterministic and both come from the same CSV,
    so they agree — but if they ever stop agreeing, every number below would
    be scored against the wrong records with nothing to show for it, so the
    check refuses rather than warns.
    """
    if str(dataset_name).lower() != DATASET:
        return None, None, (
            f'{dataset_name} has no curator flag. The flag is ChEMBL\'s '
            f'data_validity_comment and only {DATASET} keeps one row per '
            f'ChEMBL record with that column')
    try:
        from kirby.datasets.herg import load_herg_ic50_measurements
        d = load_herg_ic50_measurements()
    except Exception as exc:
        return None, None, (f'the per-record hERG IC50 pull would not load: '
                            f'{type(exc).__name__}: {exc}')
    if not d.get('curator_flag_available'):
        return None, None, (
            'the cached hERG IC50 pull has no data_validity_comment column. '
            'It was written before that column was requested. Re-fetch with: '
            'python tests/experiments/noise/_probe_chembl_herg.py '
            '--standard-type IC50')

    labels = np.asarray(data['labels'], dtype=float)
    loader_labels = np.asarray(d['labels'], dtype=float)
    if len(loader_labels) != len(labels):
        return None, None, (
            f'the loader returned {len(loader_labels)} records and the '
            f'experiment was handed {len(labels)} rows. The flag indexes the '
            f'loader\'s rows, so grading against it would score the wrong '
            f'records')
    if not np.allclose(loader_labels, labels, rtol=0, atol=1e-9,
                       equal_nan=True):
        n_off = int(np.sum(~np.isclose(loader_labels, labels, rtol=0,
                                       atol=1e-9, equal_nan=True)))
        return None, None, (
            f'{n_off} of the {len(labels)} labels the experiment was handed '
            f'differ from the loader\'s, so the two are not the same rows in '
            f'the same order and the flag cannot be lined up against them')
    return (np.asarray(d['curator_flag'], dtype=bool),
            list(d['curator_flag_text']),
            'read from the same loader that produced the labels, and checked '
            'against them row by row')


def grade(flag, suspicion, *, label):
    """The three detection numbers, plus what a random ranking would score.

    flag       one boolean per thing graded: did a curator mark it suspect
    suspicion  one float per thing graded, higher meaning more suspicious.
               NaN means the detector wrote no score for it, and those are
               dropped here rather than filled in — a filled-in score is a
               detector opinion nobody computed.
    label      what one entry counts: 'records' or 'molecules'

    Returns None when nothing can be graded, with the same meaning
    `detection_scores` gives it: a suppressed figure is readable and a
    meaningless 1.0 is not.
    """
    flag = np.asarray(flag, dtype=bool).ravel()
    sus = np.asarray(suspicion, dtype=float).ravel()
    if sus.size != flag.size:
        return None
    kept = np.isfinite(sus)
    n_dropped = int((~kept).sum())
    if kept.sum() == 0:
        return None
    out = detection_scores(flag[kept], sus[kept], CONDITION)
    if out is None:
        return None
    n_flag = int(out['detection_n_corrupted'])
    n_all = int(out['detection_n_training'])
    out = dict(out)
    # detection_scores names its two counts after injected noise, because that
    # is what every other caller hands it. Here they count flagged records and
    # scored records, so they are written again under names that say so. The
    # originals are left in place so a reader joining this file to n11's is
    # joining on the same keys.
    out['n_flagged'] = n_flag
    out['n_scored'] = n_all
    out['graded_over'] = label
    out['n_with_no_score_dropped'] = n_dropped
    out['chance_average_precision'] = n_flag / float(n_all)
    out['what_the_counts_count'] = (
        f'{n_flag} {label} carry a ChEMBL curator comment, out of {n_all} '
        f'{label} the detector scored. A detector ranking at random scores '
        f'{n_flag / float(n_all):.4f} average precision.')
    return out


def _by_molecule(values, molecule_of_row, n_molecules, how):
    """Collapse one number per record into one per molecule.

    how='any'  a molecule is flagged if any of its records is
    how='max'  a molecule's suspicion is the largest of its records'
    A molecule none of whose records kept a score comes back NaN, and grade()
    drops it.
    """
    if how == 'any':
        out = np.zeros(n_molecules, dtype=bool)
        for i, m in enumerate(molecule_of_row):
            if values[i]:
                out[int(m)] = True
        return out
    out = np.full(n_molecules, np.nan, dtype=float)
    for i, m in enumerate(molecule_of_row):
        v = values[i]
        if not np.isfinite(v):
            continue
        m = int(m)
        if not np.isfinite(out[m]) or v > out[m]:
            out[m] = v
    return out



def _matrix_signature(X_tr, X_va):
    """Same bytes in both blocks = the same space; scored once, copied."""
    from experiments.noise.noise_cache import array_hash
    return array_hash(np.ascontiguousarray(X_tr), np.ascontiguousarray(X_va))


def _recipe_summary(bp, aug_cond):
    """The recipe fields a reader needs, from a Blueprint."""
    if bp is None:
        return None
    return {'budget_features': int(getattr(bp, 'budget', 0) or 0),
            'allocation_method': getattr(bp, 'allocation_method', None),
            'importance_method': getattr(bp, 'importance_method', None),
            'internal_evaluator': getattr(bp, 'internal_evaluator', None),
            'n_reps': len(getattr(bp, 'reps', []) or []),
            'reps': sorted(getattr(bp, 'reps', []) or []),
            'aug_condition': aug_cond}


def _estimator_kwargs(est_name, *, groups, exclude, is_clf):
    """The keyword arguments every estimator is handed: the scaffold groups
    where it takes them (its own folds), the record's own molecule as the
    exclusion, and the classification switch."""
    params = inspect.signature(
        type(get_noise_estimator(est_name)).__init__).parameters
    kw = {}
    if groups is not None and 'groups' in params:
        kw['groups'] = groups
    if 'exclude_groups' in params:
        # The record's own other measurements of the same molecule, never its
        # scaffold group (kNN record axis 6 keeps scaffold-mates).
        kw['exclude_groups'] = exclude
    if is_clf and 'is_clf' in params:
        kw['is_clf'] = True
    return kw


def _score_one_fold(data, dataset_name, labels, molecule_of_row, estimators,
                    sim_estimators, sweep, is_clf, search_groups, metric, *,
                    fold_i, train_idx, val_idx, n_folds, scheme_readers=None):
    """Every detector's score for this fold's held-out records, in every
    space this fold's searches chose, by the detector's own neighbour search
    and under every neighbour setting.

    Returns (meta, {space: {estimator: held-out scores}},
             {space: {'<setting>|k=<k>|scheme=<s>': {estimator: held-out
             scores}}}).
    """
    scheme_readers = scheme_readers or {}
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    task_type = data.get('task_type') or 'regression'
    t_fold = time.time()
    print(f"\n  === fold {fold_i}: {len(train_idx)} training records, "
          f"{len(val_idx)} held-out records ===")

    # ── this fold's spaces: per decided model, 2 + 2, plus neighbour spaces ──
    entries = fold_configurations(data, dataset_name, fold_i=fold_i,
                                  train_idx=train_idx, n_folds=n_folds)
    need_nb = {s.space for s in sweep} & set(NEIGHBOUR_ONLY_SPACES)
    wanted = tuple(CONFIGURATION_SPACES) + tuple(sorted(need_nb))
    detector_X, nb_spaces, space_info = {}, {}, {}
    unavailable, configurations = {}, {}
    for m, entry in entries.items():
        factory = get_model_factories([m], task_type=task_type,
                                      n_samples=len(train_idx)).get(m)
        X, unav, names = fold_model_inputs(
            data, dataset_name, m, entry, train_idx=train_idx,
            val_idx=val_idx, y_train=labels[train_idx], factory=factory,
            metric=metric, spaces=wanted)
        configurations[m] = configuration_record(entry, names)
        for sp, why in (unav or {}).items():
            if sp in wanted:
                unavailable[f'{sp}@{m}'] = why
        recipes = (entry or {}).get('recipes') or {}
        nb_spaces[m] = {}
        for sp in wanted:
            if sp not in X:
                continue
            X_tr, X_va = X[sp]
            X_tr, X_va = _clean_matrix(X_tr), _clean_matrix(X_va)
            # The neighbour settings read every built space, the 8 detector
            # spaces included (base_1, hybrid, hybrid__general2 are settings).
            nb_spaces[m][sp] = np.vstack([X_tr, X_va])
            info = {'model': m, 'slot': sp}
            if sp.startswith('base_') and sp in CONFIGURATION_SPACES:
                info['representation'] = names.get(sp)
            else:
                rec = recipes.get(sp)
                info['recipe'] = names.get(sp)
                info.update((_recipe_summary(*rec) if rec else None) or {})
            if sp in CONFIGURATION_SPACES:
                detector_X[f'{sp}@{m}'] = (X_tr, X_va)
                info['kind'] = 'detector_space'
            else:
                info['kind'] = 'neighbour_space_only'
            space_info[f'{sp}@{m}'] = info
        print(f"  {m}: {sorted(sp for sp in X if sp in wanted)}"
              + (f"; unavailable: {sorted(unav)}" if unav else ''))

    rows = np.concatenate([train_idx, val_idx])
    y_fold = labels[rows]
    # A record's OTHER measurements of the same molecule are identical rows
    # in every feature space, so they are its own nearest neighbours and
    # would supply its label back to it. Every estimator that accepts
    # exclude_groups, and every neighbour list, is given the molecule
    # grouping. This is not the scaffold exclusion axis 6 cut.
    excl_fold = molecule_of_row[rows]
    is_val = np.zeros(len(rows), dtype=bool)
    is_val[len(train_idx):] = True
    fold_groups = (None if search_groups is None
                   else np.asarray(search_groups)[rows])
    smiles_rows = sx.smiles_for(data, rows)

    fold_scores, sim_fold_scores = {}, {}
    failures, copied_from, sim_copied_from = {}, {}, {}
    sim_unavailable, exclusion_note = {}, {}
    seconds_per_space, est_seconds = {}, {}
    first_with_signature, first_with_sim_signature = {}, {}
    # Neighbour lists of one model's settings on this fold. Every setting but
    # forest_proximity reads one of the model's spaces, so its lists serve
    # all four of that model's detector spaces; forest_proximity is fitted on
    # the detector space itself ('model_input') and is listed per space.
    lists_by_model = {}

    for space_name, (X_tr, X_va) in detector_X.items():
        model = space_name.rsplit('@', 1)[1]
        t_space = time.time()
        X_fold = np.vstack([X_tr, X_va])
        sig = _matrix_signature(X_tr, X_va)

        # ── the detectors' own neighbour search ──
        src = first_with_signature.get(sig)
        if src is not None:
            fold_scores[space_name] = {e: v.copy()
                                       for e, v in fold_scores[src].items()}
            failures[space_name] = dict(failures.get(src) or {})
            copied_from[space_name] = src
            print(f"    {space_name}: same features as {src}, copied")
        else:
            first_with_signature[sig] = space_name
            fold_scores[space_name] = {}
            est_seconds[space_name] = {}
            for est_name in estimators:
                X_e, y_e, excl_e, val_e = X_fold, y_fold, excl_fold, is_val
                g_e = fold_groups
                keep = None
                if (est_name in GP_SUBSAMPLED_ESTIMATORS
                        and len(rows) > GP_MAX_ROWS):
                    # Seeded from the fold number alone, so every space caps
                    # the SAME records and the difference between two spaces
                    # is not partly a difference in which rows ran.
                    rng = np.random.RandomState(42 + int(fold_i))
                    keep = np.sort(rng.permutation(len(rows))[:GP_MAX_ROWS])
                    X_e, y_e = X_fold[keep], y_fold[keep]
                    excl_e, val_e = excl_fold[keep], is_val[keep]
                    g_e = None if fold_groups is None else fold_groups[keep]
                out = np.full(len(val_idx), np.nan)
                t_est = time.time()
                try:
                    kw = _estimator_kwargs(est_name, groups=g_e,
                                           exclude=excl_e, is_clf=is_clf)
                    exclusion_note[est_name] = (
                        "its own molecule's other measurements are excluded "
                        "from its neighbours and its folds"
                        if 'exclude_groups' in kw
                        else ('scaffold-grouped folds' if 'groups' in kw
                              else 'nothing was excluded'))
                    est = get_noise_estimator(est_name, **kw)
                    s = np.asarray(est.estimate(X_e, y_e), dtype=float)
                    if keep is None:
                        out = s[val_e]
                    else:
                        # positions of the kept held-out rows among val_idx
                        out[keep[val_e] - len(train_idx)] = s[val_e]
                except Exception as exc:
                    failures.setdefault(space_name, {}).setdefault(
                        est_name, f'{type(exc).__name__}: {exc}')
                fold_scores[space_name][est_name] = out
                est_seconds[space_name][est_name] = round(
                    time.time() - t_est, 1)

        # ── the neighbour settings (kNN record) ──
        # The settings of a space's own model; the same matrix under another
        # model reads that model's settings, so the model is part of the
        # match here.
        sim_sig = (sig, model)
        src = first_with_sim_signature.get(sim_sig)
        if not (sweep and sim_estimators):
            pass
        elif src is not None:
            sim_fold_scores[space_name] = {
                sim: {e: v.copy() for e, v in d.items()}
                for sim, d in sim_fold_scores.get(src, {}).items()}
            sim_copied_from[space_name] = src
        else:
            first_with_sim_signature[sim_sig] = space_name
            sim_fold_scores[space_name] = {}
            lists_m = lists_by_model.setdefault(model, {})
            for setting in sweep:
                if setting.space == ss.FOREST_SPACE:
                    nb, why = sx.neighbour_lists(
                        dataset_name, setting, {'model_input': X_fold},
                        y_fold, exclude_groups=excl_fold,
                        smiles_train=smiles_rows, is_clf=is_clf)
                else:
                    if setting.name not in lists_m:
                        lists_m[setting.name] = sx.neighbour_lists(
                            dataset_name, setting, nb_spaces.get(model, {}),
                            y_fold, exclude_groups=excl_fold,
                            smiles_train=smiles_rows, is_clf=is_clf)
                    nb, why = lists_m[setting.name]
                if nb is None:
                    sim_unavailable.setdefault(
                        f'{setting.name}@{model}',
                        unavailable.get(f'{setting.space}@{model}') or why)
                    continue
                for k in sx.ks_held(nb):
                    for est_name in sim_estimators:
                        # DECIDED 2026-10-06 (Adelaide delegated): both
                        # counting schemes for the detectors that read one;
                        # the others once, under their own weighting.
                        schemes = (SCHEMES if scheme_readers.get(est_name)
                                   else (None,))
                        for scheme in schemes:
                            sim = (f'{sx.sim_name(setting, k)}|scheme='
                                   f'{scheme or OWN_SCHEME}')
                            nb_s = nb if scheme is None else dict(
                                nb, scheme=scheme)
                            try:
                                kw = _estimator_kwargs(
                                    est_name, groups=fold_groups,
                                    exclude=excl_fold, is_clf=is_clf)
                                kw.update(k=int(k), precomputed_neighbors=nb_s)
                                est = get_noise_estimator(est_name, **kw)
                                sc_ = np.asarray(
                                    est.estimate(X_fold, y_fold), dtype=float)
                            except Exception as exc:
                                sim_unavailable.setdefault(
                                    f'{sim}|{est_name}@{model}',
                                    f'{type(exc).__name__}: {exc}')
                                continue
                            sim_fold_scores[space_name].setdefault(sim, {})[
                                est_name] = sc_[is_val].astype(np.float32)
        seconds_per_space[space_name] = round(time.time() - t_space, 1)
        print(f"    {space_name}: scored [{seconds_per_space[space_name]:.0f}s]")

    meta = {
        'fold': int(fold_i),
        'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
        'models': sorted(entries),
        'configurations': configurations,
        'spaces': space_info,
        'unavailable': unavailable,
        'spaces_copied_from': copied_from,
        'similarity_axis_copied_from': sim_copied_from,
        'similarity_axis_unavailable': sim_unavailable,
        'failures': failures,
        'seconds_per_space': seconds_per_space,
        'seconds_per_estimator': est_seconds,
        'seconds_this_fold': round(time.time() - t_fold, 1),
        'exclusion_note': exclusion_note,
        'selection_scope': 'fold_train_rows',
    }
    return meta, fold_scores, sim_fold_scores


def run(data, config, output_dir, dataset_name, force_rerun=False, **kwargs):
    """N22: detector precision and recall against ChEMBL's curator flag."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N22: THE CURATOR FLAG AS GROUND TRUTH")
    print("=" * 70)

    labels = np.asarray(data['labels'], dtype=np.float64)
    n_samples = len(labels)
    task_type = data['task_type']
    is_clf = (task_type == 'classification')
    search_groups = data.get('groups')
    metric_name, _fn, lower_is_better = get_domain_metric(data)

    results = {
        # The dataset's own metric and direction: the per-fold searches that
        # chose the spaces rank in it. Every detection number below is
        # average precision against the flag, higher better.
        'metric': metric_name,
        'task_type': task_type,
        'lower_is_better': bool(lower_is_better),
        'detection_metric': 'average_precision',
        'detection_lower_is_better': False,
        'n_samples': int(n_samples),
        'what_one_row_is': 'one ChEMBL activity record, not one molecule',
        'ground_truth': 'chembl_data_validity_comment',
        'ground_truth_note': (
            "ChEMBL's curators write this field on records they found "
            "suspect. Nothing in this repository produced it, and no label "
            "here was moved. Every other detection number in the noise "
            "directory is scored against a list the same script drew."),
        'detection_condition': CONDITION,
        'result_layout': CHECKPOINT_IDENTITY,
    }

    # Before any search: every dataset but herg_ic50 stops here.
    flag, flag_text, flag_note = _flag_for_rows(data, dataset_name)
    if flag is None:
        print(f"  SKIP: {flag_note}")
        results.update({'skipped': True, 'reason': flag_note,
                        'policy': 'no_curator_flag_for_this_dataset'})
        return results
    if len(flag) != n_samples:
        reason = (f'{len(flag)} flags for {n_samples} rows')
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'flag_and_rows_disagree'})
        print(f"  SKIP: {reason}")
        return results

    results['how_the_flag_was_obtained'] = flag_note
    n_flag = int(flag.sum())
    texts = {}
    for t in flag_text:
        if t:
            texts[t] = texts.get(t, 0) + 1
    results['n_records_flagged'] = n_flag
    results['n_records'] = int(n_samples)
    results['flagged_share_of_records'] = n_flag / float(n_samples)
    results['flag_texts'] = dict(sorted(texts.items(), key=lambda kv: -kv[1]))
    print(f"  {n_flag} of {n_samples} records carry a curator comment")
    for t, c in results['flag_texts'].items():
        print(f"    {c} records: {t}")

    if n_flag == 0 or n_flag == n_samples:
        reason = (f'{n_flag} of {n_samples} records carry a flag, so '
                  f'precision is fixed by construction and says nothing')
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'flag_is_degenerate'})
        print(f"  SKIP: {reason}")
        return results

    # One molecule per repeat group, so the record-level numbers can be read
    # beside numbers where a molecule counts once.
    rep_groups = data.get('replicate_groups')
    if rep_groups is None:
        from kirby.datasets.herg import load_herg_ic50_measurements
        rep_groups = load_herg_ic50_measurements()['replicate_groups']
    molecule_of_row = np.asarray(rep_groups, dtype=np.int64)
    n_molecules = int(molecule_of_row.max()) + 1 if n_samples else 0
    flag_by_molecule = _by_molecule(flag, molecule_of_row, n_molecules, 'any')
    results['n_molecules'] = n_molecules
    results['n_molecules_with_a_flagged_record'] = int(flag_by_molecule.sum())
    results['flagged_share_of_molecules'] = (
        float(flag_by_molecule.sum()) / n_molecules if n_molecules else 0.0)
    print(f"  those sit on {int(flag_by_molecule.sum())} of the "
          f"{n_molecules} molecules")

    # What the flag is, in the terms that limit how it reads.
    results['what_the_flag_is_and_is_not'] = {
        'outside_typical_range_is_about_the_value': (
            f"{texts.get('Outside typical range', 0)} of the {n_flag} flagged "
            f"records carry 'Outside typical range', which ChEMBL assigns "
            f"because the measured value is extreme. A detector that scores a "
            f"record by the distance between its label and its neighbours' "
            f"labels is therefore graded partly against a property of the "
            f"label it was handed, not only against an independent verdict."),
        'the_two_texts_are_graded_apart_as_well': (
            "per_flag_text below holds each text's own three numbers. "
            "'Potential transcription error' is a handful of records and a "
            "number over a handful is not worth reading alone; it is there so "
            "nobody has to re-run this to see it."),
        'this_is_one_kind_of_noise': (
            "Owner's ruling, 2026-09-17. The flag is OUTLIER noise: a value "
            "out of range, or a number typed in wrong. A detector that wins "
            "here has not been shown to win on any other kind of noise, and "
            "a detector that loses here has not been shown to lose on any "
            "other kind. Nothing in this file is a result about noise in "
            "general."),
        'it_says_nothing_about_laboratory_variation': (
            "The variation between laboratories running the same experiment "
            "differently, and the variation within one laboratory between "
            "runs, are different quantities and are not measured here. n06 "
            "correlates the detectors against the disagreement between "
            "laboratories, n17 removes molecules on it and n19 uses the "
            "measurement count and the spread."),
        'what_is_not_claimed': (
            "an unflagged record is not a correct one. ChEMBL's curators "
            "flagged what they found; the rest were not certified. So recall "
            "is recall against what a curator caught, and a detector that "
            "ranks a genuinely wrong unflagged record first is punished for "
            "it here."),
    }

    estimators = resolve_estimators(is_clf=is_clf)
    results['estimators_run'] = list(estimators)
    results['estimator_list_source'] = (
        'n11_estimation_remediation_grid.BASE_ESTIMATORS, resolved through '
        'n06_bindingdb_ground_truth.resolve_estimators, imported so the three '
        'files cannot drift')
    print(f"  {len(estimators)} detectors: {', '.join(estimators)}")

    # ── what the spaces are (chosen per fold below) ──
    if kwargs.get('budget') is not None:
        print(f"  --budget {kwargs['budget']} is ignored: each fold's hybrids "
              f"are ranked over the dataset's decided budgets (2026-10-05)")
        results['budget_argument_ignored'] = int(kwargs['budget'])
    similarity_arm = bool(kwargs.get('similarity_arm', True))
    results['similarity_arm'] = similarity_arm
    results['spaces_per_model'] = {
        'detector_spaces': list(CONFIGURATION_SPACES),
        'neighbour_spaces_only': [sp for sp in NEIGHBOUR_ONLY_SPACES
                                  if similarity_arm
                                  or sp != 'hybrid_similarity'],
        'what_they_are': (
            "base_1, base_2: the model's 2 best single representations; "
            "hybrid, hybrid__general2: its 2 best general hybrids with "
            "different selected features; all four ranked by a search on the "
            "fold's training records. Every detector runs in these. "
            "hybrid__distance1 (the model's top hybrid of the fold's distance "
            "search; knn is one of its models, budgets up to 500 features) "
            "and hybrid_similarity (the top hybrid of the fold's similarity "
            "search, the same recipe for both models) are read only as "
            "neighbour settings. No downstream model is fitted in n22."),
    }
    results['selection_scope'] = 'fold_train_rows'

    # ── the neighbour settings (kNN record, Adelaide 2026-10-05) ──
    has_smiles = data.get('smiles') is not None
    # DECIDED 2026-10-06 (Adelaide delegated): herg_ic50's neighbour group is
    # set by its molecule count (9,802 molecules, so fast_knn), not by its
    # 12,008 records.
    group = ss.knn_group(dataset_name, n_molecules)
    sweep = sx.sweep_settings(dataset_name=dataset_name,
                              n_molecules=n_molecules, has_smiles=has_smiles)
    if not similarity_arm:
        sweep = [s_ for s_ in sweep if s_.space != 'hybrid_similarity']
    _nb_params = {}
    for e in estimators:
        try:
            _nb_params[e] = 'precomputed_neighbors' in inspect.signature(
                type(get_noise_estimator(e)).__init__).parameters
        except Exception:
            _nb_params[e] = False
    sim_estimators = [e for e in estimators if _nb_params.get(e)]
    scheme_readers = {e: reads_scheme(e) for e in sim_estimators}
    results['similarity_axis_settings'] = {
        'knn_group': group,
        'molecules_counted_for_the_group': int(n_molecules),
        'settings_swept': [s_.name for s_ in sweep],
        'ks_swept': list(ss.SWEPT_KS) if sweep else [],
        'estimators': sim_estimators,
        # TODO(similarity-sweep 2026-09-25): a neighbour-based estimator whose
        # class takes no precomputed_neighbors (n11's 'hybrid' estimator,
        # when it resolves) still finds its own neighbours in the space's X
        # and is not on the similarity axis. Adding it needs
        # precomputed_neighbors on that class in
        # src/kirby/utils/noise_estimation.py, outside this file.
        'estimators_not_swept': [e for e in estimators
                                 if not _nb_params.get(e)],
        'screen': ('none: n22 refits no model per setting, so every setting '
                   'is scored (kNN record, screen decision)'),
        'scaffold_mates': ("kept in every list (axis 6); only a record's own "
                           "other measurements of the same molecule are "
                           "excluded"),
        # DECIDED 2026-10-06 (Adelaide delegated): sweep both counting
        # schemes for the neighbour detectors, by handing them
        # dict(lists, scheme=s). The detector reads the scheme since the
        # 2026-10-06 change to src/kirby/utils/noise_estimation.py.
        'counting_schemes_swept': list(SCHEMES),
        'detectors_swept_over_the_scheme': [
            e for e in sim_estimators if scheme_readers.get(e)],
        'detectors_with_their_own_weighting': {
            e: ("weights its neighbours by one over the distance itself and "
                "reads no scheme; scored once per setting and k, keyed "
                "'scheme=own'")
            for e in sim_estimators if not scheme_readers.get(e)},
        'default_path': ("the detector's own neighbour search in the space "
                         "itself, at its own default k and distance; the "
                         "numbers under per_space"),
    }
    print(f"  neighbour settings ({group}): {len(sweep)} x ks "
          f"{list(ss.SWEPT_KS)} x schemes {list(SCHEMES)}, detectors "
          f"{sim_estimators}")

    # ── folds ──
    folds, n_folds = noise_outer_folds(data, config)
    results['n_folds'] = int(n_folds)
    results['folds_run'] = [int(f) for f, _, _ in folds]
    print(f"  {len(folds)} of {n_folds} outer folds "
          f"(kirby.find_best_hybrid.nested_cv_outer_folds, scaffold-grouped, "
          f"so a molecule's repeats never sit on both sides)")
    fold_of_row = np.full(n_samples, -1, dtype=np.int64)
    for fold_i, _tr, va in folds:
        fold_of_row[va] = fold_i
    n_straddle = int(sum(
        1 for m in np.unique(molecule_of_row)
        if len(np.unique(fold_of_row[molecule_of_row == m])) > 1))
    results['molecules_whose_records_span_more_than_one_fold'] = n_straddle

    # Written per fold as it finishes: one checkpoint line and one score file
    # per fold. A killed run redoes only the fold in flight.
    from experiments.noise.n07_remediation import FoldCheckpoint
    from experiments.noise.noise_cache import array_hash
    fingerprint = {
        'identity': CHECKPOINT_IDENTITY, 'dataset': dataset_name,
        'n_samples': int(n_samples), 'n_folds': int(n_folds),
        'labels': array_hash(labels), 'flag': array_hash(flag),
        'molecules': array_hash(molecule_of_row),
        'estimators': list(estimators),
        'settings': [s_.name for s_ in sweep],
        'schemes': list(SCHEMES),
        'scheme_readers': sorted(e for e, v in scheme_readers.items() if v),
        'similarity_arm': similarity_arm,
    }
    fp_tag = array_hash(np.frombuffer(
        repr(sorted(fingerprint.items())).encode(), dtype=np.uint8))[:12]
    ckpt = FoldCheckpoint(output_dir, EXP_FULL_NAME, dataset_name,
                          tag='fold_ckpt', fingerprint=fingerprint)
    metric = (metric_name, _fn, lower_is_better)

    scores = {}       # space -> estimator -> score per record (NaN = unscored)
    sim_scores = {}   # space -> '<setting>|k=<k>' -> estimator -> score
    per_fold = {}

    def _put(sp, est, vi, v, sim=None):
        if sim is None:
            arr = scores.setdefault(sp, {}).setdefault(
                est, np.full(n_samples, np.nan))
        else:
            arr = sim_scores.setdefault(sp, {}).setdefault(sim, {}).setdefault(
                est, np.full(n_samples, np.nan))
        arr[vi] = v

    for fold_i, train_idx, val_idx in folds:
        done = None if force_rerun else ckpt.get(['fold', int(fold_i)])
        if done is not None and os.path.exists(str(done.get('npz') or '')):
            meta = done['meta']
            z = np.load(done['npz'])
            vi = z['val_idx']
            for key, ident in (meta.get('npz_keys') or {}).items():
                sp, est, sim = ident
                _put(sp, est, vi, z[key], sim)
            per_fold[int(fold_i)] = meta
            print(f"\n  fold {fold_i}: read back from {done['npz']}")
            continue

        meta, fold_scores, sim_fold_scores = _score_one_fold(
            data, dataset_name, labels, molecule_of_row, estimators,
            sim_estimators, sweep, is_clf, search_groups, metric,
            fold_i=int(fold_i), train_idx=train_idx, val_idx=val_idx,
            n_folds=n_folds, scheme_readers=scheme_readers)

        payload = {'val_idx': np.asarray(val_idx)}
        npz_keys = {}
        j = 0
        for sp, d in fold_scores.items():
            for est, v in d.items():
                payload[f's{j}'] = v
                npz_keys[f's{j}'] = [sp, est, None]
                j += 1
        for sp, by_sim in sim_fold_scores.items():
            for sim, d in by_sim.items():
                for est, v in d.items():
                    payload[f's{j}'] = v
                    npz_keys[f's{j}'] = [sp, est, sim]
                    j += 1
        meta['npz_keys'] = npz_keys
        # The held-out grading of this fold alone, per space and detector, so
        # the spread over folds can be read beside the pooled number.
        vi = np.asarray(val_idx)
        meta['held_out_average_precision'] = {
            sp: {est: (grade(flag[vi], v, label='records') or {}).get(
                'detection_average_precision')
                for est, v in d.items()}
            for sp, d in fold_scores.items()}
        meta = convert_numpy(meta)
        if output_dir is not None:
            npz_path = os.path.join(
                str(output_dir), EXP_FULL_NAME,
                f'{dataset_name}.fold{fold_i}.{fp_tag}.npz')
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez_compressed(npz_path, **payload)
            ckpt.put(['fold', int(fold_i)], {'meta': meta, 'npz': npz_path})
        for key, (sp, est, sim) in npz_keys.items():
            _put(sp, est, vi, payload[key], sim)
        per_fold[int(fold_i)] = meta

    # One entry per fold: each model's configurations with the fold search's
    # own training-row scores, which representation or recipe filled each
    # space, what was unavailable and why, copies, failures, timings, and the
    # fold's own held-out average precision.
    results['per_fold'] = {str(k): {kk: vv for kk, vv in v.items()
                                    if kk != 'npz_keys'}
                           for k, v in per_fold.items()}
    results['held_out_model_score'] = (
        "none: n22 fits no downstream model. The per-fold searches' "
        "training-row scores are under per_fold[<i>]['configurations'] in "
        "the dataset's metric; the held-out numbers are average precision "
        "against the curator flag.")

    spaces = sorted(scores)
    space_model = {sp: sp.rsplit('@', 1)[1] for sp in spaces if '@' in sp}
    base_spaces = [sp for sp in spaces
                   if sp.split('@', 1)[0].startswith('base_')]
    hybrid_spaces = [sp for sp in spaces if sp not in base_spaces]
    results['space_model'] = space_model
    results['comparator_spaces'] = base_spaces
    if not spaces:
        reason = ('no feature space to score in on any fold: ' + ' | '.join(
            f"fold {k}: {v.get('unavailable')}" for k, v in per_fold.items()))
        print(f"  SKIP: {reason}")
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'no_feature_space'})
        return results

    exclusion_note = {}
    for v in per_fold.values():
        exclusion_note.update(v.get('exclusion_note') or {})
    results['replicate_exclusion'] = exclusion_note
    results['how_a_record_was_scored'] = {
        'n_folds': int(n_folds),
        'folds': ('kirby.find_best_hybrid.nested_cv_outer_folds on the '
                  'scaffold groups, the folds the per-fold searches are filed '
                  "under; a molecule and its whole scaffold family sit on one "
                  "side"),
        'features': ("each fold's representations and hybrid recipes are "
                     "chosen by searches on that fold's training records, and "
                     "each hybrid's features are selected on those records, "
                     "so no held-out record helps choose the space its own "
                     "suspicion score is computed in"),
        'scaling': ("fitted on the fold's training records. Single "
                    "representations and hybrids: kirby.scaling."
                    "FeatureScaler (0/1 features unscaled, the rest "
                    "standardised)"),
        'rows_the_detector_was_given': (
            "every record, mapped into that fold's space. The detectors score "
            "exactly the rows they are handed, so what the fold controls is "
            "the feature space and which records keep a score."),
        'rows_that_kept_a_score': "the fold's held-out records only",
    }

    def _grade_all(sus, vi_by_fold):
        row = {'records': grade(flag, sus, label='records')}
        sus_mol = _by_molecule(sus, molecule_of_row, n_molecules, 'max')
        row['molecules'] = grade(flag_by_molecule, sus_mol, label='molecules')
        # Per fold, then the spread over folds (no significance test).
        row['records_per_fold_average_precision'] = fold_spread([
            (grade(flag[vi], sus[vi], label='records') or {}).get(
                'detection_average_precision')
            for vi in vi_by_fold])
        return row

    vi_by_fold = [np.asarray(va) for _f, _tr, va in folds]

    # ── the three numbers, per space and per detector ──
    per_space = {}
    for space_name in spaces:
        per_est = {}
        for est_name in estimators:
            sus = scores[space_name].get(est_name,
                                         np.full(n_samples, np.nan))
            row = _grade_all(sus, vi_by_fold)
            # Each flag text on its own, against the same scores. A record
            # carrying a different text is not counted as a negative: it is
            # dropped, because a curator did mark it and calling it clean
            # would put a right answer in the wrong column.
            per_text = {}
            for t, c in results['flag_texts'].items():
                this = np.array([ft == t for ft in flag_text], dtype=bool)
                other = flag & ~this
                keep = ~other
                g = grade(this[keep], sus[keep], label='records')
                if g is not None:
                    g['flag_text'] = t
                    g['n_records_with_another_flag_text_dropped'] = int(
                        other.sum())
                    if c < 30:
                        g['too_few_to_read'] = (
                            f'{c} records carry this text; a precision over '
                            f'{c} records is not a number worth reading on '
                            f'its own')
                per_text[t] = g
            row['per_flag_text'] = per_text
            per_est[est_name] = row
        per_space[space_name] = per_est
    results['per_space'] = convert_numpy(per_space)

    # ── the neighbour settings, graded the same way ──
    if sim_scores:
        results['similarity_axis'] = convert_numpy({
            sp: {sim: {est: _grade_all(np.asarray(v, dtype=float), vi_by_fold)
                       for est, v in d.items()}
                 for sim, d in by_sim.items()}
            for sp, by_sim in sim_scores.items()})
        results['similarity_axis_note'] = (
            "similarity_axis[space]['<setting>|k=<k>|scheme=<s>'][detector] "
            "holds the same grading as per_space; <s> is the counting scheme, "
            "or 'own' for a detector that weights neighbours its own way. "
            "The detector's neighbours are found in that setting among the fold's records. The setting's space is "
            "the space's own model's per-fold one. The detector's own features "
            "are still the space's.")

    # ── each model's hybrids minus that model's single representations ──
    delta = {}
    for base_sp in base_spaces:
        per_hybrid = {}
        for hs in hybrid_spaces:
            if space_model.get(hs) != space_model.get(base_sp):
                continue
            d = {}
            for est_name in estimators:
                a = (per_space[hs][est_name]['records'] or {}).get(
                    'detection_average_precision')
                b = (per_space[base_sp][est_name]['records'] or {}).get(
                    'detection_average_precision')
                if a is None or b is None or not (np.isfinite(a)
                                                  and np.isfinite(b)):
                    continue
                per_fold_diff = []
                for vi in vi_by_fold:
                    ga = grade(flag[vi], scores[hs][est_name][vi],
                               label='records')
                    gb = grade(flag[vi], scores[base_sp][est_name][vi],
                               label='records')
                    if ga and gb:
                        per_fold_diff.append(
                            ga['detection_average_precision']
                            - gb['detection_average_precision'])
                d[est_name] = {
                    'average_precision_in_the_hybrid': float(a),
                    f'average_precision_in_{base_sp}': float(b),
                    'difference': float(a - b),
                    'difference_per_fold': fold_spread(per_fold_diff)}
            per_hybrid[hs] = d
        delta[base_sp] = {'model': space_model.get(base_sp),
                          'per_hybrid_space': per_hybrid}
    results['hybrid_minus_single_representation'] = convert_numpy(delta)
    results['hybrid_minus_single_representation_note'] = (
        'both numbers are average precision against the curator flag over '
        'records, and both are printed so the difference is never read '
        'without the pair it came from. A slot is a rank: which '
        'representation or recipe filled it on each fold is in per_fold.')

    # ── the ordering, so it can be put beside n11's ──
    best = {}
    for space_name in spaces:
        ranked = []
        for est_name in estimators:
            g = per_space[space_name][est_name]['records']
            if g and np.isfinite(g.get('detection_average_precision', np.nan)):
                ranked.append((_canonical(est_name),
                               float(g['detection_average_precision'])))
        ranked.sort(key=lambda kv: -kv[1])
        best[space_name] = ranked
        if ranked:
            print(f"\n  {space_name}, average precision against the flag "
                  f"({n_flag} flagged records out of {n_samples} scored; "
                  f"random ranking would score "
                  f"{n_flag / float(n_samples):.4f}):")
            for name, ap in ranked:
                print(f"    {ap:.4f}  {name}")
    results['average_precision_ranking'] = convert_numpy(best)

    # Every per-record score, so a re-analysis costs nothing.
    results['per_record'] = convert_numpy({
        'curator_flag': flag.astype(int).tolist(),
        'molecule_id': molecule_of_row.tolist(),
        'fold': fold_of_row.tolist(),
        'suspicion': {sp: {e: scores[sp][e].tolist() for e in scores[sp]}
                      for sp in spaces},
    })
    # n22's result file is written by its caller (run_all or main), not here.
    # The per-fold lines and score files stay, so a kill between this return
    # and that write costs nothing but the assembly.
    return results


def main():
    parser = create_experiment_parser(
        'N22: detector precision and recall against ChEMBL\'s curator flag')
    parser.add_argument('--no-similarity-arm', action='store_true',
                        help='leave the similarity-search hybrid out of the '
                             'neighbour settings')
    args = parser.parse_args()
    config = get_config(args)
    data = load_data(args.dataset, rep_preset=args.reps)
    results = run(data, config, args.output, args.dataset,
                  similarity_arm=not args.no_similarity_arm,
                  budget=getattr(args, 'budget', None))
    save_experiment_results(results, args.output, 'n22_curator_flag',
                            args.dataset)


if __name__ == '__main__':
    main()
