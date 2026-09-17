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

TWO FEATURE SPACES, the same pair n06 uses. One single representation, chosen
as the best in the pool lookup that chose the hybrid recipe — never a
concatenation of everything, which is not a baseline in this project. And the
hybrid built from that same pool. Features are chosen inside each fold on the
other folds' rows, and a record keeps a score only in the fold that held it
out.

THE DETECTOR LIST IS n11's, imported rather than copied, for the reason
recorded at n06_bindingdb_ground_truth.py:87-92 — two hand-kept lists drift,
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
gets a recorded skip with the reason, not a crash.

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
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments._shared import (  # noqa: E402
    load_data, get_config, create_experiment_parser,
    save_experiment_results, convert_numpy,
    make_cv_splitter,
    extract_top3_base_reps, pick_noise_base_rep, get_domain_metric,
    POOL_SUBSETS,
)
from kirby.utils.noise_estimation import get_noise_estimator  # noqa: E402

from noise_helpers import detection_scores  # noqa: E402

# n06's, imported for the same reason n06 imports n11's estimator list: a
# second hand-kept copy of how a space is named, how a recipe is fetched and
# how a fold's hybrid is built would drift, and then n06's table and this one
# could not be read against each other. `resolve_estimators` resolves n11's
# BASE_ESTIMATORS plus the torch and optional ones when they import.
from n06_bindingdb_ground_truth import (  # noqa: E402
    resolve_estimators, _canonical, _clean_matrix, _blueprint_for,
    _fold_hybrid_pair, _space_name, SPACE_FOR_FAMILY, FEATURES_CHOSEN_BY,
    POOL_SUBSET_WITH_SIMILARITY, POOL_SUBSET_WITHOUT_SIMILARITY,
    GP_SUBSAMPLED_ESTIMATORS, GP_MAX_ROWS,
)

N_FOLDS = 5

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


def run(data, config, output_dir, dataset_name, **kwargs):
    """N22: detector precision and recall against ChEMBL's curator flag."""
    print("\n" + "=" * 70)
    print("EXPERIMENT N22: THE CURATOR FLAG AS GROUND TRUTH")
    print("=" * 70)

    reps = data['reps']
    labels = np.asarray(data['labels'], dtype=np.float64)
    n_samples = len(labels)
    task_type = data['task_type']
    is_clf = (task_type == 'classification')
    augmentations = data.get('augmentations')
    search_groups = data.get('groups')
    metric_name, _fn, lower_is_better = get_domain_metric(data)

    results = {
        'metric': metric_name,
        'task_type': task_type,
        'lower_is_better': bool(lower_is_better),
        'n_samples': int(n_samples),
        'what_one_row_is': 'one ChEMBL activity record, not one molecule',
        'ground_truth': 'chembl_data_validity_comment',
        'ground_truth_note': (
            "ChEMBL's curators write this field on records they found "
            "suspect. Nothing in this repository produced it, and no label "
            "here was moved. Every other detection number in the noise "
            "directory is scored against a list the same script drew."),
        'detection_condition': CONDITION,
    }

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

    # ── feature spaces: one single representation and the hybrid ──
    budget = kwargs.get('budget')
    spaces = {}
    space_provenance = {}
    pool_subset = (POOL_SUBSET_WITH_SIMILARITY
                   if kwargs.get('similarity_arm', True)
                   else POOL_SUBSET_WITHOUT_SIMILARITY)
    eligible_families = POOL_SUBSETS[pool_subset]
    results['pool_subset'] = pool_subset
    results['pool_subset_families'] = list(eligible_families)
    results['hybrid_family'] = None

    t_pool = time.time()
    bp, aug_cond, meta, search = _blueprint_for(
        reps, labels, search_groups, data, dataset_name, task_type,
        augmentations, budget, pool_subset=pool_subset)
    results['seconds_to_resolve_the_pool'] = round(time.time() - t_pool, 1)

    winner_family = None
    pool_failure = None
    if bp is None:
        pool_failure = meta
        print(f"  no hybrid recipe from the pool subset {pool_subset}: {meta}")
    else:
        winner_family = meta.get('objective', 'predictive')
        results['hybrid_family'] = winner_family
        results['hybrid_family_choice'] = meta.get('objective_choice')
        results['hybrid_metric'] = meta.get('metric_name')
        entries = [(winner_family, aug_cond, meta.get('tier'), bp,
                    meta.get('score_mean'), meta.get('metric_name'))]
        for o, d in (meta.get('by_objective') or {}).items():
            if o == winner_family:
                continue
            o_bp = (d.get('best_config') or {}).get('blueprint')
            if o_bp is None:
                space_provenance[_space_name(o)] = {
                    'unavailable': f'the best {o} pool entry carries no '
                                   f'blueprint'}
                continue
            entries.append((o, d.get('aug_condition'), d.get('tier'), o_bp,
                            d.get('score_mean'), d.get('metric_name')))
        for o, o_aug, o_tier, o_bp, o_score, o_metric in entries:
            name = _space_name(o)
            spaces[name] = ('hybrid', o_bp, o_aug)
            space_provenance[name] = {
                'built_under_objective': o,
                'is_the_recipe_the_pool_returned': (o == winner_family),
                'features_chosen_by': FEATURES_CHOSEN_BY.get(o, o),
                'which_features': "chosen inside each fold, on the other "
                                  "folds' rows",
                'tier': o_tier, 'aug_condition': o_aug,
                'pool_score': o_score, 'pool_score_metric': o_metric,
                'budget': int(getattr(o_bp, 'budget', 0) or 0),
                'allocation_method': getattr(o_bp, 'allocation_method', None),
                'importance_method': getattr(o_bp, 'importance_method', None),
            }
    for o in eligible_families:
        name = _space_name(o)
        if name not in spaces and name not in space_provenance:
            space_provenance[name] = {'unavailable': (
                pool_failure if pool_failure is not None
                else f'no {o} search in the pool produced a result for rf')}
    for o, name in SPACE_FOR_FAMILY.items():
        if o not in eligible_families and name not in spaces:
            space_provenance[name] = {'unavailable': (
                f'the {o} family is not in the pool subset {pool_subset}')}

    base_rep = None
    if search is not None:
        try:
            top3 = extract_top3_base_reps(search)
            if top3:
                results['top3_base_reps'] = [(n, sc) for n, sc in top3]
                if top3[0][0] in reps:
                    base_rep = top3[0][0]
        except Exception as exc:
            print(f"  no base-representation ranking off the search "
                  f"({type(exc).__name__}: {exc})")
    if base_rep is None:
        base_rep = pick_noise_base_rep(reps, dataset_name)
        ranked_in = ('a fixed preference order; the pool search gave no '
                     'usable ranking of the single representations')
    else:
        ranked_in = ('the best single representation in the same search that '
                     'chose the hybrid recipe, so both arms are ranked on one '
                     'set of folds')
    print(f"  comparator representation: {base_rep}")
    spaces[base_rep] = ('base', None, None)
    space_provenance[base_rep] = {
        'features_chosen_by': 'nothing — every feature of one representation',
        'which_representation': ranked_in,
        'scaling': "z-scored inside each fold on that fold's training rows, "
                   "the same treatment the hybrid arm gets",
    }
    results['comparator_space'] = base_rep
    results['space_provenance'] = convert_numpy(space_provenance)

    # ── score every record, inside its own fold ──
    splitter, use_groups = make_cv_splitter(N_FOLDS, search_groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))
    split_args = ((sample_X, labels, search_groups) if use_groups
                  else (sample_X, labels))
    folds = list(splitter.split(*split_args))
    print(f"  {len(folds)} folds, scaffold-grouped, so a molecule's repeats "
          f"never sit on both sides")

    base_matrix = _clean_matrix(reps[base_rep])
    scores = {s: {e: np.full(n_samples, np.nan) for e in estimators}
              for s in spaces}
    failures = {s: {} for s in spaces}
    est_seconds = {s: {e: 0.0 for e in estimators} for s in spaces}
    timings = {}
    exclusion_note = {}

    for space_name, (kind, blueprint, aug) in spaces.items():
        print(f"\n  === {space_name} ===")
        t_space = time.time()
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            t_fold = time.time()
            if kind == 'hybrid':
                try:
                    X_tr, X_va = _fold_hybrid_pair(
                        train_idx, val_idx, reps, labels, search_groups,
                        blueprint, aug, augmentations)
                except Exception as exc:
                    failures[space_name][f'fold_{fold_i}'] = (
                        f'{type(exc).__name__}: {exc}')
                    print(f"    fold {fold_i}: build failed "
                          f"({type(exc).__name__}: {exc})")
                    continue
            else:
                X_tr, X_va = base_matrix[train_idx], base_matrix[val_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)

            rows = np.concatenate([train_idx, val_idx])
            X_fold = np.vstack([X_tr, X_va])
            y_fold = labels[rows]
            # A record's OTHER measurements of the same molecule are identical
            # rows in every feature space, so they are its own nearest
            # neighbours and would supply its label back to it. Every
            # estimator that accepts exclude_groups is given the molecule
            # grouping, the same way n06 gives it the repeat grouping.
            excl_fold = molecule_of_row[rows]
            is_val = np.zeros(len(rows), dtype=bool)
            is_val[len(train_idx):] = True
            fold_groups = (np.asarray(search_groups)[rows]
                           if search_groups is not None else None)

            for est_name in estimators:
                X_e, y_e, rows_e = X_fold, y_fold, rows
                excl_e, val_e = excl_fold, is_val
                g_e = fold_groups
                if (est_name in GP_SUBSAMPLED_ESTIMATORS
                        and len(rows) > GP_MAX_ROWS):
                    # Seeded from the fold number alone, so every space caps
                    # the SAME records and the difference between the two
                    # spaces is not partly a difference in which rows ran.
                    rng = np.random.RandomState(42 + fold_i)
                    keep = np.sort(rng.permutation(len(rows))[:GP_MAX_ROWS])
                    X_e, y_e, rows_e = X_fold[keep], y_fold[keep], rows[keep]
                    excl_e, val_e = excl_fold[keep], is_val[keep]
                    g_e = None if fold_groups is None else fold_groups[keep]

                t_est = time.time()
                try:
                    est_kwargs = {}
                    params = inspect.signature(
                        type(get_noise_estimator(est_name)).__init__).parameters
                    if g_e is not None and 'groups' in params:
                        est_kwargs['groups'] = g_e
                    if 'exclude_groups' in params:
                        est_kwargs['exclude_groups'] = excl_e
                    if is_clf and 'is_clf' in params:
                        est_kwargs['is_clf'] = True
                    exclusion_note[est_name] = (
                        "its own molecule's other measurements are excluded "
                        "from its neighbours and its folds"
                        if 'exclude_groups' in est_kwargs
                        else ('scaffold-grouped folds'
                              if 'groups' in est_kwargs
                              else 'nothing was excluded'))
                    est = get_noise_estimator(est_name, **est_kwargs)
                    s = np.asarray(est.estimate(X_e, y_e), dtype=float)
                    scores[space_name][est_name][rows_e[val_e]] = s[val_e]
                except Exception as exc:
                    failures[space_name].setdefault(est_name, []).append(
                        f'fold {fold_i}: {type(exc).__name__}: {exc}')
                est_seconds[space_name][est_name] += time.time() - t_est
            print(f"    fold {fold_i}: {len(train_idx)} records chose the "
                  f"features, {len(val_idx)} records scored "
                  f"[{time.time() - t_fold:.0f}s]")
        timings[space_name] = round(time.time() - t_space, 1)

    results['seconds_per_space'] = timings
    results['seconds_per_estimator'] = {
        sp: {e: round(v, 1) for e, v in sorted(d.items(), key=lambda kv: -kv[1])}
        for sp, d in est_seconds.items()}
    results['failures'] = convert_numpy(failures)
    results['replicate_exclusion'] = exclusion_note
    results['how_a_record_was_scored'] = {
        'n_folds': len(folds),
        'folds': 'scaffold-grouped, so a molecule and its whole scaffold '
                 'family sit on one side',
        'features': "each fold's features are chosen on the other folds' "
                    "rows, so no record helps choose the features its own "
                    "suspicion score is computed in",
        'rows_that_kept_a_score': 'the held-out fifth only',
    }

    # ── the three numbers, per space and per detector ──
    per_space = {}
    for space_name in spaces:
        per_est = {}
        for est_name in estimators:
            sus = scores[space_name][est_name]
            row = {'records': grade(flag, sus, label='records')}
            sus_mol = _by_molecule(sus, molecule_of_row, n_molecules, 'max')
            row['molecules'] = grade(flag_by_molecule, sus_mol,
                                     label='molecules')
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

    # ── the hybrid minus the single representation, per detector ──
    hybrid_spaces = [s for s, (k, _b, _a) in spaces.items() if k == 'hybrid']
    delta = {}
    for hs in hybrid_spaces:
        d = {}
        for est_name in estimators:
            a = (per_space[hs][est_name]['records'] or {}).get(
                'detection_average_precision')
            b = (per_space[base_rep][est_name]['records'] or {}).get(
                'detection_average_precision')
            if a is None or b is None or not (np.isfinite(a)
                                              and np.isfinite(b)):
                continue
            d[est_name] = {
                'average_precision_in_the_hybrid': float(a),
                f'average_precision_in_{base_rep}': float(b),
                'difference': float(a - b)}
        delta[hs] = d
    results['hybrid_minus_single_representation'] = convert_numpy(delta)
    results['hybrid_minus_single_representation_note'] = (
        'both numbers are average precision against the curator flag over '
        'records, and both are printed so the difference is never read '
        'without the pair it came from')

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
        'suspicion': {sp: {e: scores[sp][e].tolist() for e in estimators}
                      for sp in spaces},
    })
    return results


def main():
    parser = create_experiment_parser(
        'N22: detector precision and recall against ChEMBL\'s curator flag')
    parser.add_argument('--no-similarity-arm', action='store_true',
                        help='narrow the pool subset to prediction-built '
                             'searches alone')
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
