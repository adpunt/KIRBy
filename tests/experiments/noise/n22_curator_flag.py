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

THE FEATURE SPACES ARE n06's. Two single representations, `base_1` and
`base_2`: rf's top 2 by rf's score in the stored search the pool lookup
returned, the lookup that chose the hybrid recipe (owner's decision
2026-09-25) — never a concatenation of everything, which is not a baseline in
this project. Where that search gives no rf ranking, both are skipped and
`base_reps_missing` says why; there is no fallback list. And the hybrid built
from that same pool. Features are chosen inside each fold on the
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
    extract_top3_base_reps, get_domain_metric,
    POOL_SUBSETS, DOWNSTREAM_MODELS,
    noise_base_reps_by_model, NOISE_BASE_REP_COUNT,
    top_base_reps, top_hybrids, select_model_rep_pairs, hybrid_deltas,
    get_best_search_for_downstream,
    searched_models, resolve_comparison_models,
)
from kirby.utils.noise_estimation import get_noise_estimator  # noqa: E402
# The shared similarity axis and the per-fold checkpoint, added 2026-09-25
# (plan sections 1-3). n22 changes no training data -- it grades detectors
# against a flag -- so it writes no D-versus-D' rows (plan section 6).
from experiments.noise import _sim_axis_n15_n22 as sx  # noqa: E402
from experiments.noise.similarity_space import SWEPT_KS  # noqa: E402

# The name the checkpoint is filed under.
EXP_FULL_NAME = 'n22_curator_flag'

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

    # ── feature spaces: rf's top 2 single representations and the hybrid ──
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

    # rf's rank-1 and rank-2 single representations by rf's score in the
    # stored search the pool returned, as spaces base_1 and base_2 (owner's
    # decision 2026-09-25). No fallback list: with no rf ranking the
    # single-representation spaces are skipped and the reason is written.
    if search is not None:
        try:
            top3 = extract_top3_base_reps(search)
            if top3:
                results['top3_base_reps'] = [(n, sc) for n, sc in top3]
        except Exception as exc:
            print(f"  no top-3 listing off the search "
                  f"({type(exc).__name__}: {exc})")
        base_by_model, base_missing = noise_base_reps_by_model(
            {'rf': search}, ['rf'], reps=reps, dataset_name=dataset_name)
    else:
        base_by_model = {}
        base_missing = {'rf': (
            f'the pool lookup returned no stored search ({pool_failure}), so '
            f'there is no rf ranking of the single representations')}
    results['base_reps_by_model'] = base_by_model
    results['base_reps_missing'] = base_missing
    base_spaces = {}
    for _k, _rn in enumerate(base_by_model.get('rf', [])
                             [:NOISE_BASE_REP_COUNT], start=1):
        base_spaces[f'base_{_k}'] = _rn
    for _sp, _rn in base_spaces.items():
        spaces[_sp] = ('base', _rn, None)
        space_provenance[_sp] = {
            'representation': _rn,
            'features_chosen_by': 'nothing — every feature of one '
                                  'representation',
            'which_representation': (
                f"rf's rank-{_sp.split('_')[1]} single representation by rf's "
                f"score in the stored search the pool returned (built under "
                f"the {winner_family} objective), the same search that chose "
                f"the hybrid recipe"),
            'scaling': "z-scored inside each fold on that fold's training rows, "
                       "the same treatment the hybrid arm gets",
        }
        print(f"  {_sp}: {_rn} (rf's rank {_sp.split('_')[1]})")
    if not base_spaces:
        print(f"  no single-representation space: {base_missing.get('rf')}")
    results['comparator_spaces'] = list(base_spaces)
    results['space_provenance'] = convert_numpy(space_provenance)

    if not spaces:
        reason = ('no feature space to score in: no hybrid recipe came back '
                  f'({pool_failure}) and no single representation was ranked '
                  f'({base_missing.get("rf")})')
        print(f"  SKIP: {reason}")
        results.update({'skipped': True, 'reason': reason,
                        'policy': 'no_feature_space'})
        return results

    # ── the top 2, added beside what is already reported ───────────────────
    # Added 2026-09-18, owner's instruction: the top 2 rows everywhere as an
    # extra, changing nothing that is already reported. It reads rankings only;
    # the spaces the detectors run in are base_1, base_2 and the hybrid arms
    # above.
    #
    # What it adds: rf's ranking alone decides base_1 and base_2, and the
    # file would otherwise report no score for any other downstream model, how
    # close the ranks sat, or how the other 4 downstream models ranked the
    # same representations.
    #
    # 0 new model fits, and no detector fits either. All four helpers read rows
    # the pool search already holds in search.baselines and search.all_results.
    # n22 fits no downstream model at all; these numbers come from the search
    # that chose the hybrid recipe, in that search's own metric
    # ('hybrid_metric' above), not in average precision against the flag.
    top2_base_reps, top2_hybrids = {}, {}
    hybrid_deltas_by_model, top2_pairs_by_model = {}, []
    if search is not None:
        # ONE SEARCH PER DOWNSTREAM MODEL. `search` came back from a pool
        # lookup that asked for rf, and every model's top 2 used to be read out
        # of it. The pool picks the best stored configuration FOR the model it
        # is asked about, so rf's search and lightgbm's can be different
        # members of the pool. Each model is now asked for separately and read
        # out of its own answer; there is no fallback to rf's, because a row
        # labelled for one model that holds another model's configuration is
        # the defect this loop is being fixed for. The lookups read the stored
        # searches the rf lookup already loaded, so they cost 0 model fits.
        _searches = {}
        _search_lookup_failed = {}
        for _m in DOWNSTREAM_MODELS:
            if _m == 'rf':
                _searches[_m] = search
                continue
            try:
                _s_m, _ = get_best_search_for_downstream(
                    reps, labels, search_groups,
                    downstream_model=_m, budget=budget,
                    augmentations=augmentations,
                    data=data,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    return_fitted=False,
                    pool_subset=pool_subset,
                )
            except Exception as _e:
                _search_lookup_failed[_m] = f'{type(_e).__name__}: {_e}'
                continue
            if _s_m is not None:
                _searches[_m] = _s_m

        # WHICH DOWNSTREAM MODELS THIS TABLE COVERS. Two lists decide it and
        # they can disagree. DOWNSTREAM_MODELS (_shared.py) is the five names
        # this file asks about. The other list is the models a stored search
        # scored, which searched_models reads off that search: a dataset's skip
        # policy can drop one, and svm drops itself above 10,000 molecules. A
        # model in the first and not in the second has no rows of its own to
        # rank, and until now it left the loop with nothing written down.
        # resolve_comparison_models returns the overlap and the reason for
        # every name it drops, and substitutes nothing.
        _requested = list(DOWNSTREAM_MODELS)
        _compare, _model_dropped = [], {}
        for _m in _requested:
            _s_m = _searches.get(_m)
            if _s_m is None:
                _model_dropped[_m] = _search_lookup_failed.get(_m) or (
                    'the pool returned no stored search for this downstream '
                    'model, so there are no rows of its own to rank')
                continue
            _keep, _why = resolve_comparison_models(_s_m, [_m])
            if not _keep:
                _model_dropped[_m] = _why[0][1]
                continue
            _compare.append(_m)
        results['top2_models_requested'] = _requested
        results['top2_models_compared'] = list(_compare)
        results['top2_models_dropped'] = _model_dropped
        results['top2_models_scored_by_each_search'] = {
            _m: searched_models(_s) for _m, _s in _searches.items()}
        for _m, _why in _model_dropped.items():
            print(f"  top-2 rows absent for {_m}: {_why}")

        for _m in _compare:
            search_m = _searches[_m]
            _ctx = f'n22 {dataset_name}/{_m}'
            try:
                # require_full=False: a downstream model its own search never
                # scored comes back as a short list carrying its own
                # shortfall_reason instead of raising and taking the run down.
                _b = top_base_reps(search_m, _m, 2, require_full=False,
                                   context=_ctx)
                top2_base_reps[_m] = [(e['rep'], e['score'])
                                      for e in _b['selected']]
                _h = top_hybrids(search_m, _m, 2, require_full=False,
                                 context=_ctx)
                # Scalars only. Each selected row also carries a Blueprint
                # object and this dict is written to JSON. The recipe is named
                # by 'label'; which representations fed it is reported by
                # hybrid_deltas under 'source_reps'.
                top2_hybrids[_m] = [
                    {'label': e['label'], 'score_mean': e['score_mean'],
                     'budget': e['budget'], 'n_features': e['n_features'],
                     'rep_count': e['rep_count'],
                     'n_folds_scored': e['n_folds_scored']}
                    for e in _h['selected']]
                if _h['selected']:
                    # hybrid= is passed rather than left None: the None branch
                    # of hybrid_deltas reads ['row'] off a top_hybrids entry,
                    # and top_hybrids builds no 'row' key, so it raises
                    # KeyError. The rank-1 entry carries 'score_mean' and
                    # 'blueprint', which is what hybrid_deltas reads.
                    hybrid_deltas_by_model[_m] = hybrid_deltas(
                        search_m, _m, hybrid=_h['selected'][0], context=_ctx)
            except Exception as _e:
                print(f"  {_m}: top-2 / hybrid deltas unavailable "
                      f"({type(_e).__name__}: {_e})")
        # select_model_rep_pairs takes ONE SearchResult and walks every model in
        # it, so it is given rf's and that is written down under
        # 'top2_pairs_search_of_model'. It resolves the model list against that
        # search itself and reports every name it drops under 'shortfalls'.
        results['top2_pairs_search_of_model'] = 'rf'
        try:
            _sel = select_model_rep_pairs(
                search, models=list(DOWNSTREAM_MODELS), n_base=2, n_hybrids=2,
                require_full=False, context=f'n22 {dataset_name}')
            top2_pairs_by_model = _sel['pairs']
            results['top2_pairs_shortfalls'] = list(_sel['shortfalls'])
        except Exception as _e:
            print(f"  top-2 pair grid unavailable "
                  f"({type(_e).__name__}: {_e})")

    # New keys, keyed by downstream model name, beside 'top3_base_reps' and not
    # in place of it. A downstream model whose own search never scored it, or
    # whose pool lookup returned no search, is absent here and named in
    # 'top2_models_dropped' with the reason. Nothing else reads them:
    # _smoke_n22.py indexes 'per_space', 'average_precision_ranking',
    # 'n_records_flagged' and 'n_records', and noise_ceiling_analyse lists n22
    # under NOT_A_SWEEP and reads no key of it.
    results['top2_base_reps'] = convert_numpy(top2_base_reps)
    results['top2_hybrids'] = convert_numpy(top2_hybrids)
    # One flat row per (downstream model, slot): 5 downstream models x
    # (2 base representations + 2 hybrids) = 20 rows when every slot is filled.
    results['top2_pairs_by_model'] = convert_numpy(top2_pairs_by_model)
    # Two deltas per downstream model, both POSITIVE when the hybrid is better:
    # 'delta_vs_source_mean' against the mean of the base representations the
    # budget gave a non-zero number of features, 'delta_vs_best_rep' against the
    # single best base representation for that model.
    results['hybrid_deltas_by_model'] = convert_numpy(hybrid_deltas_by_model)
    results['top2_read_from'] = (
        "one stored search per downstream model, each from a pool lookup that "
        "asked about that model. rf's is the search that chose the hybrid "
        "recipe. Every score is in its own search's metric, not in average "
        "precision against the curator flag."
        if search is not None else
        'no search: the pool lookup returned none, so no ranking was read')

    # ── score every record, inside its own fold ──
    splitter, use_groups = make_cv_splitter(N_FOLDS, search_groups,
                                            task_type=task_type)
    sample_X = next(iter(reps.values()))
    split_args = ((sample_X, labels, search_groups) if use_groups
                  else (sample_X, labels))
    folds = list(splitter.split(*split_args))
    print(f"  {len(folds)} folds, scaffold-grouped, so a molecule's repeats "
          f"never sit on both sides")

    base_matrices = {sp: _clean_matrix(reps[rn])
                     for sp, rn in base_spaces.items()}
    scores = {s: {e: np.full(n_samples, np.nan) for e in estimators}
              for s in spaces}
    failures = {s: {} for s in spaces}
    est_seconds = {s: {e: 0.0 for e in estimators} for s in spaces}
    timings = {}
    exclusion_note = {}

    # ── The shared similarity axis and the checkpoint ──
    # Added 2026-09-25 (plan sections 1-3).
    #
    # THE SIMILARITY AXIS. Every detector that reads a neighbour list
    # (the four estimators that take precomputed_neighbors,
    # noise_estimation._neighbours_for) is re-run on each fold's rows with its
    # neighbours found in every setting of sx.sweep_settings() -- rf's top-2
    # single representations, each plus augmented features, rf's two hybrids
    # rebuilt through noise_helpers.hybrid_for_fold, ECFP4 Tanimoto on the
    # records' SMILES, forest proximity -- at every k in SWEPT_KS. Each
    # record's own other measurements stay excluded (exclude_groups, as
    # below). The same grading follows, over records and over molecules,
    # under 'similarity_axis'. The feature space each detector's X is in is
    # the space it is filed under, as before; only the neighbours move.
    # KIRBY_SIMILARITY_SETTINGS=model_input switches it off.
    #
    # CHECKPOINT. Every score array is pickled after every (space, fold), so
    # a killed run resumes at the next one.
    sweep = sx.sweep_settings()
    results['similarity_settings_swept'] = [s_.name for s_ in sweep]
    results['similarity_ks_swept'] = list(SWEPT_KS) if sweep else []
    # Which neighbours every number OUTSIDE the rows and the similarity axis was
    # computed with, so each result record names its similarity setting.
    results['similarity_setting_of_the_default_path'] = sx.TODAY
    _nb_params = {}
    for e in estimators:
        try:
            _nb_params[e] = 'precomputed_neighbors' in inspect.signature(
                type(get_noise_estimator(e)).__init__).parameters
        except Exception:
            _nb_params[e] = False
    sim_estimators = [e for e in estimators if _nb_params.get(e)]
    results['similarity_estimators'] = sim_estimators
    # TODO(similarity-sweep 2026-09-25): a neighbour-based estimator whose
    # class takes no precomputed_neighbors (n11's 'hybrid' estimator, when it
    # resolves) still finds its own neighbours in the space's X and is not on
    # the similarity axis. Adding it needs precomputed_neighbors on that class
    # in src/kirby/utils/noise_estimation.py, outside this file.
    results['similarity_estimators_not_swept'] = [
        e for e in estimators if not _nb_params.get(e)]
    # {space: {'<setting>|k=<k>': {estimator: row as per_space holds}}}
    sim_graded = {}
    sim_unavailable = {}
    # {(setting|k, estimator): full-length scores} of the space in flight.
    sim_scores = {}
    try:
        from experiments._shared import get_model_factories
        _rf_factory = get_model_factories(['rf'], task_type=task_type,
                                          n_samples=n_samples).get('rf')
    except Exception:
        _rf_factory = None
    _metric = (metric_name, _fn, lower_is_better)
    _recipes_rf = {}
    for _nm, (_kind, _bp, _aug) in spaces.items():
        if _kind == 'hybrid' and _nm == _space_name('predictive'):
            _recipes_rf['hybrid'] = _bp
        elif _kind == 'hybrid' and _nm == _space_name('similarity'):
            _recipes_rf['hybrid_similarity'] = _bp
    # The code that computes the scores is in the signature (2026-09-28).
    # run_signature holds the data, the settings and the similarity setting,
    # not the code, so a run killed at the time cap and resumed after an edit
    # to a detector or to this file would have joined folds scored by the old
    # code to folds scored by the new, under one set of numbers. Any byte
    # change to these three files now starts the scoring again.
    import hashlib as _hashlib
    import kirby.utils.noise_estimation as _ne_mod
    import n06_bindingdb_ground_truth as _n06_mod
    _code = {}
    for _mod_path in (os.path.abspath(__file__), _ne_mod.__file__,
                      _n06_mod.__file__):
        try:
            with open(_mod_path, 'rb') as _fh:
                _code[os.path.basename(_mod_path)] = _hashlib.sha1(
                    _fh.read()).hexdigest()[:12]
        except OSError:
            _code[os.path.basename(str(_mod_path))] = 'unreadable'
    ck = sx.FoldCheckpoint(
        output_dir, EXP_FULL_NAME, dataset_name,
        sx.run_signature(data, config, {
            'estimators': estimators, 'spaces': sorted(spaces),
            'base_spaces': base_spaces, 'budget': budget,
            'pool_subset': pool_subset, 'code': _code}))
    if ck.state is not None:
        (scores, failures, est_seconds, timings, exclusion_note, sim_graded,
         sim_unavailable, sim_scores) = ck.state

    def _grade_all(sus):
        row = {'records': grade(flag, sus, label='records')}
        sus_mol = _by_molecule(sus, molecule_of_row, n_molecules, 'max')
        row['molecules'] = grade(flag_by_molecule, sus_mol, label='molecules')
        return row

    def _sweep_fold(X_fold, X_tr, X_va, train_idx, val_idx, rows, y_fold,
                    excl_fold, is_val):
        """Every neighbour-reading detector under every setting and k, on
        one fold's rows, X_fold being the rows as the detectors see them.
        Fills sim_scores for the held-out records."""
        # _clean_matrix, as the base spaces above get it: a non-finite entry
        # would put nan into every distance of its row.
        train_reps = {k_: _clean_matrix(v[train_idx]) for k_, v in reps.items()}
        val_reps = {k_: _clean_matrix(v[val_idx]) for k_, v in reps.items()}
        aug_tr = aug_va = None
        if augmentations:
            aug_tr = {a: m_[train_idx] for a, m_ in augmentations.items()}
            aug_va = {a: m_[val_idx] for a, m_ in augmentations.items()}
        g_tr = (np.asarray(search_groups)[train_idx]
                if search_groups is not None else None)
        sp_tr, sp_va, unav = sx.build_spaces(
            dataset_name, sweep, X_model_tr=X_tr, X_model_va=X_va,
            train_reps=train_reps, val_reps=val_reps, train_idx=train_idx,
            base_names=list(base_spaces.values()), recipes=_recipes_rf,
            y_train=labels[train_idx], groups_train=g_tr, model_name='rf',
            model_factory=_rf_factory, metric=_metric, task_type=task_type,
            augmentations_train=aug_tr, augmentations_val=aug_va,
            want_val=True)
        # The fold's rows, training then held out, as the estimators see
        # them. 'model_input' is X_fold itself.
        sp_all = {k_: np.vstack([v, sp_va[k_]]) for k_, v in sp_tr.items()
                  if sp_va.get(k_) is not None}
        sp_all['model_input'] = X_fold
        smiles_rows = sx.smiles_for(data, rows)
        fold_groups_all = (np.asarray(search_groups)[rows]
                           if search_groups is not None else None)
        for setting in sweep:
            if setting.space in unav:
                sim_unavailable.setdefault(setting.name, unav[setting.space])
                continue
            nb, why = sx.neighbour_lists(
                dataset_name, setting, sp_all, y_fold,
                exclude_groups=excl_fold, smiles_train=smiles_rows,
                is_clf=is_clf)
            if nb is None:
                sim_unavailable.setdefault(setting.name, why)
                continue
            for k in sx.ks_held(nb):
                sim = sx.sim_name(setting, k)
                for est_name in sim_estimators:
                    try:
                        cls = type(get_noise_estimator(est_name))
                        params = inspect.signature(cls.__init__).parameters
                        kw = {'k': int(k), 'precomputed_neighbors': nb}
                        # The same arguments the default path hands the
                        # estimator (the loop below), plus the lists.
                        if fold_groups_all is not None and 'groups' in params:
                            kw['groups'] = fold_groups_all
                        if 'exclude_groups' in params:
                            kw['exclude_groups'] = excl_fold
                        if is_clf and 'is_clf' in params:
                            kw['is_clf'] = True
                        est = get_noise_estimator(est_name, **kw)
                        sc_ = np.asarray(est.estimate(X_fold, y_fold),
                                         dtype=float)
                    except Exception as exc:
                        sim_unavailable.setdefault(
                            f'{sim}|{est_name}',
                            f'{type(exc).__name__}: {exc}')
                        continue
                    arr = sim_scores.setdefault(
                        (sim, est_name),
                        np.full(n_samples, np.nan, dtype=np.float32))
                    arr[rows[is_val]] = sc_[is_val]

    for space_name, (kind, blueprint, aug) in spaces.items():
        print(f"\n  === {space_name} ===")
        if ck.is_done(f'{space_name}|done'):
            print("    already in the checkpoint, skipped")
            continue
        t_space = time.time()
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            if ck.is_done(f'{space_name}|f{fold_i}'):
                print(f"    fold {fold_i}: already in the checkpoint, skipped")
                continue
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
                X_tr = base_matrices[space_name][train_idx]
                X_va = base_matrices[space_name][val_idx]

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
                # No row cap for gaussian_process here (removed 2026-09-28).
                # This file used to hand it 1,000 of the fold's 12,008 rows,
                # so it scored about 1,000 records in all while every other
                # detector scored all 12,008, and the ranking below set its
                # average precision beside theirs over different records.
                # What removing the cap costs. GaussianProcessEstimator fits
                # each of its inner folds on at most 1,000 training rows
                # (noise_estimation.py, n_max in fold_fn) and only PREDICTS
                # the rest. With the cap its inner training folds held about
                # 800 rows; now they hold about 9,600 and are cut to 1,000.
                # A fit grows with the cube of its rows, so each inner fit is
                # about (1000/800)^3 = 1.95 times the work it was. That is
                # arithmetic, not a measured time. GP_SUBSAMPLED_ESTIMATORS
                # and GP_MAX_ROWS stay imported and unused so n06's constants
                # are not touched.

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
            if sweep and sim_estimators:
                _sweep_fold(X_fold, X_tr, X_va, train_idx, val_idx, rows,
                            y_fold, excl_fold, is_val)
            ck.save(f'{space_name}|f{fold_i}',
                    (scores, failures, est_seconds, timings, exclusion_note,
                     sim_graded, sim_unavailable, sim_scores))
            print(f"    fold {fold_i}: {len(train_idx)} records chose the "
                  f"features, {len(val_idx)} records scored "
                  f"[{time.time() - t_fold:.0f}s]")
        timings[space_name] = round(time.time() - t_space, 1)
        if sweep:
            # Grade this space's similarity axis, then free its arrays.
            _g = {}
            for (sim, est_name), sus in sorted(sim_scores.items()):
                _g.setdefault(sim, {})[est_name] = _grade_all(
                    np.asarray(sus, dtype=float))
            sim_graded[space_name] = _g
            sim_scores = {}
        ck.save(f'{space_name}|done',
                (scores, failures, est_seconds, timings, exclusion_note,
                 sim_graded, sim_unavailable, sim_scores))

    results['seconds_per_space'] = timings
    if sweep:
        results['similarity_axis'] = convert_numpy(sim_graded)
        results['similarity_axis_note'] = (
            "similarity_axis[space]['<setting>|k=<k>'][detector] holds the "
            "same grading as per_space, with the detector's neighbours found "
            "in that similarity setting. The detector's own features are still "
            "the space's.")
        if sim_unavailable:
            results['similarity_axis_unavailable'] = sim_unavailable
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

    # ── the label on its own, graded the same way (added 2026-09-28) ──
    # A control no detector here had. Each record's suspicion is read off its
    # own label alone: no features, no neighbours, no fit. If a detector does
    # not beat these, its average precision against the flag says nothing
    # about the features it was run in.
    #   lowest_label_first         minus the pIC50, so the weakest binder
    #                              ranks most suspicious
    #   farthest_from_median_label the distance of the pIC50 from the median
    #                              pIC50 of all records, either direction
    # Both use only what every detector is already handed for the record it
    # scores, its own label, and neither sees the flag.
    label_only = {}
    for _nm, _sus in (
            ('lowest_label_first', -labels),
            ('farthest_from_median_label',
             np.abs(labels - np.median(labels)))):
        _row = {'records': grade(flag, _sus, label='records')}
        _row['molecules'] = grade(
            flag_by_molecule,
            _by_molecule(_sus, molecule_of_row, n_molecules, 'max'),
            label='molecules')
        label_only[_nm] = _row
    results['label_only_controls'] = convert_numpy(label_only)

    # ── how much of the flag is a cut on the value (added 2026-09-28) ──
    # Counted, not assumed. pIC50 = 9 - log10(IC50 in nM), so pIC50 below 4
    # is IC50 above 100,000 nM.
    _below = labels < 4.0 - 1e-9
    _otr = np.array([ft == 'Outside typical range' for ft in flag_text],
                    dtype=bool)
    results['flag_versus_ic50_above_100000_nM'] = {
        'n_records_above_100000_nM': int(_below.sum()),
        'n_of_those_carrying_any_flag': int((_below & flag).sum()),
        'n_outside_typical_range_records': int(_otr.sum()),
        'n_of_those_above_100000_nM': int((_otr & _below).sum()),
        'n_flagged_records_at_or_below_100000_nM': int((flag & ~_below).sum()),
        'note': (
            "Counts of records. If every record above 100,000 nM carries a "
            "flag and every 'Outside typical range' record is above it, that "
            "text is a fixed cut on the value in this pull. Whether a person "
            "or a rule wrote it is not recorded in the pull."),
    }

    # ── the hybrid minus each single representation, per detector ──
    # One block per single-representation space (base_1, base_2), each keyed
    # by hybrid space.
    hybrid_spaces = [s for s, (k, _b, _a) in spaces.items() if k == 'hybrid']
    delta = {}
    for base_sp, base_rn in base_spaces.items():
        per_hybrid = {}
        for hs in hybrid_spaces:
            d = {}
            for est_name in estimators:
                # Both numbers over the SAME records (2026-09-28). A fold whose
                # hybrid would not build, or a detector that failed in one
                # fold of one space, leaves those records unscored in that
                # space only. Graded apart, the two average precisions were
                # then over different records with a different flagged share,
                # and their difference was partly that. When every record is
                # scored in both, this is the same number as before.
                _sa = np.asarray(scores[hs][est_name], dtype=float)
                _sb = np.asarray(scores[base_sp][est_name], dtype=float)
                _both = np.isfinite(_sa) & np.isfinite(_sb)
                if not _both.any():
                    continue
                ga = grade(flag[_both], _sa[_both], label='records')
                gb = grade(flag[_both], _sb[_both], label='records')
                a = (ga or {}).get('detection_average_precision')
                b = (gb or {}).get('detection_average_precision')
                if a is None or b is None or not (np.isfinite(a)
                                                  and np.isfinite(b)):
                    continue
                d[est_name] = {
                    'average_precision_in_the_hybrid': float(a),
                    f'average_precision_in_{base_rn}': float(b),
                    'difference': float(a - b),
                    'n_records_scored_in_both': int(_both.sum()),
                    'n_flagged_among_them': int(flag[_both].sum())}
            per_hybrid[hs] = d
        delta[base_sp] = {'comparator_representation': base_rn,
                          'per_hybrid_space': per_hybrid}
    results['hybrid_minus_single_representation'] = convert_numpy(delta)
    results['hybrid_minus_single_representation_note'] = (
        'both numbers are average precision against the curator flag over '
        'records, both over the records scored in both spaces '
        '(n_records_scored_in_both), and both are printed so the difference '
        'is never read without the pair it came from')

    # ── the ordering, so it can be put beside n11's ──
    best = {}
    # How many records each ranked number is over (2026-09-28). A detector
    # that failed in a fold is graded over fewer records than the rest, with a
    # different flagged share, and the ranking used to set it beside them with
    # nothing to say so. The ranking itself is unchanged; this sits beside it.
    ranked_n = {}
    for space_name in spaces:
        ranked = []
        ranked_n[space_name] = {}
        for est_name in estimators:
            g = per_space[space_name][est_name]['records']
            if g and np.isfinite(g.get('detection_average_precision', np.nan)):
                ranked.append((_canonical(est_name),
                               float(g['detection_average_precision'])))
                ranked_n[space_name][_canonical(est_name)] = {
                    'n_scored': int(g['n_scored']),
                    'n_flagged': int(g['n_flagged'])}
        ranked.sort(key=lambda kv: -kv[1])
        best[space_name] = ranked
        if ranked:
            print(f"\n  {space_name}, average precision against the flag "
                  f"({n_flag} flagged records out of {n_samples} scored; "
                  f"random ranking would score "
                  f"{n_flag / float(n_samples):.4f}):")
            for name, ap in ranked:
                _n = ranked_n[space_name][name]
                _short = ('' if _n['n_scored'] == n_samples else
                          f"  [over {_n['n_scored']} records, "
                          f"{_n['n_flagged']} flagged, not all {n_samples}]")
                print(f"    {ap:.4f}  {name}{_short}")
            for _nm, _row in label_only.items():
                _g = _row.get('records') or {}
                if _g:
                    print(f"    {_g['detection_average_precision']:.4f}  "
                          f"[control, label only] {_nm}")
    results['average_precision_ranking'] = convert_numpy(best)
    results['average_precision_ranking_counts'] = convert_numpy(ranked_n)
    results['average_precision_ranking_note'] = (
        'detectors only. The two label-only controls are under '
        "'label_only_controls', graded over the same records, and a detector "
        'that does not beat them has shown nothing about its feature space')

    # Every per-record score, so a re-analysis costs nothing.
    results['per_record'] = convert_numpy({
        'curator_flag': flag.astype(int).tolist(),
        'molecule_id': molecule_of_row.tolist(),
        'suspicion': {sp: {e: scores[sp][e].tolist() for e in estimators}
                      for sp in spaces},
    })
    # n22's result file is written by its caller (run_all or main), not here.
    # The snapshot is cleared on return: a kill between this return and that
    # write costs a re-run, never a stale resume.
    ck.clear()
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
