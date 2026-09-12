"""Shared configuration and runner for per-dataset noise experiments.

Each dataset gets its own script (e.g. herg.py, logd.py) that calls
run_all() with its rep list from the pipeline readiness tiers.
Every experiment here runs on one dataset. All but one are independent of each
other and may run in any order; n06 is not, and runs last. Its headline is the
rank correlation between its own ordering of the noise estimators and the
ordering n11 and n13 measured under injected noise, and it reads those two out
of the files they write in the same run.

Experiments:
  N01-N08: Original noise suite (cliffs, weighting, SAS, GP-LOO, detection,
           remediation, estimation survey)
  N06: real measurement disagreement — repeat measurements and two curators,
       never injected noise
  N09: LNL method survey with hybrids (12 training-loop methods)
  N10: Hybrid neighborhood for similarity-dependent LNL
  N11: Noise estimation × remediation grid (full confirmation report grid)
  N12: Feature importance-weighted distance for noise estimation
  N13: Alternative noise distributions (uniform, multiplicative, heavy-tailed, partial)
  N14: Activity-cliff landscape survey (gate; trains no downstream models)
  N15: Clean-searched vs noise-searched hybrid recipe (oracle advantage)
  N16: Feature importance across noise levels (gate; trains no downstream models)

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy, should_skip_experiment,
    experiment_result_is_empty,
    DOWNSTREAM_MODELS_NOISE, INTERNAL_EVALUATORS,
    input_fingerprint, fingerprint_mismatch,
)

# Reuse dataset rep configs — single source of truth
from experiments.complexity.complexity_helpers import DATASET_REPS


# ── Experiment list ────────────────────────────────────────────────────────
# All standalone, no ordering constraints.

SINGLE_DATASET_EXPERIMENTS = [
    'n01',   # cliff-stratified prediction error (sub-4 only; sub-1/2/3 cut)
    'n02',   # importance-weighted sample weighting
    'n07',   # noise remediation strategies
    'n09',   # LNL method survey with hybrids (12 methods)
    'n11',   # noise estimation x remediation grid
    'n12',   # feature importance-weighted distance
    'n13',   # alternative noise distributions
    'n14',   # activity-cliff landscape survey (gate; no downstream models)
    'n15',   # clean-searched vs noise-searched hybrid blueprint (oracle advantage)
    'n16',   # feature importance across noise levels (gate; no downstream models)
    # n06 runs LAST, and that is load-bearing: its headline is the rank
    # correlation between its own ordering of the noise estimators and the
    # ordering n11 and n13 measured under injected noise, and it reads those
    # two orderings out of the result files they write in this same run. Put it
    # earlier and it finds no stored ranking for the dataset in hand and
    # records that instead of the number.
    'n06',   # real measurement disagreement, from a ground-truth provider
]
# CUT 2026-07-07 (irredeemable by design — see AUDIT_NOISE.md):
# n03 sas_landscape, n04 gp_loo_detection, n05 noise_injection_benchmark,
# n08 estimation_survey, n10 hybrid_neighborhood_lnl.

# n06 was excluded from this list until 2026-09-11, on the grounds that it was
# dataset-specific and had to be run by hand. It no longer is: its ground truth
# comes from a provider, and a dataset where no provider finds a repeat
# structure writes a skip carrying the count it found rather than being left
# out of the run.


# ── Noisy-label hybrid search grid ─────────────────────────────────────────
# THE noise conditions for which a fairly-searched (noise-aware) hybrid blueprint
# is precomputed and cached. This list is the single source of truth: the
# precompute builds exactly these, and an experiment may only ask for a spec that
# is in here. If the two ever diverge, an experiment gets a blueprint fitted to a
# noise draw that exists nowhere in its own run — and nothing crashes. Hence: one
# list, imported by both.
#
# Levels are TARGET EFFECTIVE NOISE: the noise actually DELIVERED, measured as a
# root-mean-square, as a fraction of the label SD. A level therefore means the same
# thing on every dataset and in every condition — unlike the five ad-hoc, mutually
# incompatible `_inject_noise` definitions the suite grew, four of which scaled by the
# label RANGE and so ran at roughly 4-6x the level they claimed.
#
# ONE noise_seed. The blueprint should not swing with the random draw; if it does,
# that is itself the finding, and it is what the clean-vs-noisy experiment measures.
NOISY_SEARCH_LEVELS = (0.05, 0.10, 0.20)
NOISY_SEARCH_SEED = 0


def noisy_search_specs(task_type: str):
    """The NoiseSpec list the noisy precompute builds, for this task type."""
    from kirby.noise_spec import default_spec
    is_clf = (task_type == 'classification')
    return [default_spec(is_clf=is_clf, target_effective_noise=lvl,
                         noise_seed=NOISY_SEARCH_SEED)
            for lvl in NOISY_SEARCH_LEVELS]


# ── ONE injector for the whole noise suite ─────────────────────────────────
# n07, n09, n11, n12 and n13 each carried a private `_inject_noise`, and no two of
# them meant the same thing: n07 corrupted a fraction of rows at 0.3 of the label
# range, n09/n11/n12 added Gaussian noise to every row at a fraction of the range,
# and n13 had four distributions of its own. None went through noiseInject. They are
# replaced by the two functions below.

def noise_spec_for(dataset_name, condition, level, base_seed=42, fold_idx=0,
                   noise_seed=0, is_clf=False):
    """A NoiseSpec whose draw is unique per (condition, level, fold, seed).

    The injector advances its generator on every draw, so re-using one across a
    sweep makes a level's realisation depend on how many levels ran before it — and
    a run of part of the grid then would not reproduce the matching rows of a full
    run. The seed depends on the CONDITION as well, or two conditions at one level
    would share a realisation and a comparison between them would measure nothing.

    zlib.crc32, NOT hash(): Python randomises string hashing per process, so hash()
    would make the noise different on every run.

    WHICH molecules are hit is seeded separately, from everything EXCEPT the level.
    Without that the outlier and grouped conditions pick a different set at every
    level, so a level sweep varies the amount and the molecules at once and the
    levels do not nest. Measured on noiseInject's 120-molecule fixture: the
    level-free pattern against the realised scale gave Spearman -0.101 for
    outlier_p10 before this, and 1.000 after (noiseInject d39d611, 2026-08-27).
    """
    import zlib
    from kirby.noise_spec import (CLASSIFICATION_STRATEGIES, DEFAULT_STRATEGY_CLF,
                                  NoiseSpec)
    from experiments.dataset_policy import noise_dose_units

    if is_clf and condition not in CLASSIFICATION_STRATEGIES:
        # A flipped class label has no magnitude, so none of the dose-matched
        # conditions has a classification counterpart and the level stops being a
        # dose and becomes a flip rate. Fall back to the symmetric flip model rather
        # than hand a regression condition to the classification injector, which
        # raises. An experiment that wants a specific flip model names one of
        # CLASSIFICATION_STRATEGIES instead.
        condition = DEFAULT_STRATEGY_CLF

    digest = zlib.crc32(str(condition).encode('utf-8'))
    selection_seed = (int(base_seed) * 1000003 + digest
                      + int(fold_idx) * 104729
                      + int(noise_seed) * 15485863) & 0x7FFFFFFF
    seed = (selection_seed
            + int(round(float(level) * 1000)) * 7919) & 0x7FFFFFFF
    return NoiseSpec(strategy=condition, target_effective_noise=float(level),
                     noise_seed=seed, selection_seed=selection_seed,
                     dose_units=noise_dose_units(dataset_name))


def inject_train_labels(y_clean, spec, is_clf=False, groups=None):
    """Corrupt one fold's TRAINING labels, and say what was corrupted.

    Returns (y_noisy, corrupt_mask, provenance). `corrupt_mask` is the ground truth
    for "did the estimator find the corrupted samples" — the detection and
    remediation experiments need it and used to compute it locally and throw it away.
    `provenance` is the row every result must carry so a figure traces back to the
    amount of noise actually delivered.

    The dose is solved against THIS fold's clean training labels, matching
    rust/src/main.rs:512.
    """
    import numpy as np
    from kirby.noise_spec import make_noisy_labels, make_noisy_labels_verbose, \
        noise_provenance_row

    y_clean = np.asarray(y_clean)
    if is_clf:
        y_noisy = make_noisy_labels(y_clean, spec, is_clf=True)
        return y_noisy, (y_noisy != y_clean), {'noise_spec_id': spec.spec_id}

    res = make_noisy_labels_verbose(y_clean, spec, groups=groups)
    scale = np.asarray(res.noise_scale, dtype=float)
    if spec.scale_is_degenerate or scale.max() - scale.min() <= 1e-12:
        # Every molecule gets the same scale, so "which ones are corrupted" is every
        # one of them that actually moved. For the five shape-only conditions that is
        # all of them — the question of WHERE the noise is is undefined there, not
        # answered with an empty mask.
        mask = np.asarray(res.epsilon) != 0.0
    else:
        mask = scale > (scale.min() + 1e-12)
    return np.asarray(res.y_noisy), mask, noise_provenance_row(res, spec)


def _import_experiment(exp_id: str):
    """Dynamically import a noise experiment module by ID (e.g. 'n01')."""
    module_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
        'n16': 'n16_importance_under_noise',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.noise.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name used by save_experiment_results."""
    name_map = {
        'n01': 'n01_activity_cliff_framework',
        'n02': 'n02_sample_weighting',
        # n06 was absent here while it was hand-run. Without it the resume
        # check and the saved file both went to a directory named 'n06', which
        # is not where anything looks for it.
        'n06': 'n06_bindingdb_ground_truth',
        'n07': 'n07_remediation',
        'n09': 'n09_lnl_survey',
        'n11': 'n11_estimation_remediation_grid',
        'n12': 'n12_importance_weighted_distance',
        'n13': 'n13_noise_distributions',
        'n14': 'n14_activity_cliff_landscape',
        'n15': 'n15_clean_vs_noisy_hybrid',
        'n16': 'n16_importance_under_noise',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False):
    """Load data once and run all single-dataset noise experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki', 'logd', 'flab_thermo_tresanco').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: SINGLE_DATASET_EXPERIMENTS).
        config: Config dict override (default: built from the DEFAULT_* constants).
        rep_names: Explicit rep list override. If None, uses DATASET_REPS[dataset_name]['general'].
        pruning_mode: One of PRUNING_MODES. 'default_pruning' = legacy
            single-stage prune. 'aggressive_pruning' = F16-validated multi-
            stage pruning (checkpoints=[25,50,100], td=0.5) applied globally
            to all models, no n_samples gate. F16: ~58% fold-0 eval reduction.
    """
    if config is None:
        from kirby.find_best_hybrid import (
            DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS, DEFAULT_N_FOLDS,
        )
        # Production precompute uses FULL_BUDGETS =
        # [10,25,50,100,250,500,1000,2000], which CONTAINS the hybrid budgets
        # n10 requests ([25,100,250]). An earlier four-entry list
        # ([50,500,1000,2000]) did not, so n10's strict budget guard skipped
        # all three and the hybrid-vs-base comparison (n10's whole point)
        # came back silently empty. Both the in-memory precompute below and the SLURM
        # disk cache (tests/precompute_hybrids.py) already build
        # FULL_BUDGETS, so this only makes config['budgets'] truthful and keeps
        # n10's requested hybrid budgets reachable.
        from experiments._shared import FULL_BUDGETS
        config = {
            'n_splits': DEFAULT_N_FOLDS,
            'allocations': DEFAULT_ALLOCATION_METHODS,
            'importance_methods': DEFAULT_IMPORTANCE_METHODS,
            'internal_evaluators': INTERNAL_EVALUATORS,
            'budgets': FULL_BUDGETS,
            'models': DOWNSTREAM_MODELS_NOISE,
        }
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Resolve rep list
    if rep_names is None:
        ds_reps = DATASET_REPS.get(dataset_name)
        if ds_reps is None:
            raise ValueError(
                f"No rep list for '{dataset_name}'. Pass rep_names explicitly "
                f"or add it to DATASET_REPS in complexity_helpers.py."
            )
        rep_names = ds_reps['general']

    print(f"\n{'=' * 70}")
    print(f"NOISE EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE ──
    t0 = time.time()
    print(f"\nLoading {dataset_name}...")
    data = load_data(dataset_name, rep_preset='all', rep_names=rep_names)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # NaN guard: some reps produce NaN for certain sequences. Replace with 0 and warn.
    import numpy as np
    reps_data = data.get('reps', {})
    for rname, X in list(reps_data.items()):
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            pct = 100 * nan_count / X.size
            print(f"  WARNING: {rname} has {nan_count} NaN values ({pct:.1f}%) — imputing 0")
            reps_data[rname] = np.nan_to_num(X, nan=0.0)

    # ── Load augmentation reps ──
    aug_rep_names = DATASET_REPS.get(dataset_name, {}).get('augmentation', [])
    if aug_rep_names:
        missing = [r for r in aug_rep_names if r not in reps_data]
        if missing:
            print(f"  Loading {len(missing)} augmentation reps: {missing}")
            aug_data = load_data(dataset_name, rep_preset='all', rep_names=missing)
            aug_reps = {}
            for rn, arr in aug_data['reps'].items():
                if int(np.isnan(arr).sum()) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                aug_reps[rn] = arr
            data['augmentations'] = aug_reps
        else:
            data['augmentations'] = {r: reps_data[r] for r in aug_rep_names}
        print(f"  Augmentations ({len(data['augmentations'])}): "
              f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
    else:
        data['augmentations'] = None

    # ── Resolve pruning_mode + apply n_samples gate ──
    from experiments._shared import resolve_pruning_mode, apply_pruning_gate
    _sched, _ovr_template, _min_n = resolve_pruning_mode(pruning_mode)
    _model_overrides = apply_pruning_gate(
        _ovr_template, len(data['labels']), _min_n)
    print(f"\n  Pruning mode: {pruning_mode}")
    if _model_overrides is not None:
        print(f"    overrides active for: {sorted(_model_overrides.keys())}")
    elif _ovr_template is not None:
        print(f"    overrides gated off (n={len(data['labels'])} < {_min_n})")

    # ── Precompute hybrid searches (general reps) ──
    # precompute_hybrid_searches catches a per-aug_condition exception, stores
    # None under that key, and appends one record to `.failures`. Those records
    # are copied into data['precompute_errors'][tier] here and written to
    # _precompute_errors/<dataset>.json below, so a failed aug_condition is
    # named in the output instead of existing only as a printed line.
    _task_type = data['task_type']
    precompute_errors = data.setdefault('precompute_errors', {})
    from experiments._shared import precompute_failures
    if len(reps_data) >= 2:
        from experiments._shared import precompute_hybrid_searches
        print(f"\n  Precomputing hybrid searches (general, {len(reps_data)} reps)...")
        precompute_results = precompute_hybrid_searches(
            reps=reps_data,
            labels=data['labels'],
            groups=data.get('groups'),
            config=config,
            augmentations=data.get('augmentations'),
            task_type=_task_type,
            domain=data.get('domain', 'molecular'),
            dataset_name=dataset_name, tier='general',
            adaptive_pruning_schedule=_sched,
            model_pruning_aggressiveness=_model_overrides,
        )
        data['precomputed_searches'] = precompute_results
        precompute_errors['general'] = precompute_failures(precompute_results)
    else:
        data['precomputed_searches'] = {}
        precompute_errors['general'] = []

    # ── NOISY-LABEL hybrid searches: NOT precomputed, deliberately ──
    # Removed 2026-09-11. This block built one full search per noise level, on
    # every dataset a noise run touched, and nothing read any of them.
    #
    # Two things disagreed. The block built searches that TRAIN on corrupted
    # activity values and pick their winner against the CLEAN ones. n15 — the
    # only experiment in the suite that asks for a noisy search — passes
    # score_on_noisy_labels=True (n15:127), which asks for the version that
    # picks its winner on the corrupted values too. find_best_hybrid files the
    # two under different tiers, so they could never meet: n15 missed on every
    # layer and paid for its own searches on top of these.
    #
    # The dict stays, empty, because get_best_search_for_downstream still reads
    # it for a caller that asks for a noisy search WITHOUT
    # score_on_noisy_labels (_shared.py:5220). No such caller exists in the
    # suite today; if one appears it will build what it needs on its first call
    # and land it in the result store, the same way n15 already does.
    data['precomputed_searches_noisy'] = {}

    # ── Record precompute failures ──
    # Written before any experiment runs, so the record survives a run that
    # dies part-way. Each record carries tier, dataset, aug_condition,
    # error_type, error and traceback.
    precompute_error_audit = {
        'dataset': dataset_name,
        'n_failures': sum(len(v) for v in precompute_errors.values()),
        'by_tier': {k: list(v) for k, v in precompute_errors.items()},
    }
    if precompute_error_audit['n_failures']:
        print(f"  Precompute failures recorded: "
              f"{precompute_error_audit['n_failures']}")
    save_experiment_results(precompute_error_audit, output_dir,
                            '_precompute_errors', dataset_name)

    # What every result file written below is stamped with, and what a
    # resumed run compares its existing files against.
    _fingerprint = input_fingerprint(data, config)

    # ── Run experiments ──
    results = {}
    for exp_id in experiments:
        # Resume: skip an experiment whose output JSON already exists AND holds
        # a measurement. Lets a walled/restarted job advance instead of
        # re-running finished experiments. force_rerun=True recomputes always.
        #
        # A file existing is not proof the run produced anything. 55 files under
        # tests/results hold no finite number outside the metadata and status
        # keys — an all-failed run, a refusal, a section left as {} (listed in
        # EMPTY_RESULT_FILES_2026_08_27.md). Under an exists() test every one of
        # them was skipped on every resubmission. save_experiment_results now
        # stamps 'result_status'; experiment_result_is_empty reads it back.
        # A file carrying the {'skipped': True} marker classifies as 'skipped',
        # not 'empty', so a deliberate skip still resumes as before.
        #
        # classify_unmarked=True — the chosen meaning for a file written before
        # the stamp existed. Such a file carries neither 'result_status' nor
        # 'has_computed_results', and on 2026-08-27 every one of the 1541 files
        # under tests/results was such a file. Re-deriving the status from the
        # file's own sections is what makes this check act on them at all; left
        # at the default, an unmarked empty file still reads as finished.
        # A file that will not parse is recomputed for the same reason: no
        # measurement can be read out of it.
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id), f"{dataset_name}.json")
        if not force_rerun and os.path.exists(_out_json):
            try:
                with open(_out_json) as _f:
                    _prev = json.load(_f)
                _prev_empty = experiment_result_is_empty(
                    _prev, classify_unmarked=True)
                _why = 'records no measurement'
            except (ValueError, OSError) as _e:
                _prev_empty = True
                _why = f'could not be read ({_e})'
            if not _prev_empty:
                # A file with numbers in it can still be the wrong numbers.
                # Nothing in it said which fold grouping, labels or
                # representations produced them, so a resume after any of those
                # changed kept it. input_fingerprint records them; a mismatch
                # recomputes.
                _drift = fingerprint_mismatch(
                    _prev.get('input_fingerprint'), _fingerprint)
                if _drift:
                    _prev_empty = True
                    _why = _drift
            if _prev_empty:
                print(f"  RERUN {exp_id}: existing output {_why} — "
                      f"recomputing ({_out_json})")
            else:
                print(f"  RESUME {exp_id}: already done — skipping ({_out_json})")
                results[exp_id] = {'resumed': True}
                continue
        print(f"\n{'─' * 70}")
        t_exp = time.time()

        # Large-n / domain scope policy: write a {'skipped':...} marker so the
        # wave-audit counts it as intentionally-skipped (see _shared.should_skip_
        # experiment + REPRODUCIBILITY_LEDGER.md §E).
        _skip = should_skip_experiment(exp_id, data)
        if _skip is not None:
            print(f"  SKIP {exp_id}: {_skip['reason']}")
            results[exp_id] = _skip
            save_experiment_results(_skip, output_dir,
                                    _experiment_full_name(exp_id), dataset_name)
            continue

        try:
            mod = _import_experiment(exp_id)
        except (ImportError, KeyError) as e:
            print(f"  SKIP {exp_id}: import failed ({e})")
            continue

        try:
            exp_result = mod.run(data, config, output_dir, dataset_name)
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(exp_result, output_dir, exp_full_name,
                                    dataset_name, fingerprint=_fingerprint)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e)}

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} — {len(results)} experiments in {total:.0f}s")
    print(f"{'=' * 70}")

    # Underscore key so it does not collide with an experiment ID.
    results['_precompute_errors'] = precompute_error_audit
    return results
