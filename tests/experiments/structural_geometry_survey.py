#!/usr/bin/env python3
"""
Structural Geometry Augmentation Survey

Tests 7 structure-based geometry features as augmentations on top of base
antibody representations across all antibody tasks.

Features tested:
  vhvl_angles (6D)          — ABangle-style VH-VL orientation
  cdr_shape (60D)           — USR-like CDR loop shape moments
  cdr_canonical (12D)       — CDR canonical class + geometry
  paratope_charge (12D)     — Paratope electrostatics
  paratope_hydrophobicity (12D) — Paratope hydrophobicity profile
  cdr_contacts (30D)        — Inter-CDR pairwise distances
  vhvl_interface (30D)      — VH-VL interface composition
  all_geometry (162D)       — All 7 concatenated

For each dataset, tests:
  1. Each geometry feature standalone
  2. Each geometry feature augmented onto each base rep
  3. All geometry combined augmented onto each base rep

Usage:
    python tests/experiments/structural_geometry_survey.py flab_thermo_jain
    python tests/experiments/structural_geometry_survey.py all_inter
    python tests/experiments/structural_geometry_survey.py all_inter --base-reps esm2,ablang2,prot_t5
    python tests/experiments/structural_geometry_survey.py all --models rf,mlp

    # Report which cached geometry arrays are entirely zero (reports only):
    python tests/experiments/structural_geometry_survey.py --scan-zero-cache
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import hashlib
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# =============================================================================
# IMPORTS FROM EXISTING SURVEY
# =============================================================================

from antibody_baseline_survey import (
    DATASET_REGISTRY, DOMAIN_GROUPS, load_dataset,
    get_models, evaluate_rep, _rep_cache_path,
)


# Default base reps to test augmentation against
DEFAULT_BASE_REPS = [
    'esm2', 'ablang2', 'prot_t5', 'esm3', 'mint',
    'saprot', 'antiberty', 'blosum', 'aac',
]

# Ridge is ruled out of this project, so it is not a default here.
DEFAULT_MODELS_ARG = 'rf,mlp'


# =============================================================================
# GEOMETRY FEATURE CACHING
# =============================================================================

def _geo_cache_dir(heavy_seqs: List[str], light_seqs: List[str]) -> Path:
    """Get geometry feature cache directory, keyed by sequence hash."""
    content = '\n'.join(heavy_seqs) + '\n\n' + '\n'.join(light_seqs)
    data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    base = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    cache_dir = base / 'reps' / f'ab_{data_hash}' / 'geometry'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


GEO_ORDER = ['vhvl_angles', 'cdr_shape', 'cdr_canonical', 'paratope_charge',
             'paratope_hydrophobicity', 'cdr_contacts', 'vhvl_interface']


def _load_or_compute_geometry(
    heavy_seqs: List[str],
    light_seqs: List[str],
) -> tuple:
    """Load geometry features from cache or compute them.

    An entirely-zero array is never cached and never trusted on read — a zero
    array means structure prediction failed, and a shape match alone would let
    that be reloaded and reported as a real feature group on a later run.

    Returns (geo, failures): geo maps feature group name → (N, D) array, and
    failures maps a dropped group name → the reason it was dropped.
    Propagates GeometryUnavailableError if nothing could be computed at all.
    """
    from kirby.representations.antibody_geometry import (
        compute_all_geometry_features, GEOMETRY_DIMS)

    cache_dir = _geo_cache_dir(heavy_seqs, light_seqs)
    failures = {}

    # Try loading all from cache. Shape is necessary but not sufficient:
    # an all-zero array is rejected and forces a recompute.
    cached = {}
    all_cached = True
    for name, dim in GEOMETRY_DIMS.items():
        cache_file = cache_dir / f'{name}.npy'
        if cache_file.exists():
            try:
                X = np.load(cache_file, allow_pickle=False)
                if X.shape == (len(heavy_seqs), dim):
                    if np.all(X == 0):
                        print(f"    Cached '{name}' is entirely zero — "
                              f"discarding and recomputing: {cache_file}")
                    else:
                        cached[name] = X
                        continue
            except Exception:
                pass
        all_cached = False

    if all_cached and len(cached) == len(GEOMETRY_DIMS):
        print(f"  Loaded all geometry features from cache")
        cached['all_geometry'] = np.hstack([cached[n] for n in GEO_ORDER])
        return cached, failures

    # Compute all features — raises GeometryUnavailableError rather than
    # handing back zeros when ImmuneBuilder is missing or every sample fails.
    computed = compute_all_geometry_features(heavy_seqs, light_seqs)

    # Keep and cache only the groups that actually carry signal.
    geo = {}
    # vhvl_angles and vhvl_interface measure the heavy domain against the light
    # one. A nanobody has no light domain, so on flab_thermo_tresanco those two
    # are zero because the quantity does not exist, not because anything broke.
    n_light = sum(1 for l in light_seqs if l)
    single_domain = (n_light == 0)
    VHVL_GROUPS = ('vhvl_angles', 'vhvl_interface')

    for name in GEOMETRY_DIMS:
        X = computed[name]
        if np.all(X == 0):
            if single_domain and name in VHVL_GROUPS:
                failures[name] = (
                    f"undefined: every sample is single-domain, so there is no "
                    f"light domain to measure the heavy one against; not cached")
                print(f"    {name}: UNDEFINED — no light chain in this dataset")
            else:
                failures[name] = (
                    f"all {X.shape[0]}x{X.shape[1]} values are zero after "
                    f"computation — structure prediction produced nothing for "
                    f"this feature group; not cached")
                print(f"    {name}: FAILURE — entirely zero, not cached")
            continue
        geo[name] = X
        np.save(cache_dir / f'{name}.npy', X)

    if not geo:
        failures['all_geometry'] = "every feature group was entirely zero"
    else:
        present = [n for n in GEO_ORDER if n in geo]
        geo['all_geometry'] = np.hstack([geo[n] for n in present])
        if len(present) < len(GEO_ORDER):
            dropped = [n for n in GEO_ORDER if n not in geo]
            print(f"    all_geometry built from {len(present)}/7 groups "
                  f"({geo['all_geometry'].shape[1]}D); dropped: {dropped}")

    return geo, failures


# =============================================================================
# CACHE AUDIT
# =============================================================================

def scan_zero_geometry_cache(cache_root: Optional[str] = None) -> List[Path]:
    """Report which cached geometry arrays under KIRBY_CACHE_DIR are all zero.

    Reports only — deletes nothing. Run this on the server before a rerun:
        python tests/experiments/structural_geometry_survey.py --scan-zero-cache
    """
    base = Path(cache_root or os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby'))) / 'reps'

    print(f"Scanning geometry caches under: {base}")
    if not base.exists():
        print("  (no reps directory — nothing to scan)")
        return []

    zero_files = []
    n_checked = 0
    for npy in sorted(base.glob('*/geometry/*.npy')):
        n_checked += 1
        try:
            X = np.load(npy, allow_pickle=False)
        except Exception as e:
            print(f"  UNREADABLE {npy}: {e}")
            continue
        if X.size and np.all(X == 0):
            zero_files.append(npy)
            print(f"  ALL ZERO  {X.shape}  {npy}")

    print(f"\n  {n_checked} cached geometry arrays checked, "
          f"{len(zero_files)} entirely zero.")
    if zero_files:
        print("  Nothing was deleted. To clear them, review the list above "
              "and remove those files yourself.")
    return zero_files


# =============================================================================
# BASE REP LOADING
# =============================================================================

def _precompute_cache_path(dataset_name: str, rep_name: str) -> Path:
    """Where tests/precompute_reps.py writes a representation.

    It writes $KIRBY_CACHE_DIR/rep_cache/<dataset>/<rep>.npy, keyed on the
    dataset NAME. The antibody path this survey grew up on writes
    $KIRBY_CACHE_DIR/reps/ab_<hash of the sequence list>/<rep>.npy, keyed on the
    SEQUENCES. Two layouts, and filling one leaves the other empty — which is
    why a precompute run does not by itself feed this survey. Both are read.
    """
    base = Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))
    return base / 'rep_cache' / dataset_name / f'{rep_name}.npy'


def resolve_base_rep_names(dataset_name: str, requested: str) -> List[str]:
    """Turn the --base-reps argument into a list of representation names.

    'pruned' means DATASET_REPS[dataset]['general'], the ten names signed off
    during rep pruning for THAT dataset. They differ per dataset and they are
    not DEFAULT_BASE_REPS: abcov_kd's pool shares three names with it,
    ddg_combined shares one. Twelve of the twenty datasets this survey runs on
    are absent from DATASET_REPS altogether, and for those 'pruned' falls back
    to DEFAULT_BASE_REPS and says so.
    """
    if requested and requested != 'pruned':
        return [r.strip() for r in requested.split(',')]

    if not requested:
        return list(DEFAULT_BASE_REPS)

    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
    except Exception as e:
        print(f"  --base-reps pruned: cannot read DATASET_REPS ({e}); "
              f"using the default list")
        return list(DEFAULT_BASE_REPS)

    pool = DATASET_REPS.get(dataset_name, {}).get('general')
    if not pool:
        print(f"  --base-reps pruned: {dataset_name} is not in DATASET_REPS, "
              f"so it has no pruned pool; using the default list")
        return list(DEFAULT_BASE_REPS)
    print(f"  --base-reps pruned: {dataset_name} pool = {list(pool)}")
    return list(pool)


def _load_base_reps(
    heavy_seqs: List[str],
    light_seqs: List[str],
    base_rep_names: List[str],
    dataset_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load base representations from either cache. Skip any not found."""
    from kirby.representations.antibody import ANTIBODY_REPS

    reps = {}
    for rep_name in base_rep_names:
        cache_file = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
        candidates = [cache_file]
        if dataset_name:
            candidates.append(_precompute_cache_path(dataset_name, rep_name))

        loaded = False
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                X = np.load(cand, allow_pickle=False)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2 and X.shape[0] == len(heavy_seqs):
                    reps[rep_name] = X
                    loaded = True
                    break
            except Exception:
                pass
        if loaded:
            continue

        # Not cached — try computing if it's a fast rep
        if rep_name in ANTIBODY_REPS and rep_name in {'onehot', 'blosum', 'aac', 'paratope'}:
            try:
                X = ANTIBODY_REPS[rep_name](heavy_seqs, light_seqs)
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 2:
                    reps[rep_name] = X
                    np.save(cache_file, X)
                    continue
            except Exception:
                pass

        print(f"    WARNING: base rep '{rep_name}' not cached, skipping")

    return reps


# =============================================================================
# AUGMENTATION EVALUATION
# =============================================================================

def run_geometry_survey(
    dataset_name: str,
    base_rep_names: List[str],
    model_names: List[str],
    n_folds: int = 5,
    allow_missing_base_reps: bool = False,
    hybrid_only: bool = False,
) -> dict:
    """Run geometry augmentation survey for one dataset."""
    print(f"\n{'='*70}")
    print(f"GEOMETRY SURVEY: {dataset_name}")
    print(f"{'='*70}")
    t0 = time.time()

    # 1. Load data
    data = load_dataset(dataset_name)
    heavy_seqs = data['heavy_seqs']
    light_seqs = data['light_seqs']
    labels = data['labels']
    groups = data['groups']
    task_type = data['task_type']
    is_clf = (task_type == 'classification')

    print(f"  {data['n_samples']} samples, {data['n_groups']} groups, "
          f"task={task_type}")

    # 2. Load base representations BEFORE predicting any structure. A missing
    #    representation used to be a warning, so 20 datasets ran for 15 hours
    #    against aac and blosum alone and reported status OK. Checking here
    #    costs a directory listing; checking after the geometry block costs
    #    every structure in the dataset first.
    base_reps = _load_base_reps(heavy_seqs, light_seqs, base_rep_names,
                                dataset_name=dataset_name)
    missing = [r for r in base_rep_names if r not in base_reps]
    print(f"  Base reps loaded: {sorted(base_reps.keys())}")

    if missing and not allow_missing_base_reps:
        print(f"  REFUSING before predicting any structure: {len(missing)} of "
              f"{len(base_rep_names)} base representations are not cached: "
              f"{missing}")
        print(f"  Read from $KIRBY_CACHE_DIR/reps/ab_<sequence hash>/ and "
              f"$KIRBY_CACHE_DIR/rep_cache/{dataset_name}/. Build them, or pass "
              f"--allow-missing-base-reps to score against the "
              f"{len(base_reps)} that are present.")
        return {
            'status': 'FAILED',
            'error': f'{len(missing)} of {len(base_rep_names)} base reps not cached',
            'metadata': {
                'dataset': dataset_name,
                'n_samples': data['n_samples'],
                'base_reps_requested': list(base_rep_names),
                'base_reps_loaded': sorted(base_reps.keys()),
                'base_reps_missing': missing,
                'timestamp': datetime.now().isoformat(),
            },
            'geometry_failures': {},
            'geometry_baselines': {}, 'augmentations': {}}

    # 3. Compute geometry features. A total failure is recorded and returned,
    #    never written out as an empty-but-successful-looking run.
    from kirby.representations.antibody_geometry import GeometryUnavailableError
    try:
        geo_reps, geo_failures = _load_or_compute_geometry(heavy_seqs, light_seqs)
    except GeometryUnavailableError as e:
        print(f"  GEOMETRY UNAVAILABLE: {e}")
        return {
            'status': 'FAILED',
            'error': 'geometry unavailable',
            'metadata': {
                'dataset': dataset_name,
                'n_samples': data['n_samples'],
                'n_groups': data['n_groups'],
                'n_folds': n_folds,
                'task_type': task_type,
                'elapsed_seconds': round(time.time() - t0, 1),
                'timestamp': datetime.now().isoformat(),
            },
            'geometry_failures': {'all': str(e)},
            'geometry_baselines': {},
            'augmentations': {},
        }

    if not base_reps:
        print("  ERROR: No base reps available — run antibody_baseline_survey first")
        return {'status': 'FAILED', 'error': 'no base reps',
                'geometry_failures': geo_failures,
                'geometry_baselines': {}, 'augmentations': {}}

    # 3b. Build hybrid from all available reps (base + geometry)
    all_reps_for_hybrid = {**base_reps, **geo_reps}
    all_reps_for_hybrid.pop('all_geometry', None)
    if len(all_reps_for_hybrid) >= 2:
        try:
            from kirby.find_best_hybrid import find_best_hybrid
            from kirby.hybrid import apply_feature_selection
            from experiments._shared import FULL_BUDGETS
            _metric = 'spearman' if not is_clf else 'auc'
            # return_fitted defaults to False, and fitted_results is only
            # populated when it is True — so every call here returned an empty
            # dict and the arm reported "search returned no fitted 'rf'
            # result". It has never been built.
            search = find_best_hybrid(
                base_reps=all_reps_for_hybrid, labels=labels, groups=groups,
                budgets=FULL_BUDGETS,
                dataset_name=dataset_name, tier='general',
                metric=_metric, task_type=task_type,
                augmentation_strategy='none',
                adaptive_pruning=0.25,
                return_fitted=True)
            # Prefer rf, but take whatever the search actually fitted rather
            # than dropping the arm because one name is absent.
            fitted = search.fitted_results.get('rf')
            fitted_model = 'rf'
            if fitted is None and search.fitted_results:
                fitted_model = sorted(search.fitted_results)[0]
                fitted = search.fitted_results[fitted_model]
                print(f"  Hybrid: no fitted 'rf', using '{fitted_model}' "
                      f"(fitted: {sorted(search.fitted_results)})")
            if fitted is not None:
                fi = {r: {'selected_indices': idx}
                      for r, idx in fitted.selected_indices.items()}
                geo_reps['kirby_hybrid'] = apply_feature_selection(
                    all_reps_for_hybrid, fi)
                print(f"  Hybrid built from '{fitted_model}': "
                      f"{geo_reps['kirby_hybrid'].shape[1]}D")
            else:
                geo_failures['kirby_hybrid'] = (
                    "the search fitted no model at all "
                    "(find_best_hybrid returned an empty fitted_results)")
                print("  Hybrid build failed: no fitted 'rf' result")
        except Exception as e:
            # Keep the traceback. "name 'speed' is not defined" told us nothing
            # about where, and the failure is inside find_best_hybrid, not here.
            tb = traceback.format_exc()
            geo_failures['kirby_hybrid'] = (
                f"hybrid build raised: {type(e).__name__}: {e}\n{tb}")
            print(f"  Hybrid build failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        geo_failures['kirby_hybrid'] = (
            f"only {len(all_reps_for_hybrid)} rep(s) available, need >= 2")

    # 4. Get models
    models = get_models(model_names, n_samples=data['n_samples'],
                        task_type=task_type)
    print(f"  Models: {list(models.keys())}")

    # 5. Evaluate geometry features standalone
    print(f"\n--- Standalone geometry features ---")
    geo_baselines = {}
    geo_names = GEO_ORDER + ['all_geometry']

    # kirby_hybrid is built from base + geometry, so it is scored standalone
    # but never concatenated onto a base rep (it already contains one).
    standalone_names = geo_names + ['kirby_hybrid']
    if hybrid_only:
        # The augmentation grid is 10 reps x 8 groups x 2 models = 160 fits and
        # it is already measured. The hybrid arm is 2. Score the base reps alone
        # (20 fits, needed for any delta) and the hybrid, and nothing else.
        standalone_names = ['kirby_hybrid']

    for geo_name in standalone_names:
        X = geo_reps.get(geo_name)
        if X is None or np.all(X == 0):
            reason = geo_failures.get(
                geo_name, 'array missing or entirely zero at scoring time')
            geo_failures[geo_name] = reason
            print(f"  {geo_name}: FAILURE — {reason}")
            continue

        X = np.asarray(X, dtype=np.float64)
        dim = X.shape[1]
        print(f"  {geo_name} ({dim}D):", end='', flush=True)

        entry = {'dimension': dim, 'scores': {}}
        for mname, model in models.items():
            try:
                result = evaluate_rep(X, labels, groups, model, n_folds,
                                      task_type=task_type)
                entry['scores'][mname] = result
                if is_clf:
                    print(f" {mname}=acc:{result['accuracy_mean']:.3f}/"
                          f"F1:{result['f1_weighted_mean']:.3f}",
                          end='', flush=True)
                else:
                    print(f" {mname}={result['r2_mean']:.3f}/"
                          f"\u03c1={result['spearman_mean']:.3f}",
                          end='', flush=True)
            except Exception as e:
                print(f" {mname}=FAIL", end='', flush=True)
        geo_baselines[geo_name] = entry
        print()

    # 6. Augmentation tests: base + geometry
    augmentations = {}
    if hybrid_only:
        print(f"\n--- Augmentation grid skipped (--hybrid-only) ---")
    else:
        print(f"\n--- Augmentation tests ---")

        # Score each base representation ON ITS OWN first. Without this there is
        # nothing to subtract a concat score from, and "does geometry help" cannot
        # be answered from the result file at all — which is what the first two
        # completed runs produced. It costs one fit per base rep per model, about
        # 12% on top of the augmentation fits.
        base_baselines = {}
        print("  Base representations alone:")
        for base_name in sorted(base_reps.keys()):
            X_base = np.asarray(base_reps[base_name], dtype=np.float64)
            entry = {'dimension': X_base.shape[1], 'scores': {}}
            print(f"    {base_name} ({X_base.shape[1]}D):", end='', flush=True)
            for mname, model in models.items():
                try:
                    result = evaluate_rep(X_base, labels, groups, model, n_folds,
                                          task_type=task_type)
                    entry['scores'][mname] = result
                    key = 'accuracy_mean' if is_clf else 'spearman_mean'
                    print(f" {mname}={result[key]:.3f}", end='', flush=True)
                except Exception as e:
                    print(f" {mname}=FAIL", end='', flush=True)
            base_baselines[base_name] = entry
            print()

        for base_name in sorted(base_reps.keys()):
            X_base = base_reps[base_name]
            base_dim = X_base.shape[1]

            for geo_name in geo_names:
                X_geo = geo_reps.get(geo_name)
                if X_geo is None or np.all(X_geo == 0):
                    continue   # already recorded in geo_failures by the standalone loop

                X_aug = np.hstack([X_base, X_geo])
                aug_key = f"{base_name}+{geo_name}"
                aug_dim = X_aug.shape[1]

                print(f"  {aug_key} ({aug_dim}D):", end='', flush=True)

                entry = {
                    'base': base_name,
                    'geometry': geo_name,
                    'base_dim': base_dim,
                    'geo_dim': X_geo.shape[1],
                    'concat_dim': aug_dim,
                    'scores': {},
                }

                for mname, model in models.items():
                    try:
                        result = evaluate_rep(X_aug, labels, groups, model,
                                              n_folds, task_type=task_type)
                        entry['scores'][mname] = result
                        if is_clf:
                            print(f" {mname}=acc:{result['accuracy_mean']:.3f}/"
                                  f"F1:{result['f1_weighted_mean']:.3f}",
                                  end='', flush=True)
                        else:
                            print(f" {mname}={result['r2_mean']:.3f}/"
                                  f"\u03c1={result['spearman_mean']:.3f}",
                                  end='', flush=True)
                    except Exception as e:
                        print(f" {mname}=FAIL", end='', flush=True)

                # The delta is the whole point of the experiment, so store it
                # rather than leaving it to be recomputed by whoever reads this.
                primary_key = 'accuracy_mean' if is_clf else 'spearman_mean'
                base_scores = base_baselines.get(base_name, {}).get('scores', {})
                entry['base_scores'] = {
                    m: r.get(primary_key) for m, r in base_scores.items()}
                entry['delta'] = {}
                for mname, r in entry['scores'].items():
                    b = base_scores.get(mname, {}).get(primary_key)
                    v = r.get(primary_key)
                    if b is not None and v is not None:
                        entry['delta'][mname] = v - b

                augmentations[aug_key] = entry
                print()

    elapsed = time.time() - t0

    # A run that scored nothing is a FAILURE, not a completed run with an
    # empty results block.
    status = 'OK' if geo_baselines else 'FAILED'
    if status == 'FAILED':
        print(f"\n  FAILURE: no geometry feature group was scored. "
              f"Reasons: {geo_failures}")

    return {
        'status': status,
        'metadata': {
            'dataset': dataset_name,
            'n_samples': data['n_samples'],
            'n_groups': data['n_groups'],
            'n_folds': n_folds,
            'task_type': task_type,
            'base_reps': sorted(base_reps.keys()),
            'base_reps_requested': list(base_rep_names),
            'base_reps_missing': missing,
            'geo_reps': sorted(geo_baselines.keys()),
            'geo_reps_requested': standalone_names,
            'models': list(models.keys()),
            'elapsed_seconds': round(elapsed, 1),
            'timestamp': datetime.now().isoformat(),
        },
        'geometry_failures': geo_failures,
        'geometry_baselines': geo_baselines,
        'base_baselines': base_baselines,
        'augmentations': augmentations,
    }


# =============================================================================
# SUMMARY
# =============================================================================

def print_augmentation_summary(result: dict):
    """Print augmentation delta summary."""
    augmentations = result.get('augmentations', {})
    geo_baselines = result.get('geometry_baselines', {})
    is_clf = result.get('metadata', {}).get('task_type') == 'classification'
    primary = 'accuracy_mean' if is_clf else 'spearman_mean'
    metric_label = 'Acc' if is_clf else '\u03c1'

    if not augmentations:
        return

    print(f"\n{'='*70}")
    print(f"AUGMENTATION SUMMARY ({metric_label})")
    print(f"{'='*70}")

    # Group by base rep
    bases = {}
    for key, entry in augmentations.items():
        base = entry['base']
        if base not in bases:
            bases[base] = {}
        bases[base][entry['geometry']] = entry

    for base_name in sorted(bases.keys()):
        base_entry = result.get('base_baselines', {}).get(base_name, {})
        base_scores = base_entry.get('scores', {})
        base_line = "  ".join(
            f"{m}={sc.get(primary, float('nan')):.4f}"
            for m, sc in sorted(base_scores.items()))
        print(f"\n  Base: {base_name}  alone: {base_line or 'not scored'}")

        for geo_name, entry in sorted(bases[base_name].items()):
            scores = entry.get('scores', {})
            deltas = entry.get('delta', {})
            best_model = None
            best_val = -float('inf')
            for mname, sc in scores.items():
                val = sc.get(primary, float('nan'))
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    best_model = mname
            if not best_model:
                continue
            d = deltas.get(best_model)
            if d is None:
                print(f"    +{geo_name:<30} {metric_label}={best_val:.4f} "
                      f"({best_model})  delta unavailable: base not scored")
            else:
                print(f"    +{geo_name:<30} {metric_label}={best_val:.4f} "
                      f"({best_model})  delta {d:+.4f}")


def _convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# PRE-SUBMIT CHECK
# =============================================================================

def check_base_reps(datasets: List[str], requested: Optional[str]) -> int:
    """Say which base representations each dataset would find, and exit.

    Predicts no structure and fits no model. It does load each dataset, because
    one of the two caches is keyed on a hash of the sequence list and the hash
    cannot be known without the sequences.

    Exit code 0 only when every dataset has every representation it would ask
    for. That is the gate: a non-zero exit means submitting the survey would
    spend cluster time and record FAILED.
    """
    total_missing = 0
    print(f"Checking base representations for {len(datasets)} dataset(s)\n")

    for ds in datasets:
        try:
            data = load_dataset(ds)
        except Exception as e:
            print(f"{ds}: could not load — {type(e).__name__}: {e}")
            total_missing += 1
            continue

        names = resolve_base_rep_names(ds, requested)
        heavy_seqs = data['heavy_seqs']
        light_seqs = data['light_seqs']

        found, absent = [], []
        for rep_name in names:
            by_seq = _rep_cache_path(heavy_seqs, light_seqs, rep_name)
            by_name = _precompute_cache_path(ds, rep_name)
            if by_seq.exists():
                found.append(f"{rep_name} (by sequence hash)")
            elif by_name.exists():
                found.append(f"{rep_name} (by dataset name)")
            elif rep_name in {'onehot', 'blosum', 'aac', 'paratope'}:
                found.append(f"{rep_name} (computed by the survey itself)")
            else:
                absent.append(rep_name)

        total_missing += len(absent)
        mark = "OK  " if not absent else "MISSING"
        print(f"{mark} {ds}  {len(found)}/{len(names)}")
        for f in found:
            print(f"       have    {f}")
        for a in absent:
            print(f"       ABSENT  {a}")
        if absent:
            print(f"       looked in $KIRBY_CACHE_DIR/reps/"
                  f"ab_{_rep_cache_path(heavy_seqs, light_seqs, 'x').parent.name[3:]}/"
                  f" and $KIRBY_CACHE_DIR/rep_cache/{ds}/")
        print()

    if total_missing:
        print(f"{total_missing} representation(s) absent. The survey would "
              f"record FAILED for those datasets. Do not submit yet.")
        return 1
    print("Every dataset has every representation it would ask for. "
          "Safe to submit.")
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Structural geometry augmentation survey for antibodies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests 7 structure-based geometry features as augmentations on top of
base antibody representations. Requires ImmuneBuilder for structure
prediction (structures are cached after first computation).

Geometry features:
  vhvl_angles (6D)          CDR loop shape (60D)
  cdr_canonical (12D)       paratope_charge (12D)
  paratope_hydrophobicity (12D)   cdr_contacts (30D)
  vhvl_interface (30D)      all_geometry (162D, all combined)

Examples:
  %(prog)s flab_thermo_jain --models rf,mlp
  %(prog)s all_inter --base-reps esm2,ablang2,prot_t5
  %(prog)s all --output tests/results/geometry_survey/
        """
    )

    parser.add_argument('dataset', type=str, nargs='?', default=None,
                        help='Dataset name, group shortcut, or "all"')
    parser.add_argument('--base-reps', type=str, default=None,
                        help=f'Comma-separated base reps, or "pruned" to use '
                             f'DATASET_REPS[dataset]["general"], the pool signed '
                             f'off during rep pruning for that dataset '
                             f'(default: {",".join(DEFAULT_BASE_REPS)})')
    parser.add_argument('--hybrid-only', action='store_true',
                        help='Score the base representations alone and the '
                             'kirby_hybrid arm, and skip the augmentation grid. '
                             'The grid is 160 fits per dataset and is already '
                             'measured; the hybrid arm is 2 and has never run.')
    parser.add_argument('--check-base-reps', action='store_true',
                        help='Report, per dataset, which requested base '
                             'representations are cached and which are not, '
                             'then exit. Predicts no structure and fits no '
                             'model. Exit code is 0 only when every dataset '
                             'has every representation it would ask for, so '
                             'this is the gate to run before submitting.')
    parser.add_argument('--allow-missing-base-reps', action='store_true',
                        help='Score against whichever base representations are '
                             'cached instead of refusing. Without this a '
                             'dataset missing any requested representation is '
                             'recorded FAILED rather than run against the rest.')
    parser.add_argument('--models', type=str, default=DEFAULT_MODELS_ARG,
                        help=f'Comma-separated model names (default: {DEFAULT_MODELS_ARG})')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for JSON results')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip any dataset whose result JSON already exists '
                             'in --output and did not record status FAILED. '
                             'Makes the run re-enterable after an srun hits its '
                             'wall clock.')
    parser.add_argument('--scan-zero-cache', action='store_true',
                        help='Report which cached geometry arrays under '
                             'KIRBY_CACHE_DIR are entirely zero, then exit. '
                             'Reports only — deletes nothing.')

    args = parser.parse_args()

    if args.scan_zero_cache:
        zero_files = scan_zero_geometry_cache()
        return 1 if zero_files else 0

    if not args.dataset:
        parser.error('dataset is required (or pass --scan-zero-cache)')

    # Resolve datasets
    ds_input = args.dataset
    if ds_input in DOMAIN_GROUPS:
        datasets = list(DOMAIN_GROUPS[ds_input])
    elif ds_input in DATASET_REGISTRY:
        datasets = [ds_input]
    else:
        print(f"ERROR: Unknown dataset '{ds_input}'")
        print(f"Available: {sorted(DATASET_REGISTRY.keys())}")
        print(f"Groups: {sorted(DOMAIN_GROUPS.keys())}")
        return 1

    if args.check_base_reps:
        return check_base_reps(datasets, args.base_reps)

    # Base reps are resolved per dataset, because "pruned" means a different
    # list for each one.
    model_names = [m.strip() for m in args.models.split(',')]

    print(f"Structural Geometry Augmentation Survey")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  Base reps: {args.base_reps or ','.join(DEFAULT_BASE_REPS)}")
    print(f"  Models: {model_names}")
    print(f"  Folds: {args.n_folds}")

    # Output
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run
    all_results = {}
    pass_count = 0
    fail_count = 0

    skip_count = 0

    for ds in datasets:
        # An srun that runs out of wall clock must not cost what it already
        # finished. Each dataset writes its own JSON, so a completed one is
        # reloaded rather than recomputed. A FAILED file is retried.
        if args.skip_existing and output_dir:
            done_path = output_dir / f"geometry_{ds}.json"
            if done_path.exists():
                try:
                    with open(done_path) as f:
                        prior = json.load(f)
                    if prior.get('status') != 'FAILED':
                        print(f"\n  Skipping {ds}: already in {done_path}")
                        all_results[ds] = prior
                        skip_count += 1
                        continue
                    print(f"\n  Retrying {ds}: prior run recorded FAILED")
                except Exception as e:
                    print(f"\n  Retrying {ds}: could not read {done_path}: {e}")

        try:
            base_rep_names = resolve_base_rep_names(ds, args.base_reps)
            result = run_geometry_survey(
                ds, base_rep_names, model_names, args.n_folds,
                allow_missing_base_reps=args.allow_missing_base_reps,
                hybrid_only=args.hybrid_only)
            result = _convert_numpy(result)
            print_augmentation_summary(result)
            all_results[ds] = result
            if result.get('status') == 'FAILED':
                print(f"  FAILED: {ds}: {result.get('error', 'nothing scored')}")
                fail_count += 1
            else:
                pass_count += 1

            if output_dir:
                # A distinct PREFIX, not a suffix: the readers glob
                # geometry_*.json and a hybrid-only file must not be mistaken
                # for a full run of the same dataset.
                stem = f"hybridarm_{ds}" if args.hybrid_only else f"geometry_{ds}"
                out_path = output_dir / f"{stem}.json"
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n  FAILED: {ds}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DATASET GEOMETRY SUMMARY")
        print(f"{'='*70}")

        for ds, r in all_results.items():
            geo = r.get('geometry_baselines', {})
            is_clf = r.get('metadata', {}).get('task_type') == 'classification'
            primary = 'accuracy_mean' if is_clf else 'spearman_mean'
            metric_label = 'Acc' if is_clf else '\u03c1'

            print(f"\n  {ds} (standalone):")
            for geo_name, entry in sorted(geo.items()):
                scores = entry.get('scores', {})
                best = max(
                    (s.get(primary, float('-inf')) for s in scores.values()),
                    default=float('nan'))
                print(f"    {geo_name:<30} best {metric_label}={best:.4f}")
            for geo_name, reason in sorted(r.get('geometry_failures', {}).items()):
                print(f"    {geo_name:<30} FAILURE — {reason}")

    tail = f", {skip_count} already done" if skip_count else ""
    print(f"\nDone: {pass_count} passed, {fail_count} failed{tail}")

    # Exit non-zero only when NOTHING scored. Twelve of the twenty datasets
    # this survey runs on have no pruned pool and refuse by design, so exiting
    # on any failure made SLURM report FAILED for a job that did all the work
    # it could — indistinguishable from a crash in sacct.
    if pass_count + skip_count == 0:
        print("Nothing scored, so this run produced no result. Exit 1.")
        return 1
    if fail_count:
        print(f"{fail_count} dataset(s) failed and {pass_count} scored, so the "
              f"results directory holds real results. Exit 0. Read which "
              f"failed and why with "
              f"tests/analysis/read_geometry_survey.py <output dir>.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
