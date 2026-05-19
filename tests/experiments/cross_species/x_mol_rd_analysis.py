#!/usr/bin/env python3
"""x_mol_rd_analysis — Downstream R&D analysis on the X-Mol cache (T-Mol.5).

Loads each cached `x_mol_species_admet_<task>` hybrid blueprint (T-Mol.4),
refits the best (model, budget) on the full task dataset, attaches a
chemistry-aware conformal regressor (T-UQ.2 ``ChemBinnedConformalRegressor``)
calibrated on a scaffold-disjoint fold, then predicts per-(task, species)
point + interval for one or more candidate compounds by toggling the
species-token augmentation. Emits one JSON table with three liability flags
per (candidate, task):

  - ``out_of_domain``         — candidate falls in the rightmost distance bin.
  - ``below_potency_floor``   — interval lower bound is below the configured
                                pActivity floor for the task (optional).
  - ``species_divergence``    — for the same candidate + task, any two species
                                intervals do not overlap.

No new ML hyperparameters: the script ONLY consumes the existing T-Mol.4 cache
+ T-UQ.2 conformal API. If a task's hybrid cache is missing, the task is
skipped with a clear log line so the script is safe to run before every
T-Mol.4 SLURM run has finished.

Usage
-----

    # Dry-run on synthetic candidate, prints the result table
    python tests/experiments/cross_species/x_mol_rd_analysis.py --dry-run

    # Real candidate, all tasks
    python tests/experiments/cross_species/x_mol_rd_analysis.py \\
        --candidate-smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \\
        --candidate-name aspirin

    # Restrict to one task, raise the potency floor for that task
    python tests/experiments/cross_species/x_mol_rd_analysis.py \\
        --candidate-smiles 'CC(=O)Nc1ccc(O)cc1' --candidate-name paracetamol \\
        --tasks cox2 --potency-floor cox2=5.0
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    get_representations, assign_scaffold_groups,
)
from experiments._shared import convert_numpy
from experiments.cross_species.x_mol_species_admet import (
    SPECIES_VOCAB, species_one_hot, admet_tasks, _assemble_task,
)


# Cache layout matches T-Mol.4 exactly. If the user re-runs T-Mol.4 with a
# different tier / aug_strategy / speed, expose them here.
DEFAULT_TIER = 'general'
DEFAULT_AUG_STRATEGY = 'uncapped'
DEFAULT_SPEED = 'full'

# Conformal target coverage. alpha=0.1 → 90% intervals (regulatory default).
DEFAULT_ALPHA = 0.1


# --------------------------------------------------------------------------
# Cache loading
# --------------------------------------------------------------------------


def _cache_base() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _search_cache_path(task_name: str,
                       tier: str = DEFAULT_TIER,
                       aug_strategy: str = DEFAULT_AUG_STRATEGY,
                       speed: str = DEFAULT_SPEED) -> Path:
    """Path to a T-Mol.4 SearchResult cache.

    Mirrors `kirby.find_best_hybrid._search_cache_path` exactly so the
    analysis script reads the same file the experiment script wrote.
    """
    stem = (f'x_mol_species_admet_{task_name}__'
            f'{tier}__{aug_strategy}__{speed}')
    return _cache_base() / 'searches' / f'{stem}.pkl'


def load_search_result(task_name: str, **path_kwargs):
    """Load the cached SearchResult for a T-Mol.4 task, or None.

    Validates only that the pickle unwraps to a SearchResult-shaped object
    with at least one entry in either ``best_by_model`` or
    ``best_by_model_budget`` — meta-drift is NOT checked because this
    script never recomputes the search.
    """
    path = _search_cache_path(task_name, **path_kwargs)
    if not path.exists():
        return None
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as exc:
        print(f"  [{task_name}] cache unpickle failed: {exc}")
        return None
    payload = (wrapper.get('payload')
               if isinstance(wrapper, dict) else wrapper)
    if payload is None:
        return None
    has_model = bool(getattr(payload, 'best_by_model', None))
    has_budget = bool(getattr(payload, 'best_by_model_budget', None))
    if not (has_model or has_budget):
        return None
    return payload


def select_blueprint(search) -> Optional[Dict[str, Any]]:
    """Pick the best result-dict from a SearchResult.

    Iterates ``best_by_model`` (which is the level at which
    ``fitted_results`` is populated by ``return_fitted=True``), ranks by
    ``score_mean`` honoring the SearchResult's ``lower_is_better``, and
    returns the winning result-dict (carries ``model``, ``budget``,
    ``score_mean``, ``score_std``, ``n_features``, ``blueprint``).
    """
    by_model = getattr(search, 'best_by_model', None) or {}
    if not by_model:
        return None
    lower_is_better = bool(getattr(search, 'lower_is_better', False))

    def _key(item):
        s = item[1].get('score_mean', float('nan'))
        return -s if lower_is_better else s

    ranked = sorted(by_model.items(), key=_key, reverse=True)
    _, best_rec = ranked[0]
    return best_rec


# --------------------------------------------------------------------------
# Per-task pipeline
# --------------------------------------------------------------------------


def _build_features_from_fitted(
    smiles: List[str], organisms: List[str], fitted,
) -> np.ndarray:
    """Concatenate selected base + augmentation features using a FittedResult.

    Bypasses ``apply_blueprint`` because the cached ``FittedResult`` already
    holds the per-rep ``selected_indices`` plus the augmentation
    ``selected_indices`` — re-running the importance scoring would be
    wasteful and the candidate inference path must not depend on labels
    at all.
    """
    from kirby.hybrid import apply_augmentation_selection

    rep_names = list(fitted.selected_indices.keys())
    base_reps = get_representations(smiles, ','.join(rep_names))
    missing = [r for r in rep_names if r not in base_reps]
    if missing:
        raise RuntimeError(
            f"base reps missing after rebuild: {missing}. The cache "
            f"references reps the current code can no longer compute.")

    parts: List[np.ndarray] = []
    for rep_name, indices in fitted.selected_indices.items():
        X = np.asarray(base_reps[rep_name], dtype=np.float32)
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        if idx.max() >= X.shape[1]:
            raise RuntimeError(
                f"Cached selected_indices for '{rep_name}' max={idx.max()} "
                f"≥ rep dim {X.shape[1]} — rep was rebuilt with a smaller "
                f"dimension; cache is stale.")
        parts.append(X[:, idx])

    if fitted.augmentation_info:
        species_aug = species_one_hot(organisms)
        aug_X = apply_augmentation_selection(
            {'species_token': species_aug}, fitted.augmentation_info)
        if aug_X is not None and aug_X.size:
            parts.append(np.asarray(aug_X, dtype=np.float32))

    if not parts:
        raise RuntimeError(
            "FittedResult has no usable selected features — cache is empty.")
    return np.hstack(parts).astype(np.float32)


def _calibration_split(groups: np.ndarray, n_folds: int = 5):
    """Last GroupKFold fold = calibration; the rest = training."""
    from sklearn.model_selection import GroupKFold
    n_folds = max(2, min(n_folds, len(set(groups))))
    splitter = GroupKFold(n_splits=n_folds)
    splits = list(splitter.split(np.zeros((len(groups), 1)),
                                 groups=groups))
    # Pick the largest calibration fold (= last under sklearn's deterministic order).
    train_idx, calib_idx = splits[-1]
    return train_idx, calib_idx, n_folds


def fit_calibrated_model(
    smiles: List[str], labels: np.ndarray, organisms: List[str],
    groups: np.ndarray, model_name: str, fitted_blueprint,
    alpha: float = DEFAULT_ALPHA, n_folds_for_calib: int = 5,
):
    """Fit the model on the train split, calibrate the conformal regressor.

    ``fitted_blueprint`` is the SearchResult.fitted_results entry — it
    carries ``selected_indices`` and ``augmentation_info`` so the feature
    matrix can be reconstructed without re-running importance.

    Returns a dict with the fitted conformal regressor, the train SMILES,
    and provenance for the JSON output.
    """
    from kirby.find_best_hybrid import get_model
    from kirby.uq.conformal_chem import ChemBinnedConformalRegressor

    X_hybrid = _build_features_from_fitted(smiles, organisms, fitted_blueprint)
    train_idx, calib_idx, n_folds = _calibration_split(
        groups, n_folds=n_folds_for_calib)

    X_train, y_train = X_hybrid[train_idx], labels[train_idx]
    smiles_train = [smiles[i] for i in train_idx]
    X_calib, y_calib = X_hybrid[calib_idx], labels[calib_idx]
    smiles_calib = [smiles[i] for i in calib_idx]

    estimator = get_model(model_name, task_type='regression',
                          n_samples=len(train_idx))
    if estimator is None:
        raise RuntimeError(
            f"get_model returned None for '{model_name}' (n={len(train_idx)})")
    estimator.fit(X_train, y_train)

    conformal = ChemBinnedConformalRegressor(estimator=estimator, alpha=alpha)
    conformal.fit_prefit(smiles_train, X_calib, y_calib, smiles_calib)
    cov = conformal.score(X_calib, y_calib, smiles_calib)

    return {
        'conformal': conformal,
        'smiles_train': smiles_train,
        'n_train': len(train_idx),
        'n_calib': len(calib_idx),
        'n_folds_for_calib': n_folds,
        'model_name': model_name,
        'calibration_coverage': cov,
    }


def predict_candidate(
    fitted: Dict[str, Any], fitted_blueprint,
    candidate_smiles: str, species_for_task: List[str],
) -> Dict[str, Any]:
    """Per-species prediction + interval for one candidate.

    Returns a dict with parallel lists keyed by species (in the same order as
    `species_for_task`). The candidate SMILES is replicated len(species)
    times so the species-token augmentation flips on a single row each.
    """
    k = len(species_for_task)
    if k == 0:
        raise ValueError("species_for_task is empty")
    candidate_list = [candidate_smiles] * k
    X_cand = _build_features_from_fitted(
        candidate_list, species_for_task, fitted_blueprint)
    iv = fitted['conformal'].predict(np.asarray(X_cand, dtype=np.float64),
                                     candidate_list)
    # Distance bin per row (for the out-of-domain flag).
    conf = fitted['conformal']
    from kirby.uq.conformal_chem import _distance_to_reference
    cand_fps = conf._fp(candidate_list)
    distances = _distance_to_reference(cand_fps, conf.train_fps_)
    bins = conf._assign_bins(distances)
    return {
        'species': species_for_task,
        'point': iv.point.tolist(),
        'lower': iv.lower.tolist(),
        'upper': iv.upper.tolist(),
        'distance_to_train': distances.tolist(),
        'bin': [int(b) for b in bins],
        'n_bins': int(conf.n_bins_eff_),
    }


# --------------------------------------------------------------------------
# Liability flags
# --------------------------------------------------------------------------


def flag_out_of_domain(prediction: Dict[str, Any]) -> bool:
    """True iff ANY species lands in the rightmost (highest-distance) bin."""
    top_bin = prediction['n_bins'] - 1
    return any(b >= top_bin for b in prediction['bin'])


def flag_below_potency_floor(prediction: Dict[str, Any],
                             floor: Optional[float]) -> bool:
    """True iff floor is set and the lower bound dips below it for any species."""
    if floor is None:
        return False
    return any(lb < floor for lb in prediction['lower'])


def flag_species_divergence(prediction: Dict[str, Any]) -> bool:
    """True iff any pair of per-species intervals are non-overlapping.

    Two intervals [a,b] and [c,d] overlap iff a<=d and c<=b. So a divergence
    is the negation: ANY pair fails that test.
    """
    lowers = prediction['lower']
    uppers = prediction['upper']
    k = len(lowers)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = lowers[i], uppers[i]
            c, d = lowers[j], uppers[j]
            if not (a <= d and c <= b):
                return True
    return False


def analyse_candidate(
    candidate_name: str, candidate_smiles: str,
    tasks_to_run: Optional[List[str]],
    potency_floors: Dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    n_folds_for_calib: int = 5,
    dry_run: bool = False,
    cache_path_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all (or selected) tasks for one candidate, return one result dict."""
    cache_path_kwargs = cache_path_kwargs or {}
    wanted = set(tasks_to_run) if tasks_to_run else None
    out: Dict[str, Any] = {
        'candidate_name': candidate_name,
        'candidate_smiles': candidate_smiles,
        'alpha': alpha,
        'per_task': {},
    }

    for task_name, organisms, loader in admet_tasks():
        if wanted is not None and task_name not in wanted:
            continue
        print(f"\n[{task_name}] candidates: {organisms}")
        search = load_search_result(task_name, **cache_path_kwargs)
        if search is None:
            print(f"  cache miss — skipping (run x_mol_species_admet first)")
            out['per_task'][task_name] = {'skipped': True,
                                          'reason': 'cache miss'}
            continue
        best_rec = select_blueprint(search)
        if best_rec is None:
            out['per_task'][task_name] = {'skipped': True,
                                          'reason': 'no blueprints in cache'}
            continue
        blueprint = best_rec['blueprint']
        model_name = best_rec['model']
        budget = best_rec['budget']
        fitted_blueprint = (search.fitted_results or {}).get(model_name)
        if fitted_blueprint is None:
            out['per_task'][task_name] = {
                'skipped': True,
                'reason': f"FittedResult missing for '{model_name}' "
                          f"(was the search run with return_fitted=True?)",
            }
            continue
        print(f"  selected: {model_name} @ budget={budget} "
              f"score={best_rec.get('score_mean'):.4f}")

        bundle = _assemble_task(task_name, organisms, loader)
        if bundle is None:
            out['per_task'][task_name] = {'skipped': True,
                                          'reason': 'insufficient coverage'}
            continue
        smiles = bundle['smiles']
        y = bundle['labels']
        organisms_per_row = bundle['organisms']
        groups, n_scaffolds = assign_scaffold_groups(smiles)

        if dry_run:
            # n_folds_for_calib=2 is the smallest GroupKFold; the rest of the
            # flow is unchanged so the table matches the full-run shape.
            calib_folds = 2
        else:
            calib_folds = n_folds_for_calib

        t0 = time.time()
        fitted = fit_calibrated_model(
            smiles=smiles, labels=y, organisms=organisms_per_row,
            groups=groups, model_name=model_name,
            fitted_blueprint=fitted_blueprint,
            alpha=alpha, n_folds_for_calib=calib_folds,
        )
        # Only run the candidate against species that actually have data
        # for this task — `bundle['per_organism_counts']` is the authoritative
        # list. species_one_hot will raise on anything outside SPECIES_VOCAB.
        species_for_task = list(bundle['per_organism_counts'].keys())
        prediction = predict_candidate(
            fitted, fitted_blueprint, candidate_smiles, species_for_task)
        elapsed = time.time() - t0

        floor = potency_floors.get(task_name)
        flags = {
            'out_of_domain': flag_out_of_domain(prediction),
            'below_potency_floor': flag_below_potency_floor(prediction, floor),
            'species_divergence': flag_species_divergence(prediction),
        }
        out['per_task'][task_name] = {
            'skipped': False,
            'model': model_name,
            'budget': budget,
            'blueprint_score_mean': best_rec.get('score_mean'),
            'blueprint_score_std': best_rec.get('score_std'),
            'n_features': best_rec.get('n_features'),
            'reps_selected': list(blueprint.reps),
            'species_token_in_blueprint': bool(
                fitted_blueprint.augmentation_info
                and 'species_token' in fitted_blueprint.augmentation_info
            ),
            'n_samples': len(smiles),
            'n_scaffolds': int(n_scaffolds),
            'n_train': fitted['n_train'],
            'n_calib': fitted['n_calib'],
            'n_folds_for_calib': fitted['n_folds_for_calib'],
            'calibration_coverage': fitted['calibration_coverage'],
            'prediction': prediction,
            'potency_floor': floor,
            'flags': flags,
            'time_s': elapsed,
        }
        for sp, pt, lo, hi, b in zip(prediction['species'],
                                     prediction['point'],
                                     prediction['lower'],
                                     prediction['upper'],
                                     prediction['bin']):
            print(f"  {sp:8s}  pActivity={pt:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]  bin={b}")
        if any(flags.values()):
            on = [k for k, v in flags.items() if v]
            print(f"  flags raised: {on}")

    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_potency_floors(s: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for piece in s.split(','):
        piece = piece.strip()
        if not piece:
            continue
        if '=' not in piece:
            raise ValueError(
                f"--potency-floor expects task=value pairs, got {piece!r}")
        k, v = piece.split('=', 1)
        out[k.strip()] = float(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-smiles', type=str, default=None,
                    help='SMILES of one candidate compound.')
    ap.add_argument('--candidate-name', type=str, default='candidate')
    ap.add_argument('--candidates-json', type=str, default=None,
                    help='Optional JSON file: [{"name": ..., "smiles": ...}, ...].')
    ap.add_argument('--tasks', type=str, default=None,
                    help='Comma-separated task names (default: all).')
    ap.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                    help='Conformal miscoverage rate (0.1 → 90%% intervals).')
    ap.add_argument('--n-folds-for-calib', type=int, default=5,
                    help='GroupKFold splits; last fold is calibration.')
    ap.add_argument('--potency-floor', type=str, default=None,
                    help='Comma-separated task=floor pairs, e.g. cox2=5.0,drd2=6.0.')
    ap.add_argument('--tier', type=str, default=DEFAULT_TIER)
    ap.add_argument('--aug-strategy', type=str, default=DEFAULT_AUG_STRATEGY)
    ap.add_argument('--speed', type=str, default=DEFAULT_SPEED)
    ap.add_argument('--dry-run', action='store_true',
                    help='2-fold calib split — local sanity only.')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    if args.candidate_smiles is None and args.candidates_json is None and \
            not args.dry_run:
        raise SystemExit(
            "Provide --candidate-smiles or --candidates-json (or --dry-run).")

    candidates: List[Tuple[str, str]] = []
    if args.candidates_json:
        data = json.loads(Path(args.candidates_json).read_text())
        for entry in data:
            candidates.append((entry['name'], entry['smiles']))
    if args.candidate_smiles:
        candidates.append((args.candidate_name, args.candidate_smiles))
    if not candidates:
        # Dry-run convenience: aspirin.
        candidates.append(('aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'))

    tasks_to_run = args.tasks.split(',') if args.tasks else None
    potency_floors = _parse_potency_floors(args.potency_floor)
    path_kwargs = {'tier': args.tier,
                   'aug_strategy': args.aug_strategy,
                   'speed': args.speed}

    print('=' * 70)
    print(f"x_mol_rd_analysis  (T-Mol.5)  candidates={len(candidates)} "
          f"dry_run={args.dry_run}")
    print(f"  alpha={args.alpha}  tasks="
          f"{tasks_to_run if tasks_to_run else 'ALL'}")
    print('=' * 70)

    all_results = []
    for name, smi in candidates:
        print(f"\n{'#' * 70}\nCandidate: {name}  SMILES={smi}\n{'#' * 70}")
        res = analyse_candidate(
            candidate_name=name, candidate_smiles=smi,
            tasks_to_run=tasks_to_run, potency_floors=potency_floors,
            alpha=args.alpha, n_folds_for_calib=args.n_folds_for_calib,
            dry_run=args.dry_run, cache_path_kwargs=path_kwargs,
        )
        all_results.append(res)

    suffix = '_dry' if args.dry_run else ''
    out_default = (f'tests/results/cross_species/'
                   f'x_mol_rd_analysis{suffix}.json')
    out = Path(args.output or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(all_results), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
