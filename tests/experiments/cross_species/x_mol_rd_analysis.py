#!/usr/bin/env python3
"""x_mol_rd_analysis — Downstream R&D analysis on the X-Mol cache (T-Mol.5).

Loads each cached `x_mol_species_admet_<task>` hybrid blueprint (T-Mol.4) and
splits the task's molecules into three scaffold-disjoint blocks. The blueprint's
recipe chooses features and the model is fitted on the first block; a
chemistry-aware conformal regressor (T-UQ.2 ``ChemBinnedConformalRegressor``)
takes its interval width from the second; coverage is counted on the third. It
then predicts per-(task, species) point + interval for one or more candidate
compounds by toggling the species-token augmentation. Emits one JSON table with
three liability flags per (candidate, task):

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
# different tier / aug_strategy, expose them here.
DEFAULT_TIER = 'general'
DEFAULT_AUG_STRATEGY = 'uncapped'

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
                       aug_strategy: str = DEFAULT_AUG_STRATEGY) -> Path:
    """Path to a T-Mol.4 SearchResult cache.

    Mirrors `kirby.find_best_hybrid._search_cache_path` exactly so the
    analysis script reads the same file the experiment script wrote.
    """
    stem = (f'x_mol_species_admet_{task_name}__'
            f'{tier}__{aug_strategy}')
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
    # Metric name first, stored flag second, raise when neither. Fixed 2026-09-10.
    from experiments._shared import object_direction
    lower_is_better = object_direction(search, 'this SearchResult')

    def _key(item):
        s = item[1].get('score_mean', float('nan'))
        return -s if lower_is_better else s

    ranked = sorted(by_model.items(), key=_key, reverse=True)
    _, best_rec = ranked[0]
    return best_rec


# --------------------------------------------------------------------------
# Per-task pipeline
# --------------------------------------------------------------------------


def _reps_for(smiles: List[str], rep_names: List[str]) -> Dict[str, np.ndarray]:
    """The blueprint's base representations for a list of molecules."""
    base_reps = get_representations(smiles, ','.join(rep_names))
    missing = [r for r in rep_names if r not in base_reps]
    if missing:
        raise RuntimeError(
            f"base reps missing after rebuild: {missing}. The cache "
            f"references reps the current code can no longer compute.")
    return {r: np.asarray(base_reps[r], dtype=np.float32) for r in rep_names}


def _apply_selection(base_reps: Dict[str, np.ndarray],
                     organisms: List[str], feature_info: dict) -> np.ndarray:
    """Apply one already-chosen feature selection to a block of molecules.

    `feature_info` is what apply_blueprint returns on the fit rows. Everything
    here is an index lookup, so no label of the block being built is read.
    """
    from kirby.hybrid import (
        apply_feature_selection, apply_augmentation_selection,
    )
    sel_reps = {k: v for k, v in base_reps.items() if k in feature_info}
    X = apply_feature_selection(sel_reps, feature_info)
    aug_info = feature_info.get('augmentation_info')
    if aug_info:
        # species_token is the only donor this script supplies. A top_k recipe
        # can move a zeroed representation's columns into the donor pool inside
        # create_hybrid, and that donor cannot be rebuilt from here.
        other = [k for k in aug_info if k != 'species_token']
        if other:
            raise RuntimeError(
                f"the recipe kept extra features from {other}, which this "
                f"script cannot rebuild — it supplies species_token only.")
        aug_X = apply_augmentation_selection(
            {'species_token': species_one_hot(organisms)}, aug_info)
        if aug_X is not None and aug_X.size:
            X = np.hstack([X, np.asarray(aug_X, dtype=np.float32)])
    if X is None or X.size == 0:
        raise RuntimeError("the blueprint selected no features.")
    return np.asarray(X, dtype=np.float32)


def _metric_fn_for(blueprint):
    """The metric the search ranked with, as f(y_true, y_pred), plus direction."""
    from kirby.find_best_hybrid import BUILTIN_METRICS
    name = getattr(blueprint, 'metric', None)
    entry = BUILTIN_METRICS.get(name)
    if entry is None:
        from experiments._shared import object_direction
        return None, object_direction(blueprint, 'this blueprint')
    fn, lower = entry
    return fn, bool(lower)


def _calibration_split(groups: np.ndarray, n_folds: int = 5):
    """Three scaffold-disjoint blocks: fit rows, width rows, coverage rows.

    Returns (train_idx, calib_idx, cover_idx, n_folds).

    The last GroupKFold fold is where coverage is counted, the second-to-last
    sets the interval width, and everything else fits the model and chooses its
    features. Until 2026-09-04 there were two blocks: width came from the last
    fold and coverage was counted on that same fold, so the reported percentage
    could not come out below 1-alpha whatever the model did. The features came
    from a search over every row, so those molecules' labels had also chosen
    which features the model was fitted on.
    """
    from sklearn.model_selection import GroupKFold
    n_groups = len(set(groups))
    if n_groups < 3:
        raise ValueError(
            f"a coverage number needs three scaffold-disjoint blocks and this "
            f"task has {n_groups} scaffold group(s).")
    n_folds = max(3, min(n_folds, n_groups))
    splitter = GroupKFold(n_splits=n_folds)
    splits = list(splitter.split(np.zeros((len(groups), 1)),
                                 groups=groups))
    cover_idx = splits[-1][1]
    calib_idx = splits[-2][1]
    held = np.concatenate([cover_idx, calib_idx])
    train_idx = np.setdiff1d(np.arange(len(groups)), held)
    return train_idx, calib_idx, cover_idx, n_folds


def fit_calibrated_model(
    smiles: List[str], labels: np.ndarray, organisms: List[str],
    groups: np.ndarray, model_name: str, blueprint,
    alpha: float = DEFAULT_ALPHA, n_folds_for_calib: int = 5,
):
    """Choose features and fit on one block, set the width on a second, count
    coverage on a third.

    ``blueprint`` is the recipe the T-Mol.4 search selected — allocation
    method, importance method, evaluator, budget, augmentation. It is applied
    here to the fit block only, so the molecules whose residuals set the
    interval width, and the molecules coverage is counted on, never contribute
    a label to which features the model gets.

    Returns a dict with the fitted conformal regressor, the fit-block SMILES,
    the feature selection (so a candidate molecule can be built the same way),
    and provenance for the JSON output.
    """
    from kirby.find_best_hybrid import get_model, apply_blueprint
    from kirby.uq.conformal_chem import ChemBinnedConformalRegressor

    train_idx, calib_idx, cover_idx, n_folds = _calibration_split(
        groups, n_folds=n_folds_for_calib)

    rep_names = list(blueprint.reps)
    base_reps = _reps_for(smiles, rep_names)
    species_aug = species_one_hot(organisms)

    def _block(idx):
        return ({r: X[idx] for r, X in base_reps.items()},
                [smiles[i] for i in idx],
                [organisms[i] for i in idx],
                labels[idx])

    tr_reps, smiles_train, org_train, y_train = _block(train_idx)
    _, smiles_calib, org_calib, y_calib = _block(calib_idx)
    _, smiles_cover, org_cover, y_cover = _block(cover_idx)

    estimator = get_model(model_name, task_type='regression',
                          n_samples=len(train_idx))
    if estimator is None:
        raise RuntimeError(
            f"get_model returned None for '{model_name}' (n={len(train_idx)})")

    metric_fn, lower_is_better = _metric_fn_for(blueprint)
    X_train, feature_info = apply_blueprint(
        tr_reps, blueprint, labels=y_train,
        augmentations={'species_token': species_aug[train_idx]},
        groups=np.asarray(groups)[train_idx],
        model=(estimator if metric_fn is not None else None),
        metric_fn=metric_fn,
        lower_is_better=lower_is_better,
    )
    # create_hybrid already appended the kept species-token columns to what it
    # returned, so nothing is added here. The other two blocks go through
    # _apply_selection, which appends them itself.
    X_train = np.asarray(X_train, dtype=np.float32)

    X_calib = _apply_selection(
        {r: X[calib_idx] for r, X in base_reps.items()}, org_calib, feature_info)
    X_cover = _apply_selection(
        {r: X[cover_idx] for r, X in base_reps.items()}, org_cover, feature_info)

    estimator.fit(X_train, y_train)

    conformal = ChemBinnedConformalRegressor(estimator=estimator, alpha=alpha)
    conformal.fit_prefit(smiles_train, X_calib, y_calib, smiles_calib)
    # Counted on the third block. The width came from the second, and neither
    # block chose a feature, so this percentage can come out below 1 - alpha.
    cov = conformal.score(X_cover, y_cover, smiles_cover)

    return {
        'conformal': conformal,
        'smiles_train': smiles_train,
        'feature_info': feature_info,
        'rep_names': rep_names,
        'n_train': len(train_idx),
        'n_calib': len(calib_idx),
        'n_cover': len(cover_idx),
        'n_folds_for_calib': n_folds,
        'model_name': model_name,
        'n_features': int(X_train.shape[1]),
        'coverage': cov,
    }


def predict_candidate(
    fitted: Dict[str, Any],
    candidate_smiles: str, species_for_task: List[str],
) -> Dict[str, Any]:
    """Per-species prediction + interval for one candidate.

    Returns a dict with parallel lists keyed by species (in the same order as
    `species_for_task`). The candidate SMILES is replicated len(species)
    times so the species-token augmentation flips on a single row each.

    The candidate is built through the same feature selection the fit block
    produced, so no label of any kind is read here.
    """
    k = len(species_for_task)
    if k == 0:
        raise ValueError("species_for_task is empty")
    candidate_list = [candidate_smiles] * k
    X_cand = _apply_selection(
        _reps_for(candidate_list, fitted['rep_names']),
        species_for_task, fitted['feature_info'])
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
        if isinstance(blueprint, dict):
            from kirby.find_best_hybrid import Blueprint as _Blueprint
            blueprint = _Blueprint.from_dict(blueprint)
        model_name = best_rec['model']
        budget = best_rec['budget']
        # The cached FittedResult is no longer read: its selected_indices were
        # chosen over every molecule of the task, including the ones that set
        # the interval width and the ones coverage is counted on. The recipe is
        # re-applied per block inside fit_calibrated_model instead.
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
            blueprint=blueprint,
            alpha=alpha, n_folds_for_calib=calib_folds,
        )
        # Only run the candidate against species that actually have data
        # for this task — `bundle['per_organism_counts']` is the authoritative
        # list. species_one_hot will raise on anything outside SPECIES_VOCAB.
        species_for_task = list(bundle['per_organism_counts'].keys())
        prediction = predict_candidate(
            fitted, candidate_smiles, species_for_task)
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
            'search_n_features': best_rec.get('n_features'),
            'n_features': fitted['n_features'],
            'reps_selected': list(blueprint.reps),
            'species_token_in_blueprint': bool(
                (fitted['feature_info'].get('augmentation_info') or {})
                and 'species_token' in fitted['feature_info']['augmentation_info']
            ),
            'n_samples': len(smiles),
            'n_scaffolds': int(n_scaffolds),
            'n_train': fitted['n_train'],
            'n_calib': fitted['n_calib'],
            'n_cover': fitted['n_cover'],
            'n_folds_for_calib': fitted['n_folds_for_calib'],
            # Counted on a block that neither set the interval width nor chose
            # a feature. Not comparable with any number a run before
            # 2026-09-04 wrote under 'calibration_coverage'.
            'coverage': fitted['coverage'],
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
                   'aug_strategy': args.aug_strategy}

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
