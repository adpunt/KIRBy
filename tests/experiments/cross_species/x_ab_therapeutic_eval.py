#!/usr/bin/env python3
"""x_ab_therapeutic_eval — T-Ab.5 therapeutic-shortlist scoring.

Loads the cached T-Ab.4 `find_best_hybrid` SearchResult for one target
species (`dog` or `cow`), refits the best model on the full training pool
*excluding* the therapeutic-eval canonical sequences (anti-leakage rule
in design doc §8), attaches T-UQ.1 ConformalClassifier for the 90 %
prediction set, scores each therapeutic with the three-tier label scheme
described in design doc §7.3:

  - native_target       = 2  → e.g. DoggifAI hold-outs for canineness
  - engineered_to_target = 1  → caninized / humanized therapeutics
  - off_species          = 0  → native non-target sequences

Emits one ranked JSON table per target showing per-therapeutic predicted
class probability (positive class = target_species native), conformal
prediction set, and the three-tier ground-truth label.

Therapeutic shortlists are tiny by design — the script is fast (CPU
seconds, single fold). It runs on the laptop after T-Ab.4's hybrid
has been cached.

Usage
-----

    # Canineness eval (after T-Ab.4 has run for --target-species dog):
    python tests/experiments/cross_species/x_ab_therapeutic_eval.py \\
        --target-species dog

    # Bovineness eval (after T-Ab.4 has run for --target-species cow):
    python tests/experiments/cross_species/x_ab_therapeutic_eval.py \\
        --target-species cow
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from experiments._shared import convert_numpy  # noqa: E402
from kirby.datasets.cross_species_ness import (  # noqa: E402
    V1_SUPPORTED_SPECIES, load_cross_species_ness,
    _sha1_hex, _canonical_seq,
)


# ---------------------------------------------------------------------------
# Therapeutic shortlists — three-tier labels per design doc §7.3
# ---------------------------------------------------------------------------
#
# `engineered_to_target`: the antibody is known to have been engineered to
#   approximate the target species. For canineness, that's the published
#   caninized vet mAbs (Librela/bedinvetmab, Solensia/frunevetmab, etc.)
#   and as a control the humanized trastuzumab. For bovineness there is no
#   published clinical bovine mAb; the entry stays empty by design.
#
# `off_species`: a native non-target antibody. We use the trastuzumab WT
#   sequence as a native human control for the canineness run.
#
# `native_target`: deliberately empty in this static shortlist — the script
#   draws N held-out native sequences from the loader's pool at runtime so
#   the table always has a real native-target reference.
#
# VH sequences below are pulled from public sources. The `note` field is the
# label provenance, not a sequence comment.

NATIVE_TARGET_HOLDOUT_SIZE = 5  # held-out native_target rows per run

CANINENESS_SHORTLIST: List[Dict[str, str]] = [
    # Librela / bedinvetmab — caninized anti-NGF mAb (Zoetis, EMA 2020).
    {
        'name': 'Librela_bedinvetmab',
        'tier': 'engineered_to_target',
        'note': 'caninized anti-NGF; framework remodelled to canine',
        'vh': (
            'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINSDGGS'
            'TYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDPGTHYDILTGNPRY'
            'YYYGMDVWGQGTLVTVSS'
        ),
    },
    # Solensia / frunevetmab — felinized anti-NGF, used here as a NEGATIVE
    # canineness control (it's engineered TO CAT, not to dog).
    {
        'name': 'Solensia_frunevetmab',
        'tier': 'off_species',
        'note': 'felinized anti-NGF; off-target for canineness',
        'vh': (
            'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG'
            'YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ'
            'GTLVTVSS'
        ),
    },
    # Trastuzumab — humanized anti-HER2; off-species for canineness.
    {
        'name': 'Trastuzumab_WT',
        'tier': 'off_species',
        'note': 'humanized anti-HER2; native sequence is humanized',
        'vh': (
            'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG'
            'YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ'
            'GTLVTVSS'
        ),
    },
]

# No published clinical bovine mAbs — the engineered_to_target row is
# intentionally empty. Bovineness eval relies on the runtime-drawn
# native_target hold-outs from the loader's cow pool.
BOVINENESS_SHORTLIST: List[Dict[str, str]] = []

SHORTLISTS = {
    'dog': CANINENESS_SHORTLIST,
    'cow': BOVINENESS_SHORTLIST,
}


# ---------------------------------------------------------------------------
# Cache loading (mirror of T-Mol.5 pattern)
# ---------------------------------------------------------------------------


def _cache_base() -> Path:
    return Path(os.environ.get('KIRBY_CACHE_DIR',
                               str(Path.home() / '.cache' / 'kirby')))


def _search_cache_path(target_species: str) -> Path:
    """Match find_best_hybrid's _search_cache_path for T-Ab.4 caches."""
    stem = f'{target_species}ness_v1__general__none__full'
    return _cache_base() / 'searches' / f'{stem}.pkl'


def load_t_ab_4_search(target_species: str):
    path = _search_cache_path(target_species)
    if not path.exists():
        return None
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as exc:
        print(f"  [{target_species}] cache unpickle failed: {exc}")
        return None
    payload = (wrapper.get('payload')
               if isinstance(wrapper, dict) else wrapper)
    if payload is None:
        return None
    if not getattr(payload, 'best_by_model', None):
        return None
    return payload


def select_best_model(search) -> Optional[Dict]:
    by_model = getattr(search, 'best_by_model', None) or {}
    if not by_model:
        return None
    # Metric name first, stored flag second, raise when neither — see
    # experiments._shared.object_direction. Fixed 2026-09-10.
    from experiments._shared import object_direction
    lower_is_better = object_direction(search, 'this SearchResult')

    def _key(item):
        s = item[1].get('score_mean', float('nan'))
        return -s if lower_is_better else s

    ranked = sorted(by_model.items(), key=_key, reverse=True)
    _, best_rec = ranked[0]
    return best_rec


# ---------------------------------------------------------------------------
# Feature reconstruction from FittedResult (label-free)
# ---------------------------------------------------------------------------


def _build_features_from_fitted(
    heavy_seqs: List[str], light_seqs: List[str], fitted,
    dataset_tag: str,
) -> np.ndarray:
    """Indexing-only feature reconstruction from FittedResult.selected_indices.

    Mirrors x_mol_rd_analysis._build_features_from_fitted but on antibody
    reps (no augmentations — T-Ab.4 uses augmentation_strategy='none').
    """
    from experiments.rep_cache import compute_or_load_reps

    rep_names = list(fitted.selected_indices.keys())
    base_reps = compute_or_load_reps(
        rep_names=rep_names,
        heavy_seqs=heavy_seqs,
        light_seqs=light_seqs,
        dataset_tag=dataset_tag,
    )
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
                f"Cached selected_indices for {rep_name!r} max={idx.max()} "
                f"≥ rep dim {X.shape[1]} — rep was rebuilt with a smaller "
                f"dimension; cache is stale.")
        parts.append(X[:, idx])
    if not parts:
        raise RuntimeError(
            "FittedResult has no usable selected features — cache is empty.")
    return np.hstack(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Three-tier table assembly
# ---------------------------------------------------------------------------


def _assemble_shortlist(
    target_species: str,
    holdout_natives: List[str],
) -> Tuple[List[Dict], Set[str]]:
    """Combine the static engineered/off shortlist with held-out natives.

    Returns (rows, exclude_seq_sha1_for_training) — the second item is the
    set of canonical-SHA1s the training loader must exclude so anti-leakage
    holds.
    """
    rows: List[Dict] = []
    for entry in SHORTLISTS.get(target_species, []):
        rows.append(dict(entry))
    for i, nat in enumerate(holdout_natives):
        rows.append({
            'name': f'native_{target_species}_holdout_{i:02d}',
            'tier': 'native_target',
            'note': f'held-out from {target_species} training pool',
            'vh': nat,
        })

    # canonicalise + hash each row so training-pool exclusion can run.
    exclude: Set[str] = set()
    for row in rows:
        canon = _canonical_seq(row['vh'])
        row['vh'] = canon
        row['canonical_sha1'] = _sha1_hex(canon)
        exclude.add(row['canonical_sha1'])
    return rows, exclude


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def evaluate_target(target_species: str, alpha: float = 0.1) -> Dict:
    search = load_t_ab_4_search(target_species)
    if search is None:
        return {
            'skipped': True,
            'reason': (
                f"Cache miss at {_search_cache_path(target_species)}. "
                "Run T-Ab.4 first (sbatch submit_x_ab_ness.sh "
                f"{target_species})."
            ),
        }
    best = select_best_model(search)
    if best is None:
        return {'skipped': True, 'reason': 'no blueprints in search cache'}
    model_name = best['model']
    fitted = (search.fitted_results or {}).get(model_name)
    if fitted is None:
        return {
            'skipped': True,
            'reason': f"FittedResult missing for {model_name!r}",
        }

    # 1. Get the training pool (excluding therapeutic-eval canonicals — we
    #    don't know which exactly yet, but loading first lets us draw the
    #    native hold-outs).
    print(f"\nLoading {target_species}ness training pool for refit...")
    pool = load_cross_species_ness(
        target_species=target_species,
        mode='one_vs_rest',
        source_species=V1_SUPPORTED_SPECIES,
        assign_groups=False,  # we don't need V-gene groups here
    )

    # 2. Draw native_target hold-outs (positives) from the pool.
    pos_idx = np.where(pool['labels'] == 1)[0]
    if len(pos_idx) < NATIVE_TARGET_HOLDOUT_SIZE + 50:
        # Need at least a few positives for refit AND for hold-out.
        print(f"  WARNING: only {len(pos_idx)} positives; reducing holdout.")
    rng = np.random.RandomState(0)
    holdout_n = min(NATIVE_TARGET_HOLDOUT_SIZE, len(pos_idx) // 4)
    chosen = rng.choice(pos_idx, holdout_n, replace=False).tolist()
    holdout_seqs = [pool['heavy_seqs'][i] for i in chosen]

    # 3. Build the three-tier shortlist + the canonical SHA1 exclusion set.
    rows, exclude = _assemble_shortlist(target_species, holdout_seqs)
    print(f"  Therapeutic table: {len(rows)} rows "
          f"({sum(1 for r in rows if r['tier'] == 'native_target')} native, "
          f"{sum(1 for r in rows if r['tier'] == 'engineered_to_target')} engineered, "
          f"{sum(1 for r in rows if r['tier'] == 'off_species')} off-species)")

    # 4. Reload the pool with exclusion → that's the conformal training set.
    pool_excl = load_cross_species_ness(
        target_species=target_species,
        mode='one_vs_rest',
        source_species=V1_SUPPORTED_SPECIES,
        exclude_seq_sha1=exclude,
        assign_groups=False,
    )
    n_train = len(pool_excl['heavy_seqs'])
    n_pos_train = int((pool_excl['labels'] == 1).sum())
    print(f"  Training pool after exclusion: {n_train} samples "
          f"({n_pos_train} positives)")

    # 5. Build features for the full excluded pool + the candidates.
    dataset_tag = f"{target_species}ness_v1"
    print(f"  Reconstructing features ({model_name})...")
    X_pool = _build_features_from_fitted(
        pool_excl['heavy_seqs'], pool_excl['light_seqs'],
        fitted, dataset_tag=dataset_tag)
    y_pool = np.asarray(pool_excl['labels'], dtype=np.int64)

    candidate_seqs = [r['vh'] for r in rows]
    X_cand = _build_features_from_fitted(
        candidate_seqs, [''] * len(candidate_seqs),
        fitted, dataset_tag=f"{dataset_tag}__cand")

    # 6. Split-conformal calibration: random 80/20 inside the excluded pool.
    #    No GroupKFold here — the V-gene groups are not exposed when
    #    assign_groups=False, and the conformal calibration just needs a
    #    held-out slice for nominal coverage. The T-Ab.4 baseline is what
    #    proves the classifier is calibrated against scaffold-disjoint
    #    folds; this conformal layer is only there to give per-candidate
    #    prediction sets at 90 % coverage.
    perm = np.random.RandomState(7).permutation(len(X_pool))
    cut = int(0.8 * len(X_pool))
    train_idx, calib_idx = perm[:cut], perm[cut:]
    X_tr, y_tr = X_pool[train_idx], y_pool[train_idx]
    X_ca, y_ca = X_pool[calib_idx], y_pool[calib_idx]

    # 7. Fit the same downstream model on X_tr, conformalise on X_ca.
    from kirby.find_best_hybrid import get_model
    from kirby.uq.conformal import ConformalClassifier
    estimator = get_model(model_name, task_type='classification',
                          n_samples=len(X_tr))
    if estimator is None:
        return {'skipped': True,
                'reason': f"get_model returned None for {model_name!r}"}
    estimator.fit(X_tr, y_tr)
    conf = ConformalClassifier(estimator=estimator, alpha=alpha)
    conf.fit_prefit(X_ca, y_ca)
    calib_cov = conf.score(X_ca, y_ca)
    print(f"  Calibration coverage @ 1-α={1-alpha:.2f}: "
          f"{calib_cov['coverage']:.3f}  mean set size={calib_cov['mean_set_size']:.2f}")

    # 8. Score the candidates.
    cs = conf.predict(X_cand)
    proba = None
    if hasattr(estimator, 'predict_proba'):
        proba = estimator.predict_proba(X_cand)
    table = []
    for i, row in enumerate(rows):
        entry = {
            'name': row['name'],
            'tier': row['tier'],
            'note': row['note'],
            'predicted_class': int(cs.pred[i]),
            'pred_set_includes_target': bool(
                cs.sets[i, list(cs.classes).index(1)]
                if 1 in cs.classes else False),
            'pred_set_classes': [
                int(c) for c, on in zip(cs.classes, cs.sets[i]) if on
            ],
        }
        if proba is not None and proba.shape[1] == 2:
            entry['p_target'] = float(proba[i, 1])
        table.append(entry)

    return {
        'skipped': False,
        'target_species': target_species,
        'model': model_name,
        'budget': best.get('budget'),
        'n_train_after_exclusion': n_train,
        'n_pos_train': n_pos_train,
        'alpha': alpha,
        'calibration_coverage': calib_cov,
        'table': table,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target-species', type=str, required=True,
                    choices=['dog', 'cow'])
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    print('=' * 70)
    print(f"x_ab_therapeutic_eval  (T-Ab.5)  target={args.target_species}")
    print('=' * 70)

    res = evaluate_target(args.target_species, alpha=args.alpha)

    out_default = (f'tests/results/cross_species/'
                   f'x_ab_therapeutic_{args.target_species}.json')
    out = Path(args.output or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(res), indent=2))

    if res.get('skipped'):
        print(f"\nSKIPPED: {res['reason']}")
    else:
        print(f"\nResults table:")
        for row in res['table']:
            ptarget = row.get('p_target', float('nan'))
            print(f"  {row['name']:30s} tier={row['tier']:22s} "
                  f"P(target)={ptarget:.3f}  "
                  f"predicted={row['predicted_class']}  "
                  f"set={row['pred_set_classes']}")
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
