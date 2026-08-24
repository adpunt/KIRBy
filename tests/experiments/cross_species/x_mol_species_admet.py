#!/usr/bin/env python3
"""x_mol_species_admet — Species-stratified ADMET via species-as-token (T-Mol.4).

Per `CROSS_SPECIES_ROADMAP.md` Phase X-Mol / `docs/cross_species_admet_design.md`:
joins the per-organism slices of the cross-species ChEMBL loaders into one
table, attaches a 5-D one-hot `{human, dog, cat, cow, horse}` species token
as an augmentation, and runs `find_best_hybrid` on it. The species token is
*always included*; the hybrid search decides whether to use it.

Tasks driven by ``ADMET_TASKS`` below — each is a (loader, organisms)
pair. T-Mol.2d resolved feasible organism sets:

- ``cox2``: human + dog + cow (PTGS2, IC50)
- ``drd2``: human + cow (DRD2, Ki)
- ``canine_cyp_*``: dog (+ human comparison when JSON / CSV populated)
- ``canine_pgp``: dog (+ human)

Splits: scaffold GroupKFold using Murcko scaffolds computed on canonical
SMILES. Because the same canonical SMILES yields the same scaffold across
organisms, a compound assayed in multiple species **always** falls in the
same fold (design doc §8 anti-leakage rule applied for free). 5-fold default,
3-fold fallback for n < 150.

Output: one JSON at ``tests/results/cross_species/x_mol_species_admet.json``
with per-task baseline + hybrid metrics, sample sizes per organism, and the
selected augmentation status.

Usage
-----

    # Dry-run (1 fold, smallest budget, no augmentation search) — local sanity
    python tests/experiments/cross_species/x_mol_species_admet.py --dry-run

    # Full run — submit on ARC
    python tests/experiments/cross_species/x_mol_species_admet.py

    # Single task, custom reps
    python tests/experiments/cross_species/x_mol_species_admet.py \\
        --tasks cox2 --reps ecfp4,mordred --n-folds 3
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings('ignore')

from find_best_hybrid_molecular import (
    get_representations, assign_scaffold_groups,
)
from experiments._shared import convert_numpy, FULL_BUDGETS
from experiments.cross_species.foundation_hybrid_vet import (
    REP_PRESETS, DEFAULT_N_FOLDS,
    _resolve_folds, build_baseline_models, evaluate_one_rep_model,
)


# Species-token vocabulary — must match the registry order used by the
# augmentation. Adding to the right end is safe (existing organism indices
# unchanged); reordering breaks cache keys.
SPECIES_VOCAB: List[str] = ['human', 'dog', 'cat', 'cow', 'horse']


def species_one_hot(organisms: List[str]) -> np.ndarray:
    """Build the 5-D species-token augmentation matrix per row."""
    n = len(organisms)
    out = np.zeros((n, len(SPECIES_VOCAB)), dtype=np.float32)
    for i, org in enumerate(organisms):
        try:
            out[i, SPECIES_VOCAB.index(org)] = 1.0
        except ValueError as exc:
            raise ValueError(
                f"organism '{org}' not in SPECIES_VOCAB={SPECIES_VOCAB}. "
                f"Add it to the vocab (right end only) and bump cache."
            ) from exc
    return out


def _load_cox2_slice(organism: str) -> Tuple[List[str], np.ndarray]:
    from kirby.datasets.cox2 import load_cox2_chembl
    d = load_cox2_chembl(organism=organism)
    return d['smiles'], np.asarray(d['labels'], dtype=np.float64)


def _load_drd2_slice(organism: str) -> Tuple[List[str], np.ndarray]:
    from kirby.datasets.drd2 import load_drd2_chembl
    d = load_drd2_chembl(organism=organism)
    return d['smiles'], np.asarray(d['labels'], dtype=np.float64)


def _load_canine_cyp_slice(isoform: str, organism: str
                           ) -> Tuple[List[str], np.ndarray]:
    from kirby.datasets.canine_cyp import load_canine_cyp
    d = load_canine_cyp(isoform, organism=organism)
    return list(d['smiles']), np.asarray(d['labels'], dtype=np.float64)


def _load_canine_pgp_slice(organism: str) -> Tuple[List[str], np.ndarray]:
    from kirby.datasets.canine_pgp import load_canine_pgp
    d = load_canine_pgp(organism=organism)
    return list(d['smiles']), np.asarray(d['labels'], dtype=np.float64)


# Per-task loader + supported organism slice. The loader callable takes a
# single string `organism` arg; tasks with extra args (CYP isoform) bind
# them via a closure below.
TaskLoader = Tuple[str, List[str], "callable"]   # (task_name, organisms, fn)


def admet_tasks() -> List[TaskLoader]:
    """Build the canonical task list. Filtering / overrides happen in main()."""
    tasks: List[TaskLoader] = []
    tasks.append(('cox2', ['human', 'dog', 'cow'], _load_cox2_slice))
    tasks.append(('drd2', ['human', 'cow'], _load_drd2_slice))
    # Canine CYPs: dog default. Most isoforms will hit the documented-skip
    # path until CSVs / resolved ChEMBL IDs land; the experiment loop logs
    # and continues on empty.
    for iso in ('1A2', '2B11', '2C21', '2D15', '3A12'):
        tasks.append((
            f'canine_cyp_{iso.lower()}',
            ['dog', 'human'],
            lambda organism, _iso=iso: _load_canine_cyp_slice(_iso, organism),
        ))
    tasks.append(('canine_pgp', ['dog', 'human'], _load_canine_pgp_slice))
    return tasks


def _assemble_task(
    task_name: str, organisms: List[str], loader,
    min_rows_per_organism: int = 20,
) -> Optional[Dict]:
    """Load each organism slice, drop empties / sparse, concatenate.

    Returns None when fewer than 2 organisms have enough data (cross-species
    is moot for a single organism — the existing per-organism baseline
    covers that case). Returns a dict with ``smiles``, ``labels``,
    ``organisms`` (per-row), ``per_organism_counts``.
    """
    per_org: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for organism in organisms:
        try:
            smi, y = loader(organism)
        except Exception as exc:
            print(f"  [{task_name}/{organism}] loader error: {exc}")
            continue
        if len(smi) < min_rows_per_organism:
            print(f"  [{task_name}/{organism}] sparse "
                  f"({len(smi)} < {min_rows_per_organism}); skipped")
            continue
        per_org[organism] = (smi, y)

    if len(per_org) < 2:
        print(f"  [{task_name}] only {len(per_org)} organism(s) above the "
              f"{min_rows_per_organism}-row threshold "
              f"({list(per_org)}); cross-species training is moot — skip task.")
        return None

    all_smiles: List[str] = []
    all_labels: List[float] = []
    all_orgs: List[str] = []
    counts: Dict[str, int] = {}
    for organism, (smi, y) in per_org.items():
        if len(smi) != len(y):
            raise ValueError(
                f"[{task_name}/{organism}] smiles/labels length mismatch "
                f"({len(smi)} vs {len(y)})"
            )
        all_smiles.extend(smi)
        all_labels.extend(y.tolist())
        all_orgs.extend([organism] * len(smi))
        counts[organism] = len(smi)

    return {
        'smiles': all_smiles,
        'labels': np.asarray(all_labels, dtype=np.float64),
        'organisms': all_orgs,
        'per_organism_counts': counts,
    }


def evaluate_task(
    task_name: str, organisms: List[str], loader,
    rep_names: List[str],
    n_folds_requested: int,
    dry_run: bool = False,
) -> Dict:
    """Run baselines + find_best_hybrid for one cross-species ADMET task."""
    print(f"\n{'=' * 70}")
    print(f"Task: {task_name}  organisms={organisms}")
    print('=' * 70)

    bundle = _assemble_task(task_name, organisms, loader)
    if bundle is None:
        return {'skipped': True, 'reason': 'insufficient organism coverage'}

    smiles = bundle['smiles']
    y = bundle['labels']
    organisms_per_row = bundle['organisms']
    per_org_counts = bundle['per_organism_counts']
    n = len(smiles)
    print(f"  total n = {n}; per-organism counts = {per_org_counts}")
    print(f"  label range pActivity = [{y.min():.2f}, {y.max():.2f}]")

    groups, n_scaffolds = assign_scaffold_groups(smiles)
    n_folds = _resolve_folds(n, n_folds_requested)
    if dry_run:
        n_folds = 2  # smallest valid GroupKFold
        print(f"  DRY-RUN: forcing n_folds={n_folds}")
    print(f"  {n_scaffolds} unique scaffolds (Murcko), {n_folds}-fold")

    reps = get_representations(smiles, ','.join(rep_names))
    species_aug = species_one_hot(organisms_per_row)
    print(f"  species-token augmentation: shape={species_aug.shape}, "
          f"present_species={sorted(set(organisms_per_row))}")

    task_result: Dict = {
        'n_samples': n,
        'n_scaffolds': int(n_scaffolds),
        'n_folds': n_folds,
        'per_organism_counts': per_org_counts,
        'task_type': 'regression',
        'baselines_per_rep': {},
        'hybrid': {},
    }

    # --- Per-rep baselines (no species token) — anchor against the species-
    # augmented hybrid below. We do NOT add the species token here; the
    # comparison is "vanilla rep + downstream model" vs "find_best_hybrid +
    # species-token augmentation".
    baselines = build_baseline_models(n_samples=n, task_type='regression')
    for rep_name in rep_names:
        if rep_name not in reps:
            print(f"  {rep_name}: SKIPPED (rep failed to build)")
            continue
        X = np.asarray(reps[rep_name], dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != n:
            print(f"  {rep_name}: SKIPPED (shape mismatch {X.shape})")
            continue
        rep_block = {'dim': int(X.shape[1]), 'conditions': {}}
        print(f"  --- {rep_name} (dim={X.shape[1]}) ---")
        for cond_name, model in baselines.items():
            t0 = time.time()
            res = evaluate_one_rep_model(X, y, groups, model, n_folds)
            res['time_s'] = time.time() - t0
            rmse = res['mean'].get('rmse', float('nan'))
            rho = res['mean'].get('spearman', float('nan'))
            print(f"    {cond_name:10s}  RMSE={rmse:.4f}  ρ={rho:.4f}  "
                  f"({res['time_s']:.1f}s, folds={res['n_folds_completed']})")
            rep_block['conditions'][cond_name] = res
        task_result['baselines_per_rep'][rep_name] = rep_block

    # --- find_best_hybrid with species-token augmentation ---
    print(f"\n  --- find_best_hybrid (+ species-token augmentation) ---")
    budgets = FULL_BUDGETS if not dry_run else [FULL_BUDGETS[0]]
    try:
        from kirby.find_best_hybrid import find_best_hybrid
        t0 = time.time()
        search = find_best_hybrid(
            base_reps=reps,
            labels=y,
            groups=groups,
            budgets=budgets,
            return_fitted=True,
            verbose=False,
            cache=True,
            dataset_name=f"x_mol_species_admet_{task_name}",
            tier='general',
            metric='rmse',
            task_type='regression',
            augmentations={'species_token': species_aug},
            augmentation_strategy='uncapped',
            adaptive_pruning=0.25 if not dry_run else False,
        )
        elapsed = time.time() - t0
        per_model = {}
        for model_name, fitted in (search.fitted_results or {}).items():
            best = getattr(fitted, 'best_metric', None) or \
                   getattr(fitted, 'best_score', None)
            selected = list(getattr(fitted, 'selected_indices', {}).keys())
            per_model[model_name] = {
                'best_rmse': float(best) if best is not None else None,
                'selected_reps': selected,
                'species_token_selected': 'species_token' in selected,
            }
        task_result['hybrid'] = {
            'time_s': elapsed,
            'budgets': list(budgets),
            'per_model': per_model,
        }
        for m, r in per_model.items():
            best = r.get('best_rmse')
            tok = 'YES' if r.get('species_token_selected') else 'no '
            line = f"    {m:10s}  best RMSE={best:.4f}" if best is not None \
                   else f"    {m:10s}  (no result)"
            print(f"{line}   species-token={tok}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        task_result['hybrid'] = {'error': str(exc)[:300]}

    return task_result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', type=str, default=None,
                    help='Comma-separated task names (default: all).')
    ap.add_argument('--reps', type=str, default=None,
                    help='Comma-separated rep names. Overrides --preset.')
    ap.add_argument('--preset', type=str, default='full',
                    choices=sorted(REP_PRESETS),
                    help='Rep preset (foundation_hybrid_vet presets reused).')
    ap.add_argument('--n-folds', type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument('--dry-run', action='store_true',
                    help='Smallest budget, fewest folds — local sanity check.')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    all_tasks = admet_tasks()
    if args.tasks:
        wanted = set(args.tasks.split(','))
        all_tasks = [t for t in all_tasks if t[0] in wanted]
        if not all_tasks:
            raise SystemExit(f"No matching tasks in {args.tasks}")

    if args.reps:
        rep_names = args.reps.split(',')
        preset_label = 'custom'
    else:
        rep_names = REP_PRESETS[args.preset]
        preset_label = args.preset

    print("=" * 70)
    print(f"x_mol_species_admet  (T-Mol.4)  preset={preset_label} "
          f"reps={len(rep_names)} dry_run={args.dry_run}")
    print(f"  tasks: {[t[0] for t in all_tasks]}")
    print("=" * 70)

    results: Dict[str, Dict] = {}
    for task_name, organisms, loader in all_tasks:
        try:
            results[task_name] = evaluate_task(
                task_name, organisms, loader, rep_names,
                n_folds_requested=args.n_folds, dry_run=args.dry_run,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[task_name] = {'error': str(exc)[:300]}

    suffix = '_dry' if args.dry_run else ''
    out_default = (f'tests/results/cross_species/'
                   f'x_mol_species_admet_{preset_label}{suffix}.json')
    out = Path(args.output or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(convert_numpy(results), indent=2))
    print(f"\nResults written: {out}")


if __name__ == '__main__':
    main()
