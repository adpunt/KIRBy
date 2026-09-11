#!/usr/bin/env python3
"""PT-Rep Phase 1 — pre-trained representation experiment harness.

Runs the blueprint-based PT-Rep conditions for one (PT source, FT target) pair.
P2 (learned base rep) and P3 (joint hybrid) are deferred — they need new
training code; this harness covers the conditions that reuse existing machinery
(`find_best_hybrid` / `build_fold_hybrid` / `create_hybrid` / `Blueprint`).

B and G are the two ways to transfer a PT-run hybrid to FT, and the user asked
for both to be tested (undecided which is better):
  B  transfer PT's hybrid configuration (allocation / importance / evaluator /
     budget), then RE-RUN feature selection on the FT training data — FT
     produces its own selected feature indices.
  G  transfer PT's concrete selected feature indices and apply them to FT
     directly, with NO re-selection.

Conditions (PRETRAIN_REP_ROADMAP.md §5 Phase 1):
  A  ft_only       FT-only hybrid — FT's own search result. Baseline / current
                   production.
  B  p1_recipe     Transfer PT's configuration, re-run feature selection on FT
                   (build_fold_hybrid with the PT blueprint).
  F  p4_shortlist  PT's top-N configurations, each transferred B-style to FT;
                   the shortlist member with the best FT CV score wins.
  G  pt_frozen     Transfer PT's exact selected feature indices, applied to FT
                   with no re-selection (apply_feature_selection). Built once
                   on the full PT corpus.
  H  base_best     Best single base rep on FT (from the FT search baselines).
                   Rep-baseline reference.

  C, D, E (P2 autoencoder / P2 supervised-projection / P3 joint hybrid) are
  DEFERRED — see roadmap §4. They are intentionally not implemented here.

Compute note: `run()` calls `find_best_hybrid` (via get_precomputed_search) and
trains downstream models — it is a server workload, not a local one. The pure
helpers below (blueprint extraction, shared-rep resolution, leakage-input
resolution, aggregation) are import/unit-testable offline; see smoke_test.py.

Usage:
    python pretrain_rep_experiment.py \\
        --pt-source ginkgo_gdpa1 --pt-task ginkgo_kd \\
        --ft-dataset ginkgo_hic --conditions A,B,F,G,H --mode full
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments._shared import (
    load_data, get_config, save_experiment_results, convert_numpy,
    get_model_factories, _cv_split, build_fold_hybrid, get_domain_metric,
    get_precomputed_search, get_best_for_downstream, resolve_pruning_kwargs,
)
from experiments.pretrain_rep.pt_registry import get_pt_dataset, PTDataset
from experiments.pretrain_rep.pt_quality import apply_quality_pipeline
from experiments.pretrain_rep.pt_leakage import remove_pt_ft_leakage
from experiments.pretrain_rep.pt_reps import compute_pt_reps
from kirby.hybrid import create_hybrid, apply_feature_selection


# Fixed downstream model set for PT-Rep experiments (roadmap §5 Phase 1) —
# chosen on performance + runtime; deliberately not bound to the codebase's
# size-based auto-selection.
DOWNSTREAM_MODELS_PT = ['rf', 'lightgbm', 'svm', 'mlp']

ALL_CONDITIONS = ['A', 'B', 'F', 'G', 'H']
CONDITION_NAMES = {
    'A': 'ft_only', 'B': 'p1_recipe', 'F': 'p4_shortlist',
    'G': 'pt_frozen', 'H': 'base_best',
}


# =============================================================================
# Pure helpers (offline-testable)
# =============================================================================

def blueprint_key(bp) -> tuple:
    """Identity tuple for a Blueprint — used to dedup the P4 shortlist."""
    d = bp.to_dict() if hasattr(bp, 'to_dict') else dict(bp)
    return (
        tuple(d.get('reps', []) or []),
        d.get('allocation_method'),
        d.get('importance_method'),
        d.get('internal_evaluator'),
        d.get('budget'),
        d.get('step_size'),
    )


def extract_top_n_blueprints(search, n, lower_is_better=None) -> list:
    """Pull the top-N distinct blueprints out of a SearchResult.

    The candidate pool is `search.best_by_model_budget` — every (model, budget)
    winner. Sorted best-first by `score_mean`, deduped by `blueprint_key`,
    truncated to N. This is the P4 shortlist.
    """
    if lower_is_better is None:
        # Metric name first, stored flag second, raise when neither. Fixed 2026-09-10.
        from experiments._shared import object_direction
        lower_is_better = object_direction(search, 'this SearchResult')
    pool = []
    for _model, by_budget in (getattr(search, 'best_by_model_budget', None)
                              or {}).items():
        for _budget, cfg in (by_budget or {}).items():
            bp = cfg.get('blueprint')
            score = cfg.get('score_mean')
            if bp is None or score is None:
                continue
            if isinstance(score, float) and np.isnan(score):
                continue
            pool.append((score, bp))
    pool.sort(key=lambda t: t[0], reverse=not lower_is_better)
    seen, out = set(), []
    for _score, bp in pool:
        k = blueprint_key(bp)
        if k in seen:
            continue
        seen.add(k)
        out.append(bp)
        if len(out) >= n:
            break
    return out


def shared_rep_set(ft_reps, pt_reps, verbose=True) -> list:
    """Reps present in BOTH dicts with matching feature dimensionality."""
    shared = sorted(set(ft_reps) & set(pt_reps))
    out = []
    for rn in shared:
        if ft_reps[rn].shape[1] != pt_reps[rn].shape[1]:
            if verbose:
                print(f"  WARNING: {rn} dim mismatch "
                      f"(ft={ft_reps[rn].shape[1]}, pt={pt_reps[rn].shape[1]}) "
                      f"— excluded")
            continue
        out.append(rn)
    return out


def ft_leakage_inputs(ft_data) -> tuple:
    """Resolve (ft_inchikeys, ft_sequences) for PT/FT leakage removal.

    Exactly one is populated for a given domain; the other is None. Best-effort
    — returns (None, None) if neither can be resolved, in which case the caller
    skips the leakage step with a warning.
    """
    domain = ft_data.get('domain')
    if domain == 'antibody':
        seqs = ft_data.get('heavy_seqs')
        return None, (list(seqs) if seqs else None)
    if domain == 'molecular':
        smiles = ft_data.get('smiles')
        if not smiles:
            return None, None
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        except Exception:
            return None, None
        keys = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                inchi = MolToInchi(mol) if mol is not None else None
                keys.append(InchiToInchiKey(inchi) if inchi else None)
            except Exception:
                keys.append(None)
        keys = [k for k in keys if k]
        return (keys or None), None
    return None, None


def pt_search_metric(task_type: str) -> str:
    """Metric to optimise the PT-side search. Independent of the FT metric —
    the PT search just optimises its own corpus's task."""
    return 'r2' if task_type == 'regression' else 'f1_weighted'


def aggregate_fold_scores(scores) -> dict:
    """{mean, std, per_fold} over a list of fold scores (NaN-safe)."""
    arr = np.asarray(scores, dtype=float)
    return {
        'mean': float(np.nanmean(arr)) if len(arr) else float('nan'),
        'std': float(np.nanstd(arr)) if len(arr) else float('nan'),
        'per_fold': [float(s) for s in scores],
    }


# =============================================================================
# PT corpus preparation
# =============================================================================

def prepare_pt_corpus(pt_source, pt_task, ft_data, ft_rep_names,
                      max_samples=None, output_dir=None, verbose=True):
    """Load -> quality-filter -> leakage-strip -> rep-compute a PT corpus.

    Returns (pt_reps, pt_labels, pt_groups, pt_ds, reports).
    """
    domain = ft_data['domain']
    if verbose:
        print(f"\n  Preparing PT corpus: {pt_source} (task={pt_task})")

    pt_ds = get_pt_dataset(domain, pt_source, task=pt_task,
                           max_samples=max_samples)
    if verbose:
        print(f"    raw: n={len(pt_ds)}")

    q_path = (os.path.join(output_dir, f'pt_quality__{pt_source}.json')
              if output_dir else None)
    pt_ds, q_report = apply_quality_pipeline(pt_ds, report_path=q_path)
    if verbose:
        print(f"    after quality pipeline: n={len(pt_ds)} "
              f"({q_report.n_in - q_report.n_out} removed)")

    ft_keys, ft_seqs = ft_leakage_inputs(ft_data)
    l_report = None
    if ft_keys is None and ft_seqs is None:
        if verbose:
            print("    WARNING: could not resolve FT keys/sequences — "
                  "PT/FT leakage step SKIPPED.")
    else:
        pt_ds, l_report = remove_pt_ft_leakage(
            pt_ds, ft_keys=ft_keys, ft_sequences=ft_seqs)
        if verbose:
            print(f"    after leakage removal: n={len(pt_ds)} "
                  f"({l_report['n_removed']} removed)")

    pt_reps = compute_pt_reps(pt_ds, ft_rep_names, verbose=verbose)

    reports = {
        'quality': q_report.to_dict(),
        'leakage': l_report,
        'n_final': len(pt_ds),
    }
    return pt_reps, pt_ds.labels, pt_ds.groups, pt_ds, reports


# =============================================================================
# Main experiment
# =============================================================================

def run(ft_data, pt_source, pt_task, config, output_dir,
        conditions=None, top_n=5, pt_max_samples=None,
        pruning_mode='default_pruning'):
    """Run the PT-Rep conditions for one (PT source, FT target) pair.

    Args:
        ft_data: dict from load_data() for the fine-tuning target.
        pt_source: PT source name (pt_registry.list_pt_sources()).
        pt_task: PT sub-task selector (e.g. a GDPa1 property), or None.
        config: dict from get_config().
        output_dir: where to write results / quality reports.
        conditions: subset of ALL_CONDITIONS; default = all.
        top_n: P4 shortlist size.
        pt_max_samples: optional PT corpus size cap.
        pruning_mode: adaptive-pruning mode for the hybrid searches.
    """
    conditions = conditions or ALL_CONDITIONS
    bad = set(conditions) - set(ALL_CONDITIONS)
    if bad:
        raise ValueError(f"unknown conditions {sorted(bad)}; "
                         f"valid: {ALL_CONDITIONS}")

    ft_name = ft_data.get('dataset_name', 'ft')
    domain = ft_data['domain']
    task_type = ft_data['task_type']
    metric_name, metric_fn, lower_is_better = get_domain_metric(ft_data)
    cmp = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)

    print("\n" + "=" * 72)
    print("PT-Rep Phase 1 — pre-trained representation experiment")
    print(f"  FT target : {ft_name} ({len(ft_data['labels'])} samples, {domain})")
    print(f"  PT source : {pt_source} (task={pt_task})")
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Metric    : {metric_name} ({'lower' if lower_is_better else 'higher'} better)")
    print("=" * 72)

    ft_reps_all = ft_data['reps']
    ft_labels = ft_data['labels']
    ft_groups = ft_data['groups']
    n_folds = config['n_splits']
    model_names = config.get('models') or DOWNSTREAM_MODELS_PT

    # ── PT corpus prep (only if a PT-dependent condition is requested) ──
    needs_pt = bool({'B', 'F', 'G'} & set(conditions))
    pt_reps_all = pt_labels = pt_groups = None
    pt_reports = None
    if needs_pt:
        pt_reps_all, pt_labels, pt_groups, pt_ds, pt_reports = prepare_pt_corpus(
            pt_source, pt_task, ft_data, list(ft_reps_all.keys()),
            max_samples=pt_max_samples, output_dir=output_dir)
        if pt_labels is None:
            raise ValueError(
                f"PT source '{pt_source}' is unlabelled — conditions B/F/G "
                f"need PT labels. Use a labelled PT source or run only A/H.")

    # ── Shared rep set ──
    if needs_pt:
        shared = shared_rep_set(ft_reps_all, pt_reps_all)
    else:
        shared = sorted(ft_reps_all.keys())
    if len(shared) < 2:
        print("  ERROR: need >=2 shared reps for a meaningful hybrid")
        return {'error': 'insufficient_shared_reps', 'shared_reps': shared}
    ft = {k: ft_reps_all[k] for k in shared}
    pt = ({k: pt_reps_all[k] for k in shared} if needs_pt else None)
    print(f"\n  Shared reps ({len(shared)}): {', '.join(shared)}")

    # ── Hybrid searches (cached) ──
    ft_pkw = resolve_pruning_kwargs(pruning_mode, len(ft_labels),
                                    dataset_label=ft_name)
    needs_ft_search = bool({'A', 'H'} & set(conditions))
    ft_search = None
    if needs_ft_search:
        print(f"\n  FT hybrid search ({ft_name})")
        ft_search = get_precomputed_search(
            ft, ft_labels, ft_groups, aug_condition='none',
            task_type=task_type, domain=domain,
            dataset_name=ft_name, tier='general', **ft_pkw)

    pt_search = None
    if needs_pt:
        pt_metric = pt_search_metric(pt_ds.task_type)
        pt_pkw = resolve_pruning_kwargs(pruning_mode, len(pt_labels),
                                        dataset_label=f'pt_{pt_source}')
        print(f"\n  PT hybrid search ({pt_source})")
        pt_search = get_precomputed_search(
            pt, pt_labels, pt_groups, aug_condition='none',
            task_type=pt_ds.task_type, domain=domain,
            dataset_name=f'pt__{pt_source}', tier='general', **pt_pkw)

    # P4 shortlist (model-agnostic — extracted once)
    pt_shortlist = (extract_top_n_blueprints(pt_search, top_n)
                    if (pt_search is not None and 'F' in conditions) else [])
    if 'F' in conditions:
        print(f"\n  P4 shortlist: {len(pt_shortlist)} distinct PT blueprints")

    # ── Per-model evaluation ──
    print(f"\n  Per-model evaluation ({n_folds}-fold CV)")
    folds = list(_cv_split(ft_groups, n_folds,
                           next(iter(ft.values())), ft_labels))
    per_model = {}

    for model_name in model_names:
        mf = get_model_factories([model_name], task_type=task_type,
                                 n_samples=len(ft_labels))
        if model_name not in mf:
            print(f"    {model_name}: skipped (no factory)")
            continue
        mk = mf[model_name]

        ft_bp = (get_best_for_downstream(ft_search, model_name)
                 if ft_search is not None else None)
        pt_bp = (get_best_for_downstream(pt_search, model_name)
                 if pt_search is not None else None)
        ft_bp = ft_bp['blueprint'] if ft_bp else None
        pt_bp = pt_bp['blueprint'] if pt_bp else None

        # Best single base rep on FT (condition H) — from FT search baselines.
        best_rep = None
        if 'H' in conditions and ft_search is not None:
            best_bl = None
            for rn, mscores in getattr(ft_search, 'baselines', {}).items():
                if rn.startswith('_') or rn not in shared:
                    continue
                s = mscores.get(model_name)
                if s is not None and not np.isnan(s):
                    if best_bl is None or cmp(s, best_bl):
                        best_bl, best_rep = s, rn
            if best_rep is None:
                best_rep = shared[0]

        # Condition G: PT frozen indices — build once on FULL PT data.
        fi_pt = None
        if 'G' in conditions and pt_bp is not None:
            try:
                _, fi_pt = create_hybrid(
                    pt, pt_labels,
                    allocation_method=pt_bp.allocation_method,
                    budget=pt_bp.budget,
                    importance_method=pt_bp.importance_method,
                    internal_evaluator=pt_bp.internal_evaluator)
            except Exception as e:
                print(f"    {model_name}: G frozen-hybrid build failed: {e}")

        fold_scores = {CONDITION_NAMES[c]: [] for c in conditions}
        # P4: per-candidate per-fold scores; the shortlist winner is picked on
        # mean CV score (selection on CV, not per-fold — no leakage).
        f_candidate_scores = [[] for _ in pt_shortlist]

        for fi, (tr_idx, va_idx) in enumerate(folds):
            tr = {k: v[tr_idx] for k, v in ft.items()}
            va = {k: v[va_idx] for k, v in ft.items()}
            y_tr, y_va = ft_labels[tr_idx], ft_labels[va_idx]

            def _fit_score(X_tr, X_va):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_va = sc.transform(X_va)
                m = mk()
                m.fit(X_tr, y_tr)
                return float(metric_fn(y_va, m.predict(X_va)))

            # A — FT's own hybrid
            if 'A' in conditions:
                try:
                    Xt, Xv, _ = build_fold_hybrid(tr, va, y_tr, ft_bp,
                        groups=(ft_groups[tr_idx] if ft_groups is not None else None),
                        model=mk(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                    fold_scores['ft_only'].append(_fit_score(Xt, Xv))
                except Exception as e:
                    print(f"      fold {fi} A failed: {e}")
                    fold_scores['ft_only'].append(float('nan'))

            # B — P1 recipe transfer (PT recipe, FT re-selects features)
            if 'B' in conditions:
                try:
                    Xt, Xv, _ = build_fold_hybrid(tr, va, y_tr, pt_bp,
                        groups=(ft_groups[tr_idx] if ft_groups is not None else None),
                        model=mk(),
                        metric_fn=metric_fn,
                        lower_is_better=lower_is_better)
                    fold_scores['p1_recipe'].append(_fit_score(Xt, Xv))
                except Exception as e:
                    print(f"      fold {fi} B failed: {e}")
                    fold_scores['p1_recipe'].append(float('nan'))

            # F — P4 shortlist (each candidate recipe-transferred)
            if 'F' in conditions:
                for ci, bp in enumerate(pt_shortlist):
                    try:
                        Xt, Xv, _ = build_fold_hybrid(tr, va, y_tr, bp,
                            groups=(ft_groups[tr_idx] if ft_groups is not None else None),
                            model=mk(),
                            metric_fn=metric_fn,
                            lower_is_better=lower_is_better)
                        f_candidate_scores[ci].append(_fit_score(Xt, Xv))
                    except Exception:
                        f_candidate_scores[ci].append(float('nan'))

            # G — PT frozen indices (no FT re-selection)
            if 'G' in conditions:
                if fi_pt is None:
                    fold_scores['pt_frozen'].append(float('nan'))
                else:
                    try:
                        Xt = apply_feature_selection(tr, fi_pt)
                        Xv = apply_feature_selection(va, fi_pt)
                        fold_scores['pt_frozen'].append(_fit_score(Xt, Xv))
                    except Exception as e:
                        print(f"      fold {fi} G failed: {e}")
                        fold_scores['pt_frozen'].append(float('nan'))

            # H — best single base rep
            if 'H' in conditions:
                try:
                    fold_scores['base_best'].append(
                        _fit_score(tr[best_rep], va[best_rep]))
                except Exception as e:
                    print(f"      fold {fi} H failed: {e}")
                    fold_scores['base_best'].append(float('nan'))

        # Resolve P4 shortlist winner by mean CV score.
        f_winner = None
        if 'F' in conditions and f_candidate_scores:
            cand_means = [np.nanmean(s) if len(s) else float('nan')
                          for s in f_candidate_scores]
            valid = [(m, i) for i, m in enumerate(cand_means)
                     if not np.isnan(m)]
            if valid:
                best_m, best_i = (min if lower_is_better else max)(
                    valid, key=lambda t: t[0])
                fold_scores['p4_shortlist'] = f_candidate_scores[best_i]
                f_winner = {
                    'index': best_i,
                    'blueprint': blueprint_key(pt_shortlist[best_i]),
                }

        model_res = {c: aggregate_fold_scores(fold_scores[CONDITION_NAMES[c]])
                     for c in conditions}
        if 'H' in conditions:
            model_res['H']['base_rep'] = best_rep
        if f_winner is not None:
            model_res['F']['shortlist_winner'] = f_winner
            model_res['F']['n_candidates'] = len(pt_shortlist)

        # Deltas vs FT-only baseline (A), if A was run.
        if 'A' in conditions:
            base = model_res['A']['mean']
            for c in conditions:
                if c == 'A':
                    continue
                cm = model_res[c]['mean']
                delta = (base - cm) if lower_is_better else (cm - base)
                model_res[c]['delta_vs_ft_only'] = float(delta)

        per_model[model_name] = model_res
        line = '  '.join(
            f"{CONDITION_NAMES[c]}={model_res[c]['mean']:.4f}"
            for c in conditions)
        print(f"    {model_name}: {line}")

    if not per_model:
        return {'error': 'no_models_evaluated'}

    # ── Summary ──
    arrow = '↓' if lower_is_better else '↑'
    print(f"\n  {'='*72}")
    print(f"  SUMMARY — {ft_name} <- PT:{pt_source}   metric={metric_name} ({arrow})")
    print(f"  {'='*72}")
    hdr = f"  {'model':<10}" + ''.join(
        f"{CONDITION_NAMES[c]:>14}" for c in conditions)
    print(hdr)
    for mn, res in per_model.items():
        row = f"  {mn:<10}" + ''.join(
            f"{res[c]['mean']:>14.4f}" for c in conditions)
        print(row)

    if 'A' in conditions:
        for c in conditions:
            if c == 'A':
                continue
            deltas = [res[c].get('delta_vs_ft_only', float('nan'))
                      for res in per_model.values()]
            wins = sum(1 for d in deltas if d > 1e-4)
            print(f"  {CONDITION_NAMES[c]:>14}: mean delta vs ft_only "
                  f"{np.nanmean(deltas):+.4f}  (wins {wins}/{len(deltas)})")

    results = {
        'ft_dataset': ft_name,
        'pt_source': pt_source,
        'pt_task': pt_task,
        'domain': domain,
        'conditions': conditions,
        'metric': metric_name,
        'lower_is_better': lower_is_better,
        'n_folds': n_folds,
        'shared_reps': shared,
        'top_n': top_n,
        'pt_reports': pt_reports,
        'per_model': convert_numpy(per_model),
    }
    return results


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='PT-Rep Phase 1 experiment harness')
    ap.add_argument('--pt-source', required=True,
                    help='PT source name (pt_registry.list_pt_sources())')
    ap.add_argument('--pt-task', default=None,
                    help='PT sub-task (e.g. a GDPa1 property name)')
    ap.add_argument('--ft-dataset', required=True,
                    help='Fine-tuning target dataset name')
    ap.add_argument('--conditions', default=','.join(ALL_CONDITIONS),
                    help=f'Comma-separated subset of {ALL_CONDITIONS}')
    ap.add_argument('--top-n', type=int, default=5,
                    help='P4 shortlist size (default 5)')
    ap.add_argument('--pt-max-samples', type=int, default=None,
                    help='Optional PT corpus size cap')
    ap.add_argument('--mode', choices=['intermediary', 'full'], default='full',
                    help='intermediary (1 split) or full (5-fold CV)')
    ap.add_argument('--models', default=','.join(DOWNSTREAM_MODELS_PT),
                    help='Comma-separated downstream model list')
    ap.add_argument('--budgets', default=None,
                    help='Comma-separated budget override')
    ap.add_argument('--pruning-mode', default='default_pruning')
    ap.add_argument('--output', default='tests/results',
                    help='Output directory')
    args = ap.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(',')]
    config = get_config(args)

    ft_data = load_data(args.ft_dataset)

    results = run(
        ft_data, args.pt_source, args.pt_task, config, args.output,
        conditions=conditions, top_n=args.top_n,
        pt_max_samples=args.pt_max_samples, pruning_mode=args.pruning_mode)

    save_experiment_results(
        results, args.output, 'pretrain_rep_experiment',
        f'{args.ft_dataset}__pt_{args.pt_source}')
