#!/usr/bin/env python3
"""
Scaffold-diff reps vs the real rep pool — the honest pipeline comparison.

THE QUESTION (user, this is the one that matters): if the scaffold-diff mutation
reps go into the ACTUAL pipeline pool alongside the real representations (the
language-model embeddings esm2/prot_t5/ablang2/... and the cheap ones), on the
real models, with honest nested cross-validation — do they BEAT the base reps
as standalone representations?

A second question used to sit here: does the hybrid put scaffold-diff features
into the winning recipe? Answering it needs the number of features each
representation contributed, which nothing records, and recording it means
re-running all four datasets cold. Dropped by the owner 2026-09-11. The count
this file used to print was the number of representation NAMES in the recipe,
and the engine sets that list to every representation in the pool, so it never
moved. It is gone rather than left unread.

This is NOT the similarity/nearest-neighbour proxy. It trains the real models and
uses the honest evaluator (search on outer-train only; hybrid AND every single rep
scored on the untouched outer fold; nothing selection-leaked).

STANDALONE BY DESIGN. Nothing here is registered into MUTATION_REPS / ANTIBODY_REPS
or added to DATASET_REPS. The two scaffold-diff reps are computed in THIS file from
the exact sequences load_data returns (so rows cannot misalign) and injected into
the rep dict. The run uses a distinct cache tier ('scaffold_cmp'), so it writes to
its own rows and does NOT touch or invalidate the datasets' existing result DBs.

DATASETS: the four single-wildtype antibody DMS sets with mutation metadata.
phillips_h3 is excluded (user-cut 2026-07-18). ddg / graphinity excluded here
(multi-PDB; a separate adapter).

METRICS REPORTED (per dataset):
  - baseline mean of EACH rep on the honest outer folds, per model
    (NestedCVResult.baselines_by_model[model][rep]['mean']).
  - delta_mean_of_folds — the honest hybrid-minus-best-single-rep headline,
    printed with the number of outer folds that scored both arms and with the
    per-fold differences it averages.
The metric itself is the dataset's own (Spearman for regression, AUROC for the
binarised binding sets) — the same metric every other pipeline run uses.

SERVER RUN. The real pool contains GPU language-model reps, so this is a SLURM job
(tests/slurm_scripts/submit_scaffold_hybrid.sh). Locally, only --smoke (fast reps
only, no GPU) is sensible; do not run the full pool locally.

Usage:
    # server, full real pool, all 4 datasets
    python tests/experiments/opig_scaffold_hybrid.py --mode full

    # single dataset, custom pool
    python tests/experiments/opig_scaffold_hybrid.py --mode full \\
        --datasets flab_expression_koenig --reps esm2,prot_t5,ablang2

    # local plumbing smoke (cheap reps only; still calls the search — keep tiny)
    python tests/experiments/opig_scaffold_hybrid.py --mode fast --smoke \\
        --datasets flab_expression_koenig
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..'))          # tests/  -> experiments._shared
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'similarity_metrics'))

# --distance-metric, taken straight from the config module: this script does not
# put tests/ on sys.path, so experiments._shared is not importable here.
from kirby.similarity.distance_config import (          # noqa: E402
    DISTANCE_METRICS as _DISTANCE_METRICS,
    DEFAULT_DISTANCE_METRIC as _DEFAULT_DISTANCE_METRIC,
    set_distance_metric as _set_distance_metric,
)


def add_distance_metric_arg(parser):
    parser.add_argument('--distance-metric', type=str,
                        default=_DEFAULT_DISTANCE_METRIC,
                        choices=list(_DISTANCE_METRICS),
                        help=('Pairwise distance for every neighbour, cliff and '
                              'retrieval quantity this script computes. Default '
                              'manhattan. KIRBY_DISTANCE_METRIC in the '
                              'environment does the same thing.'))
    return parser


def apply_distance_metric_arg(args):
    return _set_distance_metric(
        getattr(args, 'distance_metric', None) or _DEFAULT_DISTANCE_METRIC)


DATASETS = ['flab_expression_koenig', 'flab_binding_koenig',
            'flab_expression_adams', 'flab_binding_adams']
SCAFFOLD_REPS = ('dphyschem', 'dphyschem_pos')
TIER = 'scaffold_cmp'          # distinct cache identity; does not touch existing DBs


def build_scaffold_reps(heavy_seqs, light_seqs):
    """Compute the two scaffold-diff reps from the SAME sequences load_data
    returned, guaranteeing row alignment with the base reps. Uses the validated
    research implementations in scaffold_reps (bit-identical to what the study ran).
    """
    import scaffold_reps as SR
    S, wt, muts = SR.encode_sequences(
        {'heavy_seqs': list(heavy_seqs), 'light_seqs': list(light_seqs)})
    return {'dphyschem': SR.rep_dphys(S, wt, muts),
            'dphyschem_pos': SR.rep_dphys_positional(S, wt, muts)}, S.shape[0]


def base_pool_for(dataset_name, override=None):
    if override:
        return [r for r in override if r not in SCAFFOLD_REPS]
    from experiments.complexity.complexity_helpers import DATASET_REPS
    return list(DATASET_REPS[dataset_name]['general'])


def summarise_one(dataset_name, block):
    """Pull the per-representation answer and the honest delta out of a
    NestedCVResult.to_dict()."""
    # to_dict() nests the paired delta under by_model[model], with no
    # top-level copy. Reading it at the top level returned None for every
    # dataset, so the headline number printed empty.
    by_model = block.get('by_model', {}) or {}
    # Spearman and AUROC both count up, but the direction is read off the
    # result rather than assumed, so a dataset scored in RMSE cannot silently
    # invert the best-base pick and the rank position below.
    lower_is_better = bool(block.get('lower_is_better', False))
    out = {'dataset': dataset_name,
           'metric_name': block.get('metric_name'),
           'lower_is_better': lower_is_better,
           'delta_mean_of_folds': {
               m: e.get('delta_mean_of_folds') for m, e in by_model.items()},
           'n_paired_folds': {
               m: e.get('n_paired_folds') for m, e in by_model.items()},
           # The per-outer-fold differences the mean above was taken over. A
           # mean over one outer fold and a mean over five print the same
           # without them.
           'per_fold_delta': {
               m: e.get('per_fold_delta') for m, e in by_model.items()}}

    # does each scaffold rep beat the base reps, per model?
    bbm = block.get('baselines_by_model', {}) or {}
    per_model = {}
    for model, reps in bbm.items():
        means = {r: v.get('mean') for r, v in reps.items()
                 if v.get('mean') is not None and np.isfinite(v.get('mean'))}
        if not means:
            continue
        base_means = {r: m for r, m in means.items() if r not in SCAFFOLD_REPS}
        if base_means:
            best_base = (min(base_means, key=base_means.get) if lower_is_better
                         else max(base_means, key=base_means.get))
        else:
            best_base = None
        per_model[model] = {
            'scaffold': {r: means.get(r) for r in SCAFFOLD_REPS},
            'best_base_rep': best_base,
            'best_base_mean': base_means.get(best_base) if best_base else None,
            'scaffold_rank': {
                r: (1 + sum(1 for v in means.values()
                            if (v < means[r] if lower_is_better
                                else v > means[r])))
                for r in SCAFFOLD_REPS if r in means},
            'n_reps_ranked': len(means),
        }
    out['baseline_by_model'] = per_model
    return out


def print_summary(summaries):
    print('\n' + '=' * 84)
    print('SCAFFOLD-DIFF REPS IN THE REAL PIPELINE POOL — honest nested-CV comparison')
    # Each dataset is scored in its own metric and carries its own direction,
    # printed beside the metric name below. A fixed "higher better" here would
    # contradict any dataset scored in RMSE.
    print('Dataset-native metric; direction per dataset.')
    print('=' * 84)
    for s in summaries:
        direction = 'lower better' if s.get('lower_is_better') else 'higher better'
        print(f'\n### {s["dataset"]}   '
              f'(metric: {s.get("metric_name")}, {direction})')
        for model, d in sorted(s.get('baseline_by_model', {}).items()):
            sc = d['scaffold']
            sc_str = ', '.join(
                f'{r}={sc[r]:.3f}(rank {d["scaffold_rank"].get(r,"?")}/{d["n_reps_ranked"]})'
                if sc.get(r) is not None else f'{r}=n/a' for r in SCAFFOLD_REPS)
            bb = d['best_base_mean']
            print(f'  {model:10s} best_base={d["best_base_rep"]}='
                  f'{bb:.3f}  |  {sc_str}' if bb is not None
                  else f'  {model:10s} (no base baselines)')
        # The delta never prints alone: one outer fold and five outer folds
        # give the same-looking number otherwise.
        deltas = s.get('delta_mean_of_folds') or {}
        if deltas:
            print('  hybrid minus best single representation, per model — '
                  'mean, outer folds that scored both arms, then those folds:')
            for model in sorted(deltas):
                dv = deltas[model]
                n = (s.get('n_paired_folds') or {}).get(model)
                pf = (s.get('per_fold_delta') or {}).get(model) or []
                d_str = ('—' if dv is None or not np.isfinite(dv)
                         else f'{dv:+.4f}')
                pf_str = ', '.join(
                    '—' if v is None or not np.isfinite(v) else f'{v:+.3f}'
                    for v in pf) or 'none'
                print(f'    {model:10s} {d_str:>9s}  '
                      f'{n} paired outer folds  |  per fold: {pf_str}')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--reps', default=None,
                    help='comma-sep base pool override (scaffold reps always added)')
    ap.add_argument('--mode', default='full', choices=['fast', 'full'])
    # --models is deliberately absent. run_nested_cv pins models=None so the
    # search's cache identity matches what the SLURM precompute wrote; a
    # per-run model list is part of that identity, so overriding it here would
    # miss every warmed search and re-run it on the spot. It was parsed and
    # never passed on until 2026-09-11.
    ap.add_argument('--budgets', default=None,
                    help='comma-separated feature budgets for the search '
                         '(default: the pipeline-wide FULL_BUDGETS)')
    ap.add_argument('--max-folds', type=int, default=None, dest='max_folds',
                    help='number of outer held-out folds (default: 5, from config)')
    ap.add_argument('--smoke', action='store_true',
                    help='cheap fast reps only, for local plumbing checks')
    ap.add_argument('--output-dir', default=os.path.join(HERE, 'results'))
    ap = add_distance_metric_arg(ap)
    args = ap.parse_args()

    apply_distance_metric_arg(args)
    from experiments._shared import load_data, run_nested_cv, get_config

    override = None
    if args.smoke:
        override = ['onehot', 'blosum', 'aac']
    elif args.reps:
        override = [r.strip() for r in args.reps.split(',')]

    config = get_config(args)
    budgets = ([int(b) for b in args.budgets.split(',') if b.strip()]
               if args.budgets else None)
    os.makedirs(args.output_dir, exist_ok=True)
    summaries, t0 = [], time.time()

    for ds in args.datasets:
        pool = base_pool_for(ds, override)
        print(f'\n=== {ds} | base pool = {len(pool)} reps + '
              f'{len(SCAFFOLD_REPS)} scaffold ({time.time()-t0:.0f}s) ===', flush=True)
        data = load_data(ds, rep_preset='full', rep_names=pool)
        scaffold, n_scaf = build_scaffold_reps(data['heavy_seqs'], data['light_seqs'])
        n_rows = len(data['labels'])
        assert n_scaf == n_rows, (
            f'row misalignment: scaffold {n_scaf} vs labels {n_rows}')
        data['reps'].update(scaffold)
        built = [r for r in (pool + list(SCAFFOLD_REPS)) if r in data['reps']]
        print(f'  reps built ({len(built)}): '
              f'scaffold present = {[r for r in SCAFFOLD_REPS if r in data["reps"]]}',
              flush=True)

        result = run_nested_cv(
            reps=data['reps'], labels=np.asarray(data['labels']),
            groups=data['groups'], config=config, dataset_name=ds,
            task_type=data['task_type'], domain='antibody', tier=TIER,
            internal_evaluators=config['internal_evaluators'],
            budgets=budgets, n_outer_folds=args.max_folds,
            verbose=True)
        block = result.to_dict()

        out_path = os.path.join(args.output_dir, f'opig_scaffold_hybrid_{ds}.json')
        with open(out_path, 'w') as f:
            json.dump(block, f, indent=1, default=lambda o: (
                o.item() if isinstance(o, (np.floating, np.integer))
                else o.tolist() if isinstance(o, np.ndarray) else str(o)))
        summaries.append(summarise_one(ds, block))
        print(f'  wrote {out_path} ({time.time()-t0:.0f}s)', flush=True)

    print_summary(summaries)
    spath = os.path.join(args.output_dir, 'opig_scaffold_hybrid_summary.json')
    with open(spath, 'w') as f:
        json.dump(summaries, f, indent=1, default=str)
    print(f'\nwrote {spath}')


if __name__ == '__main__':
    main()
