#!/usr/bin/env python3
"""
Scaffold-diff reps vs the real rep pool — the honest pipeline comparison.

THE QUESTION (user, this is the one that matters): if the scaffold-diff mutation
reps go into the ACTUAL pipeline pool alongside the real representations (the
language-model embeddings esm2/prot_t5/ablang2/... and the cheap ones), on the
real models, with honest nested cross-validation —
  (1) do they BEAT the base reps as standalone representations, and
  (2) does the hybrid actually PICK their features into the winning blueprint?

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
    (NestedCVResult.baselines_by_model[model][rep]['mean']) — answers (1).
  - how often each scaffold rep appears in the winning hybrid blueprint across
    outer folds and models (blueprints_per_fold[...]['reps']) — answers (2).
  - delta_mean_of_folds — the honest hybrid-minus-best-single-rep headline.
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
    """Pull the two answers out of a NestedCVResult.to_dict()."""
    out = {'dataset': dataset_name,
           'delta_mean_of_folds': block.get('delta_mean_of_folds')}

    # (1) does each scaffold rep beat the base reps, per model?
    bbm = block.get('baselines_by_model', {}) or {}
    per_model = {}
    for model, reps in bbm.items():
        means = {r: v.get('mean') for r, v in reps.items()
                 if v.get('mean') is not None and np.isfinite(v.get('mean'))}
        if not means:
            continue
        base_means = {r: m for r, m in means.items() if r not in SCAFFOLD_REPS}
        best_base = max(base_means, key=base_means.get) if base_means else None
        per_model[model] = {
            'scaffold': {r: means.get(r) for r in SCAFFOLD_REPS},
            'best_base_rep': best_base,
            'best_base_mean': base_means.get(best_base) if best_base else None,
            'scaffold_rank': {
                r: (1 + sum(1 for m in means.values() if m > means[r]))
                for r in SCAFFOLD_REPS if r in means},
            'n_reps_ranked': len(means),
        }
    out['baseline_by_model'] = per_model

    # (2) does the hybrid pick scaffold features?
    picks = {r: 0 for r in SCAFFOLD_REPS}
    n_blueprints = 0
    bpf = block.get('blueprints_per_fold') or {}
    # blueprints_per_fold may be {model: [bp,...]} or [bp,...]; handle both
    iterables = bpf.values() if isinstance(bpf, dict) else [bpf]
    for lst in iterables:
        for bp in (lst or []):
            if not isinstance(bp, dict):
                continue
            n_blueprints += 1
            for r in bp.get('reps', []) or []:
                if r in picks:
                    picks[r] += 1
    out['hybrid_selection'] = {'n_blueprints': n_blueprints,
                               'scaffold_picked_count': picks}
    return out


def print_summary(summaries):
    print('\n' + '=' * 84)
    print('SCAFFOLD-DIFF REPS IN THE REAL PIPELINE POOL — honest nested-CV comparison')
    print('Dataset-native metric (Spearman / AUROC). Higher better.')
    print('=' * 84)
    for s in summaries:
        print(f'\n### {s["dataset"]}   '
              f'(hybrid vs best single rep, delta = {s.get("delta_mean_of_folds")})')
        for model, d in sorted(s.get('baseline_by_model', {}).items()):
            sc = d['scaffold']
            sc_str = ', '.join(
                f'{r}={sc[r]:.3f}(rank {d["scaffold_rank"].get(r,"?")}/{d["n_reps_ranked"]})'
                if sc.get(r) is not None else f'{r}=n/a' for r in SCAFFOLD_REPS)
            bb = d['best_base_mean']
            print(f'  {model:10s} best_base={d["best_base_rep"]}='
                  f'{bb:.3f}  |  {sc_str}' if bb is not None
                  else f'  {model:10s} (no base baselines)')
        hs = s.get('hybrid_selection', {})
        print(f'  hybrid picked scaffold features: {hs.get("scaffold_picked_count")} '
              f'out of {hs.get("n_blueprints")} blueprints')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--reps', default=None,
                    help='comma-sep base pool override (scaffold reps always added)')
    ap.add_argument('--mode', default='full', choices=['fast', 'full'])
    ap.add_argument('--models', default=None)
    ap.add_argument('--budgets', default=None)
    ap.add_argument('--max-folds', type=int, default=None, dest='max_folds')
    ap.add_argument('--smoke', action='store_true',
                    help='cheap fast reps only, for local plumbing checks')
    ap.add_argument('--output-dir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()

    from experiments._shared import load_data, run_nested_cv, get_config

    override = None
    if args.smoke:
        override = ['onehot', 'blosum', 'aac']
    elif args.reps:
        override = [r.strip() for r in args.reps.split(',')]

    config = get_config(args)
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
