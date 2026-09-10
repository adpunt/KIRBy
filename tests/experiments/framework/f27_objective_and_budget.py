#!/usr/bin/env python3
"""F27: which objective builds the best hybrid, out of which representations,
at which budget?

Paper: Framework. This decides how KIRBy builds a hybrid, so it is a framework
experiment, not a noise or similarity one.

WHY THIS EXISTS -- READ THIS BEFORE ADDING ANYTHING
    f19 and f21 were asked the same two questions and answered neither. Both
    scored feature subsets written by hand inside the experiment, over five
    fingerprints. Neither calls create_hybrid, neither uses an allocation
    method, neither uses a sim_ recipe. A framework experiment that never
    builds a hybrid cannot say anything about how the framework builds one.
    They stay on disk as the example of what not to do. This one runs the real
    search through nested_cv_evaluate and reports per dataset x model x budget.

THE TWO QUESTIONS
    1. Which objective should the search maximise? Picking by prediction is the
       default. Picking by geometry is what the similarity study needs. Until
       2026-08-27 the two could not be compared: a predictive run computed no
       geometry number and a geometry run fitted no model. Every winner now
       records all three numbers, so an objective that only wins on the number
       it selected by is visible as such.
    2. Is there one best budget? Or does the best budget move with the dataset
       and the model, in which case budget has to stay a searched axis.

THE FIVE OBJECTIVES
    predictive   fit the model, score its held-out predictions
    similarity   each held-out molecule's property predicted from its five
                 nearest neighbours, Spearman'd against truth, scaffold-grouped
    struct_fid   Spearman between pairwise distance and MCS distance -- whether
                 two molecules the space calls close really do share atoms
    blend 0.75 similarity / 0.25 struct_fid
    blend 0.50 similarity / 0.50 struct_fid
                 the weighted mean of those two on each fold. Both are Spearman
                 correlations on the same rows, both higher-is-better, both
                 bounded in [-1, 1], so the mean is on the same scale as either.

    similarity_sac was CUT by the owner on 2026-08-26 and raises. It is not here.

THE THREE REPRESENTATION POOLS
    general      the dataset's signed-off pool (10 per molecular dataset)
    recipes      the 11 structure-tuned sim_ recipes alone
    both         all 21

    Three pools rather than two: with 'both' alone, a difference could come
    from the recipes or from having more representations to choose between.
    'recipes' on its own separates them.

WHAT IS REPORTED
    delta_mean_of_folds from nested_cv_evaluate -- the hybrid minus the best
    single representation, paired per outer fold, both arms having paid the
    identical held-out tax. Never search_mean, never .all_results.
    Plus, for EVERY winner whichever objective picked it, the same paired delta
    on the other two numbers (side_predictive_*, side_readacross_*,
    struct_fidelity_*), and the honest score per budget per model.

    MCS pairs are built once per dataset by kirby.similarity.mcs.get_mcs_pairs,
    keyed by a content hash of that dataset's own SMILES in the framework's row
    order. 8000 pairs measured at 0.28 s/pair on 2026-08-27, so about 37 minutes
    the first time a dataset is run and free after that.

SKIPS ARE LOGGED. Any (objective, pool) that fails is written into the output
under 'skipped' with its error. A missing cell never reads as a cell that ran
and produced nothing. MODEL_MAX_BUDGET is EMPTY since 2026-09-10, so every
model now runs at every budget asked for. 'budget_caps' stays in the output and
is empty; it fills in again only if a caller puts a cap back by name.

WALL TIME. 5 objectives x 3 pools = 15 nested_cv_evaluate runs per dataset.
f26's structural estimate for one is 3.5-11 h, so 52-165 h per dataset. That is
a SERVER job, one array task per (dataset, pool). Structural estimate from
configuration counts, NOT a measured time -- run --smoke first and set the wall
from that.

USAGE
    # one array task on the server
    python tests/experiments/framework/f27_objective_and_budget.py \\
        --dataset herg_ki --pools both

    # local plumbing check -- three cheap representations, one budget, one fold
    python tests/experiments/framework/f27_objective_and_budget.py \\
        --dataset herg_ki --smoke
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))     # tests/
sys.path.insert(0, os.path.join(HERE, '..'))           # tests/experiments/

TIER = 'f27_objbud'          # own cache identity; touches no existing result DB

# (name used in output and on the command line, objective passed to the search)
OBJECTIVES = [
    ('predictive', 'predictive'),
    ('similarity', 'similarity'),
    ('struct_fid', 'struct_fid'),
    ('blend_75_25', {'similarity': 0.75, 'struct_fid': 0.25}),
    ('blend_50_50', {'similarity': 0.50, 'struct_fid': 0.50}),
]

# (name, include_general, recipes_only) -> resolved by pool_rep_names below
POOLS = ['general', 'recipes', 'both']

# Every objective is read against this one. It is the current default, so a
# column of numbers inside the fold spread means changing the objective bought
# nothing.
REFERENCE_OBJECTIVE = 'predictive'

# The three numbers every winner carries, and the prefix each is stored under
# in NestedCVResult.by_model.
SIDE_KEYS = ['side_predictive', 'side_readacross', 'struct_fidelity']


def pool_rep_names(dataset, pool):
    """Representation names for one pool on one dataset.

    Raises for a molecular dataset with no sim_ recipes registered rather than
    silently handing back the general pool twice under two pool names.
    """
    from experiments.complexity.complexity_helpers import (
        DATASET_REPS, similarity_rep_names)
    entry = DATASET_REPS.get(dataset) or {}
    general = list(entry.get('general') or [])
    recipes = list(entry.get('similarity') or [])
    if pool in ('recipes', 'both') and not recipes:
        raise ValueError(
            f"{dataset} has no 'similarity' recipes registered in "
            f"DATASET_REPS, so pool={pool!r} would be the same run as "
            f"pool='general' under a different name. Register the recipes or "
            f"run --pools general.")
    if pool == 'general':
        if not general:
            raise ValueError(f"{dataset} has no 'general' pool in DATASET_REPS.")
        return general
    if pool == 'recipes':
        return recipes
    if pool == 'both':
        return similarity_rep_names(dataset, include_general=True)
    raise ValueError(f'unknown pool {pool!r}; choose from {POOLS}')


def budget_caps(models, budgets):
    """Which (model, budget) cells the engine will never run, and why.

    MODEL_MAX_BUDGET is a per-model ceiling in find_best_hybrid. A budget above
    a model's ceiling produces no configuration at all, so its absence from the
    per-budget table is structural. Recorded so it is never read as a loss.
    """
    from kirby.find_best_hybrid import MODEL_MAX_BUDGET
    out = {}
    for m in (models or []):
        cap = MODEL_MAX_BUDGET.get(m)
        if cap is None:
            continue
        skipped = [int(b) for b in budgets if int(b) > int(cap)]
        if skipped:
            out[m] = {'max_budget': int(cap), 'budgets_never_run': skipped}
    return out


# ------------------------------------------------------------------- one cell
def _paired_delta(hybrid_folds, baseline_folds, lower_is_better):
    """Mean of the per-fold (hybrid - best single rep) differences.

    The same arithmetic nested_cv_evaluate uses for delta_mean_of_folds, reused
    here to get the paired delta at ONE budget, which it does not report.
    """
    per_fold, paired = [], []
    n = max(len(hybrid_folds or []), len(baseline_folds or []))
    for f in range(n):
        hv = hybrid_folds[f] if hybrid_folds and f < len(hybrid_folds) else None
        bv = baseline_folds[f] if baseline_folds and f < len(baseline_folds) else None
        if hv is None or bv is None or not np.isfinite(hv) or not np.isfinite(bv):
            per_fold.append(None)
            continue
        d = (bv - hv) if lower_is_better else (hv - bv)
        per_fold.append(float(d))
        paired.append(float(d))
    return (float(np.mean(paired)) if paired else None), per_fold, len(paired)


def _f(v):
    """None for anything that is not a finite number, so JSON never carries NaN."""
    try:
        return float(v) if v is not None and np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def summarise_cell(result):
    """Everything one (objective, pool) run produced, in one JSON block."""
    lower = bool(result.lower_is_better)
    out = {
        'metric_name': result.metric_name,
        'lower_is_better': lower,
        # The metric the fitted model was scored with when the objective itself
        # fits no model. Under 'predictive' it repeats metric_name.
        'side_metric_name': getattr(result, 'side_metric_name', None),
        'side_metric_lower_is_better': bool(
            getattr(result, 'side_metric_lower_is_better', False)),
        'n_outer_folds': int(result.n_outer_folds),
        'by_model': {},
        'by_model_budget': {},
        'by_model_budget_side': {},
    }

    for model, e in (result.by_model or {}).items():
        row = {
            # THE number. Paired per outer fold, both arms held out.
            'delta_mean_of_folds': _f(e.get('delta_mean_of_folds')),
            'n_paired_folds': int(e.get('n_paired_folds') or 0),
            'hybrid_mean': _f(e.get('hybrid_mean')),
            'hybrid_std': _f(e.get('hybrid_std')),
            'best_single_mean': _f(e.get('best_single_mean')),
            # ADDED 2026-09-10. These four are on the object nested_cv_evaluate
            # returns and every writer dropped them, so across 1864 stored cells
            # carrying a delta, NONE carried the two numbers that say whether the
            # delta means anything.
            #
            # search_mean is what the inner search said the winning recipe would
            # score; bias is that minus the held-out score of the SAME recipe —
            # the winner's curse, and the hybrid arm alone pays it, because its
            # number is a maximum over hundreds of configurations while each
            # representation's is a single number with no maximum taken. On
            # flab_binding_koenig it ran 0.0317 to 0.0641 spearman against honest
            # deltas of 0.0113 to 0.1215: the same size as the win.
            #
            # The two widths say how much of a delta is method and how much is
            # width. The hybrid is held to its budget; the representation it is
            # compared against keeps every feature it has.
            'search_mean': _f(e.get('search_mean')),
            'bias': _f(e.get('bias')),
            'n_features_per_fold': e.get('n_features_per_fold'),
            'best_single_n_features_per_fold': e.get(
                'best_single_n_features_per_fold'),
            'best_single_std': _f(e.get('best_single_std')),
            'best_single_reps_per_fold': e.get('best_single_reps_per_fold'),
            'per_fold_delta': e.get('per_fold_delta'),
        }
        # The other two numbers for the SAME winner. An objective that wins only
        # on the number it selected by shows up here as a flat or negative
        # delta on the other two.
        for key in SIDE_KEYS:
            if f'{key}_mean' not in e:
                continue
            row[key] = {
                'mean': _f(e.get(f'{key}_mean')),
                'delta_mean_of_folds': _f(e.get(f'{key}_delta_mean_of_folds')),
                'n_paired_folds': int(e.get(f'{key}_n_paired_folds') or 0),
                'fold_scores': e.get(f'{key}_fold_scores'),
            }
        out['by_model'][model] = row

    for model, per_budget in (result.by_model_budget or {}).items():
        base_folds = ((result.by_model or {}).get(model, {})
                      .get('best_single_fold_scores'))
        out['by_model_budget'][model] = {}
        for budget, b in per_budget.items():
            d, per_fold, n_paired = _paired_delta(b.get('fold_scores'),
                                                  base_folds, lower)
            out['by_model_budget'][model][str(budget)] = {
                'delta_mean_of_folds': d,
                'n_paired_folds': n_paired,
                'per_fold_delta': per_fold,
                'hybrid_honest_mean': _f(b.get('honest_mean')),
                'hybrid_honest_std': _f(b.get('honest_std')),
            }

    for side_key, per_model in (getattr(result, 'by_model_budget_side', None)
                                or {}).items():
        prefix = ('struct_fidelity' if side_key == 'struct_fidelity'
                  else f'side_{side_key}')
        out['by_model_budget_side'][prefix] = {}
        for model, per_budget in per_model.items():
            out['by_model_budget_side'][prefix][model] = {
                str(budget): {'mean': _f(b.get('mean')),
                              'std': _f(b.get('std')),
                              'fold_scores': b.get('fold_scores')}
                for budget, b in per_budget.items()
            }
    return out


def best_budget_table(cell):
    """Best budget per model for one (objective, pool), with the fold spread.

    A winner whose margin over the runner-up is inside the fold-to-fold spread
    is not a winner. Both numbers are reported so that is visible rather than
    asserted.
    """
    table = {}
    for model, per_budget in (cell.get('by_model_budget') or {}).items():
        rows = [(int(b), v['delta_mean_of_folds'], v.get('hybrid_honest_std'))
                for b, v in per_budget.items()
                if v.get('delta_mean_of_folds') is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r[1], reverse=True)
        best_b, best_d, best_sd = rows[0]
        runner = rows[1][1] if len(rows) > 1 else None
        table[model] = {
            'best_budget': best_b,
            'best_delta': best_d,
            'fold_std_at_best': best_sd,
            'runner_up_delta': runner,
            'margin_over_runner_up': (None if runner is None
                                      else float(best_d - runner)),
            # The only honest verdict this table can carry on its own.
            'margin_exceeds_fold_std': (
                None if (runner is None or best_sd is None) else
                bool((best_d - runner) > best_sd)),
            'budgets_scored': sorted(int(b) for b in per_budget),
        }
    return table


# ---------------------------------------------------------------------- main
def _result_path(args, dataset):
    """The file this dataset's result is written to. One place, so the
    end-of-run write and the after-every-cell write can never disagree."""
    suffix = '__smoke' if args.smoke else ''
    pools = args.pools.replace(',', '-') if args.pools else 'all'
    return os.path.join(
        args.output_dir,
        f'f27_objective_and_budget_{dataset}__{pools}{suffix}.json')


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_result(out, path):
    """Write to a temporary file in the same directory, then rename over the
    target. A rename is atomic on one filesystem, so a job killed mid-write
    leaves the previous complete file rather than a truncated one."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.partial'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _make_flusher(args, dataset):
    """Rewrite the whole result after every (objective, pool) cell.

    Each cell is hours. Written once at the end, a job that died inside its
    wall left nothing on disk at all -- which is how f21 lost 9 hours.
    """
    path = _result_path(args, dataset)

    def flush(out, objective, pool, state, t0):
        prog = out.setdefault('progress', {
            'cells_planned': len(out['objectives']) * len(out['pools']),
            'cells_attempted': 0,
            'complete': False,
        })
        if objective is not None:
            prog['cells_attempted'] += 1
            prog['last_cell'] = {'objective': objective, 'pool': pool,
                                 'state': state}
        prog['elapsed_seconds'] = round(time.time() - t0, 1)
        out['elapsed_seconds'] = prog['elapsed_seconds']
        try:
            _write_result(out, path)
        except Exception as exc:
            print(f'  [flush] could not write {path}: {exc!r}', flush=True)
            return
        if objective is not None:
            print(f"  [flush] {prog['cells_attempted']}/"
                  f"{prog['cells_planned']} cells -> {os.path.basename(path)}",
                  flush=True)

    return flush


def load_pool(dataset, pool, args):
    """Load one pool's representations, plus the SMILES-keyed MCS pairs."""
    from experiments._shared import load_data

    if args.smoke:
        rep_names = ['ecfp4', 'maccs', 'avalon']
    else:
        rep_names = pool_rep_names(dataset, pool)
    data = load_data(dataset, 'custom', rep_names=rep_names)
    return data


def mcs_pairs_for(dataset, data, args):
    """MCS pairs in THIS dataset's row order, or None with a stated reason.

    get_mcs_pairs is keyed by a content hash of the SMILES list, so pairs built
    against a different row order are never served here -- they are rebuilt.
    The .npz files in tests/experiments/similarity_metrics/ are the July study's
    ordering and must NOT be used: row 900 is a different molecule.
    """
    smiles = data.get('smiles')
    if smiles is None:
        return None, 'dataset has no SMILES, so MCS distance is undefined'
    from kirby.similarity.mcs import get_mcs_pairs
    # --mcs-pairs wins even under --smoke. A smoke that quietly drops to 400
    # pairs cannot reach struct_fid at all: an inner validation fold holds about
    # n_pairs * (its fraction of the dataset) ** 2 of them, and fidelity is NaN
    # below 10. At n_outer=2 that fraction is ~0.1, so 400 pairs leaves ~4.
    n_pairs = args.mcs_pairs
    t = time.time()
    pairs = get_mcs_pairs(dataset, list(smiles),
                          y=np.asarray(data['labels']), n_pairs=n_pairs, seed=0)
    print(f'  MCS pairs: {len(pairs["I"])} ({time.time() - t:.0f}s)', flush=True)
    return pairs, None


def pairs_per_fold_report(pairs, labels, groups, n_outer, seed=42):
    """How many pairs land with BOTH molecules inside one outer fold.

    _structural_fidelity_score returns NaN below 10, so a dataset that cannot
    clear that has no structural number at all and must not be run under
    struct_fid. Counted before the grid rather than discovered inside it.
    """
    from kirby.find_best_hybrid import nested_cv_outer_folds
    if pairs is None:
        return None
    dummy = {'_': np.zeros((len(labels), 1))}
    folds = nested_cv_outer_folds(dummy, labels, groups, n_outer, seed,
                                  'regression')
    I = np.asarray(pairs['I']).astype(int)
    J = np.asarray(pairs['J']).astype(int)
    counts = []
    for _tr, te in folds:
        s = set(int(x) for x in te)
        counts.append(int(sum(1 for i, j in zip(I, J)
                              if int(i) in s and int(j) in s)))
    return {'per_outer_test_fold': counts,
            'min': int(min(counts)) if counts else 0,
            'threshold': 10,
            'usable': bool(counts and min(counts) >= 10)}


def run_dataset(ds, args):
    from experiments._shared import (run_nested_cv, get_config, FULL_BUDGETS,
                                     outer_rep_builders_from_factories)

    config = get_config(args)

    budgets = ([int(b) for b in args.budgets.split(',')] if args.budgets
               else ([25] if args.smoke else list(FULL_BUDGETS)))
    pools = ([p.strip() for p in args.pools.split(',')] if args.pools
             else (['general'] if args.smoke else list(POOLS)))
    wanted = ({o.strip() for o in args.objectives.split(',')}
              if args.objectives else None)
    objectives = [(n, o) for n, o in OBJECTIVES
                  if wanted is None or n in wanted]
    models = ([m.strip() for m in args.models.split(',')] if args.models
              else None)
    importance = ([i.strip() for i in args.importance.split(',')]
                  if args.importance else None)
    evaluators = ([e.strip() for e in args.evaluators.split(',')]
                  if args.evaluators else None)

    out = {
        'experiment': 'f27_objective_and_budget',
        'dataset': ds,
        'tier': TIER,
        'budgets': budgets,
        'pools': pools,
        'objectives': [n for n, _ in objectives],
        'reference_objective': REFERENCE_OBJECTIVE,
        'smoke': bool(args.smoke),
        'models': models,
        'importance_methods': importance,
        'internal_evaluators': evaluators,
        'n_outer_folds_requested': args.outer_folds or config['n_splits'],
        'budget_caps': budget_caps(models, budgets),
        'pool_reps': {},
        'mcs': {},
        'results': {},
        'skipped': [],
    }

    t0 = time.time()
    _flush = _make_flusher(args, ds)

    for pool in pools:
        print(f'\n### {ds} | pool={pool} ###', flush=True)
        try:
            data = load_pool(ds, pool, args)
        except Exception as exc:
            for name, _obj in objectives:
                out['skipped'].append({
                    'objective': name, 'pool': pool,
                    'error': f'pool did not load: {exc!r}',
                    'traceback': traceback.format_exc(limit=8)})
            print(f'  SKIPPED whole pool {pool}: {exc!r}', flush=True)
            _flush(out, None, None, 'pool-failed', t0)
            continue

        reps = data['reps']
        labels = np.asarray(data['labels'])
        groups = data.get('groups')

        # The sim_ recipes have no whole-dataset matrix: each fits its column
        # selection and its z-score statistics on the fold's training rows. They
        # arrive from load_data as per-fold factories and are rebuilt once per
        # OUTER fold here, so the recipes are scored on the outer test fold on
        # the same terms as every other representation — in the hybrid AND as a
        # baseline. Handing them to the inner search instead would leave them
        # out of both outer arms without saying so.
        _fold_names = list(data.get('fold_fitted_reps') or [])
        builders = outer_rep_builders_from_factories(
            data.get('rep_factories') or {}, len(labels), names=_fold_names)
        if len(builders) != len(_fold_names):
            raise RuntimeError(
                f'{ds}/{pool}: {sorted(set(_fold_names) - set(builders))} have '
                f'no factory, so the pool would run short of what it names.')
        out['pool_reps'][pool] = {k: list(np.shape(v))
                                  for k, v in sorted(reps.items())}
        for _bn in sorted(builders):
            out['pool_reps'][pool][_bn] = 'per-fold'
        out['n_samples'] = int(len(labels))

        pairs, why = mcs_pairs_for(ds, data, args)
        if pool not in out['mcs']:
            out['mcs'][pool] = {
                'available': pairs is not None,
                'reason': why,
                'n_pairs': (None if pairs is None else int(len(pairs['I']))),
                'per_fold': pairs_per_fold_report(
                    pairs, labels, groups,
                    args.outer_folds or config['n_splits']),
            }

        for name, objective in objectives:
            needs_mcs = (objective == 'struct_fid'
                         or (isinstance(objective, dict)
                             and 'struct_fid' in objective))
            if needs_mcs and pairs is None:
                out['skipped'].append({
                    'objective': name, 'pool': pool,
                    'error': f'objective needs MCS pairs: {why}'})
                print(f'  SKIPPED {name}/{pool}: no MCS pairs ({why})',
                      flush=True)
                _flush(out, name, pool, 'skipped', t0)
                continue

            print(f'\n=== {ds} | pool={pool} | objective={name} '
                  f'({time.time() - t0:.0f}s elapsed) ===', flush=True)
            try:
                result = run_nested_cv(
                    reps=reps, labels=labels, groups=groups, config=config,
                    dataset_name=ds,
                    task_type=data.get('task_type', 'auto'),
                    domain=data.get('domain'),
                    tier=TIER,
                    budgets=budgets,
                    objective=objective,
                    structural_pairs=pairs,
                    n_outer_folds=args.outer_folds,
                    **({'outer_rep_builders': builders,
                        # The recipes' VALUES change with the recipe code while
                        # the cache key sees only names and widths, so the id
                        # carries that change into every fold's namespace.
                        'outer_rep_builders_id': f'simrecipes_{pool}_v1'}
                       if builders else {}),
                    **({'models': models} if models else {}),
                    **({'importance_methods': importance} if importance else {}),
                    **({'internal_evaluators': evaluators} if evaluators else {}),
                    # objective and pool are in the search META but not in the
                    # search FILENAME, so without this every cell writes the
                    # same file and evicts the previous one -- correct results,
                    # all recomputed cold. condition_suffix lands in the search
                    # filename only, so the per-representation baseline files
                    # are still shared across cells that use the same pool.
                    condition_suffix=f'obj-{name}__pool-{pool}',
                    verbose=True,
                )
                cell = summarise_cell(result)
                cell['best_budget_per_model'] = best_budget_table(cell)
                out['results'].setdefault(pool, {})[name] = cell
                _flush(out, name, pool, 'done', t0)
            except Exception as exc:
                out['skipped'].append({
                    'objective': name, 'pool': pool,
                    'error': repr(exc),
                    'traceback': traceback.format_exc(limit=8),
                })
                print(f'  SKIPPED {name}/{pool}: {exc!r}', flush=True)
                _flush(out, name, pool, 'skipped', t0)

    out['elapsed_seconds'] = round(time.time() - t0, 1)
    out.setdefault('progress', {
        'cells_planned': len(out['objectives']) * len(out['pools']),
        'cells_attempted': 0,
    })['complete'] = True
    _flush(out, None, None, 'complete', t0)
    return out


def print_report(out):
    ds = out['dataset']
    print('\n' + '=' * 96)
    print(f'F27 OBJECTIVE AND BUDGET -- {ds}   (n={out.get("n_samples")})')
    print('=' * 96)

    for pool, per_obj in (out.get('results') or {}).items():
        n_reps = len(out.get('pool_reps', {}).get(pool, {}))
        mcs = (out.get('mcs') or {}).get(pool) or {}
        pf = mcs.get('per_fold') or {}
        print(f'\n--- pool = {pool}  ({n_reps} representations; '
              f'MCS pairs {mcs.get("n_pairs")}, '
              f'min per outer fold {pf.get("min")}, '
              f'usable {pf.get("usable")}) ---')

        models = sorted({m for c in per_obj.values()
                         for m in c.get('by_model', {})})
        for model in models:
            print(f'\n  model = {model}')
            print(f'    {"objective":14s} {"picked-by":>10s} '
                  f'{"predict":>9s} {"neighbour":>10s} {"MCS":>9s} '
                  f'{"best budget":>12s}')
            print(f'    {"":14s} {"(its own)":>10s} '
                  f'{"delta":>9s} {"delta":>10s} {"delta":>9s}')
            for name in out['objectives']:
                c = per_obj.get(name)
                if not c or model not in c.get('by_model', {}):
                    continue
                e = c['by_model'][model]

                def col(key):
                    v = (e.get(key) or {}).get('delta_mean_of_folds')
                    return f'{v:+9.4f}' if v is not None else f'{"n/a":>9s}'

                own = e.get('delta_mean_of_folds')
                own_s = f'{own:+10.4f}' if own is not None else f'{"n/a":>10s}'
                bb = (c.get('best_budget_per_model') or {}).get(model) or {}
                bb_s = (f'{bb["best_budget"]}' if bb.get('best_budget')
                        is not None else 'n/a')
                if bb.get('margin_exceeds_fold_std') is False:
                    bb_s += '~'      # inside the fold spread; not a winner
                print(f'    {name:14s} {own_s} {col("side_predictive")} '
                      f'{col("side_readacross"):>10s} '
                      f'{col("struct_fidelity")} {bb_s:>12s}')
            print('    ~ = the best budget beat the runner-up by less than the '
                  'fold spread, so it is a tie, not a winner.')

    caps = out.get('budget_caps') or {}
    if caps:
        print('\nBudgets that never ran, per model (MODEL_MAX_BUDGET):')
        for m, c in sorted(caps.items()):
            print(f'  {m}: cap {c["max_budget"]}, never run '
                  f'{c["budgets_never_run"]}')

    if out.get('skipped'):
        print(f'\nSKIPPED CELLS ({len(out["skipped"])}):')
        for s in out['skipped']:
            print(f'  {s["pool"]}/{s["objective"]}: {s["error"]}')
    else:
        print('\nNo cells skipped.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mode', default='full', choices=['fast', 'full',
                                                       'intermediary'])
    ap.add_argument('--pools', default=None,
                    help=f'comma-separated; default {",".join(POOLS)}')
    ap.add_argument('--objectives', default=None,
                    help='comma-separated; default all five: '
                         + ','.join(n for n, _ in OBJECTIVES))
    ap.add_argument('--budgets', default=None,
                    help='comma-separated; default the full list')
    ap.add_argument('--models', default=None,
                    help='comma-separated downstream models. Default None, '
                         'which is the roster the SLURM precompute caches '
                         'under -- change it only for a plumbing check.')
    ap.add_argument('--outer-folds', type=int, default=None, dest='outer_folds',
                    help='outer CV folds. Default the config n_splits (5).')
    ap.add_argument('--importance', default=None,
                    help='comma-separated importance methods. Default all three.')
    ap.add_argument('--evaluators', default=None,
                    help='comma-separated internal evaluators. Default all '
                         'five. lightgbm SEGFAULTS locally once the rdkit '
                         'loader has imported, so a local run wants rf.')
    ap.add_argument('--mcs-pairs', type=int, default=8000, dest='mcs_pairs',
                    help='MCS pairs per dataset. 0.28 s/pair measured, so the '
                         'default costs about 37 min the first time a dataset '
                         'is run and nothing after that.')
    ap.add_argument('--smoke', action='store_true',
                    help='plumbing check: 3 cheap representations, 1 budget, '
                         '400 MCS pairs, one pool. Proves dispatch, measures '
                         'nothing.')
    ap.add_argument('--output-dir', dest='output_dir',
                    default=os.path.join(HERE, '..', 'results'))
    args = ap.parse_args()

    out = run_dataset(args.dataset, args)
    print_report(out)

    path = _result_path(args, args.dataset)
    _write_result(out, path)
    print(f'\nwrote {path}')
    return out


if __name__ == '__main__':
    main()
