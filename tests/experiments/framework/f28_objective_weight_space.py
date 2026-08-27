#!/usr/bin/env python3
"""F28: every objective on its own, then every blend of them.

Paper: Framework. This decides what the search should maximise, so it is a
framework experiment, not a noise or similarity one.

THE QUESTION
    The search picks a feature subset by maximising ONE number, and which
    number is a free choice. Three are available:

      predict     a model is fitted on the subset and scored on held-out rows
      neighbour   each held-out molecule's property predicted from its five
                  nearest neighbours, Spearman'd against truth, scaffold-grouped
      MCS         Spearman between pairwise distance and MCS distance -- whether
                  two molecules the space calls close really do share atoms

    Nobody has run them against each other inside the real search, and nobody
    has run any mixture of them at all. This does both.

THE GRID -- 15 cells per dataset
    Weights over the three, in steps of 0.25. Fifteen points on the simplex:
    three corners (the objectives on their own) and twelve blends.

      3 singles      1.00 of one objective
      6 two-way      0.75 / 0.25 both ways round, for each of the three pairs
      3 two-way      0.50 / 0.50, for each of the three pairs
      3 three-way    0.50 / 0.25 / 0.25, each objective taking the half in turn

    Three points per pair, not two, so a pair that trades off shows a TURN
    rather than a straight line. --centre adds 1/3 each as a 16th cell; it is
    not a multiple of 0.25 so it is off by default.

ALL THREE ON ONE SCALE
    A weighted mean needs its parts on one scale, and the dataset's own metric
    is not: RMSE on seven of the eight molecular sets, against two Spearman
    correlations. So INSIDE A BLEND ONLY, prediction is scored as a correlation
    between predicted and true on the same held-out rows -- Spearman for
    regression, 2*(AUROC - 0.5) for classification, which puts chance at 0 the
    way a correlation does. All three are then bounded in [-1, 1] and
    higher-is-better.

    The plain objective='predictive' is untouched and still scores the dataset's
    own metric. Both numbers are recorded for every winner either way.

WHAT MAKES IT AFFORDABLE, and what that costs
    ★ --max-reps 3 runs the allocation methods top_2 and top_3 only, so a hybrid
    draws on at most 3 representations instead of all 10. top_3 takes the three
    BEST-SCORING representations -- it does not try all 120 trios. The cap is a
    cap, not an enumeration.
    ★ --outer-folds 3 instead of 5.

    Together those are 4.5x fewer allocation methods and 1.7x fewer outer folds,
    so 7.5x off a default cell. Everything else is identical to
    find_best_hybrid: the same create_hybrid, the same importance methods, the
    same internal evaluators, the same full budget list, the same honest outer
    fold.

    A cap on the number of representations is a real restriction on the answer.
    An objective whose advantage only appears when it can combine six
    representations cannot show it here. Stated so nobody reads a null as
    "the objective does not matter".

ONE POOL
    The dataset's signed-off pool. The representation-pool axis is what f27 is
    for; running it here as well would multiply this grid by three for a
    question already being asked elsewhere.

WHAT IS REPORTED
    delta_mean_of_folds from nested_cv_evaluate -- the hybrid minus the best
    single representation, paired per outer fold, both arms having paid the
    identical held-out tax. Never search_mean, never .all_results.
    Every winner is then read on all four recorded numbers whichever weights
    picked it: the dataset's own metric, prediction as a correlation, the
    nearest-neighbour score, and MCS fidelity. A blend that wins only on its own
    weighted sum is visible as such.

SKIPS ARE LOGGED. Any cell that fails is written into 'skipped' with its error.
MODEL_MAX_BUDGET caps svm at 500 and gp/mlp/bnn/dnn/cnn at 1000, so those
(model, budget) cells never run; they are listed under 'budget_caps' so a
missing high budget is never read as a budget that lost.

WALL TIME. 15 cells per dataset. A default cell is f26's 3.5-11 h, and the two
caps take 7.5x off, so roughly 28-88 minutes each and 7-22 h per dataset. That
is a structural estimate from configuration counts, NOT a measured time. Run
--smoke first and set the wall from it.

USAGE
    # one array task on the server
    python tests/experiments/framework/f28_objective_weight_space.py \\
        --dataset herg_ki

    # local plumbing check
    python tests/experiments/framework/f28_objective_weight_space.py \\
        --dataset herg_ki --smoke --outer-folds 2 --models rf \\
        --evaluators rf --importance treeshap --mcs-pairs 4000

    # print the grid and stop -- no data loaded, no search
    python tests/experiments/framework/f28_objective_weight_space.py \\
        --dataset herg_ki --list-grid
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import traceback
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))     # tests/
sys.path.insert(0, os.path.join(HERE, '..'))           # tests/experiments/

TIER = 'f28_objweight'       # own cache identity; touches no existing result DB

# Engine names, in the order the weights are written.
COMPONENTS = ('predictive', 'similarity', 'struct_fid')
# What each one is, in words, for the report.
PLAIN = {'predictive': 'predict', 'similarity': 'neighbour',
         'struct_fid': 'MCS'}
# Which recorded number each component reads, so a blend can be checked against
# its parts. 'side_predictive_corr' is the correlation form, NOT the dataset's
# own metric -- see ALL THREE ON ONE SCALE above.
COMPONENT_COLUMN = {'predictive': 'side_predictive_corr',
                    'similarity': 'side_readacross',
                    'struct_fid': 'struct_fidelity'}
# Every number a winner carries, whichever weights picked it.
REPORT_COLUMNS = ('side_predictive', 'side_predictive_corr',
                  'side_readacross', 'struct_fidelity')

# The objective every other cell is read against: the current default.
REFERENCE_CELL = 'predict'

STEP = 4                     # quarters


def build_grid(step=STEP, centre=False):
    """(cell name, objective) for every point on the weight simplex.

    A corner is a single objective and is passed as its plain name, so it lands
    on the same cache entry a plain run would and is scored exactly the same
    way. Only genuine mixtures are passed as a dict.
    """
    cells = []
    pts = [p for p in product(range(step + 1), repeat=len(COMPONENTS))
           if sum(p) == step]
    # corners first, then two-way, then three-way — the order the report reads in
    pts.sort(key=lambda p: (sum(1 for w in p if w), [-w for w in p]))
    for p in pts:
        weights = {c: w / step for c, w in zip(COMPONENTS, p) if w}
        if len(weights) == 1:
            name = PLAIN[next(iter(weights))]
            cells.append((name, next(iter(weights))))
        else:
            name = '+'.join(f'{PLAIN[c]}{w:.2f}'.replace('0.', '.')
                            for c, w in weights.items())
            cells.append((name, weights))
    if centre:
        third = 1.0 / 3.0
        cells.append(('centre', {c: third for c in COMPONENTS}))
    return cells


def budget_caps(models, budgets):
    """Which (model, budget) cells the engine will never run, and why."""
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


def _f(v):
    """None for anything that is not a finite number, so JSON never holds NaN."""
    try:
        return float(v) if v is not None and np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


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


def summarise_cell(result, objective):
    """One cell's numbers, plus the check that a blend equals its parts."""
    lower = bool(result.lower_is_better)
    out = {
        'metric_name': result.metric_name,
        'lower_is_better': lower,
        'side_metric_name': getattr(result, 'side_metric_name', None),
        'side_metric_lower_is_better': bool(
            getattr(result, 'side_metric_lower_is_better', False)),
        'n_outer_folds': int(result.n_outer_folds),
        'weights': (dict(objective) if isinstance(objective, dict)
                    else {objective: 1.0}),
        'by_model': {},
        'by_model_budget': {},
    }

    for model, e in (result.by_model or {}).items():
        row = {
            'delta_mean_of_folds': _f(e.get('delta_mean_of_folds')),
            'n_paired_folds': int(e.get('n_paired_folds') or 0),
            'hybrid_mean': _f(e.get('hybrid_mean')),
            'best_single_mean': _f(e.get('best_single_mean')),
            'best_single_reps_per_fold': e.get('best_single_reps_per_fold'),
            'per_fold_delta': e.get('per_fold_delta'),
            'reps_per_fold': None,
        }
        # How many representations the winner actually drew on, per fold. The
        # cap is at most 3; a hybrid that used 1 is a different thing from one
        # that used 3 and the report should not have to guess.
        bps = e.get('blueprints_per_fold') or []
        counts = []
        for bp in bps:
            reps = getattr(bp, 'reps', None)
            if reps is None and isinstance(bp, dict):
                reps = bp.get('reps')
            counts.append(None if reps is None else int(len(reps)))
        row['reps_per_fold'] = counts

        for key in REPORT_COLUMNS:
            if f'{key}_mean' not in e:
                continue
            row[key] = {
                'mean': _f(e.get(f'{key}_mean')),
                'delta_mean_of_folds': _f(e.get(f'{key}_delta_mean_of_folds')),
                'n_paired_folds': int(e.get(f'{key}_n_paired_folds') or 0),
            }

        # A blend's own delta must equal the weighted mean of its components'
        # deltas -- both are means over the same folds of the same quantities.
        # Recorded rather than asserted, so a mismatch is visible in the file
        # instead of stopping a 20-hour run.
        if isinstance(objective, dict):
            acc, complete = 0.0, True
            for comp, w in objective.items():
                v = (row.get(COMPONENT_COLUMN[comp]) or {}).get(
                    'delta_mean_of_folds')
                if v is None:
                    complete = False
                    break
                acc += float(w) * float(v)
            own = row['delta_mean_of_folds']
            row['blend_identity'] = {
                'weighted_mean_of_components': (float(acc) if complete
                                                else None),
                'own_delta': own,
                'gap': (None if (not complete or own is None)
                        else float(abs(acc - own))),
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
                'hybrid_honest_mean': _f(b.get('honest_mean')),
                'hybrid_honest_std': _f(b.get('honest_std')),
            }
    return out


# ---------------------------------------------------------------------- main
def _result_path(args, dataset):
    suffix = '__smoke' if args.smoke else ''
    return os.path.join(args.output_dir,
                        f'f28_objective_weight_space_{dataset}{suffix}.json')


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_result(out, path):
    """Write to a temporary file beside the target, then rename over it. A
    rename is atomic on one filesystem, so a job killed mid-write leaves the
    previous complete file rather than a truncated one."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.partial'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _make_flusher(args, dataset):
    """Rewrite the whole result after every cell. Written once at the end, a
    job that died inside its wall left nothing on disk at all."""
    path = _result_path(args, dataset)

    def flush(out, cell, state, t0):
        prog = out.setdefault('progress', {
            'cells_planned': len(out['grid']),
            'cells_attempted': 0,
            'complete': False,
        })
        if cell is not None:
            prog['cells_attempted'] += 1
            prog['last_cell'] = {'cell': cell, 'state': state}
        prog['elapsed_seconds'] = round(time.time() - t0, 1)
        out['elapsed_seconds'] = prog['elapsed_seconds']
        try:
            _write_result(out, path)
        except Exception as exc:
            print(f'  [flush] could not write {path}: {exc!r}', flush=True)
            return
        if cell is not None:
            print(f"  [flush] {prog['cells_attempted']}/"
                  f"{prog['cells_planned']} cells -> {os.path.basename(path)}",
                  flush=True)

    return flush


def mcs_pairs_for(dataset, data, args):
    """MCS pairs in THIS dataset's row order, or None with a stated reason.

    get_mcs_pairs is keyed by a content hash of the SMILES list, so pairs built
    against a different row order are rebuilt rather than served. The .npz files
    in tests/experiments/similarity_metrics/ are the July study's ordering and
    must NOT be used: row 900 is a different molecule.
    """
    smiles = data.get('smiles')
    if smiles is None:
        return None, 'dataset has no SMILES, so MCS distance is undefined'
    from kirby.similarity.mcs import get_mcs_pairs
    t = time.time()
    pairs = get_mcs_pairs(dataset, list(smiles),
                          y=np.asarray(data['labels']),
                          n_pairs=args.mcs_pairs, seed=0)
    print(f'  MCS pairs: {len(pairs["I"])} ({time.time() - t:.0f}s)', flush=True)
    return pairs, None


def pairs_per_fold_report(pairs, labels, groups, n_outer, seed=42):
    """How many pairs land with BOTH molecules inside one outer fold.

    _structural_fidelity_score returns NaN below 10, so a dataset that cannot
    clear that has no MCS number at all and every cell containing MCS will
    refuse. Counted before the grid rather than discovered inside it.
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
    from experiments._shared import load_data, run_nested_cv, get_config, FULL_BUDGETS
    from experiments.complexity.complexity_helpers import DATASET_REPS

    config = get_config(args)
    grid = build_grid(centre=args.centre)
    wanted = ({c.strip() for c in args.cells.split(',')} if args.cells else None)
    if wanted is not None:
        grid = [(n, o) for n, o in grid if n in wanted]

    budgets = ([int(b) for b in args.budgets.split(',')] if args.budgets
               else ([25] if args.smoke else list(FULL_BUDGETS)))
    # top_2 and top_3 are registered in NAMED_ALLOCATIONS and both resolve to
    # performance_weighted, which is one of the three allocators that can hold a
    # whole blend. max_reps=3 means top_2 and top_3; max_reps=5 adds top_5.
    allocs = [a for a in ('top_2', 'top_3', 'top_5')
              if int(a.split('_')[1]) <= args.max_reps]
    if not allocs:
        raise ValueError(f'--max-reps {args.max_reps} leaves no top_k '
                         f'allocation method; the smallest is top_2.')

    models = ([m.strip() for m in args.models.split(',')] if args.models
              else None)
    importance = ([i.strip() for i in args.importance.split(',')]
                  if args.importance else None)
    evaluators = ([e.strip() for e in args.evaluators.split(',')]
                  if args.evaluators else None)

    if args.smoke:
        rep_names = ['ecfp4', 'maccs', 'avalon']
    else:
        rep_names = list((DATASET_REPS.get(ds) or {}).get('general') or [])
        if not rep_names:
            raise ValueError(f"{ds} has no 'general' pool in DATASET_REPS.")

    data = load_data(ds, 'custom', rep_names=rep_names)
    reps = data['reps']
    labels = np.asarray(data['labels'])
    groups = data.get('groups')
    n_outer = args.outer_folds or config['n_splits']

    pairs, why = mcs_pairs_for(ds, data, args)

    out = {
        'experiment': 'f28_objective_weight_space',
        'dataset': ds,
        'tier': TIER,
        'n_samples': int(len(labels)),
        'reps': {k: list(np.shape(v)) for k, v in sorted(reps.items())},
        'max_reps': int(args.max_reps),
        'allocation_methods': allocs,
        'budgets': budgets,
        'grid': [n for n, _ in grid],
        'weights': {n: (dict(o) if isinstance(o, dict) else {o: 1.0})
                    for n, o in grid},
        'reference_cell': REFERENCE_CELL,
        'smoke': bool(args.smoke),
        'models': models,
        'importance_methods': importance,
        'internal_evaluators': evaluators,
        'n_outer_folds_requested': n_outer,
        'budget_caps': budget_caps(models, budgets),
        'mcs': {'available': pairs is not None, 'reason': why,
                'n_pairs': (None if pairs is None else int(len(pairs['I']))),
                'per_fold': pairs_per_fold_report(pairs, labels, groups,
                                                  n_outer)},
        'results': {},
        'skipped': [],
    }

    t0 = time.time()
    _flush = _make_flusher(args, ds)

    for cell_name, objective in grid:
        needs_mcs = ('struct_fid' == objective
                     or (isinstance(objective, dict)
                         and 'struct_fid' in objective))
        if needs_mcs and pairs is None:
            out['skipped'].append({'cell': cell_name,
                                   'error': f'needs MCS pairs: {why}'})
            print(f'  SKIPPED {cell_name}: no MCS pairs ({why})', flush=True)
            _flush(out, cell_name, 'skipped', t0)
            continue

        print(f'\n=== {ds} | {cell_name} | {objective} '
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
                allocation_methods=allocs,
                structural_pairs=pairs,
                n_outer_folds=args.outer_folds,
                **({'models': models} if models else {}),
                **({'importance_methods': importance} if importance else {}),
                **({'internal_evaluators': evaluators} if evaluators else {}),
                # The objective is in the search META but not in the search
                # FILENAME, so without this every cell writes the same file and
                # evicts the previous one -- correct results, all recomputed
                # cold. condition_suffix lands in the search filename only, so
                # the per-(representation, model) baseline files stay shared.
                condition_suffix=f'cell-{cell_name}',
                verbose=True,
            )
            out['results'][cell_name] = summarise_cell(result, objective)
            _flush(out, cell_name, 'done', t0)
        except Exception as exc:
            out['skipped'].append({
                'cell': cell_name,
                'weights': (dict(objective) if isinstance(objective, dict)
                            else {objective: 1.0}),
                'error': repr(exc),
                'traceback': traceback.format_exc(limit=8),
            })
            print(f'  SKIPPED {cell_name}: {exc!r}', flush=True)
            _flush(out, cell_name, 'skipped', t0)

    out['elapsed_seconds'] = round(time.time() - t0, 1)
    out.setdefault('progress', {'cells_planned': len(grid),
                                'cells_attempted': 0})['complete'] = True
    _flush(out, None, 'complete', t0)
    return out


def print_report(out):
    ds = out['dataset']
    mcs = out.get('mcs') or {}
    pf = mcs.get('per_fold') or {}
    print('\n' + '=' * 104)
    print(f'F28 OBJECTIVE WEIGHT SPACE -- {ds}   (n={out.get("n_samples")}, '
          f'{len(out.get("reps") or {})} representations, '
          f'at most {out.get("max_reps")} combined)')
    print(f'  MCS pairs {mcs.get("n_pairs")}, min per outer fold '
          f'{pf.get("min")} (needs {pf.get("threshold")}), '
          f'usable {pf.get("usable")}')
    print('=' * 104)

    per_cell = out.get('results') or {}
    models = sorted({m for c in per_cell.values() for m in (c.get('by_model') or {})})
    for model in models:
        print(f'\n  model = {model}   '
              f'(delta = hybrid minus best single representation, + is better)')
        print(f'    {"cell":22s} {"own":>9s} {"predict":>9s} {"corr":>9s} '
              f'{"neighbour":>10s} {"MCS":>9s} {"reps":>6s}  vs predict')
        ref = (per_cell.get(REFERENCE_CELL, {}).get('by_model', {})
               .get(model) or {})
        rd = ref.get('delta_mean_of_folds')
        for cell_name in out.get('grid', []):
            c = per_cell.get(cell_name)
            e = (c or {}).get('by_model', {}).get(model)
            if not e:
                continue

            def col(key, w=9):
                v = (e.get(key) or {}).get('delta_mean_of_folds')
                return f'{v:+{w}.4f}' if v is not None else f'{"n/a":>{w}s}'

            own = e.get('delta_mean_of_folds')
            own_s = f'{own:+9.4f}' if own is not None else f'{"n/a":>9s}'
            reps = [r for r in (e.get('reps_per_fold') or []) if r]
            reps_s = (f'{min(reps)}-{max(reps)}' if reps else 'n/a')
            if cell_name == REFERENCE_CELL:
                vs = '  (this)'
            elif rd is None or own is None:
                vs = '  n/a'
            else:
                # Both are in the same units only when both cells are scored on
                # the same quantity, which they are not across the grid. The
                # honest comparison is column by column, so this is the
                # PREDICTION column against the reference's prediction column.
                a = (e.get('side_predictive') or {}).get('delta_mean_of_folds')
                b = (ref.get('side_predictive') or {}).get('delta_mean_of_folds')
                vs = ('  n/a' if a is None or b is None
                      else f'  {a - b:+.4f} on predict')
            print(f'    {cell_name:22s} {own_s} {col("side_predictive")} '
                  f'{col("side_predictive_corr")} '
                  f'{col("side_readacross", 10)} {col("struct_fidelity")} '
                  f'{reps_s:>6s}{vs}')
        print('    own = the number that cell was maximising, so it is not '
              'comparable across rows.')
        print('    The four columns after it ARE comparable down a column.')

        bad = []
        for cell_name, c in per_cell.items():
            bi = ((c.get('by_model') or {}).get(model) or {}).get(
                'blend_identity') or {}
            gap = bi.get('gap')
            if gap is not None and gap > 1e-9:
                bad.append((cell_name, gap))
        if bad:
            print('    ⚠ a blend\'s own delta did NOT equal the weighted mean '
                  'of its parts:')
            for cell_name, gap in bad:
                print(f'      {cell_name}: gap {gap:.2e}')

    caps = out.get('budget_caps') or {}
    if caps:
        print('\nBudgets that never ran, per model (MODEL_MAX_BUDGET):')
        for m, c in sorted(caps.items()):
            print(f'  {m}: cap {c["max_budget"]}, never run '
                  f'{c["budgets_never_run"]}')

    if out.get('skipped'):
        print(f'\nSKIPPED CELLS ({len(out["skipped"])}):')
        for s in out['skipped']:
            print(f'  {s["cell"]}: {str(s["error"])[:160]}')
    else:
        print('\nNo cells skipped.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mode', default='full', choices=['fast', 'full',
                                                       'intermediary'])
    ap.add_argument('--max-reps', type=int, default=3, dest='max_reps',
                    help='most representations a hybrid may draw on. 3 (the '
                         'default) runs top_2 and top_3. top_3 takes the three '
                         'BEST-SCORING representations; it does not try every '
                         'trio.')
    ap.add_argument('--outer-folds', type=int, default=3, dest='outer_folds',
                    help='outer CV folds. Default 3 here, not the config 5, '
                         'because this grid is 15 cells wide.')
    ap.add_argument('--budgets', default=None,
                    help='comma-separated; default the full list')
    ap.add_argument('--cells', default=None,
                    help='comma-separated cell names; default all 15. Use it '
                         'to resubmit the cells a killed task did not reach.')
    ap.add_argument('--centre', action='store_true',
                    help='add 1/3 of each as a 16th cell. Not a multiple of '
                         '0.25, so it is off by default.')
    ap.add_argument('--models', default=None,
                    help='comma-separated downstream models. Default None, '
                         'which is the roster the SLURM precompute caches '
                         'under -- change it only for a plumbing check.')
    ap.add_argument('--importance', default=None,
                    help='comma-separated importance methods. Default all three.')
    ap.add_argument('--evaluators', default=None,
                    help='comma-separated internal evaluators. Default all '
                         'five. lightgbm SEGFAULTS locally once the rdkit '
                         'loader has imported, so a local run wants rf.')
    ap.add_argument('--mcs-pairs', type=int, default=8000, dest='mcs_pairs',
                    help='MCS pairs per dataset. 0.024 s/pair measured, so the '
                         'default costs about 3 minutes the first time a '
                         'dataset is run and nothing after that.')
    ap.add_argument('--list-grid', action='store_true', dest='list_grid',
                    help='print the 15 cells and stop. Loads no data.')
    ap.add_argument('--smoke', action='store_true',
                    help='plumbing check: 3 cheap representations, 1 budget.')
    ap.add_argument('--output-dir', dest='output_dir',
                    default=os.path.join(HERE, '..', 'results'))
    args = ap.parse_args()

    if args.list_grid:
        grid = build_grid(centre=args.centre)
        print(f'{len(grid)} cells\n')
        for i, (name, obj) in enumerate(grid, 1):
            if isinstance(obj, dict):
                w = ', '.join(f'{PLAIN[c]} {v:.2f}' for c, v in obj.items())
            else:
                w = f'{PLAIN[obj]} 1.00'
            print(f'  {i:2d}. {name:22s} {w}')
        return

    out = run_dataset(args.dataset, args)
    print_report(out)

    path = _result_path(args, args.dataset)
    _write_result(out, path)
    print(f'\nwrote {path}')
    return out


if __name__ == '__main__':
    main()
