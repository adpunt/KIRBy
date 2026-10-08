"""Loader for thesis figure F3 (THESIS_OUTLINE.md section 5.6): which base
representation each selected feature of a hybrid came from.

Source: c22_explainability, one file per dataset. For every outer fold and
every decided model, c22 stores the two best hybrids ranked on that fold's
training rows (slots 'hybrid1', 'hybrid2'). Each stored hybrid carries
'allocation_built': {representation: number of features the build took}.
Its values sum to the hybrid's 'n_features' in every file read on
2026-10-09.

Rule for one bar: slot 'hybrid1' (the top-ranked hybrid on that fold's
training rows) for model M on fold F. All models and all folds are
returned; the figure picks. Pass slots=('hybrid1', 'hybrid2') for both.

c24_blueprint_transfer stores the same counts ('features_by_rep') for the
same hybrid; on 2026-10-09 199 of 200 (dataset, fold, model) entries were
identical to c22 hybrid1 (the one difference: tcga_brca_pam50 fold 2 dnn,
same configuration, deeplift importance, counts differ by 1-3 features).

Not stored anywhere in the results: the selected feature indices. c22 and
c24 count them, then discard them; the search cache keeps only a hash of
them. So the deeper panel (named feature groups) cannot be built from disk;
load_f3_detail reports that through missing().
"""
import ast
import glob
import json
import pathlib
import sys

INPUTS = [
    'complexity/c22_explainability/*.json',
    'complexity/c22_explainability/*.partial.jsonl',
]

C22 = 'complexity/c22_explainability'
N_FOLDS = 5
SLOTS = ('hybrid1',)

_here = pathlib.Path(__file__).resolve()
REPO = _here.parents[2] if len(_here.parents) > 2 else None


def _module_assign(path, name):
    """The AST value assigned to `name` at module level of `path`."""
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value
    raise KeyError(f'{name} not found in {path}')


def expected_datasets():
    """DATASET_REPS keys minus EXCLUDED_DATASETS and CUT_DATASETS
    (plan.dataset_roster), read without importing either module."""
    tests = REPO / 'tests' / 'experiments'
    plan = tests / 'orchestrate' / 'plan.py'
    cut = set(ast.literal_eval(_module_assign(plan, 'CUT_DATASETS')))
    excl = set(ast.literal_eval(_module_assign(plan, 'EXCLUDED_DATASETS')))
    reps = _module_assign(tests / 'complexity' / 'complexity_helpers.py',
                          'DATASET_REPS')
    keys = [k.value for k in reps.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return sorted(d for d in keys if d not in cut and d not in excl)


def _decode():
    sys.path.insert(0, str(REPO / 'tests' / 'experiments'))
    from _checkpoint import decode
    return decode


def _read_dataset(results_root, ds):
    """(models expected, {fold: fold record}, source path) or None."""
    full = results_root / C22 / f'{ds}.json'
    part = results_root / C22 / f'{ds}.partial.jsonl'
    if full.exists():
        d = json.loads(full.read_text())
        folds = {int(p['fold']): p for p in d.get('per_fold', [])}
        return list(d.get('models') or []), folds, str(full.relative_to(results_root))
    if part.exists():
        decode = _decode()
        models, folds = [], {}
        for line in part.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if 'identity' in o:
                models = list((o['identity'].get('config') or {}).get('models') or [])
            elif 'payload' in o:
                p = decode(o['payload'])
                if isinstance(p, dict) and 'models' in p:
                    folds[int(p.get('fold', o['key'].get('fold')))] = p
        return models, folds, str(part.relative_to(results_root)) + ' (partial)'
    return None


def load_f3(results_root: pathlib.Path, missing, slots=SLOTS) -> list:
    """One dict per (dataset, model, fold, slot, representation) segment."""
    results_root = pathlib.Path(results_root)
    rows = []
    for ds in expected_datasets():
        got = _read_dataset(results_root, ds)
        if got is None:
            missing(f'{ds}: no c22_explainability file (.json or .partial.jsonl); '
                    f'no blueprint for any model or fold')
            continue
        models, folds, source = got
        if not models:
            models = sorted({m for p in folds.values() for m in p.get('models', {})})
        for fold in range(N_FOLDS):
            p = folds.get(fold)
            if p is None:
                missing(f'{ds} fold {fold}: not in {source}; all models')
                continue
            for model in models:
                md = (p.get('models') or {}).get(model)
                if md is None:
                    missing(f'{ds} fold {fold} {model}: model absent from {source}')
                    continue
                by_slot = {h.get('slot'): h for h in md.get('hybrids') or []}
                for slot in slots:
                    h = by_slot.get(slot)
                    if h is None:
                        missing(f'{ds} fold {fold} {model} {slot}: no such hybrid '
                                f'(search_failed={p.get("search_failed")}, '
                                f'shortfall={p.get("shortfall")})')
                        continue
                    alloc = h.get('allocation_built')
                    total = h.get('n_features')
                    if not alloc or total is None:
                        missing(f'{ds} fold {fold} {model} {slot}: no allocation_built '
                                f'or n_features in {source}')
                        continue
                    if sum(int(v) for v in alloc.values()) != int(total):
                        missing(f'{ds} fold {fold} {model} {slot}: allocation_built sums '
                                f'to {sum(alloc.values())} features, n_features says '
                                f'{total}; segments not emitted')
                        continue
                    for rep, n in alloc.items():
                        rows.append({
                            'dataset': ds, 'model': model, 'fold': fold,
                            'slot': slot, 'hybrid_label': h.get('label'),
                            'representation': rep, 'n_features': int(n),
                            'total_features': int(total), 'source': source,
                        })
    return rows


def load_f3_detail(results_root, dataset, missing) -> list:
    """Named feature groups of one dataset's hybrid. Not buildable from disk."""
    missing(
        f'{dataset}: the per-feature indices of a hybrid are not stored in any '
        f'result file. c22_explainability and c24_blueprint_transfer keep only '
        f'the count of features per representation; the search cache keeps a '
        f'hash of the indices. c22 stores names for its 30 highest-SHAP features '
        f'(tree models only; the one small-molecule dataset with them is herg), '
        f'and for herg those are unnamed ("mordred dim 1553"). Getting the '
        f'full list means re-fitting the blueprint (tests/analysis/blueprint_trace.py), '
        f'which is compute.')
    return []


if __name__ == '__main__':
    REPO = pathlib.Path.home() / 'repos' / 'KIRBy'
    root = pathlib.Path('/Volumes/seagate/kirby_results_2026-10-08')
    miss = []
    rows = load_f3(root, miss.append)
    detail = load_f3_detail(root, 'herg', miss.append)
    bars = sorted({(r['dataset'], r['model'], r['fold']) for r in rows})
    dsets = sorted({r['dataset'] for r in rows})
    print(f'segments: {len(rows)}  bars (dataset, model, fold): {len(bars)}  '
          f'datasets: {len(dsets)}')
    for ds in dsets:
        ms = sorted({(r['model']) for r in rows if r['dataset'] == ds})
        nf = sorted({r['fold'] for r in rows if r['dataset'] == ds})
        print(f'  {ds}: models {ms}, folds {nf}')
    print('\nfold 0 segments, first model per dataset:')
    for ds in dsets:
        sub = [r for r in rows if r['dataset'] == ds and r['fold'] == 0]
        if not sub:
            continue
        m = sub[0]['model']
        sub = [r for r in sub if r['model'] == m]
        segs = ', '.join(f"{r['representation']} {r['n_features']}" for r in sub)
        print(f"  {ds} {m} [{sub[0]['hybrid_label']}] total {sub[0]['total_features']}: {segs}")
    print(f'\nmissing ({len(miss)}):')
    for m in miss:
        print('  ' + m)
