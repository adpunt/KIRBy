"""Loaders for thesis figures F5 and F6 (similarity chapter, outline 6.3, 6.4).

F5: Spearman correlation between pairwise distance and pairwise activity
    difference over held-out molecules, best single representation (x) against
    hybrid (y). Source: s02_hybrid_similarity.
    Setting per dataset: no budget limit (the hybrid each fold's search chose
    over every budget, the top-level `comparisons` of each fold), unweighted,
    the first distance of the dataset's kNN group (knn_axes.metrics[0]; it is
    manhattan on every file, the same rule s05 uses for its fixed setting).
    Value per side = mean over the folds whose comparison row has a delta
    (the folds s02's own `summary` averages).
    hybrid_kind 'distance' is knn's distance hybrid against knn's rank-1 single
    representation (the outline's mark). Extra rows: 'predictive:<model>'
    (that model's hybrid 1 and 2 against that model's rank-1 single
    representation) and 'similarity' (similarity-objective hybrid against
    knn's rank-1).

F6: score of predicting each held-out molecule from its nearest training
    molecules, in the dataset's own metric. Source: s05_knn_prediction
    summary.fixed_ref (the comparison fixed before the run: distance hybrid
    against knn's rank-1 single representation, unweighted, first distance,
    k=5, rank counting). Values are the stored cell means. Extra rows at the
    same setting from `cells`: similarity hybrid and each model's hybrid 1, 2
    as feature spaces for the neighbours.

Read-only; stdlib + numpy. Paths come from results_root only.
"""
import ast
import glob
import importlib.util
import json
import pathlib
from collections import Counter

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

S02_DIR = 'similarity/s02_hybrid_similarity'
S05_DIR = 'similarity/s05_knn_prediction'
INPUTS_F5 = [S02_DIR + '/*.json', S02_DIR + '/*.partial.jsonl']
INPUTS_F6 = [S05_DIR + '/*.json', S05_DIR + '/*.partial.jsonl']


# ---------------------------------------------------------------- helpers

def _cut_datasets():
    """CUT_DATASETS + EXCLUDED_DATASETS read from plan.py without importing it."""
    src = (REPO_ROOT / 'tests/experiments/orchestrate/plan.py').read_text()
    out = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in ('CUT_DATASETS', 'EXCLUDED_DATASETS'):
                out |= set(ast.literal_eval(node.value))
    return out


def _decode_fn():
    p = REPO_ROOT / 'tests/experiments/_checkpoint.py'
    spec = importlib.util.spec_from_file_location('_kirby_checkpoint', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode


def _expected(results_root, exp, files_present, cut):
    """Datasets the ledger plans `exp` on (not cut), plus any with a file."""
    exp_ds = set(files_present)
    lp = results_root / 'ledger.json'
    status = {}
    if lp.exists():
        rows = json.loads(lp.read_text()).get('rows', {})
        for ds, r in rows.items():
            for s in r.get('stages', []):
                u = s.get('units', {}).get(exp)
                if u is not None:
                    exp_ds.add(ds)
                    status[ds] = u.get('status')
    return sorted(d for d in exp_ds if d not in cut), status


def _read_partial(path, decode):
    """{fold index: payload} from a .partial.jsonl sidecar."""
    out = {}
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        o = decode(json.loads(line))
        p = o.get('payload')
        if isinstance(p, dict) and 'fold' in p:
            out[int(p['fold'])] = p
    return out


def _fin(x):
    return x is not None and np.isfinite(x)


def _mean(xs):
    xs = [x for x in xs if _fin(x)]
    return (float(np.mean(xs)), len(xs)) if xs else (None, 0)


def _one_or_list(xs):
    c = Counter(x for x in xs if x is not None)
    if not c:
        return None
    return next(iter(c)) if len(c) == 1 else [x for x in xs]


# ---------------------------------------------------------------- F5 (s02)

def _s02_recipe(sel, arm):
    """Recipe label of a hybrid arm key from one fold's `selection`."""
    if not sel:
        return None
    if arm == 'knn/distance_hybrid':
        return (sel.get('distance_hybrid') or {}).get('label')
    if arm == 'similarity_hybrid':
        return (sel.get('similarity_hybrid') or {}).get('label')
    m, _, h = arm.partition('/hybrid')
    hy = ((sel.get('models') or {}).get(m) or {}).get('hybrids') or []
    i = int(h) - 1 if h.isdigit() else -1
    return hy[i].get('label') if 0 <= i < len(hy) else None


def load_f5(results_root: pathlib.Path, missing) -> list[dict]:
    results_root = pathlib.Path(results_root)
    cut = _cut_datasets()
    decode = None
    d = results_root / S02_DIR
    files = {pathlib.Path(p).name[:-5]: pathlib.Path(p)
             for p in glob.glob(str(d / '*.json'))}
    partials = {pathlib.Path(p).name[:-len('.partial.jsonl')]: pathlib.Path(p)
                for p in glob.glob(str(d / '*.partial.jsonl'))}
    datasets, status = _expected(results_root, 's02', set(files) | set(partials), cut)
    rows = []
    for ds in datasets:
        if ds in files:
            j = json.loads(files[ds].read_text())
            if j.get('skipped') or not j.get('has_computed_results', True):
                missing(f'F5 {ds}: s02 file is marked skipped / no computed results '
                        f'({j.get("skip_reason") or j.get("result_status")})')
                continue
            folds = j.get('folds') or []
            dist = (j.get('knn_axes') or {}).get('metrics', [None])[0]
            models = j.get('models') or []
            rel = f'{S02_DIR}/{ds}.json'
            metric = (j.get('metric_name') or 'spearman').lower()
            n_outer = j.get('n_outer_folds', len(folds))
        elif ds in partials:
            decode = decode or _decode_fn()
            pf = _read_partial(partials[ds], decode)
            if not pf:
                missing(f'F5 {ds}: only a partial checkpoint, with no fold payloads')
                continue
            n_outer = max(p.get('n_outer_folds', 0) for p in pf.values()) or (max(pf) + 1)
            folds = [pf.get(i, {}) for i in range(n_outer)]
            # first distance = first distance in any comparison row (axes order)
            dist = None
            for f in folds:
                for c in (f.get('comparisons') or {}).values():
                    if c.get('rows'):
                        dist = c['rows'][0]['distance']
                        break
                if dist:
                    break
            models = list(((folds[min(pf)].get('selection') or {}).get('models') or {}))
            rel = f'{S02_DIR}/{ds}.partial.jsonl'
            metric = 'spearman'
        else:
            missing(f'F5 {ds}: no s02 result file (ledger status '
                    f'{status.get(ds, "unknown")}); all marks missing')
            continue

        expected = [('knn distance hybrid - knn base rank 1', 'distance', 'knn', None),
                    ('similarity hybrid - knn base rank 1', 'similarity', 'knn', None)]
        for m in models:
            for h in (1, 2):
                expected.append((f'{m} hybrid {h} - {m} base rank 1',
                                 f'predictive:{m}', m, h))
        for name, kind, single_model, hrank in expected:
            hs, ss, sreps, arms, recipes, budgets = [], [], [], [], [], []
            pf_h, pf_s = [], []
            for f in folds:
                c = (f.get('comparisons') or {}).get(name)
                row = None
                if c:
                    for r in c.get('rows', []):
                        if (r.get('distance') == dist
                                and r.get('hybrid_weighting') == 'unweighted'
                                and r.get('single_weighting', 'unweighted') == 'unweighted'):
                            row = r
                            break
                if row is None or row.get('delta') is None:
                    pf_h.append(None); pf_s.append(None)
                    sreps.append(c.get('single_rep') if c else None)
                    recipes.append(None)
                    continue
                pf_h.append(row['hybrid_sac_rho']); pf_s.append(row['single_sac_rho'])
                hs.append(row['hybrid_sac_rho']); ss.append(row['single_sac_rho'])
                sreps.append(c.get('single_rep'))
                arms.append(c.get('hybrid_arm'))
                rec = _s02_recipe(f.get('selection'), c.get('hybrid_arm'))
                recipes.append(rec)
            h, n = _mean(hs)
            s, _ = _mean(ss)
            if h is None or s is None:
                why = ''
                if kind == 'similarity':
                    reasons = {str(v) for f in folds for k, v in
                               ((f.get('selection') or {}).get('unavailable') or {}).items()
                               if 'similarity' in k}
                    why = '; stored reason: ' + ' | '.join(sorted(reasons)) if reasons else ''
                missing(f'F5 {ds}: no {kind} mark ("{name}", {dist}, unweighted, '
                        f'no budget limit) on any fold{why}')
                continue
            rows.append({
                'dataset': ds,
                'metric': metric,
                'lower_is_better': False,
                'single_rep': _one_or_list(sreps),
                'single_model': single_model,
                'single_value': s,
                'hybrid_label': _one_or_list(arms),
                'hybrid_kind': kind,
                'hybrid_rank': hrank,
                'hybrid_recipes_per_fold': recipes,
                'hybrid_value': h,
                'primary': kind == 'distance',
                'setting': {'distance': dist, 'feature_limit': None,
                            'weighting': 'unweighted',
                            'n_sample_pairs': j.get('n_sample_pairs') if ds in files else None},
                'n_folds': n,
                'n_outer_folds': n_outer,
                'per_fold_single': pf_s,
                'per_fold_hybrid': pf_h,
                'source': f'{rel} :: folds[*].comparisons["{name}"].rows'
                          f'[distance={dist}, hybrid_weighting=unweighted]'
                          '.{single_sac_rho,hybrid_sac_rho}',
            })
    return rows


# ---------------------------------------------------------------- F6 (s05)

FIXED = {'condition': 'unweighted', 'k': 5, 'scheme': 'rank'}   # s05 FIXED_*


def _s05_cells_from_partial(pf):
    """cells dict {name: {'fold_scores': [...]}} from per-fold arm rows."""
    n = max(pf) + 1
    cells = {}
    for i, p in pf.items():
        for arm, a in (p.get('arms') or {}).items():
            for cond, cd in (a.get('conditions') or {}).items():
                for r in cd.get('rows', []):
                    key = f'{arm}|{cond}|{r["distance"]}|k{r["k"]}|{r["scheme"]}'
                    c = cells.setdefault(key, {'arm': arm, 'fold_scores': [None] * n})
                    c['fold_scores'][i] = r.get('score')
    for c in cells.values():
        c['score_mean'], c['n_folds_scored'] = _mean(c['fold_scores'])
    return cells


def load_f6(results_root: pathlib.Path, missing) -> list[dict]:
    results_root = pathlib.Path(results_root)
    cut = _cut_datasets()
    decode = None
    d = results_root / S05_DIR
    files = {pathlib.Path(p).name[:-5]: pathlib.Path(p)
             for p in glob.glob(str(d / '*.json'))}
    partials = {pathlib.Path(p).name[:-len('.partial.jsonl')]: pathlib.Path(p)
                for p in glob.glob(str(d / '*.partial.jsonl'))}
    datasets, status = _expected(results_root, 's05', set(files) | set(partials), cut)
    rows = []
    for ds in datasets:
        fixed_ref = None
        if ds in files:
            j = json.loads(files[ds].read_text())
            if j.get('skipped') or not j.get('has_computed_results', True):
                missing(f'F6 {ds}: s05 file is marked skipped / no computed results '
                        f'({j.get("skip_reason") or j.get("result_status")})')
                continue
            rel = f'{S05_DIR}/{ds}.json'
            metric = j['metric'].lower()
            low = bool(j['lower_is_better'])
            dist = j['knn_settings']['metrics'][0]
            cells = j.get('cells') or {}
            pf = {int(k): v for k, v in (j.get('per_fold') or {}).items()}
            models = j.get('models') or []
            fixed_ref = (j.get('summary') or {}).get('fixed_ref')
            reasons = j.get('arm_reasons_missing') or {}
            n_outer = j.get('n_outer_folds', len(pf))
        elif ds in partials:
            decode = decode or _decode_fn()
            pf = _read_partial(partials[ds], decode)
            if not pf:
                missing(f'F6 {ds}: only a partial checkpoint, with no fold payloads')
                continue
            p0 = pf[min(pf)]
            rel = f'{S05_DIR}/{ds}.partial.jsonl'
            metric = str(p0.get('metric')).lower()
            low = bool(p0.get('lower_is_better'))
            rows0 = (((p0.get('arms') or {}).get('single_knn1') or {})
                     .get('conditions', {}).get('unweighted', {}).get('rows') or [{}])
            dist = rows0[0].get('distance')
            cells = _s05_cells_from_partial(pf)
            models = list(p0.get('models') or {})
            reasons = {}
            n_outer = max(pf) + 1
        else:
            missing(f'F6 {ds}: no s05 result file (ledger status '
                    f'{status.get(ds, "unknown")}); all marks missing')
            continue

        tail = f'|{FIXED["condition"]}|{dist}|k{FIXED["k"]}|{FIXED["scheme"]}'
        single_cell = cells.get('single_knn1' + tail)
        single_reps = [((pf.get(i) or {}).get('knn_base') or [None])[0]
                       for i in range(n_outer)]
        expected = [('distance_hybrid', 'distance')]
        expected += [('similarity_hybrid', 'similarity')]
        expected += [(f'general_{m}_{h}', f'predictive:{m}') for m in models for h in (1, 2)]
        for arm, kind in expected:
            hc = cells.get(arm + tail)
            if arm == 'distance_hybrid' and fixed_ref:
                h, s = fixed_ref['hybrid_score_mean'], fixed_ref['single_score_mean']
                n = fixed_ref['n_paired_folds']
                src = f'{rel} :: summary.fixed_ref.{{single_score_mean,hybrid_score_mean}}'
            else:
                if not hc or not single_cell or not _fin(hc.get('score_mean')) \
                        or not _fin(single_cell.get('score_mean')):
                    why = '; '.join(map(str, reasons.get(arm, []))) or 'cell not scored'
                    missing(f'F6 {ds}: no {kind} mark ({arm}{tail}): {why}')
                    continue
                # paired: both sides averaged over the folds where both scored
                pairs = [(a, b) for a, b in zip(hc['fold_scores'], single_cell['fold_scores'])
                         if _fin(a) and _fin(b)]
                if not pairs:
                    missing(f'F6 {ds}: no {kind} mark ({arm}{tail}): no fold scores both sides')
                    continue
                h = float(np.mean([a for a, _ in pairs]))
                s = float(np.mean([b for _, b in pairs]))
                n = len(pairs)
                src = (f'{rel} :: cells["{arm}{tail}"].fold_scores vs '
                       f'cells["single_knn1{tail}"].fold_scores, mean over paired folds')
            budgets = None
            if arm in ('distance_hybrid', 'similarity_hybrid'):
                budgets = [((pf.get(i) or {}).get(arm) or {}).get('budget_features')
                           for i in range(n_outer)]
            rows.append({
                'dataset': ds,
                'metric': metric,
                'lower_is_better': low,
                'model': 'knn',
                'single_rep': _one_or_list(single_reps),
                'single_label': 'single_knn1',
                'single_value': s,
                'hybrid_label': arm,
                'hybrid_kind': kind,
                'hybrid_value': h,
                'primary': kind == 'distance',
                'setting': {'distance': dist, 'k': FIXED['k'], 'scheme': FIXED['scheme'],
                            'weighting': FIXED['condition'], 'feature_limit': None,
                            'hybrid_budget_per_fold': budgets},
                'n_folds': n,
                'n_outer_folds': n_outer,
                'per_fold_single': (single_cell or {}).get('fold_scores'),
                'per_fold_hybrid': (hc or {}).get('fold_scores'),
                'source': src,
            })
    return rows


# ---------------------------------------------------------------- main

if __name__ == '__main__':
    import sys
    REPO_ROOT = pathlib.Path('~/repos/KIRBy').expanduser()
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else '/Volumes/seagate/kirby_results_2026-10-08')
    for name, fn in (('F5', load_f5), ('F6', load_f6)):
        miss = []
        rows = fn(root, miss.append)
        print(f'\n===== {name}: {len(rows)} rows '
              f'({sum(r["primary"] for r in rows)} primary), {len(miss)} missing')
        for r in rows:
            sr = r['single_rep'] if isinstance(r['single_rep'], str) else '/'.join(map(str, r['single_rep'] or []))
            print(f'  {r["dataset"]:22s} {r["metric"]:11s} {r["hybrid_kind"]:16s} '
                  f'{r["hybrid_label"]!s:28s} {r["single_value"]:.4f} -> {r["hybrid_value"]:.4f} '
                  f'n={r["n_folds"]} single={sr}')
        for m in miss:
            print('  MISSING', m)
