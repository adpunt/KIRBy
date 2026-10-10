"""Loaders for thesis figures F2a and F2b (THESIS_OUTLINE.md section 5.5).

F2a: best number of features (hybrid budget) per dataset x downstream model.
F2b: best number of base representations per dataset x downstream model.

Read-only. Paths are built from results_root only:
    <results_root>/complexity/<experiment>/<dataset>.json
and, where the final file is absent, <dataset>.partial.jsonl decoded with
tests/experiments/_checkpoint.decode.

F2a sources (load_f2a(..., source=...)):
  'c09' (default). c09 stores no best budget. For each decided model it stores
        the rank-1 hybrid's held-out score at every decided budget
        (bias_variance.curves['<model>|hybrid1'].by_budget[i].val_score_mean,
        the mean over outer folds). best = argmin (rmse) / argmax (others) of
        that stored mean over c09's own budget list. Both decided models.
  'c16' c16 stores its own best:
        families[<arm>].comparison.simple_hybrid.budget, the budget with the
        best fold-mean held-out score at the default regularisation setting
        (svm C = 1, rf no depth limit), over the budgets below capped_full.
        Only svm and rf are run, so only those models get a mark.
  c06 is not offered: it stores an optimal budget per capacity setting
  (4 settings per model family) per outer fold, picked on the training-row
  score, so it has no single value per dataset x model.

F2b source: c18. c18 stores, per model, per budget and per outer fold, the
  rep_count of the search's best hybrid ('natural' arm, recipe_per_fold[k]
  .rep_count; None = no limit). It stores no single best. The loader takes the
  budget at which 'natural' has the best stored fold-mean held-out score
  (per_model[m].per_budget[b].natural.score) and returns the per-fold
  rep_counts there. best_value is set only when every fold picked the same
  count. Representation of the count:
      best_value=int,  best_label='<int>'        all folds picked that count
      best_value=None, best_label='no limit'     all folds picked no limit
      best_value=None, best_label='folds disagree'
  per_fold_labels always carries the five fold values as strings.
"""
import ast
import json
import math
import pathlib
import sys

INPUTS_F2A = ['complexity/c09_formal_analysis',
              'complexity/c16_feature_selection_as_regularization',
              'complexity/c02_complexity_matching_grid',
              'complexity/c17_information_theoretic_decomposition']
INPUTS_F2B = ['complexity/c18_diversity_vs_depth']

C09 = 'complexity/c09_formal_analysis'
C16 = 'complexity/c16_feature_selection_as_regularization'
C02 = 'complexity/c02_complexity_matching_grid'
C17 = 'complexity/c17_information_theoretic_decomposition'
C18 = 'complexity/c18_diversity_vs_depth'

# herg_ic50 is in decided_grid.json but has no complexity stage
# (orchestrate/plan.py docstring: "herg_ic50's row has no complexity
# experiments").
NO_COMPLEXITY_STAGE = ('herg_ic50',)
# c16 sweeps a regularisation setting for these models only.
C16_MODELS = ('svm', 'rf')


def _repo_root():
    # Final location: <repo>/tests/analysis/<this file>.
    return pathlib.Path(__file__).resolve().parents[2]


def _import_decode(repo):
    p = str(repo / 'tests' / 'experiments')
    if p not in sys.path:
        sys.path.insert(0, p)
    from _checkpoint import decode
    return decode


def _cut_datasets(repo):
    """CUT_DATASETS + EXCLUDED_DATASETS from orchestrate/plan.py, read with
    ast so plan.py's imports are not executed."""
    tree = ast.parse((repo / 'tests' / 'experiments' / 'orchestrate'
                      / 'plan.py').read_text())
    out = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None)
                in ('CUT_DATASETS', 'EXCLUDED_DATASETS')):
            out |= set(ast.literal_eval(node.value))
    if not out:
        raise RuntimeError('CUT_DATASETS / EXCLUDED_DATASETS not found')
    return out


def _decided(repo):
    return json.loads((repo / 'tests' / 'experiments'
                       / 'decided_grid.json').read_text())['datasets']


def _roster(repo):
    """(dataset, [decided general models]) for every dataset not cut."""
    cut = _cut_datasets(repo)
    dec = _decided(repo)
    out = []
    for ds in sorted(dec):
        if ds in cut or ds in NO_COMPLEXITY_STAGE:
            continue
        models = [m for m in dec[ds]['general']['models'] if m != 'knn']
        out.append((ds, models))
    return out


def _read_partial(path, decode):
    """(identity header, [per-fold payloads sorted by fold])."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    head = lines[0] if lines and 'identity' in lines[0] else {}
    folds = [decode(l['payload']) for l in lines
             if isinstance(l, dict) and 'payload' in l]
    folds.sort(key=lambda p: p.get('fold', 0))
    return head.get('identity') or {}, folds


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _pick(pairs, lower_is_better):
    """pairs: [(value, score)]. Returns (value, score, tied_values)."""
    ok = [(v, s) for v, s in pairs if _finite(s)]
    if not ok:
        return None, None, []
    best_s = min(s for _, s in ok) if lower_is_better else max(s for _, s in ok)
    tied = [v for v, s in ok if s == best_s]
    return tied[0], best_s, tied[1:]


def _mean(xs):
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else None


# ── F2a ────────────────────────────────────────────────────────────────────

def _f2a_c09(results_root, missing, repo, roster, decode):
    rows = []
    d = results_root / C09
    for ds, models in roster:
        fj, fp = d / f'{ds}.json', d / f'{ds}.partial.jsonl'
        if fj.exists():
            j = json.loads(fj.read_text())
            metric = (j.get('metric_name') or j.get('metric')).lower()
            lib = bool(j['lower_is_better'])
            curves = ((j.get('bias_variance') or {}).get('curves') or {})
            for m in models:
                c = curves.get(f'{m}|hybrid1')
                if not c:
                    why = ('dropped by c09: ' + str(j.get('models_dropped'))
                           if m not in (j.get('models') or []) else
                           'no bias_variance curve for hybrid1')
                    missing(f'F2a c09 {ds} {m}: {why}')
                    continue
                bb = c['by_budget']
                pairs = [(int(b['budget']), b.get('val_score_mean')) for b in bb]
                nf = {int(b['budget']): sum(1 for f in b.get('by_fold') or []
                                            if _finite(f.get('val_score')))
                      for b in bb}
                best, sc, tied = _pick(pairs, lib)
                if best is None:
                    missing(f'F2a c09 {ds} {m}: no finite held-out score '
                            f'at any budget')
                    continue
                i = [p[0] for p in pairs].index(best)
                rows.append({
                    'dataset': ds, 'model': m, 'metric': metric,
                    'lower_is_better': lib,
                    'grid': [p[0] for p in pairs],
                    'best_value': best, 'best_score': sc,
                    'n_folds': nf[best],
                    'n_folds_by_budget': [nf[p[0]] for p in pairs],
                    'score_by_budget': [p[1] for p in pairs],
                    'ties': tied,
                    'best_is': 'read off: argbest of stored fold-mean '
                               'held-out score over the budget list',
                    'source': f"{C09}/{ds}.json :: bias_variance.curves"
                              f"['{m}|hybrid1'].by_budget[{i}].val_score_mean",
                })
        elif fp.exists():
            ident, folds = _read_partial(fp, decode)
            if not folds:
                for m in models:
                    missing(f'F2a c09 {ds} {m}: partial file holds no fold')
                continue
            metric = folds[0]['metric'].lower()
            lib = bool(folds[0]['lower_is_better'])
            for m in models:
                per_b = {}
                for p in folds:
                    for bb in p.get('by_budget') or []:
                        h = next((h for h in ((bb.get('models') or {}).get(m)
                                              or {}).get('hybrids') or []
                                  if h.get('slot') == 'hybrid1'), None)
                        per_b.setdefault(int(bb['budget']), []).append(
                            h.get('val_score') if h else None)
                if not per_b:
                    missing(f'F2a c09 {ds} {m}: partial file has no hybrid1')
                    continue
                grid = sorted(per_b)
                pairs = [(b, _mean(per_b[b])) for b in grid]
                best, sc, tied = _pick(pairs, lib)
                if best is None:
                    missing(f'F2a c09 {ds} {m}: no finite held-out score')
                    continue
                nfb = [sum(1 for x in per_b[b] if _finite(x)) for b in grid]
                rows.append({
                    'dataset': ds, 'model': m, 'metric': metric,
                    'lower_is_better': lib, 'grid': grid,
                    'best_value': best, 'best_score': sc,
                    'n_folds': nfb[grid.index(best)],
                    'n_folds_by_budget': nfb,
                    'score_by_budget': [p[1] for p in pairs], 'ties': tied,
                    'best_is': 'read off: argbest of the mean over the '
                               'partial file\'s folds (run not finished)',
                    'source': f"{C09}/{ds}.partial.jsonl :: payload.by_budget"
                              f"[*].models.{m}.hybrids[slot=hybrid1].val_score",
                })
        else:
            for m in models:
                missing(f'F2a c09 {ds} {m}: no c09 result file')
    return rows


def _f2a_c16(results_root, missing, repo, roster, decode):
    rows = []
    d = results_root / C16
    for ds, models in roster:
        want = [m for m in models if m in C16_MODELS]
        if not want:
            continue                 # c16 has no sweep for these models
        fj = d / f'{ds}.json'
        if not fj.exists():
            extra = (' (partial file only, run not finished)'
                     if (d / f'{ds}.partial.jsonl').exists() else '')
            for m in want:
                missing(f'F2a c16 {ds} {m}: no c16 result file{extra}')
            continue
        j = json.loads(fj.read_text())
        metric = (j.get('metric_name') or j.get('metric')).lower()
        lib = bool(j['lower_is_better'])
        fams = j.get('families') or {}
        for m in want:
            arm = next((a for a, v in fams.items()
                        if v.get('recipe_downstream_model') == m
                        and v.get('hybrid_slot') == 1), None)
            if arm is None:
                missing(f'F2a c16 {ds} {m}: no hybrid-1 arm in families '
                        f'(families_not_run={j.get("families_not_run")}, '
                        f'skipped={j.get("skipped")})')
                continue
            v = fams[arm]
            sh = (v.get('comparison') or {}).get('simple_hybrid') or {}
            b = sh.get('budget')
            if b is None or not _finite(sh.get('score')):
                missing(f'F2a c16 {ds} {m}: comparison.simple_hybrid has no '
                        f'budget/score ({v["comparison"].get("hybrid_wins_reason")})')
                continue
            cap = int(j['capped_full'])
            hm = v['heatmap']
            dp = sh.get('default_param')
            key = str(dp)
            grid = sorted(int(x) for x in hm if int(x) < cap)
            cell = hm.get(str(b), {}).get(key, {})
            rows.append({
                'dataset': ds, 'model': m, 'metric': metric,
                'lower_is_better': lib, 'grid': grid,
                'best_value': int(b), 'best_score': sh['score'],
                'n_folds': cell.get('n_folds_scored'),
                'score_by_budget': [hm[str(x)].get(key, {}).get('val')
                                    for x in grid],
                'default_param': f"{v['param_name']}={dp}",
                'capped_full_excluded': cap,
                'best_is': 'stored by c16',
                'source': f"{C16}/{ds}.json :: families['{arm}']"
                          f".comparison.simple_hybrid.budget",
            })
    return rows


def load_f2a(results_root: pathlib.Path, missing, source='c09') -> list:
    results_root = pathlib.Path(results_root)
    repo = _repo_root()
    roster = _roster(repo)
    decode = _import_decode(repo)
    if source == 'c09':
        return _f2a_c09(results_root, missing, repo, roster, decode)
    if source == 'c16':
        return _f2a_c16(results_root, missing, repo, roster, decode)
    raise ValueError(f"source must be 'c09' or 'c16', not {source!r}")


def load_f2a_picks(results_root: pathlib.Path, missing) -> list:
    """The budget each outer fold's search picked on that fold's training rows (c02).

    One row per dataset x decided model x outer fold:
        per_fold[k].standard_selection[<model>].hybrids[slot=hybrid1].budget
    (the rank-1 hybrid of the fold's own search; n_features is the width it
    built). Final .json only: c02 has a final file for every dataset it ran.
    """
    results_root = pathlib.Path(results_root)
    rows = []
    d = results_root / C02
    for ds, models in _roster(_repo_root()):
        f = d / f'{ds}.json'
        if not f.exists():
            for m in models:
                missing(f'F2a c02 {ds} {m}: no c02 result file')
            continue
        j = json.loads(f.read_text())
        for m in models:
            got = 0
            for p in j.get('per_fold') or []:
                ss = (p.get('standard_selection') or {}).get(m) or {}
                h = next((h for h in ss.get('hybrids') or []
                          if h.get('slot') == 'hybrid1'), None)
                if not h or h.get('budget') is None:
                    continue
                got += 1
                rows.append({
                    'dataset': ds, 'model': m, 'fold': int(p['fold']),
                    'picked_budget': int(h['budget']),
                    'n_features': h.get('n_features'),
                    'train_score': h.get('train_score'),
                    'grid': j.get('budgets'),
                    'source': f"{C02}/{ds}.json :: per_fold[{p['fold']}]"
                              f".standard_selection.{m}.hybrids[slot=hybrid1].budget",
                })
            if not got:
                missing(f'F2a c02 {ds} {m}: no fold has a hybrid1 pick')
    return rows


C17_METHODS = ('rfe', 'random')


def load_f2a_c17(results_root: pathlib.Path, missing) -> list:
    """Recursive feature elimination and random features at the same budgets (c17).

    One row per dataset x decided model x method x budget:
        per_budget[<budget>][<model>][<method>].downstream_score  (mean over folds)
        per_budget[<budget>][<model>][<method>].fold_scores
    rfe: sklearn RFE with a random forest on the fold's training rows, over all
    base-representation features concatenated. random: one draw of that many
    features (seed 42). c17 runs regression datasets only.
    """
    results_root = pathlib.Path(results_root)
    rows = []
    d = results_root / C17
    for ds, models in _roster(_repo_root()):
        f = d / f'{ds}.json'
        if not f.exists():
            missing(f'F2a c17 {ds}: no c17 result file')
            continue
        j = json.loads(f.read_text())
        pb = j.get('per_budget') or {}
        if not pb:
            missing(f"F2a c17 {ds}: no per_budget ({j.get('reason') or 'skipped'})")
            continue
        for m in models:
            for meth in C17_METHODS:
                for b in sorted(pb, key=int):
                    c = (pb[b].get(m) or {}).get(meth)
                    if not c or not _finite(c.get('downstream_score')):
                        continue
                    fs = [x for x in c.get('fold_scores') or [] if _finite(x)]
                    rows.append({
                        'dataset': ds, 'model': m, 'method': meth, 'budget': int(b),
                        'score': c['downstream_score'], 'fold_scores': fs,
                        'n_folds': len(fs),
                        'source': f"{C17}/{ds}.json :: per_budget.{b}.{m}.{meth}"
                                  f".downstream_score",
                    })
    return rows


# ── F2b ────────────────────────────────────────────────────────────────────

def _label(rc):
    return 'no limit' if rc is None else str(int(rc))


def load_f2b(results_root: pathlib.Path, missing) -> list:
    results_root = pathlib.Path(results_root)
    repo = _repo_root()
    roster = _roster(repo)
    decode = _import_decode(repo)
    dec = _decided(repo)
    d = results_root / C18
    rows = []
    for ds, models in roster:
        fj, fp = d / f'{ds}.json', d / f'{ds}.partial.jsonl'
        ident, pfolds = (_read_partial(fp, decode) if fp.exists()
                         else ({}, []))
        run_grid = (ident.get('config') or {}).get('rep_counts')
        grid_now = dec[ds]['general']['rep_counts']
        grid, grid_src = ((run_grid, f'{C18}/{ds}.partial.jsonl :: '
                                     f'identity.config.rep_counts')
                          if run_grid is not None else
                          (grid_now, 'decided_grid.json (run grid not '
                                     'stored)'))
        if fj.exists():
            j = json.loads(fj.read_text())
            metric = (j.get('metric_name') or j.get('metric')).lower()
            lib = bool(j['lower_is_better'])
            per_model = j.get('per_model') or {}
            kind = 'json'
        elif pfolds:
            metric = pfolds[0]['metric'].lower()
            lib = bool(pfolds[0]['lower_is_better'])
            per_model = None
            kind = 'partial'
        else:
            for m in models:
                missing(f'F2b c18 {ds} {m}: no c18 result file')
            continue
        for m in models:
            # budget -> (fold-mean held-out score, [(fold, rep_count)])
            by_b = {}
            if kind == 'json':
                pb = (per_model.get(m) or {}).get('per_budget') or {}
                for b, arms in pb.items():
                    nat = arms.get('natural') or {}
                    rpf = nat.get('recipe_per_fold') or []
                    by_b[int(b)] = (nat.get('score'),
                                    list(zip(nat.get('folds') or [],
                                             [r.get('rep_count') for r in rpf])))
            else:
                acc = {}
                for p in pfolds:
                    pbud = ((p.get('models') or {}).get(m) or {}).get(
                        'per_budget') or {}
                    for b, brec in pbud.items():
                        nat = (brec.get('arms') or {}).get('natural') or {}
                        if nat.get('heldout_score') is None:
                            continue
                        acc.setdefault(int(b), []).append(
                            (p['fold'], nat['heldout_score'],
                             nat.get('rep_count')))
                for b, xs in acc.items():
                    by_b[b] = (_mean([x[1] for x in xs]),
                               [(x[0], x[2]) for x in xs])
            if not by_b:
                missing(f'F2b c18 {ds} {m}: no natural arm scored '
                        f'(model not in c18 models {j.get("models") if kind == "json" else "?"})')
                continue
            budgets = sorted(by_b)
            bb, sc, tied = _pick([(b, by_b[b][0]) for b in budgets], lib)
            if bb is None:
                missing(f'F2b c18 {ds} {m}: natural arm has no finite score')
                continue
            fold_rc = by_b[bb][1]
            labels = [_label(rc) for _, rc in fold_rc]
            agree = len(set(labels)) == 1
            if agree and fold_rc[0][1] is not None:
                bv, bl = int(fold_rc[0][1]), labels[0]
            elif agree:
                bv, bl = None, 'no limit'
            else:
                bv, bl = None, 'folds disagree'
            counts = {}
            for l in labels:
                counts[l] = counts.get(l, 0) + 1
            rows.append({
                'dataset': ds, 'model': m, 'metric': metric,
                'lower_is_better': lib,
                'grid': grid, 'grid_source': grid_src,
                'grid_decided_now': grid_now,
                'grid_single_value': len(grid) == 1,
                'best_value': bv, 'best_label': bl, 'folds_agree': agree,
                'per_fold_labels': labels,
                'per_fold_folds': [f for f, _ in fold_rc],
                'label_counts': counts,
                'at_budget': bb, 'budget_ties': tied,
                'best_score': sc,
                'n_folds': len(fold_rc),
                'best_is': ('stored per fold by c18 (search pick on training '
                            'rows); budget = argbest of stored natural fold-'
                            'mean held-out score'),
                'source': (f"{C18}/{ds}.json :: per_model.{m}.per_budget"
                           f"['{bb}'].natural.recipe_per_fold[*].rep_count"
                           if kind == 'json' else
                           f"{C18}/{ds}.partial.jsonl :: payload.models.{m}"
                           f".per_budget['{bb}'].arms.natural.rep_count"),
            })
    return rows


if __name__ == '__main__':
    root = pathlib.Path('/Volumes/seagate/kirby_results_2026-10-08')
    # Hard-coded repo only here: this scratch copy is not yet under tests/.
    _REPO = pathlib.Path.home() / 'repos' / 'KIRBy'
    _repo_root = lambda: _REPO  # noqa: E731

    def run(name, fn, **kw):
        miss = []
        rows = fn(root, miss.append, **kw)
        print(f'\n===== {name}: {len(rows)} rows, {len(miss)} missing')
        for r in rows:
            if 'best_label' in r:
                print(f"  {r['dataset']:24s} {r['model']:9s} "
                      f"{r['best_label']:14s} folds={r['per_fold_labels']} "
                      f"@b{r['at_budget']} {r['metric']}={r['best_score']:.4f} "
                      f"grid={r['grid']} now={r['grid_decided_now']}")
            else:
                print(f"  {r['dataset']:24s} {r['model']:9s} best={r['best_value']:<5d} "
                      f"{r['metric']}={r['best_score']:.4f} n_folds={r['n_folds']} "
                      f"grid={r['grid']} scores="
                      f"{[None if s is None else round(s, 4) for s in r['score_by_budget']]}"
                      + (f" TIES={r['ties']}" if r.get('ties') else ''))
        for m in miss:
            print('  MISSING:', m)
        return rows

    a9 = run('F2a source=c09', load_f2a, source='c09')
    a16 = run('F2a source=c16', load_f2a, source='c16')
    run('F2b c18', load_f2b)
    # c09 vs c16 where both have a mark
    k9 = {(r['dataset'], r['model']): r['best_value'] for r in a9}
    print('\n===== c09 vs c16 best budget (features), same dataset x model')
    for r in a16:
        k = (r['dataset'], r['model'])
        print(f"  {k[0]:24s} {k[1]:4s} c09={k9.get(k)} c16={r['best_value']} "
              f"{'same' if k9.get(k) == r['best_value'] else 'DIFFER'}")
