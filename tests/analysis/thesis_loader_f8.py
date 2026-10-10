"""Loader for thesis figure F8, "did the cleaning work", rebuilt 2026-10-10 to follow Chapter 7.

One row of the figure = one signed-off DEEP method (THESIS_OUTLINE.md "CHAPTER 7 METHODS, SIGNED
OFF"), named by its action and what selects the molecules (THESIS_OUTLINE.md "NAMES FOR THE NOISE
METHODS"). Rows sit in three blocks, one file each:
  detector    noise mitigation, molecules selected by a detector
  flag        noise mitigation, molecules selected by a known flag (no every-molecule method is DEEP)
  training    noise-robust training (no detector; the model changes)
One column = one dataset as delivered (level 0), QM9 with no noise added, or one dataset under one
damage kind at its main level (1.0 of the clean-label spread; censoring 0.25; class flips 0.1).

One cell = the SHARE of model-and-configuration pairs (each decided model x base_1, base_2, hybrid,
hybrid__general2) whose held-out score after the method beat the score before it. Outer fold 0,
read from <results_root>/noise/ only. Scores are averaged over noise draws first.
- A method's settings (share flagged, strength, neighbour count, distance, ...) are collapsed: a pair
  counts as improved if ANY setting improved it, so the share leans optimistic (picked on held-out
  scores). The setting with the largest gain is kept in `pair_settings`.
- "Improved": after better than before once both are rounded to 4 decimals. A tie is not improved.
- Control, where the experiment ran one: for the pair's best setting, the method's score against the
  best of the control's seeds at the same setting (random deletion of the same number of molecules in
  n24; shuffled per-molecule lifts in n30). `n_control_beaten` of `n_control_pairs`.

Sources: the uniform *.rows.jsonl files (fields dataset, method, method_setting, similarity_setting,
model_input, model, noise_condition, level, draw, fold, score_D = before, score_Dprime = after), and
for n40 the per_level block of each <dataset>__<kind>.json. Written by Claude, 2026-10-10.
"""
import glob
import json
import math
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tests' / 'analysis'))
from ch7_numbers import CH7_DATASETS  # noqa: E402

DIRS = {'n07': 'n07_remediation', 'n13': 'n13_noise_distributions', 'n20': 'n20_clean_before_search',
        'n21': 'n21_neighbour_distance', 'n23': 'n23_uncertainty_cleaning', 'n24': 'n24_delete_or_keep',
        'n30': 'n30_censored_likelihood', 'n40': 'n40_model_side_regularisation'}
INPUTS = [f'noise/{d}/*.rows.jsonl' for e, d in DIRS.items() if e != 'n40'] + \
         ['noise/n40_model_side_regularisation/*__*.json']

CONFIGS = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')
CLASS_KINDS = ('uniform', 'confusion_directed', 'class_imbalance', 'instance_noise', 'binary_asymmetric',
               'class_dependent')
REG_KINDS = ('gaussian', 'outlier_p05', 'shift_only_p10', 'sparse_dprime_up_p10', 'grouped_shifted', 'censoring',
             'dense_oofmse')   # dense_oofmse: n40's Gaussian noise sized by a model's out-of-fold error
KINDS = REG_KINDS + CLASS_KINDS
PATIENT = re.compile(r'tcga|annitia|nhanes|pam50', re.I)


def main_level(kind):
    if kind == 'censoring':
        return 0.25
    if kind in CLASS_KINDS:
        return 0.1
    return 1.0


def _st(r):
    return r.get('method_setting') or ''


# Each row of the figure. block, action (delete / count less / move the label / model), words, the
# experiment, the method code, a filter on the setting, the arm the before score is paired with, and the
# control (method code, function that maps a control setting to the method's setting, or None).
def _strip_seed(s):
    return re.sub(r'\|seed=\d+$', '', s)


ROWS = [
    # ---- mitigation selected by a detector
    dict(block='detector', action='Delete', exp='n07', method='remove',
         words='Out-of-fold residual'),
    dict(block='detector', action='Delete', exp='n24', method='remove', keep=lambda s: s.startswith('cross_val'),
         words='Out-of-fold residual, second experiment', control=('delete_at_random', _strip_seed)),
    dict(block='detector', action='Delete', exp='n24', method='remove', keep=lambda s: 'neighbor_residual' in s,
         words='Neighbor-label gap', control=('delete_at_random', _strip_seed)),
    dict(block='detector', action='Delete', exp='n23', method='activity_cliff_aware', keep=lambda s: 'drop@' in s,
         words='Cliff-aware neighbor gap'),
    dict(block='detector', action='Delete', exp='n23', method='gaussian_process', keep=lambda s: 'drop@' in s,
         words='GP predicted spread'),
    dict(block='detector', action='Delete', exp='n23', method='hetero_gp', keep=lambda s: 'drop@' in s,
         words='Heteroscedastic GP spread'),
    dict(block='detector', action='Delete', exp='n23', method='heteroscedastic_nn', keep=lambda s: 'drop@' in s,
         words='Variance-head network spread'),
    dict(block='detector', action='Delete', exp='n23', method='ngboost', keep=lambda s: 'drop@' in s,
         words='NGBoost spread'),
    dict(block='detector', action='Delete', exp='n20', method='remove',
         keep=lambda s: 'arm=search_cleaned__fit_cleaned' in s,
         words='Out-of-fold residual, before search'),
    dict(block='detector', action='Count less', exp='n07', method='downweight',
         words='Out-of-fold residual'),
    dict(block='detector', action='Count less', exp='n21', method='neighbor_residual_weight',
         words='Neighbor-label gap, best distance'),
    dict(block='detector', action='Count less', exp='n21', method='local_variance_ratio_weight',
         words='Gap over neighbor spread, best distance'),
    dict(block='detector', action='Count less', exp='n13', method='local_variance_ratio',
         words='Gap over neighbor spread'),
    dict(block='detector', action="Move the label", exp='n07', method='label_smooth',
         words="Out-of-fold residual, to neighbors' mean"),
    dict(block='detector', action="Move the label", exp='n20', method='label_smooth',
         keep=lambda s: 'arm=search_cleaned__fit_cleaned' in s,
         words="Out-of-fold residual, to neighbors' mean, before search"),
    # ---- mitigation selected by a known flag (labels at the assay limit)
    dict(block='flag', action='Delete', exp='n30', method='drop_clipped', words='Clipped molecules'),
    dict(block='flag', action='Move the label', exp='n30', method='constant_lift',
         words='Clipped labels raised by one constant'),
    dict(block='flag', action='Move the label', exp='n30', method='per_molecule_lift',
         words='Clipped labels raised per molecule',
         control=('per_molecule_lift_shuffled', lambda s: s.split('/')[0].replace('_shuffled', ''))),
    dict(block='flag', action='Move the label', exp='n30', method='truncated_normal_mean',
         words='Clipped labels set to truncated-normal mean'),
    dict(block='flag', action='Move the label', exp='n30', method='censored_em',
         words='Clipped labels set by censored EM'),
    # ---- noise-robust training (n40; before = the same model at fixed default settings)
    dict(block='training', action='Model', exp='n40', method='stack', words='Stacked models'),
    dict(block='training', action='Model', exp='n40', method='seed_avg', words='Network averaged over seeds'),
    dict(block='training', action='Model', exp='n40', method='retuned', words='Retuned on the damaged labels',
         control=('tuned_clean_reused', lambda s: s)),
    dict(block='training', action='Model', exp='n40', method='tuned_clean_reused',
         words='Tuned on clean labels, reused'),
]
BLOCK_TITLES = {'detector': 'Noise mitigation, molecules selected by a detector',
                'flag': 'Noise mitigation, molecules selected by a known flag (label at the assay limit)',
                'training': 'Noise-robust training'}
ACTION_ORDER = ['Delete', 'Count less', 'Move the label', 'Model']


def row_key(spec):
    return f"{spec['block']}|{spec['action']}|{spec['words']}"


def _cfg(model_input):
    """'dnn:hybrid__general2', 'dnn@base_1', 'base_1|asis|dnn', 'hybrid' -> configuration name."""
    for tok in re.split(r'[:@|]', str(model_input)):
        if tok in CONFIGS:
            return tok
    return None


def _bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _mean(v):
    v = [x for x in v if not _bad(x)]
    return sum(v) / len(v) if v else None


def _better(a, b, lib):
    a, b = round(a, 4), round(b, 4)
    return a < b if lib else a > b


_ROWS_CACHE = {}


def _rows_of(root, exp, missing):
    if exp in _ROWS_CACHE:
        return _ROWS_CACHE[exp]
    out, seen = [], set()
    for p in sorted(glob.glob(str(root / 'noise' / DIRS[exp] / '*.rows.jsonl'))):
        if PATIENT.search(os.path.basename(p)):
            continue
        rel = os.path.relpath(p, root)
        for ln in open(p):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get('fold') != 0 or r.get('dataset') not in CH7_DATASETS:
                continue
            if 'clean_label' in str(r.get('model_input')):
                continue   # n07 hybrids searched on the clean labels: a reference, not the method
            k = (r['dataset'], r.get('method'), _st(r), r.get('similarity_setting'), r.get('model_input'),
                 r.get('model'), r.get('noise_condition'), r.get('level'), r.get('draw'))
            if k in seen:
                continue
            seen.add(k)
            r['_src'] = rel
            out.append(r)
    _ROWS_CACHE[exp] = out
    return out


def _n40_rows(root, missing):
    """n40 per_level -> rows in the rows.jsonl layout. before = arm 'fixed' of the same model; 'stack' is
    stored under fixed and is paired with each decided model's fixed score."""
    out = []
    for p in sorted(glob.glob(str(root / 'noise' / DIRS['n40'] / '*__*.json'))):
        b = os.path.basename(p)[:-5]
        ds, kind = b.split('__', 1)
        if ds not in CH7_DATASETS or PATIENT.search(b):
            continue
        if kind not in KINDS:
            missing(f'n40 {b}: damage kind {kind!r} is not one of the decided kinds; left out')
            continue
        d = json.load(open(p))
        if not d.get('has_computed_results') or not d.get('per_level'):
            missing(f'n40 {b}: no computed results ({d.get("result_status")})')
            continue
        lib = d['lower_is_better']
        rel = os.path.relpath(p, root)
        for L, bycfg in d['per_level'].items():
            for cfg, arms in bycfg.items():
                fixed = arms.get('fixed') or {}
                for m in d.get('models') or []:
                    for dk, before in (fixed.get(m) or {}).items():
                        if not str(dk).endswith('f0'):
                            continue
                        mm = re.match(r'l([\d.]+|none)d(\d+)f0', dk)
                        draw = int(mm.group(2)) if mm else 0
                        for arm in ('stack', 'seed_avg', 'retuned', 'tuned_clean_reused'):
                            src = fixed.get('stack') if arm == 'stack' else (arms.get(arm) or {}).get(m)
                            if not src or dk not in src:
                                continue
                            out.append(dict(dataset=ds, method=arm, method_setting='', similarity_setting='',
                                            model_input=cfg, model=m, noise_condition=kind, level=float(L),
                                            draw=draw, fold=0, lower_is_better=lib, metric=d['metric'],
                                            score_D=before, score_Dprime=src[dk], _src=rel))
    return out


def _condition(r):
    """-> (grid, damage kind) or None. Level 0 and 'as_they_come' are the labels as delivered."""
    lv, kind = r.get('level'), r.get('noise_condition')
    if kind == 'as_they_come' or lv in (0, 0.0):
        return ('qm9' if r['dataset'] == 'qm9' else 'delivered'), 'none'
    if kind not in KINDS or lv is None:
        return None
    if abs(float(lv) - main_level(kind)) > 1e-9:
        return None
    return f'damage:{kind}', kind


def load_f8(results_root, missing):
    results_root = pathlib.Path(results_root)
    _ROWS_CACHE.clear()
    n40 = _n40_rows(results_root, missing)
    out = []
    for spec in ROWS:
        src_rows = n40 if spec['exp'] == 'n40' else _rows_of(results_root, spec['exp'], missing)
        keep = spec.get('keep') or (lambda s: True)
        mine = [r for r in src_rows if r.get('method') == spec['method'] and keep(_st(r))]
        if not mine:
            missing(f"{row_key(spec)}: no fold-0 rows for {spec['exp']} {spec['method']}")
            continue
        ctrl = spec.get('control')
        ctrl_rows = {}
        if ctrl:
            for r in src_rows:
                if r.get('method') != ctrl[0]:
                    continue
                c = _condition(r)
                if c is None:
                    continue
                k = (r['dataset'], c[0], r.get('model'), _cfg(r.get('model_input')), ctrl[1](_st(r)),
                     r.get('similarity_setting'))
                ctrl_rows.setdefault(k, {}).setdefault(_st(r), []).append(r['score_Dprime'])
        # cell -> pair -> setting -> before / after lists
        cells = {}
        for r in mine:
            c = _condition(r)
            if c is None:
                continue
            cfg = _cfg(r.get('model_input'))
            if cfg is None:
                missing(f"{row_key(spec)}: configuration not read from {r.get('model_input')!r}")
                continue
            cell = cells.setdefault((r['dataset'], c[0]), dict(kind=c[1], pairs={}, lib=r['lower_is_better'],
                                                               metric=r.get('metric'), src=set(), draws=set(),
                                                               levels=set()))
            cell['src'].add(r['_src'])
            cell['levels'].add(r.get('level'))
            cell['draws'].add((r.get('noise_condition'), r.get('draw')))
            st = (_st(r), r.get('similarity_setting'))
            p = cell['pairs'].setdefault((r['model'], cfg), {}).setdefault(st, [[], []])
            p[0].append(r['score_D'])
            p[1].append(r['score_Dprime'])
        for (ds, grid), cell in sorted(cells.items()):
            lib = cell['lib']
            run = imp = cb = cn = 0
            pairs, pairs_imp, settings, ctrl_txt = [], [], {}, []
            for (m, cfg), bys in sorted(cell['pairs'].items(), key=lambda t: (t[0][0], CONFIGS.index(t[0][1]))):
                scored = []
                for st, (bs, as_) in bys.items():
                    b, a = _mean(bs), _mean(as_)
                    if b is None or a is None:
                        continue
                    scored.append((st, b, a, len(as_)))
                if not scored:
                    continue
                # settings scored on fewer draws than the pair's fullest setting are left out, so the
                # best setting is not picked from one lucky draw
                full = max(t[3] for t in scored)
                thin = sum(1 for t in scored if t[3] < full)
                if thin:
                    cell.setdefault('thin', 0)
                    cell['thin'] += thin
                scored = [t[:3] for t in scored if t[3] == full]
                run += 1
                name = f'{m}@{cfg}'
                best = max(scored, key=lambda t: (t[1] - t[2]) if lib else (t[2] - t[1]))
                ok = any(_better(a, b, lib) for _, b, a in scored)
                imp += ok
                if ok:
                    pairs_imp.append(name)
                pairs.append(f'{name}:{best[1]:.4f}->{best[2]:.4f}')
                settings[name] = ' / '.join(x for x in best[0] if x and x != 'none') or '-'
                if ctrl:
                    seeds = ctrl_rows.get((ds, grid, m, cfg, best[0][0], best[0][1]))
                    if seeds:
                        cvals = [_mean(v) for v in seeds.values()]
                        cvals = [v for v in cvals if v is not None]
                        if cvals:
                            cbest = min(cvals) if lib else max(cvals)
                            cn += 1
                            won = _better(best[2], cbest, lib)
                            cb += won
                            ctrl_txt.append(f'{name}:{best[2]:.4f} vs {cbest:.4f} ({len(cvals)} seeds)')
            if not run:
                continue
            out.append(dict(
                row_key=row_key(spec), block=spec['block'], action=spec['action'], row_label=spec['words'],
                experiment=spec['exp'], method=spec['method'], dataset=ds, grid=grid, damage_kind=cell['kind'],
                level=sorted(x for x in cell['levels'] if x is not None) or ['as_they_come'],
                metric=cell['metric'], lower_is_better=lib, n_pairs=run, n_pairs_improved=imp,
                share_improved=imp / run, has_control=bool(ctrl), n_control_pairs=cn, n_control_beaten=cb,
                control_mark=('' if not ctrl or not cn else ('dot' if cb >= cn / 2 else 'cross')),
                control=ctrl[0] if ctrl else '', control_pairs=ctrl_txt, pairs=pairs, pairs_improved=pairs_imp,
                pair_settings=settings, n_draw_copies=len(cell['draws']),
                settings_left_out_fewer_draws=cell.get('thin', 0), source=sorted(cell['src'])))
            if ctrl and not cn:
                missing(f'{row_key(spec)} {ds} {grid}: control {ctrl[0]} has no score at the best setting')
    return out


if __name__ == '__main__':
    import collections
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/seagate/kirby_results_2026-10-10')
    MISS = []
    rows = load_f8(root, MISS.append)
    print(len(rows), 'cells')
    for k, n in collections.Counter(r['row_key'] for r in rows).items():
        print(f'  {n:3d}  {k}')
    print('columns:', sorted({(r['dataset'], r['grid']) for r in rows}))
    for m in MISS[:40]:
        print('  missing:', m[:200])
