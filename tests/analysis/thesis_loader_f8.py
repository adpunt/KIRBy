"""Loader for thesis figure F8: label-cleaning methods, labels as delivered vs injected damage.

One cell = one (experiment, method, dataset, grid). The value is how many of the
model-and-configuration pairs (2 downstream models x base_1, base_2, hybrid,
hybrid__general2) scored better on held-out molecules after cleaning than with the
labels the method started from. Fold 0. Read at level 0 ('delivered') and at the
main damaged level (1.0 of the clean-label spread; censoring 0.25; class-flip kinds 0.1).

Reads the raw result files under <results_root>/noise/<experiment dir>/ only. Ported
from the 2026-10-08 analysis scripts (answers/_scripts/n07.py ... n38.py and
cleaning_by_method.py), which read the same files and wrote one text line per improved
pair. Written by Claude, 2026-10-09.

Rules carried over from cleaning_by_method.py:
- "Improved": cleaned score better than the starting score after both are rounded to
  4 decimals (lower for rmse, higher for spearman/auc/mcc). A tie is not improved.
- A method's settings (flag rate, strength, neighbour screen, cleaning representation,
  winsor cut-off) are collapsed: a pair counts as improved if ANY setting improved it.
  The setting that gave the largest gain is kept in `pair_settings`.
- Reweighting (downweight, huber_weight, robust_loss, invvar, itlm, ...) and
  training-side methods are excluded.
- n25 "put cliff molecules back" is compared with the same cleanup without put-back.
- n38 caco2 and n20 caco2 have no level 1.0; the prior analysis read them at 0.5
  (LEVEL_EXCEPTIONS below). The row's `level` field says which level was read.
"""
import ast
import glob
import json
import math
import os
import pathlib
import re
import sys

EXPERIMENTS = {
    'n07': 'n07_remediation', 'n18': 'n18_clean_the_real_labels', 'n20': 'n20_clean_before_search',
    'n23': 'n23_uncertainty_cleaning', 'n24': 'n24_delete_or_keep', 'n25': 'n25_cliff_exemption',
    'n27': 'n27_removal_cost_and_propagation', 'n30': 'n30_censored_likelihood',
    'n34': 'n34_group_random_effects', 'n38': 'n38_noise_relative_to_model_error',
}
REPO = pathlib.Path(__file__).resolve().parents[2]   # tests/analysis/f8.py -> repo root
INPUTS = []
for _d in EXPERIMENTS.values():
    INPUTS += [f'noise/{_d}/*.json', f'noise/{_d}/*.partial.jsonl', f'noise/{_d}/_partial/*.json',
               f'noise/{_d}/_checkpoints/*.json']

CONFIGS = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')
CLASS_KINDS = {'binary_asymmetric', 'class_dependent', 'class_imbalance', 'confusion_directed',
               'instance_noise', 'uniform'}
# (experiment, dataset) -> level read instead of the main level, as the prior analysis did.
LEVEL_EXCEPTIONS = {('n20', 'caco2'): 0.5, ('n38', 'caco2'): 0.5}

HANDLE = {'remove': 'remove', 'delete': 'remove', 'drop': 'remove', 'drop_clipped': 'remove clipped molecules',
          'knn_smooth': "replace with neighbours' mean", 'label_smooth': "move toward neighbours' mean",
          'oof_smooth': 'move toward out-of-fold prediction', 'gp_smooth': 'move toward gp prediction',
          'js_shrink': 'shrink toward out-of-fold prediction', 'prox_smooth': "move toward forest-neighbours' mean",
          'constant_lift': 'raise clipped labels by one constant', 'per_molecule_lift': 'raise clipped labels per molecule',
          'truncated_normal_mean': 'replace clipped labels by truncated-normal mean',
          'censored_em': 'replace clipped labels by censored EM estimate',
          'move_proportional': 'move toward prediction, in proportion to uncertainty',
          'move_confidence': 'move toward prediction, by confidence',
          'group_center': 'subtract scaffold-group mean offset', 'group_center_3plus': 'subtract group offset (groups of 3+)',
          'random_intercept': 'subtract shrunk group offset', 'winsorise': 'pull extreme labels in to c SD',
          'winsor': 'pull clipped labels toward the limit'}
WEIGHTS = ('downweight', 'huber_weight', 'robust_loss', 'invvar', 'itlm', 'two_learner', 'shrink_flagged')
ASSAY_LIMIT = 'labels at the assay limit'
EVERY_LABEL = 'none (every label)'


def target_level(kind):
    if kind == 'censoring':
        return 0.25
    if kind in CLASS_KINDS:
        return 0.1
    return 1.0


def _cut_datasets(repo_root):
    """CUT_DATASETS + EXCLUDED_DATASETS from orchestrate/plan.py, read without importing it."""
    src = (repo_root / 'tests' / 'experiments' / 'orchestrate' / 'plan.py').read_text()
    out = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) in ('CUT_DATASETS', 'EXCLUDED_DATASETS')
                                                for t in node.targets):
            out |= set(ast.literal_eval(node.value))
    return out


def _decode_fn(repo_root):
    sys.path.insert(0, str(repo_root / 'tests' / 'experiments'))
    from _checkpoint import decode  # noqa: E402
    return decode


def _bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f0(d):
    """Mean over noise draws of fold 0. d: {'l0.1d2f0': score, ...} or a number.
    Keys without a fold tag are all taken."""
    if d is None:
        return None
    if not isinstance(d, dict):
        return None if _bad(d) else float(d)
    items = list(d.items())
    tagged = [(k, v) for k, v in items if re.search(r'f\d+$', str(k))]
    if tagged:
        items = [(k, v) for k, v in tagged if re.search(r'f0$', str(k))]
    v = [x for _, x in items if not _bad(x)]
    return sum(v) / len(v) if v else None


def _nd(d):
    """Number of fold-0 noise draws behind a {'l1d2f0': score} dict."""
    if not isinstance(d, dict):
        return None
    t = [k for k in d if re.search(r'f\d+$', str(k))]
    return sum(1 for k in t if str(k).endswith('f0')) if t else len(d)


def _fold0_block(b, folds_run):
    """{'mean','per_fold',...} block -> fold-0 score."""
    if not isinstance(b, dict):
        return None
    pf = b.get('per_fold')
    if pf and folds_run and 0 in folds_run and len(pf) == len(folds_run):
        v = pf[folds_run.index(0)]
        return None if _bad(v) else v
    v = b.get('mean')
    if folds_run and folds_run != [0]:
        return None   # several folds and no per-fold list: cannot isolate fold 0
    return None if _bad(v) else v


class _Rec(dict):
    pass


def _rec(exp, det, hraw, ds, kind, level, model, cfg, before, after, setting, metric, lib, source, variant='',
         nd=None):
    return _Rec(exp=exp, det=det, hraw=hraw, variant=variant, ds=ds, kind=kind, level=float(level), model=model,
                cfg=cfg, before=before, after=after, setting=setting, metric=metric, lib=lib, source=source, nd=nd)


# ------------------------------------------------------------------ per-experiment readers
# Each yields (records, expected) where expected is [(exp, det, hraw, variant, ds, kind, level, model, cfg, source)]
# for every pair the file says it ran the method on; records carry before/after (None when absent).

def _files(root, exp, cut, partial_dir=None):
    """[(ds, kind or None, path)] for final result files, plus partial-dir files with no final twin."""
    d = root / 'noise' / EXPERIMENTS[exp]
    out = {}
    for p in sorted(glob.glob(str(d / '*.json'))):
        b = os.path.basename(p)
        if 'fingerprint' in b or '.rows.' in b:
            continue
        ds, _, kind = b[:-5].partition('__')
        out[(ds, kind or None)] = p
    if partial_dir:
        for p in sorted(glob.glob(str(d / partial_dir / '*.json'))):
            b = os.path.basename(p)
            if 'fingerprint' in b:
                continue
            ds, _, kind = b[:-5].partition('__')
            out.setdefault((ds, kind or None), p)
    return [(ds, k, p) for (ds, k), p in sorted(out.items(), key=lambda t: (t[0][0], t[0][1] or '')) if ds not in cut]


def _skip(o):
    return (not o.get('has_computed_results', True) and o.get('result_status') != 'partial') or o.get('skipped')


def read_n07(root, cut, missing):
    for ds, kind, p in _files(root, 'n07', cut):
        if not kind or p.endswith('__n07.json'):
            continue
        r = json.load(open(p))
        if _skip(r) or 'by_level' not in r:
            continue
        fr, lib, met = r.get('folds_run') or [0], r['lower_is_better'], r['metric']
        for L, bl in r['by_level'].items():
            for m in r['models']:
                for cfg in r['configurations_per_model']:
                    sc = bl['scores'].get(m, {}).get(cfg, {})
                    b = _fold0_block(sc.get('noisy_baseline'), fr)
                    for s in r['strategies']:
                        if s in ('no_noise', 'noisy_baseline'):
                            continue
                        h, _, fs = s.partition('@')
                        det = {'js_shrink': EVERY_LABEL, 'constant_lift': ASSAY_LIMIT}.get(h, 'cross_val')
                        a = _fold0_block(sc.get(s), fr)
                        yield _rec('n07', det, h, ds, kind, L, m, cfg, b, a, fs or '-', met, lib, p,
                                   nd=bl.get('n_draws'))


def read_n18(root, cut, missing):
    for ds, kind, p in _files(root, 'n18', cut):
        d = json.load(open(p))
        if _skip(d) or 'per_space' not in d:
            continue
        lib, met = d['lower_is_better'], d['metric']
        for cfg, sp in d['per_space'].items():
            for det, byr in sp['per_detector'].items():
                for rate, blk in byr.items():
                    for m, pm in blk['per_model'].items():
                        b = pm.get('no_cleaning_mean')
                        for cl, v in pm.items():
                            if cl == 'no_cleaning_mean':
                                continue
                            a = v.get('cleaned_mean') if v.get('folds_scored') else None
                            dw = ASSAY_LIMIT if det == 'no_detector' else det
                            ax = v.get('swept_axis')
                            st = f"{ax} {v.get('setting')}" if ax not in (None, 'none') else '-'
                            yield _rec('n18', dw, cl, ds, 'delivered', 0, m, cfg, None if _bad(b) else b,
                                       None if _bad(a) else a, st, met, lib, p)


def read_n20(root, cut, missing):
    A, AC, CA = 'search_asis__fit_asis', 'search_asis__fit_cleaned', 'search_cleaned__fit_asis'
    COMP = ((AC, 'fit on cleaned labels'), (CA, 'cleaned for the feature search only, fit on labels as delivered'))
    reads = {'remove', 'knn_smooth'}
    for ds, kind, p in _files(root, 'n20', cut, partial_dir='_partial'):
        d = json.load(open(p))
        if _skip(d) or not d.get('per_regime'):
            continue
        lib, met = d['lower_is_better'], d['metric']
        for reg, byL in d['per_regime'].items():
            for L, b0 in byL.items():
                if not b0:
                    continue
                lv = 0.0 if reg == 'as_they_come' else float(L)
                knd = 'as_they_come' if reg == 'as_they_come' else kind
                for base, bc in b0.items():
                    for cl, bf in bc.items():
                        if cl == 'no_cleaning':
                            continue
                        for fr, bm in bf.items():
                            if cl not in reads and fr != '0.05':
                                continue
                            for mc, arms in bm.items():
                                m, cfg = mc.split('__', 1)
                                sc = {a: _f0(u) for a, u in arms.items() if u}
                                for arm, var in COMP:
                                    st = f'cleaned in {base}' + (f', flag rate {fr}' if cl in reads else '') + \
                                         (', as_they_come regime' if reg == 'as_they_come' else '')
                                    yield _rec('n20', 'cross_val', cl, ds, knd, lv, m, cfg, sc.get(A), sc.get(arm),
                                               st, met, lib, p, variant=var, nd=_nd(arms.get(arm)))


def read_n23(root, cut, missing):
    S = ['hetero_gp', 'ngboost', 'heteroscedastic_nn', 'gaussian_process', 'activity_cliff_aware']
    for ds, kind, p in _files(root, 'n23', cut):
        if not kind:
            continue
        o = json.load(open(p))
        if _skip(o) or 'per_regime' not in o:
            continue
        lib, met = o['lower_is_better'], o['metric']
        inj = o['per_regime']['injected']
        for L, byc in inj.items():
            for cfg, bys in byc.items():
                for m in o['models']:
                    base = None
                    for s in S:
                        if m in bys.get(s, {}).get('none', {}):
                            base = _f0(bys[s]['none'][m])
                            break
                    for s in S:
                        det = 'neighbour-label gap' if s == 'activity_cliff_aware' else f'{s} uncertainty'
                        for a, v in bys.get(s, {}).items():
                            if a == 'none':
                                continue
                            h, _, st = a.partition('@')
                            yield _rec('n23', det, h, ds, kind, L, m, cfg, base, _f0(v.get(m)), st or '-', met, lib, p,
                                       nd=_nd(v.get(m)))


def read_n24(root, cut, missing):
    for ds, kind, p in _files(root, 'n24', cut):
        if not kind:
            continue
        o = json.load(open(p))
        if _skip(o) or 'per_level' not in o:
            continue
        lib, met = o['lower_is_better'], o['metric']
        reads = set(o.get('arms_that_read_the_flag_rate') or [])
        cu_of = o.get('arm_cleanup') or {}
        for L, byc in o['per_level'].items():
            for cfg, byd in byc.items():
                for det, byr in byd.items():
                    for rr, arms in byr.items():
                        for m in o['models']:
                            b = _f0((arms.get('no_cleaning') or {}).get(m))
                            for a, x in arms.items():
                                if a == 'no_cleaning' or a.startswith('delete_at_random'):
                                    continue
                                cu = cu_of.get(a, a.split('/')[-1])
                                if cu not in reads and rr != '0.05':
                                    continue   # rate-blind arm: same fit under every rate
                                st = det + (f', flag rate {rr}' if cu in reads else '')
                                yield _rec('n24', det.split('@')[0], cu, ds, kind, L, m, cfg, b, _f0(x.get(m)),
                                           st, met, lib, p, nd=_nd(x.get(m)))


def read_n25(root, cut, missing):
    for ds, kind, p in _files(root, 'n25', cut):
        if not kind:
            continue
        o = json.load(open(p))
        if _skip(o) or 'per_level' not in o:
            continue
        lib, met = o['lower_is_better'], o['metric']
        fr = o.get('folds_run') or [0]
        for L, PL in o['per_level'].items():
            for cfg, sp in PL['per_space'].items():
                for m in o['models']:
                    b = _fold0_block(sp['per_model'].get(m, {}).get('no_cleaning'), fr)
                    seen = set()
                    for det, dv in sp['per_detector'].items():
                        for slot, sv in dv['per_setting_slot'].items():
                            for cu, cv in sv['per_model'].get(m, {}).items():
                                ax = cv.get('swept_axis')
                                free = ax in (None, 'none')
                                dw = (ASSAY_LIMIT if cu == 'constant_lift' else EVERY_LABEL) if free else det
                                st = '-' if free else f"{ax} {cv.get('setting')}"
                                if (dw, cu, st) in seen:
                                    continue   # setting-free cleanup repeated under every detector and slot
                                seen.add((dw, cu, st))
                                plain = cv.get('plain_mean')
                                plain = None if _bad(plain) else plain
                                yield _rec('n25', dw, cu, ds, kind, L, m, cfg, b, plain, st, met, lib, p,
                                           variant='plain cleanup')
                                ce = (cv.get('cliff_exempt') or {}).get('arm_mean')
                                if plain is not None and 'cliff_exempt' in cv:
                                    yield _rec('n25', dw, cu, ds, kind, L, m, cfg, plain, None if _bad(ce) else ce,
                                               st, met, lib, p, variant='cliff molecules put back')


def read_n27(root, cut, missing):
    for ds, kind, p in _files(root, 'n27', cut):
        o = json.load(open(p))
        if _skip(o) or 'per_fold' not in o:
            continue
        lib, met = o.get('lower_is_better', True), o['metric']
        pf = o['per_fold']
        for cfg in CONFIGS:
            for m in o['models']:
                b = ((pf['reference'].get(cfg) or {}).get(m) or {}).get('f0', {}).get('score')
                for al, byc in (pf.get('relabelled', {}).get(cfg) or {}).items():
                    a = ((byc.get(m) or {}).get('f0') or {}).get('score')
                    yield _rec('n27', EVERY_LABEL, 'oof_smooth', ds, 'delivered', 0, m, cfg, b, a, f'alpha {al}',
                               met, lib, p)


def read_n30(root, cut, missing):
    TR = ['constant_lift', 'per_molecule_lift', 'truncated_normal_mean', 'censored_em', 'drop_clipped',
          'winsor_c1', 'winsor_c1.25', 'winsor_c1.5']
    for ds, kind, p in _files(root, 'n30', cut):
        r = json.load(open(p))
        if _skip(r) or 'per_level' not in r:
            continue
        lib, met = r.get('lower_is_better', True), r['metric']
        for L, byc in r['per_level'].items():
            for cfg in CONFIGS:
                for m in r['models']:
                    c = byc.get(cfg, {})
                    b = _f0((c.get('clipped') or {}).get(m))
                    for a in TR:
                        if a not in c:
                            continue
                        h = 'winsor' if a.startswith('winsor') else a
                        yield _rec('n30', ASSAY_LIMIT, h, ds, 'censoring', L, m, cfg, b, _f0(c[a].get(m)),
                                   a if h == 'winsor' else '-', met, lib, p, nd=_nd(c[a].get(m)))


def read_n34(root, cut, missing, decode):
    d = root / 'noise' / EXPERIMENTS['n34']
    ARMS = ['group_center', 'group_center_3plus', 'random_intercept']
    stems = {os.path.basename(x)[:-5] for x in glob.glob(str(d / '*.json')) + glob.glob(str(d / '_checkpoints' / '*.json'))
             if 'fingerprint' not in x and '__' in os.path.basename(x)}
    for stem in sorted(stems):
        ds, kind = stem.split('__', 1)
        if ds in cut:
            continue
        acc = src = None
        p = d / f'{stem}.json'
        if p.exists():
            j = json.load(open(p))
            if j.get('has_computed_results') and 'per_condition' in j:
                acc, src = j['per_condition'], str(p)
        if acc is None:
            p = d / '_checkpoints' / f'{stem}.json'
            if p.exists():
                o = decode(json.load(open(p)))
                if o.get('state') and o['state'].get('acc'):
                    acc, src = o['state']['acc'], str(p)
        if acc is None:
            continue
        for c, byL in acc.items():
            for L, bysp in byL.items():
                lv = 0.0 if c == 'none' else float(L)
                for cfg, a in bysp.items():
                    for m in (a.get('none') or {}):
                        b = _f0(a['none'][m])
                        for arm in ARMS:
                            yield _rec('n34', 'scaffold groups', arm, ds, kind, lv, m, cfg, b,
                                       _f0((a.get(arm) or {}).get(m)), '-', 'rmse', True, src,
                                       nd=_nd((a.get(arm) or {}).get(m)))


def read_n38(root, cut, missing, decode):
    d = root / 'noise' / EXPERIMENTS['n38']
    stems = {os.path.basename(x)[:-5] for x in glob.glob(str(d / '*.json')) + glob.glob(str(d / '_checkpoints' / '*.json'))
             if 'fingerprint' not in x and '__' in os.path.basename(x)}
    for stem in sorted(stems):
        ds, kind = stem.split('__', 1)
        if ds in cut:
            continue
        acc = src = None
        p = d / f'{stem}.json'
        if p.exists():
            j = json.load(open(p))
            if j.get('has_computed_results') and j.get('per_level'):
                acc, src = j['per_level'], str(p)
        if acc is None:
            p = d / '_checkpoints' / f'{stem}.json'
            if p.exists():
                o = decode(json.load(open(p)))
                s = o.get('state')
                if s and s.get('acc'):
                    acc, src = s['acc'], str(p)
        if acc is None:
            continue
        for base, byL in acc.items():
            for L, byc in byL.items():
                for cfg, a in byc.items():
                    for m in (a.get('no_cleaning') or {}):
                        b = _f0(a['no_cleaning'][m])
                        x = 'winsorise_c1.5'   # the other three n38 methods are training-side or reweighting
                        if x in a:
                            yield _rec('n38', 'residual beyond c SD', 'winsorise', ds, kind, L, m, cfg, b,
                                       _f0(a[x].get(m)), f'c1.5, noise on {base} labels', 'rmse', True, src,
                                       nd=_nd(a[x].get(m)))


# ------------------------------------------------------------------ assembly

def _label(det, hw, variant):
    s = f'{hw}; flagged by {det}'
    if variant == 'cliff molecules put back':
        s = f'put cliff molecules back after: {hw}; flagged by {det}'
    elif variant and variant != 'plain cleanup':
        s += f' ({variant})'
    return s


def _improved(before, after, lib):
    b, a = round(before, 4), round(after, 4)
    return a < b if lib else a > b


def load_f8(results_root: pathlib.Path, missing) -> list:
    results_root = pathlib.Path(results_root)
    repo = REPO
    cut = _cut_datasets(repo)
    decode = _decode_fn(repo)

    # partial checkpoints (*.partial.jsonl) hold per-unit intermediate payloads, not the
    # result layout; a dataset/kind with only a .partial.jsonl is reported, not read.
    for exp, dname in EXPERIMENTS.items():
        for p in glob.glob(str(results_root / 'noise' / dname / '*.partial.jsonl')):
            stem = os.path.basename(p)[:-len('.partial.jsonl')]
            if stem.split('__')[0] in cut:
                continue
            if not os.path.exists(os.path.join(os.path.dirname(p), stem + '.json')):
                missing(f'{exp} {stem}: only a .partial.jsonl checkpoint exists; its unit payloads are not read')

    recs = []
    readers = [read_n07, read_n18, read_n20, read_n23, read_n24, read_n25, read_n27, read_n30]
    for f in readers:
        recs += list(f(results_root, cut, missing))
    recs += list(read_n34(results_root, cut, missing, decode))
    recs += list(read_n38(results_root, cut, missing, decode))

    unmapped = sorted({(r['exp'], r['hraw']) for r in recs
                       if not any(w in r['hraw'] for w in WEIGHTS) and r['hraw'] not in HANDLE})
    for exp, h in unmapped:
        missing(f'{exp}: label handling {h!r} has no words in HANDLE; left out')
    recs = [r for r in recs if r['hraw'] in HANDLE and not any(w in r['hraw'] for w in WEIGHTS)]

    # which level is read per (experiment, dataset, damage kind)
    levels_of = {}
    for r in recs:
        if r['level'] > 0:
            levels_of.setdefault((r['exp'], r['ds'], r['kind']), set()).add(r['level'])
    read_level = {}
    for (exp, ds, kind), lvs in sorted(levels_of.items()):
        want = target_level(kind)
        if any(abs(l - want) < 1e-9 for l in lvs):
            read_level[(exp, ds, kind)] = want
            continue
        alt = LEVEL_EXCEPTIONS.get((exp, ds))
        if alt is not None and any(abs(l - alt) < 1e-9 for l in lvs):
            read_level[(exp, ds, kind)] = alt
            missing(f'{exp} {ds} {kind}: main level {want} not in the file (levels {sorted(lvs)}); '
                    f'read at {alt} as the 2026-10-08 analysis did')
        else:
            missing(f'{exp} {ds} {kind}: main level {want} not in the file (levels {sorted(lvs)}); no damage cell')

    # group into cells
    cells = {}
    for r in recs:
        if r['level'] == 0:
            grid, dk = 'delivered', 'none'
        else:
            want = read_level.get((r['exp'], r['ds'], r['kind']))
            if want is None or abs(r['level'] - want) > 1e-9:
                continue
            grid, dk = f"damage:{r['kind']}", r['kind']
        key = (r['exp'], r['det'], r['hraw'], r['variant'], r['ds'], grid)
        c = cells.setdefault(key, dict(recs=[], dk=dk, levels=set(), metric=r['metric'], lib=r['lib'], sources=set(),
                                       nds=set()))
        if r['nd'] is not None:
            c['nds'].add(r['nd'])
        c['recs'].append(r)
        c['levels'].add(r['level'])
        c['sources'].add(os.path.relpath(r['source'], results_root))

    out = []
    for key, c in sorted(cells.items()):
        exp, det, hraw, variant, ds, grid = key
        lib = c['lib']
        gain = (lambda r: r['before'] - r['after']) if lib else (lambda r: r['after'] - r['before'])
        bypair = {}
        for r in c['recs']:
            bypair.setdefault((r['model'], r['cfg']), []).append(r)
        expected = sorted(bypair, key=lambda t: (t[0], CONFIGS.index(t[1]) if t[1] in CONFIGS else 9))
        pairs, pairs_imp, settings, lacking, conflicts = [], [], {}, [], []
        by_copy = {}   # level-0 copy (damage-kind file it sits in) -> [improved, run]
        for pr in expected:
            rs = bypair[pr]
            ok = [r for r in rs if r['before'] is not None and r['after'] is not None]
            name = f'{pr[0]}@{pr[1]}'
            if not ok:
                why = 'before' if all(r['before'] is None for r in rs) else 'after'
                lacking.append(f'{name} (no {why} score)')
                continue
            best = max(ok, key=gain)
            imp = any(_improved(r['before'], r['after'], lib) for r in ok)
            pairs.append(f"{name}:{best['before']:.4f}->{best['after']:.4f}")
            if imp:
                pairs_imp.append(name)
            settings[name] = best['setting'] + (f" [level 0 of the {best['kind']} file]" if grid == 'delivered' else '')
            if grid == 'delivered':
                verdict = {}
                for k in sorted({r['kind'] for r in ok}):
                    verdict[k] = any(_improved(r['before'], r['after'], lib) for r in ok if r['kind'] == k)
                    by_copy.setdefault(k, [0, 0])
                    by_copy[k][0] += verdict[k]
                    by_copy[k][1] += 1
                if len(set(verdict.values())) > 1:
                    conflicts.append(name)
        run = len(pairs)
        extra = [f'{m}@{cf}' for m in sorted({p[0] for p in expected}) for cf in CONFIGS if (m, cf) not in bypair]
        label = f'{exp} {ds} {grid} [{_label(det, HANDLE[hraw], variant)}]'
        if lacking:
            missing(f'{label}: {len(lacking)} of {len(expected)} pairs not scored: ' + ', '.join(lacking))
        if extra:
            missing(f'{label}: pairs absent from the file: ' + ', '.join(extra))
        if conflicts:
            missing(f'{label}: the level-0 copies in different damage-kind files disagree for '
                    + ', '.join(conflicts) + '; counted improved if any copy improved')
        out.append(dict(
            experiment=exp, detector=det, handling=HANDLE[hraw], handling_code=hraw, variant=variant,
            method_label=f'{exp}: ' + _label(det, HANDLE[hraw], variant),
            dataset=ds, grid=grid, damage_kind=c['dk'],
            level=sorted(c['levels'])[0] if len(c['levels']) == 1 else sorted(c['levels']),
            n_pairs_run=run, n_pairs_improved=len(pairs_imp) if run else None,
            n_pairs_expected=len(expected) + len(extra),
            metric=c['metric'], lower_is_better=lib, pairs=pairs, pairs_improved=pairs_imp,
            pair_settings=settings,
            n_pairs_improved_by_level0_copy={k: f'{v[0]} of {v[1]}' for k, v in by_copy.items()},
            n_draws=sorted(c['nds']), source=sorted(c['sources'])))
    for src in sorted({s for r in out for s in r['source'] if '/_partial/' in s or '/_checkpoints/' in s}):
        missing(f'{src}: read from an unfinished run (partial or checkpoint file); draws may be incomplete')
    return out


# ------------------------------------------------------------------ agreement with answers/_noise_lines
def _prior_parse(exp, meth):
    """cleaning_by_method.parse, copied (2026-10-08 analysis)."""
    m = meth.lower()
    if any(w in m for w in WEIGHTS): return None
    if exp == 'n07':
        h = re.split(r'[@ ]', m)[0]
        return {'js_shrink': EVERY_LABEL, 'constant_lift': ASSAY_LIMIT}.get(h, 'cross_val'), HANDLE.get(h)
    if exp == 'n18':
        p = [x.strip() for x in m.split(',')]
        return (p[1] if len(p) > 1 and p[0] != 'constant_lift' else ASSAY_LIMIT), HANDLE.get(p[0])
    if exp == 'n20':
        if not ('vs fit as they are' in m or 'vs neither' in m): return None
        return 'cross_val', HANDLE.get(m.split(',')[0])
    if exp == 'n23':
        mm = re.match(r'(\w+) uncertainty, (\w+?)@', m)
        if not mm: return None
        src, act = mm.groups()
        return ('neighbour-label gap' if src == 'activity_cliff_aware' else f'{src} uncertainty'), HANDLE.get(act)
    if exp == 'n24':
        mm = re.match(r'(\w+) \((\w+)', m)
        return (mm.group(2), HANDLE.get(mm.group(1))) if mm else None
    if exp == 'n25':
        mm = re.search(r'put back after (\w+) \((\w+)', m)
        if mm: return mm.group(2), 'put cliff molecules back after ' + HANDLE.get(mm.group(1), mm.group(1))
        mm = re.search(r'plain cleanup (\w+) \((\w+)', m)
        return (mm.group(2), HANDLE.get(mm.group(1))) if mm else None
    if exp == 'n30':
        h = 'winsor' if m.startswith('winsor') else m.split()[0]
        return ASSAY_LIMIT, HANDLE.get(h)
    if exp == 'n34': return 'scaffold groups', HANDLE.get(m.split()[0])
    if exp == 'n38': return ('residual beyond c SD', HANDLE['winsorise']) if m.startswith('winsorise') else None
    if exp == 'n27': return EVERY_LABEL, 'move toward out-of-fold prediction'
    return None


def prior_improved_sets(lines_dir, with_clean_label_features=False):
    """{(exp, det, handling words, ds, grid): set('model@config')} from _noise_lines, as cleaning_by_method read it."""
    REAL = re.compile(r'(@ 0(?![.\d])|no damage|as delivered|real labels|nothing injected|undamaged|correct labels|none \(correct)', re.I)
    out = {}
    for exp in EXPERIMENTS:
        fn = os.path.join(lines_dir, exp + '.txt')
        if not os.path.exists(fn):
            continue
        for ln in open(fn):
            if ln.startswith('#'):
                continue
            f = [x.strip() for x in ln.split(' | ')]
            if len(f) < 6:
                continue
            ds, _, cond, cfg, meth = f[:5]
            if 'features from clean labels' in cfg and not with_clean_label_features:
                continue
            r = _prior_parse(exp, meth)
            if not r or not r[1]:
                continue
            mm = re.match(r'(\w+)@(\w+)', cfg.replace(' ', ''))
            if not mm:
                continue
            if exp == 'n27':
                m = re.search(r'alpha 0 (\d+\.\d+) → alpha [\d.]+ (\d+\.\d+)', ln)
            else:
                m = re.search(r'(-?\d+\.\d+)[^|→]*?→[^|\d-]*?(-?\d+\.\d+)', ln)
            if not m:
                continue
            a, b = float(m.group(1)), float(m.group(2))
            unit = (re.search(r'\((rmse|spearman|auc|mcc|f1\w*)', ln) or [None, 'rmse'])[1]
            g = (a - b) if unit == 'rmse' else (b - a)
            if g <= 0:
                continue
            if REAL.search(cond):
                grid = 'delivered'
            else:
                kind = re.sub(r' \(.*\)$', '', cond.split(' @')[0])
                L = re.search(r'@ ([\d.]+)', cond)
                main = {'censoring': 0.25}.get(kind, 0.1 if ds == 'herg' else 1.0)
                if not L or not (float(L.group(1)) == main or
                                 (ds == 'caco2' and float(L.group(1)) == 0.5 and exp in ('n20', 'n38'))):
                    continue
                grid = f'damage:{kind}'
            out.setdefault((exp, r[0], r[1], ds, grid), set()).add(f'{mm.group(1)}@{mm.group(2)}')
    return out


def agreement(rows, lines_dir, with_clean_label_features=False):
    """Compare improved-pair sets per (exp, det, handling, ds, grid). n20's two comparisons are merged and
    n25 put-back handling is renamed, as cleaning_by_method did; n25 setting-free cleanups (one fit shown under
    every detector there) are compared with the union over detectors."""
    prior = prior_improved_sets(lines_dir, with_clean_label_features)
    mine = {}
    for r in rows:
        if r['n_pairs_run'] == 0:
            continue
        hw = r['handling']
        if r['variant'] == 'cliff molecules put back':
            hw = 'put cliff molecules back after ' + hw
        k = (r['experiment'], r['detector'], hw, r['dataset'], r['grid'])
        mine.setdefault(k, set()).update(r['pairs_improved'])
    # n25 setting-free: union prior over detectors
    for k in list(mine):
        if k[0] == 'n25' and k[1] in (ASSAY_LIMIT, EVERY_LABEL):
            u = set()
            for pk, v in prior.items():
                if pk[0] == 'n25' and pk[2] == k[2] and pk[3] == k[3] and pk[4] == k[4]:
                    u |= v
            prior[k] = u
    keys = sorted(set(mine) | {k for k in prior if not (k[0] == 'n25' and k[2].endswith(('constant', "forest-neighbours' mean")))})
    same, diff = 0, []
    for k in keys:
        a, b = mine.get(k, set()), prior.get(k, set())
        if a == b:
            same += 1
        else:
            diff.append((k, len(a), len(b), sorted(a - b), sorted(b - a)))
    return same, diff


if __name__ == '__main__':
    import collections
    REPO = pathlib.Path('~/repos/KIRBy').expanduser()
    ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/seagate/kirby_results_2026-10-08')
    LINES = '/Volumes/seagate/claude_analysis_2026-10-08/answers/_noise_lines'
    MISS = []
    rows = load_f8(ROOT, MISS.append)
    print(f'{len(rows)} cells')
    meth = sorted({r['method_label'] for r in rows})
    print(f'{len(meth)} methods over {len({r["experiment"] for r in rows})} experiments; '
          f'datasets {sorted({r["dataset"] for r in rows})}')
    print('cells by grid:', dict(collections.Counter(r['grid'] for r in rows)))
    print('cells that ran no pair (hatched):', sum(1 for r in rows if r['n_pairs_run'] == 0))
    print('\nsample:')
    for r in [x for x in rows if x['experiment'] == 'n24' and x['dataset'] == 'caco2'][:2]:
        print(json.dumps(r, indent=1))
    print(f'\n{len(MISS)} missing messages; first 10:')
    for m in MISS[:10]:
        print('  ', m[:220])
    if os.path.isdir(LINES):
        for wc in (False, True):
            same, diff = agreement(rows, LINES, wc)
            print(f"\nagreement with _noise_lines ({'with' if wc else 'without'} n07 'features from clean labels' lines): "
                  f'{same} of {same + len(diff)} (experiment, method, dataset, grid) cells give the same improved pairs')
            byexp = collections.Counter(d[0][0] for d in diff)
            print('   disagreeing cells by experiment:', dict(byexp))
            for d in diff[:12]:
                print('   ', d[0], f'mine {d[1]} vs lines {d[2]}; only mine {d[3]}; only lines {d[4]}')
