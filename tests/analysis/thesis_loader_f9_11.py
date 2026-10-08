"""Data loaders for thesis figures F9, F10 and F11 (Chapter 7, label noise).

Read-only. No model is fitted. Every value is a number the experiment itself
wrote; the only arithmetic is a mean over the noise draws of one fold, the
same mean the experiment's own summary takes.

F9   n30_censored_likelihood   the censoring ladder: labels as delivered,
                               clipped, clipped then corrected, clipped
                               molecules deleted. Held-out score per model and
                               configuration at each damaged censoring level.
F10  n32_iterative_and_dual_selection
                               detection AUROC of a detector against the
                               held-out score of the repair driven by that
                               same detector. Only three detector families in
                               n32 drive a repair (see load_f10).
F11  n07_remediation           held-out score at every damage level, for no
                               cleaning and each cleaning method.

Paths come from results_root only. The cut-dataset lists are read from
tests/experiments/orchestrate/plan.py, and partial checkpoints are decoded
with tests/experiments/_checkpoint.decode, both under the repository root.
"""
import ast
import glob
import importlib.util
import json
import os
import re
from pathlib import Path

INPUTS_F9 = ['noise/n30_censored_likelihood/*.json']
INPUTS_F10 = ['noise/n32_iterative_and_dual_selection/*.json',
              'noise/n32_iterative_and_dual_selection/_checkpoints/*.json']
INPUTS_F11 = ['noise/n07_remediation/*.json']

# Set by __main__ when this file is not inside the repository.
REPO_ROOT = None

CONFIGURATIONS = ('base_1', 'base_2', 'hybrid', 'hybrid__general2')


# --------------------------------------------------------------------------- shared

def _repo():
    return Path(REPO_ROOT) if REPO_ROOT else Path(__file__).resolve().parents[2]


_CUT = None


def _cut():
    """CUT_DATASETS + EXCLUDED_DATASETS, parsed from plan.py without importing it."""
    global _CUT
    if _CUT is None:
        src = (_repo() / 'tests/experiments/orchestrate/plan.py').read_text()
        out = set()
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if set(names) & {'CUT_DATASETS', 'EXCLUDED_DATASETS'}:
                    out |= set(ast.literal_eval(node.value))
        _CUT = frozenset(out)
    return _CUT


_DECODE = None


def _decode(obj):
    global _DECODE
    if _DECODE is None:
        p = _repo() / 'tests/experiments/_checkpoint.py'
        spec = importlib.util.spec_from_file_location('_kirby_checkpoint', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DECODE = mod.decode
    return _DECODE(obj)


def _split_name(path):
    """'<dataset>[__<kind>].json' -> (dataset, kind or None)."""
    stem = os.path.basename(path)[:-5]
    ds, _, kind = stem.partition('__')
    return ds, (kind or None)


def _result_files(results_root, pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(results_root, pattern))):
        b = os.path.basename(p)
        if 'fingerprint' in b or '.rows.' in b or b.endswith('.partial.jsonl'):
            continue
        out.append(p)
    return out


_PASS = re.compile(r'^l(?P<level>[0-9.eE+-]+)d(?P<draw>\d+)f(?P<fold>\d+)$')


def _by_fold(pass_scores):
    """{pass_key: score} -> {fold: (mean over draws, n_draws)}."""
    acc = {}
    for k, v in pass_scores.items():
        m = _PASS.match(str(k))
        fold = int(m.group('fold')) if m else None
        if v is None:
            continue
        acc.setdefault(fold, []).append(float(v))
    return {f: (sum(v) / len(v), len(v)) for f, v in acc.items()}


def _lvl(x):
    return float(x)


def _rel(results_root, p):
    return os.path.relpath(p, results_root)


# --------------------------------------------------------------------------- F9

F9_CORRECTIONS = ('constant_lift', 'per_molecule_lift', 'truncated_normal_mean',
                  'censored_em', 'winsor_c1', 'winsor_c1.25', 'winsor_c1.5')
F9_STEP_ARM = {'delivered': 'clean', 'clipped': 'clipped', 'deleted': 'drop_clipped'}
F9_PANELS = ('caco2', 'herg_ki')   # the outline's two panels


def load_f9(results_root, missing):
    """The censoring ladder from n30, one row per (dataset, damaged level, model,
    configuration, fold, step[, correction method]).

    step 'delivered' is n30's `clean` arm in the same pass: fitted on the
    training labels before the clip, in the same feature matrix as the other
    three steps. For a hybrid that matrix was rebuilt from the clipped labels
    (FAULTS.md A5), so `delivered` for a hybrid moves with the level; for a
    single representation it equals the level-0 score.
    step 'corrected' carries one row per correction method in
    F9_CORRECTIONS (correction_method = n30's arm name). The shuffled-lift
    control arms are not returned.
    Level 0 is not returned: nothing is clipped by the experiment there.
    """
    cut = _cut()
    rows = []
    seen = set()
    for pat in INPUTS_F9:
        for p in _result_files(results_root, pat):
            ds, _ = _split_name(p)
            if ds in cut:
                continue
            o = json.load(open(p))
            if not o.get('has_computed_results') or 'per_level' not in o:
                reason = o.get('skipped') or o.get('reason') or o.get('result_status')
                if reason != 'not_regression':
                    missing(f'F9 {ds}: n30 file holds no results ({reason}) [{_rel(results_root, p)}]')
                continue
            seen.add(ds)
            metric = o.get('metric_name') or o.get('metric')
            models = o.get('models') or []
            for L, by_sp in o['per_level'].items():
                if _lvl(L) == 0:
                    continue
                for sp in CONFIGURATIONS:
                    arms = by_sp.get(sp)
                    if arms is None:
                        missing(f'F9 {ds} level {L} {sp}: configuration absent, all models, all steps')
                        continue
                    for m in models:
                        steps = [(s, a, None) for s, a in F9_STEP_ARM.items()]
                        steps += [('corrected', c, c) for c in F9_CORRECTIONS]
                        for step, arm, corr in steps:
                            sc = (arms.get(arm) or {}).get(m)
                            if not sc:
                                missing(f'F9 {ds} level {L} {m} {sp}: step {step}'
                                        f'{" (" + corr + ")" if corr else ""} not in file (arm {arm})')
                                continue
                            for fold, (v, nd) in sorted(_by_fold(sc).items(), key=lambda t: (t[0] is None, t[0])):
                                rows.append(dict(
                                    dataset=ds, model=m, configuration=sp, step=step,
                                    correction_method=corr, arm=arm, metric=metric,
                                    lower_is_better=o.get('lower_is_better'),
                                    value=v, level=_lvl(L), n_draws=nd, fold=fold,
                                    source=_rel(results_root, p)))
    for ds in F9_PANELS:
        if ds not in seen:
            missing(f'F9 panel {ds}: no n30 result file with results under {INPUTS_F9[0]}')
    # A rows fingerprint without a result file means the unit started but did not finish.
    for fp in sorted(glob.glob(os.path.join(results_root, 'noise/n30_censored_likelihood/*fingerprint.json'))):
        ds = os.path.basename(fp).split('__')[0]
        if ds not in seen and ds not in cut:
            missing(f'F9 {ds}: only a rows fingerprint, no n30 result file [{_rel(results_root, fp)}]')
    return rows


# --------------------------------------------------------------------------- F10

N32_DETECTOR_LABEL = {
    'oof_residual': 'out-of-fold forest residual',
    'ngboost_rescaled': 'natural-gradient boosting residual / predicted spread',
    'quantile_forest_rescaled': 'quantile forest residual / leaf spread',
    'conformal': 'conformal score (residual / forest spread)',
    'cleanlab': 'cleanlab label quality',
}
F10_SHARES = ('0.05', '0.1', '0.2')
F10_ALPHAS = ('0.8', '0.9', '0.95')


def _det_label(d):
    if d in N32_DETECTOR_LABEL:
        return N32_DETECTOR_LABEL[d]
    m = re.match(r'knn_shapley@(.+?)\|.*\|k(\d+)$', d)
    if m:
        space = 'forest proximity' if m.group(1) == 'forest_proximity' else m.group(1)
        return f'Shapley value to a nearest-neighbour predictor, {space}, {m.group(2)} neighbours'
    m = re.match(r'itlm_a([0-9.]+)$', d)
    if m:
        return f'trimmed refit loss, keep share {m.group(1)} (last round)'
    m = re.match(r'two_learner_share([0-9.]+)$', d)
    if m:
        return f'two learners, larger of the two losses, flag share {m.group(1)}'
    return d


def _repair_label(arm):
    m = re.match(r'shrink_flagged_one_weight_share([0-9.]+)$', arm)
    if m:
        return f'shrink flagged {m.group(1)} toward out-of-fold prediction, one weight'
    m = re.match(r'shrink_flagged_per_row_share([0-9.]+)$', arm)
    if m:
        return f'shrink flagged {m.group(1)} toward out-of-fold prediction, weight per molecule'
    m = re.match(r'itlm_a([0-9.]+)$', arm)
    if m:
        return f'trimmed refit, keep {m.group(1)} of training molecules'
    m = re.match(r'two_learner_share([0-9.]+)$', arm)
    if m:
        return f'two learners, delete {m.group(1)} flagged by both'
    return arm


def _n32_units(results_root, missing, cut):
    """(dataset, kind) -> dict(src, acc, det, rec, final). Final file first, else checkpoint."""
    D = os.path.join(results_root, 'noise/n32_iterative_and_dual_selection')
    units = {}
    for p in _result_files(results_root, INPUTS_F10[0]):
        ds, kind = _split_name(p)
        if ds in cut:
            continue
        o = json.load(open(p))
        if o.get('has_computed_results') and 'per_level' in o:
            units[(ds, kind)] = dict(src=_rel(results_root, p), final=True, acc=o['per_level'],
                                     det=o.get('detection') or {}, rec=o.get('method_records') or {},
                                     metric=o.get('metric_name') or o.get('metric'),
                                     lib=o.get('lower_is_better'))
    for p in sorted(glob.glob(os.path.join(D, '_checkpoints', '*.json'))):
        ds, kind = _split_name(p)
        if ds in cut or (ds, kind) in units:
            continue
        o = _decode(json.load(open(p)))
        S = o.get('state') or {}
        if not S.get('acc'):
            missing(f'F10 {ds} {kind}: n32 checkpoint holds no finished pass [{_rel(results_root, p)}]')
            continue
        units[(ds, kind)] = dict(src=_rel(results_root, p) + ' (checkpoint, unit unfinished)', final=False,
                                 acc=S['acc'], det=S.get('det_acc') or {}, rec=S.get('rec_acc') or {},
                                 metric=None, lib=None)
    # Final files that hold no results and have no checkpoint: say why.
    for p in _result_files(results_root, INPUTS_F10[0]):
        ds, kind = _split_name(p)
        if ds in cut or (ds, kind) in units:
            continue
        o = json.load(open(p))
        reason = o.get('skipped') or o.get('reason') or o.get('result_status')
        if reason not in ('detection_undefined', 'not_regression'):
            missing(f'F10 {ds} {kind}: n32 file holds no results ({reason})')
    return units


def _auroc_by_fold(entry, final):
    """Detection entry -> {fold: (auroc, n_passes)}. Checkpoint: list of per-pass rows
    (fold not on the row, so all rows of the level are one fold group keyed None unless
    a pass key is present). Final file: one mean over every pass of the level."""
    if not entry:
        return {}
    if isinstance(entry, list):
        vals = [float(r['detection_auroc']) for r in entry if r.get('detection_auroc') is not None]
        return {None: (sum(vals) / len(vals), len(vals))} if vals else {}
    v = entry.get('detection_auroc')
    return {None: (float(v), entry.get('detection_n_passes'))} if v is not None else {}


def _rec_auroc_by_fold(recs, arm, field):
    """method_records rows -> {fold: (mean auroc over draws, n)} for one removal arm."""
    acc = {}
    for r in recs or []:
        d = (r.get(arm) or {}).get(field)
        if not d or d.get('detection_auroc') is None:
            continue
        m = _PASS.match(str(r.get('pass', '')))
        fold = int(m.group('fold')) if m else None
        acc.setdefault(fold, []).append(float(d['detection_auroc']))
    return {f: (sum(v) / len(v), len(v)) for f, v in acc.items()}


def load_f10(results_root, missing):
    """Detection AUROC against repair outcome, from n32.

    One row per (dataset, damage kind, level, model, configuration, detector,
    repair method). n32 grades seven detector families against the molecules the
    damage moved, but only these drive a repair whose held-out score is stored:
      oof_residual          -> shrink_flagged_one_weight_share<s>,
                               shrink_flagged_per_row_share<s>  (s = 0.05, 0.1, 0.2;
                               the flagged share is the top s by |y - p|, p the same
                               out-of-fold prediction the detector reads)
      itlm_a<alpha>         -> itlm_a<alpha> (its own last-round loss is the detector)
      two_learner_share<s>  -> two_learner_share<s> (larger of the two losses)
    knn_shapley, ngboost_rescaled, quantile_forest_rescaled, conformal and
    cleanlab are graded only; no repair reads them. Each is reported through
    missing(), once per (dataset, kind, level, model, configuration).

    repair_value is the repair's held-out score (the dataset's metric). Beside it:
    no_cleaning_value (same model on the damaged labels), clean_value (undamaged
    labels), best_random_value (lowest of the matched random-deletion seeds; removal
    arms only). The improvement is no_cleaning_value - repair_value for rmse; n32
    stores the two scores, not their difference.
    fold: AUROC from the checkpoint's detector list carries no fold, so fold is the
    repair's fold and the AUROC is the mean over that level's passes (every n32 unit
    so far has fold 0 only). In a final n32 file the AUROC is already a mean over
    every pass of the level.
    """
    cut = _cut()
    rows = []
    for (ds, kind), u in sorted(_n32_units(results_root, missing, cut).items(), key=lambda t: (t[0][0], t[0][1] or '')):
        acc, det, rec = u['acc'], u['det'], u['rec']
        metric = u['metric'] or 'rmse'
        for L in sorted(acc, key=_lvl):
            if _lvl(L) == 0:
                continue   # n32 grades no detector at level 0 (nothing moved)
            by_sp = acc[L]
            det_L = det.get(L, {})
            rec_L = rec.get(L, {})
            models = sorted({m for sp in by_sp.values() for m in (sp.get('no_cleaning') or {})})
            for sp in CONFIGURATIONS:
                arms = by_sp.get(sp) or {}
                for m in models:
                    ekey = f'{sp}/{m}' if f'{sp}/{m}' in det_L or f'{sp}/{m}' in rec_L else sp
                    base = _by_fold((arms.get('no_cleaning') or {}).get(m) or {})
                    clean = _by_fold((arms.get('clean') or {}).get(m) or {})
                    if not base:
                        missing(f'F10 {ds} {kind} {L} {m} {sp}: no no_cleaning score, every dot of this pair')
                        continue
                    dets = det_L.get(ekey, {})
                    recs = rec_L.get(ekey, [])

                    def emit(detector, auroc_fold, arm, field_note):
                        sc = _by_fold((arms.get(arm) or {}).get(m) or {})
                        if not sc:
                            missing(f'F10 {ds} {kind} {L} {m} {sp}: detector {detector} has AUROC but repair {arm} has no score')
                            return
                        if not auroc_fold:
                            missing(f'F10 {ds} {kind} {L} {m} {sp}: repair {arm} scored but detector {detector} has no AUROC ({field_note})')
                            return
                        rnd = []
                        for s in range(10):
                            r = _by_fold((arms.get(f'{arm}_random/seed{s}') or {}).get(m) or {})
                            if r:
                                rnd.append(r)
                        for fold, (v, nd) in sc.items():
                            a, na = auroc_fold.get(fold) or auroc_fold.get(None) or (None, None)
                            if a is None:
                                missing(f'F10 {ds} {kind} {L} {m} {sp} fold {fold}: {detector} AUROC absent for this fold')
                                continue
                            br = [r[fold][0] for r in rnd if fold in r]
                            rows.append(dict(
                                dataset=ds, damage_kind=kind, level=_lvl(L),
                                detector=detector, detector_label=_det_label(detector),
                                detection_auroc=a, detection_n_passes=na,
                                repair_method=arm, repair_label=_repair_label(arm),
                                repair_quantity_name=f'held-out {metric} after the repair',
                                repair_value=v, no_cleaning_value=base.get(fold, (None,))[0],
                                clean_value=clean.get(fold, (None,))[0],
                                best_random_value=min(br) if br else None,
                                n_random_seeds=len(br), metric=metric,
                                model=m, configuration=sp, n_draws=nd, fold=fold,
                                source=u['src']))

                    oof = _auroc_by_fold(dets.get('oof_residual'), u['final'])
                    for s in F10_SHARES:
                        for w in ('one_weight', 'per_row'):
                            emit('oof_residual', oof, f'shrink_flagged_{w}_share{s}', 'oof_residual')
                    for a in F10_ALPHAS:
                        arm = f'itlm_a{a}'
                        emit(arm, _rec_auroc_by_fold(recs, arm, 'detection_by_last_round_loss'), arm,
                             'method_records detection_by_last_round_loss')
                    for s in F10_SHARES:
                        arm = f'two_learner_share{s}'
                        emit(arm, _rec_auroc_by_fold(recs, arm, 'detection_by_larger_of_the_two_losses'), arm,
                             'method_records detection_by_larger_of_the_two_losses')
                    graded_only = sorted(d for d in dets if d != 'oof_residual')
                    if graded_only:
                        missing(f'F10 {ds} {kind} {L} {m} {sp}: {len(graded_only)} detectors graded with no repair '
                                f'reading them, no y value: {", ".join(graded_only)}')
    return rows


def n32_detection_auroc(results_root):
    """Every n32 detector AUROC per (dataset, kind, level, model, configuration),
    including the graded-only detectors. For checks and for an x-axis-only display."""
    cut = _cut()
    out = []
    for (ds, kind), u in _n32_units(results_root, lambda m: None, cut).items():
        for L, by in u['det'].items():
            for ekey, dets in by.items():
                if ekey.startswith('neighbour_spaces'):
                    continue
                sp, _, m = ekey.partition('/')
                for d, e in dets.items():
                    for fold, (a, n) in _auroc_by_fold(e, u['final']).items():
                        out.append(dict(dataset=ds, damage_kind=kind, level=_lvl(L), model=m or None,
                                        configuration=sp, detector=d, detector_label=_det_label(d),
                                        detection_auroc=a, n_passes=n, source=u['src']))
    return out


# --------------------------------------------------------------------------- F11

def _n07_label(s):
    base, _, share = s.partition('@flag_share')
    sh = f', flagged share {share}' if share else ''
    return {
        'no_noise': 'undamaged labels',
        'noisy_baseline': 'no cleaning',
        'remove': 'delete flagged molecules',
        'downweight': 'reweight by 1/(1 + score)',
        'label_smooth': "move labels toward neighbours' mean",
        'knn_smooth': "replace flagged labels by neighbours' mean",
        'oof_smooth': 'move labels toward out-of-fold prediction',
        'gp_smooth': 'move labels toward Gaussian-process mean',
        'js_shrink': 'score-scaled blend toward out-of-fold prediction',
        'constant_lift': 'raise labels at the assay limit',
        'huber_weight': 'Huber weights',
        'prox_smooth': "move labels toward forest-proximity neighbours' mean",
        'quantile_clip': 'clip onto leaf quantile range',
    }.get(base, base) + sh


def load_f11(results_root, missing):
    """Held-out score against damage level, from n07, one row per (dataset, damage
    kind, level, model, configuration, method, fold).

    method = n07's strategy name ('noisy_baseline' is no cleaning, 'no_noise' the
    undamaged labels). features_from: 'damaged labels' is n07's `scores` (each
    hybrid's features rebuilt from that draw's damaged labels, FAULTS.md A5);
    'clean labels' is `scores_clean_label_features`, the two hybrids with
    features built from the clean labels (hybrids only). Level 0 is the labels
    as delivered, one draw; every n07 detector is cross_val (out-of-fold forest
    residual) for these methods.
    """
    cut = _cut()
    rows = []
    for pat in INPUTS_F11:
        for p in _result_files(results_root, pat):
            ds, kind = _split_name(p)
            if ds in cut:
                continue
            o = json.load(open(p))
            if not o.get('has_computed_results') or 'by_level' not in o:
                missing(f'F11 {ds} {kind}: n07 file holds no results '
                        f'({o.get("skipped") or o.get("result_status")})')
                continue
            metric = o.get('metric_name') or o.get('metric')
            folds = [int(f) for f in (o.get('folds_run') or [0])]
            strategies = o.get('strategies') or []
            models = o.get('models') or []
            for L in sorted(o['by_level'], key=_lvl):
                b = o['by_level'][L]
                skipped = b.get('expensive_cleanups_skipped_width') or {}
                for feat_key, feat_label in (('scores', 'damaged labels'),
                                             ('scores_clean_label_features', 'clean labels')):
                    S = b.get(feat_key) or {}
                    for m in models:
                        spaces = CONFIGURATIONS if feat_key == 'scores' else ('hybrid', 'hybrid__general2')
                        for sp in spaces:
                            cell = (S.get(m) or {}).get(sp)
                            if cell is None:
                                missing(f'F11 {ds} {kind} level {L} {m} {sp} ({feat_label}): configuration absent')
                                continue
                            for s in strategies:
                                e = cell.get(s)
                                if not e:
                                    why = [k for k, v in skipped.items() if (v.get(m) or {}).get(sp)]
                                    reason = (f'expensive cleanup skipped at width {skipped[why[0]][m][sp]} features'
                                              if why and s in (o.get('expensive_cleanups') or []) else 'not in file')
                                    missing(f'F11 {ds} {kind} level {L} {m} {sp} ({feat_label}): {s} {reason}')
                                    continue
                                pf = e.get('per_fold') or [e.get('mean')]
                                for i, v in enumerate(pf):
                                    if v is None:
                                        missing(f'F11 {ds} {kind} level {L} {m} {sp}: {s} fold {i} is None')
                                        continue
                                    rows.append(dict(
                                        dataset=ds, damage_kind=kind, method=s, method_label=_n07_label(s),
                                        model=m, configuration=sp, features_from=feat_label,
                                        level=_lvl(L), metric=metric, lower_is_better=o.get('lower_is_better'),
                                        value=float(v), n_draws=b.get('n_draws'),
                                        fold=folds[i] if i < len(folds) else i,
                                        source=_rel(results_root, p)))
    return rows


# --------------------------------------------------------------------------- __main__

if __name__ == '__main__':
    import collections
    import sys
    REPO_ROOT = os.path.expanduser('~/repos/KIRBy')
    roots = sys.argv[1:] or ['/Volumes/seagate/kirby_results_2026-10-08']
    for root in roots:
        print(f'\n######## results_root = {root}')
        for name, fn in (('F9', load_f9), ('F10', load_f10), ('F11', load_f11)):
            miss = []
            rows = fn(root, miss.append)
            print(f'\n=== {name}: {len(rows)} rows, {len(miss)} missing marks')
            c = collections.Counter((r['dataset'], r.get('damage_kind')) for r in rows)
            for k, v in sorted(c.items(), key=lambda t: (t[0][0], t[0][1] or '')):
                print(f'   {k[0]} {k[1] or "censoring"}: {v} rows')
            for r in rows[:2]:
                print('   sample:', r)
            for msg in miss[:12]:
                print('   missing:', msg)
            if len(miss) > 12:
                print(f'   ... {len(miss) - 12} more missing')

            if name == 'F9':
                def g(ds, m, sp, step, corr=None, L=0.25):
                    for r in rows:
                        if (r['dataset'], r['model'], r['configuration'], r['step'], r['correction_method'], r['level']) == (ds, m, sp, step, corr, L):
                            return r['value']
                print('   CHECK outline: herg_ki dnn base_2 clipped 0.7485 -> constant_lift 0.6662 at 0.25:',
                      g('herg_ki', 'dnn', 'base_2', 'clipped'), '->', g('herg_ki', 'dnn', 'base_2', 'corrected', 'constant_lift'))
            if name == 'F10':
                # Outline: no removal or shrinking method beat no cleaning on half the pairs under any damage kind.
                win = collections.defaultdict(lambda: [0, set()])
                for r in rows:
                    k = (r['dataset'], r['damage_kind'], r['level'], r['repair_method'])
                    win[k][1].add((r['model'], r['configuration']))
                    if r['repair_value'] < r['no_cleaning_value']:
                        win[k][0] += 1
                worst = max(((v[0] / len(v[1]), k, v[0], len(v[1])) for k, v in win.items()), default=None)
                print('   CHECK outline "none beat no cleaning on half the pairs": highest share',
                      worst and f'{worst[2]} of {worst[3]} pairs, {worst[1]}')
                det = n32_detection_auroc(root)
                for L, dname in ((0.25, 'knn_shapley@forest_proximity|none|forest|k10'),
                                 (0.1, 'knn_shapley@forest_proximity|none|forest|k20')):
                    for d in (dname, 'oof_residual'):
                        v = [x['detection_auroc'] for x in det if x['dataset'] == 'caco2'
                             and x['damage_kind'] == 'censoring' and x['level'] == L and x['detector'] == d]
                        if v:
                            print(f'   CHECK caco2 censoring {L} {d}: AUROC {min(v):.4f} to {max(v):.4f} over {len(v)} pairs')
            if name == 'F11':
                c = collections.Counter((r['dataset'], r['damage_kind'], r['level']) for r in rows)
                lv = collections.defaultdict(set)
                for (ds, k, L) in c:
                    lv[(ds, k)].add(L)
                for k, v in sorted(lv.items()):
                    print(f'   levels {k[0]} {k[1]}: {sorted(v)}')
