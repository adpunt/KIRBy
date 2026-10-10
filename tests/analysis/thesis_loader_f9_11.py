"""Data loaders for thesis figures F9, F10 and F11 (Chapter 7, label noise).

Read-only. No model is fitted. Every value is a number the experiment itself
wrote; the only arithmetic is a mean over the noise draws of one fold, the
same mean the experiment's own summary takes.

F9   n30_censored_likelihood   the censoring ladder: labels as delivered,
                               clipped, clipped then corrected, clipped
                               molecules deleted. Held-out score per model and
                               configuration at each damaged censoring level.
F10  n32_iterative_and_dual_selection, n31_regime_and_breakeven
                               precision of the molecules the two-learner
                               deletion removed against the held-out score
                               after removing them (n32), with n31's
                               break-even precision for deleting a flagged
                               set of the same share (see load_f10).
F11  n24_delete_or_keep, n30_censored_likelihood
                               hERG under label flips: held-out score against the
                               share of molecules flagged (n24). Caco-2 under
                               clipping: held-out score against the clipped share
                               (n30).

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


F10_ARM = 'two_learner_share{s}'
F10_DAMAGE_WITH_BREAKEVEN = ('shift_only_p10', 'sparse_dprime_up_p10')
INPUTS_F10_BREAKEVEN = ['noise/n31_regime_and_breakeven/*.json']


def _tl_precision_by_fold(recs, arm):
    """method_records rows -> {fold: dict} for one two-learner arm.

    precision = moved_molecules_among_the_removed / removed, removed =
    training_molecules - kept_by_both_and_trained_on (the molecules the arm deleted).
    Mean over the noise draws of the fold; the counts are summed over the same draws."""
    acc = {}
    for r in recs or []:
        d = r.get(arm) or {}
        n_tr, kept, hit = d.get('training_molecules'), d.get('kept_by_both_and_trained_on'), \
            d.get('moved_molecules_among_the_removed')
        if n_tr is None or kept is None or hit is None or n_tr - kept <= 0:
            continue
        m = _PASS.match(str(r.get('pass', '')))
        fold = int(m.group('fold')) if m else None
        det = d.get('detection_by_larger_of_the_two_losses') or {}
        a = acc.setdefault(fold, dict(prec=[], removed=0, hits=0, train=0, moved=0, ap=[], auroc=[]))
        a['prec'].append(hit / float(n_tr - kept))
        a['removed'] += n_tr - kept
        a['hits'] += hit
        a['train'] += n_tr
        a['moved'] += r.get('molecules_moved') or 0
        if det.get('detection_average_precision') is not None:
            a['ap'].append(float(det['detection_average_precision']))
        if det.get('detection_auroc') is not None:
            a['auroc'].append(float(det['detection_auroc']))
    out = {}
    for f, a in acc.items():
        n = len(a['prec'])
        out[f] = dict(precision=sum(a['prec']) / n, n_precision_draws=n,
                      removed_mean=a['removed'] / n, removed_hits_mean=a['hits'] / n,
                      training_molecules_mean=a['train'] / n, molecules_moved_mean=a['moved'] / n,
                      average_precision=(sum(a['ap']) / len(a['ap'])) if a['ap'] else None,
                      detection_auroc=(sum(a['auroc']) / len(a['auroc'])) if a['auroc'] else None)
    return out


def load_f10(results_root, missing):
    """Precision of the deleted molecules against the held-out score after deleting them, from n32.

    One row per (dataset, damage kind, level, model, configuration, flagged share, fold).
    Only the two-learner deletion is returned (two_learner_share<s>, s = 0.05, 0.1, 0.2):
    it is the one detector-and-action pair in n32 that Chapter 7 keeps (THESIS_OUTLINE.md
    "CHAPTER 7 METHODS, SIGNED OFF": two_learner BROAD; itlm and both shrinks CUT). Its
    detector is the larger of the two learners' losses; its action deletes the molecules
    either learner flags, so the deleted share is somewhat above s (removed_mean).

    precision      share of the deleted molecules whose label the damage moved, read from
                   method_records (moved_molecules_among_the_removed over the deleted count),
                   mean over the draws of the fold.
    after_value    the held-out score after the deletion (the dataset's metric).
    no_cleaning_value, clean_value: same model on the damaged and on the undamaged labels.
    """
    cut = _cut()
    rows = []
    for (ds, kind), u in sorted(_n32_units(results_root, missing, cut).items(), key=lambda t: (t[0][0], t[0][1] or '')):
        acc, rec = u['acc'], u['rec']
        metric = u['metric'] or 'rmse'
        for L in sorted(acc, key=_lvl):
            if _lvl(L) == 0:
                continue   # nothing moved at level 0, so a deleted set has no precision
            by_sp = acc[L]
            rec_L = rec.get(L, {})
            models = sorted({m for sp in by_sp.values() for m in (sp.get('no_cleaning') or {})})
            for sp in CONFIGURATIONS:
                arms = by_sp.get(sp) or {}
                for m in models:
                    ekey = f'{sp}/{m}' if f'{sp}/{m}' in rec_L else sp
                    base = _by_fold((arms.get('no_cleaning') or {}).get(m) or {})
                    clean = _by_fold((arms.get('clean') or {}).get(m) or {})
                    if not base:
                        continue
                    recs = rec_L.get(ekey, [])
                    for s in F10_SHARES:
                        arm = F10_ARM.format(s=s)
                        sc = _by_fold((arms.get(arm) or {}).get(m) or {})
                        pr = _tl_precision_by_fold(recs, arm)
                        if not sc or not pr:
                            missing(f'F10 {ds} {kind} {L} {m} {sp}: {arm} has '
                                    f'{"no score" if not sc else "no deleted-set counts"}')
                            continue
                        for fold, (v, nd) in sc.items():
                            p = pr.get(fold) or pr.get(None)
                            if p is None:
                                missing(f'F10 {ds} {kind} {L} {m} {sp} fold {fold}: {arm} counts absent')
                                continue
                            rows.append(dict(
                                dataset=ds, damage_kind=kind, level=_lvl(L),
                                detector='two_learner', action='delete', flag_share=float(s),
                                method=arm, precision=p['precision'],
                                removed_mean=p['removed_mean'], removed_hits_mean=p['removed_hits_mean'],
                                training_molecules_mean=p['training_molecules_mean'],
                                molecules_moved_mean=p['molecules_moved_mean'],
                                average_precision=p['average_precision'], detection_auroc=p['detection_auroc'],
                                after_value=v, no_cleaning_value=base.get(fold, (None,))[0],
                                clean_value=clean.get(fold, (None,))[0], metric=metric,
                                model=m, configuration=sp, n_draws=nd, fold=fold,
                                unit_finished=u['final'], source=u['src']))
    return rows


def load_f10_breakeven(results_root, missing):
    """Break-even precision for deleting a flagged set, from n31's final files.

    {(dataset, damage kind, level, configuration, model, share): dict(value, refused, source,
    detector_precision)}. value is summary.per_condition.<kind>.<lvl>.<cfg>.per_model.<m>
    .synthetic.share<s>.remove.break_even_precision (None with the reason when n31 found no
    crossing). Only the damage kinds n32 also runs (F10_DAMAGE_WITH_BREAKEVEN): n31's outlier
    kind is outlier_only_p05 (only the 5% move), n32's is outlier_p05 (every label moves), so
    they are not the same damage. A dataset with n31 rows but no final file is reported."""
    out = {}
    cut = _cut()
    D = os.path.join(results_root, 'noise/n31_regime_and_breakeven')
    finals = set()
    for p in _result_files(results_root, INPUTS_F10_BREAKEVEN[0]):
        ds, kind = _split_name(p)
        if ds in cut or kind:
            continue
        o = json.load(open(p))
        summ = (o.get('summary') or {}).get('per_condition')
        if not summ:
            continue
        finals.add(ds)
        for cond in F10_DAMAGE_WITH_BREAKEVEN:
            for L, lv in (summ.get(cond) or {}).items():
                for sp, spv in lv.items():
                    for m, mv in (spv.get('per_model') or {}).items():
                        for shk, ops in (mv.get('synthetic') or {}).items():
                            e = ops.get('remove')
                            if not e:
                                continue
                            out[(ds, cond, _lvl(L), sp, m, float(shk.replace('share', '')))] = dict(
                                value=e.get('break_even_precision'),
                                refused=e.get('break_even_refused_because'),
                                detector_precision=e.get('detector_precision_at_this_share_mean'),
                                n_folds=len(o.get('folds_run') or []),
                                source=_rel(results_root, p))
    for p in sorted(glob.glob(os.path.join(D, '*.rows.jsonl'))):
        ds = os.path.basename(p).split('__')[0]
        if ds not in finals and ds not in cut:
            missing(f'F10 break-even {ds}: n31 has rows but no final file, so no break-even precision '
                    f'[{_rel(results_root, p)}]')
    return out


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

# hERG label flips (n24) and Caco-2 clipping (n30), fold 0 by default.
INPUTS_F11_FLIPS = ['noise/n24_delete_or_keep/herg__*.json']
F11_FLIP_ARMS = {   # n24 arm -> (method key, words). delete_at_random seeds become one 'best random deletion'.
    'no_cleaning': ('no_cleaning', 'no cleaning'),
    'delete/remove': ('delete', 'delete flagged molecules'),
    'keep_rewrite/label_smooth': ('label_smooth', "move flagged labels toward neighbors' mean"),
    'keep_reweight/downweight': ('downweight', 'down-weight by suspicion score'),
}
F11_CLIP_ARMS = {   # n30 arm -> (method key, words)
    'clipped': ('no_cleaning', 'no cleaning'),
    'censored_em': ('censored_em', 'censored EM on clipped labels'),
    'constant_lift': ('constant_lift', 'raise clipped labels by one constant'),
    'drop_clipped': ('drop_clipped', 'delete clipped molecules'),
}


def _f11_flip_rows(results_root, missing, kind, model, detector, spaces, fold):
    """n24 hERG flips: score against the share of training molecules flagged, at the
    file's one damaged level. One row per (configuration, method, flagged share).
    Share 0 is no cleaning. down-weighting does not read the flagged share (n24 fits it
    once and repeats the score), so it is returned at every share with rate_blind=True.
    best_random_deletion is the best of the five random-deletion seeds (each seed's mean over
    draws), picked on the held-out score, so it leans in the control's favour."""
    rows = []
    stage = 'noise' if fold == 0 else 'noise_full'   # noise_full holds the promoted folds 1-2
    paths = [p for p in _result_files(results_root, f'{stage}/n24_delete_or_keep/herg__*.json')
             if _split_name(p) == ('herg', kind)]
    if not paths:
        missing(f'F11 herg {kind}: no n24 result file')
        return rows
    p = paths[0]
    o = json.load(open(p))
    if not o.get('has_computed_results'):
        missing(f'F11 herg {kind}: n24 file holds no results ({o.get("result_status")})')
        return rows
    levels = sorted((o.get('per_level') or {}), key=_lvl)
    damaged = [L for L in levels if _lvl(L) > 0]
    metric = o.get('metric_name') or o.get('metric')
    for L in levels:
        for sp in spaces:
            det = ((o['per_level'][L].get(sp) or {}).get(detector))
            if not det:
                missing(f'F11 herg {kind} level {L} {sp}: detector {detector} absent')
                continue
            for fr in sorted(det, key=float):
                arms = det[fr]
                def score(arm):
                    sc = (arms.get(arm) or {}).get(model)
                    if not sc:
                        return None, 0
                    v = _by_fold(sc).get(fold)
                    return v if v else (None, 0)
                for arm, (key, words) in F11_FLIP_ARMS.items():
                    v, nd = score(arm)
                    if v is None:
                        missing(f'F11 herg {kind} level {L} {sp} {model} share {fr}: {key} not in file')
                        continue
                    share = 0.0 if key == 'no_cleaning' else float(fr)
                    rows.append(dict(panel=f'herg:{kind}', dataset='herg', damage_kind=kind, damage_level=_lvl(L),
                                     x=share, x_meaning='share of training molecules flagged',
                                     method=key, method_label=words, rate_blind=(key == 'downweight'),
                                     model=model, configuration=sp, detector=detector, metric=metric,
                                     lower_is_better=o.get('lower_is_better'), value=v, n_draws=nd,
                                     fold=fold, source=_rel(results_root, p)))
                seeds = []
                for arm in arms:
                    if arm.startswith('delete_at_random/'):
                        v, nd = score(arm)
                        if v is not None:
                            seeds.append((v, nd, arm.split('/')[-1]))
                if not seeds:
                    missing(f'F11 herg {kind} level {L} {sp} {model} share {fr}: no random-deletion seed')
                    continue
                best = (min if o.get('lower_is_better') else max)(seeds)
                rows.append(dict(panel=f'herg:{kind}', dataset='herg', damage_kind=kind, damage_level=_lvl(L),
                                 x=float(fr), x_meaning='share of training molecules flagged',
                                 method='best_random_deletion',
                                 method_label='best of five random deletions, same number',
                                 rate_blind=False, model=model, configuration=sp, detector=detector,
                                 metric=metric, lower_is_better=o.get('lower_is_better'), value=best[0],
                                 n_draws=best[1], n_seeds=len(seeds), best_seed=best[2], fold=fold,
                                 source=_rel(results_root, p)))
    if len(damaged) != 1:
        missing(f'F11 herg {kind}: n24 holds damaged levels {damaged}; the panel draws each separately')
    return rows


def _f11_clip_rows(results_root, missing, dataset, model, spaces, fold):
    """n30 clipping: score against the clipped share, one row per (configuration, method,
    level). At level 0 n30 still treats the molecules whose delivered label equals the
    largest label as clipped, so each method has its own level-0 score."""
    rows = []
    stage = 'noise' if fold == 0 else 'noise_full'
    paths = [p for p in _result_files(results_root, f'{stage}/n30_censored_likelihood/*.json')
             if _split_name(p)[0] == dataset]
    if not paths:
        missing(f'F11 {dataset} censoring: no n30 result file')
        return rows
    p = paths[0]
    o = json.load(open(p))
    if not o.get('has_computed_results') or 'per_level' not in o:
        missing(f'F11 {dataset} censoring: n30 file holds no results ({o.get("result_status")})')
        return rows
    metric = o.get('metric_name') or o.get('metric')
    for L in sorted(o['per_level'], key=_lvl):
        for sp in spaces:
            arms = o['per_level'][L].get(sp)
            if arms is None:
                missing(f'F11 {dataset} censoring level {L} {sp}: configuration absent')
                continue
            for arm, (key, words) in F11_CLIP_ARMS.items():
                sc = (arms.get(arm) or {}).get(model)
                v = _by_fold(sc).get(fold) if sc else None
                if not v:
                    missing(f'F11 {dataset} censoring level {L} {sp} {model}: {key} not in file')
                    continue
                rows.append(dict(panel=f'{dataset}:censoring', dataset=dataset, damage_kind='censoring',
                                 damage_level=_lvl(L), x=_lvl(L), x_meaning='share of training labels clipped',
                                 method=key, method_label=words, rate_blind=False, model=model,
                                 configuration=sp, detector='', metric=metric,
                                 lower_is_better=o.get('lower_is_better'), value=v[0], n_draws=v[1],
                                 fold=fold, source=_rel(results_root, p)))
    return rows


INPUTS_F11 = INPUTS_F11_FLIPS + ['noise/n30_censored_likelihood/*.json']


def load_f11(results_root, missing, panels=('herg:confusion_directed', 'herg:uniform', 'caco2:censoring'),
             flip_model='xgboost', flip_detector='neighbor_residual@screen1', clip_model='dnn',
             spaces=('hybrid', 'base_1'), fold=0):
    """F11 rows. A panel 'herg:<flip kind>' reads n24 (x = share of training molecules
    flagged, at the file's damaged level); '<dataset>:censoring' reads n30 (x = clipped share).
    spaces: 'hybrid' is the top hybrid, 'base_1' the best single representation, both chosen
    on the fold's training molecules."""
    rows = []
    for panel in panels:
        ds, kind = panel.split(':')
        if ds in _cut():
            missing(f'F11 {panel}: dataset is cut')
            continue
        if kind == 'censoring':
            rows += _f11_clip_rows(results_root, missing, ds, clip_model, spaces, fold)
        elif ds == 'herg':
            rows += _f11_flip_rows(results_root, missing, kind, flip_model, flip_detector, spaces, fold)
        else:
            missing(f'F11 {panel}: no reader for this panel')
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
                be = load_f10_breakeven(root, miss.append)
                print(f'   break-even entries: {len(be)}')
                for k in sorted(be, key=str)[:6]:
                    print('   break-even', k, be[k]['value'], be[k]['refused'])
            if name == 'F11':
                lv = collections.defaultdict(set)
                for r in rows:
                    lv[(r['dataset'], r['damage_kind'])].add(r['x'])
                for k, v in sorted(lv.items()):
                    print(f'   levels {k[0]} {k[1]}: {sorted(v)}')
