"""The similarity axis, the D-versus-D' rows, the controls and the per-fold
checkpoint, as n15-n22 use them.

Written 2026-09-25 while wiring n15, n16, n17, n18, n19, n20, n21 and n22 into
plan sections 1-3 (~/.claude/plans/pasted-content-id-0428-noise-experiment-
fizzy-ladybug.md). Everything here calls the shared modules the plan built --
similarity_space, noise_cache, d_vs_dprime, controls -- and adds only the glue
those eight files would otherwise each carry a copy of:

  sweep_settings        which similarity settings the run sweeps, from
                        KIRBY_SIMILARITY_SETTINGS (see its docstring for the
                        one rule added on top of settings_from_env)
  build_spaces          one downstream model's similarity spaces for one fold:
                        its top 2 single representations, each of those plus
                        augmented features (c21's capped and uncapped), its
                        prediction-built and similarity-built hybrids, and the
                        model-input matrix itself
  neighbour_lists       noise_cache.cached_neighbours, returning (lists, None)
                        or (None, why)
  Rows                  one JSONL file of d_vs_dprime rows per (experiment,
                        dataset[, condition]), resumed through done_keys
  fit_score / dprime_rows / control_rows
                        the D, D' and clean fits of one row, cached through
                        noise_cache.cached_fit_predict for the seeded models
  FoldCheckpoint        the accumulators an experiment holds after each fold,
                        pickled, so a killed run resumes at the next fold

⛔ NOT a shared module of the suite. It lives beside the eight files it serves
because the wiring brief allowed edits to those eight and nothing else. It
belongs in similarity_space.py / d_vs_dprime.py once the other wiring agents'
files settle; TODO(similarity-sweep 2026-09-25) move it there.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.noise import similarity_space as ss  # noqa: E402
from experiments.noise import noise_cache as nc  # noqa: E402
from experiments.noise import d_vs_dprime as dvd  # noqa: E402
from experiments.noise.controls import (  # noqa: E402
    controls_enabled, random_twin, shuffled_twin)


# The similarity_setting written on a row whose D' does not depend on any
# neighbour list: the method reads none and neither does the score driving it.
NO_SIMILARITY = 'none'

# The similarity_setting written on a row computed by the experiment's own,
# unchanged code path: neighbours in the model-input matrix under the
# process-wide distance, each method at the neighbour count it uses today (10
# for the four neighbour estimators, 9 for label_smooth / knn_smooth,
# n07:829-855, 10 for prox_smooth, n07:907). 'k=default' because those counts
# differ between methods of one row, so no single k is true of it.
TODAY = f'{ss.MODEL_INPUT.name}|k=default'

# n07's cleanups that read a neighbour list (n07.cleaned_training_set,
# similarity_neighbours=). prox_smooth reads one only under forest_proximity.
SMOOTHERS_READING_NEIGHBOURS = ('label_smooth', 'knn_smooth')

# Downstream models whose factory seeds every fit (find_best_hybrid.py:2461-
# 2519, 2805: random_state=42), so a cached prediction is the prediction a
# refit would give. dnn and the torch learners are fitted directly, never
# cached: noise_cache.cached_fit_predict's docstring forbids an unreproducible
# factory under a name that says otherwise.
CACHEABLE_MODELS = ('rf', 'svm', 'lightgbm', 'xgboost')


# ---------------------------------------------------------------------------
# Which settings
# ---------------------------------------------------------------------------

def sweep_settings(available_spaces=None, env=None):
    """The similarity settings the sweep loops over, [] for today's run only.

    similarity_space.settings_from_env, plus one rule. KIRBY_SIMILARITY_SETTINGS
    unset or 'all' is the full sweep, and enumerate_settings leaves MODEL_INPUT
    out of it ("a caller adds it when it wants today's behaviour as a row",
    similarity_space.py:190). The owner asked for every k in use today to be
    swept (2026-09-25), and neighbours in the model-input matrix are what every
    method uses today, so under 'all' MODEL_INPUT is put first: 67 settings,
    not 66. 'model_input' alone returns [], because that setting IS the
    experiment's unchanged code path, which always runs; sweeping it at the
    other ks would be more than today's run. A comma list is taken as written.
    """
    e = os.environ if env is None else env
    got = ss.settings_from_env(available_spaces, env=e)
    if len(got) == 1 and got[0].space == ss.MODEL_INPUT_SPACE:
        return []
    raw = str(e.get('KIRBY_SIMILARITY_SETTINGS') or '').strip()
    if raw in ('', 'all'):
        got = [ss.MODEL_INPUT] + list(got)
    return got


def sim_name(setting, k):
    """The row's similarity_setting: '<space>|<weighting>|<metric>|k=<k>'.

    k is folded into this column because d_vs_dprime.ROW_FIELDS has no k
    column and make_row refuses an unknown one; every row must say its k.
    """
    if setting is None:
        return NO_SIMILARITY
    return f'{setting.name}|k={int(k)}'


def strategy_reads_setting(strategy, setting):
    """Does n07.cleaned_training_set read the setting's neighbours for this
    cleanup? label_smooth and knn_smooth always; prox_smooth only under
    forest_proximity (n07.cleaned_training_set refuses any other pairing)."""
    if setting is None:
        return False
    if strategy in SMOOTHERS_READING_NEIGHBOURS:
        return True
    if strategy == 'prox_smooth':
        return setting.space == ss.FOREST_SPACE
    return False


def ks_held(nbrs, ks=None):
    """The ks of SWEPT_KS these lists can serve exactly (neighbours_at)."""
    ks = ss.SWEPT_KS if ks is None else ks
    exact = nbrs.get('exact_by_k')
    if exact is not None:
        return [int(k) for k in ks if int(k) in exact]
    return [int(k) for k in ks if int(k) <= int(nbrs.get('k_eff') or 0)]


def seed_of(*parts):
    """A tuple of ints for controls._rng from any mix of ints and strings."""
    out = []
    for p in parts:
        if isinstance(p, (int, np.integer)):
            out.append(int(p))
        else:
            out.append(zlib.crc32(str(p).encode()) & 0x7FFFFFFF)
    return tuple(out)


# ---------------------------------------------------------------------------
# The spaces
# ---------------------------------------------------------------------------

def _scale(X_tr, X_va=None):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    A = sc.fit_transform(np.asarray(X_tr, dtype=np.float64))
    return A, (None if X_va is None
               else sc.transform(np.asarray(X_va, dtype=np.float64)))


def build_spaces(dataset, settings, *, X_model_tr, train_reps, val_reps,
                 train_idx, base_names, recipes, y_train, groups_train,
                 model_name, model_factory, metric, task_type,
                 augmentations_train=None, augmentations_val=None,
                 X_model_va=None, want_val=False, hybrid_augmentations=False):
    """One downstream model's similarity spaces for one fold.

    Only the spaces some setting in `settings` reads are built. Every matrix is
    z-scored on the fold's training rows (similarity_space.neighbour_lists
    requires standardised input, preprocessing.py:396-401).

    Args:
        X_model_tr: the matrix the model is fitted on, as the experiment holds
            it ('model_input'; read by MODEL_INPUT and forest_proximity).
        train_reps / val_reps: {representation: matrix} sliced to this fold.
        train_idx: the fold's training row indices, for the hybrid cache key.
        base_names: this model's top-2 single representations, rank order
            (_shared.noise_base_reps_by_model).
        recipes: {'hybrid': Blueprint, 'hybrid_similarity': Blueprint}, this
            model's prediction-built and similarity-built recipes.
        y_train: the labels the experiment holds for the training rows (noisy
            where noise was injected). Every build ranks features on these.
        metric: the get_domain_metric triple.
        augmentations_train / _val: {name: matrix} donors sliced to this fold,
            for the four augmented spaces. None leaves them unavailable.
        hybrid_augmentations: hand those donors to the two hybrid builds too.
            Set it exactly when the experiment's own model-input hybrid build
            passes them (n21 does, n17-n20 do not), so the similarity space is
            the same build as the model input and the cache serves both.

    Returns (spaces_train, spaces_val or None, unavailable) where
    `unavailable` is {space: why}. A space that cannot be built is never
    faked; the caller skips every setting in it and records the reason.
    """
    from experiments.noise.noise_helpers import recipe_is_geometry
    need = {s.space for s in settings}
    metric_name, metric_fn, lib = metric
    tr = {'model_input': X_model_tr}
    va = {'model_input': X_model_va} if want_val else None
    unavailable = {}
    base_names = list(base_names or [])

    for i in (1, 2):
        sp = f'base_{i}'
        aug_sps = [ss.aug_space_name(i, st) for st in ss.AUG_STRATEGIES
                   if ss.aug_space_name(i, st) in need]
        if sp not in need and not aug_sps:
            continue
        if len(base_names) < i or base_names[i - 1] not in train_reps:
            why = (f'{model_name} has no rank-{i} single representation in '
                   f'this run')
            for s_ in [sp] + aug_sps:
                unavailable[s_] = why
            continue
        rn = base_names[i - 1]
        if sp in need:
            a, b = _scale(train_reps[rn], val_reps.get(rn) if want_val else None)
            tr[sp] = a
            if want_val:
                va[sp] = b
        for st in ss.AUG_STRATEGIES:
            asp = ss.aug_space_name(i, st)
            if asp not in need:
                continue
            # The donors, minus the base itself (c21:429).
            donors_tr = {a: m for a, m in (augmentations_train or {}).items()
                         if a != rn}
            if not donors_tr:
                unavailable[asp] = ('no augmentation representation is loaded '
                                    'for this dataset besides the base')
                continue
            donors_va = {a: m for a, m in (augmentations_val or {}).items()
                         if a != rn}
            bp = ss.base_aug_blueprint(
                rn, int(np.asarray(train_reps[rn]).shape[1]),
                metric_name=metric_name, task_type=task_type,
                lower_is_better=lib, strategy=st)
            got = nc.cached_hybrid_for_fold(
                dataset, {rn: train_reps[rn]},
                {rn: val_reps.get(rn, train_reps[rn])}, y_train, bp,
                groups_train, None if model_factory is None else model_factory(),
                metric_fn, lib,
                model_name=model_name, metric_name=metric_name,
                train_idx=train_idx, augmentations_train=donors_tr,
                augmentations_val=donors_va or donors_tr)
            if got[0] is None:
                unavailable[asp] = 'the c21-style augmented build failed'
                continue
            a, b = _scale(got[0], got[1] if want_val else None)
            tr[asp] = a
            if want_val:
                va[asp] = b

    for sp in ('hybrid', 'hybrid_similarity'):
        if sp not in need:
            continue
        bp = (recipes or {}).get(sp)
        if bp is None:
            unavailable[sp] = f'{model_name} has no {sp} recipe in the pool'
            continue
        geometry = recipe_is_geometry(bp)
        got = nc.cached_hybrid_for_fold(
            dataset, train_reps, val_reps, y_train, bp, groups_train,
            None if (geometry or model_factory is None) else model_factory(),
            metric_fn, lib,
            model_name=None if geometry else model_name,
            metric_name=metric_name, train_idx=train_idx,
            augmentations_train=(augmentations_train if hybrid_augmentations
                                 else None),
            augmentations_val=(augmentations_val if hybrid_augmentations
                               else None))
        if got[0] is None:
            unavailable[sp] = 'hybrid_for_fold could not build this fold'
            continue
        a, b = _scale(got[0], got[1] if want_val else None)
        tr[sp] = a
        if want_val:
            va[sp] = b
    return tr, va, unavailable


def neighbour_lists(dataset, setting, spaces_train, y_train, *,
                    exclude_groups=None, smiles_train=None, is_clf=False):
    """(lists, None) or (None, why). Training-row lists only: every consumer
    in n15-n22 reads neighbours among the training rows."""
    try:
        nb = nc.cached_neighbours(
            dataset, setting, spaces_train=spaces_train, spaces_val=None,
            y_train=y_train, exclude_groups=exclude_groups,
            smiles_train=smiles_train, is_clf=is_clf)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    if not nb or not nb.get('available'):
        return None, (nb or {}).get('reason', 'unavailable')
    return nb, None


def smiles_for(data, rows):
    """The SMILES of `rows`, or None when the loader carries none."""
    sm = data.get('smiles')
    if sm is None:
        return None
    sm = np.asarray(sm, dtype=object)
    if len(sm) == 0:
        return None
    return sm[np.asarray(rows)]


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _clean_stem(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_')


class Rows:
    """d_vs_dprime rows for one (experiment, dataset[, condition]), on disk at
    <output_dir>/<exp_full_name>/<dataset>[__<condition>].rows.jsonl, written
    as each fold finishes (d_vs_dprime.append_rows) and resumed through
    d_vs_dprime.done_keys. output_dir None keeps them in memory."""

    def __init__(self, output_dir, exp_full_name, dataset_name, condition=None):
        stem = str(dataset_name)
        if condition:
            stem += f'__{_clean_stem(condition)}'
        self.path = (None if output_dir is None else os.path.join(
            str(output_dir), exp_full_name, f'{stem}.rows.jsonl'))
        self.done = dvd.done_keys(self.path) if self.path else set()
        self.n_on_disk_at_start = len(self.done)
        self.n_written = 0
        self.memory = []

    def is_done(self, identity, models):
        """True when every model's row for this identity is already written."""
        for m in models:
            probe = dict(identity, model=m)
            if dvd.row_key(probe) not in self.done:
                return False
        return bool(list(models))

    def write(self, rows):
        new = []
        for r in rows:
            key = dvd.row_key(r)
            if key in self.done:
                continue
            self.done.add(key)
            new.append(r)
        if not new:
            return 0
        if self.path:
            dvd.append_rows(self.path, new)
        else:
            self.memory.extend(new)
        self.n_written += len(new)
        return len(new)

    def summary(self):
        return {'rows_path': self.path,
                'rows_on_disk_before_this_run': self.n_on_disk_at_start,
                'rows_written_this_run': self.n_written,
                'row_fields': list(dvd.ROW_FIELDS)}


def identity(*, dataset, experiment, method, method_setting,
             similarity_setting, model_input, noise_condition, level=None,
             draw=None, fold=None):
    """Every d_vs_dprime.KEY_FIELDS entry but `model`."""
    return {'dataset': dataset, 'experiment': experiment, 'method': method,
            'method_setting': method_setting,
            'similarity_setting': similarity_setting,
            'model_input': model_input, 'noise_condition': noise_condition,
            'level': level, 'draw': draw, 'fold': fold}


def row(ident, *, model, metric, score_D, score_Dprime, score_clean=None):
    """One row from numbers the experiment already computed. No fit."""
    metric_name, _fn, lib = metric
    nan = float('nan')
    return dvd.make_row(
        **ident, model=model, metric=metric_name, lower_is_better=lib,
        score_D=nan if score_D is None else float(score_D),
        score_Dprime=nan if score_Dprime is None else float(score_Dprime),
        score_clean=nan if score_clean is None else float(score_clean))


def fit_score(dataset, model_name, factory, X_tr, y_tr, X_va, y_va, metric,
              sample_weight=None):
    """One fit and one held-out score in the dataset's metric, nan on failure.
    Seeded models go through noise_cache.cached_fit_predict; the rest are
    fitted directly. Predicts the way n07._predict_for_metric does
    (noise_cache.predict_mode_for)."""
    metric_name, metric_fn, _lib = metric
    mode = nc.predict_mode_for(metric_name)
    try:
        if model_name in CACHEABLE_MODELS:
            pred = nc.cached_fit_predict(dataset, model_name, factory, X_tr,
                                         y_tr, X_va,
                                         sample_weight=sample_weight,
                                         predict_mode=mode)
        else:
            m = factory()
            if sample_weight is None:
                m.fit(X_tr, y_tr)
            else:
                m.fit(X_tr, y_tr, sample_weight=np.asarray(sample_weight))
            pred = nc._predict(m, X_va, mode)
        return float(metric_fn(y_va, pred))
    except Exception as exc:
        print(f"      row fit {model_name} failed: {type(exc).__name__}: {exc}")
        return float('nan')


def dprime_rows(rows_obj, dataset, ident, cleaned, *, X_tr, y_D, X_va, y_va,
                metric, factories, y_clean=None, X_va_prime=None,
                score_D=None, score_clean=None):
    """Fit D and D' (and the clean labels when given) per model and write one
    row each. `cleaned` is n07.cleaned_training_set's dict (X, y,
    sample_weight) or None (the method declined: score_Dprime nan).
    X_va_prime is the held-out matrix D' is scored on when its features differ
    from D's (n19's count feature). score_D / score_clean {model: score} reuse
    a D or clean score the experiment already holds instead of refitting it."""
    if rows_obj.is_done(ident, list(factories)):
        return 0
    out = []
    for name, f in factories.items():
        if score_D is not None and name in score_D:
            s_D = score_D[name]
        else:
            s_D = fit_score(dataset, name, f, X_tr, y_D, X_va, y_va, metric)
        if score_clean is not None and name in score_clean:
            s_C0 = score_clean[name]
        else:
            s_C0 = None
        if cleaned is None:
            s_P = float('nan')
        else:
            s_P = fit_score(dataset, name, f, cleaned['X'], cleaned['y'],
                            X_va if X_va_prime is None else X_va_prime, y_va,
                            metric, sample_weight=cleaned.get('sample_weight'))
        s_C = (s_C0 if s_C0 is not None or y_clean is None else
               fit_score(dataset, name, f, X_tr, y_clean, X_va, y_va, metric))
        out.append(row(ident, model=name, metric=metric, score_D=s_D,
                       score_Dprime=s_P, score_clean=s_C))
    return rows_obj.write(out)


def control_rows(rows_obj, dataset, ident, cleaned, *, X_tr, y_D, X_va, y_va,
                 metric, factories, seed, is_clf=False, rerun=None,
                 y_clean=None, score_D=None, score_clean=None):
    """The random and shuffled twins of one cleanup's D', as rows whose method
    is '<method>__random_twin' / '<method>__shuffled_twin'. Runs only under
    KIRBY_NOISE_CONTROLS=1 (controls.controls_enabled).

    `rerun(scores, rate)` re-runs the SAME cleanup on a score column, and makes
    the random twin n24's (controls.random_twin docstring). On a
    classification label move the shuffled twin is skipped: permuted label
    changes need not land on a class (controls.shuffled_twin docstring)."""
    if not controls_enabled() or cleaned is None:
        return 0
    n = 0
    try:
        rt, _n_t = random_twin(cleaned, X_tr, y_D, seed, rerun=rerun)
    except Exception as exc:
        print(f"      random twin failed: {type(exc).__name__}: {exc}")
        rt = None
    label_move = cleaned.get('keep') is None and \
        cleaned.get('sample_weight') is None and \
        cleaned.get('label_column') is not None
    st = None
    if not (is_clf and label_move):
        try:
            st = shuffled_twin(cleaned, X_tr, y_D, seed)
        except Exception as exc:
            print(f"      shuffled twin failed: {type(exc).__name__}: {exc}")
    for tag, twin in (('random_twin', rt), ('shuffled_twin', st)):
        if twin is None:
            continue
        n += dprime_rows(rows_obj, dataset,
                         dict(ident, method=f"{ident['method']}__{tag}"),
                         twin, X_tr=X_tr, y_D=y_D, X_va=X_va, y_va=y_va,
                         metric=metric, factories=factories, y_clean=y_clean,
                         score_D=score_D, score_clean=score_clean)
    return n


# ---------------------------------------------------------------------------
# Per-fold checkpoint
# ---------------------------------------------------------------------------

def run_signature(data, config, extra=None):
    """What a checkpoint was computed from: the input fingerprint run_all
    stamps its files with (_shared.input_fingerprint), the run's own settings,
    the similarity settings and whether controls ran. A checkpoint under a
    different signature is ignored, never resumed from."""
    from experiments._shared import input_fingerprint
    try:
        fp = input_fingerprint(data, config)
    except Exception:
        fp = {}
    blob = json.dumps({
        'fp': fp,
        'extra': repr(sorted((extra or {}).items(), key=lambda kv: str(kv[0]))),
        'sim': os.environ.get('KIRBY_SIMILARITY_SETTINGS', ''),
        'controls': controls_enabled(),
    }, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=10).hexdigest()


class FoldCheckpoint:
    """The accumulators after the last finished unit (a fold, or a (draw,
    fold) pair), pickled atomically to
    <output_dir>/<exp_full_name>/_checkpoint/<dataset>[__<cond>]__<sig>.pkl.

    Usage in a fold loop:
        ck = FoldCheckpoint(output_dir, exp, dataset, sig, condition)
        state = ck.state            # None, or what save() last wrote
        ... if state: restore accumulators from it ...
        for unit in units:
            if ck.is_done(unit): continue
            ... compute the unit ...
            ck.save(unit, {accumulators})
        ck.clear()                  # once the result file is written
    Units must run in the same order every time; a resumed run restores the
    snapshot taken after the last finished unit and skips every unit in it.
    output_dir None disables it (a smoke or a direct call)."""

    def __init__(self, output_dir, exp_full_name, dataset_name, signature,
                 condition=None):
        self.path = None
        self.done_units = []
        self.state = None
        if output_dir is None or os.environ.get('KIRBY_NOISE_CHECKPOINT',
                                                '1') == '0':
            return
        stem = str(dataset_name)
        if condition:
            stem += f'__{_clean_stem(condition)}'
        self.path = os.path.join(str(output_dir), exp_full_name, '_checkpoint',
                                 f'{stem}__{signature}.pkl')
        if os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as fh:
                    blob = pickle.load(fh)
                self.done_units = list(blob.get('done_units') or [])
                self.state = blob.get('state')
                print(f"  resuming from {self.path}: {len(self.done_units)} "
                      f"unit(s) already finished")
            except Exception as exc:
                print(f"  checkpoint {self.path} unreadable ({exc}); starting "
                      f"from the first fold")
                self.done_units, self.state = [], None

    def is_done(self, unit):
        return str(unit) in self.done_units

    def save(self, unit, state):
        self.done_units.append(str(unit))
        self.state = state
        if self.path is None:
            return
        d = os.path.dirname(self.path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_')
        try:
            with os.fdopen(fd, 'wb') as fh:
                pickle.dump({'done_units': self.done_units, 'state': state}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def clear(self):
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass
