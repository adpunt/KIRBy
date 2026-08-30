"""Executable checks for the KIRBy noise-robustness runner.

    python tests/smoke/smoke_kirby_uncertainty.py

Replaces `smoke_nine_fixes.py`. That file had 25 checks, of which 11 searched
this module's SOURCE TEXT for a matching string, two were tautologies (its
random-number check reduced to `torch.equal(mid, mid)` and never touched the
pipeline at all), two asserted a defect rather than a fix, and one defect had no
check whatsoever -- while the banner still printed ALL NINE-FIX CHECKS PASSED.
A string match passes whether or not the line it matched ever executes, which is
exactly how the dead `noise_pattern_pred` block below survived: the source
contained the line, so the check was green, and the column was empty in every
file the pipeline ever wrote.

So: EVERY check here runs the code. Where a check is cheap enough to run the
whole pipeline it runs the whole pipeline; where it is not, it calls the real
function with real arrays. Nothing is asserted about the text of this file.

Each check names the property it guards, and each is written so that removing
the fix makes it fail -- several carry an explicit CONTROL showing the value the
check would have taken under the old behaviour.

Optional: set KIRBY_PIPELINE_BAK to a copy of an older runner to enable the
prediction-drift comparison. It is SKIPPED, not failed, when unset.
"""
import contextlib
import importlib.util
import io
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
# KIRBY_PIPELINE points the whole suite at a different copy of the runner. Its
# purpose is the mutation harness: put a copy beside the original with one fix
# undone, run this file against it, and confirm the matching check goes red. A
# check nobody has ever seen fail is not evidence of anything.
PIPE = Path(os.environ.get('KIRBY_PIPELINE',
                           HERE.parent / 'alternative_data_noise_robustness.py'))

_spec = importlib.util.spec_from_file_location('pipe', PIPE)
P = importlib.util.module_from_spec(_spec)
sys.modules['pipe'] = P
_spec.loader.exec_module(P)

from noiseInject import CONDITIONS as NI_CONDITIONS   # noqa: E402
from noiseInject import NoiseInjectorRegression as NI  # noqa: E402


# ── harness ────────────────────────────────────────────────────────────────
FAILS = []
NCHECK = [0]


def check(name, ok, detail=''):
    NCHECK[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def raises(fn, exc, needle=None):
    """Run `fn`; return (did_it_raise_exc, message)."""
    try:
        fn()
    except exc as e:
        msg = str(e)
        return (needle is None or needle.lower() in msg.lower()), msg
    except Exception as e:                       # wrong type is a failure
        return False, f"raised {type(e).__name__} instead: {e}"
    return False, "did not raise"


# ── shared fixtures ────────────────────────────────────────────────────────
# Eight real ring systems, so Murcko scaffolds give a genuine grouped split.
CORES = ['c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1', 'C1CCCCC1',
         'C1CCNCC1', 'c1cc[nH]c1', 'C1CCOC1', 'c1ccsc1']
N_MOL = 240
SMILES = [CORES[i % len(CORES)] + 'C' * (1 + i // len(CORES) % 6) for i in range(N_MOL)]
_rng = np.random.RandomState(7)
# Column 1 is the molecule's position in the full dataset, so a prediction can
# be decoded back to the molecule it was made for. That is what catches an
# out-of-fold block whose rows are labelled with the wrong index.
FEATS = np.column_stack([_rng.normal(size=N_MOL),
                         np.arange(N_MOL, dtype=float),
                         _rng.normal(size=N_MOL)])
LABELS = FEATS[:, 0] * 1.4 + _rng.normal(size=N_MOL) * 0.4 + 5.0


# `INNER` is True only while the runner is inside `_oof_predict`, set by the
# wrapper below. Failing on "a small fit set" instead would also kill the main
# model on the smaller outer folds, which is a different experiment.
INNER = {'on': False, 'calls': 0, 'fail_first': 0}


class GaussianProcessStub:
    """Stands in for the gauche GP: cheap, and its prediction decodes.

    `_tree_mean_std` dispatches on the class NAME containing 'GaussianProcess',
    so this exercises the same branch the real GP does. `predict` returns the
    molecule id carried in feature column 1 (standardised, since PDV is one of
    the representations the shared spec standardises), which lets a test prove
    that the row written for `sample_idx` really is the molecule that model
    scored.
    """

    def __init__(self, *a, **k):
        pass

    def fit(self, Xf, yf):
        if INNER['on']:
            INNER['calls'] += 1
            if INNER['calls'] <= INNER['fail_first']:
                raise RuntimeError('simulated inner-fold failure')
        self.n = len(yf)
        self.ym = float(np.mean(yf))
        return self

    # The likelihood noise. A real Gaussian process has ONE of these for the
    # whole fit, which is why SUPPORT calls its aleatoric term `constant`.
    LIKELIHOOD_NOISE_VAR = 1e-3

    def predict(self, Xs, return_std=False):
        pm = np.asarray(Xs)[:, 1].astype(float).copy()
        ps = np.abs(pm - pm.mean()) / (pm.std() + 1e-9) + 0.1
        self._record_split(pm, ps)
        return (pm, ps) if return_std else pm

    def _record_split(self, pm, ps):
        """Set the two attributes `_tree_split` reads, as the real class does.

        `GaussianProcessGauche.predict` records the split as a side effect,
        because the dispatch around it requires predict to keep its (mean, std)
        signature -- so `_tree_split` reads `aleatoric_var_` and
        `epistemic_var_` off the model rather than being handed them.

        THIS STUB DID NOT SET THEM, and that is the whole of the failure
        recorded as RERUN_PLAN.md 13.17 A4: with one outer fold cross-fitted,
        GP+PDV under grouped_wider wrote no uncertainty rows and the run
        stopped. It was read as a pipeline defect. It is not one. The split was
        wired into this runner on 2026-08-28 and the stub was written before
        that, so `_tree_split` got None for the data half,
        `assert_matches_support` refused the row -- correctly, because SUPPORT
        says a Gaussian process HAS a constant aleatoric term -- the per-fold
        try/except turned that into an ERROR line, and the end-of-run gate then
        reported the silence. Every guard in that chain behaved as designed.

        The split is the real one's shape: the model half is the latent variance
        and varies per molecule, the data half is one number for the fit. They
        sum to exactly the variance `predict` reports, so the standard deviation
        this stub returns is unchanged and no test that reads it moves.
        `ps >= 0.1` by construction, so the model half stays positive.
        """
        ps = np.asarray(ps, dtype=float)
        self.aleatoric_var_ = np.full(len(ps), self.LIKELIHOOD_NOISE_VAR)
        self.epistemic_var_ = ps ** 2 - self.LIKELIHOOD_NOISE_VAR


class GaussianProcessLinearStub(GaussianProcessStub):
    """A stub whose predictions live on the LABEL scale.

    Needed wherever a test looks at a quantity computed FROM a prediction --
    the censoring shape, for instance, is `max(y - cut_point, 0)` against a
    cut-point taken from the training labels, so a stub that predicts a
    standardised feature would land the whole set below the cut-point and the
    shape would come out flat for a reason that has nothing to do with the code
    under test.
    """

    def fit(self, Xf, yf):
        super().fit(Xf, yf)
        A = np.column_stack([np.asarray(Xf)[:, 0], np.ones(len(Xf))])
        self.coef = np.linalg.lstsq(A, yf, rcond=None)[0]
        self.sd = float(np.std(yf - A @ self.coef)) + 1e-6
        return self

    def predict(self, Xs, return_std=False):
        Xs = np.asarray(Xs, dtype=float)
        pm = np.column_stack([Xs[:, 0], np.ones(len(Xs))]) @ self.coef
        ps = self.sd * (1.0 + 0.3 * np.abs(Xs[:, 2]))
        # `self.sd` is a residual spread and can be small, so the data half is
        # taken as a FRACTION of the smallest reported variance rather than a
        # fixed number -- a constant larger than ps**2 would make the model half
        # negative, which _check_nonneg refuses.
        self.LIKELIHOOD_NOISE_VAR = 0.5 * float(np.min(ps) ** 2)
        self._record_split(pm, ps)
        return (pm, ps) if return_std else pm


_REAL_OOF = P._oof_predict


def _oof_marking_inner(*a, **k):
    INNER['on'] = True
    try:
        return _REAL_OOF(*a, **k)
    finally:
        INNER['on'] = False


P._oof_predict = _oof_marking_inner


def install_stub_gp(max_n=None, fail_first=0):
    P.GaussianProcessGauche = GaussianProcessStub
    P.HAS_GP = True
    P.GP_MAX_N = max_n
    P.generate_representations = lambda s, rep_filter=None: {'PDV': FEATS}
    INNER.update(on=False, calls=0, fail_first=fail_first)


def run_pipeline(outdir, quiet=True, **kw):
    """Run the real `run_dataset` on the fixture, returning its log."""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        P.run_dataset('smoke', SMILES, LABELS, outdir,
                      model_filter=kw.pop('models', ['GP']),
                      rep_filter=['PDV'], gp_reps=['PDV'], **kw)
    return buf.getvalue()


def scratch():
    return Path(tempfile.mkdtemp(prefix='kirby_smoke_'))


T0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════
section("A1  the noise shape recomputed from the model's OWN prediction")
# The guard `if sigma in extras['oof_mean']` sat THIRTEEN LINES ABOVE the line
# that puts the level into extras['oof_mean'], inside one pass of one loop. It
# was therefore false on every pass, and `train_noise_pattern_pred` came out
# empty in every file this pipeline has ever written. Nothing noticed, because
# the check for it searched the source for the string 'noise_pattern_pred'.
# 'censoring' is used because its noise scale is keyed to the LABEL, so the
# shape recomputed from a predicted label genuinely differs from the shape
# computed from the true one -- which is the whole point of the column.
gtr = _rng.randint(0, 30, 180)
gte = _rng.randint(0, 30, 60)
_, _, ex = P.run_tree_experiment(
    FEATS[:180], LABELS[:180], FEATS[180:], LABELS[180:],
    lambda: GaussianProcessLinearStub(), 'censoring', [0.0, 0.25],
    oof_folds=3, train_groups=gtr, noise_groups=gtr,
    test_noise_groups=gte, rep_name='PDV')
tpp = ex['train_noise_pattern_pred']
check("tree runner populates it for every level",
      set(tpp) == {0.0, 0.25}, f"levels present: {sorted(tpp)} (was {{}} before the fix)")
check("and the values vary, so the ceiling can be correlated against",
      bool(tpp) and all(len(np.unique(np.round(v, 12))) > 1 for v in tpp.values()),
      f"distinct values per level: {[len(np.unique(np.round(v, 12))) for v in tpp.values()]}")
check("it differs from the shape of the TRUE label, so it is a real control",
      0.25 in tpp and not np.array_equal(tpp[0.25], ex['train_noise_pattern'][0.25]))
check("non-finite out-of-fold predictions do not blank the column",
      np.isfinite(P._pattern_from_oof(
          P._injector_for('censoring', 0.0),
          np.array([1.0, np.nan, 3.0]), LABELS[:3], None)).all())

_fake_calls = []
_real_nn = P.train_neural_regression


def _fake_nn(Xtr, ytr, Xva, yva, Xte, model_type=None, epochs=None,
             y_scaler=None):
    # y_scaler is accepted and ignored: this stub works on the raw labels, and
    # the real function's use of it is checked in the target-scaling section.
    _fake_calls.append((np.array(ytr, float), np.array(yva, float)))
    A = np.column_stack([np.asarray(Xtr)[:, 0], np.ones(len(Xtr))])
    coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
    Xte = np.asarray(Xte, dtype=float)
    pm = np.column_stack([Xte[:, 0], np.ones(len(Xte))]) @ coef
    # FOUR values, not two. The real train_neural_regression started returning
    # the aleatoric and epistemic halves alongside the mean and the spread when
    # the laboratory runner's uncertainty split landed, and this stub was not
    # moved with it -- so every section below that installs it died on
    # "not enough values to unpack (expected 4, got 2)" and the checks after it
    # never ran. None for both halves is the honest stub: this fake has no
    # split to report, and the runner already handles a model that reports none.
    return pm, np.abs(Xte[:, 2]) + 0.1, None, None


P.train_neural_regression = _fake_nn
_, _, exn = P.run_neural_experiment(
    FEATS[:140], LABELS[:140], FEATS[140:180], LABELS[140:180],
    FEATS[180:], LABELS[180:], 'deterministic', 'censoring', [0.0, 0.25],
    oof_folds=3, train_groups=gtr[:140], noise_groups=gtr[:140],
    test_noise_groups=gte, val_noise_groups=gtr[140:180], rep_name='PDV')
P.train_neural_regression = _real_nn
check("neural runner populates it too (the identical dead block lived there)",
      set(exn['train_noise_pattern_pred']) == {0.0, 0.25},
      f"levels present: {sorted(exn['train_noise_pattern_pred'])}")

# ═══════════════════════════════════════════════════════════════════════════
section("A2  leaving 0.0 out of the grid must fail loudly, not write nothing")
install_stub_gp(max_n=None)
d = scratch()
try:
    ok, msg = raises(lambda: run_pipeline(d, sigma_levels=[0.3, 0.6],
                                          unc_conditions='all', oof_folds=3,
                                          conditions=['grouped_wider'],
                                          oof_outer_folds=1),
                     P.RunIntegrityError, 'negative control')
    check("a grid without 0.0 stops the run and says why", ok, msg[:150])
    check("and it does NOT quietly write a file instead",
          not list(Path(d).glob('*uncertainty_values.csv')))
finally:
    shutil.rmtree(d, ignore_errors=True)

# The case that separates the two jobs the old single expression was doing.
# A model that emits an uncertainty at 0.6 but not at 0.0 used to make
# `uncertainties.get(0.0) is not None` false, so the run wrote NOTHING -- test
# rows included -- and exited 0. Asking "does this model emit uncertainty at
# all" instead turns that into a stated error.
_real_tms = P._tree_mean_std
_drop = {'n': 0}


def _tms_dropping_the_first(mdl, X_score):
    """Drop the uncertainty on the first scoring call, which is level 0.0's."""
    m, u = _real_tms(mdl, X_score)
    _drop['n'] += 1
    return (m, None) if _drop['n'] == 1 else (m, u)


install_stub_gp(max_n=None)
P._tree_mean_std = _tms_dropping_the_first
d = scratch()
try:
    ok, msg = raises(lambda: run_pipeline(d, sigma_levels=[0.0, 0.6],
                                          unc_conditions='all', oof_folds=0,
                                          conditions=['grouped_wider']),
                     P.RunIntegrityError, 'not at [0.0]')
    check("a model emitting uncertainty at one level but not another is an "
          "error, not a silent write of nothing", ok, msg[:140])
finally:
    P._tree_mean_std = _real_tms
    shutil.rmtree(d, ignore_errors=True)

d = scratch()
try:
    run_pipeline(d, sigma_levels=[0.0, 0.6], unc_conditions='all', oof_folds=3,
                 conditions=['grouped_wider'], oof_outer_folds=1)
    df = pd.read_csv(d / 'GP_PDV_uncertainty_values.csv')
    check("with 0.0 present, both splits are written",
          set(df['split']) == {'test', 'train_oof'}, str(sorted(set(df['split']))))
    check("every molecule in the dataset appears once per level on the test side "
          "(all five scaffold folds written)",
          int((df['split'] == 'test').sum()) == 2 * N_MOL,
          f"{int((df['split'] == 'test').sum())} test rows "
          f"(2 levels x {N_MOL} molecules)")
    z = df[(df.split == 'train_oof') & (df.sigma == 0.0)]
    check("injected_noise is EXACTLY zero at level 0 (a true negative control)",
          len(z) > 0 and (z['injected_noise'].values == 0.0).all(),
          f"{len(z)} rows, max|eps| = {np.abs(z['injected_noise'].values).max():.3e}")
    tr = df[df.split == 'train_oof']
    check("noise_pattern_pred is populated and varying on training rows",
          tr['noise_pattern_pred'].notna().all() and tr['noise_pattern_pred'].nunique() > 1,
          f"{tr['noise_pattern_pred'].nunique()} distinct values, "
          f"{int(tr['noise_pattern_pred'].isna().sum())} blank")
    pats = {s: g.sort_values('mol_idx')['noise_pattern'].values
            for s, g in tr.groupby('sigma')}
    check("noise_pattern is bit-identical across levels, so the zero-level "
          "subtraction is well posed",
          {0.0, 0.6} <= set(pats) and np.array_equal(pats[0.0], pats[0.6]))
finally:
    shutil.rmtree(d, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
section("A3  --unc-conditions defaults to all, and writing nothing is an error")
_parser = P.build_parser()
_defaults = _parser.parse_args([])
check("argparse itself returns 'all' as the default",
      _defaults.unc_conditions == 'all', f"got '{_defaults.unc_conditions}'")
check("and 'gaussian' is still selectable",
      _parser.parse_args(['--unc-conditions', 'gaussian']).unc_conditions == 'gaussian')

d = scratch()
try:
    # The old silent failure: ask for a targeted condition, leave the writer on
    # 'gaussian', and nothing at all is written while the job exits 0.
    ok, msg = raises(lambda: run_pipeline(d, sigma_levels=[0.0, 0.6],
                                          unc_conditions='gaussian',
                                          conditions=['grouped_wider'],
                                          oof_folds=3, oof_outer_folds=1),
                     P.RunIntegrityError, 'not one uncertainty row')
    check("a run that writes no uncertainty for an uncertainty model raises",
          ok, msg[:160])
finally:
    shutil.rmtree(d, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
section("A4  --oof-folds 1 is rejected instead of silently ignored")
_p = P.build_parser()
ok, msg = raises(lambda: P.validate_args(_p, _p.parse_args(['--oof-folds', '1'])),
                 SystemExit)
check("argparse exits on --oof-folds 1", ok, "(argparse writes the reason to stderr)")
for good in ('0', '3'):
    a = P.validate_args(P.build_parser(), P.build_parser().parse_args(['--oof-folds', good]))
    check(f"--oof-folds {good} is still accepted", a.oof_folds == int(good))
ok, _ = raises(lambda: P.validate_args(_p, _p.parse_args(['--oof-folds', '-2'])), SystemExit)
check("a negative fold count is rejected too", ok)

# ═══════════════════════════════════════════════════════════════════════════
section("A5  uncertainty models are matched by PREFIX, not by exact equality")
check("the plain names still match", all(P._emits_uncertainty(m) for m in P.UNCERTAINTY_MODELS))
check("a decorated name matches ('GP-Tanimoto' got no scoring under equality)",
      P._emits_uncertainty('GP-Tanimoto') and P._emits_uncertainty('NGBoost-Poisson'))
check("an unrelated model still does not match",
      not any(P._emits_uncertainty(m) for m in ('RF', 'SVM', 'DNN', 'MLP', 'LightGBM')))
check("and a name that merely starts with the same letters does not",
      not P._emits_uncertainty('GPU') and not P._emits_uncertainty('QRFast'))

install_stub_gp(max_n=None)
d = scratch()
try:
    # End to end: build the GP under a decorated name and demand training rows.
    run_pipeline(d, sigma_levels=[0.0, 0.6], unc_conditions='all', oof_folds=3,
                 conditions=['grouped_wider'], oof_outer_folds=1,
                 gp_kernel='tanimoto', models=['GP-Tanimoto'])
    df = pd.read_csv(d / 'GPTanimoto_PDV_uncertainty_values.csv')
    n_tr = int((df['split'] == 'train_oof').sum())
    check("a GP built as 'GP-Tanimoto' now gets out-of-fold training rows",
          n_tr > 0, f"{n_tr} train_oof rows (was 0 under exact-equality matching)")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
section("A6  the noise at a level depends on the LEVEL, not on the grid it sat in")
# Seeded from the grid POSITION, `--sigmas 0.0 0.6` and `--sigmas 0.0 0.3 0.6`
# corrupted the same molecules differently at 0.6, so a gap-filling
# resubmission drew noise unrelated to the original run's and the two files
# merged into one column with nothing to tell them apart.
kw = dict(oof_folds=0, train_groups=gtr, noise_groups=gtr,
          test_noise_groups=gte, rep_name='PDV')
_, _, ea = P.run_tree_experiment(FEATS[:180], LABELS[:180], FEATS[180:], LABELS[180:],
                                 lambda: GaussianProcessStub(), 'gaussian',
                                 [0.0, 0.6], **kw)
_, _, eb = P.run_tree_experiment(FEATS[:180], LABELS[:180], FEATS[180:], LABELS[180:],
                                 lambda: GaussianProcessStub(), 'gaussian',
                                 [0.0, 0.3, 0.6], **kw)
_, _, ec = P.run_tree_experiment(FEATS[:180], LABELS[:180], FEATS[180:], LABELS[180:],
                                 lambda: GaussianProcessStub(), 'gaussian',
                                 [0.6], **kw)
d1 = float(np.abs(ea['train_epsilon'][0.6] - eb['train_epsilon'][0.6]).max())
d2 = float(np.abs(ea['train_epsilon'][0.6] - ec['train_epsilon'][0.6]).max())
check("level 0.6 draws the same noise in a 2-level and a 3-level grid",
      np.array_equal(ea['train_epsilon'][0.6], eb['train_epsilon'][0.6]),
      f"max|difference| = {d1:.3e}")
check("and the same again when 0.6 is the only level in the grid",
      np.array_equal(ea['train_epsilon'][0.6], ec['train_epsilon'][0.6]),
      f"max|difference| = {d2:.3e}")
check("CONTROL: different LEVELS still draw different noise",
      not np.array_equal(eb['train_epsilon'][0.3], eb['train_epsilon'][0.6]))
check("different conditions still draw different noise",
      P._injector_for('gaussian', 0.6).random_state
      != P._injector_for('laplace', 0.6).random_state)
check("the validation stream is independent of the training stream",
      P._injector_for('gaussian', 0.6).random_state
      != P._injector_for('gaussian', 0.6, stream='validation').random_state,
      f"training {P._injector_for('gaussian', 0.6).random_state}, "
      f"validation {P._injector_for('gaussian', 0.6, stream='validation').random_state}")
check("0.6, 0.60 and 6e-1 are one level, not three",
      P._injector_for('gaussian', 0.6).random_state
      == P._injector_for('gaussian', 0.60).random_state
      == P._injector_for('gaussian', 6e-1).random_state)
check("seeding is stable across processes (crc32, not Python's salted hash)",
      P._injector_for('gaussian', 0.6).random_state ==
      (42 * 1000003 + __import__('zlib').crc32(b'training|gaussian|0.6')) & 0x7FFFFFFF)

# validation labels really are corrupted, from their own stream, and the
# training draw is untouched by whether they are  (ported from d9_val.py)
_fake_calls.clear()
P.train_neural_regression = _fake_nn
seen = {}
for flag in (True, False):
    _fake_calls.clear()
    P.run_neural_experiment(FEATS[:140], LABELS[:140], FEATS[140:180], LABELS[140:180],
                            FEATS[180:], LABELS[180:], 'deterministic', 'gaussian',
                            [0.0, 0.6], oof_folds=0, noise_validation=flag,
                            train_groups=gtr[:140], noise_groups=gtr[:140],
                            test_noise_groups=gte, val_noise_groups=gtr[140:180],
                            rep_name='PDV')
    seen[flag] = list(_fake_calls)
P.train_neural_regression = _real_nn
check("training labels are identical whether or not validation is corrupted",
      all(np.array_equal(seen[True][i][0], seen[False][i][0]) for i in range(2)))
check("validation is corrupted at a non-zero level",
      not np.array_equal(seen[True][1][1], LABELS[140:180]),
      f"max|dy| = {np.abs(seen[True][1][1] - LABELS[140:180]).max():.4f}")
check("validation is left clean at level 0 and under --no-noise-validation",
      np.array_equal(seen[True][0][1], LABELS[140:180])
      and np.array_equal(seen[False][1][1], LABELS[140:180]))

# ═══════════════════════════════════════════════════════════════════════════
section("A7  row counts are asserted, and a short or empty block stops the run")
good = pd.DataFrame([{'split': 'test', 'noise_type': 'gaussian', 'sigma': s,
                      'fold': 0, 'v': i}
                     for s in (0.0, 0.6) for i in range(10)])
check("a complete frame passes", P._assert_cell_counts(good, 10, 'ctx') is not None)
ok, msg = raises(lambda: P._assert_cell_counts(good.iloc[:-1], 10, 'ctx'),
                 P.RunIntegrityError, 'expected 10 rows')
check("one missing row is caught", ok, msg[:110])
ok, _ = raises(lambda: P._assert_cell_counts(good.iloc[:0], 10, 'ctx'),
               P.RunIntegrityError, 'no rows at all')
check("an empty block is caught", ok)
ok, _ = raises(lambda: P._assert_cell_counts(good[good.sigma == 0.0], 10, 'ctx'),
               P.RunIntegrityError)
check("CONTROL: a frame missing a whole LEVEL is not caught by the per-cell "
      "count alone -- the run's own level list is what catches that",
      not ok, "(guarded separately by _levels_written; see A7 end-to-end below)")

# the guard is wired into the writer, and the writer verifies what LANDED
d = scratch()
try:
    ragged = pd.concat([good,
                        pd.DataFrame([{'split': 'test', 'noise_type': 'laplace',
                                       'sigma': 0.0, 'fold': 0, 'v': 1}])],
                       ignore_index=True)
    ok, msg = raises(lambda: P._flush_uncertainties([('QRF', 'ECFP4', ragged)], d, 'smoke'),
                     P.RunIntegrityError, 'not constant')
    check("the writer reads the file back and rejects a ragged cell", ok, msg[:130])
finally:
    shutil.rmtree(d, ignore_errors=True)

# and re-flushing the same fold replaces rather than duplicates (ported)
d = scratch()
try:
    frame = pd.DataFrame([{'split': 'test', 'noise_type': 'gaussian', 'sigma': 0.0,
                           'fold': 0, 'sample_idx': i, 'uncertainty': 0.1}
                          for i in range(10)])
    acc = [('QRF', 'ECFP4', frame)]
    P._flush_uncertainties(acc, d, 'smoke')
    P._flush_uncertainties(acc, d, 'smoke')
    got = pd.read_csv(d / 'QRF_ECFP4_uncertainty_values.csv')
    check("re-flushing one fold replaces rather than duplicates", len(got) == 10,
          f"{len(got)} rows")
    acc.append(('QRF', 'ECFP4', frame.assign(fold=1)))
    P._flush_uncertainties(acc, d, 'smoke')
    got = pd.read_csv(d / 'QRF_ECFP4_uncertainty_values.csv')
    check("a second fold is added, not overwritten",
          len(got) == 20 and set(got['fold']) == {0, 1}, f"{len(got)} rows")
finally:
    shutil.rmtree(d, ignore_errors=True)

# end to end: the counts in the real output file are exactly what was intended
install_stub_gp(max_n=None)
d = scratch()
try:
    run_pipeline(d, sigma_levels=[0.0, 0.6], unc_conditions='all', oof_folds=3,
                 conditions=['grouped_wider', 'censoring'], oof_outer_folds=1)
    df = pd.read_csv(d / 'GP_PDV_uncertainty_values.csv')
    cells = df.groupby(['split', 'fold', 'noise_type', 'sigma']).size()
    per_fold = {k: g.nunique() for k, g in cells.groupby(level=['split', 'fold'])}
    check("every (condition, level) cell in a split/fold holds the same count",
          set(per_fold.values()) == {1}, str(per_fold))
    check("both conditions reached the file",
          set(df['noise_type']) == {'grouped_wider', 'censoring'},
          str(sorted(set(df['noise_type']))))
    check("censoring is swept on its own level axis, and includes a true zero",
          set(df[df.noise_type == 'censoring']['sigma']) == set(P.CENSORING_LEVELS),
          str(sorted(set(df[df.noise_type == 'censoring']['sigma']))))
finally:
    shutil.rmtree(d, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════════════════
section("out-of-fold construction  (ported from t3.py: coverage, leakage, identity)")


def probe_oof(n, groups, n_folds, label):
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.arange(n, dtype=float) * 10.0
    seen = []

    # `_oof_predict` hands the fit rows' scaffold ids to fit_predict as a FOURTH
    # argument -- NGBoost carves its own early-stopping split inside fit() and
    # can only carve it grouped if it knows them -- and returns SIX values, the
    # last four being the out-of-fold split, which inner fit scored each
    # molecule, and the count of folds that succeeded.
    #
    # This probe still took three arguments and unpacked three values, so every
    # call raised inside the try/except in _oof_predict, all five inner folds
    # were reported as failed, and the unpack below then raised ValueError. The
    # same staleness as the stub GP above and RERUN_PLAN.md 13.17 A4: the runner
    # moved on 2026-08-28 and the probes did not.
    def fp(Xf, yf, Xs, groups_fit=None):
        seen.append((Xf[:, 0].astype(int).copy(), Xs[:, 0].astype(int).copy()))
        if groups_fit is not None and len(groups_fit) != len(Xf):
            raise AssertionError(
                f'the fit set has {len(Xf)} rows and {len(groups_fit)} scaffold '
                f'ids travelled with it')
        return Xs[:, 0].copy(), Xs[:, 0].copy() + 0.5

    om, ou, _oof_alea, _oof_epis, oof_fold, n_ok = P._oof_predict(
        fp, X, y, n_folds, groups=groups)
    problems = []
    cnt = np.zeros(n, int)
    for keep, held in seen:
        cnt[held] += 1
        if len(np.intersect1d(keep, held)):
            problems.append('a molecule was in both the fit set and the scored set')
        if len(keep) + len(held) != n:
            problems.append(f'keep+held={len(keep) + len(held)} != n={n}')
        if groups is not None and len(np.unique(groups)) >= n_folds:
            if set(np.asarray(groups)[keep]) & set(np.asarray(groups)[held]):
                problems.append('a scaffold spanned the fit/score boundary')
    if not (cnt == 1).all():
        problems.append(f'coverage min={cnt.min()} max={cnt.max()}')
    if not np.array_equal(om, np.arange(n, dtype=float)):
        problems.append('predictions landed on the wrong rows')
    if not np.array_equal(ou, np.arange(n, dtype=float) + 0.5):
        problems.append('uncertainties landed on the wrong rows')
    if n_ok != len(seen):
        problems.append(f'n_ok {n_ok} != folds run {len(seen)}')
    # Which inner fit scored each molecule. Without it a term that is one number
    # per FIT cannot be told apart from one that varies per molecule, which is
    # the distinction the whole uncertainty question turns on.
    if not (np.asarray(oof_fold) >= 0).all():
        problems.append('some molecule was not attributed to an inner fold')
    if len(set(np.asarray(oof_fold).tolist())) != len(seen):
        problems.append(f'{len(set(np.asarray(oof_fold).tolist()))} distinct fold '
                        f'labels for {len(seen)} folds')
    # This probe's fit_predict returns the two-value form, so the split must
    # come back absent rather than as a column of zeros.
    if _oof_alea is not None or _oof_epis is not None:
        problems.append('a fit_predict that returns no split produced one')
    check(label, not problems, '; '.join(problems) or f'n={n}, {len(seen)} folds')


_r = np.random.RandomState(0)
for n, ng in [(100, 37), (1000, 250), (53, 11), (101, 5)]:
    g = _r.randint(0, ng, n)
    g[:ng] = np.arange(ng)
    probe_oof(n, g, 5, f"scaffold split, n={n}, {ng} scaffolds")
g = np.zeros(200, int)
g[:5] = np.arange(5)
probe_oof(200, g, 5, "one huge scaffold plus four singletons")
for ng in (1, 2, 4):
    g = _r.randint(0, ng, 257)
    g[:ng] = np.arange(ng)
    probe_oof(257, g, 5, f"fallback to a random split, only {ng} scaffold(s)")
probe_oof(257, None, 5, "no scaffolds supplied at all")
probe_oof(7, None, 5, "n=7 with 5 folds (uneven parts)")

# ═══════════════════════════════════════════════════════════════════════════
section("a failed inner fold is reported, not silently blanked  (ported from t5.py)")
for k, expect_rows in ((0, True), (2, True), (3, False)):
    install_stub_gp(max_n=None, fail_first=k)
    d = scratch()
    try:
        log = run_pipeline(d, sigma_levels=[0.0], unc_conditions='all',
                           oof_folds=3, conditions=['gaussian'], oof_outer_folds=1)
        df = pd.read_csv(d / 'GP_PDV_uncertainty_values.csv')
        tr = df[df.split == 'train_oof']
        if expect_rows:
            check(f"{k}/3 inner folds forced to fail: rows written, count recorded",
                  len(tr) > 0 and set(tr['oof_folds_ok']) == {3 - k},
                  f"{len(tr)} rows, oof_folds_ok={sorted(set(tr['oof_folds_ok']))}")
            check(f"{k}/3 forced to fail: the truncation is logged",
                  (k == 0) or ('inner folds succeeded' in log),
                  '' if k == 0 else 'WARNING line present')
        else:
            check("all inner folds failing writes NO training block "
                  "(a blank cell counts as present downstream)",
                  len(tr) == 0 and 'block skipped' in log,
                  f"{len(tr)} train_oof rows, test rows still written: "
                  f"{int((df.split == 'test').sum())}")
        check(f"{k}/3 forced to fail: TEST rows are unaffected",
              int((df.split == 'test').sum()) > 0)
    finally:
        shutil.rmtree(d, ignore_errors=True)
install_stub_gp(max_n=None)

# ═══════════════════════════════════════════════════════════════════════════
section("the GP subsample keeps FULL-training row indices  (ported from t4/t4b.py)")
install_stub_gp(max_n=40)
d = scratch()
try:
    run_pipeline(d, sigma_levels=[0.0, 0.6], unc_conditions='all', oof_folds=3,
                 conditions=['grouped_wider'], oof_outer_folds=1)
    df = pd.read_csv(d / 'GP_PDV_uncertainty_values.csv')
    tr = df[(df.split == 'train_oof') & (df.sigma == 0.6) & (df.fold == 0)]
    groups_all, _ = P.assign_scaffold_groups(SMILES)
    fold0_train, _ = list(GroupKFold(n_splits=P.N_FOLDS).split(SMILES, LABELS, groups_all))[0]
    # The pipeline's OWN carve, not a re-implementation: the early-stopping split
    # is grouped by scaffold now, so `np.arange(len(train)//5, ...)` names
    # different molecules than the runner used.
    tl, _vl = P.scaffold_validation_carve(
        LABELS[fold0_train], np.asarray(groups_all)[fold0_train], 0)
    y_train = LABELS[fold0_train][tl]
    X_train = FEATS[fold0_train][tl]
    gp_row_idx = np.random.RandomState(42).choice(len(X_train), 40, replace=False)
    si = tr['sample_idx'].values.astype(int)
    check("one row per subsampled molecule, no duplicates",
          len(si) == 40 and len(set(si)) == 40, f"{len(si)} rows")
    check("sample_idx indexes the FULL training array, not the subsample",
          set(si) == set(gp_row_idx) and si.max() >= 40,
          f"max sample_idx = {si.max()} (>= 40 proves it is not 0..39)")
    check("y_true is the clean label of the molecule sample_idx names",
          np.allclose(tr['y_true'].values, y_train[si]))
    check("CONTROL: it is NOT y_train[0..40), the buggy indexing",
          not np.allclose(tr['y_true'].values, y_train[:40]),
          f"max|difference| under the buggy indexing = "
          f"{np.abs(tr['y_true'].values - y_train[:40]).max():.4f}")
    # The stub predicts the molecule's own global id, read out of feature
    # column 1 -- but PDV is standardised before the model ever sees it, so undo
    # that first. The scaler was fitted on the 40 subsampled rows, which is
    # exactly what the runner scaled.
    col = X_train[gp_row_idx, 1]
    # Two transforms to undo, not one. The features are standardised before the
    # model sees them, and the TARGET is standardised too now (fitted on the
    # clean training labels), with predictions inverse-transformed on the way
    # out -- so the stub's molecule id comes back on the label scale.
    y_fit = y_train[gp_row_idx]
    decoded = ((tr['y_pred'].values - y_fit.mean()) / y_fit.std()) * col.std() + col.mean()
    check("the prediction on a row was made for the molecule that row names",
          np.allclose(np.rint(decoded), X_train[si, 1]),
          f"max|decoded - true molecule id| = "
          f"{np.abs(decoded - X_train[si, 1]).max():.3e}")
    check("CONTROL: the decode does not match a shuffled assignment",
          not np.allclose(np.rint(decoded), X_train[np.roll(si, 1), 1]))
finally:
    shutil.rmtree(d, ignore_errors=True)
install_stub_gp(max_n=None)

# ═══════════════════════════════════════════════════════════════════════════
section("the out-of-fold pass does not move the global torch generator")
# The old check for this reduced to `torch.equal(mid, mid)` and never touched
# the pipeline. This one runs the real neural path twice.
_rs = np.random.RandomState(0)
Xtr, Xva, Xte = _rs.randn(96, 6), _rs.randn(32, 6), _rs.randn(40, 6)
ytr = Xtr[:, 0] * 2 + _rs.randn(96) * 0.3
yva = Xva[:, 0] * 2 + _rs.randn(32) * 0.3
yte = Xte[:, 0] * 2 + _rs.randn(40) * 0.3
gg = _rs.randint(0, 12, 96)


def neural_run(oof):
    torch.manual_seed(12345)
    np.random.seed(999)
    random.seed(777)
    with contextlib.redirect_stdout(io.StringIO()):
        p, u, e = P.run_neural_experiment(Xtr, ytr, Xva, yva, Xte, yte,
                                          'deterministic', 'gaussian', [0.0, 0.6],
                                          oof_folds=oof, train_groups=gg,
                                          noise_groups=gg, test_noise_groups=None,
                                          val_noise_groups=None, rep_name='PDV')
    return p, u, e, torch.get_rng_state().clone()


p0, u0, e0, s0 = neural_run(0)
p3, u3, e3, s3 = neural_run(3)
check("predictions are bit-identical with and without cross-fitting",
      all(np.array_equal(np.asarray(p0[s], float), np.asarray(p3[s], float))
          for s in (0.0, 0.6)),
      "max|difference| = " + f"{max(np.abs(np.asarray(p0[s], float) - np.asarray(p3[s], float)).max() for s in (0.0, 0.6)):.3e}")
check("the training corruption is identical too",
      all(np.array_equal(e0['train_epsilon'][s], e3['train_epsilon'][s])
          for s in (0.0, 0.6)))
check("the global torch generator ends in the same state",
      torch.equal(s0, s3))
check("CONTROL: the cross-fitting pass really did run extra fits",
      bool(e3['oof_mean']) and not e0['oof_mean'],
      f"levels cross-fitted: {sorted(e3['oof_mean'])}")

# ═══════════════════════════════════════════════════════════════════════════
section("out-of-fold error exceeds in-sample error  (a real model, real fits)")
# The signature of memorisation is ERROR, not uncertainty: a model always fits
# its own training rows better than held-out ones. If this fails, molecules are
# being scored by models that saw their corrupted labels, and every per-molecule
# number in the study is measuring memory.
unc_model, unc_name = None, None
_probe_X = np.random.RandomState(0).normal(size=(60, 4))
_probe_y = _probe_X[:, 0]
try:
    # Import AND fit: quantile_forest can import cleanly and then fail at fit
    # time against a mismatched scikit-learn ('Invalid parameter monotonic_cst').
    from quantile_forest import RandomForestQuantileRegressor
    RandomForestQuantileRegressor(n_estimators=5, random_state=0).fit(_probe_X, _probe_y)
    unc_model = lambda: RandomForestQuantileRegressor(n_estimators=60, random_state=42, n_jobs=2)
    unc_name = 'QRF'
except Exception as _e:
    print(f"  NOTE  quantile_forest unusable here ({type(_e).__name__}), "
          f"falling back to NGBoost. VERIFY QRF ON THE SERVER BEFORE SUBMITTING QRF JOBS.")
    try:
        from ngboost import NGBRegressor
        NGBRegressor(n_estimators=5, verbose=False).fit(_probe_X, _probe_y)
        unc_model = lambda: NGBRegressor(n_estimators=120, learning_rate=0.03,
                                         random_state=42, verbose=False)
        unc_name = 'NGBoost'
    except Exception as _e2:
        print(f"  NOTE  ngboost unusable here too ({type(_e2).__name__})")
if unc_model is None:
    check("an uncertainty-capable model is installed", False,
          "neither quantile_forest nor ngboost fits in this interpreter")
else:
    _rs = np.random.RandomState(1)
    Xr = _rs.normal(size=(400, 10))
    w = _rs.normal(size=10)
    yr = Xr @ w + 0.25 * _rs.normal(size=400)
    gr = _rs.randint(0, 30, 400)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, exr = P.run_tree_experiment(Xr, yr, Xr[:100], yr[:100], unc_model,
                                          'outlier_p05', [0.0, 1.0], oof_folds=5,
                                          train_groups=gr, noise_groups=gr,
                                          test_noise_groups=gr[:100], rep_name='PDV')
    yn = yr + exr['train_epsilon'][1.0]
    m = unc_model()
    m.fit(Xr, yn)
    ins_mean, _ = P._tree_mean_std(m, Xr)
    err_ins = float(np.mean(np.abs(ins_mean - yn)))
    err_oof = float(np.nanmean(np.abs(exr['oof_mean'][1.0] - yn)))
    check(f"({unc_name}) out-of-fold error > in-sample error",
          err_oof > err_ins,
          f"oof {err_oof:.4f} vs in-sample {err_ins:.4f} "
          f"(+{100 * (err_oof / err_ins - 1):.1f}%)")
    check("every training molecule received an out-of-fold uncertainty",
          np.isfinite(exr['oof_unc'][1.0]).all(),
          f"{int(np.isfinite(exr['oof_unc'][1.0]).sum())}/400")
    # and the ceiling column is not simply a copy of the real shape
    a = exr['train_noise_pattern'][1.0]
    b = exr['train_noise_pattern_pred'].get(1.0, np.array([np.nan]))
    check("the sham-ceiling shape is populated for a real model",
          np.isfinite(b).all() and len(np.unique(np.round(b, 12))) > 1,
          f"{len(np.unique(np.round(b, 12)))} distinct values; "
          f"spearman against the true shape = "
          f"{spearmanr(a, b).correlation if np.std(a) > 0 else float('nan'):+.3f}")

# ═══════════════════════════════════════════════════════════════════════════
section("every condition delivers the amount it was asked for")
# The six conditions this replaced were one condition at six doses: at a common
# nominal setting they delivered between 0.49x and 2.00x the same amount, and
# their whole apparent severity ordering was that.
from noiseInject import dose_tolerance   # noqa: E402
tau = 0.5 * float(np.std(LABELS))
ygroups = _r.randint(0, 40, len(LABELS))
off, tails = [], {}
for cond in ('gaussian', 'student_t_nu3', 'grouped_wider', 'outlier_p10'):
    r = NI.from_condition(cond, random_state=0).inject_verbose(LABELS, tau, groups=ygroups)
    tol = dose_tolerance(r.epsilon, r.effective_n, nu=NI_CONDITIONS[cond].get('nu'))
    if abs(r.realised_dose_label_units / tau - 1) > tol:
        off.append(f"{cond} {100 * (r.realised_dose_label_units / tau - 1):+.1f}% "
                   f"(tolerance {100 * tol:.1f}%)")
    tails[cond] = float(np.mean(np.abs(r.epsilon) > 3 * tau))
check("the dose confound is gone", not off, '; '.join(off) or 'all within tolerance')
check("but the conditions still differ in how concentrated the damage is",
      tails['student_t_nu3'] > 3 * tails['gaussian'],
      ', '.join(f'{k} {100 * v:.2f}%' for k, v in tails.items()))
flat = NI.from_condition('gaussian', random_state=0).noise_scale(LABELS, 1.0)
check("a shape-only condition has a CONSTANT scale, so 'where is the noise' is "
      "undefined there rather than zero",
      len(np.unique(np.round(flat, 12))) == 1)
wide = NI.from_condition('grouped_wider', random_state=0).noise_scale(
    LABELS, 1.0, groups=ygroups)
check("a targeted condition does have a pattern to find",
      len(np.unique(np.round(wide, 12))) > 1,
      f"{len(np.unique(np.round(wide, 12)))} distinct scales")

# ═══════════════════════════════════════════════════════════════════════════
section("optional: drift against an older runner")
BAK = os.environ.get('KIRBY_PIPELINE_BAK')
if not BAK:
    print("  SKIP  no KIRBY_PIPELINE_BAK set — nothing to compare against.")
    print("        This is optional by design: the seed formula changed in A6, so")
    print("        predictions are EXPECTED to differ from any earlier run and")
    print("        bit-identity is not a property to preserve.")
elif not Path(BAK).exists():
    print(f"  SKIP  KIRBY_PIPELINE_BAK={BAK} does not exist.")
else:
    _s = importlib.util.spec_from_file_location('pipe_old', BAK)
    OLD = importlib.util.module_from_spec(_s)
    sys.modules['pipe_old'] = OLD
    try:
        _s.loader.exec_module(OLD)
        _, _, eo = OLD.run_tree_experiment(FEATS[:180], LABELS[:180], FEATS[180:],
                                           LABELS[180:], lambda: GaussianProcessStub(),
                                           'gaussian', [0.0, 0.6], **kw)
        same = np.array_equal(eo['train_epsilon'][0.6], ea['train_epsilon'][0.6])
        print(f"  INFO  noise at level 0.6 identical to the backup: {same} "
              f"(expected False after A6)")
    except Exception as e:
        print(f"  SKIP  the backup does not execute here: {type(e).__name__}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"{NCHECK[0]} checks in {time.time() - T0:.1f}s")
if FAILS:
    print("FAILED:", "; ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
