import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*experimental_relax_shapes.*')
warnings.filterwarnings('ignore', message='.*reduce_retracing.*')

"""
Validation Noise Robustness: 3 Regression Datasets with Scaffold CV
====================================================================

Tests noise robustness on three validated regression datasets that pass
the ECFP4+RF baseline (R² > 0.5).

Datasets (all REGRESSION). The counts are what the loaders in THIS file produce
from the files this file actually reads (openadmet_train.csv, 5,326 rows, and the
cached ChEMBL extract). The header used to quote 7,309 and 3,777, which are the
non-null LogD and Caco-2 counts in openadmet_raw.csv -- a larger extraction this
script never opens (RERUN_PLAN.md 2.13).
  1. OpenADMET-LogD — 5039 molecules, lipophilicity
  2. OpenADMET-Caco2_Efflux — 2161 molecules, P-gp efflux ratio
  3. ChEMBL-hERG-Ki — 1415 molecules, hERG binding affinity (pKi)

Split: 5-fold scaffold CV (Murcko scaffolds)
  - Groups molecules by scaffold, CHIRALITY-BLIND, with acyclic molecules as
    singletons -- the same key QM9 splits on
  - Holds out entire scaffold groups per fold
  - Carves the early-stopping rows out of each fold's training block by
    scaffold too, so stopping is not scored on frameworks the model just fitted
  - Tests generalization to novel chemotypes

Model-Representation Matrix:
  Representations: ECFP4, PDV, SNS, MHG-GNN-pretrained, Avalon, ChemBERTa (6 total)
  Models: RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV only),
          DNN, BNN-Full, VBLL-Full,                    (NN-α family — DNN base)
          MLP, MLP-BNN-Full, MLP-VBLL-Full             (NN-β family — MLP base)
          (13 total; BNN-Last implemented but not run by default)
  Total: 4 reps × 13 models × 7 conditions × 3 datasets × 5 folds

Noise conditions: read from noise_conditions.json in the qsar_qm_models
  checkout, so this file states no list of its own. As settled on 2026-08-27
  that is gaussian, grouped_wider, grouped_shifted, censoring, student_t_nu5,
  outlier_p10, laplace. The six named here before -- legacy, outlier, quantile,
  hetero, threshold, valprop -- were deleted in noiseInject 1.0.0; they were one
  condition at six doses.
Noise levels: 0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5 (7), in fractions of each
  dataset's own clean training label spread. The eleven-step 0.0-1.0 ladder
  named here before was the old raw-scale grid.

Usage:
  python alternative_data_noise_robustness.py
  python alternative_data_noise_robustness.py --datasets logd caco2 herg
  python alternative_data_noise_robustness.py --results-root results/validation
"""

import argparse
import zlib

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import sys
import time
import json
from datetime import datetime, timezone

import requests
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED MODEL SPEC
# ═══════════════════════════════════════════════════════════════════════════
# Every default this pipeline shares with QM9 lives in ONE file, in the
# qsar_qm_models checkout, and is IMPORTED rather than restated. Restating it
# here is what let the boosting learning rate, the quantile-forest tree count
# and the NN-beta width drift apart under one name each.
#
# No vendored copy and no silent fallback: if the checkout cannot be found this
# raises, because a second copy that quietly goes stale is the defect, not the
# fix. Every job script already runs `. <qsar_qm_models>/setup.sh` before this
# module is imported, so the checkout is present by construction.

def _load_model_defaults():
    candidates = []
    env = os.environ.get('QSAR_QM_MODELS_ROOT')
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent.parent / 'qsar_qm_models',   # sibling checkout
        Path('/data/stat-cadd/scat9264/qsar_qm_models'),
        Path('/data/stat-ecr/scat9264/qsar_qm_models'),
    ]
    for root in candidates:
        spec_file = Path(root) / 'models' / 'model_defaults.py'
        if spec_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location('model_defaults', spec_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"  model spec: {spec_file}  version {mod.SPEC_VERSION}  "
                  f"hash {mod.spec_hash()}")
            return mod
    raise RuntimeError(
        "Cannot find models/model_defaults.py, the shared parameter spec. This "
        "pipeline must build the same models as QM9 and refuses to guess them. "
        "Set QSAR_QM_MODELS_ROOT to the qsar_qm_models checkout, or place it "
        f"beside the KIRBy checkout. Looked in: "
        + ", ".join(str(c) for c in candidates))


def _widen_output_head(model, n_outputs):
    """Widen a network's output layer, or STOP.

    A variance head needs two outputs: the value and its own log variance. The
    layer is called `fc3` on the DNN and `output_layer` on the MLP.

    This is QM9's `widen_output_head` (models.py), including its refusal. There
    it used to be a chain of `hasattr` with no `else`, so a class with none of
    those names fell through in SILENCE: the network kept a one-column head while
    being fitted by a two-column loss, the subtraction broadcast instead of
    raising, both outputs were regressed onto the label, and the variance column
    that reached the file meant nothing (RERUN_PLAN.md 5.5a point 3).
    """
    import torch.nn as _nn
    for attr in ('fc3', 'fc_out', 'output_layer'):
        head = getattr(model, attr, None)
        if isinstance(head, _nn.Linear):
            setattr(model, attr, _nn.Linear(head.in_features, n_outputs))
            return model
    # This side's DNN wraps its layers in an nn.Sequential called `net`, so it
    # has no named head at all and the loop above cannot see it. Replace the
    # LAST Linear in the sequence, which is the head by construction.
    seq = getattr(model, 'net', None)
    if isinstance(seq, _nn.Sequential):
        for i in range(len(seq) - 1, -1, -1):
            if isinstance(seq[i], _nn.Linear):
                seq[i] = _nn.Linear(seq[i].in_features, n_outputs)
                return model
    raise RuntimeError(
        f"a {n_outputs}-output head was needed and {type(model).__name__} has "
        f"no layer this recognises (looked for fc3, fc_out, output_layer). A "
        f"one-column head fitted by a two-column loss trains both outputs onto "
        f"the label and the variance column means nothing.")


def _load_shared_module(relpath, what):
    """Import a module from the qsar_qm_models checkout, by path.

    Same search and same refusal as the two loaders below: no vendored copy and
    no silent fallback, because a second copy that goes stale is the defect and
    not the fix.
    """
    candidates = []
    env = os.environ.get('QSAR_QM_MODELS_ROOT')
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent.parent / 'qsar_qm_models',
        Path('/data/stat-cadd/scat9264/qsar_qm_models'),
        Path('/data/stat-ecr/scat9264/qsar_qm_models'),
    ]
    for root in candidates:
        mod_file = Path(root) / relpath
        if mod_file.exists():
            import importlib.util
            # The module's OWN directory goes on the path first. A shared module
            # may import its siblings by bare name -- scripts/utils.py does
            # `from uncertainty_decomposition import ...` -- and loading it by
            # path alone leaves those siblings unimportable, so the load dies
            # with ModuleNotFoundError for a file sitting next to the one that
            # was found.
            parent = str(mod_file.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            spec = importlib.util.spec_from_file_location(
                Path(relpath).stem, mod_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"  {what}: {mod_file}")
            return mod
    raise RuntimeError(
        f"Cannot find {relpath}, the shared {what}. This pipeline must use the "
        f"same one QM9 uses and refuses to define it a second time. Set "
        f"QSAR_QM_MODELS_ROOT to the qsar_qm_models checkout. Looked in: "
        + ", ".join(str(c) for c in candidates))


def _load_uncertainty_decomposition():
    """The shared aleatoric/epistemic split, from the same checkout.

    Found the same way and refused the same way as the parameter spec above: no
    vendored copy and no silent fallback, because a second copy that goes stale
    is the defect and not the fix. This runner used to compute NO split at all
    and QM9 held four different ones, which is failure mode 10 in
    `RERUN_PLAN.md` 0.6 -- one specification, two implementations, drifting.
    """
    candidates = []
    env = os.environ.get('QSAR_QM_MODELS_ROOT')
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent.parent / 'qsar_qm_models',
        Path('/data/stat-cadd/scat9264/qsar_qm_models'),
        Path('/data/stat-ecr/scat9264/qsar_qm_models'),
    ]
    for root in candidates:
        mod_file = Path(root) / 'scripts' / 'uncertainty_decomposition.py'
        if mod_file.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'uncertainty_decomposition', mod_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"  uncertainty split: {mod_file}")
            return mod
    raise RuntimeError(
        "Cannot find scripts/uncertainty_decomposition.py, the shared "
        "definition of the aleatoric/epistemic split. This pipeline must split "
        "uncertainty the same way QM9 does and refuses to define it a second "
        "time. Set QSAR_QM_MODELS_ROOT to the qsar_qm_models checkout. Looked "
        "in: " + ", ".join(str(c) for c in candidates))


def _load_noise_conditions():
    """The conditions the study runs, READ from the settled file, not restated.

    `noise_conditions.json` in the qsar_qm_models checkout is the one place the
    condition set is decided (settled 2026-08-27). It is parsed by
    rust/tests/noise_gates.rs, scripts/test_noise_conditions.py, the figure
    script and the uncertainty job generator -- all on the QM9 side. This side
    held its own hand-typed copy, and the copy had gone stale: it listed
    outlier_p05, which the settled file puts under `not_run`, and it did not
    list outlier_p10, which is the settled contamination setting. Any run
    launched without --conditions therefore produced rows for a retired
    condition and none for the settled one. The job generators were already
    fixed to state the conditions explicitly, so no launched job took this
    default -- but a hand-run invocation still did, and a second copy that
    quietly goes stale is the defect and not the fix.

    Refused rather than defaulted, for the same reason as the parameter spec
    above: guessing the condition set is how the two halves drift.
    """
    candidates = []
    env = os.environ.get('QSAR_QM_MODELS_ROOT')
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates += [
        here.parent.parent.parent / 'qsar_qm_models',
        Path('/data/stat-cadd/scat9264/qsar_qm_models'),
        Path('/data/stat-ecr/scat9264/qsar_qm_models'),
    ]
    for root in candidates:
        cond_file = Path(root) / 'noise_conditions.json'
        if cond_file.exists():
            with open(cond_file) as fh:
                doc = json.load(fh)
            names = [c['name'] for c in doc['stage_1_full_grid']]
            names += [c['name'] for c in doc['stage_2_depth_only']]
            retired = {c['name'] for c in doc.get('not_run', [])}
            clash = retired.intersection(names)
            if clash:
                raise RuntimeError(
                    f"{cond_file} lists {sorted(clash)} as both run and not_run")
            print(f"  noise conditions: {cond_file}  settled "
                  f"{doc.get('settled')}  {names}")
            return names
    raise RuntimeError(
        "Cannot find noise_conditions.json, the settled list of noise "
        "conditions. This pipeline must run the same conditions as QM9 and "
        "refuses to restate them. Set QSAR_QM_MODELS_ROOT to the "
        "qsar_qm_models checkout, or place it beside the KIRBy checkout. "
        "Looked in: " + ", ".join(str(c) for c in candidates))


MODEL_SPEC = _load_model_defaults()
SKLEARN_DEFAULTS = MODEL_SPEC.SKLEARN_DEFAULTS
GP_DEFAULTS = MODEL_SPEC.GP_DEFAULTS
NEURAL_DEFAULTS = MODEL_SPEC.NEURAL_DEFAULTS
BAYESIAN_DEFAULTS = MODEL_SPEC.BAYESIAN_DEFAULTS
sklearn_params = MODEL_SPEC.sklearn_params
should_standardise = MODEL_SPEC.should_standardise
is_binary_matrix = MODEL_SPEC.is_binary_matrix
gp_fit_threads = MODEL_SPEC.gp_fit_threads
_bnn_kl_weight = MODEL_SPEC.bnn_kl_weight
# Whether a Gaussian-process fit returned one number for every molecule it
# scored. This file wrote the same arithmetic out inline; two copies of one rule
# that decides whether a row reaches the results is failure mode 10 of
# RERUN_PLAN.md 0.6.
gp_fit_collapsed = MODEL_SPEC.gp_fit_collapsed

# EVERYTHING in and out of this module is a VARIANCE. The one conversion to a
# spread happens where the rows are written, and never as a sum of two spreads.
UNC_SPLIT = _load_uncertainty_decomposition()
decompose_forest = UNC_SPLIT.decompose_forest
decompose_gp = UNC_SPLIT.decompose_gp
decompose_hetero_gp = UNC_SPLIT.decompose_hetero_gp
decompose_sampling = UNC_SPLIT.decompose_sampling
decompose_single_distribution = UNC_SPLIT.decompose_single_distribution
# NGBoost's model-uncertainty term: the spread of several seeds' means. A single
# distributional fit predicts a variance per molecule and has NOTHING to disagree
# with, so its model half is absent -- and since 2026-08-30 the shared support
# table says NGBoost HAS one, because QM9 fits the ensemble. Without this import
# and the refit below, `assert_matches_support` refuses every NGBoost row on this
# side: 63 tasks of the uncertainty run, on the model the study highlights.
decompose_seed_ensemble = UNC_SPLIT.decompose_seed_ensemble

# THE VARIANCE-HEAD LOSS, from QM9's own loss_functions.py rather than written
# again here. A network with a variance output head -- Kendall & Gal eq. 6, the
# case the uncertainty literature actually holds up -- predicts the value and its
# own observation noise, and is fitted by the negative log likelihood of a
# Gaussian with that predicted variance. It is the only route in either pipeline
# to an aleatoric term that varies per molecule on a NETWORK.
#
# Loaded, not transcribed: the transform between the head's second output and a
# variance is part of the loss, and `split_predictive_head` on the QM9 side reads
# the variance back through that same transform. Two copies of that pair is
# exactly the drift this file refuses everywhere else.
_LOSSES = _load_shared_module('scripts/loss_functions.py', 'loss functions')
HeteroscedasticLoss = _LOSSES.HeteroscedasticLoss
_UTILS = _load_shared_module('scripts/utils.py', 'predictive head split')
split_predictive_head = _UTILS.split_predictive_head
variance_to_std = UNC_SPLIT.variance_to_std
assert_matches_support = UNC_SPLIT.assert_matches_support
unc_support = UNC_SPLIT.support


def bnn_elbo_criterion(base_criterion, model, n_train):
    """Wrap a plain loss so a torchbnn network is fitted on the ELBO.

    Mirrors `bnn_elbo_criterion` in qsar_qm_models/models/models.py, and takes
    its weight from the same spec: 'elbo' means 1 / n_train, because the
    objective is sum_i NLL_i + KL and the criterion here is the MEAN over a
    batch.
    """
    kl = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    weight = _bnn_kl_weight(n_train)

    def criterion(output, target):
        return base_criterion(output, target) + weight * kl(model)

    criterion.kl_weight = weight
    criterion.base = base_criterion
    return criterion


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# The noise conditions, named exactly as in `noiseInject.CONDITIONS` and in
# `rust/src/main.rs`, so the experimental and QM9 halves of the study can be read
# on one axis. The six they replace were one condition at six doses: at a common
# nominal setting they delivered between 0.49x and 2.00x the same amount of
# noise, and their entire apparent severity ordering was that. See
# NOISE_DESIGN.md sections 1, 2 and 2a.
#
# Shape, at matched amount:      gaussian, student_t_nu10/5/3, laplace
# Targeting, at matched amount:  grouped_wider, grouped_shifted, outlier_p01/05/10
# Not dose-matched, own axis:    censoring_10 ... censoring_50
#
# The list itself is READ from noise_conditions.json in the qsar_qm_models
# checkout -- the settled file, not a copy of it. See _load_noise_conditions.
NOISE_CONDITIONS = _load_noise_conditions()

# Which conditions need a scaffold group assignment to be injectable at all.
GROUPED_CONDITIONS = ('grouped_wider', 'grouped_shifted')

# Models that emit a PER-MOLECULE uncertainty. QRF via its quantile spread, GP
# via posterior sd, NGBoost via its predicted distribution, and the four
# Bayesian/variational nets via the spread of NEURAL_DEFAULTS mc_passes
# stochastic forward passes.
# DNN and MLP are deterministic and return None, so cross-fitting them would
# buy nothing. Kept as one list so the runner, the roster and the SLURM
# generator cannot drift apart.
# 'GP' matches 'GP-Hetero' and 'GP-Tanimoto' too: _emits_uncertainty
# compares by prefix, so a decorated variant is not silently left out of the
# out-of-fold pass the way 'GP-Tanimoto' once was.
UNCERTAINTY_MODELS = ['QRF', 'NGBoost', 'GP',
                      'BNN-Full', 'VBLL-Full', 'MLP-BNN-Full', 'MLP-VBLL-Full',
                      # The variance-head networks, added 2026-09-01. Their
                      # aleatoric term varies per molecule, which no other
                      # network here has, so they must be cross-fitted like the
                      # rest or that column is only ever written on test rows,
                      # whose injected noise is zero by construction.
                      'BNN-Full-MVE', 'MLP-BNN-Full-MVE']
# The noise LEVEL is now the amount of noise DELIVERED -- the root-mean-square
# error added, in the label's own units, which for these three datasets is log
# units. Every condition solves for the internal scale that hits it, so a level
# means the same thing across conditions. It did not before: the number was a
# raw scale each condition interpreted its own way.
#
# In log units these are anchored to published assay error (NOISE_DESIGN.md
# section 6.4): logD 0.15 within a laboratory, Caco-2 0.35 between eleven
# laboratories, hERG 0.54. Each grid brackets one unit of real error and runs to
# roughly twice it. Report the resulting fraction of the label spread alongside,
# because one unit of real error is 0.13 of the spread on logD but 0.76 on
# Caco-2 -- a factor of six, and a single shared ladder would mean six different
# experiments.
# ONE grid, in fractions of each dataset's own clean training label spread --
# the same grid QM9 runs, so the same number means the same relative corruption
# everywhere and auc_norm compares directly (author's decision, 2026-08-27).
NOISE_LEVELS_BY_DATASET = {
    'OpenADMET-LogD':          [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
    'OpenADMET-Caco2_Efflux':  [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
    'ChEMBL-hERG-Ki':          [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
}

# What each dataset's top level is worth in log units, and in multiples of its
# published assay error. This is what the shared grid costs, and it belongs in
# any caption that quotes a level.
#
#            label spread   assay error   top = 1.5 spreads   = x assay error
#   LogD          1.19          0.15            1.79               11.9
#   Caco-2        0.44          0.35            0.67                1.9
#   hERG          0.91          0.54            1.37                2.5
#
# The retired grids were in log units, anchored to that error
# (NOISE_DESIGN.md 6.4): LogD [0, 0.15, 0.3, 0.5, 0.75, 1.0], Caco-2
# [0, 0.1, 0.2, 0.35, 0.5, 0.7], hERG [0, 0.15, 0.3, 0.54, 0.8, 1.1]. Their
# problem was the other way round: one unit of real error is 0.13 of the label
# spread on LogD and 0.79 on Caco-2, so the amount of corruption relative to the
# signal differed sixfold and auc_norm was not comparable across datasets.
# Error-over-spread runs 0.13 to 0.79 whichever axis is held fixed; only one of
# the two can be constant.
# Censoring is swept on its own axis -- the fraction of labels clipped -- because
# it has no variance parameter and is not zero-mean. Its LEVEL is that fraction,
# so 0.0 really is a clean baseline. Grid set by the range-finding run: damage
# accelerates smoothly, becoming serious around 20% and severe past 30%, and
# Svensson et al. report 25-63% of labels censored in real industrial assays.
CENSORING_LEVELS = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50]

# Fallback for a dataset with no anchored grid of its own.
NOISE_LEVELS = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
N_FOLDS = 5
# From the shared spec. Uncapped, to match QM9, which fits its GP on the whole
# training split. A 2,000-molecule cap made this a DIFFERENT model, not the same
# model on less data. Set max_train_n in model_defaults.py to reinstate a cap; it
# then applies to every GP variant, not just the one named exactly 'GP'.
# (QM9 used to stack validation into training here; as of 2026-08-27 no model on
# either pipeline does.)
GP_MAX_N = GP_DEFAULTS['max_train_n']
# Anchored to THIS FILE, not the working directory. It was `Path('data_cache')`,
# which resolves against wherever the process was started, so a job launched from
# a different directory found no cache, re-fetched ChEMBL live, and trained on a
# different dataset -- and row order sets the fold geometry, so the split moved
# with it (RERUN_PLAN.md 2.13).
CACHE_DIR = Path(__file__).resolve().parent / 'data_cache'


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

try:
    import torchbnn as bnn
    # Use torchhk (same library as qsar_qm_models/models/models.py:28) so the
    # BayesLinear conversion is byte-for-byte identical between QM9 and
    # validation pipelines. bayesian_torch has a different transform_model
    # signature and was producing divergent BNN-α rows.
    from torchhk import transform_model, transform_layer
    HAS_BAYESIAN_TORCH = True
except ImportError:
    print("WARNING: torchbnn or bayesian-torch not installed, BNN experiments will be skipped")
    HAS_BAYESIAN_TORCH = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("WARNING: xgboost not installed, XGBoost experiments will be skipped")
    HAS_XGBOOST = False

try:
    # GP on PDV is implemented with gauche/gpytorch so the model exactly matches
    # qsar_qm_models/models/models.py:504 (the canonical Gauche ExactGP class).
    # This makes the validation 'GP' rows directly comparable to QM9 gauche_rbf
    # rows. Previously KIRBy used sklearn.GaussianProcessRegressor — functionally
    # equivalent (scaled RBF + learned observation noise) but a different library.
    import gpytorch
    import gauche  # noqa: F401  (verifies the gauche package is importable)
    try:
        from botorch.fit import fit_gpytorch_mll as _fit_gpytorch
    except ImportError:
        from botorch import fit_gpytorch_model as _fit_gpytorch
    HAS_GP = True
except ImportError as _e:
    print(f"WARNING: gauche/gpytorch/botorch not available ({_e}); GP experiments will be skipped")
    HAS_GP = False

try:
    from quantile_forest import RandomForestQuantileRegressor
    HAS_QRF = True
except ImportError:
    print("WARNING: quantile_forest not installed, QRF experiments will be skipped")
    HAS_QRF = False

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    print("WARNING: ngboost not installed, NGBoost experiments will be skipped")
    HAS_NGBOOST = False


def _ngboost_kwargs():
    """Resolve the shared spec's Dist and Score NAMES to the classes.

    QM9 passes Dist=Normal and Score=MLE explicitly while this pipeline used to
    pass neither, so the two agreed only for as long as ngboost's defaults
    happened to match. They are now named in one place and resolved here, and
    scripts/audit_pipeline_parity.py asserts the names still point where they
    used to.
    """
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    from sklearn.tree import DecisionTreeRegressor
    params = dict(SKLEARN_DEFAULTS['ngboost'])
    dist_name, score_name = params.pop('dist'), params.pop('score')
    params['Dist'] = {'Normal': Normal}[dist_name]
    params['Score'] = {'MLE': MLE}[score_name]
    # The base learner, same treatment as Dist and Score: named in the shared
    # spec in parts because it is a constructed object rather than a literal,
    # and built here. Depth 3 and friedman_mse are ngboost's own defaults AND
    # the values its paper used (Duan et al., ICML 2020, section 4); pinning
    # them stops a library upgrade moving them under one pipeline only.
    params['Base'] = DecisionTreeRegressor(
        max_depth=params.pop('base_max_depth'),
        criterion=params.pop('base_criterion'),
        random_state=42)
    # HOW MANY STAGES. ✅ CLOSED 2026-08-28 (author's decision) -- it used to be
    # a gap, and the gap was this:
    #
    # The spec carries the paper's stage selection: fit against a validation
    # split, read the M with the best log-likelihood off that curve, predict at
    # that M rather than at the cap. QM9 did it and this pipeline did not, so
    # NGBoost on an experimental dataset was a more heavily trained model than
    # NGBoost on QM9 while the settings fingerprint on every row said the two
    # matched.
    #
    # They are still popped, because NGBRegressor takes neither at construction:
    # `early_stopping_rounds` is a fit() argument and `use_best_iteration` is
    # ours. They are returned instead, and EarlyStoppingNGBRegressor below
    # applies them.
    stopping = {
        'early_stopping_rounds': params.pop('early_stopping_rounds', None),
        'use_best_iteration': params.pop('use_best_iteration', None),
    }
    return params, stopping


class EarlyStoppingNGBRegressor(NGBRegressor if HAS_NGBOOST else object):
    """NGBoost that stops when it stops improving, the way QM9 does.

    `n_estimators` is a CAP, not the answer. NGBoost's own paper (Duan et al.,
    ICML 2020, section 4) selects the number of rounds on a validation set and
    predicts at that number. QM9 already had a validation split held out of
    every fit and used it; this pipeline's tree path has no such split, so it
    carves one -- scaffold-grouped, by the same helper the neural path here has
    always used, so the rows early stopping is scored on share no scaffold
    framework with the rows fitted.

    Two consequences, both deliberate and both worth stating in the Methods.
    NGBoost here now fits on 80% of the training block while the other tree
    models fit on all of it -- which is already true of the four neural families
    on this side, for the same reason. And it needs the fold's scaffold ids:
    without them it cannot carve a grouped split, and it then trains to the cap
    exactly as before rather than carving an ungrouped one, because scoring
    early stopping on molecules whose frameworks were just fitted is the
    optimism the outer split exists to rule out.

    THE TRAP, and it is the reason `predict` is overridden. ngboost stops at
    (best + patience) and KEEPS every round it fitted; `predict(X)` with no
    `max_iter` then uses all of them. Measured on QM9: best round 249 against
    300 fitted, and the two predictions differ. Those extra rounds are fitted
    past the validation optimum on noisy labels -- the same defect the neural
    models had when they returned the last epoch instead of the best one.
    """

    def __init__(self, *args, early_stopping_rounds=None,
                 use_best_iteration=None, validation_fraction=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        # NOT `self.early_stopping_rounds`, and NOT `self.validation_fraction`.
        # ngboost owns both names and reads them at the top of its own fit():
        # when `self.early_stopping_rounds` is set it OVERRIDES the value passed
        # to fit, and if no validation set came with the call it builds one
        # itself with `train_test_split` -- an ungrouped random split of the
        # training rows. Measured: the fallback path below then stopped at 324
        # rounds having early-stopped against molecules sharing scaffolds with
        # the rows it had just fitted, which is precisely the optimism the outer
        # scaffold split exists to rule out, arriving silently through a library
        # default. Keeping our two settings under our own names leaves ngboost's
        # at None, so it never self-splits and the fit-time value we pass is the
        # one that acts.
        self.stop_rounds_ = int(early_stopping_rounds or 0)
        self.use_best_iteration_ = bool(use_best_iteration)
        self.val_fraction_ = float(validation_fraction)
        self.fit_groups_ = None
        self.best_iter_ = None
        self.stopped_early_ = False

    def set_fit_groups(self, groups):
        """The scaffold ids of the rows the NEXT fit will receive, or None."""
        self.fit_groups_ = None if groups is None else np.asarray(groups)
        return self

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X)
        Y = np.asarray(Y, dtype=float)
        rounds = self.stop_rounds_
        groups = self.fit_groups_
        self.best_iter_ = None
        self.stopped_early_ = False

        if rounds <= 0 or groups is None or len(groups) != len(Y):
            # No curve to read. The stage count is then whatever the cap says,
            # which is what this pipeline did before 2026-08-28.
            return super().fit(X, Y, **kwargs)

        try:
            keep, val = scaffold_validation_carve(
                Y, groups, fold_idx=0, val_frac=self.val_fraction_)
        except RunIntegrityError as exc:
            # Too few scaffolds to carve a grouped split. Say so and train to the
            # cap rather than carving an ungrouped one.
            print(f"      [ngboost] no grouped validation carve ({exc}); "
                  f"training to the cap of {self.n_estimators}", flush=True)
            return super().fit(X, Y, **kwargs)

        super().fit(X[keep], Y[keep], X_val=X[val], Y_val=Y[val],
                    early_stopping_rounds=rounds, **kwargs)
        self.stopped_early_ = True
        if self.use_best_iteration_:
            self.best_iter_ = getattr(self, 'best_val_loss_itr', None)
        return self

    def predict(self, X, max_iter=None):
        return super().predict(X, max_iter=max_iter or self.best_iter_)

    def pred_dist(self, X, max_iter=None):
        return super().pred_dist(X, max_iter=max_iter or self.best_iter_)

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    print("WARNING: lightgbm not installed, LightGBM experiments will be skipped")
    HAS_LGB = False

try:
    from sklearn.svm import SVR
    HAS_SVM = True
except ImportError:
    print("WARNING: sklearn.svm not available, SVM experiments will be skipped")
    HAS_SVM = False

# KIRBy imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_sns,
    create_mhg_gnn,
    create_avalon,
    create_chemberta
)
from kirby.noise_robustness import robustness_metrics
from noiseInject import CONDITIONS, DoseError, NoiseInjectorRegression


# ═══════════════════════════════════════════════════════════════════════════
# SMILES / SCAFFOLD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def standardise_smiles(smi):
    """Canonicalise SMILES, keep largest fragment, remove salts."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_scaffold(smi):
    """Get Murcko scaffold for a SMILES string, CHIRALITY-BLIND.

    `Chem.MolToSmiles` defaults to isomericSmiles=True, so the scaffold key used
    to carry @/@@ and E/Z and stereoisomers of one Bemis-Murcko framework landed
    in DIFFERENT groups -- which GroupKFold can then split between training and
    test. `create_ecfp4` leaves includeChirality at False, so those molecules are
    bit-identical in the fingerprint: measured on the cached data, 219 of 5,039
    LogD test rows (4.35%), 54 of 2,161 Caco-2 (2.50%) and 17 of 1,415 hERG
    (1.20%) had an ECFP4 vector identical to a molecule in their own fold's
    training set, against 0.08%, 0.09% and 0.00% under a chirality-blind key
    (RERUN_PLAN.md 2.13).

    QM9 is chirality-blind on both its split (DeepChem's ScaffoldSplitter
    defaults include_chirality=False) and its noise grouping, so this is also
    what makes 'scaffold split' one protocol across the two pipelines.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)
    except Exception:
        return smi


def assign_scaffold_groups(smiles_list):
    """Assign scaffold group IDs to molecules. Used for the CV fold geometry.

    An ACYCLIC molecule gets its own group. `get_scaffold` returns '' for one --
    MurckoScaffold.GetScaffoldForMol on an acyclic molecule gives an empty core
    -- so every acyclic molecule used to collapse into a single group that
    GroupKFold had to place entirely in one fold. `assign_noise_groups` already
    treats them as singletons, and so does QM9's own grouping; this makes the CV
    path agree. No molecule in LogD, Caco-2 or hERG is acyclic, so this moves no
    fold on the current data (RERUN_PLAN.md 2.13).
    """
    scaffolds = []
    for i, smi in enumerate(smiles_list):
        core = get_scaffold(smi)
        scaffolds.append(core if core else f'__acyclic__{i}')
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    return groups, len(unique_scaffolds)


def assign_noise_groups(smiles_list):
    """Scaffold groups for NOISE INJECTION, with acyclic molecules as singletons.

    Deliberately separate from `assign_scaffold_groups`, which defines the CV
    fold geometry: changing that would silently move every fold in the study.

    The difference is NOISE_DESIGN.md section 2a rule 2. `get_scaffold` returns
    an empty string for an acyclic molecule, so every acyclic molecule lands in
    one bucket. For splitting that is merely coarse; for grouped noise it is
    wrong, because a single group-level draw would then move the whole bucket at
    once. Measured on QM9, where 32.2% of molecules are acyclic, that swings the
    realised dose from 0.795 to 1.493 against a target of 1.0. Each acyclic
    molecule therefore becomes its own group.

    Returns (groups, n_groups, largest_group_share).
    """
    scaffolds = []
    for i, smi in enumerate(smiles_list):
        core = get_scaffold(smi)
        scaffolds.append(f'__acyclic_{i}' if core == '' else core)
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    largest = float(np.bincount(groups).max() / len(groups))
    return groups, len(unique_scaffolds), largest


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def download_openadmet(csv_path=None):
    """Download or load OpenADMET-ExpansionRx data."""
    if csv_path and Path(csv_path).exists():
        print(f"  Loading cached OpenADMET data from {csv_path}")
        return pd.read_csv(csv_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / 'openadmet_train.csv'

    if cached.exists():
        print(f"  Loading cached OpenADMET data from {cached}")
        return pd.read_csv(cached)

    url = (
        "https://huggingface.co/datasets/openadmet/"
        "openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv"
    )
    print(f"  Downloading OpenADMET data from HuggingFace...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    print(f"  Saved to {cached}")
    return pd.read_csv(cached)


def fetch_chembl_herg_ki():
    """Extract hERG (CHEMBL240) Ki data via ChEMBL REST API."""
    print("  Fetching ChEMBL hERG Ki data...")

    cached = CACHE_DIR / "chembl_herg_ki.csv"
    if cached.exists():
        print(f"  [cached] {cached}")
        stamp = CACHE_DIR / "chembl_herg_ki.provenance.json"
        if stamp.exists():
            _p = json.loads(stamp.read_text())
            print(f"    fetched {_p.get('fetched_utc')} from ChEMBL release "
                  f"{_p.get('chembl_release')}, {_p.get('n_compounds')} compounds")
        else:
            print(f"    WARNING: no provenance file beside it. This cache "
                  f"predates the stamp, so which ChEMBL release it came from is "
                  f"not recorded.")
        return pd.read_csv(cached)

    # No cache: this is about to fetch a MOVING dataset over the network. ChEMBL
    # grows, so two runs at different times train on different data, and nothing
    # recorded which release produced a result. Refuse unless the caller says so.
    if os.environ.get('KIRBY_ALLOW_CHEMBL_FETCH') != '1':
        raise RuntimeError(
            f"{cached} does not exist, and fetching hERG live from ChEMBL would "
            f"pull whatever release is current today -- which is not the dataset "
            f"any existing result was produced from. Copy the cached CSV into "
            f"place, or set KIRBY_ALLOW_CHEMBL_FETCH=1 to fetch deliberately; "
            f"the release and date are then written beside it.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_ID = "CHEMBL240"
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    all_records = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "target_chembl_id": TARGET_ID,
            "standard_type": "Ki",
            "pchembl_value__isnull": "false",
            "standard_relation": "=",
            "data_validity_comment__isnull": "true",
            "limit": limit,
            "offset": offset,
            "format": "json",
        }

        print(f"    Querying ChEMBL API (offset={offset})...")
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    ChEMBL API request failed: {e}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            all_records.append({
                "canonical_smiles": act.get("canonical_smiles"),
                "pchembl_value": act.get("pchembl_value"),
                "assay_type": act.get("assay_type"),
            })

        page_meta = data.get("page_meta", {})
        if page_meta.get("next") is None:
            break
        offset += limit
        time.sleep(0.5)

    if not all_records:
        raise RuntimeError("No hERG Ki records retrieved from ChEMBL!")

    df = pd.DataFrame(all_records)
    print(f"    Retrieved {len(df)} raw hERG Ki records")

    # Filter to binding assays
    if "assay_type" in df.columns:
        df = df[df["assay_type"] == "B"].copy()

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])

    # Deduplicate: median pchembl per compound
    grouped = df.groupby("canonical_smiles")["pchembl_value"]
    medians = grouped.median().reset_index()
    stds = grouped.std().reset_index().rename(columns={"pchembl_value": "std"})
    merged = medians.merge(stds, on="canonical_smiles")

    # Remove high-variance compounds (std > 1.0 log unit)
    merged = merged[(merged["std"].isna()) | (merged["std"] <= 1.0)]

    result = merged[["canonical_smiles", "pchembl_value"]].copy()
    result.columns = ["SMILES", "pKi"]

    result.to_csv(cached, index=False)
    # What this fetch actually pulled, beside the file it pulled it into.
    try:
        _rel = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json",
                            timeout=30).json().get('chembl_db_version', 'unknown')
    except Exception:
        _rel = 'unknown'
    (CACHE_DIR / "chembl_herg_ki.provenance.json").write_text(json.dumps({
        'fetched_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'chembl_release': _rel,
        'target': TARGET_ID,
        'n_compounds': int(len(result)),
    }, indent=1))
    print(f"    Final hERG Ki dataset: {len(result)} compounds "
          f"(ChEMBL release {_rel})")

    return result


def load_openadmet_endpoint(df, endpoint_col, log_transform=False):
    """Extract a single endpoint from OpenADMET, standardize, deduplicate."""
    smiles_col = next(c for c in df.columns if c.upper() == 'SMILES')

    sub = df[[smiles_col, endpoint_col]].dropna(subset=[endpoint_col]).copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors='coerce')
    sub = sub.dropna()

    # Standardize SMILES
    sub['std_smiles'] = sub[smiles_col].apply(standardise_smiles)
    sub = sub.dropna(subset=['std_smiles'])

    # Deduplicate by canonical SMILES: take median target
    sub = sub.groupby('std_smiles').agg({endpoint_col: 'median'}).reset_index()

    smiles_arr = sub['std_smiles'].values
    labels_arr = sub[endpoint_col].values.astype(np.float64)

    if log_transform:
        # DROP a non-positive value, never clip it. Clipping to 1e-10 made the
        # label exactly -10.0, which is finite, so the isfinite filter below kept
        # it: on Caco-2 (label mean 0.293, spread 0.445) that is 23 spreads below
        # the mean and would move the R2 of every row in the dataset. The current
        # extraction has no non-positive efflux ratio, so this is latent -- which
        # is why it survived (RERUN_PLAN.md 2.13).
        positive = labels_arr > 0
        if not positive.all():
            print(f"  {endpoint_col}: dropping {int((~positive).sum())} "
                  f"non-positive value(s) before the log transform")
            smiles_arr = smiles_arr[positive]
            labels_arr = labels_arr[positive]
        labels_arr = np.log10(labels_arr)
        valid = np.isfinite(labels_arr)
        smiles_arr = smiles_arr[valid]
        labels_arr = labels_arr[valid]

    return smiles_arr, labels_arr


def load_chembl_herg():
    """Load ChEMBL hERG Ki as regression dataset."""
    df = fetch_chembl_herg_ki()

    # Standardize SMILES
    df['std_smiles'] = df['SMILES'].apply(standardise_smiles)
    df = df.dropna(subset=['std_smiles'])

    smiles_arr = df['std_smiles'].values
    # The cached CSV has been written with different label column names by
    # different versions of fetch_chembl_herg_ki (a stale cache carrying
    # 'pChEMBL' rather than 'pKi' raised KeyError and killed the whole run).
    # Accept any of the known spellings and fail with a readable message.
    label_col = next((c for c in ('pKi', 'pChEMBL', 'pchembl_value', 'pchembl')
                      if c in df.columns), None)
    if label_col is None:
        raise KeyError(
            f"hERG label column not found. Looked for pKi/pChEMBL/pchembl_value; "
            f"the file has: {list(df.columns)}")
    # Deduplicate by STANDARDISED SMILES, taking the median -- the same rule the
    # OpenADMET loader applies.
    #
    # This step did not exist. fetch_chembl_herg_ki dedupes on the RAW ChEMBL
    # SMILES, before standardisation, so two records that standardise to the
    # same molecule survived as two rows with byte-identical features and
    # different pKi. hERG then carried an irreducible-error floor from replicate
    # labels that LogD and Caco-2 did not -- and the study's whole quantity is
    # how R2 decays as label noise is injected, so a dataset that already
    # contains replicate label noise decays differently for reasons that have
    # nothing to do with the injection (RERUN_PLAN.md 2.13).
    _before = len(df)
    df = (df[['std_smiles', label_col]]
          .groupby('std_smiles', as_index=False)
          .agg({label_col: 'median'}))
    if len(df) < _before:
        print(f"  hERG: {_before} -> {len(df)} compounds after deduplicating on "
              f"the standardised SMILES (median pKi)")
    smiles_arr = df['std_smiles'].values
    labels_arr = df[label_col].values.astype(np.float64)

    return smiles_arr, labels_arr


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK MODELS
# ═══════════════════════════════════════════════════════════════════════════

class DeterministicRegressor(nn.Module):
    """DNN matching the qsar_qm_models BNN-Full / VBLL-Full base architecture.

    Hidden sizes [128, 64], ReLU, Dropout(0.2), NO BatchNorm — mirrors
    qsar_qm_models/scripts/investigate_pdv.py DEFAULT_PARAMS['dnn'] and the
    DNNModel class in that file. Required for QM9 and validation BNN-α /
    VBLL-α rows to be directly comparable.
    """
    def __init__(self, input_dim, hidden_size1=128, hidden_size2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()


class MLPRegressor(nn.Module):
    """MLP matching qsar_qm_models/models/models.py:483-502 MLPRegressor (NN-β base).

    Built at hidden_size=128 by the caller, matching QM9: Linear(input→128) →
    ReLU → Linear(128→128) → ReLU → Dropout(0.2) → Linear(128→1). The class
    default below is 32 and must NOT be relied on — QM9 passes 128.
    Used as the base for BNN-β (`mlp_bnn_full`) and VBLL-β (`mlp_vbll`) in the
    paper Wilcoxon comparisons.
    """
    def __init__(self, input_dim, hidden_size=32, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.dropout(x)
        return self.output_layer(x).squeeze()


if HAS_BAYESIAN_TORCH:
    def apply_bayesian_transformation(model):
        """Full BNN: replace every nn.Linear with torchbnn BayesLinear.

        Mirrors qsar_qm_models/models/models.py:1046 apply_bayesian_transformation.
        """
        transform_model(
            model, nn.Linear, bnn.BayesLinear,
            args={"prior_mu": BAYESIAN_DEFAULTS['bnn_prior_mu'],
                  "prior_sigma": BAYESIAN_DEFAULTS['bnn_prior_sigma'],
                  "in_features": ".in_features",
                  "out_features": ".out_features", "bias": ".bias"},
            attrs={"weight_mu": ".weight"})
        return model

    def apply_bayesian_transformation_last_layer(model):
        """Last-layer BNN: replace only the final nn.Linear with BayesLinear.

        Kept for completeness; not used by default (see model dispatch below).
        Mirrors qsar_qm_models/models/models.py:1076.
        """
        last_name, last_mod = None, None
        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, nn.Linear):
                last_name, last_mod = name, mod
                break
        if last_mod is None:
            raise ValueError("No nn.Linear found")
        bl = transform_layer(
            last_mod, nn.Linear, bnn.BayesLinear,
            args={"prior_mu": BAYESIAN_DEFAULTS['bnn_prior_mu'],
                  "prior_sigma": BAYESIAN_DEFAULTS['bnn_prior_sigma'],
                  "in_features": ".in_features",
                  "out_features": ".out_features", "bias": ".bias"},
            attrs={"weight_mu": ".weight"})

        def _set(obj, path, val):
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        _set(model, last_name, bl)
        return model


# ─── Variational Bayesian Last Layer (full-network variant) ───────────────────
# Mirrors qsar_qm_models/models/models.py:1131-1349 verbatim — VBLLLayer,
# VBLLLoss, and the full-network transformation. NOT wrapped in HAS_BAYESIAN_TORCH
# because VBLL is a pure PyTorch implementation and has no torchbnn dependency.

class VBLLLayer(nn.Module):
    """Variational Bayesian Last Layer (Harrison 2024).

    Mean-field variational posterior q(W) = N(weight_mu, diag(exp(weight_log_sigma)^2))
    over the weight matrix, standard normal prior p(W) = N(0, I), trained via the
    reparameterization trick.
    """
    def __init__(self, in_features, out_features, prior_mu=None, prior_sigma=None,
                 heteroscedastic=False):
        super().__init__()
        # From the shared spec (RERUN_PLAN.md 2.13).
        _init_log_sigma = BAYESIAN_DEFAULTS['vbll_init_log_sigma']
        _init_log_noise = BAYESIAN_DEFAULTS['vbll_init_log_noise_var']
        self.in_features = in_features
        self.out_features = out_features
        # When False the observation noise is ONE NUMBER for the whole fit, so
        # this model's data-noise term is identical for every molecule and its
        # correlation with per-molecule injected noise is zero however good the
        # model is. When True the noise is predicted FROM THE INPUT, following
        # `HetRegression` in research_archive/f692d614/vbll_regression.py:277-370
        # -- the noise head is variational too, so it carries a KL term rather
        # than being a free function fitting residuals. MIRRORS VBLLLayer in
        # qsar_qm_models/models/models.py; the two must move together
        # (RERUN_PLAN.md 5.5f).
        self.heteroscedastic = bool(heteroscedastic)
        self.prior_mu = (BAYESIAN_DEFAULTS['vbll_prior_mu']
                         if prior_mu is None else prior_mu)
        self.prior_sigma = (BAYESIAN_DEFAULTS['vbll_prior_sigma']
                            if prior_sigma is None else prior_sigma)

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(
            torch.full((out_features, in_features), _init_log_sigma))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(
            torch.full((out_features,), _init_log_sigma))

        # Learned observation noise (aleatoric)
        self.log_noise_var = nn.Parameter(torch.tensor(_init_log_noise))

        if self.heteroscedastic:
            # A variational map from the features to a log noise variance, one
            # value per molecule. Initialised to predict nothing, so the layer
            # starts at the single-noise value above and has to earn any
            # departure from it.
            self.noise_mu = nn.Parameter(torch.zeros(1, in_features))
            self.noise_log_sigma = nn.Parameter(
                torch.full((1, in_features), _init_log_sigma))
            self.noise_bias_mu = nn.Parameter(torch.tensor([_init_log_noise]))
            self.noise_bias_log_sigma = nn.Parameter(
                torch.tensor([_init_log_sigma]))
        # What the last forward pass predicted, so the loss can read it without
        # every call site growing an argument.
        self._last_noise_var = None

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @property
    def noise_var(self):
        """The single observation noise. Meaningful only when this layer is
        homoscedastic; when it is not, use `noise_var_for(x)`."""
        return torch.exp(self.log_noise_var)

    def noise_var_for(self, x):
        """Observation noise variance, PER MOLECULE when heteroscedastic.

        Clamped the way the heteroscedastic loss clamps its log variance, so
        neither can drive the other's gradient to infinity.
        """
        if not self.heteroscedastic:
            n = torch.exp(self.log_noise_var)
            return n.expand(x.shape[0]) if x.dim() > 1 else n
        w = self.noise_mu + torch.exp(self.noise_log_sigma) * torch.randn_like(
            self.noise_mu)
        b = self.noise_bias_mu + torch.exp(
            self.noise_bias_log_sigma) * torch.randn_like(self.noise_bias_mu)
        log_var = F.linear(x, w, b).squeeze(-1)
        return torch.exp(torch.clamp(log_var, -10.0, 10.0))

    def kl_divergence(self):
        prior_var = self.prior_sigma ** 2
        w_var = torch.exp(2.0 * self.weight_log_sigma)
        kl_w = 0.5 * torch.sum(
            w_var / prior_var
            + ((self.prior_mu - self.weight_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.weight_log_sigma
        )
        b_var = torch.exp(2.0 * self.bias_log_sigma)
        kl_b = 0.5 * torch.sum(
            b_var / prior_var
            + ((self.prior_mu - self.bias_mu) ** 2) / prior_var
            - 1.0
            + math.log(prior_var) - 2.0 * self.bias_log_sigma
        )
        kl = kl_w + kl_b

        if self.heteroscedastic:
            # The noise head is variational, so it is regularised toward the
            # same prior. Without this it is free to fit residual noise
            # arbitrarily and the data-noise term stops being a posterior
            # quantity at all.
            n_var = torch.exp(2.0 * self.noise_log_sigma)
            kl = kl + 0.5 * torch.sum(
                n_var / prior_var
                + ((self.prior_mu - self.noise_mu) ** 2) / prior_var
                - 1.0
                + math.log(prior_var) - 2.0 * self.noise_log_sigma)
            nb_var = torch.exp(2.0 * self.noise_bias_log_sigma)
            kl = kl + 0.5 * torch.sum(
                nb_var / prior_var
                + ((self.prior_mu - self.noise_bias_mu) ** 2) / prior_var
                - 1.0
                + math.log(prior_var) - 2.0 * self.noise_bias_log_sigma)
        return kl

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        self._last_noise_var = self.noise_var_for(x)
        return F.linear(x, weight, bias)


class VBLLLoss(nn.Module):
    """ELBO loss for VBLL: Gaussian NLL (learned noise) + sum(KL) / n_data."""
    def __init__(self, model, n_data):
        super().__init__()
        self.model = model
        self.n_data = n_data
        self.vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
        self.output_layer = self.vbll_layers[-1] if self.vbll_layers else None

    def forward(self, pred, target):
        if self.output_layer is None:
            return nn.MSELoss()(pred, target)
        # When the layer predicts a noise PER MOLECULE, use what it predicted on
        # the forward pass that produced `pred`. Using the single `noise_var`
        # here instead would fit the mean with a noise head attached and never
        # train that head, which writes a per-molecule column meaning nothing.
        cached = getattr(self.output_layer, '_last_noise_var', None)
        if self.output_layer.heteroscedastic and cached is not None:
            noise_var = cached.reshape(-1, 1) if pred.dim() > 1 else cached
        else:
            noise_var = self.output_layer.noise_var
        nll = 0.5 * torch.log(noise_var) + 0.5 * ((pred - target) ** 2) / noise_var
        nll = nll.mean()
        kl = sum(layer.kl_divergence() for layer in self.vbll_layers)
        return nll + kl / self.n_data


def apply_bayesian_transformation_full_variational(model, heteroscedastic=False):
    """Replace ALL nn.Linear layers with VBLLLayer. Initializes each VBLLLayer
    weight_mu from the corresponding pretrained Linear weights.

    Mirrors qsar_qm_models/models/models.py:1295.
    """
    linear_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_layers.append((name, mod))

    def _set(obj, path, val):
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    # Only the OUTPUT layer gets the noise head: a hidden layer carries no
    # observation noise to make heteroscedastic.
    last_name = linear_layers[-1][0] if linear_layers else None
    for name, lin in linear_layers:
        vbll = VBLLLayer(lin.in_features, lin.out_features,
                         heteroscedastic=(heteroscedastic and name == last_name))
        with torch.no_grad():
            vbll.weight_mu.copy_(lin.weight.data)
            if lin.bias is not None:
                vbll.bias_mu.copy_(lin.bias.data)
        _set(model, name, vbll)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# GAUSSIAN PROCESS (gauche / gpytorch — mirrors qsar models.py:504)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_GP:
    def _init_rbf_lengthscale(model, train_x):
        """Start the RBF lengthscale at the median distance between training
        molecules. A transcription of `init_rbf_lengthscale` in
        `qsar_qm_models/models/models.py`, reading the same two spec keys.

        gpytorch starts it at softplus(0), about 0.69. Typical distances between
        molecules run from 17 on the descriptor vector to 1,100 on the learned
        embeddings, so at 0.69 every molecule looks infinitely far from every
        other: the kernel matrix is the identity, the marginal likelihood is flat,
        and neither botorch nor Adam has a gradient to follow. The fit returns one
        constant for every molecule, that constant still gets scored, and the
        score reads as a weak representation rather than as a failed fit.

        That is what produced R2 = -0.0158 for MHG-GNN on QM9. Started from the
        data instead, the SAME features reached 0.68 to 0.76. QM9 was fixed on
        2026-08-26; this side was not, and the word `lengthscale` appeared nowhere
        in this file -- so every Gaussian-process number on LogD, Caco-2 and hERG
        was a failed fit, on the continuous descriptor vector, which is the only
        representation the Gaussian process runs on here.

        Only the RBF kernel has a lengthscale. Tanimoto is a ratio and has none,
        so this returns None there and changes nothing.
        """
        if not GP_DEFAULTS['init_lengthscale_from_data']:
            return None
        base = getattr(model.covar_module, 'base_kernel', None)
        if base is None or not hasattr(base, 'lengthscale') or base.lengthscale is None:
            return None
        with torch.no_grad():
            n = min(GP_DEFAULTS['lengthscale_probe_n'], len(train_x))
            probe = train_x[torch.randperm(len(train_x))[:n]]
            median = torch.cdist(probe, probe).flatten().median().item()
        if not np.isfinite(median) or median <= 0:
            print("      [gp] median distance is not usable; leaving the "
                  "lengthscale at its default", flush=True)
            return None
        base.lengthscale = median
        print(f"      [gp] lengthscale started at the median distance between "
              f"molecules: {median:.4g}", flush=True)
        return median


    class _GaucheExactGP(gpytorch.models.ExactGP):
        """ExactGP with a ScaleKernel wrapper. Mirrors qsar_qm_models
        models.py:504 'Gauche' class verbatim — same mean/cov structure so
        the validation GP rows are directly comparable to QM9 gauche_rbf rows.
        """
        def __init__(self, train_x, train_y, likelihood, kernel_class):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel_class())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))


    class GaussianProcessGauche:
        """sklearn-compatible wrapper around the gauche/gpytorch ExactGP so the
        existing run_tree_experiment GP path works unchanged. Default kernel is
        RBF (matches qsar's gauche_rbf — PDV-only in this pipeline).

        Name intentionally contains 'GaussianProcess' so the dispatch substring
        check in run_tree_experiment (`'GaussianProcess' in str(type(mdl))`)
        keeps routing predictions through the (mean, std) branch.
        """
        def __init__(self, kernel=None, init_noise=None,
                     init_outputscale=None, max_fit_iter=None):
            self.kernel_name = GP_DEFAULTS['kernel'] if kernel is None else kernel
            self.init_noise = (GP_DEFAULTS['likelihood_noise']
                               if init_noise is None else init_noise)
            self.init_outputscale = (GP_DEFAULTS['outputscale']
                                     if init_outputscale is None else init_outputscale)
            self.max_fit_iter = (GP_DEFAULTS['fallback_adam_iters']
                                 if max_fit_iter is None else max_fit_iter)
            self.model_ = None
            self.likelihood_ = None
            # The split from the last predict(), as variances. The
            # heteroscedastic subclass overwrites both with per-molecule values.
            self.aleatoric_var_ = None
            self.epistemic_var_ = None
            # 'botorch' or 'adam_fallback'. Written into every results row as
            # gp_fit_method: the fallback fits a DIFFERENT optimisation and used
            # to leave no trace outside this attribute.
            self.fit_method_ = None
            self.fit_fallback_ = None

        def _kernel_class(self):
            if self.kernel_name == 'rbf':
                return gpytorch.kernels.RBFKernel
            if self.kernel_name == 'tanimoto':
                # Lazy import — only needed if caller switches to Tanimoto.
                from gauche.kernels.fingerprint_kernels.tanimoto_kernel \
                    import TanimotoKernel
                return TanimotoKernel
            raise ValueError(f"unknown kernel {self.kernel_name!r}")

        def fit(self, X, y):
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
            yt = torch.from_numpy(np.asarray(y, dtype=np.float64))
            # The labels this fit saw, so predict() can say whether it returned
            # its prior over the block it was asked to score.
            self.y_fit_ = np.asarray(y, dtype=float)
            if self.kernel_name == 'tanimoto' and not is_binary_matrix(Xt.numpy()):
                # The Tanimoto ratio <x,y>/(|x|^2 + |y|^2 - <x,y>) is a
                # similarity in [0, 1] on BINARY vectors. On centred, negative,
                # continuous values it is not a similarity at all and its
                # denominator can vanish or go negative, so the number it
                # produces would be meaningless rather than merely different
                # (RERUN_PLAN.md 2.13).
                raise ValueError(
                    "the Tanimoto kernel was asked for on a feature matrix that "
                    "is not binary. It is defined on binary fingerprints only; "
                    "use --gp-kernel rbf, or a representation that is not "
                    "standardised.")
            self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood(
                noise=self.init_noise)
            self.model_ = _GaucheExactGP(Xt, yt, self.likelihood_,
                                         self._kernel_class())
            # QM9 computed an outputscale and never applied it, so its
            # ScaleKernel actually started at gpytorch's softplus(0) ~ 0.693.
            # Both sides now apply the spec value.
            if GP_DEFAULTS['apply_outputscale']:
                self.model_.covar_module.outputscale = self.init_outputscale
            self.lengthscale_init_ = _init_rbf_lengthscale(self.model_, Xt)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood_, self.model_)
            # botorch >= ~0.12 requires its fitter's argument to be a botorch
            # Model (it calls transform_inputs); _GaucheExactGP is a plain
            # gpytorch ExactGP, so on newer botorch this raises
            #   '_GaucheExactGP' object has no attribute 'transform_inputs'
            # and -- because run_dataset catches per-condition exceptions -- the
            # whole GP arm produced NOTHING while the job still exited 0.
            # Fall back to a plain gpytorch Adam loop, which is version-proof.
            try:
                with gp_fit_threads():
                    _fit_gpytorch(mll)
                self.fit_method_ = 'botorch'
            except Exception as e:
                self.fit_fallback_ = f"{type(e).__name__}: {e}"
                self.fit_method_ = 'adam_fallback'
                print(f"      [gp] botorch fitter refused this model "
                      f"({self.fit_fallback_}); using the Adam fallback. "
                      f"Recorded as gp_fit_method=adam_fallback.", flush=True)
                self.model_.train()
                self.likelihood_.train()
                opt = torch.optim.Adam(self.model_.parameters(),
                                       lr=GP_DEFAULTS['fallback_adam_lr'])
                n_iter = self.max_fit_iter
                with gp_fit_threads():
                    for _ in range(n_iter):
                        opt.zero_grad()
                        loss = -mll(self.model_(Xt), yt)
                        loss.backward()
                        opt.step()
            return self

        def predict(self, X, return_std=False):
            self.model_.eval()
            self.likelihood_.eval()
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
            # Prediction solves against the same kernel matrix, so it can crash
            # the same way the fit does. Guard it too.
            with gp_fit_threads(), torch.no_grad():
                preds = self.model_(Xt)
                mean = preds.mean.detach().cpu().numpy()
                var = preds.variance.detach().cpu().numpy()
            # The split, recorded as VARIANCES. The latent variance is the model
            # half; the data half is the likelihood noise, and a homoscedastic
            # likelihood has exactly ONE of those, so it is the same number for
            # every molecule. That is why this process cannot answer a
            # per-molecule question and the heteroscedastic one below can.
            latent_var = np.clip(var, 1e-12, None)
            self.epistemic_var_ = latent_var
            self.aleatoric_var_ = np.full_like(
                latent_var, float(self.likelihood_.noise.item()))
            # A block the process knows nothing about gets its PRIOR back: one
            # predicted value and one latent variance for every molecule in it.
            # The out-of-fold split is scaffold-grouped, so a held-out fold is
            # whole scaffold families the fit set does not contain, and this is
            # where it happens. The model half is declared per molecule, so
            # written flat it would be refused by the writer's guard and the run
            # would stop; blanked, the row keeps its prediction and its
            # observation-noise half and says plainly that the process had no
            # per-molecule answer here (RERUN_PLAN.md 3.1d, 5.5i).
            #
            # The REPORTED total is untouched: it is a statement about the
            # observation and it is still the sum of the two variances the fit
            # produced. Only the half that is declared per molecule is withheld.
            self.collapsed_ = bool(
                getattr(self, 'y_fit_', None) is not None
                and gp_fit_collapsed(mean, self.y_fit_)[0])
            if self.collapsed_:
                print(f"      [gp] the fit returned its prior over these "
                      f"{len(mean)} molecules -- no per-molecule model "
                      f"uncertainty for them.", flush=True)
                self.epistemic_var_ = np.full(len(mean), np.nan)
            if return_std:
                # BOTH halves, added as VARIANCES and converted once. This used
                # to return the latent spread alone, which is the model half --
                # the observation noise was computed on the two lines above and
                # thrown away.
                #
                # QM9 reports the total (`models/models.py`, the gauche trainer:
                # `sqrt(likelihood(model(x)).variance)`), so the column called
                # `uncertainty` meant two different quantities on the two halves
                # of the study and every coverage number was measured against a
                # different definition. Measured on one fitted process: the total
                # is 2.94x the latent spread, and coverage at one standard
                # deviation at noise level 1.0 is 0.690 against 0.230.
                #
                # Coverage asks how often the TRUE LABEL lands inside the
                # interval, and a label is an observation, so the interval has to
                # carry the observation noise or it under-covers by construction.
                # The heteroscedastic subclass below has always done this.
                #
                # Adding the two VARIANCES, never the two standard deviations:
                # summing spreads overstates the total by up to 41%.
                return mean, np.sqrt(self.aleatoric_var_ + latent_var)
            return mean


    class HeteroscedasticGaussianProcessGauche(GaussianProcessGauche):
        """A Gaussian process that predicts a noise level for EACH molecule.

        The ordinary one above learns a single likelihood noise for the whole
        dataset, so it reports the same data-noise number for every molecule and
        cannot answer a per-molecule question at all. This one fits a small
        network alongside the process that predicts the observation noise from
        the molecule's own features, so the two halves of the predictive
        variance come apart:

            data noise   the network's output for that molecule
            model noise  the process's own latent variance at that molecule

        Measured on 4,000 QM9 molecules by chat I: the data half tracks the true
        corruption at +0.79 while the model half sits at -0.12 against the same
        thing, for the same accuracy as the ordinary process (0.5315 against
        0.5318). Both forest halves track it at +0.84 and +0.81 -- the same
        signal twice -- so this is the only model measured so far that actually
        separates them.

        MIRRORS `train_heteroscedastic_gp` in qsar_qm_models/models/models.py:
        same 64-64-1 softplus network, same joint Adam loop, same learning rates
        (0.1 for the process, 0.001 for the network), same loss
        `0.5*log(v) + r^2/(2v)` on the squared residuals. The two must move
        together or a `GP-Hetero` row from one pipeline and one from the other
        are different models under one name -- the failure RERUN_PLAN.md 3.4.2
        is about.

        The class name contains 'GaussianProcess' so `run_tree_experiment` keeps
        routing it through the (mean, std) branch, and the model is registered as
        `GP-Hetero` so `_emits_uncertainty` matches it by prefix.
        """

        # Kept as attributes rather than returned, because `predict` has to keep
        # the (mean, std) signature the dispatch expects.
        def __init__(self, *args, epochs=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.epochs = (NEURAL_DEFAULTS['training']['epochs']
                           if epochs is None else epochs)
            self.noise_net_ = None
            self.aleatoric_var_ = None
            self.epistemic_var_ = None

        @staticmethod
        def _noise_net(input_dim):
            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 1), torch.nn.Softplus())

        def fit(self, X, y):
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
            self.y_fit_ = np.asarray(y, dtype=float)
            self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood()
            self.model_ = _GaucheExactGP(Xt, yt, self.likelihood_,
                                         self._kernel_class())
            if GP_DEFAULTS['apply_outputscale']:
                self.model_.covar_module.outputscale = self.init_outputscale
            # Same start as the ordinary process above. Without it this one
            # collapses the same way, and it is the only model on either roster
            # that reports both halves of its uncertainty per molecule.
            self.lengthscale_init_ = _init_rbf_lengthscale(self.model_, Xt)
            self.noise_net_ = self._noise_net(Xt.shape[1])

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood_, self.model_)
            self.model_.train(); self.likelihood_.train(); self.noise_net_.train()
            opt = torch.optim.Adam([
                {'params': self.model_.parameters(), 'lr': 0.1},
                {'params': self.noise_net_.parameters(), 'lr': 0.001}])

            # No botorch path: the two losses are optimised together, which
            # botorch's fitter does not express. Recorded so no row claims a
            # fit method it did not use.
            self.fit_method_ = 'joint_adam'
            with gp_fit_threads():
                for _ in range(self.epochs):
                    opt.zero_grad()
                    gp_loss = -mll(self.model_(Xt), yt)
                    # THE RESIDUAL MUST COME FROM THE FITTED PROCESS, NOT THE PRIOR.
                    #
                    # In train mode gpytorch returns the PRIOR at the training
                    # inputs, so `y - mean` is `y - a constant` -- the spread of
                    # the labels themselves, which is dominated by real
                    # structure and says almost nothing about which labels are
                    # corrupted. Measured on 300 molecules where the loud half
                    # carries 100x the noise variance of the quiet half:
                    #
                    #   prior residuals   loud 6.44  quiet 5.92   ratio  1.09
                    #   fitted residuals  loud 0.88  quiet 0.033  ratio 27.08
                    #
                    # Trained on the first, the network's output is flat: the
                    # rank correlation with the true corruption sits between
                    # -0.12 and +0.09 at every learning rate from 0.001 to 0.05
                    # and every length from 60 to 2000 epochs. It is not a
                    # convergence problem, it is the wrong target.
                    with torch.no_grad():
                        self.model_.eval()
                        resid = (yt - self.model_(Xt).mean) ** 2
                        self.model_.train()
                    pred_var = self.noise_net_(Xt).squeeze(-1) + 1e-4
                    noise_loss = torch.mean(
                        0.5 * torch.log(pred_var) + resid / (2 * pred_var))
                    (gp_loss + noise_loss).backward()
                    opt.step()
            return self

        def predict(self, X, return_std=False):
            self.model_.eval(); self.likelihood_.eval(); self.noise_net_.eval()
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with gp_fit_threads(), torch.no_grad():
                posterior = self.model_(Xt)
                mean = posterior.mean.detach().cpu().numpy()
                latent_var = posterior.variance.detach().cpu().numpy()
                noise_var = (self.noise_net_(Xt).squeeze(-1)
                             + 1e-4).detach().cpu().numpy()
            # Everything here is a VARIANCE. One conversion to a spread, at the
            # point of returning, and never a sum of two spreads.
            self.aleatoric_var_ = np.clip(noise_var, 1e-12, None)
            latent_var = np.clip(latent_var, 1e-12, None)
            self.epistemic_var_ = latent_var
            # The same prior-over-a-whole-block case as the ordinary process
            # above. The NOISE half is untouched -- it comes from the network per
            # molecule and is what this model exists for -- and so is the
            # reported total.
            self.collapsed_ = bool(
                getattr(self, 'y_fit_', None) is not None
                and gp_fit_collapsed(mean, self.y_fit_)[0])
            if self.collapsed_:
                print(f"      [gp] the noise-predicting fit returned its prior "
                      f"over these {len(mean)} molecules -- no per-molecule "
                      f"model uncertainty for them.", flush=True)
                self.epistemic_var_ = np.full(len(mean), np.nan)
            if return_std:
                total = self.aleatoric_var_ + latent_var
                return mean, np.sqrt(total)
            return mean


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_neural_regression(X_train, y_train, X_val, y_val, X_test,
                            model_type='deterministic', epochs=None, lr=None,
                            y_scaler=None):
    """Train regression neural network.

    Returns (predictions, uncertainties, aleatoric_var, epistemic_var). The last
    two are on the RAW label scale and are None for a model that has no such
    component.

    Targets are z-score normalized for training and predictions/uncertainties
    are inverse-transformed back to the raw label scale before returning.
    Without this, MSE on raw-scale labels diverges for wide-range datasets
    (e.g. hERG) and high-dim reps (e.g. MHG-GNN-pretrained).

    `y_scaler` must be fitted on the CLEAN training labels and is passed in.
    It used to be fitted here, on whatever `y_train` held -- which is the
    already-corrupted labels. The divisor was then
    sqrt(sd_clean^2 + dose^2), which GROWS with the level, so the standardised
    target shrank as the dose rose and the same nominal level posed a different
    optimisation problem: a fixed learning rate, weight decay and absolute
    early-stopping tolerance all mean something else at each level, and the
    degradation curve absorbed it. At the top of the Caco-2 grid the divisor was
    about 1.8x its clean value. QM9 standardises against the clean TRAINING
    spread in Rust; this now does the same (RERUN_PLAN.md 2.13).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _t = NEURAL_DEFAULTS['training']
    epochs = _t['epochs'] if epochs is None else epochs
    lr = _t['lr'] if lr is None else lr

    if not _t['standardise_targets']:
        raise ValueError(
            "the shared spec asks for standardise_targets=False; this runner "
            "always standardises, and so does QM9 (in Rust). Flipping the key "
            "would have changed one side only (RERUN_PLAN.md 2.13).")
    if y_scaler is None:
        raise ValueError(
            "train_neural_regression needs the target scaler fitted on the "
            "CLEAN training labels. Fitting it here would fit it on the noisy "
            "ones and make the learning problem change with the noise level.")
    y_train_norm = y_scaler.transform(
        np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_norm = y_scaler.transform(
        np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
    y_scale = float(y_scaler.scale_[0])  # std used to rescale predicted uncertainty

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train_norm).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val_norm).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    # arch: 'dnn' (DNN, NN-α: [128, 64]) or 'mlp' (MLP, NN-β: [32, 32]).
    # Model types prefixed with 'mlp-' use the MLP base; others use DNN.
    if model_type.startswith('mlp-'):
        arch = 'mlp'
        subtype = model_type[len('mlp-'):]   # 'deterministic' | 'full-bnn' | 'full-vbll'
        # Width 128 to match QM9, which passes hidden_size=128 explicitly
        # (qsar models.py:2531,:2613) and which paper.tex:216 states as [128,128].
        # The class default of 32 was being taken here, giving 16x fewer
        # hidden parameters under the same model name.
        model = MLPRegressor(
            X_train.shape[1],
            hidden_size=NEURAL_DEFAULTS['mlp']['hidden_size'],
            num_hidden_layers=NEURAL_DEFAULTS['mlp']['num_hidden_layers'],
            dropout_rate=NEURAL_DEFAULTS['mlp']['dropout_rate']).to(device)
    else:
        arch = 'dnn'
        subtype = model_type
        h1, h2 = NEURAL_DEFAULTS['dnn']['hidden_sizes']
        model = DeterministicRegressor(X_train.shape[1], h1, h2).to(device)

    if subtype == 'full-bnn':
        if not HAS_BAYESIAN_TORCH:
            raise RuntimeError(f"Full BNN ({arch}) requested but torchbnn not available")
        model = apply_bayesian_transformation(model)
    elif subtype == 'last-layer-bnn':
        if not HAS_BAYESIAN_TORCH:
            raise RuntimeError(f"Last-layer BNN ({arch}) requested but torchbnn not available")
        model = apply_bayesian_transformation_last_layer(model)
    elif subtype == 'full-vbll':
        model = apply_bayesian_transformation_full_variational(model).to(device)
    elif subtype == 'full-vbll-het':
        # The same model with a noise head, so its data-noise term varies per
        # molecule instead of being one number per fit (RERUN_PLAN.md 5.5f).
        model = apply_bayesian_transformation_full_variational(
            model, heteroscedastic=True).to(device)
    elif subtype == 'full-bnn-mve':
        # A BAYESIAN NETWORK WITH A VARIANCE OUTPUT HEAD -- Kendall & Gal eq. 6,
        # the case the uncertainty literature actually holds up, and the only
        # route on this side to an aleatoric term that varies PER MOLECULE on a
        # network. The head predicts the value and its own log variance; the
        # weights are sampled, which gives the epistemic term. Two mechanisms,
        # which is why the two halves can come apart at all.
        #
        # The head is widened BEFORE the Bayesian transformation, because the
        # transformation replaces every Linear layer and would otherwise be
        # applied to the one-output head and then thrown away.
        if not HAS_BAYESIAN_TORCH:
            raise RuntimeError(
                f"a variance-head BNN ({arch}) was requested but torchbnn is "
                f"not available")
        model = _widen_output_head(model, 2).to(device)
        model = apply_bayesian_transformation(model)

    is_bayesian = subtype != 'deterministic'
    if _t['optimizer'] != 'adam':
        raise ValueError(
            f"the shared spec asks for optimizer={_t['optimizer']!r}; only "
            f"'adam' is built here. A spec key nothing reads is a guard that "
            f"cannot fire (RERUN_PLAN.md 2.13).")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=_t['weight_decay'])
    # VBLL uses its own ELBO loss. The torchbnn networks now carry a KL term too:
    # until 2026-08-27 they were fitted with plain MSE, so nothing pulled the
    # variational posterior toward the prior, `prior_sigma` was only an
    # initialisation, and the posterior width received MSE gradients that drive
    # it toward zero. What was reported as their epistemic uncertainty was the
    # residual weight noise the fit happened to leave, and it was compared with
    # VBLL, which had carried a KL term all along.
    if subtype in ('full-vbll', 'full-vbll-het'):
        criterion = VBLLLoss(model, n_data=len(X_train))
    elif subtype == 'full-bnn-mve':
        # The shared variance-head loss, wrapped in the same KL term every other
        # Bayesian network here carries. QM9 builds exactly this pair:
        # get_loss_function('heteroscedastic') then bnn_elbo_criterion.
        criterion = bnn_elbo_criterion(HeteroscedasticLoss(), model,
                                       len(X_train))
    elif subtype in ('full-bnn', 'last-layer-bnn'):
        criterion = bnn_elbo_criterion(nn.MSELoss(), model, len(X_train))
    else:
        criterion = nn.MSELoss()
    # Batch size 32, from the shared spec. This pipeline used 64 while QM9 used
    # 32 -- not listed in the original hand audit at all.
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=_t['batch_size'],
                        shuffle=True)
    # THE STOPPING SIGNAL IS AVERAGED OVER THE SAME BATCHES AS QM9's.
    #
    # This side used to score early stopping with ONE forward pass over the
    # whole validation block. For a torchbnn or VBLL network every forward pass
    # draws fresh weights, so that one number is a single Monte Carlo sample
    # while QM9's is the mean of ceil(n_val/32) of them (models.py:2970-2993,
    # gated on the same `val_loss_reduction` key). Measured on an identical
    # Bayesian network at this side's validation sizes, the single-pass value
    # has 2.9x the epoch-to-epoch spread at 10 batches and 5.1x at QM9's
    # production 1,000 validation molecules, with the two means agreeing to
    # under 1% -- so it is pure noise on the signal, not a different quantity.
    # Against the same strict-improvement rule and the same patience of 10, the
    # noisier signal locks onto a lucky early minimum: in the probe the restored
    # epoch moved from 94 to 45 and the true loss there from 0.577 to 0.880.
    # That under-trained this side's Bayesian nets relative to QM9's.
    #
    # Mean over BATCHES, not over molecules, because that is the statistic QM9
    # compares against the same threshold. It is a slightly wrong weighting when
    # n_val is not a multiple of the batch size (at n_val = 1000 the last 8
    # molecules carry 1/32 of the weight instead of 8/1000); correcting that is
    # a change to both sides at once or to neither, so it is left alone here.
    val_loader = DataLoader(TensorDataset(X_v, y_v),
                            batch_size=_t['batch_size'], shuffle=False)
    if _t['val_loss_reduction'] != 'mean':
        raise ValueError(
            f"the shared spec asks for val_loss_reduction="
            f"{_t['val_loss_reduction']!r}; only 'mean' is built here, and QM9 "
            f"reads the same key. A spec key nothing reads is a guard that "
            f"cannot fire (RERUN_PLAN.md 2.13).")

    best_val, patience_ctr = float('inf'), 0
    patience = _t['patience']
    tolerance = _t['improvement_tolerance']
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        # WEIGHTED BY MOLECULE, not by batch. Settled by the author 2026-08-29,
        # on both halves of the study at once.
        #
        # The loss inside a batch is already a mean over that batch, so adding
        # the batches up and dividing by their COUNT gives every batch the same
        # weight however many molecules it holds. The last batch is short
        # whenever the validation count is not a multiple of the batch size, and
        # its molecules then count for more than the others. With 32 to a batch
        # and, say, 226 validation molecules there are 7 batches of 32 and one of
        # 2, and those 2 molecules carry an eighth of the stopping decision
        # instead of a hundredth. Multiplying each batch's mean by its size and
        # dividing by the molecule count is the ordinary mean over molecules.
        #
        # QM9 carries the identical correction in `models/models.py`. It is both
        # halves or neither: correcting one alone would put the two studies back
        # on different stopping criteria, which is what this whole exercise is
        # about.
        val_loss, n_val_molecules = 0.0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                val_loss += criterion(model(vx), vy).item() * len(vy)
                n_val_molecules += len(vy)
        vl = val_loss / n_val_molecules if n_val_molecules else float('inf')
        if np.isfinite(vl) and vl < best_val - tolerance:
            best_val = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if not _t['restore_best_weights']:
        raise ValueError(
            "the shared spec asks for restore_best_weights=False; this runner "
            "always restores. Flipping the key in model_defaults.py would have "
            "changed QM9 and left this side alone (RERUN_PLAN.md 2.13).")
    if best_state is None:
        # A fit that diverged on the first epoch left best_state None, and
        # `load_state_dict(None)` raised TypeError -- which the caller's blanket
        # `except Exception` printed as one log line and turned into a MISSING
        # CELL. Cells that go missing this way are not missing at random: it is
        # the widest representations that diverge, so any ANOVA or Wilcoxon over
        # the model x representation grid was computed on a biased subset
        # (RERUN_PLAN.md 2.13). Stop instead, and say what happened.
        raise RunIntegrityError(
            f"{model_type}: the validation loss was never finite and never "
            f"improved in {epochs} epochs, so there is no epoch to restore. The "
            f"fit diverged; it must not be recorded as an absent cell.")
    model.load_state_dict(best_state)
    model.eval()

    # Predictions are on the normalized scale — inverse-transform back to raw
    # label units so metrics line up with the tree-model R²/RMSE.
    def _to_raw(arr):
        return y_scaler.inverse_transform(
            np.asarray(arr, dtype=float).reshape(-1, 1)).ravel()

    if is_bayesian:
        # 100 passes to match QM9 (models.py:2077 and six other sites) and
        # paper.tex:216, which already claims 100.
        #
        # The observation noise each pass predicted is kept alongside the
        # prediction. This runner used to keep only the spread over the passes,
        # so the variational layer's learned noise was computed by its own loss
        # and never left the class -- the model-uncertainty half was reported as
        # if it were the whole thing (RERUN_PLAN.md 5.5a point 6).
        vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
        vbll_layer = vbll_layers[-1] if vbll_layers else None

        # A VARIANCE HEAD outputs TWO columns -- the value and its own log
        # variance -- so the raw output cannot be flattened. `reshape(-1)` on an
        # (n, 2) tensor interleaves the two into one array of length 2n, which
        # would be read as predictions for molecules that do not exist. The
        # split uses QM9's own `split_predictive_head`, so the transform from the
        # second output to a variance is the one the loss was FITTED with.
        is_mve = subtype == 'full-bnn-mve'

        passes, noise_passes = [], []
        for _ in range(_t['mc_passes']):
            out = model(X_te)
            if is_mve:
                # DETACHED FIRST, and to numpy. `split_predictive_head` clips the
                # log variance with np.clip, which cannot touch a tensor that
                # still carries a gradient -- it raises rather than silently
                # doing the wrong thing, which is how this was caught.
                _o = out.detach().cpu().numpy().astype(float)
                _m, _v = split_predictive_head(_o, 'heteroscedastic')
                if _v is None:
                    raise RunIntegrityError(
                        "the variance head returned no variance, so the model "
                        "has a two-column head fitted by a two-column loss and "
                        "the column that reaches the file would be empty.")
                passes.append(np.asarray(_m, dtype=float).reshape(-1))
                noise_passes.append(np.asarray(_v, dtype=float).reshape(-1))
                continue
            passes.append(np.asarray(out.detach().cpu().numpy(),
                                     dtype=float).reshape(-1))
            if vbll_layer is not None and vbll_layer.heteroscedastic:
                cached = getattr(vbll_layer, '_last_noise_var', None)
                if cached is None:
                    raise RunIntegrityError(
                        "the variational layer is heteroscedastic but recorded "
                        "no per-molecule noise on this pass. Falling back to its "
                        "single noise would write a constant column under a name "
                        "that claims it varies per molecule.")
                noise_passes.append(
                    cached.detach().cpu().numpy().reshape(-1).astype(float))
        preds_norm = np.stack(passes, axis=0)

        if noise_passes:
            var_passes = np.stack(noise_passes, axis=0)
        elif vbll_layer is not None:
            var_passes = np.full_like(preds_norm,
                                      float(vbll_layer.noise_var.item()))
        else:
            # A plain Bayesian network predicts a mean and nothing else, so it
            # has NO observation-noise term -- Kendall & Gal's epistemic-only
            # model, not a zero.
            var_passes = None

        alea_var, epis_var, total_var = decompose_sampling(preds_norm, var_passes)
        mean_raw = _to_raw(preds_norm.mean(0))
        # Variances scale by the SQUARE of the label spread; a standard
        # deviation scales by the spread itself. The two are different
        # operations and the difference is invisible in the output file.
        alea_raw = None if alea_var is None else alea_var * y_scale ** 2
        epis_raw = epis_var * y_scale ** 2
        # BOTH halves, added as VARIANCES and converted once.
        #
        # This used to be `preds_norm.std(0)`, the spread over the stochastic
        # passes, which is the model half alone -- so a variational layer's
        # learned observation noise was computed on the line above and thrown
        # away. QM9 reports the total (`models/models.py`, all three neural
        # trainers: `sqrt(total_var)`), so the column called `uncertainty` meant
        # two different quantities on the two halves of the study. Measured on
        # one fitted network: the total is 1.49x the spread over passes.
        #
        # Coverage asks how often the TRUE LABEL lands inside the interval, and a
        # label is an observation, so the interval has to carry the observation
        # noise or it under-covers by construction. That is the most likely
        # explanation for the recorded VBLL coverage of 0.27-0.45 against a 0.68
        # target.
        #
        # `decompose_sampling` returns total = aleatoric + epistemic as VARIANCES
        # and returns the epistemic term alone where the network has no
        # observation-noise output -- so BNN-Full and MLP-BNN-Full are unchanged
        # to the bit, and only the four variational models move.
        std_raw = np.sqrt(np.asarray(total_var, dtype=float)) * y_scale
        return mean_raw, std_raw, alea_raw, epis_raw
    with torch.no_grad():
        return _to_raw(model(X_te).cpu().numpy()), None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def _oof_predict(fit_predict, X, y_noisy, n_folds, groups=None, seed=42):
    """Out-of-fold predictions and uncertainties over the TRAINING molecules.

    Splits the training rows into `n_folds` parts, fits on the rest and scores
    the held-out part, rotating. Every training molecule therefore receives a
    prediction from a model that never saw ITS label.

    This is the only construction under which "does predicted uncertainty
    track which labels were corrupted?" is answerable: scoring a molecule with
    a model that fitted its corrupted label measures memorisation, not
    uncertainty (a GP has zero posterior variance at its own training inputs;
    a forest has fitted those exact rows).

    The noise is injected ONCE over the whole training set before this is
    called, so a molecule carries the same corruption in whichever fold it
    lands.

    `fit_predict(X_fit, y_fit, X_score)` returns either `(mean, std_or_None)` or
    `(mean, std_or_None, aleatoric_var_or_None, epistemic_var_or_None)`; the
    four-value form carries the split out of fold. Returns (oof_mean, oof_unc,
    oof_aleatoric_var, oof_epistemic_var, oof_fold, n_folds_ok).

    `groups` are the Murcko-scaffold ids of the training rows. When supplied the
    inner split is scaffold-grouped, matching the outer CV: without it the fit
    set contains close analogues of every held-out molecule, so out-of-fold
    uncertainties come from an interpolation regime while the test set is an
    extrapolation regime, and the two are not on the same scale.

    `fit_predict(X_fit, y_fit, X_score) -> (mean, std_or_None)`.

    Returns (oof_mean, oof_unc, n_folds_ok) — the caller must check n_folds_ok,
    because a silently truncated out-of-fold pass looks exactly like a complete
    one in the output file.
    """
    n = len(y_noisy)
    oof_mean = np.full(n, np.nan)
    oof_unc = np.full(n, np.nan)

    if groups is not None and len(np.unique(groups)) >= n_folds:
        splitter = GroupKFold(n_splits=n_folds)
        folds = [(tr, te) for tr, te in splitter.split(X, y_noisy, groups)]
    else:
        if groups is not None:
            print(f"      [oof] only {len(np.unique(groups))} scaffolds for {n_folds} folds "
                  f"— falling back to a random split", flush=True)
        order = np.random.RandomState(seed).permutation(n)
        parts = np.array_split(order, n_folds)
        folds = [(np.setdiff1d(order, held), held) for held in parts]

    # The split, carried out of fold. This is where it matters: a TEST label is
    # never corrupted, so a split written only for test rows cannot answer
    # whether the uncertainty finds the corrupted labels (RERUN_PLAN.md 3.1c).
    # `oof_fold` says which inner fit scored each molecule, because a component
    # that is one number per FIT -- a homoscedastic likelihood noise, a
    # variational layer's single observation noise -- takes a different value in
    # each fold, and that is fold variation, not per-molecule variation.
    oof_alea = np.full(n, np.nan)
    oof_epis = np.full(n, np.nan)
    oof_fold = np.full(n, -1, dtype=int)
    got_alea = got_epis = False

    n_ok = 0
    for fold_i, (keep, held) in enumerate(folds):
        try:
            # The fit rows' scaffold ids travel with them: NGBoost carves its
            # own early-stopping split inside fit() and can only carve it
            # grouped if it knows which scaffold each fit row belongs to. An
            # inner fold that lost them would silently train to the cap while
            # the outer fit stopped early, so the out-of-fold rows and the test
            # rows would come from differently trained models.
            out = fit_predict(X[keep], y_noisy[keep], X[held],
                              None if groups is None else np.asarray(groups)[keep])
        except Exception as e:
            print(f"      [oof] fold failed: {type(e).__name__}: {e}", flush=True)
            continue
        if len(out) == 4:
            m, u, a, ep = out
        else:
            m, u = out
            a = ep = None
        oof_mean[held] = np.asarray(m, dtype=float).ravel()
        oof_fold[held] = fold_i
        if u is not None:
            oof_unc[held] = np.asarray(u, dtype=float).ravel()
        if a is not None:
            oof_alea[held] = np.asarray(a, dtype=float).ravel()
            got_alea = True
        if ep is not None:
            oof_epis[held] = np.asarray(ep, dtype=float).ravel()
            got_epis = True
        n_ok += 1
    if n_ok < len(folds):
        print(f"      [oof] WARNING {n_ok}/{len(folds)} inner folds succeeded", flush=True)
    return (oof_mean, oof_unc,
            oof_alea if got_alea else None,
            oof_epis if got_epis else None,
            oof_fold, n_ok)


def _pattern_from_oof(pattern_injector, oof_mean, y_train, noise_groups,
                      reference_dose=1.0):
    """The level-free noise shape recomputed from the model's OWN prediction.

    The ceiling for `noise_pattern`: an effect against the real shape that is no
    larger than the effect against this one is the model tracking its own
    prediction, not the noise.

    Non-finite out-of-fold predictions (a failed inner fold) are replaced by the
    mean of the finite ones, so one failed fold does not blank the whole column;
    the count of successful folds travels separately in `oof_folds_ok`.
    """
    om = np.asarray(oof_mean, dtype=float)
    finite = np.isfinite(om)
    fill = float(om[finite].mean()) if finite.any() else 0.0
    om = np.where(finite, om, fill)
    # The same reference dose the real pattern uses -- one label spread -- so the
    # ceiling and the thing it is a ceiling on are on one scale.
    return pattern_injector.noise_scale(om, reference_dose, reference=y_train,
                                        groups=noise_groups,
                                        reference_groups=noise_groups)


def _tree_mean_std(mdl, X_score):
    """Per-molecule mean and uncertainty from a fitted tree/GP/NGBoost model.

    Single definition, used for both the held-out test set and the
    out-of-fold training predictions, so the two are never computed
    differently.
    """
    if 'Quantile' in str(type(mdl)):
        q16, q50, q84 = mdl.predict(X_score, quantiles=[0.16, 0.5, 0.84]).T
        return q50, (q84 - q16) / 2
    if 'GaussianProcess' in str(type(mdl)):
        pm, ps = mdl.predict(X_score, return_std=True)
        return pm, ps
    if 'NGBRegressor' in str(type(mdl)):
        dist = mdl.pred_dist(X_score)
        return dist.loc, dist.scale
    return mdl.predict(X_score), None


def _tree_split(mdl, X_fit, y_fit, X_score):
    """The aleatoric/epistemic split for a fitted tree, GP or NGBoost model.

    Single definition, used for the held-out test rows and for the out-of-fold
    training rows, so the two are never computed differently -- the same rule as
    `_tree_mean_std` beside it. Everything returned is a VARIANCE.

    Where each model's split comes from:

      forests     the law of total variance over the per-tree leaf distributions,
                  walked from trees that are ALREADY FITTED, so it costs no
                  refit. Both halves vary per molecule, and only because
                  `min_samples_leaf` is 5: at a leaf of one there is no
                  within-leaf spread and the data half is identically zero,
                  which is arithmetic and not a finding (RERUN_PLAN.md 5.5c).
      GP          the latent variance is the model half; the data half is the
                  likelihood noise, ONE number for the whole fit. The
                  heteroscedastic variant predicts it per molecule instead.
      NGBoost     one fit predicts a distribution per molecule, so the data half
                  varies and there is no second fit to disagree with -- the
                  model half is ABSENT, not zero.
      anything else   no split at all.

    Returns (aleatoric_var, epistemic_var); either may be None.
    """
    type_name = str(type(mdl))
    if 'Forest' in type_name and hasattr(mdl, 'estimators_'):
        _mean, alea, epis = decompose_forest(mdl, X_fit, y_fit, X_score)
        return alea, epis
    if 'GaussianProcess' in type_name:
        # predict() records the split as a side effect, because the dispatch
        # around it requires predict to keep its (mean, std) signature.
        alea = getattr(mdl, 'aleatoric_var_', None)
        epis = getattr(mdl, 'epistemic_var_', None)
        return alea, epis
    if 'NGBRegressor' in type_name:
        # ONE FIT, and no epistemic term. `ngboost_ensemble_seeds` is 1 in the
        # shared spec (1.7.0) because the seed ensemble does not produce a usable
        # one: measured on 3,000 real QM9 molecules at four settings, the spread
        # between members is 150 to 3,581 times smaller than the aleatoric term
        # and RISES with the noise level. NGBoost's seed reaches only
        # `minibatch_frac` and `col_sample`, both pinned at 1.0, so the members
        # are the same fit (RERUN_PLAN.md 2.30).
        #
        # `_ngboost_ensemble_members` is kept below and reads that spec key, so
        # raising it re-enables the ensemble on this side in one place -- but the
        # QM9 support table must be changed to match, or the guard refuses the
        # row on one pipeline and not the other.
        members = _ngboost_ensemble_members(mdl, X_fit, y_fit)
        if len(members) < 2:
            scale = np.asarray(mdl.pred_dist(X_score).scale, dtype=float)
            alea, epis, _total = decompose_single_distribution(scale ** 2)
            return alea, epis
        dists = [m.pred_dist(X_score) for m in members]
        alea, epis, _total = decompose_seed_ensemble(
            np.stack([np.asarray(d.loc, dtype=float) for d in dists], axis=0),
            np.stack([np.asarray(d.scale, dtype=float) ** 2 for d in dists], axis=0))
        return alea, epis
    return None, None


def _ngboost_ensemble_members(fitted, X_fit, y_fit):
    """`fitted` plus the extra seeds, all fitted on the same rows.

    The seed count is the SHARED spec's (`ngboost_ensemble_seeds`, 3), not a
    literal here: QM9 reads the same key, and a second number on this side is how
    the two pipelines report different quantities under one column name.

    Member 0 is the model that was already fitted and is returned unchanged, so
    no accuracy number moves and the extra cost is (seeds - 1) fits, not seeds.
    A member that fails to fit is DROPPED with a message rather than silently
    reducing the spread -- three seeds that became two is a different statistic,
    and a quieter one.
    """
    n_seeds = int(SKLEARN_DEFAULTS.get('ngboost_ensemble_seeds', 1) or 1)
    members = [fitted]
    if n_seeds < 2:
        return members
    kwargs, stopping = _ngboost_kwargs()
    for k in range(1, n_seeds):
        try:
            m = EarlyStoppingNGBRegressor(random_state=42 + 1000 * k,
                                          verbose=False, **kwargs, **stopping)
            m.fit(X_fit, y_fit)
            members.append(m)
        except Exception as exc:
            print(f"      [ngboost] ensemble member {k} failed to fit "
                  f"({type(exc).__name__}: {exc}) -- dropped", flush=True)
    return members


def _level_key(level):
    """The canonical text form of a noise level, used to seed from its VALUE.

    `%.12g` round-trips every level any grid in this file uses, and gives the
    same text for 0.6, 0.60 and 6e-1, so a level named three ways is one level.
    """
    return f"{float(level):.12g}"


def _injector_for(condition, level, stream='training'):
    """A fresh injector for one condition at one noise LEVEL.

    Three properties this exists to guarantee, all learned the hard way.

    REPRODUCIBILITY. `NoiseInjectorRegression` advances its generator on every
    draw, so re-using one injector across the level grid makes the realisation at
    a level depend on how many levels ran before it.

    GRID-INDEPENDENCE. The seed is derived from the level's VALUE, never from its
    POSITION in the grid. It used to be the position: `--sigmas 0.0 0.6` gave 0.6
    index 1 while `--sigmas 0.0 0.3 0.6` gave it index 2, so the two runs
    corrupted the same molecules differently at the same nominal level, and any
    gap-filling resubmission drew noise unrelated to the original run's. Merging
    those files put two different corruptions of the same molecule into one
    column with nothing to tell them apart.

    SEPARATE STREAMS. The validation split draws from its own stream, so
    corrupting validation cannot change the training corruption.

    NOTE: this changes the realised noise relative to any run made before it,
    because the seed formula changed. Results from the two are not mergeable.
    """
    base = 42 if stream == 'training' else 1337
    # zlib.crc32, NOT hash(): Python randomises string hashing per process, so
    # hash() here would have made the noise different on every run while the
    # docstring above promised the opposite.
    digest = zlib.crc32(f"{stream}|{condition}|{_level_key(level)}".encode('utf-8'))
    seed = (base * 1000003 + digest) & 0x7FFFFFFF
    # WHO gets hit is seeded WITHOUT the level. The shape draw varies per level;
    # the affected set must not, or `noise_pattern` -- the level-free column the
    # whole zero-level subtraction rests on -- describes a different set of
    # molecules than the injection at that level. Measured on a 120-molecule
    # fixture before this: for outlier_p10 the pattern and the realised scale had
    # Spearman -0.10 and their top-15% sets overlapped 3 of 18; for
    # grouped_wider, 0.375 at one level and -0.250 at another
    # (RERUN_PLAN.md 2.13). The Rust injector has always been level-independent
    # here.
    selection = (base * 1000003
                 + zlib.crc32(f"{stream}|{condition}|selection".encode('utf-8'))) \
        & 0x7FFFFFFF
    return NoiseInjectorRegression.from_condition(condition, random_state=seed,
                                                  selection_state=selection)


def run_tree_experiment(X_train, y_train, X_test, y_test, model_fn, condition, sigma_levels,
                        oof_folds=0, train_groups=None, noise_groups=None,
                        test_noise_groups=None, provenance=None, rep_name=None):
    """Noise robustness for tree-based regression model.

    Returns (predictions, uncertainties, extras). `extras` carries everything
    needed to ask whether uncertainty tracks the injected noise:
      train_noise_scale[sigma]  per-molecule noise SD actually applied in training
      train_epsilon[sigma]      the noise actually added to each training label
      test_noise_scale[sigma]   the noise scale a TEST molecule's region would
                                receive, using the TRAINING distribution's
                                cut-points -- the model never saw this, so it is
                                the target for "did the model learn where the
                                data is unreliable?"
      oof_mean/oof_unc[sigma]   out-of-fold training predictions (if oof_folds>0)
    """
    # Standardise only the representations the shared spec names. This used to
    # scale EVERYTHING, including binary fingerprints -- which QM9 does not do,
    # and which costs an RBF-kernel support vector machine about 0.6 R-squared
    # because the rare bits get huge magnitudes and dominate every distance.
    if rep_name is None:
        raise ValueError(
            "run_tree_experiment needs rep_name so the standardisation "
            "decision can be reported against a representation. The decision "
            "itself is made from the features, not the name.")
    if should_standardise(X_train, rep_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    predictions, uncertainties = {}, {}
    extras = {'train_noise_scale': {}, 'train_epsilon': {}, 'test_noise_scale': {},
              'test_noise_pattern': {}, 'train_noise_pattern': {},
              'test_noise_pattern_pred': {}, 'train_noise_pattern_pred': {},
              'oof_mean': {}, 'oof_unc': {}, 'oof_folds_ok': {},
              # WHAT THE UNCERTAINTY COLUMN IS DIVIDED BY, so the two halves of
              # the study can go in one table. This runner converts predictions
              # and uncertainties back to RAW label units (log units here) before
              # writing them; QM9 leaves them on the standardised scale it fitted
              # on, and its results rows carry standardisation_mean and
              # standardisation_sd for exactly this reason. So the column called
              # `uncertainty` was in log units here and in multiples of the label
              # spread there, and a mean-uncertainty table across the four
              # datasets added the two together. The settled scale is fractions
              # of the CLEAN TRAINING LABEL SPREAD (author, 2026-08-27), which is
              # what this number is: the scaler is fitted on the clean training
              # labels, once, and used at every level.
              'label_scale': None,
              # The two halves of the uncertainty, as VARIANCES, for the test
              # rows and for the out-of-fold training rows. This pipeline wrote
              # ONE uncertainty column and performed no split at all -- and the
              # paper's uncertainty section is built on these three datasets
              # (RERUN_PLAN.md 5.5a point 6).
              'test_aleatoric_var': {}, 'test_epistemic_var': {},
              'oof_aleatoric_var': {}, 'oof_epistemic_var': {},
              'oof_fold': {},
              # PER LEVEL, not one scalar. It used to be a single string
              # overwritten inside the level loop and then broadcast onto every
              # level's row -- so if botorch succeeded at low noise and refused
              # at high noise, the file said every level used the same
              # optimiser, and a row reading 'botorch' might have come from the
              # Adam fallback, which is a different fit (RERUN_PLAN.md 2.13).
              'gp_fit_method': {}, 'gp_collapsed': {}}

    # The SHAPE of the noise -- which molecules a condition hits hardest -- taken
    # at a fixed reference level, so it is the same column at every level
    # INCLUDING zero. This is what makes the confound testable: for the
    # label-keyed conditions the noise scale is a deterministic function of the
    # label, and a model's uncertainty may already track the label for reasons
    # that have nothing to do with noise. Correlating uncertainty against this
    # pattern on a model trained at level 0 measures exactly that pre-existing
    # association, and the honest effect is (correlation at the level) minus
    # (correlation at zero).
    #
    # The pattern is reproducible because the scale map is drawn from a
    # separately seeded generator, so it does not depend on how many shape draws
    # have happened -- see `_selection_rng` in noiseInject.
    # The LEVEL is a fraction of this fold's CLEAN TRAINING label spread, and the
    # dose handed to the injector is level x spread. That is what QM9 already
    # does (`--dose-units spread`), so the same numeric level means the same
    # relative corruption on all four datasets and auc_norm is comparable
    # without rescaling anything (RERUN_PLAN.md 2.12).
    #
    # It used to be raw log units, anchored per dataset to published assay error
    # (NOISE_DESIGN.md 6.4). Those grids were close in log units -- tops 1.0, 0.7
    # and 1.1 -- but the label spreads under them are 1.19, 0.44 and 0.91, so the
    # top of the LogD grid was 0.84 of its spread and the top of the Caco-2 grid
    # was 1.57 of its. A Caco-2 model was corrupted about twice as hard as a LogD
    # model before its curve was summarised, and then the two went on one bar.
    #
    # The cost, and it belongs in the caption: a level is no longer a stated
    # multiple of assay error. At the top of the shared grid LogD carries 11.9x
    # its within-lab error of 0.15, Caco-2 1.9x its 0.35 and hERG 2.5x its 0.54.
    # CENSORING IS EXEMPT. Its level is the FRACTION of labels clipped, which is
    # already dimensionless and already shared across datasets. Multiplying it by
    # a label spread gives a fraction outside [0, 1] and the quantile raises.
    _censoring = CONDITIONS[condition]['strategy'] == 'censoring'
    if _censoring:
        dose_reference_sd = 1.0
    else:
        dose_reference_sd = float(np.std(np.asarray(y_train, dtype=float)))
        if not np.isfinite(dose_reference_sd) or dose_reference_sd <= 0:
            raise RunIntegrityError(
                f"the clean training label spread is {dose_reference_sd}, so a "
                f"level expressed as a fraction of it has no meaning.")

    def _dose(level):
        return float(level) * dose_reference_sd

    pattern_injector = _injector_for(condition, 0.0)
    tr_pattern = pattern_injector.noise_scale(y_train, dose_reference_sd,
                                              reference=y_train,
                                              groups=noise_groups,
                                              reference_groups=noise_groups)
    # reference_groups=noise_groups, NOT test_noise_groups: the affected groups
    # must be the ones the TRAINING labels were exposed to, then looked up by
    # group id for the test molecules. Re-running the selection over the test
    # set's own groups picks a different set -- measured on a 40-group split,
    # two of eight corrupted groups went unmarked -- and question B would then
    # be scored against an injection that never happened.
    te_pattern = pattern_injector.noise_scale(y_test, dose_reference_sd,
                                              reference=y_train,
                                              groups=test_noise_groups,
                                              reference_groups=noise_groups)

    # The TARGET is standardised here too, on the CLEAN training labels.
    #
    # This path used to fit on the labels in their native units -- log D, log
    # efflux ratio, pKi -- while QM9 fits every model on a z-scored label
    # (standardised in Rust against the clean training spread). The shared spec
    # is written for a unit-variance target: SVR's epsilon 0.1 is a tenth of a
    # standard deviation on QM9 and a large dead zone on a raw label whose
    # spread is a few tenths of a log unit; C = 1.0 is a different amount of
    # regularisation; the GP's likelihood noise 1e-3 and outputscale 1.0 and
    # NGBoost's learning rate over 500 stages are all stated relative to unit
    # variance. Baseline R2 and the shape of the degradation curve differed
    # between the pipelines for that reason alone (RERUN_PLAN.md 2.13).
    #
    # Trees are scale-invariant in y, so this changes nothing for RF, QRF,
    # XGBoost or LightGBM; it changes SVM, the GP and NGBoost.
    y_scaler = StandardScaler().fit(
        np.asarray(y_train, dtype=float).reshape(-1, 1))
    _y_scale = float(y_scaler.scale_[0])
    extras['label_scale'] = _y_scale

    def _to_model_scale(y):
        return y_scaler.transform(
            np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    def _to_raw(y):
        return y_scaler.inverse_transform(
            np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    for sigma in sigma_levels:
        # A FRESH injector per level, seeded from the level's VALUE. The
        # generator advances with every draw, so re-using one would make the
        # realisation at a level depend on how many levels ran before it; and
        # seeding from the level's POSITION in the grid, which is what this used
        # to do, made a subset run corrupt differently from the full run.
        injector = _injector_for(condition, sigma)
        result = injector.inject_verbose(y_train, _dose(sigma), groups=noise_groups)
        y_noisy, tr_scale, eps = result.y_noisy, result.noise_scale, result.epsilon
        # noise_scale consumes no randomness and re-seeds its own selection, so
        # calling it here cannot perturb the draw above.
        te_scale = injector.noise_scale(y_test, _dose(sigma), reference=y_train,
                                        groups=test_noise_groups,
                                        reference_groups=noise_groups)
        if provenance is not None:
            provenance[sigma] = result.as_row()

        mdl = model_fn()
        # NGBoost carves its own early-stopping split and needs the fold's
        # scaffold ids to carve it grouped. Every other model ignores this.
        if hasattr(mdl, 'set_fit_groups'):
            mdl.set_fit_groups(train_groups)
        mdl.fit(X_train, _to_model_scale(y_noisy))
        # Which optimiser actually fitted the GP -- '' for every other model.
        # The Adam fallback is a different fit, so it has to reach the results.
        if getattr(mdl, 'fit_method_', None):
            extras['gp_fit_method'][sigma] = mdl.fit_method_
        mean, std = _tree_mean_std(mdl, X_test)
        _alea, _epis = _tree_split(mdl, X_train, _to_model_scale(y_noisy), X_test)
        # Back to raw label units, so R2 and RMSE stay comparable with the
        # neural path and with the labels themselves. A standard deviation has
        # no offset, so it scales by the spread alone.
        predictions[sigma] = _to_raw(mean)
        uncertainties[sigma] = (None if std is None
                                else np.asarray(std, dtype=float) * _y_scale)
        # A fit that gave up and predicted one constant still produces a number,
        # and that number reads as a bad representation rather than a failed fit
        # -- the situation QM9 added gp_collapsed to prevent. Recorded here too,
        # per level.
        _collapsed, _spread, _threshold = gp_fit_collapsed(
            predictions[sigma], y_train)
        extras['gp_collapsed'][sigma] = int(_threshold > 0 and _collapsed)

        extras['train_noise_scale'][sigma] = tr_scale
        extras['train_epsilon'][sigma] = eps
        extras['test_noise_scale'][sigma] = te_scale
        extras['train_noise_pattern'][sigma] = tr_pattern
        extras['test_noise_pattern'][sigma] = te_pattern
        # Variances scale by the SQUARE of the spread. Multiplying by the spread
        # itself is right for a standard deviation and wrong for a variance, and
        # the difference is invisible in the output file (RERUN_PLAN.md 5.5).
        extras['test_aleatoric_var'][sigma] = (
            None if _alea is None else np.asarray(_alea, float) * _y_scale ** 2)
        extras['test_epistemic_var'][sigma] = (
            None if _epis is None else np.asarray(_epis, float) * _y_scale ** 2)
        # The sham ceiling: the same pattern computed from what the model
        # PREDICTED rather than the true label. If uncertainty tracks this just
        # as strongly, the model is following its own prediction, not the noise.
        # Computed here because quantile and outlier need the training reference,
        # which is not recoverable from the output file once folds 1-4 carry no
        # training rows.
        extras['test_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
            np.asarray(predictions[sigma], dtype=float), dose_reference_sd,
            reference=y_train,
            groups=test_noise_groups, reference_groups=noise_groups)
        if oof_folds and oof_folds > 1:
            def _fp(Xf, yf, Xs, gf=None):
                m = model_fn()
                _yf = _to_model_scale(yf)
                if hasattr(m, 'set_fit_groups'):
                    m.set_fit_groups(gf)
                m.fit(Xf, _yf)
                _m, _s = _tree_mean_std(m, Xs)
                _a, _e = _tree_split(m, Xf, _yf, Xs)
                return (_to_raw(_m),
                        None if _s is None
                        else np.asarray(_s, dtype=float) * _y_scale,
                        None if _a is None
                        else np.asarray(_a, float) * _y_scale ** 2,
                        None if _e is None
                        else np.asarray(_e, float) * _y_scale ** 2)
            om, ou, oa, oe, ofold, n_ok = _oof_predict(
                _fp, X_train, y_noisy, oof_folds, groups=train_groups)
            extras['oof_mean'][sigma] = om
            extras['oof_unc'][sigma] = ou
            extras['oof_aleatoric_var'][sigma] = oa
            extras['oof_epistemic_var'][sigma] = oe
            extras['oof_fold'][sigma] = ofold
            extras['oof_folds_ok'][sigma] = n_ok
            # The sham ceiling for the TRAINING rows, computed HERE and not
            # a dozen lines above, where it used to sit: `extras['oof_mean']`
            # is written at the bottom of this block, so a test for the level
            # being in it, placed above the block, was false on every pass of
            # every loop and the column came out empty on every training row
            # this pipeline has ever produced.
            extras['train_noise_pattern_pred'][sigma] = _pattern_from_oof(
                pattern_injector, om, y_train, noise_groups,
                reference_dose=dose_reference_sd)

    return predictions, uncertainties, extras


def run_neural_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                          model_type, condition, sigma_levels, oof_folds=0,
                          train_groups=None, noise_validation=True,
                          noise_groups=None, test_noise_groups=None,
                          val_noise_groups=None, provenance=None, rep_name=None):
    """Noise robustness for neural regression model. See run_tree_experiment
    for the contents of `extras`."""
    # Same policy as the tree path and as QM9 -- see the note there.
    if rep_name is None:
        raise ValueError(
            "run_neural_experiment needs rep_name so the standardisation "
            "decision can be reported against a representation. The decision "
            "itself is made from the features, not the name.")
    if should_standardise(X_train, rep_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    predictions, uncertainties = {}, {}
    extras = {'train_noise_scale': {}, 'train_epsilon': {}, 'test_noise_scale': {},
              'test_noise_pattern': {}, 'train_noise_pattern': {},
              'test_noise_pattern_pred': {}, 'train_noise_pattern_pred': {},
              'oof_mean': {}, 'oof_unc': {}, 'oof_folds_ok': {},
              # See the tree path for what this is and why the study needs it.
              'label_scale': None,
              # The two halves of the uncertainty, as variances. See the tree
              # path for why they are here.
              'test_aleatoric_var': {}, 'test_epistemic_var': {},
              'oof_aleatoric_var': {}, 'oof_epistemic_var': {},
              'oof_fold': {},
              # No GP on this path; kept as a dict so the writer sees the same
              # shape from both runners.
              'gp_fit_method': {}, 'gp_collapsed': {}}

    # See run_tree_experiment for why the reference pattern is recorded, and why
    # a fresh injector is built per level.
    # The LEVEL is a fraction of this fold's CLEAN TRAINING label spread, and the
    # dose handed to the injector is level x spread. That is what QM9 already
    # does (`--dose-units spread`), so the same numeric level means the same
    # relative corruption on all four datasets and auc_norm is comparable
    # without rescaling anything (RERUN_PLAN.md 2.12).
    #
    # It used to be raw log units, anchored per dataset to published assay error
    # (NOISE_DESIGN.md 6.4). Those grids were close in log units -- tops 1.0, 0.7
    # and 1.1 -- but the label spreads under them are 1.19, 0.44 and 0.91, so the
    # top of the LogD grid was 0.84 of its spread and the top of the Caco-2 grid
    # was 1.57 of its. A Caco-2 model was corrupted about twice as hard as a LogD
    # model before its curve was summarised, and then the two went on one bar.
    #
    # The cost, and it belongs in the caption: a level is no longer a stated
    # multiple of assay error. At the top of the shared grid LogD carries 11.9x
    # its within-lab error of 0.15, Caco-2 1.9x its 0.35 and hERG 2.5x its 0.54.
    # CENSORING IS EXEMPT. Its level is the FRACTION of labels clipped, which is
    # already dimensionless and already shared across datasets. Multiplying it by
    # a label spread gives a fraction outside [0, 1] and the quantile raises.
    _censoring = CONDITIONS[condition]['strategy'] == 'censoring'
    if _censoring:
        dose_reference_sd = 1.0
    else:
        dose_reference_sd = float(np.std(np.asarray(y_train, dtype=float)))
        if not np.isfinite(dose_reference_sd) or dose_reference_sd <= 0:
            raise RunIntegrityError(
                f"the clean training label spread is {dose_reference_sd}, so a "
                f"level expressed as a fraction of it has no meaning.")

    def _dose(level):
        return float(level) * dose_reference_sd

    pattern_injector = _injector_for(condition, 0.0)
    tr_pattern = pattern_injector.noise_scale(y_train, dose_reference_sd,
                                              reference=y_train,
                                              groups=noise_groups,
                                              reference_groups=noise_groups)
    # reference_groups=noise_groups, NOT test_noise_groups: the affected groups
    # must be the ones the TRAINING labels were exposed to, then looked up by
    # group id for the test molecules. Re-running the selection over the test
    # set's own groups picks a different set -- measured on a 40-group split,
    # two of eight corrupted groups went unmarked -- and question B would then
    # be scored against an injection that never happened.
    te_pattern = pattern_injector.noise_scale(y_test, dose_reference_sd,
                                              reference=y_train,
                                              groups=test_noise_groups,
                                              reference_groups=noise_groups)

    # ONE target scaler, fitted on the CLEAN training labels, used at every
    # level. Fitting it per level on the noisy labels made the divisor grow with
    # the dose, so the standardised target shrank and the same nominal level
    # posed a different optimisation problem (RERUN_PLAN.md 2.13).
    y_scaler = StandardScaler().fit(
        np.asarray(y_train, dtype=float).reshape(-1, 1))
    extras['label_scale'] = float(y_scaler.scale_[0])

    for sigma in sigma_levels:
        injector = _injector_for(condition, sigma)
        result = injector.inject_verbose(y_train, _dose(sigma), groups=noise_groups)
        y_noisy, tr_scale, eps = result.y_noisy, result.noise_scale, result.epsilon
        te_scale = injector.noise_scale(y_test, _dose(sigma), reference=y_train,
                                        groups=test_noise_groups,
                                        reference_groups=noise_groups)
        if provenance is not None:
            provenance[sigma] = result.as_row()

        # FIX: the validation labels used for early stopping were left CLEAN,
        # so the four Bayesian nets were explicitly selected NOT to fit the
        # injected noise -- suppressing the very quantity this run measures, and
        # only for the neural half of the roster. They now carry the same
        # condition at the same level, drawn from an INDEPENDENT generator so
        # the training draw is untouched.
        if noise_validation and sigma > 0.0:
            # reference=y_train matters for censoring: `inject` would otherwise
            # take the cut-point from y_val itself, so validation would carry a
            # different noise geometry from training and test.
            #
            # `reference_groups` is deliberately NOT passed, and that is a
            # correction. It used to be, which meant grouped_wider chose its
            # scaffold families on the TRAINING assignment and then looked them
            # up among the validation molecules. scaffold_validation_carve
            # guarantees the two share no family -- it raises if they ever do --
            # so the lookup matched nothing, every validation molecule took the
            # ordinary width, and the validation labels carried plain noise while
            # the training labels carried the grouped pattern. Under one
            # condition name, silently, on all three datasets, for the four
            # neural families that stop training by watching these labels.
            #
            # Omitting it draws a fresh selection over validation's OWN families
            # at the same molecule fraction, using this stream's independent
            # selection seed, so validation carries the same KIND of noise as
            # training without copying the training draw. Nothing else changes:
            # `reference` still fixes the censoring cut-point, and the other five
            # conditions do not consult the group assignment for their selection.
            # noiseInject now refuses the old call rather than injecting plain
            # noise under a grouped name.
            y_val_used = _injector_for(condition, sigma,
                                       stream='validation').inject_verbose(
                y_val, _dose(sigma), reference=y_train,
                groups=val_noise_groups).y_noisy
        else:
            y_val_used = y_val

        preds, uncs, alea, epis = train_neural_regression(
            X_train, y_noisy, X_val, y_val_used, X_test,
            model_type=model_type, epochs=NEURAL_DEFAULTS['training']['epochs'],
            y_scaler=y_scaler)
        predictions[sigma] = preds
        uncertainties[sigma] = uncs
        extras['test_aleatoric_var'][sigma] = alea
        extras['test_epistemic_var'][sigma] = epis

        extras['train_noise_scale'][sigma] = tr_scale
        extras['train_epsilon'][sigma] = eps
        extras['test_noise_scale'][sigma] = te_scale
        extras['train_noise_pattern'][sigma] = tr_pattern
        extras['test_noise_pattern'][sigma] = te_pattern
        # The sham ceiling: the same pattern computed from what the model
        # PREDICTED rather than the true label. If uncertainty tracks this just
        # as strongly, the model is following its own prediction, not the noise.
        # Computed here because quantile and outlier need the training reference,
        # which is not recoverable from the output file once folds 1-4 carry no
        # training rows.
        extras['test_noise_pattern_pred'][sigma] = pattern_injector.noise_scale(
            np.asarray(predictions[sigma], dtype=float), dose_reference_sd,
            reference=y_train,
            groups=test_noise_groups, reference_groups=noise_groups)
        if oof_folds and oof_folds > 1:
            def _fp(Xf, yf, Xs, gf=None):
                # `gf` is accepted and unused: the neural path carves its own
                # split below. The out-of-fold driver passes the fit rows'
                # scaffold ids to every inner fit because the tree path's
                # NGBoost needs them.
                #
                # Carve the early-stopping split from THIS inner fold's own fit
                # rows. Reusing the outer validation set would early-stop against
                # molecules sharing scaffolds with the rows being scored (26-42%
                # overlap on the real datasets), so the out-of-fold uncertainty
                # would not be on the same extrapolation footing as the test set.
                nv = max(1, len(yf) // 5)
                return train_neural_regression(Xf[nv:], yf[nv:], Xf[:nv], yf[:nv], Xs,
                                               model_type=model_type,
                                               epochs=NEURAL_DEFAULTS['training']['epochs'],
                                               y_scaler=y_scaler)
            # FIX: train_neural_regression consumes the GLOBAL torch generator
            # (weight init + shuffled DataLoader). Without this snapshot the
            # extra out-of-fold fits would shift the stream, so the MAIN model at
            # every later sigma would be initialised differently and this job's
            # R2/auc_norm would silently disagree with a run made without
            # --oof-folds.
            _tstate = torch.get_rng_state()
            _cstate = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
            try:
                    om, ou, oa, oe, ofold, n_ok = _oof_predict(
                    _fp, X_train, y_noisy, oof_folds, groups=train_groups)
            finally:
                torch.set_rng_state(_tstate)
                if _cstate is not None:
                    torch.cuda.set_rng_state_all(_cstate)
            extras['oof_mean'][sigma] = om
            extras['oof_unc'][sigma] = ou
            extras['oof_aleatoric_var'][sigma] = oa
            extras['oof_epistemic_var'][sigma] = oe
            extras['oof_fold'][sigma] = ofold
            extras['oof_folds_ok'][sigma] = n_ok
            # The sham ceiling for the TRAINING rows, computed HERE and not
            # a dozen lines above, where it used to sit: `extras['oof_mean']`
            # is written at the bottom of this block, so a test for the level
            # being in it, placed above the block, was false on every pass of
            # every loop and the column came out empty on every training row
            # this pipeline has ever produced.
            extras['train_noise_pattern_pred'][sigma] = _pattern_from_oof(
                pattern_injector, om, y_train, noise_groups,
                reference_dose=dose_reference_sd)

    return predictions, uncertainties, extras


def compute_metrics(y_true, predictions_dict):
    """Per-sigma regression metrics DataFrame."""
    rows = []
    for sigma in sorted(predictions_dict):
        yp = predictions_dict[sigma]
        rows.append({
            'sigma': sigma,
            'r2': r2_score(y_true, yp),
            'rmse': np.sqrt(mean_squared_error(y_true, yp)),
            'mae': mean_absolute_error(y_true, yp),
            'spearman': spearmanr(y_true, yp).correlation if np.std(yp) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# REPRESENTATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

# The settled representation set (RERUN_PLAN.md 13.7). Avalon and ChemBERTa were
# added 2026-08-26; SNS stays because the paper describes it as ECFP4's
# collision-free counterpart.
ALL_REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained', 'Avalon', 'ChemBERTa']


def generate_representations(smiles_list, rep_filter=None):
    """Generate all representations for a SMILES list. Returns dict of arrays.

    THE STRINGS THAT ARRIVE HERE CARRY STEREOCHEMISTRY AND QM9's DO NOT, AND
    THAT IS AN OPEN DECISION, NOT AN OVERSIGHT.

    `standardise_smiles` above canonicalises with RDKit's default
    isomericSmiles=True, so @/@@ and E/Z survive into the string every
    representation is built from. QM9 builds
    `Chem.MolToSmiles(mol, isomericSmiles=False)` (process_and_train.py:974,
    :1131) and features that. The four static representations do not care --
    measured on 100 stereo-bearing molecules from the cached LogD extract,
    ECFP4, Avalon, Sort & Slice and the PDV come out bit-identical either way,
    max|diff| exactly 0.0 on all four. MHG-GNN does care: max|diff| 40.0 against
    a feature spread of 11.6, i.e. 3.5 feature standard deviations, with all 100
    rows changed. So a cross-pipeline sentence about a learned embedding today
    compares a stereo-aware featurisation with a stereo-blind one. ChemBERTa is
    the other string-reading representation and was not measured here; it is not
    in this side's default grid either.

    What harmonising would cost, measured on the cached LogD extract: 1,249 of
    5,039 standardised SMILES (24.8%) carry stereochemistry. Stripping it HERE
    only -- at featurisation, leaving the loader alone -- changes no dataset and
    no split (`get_scaffold` is already chirality-blind: both forms of all 100
    test molecules land in the same scaffold group, so stereoisomers cannot be
    split between folds and made feature-identical across them), and moves the
    MHG-GNN and ChemBERTa numbers on this side alone.
    Stripping it in `standardise_smiles` instead would also merge 346 molecules
    into 173 rows by median, with label ranges inside a merged pair running to
    1.00 log units against a dataset spread of 1.19 -- a different dataset, not
    a different featurisation.

    Left as it is until the author rules, because the ruling is which half of
    the study changes. See RERUN_PLAN.md and the parity audit.
    """
    import gc
    reps = {}
    wanted = set(rep_filter) if rep_filter else set(ALL_REPS)

    if 'ECFP4' in wanted:
        print("  ECFP4...", flush=True)
        reps['ECFP4'] = create_ecfp4(smiles_list, n_bits=2048)
        print(f"    done: {reps['ECFP4'].shape}", flush=True)

    if 'PDV' in wanted:
        print("  PDV...", flush=True)
        reps['PDV'] = create_pdv(smiles_list)
        print(f"    done: {reps['PDV'].shape}", flush=True)

    if 'SNS' in wanted:
        print("  SNS...", flush=True)
        reps['SNS'], _ = create_sns(smiles_list, return_featurizer=True)
        print(f"    done: {reps['SNS'].shape}", flush=True)

    if 'MHG-GNN-pretrained' in wanted:
        print("  MHG-GNN (pretrained)...", flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reps['MHG-GNN-pretrained'] = create_mhg_gnn(smiles_list, batch_size=32)
            print(f"    done: {reps['MHG-GNN-pretrained'].shape}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e} — skipping MHG-GNN", flush=True)

    if 'Avalon' in wanted:
        print("  Avalon...", flush=True)
        reps['Avalon'] = create_avalon(smiles_list, n_bits=2048)
        print(f"    done: {reps['Avalon'].shape}", flush=True)

    if 'ChemBERTa' in wanted:
        print("  ChemBERTa...", flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reps['ChemBERTa'] = create_chemberta(smiles_list, batch_size=32)
            print(f"    done: {reps['ChemBERTa'].shape}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e} — skipping ChemBERTa", flush=True)

    # A representation that failed to build must STOP the job, exactly as a model
    # that cannot be constructed already does.
    #
    # MHG-GNN and ChemBERTa were the only two wrapped in `except Exception`, and
    # on failure the key was simply never added, so the representation vanished
    # from the roster built by `for rname in reps.keys()`. A job launched with
    # all six could exit 0, write a results file and be merged into the pooled
    # results with one representation missing FOR THAT DATASET ONLY -- unbalanced
    # coverage across the three datasets, and one line in a long log
    # (RERUN_PLAN.md 2.13).
    # An ALL-ZERO row is a molecule with no features carrying a real label into
    # training, and nothing downstream can tell it from a real one. Every builder
    # in src/kirby/representations/molecular.py catches an unparseable molecule
    # and appends np.zeros(n_bits) without logging or counting it; the Rust
    # pipeline treats the same event as a run-stopping failure, so the two had
    # opposite policies for one event (RERUN_PLAN.md 2.13). Checked here, once,
    # rather than in a builder that other experiments share.
    #
    # A learned embedding can legitimately be near zero but not exactly zero in
    # every dimension, so this is an exact test.
    blank = {}
    for name, matrix in reps.items():
        rows = np.asarray(matrix)
        if rows.ndim != 2:
            continue
        n_blank = int((~rows.any(axis=1)).sum())
        if n_blank:
            blank[name] = n_blank
    if blank:
        raise RuntimeError(
            f"all-zero feature rows: {blank}. A molecule with no features would "
            f"be trained on and scored as if it were real data, and every such "
            f"molecule looks identical to the model. The loaders drop "
            f"unparseable SMILES upstream, so this means something else failed.")

    missing = sorted(wanted - set(reps))
    if missing:
        raise RuntimeError(
            f"Refusing to run a partial roster: {missing} were requested and "
            f"could not be built. The FAILED lines above say why. A run with a "
            f"representation missing on one dataset only silently unbalances "
            f"every cross-representation aggregate.")

    return reps


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class RunIntegrityError(RuntimeError):
    """The run is misconfigured, or produced output that cannot be interpreted.

    A separate class because the per-condition loop below wraps every experiment
    in `except Exception`, logs it and carries on -- which is right for a model
    that failed to fit on one representation, and completely wrong for "the
    zero-level negative control is missing" or "this condition wrote no rows".
    Those must stop the job, so they are raised as this and re-raised there.
    """


def _emits_uncertainty(model_name):
    """Is `model_name` one of the models that emits a per-molecule uncertainty?

    Matched by PREFIX, the way the GP subsample branch matches 'GP'. Exact
    equality is what this used to do, and it silently excluded every variant
    built under a decorated name: a Gaussian process built as 'GP-Tanimoto'
    got no out-of-fold training-side scoring at all and nothing said so, so the
    output looked like a complete run of a model that simply had no signal.
    """
    return any(model_name == u or model_name.startswith(u + '-')
               for u in UNCERTAINTY_MODELS)


def _assert_cell_counts(frame, expected, context):
    """Every (split, noise_type, sigma) cell must hold exactly `expected` rows.

    The failure this catches is a block that comes out short or empty -- one
    level missing because its uncertainty was None, a condition that produced
    nothing, an out-of-fold pass that covered part of the training set. All of
    those read as a complete run in the output file, because a cell that is
    absent looks exactly like a cell that was never asked for.
    """
    if len(frame) == 0:
        raise RunIntegrityError(
            f"{context}: no rows at all. A condition that produces no rows must "
            f"stop the run -- an empty block is indistinguishable from a block "
            f"that was never requested once the file is merged.")
    counts = frame.groupby(['split', 'noise_type', 'sigma'], dropna=False).size()
    bad = counts[counts != expected]
    if len(bad):
        raise RunIntegrityError(
            f"{context}: expected {expected} rows in every "
            f"(split, condition, level) cell, got "
            + "; ".join(f"{k}={v}" for k, v in bad.items()))
    return counts


def _flush_uncertainties(all_uncertainties, results_dir, dataset_name):
    """Write the accumulated per-molecule uncertainty frames to disk.

    Called after EVERY outer fold, not only at the end. The out-of-fold pass
    multiplies the fit count by (1 + oof_folds), so a task is far more likely to
    hit its wall clock than before -- and previously everything this run exists
    to produce was held in memory until all five folds had finished, so a
    timeout at fold 5 destroyed all of it while leaving an all_results_partial
    file that made the job look productive.

    Rewriting the same rows is harmless: the merge below keys on
    (split, noise_type, fold, sigma) and replaces rather than duplicates.
    """
    results_dir = Path(results_dir)
    by_config = {}
    for model_name, rep_name, unc_df in all_uncertainties:
        by_config.setdefault((model_name, rep_name), []).append(unc_df)

    for (model_name, rep_name), frames in by_config.items():
        unc_all = pd.concat(frames, ignore_index=True)
        unc_all['dataset'] = dataset_name
        unc_all['model'] = model_name
        unc_all['rep'] = rep_name
        fname = f"{model_name.replace('-', '')}_{rep_name.replace('-', '')}_uncertainty_values.csv"
        out_path = results_dir / fname
        if out_path.exists():
            try:
                prev = pd.read_csv(out_path)
                keys = [c for c in ('split', 'noise_type', 'fold', 'sigma')
                        if c in prev.columns and c in unc_all.columns]
                if keys:
                    new_keys = set(map(tuple, unc_all[keys].drop_duplicates().values))
                    prev = prev[~prev[keys].apply(tuple, axis=1).isin(new_keys)]
                unc_all = pd.concat([prev, unc_all], ignore_index=True)
            except Exception as e:
                print(f"  WARN: could not merge existing {fname}: {e}")
        tmp = out_path.with_suffix('.csv.tmp')
        unc_all.to_csv(tmp, index=False)
        os.replace(tmp, out_path)   # atomic: a killed job never leaves a half file
        # Read the file BACK and count it. Asserting on the in-memory frame
        # would prove only that the frame was right; the thing that has to be
        # right is what landed on disk, and a truncated or half-merged write is
        # exactly the failure that reaches the analysis silently.
        _back = pd.read_csv(out_path)
        if len(_back) != len(unc_all):
            raise RunIntegrityError(
                f"{fname}: wrote {len(unc_all)} rows, read back {len(_back)}.")
        if {'split', 'noise_type', 'fold', 'sigma'} <= set(_back.columns):
            # Constant WITHIN a (split, fold), not across folds: scaffold folds
            # are not equal sizes, so the number of test molecules and the number
            # of training molecules both differ from fold to fold. What must not
            # differ is one condition or one level against another inside the
            # same fold -- that is a dropped block.
            _cells = _back.groupby(['split', 'fold', 'noise_type', 'sigma'],
                                   dropna=False).size()
            for _key, _grp in _cells.groupby(level=['split', 'fold']):
                if _grp.nunique() != 1:
                    raise RunIntegrityError(
                        f"{fname}: rows per (condition, level) cell are not "
                        f"constant within split/fold {_key} -- "
                        + "; ".join(f"{k}={v}" for k, v in _grp.items()))
        print(f"  Saved {fname}: {len(unc_all)} rows "
              f"({unc_all['split'].nunique() if 'split' in unc_all else 1} split(s), "
              f"{unc_all['noise_type'].nunique() if 'noise_type' in unc_all else 1} condition(s), "
              f"{unc_all['fold'].nunique() if 'fold' in unc_all else 1} fold(s))")


def scaffold_validation_carve(y_train_full, groups_full, fold_idx,
                              val_frac=0.2):
    """Split one fold's training block into training and early-stopping rows.

    Grouped by scaffold, like the outer CV. It used to be
    `np.arange(len(train_idx) // 5)` -- the first fifth of the block in dataset
    order, which is alphabetical by SMILES. Two things followed
    (RERUN_PLAN.md 2.12). The validation rows shared scaffolds with training, so
    early stopping was scored on molecules whose frameworks the model had just
    fitted, which is the optimism the outer scaffold split exists to rule out.
    And the same low-index molecules sit at the front of four folds out of five,
    so about a fifth of every dataset was validation nearly always and training
    almost never.

    Returns (train_idx_local, val_idx_local), positions within the fold's
    training block, each sorted.
    """
    carve = GroupShuffleSplit(n_splits=1, test_size=val_frac,
                              random_state=fold_idx)
    train_local, val_local = next(
        carve.split(np.zeros(len(y_train_full)), y_train_full, groups_full))
    train_local = np.sort(train_local)
    val_local = np.sort(val_local)
    shared = set(np.asarray(groups_full)[train_local]) & set(
        np.asarray(groups_full)[val_local])
    if shared:
        raise RunIntegrityError(
            f"fold {fold_idx}: {len(shared)} scaffold group(s) are in both the "
            f"training and the early-stopping validation split, so stopping is "
            f"scored on frameworks the model was fitted on.")
    if len(val_local) == 0 or len(train_local) == 0:
        raise RunIntegrityError(
            f"fold {fold_idx}: the validation carve produced "
            f"{len(train_local)} training and {len(val_local)} validation rows.")
    return train_local, val_local


# The Gaussian process's display names, per kernel: (plain, heteroscedastic).
#
# Written out rather than built with an f-string so that every name this file
# emits exists in it as a quoted literal. qsar_qm_models/model_names.json is the
# settled correspondence between the two pipelines and
# qsar_qm_models/scripts/test_tuning_rosters.py checks it against this file by
# searching for the literal -- which an assembled name defeats, and which was
# passing only because 'GP-Hetero' happened to appear in a comment.
#
# Identical to what the f-string produced: 'tanimoto'.capitalize() is
# 'Tanimoto'. A kernel not listed here still falls back to the assembled form,
# so adding one does not silently lose its heteroscedastic twin.
GP_DISPLAY_NAMES = {
    'rbf':      ('GP',           'GP-Hetero'),
    'tanimoto': ('GP-Tanimoto',  'GP-Tanimoto-Hetero'),
    'matern':   ('GP-Matern',    'GP-Matern-Hetero'),
}


def run_dataset(dataset_name, smiles, labels, results_dir, model_filter=None, rep_filter=None, sigma_levels=None,
                gp_kernel='rbf', gp_reps=None, unc_conditions='all', oof_folds=0,
                conditions=None, oof_outer_folds=None, noise_validation=True):
    """Run full scaffold CV experiment for one dataset."""
    if sigma_levels is None:
        # Levels are anchored to this dataset's published assay error, in log
        # units -- see NOISE_LEVELS_BY_DATASET.
        sigma_levels = NOISE_LEVELS_BY_DATASET.get(dataset_name, NOISE_LEVELS)
    # Which noise conditions this invocation runs. A subset lets one job cover
    # one condition, which is how the uncertainty runs are split across the queue.
    conditions = list(conditions) if conditions else list(NOISE_CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown noise condition(s) {unknown}. Known: {sorted(CONDITIONS)}. "
            "The six old strategy names (legacy, quantile, threshold, hetero, "
            "valprop, outlier) were deleted in noiseInject 1.0.0 -- they were one "
            "condition at six doses. See NOISE_DESIGN.md.")

    # Scaffold groups for the two grouped conditions. Separate from the CV
    # grouping: acyclic molecules are singletons here (section 2a rule 2).
    noise_groups_all, n_noise_groups, largest_share = assign_noise_groups(smiles)
    if any(c in GROUPED_CONDITIONS for c in conditions):
        print(f"  noise groups: {n_noise_groups} scaffolds, largest holds "
              f"{largest_share:.1%} of molecules")

    # Report which conditions produce a CONSTANT per-molecule noise scale on this
    # dataset. Those cannot answer "does uncertainty track WHERE the noise is" --
    # a rank correlation against a constant is undefined, not zero -- and it is
    # far better to see that here than after the cluster time is spent.
    for _c in conditions:
        _inj = NoiseInjectorRegression.from_condition(_c, random_state=0)
        _sc = _inj.noise_scale(labels, 1.0, reference=labels, groups=noise_groups_all)
        if len(np.unique(np.round(_sc, 12))) == 1:
            _note = ("expected: a shape-only condition gives every molecule the "
                     "same scale, so question B is UNDEFINED here, not zero"
                     if _inj.scale_is_degenerate is not False and
                     CONDITIONS[_c]['strategy'] == 'uniform'
                     else "DEGENERATE on this dataset -- question B cannot be "
                          "answered for this condition")
            print(f"  NOTE condition '{_c}': noise scale is CONSTANT ({_note})")
    # Reps on which to run the GP (gauche/gpytorch). Default {'PDV'} preserves
    # the original PDV-only RBF behaviour; pass gp_reps to run GP on more reps
    # (e.g. all four) so a single consistent GP model spans every rep and can
    # enter the cross-rep ANOVA.
    gp_rep_set = set(gp_reps) if gp_reps else {'PDV'}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    print(f"  N molecules: {len(smiles)}")
    print(f"  y range: [{labels.min():.3f}, {labels.max():.3f}], std={labels.std():.3f}")

    # Assign scaffold groups
    groups, n_scaffolds = assign_scaffold_groups(smiles)
    print(f"  N scaffolds: {n_scaffolds}")

    # Generate representations for ALL molecules
    print(f"\nGenerating representations for {len(smiles)} molecules...")
    reps = generate_representations(smiles, rep_filter=rep_filter)

    # Build experiment configs
    experiments = []
    for rname in reps.keys():
        # RF
        experiments.append(('RF', rname,
            lambda: RandomForestRegressor(random_state=42, n_jobs=-1,
                                          **sklearn_params('rf')), None))
        # QRF
        if HAS_QRF:
            experiments.append(('QRF', rname,
                lambda: RandomForestQuantileRegressor(random_state=42, n_jobs=-1,
                                                      **sklearn_params('qrf')), None))
        # XGBoost
        if HAS_XGBOOST:
            experiments.append(('XGBoost', rname,
                # learning_rate is the one that bit: unset falls through to the
                # booster default of 0.3, not the wrapper-documented 0.1.
                # Measured: 0.3 costs up to 0.05 R2 under injected noise.
                lambda: XGBRegressor(random_state=42,
                                     **sklearn_params('xgboost')), None))
        # NGBoost. EarlyStoppingNGBRegressor, not NGBRegressor: the spec's stage
        # selection is applied here now, the way QM9 applies it. See the class.
        if HAS_NGBOOST:
            def _make_ngboost():
                kwargs, stopping = _ngboost_kwargs()
                return EarlyStoppingNGBRegressor(
                    random_state=42, verbose=False, **kwargs, **stopping)
            experiments.append(('NGBoost', rname, _make_ngboost, None))
        # LightGBM
        if HAS_LGB:
            experiments.append(('LightGBM', rname,
                lambda: LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1,
                                      **sklearn_params('lightgbm')), None))
        # SVM — match QM9 defaults exactly: fixed RBF, C=1.0, gamma='scale',
        # for every representation. No per-rep kernel switching, no Optuna
        # tuning, so SVM is the same model on QM9 and on the validation
        # datasets and is free of the kernel--representation confound.
        if HAS_SVM:
            experiments.append(('SVM', rname,
                lambda: SVR(**sklearn_params('svm')), None))
        # GP — gauche/gpytorch ExactGP with learned GaussianLikelihood noise,
        # mirroring qsar_qm_models/models/models.py:504. Kernel selectable via
        # gp_kernel (rbf default — valid on every rep, matches QM9 gauche_rbf;
        # tanimoto only valid on binary fingerprints, e.g. ECFP4). Reps selectable
        # via gp_reps (default {'PDV'} preserves the original behaviour).
        # Display name encodes the kernel so the figure pipeline can map
        # RBF -> gauche_rbf and Tanimoto -> gauche (val_model_map update needed).
        if HAS_GP and rname in gp_rep_set:
            # Spelled out rather than assembled. Both names are checked
            # against this file by qsar_qm_models/scripts/test_tuning_rosters.py,
            # which searches for the quoted literal; an f-string leaves no
            # literal to find, so that check used to be satisfied by the word
            # appearing in a COMMENT further up. Same strings as before --
            # 'tanimoto'.capitalize() is 'Tanimoto' -- so nothing a row carries
            # changes.
            gp_name, gp_hetero_name = GP_DISPLAY_NAMES.get(
                gp_kernel,
                (f'GP-{gp_kernel.capitalize()}',
                 f'GP-{gp_kernel.capitalize()}-Hetero'))
            experiments.append((gp_name, rname,
                lambda k=gp_kernel: GaussianProcessGauche(
                    # From the shared spec, not literals: editing
                    # GP_DEFAULTS used to change QM9's GP and leave this one
                    # untouched, while the spec_hash on both files moved in
                    # lockstep asserting a parity that no longer held
                    # (RERUN_PLAN.md 2.13).
                    kernel=k,
                    init_noise=GP_DEFAULTS['likelihood_noise'],
                    init_outputscale=GP_DEFAULTS['outputscale']), None))

            # GP-Hetero — the same process with a network predicting the noise
            # for each molecule, so its data half and model half come apart.
            # It is the only model measured so far that separates them; both
            # forest halves report the same signal. Added 2026-08-28 on the
            # author's instruction, to the uncertainty runs only -- it answers
            # an uncertainty question and does not need a place on the accuracy
            # grid.
            experiments.append((gp_hetero_name, rname,
                lambda k=gp_kernel: HeteroscedasticGaussianProcessGauche(
                    kernel=k,
                    init_noise=GP_DEFAULTS['likelihood_noise'],
                    init_outputscale=GP_DEFAULTS['outputscale']), None))
        # Neural models. Display names match the val_model_map in
        # qsar_qm_models/scripts/generate_paper_figures.py:_normalize_validation_names
        # so rows merge into the canonical QM9 model namespace:
        #   DNN              -> dnn          (NN-α deterministic)
        #   BNN-Full         -> dnn_bnn_full (BNN-α, full BNN on DNN base)
        #   VBLL-Full        -> dnn_vbll     (VBLL-α, full VBLL on DNN base)
        #   MLP              -> mlp          (NN-β deterministic)
        #   MLP-BNN-Full     -> mlp_bnn_full (BNN-β, full BNN on MLP base)
        #   MLP-VBLL-Full    -> mlp_vbll     (VBLL-β, full VBLL on MLP base)
        # BNN-Last is implemented but not run by default (user directive).
        # VBLL-Full-Hetero / MLP-VBLL-Full-Hetero, added 2026-08-28: the same
        # variational models with a noise head, so their data-noise term varies
        # per molecule instead of being one number per fit. Without it their
        # correlation with per-molecule injected noise is zero however good the
        # model is -- a mechanism, not a result (RERUN_PLAN.md 5.5f). The suffix
        # matches GP-Hetero above, and _emits_uncertainty matches by prefix, so
        # both get the out-of-fold pass.
        # BNN-Full-MVE / MLP-BNN-Full-MVE, added 2026-09-01: a Bayesian network
        # with a VARIANCE OUTPUT HEAD -- Kendall & Gal eq. 6. The only route on
        # either pipeline to a network whose aleatoric term varies per molecule,
        # and the case the uncertainty literature actually holds up. QM9 reaches
        # the same model with `--bayesian-transformation full --loss
        # heteroscedastic`; here it is a model type, because this runner selects
        # by name rather than by flags.
        for mtype, mname in [('deterministic', 'DNN'),
                              ('full-bnn', 'BNN-Full'),
                              ('full-bnn-mve', 'BNN-Full-MVE'),
                              ('full-vbll', 'VBLL-Full'),
                              ('full-vbll-het', 'VBLL-Full-Hetero'),
                              ('mlp-deterministic', 'MLP'),
                              ('mlp-full-bnn', 'MLP-BNN-Full'),
                              ('mlp-full-bnn-mve', 'MLP-BNN-Full-MVE'),
                              ('mlp-full-vbll', 'MLP-VBLL-Full'),
                              ('mlp-full-vbll-het', 'MLP-VBLL-Full-Hetero')]:
            if mtype != 'deterministic' and not HAS_BAYESIAN_TORCH:
                continue
            experiments.append((mname, rname, None, mtype))

    # Filter to specific models if requested
    if model_filter:
        filter_set = set(model_filter)
        experiments = [(n, r, fn, mt) for n, r, fn, mt in experiments if n in filter_set]
        # A requested model that cannot be built must STOP the job. Previously an
        # absent optional package (gpytorch, quantile_forest, bayesian-torch)
        # only printed a warning on the first page of the log, the experiment
        # list came out empty, every fold looped over nothing, and the run
        # exited having produced no rows. That is exactly what happened to jobs
        # 12822693 and 12822694 on 2026-08-19.
        if not experiments:
            raise RuntimeError(
                f"No experiments to run: requested models {sorted(filter_set)} but none "
                f"could be constructed. Check the optional-import warnings at the top of "
                f"this log - the packages for those models are missing from this "
                f"interpreter ({sys.executable}).")
        built = {n for n, _, _, _ in experiments}
        missing = filter_set - built
        if missing:
            raise RuntimeError(
                f"Requested models {sorted(missing)} could not be constructed in this "
                f"environment ({sys.executable}). Refusing to run a partial roster - "
                f"fix the environment or drop them from --models explicitly.")
        print(f"  Filtered to models: {sorted(filter_set)}")

    print(f"\n{len(experiments)} model-rep configs × {len(conditions)} conditions "
          f"({', '.join(conditions)}) × {len(sigma_levels)} levels × {N_FOLDS} folds")

    # Scaffold CV
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_fold_results = []
    all_uncertainties = []
    # Which (model, rep) were SEEN to emit a per-molecule uncertainty, and which
    # of them actually reached the output file. Passing --conditions grouped_wider
    # without --unc-conditions all used to write nothing at all, in silence: the
    # job exited 0, produced R2 rows, and the uncertainty question it was
    # submitted to answer had no data behind it.
    unc_emitting = set()
    unc_written = set()
    unc_written_conditions = set()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(smiles, labels, groups)):
        print(f"\n{'─'*40}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─'*40}")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        # Split labels
        y_train_full = labels[train_idx]
        y_test = labels[test_idx]

        # Scaffold ids of this fold's training rows, so the validation carve and
        # the inner cross-fitting split can both be scaffold-grouped exactly like
        # the outer CV.
        groups_full = np.asarray(groups)[train_idx]

        # Carve validation from train (20% of train), BY SCAFFOLD.
        #
        # It used to be `np.arange(len(train_idx) // 5)` -- the first fifth of the
        # fold's training block in dataset order, which is alphabetical by SMILES.
        # Two things followed (RERUN_PLAN.md 2.12). The validation set shared
        # scaffolds with training, so early stopping was scored on molecules
        # whose frameworks the model had just fitted -- an optimistic stopping
        # signal that the outer scaffold CV exists to rule out. And because the
        # same low-index molecules sit at the front of four folds out of five,
        # roughly a fifth of every dataset was validation nearly always and
        # training almost never.
        train_idx_local, val_idx_local = scaffold_validation_carve(
            y_train_full, groups_full, fold_idx)
        y_train = y_train_full[train_idx_local]
        y_val = y_train_full[val_idx_local]
        print(f"  Early-stopping validation: {len(val_idx_local)} molecules, "
              f"{len(set(groups_full[val_idx_local]))} scaffold groups, "
              f"disjoint from the {len(set(groups_full[train_idx_local]))} "
              f"training groups")
        # HOW MANY ROWS ARE ACTUALLY FITTED, said out loud in every log.
        #
        # Five folds put 20% in the test block and the grouped early-stopping
        # carve takes a fifth of what is left, so a model here fits about 65% of
        # the dataset -- measured on the cached LogD extract, 0.634 to 0.668
        # across the five folds, 3,194 to 3,364 rows of 5,039. QM9 splits
        # 80/10/10 with the early-stopping block held out SEPARATELY, so it
        # fits the whole 80%. Neither side is wrong; both numbers follow from
        # settled choices (five folds here, three-way split there). It is
        # printed because a baseline R2 read across the two halves is not
        # like-for-like without it, and the Methods table needs the count.
        print(f"  Rows fitted this fold: {len(train_idx_local)} of {len(labels)} "
              f"({len(train_idx_local) / len(labels):.3f} of the dataset); QM9 "
              f"fits 0.80 of its sample")

        groups_train = groups_full[train_idx_local]

        # Re-fit the Sort & Slice vocabulary on THIS fold's training molecules.
        #
        # `generate_representations` runs before this loop, so the substructure
        # prevalence counts -- and therefore which 1,024 substructures occupy the
        # 1,024 feature slots -- used to be computed from the held-out molecules
        # as well. That is a structure-level leak into the feature basis of every
        # fold, in a study whose headline quantity is how performance degrades,
        # and it is asymmetric: only SNS has it. QM9 fits the featuriser on
        # training molecules only (RERUN_PLAN.md 2.13). Fitted on the rows the
        # model is fitted on, so the early-stopping rows are excluded too.
        fold_reps = reps
        if 'SNS' in reps:
            fold_reps = dict(reps)
            _fit_pos = train_idx[train_idx_local]
            _, _sns_featurizer = create_sns(
                [smiles[i] for i in _fit_pos], return_featurizer=True)
            fold_reps['SNS'] = create_sns(
                list(smiles), reference_featurizer=_sns_featurizer)

        # Run experiments
        for exp_idx, (model_name, rep_name, model_fn, model_type) in enumerate(experiments, 1):
            print(f"\n  [{exp_idx}/{len(experiments)}] {model_name} + {rep_name}...", flush=True)

            # Get representation splits
            X_full = fold_reps[rep_name]
            X_train_full = X_full[train_idx]
            X_test = X_full[test_idx]
            X_train = X_train_full[train_idx_local]
            X_val = X_train_full[val_idx_local]

            # Subsample for GP if needed. `gp_row_idx` keeps the position of
            # each retained molecule in the FULL training array, so the GP's
            # out-of-fold rows can be labelled with the same molecule index every
            # other model uses. Without it, joining GP to QRF on
            # (dataset, fold, sample_idx) pairs completely different molecules.
            if model_name.startswith('GP') and GP_MAX_N and len(X_train) > GP_MAX_N:
                gp_row_idx = np.random.RandomState(42).choice(
                    len(X_train), GP_MAX_N, replace=False)
                X_train_gp = X_train[gp_row_idx]
                y_train_gp = y_train[gp_row_idx]
            else:
                gp_row_idx = np.arange(len(X_train))
                X_train_gp = X_train
                y_train_gp = y_train

            for condition in conditions:
                # Pin the neural stream for THIS cell.
                #
                # There was no torch.manual_seed anywhere in this file. Weight
                # initialisation and the DataLoader shuffle drew from PyTorch's
                # global generator, seeded nondeterministically at process start
                # and consumed in sequence by every neural fit in the job -- so
                # a cell's result depended on how many fits ran before it, and a
                # `--models BNN-Full` job did not reproduce the same cell from a
                # full-roster job (RERUN_PLAN.md 2.13). Keying on the cell's own
                # identity makes it reproducible however the roster is split.
                _cell_seed = zlib.crc32(
                    f"{model_name}|{rep_name}|{condition}|{fold_idx}".encode('utf-8')
                ) & 0x7FFFFFFF
                torch.manual_seed(_cell_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_cell_seed)

                # Censoring is not dose-matched, so it is swept on its own axis:
                # the level is the FRACTION of labels clipped, not an amount of
                # noise. Sweeping it over the dose grid would clip the same
                # fraction at every level, including the clean baseline.
                levels = (CENSORING_LEVELS
                          if CONDITIONS[condition]['strategy'] == 'censoring'
                          else sigma_levels)
                print(f"    Condition: {condition}  levels {levels}", flush=True)
                # The noise groups for THIS fold's splits, taken from the
                # dataset-wide assignment so a molecule keeps its group whichever
                # fold it lands in.
                ng_train = noise_groups_all[train_idx][train_idx_local]
                ng_val = noise_groups_all[train_idx][val_idx_local]
                ng_test = noise_groups_all[test_idx]
                provenance = {}

                try:
                    # Cross-fitting only ever helps a model that emits a
                    # per-molecule uncertainty; it is wasted compute otherwise.
                    #
                    # It is also the ONLY expensive part of this run. The five
                    # scaffold folds are trained regardless (they produce the R2
                    # numbers), so saving TEST uncertainty for all five is free.
                    # Cross-fitting, by contrast, multiplies the fits, so it can
                    # be restricted to the first N outer folds: one outer fold
                    # still gives every training molecule an out-of-fold score.
                    # PREFIX match, not exact equality -- see _emits_uncertainty.
                    _oof = oof_folds if _emits_uncertainty(model_name) else 0
                    if _oof and oof_outer_folds is not None and fold_idx >= oof_outer_folds:
                        _oof = 0
                    if model_fn is not None:
                        # PREFIX, matching the subsample guard above and
                        # _emits_uncertainty. Exact equality routed 'GP-Tanimoto'
                        # down the ordinary path with the FULL training arrays,
                        # while the subsample guard had already built the
                        # subsampled ones -- two guards on one model that did not
                        # agree (RERUN_PLAN.md 2.13).
                        if model_name == 'GP' or model_name.startswith('GP-'):
                            predictions, uncertainties, extras = run_tree_experiment(
                                X_train_gp, y_train_gp, X_test, y_test,
                                model_fn, condition, levels, oof_folds=_oof,
                                train_groups=groups_train[gp_row_idx],
                                noise_groups=ng_train[gp_row_idx],
                                test_noise_groups=ng_test,
                                provenance=provenance, rep_name=rep_name)
                            y_train_used = y_train_gp
                            row_index = gp_row_idx
                        else:
                            predictions, uncertainties, extras = run_tree_experiment(
                                X_train, y_train, X_test, y_test,
                                model_fn, condition, levels, oof_folds=_oof,
                                train_groups=groups_train,
                                noise_groups=ng_train,
                                test_noise_groups=ng_test,
                                provenance=provenance, rep_name=rep_name)
                            y_train_used = y_train
                            row_index = np.arange(len(y_train))
                    else:
                        predictions, uncertainties, extras = run_neural_experiment(
                            X_train, y_train, X_val, y_val, X_test, y_test,
                            model_type, condition, levels, oof_folds=_oof,
                            train_groups=groups_train,
                            noise_validation=noise_validation,
                            noise_groups=ng_train,
                            test_noise_groups=ng_test,
                            val_noise_groups=ng_val,
                            provenance=provenance, rep_name=rep_name)
                        y_train_used = y_train
                        row_index = np.arange(len(y_train))

                    per_sigma = compute_metrics(y_test, predictions)
                    per_sigma['model'] = model_name
                    per_sigma['rep'] = rep_name
                    per_sigma['noise_type'] = condition
                    per_sigma['fold'] = fold_idx
                    per_sigma['dataset'] = dataset_name
                    # What the number in `sigma` MEASURES. These grids are in
                    # RAW LOG UNITS, anchored to published assay error, while
                    # QM9 doses in fractions of its clean label spread and the
                    # censoring axis is a fraction of labels clipped. All three
                    # used to be written into one column called `sigma` with
                    # nothing to tell them apart, and auc_norm is mean retention
                    # over each configuration's OWN level range -- so putting a
                    # QM9 auc_norm beside one of these compares retention over
                    # different spans (RERUN_PLAN.md 2.12).
                    per_sigma['level_units'] = (
                        'fraction_censored'
                        if CONDITIONS[condition]['strategy'] == 'censoring'
                        else 'label_sd')
                    # What the noise ACTUALLY delivered, per level. Nothing
                    # recorded this before, which is the single reason the
                    # dose confound survived the life of the project.
                    # RERUN_PLAN.md section 5.2 -- what every result row must
                    # carry. `shape_name` and `targeting_name` are here as well
                    # as `noise_type` so a row can be read without parsing the
                    # condition name, and `seed` and the clean label mean and
                    # spread so a row can be reproduced from itself.
                    _prov_cols = ('shape_name', 'targeting_name',
                                  'unit_dose_g', 'solved_scale',
                                  'target_dose_label_units',
                                  'realised_dose_label_units',
                                  'realised_dose_fraction_of_spread',
                                  'mean_epsilon', 'affected_molecule_fraction',
                                  'effective_n', 'censoring_limit', 'n_groups',
                                  'largest_group_share',
                                  'clean_label_mean', 'clean_label_sd',
                                  'seed', 'scale_is_degenerate')
                    for _col in _prov_cols:
                        per_sigma[_col] = per_sigma['sigma'].map(
                            lambda _s: provenance.get(_s, {}).get(_col))
                    # Provenance: which shared parameter spec built these models.
                    # A results CSV must be traceable to the numbers behind it.
                    for _k, _v in MODEL_SPEC.provenance_columns().items():
                        per_sigma[_k] = _v
                    # Which optimiser actually fitted the GP. The Adam fallback
                    # is a different fit and used to leave no trace on disk.
                    # Per level: the value recorded at THAT level, not the
                    # last one the loop happened to see.
                    # The clean training label spread this fold standardised
                    # by. It is written onto every uncertainty row so the
                    # analysis can put this pipeline's uncertainties on the same
                    # scale as QM9's -- fractions of the clean training label
                    # spread, settled 2026-08-27. A run that does not report it
                    # STOPS: an uncertainty column whose units are unknown is
                    # worse than a missing one, because it still averages.
                    _label_scale = extras.get('label_scale')
                    if _label_scale is None or not np.isfinite(_label_scale) \
                            or _label_scale <= 0:
                        raise RuntimeError(
                            f"{model_name}+{rep_name} {condition}: the runner "
                            f"reported no usable clean training label spread "
                            f"({_label_scale!r}), so the uncertainty rows would "
                            f"be written in units nothing downstream can convert.")
                    _label_scale = float(_label_scale)
                    _fitm = extras.get('gp_fit_method', {})
                    _coll = extras.get('gp_collapsed', {})
                    if isinstance(_fitm, dict):
                        per_sigma['gp_fit_method'] = per_sigma['sigma'].map(
                            lambda _s: _fitm.get(_s, ''))
                    else:
                        per_sigma['gp_fit_method'] = _fitm
                    per_sigma['gp_collapsed'] = per_sigma['sigma'].map(
                        lambda _s: _coll.get(_s, '') if isinstance(_coll, dict) else '')
                    all_fold_results.append(per_sigma)

                    # ---- Uncertainty output -------------------------------
                    # Was: one condition only, and the writer then overwrote the
                    # file once per fold so only the LAST fold survived and
                    # neither the condition nor the fold reached disk. Both fixed.
                    save_this = (unc_conditions == 'all') or (condition == 'gaussian')

                    # These were one expression, `uncertainties.get(0.0) is not
                    # None`, doing two unrelated jobs: deciding whether this model
                    # emits a per-molecule uncertainty at all, and checking that
                    # the zero level was run. Leaving 0.0 out of --sigmas
                    # therefore produced NO uncertainty output whatsoever, test
                    # rows included, and said nothing about it. Split apart:
                    emits = any(uncertainties.get(s) is not None for s in levels)
                    if emits:
                        unc_emitting.add((model_name, rep_name))
                        missing = [s for s in levels if uncertainties.get(s) is None]
                        if missing:
                            raise RunIntegrityError(
                                f"{model_name}+{rep_name} {condition}: emitted a "
                                f"per-molecule uncertainty at some levels but not at "
                                f"{missing}. A level with no uncertainty leaves a hole "
                                f"in the file that reads as a level that was never run.")
                        if 0.0 not in levels:
                            raise RunIntegrityError(
                                f"{model_name}+{rep_name} emits a per-molecule "
                                f"uncertainty but level 0.0 is not in the grid "
                                f"{list(levels)}. The zero level is the NEGATIVE "
                                f"CONTROL: the model there saw the same labels and no "
                                f"corruption, so its correlation against the noise "
                                f"shape is the label-magnitude confound and nothing "
                                f"else. Every reported effect is (correlation at the "
                                f"level) minus (correlation at zero), so without it "
                                f"the run cannot be interpreted. Add 0.0 to --sigmas.")
                        # A model that emits an uncertainty but is not recognised as
                        # one gets no out-of-fold scoring, and used to get it in
                        # silence -- exact-equality matching excluded 'GP-Tanimoto'
                        # and every other decorated name.
                        if oof_folds and oof_folds > 1 and not _emits_uncertainty(model_name):
                            raise RunIntegrityError(
                                f"--oof-folds {oof_folds} was requested and "
                                f"{model_name} emits a per-molecule uncertainty, but "
                                f"it is not matched by UNCERTAINTY_MODELS "
                                f"({UNCERTAINTY_MODELS}), so it would be cross-fitted "
                                f"nowhere and no training rows would be written. Add "
                                f"its family to UNCERTAINTY_MODELS.")

                    if save_this and emits:
                        # (B) TEST molecules: does the model become less certain
                        # in the regions the condition corrupts? `noise_scale` is
                        # what the molecule's region WOULD receive -- the model
                        # never saw it, so this is a genuine prediction target.
                        # Which SUPPORT entry describes this model, and the
                        # guard that stops a run whose numbers disagree with it.
                        # A component declared per molecule that is really one
                        # number copied onto every row correlates with
                        # per-molecule injected noise at exactly zero however
                        # good the model is, and nothing in the output file said
                        # which it was (RERUN_PLAN.md 5.5a point 5).
                        _alea_kind, _epis_kind = unc_support(model_name)
                        unc_rows = []
                        for sigma in levels:
                            ts = extras['test_noise_scale'].get(sigma)
                            tp = extras['test_noise_pattern'].get(sigma)
                            tpp = extras['test_noise_pattern_pred'].get(sigma)
                            _ta = extras['test_aleatoric_var'].get(sigma)
                            _te = extras['test_epistemic_var'].get(sigma)
                            assert_matches_support(model_name, _ta, _te,
                                                   n_molecules=len(y_test))
                            # The single conversion from a variance to a spread.
                            _ta_sd = variance_to_std(_ta)
                            _te_sd = variance_to_std(_te)
                            for i in range(len(y_test)):
                                unc_rows.append({
                                    'split': 'test',
                                    'noise_type': condition,
                                    'sigma': sigma,
                                    'fold': fold_idx,
                                    'sample_idx': i,
                                    # sample_idx is a WITHIN-FOLD position, so it
                                    # repeats across folds and identifies nothing
                                    # on its own. mol_idx is the row in the full
                                    # dataset: it is what joins a molecule across
                                    # folds, conditions and models.
                                    'mol_idx': int(test_idx[i]),
                                    'y_true': y_test[i],
                                    'y_pred': predictions[sigma][i],
                                    'uncertainty': uncertainties[sigma][i],
                                    # The two halves of it. This pipeline wrote
                                    # one column and performed no split, and the
                                    # paper's uncertainty section is built on
                                    # these three datasets.
                                    'aleatoric_uncertainty': (
                                        float(_ta_sd[i]) if _ta_sd is not None else np.nan),
                                    'epistemic_uncertainty': (
                                        float(_te_sd[i]) if _te_sd is not None else np.nan),
                                    'aleatoric_support': _alea_kind,
                                    'epistemic_support': _epis_kind,
                                    'noise_scale': (float(ts[i]) if ts is not None else np.nan),
                                    'noise_pattern': (float(tp[i]) if tp is not None else np.nan),
                                    'noise_pattern_pred': (float(tpp[i]) if tpp is not None else np.nan),
                                    # What `uncertainty` and `y_pred` above are
                                    # divided by to reach the settled scale --
                                    # fractions of the clean training label
                                    # spread. Written on the row because the
                                    # analysis pools these files with QM9's,
                                    # whose columns are already on that scale.
                                    'label_scale': _label_scale,
                                    'injected_noise': 0.0,   # test labels are never corrupted
                                    # Emitted on EVERY row, test included. If the column
                                    # appeared only where out-of-fold rows exist, task files
                                    # would have different column sets and the streaming merge
                                    # -- which writes one header and appends -- would shift
                                    # every later column silently.
                                    'oof_folds_ok': -1,
                                })
                        _test_frame = pd.DataFrame(unc_rows)
                        # What the run INTENDED: one row per test molecule per
                        # level, no more and no fewer.
                        _assert_cell_counts(
                            _test_frame, len(y_test),
                            f"{model_name}+{rep_name} {condition} fold {fold_idx} "
                            f"test rows ({len(levels)} levels x {len(y_test)} molecules)")
                        all_uncertainties.append((model_name, rep_name, _test_frame))
                        unc_written.add((model_name, rep_name))
                        unc_written_conditions.add((model_name, rep_name, condition))

                        # (A) TRAINING molecules, scored out-of-fold: do the
                        # molecules whose labels were actually corrupted come
                        # back as the uncertain ones? `injected_noise` here is
                        # the epsilon the injector really drew, not a regression
                        # estimate.
                        if extras['oof_unc']:
                            tr_rows = []
                            _levels_written = []
                            for sigma in levels:
                                ou = extras['oof_unc'].get(sigma)
                                om = extras['oof_mean'].get(sigma)
                                if ou is None:
                                    continue
                                if not np.isfinite(ou).any():
                                    # Every inner fold failed. Writing this block
                                    # would put an all-blank cell into the output
                                    # that the coverage report counts as present.
                                    #
                                    # CHECKED FIRST, before the support guard
                                    # below. When no inner fold succeeds,
                                    # `_oof_predict` returns None for both halves
                                    # of the split -- correctly, because nothing
                                    # produced one -- and the guard then refuses
                                    # the block for a model whose SUPPORT entry
                                    # says it HAS that half. That is the guard
                                    # answering a question nobody asked: it exists
                                    # to catch a model whose output disagrees with
                                    # what it can do, not a computation that did
                                    # not happen. Raising there turned a clean
                                    # "block skipped" into an exception caught by
                                    # the per-fold handler, so the line below was
                                    # never printed and the end-of-run gate
                                    # reported the silence instead (2026-08-30).
                                    print(f"      [oof] sigma={sigma}: no finite "
                                          f"out-of-fold values — block skipped",
                                          flush=True)
                                    continue
                                _oa = extras['oof_aleatoric_var'].get(sigma)
                                _oe = extras['oof_epistemic_var'].get(sigma)
                                _ofold = extras['oof_fold'].get(sigma)
                                # `blocks` because these rows come from several
                                # fits: a component that is one number per fit
                                # takes a different value in each fold, and that
                                # is fold variation, not per-molecule variation.
                                assert_matches_support(
                                    model_name, _oa, _oe,
                                    n_molecules=len(y_train_used),
                                    blocks=_ofold)
                                _oa_sd = variance_to_std(_oa)
                                _oe_sd = variance_to_std(_oe)
                                eps = extras['train_epsilon'][sigma]
                                sc = extras['train_noise_scale'][sigma]
                                pt = extras['train_noise_pattern'][sigma]
                                ptp = extras['train_noise_pattern_pred'].get(sigma)
                                n_ok = extras['oof_folds_ok'].get(sigma, 0)
                                _levels_written.append(sigma)
                                for i in range(len(y_train_used)):
                                    tr_rows.append({
                                        'split': 'train_oof',
                                        'noise_type': condition,
                                        'sigma': sigma,
                                        'fold': fold_idx,
                                        'oof_folds_ok': n_ok,
                                        'sample_idx': int(row_index[i]),
                                        'mol_idx': int(train_idx[train_idx_local[int(row_index[i])]]),
                                        'y_true': y_train_used[i],
                                        'y_pred': om[i],
                                        'uncertainty': ou[i],
                                        'aleatoric_uncertainty': (
                                            float(_oa_sd[i]) if _oa_sd is not None else np.nan),
                                        'epistemic_uncertainty': (
                                            float(_oe_sd[i]) if _oe_sd is not None else np.nan),
                                        'aleatoric_support': _alea_kind,
                                        'epistemic_support': _epis_kind,
                                        'noise_scale': float(sc[i]),
                                        'noise_pattern': float(pt[i]),
                                        'noise_pattern_pred': (float(ptp[i])
                                                               if ptp is not None else np.nan),
                                        # See the test rows for what this is.
                                        'label_scale': _label_scale,
                                        'injected_noise': float(eps[i]),
                                    })
                            if tr_rows:
                                _train_frame = pd.DataFrame(tr_rows)
                                # One row per scored TRAINING molecule per level
                                # that produced any finite out-of-fold value. A
                                # level where every inner fold failed is skipped
                                # above and is deliberately absent from
                                # `_levels_written`, so a short block cannot hide
                                # behind that skip.
                                _assert_cell_counts(
                                    _train_frame, len(y_train_used),
                                    f"{model_name}+{rep_name} {condition} fold "
                                    f"{fold_idx} out-of-fold training rows "
                                    f"({len(_levels_written)} levels x "
                                    f"{len(y_train_used)} molecules)")
                                if len(_levels_written) != len(_train_frame.groupby(
                                        ['noise_type', 'sigma'], dropna=False)):
                                    raise RunIntegrityError(
                                        f"{model_name}+{rep_name} {condition} fold "
                                        f"{fold_idx}: wrote out-of-fold blocks for "
                                        f"{_levels_written} but the frame holds a "
                                        f"different set of levels.")
                                all_uncertainties.append(
                                    (model_name, rep_name, _train_frame))
                            elif any(np.isfinite(np.asarray(_v, dtype=float)).any()
                                     for _v in extras['oof_unc'].values()):
                                # Finite out-of-fold values exist and none of
                                # them reached a row: that is a bookkeeping
                                # failure and must stop the run. A level where
                                # EVERY inner fold failed is a different thing --
                                # it is skipped above, loudly, and leaves no
                                # blank cell for the coverage report to count.
                                raise RunIntegrityError(
                                    f"{model_name}+{rep_name} {condition} fold "
                                    f"{fold_idx}: finite out-of-fold values exist "
                                    f"but not one training row was written.")

                except (RunIntegrityError, DoseError):
                    # NOT swallowed. Logging one of these and carrying on is how
                    # a run finishes green having produced output nobody can
                    # interpret; the broad handler below is for a model that
                    # failed to fit, which is a different thing entirely.
                    #
                    # DoseError belongs here for exactly that reason. It says the
                    # injector handed back an amount nobody asked for, so the
                    # cell would report a level it did not receive -- and caught
                    # below it would vanish as one printed line, which is the
                    # failure this file keeps having to fix.
                    #
                    # The injector does not raise it today: it warns and carries
                    # on, because a three-sigma band is crossed by a working
                    # injector on about 1% of draws at these dataset sizes and
                    # the delivered amount is recorded on every row either way
                    # (qsar_qm_models RERUN_PLAN.md 2.14a). This keeps the route
                    # open for anything that is ever made fatal.
                    raise
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    continue

        # Incremental save after each fold (protection against timeout)
        if all_uncertainties:
            _flush_uncertainties(all_uncertainties, results_dir, dataset_name)
        if all_fold_results:
            _partial = pd.concat(all_fold_results, ignore_index=True)
            _partial_path = results_dir / 'all_results_partial.csv'
            _partial.to_csv(_partial_path, index=False)
            print(f"  [Checkpoint] Saved {len(_partial)} rows after fold {fold_idx + 1}")

    if not all_fold_results:
        # Every (model, condition) raised. Returning quietly here made the job
        # exit 0 with no output, which is indistinguishable from success in
        # sacct and would have been discovered only at merge time.
        raise RuntimeError(
            f"No results produced for {dataset_name} — every model/condition "
            f"combination raised. Check the per-condition ERROR lines above.")

    # Combine results across folds
    combined = pd.concat(all_fold_results, ignore_index=True)

    # A requested MODEL that produced no rows at all stops the job, the way a
    # requested representation already does.
    #
    # The all-failed guard above only fires when EVERY model failed. One model
    # failing everywhere is the more likely case and the more damaging one: it
    # exits 0, writes a results file, and is merged into the pooled results with
    # that model absent for this dataset only. Missing cells are not missing at
    # random, so any ANOVA or Wilcoxon over the model x representation grid is
    # then computed on a biased subset (RERUN_PLAN.md 2.13). Seen live on
    # 2026-08-27: quantile_forest raises 'Invalid parameter monotonic_cst' on
    # this sklearn, and QRF contributed nothing while the run finished green.
    _asked = {name for name, _rep, _fn, _mt in experiments}
    _produced = set(combined['model'].unique()) if 'model' in combined else set()
    _silent = sorted(_asked - _produced)
    if _silent:
        raise RuntimeError(
            f"{_silent} were requested and produced NO rows for {dataset_name}. "
            f"The per-condition ERROR lines above say why. A results file with a "
            f"model missing on one dataset only silently unbalances every "
            f"cross-model aggregate.")

    _asked_reps = {rep for _n, rep, _fn, _mt in experiments}
    _silent_reps = sorted(_asked_reps - (set(combined['rep'].unique())
                                         if 'rep' in combined else set()))
    if _silent_reps:
        raise RuntimeError(
            f"{_silent_reps} were requested and produced NO rows for "
            f"{dataset_name}. See the ERROR lines above.")

    # Merge with existing results if running a subset of models/reps
    existing_path = results_dir / 'all_results.csv'
    # `conditions` is part of this guard, not only model and rep.
    #
    # --conditions is a first-class way to split a run across the queue -- its
    # own help text says so -- and it was NOT in the guard, so a job that ran one
    # condition rewrote all_results.csv with that condition's rows alone and
    # destroyed every other condition already in the file (RERUN_PLAN.md 2.13).
    # A condition-only run is a subset run like any other.
    _condition_subset = (conditions is not None
                         and set(conditions) != set(CONDITIONS))
    if (model_filter or rep_filter or _condition_subset) and existing_path.exists():
        existing = pd.read_csv(existing_path)
        # Remove old rows ONLY for the combinations we are re-running.
        mask = pd.Series(True, index=existing.index)
        if model_filter:
            mask &= existing['model'].isin(model_filter)
        if rep_filter:
            mask &= existing['rep'].isin(rep_filter)
        if _condition_subset and 'noise_type' in existing.columns:
            mask &= existing['noise_type'].isin(list(conditions))
        existing = existing[~mask]
        combined = pd.concat([existing, combined], ignore_index=True)
        print(f"  Merged with existing results: {len(combined)} total rows "
              f"(kept {len(existing)} rows this run did not produce)")

    combined.to_csv(results_dir / 'all_results.csv', index=False)

    # Aggregate across folds and compute curve-shape robustness metrics
    summary_rows = []
    for (model, rep, strat), grp in combined.groupby(['model', 'rep', 'noise_type']):
        # Average across folds first, then summarise the degradation curve
        fold_avgs = grp.groupby('sigma')[['r2', 'rmse', 'mae', 'spearman']].mean().reset_index()
        fold_avgs = fold_avgs.sort_values('sigma')

        sigmas = fold_avgs['sigma'].values
        r2_vals = fold_avgs['r2'].values
        rmse_vals = fold_avgs['rmse'].values

        baseline_r2 = r2_vals[0]
        baseline_rmse = rmse_vals[0]

        # Curve-shape robustness from R² retention (no linear assumption).
        # auc_norm is PRIMARY (higher = more robust); weibull_* are supplementary.
        rob = robustness_metrics(sigmas, r2_vals)

        # Retention at sigma=0.5 and 1.0
        # Retention at a GRID POSITION, not at a literal level.
        #
        # These two used to select `sigma == 0.5` and `sigma == 1.0` by exact
        # float equality. The anchored grids are LogD [0, 0.15, 0.3, 0.5, 0.75,
        # 1.0], Caco-2 [0, 0.1, 0.2, 0.35, 0.5, 0.7] and hERG [0, 0.15, 0.3,
        # 0.54, 0.8, 1.1] -- Caco-2 has 0.5 but no 1.0, hERG has neither. So
        # retention_1.0 was populated for LogD only and retention_0.5 for LogD
        # and Caco-2 only, and averaging those columns across datasets compared
        # one dataset against a blank while the hERG rows looked ordinary
        # (RERUN_PLAN.md 2.13). The level each number was read at is written
        # beside it, because it differs by dataset.
        _levels = fold_avgs['sigma'].values
        _mid = len(_levels) // 2
        _top = len(_levels) - 1
        retention_mid_level = float(_levels[_mid])
        retention_top_level = float(_levels[_top])
        retention_mid = (float(r2_vals[_mid]) / baseline_r2
                         if baseline_r2 > 0 else np.nan)
        retention_top = (float(r2_vals[_top]) / baseline_r2
                         if baseline_r2 > 0 else np.nan)

        # CV std of baseline R²
        baseline_std = grp[grp['sigma'] == 0.0]['r2'].std()

        # auc_norm PER FOLD, and the spread across them.
        #
        # The five folds were averaged into one curve above and auc_norm computed
        # once from that average, so the only spread that survived was
        # baseline_r2_std -- the folds at level 0, and nothing else. Every fold
        # already writes its own row, so this costs no extra fits: compute the
        # curve per fold, summarise each, and report the spread of the five
        # (RERUN_PLAN.md 2.15).
        #
        # It is NOT the same quantity as QM9's spread across replicates. A QM9
        # replicate reshuffles all 129,428 molecules and takes a fresh sample, so
        # its spread contains sampling variance. These folds partition one fixed
        # dataset and share training molecules, so theirs does not. The two are
        # reported in separate columns and must not share an axis or a test.
        _fold_aucs = []
        for _f, _fgrp in grp.groupby('fold'):
            _fc = _fgrp.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
            if len(_fc) < 2:
                continue
            _fr = robustness_metrics(_fc['sigma'].values, _fc['r2'].values)
            if np.isfinite(_fr['auc_norm']):
                _fold_aucs.append(float(_fr['auc_norm']))
        auc_norm_fold_mean = float(np.mean(_fold_aucs)) if _fold_aucs else np.nan
        auc_norm_fold_std = (float(np.std(_fold_aucs, ddof=1))
                             if len(_fold_aucs) > 1 else np.nan)
        auc_norm_n_folds = len(_fold_aucs)

        summary_rows.append({
            'dataset': dataset_name,
            'model': model,
            'rep': rep,
            'noise_type': strat,
            'baseline_r2': baseline_r2,
            'baseline_r2_std': baseline_std,
            'baseline_rmse': baseline_rmse,
            'auc_norm': rob['auc_norm'],
            # auc_norm computed per fold, then averaged, and the spread of the
            # per-fold values. `auc_norm` above is computed from the fold-
            # AVERAGED curve, which is a different number; both are reported.
            'auc_norm_fold_mean': auc_norm_fold_mean,
            'auc_norm_fold_std': auc_norm_fold_std,
            'auc_norm_n_folds': auc_norm_n_folds,
            'weibull_tau': rob['weibull_tau'],
            'weibull_beta': rob['weibull_beta'],
            'retention_mid': retention_mid,
            'retention_mid_level': retention_mid_level,
            'retention_top': retention_top,
            'retention_top_level': retention_top_level,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Merge with existing summary if running a subset of models/reps
    existing_summary = results_dir / 'summary.csv'
    # Same guard as all_results.csv above, for the same reason.
    if (model_filter or rep_filter or _condition_subset) and existing_summary.exists():
        existing = pd.read_csv(existing_summary)
        mask = pd.Series(True, index=existing.index)
        if model_filter:
            mask &= existing['model'].isin(model_filter)
        if rep_filter:
            mask &= existing['rep'].isin(rep_filter)
        if _condition_subset and 'noise_type' in existing.columns:
            mask &= existing['noise_type'].isin(list(conditions))
        existing = existing[~mask]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(results_dir / 'summary.csv', index=False)

    # Save uncertainty data (final pass; also written incrementally per fold).
    _flush_uncertainties(all_uncertainties, results_dir, dataset_name)

    # Print summary
    print(f"\n{'─'*80}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'─'*80}")

    working = summary_df[summary_df['baseline_r2'] >= 0.3]
    print(f"  Working configs (baseline R² >= 0.3): {len(working)}/{len(summary_df)}")

    if len(working) > 0:
        print(f"\n  Top 5 most robust (highest auc_norm):")
        top5 = working.nlargest(5, 'auc_norm')
        for _, row in top5.iterrows():
            print(f"    {row['model']:12} + {row['rep']:20} + {row['noise_type']:10}: "
                  f"auc_norm={row['auc_norm']:.4f}, baseline={row['baseline_r2']:.3f}±{row['baseline_r2_std']:.3f}")

        print(f"\n  By representation (mean across models/conditions):")
        rep_agg = working.groupby('rep')[['baseline_r2', 'auc_norm', 'retention_mid']].mean()
        print(rep_agg.round(4).to_string())

        print(f"\n  By model (mean across reps/conditions):")
        mod_agg = working.groupby('model')[['baseline_r2', 'auc_norm', 'retention_mid']].mean()
        print(mod_agg.round(4).to_string())

    # ---- integrity gates -------------------------------------------------
    # LAST, after every file has been written: a gate that fired before the
    # writes would throw away the whole run's compute to report a bookkeeping
    # fault, and the files are what a person needs in order to diagnose it. The
    # job still exits non-zero, which is the part that matters.
    _silent = sorted(unc_emitting - unc_written)
    if _silent:
        raise RunIntegrityError(
            f"{dataset_name}: {_silent} emit a per-molecule uncertainty but not "
            f"one uncertainty row was written. --unc-conditions is "
            f"'{unc_conditions}' and the conditions run were {conditions}; with "
            f"'gaussian' only the plain Gaussian condition is saved, and its "
            f"noise scale is the same for every molecule, which is the one "
            f"condition where 'does uncertainty find where the noise is' has no "
            f"answer. Pass --unc-conditions all.")
    if unc_conditions == 'all':
        # Only when every condition was MEANT to be written. Under
        # --unc-conditions gaussian the absent conditions are the point.
        _missing_cond = sorted(
            (m, r, c) for (m, r) in unc_written for c in conditions
            if (m, r, c) not in unc_written_conditions)
        if _missing_cond:
            raise RunIntegrityError(
                f"{dataset_name}: these (model, rep, condition) combinations wrote "
                f"no uncertainty rows while others for the same model did: "
                f"{_missing_cond}. A condition that produces no rows must stop the "
                f"run -- once merged, an absent condition is indistinguishable "
                f"from one that was never requested. Check the per-condition ERROR "
                f"lines above.")

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    """The command-line surface, built where a test can execute it.

    Separate from `main()` so the defaults and the rejections below can be
    exercised by running argparse itself rather than by reading this file for a
    matching string. A string match passes whether or not the line it matched
    ever runs, and this project has already shipped one of those.
    """
    parser = argparse.ArgumentParser(
        description='Validation Noise Robustness: 3 Regression Datasets with Scaffold CV')
    parser.add_argument('--datasets', nargs='+',
                        choices=['logd', 'caco2', 'herg_ki', 'all'],
                        default=['all'],
                        help='Which datasets to test (default: all)')
    parser.add_argument('--openadmet-csv', type=str, default=None,
                        help='Path to cached OpenADMET CSV')
    parser.add_argument('--results-root', type=str, default='results/validation',
                        help='Root directory for results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Only run these models (e.g. --models NGBoost SVM LightGBM)')
    parser.add_argument('--reps', nargs='+', default=None,
                        choices=ALL_REPS,
                        help='Only run these reps (e.g. --reps PDV). Merges with existing results.')
    parser.add_argument('--sigmas', nargs='+', type=float, default=None,
                        help='Only run these sigma levels (e.g. --sigmas 0.0 0.5 1.0). Default: all 11.')
    parser.add_argument('--gp-kernel', type=str, default='rbf', choices=['rbf', 'tanimoto'],
                        help="GP (gauche) kernel. 'rbf' (default) is valid on all reps and matches "
                             "QM9 gauche_rbf; 'tanimoto' is only valid on binary fingerprints (ECFP4).")
    parser.add_argument('--gp-reps', nargs='+', default=None,
                        choices=ALL_REPS,
                        help='Reps on which to run the GP. Default: PDV only (original behaviour). '
                             'Pass all four to run GP on every rep so it can enter the cross-rep ANOVA.')
    parser.add_argument('--conditions', nargs='+', default=None,
                        choices=sorted(CONDITIONS),
                        help='Only run these noise conditions (default: NOISE_CONDITIONS). '
                             'Splitting a run by condition is how the '
                             'uncertainty jobs are spread across the queue. Every condition '
                             'delivers the SAME amount of noise at a given level, so any '
                             'difference between them is a difference of shape.')
    parser.add_argument('--unc-conditions', type=str, default='all',
                        choices=['gaussian', 'all'],
                        help="Which noise conditions get their per-molecule uncertainty written "
                             "to disk. 'all' is now the DEFAULT: under 'gaussian' the only "
                             "condition written is plain Gaussian, whose noise scale is the same "
                             "for every molecule, and that is the single condition where 'does "
                             "uncertainty find where the noise is' has no answer to find. "
                             "'all' is needed to "
                             "ask whether a model learns WHERE the data is unreliable, since the "
                             "shape-only conditions give every molecule the same noise scale and "
                             "so have no pattern to learn -- for those the question is UNDEFINED, "
                             "not zero. grouped_wider and the censoring conditions are the ones "
                             "with a pattern to find.")
    parser.add_argument('--no-noise-validation', action='store_true',
                        help="Leave the early-stopping validation labels CLEAN. Default is to "
                             "corrupt them with the same condition and level as training: "
                             "otherwise the neural models are explicitly selected NOT to fit the "
                             "injected noise, which suppresses the quantity being measured and "
                             "does so for the neural half of the roster only.")
    parser.add_argument('--oof-outer-folds', type=int, default=None,
                        help="Cross-fit only the first N of the 5 scaffold folds (default: all "
                             "of them). The 5 folds are trained either way, so TEST-side "
                             "uncertainty is free for all 5; this only limits the expensive "
                             "out-of-fold TRAINING pass. --oof-outer-folds 1 cuts the added "
                             "cost about threefold and still gives every training molecule an "
                             "out-of-fold score, just without a spread across folds.")
    parser.add_argument('--oof-folds', type=int, default=0,
                        help="Cross-fit the training set into this many folds and save an "
                             "out-of-fold uncertainty for every TRAINING molecule (0 = off, "
                             "the default). Needed to ask whether uncertainty flags the labels "
                             "that were actually corrupted: scoring a molecule with a model that "
                             "fitted its corrupted label measures memorisation, not uncertainty. "
                             "Costs this many extra fits per noise level, and only applies to the "
                             "models that emit a per-molecule uncertainty.")
    return parser


def validate_args(parser, args):
    """Reject argument combinations that would run and then mean nothing."""
    # --oof-folds 1 used to be accepted and then silently ignored: the trigger in
    # both runners is `oof_folds > 1`, so the job ran, cost nothing extra, and
    # produced a file with no training rows in it and no explanation.
    if args.oof_folds == 1:
        parser.error(
            "--oof-folds 1 cannot score anything out of fold: with one fold the "
            "fit set and the scored set are the same molecules, so every "
            "training molecule would be scored by a model that fitted its own "
            "corrupted label. That measures memorisation, not uncertainty. Use "
            "0 to turn cross-fitting off, or 2 or more to turn it on.")
    if args.oof_folds < 0:
        parser.error("--oof-folds cannot be negative.")
    if args.oof_outer_folds is not None and args.oof_outer_folds < 1:
        parser.error("--oof-outer-folds must be at least 1.")
    return args


def main():
    parser = build_parser()
    args = validate_args(parser, parser.parse_args())

    ds_list = ['logd', 'caco2', 'herg_ki'] if 'all' in args.datasets else args.datasets
    # None means: use each dataset's own anchored grid (NOISE_LEVELS_BY_DATASET).
    sigma_levels = args.sigmas if args.sigmas else None

    print("=" * 80)
    print("Validation Noise Robustness Experiment")
    print("=" * 80)
    print(f"  Datasets: {ds_list}")
    print(f"  Split: {N_FOLDS}-fold scaffold CV")
    rep_list = ", ".join(args.reps) if args.reps else ", ".join(ALL_REPS)
    print(f"  Representations: {rep_list}")
    print(f"  Noise conditions: {args.conditions if args.conditions else NOISE_CONDITIONS}")
    print(f"  Noise levels: {sigma_levels if sigma_levels else 'per dataset, anchored to assay error'}")
    model_list = ("RF, QRF, XGBoost, NGBoost, LightGBM, SVM, GP(PDV), "
                  "DNN, VBLL-Full, MLP, MLP-VBLL-Full")
    if HAS_BAYESIAN_TORCH:
        model_list += ", BNN-Full, MLP-BNN-Full"
    if args.models:
        model_list = ", ".join(args.models)
    print(f"  Models: {model_list}")
    print(f"  Uncertainty saved for conditions: {args.unc_conditions}")
    print(f"  Out-of-fold training uncertainty: "
          f"{args.oof_folds if args.oof_folds > 1 else 'off'}"
          f"{' folds, models: ' + ','.join(UNCERTAINTY_MODELS) if args.oof_folds > 1 else ''}"
          f"{'' if not args.oof_folds or args.oof_outer_folds is None else f', on the first {args.oof_outer_folds} of {N_FOLDS} scaffold folds'}")
    print()

    all_summaries = []

    # Load OpenADMET data if needed
    openadmet_df = None
    if 'logd' in ds_list or 'caco2' in ds_list:
        print("Loading OpenADMET-ExpansionRx dataset...")
        openadmet_df = download_openadmet(csv_path=args.openadmet_csv)
        print(f"  Shape: {openadmet_df.shape}")

    # 1. OpenADMET-LogD
    if 'logd' in ds_list and openadmet_df is not None:
        logd_col = next((c for c in openadmet_df.columns if 'LogD' in c), None)
        if logd_col:
            print(f"\nPreparing OpenADMET-LogD (column: {logd_col})...")
            smiles, labels = load_openadmet_endpoint(openadmet_df, logd_col, log_transform=False)
            print(f"  {len(smiles)} molecules")
            summary = run_dataset('OpenADMET-LogD', smiles, labels,
                                  Path(args.results_root) / 'logd',
                                  model_filter=args.models, rep_filter=args.reps,
                                  sigma_levels=sigma_levels,
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                                  unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                                  conditions=args.conditions,
                                  oof_outer_folds=args.oof_outer_folds,
                                  noise_validation=not args.no_noise_validation,)
            all_summaries.append(summary)
        else:
            print("ERROR: Cannot find LogD column")

    # 2. OpenADMET-Caco2_Efflux
    if 'caco2' in ds_list and openadmet_df is not None:
        caco2_col = next((c for c in openadmet_df.columns if 'Caco' in c and 'Efflux' in c), None)
        if caco2_col:
            print(f"\nPreparing OpenADMET-Caco2_Efflux (column: {caco2_col})...")
            smiles, labels = load_openadmet_endpoint(openadmet_df, caco2_col, log_transform=True)
            print(f"  {len(smiles)} molecules")
            summary = run_dataset('OpenADMET-Caco2_Efflux', smiles, labels,
                                  Path(args.results_root) / 'caco2',
                                  model_filter=args.models, rep_filter=args.reps,
                                  sigma_levels=sigma_levels,
                                  gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                                  unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                                  conditions=args.conditions,
                                  oof_outer_folds=args.oof_outer_folds,
                                  noise_validation=not args.no_noise_validation,)
            all_summaries.append(summary)
        else:
            print("ERROR: Cannot find Caco2 Efflux column")

    # 3. ChEMBL-hERG-Ki
    if 'herg_ki' in ds_list:
        print("\nPreparing ChEMBL-hERG-Ki...")
        smiles, labels = load_chembl_herg()
        print(f"  {len(smiles)} molecules")
        summary = run_dataset('ChEMBL-hERG-Ki', smiles, labels,
                              Path(args.results_root) / 'herg',
                              model_filter=args.models, rep_filter=args.reps,
                              sigma_levels=sigma_levels,   # was omitted: --sigmas was ignored for hERG
                              gp_kernel=args.gp_kernel, gp_reps=args.gp_reps,
                              unc_conditions=args.unc_conditions, oof_folds=args.oof_folds,
                              conditions=args.conditions,
                              oof_outer_folds=args.oof_outer_folds,
                              noise_validation=not args.no_noise_validation,)
        all_summaries.append(summary)

    # Cross-dataset summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)

        # Merge with existing combined_summary if running a subset
        combined_path = Path(args.results_root) / 'combined_summary.csv'
        if (args.models or args.reps) and combined_path.exists():
            existing = pd.read_csv(combined_path)
            if args.models and args.reps:
                mask = existing['model'].isin(args.models) & existing['rep'].isin(args.reps)
            elif args.models:
                mask = existing['model'].isin(args.models)
            else:
                mask = existing['rep'].isin(args.reps)
            existing = existing[~mask]
            combined = pd.concat([existing, combined], ignore_index=True)

        combined.to_csv(combined_path, index=False)

        print("\n" + "=" * 80)
        print("CROSS-DATASET SUMMARY")
        print("=" * 80)

        for ds in combined['dataset'].unique():
            ds_data = combined[combined['dataset'] == ds]
            working = ds_data[ds_data['baseline_r2'] >= 0.3]
            print(f"\n  {ds}: {len(working)}/{len(ds_data)} configs pass baseline >= 0.3")

    print("\n" + "=" * 80)
    print(f"COMPLETE — Results saved to {args.results_root}")
    print("=" * 80)


if __name__ == '__main__':
    main()
