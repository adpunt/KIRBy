"""Shared configuration and runner for federated experiments.

Bridges FLuID (Lhasa Limited) and KIRBy for the federated experiment category.
FLuID uses RDKit Mol objects + DataFrames; KIRBy uses SMILES + numpy arrays.

Privacy principles:
    - No silo shares raw data with other silos or the coordinator
    - Silos may share: predictions, feature indices, importance weights, validation scores
    - The "naive pool" condition is explicitly labeled as privacy-violating baseline

Config: RF / LightGBM / MLP, 1-fold scaffold CV, fast search.
Rep lists reused from complexity_helpers.DATASET_REPS.
"""

import json
import sys
import os
import time
import warnings
from pathlib import Path

# Ensure imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))         # tests/experiments/

import numpy as np
import pandas as pd
from experiments._shared import make_nearest_neighbors  # honours --distance-metric
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score, f1_score,
    accuracy_score,
)

from experiments._shared import (
    load_data, save_experiment_results, convert_numpy,
    DOWNSTREAM_MODELS, INTERNAL_EVALUATORS,
    input_fingerprint,
)
from experiments.complexity.complexity_helpers import DATASET_REPS

# ── FLuID import ─────────────────────────────────────────────────────────────
# FLuID calls IPython display() and Plotly showPlot() inside its functions.
# Monkey-patch before import to suppress in non-notebook environments.

import builtins
_original_display = getattr(builtins, 'display', None)
builtins.display = lambda *a, **kw: None

_fluid_default = '/Users/apunt/repos/lhasa/FLuID_POC/preprocessing'
_fluid_server_candidates = [
    '/data/stat-ecr/scat9264/FLuID_POC/preprocessing',  # current ARC host
    '/data/stat-cadd/scat9264/FLuID_POC/preprocessing',  # legacy
]
for _candidate in _fluid_server_candidates:
    if os.path.isdir(_candidate):
        _fluid_default = _candidate
        break
_fluid_path = os.environ.get('FLUID_PATH', _fluid_default)
sys.path.insert(0, _fluid_path)

# FLuID imports rdkit.Chem.PandasTools, whose PandasPatcher reads
# pandas.io.formats.format.get_adjustment. pandas 2.2 (constraints.txt pins
# 2.2.0) moved it to pandas.io.formats.printing, so the import died on the
# server with AttributeError, 2026-09-26. Point the old name at the new one
# when it is missing; nothing else reads it.
try:
    import pandas.io.formats.format as _pd_fmt
    if not hasattr(_pd_fmt, 'get_adjustment'):
        from pandas.io.formats.printing import get_adjustment as _ga
        _pd_fmt.get_adjustment = _ga
except Exception:
    pass

try:
    import FLuID as fluid
    # Patch FLuID's module-level display binding (imported from IPython)
    fluid.display = lambda *a, **kw: None
    fluid.showPlot = lambda *a, **kw: None
    _FLUID_AVAILABLE = True
except ImportError:
    _FLUID_AVAILABLE = False
    warnings.warn("FLuID not available — federated experiments will fail.")

# Restore display if it existed before; otherwise remove the placeholder so
# we don't leave a no-op shim bound globally.
if _original_display is not None:
    builtins.display = _original_display
elif hasattr(builtins, 'display') and getattr(builtins, 'display') is not _original_display:
    try:
        delattr(builtins, 'display')
    except AttributeError:
        pass


# ── Lhasa hERG loader (returns data in load_data format) ─────────────────────

def load_lhasa_herg_data(rep_names=None):
    """Load Lhasa's curated hERG training data in the same format as load_data().

    Returns a data dict compatible with fed01/02/04 with:
    - Binary labels (0/1) already binarized
    - task_type='classification'
    - Scaffold groups for CV splitting
    """
    from experiments.federated.lhasa_herg_loader import load_lhasa_herg_training

    smiles, labels, inchikeys, _ = load_lhasa_herg_training()
    print(f"  Lhasa hERG: {len(labels)} molecules, "
          f"{int(labels.sum())} active, {len(labels) - int(labels.sum())} inactive")

    # Compute representations (same path as fed03)
    from find_best_hybrid_molecular import (
        get_representations as get_molecular_representations,
        assign_scaffold_groups,
    )
    # get_molecular_representations expects a preset name OR a comma-separated
    # string. run_all hands us None (no override) or a Python list (explicit
    # --reps). Normalize: None/empty -> 'all'; a list -> comma-joined string.
    # Without this, an empty list ([]) silently produced ZERO reps and a list
    # override raised AttributeError on .split — both real bugs.
    if rep_names is None or (isinstance(rep_names, (list, tuple))
                             and len(rep_names) == 0):
        rep_names = 'all'
    elif isinstance(rep_names, (list, tuple)):
        rep_names = ','.join(rep_names)
    reps = get_molecular_representations(smiles, rep_names)

    # Scaffold groups
    try:
        groups, _ = assign_scaffold_groups(smiles)
    except Exception:
        groups = None

    data = {
        'reps': reps,
        'labels': labels,
        'smiles': smiles,
        'groups': groups,
        'task_type': 'classification',
        'domain': 'molecular',
        'is_binary': True,  # Signal to experiments: labels already binarized
        'augmentations': None,
    }
    # Honor the test-only KIRBY_SMOKE_N row cap (no-op when env var unset)
    from experiments._shared import _smoke_subsample
    return _smoke_subsample(data)


# ── Smoke-test gating (test-only) ────────────────────────────────────────────
# When KIRBY_SMOKE_N is set, the federated suite shrinks its fan-out (folds,
# budgets, downstream models, noise/sigma/smooth/detector grids) so it runs in
# minutes on tiny local subsets. ALL of this is a no-op when the env var is
# unset, so normal/server runs use the full grids unchanged.

def _row_cap() -> int:
    """The KIRBY_SMOKE_N row cap, or 0 when there is none.

    KIRBY_SMOKE_N throws rows away. It is the one knob that changes what the
    experiment is measured on, so it is kept separate from the knob that
    shrinks the SEARCH below.
    """
    raw = os.environ.get('KIRBY_SMOKE_N', '').strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _smoke_active() -> bool:
    """True when the SEARCH GRID should be shrunk — not the row count.

    Two different things used to hang off KIRBY_SMOKE_N: how many molecules
    the experiment sees, and how big a grid the hybrid search covers. Setting
    it to run locally therefore threw the rows away as well, and a local run
    on 200 molecules answers nothing. KIRBY_SMOKE_GRID shrinks the search and
    leaves every row in place; KIRBY_SMOKE_N still caps the rows, and setting
    it implies the grid shrink too.
    """
    if os.environ.get('KIRBY_SMOKE_GRID', '').strip() not in ('', '0'):
        return True
    return _row_cap() > 0


def smoke_grid(full_values, smoke_values):
    """Shrink a per-experiment axis — only when the rows are capped too.

    These are the experiment's OWN axes: fed01's six flip fractions, its eight
    detectors, fed04's five smooth factors. They are what the experiment
    sweeps, not the space the hybrid search covers, and collapsing one of them
    changes the question rather than the cost of answering it. So they shrink
    only under KIRBY_SMOKE_N, where the rows are already gone and nothing is
    being answered anyway. Under KIRBY_SMOKE_GRID — every row present, a
    narrowed search — the full axis runs.
    """
    return list(smoke_values) if _row_cap() > 0 else list(full_values)


def smoke_n_folds(full_n: int) -> int:
    """How many cross-validation folds a smoke run uses, else full_n.

    Two folds at 200 capped rows; one further fold per 200 rows after that,
    up to the full count. A flat 2 leaves every spread in the result file
    computed from two numbers, which is as much a reason to raise the cap as
    to lower it.
    """
    # KIRBY_FED_FOLDS sets the fold count on its own, with every row present.
    # Cutting folds and capping rows are different things: a 3-fold run on all
    # 13,231 molecules is a coarser estimate of the same quantity, where a
    # 200-row run is a different quantity. This switch only does the first.
    raw = os.environ.get('KIRBY_FED_FOLDS', '').strip()
    if raw:
        return max(2, min(full_n, int(raw)))
    n = _row_cap()
    if n == 0:
        # Every row is present, so there is no reason to cut the folds.
        return full_n
    return max(2, min(full_n, n // 200))


def smoke_search_kwargs(config=None):
    """Test-only: narrow a get_best_search_for_downstream / find_best_hybrid call.

    Every federated file asks the engine for a recipe, and the engine defaults
    to scoring all five DOWNSTREAM_MODELS against all five internal
    evaluators. The federated sweep only ever reads the row for the family it
    is running — rf, xgb or dnn — so the other four models are scored and
    discarded. Under KIRBY_SMOKE_N this returns the two keyword arguments that
    cut the search to the families the sweep will ask for and to the smoke
    evaluator list; it returns an empty dict when the env var is unset, so a
    server run is unchanged.
    """
    if not _smoke_active():
        return {}
    cfg = config or {}
    models = sorted({fed_to_kirby_model(m)
                     for m in cfg.get('fed_models', FED_MODELS)})
    out = {'models': models}
    evals = cfg.get('internal_evaluators')
    if evals:
        out['internal_evaluators'] = list(evals)
    return out


def federated_budgets(default=(50, 100, 250, 500, 1000)):
    """The feature budgets an experiment sweeps.

    Four of the five federated files pinned their budget to 100 and never
    varied it, so every answer they gave was an answer about 100 features
    rather than about the method. fed02's KIRBy arms pick 100 features and
    are measured against models using whole representations of 2,048, 1,613
    and 768 columns, which makes width the obvious explanation for a loss
    nobody had tested.

    KIRBY_FED_BUDGETS overrides the list, comma separated. A server run takes
    the default and covers the range; a local debugging run sets it to one
    width so it finishes.
    """
    raw = os.environ.get('KIRBY_FED_BUDGETS', '').strip()
    if not raw:
        return list(default)
    return [int(x) for x in raw.split(',') if x.strip()]


def smoke_k_silos(full_k: int) -> int:
    """How many silos/clusters a smoke run splits into, else full_k.

    A silo has to clear fed04's 20-sample minimum or its importance profile
    is skipped as 'too few samples', and a smoke run has already thrown most
    of the rows away. This used to be a flat 2 whatever KIRBY_SMOKE_N said,
    which is right at 200 rows and wrong at 600: with two silos the
    flip-fraction linspace has two points, so fed01's Spearman between a
    silo's flip fraction and its share of the feature budget is +1 or -1 by
    construction and says nothing.

    One silo per 100 capped rows, never fewer than 2 and never more than the
    full count: 2 silos at 200 rows, 5 at 500 and above.
    """
    n = _row_cap()
    if n == 0:
        # Every row is present, so every silo clears the 20-sample minimum.
        return full_k
    return max(2, min(full_k, n // 100))


# ── Federated-specific config ────────────────────────────────────────────────

_FEDERATED_CONFIG_CACHE = None


def _get_federated_config():
    global _FEDERATED_CONFIG_CACHE
    if _FEDERATED_CONFIG_CACHE is None:
        from kirby.find_best_hybrid import (
            DEFAULT_ALLOCATION_METHODS, DEFAULT_IMPORTANCE_METHODS,
            DEFAULT_N_FOLDS,
        )
        if _smoke_active():
            # Test-only: shrink folds/budgets/models so the heavy hybrid
            # search fans out minimally. Gated — server runs never see this.
            _FEDERATED_CONFIG_CACHE = {
                'n_splits': 2,
                'allocations': DEFAULT_ALLOCATION_METHODS,
                'importance_methods': DEFAULT_IMPORTANCE_METHODS,
                'internal_evaluators': ['match', 'rf'],
                'budgets': [50],
                'models': ['rf', 'knn'],
                # Federated downstream sweep (test-only: 1 model, 1 seed → fast).
                'fed_models': ['rf'],
                'fed_seeds': [0],
            }
        else:
            _FEDERATED_CONFIG_CACHE = {
                'n_splits': DEFAULT_N_FOLDS,
                'allocations': DEFAULT_ALLOCATION_METHODS,
                'importance_methods': DEFAULT_IMPORTANCE_METHODS,
                'internal_evaluators': INTERNAL_EVALUATORS,
                'budgets': [100],
                'models': DOWNSTREAM_MODELS,
                # Federated downstream sweep: FLuID's RF+XGBoost+MLP families,
                # 3 seeds for error bars (silos stay at 5). Drives run_all's
                # sweep; experiments read config['fed_model']/['fed_seed'].
                'fed_models': FED_MODELS,
                'fed_seeds': FED_SEEDS,
            }
    return _FEDERATED_CONFIG_CACHE


# ── Turning a continuous endpoint into two classes ───────────────────────────
# fed01 and fed04 both ask a two-class question, and both run on datasets whose
# labels are continuous. Until 2026-09-04 each carried its own constant, pKi >=
# 6.0, and applied it to every dataset. caco2's log10 efflux ratio tops out at
# 1.678 and logd's LogD at 4.500, so on both of those every row landed in one
# class: fed04 raised inside RandomUnderSampler and wrote nothing, fed01 ran a
# single-class problem end to end.
#
# {dataset_name: (cutoff, source)} — a row's class is 1 when label >= cutoff.
# A dataset with no row here is refused by name, not silently thresholded.
#
# Each cutoff below is a published line on the same quantity the label
# measures, and the class counts beside it were measured on the loaded dataset
# on 2026-09-12. A cutoff borrowed from a neighbouring quantity is not used:
# the lipophilicity rules of thumb that are quoted most often (Hughes 2008's
# 3/75, Gleeson 2008's 4/400) are thresholds on ClogP and logP, and the logd
# label here is logD at pH 7.4, which is a different number for any molecule
# that ionises.
CLASSIFICATION_CUTOFFS = {
    # ChEMBL hERG Ki as pKi. pKi >= 6.0 is Ki <= 1 uM, the stricter of the two
    # conventions in the hERG classification literature; the other is 10 uM,
    # pKi >= 5.0. Measured on 1,412 molecules: pKi >= 6.0 gives 467 blockers
    # against 945 non-blockers (33.1% blockers), pKi >= 5.0 gives 1,059
    # against 353 (75.0%). Label range 4.070 to 9.410, median 5.580.
    # One caveat worth carrying: the 1 uM and 10 uM conventions are quoted for
    # IC50 from patch-clamp, and this label is Ki.
    'herg_ki': (6.0, 'pKi >= 6.0 (Ki <= 1 uM)'),
    # log10 of the Caco-2 A-to-B / B-to-A efflux ratio. A net efflux ratio of
    # 2 or more flags an in vitro P-glycoprotein substrate — FDA, In Vitro
    # Drug Interaction Studies guidance for industry (draft 2017, final
    # January 2020). Measured on 2,158 molecules: 756 substrates against
    # 1,402 non-substrates (35.0% substrates). Label range -0.319 to 1.679,
    # median 0.143.
    'caco2': (0.30103, 'log10(efflux ratio) >= log10(2) — P-gp substrate'),
    # logD at pH 7.4. Waring, 'Lipophilicity in drug discovery', Expert
    # Opinion on Drug Discovery 5(3):235-248, 2010: attrition analysis puts
    # the optimal logD7.4 for a small-molecule drug between 1 and 3, and 3 is
    # its upper edge — above it, ADMET liabilities and promiscuity rise. This
    # is a threshold on logD7.4 itself, not on ClogP. Measured on 5,028
    # molecules: 1,242 above against 3,786 below (24.7% above). Label range
    # -1.000 to 4.500, median 2.100.
    'logd': (3.0, 'logD(7.4) >= 3 — above Waring 2010 optimal range'),
}


def binarise_labels(dataset_name, labels, is_binary=False):
    """Two classes for a federated experiment, in the dataset's own label units.

    Returns (labels_binary, note). Raises when the dataset has no cutoff, and
    when the cutoff it has puts every row in one class.
    """
    labels = np.asarray(labels)
    uniq = set(np.unique(labels).tolist())
    if is_binary or uniq <= {0, 1, 0.0, 1.0}:
        return labels.astype(int), 'labels already binary'

    entry = CLASSIFICATION_CUTOFFS.get(dataset_name)
    if entry is None:
        raise ValueError(
            f"No classification cutoff defined for '{dataset_name}'. Its "
            f"labels are continuous and this experiment needs two classes. "
            f"Add a (cutoff, source) row for it to CLASSIFICATION_CUTOFFS in "
            f"federated_helpers.py — in that dataset's own label units — "
            f"rather than borrowing another endpoint's.")
    cutoff, note = entry
    labels_binary = (labels >= cutoff).astype(int)
    n_active = int(labels_binary.sum())
    if n_active == 0 or n_active == len(labels_binary):
        raise ValueError(
            f"The cutoff for '{dataset_name}' ({note}) puts all "
            f"{len(labels_binary)} rows in one class. Label range is "
            f"[{float(np.min(labels)):.4f}, {float(np.max(labels)):.4f}].")
    return labels_binary, note


# ── Experiment list ──────────────────────────────────────────────────────────

SINGLE_DATASET_EXPERIMENTS = [
    'fed01',   # noise heterogeneity across silos + federated noise detection
    'fed02',   # rep diversity as federated silos
    'fed03',   # synthetic private datasets + antibody federated
    'fed04',   # cluster-aware importance
    'fed05',   # antibody federated extension
]

# Experiments that load their own silo data internally — run_all skips
# load_data for these and passes a placeholder data dict instead.
SELF_LOADING_EXPERIMENTS = {
    'fed05',   # loads 3 antibody silo datasets internally
}


# =============================================================================
# MODEL FACTORY (mirrors Lhasa FLuID — keep in lock-step with FLuID.py)
# =============================================================================
# These mirror FLuID's own model construction so every federated condition uses
# the SAME downstream models FLuID uses, and so the result is comparable to the
# published FLuID numbers (Nature Mach. Intell. 2025; MCC primary metric).
#
# Reference (Lhasa FLuID_POC/preprocessing/FLuID.py):
#   - create_classifier()            FLuID.py:85-103  (hyperparameters copied verbatim)
#   - create_trained_classifier()    FLuID.py:125-147 (RandomUnderSampler then fit)
# The ONLY thing we change is random_state, which is swept over FED_SEEDS to get
# error bars (the published wave had a single seed → no variance). Silos stay 5.

# FLuID's algorithm NAMES (must match FLuID.create_classifier exactly):
# 'rf' + 'xgb' are FLuID's internal/teacher families, and the student was
# FLuID's 'mlp'.
# (FLuID also exposes nb/svm/svml/knn/gp/dt/ab — not used here.)
#
# ⛔ THE STUDENT IS dnn SINCE 2026-09-18, OWNER'S DECISION. mlp was cut from
# the suite: it did not perform and it was slow. It also never had a recipe to
# read — mlp was never scored by any search, so all five federated files fell
# to a hardcoded recipe on the mlp third of every sweep and did not record it.
# dnn is scored by every search, so that third now reads a real recipe.
FED_MODELS = ['rf', 'xgb', 'dnn']
# 3-seed sweep for error bars (user decision 2026-06-04). Silos stay at 5.
FED_SEEDS = [0, 1, 2]

# FLuID and KIRBy use DIFFERENT names for the same model: FLuID says 'xgb'
# (FLuID.create_classifier), KIRBy's own DOWNSTREAM_MODELS says 'xgboost'. Use
# FLuID names on the FLuID teacher/student side, map to KIRBy names whenever a
# swept model is handed to a KIRBy call (get_best_search_for_downstream,
# federated_feature_selection, downstream_model=...).
_FED_TO_KIRBY_MODEL = {'rf': 'rf', 'xgb': 'xgboost', 'dnn': 'dnn'}


def fed_to_kirby_model(fed_model):
    """Translate a FLuID model name to KIRBy's downstream_model name."""
    return _FED_TO_KIRBY_MODEL.get(fed_model, fed_model)


def aggregate_seed_runs(run_dicts):
    """Recursively mean/std numeric leaves across a list of per-seed result
    dicts (the 3-seed sweep → error bars). Non-numeric leaves take seed 0's
    value. Numeric leaves become {'mean','std','n_seeds','per_seed'}. Missing
    keys in some seeds are skipped. Mirrors the per-fold _aggregate already used
    inside the experiments, lifted to the seed axis."""
    run_dicts = [r for r in run_dicts if r is not None]
    if not run_dicts:
        return {}
    first = run_dicts[0]
    if isinstance(first, dict):
        keys = set()
        for r in run_dicts:
            if isinstance(r, dict):
                keys.update(r.keys())
        out = {}
        for k in keys:
            out[k] = aggregate_seed_runs(
                [r[k] for r in run_dicts if isinstance(r, dict) and k in r])
        return out
    numeric = [float(v) for v in run_dicts
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and not (isinstance(v, float) and np.isnan(v))]
    non_bool = [v for v in run_dicts if not isinstance(v, bool)]
    if numeric and non_bool and all(
            isinstance(v, (int, float)) for v in non_bool):
        arr = np.asarray(numeric, dtype=float)
        return {'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr, ddof=0)),
                'n_seeds': int(len(arr)),
                'per_seed': numeric}
    return first  # non-numeric (strings, configs) — keep seed 0's value


def _model_n_jobs():
    """How many cores one forest may use.

    sklearn defaults to one. The FLuID quality ensemble fits five folds per
    silo on whole representations, so a fed02 fold spent hours in forests that
    were each on a single core. A forest with a fixed random_state builds the
    same trees whatever this is, so raising it changes the wall clock and
    nothing else. KIRBY_LGBM_THREADS is the knob the repo already uses for
    this; it defaults to 1 here so a SLURM job keeps the cores it was given.
    """
    try:
        return int(os.environ.get('KIRBY_LGBM_THREADS', '1'))
    except ValueError:
        return 1


def make_fed_classifier(algorithm, seed=0):
    """Return an UNFITTED classifier sourced DIRECTLY from Lhasa's
    FLuID.create_classifier (FLuID.py:85-107) — no re-declared hyperparameters,
    so it can never drift from FLuID — with random_state overridden to `seed`
    for the seed sweep. `algorithm` uses FLuID's names for the two teachers
    ('rf','xgb') and KIRBy's name for the student ('dnn').

    Requires FLuID importable; the whole federated suite already does.
    """
    if not _FLUID_AVAILABLE:
        raise RuntimeError(
            "FLuID not importable — make_fed_classifier sources its models "
            "from FLuID.create_classifier.")
    clf = fluid.create_classifier(algorithm)
    # FLuID's hyperparameters stay exactly as FLuID declared them. n_jobs is
    # not one of them; it is how many cores the fit may use.
    if hasattr(clf, 'n_jobs'):
        clf.n_jobs = _model_n_jobs()
    try:
        clf.set_params(random_state=seed)
    except (ValueError, TypeError):
        pass  # nb/knn etc. have no random_state — leave as FLuID defines it
    return clf


def fit_fed_classifier(algorithm, X, y, seed=0, balance=True):
    """Train a classifier exactly as FLuID.create_trained_classifier does
    (FLuID.py:125-147): RandomUnderSampler(random_state=seed) balancing, then
    fit. The model itself comes from FLuID.create_classifier (via
    make_fed_classifier), so it stays in lock-step with Lhasa's code. This
    replaces KIRBy's old class_weight-only path (a documented deviation); per
    the 2026-06-04 decision all three models now use FLuID's actual
    undersampling (the MLP cannot be class-weighted at all)."""
    X = np.asarray(X)
    y = np.asarray(y)
    if balance:
        from imblearn.under_sampling import RandomUnderSampler
        X, y = RandomUnderSampler(random_state=seed).fit_resample(X, y)
    clf = make_fed_classifier(algorithm, seed=seed)
    clf.fit(X, y)
    return clf


def make_fed_regressor(algorithm, seed=0):
    """Regressor counterpart of make_fed_classifier. FLuID is CLASSIFICATION-
    ONLY (no regressor exists in FLuID.create_classifier), so for the federated
    REGRESSION silos these mirror FLuID's architecture/hyperparameters in
    regression form — a KIRBy extension (user decision 2026-06-04: sweep
    regression too), kept as close to FLuID's settings as possible:
    RF 150 trees/depth 15; XGBoost default depth; MLP alpha=1/(40,40)/1000 iter.
    """
    if algorithm == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=seed,
            n_jobs=_model_n_jobs())
    if algorithm == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(objective='reg:squarederror', random_state=seed)
    if algorithm == 'dnn':
        # Was FLuID's MLPRegressor(alpha=1, hidden_layer_sizes=(40, 40),
        # max_iter=1000) until 2026-09-18. It is now KIRBy's own dnn, the same
        # estimator every search scores, so the recipe this arm reads and the
        # model it fits are the same family. Settings are dnn's own defaults
        # (src/kirby/models/dnn.py): (256, 128, 64), dropout 0.2, lr 1e-3,
        # 200 epochs, batch 256, patience 20.
        from kirby.models.dnn import DNNRegressor
        return DNNRegressor(random_state=seed)
    raise ValueError(
        f"Unknown federated model '{algorithm}'. Expected one of {FED_MODELS}.")


def fit_fed_regressor(algorithm, X, y, seed=0):
    """Fit a regressor for a federated regression silo. No class balancing
    (regression has no classes → no RandomUnderSampler)."""
    reg = make_fed_regressor(algorithm, seed=seed)
    reg.fit(np.asarray(X), np.asarray(y))
    return reg


def fit_fed_model(algorithm, X, y, seed=0, task_type='classification',
                  balance=True):
    """Single entry point: dispatch to the FLuID-sourced classifier (with
    undersampling) or the regression-mirror regressor, by task_type."""
    if task_type == 'regression':
        return fit_fed_regressor(algorithm, X, y, seed=seed)
    return fit_fed_classifier(algorithm, X, y, seed=seed, balance=balance)


# =============================================================================
# FORMAT BRIDGE: KIRBy <-> FLuID
# =============================================================================

def smiles_to_fluid_df(smiles_list, labels, pki_threshold, source='KIRBy'):
    """Convert KIRBy SMILES + continuous labels into FLuID-format DataFrame.

    Args:
        smiles_list: List of SMILES strings.
        labels: Continuous labels (e.g. pKi values), or labels already 0/1.
        pki_threshold: Where to cut continuous labels — a row at or above it
            is Active. REQUIRED, with no default: a default of 6.0 lived here
            and was a second copy of the hERG cut, so a caller that forgot to
            pass one got a pKi threshold applied to whatever its label was.
            Every dataset's cut belongs in CLASSIFICATION_CUTOFFS; pass
            binarise_labels' output and any float here (it is unused), or pass
            the cut itself. Ignored when the labels are already 0/1.
        source: Source name for the SOURCE column.

    Returns:
        (fluid_df, valid_mask): DataFrame with FLuID columns and boolean mask
        of valid molecules (some SMILES may fail RDKit parsing).
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    mols = []
    fps = []
    inchis = []
    valid_mask = np.ones(len(smiles_list), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask[i] = False
            mols.append(None)
            fps.append(None)
            inchis.append(None)
            continue
        mols.append(mol)
        fps.append(fluid.compute_fp_array(mol))
        try:
            inchi = MolToInchi(mol)
            inchis.append(InchiToInchiKey(inchi) if inchi else 'UNKNOWN')
        except Exception:
            inchis.append('UNKNOWN')

    # Filter to valid only
    valid_idx = np.where(valid_mask)[0]
    labels_valid = np.asarray(labels)[valid_idx]

    # If labels are already binary (0/1), skip thresholding
    unique_vals = set(np.unique(labels_valid))
    if unique_vals <= {0, 1, 0.0, 1.0}:
        binary_labels = labels_valid.astype(int)
    else:
        binary_labels = (labels_valid >= pki_threshold).astype(int)
    activity_strs = np.where(binary_labels == 1, 'Active', 'Inactive')

    df = pd.DataFrame({
        'MOLECULE': [mols[i] for i in valid_idx],
        'FP': [fps[i] for i in valid_idx],
        'CLASS': binary_labels,
        'ACTIVITY': activity_strs,
        'INCHI': [inchis[i] for i in valid_idx],
        'SOURCE': source,
        'PKI': labels_valid,
    })
    df.index = range(len(df))

    return df, valid_mask


def cluster_into_silos(smiles_list, labels, pki_threshold, k=5,
                       smooth_factor=0.05, seed=42):
    """Cluster molecules into K silos using FLuID's K-Means on Morgan FPs.

    Args:
        smiles_list: List of SMILES strings.
        labels: Continuous labels (e.g. pKi values).
        k: Number of clusters (silos).
        smooth_factor: Fraction of samples to randomly reassign across clusters.
        pki_threshold: Where to cut continuous labels — see
            smiles_to_fluid_df. Required, no default.
        seed: Random seed for reproducibility.

    Returns:
        (silo_indices, fluid_df, cluster_data, valid_mask):
        - silo_indices: {silo_id: np.array of row indices in fluid_df}
          silo 0 = all samples, silos 1..k = clusters
        - fluid_df: FLuID-format DataFrame
        - cluster_data: list of DataFrames [full, cluster_1, ..., cluster_k]
        - valid_mask: boolean mask of valid molecules
    """
    fluid_df, valid_mask = smiles_to_fluid_df(
        smiles_list, labels, pki_threshold=pki_threshold)

    params = {
        'k': k,
        'smooth_factor': smooth_factor,
        'random_state': seed,
        # FLuID's cluster_data_space accesses these for plotting (even though
        # we patched showPlot, fig.update_layout runs before showPlot):
        'figure_font': dict(family="Arial", size=12),
    }

    # FLuID's cluster_data_space modifies df in place, adds CLUSTER/CLUSTER_ID
    cluster_data = fluid.cluster_data_space(
        fluid_df, 'Teacher', 'T', k, smooth_factor, params)

    # Extract index mappings
    silo_indices = {0: np.arange(len(fluid_df))}
    for s in range(1, k + 1):
        silo_indices[s] = np.array(
            cluster_data[s].index.tolist(), dtype=int)

    return silo_indices, fluid_df, cluster_data, valid_mask


def cluster_training_only(train_fluid_df, k=5, smooth_factor=0.05, seed=42):
    """Cluster TRAINING molecules only, matching FLuID's pattern.

    FLuID clusters trainingData only — test set never touches K-Means.
    This function takes a pre-split training DataFrame and clusters it.

    Args:
        train_fluid_df: FLuID-format DataFrame of TRAINING molecules only.
            Must have 'FP' column (Morgan fingerprints).
        k: Number of clusters (silos).
        smooth_factor: Fraction of samples to randomly reassign.
        seed: Random seed.

    Returns:
        (silo_indices, cluster_data, kmeans_model):
        - silo_indices: {silo_id: np.array of row indices} (training-relative)
        - cluster_data: list of DataFrames [full_train, cluster_1, ..., cluster_k]
        - kmeans_model: fitted KMeans for assigning new molecules to clusters
    """
    if not _FLUID_AVAILABLE:
        raise RuntimeError("FLuID not available — cannot cluster")

    from sklearn.cluster import KMeans as _KMeans

    params = {
        'k': k,
        'smooth_factor': smooth_factor,
        'random_state': seed,
        'figure_font': dict(family="Arial", size=12),
    }

    # Reset index to 0..N_train-1 so cluster indices are training-relative
    train_df = train_fluid_df.reset_index(drop=True)

    # FLuID's cluster_data_space: K-Means + smooth_factor reassignment
    # Modifies train_df in place, adds CLUSTER/CLUSTER_ID columns
    try:
        cluster_data = fluid.cluster_data_space(
            train_df, 'Teacher', 'T', k, smooth_factor, params)
    except KeyError:
        # FLuID.py:542-548 draws a stacked bar chart of Active/Inactive counts
        # per cluster, and reads groups[(cluster, 'Active')] for every cluster.
        # A cluster holding no active molecules has no such key, so the lookup
        # raises and the function never returns — even though the clustering
        # finished at FLuID.py:538 and CLUSTER is already on train_df. Both
        # fed01 and fed04 died here on 2026-09-04, on 116 training rows over
        # two clusters. fluid.display and fluid.showPlot are already no-ops
        # (see the import block above); they cannot help, because the counts
        # are computed inside the go.Bar(...) argument, before showPlot runs.
        #
        # Rebuild what cluster_data_space would have returned: the whole
        # training set, then one frame per cluster. This is the same slicing
        # it does at FLuID.py:536, so the clustering stays FLuID's.
        if 'CLUSTER' not in train_df.columns:
            raise
        cluster_data = [train_df] + [
            train_df[train_df['CLUSTER'] == t] for t in range(1, k + 1)]
        # print, not warnings.warn: _shared.py:45 and find_best_hybrid.py:63
        # both call warnings.filterwarnings('ignore') at import, so nothing
        # raised through the warnings module is ever seen in these runs.
        print("  [cluster] FLuID's cluster chart raised (a cluster holds one "
              "class only). The clustering finished; cluster_data rebuilt "
              "from the CLUSTER column.")

    # Extract index mappings (training-relative)
    silo_indices = {}
    for s in range(1, k + 1):
        silo_indices[s] = np.array(
            cluster_data[s].index.tolist(), dtype=int)

    # Fit KMeans on training FPs for test molecule assignment.
    # FLuID's cluster_data_space applies smoothing AFTER K-Means, so the
    # CLUSTER column may differ from raw K-Means labels. We fit a fresh
    # K-Means on the SMOOTHED assignments to match.
    X_train = np.asarray(list(train_df.FP))
    smoothed_labels = np.asarray(train_df['CLUSTER']) - 1  # 0-indexed for sklearn
    kmeans_model = _KMeans(n_clusters=k, random_state=seed)
    kmeans_model.fit(X_train)
    # Override cluster_centers_ with centroids of smoothed assignments
    for c in range(k):
        mask = smoothed_labels == c
        if mask.any():
            kmeans_model.cluster_centers_[c] = X_train[mask].mean(axis=0)
    # FED-29: overriding cluster_centers_ leaves labels_/inertia_ from the raw
    # K-Means fit, which no longer match the smoothed centroids. Re-derive both
    # from the overridden centroids so any downstream read of .labels_/.inertia_
    # is consistent (fed04 only calls .predict(), but keep the object coherent).
    new_labels = kmeans_model.predict(X_train)
    kmeans_model.labels_ = new_labels
    _dist_sq = ((X_train - kmeans_model.cluster_centers_[new_labels]) ** 2).sum()
    kmeans_model.inertia_ = float(_dist_sq)

    return silo_indices, cluster_data, kmeans_model


# =============================================================================
# NOISE INJECTION (PER-SILO)
# =============================================================================

def inject_noise_per_silo(labels, silo_indices, noise_levels,
                          noise_type='label_flip', seed=42):
    """Inject heterogeneous noise across silos, one silo at a time.

    Every silo draws independently and is corrupted at its own level, which is the
    point of the experiment: silos differ in how good their data is.

    Injection goes through noiseInject, the same definition the noise suite, c10 and
    the hybrid search use. Two things that follow from that:

    label_flip is CLASSIFICATION and takes the `uniform` flip model. A flipped class
    label has no magnitude, so dose matching does not apply and the level stays what
    it always was -- the fraction of that silo's labels flipped.

    gaussian is REGRESSION and the level is now the noise DELIVERED, a root-mean-square
    as a fraction of that silo's clean label SD. It used to be a fraction of the silo's
    label RANGE, which is roughly 4-6x the SD, so a level of 0.2 delivered around a
    whole standard deviation of noise rather than a fifth of one.

    Args:
        labels: 1D array of binary labels (0/1) for label_flip,
                or continuous for gaussian.
        silo_indices: {silo_id: np.array of indices}. Use silos 1..k only.
        noise_levels: {silo_id: level} e.g. {1: 0.0, 2: 0.05, ..., 5: 0.3}.
        noise_type: 'label_flip' (binary) or 'gaussian' (continuous).
        seed: Random seed. Mixed with the silo id so no two silos share a draw.

    Returns:
        (labels_noisy, noise_mask_global, noise_mask_per_silo)
    """
    import zlib
    from kirby.noise_spec import NoiseSpec, make_noisy_labels, make_noisy_labels_verbose

    labels_noisy = labels.copy()
    noise_mask_global = np.zeros(len(labels), dtype=bool)
    noise_mask_per_silo = {}

    for silo_id, fraction in noise_levels.items():
        idx = silo_indices[silo_id]
        silo_mask = np.zeros(len(labels), dtype=bool)

        if fraction <= 0:
            noise_mask_per_silo[silo_id] = silo_mask
            continue

        # A fresh injector per silo, and a seed that depends on the silo: the
        # generator advances on every draw, so one injector shared across the silos
        # would make each silo's corruption depend on how many ran before it.
        silo_seed = (int(seed) * 1000003
                     + zlib.crc32(str(silo_id).encode('utf-8'))
                     + int(round(float(fraction) * 1000)) * 7919) & 0x7FFFFFFF
        silo_labels = labels[idx]

        if noise_type == 'label_flip':
            spec = NoiseSpec(strategy='uniform',
                             target_effective_noise=float(fraction),
                             noise_seed=silo_seed)
            flipped = make_noisy_labels(silo_labels, spec, is_clf=True)
            labels_noisy[idx] = flipped
            silo_mask[idx[np.asarray(flipped) != np.asarray(silo_labels)]] = True

        elif noise_type == 'gaussian':
            spec = NoiseSpec(strategy='gaussian',
                             target_effective_noise=float(fraction),
                             noise_seed=silo_seed, dose_units='spread')
            res = make_noisy_labels_verbose(silo_labels, spec)
            noise = np.asarray(res.epsilon)
            labels_noisy[idx] = np.asarray(res.y_noisy)
            # Ground truth mask: the rows this silo moved by more than its own
            # median absolute deviation. Gaussian perturbs every row, so "which
            # rows are corrupted" has no exact answer here -- this is a
            # magnitude cut, and it is why a targeted condition (outlier,
            # grouped_wider) answers the detection question and this one does not.
            silo_mad = float(np.median(np.abs(
                silo_labels - np.median(silo_labels))))
            if silo_mad < 1e-12:
                silo_mad = float(np.std(silo_labels)) or 1.0
            silo_mask[idx[np.abs(noise) > silo_mad]] = True

        else:
            raise ValueError(f"unknown noise_type {noise_type!r}; "
                             f"expected 'label_flip' or 'gaussian'")

        noise_mask_global |= silo_mask
        noise_mask_per_silo[silo_id] = silo_mask

    return labels_noisy, noise_mask_global, noise_mask_per_silo



# =============================================================================
# FLUID TEACHER ENSEMBLE (CONDITIONS 2-4)
# =============================================================================

def train_teachers_per_silo(fluid_df, cluster_data, noisy_labels, k,
                            algorithm='rf'):
    """Train one FLuID classifier per silo using noisy labels.

    Updates fluid_df CLASS/ACTIVITY in place, then wraps build_teacher_models.
    Privacy: each silo trains locally on its own data.

    Returns:
        teacher_models: list of (classifier, size, df) tuples, length k+1
    """
    # Update labels in the DataFrame
    fluid_df['CLASS'] = noisy_labels
    fluid_df['ACTIVITY'] = np.where(noisy_labels == 1, 'Active', 'Inactive')

    # Rebuild cluster_data views with updated labels
    cluster_data_updated = [fluid_df]  # Teacher 0 = full dataset
    for s in range(1, k + 1):
        cluster_data_updated.append(
            fluid_df[fluid_df['CLUSTER'] == s])

    params = {'k': k, 'teacher_algorithm': algorithm}
    teacher_models = fluid.build_teacher_models(cluster_data_updated, params)
    return teacher_models


def predict_fluid_similarity_ensemble(teacher_models, cluster_data_train,
                                      test_fps, k, seed=42):
    """FLuID's 8-NN similarity-weighted ensemble prediction.

    Reimplements the core math from FLuID.annotate_transfer_data() and
    federate_labels() without I/O, plotting, or pickle side effects.

    Privacy: each teacher runs predict locally, shares only predictions
    and similarity weights to coordinator.

    Args:
        teacher_models: list of (clf, size, df) tuples from training.
        cluster_data_train: list of TRAIN-ONLY DataFrames per cluster.
            Must NOT include test samples (to avoid data leakage in KNN).
        test_fps: Morgan FP array for test molecules.
        k: number of silos.
        seed: random seed.

    Returns:
        (pred_classes, pred_probs_positive, teacher_weights_per_sample)
    """
    n_test = len(test_fps)

    # Collect per-teacher predictions and weights
    all_probs = {}   # {teacher_id: (n_test, 2) probabilities}
    all_weights = {} # {teacher_id: (n_test,) similarity weights}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        teacher_df = cluster_data_train[t]

        # Teacher predictions on test set
        probs = clf.predict_proba(test_fps)  # (n_test, 2)
        all_probs[t] = probs

        # 8-NN distance weighting using TRAIN-ONLY teacher data
        teacher_fps = np.asarray(list(teacher_df.FP))
        # A silo with no training rows has nothing to measure a distance
        # from. max(1, int(0.2 * 0)) asks numpy for one sample out of an
        # empty array, which raises. The student pipeline above already
        # handles this by giving that teacher a flat weight; do the same
        # here so one empty cluster cannot end the whole run.
        if len(teacher_fps) == 0:
            all_weights[t] = np.ones(len(test_fps))
            continue
        n_sample = max(1, int(0.2 * len(teacher_fps)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(teacher_fps), size=n_sample, replace=False)
        teacher_fps_sample = teacher_fps[sample_idx]

        n_neighbors = min(8, len(teacher_fps_sample))
        # metric=None follows --distance-metric. FLuID's published
        # annotate_transfer_data measures this in Euclidean; pass
        # metric='euclidean' to pin it back for a replication run.
        knn = make_nearest_neighbors(n_neighbors=n_neighbors)
        knn.fit(teacher_fps_sample)
        distances, _ = knn.kneighbors(test_fps)
        all_weights[t] = distances.mean(axis=1)  # mean distance to 8-NN

    if not all_probs:
        # All teachers were None — return uniform predictions
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    # Distances -> similarities on ONE scale shared by every teacher, as
    # FLuID's annotate_transfer_data does (FLuID.py:925-940: minWeight and
    # maxWeight are taken over teachers 1..k together). Every teacher here
    # measures in the same Morgan space, so the scale is comparable.
    #
    # This used to min-max each teacher on its own. That erased the fact the
    # weight exists to carry -- which silo is nearer -- since every teacher's
    # nearest test molecule got 1 and its farthest got 0. It also gave a
    # molecule that is the farthest from every silo weight 0 from all of
    # them, and its prediction then came out as class 0 at probability 0
    # whatever the teachers said. The student pipeline below already used
    # the shared scale.
    active_teachers = sorted(all_probs.keys())
    _raw_all = np.concatenate([np.asarray(all_weights[t], dtype=float)
                               for t in active_teachers])
    min_w, max_w = _raw_all.min(), _raw_all.max()
    delta_w = max_w - min_w if max_w > min_w else 1.0
    for t in active_teachers:
        normalized = (all_weights[t] - min_w) / delta_w
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    # A molecule with weight 0 from every teacher (only possible when all of
    # them sit at the shared maximum) is shared equally rather than
    # predicted as 0 by the clamp below.
    _all_zero = weight_sum <= 0
    if np.any(_all_zero):
        for t in active_teachers:
            all_weights[t] = np.where(_all_zero, 1.0, all_weights[t])
        weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    fed_probs = np.zeros((n_test, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    pred_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    pred_probs_positive = fed_probs[:, 1]  # P(Active) for AUROC

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return pred_classes, pred_probs_positive, teacher_weights


def predict_fluid_quality_ensemble(teacher_models, cluster_data_train,
                                   test_fps, k):
    """Quality-weighted ensemble: weight teachers by cross-val MCC.

    Privacy: each silo shares one scalar (its MCC), not data.

    Args:
        teacher_models: list of (clf, size, df) tuples.
        cluster_data_train: list of TRAIN-ONLY DataFrames per cluster.
        test_fps: Morgan FP array for test molecules.
        k: number of silos.

    Returns:
        (pred_classes, pred_probs_positive, teacher_mcc_weights)
    """
    from sklearn.model_selection import KFold

    n_test = len(test_fps)
    mccs = {}
    all_probs = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            mccs[t] = 0.0
            continue  # Skip clusters with < 2 samples (no valid teacher)
        teacher_df = cluster_data_train[t]

        # Teacher predictions on test
        all_probs[t] = clf.predict_proba(test_fps)

        # Cross-val MCC on own TRAINING data only (no test leakage)
        # FLuID uses cv=5 always. We adapt to tiny silos where 5-fold may
        # fail (e.g. <5 samples in minority class). Equivalent when viable.
        X_t = np.asarray(list(teacher_df.FP))
        y_t = np.asarray(list(teacher_df.CLASS))
        n_classes = len(np.unique(y_t))
        min_class_count = min(np.bincount(y_t.astype(int)))
        n_cv = min(5, min_class_count, len(y_t))
        if n_classes < 2 or n_cv < 2:
            mccs[t] = 0.0
            continue
        try:
            y_pred_cv = cross_val_predict(
                clf, X_t, y_t,
                cv=KFold(n_splits=n_cv, shuffle=True, random_state=42))
            mccs[t] = max(0.0, matthews_corrcoef(y_t, y_pred_cv))
        except Exception:
            mccs[t] = 0.0

    if not all_probs:
        # All teachers were None — return uniform predictions
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    active_teachers = sorted(all_probs.keys())

    # Normalize MCC weights (active teachers only)
    total_mcc = sum(mccs.get(t, 0.0) for t in active_teachers)
    if total_mcc <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: mccs.get(t, 0.0) / total_mcc for t in active_teachers}

    # Weighted average
    fed_probs = np.zeros((n_test, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * weights[t]

    pred_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    pred_probs_positive = fed_probs[:, 1]  # P(Active) for AUROC

    return pred_classes, pred_probs_positive, weights


def federate_labels_and_train_student(teacher_models, cluster_data, fluid_df,
                                      k, test_fps, algorithm='rf',
                                      student_size=None,
                                      train_indices=None, seed=42):
    """FLuID-faithful student pipeline.

    Matches FLuID's published protocol:
    1. Each teacher predicts on public pool with similarity-weighted vote
       (FLuID's federate_labels uses W-T weights, not equal weights)
    2. Student data selected via FLuID's 'balanced' mode: sort by confidence
       descending, pick student_size/2 per class
    3. Student RF trained on federated pseudo-labels

    Privacy: only predictions on shared public molecules are exchanged.
    Test molecules are excluded from the student training pool.

    Simulation note: KIRBy uses training data as the public pool (no separate
    transfer pool like FLuID's 50K ChEMBL). This is a known simulation-level
    compromise documented in the federated audit.

    Args:
        train_indices: If provided, only these indices of fluid_df are
            used as the public pool. This prevents test data leakage.
        student_size: Number of federated pseudo-labels to select for
            the student. Default: min(10000, pool_size). FLuID uses 10000.

    Returns:
        (pred_classes, pred_probs, n_federated)
    """
    from sklearn.ensemble import RandomForestClassifier

    all_fps = np.asarray(list(fluid_df.FP))

    # Use only train indices as public pool (exclude test to avoid leakage)
    if train_indices is not None:
        pool_fps = all_fps[train_indices]
    else:
        pool_fps = all_fps
    n_pool = len(pool_fps)

    # Default student_size (FLuID: 10000; scale down for small pools)
    if student_size is None:
        student_size = min(10000, n_pool)

    # ── Step 1: Similarity-weighted federation (matches FLuID's federate_labels)
    # Each teacher predicts probabilities on the pool, weighted by 8-NN similarity
    all_probs = {}  # {t: (n_pool, 2)}
    all_weights = {}  # {t: (n_pool,)} similarity weights

    rng = np.random.RandomState(seed)

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue
        probs = clf.predict_proba(pool_fps)
        all_probs[t] = probs

        # 8-NN distance weighting using 20% sample of teacher's training FPs
        # (matches FLuID's annotate_transfer_data: data.sample(frac=0.2))
        teacher_df = cluster_data[t] if t < len(cluster_data) else None
        if teacher_df is not None and len(teacher_df) > 0:
            teacher_fps = np.asarray(list(teacher_df.FP))
            n_sample = max(1, int(0.2 * len(teacher_fps)))
            sample_idx = rng.choice(len(teacher_fps), size=n_sample, replace=False)
            teacher_fps_sample = teacher_fps[sample_idx]

            n_neighbors = min(8, len(teacher_fps_sample))
            # metric=None follows --distance-metric. FLuID's published
            # annotate_transfer_data measures this in Euclidean; pass
            # metric='euclidean' to pin it back for a replication run.
            knn = make_nearest_neighbors(n_neighbors=n_neighbors)
            knn.fit(teacher_fps_sample)
            distances, _ = knn.kneighbors(pool_fps)
            all_weights[t] = distances.mean(axis=1)
        else:
            all_weights[t] = np.ones(n_pool)

    active_teachers = sorted(all_probs.keys())
    if not active_teachers:
        raise ValueError("No active teachers available for student pipeline")

    # Normalize distances → similarities: (1 - normalized_distance)^2
    # (matches FLuID's annotate_transfer_data normalization)
    all_raw = np.column_stack([all_weights[t] for t in active_teachers])
    min_w = all_raw.min()
    max_w = all_raw.max()
    delta = max_w - min_w if max_w > min_w else 1.0

    for t in active_teachers:
        normalized = (all_weights[t] - min_w) / delta
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample
    weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    fed_probs = np.zeros((n_pool, 2))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    fed_classes = (fed_probs[:, 1] > fed_probs[:, 0]).astype(int)
    fed_confidence = np.maximum(fed_probs[:, 0], fed_probs[:, 1])

    # ── Step 2: Balanced selection by confidence (matches FLuID's 'balanced' mode)
    # FLuID's select_student_data: shuffle, then pick student_size/2 per class
    # sorted by federated confidence (highest first)
    half = student_size // 2
    active_idx = np.where(fed_classes == 1)[0]
    inactive_idx = np.where(fed_classes == 0)[0]

    # Sort each class by confidence descending, take top half
    active_order = np.argsort(fed_confidence[active_idx])[::-1]
    inactive_order = np.argsort(fed_confidence[inactive_idx])[::-1]

    n_active = min(half, len(active_idx))
    n_inactive = min(half, len(inactive_idx))
    selected_active = active_idx[active_order[:n_active]]
    selected_inactive = inactive_idx[inactive_order[:n_inactive]]

    selected = np.concatenate([selected_active, selected_inactive])
    student_fps = pool_fps[selected]
    student_classes = fed_classes[selected]

    # ── Step 3: Train student on federated pseudo-labels.
    # Use the caller-supplied seed for reproducibility (P1-9 fix —
    # prior hardcoded random_state=0 ignored `seed`).
    student = RandomForestClassifier(
        n_estimators=150, max_depth=15, class_weight='balanced',
        random_state=seed)
    student.fit(student_fps, student_classes)

    # Predict on test set
    test_probs = student.predict_proba(test_fps)
    pred_classes = (test_probs[:, 1] > test_probs[:, 0]).astype(int)
    pred_probs_positive = test_probs[:, 1]  # P(Active) for AUROC

    return pred_classes, pred_probs_positive, len(fed_classes)


# =============================================================================
# KIRBY FEDERATED FEATURE SELECTION (CONDITION 5)
# =============================================================================

# Mirrors the canonical 'match' → concrete-evaluator resolution in
# src/kirby/hybrid.py:3331-3355. Used to prevent silent degradation to
# equal-budget when 'match' is forwarded into get_model_factories (which
# doesn't know 'match'), and to prevent create_hybrid's no-model branch from
# resolving 'match' to 'ridge' (forbidden — feedback_no_ridge.md).
_MATCH_RESOLVABLE_MODELS = (
    'rf', 'xgboost', 'lightgbm', 'catboost', 'mlp', 'svm',
    'ridge', 'knn', 'gp', 'cnn', 'dnn', 'bnn', 'ngboost',
)


def _resolve_match(internal_evaluator, downstream_model_name=None):
    """Resolve 'match' to a concrete evaluator name.

    If a downstream model name is provided and recognised, use it (same family
    as the downstream model). Otherwise default to 'rf' as a safe fallback
    (NOT 'ridge' — see feedback_no_ridge.md).
    """
    if internal_evaluator != 'match':
        return internal_evaluator
    if downstream_model_name in _MATCH_RESOLVABLE_MODELS:
        return downstream_model_name
    return 'rf'


def allocate_budget_across_silos(silo_scores, budget, k,
                                 rule='proportional'):
    """How much of the feature budget each silo gets, from its own score.

    The coordinator has one number per silo — the score that silo reported on
    rows it held out — and one budget to divide. 'proportional' is what this
    has always done: a silo's share is its score over the total. That rule
    had never been measured against an equal split, so whether the scores
    were buying anything at all was unknown.

    equal        — budget // k each. The null: if nothing beats this, the
                   silo scores are not being used for anything.
    proportional — share = score / total. The long-standing behaviour.
    power2       — share = score^2 / total. Widens the gap between a good
                   silo and a mediocre one when the raw scores sit close
                   together, which they do when every silo is a cluster of
                   the same dataset.
    sqrt         — share = sqrt(score) / total. Flatter, for when one silo's
                   score is an outlier and would take nearly the whole budget.
    rank_linear  — share from the silo's RANK, not its score. Immune to the
                   scale of the scores, which are R squared on one dataset
                   and spearman on another.
    """
    ids = list(range(1, k + 1))
    sc = {s: max(0.0, float(silo_scores.get(s, 0.0))) for s in ids}
    total = sum(sc.values())

    if rule == 'equal' or total <= 0:
        w = {s: 1.0 / k for s in ids}
    elif rule == 'proportional':
        w = {s: sc[s] / total for s in ids}
    elif rule == 'power2':
        t = sum(v * v for v in sc.values())
        w = {s: (sc[s] ** 2) / t for s in ids} if t > 0 else {s: 1.0 / k for s in ids}
    elif rule == 'sqrt':
        t = sum(np.sqrt(v) for v in sc.values())
        w = {s: float(np.sqrt(sc[s]) / t) for s in ids} if t > 0 else {s: 1.0 / k for s in ids}
    elif rule == 'rank_linear':
        order = sorted(ids, key=lambda s: sc[s])
        rank = {s: i + 1 for i, s in enumerate(order)}
        t = sum(rank.values())
        w = {s: rank[s] / t for s in ids}
    else:
        raise ValueError(f"unknown allocation rule {rule!r}")

    return {s: max(1, int(budget * w[s])) for s in ids}


def federated_feature_selection(rep_dict, labels, silo_indices, k,
                                budget=100, allocation_method='performance_weighted',
                                importance_method='treeshap',
                                internal_evaluator='rf', seed=42,
                                groups=None, task_type=None,
                                downstream_model=None,
                                metric_fn=None, lower_is_better=False,
                                allocation_rule='proportional'):
    """Privacy-preserving federated feature selection.

    Protocol:
        1. Each silo runs create_hybrid locally on its own subset
        2. Each silo shares: feature_info (indices + importance), validation score
        3. Coordinator allocates total budget across silos proportional to scores
        4. Each silo selects its allocated features (top by importance)
        5. At inference: features selected using same indices per silo

    Privacy: no raw features leave any silo. Only feature indices,
    importance weights, and validation scores are shared.

    Args:
        rep_dict: {rep_name: np.ndarray (n_total, d)} — full feature arrays.
        labels: Labels for all samples (binary or continuous).
        silo_indices: {silo_id: np.array} for silos 1..k.
        k: Number of silos.
        budget: Total feature budget across all silos.
        allocation_method: KIRBy allocation method for within-silo selection.
        importance_method: KIRBy importance method.
        seed: Random seed.

    Returns:
        (X_hybrid, silo_budgets, silo_scores, silo_feature_infos)
    """
    from kirby.hybrid import create_hybrid, apply_feature_selection
    from sklearn.model_selection import GroupShuffleSplit
    from experiments._shared import get_model_factories

    # Caller-provided task_type wins; cardinality heuristic only as last resort.
    if task_type is None:
        warnings.warn(
            "federated_feature_selection: task_type not provided; falling back to "
            "cardinality heuristic (<=10 unique labels = classification). "
            "Pass task_type explicitly to avoid mis-routing regression silos.",
            RuntimeWarning, stacklevel=2,
        )
        is_classification = len(np.unique(labels)) <= 10
    else:
        is_classification = (task_type == 'classification')
    silo_task_type = 'classification' if is_classification else 'regression'

    silo_scores = {}
    silo_feature_infos = {}

    # Resolve 'match' to a concrete evaluator name BEFORE calling
    # get_model_factories (which doesn't know 'match' → would return {} → None
    # factory → silo_scores=0 → equal-budget fallback). See feedback_no_ridge.md
    # for why we default to 'rf' (not 'ridge') when no downstream is given.
    resolved_evaluator = _resolve_match(internal_evaluator, downstream_model)

    # Instantiate a concrete downstream-model instance (when possible) so that
    # create_hybrid's internal 'match' branch can resolve to the same family —
    # otherwise it silently degrades to 'ridge' (hybrid.py:3354-3355).
    downstream_model_instance = None
    if downstream_model is not None:
        try:
            _ds_factory = get_model_factories(
                [downstream_model], task_type=silo_task_type,
                n_samples=len(labels)).get(downstream_model)
            if _ds_factory is not None:
                downstream_model_instance = _ds_factory()
        except Exception:
            downstream_model_instance = None

    # Phase 1: Each silo computes importance locally and shares validation score
    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            silo_scores[s] = 0.0
            silo_feature_infos[s] = None
            continue

        # Silo's local labels. The feature matrices are sliced below, after
        # the held-out rows are carved out.
        silo_labels = labels[idx]

        # Each silo ranks up to the WHOLE budget locally; the coordinator's
        # trim in Phase 3 is what decides its share.
        #
        # This used to be an equal share, min(max(10, budget // k), n_silo).
        # Phase 2 then set silo_budgets proportional to the silo's validation
        # score, and Phase 3 trimmed each silo's list to that number — but a
        # silo had only ever ranked an equal share, so a silo allocated more
        # than the equal share could not supply it. The allocation that came
        # out was min(equal share, proportional share) for every silo, which
        # is never larger than the equal split and sums to LESS than the
        # budget whenever the scores are uneven. On caco2 with two silos the
        # coordinator allocated 48 and 52 features against a per-silo ceiling
        # of 50, so the condition reported 100 features and could build at
        # most 98. With five silos the ceiling is 20 each and a silo that
        # earns 40 still hands over 20.
        #
        # Ranking the full budget locally costs the silo more of its own
        # compute and shares exactly what it shared before — feature indices,
        # importance weights and one validation score. No raw feature leaves
        # the silo either way.
        n_silo = len(idx)
        per_silo_budget = min(budget, n_silo)
        # The same scaffold groups the silo's 80/20 scoring split uses below;
        # create_hybrid needs them or it splits the silo at random to decide
        # each representation's share of the budget.
        silo_groups = groups[idx] if groups is not None else None

        # The silo's held-out rows are carved out FIRST, before any feature is
        # picked. Until 2026-09-04 create_hybrid ran on all of the silo's rows
        # and the score was then read off a 20% slice of those same rows, so
        # every feature entering the silo's set had already seen the labels it
        # was graded against. silo_scores set the budget split across silos,
        # so that inflation went straight into how many features each silo got.
        # The features the silo shares downstream are the ones fitted on the
        # 80%: one create_hybrid call, not two.
        va_local = None
        if len(idx) >= 30:
            if silo_groups is not None and len(np.unique(silo_groups)) >= 2:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=seed + s)
                tr_local, va_local = next(
                    gss.split(np.zeros(len(idx)), silo_labels,
                              groups=silo_groups))
            else:
                # No scaffold groups for this silo. A random 20% would put
                # near-duplicates on both sides, so refuse rather than report a
                # score drawn that way (feedback_no_random_splits.md). The silo
                # keeps the flat score used for tiny silos and its features are
                # fitted on all of its rows.
                # print, not warnings.warn — see the note in
                # cluster_training_only: warnings are filtered off at import.
                print(f"  [silo {s}] no usable scaffold groups; scoring on a "
                      f"random 20% is forbidden, so it takes the flat 0.5 "
                      f"score and its features are fitted on all its rows.")
                va_local = None

        if va_local is not None:
            fit_rows = idx[tr_local]
            val_rows = idx[va_local]
        else:
            fit_rows = idx
            val_rows = None

        fit_reps = {r: arr[fit_rows] for r, arr in rep_dict.items()}
        fit_labels = labels[fit_rows]
        fit_groups = groups[fit_rows] if groups is not None else None
        per_silo_budget = min(per_silo_budget, len(fit_rows))

        try:
            X_silo, fi = create_hybrid(
                fit_reps, fit_labels, groups=fit_groups,
                allocation_method=allocation_method,
                importance_method=importance_method,
                budget=per_silo_budget,
                internal_evaluator=internal_evaluator,
                model=downstream_model_instance,
                metric_fn=metric_fn, lower_is_better=lower_is_better,
            )
            silo_feature_infos[s] = fi

            # Silo shares its validation score (one scalar), measured on rows
            # no feature was chosen from.
            # Evaluator matches the downstream model family the user passed
            # in (no Ridge sensitivity check, per feedback_no_ridge.md).
            if val_rows is not None:
                X_silo_val = apply_feature_selection(
                    {r: arr[val_rows] for r, arr in rep_dict.items()}, fi)
                eval_factory = get_model_factories(
                    [resolved_evaluator], task_type=silo_task_type,
                    n_samples=len(fit_rows)).get(resolved_evaluator)
                evaluator = eval_factory() if eval_factory else None
                if evaluator is None:
                    silo_scores[s] = 0.0
                else:
                    evaluator.fit(X_silo, fit_labels)
                    # Score the silo on the metric the CALLER passed, when it
                    # passed one and higher is better on it.
                    #
                    # This used to be evaluator.score() and nothing else, which
                    # is R squared for a regressor and accuracy for a
                    # classifier — never the metric_fn just above it, which the
                    # callers pass precisely so the budget split is decided on
                    # the quantity the experiment reports. On a hard regression
                    # silo R squared is negative, max(0.0, ...) turns it into
                    # 0.0, and when every silo lands on 0.0 the coordinator has
                    # nothing to be proportional to and splits the budget
                    # EQUALLY. MEASURED on fed05's C4, 2026-09-18: all three
                    # antibody silos scored 0.0 and each took exactly 33 of the
                    # 100 features, so the condition that asks whether a noisy
                    # silo receives less could not answer — the noisy silo and
                    # the clean ones got the same number by construction.
                    #
                    # A lower-is-better metric has no meaning as a
                    # proportional weight (a budget proportional to RMSE gives
                    # the worst silo the most), so those keep evaluator.score()
                    # — which for RMSE is exactly the right shape: R squared is
                    # 1 minus the silo's squared error over the error of
                    # predicting the mean.
                    _y_val = labels[val_rows]
                    if metric_fn is not None and not lower_is_better:
                        # AUC is a ranking metric and needs a graded score.
                        # Fed the 0/1 classes from predict(), it returns
                        # (sensitivity + specificity) / 2, so every silo's
                        # share was set by balanced accuracy under the AUC
                        # name. hybrid.py's _score_rep_for_budget made the
                        # same change on 2026-09-13.
                        _pred_val = None
                        if (is_classification
                                and getattr(metric_fn, '__name__', '')
                                in ('_safe_roc_auc', 'roc_auc_score')
                                and hasattr(evaluator, 'predict_proba')):
                            try:
                                _pp = np.asarray(
                                    evaluator.predict_proba(X_silo_val))
                                if _pp.ndim == 2 and _pp.shape[1] == 2:
                                    _pred_val = _pp[:, 1]
                            except Exception:
                                _pred_val = None
                        if _pred_val is None:
                            _pred_val = evaluator.predict(X_silo_val)
                        try:
                            _sc = float(metric_fn(_y_val, _pred_val))
                            if not np.isfinite(_sc):
                                raise ValueError('metric is not finite')
                        except Exception:
                            _sc = float(evaluator.score(X_silo_val, _y_val))
                    else:
                        _sc = float(evaluator.score(X_silo_val, _y_val))
                    silo_scores[s] = max(0.0, _sc)
            else:
                silo_scores[s] = 0.5  # default for tiny / ungrouped silos

        except Exception as e:
            warnings.warn(f"Silo {s} hybrid failed: {e}")
            silo_scores[s] = 0.0
            silo_feature_infos[s] = None

    # Phase 2: Coordinator allocates budget across silos from their scores
    silo_budgets = allocate_budget_across_silos(
        silo_scores, budget, k, rule=allocation_rule)

    # Phase 3: Apply budget allocation — trim each silo's feature_info to
    # its allocated budget, then deduplicate features across silos.
    seen_features = set()  # (rep_name, feature_index) pairs already selected
    hybrid_parts = []

    for s in range(1, k + 1):
        fi = silo_feature_infos[s]
        if fi is None:
            continue

        allocated = silo_budgets.get(s, 0)
        trimmed_fi = {}

        # Scale the silo's own split down to `allocated`, keeping the shares
        # the allocation method gave each representation.
        #
        # This loop used to walk the representations in dictionary order and
        # hand each one whatever was left of the allocation, so the first
        # representation in the dict took its fill and the ones after it took
        # the remainder — dictionary order decided the split rather than
        # anything measured. It was invisible while every silo ranked exactly
        # its equal share, because the trim then almost never bit; now that a
        # silo ranks the whole budget (see Phase 1) the trim is what picks the
        # features.
        #
        # Ranking every feature the silo holds against every other one is not
        # available here: an importance score is computed per representation,
        # by a model fitted on that representation alone, so a gain of 3.1 on
        # mordred and a gain of 3.1 on ecfp4 are not the same quantity. What
        # IS comparable is the split create_hybrid already chose — how many
        # features each representation earned. That split is kept and scaled,
        # by largest remainder so the parts sum to the allocation, and within
        # one representation the features are taken in its own importance
        # order.
        avail = {}
        for rep_name, rep_info in fi.items():
            if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
                trimmed_fi[rep_name] = rep_info
                continue
            indices = rep_info['selected_indices']
            importances = np.asarray(
                rep_info.get('importance_scores', np.ones(len(indices))),
                dtype=float)
            keep_i, keep_s = [], []
            for idx_i, feat_idx in enumerate(indices):
                if (rep_name, int(feat_idx)) in seen_features:
                    continue              # another silo already took it
                keep_i.append(int(feat_idx))
                keep_s.append(float(importances[idx_i])
                              if idx_i < len(importances) else 0.0)
            order = np.argsort(keep_s)[::-1]
            avail[rep_name] = ([keep_i[o] for o in order],
                               [keep_s[o] for o in order])

        n_avail = sum(len(v[0]) for v in avail.values())
        target = min(max(0, allocated), n_avail)
        exact = {r: (len(v[0]) * target / n_avail) if n_avail else 0.0
                 for r, v in avail.items()}
        take = {r: min(int(e), len(avail[r][0])) for r, e in exact.items()}
        # Largest remainder: hand the leftover slots to the representations
        # whose exact share was cut by the most.
        leftover = target - sum(take.values())
        for r in sorted(avail, key=lambda r: -(exact[r] - take[r])):
            if leftover <= 0:
                break
            if take[r] < len(avail[r][0]):
                take[r] += 1
                leftover -= 1

        for rep_name, (idxs, scores) in avail.items():
            n = take[rep_name]
            trimmed_fi[rep_name] = {
                'selected_indices': np.array(idxs[:n], dtype=int),
                'importance_scores': np.array(scores[:n]),
                'n_features': n,
            }
            for feat_idx in idxs[:n]:
                seen_features.add((rep_name, int(feat_idx)))

        silo_feature_infos[s] = trimmed_fi

        try:
            X_silo = apply_feature_selection(rep_dict, trimmed_fi)
            if X_silo.shape[1] > 0:
                hybrid_parts.append(X_silo)
            else:
                silo_feature_infos[s] = None  # mark failed
        except Exception:
            silo_feature_infos[s] = None  # mark failed
            continue

    if not hybrid_parts:
        raise ValueError("No silo produced valid feature selections")

    X_hybrid = np.hstack(hybrid_parts)

    return X_hybrid, silo_budgets, silo_scores, silo_feature_infos


# =============================================================================
# FEDERATED NOISE DETECTION (CONDITION 6)
# =============================================================================

def stack_reps_for_distance(rep_dict, rows=None, coefficient=None):
    """Glue every representation into one matrix that a DISTANCE can be taken on.

    Each representation is z-scored on its own and then divided by the square
    root of how many features it has, so that two representations count the same
    however wide they are. Plain np.hstack + one StandardScaler does not do this:
    it leaves every feature with variance 1, so a 2560-feature protein embedding
    supplies 2560 of the terms in a squared Euclidean distance and a 200-feature
    descriptor set supplies 200 — the embedding decides the neighbours almost on
    its own.

    The divisor is kirby.similarity.preprocessing.block_divisor, the same one the
    sim_* recipes use. Which divisor is right depends on the distance: sqrt(width)
    under Euclidean, width under Manhattan, width ** (1/p) under fractional
    Minkowski, 1.0 under cosine. `coefficient=None` reads --distance-metric so the
    divisor and the detector cannot disagree. They did before: the divisor was
    fixed at sqrt(width) whatever the detector measured, which puts the width back
    as the block's weight under any non-quadratic distance.

    The returned matrix is ALREADY z-scored. Standardising it a second time
    cancels the divisor exactly and puts the width back as the weight, so every
    caller must pass standardize=False / already_scaled=True downstream.

    `rows` selects a subset of rows (a silo) before any statistic is computed, so
    a silo's own mean and spread set its scale.
    """
    from kirby.similarity.preprocessing import stack_blocks
    from kirby.similarity.distance_config import base_and_p

    base, p = base_and_p(coefficient)
    names = sorted(rep_dict.keys())
    blocks = []
    for name in names:
        X = np.asarray(rep_dict[name], dtype=np.float64)
        if rows is not None:
            X = X[rows]
        blocks.append(np.nan_to_num(np.clip(X, -1e10, 1e10), nan=0.0,
                                    posinf=1e10, neginf=-1e10))
    stacked, _stats = stack_blocks(blocks, coefficient=base, p=p,
                                   equalise=True)
    return stacked


def federated_noise_detection(rep_dict, labels, silo_indices, noise_mask_true,
                              k, detector_names=None, groups=None,
                              task_type=None):
    """Run noise detectors in three privacy settings.

    Settings:
        - centralized: all features pooled (unrealistic baseline)
        - per_silo: each silo runs detectors locally (privacy-preserving)
        - federated_hybrid: detectors on federated feature-selected data

    Privacy: per_silo and federated_hybrid never share raw features.

    groups: one scaffold-group id per row of `labels`, in the same order.
    Handed to federated_feature_selection so each silo's 80/20 scoring split
    is grouped rather than a random shuffle. Without it that split is refused
    and every silo takes the flat score.

    Returns:
        {setting: {detector: {precision, recall, f1, n_flagged}}}
    """
    from sklearn.preprocessing import StandardScaler
    from kirby.utils.noise_mitigation_methods import (
        get_mitigation_method, compute_distance_matrix,
        DISTANCE_AWARE_METHODS,
    )

    if detector_names is None:
        detector_names = [
            'knn_k5', 'knn_k10', 'lof', 'distance_weighted',
            'cv_disagreement', 'bootstrap_ensemble',
            'zscore', 'prediction_error',
        ]

    n_total = len(labels)
    n_noisy = int(noise_mask_true.sum())
    results = {}

    def _eval_detectors(X, y, mask_true, setting_name, D=None,
                        already_scaled=False):
        setting_results = {}
        # already_scaled: X came from stack_reps_for_distance, which z-scored
        # each representation and then divided it by sqrt(its width). Running
        # StandardScaler over that again restores every feature to variance 1
        # and cancels the divisor exactly, which is the thing being fixed.
        X_s = X if already_scaled else StandardScaler().fit_transform(X)
        for det_name in detector_names:
            try:
                kwargs = {}
                if det_name in DISTANCE_AWARE_METHODS and D is not None:
                    kwargs['precomputed_distances'] = D
                detector = get_mitigation_method(det_name, **kwargs)
                flagged_idx = detector.identify_noise(X_s, y)
                flagged_mask = np.zeros(len(y), dtype=bool)
                flagged_mask[flagged_idx] = True

                tp = int((flagged_mask & mask_true).sum())
                n_flagged = int(flagged_mask.sum())
                precision = tp / max(n_flagged, 1)
                recall = tp / max(mask_true.sum(), 1)
                f1 = (2 * precision * recall /
                      max(precision + recall, 1e-10))

                setting_results[det_name] = {
                    'n_flagged': n_flagged,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                }
            except Exception as e:
                setting_results[det_name] = {'error': str(e)}
        return setting_results

    # Setting 1: Centralized (unrealistic)
    X_concat = stack_reps_for_distance(rep_dict)
    D_concat = compute_distance_matrix(X_concat, standardize=False)
    results['centralized'] = _eval_detectors(
        X_concat, labels, noise_mask_true, 'centralized', D_concat,
        already_scaled=True)

    # Setting 2: Per-silo (privacy-preserving)
    # Each silo runs detectors locally, reports flagged global indices
    per_silo_flagged = np.zeros(n_total, dtype=bool)
    per_silo_detail = {}
    for s in range(1, k + 1):
        idx = silo_indices[s]
        if len(idx) < 20:
            continue
        X_silo = stack_reps_for_distance(rep_dict, rows=idx)
        silo_noise_mask = noise_mask_true[idx]
        silo_results = _eval_detectors(
            X_silo, labels[idx], silo_noise_mask, f'silo_{s}',
            already_scaled=True)
        per_silo_detail[f'silo_{s}'] = silo_results

    results['per_silo'] = per_silo_detail

    # Setting 3: Federated hybrid (privacy-preserving feature selection)
    try:
        X_fed, _budgets, _scores, _fis = federated_feature_selection(
            rep_dict, labels, silo_indices, k,
            allocation_method='performance_weighted',
            importance_method='treeshap',
            internal_evaluator='rf', seed=42,
            groups=groups, task_type=task_type)
        D_fed = compute_distance_matrix(X_fed, standardize=True)
        results['federated_hybrid'] = _eval_detectors(
            X_fed, labels, noise_mask_true, 'federated_hybrid', D_fed)
    except Exception as e:
        results['federated_hybrid'] = {'_error': str(e)}

    return results


# =============================================================================
# CLASSIFICATION METRICS HELPER
# =============================================================================

def compute_clf_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics for federated experiment evaluation.

    Returns dict with MCC, BAC, AUROC, F1 (binary and macro-F1 for multiclass).
    Also includes 'accuracy' and 'f1_weighted' (FED-26) — the registered
    primary metrics for several antibody-classification datasets, so a fed
    experiment routing such a dataset through here reports the right metric.
    `y_prob` may be a 1-D positive-class probability (binary) or the full
    (n, n_classes) probability matrix (multiclass).
    """
    n_classes = len(np.unique(y_true))
    f1_avg = 'binary' if n_classes == 2 else 'macro'
    metrics = {
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'bac': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average=f1_avg, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(
            f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob)
            if n_classes == 2:
                if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                    y_prob_arr = y_prob_arr[:, 1]
                metrics['auroc'] = float(roc_auc_score(y_true, y_prob_arr))
            elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == n_classes:
                metrics['auroc'] = float(roc_auc_score(
                    y_true, y_prob_arr, multi_class='ovr', average='macro'))
            else:
                metrics['auroc'] = float('nan')
        except ValueError:
            metrics['auroc'] = float('nan')
    return metrics


# =============================================================================
# REGRESSION METRICS AND ENSEMBLE HELPERS (for fed02+)
# =============================================================================

def compute_reg_metrics(y_true, y_pred):
    """Compute regression metrics for federated experiment evaluation.

    Returns dict with rmse, spearman_rho, spearman_p, pearson_r, pearson_p, r2.
    """
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import spearmanr as _spearmanr, pearsonr as _pearsonr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # A rank correlation against a constant is undefined, and scipy returns
    # NaN for it. NaN is not valid JSON, it survives into the result file, and
    # it names no cause — a reader cannot tell it apart from a failed call.
    # MEASURED on fed05's dms silo (abagym_EGFR), 2026-09-18: that set is 659
    # single mutants of one antibody, its 5 sequence-identity groups are 5
    # disjoint sets of mutated positions, and under a group-aware split the
    # one-hot representation has 580 columns varying in the training rows,
    # 115 in the held-out rows and ZERO varying in both. The forest saw no
    # feature that moved on the rows it was scored on, predicted one value
    # for all 109 of them, and the file recorded spearman_rho = NaN.
    #
    # Both correlations are None (JSON null) when either side is constant,
    # and `constant_prediction` says which side it was.
    _const_pred = bool(np.ptp(y_pred) == 0)
    _const_true = bool(np.ptp(y_true) == 0)
    if _const_pred or _const_true:
        sp_r = sp_p = pe_r = pe_p = None
    else:
        try:
            sp_r, sp_p = _spearmanr(y_true, y_pred)
            sp_r, sp_p = float(sp_r), float(sp_p)
            if not np.isfinite(sp_r):
                sp_r = sp_p = None
        except Exception:
            sp_r, sp_p = None, None
        try:
            pe_r, pe_p = _pearsonr(y_true, y_pred)
            pe_r, pe_p = float(pe_r), float(pe_p)
            if not np.isfinite(pe_r):
                pe_r = pe_p = None
        except Exception:
            pe_r, pe_p = None, None

    return {
        'rmse': rmse,
        'spearman_rho': sp_r,
        'spearman_p': sp_p,
        'pearson_r': pe_r,
        'pearson_p': pe_p,
        'r2': r2,
        'constant_prediction': _const_pred,
        'constant_truth': _const_true,
    }


def predict_fluid_similarity_ensemble_regression(teacher_models,
                                                 silo_train_features,
                                                 silo_test_features, k,
                                                 seed=42):
    """FLuID-style 8-NN similarity-weighted regression ensemble.

    Each silo trains a regressor on its own representation. Predictions are
    combined via weighted mean, with weights derived from 8-NN distances in
    each silo's OWN feature space.

    Privacy: each silo shares only predictions and similarity weights.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
            Indices 1..k are (regressor, n_train, X_train) tuples.
        silo_train_features: list (length k+1). Index 0 is None.
            Indices 1..k are training feature arrays for each silo.
        silo_test_features: list (length k+1). Index 0 is None.
            Indices 1..k are test feature arrays for each silo.
        k: number of silos.
        seed: random seed.

    Returns:
        (predictions, teacher_weights_per_sample)
    """
    n_test = silo_test_features[1].shape[0]

    all_preds = {}
    all_weights = {}

    for t in range(1, k + 1):
        reg = teacher_models[t][0]
        if reg is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        # Predict in this silo's feature space
        all_preds[t] = reg.predict(X_test_t)

        # 8-NN distance weighting in this silo's OWN feature space
        # A silo with no training rows has nothing to measure a distance
        # from. max(1, int(0.2 * 0)) asks numpy for one sample out of an
        # empty array, which raises. The student pipeline above already
        # handles this by giving that teacher a flat weight; do the same
        # here so one empty cluster cannot end the whole run.
        if len(X_train_t) == 0:
            all_weights[t] = np.ones(len(X_test_t))
            continue
        n_sample = max(1, int(0.2 * len(X_train_t)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(X_train_t), size=n_sample, replace=False)
        X_ref = X_train_t[sample_idx]

        n_neighbors = min(8, len(X_ref))
        # metric=None follows --distance-metric. FLuID's published
        # annotate_transfer_data measures this in Euclidean; pass
        # metric='euclidean' to pin it back for a replication run.
        knn = make_nearest_neighbors(n_neighbors=n_neighbors)
        knn.fit(X_ref)
        distances, _ = knn.kneighbors(X_test_t)
        all_weights[t] = distances.mean(axis=1)

    if not all_preds:
        # All teachers were None — return zeros
        return np.zeros(n_test), np.zeros((n_test, k))

    active_teachers = sorted(all_preds.keys())

    # Normalize distances → similarities PER SILO (each silo's feature
    # space has different scale, so global normalization would be meaningless)
    for t in active_teachers:
        raw = all_weights[t]
        min_t, max_t = raw.min(), raw.max()
        delta_t = max_t - min_t if max_t > min_t else 1.0
        normalized = (raw - min_t) / delta_t
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    # Each silo is min-max scaled on its own, so the held-out molecule
    # farthest from silo t gets weight 0 from silo t. A molecule that is the
    # farthest in EVERY silo got weight 0 from all of them, and the clamp
    # below then divided 0 by 1e-10: its prediction came out as exactly 0
    # (or class 0 at probability 0), whatever the teachers said. Measured
    # 2026-09-28 on herg_ki, fold 3 of 3: one of 470 held-out molecules.
    # All silos are equally far from such a molecule, so they share it.
    _all_zero = weight_sum <= 0
    if np.any(_all_zero):
        for t in active_teachers:
            all_weights[t] = np.where(_all_zero, 1.0, all_weights[t])
        weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted mean of continuous predictions
    predictions = np.zeros(n_test)
    for t in active_teachers:
        predictions += all_preds[t] * all_weights[t]

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return predictions, teacher_weights


def predict_fluid_quality_ensemble_regression(teacher_models,
                                              silo_train_features,
                                              silo_test_features, k):
    """Quality-weighted regression ensemble: weight by inverse cross-val RMSE.

    Privacy: each silo shares one scalar (its RMSE), not data.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
        silo_train_features: list (length k+1) of (X_train, y_train) tuples.
            Index 0 is None. Each entry is (features, labels) for that silo.
        silo_test_features: list (length k+1) of X_test arrays.
        k: number of silos.

    Returns:
        (predictions, rmse_weights_dict)
    """
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import mean_squared_error

    n_test = silo_test_features[1].shape[0]
    rmses = {}
    all_preds = {}

    for t in range(1, k + 1):
        reg = teacher_models[t][0]
        if reg is None:
            rmses[t] = float('inf')
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t, y_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_preds[t] = reg.predict(X_test_t)

        # Cross-val RMSE on own training data
        n_cv = min(5, max(2, len(y_train_t) // 5))
        try:
            y_cv = cross_val_predict(
                reg, X_train_t, y_train_t,
                cv=KFold(n_splits=n_cv, shuffle=True, random_state=42))
            rmses[t] = float(np.sqrt(mean_squared_error(y_train_t, y_cv)))
        except Exception:
            rmses[t] = float('inf')

    if not all_preds:
        # All teachers were None — return zeros
        return np.zeros(n_test), {}

    active_teachers = sorted(all_preds.keys())

    # Weights: inverse RMSE (lower RMSE = higher weight), active teachers only
    inv_rmses = {t: 1.0 / max(rmses.get(t, float('inf')), 1e-6)
                 for t in active_teachers}
    total = sum(inv_rmses.values())
    if total <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: v / total for t, v in inv_rmses.items()}

    # Quality-weighted mean
    predictions = np.zeros(n_test)
    for t in active_teachers:
        predictions += all_preds[t] * weights[t]

    return predictions, weights


def predict_fluid_similarity_ensemble_classification(
        teacher_models, silo_train_features, silo_test_features, k, seed=42):
    """FLuID-style 8-NN similarity-weighted classification ensemble.

    Mirrors predict_fluid_similarity_ensemble_regression but uses
    predict_proba and weighted probability averaging for classification.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
            Indices 1..k are (classifier, n_train, X_train) tuples.
        silo_train_features: list (length k+1). Index 0 is None.
            Indices 1..k are training feature arrays for each silo.
        silo_test_features: list (length k+1). Index 0 is None.
            Indices 1..k are test feature arrays for each silo.
        k: number of silos.
        seed: random seed.

    Returns:
        (pred_classes, pred_probs_positive, teacher_weights_per_sample)
    """
    n_test = silo_test_features[1].shape[0]

    all_probs = {}
    all_weights = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_probs[t] = clf.predict_proba(X_test_t)

        # 8-NN distance weighting in this silo's OWN feature space
        # A silo with no training rows has nothing to measure a distance
        # from. max(1, int(0.2 * 0)) asks numpy for one sample out of an
        # empty array, which raises. The student pipeline above already
        # handles this by giving that teacher a flat weight; do the same
        # here so one empty cluster cannot end the whole run.
        if len(X_train_t) == 0:
            all_weights[t] = np.ones(len(X_test_t))
            continue
        n_sample = max(1, int(0.2 * len(X_train_t)))
        rng = np.random.RandomState(seed + t)
        sample_idx = rng.choice(len(X_train_t), size=n_sample, replace=False)
        X_ref = X_train_t[sample_idx]

        n_neighbors = min(8, len(X_ref))
        # metric=None follows --distance-metric. FLuID's published
        # annotate_transfer_data measures this in Euclidean; pass
        # metric='euclidean' to pin it back for a replication run.
        knn = make_nearest_neighbors(n_neighbors=n_neighbors)
        knn.fit(X_ref)
        distances, _ = knn.kneighbors(X_test_t)
        all_weights[t] = distances.mean(axis=1)

    if not all_probs:
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), np.zeros((n_test, k))

    active_teachers = sorted(all_probs.keys())

    # Normalize distances → similarities PER SILO
    for t in active_teachers:
        raw = all_weights[t]
        min_t, max_t = raw.min(), raw.max()
        delta_t = max_t - min_t if max_t > min_t else 1.0
        normalized = (raw - min_t) / delta_t
        all_weights[t] = (1.0 - normalized) ** 2

    # Normalize weights to sum to 1 per sample (active teachers only)
    weight_sum = sum(all_weights[t] for t in active_teachers)
    # Each silo is min-max scaled on its own, so the held-out molecule
    # farthest from silo t gets weight 0 from silo t. A molecule that is the
    # farthest in EVERY silo got weight 0 from all of them, and the clamp
    # below then divided 0 by 1e-10: its prediction came out as exactly 0
    # (or class 0 at probability 0), whatever the teachers said. Measured
    # 2026-09-28 on herg_ki, fold 3 of 3: one of 470 held-out molecules.
    # All silos are equally far from such a molecule, so they share it.
    _all_zero = weight_sum <= 0
    if np.any(_all_zero):
        for t in active_teachers:
            all_weights[t] = np.where(_all_zero, 1.0, all_weights[t])
        weight_sum = sum(all_weights[t] for t in active_teachers)
    weight_sum = np.maximum(weight_sum, 1e-10)
    for t in active_teachers:
        all_weights[t] = all_weights[t] / weight_sum

    # Weighted average of probability distributions
    n_classes = all_probs[active_teachers[0]].shape[1]
    fed_probs = np.zeros((n_test, n_classes))
    for t in active_teachers:
        fed_probs += all_probs[t] * all_weights[t][:, np.newaxis]

    pred_classes = fed_probs.argmax(axis=1)
    pred_probs_positive = fed_probs[:, 1] if n_classes == 2 else None

    teacher_weights = np.column_stack(
        [all_weights.get(t, np.zeros(n_test)) for t in range(1, k + 1)])

    return pred_classes, pred_probs_positive, teacher_weights


def predict_fluid_quality_ensemble_classification(
        teacher_models, silo_train_features, silo_test_features, k):
    """Quality-weighted classification ensemble: weight by cross-val MCC.

    Mirrors predict_fluid_quality_ensemble_regression but uses classifiers.

    Args:
        teacher_models: list (length k+1). Index 0 is None placeholder.
        silo_train_features: list (length k+1) of (X_train, y_train) tuples.
            Index 0 is None.
        silo_test_features: list (length k+1) of X_test arrays.
        k: number of silos.

    Returns:
        (pred_classes, pred_probs_positive, mcc_weights_dict)
    """
    from sklearn.model_selection import KFold, cross_val_predict

    n_test = silo_test_features[1].shape[0]
    mccs = {}
    all_probs = {}

    for t in range(1, k + 1):
        clf = teacher_models[t][0]
        if clf is None:
            mccs[t] = 0.0
            continue  # Skip clusters with < 2 samples (no valid teacher)
        X_train_t, y_train_t = silo_train_features[t]
        X_test_t = silo_test_features[t]

        all_probs[t] = clf.predict_proba(X_test_t)

        # Cross-val MCC on own training data
        n_classes = len(np.unique(y_train_t))
        min_class = min(np.bincount(y_train_t.astype(int)))
        n_cv = min(5, min_class, len(y_train_t))
        if n_classes < 2 or n_cv < 2:
            mccs[t] = 0.0
            continue
        try:
            y_cv = cross_val_predict(
                clf, X_train_t, y_train_t,
                cv=KFold(n_splits=n_cv, shuffle=True, random_state=42))
            mccs[t] = max(0.0, float(matthews_corrcoef(y_train_t, y_cv)))
        except Exception:
            mccs[t] = 0.0

    if not all_probs:
        return np.zeros(n_test, dtype=int), np.full(n_test, 0.5), {}

    active_teachers = sorted(all_probs.keys())

    # Normalize MCC weights (active teachers only)
    total_mcc = sum(mccs.get(t, 0.0) for t in active_teachers)
    if total_mcc <= 0:
        weights = {t: 1.0 / len(active_teachers) for t in active_teachers}
    else:
        weights = {t: mccs.get(t, 0.0) / total_mcc for t in active_teachers}

    # Weighted average of probabilities
    n_classes = all_probs[active_teachers[0]].shape[1]
    fed_probs = np.zeros((n_test, n_classes))
    for t in active_teachers:
        fed_probs += all_probs[t] * weights[t]

    pred_classes = fed_probs.argmax(axis=1)
    pred_probs_positive = fed_probs[:, 1] if n_classes == 2 else None

    return pred_classes, pred_probs_positive, weights


def fluid_within_kirby_across(rep_dict_train, rep_dict_test, labels,
                              k_intra=3, budget=100, seed=42,
                              allocation_method='performance_weighted',
                              importance_method='treeshap',
                              internal_evaluator='rf',
                              groups=None, task_type=None,
                              downstream_model=None,
                              metric_fn=None, lower_is_better=False):
    """Condition 6: FLuID clustering within each silo, KIRBy across silos.

    For each representation (silo):
        1. PCA to reduce dimensionality
        2. K-Means to cluster molecules within that silo's feature space
    Then create virtual silo indices (one per rep × sub-cluster) and run
    federated_feature_selection across the expanded set.

    Privacy: intra-silo clustering is local. Cross-silo allocation uses
    the federated_feature_selection protocol.

    Args:
        rep_dict_train: {rep_name: X_train_array} for all silo reps.
        rep_dict_test: {rep_name: X_test_array} for all silo reps.
        labels: training labels (continuous).
        k_intra: number of sub-clusters per silo.
        budget: total feature budget.
        seed: random seed.

    Returns:
        (X_hybrid_train, X_hybrid_test, silo_budgets, silo_scores, metadata)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from kirby.hybrid import apply_feature_selection, create_hybrid

    virtual_silo_indices = {}
    metadata = {}
    silo_counter = 1

    for rep_name, X_train in rep_dict_train.items():
        # PCA to reduce high-dim reps before clustering
        n_components = max(1, min(50, X_train.shape[1], X_train.shape[0] - 1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        if n_components < X_scaled.shape[1]:
            X_reduced = PCA(n_components=n_components,
                            random_state=seed).fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        km = KMeans(n_clusters=k_intra, random_state=seed, n_init=10)
        assignments = km.fit_predict(X_reduced)

        for c in range(k_intra):
            indices = np.where(assignments == c)[0]
            if len(indices) >= 20:
                virtual_silo_indices[silo_counter] = indices
                metadata[silo_counter] = {
                    'parent_rep': rep_name,
                    'cluster': c,
                    'n_samples': len(indices),
                }
                silo_counter += 1

    k_expanded = silo_counter - 1
    if k_expanded == 0:
        raise ValueError("No virtual silos with >= 20 samples")

    # Run per-silo feature selection, restricting each silo to its parent rep.
    # This enforces the privacy boundary: silo from rep A cannot select from rep B.
    from sklearn.model_selection import GroupShuffleSplit as _GSS, ShuffleSplit as _SS
    from experiments._shared import get_model_factories

    # task_type wins; cardinality heuristic only as last resort (P0-6).
    if task_type is None:
        is_classification = len(np.unique(labels)) <= 10
    else:
        is_classification = (task_type == 'classification')
    silo_task_type = 'classification' if is_classification else 'regression'

    # Resolve 'match' → concrete evaluator name BEFORE get_model_factories
    # (it doesn't recognise 'match' and would silently fall back to equal
    # budget). Default to 'rf' when no downstream supplied (NOT 'ridge', per
    # feedback_no_ridge.md).
    resolved_evaluator = _resolve_match(internal_evaluator, downstream_model)

    # Instantiate the downstream model (if provided) so that create_hybrid's
    # internal 'match' branch resolves to the matching family instead of
    # silently degrading to 'ridge' (hybrid.py:3354-3355).
    downstream_model_instance = None
    if downstream_model is not None:
        try:
            _ds_factory = get_model_factories(
                [downstream_model], task_type=silo_task_type,
                n_samples=len(labels)).get(downstream_model)
            if _ds_factory is not None:
                downstream_model_instance = _ds_factory()
        except Exception:
            downstream_model_instance = None

    silo_fis = {}
    silo_scores = {}
    hybrid_train_parts = []

    # ── Pass 1: score each virtual silo on rows its own selection never saw ──
    # Until 2026-09-12 one pass did both jobs: it selected features from every
    # row of the silo, then split those same rows and scored a model on the
    # part it called validation. Every feature had already been chosen with
    # the validation rows' labels in view, so the score was not a held-out
    # score and the budget it went on to set was allocated on it.
    #
    # The split now comes first. A silo's probe selection sees only its
    # training part, and the score comes off the part that selection never
    # touched. The probe budget is the old equal split, which is only ever
    # used to rank the silos against each other.
    probe_budget_of = {}
    for s in range(1, k_expanded + 1):
        idx = virtual_silo_indices[s]
        parent_rep = metadata[s]['parent_rep']
        silo_labels = labels[idx]
        silo_groups = groups[idx] if groups is not None else None
        n_silo = len(idx)
        probe_budget = min(max(5, budget // k_expanded), n_silo)
        probe_budget_of[s] = probe_budget

        if n_silo < 30:
            # Too few rows to hold any out. A silo scored 0.5 by default gets
            # its share of the budget from that default, not from a measured
            # score, so say so rather than let 0.5 read as a measurement.
            silo_scores[s] = 0.5
            metadata[s]['score_source'] = 'default_0.5_too_few_rows'
            continue

        try:
            if silo_groups is not None and len(np.unique(silo_groups)) >= 2:
                splitter = _GSS(n_splits=1, test_size=0.2,
                                random_state=seed + s)
                tr_v, va_v = next(splitter.split(
                    np.zeros((n_silo, 1)), silo_labels, groups=silo_groups))
            else:
                splitter = _SS(n_splits=1, test_size=0.2,
                               random_state=seed + s)
                tr_v, va_v = next(splitter.split(np.zeros((n_silo, 1))))

            probe_tr = {parent_rep: rep_dict_train[parent_rep][idx][tr_v]}
            _, fi_probe = create_hybrid(
                probe_tr, silo_labels[tr_v],
                groups=(silo_groups[tr_v] if silo_groups is not None else None),
                allocation_method=allocation_method,
                importance_method=importance_method,
                budget=min(probe_budget, len(tr_v)),
                internal_evaluator=internal_evaluator,
                model=downstream_model_instance,
                metric_fn=metric_fn, lower_is_better=lower_is_better)

            X_probe_tr = apply_feature_selection(
                {parent_rep: rep_dict_train[parent_rep][idx][tr_v]}, fi_probe)
            X_probe_va = apply_feature_selection(
                {parent_rep: rep_dict_train[parent_rep][idx][va_v]}, fi_probe)

            eval_factory = get_model_factories(
                [resolved_evaluator], task_type=silo_task_type,
                n_samples=len(tr_v)).get(resolved_evaluator)
            if eval_factory is None or X_probe_tr.shape[1] == 0:
                silo_scores[s] = 0.5
                metadata[s]['score_source'] = 'default_0.5_no_evaluator'
            else:
                ev = eval_factory()
                ev.fit(X_probe_tr, silo_labels[tr_v])
                silo_scores[s] = max(0.0, float(
                    ev.score(X_probe_va, silo_labels[va_v])))
                metadata[s]['score_source'] = 'held_out_within_silo'
                metadata[s]['n_score_train'] = int(len(tr_v))
                metadata[s]['n_score_heldout'] = int(len(va_v))
        except Exception:
            silo_scores[s] = 0.0
            metadata[s]['score_source'] = 'failed'

    # ── Budget allocation proportional to those scores ──────────────────
    total_score = sum(silo_scores.values())
    silo_budgets = {}
    for s in range(1, k_expanded + 1):
        if total_score > 0:
            silo_budgets[s] = max(1, int(
                budget * silo_scores[s] / total_score))
        else:
            silo_budgets[s] = max(1, budget // k_expanded)
        silo_budgets[s] = min(silo_budgets[s], len(virtual_silo_indices[s]))

    # ── Pass 2: rebuild each silo at the budget its score earned ─────────
    # The score-proportional split used to be computed, reported and then
    # thrown away: every silo was rebuilt at the same equal share. It is the
    # budget that is actually used now, which is what makes this condition
    # differ from an equal split at all.
    for s in range(1, k_expanded + 1):
        idx = virtual_silo_indices[s]
        parent_rep = metadata[s]['parent_rep']
        silo_rep_tr = {parent_rep: rep_dict_train[parent_rep][idx]}
        silo_labels = labels[idx]
        silo_groups = groups[idx] if groups is not None else None
        try:
            _, fi = create_hybrid(
                silo_rep_tr, silo_labels, groups=silo_groups,
                allocation_method=allocation_method,
                importance_method=importance_method,
                budget=silo_budgets[s],
                internal_evaluator=internal_evaluator,
                model=downstream_model_instance,
                metric_fn=metric_fn, lower_is_better=lower_is_better)
            silo_fis[s] = fi
        except Exception:
            silo_fis[s] = None

    # Build training hybrid: apply each silo's feature selection to full train
    # using only the parent rep (privacy: each silo only exposes its own rep)
    # Two virtual silos cut from the SAME representation can select the same
    # column of it, and the concatenation used to stack both copies: the
    # feature then counted twice in the model and twice against the budget.
    # A column already taken is dropped from the later silo, and each silo
    # records how many of its own selections were dropped that way.
    valid_silos = []
    taken = set()          # (parent_rep, column index) already in the stack
    deduped_fis = {}
    for s in range(1, k_expanded + 1):
        fi = silo_fis.get(s)
        if fi is None:
            continue
        parent_rep = metadata[s]['parent_rep']
        info = fi.get(parent_rep)
        if not isinstance(info, dict) or 'selected_indices' not in info:
            continue
        sel = np.asarray(info['selected_indices']).ravel()
        imp = info.get('importance_scores')
        keep = [i for i, c in enumerate(sel)
                if (parent_rep, int(c)) not in taken]
        metadata[s]['n_selected_before_dedup'] = int(len(sel))
        metadata[s]['n_dropped_as_duplicate'] = int(len(sel) - len(keep))
        if not keep:
            metadata[s]['n_selected'] = 0
            continue
        kept_cols = sel[keep]
        for c in kept_cols:
            taken.add((parent_rep, int(c)))
        fi_dedup = dict(fi)
        new_info = dict(info)
        new_info['selected_indices'] = np.asarray(kept_cols, dtype=int)
        if imp is not None and len(np.asarray(imp).ravel()) == len(sel):
            new_info['importance_scores'] = np.asarray(imp).ravel()[keep]
        new_info['n_features'] = int(len(kept_cols))
        fi_dedup[parent_rep] = new_info
        metadata[s]['n_selected'] = int(len(kept_cols))
        parent_dict = {parent_rep: rep_dict_train[parent_rep]}
        try:
            X_part = apply_feature_selection(parent_dict, fi_dedup)
            if X_part.shape[1] > 0:
                hybrid_train_parts.append(X_part)
                valid_silos.append(s)
                deduped_fis[s] = fi_dedup
        except Exception:
            continue
    silo_fis = deduped_fis or silo_fis

    if not hybrid_train_parts:
        raise ValueError("No virtual silo produced valid features")

    X_hybrid_train = np.hstack(hybrid_train_parts)

    # Apply same selections to test data (same parent-rep restriction)
    hybrid_test_parts = []
    for s in valid_silos:
        fi = silo_fis[s]
        parent_rep = metadata[s]['parent_rep']
        parent_dict_te = {parent_rep: rep_dict_test[parent_rep]}
        X_part = apply_feature_selection(parent_dict_te, fi)
        if X_part.shape[1] > 0:
            hybrid_test_parts.append(X_part)

    if not hybrid_test_parts:
        raise ValueError("No virtual silo produced valid test features")

    X_hybrid_test = np.hstack(hybrid_test_parts)

    if X_hybrid_train.shape[1] != X_hybrid_test.shape[1]:
        raise RuntimeError(
            f"fluid_within_kirby_across: training features "
            f"{X_hybrid_train.shape[1]} against test features "
            f"{X_hybrid_test.shape[1]} — the two sides applied different "
            f"selections.")

    return X_hybrid_train, X_hybrid_test, silo_budgets, silo_scores, metadata


def rank_silos_train_only(silo_fit_score, silo_names, seed=0):
    """Rank silos using only rows the outer test fold never sees.

    Several federated conditions report 'the best single silo' and pick it by
    whichever silo scored best on the outer test fold — a number nobody could
    have produced in advance, because it needs the answers. Any condition that
    is then compared against it is understated by an unknown amount.

    This gives the honest counterpart. `silo_fit_score` is a callable the
    caller supplies: given a silo name it fits on part of THAT silo's own
    training rows and returns a score on the part it held out, higher always
    better (negate an error before returning it). The caller keeps its
    test-fold pick as well and labels it a ceiling.

    Returns (best_silo_name, {silo: inner score}). A silo whose probe raises
    is scored None and cannot win.
    """
    scores = {}
    for sname in silo_names:
        try:
            v = silo_fit_score(sname)
            scores[sname] = None if v is None else float(v)
        except Exception:
            scores[sname] = None
    usable = {k: v for k, v in scores.items() if v is not None}
    best = max(usable, key=usable.get) if usable else None
    return best, scores


# =============================================================================
# IMPORTANCE PROFILE ANALYSIS (for fed04 cluster-aware importance)
# =============================================================================

def extract_importance_profile(feature_info):
    """Extract per-rep importance summary from a feature_info dict.

    Returns {rep_name: total_importance_score} — one scalar per rep
    summarizing how important that rep is in this hybrid.
    """
    profile = {}
    for rep_name, rep_info in feature_info.items():
        if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
            continue
        n_features = rep_info.get('n_features', len(rep_info['selected_indices']))
        importance = rep_info.get('importance_scores', np.ones(n_features))
        profile[rep_name] = float(np.sum(importance)) if len(importance) > 0 else 0.0
    return profile


def compare_importance_profiles(profiles_dict):
    """Compare importance profiles across clusters via Spearman correlation.

    Args:
        profiles_dict: {cluster_id: {rep_name: importance_score}}

    Returns:
        (mean_spearman, pairwise_dict) where pairwise_dict maps
        '(c1,c2)' → spearman_r for each cluster pair.
    """
    from scipy.stats import spearmanr as _spearmanr
    from itertools import combinations

    cluster_ids = sorted(profiles_dict.keys())
    if len(cluster_ids) < 2:
        # One cluster (or none) means there is no pair to correlate, so the
        # stability is undefined. This used to return 1.0 — the value that
        # means two clusters agree perfectly — and fed04 averaged it in with
        # the folds that had a real measurement. MEASURED on caco2, 200
        # molecules: fold 2 had one cluster above the 20-sample minimum and
        # contributed 1.000, fold 1 measured 0.829, and the reported
        # mean_profile_stability of 0.914 was the average of a measurement
        # and a placeholder. None (JSON null) is what the <3-representation
        # branch below already returns for the same reason, and both callers
        # (fed04, fed05's C3) already print 'n/a' for it and drop it from
        # their fold averages.
        return None, {}

    # Get union of all rep names
    all_reps = sorted(set().union(
        *[set(p.keys()) for p in profiles_dict.values()]))
    if len(all_reps) < 3:
        # Stability is undefined with <3 reps — return None (JSON null), not
        # NaN (which is invalid JSON and trips the "no NaN" requirement).
        return None, {}

    # Build importance vectors (missing reps get 0)
    vectors = {}
    for cid in cluster_ids:
        vectors[cid] = np.array(
            [profiles_dict[cid].get(r, 0.0) for r in all_reps])

    # Pairwise Spearman
    pairwise = {}
    correlations = []
    for c1, c2 in combinations(cluster_ids, 2):
        try:
            r, _ = _spearmanr(vectors[c1], vectors[c2])
            r = float(r)
        except Exception:
            r = float('nan')
        # A pair whose correlation is undefined — one cluster's importance
        # vector is constant, so there is no rank to correlate — was stored
        # as NaN. NaN is not valid JSON and it is the same defect the mean
        # below already avoids by returning None. fed04 on herg_ki wrote 130
        # of them into folds[].profile_stability.pairwise.
        pairwise[f'({c1},{c2})'] = None if np.isnan(r) else r
        if not np.isnan(r):
            correlations.append(r)

    # None (JSON null) when no pairwise correlation is computable (e.g. all
    # clusters' importance vectors are constant) — never NaN.
    mean_r = float(np.mean(correlations)) if correlations else None
    return mean_r, pairwise


def _feature_label(rep_name, index):
    """A readable label for one selected feature of one representation.

    PDV and the descriptor-style representations carry real descriptor names,
    so a selected feature comes back as e.g. 'pdv:MolLogP'. A fingerprint bit
    or an embedding dimension has no name, so it comes back as 'ecfp4[1461]'
    or 'molformer[288]'. Either way the label is stable across clusters,
    folds, datasets and runs, which is what makes the selections comparable.
    """
    try:
        from kirby.importance.hybrid import _resolve_rep_type
        interp_type, name_fn = _resolve_rep_type(rep_name)
    except Exception:
        interp_type, name_fn = 'embedding_dim', None
    if interp_type == 'named' and name_fn is not None:
        try:
            names = name_fn(int(index) + 1)
            if len(names) > index:
                return f"{rep_name}:{names[int(index)]}"
        except Exception:
            pass
    return f"{rep_name}[{int(index)}]"


def extract_selected_features(feature_info):
    """The identity of every feature one hybrid selected, not just the counts.

    extract_importance_profile answers 'how much of the budget went to each
    representation'. This answers 'which features did it take', which is the
    thing two clusters can agree or disagree on at the level of a single
    descriptor or fingerprint bit.

    Returns {
        'labels': [str], every selected feature, ordered by importance,
                  highest first — 'pdv:MolLogP', 'ecfp4[1461]', ...
        'importances': [float], aligned with labels,
        'per_rep': {rep_name: [{'index': int, 'label': str,
                                'importance': float}, ...]},
        'n_selected': int,
    }
    """
    rows = []
    per_rep = {}
    for rep_name, rep_info in feature_info.items():
        if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
            continue
        idx = np.asarray(rep_info['selected_indices']).ravel()
        imp = rep_info.get('importance_scores')
        imp = (np.ones(len(idx)) if imp is None
               else np.asarray(imp, dtype=float).ravel())
        if len(imp) != len(idx):
            imp = np.ones(len(idx))
        entries = []
        for i, sc in zip(idx, imp):
            label = _feature_label(rep_name, i)
            entries.append({'index': int(i), 'label': label,
                            'importance': float(sc)})
            rows.append((label, float(sc)))
        per_rep[rep_name] = sorted(entries, key=lambda e: -e['importance'])
    rows.sort(key=lambda r: -r[1])
    return {
        'labels': [r[0] for r in rows],
        'importances': [r[1] for r in rows],
        'per_rep': per_rep,
        'n_selected': len(rows),
    }


def compare_selected_features(selections_dict):
    """How far two clusters' selected features actually overlap.

    Args:
        selections_dict: {cluster_id: extract_selected_features(...) output}

    Returns a dict with, for a set of K clusters each selecting a budget of
    features:
        'pairwise_jaccard': {'(c1,c2)': float} — of the features either
            cluster took, the fraction both took. 1.0 means identical
            selections, 0.0 means no feature in common.
        'mean_jaccard': float or None
        'shared_by_all': [str] — features every cluster took. These are what
            the dataset wants everywhere.
        'n_shared_by_all': int
        'unique_to_one': {cluster_id: [str]} — features exactly one cluster
            took. These are what that region of chemical space wants and the
            rest of the dataset does not.
        'n_unique_to_one': {cluster_id: int}
        'selection_counts': {label: int} — how many clusters took each
            feature, so the whole distribution is readable, not just its two
            ends.
    """
    from itertools import combinations

    cluster_ids = sorted(selections_dict.keys())
    sets = {c: set(selections_dict[c].get('labels', [])) for c in cluster_ids}
    if len(cluster_ids) < 2:
        # Nothing to compare. 'shared_by_all' would be the one cluster's own
        # selection, which reads as agreement when there is no second cluster
        # to agree with, so it is empty and n_clusters says why.
        return {'pairwise_jaccard': {}, 'mean_jaccard': None,
                'shared_by_all': [], 'n_shared_by_all': None,
                'unique_to_one': {}, 'n_unique_to_one': {},
                'selection_counts': {},
                'n_clusters': len(cluster_ids),
                'undefined_reason': 'fewer than two clusters produced a selection'}

    pairwise, vals = {}, []
    for c1, c2 in combinations(cluster_ids, 2):
        union = sets[c1] | sets[c2]
        j = (len(sets[c1] & sets[c2]) / len(union)) if union else 0.0
        pairwise[f'({c1},{c2})'] = float(j)
        vals.append(j)

    counts = {}
    for c in cluster_ids:
        for lab in sets[c]:
            counts[lab] = counts.get(lab, 0) + 1

    shared_all = sorted(l for l, n in counts.items() if n == len(cluster_ids))
    unique_to_one = {}
    for c in cluster_ids:
        unique_to_one[c] = sorted(l for l in sets[c] if counts[l] == 1)

    return {
        'n_clusters': len(cluster_ids),
        'pairwise_jaccard': pairwise,
        'mean_jaccard': float(np.mean(vals)) if vals else None,
        'shared_by_all': shared_all,
        'n_shared_by_all': len(shared_all),
        'unique_to_one': unique_to_one,
        'n_unique_to_one': {c: len(v) for c, v in unique_to_one.items()},
        'selection_counts': counts,
    }


# =============================================================================
# ANTIBODY CLUSTERING (for fed05 — replaces FLuID's Morgan FP K-Means)
# =============================================================================

def cluster_antibody_silos(rep_dict, k=3, seed=42, cluster_rep='esm2'):
    """K-Means clustering on protein embeddings for antibody sequences.

    Replaces FLuID's Morgan FP K-Means for the antibody domain.
    Uses ESM2 embeddings (1280D) by default, with PCA reduction.

    Args:
        rep_dict: {rep_name: np.ndarray} from load_data.
        k: number of clusters.
        seed: random seed.
        cluster_rep: which rep to cluster on (default 'esm2').

    Returns:
        silo_indices: {cluster_id: np.array of indices} (1-indexed, 1..k)
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if not rep_dict:
        raise ValueError("cluster_antibody_silos: rep_dict is empty")

    # Find clustering rep
    if cluster_rep in rep_dict:
        X = rep_dict[cluster_rep]
    else:
        # Fallback: try other large reps
        for fallback in ['esmc', 'prot_t5', 'ablang2', 'antiberty']:
            if fallback in rep_dict:
                X = rep_dict[fallback]
                cluster_rep = fallback
                break
        else:
            # FED-28: previously this hstacked EVERY rep ('concat'), which is a
            # memory bomb for high-dim antibody embeddings (ESM2 1280D × ablang2
            # × prot_t5 ...) AND silently changes the clustering geometry vs the
            # single-rep path. Instead pick the single highest-dimensional
            # available rep deterministically (PCA below reduces it anyway).
            cluster_rep = max(sorted(rep_dict.keys()),
                              key=lambda r: rep_dict[r].shape[1])
            X = rep_dict[cluster_rep]
            warnings.warn(
                f"cluster_antibody_silos: none of the preferred reps "
                f"(esm2/esmc/prot_t5/ablang2/antiberty) present; clustering on "
                f"single highest-dim rep '{cluster_rep}' "
                f"({X.shape[1]}D) instead of concatenating all reps.")

    # PCA reduction if high-dimensional
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = max(1, min(50, X_scaled.shape[1], X_scaled.shape[0] - 1))
    if n_components < X_scaled.shape[1]:
        X_reduced = PCA(n_components=n_components,
                        random_state=seed).fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    km = _KMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = km.fit_predict(X_reduced)

    silo_indices = {}
    for c in range(k):
        silo_indices[c + 1] = np.where(assignments == c)[0]

    return silo_indices


# =============================================================================
# IMPORTANCE PROFILE TRANSFER (for fed03 cross-task transfer)
# =============================================================================

def importance_profile_transfer(source_feature_info, target_rep_dict,
                                target_labels, budget=100):
    """Transfer importance profile from source silo to target silo.

    Uses source's per-rep importance rankings to guide feature selection on
    target's data. Only feature INDEX preferences (which features in which
    reps) are transferred — no raw data crosses silo boundaries.

    Protocol:
        1. Source silo shares feature_info (selected_indices + importance per rep)
        2. Target silo selects the same feature indices from its own reps
        3. If a rep in source doesn't exist in target, it's skipped
        4. Budget is distributed proportional to source importance

    Args:
        source_feature_info: feature_info dict from create_hybrid on source data.
        target_rep_dict: {rep_name: np.ndarray} for target silo.
        target_labels: labels for target silo (for optional re-ranking).
        budget: max features to select.

    Returns:
        (X_target_hybrid, transferred_fi, transfer_stats)
    """
    from kirby.hybrid import apply_feature_selection

    transferred_fi = {}
    transfer_stats = {'matched_reps': 0, 'skipped_reps': 0, 'total_features': 0}

    for rep_name, rep_info in source_feature_info.items():
        if not isinstance(rep_info, dict) or 'selected_indices' not in rep_info:
            continue

        if rep_name not in target_rep_dict:
            transfer_stats['skipped_reps'] += 1
            continue

        source_indices = rep_info['selected_indices']
        source_importance = rep_info.get('importance_scores',
                                         np.ones(len(source_indices)))
        target_n_features = target_rep_dict[rep_name].shape[1]

        # Only keep indices that are valid for target's feature dimensions
        valid = source_indices < target_n_features
        indices = source_indices[valid]
        importance = source_importance[valid] if len(source_importance) == len(source_indices) else np.ones(len(indices))

        # Trim to budget share
        remaining = budget - transfer_stats['total_features']
        if remaining <= 0:
            break
        if len(indices) > remaining:
            order = np.argsort(importance)[::-1][:remaining]
            indices = indices[order]
            importance = importance[order]

        transferred_fi[rep_name] = {
            'selected_indices': np.array(indices, dtype=int),
            'importance_scores': np.array(importance),
            'n_features': len(indices),
        }
        transfer_stats['matched_reps'] += 1
        transfer_stats['total_features'] += len(indices)

    if transfer_stats['total_features'] == 0:
        return None, transferred_fi, transfer_stats

    X_target = apply_feature_selection(target_rep_dict, transferred_fi)
    return X_target, transferred_fi, transfer_stats


# =============================================================================
# RUN INFRASTRUCTURE (follows noise_helpers.run_all pattern)
# =============================================================================

def _import_experiment(exp_id: str):
    """Dynamically import a federated experiment module by ID."""
    module_map = {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    }
    mod_name = module_map[exp_id]
    import importlib
    return importlib.import_module(f'experiments.federated.{mod_name}')


def _experiment_full_name(exp_id: str) -> str:
    """Map short ID to the experiment directory name."""
    name_map = {
        'fed01': 'fed01_noise_heterogeneity',
        'fed02': 'fed02_rep_diversity_silos',
        'fed03': 'fed03_synthetic_private',
        'fed04': 'fed04_cluster_aware_importance',
        'fed05': 'fed05_antibody_federated',
    }
    return name_map.get(exp_id, exp_id)


def run_all(dataset_name: str, output_dir: str,
            experiments: list = None, config: dict = None,
            rep_names: list = None,
            pruning_mode: str = 'default_pruning',
            force_rerun: bool = False):
    """Load data once and run all federated experiments.

    Args:
        dataset_name: Dataset name (e.g. 'herg_ki').
        output_dir: Base output directory for results JSON files.
        experiments: List of experiment IDs to run (default: all).
        config: Config dict override (default: FEDERATED_CONFIG).
        rep_names: Explicit rep list override.
        pruning_mode: One of PRUNING_MODES. 'default_pruning' = legacy
            single-stage prune. 'aggressive_pruning' = F16-validated multi-
            stage pruning (checkpoints=[25,50,100], td=0.5) applied globally
            to all models, no n_samples gate. Most fed experiments use
            create_hybrid directly per silo and bypass the cache, so this
            only affects the precompute_hybrid_searches calls below.
        force_rerun: recompute every experiment. Default False RESUMES: an
            experiment whose <output>/<experiment>/<dataset>.json exists,
            holds a measurement and carries an input_fingerprint matching the
            data just loaded is skipped -- the same rule as noise and
            similarity run_all. Added 2026-09-25; before, every resubmission
            re-ran every experiment from the start.
    """
    if not _FLUID_AVAILABLE:
        print("ERROR: FLuID not available. Federated experiments require FLuID.")
        print(f"Set FLUID_PATH env var (current: {_fluid_path}) or install FLuID.")
        sys.exit(1)

    if config is None:
        config = dict(_get_federated_config())
    if experiments is None:
        experiments = list(SINGLE_DATASET_EXPERIMENTS)

    # Check if ALL requested experiments are self-loading (skip data load)
    all_self_loading = all(e in SELF_LOADING_EXPERIMENTS for e in experiments)

    # Resolve rep list (skip if all experiments load their own data or Lhasa)
    if not all_self_loading and dataset_name not in ('herg', 'lhasa_herg'):
        if rep_names is None:
            ds_reps = DATASET_REPS.get(dataset_name)
            if ds_reps is None:
                raise ValueError(
                    f"No rep list for '{dataset_name}'. Pass rep_names "
                    f"explicitly or add it to DATASET_REPS in "
                    f"complexity_helpers.py."
                )
            rep_names = ds_reps['general']
    else:
        rep_names = rep_names or []

    print(f"\n{'=' * 70}")
    print(f"FEDERATED EXPERIMENTS: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Config:      {config['n_splits']} fold, {config['models']}, "
          f"")
    print(f"Reps ({len(rep_names)}): {', '.join(rep_names)}")
    print(f"Experiments: {experiments}")
    print(f"Output:      {output_dir}")

    # ── Load data ONCE (skip for self-loading experiments) ──
    t0 = time.time()
    if all_self_loading:
        print(f"\nSkipping load_data — all experiments load their own data")
        data = {'reps': {}, 'labels': np.array([]),
                'domain': 'federated', 'augmentations': None}
    elif dataset_name in ('herg', 'lhasa_herg'):
        print(f"\nLoading Lhasa hERG data...")
        data = load_lhasa_herg_data(rep_names=rep_names)
        print(f"  Loaded in {time.time() - t0:.1f}s")
    else:
        print(f"\nLoading {dataset_name}...")
        # Pass the resolved rep list AS the preset (comma-joined) rather than
        # 'all'. get_representations() generates only the named reps, instead of
        # generating the full 'all' set and filtering afterwards. This avoids
        # triggering heavy/optional neural reps (which may be slow or fail
        # locally) when only a small cheap rep set was requested — and on the
        # server it generates exactly the DATASET_REPS general list. Equivalent
        # results (same per-rep arrays, same cache files), strictly less work.
        _preset = ','.join(rep_names) if rep_names else 'all'
        data = load_data(dataset_name, rep_preset=_preset, rep_names=rep_names)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # NaN guard
        reps_data = data.get('reps', {})
        for rname, X in list(reps_data.items()):
            nan_count = int(np.isnan(X).sum())
            if nan_count > 0:
                pct = 100 * nan_count / X.size
                print(f"  WARNING: {rname} has {nan_count} NaN values "
                      f"({pct:.1f}%) -- imputing 0")
                reps_data[rname] = np.nan_to_num(X, nan=0.0)

        # ── Load augmentation reps ──
        aug_rep_names = DATASET_REPS.get(dataset_name, {}).get(
            'augmentation', [])
        if aug_rep_names and _smoke_active():
            # Test-only: skip augmentation reps under KIRBY_SMOKE_N. They are
            # optional enrichment and several (e.g. pharmacophore3d,
            # coulomb_matrix) require slow 3D-conformer generation over the FULL
            # dataset before the row cap applies — minutes per smoke run with no
            # value at tiny n. No-op when the env var is unset (server unchanged).
            print(f"  [SMOKE] skipping {len(aug_rep_names)} augmentation reps")
            data['augmentations'] = None
            aug_rep_names = []
        if aug_rep_names:
            missing = [r for r in aug_rep_names if r not in reps_data]
            if missing:
                print(f"  Loading {len(missing)} augmentation reps: "
                      f"{missing}")
                aug_data = load_data(dataset_name,
                                     rep_preset=','.join(missing),
                                     rep_names=missing)
                aug_reps = {}
                for rn, arr in aug_data['reps'].items():
                    if int(np.isnan(arr).sum()) > 0:
                        arr = np.nan_to_num(arr, nan=0.0)
                    aug_reps[rn] = arr
                data['augmentations'] = aug_reps
            else:
                data['augmentations'] = {
                    r: reps_data[r] for r in aug_rep_names}
            print(f"  Augmentations ({len(data['augmentations'])}): "
                  f"{', '.join(f'{k}={v.shape[1]}' for k, v in data['augmentations'].items())}")
        else:
            data['augmentations'] = None

        # ── Resolve pruning_mode + apply n_samples gate ──
        from experiments._shared import resolve_pruning_mode, apply_pruning_gate
        _sched, _ovr_template, _min_n = resolve_pruning_mode(pruning_mode)
        _model_overrides = apply_pruning_gate(
            _ovr_template, len(data['labels']), _min_n)
        print(f"\n  Pruning mode: {pruning_mode}")
        if _model_overrides is not None:
            print(f"    overrides active for: {sorted(_model_overrides.keys())}")
        elif _ovr_template is not None:
            print(f"    overrides gated off (n={len(data['labels'])} < {_min_n})")

        # ── Precompute hybrid searches (general reps) ──
        # precompute_hybrid_searches catches a per-aug_condition exception,
        # stores None under that key, and appends one record to `.failures`.
        # Those records are copied into data['precompute_errors'][tier] here
        # and written to _precompute_errors/<dataset>.json below, so a failed
        # aug_condition is named in the output, not only in a printed line.
        _task_type = data['task_type']
        precompute_errors = data.setdefault('precompute_errors', {})
        from experiments._shared import precompute_failures
        if len(reps_data) >= 2:
            from experiments._shared import precompute_hybrid_searches
            print(f"\n  Precomputing hybrid searches (general, {len(reps_data)} reps)...")
            # Test-only: under KIRBY_SMOKE_N narrow the precompute to the
            # downstream models the federated sweep will actually ask for and
            # to the two-evaluator smoke list. precompute_hybrid_searches
            # otherwise scores all five DOWNSTREAM_MODELS against all five
            # internal evaluators, which is 125 recipes x 5 folds for a run
            # that only ever reads the 'rf' row. No-op when the env var is
            # unset, so the server search space is unchanged.
            _pre_models = None
            _pre_evals = None
            if _smoke_active():
                _pre_models = sorted({
                    fed_to_kirby_model(m)
                    for m in config.get('fed_models', FED_MODELS)})
                _pre_evals = config.get('internal_evaluators')
                print(f"    [SMOKE] precompute models={_pre_models} "
                      f"evaluators={_pre_evals}")
            precompute_results = precompute_hybrid_searches(
                reps=reps_data,
                labels=data['labels'],
                groups=data.get('groups'),
                config=config,
                augmentations=data.get('augmentations'),
                task_type=_task_type,
                domain=data.get('domain', 'molecular'),
                dataset_name=dataset_name, tier='general',
                adaptive_pruning_schedule=_sched,
                model_pruning_aggressiveness=_model_overrides,
                models=_pre_models,
                internal_evaluators=_pre_evals,
            )
            data['precomputed_searches'] = precompute_results
            precompute_errors['general'] = precompute_failures(
                precompute_results)
        else:
            data['precomputed_searches'] = {}
            precompute_errors['general'] = []

    # ── Record precompute failures ──
    # Written before any experiment runs, so the record survives a run that
    # dies part-way. Each record carries tier, dataset, aug_condition,
    # error_type, error and traceback. Empty when the branch above did not run
    # (self-loading experiments, or the Lhasa hERG path: no precompute pass).
    _precompute_errors = data.setdefault('precompute_errors', {})
    precompute_error_audit = {
        'dataset': dataset_name,
        'n_failures': sum(len(v) for v in _precompute_errors.values()),
        'by_tier': {k: list(v) for k, v in _precompute_errors.items()},
    }
    if precompute_error_audit['n_failures']:
        print(f"  Precompute failures recorded: "
              f"{precompute_error_audit['n_failures']}")
    save_experiment_results(precompute_error_audit, output_dir,
                            '_precompute_errors', dataset_name)

    # What each result file below is stamped with: the row count, the label and
    # group hashes, the split count and the representation list, and what the
    # resume check below compares a file on disk against (2026-09-25).
    # ⛔ When every requested experiment loads its own data (all_self_loading)
    # `data` is a stand-in with no rows, so the fingerprint cannot see a change
    # in the data those experiments load; --force-rerun is the way to redo them.
    _fingerprint = input_fingerprint(data, config)

    from experiments._shared import (
        experiment_result_is_empty, fingerprint_mismatch)

    # ── Run experiments ──
    results = {}
    for exp_id in experiments:
        # Resume, the same rule as noise_helpers.run_all and
        # similarity_helpers.run_all: skip an experiment whose output exists,
        # holds a measurement, and was computed from the same data
        # (input_fingerprint). A file that will not parse, records nothing, or
        # whose fingerprint differs is recomputed. force_rerun recomputes
        # always. Added 2026-09-25.
        _out_json = os.path.join(
            output_dir, _experiment_full_name(exp_id), f"{dataset_name}.json")
        if not force_rerun and os.path.exists(_out_json) \
                and os.path.getsize(_out_json) > 0:
            try:
                with open(_out_json) as _f:
                    _prev = json.load(_f)
                _why = ('records no measurement'
                        if experiment_result_is_empty(_prev,
                                                      classify_unmarked=True)
                        else fingerprint_mismatch(
                            _prev.get('input_fingerprint'), _fingerprint))
            except (ValueError, OSError) as _e:
                _why = f'could not be read ({_e})'
            if _why:
                print(f"  RERUN {exp_id}: existing output {_why} — "
                      f"recomputing ({_out_json})")
            else:
                print(f"  RESUME {exp_id}: already done — skipping "
                      f"({_out_json})")
                results[exp_id] = {'resumed': True}
                continue

        print(f"\n{'~' * 70}")
        t_exp = time.time()

        # An experiment that cannot be imported is a FAILED experiment, not a
        # skip (2026-09-25): recorded with its traceback so run_dataset.py
        # writes it to <output>/_errors/ and exits 1.
        try:
            mod = _import_experiment(exp_id)
        except Exception as e:
            import traceback
            print(f"  {exp_id} FAILED: import failed "
                  f"({type(e).__name__}: {e})")
            traceback.print_exc()
            results[exp_id] = {'error': str(e) or type(e).__name__,
                               'error_kind': f'import:{type(e).__name__}',
                               'traceback': traceback.format_exc()}
            continue

        try:
            # ── Federated sweep: FLuID's RF/XGBoost/MLP families × seeds ──
            # Each experiment reads config['fed_model']/['fed_seed']; we run the
            # whole experiment per (model, seed), then average the seeds into
            # mean/std (error bars). Silos stay at 5; the expensive rep
            # precompute above is shared across all runs.
            _fed_models = config.get('fed_models', FED_MODELS)
            _fed_seeds = config.get('fed_seeds', FED_SEEDS)
            per_model = {}
            for _fed_model in _fed_models:
                _seed_runs = []
                for _fed_seed in _fed_seeds:
                    _run_cfg = dict(config, fed_model=_fed_model,
                                    fed_seed=_fed_seed)
                    print(f"  >> {exp_id}: model={_fed_model} seed={_fed_seed}")
                    _seed_runs.append(
                        mod.run(data, _run_cfg, output_dir, dataset_name))
                per_model[_fed_model] = {
                    'per_seed': _seed_runs,
                    'aggregated': aggregate_seed_runs(_seed_runs),
                }
            exp_result = {
                'sweep': {'models': _fed_models, 'seeds': _fed_seeds,
                          'silos': 5,
                          'note': 'per_model.aggregated = mean/std over seeds'},
                'per_model': per_model,
            }
            results[exp_id] = exp_result

            exp_full_name = _experiment_full_name(exp_id)
            save_experiment_results(
                exp_result, output_dir, exp_full_name, dataset_name,
                fingerprint=_fingerprint)
            print(f"  {exp_id} done in {time.time() - t_exp:.1f}s "
                  f"({len(_fed_models)}x{len(_fed_seeds)} runs)")

        except Exception as e:
            print(f"  {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'error': str(e) or type(e).__name__,
                               'error_kind': type(e).__name__,
                               'traceback': traceback.format_exc()}

        except SystemExit as e:
            # SystemExit is not an Exception; a library calling sys.exit()
            # would end the run with nothing said. Same handling as
            # complexity_helpers.run_all (complexity_helpers.py:1242-1263),
            # added 2026-09-25.
            print(f"  {exp_id} FAILED: a library called sys.exit({e.code}) "
                  f"instead of raising. Caught so the remaining experiments "
                  f"still run.")
            import traceback
            traceback.print_exc()
            results[exp_id] = {
                'error': f'SystemExit({e.code}) — a library exited the process',
                'error_kind': 'SystemExit',
                'traceback': traceback.format_exc()}

        except KeyboardInterrupt:
            raise

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL DONE: {dataset_name} -- {len(results)} experiments "
          f"in {total:.0f}s")
    print(f"{'=' * 70}")

    # Underscore key so it does not collide with an experiment ID.
    results['_precompute_errors'] = precompute_error_audit
    return results
