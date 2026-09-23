"""Canonical label-noise specification for KIRBy.

ONE definition of injected label noise, shared by the hybrid search (as a cached
precompute) and by every noise experiment. Before this module, n07/n09/n11/n12/n13
each carried a private `_inject_noise` with FOUR incompatible meanings (n07 corrupted
a fraction of rows at 30% of the label range; n09/n11/n12 added Gaussian noise to every
row scaled by the label range; n13 had four distributions). A blueprint searched under
one definition and used by an experiment injecting another is fitted to labels that
exist nowhere in that run — and nothing crashes. Hence: one definition, one module.

Backed by `noiseInject` (Punt, J. Cheminform. 2026; Zenodo 10.5281/zenodo.20532710),
which is also the source of the retention metrics already in `kirby.noise_robustness`.

TWO PROPERTIES THIS MODULE EXISTS TO GUARANTEE
----------------------------------------------
1. REPRODUCIBILITY. `NoiseInjectorRegression.__init__` builds `self.rng =
   np.random.RandomState(random_state)` ONCE, and every `inject()` call draws from it,
   advancing the state — so calling `inject()` twice on one injector yields DIFFERENT
   noise. We therefore construct a FRESH injector on every call. Without this the
   cached noisy search is not reproducible and the whole cache is void.

2. FOLD-INDEPENDENCE. The realisation is drawn ONCE over the full label column, then
   subset per fold. A given row therefore always receives the same corruption no matter
   which fold it lands in, and `noisy_labels_hash()` gives experiments a single value to
   assert against the search they are consuming.

WHAT A NOISE LEVEL MEANS
------------------------
`target_effective_noise` is read through `dose_units`, mirroring `DoseUnits { Spread,
Label }` in `qsar_qm_models/rust/src/main.rs:239` so the Python and Rust halves of the
study can be compared line by line:

    'spread'  the level is a fraction of the clean label SD. The dose delivered is
              level * SD(y). Use this where no assay error has been published to
              anchor to — QM9, the reaction sets, the antibody sets.
    'label'   the level IS the dose, in the label's own units. Use this where the
              literature gives a figure for the endpoint: 0.54 log units for pKi
              (Kramer et al. 2012), 0.35 log10 for Caco-2 (computed from Bentz et al.
              2013 Table 6), and so on. The per-dataset grids live in
              `tests/experiments/dataset_policy.py`.

Either way the noise DELIVERED is a root-mean-square, and both the dose in label units
and the dose as a fraction of the label spread are recorded on every row — see
`noise_provenance_row`.

Censoring is the exception and takes its own axis: its level is the FRACTION of labels
clipped at the assay limit, not a dose. It is not dose-matched and not zero-mean.

CHANGED IN noiseInject 1.0.0, AND THIS VOIDS THE OLD CACHE. The level used to be
mean |Δy| / SD -- the FIRST moment -- found by binary search. R² and RMSE are
second-moment quantities, so the second moment is what has to be matched: at identical
root-mean-square noise, mean|ε|/rms is 0.797 for a Gaussian but 0.642 for Student-t at
ν = 3, so the old definition handed the heavy-tailed conditions up to 24% more actual
noise at the same nominal level. The search was also run against a moving target -- one
injector was re-used across all 20 of its iterations, so every iteration drew fresh
noise. Because the strategy names changed with it, `spec_id` and therefore `tier` change
too, so cached rows from the old definition are a clean MISS rather than a wrong hit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

_INSTALL_HINT = (
    "noiseInject is required for label-noise experiments but is not installed.\n"
    "It is not on PyPI. Install it from the source repo:\n"
    "    pip install git+ssh://git@github.com/adpunt/noise_inject.git\n"
    "(or, for local development:  pip install -e ~/repos/NoiseInject )"
)


def _require_noiseinject():
    """Import noiseInject with an actionable error.

    A bare ImportError deep inside a SLURM job is a wasted queue slot; say what to
    run instead.
    """
    try:
        import noiseInject  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

# Regression conditions from noiseInject. Every one of the dose-matched ten delivers
# the same AMOUNT of noise at a given level, so a difference between them is a
# difference of shape. The six they replace (legacy, quantile, threshold, outlier,
# hetero, valprop) were one condition at six doses, and four of them assumed error
# depends on the measured value, which has been directly tested on 16,844 repeat
# measurements and refuted.
#
# These strings are the registry keys of `noiseInject.CONDITIONS` and must stay
# identical to them.
SHAPE_STRATEGIES = ('gaussian', 'student_t_nu10', 'student_t_nu5', 'student_t_nu3',
                    'laplace')
OUTLIER_STRATEGIES = ('outlier_p01', 'outlier_p05', 'outlier_p10')

# The same mechanism with the unselected molecules left exactly alone. Under the
# three above the multiplier is 3.0 on the selected fraction and 1.0 on the rest,
# so every label moves and the selected fraction carries only part of the injected
# error -- at p=0.10 the mean squared multiplier is 0.10*9 + 0.90*1 = 1.8 and the
# widened tenth therefore carries 0.9/1.8, exactly half, fixed by p and lam. So a
# method that removes precisely the molecules the mask names leaves the other half
# behind, and a null result cannot tell "the detector cannot find them" apart from
# "finding them does not help".
#
# Here the multiplier off the selected fraction is 0.0. Measured 2026-09-23 on
# 2,000 Gaussian labels at a dose of half their spread: outlier_only_p10 moved 208
# of 2,000 labels against 2,000 of 2,000, and the widened molecules carried 100.0%
# of the squared error against 47.8%.
#
# BOTH families stay in the sweep. The pair is the comparison.
OUTLIER_ONLY_STRATEGIES = ('outlier_only_p01', 'outlier_only_p05',
                           'outlier_only_p10')

# These two need one integer scaffold-group id per molecule. Build it with
# `assign_noise_groups` below, NOT with the fold-geometry grouping.
GROUPED_STRATEGIES = ('grouped_wider', 'grouped_shifted')

# Censoring's level is the fraction of labels clipped, not a dose.
CENSORING_STRATEGIES = ('censoring',)

# The eleven that need nothing but a label column.
UNGROUPED_STRATEGIES = (SHAPE_STRATEGIES + OUTLIER_STRATEGIES
                        + OUTLIER_ONLY_STRATEGIES)

# The thirteen that are dose-matched. Censoring is not one of them.
DOSE_MATCHED_STRATEGIES = UNGROUPED_STRATEGIES + GROUPED_STRATEGIES

REGRESSION_STRATEGIES = DOSE_MATCHED_STRATEGIES + CENSORING_STRATEGIES

CLASSIFICATION_STRATEGIES = ('uniform', 'class_imbalance', 'binary_asymmetric',
                             'instance_noise', 'class_dependent', 'confusion_directed')

DOSE_UNITS = ('spread', 'label')

# Conditions whose per-molecule noise scale is the same for every molecule. For these
# "which region of the label space is unreliable" is UNDEFINED, not zero, so a rank
# correlation against that scale must never be reported. Experiments that ask where
# the noise is must use a condition outside this set.
# `grouped_shifted` belongs here too: it moves each affected group by one offset,
# so the per-molecule scale is the same everywhere. noiseInject says the same at
# core.py:84 (_CONSTANT_SCALE_STRATEGIES), added 2026-08-27 in 82b60be.
CONSTANT_SCALE_STRATEGIES = SHAPE_STRATEGIES + ('grouped_shifted',)

# REMOVED 2026-09-12. `DEFAULT_NOISE_LEVELS = (0.05, 0.10, 0.20)` used to sit
# here, described as "the noise grid the cached precompute is built for". That
# precompute was deleted on 2026-09-11 and no code ever read this constant --
# grep over src/ and tests/ returns nothing. It is gone rather than updated,
# because a second copy of a level grid is how the suite ended up with two that
# disagreed. The grid every noise experiment sweeps is
# experiments.dataset_policy._DEFAULT_NOISE_LEVELS; the levels n15 sweeps are
# experiments.noise.noise_helpers.NOISY_SEARCH_LEVELS.
DEFAULT_STRATEGY_REG = 'gaussian'
DEFAULT_STRATEGY_CLF = 'uniform'


def _freeze_params(params: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, float], ...]:
    """A hashable, order-stable form of the condition parameter overrides."""
    if not params:
        return ()
    return tuple(sorted((str(k), v) for k, v in params.items()))


@dataclass(frozen=True)
class NoiseSpec:
    """A frozen, hashable description of one injected-noise condition.

    strategy               noiseInject condition name (a key of noiseInject.CONDITIONS
                           for regression; a classification strategy otherwise)
    target_effective_noise the noise level, read through `dose_units`. For regression
                           that is the delivered root-mean-square noise, either as a
                           fraction of the label SD ('spread') or in the label's own
                           units ('label'). For `censoring` it is the fraction of
                           labels clipped. For classification it is the target flip
                           rate.
    noise_seed             seed for the noise draw ONLY. Deliberately separate from the
                           search's `seed`, which drives the CV fold boundaries and sits
                           in the cache identity — keying noise off `seed` would move
                           every fold whenever the noise draw changed.
    selection_seed         seed for WHICH molecules are hit, separate from the seed
                           for how hard. A level sweep varies `noise_seed` per level,
                           so without this the affected molecules change at every
                           level and the curve mixes more dose with a different set of
                           molecules. Pin it to (condition, fold) and the levels nest.
                           None means noiseInject falls back to `noise_seed`, which is
                           the pre-2026-08-27 behaviour.
    dose_units             'spread' or 'label'. Ignored for censoring and for
                           classification.
    params                 overrides for the condition's own parameters (lam,
                           group_fraction, rho, p, side), as sorted (name, value) pairs.
                           Empty means the registry defaults, which are the sourced
                           ones: lam = 3 (Avdeef 2019), group_fraction = 0.2 (a stated
                           choice, no published number exists), rho = 0.62 (Bentz et al.
                           2013 Table 7), p = 0.01/0.05/0.10 (Hampel 2001).
    """
    strategy: str
    target_effective_noise: float
    noise_seed: int = 0
    dose_units: str = 'spread'
    params: Tuple[Tuple[str, float], ...] = ()
    selection_seed: Optional[int] = None

    def __post_init__(self):
        if self.strategy not in REGRESSION_STRATEGIES + CLASSIFICATION_STRATEGIES:
            raise ValueError(
                f"Unknown noise strategy {self.strategy!r}. "
                f"Regression: {REGRESSION_STRATEGIES}. "
                f"Classification: {CLASSIFICATION_STRATEGIES}.")
        if self.dose_units not in DOSE_UNITS:
            raise ValueError(
                f"dose_units must be one of {DOSE_UNITS}; got {self.dose_units!r}.")
        level = float(self.target_effective_noise)
        if level < 0.0:
            raise ValueError(
                f"target_effective_noise must be non-negative; got {level!r}.")
        if self.is_censoring and not 0.0 <= level < 1.0:
            raise ValueError(
                f"censoring's level is the FRACTION of labels clipped and must be in "
                f"[0, 1); got {level!r}.")
        if self.is_classification and not 0.0 <= level <= 1.0:
            raise ValueError(
                f"a classification level is a flip rate and must be in [0, 1]; "
                f"got {level!r}.")
        object.__setattr__(self, 'params', _freeze_params(dict(self.params)))

    # -- what kind of condition this is -------------------------------------
    @property
    def is_classification(self) -> bool:
        return self.strategy in CLASSIFICATION_STRATEGIES

    @property
    def is_censoring(self) -> bool:
        return self.strategy in CENSORING_STRATEGIES

    @property
    def needs_groups(self) -> bool:
        """True when injection requires one scaffold-group id per molecule."""
        return self.strategy in GROUPED_STRATEGIES

    @property
    def scale_is_degenerate(self) -> bool:
        """True when every molecule receives the same noise scale.

        For the five shape-only conditions "which molecules are unreliable" is
        UNDEFINED, not zero. A rank correlation against a constant must never be
        reported. This is a property of the condition, not a defect.
        """
        return self.strategy in CONSTANT_SCALE_STRATEGIES

    @property
    def param_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.params}

    @property
    def spec_id(self) -> str:
        """Stable, filename- and SQL-safe id. Goes into the cache `tier` string.

        Everything that changes the labels is in here. Leaving `dose_units` or the
        condition parameters out would let a grouped run and a Gaussian run at the
        same level collide on one cache row.
        """
        e = int(round(float(self.target_effective_noise) * 1000))
        sid = f"{self.strategy}_e{e:04d}_s{int(self.noise_seed)}"
        if self.selection_seed is not None:
            sid += f"_w{int(self.selection_seed)}"
        if not self.is_classification and not self.is_censoring:
            sid += f"_{self.dose_units}"
        for k, v in self.params:
            sid += f"_{k}{v:g}" if isinstance(v, (int, float)) else f"_{k}{v}"
        return sid

    def tier_name(self, base_tier: str) -> str:
        """The cache `tier` this spec produces on top of a base tier.

        `tier` is IDENTITY_COLS[0] of the SQLite result store and is documented as the
        condition name (`general_distance` / `top_distance` already use it this way).
        It is the ONLY free discriminator available: with train-only noise the global
        label array is unchanged, so `labels_hash` is IDENTICAL to the clean run and
        would serve clean rows straight back out of the DB.
        """
        return f"{base_tier}__noise_{self.spec_id}"


def is_noisy_tier(tier: str) -> bool:
    return '__noise_' in (tier or '')


def base_tier_of(tier: str) -> str:
    """'general__noise_gaussian_e0100_s0_spread' -> 'general'."""
    return (tier or '').split('__noise_', 1)[0]


# ---------------------------------------------------------------------------
# SCAFFOLD GROUPS FOR THE GROUPED CONDITIONS
# ---------------------------------------------------------------------------

def assign_noise_groups(smiles_list):
    """Scaffold groups for NOISE INJECTION, with acyclic molecules as singletons.

    Deliberately separate from the fold geometry: the grouping that defines the CV
    splits must not move when the noise scheme changes, and vice versa.

    The difference is NOISE_DESIGN.md section 2a rule 2. A Murcko scaffold is the
    empty string for an acyclic molecule, so every acyclic molecule otherwise lands in
    one bucket. For splitting that is merely coarse; for grouped noise it is wrong,
    because a single group-level draw would then move the whole bucket at once.
    Measured on QM9, where 32.2% of molecules are acyclic, that swings the realised
    dose from 0.795 to 1.493 against a target of 1.0. Each acyclic molecule therefore
    becomes its own group.

    Returns (groups, n_groups, largest_group_share).
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for i, smi in enumerate(smiles_list):
        core = ''
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            try:
                core = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),
                                        canonical=True)
            except Exception:
                core = ''
        # An unparseable SMILES is its own group too: guessing it into a shared
        # bucket would give one offset draw a say over molecules it has nothing in
        # common with.
        scaffolds.append(f'__acyclic_{i}' if not core else core)
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds], dtype=np.int64)
    largest = float(np.bincount(groups).max() / len(groups)) if len(groups) else 0.0
    return groups, len(unique_scaffolds), largest


# ---------------------------------------------------------------------------
# LEVEL -> DOSE
# ---------------------------------------------------------------------------

def calibrate(labels: np.ndarray, spec: NoiseSpec, is_clf: bool) -> float:
    """Resolve a spec's level into the magnitude the injector takes.

    Resolved ONCE on the full label column, so every fold and every experiment shares
    one magnitude. The Rust half resolves it against the clean TRAINING labels
    instead (`rust/src/main.rs:512`); the two differ by a few percent on a scaffold
    split, and the full column is used here because fold-independence is what makes
    the cached noisy search reproducible.

    For REGRESSION there is nothing to search any more: the delivered noise is a
    root-mean-square and the condition solves for its own internal scale in closed
    form. Exact and deterministic, where the old binary search was neither.

    Classification still calibrates, because a flip rate is not a closed form.
    """
    _require_noiseinject()
    y = np.asarray(labels).ravel()
    level = float(spec.target_effective_noise)
    if is_clf:
        from noiseInject import calibrate_flip_probability
        return float(calibrate_flip_probability(
            y, target_flip_rate=level,
            strategy=spec.strategy, random_state=int(spec.noise_seed)))
    if spec.is_censoring:
        # Not a dose. The level IS the fraction of labels clipped at the assay limit.
        return level
    if spec.dose_units == 'label':
        return level
    return level * float(np.std(y))


# ---------------------------------------------------------------------------
# INJECTION
# ---------------------------------------------------------------------------

def _build_injector(spec: NoiseSpec):
    from noiseInject import NoiseInjectorRegression
    inj = NoiseInjectorRegression.from_condition(
        spec.strategy,
        random_state=int(spec.noise_seed),
        selection_state=(None if spec.selection_seed is None
                         else int(spec.selection_seed)))
    inj.params.update(spec.param_dict)
    return inj


def make_noisy_labels_verbose(labels: np.ndarray, spec: NoiseSpec,
                              magnitude: Optional[float] = None,
                              groups: Optional[np.ndarray] = None,
                              reference: Optional[np.ndarray] = None):
    """One deterministic noisy realisation, with the provenance that produced it.

    Regression only — `InjectionResult` has no classification counterpart.

    A FRESH injector is constructed here on purpose — see the module docstring.

    Returns a `noiseInject.InjectionResult`, which also unpacks as
    (y_noisy, noise_scale, epsilon).
    """
    if spec is None:
        raise ValueError("make_noisy_labels_verbose needs a spec; a clean run has no "
                         "injection provenance to report.")
    if spec.is_classification:
        raise ValueError(
            f"{spec.strategy!r} is a classification condition and has no "
            f"InjectionResult. Use make_noisy_labels.")
    _require_noiseinject()
    y = np.asarray(labels).ravel()
    if spec.needs_groups and groups is None:
        raise ValueError(
            f"{spec.strategy!r} needs one scaffold-group id per molecule. Build it "
            f"with kirby.noise_spec.assign_noise_groups(smiles) — NOT with the "
            f"fold-geometry grouping, which merges every acyclic molecule into one "
            f"group and would let a single draw move all of them at once.")
    if magnitude is None:
        magnitude = calibrate(y, spec, is_clf=False)
    inj = _build_injector(spec)
    return inj.inject_verbose(y, float(magnitude), groups=groups, reference=reference)


def make_noisy_labels(labels: np.ndarray, spec: Optional[NoiseSpec],
                      is_clf: bool, magnitude: Optional[float] = None,
                      groups: Optional[np.ndarray] = None,
                      reference: Optional[np.ndarray] = None) -> np.ndarray:
    """The ONE deterministic noisy realisation of a label column.

    Returns a copy of `labels` with noise applied to EVERY row. Callers take the
    training subset of this; validation rows are simply never read, which keeps
    evaluation labels clean while the realisation stays fold-independent.

    A FRESH injector is constructed here on purpose — see the module docstring.
    """
    y = np.asarray(labels).ravel()
    if spec is None:
        return y.copy()
    _require_noiseinject()
    if is_clf:
        if magnitude is None:
            magnitude = calibrate(y, spec, is_clf=True)
        from noiseInject import NoiseInjectorClassification
        inj = NoiseInjectorClassification(strategy=spec.strategy,
                                          random_state=int(spec.noise_seed))
        return np.asarray(inj.inject(y, magnitude))
    return np.asarray(make_noisy_labels_verbose(
        y, spec, magnitude=magnitude, groups=groups, reference=reference).y_noisy)


def noise_scale(labels: np.ndarray, spec: NoiseSpec,
                magnitude: Optional[float] = None,
                groups: Optional[np.ndarray] = None,
                reference: Optional[np.ndarray] = None) -> np.ndarray:
    """The noise scale each molecule's region receives, without corrupting it.

    This is what scores held-out molecules against the pattern the TRAINING labels
    were exposed to: pass reference=y_train and the training group assignment. It is
    the input to "did the model learn where the data is unreliable".

    UNDEFINED for the five shape-only conditions, where every molecule gets the same
    scale. `spec.scale_is_degenerate` says so; do not report a rank correlation
    against a constant.
    """
    _require_noiseinject()
    y = np.asarray(labels).ravel()
    if magnitude is None:
        magnitude = calibrate(y, spec, is_clf=False)
    inj = _build_injector(spec)
    return np.asarray(inj.noise_scale(y, float(magnitude), reference=reference,
                                      groups=groups))


def noise_provenance_row(result, spec: Optional[NoiseSpec] = None) -> Dict[str, Any]:
    """The provenance columns every noisy result row carries.

    Names come straight from `InjectionResult.as_row()`, which matches `NoisePlan` in
    `qsar_qm_models/rust/src/main.rs` field for field. Two names for one quantity is
    how the same values ended up captioned with two different metrics on facing pages.
    """
    row = dict(result.as_row())
    if spec is not None:
        row['noise_spec_id'] = spec.spec_id
        row['dose_units'] = ('fraction_clipped' if spec.is_censoring
                             else spec.dose_units)
        row['noise_level'] = float(spec.target_effective_noise)
    return row


def noisy_labels_hash(noisy_labels: np.ndarray) -> str:
    """Short content hash of a noisy label column.

    Experiments assert this against the value the cached search was built under, so a
    blueprint fitted to a DIFFERENT noise draw fails loudly instead of silently.
    """
    a = np.ascontiguousarray(np.asarray(noisy_labels).ravel())
    h = hashlib.sha1()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def default_spec(is_clf: bool, target_effective_noise: float,
                 noise_seed: int = 0, dose_units: str = 'spread') -> NoiseSpec:
    return NoiseSpec(
        strategy=DEFAULT_STRATEGY_CLF if is_clf else DEFAULT_STRATEGY_REG,
        target_effective_noise=float(target_effective_noise),
        noise_seed=int(noise_seed),
        dose_units=dose_units,
    )
