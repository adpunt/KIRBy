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

Noise LEVEL is expressed as `target_effective_noise` (mean |Δy| as a fraction of the
label SD) and converted to a raw sigma by binary search on the actual labels. This is
what makes a level comparable across datasets; the four ad-hoc definitions never were.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

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

# Regression strategies from noiseInject. 'legacy' = homogeneous Gaussian (the default).
REGRESSION_STRATEGIES = ('legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop')
CLASSIFICATION_STRATEGIES = ('uniform', 'class_imbalance', 'binary_asymmetric',
                             'instance_noise', 'class_dependent', 'confusion_directed')

# The noise grid the cached precompute is built for. Levels are target EFFECTIVE noise.
DEFAULT_NOISE_LEVELS = (0.05, 0.10, 0.20)
DEFAULT_STRATEGY_REG = 'legacy'
DEFAULT_STRATEGY_CLF = 'uniform'


@dataclass(frozen=True)
class NoiseSpec:
    """A frozen, hashable description of one injected-noise condition.

    strategy               noiseInject strategy name
    target_effective_noise mean |Δy| as a fraction of label SD (regression), or the
                           target flip rate (classification)
    noise_seed             seed for the noise draw ONLY. Deliberately separate from the
                           search's `seed`, which drives the CV fold boundaries and sits
                           in the cache identity — keying noise off `seed` would move
                           every fold whenever the noise draw changed.
    """
    strategy: str
    target_effective_noise: float
    noise_seed: int = 0

    def __post_init__(self):
        if self.strategy not in REGRESSION_STRATEGIES + CLASSIFICATION_STRATEGIES:
            raise ValueError(
                f"Unknown noise strategy {self.strategy!r}. "
                f"Regression: {REGRESSION_STRATEGIES}. "
                f"Classification: {CLASSIFICATION_STRATEGIES}.")
        if not 0.0 < float(self.target_effective_noise) <= 1.0:
            raise ValueError(
                f"target_effective_noise must be in (0, 1]; got "
                f"{self.target_effective_noise!r}. Use label_noise=None for a clean search.")

    @property
    def spec_id(self) -> str:
        """Stable, filename- and SQL-safe id. Goes into the cache `tier` string."""
        e = int(round(float(self.target_effective_noise) * 1000))
        return f"{self.strategy}_e{e:03d}_s{int(self.noise_seed)}"

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
    """'general__noise_legacy_e100_s0' -> 'general'."""
    return (tier or '').split('__noise_', 1)[0]


def calibrate(labels: np.ndarray, spec: NoiseSpec, is_clf: bool) -> float:
    """Resolve a spec's target level into a raw sigma (or flip probability).

    Calibrated ONCE on the full label column, so every fold and every experiment shares
    one magnitude. Deterministic: `calibrate_sigma` binary-searches with an injector
    seeded by `random_state`, so a fixed seed gives a fixed answer.
    """
    _require_noiseinject()
    y = np.asarray(labels).ravel()
    if is_clf:
        from noiseInject import calibrate_flip_probability
        return float(calibrate_flip_probability(
            y, target_flip_rate=float(spec.target_effective_noise),
            strategy=spec.strategy, random_state=int(spec.noise_seed)))
    from noiseInject import calibrate_sigma
    return float(calibrate_sigma(
        y, target_effective_noise=float(spec.target_effective_noise),
        strategy=spec.strategy, random_state=int(spec.noise_seed)))


def make_noisy_labels(labels: np.ndarray, spec: Optional[NoiseSpec],
                      is_clf: bool, magnitude: Optional[float] = None) -> np.ndarray:
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
    if magnitude is None:
        magnitude = calibrate(y, spec, is_clf)
    if is_clf:
        from noiseInject import NoiseInjectorClassification
        inj = NoiseInjectorClassification(strategy=spec.strategy,
                                          random_state=int(spec.noise_seed))
        return np.asarray(inj.inject(y, magnitude))
    from noiseInject import NoiseInjectorRegression
    inj = NoiseInjectorRegression(strategy=spec.strategy,
                                  random_state=int(spec.noise_seed))
    return np.asarray(inj.inject(y, magnitude))


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
                 noise_seed: int = 0) -> NoiseSpec:
    return NoiseSpec(
        strategy=DEFAULT_STRATEGY_CLF if is_clf else DEFAULT_STRATEGY_REG,
        target_effective_noise=float(target_effective_noise),
        noise_seed=int(noise_seed),
    )
