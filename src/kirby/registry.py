"""Canonical per-dataset metric / task / direction registry.

Single source of truth for:
- task_type ('classification' or 'regression')
- metric name (must match a key in find_best_hybrid.BUILTIN_METRICS)
- lower_is_better (derived from BUILTIN_METRICS; stored here for readability)
- domain ('molecular', 'antibody', 'dta', 'tcga', 'reaction', 'nucleotide')

Every consumer reads from here. There is no other metric-direction table in
the codebase. If you need to override per-experiment, pass `metric=` explicitly
to find_best_hybrid — but that will bypass the registry and force a mismatch
check on load.

Adding a new dataset:
1. Add an entry here with the four required fields.
2. Confirm the loader sets `data['task_type']` identically (load_data cross-
   checks this on every call and raises loudly on mismatch).
3. If the metric isn't in BUILTIN_METRICS, add it there first.
4. Run `verify_registry()` — it's called at module import and at the start of
   every precompute run, so broken entries fail immediately.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# Valid values — any value outside these sets fails verify_registry().
_VALID_TASK_TYPES = ('classification', 'regression')
_VALID_DOMAINS = ('molecular', 'antibody', 'dta', 'tcga', 'reaction',
                  'nucleotide')


@dataclass(frozen=True)
class DatasetInfo:
    """Per-dataset metadata. Immutable on purpose — drift must be a code edit."""

    name: str
    task_type: str
    metric: str
    lower_is_better: bool
    domain: str
    n_samples: Optional[int] = None
    notes: str = ''


# =============================================================================
# REGISTRY
# =============================================================================
#
# The registry ships empty. The experiment layer populates it at startup by
# calling `register_datasets()` with its dataset table — the framework stays
# free of a baked-in dataset list.

DATASET_REGISTRY: Dict[str, DatasetInfo] = {}  # populated by the experiment layer via register_datasets()


def register_datasets(mapping):
    """Merge a {name: DatasetInfo} mapping into the registry. Called by the
    experiment layer so the framework ships without a baked-in dataset list."""
    DATASET_REGISTRY.update(mapping)


# =============================================================================
# LOOKUP + VERIFICATION
# =============================================================================

def lookup(dataset_name: str) -> Optional[DatasetInfo]:
    """Return the DatasetInfo for `dataset_name`, or None if not registered.

    Returning None (rather than raising) lets callers decide whether the miss
    is fatal. `find_best_hybrid(cache=True)` requires a registered dataset
    when `metric=` isn't passed explicitly.
    """
    return DATASET_REGISTRY.get(dataset_name)


def require(dataset_name: str) -> DatasetInfo:
    """Like lookup but raises if the dataset is not registered."""
    info = DATASET_REGISTRY.get(dataset_name)
    if info is None:
        raise KeyError(
            f"Dataset '{dataset_name}' is not in DATASET_REGISTRY. "
            f"Add an entry to src/kirby/registry.py or pass metric= "
            f"and task_type= explicitly to find_best_hybrid."
        )
    return info


def crosscheck_task_type(dataset_name: str, loader_task_type: str) -> None:
    """Raise if a loader's task_type disagrees with the registry.

    Use this at the top of any `load_dataset`-style wrapper that
    doesn't route through `_shared.load_data` (which already does the
    cross-check). Silent drift between loader and registry produces
    metric-direction bugs that are expensive to find — fail loud.

    Returns None on match (or unregistered dataset). Raises ValueError
    on mismatch.
    """
    info = DATASET_REGISTRY.get(dataset_name)
    if info is None:
        return  # unregistered is fine — caller is on their own
    if loader_task_type not in ('classification', 'regression'):
        raise ValueError(
            f"crosscheck_task_type({dataset_name!r}): loader returned "
            f"task_type={loader_task_type!r}. Must be 'classification' "
            f"or 'regression'."
        )
    if loader_task_type != info.task_type:
        raise ValueError(
            f"Registry/loader mismatch for {dataset_name!r}: "
            f"loader returned task_type={loader_task_type!r}, but "
            f"DATASET_REGISTRY says {info.task_type!r}. Fix one side — "
            f"they must agree."
        )


def verify_registry() -> None:
    """Assert every entry is internally consistent.

    Called once at module import. Also safe to call from tests /
    precompute scripts as a pre-flight check.

    Raises ValueError on the first offending entry.
    """
    # Import lazily to avoid a hard circular dependency with find_best_hybrid.
    from kirby.find_best_hybrid import BUILTIN_METRICS

    for name, info in DATASET_REGISTRY.items():
        if name != info.name:
            raise ValueError(
                f"Registry key '{name}' does not match info.name '{info.name}'"
            )
        if info.task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Dataset '{name}': invalid task_type {info.task_type!r}. "
                f"Must be one of {_VALID_TASK_TYPES}."
            )
        if info.domain not in _VALID_DOMAINS:
            raise ValueError(
                f"Dataset '{name}': invalid domain {info.domain!r}. "
                f"Must be one of {_VALID_DOMAINS}."
            )
        if info.metric not in BUILTIN_METRICS:
            raise ValueError(
                f"Dataset '{name}': metric {info.metric!r} not in BUILTIN_METRICS. "
                f"Add it there before registering the dataset."
            )
        _, builtin_lib = BUILTIN_METRICS[info.metric]
        if info.lower_is_better != builtin_lib:
            raise ValueError(
                f"Dataset '{name}': lower_is_better={info.lower_is_better} "
                f"does not match BUILTIN_METRICS[{info.metric!r}]={builtin_lib}. "
                f"Direction must agree."
            )


def all_datasets() -> Dict[str, DatasetInfo]:
    """Return a copy of the registry. Useful for tests / diagnostics."""
    return dict(DATASET_REGISTRY)


def datasets_by_domain(domain: str) -> Dict[str, DatasetInfo]:
    """Return all registry entries in `domain`."""
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}. Expected {_VALID_DOMAINS}.")
    return {k: v for k, v in DATASET_REGISTRY.items() if v.domain == domain}


# Verify at import — if someone edits the registry into an inconsistent state,
# every entry point fails loudly on first import, not later at runtime.
verify_registry()
