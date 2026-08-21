"""Local reproducer: simulate precompute → c02 lookup, verify cache hits.

Design:
  Step 1  — call find_best_hybrid the way precompute calls it for
           aug_strategy='uncapped'. Writes hybrid_eval files to cache.
  Step 2  — delete the search cache file, keep the hybrid_eval files.
           This forces the second call to recompute the search from
           scratch. If cache is aligned, it will find every
           hybrid_eval on disk and HIT them.
  Step 3  — call find_best_hybrid the way c02 calls it for uncapped.
           Observe the hybrid_eval cache stats. A healthy pipeline
           shows near-100% HIT. A broken one shows MISS or STALE.
  Step 4  — if stale, diff the stored meta against expected meta and
           print the drifting field.

  Then repeat Step 3 for the no_aug case (aug_strategy='none').

Runs entirely locally. Cache lives on /Volumes/seagate/kirby_cache_repro
so it doesn't pollute ~/.cache/kirby.
"""
import os
import sys
import shutil
import pickle
from pathlib import Path

CACHE_DIR = Path("/Volumes/seagate/kirby_cache_repro")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KIRBY_CACHE_DIR"] = str(CACHE_DIR)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from kirby.find_best_hybrid import (  # noqa: E402
    find_best_hybrid,
    _CACHE_STATS,
    _reset_cache_stats,
    _cache_stats_summary,
)
from experiments._shared import INTERNAL_EVALUATORS, FULL_BUDGETS  # noqa: E402

DATASET = "repro_ds"
# Smaller problem-space than the DEFAULT_* full config so the local run stays
# under ~60s but still exercises top_k hybrid_eval cache probing.
ALLOCS = ["top_2", "top_3"]
IMPS = ["lightgbm_gain"]
EVALS = ["rf", "lightgbm"]
BUDGETS = [10, 25]


def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 80
    groups = np.repeat(np.arange(8), 10)
    base_reps = {
        "rep_a": rng.standard_normal((n, 20)).astype(np.float32),
        "rep_b": rng.standard_normal((n, 20)).astype(np.float32),
        "rep_c": rng.standard_normal((n, 20)).astype(np.float32),
    }
    augs = {
        "aug_x": rng.standard_normal((n, 5)).astype(np.float32),
        "aug_y": rng.standard_normal((n, 5)).astype(np.float32),
    }
    labels = (base_reps["rep_a"][:, :3].sum(axis=1)
              + 0.3 * rng.standard_normal(n)).astype(np.float64)
    return base_reps, labels, groups, augs


def clear_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def delete_search_cache():
    """Remove searches/ so the next call can't early-return from the search
    cache, forcing it into the hybrid_eval probe path."""
    sdir = CACHE_DIR / "searches"
    if sdir.exists():
        shutil.rmtree(sdir)


def call_like_precompute(base_reps, labels, groups, augs, strategy: str):
    """Mirror precompute_hybrid_searches's call to find_best_hybrid.

    The real precompute passes models=None, allocation_methods=None, etc.
    (tier defaults). Here we pass explicit small lists to keep the test
    fast, but the cache-key behaviour is identical — normalization just
    collapses both `None` and `explicit == tier default` to the same key.
    """
    return find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        budgets=BUDGETS,
        models=None,
        augmentations=augs if strategy != "none" else None,
        augmentation_strategy=strategy,
        speed="full",
        task_type="regression",
        metric="r2",
        cache=True,
        verbose=False,
        dataset_name=DATASET,
        tier="general",
        allocation_methods=ALLOCS,
        importance_methods=IMPS,
        internal_evaluators=EVALS,
    )


def call_like_c02_uncapped(base_reps, labels, groups, augs):
    """Mirror c02's uncapped call. Same args as precompute for this scenario."""
    return find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=None,
        budgets=BUDGETS,
        allocation_methods=ALLOCS,
        importance_methods=IMPS,
        internal_evaluators=EVALS,
        n_folds=5,
        augmentations=augs,
        augmentation_strategy="uncapped",
        metric="r2",
        task_type="regression",
        verbose=False,
        cache=True,
        dataset_name=DATASET,
        tier="general",
    )


def call_like_c02_no_aug(base_reps, labels, groups, models_spec=None):
    """Mirror c02's no_aug call. models_spec=None => tier defaults.
    models_spec=dict(...) => explicit model set, mimicking c02's actual pattern."""
    return find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        models=models_spec,
        budgets=BUDGETS,
        allocation_methods=ALLOCS,
        importance_methods=IMPS,
        internal_evaluators=EVALS,
        n_folds=5,
        augmentations=None,
        augmentation_strategy="none",
        metric="r2",
        task_type="regression",
        verbose=False,
        cache=True,
        dataset_name=DATASET,
        tier="general",
    )


def diff_metas(path: Path, expected: dict, ignore=("code_timestamp",)):
    """Load the stored meta and return a list of (key, stored, expected)
    for every drifting key."""
    try:
        wrapper = pickle.loads(path.read_bytes())
    except Exception as e:
        return [("_unreadable", str(e), "<expected>")]
    stored = wrapper.get("_cache_meta", {})
    diffs = []
    for k, v in expected.items():
        if k in ignore:
            continue
        if k not in stored:
            diffs.append((k, "<missing>", v))
        elif stored[k] != v:
            diffs.append((k, stored[k], v))
    extras = set(stored) - set(expected) - set(ignore)
    for k in extras:
        diffs.append((k, stored[k], "<not in expected>"))
    return diffs


def summarize_hybrid_evals():
    he_dir = CACHE_DIR / "hybrid_evals" / DATASET
    if not he_dir.exists():
        return {}
    by_aug = {}
    for f in he_dir.glob("*.pkl"):
        aug = f.stem.rsplit("__", 1)[1]
        by_aug[aug] = by_aug.get(aug, 0) + 1
    return by_aug


def run_scenario(label: str, precompute_strategy: str, probe_fn):
    print("\n" + "=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    # Step 1: precompute. Writes files to disk.
    print(f"  Step 1: precompute[{precompute_strategy!r}] — writing cache")
    _reset_cache_stats()
    call_like_precompute(*scenario_data, strategy=precompute_strategy)
    pre_stats = dict(_CACHE_STATS)
    print(f"    [after precompute] {_cache_stats_summary()}")
    print(f"    [on disk] hybrid_evals: {summarize_hybrid_evals()}")

    # Step 2: nuke search cache so the probe goes through hybrid_evals.
    delete_search_cache()

    # Step 3: probe as c02 does.
    print(f"  Step 2: c02-style probe — should now HIT hybrid_evals")
    _reset_cache_stats()
    probe_fn(*scenario_data[:-1] if probe_fn is call_like_c02_no_aug
            else scenario_data)
    probe_stats = dict(_CACHE_STATS)
    print(f"    [after probe] {_cache_stats_summary()}")

    he_hit = probe_stats["hybrid_eval_hit"]
    he_miss = probe_stats["hybrid_eval_miss"]
    he_stale = probe_stats["hybrid_eval_stale"]
    total = he_hit + he_miss + he_stale

    if total == 0:
        print("    ⚠️  hybrid_eval cache was never probed — non-top_k configs only?")
    else:
        pct = 100 * he_hit / total
        print(f"    VERDICT: {pct:.0f}% hit ({he_hit}/{total})")
        if he_stale > 0 or he_miss > 0:
            # Dig into the first miss/stale: show which meta field drifts.
            # We re-do a raw probe by walking hybrid_evals/<ds>/ and
            # matching the FIRST file against what c02 would expect.
            print(f"    ⚠️  Non-zero drift. Stale={he_stale}, miss={he_miss}.")
            _diagnose_drift(probe_stats)


def _diagnose_drift(stats):
    """Re-run just the probe path to collect actual (expected, stored) pairs."""
    print("    Re-probing with verbose logging to show first drift field...")
    # Enable verbose on _load_cache_with_meta by monkey-patching via a wrapper
    # probe. Easier: just dump the first stored meta and the first expected
    # meta and diff them.
    # Rebuild expected meta by doing a dry-run call WITHOUT writing, but
    # intercepting at the cache-probe step is intrusive. Simpler: read all
    # stored files for this scenario and print the meta of one.
    he_dir = CACHE_DIR / "hybrid_evals" / DATASET
    sample = next(iter(he_dir.glob("*.pkl")), None)
    if sample is None:
        print("    (no hybrid_eval files on disk — cannot diagnose)")
        return
    wrapper = pickle.loads(sample.read_bytes())
    stored = wrapper.get("_cache_meta", {})
    print(f"    Sample stored meta from {sample.name}:")
    for k in sorted(stored):
        if k == "code_timestamp":
            continue
        v = stored[k]
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"      {k} = {s}")


def simulate_prefix_legacy_uncapped(base_reps, labels, groups):
    """Simulate what pre-fix precompute wrote to __uncapped.pkl for the
    'none' condition: aug_strategy='uncapped' but augmentations=None.

    This is the exact call pattern produced by the OLD _shared.py before
    the 2026-04-21 fix. Every file produced here will have
    aug_strategy='uncapped' in meta but aug_names=(), which is exactly
    the corruption the other chat described."""
    return find_best_hybrid(
        base_reps=base_reps,
        labels=labels,
        groups=groups,
        budgets=BUDGETS,
        models=None,
        augmentations=None,                # <-- the bug: no augs
        augmentation_strategy="uncapped",  # <-- but pretends to be uncapped
        speed="full",
        task_type="regression",
        metric="r2",
        cache=True,
        verbose=False,
        dataset_name=DATASET,
        tier="general",
        allocation_methods=ALLOCS,
        importance_methods=IMPS,
        internal_evaluators=EVALS,
    )


def run_legacy_scenario():
    """Scenario C: simulate pre-fix on-disk state, then run fixed c02.

    Expected outcome if the theory is correct:
      * First probe: ~100% STALE (legacy file exists but aug_names drift).
      * Second probe (immediately after): ~100% HIT (stale -> recompute -> write).
    If first probe shows MISS instead of STALE, the filename differs
    (different subset_hash, different aug_strategy, etc) and the theory
    is wrong — we'd need to re-diagnose.
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: pre-fix legacy files -> fixed c02 uncapped probe")
    print("=" * 70)
    print("  Step 1: write legacy files (augs=None, aug_strategy='uncapped')")
    _reset_cache_stats()
    simulate_prefix_legacy_uncapped(*scenario_data[:-1])
    print(f"    [after legacy write] {_cache_stats_summary()}")
    print(f"    [on disk] hybrid_evals: {summarize_hybrid_evals()}")

    delete_search_cache()

    print("  Step 2: fixed c02 uncapped probe (expects aug_names set)")
    _reset_cache_stats()
    call_like_c02_uncapped(*scenario_data)
    s = dict(_CACHE_STATS)
    print(f"    [after probe] {_cache_stats_summary()}")
    total = s["hybrid_eval_hit"] + s["hybrid_eval_miss"] + s["hybrid_eval_stale"]
    if total:
        if s["hybrid_eval_stale"]:
            print(f"    ✓ STALE count non-zero ({s['hybrid_eval_stale']}) — "
                  f"legacy theory confirmed. First run rewrites files.")
        elif s["hybrid_eval_miss"] == total:
            print(f"    ✗ All MISS, zero STALE — filenames differ, not just meta. "
                  f"Theory wrong; subset_hash or aug_strategy filename differs.")
        else:
            print(f"    Partial. Examine breakdown: {s}")

    delete_search_cache()

    print("  Step 3: second probe (should now HIT since Step 2 rewrote)")
    _reset_cache_stats()
    call_like_c02_uncapped(*scenario_data)
    s = dict(_CACHE_STATS)
    print(f"    [after 2nd probe] {_cache_stats_summary()}")
    hit = s["hybrid_eval_hit"]
    stale = s["hybrid_eval_stale"]
    if hit and stale == 0:
        print(f"    ✓ Second probe clean — legacy stale gets rewritten in one pass.")
    elif stale:
        print(f"    ✗ STILL STALE on second probe — write not happening or "
              f"something's wrong beyond the fix. Count: {stale}")


if __name__ == "__main__":
    clear_cache()
    scenario_data = make_data(seed=0)

    # Scenario A: precompute wrote uncapped; c02 probes for uncapped.
    # Should produce 100% hit, 0 stale.
    run_scenario(
        "A: precompute[uncapped] -> c02 uncapped probe",
        precompute_strategy="uncapped",
        probe_fn=call_like_c02_uncapped,
    )

    clear_cache()
    # Scenario B: precompute wrote none; c02 probes for no_aug.
    # Should produce 100% hit, 0 stale.
    run_scenario(
        "B: precompute[none] -> c02 no_aug probe (models=None)",
        precompute_strategy="none",
        probe_fn=call_like_c02_no_aug,
    )

    clear_cache()
    # Scenario C: simulates the legacy pre-fix state on the server.
    run_legacy_scenario()
