"""Server-side diagnostic: check hybrid_eval cache state.

Single-dataset mode:
  python tests/scripts/diagnose_dataset_cache.py ginkgo_heparin

All-dataset mode (one-line summary per dataset):
  python tests/scripts/diagnose_dataset_cache.py --all

It answers:
  1. What aug_strategy suffixes exist for each dataset?
  2. Are the stored aug_names what fixed c02 would expect, or
     do they show pre-fix corruption (aug_names=() in __uncapped.pkl)?
  3. Predicted hit/miss/stale for c02's no_aug / uncapped / capped probes.
"""
import argparse
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Cache location — allow KIRBY_CACHE_DIR override, else server default.
import os
CACHE_ROOT = Path(os.environ.get(
    "KIRBY_CACHE_DIR", "/data/stat-cadd/scat9264/.cache/kirby"
))


def _aug_names_summary(files_by_aug: dict, sample_size: int) -> dict:
    """Per suffix, return majority aug_names stored in meta (None if mixed)."""
    out = {}
    for aug_suffix, files in files_by_aug.items():
        sample = random.sample(files, min(sample_size, len(files)))
        dist = Counter()
        for f in sample:
            try:
                w = pickle.loads(f.read_bytes())
                meta = w.get("_cache_meta", {})
                dist[meta.get("aug_names", "<missing>")] += 1
            except Exception:
                dist["<unreadable>"] += 1
        out[aug_suffix] = dist
    return out


def _load_dataset_aug_map() -> dict:
    """Return {dataset_name: has_augs_bool} from DATASET_REPS.

    Datasets without an 'augmentation' key in DATASET_REPS should only
    ever have __none.pkl files — absence of __uncapped/__capped is
    correct, not a bug."""
    try:
        from experiments.complexity.complexity_helpers import DATASET_REPS
    except Exception:
        return {}
    return {name: bool(cfg.get("augmentation"))
            for name, cfg in DATASET_REPS.items()}


def _list_search_caches(ds: str) -> dict[str, dict]:
    """Return {aug_strategy: {has_file: bool, has_ckpt: bool}} for this
    dataset's search cache. Tells us whether precompute FINISHED writing
    a full SearchResult (search cache), vs is still in-progress (only
    a .ckpt.pkl exists)."""
    sdir = CACHE_ROOT / "searches"
    if not sdir.exists():
        return {}
    out: dict[str, dict] = {}
    for p in sdir.glob(f"{ds}__*.pkl"):
        is_ckpt = ".ckpt" in p.name
        # Filename pattern: {ds}__{tier}__{aug_strategy}__{speed}[.ckpt].pkl
        stem = p.stem.replace(".ckpt", "")
        parts = stem.split("__")
        if len(parts) < 4:
            continue
        aug_strategy = parts[2]
        entry = out.setdefault(aug_strategy, {"has_file": False, "has_ckpt": False})
        if is_ckpt:
            entry["has_ckpt"] = True
        else:
            entry["has_file"] = True
    return out


def _scan_one(ds: str, sample: int, has_augs: bool) -> dict:
    """Return a compact summary for one dataset: file counts, aug_names
    distributions, and a verdict string."""
    he_dir = CACHE_ROOT / "hybrid_evals" / ds
    if not he_dir.exists():
        return {"exists": False}
    files_by_aug: dict = {}
    for f in he_dir.glob("*.pkl"):
        aug = f.stem.rsplit("__", 1)[1]
        files_by_aug.setdefault(aug, []).append(f)
    names_by_aug = _aug_names_summary(files_by_aug, sample)
    n_none = len(files_by_aug.get("none", []))
    n_uncapped = len(files_by_aug.get("uncapped", []))
    n_capped = len(files_by_aug.get("capped", []))

    # Classify uncapped files: aug_names=() is pre-fix corruption
    uncapped_dist = names_by_aug.get("uncapped", Counter())
    total_uncapped_sampled = sum(uncapped_dist.values())
    empty_in_uncapped = uncapped_dist.get((), 0)
    pct_corrupt_uncapped = (100.0 * empty_in_uncapped / total_uncapped_sampled
                            if total_uncapped_sampled else 0.0)

    # Search cache state: tells us whether precompute finished its
    # per-aug-condition top-level write.
    search_caches = _list_search_caches(ds)

    # Verdict considers whether augs are expected for this dataset.
    verdict_parts = []
    if n_none == 0:
        verdict_parts.append("NO __none")
    if has_augs:
        if n_uncapped == 0:
            if search_caches.get("uncapped", {}).get("has_file"):
                verdict_parts.append("__uncapped files missing despite search cache present")
            else:
                verdict_parts.append("__uncapped incomplete (precompute not done)")
        elif pct_corrupt_uncapped > 50:
            verdict_parts.append(f"__uncapped corrupt ({pct_corrupt_uncapped:.0f}% empty aug_names)")
        if n_capped == 0:
            if search_caches.get("capped", {}).get("has_file"):
                verdict_parts.append("__capped files missing despite search cache present")
            else:
                verdict_parts.append("__capped incomplete (precompute not done)")
    verdict = "; ".join(verdict_parts) if verdict_parts else "OK"

    return {
        "exists": True,
        "has_augs": has_augs,
        "total": sum(len(v) for v in files_by_aug.values()),
        "n_none": n_none,
        "n_uncapped": n_uncapped,
        "n_capped": n_capped,
        "pct_corrupt_uncapped": pct_corrupt_uncapped,
        "verdict": verdict,
        "search_caches": search_caches,
        "names_by_aug": names_by_aug,
        "files_by_aug": files_by_aug,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None,
                    help="Dataset name. Omit with --all to scan every dataset.")
    ap.add_argument("--all", action="store_true",
                    help="Scan every dataset under hybrid_evals/.")
    ap.add_argument("--sample", type=int, default=200,
                    help="Files per aug_strategy to open for meta inspection (default 200)")
    args = ap.parse_args()

    if args.all:
        he_root = CACHE_ROOT / "hybrid_evals"
        if not he_root.exists():
            print(f"ERROR: {he_root} does not exist.")
            sys.exit(1)
        aug_map = _load_dataset_aug_map()
        # Smaller per-dataset sample so the scan finishes in reasonable time.
        sample_per_ds = min(args.sample, 50)
        datasets = sorted(p.name for p in he_root.iterdir() if p.is_dir())
        print(f"Scanning {len(datasets)} datasets "
              f"(sampling {sample_per_ds} files per suffix)\n")
        print(f"Legend: 'augs?' = dataset has augmentations per DATASET_REPS.")
        print(f"        'S_n/u/c' = search cache file present for "
              f"none/uncapped/capped.\n")
        header = (f"{'dataset':<38} {'augs?':>5} {'total':>7} {'none':>7} "
                  f"{'uncap':>7} {'cap':>5} {'%cor':>4} {'S_n/u/c':>7}  verdict")
        print(header)
        print("-" * len(header))
        bad_with_augs = []
        bad_without_augs = []
        for ds in datasets:
            has_augs = aug_map.get(ds, False)
            r = _scan_one(ds, sample_per_ds, has_augs)
            if not r["exists"]:
                continue
            v = r["verdict"]
            sc = r["search_caches"]
            # Compact search cache flag: Y = file, C = ckpt only, - = none
            def _sc_flag(strategy):
                s = sc.get(strategy, {})
                if s.get("has_file"):
                    return "Y"
                if s.get("has_ckpt"):
                    return "C"
                return "-"
            s_flags = f"{_sc_flag('none')}/{_sc_flag('uncapped')}/{_sc_flag('capped')}"
            augs_str = "yes" if has_augs else "no"
            line = (f"{ds:<38} {augs_str:>5} {r['total']:>7} {r['n_none']:>7} "
                    f"{r['n_uncapped']:>7} {r['n_capped']:>5} "
                    f"{r['pct_corrupt_uncapped']:>3.0f}% {s_flags:>7}  {v}")
            print(line)
            if v != "OK":
                if has_augs:
                    bad_with_augs.append((ds, v))
                else:
                    bad_without_augs.append((ds, v))
        print()
        if bad_with_augs:
            print(f"=== {len(bad_with_augs)} antibody/aug datasets need attention ===")
            for ds, v in bad_with_augs:
                print(f"  {ds}: {v}")
        else:
            print("All aug-using datasets OK.")
        if bad_without_augs:
            print(f"\n=== {len(bad_without_augs)} non-aug datasets flagged (should be rare) ===")
            for ds, v in bad_without_augs:
                print(f"  {ds}: {v}")
        return

    if not args.dataset:
        print("ERROR: provide a dataset name or --all")
        sys.exit(1)

    ds = args.dataset
    he_dir = CACHE_ROOT / "hybrid_evals" / ds
    if not he_dir.exists():
        print(f"ERROR: {he_dir} does not exist. Wrong dataset or cache dir?")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 1. What suffixes exist on disk?
    # ----------------------------------------------------------------
    files_by_aug: dict[str, list[Path]] = {}
    for f in he_dir.glob("*.pkl"):
        aug = f.stem.rsplit("__", 1)[1]
        files_by_aug.setdefault(aug, []).append(f)

    print(f"\n=== {ds}: hybrid_eval files on disk ===")
    total = sum(len(v) for v in files_by_aug.values())
    print(f"  Total: {total}")
    for aug in sorted(files_by_aug):
        print(f"    __{aug}.pkl: {len(files_by_aug[aug])}")
    print()

    # ----------------------------------------------------------------
    # 2. Sample each suffix. Group by aug_names stored in meta.
    # ----------------------------------------------------------------
    print(f"=== Stored aug_names distribution (sample of up to {args.sample} per suffix) ===")
    for aug_suffix in sorted(files_by_aug):
        files = files_by_aug[aug_suffix]
        sample = random.sample(files, min(args.sample, len(files)))
        aug_names_dist = Counter()
        aug_strategy_dist = Counter()
        unreadable = 0
        for f in sample:
            try:
                w = pickle.loads(f.read_bytes())
                meta = w.get("_cache_meta", {})
            except Exception:
                unreadable += 1
                continue
            aug_names_dist[meta.get("aug_names", "<missing>")] += 1
            aug_strategy_dist[meta.get("aug_strategy", "<missing>")] += 1

        print(f"\n  __{aug_suffix}.pkl ({len(sample)} sampled):")
        print(f"    unreadable: {unreadable}")
        print(f"    aug_strategy counts:")
        for v, c in aug_strategy_dist.most_common():
            print(f"      {v!r}: {c}")
        print(f"    aug_names counts:")
        for v, c in aug_names_dist.most_common():
            v_str = str(v)
            if len(v_str) > 70:
                v_str = v_str[:67] + "..."
            print(f"      {v_str}: {c}")

    # ----------------------------------------------------------------
    # 3. Final verdict — is c02 going to HIT or MISS/STALE?
    # ----------------------------------------------------------------
    print("\n=== Verdict for fixed c02 runs ===")
    none_count = len(files_by_aug.get("none", []))
    uncapped_count = len(files_by_aug.get("uncapped", []))
    capped_count = len(files_by_aug.get("capped", []))

    print(f"  c02 no_aug (reads __none.pkl):")
    if none_count == 0:
        print(f"    ✗ ZERO __none.pkl files exist. Every probe will MISS.")
        print(f"      → c02 no_aug will recompute EVERY config from scratch.")
        print(f"      → FIX: re-run precompute for this dataset.")
    else:
        print(f"    ✓ {none_count} __none.pkl files exist — probes will hit.")

    print(f"  c02 uncapped (reads __uncapped.pkl):")
    if uncapped_count == 0:
        print(f"    ✗ ZERO __uncapped.pkl files exist.")
    else:
        # Check a sample for aug_names drift.
        sample = random.sample(files_by_aug["uncapped"],
                               min(50, uncapped_count))
        empty_aug = 0
        non_empty = 0
        for f in sample:
            try:
                w = pickle.loads(f.read_bytes())
                meta = w.get("_cache_meta", {})
                if meta.get("aug_names") == ():
                    empty_aug += 1
                else:
                    non_empty += 1
            except Exception:
                pass
        if empty_aug > 0 and non_empty == 0:
            print(f"    ✗ All sampled __uncapped.pkl files have aug_names=(). "
                  f"Pre-fix corruption — c02 uncapped will STALE every probe.")
            print(f"      → First run rewrites them; second run HITs. "
                  f"Or re-run precompute for a clean rebuild.")
        elif empty_aug > 0:
            print(f"    ⚠ Mixed state: {empty_aug}/{len(sample)} "
                  f"have aug_names=(). Some will STALE, some HIT.")
        else:
            print(f"    ✓ All sampled __uncapped.pkl files have non-empty "
                  f"aug_names — c02 uncapped probes will HIT.")

    print(f"  c02 capped (reads __capped.pkl):")
    if capped_count == 0:
        print(f"    ✗ ZERO __capped.pkl files — will recompute.")
    else:
        print(f"    ✓ {capped_count} files exist.")


if __name__ == "__main__":
    main()
