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


def _scan_one(ds: str, sample: int) -> dict:
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

    verdict_parts = []
    if n_none == 0:
        verdict_parts.append("NO __none (no_aug will MISS)")
    if total_uncapped_sampled == 0:
        verdict_parts.append("NO __uncapped")
    elif pct_corrupt_uncapped > 50:
        verdict_parts.append(f"__uncapped corrupt ({pct_corrupt_uncapped:.0f}% empty aug_names)")
    if n_capped == 0:
        verdict_parts.append("NO __capped")
    verdict = "; ".join(verdict_parts) if verdict_parts else "OK"

    return {
        "exists": True,
        "total": sum(len(v) for v in files_by_aug.values()),
        "n_none": n_none,
        "n_uncapped": n_uncapped,
        "n_capped": n_capped,
        "pct_corrupt_uncapped": pct_corrupt_uncapped,
        "verdict": verdict,
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
        # Smaller per-dataset sample so the scan finishes in reasonable time.
        sample_per_ds = min(args.sample, 50)
        datasets = sorted(p.name for p in he_root.iterdir() if p.is_dir())
        print(f"Scanning {len(datasets)} datasets "
              f"(sampling {sample_per_ds} files per suffix)\n")
        header = (f"{'dataset':<38} {'total':>7} {'none':>7} {'uncap':>7} "
                  f"{'cap':>5} {'%corr':>5}  verdict")
        print(header)
        print("-" * len(header))
        bad = []
        for ds in datasets:
            r = _scan_one(ds, sample_per_ds)
            if not r["exists"]:
                continue
            v = r["verdict"]
            line = (f"{ds:<38} {r['total']:>7} {r['n_none']:>7} "
                    f"{r['n_uncapped']:>7} {r['n_capped']:>5} "
                    f"{r['pct_corrupt_uncapped']:>4.0f}%  {v}")
            print(line)
            if v != "OK":
                bad.append((ds, v))
        print()
        if bad:
            print(f"=== {len(bad)} datasets need attention ===")
            for ds, v in bad:
                print(f"  {ds}: {v}")
        else:
            print("All datasets OK.")
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
