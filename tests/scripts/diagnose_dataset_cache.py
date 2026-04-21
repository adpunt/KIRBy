"""Server-side diagnostic: check hybrid_eval cache state for a dataset.

Run this on the server with the dataset you care about, e.g.:
  cd /data/stat-cadd/scat9264/KIRBy
  python tests/scripts/diagnose_dataset_cache.py ginkgo_heparin

It answers three questions:
  1. What aug_strategy suffixes exist on disk for this dataset?
  2. For a sample of __uncapped.pkl files, does the stored meta match
     what fixed c02 would expect? Or is aug_names=() (pre-fix corruption)?
  3. Do __none.pkl files exist? (If no, c02's no_aug call will MISS every
     probe and recompute from scratch.)

Outputs one "[cache] ..." line-style summary at the end so it's
pipe-friendly.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Dataset name, e.g. ginkgo_heparin")
    ap.add_argument("--sample", type=int, default=200,
                    help="How many files per aug_strategy to open (default 200)")
    args = ap.parse_args()

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
