#!/usr/bin/env python
"""Preliminary-results summary from c11 (ensemble vs hybrid).

c02 crashed in the previous wave so its results aren't on disk yet. c11
gives us the same headline: per-(dataset, model), best_single vs hybrid,
plus ensemble and PCA-matched as bonus comparisons.

Per dataset and per downstream model:
  - best single rep + score
  - best hybrid score (across budgets)
  - delta = hybrid - baseline
  - ensemble + pca_matched scores for context

Aggregates:
  - Mean delta per model
  - Win rate (hybrid > baseline)
  - Per-domain breakdown
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
C11_DIR = REPO / "tests/results/complexity/c11_ensemble_vs_hybrid"


def domain_of(name):
    if name.startswith(("tcga_",)):
        return "tcga"
    if name.startswith(("kiba", "bindingdb", "davis", "pdbbind")):
        return "dta"
    if name.startswith(("buchwald", "bh_", "suzuki")):
        return "reaction"
    if name.startswith(("nt_", "gb_")):
        return "nucleotide"
    if any(name.startswith(p) for p in (
        "flab_", "abagym_", "abcov_", "ddg_", "ginkgo_", "tz_",
        "humanness_", "graphinity_", "sabdab_", "abbind",
    )):
        return "antibody"
    return "molecular"


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def best_score(rows, lower=False):
    if not rows:
        return None
    return min(rows, key=lambda r: r["score_mean"]) if lower \
        else max(rows, key=lambda r: r["score_mean"])


def main():
    if not C11_DIR.exists():
        print(f"No c11 results at {C11_DIR}")
        return

    per_dataset = []
    per_model_deltas = defaultdict(list)
    per_domain_deltas = defaultdict(list)
    win_count = 0
    total_count = 0

    for f in sorted(C11_DIR.glob("*.json")):
        ds = f.stem
        d = load_json(f)
        if not d or "results" not in d:
            continue
        results = d["results"]

        # Group by model
        by_model = defaultdict(list)
        for r in results:
            by_model[r["model"]].append(r)

        for model, rows in sorted(by_model.items()):
            single = [r for r in rows if r["method"] == "best_single" and not r.get("augmented")]
            hybrid = [r for r in rows if r["method"] == "hybrid" and not r.get("augmented")]
            ens = [r for r in rows if r["method"] == "ensemble" and not r.get("augmented")]
            pca = [r for r in rows if r["method"] == "pca_matched" and not r.get("augmented")]

            if not single or not hybrid:
                continue

            # heuristic: regression metric is R²/spearman (higher better) — true for everything in this run
            single_best = best_score(single, lower=False)
            hybrid_best = best_score(hybrid, lower=False)
            ens_best = best_score(ens, lower=False) if ens else None
            pca_best = best_score(pca, lower=False) if pca else None

            base_score = single_best["score_mean"]
            base_rep = single_best.get("detail", {}).get("best_rep", "?")
            hybrid_score = hybrid_best["score_mean"]
            hybrid_budget = hybrid_best.get("budget", "?")
            delta = hybrid_score - base_score

            per_dataset.append({
                "dataset": ds,
                "domain": domain_of(ds),
                "model": model,
                "base_rep": base_rep,
                "base_score": base_score,
                "hybrid_score": hybrid_score,
                "hybrid_budget": hybrid_budget,
                "ensemble_score": ens_best["score_mean"] if ens_best else None,
                "pca_score": pca_best["score_mean"] if pca_best else None,
                "delta": delta,
            })
            per_model_deltas[model].append(delta)
            per_domain_deltas[(domain_of(ds), model)].append(delta)
            total_count += 1
            if delta > 0:
                win_count += 1

    if not per_dataset:
        print("No usable c11 rows found.")
        return

    # --- Per-dataset table ---
    print("=" * 115)
    print(f"{'dataset':<32} {'dom':<6} {'model':<10} {'base_rep':<22} "
          f"{'base':>7} {'hybrid':>7} {'b':>5} {'delta':>9} "
          f"{'ens':>7} {'pca':>7}")
    print("-" * 115)
    last_ds = None
    for r in per_dataset:
        ds = r["dataset"] if r["dataset"] != last_ds else ""
        last_ds = r["dataset"]
        ens = f"{r['ensemble_score']:.4f}" if r["ensemble_score"] is not None else "  -  "
        pca = f"{r['pca_score']:.4f}" if r["pca_score"] is not None else "  -  "
        print(f"{ds:<32} {r['domain']:<6} {r['model']:<10} "
              f"{r['base_rep'][:22]:<22} {r['base_score']:>7.4f} "
              f"{r['hybrid_score']:>7.4f} {str(r['hybrid_budget']):>5} "
              f"{r['delta']:>+9.4f} {ens:>7} {pca:>7}")

    # --- Aggregate per model ---
    print()
    print("=" * 115)
    print("AGGREGATE PER MODEL")
    print("-" * 115)
    print(f"{'model':<10} {'n':>5} {'mean_delta':>12} {'median':>12} {'min':>12} {'max':>12} {'win_rate':>14}")
    for model in sorted(per_model_deltas):
        deltas = sorted(per_model_deltas[model])
        n = len(deltas)
        mean_d = sum(deltas) / n
        med_d = deltas[n // 2]
        wins = sum(1 for d in deltas if d > 0)
        print(f"{model:<10} {n:>5} {mean_d:>+12.4f} {med_d:>+12.4f} "
              f"{deltas[0]:>+12.4f} {deltas[-1]:>+12.4f} "
              f"  {wins}/{n} ({100*wins/n:.0f}%)")

    # --- Aggregate per domain x model ---
    print()
    print("=" * 115)
    print("AGGREGATE PER DOMAIN × MODEL")
    print("-" * 115)
    print(f"{'domain':<10} {'model':<10} {'n':>5} {'mean_delta':>12} {'win_rate':>10}")
    for (dom, model) in sorted(per_domain_deltas):
        deltas = per_domain_deltas[(dom, model)]
        n = len(deltas)
        mean_d = sum(deltas) / n
        wins = sum(1 for d in deltas if d > 0)
        print(f"{dom:<10} {model:<10} {n:>5} {mean_d:>+12.4f} "
              f"  {wins}/{n} ({100*wins/n:.0f}%)")

    # --- Headline ---
    print()
    print("=" * 115)
    print("HEADLINE")
    print("-" * 115)
    n_ds = len(set(r["dataset"] for r in per_dataset))
    print(f"  Datasets:                    {n_ds}")
    print(f"  Total (dataset × model):     {total_count}")
    print(f"  Hybrid > best single rep:    {win_count}/{total_count}  ({100*win_count/total_count:.0f}%)")
    all_d = [r["delta"] for r in per_dataset]
    print(f"  Mean delta (all cells):      {sum(all_d)/len(all_d):+.4f}")
    print(f"  Median delta:                {sorted(all_d)[len(all_d)//2]:+.4f}")
    print()
    print("  Per-domain win rate:")
    by_dom_total = defaultdict(int)
    by_dom_wins = defaultdict(int)
    for r in per_dataset:
        by_dom_total[r["domain"]] += 1
        if r["delta"] > 0:
            by_dom_wins[r["domain"]] += 1
    for dom in sorted(by_dom_total):
        t = by_dom_total[dom]
        w = by_dom_wins[dom]
        print(f"    {dom:<10}  {w}/{t} ({100*w/t:.0f}%)")


if __name__ == "__main__":
    main()
