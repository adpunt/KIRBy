"""Blend search over top Bayesian-LB candidates.

Generates rank-blended submissions (pairs and triples), scores them with the
Bayesian LB predictor from lb_reverse_engineer.py, and emits the top-5 blends
that beat any single anchor.

Outputs:
  tests/results/annitia/full/sub_master_blend_top{1..5}.csv
  tests/results/annitia/full/blend_search_results.csv
  AUDIT_BLEND_SEARCH.md
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/experiments/annitia"))

from lb_reverse_engineer import (  # noqa: E402
    centered_ranks,
    fit_bayesian,
    load_anchors,
    predict_bayesian,
)

RESULTS = REPO / "tests/results/annitia/full"
LB_PRED = RESULTS / "lb_predictions.csv"
AUDIT_OUT = REPO / "AUDIT_BLEND_SEARCH.md"
SEARCH_OUT = RESULTS / "blend_search_results.csv"

TOP_PAIR = 15
TOP_TRIPLE = 10
PAIR_STEPS = np.round(np.arange(0.0, 1.0001, 0.1), 2)
TRIPLE_STEPS = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def load_candidate(stem: str) -> dict:
    path = RESULTS / f"{stem}.csv"
    df = pd.read_csv(path).sort_values("trustii_id").reset_index(drop=True)
    return {
        "stem": stem,
        "trustii_id": df["trustii_id"].to_numpy(),
        "hep": df["risk_hepatic_event"].to_numpy(),
        "death": df["risk_death"].to_numpy(),
    }


def blend(comps: list[dict], weights: np.ndarray) -> dict:
    """Blend by rank: weighted sum of per-endpoint ranks. Returns trustii-aligned dict."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    hep_rank = np.zeros_like(comps[0]["hep"], dtype=float)
    death_rank = np.zeros_like(comps[0]["death"], dtype=float)
    for w, c in zip(weights, comps):
        hep_rank += w * rankdata(c["hep"])
        death_rank += w * rankdata(c["death"])
    return {
        "trustii_id": comps[0]["trustii_id"],
        "hep": hep_rank,
        "death": death_rank,
        "rh": centered_ranks(hep_rank),
        "rd": centered_ranks(death_rank),
    }


def score(model: dict, blended: dict) -> tuple[float, float]:
    cand = {"hep": blended["hep"], "death": blended["death"], "rh": blended["rh"], "rd": blended["rd"]}
    return predict_bayesian(model, cand)


def write_submission(path: Path, blended: dict) -> None:
    df = pd.DataFrame({
        "trustii_id": blended["trustii_id"].astype(int),
        "risk_hepatic_event": blended["hep"],
        "risk_death": blended["death"],
    })
    df.to_csv(path, index=False)


def main() -> None:
    print("[info] loading anchors and fitting Bayesian model", file=sys.stderr)
    anchors = load_anchors()
    bm = fit_bayesian(anchors)

    lb = pd.read_csv(LB_PRED).sort_values("predicted_LB_bayesian", ascending=False)
    top_stems = lb["variant"].head(TOP_PAIR).tolist()
    print(f"[info] top {TOP_PAIR} candidates: {top_stems}", file=sys.stderr)

    candidates = {s: load_candidate(s) for s in top_stems}
    # Self-score each anchor to set baseline (recompute through same scoring path)
    self_scores = {}
    for s, c in candidates.items():
        cand = {
            "hep": c["hep"], "death": c["death"],
            "rh": centered_ranks(c["hep"]), "rd": centered_ranks(c["death"]),
        }
        m, sd = predict_bayesian(bm, cand)
        self_scores[s] = (m, sd)
    best_solo_stem = max(self_scores, key=lambda k: self_scores[k][0])
    best_solo = self_scores[best_solo_stem]
    print(f"[info] best solo: {best_solo_stem} pred={best_solo[0]:.5f} std={best_solo[1]:.5f}", file=sys.stderr)

    rows = []

    # Pairs
    pair_stems = top_stems[:TOP_PAIR]
    for a, b in combinations(pair_stems, 2):
        comps = [candidates[a], candidates[b]]
        for w in PAIR_STEPS:
            if w in (0.0, 1.0):
                continue  # reduces to a single anchor
            blended = blend(comps, np.array([w, 1.0 - w]))
            mean, std = score(bm, blended)
            rows.append({
                "components": f"{a}|{b}",
                "n_components": 2,
                "weights": f"{w:.2f},{1-w:.2f}",
                "predicted_LB": mean,
                "std": std,
                "anchor_a": self_scores[a][0],
                "anchor_b": self_scores[b][0],
                "best_anchor_pred": max(self_scores[a][0], self_scores[b][0]),
            })
    print(f"[info] pair blends scored: {len(rows)}", file=sys.stderr)

    # Triples
    triple_stems = top_stems[:TOP_TRIPLE]
    for a, b, c in combinations(triple_stems, 3):
        comps = [candidates[a], candidates[b], candidates[c]]
        for wa in TRIPLE_STEPS:
            for wb in TRIPLE_STEPS:
                wc = round(1.0 - wa - wb, 2)
                if wc < 0 or wc > 1:
                    continue
                # Skip cases that reduce to <=2 components
                weights = np.array([wa, wb, wc])
                if np.sum(weights > 0) < 3:
                    continue
                blended = blend(comps, weights)
                mean, std = score(bm, blended)
                rows.append({
                    "components": f"{a}|{b}|{c}",
                    "n_components": 3,
                    "weights": f"{wa:.2f},{wb:.2f},{wc:.2f}",
                    "predicted_LB": mean,
                    "std": std,
                    "anchor_a": self_scores[a][0],
                    "anchor_b": self_scores[b][0],
                    "best_anchor_pred": max(self_scores[a][0], self_scores[b][0], self_scores[c][0]),
                })
    print(f"[info] total blends scored: {len(rows)}", file=sys.stderr)

    df = pd.DataFrame(rows).sort_values("predicted_LB", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.to_csv(SEARCH_OUT, index=False)
    print(f"[out] {SEARCH_OUT}", file=sys.stderr)

    # Filter blends that exceed the best solo anchor in the top-15
    threshold = best_solo[0]
    beats = df[df["predicted_LB"] > threshold].reset_index(drop=True)
    print(f"[info] {len(beats)} blends exceed best solo (pred {threshold:.5f})", file=sys.stderr)

    # Write top-5 sub_master_blend_*.csv
    chosen = beats.head(5) if len(beats) >= 5 else df.head(5)
    chosen_paths = []
    for i, r in chosen.iterrows():
        comp_stems = r["components"].split("|")
        weights = np.array([float(x) for x in r["weights"].split(",")])
        comps = [candidates[s] for s in comp_stems]
        blended = blend(comps, weights)
        out_path = RESULTS / f"sub_master_blend_top{i+1}.csv"
        write_submission(out_path, blended)
        chosen_paths.append(out_path)
        print(f"[out] {out_path} pred={r['predicted_LB']:.5f}", file=sys.stderr)

    # Audit
    write_audit(df, beats, chosen, best_solo_stem, best_solo, self_scores, top_stems)


def write_audit(df: pd.DataFrame, beats: pd.DataFrame, chosen: pd.DataFrame,
                best_solo_stem: str, best_solo: tuple[float, float],
                self_scores: dict, top_stems: list[str]) -> None:
    lines: list[str] = []
    A = lines.append
    A("# AUDIT — Blend Search")
    A("")
    A("Predictor: Bayesian LB estimator from `lb_reverse_engineer.py`. LOO MAE on")
    A("anchors is ~0.005-0.012 LB units, so improvements smaller than ~0.005")
    A("should be considered indistinguishable from noise.")
    A("")
    A("## Setup")
    A("")
    A(f"- Top {TOP_PAIR} candidates by Bayesian predicted LB used for pair blends.")
    A(f"- Top {TOP_TRIPLE} candidates used for triple blends.")
    A("- Pair grid: w ∈ {0.1, 0.2, ..., 0.9} (10 values, excluding 0/1 trivial).")
    A("- Triple grid: simplex (wa, wb, wc) with step 0.1 and all weights > 0.")
    A("- Blending is per-endpoint rank-mean: blended_rank = Σ w_i · rank(comp_i).")
    A("")
    A("## Best solo anchor (from top-15)")
    A("")
    A(f"- `{best_solo_stem}`: predicted_LB = {best_solo[0]:.5f}, std = {best_solo[1]:.5f}")
    A("")
    A("Solo predictions (re-scored through same Bayesian path) for top-15:")
    A("")
    A("| Stem | predicted_LB | std |")
    A("|---|---|---|")
    for s in top_stems:
        m, sd = self_scores[s]
        A(f"| `{s}` | {m:.5f} | {sd:.5f} |")
    A("")
    A("## Top 10 blends by predicted LB")
    A("")
    A("| Rank | n | Components | Weights | predicted_LB | std | best_anchor | Δ vs anchor |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in df.head(10).iterrows():
        delta = r["predicted_LB"] - r["best_anchor_pred"]
        comp = r["components"].replace("|", " + ")
        A(f"| {r['rank']} | {r['n_components']} | {comp} | {r['weights']} | "
          f"{r['predicted_LB']:.5f} | {r['std']:.5f} | {r['best_anchor_pred']:.5f} | {delta:+.5f} |")
    A("")
    A("## Beats best solo anchor?")
    A("")
    threshold = best_solo[0]
    A(f"- Threshold (best solo predicted_LB): **{threshold:.5f}**")
    A(f"- Blends predicted to exceed threshold: **{len(beats)}**")
    if len(beats):
        best_blend = beats.iloc[0]
        improvement = best_blend["predicted_LB"] - threshold
        A(f"- Best blend gain over best solo: **{improvement:+.5f}**")
    A("")
    A("## Honest verdict")
    A("")
    A("- LOO MAE of the Bayesian predictor is ~0.005-0.012; gains below ~0.005 are not distinguishable from predictor noise.")
    if len(beats):
        improvement = beats.iloc[0]["predicted_LB"] - threshold
        if improvement < 0.001:
            A(f"- Top blend improves over the best solo anchor by only {improvement:+.5f} — well within predictor noise. **No real signal of improvement.**")
        elif improvement < 0.005:
            A(f"- Top blend improves by {improvement:+.5f} — within LOO MAE. Treat as a coin-flip relative to the best solo anchor.")
        else:
            A(f"- Top blend improves by {improvement:+.5f}, comparable to the predictor's MAE. Slight edge plausible but not certain.")
    else:
        A("- No blend beat the best solo anchor in this grid. The Bayesian predictor prefers the solo top.")
    A("")
    A("- The top-15 anchors are highly rank-correlated (>0.97) so blends move only marginally in the rank space the predictor sees.")
    A("- Posterior std understates true uncertainty (Gaussian relaxation, no permutation constraint).")
    A("- C-index ≈ (Spearman+1)/2 is approximate; true LB can differ from prediction by 0.005-0.01 even on anchor data.")
    A("")
    A("## Chosen blends (written to disk)")
    A("")
    A("| File | Components | Weights | predicted_LB |")
    A("|---|---|---|---|")
    for i, r in chosen.iterrows():
        comp = r["components"].replace("|", " + ")
        A(f"| `sub_master_blend_top{i+1}.csv` | {comp} | {r['weights']} | {r['predicted_LB']:.5f} |")
    A("")
    AUDIT_OUT.write_text("\n".join(lines))
    print(f"[out] {AUDIT_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
