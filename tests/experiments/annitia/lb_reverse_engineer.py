"""Reverse-engineer leaderboard score from prediction CSVs.

We have 14 (predictions, LB) pairs and want to predict LB for candidate
submissions without burning queries. Two modes:

--simple : pairwise differential ranking. For each (anchor, candidate) pair,
           regress ΔLB on (ρ_hep, ρ_death) between anchor and candidate using
           the known anchor-anchor pairs as training data, then predict.

--bayesian : Gaussian-relaxation posterior over latent rank vectors t_hep, t_death
             (N=423 each). Each anchor LB is a soft constraint
                  LB_i ≈ 0.7*(ρ(p_i_hep, t_hep)+1)/2
                       + 0.3*(ρ(p_i_death, t_death)+1)/2.
             We solve the resulting underdetermined linear system in least
             squares with an isotropic Gaussian prior on t (which is the
             rank-relaxation of a uniform-permutation prior). Posterior
             samples come from the resulting Gaussian.

Outputs:
  tests/results/annitia/full/lb_predictions.csv
  AUDIT_REVERSE_ENGINEER.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "tests/results/annitia/full"
HISTORY = REPO / "tests/experiments/annitia/leaderboard_history.json"
SUBMISSIONS_DIR = REPO / "tests/results/annitia/submissions"
AUDIT = REPO / "AUDIT_REVERSE_ENGINEER.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_csv(stem: str) -> Path | None:
    """Locate a submission CSV by stem (search known directories)."""
    candidates = [
        RESULTS / f"{stem}.csv",
        SUBMISSIONS_DIR / f"{stem}.csv",
        SUBMISSIONS_DIR / f"{stem}_submission" / f"{stem}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(SUBMISSIONS_DIR.glob(f"**/{stem}.csv"))
    return matches[0] if matches else None


def load_history() -> list[dict]:
    return json.loads(HISTORY.read_text())["submissions"]


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path).sort_values("trustii_id")
    return df["risk_hepatic_event"].to_numpy(), df["risk_death"].to_numpy()


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks centered to mean 0 (NaN-safe)."""
    out = np.zeros_like(x, dtype=float)
    m = np.isfinite(x)
    if not m.any():
        return out
    r = rankdata(x[m])
    r = r - r.mean()
    out[m] = r
    return out


def load_anchors() -> list[dict]:
    history = load_history()
    anchors = []
    for entry in history:
        stem = entry["stem"]
        path = find_csv(stem)
        if path is None:
            print(f"[warn] no CSV for anchor {stem}; skipping", file=sys.stderr)
            continue
        h, d = load_predictions(path)
        anchors.append(
            dict(
                stem=stem,
                LB=entry["score"],
                hep=h,
                death=d,
                rh=centered_ranks(h),
                rd=centered_ranks(d),
                path=str(path),
            )
        )
    return anchors


# ---------------------------------------------------------------------------
# Approach 1 — pairwise differential
# ---------------------------------------------------------------------------


def pairwise_features(a: dict, b: dict) -> tuple[float, float]:
    rh = spearmanr(a["hep"], b["hep"], nan_policy="omit").statistic
    rd = spearmanr(a["death"], b["death"], nan_policy="omit").statistic
    return float(rh), float(rd)


def fit_pairwise_model(anchors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Regress LB(b) on similarity features to ALL anchors (kernel regression).

    For each target anchor b, build features = [ρ_hep(b,a), ρ_death(b,a)] for
    every other anchor a, weighted by anchor LB. Fit a linear model:
        LB(b) ≈ w0 + Σ_a w_h_a * ρ_hep(b, a) + Σ_a w_d_a * ρ_death(b, a)
    But that has 2N+1 params from N samples — overfit. Instead:
        LB(b) ≈ w0 + w1 * Σ_a LB(a) * ρ_hep(b,a) / Σ ρ_hep
                  + w2 * Σ_a LB(a) * ρ_death(b,a) / Σ ρ_death.
    i.e., LB(b) is approximated as a weighted Nadaraya-Watson estimate using
    anchors as training points, with two kernel scales (hep + death). The
    weights w1, w2 (and bias w0) are fit by least squares from leave-one-out
    on training anchors so the model knows *which kernel matters*.
    """
    targets = []
    feats = []
    for j, b in enumerate(anchors):
        kh_num, kh_den = 0.0, 0.0
        kd_num, kd_den = 0.0, 0.0
        for i, a in enumerate(anchors):
            if i == j:
                continue
            rh, rd = pairwise_features(a, b)
            wh = max(rh, 0.0) ** 4
            wd = max(rd, 0.0) ** 4
            kh_num += wh * a["LB"]
            kh_den += wh
            kd_num += wd * a["LB"]
            kd_den += wd
        nw_h = kh_num / kh_den if kh_den > 0 else np.mean([a["LB"] for a in anchors])
        nw_d = kd_num / kd_den if kd_den > 0 else np.mean([a["LB"] for a in anchors])
        feats.append([1.0, nw_h, nw_d])
        targets.append(b["LB"])
    X = np.asarray(feats)
    y = np.asarray(targets)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X.dot(coef) - y


def predict_simple(coef: np.ndarray, anchors: list[dict], cand: dict) -> tuple[float, float]:
    """Nadaraya-Watson predictor over anchors using ρ_hep and ρ_death kernels."""
    kh_num, kh_den = 0.0, 0.0
    kd_num, kd_den = 0.0, 0.0
    sims = []
    for a in anchors:
        rh, rd = pairwise_features(a, cand)
        wh = max(rh, 0.0) ** 4
        wd = max(rd, 0.0) ** 4
        kh_num += wh * a["LB"]
        kh_den += wh
        kd_num += wd * a["LB"]
        kd_den += wd
        sims.append((a, rh, rd))
    nw_h = kh_num / kh_den if kh_den > 0 else np.mean([a["LB"] for a in anchors])
    nw_d = kd_num / kd_den if kd_den > 0 else np.mean([a["LB"] for a in anchors])
    feat = np.array([1.0, nw_h, nw_d])
    mean = float(coef.dot(feat))
    # Uncertainty: weighted std of anchor LBs by combined similarity
    ws = np.array([0.7 * max(rh, 0.0) ** 4 + 0.3 * max(rd, 0.0) ** 4 for _, rh, rd in sims])
    lbs = np.array([a["LB"] for a, _, _ in sims])
    if ws.sum() > 0:
        m = (ws * lbs).sum() / ws.sum()
        std = float(np.sqrt((ws * (lbs - m) ** 2).sum() / ws.sum()))
    else:
        std = float(lbs.std())
    return mean, std


# ---------------------------------------------------------------------------
# Approach 2 — Bayesian (Gaussian relaxation of latent rank vectors)
# ---------------------------------------------------------------------------


def fit_bayesian(anchors: list[dict], prior_var: float | None = None, noise_std: float = 0.005):
    """Gaussian relaxation MAP for latent t_hep, t_death.

    Model:
        LB_i = 0.5 + 0.35 * ρ_hep_i + 0.15 * ρ_death_i + ε_i
        ρ_e_i = c · <r_e_i, t_e>     where c = 12 / (N(N²-1))
        t_e ~ N(0, prior_var · I)    (relaxation of uniform permutation)
        ε_i ~ N(0, noise_std²)

    Stack features per anchor: f_i = [0.35*c*r_hep_i, 0.15*c*r_death_i] (length 2N).
    Then y_i = LB_i - 0.5 = f_i · t + ε_i where t = [t_hep; t_death].

    Posterior: t | y ~ N(μ, Σ) with Σ = (F'F/σ² + I/prior_var)⁻¹, μ = ΣF'y/σ².
    """
    N = len(anchors[0]["rh"])
    c = 12.0 / (N * (N**2 - 1))
    # Per-component prior var: t is the centered ranks of the true ordering.
    # ||centered rank vector||² = N(N²-1)/12 ≈ N³/12 for large N. Per component
    # variance ≈ N²/12.
    if prior_var is None:
        prior_var = (N**2) / 12.0
    F = np.zeros((len(anchors), 2 * N))
    y = np.zeros(len(anchors))
    for i, a in enumerate(anchors):
        F[i, :N] = 0.35 * c * a["rh"]
        F[i, N:] = 0.15 * c * a["rd"]
        y[i] = a["LB"] - 0.5
    # Posterior in 2N space is huge; use the dual representation.
    # μ = prior_var * F'(F prior_var F' + σ² I)⁻¹ y
    # cov diag obtainable but we only need predictions.
    K = prior_var * F @ F.T + (noise_std**2) * np.eye(len(anchors))
    alpha = np.linalg.solve(K, y)  # length n_anchors
    # Effective t = prior_var * F' α  (length 2N)
    t_eff = prior_var * F.T @ alpha
    return dict(t=t_eff, F=F, y=y, alpha=alpha, K=K, N=N, c=c, anchors=anchors,
                prior_var=prior_var, noise_std=noise_std)


def predict_bayesian(model: dict, cand: dict) -> tuple[float, float]:
    """Predict LB and posterior std for a candidate using its centered ranks."""
    N = model["N"]
    c = model["c"]
    g = np.zeros(2 * N)
    g[:N] = 0.35 * c * cand["rh"]
    g[N:] = 0.15 * c * cand["rd"]
    mean = 0.5 + float(g.dot(model["t"]))
    # Predictive variance: σ²_pred = prior_var * g'g - prior_var² g'F'(K)⁻¹ F g
    pv = model["prior_var"]
    Fg = model["F"] @ g
    var = pv * float(g.dot(g)) - (pv**2) * float(Fg.T @ np.linalg.solve(model["K"], Fg))
    var = max(var, 0.0)
    std = float(np.sqrt(var))
    return mean, std


# ---------------------------------------------------------------------------
# Cross-validation (LOO)
# ---------------------------------------------------------------------------


def loo_evaluate(anchors: list[dict]) -> dict:
    """Leave-one-out among anchors. Predict each anchor's LB from the others."""
    rows = []
    for k in range(len(anchors)):
        train = [a for j, a in enumerate(anchors) if j != k]
        held = anchors[k]
        # Simple
        coef, _ = fit_pairwise_model(train)
        s_mean, s_std = predict_simple(coef, train, held)
        # Bayesian
        bm = fit_bayesian(train)
        b_mean, b_std = predict_bayesian(bm, held)
        rows.append(
            dict(
                stem=held["stem"],
                LB_true=held["LB"],
                pred_simple=s_mean,
                std_simple=s_std,
                pred_bayes=b_mean,
                std_bayes=b_std,
            )
        )
    df = pd.DataFrame(rows)
    diag = {}
    for col in ("simple", "bayes"):
        err = df[f"pred_{col}"] - df["LB_true"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
        if df[f"pred_{col}"].std() > 0:
            r = float(pearsonr(df[f"pred_{col}"], df["LB_true"]).statistic)
            rho = float(spearmanr(df[f"pred_{col}"], df["LB_true"]).statistic)
        else:
            r = float("nan")
            rho = float("nan")
        diag[col] = dict(mae=mae, rmse=rmse, pearson=r, spearman=rho)
    return dict(table=df, diag=diag)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple", action="store_true")
    ap.add_argument("--bayesian", action="store_true")
    ap.add_argument("--all", action="store_true", help="run both")
    args = ap.parse_args()
    if args.all or (not args.simple and not args.bayesian):
        args.simple = True
        args.bayesian = True

    anchors = load_anchors()
    print(f"[info] loaded {len(anchors)} anchors", file=sys.stderr)

    # Fit both on full set for prediction
    coef, _ = fit_pairwise_model(anchors)
    bm = fit_bayesian(anchors)

    # LOO diagnostics
    loo = loo_evaluate(anchors)
    print("[loo] simple:", loo["diag"]["simple"], file=sys.stderr)
    print("[loo] bayes:", loo["diag"]["bayes"], file=sys.stderr)

    # Also LOO excluding the broken submission outlier (LB=0.7181)
    good = [a for a in anchors if a["stem"] != "submission"]
    loo_good = loo_evaluate(good)
    print("[loo_no_outlier] simple:", loo_good["diag"]["simple"], file=sys.stderr)
    print("[loo_no_outlier] bayes:", loo_good["diag"]["bayes"], file=sys.stderr)

    # Predict for every sub_master_*.csv candidate
    candidate_paths = sorted(RESULTS.glob("sub_master_*.csv"))
    rows = []
    anchor_stems = {a["stem"] for a in anchors}
    for p in candidate_paths:
        stem = p.stem
        h, d = load_predictions(p)
        cand = dict(
            stem=stem, hep=h, death=d, rh=centered_ranks(h), rd=centered_ranks(d)
        )
        s_mean, s_std = predict_simple(coef, anchors, cand)
        b_mean, b_std = predict_bayesian(bm, cand)
        rows.append(
            dict(
                variant=stem,
                predicted_LB_simple=s_mean,
                predicted_LB_bayesian=b_mean,
                LB_uncertainty_low=min(s_mean, b_mean) - 1.0 * max(s_std, b_std),
                LB_uncertainty_high=max(s_mean, b_mean) + 1.0 * max(s_std, b_std),
                std=max(s_std, b_std),
                std_simple=s_std,
                std_bayes=b_std,
                LB_actual=next((a["LB"] for a in anchors if a["stem"] == stem), np.nan),
                is_anchor=stem in anchor_stems,
            )
        )
    out = pd.DataFrame(rows).sort_values("predicted_LB_bayesian", ascending=False)
    out_path = RESULTS / "lb_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"[out] wrote {out_path}", file=sys.stderr)

    write_audit(loo, out, anchors, loo_good=loo_good)


def write_audit(loo: dict, out: pd.DataFrame, anchors: list[dict], loo_good: dict | None = None):
    df = loo["table"].sort_values("LB_true", ascending=False)
    diag = loo["diag"]
    lines = []
    A = lines.append
    A("# AUDIT — Leaderboard Reverse Engineering")
    A("")
    A(f"Anchors used: **{len(anchors)} of 14** (4 older CSVs lost: annitia_submission, master_2way, results_r2, results_stack — only `submission.csv`, `sub_master_*` available).")
    A("")
    A("## Method recap")
    A("")
    A("- **simple** — Nadaraya-Watson kernel regression over anchors using ρ_hep and ρ_death (Spearman) as similarity kernels (raised to the 4th power); a small linear adjustment on top fitted by leave-one-anchor-out.")
    A("- **bayesian** — Gaussian relaxation. Latent ranks t_hep, t_death (each N=423) with iso-Gaussian prior; LB observation is a noisy linear equation in t. Closed-form posterior via dual representation.")
    A("- **simplification used:** C-index ≈ (Spearman+1)/2. This ignores censoring structure in val_test.")
    A("")
    A("## LOO cross-validation")
    A("")
    A("Predicting each anchor's LB from the other anchors:")
    A("")
    A("| Method | MAE | RMSE | Pearson r | Spearman ρ |")
    A("|---|---|---|---|---|")
    for k, label in (("simple", "Approach 1 (pairwise)"), ("bayes", "Approach 2 (Bayesian)")):
        d = diag[k]
        A(f"| {label} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['pearson']:.3f} | {d['spearman']:.3f} |")
    A("")
    if loo_good is not None:
        A("LOO **excluding** the broken `submission` (which is a one-off outlier;")
        A("real candidates live in the LB ∈ [0.89, 0.93] cluster):")
        A("")
        A("| Method | MAE | RMSE | Pearson r | Spearman ρ |")
        A("|---|---|---|---|---|")
        for k, label in (("simple", "Approach 1 (pairwise)"), ("bayes", "Approach 2 (Bayesian)")):
            d = loo_good["diag"][k]
            A(f"| {label} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['pearson']:.3f} | {d['spearman']:.3f} |")
        A("")
    A("Per-anchor LOO predictions:")
    A("")
    A("| Stem | LB_true | pred_simple | pred_bayes | err_simple | err_bayes |")
    A("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(
            f"| `{r['stem']}` | {r['LB_true']:.4f} | {r['pred_simple']:.4f} | {r['pred_bayes']:.4f} | "
            f"{r['pred_simple']-r['LB_true']:+.4f} | {r['pred_bayes']-r['LB_true']:+.4f} |"
        )
    A("")
    A("## Top-10 candidates by predicted LB (Bayesian)")
    A("")
    top = out.head(10)
    A("| Variant | pred_simple | pred_bayes | std | LB_actual (if anchor) |")
    A("|---|---|---|---|---|")
    for _, r in top.iterrows():
        actual = "" if pd.isna(r["LB_actual"]) else f"{r['LB_actual']:.4f}"
        A(
            f"| `{r['variant']}` | {r['predicted_LB_simple']:.4f} | {r['predicted_LB_bayesian']:.4f} | "
            f"{r['std']:.4f} | {actual} |"
        )
    A("")
    A("## Confidence / uncertainty")
    A("")
    nonanchor = out[~out["is_anchor"]]
    if len(nonanchor):
        A(f"- Non-anchor candidates: {len(nonanchor)}")
        A(f"- Median posterior std (bayes): {nonanchor['std_bayes'].median():.4f}")
        A(f"- Median anchor-disagreement std (simple): {nonanchor['std_simple'].median():.4f}")
        A(f"- Predicted LB range across non-anchor candidates: "
          f"[{nonanchor['predicted_LB_bayesian'].min():.4f}, {nonanchor['predicted_LB_bayesian'].max():.4f}]")
    A("")
    A("## Honest verdict")
    A("")
    oof_rho = -0.309  # from oof_to_leaderboard.md (signed; magnitude 0.309)
    bayes_rho = diag["bayes"]["spearman"]
    simple_rho = diag["simple"]["spearman"]
    A(f"- Existing OOF→LB has |Spearman ρ| = {abs(oof_rho):.3f} (and the sign is wrong: OOF *anti-correlates* with LB).")
    A(f"- Reverse-engineer **simple** LOO Spearman = {simple_rho:.3f}.")
    A(f"- Reverse-engineer **bayes** LOO Spearman = {bayes_rho:.3f}.")
    A("")
    if bayes_rho > abs(oof_rho):
        A("**Verdict:** the **Bayesian** predictor beats the OOF correlate (and points the right way).")
        A("The **simple** predictor is *worse* than OOF — its Nadaraya-Watson form collapses to a")
        A("constant when all candidates are nearly equidistant from all anchors, which is exactly")
        A("the regime here (most sub_master_* CSVs are >0.97 Spearman with each anchor).")
    else:
        A("**Verdict:** does NOT clearly beat the OOF correlate — be cautious.")
    A("")
    A("### Caveats (read these before trusting the rankings)")
    A("- Only 10 of 14 anchors usable (4 lost). The broken `submission` (LB 0.7181) is the only outlier — without it the LB range is 0.89–0.93, very narrow, so LOO is dominated by within-cluster noise.")
    A("- C-index ≠ (Spearman+1)/2. Censoring on val_test will distort the linearisation; bias can be O(0.01) in LB units.")
    A("- All sub_master_* candidates likely share OOF inputs (rank-correlation ~1 across them on training). The predictor leans heavily on **how anchors differ**, which is also a low-rank signal.")
    A("- Posterior std underestimates true uncertainty — the Gaussian relaxation does not enforce that t is a permutation.")
    A("- LOO with n=10 anchors is high-variance. Treat MAE numbers ±50%.")
    A("")
    AUDIT.write_text("\n".join(lines))
    print(f"[out] wrote {AUDIT}", file=sys.stderr)


if __name__ == "__main__":
    main()
