#!/usr/bin/env python3
"""Davis cold-protein baseline investigation.

Hypothesis: the eight (rep, model) baselines that hit identical CI=0.7567 /
R²=0.224 / RMSE=0.788 to 4dp on davis aren't learning anything rep-specific.
They're collapsing to "predict per-drug mean training affinity" because in a
cold-protein split every drug still appears in train (paired with a different
subset of proteins).

Tests:
  1. Compute the trivial drug-mean-affinity predictor's CI/R²/RMSE on the
     same splits used by the baseline survey.
  2. Refit ridge with each tied rep, dump per-fold predictions, and check:
       - pairwise correlation across reps (should be ~1.0 if rep-agnostic)
       - correlation with the drug-mean predictor (should be ~1.0)

Run on server (needs DTA reps + label loader):
  python tests/scripts/davis_baseline_investigation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from kirby.find_best_hybrid import _concordance_index  # noqa: E402

# The eight tied baselines reported in davis.json (CI=0.7567 / R²=0.224)
TIED_REPS = ["drug_avalon", "drug_chemberta", "drug_ecfp4", "drug_maccs"]

OUT = Path("tests/results/phase5_baselines/dta/davis_investigation.json")
N_FOLDS = 5


def load_davis():
    """Returns (drug_smiles, protein_seqs, y, reps_dict, groups)."""
    from kirby.datasets.dta import load_dataset
    from kirby.representations.molecular import (
        create_avalon, create_chemberta, create_ecfp, create_maccs,
    )

    data = load_dataset("davis")
    drug_smiles = list(data["drug_smiles"])
    protein_seqs = list(data["protein_seqs"])
    y = np.asarray(data["labels"], dtype=np.float64)

    print(f"  davis: n={len(y)} pairs, "
          f"{len(set(drug_smiles))} unique drugs, "
          f"{len(set(protein_seqs))} unique proteins")

    # Cold-protein groups: one group per unique protein
    proteins_unique = sorted(set(protein_seqs))
    p_to_idx = {p: i for i, p in enumerate(proteins_unique)}
    groups = np.array([p_to_idx[p] for p in protein_seqs])

    # Generate the four tied drug reps (cheap: ECFP/MACCS/Avalon/ChemBERTa)
    drugs_unique = sorted(set(drug_smiles))
    d_to_idx = {d: i for i, d in enumerate(drugs_unique)}
    drug_indices = np.array([d_to_idx[d] for d in drug_smiles])

    rep_funcs = {
        "drug_ecfp4": create_ecfp,
        "drug_maccs": create_maccs,
        "drug_avalon": create_avalon,
        "drug_chemberta": create_chemberta,
    }
    reps = {}
    for name, fn in rep_funcs.items():
        print(f"  generating {name}...", end=" ", flush=True)
        t0 = time.time()
        unique_rep = fn(drugs_unique)
        reps[name] = np.asarray(unique_rep)[drug_indices]
        print(f"{reps[name].shape} ({time.time()-t0:.1f}s)")

    return drug_smiles, drug_indices, y, reps, groups


def drug_mean_predictor(y_train, drug_idx_train, drug_idx_test, global_mean):
    """For each test drug, return the mean training affinity for that drug.

    If a drug has no training samples, fall back to global_mean.
    """
    sums = np.zeros(int(max(drug_idx_train.max(), drug_idx_test.max())) + 1)
    counts = np.zeros_like(sums)
    np.add.at(sums, drug_idx_train, y_train)
    np.add.at(counts, drug_idx_train, 1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
    return means[drug_idx_test]


def score(y_true, y_pred):
    return {
        "ci": float(_concordance_index(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def main():
    drug_smiles, drug_idx, y, reps, groups = load_davis()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.zeros(len(y)), y, groups))

    # 1. Drug-mean baseline per fold
    drug_mean_metrics = []
    drug_mean_preds = []
    for train_idx, test_idx in splits:
        gm = float(np.mean(y[train_idx]))
        pred = drug_mean_predictor(
            y[train_idx], drug_idx[train_idx], drug_idx[test_idx], gm)
        drug_mean_metrics.append(score(y[test_idx], pred))
        drug_mean_preds.append((test_idx, pred))

    # 2. Ridge with each tied rep, dump preds per fold
    ridge_preds = {r: [] for r in TIED_REPS}
    ridge_metrics = {r: [] for r in TIED_REPS}
    for rep_name in TIED_REPS:
        X = reps[rep_name]
        for train_idx, test_idx in splits:
            scaler = StandardScaler()
            X_tr = np.nan_to_num(scaler.fit_transform(X[train_idx]))
            X_te = np.nan_to_num(scaler.transform(X[test_idx]))
            model = Ridge(alpha=1.0)
            model.fit(X_tr, y[train_idx])
            pred = model.predict(X_te)
            ridge_preds[rep_name].append((test_idx, pred))
            ridge_metrics[rep_name].append(score(y[test_idx], pred))

    # 3. Cross-rep prediction correlations (per fold) + vs drug-mean
    corr_matrix = {}  # {fold: {(rep_a, rep_b): corr}}
    corr_vs_drug_mean = {}  # {rep: [per-fold corr]}
    for rep_name in TIED_REPS:
        corr_vs_drug_mean[rep_name] = []
        for fold_i in range(N_FOLDS):
            _, p_rep = ridge_preds[rep_name][fold_i]
            _, p_dm = drug_mean_preds[fold_i]
            corr_vs_drug_mean[rep_name].append(
                float(np.corrcoef(p_rep, p_dm)[0, 1]))

    for fold_i in range(N_FOLDS):
        corr_matrix[fold_i] = {}
        for i, ra in enumerate(TIED_REPS):
            for rb in TIED_REPS[i + 1:]:
                _, p_a = ridge_preds[ra][fold_i]
                _, p_b = ridge_preds[rb][fold_i]
                corr_matrix[fold_i][f"{ra}__vs__{rb}"] = (
                    float(np.corrcoef(p_a, p_b)[0, 1]))

    # Aggregate
    def agg(metric_list):
        keys = metric_list[0].keys()
        return {k: {"mean": float(np.mean([m[k] for m in metric_list])),
                    "std": float(np.std([m[k] for m in metric_list]))}
                for k in keys}

    out = {
        "n_folds": N_FOLDS,
        "n_samples": int(len(y)),
        "n_drugs": int(len(np.unique(drug_idx))),
        "n_proteins": int(len(np.unique(groups))),
        "drug_mean_predictor": agg(drug_mean_metrics),
        "ridge_per_rep": {r: agg(ridge_metrics[r]) for r in TIED_REPS},
        "corr_pred_vs_drug_mean_per_fold": corr_vs_drug_mean,
        "corr_pred_vs_drug_mean_mean": {
            r: float(np.mean(corr_vs_drug_mean[r])) for r in TIED_REPS},
        "cross_rep_pred_correlations": corr_matrix,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")

    # Console summary
    dm = out["drug_mean_predictor"]
    print("\n=== Drug-mean predictor (5-fold, cold-protein) ===")
    print(f"  CI={dm['ci']['mean']:.4f}±{dm['ci']['std']:.4f}  "
          f"R²={dm['r2']['mean']:.4f}±{dm['r2']['std']:.4f}  "
          f"RMSE={dm['rmse']['mean']:.4f}")
    print("\n=== Ridge per rep ===")
    for r in TIED_REPS:
        m = out["ridge_per_rep"][r]
        print(f"  {r:<18} CI={m['ci']['mean']:.4f}  R²={m['r2']['mean']:.4f}  "
              f"corr_vs_drug_mean={out['corr_pred_vs_drug_mean_mean'][r]:.4f}")
    print("\nIf corr_vs_drug_mean ≈ 1.0 across all reps, the 'best baseline' "
          "is just per-drug mean fitting — rep choice is irrelevant.")


if __name__ == "__main__":
    main()
