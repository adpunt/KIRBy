#!/usr/bin/env python3
"""
QSAR Baseline Benchmark: ECFP4 + RF/LightGBM under Scaffold Splits
====================================================================

Tests the "boring baseline" (ECFP4 Morgan r=2 + RandomForest / LightGBM)
across datasets recommended by Pat Walters / Greg Landrum standards,
under Bemis-Murcko scaffold-grouped cross-validation.

Datasets tested:
  1. OpenADMET-ExpansionRx (HuggingFace) — LogD, KSOL, HLM CLint, MLM CLint
  2. Genentech Computational-ADME (GitHub) — HLM, RLM, Solubility, hPPB, rPPB, MDR1-MDCK ER
  3. AstraZeneca ChEMBL Solubility (DGL) — Aqueous solubility pH 7.4
  4. ChEMBL hERG Ki (REST API, CHEMBL240) — Binding affinity (pChEMBL)

Protocol:
  - 5-fold scaffold GroupKFold, repeated 3x with different scaffold-group shuffles
  - Reports R², MAE, Spearman ρ per fold + aggregated mean ± std
  - Logs fold diagnostics: N(train/test), y-range, NN-similarity distribution
  - Models: RandomForest (n_estimators=1000) and LightGBM

Requirements:
  pip install rdkit scikit-learn lightgbm pandas numpy scipy requests tqdm

Usage:
  python qsar_baseline_benchmark.py                    # run everything
  python qsar_baseline_benchmark.py --datasets openadmet genentech  # subset
  python qsar_baseline_benchmark.py --skip-download     # if data already cached
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ── RDKit ──────────────────────────────────────────────────────────────────
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

# ── LightGBM ──────────────────────────────────────────────────────────────
import lightgbm as lgb

import requests

warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.logger().setLevel(RDLogger.ERROR)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path("data_cache")
RESULTS_DIR = Path("results")
N_FOLDS = 5
N_REPEATS = 3
FP_RADIUS = 2
FP_NBITS = 2048

RF_PARAMS = dict(
    n_estimators=1000,
    max_features=0.3,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
)

LGBM_PARAMS = dict(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Molecule utilities
# ═══════════════════════════════════════════════════════════════════════════

def standardise_smiles(smi: str) -> Optional[str]:
    """Canonicalise SMILES, keep largest fragment, remove salts."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Largest fragment (salt removal)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_ecfp(smi: str, radius: int = FP_RADIUS, nbits: int = FP_NBITS) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan/ECFP fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_scaffold(smi: str) -> str:
    """Bemis-Murcko scaffold from SMILES. Falls back to SMILES itself."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return smi


def tanimoto_nn_similarity(fps_train: np.ndarray, fps_test: np.ndarray, sample_n: int = 500) -> np.ndarray:
    """Compute nearest-neighbour Tanimoto similarity from test→train.
    Samples at most `sample_n` test points for efficiency.
    """
    if len(fps_test) > sample_n:
        idx = np.random.choice(len(fps_test), sample_n, replace=False)
        fps_test = fps_test[idx]

    nn_sims = []
    for fp_test in fps_test:
        # Tanimoto for binary vectors: |A∩B| / |A∪B|
        intersect = np.sum(np.minimum(fp_test, fps_train), axis=1)
        union = np.sum(np.maximum(fp_test, fps_train), axis=1)
        union = np.where(union == 0, 1, union)  # avoid div/0
        sims = intersect / union
        nn_sims.append(np.max(sims))
    return np.array(nn_sims)


# ═══════════════════════════════════════════════════════════════════════════
# Data acquisition
# ═══════════════════════════════════════════════════════════════════════════

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"  [cached] {dest}")
        return dest
    log.info(f"  Downloading {desc or url}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"  → saved to {dest}")
    return dest


def fetch_openadmet() -> pd.DataFrame:
    """Download OpenADMET-ExpansionRx full dataset from HuggingFace."""
    log.info("━━━ Fetching OpenADMET-ExpansionRx dataset ━━━")

    # The full dataset (train+test with labels) released Jan 2026
    csv_url = (
        "https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/"
        "resolve/main/expansion_data_train.csv"
    )
    dest = CACHE_DIR / "openadmet_train.csv"
    download_file(csv_url, dest, "OpenADMET train split")

    # Also try to get the full combined data (train+test with labels)
    # The raw dataset has more rows including out-of-range measurements
    raw_url = (
        "https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/"
        "resolve/main/expansion_data_raw.csv"
    )
    raw_dest = CACHE_DIR / "openadmet_raw.csv"
    try:
        download_file(raw_url, raw_dest, "OpenADMET raw (full)")
        # Use raw if it's bigger and contains more data
        df_raw = pd.read_csv(raw_dest)
        df_train = pd.read_csv(dest)
        if len(df_raw) > len(df_train):
            log.info(f"  Using raw dataset ({len(df_raw)} rows) over train ({len(df_train)} rows)")
            return df_raw
    except Exception as e:
        log.warning(f"  Could not fetch raw dataset: {e}")

    return pd.read_csv(dest)


def fetch_genentech() -> pd.DataFrame:
    """Download Genentech Computational-ADME dataset from GitHub."""
    log.info("━━━ Fetching Genentech Computational-ADME dataset ━━━")
    csv_url = (
        "https://raw.githubusercontent.com/molecularinformatics/"
        "Computational-ADME/main/ADME_public_set_3521.csv"
    )
    dest = CACHE_DIR / "genentech_adme_3521.csv"
    download_file(csv_url, dest, "Genentech ADME 3521")
    return pd.read_csv(dest)


def fetch_az_solubility() -> pd.DataFrame:
    """Download AstraZeneca ChEMBL Solubility from DGL data server."""
    log.info("━━━ Fetching AstraZeneca ChEMBL Solubility dataset ━━━")
    csv_url = "https://data.dgl.ai/dataset/AstraZeneca_ChEMBL_Solubility.csv"
    dest = CACHE_DIR / "az_chembl_solubility.csv"
    download_file(csv_url, dest, "AstraZeneca ChEMBL Solubility")
    return pd.read_csv(dest)


def fetch_chembl_herg_ki() -> pd.DataFrame:
    """Extract hERG (CHEMBL240) Ki data via ChEMBL REST API.
    
    Implements Landrum's strict curation protocol:
      - standard_type = Ki only
      - pchembl_value not null
      - standard_relation = '='
      - data_validity_comment is null
      - confidence_score >= 9
      - target_type = SINGLE PROTEIN
    """
    log.info("━━━ Fetching ChEMBL hERG (CHEMBL240) Ki data via REST API ━━━")
    
    cached = CACHE_DIR / "chembl_herg_ki.csv"
    if cached.exists():
        log.info(f"  [cached] {cached}")
        return pd.read_csv(cached)
    
    TARGET_ID = "CHEMBL240"
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    
    all_records = []
    offset = 0
    limit = 1000
    
    while True:
        params = {
            "target_chembl_id": TARGET_ID,
            "standard_type": "Ki",
            "pchembl_value__isnull": "false",
            "standard_relation": "=",
            "data_validity_comment__isnull": "true",
            "limit": limit,
            "offset": offset,
            "format": "json",
        }
        
        log.info(f"  Querying ChEMBL API (offset={offset})...")
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            log.error(f"  ChEMBL API request failed: {e}")
            break
        
        activities = data.get("activities", [])
        if not activities:
            break
        
        for act in activities:
            all_records.append({
                "molecule_chembl_id": act.get("molecule_chembl_id"),
                "canonical_smiles": act.get("canonical_smiles"),
                "pchembl_value": act.get("pchembl_value"),
                "standard_type": act.get("standard_type"),
                "standard_relation": act.get("standard_relation"),
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units"),
                "assay_chembl_id": act.get("assay_chembl_id"),
                "assay_type": act.get("assay_type"),
                "target_chembl_id": act.get("target_chembl_id"),
                "data_validity_comment": act.get("data_validity_comment"),
            })
        
        # Check if there are more pages
        page_meta = data.get("page_meta", {})
        next_url = page_meta.get("next")
        if next_url is None:
            break
        offset += limit
        time.sleep(0.5)  # Be polite to the API
    
    if not all_records:
        log.error("  No hERG Ki records retrieved from ChEMBL!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    log.info(f"  Retrieved {len(df)} raw hERG Ki activity records")
    
    # Post-filter: confidence_score and target_type are on the assay, not activity.
    # We already filtered by target_chembl_id directly, which is sufficient.
    # For extra strictness, we filter binding assays only:
    if "assay_type" in df.columns:
        n_before = len(df)
        df = df[df["assay_type"] == "B"].copy()
        log.info(f"  Filtered to binding assays: {n_before} → {len(df)}")
    
    # Convert pchembl to float
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])
    
    # Deduplicate: median pchembl per compound, exclude high-variance compounds
    grouped = df.groupby("canonical_smiles")["pchembl_value"]
    medians = grouped.median().reset_index()
    stds = grouped.std().reset_index().rename(columns={"pchembl_value": "std"})
    merged = medians.merge(stds, on="canonical_smiles")
    
    # Remove compounds with std > 1.0 log unit (Landrum protocol)
    n_before = len(merged)
    merged = merged[(merged["std"].isna()) | (merged["std"] <= 1.0)]
    log.info(f"  Removed high-variance compounds: {n_before} → {len(merged)}")
    
    result = merged[["canonical_smiles", "pchembl_value"]].copy()
    result.columns = ["SMILES", "pChEMBL"]
    
    cached.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cached, index=False)
    log.info(f"  Final hERG Ki dataset: {len(result)} compounds, saved to {cached}")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Dataset preparation (standardise, featurise, split-ready)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    dataset_name: str,
    log_transform: bool = False,
) -> Optional[dict]:
    """Standardise molecules, compute ECFP4, assign scaffolds.
    
    Returns dict with keys: X, y, smiles, scaffolds, dataset_name, target_name
    or None if the dataset is too small after cleaning.
    """
    log.info(f"  Preparing: {dataset_name} / {target_col}")
    
    # ── Clean target values ────────────────────────────────────────────
    sub = df[[smiles_col, target_col]].copy()
    sub = sub.dropna(subset=[target_col])
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna(subset=[target_col])
    
    if len(sub) < 100:
        log.warning(f"    Only {len(sub)} valid rows — skipping")
        return None
    
    # ── Standardise SMILES ─────────────────────────────────────────────
    sub["std_smiles"] = sub[smiles_col].apply(standardise_smiles)
    sub = sub.dropna(subset=["std_smiles"])
    
    # Deduplicate by canonical SMILES: take median target
    sub = sub.groupby("std_smiles").agg({target_col: "median"}).reset_index()
    
    if len(sub) < 100:
        log.warning(f"    Only {len(sub)} unique molecules — skipping")
        return None
    
    # ── Target transform ───────────────────────────────────────────────
    y = sub[target_col].values.copy()
    if log_transform:
        # Clip to positive values before log
        y = np.clip(y, 1e-10, None)
        y = np.log10(y)
        log.info(f"    Applied log10 transform")
    
    # Remove infinite / nan after transform
    valid_mask = np.isfinite(y)
    sub = sub[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    
    if len(sub) < 100:
        log.warning(f"    Only {len(sub)} valid after transform — skipping")
        return None
    
    # ── Featurise (ECFP4) ─────────────────────────────────────────────
    fps = []
    valid_idx = []
    for i, smi in enumerate(sub["std_smiles"]):
        fp = smiles_to_ecfp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    
    X = np.array(fps)
    y = y[valid_idx]
    smiles_arr = sub["std_smiles"].values[valid_idx]
    
    if len(X) < 100:
        log.warning(f"    Only {len(X)} valid fingerprints — skipping")
        return None
    
    # ── Scaffolds ──────────────────────────────────────────────────────
    scaffolds = np.array([get_scaffold(s) for s in smiles_arr])
    n_unique_scaffolds = len(set(scaffolds))
    
    log.info(
        f"    Ready: {len(X)} molecules, {n_unique_scaffolds} unique scaffolds, "
        f"y range [{y.min():.2f}, {y.max():.2f}], y std {y.std():.3f}"
    )
    
    return {
        "X": X,
        "y": y,
        "smiles": smiles_arr,
        "scaffolds": scaffolds,
        "dataset_name": dataset_name,
        "target_name": target_col,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold-grouped cross-validation
# ═══════════════════════════════════════════════════════════════════════════

def run_scaffold_cv(
    data: dict,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
) -> dict:
    """Run scaffold-grouped K-fold CV with RF and LightGBM.
    
    Returns a results dict with per-fold and aggregated metrics.
    """
    X, y = data["X"], data["y"]
    scaffolds = data["scaffolds"]
    dataset_name = data["dataset_name"]
    target_name = data["target_name"]
    
    log.info(f"\n{'='*70}")
    log.info(f"Running scaffold CV: {dataset_name} / {target_name}")
    log.info(f"  N={len(X)}, folds={n_folds}, repeats={n_repeats}")
    log.info(f"{'='*70}")
    
    # Encode scaffolds as integer groups
    unique_scaffolds = list(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    groups = np.array([scaffold_to_id[s] for s in scaffolds])
    
    all_results = {"RF": defaultdict(list), "LGBM": defaultdict(list)}
    fold_diagnostics = []
    
    for repeat_i in range(n_repeats):
        # Shuffle scaffold group assignments for this repeat
        rng = np.random.RandomState(repeat_i * 1000 + 42)
        
        # Shuffle the group labels (permute which scaffolds go in which fold)
        perm = rng.permutation(len(unique_scaffolds))
        shuffled_groups = np.array([perm[scaffold_to_id[s]] for s in scaffolds])
        
        gkf = GroupKFold(n_splits=n_folds)
        
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=shuffled_groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # ── Fold diagnostics ───────────────────────────────────────
            diag = {
                "repeat": repeat_i,
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "y_train_min": float(y_train.min()),
                "y_train_max": float(y_train.max()),
                "y_train_std": float(y_train.std()),
                "y_test_min": float(y_test.min()),
                "y_test_max": float(y_test.max()),
                "y_test_std": float(y_test.std()),
            }
            
            # NN similarity (sampled for speed)
            try:
                nn_sims = tanimoto_nn_similarity(X_train, X_test, sample_n=200)
                diag["nn_sim_mean"] = float(nn_sims.mean())
                diag["nn_sim_median"] = float(np.median(nn_sims))
                diag["frac_nn_sim_gt_05"] = float(np.mean(nn_sims > 0.5))
            except Exception:
                diag["nn_sim_mean"] = None
                diag["nn_sim_median"] = None
                diag["frac_nn_sim_gt_05"] = None
            
            fold_diagnostics.append(diag)
            
            # ── Random Forest ──────────────────────────────────────────
            rf = RandomForestRegressor(**RF_PARAMS)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            r2_rf = r2_score(y_test, y_pred_rf)
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            rho_result = stats.spearmanr(y_test, y_pred_rf)
            rho_rf = rho_result.correlation if np.isfinite(rho_result.correlation) else 0.0
            
            all_results["RF"]["r2"].append(r2_rf)
            all_results["RF"]["mae"].append(mae_rf)
            all_results["RF"]["spearman"].append(rho_rf)
            
            # ── LightGBM ──────────────────────────────────────────────
            lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
            lgbm.fit(X_train, y_train)
            y_pred_lgbm = lgbm.predict(X_test)
            
            r2_lgbm = r2_score(y_test, y_pred_lgbm)
            mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
            rho_result = stats.spearmanr(y_test, y_pred_lgbm)
            rho_lgbm = rho_result.correlation if np.isfinite(rho_result.correlation) else 0.0
            
            all_results["LGBM"]["r2"].append(r2_lgbm)
            all_results["LGBM"]["mae"].append(mae_lgbm)
            all_results["LGBM"]["spearman"].append(rho_lgbm)
            
            log.info(
                f"  repeat={repeat_i} fold={fold_i}: "
                f"RF R²={r2_rf:.3f} MAE={mae_rf:.3f} ρ={rho_rf:.3f} | "
                f"LGBM R²={r2_lgbm:.3f} MAE={mae_lgbm:.3f} ρ={rho_lgbm:.3f} | "
                f"N_train={len(train_idx)} N_test={len(test_idx)} "
                f"NN_sim>{0.5}={diag.get('frac_nn_sim_gt_05', 'N/A')}"
            )
    
    # ── Aggregate ──────────────────────────────────────────────────────
    summary = {}
    for model_name, metrics in all_results.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary[model_name][metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": [float(v) for v in arr],
            }
    
    log.info(f"\n{'─'*70}")
    log.info(f"SUMMARY: {dataset_name} / {target_name}")
    log.info(f"{'─'*70}")
    for model_name in ["RF", "LGBM"]:
        r2 = summary[model_name]["r2"]
        mae = summary[model_name]["mae"]
        rho = summary[model_name]["spearman"]
        passes = "✓ PASS" if r2["mean"] > 0.5 else "✗ FAIL"
        log.info(
            f"  {model_name:5s}: R² = {r2['mean']:.3f} ± {r2['std']:.3f}  "
            f"MAE = {mae['mean']:.3f} ± {mae['std']:.3f}  "
            f"ρ = {rho['mean']:.3f} ± {rho['std']:.3f}  "
            f"[R²>0.5? {passes}]"
        )
    log.info(f"{'─'*70}\n")
    
    return {
        "dataset": dataset_name,
        "target": target_name,
        "n_molecules": len(X),
        "n_scaffolds": len(set(scaffolds)),
        "y_range": [float(y.min()), float(y.max())],
        "y_std": float(y.std()),
        "summary": summary,
        "fold_diagnostics": fold_diagnostics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dataset-specific preparation pipelines
# ═══════════════════════════════════════════════════════════════════════════

def prepare_openadmet_tasks(df: pd.DataFrame) -> list[dict]:
    """Prepare individual endpoint tasks from OpenADMET dataset."""
    log.info("\n━━━ Preparing OpenADMET-ExpansionRx endpoints ━━━")
    
    # Column names as found in the HuggingFace dataset
    # LogD: leave untransformed
    # KSOL, HLM CLint, MLM CLint: log10 transform (strictly positive)
    # Caco-2 Permeability: log10 transform
    # MPPB, MBPB, MGMB: these are % bound; could use logit but we'll try raw first
    
    # Auto-detect SMILES column
    smiles_col = None
    for candidate in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        log.error(f"  Cannot find SMILES column in OpenADMET data. Columns: {list(df.columns)}")
        return []
    
    log.info(f"  Columns found: {list(df.columns)}")
    log.info(f"  SMILES column: {smiles_col}")
    log.info(f"  Shape: {df.shape}")
    
    # Define endpoints and their transform strategy
    endpoint_configs = [
        # (column_name_pattern, display_name, log_transform)
        ("LogD", "LogD", False),
        ("KSOL", "KSOL", True),
        ("HLM CLint", "HLM_CLint", True),
        ("MLM CLint", "MLM_CLint", True),
        ("Caco-2 Permeability Papp", "Caco2_Papp", True),
        ("Caco-2 Permeability Efflux", "Caco2_Efflux", True),
    ]
    
    tasks = []
    for col_pattern, display_name, do_log in endpoint_configs:
        # Find matching column
        matching = [c for c in df.columns if col_pattern in c]
        if not matching:
            log.info(f"  Endpoint '{col_pattern}' not found, skipping")
            continue
        col = matching[0]
        n_valid = df[col].notna().sum()
        log.info(f"  Found endpoint: {col} ({n_valid} non-null values)")
        
        result = prepare_dataset(
            df, smiles_col, col,
            dataset_name=f"OpenADMET-{display_name}",
            log_transform=do_log,
        )
        if result is not None:
            tasks.append(result)
    
    return tasks


def prepare_genentech_tasks(df: pd.DataFrame) -> list[dict]:
    """Prepare individual endpoint tasks from Genentech ADME dataset."""
    log.info("\n━━━ Preparing Genentech Computational-ADME endpoints ━━━")
    
    log.info(f"  Columns found: {list(df.columns)}")
    log.info(f"  Shape: {df.shape}")
    
    # Find SMILES column
    smiles_col = None
    for candidate in ["Smiles", "SMILES", "smiles", "canonical_smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        log.error(f"  Cannot find SMILES column. Columns: {list(df.columns)}")
        return []
    
    # The values in the CSV are already log-transformed per the README:
    # "experimental log(properties) for six endpoints"
    # So we do NOT apply additional log transform
    
    # Auto-detect endpoint columns (everything that's not SMILES/vendor/ID)
    exclude_patterns = ["smiles", "vendor", "id", "name", "Unnamed"]
    endpoint_cols = [
        c for c in df.columns
        if not any(p.lower() in c.lower() for p in exclude_patterns)
        and df[c].dtype in [np.float64, np.float32, np.int64, "float64", "float32"]
    ]
    
    # Also try to find them by known patterns
    if not endpoint_cols:
        endpoint_cols = [c for c in df.columns if "LOG" in c.upper() or "HLM" in c or "RLM" in c]
    
    log.info(f"  Detected endpoint columns: {endpoint_cols}")
    
    tasks = []
    for col in endpoint_cols:
        n_valid = df[col].notna().sum()
        if n_valid < 100:
            log.info(f"  Skipping {col}: only {n_valid} values")
            continue
        
        # Clean display name
        display_name = col.replace("LOG ", "").replace("(", "").replace(")", "").replace(" ", "_")
        display_name = display_name[:30]  # truncate long names
        
        result = prepare_dataset(
            df, smiles_col, col,
            dataset_name=f"Genentech-{display_name}",
            log_transform=False,  # Already log-transformed
        )
        if result is not None:
            tasks.append(result)
    
    return tasks


def prepare_az_solubility_tasks(df: pd.DataFrame) -> list[dict]:
    """Prepare AstraZeneca solubility task."""
    log.info("\n━━━ Preparing AstraZeneca ChEMBL Solubility ━━━")
    
    log.info(f"  Columns found: {list(df.columns)}")
    log.info(f"  Shape: {df.shape}")
    
    # Find SMILES column
    smiles_col = None
    for candidate in ["Smiles", "SMILES", "smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        log.error(f"  Cannot find SMILES column. Columns: {list(df.columns)}")
        return []
    
    # Find solubility column
    sol_col = None
    for candidate in ["Solubility", "solubility", "SOLUBILITY"]:
        if candidate in df.columns:
            sol_col = candidate
            break
    if sol_col is None:
        # Try any numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sol_col = numeric_cols[0]
            log.info(f"  Using numeric column '{sol_col}' as solubility")
        else:
            log.error(f"  Cannot find solubility column")
            return []
    
    # Values are in nM; the DGL loader applies log by default
    # Raw range: [100, 1513600], log range: [4.61, 14.23]
    # We'll apply log10 ourselves
    result = prepare_dataset(
        df, smiles_col, sol_col,
        dataset_name="AZ-ChEMBL-Solubility",
        log_transform=True,  # Convert from nM to log10(nM)
    )
    
    return [result] if result is not None else []


def prepare_herg_tasks(df: pd.DataFrame) -> list[dict]:
    """Prepare hERG Ki task from ChEMBL extraction."""
    log.info("\n━━━ Preparing ChEMBL hERG Ki ━━━")
    
    if df.empty:
        log.error("  hERG dataset is empty!")
        return []
    
    log.info(f"  Columns found: {list(df.columns)}")
    log.info(f"  Shape: {df.shape}")
    
    smiles_col = "SMILES" if "SMILES" in df.columns else df.columns[0]
    target_col = "pChEMBL" if "pChEMBL" in df.columns else df.columns[1]
    
    # pChEMBL values are already -log10(molar), no additional transform needed
    result = prepare_dataset(
        df, smiles_col, target_col,
        dataset_name="ChEMBL-hERG-Ki",
        log_transform=False,
    )
    
    return [result] if result is not None else []


# ═══════════════════════════════════════════════════════════════════════════
# Results output
# ═══════════════════════════════════════════════════════════════════════════

def save_results(all_results: list[dict]):
    """Save results to JSON and print a summary table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    json_path = RESULTS_DIR / "qsar_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nFull results saved to {json_path}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("QSAR BASELINE BENCHMARK RESULTS")
    print("ECFP4 (Morgan r=2, 2048 bits) + RF/LightGBM under scaffold-grouped 5-fold CV × 3 repeats")
    print("=" * 100)
    
    header = f"{'Dataset':<40s} {'Model':>5s} {'N':>6s} {'R² mean±std':>14s} {'MAE mean±std':>14s} {'ρ mean±std':>14s} {'R²>0.5?':>8s}"
    print(header)
    print("-" * 100)
    
    for res in all_results:
        ds = f"{res['dataset']}/{res['target']}"
        if len(ds) > 38:
            ds = ds[:35] + "..."
        n = res["n_molecules"]
        
        for model_name in ["RF", "LGBM"]:
            s = res["summary"][model_name]
            r2_str = f"{s['r2']['mean']:.3f}±{s['r2']['std']:.3f}"
            mae_str = f"{s['mae']['mean']:.3f}±{s['mae']['std']:.3f}"
            rho_str = f"{s['spearman']['mean']:.3f}±{s['spearman']['std']:.3f}"
            passes = "✓" if s["r2"]["mean"] > 0.5 else "✗"
            
            print(f"{ds:<40s} {model_name:>5s} {n:>6d} {r2_str:>14s} {mae_str:>14s} {rho_str:>14s} {passes:>8s}")
            ds = ""  # Only show dataset name once
    
    print("=" * 100)
    
    # Print fold diagnostics summary
    print("\nFOLD DIAGNOSTICS (averaged across all folds):")
    print("-" * 80)
    for res in all_results:
        diags = res.get("fold_diagnostics", [])
        if not diags:
            continue
        nn_sims = [d["frac_nn_sim_gt_05"] for d in diags if d.get("frac_nn_sim_gt_05") is not None]
        avg_nn = np.mean(nn_sims) if nn_sims else float("nan")
        test_sizes = [d["n_test"] for d in diags]
        train_sizes = [d["n_train"] for d in diags]
        
        print(
            f"  {res['dataset']}/{res['target']}: "
            f"avg train={np.mean(train_sizes):.0f}, avg test={np.mean(test_sizes):.0f}, "
            f"frac(NN_sim>0.5)={avg_nn:.2%}, "
            f"y_range=[{res['y_range'][0]:.2f}, {res['y_range'][1]:.2f}]"
        )
    print()
    
    # Save CSV summary for easy import
    csv_rows = []
    for res in all_results:
        for model_name in ["RF", "LGBM"]:
            s = res["summary"][model_name]
            csv_rows.append({
                "dataset": res["dataset"],
                "target": res["target"],
                "model": model_name,
                "n_molecules": res["n_molecules"],
                "n_scaffolds": res["n_scaffolds"],
                "r2_mean": s["r2"]["mean"],
                "r2_std": s["r2"]["std"],
                "mae_mean": s["mae"]["mean"],
                "mae_std": s["mae"]["std"],
                "spearman_mean": s["spearman"]["mean"],
                "spearman_std": s["spearman"]["std"],
                "r2_gt_05": s["r2"]["mean"] > 0.5,
            })
    
    csv_path = RESULTS_DIR / "qsar_baseline_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info(f"Summary CSV saved to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

AVAILABLE_DATASETS = ["openadmet", "genentech", "az_solubility", "herg"]


def main():
    parser = argparse.ArgumentParser(
        description="QSAR Baseline Benchmark: ECFP4 + RF/LightGBM under Scaffold Splits"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE_DATASETS + ["all"],
        default=["all"],
        help="Which datasets to test (default: all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, use cached data only",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS,
        help=f"Number of CV folds (default: {N_FOLDS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help=f"Number of CV repeats (default: {N_REPEATS})",
    )
    parser.add_argument(
        "--fp-bits",
        type=int,
        default=FP_NBITS,
        help=f"Fingerprint bits (default: {FP_NBITS})",
    )
    args = parser.parse_args()
    
    n_folds = args.folds
    n_repeats = args.repeats
    fp_nbits = args.fp_bits
    
    datasets_to_run = AVAILABLE_DATASETS if "all" in args.datasets else args.datasets
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 70)
    log.info("QSAR Baseline Benchmark")
    log.info(f"  Datasets: {datasets_to_run}")
    log.info(f"  FP: Morgan r={FP_RADIUS}, {fp_nbits} bits")
    log.info(f"  CV: {n_folds}-fold scaffold GroupKFold × {n_repeats} repeats")
    log.info(f"  Models: RF, LightGBM")
    log.info("=" * 70)
    
    all_tasks = []
    
    # ── 1. OpenADMET-ExpansionRx ───────────────────────────────────────
    if "openadmet" in datasets_to_run:
        try:
            df = fetch_openadmet()
            tasks = prepare_openadmet_tasks(df)
            all_tasks.extend(tasks)
        except Exception as e:
            log.error(f"Failed to load OpenADMET: {e}", exc_info=True)
    
    # ── 2. Genentech Computational-ADME ────────────────────────────────
    if "genentech" in datasets_to_run:
        try:
            df = fetch_genentech()
            tasks = prepare_genentech_tasks(df)
            all_tasks.extend(tasks)
        except Exception as e:
            log.error(f"Failed to load Genentech ADME: {e}", exc_info=True)
    
    # ── 3. AstraZeneca ChEMBL Solubility ───────────────────────────────
    if "az_solubility" in datasets_to_run:
        try:
            df = fetch_az_solubility()
            tasks = prepare_az_solubility_tasks(df)
            all_tasks.extend(tasks)
        except Exception as e:
            log.error(f"Failed to load AZ Solubility: {e}", exc_info=True)
    
    # ── 4. ChEMBL hERG Ki ─────────────────────────────────────────────
    if "herg" in datasets_to_run:
        try:
            df = fetch_chembl_herg_ki()
            tasks = prepare_herg_tasks(df)
            all_tasks.extend(tasks)
        except Exception as e:
            log.error(f"Failed to load hERG Ki: {e}", exc_info=True)
    
    # ── Run benchmarks ─────────────────────────────────────────────────
    if not all_tasks:
        log.error("No valid tasks to benchmark! Check data downloads and errors above.")
        sys.exit(1)
    
    log.info(f"\n{'#'*70}")
    log.info(f"  {len(all_tasks)} tasks ready for benchmarking")
    log.info(f"{'#'*70}\n")
    
    all_results = []
    for task in all_tasks:
        try:
            result = run_scaffold_cv(task, n_folds=n_folds, n_repeats=n_repeats)
            all_results.append(result)
        except Exception as e:
            log.error(f"Failed on {task['dataset_name']}/{task['target_name']}: {e}", exc_info=True)
    
    # ── Save & display ─────────────────────────────────────────────────
    if all_results:
        save_results(all_results)
    else:
        log.error("No results generated!")
        sys.exit(1)


if __name__ == "__main__":
    main()