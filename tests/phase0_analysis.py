"""
Phase 0 Analysis - Global Screening of Model-Representation Pairs
Generates Figures 1-2 and Supplementary S1-S3

COMBINED VERSION:
- Loads OLD phase0c data from ../../qsar_qm_models/results
- Loads NEW graph data from results/phase1_graphs_updated
- Generates ALL original figures with combined data

Key metrics used (NO AUC):
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention percentage: (R²_high / R²_baseline) * 100
- Baseline R² at σ=0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL OF CHEMINFORMATICS STYLE
# ============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Color palettes
REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'graph': '#949494',
    'pdv-3d': '#56B4E9',
    'random_smiles': '#756bb1',
    'randomized_smiles': '#756bb1',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'xgboost': '#e74c3c',
    'gauche': '#9b59b6',
    'qrf': '#16a085',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'dnn_bnn_full': '#e67e22',
    'dnn_bnn_last': '#95a5a6',
    'dnn_bnn_variational': '#d35400',
    'GCN': '#1f77b4',
    'GAT': '#ff7f0e',
    'GIN': '#2ca02c',
    'MPNN': '#d62728',
}

# Models to exclude from analysis
mlp_mtl_models = ['mlp', 'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational',
                  'residual_mlp', 'residual_mlp_bnn_full', 'residual_mlp_bnn_last',
                  'residual_mlp_bnn_variational',
                  'mtl', 'mtl_bnn_full', 'mtl_bnn_last', 'mtl_bnn_variational',
                  'flexible_dnn', 'flexible_bnn_full', 'flexible_bnn_last',
                  'flexible_bnn_variational']

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_representation(rep):
    """Format representation names for display"""
    rep_map = {
        'ecfp4': 'ECFP4',
        'pdv': 'PDV',
        'pdv-3d': 'PDV-3D',
        'sns': 'SNS',
        'smiles': 'SMILES',
        'randomized_smiles': 'R-SMILES',
        'random_smiles': 'R-SMILES',
        'graph': 'Graph',
    }
    return rep_map.get(rep.lower(), rep.upper())


def format_model(model):
    """Format model names for display"""
    model_map = {
        'rf': 'RF',
        'qrf': 'QRF',
        'xgboost': 'XGBoost',
        'ngboost': 'NGBoost',
        'dnn': 'DNN',
        'mlp': 'MLP',
        'gauche': 'GP',
        'gcn': 'GCN',
        'gin': 'GIN',
        'gat': 'GAT',
        'mpnn': 'MPNN',
        'svm': 'SVM',
        'dnn_bnn_full': 'DNN-BNN-Full',
        'dnn_bnn_last': 'DNN-BNN-Last',
        'dnn_bnn_variational': 'DNN-BNN-Var',
    }
    return model_map.get(model.lower(), model.capitalize())

# ============================================================================
# DATA LOADING - COMBINED OLD + NEW
# ============================================================================

def load_old_phase0c_data(results_dir="../../qsar_qm_models/results"):
    """Load original Phase 0c screening results"""
    print("\n" + "="*80)
    print("LOADING OLD PHASE 0C SCREENING DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    screening_files = list(results_dir.glob("phase0c_screen_*.csv"))
    
    if not screening_files:
        print("WARNING: No phase0c_screen_*.csv files found!")
        return pd.DataFrame()
    
    print(f"Found {len(screening_files)} screening files")
    
    all_data = []
    for filepath in screening_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['model'] = combined_df['model'].str.replace('_split', '', regex=False)
    
    print(f"\nRaw data loaded: {len(combined_df)} rows")
    
    # Filter catastrophic failures
    print("Filtering catastrophic failures...")
    print(f"  Before: {len(combined_df)} rows")
    combined_df = combined_df[combined_df['r2'] > -10]
    print(f"  After R² > -10 filter: {len(combined_df)} rows")
    
    # Filter configs with terrible baseline
    baseline_check = combined_df[combined_df['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    bad_configs = baseline_check[baseline_check < 0.1].index
    
    if len(bad_configs) > 0:
        print(f"  Removing {len(bad_configs)} configs with baseline R² < 0.1")
        for model, rep in bad_configs:
            combined_df = combined_df[~((combined_df['model'] == model) & (combined_df['rep'] == rep))]
        print(f"  After baseline filter: {len(combined_df)} rows")
    
    # Average across iterations for same model/rep/sigma
    results = combined_df.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    # Filter out MLP/MTL
    print(f"\nFiltering MLP/MTL models...")
    print(f"  Before: {len(results)} rows")
    results = results[~results['model'].isin(mlp_mtl_models)]
    print(f"  After: {len(results)} rows")
    
    print(f"Final aggregated data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    
    return results


def load_new_graph_data(results_dir="results"):
    """Load NEW graph screening data from phase1_graphs_updated"""
    print("\n" + "="*80)
    print("LOADING NEW GRAPH SCREENING DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    graph_dir = results_dir / "phase1_graphs_updated"
    
    if not graph_dir.exists():
        print(f"WARNING: Graph directory not found: {graph_dir}")
        return pd.DataFrame()
    
    all_data = []
    for seed_dir in sorted(graph_dir.glob("seed_*")):
        seed = seed_dir.name.split('_')[1]
        all_results = seed_dir / "all_results.csv"
        
        if all_results.exists():
            try:
                df = pd.read_csv(all_results)
                df['iteration'] = int(seed)
                all_data.append(df)
            except Exception as e:
                print(f"Warning: {seed_dir.name}: {e}")
    
    if not all_data:
        print("WARNING: No graph data found")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure representation column
    if 'representation' not in combined_df.columns:
        combined_df['representation'] = 'graph'
    
    print(f"Loaded {len(combined_df)} graph result rows")
    print(f"Unique models: {combined_df['model'].nunique()}")
    
    # Filter catastrophic failures
    print("Filtering catastrophic failures...")
    print(f"  Before: {len(combined_df)} rows")
    combined_df = combined_df[combined_df['r2'] > -10]
    print(f"  After R² > -10 filter: {len(combined_df)} rows")
    
    # Average across iterations
    results = combined_df.groupby(['model', 'representation', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'iteration': 'n_seeds'}, inplace=True)
    
    # Filter out MLP/MTL
    print(f"\nFiltering MLP/MTL models...")
    print(f"  Before: {len(results)} rows")
    results = results[~results['model'].isin(mlp_mtl_models)]
    print(f"  After: {len(results)} rows")
    
    print(f"Final aggregated data: {len(results)} rows")
    print(f"Models: {sorted(results['model'].unique())}")
    
    return results


def load_screening_results(old_results_dir="../../qsar_qm_models/results", 
                           new_results_dir="results"):
    """Combine old phase0c and new graph screening data"""
    print("\n" + "="*80)
    print("COMBINING ALL SCREENING DATA")
    print("="*80)
    
    phase0c_data = load_old_phase0c_data(old_results_dir)
    graph_data = load_new_graph_data(new_results_dir)
    
    all_dfs = [df for df in [phase0c_data, graph_data] if len(df) > 0]
    
    if not all_dfs:
        print("ERROR: No data loaded!")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nCombined data: {len(combined)} rows")
    print(f"Unique models: {combined['model'].nunique()}")
    print(f"Unique representations: {combined['representation'].nunique()}")
    print(f"Sigma values: {sorted(combined['sigma'].unique())}")
    
    return combined


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.6):
    """
    Calculate robustness metrics for each model-representation pair
    
    Metrics calculated:
    - baseline_r2: R² at σ=0
    - r2_high: R² at σ=sigma_high
    - retention_pct: (r2_high / baseline_r2) * 100
    - nsi_r2: slope of R² vs σ (Noise Sensitivity Index)
    - nsi_rmse: slope of RMSE vs σ
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS (σ_high = {sigma_high})")
    print("="*80)
    
    metrics_list = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'model': model,
            'representation': rep,
        }
        
        # Baseline performance at σ=0
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # Performance at high noise
        sigma_h = group[np.abs(group['sigma'] - sigma_high) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Retention percentage
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            if metrics['baseline_r2'] != 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # Calculate NSI (slope of performance vs sigma)
        if len(group) >= 3:
            try:
                # R² NSI
                slope_r2, intercept_r2, r_val_r2, p_val_r2, _ = stats.linregress(
                    group['sigma'], group['r2']
                )
                metrics['nsi_r2'] = slope_r2
                metrics['nsi_r2_pval'] = p_val_r2
                metrics['nsi_r2_r'] = r_val_r2
                
                # Relative NSI (normalized by baseline)
                if intercept_r2 != 0:
                    metrics['nsi_r2_relative'] = slope_r2 / abs(intercept_r2)
                else:
                    metrics['nsi_r2_relative'] = np.nan
                
                # RMSE NSI
                slope_rmse, _, r_val_rmse, p_val_rmse, _ = stats.linregress(
                    group['sigma'], group['rmse']
                )
                metrics['nsi_rmse'] = slope_rmse
                metrics['nsi_rmse_pval'] = p_val_rmse
                metrics['nsi_rmse_r'] = r_val_rmse
                
            except Exception as e:
                metrics['nsi_r2'] = np.nan
                metrics['nsi_r2_relative'] = np.nan
                metrics['nsi_rmse'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
            metrics['nsi_r2_relative'] = np.nan
            metrics['nsi_rmse'] = np.nan
        
        # Performance at intermediate sigmas
        for sig in [0.2, 0.3, 0.4]:
            sigma_val = group[np.abs(group['sigma'] - sig) < 0.05]
            if len(sigma_val) > 0:
                metrics[f'r2_s{sig}'] = sigma_val['r2'].values[0]
                metrics[f'rmse_s{sig}'] = sigma_val['rmse'].values[0]
            else:
                metrics[f'r2_s{sig}'] = np.nan
                metrics[f'rmse_s{sig}'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    # Summary statistics
    print(f"\nBaseline R² statistics:")
    print(f"  Mean: {metrics_df['baseline_r2'].mean():.4f}")
    print(f"  Median: {metrics_df['baseline_r2'].median():.4f}")
    print(f"  Range: [{metrics_df['baseline_r2'].min():.4f}, {metrics_df['baseline_r2'].max():.4f}]")
    
    print(f"\nRetention at σ={sigma_high} statistics:")
    print(f"  Mean: {metrics_df['retention_pct'].mean():.2f}%")
    print(f"  Median: {metrics_df['retention_pct'].median():.2f}%")
    print(f"  Range: [{metrics_df['retention_pct'].min():.2f}%, {metrics_df['retention_pct'].max():.2f}%]")
    
    print(f"\nNSI (R²) statistics:")
    print(f"  Mean: {metrics_df['nsi_r2'].mean():.4f}")
    print(f"  Median: {metrics_df['nsi_r2'].median():.4f}")
    
    return metrics_df


def define_robustness_score(metrics_df):
    """
    Define composite robustness score based on:
    1. Low absolute NSI (slow degradation)
    2. High retention percentage
    3. Reasonable baseline (filtered already)
    """
    # Use retention and NSI for ranking
    # Higher retention = better, lower |NSI| = better
    
    # Normalize both to 0-1 range
    ret_normalized = (metrics_df['retention_pct'] - metrics_df['retention_pct'].min()) / \
                     (metrics_df['retention_pct'].max() - metrics_df['retention_pct'].min())
    
    nsi_abs = metrics_df['nsi_r2'].abs()
    nsi_normalized = 1 - ((nsi_abs - nsi_abs.min()) / (nsi_abs.max() - nsi_abs.min()))
    
    # Combined score (equal weight)
    metrics_df['robustness_score'] = (ret_normalized + nsi_normalized) / 2
    
    return metrics_df


# ============================================================================
# ANOVA ANALYSIS
# ============================================================================

def perform_variance_decomposition(df, output_dir):
    """ANOVA variance decomposition"""
    print("\n" + "="*80)
    print("PERFORMING ANOVA VARIANCE DECOMPOSITION")
    print("="*80)
    
    df_03 = df[np.abs(df['sigma'] - 0.3) < 0.05].copy()
    
    if len(df_03) == 0:
        print("⚠️  No data at σ=0.3 for ANOVA")
        return
    
    df_03 = df_03[df_03['r2'] > -10].dropna(subset=['r2', 'model', 'representation'])
    df_03 = df_03[~df_03['representation'].isin(['random_smiles', 'randomized_smiles'])].copy()
    
    print(f"\nData for ANOVA:")
    print(f"  {len(df_03)} observations")
    print(f"  {df_03['model'].nunique()} models")
    print(f"  {df_03['representation'].nunique()} representations")
    
    grand_mean = df_03['r2'].mean()
    total_ss = ((df_03['r2'] - grand_mean) ** 2).sum()
    
    model_means = df_03.groupby('model')['r2'].mean()
    model_counts = df_03.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)
    
    rep_means = df_03.groupby('representation')['r2'].mean()
    rep_counts = df_03.groupby('representation').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)
    
    interaction_means = df_03.groupby(['model', 'representation'])['r2'].mean()
    interaction_counts = df_03.groupby(['model', 'representation']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        cell_mean = interaction_means[(model, rep)]
        expected = model_means[model] + rep_means[rep] - grand_mean
        ss_interaction += count * (cell_mean - expected) ** 2
    
    ss_residual = total_ss - ss_model - ss_rep - ss_interaction
    
    n = len(df_03)
    df_model = df_03['model'].nunique() - 1
    df_rep = df_03['representation'].nunique() - 1
    df_interaction = df_model * df_rep
    df_residual = n - (df_model + 1) * (df_rep + 1)
    
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_rep = ss_rep / df_rep if df_rep > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    
    f_model = ms_model / ms_residual if ms_residual > 0 else np.nan
    f_rep = ms_rep / ms_residual if ms_residual > 0 else np.nan
    f_interaction = ms_interaction / ms_residual if ms_residual > 0 else np.nan
    
    p_model = 1 - stats.f.cdf(f_model, df_model, df_residual) if not np.isnan(f_model) else np.nan
    p_rep = 1 - stats.f.cdf(f_rep, df_rep, df_residual) if not np.isnan(f_rep) else np.nan
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual) if not np.isnan(f_interaction) else np.nan
    
    eta2_model = ss_model / total_ss * 100
    eta2_rep = ss_rep / total_ss * 100
    eta2_interaction = ss_interaction / total_ss * 100
    eta2_residual = ss_residual / total_ss * 100
    
    print("\n" + "="*80)
    print("ANOVA TABLE (R² at σ=0.3)")
    print("="*80)
    print(f"{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}")
    print("-"*80)
    print(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}")
    print(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}")
    print(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}")
    print(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}")
    print("-"*80)
    print(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}")
    print("="*80)
    
    output_path = Path(output_dir) / "anova_variance_decomposition.txt"
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANOVA VARIANCE DECOMPOSITION\n")
        f.write("Dependent Variable: R² at σ=0.3\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {len(df_03)} observations, {df_03['model'].nunique()} models, {df_03['representation'].nunique()} representations\n\n")
        f.write("ANOVA TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}\n")
        f.write(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}\n")
        f.write(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}\n")
        f.write(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}\n")
        f.write("="*80 + "\n\n")
        f.write("INTERPRETATION:\n")
        f.write(f"  - Model architecture explains {eta2_model:.1f}% of variance\n")
        f.write(f"  - Representation explains {eta2_rep:.1f}% of variance\n")
        f.write(f"  - Interaction explains {eta2_interaction:.1f}% of variance\n")
        f.write(f"  - Residual (unexplained) is {eta2_residual:.1f}% of variance\n")
    
    print(f"\n✓ Saved ANOVA results to {output_path}")
    
    return {
        'eta2_model': eta2_model,
        'eta2_representation': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual
    }


def perform_noise_robustness_anova(metrics_df, output_dir):
    """ANOVA for noise robustness metrics"""
    print("\n" + "="*80)
    print("ANOVA VARIANCE DECOMPOSITION - NOISE ROBUSTNESS")
    print("="*80)
    
    analysis_df = metrics_df.dropna(subset=['nsi_r2', 'retention_pct', 'model', 'representation']).copy()
    analysis_df = analysis_df[~analysis_df['representation'].isin(['random_smiles', 'randomized_smiles'])].copy()
    analysis_df['abs_nsi'] = analysis_df['nsi_r2'].abs()
    
    print(f"\nData for ANOVA:")
    print(f"  {len(analysis_df)} observations")
    print(f"  {analysis_df['model'].nunique()} models")
    print(f"  {analysis_df['representation'].nunique()} representations")
    
    dependent_vars = {
        'NSI (R²)': 'nsi_r2',
        'Retention (%)': 'retention_pct',
        '|NSI|': 'abs_nsi'
    }
    
    all_results = {}
    output_text = []
    output_text.append("="*80)
    output_text.append("ANOVA VARIANCE DECOMPOSITION - NOISE ROBUSTNESS METRICS")
    output_text.append("="*80)
    output_text.append("")
    
    for metric_name, metric_col in dependent_vars.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING: {metric_name}")
        print(f"{'='*80}")
        
        output_text.append(f"\n{'='*80}")
        output_text.append(f"METRIC: {metric_name}")
        output_text.append(f"{'='*80}")
        
        y_values = analysis_df[metric_col]
        
        grand_mean = y_values.mean()
        total_ss = ((y_values - grand_mean) ** 2).sum()
        
        model_means = analysis_df.groupby('model')[metric_col].mean()
        model_counts = analysis_df.groupby('model').size()
        ss_model = sum(model_counts * (model_means - grand_mean) ** 2)
        
        rep_means = analysis_df.groupby('representation')[metric_col].mean()
        rep_counts = analysis_df.groupby('representation').size()
        ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)
        
        interaction_means = analysis_df.groupby(['model', 'representation'])[metric_col].mean()
        interaction_counts = analysis_df.groupby(['model', 'representation']).size()
        ss_interaction = 0
        for (model, rep), count in interaction_counts.items():
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2
        
        ss_residual = total_ss - ss_model - ss_rep - ss_interaction
        
        n = len(analysis_df)
        df_model = analysis_df['model'].nunique() - 1
        df_rep = analysis_df['representation'].nunique() - 1
        df_interaction = df_model * df_rep
        df_residual = n - (df_model + 1) * (df_rep + 1)
        
        ms_model = ss_model / df_model if df_model > 0 else 0
        ms_rep = ss_rep / df_rep if df_rep > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_residual = ss_residual / df_residual if df_residual > 0 else 0
        
        f_model = ms_model / ms_residual if ms_residual > 0 else np.nan
        f_rep = ms_rep / ms_residual if ms_residual > 0 else np.nan
        f_interaction = ms_interaction / ms_residual if ms_residual > 0 else np.nan
        
        p_model = 1 - stats.f.cdf(f_model, df_model, df_residual) if not np.isnan(f_model) else np.nan
        p_rep = 1 - stats.f.cdf(f_rep, df_rep, df_residual) if not np.isnan(f_rep) else np.nan
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual) if not np.isnan(f_interaction) else np.nan
        
        eta2_model = (ss_model / total_ss * 100) if total_ss > 0 else 0
        eta2_rep = (ss_rep / total_ss * 100) if total_ss > 0 else 0
        eta2_interaction = (ss_interaction / total_ss * 100) if total_ss > 0 else 0
        eta2_residual = (ss_residual / total_ss * 100) if total_ss > 0 else 0
        
        all_results[metric_name] = {
            'eta2_model': eta2_model,
            'eta2_representation': eta2_rep,
            'eta2_interaction': eta2_interaction,
            'eta2_residual': eta2_residual,
            'p_model': p_model,
            'p_rep': p_rep,
            'p_interaction': p_interaction
        }
        
        print(f"\n{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}")
        print("-"*80)
        print(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}")
        print(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}")
        print(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}")
        print(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}")
        print("-"*80)
        print(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}")
        
        output_text.append(f"\n{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}")
        output_text.append("-"*80)
        output_text.append(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}")
        output_text.append(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}")
        output_text.append(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}")
        output_text.append(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}")
        output_text.append("-"*80)
        output_text.append(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}")
        
        print(f"\nINTERPRETATION:")
        print(f"  - Model architecture explains {eta2_model:.1f}% of variance in {metric_name}")
        print(f"  - Representation explains {eta2_rep:.1f}% of variance in {metric_name}")
        print(f"  - Interaction explains {eta2_interaction:.1f}% of variance in {metric_name}")
        print(f"  - Residual (unexplained) is {eta2_residual:.1f}% of variance")
        
        output_text.append(f"\nINTERPRETATION:")
        output_text.append(f"  - Model architecture explains {eta2_model:.1f}% of variance in {metric_name}")
        output_text.append(f"  - Representation explains {eta2_rep:.1f}% of variance in {metric_name}")
        output_text.append(f"  - Interaction explains {eta2_interaction:.1f}% of variance in {metric_name}")
        output_text.append(f"  - Residual (unexplained) is {eta2_residual:.1f}% of variance")
        output_text.append("")
    
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON ACROSS METRICS")
    print(f"{'='*80}")
    print(f"\n{'Metric':<20} {'Model η²':>12} {'Rep η²':>12} {'Interaction η²':>12} {'Residual η²':>12}")
    print("-"*80)
    
    output_text.append(f"\n{'='*80}")
    output_text.append("SUMMARY COMPARISON ACROSS METRICS")
    output_text.append(f"{'='*80}")
    output_text.append(f"\n{'Metric':<20} {'Model η²':>12} {'Rep η²':>12} {'Interaction η²':>12} {'Residual η²':>12}")
    output_text.append("-"*80)
    
    for metric_name, results in all_results.items():
        print(f"{metric_name:<20} {results['eta2_model']:>11.1f}% {results['eta2_representation']:>11.1f}% {results['eta2_interaction']:>11.1f}% {results['eta2_residual']:>11.1f}%")
        output_text.append(f"{metric_name:<20} {results['eta2_model']:>11.1f}% {results['eta2_representation']:>11.1f}% {results['eta2_interaction']:>11.1f}% {results['eta2_residual']:>11.1f}%")
    
    print("="*80)
    output_text.append("="*80)
    
    print("\nKEY INSIGHTS:")
    output_text.append("\nKEY INSIGHTS:")
    
    for metric_name, results in all_results.items():
        factors = {
            'Model': results['eta2_model'],
            'Representation': results['eta2_representation'],
            'Interaction': results['eta2_interaction']
        }
        dominant = max(factors.items(), key=lambda x: x[1])
        
        insight = f"  - {metric_name}: {dominant[0]} dominates ({dominant[1]:.1f}% variance explained)"
        print(insight)
        output_text.append(insight)
    
    output_path = Path(output_dir) / "anova_noise_robustness_decomposition.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_text))
    
    print(f"\n✓ Saved noise robustness ANOVA results to {output_path}")
    
    return all_results


# ============================================================================
# FIGURE 1: GLOBAL NOISE ROBUSTNESS LANDSCAPE
# ============================================================================

def create_figure1_global_landscape(df, metrics_df, output_dir):
    """Figure 1: Global noise robustness landscape"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: GLOBAL ROBUSTNESS LANDSCAPE")
    print("="*80)
    
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.35, wspace=0.20,
                          left=0.10, right=0.95, top=0.96, bottom=0.04)
    
    # Panel A: Top 20
    ax_a = fig.add_subplot(gs[0, 0])
    top_20 = metrics_df.nlargest(20, 'robustness_score').sort_values('robustness_score')
    y_pos = np.arange(len(top_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in top_20['representation']]
    
    ax_a.barh(y_pos, top_20['robustness_score'], 
             color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(top_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, R²_h={row['r2_high']:.2f}"
        ax_a.text(row['robustness_score'] + 0.01, i, text, 
                 va='center', fontsize=7, color='black')
    
    ax_a.set_yticks(y_pos)
    labels = [f"{format_model(row['model'])}/{format_representation(row['representation'])}" 
              for _, row in top_20.iterrows()]
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.set_xlabel('Robustness Score', fontsize=10)
    ax_a.set_title('A. Top 20 Most Noise-Robust Configurations', 
                   fontsize=11, fontweight='bold', pad=12)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_xlim(0, max(top_20['robustness_score']) * 1.25)
    ax_a.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: Bottom 20
    ax_b = fig.add_subplot(gs[1, 0])
    bottom_20 = metrics_df.nsmallest(20, 'robustness_score').sort_values('robustness_score', ascending=False)
    y_pos = np.arange(len(bottom_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in bottom_20['representation']]
    
    ax_b.barh(y_pos, bottom_20['robustness_score'], 
             color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(bottom_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, R²_h={row['r2_high']:.2f}"
        ax_b.text(row['robustness_score'] + 0.01, i, text, 
                 va='center', fontsize=7, color='black')
    
    ax_b.set_yticks(y_pos)
    labels = [f"{format_model(row['model'])}/{format_representation(row['representation'])}" 
              for _, row in bottom_20.iterrows()]
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.set_xlabel('Robustness Score', fontsize=10)
    ax_b.set_title('B. Bottom 20 Least Robust Configurations', 
                   fontsize=11, fontweight='bold', pad=12)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.set_xlim(0, max(bottom_20['robustness_score']) * 1.25)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: Heatmap
    ax_c = fig.add_subplot(gs[2, 0])
    heatmap_data = metrics_df.pivot_table(
        index='model',
        columns='representation',
        values='retention_pct',
        aggfunc='mean'
    )
    heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]
    
    formatted_models = [format_model(m) for m in heatmap_data.index]
    formatted_reps = [format_representation(r) for r in heatmap_data.columns]
    
    im = ax_c.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn', 
                     vmin=0, vmax=100, interpolation='nearest')
    
    ax_c.set_xticks(np.arange(len(heatmap_data.columns)))
    ax_c.set_yticks(np.arange(len(heatmap_data.index)))
    ax_c.set_xticklabels(formatted_reps, rotation=45, ha='right', fontsize=9)
    ax_c.set_yticklabels(formatted_models, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.04)
    cbar.set_label('Retention at high noise (%)', fontsize=9, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=8)
    
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < 50 else 'black'
                ax_c.text(j, i, f'{value:.0f}', ha='center', va='center',
                         color=text_color, fontsize=7, fontweight='bold')
    
    ax_c.set_title('C. Global Heatmap: Retention % at High Noise', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.set_xlabel('Representation', fontsize=9)
    ax_c.set_ylabel('Model', fontsize=9)
    
    output_path = Path(output_dir) / "figure1_global_robustness_landscape.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1 to {output_path}")
    plt.close()


def create_supplementary_degradation_curves(df, metrics_df, output_dir):
    """Supplementary: R² degradation curves"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY: DEGRADATION CURVES")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    top_5 = metrics_df.nlargest(5, 'robustness_score')
    bottom_5 = metrics_df.nsmallest(5, 'robustness_score')
    
    for idx, (_, row) in enumerate(top_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                 marker='o', markersize=4, linewidth=2, alpha=0.9,
                 label=f"{model}/{rep} (top #{idx+1})", color=color, linestyle='-')
    
    for idx, (_, row) in enumerate(bottom_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                 marker='s', markersize=4, linewidth=2, alpha=0.8,
                 label=f"{model}/{rep} (bottom #{idx+1})", color=color, linestyle='--')
    
    ax.set_xlabel('Noise level (σ)', fontsize=9)
    ax.set_ylabel('R² score', fontsize=9)
    ax.set_title('R² Degradation Curves: Top 5 vs Bottom 5 Configurations', 
                   fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=7, loc='best', ncol=2, frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "supplementary_degradation_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved degradation curves to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: REPRESENTATION EFFECTS
# ============================================================================

def create_figure2_representation_effects(df, metrics_df, output_dir):
    """Figure 2: Representation effects"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 2: REPRESENTATION EFFECTS")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.98, top=0.96, bottom=0.06)
    
    # Panel A: Performance at σ=0.3
    ax_a = fig.add_subplot(gs[0, 0])
    r2_at_03 = []
    for (model, rep), group in df.groupby(['model', 'representation']):
        sigma_03 = group[np.abs(group['sigma'] - 0.3) < 0.05]
        if len(sigma_03) > 0:
            r2_at_03.append({
                'model': model,
                'representation': rep,
                'r2_at_03': sigma_03['r2'].values[0]
            })
    r2_at_03_df = pd.DataFrame(r2_at_03)
    
    rep_order = r2_at_03_df.groupby('representation')['r2_at_03'].median().sort_values(ascending=False).index
    valid_reps = []
    valid_data = []
    for rep in rep_order:
        data = r2_at_03_df[r2_at_03_df['representation'] == rep]['r2_at_03'].dropna()
        if len(data) >= 2:
            valid_reps.append(rep)
            valid_data.append(data)
    
    if len(valid_data) > 0:
        parts = ax_a.violinplot(valid_data, positions=range(len(valid_reps)), widths=0.7,
                                showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            rep = valid_reps[i]
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        
        for i, rep in enumerate(valid_reps):
            rep_data = r2_at_03_df[r2_at_03_df['representation'] == rep]['r2_at_03'].dropna()
            y = rep_data.values
            x = np.random.normal(i, 0.04, size=len(y))
            ax_a.scatter(x, y, alpha=0.4, s=8, color='black', zorder=3)
        
        ax_a.set_xticks(range(len(valid_reps)))
        formatted_reps = [format_representation(r) for r in valid_reps]
        ax_a.set_xticklabels(formatted_reps, rotation=45, ha='right')
        ax_a.set_ylabel('R² at σ=0.3', fontsize=9)
        ax_a.set_title('A. Performance at Moderate Noise', fontsize=10, fontweight='bold', pad=10)
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)
        ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax_a.set_ylim(bottom=0)
    
    # Panel B: Retention by representation
    ax_b = fig.add_subplot(gs[0, 1])
    rep_order_robust = metrics_df.groupby('representation')['retention_pct'].median().sort_values(ascending=False).index
    valid_reps_robust = []
    valid_data_robust = []
    for rep in rep_order_robust:
        data = metrics_df[metrics_df['representation'] == rep]['retention_pct'].dropna()
        if len(data) >= 2:
            valid_reps_robust.append(rep)
            valid_data_robust.append(data)
    
    if len(valid_data_robust) > 0:
        parts = ax_b.violinplot(valid_data_robust, positions=range(len(valid_reps_robust)),
                                widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            rep = valid_reps_robust[i]
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        
        for i, rep in enumerate(valid_reps_robust):
            rep_data = metrics_df[metrics_df['representation'] == rep]['retention_pct'].dropna()
            y = rep_data.values
            x = np.random.normal(i, 0.04, size=len(y))
            ax_b.scatter(x, y, alpha=0.4, s=8, color='black', zorder=3)
        
        ax_b.set_xticks(range(len(valid_reps_robust)))
        formatted_reps = [format_representation(r) for r in valid_reps_robust]
        ax_b.set_xticklabels(formatted_reps, rotation=45, ha='right')
        ax_b.set_ylabel('Retention at high noise (%)', fontsize=9)
        ax_b.set_title('B. Noise Robustness by Representation', fontsize=10, fontweight='bold', pad=10)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax_b.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_b.set_ylim(bottom=0)
    
    # Panel C: Noise sensitivity
    ax_c = fig.add_subplot(gs[0, 2])
    noise_difficulty = metrics_df.groupby('representation').agg({
        'nsi_r2': lambda x: np.abs(x).mean(),
        'retention_pct': 'mean'
    }).reset_index()
    noise_difficulty['difficulty'] = noise_difficulty['nsi_r2'].abs()
    noise_difficulty = noise_difficulty.sort_values('difficulty', ascending=False)
    
    colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in noise_difficulty['representation']]
    ax_c.bar(range(len(noise_difficulty)), noise_difficulty['difficulty'],
            color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax_c.set_xticks(range(len(noise_difficulty)))
    formatted_reps = [format_representation(r) for r in noise_difficulty['representation']]
    ax_c.set_xticklabels(formatted_reps, rotation=45, ha='right')
    ax_c.set_ylabel('Average |NSI|', fontsize=9)
    ax_c.set_title('C. Representation Sensitivity to Noise', fontsize=10, fontweight='bold', pad=10)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel D: Degradation across noise
    ax_d = fig.add_subplot(gs[1, 0])
    
    print(f"\n{'='*80}")
    print("PANEL D DEBUG - GRAPH REPRESENTATION")
    print(f"{'='*80}")
    
    # Check raw graph data
    graph_data_raw = df[df['representation'] == 'graph'].copy()
    print(f"\nRaw graph data: {len(graph_data_raw)} rows")
    print(f"Unique models in graph: {sorted(graph_data_raw['model'].unique())}")
    print(f"Unique sigmas in graph: {sorted(graph_data_raw['sigma'].unique())}")
    
    # HARDCODED EXCLUSION: Remove full Bayesian GNNs AND lowercase gcn
    exclude_models = ['gcn_bnn_full', 'mpnn_bnn_full', 'gin_bnn_full', 'gat_bnn_full', 'gcn']
    df_panel_d = df[~((df['model'].isin(exclude_models)) & (df['representation'] == 'graph'))].copy()
    
    graph_data_filtered = df_panel_d[df_panel_d['representation'] == 'graph'].copy()
    print(f"\nAfter filtering {exclude_models}:")
    print(f"  Filtered graph data: {len(graph_data_filtered)} rows")
    print(f"  Remaining models: {sorted(graph_data_filtered['model'].unique())}")
    
    representations = sorted(df_panel_d['representation'].unique())
    for rep in representations:
        rep_data = df_panel_d[df_panel_d['representation'] == rep]
        if len(rep_data) == 0:
            continue
        
        # CRITICAL FIX: Round sigma to avoid floating point duplicates
        rep_data = rep_data.copy()
        rep_data['sigma_rounded'] = rep_data['sigma'].round(2)
        avg_by_sigma = rep_data.groupby('sigma_rounded')['r2'].median().reset_index()
        avg_by_sigma.rename(columns={'sigma_rounded': 'sigma'}, inplace=True)
        
        if rep == 'graph':
            print(f"\nGraph median by sigma (after rounding):")
            print(avg_by_sigma)
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax_d.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                 marker='o', markersize=4, linewidth=2, alpha=0.8,
                 label=format_representation(rep), color=color)
    
    print(f"{'='*80}\n")
    
    ax_d.set_xlabel('Noise level (σ)', fontsize=9)
    ax_d.set_ylabel('Median R²', fontsize=9)
    ax_d.set_title('D. Performance Across Noise Levels\n(Median, full Bayesian GNNs excluded)', 
                   fontsize=9, fontweight='bold', pad=10)
    ax_d.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9, ncol=2)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel E: SNS top performers
    ax_e = fig.add_subplot(gs[1, 1])
    target_models = ['xgboost', 'ngboost', 'rf', 'gauche']
    for model in target_models:
        model_data = df[(df['model'] == model) & (df['representation'] == 'sns')]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#999999')
            ax_e.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', markersize=5, linewidth=2.5, alpha=0.9,
                     label=format_model(model), color=color)
    
    ax_e.set_xlabel('Noise level (σ)', fontsize=9)
    ax_e.set_ylabel('R²', fontsize=9)
    ax_e.set_title('E. SNS: Top Performers', fontsize=10, fontweight='bold', pad=10)
    ax_e.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_e.set_ylim(bottom=0)
    
    # Panel F: PDV top performers
    ax_f = fig.add_subplot(gs[1, 2])
    for model in target_models:
        model_data = df[(df['model'] == model) & (df['representation'] == 'pdv')]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#999999')
            ax_f.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', markersize=5, linewidth=2.5, alpha=0.9,
                     label=format_model(model), color=color)
    
    ax_f.set_xlabel('Noise level (σ)', fontsize=9)
    ax_f.set_ylabel('R²', fontsize=9)
    ax_f.set_title('F. PDV: Top Performers', fontsize=10, fontweight='bold', pad=10)
    ax_f.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_f.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "figure2_representation_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================

def create_supplementary_s1(metrics_df, output_dir):
    """Supplementary S1: Baseline vs retention scatter"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S1")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    representations = metrics_df['representation'].unique()
    for rep in representations:
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['retention_pct'],
                  alpha=0.7, s=60, color=color, label=rep.upper(),
                  edgecolors='black', linewidth=0.5)
    
    median_baseline = metrics_df['baseline_r2'].median()
    median_retention = metrics_df['retention_pct'].median()
    ax.axvline(median_baseline, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(median_retention, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Baseline R² (σ=0)', fontsize=10)
    ax.set_ylabel('Retention at high noise (%)', fontsize=10)
    ax.set_title('Supplementary S1: Baseline vs Retention', fontsize=11, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2, frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "supplementary_s1_baseline_vs_retention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S1 to {output_path}")
    plt.close()


def create_supplementary_s2(metrics_df, output_dir):
    """Supplementary S2: NSI distributions"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S2")
    print("="*80)
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    
    # Panel A: NSI histogram
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(metrics_df['nsi_r2'].dropna(), bins=30, 
             color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_a.axvline(metrics_df['nsi_r2'].median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median = {metrics_df["nsi_r2"].median():.4f}')
    ax_a.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_a.set_xlabel('NSI (R²)', fontsize=9)
    ax_a.set_ylabel('Frequency', fontsize=9)
    ax_a.set_title('A. Distribution of NSI', fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: NSI by representation
    ax_b = fig.add_subplot(gs[0, 1])
    rep_order = metrics_df.groupby('representation')['nsi_r2'].apply(lambda x: np.abs(x).median()).sort_values().index
    parts = ax_b.violinplot(
        [metrics_df[metrics_df['representation'] == rep]['nsi_r2'].dropna() for rep in rep_order],
        positions=range(len(rep_order)), widths=0.7, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        rep = rep_order[i]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    ax_b.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.set_xticks(range(len(rep_order)))
    ax_b.set_xticklabels(rep_order, rotation=45, ha='right')
    ax_b.set_ylabel('NSI (R²)', fontsize=9)
    ax_b.set_title('B. NSI by Representation', fontsize=10, fontweight='bold')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: RMSE NSI
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.hist(metrics_df['nsi_rmse'].dropna(), bins=30, 
             color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_c.axvline(metrics_df['nsi_rmse'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f'Median = {metrics_df["nsi_rmse"].median():.4f}')
    ax_c.set_xlabel('NSI (RMSE)', fontsize=9)
    ax_c.set_ylabel('Frequency', fontsize=9)
    ax_c.set_title('C. Distribution of NSI (RMSE)', fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel D: NSI by model
    ax_d = fig.add_subplot(gs[1, 1])
    model_order = metrics_df.groupby('model')['nsi_r2'].apply(lambda x: np.abs(x).median()).sort_values().index[:10]
    parts = ax_d.violinplot(
        [metrics_df[metrics_df['model'] == model]['nsi_r2'].dropna() for model in model_order],
        positions=range(len(model_order)), widths=0.7, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        model = model_order[i]
        color = MODEL_COLORS.get(model, '#999999')
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    
    ax_d.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_xticks(range(len(model_order)))
    ax_d.set_xticklabels(model_order, rotation=45, ha='right', fontsize=7)
    ax_d.set_ylabel('NSI (R²)', fontsize=9)
    ax_d.set_title('D. NSI by Model (Top 10)', fontsize=10, fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "supplementary_s2_nsi_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S2 to {output_path}")
    plt.close()


def create_supplementary_s3(df, metrics_df, output_dir):
    """Supplementary S3: Per-target breakdown (if applicable)"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S3")
    print("="*80)
    
    if 'target' not in df.columns or df['target'].nunique() <= 1:
        print("⚠️  Only one target - skipping S3")
        return
    
    targets = df['target'].unique()
    print(f"Found {len(targets)} targets")
    
    n_targets = len(targets)
    ncols = min(3, n_targets)
    nrows = (n_targets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        target_metrics = metrics_df[metrics_df.get('target') == target]
        
        if len(target_metrics) == 0:
            ax.text(0.5, 0.5, f'No data for {target}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        heatmap_data = target_metrics.pivot_table(
            index='model', columns='representation',
            values='retention_pct', aggfunc='mean')
        
        im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn', 
                      vmin=0, vmax=100, interpolation='nearest')
        
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(heatmap_data.index, fontsize=7)
        ax.set_title(f'{target}', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for idx in range(len(targets), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "supplementary_s3_per_target_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S3 to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Top 20
    table1 = metrics_df.nlargest(20, 'robustness_score')[
        ['model', 'representation', 'baseline_r2', 'r2_high', 
         'retention_pct', 'nsi_r2', 'robustness_score']
    ].copy()
    table1.columns = ['Model', 'Representation', 'Baseline R²', 'R² (high noise)',
                     'Retention (%)', 'NSI (R²)', 'Robustness Score']
    table1.to_csv(output_dir / "table1_top20_robust_configs.csv", index=False, float_format='%.4f')
    with open(output_dir / "table1_top20_robust_configs.tex", 'w') as f:
        f.write(table1.to_latex(index=False, float_format="%.4f"))
    print(f"✓ Saved Table 1")
    
    # Table 2: By representation
    table2 = metrics_df.groupby('representation').agg({
        'baseline_r2': ['mean', 'std', 'median'],
        'retention_pct': ['mean', 'std', 'median'],
        'nsi_r2': lambda x: np.abs(x).mean(),
        'robustness_score': ['mean', 'std']
    }).round(4)
    table2.to_csv(output_dir / "table2_performance_by_representation.csv")
    with open(output_dir / "table2_performance_by_representation.tex", 'w') as f:
        f.write(table2.to_latex(float_format="%.4f"))
    print(f"✓ Saved Table 2")
    
    # Table 3: By model
    table3 = metrics_df.groupby('model').agg({
        'baseline_r2': ['mean', 'std', 'median'],
        'retention_pct': ['mean', 'std', 'median'],
        'nsi_r2': lambda x: np.abs(x).mean(),
        'robustness_score': ['mean', 'std']
    }).round(4)
    table3 = table3.sort_values(('robustness_score', 'mean'), ascending=False).head(20)
    table3.to_csv(output_dir / "table3_performance_by_model.csv")
    with open(output_dir / "table3_performance_by_model.tex", 'w') as f:
        f.write(table3.to_latex(float_format="%.4f"))
    print(f"✓ Saved Table 3")


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def perform_statistical_tests(df, metrics_df, output_dir):
    """Perform statistical tests"""
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL TESTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    results_text = []
    results_text.append("STATISTICAL COMPARISONS - PHASE 0")
    results_text.append("="*80)
    results_text.append("")
    
    # Test 1: Representation comparisons
    results_text.append("TEST 1: Representation Comparisons")
    results_text.append("-"*80)
    
    reps = metrics_df['representation'].value_counts().head(5).index
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            rep1, rep2 = reps[i], reps[j]
            data1 = metrics_df[metrics_df['representation'] == rep1]['retention_pct'].dropna()
            data2 = metrics_df[metrics_df['representation'] == rep2]['retention_pct'].dropna()
            
            if len(data1) >= 3 and len(data2) >= 3:
                from scipy.stats import mannwhitneyu
                stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                mean1, mean2 = data1.mean(), data2.mean()
                
                results_text.append(f"{rep1.upper()} vs {rep2.upper()}:")
                results_text.append(f"  Mean retention: {mean1:.2f}% vs {mean2:.2f}%")
                results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.4f}")
                
                if p_val < 0.05:
                    winner = rep1 if mean1 > mean2 else rep2
                    results_text.append(f"  → Significant difference, {winner.upper()} superior")
                else:
                    results_text.append(f"  → No significant difference")
                results_text.append("")
    
    # Test 2: Model comparisons
    results_text.append("\nTEST 2: Model Family Comparisons")
    results_text.append("-"*80)
    
    models = metrics_df['model'].value_counts().head(5).index
    for i in range(min(3, len(models))):
        for j in range(i+1, min(3, len(models))):
            model1, model2 = models[i], models[j]
            data1 = metrics_df[metrics_df['model'] == model1]['retention_pct'].dropna()
            data2 = metrics_df[metrics_df['model'] == model2]['retention_pct'].dropna()
            
            if len(data1) >= 3 and len(data2) >= 3:
                from scipy.stats import mannwhitneyu
                stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                mean1, mean2 = data1.mean(), data2.mean()
                
                results_text.append(f"{model1} vs {model2}:")
                results_text.append(f"  Mean retention: {mean1:.2f}% vs {mean2:.2f}%")
                results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.4f}")
                
                if p_val < 0.05:
                    winner = model1 if mean1 > mean2 else model2
                    results_text.append(f"  → Significant difference, {winner} superior")
                else:
                    results_text.append(f"  → No significant difference")
                results_text.append("")
    
    output_path = output_dir / "statistical_tests_phase0.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    print(f"✓ Saved statistical tests to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(old_results_dir="../../qsar_qm_models/results", 
         new_results_dir="results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 0 ANALYSIS - GLOBAL SCREENING (COMBINED)")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    df = load_screening_results(old_results_dir, new_results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    
    # FORCE FILTER MLP/MTL
    print(f"\n{'='*80}")
    print("FORCING MLP/MTL FILTER ON METRICS")
    print(f"{'='*80}")
    print(f"Before filter: {len(metrics_df)} configs")
    print(f"Models present: {sorted(metrics_df['model'].unique())}")
    metrics_df = metrics_df[~metrics_df['model'].isin(mlp_mtl_models)].copy()
    print(f"After filter: {len(metrics_df)} configs")
    print(f"Models remaining: {sorted(metrics_df['model'].unique())}")
    print(f"{'='*80}\n")
    
    metrics_df = define_robustness_score(metrics_df)
    
    output_dir = Path(new_results_dir) / "phase0_figures_combined"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_df.to_csv(output_dir / "phase0_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase0_robustness_metrics.csv'}")
    
    perform_variance_decomposition(df, output_dir)
    perform_noise_robustness_anova(metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure1_global_landscape(df, metrics_df, output_dir)
    create_supplementary_degradation_curves(df, metrics_df, output_dir)
    create_figure2_representation_effects(df, metrics_df, output_dir)
    create_supplementary_s1(metrics_df, output_dir)
    create_supplementary_s2(metrics_df, output_dir)
    create_supplementary_s3(df, metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_tests(df, metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("PHASE 0 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    print("\nTop 10 Most Robust Configurations:")
    for idx, (_, row) in enumerate(metrics_df.nlargest(10, 'robustness_score').iterrows(), 1):
        print(f"  {idx}. {format_model(row['model'])}/{format_representation(row['representation'])}: "
              f"Score={row['robustness_score']:.3f}, "
              f"R²₀={row['baseline_r2']:.3f}, "
              f"Retention={row['retention_pct']:.1f}%")
    
    print("\nRepresentation Rankings (by median retention):")
    rep_ranking = metrics_df.groupby('representation')['retention_pct'].median().sort_values(ascending=False)
    for idx, (rep, val) in enumerate(rep_ranking.items(), 1):
        print(f"  {idx}. {format_representation(rep)}: {val:.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        old_dir = sys.argv[1]
        new_dir = sys.argv[2]
        main(old_dir, new_dir)
    elif len(sys.argv) > 1:
        main(new_results_dir=sys.argv[1])
    else:
        main()