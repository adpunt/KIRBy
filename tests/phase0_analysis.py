"""
Phase 0 Analysis - Global Screening of Model-Representation Pairs
IMPROVED VERSION - Fixes for visualization issues

Key improvements:
1. Fixed retention calculation - properly handles baseline R² ≈ 0 cases
2. Clean label formatting (GAT-BNN-Last not Gat_bnn_last)
3. Graph representation in separate column in heatmap
4. Consistent filtering of catastrophic failures throughout
5. Panel B with proper exclusion of outliers + inset
6. Panel C excludes/annotates misleading Graph NSI
7. S1 with zoomed inset for main cluster
8. Improved legend in degradation curves
9. Consistent filtering across ALL figures
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

# Catastrophically failing models (baseline R² < 0.1 or R² goes negative)
CATASTROPHIC_GRAPH_MODELS = ['gcn_bnn_full', 'mpnn_bnn_full', 'gin_bnn_full', 
                              'gat_bnn_full', 'gcn']

# Minimum baseline R² to be considered valid
MIN_BASELINE_R2 = 0.1

# ============================================================================
# IMPROVED FORMATTING HELPERS
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
    """Format model names for display - IMPROVED"""
    # First normalize to lowercase for matching
    model_lower = model.lower()
    
    # Direct mappings for simple models
    simple_map = {
        'rf': 'RF',
        'qrf': 'QRF',
        'xgboost': 'XGBoost',
        'ngboost': 'NGBoost',
        'dnn': 'DNN',
        'mlp': 'MLP',
        'gauche': 'GP',
        'svm': 'SVM',
    }
    
    if model_lower in simple_map:
        return simple_map[model_lower]
    
    # Handle GNN variants
    gnn_base = {'gcn': 'GCN', 'gin': 'GIN', 'gat': 'GAT', 'mpnn': 'MPNN'}
    
    for base, formatted in gnn_base.items():
        if model_lower.startswith(base):
            if 'bnn_full' in model_lower:
                return f'{formatted}-BNN-Full'
            elif 'bnn_last' in model_lower:
                return f'{formatted}-BNN-Last'
            elif 'bnn_variational' in model_lower or 'bnn_var' in model_lower:
                return f'{formatted}-BNN-Var'
            else:
                return formatted
    
    # Handle DNN BNN variants
    if 'dnn' in model_lower:
        if 'bnn_full' in model_lower:
            return 'DNN-BNN-Full'
        elif 'bnn_last' in model_lower:
            return 'DNN-BNN-Last'
        elif 'bnn_variational' in model_lower or 'bnn_var' in model_lower:
            return 'DNN-BNN-Var'
        return 'DNN'
    
    # Handle conformal variants
    if 'conformal' in model_lower:
        parts = model_lower.replace('conformal_', '').split('_')
        base_model = parts[0]
        formatted_base = simple_map.get(base_model, base_model.upper())
        return f'Conformal-{formatted_base}'
    
    # Fallback: capitalize
    return model.replace('_', '-').title()


def format_config_label(model, rep):
    """Format model/representation pair for display"""
    return f"{format_model(model)}/{format_representation(rep)}"


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
    bad_configs = baseline_check[baseline_check < MIN_BASELINE_R2].index
    
    if len(bad_configs) > 0:
        print(f"  Removing {len(bad_configs)} configs with baseline R² < {MIN_BASELINE_R2}")
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
# METRICS CALCULATION - IMPROVED
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.6):
    """
    Calculate robustness metrics for each model-representation pair
    
    IMPROVED: Properly handles edge cases where baseline R² ≈ 0
    
    Metrics calculated:
    - baseline_r2: R² at σ=0
    - r2_high: R² at σ=sigma_high
    - retention_pct: (r2_high / baseline_r2) * 100, bounded 0-100 for valid configs
    - nsi_r2: slope of R² vs σ (Noise Sensitivity Index)
    - is_valid: flag for whether config has meaningful baseline
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
        
        # IMPROVED: Check validity before calculating retention
        metrics['is_valid'] = (
            not np.isnan(metrics['baseline_r2']) and 
            metrics['baseline_r2'] >= MIN_BASELINE_R2
        )
        
        # IMPROVED: Retention percentage - only for valid configs
        if metrics['is_valid'] and not np.isnan(metrics['r2_high']):
            # Calculate raw retention
            raw_retention = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            # Bound to reasonable range (can exceed 100 slightly due to noise)
            metrics['retention_pct'] = np.clip(raw_retention, 0, 150)
            metrics['retention_pct_raw'] = raw_retention  # Keep raw for analysis
        else:
            metrics['retention_pct'] = np.nan
            metrics['retention_pct_raw'] = np.nan
        
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
                
                # IMPROVED: Relative NSI only for valid configs
                if metrics['is_valid'] and intercept_r2 > MIN_BASELINE_R2:
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
    print(f"Valid configurations (baseline R² >= {MIN_BASELINE_R2}): {metrics_df['is_valid'].sum()}")
    print(f"Invalid configurations: {(~metrics_df['is_valid']).sum()}")
    
    # List invalid configs
    invalid = metrics_df[~metrics_df['is_valid']]
    if len(invalid) > 0:
        print("\nInvalid configurations (catastrophic failure):")
        for _, row in invalid.iterrows():
            print(f"  {row['model']}/{row['representation']}: baseline R² = {row['baseline_r2']:.4f}")
    
    # Summary statistics for VALID configs only
    valid_df = metrics_df[metrics_df['is_valid']]
    print(f"\nBaseline R² statistics (valid only):")
    print(f"  Mean: {valid_df['baseline_r2'].mean():.4f}")
    print(f"  Median: {valid_df['baseline_r2'].median():.4f}")
    print(f"  Range: [{valid_df['baseline_r2'].min():.4f}, {valid_df['baseline_r2'].max():.4f}]")
    
    print(f"\nRetention at σ={sigma_high} statistics (valid only):")
    print(f"  Mean: {valid_df['retention_pct'].mean():.2f}%")
    print(f"  Median: {valid_df['retention_pct'].median():.2f}%")
    print(f"  Range: [{valid_df['retention_pct'].min():.2f}%, {valid_df['retention_pct'].max():.2f}%]")
    
    return metrics_df


def define_robustness_score(metrics_df):
    """
    Define composite robustness score based on:
    1. Low absolute NSI (slow degradation)
    2. High retention percentage
    3. Reasonable baseline (filtered already)
    
    IMPROVED: Only scores valid configurations
    """
    # Work only with valid configs for scoring
    valid_mask = metrics_df['is_valid'] & metrics_df['retention_pct'].notna()
    
    metrics_df['robustness_score'] = np.nan
    
    if valid_mask.sum() > 0:
        valid_retention = metrics_df.loc[valid_mask, 'retention_pct']
        valid_nsi = metrics_df.loc[valid_mask, 'nsi_r2'].abs()
        
        # Normalize both to 0-1 range
        ret_normalized = (valid_retention - valid_retention.min()) / \
                         (valid_retention.max() - valid_retention.min() + 1e-8)
        
        nsi_normalized = 1 - ((valid_nsi - valid_nsi.min()) / 
                              (valid_nsi.max() - valid_nsi.min() + 1e-8))
        
        # Combined score (equal weight)
        metrics_df.loc[valid_mask, 'robustness_score'] = (ret_normalized + nsi_normalized) / 2
    
    return metrics_df


# ============================================================================
# HELPER FUNCTIONS FOR FILTERING
# ============================================================================

def get_valid_metrics(metrics_df):
    """Return only valid (non-catastrophic) configurations"""
    return metrics_df[metrics_df['is_valid']].copy()


def get_valid_data(df, metrics_df):
    """Filter raw data to exclude catastrophic configurations"""
    valid_configs = metrics_df[metrics_df['is_valid']][['model', 'representation']]
    return df.merge(valid_configs, on=['model', 'representation'])


def filter_catastrophic_graph_models(df):
    """Remove known catastrophically failing graph models"""
    mask = ~((df['model'].isin(CATASTROPHIC_GRAPH_MODELS)) & 
             (df['representation'] == 'graph'))
    return df[mask].copy()


# ============================================================================
# ANOVA ANALYSIS
# ============================================================================

def perform_variance_decomposition(df, output_dir):
    """ANOVA variance decomposition - uses only valid configs"""
    print("\n" + "="*80)
    print("PERFORMING ANOVA VARIANCE DECOMPOSITION")
    print("="*80)
    
    # Filter to valid configurations
    df_filtered = filter_catastrophic_graph_models(df)
    df_03 = df_filtered[np.abs(df_filtered['sigma'] - 0.3) < 0.05].copy()
    
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
        f.write("Note: Catastrophically failing models excluded\n")
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


# ============================================================================
# FIGURE 1: GLOBAL NOISE ROBUSTNESS LANDSCAPE - IMPROVED
# ============================================================================

def create_figure1_global_landscape(df, metrics_df, output_dir):
    """Figure 1: Global noise robustness landscape - IMPROVED"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: GLOBAL ROBUSTNESS LANDSCAPE")
    print("="*80)
    
    # Use only valid configurations
    valid_metrics = get_valid_metrics(metrics_df)
    
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.35, wspace=0.20,
                          left=0.12, right=0.95, top=0.96, bottom=0.04)
    
    # Panel A: Top 20
    ax_a = fig.add_subplot(gs[0, 0])
    top_20 = valid_metrics.nlargest(20, 'robustness_score').sort_values('robustness_score')
    y_pos = np.arange(len(top_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in top_20['representation']]
    
    ax_a.barh(y_pos, top_20['robustness_score'], 
             color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(top_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, R²ₕ={row['r2_high']:.2f}"
        ax_a.text(row['robustness_score'] + 0.01, i, text, 
                 va='center', fontsize=7, color='black')
    
    ax_a.set_yticks(y_pos)
    labels = [format_config_label(row['model'], row['representation']) 
              for _, row in top_20.iterrows()]
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.set_xlabel('Robustness Score', fontsize=10)
    ax_a.set_title('A. Top 20 Most Noise-Robust Configurations', 
                   fontsize=11, fontweight='bold', pad=12)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_xlim(0, max(top_20['robustness_score']) * 1.25)
    ax_a.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: Bottom 20 (of VALID configs - not catastrophic failures)
    ax_b = fig.add_subplot(gs[1, 0])
    bottom_20 = valid_metrics.nsmallest(20, 'robustness_score').sort_values('robustness_score', ascending=False)
    y_pos = np.arange(len(bottom_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in bottom_20['representation']]
    
    ax_b.barh(y_pos, bottom_20['robustness_score'], 
             color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(bottom_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, R²ₕ={row['r2_high']:.2f}"
        ax_b.text(row['robustness_score'] + 0.01, i, text, 
                 va='center', fontsize=7, color='black')
    
    ax_b.set_yticks(y_pos)
    labels = [format_config_label(row['model'], row['representation']) 
              for _, row in bottom_20.iterrows()]
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.set_xlabel('Robustness Score', fontsize=10)
    ax_b.set_title('B. Bottom 20 Least Robust Configurations (Valid Only)', 
                   fontsize=11, fontweight='bold', pad=12)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.set_xlim(0, max(bottom_20['robustness_score']) * 1.25)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: Heatmap - IMPROVED ordering with Graph on right
    ax_c = fig.add_subplot(gs[2, 0])
    
    # Pivot with valid metrics only
    heatmap_data = valid_metrics.pivot_table(
        index='model',
        columns='representation',
        values='retention_pct',
        aggfunc='mean'
    )
    
    # Order representations: fingerprints first, then Graph on right
    rep_order = ['ecfp4', 'pdv', 'sns', 'smiles', 'randomized_smiles', 'graph']
    rep_order = [r for r in rep_order if r in heatmap_data.columns]
    heatmap_data = heatmap_data[rep_order]
    
    # Sort models by mean retention
    heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]
    
    # Format labels
    formatted_models = [format_model(m) for m in heatmap_data.index]
    formatted_reps = [format_representation(r) for r in heatmap_data.columns]
    
    # Create heatmap with FIXED colorscale 0-100
    im = ax_c.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn', 
                     vmin=0, vmax=100, interpolation='nearest')
    
    ax_c.set_xticks(np.arange(len(heatmap_data.columns)))
    ax_c.set_yticks(np.arange(len(heatmap_data.index)))
    ax_c.set_xticklabels(formatted_reps, rotation=45, ha='right', fontsize=9)
    ax_c.set_yticklabels(formatted_models, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.04)
    cbar.set_label('Retention at σ=0.6 (%)', fontsize=9, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=8)
    
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < 50 else 'black'
                ax_c.text(j, i, f'{value:.0f}', ha='center', va='center',
                         color=text_color, fontsize=7, fontweight='bold')
    
    ax_c.set_title('C. Retention (%) at High Noise — Valid Configurations Only', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.set_xlabel('Representation', fontsize=9)
    ax_c.set_ylabel('Model', fontsize=9)
    
    # Add vertical line before Graph column
    if 'graph' in rep_order:
        graph_idx = rep_order.index('graph')
        ax_c.axvline(x=graph_idx - 0.5, color='black', linewidth=2, linestyle='-')
    
    output_path = Path(output_dir) / "figure1_global_robustness_landscape.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY: DEGRADATION CURVES - IMPROVED
# ============================================================================

def create_supplementary_degradation_curves(df, metrics_df, output_dir):
    """Supplementary: R² degradation curves - IMPROVED"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY: DEGRADATION CURVES")
    print("="*80)
    
    valid_metrics = get_valid_metrics(metrics_df)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Top 5 from valid metrics
    top_5 = valid_metrics.nlargest(5, 'robustness_score')
    # Bottom 5 from valid metrics (not catastrophic failures)
    bottom_5 = valid_metrics.nsmallest(5, 'robustness_score')
    
    # Color palettes for clear distinction
    top_colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#ffffbf']
    bottom_colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf']
    
    # Plot top 5 with solid lines
    for idx, (_, row) in enumerate(top_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        label = f"{format_model(model)}/{format_representation(rep)}"
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                marker='o', markersize=5, linewidth=2.5, alpha=0.9,
                label=f"⬆ {label}", color=top_colors[idx], linestyle='-')
    
    # Plot bottom 5 with dashed lines
    for idx, (_, row) in enumerate(bottom_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        label = f"{format_model(model)}/{format_representation(rep)}"
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                marker='s', markersize=5, linewidth=2.5, alpha=0.9,
                label=f"⬇ {label}", color=bottom_colors[idx], linestyle='--')
    
    ax.set_xlabel('Noise level (σ)', fontsize=11)
    ax.set_ylabel('R² score', fontsize=11)
    ax.set_title('R² Degradation: Top 5 vs Bottom 5 Valid Configurations', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Improved legend - two columns, clear grouping
    ax.legend(fontsize=8, loc='upper right', ncol=2, frameon=True, 
              framealpha=0.95, edgecolor='gray',
              title='⬆ Most Robust    ⬇ Least Robust', title_fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.02, 1.02)
    
    output_path = Path(output_dir) / "supplementary_degradation_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved degradation curves to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: REPRESENTATION EFFECTS - IMPROVED
# ============================================================================

def create_figure2_representation_effects(df, metrics_df, output_dir):
    """Figure 2: Representation effects - IMPROVED"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 2: REPRESENTATION EFFECTS")
    print("="*80)
    
    # Use valid metrics throughout
    valid_metrics = get_valid_metrics(metrics_df)
    
    # Filter data for catastrophic graph failures
    df_filtered = filter_catastrophic_graph_models(df)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.98, top=0.94, bottom=0.08)
    
    # Panel A: Performance at σ=0.3
    ax_a = fig.add_subplot(gs[0, 0])
    r2_at_03 = []
    for (model, rep), group in df_filtered.groupby(['model', 'representation']):
        sigma_03 = group[np.abs(group['sigma'] - 0.3) < 0.05]
        if len(sigma_03) > 0:
            r2_at_03.append({
                'model': model,
                'representation': rep,
                'r2_at_03': sigma_03['r2'].values[0]
            })
    r2_at_03_df = pd.DataFrame(r2_at_03)
    
    # Order by median performance
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
    
    # Panel B: Retention by representation - IMPROVED (valid configs only)
    ax_b = fig.add_subplot(gs[0, 1])
    rep_order_robust = valid_metrics.groupby('representation')['retention_pct'].median().sort_values(ascending=False).index
    valid_reps_robust = []
    valid_data_robust = []
    for rep in rep_order_robust:
        data = valid_metrics[valid_metrics['representation'] == rep]['retention_pct'].dropna()
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
            rep_data = valid_metrics[valid_metrics['representation'] == rep]['retention_pct'].dropna()
            y = rep_data.values
            x = np.random.normal(i, 0.04, size=len(y))
            ax_b.scatter(x, y, alpha=0.4, s=8, color='black', zorder=3)
        
        ax_b.set_xticks(range(len(valid_reps_robust)))
        formatted_reps = [format_representation(r) for r in valid_reps_robust]
        ax_b.set_xticklabels(formatted_reps, rotation=45, ha='right')
        ax_b.set_ylabel('Retention at σ=0.6 (%)', fontsize=9)
        ax_b.set_title('B. Noise Robustness by Representation\n(Valid Configurations Only)', 
                       fontsize=10, fontweight='bold', pad=10)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
        ax_b.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_b.set_ylim(60, 105)  # Focused range for valid configs
    
    # Panel C: Noise sensitivity - IMPROVED (exclude Graph or annotate)
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Calculate NSI only for valid configs
    noise_difficulty = valid_metrics.groupby('representation').agg({
        'nsi_r2': lambda x: np.abs(x).mean(),
        'retention_pct': 'mean'
    }).reset_index()
    noise_difficulty['difficulty'] = noise_difficulty['nsi_r2'].abs()
    noise_difficulty = noise_difficulty.sort_values('difficulty', ascending=False)
    
    colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in noise_difficulty['representation']]
    bars = ax_c.bar(range(len(noise_difficulty)), noise_difficulty['difficulty'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # Annotate Graph bar if present
    for i, (_, row) in enumerate(noise_difficulty.iterrows()):
        if row['representation'] == 'graph':
            ax_c.annotate('*Well-tuned\nGNNs only', 
                         xy=(i, row['difficulty']), 
                         xytext=(i + 0.5, row['difficulty'] + 0.05),
                         fontsize=7, ha='left',
                         arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    ax_c.set_xticks(range(len(noise_difficulty)))
    formatted_reps = [format_representation(r) for r in noise_difficulty['representation']]
    ax_c.set_xticklabels(formatted_reps, rotation=45, ha='right')
    ax_c.set_ylabel('Average |NSI|', fontsize=9)
    ax_c.set_title('C. Representation Sensitivity to Noise\n(Valid Configurations Only)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel D: Degradation across noise - IMPROVED filtering
    ax_d = fig.add_subplot(gs[1, 0])
    
    # Use filtered data (catastrophic failures removed)
    representations = sorted(df_filtered['representation'].unique())
    
    for rep in representations:
        rep_data = df_filtered[df_filtered['representation'] == rep].copy()
        if len(rep_data) == 0:
            continue
        
        # Round sigma and compute median
        rep_data['sigma_rounded'] = rep_data['sigma'].round(2)
        avg_by_sigma = rep_data.groupby('sigma_rounded')['r2'].median().reset_index()
        avg_by_sigma.rename(columns={'sigma_rounded': 'sigma'}, inplace=True)
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax_d.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                 marker='o', markersize=4, linewidth=2, alpha=0.8,
                 label=format_representation(rep), color=color)
    
    ax_d.set_xlabel('Noise level (σ)', fontsize=9)
    ax_d.set_ylabel('Median R²', fontsize=9)
    ax_d.set_title('D. Performance Across Noise Levels\n(Catastrophic Failures Excluded)', 
                   fontsize=9, fontweight='bold', pad=10)
    ax_d.legend(fontsize=7, loc='lower left', frameon=True, framealpha=0.9, ncol=2)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_d.set_ylim(0, 0.95)
    
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
    ax_e.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
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
    ax_f.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_f.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "figure2_representation_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S1: BASELINE VS RETENTION - IMPROVED WITH INSET
# ============================================================================

def create_supplementary_s1(metrics_df, output_dir):
    """Supplementary S1: Baseline vs retention scatter - IMPROVED with inset"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S1")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ALL data first (including invalid)
    representations = metrics_df['representation'].unique()
    for rep in representations:
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['retention_pct'],
                  alpha=0.7, s=60, color=color, label=format_representation(rep).upper(),
                  edgecolors='black', linewidth=0.5)
    
    # Reference lines
    ax.axvline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Baseline R² (σ=0)', fontsize=10)
    ax.set_ylabel('Retention at σ=0.6 (%)', fontsize=10)
    ax.set_title('Supplementary S1: Baseline vs Retention\n(All Configurations)', 
                 fontsize=11, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='lower left', ncol=2, frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add zoomed inset for main cluster
    axins = inset_axes(ax, width="45%", height="45%", loc='upper right',
                       bbox_to_anchor=(0.02, 0.02, 0.96, 0.96),
                       bbox_transform=ax.transAxes)
    
    # Plot only valid configs in inset
    valid_metrics = get_valid_metrics(metrics_df)
    for rep in representations:
        rep_data = valid_metrics[valid_metrics['representation'] == rep]
        if len(rep_data) > 0:
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            axins.scatter(rep_data['baseline_r2'], rep_data['retention_pct'],
                         alpha=0.7, s=40, color=color,
                         edgecolors='black', linewidth=0.3)
    
    # Set inset limits to focus on main cluster
    axins.set_xlim(0.55, 0.92)
    axins.set_ylim(68, 102)
    axins.set_xlabel('Baseline R²', fontsize=7)
    axins.set_ylabel('Retention (%)', fontsize=7)
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.3, linestyle=':', linewidth=0.3)
    axins.set_title('Main Cluster (Valid Only)', fontsize=8, fontweight='bold')
    
    # Mark the inset region on main plot
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    # Annotate outliers
    invalid_metrics = metrics_df[~metrics_df['is_valid']]
    for _, row in invalid_metrics.iterrows():
        if row['retention_pct'] < -100:
            ax.annotate(f"{format_model(row['model'])}\n(catastrophic)", 
                       xy=(row['baseline_r2'], row['retention_pct']),
                       xytext=(0.15, row['retention_pct'] + 50),
                       fontsize=7, ha='center',
                       arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    output_path = Path(output_dir) / "supplementary_s1_baseline_vs_retention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S1 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S2: NSI DISTRIBUTIONS - IMPROVED
# ============================================================================

def create_supplementary_s2(metrics_df, output_dir):
    """Supplementary S2: NSI distributions - IMPROVED"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S2")
    print("="*80)
    
    # Use valid metrics only
    valid_metrics = get_valid_metrics(metrics_df)
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    
    # Panel A: NSI histogram (valid only)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(valid_metrics['nsi_r2'].dropna(), bins=25, 
             color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.8)
    median_val = valid_metrics['nsi_r2'].median()
    ax_a.axvline(median_val, color='red', linestyle='--', 
                linewidth=2, label=f'Median = {median_val:.4f}')
    ax_a.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_a.set_xlabel('NSI (R²)', fontsize=9)
    ax_a.set_ylabel('Frequency', fontsize=9)
    ax_a.set_title('A. Distribution of NSI\n(Valid Configurations)', fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: NSI by representation (valid only)
    ax_b = fig.add_subplot(gs[0, 1])
    rep_order = valid_metrics.groupby('representation')['nsi_r2'].apply(
        lambda x: np.abs(x).median()
    ).sort_values().index
    
    valid_rep_data = []
    valid_rep_names = []
    for rep in rep_order:
        data = valid_metrics[valid_metrics['representation'] == rep]['nsi_r2'].dropna()
        if len(data) >= 2:
            valid_rep_data.append(data)
            valid_rep_names.append(rep)
    
    if len(valid_rep_data) > 0:
        parts = ax_b.violinplot(valid_rep_data, positions=range(len(valid_rep_names)), 
                               widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            rep = valid_rep_names[i]
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
    
    ax_b.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.set_xticks(range(len(valid_rep_names)))
    ax_b.set_xticklabels([format_representation(r) for r in valid_rep_names], 
                         rotation=45, ha='right')
    ax_b.set_ylabel('NSI (R²)', fontsize=9)
    ax_b.set_title('B. NSI by Representation\n(Valid Configurations)', fontsize=10, fontweight='bold')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: RMSE NSI (valid only)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.hist(valid_metrics['nsi_rmse'].dropna(), bins=25, 
             color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.8)
    median_rmse = valid_metrics['nsi_rmse'].median()
    ax_c.axvline(median_rmse, color='blue', linestyle='--', 
                linewidth=2, label=f'Median = {median_rmse:.4f}')
    ax_c.set_xlabel('NSI (RMSE)', fontsize=9)
    ax_c.set_ylabel('Frequency', fontsize=9)
    ax_c.set_title('C. Distribution of NSI (RMSE)\n(Valid Configurations)', fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel D: NSI by model - IMPROVED: show most SENSITIVE (worst NSI)
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Sort by absolute NSI (most negative = most sensitive to noise)
    model_nsi = valid_metrics.groupby('model')['nsi_r2'].median().sort_values()
    top_sensitive = model_nsi.head(10)  # Most negative NSI = most degradation
    
    colors = ['#e74c3c' if nsi < -0.3 else '#f39c12' if nsi < -0.2 else '#3498db' 
              for nsi in top_sensitive.values]
    
    y_pos = np.arange(len(top_sensitive))
    ax_d.barh(y_pos, top_sensitive.values, color=colors, alpha=0.8, 
             edgecolor='black', linewidth=0.5)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([format_model(m) for m in top_sensitive.index], fontsize=8)
    ax_d.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_xlabel('Median NSI (R²)', fontsize=9)
    ax_d.set_title('D. Most Noise-Sensitive Models\n(Lower = More Degradation)', 
                   fontsize=10, fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "supplementary_s2_nsi_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S2 to {output_path}")
    plt.close()


# ============================================================================
# NEW: FIGURE FOR GRAPH NEURAL NETWORK ANALYSIS
# ============================================================================

def create_figure_gnn_analysis(df, metrics_df, output_dir):
    """Create dedicated figure showing GNN performance when properly tuned"""
    print("\n" + "="*80)
    print("GENERATING GNN ANALYSIS FIGURE")
    print("="*80)
    
    # Get graph data only
    graph_data = df[df['representation'] == 'graph'].copy()
    graph_metrics = metrics_df[metrics_df['representation'] == 'graph'].copy()
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.08, right=0.95, top=0.88, bottom=0.15)
    
    # Panel A: All GNN models - showing failure vs success
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Group models
    working_models = [m for m in graph_data['model'].unique() 
                     if m not in CATASTROPHIC_GRAPH_MODELS]
    failing_models = [m for m in graph_data['model'].unique() 
                     if m in CATASTROPHIC_GRAPH_MODELS]
    
    # Plot working models (solid lines)
    for model in working_models:
        model_data = graph_data[graph_data['model'] == model].copy()
        model_data['sigma_rounded'] = model_data['sigma'].round(2)
        avg_by_sigma = model_data.groupby('sigma_rounded')['r2'].mean().reset_index()
        
        color = MODEL_COLORS.get(model.upper(), '#333333')
        ax_a.plot(avg_by_sigma['sigma_rounded'], avg_by_sigma['r2'],
                 marker='o', markersize=5, linewidth=2.5, alpha=0.9,
                 label=format_model(model), color=color, linestyle='-')
    
    # Plot failing models (dashed, lighter)
    for model in failing_models[:3]:  # Limit to avoid clutter
        model_data = graph_data[graph_data['model'] == model].copy()
        model_data['sigma_rounded'] = model_data['sigma'].round(2)
        avg_by_sigma = model_data.groupby('sigma_rounded')['r2'].mean().reset_index()
        
        ax_a.plot(avg_by_sigma['sigma_rounded'], avg_by_sigma['r2'],
                 marker='x', markersize=4, linewidth=1.5, alpha=0.5,
                 label=f"{format_model(model)} (fail)", color='gray', linestyle='--')
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=10)
    ax_a.set_ylabel('R²', fontsize=10)
    ax_a.set_title('A. GNN Performance: Well-Tuned vs Catastrophic Failure', 
                   fontsize=11, fontweight='bold', pad=10)
    ax_a.legend(fontsize=7, loc='lower left', ncol=2, frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(-0.1, 0.95)
    ax_a.axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    # Panel B: Comparison of BNN variants
    ax_b = fig.add_subplot(gs[0, 1])
    
    bnn_variants = {
        'bnn_last': {'color': '#2ecc71', 'label': 'BNN-Last (Works)'},
        'bnn_full': {'color': '#e74c3c', 'label': 'BNN-Full (Fails)'},
    }
    
    base_gnns = ['GCN', 'GAT', 'GIN', 'MPNN']
    
    x_positions = np.arange(len(base_gnns))
    width = 0.35
    
    # Get retention for each variant
    for idx, (variant, style) in enumerate(bnn_variants.items()):
        retentions = []
        for base in base_gnns:
            model_name = f"{base.lower()}_{variant}"
            model_metrics = graph_metrics[graph_metrics['model'] == model_name]
            if len(model_metrics) > 0 and model_metrics['is_valid'].values[0]:
                retentions.append(model_metrics['retention_pct'].values[0])
            else:
                retentions.append(0)  # Failure
        
        offset = width * (idx - 0.5)
        bars = ax_b.bar(x_positions + offset, retentions, width, 
                       color=style['color'], alpha=0.8, label=style['label'],
                       edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, ret in zip(bars, retentions):
            if ret > 0:
                ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                         f'{ret:.0f}%', ha='center', va='bottom', fontsize=7)
            else:
                ax_b.text(bar.get_x() + bar.get_width()/2, 5,
                         'FAIL', ha='center', va='bottom', fontsize=6, color='red')
    
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(base_gnns, fontsize=9)
    ax_b.set_ylabel('Retention at σ=0.6 (%)', fontsize=10)
    ax_b.set_title('B. BNN Variant Comparison Across GNN Architectures', 
                   fontsize=11, fontweight='bold', pad=10)
    ax_b.legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax_b.set_ylim(0, 110)
    ax_b.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add annotation
    fig.text(0.5, 0.02, 
             'Note: Full Bayesian GNNs (BNN-Full) show catastrophic failure with baseline R² ≈ 0, '
             'while last-layer Bayesian variants (BNN-Last) maintain strong performance.',
             ha='center', fontsize=8, style='italic')
    
    output_path = Path(output_dir) / "figure_gnn_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved GNN analysis figure to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables - using valid configs only"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    valid_metrics = get_valid_metrics(metrics_df)
    
    # Table 1: Top 20
    table1 = valid_metrics.nlargest(20, 'robustness_score')[
        ['model', 'representation', 'baseline_r2', 'r2_high', 
         'retention_pct', 'nsi_r2', 'robustness_score']
    ].copy()
    
    # Format model names
    table1['model'] = table1['model'].apply(format_model)
    table1['representation'] = table1['representation'].apply(format_representation)
    
    table1.columns = ['Model', 'Representation', 'Baseline R²', 'R² (σ=0.6)',
                     'Retention (%)', 'NSI (R²)', 'Robustness Score']
    table1.to_csv(output_dir / "table1_top20_robust_configs.csv", index=False, float_format='%.4f')
    
    with open(output_dir / "table1_top20_robust_configs.tex", 'w') as f:
        f.write(table1.to_latex(index=False, float_format="%.3f", escape=False))
    print(f"✓ Saved Table 1")
    
    # Table 2: By representation
    table2 = valid_metrics.groupby('representation').agg({
        'baseline_r2': ['mean', 'std', 'count'],
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean(),
        'robustness_score': ['mean', 'std']
    }).round(4)
    
    # Format index
    table2.index = table2.index.map(format_representation)
    
    table2.to_csv(output_dir / "table2_performance_by_representation.csv")
    print(f"✓ Saved Table 2")
    
    # Table 3: By model (top 20)
    table3 = valid_metrics.groupby('model').agg({
        'baseline_r2': ['mean', 'std'],
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean(),
        'robustness_score': ['mean', 'std']
    }).round(4)
    table3 = table3.sort_values(('robustness_score', 'mean'), ascending=False).head(20)
    
    # Format index
    table3.index = table3.index.map(format_model)
    
    table3.to_csv(output_dir / "table3_performance_by_model.csv")
    print(f"✓ Saved Table 3")
    
    # Table 4: Catastrophic failures summary
    invalid_metrics = metrics_df[~metrics_df['is_valid']]
    if len(invalid_metrics) > 0:
        table4 = invalid_metrics[['model', 'representation', 'baseline_r2', 'r2_high']].copy()
        table4['model'] = table4['model'].apply(format_model)
        table4['representation'] = table4['representation'].apply(format_representation)
        table4.columns = ['Model', 'Representation', 'Baseline R²', 'R² at σ=0.6']
        table4.to_csv(output_dir / "table4_catastrophic_failures.csv", index=False, float_format='%.4f')
        print(f"✓ Saved Table 4 (catastrophic failures)")


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def perform_statistical_tests(df, metrics_df, output_dir):
    """Perform statistical tests - using valid configs"""
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL TESTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    valid_metrics = get_valid_metrics(metrics_df)
    
    results_text = []
    results_text.append("STATISTICAL COMPARISONS - PHASE 0")
    results_text.append("Note: Analysis uses only valid configurations (baseline R² >= 0.1)")
    results_text.append("="*80)
    results_text.append("")
    
    # Test 1: Representation comparisons
    results_text.append("TEST 1: Representation Comparisons (Retention %)")
    results_text.append("-"*80)
    
    reps = valid_metrics['representation'].value_counts().head(6).index
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            rep1, rep2 = reps[i], reps[j]
            data1 = valid_metrics[valid_metrics['representation'] == rep1]['retention_pct'].dropna()
            data2 = valid_metrics[valid_metrics['representation'] == rep2]['retention_pct'].dropna()
            
            if len(data1) >= 3 and len(data2) >= 3:
                from scipy.stats import mannwhitneyu
                stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                mean1, mean2 = data1.mean(), data2.mean()
                
                results_text.append(f"{format_representation(rep1)} vs {format_representation(rep2)}:")
                results_text.append(f"  Mean retention: {mean1:.2f}% vs {mean2:.2f}%")
                results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.4f}")
                
                if p_val < 0.05:
                    winner = rep1 if mean1 > mean2 else rep2
                    results_text.append(f"  → Significant difference, {format_representation(winner)} superior")
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
    print("PHASE 0 ANALYSIS - GLOBAL SCREENING (IMPROVED)")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    df = load_screening_results(old_results_dir, new_results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    
    # Filter MLP/MTL from metrics
    print(f"\n{'='*80}")
    print("APPLYING FINAL FILTERS")
    print(f"{'='*80}")
    print(f"Before filter: {len(metrics_df)} configs")
    metrics_df = metrics_df[~metrics_df['model'].isin(mlp_mtl_models)].copy()
    print(f"After MLP/MTL filter: {len(metrics_df)} configs")
    print(f"Valid configs: {metrics_df['is_valid'].sum()}")
    print(f"Invalid configs: {(~metrics_df['is_valid']).sum()}")
    
    metrics_df = define_robustness_score(metrics_df)
    
    output_dir = Path(new_results_dir) / "phase0_figures_improved"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save full metrics (including invalid for reference)
    metrics_df.to_csv(output_dir / "phase0_robustness_metrics_full.csv", index=False)
    
    # Save valid metrics only
    valid_metrics = get_valid_metrics(metrics_df)
    valid_metrics.to_csv(output_dir / "phase0_robustness_metrics_valid.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir}")
    
    # ANOVA
    perform_variance_decomposition(df, output_dir)
    
    # Figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure1_global_landscape(df, metrics_df, output_dir)
    create_supplementary_degradation_curves(df, metrics_df, output_dir)
    create_figure2_representation_effects(df, metrics_df, output_dir)
    create_supplementary_s1(metrics_df, output_dir)
    create_supplementary_s2(metrics_df, output_dir)
    create_figure_gnn_analysis(df, metrics_df, output_dir)
    
    # Tables and stats
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_tests(df, metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 0 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    print("\nTop 10 Most Robust Configurations:")
    for idx, (_, row) in enumerate(valid_metrics.nlargest(10, 'robustness_score').iterrows(), 1):
        print(f"  {idx}. {format_config_label(row['model'], row['representation'])}: "
              f"Score={row['robustness_score']:.3f}, "
              f"R²₀={row['baseline_r2']:.3f}, "
              f"Retention={row['retention_pct']:.1f}%")
    
    print("\nRepresentation Rankings (by median retention, valid only):")
    rep_ranking = valid_metrics.groupby('representation')['retention_pct'].median().sort_values(ascending=False)
    for idx, (rep, val) in enumerate(rep_ranking.items(), 1):
        print(f"  {idx}. {format_representation(rep)}: {val:.1f}%")
    
    print("\nCatastrophic Failures (excluded from main analysis):")
    invalid = metrics_df[~metrics_df['is_valid']]
    for _, row in invalid.iterrows():
        print(f"  - {format_config_label(row['model'], row['representation'])}: "
              f"baseline R² = {row['baseline_r2']:.4f}")
    
    print("\nGNN Analysis Summary:")
    graph_valid = valid_metrics[valid_metrics['representation'] == 'graph']
    if len(graph_valid) > 0:
        print(f"  Valid GNN configs: {len(graph_valid)}")
        print(f"  Mean retention: {graph_valid['retention_pct'].mean():.1f}%")
        print(f"  Best GNN: {format_model(graph_valid.loc[graph_valid['retention_pct'].idxmax(), 'model'])} "
              f"({graph_valid['retention_pct'].max():.1f}%)")
    
    graph_invalid = invalid[invalid['representation'] == 'graph']
    if len(graph_invalid) > 0:
        print(f"  Failed GNN configs: {len(graph_invalid)}")
        print(f"  → Full Bayesian GNNs (BNN-Full) show catastrophic failure")
        print(f"  → Last-layer Bayesian GNNs (BNN-Last) perform excellently")


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