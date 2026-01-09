"""
Alternative Datasets Analysis - ESOL, hERG
Generates cross-dataset comparison figures

Handles BOTH regression (ESOL) and classification (hERG) tasks
File format: {MODEL}_{REP}_{STRATEGY}.csv

For regression (ESOL):
- Columns: sigma, r2, rmse, mae, model, rep, strategy
- Noise parameter: sigma (0.0-1.0)
- Primary metric: R²

For classification (hERG):
- Columns: flip_prob, accuracy, auc, precision, recall, f1, mcc, model, rep, strategy  
- Noise parameter: flip_prob (0.0-1.0)
- Primary metrics: Accuracy, AUC

Generates:
- Figure: Cross-dataset robustness comparison
- Figure: Task-specific degradation curves
- Tables: Best configurations per dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
DATASET_COLORS = {
    'esol': '#3498db',
    'herg': '#e74c3c',
    'qm9': '#9b59b6',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#16a085',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'gp': '#9b59b6',
    'fullbnn': '#8e44ad',
    'full_bnn': '#8e44ad',
    'lastlayerbnn': '#e67e22',
    'lastlayer_bnn': '#e67e22',
    'varbnn': '#c0392b',
    'var_bnn': '#c0392b',
}

REPRESENTATION_COLORS = {
    'ecfp4': '#DE8F05',
    'pdv': '#56B4E9',
    'sns': '#029E73',
    'mhg_gnn_pretrained': '#CC78BC',
}


def format_label(model, rep):
    """Format model/representation labels for clean display."""
    model_map = {
        'rf': 'RF', 'qrf': 'QRF', 'xgboost': 'XGBoost', 'ngboost': 'NGBoost',
        'dnn': 'DNN', 'gp': 'GP', 'rf_calibrated': 'RF-Cal',
        'fullbnn': 'FullBNN', 'full_bnn': 'FullBNN',
        'lastlayerbnn': 'LLBNN', 'lastlayer_bnn': 'LLBNN',
        'varbnn': 'VarBNN', 'var_bnn': 'VarBNN',
    }
    rep_map = {
        'ecfp4': 'ECFP4', 'ecfp4_class': 'ECFP4', 'ecfp4_calibrated': 'ECFP4-Cal',
        'pdv': 'PDV', 'pdv_class': 'PDV',
        'sns': 'SNS', 'sns_class': 'SNS',
        'mhg_gnn_pretrained': 'MHG-GNN', 'mhggnn_pretrained_class': 'MHG-GNN',
    }
    m = model_map.get(model, model.upper())
    r = rep_map.get(rep, rep.upper().replace('_', '-'))
    return f"{m}/{r}"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset_results(dataset_name, results_dir="../results"):
    """
    Load all results for a specific dataset
    
    File format: {MODEL}_{REP}_{STRATEGY}.csv
    Examples: RF_ECFP4_legacy.csv, DNN_PDV_uniform.csv, FullBNN_PDV_hetero.csv
    
    Args:
        dataset_name: 'esol' or 'herg'
        results_dir: Base results directory
    
    Returns:
        DataFrame with all results, task_type ('regression' or 'classification')
    """
    print(f"\n{'='*80}")
    print(f"LOADING {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    
    dataset_dir = Path(results_dir) / dataset_name
    
    if not dataset_dir.exists():
        print(f"⚠️  Directory not found: {dataset_dir}")
        return pd.DataFrame(), None
    
    # Find all CSV files (excluding special files)
    csv_files = list(dataset_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not any(x in f.name.lower() for x in [
        'uncertainty', 'calibration', 'summary', 'all_results', 'uncalibrated'
    ])]
    
    if not csv_files:
        print(f"⚠️  No result files found in {dataset_dir}")
        return pd.DataFrame(), None
    
    print(f"Found {len(csv_files)} result files")
    
    all_data = []
    task_type = None
    
    for filepath in sorted(csv_files):
        try:
            df = pd.read_csv(filepath)
            
            # Determine task type from columns
            if task_type is None:
                if 'r2' in df.columns or 'rmse' in df.columns:
                    task_type = 'regression'
                elif 'accuracy' in df.columns or 'auc' in df.columns:
                    task_type = 'classification'
                else:
                    print(f"⚠️  Cannot determine task type from {filepath.name}")
                    continue
            
            # Parse filename: {MODEL}_{REP}_{STRATEGY}.csv
            # Handle: RF_ECFP4_legacy, FullBNN_PDV_hetero, RF_MHGGNNpretrained_outlier
            name_parts = filepath.stem.split('_')
            
            if len(name_parts) < 3:
                print(f"⚠️  Cannot parse filename: {filepath.name}")
                continue
            
            # Strategy is always last part
            strategy = name_parts[-1]
            
            # Model and rep are before strategy
            model_rep_parts = name_parts[:-1]
            
            # Parse model (handle multi-part like FullBNN, LastLayerBNN)
            if len(model_rep_parts) == 2:
                model, rep = model_rep_parts
            elif len(model_rep_parts) == 3:
                # Check if first two parts are model (FullBNN, LastLayerBNN, etc)
                potential_model = f"{model_rep_parts[0]}_{model_rep_parts[1]}"
                if any(x in potential_model.lower() for x in ['bnn', 'full', 'layer', 'var']):
                    model = potential_model
                    rep = model_rep_parts[2]
                else:
                    # Likely: RF_MHGGNN_pretrained
                    model = model_rep_parts[0]
                    rep = f"{model_rep_parts[1]}_{model_rep_parts[2]}"
            elif len(model_rep_parts) == 4:
                # Could be: Full_BNN_MHG_GNN or similar
                # Assume first 2 are model, last 2 are rep
                model = f"{model_rep_parts[0]}_{model_rep_parts[1]}"
                rep = f"{model_rep_parts[2]}_{model_rep_parts[3]}"
            else:
                # Fallback
                model = model_rep_parts[0]
                rep = '_'.join(model_rep_parts[1:])
            
            # Normalize names from filename
            model = model.lower().replace('-', '_')
            rep = rep.lower().replace('-', '_')
            strategy = strategy.lower()
            
            # Normalize representation names from filename
            if rep in ['mhggnn', 'mhg_gnn', 'mhggnn_pretrained', 'mhggnnpretrained']:
                rep = 'mhg_gnn_pretrained'
            
            # Add metadata if not already present
            if 'model' not in df.columns:
                df['model'] = model
            if 'representation' not in df.columns:
                df['representation'] = rep
            if 'rep' not in df.columns:
                df['rep'] = rep
            if 'strategy' not in df.columns:
                df['strategy'] = strategy
            
            # Normalize existing columns (in case CSV has different casing/formatting)
            if 'model' in df.columns:
                df['model'] = df['model'].str.lower().str.replace('-', '_')
            if 'representation' in df.columns:
                df['representation'] = df['representation'].str.lower().str.replace('-', '_')
                # Normalize MHG-GNN variants
                df.loc[df['representation'].isin(['mhggnn', 'mhg_gnn', 'mhggnn_pretrained', 
                                                  'mhggnnpretrained', 'mhg-gnn-pretrained']), 
                       'representation'] = 'mhg_gnn_pretrained'
            if 'rep' in df.columns:
                df['rep'] = df['rep'].str.lower().str.replace('-', '_')
                # Normalize MHG-GNN variants
                df.loc[df['rep'].isin(['mhggnn', 'mhg_gnn', 'mhggnn_pretrained', 
                                       'mhggnnpretrained', 'mhg-gnn-pretrained']), 
                       'rep'] = 'mhg_gnn_pretrained'
            if 'strategy' in df.columns:
                df['strategy'] = df['strategy'].str.lower()
            
            df['dataset'] = dataset_name
            df['source_file'] = filepath.name
            
            all_data.append(df)
            print(f"  ✓ Loaded {filepath.name}: {model}/{rep}/{strategy}")
            
        except Exception as e:
            print(f"  ❌ Error loading {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame(), task_type
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Standardize column names
    if 'rep' not in combined_df.columns and 'representation' in combined_df.columns:
        combined_df['rep'] = combined_df['representation']
    
    # Standardize noise parameter column
    if task_type == 'regression':
        if 'sigma' not in combined_df.columns and 'sigma_noise' in combined_df.columns:
            combined_df['sigma'] = combined_df['sigma_noise']
        noise_col = 'sigma'
    else:  # classification
        if 'flip_prob' not in combined_df.columns and 'noise_level' in combined_df.columns:
            combined_df['flip_prob'] = combined_df['noise_level']
        noise_col = 'flip_prob'
    
    # CRITICAL: Filter out known problematic configurations
    n_before = len(combined_df)
    
    if dataset_name == 'herg':
        # DNN + PDV: training failure
        # DNN + PDV_CLASS: 100% retention but predicts majority class only (broken)
        combined_df = combined_df[~((combined_df['model'] == 'dnn') & 
                                    (combined_df['representation'].isin(['pdv', 'pdv_class'])))]
        n_excluded = n_before - len(combined_df)
        if n_excluded > 0:
            print(f"\n⚠️  EXCLUDED {n_excluded} rows: DNN+PDV, DNN+PDV_CLASS (training failures)")
            print(f"    See KNOWN_ISSUES.md for details")
    
    elif dataset_name == 'esol':
        # DNN + PDV: negative R², model failed to learn
        # GP + PDV: very poor baseline (~0.08 R²), unreliable retention metrics
        n_before_esol = len(combined_df)
        combined_df = combined_df[~((combined_df['model'] == 'dnn') & 
                                    (combined_df['representation'] == 'pdv'))]
        combined_df = combined_df[~((combined_df['model'] == 'gp') & 
                                    (combined_df['representation'] == 'pdv'))]
        n_excluded = n_before_esol - len(combined_df)
        if n_excluded > 0:
            print(f"\n⚠️  EXCLUDED {n_excluded} rows: DNN+PDV (negative R²), GP+PDV (poor baseline)")
            print(f"    See KNOWN_ISSUES.md for details")
    
    # Filter noise strategies per dataset
    n_before_strategy = len(combined_df)
    
    if dataset_name == 'esol':
        # Keep only: hetero, legacy
        # Exclude: outlier, quantile, threshold, valprop
        retained_strategies = ['hetero', 'legacy']
        combined_df = combined_df[combined_df['strategy'].isin(retained_strategies)]
        n_excluded_strategy = n_before_strategy - len(combined_df)
        if n_excluded_strategy > 0:
            print(f"\n⚠️  ESOL: Retained strategies: {retained_strategies}")
            print(f"    Excluded {n_excluded_strategy} rows from other strategies")
            print(f"    Reason: Focus on representative noise patterns for cross-dataset comparison")
    
    elif dataset_name == 'herg':
        # Keep only: class_imbalance, uniform
        # Exclude: binary_asymmetric (too easy, no degradation)
        retained_strategies = ['class_imbalance', 'uniform']
        combined_df = combined_df[combined_df['strategy'].isin(retained_strategies)]
        n_excluded_strategy = n_before_strategy - len(combined_df)
        if n_excluded_strategy > 0:
            print(f"\n⚠️  hERG: Retained strategies: {retained_strategies}")
            print(f"    Excluded {n_excluded_strategy} rows from binary_asymmetric")
            print(f"    Reason: binary_asymmetric shows no degradation (uninformative)")
    
    print(f"\n✓ Loaded {len(combined_df)} total rows after filtering")
    print(f"  Task type: {task_type}")
    print(f"  Noise parameter: {noise_col}")
    print(f"  Models ({len(combined_df['model'].unique())}): {sorted(combined_df['model'].unique())}")
    print(f"  Representations ({len(combined_df['representation'].unique())}): {sorted(combined_df['representation'].unique())}")
    print(f"  Strategies ({len(combined_df['strategy'].unique())}): {sorted(combined_df['strategy'].unique())}")
    
    return combined_df, task_type


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_robustness_metrics_regression(df, noise_high=0.5):
    """Calculate robustness metrics for regression tasks"""
    print(f"\n{'='*80}")
    print(f"CALCULATING REGRESSION METRICS (σ_high = {noise_high})")
    print(f"{'='*80}")
    
    metrics_list = []
    
    for (dataset, model, rep, strategy), group in df.groupby(
        ['dataset', 'model', 'representation', 'strategy']
    ):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'dataset': dataset,
            'model': model,
            'representation': rep,
            'strategy': strategy,
        }
        
        # Baseline at σ=0
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # Performance at high noise
        sigma_h = group[np.abs(group['sigma'] - noise_high) < 0.1]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Retention
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            # Only calculate retention for models with reasonable baseline performance
            # Negative or near-zero R² indicates model failure
            if metrics['baseline_r2'] > 0.05:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
                # Cap at 150% to catch edge cases and data issues
                # (retention >100% can occur with noisy metrics but shouldn't be extreme)
                if metrics['retention_pct'] > 150:
                    print(f"  ⚠️  Capped retention from {metrics['retention_pct']:.1f}% to 150% "
                          f"for {model}/{rep}/{strategy} (baseline R²={metrics['baseline_r2']:.4f})")
                    metrics['retention_pct'] = 150.0
            else:
                # Model with very poor baseline - exclude from retention analysis
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # NSI
        if len(group) >= 3:
            try:
                slope_r2, _, _, _, _ = stats.linregress(group['sigma'], group['r2'])
                metrics['nsi_r2'] = slope_r2
            except:
                metrics['nsi_r2'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    print(f"✓ Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


def calculate_robustness_metrics_classification(df, noise_high=0.5):
    """Calculate robustness metrics for classification tasks"""
    print(f"\n{'='*80}")
    print(f"CALCULATING CLASSIFICATION METRICS (flip_prob_high = {noise_high})")
    print(f"{'='*80}")
    
    metrics_list = []
    
    # Find AUC column name (could be 'auc', 'roc_auc', 'auroc', etc.)
    auc_col = None
    for col in df.columns:
        if 'auc' in col.lower() and 'pr' not in col.lower():
            auc_col = col
            break
    
    for (dataset, model, rep, strategy), group in df.groupby(
        ['dataset', 'model', 'representation', 'strategy']
    ):
        group = group.sort_values('flip_prob')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'dataset': dataset,
            'model': model,
            'representation': rep,
            'strategy': strategy,
        }
        
        # Baseline at flip_prob=0
        fp_0 = group[group['flip_prob'] == 0.0]
        if len(fp_0) > 0:
            metrics['baseline_accuracy'] = fp_0['accuracy'].values[0]
            if auc_col and auc_col in fp_0.columns:
                metrics['baseline_auc'] = fp_0[auc_col].values[0]
            else:
                metrics['baseline_auc'] = np.nan
            if 'f1' in fp_0.columns:
                metrics['baseline_f1'] = fp_0['f1'].values[0]
            else:
                metrics['baseline_f1'] = np.nan
        else:
            metrics['baseline_accuracy'] = np.nan
            metrics['baseline_auc'] = np.nan
            metrics['baseline_f1'] = np.nan
        
        # Performance at high noise
        fp_h = group[np.abs(group['flip_prob'] - noise_high) < 0.1]
        if len(fp_h) > 0:
            metrics['accuracy_high'] = fp_h['accuracy'].values[0]
            if auc_col and auc_col in fp_h.columns:
                metrics['auc_high'] = fp_h[auc_col].values[0]
            else:
                metrics['auc_high'] = np.nan
            if 'f1' in fp_h.columns:
                metrics['f1_high'] = fp_h['f1'].values[0]
            else:
                metrics['f1_high'] = np.nan
        else:
            metrics['accuracy_high'] = np.nan
            metrics['auc_high'] = np.nan
            metrics['f1_high'] = np.nan
        
        # Retention (accuracy)
        if not np.isnan(metrics['baseline_accuracy']) and not np.isnan(metrics['accuracy_high']):
            if metrics['baseline_accuracy'] != 0:
                metrics['retention_pct_acc'] = (metrics['accuracy_high'] / metrics['baseline_accuracy']) * 100
            else:
                metrics['retention_pct_acc'] = np.nan
        else:
            metrics['retention_pct_acc'] = np.nan
        
        # Retention (AUC)
        if not np.isnan(metrics['baseline_auc']) and not np.isnan(metrics['auc_high']):
            if metrics['baseline_auc'] != 0:
                metrics['retention_pct_auc'] = (metrics['auc_high'] / metrics['baseline_auc']) * 100
            else:
                metrics['retention_pct_auc'] = np.nan
        else:
            metrics['retention_pct_auc'] = np.nan
        
        # NSI (accuracy)
        if len(group) >= 3:
            try:
                slope_acc, _, _, _, _ = stats.linregress(group['flip_prob'], group['accuracy'])
                metrics['nsi_accuracy'] = slope_acc
            except:
                metrics['nsi_accuracy'] = np.nan
        else:
            metrics['nsi_accuracy'] = np.nan
        
        # NSI (AUC)
        if len(group) >= 3 and auc_col and auc_col in group.columns:
            try:
                slope_auc, _, _, _, _ = stats.linregress(group['flip_prob'], group[auc_col])
                metrics['nsi_auc'] = slope_auc
            except:
                metrics['nsi_auc'] = np.nan
        else:
            metrics['nsi_auc'] = np.nan
        
        # For classification, use accuracy retention as primary retention metric
        metrics['retention_pct'] = metrics['retention_pct_acc']
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    print(f"✓ Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


# ============================================================================
# FIGURE: CROSS-DATASET COMPARISON
# ============================================================================

def create_cross_dataset_figure(all_metrics, output_dir):
    """
    Cross-dataset robustness comparison
    
    Shows how the same model-representation pairs perform across datasets
    """
    print(f"\n{'='*80}")
    print("GENERATING CROSS-DATASET COMPARISON FIGURE")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.98, top=0.94, bottom=0.06)
    
    # ========================================================================
    # Panel A: Robustness ranking consistency across datasets
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Find common model-rep pairs across datasets
    common_configs = set()
    for dataset in all_metrics['dataset'].unique():
        dataset_configs = set(
            zip(all_metrics[all_metrics['dataset'] == dataset]['model'],
                all_metrics[all_metrics['dataset'] == dataset]['representation'])
        )
        if len(common_configs) == 0:
            common_configs = dataset_configs
        else:
            common_configs &= dataset_configs
    
    print(f"Found {len(common_configs)} common model-rep pairs across datasets")
    
    if len(common_configs) >= 3:
        # For each dataset, calculate average retention across strategies
        retention_col = 'retention_pct' if 'retention_pct' in all_metrics.columns else 'retention_pct_acc'
        
        dataset_rankings = {}
        for dataset in sorted(all_metrics['dataset'].unique()):
            dataset_data = all_metrics[all_metrics['dataset'] == dataset]
            avg_retention = dataset_data.groupby(['model', 'representation'])[retention_col].mean()
            dataset_rankings[dataset] = avg_retention
        
        # Plot rankings for top configs
        if len(dataset_rankings) >= 2:
            datasets = list(dataset_rankings.keys())
            x_pos = np.arange(len(common_configs))
            width = 0.8 / len(datasets)
            
            for idx, dataset in enumerate(datasets):
                ranking = dataset_rankings[dataset]
                # Get values for common configs in same order
                values = [ranking.get((m, r), np.nan) for m, r in sorted(common_configs)]
                
                color = DATASET_COLORS.get(dataset, '#999999')
                ax_a.bar(x_pos + idx * width, values, width, 
                        label=dataset.upper(), color=color, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
            
            ax_a.set_xticks(x_pos + width * (len(datasets) - 1) / 2)
            ax_a.set_xticklabels([format_label(m, r) for m, r in sorted(common_configs)], 
                                rotation=45, ha='right', fontsize=6)
            ax_a.set_ylabel('Average Retention (%)', fontsize=9)
            ax_a.set_title('A. Robustness Across Datasets\n(Common Configurations)', 
                          fontsize=10, fontweight='bold', pad=10)
            ax_a.legend(fontsize=7, loc='best')
            ax_a.spines['top'].set_visible(False)
            ax_a.spines['right'].set_visible(False)
            ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax_a.text(0.5, 0.5, 'Insufficient common configurations',
                 ha='center', va='center', transform=ax_a.transAxes, fontsize=10)
        ax_a.axis('off')
    
    # ========================================================================
    # Panel B: Dataset difficulty comparison
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Average retention by dataset
    retention_col = 'retention_pct' if 'retention_pct' in all_metrics.columns else 'retention_pct_acc'
    dataset_difficulty = all_metrics.groupby('dataset')[retention_col].agg(['mean', 'std']).reset_index()
    dataset_difficulty = dataset_difficulty.sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(dataset_difficulty))
    colors = [DATASET_COLORS.get(d, '#999999') for d in dataset_difficulty['dataset']]
    
    ax_b.barh(y_pos, dataset_difficulty['mean'],
             xerr=dataset_difficulty['std'],
             color=colors, alpha=0.8, height=0.6,
             edgecolor='black', linewidth=0.8, capsize=4)
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([d.upper() for d in dataset_difficulty['dataset']], fontsize=8)
    ax_b.set_xlabel('Average Retention % (±std)', fontsize=9)
    ax_b.set_title('B. Dataset Difficulty\n(Lower = Harder)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.axvline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # Panel C: Representation consistency
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Normalize representations to base names before aggregating
    def get_base_rep(rep):
        """Map representation variants to base name."""
        rep = rep.lower()
        if 'ecfp4' in rep and 'calibrated' in rep:
            return 'ECFP4-Cal'
        elif 'ecfp4' in rep:
            return 'ECFP4'
        elif 'mhg' in rep or 'mhggnn' in rep:
            return 'MHG-GNN'
        elif 'pdv' in rep:
            return 'PDV'
        elif 'sns' in rep:
            return 'SNS'
        else:
            return rep.upper()
    
    # Create copy with base representation
    metrics_for_heatmap = all_metrics.copy()
    metrics_for_heatmap['base_rep'] = metrics_for_heatmap['representation'].apply(get_base_rep)
    
    # Exclude ECFP4-Cal (only present in one dataset)
    metrics_for_heatmap = metrics_for_heatmap[metrics_for_heatmap['base_rep'] != 'ECFP4-Cal']
    
    # Average retention by base representation and dataset
    rep_performance = metrics_for_heatmap.groupby(['base_rep', 'dataset'])[retention_col].mean().reset_index()
    
    # Create pivot for heatmap
    if len(rep_performance) > 0:
        pivot = rep_performance.pivot(index='base_rep', 
                                      columns='dataset', 
                                      values=retention_col)
        
        im = ax_c.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=100, interpolation='nearest')
        
        ax_c.set_xticks(np.arange(len(pivot.columns)))
        ax_c.set_yticks(np.arange(len(pivot.index)))
        ax_c.set_xticklabels([d.upper() for d in pivot.columns], fontsize=8)
        ax_c.set_yticklabels(pivot.index, fontsize=8)
        
        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < 50 else 'black'
                    ax_c.text(j, i, f'{value:.0f}', ha='center', va='center',
                            color=text_color, fontsize=7, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label('Retention (%)', fontsize=8, rotation=270, labelpad=15)
        
        ax_c.set_title('C. Representation Performance\nAcross Datasets', 
                      fontsize=10, fontweight='bold', pad=10)
        ax_c.set_xlabel('Dataset', fontsize=9)
        ax_c.set_ylabel('Representation', fontsize=9)
    else:
        ax_c.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax_c.transAxes, fontsize=10)
        ax_c.axis('off')
    
    # ========================================================================
    # Panels D-F: Per-dataset top performers
    # ========================================================================
    for idx, dataset in enumerate(sorted(all_metrics['dataset'].unique())[:3]):
        ax = fig.add_subplot(gs[1, idx])
        
        dataset_data = all_metrics[all_metrics['dataset'] == dataset]
        
        # Get top 10 configurations
        top_10 = dataset_data.groupby(['model', 'representation'])[retention_col].mean().nlargest(10).reset_index()
        
        if len(top_10) > 0:
            y_pos = np.arange(len(top_10))
            labels = [format_label(row['model'], row['representation']) for _, row in top_10.iterrows()]
            
            color = DATASET_COLORS.get(dataset, '#999999')
            ax.barh(y_pos, top_10[retention_col], color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel('Retention (%)', fontsize=8)
            ax.set_title(f'{chr(68+idx)}. {dataset.upper()}: Top 10', 
                        fontsize=9, fontweight='bold', pad=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_xlim(0, 110)
    
    # ========================================================================
    # Save
    # ========================================================================
    output_path = Path(output_dir) / "cross_dataset_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved cross-dataset figure to {output_path}")
    plt.close()


# ============================================================================
# FIGURE: PER-DATASET DEGRADATION CURVES
# ============================================================================

def create_per_dataset_degradation_figures(all_data, output_dir):
    """
    Create degradation curve figures for each dataset.
    Shows TOP 5 configurations only (no bottom 5).
    """
    print(f"\n{'='*80}")
    print("GENERATING PER-DATASET DEGRADATION FIGURES (TOP 5 ONLY)")
    print(f"{'='*80}")
    
    for dataset_name, dataset_info in all_data.items():
        df = dataset_info['data']
        task_type = dataset_info['task_type']
        
        print(f"\nGenerating figure for {dataset_name.upper()}...")
        
        if task_type == 'regression':
            noise_col = 'sigma'
            metric_col = 'r2'
            metric_label = 'R²'
        else:
            noise_col = 'flip_prob'
            metric_col = 'accuracy'
            metric_label = 'Accuracy'
        
        # Calculate retention for each config
        config_retention = []
        for (model, rep, strategy), group in df.groupby(['model', 'representation', 'strategy']):
            group_sorted = group.sort_values(noise_col)
            if len(group_sorted) >= 3:
                baseline = group_sorted[group_sorted[noise_col] == 0.0]
                high_noise = group_sorted[group_sorted[noise_col] >= 0.5]
                
                if len(baseline) > 0 and len(high_noise) > 0:
                    baseline_val = baseline[metric_col].values[0]
                    high_val = high_noise.iloc[0][metric_col]
                    
                    if baseline_val > 0:
                        retention = (high_val / baseline_val) * 100
                        config_retention.append({
                            'model': model,
                            'rep': rep,
                            'strategy': strategy,
                            'retention': retention
                        })
        
        if len(config_retention) == 0:
            print(f"  ⚠️  No configurations with sufficient data for {dataset_name}")
            continue
        
        retention_df = pd.DataFrame(config_retention).sort_values('retention', ascending=False)
        strategies = sorted(df['strategy'].unique())
        n_strategies = len(strategies)
        
        # Single row of panels for top 5 only
        fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))
        if n_strategies == 1:
            axes = [axes]
        
        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            
            # Get top 5 configs for this strategy
            strategy_configs = retention_df[retention_df['strategy'] == strategy].head(5)
            
            for _, config in strategy_configs.iterrows():
                model, rep = config['model'], config['rep']
                config_data = df[
                    (df['model'] == model) &
                    (df['representation'] == rep) &
                    (df['strategy'] == strategy)
                ].sort_values(noise_col)
                
                if len(config_data) > 2:
                    label = format_label(model, rep)
                    ax.plot(config_data[noise_col], config_data[metric_col],
                           marker='o', linewidth=1.5, markersize=4, alpha=0.8,
                           label=label)
            
            ax.set_xlabel(f'Noise Level ({noise_col})', fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(f'Top 5: {strategy}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='lower left')
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle(f'{dataset_name.upper()}: Performance Degradation Curves',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = Path(output_dir) / f"{dataset_name}_degradation_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {dataset_name} degradation curves to {output_path}")
        plt.close()


# ============================================================================
# TABLES
# ============================================================================

def create_summary_tables(all_metrics, output_dir):
    """Create summary tables"""
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY TABLES")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    
    # Table 1: Best configurations per dataset
    retention_col = 'retention_pct' if 'retention_pct' in all_metrics.columns else 'retention_pct_acc'
    
    for dataset in sorted(all_metrics['dataset'].unique()):
        dataset_data = all_metrics[all_metrics['dataset'] == dataset]
        
        top_20 = dataset_data.groupby(['model', 'representation']).agg({
            retention_col: 'mean',
            'baseline_r2' if 'baseline_r2' in dataset_data.columns else 'baseline_accuracy': 'mean'
        }).reset_index().nlargest(20, retention_col)
        
        # Add note about exclusions
        strategies_used = sorted(dataset_data['strategy'].unique())
        
        top_20.to_csv(output_dir / f"table_{dataset}_top20.csv", index=False)
        
        # Write exclusion note
        with open(output_dir / f"table_{dataset}_top20_README.txt", 'w') as f:
            f.write(f"{dataset.upper()} Top 20 Configurations\n")
            f.write("="*50 + "\n\n")
            f.write(f"Strategies included: {', '.join(strategies_used)}\n\n")
            if dataset == 'esol':
                f.write("Excluded configurations: DNN+PDV (negative R²), GP+PDV (poor baseline)\n")
                f.write("Excluded strategies: outlier, quantile, threshold, valprop\n")
                f.write("Reason: Focus on representative noise patterns for cross-dataset comparison\n")
            elif dataset == 'herg':
                f.write("Excluded configurations: DNN+PDV, DNN+PDV_CLASS\n")
                f.write("Reason: Training convergence failure (see KNOWN_ISSUES.md)\n\n")
                f.write("Excluded strategies: binary_asymmetric\n")
                f.write("Reason: No degradation observed (uninformative)\n")
        
        print(f"✓ Saved {dataset} top 20 table with README")
    
    # Table 2: Cross-dataset summary
    summary = all_metrics.groupby(['dataset', 'model', 'representation']).agg({
        retention_col: 'mean'
    }).reset_index()
    
    summary.to_csv(output_dir / "table_cross_dataset_summary.csv", index=False)
    
    # Write cross-dataset README
    with open(output_dir / "table_cross_dataset_summary_README.txt", 'w') as f:
        f.write("Cross-Dataset Summary\n")
        f.write("="*50 + "\n\n")
        f.write("ESOL strategies: hetero, legacy\n")
        f.write("hERG strategies: class_imbalance, uniform\n\n")
        f.write("Exclusions:\n")
        f.write("  ESOL: DNN+PDV, GP+PDV configurations removed (model failures)\n")
        f.write("  ESOL: outlier, quantile, threshold, valprop strategies removed\n")
        f.write("  hERG: DNN+PDV, DNN+PDV_CLASS configurations removed (training failures)\n")
        f.write("  hERG: binary_asymmetric strategy removed (no degradation)\n")
    
    print(f"✓ Saved cross-dataset summary table with README")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution"""
    print("="*80)
    print("ALTERNATIVE DATASETS ANALYSIS")
    print("="*80)
    
    # Try to load each dataset
    datasets_to_load = ['esol', 'herg']
    
    all_data = {}
    all_metrics = []
    
    for dataset_name in datasets_to_load:
        df, task_type = load_dataset_results(dataset_name, results_dir)
        
        if len(df) > 0:
            all_data[dataset_name] = {'data': df, 'task_type': task_type}
            
            # Calculate metrics based on task type
            if task_type == 'regression':
                metrics_df = calculate_robustness_metrics_regression(df, noise_high=0.5)
            else:  # classification
                metrics_df = calculate_robustness_metrics_classification(df, noise_high=0.5)
            
            all_metrics.append(metrics_df)
    
    if not all_metrics:
        print("\n❌ No datasets loaded!")
        return
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Create output directory
    output_dir = Path(results_dir) / "alternative_datasets_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save combined metrics
    combined_metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    print(f"\n✓ Saved combined metrics")
    
    # Generate figures
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    
    create_cross_dataset_figure(combined_metrics, output_dir)
    create_per_dataset_degradation_figures(all_data, output_dir)
    
    # Generate tables
    create_summary_tables(combined_metrics, output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}")
    
    print("\nGenerated files:")
    print("  FIGURES:")
    print("    - cross_dataset_comparison.png (6 panels)")
    for dataset in all_data.keys():
        print(f"    - {dataset}_degradation_curves.png (top 5 per strategy)")
    
    print("  TABLES:")
    for dataset in all_data.keys():
        print(f"    - table_{dataset}_top20.csv")
    print("    - table_cross_dataset_summary.csv")
    print("    - all_metrics.csv")
    
    print("\nDatasets analyzed:")
    for dataset in all_data.keys():
        task = all_data[dataset]['task_type']
        n_configs = len(all_data[dataset]['data'].groupby(['model', 'representation']))
        n_strategies = len(all_data[dataset]['data']['strategy'].unique())
        strategies = sorted(all_data[dataset]['data']['strategy'].unique())
        print(f"  - {dataset.upper()}: {task}, {n_configs} configurations, {n_strategies} strategies")
        print(f"    Strategies: {', '.join(strategies)}")
    
    print("\n" + "="*80)
    print("EXCLUSIONS APPLIED")
    print("="*80)
    print("ESOL:")
    print("  Excluded configurations: DNN+PDV (negative R²), GP+PDV (poor baseline)")
    print("  Excluded strategies: outlier, quantile, threshold, valprop")
    print("  Reason: Focus on representative noise patterns for cross-dataset comparison")
    print("  Retained: hetero, legacy")
    print("\nhERG:")
    print("  Excluded configurations: DNN+PDV, DNN+PDV_CLASS (training failures)")
    print("  Reason: Training convergence failure (see KNOWN_ISSUES.md)")
    print("  Excluded strategies: binary_asymmetric")
    print("  Reason: No degradation observed (uninformative)")
    print("  Retained: class_imbalance, uniform")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)