"""
Phase 2 Uncertainty Analysis - Fixed Version

Addresses feedback:
- Figure 4: Separate panels per model, fixed y-axis scaling, removed misleading threshold
- Figure 5: Extended ECE color scale, clearer calibration curves
- Figure 6: Removed flat ratio plots, standardized y-axes, table for ratios
- Added: Model consistency, MAE degradation tracking, summary comparison figure

Usage:
    python phase2_analysis_fixed.py results/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL STYLE
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

# Consistent model set for all figures
# Primary models for non-graph representations
CORE_MODELS = ['qrf', 'ngboost', 'bnn_full', 'gauche']
# Primary models for graph representations (exclude gauche - it's GP-based, not GNN)
GRAPH_CORE_MODELS = ['gcn_bnn_full', 'gat_bnn_full', 'gin_bnn_full']
# Models that actually work for UQ (for Figure 7 cross-rep comparison)
GOOD_UQ_MODELS = ['qrf', 'ngboost']  # These show actual uncertainty-error correlation

MODEL_COLORS = {
    'qrf': '#3498db',
    'ngboost': '#e74c3c',
    'bnn_full': '#2ecc71',
    'bnn_last': '#27ae60',
    'bnn_variational': '#9b59b6',
    'gauche': '#f39c12',
    'gcn_bnn_full': '#1abc9c',
    'gcn_bnn_last': '#16a085',
    'gcn_bnn_variational': '#148f77',
    'gat_bnn_full': '#9b59b6',
    'gat_bnn_last': '#8e44ad',
    'gat_bnn_variational': '#7d3c98',
    'gin_bnn_full': '#e67e22',
    'gin_bnn_last': '#d35400',
    'gin_bnn_variational': '#ba4a00',
}

MODEL_MARKERS = {
    'qrf': 'o',
    'ngboost': 's',
    'bnn_full': '^',
    'bnn_last': 'v',
    'bnn_variational': 'D',
    'gauche': 'p',
    'gcn_bnn_full': 'o',
    'gcn_bnn_last': 'v',
    'gcn_bnn_variational': 'D',
    'gat_bnn_full': 's',
    'gat_bnn_last': '<',
    'gat_bnn_variational': '>',
    'gin_bnn_full': '^',
    'gin_bnn_last': 'p',
    'gin_bnn_variational': 'h',
}

# Display names for cleaner labels
MODEL_DISPLAY_NAMES = {
    'qrf': 'QRF',
    'ngboost': 'NGBoost', 
    'bnn_full': 'BNN-Full',
    'bnn_last': 'BNN-Last',
    'bnn_variational': 'BNN-Var',
    'gauche': 'GAUCHE',
    'gcn_bnn_full': 'GCN-Full',
    'gcn_bnn_last': 'GCN-Last',
    'gcn_bnn_variational': 'GCN-Var',
    'gat_bnn_full': 'GAT-Full',
    'gat_bnn_last': 'GAT-Last',
    'gat_bnn_variational': 'GAT-Var',
    'gin_bnn_full': 'GIN-Full',
    'gin_bnn_last': 'GIN-Last',
    'gin_bnn_variational': 'GIN-Var',
}

REP_DISPLAY_NAMES = {
    'graph': 'Graph',
    'pdv': 'PDV',
    'smiles_ohe': 'SMILES-OHE',
    'sns': 'SNS',
}

def get_display_name(model):
    return MODEL_DISPLAY_NAMES.get(model, model)

def get_rep_display_name(rep):
    return REP_DISPLAY_NAMES.get(rep, rep)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_uncertainty_data(results_dir):
    """Load all phase2*_uncertainty_values.csv files"""
    print("\n" + "="*80)
    print("LOADING UNCERTAINTY DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    files = list(results_dir.glob("phase2*_uncertainty_values.csv"))
    
    if not files:
        raise FileNotFoundError(f"No phase2*_uncertainty_values.csv files found in {results_dir}")
    
    print(f"\nFound {len(files)} files")
    
    all_data = []
    for filepath in sorted(files):
        df = pd.read_csv(filepath)
        
        if 'model' in df.columns:
            df['model_name'] = df['model']
        
        models = sorted(df['model_name'].unique())
        reps = sorted(df['representation'].unique())
        sigmas = sorted(df['sigma'].unique())
        
        print(f"✓ {filepath.name}:")
        print(f"    {len(df):,} rows | models={models} | reps={reps} | σ={sigmas}")
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data loaded from any files")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"COMBINED DATA")
    print(f"{'='*80}")
    print(f"Total rows: {len(combined):,}")
    print(f"Models: {sorted(combined['model_name'].unique())}")
    print(f"Representations: {sorted(combined['representation'].unique())}")
    print(f"Sigma levels: {sorted(combined['sigma'].unique())}")
    
    has_decomp = ('epistemic_uncertainty' in combined.columns and 
                  'aleatoric_uncertainty' in combined.columns)
    print(f"Decomposition: {'✓ YES' if has_decomp else '✗ NO'}")
    
    return combined


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_ece(uncertainties, errors, n_bins=10):
    """Calculate Expected Calibration Error"""
    if len(uncertainties) < n_bins:
        return np.nan
    
    bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            expected = uncertainties[in_bin].mean()
            observed = errors[in_bin].mean()
            ece += (in_bin.sum() / len(uncertainties)) * abs(expected - observed)
    
    return ece


def calculate_metrics(uncertainty_df):
    """Calculate comprehensive metrics for each model/rep/sigma configuration"""
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    
    metrics = []
    
    for (model, rep, sigma), group in uncertainty_df.groupby(['model_name', 'representation', 'sigma']):
        errors = np.abs(group['y_true_noisy'] - group['y_pred_mean'])
        uncertainties = group['y_pred_std_calibrated']
        
        if uncertainties.isna().all() or len(uncertainties) < 10:
            continue
        
        valid_mask = ~(uncertainties.isna() | errors.isna())
        uncertainties = uncertainties[valid_mask]
        errors = errors[valid_mask]
        
        if len(uncertainties) < 10:
            continue
        
        # Uncertainty-error correlation
        if len(errors) > 1 and uncertainties.std() > 0:
            correlation, p_value = stats.pearsonr(uncertainties, errors)
        else:
            correlation, p_value = np.nan, np.nan
        
        # Expected Calibration Error
        ece = calculate_ece(uncertainties.values, errors.values)
        
        # Coverage
        coverage_1std = np.mean(errors <= uncertainties)
        coverage_2std = np.mean(errors <= 2 * uncertainties)
        
        # MAE for tracking degradation
        mae = errors.mean()
        
        metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std(),
            'mean_absolute_error': mae,
            'ece': ece,
            'coverage_1std': coverage_1std,
            'coverage_2std': coverage_2std,
            'n_samples': len(group),
            'median_uncertainty': uncertainties.median(),
            'median_error': errors.median()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"✓ Calculated metrics for {len(metrics_df)} configurations")
    print(f"  Models: {len(metrics_df['model_name'].unique())}")
    print(f"  Representations: {len(metrics_df['representation'].unique())}")
    print(f"  Sigma levels: {len(metrics_df['sigma'].unique())}")
    
    return metrics_df


def calculate_decomposition(uncertainty_df):
    """Calculate epistemic/aleatoric decomposition metrics"""
    print("\n" + "="*80)
    print("CALCULATING DECOMPOSITION")
    print("="*80)
    
    if 'epistemic_uncertainty' not in uncertainty_df.columns:
        print("⚠️  No epistemic/aleatoric columns found")
        return pd.DataFrame()
    
    decomp_metrics = []
    
    for (model, rep, sigma), group in uncertainty_df.groupby(['model_name', 'representation', 'sigma']):
        epistemic = group['epistemic_uncertainty']
        aleatoric = group['aleatoric_uncertainty']
        
        if epistemic.isna().all() or aleatoric.isna().all():
            continue
        
        valid_mask = ~(epistemic.isna() | aleatoric.isna())
        epistemic = epistemic[valid_mask]
        aleatoric = aleatoric[valid_mask]
        
        if len(epistemic) < 10:
            continue
        
        mean_epistemic = epistemic.mean()
        mean_aleatoric = aleatoric.mean()
        total_uncertainty = np.sqrt(epistemic**2 + aleatoric**2).mean()
        
        ratio = mean_epistemic / mean_aleatoric if mean_aleatoric > 0 else np.nan
        
        decomp_metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'mean_epistemic': mean_epistemic,
            'mean_aleatoric': mean_aleatoric,
            'mean_total': total_uncertainty,
            'epistemic_aleatoric_ratio': ratio,
            'n_samples': len(epistemic)
        })
    
    decomp_df = pd.DataFrame(decomp_metrics)
    print(f"✓ Calculated decomposition for {len(decomp_df)} configurations")
    
    return decomp_df


def get_consistent_models(metrics_df, rep, max_models=4):
    """Get models that exist for this representation, prioritizing models with good data"""
    rep_data = metrics_df[metrics_df['representation'] == rep]
    available = set(rep_data['model_name'].unique())
    
    # Check which models have sufficient data (at least 3 sigma levels)
    good_models = []
    for model in available:
        model_data = rep_data[rep_data['model_name'] == model]
        if len(model_data['sigma'].unique()) >= 3:
            good_models.append(model)
    
    available = set(good_models)
    
    if 'graph' in rep.lower():
        # For graph, prioritize actual GNN models (not gauche)
        priority_models = GRAPH_CORE_MODELS
    else:
        priority_models = CORE_MODELS
    
    result = [m for m in priority_models if m in available]
    
    # Cap at max_models to avoid cluttered legends
    return result[:max_models]


def get_scatter_models(metrics_df, rep):
    """Get exactly 2 models for scatter plot comparison - pick the best ones"""
    rep_data = metrics_df[metrics_df['representation'] == rep]
    available = set(rep_data['model_name'].unique())
    
    if 'graph' in rep.lower():
        # For graph, show one GNN and gauche for comparison
        candidates = ['gcn_bnn_full', 'gat_bnn_full', 'gauche']
    else:
        # For non-graph, QRF and NGBoost are the most informative
        candidates = ['qrf', 'ngboost', 'bnn_full', 'gauche']
    
    result = [m for m in candidates if m in available]
    return result[:2]


# ============================================================================
# FIGURE 4: UNCERTAINTY-ERROR RELATIONSHIPS (FIXED)
# ============================================================================

def create_figure4(uncertainty_df, metrics_df, output_dir):
    """
    Figure 4: Uncertainty-Error Relationships
    
    FIXES:
    - Separate scatter subpanels per model (no overlap)
    - Consistent y-axis scaling across rows
    - Clean styling without garish colored zones
    - Sequential panel labels
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 4 (FIXED)")
    print("="*80)
    
    available_reps = sorted(metrics_df['representation'].unique())
    n_rows = len(available_reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    # Calculate global y-axis limits for UQ Quality consistency
    valid_corr = metrics_df['correlation'].dropna()
    if len(valid_corr) > 0:
        global_corr_min = max(-0.2, valid_corr.min() - 0.05)
        global_corr_max = min(1.0, valid_corr.max() + 0.05)
    else:
        global_corr_min, global_corr_max = -0.2, 0.6
    
    fig = plt.figure(figsize=(18, 4.5*n_rows))
    
    # 4 columns: 2 scatter subpanels + correlation + inflation
    gs = fig.add_gridspec(n_rows, 4, hspace=0.35, wspace=0.28,
                          left=0.05, right=0.98, top=0.94, bottom=0.06,
                          width_ratios=[1, 1, 1.2, 1.2])
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(available_reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        scatter_models = get_scatter_models(metrics_df, rep)
        line_models = get_consistent_models(metrics_df, rep)
        
        if len(scatter_models) == 0:
            continue
        
        rep_name = get_rep_display_name(rep)
        print(f"  {rep}: scatter={scatter_models}, lines={line_models}")
        
        # ====================================================================
        # Panels A-B: Separate scatter plots per model
        # ====================================================================
        for scatter_idx, model in enumerate(scatter_models):
            ax = fig.add_subplot(gs[row_idx, scatter_idx])
            
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.05)
            ]
            
            if len(data) < 50:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(data)})',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: {get_display_name(model)}', 
                            fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                panel_idx += 1
                continue
            
            # Subsample for clarity
            if len(data) > 2000:
                data = data.sample(2000, random_state=42)
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            if len(errors) < 30:
                ax.text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center', 
                       transform=ax.transAxes)
                panel_idx += 1
                continue
            
            color = MODEL_COLORS.get(model, '#999999')
            
            # Clean scatter with good alpha
            ax.scatter(uncertainties, errors, s=6, alpha=0.25, color=color, 
                      edgecolors='none', rasterized=True)
            
            # Perfect calibration line
            max_val = max(uncertainties.max(), errors.max()) * 1.05
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5, 
                   label='y=x', zorder=10)
            
            # Correlation annotation
            if len(uncertainties) > 10 and uncertainties.std() > 0:
                corr, _ = stats.pearsonr(uncertainties, errors)
                ax.annotate(f'r = {corr:.2f}', xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='gray', alpha=0.9))
            
            ax.set_xlabel('Predicted Uncertainty')
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: {get_display_name(model)} (σ=0.3)', 
                        fontweight='bold')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
            sns.despine(ax=ax)
            panel_idx += 1
        
        # Fill empty scatter slots if only 1 model
        while scatter_idx < 1:
            scatter_idx += 1
            ax = fig.add_subplot(gs[row_idx, scatter_idx])
            ax.axis('off')
            panel_idx += 1
        
        # ====================================================================
        # Panel C: UQ Quality (correlation across σ)
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        for model in line_models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['correlation'],
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        # Simple reference line instead of garish zones
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(0.3, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Uncertainty-Error Correlation')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: UQ Quality', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_ylim(global_corr_min, global_corr_max)
        ax.set_xlim(-0.02, 1.02)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel D: Uncertainty inflation
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 3])
        
        for model in line_models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_uncertainty'],
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        # Ideal inflation line
        sigma_range = np.linspace(0, 1.0, 20)
        baseline_data = rep_metrics[np.abs(rep_metrics['sigma']) < 0.05]
        if len(baseline_data) > 0:
            baseline = baseline_data['mean_uncertainty'].median()
            if not np.isnan(baseline):
                ax.plot(sigma_range, baseline + sigma_range, 'k--', 
                       linewidth=1.5, alpha=0.5, label='Ideal')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Uncertainty Inflation', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        sns.despine(ax=ax)
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure4_uncertainty_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: CALIBRATION ANALYSIS (FIXED)
# ============================================================================

def create_figure5(uncertainty_df, metrics_df, output_dir):
    """
    Figure 5: Calibration Analysis
    
    FIXES:
    - Extended ECE color scale to capture full range
    - Cleaner coverage plot without ugly text annotations
    - Consistent model selection
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 5 (FIXED)")
    print("="*80)
    
    available_reps = sorted(metrics_df['representation'].unique())
    n_rows = len(available_reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    # Calculate actual ECE range for proper color scaling
    ece_max = metrics_df['ece'].max()
    ece_vmax = max(0.5, np.ceil(ece_max * 10) / 10)  # Round up to nearest 0.1
    print(f"  ECE range: 0 - {ece_max:.3f}, using vmax={ece_vmax}")
    
    fig = plt.figure(figsize=(16, 4.5*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.40, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(available_reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        available_models = get_consistent_models(metrics_df, rep)
        rep_name = get_rep_display_name(rep)
        
        if len(available_models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Reliability diagram at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        plotted_any = False
        for model in available_models:
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.05)
            ]
            
            if len(data) < 50:
                continue
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            if len(errors) < 30:
                continue
            
            # Binned reliability diagram
            n_bins = 10
            try:
                bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8
            except Exception:
                continue
            
            bin_centers, bin_rmse = [], []
            for i in range(n_bins):
                in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
                if in_bin.sum() > 5:
                    bin_centers.append(uncertainties[in_bin].mean())
                    bin_rmse.append(np.sqrt(np.mean(errors[in_bin]**2)))
            
            if len(bin_centers) >= 3:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(bin_centers, bin_rmse, marker=marker, linewidth=2, 
                       markersize=6, color=color, alpha=0.8, label=get_display_name(model))
                plotted_any = True
        
        # Perfect calibration line
        if plotted_any and len(ax.lines) > 0:
            all_vals = []
            for line in ax.lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(xdata) > 0:
                    all_vals.extend(xdata)
                    all_vals.extend(ydata)
            if all_vals:
                max_val = max(all_vals) * 1.05
                ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Mean Predicted Uncertainty')
        ax.set_ylabel('Observed RMSE')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Reliability (σ=0.3)', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel B: Coverage across σ
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in available_models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                coverage = model_data['coverage_1std'] * 100
                
                # Use alpha to indicate poor coverage, but no special styling
                ax.plot(model_data['sigma'], coverage,
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        ax.axhline(68, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (68%)')
        
        # Subtle shading for problematic region instead of ugly text
        ax.axhspan(0, 20, alpha=0.08, color='red', zorder=0)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Coverage at 1σ (%)')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Coverage', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 105)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel C: ECE heatmap
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        # Get all models for this representation (for heatmap we can show more)
        all_rep_models = sorted(rep_metrics['model_name'].unique())
        
        pivot = rep_metrics.pivot_table(
            values='ece',
            index='model_name',
            columns='sigma',
            aggfunc='mean'
        )
        
        if len(pivot) > 0:
            # Reorder to put important models first
            ordered_models = [m for m in available_models if m in pivot.index]
            other_models = [m for m in pivot.index if m not in ordered_models]
            new_order = ordered_models + other_models
            pivot = pivot.reindex([m for m in new_order if m in pivot.index])
            
            im = ax.imshow(pivot.values, cmap='RdYlGn_r', 
                          aspect='auto', vmin=0, vmax=ece_vmax)
            
            # Add value annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > ece_vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=6, color=text_color)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], fontsize=7)
            ax.set_yticklabels([get_display_name(m) for m in pivot.index], fontsize=7)
            
            ax.set_xlabel('Noise Level (σ)')
            ax.set_ylabel('Model')
            ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: ECE', fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('ECE (lower = better)', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure5_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 6: EPISTEMIC/ALEATORIC DECOMPOSITION (FIXED)
# ============================================================================

def create_figure6(decomp_df, output_dir):
    """
    Figure 6: Epistemic/Aleatoric Decomposition
    
    FIXES:
    - Standardized y-axis scales across panels
    - Cleaner legends with display names
    - Ratios annotated on bars instead of separate plot
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 6 (FIXED)")
    print("="*80)
    
    if len(decomp_df) == 0:
        print("⚠️  No decomposition data available")
        return
    
    available_reps = sorted(decomp_df['representation'].unique())
    n_rows = len(available_reps)
    
    # Calculate global y-axis limits
    global_epistemic_max = decomp_df['mean_epistemic'].max() * 1.15
    global_aleatoric_max = decomp_df['mean_aleatoric'].max() * 1.15
    global_total_max = (decomp_df['mean_epistemic'] + decomp_df['mean_aleatoric']).max() * 1.2
    
    fig = plt.figure(figsize=(16, 4.5*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.40, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(available_reps):
        rep_decomp = decomp_df[decomp_df['representation'] == rep]
        
        # Get models with sufficient data
        model_counts = rep_decomp.groupby('model_name')['sigma'].nunique()
        available_models = [m for m in model_counts[model_counts >= 3].index]
        
        # Limit to reasonable number
        if 'graph' in rep.lower():
            priority = GRAPH_CORE_MODELS + ['gauche']
        else:
            priority = CORE_MODELS
        available_models = [m for m in priority if m in available_models][:5]
        
        rep_name = get_rep_display_name(rep)
        
        if len(available_models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Epistemic uncertainty across noise
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        for model in available_models:
            model_data = rep_decomp[rep_decomp['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_epistemic'],
                       marker=marker, linewidth=2, markersize=5,
                       label=get_display_name(model), color=color, alpha=0.9)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Epistemic Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Epistemic', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, global_epistemic_max)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel B: Aleatoric uncertainty across noise
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in available_models:
            model_data = rep_decomp[rep_decomp['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_aleatoric'],
                       marker=marker, linewidth=2, markersize=5,
                       label=get_display_name(model), color=color, alpha=0.9)
        
        # Ideal: aleatoric should track noise
        sigma_range = np.linspace(0, 1.0, 20)
        ax.plot(sigma_range, sigma_range, 'k--', linewidth=1.5, alpha=0.5, label='Ideal (σ)')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Aleatoric Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Aleatoric', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, global_aleatoric_max)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel C: Stacked bar at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        sigma_03 = rep_decomp[np.abs(rep_decomp['sigma'] - 0.3) < 0.05]
        
        if len(sigma_03) > 0:
            # Get models that have data at σ=0.3
            models_with_data = [m for m in available_models 
                               if m in sigma_03['model_name'].values]
            
            if len(models_with_data) > 0:
                x = np.arange(len(models_with_data))
                
                epist = [sigma_03[sigma_03['model_name'] == m]['mean_epistemic'].mean() 
                        for m in models_with_data]
                alea = [sigma_03[sigma_03['model_name'] == m]['mean_aleatoric'].mean() 
                       for m in models_with_data]
                ratios = [sigma_03[sigma_03['model_name'] == m]['epistemic_aleatoric_ratio'].mean() 
                         for m in models_with_data]
                
                ax.bar(x, alea, label='Aleatoric', alpha=0.85, color='#3498db', edgecolor='white')
                ax.bar(x, epist, bottom=alea, label='Epistemic', alpha=0.85, color='#e74c3c', edgecolor='white')
                
                # Add ratio annotations
                for i, (e, a, r) in enumerate(zip(epist, alea, ratios)):
                    if not np.isnan(r):
                        ax.text(i, e + a + global_total_max * 0.02, f'{r:.2f}',
                               ha='center', va='bottom', fontsize=7, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels([get_display_name(m) for m in models_with_data], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Uncertainty')
                ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Decomposition (σ=0.3)', fontweight='bold')
                ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
                ax.set_ylim(0, global_total_max)
                
                # Add text explaining the numbers
                ax.text(0.98, 0.98, 'E/A ratio', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=6, style='italic', color='gray')
        
        sns.despine(ax=ax)
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure6_decomposition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# NEW: FIGURE 7 - CROSS-REPRESENTATION COMPARISON
# ============================================================================

def create_figure7(metrics_df, output_dir):
    """
    Figure 7: Cross-representation comparison for a consistent method
    
    Shows how a well-performing model behaves across all representations
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 7 (CROSS-REP COMPARISON)")
    print("="*80)
    
    # Pick a model that: 1) exists across most representations, 2) actually has good UQ
    # Prefer NGBoost or QRF over Gauche
    available_reps = set(metrics_df['representation'].unique())
    
    best_model = None
    best_score = -1
    
    for model in GOOD_UQ_MODELS + CORE_MODELS:
        model_data = metrics_df[metrics_df['model_name'] == model]
        rep_count = model_data['representation'].nunique()
        mean_corr = model_data['correlation'].mean()
        
        # Score by number of reps * mean correlation
        if not np.isnan(mean_corr):
            score = rep_count * (mean_corr + 0.5)  # Add offset to handle negative corr
            if score > best_score:
                best_score = score
                best_model = model
    
    if best_model is None:
        print("⚠️  No suitable model found for cross-rep comparison")
        return
    
    model_data = metrics_df[metrics_df['model_name'] == best_model]
    reps = sorted(model_data['representation'].unique())
    
    if len(reps) < 2:
        print(f"⚠️  Model {best_model} only has {len(reps)} representation(s)")
        return
    
    print(f"  Selected model: {best_model} (available in {len(reps)} representations)")
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    rep_colors = {rep: plt.cm.Set2(i/len(reps)) for i, rep in enumerate(reps)}
    
    model_display = get_display_name(best_model)
    
    # Panel A: MAE degradation
    ax = axes[0]
    for rep in reps:
        rep_data = model_data[model_data['representation'] == rep].sort_values('sigma')
        if len(rep_data) >= 2:
            ax.plot(rep_data['sigma'], rep_data['mean_absolute_error'],
                   marker='o', linewidth=2, markersize=5,
                   label=get_rep_display_name(rep), color=rep_colors[rep])
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('MAE')
    ax.set_title(f'A. {model_display}: MAE Degradation', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    # Panel B: Correlation comparison
    ax = axes[1]
    for rep in reps:
        rep_data = model_data[model_data['representation'] == rep].sort_values('sigma')
        if len(rep_data) >= 2:
            ax.plot(rep_data['sigma'], rep_data['correlation'],
                   marker='o', linewidth=2, markersize=5,
                   label=get_rep_display_name(rep), color=rep_colors[rep])
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Uncertainty-Error Correlation')
    ax.set_title(f'B. {model_display}: UQ Quality', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    # Panel C: Coverage comparison
    ax = axes[2]
    for rep in reps:
        rep_data = model_data[model_data['representation'] == rep].sort_values('sigma')
        if len(rep_data) >= 2:
            ax.plot(rep_data['sigma'], rep_data['coverage_1std'] * 100,
                   marker='o', linewidth=2, markersize=5,
                   label=get_rep_display_name(rep), color=rep_colors[rep])
    ax.axhline(68, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'C. {model_display}: Coverage', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    
    # Panel D: ECE comparison
    ax = axes[3]
    for rep in reps:
        rep_data = model_data[model_data['representation'] == rep].sort_values('sigma')
        if len(rep_data) >= 2:
            ax.plot(rep_data['sigma'], rep_data['ece'],
                   marker='o', linewidth=2, markersize=5,
                   label=get_rep_display_name(rep), color=rep_colors[rep])
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('ECE')
    ax.set_title(f'D. {model_display}: Calibration Error', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    # Panel E: Uncertainty inflation comparison
    ax = axes[4]
    for rep in reps:
        rep_data = model_data[model_data['representation'] == rep].sort_values('sigma')
        if len(rep_data) >= 2:
            ax.plot(rep_data['sigma'], rep_data['mean_uncertainty'],
                   marker='o', linewidth=2, markersize=5,
                   label=get_rep_display_name(rep), color=rep_colors[rep])
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Mean Uncertainty')
    ax.set_title(f'E. {model_display}: Uncertainty Inflation', fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    # Panel F: Summary bar chart at σ=0.3
    ax = axes[5]
    sigma_03 = model_data[np.abs(model_data['sigma'] - 0.3) < 0.05]
    
    if len(sigma_03) > 0:
        reps_with_data = [r for r in reps if r in sigma_03['representation'].values]
        x = np.arange(len(reps_with_data))
        width = 0.35
        
        corrs = [sigma_03[sigma_03['representation'] == r]['correlation'].mean() 
                for r in reps_with_data]
        covs = [sigma_03[sigma_03['representation'] == r]['coverage_1std'].mean() 
               for r in reps_with_data]
        
        ax.bar(x - width/2, corrs, width, label='Correlation', alpha=0.85, color='#3498db')
        ax.bar(x + width/2, covs, width, label='Coverage (norm)', alpha=0.85, color='#2ecc71')
        
        ax.set_xticks(x)
        ax.set_xticklabels([get_rep_display_name(r) for r in reps_with_data], 
                         rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Value')
        ax.set_title(f'F. {model_display}: Summary (σ=0.3)', fontweight='bold')
        ax.legend(fontsize=7, framealpha=0.9)
        ax.axhline(0, color='gray', linewidth=0.8)
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "figure7_representation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# NEW: FIGURE 8 - GNN FAILURE ANALYSIS
# ============================================================================

def create_figure8(metrics_df, output_dir):
    """
    Figure 8: GNN UQ Failure Analysis
    
    Highlights that graph neural networks fail at uncertainty quantification
    Note: Gauche is GP-based, not a GNN, so excluded from GNN analysis
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8 (GNN ANALYSIS)")
    print("="*80)
    
    # Separate GNN models (on graph representation) from non-GNN
    # Gauche is NOT a GNN - it's GP-based and works on all representations
    graph_data = metrics_df[
        (metrics_df['representation'].str.contains('graph', case=False)) &
        (~metrics_df['model_name'].str.contains('gauche', case=False))
    ]
    
    non_graph_data = metrics_df[
        ~metrics_df['representation'].str.contains('graph', case=False)
    ]
    
    if len(graph_data) == 0:
        print("⚠️  No graph representation data found")
        return
    
    if len(non_graph_data) == 0:
        print("⚠️  No non-graph data for comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Get GNN models to show (limit to core ones)
    gnn_models = [m for m in GRAPH_CORE_MODELS 
                  if m in graph_data['model_name'].unique()]
    
    # Panel A: GNN coverage collapse
    ax = axes[0]
    
    for model in gnn_models:
        model_data = graph_data[graph_data['model_name'] == model].sort_values('sigma')
        if len(model_data) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax.plot(model_data['sigma'], model_data['coverage_1std'] * 100,
                   marker=marker, linewidth=2, markersize=5,
                   label=get_display_name(model), color=color)
    
    ax.axhline(68, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8, label='Target')
    ax.axhspan(0, 20, alpha=0.1, color='red', zorder=0)
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('A. GNN Models: Coverage Collapse', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    
    # Panel B: GNN vs Non-GNN correlation comparison
    ax = axes[1]
    
    # Average across models for each group
    gnn_avg = graph_data.groupby('sigma')['correlation'].agg(['mean', 'std']).reset_index()
    non_gnn_avg = non_graph_data.groupby('sigma')['correlation'].agg(['mean', 'std']).reset_index()
    
    ax.plot(gnn_avg['sigma'], gnn_avg['mean'], 'o-', linewidth=2.5, 
           label='GNN methods', color='#e74c3c', markersize=6)
    ax.fill_between(gnn_avg['sigma'], 
                   gnn_avg['mean'] - gnn_avg['std'],
                   gnn_avg['mean'] + gnn_avg['std'],
                   alpha=0.2, color='#e74c3c')
    
    ax.plot(non_gnn_avg['sigma'], non_gnn_avg['mean'], 's-', linewidth=2.5,
           label='Non-GNN methods', color='#3498db', markersize=6)
    ax.fill_between(non_gnn_avg['sigma'],
                   non_gnn_avg['mean'] - non_gnn_avg['std'],
                   non_gnn_avg['mean'] + non_gnn_avg['std'],
                   alpha=0.2, color='#3498db')
    
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Uncertainty-Error Correlation')
    ax.set_title('B. GNN vs Non-GNN: UQ Quality', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    # Panel C: ECE comparison
    ax = axes[2]
    
    gnn_ece = graph_data.groupby('sigma')['ece'].agg(['mean', 'std']).reset_index()
    non_gnn_ece = non_graph_data.groupby('sigma')['ece'].agg(['mean', 'std']).reset_index()
    
    ax.plot(gnn_ece['sigma'], gnn_ece['mean'], 'o-', linewidth=2.5,
           label='GNN methods', color='#e74c3c', markersize=6)
    ax.fill_between(gnn_ece['sigma'],
                   gnn_ece['mean'] - gnn_ece['std'],
                   gnn_ece['mean'] + gnn_ece['std'],
                   alpha=0.2, color='#e74c3c')
    
    ax.plot(non_gnn_ece['sigma'], non_gnn_ece['mean'], 's-', linewidth=2.5,
           label='Non-GNN methods', color='#3498db', markersize=6)
    ax.fill_between(non_gnn_ece['sigma'],
                   non_gnn_ece['mean'] - non_gnn_ece['std'],
                   non_gnn_ece['mean'] + non_gnn_ece['std'],
                   alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Expected Calibration Error')
    ax.set_title('C. GNN vs Non-GNN: Calibration Error', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "figure8_gnn_failure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES (ENHANCED)
# ============================================================================

def create_tables(metrics_df, decomp_df, output_dir):
    """Create summary tables with cleaner formatting"""
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Overall performance summary
    table1 = metrics_df.groupby(['model_name', 'representation']).agg({
        'correlation': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'coverage_1std': ['mean', 'std'],
        'mean_absolute_error': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(4)
    table1.columns = ['_'.join(col).strip() for col in table1.columns.values]
    table1.to_csv(output_dir / "table1_model_summary.csv")
    print(f"✓ table1_model_summary.csv")
    
    # Table 2: Performance at key noise levels
    for sigma in [0.0, 0.3, 0.6, 1.0]:
        sigma_data = metrics_df[np.abs(metrics_df['sigma'] - sigma) < 0.05]
        if len(sigma_data) > 0:
            table = sigma_data.pivot_table(
                values=['correlation', 'ece', 'coverage_1std', 'mean_absolute_error'],
                index='model_name',
                columns='representation',
                aggfunc='mean'
            ).round(4)
            table.to_csv(output_dir / f"table2_sigma{sigma:.1f}.csv")
            print(f"✓ table2_sigma{sigma:.1f}.csv")
    
    # Table 3: Decomposition ratios
    if len(decomp_df) > 0:
        ratio_table = decomp_df.groupby(['model_name', 'representation']).agg({
            'epistemic_aleatoric_ratio': ['mean', 'std'],
            'mean_epistemic': 'mean',
            'mean_aleatoric': 'mean'
        }).round(4)
        ratio_table.columns = ['_'.join(col).strip() for col in ratio_table.columns.values]
        ratio_table.to_csv(output_dir / "table3_decomposition_ratios.csv")
        print(f"✓ table3_decomposition_ratios.csv")
        
        # Print key finding
        print("\n  📊 Epistemic/Aleatoric ratios by model:")
        ratio_summary = decomp_df.groupby('model_name')['epistemic_aleatoric_ratio'].agg(['mean', 'std'])
        print(ratio_summary.round(3).to_string())
    
    # Table 4: MAE degradation
    mae_pivot = metrics_df.pivot_table(
        values='mean_absolute_error',
        index='model_name',
        columns='sigma',
        aggfunc='mean'
    ).round(4)
    mae_pivot.to_csv(output_dir / "table4_mae_degradation.csv")
    print(f"✓ table4_mae_degradation.csv")


# ============================================================================
# GAUCHE ANALYSIS (NEW)
# ============================================================================

def analyze_gauche_behavior(metrics_df, output_dir):
    """
    Analyze Gauche's flat uncertainty inflation behavior
    """
    print("\n" + "="*80)
    print("ANALYZING GAUCHE BEHAVIOR")
    print("="*80)
    
    gauche_data = metrics_df[metrics_df['model_name'] == 'gauche']
    
    if len(gauche_data) == 0:
        print("⚠️  No Gauche data found")
        return
    
    analysis_lines = []
    analysis_lines.append("GAUCHE UNCERTAINTY INFLATION ANALYSIS")
    analysis_lines.append("=" * 50)
    analysis_lines.append("")
    analysis_lines.append("FINDING: Gauche shows nearly flat uncertainty inflation")
    analysis_lines.append("compared to other methods that increase with noise.")
    analysis_lines.append("")
    analysis_lines.append("Mean uncertainty by noise level:")
    
    for rep in gauche_data['representation'].unique():
        rep_data = gauche_data[gauche_data['representation'] == rep].sort_values('sigma')
        analysis_lines.append(f"\n  {rep}:")
        for _, row in rep_data.iterrows():
            analysis_lines.append(f"    σ={row['sigma']:.1f}: uncertainty={row['mean_uncertainty']:.4f}")
        
        # Calculate inflation rate
        if len(rep_data) >= 2:
            sigma_range = rep_data['sigma'].max() - rep_data['sigma'].min()
            unc_range = rep_data['mean_uncertainty'].max() - rep_data['mean_uncertainty'].min()
            inflation_rate = unc_range / sigma_range if sigma_range > 0 else 0
            analysis_lines.append(f"    Inflation rate: {inflation_rate:.4f} (ideal: 1.0)")
    
    analysis_lines.append("")
    analysis_lines.append("POSSIBLE EXPLANATIONS:")
    analysis_lines.append("1. Gauche uses GP-based uncertainty that may be dominated by")
    analysis_lines.append("   kernel hyperparameters rather than noise estimation")
    analysis_lines.append("2. The model may be underfitting and capturing only epistemic")
    analysis_lines.append("   uncertainty from limited training data")
    analysis_lines.append("3. Hyperparameters may not be re-optimized for noisy conditions")
    
    output_path = Path(output_dir) / "gauche_analysis.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(analysis_lines))
    print(f"✓ Saved analysis to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="results"):
    """Main analysis function"""
    print("="*80)
    print("PHASE 2: UNCERTAINTY ANALYSIS")
    print("="*80)
    
    # Load data
    uncertainty_df = load_uncertainty_data(results_dir)
    if len(uncertainty_df) == 0:
        raise ValueError("No data loaded")
    
    # Calculate metrics
    metrics_df = calculate_metrics(uncertainty_df)
    if len(metrics_df) == 0:
        raise ValueError("No metrics calculated")
    
    # Calculate decomposition
    decomp_df = calculate_decomposition(uncertainty_df)
    
    # Create output directory
    output_dir = Path(results_dir) / "phase2_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\n✓ Saved metrics.csv")
    
    if len(decomp_df) > 0:
        decomp_df.to_csv(output_dir / "decomposition.csv", index=False)
        print(f"✓ Saved decomposition.csv")
    
    # Generate figures
    create_figure4(uncertainty_df, metrics_df, output_dir)
    create_figure5(uncertainty_df, metrics_df, output_dir)
    
    if len(decomp_df) > 0:
        create_figure6(decomp_df, output_dir)
    
    create_figure7(metrics_df, output_dir)
    create_figure8(metrics_df, output_dir)
    
    # Generate tables
    create_tables(metrics_df, decomp_df, output_dir)
    
    # Gauche analysis
    analyze_gauche_behavior(metrics_df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nData: {len(uncertainty_df):,} samples")
    print(f"Configurations: {len(metrics_df)}")
    print(f"Models: {sorted(metrics_df['model_name'].unique())}")
    print(f"Representations: {sorted(metrics_df['representation'].unique())}")
    
    print("\n" + "-"*40)
    print("KEY METRICS BY MODEL")
    print("-"*40)
    
    summary = metrics_df.groupby('model_name').agg({
        'correlation': 'mean',
        'coverage_1std': lambda x: x.mean() * 100,
        'ece': 'mean',
        'mean_absolute_error': 'mean'
    }).round(3)
    summary.columns = ['Corr', 'Cov%', 'ECE', 'MAE']
    print(summary.to_string())
    
    # Identify problematic methods
    low_coverage = summary[summary['Cov%'] < 30]
    if len(low_coverage) > 0:
        print(f"\n⚠️  Methods with low coverage (<30%): {list(low_coverage.index)}")
    
    low_corr = summary[summary['Corr'] < 0.1]
    if len(low_corr) > 0:
        print(f"⚠️  Methods with poor UQ correlation (<0.1): {list(low_corr.index)}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_dir)