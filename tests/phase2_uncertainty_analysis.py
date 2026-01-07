"""
Phase 2 Uncertainty Analysis - Clean Version

Loads per-sample uncertainty data and calculates:
- Uncertainty-error correlation
- Calibration metrics (ECE, coverage)
- Epistemic/aleatoric decomposition

Generates:
- Figure 4: Uncertainty-error relationships
- Figure 5: Calibration analysis
- Figure 6: Epistemic/aleatoric decomposition
- Summary tables

Usage:
    python phase2_analysis.py results/
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

MODEL_COLORS = {
    'qrf': '#3498db',
    'ngboost': '#e74c3c',
    'bnn_full': '#2ecc71',
    'bnn_last': '#95a5a6',
    'bnn_variational': '#9b59b6',
    'gauche': '#f39c12',
    # Graph models
    'gcn_bnn_full': '#2ecc71',
    'gcn_bnn_last': '#95a5a6',
    'gcn_bnn_variational': '#9b59b6',
    'gat_bnn_full': '#1abc9c',
    'gat_bnn_last': '#16a085',
    'gat_bnn_variational': '#d35400',
    'gin_bnn_full': '#e67e22',
    'gin_bnn_last': '#c0392b',
    'gin_bnn_variational': '#8e44ad',
}

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
        print("❌ ERROR: No phase2*_uncertainty_values.csv files found!")
        return pd.DataFrame()
    
    print(f"\nFound {len(files)} files")
    
    all_data = []
    for filepath in sorted(files):
        try:
            df = pd.read_csv(filepath)
            
            # Rename 'model' column to 'model_name' for consistency
            if 'model' in df.columns:
                df['model_name'] = df['model']
            
            models = sorted(df['model_name'].unique())
            reps = sorted(df['representation'].unique())
            sigmas = sorted(df['sigma'].unique())
            
            print(f"✓ {filepath.name}:")
            print(f"    {len(df):,} rows | models={models} | reps={reps} | σ={sigmas}")
            
            all_data.append(df)
            
        except Exception as e:
            print(f"❌ ERROR: {filepath.name}: {e}")
    
    if not all_data:
        print("\n❌ No valid data loaded!")
        return pd.DataFrame()
    
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
        # Calculate errors and uncertainties
        errors = np.abs(group['y_true_noisy'] - group['y_pred_mean'])
        uncertainties = group['y_pred_std_calibrated']
        
        # Skip if insufficient data
        if uncertainties.isna().all() or len(uncertainties) < 10:
            continue
        
        # Remove NaN values
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
        
        # Coverage (fraction of errors within predicted uncertainty bounds)
        coverage_1std = np.mean(errors <= uncertainties)
        coverage_2std = np.mean(errors <= 2 * uncertainties)
        
        metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std(),
            'mean_absolute_error': errors.mean(),
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


# ============================================================================
# FIGURE 4: UNCERTAINTY-ERROR RELATIONSHIPS
# ============================================================================

def create_figure4(uncertainty_df, metrics_df, output_dir):
    """
    Figure 4: Uncertainty-Error Relationships
    - Panel A: Scatter plot at σ=0.3
    - Panel B: Correlation across noise levels
    - Panel C: Uncertainty inflation
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 4")
    print("="*80)
    
    # Get available representations
    available_reps = sorted(metrics_df['representation'].unique())
    n_rows = len(available_reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    # Create figure with dynamic rows
    fig = plt.figure(figsize=(18, 5*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.30, wspace=0.25,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    for row_idx, rep in enumerate(available_reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        available_models = sorted(rep_metrics['model_name'].unique())
        
        if len(available_models) == 0:
            continue
        
        print(f"  {rep}: {available_models}")
        
        # ====================================================================
        # Panel A: Scatter at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        for model in available_models[:2]:  # First 2 models for clarity
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.05)
            ]
            
            if len(data) < 100:
                continue
            
            # Subsample if too many points
            if len(data) > 3000:
                data = data.sample(3000, random_state=42)
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            if len(errors) < 50:
                continue
            
            color = MODEL_COLORS.get(model, '#999999')
            ax.scatter(uncertainties, errors, s=1, alpha=0.3, 
                      color=color, label=model)
        
        # Perfect calibration line
        if len(ax.collections) > 0:
            all_unc = np.concatenate([col.get_offsets()[:, 0] 
                                     for col in ax.collections])
            all_err = np.concatenate([col.get_offsets()[:, 1] 
                                     for col in ax.collections])
            max_val = max(all_unc.max(), all_err.max())
            ax.plot([0, max_val], [0, max_val], 'r--', 
                   alpha=0.7, linewidth=1.5, label='Perfect')
        
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('|Error|')
        panel_label = chr(65 + row_idx * 3)  # A, D, G, ...
        ax.set_title(f'{panel_label}. {rep}: Unc-Error at σ=0.3', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.2)
        
        # ====================================================================
        # Panel B: Correlation across σ
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in available_models:
            model_data = rep_metrics[
                rep_metrics['model_name'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], model_data['correlation'],
                       marker='o', linewidth=2, markersize=4, alpha=0.9,
                       label=model, color=color)
        
        ax.axhline(0.5, color='green', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Good (>0.5)')
        ax.set_xlabel('Noise level (σ)')
        ax.set_ylabel('Uncertainty-Error Correlation')
        panel_label = chr(66 + row_idx * 3)  # B, E, H, ...
        ax.set_title(f'{panel_label}. {rep}: UQ Quality', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlim(-0.05, 1.05)
        
        # ====================================================================
        # Panel C: Uncertainty inflation
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        for model in available_models:
            model_data = rep_metrics[
                rep_metrics['model_name'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], model_data['mean_uncertainty'],
                       marker='o', linewidth=2, markersize=4, alpha=0.9,
                       label=model, color=color)
        
        # Ideal inflation line
        sigma_range = np.linspace(0, 1.0, 20)
        baseline = rep_metrics[
            rep_metrics['sigma'] == 0.0
        ]['mean_uncertainty'].median()
        
        if not np.isnan(baseline):
            ax.plot(sigma_range, baseline + sigma_range, 'k--', 
                   linewidth=1.5, alpha=0.5, label='Ideal: +σ')
        
        ax.set_xlabel('Noise level (σ)')
        ax.set_ylabel('Mean Uncertainty')
        panel_label = chr(67 + row_idx * 3)  # C, F, I, ...
        ax.set_title(f'{panel_label}. {rep}: Uncertainty Inflation', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
    
    output_path = Path(output_dir) / "figure4_uncertainty_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: CALIBRATION ANALYSIS
# ============================================================================

def create_figure5(uncertainty_df, metrics_df, output_dir):
    """
    Figure 5: Calibration Analysis
    - Panel A: Calibration curves at σ=0.3
    - Panel B: Coverage across noise levels
    - Panel C: ECE heatmap
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 5")
    print("="*80)
    
    available_reps = sorted(metrics_df['representation'].unique())
    n_rows = len(available_reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    fig = plt.figure(figsize=(18, 5*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    for row_idx, rep in enumerate(available_reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        available_models = sorted(rep_metrics['model_name'].unique())
        
        if len(available_models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Calibration curve at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        for model in available_models[:2]:
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.05)
            ]
            
            if len(data) < 100:
                continue
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            # Binned calibration
            n_bins = 10
            bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 1e-8
            
            bin_pred_unc, bin_obs_rmse = [], []
            for i in range(n_bins):
                in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
                if in_bin.sum() > 5:
                    bin_pred_unc.append(uncertainties[in_bin].mean())
                    bin_obs_rmse.append(np.sqrt(np.mean(errors[in_bin]**2)))
            
            if len(bin_pred_unc) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(bin_pred_unc, bin_obs_rmse, 'o-', linewidth=2, 
                       markersize=5, color=color, alpha=0.8, label=model)
        
        # Perfect calibration line
        if len(ax.lines) > 0:
            all_x = np.concatenate([line.get_xdata() for line in ax.lines])
            all_y = np.concatenate([line.get_ydata() for line in ax.lines])
            max_val = max(all_x.max(), all_y.max())
            ax.plot([0, max_val], [0, max_val], 'k--', 
                   linewidth=1.5, alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Observed RMSE')
        panel_label = chr(65 + row_idx * 3)
        ax.set_title(f'{panel_label}. {rep}: Calibration at σ=0.3', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        
        # ====================================================================
        # Panel B: Coverage across σ
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in available_models:
            model_data = rep_metrics[
                rep_metrics['model_name'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], 
                       model_data['coverage_1std'] * 100,
                       marker='o', linewidth=2, markersize=4, alpha=0.8,
                       label=model, color=color)
        
        ax.axhline(68, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.6, label='Target (68%)')
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Coverage at 1σ (%)')
        panel_label = chr(66 + row_idx * 3)
        ax.set_title(f'{panel_label}. {rep}: Coverage', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 100)
        
        # ====================================================================
        # Panel C: ECE heatmap
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        pivot = rep_metrics.pivot_table(
            values='ece',
            index='model_name',
            columns='sigma',
            aggfunc='mean'
        )
        
        if len(pivot) > 0:
            im = ax.imshow(pivot.values, cmap='RdYlGn_r', 
                          aspect='auto', vmin=0, vmax=0.3)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], 
                              fontsize=7, rotation=45)
            ax.set_yticklabels(pivot.index, fontsize=8)
            
            ax.set_xlabel('Noise Level (σ)')
            ax.set_ylabel('Model')
            panel_label = chr(67 + row_idx * 3)
            ax.set_title(f'{panel_label}. {rep}: ECE Heatmap', 
                        fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('ECE', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    
    output_path = Path(output_dir) / "figure5_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 6: EPISTEMIC/ALEATORIC DECOMPOSITION
# ============================================================================

def create_figure6(decomp_df, output_dir):
    """
    Figure 6: Epistemic/Aleatoric Decomposition
    - Panel A: Epistemic vs Aleatoric across noise levels
    - Panel B: Epistemic/Aleatoric ratio
    - Panel C: Stacked bar chart at σ=0.3
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 6")
    print("="*80)
    
    if len(decomp_df) == 0:
        print("⚠️  No decomposition data available")
        return
    
    available_reps = sorted(decomp_df['representation'].unique())
    n_rows = len(available_reps)
    
    fig = plt.figure(figsize=(18, 5*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    for row_idx, rep in enumerate(available_reps):
        rep_decomp = decomp_df[decomp_df['representation'] == rep]
        available_models = sorted(rep_decomp['model_name'].unique())
        
        if len(available_models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Epistemic vs Aleatoric
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        for model in available_models:
            model_data = rep_decomp[
                rep_decomp['model_name'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], model_data['mean_epistemic'],
                       marker='o', linewidth=2, linestyle='-',
                       label=f'{model} (Epi)', color=color, alpha=0.8)
                ax.plot(model_data['sigma'], model_data['mean_aleatoric'],
                       marker='s', linewidth=2, linestyle='--',
                       label=f'{model} (Alea)', color=color, alpha=0.5)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Uncertainty')
        panel_label = chr(65 + row_idx * 3)
        ax.set_title(f'{panel_label}. {rep}: Epistemic vs Aleatoric', 
                    fontweight='bold')
        ax.legend(fontsize=6, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # ====================================================================
        # Panel B: Epistemic/Aleatoric Ratio
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in available_models:
            model_data = rep_decomp[
                rep_decomp['model_name'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], 
                       model_data['epistemic_aleatoric_ratio'],
                       marker='o', linewidth=2, markersize=4,
                       label=model, color=color, alpha=0.8)
        
        ax.axhline(1.0, color='black', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Equal')
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Epistemic / Aleatoric Ratio')
        panel_label = chr(66 + row_idx * 3)
        ax.set_title(f'{panel_label}. {rep}: Uncertainty Ratio', 
                    fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        
        # ====================================================================
        # Panel C: Stacked bar at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        sigma_03 = rep_decomp[np.abs(rep_decomp['sigma'] - 0.3) < 0.05]
        
        if len(sigma_03) > 0:
            models = [m for m in available_models 
                     if m in sigma_03['model_name'].values]
            x = np.arange(len(models))
            
            epist = [sigma_03[sigma_03['model_name'] == m]['mean_epistemic'].mean() 
                    for m in models]
            alea = [sigma_03[sigma_03['model_name'] == m]['mean_aleatoric'].mean() 
                   for m in models]
            
            ax.bar(x, alea, label='Aleatoric', alpha=0.8, color='#3498db')
            ax.bar(x, epist, bottom=alea, label='Epistemic', 
                  alpha=0.8, color='#e74c3c')
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Uncertainty')
            panel_label = chr(67 + row_idx * 3)
            ax.set_title(f'{panel_label}. {rep}: Decomposition at σ=0.3', 
                        fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / "figure6_decomposition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_tables(metrics_df, decomp_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Overall performance
    table1 = metrics_df.groupby(['model_name', 'representation']).agg({
        'correlation': 'mean',
        'mean_uncertainty': 'mean',
        'ece': 'mean',
        'coverage_1std': 'mean',
        'mean_absolute_error': 'mean',
        'n_samples': 'sum'
    }).round(4)
    
    table1.to_csv(output_dir / "table1_model_summary.csv")
    print(f"✓ table1_model_summary.csv")
    
    # Table 2: Performance at key noise levels
    for sigma in [0.0, 0.3, 0.6, 1.0]:
        sigma_data = metrics_df[np.abs(metrics_df['sigma'] - sigma) < 0.05]
        if len(sigma_data) > 0:
            table = sigma_data.groupby(['model_name', 'representation']).agg({
                'correlation': 'mean',
                'ece': 'mean',
                'coverage_1std': 'mean',
                'mean_absolute_error': 'mean'
            }).round(4)
            
            table.to_csv(output_dir / f"table2_sigma{sigma:.1f}.csv")
            print(f"✓ table2_sigma{sigma:.1f}.csv")
    
    # Table 3: Decomposition summary
    if len(decomp_df) > 0:
        table3 = decomp_df.groupby(['model_name', 'representation']).agg({
            'mean_epistemic': 'mean',
            'mean_aleatoric': 'mean',
            'epistemic_aleatoric_ratio': 'mean'
        }).round(4)
        
        table3.to_csv(output_dir / "table3_decomposition.csv")
        print(f"✓ table3_decomposition.csv")


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
        print("\n❌ ERROR: No data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_metrics(uncertainty_df)
    if len(metrics_df) == 0:
        print("\n❌ ERROR: No metrics calculated!")
        return
    
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
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure4(uncertainty_df, metrics_df, output_dir)
    create_figure5(uncertainty_df, metrics_df, output_dir)
    
    if len(decomp_df) > 0:
        create_figure6(decomp_df, output_dir)
    
    # Generate tables
    create_tables(metrics_df, decomp_df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nData: {len(uncertainty_df):,} samples")
    print(f"Configurations: {len(metrics_df)}")
    print(f"Models: {sorted(metrics_df['model_name'].unique())}")
    print(f"Representations: {sorted(metrics_df['representation'].unique())}")
    print(f"Sigma levels: {sorted(metrics_df['sigma'].unique())}")
    
    print("\nCorrelation Statistics:")
    print(metrics_df.groupby('model_name')['correlation'].agg(['mean', 'std']).round(3))
    
    print("\nCoverage Statistics (1σ):")
    print((metrics_df.groupby('model_name')['coverage_1std'].agg(['mean', 'std']) * 100).round(1))
    
    if len(decomp_df) > 0:
        print("\nEpistemic/Aleatoric Ratio:")
        print(decomp_df.groupby('model_name')['epistemic_aleatoric_ratio'].mean().round(3))
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_dir)