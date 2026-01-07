"""
Phase 3 Analysis - UPDATED FOR CALIBRATION GRID
Generates Figure 7, 8, 9, 10 and Supplementary S7

UPDATES:
1. Support for calibration_grid/ directory (NEW primary data source)
2. Support for continuous_pdv and mhg_gnn representations
3. Support for conformal_hetero model
4. Parse calibration size from filenames
5. Proper representation naming (continuous_pdv → pdv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ============================================================================
# JOURNAL OF CHEMINFORMATICS STYLE
# ============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
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

# Color palette
MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#16a085',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'gauche': '#9b59b6',
    'conformal_rf': '#3498db',
    'conformal_qrf': '#16a085',
    'conformal_dnn': '#34495e',
    'conformal_hetero': '#e67e22',
}

REPRESENTATION_COLORS = {
    'ecfp4': '#DE8F05',
    'pdv': '#56B4E9',
    'binary_pdv': '#0173B2',
    'continuous_pdv': '#56B4E9',
    'mhg_gnn': '#CC78BC',
    'sns': '#029E73',
}

# ============================================================================
# DATA LOADING - CALIBRATION GRID VERSION
# ============================================================================

def load_calibration_grid_file(filepath):
    """
    Load a single calibration grid file
    
    Filename format: {rep}_conformal_{base_model}_cal{size}.csv
    Examples:
    - continuous_pdv_conformal_rf_cal10.csv
    - mhggnn_conformal_hetero_cal20.csv
    """
    try:
        df = pd.read_csv(filepath)
        filename = filepath.stem
        
        # Parse filename
        # Pattern: {rep}_conformal_{model}_cal{size}
        # or: {rep}_conformal_{base_model}_cal{size}
        
        parts = filename.split('_')
        
        if 'conformal' not in parts:
            print(f"  ⚠️  {filepath.name}: no 'conformal' in filename")
            return None
        
        conf_idx = parts.index('conformal')
        
        # Extract representation (before 'conformal')
        rep_parts = parts[:conf_idx]
        rep = '_'.join(rep_parts)
        
        # Normalize representation names
        if rep == 'continuous_pdv':
            rep = 'pdv'
        elif rep == 'mhggnn':
            rep = 'mhg_gnn'
        
        # Extract model and calibration size (after 'conformal')
        # Format: conformal_{model}_cal{size} or conformal_hetero_cal{size}
        model_parts = parts[conf_idx + 1:]
        
        # Find calibration size
        calib_size = None
        model_only_parts = []
        for part in model_parts:
            if part.startswith('cal'):
                try:
                    calib_size = int(part.replace('cal', ''))
                except:
                    pass
            else:
                model_only_parts.append(part)
        
        # Model is everything after conformal, before cal
        if model_only_parts:
            model = '_'.join(model_only_parts)
        else:
            model = 'unknown'
        
        if not rep or not model:
            print(f"  ⚠️  {filepath.name}: couldn't parse rep/model")
            return None
        
        # Add metadata
        if 'model' not in df.columns:
            df['model'] = model
        if 'base_model' not in df.columns:
            df['base_model'] = model
        if 'model_name' not in df.columns:
            df['model_name'] = f'conformal_{model}'
        if 'representation' not in df.columns:
            df['representation'] = rep
        if 'rep' not in df.columns:
            df['rep'] = rep
        if calib_size is not None and 'calibration_size' not in df.columns:
            df['calibration_size'] = calib_size
        
        # Ensure sigma column
        if 'sigma' not in df.columns and 'sigma_noise' not in df.columns:
            # Try to parse from filename
            import re
            sigma_match = re.search(r'sigma([0-9.]+)', filename)
            if sigma_match:
                df['sigma'] = float(sigma_match.group(1))
            else:
                df['sigma'] = 0.0
        
        # Standardize sigma naming
        if 'sigma_noise' in df.columns and 'sigma' not in df.columns:
            df['sigma'] = df['sigma_noise']
        elif 'sigma' in df.columns and 'sigma_noise' not in df.columns:
            df['sigma_noise'] = df['sigma']
        
        # Rename 'iteration' to match expected format
        if 'iteration' in df.columns and 'iter' not in df.columns:
            df['iter'] = df['iteration']
        
        # Check for essential columns
        if 'alpha' not in df.columns:
            print(f"  ⚠️  {filepath.name}: missing 'alpha' column")
            return None
        
        # Ensure y_true
        if 'y_true' not in df.columns:
            if 'y_true_noisy' in df.columns:
                df['y_true'] = df['y_true_noisy']
            elif 'y_true_original' in df.columns:
                df['y_true'] = df['y_true_original']
            else:
                print(f"  ⚠️  {filepath.name}: no y_true column")
                return None
        
        # Calculate coverage if missing
        if 'coverage' not in df.columns:
            if all(col in df.columns for col in ['y_true', 'lower_bound', 'upper_bound']):
                df['coverage'] = ((df['y_true'] >= df['lower_bound']) & 
                                 (df['y_true'] <= df['upper_bound'])).astype(int)
            elif all(col in df.columns for col in ['y_true', 'lower', 'upper']):
                df['coverage'] = ((df['y_true'] >= df['lower']) & 
                                 (df['y_true'] <= df['upper'])).astype(int)
            else:
                print(f"  ⚠️  {filepath.name}: cannot calculate coverage (missing bounds)")
                return None
        
        # Calculate interval_width if missing
        if 'interval_width' not in df.columns:
            if all(col in df.columns for col in ['lower_bound', 'upper_bound']):
                df['interval_width'] = df['upper_bound'] - df['lower_bound']
            elif all(col in df.columns for col in ['lower', 'upper']):
                df['interval_width'] = df['upper'] - df['lower']
            else:
                print(f"  ⚠️  {filepath.name}: cannot calculate interval_width")
                return None
        
        # Ensure y_pred
        if 'y_pred' not in df.columns:
            if 'y_pred_mean' in df.columns:
                df['y_pred'] = df['y_pred_mean']
            elif 'prediction' in df.columns:
                df['y_pred'] = df['prediction']
        
        df['source_file'] = filepath.name
        df['source_type'] = 'calibration_grid'
        
        # Report successful load
        n_alphas = len(df['alpha'].unique())
        n_sigmas = len(df['sigma'].unique())
        n_iters = len(df.get('iter', df.get('iteration', [0])).unique())
        
        print(f"  ✓ {filepath.name}: {len(df):,} rows")
        print(f"      model={model}, rep={rep}, calib_size={calib_size}")
        print(f"      {n_alphas} alphas × {n_sigmas} sigmas × {n_iters} iterations")
        
        return df
        
    except Exception as e:
        print(f"  ❌ {filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_conformal_intervals(results_dir="../results"):
    """
    Load Phase 3 conformal prediction results - UPDATED FOR CALIBRATION GRID
    
    Handles THREE types of data structures:
    1. CALIBRATION GRID FILES: calibration_grid/*.csv (NEW - primary source)
    2. SUMMARY FILES: phase3a/b/c_*.csv with per-row predictions and metadata
    3. RAW INTERVAL FILES: conformal_intervals/*.csv with granular predictions
    
    Strategy:
    - Try calibration_grid first (new data location)
    - Fall back to summary files (old data)
    - Fall back to raw interval files if needed
    """
    print("\n" + "="*80)
    print("LOADING PHASE 3 CONFORMAL PREDICTION DATA - UPDATED VERSION")
    print("="*80)
    
    results_dir = Path(results_dir)
    print(f"Searching in: {results_dir.absolute()}\n")
    
    all_data = []
    
    # ========================================================================
    # STEP 1: Try to load CALIBRATION GRID files (NEW)
    # ========================================================================
    print("-"*80)
    print("STEP 1: Looking for calibration grid files (calibration_grid/*.csv)")
    print("-"*80)
    
    calib_dir = results_dir / "calibration_grid"
    
    if calib_dir.exists():
        calib_files = list(calib_dir.glob("*.csv"))
        # Filter out per_epoch and uncertainty files
        calib_files = [f for f in calib_files if 'per_epoch' not in f.name and 'uncertainty' not in f.name]
        
        print(f"Found {len(calib_files)} calibration grid files\n")
        
        if calib_files:
            print("Loading calibration grid files...")
            for filepath in sorted(calib_files):
                loaded_df = load_calibration_grid_file(filepath)
                if loaded_df is not None and len(loaded_df) > 0:
                    all_data.append(loaded_df)
    else:
        print("❌ No calibration_grid directory found")
    
    # ========================================================================
    # STEP 2: Try to load SUMMARY files (phase3a/b/c_*.csv) - OLD DATA
    # ========================================================================
    if not all_data:
        print("\n" + "-"*80)
        print("STEP 2: No calibration grid, looking for summary files (phase3a/b/c/*.csv)")
        print("-"*80)
        
        summary_patterns = [
            "phase3a_*_conformal_*.csv",
            "phase3b_*_conformal_*.csv",
            "phase3c_*_conformal_*.csv",
            "phase3d_*_conformal_*.csv",
            "phase3e_*_conformal_*.csv",
            "phase3f_*_conformal_*.csv",
            "phase3_*_conformal_*.csv",
        ]
        
        summary_files = []
        for pattern in summary_patterns:
            found = list(results_dir.glob(pattern))
            summary_files.extend(found)
        
        summary_files = list(set(summary_files))
        summary_files = [f for f in summary_files if 'uncertainty_values' not in f.name]
        summary_files = sorted(summary_files)
        
        print(f"Found {len(summary_files)} summary files")
        
        if summary_files:
            # Load using old loader (would need to implement separately)
            print("⚠️  Old summary file format - may need custom loader")
    
    # ========================================================================
    # STEP 3: Combine and validate
    # ========================================================================
    if not all_data:
        print("\n❌ ERROR: No conformal data could be loaded!")
        print("\nSearched for:")
        print("  1. Calibration grid: calibration_grid/*.csv")
        print("  2. Summary files: phase3a/b/c_*_conformal_*.csv")
        return pd.DataFrame()
    
    intervals_df = pd.concat(all_data, ignore_index=True)
    
    # Final data cleaning and standardization
    intervals_df = standardize_columns(intervals_df)
    
    # Print comprehensive summary
    print_data_summary(intervals_df)
    
    return intervals_df


def standardize_columns(df):
    """Standardize column names across different data sources"""
    # Model names
    if 'model_name' not in df.columns and 'model' in df.columns:
        df['model_name'] = 'conformal_' + df['model'].astype(str)
    
    if 'base_model' not in df.columns and 'model' in df.columns:
        df['base_model'] = df['model'].str.replace('conformal_', '', regex=False)
    
    # Representation
    if 'rep' not in df.columns and 'representation' in df.columns:
        df['rep'] = df['representation']
    elif 'representation' not in df.columns and 'rep' in df.columns:
        df['representation'] = df['rep']
    
    # Normalize representation names
    if 'representation' in df.columns:
        df['representation'] = df['representation'].replace({
            'continuous_pdv': 'pdv',
            'mhggnn': 'mhg_gnn'
        })
    if 'rep' in df.columns:
        df['rep'] = df['rep'].replace({
            'continuous_pdv': 'pdv',
            'mhggnn': 'mhg_gnn'
        })
    
    # Sigma
    if 'sigma_noise' not in df.columns and 'sigma' in df.columns:
        df['sigma_noise'] = df['sigma']
    elif 'sigma' not in df.columns and 'sigma_noise' in df.columns:
        df['sigma'] = df['sigma_noise']
    
    return df


def print_data_summary(df):
    """Print comprehensive data summary"""
    print(f"\n{'='*80}")
    print("✅ SUCCESSFULLY LOADED CONFORMAL DATA")
    print(f"{'='*80}")
    print(f"\nTotal predictions: {len(df):,}")
    
    if 'source_type' in df.columns:
        print(f"Source types: {df['source_type'].value_counts().to_dict()}")
    
    print(f"\n📊 Data Summary:")
    print(f"  Models ({len(df['base_model'].unique())}): {sorted(df['base_model'].unique())}")
    print(f"  Representations ({len(df['rep'].unique())}): {sorted(df['rep'].unique())}")
    print(f"  Alpha values ({len(df['alpha'].unique())}): {sorted(df['alpha'].unique())}")
    print(f"  Sigma values ({len(df['sigma_noise'].unique())}): {sorted(df['sigma_noise'].unique())}")
    
    if 'calibration_size' in df.columns:
        calib_sizes = sorted(df['calibration_size'].dropna().unique())
        print(f"  Calibration sizes ({len(calib_sizes)}): {calib_sizes}")
    
    print(f"\n📈 Data Completeness by Model/Representation:")
    print(f"  {'Model':<20s} {'Rep':<10s} {'Rows':>10s} {'Alphas':>8s} {'Sigmas':>8s} {'Cal Sizes':>10s}")
    print(f"  {'-'*80}")
    
    for model in sorted(df['base_model'].unique()):
        for rep in sorted(df['rep'].unique()):
            subset = df[(df['base_model'] == model) & (df['rep'] == rep)]
            if len(subset) > 0:
                n_alphas = len(subset['alpha'].unique())
                n_sigmas = len(subset['sigma_noise'].unique())
                if 'calibration_size' in subset.columns:
                    calib_info = str(sorted(subset['calibration_size'].dropna().unique()))
                else:
                    calib_info = 'N/A'
                print(f"  {model:<20s} {rep:<10s} {len(subset):>10,} {n_alphas:>8} {n_sigmas:>8} {calib_info:>10s}")


# Continue with calculate_conformal_metrics and all figure generation functions from original...
# (The rest of the script remains the same as doc 2)

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_conformal_metrics(intervals_df):
    """
    Calculate conformal prediction metrics
    
    Metrics:
    - Expected coverage: 1 - alpha
    - Empirical coverage: actual fraction within intervals
    - Coverage deviation: empirical - expected
    - Mean/median interval width
    - Efficiency: width / RMSE ratio
    - Calibration status
    """
    print("\n" + "="*80)
    print("CALCULATING CONFORMAL METRICS")
    print("="*80)
    
    metrics = []
    
    groupby_cols = ['model_name', 'rep', 'sigma_noise', 'alpha']
    
    # Add calibration_size if it exists
    if 'calibration_size' in intervals_df.columns:
        groupby_cols.append('calibration_size')
    
    # Check which columns exist
    existing_cols = [col for col in groupby_cols if col in intervals_df.columns]
    
    if len(existing_cols) < 4:
        print(f"⚠️  Warning: Missing groupby columns. Have: {existing_cols}")
        # Try alternate column names
        if 'base_model' in intervals_df.columns and 'model_name' not in intervals_df.columns:
            groupby_cols[0] = 'base_model'
        if 'representation' in intervals_df.columns and 'rep' not in intervals_df.columns:
            groupby_cols[1] = 'representation'
        if 'sigma' in intervals_df.columns and 'sigma_noise' not in intervals_df.columns:
            groupby_cols[2] = 'sigma'
    
    for group_keys, group in intervals_df.groupby(groupby_cols):
        if len(groupby_cols) == 4:
            model, rep, sigma, alpha = group_keys
            calib_size = None
        elif len(groupby_cols) == 5:
            model, rep, sigma, alpha, calib_size = group_keys
        else:
            continue
        
        # Expected coverage
        expected_coverage = 1 - alpha
        
        # Empirical coverage
        empirical_coverage = group['coverage'].mean()
        
        # Coverage deviation
        coverage_deviation = empirical_coverage - expected_coverage
        
        # Interval width stats
        mean_width = group['interval_width'].mean()
        median_width = group['interval_width'].median()
        std_width = group['interval_width'].std()
        
        # Prediction error
        if 'y_pred' in group.columns and 'y_true' in group.columns:
            rmse = np.sqrt(((group['y_true'] - group['y_pred'])**2).mean())
            mae = np.abs(group['y_true'] - group['y_pred']).mean()
        else:
            rmse = np.nan
            mae = np.nan
        
        # Efficiency: narrower intervals relative to error are better
        efficiency = mean_width / rmse if rmse > 0 and not np.isnan(rmse) else np.inf
        
        # Calibration check (within 5%)
        is_calibrated = abs(coverage_deviation) < 0.05
        
        metric_dict = {
            'model_name': model,
            'base_model': model.replace('conformal_', ''),
            'representation': rep,
            'sigma': sigma,
            'alpha': alpha,
            'expected_coverage': expected_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_deviation': coverage_deviation,
            'mean_width': mean_width,
            'median_width': median_width,
            'std_width': std_width,
            'rmse': rmse,
            'mae': mae,
            'efficiency': efficiency,
            'is_calibrated': is_calibrated,
            'n_samples': len(group)
        }
        
        if calib_size is not None:
            metric_dict['calibration_size'] = calib_size
        
        metrics.append(metric_dict)
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"\n✓ Calculated metrics for {len(metrics_df)} configurations")
    
    # Summary stats
    if len(metrics_df) > 0:
        print(f"\nCalibration summary:")
        calibrated_count = metrics_df['is_calibrated'].sum()
        total_count = len(metrics_df)
        print(f"  Well-calibrated (|deviation| < 0.05): {calibrated_count}/{total_count} ({calibrated_count/total_count*100:.1f}%)")
        
        print(f"\nCoverage statistics:")
        print(f"  Mean coverage deviation: {metrics_df['coverage_deviation'].abs().mean():.4f}")
        print(f"  Median coverage deviation: {metrics_df['coverage_deviation'].abs().median():.4f}")
        
        print(f"\nEfficiency statistics:")
        finite_eff = metrics_df[np.isfinite(metrics_df['efficiency'])]['efficiency']
        if len(finite_eff) > 0:
            print(f"  Mean efficiency: {finite_eff.mean():.2f}")
            print(f"  Median efficiency: {finite_eff.median():.2f}")
    
    return metrics_df


# ============================================================================
# FIGURE 7: CONFORMAL PREDICTION VALIDITY AND EFFICIENCY
# ============================================================================

def create_figure7_conformal_validity_efficiency(intervals_df, metrics_df, output_dir):
    """
    Figure 7: Conformal prediction validity (SINGLE PANEL)
    
    Panel A: Coverage calibration vs target confidence and noise
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 7: CONFORMAL VALIDITY")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Select key models and sigma levels
    sigma_levels = sorted(metrics_df['sigma'].unique())[:3]
    
    available_models = metrics_df['base_model'].unique()
    key_models = [m for m in ['rf', 'qrf', 'dnn', 'gauche', 'hetero'] if m in available_models][:3]
    
    if len(key_models) == 0 or len(sigma_levels) == 0:
        ax.text(0.5, 0.5, 'Insufficient conformal prediction data',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    else:
        plotted_something = False
        
        for model in key_models:
            for sigma_idx, sigma in enumerate(sigma_levels):
                model_sigma = metrics_df[
                    (metrics_df['base_model'] == model) &
                    (metrics_df['sigma'] == sigma)
                ].sort_values('alpha', ascending=False)
                
                if len(model_sigma) == 0:
                    continue
                
                color = MODEL_COLORS.get(model, MODEL_COLORS.get(f'conformal_{model}', '#999999'))
                linestyle = ['-', '--', ':'][sigma_idx % 3]
                
                ax.plot(model_sigma['expected_coverage'], 
                       model_sigma['empirical_coverage'],
                       marker='o', linewidth=2, markersize=6, alpha=0.9,
                       color=color, linestyle=linestyle,
                       label=f'{model}, σ={sigma:.1f}')
                
                plotted_something = True
        
        if plotted_something:
            # Perfect calibration line
            ax.plot([0.8, 1.0], [0.8, 1.0], 'k--', linewidth=2, alpha=0.7,
                   label='Perfect calibration')
            
            # Acceptable range (±5%)
            ax.fill_between([0.8, 1.0], [0.75, 0.95], [0.85, 1.05],
                          color='green', alpha=0.15, label='±5% range')
            
            ax.set_xlabel('Target Coverage (1-α)', fontsize=9)
            ax.set_ylabel('Empirical Coverage', fontsize=9)
            ax.set_title('Conformal Prediction: Coverage Calibration', 
                       fontsize=10, fontweight='bold', pad=10)
            ax.legend(fontsize=7, loc='lower right', ncol=1, frameon=True, framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_xlim(0.78, 1.02)
            ax.set_ylim(0.78, 1.02)
        else:
            ax.text(0.5, 0.5, 'No plottable data found',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure7_conformal_validity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 7 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S7
# ============================================================================

def create_supplementary_s7(intervals_df, metrics_df, output_dir):
    """
    Supplementary S7: Per-target conformal performance
    (Only if multiple targets available)
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S7: PER-TARGET CONFORMAL")
    print("="*80)
    
    if 'target' not in intervals_df.columns:
        print("⚠️  No 'target' column found. Skipping S7.")
        return
    
    targets = intervals_df['target'].unique()
    
    if len(targets) <= 1:
        print(f"⚠️  Only one target found: {targets}. Skipping S7.")
        return
    
    print(f"Found {len(targets)} targets: {targets}")
    
    # Create grid
    n_targets = len(targets)
    ncols = min(3, n_targets)
    nrows = (n_targets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    alpha_target = 0.05
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        if 'target' in metrics_df.columns:
            target_metrics = metrics_df[
                (metrics_df['target'] == target) &
                (metrics_df['alpha'] == alpha_target)
            ]
        else:
            target_metrics = pd.DataFrame()
        
        if len(target_metrics) == 0:
            ax.text(0.5, 0.5, f'No data for {target}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target}', fontsize=9, fontweight='bold')
            continue
        
        for model in target_metrics['base_model'].unique():
            model_data = target_metrics[
                target_metrics['base_model'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], model_data['median_width'],
                       marker='o', linewidth=2, label=model, color=color, alpha=0.8)
        
        ax.set_xlabel('σ', fontsize=8)
        ax.set_ylabel('Median Width', fontsize=8)
        ax.set_title(f'{target}', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused
    for idx in range(len(targets), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "supplementary_s7_per_target_conformal.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S7 to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables for Phase 3"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    if len(metrics_df) == 0:
        print("⚠️  No metrics to create tables from")
        return
    
    # Table 4: Simplified summary
    print("\nCreating Table 4: Simplified Conformal Summary")
    
    alpha_target = 0.05
    table4_data = []
    
    for (model, rep), group in metrics_df[metrics_df['alpha'] == alpha_target].groupby(
        ['base_model', 'representation']
    ):
        sigma_0 = group[group['sigma'] == 0.0]
        sigma_mid = group[group['sigma'] > 0.2]
        
        if len(sigma_0) > 0 and len(sigma_mid) > 0:
            sigma_mid_val = sigma_mid.iloc[0]['sigma']
            
            row = {
                'Model': model,
                'Representation': rep,
                'Target Coverage': '95%',
                'Coverage @ σ=0': f"{sigma_0.iloc[0]['empirical_coverage']:.3f}",
                f'Coverage @ σ={sigma_mid_val:.1f}': f"{sigma_mid.iloc[0]['empirical_coverage']:.3f}",
                'Width @ σ=0': f"{sigma_0.iloc[0]['median_width']:.3f}",
                f'Width @ σ={sigma_mid_val:.1f}': f"{sigma_mid.iloc[0]['median_width']:.3f}",
                'Well-Calibrated': '✓' if sigma_mid.iloc[0]['is_calibrated'] else '✗'
            }
            table4_data.append(row)
    
    if table4_data:
        table4 = pd.DataFrame(table4_data)
        table4.to_csv(output_dir / "table4_conformal_summary_simplified.csv", index=False)
        print(f"✓ Saved Table 4")
    
    # Detailed tables
    alpha_key = [0.01, 0.05, 0.1, 0.2]
    
    table1 = metrics_df[metrics_df['alpha'].isin(alpha_key)].groupby(
        ['base_model', 'representation', 'alpha']
    ).agg({
        'expected_coverage': 'mean',
        'empirical_coverage': 'mean',
        'coverage_deviation': 'mean',
        'median_width': 'mean',
        'efficiency': 'mean',
        'is_calibrated': 'mean'
    }).reset_index()
    
    table1 = table1.round(4)
    table1.to_csv(output_dir / "table_phase3_conformal_summary.csv", index=False)
    print(f"✓ Saved detailed conformal summary")


# ============================================================================
# FIGURE 8: NOISE ROBUSTNESS ANALYSIS 
# ============================================================================

def create_figure8_noise_robustness(metrics_df, output_dir):
    """
    Figure 8: How CP quality degrades with noise
    2x2: Width inflation (ECFP4/PDV), Coverage stability heatmap, Efficiency
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8: NOISE ROBUSTNESS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    alpha_focus = 0.1
    
    # Get available representations
    available_reps = sorted(metrics_df['representation'].unique())
    rep_a = available_reps[0] if len(available_reps) > 0 else 'ecfp4'
    rep_b = available_reps[1] if len(available_reps) > 1 else 'pdv'
    
    # Panel A: Width inflation - Rep A
    ax = axes[0, 0]
    rep = rep_a
    rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
    
    for model in rep_models:
        model_data = metrics_df[
            (metrics_df['base_model'] == model) &
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ].sort_values('sigma')
        
        if len(model_data) > 2:
            baseline = model_data[model_data['sigma'] == 0.0]['median_width']
            if len(baseline) > 0:
                relative_width = (model_data['median_width'] / baseline.iloc[0] - 1) * 100
                color = MODEL_COLORS.get(model, MODEL_COLORS.get(f'conformal_{model}', '#999999'))
                ax.plot(model_data['sigma'], relative_width,
                       marker='o', linewidth=2.5, markersize=6,
                       color=color, label=model, alpha=0.85)
    
    sigma_range = np.linspace(0, metrics_df['sigma'].max(), 50)
    ax.plot(sigma_range, sigma_range * 50, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Increase (%)', fontsize=10)
    ax.set_title(f'A. Interval Width Inflation ({rep.upper()})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel B: Width inflation - Rep B
    ax = axes[0, 1]
    rep = rep_b
    rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
    
    for model in rep_models:
        model_data = metrics_df[
            (metrics_df['base_model'] == model) &
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ].sort_values('sigma')
        
        if len(model_data) > 2:
            baseline = model_data[model_data['sigma'] == 0.0]['median_width']
            if len(baseline) > 0:
                relative_width = (model_data['median_width'] / baseline.iloc[0] - 1) * 100
                color = MODEL_COLORS.get(model, MODEL_COLORS.get(f'conformal_{model}', '#999999'))
                ax.plot(model_data['sigma'], relative_width,
                       marker='s', linewidth=2.5, markersize=6,
                       color=color, label=model, alpha=0.85)
    
    ax.plot(sigma_range, sigma_range * 50, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Increase (%)', fontsize=10)
    ax.set_title(f'B. Interval Width Inflation ({rep.upper()})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel C: Coverage stability heatmap
    ax = axes[1, 0]
    plot_data = metrics_df[metrics_df['alpha'] == alpha_focus].copy()
    plot_data['abs_deviation'] = plot_data['coverage_deviation'].abs()
    
    pivot = plot_data.pivot_table(
        values='abs_deviation',
        index=['base_model', 'representation'],
        columns='sigma',
        aggfunc='mean'
    )
    
    if len(pivot) > 0:
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=0.10, interpolation='nearest')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], fontsize=8)
        ax.set_yticklabels([f'{idx[0]}\n{idx[1]}' for idx in pivot.index], fontsize=7)
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color='white' if val > 0.05 else 'black', fontsize=6, fontweight='bold')
        
        ax.set_xlabel('Noise Level (σ)', fontsize=10)
        ax.set_ylabel('Model / Representation', fontsize=10)
        ax.set_title('C. Coverage Calibration Error', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|Coverage Deviation|', fontsize=9)
    
    # Panel D: Efficiency vs noise
    ax = axes[1, 1]
    for rep in available_reps[:2]:
        rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
        for model in rep_models:
            model_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ].sort_values('sigma')
            model_data = model_data[np.isfinite(model_data['efficiency'])]
            
            if len(model_data) > 2:
                color = MODEL_COLORS.get(model, MODEL_COLORS.get(f'conformal_{model}', '#999999'))
                marker = 'o' if rep == available_reps[0] else 's'
                linestyle = '-' if rep == available_reps[0] else '--'
                ax.plot(model_data['sigma'], model_data['efficiency'],
                       marker=marker, linewidth=2, markersize=5,
                       color=color, linestyle=linestyle, alpha=0.75,
                       label=f'{model}/{rep}')
    
    ax.axhline(3.0, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Efficiency (Width/RMSE)', fontsize=10)
    ax.set_title('D. Efficiency Across Noise', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure8_noise_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 8 to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 9: DETAILED CALIBRATION CURVES
# ============================================================================

def create_figure9_detailed_calibration(metrics_df, output_dir):
    """Figure 9: Per-model calibration curves at multiple noise levels"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 9: DETAILED CALIBRATION CURVES")
    print("="*80)
    
    # First, identify which model/rep combinations have data
    available_combos = []
    models = sorted(metrics_df['base_model'].unique())
    reps = sorted(metrics_df['representation'].unique())[:2]
    
    for model in models:
        for rep in reps:
            # Check if this combo has data at multiple sigmas
            combo_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep)
            ]
            if len(combo_data) > 3:
                available_combos.append((model, rep))
    
    if len(available_combos) == 0:
        print("⚠️  No data for Figure 9 (need model/rep with multiple noise levels)")
        return
    
    print(f"Found data for {len(available_combos)} model/rep combinations")
    
    # Create grid based on available combos
    ncols = 2
    nrows = len([m for m, r in available_combos if r == reps[0]])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    sigma_levels = [0.0, 0.3, 0.6, 0.9]
    colors_sigma = ['#2c3e50', '#3498db', '#e74c3c', '#9b59b6']
    
    row_idx = 0
    models_with_data = sorted(list(set([m for m, r in available_combos])))
    
    for model in models_with_data:
        for col_idx, rep in enumerate(reps):
            if (model, rep) in available_combos:
                ax = axes[row_idx, col_idx]
                plotted = False
                
                for sigma_idx, sigma in enumerate(sigma_levels):
                    data = metrics_df[
                        (metrics_df['base_model'] == model) &
                        (metrics_df['representation'] == rep) &
                        (np.abs(metrics_df['sigma'] - sigma) < 0.01)
                    ].sort_values('alpha', ascending=False)
                    
                    if len(data) > 1:
                        ax.plot(data['expected_coverage'], data['empirical_coverage'],
                               marker='o', linewidth=2.5, markersize=7,
                               color=colors_sigma[sigma_idx], alpha=0.8,
                               label=f'σ={sigma:.1f}')
                        plotted = True
                
                if plotted:
                    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', linewidth=2, alpha=0.6, label='Perfect')
                    ax.fill_between([0.7, 1.0], [0.65, 0.95], [0.75, 1.05],
                                   color='green', alpha=0.12, label='±5%')
                    ax.set_xlabel('Target Coverage (1-α)', fontsize=10)
                    ax.set_ylabel('Empirical Coverage', fontsize=10)
                    ax.set_title(f'{model.upper()} / {rep.upper()}', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='lower right', ncol=2)
                    ax.grid(alpha=0.25, linestyle=':')
                    ax.set_xlim(0.69, 1.01)
                    ax.set_ylim(0.69, 1.01)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            else:
                ax = axes[row_idx, col_idx]
                ax.axis('off')
        
        row_idx += 1
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure9_detailed_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 9 to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 10: CP FOR NOISE DETECTION
# ============================================================================

def create_figure10_noise_detection(metrics_df, output_dir):
    """Figure 10: Can CP intervals detect/quantify noise? Width-noise correlation"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 10: NOISE DETECTION POTENTIAL")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    alpha_focus = 0.1
    
    available_reps = sorted(metrics_df['representation'].unique())[:2]
    
    # Panels A & B: Width vs RMSE colored by noise
    for rep_idx, rep in enumerate(available_reps):
        ax = axes[0, rep_idx]
        rep_data = metrics_df[
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ]
        
        scatter = ax.scatter(rep_data['rmse'], rep_data['median_width'],
                           c=rep_data['sigma'], cmap='viridis',
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        rmse_range = np.linspace(rep_data['rmse'].min(), rep_data['rmse'].max(), 100)
        for eff in [2, 3, 4]:
            ax.plot(rmse_range, rmse_range * eff, '--', alpha=0.25, linewidth=1, color='gray')
        
        ax.set_xlabel('RMSE', fontsize=10)
        ax.set_ylabel('Median Interval Width', fontsize=10)
        ax.set_title(f'{"A" if rep_idx == 0 else "B"}. Width-Error ({rep.upper()})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Noise (σ)', fontsize=9)
    
    # Panel C: Width-noise correlation
    ax = axes[1, 0]
    correlations = []
    for model in metrics_df['base_model'].unique():
        for rep in metrics_df['representation'].unique():
            model_rep = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ]
            if len(model_rep) > 3:
                corr = model_rep[['sigma', 'median_width']].corr().iloc[0, 1]
                correlations.append({'model': model, 'rep': rep, 'correlation': corr, 'label': f'{model}/{rep}'})
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('correlation')
        colors_list = [MODEL_COLORS.get(row['model'], MODEL_COLORS.get(f"conformal_{row['model']}", '#999999')) for _, row in corr_df.iterrows()]
        y_pos = np.arange(len(corr_df))
        ax.barh(y_pos, corr_df['correlation'], color=colors_list, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(corr_df['label'], fontsize=8)
        ax.set_xlabel('Correlation (σ vs Width)', fontsize=10)
        ax.set_title('C. CP Width Tracks Noise', fontsize=11, fontweight='bold')
        ax.axvline(0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Strong')
        ax.grid(alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    
    # Panel D: Width ratio vs true noise
    ax = axes[1, 1]
    for rep in available_reps:
        for model in metrics_df[metrics_df['representation'] == rep]['base_model'].unique()[:3]:
            model_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ].sort_values('sigma')
            
            if len(model_data) > 2:
                baseline = model_data[model_data['sigma'] == 0.0]['median_width']
                if len(baseline) > 0:
                    width_ratio = model_data['median_width'] / baseline.iloc[0]
                    color = MODEL_COLORS.get(model, MODEL_COLORS.get(f'conformal_{model}', '#999999'))
                    marker = 'o' if rep == available_reps[0] else 's'
                    ax.plot(model_data['sigma'], width_ratio,
                           marker=marker, linewidth=2, markersize=6,
                           color=color, alpha=0.75, label=f'{model}/{rep}')
    
    sigma_range = np.linspace(0, metrics_df['sigma'].max(), 50)
    ax.plot(sigma_range, 1 + sigma_range * 0.5, 'k--', linewidth=2, alpha=0.5, label='Ideal')
    ax.set_xlabel('True Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Ratio (vs σ=0)', fontsize=10)
    ax.set_title('D. Noise Detection via CP Width', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure10_noise_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 10 to {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 3 ANALYSIS - UPDATED FOR CALIBRATION GRID")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    # Load data
    intervals_df = load_conformal_intervals(results_dir)
    
    if len(intervals_df) == 0:
        print("\n❌ ERROR: No conformal data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_conformal_metrics(intervals_df)
    
    if len(metrics_df) == 0:
        print("\n❌ ERROR: No metrics calculated!")
        return
    
    # Create output directory
    output_dir = Path(results_dir) / "phase3_figures_updated"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase3_conformal_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase3_conformal_metrics.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure7_conformal_validity_efficiency(intervals_df, metrics_df, output_dir)
    create_figure8_noise_robustness(metrics_df, output_dir)
    create_figure9_detailed_calibration(metrics_df, output_dir)
    create_figure10_noise_detection(metrics_df, output_dir)
    create_supplementary_s7(intervals_df, metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 3 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  FIGURES:")
    print("    - figure7_conformal_validity.png")
    print("    - figure8_noise_robustness.png (width inflation, coverage stability)")
    print("    - figure9_detailed_calibration.png (per-model calibration curves)")
    print("    - figure10_noise_detection.png (CP for noise quantification)")
    print("  TABLES:")
    print("    - table4_conformal_summary_simplified.csv")
    print("    - table_phase3_conformal_summary.csv")
    print("  DATA:")
    print("    - phase3_conformal_metrics.csv")
    
    # Stats
    if len(metrics_df) > 0:
        total = len(metrics_df)
        calibrated = metrics_df['is_calibrated'].sum()
        
        print(f"\nCalibration: {calibrated}/{total} configs well-calibrated ({calibrated/total*100:.1f}%)")
        print(f"Mean coverage deviation: {metrics_df['coverage_deviation'].abs().mean():.4f}")
        
        finite_eff = metrics_df[np.isfinite(metrics_df['efficiency'])]['efficiency']
        if len(finite_eff) > 0:
            print(f"Mean efficiency: {finite_eff.mean():.2f}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)