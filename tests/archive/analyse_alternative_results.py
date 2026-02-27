"""
Pharmaceutical Dataset Noise Robustness Analysis
=================================================

Generates figures for COX-2 IC50 and DRD2 Ki noise robustness experiments.

Input: results/cox2/all_results.csv, results/drd2/all_results.csv
       results/cox2/summary.csv, results/drd2/summary.csv

Key metric: Noise Degradation Slope (NDS)
  - NDS_r2: slope of R² vs σ (negative = degrades with noise)
  - Thresholded: only reported for configs with baseline R² above threshold
  - Focus on legacy (Gaussian) strategy as primary, alternatives for comparison

Figures:
  1. Per-dataset degradation curves by strategy (COX-2, DRD2)
  2. Per-dataset NDS heatmaps (model × representation, legacy strategy)
  3. Cross-dataset comparison
  4. Strategy concordance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, linregress
import warnings
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL OF CHEMINFORMATICS STYLE
# ============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
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

# ============================================================================
# DISPLAY LABELS AND COLOURS
# ============================================================================

STRATEGY_LABELS = {
    'legacy': 'Legacy (Gaussian)',
    'outlier': 'Outlier',
    'quantile': 'Quantile',
    'hetero': 'Heteroscedastic',
    'threshold': 'Threshold',
    'valprop': 'Value-Proportional',
}

STRATEGY_COLOURS = {
    'legacy': '#2ecc71',
    'outlier': '#e67e22',
    'quantile': '#f39c12',
    'hetero': '#e74c3c',
    'threshold': '#7f8c8d',
    'valprop': '#9b59b6',
}

# Consistent ordering for plots
STRATEGY_ORDER = ['legacy', 'hetero', 'quantile', 'outlier', 'threshold', 'valprop']

MODEL_LABELS = {
    'RF': 'RF',
    'QRF': 'QRF',
    'XGBoost': 'XGBoost',
    'GP': 'GP',
    'DNN': 'DNN',
    'Full-BNN': 'Full-BNN',
    'LastLayer-BNN': 'LL-BNN',
    'Var-BNN': 'Var-BNN',
}

MODEL_COLOURS = {
    'RF': '#3498db',
    'QRF': '#16a085',
    'XGBoost': '#e74c3c',
    'GP': '#f39c12',
    'DNN': '#34495e',
    'Full-BNN': '#8e44ad',
    'LastLayer-BNN': '#2980b9',
    'Var-BNN': '#c0392b',
}

REP_LABELS = {
    'ECFP4': 'ECFP4',
    'PDV': 'PDV',
    'SNS': 'SNS',
    'MHG-GNN-pretrained': 'MHG-GNN',
}

DATASET_LABELS = {
    'cox2': 'COX-2 (IC50)',
    'drd2': 'DRD2 (Ki)',
}

DATASET_COLOURS = {
    'cox2': '#e74c3c',
    'drd2': '#3498db',
}


def fmt(value, mapping):
    """Get display label, fallback to value itself."""
    return mapping.get(value, value)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset_results(results_dir, dataset_name):
    """Load all_results.csv and summary.csv for a dataset."""
    rdir = Path(results_dir) / dataset_name
    
    all_results_path = rdir / 'all_results.csv'
    summary_path = rdir / 'summary.csv'
    
    if not all_results_path.exists():
        print(f"  WARNING: {all_results_path} not found")
        return None, None
    
    df = pd.read_csv(all_results_path)
    df['dataset'] = dataset_name
    print(f"  {dataset_name}: {len(df)} rows, "
          f"{df['model'].nunique()} models, "
          f"{df['rep'].nunique()} reps, "
          f"{df['strategy'].nunique()} strategies")
    
    summary = None
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary['dataset'] = dataset_name
    
    return df, summary


def load_all_results(results_dir):
    """Load results for both COX-2 and DRD2."""
    print("=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)
    
    all_dfs = []
    all_summaries = []
    
    for dataset in ['cox2', 'drd2']:
        df, summary = load_dataset_results(results_dir, dataset)
        if df is not None:
            all_dfs.append(df)
        if summary is not None:
            all_summaries.append(summary)
    
    if not all_dfs:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    summary_combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else None
    
    print(f"\nCombined: {len(combined)} rows across {combined['dataset'].nunique()} datasets")
    
    return combined, summary_combined


# ============================================================================
# NDS CALCULATION
# ============================================================================

def calculate_nds(df, baseline_threshold=None):
    """
    Calculate Noise Degradation Slope for each config.
    
    If baseline_threshold is None, NDS is calculated for all configs
    but meets_threshold column marks which ones are reliable.
    """
    metrics = []
    
    groups = df.groupby(['dataset', 'model', 'rep', 'strategy'])
    
    for (dataset, model, rep, strategy), group in groups:
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        sigmas = group['sigma'].values
        r2_vals = group['r2'].values
        rmse_vals = group['rmse'].values
        
        baseline_r2 = r2_vals[sigmas == 0.0]
        baseline_rmse = rmse_vals[sigmas == 0.0]
        
        if len(baseline_r2) == 0:
            continue
        
        baseline_r2 = baseline_r2[0]
        baseline_rmse = baseline_rmse[0]
        
        # Linear regression across all sigma levels
        slope_r2, intercept, r_value, p_value, std_err = linregress(sigmas, r2_vals)
        slope_rmse, _, _, _, _ = linregress(sigmas, rmse_vals)
        
        entry = {
            'dataset': dataset,
            'model': model,
            'rep': rep,
            'strategy': strategy,
            'baseline_r2': baseline_r2,
            'baseline_rmse': baseline_rmse,
            'NDS_r2': slope_r2,
            'NDS_rmse': slope_rmse,
            'r2_fit': r_value ** 2,
            'p_value': p_value,
        }
        
        if baseline_threshold is not None:
            entry['meets_threshold'] = baseline_r2 > baseline_threshold
        else:
            entry['meets_threshold'] = True
        
        metrics.append(entry)
    
    metrics_df = pd.DataFrame(metrics)
    
    if baseline_threshold is not None:
        n_meets = metrics_df['meets_threshold'].sum()
        n_total = len(metrics_df)
        print(f"  NDS calculated: {n_total} configs, "
              f"{n_meets} meet baseline threshold (R² > {baseline_threshold})")
    
    return metrics_df


# ============================================================================
# FIGURE 1: PER-DATASET DEGRADATION CURVES BY STRATEGY
# ============================================================================

def figure_degradation_by_strategy(df, output_dir):
    """
    Two panels (COX-2, DRD2): R² degradation curves averaged across
    all model/rep configs, one line per noise strategy.
    """
    print("\nGenerating: Degradation curves by strategy...")
    
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5.5 * n_datasets, 4),
                              squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = df[df['dataset'] == dataset]
        
        strategies = [s for s in STRATEGY_ORDER if s in dset['strategy'].unique()]
        
        for strategy in strategies:
            sdata = dset[dset['strategy'] == strategy]
            avg = sdata.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
            
            if len(avg) < 3:
                continue
            
            colour = STRATEGY_COLOURS.get(strategy, '#999999')
            label = fmt(strategy, STRATEGY_LABELS)
            
            ax.plot(avg['sigma'], avg['r2'],
                    marker='o', markersize=4, linewidth=1.8,
                    color=colour, label=label, alpha=0.9)
        
        ax.set_xlabel('Noise level (σ)')
        ax.set_ylabel('Mean R² (across all configs)')
        ax.set_title(f'{fmt(dataset, DATASET_LABELS)}',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=6.5, loc='lower left', frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    
    plt.tight_layout()
    path = Path(output_dir) / "fig_degradation_by_strategy.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 2: LEGACY STRATEGY — MODEL × REP DEGRADATION CURVES
# ============================================================================

def figure_legacy_degradation(df, output_dir):
    """
    For the legacy (Gaussian) strategy: R² vs σ curves coloured by model,
    with line style by representation. One panel per dataset.
    """
    print("\nGenerating: Legacy strategy degradation by model/rep...")
    
    legacy = df[df['strategy'] == 'legacy']
    if len(legacy) == 0:
        print("  WARNING: No legacy strategy data found")
        return
    
    datasets = sorted(legacy['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4.5),
                              squeeze=False)
    
    rep_styles = {'ECFP4': '-', 'PDV': '--', 'SNS': '-.', 'MHG-GNN-pretrained': ':'}
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = legacy[legacy['dataset'] == dataset]
        
        for (model, rep), group in dset.groupby(['model', 'rep']):
            avg = group.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
            
            if len(avg) < 3:
                continue
            
            colour = MODEL_COLOURS.get(model, '#999999')
            style = rep_styles.get(rep, '-')
            label = f"{fmt(model, MODEL_LABELS)}/{fmt(rep, REP_LABELS)}"
            
            ax.plot(avg['sigma'], avg['r2'],
                    linestyle=style, marker='o', markersize=3,
                    linewidth=1.5, color=colour, label=label, alpha=0.85)
        
        ax.set_xlabel('Noise level (σ)')
        ax.set_ylabel('R²')
        ax.set_title(f'{fmt(dataset, DATASET_LABELS)} — Legacy (Gaussian)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=5.5, loc='lower left', ncol=2,
                  frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    
    plt.tight_layout()
    path = Path(output_dir) / "fig_legacy_degradation_by_config.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 3: NDS HEATMAPS — MODEL × REP (LEGACY STRATEGY)
# ============================================================================

def figure_nds_heatmaps(nds_df, output_dir, baseline_threshold=None):
    """
    Heatmap of NDS_r2 for legacy strategy: rows = models, cols = representations.
    One panel per dataset. Cells annotated with NDS value.
    Configs below baseline threshold shown greyed out.
    """
    print("\nGenerating: NDS heatmaps (legacy strategy)...")
    
    legacy_nds = nds_df[nds_df['strategy'] == 'legacy'].copy()
    if len(legacy_nds) == 0:
        print("  WARNING: No legacy NDS data")
        return
    
    datasets = sorted(legacy_nds['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets + 1, 4.5),
                              squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = legacy_nds[legacy_nds['dataset'] == dataset]
        
        pivot = dset.pivot_table(
            index='model', columns='rep', values='NDS_r2', aggfunc='mean'
        )
        
        # Baseline R² pivot for threshold annotation
        baseline_pivot = dset.pivot_table(
            index='model', columns='rep', values='baseline_r2', aggfunc='mean'
        )
        
        # Order rows/columns sensibly
        model_order = [m for m in ['RF', 'QRF', 'XGBoost', 'GP', 'DNN',
                                    'Full-BNN', 'LastLayer-BNN', 'Var-BNN']
                       if m in pivot.index]
        rep_order = [r for r in ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']
                     if r in pivot.columns]
        
        pivot = pivot.reindex(index=model_order, columns=rep_order)
        baseline_pivot = baseline_pivot.reindex(index=model_order, columns=rep_order)
        
        # All NDS values are negative (degradation), so vmax ~ 0
        vmin = pivot.min().min()
        vmax = 0
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Labels
        row_labels = [fmt(m, MODEL_LABELS) for m in pivot.index]
        col_labels = [fmt(r, REP_LABELS) for r in pivot.columns]
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                nds_val = pivot.iloc[i, j]
                bl_val = baseline_pivot.iloc[i, j]
                
                if pd.isna(nds_val):
                    continue
                
                # Grey out below-threshold configs
                below_threshold = (baseline_threshold is not None and
                                   pd.notna(bl_val) and bl_val <= baseline_threshold)
                
                if below_threshold:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=True, facecolor='#d5d5d5',
                                               edgecolor='white', linewidth=0.5,
                                               zorder=2))
                    ax.text(j, i, f'{nds_val:.2f}*', ha='center', va='center',
                            color='#999999', fontsize=6.5, fontstyle='italic', zorder=3)
                else:
                    text_colour = 'white' if nds_val < (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{nds_val:.2f}', ha='center', va='center',
                            color=text_colour, fontsize=6.5, fontweight='medium', zorder=3)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDS (R²)', fontsize=8, rotation=270, labelpad=12)
        cbar.ax.tick_params(labelsize=7)
        
        title = f'{fmt(dataset, DATASET_LABELS)} — Legacy'
        if baseline_threshold is not None:
            title += f'\n(* = baseline R² ≤ {baseline_threshold})'
        ax.set_title(title, fontweight='bold', fontsize=10, pad=10)
        ax.set_xlabel('Representation')
        ax.set_ylabel('Model')
    
    plt.tight_layout()
    path = Path(output_dir) / "fig_nds_heatmap_legacy.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 4: NDS BY STRATEGY — BAR CHART
# ============================================================================

def figure_nds_by_strategy(nds_df, output_dir, baseline_threshold=None):
    """
    Bar chart of mean NDS_r2 per strategy, one panel per dataset.
    Only includes configs above baseline threshold.
    """
    print("\nGenerating: NDS by strategy...")
    
    if baseline_threshold is not None:
        plot_df = nds_df[nds_df['meets_threshold']].copy()
    else:
        plot_df = nds_df.copy()
    
    if len(plot_df) == 0:
        print("  WARNING: No data above threshold")
        return
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4),
                              squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = plot_df[plot_df['dataset'] == dataset]
        
        strat_stats = dset.groupby('strategy').agg(
            nds_mean=('NDS_r2', 'mean'),
            nds_std=('NDS_r2', 'std'),
            n_configs=('NDS_r2', 'count'),
        ).reset_index()
        
        # Sort by |NDS| ascending (most robust first)
        strat_stats['abs_nds'] = strat_stats['nds_mean'].abs()
        strat_stats = strat_stats.sort_values('abs_nds', ascending=True)
        
        y_pos = np.arange(len(strat_stats))
        colours = [STRATEGY_COLOURS.get(s, '#999999') for s in strat_stats['strategy']]
        
        bars = ax.barh(y_pos, strat_stats['nds_mean'],
                       xerr=strat_stats['nds_std'].fillna(0),
                       color=colours, alpha=0.85, height=0.65,
                       edgecolor='black', linewidth=0.5, capsize=3)
        
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Annotate with n_configs
        for i, (_, row) in enumerate(strat_stats.iterrows()):
            ax.text(0.01, i, f'n={int(row["n_configs"])}',
                    va='center', fontsize=6, color='gray',
                    transform=ax.get_yaxis_transform())
        
        labels = [fmt(s, STRATEGY_LABELS) for s in strat_stats['strategy']]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Mean NDS (R²) ± std')
        ax.set_title(f'{fmt(dataset, DATASET_LABELS)}',
                     fontweight='bold', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    fig.suptitle('Noise Degradation Slope by Strategy\n(closer to 0 = more stable)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "fig_nds_by_strategy.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 5: MODEL × STRATEGY HEATMAP (NDS)
# ============================================================================

def figure_model_strategy_heatmap(nds_df, output_dir, baseline_threshold=None):
    """
    Heatmap: rows = model/rep configs, columns = noise strategies.
    Values = NDS_r2. One panel per dataset.
    Only includes configs above baseline threshold.
    """
    print("\nGenerating: Model × Strategy NDS heatmap...")
    
    if baseline_threshold is not None:
        plot_df = nds_df[nds_df['meets_threshold']].copy()
    else:
        plot_df = nds_df.copy()
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets + 1, 6),
                              squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = plot_df[plot_df['dataset'] == dataset]
        
        pivot = dset.pivot_table(
            index=['model', 'rep'],
            columns='strategy',
            values='NDS_r2',
            aggfunc='mean'
        )
        
        if len(pivot) == 0:
            ax.text(0.5, 0.5, 'No data above threshold',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Sort by mean |NDS| (most robust at top)
        pivot['_abs_mean'] = pivot.abs().mean(axis=1)
        pivot = pivot.sort_values('_abs_mean', ascending=True).drop('_abs_mean', axis=1)
        
        # Order columns
        col_order = [s for s in STRATEGY_ORDER if s in pivot.columns]
        pivot = pivot[col_order]
        
        vmin = pivot.min().min()
        vmax = 0
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        
        row_labels = [f"{fmt(m, MODEL_LABELS)}/{fmt(r, REP_LABELS)}"
                      for m, r in pivot.index]
        col_labels = [fmt(s, STRATEGY_LABELS) for s in pivot.columns]
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    text_colour = 'white' if val < (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=text_colour, fontsize=5.5)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDS (R²)', fontsize=8, rotation=270, labelpad=12)
        cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(f'{fmt(dataset, DATASET_LABELS)}',
                     fontweight='bold', fontsize=10, pad=10)
        ax.set_xlabel('Noise Strategy')
        ax.set_ylabel('Model / Representation')
    
    fig.suptitle('Noise Degradation Slope: Model × Strategy',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "fig_nds_model_strategy_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 6: CROSS-DATASET COMPARISON
# ============================================================================

def figure_cross_dataset(nds_df, output_dir, baseline_threshold=None):
    """
    Compare COX-2 vs DRD2 NDS for the legacy strategy.
    
    Panel A: Scatter of NDS_r2 (COX-2 vs DRD2) per model/rep
    Panel B: Paired bar chart of baseline R² by model
    Panel C: Degradation slope comparison by model
    """
    print("\nGenerating: Cross-dataset comparison...")
    
    legacy = nds_df[nds_df['strategy'] == 'legacy'].copy()
    
    datasets = legacy['dataset'].unique()
    if len(datasets) < 2:
        print("  WARNING: Need both datasets for cross-comparison")
        return
    
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.06, right=0.98,
                          top=0.88, bottom=0.15)
    
    # --- Panel A: Scatter NDS COX-2 vs DRD2 ---
    ax_a = fig.add_subplot(gs[0, 0])
    
    cox2_nds = legacy[legacy['dataset'] == 'cox2'].set_index(['model', 'rep'])
    drd2_nds = legacy[legacy['dataset'] == 'drd2'].set_index(['model', 'rep'])
    
    common = cox2_nds.index.intersection(drd2_nds.index)
    
    if len(common) >= 3:
        x = cox2_nds.loc[common, 'NDS_r2'].values
        y = drd2_nds.loc[common, 'NDS_r2'].values
        
        colours = [MODEL_COLOURS.get(m, '#999999') for m, r in common]
        
        ax_a.scatter(x, y, c=colours, s=60, alpha=0.8,
                     edgecolors='black', linewidth=0.5, zorder=3)
        
        # Diagonal
        lim_min = min(x.min(), y.min()) * 1.1
        lim_max = 0.05
        ax_a.plot([lim_min, lim_max], [lim_min, lim_max],
                  'k--', alpha=0.4, linewidth=1)
        
        corr, pval = spearmanr(x, y)
        ax_a.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {pval:.4f}',
                  transform=ax_a.transAxes, fontsize=7, va='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Label points
        for (model, rep), xi, yi in zip(common, x, y):
            ax_a.annotate(f'{fmt(model, MODEL_LABELS)}/{fmt(rep, REP_LABELS)}',
                         (xi, yi), fontsize=4.5, alpha=0.7,
                         xytext=(3, 3), textcoords='offset points')
    
    ax_a.set_xlabel('NDS (R²) — COX-2')
    ax_a.set_ylabel('NDS (R²) — DRD2')
    ax_a.set_title('A. NDS concordance (Legacy)', fontweight='bold', pad=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # --- Panel B: Baseline R² comparison ---
    ax_b = fig.add_subplot(gs[0, 1])
    
    baseline_comp = []
    for (model, rep) in common:
        baseline_comp.append({
            'config': f'{fmt(model, MODEL_LABELS)}/{fmt(rep, REP_LABELS)}',
            'model': model,
            'COX-2': cox2_nds.loc[(model, rep), 'baseline_r2'],
            'DRD2': drd2_nds.loc[(model, rep), 'baseline_r2'],
        })
    
    if baseline_comp:
        bdf = pd.DataFrame(baseline_comp).sort_values('COX-2', ascending=True)
        
        y_pos = np.arange(len(bdf))
        height = 0.35
        
        ax_b.barh(y_pos - height/2, bdf['COX-2'], height=height,
                  color=DATASET_COLOURS['cox2'], alpha=0.8,
                  edgecolor='black', linewidth=0.5, label='COX-2')
        ax_b.barh(y_pos + height/2, bdf['DRD2'], height=height,
                  color=DATASET_COLOURS['drd2'], alpha=0.8,
                  edgecolor='black', linewidth=0.5, label='DRD2')
        
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(bdf['config'], fontsize=6)
        ax_b.set_xlabel('Baseline R² (σ = 0)')
        ax_b.legend(fontsize=7, loc='lower right')
    
    ax_b.set_title('B. Baseline performance', fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # --- Panel C: NDS comparison by model (averaged across reps) ---
    ax_c = fig.add_subplot(gs[0, 2])
    
    model_nds = legacy.groupby(['dataset', 'model']).agg(
        nds_mean=('NDS_r2', 'mean'),
        nds_std=('NDS_r2', 'std'),
    ).reset_index()
    
    models = sorted(legacy['model'].unique(),
                    key=lambda m: list(MODEL_LABELS.keys()).index(m)
                    if m in MODEL_LABELS else 99)
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    for i, dataset in enumerate(['cox2', 'drd2']):
        dset = model_nds[model_nds['dataset'] == dataset]
        vals = [dset[dset['model'] == m]['nds_mean'].values[0]
                if m in dset['model'].values else np.nan for m in models]
        errs = [dset[dset['model'] == m]['nds_std'].values[0]
                if m in dset['model'].values else 0 for m in models]
        
        offset = (i - 0.5) * width
        ax_c.bar(x_pos + offset, vals, width=width,
                 yerr=errs, color=DATASET_COLOURS[dataset],
                 alpha=0.8, edgecolor='black', linewidth=0.5,
                 capsize=3, label=fmt(dataset, DATASET_LABELS))
    
    ax_c.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([fmt(m, MODEL_LABELS) for m in models],
                          rotation=45, ha='right', fontsize=7)
    ax_c.set_ylabel('NDS (R²)')
    ax_c.set_title('C. NDS by model (Legacy)', fontweight='bold', pad=10)
    ax_c.legend(fontsize=7, loc='lower left')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    path = Path(output_dir) / "fig_cross_dataset_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# FIGURE 7: STRATEGY RANKING CONCORDANCE
# ============================================================================

def figure_strategy_concordance(nds_df, output_dir, baseline_threshold=None):
    """
    Spearman correlation of model rankings between noise strategies.
    One panel per dataset.
    """
    print("\nGenerating: Strategy ranking concordance...")
    
    if baseline_threshold is not None:
        plot_df = nds_df[nds_df['meets_threshold']].copy()
    else:
        plot_df = nds_df.copy()
    
    datasets = sorted(plot_df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5.5 * n_datasets, 5),
                              squeeze=False)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        dset = plot_df[plot_df['dataset'] == dataset]
        
        pivot = dset.pivot_table(
            index=['model', 'rep'],
            columns='strategy',
            values='NDS_r2',
            aggfunc='mean'
        )
        
        strategies = [s for s in STRATEGY_ORDER
                      if s in pivot.columns and pivot[s].notna().sum() >= 3]
        
        if len(strategies) < 2:
            ax.text(0.5, 0.5, 'Need ≥ 2 strategies with ≥ 3 configs',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.axis('off')
            continue
        
        n = len(strategies)
        corr_matrix = np.ones((n, n))
        
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                if i != j:
                    d1 = pivot[s1].dropna()
                    d2 = pivot[s2].dropna()
                    common = d1.index.intersection(d2.index)
                    if len(common) >= 3:
                        corr, _ = spearmanr(d1[common], d2[common])
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = np.nan
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                       aspect='equal', interpolation='nearest')
        
        labels = [fmt(s, STRATEGY_LABELS) for s in strategies]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        
        for i in range(n):
            for j in range(n):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    text_colour = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=text_colour, fontsize=8, fontweight='medium')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman ρ', fontsize=8, rotation=270, labelpad=12)
        cbar.ax.tick_params(labelsize=7)
        
        ax.set_title(f'{fmt(dataset, DATASET_LABELS)}',
                     fontweight='bold', fontsize=10, pad=10)
    
    fig.suptitle('Strategy Ranking Concordance\n(Spearman ρ of NDS rankings)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "fig_strategy_concordance.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(nds_df, output_dir, baseline_threshold=None):
    """Save CSV summary tables."""
    print("\nGenerating: Summary tables...")
    
    output_dir = Path(output_dir)
    
    # Full NDS table
    nds_df.to_csv(output_dir / "nds_all_configs.csv", index=False)
    print(f"  ✓ nds_all_configs.csv")
    
    if baseline_threshold is not None:
        thresh_df = nds_df[nds_df['meets_threshold']].copy()
    else:
        thresh_df = nds_df.copy()
    
    # By strategy per dataset
    strat_table = thresh_df.groupby(['dataset', 'strategy']).agg(
        mean_NDS_r2=('NDS_r2', 'mean'),
        std_NDS_r2=('NDS_r2', 'std'),
        mean_baseline_r2=('baseline_r2', 'mean'),
        n_configs=('NDS_r2', 'count'),
    ).reset_index()
    strat_table['abs_mean_NDS'] = strat_table['mean_NDS_r2'].abs()
    strat_table = strat_table.sort_values(['dataset', 'abs_mean_NDS'])
    strat_table = strat_table.round(4)
    strat_table.to_csv(output_dir / "table_nds_by_strategy.csv", index=False)
    print(f"  ✓ table_nds_by_strategy.csv")
    
    # Top configs per dataset (legacy)
    legacy = thresh_df[thresh_df['strategy'] == 'legacy']
    top_table = legacy.sort_values(['dataset', 'NDS_r2'], ascending=[True, False])
    top_table = top_table[['dataset', 'model', 'rep', 'baseline_r2',
                           'NDS_r2', 'NDS_rmse', 'r2_fit']].round(4)
    top_table.to_csv(output_dir / "table_nds_legacy_ranked.csv", index=False)
    print(f"  ✓ table_nds_legacy_ranked.csv")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="results", baseline_threshold=None):
    """Main execution."""
    print("=" * 80)
    print("PHARMACEUTICAL NOISE ROBUSTNESS ANALYSIS")
    print("  COX-2 (IC50) + DRD2 (Ki)")
    print("=" * 80)
    
    if baseline_threshold is not None:
        print(f"  Baseline threshold: R² > {baseline_threshold}")
    else:
        print("  Baseline threshold: None (all configs included)")
    
    # Load
    df, precomputed_summary = load_all_results(results_dir)
    
    # Dataset summary
    for dataset in df['dataset'].unique():
        dset = df[df['dataset'] == dataset]
        print(f"\n  {fmt(dataset, DATASET_LABELS)}:")
        print(f"    Models: {sorted(dset['model'].unique())}")
        print(f"    Reps: {sorted(dset['rep'].unique())}")
        print(f"    Strategies: {sorted(dset['strategy'].unique())}")
        print(f"    σ range: {dset['sigma'].min():.1f} – {dset['sigma'].max():.1f}")
        
        # Baseline R² range
        baseline = dset[dset['sigma'] == 0.0]
        if len(baseline) > 0:
            print(f"    Baseline R² range: {baseline['r2'].min():.3f} – {baseline['r2'].max():.3f}")
    
    # Calculate NDS
    print("\n" + "=" * 80)
    print("CALCULATING NOISE DEGRADATION SLOPES")
    print("=" * 80)
    
    nds_df = calculate_nds(df, baseline_threshold=baseline_threshold)
    
    if len(nds_df) == 0:
        print("ERROR: No NDS metrics calculated")
        return
    
    # Output directory
    output_dir = Path(results_dir) / "pharma_figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    
    figure_degradation_by_strategy(df, output_dir)
    figure_legacy_degradation(df, output_dir)
    figure_nds_heatmaps(nds_df, output_dir, baseline_threshold)
    figure_nds_by_strategy(nds_df, output_dir, baseline_threshold)
    figure_model_strategy_heatmap(nds_df, output_dir, baseline_threshold)
    figure_cross_dataset(nds_df, output_dir, baseline_threshold)
    figure_strategy_concordance(nds_df, output_dir, baseline_threshold)
    create_summary_tables(nds_df, output_dir, baseline_threshold)
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    for dataset in nds_df['dataset'].unique():
        dset = nds_df[(nds_df['dataset'] == dataset) & (nds_df['meets_threshold'])]
        legacy = dset[dset['strategy'] == 'legacy']
        
        print(f"\n{fmt(dataset, DATASET_LABELS)}:")
        
        if len(legacy) > 0:
            print(f"  Legacy strategy — most robust (least negative NDS):")
            best = legacy.nlargest(3, 'NDS_r2')
            for _, row in best.iterrows():
                print(f"    {fmt(row['model'], MODEL_LABELS)}/{fmt(row['rep'], REP_LABELS)}: "
                      f"NDS={row['NDS_r2']:.4f}, baseline R²={row['baseline_r2']:.3f}")
        
        # Strategy ranking
        strat_rank = dset.groupby('strategy')['NDS_r2'].apply(
            lambda x: x.abs().mean()
        ).sort_values()
        
        if len(strat_rank) > 0:
            print(f"  Strategy ranking (|NDS|, lower = more stable):")
            for i, (strat, val) in enumerate(strat_rank.items(), 1):
                print(f"    {i}. {fmt(strat, STRATEGY_LABELS)}: |NDS|={val:.4f}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    # Parse optional threshold
    threshold = None
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])
    
    main(results_dir, baseline_threshold=threshold)
