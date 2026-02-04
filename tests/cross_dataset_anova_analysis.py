#!/usr/bin/env python3
"""
Cross-Dataset ANOVA Analysis
=============================

Tests whether the "representation drives performance, model drives robustness"
finding generalizes across datasets.

Paper Hypothesis (Part 1):
- Representation explains ~72% variance in predictive performance (R²)
- Model explains ~67% variance in noise robustness (NDS)

This script tests if this pattern holds across:
- OpenADMET-LogD (lipophilicity)
- OpenADMET-Caco2 (P-gp efflux)
- FLuID-hERG (cardiotoxicity, classification)

Outputs:
- Figure: Side-by-side ANOVA variance decomposition across datasets
- Table: η² values per dataset for model/representation effects
- Statistical summary of pattern consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS (Journal of Cheminformatics)
# =============================================================================

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
    'legend.frameon': False,
    'lines.linewidth': 1.5,
})

DATASET_COLORS = {
    'OpenADMET-LogD': '#3498db',
    'OpenADMET-Caco2': '#e74c3c',
    'FLuID-hERG': '#2ecc71',
}

DATASET_LABELS = {
    'OpenADMET-LogD': 'LogD',
    'OpenADMET-Caco2': 'Caco2',
    'FLuID-hERG': 'hERG',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_results(results_dir, dataset_name):
    """Load all_results.csv for a dataset."""
    results_dir = Path(results_dir)

    # Try different possible locations
    possible_paths = [
        results_dir / dataset_name / 'all_results.csv',
        results_dir / f'{dataset_name}_all_results.csv',
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            df['dataset'] = dataset_name
            print(f"Loaded {dataset_name}: {len(df)} rows")
            return df

    print(f"Warning: No data found for {dataset_name}")
    return None


def load_all_datasets(results_dir):
    """Load all available datasets."""
    results_dir = Path(results_dir)

    datasets = {}
    # Dataset folder names from alternative_data_noise_robustness.py
    for name in ['OpenADMET-LogD', 'OpenADMET-Caco2', 'FLuID-hERG']:
        df = load_dataset_results(results_dir, name)
        if df is not None:
            datasets[name] = df

    return datasets


# =============================================================================
# ANOVA FUNCTIONS
# =============================================================================

def run_performance_anova(df, sigma_value=0.3):
    """
    Run ANOVA on R² at a specific sigma level.

    Tests: "Representation drives predictive performance"
    Expected: ~72% variance explained by representation
    """
    # Filter to specific sigma
    df_sigma = df[np.abs(df['sigma'] - sigma_value) < 0.05].copy()

    if len(df_sigma) == 0:
        return None

    # Clean data
    df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])

    # Standardize column names
    if 'representation' in df_sigma.columns and 'rep' not in df_sigma.columns:
        df_sigma['rep'] = df_sigma['representation']

    if len(df_sigma) < 10:
        return None

    # Calculate ANOVA components
    grand_mean = df_sigma['r2'].mean()
    total_ss = ((df_sigma['r2'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    # Model effect
    model_means = df_sigma.groupby('model')['r2'].mean()
    model_counts = df_sigma.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    # Representation effect
    rep_means = df_sigma.groupby('rep')['r2'].mean()
    rep_counts = df_sigma.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    # Interaction effect
    interaction_means = df_sigma.groupby(['model', 'rep'])['r2'].mean()
    interaction_counts = df_sigma.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    # Residual
    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    # Calculate η² (proportion of variance)
    eta2_model = (ss_model / total_ss) * 100
    eta2_rep = (ss_rep / total_ss) * 100
    eta2_interaction = (ss_interaction / total_ss) * 100
    eta2_residual = (ss_residual / total_ss) * 100

    return {
        'n_observations': len(df_sigma),
        'n_models': df_sigma['model'].nunique(),
        'n_reps': df_sigma['rep'].nunique(),
        'grand_mean_r2': grand_mean,
        'eta2_model': eta2_model,
        'eta2_rep': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual,
    }


def run_robustness_anova(df, baseline_threshold=0.4):
    """
    Run ANOVA on Noise Degradation Slope (NDS).

    Tests: "Model drives noise robustness"
    Expected: ~67% variance explained by model
    """
    # Calculate NDS for each model-rep combination
    nds_data = []

    for (model, rep), group in df.groupby(['model', 'rep']):
        group = group.sort_values('sigma')

        if len(group) < 3:
            continue

        # Check baseline threshold
        baseline = group[group['sigma'] == 0.0]
        if len(baseline) == 0:
            # Try closest to 0
            baseline = group[group['sigma'] == group['sigma'].min()]

        if len(baseline) == 0 or baseline['r2'].values[0] < baseline_threshold:
            continue

        # Calculate slope (NDS)
        try:
            slope, _, r_val, p_val, _ = stats.linregress(group['sigma'], group['r2'])
            nds_data.append({
                'model': model,
                'rep': rep,
                'nds': slope,
                'baseline_r2': baseline['r2'].values[0],
            })
        except:
            continue

    if len(nds_data) < 6:
        return None

    nds_df = pd.DataFrame(nds_data)

    # Run ANOVA on NDS
    grand_mean = nds_df['nds'].mean()
    total_ss = ((nds_df['nds'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    # Model effect on robustness
    model_means = nds_df.groupby('model')['nds'].mean()
    model_counts = nds_df.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    # Representation effect on robustness
    rep_means = nds_df.groupby('rep')['nds'].mean()
    rep_counts = nds_df.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    # Interaction
    interaction_means = nds_df.groupby(['model', 'rep'])['nds'].mean()
    interaction_counts = nds_df.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    eta2_model = (ss_model / total_ss) * 100
    eta2_rep = (ss_rep / total_ss) * 100
    eta2_interaction = (ss_interaction / total_ss) * 100
    eta2_residual = (ss_residual / total_ss) * 100

    return {
        'n_configs': len(nds_df),
        'n_models': nds_df['model'].nunique(),
        'n_reps': nds_df['rep'].nunique(),
        'mean_nds': grand_mean,
        'eta2_model': eta2_model,
        'eta2_rep': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual,
    }


# =============================================================================
# ANALYSIS BY STRATEGY
# =============================================================================

def analyze_by_strategy(df):
    """Run ANOVA for each noise strategy separately."""
    strategies = df['strategy'].unique()

    results = {}
    for strategy in strategies:
        df_strat = df[df['strategy'] == strategy]

        perf = run_performance_anova(df_strat)
        robust = run_robustness_anova(df_strat)

        if perf and robust:
            results[strategy] = {
                'performance': perf,
                'robustness': robust,
            }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_cross_dataset_anova(all_results, output_dir):
    """Create side-by-side ANOVA comparison figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Prepare data for plotting
    datasets = list(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Performance ANOVA (left panel)
    ax1 = axes[0]
    x = np.arange(len(datasets))
    width = 0.35

    rep_vals = [all_results[d]['legacy']['performance']['eta2_rep'] for d in datasets]
    model_vals = [all_results[d]['legacy']['performance']['eta2_model'] for d in datasets]

    bars1 = ax1.bar(x - width/2, rep_vals, width, label='Representation', color='#3498db')
    bars2 = ax1.bar(x + width/2, model_vals, width, label='Model', color='#e74c3c')

    ax1.set_ylabel('Variance Explained (η², %)')
    ax1.set_title('A. Predictive Performance (R²)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS.get(d, d.upper()) for d in datasets])
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=72, color='#3498db', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(len(datasets)-0.5, 74, 'QM9 ref: 72%', fontsize=6, color='#3498db')

    # Add value labels
    for bar, val in zip(bars1, rep_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, model_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=6)

    # Robustness ANOVA (right panel)
    ax2 = axes[1]

    rep_vals = [all_results[d]['legacy']['robustness']['eta2_rep'] for d in datasets]
    model_vals = [all_results[d]['legacy']['robustness']['eta2_model'] for d in datasets]

    bars1 = ax2.bar(x - width/2, rep_vals, width, label='Representation', color='#3498db')
    bars2 = ax2.bar(x + width/2, model_vals, width, label='Model', color='#e74c3c')

    ax2.set_ylabel('Variance Explained (η², %)')
    ax2.set_title('B. Noise Robustness (NDS)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS.get(d, d.upper()) for d in datasets])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=67, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.text(len(datasets)-0.5, 69, 'QM9 ref: 67%', fontsize=6, color='#e74c3c')

    for bar, val in zip(bars1, rep_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, model_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_anova_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cross_dataset_anova_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'cross_dataset_anova_comparison.png'}")


def plot_strategy_consistency(all_results, output_dir):
    """Show ANOVA decomposition across all strategies for each dataset."""
    output_dir = Path(output_dir)

    datasets = list(all_results.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, 2, figsize=(8, 2.5 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for i, dataset in enumerate(datasets):
        strategies = sorted(all_results[dataset].keys())

        # Performance
        ax1 = axes[i, 0]
        x = np.arange(len(strategies))
        width = 0.35

        rep_vals = [all_results[dataset][s]['performance']['eta2_rep'] for s in strategies]
        model_vals = [all_results[dataset][s]['performance']['eta2_model'] for s in strategies]

        ax1.bar(x - width/2, rep_vals, width, label='Rep', color='#3498db')
        ax1.bar(x + width/2, model_vals, width, label='Model', color='#e74c3c')
        ax1.set_ylabel('η² (%)')
        ax1.set_title(f'{DATASET_LABELS.get(dataset, dataset.upper())} - Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        if i == 0:
            ax1.legend(loc='upper right', fontsize=6)

        # Robustness
        ax2 = axes[i, 1]

        rep_vals = [all_results[dataset][s]['robustness']['eta2_rep'] for s in strategies]
        model_vals = [all_results[dataset][s]['robustness']['eta2_model'] for s in strategies]

        ax2.bar(x - width/2, rep_vals, width, label='Rep', color='#3498db')
        ax2.bar(x + width/2, model_vals, width, label='Model', color='#e74c3c')
        ax2.set_ylabel('η² (%)')
        ax2.set_title(f'{DATASET_LABELS.get(dataset, dataset.upper())} - Robustness')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_dataset_strategy_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_dir / 'cross_dataset_strategy_consistency.png'}")


# =============================================================================
# REPORTING
# =============================================================================

def generate_summary_table(all_results, output_dir):
    """Generate CSV and LaTeX summary tables."""
    output_dir = Path(output_dir)

    rows = []
    for dataset, strategies in all_results.items():
        for strategy, results in strategies.items():
            rows.append({
                'Dataset': DATASET_LABELS.get(dataset, dataset.upper()),
                'Strategy': strategy,
                'Perf_η²_Rep': results['performance']['eta2_rep'],
                'Perf_η²_Model': results['performance']['eta2_model'],
                'Robust_η²_Rep': results['robustness']['eta2_rep'],
                'Robust_η²_Model': results['robustness']['eta2_model'],
                'N_obs': results['performance']['n_observations'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'cross_dataset_anova_summary.csv', index=False)

    # LaTeX table (legacy strategy only for simplicity)
    legacy_df = df[df['Strategy'] == 'legacy'].copy()
    latex = legacy_df.to_latex(index=False, float_format='%.1f')
    with open(output_dir / 'cross_dataset_anova_table.tex', 'w') as f:
        f.write(latex)

    print(f"Saved tables to {output_dir}")
    return df


def generate_report(all_results, output_dir):
    """Generate text report summarizing findings."""
    output_dir = Path(output_dir)

    lines = []
    lines.append("=" * 70)
    lines.append("CROSS-DATASET ANOVA ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("HYPOTHESIS: Representation drives performance (~72%), Model drives robustness (~67%)")
    lines.append("")

    # Summary for each dataset
    for dataset, strategies in all_results.items():
        lines.append("-" * 70)
        lines.append(f"DATASET: {DATASET_LABELS.get(dataset, dataset.upper())}")
        lines.append("-" * 70)

        if 'legacy' in strategies:
            perf = strategies['legacy']['performance']
            robust = strategies['legacy']['robustness']

            lines.append(f"  Performance (R² at σ=0.3):")
            lines.append(f"    Representation η²: {perf['eta2_rep']:.1f}%")
            lines.append(f"    Model η²:          {perf['eta2_model']:.1f}%")
            lines.append(f"    N observations:    {perf['n_observations']}")
            lines.append("")
            lines.append(f"  Robustness (NDS):")
            lines.append(f"    Representation η²: {robust['eta2_rep']:.1f}%")
            lines.append(f"    Model η²:          {robust['eta2_model']:.1f}%")
            lines.append(f"    N configs:         {robust['n_configs']}")
            lines.append("")

            # Check if pattern holds
            perf_matches = perf['eta2_rep'] > perf['eta2_model']
            robust_matches = robust['eta2_model'] > robust['eta2_rep']

            if perf_matches and robust_matches:
                lines.append("  ✓ Pattern CONFIRMED: Rep→Performance, Model→Robustness")
            else:
                lines.append("  ✗ Pattern NOT confirmed")
                if not perf_matches:
                    lines.append(f"    - Performance: Model ({perf['eta2_model']:.1f}%) > Rep ({perf['eta2_rep']:.1f}%)")
                if not robust_matches:
                    lines.append(f"    - Robustness: Rep ({robust['eta2_rep']:.1f}%) > Model ({robust['eta2_model']:.1f}%)")

        lines.append("")

    # Overall summary
    lines.append("=" * 70)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 70)

    n_confirmed = 0
    n_total = 0
    for dataset, strategies in all_results.items():
        if 'legacy' in strategies:
            perf = strategies['legacy']['performance']
            robust = strategies['legacy']['robustness']
            if perf['eta2_rep'] > perf['eta2_model'] and robust['eta2_model'] > robust['eta2_rep']:
                n_confirmed += 1
            n_total += 1

    lines.append(f"Pattern confirmed in {n_confirmed}/{n_total} datasets")
    lines.append("")

    report = '\n'.join(lines)

    with open(output_dir / 'cross_dataset_anova_report.txt', 'w') as f:
        f.write(report)

    print(report)
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Cross-dataset ANOVA analysis')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing dataset result folders')
    parser.add_argument('--output-dir', type=str, default='results/cross_dataset_analysis',
                        help='Output directory for figures and tables')
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    output_dir = script_dir / args.output_dir

    print(f"Loading data from {results_dir}...")
    datasets = load_all_datasets(results_dir)

    if not datasets:
        print("ERROR: No datasets found!")
        return

    print(f"\nFound {len(datasets)} datasets: {list(datasets.keys())}")

    # Run analysis for each dataset
    all_results = {}
    for name, df in datasets.items():
        print(f"\nAnalyzing {name}...")
        all_results[name] = analyze_by_strategy(df)
        print(f"  Completed {len(all_results[name])} strategies")

    # Generate outputs
    print("\nGenerating outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_cross_dataset_anova(all_results, output_dir)
    plot_strategy_consistency(all_results, output_dir)
    generate_summary_table(all_results, output_dir)
    generate_report(all_results, output_dir)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
