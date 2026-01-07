#!/usr/bin/env python3
"""
Noise Robustness Experiment Analysis
=====================================

Comprehensive analysis, visualization, and summary tables for noise mitigation experiments.

Key Questions Addressed:
1. Do hybrids outperform base representations for noise detection?
2. Which mitigation method works best?
3. Which distance metric is optimal?
4. Does AC-aware detection (neighborhood_consensus) help?
5. Does feature weighting improve results?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'hybrid': '#2ecc71',
    'base': '#3498db', 
    'best': '#e74c3c',
    'ac_aware': '#9b59b6',
    'baseline': '#95a5a6'
}

METHOD_CATEGORIES = {
    'distance_knn': ['knn_k5', 'knn_k10', 'knn_k20'],
    'distance_lof': ['lof', 'lof_k10', 'lof_k20'],
    'distance_advanced': ['distance_weighted', 'neighborhood_consensus', 'activity_cliffs'],
    'ensemble': ['cv_disagreement', 'bootstrap_ensemble'],
    'model_based': ['co_teaching', 'dividemix'],
    'baseline': ['zscore', 'prediction_error', 'mahalanobis']
}

AC_AWARE_METHODS = {'neighborhood_consensus'}
DISTANCE_AWARE_METHODS = {'knn_k5', 'knn_k10', 'knn_k20', 'lof', 'lof_k10', 'lof_k20',
                          'distance_weighted', 'neighborhood_consensus', 'activity_cliffs'}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(results_path: Path) -> pd.DataFrame:
    """Load results from CSV or pickle."""
    if results_path.suffix == '.csv':
        df = pd.read_csv(results_path)
    elif results_path.suffix == '.pkl':
        df = pd.read_pickle(results_path)
    else:
        raise ValueError(f"Unknown format: {results_path.suffix}")
    
    # Ensure required columns exist
    required = ['representation', 'rep_type', 'sigma', 'method', 'distance_metric', 
                'weighted', 'recovery_rate', 'r2_clean', 'r2_noisy', 'r2_cleaned']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
    
    return df


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def table_overall_rankings(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Table 1: Overall Top Configurations
    
    Shows the best performing (representation, method, metric, weighted) combinations.
    """
    ranking_cols = ['representation', 'rep_type', 'method', 'distance_metric', 'weighted', 'sigma']
    metric_cols = ['recovery_rate', 'precision', 'recall', 'f1', 'r2_cleaned']
    
    available_metrics = [c for c in metric_cols if c in df.columns]
    
    summary = df.groupby(ranking_cols)[available_metrics].mean().reset_index()
    summary = summary.sort_values('recovery_rate', ascending=False).head(top_n)
    
    # Format for display
    summary['recovery_rate'] = summary['recovery_rate'].apply(lambda x: f"{x:+.1%}")
    if 'precision' in summary.columns:
        summary['precision'] = summary['precision'].apply(lambda x: f"{x:.1%}")
        summary['recall'] = summary['recall'].apply(lambda x: f"{x:.1%}")
    if 'r2_cleaned' in summary.columns:
        summary['r2_cleaned'] = summary['r2_cleaned'].apply(lambda x: f"{x:.4f}")
    
    return summary


def table_representation_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Hybrid vs Base Representation Performance
    
    THE KEY TABLE - answers whether hybrids provide better noise detection.
    """
    # Group by representation type and representation
    summary = df.groupby(['rep_type', 'representation', 'sigma']).agg({
        'recovery_rate': ['mean', 'std', 'max'],
        'r2_clean': 'first',
        'r2_noisy': 'first',
        'r2_cleaned': 'mean'
    }).reset_index()
    
    summary.columns = ['rep_type', 'representation', 'sigma', 
                       'recovery_mean', 'recovery_std', 'recovery_max',
                       'r2_clean', 'r2_noisy', 'r2_cleaned_mean']
    
    # Calculate "best possible" - what's the max recovery achieved?
    summary['best_config_recovery'] = summary['recovery_max']
    
    # Sort by rep_type (hybrid first) then by recovery
    summary = summary.sort_values(['rep_type', 'recovery_mean'], 
                                   ascending=[True, False])
    
    return summary


def table_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Method Performance Comparison
    
    Which noise mitigation method works best across all settings?
    """
    # Add method category
    def get_category(method):
        for cat, methods in METHOD_CATEGORIES.items():
            if method in methods:
                return cat
        return 'other'
    
    df = df.copy()
    df['method_category'] = df['method'].apply(get_category)
    df['is_ac_aware'] = df['method'].isin(AC_AWARE_METHODS)
    
    summary = df.groupby(['method', 'method_category', 'is_ac_aware']).agg({
        'recovery_rate': ['mean', 'std', 'max', 'count'],
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    summary.columns = ['method', 'category', 'ac_aware',
                       'recovery_mean', 'recovery_std', 'recovery_max', 'n_experiments',
                       'precision_mean', 'recall_mean']
    
    summary = summary.sort_values('recovery_mean', ascending=False)
    
    return summary


def table_distance_metric_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4: Distance Metric Performance
    
    Which distance metric works best for noise detection?
    """
    # Filter to only distance-aware methods
    df_dist = df[df['method'].isin(DISTANCE_AWARE_METHODS)].copy()
    
    summary = df_dist.groupby(['distance_metric', 'weighted']).agg({
        'recovery_rate': ['mean', 'std', 'max'],
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    summary.columns = ['distance_metric', 'weighted',
                       'recovery_mean', 'recovery_std', 'recovery_max',
                       'precision_mean', 'recall_mean']
    
    summary = summary.sort_values('recovery_mean', ascending=False)
    
    return summary


def table_weighting_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 5: Effect of Feature Weighting
    
    Paired comparison: does weighting improve performance?
    """
    # Filter to methods that can use weighting
    df_weight = df[df['method'].isin(DISTANCE_AWARE_METHODS)].copy()
    
    # Pivot to get weighted vs unweighted side by side
    pivot_cols = ['representation', 'method', 'distance_metric', 'sigma']
    
    weighted = df_weight[df_weight['weighted'] == True].set_index(pivot_cols)['recovery_rate']
    unweighted = df_weight[df_weight['weighted'] == False].set_index(pivot_cols)['recovery_rate']
    
    comparison = pd.DataFrame({
        'unweighted': unweighted,
        'weighted': weighted
    }).dropna()
    
    comparison['improvement'] = comparison['weighted'] - comparison['unweighted']
    comparison['pct_improved'] = (comparison['improvement'] > 0).astype(float)
    
    # Summary statistics
    summary = pd.DataFrame({
        'metric': ['Mean improvement', 'Median improvement', '% configs improved', 
                   'Max improvement', 'Min improvement'],
        'value': [
            f"{comparison['improvement'].mean():+.2%}",
            f"{comparison['improvement'].median():+.2%}",
            f"{comparison['pct_improved'].mean():.1%}",
            f"{comparison['improvement'].max():+.2%}",
            f"{comparison['improvement'].min():+.2%}"
        ]
    })
    
    return summary


def table_ac_aware_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6: AC-Aware vs Standard Methods
    
    Does neighborhood_consensus (which preserves activity cliffs) outperform
    naive k-NN methods that might incorrectly remove activity cliffs as noise?
    """
    # Compare neighborhood_consensus to knn methods
    knn_methods = ['knn_k5', 'knn_k10', 'knn_k20']
    ac_method = 'neighborhood_consensus'
    
    df_compare = df[df['method'].isin(knn_methods + [ac_method])].copy()
    df_compare['is_ac_aware'] = df_compare['method'] == ac_method
    
    summary = df_compare.groupby(['representation', 'sigma', 'is_ac_aware']).agg({
        'recovery_rate': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    # Pivot for comparison
    pivot = summary.pivot_table(
        index=['representation', 'sigma'],
        columns='is_ac_aware',
        values=['recovery_rate', 'precision', 'recall']
    )
    
    # Calculate improvement
    result = pd.DataFrame({
        'representation': pivot.index.get_level_values(0),
        'sigma': pivot.index.get_level_values(1),
        'knn_recovery': pivot[('recovery_rate', False)].values,
        'ac_aware_recovery': pivot[('recovery_rate', True)].values,
    })
    result['ac_advantage'] = result['ac_aware_recovery'] - result['knn_recovery']
    
    return result


def table_best_per_representation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 7: Best Configuration Per Representation
    
    What's the optimal (method, metric, weighting) for each representation?
    """
    idx = df.groupby(['representation', 'sigma'])['recovery_rate'].idxmax()
    best = df.loc[idx][['representation', 'rep_type', 'sigma', 'method', 
                        'distance_metric', 'weighted', 'recovery_rate',
                        'r2_clean', 'r2_noisy', 'r2_cleaned']]
    
    return best.sort_values(['rep_type', 'representation', 'sigma'])


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_hybrid_vs_base(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 1: The Key Result - Hybrid vs Base Recovery Rates
    
    Box/violin plot comparing recovery rates for hybrid vs base representations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, sigma in enumerate(df['sigma'].unique()):
        ax = axes[i]
        df_sigma = df[df['sigma'] == sigma]
        
        # Violin plot
        parts = ax.violinplot(
            [df_sigma[df_sigma['rep_type'] == 'base']['recovery_rate'].values,
             df_sigma[df_sigma['rep_type'] == 'hybrid']['recovery_rate'].values],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        
        # Color the violins
        parts['bodies'][0].set_facecolor(COLORS['base'])
        parts['bodies'][1].set_facecolor(COLORS['hybrid'])
        
        # Add individual points
        for j, rep_type in enumerate(['base', 'hybrid']):
            data = df_sigma[df_sigma['rep_type'] == rep_type]['recovery_rate']
            x = np.random.normal(j, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.3, s=20, 
                      color=COLORS[rep_type], edgecolor='white', linewidth=0.5)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base\nRepresentations', 'Hybrid\nRepresentations'])
        ax.set_ylabel('Recovery Rate')
        ax.set_title(f'σ = {sigma}', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add mean annotations
        for j, rep_type in enumerate(['base', 'hybrid']):
            mean_val = df_sigma[df_sigma['rep_type'] == rep_type]['recovery_rate'].mean()
            ax.annotate(f'μ = {mean_val:+.1%}', xy=(j, mean_val), 
                       xytext=(j + 0.3, mean_val),
                       fontsize=11, fontweight='bold')
    
    fig.suptitle('Hybrid vs Base Representations for Noise Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 2: Method × Representation Heatmap
    
    Shows which methods work best for which representations.
    """
    # Pivot: method vs representation, value = mean recovery
    pivot = df.pivot_table(
        index='method',
        columns='representation',
        values='recovery_rate',
        aggfunc='mean'
    )
    
    # Order methods by overall performance
    method_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[method_order]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Recovery Rate'})
    
    ax.set_xlabel('Representation', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Recovery Rate: Method × Representation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distance_metric_comparison(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 3: Distance Metric Performance
    
    Bar chart comparing different distance metrics.
    """
    df_dist = df[df['method'].isin(DISTANCE_AWARE_METHODS)].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: By metric (averaged over weighted/unweighted)
    ax = axes[0]
    metric_perf = df_dist.groupby('distance_metric')['recovery_rate'].agg(['mean', 'std'])
    metric_perf = metric_perf.sort_values('mean', ascending=True)
    
    colors = [COLORS['best'] if m == metric_perf['mean'].idxmax() else COLORS['base'] 
              for m in metric_perf.index]
    
    ax.barh(range(len(metric_perf)), metric_perf['mean'], 
            xerr=metric_perf['std'], color=colors, capsize=3)
    ax.set_yticks(range(len(metric_perf)))
    ax.set_yticklabels(metric_perf.index)
    ax.set_xlabel('Mean Recovery Rate')
    ax.set_title('Performance by Distance Metric', fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Right: Weighted vs Unweighted for each metric
    ax = axes[1]
    pivot = df_dist.pivot_table(index='distance_metric', columns='weighted',
                                 values='recovery_rate', aggfunc='mean')
    pivot = pivot.sort_values(True, ascending=True)
    
    x = np.arange(len(pivot))
    width = 0.35
    
    ax.barh(x - width/2, pivot[False], width, label='Unweighted', color=COLORS['base'])
    ax.barh(x + width/2, pivot[True], width, label='Weighted', color=COLORS['hybrid'])
    
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Mean Recovery Rate')
    ax.set_title('Weighted vs Unweighted', fontweight='bold')
    ax.legend()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_scatter(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 4: Precision vs Recall Scatter
    
    Shows the precision-recall tradeoff for different methods.
    """
    if 'precision' not in df.columns or 'recall' not in df.columns:
        print("Precision/recall columns not found, skipping this plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Aggregate by method
    method_perf = df.groupby('method').agg({
        'precision': 'mean',
        'recall': 'mean',
        'recovery_rate': 'mean'
    }).reset_index()
    
    # Color by category
    def get_color(method):
        if method in AC_AWARE_METHODS:
            return COLORS['ac_aware']
        for cat, methods in METHOD_CATEGORIES.items():
            if method in methods:
                if cat == 'baseline':
                    return COLORS['baseline']
                elif 'distance' in cat:
                    return COLORS['base']
                else:
                    return COLORS['hybrid']
        return 'gray'
    
    colors = [get_color(m) for m in method_perf['method']]
    sizes = 100 + 500 * (method_perf['recovery_rate'] - method_perf['recovery_rate'].min()) / \
            (method_perf['recovery_rate'].max() - method_perf['recovery_rate'].min() + 1e-10)
    
    scatter = ax.scatter(method_perf['recall'], method_perf['precision'],
                        c=colors, s=sizes, alpha=0.7, edgecolor='white', linewidth=2)
    
    # Annotate points
    for _, row in method_perf.iterrows():
        ax.annotate(row['method'], (row['recall'], row['precision']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.set_title('Noise Detection: Precision vs Recall\n(size = recovery rate)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ac_aware'], label='AC-Aware'),
        Patch(facecolor=COLORS['base'], label='Distance-based'),
        Patch(facecolor=COLORS['hybrid'], label='Ensemble/Model'),
        Patch(facecolor=COLORS['baseline'], label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_r2_trajectory(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 5: R² Trajectory (Clean → Noisy → Cleaned)
    
    Shows how well each representation recovers from noise.
    """
    # Get best config per representation
    idx = df.groupby(['representation', 'sigma'])['recovery_rate'].idxmax()
    best = df.loc[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, sigma in enumerate(sorted(df['sigma'].unique())):
        ax = axes[i]
        df_sigma = best[best['sigma'] == sigma].sort_values('r2_clean', ascending=False)
        
        x = np.arange(len(df_sigma))
        width = 0.25
        
        ax.bar(x - width, df_sigma['r2_clean'], width, label='Clean', color=COLORS['hybrid'])
        ax.bar(x, df_sigma['r2_noisy'], width, label='Noisy', color=COLORS['baseline'])
        ax.bar(x + width, df_sigma['r2_cleaned'], width, label='Cleaned', color=COLORS['best'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_sigma['representation'], rotation=45, ha='right')
        ax.set_ylabel('R²')
        ax.set_title(f'σ = {sigma}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
    
    fig.suptitle('R² Trajectory: Clean → Noisy → Cleaned (Best Config per Rep)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_category_boxplot(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Figure 6: Method Category Comparison
    
    Box plots comparing different categories of methods.
    """
    df = df.copy()
    
    def get_category(method):
        for cat, methods in METHOD_CATEGORIES.items():
            if method in methods:
                return cat
        return 'other'
    
    df['category'] = df['method'].apply(get_category)
    
    # Order categories by mean performance
    cat_order = df.groupby('category')['recovery_rate'].mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=df, x='category', y='recovery_rate', order=cat_order,
                palette='viridis', ax=ax)
    sns.stripplot(data=df, x='category', y='recovery_rate', order=cat_order,
                  color='black', alpha=0.3, size=3, ax=ax)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Method Category', fontsize=12)
    ax.set_ylabel('Recovery Rate', fontsize=12)
    ax.set_title('Performance by Method Category', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_full_analysis(results_path: Path, output_dir: Optional[Path] = None):
    """
    Run complete analysis pipeline.
    
    Generates all tables and figures, saves to output_dir if provided.
    """
    print("="*80)
    print("NOISE ROBUSTNESS EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📊 Loading results...")
    df = load_results(results_path)
    print(f"   Loaded {len(df)} experiment results")
    print(f"   Representations: {df['representation'].nunique()}")
    print(f"   Methods: {df['method'].nunique()}")
    print(f"   Noise levels: {sorted(df['sigma'].unique())}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # TABLES
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)
    
    # Table 1: Overall Rankings
    print("\n📋 TABLE 1: Top 15 Configurations (by Recovery Rate)")
    print("-"*60)
    t1 = table_overall_rankings(df, top_n=15)
    print(t1.to_string(index=False))
    if output_dir:
        t1.to_csv(output_dir / 'table1_top_configurations.csv', index=False)
    
    # Table 2: Representation Comparison (THE KEY TABLE)
    print("\n📋 TABLE 2: Hybrid vs Base Representation Performance")
    print("-"*60)
    t2 = table_representation_comparison(df)
    print(t2.to_string(index=False))
    if output_dir:
        t2.to_csv(output_dir / 'table2_representation_comparison.csv', index=False)
    
    # Statistical test: hybrid vs base
    hybrid_recovery = df[df['rep_type'] == 'hybrid']['recovery_rate']
    base_recovery = df[df['rep_type'] == 'base']['recovery_rate']
    print(f"\n   ⭐ KEY RESULT:")
    print(f"      Hybrid mean recovery: {hybrid_recovery.mean():+.2%} (±{hybrid_recovery.std():.2%})")
    print(f"      Base mean recovery:   {base_recovery.mean():+.2%} (±{base_recovery.std():.2%})")
    print(f"      Difference:           {hybrid_recovery.mean() - base_recovery.mean():+.2%}")
    
    # Table 3: Method Comparison
    print("\n📋 TABLE 3: Method Performance Comparison")
    print("-"*60)
    t3 = table_method_comparison(df)
    print(t3.to_string(index=False))
    if output_dir:
        t3.to_csv(output_dir / 'table3_method_comparison.csv', index=False)
    
    # Table 4: Distance Metric Comparison
    print("\n📋 TABLE 4: Distance Metric Performance")
    print("-"*60)
    t4 = table_distance_metric_comparison(df)
    print(t4.to_string(index=False))
    if output_dir:
        t4.to_csv(output_dir / 'table4_distance_metric_comparison.csv', index=False)
    
    # Table 5: Weighting Effect
    print("\n📋 TABLE 5: Effect of Feature Weighting")
    print("-"*60)
    t5 = table_weighting_effect(df)
    print(t5.to_string(index=False))
    if output_dir:
        t5.to_csv(output_dir / 'table5_weighting_effect.csv', index=False)
    
    # Table 6: AC-Aware Comparison
    print("\n📋 TABLE 6: AC-Aware vs Standard k-NN Methods")
    print("-"*60)
    t6 = table_ac_aware_comparison(df)
    print(t6.to_string(index=False))
    if output_dir:
        t6.to_csv(output_dir / 'table6_ac_aware_comparison.csv', index=False)
    
    ac_advantage = t6['ac_advantage'].mean()
    print(f"\n   ⭐ AC-AWARE ADVANTAGE: {ac_advantage:+.2%} mean improvement over naive k-NN")
    
    # Table 7: Best per Representation
    print("\n📋 TABLE 7: Best Configuration Per Representation")
    print("-"*60)
    t7 = table_best_per_representation(df)
    print(t7.to_string(index=False))
    if output_dir:
        t7.to_csv(output_dir / 'table7_best_per_representation.csv', index=False)
    
    # ========================================================================
    # FIGURES
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    print("\n📈 Figure 1: Hybrid vs Base (Key Result)")
    plot_hybrid_vs_base(df, output_dir / 'fig1_hybrid_vs_base.png' if output_dir else None)
    
    print("\n📈 Figure 2: Method × Representation Heatmap")
    plot_method_heatmap(df, output_dir / 'fig2_method_heatmap.png' if output_dir else None)
    
    print("\n📈 Figure 3: Distance Metric Comparison")
    plot_distance_metric_comparison(df, output_dir / 'fig3_distance_metrics.png' if output_dir else None)
    
    print("\n📈 Figure 4: Precision vs Recall")
    plot_precision_recall_scatter(df, output_dir / 'fig4_precision_recall.png' if output_dir else None)
    
    print("\n📈 Figure 5: R² Trajectory")
    plot_r2_trajectory(df, output_dir / 'fig5_r2_trajectory.png' if output_dir else None)
    
    print("\n📈 Figure 6: Method Category Comparison")
    plot_method_category_boxplot(df, output_dir / 'fig6_method_categories.png' if output_dir else None)
    
    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    # Best overall config
    best_idx = df['recovery_rate'].idxmax()
    best = df.loc[best_idx]
    print(f"""
    🏆 BEST CONFIGURATION:
       Representation: {best['representation']} ({best['rep_type']})
       Method:         {best['method']}
       Distance:       {best['distance_metric']}
       Weighted:       {best['weighted']}
       Recovery:       {best['recovery_rate']:+.1%}
       R² Cleaned:     {best['r2_cleaned']:.4f}
    
    📊 KEY FINDINGS:
       1. Hybrid vs Base: {hybrid_recovery.mean() - base_recovery.mean():+.2%} advantage for hybrids
       2. Best Method:    {t3.iloc[0]['method']} ({t3.iloc[0]['recovery_mean']:+.2%} mean recovery)
       3. Best Metric:    {t4.iloc[0]['distance_metric']} ({t4.iloc[0]['recovery_mean']:+.2%} mean recovery)
       4. AC-Aware:       {ac_advantage:+.2%} advantage over naive k-NN
       5. Weighting:      {'Helps' if t5[t5['metric'] == 'Mean improvement']['value'].values[0][0] == '+' else 'Hurts'}
    """)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze noise robustness experiment results')
    parser.add_argument('results', type=Path, help='Path to results CSV or pickle file')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output directory for tables and figures')
    
    args = parser.parse_args()
    
    run_full_analysis(args.results, args.output)