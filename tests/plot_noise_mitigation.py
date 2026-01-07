#!/usr/bin/env python3
"""
Visualize Noise Robustness Results
===================================

Generate plots and summary statistics from noise mitigation experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
sns.set_palette('Set2')


def load_results(results_dir='results/noise_robustness'):
    """Load all results from experiments"""
    results_dir = Path(results_dir)
    
    datasets = {}
    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name.upper()
            all_file = dataset_dir / 'all_results.csv'
            if all_file.exists():
                datasets[dataset_name] = pd.read_csv(all_file)
                print(f"Loaded {len(datasets[dataset_name])} results for {dataset_name}")
    
    return datasets


def plot_recovery_rate_comparison(datasets, output_dir):
    """
    Plot recovery rate across representations and noise levels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, df in datasets.items():
        
        # Average across seeds and methods
        summary = df.groupby(['representation', 'sigma']).agg({
            'recovery_rate': ['mean', 'std']
        }).reset_index()
        
        summary.columns = ['representation', 'sigma', 'recovery_mean', 'recovery_std']
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for rep in ['pdv', 'mhggnn', 'hybrid']:
            rep_data = summary[summary['representation'] == rep]
            ax.plot(rep_data['sigma'], rep_data['recovery_mean'], 
                   marker='o', label=rep.upper(), linewidth=2, markersize=8)
            ax.fill_between(rep_data['sigma'],
                           rep_data['recovery_mean'] - rep_data['recovery_std'],
                           rep_data['recovery_mean'] + rep_data['recovery_std'],
                           alpha=0.2)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='No recovery')
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Full recovery')
        
        ax.set_xlabel('Noise Level (σ)', fontsize=12)
        ax.set_ylabel('Performance Recovery Rate', fontsize=12)
        ax.set_title(f'{dataset_name}: Average Recovery Rate Across All Methods', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_recovery_rate.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved recovery rate plot for {dataset_name}")


def plot_cleaning_accuracy(datasets, output_dir):
    """
    Plot F1 score (cleaning accuracy) across representations.
    """
    output_dir = Path(output_dir)
    
    for dataset_name, df in datasets.items():
        
        # Average across seeds and methods
        summary = df.groupby(['representation', 'sigma']).agg({
            'f1': ['mean', 'std'],
            'precision': 'mean',
            'recall': 'mean'
        }).reset_index()
        
        summary.columns = ['representation', 'sigma', 'f1_mean', 'f1_std', 'precision', 'recall']
        
        # Plot F1
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for rep in ['pdv', 'mhggnn', 'hybrid']:
            rep_data = summary[summary['representation'] == rep]
            ax.plot(rep_data['sigma'], rep_data['f1_mean'], 
                   marker='o', label=rep.upper(), linewidth=2, markersize=8)
            ax.fill_between(rep_data['sigma'],
                           rep_data['f1_mean'] - rep_data['f1_std'],
                           rep_data['f1_mean'] + rep_data['f1_std'],
                           alpha=0.2)
        
        ax.set_xlabel('Noise Level (σ)', fontsize=12)
        ax.set_ylabel('F1 Score (Cleaning Accuracy)', fontsize=12)
        ax.set_title(f'{dataset_name}: Average F1 Score Across All Methods', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_f1_score.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved F1 score plot for {dataset_name}")


def plot_method_comparison(datasets, output_dir, sigma=0.6):
    """
    Compare methods at high noise level.
    """
    output_dir = Path(output_dir)
    
    for dataset_name, df in datasets.items():
        
        # Filter to high noise
        high_noise = df[df['sigma'] == sigma]
        
        # Average across seeds
        method_summary = high_noise.groupby(['representation', 'method']).agg({
            'recovery_rate': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, rep in enumerate(['pdv', 'mhggnn', 'hybrid']):
            rep_data = method_summary[method_summary['representation'] == rep]
            rep_data = rep_data.sort_values('recovery_rate', ascending=False).head(10)
            
            ax = axes[i]
            
            # Plot recovery rate bars
            bars = ax.barh(range(len(rep_data)), rep_data['recovery_rate'], alpha=0.7)
            ax.set_yticks(range(len(rep_data)))
            ax.set_yticklabels(rep_data['method'], fontsize=9)
            ax.set_xlabel('Recovery Rate', fontsize=10)
            ax.set_title(f'{rep.upper()}', fontsize=12, fontweight='bold')
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Color bars by F1 score
            norm = plt.Normalize(vmin=0, vmax=1)
            cmap = plt.cm.RdYlGn
            for bar, f1 in zip(bars, rep_data['f1']):
                bar.set_color(cmap(norm(f1)))
        
        fig.suptitle(f'{dataset_name}: Top 10 Methods by Recovery Rate (σ={sigma})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_method_comparison.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved method comparison for {dataset_name}")


def plot_distance_metric_comparison(datasets, output_dir, sigma=0.6):
    """
    Compare distance metrics for distance-based methods.
    """
    output_dir = Path(output_dir)
    
    distance_methods = ['knn_k5', 'knn_k10', 'activity_cliffs']
    
    for dataset_name, df in datasets.items():
        
        # Filter to high noise and distance-based methods
        high_noise = df[
            (df['sigma'] == sigma) & 
            (df['method'].isin(distance_methods)) &
            (df['distance_metric'] != 'none')
        ]
        
        # Average across seeds
        metric_summary = high_noise.groupby(['representation', 'distance_metric']).agg({
            'recovery_rate': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, metric_name in enumerate(['recovery_rate', 'f1']):
            ax = axes[i]
            
            pivot = metric_summary.pivot(index='distance_metric', 
                                        columns='representation', 
                                        values=metric_name)
            
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xlabel('Distance Metric', fontsize=11)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(title='Representation', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            if metric_name == 'recovery_rate':
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        fig.suptitle(f'{dataset_name}: Distance Metric Comparison (σ={sigma})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_distance_metrics.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved distance metric comparison for {dataset_name}")


def plot_weighting_impact(datasets, output_dir, sigma=0.6):
    """
    Compare weighted vs unweighted for distance-based methods.
    """
    output_dir = Path(output_dir)
    
    distance_methods = ['knn_k5', 'knn_k10', 'activity_cliffs']
    
    for dataset_name, df in datasets.items():
        
        # Filter to high noise and distance-based methods
        high_noise = df[
            (df['sigma'] == sigma) & 
            (df['method'].isin(distance_methods)) &
            (df['distance_metric'] != 'none')
        ]
        
        # Average across seeds and distance metrics
        weight_summary = high_noise.groupby(['representation', 'weighted']).agg({
            'recovery_rate': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, metric_name in enumerate(['recovery_rate', 'f1']):
            ax = axes[i]
            
            for rep in ['pdv', 'mhggnn', 'hybrid']:
                rep_data = weight_summary[weight_summary['representation'] == rep]
                
                unweighted = rep_data[rep_data['weighted'] == False][metric_name].values[0] if len(rep_data[rep_data['weighted'] == False]) > 0 else 0
                weighted = rep_data[rep_data['weighted'] == True][metric_name].values[0] if len(rep_data[rep_data['weighted'] == True]) > 0 else 0
                
                x = ['Unweighted', 'Weighted']
                y = [unweighted, weighted]
                
                ax.plot(x, y, marker='o', label=rep.upper(), linewidth=2, markersize=10)
            
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            if metric_name == 'recovery_rate':
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        fig.suptitle(f'{dataset_name}: Feature Weighting Impact (σ={sigma})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_weighting_impact.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved weighting impact plot for {dataset_name}")


def generate_summary_table(datasets, output_dir):
    """
    Generate summary tables.
    """
    output_dir = Path(output_dir)
    
    for dataset_name, df in datasets.items():
        
        # Best method per representation at each noise level
        best_methods = []
        
        for sigma in df['sigma'].unique():
            sigma_data = df[df['sigma'] == sigma]
            
            for rep in ['pdv', 'mhggnn', 'hybrid']:
                rep_data = sigma_data[sigma_data['representation'] == rep]
                
                # Get best by recovery rate
                best_by_recovery = rep_data.nlargest(1, 'recovery_rate').iloc[0]
                
                best_methods.append({
                    'dataset': dataset_name,
                    'representation': rep,
                    'sigma': sigma,
                    'best_method': best_by_recovery['method'],
                    'distance_metric': best_by_recovery['distance_metric'],
                    'weighted': best_by_recovery['weighted'],
                    'recovery_rate': best_by_recovery['recovery_rate'],
                    'f1': best_by_recovery['f1']
                })
        
        best_df = pd.DataFrame(best_methods)
        best_df.to_csv(output_dir / f'{dataset_name.lower()}_best_methods.csv', index=False)
        
        print(f"✓ Saved best methods table for {dataset_name}")


def print_key_findings(datasets):
    """
    Print key findings from experiments.
    """
    
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    
    for dataset_name, df in datasets.items():
        
        print(f"\n{dataset_name}:")
        print("-"*100)
        
        # Best overall method
        high_noise = df[df['sigma'] == 0.6]
        
        for rep in ['pdv', 'mhggnn', 'hybrid']:
            rep_data = high_noise[high_noise['representation'] == rep]
            best = rep_data.nlargest(1, 'recovery_rate').iloc[0]
            
            print(f"\n  {rep.upper()}:")
            print(f"    Best method: {best['method']} "
                  f"({best['distance_metric']}, weighted={best['weighted']})")
            print(f"    Recovery: {best['recovery_rate']:+.2%}")
            print(f"    F1: {best['f1']:.3f}")
        
        # Representation comparison
        print(f"\n  Representation Comparison (σ=0.6, averaged across all methods):")
        rep_avg = high_noise.groupby('representation').agg({
            'recovery_rate': 'mean',
            'f1': 'mean'
        })
        
        for rep in ['pdv', 'mhggnn', 'hybrid']:
            print(f"    {rep.upper():10s}: Recovery={rep_avg.loc[rep, 'recovery_rate']:+.2%}, "
                  f"F1={rep_avg.loc[rep, 'f1']:.3f}")
        
        # Distance metric comparison
        distance_data = high_noise[
            (high_noise['distance_metric'] != 'none')
        ]
        
        if len(distance_data) > 0:
            print(f"\n  Distance Metric Comparison (distance-based methods, σ=0.6):")
            metric_avg = distance_data.groupby('distance_metric').agg({
                'recovery_rate': 'mean',
                'f1': 'mean'
            })
            
            for metric in ['euclidean', 'manhattan', 'cosine']:
                if metric in metric_avg.index:
                    print(f"    {metric:12s}: Recovery={metric_avg.loc[metric, 'recovery_rate']:+.2%}, "
                          f"F1={metric_avg.loc[metric, 'f1']:.3f}")
        
        # Weighting impact
        weight_data = high_noise[
            (high_noise['distance_metric'] != 'none')
        ]
        
        if len(weight_data) > 0:
            print(f"\n  Feature Weighting Impact (distance-based methods, σ=0.6):")
            weight_avg = weight_data.groupby('weighted').agg({
                'recovery_rate': 'mean',
                'f1': 'mean'
            })
            
            for weighted in [False, True]:
                if weighted in weight_avg.index:
                    label = "Weighted" if weighted else "Unweighted"
                    print(f"    {label:12s}: Recovery={weight_avg.loc[weighted, 'recovery_rate']:+.2%}, "
                          f"F1={weight_avg.loc[weighted, 'f1']:.3f}")


def main():
    
    results_dir = 'results/noise_robustness'
    output_dir = 'results/noise_robustness/plots'
    
    print("="*100)
    print("VISUALIZING NOISE ROBUSTNESS RESULTS")
    print("="*100)
    
    # Load results
    datasets = load_results(results_dir)
    
    if not datasets:
        print("ERROR: No results found!")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    plot_recovery_rate_comparison(datasets, output_dir)
    plot_cleaning_accuracy(datasets, output_dir)
    plot_method_comparison(datasets, output_dir)
    plot_distance_metric_comparison(datasets, output_dir)
    plot_weighting_impact(datasets, output_dir)
    
    # Generate tables
    print("\nGenerating summary tables...")
    generate_summary_table(datasets, output_dir)
    
    # Print key findings
    print_key_findings(datasets)
    
    print("\n" + "="*100)
    print("VISUALIZATION COMPLETE")
    print("="*100)
    print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()