"""
Phase 0 Analysis - Global Screening (Extended with Graphs)
===========================================================

Analyzes:
1. Original phase0c screening data (from ../../qsar_qm_models/results)
2. NEW: Graph models from phase1_graphs_updated (from ../results)

Generates robustness metrics for all model-representation pairs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
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
    
    print(f"Final aggregated data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    
    return results


def load_new_graph_data(results_dir="../results"):
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
    
    print(f"Final aggregated data: {len(results)} rows")
    print(f"Models: {sorted(results['model'].unique())}")
    
    return results


def combine_all_screening_data(old_results_dir="../../qsar_qm_models/results", 
                               new_results_dir="../results"):
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
# ROBUSTNESS METRICS
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.6):
    """Calculate robustness metrics"""
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
        
        # Baseline
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # High noise
        sigma_h = group[np.abs(group['sigma'] - sigma_high) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Retention
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            if metrics['baseline_r2'] != 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # NSI
        if len(group) >= 3:
            try:
                slope_r2, intercept_r2, r_val, p_val, _ = stats.linregress(
                    group['sigma'], group['r2']
                )
                metrics['nsi_r2'] = slope_r2
                metrics['nsi_r2_pval'] = p_val
                
                if intercept_r2 != 0:
                    metrics['nsi_r2_relative'] = slope_r2 / abs(intercept_r2)
                else:
                    metrics['nsi_r2_relative'] = np.nan
                
            except:
                metrics['nsi_r2'] = np.nan
                metrics['nsi_r2_relative'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
            metrics['nsi_r2_relative'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


def define_robustness_score(metrics_df):
    """Define composite robustness score"""
    ret_normalized = (metrics_df['retention_pct'] - metrics_df['retention_pct'].min()) / \
                     (metrics_df['retention_pct'].max() - metrics_df['retention_pct'].min())
    
    nsi_abs = metrics_df['nsi_r2'].abs()
    nsi_normalized = 1 - ((nsi_abs - nsi_abs.min()) / (nsi_abs.max() - nsi_abs.min()))
    
    metrics_df['robustness_score'] = (ret_normalized + nsi_normalized) / 2
    
    return metrics_df


# ============================================================================
# MAIN
# ============================================================================

def main(old_results_dir="../../qsar_qm_models/results", 
         new_results_dir="../results"):
    """Main execution"""
    print("="*80)
    print("PHASE 0 ANALYSIS - GLOBAL SCREENING (EXTENDED WITH GRAPHS)")
    print("="*80)
    
    # Load and combine all data
    df = combine_all_screening_data(old_results_dir, new_results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    metrics_df = define_robustness_score(metrics_df)
    
    # Create output directory
    output_dir = Path(new_results_dir) / "phase0_extended"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase0_robustness_metrics_extended.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase0_robustness_metrics_extended.csv'}")
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 0 ANALYSIS COMPLETE (EXTENDED)")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    print("\nTop 10 Most Robust Configurations (including graphs):")
    for idx, (_, row) in enumerate(metrics_df.nlargest(10, 'robustness_score').iterrows(), 1):
        print(f"  {idx}. {row['model']}/{row['representation']}: "
              f"Score={row['robustness_score']:.3f}, "
              f"R²₀={row['baseline_r2']:.3f}, "
              f"Retention={row['retention_pct']:.1f}%")
    
    print("\nRepresentation Rankings (by median retention):")
    rep_ranking = metrics_df.groupby('representation')['retention_pct'].median().sort_values(ascending=False)
    for idx, (rep, val) in enumerate(rep_ranking.items(), 1):
        print(f"  {idx}. {rep.upper()}: {val:.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        old_dir = sys.argv[1]
        new_dir = sys.argv[2]
        main(old_dir, new_dir)
    elif len(sys.argv) > 1:
        # If only one arg, assume it's new_dir
        main(new_results_dir=sys.argv[1])
    else:
        main()