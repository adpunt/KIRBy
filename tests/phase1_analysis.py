"""
Phase 1 Analysis - Deterministic vs Probabilistic (UPDATED)
===========================================================

Combines:
1. Old phase1a/b/c data (RF/QRF, XGBoost/NGBoost, DNN/BNN on Binary PDV)
2. New representations (same pairs on PDV, MHG-GNN)
3. Graph models (GCN/GAT/GIN/MPNN vs their Bayesian variants)

Generates Figure 3 and Supplementary S4
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
COLORS = {
    'deterministic': '#0173B2',
    'probabilistic': '#DE8F05',
}

REPRESENTATION_COLORS = {
    'binary_pdv': '#0173B2',
    'pdv': '#56B4E9',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'mhg_gnn': '#CC78BC',
    'graph': '#949494',
}

# ============================================================================
# UTILITY FUNCTIONS  
# ============================================================================

def format_representation(rep):
    """Format representation name for display"""
    mapping = {
        'binary_pdv': 'Binary PDV',
        'pdv': 'PDV',
        'sns': 'SNS',
        'ecfp4': 'ECFP4',
        'mhg_gnn': 'MHG-GNN',
        'graph': 'Graph'
    }
    return mapping.get(rep, rep)

def format_model(model):
    """Format model name for display"""
    return model.replace('_', ' ').upper()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_old_phase1_data(results_dir="../../qsar_qm_models/results"):
    """Load original phase1a/b/c data"""
    print("\n" + "="*80)
    print("LOADING OLD PHASE 1 DATA (phase1a/b/c)")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    phase1_files = list(results_dir.glob("phase1a_*.csv")) + \
                   list(results_dir.glob("phase1b_*.csv")) + \
                   list(results_dir.glob("phase1c_*.csv"))
    
    # Exclude per_epoch and uncertainty files
    phase1_files = [f for f in phase1_files if 'per_epoch' not in f.name and 'uncertainty' not in f.name]
    
    if not phase1_files:
        print("WARNING: No phase1a/b/c files found!")
        return pd.DataFrame()
    
    print(f"Found {len(phase1_files)} files")
    
    all_data = []
    for filepath in phase1_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            
            # Parse from filename
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                phase = parts[0]  # phase1a, phase1b, phase1c
                rep = parts[1] if len(parts) > 1 else 'unknown'
                model = '_'.join(parts[2:])
                
                if 'model' not in df.columns:
                    df['model'] = model
                if 'rep' not in df.columns:
                    df['rep'] = rep
                df['phase'] = phase
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Warning: {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Rename pdv to binary_pdv
    combined_df['rep'] = combined_df['rep'].replace('pdv', 'binary_pdv')
    
    print(f"Loaded {len(combined_df)} rows")
    print(f"Unique models: {combined_df['model'].unique()}")
    print(f"Unique reps: {combined_df['rep'].unique()}")
    
    return combined_df


def load_new_phase1_data(results_dir="results"):
    """Load new phase1 data (continuous_pdv and mhg_gnn)"""
    print("\n" + "="*80)
    print("LOADING NEW PHASE 1 DATA (PDV, MHG-GNN)")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    new_files = list(results_dir.glob("phase1_continuous_pdv_*.csv")) + \
                list(results_dir.glob("phase1_mhggnn_*.csv"))
    
    new_files = [f for f in new_files if 'per_epoch' not in f.name and 'uncertainty' not in f.name]
    
    if not new_files:
        print("WARNING: No new phase1 files found!")
        return pd.DataFrame()
    
    print(f"Found {len(new_files)} files")
    
    all_data = []
    for filepath in new_files:
        try:
            df = pd.read_csv(filepath)
            
            # Parse from filename
            parts = filepath.stem.split('_')
            
            if 'continuous_pdv' in filepath.name:
                rep = 'pdv'
                model_parts = parts[3:]
            elif 'mhggnn' in filepath.name:
                rep = 'mhg_gnn'
                model_parts = parts[2:]
            else:
                continue
            
            model = '_'.join(model_parts)
            
            if 'model' not in df.columns:
                df['model'] = model
            if 'rep' not in df.columns:
                df['rep'] = rep
            
            df['source_file'] = filepath.name
            all_data.append(df)
            
        except Exception as e:
            print(f"Warning: {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Loaded {len(combined_df)} rows")
    print(f"Unique models: {combined_df['model'].unique()}")
    print(f"Unique reps: {combined_df['rep'].unique()}")
    
    return combined_df


def load_graph_phase1_data(results_dir="results"):
    """Load graph deterministic and Bayesian models"""
    print("\n" + "="*80)
    print("LOADING GRAPH PHASE 1 DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Try new unified format first (phase1_graphs_updated)
    graph_dir_new = results_dir / "phase1_graphs_updated"
    
    all_data = []
    
    if graph_dir_new.exists():
        print(f"Loading from unified format: {graph_dir_new}")
        for seed_dir in sorted(graph_dir_new.glob("seed_*")):
            seed = seed_dir.name.split('_')[1]
            all_results = seed_dir / "all_results.csv"
            if all_results.exists():
                try:
                    df = pd.read_csv(all_results)
                    df['seed'] = int(seed)
                    df = df.rename(columns={'seed': 'iteration'})
                    
                    # Ensure model_type is present
                    if 'model_type' not in df.columns:
                        df['model_type'] = df['model'].apply(
                            lambda x: 'probabilistic' if 'bnn' in x.lower() else 'deterministic'
                        )
                    
                    # Add representation
                    if 'rep' not in df.columns and 'representation' not in df.columns:
                        df['representation'] = 'graph'
                    elif 'rep' in df.columns:
                        df = df.rename(columns={'rep': 'representation'})
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"Warning: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(combined)} rows from unified format")
            print(f"Unique models: {combined['model'].unique()}")
            return combined
    
    # Fallback to old format (qm9_graphs + qm9_graphs_uncertainty)
    print("Unified format not found, trying legacy format...")
    
    # Load deterministic graphs from robustness
    det_data = []
    graph_dir = results_dir / "qm9_graphs"
    
    if graph_dir.exists():
        for seed_dir in sorted(graph_dir.glob("seed_*")):
            seed = seed_dir.name.split('_')[1]
            all_results = seed_dir / "all_results.csv"
            if all_results.exists():
                try:
                    df = pd.read_csv(all_results)
                    df['seed'] = int(seed)
                    # Keep only deterministic models (not Graph-GP)
                    df = df[df['model'].isin(['GCN', 'GAT', 'GIN', 'MPNN'])]
                    det_data.append(df)
                except Exception as e:
                    print(f"Warning: {e}")
    
    # Load Bayesian graphs from uncertainty
    bnn_data = []
    unc_dir = results_dir / "qm9_graphs_uncertainty"
    
    if unc_dir.exists():
        for seed_dir in sorted(unc_dir.glob("seed_*")):
            seed = seed_dir.name.split('_')[1]
            unc_file = seed_dir / "uncertainty_values.csv"
            if unc_file.exists():
                try:
                    df = pd.read_csv(unc_file)
                    df['seed'] = int(seed)
                    bnn_data.append(df)
                except Exception as e:
                    print(f"Warning: {e}")
    
    # Combine deterministic
    if det_data:
        det_df = pd.concat(det_data, ignore_index=True)
        det_df = det_df.rename(columns={'seed': 'iteration'})
        det_df['model_type'] = 'deterministic'
        if 'representation' not in det_df.columns:
            det_df['representation'] = 'graph'
        print(f"Loaded {len(det_df)} deterministic graph rows")
    else:
        det_df = pd.DataFrame()
    
    # Process Bayesian (aggregate from per-sample to metrics)
    if bnn_data:
        bnn_df = pd.concat(bnn_data, ignore_index=True)
        
        # Aggregate to metrics
        metrics_list = []
        for (model, sigma, seed), group in bnn_df.groupby(['model', 'sigma', 'seed']):
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            mae = np.mean(np.abs(y_true - y_pred))
            
            metrics_list.append({
                'model': model,
                'representation': 'graph',
                'sigma': sigma,
                'iteration': seed,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'model_type': 'probabilistic'
            })
        
        bnn_metrics_df = pd.DataFrame(metrics_list)
        print(f"Loaded {len(bnn_metrics_df)} Bayesian graph rows")
    else:
        bnn_metrics_df = pd.DataFrame()
    
    # Combine
    all_graph = []
    if len(det_df) > 0:
        all_graph.append(det_df)
    if len(bnn_metrics_df) > 0:
        all_graph.append(bnn_metrics_df)
    
    if all_graph:
        combined = pd.concat(all_graph, ignore_index=True)
        print(f"Total graph data: {len(combined)} rows")
        return combined
    else:
        print("WARNING: No graph data found!")
        return pd.DataFrame()


def parse_model_type(df):
    """Parse model into deterministic/probabilistic type"""
    
    def get_model_type(model_name):
        model_lower = model_name.lower()
        
        # Probabilistic models
        prob_keywords = ['qrf', 'ngboost', 'gauche', 'bnn', 'gp']
        if any(kw in model_lower for kw in prob_keywords):
            return 'probabilistic'
        
        # Deterministic
        return 'deterministic'
    
    if 'model_type' not in df.columns:
        df['model_type'] = df['model'].apply(get_model_type)
    
    return df


def combine_phase1_data(old_results_dir="../../qsar_qm_models/results",
                       new_results_dir="results"):
    """Combine all Phase 1 data sources"""
    print("\n" + "="*80)
    print("COMBINING PHASE 1 DATA")
    print("="*80)
    
    old_data = load_old_phase1_data(old_results_dir)
    new_data = load_new_phase1_data(new_results_dir)
    graph_data = load_graph_phase1_data(new_results_dir)
    
    # Standardize column names BEFORE concatenating
    standardized_dfs = []
    for df in [old_data, new_data, graph_data]:
        if len(df) > 0:
            df = df.copy()  # Don't modify original
            
            # Standardize representation column
            if 'rep' in df.columns and 'representation' not in df.columns:
                df = df.rename(columns={'rep': 'representation'})
            elif 'rep' in df.columns and 'representation' in df.columns:
                # Both exist - drop 'rep', keep 'representation'
                df = df.drop(columns=['rep'])
            elif 'rep' not in df.columns and 'representation' not in df.columns:
                # Neither exists - skip this dataframe
                print(f"WARNING: Dataframe missing both 'rep' and 'representation' columns")
                continue
            
            standardized_dfs.append(df)
    
    if not standardized_dfs:
        print("ERROR: No Phase 1 data loaded!")
        return pd.DataFrame()
    
    combined = pd.concat(standardized_dfs, ignore_index=True)
    
    # Parse model types
    combined = parse_model_type(combined)
    
    # Filter catastrophic failures
    print(f"\nBefore filtering: {len(combined)} rows")
    combined = combined[combined['r2'] > -10]
    print(f"After R² > -10 filter: {len(combined)} rows")
    
    # Average across iterations
    print("\nAggregating across iterations...")
    results = combined.groupby(['model', 'representation', 'sigma', 'model_type']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
    }).reset_index()
    
    print(f"\nFinal data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    print(f"\nDeterministic models: {results[results['model_type']=='deterministic']['model'].unique()}")
    print(f"Probabilistic models: {results[results['model_type']=='probabilistic']['model'].unique()}")
    
    return results


def calculate_robustness_metrics(df, sigma_high=0.6):
    """Calculate robustness metrics"""
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS (σ_high = {sigma_high})")
    print("="*80)
    
    metrics_list = []
    
    for (model, rep, model_type), group in df.groupby(['model', 'representation', 'model_type']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'model': model,
            'representation': rep,
            'model_type': model_type,
        }
        
        # Baseline
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
        
        # High noise
        sigma_h = group[np.abs(group['sigma'] - sigma_high) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
        else:
            metrics['r2_high'] = np.nan
        
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
                slope_r2, _, _, p_val, _ = stats.linregress(group['sigma'], group['r2'])
                metrics['nsi_r2'] = slope_r2
                metrics['nsi_r2_pval'] = p_val
            except:
                metrics['nsi_r2'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


# ============================================================================
# FIGURE 3: DETERMINISTIC VS PROBABILISTIC
# ============================================================================

def create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir):
    """Figure 3: Deterministic vs probabilistic models"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: DETERMINISTIC VS PROBABILISTIC")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.12)
    
    # Panel A: Degradation curves for paired models
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Find most common representation
    available_reps = df['representation'].unique()
    if 'pdv' in available_reps:
        primary_rep = 'pdv'
    elif 'binary_pdv' in available_reps:
        primary_rep = 'binary_pdv'
    elif 'sns' in available_reps:
        primary_rep = 'sns'
    else:
        primary_rep = available_reps[0]
    
    print(f"Using representation: {format_representation(primary_rep)}")
    
    # Plot paired models
    pairs_to_plot = [
        ('rf', 'qrf', 'RF vs QRF'),
        ('xgboost', 'ngboost', 'XGBoost vs NGBoost'),
    ]
    
    for det_model, prob_model, label in pairs_to_plot:
        # Deterministic
        det_data = df[(df['model'].str.lower().str.contains(det_model, na=False)) & 
                      (df['representation'] == primary_rep) &
                      (df['model_type'] == 'deterministic')]
        
        if len(det_data) > 0:
            det_avg = det_data.groupby('sigma')['r2'].mean().reset_index()
            ax_a.plot(det_avg['sigma'], det_avg['r2'],
                     marker='o', linestyle='--', linewidth=2, alpha=0.7,
                     label=f'{det_model.upper()} (det)', color=COLORS['deterministic'])
        
        # Probabilistic
        prob_data = df[(df['model'].str.lower().str.contains(prob_model, na=False)) & 
                       (df['representation'] == primary_rep) &
                       (df['model_type'] == 'probabilistic')]
        
        if len(prob_data) > 0:
            prob_avg = prob_data.groupby('sigma')['r2'].mean().reset_index()
            ax_a.plot(prob_avg['sigma'], prob_avg['r2'],
                     marker='s', linestyle='-', linewidth=2, alpha=0.9,
                     label=f'{prob_model.upper()} (prob)', color=COLORS['probabilistic'])
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9)
    ax_a.set_ylabel('R² score', fontsize=9)
    ax_a.set_title(f'A. Degradation Curves ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=7, loc='best', ncol=2, frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: Baseline vs Retention scatter
    ax_b = fig.add_subplot(gs[0, 1])
    
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_df[metrics_df['model_type'] == model_type]
        
        if len(subset) > 0:
            color = COLORS[model_type]
            marker = 'o' if model_type == 'deterministic' else 's'
            
            ax_b.scatter(subset['baseline_r2'], subset['retention_pct'],
                        alpha=0.7, s=80, color=color, marker=marker,
                        label=model_type.capitalize(),
                        edgecolors='black', linewidth=0.8)
    
    ax_b.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                label='No degradation')
    
    ax_b.set_xlabel('Baseline R² (σ=0)', fontsize=9)
    ax_b.set_ylabel('Retention at high noise (%)', fontsize=9)
    ax_b.set_title('B. Baseline vs Robustness', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: BNN transformations (if available)
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Find DNN/BNN data
    bnn_data = metrics_df[metrics_df['model'].str.lower().str.contains('dnn|bnn', na=False)]
    
    if len(bnn_data) > 0:
        # Group by transformation type
        transforms = ['deterministic', 'probabilistic']
        transform_labels = ['Deterministic', 'Bayesian']
        
        # Find unique transformation variants
        bnn_variants = bnn_data['model'].unique()
        
        # Simplified bar plot
        x_pos = np.arange(len(bnn_variants))
        baseline_vals = []
        retention_vals = []
        
        for variant in bnn_variants:
            var_data = bnn_data[bnn_data['model'] == variant]
            baseline_vals.append(var_data['baseline_r2'].mean())
            retention_vals.append(var_data['retention_pct'].mean() / 100)
        
        width = 0.35
        ax_c.bar(x_pos - width/2, baseline_vals, width, label='Baseline R²',
                alpha=0.8, edgecolor='black', linewidth=0.5)
        ax_c.bar(x_pos + width/2, retention_vals, width, label='Retention (norm)',
                alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels([format_model(v) for v in bnn_variants], 
                            rotation=45, ha='right', fontsize=7)
        ax_c.set_ylabel('Score', fontsize=9)
        ax_c.set_title('C. DNN Variants Comparison', 
                      fontsize=10, fontweight='bold', pad=10)
        ax_c.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax_c.text(0.5, 0.5, 'No DNN/BNN data available',
                 ha='center', va='center', transform=ax_c.transAxes,
                 fontsize=10, style='italic')
        ax_c.axis('off')
    
    output_path = Path(output_dir) / "figure3_deterministic_vs_probabilistic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to {output_path}")
    plt.close()


def create_supplementary_s4(metrics_df, output_dir):
    """Supplementary S4: Pairwise difference plots"""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S4: PAIRWISE DIFFERENCES")
    print("="*80)
    
    # Define pairs
    pairs = [
        ('rf', 'qrf'),
        ('xgboost', 'ngboost'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for idx, (ax, metric) in enumerate(zip(axes, ['retention_pct', 'nsi_r2'])):
        differences = []
        labels = []
        colors_list = []
        
        for det_model, prob_model in pairs:
            for rep in metrics_df['representation'].unique():
                # Deterministic
                det = metrics_df[(metrics_df['model'].str.lower().str.contains(det_model, na=False)) &
                                (metrics_df['representation'] == rep) &
                                (metrics_df['model_type'] == 'deterministic')]
                
                # Probabilistic
                prob = metrics_df[(metrics_df['model'].str.lower().str.contains(prob_model, na=False)) &
                                 (metrics_df['representation'] == rep) &
                                 (metrics_df['model_type'] == 'probabilistic')]
                
                if len(det) > 0 and len(prob) > 0:
                    det_val = det[metric].mean()
                    prob_val = prob[metric].mean()
                    
                    if metric == 'nsi_r2':
                        diff = det_val - prob_val
                    else:
                        diff = prob_val - det_val
                    
                    differences.append(diff)
                    labels.append(f'{det_model} vs {prob_model}\n{format_representation(rep)}')
                    
                    if diff > 0:
                        colors_list.append(COLORS['probabilistic'])
                    else:
                        colors_list.append(COLORS['deterministic'])
        
        if differences:
            y_pos = np.arange(len(differences))
            
            ax.barh(y_pos, differences, color=colors_list, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            
            if metric == 'retention_pct':
                ax.set_xlabel('Retention difference (% points)', fontsize=9)
                ax.set_title('Retention % Gain\n(Probabilistic - Deterministic)', 
                           fontsize=10, fontweight='bold')
            else:
                ax.set_xlabel('NSI difference', fontsize=9)
                ax.set_title('NSI Improvement\n(Det - Prob)', 
                           fontsize=10, fontweight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "supplementary_s4_pairwise_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S4 to {output_path}")
    plt.close()


def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Deterministic vs Probabilistic summary
    table1 = metrics_df.groupby(['model', 'representation', 'model_type']).agg({
        'baseline_r2': 'mean',
        'r2_high': 'mean',
        'retention_pct': 'mean',
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table1 = table1.round(4)
    table1.to_csv(output_dir / "table_phase1_summary.csv", index=False)
    print(f"✓ Saved summary table")
    
    # Table 2: BNN transformations (if available)
    bnn_data = metrics_df[metrics_df['model'].str.lower().str.contains('dnn|bnn', na=False)]
    
    if len(bnn_data) > 0:
        table2 = bnn_data.groupby(['model', 'representation']).agg({
            'baseline_r2': 'mean',
            'r2_high': 'mean',
            'retention_pct': 'mean',
            'nsi_r2': lambda x: np.abs(x).mean()
        }).reset_index()
        
        table2 = table2.round(4)
        table2.to_csv(output_dir / "table_phase1_dnn_transforms.csv", index=False)
        print(f"✓ Saved DNN transformation table")


def perform_statistical_comparisons(df, metrics_df, output_dir):
    """Statistical comparisons between deterministic and probabilistic"""
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL COMPARISONS")
    print("="*80)
    
    results_text = []
    results_text.append("STATISTICAL COMPARISONS - PHASE 1")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Comparing deterministic vs probabilistic variants")
    results_text.append("Tests: Wilcoxon signed-rank (paired samples across noise levels)")
    results_text.append("")
    
    # Define pairs
    pairs = [
        ('rf', 'qrf', 'Random Forest'),
        ('xgboost', 'ngboost', 'Gradient Boosting'),
        ('dnn', 'bnn', 'Neural Network'),
    ]
    
    for det_model, prob_model, desc in pairs:
        results_text.append(f"\n{'='*80}")
        results_text.append(f"{desc.upper()}: {det_model} vs {prob_model}")
        results_text.append(f"{'='*80}")
        
        for rep in df['representation'].unique():
            results_text.append(f"\nRepresentation: {format_representation(rep)}")
            results_text.append("-"*80)
            
            # Get data
            det_data = df[(df['model'].str.lower().str.contains(det_model, na=False)) &
                         (df['representation'] == rep) &
                         (df['model_type'] == 'deterministic')]
            
            prob_data = df[(df['model'].str.lower().str.contains(prob_model, na=False)) &
                          (df['representation'] == rep) &
                          (df['model_type'] == 'probabilistic')]
            
            if len(det_data) == 0 or len(prob_data) == 0:
                results_text.append("  No data available for comparison")
                continue
            
            # Align by sigma
            det_avg = det_data.groupby('sigma')['r2'].mean()
            prob_avg = prob_data.groupby('sigma')['r2'].mean()
            
            common_sigmas = set(det_avg.index) & set(prob_avg.index)
            
            if len(common_sigmas) < 3:
                results_text.append("  Insufficient overlapping sigma values")
                continue
            
            det_vals = [det_avg[s] for s in sorted(common_sigmas)]
            prob_vals = [prob_avg[s] for s in sorted(common_sigmas)]
            
            # Wilcoxon signed-rank test
            try:
                stat, p_val = stats.wilcoxon(det_vals, prob_vals)
                
                mean_det = np.mean(det_vals)
                mean_prob = np.mean(prob_vals)
                
                results_text.append(f"  Mean R² (deterministic): {mean_det:.4f}")
                results_text.append(f"  Mean R² (probabilistic): {mean_prob:.4f}")
                results_text.append(f"  Wilcoxon statistic: {stat:.2f}")
                results_text.append(f"  p-value: {p_val:.6f}")
                
                if p_val < 0.05:
                    winner = 'probabilistic' if mean_prob > mean_det else 'deterministic'
                    results_text.append(f"  → Significant difference (p<0.05), {winner} superior")
                else:
                    results_text.append(f"  → No significant difference")
                
            except Exception as e:
                results_text.append(f"  Error in statistical test: {e}")
    
    # Save
    output_path = Path(output_dir) / "statistical_comparisons_phase1.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"✓ Saved statistical comparisons to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(old_results_dir="../../qsar_qm_models/results",
         new_results_dir="results"):
    """Main execution"""
    print("="*80)
    print("PHASE 1 ANALYSIS - DETERMINISTIC VS PROBABILISTIC (UPDATED)")
    print("="*80)
    
    # Load data
    df = combine_phase1_data(old_results_dir, new_results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    
    # Create output directory
    output_dir = Path(new_results_dir) / "phase1_figures_updated"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase1_robustness_metrics_all.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase1_robustness_metrics_all.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir)
    create_supplementary_s4(metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Statistical tests
    perform_statistical_comparisons(df, metrics_df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - figure3_deterministic_vs_probabilistic.png")
    print("    - supplementary_s4_pairwise_differences.png")
    print("  Tables:")
    print("    - table_phase1_summary.csv")
    print("    - table_phase1_dnn_transforms.csv (if applicable)")
    print("  Data:")
    print("    - phase1_robustness_metrics_all.csv")
    print("    - statistical_comparisons_phase1.txt")
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    print("\nDeterministic vs Probabilistic counts:")
    print(metrics_df.groupby('model_type').size())
    
    print("\nBy representation:")
    print(metrics_df.groupby(['representation', 'model_type']).size())
    
    # Identify and summarize pairs
    print("\n" + "="*80)
    print("DETERMINISTIC-PROBABILISTIC PAIRS FOUND")
    print("="*80)
    
    pairs = [
        ('rf', 'qrf'),
        ('xgboost', 'ngboost'),
        ('dnn', 'bnn'),
        ('gcn', 'gcn_bnn'),
        ('gat', 'gat_bnn'),
        ('gin', 'gin_bnn'),
        ('mpnn', 'mpnn_bnn'),
    ]
    
    for det, prob in pairs:
        det_configs = metrics_df[metrics_df['model'].str.lower().str.contains(det, na=False) & 
                                (metrics_df['model_type'] == 'deterministic')]
        prob_configs = metrics_df[metrics_df['model'].str.lower().str.contains(prob, na=False) &
                                 (metrics_df['model_type'] == 'probabilistic')]
        
        if len(det_configs) > 0 and len(prob_configs) > 0:
            print(f"\n{det.upper()} vs {prob.upper()}:")
            common_reps = set(det_configs['representation']) & set(prob_configs['representation'])
            print(f"  Common representations: {[format_representation(r) for r in common_reps]}")
            
            for rep in common_reps:
                det_ret = det_configs[det_configs['representation'] == rep]['retention_pct'].mean()
                prob_ret = prob_configs[prob_configs['representation'] == rep]['retention_pct'].mean()
                print(f"    {format_representation(rep)}: Det={det_ret:.1f}% vs Prob={prob_ret:.1f}%")


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