"""
Phase 1 Analysis - Deterministic vs Probabilistic (COMBINED)
============================================================

COMBINED VERSION:
- Loads OLD phase1a/b/c data from ../../qsar_qm_models/results
- Loads NEW representations + graph data from results/
- Generates ALL original figures with combined data

Key metrics (NO AUC):
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention percentage: (R²_high / R²_baseline) * 100
- Baseline R² at σ=0
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
# FORMATTING
# ============================================================================

def format_representation(rep):
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
    return model.replace('_', ' ').upper()

# ============================================================================
# DATA LOADING - FROM NEW SCRIPT
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
            
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                phase = parts[0]
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
    combined_df['rep'] = combined_df['rep'].replace('pdv', 'binary_pdv')
    
    print(f"Loaded {len(combined_df)} rows")
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
    return combined_df


def load_graph_phase1_data(results_dir="results"):
    """Load graph deterministic and Bayesian models"""
    print("\n" + "="*80)
    print("LOADING GRAPH PHASE 1 DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
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
                    
                    if 'model_type' not in df.columns:
                        df['model_type'] = df['model'].apply(
                            lambda x: 'probabilistic' if 'bnn' in x.lower() else 'deterministic'
                        )
                    
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
            return combined
    
    print("WARNING: No graph data found!")
    return pd.DataFrame()


def parse_model_type(df):
    """Parse model into deterministic/probabilistic type"""
    
    def get_model_type(model_name):
        model_lower = model_name.lower()
        prob_keywords = ['qrf', 'ngboost', 'gauche', 'bnn', 'gp']
        if any(kw in model_lower for kw in prob_keywords):
            return 'probabilistic'
        return 'deterministic'
    
    if 'model_type' not in df.columns:
        df['model_type'] = df['model'].apply(get_model_type)
    
    return df


def load_phase1_results(old_results_dir="../../qsar_qm_models/results",
                        new_results_dir="results"):
    """Combine all Phase 1 data sources"""
    print("\n" + "="*80)
    print("COMBINING PHASE 1 DATA")
    print("="*80)
    
    old_data = load_old_phase1_data(old_results_dir)
    graph_data = load_graph_phase1_data(new_results_dir)
    
    # REMOVED: new_data loading - those files don't exist
    
    standardized_dfs = []
    for df in [old_data, graph_data]:
        if len(df) > 0:
            df = df.copy()
            
            # Standardize representation column
            if 'rep' in df.columns and 'representation' not in df.columns:
                df = df.rename(columns={'rep': 'representation'})
            elif 'rep' in df.columns and 'representation' in df.columns:
                df = df.drop(columns=['rep'])
            elif 'rep' not in df.columns and 'representation' not in df.columns:
                print(f"WARNING: Dataframe missing both 'rep' and 'representation' columns")
                continue
            
            # CRITICAL: Parse model_type BEFORE concat
            df = parse_model_type(df)
            
            # Ensure required columns exist
            required_cols = ['model', 'representation', 'sigma', 'r2', 'rmse', 'mae', 'model_type']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"WARNING: Missing required columns {missing}, skipping this dataframe")
                continue
            
            print(f"  Adding {len(df)} rows with representations: {df['representation'].unique()}")
            standardized_dfs.append(df)
    
    if not standardized_dfs:
        print("ERROR: No Phase 1 data loaded!")
        return pd.DataFrame()
    
    combined = pd.concat(standardized_dfs, ignore_index=True)
    
    print(f"\nAfter concat: {len(combined)} rows")
    print(f"  Unique representations: {combined['representation'].unique()}")
    print(f"  Unique models: {combined['model'].nunique()}")
    
    combined = combined[combined['r2'] > -10]
    print(f"After R² > -10 filter: {len(combined)} rows")
    
    print("\nAggregating across iterations...")
    results = combined.groupby(['model', 'representation', 'sigma', 'model_type']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
    }).reset_index()
    
    print(f"\nFinal data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    print(f"Representations: {sorted(results['representation'].unique())}")
    
    return results

# ============================================================================
# METRICS CALCULATION - FROM OLD SCRIPT
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.6):
    """Calculate robustness metrics and filter outliers"""
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
    
    all_metrics = pd.DataFrame(metrics_list)
    print(f"Calculated metrics for {len(all_metrics)} configurations")
    
    # FILTER OUTLIERS
    print("\n" + "="*80)
    print("FILTERING OUTLIERS")
    print("="*80)
    
    outliers = all_metrics[
        (all_metrics['baseline_r2'] < 0.1) |
        (all_metrics['retention_pct'] < -50) |
        (all_metrics['retention_pct'] > 150)
    ].copy()
    
    if len(outliers) > 0:
        print(f"\nFound {len(outliers)} outlier configurations to exclude:")
        for _, row in outliers.iterrows():
            print(f"  - {row['model']}/{row['representation']}: "
                  f"baseline_r2={row['baseline_r2']:.3f}, "
                  f"retention={row['retention_pct']:.1f}%")
    else:
        print("\nNo outliers detected")
    
    metrics_df = all_metrics[
        (all_metrics['baseline_r2'] >= 0.1) &
        (all_metrics['retention_pct'] >= -50) &
        (all_metrics['retention_pct'] <= 150)
    ].copy()
    
    print(f"\nAfter filtering: {len(metrics_df)} configurations retained")
    print(f"Excluded: {len(outliers)} configurations")
    
    return metrics_df

# ============================================================================
# FIGURE GENERATION - ALL FROM OLD SCRIPT
# ============================================================================

def create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir):
    """Figure 3: Deterministic vs probabilistic models"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: DETERMINISTIC VS PROBABILISTIC")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.12)
    
    # Panel A: Degradation curves
    ax_a = fig.add_subplot(gs[0, 0])
    
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
    
    pairs_to_plot = [
        ('rf', 'qrf', 'RF vs QRF'),
        ('xgboost', 'ngboost', 'XGBoost vs NGBoost'),
    ]
    
    for det_model, prob_model, label in pairs_to_plot:
        det_data = df[(df['model'].str.lower().str.contains(det_model, na=False)) & 
                      (df['representation'] == primary_rep) &
                      (df['model_type'] == 'deterministic')]
        
        if len(det_data) > 0:
            det_avg = det_data.groupby('sigma')['r2'].mean().reset_index()
            ax_a.plot(det_avg['sigma'], det_avg['r2'],
                     marker='o', linestyle='--', linewidth=2, alpha=0.7,
                     label=f'{det_model.upper()} (det)', color=COLORS['deterministic'])
        
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
    
    # Panel C: BNN transformations
    ax_c = fig.add_subplot(gs[0, 2])
    
    bnn_data = metrics_df[metrics_df['model'].str.lower().str.contains('dnn|bnn', na=False)]
    
    if len(bnn_data) > 0:
        bnn_variants = bnn_data['model'].unique()
        
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
                det = metrics_df[(metrics_df['model'].str.lower().str.contains(det_model, na=False)) &
                                (metrics_df['representation'] == rep) &
                                (metrics_df['model_type'] == 'deterministic')]
                
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
    
    table1 = metrics_df.groupby(['model', 'representation', 'model_type']).agg({
        'baseline_r2': 'mean',
        'r2_high': 'mean',
        'retention_pct': 'mean',
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table1 = table1.round(4)
    table1.to_csv(output_dir / "table_phase1_summary.csv", index=False)
    print(f"✓ Saved summary table")
    
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
            
            det_data = df[(df['model'].str.lower().str.contains(det_model, na=False)) &
                         (df['representation'] == rep) &
                         (df['model_type'] == 'deterministic')]
            
            prob_data = df[(df['model'].str.lower().str.contains(prob_model, na=False)) &
                          (df['representation'] == rep) &
                          (df['model_type'] == 'probabilistic')]
            
            if len(det_data) == 0 or len(prob_data) == 0:
                results_text.append("  No data available for comparison")
                continue
            
            det_avg = det_data.groupby('sigma')['r2'].mean()
            prob_avg = prob_data.groupby('sigma')['r2'].mean()
            
            common_sigmas = set(det_avg.index) & set(prob_avg.index)
            
            if len(common_sigmas) < 3:
                results_text.append("  Insufficient overlapping sigma values")
                continue
            
            det_vals = [det_avg[s] for s in sorted(common_sigmas)]
            prob_vals = [prob_avg[s] for s in sorted(common_sigmas)]
            
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
    print("PHASE 1 ANALYSIS - DETERMINISTIC VS PROBABILISTIC (COMBINED)")
    print("="*80)
    
    df = load_phase1_results(old_results_dir, new_results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)    
    output_dir = Path(new_results_dir) / "phase1_figures_combined"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_df.to_csv(output_dir / "phase1_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase1_robustness_metrics.csv'}")
    
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir)
    create_supplementary_s4(metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_comparisons(df, metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        old_dir = sys.argv[1]
        new_dir = sys.argv[2]
        main(old_dir, new_dir)
    elif len(sys.argv) > 1:
        main(new_results_dir=sys.argv[1])
    else:
        main()