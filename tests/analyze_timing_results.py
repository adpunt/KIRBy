#!/usr/bin/env python3
"""
Quick analysis of existing timing results

Checks what timing data is available and provides summary statistics
without needing to re-run the full test suite.
"""

import json
from pathlib import Path
import sys


def analyze_timing_file(filepath):
    """Analyze a single timing results file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = data.get('dataset', 'unknown')
        total_time = data.get('total_time', 0)
        
        reps = data.get('representations', {})
        hybrids = data.get('hybrids', {})
        
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"{'='*80}")
        
        # Representations summary
        if reps:
            print(f"\nRepresentations tested: {len(reps)}")
            print(f"{'Name':<20} {'Time (s)':>10} {'Features':>10} {'Metric':>10}")
            print("-"*80)
            
            sorted_reps = sorted(reps.items(), key=lambda x: x[1]['total_time'])
            for name, data in sorted_reps:
                time_s = data['total_time']
                n_feat = data.get('n_features', 'N/A')
                
                # Get primary metric
                if 'rmse' in data:
                    metric = f"RMSE:{data['rmse']:.3f}"
                elif 'auc' in data and data['auc'] is not None:
                    metric = f"AUC:{data['auc']:.3f}"
                elif 'mae' in data:
                    metric = f"MAE:{data['mae']:.3f}"
                else:
                    metric = "N/A"
                
                print(f"{name:<20} {time_s:>10.2f} {str(n_feat):>10} {metric:>10}")
        
        # Hybrids summary
        if hybrids:
            print(f"\nHybrids tested: {len(hybrids)}")
            print(f"{'Name':<30} {'Time (s)':>10} {'Features':>10} {'Metric':>10}")
            print("-"*80)
            
            sorted_hybrids = sorted(hybrids.items(), key=lambda x: x[1]['total_time'])
            for name, data in sorted_hybrids:
                time_s = data['total_time']
                n_feat = data.get('n_features', 'N/A')
                
                # Get primary metric
                if 'rmse' in data:
                    metric = f"RMSE:{data['rmse']:.3f}"
                elif 'auc' in data and data['auc'] is not None:
                    metric = f"AUC:{data['auc']:.3f}"
                elif 'mae' in data:
                    metric = f"MAE:{data['mae']:.3f}"
                else:
                    metric = "N/A"
                
                print(f"{name:<30} {time_s:>10.2f} {str(n_feat):>10} {metric:>10}")
        
        # Speed recommendations
        print(f"\n{'Speed Tiers':<30} {'Time Range':<20} {'Candidates'}")
        print("-"*80)
        
        all_items = [(name, data['total_time'], 'rep') for name, data in reps.items()]
        all_items += [(name, data['total_time'], 'hybrid') for name, data in hybrids.items()]
        all_items.sort(key=lambda x: x[1])
        
        # Categorize by speed
        super_fast = [x for x in all_items if x[1] < 10]
        fast = [x for x in all_items if 10 <= x[1] < 30]
        medium = [x for x in all_items if 30 <= x[1] < 60]
        slow = [x for x in all_items if x[1] >= 60]
        
        print(f"{'Super Fast (<10s)':<30} {'<10s':<20} {len(super_fast)}")
        print(f"{'Fast (10-30s)':<30} {'10-30s':<20} {len(fast)}")
        print(f"{'Medium (30-60s)':<30} {'30-60s':<20} {len(medium)}")
        print(f"{'Slow (>60s)':<30} {'>60s':<20} {len(slow)}")
        
        if super_fast:
            print(f"\nSuper fast options: {', '.join([x[0] for x in super_fast[:5]])}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return False


def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         KIRBy Timing Results Quick Analysis                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Check for timing result files
    result_files = {
        'ESOL': 'esol_timing_results.json',
        'hERG FLuID': 'herg_fluid_timing_results.json',
        'hERG ChEMBL': 'herg_chembl_timing_results.json',
        'QM9': 'qm9_timing_results.json',
        'ALL': 'all_timing_results.json'
    }
    
    available = {}
    missing = []
    
    for name, filepath in result_files.items():
        if Path(filepath).exists():
            available[name] = filepath
        else:
            missing.append(name)
    
    # Summary
    print(f"\nAvailable results: {len(available)}/{len(result_files)}")
    
    if missing:
        print(f"\nMissing results:")
        for name in missing:
            print(f"  ✗ {name}")
        print(f"\nRun 'python run_all_timing_tests.py' to generate missing results")
    
    # Analyze available results
    if available:
        print(f"\n{'='*80}")
        print("ANALYZING AVAILABLE RESULTS")
        print(f"{'='*80}")
        
        for name, filepath in available.items():
            if name == 'ALL':
                continue  # Skip the master file
            
            analyze_timing_file(filepath)
    else:
        print("\n⚠ No timing results found!")
        print("Run 'python run_all_timing_tests.py' to generate timing data")
        sys.exit(1)
    
    # Overall recommendations only if we have data
    if available:
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS FOR HYBRID MASTER SCRIPT")
        print(f"{'='*80}")
        
        print("\nBased on timing data, consider:")
        print("\n1. SUPER FAST TEST (~30-60s total)")
        print("   - Use fastest representations only (ECFP4, PDV, mol2vec)")
        print("   - 1-2 simple hybrids (ECFP4+PDV)")
        print("   - 1-2 datasets (ESOL + hERG FLuID)")
        print("   - Skip slow models (GNN, GP)")
        
        print("\n2. MEDIUM SPEED TEST (~5-10min total)")
        print("   - Add pretrained reps (Graph Kernel, MHG-GNN)")
        print("   - 3-5 hybrids with varying complexity")
        print("   - 3 datasets (ESOL, hERG, QM9)")
        print("   - Include one finetuned model (GNN with reduced epochs)")
        
        print("\n3. COMPREHENSIVE TEST (~30-60min total)")
        print("   - All representations")
        print("   - 8-10 diverse hybrids")
        print("   - All datasets")
        print("   - All model types")
        
        print(f"\n{'='*80}")
        print("Analysis complete! Ready to design hybrid master scripts.")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
