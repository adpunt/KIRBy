#!/usr/bin/env python3
"""
Master runner for all KIRBy timing tests.

Executes all test files and collects comprehensive timing data.
Expected runtime: ~2-3 hours depending on hardware.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


def run_test(script, args=None, description="Test"):
    """Run a timing test and return results"""
    print(f"\n{'='*100}")
    print(f"Running: {description}")
    print(f"{'='*100}")
    
    cmd = ['python', script]
    if args:
        cmd.extend(args)
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\n✓ {description} completed in {elapsed:.2f}s ({elapsed/60:.2f}min)")
        
        return {
            'success': True,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        
        print(f"\n✗ {description} failed after {elapsed:.2f}s")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        
        return {
            'success': False,
            'elapsed_time': elapsed,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }


def main():
    print("="*100)
    print("KIRBy Comprehensive Timing Test Suite")
    print("="*100)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"\nThis will run ALL timing tests across:")
    print("  - ESOL (small molecule solubility)")
    print("  - hERG FLuID (cardiotoxicity - FLuID split)")
    print("  - hERG ChEMBL (cardiotoxicity - ChEMBL split)")
    print("  - QM9 (quantum properties)")
    print("  - TCGA BRCA (multimodal genomics)")
    print(f"\nExpected total runtime: ~2-3 hours")
    print("="*100)
    
    overall_start = time.time()
    
    # Collect all results
    results = {
        'start_time': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Define all tests
    tests = [
        {
            'name': 'esol',
            'script': 'test_esol_timed.py',
            'description': 'ESOL Solubility (Regression)',
            'output_file': 'esol_timing_results.json'
        },
        {
            'name': 'herg_fluid',
            'script': 'test_herg_timed.py',
            'description': 'hERG FLuID (Classification)',
            'args': ['--source', 'fluid'],
            'output_file': 'herg_fluid_timing_results.json'
        },
        {
            'name': 'herg_chembl',
            'script': 'test_herg_timed.py',
            'description': 'hERG ChEMBL (Classification)',
            'args': ['--source', 'chembl'],
            'output_file': 'herg_chembl_timing_results.json'
        },
        {
            'name': 'qm9',
            'script': 'test_qm9_timed.py',
            'description': 'QM9 HOMO-LUMO Gap (Regression)',
            'output_file': 'qm9_timing_results.json'
        },
        {
            'name': 'tcga_brca',
            'script': 'test_tcga_timed.py',
            'description': 'TCGA BRCA (Multimodal Classification)',
            'args': ['--cancer-type', 'BRCA', '--n-samples', '500'],
            'output_file': 'tcga_BRCA_timing_results.json'
        }
    ]
    
    # Run each test
    for i, test_config in enumerate(tests, 1):
        print(f"\n\n{'#'*100}")
        print(f"TEST {i}/{len(tests)}: {test_config['description']}")
        print(f"{'#'*100}")
        
        test_result = run_test(
            test_config['script'],
            test_config.get('args'),
            test_config['description']
        )
        
        # Store result
        results['tests'][test_config['name']] = {
            'description': test_config['description'],
            'script': test_config['script'],
            'elapsed_time': test_result['elapsed_time'],
            'success': test_result['success']
        }
        
        # Load timing data from JSON if successful
        output_file = test_config.get('output_file')
        if test_result['success'] and output_file:
            if Path(output_file).exists():
                try:
                    with open(output_file, 'r') as f:
                        timing_data = json.load(f)
                    results['tests'][test_config['name']]['timing_data'] = timing_data
                except Exception as e:
                    print(f"Warning: Could not load timing data from {output_file}: {e}")
            else:
                print(f"Warning: Expected output file not created: {output_file}")
        
        if not test_result['success']:
            print(f"\n⚠️  {test_config['description']} FAILED!")
            print("Continuing with remaining tests...")
    
    # Calculate total time
    results['end_time'] = datetime.now().isoformat()
    results['total_time'] = time.time() - overall_start
    
    # Save comprehensive results
    output_file = 'all_timing_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n\n" + "="*100)
    print("COMPREHENSIVE TIMING TEST SUMMARY")
    print("="*100)
    
    print(f"\nTotal runtime: {results['total_time']:.2f}s ({results['total_time']/60:.2f}min, {results['total_time']/3600:.2f}hr)")
    print(f"\nResults by test:")
    print(f"{'Test':<30} {'Status':<15} {'Time (min)':<15}")
    print("-" * 100)
    
    for test_name, test_data in results['tests'].items():
        status = "✓ SUCCESS" if test_data['success'] else "✗ FAILED"
        elapsed_min = test_data['elapsed_time'] / 60
        print(f"{test_data['description']:<30} {status:<15} {elapsed_min:>10.2f}")
    
    # Quick reference of fastest reps/hybrids per dataset
    print("\n" + "="*100)
    print("QUICK REFERENCE - FASTEST METHODS PER DATASET")
    print("="*100)
    
    for test_name, test_data in results['tests'].items():
        if 'timing_data' in test_data and test_data['success']:
            timing_data = test_data['timing_data']
            
            print(f"\n{test_data['description']}:")
            
            # Fastest representation
            if 'representations' in timing_data:
                reps = timing_data['representations']
                fastest_rep = min(reps.items(), key=lambda x: x[1].get('total_time', float('inf')))
                print(f"  Fastest rep: {fastest_rep[0]} ({fastest_rep[1].get('total_time', 0):.2f}s)")
            
            # Fastest hybrid
            if 'hybrids' in timing_data:
                hybrids = timing_data['hybrids']
                if hybrids:
                    fastest_hybrid = min(hybrids.items(), key=lambda x: x[1].get('total_time', float('inf')))
                    print(f"  Fastest hybrid: {fastest_hybrid[0]} ({fastest_hybrid[1].get('total_time', 0):.2f}s)")
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE RESULTS SAVED TO: {output_file}")
    print(f"{'='*100}")
    
    print(f"\nNext steps:")
    print(f"1. Review timing data in {output_file}")
    print(f"2. Use analyze_timing_results.py to categorize by speed")
    print(f"3. Build hybrid master scripts based on timing tiers")
    
    # Success/failure count
    n_success = sum(1 for t in results['tests'].values() if t['success'])
    n_total = len(results['tests'])
    
    print(f"\n{'='*100}")
    if n_success == n_total:
        print(f"✓ ALL TESTS PASSED ({n_success}/{n_total})")
    else:
        print(f"⚠️  {n_total - n_success} TEST(S) FAILED ({n_success}/{n_total} passed)")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
