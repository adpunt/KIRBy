#!/usr/bin/env python3
"""
Precache Molecular Representations (subprocess-isolated)

Runs each (dataset, rep) pair in an isolated subprocess so that crashes
(SIGBUS), hangs, and import errors in one rep can't take down the rest.

Re-runnable: checks cache first, skips reps that already have a .npy file.

Usage:
    # Single dataset, all reps in 'all' preset
    python scripts/precache_molecular_reps.py logd

    # All 4 datasets
    python scripts/precache_molecular_reps.py --all-datasets

    # Specific reps only
    python scripts/precache_molecular_reps.py logd --reps chemberta,molformer

    # With GPU (faster for HF transformers)
    python scripts/precache_molecular_reps.py logd --gpu

    # Custom timeout (seconds)
    python scripts/precache_molecular_reps.py logd --reps schnet --timeout 7200
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

# All molecular datasets
ALL_DATASETS = ['logd', 'caco2', 'herg', 'qm9']

# Reps in the 'all' preset + chemformer (implemented but not in preset)
ALL_REPS = [
    'ecfp4', 'maccs', 'pdv', 'mol2vec', 'chemberta', 'molformer',
    'selformer', 'chembert', 'molclr', 'graphmvp', 'smited',
    'mhg_gnn', 'grover', 'graph_kernel', 'schnet', 'chemformer'
]

# Per-rep timeout tiers (seconds)
REP_TIMEOUTS = {
    # Fast CPU fingerprints
    'ecfp4': 600, 'maccs': 600, 'pdv': 600,
    # HF transformers (auto-download, GPU helps)
    'chemberta': 1800, 'molformer': 1800, 'selformer': 1800, 'chembert': 1800,
    # Fast CPU
    'mol2vec': 600,
    # Manual setup needed, GPU helps
    'grover': 3600, 'chemformer': 3600, 'molclr': 3600, 'graphmvp': 3600,
    'smited': 3600,
    # Fragile (SIGBUS history)
    'mhg_gnn': 3600,
    # Slow (conformer generation / kernel computation)
    'schnet': 7200, 'graph_kernel': 7200,
}

WORKER = str(ROOT / 'scripts' / '_generate_single_rep.py')


def run_one(dataset: str, rep_name: str, gpu: bool, timeout: int,
            python: str) -> dict:
    """Run a single (dataset, rep) pair in a subprocess.

    Returns dict with keys: dataset, rep, status, elapsed, error.
    """
    cmd = [python, WORKER, dataset, rep_name]
    if gpu:
        cmd.append('--gpu')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    t0 = time.time()
    result = {
        'dataset': dataset,
        'rep': rep_name,
        'status': 'unknown',
        'elapsed': 0,
        'error': None,
    }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,  # isolate process group
        )

        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the subprocess's process group (not ours)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            proc.wait()
            result['status'] = 'TIMEOUT'
            result['error'] = f'Killed after {timeout}s'
            result['elapsed'] = time.time() - t0
            return result

        result['elapsed'] = time.time() - t0

        if proc.returncode == 0:
            # Check if it was a cache hit
            if stdout and 'CACHED' in stdout:
                result['status'] = 'CACHED'
            else:
                result['status'] = 'OK'
        elif proc.returncode < 0:
            sig_name = signal.Signals(-proc.returncode).name
            result['status'] = 'CRASH'
            result['error'] = f'Signal {sig_name} (exit {proc.returncode})'
        else:
            result['status'] = 'FAILED'
            # Grab last line of stdout as error message
            lines = (stdout or '').strip().split('\n')
            result['error'] = lines[-1][:200] if lines else f'exit {proc.returncode}'

        # Print worker output indented
        if stdout and stdout.strip():
            for line in stdout.strip().split('\n'):
                print(f"      {line}")

    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)[:200]
        result['elapsed'] = time.time() - t0

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Precache molecular reps in isolated subprocesses"
    )
    parser.add_argument("dataset", nargs='?', default=None,
                        help="Dataset name (logd, caco2, herg, qm9)")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run all 4 molecular datasets")
    parser.add_argument("--reps", type=str, default=None,
                        help="Comma-separated rep names (default: all)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for transformer reps")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Override per-rep timeout (seconds)")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable to use")
    args = parser.parse_args()

    if args.all_datasets:
        datasets = ALL_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify a dataset or use --all-datasets")

    if args.reps:
        reps = [r.strip().lower() for r in args.reps.split(',')]
    else:
        reps = ALL_REPS

    print("=" * 70)
    print("MOLECULAR REPRESENTATION PRECACHE")
    print(f"  Datasets:  {', '.join(datasets)}")
    print(f"  Reps:      {len(reps)} ({', '.join(reps)})")
    print(f"  GPU:       {args.gpu}")
    print(f"  Python:    {args.python}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*50}")

        for rep_name in reps:
            timeout = args.timeout or REP_TIMEOUTS.get(rep_name, 3600)
            print(f"\n  [{rep_name}] timeout={timeout}s ...")

            result = run_one(dataset, rep_name, args.gpu, timeout, args.python)
            all_results.append(result)

            status_icon = {
                'OK': '+', 'CACHED': '~', 'FAILED': 'X',
                'CRASH': '!', 'TIMEOUT': 'T', 'ERROR': '?',
            }.get(result['status'], '?')

            msg = f"  [{status_icon}] {rep_name}: {result['status']} ({result['elapsed']:.1f}s)"
            if result['error']:
                msg += f" — {result['error']}"
            print(msg)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_status = {}
    for r in all_results:
        by_status.setdefault(r['status'], []).append(r)

    for status in ['OK', 'CACHED', 'FAILED', 'CRASH', 'TIMEOUT', 'ERROR']:
        items = by_status.get(status, [])
        if items:
            names = [f"{r['dataset']}/{r['rep']}" for r in items]
            print(f"  {status} ({len(items)}): {', '.join(names)}")

    total = len(all_results)
    ok = len(by_status.get('OK', [])) + len(by_status.get('CACHED', []))
    print(f"\n  Total: {ok}/{total} succeeded")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Exit with failure if any non-cached rep failed
    failures = total - ok
    if failures > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
