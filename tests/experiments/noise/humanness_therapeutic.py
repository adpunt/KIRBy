#!/usr/bin/env python3
"""Run all single-dataset noise experiments for HUMANNESS_THERAPEUTIC.

Usage:
    python tests/experiments/noise/humanness_therapeutic.py
    python tests/experiments/noise/humanness_therapeutic.py --experiments n01 n02
    python tests/experiments/noise/humanness_therapeutic.py --output tests/results/noise
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.noise.noise_helpers import (
    run_all, SINGLE_DATASET_EXPERIMENTS,
)

DATASET = 'humanness_therapeutic'
DEFAULT_OUTPUT = 'tests/results/noise'


def main():
    parser = argparse.ArgumentParser(
        description=f'Noise experiments for humanness_therapeutic')
    parser.add_argument(
        '--experiments', nargs='+', default=None,
        help=f'Experiment IDs to run (default: all). '
             f'Options: {", ".join(SINGLE_DATASET_EXPERIMENTS)}')
    parser.add_argument(
        '--output', type=str, default=DEFAULT_OUTPUT,
        help=f'Output directory (default: {DEFAULT_OUTPUT})')
    args = parser.parse_args()
    run_all(dataset_name=DATASET, output_dir=args.output, experiments=args.experiments)


if __name__ == '__main__':
    main()
