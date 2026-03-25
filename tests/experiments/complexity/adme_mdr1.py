#!/usr/bin/env python3
"""Run all single-dataset complexity experiments for ADME_MDR1.

Usage:
    python tests/experiments/complexity/adme_mdr1.py 
    python tests/experiments/complexity/adme_mdr1.py --experiments c01 c02 c05
    python tests/experiments/complexity/adme_mdr1.py --output tests/results/complexity
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.complexity.complexity_helpers import (
    run_all, SINGLE_DATASET_EXPERIMENTS,
)

DATASET = 'adme_mdr1'
DEFAULT_OUTPUT = 'tests/results/complexity'


def main():
    parser = argparse.ArgumentParser(
        description=f'Complexity experiments for adme_mdr1')
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
