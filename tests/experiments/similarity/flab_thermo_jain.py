#!/usr/bin/env python3
"""Run all single-dataset similarity experiments for FLAB_THERMO_JAIN.

Usage:
    python tests/experiments/similarity/flab_thermo_jain.py
    python tests/experiments/similarity/flab_thermo_jain.py --experiments s01 s02 s05
    python tests/experiments/similarity/flab_thermo_jain.py --output tests/results/similarity
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.similarity.similarity_helpers import (
    run_all, SINGLE_DATASET_EXPERIMENTS,
)

DATASET = 'flab_thermo_jain'
DEFAULT_OUTPUT = 'tests/results/similarity'


def main():
    parser = argparse.ArgumentParser(
        description=f'Similarity experiments for flab_thermo_jain')
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
