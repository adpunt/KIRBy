#!/usr/bin/env python3
"""Run all single-dataset similarity experiments for GINKGO_COLLOIDAL.

Usage:
    python tests/experiments/similarity/ginkgo_colloidal.py
    python tests/experiments/similarity/ginkgo_colloidal.py --experiments s01 s02 s05
    python tests/experiments/similarity/ginkgo_colloidal.py --output tests/results/similarity
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

DATASET = 'ginkgo_colloidal'
DEFAULT_OUTPUT = 'tests/results/similarity'


def main():
    parser = argparse.ArgumentParser(
        description=f'Similarity experiments for ginkgo_colloidal')
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
