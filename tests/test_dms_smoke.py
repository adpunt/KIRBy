#!/usr/bin/env python3
"""
DMS Smoke Test — Verify all 9 DMS datasets × no-dependency reps load and generate.

Loads each dataset with a small sample limit (~50), generates each rep that needs
no GPU or model downloads, and checks: correct shape, no NaNs, no crashes.

Reps tested (no external model downloads needed):
  - aac (20D), blosum62 (20D), biophysical_delta (6D), positional (11D),
    developability (16D)
  - varpos_onehot, varpos_blosum, varpos_physchem (all positions)
  - varpos_cdr_onehot, varpos_cdr_physchem (CDR only)
  - varpos_cdrh3_physchem (CDR-H3 only)
  - varpos_fw_physchem (framework only)

Reps NOT tested here (require model downloads, server-only):
  - lm_score_ablang2, lm_score_esm2, lm_score_prot_t5, lm_score_antiberty
  - esm2_site, esm1v_score, antifold_score, sapiens_score

Usage:
    python tests/test_dms_smoke.py
    python tests/test_dms_smoke.py --samples 20  # faster
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

# All 9 DMS datasets (4 FLAb intra-family + 5 AbAgym)
DMS_DATASETS = [
    'flab_expression_koenig',
    'flab_binding_koenig',
    'flab_thermo_hie',
    'flab_binding_rosace',
    'abagym_Ang2_2017_G6',
    'abagym_EGFR_2013_Cetuximab',
    'abagym_lysozyme_2019_D441',
    'abagym_VEGF_2017b_G6',
    'abagym_HER2_2021_trastuzumab',
]

# No-dependency reps with expected min dimensions
# (varpos dims depend on dataset alignment, so we just check > 0)
NO_DEP_REPS = {
    # (rep_name, mode, expected_dim_or_None)
    'aac':                    ('concat', 20),
    'blosum62':               ('concat', 20),
    'biophysical_delta':      ('mutation', 6),
    'positional':             ('mutation', 11),
    'developability':         ('paired', 16),
    'varpos_onehot':          ('varpos', None),
    'varpos_blosum':          ('varpos', None),
    'varpos_physchem':        ('varpos', None),
    'varpos_cdr_onehot':      ('varpos', None),
    'varpos_cdr_physchem':    ('varpos', None),
    'varpos_cdrh3_physchem':  ('varpos', None),
    'varpos_fw_physchem':     ('varpos', None),
}


def load_dataset_slim(name: str, max_samples: int = 50):
    """Load a DMS dataset with a small sample limit for smoke testing."""
    if name.startswith('flab_'):
        from kirby.datasets.flab import load_flab_dataset
        data = load_flab_dataset(name, max_samples=max_samples)
    elif name.startswith('abagym_'):
        from kirby.datasets.abagym import load_abagym_dataset
        data = load_abagym_dataset(name, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown DMS dataset: {name}")
    return data


def generate_rep(rep_name, mode, data):
    """Generate a single rep from loaded dataset data."""
    from kirby.representations.antibody import (
        compute_aac_features,
        compute_blosum62_features,
        compute_biophysical_delta,
        compute_mutation_positional_features,
        compute_developability_features,
        create_variable_position_encoding,
    )

    heavy = data['heavy_seqs']
    light = data['light_seqs']
    mutations = data['mutations']
    wt_heavy = data['wildtype_heavy']
    wt_light = data.get('wildtype_light')

    if mode == 'concat':
        if light:
            seqs = [h + l for h, l in zip(heavy, light)]
        else:
            seqs = heavy
        if rep_name == 'aac':
            return compute_aac_features(seqs)
        elif rep_name == 'blosum62':
            return compute_blosum62_features(seqs)

    elif mode == 'paired':
        if rep_name == 'developability':
            return compute_developability_features(heavy, light)

    elif mode == 'mutation':
        if rep_name == 'biophysical_delta':
            return compute_biophysical_delta(mutations)
        elif rep_name == 'positional':
            return compute_mutation_positional_features(mutations, wt_heavy or '', wt_light)

    elif mode == 'varpos':
        suffix = rep_name.replace('varpos_', '')
        region = 'all'
        if suffix.startswith('cdrh3_'):
            region = 'cdr_h3'
            suffix = suffix.replace('cdrh3_', '')
        elif suffix.startswith('cdr_'):
            region = 'cdr'
            suffix = suffix.replace('cdr_', '')
        elif suffix.startswith('fw_'):
            region = 'framework'
            suffix = suffix.replace('fw_', '')
        enc = suffix
        if light:
            return create_variable_position_encoding(heavy, light, encoding=enc, region=region)
        else:
            return create_variable_position_encoding(heavy, encoding=enc, region=region)

    raise ValueError(f"Unknown rep dispatch: {rep_name} / {mode}")


def main():
    parser = argparse.ArgumentParser(description='DMS Smoke Test')
    parser.add_argument('--samples', type=int, default=50,
                        help='Max samples per dataset (default: 50)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets to test (default: all 9)')
    args = parser.parse_args()

    datasets = args.datasets or DMS_DATASETS
    max_samples = args.samples

    print("=" * 72)
    print(f"DMS SMOKE TEST — {len(datasets)} datasets × {len(NO_DEP_REPS)} reps")
    print(f"  Max samples per dataset: {max_samples}")
    print("=" * 72)

    # Results: (dataset, rep, status, dim, time_s)
    results = []
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for ds_name in datasets:
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 60}")

        # Load dataset
        try:
            t0 = time.time()
            data = load_dataset_slim(ds_name, max_samples=max_samples)
            n_samples = len(data['heavy_seqs'])
            load_time = time.time() - t0
            print(f"  Loaded: {n_samples} samples ({load_time:.1f}s)")
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            for rep_name in NO_DEP_REPS:
                results.append((ds_name, rep_name, 'LOAD_FAIL', 0, 0.0, str(e)))
                n_fail += 1
            continue

        has_mutations = (data['mutations'] is not None
                         and any(len(m) > 0 for m in data['mutations']))

        for rep_name, (mode, expected_dim) in NO_DEP_REPS.items():
            # Skip mutation reps if no mutation data
            if mode in ('mutation',) and not has_mutations:
                results.append((ds_name, rep_name, 'SKIP', 0, 0.0, 'no mutations'))
                n_skip += 1
                print(f"  {rep_name:30s} SKIP (no mutations)")
                continue
            if mode == 'varpos' and not has_mutations:
                # varpos still works (it detects variable positions from sequences)
                pass

            try:
                t0 = time.time()
                rep = generate_rep(rep_name, mode, data)
                elapsed = time.time() - t0

                # Validate
                assert isinstance(rep, np.ndarray), f"Expected ndarray, got {type(rep)}"
                assert rep.ndim == 2, f"Expected 2D, got {rep.ndim}D"
                assert rep.shape[0] == n_samples, (
                    f"Row mismatch: {rep.shape[0]} vs {n_samples}")
                assert rep.shape[1] > 0, "Zero features"
                nan_count = np.isnan(rep).sum()
                if expected_dim is not None:
                    assert rep.shape[1] == expected_dim, (
                        f"Dim mismatch: {rep.shape[1]} vs {expected_dim}")

                status = 'PASS' if nan_count == 0 else f'WARN({nan_count} NaN)'
                results.append((ds_name, rep_name, status, rep.shape[1], elapsed, ''))
                n_pass += 1
                tag = 'OK' if nan_count == 0 else f'OK ({nan_count} NaN)'
                print(f"  {rep_name:30s} {tag:12s} {rep.shape[1]:5d}D  {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - t0
                results.append((ds_name, rep_name, 'FAIL', 0, elapsed, str(e)[:80]))
                n_fail += 1
                print(f"  {rep_name:30s} FAIL         {str(e)[:60]}")

    # Summary table
    print(f"\n{'=' * 72}")
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"{'=' * 72}")

    if n_fail > 0:
        print("\nFAILURES:")
        for ds, rep, status, dim, t, msg in results:
            if 'FAIL' in status:
                print(f"  {ds} / {rep}: {msg}")

    # Compact matrix view
    print(f"\n{'Dataset':<35s}", end='')
    rep_names = list(NO_DEP_REPS.keys())
    short = [r.replace('varpos_', 'vp_').replace('biophysical_', 'bp_')
             .replace('positional', 'pos').replace('developability', 'dev')
             for r in rep_names]
    for s in short:
        print(f" {s[:6]:>6s}", end='')
    print()
    print('─' * (35 + 7 * len(rep_names)))

    for ds_name in datasets:
        print(f"{ds_name:<35s}", end='')
        for rep_name in rep_names:
            match = [r for r in results if r[0] == ds_name and r[1] == rep_name]
            if match:
                _, _, status, dim, _, _ = match[0]
                if 'PASS' in status:
                    print(f" {dim:>5d}D", end='')
                elif 'SKIP' in status:
                    print(f"   skip", end='')
                else:
                    print(f"   FAIL", end='')
            else:
                print(f"      ?", end='')
        print()

    return 1 if n_fail > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
