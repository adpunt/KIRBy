#!/usr/bin/env python3
"""
Hybrid Master Fast Test - Comprehensive Sanity Check

Quick pipeline validation for all KIRBy representations:
1. Molecular: ECFP4, MACCS, PDV, SNS, mol2vec, ChemBERTa, MolFormer, MHG-GNN, graph kernel, augmentations
2. Antibody: AntiBERTy, AbLang, CDR-stratified, developability, humanness
3. Sequence (DNA/RNA): Nucleotide Transformer, DNABERT-2, HyenaDNA, Caduceus, k-mer, augmentations
4. Hybrid: Greedy allocation, augmentation strategies

This is NOT for benchmarking - just verifying the pipeline works end-to-end.

Usage:
    python -m pytest tests/hybrid_master_fast.py -v          # Run as pytest
    python tests/hybrid_master_fast.py                        # Run directly
    python tests/hybrid_master_fast.py --skip-slow           # Skip slow pretrained models
"""

import sys
import os
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Small sample sizes for quick testing
N_MOLECULES = 15
N_ANTIBODIES = 10
N_SEQUENCES = 10

# Test SMILES (valid, diverse molecules)
TEST_SMILES = [
    'CCO',                          # ethanol
    'CC(=O)O',                      # acetic acid
    'c1ccccc1',                     # benzene
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # ibuprofen
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # caffeine
    'CC(=O)Nc1ccc(cc1)O',          # paracetamol
    'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',  # chloroquine
    'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',      # salbutamol
    'COc1ccc2[nH]cc(CCNC(C)=O)c2c1',      # melatonin
    'CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O',  # metamizole
    'CCCC',                         # butane
    'c1ccc2ccccc2c1',              # naphthalene
    'CC(C)C',                       # isobutane
    'C1CCCCC1',                     # cyclohexane
    'CCN',                          # ethylamine
]

# Test antibody sequences (simplified, valid)
TEST_HEAVY_CHAINS = [
    'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
    'QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCAR',
    'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR',
    'EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAR',
    'EVQLVESGGGLVQPGGSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR',
    'QITLKESGPTLVKPTQTLTLTCTFSGFSLSTSGVGVGWIRQPPGKALEWLALIYWDDDKRYSPSLKSRLTITKDTSKNQVVLTMTNMDPVDTATYYCAHR',
    'EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCAR',
    'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCAR',
]

TEST_LIGHT_CHAINS = [
    'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLT',
    'DIQLTQSPSFLSASVGDRVTITCRASQGISSYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQQLNSYPLT',
    'EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPRT',
    'DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPLT',
    'QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNTVNWYQQLPGTAPKLLIYSDNQRPSGVPDRFSGSKSGTSASLAISGLQSEDEADYYCAAWDDSLNGWV',
    'SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQAWDSSTAV',
    'DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPLT',
    'EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISSLQSEDFAVYYCQQYNNWPLT',
    'QSALTQPRSVSGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTLVV',
    'NFMLTQPHSVSESPGKTVTISCTRSSGSIASNYVQWYQQRPGSSPTTVIYDDDKRPSGVPDRFSGSIDSSSNSASLTISGLKTEDEADYYCQSYDSSNHWV',
]

# Test DNA sequences
TEST_DNA_SEQUENCES = [
    'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
    'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG',
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT',
    'GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG',
    'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC',
    'ATATATATATATATATATATATATATATATATATATATATATATATATATAT',
    'GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
    'ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC',
    'TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',
]

# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    time_seconds: float
    output_shape: Optional[Tuple] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.skip_slow = False

    def run_test(self, name: str, test_fn, skip_if_slow: bool = False):
        """Run a single test and record result."""
        if skip_if_slow and self.skip_slow:
            self.results.append(TestResult(
                name=name, passed=True, time_seconds=0,
                skipped=True, skip_reason="--skip-slow flag"
            ))
            print(f"  SKIP: {name} (slow)")
            return None

        t0 = time.time()
        try:
            result = test_fn()
            elapsed = time.time() - t0

            shape = None
            if isinstance(result, np.ndarray):
                shape = result.shape
            elif isinstance(result, dict):
                shape = {k: v.shape if isinstance(v, np.ndarray) else type(v).__name__
                        for k, v in result.items() if not k.startswith('_')}
            elif isinstance(result, tuple) and len(result) == 2:
                if isinstance(result[0], np.ndarray):
                    shape = result[0].shape

            self.results.append(TestResult(
                name=name, passed=True, time_seconds=elapsed, output_shape=shape
            ))
            print(f"  PASS: {name} ({elapsed:.2f}s) -> {shape}")
            return result

        except Exception as e:
            elapsed = time.time() - t0
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.results.append(TestResult(
                name=name, passed=False, time_seconds=elapsed, error=error_msg
            ))
            print(f"  FAIL: {name} ({elapsed:.2f}s)")
            print(f"        {error_msg}")
            if os.environ.get('DEBUG'):
                traceback.print_exc()
            return None

    def summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed and not r.skipped)
        failed = sum(1 for r in self.results if not r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        total = len(self.results)

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed:  {passed}/{total - skipped}")
        print(f"Failed:  {failed}/{total - skipped}")
        print(f"Skipped: {skipped}/{total}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")

        return failed == 0


runner = TestRunner()


# =============================================================================
# MOLECULAR REPRESENTATION TESTS
# =============================================================================

def test_molecular_representations():
    """Test all molecular representations."""
    print("\n" + "=" * 60)
    print("MOLECULAR REPRESENTATIONS")
    print("=" * 60)

    smiles = TEST_SMILES[:N_MOLECULES]
    labels = np.random.randn(len(smiles))

    # Import molecular module
    try:
        from kirby.representations import molecular
    except ImportError:
        from kirby.representations.molecular import (
            create_ecfp4, create_maccs, create_pdv, create_sns,
            create_mol2vec, create_chemberta, create_molformer, create_mhg_gnn,
            create_graph_kernel,
            compute_graph_topology, compute_spectral_features,
            compute_subgraph_counts, compute_graph_distances
        )
        molecular = None

    # --- STATIC REPRESENTATIONS ---
    print("\n[Static Representations]")

    if molecular:
        runner.run_test("ECFP4", lambda: molecular.create_ecfp4(smiles))
        runner.run_test("MACCS", lambda: molecular.create_maccs(smiles))
        runner.run_test("PDV", lambda: molecular.create_pdv(smiles))
        runner.run_test("SNS (fit)", lambda: molecular.create_sns(smiles, return_featurizer=True))
    else:
        runner.run_test("ECFP4", lambda: create_ecfp4(smiles))
        runner.run_test("MACCS", lambda: create_maccs(smiles))
        runner.run_test("PDV", lambda: create_pdv(smiles))
        runner.run_test("SNS (fit)", lambda: create_sns(smiles, return_featurizer=True))

    # --- PRETRAINED (FROZEN) ---
    print("\n[Pretrained Embeddings (Frozen)]")

    if molecular:
        runner.run_test("mol2vec", lambda: molecular.create_mol2vec(smiles), skip_if_slow=True)
        runner.run_test("ChemBERTa", lambda: molecular.create_chemberta(smiles, batch_size=8), skip_if_slow=True)
        runner.run_test("MolFormer", lambda: molecular.create_molformer(smiles, batch_size=8), skip_if_slow=True)
        runner.run_test("MHG-GNN", lambda: molecular.create_mhg_gnn(smiles, batch_size=8), skip_if_slow=True)
    else:
        runner.run_test("mol2vec", lambda: create_mol2vec(smiles), skip_if_slow=True)
        runner.run_test("ChemBERTa", lambda: create_chemberta(smiles, batch_size=8), skip_if_slow=True)
        runner.run_test("MolFormer", lambda: create_molformer(smiles, batch_size=8), skip_if_slow=True)
        runner.run_test("MHG-GNN", lambda: create_mhg_gnn(smiles, batch_size=8), skip_if_slow=True)

    # --- GRAPH KERNEL ---
    print("\n[Graph Kernel]")

    if molecular:
        runner.run_test("WL Graph Kernel", lambda: molecular.create_graph_kernel(smiles, n_iter=3, return_vocabulary=True))
    else:
        runner.run_test("WL Graph Kernel", lambda: create_graph_kernel(smiles, n_iter=3, return_vocabulary=True))

    # --- AUGMENTATIONS ---
    print("\n[Augmentations]")

    if molecular:
        runner.run_test("Graph Topology", lambda: molecular.compute_graph_topology(smiles))
        runner.run_test("Spectral Features", lambda: molecular.compute_spectral_features(smiles, k=5))
        runner.run_test("Subgraph Counts", lambda: molecular.compute_subgraph_counts(smiles))
        runner.run_test("Graph Distances", lambda: molecular.compute_graph_distances(smiles))
    else:
        runner.run_test("Graph Topology", lambda: compute_graph_topology(smiles))
        runner.run_test("Spectral Features", lambda: compute_spectral_features(smiles, k=5))
        runner.run_test("Subgraph Counts", lambda: compute_subgraph_counts(smiles))
        runner.run_test("Graph Distances", lambda: compute_graph_distances(smiles))


# =============================================================================
# ANTIBODY REPRESENTATION TESTS
# =============================================================================

def test_antibody_representations():
    """Test all antibody representations."""
    print("\n" + "=" * 60)
    print("ANTIBODY REPRESENTATIONS")
    print("=" * 60)

    heavy = TEST_HEAVY_CHAINS[:N_ANTIBODIES]
    light = TEST_LIGHT_CHAINS[:N_ANTIBODIES]

    # Import antibody module
    try:
        from kirby.representations import antibody
    except ImportError:
        from kirby.representations.antibody import (
            create_antiberty_embeddings, create_antiberty_embeddings_batch,
            create_ablang_embeddings, create_ablang2_embeddings,
            compute_developability_features, compute_humanness_scores
        )
        antibody = None

    # --- LANGUAGE MODEL EMBEDDINGS ---
    print("\n[Language Model Embeddings]")

    if antibody:
        runner.run_test("AntiBERTy (heavy)",
                       lambda: antibody.create_antiberty_embeddings(heavy, chain_type='heavy'),
                       skip_if_slow=True)
        runner.run_test("AntiBERTy Batch",
                       lambda: antibody.create_antiberty_embeddings_batch(heavy, light, aggregations=['mean']),
                       skip_if_slow=True)
        runner.run_test("AbLang (heavy)",
                       lambda: antibody.create_ablang_embeddings(heavy),
                       skip_if_slow=True)
        runner.run_test("AbLang2 (paired)",
                       lambda: antibody.create_ablang2_embeddings(heavy, light),
                       skip_if_slow=True)
    else:
        runner.run_test("AntiBERTy (heavy)",
                       lambda: create_antiberty_embeddings(heavy, chain_type='heavy'),
                       skip_if_slow=True)
        runner.run_test("AntiBERTy Batch",
                       lambda: create_antiberty_embeddings_batch(heavy, light, aggregations=['mean']),
                       skip_if_slow=True)
        runner.run_test("AbLang (heavy)",
                       lambda: create_ablang_embeddings(heavy),
                       skip_if_slow=True)
        runner.run_test("AbLang2 (paired)",
                       lambda: create_ablang2_embeddings(heavy, light),
                       skip_if_slow=True)

    # --- AUGMENTATIONS ---
    print("\n[Augmentations]")

    if antibody:
        runner.run_test("Developability Features",
                       lambda: antibody.compute_developability_features(heavy, light))
        runner.run_test("Humanness Scores (heavy)",
                       lambda: antibody.compute_humanness_scores(heavy, chain_type='heavy'))
        runner.run_test("Humanness Scores (light)",
                       lambda: antibody.compute_humanness_scores(light, chain_type='light'))
    else:
        runner.run_test("Developability Features",
                       lambda: compute_developability_features(heavy, light))
        runner.run_test("Humanness Scores (heavy)",
                       lambda: compute_humanness_scores(heavy, chain_type='heavy'))
        runner.run_test("Humanness Scores (light)",
                       lambda: compute_humanness_scores(light, chain_type='light'))


# =============================================================================
# SEQUENCE (DNA/RNA) REPRESENTATION TESTS
# =============================================================================

def test_sequence_representations():
    """Test all sequence (DNA/RNA) representations."""
    print("\n" + "=" * 60)
    print("SEQUENCE (DNA/RNA) REPRESENTATIONS")
    print("=" * 60)

    seqs = TEST_DNA_SEQUENCES[:N_SEQUENCES]

    # Import sequence module
    try:
        from kirby.representations import sequence
    except ImportError:
        from kirby.representations.sequence import (
            create_nucleotide_transformer, create_dnabert2, create_hyenadna, create_caduceus,
            create_kmer_features, create_onehot_encoding,
            compute_gc_content, compute_sequence_complexity,
            compute_positional_features, compute_motif_features
        )
        sequence = None

    # --- STATIC REPRESENTATIONS ---
    print("\n[Static Representations]")

    if sequence:
        runner.run_test("k-mer (k=4)", lambda: sequence.create_kmer_features(seqs, k=4))
        runner.run_test("One-hot encoding", lambda: sequence.create_onehot_encoding(seqs, max_length=60))
    else:
        runner.run_test("k-mer (k=4)", lambda: create_kmer_features(seqs, k=4))
        runner.run_test("One-hot encoding", lambda: create_onehot_encoding(seqs, max_length=60))

    # --- PRETRAINED (FROZEN) ---
    print("\n[Pretrained Embeddings (Frozen)]")

    if sequence:
        runner.run_test("Nucleotide Transformer",
                       lambda: sequence.create_nucleotide_transformer(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("DNABERT-2",
                       lambda: sequence.create_dnabert2(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("HyenaDNA",
                       lambda: sequence.create_hyenadna(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("Caduceus",
                       lambda: sequence.create_caduceus(seqs, batch_size=4),
                       skip_if_slow=True)
    else:
        runner.run_test("Nucleotide Transformer",
                       lambda: create_nucleotide_transformer(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("DNABERT-2",
                       lambda: create_dnabert2(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("HyenaDNA",
                       lambda: create_hyenadna(seqs, batch_size=4),
                       skip_if_slow=True)
        runner.run_test("Caduceus",
                       lambda: create_caduceus(seqs, batch_size=4),
                       skip_if_slow=True)

    # --- AUGMENTATIONS ---
    print("\n[Augmentations]")

    if sequence:
        runner.run_test("GC Content", lambda: sequence.compute_gc_content(seqs))
        runner.run_test("Sequence Complexity", lambda: sequence.compute_sequence_complexity(seqs))
        runner.run_test("Positional Features", lambda: sequence.compute_positional_features(seqs, n_bins=5))
        runner.run_test("Motif Features", lambda: sequence.compute_motif_features(seqs))
    else:
        runner.run_test("GC Content", lambda: compute_gc_content(seqs))
        runner.run_test("Sequence Complexity", lambda: compute_sequence_complexity(seqs))
        runner.run_test("Positional Features", lambda: compute_positional_features(seqs, n_bins=5))
        runner.run_test("Motif Features", lambda: compute_motif_features(seqs))


# =============================================================================
# HYBRID CREATION TESTS
# =============================================================================

def test_hybrid_creation():
    """Test hybrid creation with different allocation methods."""
    print("\n" + "=" * 60)
    print("HYBRID CREATION")
    print("=" * 60)

    # Import hybrid module
    try:
        from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection
    except ImportError:
        from kirby.hybrid import create_hybrid, apply_feature_selection, apply_augmentation_selection

    # Import molecular for base reps
    try:
        from kirby.representations.molecular import (
            create_ecfp4, create_maccs, create_pdv,
            compute_graph_topology, compute_spectral_features
        )
    except ImportError:
        from kirby.representations.molecular import (
            create_ecfp4, create_maccs, create_pdv,
            compute_graph_topology, compute_spectral_features
        )

    smiles = TEST_SMILES[:N_MOLECULES]
    labels = np.random.randn(len(smiles))

    # Generate base representations
    print("\n[Generating base representations for hybrid tests]")
    base_reps = {}
    base_reps['ecfp4'] = create_ecfp4(smiles)
    base_reps['maccs'] = create_maccs(smiles)
    base_reps['pdv'] = create_pdv(smiles)
    print(f"  Base reps: {list(base_reps.keys())}")

    # Generate augmentations
    augmentations = {}
    augmentations['topology'] = compute_graph_topology(smiles)
    augmentations['spectral'] = compute_spectral_features(smiles, k=5)
    print(f"  Augmentations: {list(augmentations.keys())}")

    # --- ALLOCATION METHODS ---
    print("\n[Allocation Methods]")

    runner.run_test("Greedy Allocation",
                   lambda: create_hybrid(base_reps, labels, allocation_method='greedy', budget=50, step_size=10))

    runner.run_test("Fixed Allocation",
                   lambda: create_hybrid(base_reps, labels, allocation_method='fixed', n_per_rep=20))

    runner.run_test("Performance-Weighted Allocation",
                   lambda: create_hybrid(base_reps, labels, allocation_method='performance_weighted', budget=50))

    # --- AUGMENTATION STRATEGIES ---
    print("\n[Augmentation Strategies]")

    runner.run_test("Augmentation: none",
                   lambda: create_hybrid(base_reps, labels, allocation_method='greedy', budget=50,
                                        augmentations=augmentations, augmentation_strategy='none'))

    runner.run_test("Augmentation: all",
                   lambda: create_hybrid(base_reps, labels, allocation_method='greedy', budget=50,
                                        augmentations=augmentations, augmentation_strategy='all',
                                        augmentation_budget=10))

    runner.run_test("Augmentation: greedy_ablation",
                   lambda: create_hybrid(base_reps, labels, allocation_method='greedy', budget=50,
                                        augmentations=augmentations, augmentation_strategy='greedy_ablation',
                                        augmentation_budget=10))

    # --- TEST/VAL APPLICATION ---
    print("\n[Train/Test Split Application]")

    def test_train_test_split():
        # Simulate train/test
        train_idx = list(range(10))
        test_idx = list(range(10, N_MOLECULES))

        train_smiles = [smiles[i] for i in train_idx]
        test_smiles = [smiles[i] for i in test_idx]
        train_labels = labels[train_idx]

        # Generate reps for train and test
        train_reps = {
            'ecfp4': create_ecfp4(train_smiles),
            'maccs': create_maccs(train_smiles),
        }
        test_reps = {
            'ecfp4': create_ecfp4(test_smiles),
            'maccs': create_maccs(test_smiles),
        }

        # Create hybrid on train
        hybrid_train, feature_info = create_hybrid(
            train_reps, train_labels,
            allocation_method='greedy', budget=30
        )

        # Apply same selection to test
        hybrid_test = apply_feature_selection(test_reps, feature_info)

        return {'train_shape': hybrid_train.shape, 'test_shape': hybrid_test.shape}

    runner.run_test("Train/Test Feature Selection", test_train_test_split)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='KIRBy Hybrid Master Fast Test')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow pretrained model tests')
    parser.add_argument('--molecular-only', action='store_true',
                       help='Only test molecular representations')
    parser.add_argument('--antibody-only', action='store_true',
                       help='Only test antibody representations')
    parser.add_argument('--sequence-only', action='store_true',
                       help='Only test sequence representations')
    parser.add_argument('--hybrid-only', action='store_true',
                       help='Only test hybrid creation')

    args = parser.parse_args()

    runner.skip_slow = args.skip_slow

    print("=" * 60)
    print("KIRBy HYBRID MASTER FAST TEST")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  N_MOLECULES:  {N_MOLECULES}")
    print(f"  N_ANTIBODIES: {N_ANTIBODIES}")
    print(f"  N_SEQUENCES:  {N_SEQUENCES}")
    print(f"  Skip slow:    {args.skip_slow}")

    # Determine which tests to run
    run_all = not (args.molecular_only or args.antibody_only or
                   args.sequence_only or args.hybrid_only)

    start_time = time.time()

    if run_all or args.molecular_only:
        test_molecular_representations()

    if run_all or args.antibody_only:
        test_antibody_representations()

    if run_all or args.sequence_only:
        test_sequence_representations()

    if run_all or args.hybrid_only:
        test_hybrid_creation()

    total_time = time.time() - start_time

    print(f"\nTotal time: {total_time:.1f}s")

    success = runner.summary()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
