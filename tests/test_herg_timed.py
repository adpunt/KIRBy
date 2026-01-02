import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
Test hERG cardiotoxicity dataset with ALL KIRBy representations.

This is BINARY CLASSIFICATION (blocker vs. non-blocker) unlike ESOL/QM9 regression.

Tests:
1. Every single representation individually
2. Sample hybrid combinations covering the representation space

Metrics: Accuracy, AUC-ROC, Precision, Recall, F1, MCC
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import time
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.herg import load_herg, get_herg_splits

from kirby.representations.molecular import (
    create_ecfp4,
    create_pdv,
    create_mol2vec,
    create_graph_kernel,
    create_mhg_gnn,
    finetune_molformer,
    finetune_chemberta,
    finetune_gnn,
    encode_from_model,
    create_hybrid
)


def evaluate_classification(y_true, y_pred, y_prob=None, name="Model"):
    """
    Calculate and print classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        name: Model name
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # AUC requires probabilities
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"{name:40s} - ACC: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
        return acc, auc, f1, mcc
    else:
        print(f"{name:40s} - ACC: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
        return acc, None, f1, mcc


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test KIRBy representations on hERG cardiotoxicity')
    parser.add_argument('--source', choices=['tdc', 'chembl', 'fluid'],
                       default='tdc', help='Data source')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples (default: all)')
    parser.add_argument('--splitter', choices=['scaffold', 'random', 'stratified'],
                       default='scaffold', help='Split method')
    args = parser.parse_args()
    
    print("="*100)
    print("hERG Cardiotoxicity - Complete KIRBy Test Suite")
    print("="*100)
    
    # Track all timings
    timings = {
        'dataset': 'herg',
        'source': args.source,
        'start_time': datetime.now().isoformat(),
        'representations': {},
        'hybrids': {},
        'total_time': 0
    }
    
    overall_start = time.time()
    
    # Load data
    print(f"\nLoading hERG dataset from {args.source}...")
    data_start = time.time()
    
    if args.source == 'fluid':
        # FLuID already has train/test split
        print("Loading FLuID training set...")
        train_data = load_herg(source='fluid', use_test=False)
        print("Loading FLuID test set...")
        test_data = load_herg(source='fluid', use_test=True)
        
        train_smiles = train_data['smiles']
        train_labels = train_data['labels']
        test_smiles = test_data['smiles']
        test_labels = test_data['labels']
        
        # Create internal validation split from training data
        n_val = len(train_smiles) // 5
        val_smiles = train_smiles[:n_val]
        val_labels = train_labels[:n_val]
        train_smiles_fit = train_smiles[n_val:]
        train_labels_fit = train_labels[n_val:]
        
    else:
        # TDC/ChEMBL: load and split
        data = load_herg(source=args.source, n_samples=args.n_samples)
        splits = get_herg_splits(data, splitter=args.splitter)
        
        # Combine train+val for final model
        train_smiles = splits['train']['smiles'] + splits['val']['smiles']
        train_labels = np.concatenate([splits['train']['labels'], splits['val']['labels']])
        test_smiles = splits['test']['smiles']
        test_labels = splits['test']['labels']
        
        # Validation split for fine-tuning
        n_val = len(train_smiles) // 5
        val_smiles = train_smiles[:n_val]
        val_labels = train_labels[:n_val]
        train_smiles_fit = train_smiles[n_val:]
        train_labels_fit = train_labels[n_val:]
    
    data_time = time.time() - data_start
    timings['data_loading_time'] = data_time
    print(f"  Data loading: {data_time:.2f}s")
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(train_smiles_fit)} ({train_labels_fit.sum()} blockers, {100*train_labels_fit.mean():.1f}%)")
    print(f"  Val:   {len(val_smiles)} ({val_labels.sum()} blockers, {100*val_labels.mean():.1f}%)")
    print(f"  Test:  {len(test_smiles)} ({test_labels.sum()} blockers, {100*test_labels.mean():.1f}%)")
    
    results = {}
    
    # =========================================================================
    # PART 1: STATIC REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 1: STATIC REPRESENTATIONS (No Training)")
    print("="*100)
    
    # ECFP4
    print("\n[1/10] ECFP4 Fingerprints...")
    start = time.time()
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    rep_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(ecfp4_train, train_labels_fit)
    pred = rf.predict(ecfp4_test)
    prob = rf.predict_proba(ecfp4_test)[:, 1]  # Probability of blocker class
    results['ecfp4'] = evaluate_classification(test_labels, pred, prob, "ECFP4")
    
    total_time = time.time() - start
    timings['representations']['ecfp4'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': ecfp4_train.shape[1],
        'acc': results['ecfp4'][0],
        'auc': results['ecfp4'][1],
        'f1': results['ecfp4'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # PDV
    print("\n[2/10] PDV Descriptors...")
    start = time.time()
    pdv_train = create_pdv(train_smiles_fit)
    pdv_test = create_pdv(test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(pdv_train, train_labels_fit)
    pred = rf.predict(pdv_test)
    prob = rf.predict_proba(pdv_test)[:, 1]
    results['pdv'] = evaluate_classification(test_labels, pred, prob, "PDV")
    
    total_time = time.time() - start
    timings['representations']['pdv'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': pdv_train.shape[1],
        'acc': results['pdv'][0],
        'auc': results['pdv'][1],
        'f1': results['pdv'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # =========================================================================
    # PART 2: PRETRAINED REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 2: PRETRAINED REPRESENTATIONS")
    print("="*100)
    
    # mol2vec
    print("\n[3/10] mol2vec...")
    start = time.time()
    mol2vec_train = create_mol2vec(train_smiles_fit)
    mol2vec_test = create_mol2vec(test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(mol2vec_train, train_labels_fit)
    pred = rf.predict(mol2vec_test)
    prob = rf.predict_proba(mol2vec_test)[:, 1]
    results['mol2vec'] = evaluate_classification(test_labels, pred, prob, "mol2vec")
    
    total_time = time.time() - start
    timings['representations']['mol2vec'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': mol2vec_train.shape[1],
        'acc': results['mol2vec'][0],
        'auc': results['mol2vec'][1],
        'f1': results['mol2vec'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # Graph Kernel
    print("\n[4/10] Graph Kernel...")
    start = time.time()
    graphkernel_train, vocab = create_graph_kernel(
        train_smiles_fit, 
        kernel='weisfeiler_lehman', 
        n_features=100,
        return_vocabulary=True
    )
    graphkernel_test = create_graph_kernel(
        test_smiles, 
        kernel='weisfeiler_lehman', 
        n_features=100,
        reference_vocabulary=vocab
    )
    rep_time = time.time() - start

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(graphkernel_train, train_labels_fit)
    pred = rf.predict(graphkernel_test)
    prob = rf.predict_proba(graphkernel_test)[:, 1]
    results['graph_kernel'] = evaluate_classification(test_labels, pred, prob, "Graph Kernel (WL)")
    
    total_time = time.time() - start
    timings['representations']['graph_kernel'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': graphkernel_train.shape[1],
        'acc': results['graph_kernel'][0],
        'auc': results['graph_kernel'][1],
        'f1': results['graph_kernel'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # MHG-GNN
    print("\n[5/10] MHG-GNN (Pretrained)...")
    start = time.time()
    mhggnn_train = create_mhg_gnn(train_smiles_fit)
    mhggnn_test = create_mhg_gnn(test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(mhggnn_train, train_labels_fit)
    pred = rf.predict(mhggnn_test)
    prob = rf.predict_proba(mhggnn_test)[:, 1]
    results['mhggnn'] = evaluate_classification(test_labels, pred, prob, "MHG-GNN (pretrained)")
    
    total_time = time.time() - start
    timings['representations']['mhggnn'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': mhggnn_train.shape[1],
        'acc': results['mhggnn'][0],
        'auc': results['mhggnn'][1],
        'f1': results['mhggnn'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")

    
    # =========================================================================
    # PART 3: FINE-TUNED NEURAL REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 3: FINE-TUNED NEURAL REPRESENTATIONS")
    print("="*100)
    
    # TODO: Uncomment when on server with GPU
    # # ChemBERTa
    # print("\n[6/10] ChemBERTa (SMILES)...")
    # start = time.time()
    # chemberta_smiles_model = finetune_chemberta(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=3, batch_size=32, use_selfies=False  # TODO: Increase epochs for production
    # )
    # chemberta_smiles_train = encode_from_model(chemberta_smiles_model, train_smiles_fit)
    # chemberta_smiles_test = encode_from_model(chemberta_smiles_model, test_smiles)
    # rep_time = time.time() - start
    
    # rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    # rf.fit(chemberta_smiles_train, train_labels_fit)
    # pred = rf.predict(chemberta_smiles_test)
    # prob = rf.predict_proba(chemberta_smiles_test)[:, 1]
    # results['chemberta_smiles'] = evaluate_classification(test_labels, pred, prob, "ChemBERTa (SMILES, finetuned)")
    
    # total_time = time.time() - start
    # timings['representations']['chemberta_smiles'] = {
    #     'representation_time': rep_time,
    #     'total_time': total_time,
    #     'n_features': chemberta_smiles_train.shape[1],
    #     'acc': results['chemberta_smiles'][0],
    #     'auc': results['chemberta_smiles'][1],
    #     'f1': results['chemberta_smiles'][2]
    # }
    # print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # GNN
    print("\n[7/10] GNN (GCN)...")
    start = time.time()
    gnn_model = finetune_gnn(
        train_smiles_fit, train_labels_fit,
        val_smiles, val_labels,
        gnn_type='gcn', hidden_dim=128, epochs=50  # TODO: Increase for production
    )
    gnn_train = encode_from_model(gnn_model, train_smiles_fit)
    gnn_test = encode_from_model(gnn_model, test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(gnn_train, train_labels_fit)
    pred = rf.predict(gnn_test)
    prob = rf.predict_proba(gnn_test)[:, 1]
    results['gnn'] = evaluate_classification(test_labels, pred, prob, "GNN (GCN, finetuned)")
    
    total_time = time.time() - start
    timings['representations']['gnn'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': gnn_train.shape[1],
        'acc': results['gnn'][0],
        'auc': results['gnn'][1],
        'f1': results['gnn'][2]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # =========================================================================
    # PART 4: HYBRID REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 4: HYBRID REPRESENTATIONS")
    print("="*100)
    
    # Hybrid 1: ECFP4 + PDV
    print("\n[Hybrid 1] ECFP4 + PDV...")
    start = time.time()
    hybrid1_train = create_hybrid({'ecfp4': ecfp4_train, 'pdv': pdv_train})
    hybrid1_test = create_hybrid({'ecfp4': ecfp4_test, 'pdv': pdv_test})
    hybrid_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(hybrid1_train, train_labels_fit)
    pred = rf.predict(hybrid1_test)
    prob = rf.predict_proba(hybrid1_test)[:, 1]
    results['hybrid_ecfp4_pdv'] = evaluate_classification(test_labels, pred, prob, "Hybrid: ECFP4 + PDV")
    
    total_time = time.time() - start
    timings['hybrids']['ecfp4_pdv'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid1_train.shape[1],
        'components': ['ecfp4', 'pdv'],
        'acc': results['hybrid_ecfp4_pdv'][0],
        'auc': results['hybrid_ecfp4_pdv'][1],
        'f1': results['hybrid_ecfp4_pdv'][2]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # Hybrid 2: ECFP4 + Graph Kernel
    print("\n[Hybrid 2] ECFP4 + Graph Kernel...")
    start = time.time()
    hybrid2_train = create_hybrid({'ecfp4': ecfp4_train, 'graphkernel': graphkernel_train})
    hybrid2_test = create_hybrid({'ecfp4': ecfp4_test, 'graphkernel': graphkernel_test})
    hybrid_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(hybrid2_train, train_labels_fit)
    pred = rf.predict(hybrid2_test)
    prob = rf.predict_proba(hybrid2_test)[:, 1]
    results['hybrid_ecfp4_graphkernel'] = evaluate_classification(test_labels, pred, prob, "Hybrid: ECFP4 + GraphKernel")
    
    total_time = time.time() - start
    timings['hybrids']['ecfp4_graphkernel'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid2_train.shape[1],
        'components': ['ecfp4', 'graph_kernel'],
        'acc': results['hybrid_ecfp4_graphkernel'][0],
        'auc': results['hybrid_ecfp4_graphkernel'][1],
        'f1': results['hybrid_ecfp4_graphkernel'][2]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # # Hybrid 3: All Neural
    # print("\n[Hybrid 3] All Neural (ChemBERTa + GNN)...")
    # start = time.time()
    # hybrid3_train = create_hybrid({
    #     'chemberta': chemberta_smiles_train,
    #     'gnn': gnn_train
    # })
    # hybrid3_test = create_hybrid({
    #     'chemberta': chemberta_smiles_test,
    #     'gnn': gnn_test
    # })
    # hybrid_time = time.time() - start
    
    # rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    # rf.fit(hybrid3_train, train_labels_fit)
    # pred = rf.predict(hybrid3_test)
    # prob = rf.predict_proba(hybrid3_test)[:, 1]
    # results['hybrid_all_neural'] = evaluate_classification(test_labels, pred, prob, "Hybrid: All Neural")
    
    # total_time = time.time() - start
    # timings['hybrids']['all_neural'] = {
    #     'hybrid_creation_time': hybrid_time,
    #     'total_time': total_time,
    #     'n_features': hybrid3_train.shape[1],
    #     'components': ['chemberta', 'gnn'],
    #     'acc': results['hybrid_all_neural'][0],
    #     'auc': results['hybrid_all_neural'][1],
    #     'f1': results['hybrid_all_neural'][2]
    # }
    # print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # Hybrid 4: Everything
    print("\n[Hybrid 4] Everything...")
    start = time.time()
    hybrid4_train = create_hybrid({
        'ecfp4': ecfp4_train,
        'pdv': pdv_train,
        'mol2vec': mol2vec_train,
        'graphkernel': graphkernel_train,
        'mhggnn': mhggnn_train,
        # 'chemberta': chemberta_smiles_train,
        'gnn': gnn_train
    })
    hybrid4_test = create_hybrid({
        'ecfp4': ecfp4_test,
        'pdv': pdv_test,
        'mol2vec': mol2vec_test,
        'graphkernel': graphkernel_test,
        'mhggnn': mhggnn_test,
        # 'chemberta': chemberta_smiles_test,
        'gnn': gnn_test
    })
    hybrid_time = time.time() - start
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(hybrid4_train, train_labels_fit)
    pred = rf.predict(hybrid4_test)
    prob = rf.predict_proba(hybrid4_test)[:, 1]
    results['hybrid_everything'] = evaluate_classification(test_labels, pred, prob, "Hybrid: Everything")
    
    total_time = time.time() - start
    timings['hybrids']['everything'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid4_train.shape[1],
        'components': ['ecfp4', 'pdv', 'mol2vec', 'graph_kernel', 'mhggnn', 'gnn'],
        'acc': results['hybrid_everything'][0],
        'auc': results['hybrid_everything'][1],
        'f1': results['hybrid_everything'][2]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    timings['total_time'] = time.time() - overall_start
    timings['end_time'] = datetime.now().isoformat()
    
    print("\n" + "="*100)
    print("SUMMARY - hERG CARDIOTOXICITY PREDICTION")
    print("="*100)
    
    # Sort by AUC (best metric for imbalanced classification)
    sorted_results = sorted(results.items(), key=lambda x: x[1][1] if x[1][1] is not None else 0, reverse=True)
    
    print(f"\n{'Representation':<40} {'ACC':>8} {'AUC':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 100)
    
    for i, (name, (acc, auc, f1, mcc)) in enumerate(sorted_results, 1):
        marker = "🏆" if i == 1 else "⭐" if i <= 3 else "  "
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{marker} {name:<37} {acc:>8.4f} {auc_str:>8} {f1:>8.4f} {mcc:>8.4f}")
    
    print(f"\n{'='*100}")
    print(f"TOTAL TIME: {timings['total_time']:.2f}s ({timings['total_time']/60:.2f}min)")
    print(f"{'='*100}")
    
    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("="*100)
    
    # Best model
    best_name, (best_acc, best_auc, best_f1, best_mcc) = sorted_results[0]
    print(f"1. BEST: {best_name}")
    print(f"   ACC={best_acc:.4f}, AUC={best_auc:.4f}, F1={best_f1:.4f}, MCC={best_mcc:.4f}")
    
    # Class balance info
    test_blocker_rate = test_labels.mean()
    print(f"\n2. CLASS BALANCE: Test set is {100*test_blocker_rate:.1f}% blockers")
    print(f"   Random baseline: ACC={max(test_blocker_rate, 1-test_blocker_rate):.4f}, AUC=0.5")
    
    # Compare baseline vs SOTA
    baseline_results = {k: v for k, v in results.items() if k in ['ecfp4', 'pdv']}
    if baseline_results:
        baseline_best = max(baseline_results.items(), key=lambda x: x[1][1] if x[1][1] else 0)
        print(f"\n3. BASELINE: {baseline_best[0]} achieved AUC={baseline_best[1][1]:.4f}")
    
    # Hybrid performance
    hybrid_results = {k: v for k, v in results.items() if k.startswith('hybrid')}
    if hybrid_results:
        hybrid_best = max(hybrid_results.items(), key=lambda x: x[1][1] if x[1][1] else 0)
        print(f"\n4. BEST HYBRID: {hybrid_best[0]}")
        print(f"   AUC={hybrid_best[1][1]:.4f}")
    
    print("\n" + "="*100)
    print("DONE! Results ready for hERG cardiotoxicity prediction benchmarking.")
    print("="*100)
    
    # Save timing results
    output_file = f'herg_{args.source}_timing_results.json'
    with open(output_file, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"\nTiming results saved to: {output_file}")


if __name__ == '__main__':
    main()
