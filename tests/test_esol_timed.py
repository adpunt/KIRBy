import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
Test ESOL dataset with ALL KIRBy representations.

Tests:
1. Every single representation individually
2. Sample hybrid combinations covering the representation space
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import json
from datetime import datetime

import sys
sys.path.insert(0, '/mnt/user-data/outputs/src')

from kirby.datasets.esol import load_esol_combined
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
    train_gauche_gp,
    predict_gauche_gp,
    create_hybrid
)


def evaluate(y_true, y_pred, name="Model"):
    """Calculate and print metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:40s} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rmse, r2


def main():
    print("="*100)
    print("ESOL Dataset - Complete KIRBy Test Suite")
    print("="*100)
    
    # Track all timings
    timings = {
        'dataset': 'esol',
        'start_time': datetime.now().isoformat(),
        'representations': {},
        'hybrids': {},
        'total_time': 0
    }
    
    overall_start = time.time()
    
    # Load data
    print("\nLoading ESOL dataset...")
    data_start = time.time()
    data = load_esol_combined(splitter='scaffold')
    data_time = time.time() - data_start
    timings['data_loading_time'] = data_time
    print(f"  Data loading: {data_time:.2f}s")
    
    train_smiles = data['train']['smiles']
    train_labels = data['train']['labels']
    test_smiles = data['test']['smiles']
    test_labels = data['test']['labels']
    
    # Validation split
    n_val = len(train_smiles) // 5
    val_smiles = train_smiles[:n_val]
    val_labels = train_labels[:n_val]
    train_smiles_fit = train_smiles[n_val:]
    train_labels_fit = train_labels[n_val:]
    
    print(f"Train: {len(train_smiles_fit)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")
    
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
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(ecfp4_train, train_labels_fit)
    pred = rf.predict(ecfp4_test)
    results['ecfp4'] = evaluate(test_labels, pred, "ECFP4")
    
    total_time = time.time() - start
    timings['representations']['ecfp4'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': ecfp4_train.shape[1],
        'rmse': results['ecfp4'][0],
        'r2': results['ecfp4'][1]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # PDV
    print("\n[2/10] PDV Descriptors...")
    start = time.time()
    pdv_train = create_pdv(train_smiles_fit)
    pdv_test = create_pdv(test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(pdv_train, train_labels_fit)
    pred = rf.predict(pdv_test)
    results['pdv'] = evaluate(test_labels, pred, "PDV")
    
    total_time = time.time() - start
    timings['representations']['pdv'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': pdv_train.shape[1],
        'rmse': results['pdv'][0],
        'r2': results['pdv'][1]
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
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(mol2vec_train, train_labels_fit)
    pred = rf.predict(mol2vec_test)
    results['mol2vec'] = evaluate(test_labels, pred, "mol2vec")
    
    total_time = time.time() - start
    timings['representations']['mol2vec'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': mol2vec_train.shape[1],
        'rmse': results['mol2vec'][0],
        'r2': results['mol2vec'][1]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # Graph Kernel
    print("\n[4/10] Graph Kernel...")
    start = time.time()
    graphkernel_train, vocab = create_graph_kernel(
        train_smiles_fit, 
        kernel='weisfeiler_lehman', 
        n_features=100,
        return_vocabulary=True  # Get vocabulary from training
    )
    graphkernel_test = create_graph_kernel(
        test_smiles, 
        kernel='weisfeiler_lehman', 
        n_features=100,
        reference_vocabulary=vocab  # Use training vocabulary!
    )
    rep_time = time.time() - start

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(graphkernel_train, train_labels_fit)
    pred = rf.predict(graphkernel_test)
    results['graph_kernel'] = evaluate(test_labels, pred, "Graph Kernel (WL)")
    
    total_time = time.time() - start
    timings['representations']['graph_kernel'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': graphkernel_train.shape[1],
        'rmse': results['graph_kernel'][0],
        'r2': results['graph_kernel'][1]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # MHG-GNN
    print("\n[5/10] MHG-GNN (Pretrained)...")
    start = time.time()
    mhggnn_train = create_mhg_gnn(train_smiles_fit)  # Use full 1024-dim embeddings
    mhggnn_test = create_mhg_gnn(test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(mhggnn_train, train_labels_fit)
    pred = rf.predict(mhggnn_test)
    results['mhggnn'] = evaluate(test_labels, pred, "MHG-GNN (pretrained)")
    
    total_time = time.time() - start
    timings['representations']['mhggnn'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': mhggnn_train.shape[1],
        'rmse': results['mhggnn'][0],
        'r2': results['mhggnn'][1]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")

    
    # =========================================================================
    # PART 3: FINE-TUNED NEURAL REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 3: FINE-TUNED NEURAL REPRESENTATIONS")
    print("="*100)
    
    # TODO: uncomment this when you get to the server
    # # MoLFormer (SMILES)
    # print("\n[6/10] MoLFormer (SMILES)...")
    # start = time.time()
    # molformer_smiles_model = finetune_molformer(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=10, batch_size=32, use_selfies=False
    # )
    # molformer_smiles_train = encode_from_model(molformer_smiles_model, train_smiles_fit)
    # molformer_smiles_test = encode_from_model(molformer_smiles_model, test_smiles)
    # rep_time = time.time() - start
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(molformer_smiles_train, train_labels_fit)
    # pred = rf.predict(molformer_smiles_test)
    # results['molformer_smiles'] = evaluate(test_labels, pred, "MoLFormer (SMILES, finetuned)")
    
    # total_time = time.time() - start
    # timings['representations']['molformer_smiles'] = {
    #     'representation_time': rep_time,
    #     'total_time': total_time,
    #     'n_features': molformer_smiles_train.shape[1],
    #     'rmse': results['molformer_smiles'][0],
    #     'r2': results['molformer_smiles'][1]
    # }
    # print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # # MoLFormer (SELFIES)
    # print("\n[7/10] MoLFormer (SELFIES)...")
    # start = time.time()
    # molformer_selfies_model = finetune_molformer(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=10, batch_size=32, use_selfies=True
    # )
    # molformer_selfies_train = encode_from_model(molformer_selfies_model, train_smiles_fit)
    # molformer_selfies_test = encode_from_model(molformer_selfies_model, test_smiles)
    # rep_time = time.time() - start
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(molformer_selfies_train, train_labels_fit)
    # pred = rf.predict(molformer_selfies_test)
    # results['molformer_selfies'] = evaluate(test_labels, pred, "MoLFormer (SELFIES, finetuned)")
    
    # total_time = time.time() - start
    # timings['representations']['molformer_selfies'] = {
    #     'representation_time': rep_time,
    #     'total_time': total_time,
    #     'n_features': molformer_selfies_train.shape[1],
    #     'rmse': results['molformer_selfies'][0],
    #     'r2': results['molformer_selfies'][1]
    # }
    # print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # TODO: change epochs
    # # ChemBERTa (SMILES)
    # print("\n[8/10] ChemBERTa (SMILES)...")
    # start = time.time()
    # chemberta_smiles_model = finetune_chemberta(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=1, batch_size=32, use_selfies=False
    # )
    # chemberta_smiles_train = encode_from_model(chemberta_smiles_model, train_smiles_fit)
    # chemberta_smiles_test = encode_from_model(chemberta_smiles_model, test_smiles)
    # rep_time = time.time() - start
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(chemberta_smiles_train, train_labels_fit)
    # pred = rf.predict(chemberta_smiles_test)
    # results['chemberta_smiles'] = evaluate(test_labels, pred, "ChemBERTa (SMILES, finetuned)")
    
    # total_time = time.time() - start
    # timings['representations']['chemberta_smiles'] = {
    #     'representation_time': rep_time,
    #     'total_time': total_time,
    #     'n_features': chemberta_smiles_train.shape[1],
    #     'rmse': results['chemberta_smiles'][0],
    #     'r2': results['chemberta_smiles'][1]
    # }
    # print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # GNN
    print("\n[9/10] GNN (GCN)...")
    start = time.time()
    gnn_model = finetune_gnn(
        train_smiles_fit, train_labels_fit,
        val_smiles, val_labels,
        gnn_type='gcn', hidden_dim=128, epochs=150
    )
    gnn_train = encode_from_model(gnn_model, train_smiles_fit)
    gnn_test = encode_from_model(gnn_model, test_smiles)
    rep_time = time.time() - start
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(gnn_train, train_labels_fit)
    pred = rf.predict(gnn_test)
    results['gnn'] = evaluate(test_labels, pred, "GNN (GCN, finetuned)")
    
    total_time = time.time() - start
    timings['representations']['gnn'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': gnn_train.shape[1],
        'rmse': results['gnn'][0],
        'r2': results['gnn'][1]
    }
    print(f"  Time: {total_time:.2f}s (rep: {rep_time:.2f}s)")
    
    # GAUCHE GP
    print("\n[10/10] GAUCHE GP...")
    start = time.time()
    gp_dict = train_gauche_gp(
        train_smiles_fit, train_labels_fit,
        kernel='weisfeiler_lehman',
        num_epochs=50
    )
    gp_results = predict_gauche_gp(gp_dict, test_smiles)
    rep_time = time.time() - start
    
    results['gauche_gp'] = evaluate(test_labels, gp_results['predictions'], "GAUCHE GP (WL kernel)")
    
    total_time = time.time() - start
    timings['representations']['gauche_gp'] = {
        'representation_time': rep_time,
        'total_time': total_time,
        'n_features': 'N/A (GP)',
        'rmse': results['gauche_gp'][0],
        'r2': results['gauche_gp'][1]
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
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid1_train, train_labels_fit)
    pred = rf.predict(hybrid1_test)
    results['hybrid_ecfp4_pdv'] = evaluate(test_labels, pred, "Hybrid: ECFP4 + PDV")
    
    total_time = time.time() - start
    timings['hybrids']['ecfp4_pdv'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid1_train.shape[1],
        'components': ['ecfp4', 'pdv'],
        'rmse': results['hybrid_ecfp4_pdv'][0],
        'r2': results['hybrid_ecfp4_pdv'][1]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # Hybrid 2: ECFP4 + Graph Kernel
    print("\n[Hybrid 2] ECFP4 + Graph Kernel...")
    start = time.time()
    hybrid2_train = create_hybrid({'ecfp4': ecfp4_train, 'graphkernel': graphkernel_train})
    hybrid2_test = create_hybrid({'ecfp4': ecfp4_test, 'graphkernel': graphkernel_test})
    hybrid_time = time.time() - start
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid2_train, train_labels_fit)
    pred = rf.predict(hybrid2_test)
    results['hybrid_ecfp4_graphkernel'] = evaluate(test_labels, pred, "Hybrid: ECFP4 + GraphKernel")
    
    total_time = time.time() - start
    timings['hybrids']['ecfp4_graphkernel'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid2_train.shape[1],
        'components': ['ecfp4', 'graph_kernel'],
        'rmse': results['hybrid_ecfp4_graphkernel'][0],
        'r2': results['hybrid_ecfp4_graphkernel'][1]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # Hybrid 3: All Neural
    # print("\n[Hybrid 3] All Neural (MoLFormer + ChemBERTa + GNN)...")
    # start = time.time()
    # hybrid3_train = create_hybrid({
    #     'molformer': molformer_smiles_train,
    #     'chemberta': chemberta_smiles_train,
    #     'gnn': gnn_train
    # })
    # hybrid3_test = create_hybrid({
    #     'molformer': molformer_smiles_test,
    #     'chemberta': chemberta_smiles_test,
    #     'gnn': gnn_test
    # })
    # hybrid_time = time.time() - start
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(hybrid3_train, train_labels_fit)
    # pred = rf.predict(hybrid3_test)
    # results['hybrid_all_neural'] = evaluate(test_labels, pred, "Hybrid: All Neural")
    
    # total_time = time.time() - start
    # timings['hybrids']['all_neural'] = {
    #     'hybrid_creation_time': hybrid_time,
    #     'total_time': total_time,
    #     'n_features': hybrid3_train.shape[1],
    #     'components': ['molformer', 'chemberta', 'gnn'],
    #     'rmse': results['hybrid_all_neural'][0],
    #     'r2': results['hybrid_all_neural'][1]
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
        # 'molformer': molformer_smiles_train,
        # 'chemberta': chemberta_smiles_train,
        'gnn': gnn_train
    })
    hybrid4_test = create_hybrid({
        'ecfp4': ecfp4_test,
        'pdv': pdv_test,
        'mol2vec': mol2vec_test,
        'graphkernel': graphkernel_test,
        'mhggnn': mhggnn_test,
        # 'molformer': molformer_smiles_test,
        # 'chemberta': chemberta_smiles_test,
        'gnn': gnn_test
    })
    hybrid_time = time.time() - start
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid4_train, train_labels_fit)
    pred = rf.predict(hybrid4_test)
    results['hybrid_everything'] = evaluate(test_labels, pred, "Hybrid: Everything")
    
    total_time = time.time() - start
    timings['hybrids']['everything'] = {
        'hybrid_creation_time': hybrid_time,
        'total_time': total_time,
        'n_features': hybrid4_train.shape[1],
        'components': ['ecfp4', 'pdv', 'mol2vec', 'graph_kernel', 'mhggnn', 'gnn'],
        'rmse': results['hybrid_everything'][0],
        'r2': results['hybrid_everything'][1]
    }
    print(f"  Time: {total_time:.2f}s (hybrid: {hybrid_time:.2f}s)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    timings['total_time'] = time.time() - overall_start
    timings['end_time'] = datetime.now().isoformat()
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])  # Sort by RMSE
    
    print("\nRanked by RMSE (best first):")
    for i, (name, (rmse, r2)) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:40s} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    print(f"\n{'='*100}")
    print(f"TOTAL TIME: {timings['total_time']:.2f}s ({timings['total_time']/60:.2f}min)")
    print(f"{'='*100}")
    
    # Save timing results
    output_file = 'esol_timing_results.json'
    with open(output_file, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"\nTiming results saved to: {output_file}")


if __name__ == '__main__':
    main()
