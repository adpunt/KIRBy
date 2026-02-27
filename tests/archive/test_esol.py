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
    
    # Load data
    print("\nLoading ESOL dataset...")
    data = load_esol_combined(splitter='scaffold')
    
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
    ecfp4_train = create_ecfp4(train_smiles_fit, n_bits=2048)
    ecfp4_test = create_ecfp4(test_smiles, n_bits=2048)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(ecfp4_train, train_labels_fit)
    pred = rf.predict(ecfp4_test)
    results['ecfp4'] = evaluate(test_labels, pred, "ECFP4")
    
    # PDV
    print("\n[2/10] PDV Descriptors...")
    pdv_train = create_pdv(train_smiles_fit)
    pdv_test = create_pdv(test_smiles)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(pdv_train, train_labels_fit)
    pred = rf.predict(pdv_test)
    results['pdv'] = evaluate(test_labels, pred, "PDV")
    
    # =========================================================================
    # PART 2: PRETRAINED REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 2: PRETRAINED REPRESENTATIONS")
    print("="*100)
    
    # mol2vec
    print("\n[3/10] mol2vec...")
    mol2vec_train = create_mol2vec(train_smiles_fit)
    mol2vec_test = create_mol2vec(test_smiles)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(mol2vec_train, train_labels_fit)
    pred = rf.predict(mol2vec_test)
    results['mol2vec'] = evaluate(test_labels, pred, "mol2vec")
    
    # Graph Kernel
    print("\n[4/10] Graph Kernel...")
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

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(graphkernel_train, train_labels_fit)
    pred = rf.predict(graphkernel_test)
    results['graph_kernel'] = evaluate(test_labels, pred, "Graph Kernel (WL)")
    
    # MHG-GNN
    print("\n[5/10] MHG-GNN (Pretrained)...")
    mhggnn_train = create_mhg_gnn(train_smiles_fit)  # Use full 1024-dim embeddings
    mhggnn_test = create_mhg_gnn(test_smiles)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(mhggnn_train, train_labels_fit)
    pred = rf.predict(mhggnn_test)
    results['mhggnn'] = evaluate(test_labels, pred, "MHG-GNN (pretrained)")

    
    # =========================================================================
    # PART 3: FINE-TUNED NEURAL REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 3: FINE-TUNED NEURAL REPRESENTATIONS")
    print("="*100)
    
    # TODO: uncomment this when you get to the server
    # # MoLFormer (SMILES)
    # print("\n[6/10] MoLFormer (SMILES)...")
    # molformer_smiles_model = finetune_molformer(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=10, batch_size=32, use_selfies=False
    # )
    # molformer_smiles_train = encode_from_model(molformer_smiles_model, train_smiles_fit)
    # molformer_smiles_test = encode_from_model(molformer_smiles_model, test_smiles)
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(molformer_smiles_train, train_labels_fit)
    # pred = rf.predict(molformer_smiles_test)
    # results['molformer_smiles'] = evaluate(test_labels, pred, "MoLFormer (SMILES, finetuned)")
    
    # # MoLFormer (SELFIES)
    # print("\n[7/10] MoLFormer (SELFIES)...")
    # molformer_selfies_model = finetune_molformer(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=10, batch_size=32, use_selfies=True
    # )
    # molformer_selfies_train = encode_from_model(molformer_selfies_model, train_smiles_fit)
    # molformer_selfies_test = encode_from_model(molformer_selfies_model, test_smiles)
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(molformer_selfies_train, train_labels_fit)
    # pred = rf.predict(molformer_selfies_test)
    # results['molformer_selfies'] = evaluate(test_labels, pred, "MoLFormer (SELFIES, finetuned)")
    
    # TODO: change epochs
    # # ChemBERTa (SMILES)
    # print("\n[8/10] ChemBERTa (SMILES)...")
    # chemberta_smiles_model = finetune_chemberta(
    #     train_smiles_fit, train_labels_fit,
    #     val_smiles, val_labels,
    #     epochs=1, batch_size=32, use_selfies=False
    # )
    # chemberta_smiles_train = encode_from_model(chemberta_smiles_model, train_smiles_fit)
    # chemberta_smiles_test = encode_from_model(chemberta_smiles_model, test_smiles)
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(chemberta_smiles_train, train_labels_fit)
    # pred = rf.predict(chemberta_smiles_test)
    # results['chemberta_smiles'] = evaluate(test_labels, pred, "ChemBERTa (SMILES, finetuned)")
    
    # GNN
    print("\n[9/10] GNN (GCN)...")
    gnn_model = finetune_gnn(
        train_smiles_fit, train_labels_fit,
        val_smiles, val_labels,
        gnn_type='gcn', hidden_dim=128, epochs=150
    )
    gnn_train = encode_from_model(gnn_model, train_smiles_fit)
    gnn_test = encode_from_model(gnn_model, test_smiles)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(gnn_train, train_labels_fit)
    pred = rf.predict(gnn_test)
    results['gnn'] = evaluate(test_labels, pred, "GNN (GCN, finetuned)")
    
    # GAUCHE GP
    print("\n[10/10] GAUCHE GP...")
    gp_dict = train_gauche_gp(
        train_smiles_fit, train_labels_fit,
        kernel='weisfeiler_lehman',
        num_epochs=50
    )
    gp_results = predict_gauche_gp(gp_dict, test_smiles)
    results['gauche_gp'] = evaluate(test_labels, gp_results['predictions'], "GAUCHE GP (WL kernel)")
    
    # =========================================================================
    # PART 4: HYBRID REPRESENTATIONS
    # =========================================================================
    print("\n" + "="*100)
    print("PART 4: HYBRID REPRESENTATIONS")
    print("="*100)
    
    # Hybrid 1: ECFP4 + PDV
    print("\n[Hybrid 1] ECFP4 + PDV...")
    hybrid1_train = create_hybrid({'ecfp4': ecfp4_train, 'pdv': pdv_train})
    hybrid1_test = create_hybrid({'ecfp4': ecfp4_test, 'pdv': pdv_test})
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid1_train, train_labels_fit)
    pred = rf.predict(hybrid1_test)
    results['hybrid_ecfp4_pdv'] = evaluate(test_labels, pred, "Hybrid: ECFP4 + PDV")
    
    # Hybrid 2: ECFP4 + Graph Kernel
    print("\n[Hybrid 2] ECFP4 + Graph Kernel...")
    hybrid2_train = create_hybrid({'ecfp4': ecfp4_train, 'graphkernel': graphkernel_train})
    hybrid2_test = create_hybrid({'ecfp4': ecfp4_test, 'graphkernel': graphkernel_test})
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid2_train, train_labels_fit)
    pred = rf.predict(hybrid2_test)
    results['hybrid_ecfp4_graphkernel'] = evaluate(test_labels, pred, "Hybrid: ECFP4 + GraphKernel")
    
    # Hybrid 3: All Neural
    # print("\n[Hybrid 3] All Neural (MoLFormer + ChemBERTa + GNN)...")
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
    
    # rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # rf.fit(hybrid3_train, train_labels_fit)
    # pred = rf.predict(hybrid3_test)
    # results['hybrid_all_neural'] = evaluate(test_labels, pred, "Hybrid: All Neural")
    
    # Hybrid 4: Everything
    print("\n[Hybrid 4] Everything...")
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
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(hybrid4_train, train_labels_fit)
    pred = rf.predict(hybrid4_test)
    results['hybrid_everything'] = evaluate(test_labels, pred, "Hybrid: Everything")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])  # Sort by RMSE
    
    print("\nRanked by RMSE (best first):")
    for i, (name, (rmse, r2)) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:40s} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    main()