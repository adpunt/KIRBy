#!/usr/bin/env python3
"""
TCGA Lung Cancer - SOTA Multimodal Fusion Suite

Tests KIRBy hypothesis: Can hybrids of SOTA embeddings + classical features
beat SOTA embeddings alone?

Implements multiple SOTA architectures:
  - Bilinear fusion (PORPOISE)
  - Cross-modal attention (MCAT)
  - Gated fusion
  - Tensor fusion
  - Ensemble fusion
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import h5py
import gzip
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             roc_auc_score, f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# TIMING IMPORTS
import time
import json
from datetime import datetime

# KIRBy imports
import sys
sys.path.append('../src')
from kirby.hybrid import create_hybrid

# ============================================================================
# SOTA MULTIMODAL FUSION MODELS
# ============================================================================

class SelfNormalizingNetwork(nn.Module):
    """Self-normalizing network for genomic features (SELU activation)"""
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, dropout=0.25):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # SELU initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


class VariationalGenomicEncoder(nn.Module):
    """Variational encoder for genomic features with reparameterization"""
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256]):
        super().__init__()
        
        # Encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.25)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Variational parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class GraphGenomicEncoder(nn.Module):
    """Graph-based encoder for genomic features (pathway structure)"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, n_layers=3):
        super().__init__()
        
        # Simple GCN-like layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        # Without explicit graph structure, use self-attention
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        x = self.norm(x)
        return x


class BilinearFusion(nn.Module):
    """Bilinear pooling (Kronecker product) for multimodal fusion"""
    def __init__(self, dim1=128, dim2=128, fusion_dim=128, n_classes=2):
        super().__init__()
        
        bilinear_dim = dim1 * dim2
        
        self.fusion = nn.Sequential(
            nn.Linear(bilinear_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(fusion_dim, n_classes)
        )
    
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x1_expanded = x1.unsqueeze(2)      # (batch, dim1, 1)
        x2_expanded = x2.unsqueeze(1)      # (batch, 1, dim2)
        bilinear = torch.bmm(x1_expanded, x2_expanded)  # (batch, dim1, dim2)
        bilinear = bilinear.view(batch_size, -1)
        return self.fusion(bilinear)


class CrossModalAttention(nn.Module):
    """Cross-modal attention (MCAT-style) - each modality attends to the other"""
    def __init__(self, dim1=128, dim2=128, n_heads=4):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = dim1 // n_heads
        
        # Modality 1 attends to Modality 2
        self.q1 = nn.Linear(dim1, dim1)
        self.k2 = nn.Linear(dim2, dim1)
        self.v2 = nn.Linear(dim2, dim1)
        
        # Modality 2 attends to Modality 1
        self.q2 = nn.Linear(dim2, dim2)
        self.k1 = nn.Linear(dim1, dim2)
        self.v1 = nn.Linear(dim1, dim2)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # x1 attends to x2
        q1 = self.q1(x1).view(batch_size, self.n_heads, self.head_dim)
        k2 = self.k2(x2).view(batch_size, self.n_heads, self.head_dim)
        v2 = self.v2(x2).view(batch_size, self.n_heads, self.head_dim)
        
        attn1 = torch.softmax(torch.sum(q1 * k2, dim=-1, keepdim=True) * self.scale, dim=1)
        x1_attended = (attn1 * v2).view(batch_size, -1)
        
        # x2 attends to x1
        q2 = self.q2(x2).view(batch_size, self.n_heads, self.head_dim)
        k1 = self.k1(x1).view(batch_size, self.n_heads, self.head_dim)
        v1 = self.v1(x1).view(batch_size, self.n_heads, self.head_dim)
        
        attn2 = torch.softmax(torch.sum(q2 * k1, dim=-1, keepdim=True) * self.scale, dim=1)
        x2_attended = (attn2 * v1).view(batch_size, -1)
        
        return x1_attended, x2_attended


class GatedFusion(nn.Module):
    """Gated fusion - learns what to attend to from each modality"""
    def __init__(self, dim1=128, dim2=128, n_classes=2):
        super().__init__()
        
        combined_dim = dim1 + dim2
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gate = self.gate(combined)
        gated = combined * gate
        return self.classifier(gated)


class TensorFusion(nn.Module):
    """Tensor fusion - captures higher-order interactions"""
    def __init__(self, dim1=128, dim2=128, fusion_dim=128, n_classes=2):
        super().__init__()
        
        # Add bias term for outer product
        self.dim1 = dim1 + 1
        self.dim2 = dim2 + 1
        tensor_dim = self.dim1 * self.dim2
        
        self.fusion = nn.Sequential(
            nn.Linear(tensor_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(fusion_dim, n_classes)
        )
    
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # Add bias term
        x1_bias = torch.cat([x1, torch.ones(batch_size, 1).to(x1.device)], dim=1)
        x2_bias = torch.cat([x2, torch.ones(batch_size, 1).to(x2.device)], dim=1)
        
        # Tensor fusion
        x1_expanded = x1_bias.unsqueeze(2)
        x2_expanded = x2_bias.unsqueeze(1)
        tensor = torch.bmm(x1_expanded, x2_expanded)
        tensor = tensor.view(batch_size, -1)
        
        return self.fusion(tensor)


class MultimodalClassifier(nn.Module):
    """Base multimodal classifier - choose encoder and fusion type"""
    def __init__(self, genomic_dim, imaging_dim, hidden_dim=128, 
                 genomic_encoder='snn', fusion_type='bilinear', n_classes=2):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Genomic encoders
        if genomic_encoder == 'snn':
            self.genomic_encoder = SelfNormalizingNetwork(
                input_dim=genomic_dim, hidden_dims=[512, 256], output_dim=hidden_dim
            )
        elif genomic_encoder == 'variational':
            self.genomic_encoder = VariationalGenomicEncoder(
                input_dim=genomic_dim, latent_dim=hidden_dim, hidden_dims=[512, 256]
            )
            self.use_vae = True
        elif genomic_encoder == 'graph':
            self.genomic_encoder = GraphGenomicEncoder(
                input_dim=genomic_dim, hidden_dim=256, output_dim=hidden_dim, n_layers=3
            )
        else:
            raise ValueError(f"Unknown genomic encoder: {genomic_encoder}")
        
        self.genomic_encoder_type = genomic_encoder
        
        # Imaging encoder
        self.imaging_encoder = nn.Sequential(
            nn.Linear(imaging_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, hidden_dim)
        )
        
        # Fusion modules
        if fusion_type == 'bilinear':
            self.fusion = BilinearFusion(hidden_dim, hidden_dim, 128, n_classes)
        elif fusion_type == 'cross_attention':
            self.cross_attn = CrossModalAttention(hidden_dim, hidden_dim, n_heads=4)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, n_classes)
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(hidden_dim, hidden_dim, n_classes)
        elif fusion_type == 'tensor':
            self.fusion = TensorFusion(hidden_dim, hidden_dim, 128, n_classes)
        elif fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, n_classes)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, genomic, imaging):
        # Encode genomic
        if self.genomic_encoder_type == 'variational':
            genomic_emb, mu, logvar = self.genomic_encoder(genomic)
        else:
            genomic_emb = self.genomic_encoder(genomic)
        
        # Encode imaging
        imaging_emb = self.imaging_encoder(imaging)
        
        # Fuse
        if self.fusion_type == 'cross_attention':
            g_att, i_att = self.cross_attn(genomic_emb, imaging_emb)
            combined = torch.cat([g_att, i_att], dim=1)
            logits = self.fusion(combined)
        elif self.fusion_type == 'concat':
            combined = torch.cat([genomic_emb, imaging_emb], dim=1)
            logits = self.fusion(combined)
        else:
            logits = self.fusion(genomic_emb, imaging_emb)
        
        return logits
    
    def get_embeddings(self, genomic, imaging):
        """Extract embeddings before fusion"""
        self.eval()
        with torch.no_grad():
            # Encode genomic
            if self.genomic_encoder_type == 'variational':
                genomic_emb, _, _ = self.genomic_encoder(genomic)
            else:
                genomic_emb = self.genomic_encoder(genomic)
            
            # Encode imaging
            imaging_emb = self.imaging_encoder(imaging)
        
        return genomic_emb, imaging_emb


class EnsembleMultimodal(nn.Module):
    """Ensemble of all encoder/fusion combinations"""
    def __init__(self, genomic_dim, imaging_dim, hidden_dim=128):
        super().__init__()
        
        # Only use SNN and Variational encoders (transformers/graph perform poorly)
        configs = [
            ('snn', 'concat'),
            ('snn', 'bilinear'), 
            ('snn', 'cross_attention'), 
            ('snn', 'gated'),
            ('variational', 'bilinear'), 
            ('variational', 'cross_attention'),
        ]
        
        self.models = nn.ModuleList([
            MultimodalClassifier(genomic_dim, imaging_dim, hidden_dim, 
                               genomic_encoder=enc, fusion_type=fus, n_classes=2)
            for enc, fus in configs
        ])
        
        # Ensemble weights (learned)
        self.ensemble_weights = nn.Parameter(torch.ones(len(configs)) / len(configs))
    
    def forward(self, genomic, imaging):
        # Get predictions from all models
        logits_list = [model(genomic, imaging) for model in self.models]
        logits_stack = torch.stack(logits_list, dim=0)  # (n_models, batch, 2)
        
        # Weighted average with softmax weights
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        logits = torch.sum(logits_stack * weights, dim=0)
        
        return logits


def extract_sota_embeddings(model, genomic_data, imaging_data, scaler_g, scaler_i):
    """Extract learned embeddings from trained SOTA model for full dataset"""
    model.eval()
    
    # Normalize full dataset using training scalers
    genomic_scaled = scaler_g.transform(genomic_data)
    imaging_scaled = scaler_i.transform(imaging_data)
    
    genomic_t = torch.FloatTensor(genomic_scaled).to(device)
    imaging_t = torch.FloatTensor(imaging_scaled).to(device)
    
    genomic_emb, imaging_emb = model.get_embeddings(genomic_t, imaging_t)
    
    return genomic_emb.cpu().numpy(), imaging_emb.cpu().numpy()


def train_sota_model(genomic_data, imaging_data, labels, 
                    genomic_encoder='snn', fusion_type='bilinear', 
                    epochs=50, use_ensemble=False):
    """Train SOTA multimodal model and return model + embeddings"""
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, 
                                          random_state=42, stratify=labels)
    
    # Normalize
    scaler_g = StandardScaler()
    scaler_i = StandardScaler()
    
    genomic_train = scaler_g.fit_transform(genomic_data[train_idx])
    genomic_test = scaler_g.transform(genomic_data[test_idx])
    
    imaging_train = scaler_i.fit_transform(imaging_data[train_idx])
    imaging_test = scaler_i.transform(imaging_data[test_idx])
    
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Convert to tensors
    genomic_train_t = torch.FloatTensor(genomic_train).to(device)
    imaging_train_t = torch.FloatTensor(imaging_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    genomic_test_t = torch.FloatTensor(genomic_test).to(device)
    imaging_test_t = torch.FloatTensor(imaging_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Initialize model
    if use_ensemble:
        model = EnsembleMultimodal(
            genomic_dim=genomic_train.shape[1],
            imaging_dim=imaging_train.shape[1],
            hidden_dim=128
        ).to(device)
        model_name = "Ensemble"
    else:
        model = MultimodalClassifier(
            genomic_dim=genomic_train.shape[1],
            imaging_dim=imaging_train.shape[1],
            hidden_dim=128,
            genomic_encoder=genomic_encoder,
            fusion_type=fusion_type
        ).to(device)
        model_name = f"{genomic_encoder}+{fusion_type}"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    print(f"\nTraining {model_name} for {epochs} epochs...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_acc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(genomic_train_t, imaging_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = logits.argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            
            test_logits = model(genomic_test_t, imaging_test_t)
            test_pred = test_logits.argmax(dim=1)
            test_acc = (test_pred == y_test_t).float().mean().item()
            test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:02d}: Loss={loss.item():.4f}  "
                  f"Train_Acc={train_acc:.3f}  Test_Acc={test_acc:.3f}")
        
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_probs = test_probs
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final metrics
    model.eval()
    with torch.no_grad():
        test_logits = model(genomic_test_t, imaging_test_t)
        test_pred = test_logits.argmax(dim=1).cpu().numpy()
        test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
    
    metrics = {
        'name': model_name,
        'acc': accuracy_score(y_test, test_pred),
        'balanced_acc': balanced_accuracy_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, test_probs),
        'f1': f1_score(y_test, test_pred)
    }
    
    # Extract embeddings from full dataset for hybrid creation
    genomic_emb, imaging_emb = extract_sota_embeddings(
        model, genomic_data, imaging_data, scaler_g, scaler_i
    )
    
    return metrics, model, genomic_emb, imaging_emb


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = "/Volumes/seagate"
DATA_CONFIG = {
    'rna_dir': f"{BASE_DIR}/tcga_lung",
    'mutations_dir': f"{BASE_DIR}/tcga_lung_raw",
    'wsi_dir': f"{BASE_DIR}/tcga_lung_wsi_features/small_cnn_320/feats_pt",
}

LUNG_DRIVER_GENES = [
    'TP53', 'KRAS', 'EGFR', 'STK11', 'KEAP1', 'NF1', 'RBM10', 
    'BRAF', 'PIK3CA', 'MET', 'ALK', 'ROS1', 'RET', 'ERBB2',
    'CDKN2A', 'SMARCA4', 'ARID1A', 'PTEN', 'ATM', 'SETD2'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mutations(sample_ids):
    """Parse MAF files using TCGA barcode matching"""
    maf_dir = Path(DATA_CONFIG['mutations_dir'])
    maf_files = list(maf_dir.glob("*.maf*"))
    maf_files = [f for f in maf_files if not f.name.startswith('._')]
    
    print(f"  Found {len(maf_files)} MAF files")
    
    # Load UUID -> barcode mapping
    uuid_to_barcode = np.load(f"{DATA_CONFIG['rna_dir']}/uuid_to_barcode.npy", allow_pickle=True).item()
    
    # Extract patient barcodes (first 12 chars: TCGA-XX-XXXX)
    sample_uuids = [sid.split('.')[0] for sid in sample_ids]
    sample_barcodes = {}
    for uuid in sample_uuids:
        if uuid in uuid_to_barcode:
            full_barcode = uuid_to_barcode[uuid]
            patient_barcode = full_barcode[:12]  # TCGA-XX-XXXX
            sample_barcodes[patient_barcode] = uuid
    
    print(f"  Mapped {len(sample_barcodes)} samples to patient barcodes")
    
    # Initialize mutation matrix
    mutation_dict = {uuid: {gene: 0 for gene in LUNG_DRIVER_GENES} 
                     for uuid in sample_uuids}
    
    matched_files = 0
    for maf_file in maf_files:
        try:
            # Read MAF file
            if maf_file.suffix == '.gz':
                with gzip.open(maf_file, 'rt') as f:
                    for line in f:
                        if not line.startswith('#'):
                            break
                    df = pd.read_csv(f, sep='\t', low_memory=False)
            else:
                with open(maf_file, 'r') as f:
                    for line in f:
                        if not line.startswith('#'):
                            break
                    df = pd.read_csv(f, sep='\t', low_memory=False)
            
            # Column 15 has tumor barcode
            if len(df.columns) > 15:
                tumor_barcode_col = df.columns[15]
                tumor_barcodes = df[tumor_barcode_col].unique()
                
                for tumor_barcode in tumor_barcodes:
                    patient_barcode = str(tumor_barcode)[:12]
                    
                    if patient_barcode in sample_barcodes:
                        matched_files += 1
                        uuid = sample_barcodes[patient_barcode]
                        
                        # Get mutations for this sample
                        sample_df = df[df[tumor_barcode_col] == tumor_barcode]
                        
                        # Column 0 has Hugo_Symbol (gene name)
                        gene_col = df.columns[0]
                        for gene in sample_df[gene_col].unique():
                            if gene in LUNG_DRIVER_GENES:
                                mutation_dict[uuid][gene] = 1
                        break  # Found match, move to next file
        except Exception as e:
            continue
    
    print(f"  Matched {matched_files} MAF files to samples")
    
    mutation_matrix = np.array([
        [mutation_dict[uuid][gene] for gene in LUNG_DRIVER_GENES]
        for uuid in sample_uuids
    ], dtype=np.float32)
    
    samples_with_mutations = (mutation_matrix.sum(axis=1) > 0).sum()
    print(f"  Samples with mutations: {samples_with_mutations}/{len(sample_uuids)}")
    
    return mutation_matrix


def load_wsi_patches(sample_ids):
    """Load pre-extracted WSI features (320-dim)"""
    
    wsi_dir = Path(DATA_CONFIG['wsi_dir'])
    
    if not wsi_dir.exists():
        raise FileNotFoundError(f"WSI feature directory not found: {wsi_dir}")
    
    print(f"  Loading 320-dim features from {wsi_dir}")
    
    sample_uuids = [sid.split('.')[0] for sid in sample_ids]
    wsi_features = []
    matched = 0
    
    for uuid in sample_uuids:
        pt_file = wsi_dir / f"{uuid}.pt"
        if pt_file.exists():
            feat = torch.load(pt_file, map_location='cpu').numpy()
            # Mean pooling if multiple patches
            if len(feat.shape) == 2:
                feat = feat.mean(axis=0)
            wsi_features.append(feat)
            matched += 1
        else:
            wsi_features.append(np.zeros(320, dtype=np.float32))
    
    print(f"  Matched {matched}/{len(sample_uuids)} samples to WSI features")
    
    return np.array(wsi_features)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_tybalt_vae(X_rna):
    """Load pretrained Tybalt VAE encoder - requires top 5000 genes"""
    import tensorflow as tf
    from tensorflow import keras
    
    model_path = 'tybalt_encoder.hdf5'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tybalt encoder not found: {model_path}")
    
    print("    Loading Tybalt encoder...")
    
    # Select top 5000 most variable genes (Tybalt was trained on this)
    variances = np.var(X_rna, axis=0)
    top_5000_idx = np.argsort(variances)[-5000:]
    X_rna_5000 = X_rna[:, top_5000_idx]
    
    encoder = keras.models.load_model(model_path, compile=False)
    X_vae = encoder.predict(X_rna_5000, verbose=0)
    
    return X_vae

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_representation(X, y, name):
    """Quick Random Forest evaluation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    return {
        'name': name,
        'n_features': X.shape[1],
        'acc': accuracy_score(y_test, y_pred),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║   TCGA Lung: KIRBy Hybrid Testing                                 ║")
    print("║   Can hybrids of SOTA embeddings beat SOTA alone?                 ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # TIMING - Initialize
    timings = {
        'dataset': 'tcga_lung',
        'start_time': datetime.now().isoformat(),
        'data_loading_time': 0,
        'feature_extraction': {},
        'classical_baselines': {},
        'classical_hybrids': {},
        'sota_training': {},
        'sota_embeddings': {},
        'sota_hybrids': {},
        'total_time': 0
    }
    
    overall_start = time.time()
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_start = time.time()
    
    # RNA-seq
    print("\n[RNA-seq]")
    rna_seq = np.load(f"{DATA_CONFIG['rna_dir']}/rna_seq.npy")
    gene_names = np.load(f"{DATA_CONFIG['rna_dir']}/gene_names.npy", allow_pickle=True)
    sample_ids = np.load(f"{DATA_CONFIG['rna_dir']}/sample_ids.npy", allow_pickle=True)
    labels = np.load(f"{DATA_CONFIG['rna_dir']}/labels.npy")
    rna_seq = np.log2(rna_seq + 1)
    print(f"  Shape: {rna_seq.shape}")
    print(f"  LUAD: {np.sum(labels==0)}, LUSC: {np.sum(labels==1)}")
    
    # Mutations
    print("\n[Mutations]")
    mutations = load_mutations(sample_ids)
    print(f"  Shape: {mutations.shape}")
    print(f"  Avg mutations/sample: {mutations.sum(axis=1).mean():.1f}")
    
    # WSI 320-dim
    print("\n[WSI Features - 320-dim]")
    wsi_320 = load_wsi_patches(sample_ids)
    print(f"  Shape: {wsi_320.shape}")
    
    y = labels
    
    data_time = time.time() - data_start
    timings['data_loading_time'] = data_time
    print(f"\nData loading time: {data_time:.2f}s")
    
    # Extract all features
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    representations = {}
    
    # ========== GENOMIC ==========
    print("\n[GENOMIC]")
    
    extraction_start = time.time()
    
    # Top variable genes
    for n in [500, 1000, 2000, 5000]:
        start = time.time()
        var = np.var(rna_seq, axis=0)
        idx = np.argsort(var)[-n:]
        representations[f'RNA_Top{n}'] = rna_seq[:, idx]
        elapsed = time.time() - start
        timings['feature_extraction'][f'RNA_Top{n}'] = elapsed
        print(f"  Top-{n}: {rna_seq[:, idx].shape} ({elapsed:.2f}s)")
    
    # PCA
    for n in [50, 100, 200]:
        start = time.time()
        pca = PCA(n_components=n, random_state=42)
        representations[f'RNA_PCA{n}'] = pca.fit_transform(rna_seq)
        elapsed = time.time() - start
        timings['feature_extraction'][f'RNA_PCA{n}'] = elapsed
        print(f"  PCA-{n}: ({pca.explained_variance_ratio_.sum():.2%} var, {elapsed:.2f}s)")
    
    # Tybalt VAE
    start = time.time()
    X_vae = extract_tybalt_vae(rna_seq)
    representations['RNA_TybaltVAE'] = X_vae
    elapsed = time.time() - start
    timings['feature_extraction']['RNA_TybaltVAE'] = elapsed
    print(f"  Tybalt VAE: {X_vae.shape} ({elapsed:.2f}s)")
    
    # Mutations
    representations['Mutations'] = mutations
    print(f"  Mutations: {mutations.shape}")
    
    # ========== IMAGING ==========
    print("\n[IMAGING]")
    
    # 320-dim features
    representations['WSI_320'] = wsi_320
    print(f"  WSI 320-dim: {wsi_320.shape}")
    
    extraction_time = time.time() - extraction_start
    print(f"\nTotal feature extraction time: {extraction_time:.2f}s")
    
    # Test individual representations
    print("\n" + "="*70)
    print("STEP 1: BASELINE - CLASSICAL REPRESENTATIONS")
    print("="*70)
    
    baseline_start = time.time()
    
    results = []
    for name, X in representations.items():
        start = time.time()
        res = evaluate_representation(X, y, name)
        elapsed = time.time() - start
        timings['classical_baselines'][name] = {
            'time': elapsed,
            'n_features': res['n_features'],
            'balanced_acc': res['balanced_acc'],
            'roc_auc': res['roc_auc']
        }
        results.append(res)
        print(f"{name:<20} {res['n_features']:>6} feats  "
              f"Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({elapsed:.2f}s)")
    
    baseline_time = time.time() - baseline_start
    print(f"\nBaseline evaluation time: {baseline_time:.2f}s")
    
    # Test classical KIRBy hybrids
    print("\n" + "="*70)
    print("STEP 2: CLASSICAL KIRBY HYBRIDS")
    print("="*70)
    
    hybrids_start = time.time()
    
    # Hybrid 1: Genomic only
    print("\n[1] Genomic Hybrid (Tybalt + Top2000 + Mutations)")
    start = time.time()
    genomic_dict = {
        'Tybalt': representations['RNA_TybaltVAE'],
        'Top2000': representations['RNA_Top2000'],
        'Mutations': representations['Mutations']
    }
    X_hybrid_genomic, _ = create_hybrid(genomic_dict, y, n_per_rep=50)
    hybrid_time = time.time() - start
    res = evaluate_representation(X_hybrid_genomic, y, 'Hybrid_Genomic_Classical')
    total_time = time.time() - start
    timings['classical_hybrids']['genomic'] = {
        'hybrid_time': hybrid_time,
        'total_time': total_time,
        'n_features': X_hybrid_genomic.shape[1],
        'balanced_acc': res['balanced_acc'],
        'roc_auc': res['roc_auc']
    }
    results.append(res)
    print(f"  → {X_hybrid_genomic.shape}  Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({total_time:.2f}s)")
    
    # Hybrid 2: Multimodal classical
    print("\n[2] Multimodal Hybrid (Tybalt + WSI_320)")
    start = time.time()
    multimodal_classical = {
        'Tybalt': representations['RNA_TybaltVAE'],
        'WSI_320': representations['WSI_320']
    }
    X_hybrid_classical_multi, _ = create_hybrid(multimodal_classical, y, n_per_rep=50)
    hybrid_time = time.time() - start
    res = evaluate_representation(X_hybrid_classical_multi, y, 'Hybrid_Classical_Multi')
    total_time = time.time() - start
    timings['classical_hybrids']['multimodal'] = {
        'hybrid_time': hybrid_time,
        'total_time': total_time,
        'n_features': X_hybrid_classical_multi.shape[1],
        'balanced_acc': res['balanced_acc'],
        'roc_auc': res['roc_auc']
    }
    results.append(res)
    print(f"  → {X_hybrid_classical_multi.shape}  Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({total_time:.2f}s)")
    
    # Hybrid 3: Full classical
    print("\n[3] Full Classical Hybrid (All features)")
    start = time.time()
    full_classical = {
        'Tybalt': representations['RNA_TybaltVAE'],
        'Top2000': representations['RNA_Top2000'],
        'Mutations': representations['Mutations'],
        'WSI_320': representations['WSI_320']
    }
    X_hybrid_full_classical, _ = create_hybrid(full_classical, y, n_per_rep=50)
    hybrid_time = time.time() - start
    res = evaluate_representation(X_hybrid_full_classical, y, 'Hybrid_Full_Classical')
    total_time = time.time() - start
    timings['classical_hybrids']['full'] = {
        'hybrid_time': hybrid_time,
        'total_time': total_time,
        'n_features': X_hybrid_full_classical.shape[1],
        'balanced_acc': res['balanced_acc'],
        'roc_auc': res['roc_auc']
    }
    results.append(res)
    print(f"  → {X_hybrid_full_classical.shape}  Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({total_time:.2f}s)")
    
    hybrids_time = time.time() - hybrids_start
    print(f"\nClassical hybrids time: {hybrids_time:.2f}s")
    
    # SOTA MULTIMODAL TRAINING
    print("\n" + "="*70)
    print("STEP 3: TRAIN SOTA MODELS & EXTRACT EMBEDDINGS")
    print("="*70)
    
    sota_training_start = time.time()
    
    best_genomic = representations['RNA_TybaltVAE']
    wsi_features = wsi_320
    
    # Train best SOTA configs
    configs = [
        ('snn', 'bilinear'),
        ('snn', 'cross_attention'),
        ('variational', 'bilinear'),
        ('variational', 'cross_attention'),
    ]
    
    sota_models = {}
    sota_metrics = []
    
    for genomic_enc, fusion in configs:
        print(f"\n{'='*70}")
        print(f"Training: {genomic_enc.upper()} + {fusion.upper()}")
        print('='*70)
        
        start = time.time()
        metrics, model, genomic_emb, imaging_emb = train_sota_model(
            best_genomic, wsi_features, y,
            genomic_encoder=genomic_enc,
            fusion_type=fusion,
            epochs=50
        )
        training_time = time.time() - start
        
        model_name = f"{genomic_enc}+{fusion}"
        sota_models[model_name] = {
            'model': model,
            'genomic_emb': genomic_emb,
            'imaging_emb': imaging_emb,
            'metrics': metrics
        }
        sota_metrics.append(metrics)
        
        timings['sota_training'][model_name] = {
            'training_time': training_time,
            'balanced_acc': metrics['balanced_acc'],
            'roc_auc': metrics['roc_auc']
        }
        
        print(f"\nSOTA Model Performance:")
        print(f"  Bal_Acc: {metrics['balanced_acc']:.4f}  AUC: {metrics['roc_auc']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
    
    sota_training_time = time.time() - sota_training_start
    print(f"\nTotal SOTA training time: {sota_training_time:.2f}s")
    
    # Test SOTA embeddings alone
    print("\n" + "="*70)
    print("STEP 4: TEST SOTA EMBEDDINGS ALONE")
    print("="*70)
    
    embedding_start = time.time()
    
    embedding_results = []
    for model_name, model_data in sota_models.items():
        # Concatenate genomic + imaging embeddings
        start = time.time()
        combined_emb = np.concatenate([
            model_data['genomic_emb'],
            model_data['imaging_emb']
        ], axis=1)
        
        res = evaluate_representation(combined_emb, y, f"{model_name}_embedding")
        elapsed = time.time() - start
        embedding_results.append(res)
        
        timings['sota_embeddings'][model_name] = {
            'eval_time': elapsed,
            'n_features': combined_emb.shape[1],
            'balanced_acc': res['balanced_acc'],
            'roc_auc': res['roc_auc']
        }
        
        print(f"{model_name}_embedding: "
              f"Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({elapsed:.2f}s)")
    
    embedding_time = time.time() - embedding_start
    print(f"\nEmbedding evaluation time: {embedding_time:.2f}s")
    
    # Create SOTA+Classical hybrids
    print("\n" + "="*70)
    print("STEP 5: KIRBY HYBRIDS (SOTA embeddings + Classical)")
    print("="*70)
    
    sota_hybrids_start = time.time()
    
    hybrid_sota_results = []
    
    for model_name, model_data in sota_models.items():
        print(f"\n[Hybrid: {model_name} embeddings + classical]")
        
        start = time.time()
        
        # Create hybrid with SOTA embeddings + classical features
        hybrid_dict = {
            f'{model_name}_genomic': model_data['genomic_emb'],
            f'{model_name}_imaging': model_data['imaging_emb'],
            'Tybalt': representations['RNA_TybaltVAE'],
            'Top2000': representations['RNA_Top2000'],
            'Mutations': representations['Mutations'],
            'WSI_320': representations['WSI_320']
        }
        
        X_hybrid_sota, _ = create_hybrid(hybrid_dict, y, n_per_rep=50)
        hybrid_time = time.time() - start
        res = evaluate_representation(X_hybrid_sota, y, f'Hybrid_SOTA_{model_name}')
        total_time = time.time() - start
        hybrid_sota_results.append(res)
        
        timings['sota_hybrids'][model_name] = {
            'hybrid_time': hybrid_time,
            'total_time': total_time,
            'n_features': X_hybrid_sota.shape[1],
            'balanced_acc': res['balanced_acc'],
            'roc_auc': res['roc_auc']
        }
        
        print(f"  Shape: {X_hybrid_sota.shape}")
        print(f"  Bal_Acc={res['balanced_acc']:.3f}  AUC={res['roc_auc']:.3f} ({total_time:.2f}s)")
        
        # Compare to embedding alone
        emb_res = [r for r in embedding_results if r['name'] == f"{model_name}_embedding"][0]
        if res['balanced_acc'] > emb_res['balanced_acc']:
            improvement = (res['balanced_acc'] - emb_res['balanced_acc']) * 100
            print(f"  ✓ HYBRID WINS! +{improvement:.1f}% over embedding alone")
        else:
            decline = (emb_res['balanced_acc'] - res['balanced_acc']) * 100
            print(f"  ✗ Embedding alone better by {decline:.1f}%")
    
    sota_hybrids_time = time.time() - sota_hybrids_start
    print(f"\nSOTA hybrids time: {sota_hybrids_time:.2f}s")
    
    # FINAL COMPARISON
    print("\n" + "="*70)
    print("FINAL COMPREHENSIVE COMPARISON")
    print("="*70)
    
    all_results = results + embedding_results + hybrid_sota_results
    all_sorted = sorted(all_results, key=lambda x: x['roc_auc'], reverse=True)
    
    print(f"\n{'Method':<50} {'Type':<20} {'Bal_Acc':>10} {'AUC':>10}")
    print("-" * 90)
    
    for r in all_sorted[:20]:
        if 'Hybrid_SOTA' in r['name']:
            typ = 'SOTA+Classical Hybrid'
        elif 'Hybrid' in r['name']:
            typ = 'Classical Hybrid'
        elif 'embedding' in r['name']:
            typ = 'SOTA Embedding'
        else:
            typ = 'Classical Baseline'
        
        print(f"{r['name']:<50} {typ:<20} {r['balanced_acc']:>10.4f} {r['roc_auc']:>10.4f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    # Find best of each category
    best_classical = sorted([r for r in results], 
                           key=lambda x: x['roc_auc'], reverse=True)[0]
    best_classical_hybrid = sorted([r for r in results if 'Hybrid' in r['name']], 
                                   key=lambda x: x['roc_auc'], reverse=True)[0]
    best_embedding = sorted(embedding_results, key=lambda x: x['roc_auc'], reverse=True)[0]
    best_sota_hybrid = sorted(hybrid_sota_results, key=lambda x: x['roc_auc'], reverse=True)[0]
    
    print(f"\nBest Classical:           {best_classical['name']:<40} AUC={best_classical['roc_auc']:.4f}")
    print(f"Best Classical Hybrid:    {best_classical_hybrid['name']:<40} AUC={best_classical_hybrid['roc_auc']:.4f}")
    print(f"Best SOTA Embedding:      {best_embedding['name']:<40} AUC={best_embedding['roc_auc']:.4f}")
    print(f"Best SOTA+Classical Hybrid: {best_sota_hybrid['name']:<40} AUC={best_sota_hybrid['roc_auc']:.4f}")
    
    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Do SOTA+Classical hybrids beat SOTA embeddings?")
    print("="*70)
    
    if best_sota_hybrid['roc_auc'] > best_embedding['roc_auc']:
        improvement = (best_sota_hybrid['roc_auc'] - best_embedding['roc_auc']) * 100
        print(f"\n✓ YES! KIRBY HYBRID WINS!")
        print(f"  → {best_sota_hybrid['name']}")
        print(f"  → {improvement:.1f}% AUC improvement over best SOTA embedding alone")
        print(f"  → This validates the KIRBy approach: combining SOTA with classical")
    else:
        decline = (best_embedding['roc_auc'] - best_sota_hybrid['roc_auc']) * 100
        print(f"\n✗ NO. SOTA embedding alone is better")
        print(f"  → {best_embedding['name']} beats hybrids by {decline:.1f}%")
        print(f"  → Need to investigate why hybrids underperform")
    
    # TIMING SUMMARY
    timings['total_time'] = time.time() - overall_start
    timings['end_time'] = datetime.now().isoformat()
    
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Data loading:        {timings['data_loading_time']:.2f}s")
    print(f"Feature extraction:  {extraction_time:.2f}s")
    print(f"Classical baselines: {baseline_time:.2f}s")
    print(f"Classical hybrids:   {hybrids_time:.2f}s")
    print(f"SOTA training:       {sota_training_time:.2f}s")
    print(f"SOTA embeddings:     {embedding_time:.2f}s")
    print(f"SOTA hybrids:        {sota_hybrids_time:.2f}s")
    print(f"TOTAL:               {timings['total_time']:.2f}s ({timings['total_time']/60:.2f}min)")
    
    # Save timing results
    output_file = 'tcga_lung_timing_results.json'
    with open(output_file, 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"\nTiming results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("Done!")


if __name__ == '__main__':
    main()