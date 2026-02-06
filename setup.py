"""
KIRBy: Knowledge Integration by Representation Borrowing
Automated Representation Optimization Through Complexity-Guided Feature Selection
"""
from setuptools import setup, find_packages

setup(
    name="kirby",
    version="0.2.0",
    author="Adelaide Punt",
    description="Automated representation optimization through complexity-guided feature selection",
    long_description=(
        "KIRBy optimizes molecular and genomic representations by intelligently combining "
        "multiple feature types through complexity-guided selection."
    ),
    long_description_content_type="text/plain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        # Core
        "numpy>=1.26.4",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.2.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        # =================================================================
        # MOLECULAR REPRESENTATIONS (organized by model type)
        # =================================================================

        # Fingerprints only (ECFP4, MACCS, PDV) - no extra deps, just RDKit
        # Install RDKit via: conda install -c conda-forge rdkit

        # HuggingFace transformers (ChemBERTa, MolFormer, SELFormer, ChemBERT)
        "transformers": [
            "torch>=2.0.0",
            "transformers>=4.41.0",
            "huggingface_hub>=0.16.0",
            "selfies>=2.1.1",  # For SELFormer
        ],

        # Graph neural networks (SchNet, MHG-GNN, MolCLR, GraphMVP)
        # NOTE: torch-scatter must be installed separately - see docs
        "gnn": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "ogb>=1.3.0",  # For GraphMVP OGB-style encoders
            "ase>=3.22.0",  # For SchNet/GraphMVP 3D
            # torch-scatter installed separately - see docs
        ],

        # External repos (GROVER) - need manual setup
        "grover": [
            "typed-argument-parser>=1.7.0",
            "descriptastorus>=2.6.0",
        ],

        # Chemformer - uses built-in tokenizer, no pysmilesutils needed

        # Uni-Mol (auto-downloads weights)
        "unimol": [
            "unimol-tools>=1.0.0",
        ],

        # mol2vec (needs wget of model_300dim.pkl)
        "mol2vec": [
            "gensim>=4.0.0",
            "mol2vec>=0.1",
        ],

        # =================================================================
        # USE CASE BUNDLES
        # =================================================================

        # Minimal QSAR: fingerprints + basic ML
        "qsar-minimal": [
            "xgboost>=1.5.0",
            "lightgbm>=3.0.0",
        ],

        # Full QSAR: transformers + GNNs + GP models
        "qsar": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "transformers>=4.41.0",
            "huggingface_hub>=0.16.0",
            "selfies>=2.1.1",
            "gensim>=4.0.0",
            "mol2vec>=0.1",
            "gpytorch>=1.9.0",
            "gauche>=0.1.0",
            "grakel>=0.1.8",
            "xgboost>=1.5.0",
            "lightgbm>=3.0.0",
        ],

        # Uncertainty quantification
        "uncertainty": [
            "gpytorch>=1.9.0",
            "gauche>=0.1.0",
            "quantile-forest>=1.2.0",
            "ngboost>=0.4.0",
            "torchbnn>=1.0.0",
        ],

        # Feature importance / interpretability
        "interpret": [
            "shap>=0.42.0",
            "captum>=0.6.0",
            "lime>=0.2.0",
            "boruta>=0.3.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.0.0",
        ],

        # =================================================================
        # DOMAIN-SPECIFIC
        # =================================================================

        # Antibody representations
        "antibody": [
            "torch>=2.0.0",
            "transformers>=4.41.0",
            "biopython>=1.79",
            "requests>=2.26.0",
            "ablang2",
            # ANARCI: conda install -c bioconda anarci
        ],

        # Medical imaging / genomics
        "bio": [
            "h5py>=3.7.0",
            "scikit-image>=0.19.0",
            "tensorflow>=2.10.0",
            "keras>=2.10.0",
            "timm>=0.9.0",
        ],

        # =================================================================
        # DEVELOPMENT
        # =================================================================
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)