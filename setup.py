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
        # Chemistry / QSAR stack (install when needed)
        "chem": [
            # NOTE: RDKit is intentionally NOT pinned here because pip installs can be fragile.
            # Prefer conda/mamba for RDKit, or install `rdkit-pypi` manually where supported.
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "selfies>=2.1.1",
            "huggingface_hub>=0.16.0",
            "transformers>=4.41.0",
            "gensim>=4.0.0",
            "mol2vec>=0.1",
            "deepchem>=2.7.0",
            "grakel>=0.1.8",
            "gauche>=0.1.0",
            "gpytorch>=1.9.0",
        ],
        # MHG-GNN feature extraction support
        "mhg": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "selfies>=2.1.1",
            "huggingface_hub>=0.16.0",
        ],
        # Antibody representation support
        "antibody": [
            "torch>=2.0.0",
            "transformers>=4.41.0",
            "biopython>=1.79",
            "requests>=2.26.0",
            "ablang2",  # AbLang-2 antibody language model
            # ANARCI is conda-only: conda install -c bioconda anarci
            # Required for CDR numbering
        ],
        # Medical / imaging / genomics utilities
        "bio": [
            "h5py>=3.7.0",
            "scikit-image>=0.19.0",
            "tensorflow>=2.10.0",
            "keras>=2.10.0",
            "timm>=0.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        # Convenience meta-extra: everything (except RDKit and ANARCI which need conda)
        "all": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "selfies>=2.1.1",
            "huggingface_hub>=0.16.0",
            "transformers>=4.41.0",
            "gensim>=4.0.0",
            "mol2vec>=0.1",
            "deepchem>=2.7.0",
            "grakel>=0.1.8",
            "gauche>=0.1.0",
            "gpytorch>=1.9.0",
            "h5py>=3.7.0",
            "scikit-image>=0.19.0",
            "tensorflow>=2.10.0",
            "keras>=2.10.0",
            "timm>=0.9.0",
            "biopython>=1.79",
            "requests>=2.26.0",
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

# TODO: add `pip install shap`