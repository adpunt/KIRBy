#!/usr/bin/env python3
"""Quick import check for all dependencies used by hybrid experiments."""

import sys

failures = []

checks = [
    # Core
    ("numpy", "import numpy"),
    ("scipy", "import scipy"),
    ("sklearn", "import sklearn"),
    ("pandas", "import pandas"),
    # Representations
    ("rdkit", "from rdkit import Chem"),
    # Models
    ("xgboost", "import xgboost"),
    ("lightgbm", "import lightgbm"),
    # Importance methods
    ("shap", "import shap"),
    ("captum", "import captum"),
    ("lime", "import lime"),
    ("boruta", "from boruta import BorutaPy"),
    # Deep learning
    ("torch", "import torch"),
    # KIRBy modules
    ("kirby.hybrid", "from kirby.hybrid import compute_feature_importance, create_hybrid"),
    ("kirby.importance", "from kirby.importance import explain_hybrid"),
    ("kirby.representations", "from kirby.representations.molecular import create_ecfp4, create_maccs, create_pdv"),
]

for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failures.append(name)

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    print("Fix with: pip install " + " ".join(failures))
    sys.exit(1)
else:
    print("All imports OK")
