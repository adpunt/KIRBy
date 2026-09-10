"""
QM9 Dataset Loader

QM9 contains 134k molecules with computed quantum mechanical properties.
This module loads HOMO-LUMO gap prediction task with scaffold splitting support.

References:
- Ramakrishnan et al. (2014). Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data.
- PyTorch Geometric QM9 dataset
"""

import os
import sys
import numpy as np
import torch
import os.path as osp
from pathlib import Path


def _default_qm9_dir() -> str:
    """Where QM9 lives: the external drive if mounted, else the cache root.

    The 235MB raw structures and the 340MB processed file are staged at
    /Volumes/seagate/QM9. Every other molecular loader already prefers the
    drive when it is mounted (see abbind.py, chembl_qsar.py, ginkgo_dev.py);
    QM9 was the one that did not, so it tried to re-download instead.
    """
    seagate = Path('/Volumes/seagate/QM9')
    if seagate.exists():
        return str(seagate)
    root = Path(os.environ.get(
        'KIRBY_CACHE_DIR', str(Path.home() / '.cache' / 'kirby')))
    return str(root / 'QM9')


def _disable_broken_pyg_extensions():
    """Hide torch_cluster/torch_scatter/torch_sparse when their .so won't load.

    Those four packages ship compiled extensions pinned to one torch build.
    On this laptop they raise OSError (symbol not found) against the installed
    torch, and torch_geometric re-raises it, so QM9 could not be opened at all.
    torch_geometric 2.4.0 runs fine without them for everything QM9 needs.

    Only fires when the import actually fails, so a machine where they work
    (the server, where SchNet needs torch_cluster.radius_graph) is untouched.
    """
    import importlib
    import importlib.abc

    broken = set()
    for name in ('torch_cluster', 'torch_scatter', 'torch_sparse',
                 'torch_spline_conv'):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except OSError:
            broken.add(name)
        except ImportError:
            pass  # not installed at all; torch_geometric handles that already
    if not broken:
        return

    class _BlockBroken(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in broken:
                raise ImportError(
                    f"{fullname} has a compiled extension incompatible with "
                    f"the installed torch; hidden so torch_geometric falls "
                    f"back to its pure-Python path")
            return None

    sys.meta_path.insert(0, _BlockBroken())
    print(f"  QM9: hid {len(broken)} torch_geometric extension package(s) "
          f"with incompatible compiled libraries: {sorted(broken)}")


def _patch_torch_load_for_pyg():
    """Patch torch.load for PyTorch 2.6+ compatibility with torch_geometric."""
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    if torch_version >= (2, 6):
        try:
            from torch_geometric.data.data import Data
            from torch_geometric.data.storage import GlobalStorage
            torch.serialization.add_safe_globals([Data, GlobalStorage])
        except (ImportError, AttributeError):
            pass


def _load_smiles_from_sdf(data_dir):
    """Extract SMILES from QM9 SDF file using RDKit."""
    from rdkit import Chem
    sdf_path = osp.join(data_dir, 'raw', 'gdb9.sdf')
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    smiles = []
    for mol in suppl:
        if mol is not None:
            smiles.append(Chem.MolToSmiles(mol))
        else:
            smiles.append(None)
    return smiles


def load_qm9(n_samples=5000, property_idx=4, data_dir=None, random_seed=42):
    """
    Load QM9 dataset and extract SMILES + labels.

    Matches qsar_qm_models pipeline:
      1. Load full QM9 via PyG
      2. Filter to RDKit-valid molecules (valid_qm9_indices.pth)
      3. Use data.smiles from PyG (not SDF extraction)
      4. Subsample if n_samples < total

    Args:
        n_samples: Number of molecules to sample (None = all valid)
        property_idx: Which property to predict (0-11)
        data_dir: Directory to store/load QM9 data
        random_seed: Random seed for reproducibility

    Returns:
        dict: {'smiles': [...], 'labels': [...], 'property_name': str}
    """
    _disable_broken_pyg_extensions()
    try:
        from torch_geometric.datasets import QM9
    except ImportError as e:
        raise ImportError(f"QM9 loader requires torch-geometric: {e}")

    import os.path as osp

    property_names = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
        'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]

    # Fix PyTorch 2.6+ weights_only compatibility
    _patch_torch_load_for_pyg()

    if data_dir is None:
        data_dir = _default_qm9_dir()
    print(f"Loading QM9 (property: {property_names[property_idx]})...")

    try:
        qm9 = QM9(root=data_dir)
    except (ValueError, RuntimeError):
        # Stale processed file from an older PyG version — delete and retry
        import glob
        processed_dir = osp.join(data_dir, 'processed')
        stale = glob.glob(osp.join(processed_dir, '*.pt'))
        if stale:
            print(f"  Removing stale PyG cache ({len(stale)} files) and reprocessing...")
            for f in stale:
                os.remove(f)
            qm9 = QM9(root=data_dir)
        else:
            raise

    # Filter to RDKit-valid molecules (matches qsar_qm_models/process_and_train.py)
    valid_indices_path = osp.join(osp.dirname(data_dir), 'valid_qm9_indices.pth')
    if not osp.exists(valid_indices_path):
        # Fallback: the file in KIRBy's own data/ directory, addressed from
        # this file rather than from the process's working directory. As a
        # relative path it was found only by a run started in the repo root:
        # a run started anywhere else printed the WARNING below, kept all
        # 130,831 molecules instead of the 129,428 RDKit-valid ones, and drew
        # a DIFFERENT 10,000-molecule sample under the same dataset name. The
        # representation cache stores no molecule identity, so the two samples
        # are told apart only by their row count.
        valid_indices_path = str(
            Path(__file__).resolve().parents[3] / 'data' / 'valid_qm9_indices.pth')
    if osp.exists(valid_indices_path):
        valid_indices = torch.load(valid_indices_path, weights_only=True)
        qm9 = qm9.index_select(valid_indices)
        print(f"  Filtered to {len(qm9)} RDKit-valid molecules")
    else:
        print("  WARNING: valid_qm9_indices.pth not found, using all molecules")

    # Use PyG's data.smiles attribute (matches qsar repo)
    # Fall back to SDF extraction only if .smiles not available
    has_smiles_attr = hasattr(qm9[0], 'smiles') and qm9[0].smiles is not None
    if has_smiles_attr:
        print("  Using PyG SMILES attribute")
    else:
        print("  PyG SMILES not available, extracting from SDF...")

    total = len(qm9)
    if n_samples is not None and n_samples < total:
        import random
        random.seed(random_seed)
        indices = sorted(random.sample(range(total), n_samples))
    else:
        indices = list(range(total))
        n_samples = total

    smiles_list = []
    targets = []

    # SDF fallback SMILES (only loaded if needed)
    sdf_smiles = None
    if not has_smiles_attr:
        # The SDF is read in raw file order, but `qm9` has already been cut
        # down to the RDKit-valid molecules, so position i in one is not
        # molecule i in the other. Indexing the SDF by the filtered position
        # pairs every structure with a different molecule's label. Refuse
        # rather than return silently mismatched structures and labels.
        raise RuntimeError(
            "QM9: the processed file carries no SMILES, and the SDF fallback "
            "indexes raw file positions against RDKit-filtered positions "
            f"({len(qm9)} of 133885 kept), which mismatches every structure "
            "with another molecule's label. Rebuild the processed file with a "
            "torch_geometric version that stores `data.smiles`.")
        sdf_smiles = _load_smiles_from_sdf(data_dir)

    print(f"  Extracting {len(indices)} molecules...")
    for index in indices:
        data = qm9[index]

        if has_smiles_attr:
            smi = data.smiles
        else:
            smi = sdf_smiles[index] if index < len(sdf_smiles) else None

        if smi is None:
            continue

        smiles_list.append(smi)
        targets.append(data.y[0, property_idx].item())

    targets = np.array(targets)

    # Return raw targets — normalization happens per-fold in the evaluation pipeline
    print(f"  Loaded {len(smiles_list)} molecules")
    print(f"  Range: [{targets.min():.3f}, {targets.max():.3f}]")

    return {
        'smiles': smiles_list,
        'labels': targets,
        'property_name': property_names[property_idx],
        'property_idx': property_idx
    }


def load_qm9_with_3d(n_samples=None, property_idx=4, data_dir=None,
                      random_seed=42):
    """Load QM9 with 3D coordinates for end-to-end 3D models (SchNet, DimeNet++).

    Unlike load_qm9(), this returns PyG Data objects with atomic positions,
    atomic numbers, and edge information for 3D GNN training.

    Args:
        n_samples: Number of molecules (None = full dataset, ~130K)
        property_idx: Which property to predict (0-11)
        data_dir: Directory to store/load QM9 data
        random_seed: Random seed for subsampling

    Returns:
        dict with keys:
            'data_list': list of PyG Data objects with .pos, .z, .y
            'smiles': list of SMILES strings
            'labels': np.ndarray of target values
            'property_name': str
    """
    _disable_broken_pyg_extensions()
    try:
        from torch_geometric.datasets import QM9
    except ImportError as e:
        raise ImportError(f"QM9 loader requires torch-geometric: {e}")

    property_names = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
        'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]

    _patch_torch_load_for_pyg()

    if data_dir is None:
        data_dir = _default_qm9_dir()
    print(f"Loading QM9 with 3D coords (property: {property_names[property_idx]})...")
    try:
        qm9 = QM9(root=data_dir)
    except (ValueError, RuntimeError):
        import glob
        processed_dir = osp.join(data_dir, 'processed')
        stale = glob.glob(osp.join(processed_dir, '*.pt'))
        if stale:
            for f in stale:
                os.remove(f)
            qm9 = QM9(root=data_dir)
        else:
            raise

    # Filter to RDKit-valid molecules (matches qsar_qm_models)
    valid_indices_path = osp.join(osp.dirname(data_dir), 'valid_qm9_indices.pth')
    if not osp.exists(valid_indices_path):
        # Same working-directory trap as in load_qm9 above.
        valid_indices_path = str(
            Path(__file__).resolve().parents[3] / 'data' / 'valid_qm9_indices.pth')
    if osp.exists(valid_indices_path):
        valid_indices = torch.load(valid_indices_path, weights_only=True)
        qm9 = qm9.index_select(valid_indices)
        print(f"  Filtered to {len(qm9)} RDKit-valid molecules")
    else:
        print("  WARNING: valid_qm9_indices.pth not found, using all molecules")

    # Use PyG's data.smiles if available
    has_smiles_attr = hasattr(qm9[0], 'smiles') and qm9[0].smiles is not None
    sdf_smiles = None
    if not has_smiles_attr:
        print("  Extracting SMILES from SDF...")
        sdf_smiles = _load_smiles_from_sdf(data_dir)

    # Subsample if requested
    total = len(qm9)
    if n_samples is not None and n_samples < total:
        import random
        random.seed(random_seed)
        indices = sorted(random.sample(range(total), n_samples))
    else:
        indices = list(range(total))
        n_samples = total

    print(f"  Extracting {len(indices)} molecules with 3D coordinates...")
    data_list = []
    smiles_list = []
    targets = []

    for i, index in enumerate(indices):
        data = qm9[index]

        if has_smiles_attr:
            smi = data.smiles
        else:
            smi = sdf_smiles[index] if sdf_smiles and index < len(sdf_smiles) else None

        if smi is None:
            continue

        # Verify 3D coordinates exist
        if not hasattr(data, 'pos') or data.pos is None:
            continue

        smiles_list.append(smi)
        targets.append(data.y[0, property_idx].item())
        data_list.append(data)

    targets = np.array(targets)
    print(f"  Loaded {len(data_list)} molecules with 3D coords")
    print(f"  Range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  Avg atoms/molecule: {np.mean([d.pos.shape[0] for d in data_list]):.1f}")

    return {
        'data_list': data_list,
        'smiles': smiles_list,
        'labels': targets,
        'property_name': property_names[property_idx],
        'property_idx': property_idx,
    }

def get_qm9_splits(data, splitter='scaffold', train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """
    Split QM9 data into train/val/test sets.
    
    Args:
        data: Dictionary from load_qm9()
        splitter: How to split the data (default: 'scaffold')
            'scaffold': Scaffold split (more realistic, harder, tests generalization)
            'random': Random split
            'stratified': Stratified split
        train_ratio: Fraction for training (default: 0.8) - used for random/stratified
        val_ratio: Fraction for validation (default: 0.1) - used for random/stratified
        random_seed: Random seed
    
    Returns:
        dict: {
            'train': {'smiles': [...], 'labels': [...]},
            'val': {'smiles': [...], 'labels': [...]},
            'test': {'smiles': [...], 'labels': [...]}
        }
    """
    smiles = data['smiles']
    labels = data['labels']
    n_samples = len(labels)
    
    if splitter == 'scaffold':
        # Use DeepChem's scaffold splitter
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError("Scaffold splitting requires DeepChem: pip install deepchem")
        
        print(f"Splitting {n_samples} molecules using scaffold split...")
        
        # Create DeepChem dataset
        dataset = dc.data.NumpyDataset(
            X=np.zeros((n_samples, 1)),  # Dummy features
            y=labels.reshape(-1, 1),
            ids=smiles
        )
        
        # Scaffold split
        splitter_obj = dc.splits.ScaffoldSplitter()
        train_dataset, val_dataset, test_dataset = splitter_obj.train_valid_test_split(
            dataset,
            frac_train=train_ratio,
            frac_valid=val_ratio,
            frac_test=1.0 - train_ratio - val_ratio,
            seed=random_seed
        )
        
        # Extract SMILES and labels
        train_idx = [smiles.index(s) for s in train_dataset.ids]
        val_idx = [smiles.index(s) for s in val_dataset.ids]
        test_idx = [smiles.index(s) for s in test_dataset.ids]
        
    elif splitter == 'random':
        print(f"Splitting {n_samples} molecules using random split...")
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
    elif splitter == 'stratified':
        print(f"Splitting {n_samples} molecules using stratified split...")
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(n_samples),
            train_size=train_ratio,
            random_state=random_seed,
            stratify=np.digitize(labels, bins=np.percentile(labels, [25, 50, 75]))
        )
        
        # Second split: val vs test
        val_size = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_seed,
            stratify=np.digitize(labels[temp_idx], bins=np.percentile(labels[temp_idx], [25, 50, 75]))
        )
    else:
        raise ValueError(f"Unknown splitter: {splitter}. Use 'scaffold', 'random', or 'stratified'")
    
    print(f"  Train: {len(train_idx)} molecules")
    print(f"  Val:   {len(val_idx)} molecules")
    print(f"  Test:  {len(test_idx)} molecules")
    
    return {
        'train': {
            'smiles': [smiles[i] for i in train_idx],
            'labels': labels[train_idx]
        },
        'val': {
            'smiles': [smiles[i] for i in val_idx],
            'labels': labels[val_idx]
        },
        'test': {
            'smiles': [smiles[i] for i in test_idx],
            'labels': labels[test_idx]
        }
    }


def load_qm9_homo_lumo(n_samples=1000, data_dir=None, random_seed=42):
    """
    Load QM9 for HOMO-LUMO gap prediction (most common task).
    
    This is a convenience wrapper around load_qm9() with property_idx=4.
    """
    return load_qm9(
        n_samples=n_samples,
        property_idx=4,  # HOMO-LUMO gap
        data_dir=data_dir,
        random_seed=random_seed
    )