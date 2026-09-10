"""
Forward-compatibility molecular representation builders (c13 candidates).

These are the molecular models on the c13 candidate list that had no builder in
`molecular.py`. c13 adds ONE of these to a dataset's existing pool, searches a
hybrid with it and without it, and subtracts. So every function here does one
thing: turn a list of SMILES into a frozen feature block, one row per molecule,
in the order the SMILES came in. No training, no fine-tuning, no labels.

CONTRACT (all functions):
    create_<name>(smiles_list, batch_size=32, device=None) -> np.ndarray
    shape (len(smiles_list), D), dtype float32, row i is smiles_list[i].
    A molecule that fails to parse or fails to encode gets a ZERO ROW and is
    counted in a printed warning. Rows are never dropped and never reordered.
    A missing package or missing weights raises ImportError / RuntimeError /
    FileNotFoundError with what to install and from where — the domain loader in
    find_best_hybrid_molecular.get_representations catches it, prints, and moves on.

STATUS OF EACH BUILDER (checked 2026-09-10 against the pages named in each docstring):
    create_chemeleon   IMPLEMENTED  loading code read from the CheMeleon repo file
    create_mol_jepa    IMPLEMENTED  loading code read from the HF model card
    create_glacier     IMPLEMENTED  loading code read from the HF model card + source
    create_ouroboros   IMPLEMENTED  class + encode() read from the GitHub source
    create_chemfm      IMPLEMENTED  from_pretrained ids read from the GitHub README;
                                    pooling is our choice, stated in the docstring
    create_mist        STUB         no weights URL and no loading call found anywhere
    create_coati       IMPLEMENTED  load + embed calls read from the repo README/source

None of these has ever been run in this repo. Each docstring says what was read and
what was not. Torch, transformers and every model package are imported INSIDE the
function bodies, so importing this module costs nothing.
"""

import os
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _parse_inputs(smiles_list, canonical=False):
    """Split inputs into the ones RDKit can parse and the ones it cannot.

    Returns (keep_idx, keep_smiles). keep_idx[j] is the position in the original
    list that keep_smiles[j] came from, so rows can be scattered back in order.
    """
    keep_idx, keep_smiles = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        # RDKit parses '' into a molecule with zero atoms rather than returning
        # None. A zero-atom graph raises inside chemprop/DGL, which would fail the
        # whole batch and zero every molecule sharing it, so send it down the
        # zero-row path on its own.
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        keep_idx.append(i)
        keep_smiles.append(Chem.MolToSmiles(mol) if canonical else smi)
    return keep_idx, keep_smiles


def _assemble(n_inputs, pieces, name):
    """Build the (n_inputs, D) matrix from encoded pieces, zero-filling the rest.

    pieces is a list of (original_indices, array_of_rows). D is taken from the
    first piece. Everything that never got a row is left at zero and counted.
    """
    if not pieces:
        raise RuntimeError(
            f"{name}: nothing encoded — either RDKit parsed none of the "
            f"{n_inputs} SMILES, or every batch raised. See the warnings above "
            f"for the underlying error."
        )
    pieces = [(idx, np.atleast_2d(np.asarray(rows, dtype=np.float32)))
              for idx, rows in pieces]
    dim = pieces[0][1].shape[1]
    out = np.zeros((n_inputs, dim), dtype=np.float32)
    filled = np.zeros(n_inputs, dtype=bool)
    for idx, rows in pieces:
        # A model that returns a different number of rows than the batch it was
        # given has already lost the input order: row j is no longer molecule
        # idx[j]. Refuse the whole matrix rather than write misaligned rows.
        if rows.shape[0] != len(idx):
            raise RuntimeError(
                f"{name}: a batch of {len(idx)} molecules came back with "
                f"{rows.shape[0]} rows. Row order is no longer trustworthy — "
                f"the model dropped or reordered molecules internally, and c13 "
                f"matches these rows to labels by position."
            )
        if rows.shape[1] != dim:
            raise RuntimeError(
                f"{name}: a batch came back {rows.shape[1]} features wide, "
                f"another {dim} wide. The feature block would not be rectangular."
            )
        for j, i in enumerate(idx):
            out[i] = rows[j]
            filled[i] = True
    n_zero = int((~filled).sum())
    if n_zero:
        print(f"  WARNING: {name}: {n_zero} of {n_inputs} molecules zero-filled "
              f"(unparseable SMILES or failed batch)")
    return out


def _to_device(obj, device):
    """Best-effort move of a collated batch (dict / list / tensor / graph) to device."""
    if hasattr(obj, 'to'):
        try:
            return obj.to(device)
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_to_device(v, device) for v in obj]
        # chemprop collates into a NamedTuple (TrainingBatch). Rebuilding it as a
        # plain tuple loses the field names the model's forward reads by name, so
        # rebuild it as its own type.
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*moved)
        return tuple(moved) if isinstance(obj, tuple) else type(obj)(moved)
    return obj


def _pick_device(device):
    import torch
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


# =============================================================================
# CheMeleon — D-MPNN pretrained on Mordred descriptors
# =============================================================================

_CHEMELEON_FINGERPRINT = None


def create_chemeleon(smiles_list, batch_size=32, device=None):
    """
    Create CheMeleon embeddings (frozen D-MPNN fingerprint, no fine-tuning).

    CheMeleon is a directed message-passing network pretrained by regressing 1M
    PubChem molecules onto their computed Mordred descriptors. The published
    fingerprint width is 2048; the actual width is whatever the checkpoint's
    message-passing block reports (mp.output_dim), read at load time.

    POOLING: chemprop's MeanAggregation — per-atom hidden states after message
    passing are averaged over the atoms of the molecule, giving one row per
    molecule. This is exactly what CheMeleonFingerprint does upstream.

    Reference: Burns & Green, "CheMeleon: Foundation Models on a Budget"
    (arXiv 2506.15792).

    Args:
        smiles_list: List of SMILES strings
        batch_size: Molecules per forward pass (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: CheMeleon embeddings (n_molecules, 2048)

    SETUP REQUIRED:
        pip install 'chemprop>=2.2.0'
        The 2048-wide message-passing checkpoint (chemeleon_mp.pt) downloads
        automatically on first use from
        https://zenodo.org/records/15460715/files/chemeleon_mp.pt
        into $KIRBY_MODELS_DIR/chemeleon (or ~/.chemprop when that is unset).

    API SOURCE: https://github.com/JacksonBurns/CheMeleon — the loading code below
    reproduces chemeleon_fingerprint.py from that repo (read 2026-09-10). The class
    is NOT importable from the chemprop package; the repo README says the file is
    meant to be copied, which is what this function does.
    """
    from pathlib import Path
    from urllib.request import urlretrieve

    import torch

    global _CHEMELEON_FINGERPRINT

    device = _pick_device(device)

    try:
        from chemprop import featurizers, nn as chemprop_nn
        from chemprop.data import BatchMolGraph
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN
    except ImportError as e:
        raise ImportError(
            f"CheMeleon needs chemprop: {e}\n"
            "  pip install 'chemprop>=2.2.0'"
        )

    if _CHEMELEON_FINGERPRINT is None:
        models_dir = os.environ.get('KIRBY_MODELS_DIR')
        ckpt_dir = Path(models_dir) / 'chemeleon' if models_dir else Path.home() / '.chemprop'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mp_path = ckpt_dir / 'chemeleon_mp.pt'
        if not mp_path.exists():
            print(f"Downloading CheMeleon checkpoint to {mp_path}...", flush=True)
            try:
                urlretrieve(
                    'https://zenodo.org/records/15460715/files/chemeleon_mp.pt',
                    mp_path,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download the CheMeleon checkpoint: {e}\n"
                    "Download it by hand and put it at "
                    f"{mp_path}:\n"
                    "  https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
                )

        print("Loading CheMeleon...", flush=True)
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = chemprop_nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        model = MPNN(
            message_passing=mp,
            agg=chemprop_nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=mp.output_dim),  # never used
        )
        model.eval()
        model.to(device=device)
        _CHEMELEON_FINGERPRINT = (model, featurizers.SimpleMoleculeMolGraphFeaturizer(),
                                  BatchMolGraph)
        print(f"  Loaded on {device} ({mp.output_dim} features)")

    model, featurizer, BatchMolGraph = _CHEMELEON_FINGERPRINT

    keep_idx, keep_smiles = _parse_inputs(smiles_list)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch = keep_smiles[i:i + batch_size]
            try:
                bmg = BatchMolGraph(
                    [featurizer(Chem.MolFromSmiles(s)) for s in batch]
                )
                bmg.to(device=model.device)
                rows = model.fingerprint(bmg).numpy(force=True)
                pieces.append((idx, np.asarray(rows, dtype=np.float32)))
            except Exception as e:
                print(f"  WARNING: chemeleon batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'chemeleon')


# =============================================================================
# Mol-JEPA — joint-embedding predictive model over 14 molecular modalities
# =============================================================================

_MOL_JEPA_MODEL = None


def create_mol_jepa(smiles_list, batch_size=32, device=None):
    """
    Create Mol-JEPA embeddings (frozen, no fine-tuning).

    Mol-JEPA masks whole molecular modalities and predicts their latent
    representations. The model takes raw SMILES strings — it does its own
    featurisation internally — and returns a ModelOutput with three fields:
    predictions [batch, 12, 512], cls [batch, 512], embeddings [batch, 13, 512].

    POOLING: the learnable CLS token, out.cls, one row of 512 features per
    molecule. No averaging. (The per-modality latents out.embeddings could be
    mean-pooled to 512 instead; this function does not do that.)

    Reference: Mol-JEPA (Boehringer Ingelheim), arXiv 2608.22642.
    License note: weights are CC BY-NC 4.0 — non-commercial.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Molecules per forward pass (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: Mol-JEPA CLS embeddings (n_molecules, 512)

    SETUP REQUIRED:
        pip install torch torch-geometric transformers safetensors rdkit molfeat
        Weights download automatically from HuggingFace Flogrammer/Mol-JEPA.

    API SOURCE: https://huggingface.co/Flogrammer/Mol-JEPA — the model card's usage
    snippet (read 2026-09-10) is `AutoModel.from_pretrained("Flogrammer/Mol-JEPA",
    trust_remote_code=True)` followed by `out = model(smiles_list)`, and it names
    out.predictions / out.cls / out.embeddings with the shapes above.
    """
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import torch
    from transformers import AutoModel

    global _MOL_JEPA_MODEL

    device = _pick_device(device)

    if _MOL_JEPA_MODEL is None:
        print("Loading Mol-JEPA...", flush=True)
        try:
            _MOL_JEPA_MODEL = AutoModel.from_pretrained(
                "Flogrammer/Mol-JEPA",
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Flogrammer/Mol-JEPA: {e}\n"
                "The custom code needs: pip install torch-geometric molfeat safetensors rdkit\n"
                "Weights: https://huggingface.co/Flogrammer/Mol-JEPA (CC BY-NC 4.0)"
            )
        _MOL_JEPA_MODEL = _MOL_JEPA_MODEL.to(device).eval()
        print(f"  Loaded on {device}")

    keep_idx, keep_smiles = _parse_inputs(smiles_list)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch = keep_smiles[i:i + batch_size]
            try:
                out = _MOL_JEPA_MODEL(batch)
                cls = out.cls if hasattr(out, 'cls') else out[1]
                pieces.append((idx, cls.detach().cpu().numpy().astype(np.float32)))
            except Exception as e:
                print(f"  WARNING: mol_jepa batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'mol_jepa')


# =============================================================================
# GLACIER — graph + SMILES-text + RDKit-descriptor student, distilled from MiniMol
# =============================================================================

_GLACIER_MODEL = None
_GLACIER_DATA = None


def create_glacier(smiles_list, batch_size=32, device=None):
    """
    Create GLACIER embeddings (frozen student encoder, no fine-tuning).

    GLACIER fuses a graph encoder, a SMILES text encoder and a tabular encoder
    over 217 RDKit descriptors, distilled from a MiniMol teacher. The checkpoint
    used here is GLACIER-100k-MiniMol; its config.json gives output_dim 512 and
    fusion_dim 512, so one row is 512 features.

    POOLING: none is applied here. The model's own forward returns the fused
    student embedding, [batch, output_dim] — the atom-level readout is an
    attentive aggregation inside the graph encoder, not a mean this function takes.

    Reference: GLACIER, arXiv 2606.11382. MIT.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Molecules per forward pass (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: GLACIER fused embeddings (n_molecules, 512)

    SETUP REQUIRED:
        pip install 'chemprop==2.2.1' lightning transformers huggingface_hub
        The repo files (custom architecture) and weights download automatically
        via snapshot_download from glacier-hf/GLACIER-100k-MiniMol.

    ROW ORDER: build_dataloader's signature (read from the repo's data/dataloader.py)
    is (dataset, batch_size=1024, num_workers=0, shuffle=False, **kwargs) — shuffle
    is off by default, so the batch comes back in dataset order. The code below also
    checks the sampler is sequential and falls back to one molecule per batch if it
    is not, because a shuffled batch would put a molecule's features on another
    molecule's row.

    API SOURCE: https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol — the model
    card's snippet (read 2026-09-10) is snapshot_download + sys.path.append, then
    `from glacier_student import Glacier`, `Glacier.from_pretrained(...)`,
    `SmilesMoleculeDataset(smiles=[...])`, `build_dataloader(dataset, batch_size=1)`,
    `model(batch)`. glacier_student.py confirms forward(batch) -> Tensor [B, output_dim].
    The dataset's is_train defaults to False, so no SMILES randomisation, and the
    217 RDKit descriptors are computed by the dataset itself when rdkit=None.
    Forward's modality dropout is guarded by self.training, so eval() removes it.
    """
    import sys

    import torch

    global _GLACIER_MODEL, _GLACIER_DATA

    device = _pick_device(device)

    if _GLACIER_MODEL is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                f"GLACIER needs huggingface_hub: {e}\n"
                "  pip install huggingface_hub"
            )
        print("Loading GLACIER (GLACIER-100k-MiniMol)...", flush=True)
        try:
            repo_dir = snapshot_download(repo_id="glacier-hf/GLACIER-100k-MiniMol")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download glacier-hf/GLACIER-100k-MiniMol: {e}\n"
                "Weights: https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol (MIT)"
            )
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)
        try:
            from data.dataloader import SmilesMoleculeDataset, build_dataloader
            from glacier_student import Glacier
        except ImportError as e:
            raise ImportError(
                f"GLACIER's custom architecture failed to import: {e}\n"
                "It needs chemprop, lightning and transformers:\n"
                "  pip install 'chemprop==2.2.1' lightning transformers"
            )
        _GLACIER_MODEL = Glacier.from_pretrained("glacier-hf/GLACIER-100k-MiniMol")
        _GLACIER_MODEL = _GLACIER_MODEL.to(device).eval()
        _GLACIER_DATA = (SmilesMoleculeDataset, build_dataloader)
        print(f"  Loaded on {device}")

    SmilesMoleculeDataset, build_dataloader = _GLACIER_DATA

    keep_idx, keep_smiles = _parse_inputs(smiles_list)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch_smiles = keep_smiles[i:i + batch_size]
            try:
                dataset = SmilesMoleculeDataset(smiles=batch_smiles)
                loader = build_dataloader(dataset, batch_size=len(batch_smiles),
                                          shuffle=False)
                sampler = getattr(loader, 'sampler', None)
                if sampler is not None and not isinstance(
                        sampler, torch.utils.data.SequentialSampler):
                    # Order is not guaranteed — encode one molecule at a time so
                    # row i cannot pick up molecule j's features.
                    rows = []
                    for smi in batch_smiles:
                        one = build_dataloader(SmilesMoleculeDataset(smiles=[smi]),
                                               batch_size=1, shuffle=False)
                        b = _to_device(next(iter(one)), device)
                        rows.append(_GLACIER_MODEL(b).detach().cpu().numpy())
                    arr = np.vstack(rows)
                else:
                    b = _to_device(next(iter(loader)), device)
                    arr = _GLACIER_MODEL(b).detach().cpu().numpy()
                pieces.append((idx, np.asarray(arr, dtype=np.float32)))
            except Exception as e:
                print(f"  WARNING: glacier batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'glacier')


# =============================================================================
# Ouroboros — WLN graph encoder with global attention pooling, 2048 features
# =============================================================================

_OUROBOROS_MODEL = None


def create_ouroboros(smiles_list, batch_size=32, device=None):
    """
    Create Ouroboros encodings (frozen encoder, no fine-tuning).

    Ouroboros encodes a molecular graph with a Weisfeiler-Lehman Network and
    pools atoms with DGL's GlobalAttentionPooling. encoder_config.json in the
    released checkpoint sets encoding_features to 2048, which matches the paper's
    "a representation module of 1D molecular representation with a size of 2048".

    POOLING: the model's own global attention pooling over atoms (a learned
    weighted sum), taken from GeminiMol.encode -> self.pooling(self.Encoder(smiles)).
    This function adds no pooling of its own.

    Reference: Ouroboros, Advanced Science 13(17), 2026.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Molecules per forward pass (default: 32)
        device: IGNORED — Ouroboros hardcodes .cuda() in MolecularEncoder and
                MolecularPooling, so it runs on GPU only. A RuntimeError is raised
                when no CUDA device is visible.

    Returns:
        np.ndarray: Ouroboros encodings (n_molecules, 2048)

    SETUP REQUIRED:
        1. Clone and install (needs DGL and dgllife):
             cd $KIRBY_MODELS_DIR
             git clone https://github.com/Wang-Lin-boop/Ouroboros
             cd Ouroboros && pip install . \\
               -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html \\
               --extra-index-url https://download.pytorch.org/whl/cu121
        2. Download a checkpoint and unzip it next to the repo:
             https://aideepmed.com/Ouroboros/Ouroboros_M1c.zip
             (M1d also exists: https://aideepmed.com/Ouroboros/Ouroboros_M1d.zip)
           The unzipped directory must contain MolEncoder.pt, Pooling.pt,
           encoder_config.json and pooling_config.json.
        3. Point KIRBY_OUROBOROS_MODEL at that directory, or place it as
           $KIRBY_MODELS_DIR/Ouroboros_M1c.

    API SOURCE: https://github.com/Wang-Lin-boop/Ouroboros — the README documents
    only the application scripts, so the call below was taken from the source
    itself (read 2026-09-10): ouroboros/model/Ouroboros.py defines
    `class Ouroboros(GeminiMol)` with __init__(model_name, batch_size=512,
    predictor_info={}, generator=True, ...), and ouroboros/model/GeminiMol.py
    defines `encode(self, smiles_list)` returning self.pooling(self.Encoder(smiles_list)).
    generator=False below skips the Generator weights, which are not needed for
    features. NOT RUN — no page shows an end-to-end feature-extraction example.

    Args note: batch_size is passed straight to the Ouroboros constructor.
    """
    import sys

    import torch

    global _OUROBOROS_MODEL

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Ouroboros calls .cuda() unconditionally in MolecularEncoder and "
            "MolecularPooling, so it cannot run on CPU. Submit this representation "
            "to a GPU node."
        )

    if _OUROBOROS_MODEL is None:
        from kirby.representations.molecular import get_models_dir
        models_dir = get_models_dir()

        # Find the repo (for the import) and the checkpoint directory (for the weights)
        repo_candidates = [
            os.path.join(models_dir, 'Ouroboros'),
            os.path.expanduser('~/repos/Ouroboros'),
            os.path.expanduser('~/kirby_models/Ouroboros'),
        ]
        repo_dir = next((p for p in repo_candidates
                         if os.path.exists(os.path.join(p, 'ouroboros', 'model',
                                                        'Ouroboros.py'))), None)
        if repo_dir is None:
            raise FileNotFoundError(
                "Ouroboros repo not found. Setup:\n"
                "  cd $KIRBY_MODELS_DIR\n"
                "  git clone https://github.com/Wang-Lin-boop/Ouroboros\n"
                "  cd Ouroboros && pip install . "
                "-f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html"
            )

        model_dir = os.environ.get('KIRBY_OUROBOROS_MODEL')
        if model_dir is None:
            for name in ('Ouroboros_M1c', 'Ouroboros_M1d'):
                candidate = os.path.join(models_dir, name)
                if os.path.exists(os.path.join(candidate, 'MolEncoder.pt')):
                    model_dir = candidate
                    break
        if model_dir is None or not os.path.exists(os.path.join(model_dir, 'MolEncoder.pt')):
            raise FileNotFoundError(
                "Ouroboros weights not found (need MolEncoder.pt + Pooling.pt).\n"
                "Download and unzip one of:\n"
                "  https://aideepmed.com/Ouroboros/Ouroboros_M1c.zip\n"
                "  https://aideepmed.com/Ouroboros/Ouroboros_M1d.zip\n"
                "then set KIRBY_OUROBOROS_MODEL to the unzipped directory."
            )

        pkg_dir = os.path.join(repo_dir, 'ouroboros')
        for path in (pkg_dir, repo_dir):
            if path not in sys.path:
                sys.path.insert(0, path)

        print(f"Loading Ouroboros from {model_dir}...", flush=True)
        try:
            from model.Ouroboros import Ouroboros
        except ImportError as e:
            raise ImportError(
                f"Failed to import Ouroboros: {e}\n"
                "It needs dgl and dgllife:\n"
                "  pip install . -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html\n"
                "  (run from the cloned Ouroboros repo)"
            )
        _OUROBOROS_MODEL = Ouroboros(
            model_name=model_dir,
            batch_size=batch_size,
            predictor_info={},
            generator=False,
        )
        print("  Loaded on cuda")

    keep_idx, keep_smiles = _parse_inputs(smiles_list, canonical=True)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch = keep_smiles[i:i + batch_size]
            try:
                feats = _OUROBOROS_MODEL.encode(batch)
                pieces.append((idx, feats.detach().cpu().numpy().astype(np.float32)))
            except Exception as e:
                print(f"  WARNING: ouroboros batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'ouroboros')


# =============================================================================
# ChemFM — 1B/3B LLaMA-style decoder on SMILES
# =============================================================================

_CHEMFM_MODEL = None
_CHEMFM_TOKENIZER = None


def create_chemfm(smiles_list, batch_size=32, device=None):
    """
    Create ChemFM embeddings (frozen decoder hidden states, no fine-tuning).

    ChemFM-1B is a LLaMA architecture over a 320-token SMILES vocabulary
    (config.json: architectures ["LlamaForCausalLM"], hidden_size 2048,
    vocab_size 320, max_position_embeddings 512), so one row is 2048 features.
    Set KIRBY_CHEMFM_MODEL to "ChemFM/ChemFM-3B" or "ChemFM/ChemFMv2-20M" to use
    a different size; the width then follows that checkpoint's hidden_size.

    POOLING: mean over the non-padding token hidden states of the last layer
    (last_hidden_state weighted by attention_mask, divided by the token count).
    This is OUR choice — the model card gives no pooling recipe. The published
    frozen-feature comparison for ChemFM does not say how it pooled either.

    Reference: ChemFM, PMC12764477.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Molecules per forward pass (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: ChemFM embeddings (n_molecules, 2048 for ChemFM-1B)

    SETUP REQUIRED:
        pip install transformers torch
        Weights (~2GB for 1B) download automatically from HuggingFace ChemFM/ChemFM-1B.
        A GPU is strongly preferred: this is a 1B-parameter decoder.

    API SOURCE: https://github.com/TheLuoFengLab/ChemFM — the Quick Start
    (read 2026-09-10) is `AutoModel.from_pretrained(model_name)` /
    `AutoTokenizer.from_pretrained(model_name)` with model_name one of
    "ChemFM/ChemFM-1B", "ChemFM/ChemFM-3B", "ChemFM/ChemFMv2-20M". Dimensions and
    architecture read from https://huggingface.co/ChemFM/ChemFM-1B/raw/main/config.json.
    CAVEAT: the HuggingFace model card README is EMPTY, so neither an embedding
    example nor a SMILES formatting rule was confirmed anywhere. The SMILES string
    is passed as-is, and the tokenizer's pad token is set to its eos token when it
    has none (config.json has no pad_token_id).
    """
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import torch
    from transformers import AutoModel, AutoTokenizer

    global _CHEMFM_MODEL, _CHEMFM_TOKENIZER

    device = _pick_device(device)
    model_name = os.environ.get('KIRBY_CHEMFM_MODEL', 'ChemFM/ChemFM-1B')

    if _CHEMFM_MODEL is None:
        print(f"Loading {model_name}...", flush=True)
        try:
            _CHEMFM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _CHEMFM_MODEL = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_name}: {e}\n"
                "  pip install -U transformers\n"
                f"Weights: https://huggingface.co/{model_name}"
            )
        if _CHEMFM_TOKENIZER.pad_token is None:
            _CHEMFM_TOKENIZER.pad_token = _CHEMFM_TOKENIZER.eos_token
        _CHEMFM_MODEL = _CHEMFM_MODEL.to(device).eval()
        print(f"  Loaded on {device}")

    keep_idx, keep_smiles = _parse_inputs(smiles_list)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch = keep_smiles[i:i + batch_size]
            try:
                encoded = _CHEMFM_TOKENIZER(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                )
                inputs = {k: v.to(device) for k, v in encoded.items()}
                outputs = _CHEMFM_MODEL(**inputs)
                token_embeddings = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).to(token_embeddings.dtype)
                summed = torch.sum(token_embeddings * mask, dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                emb = (summed / counts).float().cpu().numpy()
                pieces.append((idx, emb.astype(np.float32)))
            except Exception as e:
                print(f"  WARNING: chemfm batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'chemfm')


# =============================================================================
# MIST — STUB, weights location never confirmed
# =============================================================================

def create_mist(smiles_list, batch_size=32, device=None):
    """
    MIST embeddings — NOT IMPLEMENTED. The loading API could not be confirmed.

    STATUS: STUB. Raises NotImplementedError so the domain loader skips it.

    WHAT WAS CONFIRMED (2026-09-10):
      - The paper is "Foundation Models for Discovery and Exploration in Chemical
        Space", arXiv 2510.18900, and MIST is a family of masked-language models
        over SMILES tokenised with Smirk, 28M to 1.8B parameters, pretrained on up
        to 6B Enamine REALSpace molecules.
      - The paper states "All code, model weights, and training recipes have been
        open-sourced".

    WHAT COULD NOT BE CONFIRMED:
      - No GitHub URL, no HuggingFace repo id and no Zenodo DOI appears in the
        arXiv HTML (v2) or on the arXiv abstract page.
      - A HuggingFace model-name search for MIST returned only Mistral language
        models, nothing molecular.
      - So: no loading call, no checkpoint name, no embedding width, and no
        statement of how per-token states are pooled.

    Writing a from_pretrained guess here would fail on the cluster, so it is not
    written. To finish this builder: find the real repository, read its README,
    then replace this body — the pooling the candidate list intends is a mean over
    the transformer's token hidden states, one row per molecule.
    """
    raise NotImplementedError(
        "create_mist: MIST's weights location and loading call are UNCONFIRMED. "
        "arXiv 2510.18900 says code and weights are open-sourced but names no "
        "GitHub, HuggingFace or Zenodo location, and a HuggingFace search finds no "
        "molecular MIST model. No API is guessed here. See the docstring."
    )


# =============================================================================
# COATI — multimodal E(3)-equivariant SMILES + 3D encoder
# =============================================================================

_COATI_ENCODER = None
_COATI_TOKENIZER = None


def create_coati(smiles_list, batch_size=32, device=None):
    """
    Create COATI embeddings (frozen encoder, no fine-tuning).

    COATI is a contrastively trained multimodal encoder over a SMILES transformer
    and an E(3)-equivariant point-cloud network. This function uses the checkpoint
    the repo README calls recommended, grande_closed.pkl, whose embedding width
    the README gives as 256. (COATI2 is a one-line change: load_coati2 from
    coati.models.simple_coati2.io with
    doc_url="s3://terray-public/models/coati2_chiral_03-08-24.pkl", width 512.)

    POOLING: none is applied here. embed_smiles_batch returns the encoder's own
    single vector per molecule, taken from the SMILES-side encoder after the
    "[SMILES]" ... "[STOP]" token sequence, one row per input.

    Reference: Kaufman et al., COATI, JCIM 2024.

    Args:
        smiles_list: List of SMILES strings (canonicalised with RDKit first, which
                     is what the repo's own single-molecule embed_smiles does)
        batch_size: Molecules per forward pass (default: 32)
        device: 'cpu' or 'cuda' (default: auto-detect)

    Returns:
        np.ndarray: COATI embeddings (n_molecules, 256)

    SETUP REQUIRED:
        git clone https://github.com/terraytherapeutics/COATI.git
        cd COATI && pip install .
        The checkpoint is pulled from the public S3 bucket
        s3://terray-public/models/grande_closed.pkl on first load, so the node
        needs outbound network access (the loader fetches it itself).

    API SOURCE: https://github.com/terraytherapeutics/COATI (README, read 2026-09-10)
    gives `from coati.models.io import load_e3gnn_smiles_clip_e2e` and
    `encoder, tokenizer = load_e3gnn_smiles_clip_e2e(freeze=True, device=...,
    doc_url="s3://terray-public/models/grande_closed.pkl")`.
    coati/generative/coati_purifications.py (read 2026-09-10) defines
    `embed_smiles_batch(smiles_list, encoder, tokenizer) -> torch.Tensor` of shape
    (batch, embedding_dim) on the encoder's device.
    """
    import torch

    global _COATI_ENCODER, _COATI_TOKENIZER

    device = _pick_device(device)

    if _COATI_ENCODER is None:
        try:
            from coati.models.io import load_e3gnn_smiles_clip_e2e
        except ImportError as e:
            raise ImportError(
                f"COATI is not installed: {e}\n"
                "  git clone https://github.com/terraytherapeutics/COATI.git\n"
                "  cd COATI && pip install ."
            )
        print("Loading COATI (grande_closed)...", flush=True)
        try:
            _COATI_ENCODER, _COATI_TOKENIZER = load_e3gnn_smiles_clip_e2e(
                freeze=True,
                device=torch.device(device),
                doc_url="s3://terray-public/models/grande_closed.pkl",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the COATI checkpoint: {e}\n"
                "It is fetched from the public bucket "
                "s3://terray-public/models/grande_closed.pkl — the node needs "
                "outbound network access."
            )
        print(f"  Loaded on {device}")

    from coati.generative.coati_purifications import embed_smiles_batch

    keep_idx, keep_smiles = _parse_inputs(smiles_list, canonical=True)
    pieces = []
    with torch.no_grad():
        for i in range(0, len(keep_smiles), batch_size):
            idx = keep_idx[i:i + batch_size]
            batch = keep_smiles[i:i + batch_size]
            try:
                vecs = embed_smiles_batch(batch, _COATI_ENCODER, _COATI_TOKENIZER)
                pieces.append((idx, vecs.detach().cpu().numpy().astype(np.float32)))
            except Exception as e:
                print(f"  WARNING: coati batch at {i} failed: {e}")

    return _assemble(len(smiles_list), pieces, 'coati')
