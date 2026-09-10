"""Forward-compatibility reaction representations for c13.

One builder per candidate reaction encoder. Each takes the dataset's reaction
SMILES and returns one row per reaction, float32, same order as the input.
No training, no fine-tuning, no label use.

RXNGraphormer (IMPLEMENTED):
  Graph encoder + transformer pretrained on 13M reactions. Reaction embedding
  taken from the package's own RXNEMB.gen_rxn_emb.
  Xu et al., Nature Machine Intelligence 7(9) 1561-1571, 2025.

ReaMVP (STUB):
  Bidirectional-GRU sequence encoder concatenated with a SchNet conformer
  encoder. Checkpoints are in the repo, but the repo exposes no embedding call.
  Han et al., Journal of Cheminformatics 16, 2024.

Egret (STUB, and refuses by default):
  BERT over reaction SMILES, 12 layers, 4 heads, hidden size 256. The released
  checkpoint is the Buchwald-Hartwig yield predictor, so embedding either
  reaction dataset with it uses the labels.
  Yin et al., Research (AAAS), 2024.

Uni-Mol3 (STUB):
  T5 over discrete 3D-aware molecule tokens. Ships Uni-Core training scripts
  only; no embedding entry point.
  Wu et al., arXiv 2508.00920, 2025.

diff_chemeleon / concat_chemeleon (IMPLEMENTED):
  Not new architectures. Each reaction component is embedded with CheMeleon,
  then the component vectors are differenced or concatenated in fixed slot
  order — the construction diff_ecfp4 and concat_ecfp4 already use in
  reaction.py. The CheMeleon builder itself lives in forward_compat_molecular.
  Burns et al., arXiv 2506.15792, 2025.

Every builder raises ImportError or RuntimeError (never sys.exit, never a bare
assert) when a package or a weight file is missing, so the reaction loader in
tests/experiments/_shared.py can warn and move on. A builder whose loading call
could not be confirmed against the model's own repository raises
NotImplementedError with what was confirmed and what was not.
"""

import os
from typing import List, Tuple

import numpy as np


# Cached RXNGraphormer embedding object, keyed by checkpoint directory.
_RXNGRAPHORMER_EMB = None
_RXNGRAPHORMER_PATH = None


# =========================================================================
# SHARED PARSING
# =========================================================================

def _split_rxn(rxn: str) -> Tuple[List[str], List[str]]:
    """Split one reaction SMILES into (reactant SMILES, product SMILES).

    Handles both forms the reaction datasets produce:
      'reactants>>products'          — what reaction_yields.py writes
      'reactants>reagents>products'  — reagents are folded onto the reactant
                                       side, which is where the Buchwald-
                                       Hartwig loader already puts ligand,
                                       base and additive.

    Empty fragments are dropped.
    """
    if '>>' in rxn:
        left, right = rxn.split('>>', 1)
        mid = ''
    else:
        parts = rxn.split('>')
        if len(parts) == 3:
            left, mid, right = parts
        elif len(parts) == 2:
            left, mid, right = parts[0], '', parts[1]
        else:
            left, mid, right = rxn, '', ''

    reactants = [s.strip() for s in (left + ('.' + mid if mid else '')).split('.')
                 if s.strip()]
    products = [s.strip() for s in right.split('.') if s.strip()]
    return reactants, products


def _to_double_arrow(rxn_smiles: List[str]) -> List[str]:
    """Normalise every reaction to 'reactants>>products'.

    RXNGraphormer's gen_rxn_emb splits on '>>' and indexes [1], so a
    three-field reaction SMILES raises there. This folds reagents onto the
    reactant side first.
    """
    out = []
    for rxn in rxn_smiles:
        reactants, products = _split_rxn(rxn)
        out.append('.'.join(reactants) + '>>' + '.'.join(products))
    return out


def _embed_unique(smiles: List[str], mol_rep_fn, batch_size: int) -> Tuple[dict, int]:
    """Embed each distinct molecule once with a molecular rep function.

    Returns (mapping from SMILES to its row, embedding width). Empty strings
    are never sent to the model; they get a zero row from the caller.
    """
    uniq = sorted({s for s in smiles if s})
    if not uniq:
        raise RuntimeError("No parseable component SMILES in this reaction set")

    # Decide whether the builder takes batch_size by reading its signature, not
    # by calling it and catching TypeError. Catching TypeError around the call
    # would also swallow a TypeError raised deep inside the builder, then embed
    # every molecule a second time before failing with the error pointing at
    # the wrong line.
    import inspect
    try:
        takes_batch = 'batch_size' in inspect.signature(mol_rep_fn).parameters
    except (TypeError, ValueError):
        takes_batch = False
    reps = mol_rep_fn(uniq, batch_size=batch_size) if takes_batch else mol_rep_fn(uniq)

    reps = np.asarray(reps, dtype=np.float32)
    if reps.shape[0] != len(uniq):
        raise RuntimeError(
            f"Molecular builder returned {reps.shape[0]} rows for "
            f"{len(uniq)} molecules — rows would not line up with reactions"
        )
    return {s: reps[i] for i, s in enumerate(uniq)}, int(reps.shape[1])


def _load_chemeleon():
    """Return the CheMeleon molecular builder, imported lazily.

    Lives in the molecular lane so both lanes load the same weights.
    """
    try:
        from .forward_compat_molecular import create_chemeleon
    except ImportError as e:
        raise RuntimeError(
            "CheMeleon builder not found: "
            "src/kirby/representations/forward_compat_molecular.py must define "
            f"create_chemeleon(smiles, batch_size). Import failed with: {e}"
        )
    return create_chemeleon


# =========================================================================
# RXNGRAPHORMER
# =========================================================================

def create_rxngraphormer(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """RXNGraphormer reaction embedding from the pretrained classification model.

    Model: RXNGraphormer, Xu et al., Nature Machine Intelligence 7(9)
    1561-1571 (2025), DOI 10.1038/s42256-025-01098-4. GNN over each molecule
    graph plus a transformer across molecules, pretrained on 13 million
    reactions (6.8M real, an equal number with fictitious bond changes).

    Pooling — this is what rxngraphormer/rxn_emb.py does, read from the repo:
    the reactant-side and product-side transformer states are each mean-pooled
    over tokens (trans_readout 'mean'), then the reactant vector, the product
    vector and their absolute difference are concatenated
    (split_merge_method 'all'), then pushed through every decoder linear +
    batch-norm layer except the last. One row per reaction.

    Width: emb_dim is 256 in the pretraining config and the decoder narrows
    the 3 x 256 concatenation, so the width is whatever that checkpoint's
    decoder emits (the paper entry reports 768). This function reads it off
    the returned tensor rather than assuming it.

    Args:
        rxn_smiles: Reaction SMILES. Both 'reactants>>products' and
            'reactants>reagents>products' are accepted; reagents are folded
            onto the reactant side.
        batch_size: Reactions per forward pass (default 16).

    Returns:
        np.ndarray of shape (len(rxn_smiles), D), float32, input order.
        A chunk that fails to encode leaves zero rows and prints a count.

    Requires: pip install rxngraphormer
        (needs torch 1.12.1 + torch_geometric 2.3.1; see the repo's install
        line at https://github.com/licheng-xu-echo/RXNGraphormer)

    SETUP REQUIRED — weights are a manual download, not auto-fetched:
        1. Download pretrained_classification_model.zip from the figshare
           record https://doi.org/10.6084/m9.figshare.28356077 (MIT).
        2. Unzip it. The directory must contain parameters.json and
           model/valid_checkpoint.pt.
        3. Point KIRBY_RXNGRAPHORMER_MODEL at that directory, or place it at
           $KIRBY_CACHE_DIR/rxngraphormer/pretrained_classification_model.

    Note: gen_rxn_emb writes a temporary ./rxn_emb_tmp directory relative to
    the working directory, so this runs inside a scratch directory and
    restores the working directory afterwards.
    """
    global _RXNGRAPHORMER_EMB, _RXNGRAPHORMER_PATH

    n = len(rxn_smiles)
    if n == 0:
        raise RuntimeError("RXNGraphormer: empty reaction list")

    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "RXNGraphormer requires PyTorch: pip install torch==1.12.1"
        )
    try:
        from rxngraphormer.rxn_emb import RXNEMB
    except ImportError as e:
        raise ImportError(
            "RXNGraphormer is not installed. pip install rxngraphormer "
            "-f https://data.pyg.org/whl/torch-1.12.0+cu113.html "
            f"(needs torch_geometric). Import error: {e}"
        )

    model_path = os.environ.get('KIRBY_RXNGRAPHORMER_MODEL')
    if not model_path:
        cache_root = os.environ.get(
            'KIRBY_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'kirby'))
        model_path = os.path.join(
            cache_root, 'rxngraphormer', 'pretrained_classification_model')

    param_json = os.path.join(model_path, 'parameters.json')
    ckpt = os.path.join(model_path, 'model', 'valid_checkpoint.pt')
    if not (os.path.isfile(param_json) and os.path.isfile(ckpt)):
        raise RuntimeError(
            f"RXNGraphormer weights not found at {model_path}. Download "
            "pretrained_classification_model.zip from "
            "https://doi.org/10.6084/m9.figshare.28356077, unzip it so the "
            "directory holds parameters.json and model/valid_checkpoint.pt, "
            "and set KIRBY_RXNGRAPHORMER_MODEL to that directory."
        )

    if _RXNGRAPHORMER_EMB is None or _RXNGRAPHORMER_PATH != model_path:
        print(f"  Loading RXNGraphormer from {model_path}...")
        _RXNGRAPHORMER_EMB = RXNEMB(pretrained_model_path=model_path,
                                    model_type="classifier")
        _RXNGRAPHORMER_PATH = model_path
    emb_calc = _RXNGRAPHORMER_EMB

    normalised = _to_double_arrow(rxn_smiles)

    import shutil
    import tempfile

    # gen_rxn_emb asserts at least 2 reactions and writes ./rxn_emb_tmp in
    # the working directory, so chunks are at least 2 long and everything
    # runs in a scratch directory.
    chunk = max(2, batch_size)
    X = None
    D = None
    n_zero_rows = 0
    cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix='kirby_rxngraphormer_')
    try:
        os.chdir(scratch)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            batch = normalised[start:end]
            pad = 0
            if len(batch) < 2:  # final chunk of a single reaction
                batch = batch + [batch[-1]]
                pad = 1
            try:
                emb = emb_calc.gen_rxn_emb(batch, batch_size=batch_size)
                emb = np.asarray(emb.detach().cpu().numpy()
                                 if hasattr(emb, 'detach') else emb,
                                 dtype=np.float32)
                if pad:
                    emb = emb[:-1]
                if emb.shape[0] != (end - start):
                    raise RuntimeError(
                        f"returned {emb.shape[0]} rows for {end - start} reactions")
                if X is None:
                    D = int(emb.shape[1])
                    X = np.zeros((n, D), dtype=np.float32)
                X[start:end] = emb
            except Exception as e:
                n_zero_rows += (end - start)
                print(f"  WARNING: RXNGraphormer chunk {start}:{end} failed: {e}")
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch, ignore_errors=True)

    if X is None:
        raise RuntimeError(
            "RXNGraphormer produced no embedding for any reaction, so the "
            "embedding width is unknown. Check the checkpoint directory and "
            "that the reaction SMILES parse in RDKit."
        )
    if n_zero_rows:
        print(f"  WARNING: RXNGraphormer zero-filled {n_zero_rows}/{n} rows")
    print(f"  RXNGraphormer embeddings: {X.shape}")
    return X


# =========================================================================
# ReaMVP — STUB, loading call not confirmed
# =========================================================================

def create_reamvp(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """ReaMVP reaction embedding — STUB, the loading call is UNVERIFIED.

    Model: ReaMVP, Han et al., Journal of Cheminformatics 16 (2024),
    DOI 10.1186/s13321-024-00815-2. A 256-wide bidirectional GRU over
    reaction SMILES concatenated with a SchNet 3D-conformer encoder, both
    pretrained in two stages.

    Intended pooling if it is ever written: the sequence encoder's final
    hidden state concatenated with the conformer encoder's pooled output,
    one row per reaction.

    CONFIRMED by reading https://github.com/Meteor-han/ReaMVP (MIT):
      - The stage-2 checkpoints ship in the repo:
        checkpoint/rnn_geo/stage2_256_0.001_cos_mse_1.0_1.0_1_best.pt
        (also rnn/, geo/, gnn_geo/ variants and a stage-1 checkpoint).
      - models/reaction.py builds ReactionCLModel out of EmbedGRUSMILES (or
        EmbedLSTMSMILES), SchNet and GeoGRU (or GeoLSTM).
      - Requirements: python 3.10, torch 2.0.0, dgl 1.0.2, rdkit 2022.9.5.

    NOT CONFIRMED, which is why this raises:
      - The repo is not a package and has no embedding entry point. Every
        module does `from models.rnn import *` / `from utils import *`, so it
        only imports from the repo root.
      - ReactionCLModel takes an `args` namespace of about twenty
        hyperparameters (vocab_size, smiles_embed_size, smiles_hidden_dim,
        rnn_hidden_dim, sch_n_interaction, ...). The values that match the
        released checkpoint are documented nowhere; the README shows only the
        training command lines.
      - The SMILES vocabulary and the conformer construction are produced
        inside downstream/prepare_data.py, with no documented call.

    Raises:
        NotImplementedError, always.
    """
    raise NotImplementedError(
        "ReaMVP has no embedding entry point. Confirmed: MIT checkpoints at "
        "https://github.com/Meteor-han/ReaMVP/tree/main/checkpoint/rnn_geo and "
        "the encoder classes in models/reaction.py. Not confirmed: the repo is "
        "not importable as a package, the args namespace matching the released "
        "checkpoint is undocumented, and the SMILES vocabulary and conformer "
        "construction have no documented call. Writing this needs the repo "
        "cloned and its stage-2 training script read end to end."
    )


# =========================================================================
# EGRET — STUB, and refuses the released checkpoint
# =========================================================================

def create_egret(rxn_smiles: List[str], batch_size: int = 16,
                 allow_finetuned_checkpoint: bool = False) -> np.ndarray:
    """Egret reaction embedding — REFUSES by default, and is a STUB besides.

    Model: Egret, Yin et al., Research (AAAS) 2024, PMC10777739. BERT over
    tokenised reaction SMILES, 12 layers, 4 attention heads, hidden size 256,
    pretrained on Pistachio with masked language modelling plus a contrastive
    loss over reaction conditions.

    Intended pooling if it is ever written: mean over reaction-SMILES tokens,
    the same pooling create_rxnfp uses, giving 256 features per reaction.

    ⚠ LABEL LEAKAGE — why this refuses. The checkpoint behind the Drive link
    in the candidate list
    (https://drive.google.com/file/d/1MNzGyMYtNDt-TshKtsJkEEjHGtXTJhLN/view)
    is the yield-prediction model, not the pretrained-only encoder. Confirmed
    by reading inference/yield_predict.py in the repo: it loads the download
    as a single-output regression head and denormalises predictions with a
    stored yield mean of 79.29 and standard deviation of 18.86 — statistics
    fitted on the Buchwald-Hartwig yields. Embedding buchwald_hartwig or
    bh_doyle_plates with those weights puts the labels into the features, and
    the c13 delta for this representation would measure nothing.
    A pretrained-only checkpoint may be inside the larger bundle at
    https://drive.google.com/file/d/1K_EVByx5Vul5HuO3z_tWiVwVm37_jxPa/view
    (models and datasets), but nobody has confirmed that.

    CONFIRMED by reading https://github.com/xiaodanyin/Egret (readme.md,
    inference/yield_predict.py, inference/models.py):
      - Loading goes through SmilesClassificationModel('bert', <model dir>,
        num_labels=1), a simpletransformers subclass in inference/models.py.
      - Dependencies: transformers 4.18.0, simpletransformers 0.61.13,
        torch >= 1.13, rdkit >= 2020; tokenisation is Schwaller's SMILES
        tokenizer, shipped as inference/tokenization.py.

    NOT CONFIRMED, which is why this raises even when the refusal is waived:
      - No embedding-extraction call anywhere in the repo — the only inference
        path is yield prediction to a CSV.
      - Whether the downloaded directory is a HuggingFace bert directory that
        transformers could reopen for hidden states was not verified against
        any page; the Drive files cannot be listed without downloading them.

    Args:
        rxn_smiles: Reaction SMILES.
        batch_size: Sequences per batch.
        allow_finetuned_checkpoint: Must be True to get past the leakage
            refusal. Only set it on a dataset that is NOT Buchwald-Hartwig
            and shares no reactions with it.

    Raises:
        RuntimeError if allow_finetuned_checkpoint is False.
        NotImplementedError otherwise.
    """
    if not allow_finetuned_checkpoint:
        raise RuntimeError(
            "Egret refused: the released checkpoint is the Buchwald-Hartwig "
            "yield head (inference/yield_predict.py denormalises with yield "
            "mean 79.29 / std 18.86), so embedding buchwald_hartwig or "
            "bh_doyle_plates with it trains on the labels. Find the "
            "pretrained-only checkpoint, or pass "
            "allow_finetuned_checkpoint=True for a reaction set that shares "
            "no reactions with Buchwald-Hartwig."
        )
    raise NotImplementedError(
        "Egret has no embedding entry point. Confirmed: "
        "SmilesClassificationModel('bert', <dir>, num_labels=1) from "
        "https://github.com/xiaodanyin/Egret is the loading call for yield "
        "prediction, BERT 12 layers / 4 heads / hidden 256. Not confirmed: no "
        "call anywhere returns hidden states, and the layout of the "
        "downloaded checkpoint directory was never seen."
    )


# =========================================================================
# UNI-MOL3 — STUB, no embedding entry point
# =========================================================================

def create_unimol3(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """Uni-Mol3 reaction embedding — STUB, no embedding entry point exists.

    Model: Uni-Mol3, Wu et al., arXiv 2508.00920 (2025). T5 backbone, 8 heads,
    8 encoder and 8 decoder layers, hidden dimension 768, initialised from
    Uni-Mol2 84M. A molecule becomes discrete 3D-aware tokens through the
    Mol-Tokenizer, and a reaction is the resulting token sequence.

    Intended pooling if it is ever written: mean over encoder token states,
    768 features per reaction.

    CONFIRMED by reading https://github.com/LirongWu/Uni-Mol3 (MIT):
      - Four downloads: Uni-Mol2 84M base (321 MB) at
        https://huggingface.co/dptech/Uni-Mol2, Uni-Tokenizer (1.44 GB),
        molecular pre-training (1.25 GB) and reaction pre-training (1.25 GB),
        the last three on Google Drive.
      - Training and fine-tuning run through the Uni-Core CLI (unicore-train)
        with architectures 'T5_base' / 'unimol3_84M' and tasks 'unimol3_T5',
        'unimol3_tokenizer', 'unimol3_regression'; checkpoints are passed as
        --finetune-from-model / --weight-dir-path.

    NOT CONFIRMED, which is why this raises:
      - The README contains no Python at all — no model class, no
        from_pretrained call, no tokenizer call, and no forward pass that
        returns hidden states.
      - Extracting a representation means installing Uni-Core, registering
        the repo as a user directory, rebuilding the task, and writing a
        pooling hook against the encoder. That hook does not exist yet.

    Raises:
        NotImplementedError, always.
    """
    raise NotImplementedError(
        "Uni-Mol3 ships no embedding entry point. Confirmed: MIT repo "
        "https://github.com/LirongWu/Uni-Mol3, reaction pre-training "
        "checkpoint on Google Drive, Uni-Mol2 84M base at "
        "https://huggingface.co/dptech/Uni-Mol2, Uni-Core CLI training with "
        "architecture T5_base and task unimol3_T5. Not confirmed: no Python "
        "loading call, no tokenizer call, no forward pass returning hidden "
        "states. A pooling hook has to be written against the encoder first."
    )


# =========================================================================
# CHEMELEON THROUGH THE EXISTING PER-COMPONENT CONSTRUCTION
# =========================================================================

def create_diff_chemeleon(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """Product CheMeleon minus reactant CheMeleon — 2048 features.

    Model: CheMeleon, Burns et al., arXiv 2506.15792 (2025). MIT code, CC0
    weights, https://doi.org/10.5281/zenodo.15426600. The molecule builder is
    create_chemeleon in forward_compat_molecular.py; this file only arranges
    reaction components, exactly as create_diff_ecfp4 does.

    Pooling: each molecule gets one 2048-length CheMeleon row. The product
    side is the mean over product molecules (one product in both reaction
    datasets). The reactant side is the mean over reactant molecules —
    mean-pooling is what _make_reactant_rep in reaction.py already does for
    continuous molecular reps, where create_diff_ecfp4 uses OR because its
    inputs are bits. The returned row is product mean minus reactant mean.

    What an unparseable component actually does: create_chemeleon returns one
    row per molecule it was given and zero-fills a molecule it cannot
    featurise rather than dropping it. So a component RDKit rejects arrives
    here as a zero vector and is averaged in with the rest — it pulls that
    side's mean toward zero, it is not left out of the mean. The printed
    zero-row count below therefore only fires for a reaction that yielded no
    component SMILES at all on either side, which is rarer than the count
    suggests. Rows are never dropped and never reordered either way.

    Args:
        rxn_smiles: Reaction SMILES. Reagents in a three-field SMILES are
            folded onto the reactant side.
        batch_size: Molecules per CheMeleon batch (default 16).

    Returns:
        np.ndarray of shape (len(rxn_smiles), 2048), float32, input order.

    Requires: whatever create_chemeleon requires (see
        src/kirby/representations/forward_compat_molecular.py).
    """
    n = len(rxn_smiles)
    if n == 0:
        raise RuntimeError("diff_chemeleon: empty reaction list")

    create_chemeleon = _load_chemeleon()

    parsed = [_split_rxn(r) for r in rxn_smiles]
    flat = [s for reactants, products in parsed for s in (reactants + products)]
    lookup, D = _embed_unique(flat, create_chemeleon, batch_size)

    X = np.zeros((n, D), dtype=np.float32)
    n_zero = 0
    for i, (reactants, products) in enumerate(parsed):
        r_rows = [lookup[s] for s in reactants if s in lookup]
        p_rows = [lookup[s] for s in products if s in lookup]
        if not r_rows and not p_rows:
            n_zero += 1
            continue
        r_mean = np.mean(r_rows, axis=0) if r_rows else np.zeros(D, dtype=np.float32)
        p_mean = np.mean(p_rows, axis=0) if p_rows else np.zeros(D, dtype=np.float32)
        X[i] = p_mean - r_mean

    if n_zero:
        print(f"  WARNING: diff_chemeleon zero-filled {n_zero}/{n} rows")
    print(f"  diff_chemeleon: {X.shape}")
    return X


def create_concat_chemeleon(rxn_smiles: List[str], batch_size: int = 16) -> np.ndarray:
    """CheMeleon rows for every reaction component, concatenated by slot.

    Model: CheMeleon, Burns et al., arXiv 2506.15792 (2025). MIT code, CC0
    weights, https://doi.org/10.5281/zenodo.15426600. The molecule builder is
    create_chemeleon in forward_compat_molecular.py.

    Pooling and slot order: each molecule gets one 2048-length CheMeleon row.
    Slots come from _parse_rxn_components in kirby.datasets.reaction_yields —
    the same function whose output feeds create_concat_ecfp4 — which splits
    reactants and products on '.', pads every reaction to the widest one in
    the set, and names the slots reactant_0 ... reactant_k, product_0 ...
    Slots are then laid out in sorted key order, as create_concat_ecfp4 lays
    them out. Component order is fixed per row by the dataset loader
    (Buchwald-Hartwig writes aryl halide, amine, ligand, base, additive in
    that order), so a slot means the same role in every row.

    Width is 2048 x (number of slots), which the widest reaction in the set
    decides. An empty slot stays zero. A component that fails to parse also
    ends up zero, but by a different route: create_chemeleon zero-fills a
    molecule it cannot featurise instead of dropping it, so that slot is
    written with a zero row rather than skipped. The printed zero-row count
    below counts rows where no slot was written at all, so it under-reports
    reactions that had every component rejected by RDKit. Rows are never
    dropped and never reordered either way.

    Args:
        rxn_smiles: Reaction SMILES. Reagents in a three-field SMILES are
            folded onto the reactant side before slotting.
        batch_size: Molecules per CheMeleon batch (default 16).

    Returns:
        np.ndarray of shape (len(rxn_smiles), 2048 * n_slots), float32,
        input order.

    Requires: whatever create_chemeleon requires (see
        src/kirby/representations/forward_compat_molecular.py).
    """
    n = len(rxn_smiles)
    if n == 0:
        raise RuntimeError("concat_chemeleon: empty reaction list")

    create_chemeleon = _load_chemeleon()

    try:
        from kirby.datasets.reaction_yields import _parse_rxn_components
    except ImportError as e:
        raise RuntimeError(
            "concat_chemeleon needs _parse_rxn_components from "
            f"kirby.datasets.reaction_yields: {e}"
        )

    components = _parse_rxn_components(_to_double_arrow(rxn_smiles))
    keys = sorted(components[0].keys())

    flat = [str(comp.get(k, '')) for comp in components for k in keys]
    lookup, D = _embed_unique(flat, create_chemeleon, batch_size)

    X = np.zeros((n, len(keys) * D), dtype=np.float32)
    n_zero = 0
    for i, comp in enumerate(components):
        filled = 0
        for j, key in enumerate(keys):
            row = lookup.get(str(comp.get(key, '')))
            if row is not None:
                X[i, j * D:(j + 1) * D] = row
                filled += 1
        if filled == 0:
            n_zero += 1

    if n_zero:
        print(f"  WARNING: concat_chemeleon zero-filled {n_zero}/{n} rows")
    print(f"  concat_chemeleon: {X.shape} ({len(keys)} slots x {D})")
    return X
