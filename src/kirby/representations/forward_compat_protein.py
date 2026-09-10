"""
Forward-Compatibility Protein Representations (DTA target side)

Candidate protein encoders for the c13 forward-compatibility experiment:
one new representation is added to a dataset's existing pool and the hybrid
search is re-run with and without it. Inputs are target amino-acid sequences
from davis, kiba and bindingdb_kd. Every function returns one row per input
sequence, float32, in the input order.

These are frozen feature extractors. No training, no fine-tuning, no labels.

Functions (all take `sequences` and return (len(sequences), D) float32):
- create_protrek_protein_embeddings:      ProTrek 650M contrastive vector (1024D)
- create_ism_protein_embeddings:          ISM t33 650M, mean over residues (1280D)
- create_profluent_e1_protein_embeddings: Profluent E1-600m, mean over residues (1280D)
- create_ankh3_protein_embeddings:        Ankh3-Large T5 encoder, [NLU] (1536D)
- create_prostt5_protein_embeddings:      ProstT5 encoder, mean over residues (1024D)
- create_dplm2_protein_embeddings:        STUB — embedding API not confirmed

Truncation: every function truncates a sequence to a stated residue count and
says so in its docstring. DTA targets run past 2000 residues in bindingdb_kd.

Failure policy: a sequence that fails to encode gets a zero row and the
function prints how many rows were zero-filled. If EVERY sequence fails the
function raises RuntimeError instead of returning an all-zero matrix. A
missing package or missing weights raises ImportError / RuntimeError naming
what to install and from where, which the DTA loader catches, warns on, and
skips.

Not yet dispatched: REP_DISPATCH in tests/find_best_hybrid_dta.py has no
entry for any of these six names, so c13 currently reports all of them as
"not loaded" for the dta domain. Adding the entries is a separate change.
"""

import os
import numpy as np
from typing import List, Optional

# Cached models — populated on first call, reused across calls in a process.
_PROTREK_MODEL = None
_ISM_MODEL = None
_ISM_TOKENIZER = None
_E1_MODEL = None
_E1_BATCH_PREPARER = None
_ANKH3_MODEL = None
_ANKH3_TOKENIZER = None
_FC_PROSTT5_ENCODER = None
_FC_PROSTT5_TOKENIZER = None


# =============================================================================
# Shared helpers (local to this module — no cross-lane imports)
# =============================================================================

def _default_device(device=None):
    """Return device, auto-detecting CUDA if not specified."""
    if device is not None:
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _models_dir() -> str:
    """Directory holding cloned model repos and manual weight downloads."""
    if 'KIRBY_MODELS_DIR' in os.environ:
        return os.environ['KIRBY_MODELS_DIR']
    for path in ['/Volumes/seagate/kirby_models',
                 os.path.expanduser('~/kirby_models'),
                 os.path.expanduser('~/repos')]:
        if os.path.exists(path):
            return path
    return os.path.expanduser('~')


def _stack_rows(rows: List[Optional[np.ndarray]], dim: int, name: str) -> np.ndarray:
    """Stack per-sequence vectors, zero-filling the ones that failed.

    rows holds one entry per input sequence in input order; None marks a
    sequence that raised. Returns (len(rows), dim) float32 and prints the
    zero-filled count so a silently empty representation cannot pass unnoticed.

    Two conditions raise instead of returning a matrix:

    - EVERY sequence failed. The result would be an all-zero matrix, which c13
      would score as a real representation against real labels. RuntimeError,
      which the DTA loader catches and skips, is the honest outcome.
    - A vector came back at a width other than `dim`. Cutting it to `dim` would
      hand back a valid-looking row that is not the model's output, so the
      width disagreement is reported instead of hidden.
    """
    n = len(rows)
    n_zero = sum(1 for r in rows if r is None)
    if n and n_zero == n:
        raise RuntimeError(
            f"{name}: all {n} sequences failed to encode, so every row would be "
            f"zeros. See the per-batch WARNING lines above for the underlying "
            f"error; nothing is returned rather than hand back an all-zero "
            f"matrix that would be scored as a representation.")
    out = np.zeros((n, dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r is None:
            continue
        vec = np.asarray(r, dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim:
            raise RuntimeError(
                f"{name}: sequence {i} produced a {vec.shape[0]}-wide vector "
                f"but the builder expected {dim}. Fix the width the builder "
                f"declares rather than trimming the row.")
        out[i] = vec
    if n_zero:
        print(f"    WARNING: {name}: {n_zero}/{n} sequences failed to "
              f"encode and were zero-filled")
    print(f"  Generated {name} protein embeddings: {out.shape}")
    return out


# =============================================================================
# ProTrek 650M — contrastive sequence/structure/text model
# =============================================================================

def create_protrek_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract ProTrek 650M protein-sequence embeddings.

    Model: ProTrek 650M (Westlake), tri-modal contrastive sequence/structure/
    text. Weights: https://huggingface.co/westlake-repl/ProTrek_650M_UniRef50
    Code: https://github.com/westlake-repl/ProTrek (not a pip package).

    Pooling: none applied here. `model.get_protein_repr([seq])` already returns
    ONE contrastive-projection vector per sequence — the README prints
    torch.Size([1, 1024]) — so a row is that 1024-wide vector as the model
    emits it, not a mean over residues.

    Truncation: 1022 residues. ProTrek's protein tower is esm2_t33_650M_UR50D,
    whose config caps max_position_embeddings at 1026.

    Returns:
        np.ndarray: (len(sequences), 1024) float32.

    SETUP REQUIRED:
        1. Clone the code (it is imported by path, there is no pip package):
           cd $KIRBY_MODELS_DIR   # or ~/repos
           git clone https://github.com/westlake-repl/ProTrek.git
        2. Download the weights into that clone:
           huggingface-cli download westlake-repl/ProTrek_650M_UniRef50 \\
               --repo-type model \\
               --local-dir $KIRBY_MODELS_DIR/ProTrek/weights/ProTrek_650M_UniRef50
        3. pip install transformers torch
        Override either location with KIRBY_PROTREK_DIR / KIRBY_PROTREK_WEIGHTS.

    NOTE: the clone's top-level package is named `model`, and it is put on
    sys.path to import ProTrekTrimodalModel. Importing it can shadow any other
    top-level `model` module in the same process.

    API source: the repo README's own example
    (ProTrekTrimodalModel(**config).eval().to(device); model.get_protein_repr).
    The four config paths and the printed torch.Size([1, 1024]) are read off
    that README. The checkpoint FILENAME differs between the two weight repos
    (ProTrek_650M.pt in one), so the code globs for the .pt rather than
    hardcoding a name.
    """
    import sys
    import glob
    import torch

    global _PROTREK_MODEL

    device = _default_device()

    if _PROTREK_MODEL is None:
        repo_dir = os.environ.get('KIRBY_PROTREK_DIR')
        if repo_dir is None:
            for path in [os.path.join(_models_dir(), 'ProTrek'),
                         os.path.expanduser('~/repos/ProTrek'),
                         os.path.expanduser('~/kirby_models/ProTrek')]:
                if os.path.isdir(os.path.join(path, 'model')):
                    repo_dir = os.path.abspath(path)
                    break
        if repo_dir is None or not os.path.isdir(repo_dir):
            raise RuntimeError(
                "ProTrek code not found. Clone it:\n"
                "  cd $KIRBY_MODELS_DIR  # or ~/repos\n"
                "  git clone https://github.com/westlake-repl/ProTrek.git\n"
                "or set KIRBY_PROTREK_DIR to an existing clone."
            )

        weights_dir = os.environ.get(
            'KIRBY_PROTREK_WEIGHTS',
            os.path.join(repo_dir, 'weights', 'ProTrek_650M_UniRef50'))
        if not os.path.isdir(weights_dir):
            raise RuntimeError(
                f"ProTrek weights not found at {weights_dir}. Download them:\n"
                "  huggingface-cli download westlake-repl/ProTrek_650M_UniRef50 "
                "--repo-type model --local-dir "
                f"{weights_dir}\n"
                "or set KIRBY_PROTREK_WEIGHTS."
            )

        checkpoints = sorted(glob.glob(os.path.join(weights_dir, '*.pt')))
        if not checkpoints:
            raise RuntimeError(
                f"No .pt checkpoint in {weights_dir}. The HuggingFace repo "
                "carries one (ProTrek_650M_UniRef50.pt / ProTrek_650M.pt); "
                "re-run the huggingface-cli download."
            )

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
        except ImportError as e:
            raise ImportError(
                f"Could not import ProTrekTrimodalModel from {repo_dir}: {e}\n"
                "Install its dependencies: pip install torch transformers "
                "(the repo's environment.sh installs requirements.txt)."
            )

        config = {
            "protein_config": os.path.join(weights_dir, "esm2_t33_650M_UR50D"),
            "text_config": os.path.join(
                weights_dir,
                "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
            "structure_config": os.path.join(weights_dir, "foldseek_t30_150M"),
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": checkpoints[0],
        }
        print(f"  Loading ProTrek 650M from {weights_dir}...")
        _PROTREK_MODEL = ProTrekTrimodalModel(**config).eval().to(device)

    model = _PROTREK_MODEL
    n = len(sequences)
    rows: List[Optional[np.ndarray]] = [None] * n

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            batch = [sequences[i].upper().strip()[:1022] for i in batch_idx]
            try:
                emb = model.get_protein_repr(batch)
                emb = emb.detach().float().cpu().numpy()
                for j, i in enumerate(batch_idx):
                    rows[i] = emb[j]
            except Exception as e:
                print(f"    WARNING: ProTrek failed on sequences "
                      f"{batch_idx[0]}-{batch_idx[-1]}: {e}")
            end = min(start + batch_size, n)
            if end < n and end % 50 < batch_size:
                print(f"    ProTrek progress: {end}/{n}")

    return _stack_rows(rows, 1024, 'ProTrek')


# =============================================================================
# ISM — ESM-2 650M architecture fine-tuned on structure tokens
# =============================================================================

def create_ism_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract ISM (Implicit Structure Model) mean-pooled embeddings.

    Model: ISM t33 650M uc30pdb — ESM-2 650M architecture, weights fine-tuned
    to predict structure tokens, sequence-only at inference.
    Weights: https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb — MIT.

    Pooling: mean over residue positions of the final hidden layer, excluding
    the BOS and EOS tokens and any padding. Width 1280.

    Truncation: 1022 residues (config max_position_embeddings is 1026), which
    matches what create_esm2_protein_embeddings already does in protein.py.

    Returns:
        np.ndarray: (len(sequences), 1280) float32.

    Requires: pip install transformers torch. Weights download on first call
    (~2.6 GB), ungated.

    API source: the model card
    (https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb) shows
    AutoModel.from_pretrained / AutoTokenizer.from_pretrained with no
    trust_remote_code; its config.json gives architectures EsmForMaskedLM,
    model_type esm, hidden_size 1280, max_position_embeddings 1026.

    NOTE: the same checkpoint is a candidate in the antibody lane. This is the
    protein-side pooling, written independently; nothing is imported across.
    """
    import torch

    global _ISM_MODEL, _ISM_TOKENIZER

    device = _default_device()
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("ISM requires transformers: pip install transformers")

    model_id = 'jozhang97/ism_t33_650M_uc30pdb'
    if _ISM_MODEL is None:
        print(f"  Loading ISM ({model_id})...")
        _ISM_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _ISM_MODEL = AutoModel.from_pretrained(model_id).to(device).eval()

    model, tokenizer = _ISM_MODEL, _ISM_TOKENIZER
    dim = int(getattr(model.config, 'hidden_size', 1280))
    n = len(sequences)
    rows: List[Optional[np.ndarray]] = [None] * n

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            batch = [sequences[i].upper().strip()[:1022] for i in batch_idx]
            try:
                inputs = tokenizer(batch, return_tensors='pt', padding=True,
                                   truncation=True, max_length=1024)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs)
                hidden = out.last_hidden_state          # (B, L, 1280)
                mask = inputs['attention_mask'].clone()
                mask[:, 0] = 0                          # drop BOS
                for j in range(mask.shape[0]):
                    n_tok = int(inputs['attention_mask'][j].sum().item())
                    if n_tok > 0:
                        mask[j, n_tok - 1] = 0          # drop EOS
                mask = mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = pooled.float().cpu().numpy()
                for j, i in enumerate(batch_idx):
                    rows[i] = pooled[j]
            except Exception as e:
                print(f"    WARNING: ISM failed on sequences "
                      f"{batch_idx[0]}-{batch_idx[-1]}: {e}")
            end = min(start + batch_size, n)
            if end < n and end % 50 < batch_size:
                print(f"    ISM progress: {end}/{n}")

    return _stack_rows(rows, dim, 'ISM')


# =============================================================================
# Profluent E1-600m
# =============================================================================

def create_profluent_e1_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract Profluent E1-600m mean-pooled embeddings.

    Model: Profluent E1-600m, a drop-in ESM replacement trained on 7T tokens.
    Weights: https://huggingface.co/Profluent-Bio/E1-600m
    Code: https://github.com/Profluent-AI/E1 (installed from the clone, it is
    not on PyPI). License carries attribution requirements — see its NOTICE.

    Pooling: mean over residue positions of `outputs.embeddings`, which the
    README states is (B, L, E). Boundary tokens are removed with the repo's
    own masks: `sequence_ids == sequence_ids.max()` keeps the final sequence,
    `~batch_preparer.get_boundary_token_mask(input_ids)` drops BOS/EOS/pad, and
    the two are ANDed. Each sequence is passed alone with no retrieved
    homologs, so the query IS the only sequence in its row.

    Truncation: 2048 residues. The config allows 8192 within-sequence
    positions; 2048 is a memory bound chosen here, not a model limit.

    Returns:
        np.ndarray: (len(sequences), 1280) float32 — hidden_size 1280 from the
        E1-600m config.json.

    SETUP REQUIRED:
        git clone https://github.com/Profluent-AI/E1.git
        cd E1 && pip3 install -e .
        (optional, CUDA 8.0+) pip3 install flash-attn --no-build-isolation

    API source: the repo README
    (https://github.com/Profluent-AI/E1) — E1ForMaskedLM.from_pretrained,
    E1BatchPreparer().get_batch_kwargs(sequences, device=...), the forward call
    taking input_ids / within_seq_position_ids / global_position_ids /
    sequence_ids, outputs.embeddings, and the three selector lines quoted
    above. Width confirmed from the model repo's config.json.

    CAVEAT: the weights are BF16 and the README's example runs under CUDA
    autocast. On CPU the model is cast to float32 here, which is slower and
    not the numerics the authors report.
    """
    import torch

    global _E1_MODEL, _E1_BATCH_PREPARER

    device = _default_device()
    try:
        from E1.modeling import E1ForMaskedLM
        from E1.batch_preparer import E1BatchPreparer
    except ImportError as e:
        raise ImportError(
            f"Profluent E1 package not importable ({e}). Install it from the "
            "repo — it is not on PyPI:\n"
            "  git clone https://github.com/Profluent-AI/E1.git\n"
            "  cd E1 && pip3 install -e ."
        )

    model_id = os.environ.get('KIRBY_E1_MODEL', 'Profluent-Bio/E1-600m')
    if _E1_MODEL is None:
        print(f"  Loading Profluent E1 ({model_id})...")
        _E1_MODEL = E1ForMaskedLM.from_pretrained(model_id).to(device)
        if device == 'cpu':
            _E1_MODEL = _E1_MODEL.float()
        _E1_MODEL = _E1_MODEL.eval()
        _E1_BATCH_PREPARER = E1BatchPreparer()

    model, preparer = _E1_MODEL, _E1_BATCH_PREPARER
    dim = int(getattr(model.config, 'hidden_size', 1280))
    n = len(sequences)
    rows: List[Optional[np.ndarray]] = [None] * n

    use_amp = device.startswith('cuda')
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            batch = [sequences[i].upper().strip()[:2048] for i in batch_idx]
            try:
                kw = preparer.get_batch_kwargs(batch, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    out = model(
                        input_ids=kw['input_ids'],
                        within_seq_position_ids=kw['within_seq_position_ids'],
                        global_position_ids=kw['global_position_ids'],
                        sequence_ids=kw['sequence_ids'],
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                emb = out.embeddings.float()            # (B, L, 1280)
                last_seq = kw['sequence_ids'] == kw['sequence_ids'].max(dim=1)[0][:, None]
                residue = ~preparer.get_boundary_token_mask(kw['input_ids'])
                keep = last_seq & residue
                for j, i in enumerate(batch_idx):
                    sel = emb[j, keep[j]]
                    if sel.shape[0] == 0:
                        print(f"    WARNING: E1 selected 0 residues for "
                              f"sequence {i}")
                        continue
                    rows[i] = sel.mean(dim=0).cpu().numpy()
            except Exception as e:
                print(f"    WARNING: E1 failed on sequences "
                      f"{batch_idx[0]}-{batch_idx[-1]}: {e}")
            end = min(start + batch_size, n)
            if end < n and end % 50 < batch_size:
                print(f"    E1 progress: {end}/{n}")

    return _stack_rows(rows, dim, 'Profluent E1')


# =============================================================================
# Ankh3-Large — T5 encoder, [NLU] prefix
# =============================================================================

def create_ankh3_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract Ankh3-Large T5-encoder mean-pooled embeddings.

    Model: Ankh3-Large, a T5 encoder-decoder protein LM; only the encoder runs
    here. Weights: https://huggingface.co/ElnaggarLab/ankh3-large —
    CC-BY-NC-SA-4.0, non-commercial share-alike.

    Pooling: mean over residue positions of the encoder's last_hidden_state.
    The model card prepends the literal string "[NLU]" to the sequence. That
    string is NOT a single added token in this tokenizer — tokenizer_config.json
    lists only <extra_id_*> as additional special tokens — so the number of
    leading pieces it produces is measured at runtime by tokenizing "[NLU]"
    alone with add_special_tokens=False, and that many leading positions are
    dropped along with the trailing EOS and any padding.

    Truncation: 1024 residues. T5 has no hard positional cap; this is a memory
    bound chosen here.

    Returns:
        np.ndarray: (len(sequences), 1536) float32 — d_model 1536 from the
        model's config.json (48 encoder layers).

    Requires: pip install transformers sentencepiece torch. Weights download on
    first call, ungated.

    API source: the model card
    (https://huggingface.co/ElnaggarLab/ankh3-large) — "you must use
    T5Tokenizer not AutoTokenizer", T5EncoderModel.from_pretrained(ckpt).eval(),
    nlu_sequence = "[NLU]" + sequence, tokenizer(..., add_special_tokens=True,
    return_tensors='pt', is_split_into_words=False).

    CAVEAT: the card notes the [S2S] prefix sometimes gives better embeddings
    than [NLU]. [NLU] is used here because that is the prefix the card's
    embedding example uses and the one the list entry names.
    """
    import torch

    global _ANKH3_MODEL, _ANKH3_TOKENIZER

    device = _default_device()
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        raise ImportError(
            "Ankh3 requires transformers and sentencepiece: "
            "pip install transformers sentencepiece")

    model_id = 'ElnaggarLab/ankh3-large'
    if _ANKH3_MODEL is None:
        print(f"  Loading Ankh3 ({model_id})...")
        _ANKH3_TOKENIZER = T5Tokenizer.from_pretrained(model_id)
        _ANKH3_MODEL = T5EncoderModel.from_pretrained(model_id).to(device).eval()

    model, tokenizer = _ANKH3_MODEL, _ANKH3_TOKENIZER
    dim = int(getattr(model.config, 'd_model', 1536))
    # "[NLU]" always occupies at least one position, so never let a
    # zero-piece measurement leave the prefix inside the mean.
    n_prefix = max(1, len(tokenizer('[NLU]', add_special_tokens=False)['input_ids']))

    n = len(sequences)
    rows: List[Optional[np.ndarray]] = [None] * n

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            batch = ['[NLU]' + sequences[i].upper().strip()[:1024]
                     for i in batch_idx]
            try:
                inputs = tokenizer(batch, add_special_tokens=True,
                                   return_tensors='pt', padding=True,
                                   is_split_into_words=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(**inputs).last_hidden_state    # (B, L, 1536)
                mask = inputs['attention_mask'].clone()
                mask[:, :n_prefix] = 0                        # drop [NLU] pieces
                for j in range(mask.shape[0]):
                    n_tok = int(inputs['attention_mask'][j].sum().item())
                    if n_tok > 0:
                        mask[j, n_tok - 1] = 0                # drop EOS
                mask = mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = pooled.float().cpu().numpy()
                for j, i in enumerate(batch_idx):
                    rows[i] = pooled[j]
            except Exception as e:
                print(f"    WARNING: Ankh3 failed on sequences "
                      f"{batch_idx[0]}-{batch_idx[-1]}: {e}")
            end = min(start + batch_size, n)
            if end < n and end % 50 < batch_size:
                print(f"    Ankh3 progress: {end}/{n}")

    return _stack_rows(rows, dim, 'Ankh3')


# =============================================================================
# ProstT5 — bilingual amino-acid / 3Di T5, encoder only
# =============================================================================

def create_prostt5_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract ProstT5 encoder mean-pooled embeddings.

    Model: ProstT5 (Rostlab), a T5 trained to translate between amino acids and
    Foldseek 3Di in both directions. Weights:
    https://huggingface.co/Rostlab/ProstT5 (~5.5 GB, ungated).

    Pooling: mean over residue positions of the T5 encoder's last_hidden_state,
    dropping the "<AA2fold>" prefix token, the trailing EOS and any padding.
    Width 1024, which the model card shows directly (last_hidden_state[0,1:8]
    for a 7-residue input is "shape (7 x 1024)").

    Preprocessing follows the card: rare and ambiguous residues U, Z, O and B
    are replaced by X, residues are uppercase and separated by single spaces,
    and the direction token "<AA2fold> " is prepended.

    Truncation: 1000 residues, matching the ProstT5 practical limit already
    used by _predict_3di_with_prostt5 in protein.py.

    Returns:
        np.ndarray: (len(sequences), 1024) float32.

    Requires: pip install transformers sentencepiece torch.

    API source: the model card (https://huggingface.co/Rostlab/ProstT5) —
    T5Tokenizer.from_pretrained('Rostlab/ProstT5'),
    T5EncoderModel.from_pretrained("Rostlab/ProstT5"), the "<AA2fold>" /
    "<fold2AA>" direction tokens, the X substitution and the whitespace rule.

    NOTE: protein.py already loads Rostlab/ProstT5 as a
    T5ForConditionalGeneration to predict 3Di tokens for SaProt. This is the
    encoder-only embedding of the same checkpoint, cached in its own global so
    the two do not fight over one handle.
    """
    import re
    import torch

    global _FC_PROSTT5_ENCODER, _FC_PROSTT5_TOKENIZER

    device = _default_device()
    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        raise ImportError(
            "ProstT5 requires transformers and sentencepiece: "
            "pip install transformers sentencepiece")

    if _FC_PROSTT5_ENCODER is None:
        print("  Loading ProstT5 encoder (Rostlab/ProstT5)...")
        _FC_PROSTT5_TOKENIZER = T5Tokenizer.from_pretrained(
            'Rostlab/ProstT5', do_lower_case=False)
        _FC_PROSTT5_ENCODER = T5EncoderModel.from_pretrained(
            'Rostlab/ProstT5').to(device).eval()

    model, tokenizer = _FC_PROSTT5_ENCODER, _FC_PROSTT5_TOKENIZER
    dim = int(getattr(model.config, 'd_model', 1024))
    n = len(sequences)
    rows: List[Optional[np.ndarray]] = [None] * n

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = list(range(start, min(start + batch_size, n)))
            batch = []
            for i in batch_idx:
                seq = re.sub(r'[UZOB]', 'X', sequences[i].upper().strip()[:1000])
                batch.append('<AA2fold> ' + ' '.join(list(seq)))
            try:
                inputs = tokenizer(batch, add_special_tokens=True,
                                   return_tensors='pt', padding='longest')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(**inputs).last_hidden_state    # (B, L, 1024)
                mask = inputs['attention_mask'].clone()
                mask[:, 0] = 0                                # drop <AA2fold>
                for j in range(mask.shape[0]):
                    n_tok = int(inputs['attention_mask'][j].sum().item())
                    if n_tok > 0:
                        mask[j, n_tok - 1] = 0                # drop EOS
                mask = mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = pooled.float().cpu().numpy()
                for j, i in enumerate(batch_idx):
                    rows[i] = pooled[j]
            except Exception as e:
                print(f"    WARNING: ProstT5 failed on sequences "
                      f"{batch_idx[0]}-{batch_idx[-1]}: {e}")
            end = min(start + batch_size, n)
            if end < n and end % 50 < batch_size:
                print(f"    ProstT5 progress: {end}/{n}")

    return _stack_rows(rows, dim, 'ProstT5')


# =============================================================================
# DPLM-2 650M — STUB, embedding API not confirmed
# =============================================================================

def create_dplm2_protein_embeddings(
    sequences: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """STUB — DPLM-2 650M embeddings. UNVERIFIED, raises NotImplementedError.

    Model: DPLM-2 650M, multimodal discrete-diffusion protein LM over a joint
    amino-acid and structure-token vocabulary.
    Weights: https://huggingface.co/airkingbd/dplm2_650m
    Code: https://github.com/bytedance/dplm

    Intended pooling (from the list entry, NOT implemented): mean over residues
    of the 1280-wide final hidden layer.

    WHAT WAS CONFIRMED by opening the pages:
      - config.json: architectures ["EsmForDPLM2"], model_type "esm",
        hidden_size 1280, vocab_size 8229.
      - tokenizer_config.json: tokenizer_class "DPLM2Tokenizer", with SEPARATE
        per-modality specials — <cls_aa> 0, <eos_aa> 2, <mask_aa> 32 against
        <cls_struct> 33, <eos_struct> 34, <mask_struct> 8229. So an amino-acid
        string has to be tokenised under the aa modality, and getting that
        wrong produces a silently valid but meaningless row.
      - The model card's two loading snippets:
        `from transformers import AutoTokenizer, EsmForDPLM2` and
        `from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel`.

    WHAT COULD NOT BE CONFIRMED, which is why this is a stub:
      - `EsmForDPLM2` is not a class in stock transformers, and the config
        carries NO auto_map, so the card's transformers snippet cannot load
        without the byprot package. The class therefore comes from a source not
        opened here.
      - No page shows how to get hidden states out of either object. The
        bytedance/dplm README lists "Representation learning of DPLM-2" as an
        unfinished TODO and documents generation only (co-generation, folding,
        inverse folding, scaffolding).
      - How to tokenise a bare amino-acid sequence under the aa modality — the
        exact call and whether a modality tag is prepended — is not shown
        anywhere that was read.

    TO IMPLEMENT: install byprot
    (git clone --recursive https://github.com/bytedance/dplm.git; cd dplm;
    bash scripts/install.sh), then read byprot/models/dplm2 and
    byprot/datamodules/dataset/tokenized_protein.py to find the aa-modality
    tokenisation and the forward call that returns hidden states. Do not guess
    them from the ESM-2 API — the vocabulary is 8229 tokens, not 33.

    Returns:
        Never returns. Raises NotImplementedError.
    """
    raise NotImplementedError(
        "dplm2 protein embeddings are not implemented: airkingbd/dplm2_650m "
        "has hidden_size 1280 and a DPLM2Tokenizer with separate <cls_aa>/"
        "<cls_struct> modalities, but no opened page shows how to extract "
        "hidden states — bytedance/dplm lists representation learning as an "
        "open TODO and EsmForDPLM2 is not in stock transformers (no auto_map "
        "in the config). Implement it against the byprot source, not from the "
        "ESM-2 API."
    )
