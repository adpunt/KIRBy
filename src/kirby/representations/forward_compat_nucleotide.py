"""
Forward-compatibility nucleotide representations (c13 candidates).

One function per newly published DNA model. Each turns a list of DNA strings into
ONE ROW PER SEQUENCE of float32 features, frozen — no fine-tuning, no label use.

Contract every function here keeps:
  f(sequences: List[str]) -> np.ndarray of shape (len(sequences), D), float32,
  rows in the SAME ORDER as the input list. A sequence that fails to encode gets a
  zero row and a printed count of how many rows were zero-filled; nothing is dropped
  and nothing is reordered.

Missing packages and missing weights raise ImportError or RuntimeError naming what
to install and from where, so the nucleotide loader can print a warning and move on.
Nothing here imports torch or transformers at module import time.

Models (9), width and pooling:
  nucel               512D   mean over tokens        FreakingPotato/NucEL
  generator_euk_1_2b  2048D  mean over tokens        GenerTeam/GENERator-eukaryote-1.2b-base
  omni_dna_116m       768D   mean over tokens        zehui127/Omni-DNA-116M
  metagene_1          4096D  mean over tokens        metagene-ai/METAGENE-1
  evo2_1b             read at runtime, mean over tokens   arcinstitute/evo2_1b_base
  janusdna            STUB — no documented load call found
  moderngena_large    1024D  mean over tokens        AIRI-Institute/moderngena-large
  borzoi              1536D  mean over the 32 bp bins covering the window
  plantcad2_large     1536D  strand-averaged, then mean over positions

Statuses: janusdna is a labelled STUB. The other eight were written against a model
card, config.json or source file opened on 2026-09-10; each docstring names the page.
"""

from typing import List

import numpy as np

from .nucleotide import _hf_embed


# Bespoke loaders cache their model here, keyed by whatever identifies the load.
_FC_MODEL_CACHE: dict = {}

_DNA_OK = set('ACGTN')


def _clean_dna(seq: str) -> str:
    """Uppercase, drop whitespace, map anything outside A/C/G/T to N."""
    s = ''.join(str(seq).upper().split())
    return ''.join(c if c in _DNA_OK else 'N' for c in s)


def _report_zero_rows(name: str, n_zero: int, n_total: int) -> None:
    if n_zero:
        print(f"  WARNING {name}: {n_zero}/{n_total} rows zero-filled "
              f"(encode failed); rows kept in input order.")


# =============================================================================
# NucEL — plain HuggingFace AutoModel, uses the shared pipeline
# =============================================================================

def _embed_nucel(sequences: List[str]) -> np.ndarray:
    """NucEL single-nucleotide ELECTRA discriminator, 93M. 512 features.

    Model: FreakingPotato/NucEL (Apache-2.0), a ModernBertModel with
    hidden_size 512 and model_max_length 2048 per its config.json.
    Pooling: mean over non-padding tokens. The paper also uses the [CLS]
    token; mean pooling is what every other nucleotide rep in this repo does,
    so mean is what is used here.

    API confirmed on the model card https://huggingface.co/FreakingPotato/NucEL:
        tokenizer = AutoTokenizer.from_pretrained("FreakingPotato/NucEL",
                                                  trust_remote_code=True)
        model = AutoModel.from_pretrained("FreakingPotato/NucEL").eval()
        outputs.last_hidden_state  # (1, L, 512)

    The tokenizer is a custom class (NucEL_Tokenizer) reached through
    trust_remote_code. On transformers 5.4+ the card says AutoTokenizer
    auto-discovery for ModernBERT is blocked and the class must be imported from
    a snapshot_download instead; if that is the installed version this call will
    fail at the tokenizer and the rep is skipped.

    Requires: pip install torch transformers
    """
    return _hf_embed(
        sequences,
        model_id='FreakingPotato/NucEL',
        batch_size=8, max_length=1024,
        trust_remote_code=True,
        auto_class='AutoModel',
    )


# =============================================================================
# GENERator-eukaryote-1.2b-base
# =============================================================================

def _embed_generator_euk_1_2b(sequences: List[str]) -> np.ndarray:
    """GENERator-eukaryote-1.2b-base. 2048 features.

    Model: GenerTeam/GENERator-eukaryote-1.2b-base (CC BY 4.0). Llama-style
    causal decoder over non-overlapping 6-mer tokens; config.json gives
    hidden_size 2048, 26 layers, vocab 4128, max_position_embeddings 16384.
    Pooling: mean of the last hidden layer over non-padding tokens.

    API confirmed on the model card
    https://huggingface.co/GenerTeam/GENERator-eukaryote-1.2b-base:
        model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(..., trust_remote_code=True)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
    and the card states mean pooling with an attention mask as one of its two
    published pooling options.

    The tokenizer is 6-mer, and the card requires input lengths that are a
    multiple of 6, so each sequence is trimmed at the 3' end to the nearest
    multiple of 6 before tokenising. The card also loads with
    attn_implementation="flash_attention_2"; that is not used here, so the model
    runs with whatever attention implementation transformers picks and no
    flash-attn install is needed.

    Requires: pip install torch transformers accelerate
    Weights: ~5 GB, downloaded from HuggingFace on first call.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = 'GenerTeam/GENERator-eukaryote-1.2b-base'
    if model_id not in _FC_MODEL_CACHE:
        print(f"  Loading {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_id}: {e}\n"
                "Needs: pip install torch transformers accelerate\n"
                "Weights come from https://huggingface.co/"
                "GenerTeam/GENERator-eukaryote-1.2b-base (CC BY 4.0)."
            )
        # The model card uses padding_side='left' for GENERATION and
        # padding_side='right' for EMBEDDING EXTRACTION. This is embedding
        # extraction. Under left padding this causal decoder gives the real
        # tokens rotary positions shifted by however much padding the batch
        # happened to need, so one sequence would embed differently depending on
        # which other sequences shared its batch.
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[model_id] = (model, tokenizer, device)
        print(f"  Loaded on {device}")

    model, tokenizer, device = _FC_MODEL_CACHE[model_id]

    cleaned = []
    for s in sequences:
        s = _clean_dna(s)
        if len(s) < 6:
            s = s + 'N' * (6 - len(s))       # the 6-mer tokenizer needs 6 bases
        else:
            s = s[:len(s) - (len(s) % 6)]
        cleaned.append(s)

    rows, n_zero = [], 0
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            try:
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=2048, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out = model(input_ids=input_ids, attention_mask=mask,
                            output_hidden_states=True)
                hidden = out.hidden_states[-1]
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                rows.append(pooled.float().cpu().numpy())
            except Exception as e:
                print(f"  {model_id}: batch at index {i} failed ({e}); zero rows.")
                rows.append(np.zeros((len(batch), model.config.hidden_size),
                                     dtype=np.float32))
                n_zero += len(batch)

    if sequences and n_zero == len(sequences):
        raise RuntimeError(
            "generator_euk_1_2b failed on every batch, so every row would be zeros. "
            "An all-zero feature block reads as a real representation to c13, "
            "so this raises instead and the loader skips the rep.")

    result = np.vstack(rows).astype(np.float32)
    _report_zero_rows('generator_euk_1_2b', n_zero, len(sequences))
    print(f"  generator_euk_1_2b: {result.shape}")
    return result


# =============================================================================
# Omni-DNA 116M
# =============================================================================

def _embed_omni_dna_116m(sequences: List[str]) -> np.ndarray:
    """Omni-DNA 116M. 768 features.

    Model: zehui127/Omni-DNA-116M (CC BY 4.0). OLMo autoregressive transformer,
    BPE vocabulary 4096; its config.json gives d_model 768, 12 layers, 12 heads
    and max_sequence_length 250, so tokenised input is truncated at 250 tokens.
    Pooling: mean of the last hidden layer over non-padding tokens.

    API confirmed on the model card https://huggingface.co/zehui127/Omni-DNA-116M
    (AutoTokenizer + AutoModelForCausalLM, trust_remote_code=True) and in the
    repo's own modeling_olmo.py, whose OLMoForCausalLM.forward takes
    output_hidden_states and returns CausalLMOutputWithPast(hidden_states=...),
    so hidden_states[-1] is (batch, tokens, 768).

    SETUP REQUIRED: modeling_olmo.py imports `olmo.config` and `olmo.model`, so
    the OLMo package must be installed:
        pip install ai2-olmo
    Requires: pip install torch transformers ai2-olmo
    """
    import torch

    try:
        import olmo  # noqa: F401 — modeling_olmo.py imports olmo.config/olmo.model
    except ImportError:
        raise ImportError(
            "Omni-DNA's remote code (modeling_olmo.py) imports the OLMo package.\n"
            "Install with:\n  pip install ai2-olmo\n"
            "Weights: https://huggingface.co/zehui127/Omni-DNA-116M (CC BY 4.0)."
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = 'zehui127/Omni-DNA-116M'
    if model_id not in _FC_MODEL_CACHE:
        print(f"  Loading {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_id}: {e}\n"
                "Needs: pip install torch transformers ai2-olmo\n"
                "Weights: https://huggingface.co/zehui127/Omni-DNA-116M"
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[model_id] = (model, tokenizer, device)
        print(f"  Loaded on {device}")

    model, tokenizer, device = _FC_MODEL_CACHE[model_id]
    cleaned = [_clean_dna(s) or 'N' for s in sequences]

    rows, n_zero = [], 0
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            try:
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=250, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out = model(input_ids=input_ids, attention_mask=mask,
                            output_hidden_states=True)
                hidden = out.hidden_states[-1]
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                rows.append(pooled.float().cpu().numpy())
            except Exception as e:
                print(f"  {model_id}: batch at index {i} failed ({e}); zero rows.")
                rows.append(np.zeros((len(batch), 768), dtype=np.float32))
                n_zero += len(batch)

    if sequences and n_zero == len(sequences):
        raise RuntimeError(
            "omni_dna_116m failed on every batch, so every row would be zeros. "
            "An all-zero feature block reads as a real representation to c13, "
            "so this raises instead and the loader skips the rep.")

    result = np.vstack(rows).astype(np.float32)
    _report_zero_rows('omni_dna_116m', n_zero, len(sequences))
    print(f"  omni_dna_116m: {result.shape}")
    return result


# =============================================================================
# METAGENE-1
# =============================================================================

def _embed_metagene_1(sequences: List[str]) -> np.ndarray:
    """METAGENE-1 metagenomic foundation model. 4096 features.

    Model: metagene-ai/METAGENE-1 (Apache-2.0, not gated). Its config.json is a
    plain LlamaForCausalLM: hidden_size 4096, 32 layers, byte-pair vocab 1024,
    max_position_embeddings 512 — so tokenised input is truncated at 512 tokens.
    Pooling: mean of last_hidden_state over non-padding tokens. PROVENANCE
    UNCONFIRMED — the model card shows generation and a feature-extraction
    pipeline call but names no pooling and shows no embedding example, so mean
    pooling here is this repo's own choice (what every other nucleotide rep
    does), not something the cited page documents.

    API confirmed on the model card https://huggingface.co/metagene-ai/METAGENE-1:
        tokenizer = AutoTokenizer.from_pretrained("metagene-ai/METAGENE-1")
        model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    The card shows generation, not embedding. Here AutoModel is used instead of
    AutoModelForCausalLM so the trunk returns last_hidden_state directly; that is
    the standard llama mapping in transformers, not a custom class.

    COST: 7B parameters. The six safetensors shards in the HuggingFace repo total
    25.9 GB and config.json gives torch_dtype float32, so 25.9 GB is what is
    downloaded on first call; the cast to bfloat16 at load time brings the
    resident copy to about 14 GB. Practically this wants a GPU.

    Requires: pip install torch transformers accelerate
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_id = 'metagene-ai/METAGENE-1'
    if model_id not in _FC_MODEL_CACHE:
        print(f"  Loading {model_id} (7B, bfloat16, ~14 GB)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_id}: {e}\n"
                "Needs: pip install torch transformers accelerate, and about "
                "14 GB of disk plus the same in RAM or VRAM.\n"
                "Weights: https://huggingface.co/metagene-ai/METAGENE-1 (Apache-2.0)."
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[model_id] = (model, tokenizer, device)
        print(f"  Loaded on {device}")

    model, tokenizer, device = _FC_MODEL_CACHE[model_id]
    cleaned = [_clean_dna(s) or 'N' for s in sequences]

    rows, n_zero = [], 0
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            try:
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out = model(input_ids=input_ids, attention_mask=mask)
                hidden = out.last_hidden_state.float()
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                rows.append(pooled.cpu().numpy())
            except Exception as e:
                print(f"  {model_id}: batch at index {i} failed ({e}); zero rows.")
                rows.append(np.zeros((len(batch), 4096), dtype=np.float32))
                n_zero += len(batch)

    if sequences and n_zero == len(sequences):
        raise RuntimeError(
            "metagene_1 failed on every batch, so every row would be zeros. "
            "An all-zero feature block reads as a real representation to c13, "
            "so this raises instead and the loader skips the rep.")

    result = np.vstack(rows).astype(np.float32)
    _report_zero_rows('metagene_1', n_zero, len(sequences))
    print(f"  metagene_1: {result.shape}")
    return result


# =============================================================================
# Evo 2 1B base
# =============================================================================

def _embed_evo2_1b(sequences: List[str]) -> np.ndarray:
    """Evo 2 1B base, StripedHyena 2, single-nucleotide tokens.

    Model: arcinstitute/evo2_1b_base (Apache-2.0), loaded through the authors'
    `evo2` package rather than transformers — the HuggingFace repo holds one
    evo2_1b_base.pt checkpoint and no config for AutoModel.
    Width: whatever the chosen layer emits; it is read off the first forward pass
    and not hardcoded here.
    Pooling: mean over the sequence's tokens, one sequence per forward pass (no
    padding, so no mask is needed).

    API confirmed in https://raw.githubusercontent.com/ArcInstitute/evo2/main/README.md:
        from evo2 import Evo2
        evo2_model = Evo2('evo2_7b')
        input_ids = torch.tensor(evo2_model.tokenizer.tokenize(sequence),
                                 dtype=torch.int).unsqueeze(0).to('cuda:0')
        outputs, embeddings = evo2_model(input_ids, return_embeddings=True,
                                         layer_names=[layer_name])
    The README's layer_name 'blocks.28.mlp.l3' is for the 7B checkpoint. Rather
    than guess the equivalent index for 1B, this function lists the loaded
    model's modules and takes the DEEPEST module matching blocks.<N>.mlp.l3.

    SETUP REQUIRED (Linux + NVIDIA GPU only, per the Evo 2 README):
        conda install -c nvidia cuda-nvcc cuda-cudart-dev
        conda install -c conda-forge transformer-engine-torch=2.3.0
        pip install flash-attn==2.8.0.post2 --no-build-isolation
        pip install evo2
    The README states the 1B checkpoint needs FP8 through Transformer Engine,
    which needs an NVIDIA Hopper GPU. On anything else this raises RuntimeError
    and the rep is skipped.
    """
    import re

    import torch

    try:
        from evo2 import Evo2
    except ImportError:
        raise ImportError(
            "Evo 2 needs the authors' evo2 package (Linux + NVIDIA GPU).\n"
            "Install with:\n"
            "  conda install -c nvidia cuda-nvcc cuda-cudart-dev\n"
            "  conda install -c conda-forge transformer-engine-torch=2.3.0\n"
            "  pip install flash-attn==2.8.0.post2 --no-build-isolation\n"
            "  pip install evo2\n"
            "Weights: https://huggingface.co/arcinstitute/evo2_1b_base (Apache-2.0)."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Evo 2 requires CUDA; no GPU is visible. The Evo 2 README also states "
            "the 1B checkpoint uses FP8 through Transformer Engine, which needs an "
            "NVIDIA Hopper GPU. Skip this rep on CPU nodes."
        )

    key = 'evo2_1b_base'
    if key not in _FC_MODEL_CACHE:
        print("  Loading evo2_1b_base...")
        try:
            model = Evo2('evo2_1b_base')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load evo2_1b_base: {e}\n"
                "The checkpoint is downloaded from "
                "https://huggingface.co/arcinstitute/evo2_1b_base (Apache-2.0); "
                "FP8 through transformer-engine on a Hopper GPU is required."
            )
        inner = getattr(model, 'model', model)
        pattern = re.compile(r'blocks\.(\d+)\.mlp\.l3$')
        candidates = []
        for name, _ in inner.named_modules():
            m = pattern.search(name)
            if m:
                candidates.append((int(m.group(1)), name))
        if not candidates:
            raise RuntimeError(
                "evo2_1b_base loaded but no module named blocks.<N>.mlp.l3 was "
                "found, so there is no layer name to ask for embeddings from. "
                "The Evo 2 README's example uses that name for evo2_7b; check the "
                "1B module names before using this rep."
            )
        layer_name = max(candidates)[1]
        print(f"  Embedding layer: {layer_name}")
        _FC_MODEL_CACHE[key] = (model, layer_name)

    model, layer_name = _FC_MODEL_CACHE[key]
    cleaned = [_clean_dna(s) or 'N' for s in sequences]

    vectors, n_zero, dim = [], 0, None
    with torch.no_grad():
        for idx, seq in enumerate(cleaned):
            try:
                input_ids = torch.tensor(
                    model.tokenizer.tokenize(seq), dtype=torch.int
                ).unsqueeze(0).to('cuda:0')
                _, embeddings = model(input_ids, return_embeddings=True,
                                      layer_names=[layer_name])
                emb = embeddings[layer_name].float()   # (1, tokens, D)
                vec = emb.mean(dim=1).squeeze(0).cpu().numpy()
                dim = vec.shape[0] if dim is None else dim
                vectors.append(vec)
            except Exception as e:
                print(f"  evo2_1b: sequence {idx} failed ({e}); zero row.")
                vectors.append(None)
                n_zero += 1

    if dim is None:
        raise RuntimeError(
            "evo2_1b produced no embedding for any sequence; nothing to return.")

    result = np.vstack([
        v if v is not None else np.zeros(dim, dtype=np.float32) for v in vectors
    ]).astype(np.float32)
    _report_zero_rows('evo2_1b', n_zero, len(sequences))
    print(f"  evo2_1b: {result.shape}")
    return result


# =============================================================================
# JanusDNA — STUB
# =============================================================================

def _embed_janusdna(sequences: List[str]) -> np.ndarray:
    """JanusDNA (Mamba-Attention mixture-of-experts). STUB — API NOT CONFIRMED.

    This function raises NotImplementedError on purpose. Do not replace the body
    with a guess.

    What IS confirmed (GitHub Qihao-Duan/JanusDNA, read 2026-09-10):
      - Pretraining weights live at Harvard Dataverse
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HDT0RN
        as a manual download, not on HuggingFace.
      - Setup installs mamba-ssm from a prebuilt CUDA wheel
        (mamba_ssm-2.2.4+cu12torch2.5..., Linux + CUDA only).
      - Training scripts name 32-dim, 72-dim and 144-dim variants, matching the
        32-144 feature width in the candidate list.

    What could NOT be confirmed: there is no from_pretrained call, no model class
    name and no embedding-extraction example anywhere in the README, so there is
    no way to write the load without inventing it. A checkpoint file also has to
    be fetched by hand from Dataverse first.

    To finish this: download a checkpoint, read the training script that builds
    the model, and mirror how it instantiates the model and reads its final
    hidden states (mean-pooled over tokens is the intended pooling).
    """
    raise NotImplementedError(
        "janusdna is a stub: weights are a manual Harvard Dataverse download "
        "(doi:10.7910/DVN/HDT0RN) and the JanusDNA README documents no model "
        "class, no from_pretrained call and no embedding example, so the loading "
        "code could not be verified. Skipping rather than guessing."
    )


# =============================================================================
# ModernGENA-large
# =============================================================================

def _embed_moderngena_large(sequences: List[str]) -> np.ndarray:
    """ModernGENA-large (377M), AIRI. 1024 features.

    Model: AIRI-Institute/moderngena-large. Its config.json is
    ModernBertForMaskedLM with hidden_size 1024 and max_position_embeddings 1024,
    so tokenised input is truncated at 1024 tokens.
    Pooling: mean of last_hidden_state over non-padding tokens.

    API confirmed on the model card https://huggingface.co/AIRI-Institute/moderngena-large:
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t")
        model = AutoModel.from_pretrained("AIRI-Institute/moderngena-large",
                                          trust_remote_code=True,
                                          attn_implementation="flash_attention_2")
    Note the TOKENIZER COMES FROM A DIFFERENT REPO — the 32k BPE vocabulary of
    gena-lm-bert-base-t2t — which is why this cannot use the shared _hf_embed
    helper, since that loads the tokenizer from the same id as the model.

    flash_attention_2 is tried first because that is what the card shows; if
    flash-attn is not installed the load is retried without it, and only a
    failure of both raises.

    Requires: pip install torch transformers  (optionally flash-attn)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_id = 'AIRI-Institute/moderngena-large'
    tokenizer_id = 'AIRI-Institute/gena-lm-bert-base-t2t'
    if model_id not in _FC_MODEL_CACHE:
        print(f"  Loading {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the ModernGENA tokenizer {tokenizer_id}: {e}\n"
                "ModernGENA uses the GENA-LM 32k BPE vocabulary from that repo."
            )
        model = None
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True,
                attn_implementation='flash_attention_2')
        except Exception as e_flash:
            print(f"  flash_attention_2 unavailable ({e_flash}); "
                  f"retrying with the default attention implementation.")
            try:
                model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {model_id}: {e}\n"
                    "Needs: pip install torch transformers (ModernBERT support "
                    "arrived in transformers 4.48). The model card loads it with "
                    "attn_implementation='flash_attention_2', which needs "
                    "pip install flash-attn on a CUDA machine.\n"
                    "Weights: https://huggingface.co/AIRI-Institute/moderngena-large"
                )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[model_id] = (model, tokenizer, device)
        print(f"  Loaded on {device}")

    model, tokenizer, device = _FC_MODEL_CACHE[model_id]
    cleaned = [_clean_dna(s) or 'N' for s in sequences]

    rows, n_zero = [], 0
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i + batch_size]
            try:
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=1024, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out = model(input_ids=input_ids, attention_mask=mask)
                hidden = getattr(out, 'last_hidden_state', None)
                if hidden is None:
                    hidden = out[0]
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                rows.append(pooled.float().cpu().numpy())
            except Exception as e:
                print(f"  {model_id}: batch at index {i} failed ({e}); zero rows.")
                rows.append(np.zeros((len(batch), 1024), dtype=np.float32))
                n_zero += len(batch)

    if sequences and n_zero == len(sequences):
        raise RuntimeError(
            "moderngena_large failed on every batch, so every row would be zeros. "
            "An all-zero feature block reads as a real representation to c13, "
            "so this raises instead and the loader skips the rep.")

    result = np.vstack(rows).astype(np.float32)
    _report_zero_rows('moderngena_large', n_zero, len(sequences))
    print(f"  moderngena_large: {result.shape}")
    return result


# =============================================================================
# Borzoi
# =============================================================================

def _embed_borzoi(sequences: List[str]) -> np.ndarray:
    """Borzoi sequence-to-function trunk. 1536 features.

    *** READ THIS BEFORE READING ANY RESULT FROM THIS REP ***
    Borzoi is the one SUPERVISED model in this set. It was trained to predict
    genomic coverage tracks, and the assay families among its training targets
    include H3K27me3, H3K4me3, H3K9ac and H3K9me3 — the same histone marks that
    four of the six nucleotide datasets here use as their labels. So a positive
    c13 delta from borzoi is not evidence that the hybrid search absorbed a new
    model; it is at least as consistent with the label information already having
    been baked into the weights during Borzoi's own training. Report any borzoi
    number with that overlap stated next to it, or leave the model out.

    Model: johahi/borzoi-replicate-0 (MIT), through pip install borzoi-pytorch.
    API confirmed in the README (https://github.com/johahi/borzoi-pytorch) and in
    the package source pytorch_borzoi_model.py, read 2026-09-10:
        from borzoi_pytorch import Borzoi
        borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
        borzoi.get_embs_after_crop(x)  # x is (N, 4, L); returns (N, dim, bins)
    From the same source: config.dim is 1536, bins_to_return is 6144 and
    return_center_bins_only is True, so the trunk output is 6144 bins covering the
    central 196,608 bp of the input at 32 bp per bin.

    INPUT HANDLING. Borzoi reads a fixed 524,288 bp window and these datasets'
    sequences are a few hundred bp. Each sequence is one-hot encoded in A,C,G,T
    order (N and any other character become an all-zero column) and placed at the
    CENTRE of an otherwise all-zero 524,288 bp window.
    POOLING: mean over only those 32 bp bins that overlap the inserted sequence
    (at least one bin), never over the zero-padding.

    COST: one forward pass per sequence over a 524k bp input. This wants a GPU;
    on CPU it is minutes per sequence.

    Requires: pip install borzoi-pytorch
    Weights: downloaded from https://huggingface.co/johahi/borzoi-replicate-0
    """
    import torch

    try:
        from borzoi_pytorch import Borzoi
    except ImportError:
        raise ImportError(
            "Borzoi needs the borzoi-pytorch package.\n"
            "Install with:\n  pip install borzoi-pytorch\n"
            "Weights are pulled from https://huggingface.co/johahi/borzoi-replicate-0 "
            "(MIT) on first use."
        )

    repo_id = 'johahi/borzoi-replicate-0'
    if repo_id not in _FC_MODEL_CACHE:
        print(f"  Loading {repo_id}...")
        try:
            model = Borzoi.from_pretrained(repo_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {repo_id}: {e}\n"
                "Needs: pip install borzoi-pytorch, and the checkpoint from "
                "https://huggingface.co/johahi/borzoi-replicate-0 (MIT)."
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[repo_id] = (model, device)
        print(f"  Loaded on {device}")

    model, device = _FC_MODEL_CACHE[repo_id]

    SEQ_LEN = 524288          # Borzoi's input window, base pairs
    BIN = 32                  # base pairs per trunk bin
    N_BINS_OUT = 6144         # central bins the trunk returns
    CROP_BP = N_BINS_OUT * BIN                 # 196,608 bp kept
    CROP_START = (SEQ_LEN - CROP_BP) // 2      # 163,840 bp offset of bin 0

    base_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    vectors, n_zero, dim = [], 0, None
    with torch.no_grad():
        for idx, raw in enumerate(sequences):
            try:
                seq = _clean_dna(raw)[:SEQ_LEN]
                if not seq:
                    raise ValueError("empty sequence")
                width = len(seq)
                start = (SEQ_LEN - width) // 2

                x = np.zeros((4, SEQ_LEN), dtype=np.float32)
                for pos, ch in enumerate(seq):
                    j = base_index.get(ch)
                    if j is not None:
                        x[j, start + pos] = 1.0

                tensor = torch.from_numpy(x).unsqueeze(0).to(device)
                embs = model.get_embs_after_crop(tensor)   # (1, dim, bins)

                n_bins = embs.shape[-1]
                lo = max(0, (start - CROP_START) // BIN)
                hi = min(n_bins, -(-(start + width - CROP_START) // BIN))
                if hi <= lo:
                    lo, hi = 0, n_bins
                vec = embs[0, :, lo:hi].float().mean(dim=1).cpu().numpy()
                dim = vec.shape[0] if dim is None else dim
                vectors.append(vec)
            except Exception as e:
                print(f"  borzoi: sequence {idx} failed ({e}); zero row.")
                vectors.append(None)
                n_zero += 1

    if dim is None:
        raise RuntimeError(
            "borzoi produced no embedding for any sequence; nothing to return.")

    result = np.vstack([
        v if v is not None else np.zeros(dim, dtype=np.float32) for v in vectors
    ]).astype(np.float32)
    _report_zero_rows('borzoi', n_zero, len(sequences))
    print(f"  borzoi: {result.shape}")
    return result


# =============================================================================
# PlantCAD2-Large
# =============================================================================

def _embed_plantcad2_large(sequences: List[str]) -> np.ndarray:
    """PlantCAD2-Large (676M, Mamba2/Caduceus, 65 angiosperm genomes). 1536 features.

    Model: kuleshov-group/PlantCAD2-Large-l48-d1536 (apache-2.0 on the repo's
    cardData). Its config.json gives d_model 1536, n_layer 48, rcps True and an
    auto_map with AutoModelForMaskedLM -> modeling_caduceus.CaduceusForMaskedLM.

    API confirmed in the PlantCaduceus README
    (https://github.com/kuleshov-group/PlantCaduceus), which covers both the
    PlantCaduceus and PlantCAD2 checkpoints:
        model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        hidden_size = embeddings.shape[-1] // 2
        forward = embeddings[..., 0:hidden_size]
        reverse = embeddings[..., hidden_size:][:, ::-1, :]
        averaged_embeddings = (forward + reverse) / 2
    Because the model is reverse-complement equivariant, the last hidden state is
    2 x 1536 wide and the two halves are averaged exactly as above, giving 1536.

    POOLING: the strand-average above, then the mean over the sequence's own token
    positions — padding positions are excluded.

    CONTEXT. The README lists PlantCAD2's max input as 8192 bp and its scoring
    script says PlantCAD2 enforces a minimum context of 2048 bp, so a short window
    is padded with N up to 2048 bp (padding excluded from the pooling) and a long
    one is truncated at 8192 bp.

    NOTE: every published number for this model is on plant tasks; the nucleotide
    datasets here are human.

    SETUP REQUIRED (CUDA):
        pip install mamba-ssm causal-conv1d
    Requires: pip install torch transformers mamba-ssm causal-conv1d
    """
    import torch

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        raise ImportError(
            "PlantCAD2 is a Caduceus/Mamba2 model and needs the mamba-ssm package "
            "(the pip package, not the conda tool).\n"
            "Install with:\n  pip install mamba-ssm causal-conv1d\n"
            "mamba-ssm requires CUDA; skip this rep on CPU-only nodes.\n"
            "Weights: https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536"
        )

    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_id = 'kuleshov-group/PlantCAD2-Large-l48-d1536'
    if model_id not in _FC_MODEL_CACHE:
        print(f"  Loading {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_id}: {e}\n"
                "Needs: pip install torch transformers mamba-ssm causal-conv1d "
                "on a CUDA machine.\n"
                "Weights: https://huggingface.co/kuleshov-group/"
                "PlantCAD2-Large-l48-d1536"
            )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        _FC_MODEL_CACHE[model_id] = (model, tokenizer, device)
        print(f"  Loaded on {device}")

    model, tokenizer, device = _FC_MODEL_CACHE[model_id]

    MIN_CTX, MAX_CTX = 2048, 8192
    prepared, real_len = [], []
    for s in sequences:
        seq = _clean_dna(s)[:MAX_CTX] or 'N'
        real_len.append(len(seq))
        if len(seq) < MIN_CTX:                       # pad, centred, with N
            pad = MIN_CTX - len(seq)
            left = pad // 2
            seq = 'N' * left + seq + 'N' * (pad - left)
        prepared.append(seq)

    rows, n_zero = [], 0
    with torch.no_grad():
        for idx, seq in enumerate(prepared):
            try:
                enc = tokenizer.encode_plus(
                    seq, return_tensors='pt',
                    return_attention_mask=False, return_token_type_ids=False)
                input_ids = enc['input_ids'].to(device)
                out = model(input_ids=input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1].to(torch.float32).cpu().numpy()

                half = hidden.shape[-1] // 2
                forward = hidden[..., 0:half]
                reverse = hidden[..., half:][:, ::-1, :]
                strand_avg = (forward + reverse) / 2.0      # (1, tokens, 1536)

                tokens = strand_avg.shape[1]
                w = real_len[idx]
                lo = max(0, (tokens - w) // 2)
                hi = min(tokens, lo + w)
                if hi <= lo:
                    lo, hi = 0, tokens
                rows.append(strand_avg[0, lo:hi, :].mean(axis=0))
            except Exception as e:
                print(f"  plantcad2_large: sequence {idx} failed ({e}); zero row.")
                rows.append(None)
                n_zero += 1

    dim = next((r.shape[0] for r in rows if r is not None), None)
    if dim is None:
        raise RuntimeError(
            "plantcad2_large produced no embedding for any sequence; "
            "nothing to return.")

    result = np.vstack([
        r if r is not None else np.zeros(dim, dtype=np.float32) for r in rows
    ]).astype(np.float32)
    _report_zero_rows('plantcad2_large', n_zero, len(sequences))
    print(f"  plantcad2_large: {result.shape}")
    return result


# =============================================================================
# Names c13 requests, mapped to the functions above.
#
# NOT WIRED. Nothing imports this dict, and NUCLEOTIDE_REPS in nucleotide.py holds
# none of these nine names, so the loader cannot reach any of them yet. Adding them
# changes the representation pool, which means wiping the cache and re-running —
# the user's decision, not this module's.
# =============================================================================

FORWARD_COMPAT_NUCLEOTIDE_REPS = {
    'nucel':              _embed_nucel,               # 512D
    'generator_euk_1_2b': _embed_generator_euk_1_2b,  # 2048D
    'omni_dna_116m':      _embed_omni_dna_116m,       # 768D
    'metagene_1':         _embed_metagene_1,          # 4096D
    'evo2_1b':            _embed_evo2_1b,             # width read at runtime
    'janusdna':           _embed_janusdna,            # STUB
    'moderngena_large':   _embed_moderngena_large,    # 1024D
    'borzoi':             _embed_borzoi,              # 1536D, TRAINING OVERLAP
    'plantcad2_large':    _embed_plantcad2_large,     # 1536D
}
