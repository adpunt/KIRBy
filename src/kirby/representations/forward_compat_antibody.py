"""
Forward-compatibility antibody representations — the c13 candidate models.

Each function takes heavy_seqs + light_seqs (lists of str, equal length) and
returns an (N, D) float32 array, one row per input in input order. Light-chain
entries may be empty strings (heavy-only datasets); every function says in its
docstring what it does in that case.

Nothing here trains, fine-tunes or reads a label. Heavy imports and weight
downloads happen inside the function bodies, so importing this module pulls in
numpy only.

Models wired here (researched entries in
tests/experiments/complexity/forward_compat_reps.py, block ANTIBODY):

    ism          ISM 650M, jozhang97/ism_t33_650M_uc30pdb          2560D
    igbert       IgBert 420M, Exscientia/IgBert                    1024D
    mint         MINT 650M, varunullanat2012/mint                  2560D
    abcdr_esmc   AbCDR-ESMC, NOC-Lab/AbCDR-ESMC                    1152D
    antifold     AntiFold inverse folding, oxpig/AntiFold            40D
    abmpnn       AbMPNN — STUB, embedding API unconfirmed            —
    p_iggen      p-IgGen 22M, ollieturnbull/p-IgGen                 768D
    antiberta2   AntiBERTa2 202M, alchemab/antiberta2              2048D
    currab       CurrAb 650M, brineylab/CurrAb                     1280D

Widths above are for a paired dataset. On a heavy-only dataset every
chain-concatenating function returns half that width, the same convention the
existing antibody reps use via _concat_hl.

Two of these already existed and were recorded dead. What was actually wrong:

  igbert — antibody.py:443 passes raw un-spaced sequences to a BertTokenizer
    whose whole vocabulary is 30 single-character tokens and whose splitter is
    whitespace. "EVQLVESGGG..." arrives as ONE word and comes back [UNK], so
    every antibody in the dataset tokenizes to the same three-token sequence
    and every embedding row is near-identical. That is the recorded signature:
    R2 ~ 0, AUROC 0.50, F1w 0.11 (PIPELINE_READINESS_TODO.md:652/663/675,
    BILL_OF_MATERIALS.md:320). The model card requires
    ' '.join(heavy) + ' [SEP] ' + ' '.join(light). embed_igbert below spaces
    the residues. The sibling embed_igt5 at antibody.py:1213 already spaces
    them, and igt5 is alive in the pools — same corpus, same lab, one
    difference. Also: the old docstring says 768 features; config.json says
    hidden_size 1024, which matches the 1024 measured in
    PERTURBATION_SIMILARITY_SUITE_DESIGN.md:139.

  mint — protein.py:2757 calls model.get_embedding(seq). MINTWrapper has no
    such method; the README calls the wrapper directly, wrapper(chains,
    chain_ids), on a batch built by CollateFn. Every sequence therefore hits
    the AttributeError arm of that function's try/except and gets a zero row,
    which is the recorded "NaN everywhere". Two further problems in the same
    call path: the chains were fed one at a time, which throws away the
    cross-chain attention MINT exists for, and the wrapper's sep_chains flag
    (the one that returns per-chain means concatenated) was never set.

The old embed_igbert / embed_mint in antibody.py are left untouched. Register
the versions here instead — see the registration snippet in the handoff.
"""

import numpy as np
from typing import Callable, Dict, List

# Module-level model caches, populated on first call, in the style of
# create_molformer / _SAPROT_MODEL. Never touched at import time.
_ISM_MODEL = None
_ISM_TOKENIZER = None
_IGBERT_MODEL = None
_IGBERT_TOKENIZER = None
_MINT_WRAPPER = None
_ABCDR_MODEL = None
_ABCDR_TOKENIZER = None
_PIGGEN_MODEL = None
_PIGGEN_TOKENIZER = None
_ANTIBERTA2_MODEL = None
_ANTIBERTA2_TOKENIZER = None
_CURRAB_MODEL = None
_CURRAB_TOKENIZER = None
_ANTIFOLD_MODEL = None


# ── shared helpers ───────────────────────────────────────────────────────────


def _device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _has_light_local(light_seqs: List[str]) -> bool:
    """True when at least one light chain is a non-empty string."""
    return any(s is not None and s != '' for s in light_seqs)


def _concat_hl_local(h_embs: np.ndarray, l_embs: np.ndarray,
                     light_seqs: List[str]) -> np.ndarray:
    """Concatenate heavy and light embeddings; heavy only when no light chains.

    Deliberately a local copy rather than an import of antibody._concat_hl:
    registering these builders adds an import of this module to antibody.py,
    and a module-level import back the other way would be circular.
    """
    if _has_light_local(light_seqs):
        return np.hstack([h_embs, l_embs]).astype(np.float32)
    return h_embs.astype(np.float32)


def _embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
    max_length: int,
    dim: int,
    batch_size: int,
    label: str,
    exclude_leading: int = 0,
) -> np.ndarray:
    """Tokenize, forward, mean-pool over non-special tokens. (len(texts), dim).

    Pools the final hidden layer over tokens that are neither padding nor a
    special token, so [CLS], [SEP], <cls>, <eos> and padding are all excluded.
    exclude_leading drops that many leading positions as well, for models whose
    start marker is an ordinary vocabulary character.

    Two things decide "special", and BOTH are needed. The tokenizer's own
    special_tokens_mask only marks the specials the tokenizer ADDED — measured
    on this machine, ' '.join('EVQL') + ' [SEP] ' + ' '.join('DIQM') through
    Exscientia/IgBert gives special_tokens_mask [1,0,0,0,0,0,0,0,0,0,1]: the
    inner [SEP] at position 5 is marked 0, because it came from the input
    string rather than from the post-processor. Same for the '<cls>' CurrAb
    puts between its chains — 'MKT<cls>VRQ' through an ESM tokenizer marks
    only the outer <cls> and <eos>. So the token IDS are checked against
    tokenizer.all_special_ids as well, which is what makes the inner separator
    actually drop out of the mean.

    A batch that raises is retried one sequence at a time; a single sequence
    that still raises gets a zero row and is counted. No row is dropped and no
    row is reordered.
    """
    import inspect
    import torch

    n = len(texts)
    out = np.zeros((n, dim), dtype=np.float32)
    n_zero = 0

    # A tokenizer emits keys its own model family wants — p-IgGen's returns
    # token_type_ids, and GPTNeoXForCausalLM.forward() rejects that keyword
    # outright. Keep only what this model's forward actually takes.
    #
    # A forward() that declares **kwargs would make this filter drop input_ids
    # and forward nothing, zero-filling every row. None of the five classes
    # used here does that under transformers 4.45.1 (checked: GPTNeoX,
    # BertModel, RoFormerForMaskedLM, EsmForMaskedLM, EsmModel all declare
    # input_ids explicitly and no VAR_KEYWORD), but a version bump could, so
    # the filter switches itself off rather than silently emptying the call.
    try:
        _params = inspect.signature(model.forward).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in _params.values()):
            _accepted = None
        else:
            _accepted = set(_params)
    except (TypeError, ValueError):
        _accepted = None

    # Special-token IDs typed inside the input string (an inner [SEP], an inner
    # <cls>) that special_tokens_mask will not flag.
    _special_ids = None
    try:
        _ids = sorted({i for i in (getattr(tokenizer, 'all_special_ids', None) or [])
                       if isinstance(i, int)})
        if _ids:
            _special_ids = torch.tensor(_ids, dtype=torch.long)
    except Exception:  # noqa: BLE001 — pooling still works off the mask alone
        _special_ids = None

    def _run(batch_texts):
        enc = tokenizer(
            batch_texts, return_tensors='pt', padding=True, truncation=True,
            max_length=max_length, return_special_tokens_mask=True,
        )
        special = enc.pop('special_tokens_mask')
        enc = {k: v.to(device) for k, v in enc.items()}
        special = special.to(device)
        call = {k: v for k, v in enc.items()
                if _accepted is None or k in _accepted}
        if _accepted is None or 'output_hidden_states' in _accepted:
            call['output_hidden_states'] = True
        with torch.no_grad():
            outputs = model(**call)
        hidden = getattr(outputs, 'last_hidden_state', None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        mask = enc['attention_mask'] * (1 - special)
        if _special_ids is not None:
            inner = torch.isin(enc['input_ids'], _special_ids.to(device))
            mask = mask * (~inner).to(mask.dtype)
        if exclude_leading:
            mask[:, :exclude_leading] = 0
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.cpu().numpy()

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        try:
            out[idx] = _run([texts[i] for i in idx])
        except Exception as exc:  # noqa: BLE001 — retry smaller, then zero-fill
            print(f"    WARNING: {label} batch at row {start} failed "
                  f"({type(exc).__name__}: {exc}) — retrying one at a time")
            for i in idx:
                try:
                    out[i] = _run([texts[i]])[0]
                except Exception as exc2:  # noqa: BLE001
                    print(f"    WARNING: {label} failed on row {i} "
                          f"(len={len(texts[i])}): {type(exc2).__name__}: {exc2}")
                    n_zero += 1

    if n_zero:
        print(f"  WARNING: {label} zero-filled {n_zero}/{n} rows")
    return out


def _load_hf(loader: Callable, model_id: str, what: str, **kwargs):
    """from_pretrained with a message that names what to download and from where."""
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load {what} from HuggingFace repo '{model_id}': "
            f"{type(exc).__name__}: {exc}. "
            f"Download it with: huggingface-cli download {model_id} "
            f"(or set HF_HOME / HF_TOKEN if the machine has no network or the "
            f"repo needs an approved access request)."
        ) from exc


def _require_transformers():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This representation needs torch + transformers: "
            "pip install torch transformers"
        ) from exc


# ── ISM 650M (ESM-2 architecture, structure-token pretraining) ───────────────


def embed_ism(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """ISM 650M embeddings: H+L concatenated → (N, 2560). 1280 per chain.

    ISM (Implicit Structure Model, ICLR 2025) is ESM-2 650M continued-trained
    to predict discrete structure tokens. Same architecture, same tokenizer,
    same 1280-wide hidden state as the existing esm2 rep — and, unlike saprot,
    it needs no structure token at inference, so it runs on bare sequence.

    Pooling: mean of the final hidden layer over residue tokens (special tokens
    and padding excluded), per chain, then heavy and light concatenated.
    Heavy-only datasets return (N, 1280).

    Confirmed from the model card: AutoTokenizer + AutoModel, no
    trust_remote_code. config.json: model_type esm, hidden_size 1280,
    num_hidden_layers 33, max_position_embeddings 1026.

    HF model: jozhang97/ism_t33_650M_uc30pdb (MIT, ~2.6GB, auto-downloads)
    Requires: pip install torch transformers
    """
    global _ISM_MODEL, _ISM_TOKENIZER
    _require_transformers()
    from transformers import AutoTokenizer, AutoModel

    model_id = 'jozhang97/ism_t33_650M_uc30pdb'
    device = _device()

    if _ISM_MODEL is None:
        print(f"  Loading ISM ({model_id})...")
        _ISM_TOKENIZER = _load_hf(AutoTokenizer, model_id, 'the ISM tokenizer')
        _ISM_MODEL = _load_hf(AutoModel, model_id, 'the ISM weights')
        _ISM_MODEL = _ISM_MODEL.eval().to(device)

    def _run(seqs, label):
        texts = [s.upper().strip() for s in seqs]
        return _embed_texts(texts, _ISM_TOKENIZER, _ISM_MODEL, device,
                            max_length=1024, dim=1280, batch_size=8, label=label)

    h_embs = _run(heavy_seqs, 'ism/heavy')
    if not _has_light_local(light_seqs):
        return h_embs.astype(np.float32)
    l_embs = _run(light_seqs, 'ism/light')
    return _concat_hl_local(h_embs, l_embs, light_seqs)


# ── IgBert 420M (paired, Exscientia) — the fixed version ────────────────────


def embed_igbert(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """IgBert paired embeddings → (N, 1024). One row, both chains in one pass.

    Replaces antibody.py:443, which fed un-spaced sequences to a whitespace
    tokenizer with a 30-token single-character vocabulary — every antibody came
    back as [UNK] and every row was near-identical, which is the R2 ~ 0 /
    AUROC 0.50 recorded against 'igbert' in the surveys. The model card
    requires spaces:

        ' '.join(sequence_heavy) + ' [SEP] ' + ' '.join(sequence_light)

    which is what this builds. [SEP] inside the string is a special token, so
    BertTokenizer splits it out rather than reading it as residues.

    Pooling: mean of the final hidden layer over residue tokens of both chains,
    with [CLS], both [SEP] and padding excluded — the paper's "average over
    both chains' residue tokens". The inner [SEP] takes an extra step: the
    tokenizer's special_tokens_mask flags only the specials it added itself,
    and comes back [1,0,0,0,0,0,0,0,0,0,1] on 'E V Q L [SEP] D I Q M', leaving
    the separator at position 5 unflagged. _embed_texts also drops any token
    whose id is in tokenizer.all_special_ids, which is what actually keeps it
    out of the mean. Heavy-only datasets send the heavy chain alone and still
    return 1024 features.

    Confirmed from the model card: BertTokenizer(do_lower_case=False) +
    BertModel(add_pooling_layer=False). config.json: hidden_size 1024,
    num_hidden_layers 30, vocab_size 30, max_position_embeddings 40000.

    HF model: Exscientia/IgBert (CC BY 4.0, ~1.7GB, auto-downloads)
    Requires: pip install torch transformers
    """
    global _IGBERT_MODEL, _IGBERT_TOKENIZER
    _require_transformers()
    from transformers import BertModel, BertTokenizer

    model_id = 'Exscientia/IgBert'
    device = _device()

    if _IGBERT_MODEL is None:
        print(f"  Loading IgBert ({model_id})...")
        _IGBERT_TOKENIZER = _load_hf(BertTokenizer, model_id,
                                     'the IgBert tokenizer', do_lower_case=False)
        _IGBERT_MODEL = _load_hf(BertModel, model_id, 'the IgBert weights',
                                 add_pooling_layer=False)
        _IGBERT_MODEL = _IGBERT_MODEL.eval().to(device)

    paired = _has_light_local(light_seqs)
    texts = []
    for i in range(len(heavy_seqs)):
        h = ' '.join(list(heavy_seqs[i].upper().strip()))
        if paired and light_seqs[i]:
            l = ' '.join(list(light_seqs[i].upper().strip()))
            texts.append(f"{h} [SEP] {l}")
        else:
            texts.append(h)

    return _embed_texts(texts, _IGBERT_TOKENIZER, _IGBERT_MODEL, device,
                        max_length=512, dim=1024, batch_size=8, label='igbert')


# ── MINT 650M (cross-chain attention) — the fixed version ───────────────────


def _find_mint_checkpoint() -> str:
    """Locate mint.ckpt on disk, else download it from HuggingFace."""
    import os

    candidates = [
        os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint', 'mint.ckpt'),
        os.path.expanduser('~/kirby_models/mint/mint.ckpt'),
        'mint.ckpt',
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "MINT checkpoint not found on disk and huggingface_hub is not "
            "installed. Either pip install huggingface_hub, or download "
            "https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt "
            "to $KIRBY_MODELS_DIR/mint/mint.ckpt"
        ) from exc

    try:
        return hf_hub_download(repo_id='varunullanat2012/mint',
                               filename='mint.ckpt')
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not fetch mint.ckpt from varunullanat2012/mint: "
            f"{type(exc).__name__}: {exc}. Download "
            "https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt "
            "to $KIRBY_MODELS_DIR/mint/mint.ckpt"
        ) from exc


def _find_mint_config() -> str:
    """Locate MINT's esm2_t33_650M_UR50D.json, which ships with the mint repo."""
    import os

    try:
        import mint as _mint_pkg
    except ImportError as exc:
        raise ImportError(
            "MINT requires the mint package: "
            "git clone https://github.com/VarunUllanat/mint.git && "
            "pip install -e mint"
        ) from exc

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(_mint_pkg.__file__)))
    candidates = [
        os.path.join(pkg_dir, 'data', 'esm2_t33_650M_UR50D.json'),
        os.path.join(os.environ.get('KIRBY_MODELS_DIR', ''), 'mint',
                     'esm2_t33_650M_UR50D.json'),
        os.path.expanduser('~/kirby_models/mint/esm2_t33_650M_UR50D.json'),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise RuntimeError(
        "MINT config esm2_t33_650M_UR50D.json not found. It ships in the mint "
        f"repo at data/esm2_t33_650M_UR50D.json (looked in {candidates}). "
        "Clone https://github.com/VarunUllanat/mint.git and pip install -e it, "
        "or copy that file to $KIRBY_MODELS_DIR/mint/."
    )


def embed_mint(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """MINT embeddings, both chains in one pass → (N, 2560). 1280 per chain.

    Replaces antibody.py:921, which routed each chain separately through
    protein.py's create_mint_protein_embeddings. That function calls
    model.get_embedding(seq); MINTWrapper has no get_embedding, so every
    sequence hit the except arm and got a zero row — the recorded "NaN
    everywhere". It also fed one chain at a time, which discards the
    cross-chain attention MINT is built around.

    Pooling: MINTWrapper(..., sep_chains=True) mean-pools each chain over its
    own residues inside the model and returns the two means concatenated,
    heavy first — 1280 + 1280. This function does no pooling of its own.

    Heavy-only datasets: MINT takes a PAIR of chains. A heavy-only row is sent
    as (heavy, heavy) and only the first 1280 features — the heavy chain's own
    mean — are kept, so the returned width is 1280. The heavy chain still
    attends to a copy of itself, which is not the same number a true
    single-chain model would give. On a dataset that HAS light chains but
    where one row's light string is empty, that row is sent as (heavy, heavy)
    too, so its second 1280 features repeat its first 1280.

    Confirmed from the repo README and mint/helpers/extract.py:
        cfg = load_config("data/esm2_t33_650M_UR50D.json")
        wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)
        chains, chain_ids = CollateFn(512)(list_of_(seq1, seq2)_tuples)
        embeddings = wrapper(chains, chain_ids)
    MINTWrapper.__init__(cfg, checkpoint_path, freeze_percent=1.0,
    use_multimer=True, sep_chains=False, device="cuda:0"); with sep_chains=True
    forward returns torch.cat(mean_chain_outs, dim=-1).

    SETUP REQUIRED
      1. git clone https://github.com/VarunUllanat/mint.git && pip install -e mint
      2. mint.ckpt (~2.6GB, CC BY 4.0) from
         https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
         placed at $KIRBY_MODELS_DIR/mint/mint.ckpt, or left to
         huggingface_hub to fetch.
    """
    global _MINT_WRAPPER
    import torch

    try:
        from mint.helpers.extract import load_config, CollateFn, MINTWrapper
    except ImportError as exc:
        raise ImportError(
            "MINT requires the mint package: "
            "git clone https://github.com/VarunUllanat/mint.git && "
            "pip install -e mint"
        ) from exc

    device = _device()
    device_str = 'cuda:0' if device.type == 'cuda' else 'cpu'

    if _MINT_WRAPPER is None:
        config_path = _find_mint_config()
        checkpoint_path = _find_mint_checkpoint()
        print(f"  Loading MINT from {checkpoint_path}...")
        try:
            cfg = load_config(config_path)
            _MINT_WRAPPER = MINTWrapper(cfg, checkpoint_path, sep_chains=True,
                                        device=device_str)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"MINTWrapper failed to build from {checkpoint_path} with "
                f"config {config_path}: {type(exc).__name__}: {exc}"
            ) from exc
        _MINT_WRAPPER = _MINT_WRAPPER.to(device).eval()

    paired = _has_light_local(light_seqs)
    n = len(heavy_seqs)
    width = 2560 if paired else 1280
    out = np.zeros((n, width), dtype=np.float32)
    n_zero = 0
    collate = CollateFn(512)
    batch_size = 4

    def _pairs(indices):
        pairs = []
        for i in indices:
            h = heavy_seqs[i].upper().strip()
            l = light_seqs[i].upper().strip() if (paired and light_seqs[i]) else h
            pairs.append((h, l))
        return pairs

    def _run(indices):
        chains, chain_ids = collate(_pairs(indices))
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        with torch.no_grad():
            embs = _MINT_WRAPPER(chains, chain_ids)
        embs = embs.float().cpu().numpy()
        return embs[:, :width]

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        try:
            out[idx] = _run(idx)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: mint batch at row {start} failed "
                  f"({type(exc).__name__}: {exc}) — retrying one at a time")
            for i in idx:
                try:
                    out[i] = _run([i])[0]
                except Exception as exc2:  # noqa: BLE001
                    print(f"    WARNING: mint failed on row {i}: "
                          f"{type(exc2).__name__}: {exc2}")
                    n_zero += 1

    if n_zero:
        print(f"  WARNING: mint zero-filled {n_zero}/{n} rows")
    return out


# ── AbCDR-ESMC (CDR-masked ESM-C 600M) ──────────────────────────────────────


def embed_abcdr_esmc(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AbCDR-ESMC embeddings, both chains in one pass → (N, 1152).

    ESM-C 600M continued-trained on paired variable domains with masking
    concentrated in the six CDR loops. The model card's own call is followed
    exactly: base ESMC 600M weights are loaded first, then the fine-tuned
    state dict from the hub is applied over them.

    Input format: heavy and light joined by a single '-' (the card's SEP_TOKEN,
    which is the ESM alphabet's gap character). Heavy-only datasets send the
    heavy chain alone; the width is 1152 either way.

    Pooling: mean over ALL positions of the returned (1, L, 1152) embeddings,
    including BOS and EOS — that is what the model card does
    (embeddings.mean(dim=1)), so this matches the published usage rather than
    the exclude-BOS/EOS convention the ESM-C rep in antibody.py uses.

    SETUP REQUIRED
      pip install esm==3.1.4 safetensors huggingface_hub
      Base weights esmc_600m download on first use (EvolutionaryScale Cambrian
      licence, non-commercial). Fine-tuned weights: NOC-Lab/AbCDR-ESMC
      model.safetensors (MIT), fetched with hf_hub_download; set HF_TOKEN if
      the repo has since been gated.
    """
    global _ABCDR_MODEL, _ABCDR_TOKENIZER
    import os
    import torch

    try:
        from huggingface_hub import hf_hub_download
        from esm.tokenization import get_esmc_model_tokenizers
        from esm.models.esmc import ESMC
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "AbCDR-ESMC requires the EvolutionaryScale esm package plus "
            "safetensors and huggingface_hub: "
            "pip install esm safetensors huggingface_hub"
        ) from exc

    repo_id = 'NOC-Lab/AbCDR-ESMC'
    device = _device()

    if _ABCDR_MODEL is None:
        print(f"  Loading AbCDR-ESMC (esmc_600m + {repo_id})...")
        try:
            _ABCDR_TOKENIZER = get_esmc_model_tokenizers()
            model = ESMC.from_pretrained("esmc_600m").to(device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load base ESM-C 600M weights: "
                f"{type(exc).__name__}: {exc}. They download on first use via "
                "the esm package; check network access and disk space."
            ) from exc
        try:
            ckpt = hf_hub_download(repo_id=repo_id, filename='model.safetensors',
                                   token=os.getenv('HF_TOKEN', None))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not download model.safetensors from {repo_id}: "
                f"{type(exc).__name__}: {exc}. Fetch it with "
                f"'huggingface-cli download {repo_id} model.safetensors', and "
                "set HF_TOKEN if the repo needs an approved access request."
            ) from exc

        state = {}
        with safe_open(ckpt, framework='pt') as sf:
            for key in sf.keys():
                state[key] = sf.get_tensor(key)
        renamed = {
            (k[len('esmC_model.'):] if k.startswith('esmC_model.') else k): v
            for k, v in state.items()
        }
        # strict=False is what the model card uses, and on its own it is
        # silent: if the 'esmC_model.' prefix ever stops matching, every key
        # lands in unexpected_keys, nothing is overwritten, and this returns
        # plain esmc_600m embeddings that would be scored and published under
        # the AbCDR-ESMC name. Count what actually landed and refuse if the
        # fine-tuned weights did not reach the model.
        incompatible = model.load_state_dict(renamed, strict=False)
        n_ckpt = len(renamed)
        n_unexpected = len(getattr(incompatible, 'unexpected_keys', []))
        n_loaded = n_ckpt - n_unexpected
        if n_loaded == 0:
            raise RuntimeError(
                f"AbCDR-ESMC fine-tuned weights did not load: all {n_ckpt} "
                f"tensors in model.safetensors were rejected as unexpected "
                f"keys, so the model still holds the base esmc_600m weights. "
                f"The key prefix stripped here is 'esmC_model.'; check what "
                f"the checkpoint's keys actually start with before using this "
                f"representation — otherwise it scores base ESM-C under the "
                f"AbCDR-ESMC name."
            )
        print(f"    AbCDR-ESMC: {n_loaded}/{n_ckpt} checkpoint tensors loaded "
              f"onto esmc_600m ({n_unexpected} unexpected, "
              f"{len(getattr(incompatible, 'missing_keys', []))} missing)")
        _ABCDR_MODEL = model.eval()

    paired = _has_light_local(light_seqs)
    n = len(heavy_seqs)
    out = np.zeros((n, 1152), dtype=np.float32)
    n_zero = 0

    for i in range(n):
        h = heavy_seqs[i].upper().strip()
        seq = f"{h}-{light_seqs[i].upper().strip()}" if (paired and light_seqs[i]) else h
        try:
            enc = _ABCDR_TOKENIZER(seq, return_tensors='pt')
            ids = enc['input_ids'].to(device)
            with torch.no_grad():
                outputs = _ABCDR_MODEL(sequence_tokens=ids)
            out[i] = outputs.embeddings.float().mean(dim=1).squeeze(0).cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: abcdr_esmc failed on row {i} "
                  f"(len={len(seq)}): {type(exc).__name__}: {exc}")
            n_zero += 1
        if (i + 1) % 200 == 0:
            print(f"    abcdr_esmc progress: {i + 1}/{n}")

    if n_zero:
        print(f"  WARNING: abcdr_esmc zero-filled {n_zero}/{n} rows")
    return out


# ── AntiFold (inverse folding, needs a built structure) ─────────────────────


_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def _antifold_chain_logprobs(model, pdb_path, has_light: bool) -> Dict[str, np.ndarray]:
    """Per-chain (n_residues, 20) amino-acid log-probabilities from one PDB.

    Mirrors structural.py:_run_antifold, which keeps the heavy chain only;
    this keeps both, keyed 'H' and 'L'.
    """
    import pandas as pd

    try:
        from antifold.antiscripts import get_pdbs_logits, df_logits_to_logprobs
    except ImportError:
        from antifold.main import get_pdbs_logits, df_logits_to_logprobs

    pdb_dir = str(pdb_path.parent)
    row = {"pdb": [pdb_path.stem], "Hchain": ["H"]}
    if has_light:
        row["Lchain"] = ["L"]

    df_logits_list = get_pdbs_logits(
        model=model,
        pdbs_csv_or_dataframe=pd.DataFrame(row),
        pdb_dir=pdb_dir,
    )
    df_logprobs = df_logits_to_logprobs(df_logits_list[0])

    per_chain = {}
    for chain in (['H', 'L'] if has_light else ['H']):
        sub = df_logprobs[df_logprobs['pdb_chain'] == chain]
        if len(sub) == 0:
            continue
        per_chain[chain] = sub[_AA_ORDER].values.astype(np.float64)
    return per_chain


def embed_antifold(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AntiFold inverse-folding log-probabilities → (N, 40). 20 per chain.

    AntiFold scores sequence given backbone geometry — the opposite
    conditioning to every language model in the pools. It needs a 3D structure
    per antibody, so this calls the repo's existing
    structural.predict_structure (ABodyBuilder2 for paired, NanoBodyBuilder2
    for heavy-only), which caches each PDB to disk by sequence hash.

    Pooling: AntiFold emits 20 amino-acid log-probabilities per residue. Those
    are averaged over ALL residues of a chain, giving 20 features per chain,
    heavy then light — 40 for a paired dataset, 20 for heavy-only. A row with
    an empty light string inside a paired dataset goes to NanoBodyBuilder2 and
    leaves its last 20 features zero.

    NOT the hidden states. AntiFold's per-residue latent representations are
    reachable only through the command-line flag --extract_embeddings; the
    README documents no Python accessor for them, and their width is not stated
    anywhere on the pages read. This builder therefore uses the
    log-probability track, which get_pdbs_logits + df_logits_to_logprobs return
    directly and which structural.py already calls the same way.

    Cost: one ABodyBuilder2 structure per unique antibody (~5s GPU, ~60s CPU)
    on first use, cached afterwards. A row whose structure prediction fails is
    zero-filled and counted.

    SETUP REQUIRED
      pip install ImmuneBuilder biopython
      pip install git+https://github.com/oxpig/AntiFold.git  (BSD 3-Clause)
      Checkpoint model.pt from
      https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt
      (antifold's load_model() fetches/points to it).
    """
    global _ANTIFOLD_MODEL

    try:
        from antifold.antiscripts import load_model
    except ImportError:
        try:
            from antifold.main import load_model
        except ImportError as exc:
            raise ImportError(
                "AntiFold is not installed: "
                "pip install git+https://github.com/oxpig/AntiFold.git "
                "and put the checkpoint from "
                "https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt "
                "where antifold.load_model() looks for it (models/model.pt)."
            ) from exc

    try:
        from kirby.representations.structural import predict_structure
    except ImportError as exc:
        raise ImportError(
            "AntiFold needs ImmuneBuilder for structures: "
            "pip install ImmuneBuilder biopython"
        ) from exc

    if _ANTIFOLD_MODEL is None:
        print("  Loading AntiFold...")
        try:
            _ANTIFOLD_MODEL = load_model()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"AntiFold load_model() failed: {type(exc).__name__}: {exc}. "
                "Download the checkpoint from "
                "https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt"
            ) from exc

    paired = _has_light_local(light_seqs)
    n = len(heavy_seqs)
    width = 40 if paired else 20
    out = np.zeros((n, width), dtype=np.float32)
    n_zero = 0

    for i in range(n):
        heavy = heavy_seqs[i].upper().strip()
        light = light_seqs[i].upper().strip() if (paired and light_seqs[i]) else None
        try:
            pdb_path = predict_structure(heavy, light)
            if pdb_path is None:
                raise RuntimeError("ABodyBuilder2/NanoBodyBuilder2 returned no structure")
            per_chain = _antifold_chain_logprobs(_ANTIFOLD_MODEL, pdb_path,
                                                 has_light=light is not None)
            if 'H' not in per_chain:
                raise RuntimeError("AntiFold returned no heavy-chain rows")
            out[i, :20] = per_chain['H'].mean(axis=0)
            if paired and 'L' in per_chain:
                out[i, 20:40] = per_chain['L'].mean(axis=0)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: antifold failed on row {i}: "
                  f"{type(exc).__name__}: {exc}")
            n_zero += 1
        if (i + 1) % 100 == 0:
            print(f"    antifold progress: {i + 1}/{n}")

    if n_zero:
        print(f"  WARNING: antifold zero-filled {n_zero}/{n} rows")
    return out


# ── AbMPNN — STUB, embedding API unconfirmed ────────────────────────────────


def embed_abmpnn(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AbMPNN structure-conditioned features — STUB, NOT WIRED.

    CONFIRMED (Zenodo record 8164693, opened): the record is titled "Inverse
    folding for antibody sequence design using deep learning", holds
    abmpnn.pt (20.0 MB) plus oas_splits.csv and sabdab_splits.csv, is
    CC BY 4.0, and states the model "is based on ProteinMPNN and can be run
    using the corresponding code" at https://github.com/dauparas/ProteinMPNN.

    NOT CONFIRMED, which is why this raises instead of guessing:
      - ProteinMPNN ships a command-line script (protein_mpnn_run.py). No page
        opened documents a Python call that returns per-residue decoder
        features or log-probabilities as an array.
      - The feature width is unstated. ProteinMPNN's internal hidden width is
        not the same thing as whatever an embedding export would emit, and no
        source read gives a number for AbMPNN.
      - How the AbMPNN checkpoint is passed to that codebase (--path_to_model_weights
        plus a model name, or a direct torch.load into a rebuilt model) is not
        documented on the Zenodo record.

    It would also need a built structure per antibody, the same ABodyBuilder2
    step embed_antifold uses.

    Writing a plausible-looking loader here would fail on the cluster, so it
    raises. To wire it: read ProteinMPNN's protein_mpnn_run.py, establish the
    call that returns per-residue features and its width, then implement here
    with the pooling stated in the docstring.
    """
    raise NotImplementedError(
        "abmpnn is not wired: the weights are confirmed (abmpnn.pt, Zenodo "
        "record 8164693, CC BY 4.0, runs on github.com/dauparas/ProteinMPNN) "
        "but no documented Python call returning per-residue features — or its "
        "width — could be confirmed. Implement against ProteinMPNN's "
        "protein_mpnn_run.py before using this rep."
    )


# ── p-IgGen (17-22M autoregressive decoder, OPIG) ───────────────────────────


def embed_p_iggen(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """p-IgGen embeddings, both chains in one string → (N, 768).

    An autoregressive decoder — the only left-to-right model among these
    candidates. Heavy and light are concatenated into one string behind the
    start marker '1', which is p-IgGen's forward-direction token
    (src/piggen/model.py builds its prompts as f"1{heavy_chain}"). The
    tokenizer is character-level over 26 tokens with no whitespace splitting,
    so residues are passed un-spaced.

    Pooling: mean of the final hidden layer over every token except the leading
    '1' and any padding. One row of 768 features for the pair — chains are NOT
    pooled separately, so the width is 768 whether or not a light chain is
    present. Heavy-only datasets send '1' + heavy.

    Confirmed: HF card gives AutoTokenizer + AutoModelForCausalLM on
    'ollieturnbull/p-IgGen'; config.json gives model_type gpt_neox,
    hidden_size 768, num_hidden_layers 4, vocab_size 26,
    max_position_embeddings 400; tokenizer.json is character-level with <PAD>
    the only special token and ByteLevel pre-tokenization
    (add_prefix_space false).

    CAVEAT: the researched entry says 3 layers / 17,349,888 params. The live
    config.json says 4 hidden layers and the HF card says 22.1M params, so the
    checkpoint at this repo id is not exactly the one those numbers describe —
    ollieturnbull/p-IgGen-developable is the other published variant. Width 768
    is the same either way.

    HF model: ollieturnbull/p-IgGen (BSD 3-Clause, ~90MB, auto-downloads)
    Requires: pip install torch transformers
    """
    global _PIGGEN_MODEL, _PIGGEN_TOKENIZER
    _require_transformers()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = 'ollieturnbull/p-IgGen'
    device = _device()

    if _PIGGEN_MODEL is None:
        print(f"  Loading p-IgGen ({model_id})...")
        _PIGGEN_TOKENIZER = _load_hf(AutoTokenizer, model_id,
                                     'the p-IgGen tokenizer')
        if _PIGGEN_TOKENIZER.pad_token is None:
            _PIGGEN_TOKENIZER.pad_token = '<PAD>'
        _PIGGEN_MODEL = _load_hf(AutoModelForCausalLM, model_id,
                                 'the p-IgGen weights')
        _PIGGEN_MODEL = _PIGGEN_MODEL.eval().to(device)

    paired = _has_light_local(light_seqs)
    texts = []
    for i in range(len(heavy_seqs)):
        h = heavy_seqs[i].upper().strip()
        l = light_seqs[i].upper().strip() if (paired and light_seqs[i]) else ''
        texts.append(f"1{h}{l}")

    return _embed_texts(texts, _PIGGEN_TOKENIZER, _PIGGEN_MODEL, device,
                        max_length=400, dim=768, batch_size=16,
                        label='p_iggen', exclude_leading=1)


# ── AntiBERTa2 (202M RoFormer, Alchemab) ────────────────────────────────────


def embed_antiberta2(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """AntiBERTa2 embeddings: H+L concatenated → (N, 2048). 1024 per chain.

    A rotary-embedding RoFormer trained on OAS plus SAbDab. Its tokenizer is a
    30-token single-character vocabulary split on whitespace, so residues MUST
    be space-separated — the same requirement that killed the old igbert
    builder.

    Its vocabulary also carries two chain-marker characters beyond the 20 amino
    acids and X: 'Ḣ' and 'Ḷ'. The model card's only example input opens with
    'Ḣ Q V Q ... T V S S', so a heavy chain is presented to this model as the
    heavy marker followed by the spaced residues. Read locally out of the
    cached vocab.txt, and tokenizing 'Ḣ ' + ' '.join(...) gives
    ['[CLS]','Ḣ','Q','V',...,'[SEP]'] — one marker token, not [UNK]. The heavy
    chain therefore goes in behind 'Ḣ' and the light chain behind 'Ḷ'. The
    card shows only the heavy marker; 'Ḷ' for the light chain is read off the
    vocabulary, not off the card.

    Pooling: mean of the final hidden layer over residue tokens (specials and
    padding excluded), per chain, then heavy and light concatenated. The chain
    marker is an ordinary vocabulary token, not a special one, so it stays in
    the mean — the same constant offset on every row.
    Heavy-only datasets return (N, 1024).

    Confirmed from the model card: AutoTokenizer + AutoModelForMaskedLM on
    'alchemab/antiberta2' (the card also names RoFormerTokenizer /
    RoFormerForMaskedLM). config.json: model_type roformer, hidden_size 1024,
    num_hidden_layers 16, vocab_size 30, max_position_embeddings 256 — hence
    max_length 256, which is ample for one variable domain.

    LICENCE: non-commercial only per the model card; commercial use needs
    Alchemab's permission. Check that against where the results get published.

    HF model: alchemab/antiberta2 (~800MB, auto-downloads)
    Requires: pip install torch transformers
    """
    global _ANTIBERTA2_MODEL, _ANTIBERTA2_TOKENIZER
    _require_transformers()
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_id = 'alchemab/antiberta2'
    device = _device()

    if _ANTIBERTA2_MODEL is None:
        print(f"  Loading AntiBERTa2 ({model_id})...")
        _ANTIBERTA2_TOKENIZER = _load_hf(AutoTokenizer, model_id,
                                         'the AntiBERTa2 tokenizer')
        _ANTIBERTA2_MODEL = _load_hf(AutoModelForMaskedLM, model_id,
                                     'the AntiBERTa2 weights')
        _ANTIBERTA2_MODEL = _ANTIBERTA2_MODEL.eval().to(device)

    def _run(seqs, label, marker):
        texts = []
        for s in seqs:
            spaced = ' '.join(list(s.upper().strip()))
            texts.append(f"{marker} {spaced}" if spaced else marker)
        return _embed_texts(texts, _ANTIBERTA2_TOKENIZER, _ANTIBERTA2_MODEL,
                            device, max_length=256, dim=1024, batch_size=8,
                            label=label)

    h_embs = _run(heavy_seqs, 'antiberta2/heavy', 'Ḣ')
    if not _has_light_local(light_seqs):
        return h_embs.astype(np.float32)
    l_embs = _run(light_seqs, 'antiberta2/light', 'Ḷ')
    return _concat_hl_local(h_embs, l_embs, light_seqs)


# ── CurrAb (curriculum-trained paired antibody LM, 650M) ────────────────────


def embed_currab(heavy_seqs: List[str], light_seqs: List[str]) -> np.ndarray:
    """CurrAb paired embeddings, both chains in one pass → (N, 1280).

    ESM-2 650M architecture with rotary position embeddings, trained
    unpaired-first then annealed onto paired sequences. The model card joins
    the chains with a literal '<cls>' between them:

        "VQ..SS<cls>EV..IK"

    which the ESM tokenizer resolves to its own <cls> token rather than to
    residue characters (verified locally against an ESM vocabulary: '<cls>'
    inside a sequence string comes back as one special token).

    Pooling: mean of the final hidden layer over residue tokens of both chains,
    with the leading <cls>, the separator <cls>, <eos> and padding all
    excluded. One row of 1280 features for the pair — chains are NOT pooled
    separately, so the width is 1280 with or without a light chain.

    Heavy-only: the card gives a separate format for an unpaired chain — a
    heavy chain alone is "VQ..SS<cls>", with the separator still there and
    nothing after it, and an unpaired light chain is "<cls>EV..IK". So a
    heavy-only row goes in as heavy + "<cls>", not as a bare sequence, which
    is the slot the model was trained to see. A row with an empty light string
    inside a paired dataset gets the same trailing separator.

    Confirmed from the model card: EsmTokenizer + EsmForMaskedLM on
    'brineylab/CurrAb', MIT, not gated. config.json: model_type esm,
    hidden_size 1280, num_hidden_layers 33, vocab_size 33,
    max_position_embeddings 322, position_embedding_type rotary — hence
    max_length 322, which fits a paired VH+VL of ~250 residues plus specials.

    HF model: brineylab/CurrAb (~2.6GB, auto-downloads)
    Requires: pip install torch transformers
    """
    global _CURRAB_MODEL, _CURRAB_TOKENIZER
    _require_transformers()
    from transformers import EsmForMaskedLM, EsmTokenizer

    model_id = 'brineylab/CurrAb'
    device = _device()

    if _CURRAB_MODEL is None:
        print(f"  Loading CurrAb ({model_id})...")
        _CURRAB_TOKENIZER = _load_hf(EsmTokenizer, model_id,
                                     'the CurrAb tokenizer')
        _CURRAB_MODEL = _load_hf(EsmForMaskedLM, model_id, 'the CurrAb weights')
        _CURRAB_MODEL = _CURRAB_MODEL.eval().to(device)

    paired = _has_light_local(light_seqs)
    texts = []
    for i in range(len(heavy_seqs)):
        h = heavy_seqs[i].upper().strip()
        if paired and light_seqs[i]:
            texts.append(f"{h}<cls>{light_seqs[i].upper().strip()}")
        else:
            texts.append(f"{h}<cls>")

    return _embed_texts(texts, _CURRAB_TOKENIZER, _CURRAB_MODEL, device,
                        max_length=322, dim=1280, batch_size=8, label='currab')


# Names as c13 will request them. The registration snippet in the handoff adds
# these to ANTIBODY_REPS in antibody.py — do not import this dict from there at
# module level, it would be circular.
FORWARD_COMPAT_ANTIBODY_REPS: Dict[str, Callable] = {
    'ism': embed_ism,                  # 2560D (1280 x 2)
    'igbert': embed_igbert,            # 1024D (paired, one pass)
    'mint': embed_mint,                # 2560D (1280 x 2, sep_chains)
    'abcdr_esmc': embed_abcdr_esmc,    # 1152D (paired, one pass)
    'antifold': embed_antifold,        #   40D (20 log-probs x 2 chains)
    'abmpnn': embed_abmpnn,            # STUB — raises NotImplementedError
    'p_iggen': embed_p_iggen,          #  768D (paired, one pass)
    'antiberta2': embed_antiberta2,    # 2048D (1024 x 2)
    'currab': embed_currab,            # 1280D (paired, one pass)
}
