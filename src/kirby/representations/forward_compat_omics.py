"""Forward-compatibility representation builders — omics / transcriptomics domain.

WHAT THIS FILE IS. c13 takes one newly published embedding model at a time, searches a
hybrid over a dataset's existing pool without it, searches again with it added, and
subtracts. Each function here turns a TCGA bulk profile into ONE ROW PER SAMPLE of
float32 features so that c13 has something to add. No training, no fine-tuning, no
label use.

THE INPUT IS BULK, MOST OF THESE MODELS WERE PRETRAINED ON SINGLE CELLS. Seven of the
nine models below saw only single-cell profiles during pretraining; only BulkFormer saw
bulk RNA-seq. Each docstring says whether a bulk profile is a legitimate input for that
model or a stretch, and how the dataset's gene symbols are mapped onto the model's
vocabulary.

RETURN CONTRACT, THE SAME FOR EVERY FUNCTION HERE:
  np.ndarray, dtype float32, shape (n_inputs, D), rows in the input order.
  n_inputs is exactly expression.shape[0] (or beta.shape[0] for the two methylation
  models). A sample that fails to encode gets a zero row and a printed count of how
  many rows were zero-filled — it is never dropped, because c13 scores row i of this
  matrix against label i.

HOW A MISSING DEPENDENCY BEHAVES. Every caller in this repo catches an exception from a
builder, prints a warning and moves on. So a missing package or missing weights raises
ImportError or RuntimeError naming the package and the download, and nothing here calls
sys.exit or asserts.

NO NETWORK OR HEAVY IMPORT AT MODULE LOAD. torch, transformers and every model package
are imported inside the function bodies.

WHAT WAS CONFIRMED AND WHAT WAS NOT. Every loading call below was read off the model's
own model card, README or source file in the session that wrote this file; each function
docstring names the page. One model — CellFM — could not be confirmed and raises
NotImplementedError with what is known and what is missing. Do not replace that body
with a guess.
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


# =============================================================================
# SHARED HELPERS
# =============================================================================

_HUGO_TO_ENSEMBL: Optional[Dict[str, str]] = None


def _hugo_to_ensembl() -> Dict[str, str]:
    """HUGO gene symbol -> Ensembl gene ID, from the Geneformer dictionaries.

    Same source the existing `create_bulkrnabert_embeddings` builder uses, so the two
    agree on which symbol becomes which Ensembl ID. Cached in a module global.
    """
    global _HUGO_TO_ENSEMBL
    if _HUGO_TO_ENSEMBL is not None:
        return _HUGO_TO_ENSEMBL

    try:
        from kirby.finetuning.tcga_geneformer import _load_geneformer_dictionaries
        _, _, name_id_dict = _load_geneformer_dictionaries()
    except Exception as e:
        raise RuntimeError(
            "No HUGO->Ensembl mapping available. The mapping is downloaded from the "
            "HuggingFace repo 'ctheodoris/Geneformer' by "
            "kirby.finetuning.tcga_geneformer._load_geneformer_dictionaries; that call "
            f"failed with: {e}"
        )

    mapping: Dict[str, str] = {}
    for hugo, ensembl in name_id_dict.items():
        mapping[str(hugo)] = str(ensembl)
        mapping[str(hugo).upper()] = str(ensembl)
    _HUGO_TO_ENSEMBL = mapping
    return mapping


def _ensembl_for(gene_names: np.ndarray) -> List[Optional[str]]:
    """One Ensembl ID (or None) per entry of gene_names, in the input order."""
    mapping = _hugo_to_ensembl()
    out: List[Optional[str]] = []
    for g in gene_names:
        gs = str(g)
        out.append(mapping.get(gs) or mapping.get(gs.upper()))
    return out


def _log2tpm_to_tpm(expression: np.ndarray) -> np.ndarray:
    """Undo the log2(TPM+1) the TCGA loader applies, giving a TPM matrix.

    The TCGA expression matrix in this repo is log2(TPM+1) (see the docstrings on
    create_gene_pca100 and create_bulkrnabert_embeddings in genomic.py). Models that
    want raw counts get TPM here instead: TPM is proportional to counts within a sample
    up to gene length, and the depth term every one of these models computes from the
    row sum then lands near 1e6 rather than at the library size. Stated so the
    approximation is visible; nothing here reconstructs true counts, which TCGA
    log2(TPM+1) does not carry.
    """
    tpm = np.power(2.0, np.asarray(expression, dtype=np.float64)) - 1.0
    return np.maximum(tpm, 0.0).astype(np.float32)


def _device() -> str:
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _finish(matrix: np.ndarray, n_expected: int, n_zero_rows: int, name: str) -> np.ndarray:
    """Enforce the return contract and report zero-filled rows."""
    out = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if out.ndim != 2:
        raise RuntimeError(f"{name}: expected a 2-D matrix, got shape {out.shape}")
    if out.shape[0] != n_expected:
        raise RuntimeError(
            f"{name}: produced {out.shape[0]} rows for {n_expected} samples. "
            f"Refusing to return a matrix whose rows do not line up with the labels."
        )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if n_expected and n_zero_rows >= n_expected:
        raise RuntimeError(
            f"{name}: every one of the {n_expected} sample rows failed to encode, so "
            f"the matrix is all zeros. Returning it would give c13 a constant feature "
            f"block to score instead of an error. The per-batch WARNING lines above "
            f"carry the underlying exception."
        )
    if n_zero_rows:
        print(f"    WARNING: {name} zero-filled {n_zero_rows}/{n_expected} sample rows")
    print(f"    {name}: {out.shape[0]} samples x {out.shape[1]} features")
    return out


def _require_dir(path: Optional[str], env_var: str, what: str, where: str) -> str:
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            f"{what} not found. Set {env_var} to the directory holding it. "
            f"Download: {where}"
        )
    return path


def _build_anndata(matrix: np.ndarray,
                   var_ids: Sequence[str],
                   var_key: Optional[str] = None,
                   obs_names: Optional[Sequence[str]] = None):
    """A minimal AnnData: one row per sample, one column per gene, no filtering."""
    try:
        import anndata as ad
    except ImportError:
        raise ImportError(
            "anndata is required to feed these models: pip install anndata"
        )
    import pandas as pd

    X = np.asarray(matrix, dtype=np.float32)
    var = pd.DataFrame(index=pd.Index([str(v) for v in var_ids], name=None))
    if var_key is not None:
        var[var_key] = var.index.astype(str)
    adata = ad.AnnData(X=X, var=var)
    adata.obs_names = [str(o) for o in (obs_names or range(X.shape[0]))]
    adata.var_names_make_unique()
    return adata


# =============================================================================
# BULKFORMER — the only bulk-pretrained model in this file
# =============================================================================

_BULKFORMER_MODEL = None
_BULKFORMER_GENES: Optional[List[str]] = None
_BULKFORMER_INTERESTED = None


def create_bulkformer_embeddings(expression: np.ndarray,
                                 gene_names: np.ndarray) -> np.ndarray:
    """BulkFormer-147M sample-level embeddings. 640 features per sample.

    MODEL. BulkFormer (Cell Systems 2026, bioRxiv 2025-06-11), a 147M-parameter hybrid
    graph/performer encoder pretrained on 581,503 human BULK RNA-seq profiles over a
    fixed vocabulary of 20,010 protein-coding genes.

    BULK INPUT IS LEGITIMATE HERE, and this is the only model in this file for which
    that is true without qualification: BulkFormer was pretrained on bulk profiles of
    roughly this width, with no single-cell sparsity assumption.

    GENE MAPPING. BulkFormer indexes genes by Ensembl ID in the fixed order of
    data/bulkformer_gene_info.csv (column 'ensg_id', 20,010 entries). The dataset's HUGO
    symbols are mapped to Ensembl IDs through the Geneformer symbol dictionary and
    placed at their position in that order. A gene BulkFormer knows that the dataset
    does not carry is filled with the repo's own placeholder value of -10, exactly as
    the repo's main_gene_selection does, and the fraction of such genes is passed to the
    model as mask_prob, again as the repo's notebook does.

    POOLING. The encoder emits one contextual vector per gene. One row here is the MEAN
    over gene positions of that per-gene matrix — restricted to the repo's
    data/interested_gene_list.pt indices when that file is present, and over all 20,010
    gene positions when it is not. This is the notebook's output_feature_type
    'sample_level' with aggregate_type 'mean'.

    INPUT VALUES. The notebook feeds log1p(TPM). The TCGA matrix here is log2(TPM+1), so
    it is converted back to TPM and re-transformed with log1p.

    API SOURCE. https://github.com/KangBoming/BulkFormer — README, model/README.md,
    model/config.py and bulkformer_extract_feature.ipynb, all read in the session that
    wrote this function. The notebook defines extract_feature itself rather than
    shipping it in a module, so the loop below re-implements that notebook cell.

    SETUP REQUIRED (three separate downloads, none of them pip-installable):
      1. git clone https://github.com/KangBoming/BulkFormer  -> set KIRBY_BULKFORMER_DIR
         to that checkout (it holds utils/BulkFormer.py and model/config.py).
      2. Checkpoint bulkformer_147M.pt from the Google Drive links in model/README.md ->
         place at <dir>/model/bulkformer_147M.pt, or set KIRBY_BULKFORMER_CKPT.
      3. Resource files from Zenodo https://doi.org/10.5281/zenodo.15744294 —
         G_tcga.pt, G_tcga_weight.pt, esm2_feature_concat.pt, bulkformer_gene_info.csv
         and (optional) interested_gene_list.pt -> place in <dir>/data/.
      Python packages: torch, torch_geometric (for SparseTensor), pandas.

    LICENSE. MIT.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, 640) float32
    """
    global _BULKFORMER_MODEL, _BULKFORMER_GENES, _BULKFORMER_INTERESTED

    import sys
    n_samples = int(expression.shape[0])

    repo_dir = _require_dir(
        os.environ.get('KIRBY_BULKFORMER_DIR'),
        'KIRBY_BULKFORMER_DIR',
        'The BulkFormer repository checkout',
        'git clone https://github.com/KangBoming/BulkFormer',
    )
    data_dir = os.path.join(repo_dir, 'data')
    ckpt_path = os.environ.get(
        'KIRBY_BULKFORMER_CKPT', os.path.join(repo_dir, 'model', 'bulkformer_147M.pt'))

    try:
        import torch
    except ImportError:
        raise ImportError("BulkFormer requires torch: pip install torch")
    try:
        from torch_geometric.typing import SparseTensor
    except ImportError:
        raise ImportError(
            "BulkFormer requires torch_geometric for SparseTensor: "
            "pip install torch_geometric"
        )
    import pandas as pd

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from utils.BulkFormer import BulkFormer
        from model.config import model_params
    except ImportError as e:
        raise RuntimeError(
            f"Could not import BulkFormer from {repo_dir}: {e}. "
            f"KIRBY_BULKFORMER_DIR must point at the repository root that contains "
            f"utils/BulkFormer.py and model/config.py."
        )

    gene_info_path = os.path.join(data_dir, 'bulkformer_gene_info.csv')
    for required in (gene_info_path,
                     os.path.join(data_dir, 'G_tcga.pt'),
                     os.path.join(data_dir, 'G_tcga_weight.pt'),
                     os.path.join(data_dir, 'esm2_feature_concat.pt')):
        if not os.path.exists(required):
            raise RuntimeError(
                f"BulkFormer resource file missing: {required}. Download the resource "
                f"files from Zenodo https://doi.org/10.5281/zenodo.15744294 into "
                f"{data_dir}/"
            )
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"BulkFormer checkpoint missing: {ckpt_path}. Download bulkformer_147M.pt "
            f"from the Google Drive link in {repo_dir}/model/README.md, or set "
            f"KIRBY_BULKFORMER_CKPT."
        )

    device = _device()

    if _BULKFORMER_GENES is None:
        gene_info = pd.read_csv(gene_info_path)
        _BULKFORMER_GENES = [str(g) for g in gene_info['ensg_id'].tolist()]
    bulkformer_genes = _BULKFORMER_GENES
    n_vocab = len(bulkformer_genes)

    if _BULKFORMER_MODEL is None:
        from collections import OrderedDict
        graph = torch.load(os.path.join(data_dir, 'G_tcga.pt'),
                           map_location='cpu', weights_only=False)
        weights = torch.load(os.path.join(data_dir, 'G_tcga_weight.pt'),
                             map_location='cpu', weights_only=False)
        graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
        gene_emb = torch.load(os.path.join(data_dir, 'esm2_feature_concat.pt'),
                              map_location='cpu', weights_only=False)
        params = dict(model_params)
        params['graph'] = graph
        params['gene_emb'] = gene_emb
        model = BulkFormer(**params).to(device)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = OrderedDict()
        for key, value in ckpt.items():
            state[key[7:] if key.startswith('module.') else key] = value
        model.load_state_dict(state)
        model.eval()
        _BULKFORMER_MODEL = model
        print(f"    Loaded BulkFormer ({params.get('dim')}D, device={device})")

        interested_path = os.path.join(data_dir, 'interested_gene_list.pt')
        if os.path.exists(interested_path):
            _interested = torch.load(
                interested_path, map_location='cpu', weights_only=False)
            # torch.load returns a tensor here as often as a list, and `if tensor:`
            # raises "Boolean value of Tensor with more than one element is ambiguous".
            # Normalise to a plain numpy index array so both the emptiness test below
            # and the numpy fancy-index on gene_out behave.
            if hasattr(_interested, 'detach'):
                _interested = _interested.detach().cpu().numpy()
            _BULKFORMER_INTERESTED = np.asarray(_interested).ravel().astype(int)
            print(f"    Pooling over {len(_BULKFORMER_INTERESTED)} genes of interest")
        else:
            _BULKFORMER_INTERESTED = None
            print("    interested_gene_list.pt absent — pooling over all "
                  f"{n_vocab} gene positions")

    model = _BULKFORMER_MODEL

    # Align the dataset onto BulkFormer's 20,010-gene Ensembl order.
    log_tpm = np.log1p(_log2tpm_to_tpm(expression))
    pos_of = {eid: i for i, eid in enumerate(bulkformer_genes)}
    aligned = np.full((n_samples, n_vocab), -10.0, dtype=np.float32)
    filled = np.zeros(n_vocab, dtype=bool)
    mapped = 0
    for col, ensembl in enumerate(_ensembl_for(gene_names)):
        if ensembl is None:
            continue
        pos = pos_of.get(ensembl)
        if pos is None:
            continue
        aligned[:, pos] = log_tpm[:, col]
        filled[pos] = True
        mapped += 1
    n_missing = int((~filled).sum())
    mask_prob = float(n_missing) / float(n_vocab)
    print(f"    Mapped {mapped}/{len(gene_names)} genes onto BulkFormer's {n_vocab}; "
          f"{n_missing} vocabulary genes absent (mask_prob {mask_prob:.4f})")

    embed_dim = int(model_params.get('dim', 640))
    embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
    n_zero_rows = 0
    batch_size = int(os.environ.get('KIRBY_BULKFORMER_BATCH', '4'))

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            try:
                X = torch.tensor(aligned[start:end], dtype=torch.float32, device=device)
                gene_out = model(X, mask_prob=mask_prob, output_expr=False)
                gene_out = gene_out.detach().cpu().numpy()
                if _BULKFORMER_INTERESTED is not None and len(_BULKFORMER_INTERESTED):
                    gene_out = gene_out[:, _BULKFORMER_INTERESTED, :]
                embeddings[start:end] = np.mean(gene_out, axis=1)
            except Exception as e:
                print(f"    WARNING: BulkFormer failed on samples {start}-{end}: {e}")
                n_zero_rows += (end - start)
            if device == 'cuda':
                torch.cuda.empty_cache()

    return _finish(embeddings, n_samples, n_zero_rows, 'bulkformer')


# =============================================================================
# TRANSCRIPTFORMER
# =============================================================================

def create_transcriptformer_embeddings(expression: np.ndarray,
                                       gene_names: np.ndarray) -> np.ndarray:
    """TranscriptFormer (CZI) cell embeddings, one row per bulk sample.

    MODEL. TranscriptFormer (bioRxiv 2025-04-25), generatively pretrained on 112 million
    cells across 12 species. Three checkpoints: tf-sapiens (human), tf-exemplar,
    tf-metazoa. Default here is tf_sapiens.

    BULK INPUT IS A STRETCH. TranscriptFormer was pretrained on single cells and its
    dataloader expects count-like values it will clip at 30 (data_config.clip_counts).
    A bulk TCGA profile is a population average, not a cell; nothing about the model
    refuses it, but the c13 number for this rep is a number for a single-cell encoder
    reading a bulk average.

    GENE MAPPING. The model keys genes by Ensembl ID: data_config.gene_col_name is
    'ensembl_id' and filter_to_vocabs drops genes outside its vocabulary. The dataset's
    HUGO symbols are mapped to Ensembl IDs through the Geneformer symbol dictionary and
    written into adata.var['ensembl_id']; a symbol with no Ensembl ID is dropped, which
    drops a COLUMN, never a sample row.

    POOLING. inference_config.emb_type is 'cell', which is the model's own mean over the
    final-layer contextual gene embeddings of the expressed genes, excluding the assay
    token. Written by the package to obsm['embeddings']. The rows that come back are
    re-indexed onto the input order by sample name before returning, so a reordering
    inside the package cannot silently shift labels.

    INPUT VALUES. log2(TPM+1) is converted back to TPM, which is what is fed as the
    count-like matrix. Cell filtering is disabled (min_expressed_genes 0,
    filter_outliers 0.0) so no sample can be dropped.

    API SOURCE. https://github.com/czi-ai/transcriptformer — README, plus
    src/transcriptformer/model/inference.py (run_inference), src/transcriptformer/cli/
    inference.py (how the config is assembled) and src/transcriptformer/cli/conf/
    inference_config.yaml, all read in the session that wrote this function.

    RE-VERIFIED 2026-09-10 against those two source files. Confirmed: run_inference is
    typed `data_files: list[str] | list[anndata.AnnData]`, so passing the AnnData
    in memory is supported; the CLI's merge order is OmegaConf.merge(mlflow_cfg, cfg)
    with the checkpoint's config.json first and the YAML second, which is the order
    used below; obsm keys come from inference_config.output_keys. Also confirmed, and
    it is why the sample name is written as an obs column: run_inference returns a
    NEWLY BUILT anndata.AnnData(obs=pd.DataFrame(concat_output['obs']), obsm=..., ...)
    that does not carry the input index, so the returned obs_names are 0..n-1.

    SETUP REQUIRED:
      pip install transcriptformer
      transcriptformer download tf-sapiens     # writes ./checkpoints/tf_sapiens
      Set KIRBY_TRANSCRIPTFORMER_CKPT to that checkpoint directory (it must contain
      config.json, model_weights.pt and vocabs/).

    LICENSE. Not stated on the repository pages read.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, D) float32. The published entry states 2048 features; the width
        actually returned by the checkpoint is used, whatever it is.
    """
    n_samples = int(expression.shape[0])

    try:
        from transcriptformer.model.inference import run_inference
    except ImportError as e:
        raise ImportError(
            f"TranscriptFormer is not installed ({e}). Install it with "
            f"'pip install transcriptformer', then download weights with "
            f"'transcriptformer download tf-sapiens'."
        )
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("TranscriptFormer requires omegaconf: pip install omegaconf")

    ckpt_dir = os.environ.get('KIRBY_TRANSCRIPTFORMER_CKPT',
                              os.path.join('checkpoints', 'tf_sapiens'))
    ckpt_dir = _require_dir(
        ckpt_dir, 'KIRBY_TRANSCRIPTFORMER_CKPT',
        'The TranscriptFormer checkpoint directory',
        "transcriptformer download tf-sapiens (writes ./checkpoints/tf_sapiens)",
    )
    config_json = os.path.join(ckpt_dir, 'config.json')
    weights_pt = os.path.join(ckpt_dir, 'model_weights.pt')
    if not os.path.exists(config_json) or not os.path.exists(weights_pt):
        raise RuntimeError(
            f"{ckpt_dir} is not a TranscriptFormer checkpoint: it needs config.json and "
            f"model_weights.pt. Re-run 'transcriptformer download tf-sapiens'."
        )

    # Build the AnnData: TPM values, genes keyed by Ensembl ID.
    tpm = _log2tpm_to_tpm(expression)
    ensembl = _ensembl_for(gene_names)
    keep = [i for i, e in enumerate(ensembl) if e is not None]
    if not keep:
        raise RuntimeError(
            "No dataset gene symbol mapped to an Ensembl ID, so TranscriptFormer has "
            "no input. Check the Geneformer symbol dictionary download."
        )
    print(f"    Mapped {len(keep)}/{len(gene_names)} genes to Ensembl IDs")
    obs_names = [f's{i}' for i in range(n_samples)]
    adata = _build_anndata(tpm[:, keep], [ensembl[i] for i in keep],
                           var_key='ensembl_id', obs_names=obs_names)
    # run_inference rebuilds its output AnnData as anndata.AnnData(obs=pd.DataFrame(
    # concat_output['obs']), ...) — it carries obs COLUMNS through but NOT the input
    # index, so the returned obs_names are a fresh 0..n-1 integer index. The sample name
    # therefore has to travel as an obs column to survive the round trip.
    adata.obs['kirby_row_name'] = obs_names

    # Assemble the config exactly as the package's own CLI entry point does.
    import json
    from transcriptformer.cli import inference as tf_cli
    default_yaml = os.path.join(os.path.dirname(tf_cli.__file__), 'conf',
                                'inference_config.yaml')
    cfg = OmegaConf.load(default_yaml)
    with open(config_json) as f:
        cfg = OmegaConf.merge(OmegaConf.create(json.load(f)), cfg)

    cfg.model.checkpoint_path = ckpt_dir
    cfg.model.inference_config.load_checkpoint = weights_pt
    cfg.model.data_config.aux_vocab_path = os.path.join(ckpt_dir, 'vocabs')
    cfg.model.data_config.esm2_mappings_path = os.path.join(ckpt_dir, 'vocabs')
    cfg.model.data_config.gene_col_name = 'ensembl_id'
    cfg.model.data_config.min_expressed_genes = 0
    cfg.model.data_config.filter_outliers = 0.0
    cfg.model.inference_config.output_keys = ['embeddings']
    cfg.model.inference_config.obs_keys = ['all']
    cfg.model.inference_config.emb_type = 'cell'
    cfg.model.inference_config.batch_size = int(
        os.environ.get('KIRBY_TRANSCRIPTFORMER_BATCH', '8'))
    cfg.model.inference_config.device = 'cuda' if _device() == 'cuda' else 'cpu'
    if cfg.model.inference_config.device == 'cpu':
        cfg.model.inference_config.precision = 32

    out = run_inference(cfg, data_files=[adata])
    if 'embeddings' not in out.obsm:
        raise RuntimeError(
            f"TranscriptFormer returned no 'embeddings' in obsm; got "
            f"{list(out.obsm.keys())}"
        )
    emb = np.asarray(out.obsm['embeddings'], dtype=np.float32)

    # Re-index onto the input order by sample name; zero-fill anything the run lost.
    # Preferred route: the kirby_row_name obs column written above. If the package ever
    # stops carrying obs columns, fall back to position, which is only safe because
    # min_expressed_genes and filter_outliers were forced to 0 so no cell can be
    # dropped and the returned row count still equals the input row count.
    if 'kirby_row_name' in out.obs.columns:
        returned = {str(name): i for i, name in enumerate(out.obs['kirby_row_name'])}
    else:
        returned = {str(name): i for i, name in enumerate(out.obs_names)}
        if not any(str(n) in returned for n in obs_names):
            if emb.shape[0] != n_samples:
                raise RuntimeError(
                    f"TranscriptFormer returned {emb.shape[0]} rows for {n_samples} "
                    f"samples and carried neither the kirby_row_name column nor the "
                    f"input obs_names, so the rows cannot be matched to labels."
                )
            print("    TranscriptFormer carried no sample name back; falling back to "
                  "positional order (cell filtering is off, row counts match)")
            returned = {str(name): i for i, name in enumerate(obs_names)}
    matrix = np.zeros((n_samples, emb.shape[1]), dtype=np.float32)
    n_zero_rows = 0
    for i, name in enumerate(obs_names):
        j = returned.get(name)
        if j is None or j >= emb.shape[0]:
            n_zero_rows += 1
        else:
            matrix[i] = emb[j]

    return _finish(matrix, n_samples, n_zero_rows, 'transcriptformer')


# =============================================================================
# CELLFM — STUB, the embedding call could not be confirmed
# =============================================================================

def create_cellfm_embeddings(expression: np.ndarray,
                             gene_names: np.ndarray) -> np.ndarray:
    """CellFM (80M) — NOT IMPLEMENTED. The frozen-embedding call is unconfirmed.

    THIS IS A STUB AND IT RAISES. Everything below was confirmed by reading the pages
    named; the one thing that could not be confirmed is how to get a frozen sample
    embedding out of the released code, so no loading code is written here. A stub that
    says unverified is worth more than a plausible invention that fails on the cluster.

    WHAT WAS CONFIRMED, by reading https://huggingface.co/ShangguanNingyuan/CellFM
    (model card), https://github.com/biomed-AI/CellFM (config.py, model.py,
    tutorials/CellAnnotation/CellAnnotation_zeroshot.ipynb):
      - Weights: HuggingFace ShangguanNingyuan/CellFM, checkpoint CellFM_80M_weight.ckpt,
        license CC BY-NC-ND 4.0 (non-commercial, no derivatives).
      - The reference implementation is MINDSPORE, not PyTorch. The card's own install
        list is mindspore=2.2.10, scanpy=1.10, scib=1.1.5. Nothing in the repo runs on
        torch. That alone makes it a different software stack from every other rep here.
      - Width: config.py Config_80M.enc_dims = 1536, so a sample row would be 1536
        features, matching the published entry.
      - Gene cap: Config_80M.nonz_len = 2048 and Prepare.n_genes = 24078. When a sample
        carries more than 2048 nonzero genes the repo's Prepare.sample draws 2048 of
        them with probability proportional to log1p(expression), so a bulk TCGA profile
        — which is nonzero almost everywhere — enters as a weighted sample of its
        high-expression genes, not as its whole transcriptome.
      - Vocabulary: csv/expand_gene_info.csv in the repo, indexed by gene symbol, with
        token id = row position + 1.
      - model.py CellFM.forward(expr, gene, zero_idx) returns (expr_emb, gene_emb,
        cls_token) and cls_token is (batch, enc_dims) — that is the sample embedding.

    WHAT COULD NOT BE CONFIRMED, and why this raises: the three tensors that forward
    wants (masked_nzdata, nonz_gene, zero_idx) are produced by a MindSpore
    mindspore.dataset pipeline of six chained maps over Prepare.seperate / sample /
    compress / attn_mask / pad_zero. The repo publishes no frozen-embedding entry point
    and no tutorial that calls forward directly — the only two notebooks fine-tune a
    classifier. Reconstructing that pipeline from the training code would be a guess at
    the padding, masking and zero-token conventions, and a wrong guess produces a matrix
    that looks fine and means nothing.

    TO IMPLEMENT IT: get one worked example from the authors, or from a repo tutorial
    that extracts cls_token, showing the exact contents of nonz_data / nonz_gene /
    zero_idx for a single sample under Config_80M. Then this function is short.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(
        "cellfm has no builder. Confirmed: weights ShangguanNingyuan/CellFM "
        "(CellFM_80M_weight.ckpt, CC BY-NC-ND 4.0), MindSpore-only reference "
        "implementation, 1536-D cls_token, 2048-gene input cap sampled from the "
        "high-expression genes, gene vocabulary csv/expand_gene_info.csv. "
        "NOT confirmed: how to build the (masked_nzdata, nonz_gene, zero_idx) tensors "
        "that CellFM.forward needs outside the repo's MindSpore training dataloader. "
        "The repo publishes no frozen-embedding entry point. Do not guess it."
    )


# =============================================================================
# scPRINT-2
# =============================================================================

_SCPRINT2_MODEL = None


def create_scprint2_embeddings(expression: np.ndarray,
                               gene_names: np.ndarray) -> np.ndarray:
    """scPRINT-2 cell embeddings, one row per bulk sample.

    MODEL. scPRINT-2 (bioRxiv 2025-12-11), an encoder-compressor-decoder model with ~20M
    active parameters whose embedding is a genuine low-dimensional bottleneck rather
    than a pooled token matrix. The published entry gives 256 features for the base
    checkpoint and 512 for medium.

    BULK INPUT IS A STRETCH. Pretrained on single cells; a bulk TCGA profile is fed as
    one pseudo-cell.

    GENE MAPPING. scPRINT-2 accepts var_names as either Ensembl IDs or HUGO symbols (its
    own FAQ says so). This builder writes Ensembl IDs, mapped from the dataset's HUGO
    symbols through the Geneformer symbol dictionary, because the model's gene list is
    Ensembl-keyed. A symbol with no Ensembl ID is dropped as a COLUMN. The model's own
    Collator then selects which genes go into the context.

    POOLING. Done inside the package: Embedder(...) with pred_embedding=['all'] writes
    one row per cell to obsm['scprint_emb']. how='most expr' is used here so the context
    is the sample's top-expressed genes rather than a random draw, which makes the row
    deterministic. Classification and plotting are turned off (doclass=False,
    doplot=False) so nothing but the embedding is computed.

    ROW SAFETY. scdataloader's Preprocessor is deliberately NOT run: it can filter cells,
    and a dropped cell here would misalign every later row against its label. The
    AnnData goes to the Embedder as built. Rows are re-indexed by sample name on the way
    out and anything missing is zero-filled.

    INPUT VALUES. scPRINT-2 takes raw counts and does its own normalisation, so the
    log2(TPM+1) matrix is converted back to TPM and passed as the count-like matrix.

    API SOURCE. https://github.com/cantinilab/scPRINT-2 — README (install, checkpoint
    download, scPRINT2.load_from_checkpoint, the anndata requirements) and
    scprint2/tasks/cell_emb.py (Embedder.__init__ and __call__ signatures, and the
    obsm['scprint_emb'] key), both read in the session that wrote this function.

    SETUP REQUIRED:
      uv pip install scprint2      (or pip install scprint2)
      lamin ontology setup, which scPRINT-2 needs to resolve organisms and genes:
        from scdataloader.utils import populate_my_ontology; populate_my_ontology()
        then _adding_scbasecamp_genes()
      Checkpoint:  hf download jkobject/scPRINT small-v2.ckpt --local-dir .
      Set KIRBY_SCPRINT2_CKPT to that .ckpt file.

    LICENSE. The paper states the pretrained weights and all 42 ablation checkpoints are
    GPL-v3.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, D) float32, D from the checkpoint (256 base / 512 medium).
    """
    global _SCPRINT2_MODEL

    n_samples = int(expression.shape[0])

    try:
        from scprint2 import scPRINT2
        from scprint2.tasks import Embedder
    except ImportError as e:
        raise ImportError(
            f"scPRINT-2 is not installed ({e}). Install with 'pip install scprint2', "
            f"then set up the lamin ontology "
            f"(scdataloader.utils.populate_my_ontology()) and download a checkpoint "
            f"with 'hf download jkobject/scPRINT small-v2.ckpt --local-dir .'"
        )

    ckpt = os.environ.get('KIRBY_SCPRINT2_CKPT')
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError(
            "scPRINT-2 checkpoint not found. Set KIRBY_SCPRINT2_CKPT to a .ckpt file. "
            "Download: hf download jkobject/scPRINT small-v2.ckpt --local-dir ."
        )

    if _SCPRINT2_MODEL is None:
        model = scPRINT2.load_from_checkpoint(
            ckpt, precpt_gene_emb=None, gene_pos_file=None)
        model.eval()
        _SCPRINT2_MODEL = model
        print(f"    Loaded scPRINT-2 from {ckpt}")
    model = _SCPRINT2_MODEL

    tpm = _log2tpm_to_tpm(expression)
    ensembl = _ensembl_for(gene_names)
    keep = [i for i, e in enumerate(ensembl) if e is not None]
    if not keep:
        raise RuntimeError(
            "No dataset gene symbol mapped to an Ensembl ID, so scPRINT-2 has no input."
        )
    print(f"    Mapped {len(keep)}/{len(gene_names)} genes to Ensembl IDs")
    obs_names = [f's{i}' for i in range(n_samples)]
    adata = _build_anndata(tpm[:, keep], [ensembl[i] for i in keep],
                           obs_names=obs_names)
    adata.obs['organism_ontology_term_id'] = 'NCBITaxon:9606'

    embedder = Embedder(
        batch_size=int(os.environ.get('KIRBY_SCPRINT2_BATCH', '8')),
        num_workers=0,
        how='most expr',
        max_len=int(os.environ.get('KIRBY_SCPRINT2_MAXLEN', '2000')),
        doclass=False,
        pred_embedding=['all'],
        doplot=False,
    )
    out, _metrics = embedder(model, adata)

    if 'scprint_emb' not in out.obsm:
        raise RuntimeError(
            f"scPRINT-2 returned no 'scprint_emb' in obsm; got {list(out.obsm.keys())}"
        )
    emb = np.asarray(out.obsm['scprint_emb'], dtype=np.float32)

    returned = {str(name): i for i, name in enumerate(out.obs_names)}
    matrix = np.zeros((n_samples, emb.shape[1]), dtype=np.float32)
    n_zero_rows = 0
    for i, name in enumerate(obs_names):
        j = returned.get(name)
        if j is None:
            n_zero_rows += 1
        else:
            matrix[i] = emb[j]

    return _finish(matrix, n_samples, n_zero_rows, 'scprint2')


# =============================================================================
# AIDO.Cell-100M  (released a second time as GB.Cell-100M)
# =============================================================================

_AIDO_CELL_MODEL = None
_AIDO_CELL_GENES: Optional[np.ndarray] = None


def _aido_cell_gene_list() -> np.ndarray:
    """The 19,264 gene symbols AIDO.Cell was pretrained on, in the model's own order.

    Read from OS_scRNA_gene_index.19264.tsv, column 'gene_name', which ships inside the
    installed package (gb_cell or modelgenerator). Not downloaded here.
    """
    global _AIDO_CELL_GENES
    if _AIDO_CELL_GENES is not None:
        return _AIDO_CELL_GENES

    import pandas as pd
    candidates = []
    try:
        import gb_cell.models as _m
        candidates.append(os.path.join(
            os.path.dirname(_m.__file__), 'gene_lists',
            'OS_scRNA_gene_index.19264.tsv'))
    except Exception:
        pass
    try:
        import modelgenerator.cell as _c
        candidates.append(os.path.join(
            os.path.dirname(_c.__file__), 'gene_lists',
            'OS_scRNA_gene_index.19264.tsv'))
    except Exception:
        pass
    env_path = os.environ.get('KIRBY_AIDO_CELL_GENE_LIST')
    if env_path:
        candidates.insert(0, env_path)

    for path in candidates:
        if path and os.path.exists(path):
            _AIDO_CELL_GENES = pd.read_csv(path, sep='\t')['gene_name'].to_numpy()
            return _AIDO_CELL_GENES

    raise RuntimeError(
        "AIDO.Cell gene list OS_scRNA_gene_index.19264.tsv not found. It ships inside "
        "the gb_cell package (huggingface/gb.cell in github.com/genbio-ai/ModelGenerator). "
        "Install that package, or set KIRBY_AIDO_CELL_GENE_LIST to the .tsv path."
    )


def create_aido_cell_embeddings(expression: np.ndarray,
                                gene_names: np.ndarray) -> np.ndarray:
    """AIDO.Cell-100M embeddings, one row per bulk sample.

    MODEL. AIDO.Cell-100M (bioRxiv 2024-11-28; NeurIPS 2024 workshop), also released as
    GB.Cell-100M: 18 layers, hidden size 650, 20 heads, with a read-depth-aware
    objective and auto-discretised continuous expression values.

    BULK INPUT IS THE CLOSEST FIT AFTER BULKFORMER. Its context is the ENTIRE
    19,264-gene transcriptome with no truncation, so a dense bulk vector goes in whole
    rather than being cut to a top-k. It was still pretrained on single cells.

    GENE MAPPING. AIDO.Cell indexes by HUGO SYMBOL, not Ensembl: the fixed order is the
    'gene_name' column of OS_scRNA_gene_index.19264.tsv. The dataset's symbols are
    matched to that list directly. Genes in the dataset that AIDO.Cell does not know are
    dropped; genes AIDO.Cell knows that the dataset lacks are set to zero and masked out
    of attention, which is exactly the repo's align_adata behaviour, re-implemented here
    in numpy so no AnnData round-trip is needed.

    PREPROCESSING, matching the repo's preprocess_counts step for step: two depth tokens
    equal to max(log10(row sum), 5) are appended to the sequence, values are normalised
    to log1p(counts / row_sum * 1e4), and the result is clipped at 20.

    POOLING. Attention-mask-weighted MEAN over the final hidden state across all
    positions, including the two depth tokens — the repo's own embed.py pools exactly
    that way, and this follows it rather than diverging.

    INPUT VALUES. The model wants raw counts. log2(TPM+1) is converted back to TPM and
    passed as the count-like matrix; the depth tokens then land near log10(1e6) = 6.

    API SOURCE. https://huggingface.co/genbio-ai/AIDO.Cell-100M points at the GB.Cell
    quickstart repo; the loading call, the alignment and the preprocessing were read
    from https://github.com/genbio-ai/ModelGenerator, files huggingface/gb.cell/embed.py,
    huggingface/gb.cell/gb_cell/utils/gene_alignment.py and .../preprocessing.py, in the
    session that wrote this function.

    SETUP REQUIRED:
      git clone https://github.com/genbio-ai/ModelGenerator
      pip install -e ModelGenerator/huggingface/gb.cell
      (or pip install modelgenerator, which ships the same CellFoundation classes at
       modelgenerator.huggingface_models.cellfoundation)
      Weights download from HuggingFace on first call. Set KIRBY_AIDO_CELL_MODEL to pick
      a different handle; default tries genbio-ai/AIDO.Cell-100M then
      genbio-ai/GB.Cell-100M.

    LICENSE. GenBio AI Community License — academic / non-commercial with attribution.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, 650) float32 for the 100M checkpoint; whatever hidden size the
        loaded checkpoint has in general.
    """
    global _AIDO_CELL_MODEL

    n_samples = int(expression.shape[0])

    try:
        import torch
    except ImportError:
        raise ImportError("AIDO.Cell requires torch: pip install torch")

    CellFoundationModel = CellFoundationConfig = None
    try:
        from gb_cell.models import CellFoundationModel, CellFoundationConfig
    except ImportError:
        try:
            from modelgenerator.huggingface_models.cellfoundation import (
                CellFoundationModel, CellFoundationConfig)
        except ImportError as e:
            raise ImportError(
                f"AIDO.Cell needs the CellFoundation classes ({e}). Install them with "
                f"'pip install -e ModelGenerator/huggingface/gb.cell' after "
                f"'git clone https://github.com/genbio-ai/ModelGenerator', or "
                f"'pip install modelgenerator'."
            )

    model_genes = _aido_cell_gene_list()
    n_vocab = len(model_genes)

    if _AIDO_CELL_MODEL is None:
        handles = [os.environ.get('KIRBY_AIDO_CELL_MODEL')] if os.environ.get(
            'KIRBY_AIDO_CELL_MODEL') else ['genbio-ai/AIDO.Cell-100M',
                                           'genbio-ai/GB.Cell-100M']
        last_error = None
        for handle in handles:
            try:
                config = CellFoundationConfig.from_pretrained(handle)
                model = CellFoundationModel.from_pretrained(handle, config=config)
                _AIDO_CELL_MODEL = model
                print(f"    Loaded AIDO.Cell from {handle}")
                break
            except Exception as e:
                last_error = e
        if _AIDO_CELL_MODEL is None:
            raise RuntimeError(
                f"Could not load AIDO.Cell weights from {handles}: {last_error}. "
                f"Weights: https://huggingface.co/genbio-ai/AIDO.Cell-100M"
            )
        device = _device()
        _AIDO_CELL_MODEL = _AIDO_CELL_MODEL.to(device)
        if device == 'cuda':
            _AIDO_CELL_MODEL = _AIDO_CELL_MODEL.to(torch.bfloat16)
        _AIDO_CELL_MODEL.eval()

    model = _AIDO_CELL_MODEL
    device = next(model.parameters()).device

    # Align to the model's fixed gene order; zero-fill and mask what it does not have.
    tpm = _log2tpm_to_tpm(expression)
    pos_of = {str(g): i for i, g in enumerate(model_genes)}
    aligned = np.zeros((n_samples, n_vocab), dtype=np.float32)
    present = np.zeros(n_vocab, dtype=np.float32)
    mapped = 0
    for col, g in enumerate(gene_names):
        pos = pos_of.get(str(g))
        if pos is None:
            pos = pos_of.get(str(g).upper())
        if pos is None:
            continue
        aligned[:, pos] = tpm[:, col]
        present[pos] = 1.0
        mapped += 1
    print(f"    Mapped {mapped}/{len(gene_names)} symbols onto AIDO.Cell's {n_vocab}; "
          f"{int(n_vocab - present.sum())} vocabulary genes zero-filled and masked")

    hidden = int(getattr(model.config, 'hidden_size', 650))
    embeddings = np.zeros((n_samples, hidden), dtype=np.float32)
    n_zero_rows = 0
    batch_size = int(os.environ.get('KIRBY_AIDO_CELL_BATCH', '8'))
    gene_mask = torch.from_numpy(present).unsqueeze(0).to(device)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            try:
                counts = torch.from_numpy(aligned[start:end]).to(
                    dtype=torch.float32, device=device)
                total = counts.sum(dim=1, keepdim=True)
                total = torch.clamp(total, min=1e-6)
                depth = torch.maximum(torch.log10(total),
                                      torch.tensor(5.0, device=device))
                values = torch.log1p(counts / total * 10000)
                values = torch.cat((values, depth, depth), dim=1)
                values = torch.clamp(values, max=20)
                if device.type == 'cuda':
                    values = values.to(torch.bfloat16)

                attn = gene_mask.repeat(values.shape[0], 1)
                attn = torch.cat(
                    [attn, torch.ones((values.shape[0], 2), device=device)], dim=1)

                outputs = model(input_ids=values, attention_mask=attn,
                                output_hidden_states=True)
                last = outputs.last_hidden_state
                m = attn.unsqueeze(-1).float()
                pooled = (last.float() * m).sum(dim=1) / m.sum(dim=1)
                embeddings[start:end] = pooled.cpu().numpy()
            except Exception as e:
                print(f"    WARNING: AIDO.Cell failed on samples {start}-{end}: {e}")
                n_zero_rows += (end - start)

    return _finish(embeddings, n_samples, n_zero_rows, 'aido_cell')


# =============================================================================
# METHYLATION MODELS — these take a beta matrix, NOT expression
# =============================================================================
# Both functions below have the signature (beta, cpg_ids), not
# (expression, gene_names): they read DNA methylation, a different measurement from
# RNA-seq expression. beta is (n_samples, n_cpgs) with values in [0, 1] and cpg_ids is
# the matching array of Illumina probe IDs (cg#######). The two adapters at the bottom
# of this section let the TCGA dispatch call them with (sample_ids, cohort), which is
# how methyl_pca50 and methyl_hvg are already dispatched.

_METHYLGPT_MODEL = None
_METHYLGPT_VOCAB = None


def create_methyl_methylgpt_embeddings(beta: np.ndarray,
                                       cpg_ids: np.ndarray) -> np.ndarray:
    """MethylGPT sample embeddings from a beta matrix. A METHYLATION block.

    SIGNATURE IS (beta, cpg_ids), NOT (expression, gene_names). beta is
    (n_samples, n_cpgs) of HM450 beta values in [0, 1]; cpg_ids is the matching array of
    Illumina probe IDs.

    MODEL. MethylGPT (bioRxiv 2024-10-30 v2), a transformer over 49,156 curated CpG
    sites, which the TCGA 450K arrays cover. The published entry gives 64 features per
    sample; the docs' own worked example prints 128, so the width the loaded checkpoint
    returns is used rather than a hardcoded number.

    THE BULK QUESTION DOES NOT ARISE. MethylGPT was pretrained on bulk array methylomes
    (Generation Scotland and similar cohorts, n=18,859 in its disease-prediction
    evidence), so a TCGA bulk beta profile is exactly the kind of input it saw.

    PROBE MAPPING. The model's vocabulary is a fixed list of CpG probe IDs
    (vocab.CpG_ids, built from probe_ids_type3.csv). The dataset's beta columns are
    placed at their position in that list; a probe the model knows that the dataset does
    not carry is written as the config's pad_value and MethylGPT's own prepare_data
    turns NaN into pad_value as well.

    POOLING. The <cls> token. model.get_cell_embeddings returns output_dict['cell_emb'],
    which is the sample-level vector from the [CLS] position prepended to the CpG
    sequence — no averaging over CpGs.

    MASKING IS TURNED OFF. config['mask_ratio'] is set to 0.0 before prepare_data, so the
    row is a deterministic encoding of the sample rather than one draw of a masked view.

    API SOURCE. https://github.com/albert-ying/MethylGPT — docs/inference_guide.md (the
    MethylVocab / MethylGPTModel.from_pretrained / extract_embeddings sequence),
    methylgpt/inference/embeddings.py and methylgpt/model/methyl_model.py
    (prepare_data and get_cell_embeddings), all read in the session that wrote this
    function.

    SETUP REQUIRED:
      git clone https://github.com/albert-ying/MethylGPT && pip install -e MethylGPT
      Checkpoints come from the Google Drive links in that README. A model directory
      must hold args.json and best_model.pt.
      Set KIRBY_METHYLGPT_MODEL_DIR to that directory.
      Set KIRBY_METHYLGPT_PROBE_CSV to the probe id file (data/probe_ids_type3.csv in
      the repo) if it is not at <model dir>/../data/probe_ids_type3.csv.

    LICENSE. Apache 2.0 on the code. The weight files' terms are not stated separately.

    Args:
        beta: (n_samples, n_cpgs) methylation beta values
        cpg_ids: (n_cpgs,) Illumina probe IDs

    Returns:
        (n_samples, D) float32
    """
    global _METHYLGPT_MODEL, _METHYLGPT_VOCAB

    import json
    n_samples = int(beta.shape[0])

    try:
        import torch
    except ImportError:
        raise ImportError("MethylGPT requires torch: pip install torch")
    try:
        from methylgpt import MethylGPTModel, MethylVocab
    except ImportError as e:
        raise ImportError(
            f"MethylGPT is not installed ({e}). "
            f"git clone https://github.com/albert-ying/MethylGPT && "
            f"pip install -e MethylGPT"
        )

    model_dir = _require_dir(
        os.environ.get('KIRBY_METHYLGPT_MODEL_DIR'),
        'KIRBY_METHYLGPT_MODEL_DIR',
        'The MethylGPT pretrained model directory (args.json + best_model.pt)',
        'the Google Drive links in https://github.com/albert-ying/MethylGPT',
    )
    args_json = os.path.join(model_dir, 'args.json')
    weights = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(args_json) or not os.path.exists(weights):
        raise RuntimeError(
            f"{model_dir} must contain args.json and best_model.pt. Download a "
            f"checkpoint from the Google Drive links in the MethylGPT README."
        )

    probe_csv = os.environ.get('KIRBY_METHYLGPT_PROBE_CSV') or os.path.join(
        os.path.dirname(os.path.abspath(model_dir)), 'data', 'probe_ids_type3.csv')
    if not os.path.exists(probe_csv):
        raise RuntimeError(
            f"MethylGPT probe id file not found at {probe_csv}. It is "
            f"data/probe_ids_type3.csv in the MethylGPT repository. Set "
            f"KIRBY_METHYLGPT_PROBE_CSV."
        )

    if _METHYLGPT_MODEL is None:
        vocab = MethylVocab(
            probe_id_dir=probe_csv,
            pad_token='<pad>',
            special_tokens=['<pad>', '<cls>', '<eoc>'],
            save_dir=None,
        )
        with open(args_json) as f:
            config = json.load(f)
        config['load_model'] = True
        config['pretrained_file'] = weights
        config['mask_ratio'] = 0.0          # frozen encoding, not a masked view
        model = MethylGPTModel.from_pretrained(config, vocab)
        model.eval()
        device = _device()
        model = model.to(device)
        if device == 'cuda':
            model = model.half()
        _METHYLGPT_MODEL = model
        _METHYLGPT_VOCAB = vocab
        print(f"    Loaded MethylGPT from {model_dir} (device={device})")

    model = _METHYLGPT_MODEL
    vocab = _METHYLGPT_VOCAB
    device = next(model.parameters()).device
    model.config['mask_ratio'] = 0.0

    if not hasattr(vocab, 'CpG_ids'):
        raise RuntimeError(
            "The loaded MethylVocab has no CpG_ids attribute, which prepare_data "
            "requires. Check that KIRBY_METHYLGPT_PROBE_CSV points at "
            "probe_ids_type3.csv."
        )
    model_cpgs = [str(c) for c in vocab.CpG_ids]
    n_vocab = len(model_cpgs)
    pad_value = float(model.config.get('pad_value', -2))

    pos_of = {c: i for i, c in enumerate(model_cpgs)}
    aligned = np.full((n_samples, n_vocab), pad_value, dtype=np.float32)
    mapped = 0
    for col, probe in enumerate(cpg_ids):
        pos = pos_of.get(str(probe))
        if pos is None:
            continue
        aligned[:, pos] = beta[:, col]
        mapped += 1
    print(f"    Mapped {mapped}/{len(cpg_ids)} probes onto MethylGPT's {n_vocab} CpGs")

    embeddings = None
    n_zero_rows = 0
    batch_size = int(os.environ.get('KIRBY_METHYLGPT_BATCH', '16'))

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            try:
                batch = {
                    'id': [str(i) for i in range(start, end)],
                    'data': torch.from_numpy(aligned[start:end]).float(),
                }
                prepared = model.prepare_data(batch)
                gene_ids = prepared['gene_ids'].to(device)
                values = prepared['values'].to(device)
                values = values.to(next(model.parameters()).dtype)
                cell_embs = model.get_cell_embeddings(gene_ids, values)
                cell_embs = cell_embs.float().cpu().numpy()
                if embeddings is None:
                    embeddings = np.zeros((n_samples, cell_embs.shape[1]),
                                          dtype=np.float32)
                embeddings[start:end] = cell_embs
            except Exception as e:
                print(f"    WARNING: MethylGPT failed on samples {start}-{end}: {e}")
                n_zero_rows += (end - start)
                if embeddings is None:
                    raise RuntimeError(
                        f"MethylGPT failed on the first batch, so the embedding width "
                        f"is unknown and no matrix can be returned: {e}"
                    )

    return _finish(embeddings, n_samples, n_zero_rows, 'methyl_methylgpt')


def create_methyl_cpgpt_embeddings(beta: np.ndarray,
                                   cpg_ids: np.ndarray) -> np.ndarray:
    """CpGPT sample embeddings from a beta matrix. A second, independent METHYLATION block.

    SIGNATURE IS (beta, cpg_ids), NOT (expression, gene_names). beta is
    (n_samples, n_cpgs) of HM450 beta values in [0, 1]; cpg_ids is the matching array of
    Illumina probe IDs.

    MODEL. CpGPT (bioRxiv 2024-10-24), pretrained on 150,000+ samples from 2,000+
    studies with cross-platform reconstruction built in. Two pretrained sizes: 'small'
    (~2.5M parameters) and 'large' (~101M).

    THE BULK QUESTION DOES NOT ARISE. Trained on bulk array methylomes, so a TCGA bulk
    beta profile is the input it expects.

    PROBE MAPPING. CpGPT's checkpoint ships a vocab JSON whose 'input' list is the set of
    probe IDs it accepts. The dataset's beta matrix is restricted to that intersection —
    the tutorial's own step — and written to an Arrow file, keeping every sample ROW and
    dropping only probe COLUMNS. The package then locates each probe's genomic position
    and looks up its DNA language-model embedding.

    POOLING. Done inside the package: predict_mode='forward' with
    return_keys=['sample_embedding'] returns one aggregated vector per sample, from a
    learnable classification token over the CpG sequence. The published entry does not
    state the width and it varies by checkpoint, so the width returned is used as-is.

    RE-VERIFIED 2026-09-10 against tutorials/quick_setup.ipynb: download_cpgpt returning
    an object with .dependencies_path/.checkpoint_path/.config_path/.vocab_path, the
    CpGPTInferencer / load_cpgpt_config / load_cpgpt_model(strict_load=True) sequence,
    the DNALLMEmbedder / IlluminaMethylationProber / CpGPTDataSaver.process_files
    sequence and every CpGPTDataModule keyword below all match the tutorial. The
    tutorial's own predict call uses return_keys=['pred_conditions'] and names other
    keys only generically, so 'sample_embedding' as a key name is still UNCONFIRMED
    alongside the row order.

    ROW ORDER CAVEAT, UNVERIFIED. The rows come back in the order the CpGPT datamodule
    yields them. This builder writes the Arrow file in the input order, uses batch_size 1
    and no shuffling, and checks that the row count matches — but that the datamodule
    preserves file order was NOT confirmed from the package source in the session that
    wrote this. If CpGPT is ever run for a real c13 cell, confirm the order first, for
    instance by encoding two obviously different samples and checking which row moves.

    API SOURCE. https://github.com/lcamillo/CpGPT — README (pip install CpGPT,
    download_cpgpt, the model zoo table) and tutorials/quick_setup.ipynb (the
    CpGPTInferencer / DNALLMEmbedder / IlluminaMethylationProber / CpGPTDataSaver /
    CpGPTDataModule / CpGPTTrainer.predict sequence), both read in the session that
    wrote this function.

    SETUP REQUIRED:
      pip install CpGPT
      The first call downloads the checkpoint, config, vocab and the human DNA
      language-model dependencies via download_cpgpt(model=..., species='human'); that
      download is large. Set KIRBY_CPGPT_MODEL to 'small' or 'large' (default 'small')
      and KIRBY_CPGPT_DIR to a writable working directory for the dependency and
      processed caches.
      Trainer precision is left at the package default of 16-mixed, which the README
      says is required to reproduce the published behaviour.

    LICENSE. MIT, with a carve-out forbidding use in biomarkers-of-aging prize
    competitions.

    Args:
        beta: (n_samples, n_cpgs) methylation beta values
        cpg_ids: (n_cpgs,) Illumina probe IDs

    Returns:
        (n_samples, D) float32
    """
    import json
    import tempfile

    n_samples = int(beta.shape[0])

    try:
        from cpgpt import download_cpgpt
        from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
        from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        from cpgpt.data.components.illumina_methylation_prober import (
            IlluminaMethylationProber)
        from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
        from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer
    except ImportError as e:
        raise ImportError(
            f"CpGPT is not installed ({e}). pip install CpGPT — then the first call "
            f"downloads weights and human dependencies from HuggingFace "
            f"(lucascamillomd/cpgpt-models and lucascamillomd/cpgpt-human-dependencies)."
        )
    import pandas as pd

    model_name = os.environ.get('KIRBY_CPGPT_MODEL', 'small')
    work_dir = os.environ.get('KIRBY_CPGPT_DIR') or tempfile.mkdtemp(prefix='cpgpt_')
    deps_dir = os.path.join(work_dir, 'dependencies')
    data_dir = os.path.join(work_dir, 'data')
    processed_dir = os.path.join(work_dir, 'processed')
    for d in (deps_dir, data_dir, processed_dir):
        os.makedirs(d, exist_ok=True)

    try:
        resources = download_cpgpt(model=model_name, species='human')
    except Exception as e:
        raise RuntimeError(
            f"CpGPT resource download failed for model '{model_name}': {e}. "
            f"Weights: https://huggingface.co/lucascamillomd/cpgpt-models ; human "
            f"dependencies: https://huggingface.co/lucascamillomd/cpgpt-human-dependencies"
        )

    llm_deps_dir = str(resources.dependencies_path)
    ckpt_path = str(resources.checkpoint_path)
    config_path = str(resources.config_path)
    vocab_path = str(resources.vocab_path) if resources.vocab_path is not None else None
    if vocab_path is None:
        raise RuntimeError(
            f"The CpGPT '{model_name}' resources carry no vocab file, so the beta "
            f"matrix cannot be restricted to the probes the model accepts."
        )

    inferencer = CpGPTInferencer(dependencies_dir=deps_dir, data_dir=data_dir)
    config = inferencer.load_cpgpt_config(config_path)
    model = inferencer.load_cpgpt_model(config, model_ckpt_path=ckpt_path,
                                        strict_load=True)

    with open(vocab_path) as f:
        vocab = json.load(f)
    accepted = set(str(c) for c in vocab['input'])
    keep = [i for i, c in enumerate(cpg_ids) if str(c) in accepted]
    if not keep:
        raise RuntimeError(
            f"None of the {len(cpg_ids)} probes is in the CpGPT '{model_name}' input "
            f"vocabulary, so there is nothing to encode."
        )
    print(f"    Kept {len(keep)}/{len(cpg_ids)} probes that CpGPT accepts")

    df = pd.DataFrame(np.asarray(beta, dtype=np.float32)[:, keep],
                      columns=[str(cpg_ids[i]) for i in keep])
    arrow_path = os.path.join(data_dir, 'kirby_betas.arrow')
    df.to_feather(arrow_path)

    embedder = DNALLMEmbedder(dependencies_dir=llm_deps_dir)
    prober = IlluminaMethylationProber(dependencies_dir=llm_deps_dir, embedder=embedder)
    saver = CpGPTDataSaver(data_paths=arrow_path, processed_dir=processed_dir)
    saver.process_files(prober, embedder)

    datamodule = CpGPTDataModule(
        predict_dir=processed_dir,
        dependencies_dir=llm_deps_dir,
        batch_size=1,
        num_workers=0,
        max_length=int(os.environ.get('KIRBY_CPGPT_MAX_LENGTH', '20000')),
        dna_llm=config.data.dna_llm,
        dna_context_len=config.data.dna_context_len,
        sorting_strategy=config.data.sorting_strategy,
        pin_memory=False,
    )

    trainer = CpGPTTrainer()   # precision fixed to 16-mixed by default, per the README
    out = trainer.predict(model=model, datamodule=datamodule,
                          predict_mode='forward', return_keys=['sample_embedding'])

    emb = out['sample_embedding'] if isinstance(out, dict) else out
    if hasattr(emb, 'detach'):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 3 and emb.shape[1] == 1:
        emb = emb[:, 0, :]
    if emb.shape[0] != n_samples:
        raise RuntimeError(
            f"CpGPT returned {emb.shape[0]} rows for {n_samples} samples. Refusing to "
            f"return a matrix whose rows do not line up with the labels."
        )

    return _finish(emb, n_samples, 0, 'methyl_cpgpt')


def create_methyl_methylgpt_from_samples(sample_ids: List[str],
                                         cohort: str = '') -> np.ndarray:
    """Adapter so the TCGA dispatch can call MethylGPT with (sample_ids, cohort).

    Loads the HM450 beta matrix for these samples with the same loader that
    methyl_pca50 and methyl_hvg already use, then calls
    create_methyl_methylgpt_embeddings. Sample order is the order of sample_ids.
    """
    from kirby.representations.genomic import _load_methylation_for_samples
    beta, probe_names = _load_methylation_for_samples(sample_ids, cohort)
    return create_methyl_methylgpt_embeddings(beta, np.asarray(probe_names))


def create_methyl_cpgpt_from_samples(sample_ids: List[str],
                                     cohort: str = '') -> np.ndarray:
    """Adapter so the TCGA dispatch can call CpGPT with (sample_ids, cohort).

    Loads the HM450 beta matrix for these samples with the same loader that
    methyl_pca50 and methyl_hvg already use, then calls
    create_methyl_cpgpt_embeddings. Sample order is the order of sample_ids.
    """
    from kirby.representations.genomic import _load_methylation_for_samples
    beta, probe_names = _load_methylation_for_samples(sample_ids, cohort)
    return create_methyl_cpgpt_embeddings(beta, np.asarray(probe_names))


# =============================================================================
# TAHOE-X1
# =============================================================================

_TAHOE_X1 = None


def create_tahoe_x1_embeddings(expression: np.ndarray,
                               gene_names: np.ndarray) -> np.ndarray:
    """Tahoe-x1 cell embeddings, one row per bulk sample.

    MODEL. Tahoe-x1 (bioRxiv 2025-10-23, Tahoe Therapeutics), pretrained on 266 million
    cells including the Tahoe-100M drug perturbation compendium. Three sizes, with the
    hidden width read off each checkpoint's model_config.yml on HuggingFace:
    70m -> 512, 1b -> 2048, 3b -> 2560. Default here is 70m; set KIRBY_TAHOE_X1_SIZE to
    '1b' or '3b'.

    BULK INPUT IS A STRETCH. Pretrained on single cells; a bulk TCGA profile is fed as
    one pseudo-cell into a 2048-gene context (seq_len_dataset 2048 in the repo's own
    inference config), so the sample enters as a truncated view of its transcriptome,
    not the whole thing.

    GENE MAPPING. The repo's inference script keys genes by Ensembl ID
    (data.gene_id_key = 'ensembl_id') and maps each to a vocabulary id, dropping genes
    the vocabulary does not know. The dataset's HUGO symbols are mapped to Ensembl IDs
    through the Geneformer symbol dictionary and written into adata.var['ensembl_id'];
    adata.var['id_in_vocab'] is then filled the way predict_embeddings.py fills it, and
    genes with id -1 are dropped as COLUMNS.

    POOLING. Done inside the package: get_batch_embeddings returns one cell vector per
    row of the AnnData, in AnnData row order — it consumes the whole AnnData in one call
    and never filters cells.

    INPUT VALUES. The model wants counts; log2(TPM+1) is converted back to TPM and
    passed as the count-like matrix.

    API SOURCE. https://huggingface.co/tahoebio/Tahoe-x1 (model card: ComposerTX.from_hf
    and get_batch_embeddings), plus https://github.com/tahoebio/tahoe-x1 files
    tahoe_x1/tasks/emb_extractor.py (the get_batch_embeddings signature),
    tahoe_x1/tasks/__init__.py and scripts/inference/predict_embeddings.py (the
    id_in_vocab mapping), all read in the session that wrote this function. The card
    prints one import as 'tahoex.tasks'; the repository's actual package directory is
    'tahoe_x1', so both are tried.

    SETUP REQUIRED:
      uv pip install tahoe-x1 --no-build-isolation-package flash-attn
      (or git clone https://github.com/tahoebio/tahoe-x1 and pip install -e .)
      Weights download from HuggingFace tahoebio/Tahoe-x1 on first call.

    LICENSE. Apache 2.0.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, D) float32; D is 512 / 2048 / 2560 for 70m / 1b / 3b.
    """
    global _TAHOE_X1

    n_samples = int(expression.shape[0])
    size = os.environ.get('KIRBY_TAHOE_X1_SIZE', '70m')

    ComposerTX = None
    get_batch_embeddings = None
    try:
        from tahoe_x1.model import ComposerTX
        from tahoe_x1.tasks import get_batch_embeddings
    except ImportError:
        try:
            from tahoex.model import ComposerTX
            from tahoex.tasks import get_batch_embeddings
        except ImportError as e:
            raise ImportError(
                f"Tahoe-x1 is not installed ({e}). "
                f"uv pip install tahoe-x1 --no-build-isolation-package flash-attn, or "
                f"git clone https://github.com/tahoebio/tahoe-x1 && pip install -e ."
            )

    if _TAHOE_X1 is None:
        try:
            model, vocab, model_cfg, collator_cfg = ComposerTX.from_hf(
                repo_id='tahoebio/Tahoe-x1', model_size=size)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Tahoe-x1 '{size}' from tahoebio/Tahoe-x1: {e}. "
                f"Weights: https://huggingface.co/tahoebio/Tahoe-x1"
            )
        _TAHOE_X1 = (model, vocab, model_cfg, collator_cfg)
        print(f"    Loaded Tahoe-x1 ({size}, d_model={model_cfg.get('d_model')})")
    model, vocab, model_cfg, collator_cfg = _TAHOE_X1

    tpm = _log2tpm_to_tpm(expression)
    ensembl = _ensembl_for(gene_names)
    keep = [i for i, e in enumerate(ensembl) if e is not None]
    if not keep:
        raise RuntimeError(
            "No dataset gene symbol mapped to an Ensembl ID, so Tahoe-x1 has no input."
        )
    adata = _build_anndata(tpm[:, keep], [ensembl[i] for i in keep],
                           var_key='ensembl_id',
                           obs_names=[f's{i}' for i in range(n_samples)])

    adata.var['id_in_vocab'] = [
        vocab[g] if g in vocab else -1 for g in adata.var['ensembl_id']
    ]
    in_vocab = np.asarray(adata.var['id_in_vocab']) >= 0
    print(f"    {int(in_vocab.sum())}/{len(keep)} Ensembl IDs are in the Tahoe-x1 vocabulary")
    if not in_vocab.any():
        raise RuntimeError(
            "No gene is in the Tahoe-x1 vocabulary, so there is nothing to encode."
        )
    adata = adata[:, in_vocab].copy()
    gene_ids = np.array([vocab[g] for g in adata.var['ensembl_id']], dtype=int)

    cell_embs = get_batch_embeddings(
        adata=adata,
        model=model,
        vocab=vocab,
        model_cfg=model_cfg,
        collator_cfg=collator_cfg,
        gene_ids=gene_ids,
        batch_size=int(os.environ.get('KIRBY_TAHOE_X1_BATCH', '8')),
        num_workers=0,
        max_length=int(os.environ.get('KIRBY_TAHOE_X1_MAXLEN', '2048')),
        return_gene_embeddings=False,
    )
    if isinstance(cell_embs, tuple):
        cell_embs = cell_embs[0]
    emb = np.asarray(cell_embs, dtype=np.float32)

    return _finish(emb, n_samples, 0, 'tahoe_x1')


# =============================================================================
# C2S-SCALE — a text LLM reading the profile as a sentence of gene names
# =============================================================================

_C2S_MODEL = None
_C2S_TOKENIZER = None


def create_c2s_scale_embeddings(expression: np.ndarray,
                                gene_names: np.ndarray) -> np.ndarray:
    """C2S-Scale (Gemma-2) embeddings: the profile written as a sentence of gene names.

    MODEL. C2S-Scale (bioRxiv 2025-04-14; Yale van Dijk lab with Google Research and
    Google DeepMind), Gemma-2 instruction-tuned on 57M+ cells rewritten as "cell
    sentences". Sizes 1B / 2B / 9B / 27B. Default here is the 27B checkpoint,
    vandijklab/C2S-Scale-Gemma-2-27B, whose config.json gives hidden_size 4608, so one
    row is 4608 features. Set KIRBY_C2S_MODEL to a smaller checkpoint from the
    vandijklab/cell2sentence-models collection to cut the cost.

    COST. THIS IS THE EXPENSIVE ONE: a 27B-parameter forward pass PER SAMPLE, over a
    sentence of ~1000 gene symbols. On a single A100 that is on the order of a second
    per sample once the ~55 GB of bf16 weights are resident, and the weights do not fit
    on a 40 GB card without sharding or a smaller checkpoint. For a TCGA cohort of about
    1000 samples, budget the download (tens of GB), the load, and a pass that runs in
    minutes to tens of minutes on one large GPU — and note the 9B or 2B checkpoint
    answers the same c13 question at a fraction of that.

    BULK INPUT IS A STRETCH, AND SO IS THE EVIDENCE. Pretrained on single cells. Its
    published number (95.43% cell type annotation) measures the accuracy of GENERATED
    TEXT naming a cell type, not the quality of the mean-pooled hidden state this
    function builds — the number does not describe this object.

    GENE MAPPING. There is no vocabulary to map onto: the profile becomes a
    space-separated list of the sample's top-expressed HUGO SYMBOLS in descending
    expression order, which is what a cell sentence is. Symbols go in as the dataset
    writes them. The number of genes in the sentence is 1000 by default
    (KIRBY_C2S_N_GENES); the model card says to use at least 200 for inference.

    POOLING. Attention-mask-weighted MEAN over the final hidden state across the tokens
    of the cell sentence. No task prompt is prepended — the prompt in the model card is
    for generation, and adding it would put the same instruction text in every row.

    API SOURCE. https://huggingface.co/vandijklab/C2S-Scale-Gemma-2-27B — the model
    card's own "Running the model on predictive tasks" snippet (AutoTokenizer /
    AutoModelForCausalLM, and the cell-sentence definition), plus that repo's
    config.json for hidden_size, both read in the session that wrote this function. The
    card shows generation; output_hidden_states pooling is the standard transformers
    route on the same loaded model, not a separate API.

    SETUP REQUIRED:
      pip install accelerate transformers sentencepiece torch
      Weights download from HuggingFace on first call. Not gated: the repo's config.json
      serves without a token.

    LICENSE. CC-BY-4.0.

    Args:
        expression: (n_samples, n_genes) log2(TPM+1) matrix
        gene_names: (n_genes,) HUGO gene symbols

    Returns:
        (n_samples, 4608) float32 for the 27B checkpoint; the loaded model's hidden size
        in general.
    """
    global _C2S_MODEL, _C2S_TOKENIZER

    n_samples = int(expression.shape[0])

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            f"C2S-Scale requires torch and transformers ({e}): "
            f"pip install accelerate transformers sentencepiece torch"
        )

    model_id = os.environ.get('KIRBY_C2S_MODEL', 'vandijklab/C2S-Scale-Gemma-2-27B')
    n_genes_in_sentence = int(os.environ.get('KIRBY_C2S_N_GENES', '1000'))

    if _C2S_MODEL is None:
        device = _device()
        try:
            _C2S_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            kwargs = {}
            if device == 'cuda':
                kwargs['dtype'] = torch.bfloat16
                kwargs['device_map'] = 'auto'
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if device != 'cuda':
                model = model.to(device)
            model.eval()
            _C2S_MODEL = model
        except Exception as e:
            raise RuntimeError(
                f"Could not load C2S-Scale '{model_id}': {e}. Weights: "
                f"https://huggingface.co/{model_id} — needs transformers, accelerate "
                f"and sentencepiece, and the 27B checkpoint needs a large GPU. Set "
                f"KIRBY_C2S_MODEL to a smaller checkpoint from the "
                f"vandijklab/cell2sentence-models collection."
            )
        print(f"    Loaded C2S-Scale {model_id} "
              f"(hidden {_C2S_MODEL.config.hidden_size}D)")

    model = _C2S_MODEL
    tokenizer = _C2S_TOKENIZER
    hidden = int(model.config.hidden_size)
    device = next(model.parameters()).device

    symbols = np.array([str(g) for g in gene_names])
    embeddings = np.zeros((n_samples, hidden), dtype=np.float32)
    n_zero_rows = 0
    k = min(n_genes_in_sentence, symbols.shape[0])

    with torch.no_grad():
        for i in range(n_samples):
            try:
                row = np.asarray(expression[i], dtype=np.float64)
                order = np.argsort(-row)[:k]
                sentence = ' '.join(symbols[order].tolist())
                enc = tokenizer(sentence, return_tensors='pt', truncation=True,
                                max_length=int(os.environ.get(
                                    'KIRBY_C2S_MAX_TOKENS', '4096')))
                enc = {kk: v.to(device) for kk, v in enc.items()}
                out = model(**enc, output_hidden_states=True)
                last = out.hidden_states[-1].float()
                mask = enc['attention_mask'].unsqueeze(-1).float()
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings[i] = pooled[0].cpu().numpy()
            except Exception as e:
                print(f"    WARNING: C2S-Scale failed on sample {i}: {e}")
                n_zero_rows += 1
            if (i + 1) % 50 == 0:
                print(f"      Progress: {i + 1}/{n_samples}")

    return _finish(embeddings, n_samples, n_zero_rows, 'c2s_scale')


# =============================================================================
# WHAT c13 CAN REQUEST FROM THIS FILE
# =============================================================================
# name              -> function                                  status
# bulkformer        -> create_bulkformer_embeddings              IMPLEMENTED
# transcriptformer  -> create_transcriptformer_embeddings        IMPLEMENTED
# cellfm            -> create_cellfm_embeddings                  STUB, raises
# scprint2          -> create_scprint2_embeddings                IMPLEMENTED
# aido_cell         -> create_aido_cell_embeddings               IMPLEMENTED
# methyl_methylgpt  -> create_methyl_methylgpt_from_samples      IMPLEMENTED (beta input)
# methyl_cpgpt      -> create_methyl_cpgpt_from_samples          IMPLEMENTED (beta input)
# tahoe_x1          -> create_tahoe_x1_embeddings                IMPLEMENTED
# c2s_scale         -> create_c2s_scale_embeddings               IMPLEMENTED

# The keys are the bare names c13 asks for: forward_compat_reps.py registers the
# omics_transcriptomics block under the tcga domain with an empty prefix, so the name
# that reaches the loader is the entry key itself.
FORWARD_COMPAT_OMICS_REPS = {
    'bulkformer':       ('create_bulkformer_embeddings',         'forward_compat_omics'),
    'transcriptformer': ('create_transcriptformer_embeddings',   'forward_compat_omics'),
    'cellfm':           ('create_cellfm_embeddings',             'forward_compat_omics'),
    'scprint2':         ('create_scprint2_embeddings',           'forward_compat_omics'),
    'aido_cell':        ('create_aido_cell_embeddings',          'forward_compat_omics'),
    'tahoe_x1':         ('create_tahoe_x1_embeddings',           'forward_compat_omics'),
    'c2s_scale':        ('create_c2s_scale_embeddings',          'forward_compat_omics'),
    'methyl_methylgpt': ('create_methyl_methylgpt_from_samples', 'forward_compat_omics_methyl'),
    'methyl_cpgpt':     ('create_methyl_cpgpt_from_samples',     'forward_compat_omics_methyl'),
}
