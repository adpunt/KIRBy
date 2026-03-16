#!/usr/bin/env python
"""Pre-download all HuggingFace / pip-managed model weights to local cache.

Run this on the LOGIN NODE (which has internet) before submitting SLURM jobs
to compute nodes (which may not have internet access).

Usage:
    python tests/slurm_scripts/precache_models.py          # everything
    python tests/slurm_scripts/precache_models.py protein   # protein models only
    python tests/slurm_scripts/precache_models.py antibody  # antibody models only
    python tests/slurm_scripts/precache_models.py molecular # molecular models only
    python tests/slurm_scripts/precache_models.py genomic   # genomic/TCGA models only
    python tests/slurm_scripts/precache_models.py reaction  # reaction models only
"""
import sys
import importlib


def _try(name, fn):
    """Run fn, print success/failure."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"  OK: {name}")
    except Exception as e:
        print(f"  SKIP: {name} — {type(e).__name__}: {e}")


# ── Protein models ──────────────────────────────────────────────────────────

def cache_protein():
    """Pre-cache all protein representation models."""

    def _esm2():
        from transformers import AutoTokenizer, AutoModel
        name = 'facebook/esm2_t33_650M_UR50D'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    def _esmc():
        from esm.models.esmc import ESMC
        print("    Downloading ESMC 600M...")
        ESMC.from_pretrained("esmc_600m")

    def _esm3():
        from esm.models.esm3 import ESM3
        print("    Downloading ESM3 open v1...")
        ESM3.from_pretrained("esm3_sm_open_v1")

    def _esmfold():
        from transformers import AutoTokenizer, EsmForProteinFolding
        name = 'facebook/esmfold_v1'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        EsmForProteinFolding.from_pretrained(name)

    def _aido():
        from transformers import AutoTokenizer, AutoModel
        name = 'genbio-ai/AIDO.Protein-16B'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        AutoModel.from_pretrained(name, trust_remote_code=True)

    def _prostt5():
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        name = 'Rostlab/ProstT5'
        print(f"    Downloading {name}...")
        T5Tokenizer.from_pretrained(name)
        T5ForConditionalGeneration.from_pretrained(name)

    def _saprot():
        from transformers import EsmTokenizer, EsmModel
        name = 'westlake-repl/SaProt_650M_AF2'
        print(f"    Downloading {name}...")
        EsmTokenizer.from_pretrained(name)
        EsmModel.from_pretrained(name)

    def _amplify():
        from transformers import AutoTokenizer, AutoModel
        name = 'chandar-lab/AMPLIFY_350M'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    def _xtrimopglm():
        from transformers import AutoTokenizer, AutoModel
        name = 'biomap-research/proteinglm-1b-mlm'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        AutoModel.from_pretrained(name, trust_remote_code=True)

    _try("ESM-2 650M", _esm2)
    _try("ESM-C 600M", _esmc)
    _try("ESM3 open v1", _esm3)
    _try("ESMFold v1", _esmfold)
    _try("AIDO.Protein-16B", _aido)
    _try("ProstT5", _prostt5)
    _try("SaProt 650M", _saprot)
    _try("AMPLIFY 350M", _amplify)
    _try("xTrimoPGLM-1B", _xtrimopglm)


# ── Antibody models ─────────────────────────────────────────────────────────

def cache_antibody():
    """Pre-cache all antibody representation models."""

    def _igbert():
        from transformers import AutoTokenizer, AutoModel
        name = 'Exscientia/IgBert'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    def _igt5():
        from transformers import T5Tokenizer, T5EncoderModel
        name = 'Exscientia/IgT5'
        print(f"    Downloading {name}...")
        T5Tokenizer.from_pretrained(name, do_lower_case=False)
        T5EncoderModel.from_pretrained(name)

    def _prot_t5():
        from transformers import T5Tokenizer, T5EncoderModel
        name = 'Rostlab/prot_t5_xl_uniref50'
        print(f"    Downloading {name}...")
        T5Tokenizer.from_pretrained(name, do_lower_case=False)
        T5EncoderModel.from_pretrained(name)

    def _abroberta():
        from transformers import AutoTokenizer, AutoModel
        name = 'mogam-ai/Ab-RoBERTa'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        AutoModel.from_pretrained(name, trust_remote_code=True)

    def _antiberty():
        from antiberty import AntiBERTyRunner
        print("    Initialising AntiBERTy (downloads on first use)...")
        AntiBERTyRunner()

    # ESM-2, ESM-C, ESM3, SaProt are shared with protein — already cached above
    _try("IgBERT", _igbert)
    _try("IgT5", _igt5)
    _try("ProtT5 XL", _prot_t5)
    _try("Ab-RoBERTa", _abroberta)
    _try("AntiBERTy", _antiberty)


# ── Molecular models ────────────────────────────────────────────────────────

def cache_molecular():
    """Pre-cache molecular representation models."""

    def _chemberta():
        from transformers import AutoTokenizer, AutoModel
        name = 'DeepChem/ChemBERTa-77M-MTR'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    def _molformer():
        from transformers import AutoTokenizer, AutoModel
        name = 'ibm/MoLFormer-XL-both-10pct'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        AutoModel.from_pretrained(name, trust_remote_code=True)

    def _selformer():
        from transformers import AutoTokenizer, AutoModel
        name = 'HUBioDataLab/SELFormer'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    def _chembert():
        from transformers import AutoTokenizer, AutoModel
        name = 'jonghyunlee/ChemBERT_ChEMBL_pretrained'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)

    _try("ChemBERTa-77M", _chemberta)
    _try("MolFormer-XL", _molformer)
    _try("SELFormer", _selformer)
    _try("ChemBERT", _chembert)


# ── Genomic / TCGA models ──────────────────────────────────────────────────

def cache_genomic():
    """Pre-cache genomic/TCGA representation models."""

    def _geneformer():
        from transformers import AutoConfig, AutoModel
        name = 'ctheodoris/Geneformer'
        print(f"    Downloading {name}...")
        AutoConfig.from_pretrained(name, trust_remote_code=True)
        AutoModel.from_pretrained(name, trust_remote_code=True)

    def _bulkrnabert():
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download
        name = 'InstaDeepAI/BulkRNABert'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)
        # Also download the gene order file
        hf_hub_download(repo_id=name, filename='gene_to_ens.csv')

    def _scfoundation():
        from transformers import AutoModel
        name = 'biomap-research/scFoundation'
        print(f"    Downloading {name}...")
        AutoModel.from_pretrained(name, trust_remote_code=True)

    _try("Geneformer", _geneformer)
    _try("BulkRNABert", _bulkrnabert)
    _try("scFoundation", _scfoundation)


# ── Reaction models ─────────────────────────────────────────────────────────

def cache_reaction():
    """Pre-cache reaction representation models."""

    def _reactiont5():
        from transformers import AutoTokenizer, T5EncoderModel
        name = 'sagawa/ReactionT5v2-forward'
        print(f"    Downloading {name}...")
        AutoTokenizer.from_pretrained(name)
        T5EncoderModel.from_pretrained(name)

    def _rxnfp():
        # rxnfp auto-downloads its BERT model on first import
        print("    Importing rxnfp (downloads RXNBERT on first use)...")
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
        )
        model, tokenizer = get_default_model_and_tokenizer()

    _try("ReactionT5", _reactiont5)
    _try("RXNFP/RXNBERT", _rxnfp)


# ── Main ────────────────────────────────────────────────────────────────────

GROUPS = {
    'protein': cache_protein,
    'antibody': cache_antibody,
    'molecular': cache_molecular,
    'genomic': cache_genomic,
    'reaction': cache_reaction,
}

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if mode == 'all':
        for name, fn in GROUPS.items():
            print(f"\n\n{'#'*60}")
            print(f"#  {name.upper()} MODELS")
            print(f"{'#'*60}")
            fn()
    elif mode in GROUPS:
        GROUPS[mode]()
    else:
        print(f"Unknown mode: {mode}")
        print(f"Usage: python {sys.argv[0]} [{'|'.join(['all'] + list(GROUPS))}]")
        sys.exit(1)

    print(f"\n\n{'='*60}")
    print("  Pre-caching complete. Models saved to ~/.cache/huggingface/")
    print(f"{'='*60}")
