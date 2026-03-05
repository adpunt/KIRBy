"""
Fine-tuning integration for KIRBy.

Provides factory constructors that produce per-fold fine-tuned embeddings.
Each factory is a closure that, given (train_idx, val_idx, train_labels),
returns (train_embeddings, val_embeddings) as numpy arrays.

Usage:
    from kirby.finetuning import make_geneformer_factory

    factory = make_geneformer_factory(expression, gene_names, ...)
    train_embs, val_embs = factory(train_idx, val_idx, train_labels)
"""

from kirby.finetuning._base import get_device, get_cache_dir, fold_cache_key, load_or_compute

__all__ = [
    'get_device',
    'get_cache_dir',
    'fold_cache_key',
    'load_or_compute',
]

# Lazy imports for domain-specific factories (avoid importing torch at module level)


def make_molecular_lora_factory(*args, **kwargs):
    from kirby.finetuning.molecular_lora import make_molecular_lora_factory as _f
    return _f(*args, **kwargs)


def make_geneformer_factory(*args, **kwargs):
    from kirby.finetuning.tcga_geneformer import make_geneformer_factory as _f
    return _f(*args, **kwargs)


def make_molecular_freeze_factory(*args, **kwargs):
    from kirby.finetuning.molecular_freeze import make_molecular_freeze_factory as _f
    return _f(*args, **kwargs)


def make_molecular_gnn_factory(*args, **kwargs):
    from kirby.finetuning.molecular_gnn import make_molecular_gnn_factory as _f
    return _f(*args, **kwargs)


def make_molecular_contrastive_factory(*args, **kwargs):
    from kirby.finetuning.molecular_contrastive import make_molecular_contrastive_factory as _f
    return _f(*args, **kwargs)


def make_molecular_grover_factory(*args, **kwargs):
    from kirby.finetuning.molecular_grover import make_molecular_grover_factory as _f
    return _f(*args, **kwargs)


def make_scfoundation_factory(*args, **kwargs):
    from kirby.finetuning.tcga_scfoundation import make_scfoundation_factory as _f
    return _f(*args, **kwargs)


def make_dta_protein_lora_factory(*args, **kwargs):
    from kirby.finetuning.dta_protein_lora import make_dta_protein_lora_factory as _f
    return _f(*args, **kwargs)


def make_antibody_lora_factory(*args, **kwargs):
    from kirby.finetuning.antibody_lora import make_antibody_lora_factory as _f
    return _f(*args, **kwargs)


def make_molecular_3d_factory(*args, **kwargs):
    from kirby.finetuning.molecular_3d import make_molecular_3d_factory as _f
    return _f(*args, **kwargs)
