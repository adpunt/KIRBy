"""
Fine-tuning integration for KIRBy.

Provides factory constructors that produce per-fold fine-tuned embeddings.
Each factory is a closure that, given (train_idx, val_idx, train_labels),
returns (train_embeddings, val_embeddings) as numpy arrays.

Usage:
    from kirby.finetuning import make_antibody_lora_factory

    factory = make_antibody_lora_factory(seqs, mutations, dataset_name='my_dms')
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


def make_antibody_lora_factory(*args, **kwargs):
    from kirby.finetuning.antibody_lora import make_antibody_lora_factory as _f
    return _f(*args, **kwargs)


def make_molecular_lora_factory(*args, **kwargs):
    from kirby.finetuning.molecular_lora import make_molecular_lora_factory as _f
    return _f(*args, **kwargs)


def make_antibody_mlm_factory(*args, **kwargs):
    from kirby.finetuning.antibody_mlm import make_antibody_mlm_factory as _f
    return _f(*args, **kwargs)


def make_geneformer_factory(*args, **kwargs):
    from kirby.finetuning.tcga_geneformer import make_geneformer_factory as _f
    return _f(*args, **kwargs)
