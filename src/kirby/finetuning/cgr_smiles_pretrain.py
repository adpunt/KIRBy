"""CGR-SMILES corpus generation and continued MLM pretraining.

Extends ChemBERTa's tokenizer with ~30 dynamic bond tokens from chython's
CGR notation, then continues MLM pretraining on a CGR-SMILES corpus built
from USPTO-STEREO (~1M atom-mapped reactions).

Phase 4 of CGR_EMBEDDING_RESEARCH.md.

Usage (as CLI):
    # Full pipeline: corpus + pretrain
    python -m kirby.finetuning.cgr_smiles_pretrain \\
        --stage all --output-dir /path/to/output --n-workers 16

    # Corpus only (CPU, ~30 min on 16 cores for 1M reactions)
    python -m kirby.finetuning.cgr_smiles_pretrain \\
        --stage corpus --output-dir /path/to/output --max-reactions 1000000

    # Pretrain only (GPU, ~2-4 hours for 50K steps)
    python -m kirby.finetuning.cgr_smiles_pretrain \\
        --stage pretrain --output-dir /path/to/output --max-steps 50000

Requires: chython, transformers, torch, datasets
"""

import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional


# =========================================================================
# DYNAMIC BOND TOKENS
# =========================================================================

def get_dynamic_bond_tokens() -> List[str]:
    """Extract the 30 dynamic bond tokens from chython.

    Filters dyn_order_str to bracket-enclosed tokens only
    (excludes static bonds like '', '=', '#', ':', '~').

    Returns:
        Sorted list of 30 CGR dynamic bond tokens.
    """
    from chython.algorithms.smiles import dyn_order_str
    return sorted(v for v in dyn_order_str.values() if v and '[' in v)


# =========================================================================
# CORPUS GENERATION
# =========================================================================

def _rxn_to_cgr_smiles(mapped_rxn_smi: str) -> Optional[str]:
    """Convert one atom-mapped reaction SMILES to CGR-SMILES.

    Returns None on failure (unmapped atoms, invalid SMILES, etc).
    """
    from chython import smiles as chython_smiles
    try:
        rxn = chython_smiles(mapped_rxn_smi)
        if rxn is None:
            return None
        rxn.canonicalize()
        cgr = rxn.compose()
        return str(cgr)
    except Exception:
        return None


# Shared list for collecting error samples across workers
_ERROR_SAMPLES = []


def _worker_convert(batch: List[str]) -> List[Optional[str]]:
    """Worker function for multiprocessing CGR conversion."""
    return [_rxn_to_cgr_smiles(smi) for smi in batch]


def _diagnose_failures(rxn_smiles: List[str], n_samples: int = 5):
    """Print diagnostic info for CGR conversion failures.

    Tests a few reactions with full error tracing to identify the
    root cause of systematic failures.
    """
    from chython import smiles as chython_smiles

    print(f"\n  === CGR CONVERSION DIAGNOSTICS ({n_samples} samples) ===")
    for i, smi in enumerate(rxn_smiles[:n_samples]):
        print(f"\n  Sample {i+1}: {smi[:120]}{'...' if len(smi) > 120 else ''}")
        try:
            rxn = chython_smiles(smi)
            print(f"    parse: {type(rxn).__name__}")
            if rxn is None:
                print(f"    FAIL: chython_smiles() returned None")
                continue
            rxn.canonicalize()
            print(f"    canonicalize: OK")
            cgr = rxn.compose()
            print(f"    compose: OK -> {str(cgr)[:80]}")
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {e}")
    print()


def build_cgr_corpus(
    rxn_smiles: List[str],
    output_path: Path,
    n_workers: int = 8,
    chunk_size: int = 500,
) -> dict:
    """Convert reaction SMILES to CGR-SMILES corpus file.

    Uses multiprocessing for parallelism. Writes one CGR-SMILES per line.

    Args:
        rxn_smiles: List of atom-mapped reaction SMILES.
        output_path: Path to write corpus.txt.
        n_workers: Number of parallel workers.
        chunk_size: Reactions per worker chunk.

    Returns:
        Stats dict: {n_total, n_success, n_failed}.
    """
    # Fail fast if chython is not installed
    try:
        from chython import smiles as _check  # noqa: F401
    except ImportError:
        raise ImportError(
            "chython is required for CGR-SMILES conversion: "
            "pip install 'chython>=1.0' or pip install -e '.[reaction]'"
        )

    # Smoke test: try a single reaction before spawning workers
    print(f"  Smoke test: converting first reaction...")
    test_result = _rxn_to_cgr_smiles(rxn_smiles[0])
    if test_result is None:
        print(f"  Smoke test FAILED — running diagnostics...")
        _diagnose_failures(rxn_smiles, n_samples=5)
        raise RuntimeError(
            "CGR conversion smoke test failed on the first reaction. "
            "See diagnostics above. Common causes: reactions are not "
            "atom-mapped, chython version incompatibility, or data "
            "format issue."
        )
    print(f"  Smoke test OK: {test_result[:80]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split into chunks for multiprocessing
    chunks = [rxn_smiles[i:i + chunk_size]
              for i in range(0, len(rxn_smiles), chunk_size)]

    print(f"  Converting {len(rxn_smiles)} reactions to CGR-SMILES "
          f"({n_workers} workers, {len(chunks)} chunks)...")

    n_success = 0
    n_failed = 0
    tmp_path = output_path.with_suffix('.tmp')

    with open(tmp_path, 'w') as f:
        if n_workers > 1:
            with Pool(n_workers) as pool:
                for i, results in enumerate(
                        pool.imap(_worker_convert, chunks)):
                    for cgr_smi in results:
                        if cgr_smi is not None:
                            f.write(cgr_smi + '\n')
                            n_success += 1
                        else:
                            n_failed += 1
                    if (i + 1) % 100 == 0:
                        total = n_success + n_failed
                        print(f"    {total}/{len(rxn_smiles)} processed "
                              f"({n_success} success, {n_failed} failed)")
        else:
            for smi in rxn_smiles:
                cgr_smi = _rxn_to_cgr_smiles(smi)
                if cgr_smi is not None:
                    f.write(cgr_smi + '\n')
                    n_success += 1
                else:
                    n_failed += 1

    os.replace(str(tmp_path), str(output_path))

    stats = {
        'n_total': len(rxn_smiles),
        'n_success': n_success,
        'n_failed': n_failed,
        'success_rate': n_success / max(len(rxn_smiles), 1),
    }
    print(f"  Corpus: {n_success} CGR-SMILES "
          f"({stats['success_rate']:.1%} success rate)")
    return stats


# =========================================================================
# TOKENIZER EXTENSION
# =========================================================================

def extend_tokenizer(
    base_model: str = 'DeepChem/ChemBERTa-77M-MLM',
    save_dir: Optional[Path] = None,
):
    """Load ChemBERTa tokenizer and add 30 dynamic bond special tokens.

    Args:
        base_model: HuggingFace model ID.
        save_dir: Directory to save extended tokenizer. If None, not saved.

    Returns:
        (tokenizer, n_new_tokens) tuple.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    new_tokens = get_dynamic_bond_tokens()

    # Filter tokens already in vocabulary
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in new_tokens if t not in existing]

    n_added = tokenizer.add_special_tokens(
        {'additional_special_tokens': to_add})
    print(f"  Added {n_added} dynamic bond tokens to tokenizer "
          f"(vocab size: {len(tokenizer)})")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))

    return tokenizer, n_added


# =========================================================================
# MLM PRETRAINING
# =========================================================================

def pretrain_mlm(
    corpus_path: Path,
    output_dir: Path,
    base_model: str = 'DeepChem/ChemBERTa-77M-MLM',
    max_steps: int = 50_000,
    batch_size: int = 32,
    lr: float = 5e-5,
    warmup_fraction: float = 0.05,
    mlm_probability: float = 0.15,
    max_length: int = 256,
    save_every: int = 10_000,
    fp16: bool = True,
    seed: int = 42,
) -> Path:
    """Continue MLM pretraining of ChemBERTa on CGR-SMILES corpus.

    1. Load base model + extend tokenizer
    2. Resize token embeddings
    3. Initialize new token embeddings to mean of existing
    4. Train with HuggingFace Trainer + DataCollatorForLanguageModeling
    5. Save final checkpoint

    Args:
        corpus_path: Path to corpus.txt (one CGR-SMILES per line).
        output_dir: Directory for checkpoints.
        base_model: HuggingFace model ID.
        max_steps: Training steps (default 50K).
        batch_size: Per-device batch size.
        lr: Peak learning rate.
        warmup_fraction: Fraction of steps for warmup.
        mlm_probability: Mask probability for MLM.
        max_length: Max sequence length.
        save_every: Checkpoint interval in steps.
        fp16: Use mixed precision.
        seed: Random seed.

    Returns:
        Path to final checkpoint directory.
    """
    import torch
    from transformers import (
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / 'checkpoint'

    # Extend tokenizer
    tokenizer, n_new = extend_tokenizer(base_model, save_dir=checkpoint_dir)

    # Load model + resize embeddings
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings to mean of existing
    if n_new > 0:
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean_emb = emb[:old_vocab_size].mean(dim=0)
            emb[old_vocab_size:] = mean_emb
        print(f"  Initialized {n_new} new token embeddings to mean of existing")

    # Load corpus
    print(f"  Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  Corpus: {len(lines)} CGR-SMILES strings")

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    dataset = Dataset.from_dict({'text': lines})
    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=['text'],
        desc='Tokenizing corpus')

    # Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # Training args
    warmup_steps = int(max_steps * warmup_fraction)
    device_has_gpu = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(output_dir / 'trainer_output'),
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        fp16=fp16 and device_has_gpu,
        logging_steps=500,
        save_steps=save_every,
        save_total_limit=3,
        seed=seed,
        dataloader_num_workers=4,
        report_to='none',
    )

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(f"  Starting MLM pretraining: {max_steps} steps, "
          f"batch_size={batch_size}, lr={lr}")
    trainer.train()

    # Save final checkpoint
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"  Saved checkpoint to {checkpoint_dir}")

    return checkpoint_dir


# =========================================================================
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='CGR-SMILES corpus generation and MLM pretraining')
    parser.add_argument(
        '--stage', choices=['corpus', 'pretrain', 'all'], default='all',
        help='Which stage to run (default: all)')
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for corpus and checkpoints')
    parser.add_argument(
        '--max-reactions', type=int, default=None,
        help='Max reactions from USPTO (None = all ~1M)')
    parser.add_argument(
        '--max-steps', type=int, default=50_000,
        help='MLM pretraining steps (default: 50000)')
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Per-device batch size (default: 32)')
    parser.add_argument(
        '--lr', type=float, default=5e-5,
        help='Peak learning rate (default: 5e-5)')
    parser.add_argument(
        '--n-workers', type=int, default=8,
        help='Parallel workers for corpus generation (default: 8)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.output_dir / 'corpus.txt'
    stats_path = args.output_dir / 'corpus_stats.json'

    if args.stage in ('corpus', 'all'):
        from kirby.datasets.uspto import load_uspto_stereo

        print("=" * 60)
        print("STAGE 1: CGR-SMILES CORPUS GENERATION")
        print("=" * 60)

        rxn_smiles = load_uspto_stereo(max_reactions=args.max_reactions)
        stats = build_cgr_corpus(
            rxn_smiles, corpus_path, n_workers=args.n_workers)

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved to {stats_path}")

    if args.stage in ('pretrain', 'all'):
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. "
                f"Run --stage corpus first.")

        # Check corpus is non-empty before spending GPU time
        with open(corpus_path) as _f:
            first_line = _f.readline()
        if not first_line.strip():
            raise RuntimeError(
                f"Corpus at {corpus_path} is empty (0 CGR-SMILES). "
                f"Check that chython is installed and reactions are "
                f"atom-mapped. Aborting pretraining.")

        print("\n" + "=" * 60)
        print("STAGE 2: MLM PRETRAINING")
        print("=" * 60)

        pretrain_mlm(
            corpus_path=corpus_path,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
