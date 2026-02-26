# Fine-Tuning Experiments TODO

Untried fine-tuning approaches beyond the LoRA/MLM survey (§7 in BOM).

## Molecular

| Approach | Base Model(s) | How it differs from what was tested | Priority |
|----------|--------------|-------------------------------------|----------|
| **Freeze+head** (learn projection, freeze backbone) | ChemBERTa, MolFormer | LoRA updates backbone weights which may be destructive on small data. Freeze+head is the pattern that works for Geneformer. | HIGH |
| **GNN fine-tuning** (published protocols) | MolCLR, GraphMVP | These models were designed for fine-tuning — their papers report gains. We only tested them frozen. | HIGH |
| **Different LoRA ranks** | ChemBERTa, MolFormer | Survey only tested rank=8. Lower rank (2-4) may regularize better on small datasets. | LOW |
| **Contrastive fine-tuning** | ChemBERTa, MolFormer | Learn to pull similar-property molecules together in embedding space, rather than regression. | MEDIUM |
| **GROVER fine-tuning** | GROVER | Has its own fine-tuning protocol (message-passing + transformer). 5000D frozen is very high-dim; fine-tuned projection could help. | MEDIUM |

## Antibody

| Approach | Base Model(s) | How it differs | Priority |
|----------|--------------|----------------|----------|
| **ESM-1v marginal scores** | ESM-1v | Zero-shot fitness prediction — fundamentally different from embedding extraction. Published strong DMS correlation. | HIGH |
| **SaProt** | SaProt | Structure-aware protein LM using 3Di alphabet (Foldseek). Different input modality entirely. | MEDIUM |
| **AbLang2 LoRA** | AbLang2 | Only positive inter-family LM (R²=0.149 on thermo_jain). ESM-2 LoRA failed but AbLang2 is antibody-specific. | MEDIUM |
| **Freeze+head** | ESM-2, AbLang2 | Same reasoning as molecular — LoRA may be too aggressive for small DMS datasets. | LOW |

## TCGA

| Approach | Base Model(s) | How it differs | Priority |
|----------|--------------|----------------|----------|
| **scFoundation** (frozen or FT) | scFoundation | 2048D, different pretraining from Geneformer. Implemented (`gene_scfoundation`) but never tested. | HIGH |

## What was already tested and failed (do NOT re-run)

- LoRA ChemBERTa / MolFormer on molecular (worse than frozen ecfp4/pdv)
- LoRA ESM-2 (650M and 8M) on antibody (negative R²)
- MLM ESM-2 on antibody (negative R²)
- See BOM §7 for full results
