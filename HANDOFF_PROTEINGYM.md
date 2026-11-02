# Handoff: reproduce ProteinGym's supervised baselines, then test the hybrid

## The job

Reproduce ProteinGym's supervised DMS baselines inside KIRBy, on ProteinGym's
own folds, so the hybrid has a comparison Adelaide produced herself. Do not
quote their published numbers as a result. They are a check on your own run,
nothing else.

## What is already done (verified 2026-10-26, this machine)

Assay: `ESTA_BACSU_Nutschel_2020`, 2,172 variants, 212 residues, thermostability.

- `src/kirby/datasets/proteingym.py` loads it as `pgym_ESTA_BACSU_Nutschel_2020`.
  `d['groups']` under `GroupKFold(n_splits=5)` gives **exactly** ProteinGym's
  `fold_modulo_5` partition, relabelled by one position (my fold 0 = their fold
  1, ... my fold 4 = their fold 0). Checked variant by variant; sizes
  470/435/429/421/417. So no fold work is needed for modulo.
- ProteinGym's fold file is on seagate:
  `/Volumes/seagate/proteingym_raw/ex_folds/cv_folds_singles_substitutions/<DMS_id>.csv`
  with columns `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`.
  **Random is not to be used** (Adelaide's standing rule). Modulo and
  contiguous are hers to use; only modulo has been checked against the loader.
- The 95 per-variant zero-shot scores are on seagate:
  `/Volumes/seagate/proteingym_raw/ex_zeroshot/<DMS_id>.csv`. Columns are
  `mutant, mutated_sequence, DMS_score, DMS_score_bin` then 95 method columns.
  No blanks on this assay.
- Scored locally, five modulo folds, Spearman, one model per representation
  (`scratchpad/pgym_base_reps.py`, log `scratchpad/base_reps_esta.log`):

  | representation | features | LightGBM | SVM |
  |---|---|---|---|
  | all 95 zero-shot scores | 95 | 0.5958 | 0.6330 |
  | one-hot of the sequence | 4240 | undefined | undefined |
  | aac | 40 | 0.1216 | 0.0917 |
  | mutation_features | 48 | 0.1641 | 0.1374 |

  One-hot is undefined because the folds are position-disjoint: each held-out
  fold mutates 33 or 34 positions and **0** of them appear in the training rows
  (measured). Every one-hot feature that varies in the held-out fold is
  constant in training, so the predictions are constant. ProteinGym publishes
  One-Hot for this assay as −0.021 under modulo and −0.094 under contiguous,
  so their own run finds no signal either. One-hot is therefore **degenerate
  under position-disjoint folds** — that is a stated reason to drop it from
  those folds, and the only one on record.

## The embeddings: what was settled 2026-10-26, and what is left

**No precomputed embeddings are published anywhere.** ProteinGym's README
resources table lists 20 downloads — DMS data, zero-shot scores, supervised
scores, MSAs, MSA weights, AF2 structures, CV folds — and none is embeddings.
ProteinNPT's README says you create them yourself with `embeddings_subs.sh`,
and that saving all MSA Transformer embeddings takes about 1TB. Read both
READMEs on 2026-10-26. So the forward passes cannot be avoided; the question is
only where they run.

This laptop is an Intel i5-8257U with no usable GPU (torch reports MPS
unavailable), which is where the hours below come from.

### 1. ESM-1v — runs here

Layer 33 mean-pooled over residues, 1280 features
(`proteinnpt/baselines/model.py`, `last_layer_index = 33` for ESM1v, then
`x.mean(dim=-2)` with no mask, so BOS and EOS are averaged in).

`facebook/esm1v_t33_650M_UR90S_1` through HuggingFace `EsmModel`, so fair-esm
is not needed for this one. Script `scratchpad/pgym_esm1v_embed.py`, weights
cached under `/Volumes/seagate/hf_cache`.

Measured: 4.08 s per variant at batch 16 over the finished run, 8,870 s for the
assay. Output
`/Volumes/seagate/proteingym_embeddings/ESTA_BACSU_Nutschel_2020/esm1v_mean_l33.npy`,
2,172 by 1,280, with a `.mutants.txt` beside it in the same row order.

**Done and scored on ESTA, 2026-10-26.** Five modulo folds, Spearman, mean
with the spread across folds:

| model | plain, 1280 features | plus the paired zero-shot score, 1281 |
|---|---|---|
| LightGBM | 0.3326 (0.0787) | 0.4014 (0.0488) |
| SVM | 0.3895 (0.0793) | 0.3956 (0.0782) |

ProteinGym publishes 0.408 for ESM-1v Embeddings on this assay under
`fold_modulo_5`. The comparable row is plus_zs, since their column carries the
covariate.

The zero-shot score does different work under the two models: +0.069 Spearman
for LightGBM, +0.006 for SVM. The SVM gets to 0.3895 from the embeddings
alone. The downstream model is not something their column varies, so "how
close is this to their number" has two answers.

A first run of this died silently at 1,376 of 2,172 and lost two hours,
because the script saved nothing until the end. It now checkpoints every 160
variants to the scratchpad and resumes from the same command. Cause of that
death not established: the volume was briefly missing when the watch looked,
the machine has 16GB with swap in use, and no kernel kill record was found.

### 2. MSA Transformer — the input chain is reproduced, the cost is not viable here

Needs a tool nothing in the earlier handoff mentioned: **hhfilter**, from
HH-suite. `msa_utils.py:363` filters the alignment at `-cov 75 -id 90 -qid 0`
before the model sees it. Homebrew has no formula; installed 3.3.0 from
bioconda into a micromamba environment of its own, binary at
`/usr/local/Cellar/micromamba/2.0.8/envs/hhsuite/bin/hhfilter`.

The chain is verified end to end on ESTA, not assumed:

| step | count |
|---|---|
| sequences in the downloaded a2m | 234,310 |
| what hhfilter reads before its own ceiling | 65,535 |
| after `-cov 75 -id 90 -qid 0` | 23,479 |
| after their preprocessing | 23,428 |
| weights in their shipped file | 23,428 |

Zero difference at the bottom, so the weighted sampling follows their code with
nothing guessed. `scratchpad/check_msa_lineup.py` recomputes that count in its
own code — ProteinNPT's own module could not be executed here, the sandbox
blocks running downloaded repo code, and reimplementing was the way round it.

Note hhfilter 3.3.0 prints `WARNING: Maximum number 65535 of sequences
exceeded`, so the alignment MSA Transformer sees is drawn from 28% of the
file. Their count matching exactly is evidence their run hit the same ceiling;
which hhfilter build they used is not recorded anywhere I could find.

Two files per assay ship in the weights archive and **only one matches**:
`DMS_msa_weights/DMS_msa_weights_for_MSA_Transformer/<assay>_theta_0.2.npy`
holds 23,428, the top-level `DMS_msa_weights/<assay>_theta_0.2.npy` holds
233,637.

Mechanism, all read on 2026-10-26: 384 alignment rows per variant
(`embeddings_subs.sh:10`), the reference first then 383 drawn **with
replacement** by `random.choices` weighted by that file
(`msa_utils.py:485-487`), drawn **once** and reused for every variant
(`embedding_utils.py:98`), each variant placed first in its own batch
(`embedding_utils.py:157`), `repr_layers=[12]`, row 0 taken
(`embedding_utils.py:247, 263`), mean over residues, 768 features.

Script `scratchpad/pgym_msat_embed.py`, checkpoint
`esm_msa1b_t12_100M_UR50S` under `/Volumes/seagate/torch_hub`. It runs
correctly — 2 variants, exit 0, matrix written.

**Measured 625 s per variant**, because each variant is a 385-row by
212-column alignment through the model. 2,172 variants is about 377 hours.
That is the blocker, and there are two ways past it, which are not equivalent:

- `fast_MSA_mode` (`embedding_utils.py:133-137`) puts 64 variants in one MSA
  batch alongside the 384 rows, matching their config's
  `eval_num_sequences_to_score_per_batch_per_gpu: 64`. 34 forward passes
  instead of 2,172. But MSA Transformer attends across rows, so 64 variants
  sharing a batch see each other and the numbers are not the same as
  one-variant-per-batch. Which of the two produced their published column is
  not determinable from their code.
- The GPU server, one variant per batch, which is what the model was built for.

**Adelaide picks this one.** Server rules: local or `srun` only, never
`sbatch`; one `srun ... --pty /bin/bash` line with plain commands inside; never
`ssh` or `scp`; never `rm` in a run block.

### 3. Tranception — not started

Repo cloned to `/Volumes/seagate/Tranception`; the checkpoint (about 3.1GB) is
not fetched. Their code disagrees with itself and this is confirmed, not a
typo: `proteinnpt/baselines/model.py:158` takes `output.hidden_states[0]`,
`proteinnpt/utils/embedding_utils.py:300` takes `output.hidden_states[-1]` with
the comment "Extract embeddings from last layer". In a HuggingFace model index
0 is the input embedding layer, before any transformer block, and index -1 is
the final layer — two different representations. `model.py` is the path that
runs when embeddings are not precomputed. For ESM-1v both files agree on layer
33; Tranception is the only one where they differ.

**Adelaide picks the layer.** Record which was used in the script.

## How their baselines actually work, so the reproduction is honest

Read from `proteinnpt/baselines/model.py` on 2026-10-26.

- The zero-shot score is **not a feature**. The config calls it
  `augmentation: "zero_shot_fitness_predictions_covariate"`. The prediction is
  `linear(features) + w × zero_shot`, with `w` a learned scalar per target
  initialised to 1.0. So the model starts by trusting the zero-shot score
  exactly and learns a correction. It never passes through the network, so
  nothing can interact with it.
- The downstream head is one `nn.Linear(dim, 1)` per target. MSELoss for
  continuous targets. AdamW, weight decay 5e-3.
- Their one-hot covers the full sequence including the ESM alphabet's BOS/EOS
  positions, flattened.

So "ESM-1v + One-Hot Encodings" is **one** zero-shot score added to a linear
fit on one-hot features. Nothing in their table combines more than two inputs,
and nothing uses more than one language model.

**Their three embedding columns are not representations on their own either.**
Read 2026-10-26 in `proteinnpt/baselines/model_configs/`:
`Embeddings_ESM1v_final.json`, `Embeddings_MSAT_final.json` and
`Embeddings_Tranception_final.json` each carry
`augmentation = zero_shot_fitness_predictions_covariate`. So do the five
`OHE_<model>_final.json`. Only `OHE_not_augmented.json` carries
`augmentation = None`, and it is the only column in their table that is one
representation and nothing else.

That means scoring a plain embedding against their "ESM-1v Embeddings" number
differs in two ways at once, not one: the representation, and whether a
zero-shot score was added. `scratchpad/pgym_score_embeddings.py` therefore
scores both versions of every embedding set — `plain`, and `plus_zs` with the
paired score as one more feature. `plus_zs` is not their mechanism: theirs adds
`w * zero_shot` outside the network with `w` starting at 1.0, which a tree or
an SVM cannot express. The write-up has to say that. The pairings from their
configs are ESM-1v with `ESM1v_ensemble`, MSA Transformer with
`MSA_Transformer_ensemble`, Tranception with `Tranception_L`.

`pgym_score_embeddings.py` was tested end to end on a random 64-feature matrix
in shuffled row order: the row-alignment branch fired, the incomplete-matrix
guard refused a 32-row file, and random features scored -0.0059 with LightGBM
and -0.0165 with SVM.

## Then, and only then, the hybrid test

Once the three embedding representations exist, run the per-fold path:
`search_on_fold` → `top_base_reps` / `top_hybrids` → `build_fold_hybrid`, one
search per outer fold on that fold's training rows only. Working example:
`scratchpad/pgym_perfold.py`. **Its grid is one entry per axis and therefore
not a search** — fix that first. Adelaide's three options for a cheaper search,
with counts, are in `/Users/apunt/repos/KIRBy/SEARCH_SETTINGS.md`; she has not
picked one yet and it is hers to pick.

The comparison is **best base representation against hybrid**. Never an
ensemble of all representations — that is `all_concat` and it is banned.

Report one row per budget. The budget is a maximum, not a target, and it is not
an axis inside the search: it is explored by running at a range of budgets and
reading the range (`src/kirby/find_best_hybrid.py:11319`).

No ProteinGym assay has a `DECISIONS.md` row, so `search_on_fold` needs
`fixed_grid=` or it raises `decided_grid.UndecidedSearch`.

## A finding that needs explaining, not burying

All 95 zero-shot scores handed to one SVM as a single 95-feature
representation scored **0.6330**. The earlier hybrid run over six hand-made
families of 60 of those scores plus aac and mutation_features scored **0.5897**
with SVM. Grouping them lost ground against not grouping them.

That run had two defects — the search was one recipe at three budgets, and it
used 60 of 95 scores — so it is not a verdict. But the 0.6330 number is the
base representation the hybrid has to beat on this assay, and it costs seconds.

## Two things about the representations I used that were never justified

- The six families (`zs_sitewise`, `zs_alignment`, `zs_tranception`,
  `zs_structure`, `zs_lm_esm`, `zs_lm_generative`) were invented in a scratch
  script. They appear nowhere in `src/` or `tests/` — grepped. ProteinGym
  publishes no family taxonomy; the nearest thing is an "Input modalities"
  column in its README (Single sequence, MSA, Structure, Single sequence &
  Structure, MSA & Structure, and three qualified variants), which is not the
  grouping used.
- 35 of the 95 methods never entered the pool: Wavenet, CARP ×4, SaProt ×2,
  PoET, MULAN_small, ProSST ×6, VenusREM, RSALOR, S2F/S2F_MSA/S3F/S3F_MSA,
  SiteRM, xTrimoPGLM ×7, Progen3 ×6. None is degenerate, broken or undefined,
  so under Adelaide's rule there was no reason to cut them.

## Conceptual point to keep straight

Each of the 95 columns is another model's guess at the fitness of that variant —
the same quantity being predicted. A hybrid over them is selecting among
predictors, which is stacking. The thesis claim is about combining
*descriptions* of the molecule. One-hot, the three embedding sets, aac and
mutation_features are descriptions. The zero-shot scores are not.

So a result showing the hybrid beats the best single zero-shot score says the
hybrid is a good stacker. It does not test the sentence from the thesis about
antibody representations carrying the same information, which is what this
whole exercise was for.

## Standing constraints

- Local runs are debug, not results. "Does it run" means normal settings,
  time-boxed, time recorded, anything slow named.
- Do not prune the search space for compute. Cutting an axis, arm, budget or
  dataset needs a stated reason: degenerate, broken, or undefined.
- No random splits.
- Adelaide decides. Offer options with numbers attached.
- Large files to `/Volumes/seagate`.
- No significance testing. Per-fold scores and spread only.
- Never write an AI verdict into her documents.
