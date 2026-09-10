#!/bin/bash
# =============================================================================
# Install what the c13 forward-compatibility candidates need, in risk order.
# =============================================================================
# NOT a SLURM job — run it on a login node or an interactive session, because
# most of it is downloads.
#
#   bash tests/slurm_scripts/install_forward_compat_deps.sh 1     # tier 1 only
#   bash tests/slurm_scripts/install_forward_compat_deps.sh 1 2   # tiers 1 and 2
#
# After ANY tier, run the gate to see what changed:
#   sbatch tests/slurm_scripts/run_forward_compat_gate.sh
#
# The tiers are ordered by how much they can break. Tier 1 touches nothing that
# an existing experiment depends on. Tier 4 pins torch and CUDA and can break
# every other experiment on the cluster — read the warning before running it.
#
# WHY TIERS AND NOT ONE BIG INSTALL. Several candidates pin an old torch
# (rxngraphormer wants torch 1.12.1, egret wants transformers 4.18.0). Installing
# those into the shared environment would downgrade torch under every other
# experiment. Tier 3 and 4 are therefore documented as SEPARATE ENVIRONMENTS,
# not as things to pip into the main one.
# =============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
. ./setup.sh

TIERS="${*:-1}"
run() { echo; echo ">>> $*"; "$@" || echo "    FAILED (continuing): $*"; }
has() { case " $TIERS " in *" $1 "*) return 0;; *) return 1;; esac; }

# -----------------------------------------------------------------------------
# TIER 1 — plain HuggingFace models. No build step, no CUDA pin, no new torch.
#          Weights download on first call. This is the cheapest 20 candidates.
# -----------------------------------------------------------------------------
if has 1; then
echo "=============== TIER 1: transformers-only candidates ==============="
# Covers, by domain:
#   antibody   ism igbert p_iggen antiberta2 currab abcdr_esmc
#   protein    ism ankh3 prostt5
#   nucleotide nucel generator_euk_1_2b metagene_1 moderngena_large
#   molecular  chemfm
#   omics      c2s_scale
# ModernGENA needs transformers >= 4.48 for ModernBERT; ankh3/prostt5 need
# sentencepiece; several need accelerate for large shards.
run pip install --upgrade "transformers>=4.48" sentencepiece accelerate safetensors huggingface_hub
run pip install esm                       # abcdr_esmc base checkpoint (ESM-C)
fi

# -----------------------------------------------------------------------------
# TIER 2 — pip-installable but with a real package behind them. Still no torch
#          repin. Each is independent; a failure here costs you that one model.
# -----------------------------------------------------------------------------
if has 2; then
echo "=============== TIER 2: pip packages with a build ==============="
run pip install "chemprop>=2.2.0"          # chemeleon, glacier, diff/concat_chemeleon
run pip install lightning                  # glacier
run pip install borzoi-pytorch             # borzoi   ⛔ SEE WARNING BELOW
run pip install ai2-olmo                   # omni_dna_116m
run pip install transcriptformer           # transcriptformer
run pip install scprint2                   # scprint2
run pip install CpGPT                      # methyl_cpgpt
run pip install moozy h5py                 # wsi_moozy + .h5 tile features
run pip install timm torchvision pillow    # wsi_gpfm, wsi_midnight, wsi_distillpath
run pip install torch-geometric molfeat    # mol_jepa
echo
echo "  ⛔ borzoi: its pretraining targets INCLUDE H3K27me3 / H3K4me3 / H3K9ac /"
echo "     H3K9me3 — the labels four of the nucleotide datasets predict. A"
echo "     positive c13 delta from it is label leakage through the pretraining"
echo "     set, not the hybrid absorbing a new model. Run it labelled, or not."
fi

# -----------------------------------------------------------------------------
# TIER 3 — git clone + a manual weights download. No pip conflict, but each one
#          needs a path exported so the builder can find it.
# -----------------------------------------------------------------------------
if has 3; then
echo "=============== TIER 3: clone + manual weights ==============="
MODELS="${KIRBY_MODELS_DIR:-$PWD/models}"
mkdir -p "$MODELS"
echo "  cloning into $MODELS"

run git clone https://github.com/VarunUllanat/mint.git        "$MODELS/mint"
run pip install -e "$MODELS/mint"                              # antibody: mint
run pip install git+https://github.com/oxpig/AntiFold.git      # antibody: antifold
run pip install ImmuneBuilder biopython                        # antifold needs a structure
run git clone https://github.com/terraytherapeutics/COATI.git "$MODELS/COATI"
run pip install -e "$MODELS/COATI"                             # molecular: coati
run git clone https://github.com/Profluent-AI/E1.git          "$MODELS/E1"
run pip install -e "$MODELS/E1"                                # protein: profluent_e1
run git clone https://github.com/westlake-repl/ProTrek.git    "$MODELS/ProTrek"
run git clone https://github.com/albert-ying/MethylGPT.git    "$MODELS/MethylGPT"
run pip install -e "$MODELS/MethylGPT"                         # omics: methyl_methylgpt
run git clone https://github.com/KangBoming/BulkFormer.git    "$MODELS/BulkFormer"

cat <<'EOF'

  STILL MANUAL after this tier — each needs a file fetched by hand:
    ouroboros    https://aideepmed.com/Ouroboros/Ouroboros_M1c.zip
                 unzip, then: export KIRBY_OUROBOROS_MODEL=<dir>
                 ⛔ GPU ONLY — its encoder calls .cuda() unconditionally.
    protrek      huggingface-cli download westlake-repl/ProTrek_650M_UniRef50
                 then: export KIRBY_PROTREK_DIR=<repo> and point at the weights
    bulkformer   bulkformer_147M.pt from the repo's release page
                 then: export KIRBY_BULKFORMER_DIR=<clone>
    methylgpt    args.json + best_model.pt from the README's Google Drive link
    abmpnn       abmpnn.pt from Zenodo 8164693 (20 MB) + ProteinMPNN
    unimol3      Uni-Mol2 84M + Uni-Tokenizer + two Google Drive checkpoints
    rxngraphormer pretrained_classification_model.zip from figshare 28356077
EOF
fi

# -----------------------------------------------------------------------------
# TIER 4 — DO NOT RUN THIS INTO THE SHARED ENVIRONMENT.
# -----------------------------------------------------------------------------
if has 4; then
cat <<'EOF'
=============== TIER 4: separate environments only ===============

  ⛔ Every entry here pins torch, CUDA or a different framework. Installing any
     of them into the environment the rest of KIRBy uses will break other
     experiments. Make a separate conda env per line, or skip them.

    evo2_1b        conda cuda-nvcc + transformer-engine-torch==2.3.0 +
                   flash-attn==2.8.0.post2 + pip evo2.  CUDA only.
    plantcad2      mamba-ssm + causal-conv1d, both CUDA-only builds.
    janusdna       mamba-ssm from a prebuilt CUDA wheel + a Harvard Dataverse
                   checkpoint.  Currently a STUB either way.
    cellfm         mindspore==2.2.10 — not PyTorch at all.
    rxngraphormer  pins torch 1.12.1 and torch_geometric 2.3.1.
    egret          pins transformers 4.18.0 and simpletransformers 0.61.13.
                   ⛔ AND its released checkpoint is the Buchwald-Hartwig
                   FINETUNED yield head — embedding those datasets with it is
                   training on the labels. create_egret refuses unless
                   allow_finetuned_checkpoint=True is passed.
    tahoe_x1       wants flash-attn built without build isolation.
EOF
fi

echo
echo "=================================================================="
echo "Done. Now find out what actually builds:"
echo "  sbatch tests/slurm_scripts/run_forward_compat_gate.sh"
echo "=================================================================="
