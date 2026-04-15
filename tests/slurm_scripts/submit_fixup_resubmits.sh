#!/bin/bash
# Resubmits for the 11 jobs broken by the sbatch --wrap line-splitting bug.
# All jobs run on HTC (user's original submissions were HTC; 7-digit job IDs).
#
# Uses heredoc-style sbatch (multi-line slurm script via stdin) so each
# command lands on its own literal line — impossible to trigger whatever
# slurmd is doing to --wrap strings that splits them mid-argument.
#
# Usage (on HTC login):
#   bash tests/slurm_scripts/submit_fixup_resubmits.sh gpu   # abclf + dta_kiba_fill (GPU, long, 24h)
#   bash tests/slurm_scripts/submit_fixup_resubmits.sh cpu   # nuc + gb_demo OOM  (CPU, medium, 48h)
#   bash tests/slurm_scripts/submit_fixup_resubmits.sh all   # both

set -euo pipefail

MODE=${1:-}
EMAIL="adelaide.punt@stcatz.ox.ac.uk"
KIRBY="/data/stat-cadd/scat9264/KIRBy"

ALL_AB_REPS="onehot,blosum,aac,paratope,esm2,esmc,ablang2,igbert,igbert_cls,antiberty,prot_t5,amplify,igt5,esm3,esm3_cdr,saprot,saprot_cdr,esm2_cdr,esmc_cdr,antiberty_cdr,prot_t5_cdr,amplify_cdr,esm2_midlayers,esm2_layer24,esm2_early,esm2_concat,esm2_cdr3,abroberta,abroberta_cdr,structural,mutation_features,esm2_mutsite,esm2_wt_delta,esm2_llr,esmc_llr,antiberty_llr,esm2_entropy_top20,esm2_per_cdr,esm2_per_region,mint,aido_protein,xtrimopglm"

submit_abclf() {
    local ds=$1
    local mem=$2
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=abclf_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=${mem}
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

cd ${KIRBY}
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
mkdir -p tests/results/opig/clf
python tests/antibody_baseline_survey.py ${ds} \\
    --output tests/results/opig/clf/ \\
    --n-folds 5 \\
    --augment \\
    --skip-existing \\
    --reps ${ALL_AB_REPS}
SBATCH_EOF
)
    echo "  ${jid}  abclf_${ds}  (long, ${mem}, 24h, HTC+GPU)"
}

submit_dta_kiba_fill() {
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dta_kiba_fill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

cd ${KIRBY}
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
mkdir -p tests/results/phase5_baselines/dta
python tests/dta_baseline_survey.py kiba \\
    --n-folds 5 \\
    --skip-existing \\
    --output tests/results/phase5_baselines/dta/kiba.json
SBATCH_EOF
)
    echo "  ${jid}  dta_kiba_fill  (long, 256G, 24h, HTC+GPU)"
}

submit_schnet_e2e() {
    # SchNet E2E rerun — previous 7440325-327 hit the zero-byte FT cache
    # and every FT factory returned "No data left in file". fc36c6c fixes
    # the cache atomic write and adds auto-healing on read, so this run
    # will produce real ft_schnet/lora_chemberta/lora_molformer scores.
    local ds=$1
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=schnet_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

cd ${KIRBY}
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
mkdir -p tests/results/schnet_e2e
python tests/run_baseline_survey.py ${ds} \\
    --finetune \\
    --models ridge,rf,xgboost \\
    --n-folds 3 \\
    --output tests/results/schnet_e2e/
SBATCH_EOF
)
    echo "  ${jid}  schnet_${ds}  (long, 128G, 24h, HTC+GPU)"
}

submit_dta_bindingdb_kd() {
    # Full fresh resubmit. Previous run (7440293) hit 4d wall and
    # produced no output because saves were end-only. Commit a81b0eb
    # adds per-rep incremental checkpointing, so if this 48h run
    # times out we resubmit with --skip-existing and pick up.
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=dta_bindingdb_kd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=medium
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

cd ${KIRBY}
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
mkdir -p tests/results/phase5_baselines/dta
python tests/dta_baseline_survey.py bindingdb_kd \\
    --n-folds 5 \\
    --skip-existing \\
    --output tests/results/phase5_baselines/dta/bindingdb_kd.json
SBATCH_EOF
)
    echo "  ${jid}  dta_bindingdb_kd  (medium, 256G, 48h, HTC+GPU)"
}

submit_nuc() {
    local ds=$1
    local mem=$2
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=nuc_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=medium
#SBATCH --mem=${mem}
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

cd ${KIRBY}
git pull 2>/dev/null || true
. ./setup.sh
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
mkdir -p tests/results/nucleotide_survey
python tests/run_baseline_survey.py ${ds} \\
    --n-folds 5 \\
    --skip-existing \\
    --output tests/results/nucleotide_survey/
SBATCH_EOF
)
    echo "  ${jid}  nuc_${ds}  (medium, ${mem}, 48h, HTC CPU)"
}

case "${MODE}" in
    gpu)
        echo "=== HTC GPU resubmits (antibody_clf + dta_kiba_fill) ==="
        submit_abclf tz_her2_binary       256G
        submit_abclf tz_her2_multiclass   256G
        submit_abclf humanness_species    128G
        submit_abclf humanness_therapeutic 128G
        submit_dta_kiba_fill
        ;;
    cpu)
        echo "=== HTC CPU resubmits (nucleotide + gb_demo OOM bumps) ==="
        submit_nuc nt_enhancers                     128G
        submit_nuc nt_enhancers_types               128G
        submit_nuc nt_splice_sites_acceptors        128G
        submit_nuc nt_splice_sites_donors           128G
        submit_nuc gb_demo_coding_vs_intergenomic_seqs 256G
        submit_nuc gb_demo_human_or_worm            256G
        ;;
    bindingdb)
        echo "=== HTC bindingdb_kd fresh resubmit (medium+GPU+256G, 48h) ==="
        submit_dta_bindingdb_kd
        ;;
    schnet)
        echo "=== HTC SchNet E2E rerun (3 datasets, long+GPU+128G, 24h) ==="
        submit_schnet_e2e logd
        submit_schnet_e2e herg
        submit_schnet_e2e caco2
        ;;
    all)
        echo "=== HTC GPU resubmits (antibody_clf + dta_kiba_fill) ==="
        submit_abclf tz_her2_binary       256G
        submit_abclf tz_her2_multiclass   256G
        submit_abclf humanness_species    128G
        submit_abclf humanness_therapeutic 128G
        submit_dta_kiba_fill
        echo ""
        echo "=== HTC CPU resubmits (nucleotide + gb_demo OOM bumps) ==="
        submit_nuc nt_enhancers                     128G
        submit_nuc nt_enhancers_types               128G
        submit_nuc nt_splice_sites_acceptors        128G
        submit_nuc nt_splice_sites_donors           128G
        submit_nuc gb_demo_coding_vs_intergenomic_seqs 256G
        submit_nuc gb_demo_human_or_worm            256G
        echo ""
        echo "=== HTC bindingdb_kd fresh resubmit ==="
        submit_dta_bindingdb_kd
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_fixup_resubmits.sh [gpu|cpu|bindingdb|schnet|all]"
        echo ""
        echo "  gpu       — 4 abclf jobs + dta_kiba_fill (GPU, long, 24h)"
        echo "  cpu       — 4 nucleotide resubmits (128G) + 2 gb_demo OOM bumps (256G) (medium, 48h)"
        echo "  bindingdb — dta_bindingdb_kd fresh (GPU, medium, 48h, 256G)"
        echo "  schnet    — SchNet E2E rerun: logd, herg, caco2 (GPU, long, 24h, 128G)"
        echo "  all       — every set (15 jobs total)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
