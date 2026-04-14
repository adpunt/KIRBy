#!/bin/bash
# Resubmits for the 13 jobs broken by the sbatch --wrap line-splitting bug.
# Uses heredoc-style sbatch (multi-line slurm script via stdin) so each
# command lands on its own literal line — impossible to trigger whatever
# slurmd is doing to --wrap strings that splits them mid-argument.
#
# Usage:
#   bash tests/slurm_scripts/submit_fixup_resubmits.sh htc   # from HTC login
#   bash tests/slurm_scripts/submit_fixup_resubmits.sh arc   # from ARC login
#
# Clusters:
#   htc — antibody_clf (4 jobs) + dta_kiba_fill (1 job) — GPU, long, 24h
#   arc — nucleotide resubmits (4 jobs) + gb_demo OOM bumps (2 jobs) — CPU, long, 48h

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

submit_nuc() {
    local ds=$1
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=nuc_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
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
    echo "  ${jid}  nuc_${ds}  (long, 128G, 48h, ARC CPU)"
}

submit_gb_oom() {
    local ds=$1
    local jid
    jid=$(sbatch --parsable <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=nuc_${ds}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
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
    echo "  ${jid}  nuc_${ds}  (long, 256G, 48h, ARC CPU — OOM resubmit)"
}

case "${MODE}" in
    htc)
        echo "=== HTC resubmits (antibody_clf + dta_kiba_fill) ==="
        submit_abclf tz_her2_binary       256G
        submit_abclf tz_her2_multiclass   256G
        submit_abclf humanness_species    128G
        submit_abclf humanness_therapeutic 128G
        submit_dta_kiba_fill
        ;;
    arc)
        echo "=== ARC resubmits (nucleotide + gb_demo OOM bumps) ==="
        submit_nuc nt_enhancers
        submit_nuc nt_enhancers_types
        submit_nuc nt_splice_sites_acceptors
        submit_nuc nt_splice_sites_donors
        submit_gb_oom gb_demo_coding_vs_intergenomic_seqs
        submit_gb_oom gb_demo_human_or_worm
        ;;
    *)
        echo "Usage: bash tests/slurm_scripts/submit_fixup_resubmits.sh [htc|arc]"
        echo ""
        echo "  htc — run from HTC login"
        echo "        4 abclf jobs + dta_kiba_fill (all GPU, long, 24h)"
        echo "  arc — run from ARC login"
        echo "        4 nucleotide resubmits (128G) + 2 gb_demo OOM bumps (256G)"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u \$USER"
