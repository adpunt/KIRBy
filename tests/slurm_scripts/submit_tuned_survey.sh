#!/bin/bash
# Submit tuned survey: same as post-preprocessing survey but with --tune.
# One SLURM job per dataset. Each job tunes HPs at start then runs hybrid search.
# Run on login node: bash tests/slurm_scripts/submit_tuned_survey.sh [search] [tune_iter]

cd /data/stat-cadd/scat9264/KIRBy
SEARCH=${1:-medium}
TUNE_ITER=${2:-20}
OUTPUT=tests/results/tuned_survey
mkdir -p "$OUTPUT"

echo "Submitting tuned survey (--tune --tune-iter $TUNE_ITER, search=$SEARCH)..."

# Molecular: 4 jobs (64G each)
for ds in logd caco2 herg qm9; do
    sbatch --job-name="tune_${ds}" --mem=64G --cpus-per-task=8 --time=23:59:00 \
        --partition=long --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python -u tests/find_best_hybrid_molecular.py ${ds} --reps all --search ${SEARCH} --tune --tune-iter ${TUNE_ITER} --output ${OUTPUT}/mol_${ds}.json"
done

# Antibody intra-family DMS: 7 jobs (64G each)
for ds in flab_expression_koenig flab_binding_koenig \
          abagym_Ang2_2017_G6 abagym_EGFR_2013_Cetuximab \
          abagym_lysozyme_2019_D441 abagym_VEGF_2017b_G6 \
          abagym_HER2_2021_trastuzumab; do
    sbatch --job-name="tune_${ds}" --mem=64G --cpus-per-task=8 --time=23:59:00 \
        --partition=long --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python -u tests/find_best_hybrid_antibody.py ${ds} --reps dms --search ${SEARCH} --split modulo_position --augment --tune --tune-iter ${TUNE_ITER} --output ${OUTPUT}/dms_${ds}.json"
done

# Antibody inter-family: 5 jobs (64G each)
for ds in ddg_combined abagym_all abcov_kd flab_thermo_jain flab_aggregation_jain; do
    SPLIT=identity
    [ "$ds" = "abagym_all" ] && SPLIT=experiment
    sbatch --job-name="tune_${ds}" --mem=64G --cpus-per-task=8 --time=23:59:00 \
        --partition=long --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python -u tests/find_best_hybrid_antibody.py ${ds} --reps all_server --search ${SEARCH} --split ${SPLIT} --tune --tune-iter ${TUNE_ITER} --output ${OUTPUT}/ab_${ds}.json"
done

# TCGA: 3 jobs (96G — expression data is large)
for ds in tcga_luad_vs_lusc tcga_brca_pam50 tcga_brca_survival; do
    sbatch --job-name="tune_${ds}" --mem=96G --cpus-per-task=8 --time=23:59:00 \
        --partition=long --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python -u tests/find_best_hybrid_tcga.py ${ds} --reps all --search ${SEARCH} --tune --tune-iter ${TUNE_ITER} --output ${OUTPUT}/tcga_${ds}.json"
done

# DTA: 2 jobs (96G — large interaction matrices)
for ds in davis kiba; do
    sbatch --job-name="tune_${ds}" --mem=96G --cpus-per-task=8 --time=23:59:00 \
        --partition=long --mail-user=adelaide.punt@stcatz.ox.ac.uk \
        --wrap="cd /data/stat-cadd/scat9264/KIRBy && . ./setup.sh && mkdir -p ${OUTPUT} && python -u tests/find_best_hybrid_dta.py ${ds} --search ${SEARCH} --tune --tune-iter ${TUNE_ITER} --output ${OUTPUT}/dta_${ds}.json"
done

echo "Submitted 21 parallel jobs. Monitor: squeue -u \$USER"
echo "Results will be in: $OUTPUT/"
