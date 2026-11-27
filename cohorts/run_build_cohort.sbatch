#!/usr/bin/env bash
# Build the YAIB frames (cohort, labels, statics, dynamics from the CDM) for one (dataset, task) as a SLURM job:
#   source env.sh
#   OMOP_CDM=<cdm> sbatch -o <log> cohorts/run_build_cohort.sbatch <dataset> <task> [duckdb memory]
# The job runs the checkout env.sh was sourced from (ICU_OMOP_REPO). OMOP_CDM, COHORT_CACHE, FRAMES_OUT,
# DUCK_TMP_BASE, CAI_CONVERT, RICU_HIRID_URINE and COHORT_DIAG pass through from the submitting environment, and so
# do SBATCH_PARTITION and SBATCH_QOS from config.env. omop_etl/tools/run_cohorts.sh submits a whole chain.
#SBATCH -J omop_build
#SBATCH -c 8
# miiv/hirid peak at ~90G (duckdb fills its default 90GB limit). eicu: static tasks --mem=178G with a
# 40GB duckdb limit; hourly tasks (los/aki/sepsis) --mem=300G with 250GB (see RUNBOOK gotchas)
#SBATCH --mem=110G
#SBATCH -t 05:00:00

set -euo pipefail
DATASET="${1:?usage: sbatch run_build_cohort.sbatch <dataset> <task> [duckmem]}"
TASK="${2:?usage: sbatch run_build_cohort.sbatch <dataset> <task> [duckmem]}"
DUCKMEM="${3:-90GB}"
REPO="${ICU_OMOP_REPO:?source env.sh before sbatch}"
: "${OMOP_CDM:?set OMOP_CDM to the CDM to read}"
export DUCK_TMP_BASE=${DUCK_TMP_BASE:-$(dirname "$OMOP_CDM")/cohorts/_duckdb_tmp}

echo "=== node=$(hostname) job=$SLURM_JOB_ID build $DATASET/$TASK duckmem=$DUCKMEM ==="
# python cannot clean up after an OOM kill
trap 'rm -rf "$DUCK_TMP_BASE/${SLURM_JOB_ID}"_* 2>/dev/null || true' EXIT
cd "$REPO/cohorts" && source ../env.sh
[ -n "${VIRTUAL_ENV:-}" ] || { echo "env.sh did not activate the environment" >&2; exit 1; }
python build_cohort.py --dataset "$DATASET" --task "$TASK" --mem "$DUCKMEM" --threads 8
echo "=== build done ==="
