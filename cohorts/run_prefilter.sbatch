#!/usr/bin/env bash
# The prefilter of one dataset (prefilter.py) as a SLURM job:
#   source env.sh
#   OMOP_CDM=<cdm> sbatch -o <log> cohorts/run_prefilter.sbatch <dataset>
# The job runs the checkout env.sh was sourced from (ICU_OMOP_REPO). OMOP_CDM and COHORT_CACHE pass through from the
# submitting environment, and so do SBATCH_PARTITION and SBATCH_QOS from config.env.
#SBATCH -J omop_prefilter
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00

set -euo pipefail
DATASET="${1:-miiv}"
REPO="${ICU_OMOP_REPO:?source env.sh before sbatch}"
: "${OMOP_CDM:?set OMOP_CDM to the CDM to read}"
CACHE=${COHORT_CACHE:-$(dirname "$OMOP_CDM")/cohorts/cohort_cache}

echo "=== node=$(hostname) job=$SLURM_JOB_ID dataset=$DATASET ==="
echo "OMOP_CDM=$OMOP_CDM COHORT_CACHE=$CACHE"
# duckdb's spill; python cannot clean up after an OOM kill
trap 'rm -rf "$CACHE/_duckdb_tmp/${SLURM_JOB_ID}"_* 2>/dev/null || true' EXIT

cd "$REPO/cohorts" && source ../env.sh
[ -n "${VIRTUAL_ENV:-}" ] || { echo "env.sh did not activate the environment" >&2; exit 1; }
python prefilter.py --dataset "$DATASET" --mem 56GB --threads 8

echo "=== done ==="
