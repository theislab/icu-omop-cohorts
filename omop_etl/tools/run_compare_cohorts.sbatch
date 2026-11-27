#!/bin/bash
# Compare two cohort builds (aggregates only) as a SLURM job, see README.md:
#   source env.sh
#   sbatch -o <log> omop_etl/tools/run_compare_cohorts.sbatch <base_frames_root> <cand_frames_root> [compare_cohorts.py args]
#SBATCH -J omop-compare-cohorts
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: sbatch run_compare_cohorts.sbatch <base_root> <cand_root> [args]" >&2; exit 2; }
REPO="${ICU_OMOP_REPO:?source env.sh before sbatch}"
cd "$REPO" && source env.sh >/dev/null
echo "job ${SLURM_JOB_ID:-} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, ${SLURM_MEM_PER_NODE:-?} MB"
export CC_MEMORY="${CC_MEMORY:-48GB}"
time python omop_etl/tools/compare_cohorts.py "$@"
