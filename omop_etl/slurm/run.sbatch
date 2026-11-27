#!/bin/bash
# One step of the ETL as a SLURM job:
#   sbatch slurm/run.sbatch <convert|stage|validate|cdm> [dataset] [extra args for python -m omop_etl]
# Resources here are the defaults; slurm/submit_all.sh passes per-step values on the command line.
#SBATCH -J omop-etl
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
set -euo pipefail
step=${1:?usage: sbatch slurm/run.sbatch <step> [dataset] [args]}; shift
dataset_args=()
if [ "$step" != cdm ] && [ $# -gt 0 ]; then dataset_args=(--dataset "$1"); shift; fi
cd "${OMOP_ETL_DIR:-$SLURM_SUBMIT_DIR}"
source ../env.sh >/dev/null
echo "job ${SLURM_JOB_ID} on $(hostname): $step ${dataset_args[*]} $*, ${SLURM_CPUS_PER_TASK} cpus, ${SLURM_MEM_PER_NODE} MB"
python -m omop_etl "$step" "${dataset_args[@]}" "$@"
sstat -n -j "${SLURM_JOB_ID}.batch" --format=MaxRSS 2>/dev/null | sed 's/^/MaxRSS so far: /' || true
