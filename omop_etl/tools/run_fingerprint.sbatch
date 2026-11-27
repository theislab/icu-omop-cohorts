#!/bin/bash
# Fingerprint an OMOP CDM parquet directory (aggregates only) as a SLURM job, see README.md:
#   source env.sh
#   sbatch -o <log> omop_etl/tools/run_fingerprint.sbatch <cdm_dir> <out_dir> [extra fingerprint_cdm.py args]
# MEASUREMENT of 1.4 B rows (4 grouped passes): ~3 min, MaxRSS 24 GB. duckdb gets 80% of --mem.
#SBATCH -J omop-fingerprint
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:00:00
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: sbatch run_fingerprint.sbatch <cdm_dir> <out_dir> [args]" >&2; exit 2; }
REPO="${ICU_OMOP_REPO:?source env.sh before sbatch}"
cd "$REPO" && source env.sh >/dev/null
echo "job ${SLURM_JOB_ID:-} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, ${SLURM_MEM_PER_NODE:-?} MB"
time python omop_etl/tools/fingerprint_cdm.py "$@"
du -sh "$2"
