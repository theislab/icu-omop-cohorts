#!/bin/bash
# Submit the whole ETL: convert + stage per dataset (datasets in parallel), then cdm once all
# staging jobs succeeded.
#   slurm/submit_all.sh [--config config.local.yaml] [--skip-convert] [datasets ...]
# Datasets default to mimic4 eicu hirid. Logs: <out_dir>/_logs/<job name>_<job id>.out
set -euo pipefail
cd "$(dirname "$0")/.."
# SBATCH_PARTITION, SBATCH_QOS and the other settings of this checkout (../config.example.env)
if [ -f ../config.env ]; then source ../config.env; fi
CONFIG=config.local.yaml
SKIP_CONVERT=0
DATASETS=()
while [ $# -gt 0 ]; do
    case $1 in
        --config) CONFIG=$2; shift 2 ;;
        --skip-convert) SKIP_CONVERT=1; shift ;;
        *) DATASETS+=("$1"); shift ;;
    esac
done
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(mimic4 eicu hirid)
OUT=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['out_dir'])" 2>/dev/null \
      || sed -n 's/^out_dir: *//p' "$CONFIG")
mkdir -p "$OUT/_logs"
export OMOP_ETL_CONFIG=$(realpath "$CONFIG") OMOP_ETL_DIR=$(pwd)

# measured peaks (README.md "Timings"): convert <= 10 GB, stage <= 14 GB, cdm 15 GB
declare -A RES=(
    [convert_mimic4]="-c 16 --mem=32G -t 01:00:00"  [stage_mimic4]="-c 16 --mem=48G -t 01:00:00"
    [convert_eicu]="-c 16 --mem=32G -t 01:00:00"    [stage_eicu]="-c 16 --mem=48G -t 01:00:00"
    [convert_hirid]="-c 16 --mem=32G -t 01:00:00"   [stage_hirid]="-c 16 --mem=48G -t 01:00:00"
)
CDM_RES="-c 32 --mem=96G -t 01:00:00"

sub() {  # sub <name> <resources> [dependency] -- <run.sbatch args>
    local name=$1 res=$2 dep=$3; shift 4
    sbatch --parsable --export=ALL -J "$name" $res ${dep:+--dependency=afterok:$dep} \
        -o "$OUT/_logs/%x_%j.out" slurm/run.sbatch "$@"
}

STAGE_JOBS=()
for ds in "${DATASETS[@]}"; do
    dep=""
    if [ $SKIP_CONVERT = 0 ]; then
        dep=$(sub "convert_$ds" "${RES[convert_$ds]}" "" -- convert "$ds")
        echo "convert $ds: $dep"
    fi
    s=$(sub "stage_$ds" "${RES[stage_$ds]}" "$dep" -- stage "$ds")
    echo "stage $ds: $s"
    STAGE_JOBS+=("$s")
done
DEPS=$(IFS=:; echo "${STAGE_JOBS[*]}")
c=$(sub cdm "$CDM_RES" "$DEPS" -- cdm --datasets "$(IFS=,; echo "${DATASETS[*]}")")
echo "cdm: $c   (logs in $OUT/_logs)"
