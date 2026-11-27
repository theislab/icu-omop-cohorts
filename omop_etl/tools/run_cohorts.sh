#!/usr/bin/env bash
# Build the YAIB cohorts (the prefilter and the builder of cohorts/) from a CDM into a new
# directory as SLURM jobs: submits the whole chain at once (SLURM forgets finished jobs, so
# afterok on an old id fails) and prints the job ids.
#
#   omop_etl/tools/run_cohorts.sh <cdm_dir> <out_root> [datasets] [tasks]
#     datasets  comma-separated, default miiv,eicu,hirid
#     tasks     comma-separated, default mortality24,kidney_function,los,aki,sepsis
#     REUSE_CACHE=1  skip the prefilter when <out_root>/cohort_cache/measurement_<ds>.parquet exists
#     BUILD=0        skip the builds (prefilter only)
#     CAI_CONVERT=1  convert eICU ionized calcium by its unit (default: all mmol/L, as ricu reads
#                    it); the eICU frames go to frames_caiconverted/
#     PYTHONPATH     put another ehrapy or ehrdata first, such as a worktree; the jobs run a copy of whatever
#                    `import ehrapy` and `import ehrdata` resolve to, named in _code/*/REVISIONS
#     COHORT_DIAG=1  also write the per-stay label ingredients (diag_*.parquet) next to the frames,
#                    for an aggregate comparison with ricu's
#     EXCLUDE_NODES  nodes no job may run on, as sbatch --exclude (e.g. one whose local disk is full)
#
# Writes <out_root>/cohort_cache/measurement_<ds>.parquet, <out_root>/frames/<task>/<ds>/
# {dyn,sta,outc}.parquet, logs in <out_root>/_logs/, and a job list <out_root>/_logs/jobs_<stamp>.tsv.
# The builder's duckdb spill goes to <out_root>/_duckdb_tmp/ (run_cohort_cell.sbatch). Reads ../../config.env: the jobs
# get SBATCH_PARTITION and SBATCH_QOS, and <out_root> must lie under ICU_OMOP_DATA if that is set.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,23p' "$0" >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ -f "$REPO/config.env" ]; then source "$REPO/config.env"; fi
export ICU_OMOP_REPO=$REPO
C=$REPO/cohorts
TOOLS=$REPO/omop_etl/tools

CDM=$(realpath -e "$1")
mkdir -p "$2"
OUT=$(realpath -e "$2")
IFS=, read -ra DATASETS <<< "${3:-miiv,eicu,hirid}"
IFS=, read -ra TASKS <<< "${4:-mortality24,kidney_function,los,aki,sepsis}"

if [ -n "${ICU_OMOP_DATA:-}" ]; then
  case "$OUT/" in
    "$(realpath -m "$ICU_OMOP_DATA")"/*) ;;
    *) echo "out_root must lie under ICU_OMOP_DATA=$ICU_OMOP_DATA (got $OUT)" >&2; exit 2 ;;
  esac
fi
for t in VISIT_OCCURRENCE PERSON MEASUREMENT DEATH DRUG_EXPOSURE PROCEDURE_OCCURRENCE SPECIMEN CARE_SITE; do
  [ -f "$CDM/$t.parquet" ] || { echo "missing $CDM/$t.parquet" >&2; exit 2; }
done
for ds in "${DATASETS[@]}"; do
  case $ds in miiv|eicu|hirid) ;; *) echo "unknown dataset $ds" >&2; exit 2 ;; esac
done
for task in "${TASKS[@]}"; do
  case $task in mortality24|kidney_function|los|aki|sepsis) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
done

export OMOP_CDM=$CDM
export COHORT_CACHE=$OUT/cohort_cache
export FRAMES_OUT=$OUT/frames
# The jobs run a copy of the cohort code and of ehrdata's and ehrapy's sources taken now, so that
# edits made while they wait in the queue do not reach them. The revisions go to _code/REVISIONS.
EHRDATA_DIR=$(cd "$C" && source ../env.sh >/dev/null 2>&1 && python -c "import ehrdata, os; print(os.path.dirname(ehrdata.__file__))")
EHRAPY_DIR=$(cd "$C" && source ../env.sh >/dev/null 2>&1 && python -c "import ehrapy, os; print(os.path.dirname(ehrapy.__file__))")
SNAP=$OUT/_code/$(date +%Y%m%d_%H%M%S)
mkdir -p "$SNAP"
mkdir "$SNAP/cohorts" && cp "$C"/*.py "$SNAP/cohorts/" && cp -r "$EHRDATA_DIR" "$SNAP/ehrdata" && cp -r "$EHRAPY_DIR" "$SNAP/ehrapy"
find "$SNAP" -name __pycache__ -prune -exec rm -rf {} +
# a package's revision: the commit of its checkout, or the version and commit it was installed from (direct_url.json)
revision() {
  if git -C "$2" rev-parse HEAD >/dev/null 2>&1; then
    echo "$(git -C "$2" rev-parse HEAD) $(git -C "$2" rev-parse --abbrev-ref HEAD) $(git -C "$2" status --porcelain -- . | wc -l) changed files"
  else
    (cd "$C" && source ../env.sh >/dev/null 2>&1 && python -c '
import json, sys
from importlib.metadata import distribution
dist = distribution(sys.argv[1])
url = json.loads(dist.read_text("direct_url.json") or "{}")
print(dist.version, url.get("url", ""), url.get("vcs_info", {}).get("commit_id", ""))' "$1")
  fi
}
{ echo "cohorts $(git -C "$C" rev-parse HEAD) $(git -C "$C" status --porcelain -- . | wc -l) changed files"
  echo "ehrdata $EHRDATA_DIR $(revision ehrdata "$EHRDATA_DIR")"
  echo "ehrapy $EHRAPY_DIR $(revision ehrapy "$EHRAPY_DIR")"
  echo "switches CAI_CONVERT=${CAI_CONVERT:-0} COHORT_DIAG=${COHORT_DIAG:-0}"
} > "$SNAP/REVISIONS"
export COHORTS_CODE=$SNAP/cohorts EHRDATA_SRC=$SNAP
echo "code snapshot: $SNAP"
unset RICU_HIRID_URINE DYN_AGG
mkdir -p "$OUT/_logs" "$COHORT_CACHE" "$FRAMES_OUT"
L=(-o "$OUT/_logs/%x_%j.log" -e "$OUT/_logs/%x_%j.log")
[ -n "${EXCLUDE_NODES:-}" ] && L+=(--exclude="$EXCLUDE_NODES")
JOBS="$OUT/_logs/jobs_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "job_id\tname\tdepends_on" > "$JOBS"
echo "OMOP_CDM=$OMOP_CDM"; echo "COHORT_CACHE=$COHORT_CACHE"; echo "FRAMES_OUT=$FRAMES_OUT"

cd "$C"
declare -A PRE
for ds in "${DATASETS[@]}"; do
  if [ "${REUSE_CACHE:-0}" = 1 ] && [ -f "$COHORT_CACHE/measurement_$ds.parquet" ]; then
    PRE[$ds]=""; echo "pre_$ds: reusing $COHORT_CACHE/measurement_$ds.parquet"; continue
  fi
  PRE[$ds]=$(sbatch --parsable --export=ALL "${L[@]}" -J "pre_$ds" \
    run_prefilter.sbatch "$ds")
  echo -e "${PRE[$ds]}\tpre_$ds\t" >> "$JOBS"; echo "pre_$ds: ${PRE[$ds]}"
done
for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    # measured on the full data: miiv/hirid fill a 90 GB duckdb limit; eicu needs more
    mem=110G; duck=90GB
    if [ "$ds" = eicu ]; then
      case $task in
        mortality24|kidney_function) mem=178G; duck=40GB ;;
        aki|sepsis)                  mem=300G; duck=250GB ;;
        los)                         mem=360G; duck=250GB ;;
      esac
    fi
    dep=()
    [ -n "${PRE[$ds]}" ] && dep=(--dependency=afterok:"${PRE[$ds]}")
    if [ "${BUILD:-1}" = 1 ]; then
    id=$(sbatch --parsable --export=ALL "${L[@]}" -J "build_${ds}_${task}" --mem=$mem -t 08:00:00 \
      "${dep[@]}" "$TOOLS/run_cohort_cell.sbatch" "$ds" "$task" "$duck")
    echo -e "$id\tbuild_${ds}_${task}\t${PRE[$ds]}" >> "$JOBS"; echo "build_${ds}_${task}: $id (mem $mem, duckdb $duck)"
    fi
  done
done
echo "jobs: $JOBS"
