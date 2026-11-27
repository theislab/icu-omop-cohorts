#!/usr/bin/env bash
# One cohort cell (dataset x task) from $OMOP_CDM / $COHORT_CACHE into $FRAMES_OUT: cohorts/build_cohort.py, with
# its duckdb spill in <out_root>/_duckdb_tmp (DUCK_TMP_BASE overrides) and its resolved paths checked against the
# exported ones. Submitted by run_cohorts.sh, which sets --mem, the log path and the dependency:
#   sbatch --export=ALL run_cohort_cell.sbatch <dataset> <task> <duckdb_mem>
#SBATCH -J omop_cell
#SBATCH -c 8
#SBATCH --mem=110G
#SBATCH -t 08:00:00
set -euo pipefail
DATASET="${1:?usage: run_cohort_cell.sbatch <dataset> <task> <duckdb_mem>}"
TASK="${2:?}"
DUCKMEM="${3:?}"
: "${OMOP_CDM:?}" "${COHORT_CACHE:?}" "${FRAMES_OUT:?}"

REPO="${ICU_OMOP_REPO:?submit through run_cohorts.sh, or source env.sh before sbatch}"
C=$REPO/cohorts
DUCK_TMP_BASE=${DUCK_TMP_BASE:-$(dirname "$FRAMES_OUT")/_duckdb_tmp}
mkdir -p "$DUCK_TMP_BASE"
# python cannot clean up after an OOM kill
trap 'rm -rf "$DUCK_TMP_BASE/${SLURM_JOB_ID}"_* 2>/dev/null || true' EXIT

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-} $DATASET/$TASK duckmem=$DUCKMEM ==="
cd "$C" && source ../env.sh
[ -n "${VIRTUAL_ENV:-}" ] || { echo "env.sh did not activate the environment" >&2; exit 1; }
# COHORTS_CODE / EHRDATA_SRC: the code snapshot of run_cohorts.sh (default: the checkout)
CODE=${COHORTS_CODE:-$C}
[ -n "${EHRDATA_SRC:-}" ] && export PYTHONPATH=$EHRDATA_SRC${PYTHONPATH:+:$PYTHONPATH}
echo "code=$CODE ehrdata_src=${EHRDATA_SRC:-installed}"
python - "$CODE" "$DATASET" "$TASK" "$DUCKMEM" "$DUCK_TMP_BASE" <<'PY'
import os, sys
c, ds, task, duck, tmp = sys.argv[1:]
sys.path.insert(0, c)
import build_cohort as b
b.DUCK_TMP_BASE = tmp
import ehrdata
resolved = {"ehrdata": ehrdata.__file__, "builder": b.__file__, "OMOP": b.OMOP, "CACHE": b.CACHE, "OUT": b.OUT, "DUCK_TMP_BASE": b.DUCK_TMP_BASE}
print("[paths]", resolved, flush=True)
assert (b.OMOP, b.CACHE, b.OUT) == (os.environ["OMOP_CDM"], os.environ["COHORT_CACHE"], os.environ["FRAMES_OUT"]), resolved
assert not b.RICU_HIRID_URINE, resolved
sys.argv = ["build_cohort.py", "--dataset", ds, "--task", task, "--mem", duck, "--threads", "8"]
b.main()
PY
echo "=== cell done ==="
