#!/usr/bin/env bash
# The whole pipeline on the two open demos, as the demo job of CI runs it. omop_etl turns the MIMIC-IV and the eICU demo
# into an OMOP CDM, the cohort builder makes the frames of every task and both datasets from it, and the notebook runs
# on a CDM of the MIMIC-IV demo alone, which its text describes.
#
#   ci/demo.sh <work dir>
#
# Everything is written below <work dir>. The demos are downloaded to <work dir>/mimic-iv-demo and
# <work dir>/eicu-crd-demo unless MIMIC_IV_DEMO and EICU_DEMO point at copies. The vocabulary is the fixture in
# omop_etl/tests/fixtures/vocabulary unless OMOP_VOCABULARY points at an Athena download. Run it in the environment of
# the repository (source env.sh, or uv run).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${1:?usage: ci/demo.sh <work dir>}"
WORK=$(cd "$1" && pwd)
MIMIC=${MIMIC_IV_DEMO:-$WORK/mimic-iv-demo}
EICU=${EICU_DEMO:-$WORK/eicu-crd-demo}
VOCAB=${OMOP_VOCABULARY:-$REPO/omop_etl/tests/fixtures/vocabulary}
MEM=${DEMO_MEMORY:-4GB}
THREADS=${DEMO_THREADS:-2}

bash "$REPO/ci/get_mimic_iv_demo.sh" "$MIMIC"
bash "$REPO/ci/get_eicu_demo.sh" "$EICU"

echo "=== omop_etl: the demos into an OMOP CDM"
cat > "$WORK/config.yaml" <<YAML
out_dir: $WORK/omop
vocabulary_dir: $VOCAB
sources:
  mimic4: $MIMIC
  eicu: $EICU
duckdb:
  memory_limit: $MEM
  threads: $THREADS
YAML
cd "$REPO"
for dataset in mimic4 eicu; do
    python -m omop_etl convert --dataset "$dataset" --config "$WORK/config.yaml"
    python -m omop_etl stage --dataset "$dataset" --config "$WORK/config.yaml"
done
# the CDM of the MIMIC-IV demo alone, for the notebook, then the CDM of both, for the cohorts
rm -rf "$WORK/omop/cdm_mimic4"
python -m omop_etl cdm --datasets mimic4 --config "$WORK/config.yaml"
mv "$WORK/omop/cdm" "$WORK/omop/cdm_mimic4"
python -m omop_etl cdm --config "$WORK/config.yaml"

echo "=== cohorts: the frames of every task and dataset"
export OMOP_CDM=$WORK/omop/cdm
for dataset in miiv eicu; do
    python cohorts/prefilter.py --dataset "$dataset" --mem "$MEM" --threads "$THREADS"
    for task in mortality24 kidney_function aki sepsis los; do
        python cohorts/build_cohort.py --dataset "$dataset" --task "$task" --mem "$MEM" --threads "$THREADS"
    done
done
python - "$WORK/omop/cohorts/frames" <<'PY'
import sys
from pathlib import Path

import pandas as pd

for task in sorted(Path(sys.argv[1]).iterdir()):
    for dataset in ("miiv", "eicu"):
        frames = {name: pd.read_parquet(task / dataset / f"{name}.parquet") for name in ("dyn", "sta", "outc")}
        stays = frames["sta"]["stay_id"].nunique()
        print(f"{task.name:16} {dataset:5} {stays:5} stays, dyn {frames['dyn'].shape}, outc {frames['outc'].shape}")
        assert stays > 0, (task.name, dataset)
PY

echo "=== the notebook, on the CDM of the MIMIC-IV demo"
mkdir -p "$WORK/notebooks"
OMOP_CDM=$WORK/omop/cdm_mimic4 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
    --output-dir "$WORK/notebooks" cohorts/notebooks/omop_ehrdata_ehrapy_pypots.ipynb
echo "=== demo done: $WORK"
