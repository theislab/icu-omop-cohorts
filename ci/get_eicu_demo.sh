#!/usr/bin/env bash
# Download the tables of the eICU Collaborative Research Database Demo 2.0.1 (open data,
# https://physionet.org/content/eicu-crd-demo/2.0.1/) that omop_etl reads into <dir>, unless they are there already.
#
#   ci/get_eicu_demo.sh <dir>
set -euo pipefail
DIR=${1:?usage: ci/get_eicu_demo.sh <dir>}
mkdir -p "$DIR"
for table in patient vitalPeriodic vitalAperiodic nurseCharting lab intakeOutput respiratoryCharting respiratoryCare \
             medication infusiondrug admissiondrug; do
    [ -f "$DIR/$table.csv.gz" ] && continue
    curl -fsSL --retry 5 --retry-delay 10 -o "$DIR/$table.csv.gz.part" \
        "https://physionet.org/files/eicu-crd-demo/2.0.1/$table.csv.gz?download"
    mv "$DIR/$table.csv.gz.part" "$DIR/$table.csv.gz"
done
echo "eICU demo tables in $DIR"
