#!/usr/bin/env bash
# Download the MIMIC-IV Clinical Database Demo 2.2 (open data, https://physionet.org/content/mimic-iv-demo/2.2/)
# into <dir>, unless it is there already.
#
#   ci/get_mimic_iv_demo.sh <dir>
set -euo pipefail
DIR=${1:?usage: ci/get_mimic_iv_demo.sh <dir>}
if [ -f "$DIR/icu/icustays.csv.gz" ]; then
    echo "MIMIC-IV demo found in $DIR"
    exit 0
fi
PARENT=$(dirname "$DIR")
mkdir -p "$PARENT"
curl -fsSL --retry 5 --retry-delay 10 -o "$PARENT/mimic-iv-demo-2.2.zip" https://physionet.org/content/mimic-iv-demo/get-zip/2.2/
unzip -q -o "$PARENT/mimic-iv-demo-2.2.zip" -d "$PARENT"
mv "$PARENT/mimic-iv-clinical-database-demo-2.2" "$DIR"
rm "$PARENT/mimic-iv-demo-2.2.zip"
echo "MIMIC-IV demo downloaded to $DIR"
