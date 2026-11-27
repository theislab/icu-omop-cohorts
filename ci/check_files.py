"""Reject files that must not enter the repository: data files, tables outside the mapping tables, the CDM schema
and the test fixtures, and notebooks stored with their outputs. Run by pre-commit on the files of a commit.

One notebook may keep its outputs, those of a run on the MIMIC-IV demo (open data): its cell that counts the CDM's
stays must show the demo's 140 ICU stays of 100 persons, its outputs must not name a local path, and it must stay
under 1 MB. A run on any other CDM, such as a credentialed one, is rejected.

    python ci/check_files.py <file> ...
"""

import json
import os
import re
import sys
from pathlib import PurePosixPath

DATA = {".parquet", ".duckdb", ".db", ".sqlite", ".h5", ".h5ad", ".hdf5", ".npy", ".npz", ".pkl", ".pickle", ".pt",
        ".pth", ".feather", ".arrow", ".zarr", ".fst", ".rds", ".rdata"}
TABLES = {".csv", ".tsv", ".gz", ".zip"}
TABLE_DIRS = ("omop_etl/mappings/", "omop_etl/schema/", "omop_etl/tests/fixtures/")

WITH_OUTPUTS = "cohorts/notebooks/omop_ehrdata_ehrapy_pypots.ipynb"
DEMO_COUNTS = re.compile(r"icu_stays\s+persons\s*\n0\s+140\s+100\s*$")
LOCAL_PATH = re.compile(r"/(?:home|Users|tmp|scratch|localscratch|lustre|groups|mnt)/")
MAX_BYTES = 1_000_000


def outputs_text(cell: dict) -> str:
    parts = []
    for output in cell.get("outputs", []):
        parts.append("".join(output.get("text", "")))
        for value in output.get("data", {}).values():
            parts.append("".join(value) if isinstance(value, list) else json.dumps(value))
    return "\n".join(parts)


def notebook_problem(path: str) -> str | None:
    with open(path) as f:
        cells = [c for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    if not any(c.get("outputs") or c.get("execution_count") for c in cells):
        return None
    if path != WITH_OUTPUTS:
        return f"a notebook with outputs; store it without them (only {WITH_OUTPUTS} keeps its outputs)"
    plain = ["".join(o.get("data", {}).get("text/plain", "")) for c in cells for o in c.get("outputs", [])]
    if not any(DEMO_COUNTS.search(text) for text in plain):
        return "outputs that do not show the MIMIC-IV demo's 140 ICU stays of 100 persons; run it on the demo"
    if any(LOCAL_PATH.search(outputs_text(c)) for c in cells):
        return "outputs that name a local path"
    if os.path.getsize(path) > MAX_BYTES:
        return f"larger than {MAX_BYTES // 1_000_000} MB"
    return None


def problems(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in DATA:
        return "a data file"
    if suffix in TABLES and not path.startswith(TABLE_DIRS):
        return f"a table outside {', '.join(TABLE_DIRS)}"
    if suffix == ".ipynb":
        return notebook_problem(path)
    return None


def main(paths: list[str]) -> int:
    found = [(p, why) for p in paths if (why := problems(p))]
    for path, why in found:
        print(f"{path}: {why}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
