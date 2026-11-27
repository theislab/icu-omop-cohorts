"""Pre-filter the OMOP MEASUREMENT table (~1.4 B rows) to one dataset's cohort cache.

Reads the CDM's VISIT_OCCURRENCE and MEASUREMENT. Keeps the rows whose concept is a target of the
crosswalk (`TARGET_MEASUREMENT_CONCEPT_IDS`: the 48 variables' concepts, the statics, the SOFA
inputs) or whose source value it names (`SOURCE_VALUE_CONCEPTS`: HiRID fluid balance and airway),
for the persons with a visit of the dataset (visit_source_value LIKE '{prefix}-%'). Writes their
ids, times, value, unit and source columns to `<COHORT_CACHE>/measurement_<dataset>.parquet`, which
the builder reads instead of the full table. Task-agnostic, so one cache serves all 5 tasks.

One heavy pass over MEASUREMENT, best run as a job (run_prefilter.sbatch). DuckDB is out-of-core and
reads only the 8 needed columns; memory_limit and a temp_directory next to the cache guard against
OOM and ENOSPC (the spill of filtered patient rows is derived data and must stay where the outputs
may be, not in /tmp).
Prints metadata/counts only (no patient values).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb  # noqa: E402
from duck_tmp import make_job_tmp  # noqa: E402
from yaib_omop_crosswalk import (  # noqa: E402
    COHORT_CACHE,
    DATASET_PREFIX,
    OMOP_CDM,
    SOURCE_VALUE_CONCEPTS,
    TARGET_MEASUREMENT_CONCEPT_IDS,
    require_cdm,
)

# OMOP_CDM / COHORT_CACHE (environment) select the CDM and the cache dir; see the crosswalk
OMOP = OMOP_CDM
OUT_DIR = COHORT_CACHE
# NB previously a single shared "_duckdb_tmp" for every invocation -- concurrent prefilters (the
# post-pipeline chain runs one per dataset) would then share a temp_directory and corrupt each
# other's spill files. Now per-job, and removed at exit.
DUCK_TMP_BASE = OUT_DIR + "/_duckdb_tmp"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="miiv", choices=list(DATASET_PREFIX))
    ap.add_argument("--mem", default="56GB")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    require_cdm()

    prefix = DATASET_PREFIX[args.dataset]
    os.makedirs(OUT_DIR, exist_ok=True)
    duck_tmp = make_job_tmp(DUCK_TMP_BASE, args.dataset)
    out = f"{OUT_DIR}/measurement_{args.dataset}.parquet"

    print(f"[env] duckdb={duckdb.__version__}  dataset={args.dataset} prefix='{prefix}-' "
          f"n_target_concepts={len(TARGET_MEASUREMENT_CONCEPT_IDS)} cdm={OMOP} out={out}", flush=True)

    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{args.mem}'")
    con.execute(f"SET temp_directory='{duck_tmp}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"CREATE VIEW visit_occurrence AS SELECT * FROM parquet_scan('{OMOP}/VISIT_OCCURRENCE.parquet')")
    con.execute(f"CREATE VIEW measurement AS SELECT * FROM parquet_scan('{OMOP}/MEASUREMENT.parquet')")

    con.execute(
        f"CREATE TABLE ds_persons AS SELECT DISTINCT person_id FROM visit_occurrence "
        f"WHERE visit_source_value LIKE '{prefix}-%'"
    )
    n_persons = con.execute("SELECT COUNT(*) FROM ds_persons").fetchone()[0]
    n_visits = con.execute(
        f"SELECT COUNT(*) FROM visit_occurrence WHERE visit_source_value LIKE '{prefix}-%'"
    ).fetchone()[0]
    print(f"[{args.dataset}] persons={n_persons} visits={n_visits}", flush=True)

    ids = ",".join(str(i) for i in TARGET_MEASUREMENT_CONCEPT_IDS)
    # the CDM carries a text value (e.g. GCS "No Response-ETT")
    has_text = "value_source_value" in {r[0] for r in con.execute("DESCRIBE measurement").fetchall()}
    text_col = ", m.value_source_value" if has_text else ""

    src_vals = ", ".join(f"'{v}'" for v in SOURCE_VALUE_CONCEPTS)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
          SELECT m.person_id, m.visit_occurrence_id, m.measurement_concept_id,
                 m.measurement_date, m.measurement_datetime,
                 m.value_as_number, m.unit_concept_id, m.unit_source_value,
                 -- carries the SOURCE itemid for the per-item height/weight rows, so a consumer
                 -- can select ricu's items (226512 weight / 226707 height) instead of BlendedICU's
                 -- mean over five. Averaged rows have 'raw_height'/'raw_weight' here.
                 m.measurement_source_value{text_col}
          FROM measurement m
          WHERE (m.measurement_concept_id IN ({ids})
                 -- Concepts identified by SOURCE VALUE rather than a standard concept id. HiRID's
                 -- cumulative fluid balance has no standard OMOP concept (every LOINC candidate is
                 -- a fixed window, not a running total), so it is written with concept_id 0 and
                 -- carries its identity in measurement_source_value. Filtering on id 0 would pull
                 -- in every unmapped measurement, hence the explicit name.
                 OR m.measurement_source_value IN ({src_vals}))
            AND m.person_id IN (SELECT person_id FROM ds_persons)
        ) TO '{out}' (FORMAT PARQUET)
        """
    )
    dt = time.time() - t0

    n_out = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{out}')").fetchone()[0]
    n_conc = con.execute(f"SELECT COUNT(DISTINCT measurement_concept_id) FROM parquet_scan('{out}')").fetchone()[0]
    sz = os.path.getsize(out) / 1e9
    print(f"[{args.dataset}] WROTE {out}", flush=True)
    print(f"[{args.dataset}] rows={n_out} matched_concepts={n_conc}/{len(TARGET_MEASUREMENT_CONCEPT_IDS)} "
          f"size={sz:.2f}GB scan={dt:.0f}s", flush=True)
    print("[OK] pre-filter done.", flush=True)


if __name__ == "__main__":
    main()
