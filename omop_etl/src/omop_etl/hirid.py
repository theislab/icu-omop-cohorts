"""HiRID 1.1.1 -> raw parquet (step 1) -> staging (step 2).

HiRID has one ICU admission per patient (`general_table.patientid`), so patient = stay = person.
All times are offsets from `general_table.admissiontime`. The rules follow ricu's HiRID callbacks
(`ricu-src/R/callback-itm.R`, `data-utils.R`). Values are stored as recorded (HiRID's float32),
with the unit of their variable.

Source timestamps are Spark INT96 with millisecond precision; duckdb reads them as µs TIMESTAMP.
"""

from __future__ import annotations

import logging

from omop_etl import config, db, staging
from omop_etl import convert as cv

log = logging.getLogger("omop_etl")

DATASET = "hirid"
MEASUREMENTS_CSV = config.MAPPINGS / DATASET / "measurements.csv"

HOSPITAL = "Bern University Hospital"

# ricu hirid_death: the death time of a patient discharged dead is the last row of
# invasive MAP (110) or heart rate (200)
DEATH_TIME_ITEMS = (110, 200)

# averaged admission height/weight: rows less than 5 h after admission; there is no lower bound,
# so pre-admission rows count
HEIGHT_ITEM, WEIGHT_ITEM = 10000450, 10000400
ADMISSION_WINDOW_US = 5 * 3600 * 10**6

GCS_TOTAL_CONCEPT = 3007194

MICROGRAM = "µg"  # HiRID writes the micro sign U+00B5, not the Greek mu U+03BC


# ---------------------------------------------------------------------------------------------
# step 1: source -> <out>/raw/hirid/
# ---------------------------------------------------------------------------------------------

def convert(cfg, con) -> None:
    """Copy the columns the staging step needs; the parquet parts of a table become one file."""
    src, raw = cfg.source(DATASET), cfg.raw_dir(DATASET)
    cv.csv_to_parquet(con, src / "reference_data" / "general_table.csv", raw / "general.parquet", {
        "patientid": "BIGINT",
        "admissiontime": "TIMESTAMP",
        "sex": "VARCHAR",
        "age": "BIGINT",
        "discharge_status": "VARCHAR",
    })
    # column names with spaces; pandas' NA strings ('NA', 'None', ...) are NULL
    cv.table_to_parquet(
        con, cv.read_csv_sql(src / "reference_data" / "hirid_variable_reference.csv"),
        raw / "variable_reference.parquet", {
            "source_table": cv.na('"Source Table"'),
            "id": f"CAST({cv.na('ID')} AS BIGINT)",
            "name": cv.na('"Variable Name"'),
        })
    cv.table_to_parquet(
        con, f"read_parquet('{src}/observation_tables/parquet/*.parquet')",
        raw / "observations.parquet", {
            "patientid": "BIGINT",
            "variableid": "BIGINT",
            "datetime": "TIMESTAMP",
            "value": "FLOAT",
        })
    cv.table_to_parquet(
        con, f"read_parquet('{src}/pharma_records/parquet/*.parquet')",
        raw / "pharma.parquet", {
            "patientid": "BIGINT",
            "pharmaid": "BIGINT",
            "infusionid": "BIGINT",
            "givenat": "TIMESTAMP",
            "givendose": "FLOAT",
            "doseunit": "VARCHAR",
            "route": "VARCHAR",
        })


# ---------------------------------------------------------------------------------------------
# step 2: raw -> <out>/staging/hirid/
# ---------------------------------------------------------------------------------------------

def stage(cfg, con) -> None:
    raw, out = cfg.raw_dir(DATASET), cfg.staging_dir(DATASET)
    for table in ("general", "variable_reference", "observations", "pharma"):
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{raw / table}.parquet')")
    n_rows, n_patients = con.sql("SELECT count(*), count(DISTINCT patientid) FROM general").fetchone()
    if n_rows != n_patients:
        raise ValueError(f"general_table has {n_rows} rows for {n_patients} patients; "
                         "hirid staging assumes one ICU stay per patient")
    with db.step("hirid stays"):
        staging.write(con, out, "stays", stays_sql())
    with db.step("hirid measurement"):
        staging.write(con, out, "measurement", measurement_sql())
    with db.step("hirid drug"):
        staging.write(con, out, "drug", drug_sql())


def stays_sql() -> str:
    return f"""
    WITH last_observation AS (
        -- the stay end: last raw observation of any variable, null values included
        -- (ricu id_win_helper.hirid_env; HiRID has no discharge time)
        SELECT patientid, max(datetime) AS last_datetime
        FROM observations
        GROUP BY patientid
    ),
    last_vital AS (
        SELECT patientid, max(datetime) AS last_datetime
        FROM observations
        WHERE variableid IN {DEATH_TIME_ITEMS}
        GROUP BY patientid
    ),
    admission_size AS (
        SELECT o.patientid,
               avg(o.value) FILTER (WHERE o.variableid = {HEIGHT_ITEM}) AS height,
               avg(o.value) FILTER (WHERE o.variableid = {WEIGHT_ITEM}) AS weight
        FROM observations o
        JOIN general g USING (patientid)
        WHERE o.variableid IN ({HEIGHT_ITEM}, {WEIGHT_ITEM})
          AND epoch_us(o.datetime) - epoch_us(g.admissiontime) < {ADMISSION_WINDOW_US}
        GROUP BY o.patientid
    )
    SELECT
        g.patientid AS stay_id,
        CAST(g.patientid AS VARCHAR) AS person_key,
        row_number() OVER (ORDER BY g.patientid) AS stay_order,
        epoch_us(g.admissiontime) AS admit_us,  -- HiRID's own date-shifted time
        year(g.admissiontime) - g.age AS year_of_birth,
        epoch_us(lo.last_datetime) - epoch_us(g.admissiontime) AS los_us,
        g.age,
        CASE WHEN g.sex IN ('M', 'F') THEN g.sex END AS sex,
        '{HOSPITAL}' AS hospital,
        NULL AS unit_type,
        NULL AS admission_source,
        g.discharge_status AS discharge_location,
        CASE g.discharge_status WHEN 'dead' THEN true WHEN 'alive' THEN false END AS mortality,
        coalesce(g.discharge_status = 'dead', false) AS icu_death,
        CASE WHEN g.discharge_status = 'dead'
             THEN epoch_us(lv.last_datetime) - epoch_us(g.admissiontime) END AS death_t_us,
        s.height,
        s.weight
    FROM general g
    LEFT JOIN last_observation lo USING (patientid)
    LEFT JOIN last_vital lv USING (patientid)
    LEFT JOIN admission_size s USING (patientid)
    """


def measurement_sql() -> str:
    """Observations of the variableids in mappings/hirid/measurements.csv, every source row kept.

    Values as recorded, with the unit of their variable in the mapping.
    """
    return f"""
    WITH item AS (
        SELECT * FROM read_csv('{MEASUREMENTS_CSV}', header = true, columns = {{
            'source_table': 'VARCHAR', 'source_code': 'BIGINT', 'source_label': 'VARCHAR',
            'variable': 'VARCHAR', 'concept_id': 'INTEGER', 'unit_concept_id': 'INTEGER',
            'rule': 'VARCHAR'}})
    ),
    scaled AS NOT MATERIALIZED (
        SELECT
            o.patientid AS stay_id,
            epoch_us(o.datetime) - epoch_us(g.admissiontime) AS t_us,
            i.concept_id,
            CAST(o.variableid AS VARCHAR) AS source_value,
            CAST(o.value AS DOUBLE) AS value,
            i.unit_concept_id,
            i.rule
        FROM observations o
        JOIN general g USING (patientid)
        JOIN item i ON i.source_code = o.variableid
        WHERE o.value IS NOT NULL AND o.datetime IS NOT NULL
    ),
    gcs_component AS (
        SELECT stay_id, t_us, concept_id, median(value) AS value
        FROM scaled
        WHERE rule = 'gcs_component'
        GROUP BY stay_id, t_us, concept_id
    ),
    gcs_total AS (
        -- the total where eye, motor and verbal are all recorded at one time (HiRID records no
        -- total itself); a partial sum would look like a real, lower score
        SELECT stay_id, t_us, {GCS_TOTAL_CONCEPT} AS concept_id, 'derived:gcs_total' AS source_value,
               sum(value) AS value, 0 AS unit_concept_id
        FROM gcs_component
        GROUP BY stay_id, t_us
        HAVING count(*) = 3
    ),
    measurement AS (
        SELECT stay_id, t_us, concept_id, source_value, value, unit_concept_id
        FROM scaled
        UNION ALL
        SELECT stay_id, t_us, concept_id, source_value, value, unit_concept_id FROM gcs_total
    )
    SELECT stay_id, t_us, concept_id, source_value, value,
           NULL AS value_text, NULL AS unit_source_value, unit_concept_id
    FROM measurement
    """


def drug_sql() -> str:
    """Every pharma row (mapped to an ingredient or not), doses of g and µg in mg."""
    return f"""
    WITH pharma_name AS (
        SELECT id AS pharmaid, name
        FROM variable_reference
        WHERE source_table = 'Pharma' AND id IS NOT NULL AND name IS NOT NULL
    ),
    administration AS (
        SELECT
            p.patientid AS stay_id,
            epoch_us(p.givenat) - epoch_us(g.admissiontime) AS t_us,
            -- the reference name verbatim (some contain double spaces); the id if it has none
            coalesce(n.name, CAST(p.pharmaid AS VARCHAR)) AS source_value,
            CASE p.doseunit
                WHEN 'g' THEN CAST(p.givendose AS DOUBLE) * 1000
                WHEN '{MICROGRAM}' THEN CAST(p.givendose AS DOUBLE) * 0.001
                ELSE CAST(p.givendose AS DOUBLE)
            END AS dose,
            CASE WHEN p.doseunit IN ('g', '{MICROGRAM}') THEN 'mg' ELSE p.doseunit END AS dose_unit,
            p.route,
            -- infusionid groups the rows of one infusion (ricu hirid_rate/hirid_duration);
            -- the consumer reads item= as the pharmaid
            nullif(concat_ws(';', 'group=' || p.infusionid, 'item=' || p.pharmaid), '') AS sig
        FROM pharma p
        JOIN general g USING (patientid)
        LEFT JOIN pharma_name n USING (pharmaid)
        WHERE p.givenat IS NOT NULL
    )
    -- exact duplicates removed (the pharma table repeats rows)
    SELECT DISTINCT
        stay_id,
        t_us AS t_start_us,
        t_us AS t_end_us,  -- pharma rows are point administrations
        source_value,
        CAST(dose AS DOUBLE) AS quantity,
        dose_unit,
        route,
        sig,
        false AS admission_drug
    FROM administration
    """
