"""MIMIC-IV 3.1 (`mimic4`): convert the csv.gz tables, stage them for the CDM step.

Every staged row belongs to one ICU stay (`icustays.stay_id`) and carries its time as
microseconds since that stay's `intime`. Rows before admission and after discharge are kept, as
ricu loads them. Hospital-level rows (labevents, microbiologyevents) go to one ICU stay of their
admission as ricu's `upgrade_id` does. Item selection, concepts and the units the items are
recorded in are in `mappings/mimic4/measurements.csv`; values are stored as recorded. Dose units
are in `mappings/mimic4/drug_units.csv`.
"""

from __future__ import annotations

import logging

from omop_etl import config, db, staging, units
from omop_etl.convert import check_raw, csv_to_parquet, run_parallel

log = logging.getLogger("omop_etl")

HOSPITAL = "Beth Israel Deaconess Medical Center"

# hours after ICU admission within which the admission height/weight is taken
ADMISSION_WINDOW_H = 5

# Source tables and the columns staging uses, with their types. Timestamps are
# 'YYYY-MM-DD HH:MM:SS' text in the csv.
TABLES = {
    "hosp/admissions": dict(hadm_id="BIGINT", admission_location="VARCHAR", discharge_location="VARCHAR",
                            hospital_expire_flag="INTEGER", deathtime="TIMESTAMP"),
    "hosp/patients": dict(subject_id="BIGINT", gender="VARCHAR", anchor_age="INTEGER", anchor_year="INTEGER"),
    "hosp/transfers": dict(hadm_id="BIGINT", eventtype="VARCHAR", careunit="VARCHAR", intime="TIMESTAMP"),
    "hosp/labevents": dict(hadm_id="BIGINT", itemid="INTEGER", charttime="TIMESTAMP", valuenum="DOUBLE",
                           valueuom="VARCHAR"),
    # chartdate: the sample date when there is no charttime (specimen_sql)
    "hosp/microbiologyevents": dict(subject_id="BIGINT", hadm_id="BIGINT", chartdate="VARCHAR",
                                    charttime="TIMESTAMP", spec_itemid="VARCHAR", spec_type_desc="VARCHAR"),
    "hosp/diagnoses_icd": dict(subject_id="BIGINT", hadm_id="BIGINT", icd_code="VARCHAR", icd_version="INTEGER"),
    "icu/icustays": dict(subject_id="BIGINT", hadm_id="BIGINT", stay_id="BIGINT", first_careunit="VARCHAR",
                         intime="TIMESTAMP", outtime="TIMESTAMP", los="DOUBLE"),
    "icu/d_items": dict(itemid="INTEGER", label="VARCHAR"),
    "icu/chartevents": dict(stay_id="BIGINT", charttime="TIMESTAMP", itemid="INTEGER", value="VARCHAR",
                            valuenum="DOUBLE", valueuom="VARCHAR"),
    "icu/outputevents": dict(stay_id="BIGINT", charttime="TIMESTAMP", itemid="INTEGER", value="DOUBLE",
                             valueuom="VARCHAR"),
    "icu/inputevents": dict(stay_id="BIGINT", starttime="TIMESTAMP", endtime="TIMESTAMP", itemid="INTEGER",
                            amount="DOUBLE", amountuom="VARCHAR", rate="DOUBLE", rateuom="VARCHAR",
                            patientweight="DOUBLE", linkorderid="BIGINT"),
    "icu/procedureevents": dict(stay_id="BIGINT", itemid="INTEGER", starttime="TIMESTAMP", endtime="TIMESTAMP"),
}


# --- step 1 -------------------------------------------------------------------------------------
def convert(cfg: config.Config, con) -> None:
    src, out = cfg.source("mimic4"), cfg.raw_dir("mimic4")
    jobs = {}
    for name, columns in TABLES.items():
        table = name.split("/")[1]
        jobs[table] = (csv_to_parquet, src / f"{name}.csv.gz", out / f"{table}.parquet", columns)
    # chartevents and labevents dominate; the rest finishes while they run
    run_parallel(con, jobs, workers=len(jobs))


# --- step 2 -------------------------------------------------------------------------------------
def stage(cfg: config.Config, con) -> None:
    raw, out = cfg.raw_dir("mimic4"), cfg.staging_dir("mimic4")
    for source, columns in TABLES.items():
        table = source.split("/")[1]
        check_raw(con, raw / f"{table}.parquet", columns, "mimic4")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{raw / table}.parquet')")
    con.execute(f"""CREATE OR REPLACE TABLE item_map AS
                    SELECT * FROM read_csv('{config.MAPPINGS / 'mimic4/measurements.csv'}', header = true,
                                           types = {{'source_code': 'INTEGER'}})""")
    con.execute(f"""CREATE OR REPLACE TABLE dose_units AS
                    SELECT * FROM read_csv('{config.MAPPINGS / 'mimic4/drug_units.csv'}', header = true,
                                           types = {{'factor': 'DOUBLE'}})""")
    units.load(con)
    con.execute(STAYS_SQL)
    con.execute(STAY_WINDOWS_SQL)
    with db.step("mimic4 stays"):
        staging.write(con, out, "stays", stays_sql())
    with db.step("mimic4 measurement"):
        staging.write(con, out, "measurement", measurement_sql())
    with db.step("mimic4 drug"):
        staging.write(con, out, "drug", drug_sql())
    with db.step("mimic4 procedure"):
        staging.write(con, out, "procedure", procedure_sql())
    with db.step("mimic4 specimen"):
        staging.write(con, out, "specimen", specimen_sql())
    with db.step("mimic4 condition"):
        staging.write(con, out, "condition", condition_sql())


def us(t, origin="s.intime") -> str:
    """SQL: microseconds from `origin` to `t`."""
    return f"(epoch_us({t}) - epoch_us({origin}))"


# Every icustays row of an admission that exists is a stay (no LoS, age or first-stay filter).
# `los` is the float days of icustays (not outtime - intime), turned into microseconds
# (los * 86400 seconds, truncated to µs).
STAYS_SQL = """
CREATE OR REPLACE TABLE stay AS
SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.first_careunit,
       CAST(trunc(i.los * 86400 * 1000000) AS BIGINT) AS los_us,
       a.admission_location, a.discharge_location, a.hospital_expire_flag, a.deathtime
FROM icustays i JOIN admissions a USING (hadm_id)
"""

# ricu's upgrade_id for hospital-level rows: stays of an admission ordered by outtime; stay k
# takes (outtime_{k-1}, outtime_k], the first has no lower and the last no upper bound.
STAY_WINDOWS_SQL = """
CREATE OR REPLACE TABLE stay_window AS
SELECT stay_id, hadm_id, intime,
       lag(outtime) OVER w AS lower_t,
       CASE WHEN row_number() OVER w < count(*) OVER (PARTITION BY hadm_id) THEN outtime END AS upper_t
FROM stay
WINDOW w AS (PARTITION BY hadm_id ORDER BY outtime NULLS FIRST, stay_id)
"""


def assign_to_stay(table: str, time_col: str) -> str:
    """Join hospital-level rows to a stay of their admission. Rows without hadm_id are left out, as
    ricu keys labevents and microbiologyevents by hadm_id and drops rows with a missing id.
    RICU CHOICE (RICU.md at the repository root, "Where we follow ricu"): a row without hadm_id inside an ICU
    stay of the same subject belongs to that stay; adding them is left for later."""
    return f"""
    {table} r JOIN stay_window s ON r.hadm_id = s.hadm_id
        AND (s.lower_t IS NULL OR r.{time_col} > s.lower_t)
        AND (s.upper_t IS NULL OR r.{time_col} <= s.upper_t)"""


def stays_sql() -> str:
    return f"""
WITH
-- ricu: a hospital death goes to the first stay of the admission whose outtime is >= deathtime,
-- else to the last stay
death_stay AS (
    SELECT hadm_id,
           coalesce(arg_min(stay_id, outtime) FILTER (outtime >= deathtime),
                    arg_max(stay_id, outtime), max(stay_id)) AS stay_id
    FROM stay
    WHERE hospital_expire_flag = 1 AND deathtime IS NOT NULL
    GROUP BY hadm_id
),
-- YAIB's miiv_death_icu: an ICU death unless the last ward before the discharge event is not an
-- ICU (ICU wards = distinct first_careunit). The discharge row itself (careunit UNKNOWN) is not
-- the last ward; the row before it is.
transfer_next AS (
    SELECT hadm_id, careunit, intime,
           lead(eventtype) OVER (PARTITION BY hadm_id ORDER BY intime NULLS FIRST) AS next_event
    FROM transfers WHERE hadm_id IS NOT NULL
),
last_ward AS (
    SELECT hadm_id, careunit AS last_ward FROM (
        SELECT *, row_number() OVER (PARTITION BY hadm_id ORDER BY intime DESC NULLS LAST) AS rn
        FROM transfer_next WHERE next_event = 'discharge')
    WHERE rn = 1
),
icu_ward AS (SELECT DISTINCT first_careunit AS ward FROM stay WHERE first_careunit IS NOT NULL),
death AS (
    SELECT d.stay_id, {us('s.deathtime')} AS death_t_us,
           NOT coalesce(w.last_ward IS NOT NULL AND w.last_ward NOT IN (SELECT ward FROM icu_ward), false)
               AS icu_death
    FROM death_stay d JOIN stay s USING (stay_id) LEFT JOIN last_ward w ON w.hadm_id = d.hadm_id
),
-- admission height/weight (stays.height/weight, BlendedICU's averaged value): per charttime the
-- mean over the items present, then per stay the latest non-null value up to
-- {ADMISSION_WINDOW_H} h after intime (no lower bound). The items are charted in different units
-- (in/cm, lb/kg), so averaging them needs one unit: cm and kg. The items themselves are staged
-- in their own units (measurement_sql).
hw_item AS (
    SELECT c.stay_id, c.charttime, c.itemid,
           c.valuenum * CASE c.itemid WHEN 226707 THEN 2.54 WHEN 226531 THEN 0.45359237 ELSE 1.0 END AS v
    FROM chartevents c JOIN stay s USING (stay_id)
    WHERE c.itemid IN (226730, 226707, 226512, 226531, 224639)
      AND c.charttime - s.intime <= INTERVAL {ADMISSION_WINDOW_H} HOUR
),
hw_time AS (
    SELECT stay_id, charttime,
           avg(v) FILTER (itemid IN (226730, 226707)) AS height,
           avg(v) FILTER (itemid IN (226512, 226531, 224639)) AS weight
    FROM hw_item GROUP BY stay_id, charttime
),
hw AS (
    SELECT stay_id,
           arg_max(height, charttime) FILTER (height IS NOT NULL) AS height,
           arg_max(weight, charttime) FILTER (weight IS NOT NULL) AS weight
    FROM hw_time GROUP BY stay_id
)
SELECT s.stay_id,
       CAST(s.subject_id AS VARCHAR) AS person_key,
       row_number() OVER (ORDER BY s.stay_id) AS stay_order,
       epoch_us(s.intime) AS admit_us,  -- MIMIC-IV's own date-shifted time
       p.anchor_year - p.anchor_age AS year_of_birth,
       s.los_us,
       p.anchor_age AS age,  -- anchor_age, not the age at this stay (as ricu)
       p.gender AS sex,
       '{HOSPITAL}' AS hospital,
       s.first_careunit AS unit_type,
       s.admission_location AS admission_source,
       s.discharge_location,
       max(s.hospital_expire_flag) OVER (PARTITION BY s.hadm_id) = 1 AS mortality,
       coalesce(d.icu_death, false) AS icu_death,
       d.death_t_us,
       hw.height, hw.weight
FROM stay s
JOIN patients p USING (subject_id)
LEFT JOIN death d USING (stay_id)
LEFT JOIN hw USING (stay_id)
"""


def measurement_sql() -> str:
    """chartevents, labevents, outputevents rows of the mapped items, as recorded and with the unit
    they were recorded in; plus the per-item height/weight rows and a derived GCS total."""
    # the unit of a row (units.py): temperatures above the threshold are Fahrenheit, also on the
    # Celsius item; CRP from its unit string, mg/L without one (as ricu)
    unit = units.unit_sql("m.rule", "v.value", "us.unit_concept_id", "m.unit_concept_id")
    return f"""
WITH
mapped AS (SELECT * FROM item_map WHERE rule IS DISTINCT FROM 'height_weight'),
chart AS (
    SELECT c.stay_id, {us('c.charttime')} AS t_us, c.itemid, 'chartevents' AS source_table,
           c.valuenum AS value,
           CASE WHEN m.rule = 'text' THEN c.value END AS value_text, c.valueuom AS unit_source_value
    FROM chartevents c JOIN stay s USING (stay_id)
    JOIN mapped m ON m.source_code = c.itemid AND m.source_table = 'chartevents'
    WHERE c.charttime IS NOT NULL
),
lab AS (
    SELECT s.stay_id, {us('r.charttime')} AS t_us, r.itemid, 'labevents' AS source_table,
           r.valuenum AS value, NULL AS value_text, r.valueuom AS unit_source_value
    FROM {assign_to_stay('labevents', 'charttime')}
    WHERE r.itemid IN (SELECT source_code FROM mapped WHERE source_table = 'labevents')
      AND r.charttime IS NOT NULL AND r.valuenum IS NOT NULL
),
output AS (
    SELECT o.stay_id, {us('o.charttime')} AS t_us, o.itemid, 'outputevents' AS source_table,
           o.value AS value, NULL AS value_text, o.valueuom AS unit_source_value
    FROM outputevents o JOIN stay s USING (stay_id)
    WHERE o.itemid IN (SELECT source_code FROM mapped WHERE source_table = 'outputevents')
      AND o.charttime IS NOT NULL
),
timeseries AS (
    SELECT v.stay_id, v.t_us, m.concept_id, CAST(v.itemid AS VARCHAR) AS source_value, m.variable,
           v.value, v.value_text, v.unit_source_value, {unit} AS unit_concept_id
    FROM (SELECT * FROM chart UNION ALL SELECT * FROM lab UNION ALL SELECT * FROM output) v
    JOIN mapped m ON m.source_code = v.itemid AND m.source_table = v.source_table
    LEFT JOIN unit_strings us ON us.unit_string = {units.key('v.unit_source_value')}
    WHERE v.value IS NOT NULL OR v.value_text IS NOT NULL
),
-- GCS total where eye, motor and verbal are all charted at one time (median of repeats)
gcs_part AS (
    SELECT stay_id, t_us, variable, median(value) AS value
    FROM timeseries
    WHERE variable IN ('glasgow_coma_score_eye', 'glasgow_coma_score_motor', 'glasgow_coma_score_verbal')
      AND value IS NOT NULL
    GROUP BY ALL
),
gcs_total AS (
    SELECT stay_id, t_us, 3007194 AS concept_id, 'derived:gcs_total' AS source_value,
           sum(value) AS value, NULL AS value_text, NULL AS unit_source_value,
           0 AS unit_concept_id
    FROM gcs_part GROUP BY stay_id, t_us HAVING count(*) = 3
),
-- every charted height/weight, in the unit of its item
height_weight AS (
    SELECT c.stay_id, {us('c.charttime')} AS t_us, m.concept_id, CAST(c.itemid AS VARCHAR) AS source_value,
           c.valuenum AS value, NULL AS value_text, c.valueuom AS unit_source_value,
           m.unit_concept_id
    FROM chartevents c JOIN stay s USING (stay_id)
    JOIN item_map m ON m.source_code = c.itemid AND m.rule = 'height_weight'
    WHERE c.valuenum IS NOT NULL AND c.charttime IS NOT NULL
)
SELECT stay_id, t_us, concept_id, source_value, value, value_text, unit_source_value, unit_concept_id
FROM timeseries
UNION ALL SELECT * FROM gcs_total
UNION ALL SELECT * FROM height_weight
"""


def drug_sql() -> str:
    """Every inputevents row with its d_items label. The infusion context ricu's rate callbacks
    need (rate, rate unit, weight, infusion group, itemid) goes into `sig` as key=value pairs."""
    return f"""
WITH ie AS (
    SELECT ie.stay_id, d.label,
           {us('ie.starttime')} AS t_start_us,
           {us('coalesce(ie.endtime, ie.starttime)')} AS t_end_us,
           ie.amount, ie.amountuom,
           nullif(concat_ws(';', 'rate=' || CAST(ie.rate AS VARCHAR), 'rate_unit=' || ie.rateuom,
                            'weight_kg=' || CAST(ie.patientweight AS VARCHAR), 'group=' || ie.linkorderid,
                            'item=' || ie.itemid), '') AS sig
    FROM inputevents ie
    JOIN d_items d USING (itemid)
    JOIN stay s USING (stay_id)
    WHERE ie.starttime IS NOT NULL
),
dosed AS (
    SELECT ie.*,
           ie.amount * coalesce(u.factor, 1.0) AS quantity,
           coalesce(u.unit, ie.amountuom) AS dose_unit
    FROM ie LEFT JOIN dose_units u ON u.source_unit = ie.amountuom
)
-- an end before the start becomes the start; the source stop is kept in sig (minutes, truncated)
SELECT DISTINCT stay_id, t_start_us,
       greatest(t_end_us, t_start_us) AS t_end_us,
       label AS source_value, quantity, dose_unit, NULL AS route,
       CASE WHEN t_end_us < t_start_us
            THEN concat_ws(';', sig, 'source_stop_min=' || CAST(trunc((t_end_us - t_start_us) / 60000000) AS BIGINT))
            ELSE sig END AS sig,
       false AS admission_drug
FROM dosed
"""


def procedure_sql() -> str:
    """Ventilation episodes (ricu `mech_vent`): 225792 invasive, 225794 non-invasive."""
    return f"""
SELECT p.stay_id, {us('p.starttime')} AS t_start_us, {us('p.endtime')} AS t_end_us,
       CASE p.itemid WHEN 225792 THEN 37158404 WHEN 225794 THEN 4177224 END AS concept_id,
       CAST(p.itemid AS VARCHAR) AS source_value
FROM procedureevents p JOIN stay s USING (stay_id)
WHERE p.itemid IN (225792, 225794) AND p.starttime IS NOT NULL
"""


def specimen_sql() -> str:
    """Cultures for the Sepsis-3 suspicion of infection (ricu `samp`): one row per distinct
    (stay, time, specimen type), every culture counts regardless of organism.

    A culture without charttime is taken at chartdate + 12 h, as ricu does.
    RICU CHOICE (RICU.md at the repository root, "Where we follow ricu"): a culture goes to the stay of its admission
    by its chartdate (midnight), not its charttime: ricu moves microbiologyevents from the admission to
    a stay with a rolling join on the table's index_var, which is chartdate (`upgrade_id`,
    ricu-src/R/data-utils.R). A culture charted after a stay's discharge on the day of it therefore
    belongs to that stay, with a time after its end."""
    return f"""
SELECT DISTINCT s.stay_id, {us('r.charttime')} AS t_us, 4133843 AS concept_id,
       r.spec_type_desc AS source_value, r.spec_itemid AS source_id
FROM {assign_to_stay("(SELECT * REPLACE (coalesce(charttime, TRY_CAST(chartdate AS TIMESTAMP) + INTERVAL 12 HOUR) AS charttime), TRY_CAST(chartdate AS TIMESTAMP) AS chart_day FROM microbiologyevents)", 'chart_day')}
WHERE r.hadm_id IS NOT NULL AND r.charttime IS NOT NULL
"""


def condition_sql() -> str:
    """Every diagnosis of the admission on each of its ICU stays, at the stay's end (its start if
    the length of stay is unknown). Concepts from mappings/conditions.csv, else 0."""
    return f"""
SELECT s.stay_id, coalesce(s.los_us, 0) AS t_us, coalesce(c.concept_id, 0) AS concept_id,
       d.icd_code AS source_value
FROM diagnoses_icd d JOIN stay s USING (hadm_id)
LEFT JOIN read_csv('{config.MAPPINGS / 'conditions.csv'}', header = true,
                   types = {{'icd_version': 'INTEGER', 'icd_code': 'VARCHAR', 'concept_id': 'INTEGER'}}) c
    ON c.icd_version = d.icd_version AND c.icd_code = d.icd_code
"""
