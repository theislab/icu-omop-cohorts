"""eICU-CRD 2.0 -> staging.

Every eICU time is an integer number of minutes from unit admission (`*offset` columns); rows
before admission and after discharge are kept. The stay is `patientunitstayid`, the person
`uniquepid`. Source docs: https://eicu-crd.mit.edu/eicutables/ ; item choices follow ricu's
concept dictionary (eth-mds/ricu, `inst/extdata/config/concept-dict.json`).

Values are stored as recorded, with the unit they were recorded in; converting units is left to
the consumer. Text is read as a number where it is one after trimming (`num`), else kept as text.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omop_etl import config, db, staging, units
from omop_etl.convert import check_raw, csv_to_parquet, run_parallel

log = logging.getLogger("omop_etl")

US_PER_MIN = 60_000_000
DAY_US = 86_400 * 10**6

# --- step 1: csv.gz -> parquet ----------------------------------------------------------------

# table -> (source file, {column: type}); only the columns `stage` reads
SOURCE_TABLES = {
    "vitalperiodic": ("vitalPeriodic.csv.gz", {
        "patientunitstayid": "INTEGER", "observationoffset": "INTEGER",
        **{c: "DOUBLE" for c in ("temperature", "sao2", "heartrate", "respiration", "cvp",
                                 "systemicsystolic", "systemicdiastolic", "systemicmean",
                                 "st1", "st2", "st3")}}),
    "nursecharting": ("nurseCharting.csv.gz", {
        "patientunitstayid": "INTEGER", "nursingchartoffset": "INTEGER",
        "nursingchartcelltypevalname": "VARCHAR", "nursingchartvalue": "VARCHAR"}),
    "lab": ("lab.csv.gz", {
        "patientunitstayid": "INTEGER", "labresultoffset": "INTEGER", "labname": "VARCHAR",
        "labresult": "DOUBLE", "labmeasurenameinterface": "VARCHAR"}),
    "vitalaperiodic": ("vitalAperiodic.csv.gz", {
        "patientunitstayid": "INTEGER", "observationoffset": "INTEGER",
        "noninvasivesystolic": "DOUBLE", "noninvasivediastolic": "DOUBLE", "noninvasivemean": "DOUBLE"}),
    "intakeoutput": ("intakeOutput.csv.gz", {
        "patientunitstayid": "INTEGER", "intakeoutputoffset": "INTEGER", "celllabel": "VARCHAR",
        "cellvaluenumeric": "DOUBLE"}),
    "respiratorycharting": ("respiratoryCharting.csv.gz", {
        "patientunitstayid": "INTEGER", "respchartoffset": "INTEGER", "respcharttypecat": "VARCHAR",
        "respchartvaluelabel": "VARCHAR", "respchartvalue": "VARCHAR"}),
    "medication": ("medication.csv.gz", {
        "patientunitstayid": "INTEGER", "drugstartoffset": "INTEGER", "drugstopoffset": "INTEGER",
        "drugname": "VARCHAR", "dosage": "VARCHAR", "routeadmin": "VARCHAR"}),
    "infusiondrug": ("infusionDrug.csv.gz", {
        "patientunitstayid": "INTEGER", "infusionoffset": "INTEGER", "drugname": "VARCHAR",
        "drugrate": "VARCHAR", "drugamount": "DOUBLE", "patientweight": "DOUBLE"}),
    "admissiondrug": ("admissionDrug.csv.gz", {
        "patientunitstayid": "INTEGER", "drugoffset": "INTEGER", "drugname": "VARCHAR",
        "drugdosage": "DOUBLE", "drugunit": "VARCHAR"}),
    "respiratorycare": ("respiratoryCare.csv.gz", {
        "patientunitstayid": "INTEGER", "ventstartoffset": "INTEGER", "ventendoffset": "INTEGER",
        "priorventstartoffset": "INTEGER", "priorventendoffset": "INTEGER"}),
    "patient": ("patient.csv.gz", {
        "patientunitstayid": "INTEGER", "uniquepid": "VARCHAR", "hospitalid": "INTEGER",
        "unittype": "VARCHAR", "unitadmitsource": "VARCHAR", "unitdischargelocation": "VARCHAR",
        "unitdischargestatus": "VARCHAR", "unitdischargeoffset": "INTEGER", "gender": "VARCHAR",
        "age": "VARCHAR", "admissionheight": "DOUBLE", "admissionweight": "DOUBLE",
        "patienthealthsystemstayid": "INTEGER", "hospitaldischargeyear": "INTEGER",
        "hospitaladmittime24": "VARCHAR", "hospitaladmitoffset": "INTEGER",
        "hospitaldischargeoffset": "INTEGER"}),
}


def source_file(src: Path, name: str) -> Path:
    """The file `name` in `src`, matched without case: the demo (eicu-crd-demo 2.0.1) names `infusiondrug.csv.gz` and
    `admissiondrug.csv.gz` in lower case, the full release `infusionDrug.csv.gz` and `admissionDrug.csv.gz`."""
    if (src / name).exists():
        return src / name
    matches = [p for p in src.iterdir() if p.name.lower() == name.lower()]
    return matches[0] if matches else src / name


def convert(cfg: config.Config, con) -> None:
    src, out = cfg.source("eicu"), cfg.raw_dir("eicu")
    # gzip is read by one thread per file: the big tables first, side by side
    jobs = {t: (csv_to_parquet, source_file(src, f), out / f"{t}.parquet", cols)
            for t, (f, cols) in SOURCE_TABLES.items()}
    rows = run_parallel(con, jobs, workers=6)
    log.info("eicu raw rows: %s", rows)


# --- step 2: raw parquet -> staging -----------------------------------------------------------

def stage(cfg: config.Config, con) -> None:
    raw, out = cfg.raw_dir("eicu"), cfg.staging_dir("eicu")
    for t, (_, columns) in SOURCE_TABLES.items():
        check_raw(con, raw / f"{t}.parquet", columns, "eicu")
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{raw / (t + '.parquet')}')")
    create_macros(con)
    units.load(con)
    with db.step("eicu stays"):
        staging.write(con, out, "stays", STAYS_SQL)
        n, late = con.sql(f"WITH RECURSIVE {HOSPITAL_PLACEMENT} SELECT count(*), count(*) FILTER "
                          "(WHERE year(make_timestamp(discharge_us)) > year) FROM placed").fetchone()
        log.info("eicu hospital stays placed: %s, ending after their discharge year: %s", f"{n:,}", f"{late:,}")
    with db.step("eicu measurement"):
        n, empty = con.sql("SELECT count(*), count(*) FILTER (WHERE labresult IS NULL) FROM lab").fetchone()
        log.info("eicu lab rows without a result, left out: %s of %s", f"{empty:,}", f"{n:,}")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE mapping AS SELECT * FROM read_csv(
            '{config.MAPPINGS / 'eicu' / 'measurements.csv'}', header = true, quote = '"',
            columns = {{'source_table': 'VARCHAR', 'source_code': 'VARCHAR', 'source_label': 'VARCHAR',
                        'variable': 'VARCHAR', 'concept_id': 'INTEGER', 'unit_concept_id': 'INTEGER',
                        'rule': 'VARCHAR'}})""")
        staging.write(con, out, "measurement", MEASUREMENT_SQL)
    with db.step("eicu drug"):
        staging.write(con, out, "drug", DRUG_SQL)
    with db.step("eicu procedure"):
        staging.write(con, out, "procedure", PROCEDURE_SQL)


def create_macros(con) -> None:
    con.execute(f"""
        CREATE OR REPLACE MACRO num(s) AS TRY_CAST(trim(s) AS DOUBLE);
        CREATE OR REPLACE MACRO is_int(s) AS regexp_matches(trim(s), '^[+-]?\\d+$');
        CREATE OR REPLACE MACRO minutes_us(m) AS CAST(m AS BIGINT) * {US_PER_MIN};
        CREATE OR REPLACE MACRO unit_is(unit, pattern) AS
            coalesce(regexp_matches(unit, '(?i)' || pattern), false);
    """)


# One row per unit stay. stay_order is ascending patientunitstayid; it picks the stay PERSON takes
# sex and year of birth from. Unit type, admission source and discharge location are the source
# strings.
#
# eICU gives times as minutes from unit admission, plus the hospital discharge year, the clock time
# of the hospital admission and the unit admission's offset from it. A person's hospital stays
# (patienthealthsystemstayid) are taken in order of discharge year, age, a fatal stay last, and id.
# The first starts on 1 January of its discharge year, at its admission clock time (a year earlier
# if it would otherwise end after its discharge year); each later one on the same terms, but not
# before the calendar day after the previous discharge. A person with many stays in one year can so
# end up past that year (logged). Unit stays sit at their true offset from their hospital
# admission, so the unit stays of one hospital stay keep their real spacing.
HOSPITAL_PLACEMENT = f"""
p AS (
    -- ages above 89 are recorded as '> 89'; ricu reads them as 90
    SELECT *, CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS DOUBLE) END AS age_years
    FROM patient
),
hospital_stay AS (
    SELECT uniquepid, patienthealthsystemstayid AS hs,
           any_value(hospitaldischargeyear) AS year,
           coalesce(CAST(epoch(TRY_CAST(any_value(hospitaladmittime24) AS TIME)) * 1000000 AS BIGINT), 0) AS tod_us,
           CAST(greatest(coalesce(max(hospitaldischargeoffset - hospitaladmitoffset), 0), 0) AS BIGINT)
               * {US_PER_MIN} AS los_us,
           min(age_years) AS age_min,
           coalesce(bool_or(unitdischargestatus = 'Expired'), false) AS fatal
    FROM p WHERE hospitaldischargeyear IS NOT NULL
    GROUP BY uniquepid, patienthealthsystemstayid
),
earliest AS (
    SELECT *,
           epoch_us(make_timestamp(year - CASE WHEN year(make_timestamp(
                        epoch_us(make_timestamp(year, 1, 1, 0, 0, 0)) + tod_us + los_us)) > year THEN 1 ELSE 0 END,
                    1, 1, 0, 0, 0)) + tod_us AS earliest_us,
           row_number() OVER (PARTITION BY uniquepid ORDER BY year, age_min, fatal, hs) AS k
    FROM hospital_stay
),
placed(uniquepid, k, hs, year, admit_us, discharge_us) AS (
    SELECT uniquepid, k, hs, year, earliest_us, earliest_us + los_us FROM earliest WHERE k = 1
    UNION ALL
    SELECT uniquepid, k, hs, year, start_us, start_us + los_us FROM (
        SELECT e.uniquepid, e.k, e.hs, e.year, e.los_us,
               greatest(e.earliest_us, (pl.discharge_us // {DAY_US} + 1) * {DAY_US} + e.tod_us) AS start_us
        FROM placed pl JOIN earliest e ON e.uniquepid = pl.uniquepid AND e.k = pl.k + 1)
)"""
STAYS_SQL = f"""
WITH RECURSIVE {HOSPITAL_PLACEMENT}
SELECT
    p.patientunitstayid AS stay_id,
    p.uniquepid AS person_key,
    row_number() OVER (ORDER BY p.patientunitstayid) AS stay_order,
    -- hospitaladmitoffset: minutes from the unit admission to the hospital admission
    pl.admit_us - minutes_us(p.hospitaladmitoffset) AS admit_us,
    p.hospitaldischargeyear - p.age_years AS year_of_birth,
    minutes_us(p.unitdischargeoffset) AS los_us,
    p.age_years AS age,
    CASE p.gender WHEN 'Male' THEN 'M' WHEN 'Female' THEN 'F' END AS sex,
    CAST(p.hospitalid AS VARCHAR) AS hospital,
    p.unittype AS unit_type,
    p.unitadmitsource AS admission_source,
    p.unitdischargelocation AS discharge_location,
    CASE p.unitdischargestatus WHEN 'Expired' THEN true WHEN 'Alive' THEN false END AS mortality,
    -- 'Expired' at unit discharge is ricu's ICU death; its time is the discharge
    coalesce(p.unitdischargestatus = 'Expired', false) AS icu_death,
    CASE WHEN p.unitdischargestatus = 'Expired' THEN minutes_us(p.unitdischargeoffset) END AS death_t_us,
    p.admissionheight AS height,
    p.admissionweight AS weight
FROM p LEFT JOIN placed pl ON pl.uniquepid = p.uniquepid AND pl.hs = p.patienthealthsystemstayid
"""

# Source rows of every charted table as (stay_id, minute, map_code, source_value, value,
# value_text, unit), joined to mappings/eicu/measurements.csv on map_code. Only mapped labels
# survive. Wide vitals are unpivoted table by table (UNPIVOT drops NULL values).

MEASUREMENT_SQL = rf"""
WITH
lab_rows AS (
    SELECT patientunitstayid AS stay_id, labresultoffset AS minute,
           'lab:' || labname AS source_value,
           labresult AS value, NULL AS value_text,
           -- the unit the row was recorded in (several conversions depend on it)
           labmeasurenameinterface AS unit_source_value
    FROM lab
    WHERE labresult IS NOT NULL
),
resp_rows AS (
    -- '35%' -> 35; text values such as
    -- the ventilator mode keep the text
    SELECT patientunitstayid AS stay_id, respchartoffset AS minute,
           'respiratorycharting:' || respchartvaluelabel AS source_value,
           num(replace(respchartvalue, '%', '')) AS value,
           respchartvalue AS raw_text, NULL AS unit_source_value
    FROM respiratorycharting
    WHERE respchartvalue IS NOT NULL
),
nurse_rows AS (
    -- keyed on nursingchartcelltypevalname: a ...vallabel bundles several measurements
    SELECT patientunitstayid AS stay_id, nursingchartoffset AS minute,
           'nursecharting:' || nursingchartcelltypevalname AS source_value,
           num(nursingchartvalue) AS value,
           nursingchartvalue AS raw_text, NULL AS unit_source_value
    FROM nursecharting
    WHERE nursingchartvalue IS NOT NULL
),
io_labels AS (
    -- catheter output cells are urine output (ricu's urine regex); ventricular (CSF) drains
    -- match the regex too and are left out. 'URINE CATHETER' does not match and maps itself.
    SELECT celllabel,
           CASE WHEN regexp_matches(celllabel, '(?i)catheter.+output|output.+catheter')
                     AND NOT regexp_matches(celllabel, '(?i)ventric')
                THEN 'Urine' ELSE celllabel END AS map_label
    FROM (SELECT DISTINCT celllabel FROM intakeoutput)
),
io_rows AS (
    SELECT io.patientunitstayid AS stay_id, io.intakeoutputoffset AS minute,
           'intakeoutput:' || l.map_label AS map_code,
           'intakeoutput:' || io.celllabel AS source_value,
           io.cellvaluenumeric AS value, NULL AS value_text, NULL AS unit_source_value
    FROM intakeoutput io JOIN io_labels l USING (celllabel)
    WHERE io.cellvaluenumeric IS NOT NULL
),
periodic_rows AS (
    UNPIVOT (SELECT patientunitstayid, observationoffset, temperature, sao2, heartrate, respiration,
                    cvp, systemicsystolic, systemicdiastolic, systemicmean, st1, st2, st3
             FROM vitalperiodic)
    ON temperature, sao2, heartrate, respiration, cvp, systemicsystolic, systemicdiastolic,
       systemicmean, st1, st2, st3
    INTO NAME col VALUE raw
),
aperiodic_rows AS (
    UNPIVOT (SELECT patientunitstayid, observationoffset, noninvasivesystolic,
                    noninvasivediastolic, noninvasivemean FROM vitalaperiodic)
    ON noninvasivesystolic, noninvasivediastolic, noninvasivemean
    INTO NAME col VALUE raw
),
source_rows AS (
    SELECT stay_id, minute, source_value AS map_code, source_value, value, value_text, unit_source_value
    FROM lab_rows
    UNION ALL
    SELECT stay_id, minute, source_value, source_value, value,
           CASE WHEN value IS NULL THEN raw_text END, unit_source_value
    FROM resp_rows
    UNION ALL
    SELECT stay_id, minute, source_value, source_value, value,
           CASE WHEN value IS NULL THEN raw_text END, unit_source_value
    FROM nurse_rows
    UNION ALL
    SELECT stay_id, minute, map_code, source_value, value, value_text, unit_source_value FROM io_rows
    UNION ALL
    SELECT patientunitstayid, observationoffset, 'vitalperiodic:' || col, 'vitalperiodic:' || col,
           CAST(raw AS DOUBLE), NULL, NULL
    FROM periodic_rows
    UNION ALL
    SELECT patientunitstayid, observationoffset, 'vitalaperiodic:' || col, 'vitalaperiodic:' || col,
           CAST(raw AS DOUBLE), NULL, NULL
    FROM aperiodic_rows
),
converted AS (
    -- the value as recorded, with the unit it was recorded in (units.py: a unit varying per row
    -- from the row's unit string, a temperature above the threshold in Fahrenheit). A lab FiO2 in
    -- mm(hg) is no fraction; ricu sets it to NA, and the row is left out.
    SELECT r.stay_id, r.minute, m.concept_id, r.source_value, r.value_text, r.unit_source_value,
           {units.unit_sql('m.rule', 'r.value', 'us.unit_concept_id', 'm.unit_concept_id')} AS unit_concept_id,
           CASE WHEN m.rule = 'fio2_mmhg_to_null' AND unit_is(r.unit_source_value, 'mm\s*\(?\s*hg')
                THEN NULL ELSE r.value END AS value
    FROM source_rows r JOIN mapping m ON m.source_code = r.map_code
    LEFT JOIN unit_strings us ON us.unit_string = {units.key('r.unit_source_value')}
)
SELECT stay_id, minutes_us(minute) AS t_us, concept_id, source_value, value, value_text,
       unit_source_value, unit_concept_id
FROM converted
WHERE minute IS NOT NULL AND (value IS NOT NULL OR value_text IS NOT NULL)
"""

# infusionDrug rate units accepted as a dose unit
INFUSION_DOSE_UNITS = ("ml/hr", "mcg/kg/min", "units/hr", "mg/hr", "mcg/min", "mcg/hr", "mcg/kg/hr",
                       "mg/min", "units/min", "ml", "units/kg/hr", "mg/kg/min", "mg/kg/hr",
                       "nanograms/kg/min")
_UNITS = ", ".join(f"'{u}'" for u in INFUSION_DOSE_UNITS)

# All rows of medication, infusionDrug and admissionDrug, drug names verbatim (the consumer
# selects antibiotics and vasopressors by name). sig carries what DRUG_EXPOSURE has no column for.

# infusionDrug: the rate is `drugrate`, as ricu's eicu_rate_kg reads it (`infusionrate`, the volume
# rate, is not used). medication: `dosage` is '<dose> <unit>'; a second number ('40 3') is no unit
# and gives none, and a dosage without a number ('ML') gives neither dose nor unit. The cohorts
# read medication only for antibiotic names.
DRUG_SQL = f"""
WITH
infusion AS (
    -- 'Norepinephrine (mcg/min)': the rate unit is the text in the first parentheses
    SELECT *, trim(replace(CASE WHEN contains(drugname, '(') THEN split_part(drugname, '(', 2) END,
                           ')', '')) AS rate_unit
    FROM infusiondrug
),
medication_fields AS (
    -- dosage is '<dose> <unit>' ('12 mg'); a second number ('125 12') is no unit
    SELECT *,
           CASE WHEN dosage IS NOT NULL THEN split_part(dosage, ' ', 1) END AS dose_text,
           CASE WHEN contains(dosage, ' ') THEN split_part(dosage, ' ', 2) END AS unit_text
    FROM medication
),
drug_rows AS (
    SELECT patientunitstayid AS stay_id, infusionoffset AS start_min, NULL AS stop_min, drugname,
           drugamount AS dose,
           CASE WHEN rate_unit IN ({_UNITS}) THEN rate_unit END AS dose_unit,
           NULL AS route,
           -- the rate, its unit and the charted weight, which ricu's eicu_rate_kg needs
           concat_ws(';', 'table=infusiondrug', 'rate=' || drugrate, 'rate_unit=' || rate_unit,
                     'weight_kg=' || CAST(patientweight AS VARCHAR)) AS sig,
           false AS admission_drug
    FROM infusion
    UNION ALL
    SELECT patientunitstayid, drugstartoffset, drugstopoffset, drugname,
           num(replace(dose_text, ',', '')),
           CASE WHEN NOT coalesce(is_int(unit_text), false) THEN unit_text END,
           routeadmin, 'table=medication', false
    FROM medication_fields
    UNION ALL
    -- home medications recorded at admission, not ICU administrations
    SELECT patientunitstayid, drugoffset, NULL, drugname, drugdosage,
           CASE WHEN drugunit = ' ' THEN NULL ELSE drugunit END,
           NULL, 'table=admissiondrug', true
    FROM admissiondrug
),
timed AS (
    -- a stop before the start (medication has them, often a stop of 0) is no usable end: the end
    -- is the start, and the source stop goes to sig as minutes from the start
    SELECT stay_id, start_min, drugname, dose, dose_unit, route, admission_drug,
           CASE WHEN stop_min < start_min THEN start_min ELSE coalesce(stop_min, start_min) END AS end_min,
           CASE WHEN stop_min < start_min
                THEN concat_ws(';', sig, 'source_stop_min=' || CAST(stop_min - start_min AS VARCHAR))
                ELSE sig END AS sig
    FROM drug_rows
    WHERE start_min IS NOT NULL AND stay_id IN (SELECT patientunitstayid FROM patient)
),
distinct_rows AS (
    -- exact duplicates collapse (the medication table repeats rows) # drops ca 100k from 13M rows
    SELECT DISTINCT stay_id, start_min, end_min, drugname, dose, dose_unit, route, sig, admission_drug
    FROM timed
)
SELECT stay_id, minutes_us(start_min) AS t_start_us, minutes_us(end_min) AS t_end_us,
       drugname AS source_value, CAST(dose AS DOUBLE) AS quantity, dose_unit, route, sig,
       admission_drug
FROM distinct_rows
"""

# Ventilation as ricu's eicu vent_start / vent_end sources, stored as recorded; the consumer pairs
# starts with ends. respiratoryCare offsets of 0 mean missing (ricu's vent_flag).
VENT = 4230167
PROCEDURE_SQL = f"""
WITH
episodes AS (
    SELECT patientunitstayid AS stay_id, ventstartoffset AS start_min,
           CASE WHEN ventendoffset <> 0 THEN ventendoffset END AS end_min,
           {VENT} AS concept_id, 'respiratorycare:vent' AS source_value
    FROM respiratorycare WHERE ventstartoffset <> 0
    UNION ALL
    SELECT patientunitstayid, priorventstartoffset,
           CASE WHEN priorventendoffset <> 0 THEN priorventendoffset END,
           {VENT}, 'respiratorycare:priorvent'
    FROM respiratorycare WHERE priorventstartoffset <> 0
),
markers AS (
    SELECT patientunitstayid AS stay_id, respchartoffset AS start_min, NULL AS end_min,
           {VENT} AS concept_id, 'respiratorycharting:' || respcharttypecat AS source_value
    FROM respiratorycharting
    WHERE respcharttypecat IN ('Start', 'Continued', 'respFlowPtVentData')
    UNION ALL
    -- 'off' markers end an episode; concept 0. A row can be a start and an end marker.
    SELECT patientunitstayid, respchartoffset, NULL, 0,
           'respiratorycharting:' || respchartvaluelabel || '=' || respchartvalue
    FROM respiratorycharting
    WHERE respchartvalue IN ('off', 'Off', 'Suspended')
)
SELECT stay_id, minutes_us(start_min) AS t_start_us, minutes_us(end_min) AS t_end_us,
       concept_id, source_value
FROM (SELECT * FROM episodes UNION ALL SELECT * FROM markers)
WHERE start_min IS NOT NULL AND stay_id IN (SELECT patientunitstayid FROM patient)
"""
