"""Step 3: every staged dataset -> the OMOP CDM v5.4 tables, one `<TABLE>.parquet` each.

Keys: a stay is `<dataset>-<stay_id>` (VISIT_OCCURRENCE.visit_source_value), a person
`<dataset>-<person_key>` (PERSON.person_source_value); person, visit and care-site ids are md5
numbers of these strings (timeline.md5_id), the other ids row numbers. Times (timeline.py): every
row at its stay's admission + staged offset; the admission is the source's own (staged `admit_us`),
or synthetic where a dataset stages none.
"""

from __future__ import annotations

import logging

from omop_etl import config, db, schema, staging, timeline, vocabulary

log = logging.getLogger("omop_etl")

COUNTRY = {"mimic4": ("US", 4330442), "eicu": ("US", 4330442), "hirid": ("CH", 4330427)}
CITY = {"Beth Israel Deaconess Medical Center": "Boston", "Bern University Hospital": "Bern"}

EHR = 32817  # type concept 'EHR' for every event row
VISIT_CONCEPT = 32037  # 'Intensive Care'
VISIT_TYPE = 44818518
ICU_DEATH = 4216643  # discharged_to 'Patient died', on the stay ricu attributes the ICU death to
DRUG_TYPE = 43542358
DRUG_TYPE_ADMISSION = 32865  # 'Patient self-report': eicu admissionDrug home medications
AGE_CONCEPT, AGE_UNIT, AGE_TYPE = 44804452, 9448, 38000280
HEIGHT, WEIGHT, CM, KG = 3036277, 3025315, 8582, 9529  # LOINC Body height, Body weight, as OHDSI's MIMIC-IV ETL
GENDER = {"M": 8507, "F": 8532}


def build(cfg: config.Config, con, datasets: list[str] | None = None) -> None:
    datasets = staged_datasets(cfg, datasets)
    log.info("cdm from %s", datasets)
    out = cfg.cdm_dir
    out.mkdir(parents=True, exist_ok=True)
    timeline.create_macros(con)
    with db.step("vocabulary"):
        vocabulary.load_concept(con, cfg.vocabulary_dir)
    register_staging(cfg, con, datasets)
    with db.step("stays and timeline"):
        build_stays(con)
        build_specimen_link(con)
    log_orphans(con)

    # The averaged height/weight MEASUREMENT rows and the age OBSERVATION rows are built here from
    # the stays table, not staged by a dataset module, so their unit names are looked up here.
    unit_names = dict(con.sql(f"SELECT concept_id, concept_name FROM athena_concept "
                              f"WHERE concept_id IN ({CM}, {KG}, {AGE_UNIT})").fetchall())
    tables = {
        "LOCATION": location_sql(),
        "CARE_SITE": care_site_sql(),
        "PERSON": person_sql(con),
        "VISIT_OCCURRENCE": visit_occurrence_sql(),
        "DEATH": death_sql(),
        "OBSERVATION": observation_sql(unit_names),
        "OBSERVATION_PERIOD": observation_period_sql(),
        "MEASUREMENT": measurement_sql(unit_names),
        "DRUG_EXPOSURE": drug_exposure_sql(con),
        "PROCEDURE_OCCURRENCE": procedure_sql(),
        "SPECIMEN": specimen_sql(),
        "FACT_RELATIONSHIP": fact_relationship_sql(),
        "CONDITION_OCCURRENCE": condition_sql(),
    }
    # LOCATION has a row per hospital, which CARE_SITE refers to, and a row per country, which
    # PERSON refers to: a person's residence is not known beyond the country of the dataset.
    year_type = {"year_of_birth": person_year_type(con)}
    for name, sql in tables.items():
        with db.step(name):
            write(con, out, name, sql, types=year_type if name == "PERSON" else None,
                  row_group_size=1_000_000 if name == "MEASUREMENT" else None)
            con.execute(f"CREATE OR REPLACE VIEW cdm_{name.lower()} AS SELECT * FROM read_parquet('{out / name}.parquet')")
    with db.step("CONCEPT, DOMAIN"):
        write(con, out, "CONCEPT", concept_sql(out, list(tables)))
        missing = con.sql(f"""SELECT list(concept_id ORDER BY concept_id) FROM ({used_concepts_sql(out, list(tables))})
                              WHERE concept_id NOT IN (SELECT concept_id FROM athena_concept)""").fetchone()[0]
        if missing:
            log.warning("concept ids used in the CDM but not in the vocabulary download: %s", missing)
        write(con, out, "DOMAIN", f"""
            SELECT * FROM ({vocabulary.domain_sql(cfg.vocabulary_dir)})
            WHERE domain_id IN (SELECT domain_id FROM read_parquet('{out}/CONCEPT.parquet'))""")
    for name in schema.tables():
        if name not in tables and name not in ("CONCEPT", "DOMAIN"):
            db.copy(con, schema.empty_sql(name), out / f"{name}.parquet")


def staged_datasets(cfg: config.Config, datasets: list[str] | None) -> list[str]:
    wanted = datasets or list(cfg.datasets)
    found = [ds for ds in wanted if staging.path(cfg.staging_dir(ds), "stays").exists()]
    missing = sorted(set(wanted) - set(found))
    if datasets and missing:
        raise FileNotFoundError(f"no staging for {missing} under {cfg.out_dir / 'staging'}")
    if missing:
        log.info("not staged, left out: %s", missing)
    if not found:
        raise FileNotFoundError(f"no staged dataset under {cfg.out_dir / 'staging'}")
    return found


def register_staging(cfg: config.Config, con, datasets: list[str]) -> None:
    """Views `stg_<table>`: the staging tables of all datasets with a `dataset` column, and a
    `row_id` (1..n over all datasets, from the parquet row numbers) that becomes the surrogate id
    of the big tables without a serial row_number() over a billion rows."""
    for table, columns in staging.TABLES.items():
        parts, offset = [], 0
        for ds in datasets:
            path = staging.path(cfg.staging_dir(ds), table)
            if not path.exists():
                continue
            parts.append(f"SELECT '{ds}' AS dataset, {offset} + file_row_number + 1 AS row_id, "
                         f"* EXCLUDE (file_row_number) FROM read_parquet('{path}', file_row_number = true)")
            offset += con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        if not parts:
            cols = ", ".join(f"CAST(NULL AS {t}) AS {c}" for c, t in columns.items())
            parts = [f"SELECT CAST(NULL AS VARCHAR) AS dataset, CAST(NULL AS BIGINT) AS row_id, {cols} LIMIT 0"]
        con.execute(f"CREATE OR REPLACE VIEW stg_{table} AS " + "\nUNION ALL ".join(parts))
        con.execute(f"CREATE OR REPLACE TABLE stg_{table}_rows AS SELECT {offset} AS n")


def build_stays(con) -> None:
    """Table `stay`: staged stays with keys, ids, and their admission on the CDM timeline."""
    con.execute("""
        CREATE OR REPLACE TABLE span AS
        SELECT dataset, stay_id, min(t) AS t_min_us, max(t) AS t_max_us FROM (
            SELECT dataset, stay_id, t_us AS t FROM stg_measurement
            UNION ALL SELECT dataset, stay_id, t_start_us FROM stg_drug
            UNION ALL SELECT dataset, stay_id, t_end_us FROM stg_drug
            UNION ALL SELECT dataset, stay_id, t_start_us FROM stg_procedure
            UNION ALL SELECT dataset, stay_id, t_end_us FROM stg_procedure
            UNION ALL SELECT dataset, stay_id, t_us FROM stg_specimen
            UNION ALL SELECT dataset, stay_id, death_t_us FROM stg_stays)
        GROUP BY dataset, stay_id""")
    con.execute(f"""
        CREATE OR REPLACE TABLE stay AS
        SELECT s.*,
               s.dataset || '-' || s.stay_id AS visit_source_value,
               s.dataset || '-' || s.person_key AS person_source_value,
               md5_id('2', s.dataset || '-' || s.stay_id) AS visit_occurrence_id,
               md5_id('1', s.dataset || '-' || s.person_key) AS person_id,
               md5_id('6', coalesce(s.hospital, 'nan') || coalesce(s.unit_type, 'nan')) AS care_site_id,
               CAST(b.base_us AS BIGINT) AS base_us,
               sp.t_min_us, sp.t_max_us
        FROM stg_stays s
        JOIN ({timeline.visit_base_sql('stg_stays', 'span')}) b USING (dataset, stay_id)
        LEFT JOIN span sp USING (dataset, stay_id)""")
    n, persons = con.sql("SELECT count(*), count(DISTINCT person_id) FROM stay").fetchone()
    dup = con.sql("SELECT count(*) - count(DISTINCT visit_occurrence_id) FROM stay").fetchone()[0]
    if dup:
        raise ValueError(f"{dup} visit_occurrence_id collisions")
    log.info("stays: %s stays of %s persons", f"{n:,}", f"{persons:,}")


def log_orphans(con) -> None:
    """Staged rows whose stay is not in stays.parquet are dropped; count them."""
    for table in ("measurement", "drug", "procedure", "specimen", "condition"):
        rows = con.sql(f"""SELECT dataset, count(*) FROM stg_{table} t
                           WHERE NOT EXISTS (SELECT 1 FROM stay s WHERE s.dataset = t.dataset AND s.stay_id = t.stay_id)
                           GROUP BY dataset""").fetchall()
        for ds, n in rows:
            log.warning("%s: %s rows of %s stays not in stays.parquet are dropped", table, f"{n:,}", ds)


def write(con, out, name: str, sql: str, types: dict | None = None, row_group_size: int | None = None) -> int:
    got = [r[0] for r in con.sql(f"DESCRIBE {sql}").fetchall()]
    unknown = set(got) - set(schema.tables()[name])
    if unknown:
        raise ValueError(f"{name}: columns not in CDM v5.4: {sorted(unknown)}")
    return db.copy(con, schema.select_sql(name, sql, set(got), types), out / f"{name}.parquet", row_group_size)


def ts(us_expr: str) -> str:
    """SQL: a time on the CDM timeline from microseconds since 2000-01-01."""
    return f"at_us({us_expr})"


# --- persons, visits, sites -------------------------------------------------------------------
def location_sql() -> str:
    """One row per hospital (CARE_SITE) and per country (PERSON)."""
    country = " UNION ALL ".join(f"SELECT '{ds}' AS dataset, '{c}' AS country, {cid} AS country_concept_id"
                                 for ds, (c, cid) in COUNTRY.items())
    city = " UNION ALL ".join(f"SELECT '{h}' AS hospital, '{c}' AS city" for h, c in CITY.items())
    return f"""
WITH country AS ({country}), city AS ({city}),
places AS (
    SELECT DISTINCT s.hospital AS location_source_value, city.city, c.country AS country_source_value,
           c.country_concept_id, 0 AS is_country
    FROM stay s JOIN country c USING (dataset) LEFT JOIN city USING (hospital)
    WHERE s.hospital IS NOT NULL
    UNION
    SELECT DISTINCT c.country, NULL, c.country, c.country_concept_id, 1
    FROM stay s JOIN country c USING (dataset)
)
SELECT row_number() OVER (ORDER BY is_country, location_source_value) AS location_id,
       city, location_source_value, country_concept_id, country_source_value
FROM places
"""


def care_site_sql() -> str:
    """One care site per (hospital, ICU unit type as the source names it)."""
    return """
SELECT DISTINCT s.care_site_id, s.hospital AS care_site_name, l.location_id,
       s.hospital AS care_site_source_value, s.unit_type AS place_of_service_source_value
FROM stay s LEFT JOIN cdm_location l ON l.location_source_value = s.hospital
"""


# the staged year of birth, else the year of the admission minus the age
YEAR_OF_BIRTH = "coalesce(year_of_birth, year(at_us(base_us)) - age)"


def person_year_type(con) -> str:
    """year_of_birth: INTEGER when every value is a whole number, else DOUBLE."""
    n = con.sql(f"SELECT count(*) FROM stay WHERE {YEAR_OF_BIRTH} != floor({YEAR_OF_BIRTH})").fetchone()[0]
    if n:
        log.warning("PERSON: %s stays with a fractional year of birth, year_of_birth stays DOUBLE", n)
        return "DOUBLE"
    return "INTEGER"


def person_sql(con) -> str:
    """One row per person; year of birth and sex of the first stay (stay_order) where age and sex
    are known, else of the first stay. The age of every stay is its OBSERVATION age row."""
    gender = " ".join(f"WHEN '{k}' THEN {v}" for k, v in GENDER.items())
    return f"""
WITH chosen AS (
    SELECT *, {YEAR_OF_BIRTH} AS yob FROM stay
    QUALIFY row_number() OVER (PARTITION BY person_id
                               ORDER BY (age IS NOT NULL AND sex IN ('M', 'F')) DESC, stay_order) = 1
),
country AS ({" UNION ALL ".join(f"SELECT '{ds}' AS dataset, '{c}' AS country" for ds, (c, _) in COUNTRY.items())})
SELECT c.person_id,
       CASE c.sex {gender} END AS gender_concept_id,
       c.yob AS year_of_birth,
       make_timestamp(CAST(trunc(c.yob) AS BIGINT), 1, 1, 0, 0, 0) AS birth_datetime,
       l.location_id,
       c.person_source_value,
       c.sex AS gender_source_value
FROM chosen c JOIN country USING (dataset)
LEFT JOIN cdm_location l ON l.location_source_value = country.country
"""


def visit_occurrence_sql() -> str:
    return f"""
SELECT visit_occurrence_id, person_id, {VISIT_CONCEPT} AS visit_concept_id,
       CAST({ts('base_us')} AS DATE) AS visit_start_date,
       {ts('base_us')} AS visit_start_datetime,
       CAST({ts('base_us + los_us')} AS DATE) AS visit_end_date,
       {ts('base_us + los_us')} AS visit_end_datetime,
       {VISIT_TYPE} AS visit_type_concept_id,
       care_site_id, visit_source_value,
       admission_source AS admitted_from_source_value,
       CASE WHEN icu_death THEN {ICU_DEATH} END AS discharged_to_concept_id,
       discharge_location AS discharged_to_source_value
FROM stay
"""


def death_sql() -> str:
    """One row per person: the earliest death on the timeline of the stay it is attributed to.
    A dead stay without a death time stands in with its visit end, only for a person without
    any timed death."""
    return f"""
WITH stays AS (
    SELECT *, bool_or(death_t_us IS NOT NULL) OVER (PARTITION BY person_id) AS person_timed FROM stay
),
death AS (
    SELECT person_id, stay_order, base_us,
           coalesce(death_t_us, CASE WHEN mortality AND NOT person_timed THEN los_us END) AS off_us
    FROM stays
)
SELECT person_id,
       CAST({ts('base_us + off_us')} AS DATE) AS death_date,
       {ts('base_us + off_us')} AS death_datetime,
       {EHR} AS death_type_concept_id
FROM death WHERE off_us IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY person_id ORDER BY base_us + off_us, stay_order) = 1
"""


def observation_sql(unit_names: dict) -> str:
    """Age at admission + 5 h, with the 5-year band '(a, b]' as 'a-b' in value_as_string."""
    at = f"base_us + {timeline.STATIC_OFFSET_H} * 3600000000"
    band = "CAST(ceil(age / 5) * 5 - 5 AS INTEGER) || '-' || CAST(ceil(age / 5) * 5 AS INTEGER)"
    return f"""
SELECT row_number() OVER (ORDER BY visit_occurrence_id) AS observation_id, person_id,
       {AGE_CONCEPT} AS observation_concept_id,
       CAST({ts(at)} AS DATE) AS observation_date, {ts(at)} AS observation_datetime,
       {AGE_TYPE} AS observation_type_concept_id,
       age AS value_as_number,
       CASE WHEN age > 0 AND age <= 95 THEN {band} END AS value_as_string,
       {AGE_UNIT} AS unit_concept_id, visit_occurrence_id,
       {sql_str(unit_names.get(AGE_UNIT))} AS unit_source_value,
       CAST(age AS VARCHAR) AS value_source_value
FROM stay
"""


def observation_period_sql() -> str:
    """Per person from the earliest to the latest date of any visit or staged row."""
    at = f"base_us + {timeline.STATIC_OFFSET_H} * 3600000000"
    return f"""
SELECT row_number() OVER (ORDER BY person_id) AS observation_period_id, person_id,
       min(CAST({ts('least(base_us, base_us + t_min_us)')} AS DATE)) AS observation_period_start_date,
       max(CAST({ts(f'greatest(base_us + coalesce(los_us, 0), base_us + t_max_us, {at})')} AS DATE))
           AS observation_period_end_date,
       {EHR} AS period_type_concept_id
FROM stay GROUP BY person_id
"""


def sql_str(s) -> str:
    return "NULL" if s is None else "'" + str(s).replace("'", "''") + "'"


# --- events -----------------------------------------------------------------------------------
def measurement_sql(unit_names: dict) -> str:
    """Staged rows plus one averaged height and weight row per stay (admission + 5 h).
    The averaged height/weight rows carry the unit concept's name as unit_source_value."""
    unit_name = f"CASE unit_concept_id WHEN {CM} THEN {sql_str(unit_names.get(CM))} WHEN {KG} THEN {sql_str(unit_names.get(KG))} END"
    at = f"s.base_us + {timeline.STATIC_OFFSET_H} * 3600000000"
    return f"""
WITH rows AS (
    SELECT m.row_id, s.person_id, s.visit_occurrence_id, s.base_us + m.t_us AS at_us, m.concept_id, m.value,
           m.unit_concept_id, m.source_value, m.unit_source_value, m.value_text
    FROM stg_measurement m JOIN stay s USING (dataset, stay_id)
    UNION ALL
    SELECT n + 2 * row_number() OVER (ORDER BY s.visit_occurrence_id) - 1, s.person_id, s.visit_occurrence_id, {at},
           {HEIGHT}, s.height, {CM}, 'raw_height', NULL, NULL
    FROM stay s, stg_measurement_rows WHERE s.height IS NOT NULL
    UNION ALL
    SELECT n + 2 * row_number() OVER (ORDER BY s.visit_occurrence_id), s.person_id, s.visit_occurrence_id, {at},
           {WEIGHT}, s.weight, {KG}, 'raw_weight', NULL, NULL
    FROM stay s, stg_measurement_rows WHERE s.weight IS NOT NULL
)
SELECT row_id AS measurement_id, person_id, concept_id AS measurement_concept_id,
       CAST({ts('at_us')} AS DATE) AS measurement_date, {ts('at_us')} AS measurement_datetime,
       strftime({ts('at_us')}, '%H:%M:%S') AS measurement_time,
       {EHR} AS measurement_type_concept_id, value AS value_as_number, unit_concept_id,
       visit_occurrence_id, source_value AS measurement_source_value,
       CASE WHEN unit_source_value IS NULL AND concept_id IN ({HEIGHT}, {WEIGHT}) THEN {unit_name}
            ELSE unit_source_value END AS unit_source_value,
       value_text AS value_source_value
FROM rows
"""


def drug_exposure_sql(con) -> str:
    """drug_concept_id: the first pattern of mappings/drug_ingredients.csv that matches the
    source name (case-insensitive), else 0."""
    con.execute("SET preserve_insertion_order = true")  # file order is the match priority
    con.execute(f"""CREATE OR REPLACE TABLE ingredient AS
                    SELECT row_number() OVER () AS priority, *
                    FROM read_csv('{config.MAPPINGS / 'drug_ingredients.csv'}', header = true,
                                  types = {{'concept_id': 'INTEGER', 'pattern': 'VARCHAR'}})""")
    con.execute("SET preserve_insertion_order = false")
    con.execute("""CREATE OR REPLACE TABLE drug_concept AS
                   SELECT d.source_value, arg_min(i.concept_id, i.priority) AS concept_id
                   FROM (SELECT DISTINCT source_value FROM stg_drug WHERE source_value IS NOT NULL) d
                   JOIN ingredient i ON regexp_matches(d.source_value, '(?i)' || i.pattern)
                   GROUP BY d.source_value""")
    return f"""
SELECT d.row_id AS drug_exposure_id, s.person_id,
       coalesce(c.concept_id, 0) AS drug_concept_id,
       CAST({ts('s.base_us + d.t_start_us')} AS DATE) AS drug_exposure_start_date,
       {ts('s.base_us + d.t_start_us')} AS drug_exposure_start_datetime,
       CAST({ts('s.base_us + d.t_end_us')} AS DATE) AS drug_exposure_end_date,
       {ts('s.base_us + d.t_end_us')} AS drug_exposure_end_datetime,
       CASE WHEN d.admission_drug THEN {DRUG_TYPE_ADMISSION} ELSE {DRUG_TYPE} END AS drug_type_concept_id,
       d.quantity, d.sig, s.visit_occurrence_id,
       d.source_value AS drug_source_value, d.route AS route_source_value,
       d.dose_unit AS dose_unit_source_value
FROM stg_drug d JOIN stay s USING (dataset, stay_id)
LEFT JOIN drug_concept c ON c.source_value = d.source_value
"""


def procedure_sql() -> str:
    return f"""
SELECT row_number() OVER () AS procedure_occurrence_id, s.person_id, p.concept_id AS procedure_concept_id,
       CAST({ts('s.base_us + p.t_start_us')} AS DATE) AS procedure_date,
       {ts('s.base_us + p.t_start_us')} AS procedure_datetime,
       CAST({ts('s.base_us + p.t_end_us')} AS DATE) AS procedure_end_date,
       {ts('s.base_us + p.t_end_us')} AS procedure_end_datetime,
       {EHR} AS procedure_type_concept_id, s.visit_occurrence_id,
       p.source_value AS procedure_source_value
FROM stg_procedure p JOIN stay s USING (dataset, stay_id)
"""


# FACT_RELATIONSHIP: a specimen and the visit of the stay it was staged for
SPECIMEN_DOMAIN, VISIT_DOMAIN = 36, 8
HAS_ASSOCIATED_VISIT, ASSOCIATED_VISIT_OF = 33136, 33137


def build_specimen_link(con) -> None:
    """Table `specimen_link`: the staged specimens with their specimen_id and visit. SPECIMEN has no
    visit id in CDM v5.4, so the visit is recorded in FACT_RELATIONSHIP."""
    con.execute("""
        CREATE OR REPLACE TABLE specimen_link AS
        SELECT row_number() OVER (ORDER BY p.row_id) AS specimen_id, s.person_id, s.visit_occurrence_id,
               s.base_us, p.*
        FROM stg_specimen p JOIN stay s USING (dataset, stay_id)""")


def specimen_sql() -> str:
    return f"""
SELECT specimen_id, person_id, concept_id AS specimen_concept_id,
       {EHR} AS specimen_type_concept_id,
       CAST({ts('base_us + t_us')} AS DATE) AS specimen_date,
       {ts('base_us + t_us')} AS specimen_datetime,
       source_id AS specimen_source_id, source_value AS specimen_source_value
FROM specimen_link
"""


def fact_relationship_sql() -> str:
    """Each specimen and its visit, in both directions."""
    return f"""
SELECT {SPECIMEN_DOMAIN} AS domain_concept_id_1, specimen_id AS fact_id_1,
       {VISIT_DOMAIN} AS domain_concept_id_2, visit_occurrence_id AS fact_id_2,
       {HAS_ASSOCIATED_VISIT} AS relationship_concept_id
FROM specimen_link
UNION ALL
SELECT {VISIT_DOMAIN}, visit_occurrence_id, {SPECIMEN_DOMAIN}, specimen_id, {ASSOCIATED_VISIT_OF}
FROM specimen_link
"""


def condition_sql() -> str:
    return f"""
SELECT row_number() OVER () AS condition_occurrence_id, s.person_id, c.concept_id AS condition_concept_id,
       CAST({ts('s.base_us + c.t_us')} AS DATE) AS condition_start_date,
       {ts('s.base_us + c.t_us')} AS condition_start_datetime,
       {EHR} AS condition_type_concept_id, s.visit_occurrence_id,
       c.source_value AS condition_source_value
FROM stg_condition c JOIN stay s USING (dataset, stay_id)
"""


# --- vocabulary -------------------------------------------------------------------------------
def used_concepts_sql(out, written: list[str]) -> str:
    parts = []
    for name in written:
        for col in schema.tables()[name]:
            if col.endswith("concept_id"):
                parts.append(f"SELECT DISTINCT {col} AS concept_id FROM read_parquet('{out}/{name}.parquet') "
                             f"WHERE {col} IS NOT NULL")
    return " UNION ".join(parts)


def concept_sql(out, written: list[str]) -> str:
    """Every concept id used in a written table, with its row from the user's Athena CONCEPT."""
    return f"""
WITH used AS ({used_concepts_sql(out, written)})
SELECT c.* FROM used JOIN athena_concept c USING (concept_id)
"""
