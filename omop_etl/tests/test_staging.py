import pytest

from omop_etl import db, staging


def _stays_sql():
    return """
        SELECT 1 AS stay_id, 'p1' AS person_key, 1 AS stay_order, NULL AS admit_us, NULL AS year_of_birth,
               INTERVAL 36 HOUR AS los,
               70 AS age, 'M' AS sex, 'h' AS hospital, 'MICU' AS unit_type,
               NULL AS admission_source, NULL AS discharge_location, false AS mortality,
               false AS icu_death, NULL AS death_t_us, 180.0 AS height, NULL AS weight"""


def test_write_casts_and_converts_intervals(tmp_path):
    con = db.connect_plain()
    staging.write(con, tmp_path, "stays", _stays_sql())
    staging.write(con, tmp_path, "measurement", """
        SELECT 1 AS stay_id, INTERVAL '-90 minutes 0.000123 seconds' AS t, 3027018 AS concept_id,
               '220045' AS source_value, 80.5 AS value, NULL AS value_text,
               'bpm' AS unit_source_value, 8541 AS unit_concept_id""")
    staging.write(con, tmp_path, "drug", """
        SELECT 1 AS stay_id, 0 AS t_start_us, 3600000000 AS t_end_us, 'x' AS source_value,
               1 AS quantity, 'mg' AS dose_unit, NULL AS route, NULL AS sig, false AS admission_drug""")
    counts = staging.validate(tmp_path, con)
    assert counts == {"stays": 1, "measurement": 1, "drug": 1}
    los, = con.sql(f"SELECT los_us FROM '{tmp_path}/stays.parquet'").fetchone()
    assert los == 36 * 3600 * 10**6
    t, = con.sql(f"SELECT t_us FROM '{tmp_path}/measurement.parquet'").fetchone()
    assert t == -90 * 60 * 10**6 + 123


def test_validate_reports_problems(tmp_path):
    con = db.connect_plain()
    db.copy(con, "SELECT 1::BIGINT AS stay_id", staging.path(tmp_path, "stays"))
    with pytest.raises(ValueError, match="measurement.parquet missing"):
        staging.validate(tmp_path, con)
