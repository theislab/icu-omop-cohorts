"""The whole pipeline on the MIMIC-IV demo (open data), compared with the BlendedICU ETL's demo CDM.

Needs the demo (https://physionet.org/content/mimic-iv-demo/2.2/) at $MIMIC_IV_DEMO. The vocabulary is
the fixture in fixtures/vocabulary (tools/make_vocabulary_fixture.py), or an Athena download at
$OMOP_VOCABULARY. The comparison with the BlendedICU ETL's demo CDM, which this ETL replaced, needs that CDM at
$BLENDEDICU_DEMO_CDM and skips without it.
"""
import os
from pathlib import Path

import pytest

from omop_etl import cdm, config, db, mimic4, schema, staging

DEMO = Path(os.environ.get("MIMIC_IV_DEMO", "mimic-iv-demo-not-set"))
VOCAB = Path(os.environ.get("OMOP_VOCABULARY", Path(__file__).parent / "fixtures" / "vocabulary"))
OLD = Path(os.environ.get("BLENDEDICU_DEMO_CDM", "blendedicu-demo-cdm-not-set"))

pytestmark = pytest.mark.skipif(not DEMO.exists(), reason="MIMIC-IV demo missing")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    cfg = config.Config(out_dir=out, vocabulary_dir=VOCAB, sources={"mimic4": DEMO}, datasets=("mimic4",))
    con = db.connect(cfg, memory_limit="8GB", threads=4)
    mimic4.convert(cfg, con)
    mimic4.stage(cfg, con)
    staging.validate(cfg.staging_dir("mimic4"), con)
    cdm.build(cfg, con)
    return cfg, con


def test_staging_counts(built):
    cfg, con = built
    counts = staging.validate(cfg.staging_dir("mimic4"), con)
    assert counts["stays"] == 140


def test_concepts_in_vocabulary(built):
    """Every concept the CDM uses has its CONCEPT row, so the vocabulary fixture is not out of date. Except MALE and
    FEMALE: the Athena download the fixture comes from lacks the Gender vocabulary."""
    cfg, con = built
    tables = [p.stem for p in cfg.cdm_dir.glob("*.parquet") if p.stem in schema.tables() and p.stem not in ("CONCEPT", "DOMAIN")]
    missing = con.sql(f"""SELECT concept_id FROM ({cdm.used_concepts_sql(cfg.cdm_dir, tables)})
                          WHERE concept_id NOT IN (SELECT concept_id FROM read_parquet('{cfg.cdm_dir}/CONCEPT.parquet'))""")
    assert {r[0] for r in missing.fetchall()} <= {8507, 8532}


# The rows the consumer reads are the BlendedICU CDM's. Times are compared relative to the row's visit
# start, since the old CDM put stays on a synthetic timeline and this one keeps MIMIC-IV's dates
# (DEATH and SPECIMEN, which have no visit id, relative to the person's nearest visit start in the
# old CDM). Not compared: values and units of MEASUREMENT (stored as recorded here, converted
# there), DRUG_EXPOSURE quantity and sig (number formats), the time of the averaged height/weight,
# PERSON.year_of_birth (anchor_year - anchor_age here), sub-second visit ends (rounded there).
V = "read_parquet('{d}/VISIT_OCCURRENCE.parquet')"
COMPARE = {
    "VISIT_OCCURRENCE": "SELECT visit_source_value, visit_occurrence_id, person_id, "
                        "round(epoch(visit_end_datetime - visit_start_datetime)), "
                        "discharged_to_concept_id FROM read_parquet('{d}/VISIT_OCCURRENCE.parquet')",
    "PERSON": "SELECT person_id, CAST(gender_concept_id AS INTEGER) FROM read_parquet('{d}/PERSON.parquet')",
    "DEATH": "SELECT d.person_id, arg_min(d.death_datetime - v.visit_start_datetime, abs(epoch(d.death_datetime - v.visit_start_datetime))) "
             "FROM read_parquet('{d}/DEATH.parquet') d JOIN " + V + " v USING (person_id) GROUP BY d.person_id",
    "MEASUREMENT": "SELECT m.visit_occurrence_id, m.person_id, {concept}, "
                   "CASE WHEN m.measurement_source_value LIKE 'raw_%' THEN NULL ELSE m.measurement_datetime - v.visit_start_datetime END, "
                   "m.measurement_source_value, m.value_source_value "
                   "FROM read_parquet('{d}/MEASUREMENT.parquet') m JOIN " + V + " v USING (visit_occurrence_id) "
                   "WHERE m.measurement_source_value NOT IN ('52569', '53154')",
    "DRUG_EXPOSURE": "SELECT d.visit_occurrence_id, d.drug_source_value, d.drug_exposure_start_datetime - v.visit_start_datetime, "
                     "d.drug_exposure_end_datetime - v.visit_start_datetime, d.drug_type_concept_id, "
                     "d.dose_unit_source_value FROM read_parquet('{d}/DRUG_EXPOSURE.parquet') d JOIN " + V + " v USING (visit_occurrence_id)",
    "PROCEDURE_OCCURRENCE": "SELECT p.visit_occurrence_id, p.procedure_concept_id, p.procedure_datetime - v.visit_start_datetime, "
                            "p.procedure_end_datetime - v.visit_start_datetime "
                            "FROM read_parquet('{d}/PROCEDURE_OCCURRENCE.parquet') p JOIN " + V + " v USING (visit_occurrence_id)",
}
# Labs whose concept now matches the unit they are recorded in (mass for mg/dL), and lactate's moles
# concept, compared under the BlendedICU concept; labevents 52569 and 53154 are mapped here only.
RENAMED_CONCEPTS = {3047181: 40762125, 3000483: 3020491, 3004501: 3020491, 3001420: 3033836,
                    3016723: 40762887, 3006906: 3005162, 3024128: 3006140, 3011904: 3018913,
                    3013682: 43534100}
# Since 2026-11-26 the concepts of OHDSI's MIMIC-IV ETL (docs/OHDSI-MIMIC.md), compared under BlendedICU's; the
# observed and set tidal volumes share one concept there and are told apart by their item.
OHDSI_CONCEPTS = {1175625: 4313591, 3000905: 3010813, 3004249: 4353843, 3004501: 3000483, 3004809: 4100147, 3005785: 4017058, 3007220: 4265595, 3007469: 4108138, 3007930: 4204561, 3008342: 4148615, 3009744: 4290193, 3010421: 3019977, 3012030: 4182871, 3012501: 4284393, 3012888: 4354253, 3013290: 3027946, 3013502: 40483579, 3014315: 4145209, 3014576: 3018572, 3016293: 3008152, 3016407: 4094436, 3018677: 3013466, 3019550: 3000285, 3019800: 4005525, 3020460: 4208414, 3020716: 4353936, 3020891: 4174894, 3021119: 46272910, 3022217: 4306239, 3022875: 4353713, 3023103: 3005456, 3023599: 4016239, 3024171: 4313591, 3024929: 3007461, 3025315: 4099154, 3027018: 4239408, 3027315: 3027801, 3027597: 4216632, 3027598: 4108290, 3035995: 3001110, 3036277: 607590, 3037511: 4254663, 21492239: 4152194, 21492240: 4154790, 21492241: 4239021, 36303946: 4139635, 40762499: 4196147}
OHDSI_BY_ITEM = {'224684': 4220163, '224685': 44782827}
RENAMED_CONCEPTS = {**{n: RENAMED_CONCEPTS.get(o, o) for n, o in OHDSI_CONCEPTS.items()}, **RENAMED_CONCEPTS}
NEW_CONCEPT = ("CASE " + " ".join(f"WHEN m.measurement_source_value = '{i}' THEN {o}" for i, o in OHDSI_BY_ITEM.items())
               + " ELSE CASE m.measurement_concept_id "
               + " ".join(f"WHEN {n} THEN {o}" for n, o in RENAMED_CONCEPTS.items())
               + " ELSE m.measurement_concept_id END END")
# SPECIMEN rows go to their stay through FACT_RELATIONSHIP here, to the nearest visit in the old CDM
SPECIMEN_NEW = ("SELECT s.person_id, s.specimen_datetime - v.visit_start_datetime, s.specimen_source_value "
                "FROM read_parquet('{d}/SPECIMEN.parquet') s JOIN read_parquet('{d}/FACT_RELATIONSHIP.parquet') f "
                "ON f.domain_concept_id_1 = 36 AND f.fact_id_1 = s.specimen_id AND f.relationship_concept_id = 33136 "
                "JOIN " + V + " v ON v.visit_occurrence_id = f.fact_id_2")
SPECIMEN_OLD = ("SELECT any_value(s.person_id), arg_min(s.specimen_datetime - v.visit_start_datetime, "
                "abs(epoch(s.specimen_datetime - v.visit_start_datetime))), any_value(s.specimen_source_value) "
                "FROM read_parquet('{d}/SPECIMEN.parquet') s JOIN " + V + " v USING (person_id) GROUP BY s.specimen_id")
# The demo's distinct cultures (stay, time, specimen type) whose chartdate puts them on another stay of their admission
# than their charttime: the first stay with outtime >= the day (midnight) or the time, else the last.
MOVED_BY_CHARTDATE = f"""
    WITH st AS (SELECT stay_id, hadm_id, outtime FROM read_csv('{DEMO}/icu/icustays.csv.gz')),
    mb AS (SELECT DISTINCT hadm_id, CAST(chartdate AS TIMESTAMP) AS day,
                  coalesce(charttime, CAST(chartdate AS TIMESTAMP) + INTERVAL 12 HOUR) AS t, spec_type_desc
           FROM read_csv('{DEMO}/hosp/microbiologyevents.csv.gz') WHERE hadm_id IN (SELECT hadm_id FROM st)),
    both_rules AS (
        SELECT coalesce(arg_min(st.stay_id, st.outtime) FILTER (st.outtime >= mb.day), arg_max(st.stay_id, st.outtime)) AS by_day,
               coalesce(arg_min(st.stay_id, st.outtime) FILTER (st.outtime >= mb.t), arg_max(st.stay_id, st.outtime)) AS by_time,
               mb.t, mb.spec_type_desc
        FROM mb JOIN st USING (hadm_id) GROUP BY mb.hadm_id, mb.day, mb.t, mb.spec_type_desc)
    SELECT count(*) FROM (SELECT DISTINCT by_day, t, spec_type_desc FROM both_rules WHERE by_day <> by_time)"""


@pytest.mark.skipif(not OLD.exists(), reason="BlendedICU ETL demo CDM missing")
@pytest.mark.parametrize("table", [*COMPARE, "SPECIMEN"])
def test_same_rows_as_blendedicu_etl(built, table):
    cfg, con = built
    if table == "SPECIMEN":
        old, new = SPECIMEN_OLD.format(d=OLD), SPECIMEN_NEW.format(d=cfg.cdm_dir)
    else:
        old = COMPARE[table].format(d=OLD, concept="m.measurement_concept_id")
        new = COMPARE[table].format(d=cfg.cdm_dir, concept=NEW_CONCEPT)
    only_old = con.sql(f"SELECT count(*) FROM ({old} EXCEPT ALL {new})").fetchone()[0]
    only_new = con.sql(f"SELECT count(*) FROM ({new} EXCEPT ALL {old})").fetchone()[0]
    # the old CDM put a culture on the stay nearest to it, as the charttime does; the cultures that their chartdate
    # puts on another stay (test_cultures_by_chartdate) differ
    moved = con.sql(MOVED_BY_CHARTDATE).fetchone()[0] if table == "SPECIMEN" else 0
    assert (only_old, only_new) == (moved, moved)


def test_cultures_by_chartdate(built):
    """A culture belongs to the first stay of its admission whose outtime is at or after the culture's chartdate
    (midnight), else to the admission's last stay: ricu's rolling join on microbiologyevents' index_var."""
    cfg, con = built
    # `expected` and `linked` are read by name in the SQL below (duckdb's replacement scan)
    expected = con.sql(f"""
        WITH st AS (SELECT stay_id, hadm_id, outtime FROM read_csv('{DEMO}/icu/icustays.csv.gz')),
        mb AS (SELECT DISTINCT hadm_id, CAST(chartdate AS TIMESTAMP) AS day,
                      coalesce(charttime, CAST(chartdate AS TIMESTAMP) + INTERVAL 12 HOUR) AS t, spec_type_desc
               FROM read_csv('{DEMO}/hosp/microbiologyevents.csv.gz') WHERE hadm_id IN (SELECT hadm_id FROM st))
        SELECT DISTINCT coalesce(arg_min(st.stay_id, st.outtime) FILTER (st.outtime >= mb.day),
                                 arg_max(st.stay_id, st.outtime)) AS stay_id, mb.t, mb.spec_type_desc
        FROM mb JOIN st USING (hadm_id) GROUP BY mb.hadm_id, mb.day, mb.t, mb.spec_type_desc""")  # noqa: F841
    linked = con.sql(f"""
        SELECT DISTINCT CAST(split_part(v.visit_source_value, '-', 2) AS BIGINT) AS stay_id, s.specimen_datetime AS t,
               s.specimen_source_value AS spec_type_desc
        FROM read_parquet('{cfg.cdm_dir}/SPECIMEN.parquet') s
        JOIN read_parquet('{cfg.cdm_dir}/FACT_RELATIONSHIP.parquet') f
          ON f.domain_concept_id_1 = 36 AND f.fact_id_1 = s.specimen_id AND f.relationship_concept_id = 33136
        JOIN read_parquet('{cfg.cdm_dir}/VISIT_OCCURRENCE.parquet') v ON v.visit_occurrence_id = f.fact_id_2""")  # noqa: F841
    assert con.sql("SELECT count(*) FROM linked").fetchone()[0] > 0
    assert con.sql("SELECT count(*) FROM (SELECT * FROM expected EXCEPT ALL SELECT * FROM linked)").fetchone()[0] == 0
    assert con.sql("SELECT count(*) FROM (SELECT * FROM linked EXCEPT ALL SELECT * FROM expected)").fetchone()[0] == 0


def test_mimic_dates_kept(built):
    """Visits start at MIMIC-IV's own (shifted) intime; year_of_birth = anchor_year - anchor_age."""
    cfg, con = built
    years = con.sql(f"SELECT min(year(visit_start_datetime)), max(year(visit_start_datetime)) "
                    f"FROM read_parquet('{cfg.cdm_dir}/VISIT_OCCURRENCE.parquet')").fetchone()
    assert years[0] > 2100
    mismatch = con.sql(f"""
        SELECT count(*) FROM read_parquet('{cfg.cdm_dir}/VISIT_OCCURRENCE.parquet') v
        JOIN read_parquet('{cfg.raw_dir("mimic4")}/icustays.parquet') i
          ON v.visit_source_value = 'mimic4-' || i.stay_id
        WHERE v.visit_start_datetime != i.intime""").fetchone()[0]
    assert mismatch == 0


def test_cdm_types(built):
    cfg, con = built
    types = dict(con.sql(f"DESCRIBE SELECT * FROM read_parquet('{cfg.cdm_dir}/MEASUREMENT.parquet')").fetchall()[i][:2]
                 for i in range(23))
    assert types["value_as_number"] == "DOUBLE"
    assert types["measurement_concept_id"] == "INTEGER"
    assert types["visit_occurrence_id"] == "BIGINT"
    person = dict(r[:2] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{cfg.cdm_dir}/PERSON.parquet')").fetchall())
    assert person["year_of_birth"] == "INTEGER"
