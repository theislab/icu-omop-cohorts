"""Mapping files: well-formed, and the drug patterns pick the intended ingredient."""
import pytest

from omop_etl import config, db


@pytest.fixture(scope="module")
def con():
    c = db.connect_plain()
    c.execute(f"""CREATE TABLE ingredient AS SELECT row_number() OVER () AS priority, *
                  FROM read_csv('{config.MAPPINGS / 'drug_ingredients.csv'}', header = true)""")
    return c


def first_ingredient(con, name):
    row = con.execute("""SELECT ingredient FROM ingredient WHERE regexp_matches(?, '(?i)' || pattern)
                         ORDER BY priority LIMIT 1""", [name]).fetchone()
    return row and row[0]


@pytest.mark.parametrize("name,ingredient", [
    ("Norepinephrine", "norepinephrine"), ("Epinephrine", "epinephrine"), ("Noradrenalin 10 mg", "norepinephrine"),
    ("Acetaminophen-IV", "acetaminophen"), ("Perfalgan 1g", "acetaminophen"), ("TYLENOL", "acetaminophen"),
    ("HYDROCODONE-ACETAMINOPHEN 5-325 MG PO TABS", "hydrocodone"), ("Piperacillin/Tazobactam (Zosyn)", "piperacillin"),
    ("Cisatracurium", "cisatracurium"), ("Insulin - Glargine", "insulin glargine"), ("Insulin - Regular", "insulin regular"),
    ("Heparin Sodium", "heparin"), ("NaCl 0.9%", "sodium chloride"), ("Sodium nitro", "nitroprusside"),
    ("Dopamine", "dopamine"), ("Dobutamine", "dobutamine"), ("Trace elements", None),
])
def test_drug_pattern_priority(con, name, ingredient):
    assert first_ingredient(con, name) == ingredient


def test_measurement_maps_are_unique(con):
    for ds in ("mimic4", "eicu", "hirid"):
        path = config.MAPPINGS / ds / "measurements.csv"
        if path.exists():
            dup = con.sql(f"""SELECT count(*) - count(DISTINCT (source_table, source_code))
                              FROM read_csv('{path}', header = true, all_varchar = true)""").fetchone()[0]
            assert dup == 0, ds
