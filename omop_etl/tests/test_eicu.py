"""eicu convert and stage on hand-made toy tables (no real data)."""

import gzip
import struct

import pytest

from omop_etl import config, db, eicu, staging


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_mul(a: float, b: float) -> float:
    # the double product of two float32 is exact, so rounding it once is IEEE float32 multiply
    return f32(f32(a) * f32(b))


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def write_raw(con, raw, table, rows):
    """rows: list of tuples in SOURCE_TABLES column order."""
    cols = eicu.SOURCE_TABLES[table][1]
    raw.mkdir(parents=True, exist_ok=True)
    if rows:
        values = ", ".join("(" + ", ".join(lit(v) for v in r) + ")" for r in rows)
        sel = ", ".join(f"CAST(c{i} AS {t}) AS {c}" for i, (c, t) in enumerate(cols.items()))
        names = ", ".join(f"c{i}" for i in range(len(cols)))
        sql = f"SELECT {sel} FROM (VALUES {values}) v({names})"
    else:
        sql = "SELECT " + ", ".join(f"CAST(NULL AS {t}) AS {c}" for c, t in cols.items()) + " LIMIT 0"
    db.copy(con, sql, raw / f"{table}.parquet")


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return repr(v)


TOY = {
    "patient": [
        # stay, uniquepid, hospital, unittype, admitsource, dischargelocation, status, discharge, gender, age, height,
        # weight, hospital stay, discharge year, hospital admission clock time, unit -> hospital admission offset,
        # unit -> hospital discharge offset
        (10, "002-1", 73, "MICU", "Emergency Department", "Death", "Expired", 1500, "Male", "> 89", 170.2, 80.5,
         7, 2015, "08:30:00", -300, 1500),
        (11, "002-1", 73, "MICU", None, "Home", "Alive", 600, "Female", "45", None, 60.0, 6, 2015, "23:00:00", 0, 2000),
        (12, "002-2", 75, "CCU", None, None, None, 100, "Unknown", None, None, None, 8, None, None, 0, 100),
    ],
    "lab": [
        (10, -30, "calcium", 8.0, "mg/dL"),
        (10, 10, "calcium", 2.1, "mmol/L"),
        (10, 20, "calcium", 9.3, None),
        (10, 30, "magnesium", 2.0, "mEq/L"),
        (10, 40, "magnesium", 0.9, "mmol/L"),
        (10, 50, "magnesium", 2.2, "mg/dL"),
        (10, 60, "CRP", 1.3, "mg/dL"),
        (10, 70, "CRP", 13.0, "mg/L"),
        (10, 80, "FiO2", 50.0, "mm(hg)"),
        (10, 90, "FiO2", 40.0, "%"),
        (10, 100, "glucose", 98.6, "mg/dL"),
        (10, 110, "creatinine", 1.1, "mg/dL"),
        (10, 120, "creatinine", None, "mg/dL"),
        (10, 130, "HCO3", 24.0, "mmol/L"),  # not mapped
        (10, 140, "ionized calcium", 1.1, "mmol/L"),
        (10, 150, "ionized calcium", 4.4, "MG/DL"),
        (10, 160, "ionized calcium", 2.3, "mEq/L"),
        (10, 170, "CRP", 5.0, None),
    ],
    "respiratorycharting": [
        (10, 5, "Vent Rate", "FiO2", "35%"),
        (10, 6, "respFlowPtVentData", "Mechanical Ventilator Mode", "AC/VC"),
        (10, 7, "Start", "PEEP", "5"),
        (10, 8, "x", "RT Vent On/Off", "Off"),
        (10, 9, "x", "FiO2", None),
    ],
    "nursecharting": [
        (10, 1, "Heart Rate", "80"),
        (10, 2, "Heart Rate", " 81"),
        (10, 3, "Temperature (C)", "100.4"),
        (10, 4, "Temperature (C)", "36.9"),
        (10, 7, "Temperature (C)", "60"),
        (10, 8, "Temperature (F)", "98.6"),
        (10, 5, "Sedation Score", "-3"),
        (10, 6, "Eyes", "4"),  # not mapped
    ],
    "intakeoutput": [
        (10, 60, "Urine", 200.0),
        (10, 61, "Output (mL)-Urethral Catheter Latex 14 Fr.", 150.0),
        (10, 62, "Ventriculostomy catheter output", 20.0),
        (10, 63, "URINE CATHETER", 100.0),
        (10, 64, "albumin", 250.0),
        (10, 65, "Urine", None),
    ],
    "vitalperiodic": [
        (10, 5, 99.5, 97.0, 80.0, 16.0, 8.0, None, None, None, 0.1, None, None),
    ],
    "vitalaperiodic": [
        (11, 7, 120.0, None, 85.0),
    ],
    "infusiondrug": [
        (10, 30, "Norepinephrine (mcg/min)", "5", 4.0, 70.0),
        (10, 30, "Norepinephrine (mcg/min)", "5", 4.0, 70.0),  # exact duplicate
        (10, 40, "Amiodarone (load) 150 mg (mg/min)", "OFF", None, None),
        (10, 50, "Heparin", None, None, 82.5),
    ],
    "medication": [
        (10, 100, 0, "VANCOMYCIN 1 G", "1 g", "IV"),
        (10, 100, 200, "drug X", "125 12", "PO"),
        (10, 100, 200, "drug Y", "5  mg", "PO"),
        (10, 100, 200, "drug Z", "1,000 mg", "PO"),
        (10, None, 200, "drug W", "1 mg", "PO"),
        (99, 100, 200, "orphan stay", "1 mg", "PO"),
        (11, 100, 200, None, None, None),
    ],
    "admissiondrug": [
        (11, -60, "ASPIRIN   ", 81.0, " "),
        (11, -60, "METOPROLOL", 25.0, "mg "),
    ],
    "respiratorycare": [
        (10, 30, 0, 0, 0),
        (10, 0, 300, 10, 20),
    ],
}


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = config.Config(out_dir=out, vocabulary_dir=out, sources={"eicu": out / "src"})
    con = db.connect_plain()
    for t in eicu.SOURCE_TABLES:
        write_raw(con, cfg.raw_dir("eicu"), t, TOY.get(t, []))
    eicu.stage(cfg, con)
    counts = staging.validate(cfg.staging_dir("eicu"), con)
    return con, cfg.staging_dir("eicu"), counts


def rows(con, d, table, cols, where="TRUE"):
    return con.sql(f"SELECT {cols} FROM '{d}/{table}.parquet' WHERE {where} ORDER BY ALL").fetchall()


def test_stays(staged):
    con, d, counts = staged
    assert counts["stays"] == 3
    r = rows(con, d, "stays", "stay_id, person_key, stay_order, los_us, age, sex, hospital, mortality, "
                              "icu_death, death_t_us, height, weight")
    assert r[0] == (10, "002-1", 1, 1500 * 60_000_000, 90.0, "M", "73", True, True, 1500 * 60_000_000, 170.2, 80.5)
    assert r[1][4:10] == (45.0, "F", "73", False, False, None)
    assert r[2][4:10] == (None, None, "75", None, False, None)


def test_stays_placed_in_their_discharge_year(staged):
    """A person's hospital stays in order of year and age: the first on 1 January of its discharge
    year at its admission clock time, the next on the calendar day after the previous discharge;
    unit stays at their offset from the hospital admission; no year, no admission time."""
    con, d, _ = staged
    t = {s: (a, y) for s, a, y in rows(con, d, "stays", "stay_id, admit_us, year_of_birth")}
    minute = 60 * 10**6
    at = lambda ts: con.sql(f"SELECT epoch_us(TIMESTAMP '{ts}')").fetchone()[0]
    # hospital stay 6 (age 45, unit stay 11 at offset 0): 2015-01-01 23:00, discharged 2000 min later
    assert t[11][0] == at("2015-01-01 23:00:00")
    # hospital stay 7 (age 90): the day after 2015-01-03 08:20 at 08:30; unit 10 300 min after it
    assert t[10][0] == at("2015-01-04 08:30:00") + 300 * minute
    assert (t[10][1], t[11][1]) == (2015 - 90, 2015 - 45)
    assert t[12] == (None, None)


def value_of(con, d, source, t_min):
    r = rows(con, d, "measurement", "value, value_text", f"source_value = '{source}' AND t_us = {t_min * 60_000_000}")
    assert len(r) == 1, (source, t_min, r)
    return r[0]


def value_unit(con, d, source, t_min):
    r = rows(con, d, "measurement", "value, unit_concept_id", f"source_value = '{source}' AND t_us = {t_min * 60_000_000}")
    assert len(r) == 1, (source, t_min, r)
    return r[0]


MG_DL, MMOL_L, MEQ_L, MG_L, CELSIUS, FAHRENHEIT = 8840, 8753, 9557, 8751, 586323, 9289


def test_lab_values_as_recorded_with_their_unit(staged):
    con, d, _ = staged
    assert value_unit(con, d, "lab:calcium", -30) == (8.0, MG_DL)
    assert value_unit(con, d, "lab:calcium", 10) == (2.1, MMOL_L)
    assert value_unit(con, d, "lab:calcium", 20) == (9.3, MG_DL)  # no unit: mg/dL assumed
    assert value_unit(con, d, "lab:magnesium", 30) == (2.0, MEQ_L)
    assert value_unit(con, d, "lab:magnesium", 40) == (0.9, MMOL_L)
    assert value_unit(con, d, "lab:magnesium", 50) == (2.2, MG_DL)
    assert value_unit(con, d, "lab:CRP", 60) == (1.3, MG_DL)
    assert value_unit(con, d, "lab:CRP", 70) == (13.0, MG_L)
    assert value_of(con, d, "lab:FiO2", 90)[0] == 40.0
    assert not rows(con, d, "measurement", "*", "source_value = 'lab:FiO2' AND t_us = 4800000000")
    assert value_unit(con, d, "lab:glucose", 100) == (98.6, MG_DL)
    assert value_unit(con, d, "lab:creatinine", 110) == (1.1, MG_DL)
    assert not rows(con, d, "measurement", "*", "source_value IN ('lab:HCO3', 'nursecharting:Eyes')")
    assert rows(con, d, "measurement", "count(*)", "source_value = 'lab:creatinine'") == [(1,)]


def test_temperature_unit_by_value(staged):
    con, d, _ = staged
    assert value_unit(con, d, "nursecharting:Temperature (C)", 3) == (100.4, FAHRENHEIT)
    assert value_unit(con, d, "nursecharting:Temperature (C)", 4) == (36.9, CELSIUS)
    assert value_unit(con, d, "vitalperiodic:temperature", 5) == (99.5, FAHRENHEIT)
    assert value_unit(con, d, "nursecharting:Temperature (C)", 7) == (60.0, FAHRENHEIT)
    assert value_unit(con, d, "nursecharting:Temperature (F)", 8) == (98.6, FAHRENHEIT)


def test_unit_from_the_unit_string(staged):
    con, d, _ = staged
    assert value_unit(con, d, "lab:ionized calcium", 140) == (1.1, MMOL_L)
    assert value_unit(con, d, "lab:ionized calcium", 150) == (4.4, MG_DL)
    assert value_unit(con, d, "lab:ionized calcium", 160) == (2.3, MEQ_L)
    assert value_unit(con, d, "lab:CRP", 170) == (5.0, MG_L)  # no unit: the item's mg/L


def test_text_values(staged):
    con, d, _ = staged
    assert value_of(con, d, "nursecharting:Heart Rate", 1) == (80.0, None)
    # a number after trimming is a number
    assert value_of(con, d, "nursecharting:Heart Rate", 2) == (81.0, None)
    assert value_of(con, d, "respiratorycharting:FiO2", 5) == (35.0, None)
    assert value_of(con, d, "respiratorycharting:Mechanical Ventilator Mode", 6) == (None, "AC/VC")
    assert value_of(con, d, "nursecharting:Sedation Score", 5) == (-3.0, None)
    assert rows(con, d, "measurement", "count(*)", "source_value = 'respiratorycharting:FiO2'") == [(1,)]


def test_intakeoutput_urine(staged):
    con, d, _ = staged
    r = rows(con, d, "measurement", "source_value, concept_id, value", "source_value LIKE 'intakeoutput:%'")
    assert r == [("intakeoutput:Output (mL)-Urethral Catheter Latex 14 Fr.", 3014315, 150.0),
                 ("intakeoutput:URINE CATHETER", 3014315, 100.0),
                 ("intakeoutput:Urine", 3014315, 200.0),
                 ("intakeoutput:albumin", 3024561, 250.0)]


def test_wide_vitals(staged):
    con, d, _ = staged
    r = rows(con, d, "measurement", "source_value, concept_id, unit_concept_id, value",
             "source_value LIKE 'vital%' AND source_value <> 'vitalperiodic:temperature'")
    assert r == [("vitalaperiodic:noninvasivemean", 21492241, 8876, 85.0),
                 ("vitalaperiodic:noninvasivesystolic", 21492239, 8876, 120.0),
                 ("vitalperiodic:cvp", 0, 0, 8.0),
                 ("vitalperiodic:heartrate", 3027018, 8541, 80.0),
                 ("vitalperiodic:respiration", 3024171, 8541, 16.0),
                 ("vitalperiodic:sao2", 40762499, 8554, 97.0),
                 ("vitalperiodic:st1", 0, 0, 0.1)]


def test_drugs(staged):
    con, d, counts = staged
    r = {(x[0], x[1]): x for x in rows(con, d, "drug", "source_value, t_start_us // 60000000, t_end_us // 60000000, "
                                                      "quantity, dose_unit, route, sig, admission_drug")}
    assert ("orphan stay", 100) not in r and ("drug W", None) not in r
    assert counts["drug"] == 10
    assert r[("Norepinephrine (mcg/min)", 30)][2:] == (
        30, 4.0, "mcg/min", None, "table=infusiondrug;rate=5;rate_unit=mcg/min;weight_kg=70.0", False)
    assert r[("Amiodarone (load) 150 mg (mg/min)", 40)][4:7] == (
        None, None, "table=infusiondrug;rate=OFF;rate_unit=load 150 mg")
    assert r[("Heparin", 50)][6] == "table=infusiondrug;weight_kg=82.5"
    assert r[("VANCOMYCIN 1 G", 100)][2:7] == (100, 1.0, "g", "IV", "table=medication;source_stop_min=-100")
    assert r[("drug X", 100)][2:5] == (200, 125.0, None)
    assert r[("drug Y", 100)][3:5] == (5.0, "")
    assert r[("drug Z", 100)][3:5] == (1000.0, "mg")
    assert r[("ASPIRIN   ", -60)][3:] == (81.0, None, None, "table=admissiondrug", True)
    assert r[("METOPROLOL", -60)][4] == "mg "
    assert rows(con, d, "drug", "count(*)", "source_value IS NULL") == [(1,)]


def test_ventilation(staged):
    con, d, _ = staged
    r = rows(con, d, "procedure", "source_value, concept_id, t_start_us // 60000000, t_end_us // 60000000")
    assert r == [("respiratorycare:priorvent", 4230167, 10, 20),
                 ("respiratorycare:vent", 4230167, 30, None),
                 ("respiratorycharting:RT Vent On/Off=Off", 0, 8, None),
                 ("respiratorycharting:Start", 4230167, 7, None),
                 ("respiratorycharting:respFlowPtVentData", 4230167, 6, None)]


def test_convert_reads_like_pandas(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    lines = ["patientunitstayid,uniquepid,hospitalid,unittype,unitadmitsource,unitdischargelocation,"
             "unitdischargestatus,unitdischargeoffset,gender,age,admissionheight,admissionweight,extra,"
             "patienthealthsystemstayid,hospitaldischargeyear,hospitaladmittime24,hospitaladmitoffset,"
             "hospitaldischargeoffset",
             '1,"002-1",73,MICU,NA,"Home ",Alive,100,Male,> 89,170.5,,"a,b",5,2014,07:15:00,-60,400',
             '2,002-2,73,MICU,None,n/a,,100,,"",NA,80,x,6,2015,,0,100']
    with gzip.open(src / "patient.csv.gz", "wt") as f:
        f.write("\n".join(lines) + "\n")
    cfg = config.Config(out_dir=tmp_path / "out", vocabulary_dir=tmp_path, sources={"eicu": src})
    con = db.connect_plain()
    f, cols = eicu.SOURCE_TABLES["patient"]
    from omop_etl import convert
    convert.csv_to_parquet(con, src / f, cfg.raw_dir("eicu") / "patient.parquet", cols)
    r = con.sql(f"SELECT * FROM '{cfg.raw_dir('eicu')}/patient.parquet' ORDER BY 1").fetchall()
    assert r[0] == (1, "002-1", 73, "MICU", None, "Home ", "Alive", 100, "Male", "> 89", 170.5, None,
                    5, 2014, "07:15:00", -60, 400)
    assert r[1] == (2, "002-2", 73, "MICU", None, None, None, 100, None, None, None, 80.0, 6, 2015, None, 0, 100)
