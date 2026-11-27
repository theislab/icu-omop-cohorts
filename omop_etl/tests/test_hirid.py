"""hirid convert + stage on a hand-made toy HiRID tree (no real data)."""

import struct

import pytest

from omop_etl import config, db, hirid, staging

ADM = "2150-01-01 08:00:00"
H = 3600 * 10**6  # one hour in µs


def f32(x: float) -> float:
    """Round a Python float to float32. Products and quotients of two float32 values computed in
    double and rounded once are exactly the float32 results."""
    return struct.unpack("f", struct.pack("f", x))[0]


def at(hours: float) -> str:
    return f"TIMESTAMP '{ADM}' + to_microseconds({round(hours * H)})"


def _toy_source(root, con):
    ref = root / "reference_data"
    ref.mkdir(parents=True)
    (ref / "general_table.csv").write_text(
        "patientid,admissiontime,sex,age,discharge_status\n"
        f"1,{ADM},M,65,dead\n"
        f"2,{ADM},F,40,alive\n"
        f"3,{ADM},M,90,\n"           # unknown outcome
        f"4,{ADM},F,20,dead\n")      # no observation at all
    (ref / "hirid_variable_reference.csv").write_text(
        "Source Table,ID,Variable Name,Unit,Additional information\n"
        "Observation,200,Heart rate,b/min,\n"
        "Pharma,1000462,Noradrenalin  20 mg/50 ml,,\n"   # double space kept verbatim
        "Pharma,1000999,Gramdrug,,\n"
        "Pharma,1000777,NA,,\n")                          # pandas reads 'NA' as missing
    rows = [
        # patientid, variableid, hours after admission, value
        (1, 200, -1.0, 80),          # heart rate before admission
        (1, 200, 10.0, 90),
        (1, 110, 12.5, None),        # last MAP row, null value: still the death time
        (1, 999, 20.25, 1),          # unmapped variable: only the stay end
        (1, 24000548, 1.0, 123),     # hemoglobin g/L -> g/dL
        (1, 24000806, 1.0, 7),       # troponin T in ng/L
        (1, 20000700, 2.0, 8),       # wbc
        (1, 20000700, 2.0, 10),      # second wbc at the same time: median 9
        (1, 24000550, 2.0, 6),       # neutrophils, absolute count, as recorded
        (1, 24000550, 3.0, 6),       # also without a wbc at that time
        (1, 10000300, 4.0, 4), (1, 10000200, 4.0, 6), (1, 10000100, 4.0, 5),
        (1, 10000100, 4.0, 4),       # repeated verbal: median 4.5
        (1, 10000300, 5.0, 4), (1, 10000200, 5.0, 6),   # verbal missing: no total
        (1, 10000450, -2.0, 180), (1, 10000450, 4.9, 170), (1, 10000450, 6.0, 175),
        (1, 10000400, 1.0, None),    # null weight: neither averaged nor staged
        (2, 30005110, 1.0, 250), (2, 15001552, 1.0, 2),
        (2, 20000700, 1.0, 0), (2, 24000550, 1.0, 5),
        (3, 200, 7.0, 70),
        (9, 200, 1.0, 70),           # patient not in general_table
    ]
    values = ",\n".join(
        f"({p}, {v}, {at(h)}, {'NULL' if x is None else x})" for p, v, h, x in rows)
    obs = root / "observation_tables" / "parquet"
    obs.mkdir(parents=True)
    con.execute(f"""COPY (SELECT col0::INTEGER AS patientid, col1::INTEGER AS variableid,
                                 col2 AS datetime, col3::FLOAT AS value, NULL::VARCHAR AS stringvalue
                          FROM (VALUES {values})) TO '{obs}/part-0.parquet' (FORMAT parquet)""")
    drugs = [
        # patientid, pharmaid, infusionid, hours, dose, unit, route
        (1, 1000462, 11, 1.0, 0.1, "µg", "iv-inf"),
        (1, 1000462, 11, 1.0, 0.1, "µg", "iv-inf"),   # exact duplicate
        (1, 1000999, 12, 2.0, 1.3, "g", "p os"),
        (1, 1000777, None, 3.0, 5, "ml", None),           # name 'NA' -> id; no infusion
        (2, 555, 13, 4.0, 2, "U", "sc-inj"),              # id without reference row
        (9, 1000462, 14, 1.0, 1, "mg", "iv-inf"),         # unknown patient
    ]
    values = ",\n".join(
        f"({p}, {ph}, {'NULL' if inf is None else inf}, {at(h)}, {d}, '{u}', "
        f"{'NULL' if r is None else repr(r)})" for p, ph, inf, h, d, u, r in drugs)
    ph = root / "pharma_records" / "parquet"
    ph.mkdir(parents=True)
    con.execute(f"""COPY (SELECT col0::INTEGER AS patientid, col1::INTEGER AS pharmaid,
                                 col2::BIGINT AS infusionid, col3 AS givenat, col4::FLOAT AS givendose,
                                 col5 AS doseunit, col6 AS route
                          FROM (VALUES {values})) TO '{ph}/part-0.parquet' (FORMAT parquet)""")


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("hirid")
    con = db.connect_plain()
    _toy_source(root / "src", con)
    cfg = config.Config(out_dir=root / "out", vocabulary_dir=root, sources={"hirid": root / "src"})
    hirid.convert(cfg, con)
    hirid.stage(cfg, con)
    staging.validate(cfg.staging_dir("hirid"), con)
    return con, cfg.staging_dir("hirid")


def _rows(con, d, table, sql):
    return con.sql(sql.replace("{t}", f"read_parquet('{d}/{table}.parquet')")).fetchall()


def test_stays(staged):
    con, d = staged
    got = {r[0]: r[1:] for r in _rows(con, d, "stays", """
        SELECT stay_id, person_key, los_us, age, sex, mortality, icu_death, death_t_us, height, weight
        FROM {t} ORDER BY stay_id""")}
    # stay end = last observation of any variable (the unmapped 999 at 20.25 h)
    assert got[1] == ("1", round(20.25 * H), 65.0, "M", True, True, round(12.5 * H),
                      pytest.approx((180 + 170) / 2), None)
    assert got[2][1:] == (1 * H, 40.0, "F", False, False, None, None, None)
    assert got[3][4:7] == (None, False, None)    # unknown outcome
    assert got[4][1] is None and got[4][5:7] == (True, None)   # dead, no observation, no time
    assert sorted(got) == [1, 2, 3, 4]


def test_measurement_values(staged):
    con, d = staged
    got = {}
    for s, t, c, v, u in _rows(con, d, "measurement", """
            SELECT source_value, t_us, concept_id, value, unit_concept_id FROM {t}"""):
        got.setdefault(s, []).append((t, c, v, u))
    # as recorded, in the unit of the variable (g/L, ng/L)
    assert got["24000548"] == [(1 * H, 3000963, 123.0, 8636)]
    assert got["24000806"] == [(1 * H, 3019800, 7.0, 8725)]
    # neutrophils as an absolute count (the % of wbc is computed by the cohorts, per hour)
    assert sorted(got["24000550"]) == [(1 * H, 3013650, 5.0, 8848), (2 * H, 3013650, 6.0, 8848),
                                       (3 * H, 3013650, 6.0, 8848)]
    assert got["derived:gcs_total"] == [(4 * H, 3007194, 4 + 6 + 4.5, 0)]
    assert sorted(got["200"]) == [(-1 * H, 3027018, 80.0, 8541), (7 * H, 3027018, 70.0, 8541),
                                  (10 * H, 3027018, 90.0, 8541)]
    assert sorted(got["10000450"]) == [(-2 * H, 3036277, 180.0, 8582),
                                       (round(4.9 * H), 3036277, 170.0, 8582),
                                       (6 * H, 3036277, 175.0, 8582)]
    assert got["30005110"] == [(1 * H, 0, 250.0, 8587)]
    assert got["15001552"] == [(1 * H, 0, 2.0, 0)]
    assert len(got["20000700"]) == 3
    # unmapped items, the null weight, the 110 null row and the unknown patient are absent
    assert not {"999", "10000400", "110"} & set(got)
    # 200 x3, hgb, tnt, neut x3, gcs total, height x3, fluid balance, airway, wbc x3, gcs parts x6
    assert sum(len(v) for v in got.values()) == 23


def test_measurement_text_columns_null(staged):
    con, d = staged
    assert _rows(con, d, "measurement", """
        SELECT count(value_text), count(unit_source_value) FROM {t}""") == [(0, 0)]


def test_drugs(staged):
    con, d = staged
    got = _rows(con, d, "drug", """
        SELECT stay_id, t_start_us, t_end_us, source_value, quantity, dose_unit, route, sig, admission_drug
        FROM {t} ORDER BY stay_id, t_start_us""")
    assert got == [
        (1, 1 * H, 1 * H, "Noradrenalin  20 mg/50 ml", f32(0.1) * 0.001, "mg", "iv-inf",
         "group=11;item=1000462", False),
        (1, 2 * H, 2 * H, "Gramdrug", f32(1.3) * 1000.0, "mg", "p os", "group=12;item=1000999", False),
        (1, 3 * H, 3 * H, "1000777", 5.0, "ml", None, "item=1000777", False),
        (2, 4 * H, 4 * H, "555", 2.0, "U", "sc-inj", "group=13;item=555", False),
    ]


def test_mapping_file_is_consistent():
    con = db.connect_plain()
    rows = con.sql(f"""SELECT count(*), count(DISTINCT source_code),
                              count(*) FILTER (WHERE rule <> 'gcs_component'),
                              count(*) FILTER (WHERE rule = 'gcs_component')
                       FROM read_csv('{hirid.MEASUREMENTS_CSV}', header = true)""").fetchone()
    n, n_codes, unknown_rules, n_gcs = rows
    assert n == n_codes and unknown_rules == 0 and n_gcs == 3
