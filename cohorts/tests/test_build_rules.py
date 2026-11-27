"""Builder rules against ricu's task scripts (YAIB-cohorts, R/). Synthetic data only.

Run:  python -m pytest tests/test_build_rules.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

duckdb = pytest.importorskip("duckdb")
ed = pytest.importorskip("ehrdata")


def _builder():
    if "bn_rules" in sys.modules:
        return sys.modules["bn_rules"]
    spec = importlib.util.spec_from_file_location("bn_rules", os.path.join(HERE, "..", "build_cohort.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["bn_rules"] = m
    spec.loader.exec_module(m)
    return m


def test_los_label_is_floored_los_minus_t():
    bn = _builder()
    t = np.arange(4)
    assert list(bn.los_label(10.7, t)) == [10.0, 9.0, 8.0, 7.0]
    assert bn.los_label(400.0, np.array([0]))[0] == 168.0


def test_kidney_function_label_is_median_of_hourly_medians():
    bn = _builder()
    x = np.full((2, 24), np.nan)
    x[0, [0, 1, 2]] = [1.0, 2.0, 10.0]
    assert bn.kidney_function_label(x)[0] == 2.0
    assert np.isnan(bn.kidney_function_label(x)[1]), "no creatinine in hours 25-48 -> NaN, excluded"


def test_missing_age_is_not_an_exclusion():
    bn = _builder()
    df = pd.DataFrame({"los_hours": [20.0, 20.0, np.nan], "n_hours": [10, 10, 10],
                       "max_empty_run": [0, 0, 0], "age": [np.nan, 10.0, 50.0]})
    kept = bn.base_exclusions(df)
    assert list(kept.index) == [0, 2]


def test_stay_without_measurements_skips_excl3():
    """ricu counts hours per stay over the rows it loaded; a stay with no row is not in that
    table, so only the empty-run rule can drop it."""
    bn = _builder()
    df = pd.DataFrame({"los_hours": [8.0, 8.0, 30.0], "n_hours": [np.nan, 2, np.nan],
                       "max_empty_run": [9, 0, 31], "age": [50.0, 50.0, 50.0]})
    assert list(bn.base_exclusions(df).index) == [0]


HR = 3027018          # heart_rate, ricu bounds 0-300
T0 = pd.Timestamp("2100-01-01 00:00:00")


def _rows(voi, hours, value=80.0):
    return [(voi, voi, HR, T0 + pd.Timedelta(hours=h, minutes=30), value) for h in hours]


def _scoped(tmp_path, meas, vois, los_h=None, ds="miiv"):
    """A connection scoped (ehrdata_dynamics.scope_cohort) to the stays `vois` of a small CDM: one
    person per stay, starts at T0, ends after `los_h[voi]` hours (default 200), with `meas` as the
    cohort cache."""
    import ehrdata_dynamics as ED
    meas = meas.copy()
    if "person_id" not in meas:
        meas["person_id"] = meas["visit_occurrence_id"]
    meas["measurement_date"] = meas["measurement_datetime"].dt.date
    for col, default in (("unit_concept_id", 0), ("unit_source_value", "x")):
        if col not in meas:
            meas[col] = default
    meas.to_parquet(tmp_path / f"measurement_{ds}.parquet")
    los_h = los_h or {}
    ends = [T0 + pd.Timedelta(hours=los_h.get(v, 200)) for v in vois]
    pd.DataFrame({"visit_occurrence_id": vois, "person_id": vois, "visit_source_value": [f"mimic4-{v}" for v in vois],
                  "visit_concept_id": 32037, "visit_start_date": T0.date(), "visit_start_datetime": T0,
                  "visit_end_date": [e.date() for e in ends], "visit_end_datetime": ends,
                  "care_site_id": 1, "discharged_to_concept_id": 0}).to_parquet(tmp_path / "VISIT_OCCURRENCE.parquet")
    pd.DataFrame({"person_id": vois, "year_of_birth": 1950, "gender_concept_id": 8507,
                  "birth_datetime": pd.Timestamp("1950-01-01")}).to_parquet(tmp_path / "PERSON.parquet")
    c = duckdb.connect()
    c.execute(f"CREATE TABLE cohort_voi AS SELECT * FROM (VALUES {', '.join(f'({v})' for v in vois)}) t(voi)")
    assert ED.scope_cohort(c, str(tmp_path), str(tmp_path / f"measurement_{ds}.parquet"), ds,
                           voi_table="cohort_voi") == len(vois)
    c.execute("CREATE VIEW observation AS SELECT 0::BIGINT AS visit_occurrence_id, 0 AS observation_concept_id, "
              "0.0 AS value_as_number WHERE false")
    return c


@pytest.fixture()
def con(tmp_path):
    stays = {  # voi: (los_h, hours with an in-range value, extra out-of-range hours)
        1: (20.5, range(0, 21), []),                               # dense: kept
        2: (20.5, [0, 1, 2, 3, 16, 17, 18, 19, 20], []),           # 12 empty bins: kept
        3: (20.5, [0, 1, 2, 3, 17, 18, 19, 20], []),               # 13 empty bins: excluded
        4: (20.5, [0, 10, 20], []),                                # 3 hours: excluded
        5: (20.5, [0, 1, 2, 3, 18, 19, 20], range(4, 18)),         # out-of-range rows do not count
        7: (6.5, [-3, -2, -1, 6], []),                             # pre-ICU hours count for excl3
    }
    rows = []
    for voi, (_los, hrs, bad) in stays.items():
        rows += _rows(voi, hrs) + _rows(voi, bad, value=999.0)
    meas = pd.DataFrame(rows, columns=["person_id", "visit_occurrence_id", "measurement_concept_id",
                                       "measurement_datetime", "value_as_number"])
    meas["unit_concept_id"] = 8541  # beats per minute
    return _scoped(tmp_path, meas, list(stays), {v: s[0] for v, s in stays.items()})


def test_base_exclusions_follow_base_cohort_R(con):
    bn = _builder()
    obs = bn.base_candidates(con, "miiv").obs
    assert list(obs["n_hours"]) == [21, 9, 8, 3, 7, 4]
    assert list(obs["max_empty_run"]) == [0, 12, 13, 9, 14, 6]
    assert sorted(bn.base_exclusions(obs)["voi"]) == [1, 2, 7]


def test_setup_arguments_aggregate_bounded_rows(con):
    import ehrdata_dynamics as ED
    stays = ED.setup_stays(con)
    rows = np.flatnonzero(stays.obs["voi"].isin([5, 7]).to_numpy())
    e = ed.io.omop.setup_variables(stays[rows], backend_handle=con, layer=ED.LAYER, sparse=True,
                                   **ED.setup_arguments({"hr": ED.dynamic_set("miiv", "hr")}, -3, 21))
    g = ED.variable(e, "hr")
    assert g.shape == (2, 24)
    assert np.isnan(g[0, 3 + 4]), "an hour with only out-of-range rows is empty"
    assert g[0, 3 + 0] == 80.0
    assert g[1, 0] == 80.0, "column 0 is hour -3"
    sets = {"map": ED.dynamic_set("miiv", "map"), "hr": ED.dynamic_set("miiv", "hr")}
    two = ed.io.omop.setup_variables(stays[rows], backend_handle=con, layer=ED.LAYER, sparse=True,
                                     **ED.setup_arguments(sets, -3, 21))
    assert list(two.var_names) == ["map", "hr"], "the variables in the order of the sets, which the layer's users index"


SBP_INV, SBP_NI, URINE = 3004249, 21492239, 3014315


def test_dynamics_median_over_pathways_and_urine_sum(tmp_path):
    """ehrdata on the cohort's rows: median per hour over the union of a variable's pathways
    (ricu's numeric default), urine summed, out-of-range rows dropped before aggregation, and a row
    attributed to the stay its visit_occurrence_id names even when another stay of the person overlaps."""
    import ehrdata_dynamics as ED
    t = lambda h, m=0: T0 + pd.Timedelta(hours=h, minutes=m)
    rows = [
        (1, 1, SBP_INV, t(0, 10), 100.0), (1, 1, SBP_NI, t(0, 20), 120.0), (1, 1, SBP_NI, t(0, 30), 140.0),
        (1, 1, SBP_INV, t(1, 10), 110.0), (1, 1, SBP_INV, t(1, 50), 999.0),     # 999 out of bounds
        (1, 1, URINE, t(0, 5), 50.0), (1, 1, URINE, t(0, 40), 30.0),
        (1, 2, SBP_INV, t(0, 15), 90.0),                                          # the other stay
    ]
    meas = pd.DataFrame(rows, columns=["person_id", "visit_occurrence_id", "measurement_concept_id",
                                       "measurement_datetime", "value_as_number"])
    meas["measurement_date"] = meas["measurement_datetime"].dt.date
    meas["unit_concept_id"] = 0
    meas["unit_source_value"] = "x"
    meas.to_parquet(tmp_path / "measurement_miiv.parquet")
    pd.DataFrame({"visit_occurrence_id": [1, 2], "person_id": [1, 1], "visit_source_value": ["mimic4-11", "mimic4-12"],
                  "visit_concept_id": [9201, 9201],
                  "visit_start_date": [T0.date()] * 2, "visit_start_datetime": [T0] * 2,
                  "visit_end_date": [t(5).date()] * 2, "visit_end_datetime": [t(5)] * 2}).to_parquet(tmp_path / "VISIT_OCCURRENCE.parquet")
    pd.DataFrame({"person_id": [1], "year_of_birth": [1950], "gender_concept_id": [8507],
                  "birth_datetime": [pd.Timestamp("1950-01-01")]}).to_parquet(tmp_path / "PERSON.parquet")
    c = duckdb.connect()
    c.execute("CREATE TABLE cohort_voi AS SELECT * FROM (VALUES (1), (2)) t(voi)")
    assert ED.scope_cohort(c, str(tmp_path), str(tmp_path / "measurement_miiv.parquet"), "miiv",
                           voi_table="cohort_voi") == 2
    edata = _builder().build_object(c, ED.setup_stays(c), "miiv", 0, 3)
    y = edata.layers[ED.LAYER].todense()
    vi = {v: j for j, v in enumerate(ED.YAIB_DYNAMIC_VARS)}
    assert list(edata.obs["stay_id"]) == [11, 12]
    assert y[0, vi["sbp"], 0] == 120.0, "median over invasive and non-invasive rows"
    assert y[0, vi["sbp"], 1] == 110.0, "the out-of-range row is dropped before aggregation"
    assert y[0, vi["urine"], 0] == 80.0, "urine is summed per hour"
    assert y[1, vi["sbp"], 0] == 90.0, "each stay keeps its own rows"


def test_hospital_without_positive_reads_the_visits_care_site(tmp_path, monkeypatch):
    """eicu exclusion 6: stays of a hospital where no stay is positive. PERSON's empty care_site_id
    must not hide the visit's."""
    bn = _builder()
    meas = pd.DataFrame({"visit_occurrence_id": [1], "measurement_concept_id": [HR],
                         "measurement_datetime": [T0], "value_as_number": [80.0]})
    c = _scoped(tmp_path, meas, [1, 2, 3], ds="eicu")
    vo = pd.read_parquet(tmp_path / "VISIT_OCCURRENCE.parquet").assign(
        visit_source_value=["eicu-1", "eicu-2", "eicu-3"], care_site_id=[10, 10, 20])
    vo.to_parquet(tmp_path / "VISIT_OCCURRENCE.parquet")
    pd.read_parquet(tmp_path / "PERSON.parquet").assign(care_site_id=None).to_parquet(tmp_path / "PERSON.parquet")
    pd.DataFrame({"care_site_id": [10, 20], "care_site_source_value": ["73", "75"]}).to_parquet(tmp_path / "CARE_SITE.parquet")
    bn.ED.scope_cohort(c, str(tmp_path), str(tmp_path / "measurement_eicu.parquet"), "eicu", voi_table="cohort_voi")
    monkeypatch.setattr(bn, "OMOP", str(tmp_path))
    monkeypatch.setattr(bn, "CACHE", str(tmp_path))
    monkeypatch.setattr(bn, "HOSP_CHUNK", 2)
    monkeypatch.setattr(bn, "_statics_height_weight", lambda con, path, voi, ds: (None, np.full(len(voi), 70.0)))
    chunks = []
    onset = lambda s, w: chunks.append(list(s.obs["voi"])) or np.isin(s.obs["voi"], [1])  # only stay 1 (hospital 73)
    excl = bn._hospital_exclusion(c, "eicu", np.array([1, 3]), onset)
    assert excl.tolist() == [False, True], "the cohort's stays of hospitals without a positive"
    assert chunks == [[1, 2], [3]], "every stay of the scope, in chunks"


def test_hirid_neutrophils_are_the_percent_of_wbc_per_hour(tmp_path):
    """ricu-extensions' blood_cell_ratio: 100 x count / wbc of the same hour (not the same minute);
    no wbc, a wbc of 0 or a ratio outside ricu's 0-100 leave the hour empty."""
    import ehrdata_dynamics as ED
    from yaib_omop_crosswalk import NEUTROPHILS_ABSOLUTE
    wbc = int(ED.concept_sets("hirid")["wbc"][0])
    t = lambda h, m: T0 + pd.Timedelta(hours=h, minutes=m)
    rows = [(1, NEUTROPHILS_ABSOLUTE, t(0, 10), 6.0), (1, wbc, t(0, 40), 8.0), (1, wbc, t(0, 50), 10.0),
            (1, NEUTROPHILS_ABSOLUTE, t(1, 10), 5.0),                                   # no wbc
            (1, NEUTROPHILS_ABSOLUTE, t(2, 10), 5.0), (1, wbc, t(2, 20), 0.0),          # wbc 0
            (1, NEUTROPHILS_ABSOLUTE, t(3, 10), 12.0), (1, wbc, t(3, 20), 10.0)]        # 120 %
    meas = pd.DataFrame(rows, columns=["visit_occurrence_id", "measurement_concept_id",
                                       "measurement_datetime", "value_as_number"])
    meas["unit_concept_id"] = 8848
    c = _scoped(tmp_path, meas, [1], ds="hirid")
    y = _builder().build_object(c, ED.setup_stays(c), "hirid", 0, 4).layers[ED.LAYER].todense()
    neut = y[0, ED.YAIB_DYNAMIC_VARS.index("neut")]
    assert neut[0] == 100 * 6 / 9, "the hour's median wbc"
    assert np.isnan(neut[1:]).all()
    assert y[0, ED.YAIB_DYNAMIC_VARS.index("wbc"), 0] == 9.0


def test_dyn_frame_rows_and_lengths():
    import ehrdata_dynamics as ED
    import sparse
    x = sparse.COO(np.array([[0, 0, 1], [0, 3, 0], [0, 2, 1]]), np.array([1.0, 2.0, 3.0]),
                   shape=(2, 48, 3), fill_value=np.nan)
    dyn = ED.dyn_frame(x, np.array([5, 7]), np.array([3, 1]))
    assert list(dyn["stay_id"]) == [5, 5, 5, 7] and list(dyn["time"]) == [0, 1, 2, 0]
    assert dyn.loc[0, ED.YAIB_DYNAMIC_VARS[0]] == 1.0 and dyn.loc[2, ED.YAIB_DYNAMIC_VARS[3]] == 2.0
    assert np.isnan(dyn.iloc[3, 2:]).all(), "stay 7's hour 1 lies beyond its grid"
    sub = ED.take_rows(x, np.array([1]))
    assert sub.shape == (1, 48, 3) and sub.nnz == 1


def _object_inputs(tmp_path, monkeypatch, rows, ds="miiv"):
    """The builder's scope (`_scope`) of a small CDM with MEASUREMENT rows (voi, concept, hour, value,
    source value), every concept in YAIB's unit, and the build's EHRData of its stays on [-24, 10)."""
    bn = _builder()
    from yaib_omop_crosswalk import CONCEPT_TO_YAIB, UNIT_CONVERSIONS
    meas = pd.DataFrame([(v, cid, T0 + pd.Timedelta(hours=h, minutes=10), x, src) for v, cid, h, x, src in rows],
                        columns=["visit_occurrence_id", "measurement_concept_id", "measurement_datetime",
                                 "value_as_number", "measurement_source_value"])
    meas["unit_concept_id"] = [UNIT_CONVERSIONS.get(CONCEPT_TO_YAIB.get(c), (0,))[0] for c in meas["measurement_concept_id"]]
    vois = sorted({r[0] for r in rows})
    c = _scoped(tmp_path, meas, vois, ds=ds)
    monkeypatch.setattr(bn, "OMOP", str(tmp_path))
    bn._scope(c, str(tmp_path / f"measurement_{ds}.parquet"), ds, vois)
    return bn, c, bn.build_object(c, bn.ED.setup_stays(c), ds, -24, 10)


def test_object_frames_equal_a_setup_on_the_frame_hours(tmp_path, monkeypatch):
    """The frames take hours [0, T) of the object, which equal the 48 variables set up on [0, T) alone."""
    import ehrdata_dynamics as ED
    hr, crea = int(ED.concept_sets("miiv")["hr"][0]), int(ED.concept_sets("miiv")["crea"][0])
    rows = [(1, hr, h, 80.0 + h, "x") for h in (-20, -1, 0, 4, 9, 12)] + [(1, crea, h, 1.0, "x") for h in (-3, 2)]
    rows += [(2, hr, h, 70.0, "x") for h in (-5, 3)]
    bn, c, edata = _object_inputs(tmp_path, monkeypatch, rows)
    alone = ed.io.omop.setup_variables(ED.setup_stays(c), backend_handle=c, layer=ED.LAYER, sparse=True,
                                       **ED.setup_arguments({v: ED.dynamic_set("miiv", v) for v in ED.YAIB_DYNAMIC_VARS}, 0, 7))
    frame = edata[:, :48, 24:31].layers[ED.LAYER]   # the frames' slice: hours [0, 7) of an axis from -24
    assert frame.shape == (2, 48, 7)
    assert np.array_equal(frame.todense(), alone.layers[ED.LAYER].todense(), equal_nan=True)
    assert list(edata.var_names[48:]) == ["egcs", "vgcs", "mgcs", "tgcs", *(f"{v}_{h}" for v, h in bn.SECOND_AGGREGATIONS)]
    assert ED.variable(edata, "hr")[0, 24 - 20] == 60.0, "the object holds the hours before admission"


def test_object_label_inputs_equal_the_hourly_grids(tmp_path, monkeypatch):
    """The label inputs of the object, rounded as the labels read them, equal a setup of each alone for the second
    aggregations, and the GCS components of `LI.gcs` without the ETL's derived total."""
    import ehrdata_dynamics as ED
    import label_inputs as LI
    cs = ED.concept_sets("miiv")
    rows = [(1, cs["crea"][0], h, x, "x") for h, x in [(-22, 0.9), (-22, 1.4), (1, 1.1), (1, 1.3), (1, 1.6), (5, 20.0)]]
    rows += [(1, cs["po2"][0], 2, x, "x") for x in (80.0, 95.0, 30.0)]
    rows += [(1, cs["fio2"][0], 2, x, "x") for x in (40.0, 60.0)]
    rows += [(2, cs["plt"][0], h, x, "x") for h, x in [(0, 150.0), (0, 90.0), (-1, 300.0)]]
    rows += [(2, cs["map"][0], 3, 65.0, "x"), (2, cs["map"][1], 3, 55.0, "x"), (2, cs["bili"][0], 4, 2.5, "x")]
    rows += [(1, LI.GCS_CONCEPTS["egcs"], h, x, "x") for h, x in [(0, 3.0), (0, 2.0), (1, 9.0), (-2, 4.0)]]
    rows += [(1, LI.GCS_CONCEPTS["vgcs"], 0, 4.0, "x"), (1, LI.GCS_CONCEPTS["mgcs"], 6, 5.0, "x")]
    rows += [(1, LI.GCS_CONCEPTS["tgcs"], 0, 12.0, LI.DERIVED_GCS_TOTAL), (2, LI.GCS_CONCEPTS["tgcs"], 1, 9.0, "x")]
    bn, c, edata = _object_inputs(tmp_path, monkeypatch, rows)
    stays = ED.setup_stays(c)
    for var, how in bn.SECOND_AGGREGATIONS:
        name = f"{var}_{how}"
        alone = ed.io.omop.setup_variables(stays, backend_handle=c, layer=ED.LAYER, sparse=True,
                                           **ED.setup_arguments({name: ED.dynamic_set("miiv", var, how)}, -24, 10))
        got = ED.variable(edata, name, rounded=True)
        assert np.array_equal(got, ED.variable(alone, name, rounded=True), equal_nan=True), var
    assert ED.variable(edata, "crea_max")[0, 24 + 1] == 1.6 and ED.variable(edata, "crea")[0, 24 + 1] == 1.3
    assert np.isnan(ED.variable(edata, "crea_max")[0, 24 + 5]), "ricu's bounds"
    gcs = {k: ED.variable(edata, k) for k in LI.GCS_CONCEPTS}
    assert gcs["egcs"][0, 24] == 2.0 and np.isnan(gcs["egcs"][0, 25]), "the hourly minimum inside ricu's bounds"
    assert np.isnan(gcs["tgcs"][0]).all() and gcs["tgcs"][1, 25] == 9.0, "the derived total is left out"
    total = LI.gcs(c, str(tmp_path / "measurement_miiv.parquet"), "hirid", edata.obs, gcs, -24, 10)
    assert total[0, 24] == 2.0 + 4.0 + 6.0, "the hour's eye and verbal minima, the missing motor at its maximum"


def test_first_last_hours_bin_and_bound_as_ehrdata(tmp_path):
    """The first and last hour at any time are those of ehrdata's observed hours: rows converted by their unit
    before ricu's bounds, rows before admission, a row without a datetime at midnight of its date."""
    import ehrdata_dynamics as ED
    from yaib_omop_crosswalk import RICU_BOUNDS
    crea = int(ED.concept_sets("hirid")["crea"][0])
    lo, hi = RICU_BOUNDS["crea"]
    t = lambda h, m=0: T0 + pd.Timedelta(hours=h, minutes=m)
    rows = [(1, t(-31, 1), 1.0, 8840), (1, t(-29), hi * 88.4 + 50, 8749), (1, t(5, 30), 2.0 * 88.4, 8749),
            (1, t(40), hi + 1, 8840),                                              # out of bounds after conversion
            (2, t(-2, 1), 0.9, 8840), (2, t(3), 1.0, 8840),
            (3, t(7), lo - 1, 8840)]
    meas = pd.DataFrame(rows, columns=["visit_occurrence_id", "measurement_datetime", "value_as_number", "unit_concept_id"])
    meas["measurement_concept_id"] = crea
    c = _scoped(tmp_path, meas, [1, 2, 3], ds="hirid")
    pd.read_parquet(tmp_path / "measurement_hirid.parquet").assign(
        measurement_datetime=lambda d: d["measurement_datetime"].where(d["value_as_number"] != 0.9)
    ).to_parquet(tmp_path / "measurement_hirid.parquet")
    stays = ED.setup_stays(c)
    got = ED.first_last_hours(stays, c, {"crea": ED.dynamic_set("hirid", "crea")})
    cells = ED.observed_hours(stays, c, "hirid", ["crea"]).groupby("row")["h"]
    assert np.array_equal(got["crea_first"], cells.min().reindex(range(3)).to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(got["crea_last"], cells.max().reindex(range(3)).to_numpy(dtype=float), equal_nan=True)
    assert list(got["crea_first"][:2]) == [-31.0, -24.0] and list(got["crea_last"][:2]) == [5.0, 3.0]
    assert np.isnan(got.loc[2]).all()


def test_eicu_rass_in_the_object_is_the_sedated_range(tmp_path, monkeypatch):
    """eicu's sedation reads RASS <= -3 (ricu's -5..-3); other RASS rows stay out of the object."""
    import ehrdata_dynamics as ED
    from yaib_omop_crosswalk import RASS_CONCEPT_ID
    rows = [(1, RASS_CONCEPT_ID, h, x, "x") for h, x in [(0, -4.0), (1, -2.0), (2, -5.0), (2, -3.0), (3, 0.0)]]
    _, _, edata = _object_inputs(tmp_path, monkeypatch, rows, ds="eicu")
    rass = ED.variable(edata, "rass")[0, 24:28]
    assert rass[0] == -4.0 and np.isnan(rass[1]) and rass[2] == -5.0 and np.isnan(rass[3])


def test_label_inputs_are_the_charted_values(tmp_path):
    """mg/dL creatinine comes back as charted, and an hour's median of two rows is their plain mean,
    as ricu computes it; label inputs are rounded to 6 decimals, which leaves both as they are."""
    import ehrdata_dynamics as ED
    crea = 3016723
    meas = pd.DataFrame({"visit_occurrence_id": [1, 1, 1], "measurement_concept_id": [crea] * 3,
                         "measurement_datetime": [T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=1, minutes=5)],
                         "value_as_number": [1.3, 0.85, 0.86], "unit_concept_id": [8840] * 3})
    c = _scoped(tmp_path, meas, [1])
    e = ed.io.omop.setup_variables(ED.setup_stays(c), backend_handle=c, layer=ED.LAYER, sparse=True,
                                   **ED.setup_arguments({"crea": ED.dynamic_set("miiv", "crea")}, 0, 2))
    v = ED.variable(e, "crea",
                    rounded=True)[0]
    assert v[0] == 1.3
    assert v[1] == (0.85 + 0.86) / 2


def test_converted_values_are_rounded(tmp_path):
    """1.3 mg/dL stored as umol/L comes back as the charted decimal, so KDIGO's +0.3 tie holds."""
    import ehrdata_dynamics as ED
    crea = 40762887
    meas = pd.DataFrame({"visit_occurrence_id": [1, 1], "measurement_concept_id": [crea, crea],
                         "measurement_datetime": [T0, T0 + pd.Timedelta(hours=1)],
                         "value_as_number": [1.0 * 88.4, 1.3 * 88.4], "unit_concept_id": [8749, 8749]})
    c = _scoped(tmp_path, meas, [1], ds="hirid")
    e = ed.io.omop.setup_variables(ED.setup_stays(c), backend_handle=c, layer=ED.LAYER, sparse=True,
                                   **ED.setup_arguments({"crea": ED.dynamic_set("hirid", "crea")}, 0, 2))
    v = ED.variable(e, "crea",
                    rounded=True)[0]
    assert list(v) == [1.0, 1.3]
    assert v[1] >= v[0] + 0.3


@pytest.mark.parametrize("convert", [False, True])
def test_ionized_calcium_as_ricu_or_converted(tmp_path, monkeypatch, convert):
    """eICU ionized calcium in mg/dL or mEq/L is read as mmol/L by default, as ricu reads it, so
    ricu's 0.5-2 mmol/L bounds drop it; with CAI_CONVERT it is converted by its unit and kept."""
    import ehrdata_dynamics as ED
    import yaib_omop_crosswalk as cw
    monkeypatch.setitem(cw.UNIT_CONVERSIONS, "cai", (cw.MMOL_L, cw.CAI_CONVERTED if convert else cw.CAI_AS_RICU))
    cai = int(cw.var_concepts("cai", "eicu")[0])
    meas = pd.DataFrame({"visit_occurrence_id": [1] * 4, "measurement_concept_id": [cai] * 4,
                         "measurement_datetime": [T0 + pd.Timedelta(hours=h) for h in range(4)],
                         "value_as_number": [1.1, 4.6, 2.3, 2.4],
                         "unit_concept_id": [cw.MMOL_L, cw.MG_DL, cw.MEQ_L, cw.MMOL_L]})
    c = _scoped(tmp_path, meas, [1], ds="eicu")
    X = _builder().build_object(c, ED.setup_stays(c), "eicu", 0, 4).layers[ED.LAYER].todense()
    v = X[0, cw.YAIB_DYNAMIC_VARS.index("cai")]
    assert v[0] == 1.1 and np.isnan(v[3])  # 2.4 mmol/L is outside ricu's bounds either way
    if convert:
        assert v[1] == pytest.approx(4.6 / 4.008) and v[2] == pytest.approx(1.15)
    else:
        assert np.isnan(v[1]) and np.isnan(v[2])


def test_aki_reads_creatinine_before_the_label_grid(tmp_path, monkeypatch):
    """ricu runs KDIGO and the excl8 baseline on every earlier creatinine value; a stay with
    values before -LABEL_PRE_H gets its own longer grid."""
    if not hasattr(pytest.importorskip("ehrapy").tl, "kdigo"):
        pytest.skip("this ehrapy has no tl.kdigo")
    bn = _builder()
    cid = int(bn.ED.concept_sets("miiv")["crea"][0])
    pre = bn.LABEL_PRE_H
    rows = [(1, cid, T0 + pd.Timedelta(hours=h), x) for h, x in
            [(-pre - 112, 0.8), (-pre - 102, 5.0), (10, 1.0)]]
    rows += [(2, cid, T0 + pd.Timedelta(hours=h), x) for h, x in [(-20, 1.0), (10, 1.7)]]
    meas = pd.DataFrame(rows, columns=["visit_occurrence_id", "measurement_concept_id",
                                       "measurement_datetime", "value_as_number"])
    meas["unit_concept_id"] = 8840  # mg/dL
    c = _scoped(tmp_path, meas, [1, 2])
    monkeypatch.setattr(bn, "CACHE", str(tmp_path))
    edata = bn._aki_onset(c, "miiv", bn.ED.setup_stays(c), np.array([70.0, 70.0]), 24)
    assert list(edata.var_names) == ["crea", "urine_rate"] and edata.obsm["kdigo"].shape == (2, pre + 24)
    onset, obs = bn._obs_hours(edata.obs, "aki_first"), edata.obs
    assert onset[0] == -pre - 102, "5.0 >= 3 x 0.8 ten hours later, before the grid"
    assert np.isclose(obs["crea_baseline"].iloc[0], 0.8), "min of all pre-ICU values and the first ICU value"
    assert obs["crea_hours_pre"].iloc[0] == 2 and obs["crea_hours_icu"].iloc[0] == 1
    assert onset[1] == 10, "1.7 >= 1.5 x 1.0"
    assert np.isclose(obs["crea_baseline"].iloc[1], 1.0)


@pytest.mark.parametrize("ds", ["miiv", "hirid"])
def test_urine_rate_from_ehrdata_as_ricu(tmp_path, ds):
    """The builder's urine rate from the urine rows as ehrdata sets them up: per minute and ricu's divisor on miiv,
    the largest row of an hour on hirid, which records the rate; rows outside ricu's bounds left out."""
    bn = _builder()
    cid = int(bn.ED.concept_sets(ds)["urine"][0])
    rows = [(-180, 50.0), (10, 100.0), (40, 60.0), (65, 80.0), (70, 3000.0), (1800, 40.0)]
    meas = pd.DataFrame({"visit_occurrence_id": 1, "measurement_concept_id": cid,
                         "measurement_datetime": [T0 + pd.Timedelta(minutes=m) for m, _ in rows],
                         "value_as_number": [x for _, x in rows]})
    c = _scoped(tmp_path, meas, [1], ds=ds)
    u = bn.urine_rate(c, bn.ED.setup_stays(c), ds, 32)
    if ds == "hirid":
        assert (u[0, 0], u[0, 1], u[0, 30]) == (100.0, 80.0, 40.0), "the largest row of the hour"
    else:
        assert (u[0, 0], u[0, 1], u[0, 30]) == (60.0, 40.0, 40.0), "as test_label_inputs: 3000 mL is out of bounds"
    assert np.isnan(u[0, 2:30]).all()


def test_abx_stop_before_start_is_a_negative_duration(tmp_path):
    """ricu's abx_duration keeps drugstopoffset - drugstartoffset even when negative; the CDM
    sets the end to the start and keeps the source stop in sig. ricu floors the duration to the hour."""
    bn = _builder()
    t = lambda h: T0 + pd.Timedelta(hours=h)
    pd.DataFrame({"visit_occurrence_id": [1, 1], "drug_source_value": ["Zosyn", "Zosyn"],
                  "drug_exposure_start_datetime": [t(2), t(10)], "drug_exposure_end_datetime": [t(2), t(13)],
                  "sig": ["table=medication;source_stop_min=-90", "table=medication"],
                  "drug_type_concept_id": [0, 0]}).to_parquet(tmp_path / "DRUG_EXPOSURE.parquet")
    c = duckdb.connect()
    stays = pd.DataFrame({"voi": [1], "person_id": [1], "visit_start_datetime": [T0]})
    rows = bn.LI.infection_tables(c, str(tmp_path), "eicu", stays)
    got = c.execute(f"SELECT dur_floor FROM {bn.LI.ABX_TABLE} ORDER BY drug_exposure_start_datetime").fetchall()
    assert got == [(-2.0,), (3.0,)], got
    assert rows["n_abx"].tolist() == [2] and rows["abx_first"].tolist() == [2.0]


def _sofa_inputs(tmp_path, rows):
    """A miiv-like CDM with MEASUREMENT rows only: (voi, var, hour, value) in YAIB units."""
    bn = _builder()
    t = lambda h: T0 + pd.Timedelta(hours=h, minutes=10)
    from yaib_omop_crosswalk import UNIT_CONVERSIONS
    meas = pd.DataFrame([(v, v, int(bn.ED.concept_sets("miiv")[var][0]), t(h), x, UNIT_CONVERSIONS.get(var, (0,))[0])
                         for v, var, h, x in rows],
                        columns=["person_id", "visit_occurrence_id", "measurement_concept_id",
                                 "measurement_datetime", "value_as_number", "unit_concept_id"])
    meas["measurement_source_value"] = "x"
    meas["value_source_value"] = None
    # one row of another stay gives the empty tables their types
    pd.DataFrame({"visit_occurrence_id": [999], "procedure_concept_id": [0], "procedure_datetime": [T0],
                  "procedure_end_datetime": [T0]}).to_parquet(tmp_path / "PROCEDURE_OCCURRENCE.parquet")
    pd.DataFrame({"visit_occurrence_id": [999], "drug_source_value": ["x"], "drug_exposure_start_datetime": [T0],
                  "drug_exposure_end_datetime": [T0], "sig": ["item=0"], "quantity": [0.0],
                  "dose_unit_source_value": ["mg"]}).to_parquet(tmp_path / "DRUG_EXPOSURE.parquet")
    vois = sorted({r[0] for r in rows})
    c = _scoped(tmp_path, meas, vois)
    return bn, c, np.array(vois)


def test_sep3_onset_at_any_time_of_the_stay(tmp_path, monkeypatch):
    """ricu evaluates the SOFA rise in [si - 48, si + 24] wherever the first SI lies, with SOFA
    rows from the stay's first component row and urine24 over all urine rows."""
    if not hasattr(pytest.importorskip("ehrapy").tl, "sepsis3"):
        pytest.skip("this ehrapy has no tl.sepsis3")
    rows = [(1, "plt", -410, 200.0), (1, "plt", -395, 40.0),          # SI long before the label grid
            (2, "plt", 290, 200.0), (2, "plt", 305, 40.0),            # SI after the label grid
            (3, "crea", -2000, 0.8), (3, "plt", 12, 40.0),            # an old row opens the SOFA rows
            (4, "plt", 12, 40.0)]                                     # no earlier row: no baseline
    rows += [(5, "urine", h, 100.0) for h in range(-30, -19)] + [(5, "urine", h, 100.0) for h in range(30, 41)]
    bn, c, voi = _sofa_inputs(tmp_path, rows)
    monkeypatch.setattr(bn, "CACHE", str(tmp_path))
    monkeypatch.setattr(bn, "OMOP", str(tmp_path))
    si = np.array([-400, 300, 10, 10, 10])
    stays = bn.ED.setup_stays(c)
    starts = stays.obs["visit_start_datetime"].copy()
    stays.obs["si_first"] = pd.to_timedelta(si, unit="h")
    bn._sep3_at_si(c, "miiv", stays, np.full(5, 70.0))
    has, first = stays.obs["sepsis3"].to_numpy(), bn._obs_hours(stays.obs, "sepsis3_onset")
    assert list(has) == [True, True, True, False, True]
    assert first[0] == -395 and first[1] == 305 and first[2] == 12
    assert first[4] == 0, "pre-ICU urine opens urine24 rows; 5 of its rows in [-24, 0] read as 480 mL/24 h"
    assert stays.obs["visit_start_datetime"].equals(starts), "the starts are moved on a copy"


def test_tracked_filters_keep_missing_values_whatever_the_dtype():
    """ricu: a missing value never excludes a stay, also in nullable pandas columns."""
    import cohort_filters as CF
    import ehrdata as ed
    obs = pd.DataFrame({"age": pd.array([17, None, 40], dtype="Int64"),
                        "los_hours": pd.array([10.0, None, 3.0], dtype="Float64"),
                        "sex": ["M", "F", "M"],
                        "flag": pd.array([False, None, True], dtype="boolean")},
                       index=["a", "b", "c"])
    edata = ed.EHRData(obs=obs)
    CF.apply(edata, ["~(age < 18)", "~(los_hours < 6)", "~flag"], CF.tracker(edata))
    assert list(edata.obs_names) == ["b"]
