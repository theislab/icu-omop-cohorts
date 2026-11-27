"""The label inputs (label_inputs.py) against ricu's callbacks. Synthetic data."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import label_inputs as LI  # noqa: E402

nan = np.nan


def test_eicu_vent_windows_as_ricu_computes_them():
    w = LI.eicu_vent_windows([0.0, 10.0, 20.0, -8.0, -5.0], [3.0, 30.0, 20.2])
    assert (0.0, 3.0) in w, "an end within 6 h (matched 1 min earlier): lasts until that end"
    assert (10.0, 26.0) in w, "no end within 6 h: ricu's duration is start + 6 h"
    assert (-5.0, 0.0) in w, "a window ending before admission is moved to admission"
    assert len(w) == 3, "under 30 min dropped: the 0.2 h match, and a start 8 h before admission"
    assert LI.eicu_vent_windows([10.0], [], origin=100.0) == [(10.0, 126.0)], "start counted from admission"


def test_window_hours_floor_the_start_and_move_pre_icu_ends():
    assert LI.window_hours(10.8, 13.0) == (10.0, 12.0), "end = floor(floor(start) + duration)"
    assert LI.window_hours(-10.5, -8.0) == (-11.0, 0.0), "an end before admission moves to admission"
    assert LI.window_hours(-10.5, -8.0, origin=20.0) == (-11.0, -9.0), "the grid origin is 20 h after admission"


def test_hirid_vent_lasts_to_next_row_capped():
    w = LI.hirid_vent_windows([0.0, 2.0, 20.0], [1, 3, 9])
    assert w[0] == (0.0, 2.0, True), "code 1 is invasive, lasts until the next row"
    assert w[1] == (2.0, 6.0, False), "gap > 12 h -> 4 h"
    assert len(w) == 2, "codes outside 1-6 are not ventilation"


def test_hirid_vent_rows_at_one_time_ordered_by_code():
    """Rows at the same time come in the order of their code, whatever order they are read in: the first lasts 0 h."""
    for times, codes in (([0.0, 0.0, 5.0], [3, 1, 9]), ([0.0, 0.0, 5.0], [1, 3, 9]), ([5.0, 0.0, 0.0], [9, 3, 1])):
        assert LI.hirid_vent_windows(times, codes) == [(0.0, 0.0, True), (0.0, 5.0, False)]


def test_tied_rows_read_in_a_fixed_order(tmp_path):
    """eicu rates sharing a start and hirid fluid balance rows at one time give the same result in any row order."""
    import duckdb
    import pandas as pd
    t0 = pd.Timestamp("2100-01-01")
    stays = pd.DataFrame({"voi": [1], "visit_start_datetime": [t0]})
    drugs = pd.DataFrame({"visit_occurrence_id": 1,
                          "drug_source_value": ["Norepinephrine (mcg/min)", "Norepinephrine (mcg/kg/min)",
                                                "Norepinephrine (mcg/kg/min)", "Norepinephrine (mcg/kg/min)"],
                          "drug_exposure_start_datetime": [t0 + pd.Timedelta(minutes=m) for m in (60, 60, 60, 240)],
                          "drug_exposure_end_datetime": pd.NaT,
                          "sig": ["table=infusiondrug;rate=20;weight_kg=100", "table=infusiondrug;rate=0.3",
                                  "table=infusiondrug;rate=0.05", "table=infusiondrug;rate=0.1"],
                          "quantity": 0.0, "dose_unit_source_value": None})
    fluid = pd.DataFrame({"visit_occurrence_id": 1, "measurement_source_value": "30005110",
                          "measurement_datetime": [t0 + pd.Timedelta(minutes=m) for m in (10, 70, 70, 130)],
                          "value_as_number": [500.0, 450.0, 600.0, 700.0]})
    results = set()
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        drugs.iloc[order].to_parquet(tmp_path / "DRUG_EXPOSURE.parquet")
        fluid.iloc[order].to_parquet(tmp_path / "m.parquet")
        con = duckdb.connect()
        vaso, _ = LI.vasopressors(con, str(tmp_path), "eicu", stays, np.array([100.0]), 0, 6)
        urine, _, _ = LI.hirid_fluid_urine(con, str(tmp_path / "m.parquet"), stays, 0, 3)
        results.add((tuple(np.nan_to_num(vaso["norepi"][0], nan=-1)), tuple(np.nan_to_num(urine[0], nan=-1))))
    assert len(results) == 1, results
    (norepi, urine), = results
    assert norepi[1] == 0.2, "rates sharing a start by name, then rate: the last, 20 mcg/min at 100 kg, lasts until the next"
    assert urine[1] == 150.0, "the rows at one time by value: 450 counts 0, then 600 - 450"


def test_eicu_rate_kg_units():
    r = LI.eicu_rate_kg([10.0, 60.0, 1.0, 5.0, 2.0], ["mcg/min", "mcg/kg/hr", "mg/hr", "ml/hr", "units/min"],
                        [100.0] * 5, ml_to_mcg=32)
    assert np.isclose(r[0], 0.1)
    assert np.isclose(r[1], 1.0)
    assert np.isclose(r[2], 1.0 / 100 / 60 * 1000)
    assert np.isclose(r[3], 5.0 / 100 / 60 * 32)
    assert np.isnan(r[4])


def test_point_intervals_and_vaso60():
    ends = LI.point_intervals([0.0, 2.0, 10.0])
    assert np.allclose(ends, [2 - 1 / 60, 8 - 1 / 60, 11 - 1 / 60]), "to next row, max 6 h, last +1 h"
    rates = [(0.0, 0.5, 0.2), (2.0, 3.0, 0.05)]
    assert LI.vaso60_stay(rates, [(0.0, 0.5)]) == [], "under 1 h of administration does not count"
    out = LI.vaso60_stay(rates, [(0.0, 0.5), (0.55, 3.0)])
    assert [x[2] for x in out] == [0.2, 0.05], "ranges within 5 min merge to >= 1 h"


def test_mark_windows_max_value():
    g = LI.mark_windows([(1, 0.2, 2.5), (1, 2.1, 2.9)], [1], 0, 4, values=np.array([0.1, 0.3]))
    assert list(g[0, :3]) == [0.1, 0.1, 0.3] and np.isnan(g[0, 3])


def test_gcs_ricu_sedation_and_missing_components():
    comp = {k: np.full((1, 10), nan) for k in ("egcs", "vgcs", "mgcs", "tgcs")}
    comp["egcs"][0, 0] = 1
    comp["mgcs"][0, 2] = 3
    sed = np.zeros((1, 10), dtype=bool); sed[0, 2] = True
    g = LI.gcs_ricu(comp, sed)
    assert g[0, 0] == 1 + 5 + 6, "missing components take their maximum"
    assert g[0, 2] == 15, "sedated -> 15"
    assert np.isnan(g[0, 1]), "no row where nothing is charted"


def test_merge_ranges_as_ricu():
    assert LI.merge_ranges([(0.0, 0.5), (0.55, 0.9)]) == [[0.0, 0.9]], "within 5 min: merged, the gap subtracted again"
    assert LI.vaso60_stay([(0.0, 1.0, 0.2)], [(0.0, 0.95)]) == [], "57 min of administration is under 1 h"
    # a chain: C overlaps B but not A, so it groups with B (foverlaps' first match), as in ricu
    assert LI.merge_ranges([(0.0, 1.0), (0.9, 2.0), (1.95, 3.0)]) == [[0.0, 2.0], [0.9, 3.0]]


def test_urine_rate_per_row_as_ricu():
    """ricu's divisor on the rows of a stay per minute (`top`, `count`), from 24 h before its start."""
    import sparse
    m_lo, T = -60 * LI.URINE_RATE_MAX_GAP_H, 32
    rows = {-180: (50.0, 1), 10: (100.0, 1), 40: (60.0, 1), 65: (80.0, 1), 125: (30.0, 2), 185: (20.0, 1),
            1800: (40.0, 1)}
    coords = np.array([[0] * len(rows), [0] * len(rows), [m - m_lo for m in rows]])
    top = sparse.COO(coords, [x for x, _ in rows.values()], shape=(1, 1, 60 * T - m_lo), fill_value=np.nan)
    count = sparse.COO(coords, [n for _, n in rows.values()], shape=(1, 1, 60 * T - m_lo), fill_value=0)
    u = LI.ricu_urine_rate(top, count, m_lo, T)
    assert u[0, 0] == 60.0, "the first ICU row / (3 h since the pre-ICU row + 1) = 25; the second row of the hour / 1"
    assert u[0, 1] == 40.0, "80 / (1 h + 1)"
    assert u[0, 2] == 30.0, "two rows of one minute: the smaller takes the divisor, the larger / 1"
    assert u[0, 3] == 10.0, "20 / (1 h + 1)"
    assert u[0, 30] == 40.0, "after more than 24 h the divisor is 1"
    assert np.isnan(u[0, 4:30]).all() and u.shape == (1, T)


def test_hirid_fluid_urine_decumulates_per_row(tmp_path):
    import duckdb
    import pandas as pd
    t0 = pd.Timestamp("2100-01-01")
    rows = [(1, t0 + pd.Timedelta(minutes=m), x) for m, x in
            [(10, 100.0), (40, 150.0), (70, 120.0), (80, 3000.0), (130, 3100.0)]]
    cache = pd.DataFrame(rows, columns=["visit_occurrence_id", "measurement_datetime", "value_as_number"])
    cache["measurement_source_value"] = "30005110"
    cache.to_parquet(tmp_path / "m.parquet")
    stays = pd.DataFrame({"voi": [1], "visit_start_datetime": [t0]})
    u, first, last = LI.hirid_fluid_urine(duckdb.connect(), str(tmp_path / "m.parquet"), stays, 0, 4)
    assert u[0, 0] == 150.0, "first value, then +50"
    assert u[0, 1] == 0.0, "a drop counts 0; the 2880 mL jump is out of bounds and dropped"
    assert u[0, 2] == 100.0
    assert (first[0], last[0]) == (0.0, 2.0)
    later = stays.assign(visit_start_datetime=t0 + pd.Timedelta(hours=1))
    u, first, _ = LI.hirid_fluid_urine(duckdb.connect(), str(tmp_path / "m.parquet"), later, -1, 3)
    assert u[0, 0] == 150.0 and first[0] == -1.0, "hours count from the start in `stays`"


def test_eicu_sedation_is_a_rass_row_in_the_last_6_hours():
    """ricu's eicu GCS is 15 where a RASS of -5..-3 lies in the hour or the 6 h before it."""
    import pandas as pd
    comp = {k: np.full((1, 12), np.nan) for k in LI.GCS_CONCEPTS}
    comp["egcs"][0] = 1.0
    rass = np.full((1, 12), np.nan)
    rass[0, 2] = -4.0
    stays = pd.DataFrame({"voi": [1], "visit_start_datetime": [pd.Timestamp("2100-01-01")]})
    g = LI.gcs(None, None, "eicu", stays, comp, 0, 12, rass=rass)
    assert list(g[0]) == [12.0] * 2 + [15.0] * 7 + [12.0] * 3


def _visits_and(tmp_path, table, rows):
    """VISIT_OCCURRENCE of one person with visits [0, 10 h] and [100 d, 100 d + 10 h], plus a
    person-level table."""
    import duckdb
    import pandas as pd
    t0 = pd.Timestamp("2000-01-01")
    pd.DataFrame({"visit_occurrence_id": [1, 2], "person_id": [7, 7],
                  "visit_start_datetime": [t0, t0 + pd.Timedelta(days=100)],
                  "visit_end_datetime": [t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(days=100, hours=10)],
                  "discharged_to_concept_id": [LI.ICU_DEATH_CONCEPT_ID, LI.ICU_DEATH_CONCEPT_ID]}
                 ).to_parquet(tmp_path / "VISIT_OCCURRENCE.parquet")
    pd.DataFrame(rows).assign(**{k: [t0 + pd.Timedelta(hours=h) for h in v]
                                 for k, v in rows.items() if k.endswith("datetime")}
                              ).to_parquet(tmp_path / f"{table}.parquet")
    con = duckdb.connect()
    con.execute("CREATE TABLE _cohort_visits AS SELECT * FROM (VALUES (1), (2)) t(visit_occurrence_id)")
    return con


def test_death_belongs_to_the_nearest_visit():
    """The death lies 30 d after the second visit's end (a ward death), far from the first visit;
    both visits are flagged. Read from the stays' .obs (build_cohort._icu_death)."""
    import importlib.util

    import pandas as pd
    spec = importlib.util.spec_from_file_location("bn_death", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build_cohort.py"))
    bn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bn)
    t0 = pd.Timestamp("2000-01-01")
    obs = pd.DataFrame({"person_id": [7, 7, 8, 9],
                        "visit_start_datetime": [t0, t0 + pd.Timedelta(days=100), t0, t0],
                        "visit_end_datetime": [t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(days=100, hours=10),
                                               t0 + pd.Timedelta(hours=10), t0 + pd.Timedelta(hours=10)],
                        "discharged_to_concept_id": [LI.ICU_DEATH_CONCEPT_ID, LI.ICU_DEATH_CONCEPT_ID,
                                                     LI.ICU_DEATH_CONCEPT_ID, 0],
                        "death_datetime": [t0 + pd.Timedelta(days=130)] * 2 + [pd.NaT, t0 + pd.Timedelta(hours=5)]})
    obs["visit_occurrence_id"] = [1, 2, 3, 4]
    died, h = bn._icu_death(obs)
    assert died.tolist() == [True, True, True, False]
    assert np.isnan(h[0]), "the other flagged visit has no death time"
    assert h[1] == 24 * 30
    # a visit outside the cohort (not in obs) nearer to the death takes it
    visits = pd.concat([obs[["visit_occurrence_id", "person_id", "visit_start_datetime", "visit_end_datetime"]],
                        pd.DataFrame({"visit_occurrence_id": [5], "person_id": [7],
                                      "visit_start_datetime": [t0 + pd.Timedelta(days=129)],
                                      "visit_end_datetime": [t0 + pd.Timedelta(days=131)]})])
    assert np.isnan(bn._icu_death(obs, visits)[1][1])
    assert np.isnan(h[2]), "flagged without a DEATH row: no death time"
    assert np.isnan(h[3]), "a death on a stay not flagged as ICU death has no ICU death time"


def test_specimen_belongs_to_the_nearest_visit(tmp_path):
    """Without FACT_RELATIONSHIP, a sample belongs to its person's nearest visit: 40 d before the second visit
    (pre-ICU, nearer to it), 20 d after the first visit."""
    import pandas as pd
    con = _visits_and(tmp_path, "SPECIMEN", {"specimen_id": [1, 2, 3], "person_id": [7, 7, 7],
                                             "specimen_datetime": [24 * 60, 24 * 20, 5]})
    t0 = pd.Timestamp("2000-01-01")
    pd.DataFrame({"visit_occurrence_id": [99], "drug_source_value": ["Vancomycin"], "drug_exposure_start_datetime": [t0],
                  "drug_exposure_end_datetime": [t0], "sig": [""], "drug_type_concept_id": [0]}
                 ).to_parquet(tmp_path / "DRUG_EXPOSURE.parquet")
    stays = pd.DataFrame({"voi": [1, 2], "person_id": [7, 7], "visit_start_datetime": [t0, t0 + pd.Timedelta(days=100)]})
    rows = LI.infection_tables(con, str(tmp_path), "miiv", stays)
    assert rows["first"].tolist() == [5.0, -24 * 40] and rows["last"].tolist() == [24 * 20, -24 * 40]
    assert rows["n_abx"].tolist() == [0, 0], "the antibiotic row of another visit is not read"
    assert con.execute(f"SELECT visit_occurrence_id, count(*) FROM {LI.SPECIMEN_TABLE} GROUP BY 1 ORDER BY 1"
                       ).fetchall() == [(1, 2), (2, 1)]
