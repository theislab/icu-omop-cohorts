"""ricu's sepsis machinery (tests/ricu_labels.py, the reference of the ehrapy tests) and the antibiotic sets.
Synthetic arrays only."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import abx_sets  # noqa: E402
import ricu_labels as R  # noqa: E402

nan = np.nan


def test_pafi_matches_within_two_hours_and_defaults_fio2_to_21():
    po2 = np.array([[80.0, nan, nan, nan, nan, 90.0]])
    fio2 = np.array([[nan, nan, 40.0, nan, nan, nan]])
    p = R.pafi(po2, fio2)
    assert p[0, 0] == 100 * 80 / 21, "no fio2 at or before the po2 -> 21 %"
    assert p[0, 2] == 100 * 80 / 40, "the fio2 row takes the po2 from 2 h before"
    assert p[0, 5] == 100 * 90 / 21, "fio2 3 h earlier is outside the match window"


def test_vent_start_lasts_six_hours():
    v = np.zeros((1, 10), dtype=bool); v[0, 2] = True
    assert list(np.flatnonzero(R.vent_hours(v)[0])) == [2, 3, 4, 5, 6, 7]


def test_urine24_spans_first_to_last_row_and_needs_12_rows():
    u = np.full((1, 40), nan); u[0, 5:30] = 10.0
    u24, rows = R.urine24(u)
    assert np.isnan(u24[0, :5]).all() and np.isnan(u24[0, 30:]).all(), "no rows outside first..last"
    assert np.isnan(u24[0, 15]), "11 rows in the window: NA"
    assert u24[0, 16] == 240.0, "12 rows of 10 mL -> 120 * 24 / 12"
    assert np.isclose(u24[0, 29], 10 * 25 * 24 / 25), "window [t-24, t] holds 25 rows"


def test_sofa_score_takes_the_worst_over_the_trailing_24h_and_spans_observed_range():
    H = 60
    coag = np.full((1, H), nan); coag[0, 10] = 3.0; coag[0, 40] = 0.0
    comps = {"coag": coag, "liver": np.full((1, H), nan)}
    s = R.sofa_score(comps)
    assert np.isnan(s[0, 9]) and np.isnan(s[0, 41]), "NaN outside first..last component row"
    assert s[0, 10] == 3 and s[0, 34] == 3, "worst value held over [t-24, t]"
    assert s[0, 35] == 0, "older than 24 h"


def test_rows_follow_the_whole_timeline_when_the_grid_holds_part_of_it():
    u = np.full((1, 40), nan); u[0, 30:35] = 10.0
    u24, rows = R.urine24(u, first=np.array([-50.0]), last=np.array([80.0]))
    assert rows[0].all(), "urine before and after the grid: every grid hour is a urine24 row"
    assert u24[0, 20] == 0.0, "an hour without urine in its window counts as 0 mL"
    coag = np.full((1, 60), nan); coag[0, 40] = 3.0
    s = R.sofa_score({"coag": coag}, first=np.array([-500.0]))
    assert s[0, 0] == 0 and s[0, 40] == 3, "data before the grid: rows from the grid start, 0 without data"
    assert R.sep3_onset(s, np.array([30]))[0] == 40, "the zero rows give the rise its baseline"
    assert R.sep3_onset(R.sofa_score({"coag": coag}), np.array([30]))[0] == -1


def test_sep3_onset_can_be_before_admission_and_uses_window_cummin():
    sofa = np.full((1, 100), nan); sofa[0, 10:] = 1.0; sofa[0, 30:] = 4.0
    assert R.sep3_onset(sofa, np.array([40]))[0] == 30
    assert R.sep3_onset(sofa, np.array([-1]))[0] == -1, "no suspected infection"
    flat = np.full((1, 100), 4.0); flat[0, :3] = 0
    assert R.sep3_onset(flat, np.array([90]))[0] == -1, "baseline is the minimum within the window"


def test_abx_sets():
    assert len(abx_sets.MIIV_ABX) == 51 and len(abx_sets.HIRID_ABX) == 81
    assert any(n == "Vancocin  Amp 500 mg" for _, n in abx_sets.HIRID_ABX)
    import re
    m = lambda s: any(re.search(r, s, re.I) for r in abx_sets.EICU_ABX_REGEX)
    assert m("VANCOMYCIN HCL 1000 MG IV SOLR") and m("ZOSYN") and m("cefepime") and m("Rocephin")
    assert not m("cefTRIAXone") and not m("meropenem") and not m("LACTATED RINGERS")
