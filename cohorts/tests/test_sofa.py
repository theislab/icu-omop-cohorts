"""SOFA components against ricu 0.6.0's actual thresholds. Synthetic arrays only.

Run:  python tests/test_sofa.py

The bounds are transcribed from closures extracted with lazyLoad, not from a textbook -- these
tests are what pin that transcription.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sofa as S  # noqa: E402

A = lambda *v: np.array(v, dtype=float)


def test_cns_matches_ricu_gcs_bounds():
    # 4 - findInterval(gcs, c(6,10,13,15))
    assert list(S.sofa_cns(A(3, 6, 10, 13, 15))) == [4, 3, 2, 1, 0]


def test_coag_matches_ricu_platelet_bounds():
    # 4 - findInterval(plt, c(20,50,100,150))
    assert list(S.sofa_coag(A(10, 20, 50, 100, 150, 400))) == [4, 3, 2, 1, 0, 0]


def test_liver_matches_ricu_bilirubin_bounds():
    # findInterval(bili, c(1.2,2,6,12))
    assert list(S.sofa_liver(A(0.5, 1.2, 2, 6, 12, 30))) == [0, 1, 2, 3, 4, 4]


def test_renal_uses_creatinine_and_urine():
    assert list(S.sofa_renal(A(0.5, 1.2, 2.0, 3.5, 5.0), A(1000, 1000, 1000, 1000, 1000))) == [0, 1, 2, 3, 4]
    # urine can drive the score on its own
    assert list(S.sofa_renal(A(0.5, 0.5), A(400, 150))) == [3, 4]


def test_resp_floors_pafi_at_200_when_not_ventilated():
    """The component's whole point: a low P/F ratio only counts as severe if it is supported.

    NB fio2 is in PERCENT, as ricu declares it (unit "%", min 21, max 100) and as our CDM stores
    it. The earlier version of this test passed fio2=1.0 (a fraction) and therefore PASSED against
    a 100x-wrong pafi -- the test encoded the bug it should have caught.
    """
    pao2, fio2 = A(60, 60), A(100.0, 100.0)        # pafi = 100*60/100 = 60 -> would be 4
    vent = np.array([True, False])
    assert list(S.sofa_resp(pao2, fio2, vent)) == [4, 2], "unventilated must be capped at 2"


def test_resp_bounds():
    # fio2 = 100% -> pafi == pao2, so these map straight onto the c(400,300,200,100) bounds
    pao2 = A(90, 150, 250, 350, 500); fio2 = np.full(5, 100.0)
    assert list(S.sofa_resp(pao2, fio2, np.ones(5, dtype=bool))) == [4, 3, 2, 1, 0]


def test_resp_uses_percent_fio2_not_a_fraction():
    """Guard the 100x. A realistic blood gas on 40% O2 is pafi 250 -> score 2, not score 4."""
    assert S.sofa_resp(A(100.0), A(40.0), np.array([True]))[0] == 2


def test_cardio_map_only():
    assert list(S.sofa_cardio(A(80, 65))) == [0, 1]


def test_cardio_vasopressors_outrank_map():
    map_ = A(80, 80, 80, 80)
    norepi = A(0, 0.05, 0.2, 0)
    dopa = A(0, 0, 0, 20)
    assert list(S.sofa_cardio(map_, dopa=dopa, norepi=norepi)) == [0, 3, 4, 4]


def test_missing_input_scores_zero_not_the_worst_grade():
    """The sharp edge of the reverse-coded components: `sofa_cns` is 4 - findInterval(gcs, ...),
    so an unmeasured GCS naively scores 4 and an unmonitored patient looks like total neurological
    failure. Caught by this test on first run.
    """
    nan = A(np.nan, np.nan)
    for fn in (S.sofa_cns, S.sofa_coag, S.sofa_liver):
        assert list(fn(nan)) == [0, 0], fn.__name__
    assert list(S.sofa_resp(nan, nan, np.zeros(2, dtype=bool))) == [0, 0]


def test_sep3_requires_both_a_sofa_rise_and_suspected_infection():
    T = 60
    sofa = np.zeros((2, T), dtype=np.int16)
    sofa[:, 30:] = 3                              # a 3-point rise at t=30 in both stays
    susp = np.zeros((2, T), dtype=bool)
    susp[0, 30] = True                            # only stay 0 has suspected infection
    out = S.sep3(sofa, susp)
    assert out[0].any(), "rise + infection must flag"
    assert not out[1].any(), "a SOFA rise without suspected infection is not sepsis"


def test_sep3_ignores_a_rise_below_threshold():
    T = 60
    sofa = np.zeros((1, T), dtype=np.int16); sofa[:, 30:] = 1     # only +1
    susp = np.zeros((1, T), dtype=bool); susp[0, 30] = True
    assert not S.sep3(sofa, susp).any()


def test_carry_forward_holds_the_last_value():
    x = np.array([[1.0, np.nan, np.nan, 4.0, np.nan]])
    assert list(S.carry_forward(x)[0]) == [1.0, 1.0, 1.0, 4.0, 4.0]


def test_carry_forward_respects_max_gap():
    x = np.array([[1.0] + [np.nan] * 5])
    got = S.carry_forward(x, max_gap=2)[0]
    assert list(got[:3]) == [1.0, 1.0, 1.0]
    assert np.isnan(got[3:]).all(), "a value older than max_gap must expire"


def test_carry_forward_leaves_leading_nans():
    x = np.array([[np.nan, np.nan, 3.0]])
    got = S.carry_forward(x)[0]
    assert np.isnan(got[0]) and np.isnan(got[1]) and got[2] == 3.0


def test_carry_forward_must_run_on_the_INPUTS_not_the_scores():
    """The subtlety that cost two rebuilds: carrying forward a SOFA *score* array does nothing,
    because `_score` has already turned missing into 0. Missingness has to be preserved as NaN and
    carried at the INPUT level -- which is what the builder now does (cf("plt") etc).
    """
    T = 30
    # a score array: gaps are already 0, so there is nothing to carry
    scores = np.zeros((1, T)); scores[0, ::6] = 3
    assert np.array_equal(S.carry_forward(scores), scores), "carrying scores is a no-op"

    # an INPUT array: gaps are NaN, so carrying holds the clinical value
    plt = np.full((1, T), np.nan); plt[0, ::6] = 30.0        # thrombocytopenia, measured 6-hourly
    held = S.carry_forward(plt, max_gap=24)
    assert not np.isnan(held[0, 1:6]).any(), "the platelet value must persist between draws"
    raw_score = S.sofa_coag(plt)
    held_score = S.sofa_coag(held)
    assert (raw_score > 0).sum() < (held_score > 0).sum(), \
        "scoring carried-forward inputs gives a STABLE component instead of an oscillating one"


def test_sep3_flags_at_most_one_hour_per_stay():
    """ricu: res[, head(.SD, n = 1L), by = id]. One onset per stay, not a run."""
    T = 168
    sofa = np.zeros((1, T), dtype=np.int16); sofa[:, 40:] = 3
    susp = np.zeros((1, T), dtype=bool); susp[0, 30] = True
    assert S.sep3(sofa, susp).sum() == 1


def test_sep3_does_not_flag_before_the_rise_happened():
    """The original bug: a rise at h40 flagged hours 16-20, i.e. ~24 h BEFORE it occurred."""
    T = 168
    sofa = np.zeros((1, T), dtype=np.int16); sofa[:, 40:] = 3
    susp = np.zeros((1, T), dtype=bool); susp[0, 20] = True
    out = S.sep3(sofa, susp)
    if out.any():
        assert np.flatnonzero(out[0])[0] >= 40, "cannot flag before the SOFA rise"


def test_sep3_uses_only_the_first_suspected_infection():
    """si_window = "first". A later infection must not pair with an unrelated later rise."""
    T = 168
    sofa = np.zeros((1, T), dtype=np.int16); sofa[:, 120:] = 4     # rise far from the first abx
    susp = np.zeros((1, T), dtype=bool); susp[0, 5] = True; susp[0, 118] = True
    assert not S.sep3(sofa, susp).any(), "only the FIRST infection window counts"


def test_sep3_baseline_is_scoped_to_the_si_window():
    """ricu evaluates delta_cummin inside the non-equi join (by = .EACHI), so the baseline is the
    cumulative minimum WITHIN [t_si - 48, t_si + 24] -- not from ICU admission. A stay that was
    sick before the window opened must not count its pre-window low as the baseline.
    """
    T = 168
    sofa = np.full((1, T), 5, dtype=np.int16)
    sofa[:, :3] = 0                       # low only at the very start, far before the window
    susp = np.zeros((1, T), dtype=bool); susp[0, 120] = True   # window is [72, 144], all SOFA 5
    assert not S.sep3(sofa, susp).any(), "a flat window has no rise, whatever happened at hour 0"

    rises = np.full((1, T), 5, dtype=np.int16)
    rises[:, 100:] = 9                    # a 4-point rise inside the window
    assert S.sep3(rises, susp).any(), "a rise inside the window must fire"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
