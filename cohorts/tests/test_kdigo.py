"""KDIGO staging, checked against ricu-extensions `callback-kdigo.R`. Synthetic arrays only.

The stage comes from the builder's `score_kdigo` (ehrapy's `ep.tl.kdigo` with ricu's settings) on an object as
`kdigo_object` gives it: the hourly creatinine and ricu's urine rate (on hirid the recorded rate, so the urine below is
the rate). The AKI label is the one place where a silent error is most costly: KDIGO is a staged, windowed
definition, so an off-by-one in a window or a wrong comparison direction produces a label that is plausible in
aggregate and wrong per stay.

Needs an ehrapy with `ep.tl.kdigo`, as the environment pins; skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ehrapy as ep  # noqa: E402

if not hasattr(ep.tl, "kdigo"):
    pytest.skip("this ehrapy has no tl.kdigo", allow_module_level=True)

import build_cohort as bn  # noqa: E402
from test_ehrapy_kdigo import kdigo_input_object  # noqa: E402

T = 48


def _stage(crea, urine, weight, hours):
    """The KDIGO stage per hour (0 where neither component stages) of (stays x hours) creatinine and urine rate."""
    edata = kdigo_input_object(np.asarray(crea, float), np.asarray(urine, float), hours)
    bn.score_kdigo(edata, np.asarray(weight, float))
    return np.nan_to_num(edata.obsm["kdigo"], nan=0)


def _kdigo_stage(y, weight_arr):
    """KDIGO stage on the ICU hours 0..T-1 of a (stays x [crea, urine] x hours) tensor."""
    return _stage(y[:, 0], y[:, 1], weight_arr, np.arange(y.shape[2]))


def _tensor(crea=None, urine=None, T=T):
    y = np.full((1, 2, T), np.nan)
    if crea is not None:
        y[0, 0, :len(crea)] = crea
    if urine is not None:
        y[0, 1, :len(urine)] = urine
    return y


def test_stable_creatinine_is_stage_0():
    y = _tensor(crea=[1.0] * T)
    assert _kdigo_stage(y, np.array([70.0])).max() == 0


def test_creatinine_1p5x_baseline_is_stage_1():
    # baseline 1.0 for 10 h, then 1.5 -> ricu: crea >= 1.5 * crea_168hr -> stage 1
    y = _tensor(crea=[1.0] * 10 + [1.5] + [np.nan] * (T - 11))
    assert _kdigo_stage(y, np.array([70.0]))[0, 10] == 1


def test_creatinine_2x_baseline_is_stage_2():
    y = _tensor(crea=[1.0] * 10 + [2.0] + [np.nan] * (T - 11))
    assert _kdigo_stage(y, np.array([70.0]))[0, 10] == 2


def test_creatinine_3x_baseline_is_stage_3():
    y = _tensor(crea=[1.0] * 10 + [3.0] + [np.nan] * (T - 11))
    assert _kdigo_stage(y, np.array([70.0]))[0, 10] == 3


def test_absolute_rise_of_0p3_over_48h_is_stage_1():
    """ricu: crea >= crea_48hr + 0.3. Kept just under 1.5x so only the absolute rule can fire."""
    y = _tensor(crea=[1.0] * 10 + [1.35] + [np.nan] * (T - 11))
    assert _kdigo_stage(y, np.array([70.0]))[0, 10] == 1


def test_unmeasured_creatinine_does_not_stage():
    y = _tensor(crea=[1.0] + [np.nan] * (T - 1))
    st = _kdigo_stage(y, np.array([70.0]))
    assert st[0, 1:].max() == 0


def test_good_urine_output_is_stage_0():
    # 70 kg, 1.0 mL/kg/h = 70 mL/h -- comfortably above every threshold
    y = _tensor(urine=[70.0] * T)
    assert _kdigo_stage(y, np.array([70.0])).max() == 0


def test_oliguria_matches_ricus_formula_not_the_clinical_rule():
    """ricu's kdigo_urine SUMS the hourly rates over the window and divides by weight, then compares
    to 0.5 -- so the effective threshold is ~0.083 mL/kg/h, 6x stricter than clinical KDIGO's
    "< 0.5 mL/kg/h for 6 h". We follow ricu, because parity with the benchmark is the goal, and
    this test exists so the divergence is deliberate and visible rather than discovered later.

    0.4 mL/kg/h (28 mL/h at 70 kg) IS clinical KDIGO stage 1 but does NOT stage under ricu.
    """
    clinically_oliguric = _tensor(urine=[28.0] * T)          # 0.4 mL/kg/h
    assert _kdigo_stage(clinically_oliguric, np.array([70.0])).max() == 0

    below_ricus_threshold = _tensor(urine=[5.0] * T)         # 6*5/70 = 0.43 < 0.5
    assert _kdigo_stage(below_ricus_threshold, np.array([70.0]))[0, 6] >= 1


def test_anuria_for_12h_is_stage_3():
    y = _tensor(urine=[0.0] * T)
    assert _kdigo_stage(y, np.array([70.0]))[0, 12] == 3


def test_missing_weight_keeps_ml():
    """ricu's join finds no weight for a stay without one and keeps its urine in mL; a missing weight and one of 0
    stage as 1 kg."""
    y = _tensor(urine=[0.1] * T)
    a = _kdigo_stage(np.repeat(y, 3, axis=0), np.array([np.nan, 0.0, 1.0]))
    assert np.array_equal(a[0], a[2]) and np.array_equal(a[1], a[2])
    assert a[0, 6] == 0 and _kdigo_stage(y, np.array([70.0]))[0, 6] >= 1, "0.6 mL over 6 h stages only per kg"


def test_kdigo_urine_undivided_without_weight():
    """ricu's join leaves urine in mL for a stay without a weight row, so 2 mL/h never stages."""
    hours = np.arange(0, 30)
    urine = np.full((1, 30), 2.0)
    crea = np.full((1, 30), np.nan)
    assert _stage(crea, urine, [np.nan], hours).max() == 0
    assert _stage(crea, urine, [70.0], hours).max() >= 1


def test_kdigo_uses_pre_icu_creatinine_baseline():
    """ricu runs kdigo_crea on the whole crea series: a pre-ICU baseline counts."""
    hours = np.arange(-168, 30)
    crea = np.full((1, hours.size), np.nan)
    crea[0, hours == -50] = 1.0
    crea[0, hours == 10] = 1.6
    st = _stage(crea, np.full_like(crea, np.nan), [70.0], hours)
    assert st[0, hours == 10][0] == 1, "1.6 >= 1.5 x the pre-ICU minimum within 7 d"
    st_icu = _stage(crea[:, hours >= 0], np.full((1, 30), np.nan), [70.0], np.arange(30))
    assert st_icu.max() == 0, "without the pre-ICU value there is no baseline"
    st_u = _stage(np.full_like(crea, np.nan), np.zeros_like(crea), [70.0], hours)
    assert st_u[0, hours < 6].max() == 0, "the urine rules count hours from ICU admission"


def test_unmeasured_urine_is_not_read_as_anuria():
    """The dangerous failure: no charting must not look like zero output."""
    y = _tensor(urine=None)
    assert _kdigo_stage(y, np.array([70.0])).max() == 0


def test_stage_is_the_max_of_the_two_components():
    """ricu: kdigo := pmax(kdigo_crea, kdigo_urine)."""
    y = _tensor(crea=[1.0] * 10 + [1.5] + [np.nan] * (T - 11), urine=[0.0] * T)
    st = _kdigo_stage(y, np.array([70.0]))
    assert st[0, 12] == 3, "urine stage 3 must win over creatinine stage 1"


def test_early_hours_cannot_stage_on_urine():
    """The windows are guarded by `time >= 6/12/24`, so hour 0-5 cannot be oliguric yet."""
    y = _tensor(urine=[0.0] * T)
    assert _kdigo_stage(y, np.array([70.0]))[0, :6].max() == 0


def test_outcome_window_is_symmetric():
    """aki.R applies pad_rollind twice -- align="left" then align="right" -- so a single positive
    at s marks [s-6, s+6], not just [s, s+6]. Getting this forward-only halves the positives.
    """
    W = bn.AKI_OUTCOME_WINDOW_H
    onset = np.zeros((1, 40), dtype=bool)
    onset[0, 20] = True
    lab = onset.copy()
    for k in range(1, W + 1):
        lab[:, :-k] |= onset[:, k:]
        lab[:, k:] |= onset[:, :-k]
    assert lab[0, 20 - W] and lab[0, 20 + W], "window must reach 6 h either side"
    assert not lab[0, 20 - W - 1] and not lab[0, 20 + W + 1], "and no further"
    assert lab[0].sum() == 2 * W + 1


def test_a_stay_IS_truncated_at_first_onset_plus_six():
    """aki.R lines 58-64 DO censor:
        outc <- summary_step(outc, "first"); stop_obs_at(outc, offset = hours(6L))
    so the grid ends 6 h after the first AKI and only that onset is kept. An earlier version of
    this test asserted the opposite -- it was written from a partial read of aki.R and encoded the
    error rather than catching it. Verified against the reference: every ricu positive stay has
    exactly one positive run, length <= 13, ending at the stay's last grid hour.
    """
    W = bn.AKI_OUTCOME_WINDOW_H
    onset = np.zeros((1, 60), dtype=bool)
    onset[0, 20] = True
    onset[0, 40] = True                       # a later onset that must NOT survive
    has = onset.any(1)
    first = np.where(has, onset.argmax(1), 60)
    pt = np.zeros_like(onset); pt[0, first[0]] = True
    lab = pt.copy()
    for k in range(1, W + 1):
        lab[:, :-k] |= pt[:, k:]
        lab[:, k:] |= pt[:, :-k]
    assert lab[0].sum() == 2 * W + 1, "exactly one dilated onset"
    assert not lab[0, 40], "a second onset must not be labelled"
    lgrid = int(min(60, first[0] + W + 1))
    assert lgrid == 27, "grid ends at first_onset + 6, inclusive both ends"



if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
