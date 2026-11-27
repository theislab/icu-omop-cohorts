"""ricu's SOFA component scores and Sepsis-3 rule. Both are the references of `test_ehrapy_labels.py` for
`ep.tl.sofa` and `ep.tl.sepsis3`, which the builder uses; they were the builder's until step 4b.

The thresholds are ricu's, checked against the installed package rather than a textbook. Three of the six
components count how many of their bounds a value reaches: the GCS (6, 10, 13, 15) and the platelets
(20, 50, 100, 150) score 4 minus that count, the bilirubin (1.2, 2, 6, 12) scores the count itself.
"""

from __future__ import annotations

import numpy as np

# ricu: PaO2/FiO2 below 100 scores 4, below 200 3, below 300 2, below 400 1, else 0, after the vent correction
SOFA_RESP_BOUNDS = (400.0, 300.0, 200.0, 100.0)
SOFA_CNS_BOUNDS = (6.0, 10.0, 13.0, 15.0)
SOFA_COAG_BOUNDS = (20.0, 50.0, 100.0, 150.0)
SOFA_LIVER_BOUNDS = (1.2, 2.0, 6.0, 12.0)
SEP3_SOFA_RISE = 2          # Sepsis-3: an increase of >= 2 SOFA points
SEP3_WINDOW_H = (48, 24)    # ricu sep3 default: si_window looks back 48 h / forward 24 h


def _find_interval(x, bounds):
    """R's findInterval: number of bounds <= x. NaN in -> NaN out (float), as in R."""
    out = np.zeros(x.shape, dtype=np.float64)
    for b in bounds:
        with np.errstate(invalid="ignore"):
            out += (x >= b).astype(np.float64)
    return np.where(np.isnan(x), np.nan, out)


def _score(raw):
    """Missing input scores 0, NOT the worst grade.

    This is the sharp edge of the reverse-coded components. `sofa_cns` is `4 - findInterval(gcs, ...)`,
    so an unmeasured GCS naively yields 4 -- the maximum -- and an unmonitored patient would look
    like total neurological failure. R returns NA here and the NA propagates; we score 0 and let the
    component simply not contribute, which is the same decision already taken for KDIGO staging.
    """
    return np.where(np.isnan(raw), 0, raw).astype(np.int8)


def sofa_resp(pao2, fio2, ventilated):
    """pafi = pao2/fio2; ricu floors it at 200 when the patient is NOT ventilated.

    That floor is the whole point of the component: a low P/F ratio only counts as severe organ
    failure if it is being supported, otherwise the score is capped at 2.
    """
    # ricu: `pafi := 100 * po2 / fio2`, because its fio2 concept is declared in PERCENT
    # (unit "%", min 21, max 100) -- verified in concept-dict.json and by deparsing pafi().
    # Our CDM stores fio2 the same way (concept 3020716: p05=30, p50=40, p95=100), and so does
    # ricu's own miiv dyn.parquet. Without the 100x, pafi is ~2 instead of ~222, every blood gas
    # scores 2 or 4, and sofa_resp becomes a two-valued switch.
    with np.errstate(invalid="ignore", divide="ignore"):
        pafi = np.where((fio2 > 0), 100.0 * pao2 / fio2, np.nan)
        pafi = np.where(np.isfinite(pafi) & (pafi < 200) & ~ventilated, 200.0, pafi)
    return _score(4 - _find_interval(pafi, SOFA_RESP_BOUNDS))


def sofa_cns(gcs):
    return _score(4 - _find_interval(gcs, SOFA_CNS_BOUNDS))


def sofa_coag(plt):
    return _score(4 - _find_interval(plt, SOFA_COAG_BOUNDS))


def sofa_liver(bili):
    return _score(_find_interval(bili, SOFA_LIVER_BOUNDS))


def sofa_cardio(map_, dopa=None, norepi=None, dobu=None, epi=None):
    """ricu scores vasopressor dose first and falls back to MAP < 70 only if none is running."""
    z = np.zeros_like(map_, dtype=np.float64)
    dopa, norepi, dobu, epi = (z if v is None else np.nan_to_num(v) for v in (dopa, norepi, dobu, epi))
    eps = np.sqrt(np.finfo(float).eps)
    out = np.zeros(map_.shape, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out = np.where(map_ < 70, 1, out)
        out = np.where((dopa > eps) & (dopa <= 5 + eps) | (dobu > eps), 2, out)
        out = np.where((dopa > 5 + eps) | ((epi > eps) & (epi <= 0.1 + eps))
                       | ((norepi > eps) & (norepi <= 0.1 + eps)), 3, out)
        out = np.where((dopa > 15 + eps) | (epi > 0.1 + eps) | (norepi > 0.1 + eps), 4, out)
    return out.astype(np.int8)


def sofa_renal(crea, urine24):
    """ricu: crea >= 5 OR urine24 < 200 -> 4; (3.5<=crea<5) OR urine24 < 500 -> 3; ..."""
    out = np.zeros(crea.shape, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        out = np.where((crea >= 1.2) & (crea < 2), 1, out)
        out = np.where((crea >= 2) & (crea < 3.5), 2, out)
        out = np.where(((crea >= 3.5) & (crea < 5)) | (urine24 < 500), 3, out)
        out = np.where((crea >= 5) | (urine24 < 200), 4, out)
    return out.astype(np.int8)


def carry_forward(x, max_gap=None):
    """Last-observation-carried-forward along time, which is what ricu's SOFA operates on.

    THIS IS LOAD-BEARING, not tidying. ricu computes each SOFA component from a concept series that
    has been carried forward onto the grid, so an hour with nothing newly charted keeps the previous
    value. Scoring straight off the raw hourly tensor instead means every unmeasured hour scores 0,
    `cummin` collapses to 0 within a few hours, and `delta_cummin` degenerates to the raw SOFA --
    so every moderately sick hour reads as a fresh 2-point rise. Measured effect of getting this
    wrong: the builder's sepsis fired in 42% of miiv stays against ricu's 5.2%.
    """
    x = np.asarray(x, dtype=np.float64)
    n, T = x.shape
    idx = np.where(~np.isnan(x), np.arange(T)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.where(idx >= 0, np.take_along_axis(x, np.maximum(idx, 0), axis=1), np.nan)
    if max_gap is not None:
        age = np.arange(T)[None, :] - idx
        out = np.where((idx >= 0) & (age <= max_gap), out, np.nan)
    return out


def sofa_total(**components):
    """Sum of the six components. ricu carries each forward before summing."""
    return sum(np.asarray(v, dtype=np.int16) for v in components.values())


def sep3(sofa, susp_inf, si_lwr=48, si_upr=24, sofa_thresh=SEP3_SOFA_RISE):
    """Sepsis-3, matching ricu's `sep3` defaults. Three things my first version got wrong:

      1. `si_window = "first"` (the DEFAULT) -- only the FIRST suspected-infection time per stay is
         considered, not every one. Using all of them let any later infection pair with any earlier
         deterioration.
      2. `delta_fun = delta_cummin` -- `x - cummin(x)`, evaluated INSIDE the non-equi join, so the
         baseline is the cumulative minimum WITHIN the SI window, not from ICU admission.
      3. `res[, head(.SD, n = 1L), by = id]` -- only the FIRST qualifying hour is kept, so a stay
         contributes one onset, not a run of them.

    The window is asymmetric around that first infection time: [t - si_lwr, t + si_upr], i.e. 48 h
    before to 24 h after, and the qualifying SOFA rise must fall INSIDE it.
    """
    n, T = sofa.shape
    out = np.zeros((n, T), dtype=bool)
    for i in range(n):
        si = np.flatnonzero(susp_inf[i])
        if si.size == 0:
            continue
        t0 = si[0]                                    # si_window = "first"
        lo, hi = max(0, t0 - si_lwr), min(T - 1, t0 + si_upr)
        # delta_cummin is evaluated INSIDE ricu's non-equi join (`by = .EACHI`), so the baseline
        # is the cumulative minimum WITHIN the SI window -- not from ICU admission. Reproduced in
        # data.table to confirm: a stay at SOFA 0 then flat 3 gives delta 0 throughout a window
        # starting at t=5 under ricu, but delta 3 under a global cummin.
        seg = sofa[i, lo:hi + 1]
        delta = seg - np.minimum.accumulate(seg)
        hits = np.flatnonzero(delta >= sofa_thresh)
        if hits.size:
            out[i, lo + hits[0]] = True               # head(.SD, n = 1L)
    return out
