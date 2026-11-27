"""ricu's sepsis-3 machinery on hourly grids that may start before ICU admission, the reference of
`test_ehrapy_labels.py` for `ep.tl.sofa` and `ep.tl.sepsis3` (the builder's until step 4b).

Every grid here is (n_stays, H) with column j = hour `h0 + j` from ICU admission, NaN where ricu
has no row. ricu sources: `ricu-src/R/callback-sofa.R` (sofa_score, components),
`callback-cncpt.R` (pafi/match_fio2, urine24, vent_ind), `callback-sep3.R` (sep3, delta_cummin).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sofa as S

SOFA_WIN_H = 24          # sofa_score win_length; slide(before = 24 h) covers [t - 24, t]
PAFI_MATCH_H = 2         # pafi match_win
FIO2_DEFAULT = 21.0      # match_fio2: missing fio2 -> 21
VENT_START_H = 6         # vent_ind: a start without an end lasts match_win = 6 h
URINE24_MIN_STEPS = 12   # urine24 min_win


def locf(x, max_gap):
    """Last observation carried forward for at most `max_gap` steps."""
    return S.carry_forward(x, max_gap=max_gap)


def _rolling(x, window, how):
    """Trailing window of `window` rows, inclusive of t, NaN-aware, along axis 1."""
    df = pd.DataFrame(x.T)
    r = df.rolling(window=window, min_periods=1)
    out = {"max": r.max, "min": r.min, "sum": r.sum, "count": r.count}[how]()
    return out.to_numpy().T


def pafi(po2, fio2, match_h=PAFI_MATCH_H, default=FIO2_DEFAULT):
    """ricu `pafi` with `match_vals`: each po2 row takes the latest fio2 at most `match_h` before
    it (21 if none), each fio2 row the latest po2 at most `match_h` before it (dropped if none);
    pafi = 100 * po2 / fio2. Inputs are the hourly po2 min and fio2 max. Per hour the lowest
    (worst) pafi is kept."""
    f_for_p = locf(fio2, match_h)
    f_for_p = np.where(np.isnan(f_for_p), default, f_for_p)
    p_for_f = locf(po2, match_h)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(~np.isnan(po2) & (f_for_p != 0), 100.0 * po2 / f_for_p, np.nan)
        b = np.where(~np.isnan(fio2) & ~np.isnan(p_for_f) & (fio2 != 0), 100.0 * p_for_f / fio2, np.nan)
    return np.fmin(a, b)


def vent_hours(charted, start_h=VENT_START_H):
    """Hours covered by a ventilation start that has no end: [t, t + 6 h) after each charted row."""
    x = np.where(charted, 1.0, np.nan)
    return ~np.isnan(locf(x, start_h - 1))


def _row_range(seen, first=None, last=None):
    """Per stay the grid indices of the first and last row. `first` / `last` (grid indices, may
    lie outside the grid, NaN if none) give the range over all times when the grid holds only
    part of it; the grid's own rows widen it."""
    n, H = seen.shape
    any_ = seen.any(1)
    lo = np.where(any_, seen.argmax(1), H).astype(float)
    hi = np.where(any_, H - 1 - seen[:, ::-1].argmax(1), -1).astype(float)
    if first is not None:
        lo = np.fmin(lo, np.maximum(np.asarray(first, float), 0))
    if last is not None:
        hi = np.fmax(hi, np.minimum(np.asarray(last, float), H - 1))
    return lo, hi


def urine24(urine, min_steps=URINE24_MIN_STEPS, win=SOFA_WIN_H, first=None, last=None):
    """ricu `urine24`: rows from each stay's first to last urine row (fill_gaps), then over the
    window [t - 24, t] within those rows: NA if fewer than `min_steps` rows, else
    sum * 24 / rows. NaN outside the first..last range. `first` / `last`: that range over all
    times, as grid indices, when the grid holds only part of the stay's rows."""
    n, H = urine.shape
    first, last = _row_range(~np.isnan(urine), first, last)
    j = np.arange(H)[None, :]
    rows = (j >= first[:, None]) & (j <= last[:, None])
    total = _rolling(np.where(rows, np.nan_to_num(urine), np.nan), win + 1, "sum")
    cnt = _rolling(np.where(rows, 1.0, np.nan), win + 1, "count")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt >= min_steps, total * 24.0 / cnt, np.nan)
    return np.where(rows, out, np.nan), rows


def _component(score, exists):
    """A component score as ricu tables it: a value where the component has a row, else NaN."""
    return np.where(exists, score.astype(float), np.nan)


def sofa_components(*, po2, fio2, vent, plt, bili, map_, vaso, gcs, crea, urine, urine_range=(None, None)):
    """The six SOFA components on the hourly grid, NaN where ricu has no component row.

    Inputs use ricu's per-concept hourly aggregation: po2 min, fio2 max, plt min, bili max,
    map min, crea max, gcs min, urine sum, pressors max. `vent` is boolean, `vaso` a dict of
    rate-like arrays (0 where not running). `urine_range`: the first and last urine row over
    all times, as grid indices (urine24)."""
    pf = pafi(po2, fio2)
    resp = _component(_sofa_resp(pf, vent), ~np.isnan(pf) | vent)
    coag = _component(S.sofa_coag(plt), ~np.isnan(plt))
    liver = _component(S.sofa_liver(bili), ~np.isnan(bili))
    running = np.zeros(map_.shape, dtype=bool)
    for v in vaso.values():
        running |= np.nan_to_num(v) > 0
    cardio = _component(S.sofa_cardio(map_, dopa=vaso.get("dopa"), norepi=vaso.get("norepi"),
                                      dobu=vaso.get("dobu"), epi=vaso.get("epi")),
                        ~np.isnan(map_) | running)
    cns = _component(S.sofa_cns(gcs), ~np.isnan(gcs))
    u24, urows = urine24(urine, first=urine_range[0], last=urine_range[1])
    renal = _component(S.sofa_renal(crea, u24), ~np.isnan(crea) | urows)
    return {"resp": resp, "coag": coag, "liver": liver, "cardio": cardio, "cns": cns, "renal": renal}


def _sofa_resp(pf, vent):
    """sofa_resp on an already matched pafi: floored at 200 when not ventilated."""
    with np.errstate(invalid="ignore"):
        x = np.where(np.isfinite(pf) & (pf < 200) & ~vent, 200.0, pf)
    return S._score(4 - S._find_interval(x, S.SOFA_RESP_BOUNDS))


def sofa_score(components, win=SOFA_WIN_H, first=None):
    """ricu `sofa_score`: rows from the first to the last hour with any component row
    (fill_gaps), per component the worst (max) over [t - 24, t], NA counted as 0 in the sum.
    NaN outside that range. `first`: the first component row over all times, as a grid index;
    a stay with data before the grid has rows (score 0 without data in the window) from the
    grid start."""
    exists = np.zeros(next(iter(components.values())).shape, dtype=bool)
    for c in components.values():
        exists |= ~np.isnan(c)
    n, H = exists.shape
    first, last = _row_range(exists, first)
    j = np.arange(H)[None, :]
    rows = (j >= first[:, None]) & (j <= last[:, None])
    total = np.zeros((n, H))
    for c in components.values():
        total += np.nan_to_num(_rolling(c, win + 1, "max"))
    return np.where(rows, total, np.nan)


def sep3_onset(sofa, si_idx, lwr=48, upr=24, thresh=2, has_si=None):
    """ricu `sep3` with si_window "first" and delta_cummin. `sofa` is NaN where no row exists,
    `si_idx` the grid index of each stay's first suspected infection, which may lie outside the
    grid (the window is clipped to it); `has_si` marks stays with one (default: si_idx >= 0).
    Returns the onset grid index, -1 if none."""
    n, H = sofa.shape
    si_idx = np.asarray(si_idx)
    has_si = (si_idx >= 0) if has_si is None else np.asarray(has_si, dtype=bool)
    out = np.full(n, -1, dtype=np.int64)
    for i in np.flatnonzero(has_si):
        lo, hi = max(0, si_idx[i] - lwr), min(H - 1, si_idx[i] + upr)
        if lo > hi:
            continue
        seg = sofa[i, lo:hi + 1]
        cm = np.minimum.accumulate(np.where(np.isnan(seg), np.inf, seg))
        with np.errstate(invalid="ignore"):
            hits = np.flatnonzero((seg - cm) >= thresh)
        if hits.size:
            out[i] = lo + hits[0]
    return out
