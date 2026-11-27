"""The builder's KDIGO stage (`score_kdigo`, ehrapy's `ep.tl.kdigo` with ricu's settings) against the builder's former
`_kdigo_stage_arrays`, kept here as the reference, cell by cell, on random hourly grids that start before the ICU
admission, with creatinine and urine at and around every threshold. Synthetic arrays only.

The object holds the creatinine as measured, which `label_view` rounds as the reference's input was rounded, and ricu's
urine rate. The reference sets a component to 0 where ehrapy leaves it missing (no creatinine or no urine at the hour),
so ehrapy's NaN counts as 0 in the comparison; this is exact, every stage 0 to 3 is compared.

Needs an ehrapy with `ep.tl.kdigo`, as the environment pins; skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ehrapy as ep  # noqa: E402

if not hasattr(ep.tl, "kdigo"):
    pytest.skip("this ehrapy has no tl.kdigo", allow_module_level=True)

import build_cohort as bn  # noqa: E402
import ehrdata as ed  # noqa: E402
import ehrdata_dynamics as ED  # noqa: E402
import sparse  # noqa: E402


def _roll(arr, window, how):
    """(n_obs, T) -> NaN-aware rolling min/sum over the trailing `window` hours, inclusive of t."""
    r = pd.DataFrame(arr.T).rolling(window=window, min_periods=1)
    return (r.min() if how == "min" else r.sum()).to_numpy().T


def _kdigo_stage_arrays(crea, rate, weight_arr, t_hours):
    """The builder's KDIGO stage until step 4c: stage, creatinine and urine components on a grid whose column j is hour
    `t_hours[j]` from ICU admission, from the hourly creatinine and ricu's urine rate.

    The creatinine stage (ricu's `kdigo_crea`, windows (t - 48 h, t] and (t - 7 d, t]):
      stage 3: at least 3 x the 7-day minimum, or at least 4 mg/dL with a stage-1 rise;
      stage 2: at least 2 x the 7-day minimum;
      stage 1: at least 0.3 mg/dL above the 48 h minimum, or at least 1.5 x the 7-day minimum.
    The urine stage (ricu's `kdigo_urine`): below 0.5 mL/kg over 6 h (1), 0.5 mL/kg over 12 h (2), 0.3 mL/kg over
    24 h or none over 12 h (3), as window sums of the rate per kg, a stay without weight in mL, only at hours with a
    urine rate."""
    crea_48 = _roll(crea, 48, "min")
    crea_168 = _roll(crea, 168, "min")
    with np.errstate(invalid="ignore"):
        kc = np.select(
            [crea >= 3 * crea_168,
             (crea >= 4) & ((crea - crea_48 >= 0.3) | (crea >= 1.5 * crea_168)),
             crea >= 2 * crea_168,
             crea >= crea_48 + 0.3,
             crea >= 1.5 * crea_168],
            [3, 3, 2, 1, 1], default=0).astype(np.int8)
    kc[np.isnan(crea)] = 0
    w = np.where(np.isfinite(weight_arr) & (weight_arr > 0), weight_arr, 1.0)[:, None]
    u6, u12, u24 = (_roll(rate, h, "sum") / w for h in (6, 12, 24))
    t_idx = np.asarray(t_hours)[None, :]
    with np.errstate(invalid="ignore"):
        ku = np.select(
            [(t_idx >= 24) & (u24 < 0.3),
             (t_idx >= 12) & (u12 == 0),
             (t_idx >= 12) & (u12 < 0.5),
             (t_idx >= 6) & (u6 < 0.5)],
            [3, 3, 2, 1], default=0).astype(np.int8)
    ku[np.isnan(rate)] = 0
    return np.maximum(kc, ku), kc, ku


def _hourly_rate(urine, ds, max_gap=24):
    """ricu's urine rate applied to hourly volumes (the gap to the previous hour plus 1), to make rates at the
    thresholds; hirid records the rate."""
    if ds == "hirid":
        return urine
    out = np.full_like(urine, np.nan)
    for i in range(urine.shape[0]):
        hrs = np.flatnonzero(~np.isnan(urine[i]))
        tm = np.diff(hrs, prepend=np.nan) + 1.0
        out[i, hrs] = urine[i, hrs] / np.where(np.isnan(tm) | (tm > max_gap), 1.0, tm)
    return out


def _creatinine(rng, n, H):
    """Sparse creatinine around a per-stay level that drifts: the level, a rise of 0.3, 1.5, 2 and 3 times it and
    4 mg/dL, each also just below and above, and uniform values."""
    levels = np.array([0.4, 0.5, 0.7, 0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 2.7, 3.0, 3.7, 3.8])
    crea = np.full((n, H), np.nan)
    for i in range(n):
        density = rng.choice([0.03, 0.1, 0.3])
        level = rng.choice(levels)
        for j in np.flatnonzero(rng.random(H) < density):
            if rng.random() < 0.15:
                level = rng.choice(levels)
            target = rng.choice([level, level + 0.3, 1.5 * level, 2 * level, 3 * level, 4.0, rng.uniform(0.3, 8)])
            crea[i, j] = target + rng.choice([0, 0, 0, -0.01, 0.01])
    crea[rng.random(n) < 0.05] = np.nan
    return crea


def _urine(rng, n, H):
    """Sparse hourly urine volumes with zeros and runs of zeros, and values that ricu's rate makes 0.3 and 0.5 mL/kg/h
    at 100 kg."""
    values = np.array([0, 0, 0, 1, 5, 10, 20, 30, 37.5, 50, 60, 100, 150, 300])
    urine = np.full((n, H), np.nan)
    for i in range(n):
        density = rng.choice([0.05, 0.3, 0.7, 1.0])
        seen = rng.random(H) < density
        urine[i, seen] = rng.choice(values, size=int(seen.sum()))
        for _ in range(rng.integers(0, 3)):
            a = rng.integers(0, H)
            urine[i, a : a + rng.integers(6, 30)] = 0.0
    urine[rng.random(n) < 0.1] = np.nan
    return urine


def kdigo_input_object(crea, rate, t_hours):
    """An EHRData as `kdigo_object` gives it: `crea` and `urine_rate` in layer `values` on the hours `t_hours` of
    ehrdata's OMOP setup."""
    n = crea.shape[0]
    return ed.EHRData(
        obs=pd.DataFrame(index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=["crea", "urine_rate"]),
        tem=pd.DataFrame({"interval_step": np.asarray(t_hours, dtype=np.int64)}),
        uns={"omop_io_variable_setup": {"interval_length_number": 1, "interval_length_unit": "h"}},
        layers={ED.LAYER: sparse.COO.from_numpy(np.stack([crea, rate], axis=1), fill_value=np.nan)},
    )


@pytest.mark.parametrize(("seed", "h0", "ds"), [(0, -48, "miiv"), (1, -7, "miiv"), (2, -150, "miiv"), (3, -30, "hirid")])
def test_kdigo_equals_the_former_builder(seed, h0, ds):
    rng = np.random.default_rng(seed)
    n, T = 400, 200
    t_hours = np.arange(h0, T)
    H = len(t_hours)
    crea, rate = _creatinine(rng, n, H), _hourly_rate(_urine(rng, n, H), ds)
    weight = rng.choice([50.0, 70.5, 80.0, 100.0, 100.0, 120.0], size=n)
    weight[rng.random(n) < 0.15] = np.nan
    weight[rng.random(n) < 0.03] = 0.0

    stage, kc, ku = _kdigo_stage_arrays(np.round(crea, ED.VALUE_DECIMALS), rate, weight, t_hours)
    edata = kdigo_input_object(crea, rate, t_hours)
    bn.score_kdigo(edata, weight)
    for name, ours in (("kdigo", stage), ("kdigo_creatinine", kc), ("kdigo_urine", ku)):
        differing = int((ours != np.nan_to_num(edata.obsm[name], nan=0)).sum())
        assert differing == 0, f"{name}: {differing} of {ours.size} cells differ"
    # ehrapy leaves a component missing exactly where its input is
    np.testing.assert_array_equal(np.isnan(edata.obsm["kdigo_creatinine"]), np.isnan(crea))
    np.testing.assert_array_equal(np.isnan(edata.obsm["kdigo_urine"]), np.isnan(rate))
    # the first hour at stage 1 or more, as the former builder took it
    for key, ours in (("aki", stage), ("aki_creatinine", kc), ("aki_urine", ku)):
        first = np.where((ours >= 1).any(1), t_hours[(ours >= 1).argmax(1)], np.nan)
        got = (edata.obs[f"{key}_first"] / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        np.testing.assert_array_equal(got, first, err_msg=key)

    # the grids exercise every stage of both components and creatinine and urine at the thresholds
    for component in (kc, ku):
        assert set(np.unique(component)) == {0, 1, 2, 3}
    crea = np.round(crea, ED.VALUE_DECIMALS)
    low48, low168 = _roll(crea, 48, "min"), _roll(crea, 168, "min")
    filled = np.where(np.isfinite(weight) & (weight > 0), weight, 1.0)
    with np.errstate(invalid="ignore"):
        assert (crea == low48 + 0.3).sum() > 0
        for ratio in (1.5, 2, 3):
            assert (crea == ratio * low168).sum() > 0
        assert ((crea == 4.0) & (crea - low48 >= 0.3)).sum() > 0
        per_kg = {h: _roll(rate, h, "sum") / filled[:, None] for h in (6, 12, 24)}
        observed = ~np.isnan(rate) & (t_hours >= 24)[None, :]
        assert (observed & (per_kg[6] == 0.5)).sum() > 0
        assert (observed & (per_kg[12] == 0)).sum() > 0
