"""ehrapy's SOFA and Sepsis-3 onset (`ep.tl.sofa`, `ep.tl.sepsis3`) against ricu_labels.py, cell by cell, on random
hourly grids with values at and around every threshold. Synthetic arrays only.

The grids are laid out as in the builder: variables of a sparse layer `values` named as the build's EHRData names
them, read through `label_view`, a time axis of ehrdata's `interval_step` in 1 h intervals, and the per-stay times in
obs as time differences. Measured inputs come rounded, as the builder's are.

Needs an ehrapy with `ep.tl.sofa`, as the environment pins; skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ehrapy as ep  # noqa: E402

if not hasattr(ep.tl, "sofa"):
    pytest.skip("this ehrapy has no tl.sofa", allow_module_level=True)

import build_cohort as bn  # noqa: E402
import ehrdata as ed  # noqa: E402
import ehrdata_dynamics as ED  # noqa: E402
import ricu_labels as RL  # noqa: E402

# ricu_labels' vasopressor names -> the build's EHRData's
VASO = {"dopa": "dopa", "dobu": "dobu", "epi": "epi", "norepi": "norepi"}


def _sparse(rng, shape, p, values):
    x = values(shape)
    x[rng.random(shape) > p] = np.nan
    return x


def _inputs(rng, n, H):
    pick = lambda options: (lambda shape: rng.choice(np.asarray(options, dtype=float), size=shape))  # noqa: E731

    def mixed(lo, hi, exact):
        def draw(shape):
            x = rng.uniform(lo, hi, size=shape)
            hit = rng.random(shape) < 0.3
            x[hit] = rng.choice(np.asarray(exact, dtype=float), size=int(hit.sum()))
            return x
        return draw

    measured = lambda p, draw: np.round(_sparse(rng, (n, H), p, draw), ED.VALUE_DECIMALS)  # noqa: E731
    vent = np.zeros((n, H), dtype=bool)
    for i in range(n):
        for _ in range(rng.integers(0, 3)):
            a = rng.integers(0, H)
            vent[i, a:a + rng.integers(1, 40)] = True
    rate = lambda lo, hi, exact: _sparse(rng, (n, H), 0.15, mixed(lo, hi, exact))  # noqa: E731
    vaso = {"dopa": rate(0, 20, [0, 5, 15, 5 + 1e-12, 15 + 1e-9]), "dobu": rate(0, 10, [0, 1e-12]),
            "epi": rate(0, 0.3, [0, 0.1, 0.1 + 1e-12, 0.1 + 1e-7]), "norepi": rate(0, 0.3, [0, 0.1, 0.1 + 1e-12])}
    for k in ("dopa", "epi"):  # a drug without rows, as hirid's dopamine
        if rng.random() < 0.3:
            vaso[k] = np.full((n, H), np.nan)
    urine = measured(0.5, mixed(0, 120, [0, 37.5, 0.1, 12.3, 250]))
    urine[rng.random(n) < 0.2] = np.nan
    return dict(
        po2=measured(0.2, mixed(30, 450, [60, 100, 200])),
        fio2=measured(0.15, pick([0, 21, 30, 40, 50, 60, 100])),
        vent=vent,
        plt=measured(0.1, mixed(5, 300, [20, 50, 100, 150])),
        bili=measured(0.05, mixed(0.2, 15, [1.2, 2, 6, 12])),
        map_=measured(0.6, mixed(30, 120, [70])),
        vaso=vaso,
        gcs=_sparse(rng, (n, H), 0.15, pick(np.arange(3, 16))),
        crea=measured(0.08, mixed(0.3, 8, [1.2, 2, 3.5, 5])),
        urine=urine,
    )


def _object(grids: dict, h_lo: int, obs: dict) -> ed.EHRData:
    """An EHRData of `grids` as the build's: layer `values`, hours h_lo, h_lo + 1, ... as ehrdata's OMOP setup records
    them, the hours `obs` in obs as time differences, and `label_view` in blocks of 97 stays."""
    n, H = next(iter(grids.values())).shape
    edata = ed.EHRData(
        shape=(n, len(grids)),
        obs=pd.DataFrame({k: pd.to_timedelta(np.asarray(h, dtype=float), unit="h") for k, h in obs.items()},
                         index=pd.RangeIndex(n).astype(str)),
        var=pd.DataFrame(index=list(grids)),
        tem=pd.DataFrame({"interval_step": np.arange(h_lo, h_lo + H)}),
        uns={"omop_io_variable_setup": {"interval_length_number": 1, "interval_length_unit": "h"}},
        layers={ED.LAYER: sparse.COO.from_numpy(np.stack(list(grids.values()), axis=1).astype(float), fill_value=np.nan)})
    ED.label_view(edata, exact=bn.COMPUTED_INPUTS, chunk=97)
    return edata


def _hours(rng, n, h_lo, H, p_nan=0.2):
    """Hours on the grid's clock, before, in and after the grid."""
    h = rng.integers(h_lo - 60, h_lo + H + 20, size=n).astype(float)
    h[rng.random(n) < p_nan] = np.nan
    return h


def _differing(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return int(((np.isnan(a) != np.isnan(b)) | (~np.isnan(a) & ~np.isnan(b) & (a != b))).sum())


@pytest.mark.parametrize(("seed", "h_lo"), [(0, 0), (1, -288), (2, 17), (3, -5)])
def test_sofa_equals_ricu_labels(seed, h_lo):
    rng = np.random.default_rng(seed)
    n, H = 300, 160
    inputs = _inputs(rng, n, H)
    first = _hours(rng, n, h_lo, H)
    u_lo = _hours(rng, n, h_lo, H)
    u_hi = np.where(np.isnan(u_lo), np.nan, u_lo + rng.integers(0, 200, size=n))

    comps = RL.sofa_components(**{**inputs, "vaso": {k: np.nan_to_num(v) for k, v in inputs["vaso"].items()}},
                               urine_range=(u_lo - h_lo, u_hi - h_lo))
    total = RL.sofa_score(comps, first=first - h_lo)
    grids = {"po2_min": inputs["po2"], "fio2_max": inputs["fio2"], "vent": np.where(inputs["vent"], 1.0, np.nan),
             "plt_min": inputs["plt"], "bili_max": inputs["bili"], "map_min": inputs["map_"],
             **{VASO[k]: v for k, v in inputs["vaso"].items()}, "gcs": inputs["gcs"], "crea_max": inputs["crea"],
             "urine": inputs["urine"]}
    edata = _object(grids, h_lo, {"sofa_first": first, "urine_first": u_lo, "urine_last": u_hi})
    bn.score_sofa(edata, "miiv")

    assert _differing(total, edata.obsm["sofa"]) == 0
    for k, organ in bn.SOFA_ORGANS.items():
        assert _differing(comps[k], edata.obsm[f"sofa_{organ}"]) == 0, k
    # the grids exercise every score of every organ, and stays scored before their first input
    for k in bn.SOFA_ORGANS:
        assert set(np.unique(comps[k][~np.isnan(comps[k])])) == {0, 1, 2, 3, 4}, k
    assert np.isfinite(total).sum() > n * H / 2
    seen = np.logical_or.accumulate(np.any([~np.isnan(c) for c in comps.values()], axis=0), axis=1)
    assert (np.isfinite(total) & ~seen).sum() > 0


def test_label_view_rounds_measured_inputs_only():
    """`label_view` rounds as the label inputs are rounded and leaves the inputs computed by cohort code as they are,
    so a rate just above a SOFA threshold stays above it."""
    x = np.array([[0.1 + 1e-7, 1.0 / 3.0, np.nan]])
    edata = _object({"norepi": x, "crea_max": x}, 0, {})
    view = np.asarray(edata.layers[ED.LABEL_VIEW])
    assert view[0, 0, 0] == 0.1 + 1e-7 and view[0, 0, 1] == 1.0 / 3.0
    assert view[0, 1, 0] == 0.1 and view[0, 1, 1] == round(1.0 / 3.0, ED.VALUE_DECIMALS)
    assert np.isnan(view[0, :, 2]).all()


@pytest.mark.parametrize("seed", range(4))
def test_sep3_onset_equals_ricu_labels(seed):
    rng = np.random.default_rng(seed)
    n, H = 400, 121
    total = rng.integers(0, 8, size=(n, H)).astype(float)
    total = np.minimum.accumulate(total[:, ::-1], axis=1)[:, ::-1] + rng.integers(0, 3, size=(n, H))
    total[rng.random((n, H)) < 0.3] = np.nan
    total[rng.random(n) < 0.1] = np.nan
    si = rng.integers(-40, H + 40, size=n)

    for lwr, upr, thresh in ((48, 24, 2), (10, 5, 3), (0, 6, 1)):
        on = RL.sep3_onset(total, si, lwr=lwr, upr=upr, thresh=thresh, has_si=np.ones(n, dtype=bool))
        edata = _object({"x": np.full((n, H), np.nan)}, 0, {})
        edata.obs["suspected_infection"] = si.astype(float)   # a position, as the grid starts at hour 0
        edata.obsm["sofa"] = total
        ep.tl.sepsis3(edata, infection_key="suspected_infection", before=lwr, after=upr, rise=thresh)
        onset = edata.obs["sepsis3_onset"].to_numpy(dtype=float)
        np.testing.assert_array_equal(on, np.where(edata.obs["sepsis3"], onset, -1).astype(np.int64))
        assert 0 < (on >= 0).sum() < n
