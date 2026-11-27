"""Cohort exclusions as ehrapy filters on `.obs`, tracked step by step.

Every exclusion of ricu's cohort scripts is a query on an `.obs` column of the stays' EHRData,
applied with `ehrapy.pp.filter_observations` and recorded by one `ehrapy.tl.CohortTracker`, so a
build leaves an attrition table and a flowchart of the whole cohort definition.

ricu excludes a stay only where the comparison is TRUE (`filter_step(~ . < thr)`), so a missing
value is never an exclusion. The queries are written as `~(x < thr)`: a NaN compares False and the
stay is kept. That holds for numpy dtypes only (a pandas NA compares NA, which ehrapy counts as
failing), so `apply` turns the nullable columns a criterion uses into float64, and nullable flags
into bool with NA as False. Prints and writes
counts only.
"""

from __future__ import annotations

import os

import ehrapy as ep
import ehrdata as ed
import pandas as pd

# base_cohort.R, exclusions 1-5
BASE_CRITERIA = [
    "~(los_hours < 0)",           # (1) negative length of stay
    "~(los_hours < 6)",           # (2) ICU stay shorter than 6 h
    "~(n_hours < 4)",             # (3) fewer than 4 hours with any of the 48 variables, at any time
    "~(max_empty_run > 12)",      # (4) a run of more than 12 empty hours on 0..min(floor(LoS), 168)
    "~(age < 18)",                # (5) younger than 18
]
TRACKED_COLUMNS = ["age", "los_hours", "sex"]


def tracker(edata: ed.EHRData) -> ep.tl.CohortTracker:
    return ep.tl.CohortTracker(edata, columns=TRACKED_COLUMNS, categorical=["sex"])


def apply(edata: ed.EHRData, criteria: list[str], ct: ep.tl.CohortTracker) -> list[tuple[str, int]]:
    """Filter `edata` in place by each criterion in turn; returns (criterion, stays left) per step."""
    referenced = [c for c in edata.obs.columns if any(c in criterion for criterion in criteria)]
    for column in referenced:
        values, dtype = edata.obs[column], str(edata.obs[column].dtype)
        if dtype == "boolean":  # a missing flag excludes nothing
            edata.obs[column] = values.fillna(False).astype(bool)
        elif dtype in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"):
            edata.obs[column] = values.astype("float64")
    steps = []
    for criterion in criteria:
        before = edata.n_obs
        ep.pp.filter_observations(edata, query=criterion, tracker=ct)
        steps.append((criterion, edata.n_obs))
        print(f"[filter] {criterion}: {before} -> {edata.n_obs}", flush=True)
    return steps


def obs_only(edata: ed.EHRData, columns: list[str]) -> ed.EHRData:
    """An EHRData of `edata`'s obs columns alone, to filter without copying its layers."""
    return ed.EHRData(obs=edata.obs[list(dict.fromkeys(TRACKED_COLUMNS + columns))].copy())


def write(out_dir: str, n_start: int, steps: list[tuple[str, int]], ct: ep.tl.CohortTracker) -> None:
    """attrition.csv (one row per step) and attrition_flowchart.html (ehrapy's flowchart)."""
    rows, n = [], n_start
    for criterion, left in steps:
        rows.append({"criterion": criterion, "excluded": n - left, "remaining": left})
        n = left
    pd.DataFrame([{"criterion": "candidates", "excluded": 0, "remaining": n_start}, *rows]).to_csv(
        os.path.join(out_dir, "attrition.csv"), index=False)
    try:
        import holoviews as hv

        hv.save(ct.plot_flowchart(title="cohort"), os.path.join(out_dir, "attrition_flowchart.html"), backend="bokeh")
    except Exception as error:  # the flowchart is a report, never a reason to fail a build
        print(f"[filter] flowchart not written: {error!r}", flush=True)
