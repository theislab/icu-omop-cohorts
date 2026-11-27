"""The 48 YAIB dynamic variables of a cohort of ICU stays, through ehrdata's OMOP IO.

One observation is one ICU stay (`setup_obs(..., "person_visit_occurrence")`). ehrdata attributes a
MEASUREMENT row to the stay its `visit_occurrence_id` names, so the stays of a person stay apart even
where their time windows overlap; `scope_cohort` keeps only rows that name a cohort stay. ehrdata pools the concepts of a variable
(`concept_ids` as concept sets), drops rows outside ricu's bounds before aggregating (`value_range`),
aggregates per hour as ricu does (median, urine summed) and returns a sparse (stays x variables x hours)
array. The CDM stores values in the unit they were recorded in; ehrdata converts them into YAIB's units
with the crosswalk's UNIT_CONVERSIONS (`unit_conversions`) before the bounds and the aggregation.

A build keeps one EHRData of its stays, with the statics and every per-stay result in `obs`: the layer
`values` holds the 48 variables and the label inputs set up as concept sets (`setup_arguments`), several
calls and the inputs computed by cohort code joined along the variables (`add_variables`); the frames
take the 48 variables on hours [0, T), the ehrapy label tools read the lazy layer `label_view`.

Prints counts only (no patient values).
"""

from __future__ import annotations

import os

import ehrdata as ed
import numpy as np
import pandas as pd
from yaib_omop_crosswalk import RATIO_OF_WBC, RICU_BOUNDS, YAIB_DYNAMIC_VARS, unit_conversion_rows, var_concepts

LAYER = "values"
# The layer the ehrapy label tools read (`label_view`), and its stays per block.
LABEL_VIEW = "label_view"
LABEL_VIEW_CHUNK = 10_000
# The variables ricu sums per hour; every other numeric variable takes the hourly median.
SUM_VARS = ("urine",)
# Label inputs are rounded to this many decimals: a value converted from another unit (115 umol/L / 88.4)
# must compare with KDIGO's +0.3 and x1.5 and SOFA's 1.2 as the decimal it stands for, and a sum must not
# depend on the order of its rows. The unit conversion itself happens in ehrdata (`unit_conversions`).
VALUE_DECIMALS = 6


def concept_sets(ds: str) -> dict[str, list[int]]:
    """{variable: its concept ids on `ds`} for the 48 variables, in YAIB's order."""
    return {var: [int(c) for c in var_concepts(var, ds)] for var in YAIB_DYNAMIC_VARS}


def scope_cohort(con, omop_dir: str, cache_path: str, ds: str, *, voi_table: str | None = None,
                 key_table: str | None = None, label_concepts=(), drop_source_values=()) -> int:
    """Table `_cohort_visits` and the views `visit_occurrence`, `person`, `measurement` for a cohort.

    The cohort is a table of visit_occurrence_ids (`voi_table`, column `voi`) or of
    visit_source_values (`key_table`, column `key`). `measurement` holds the cache rows of the cohort's
    visits that feed one of the 48 variables (also through RATIO_OF_WBC) or have one of the
    `label_concepts`, as recorded, without the rows whose `measurement_source_value` is one of
    `drop_source_values`. Returns the number of cohort visits.
    """
    if (voi_table is None) == (key_table is None):
        raise ValueError("pass exactly one of voi_table and key_table")
    visit_filter = (f"visit_occurrence_id IN (SELECT voi FROM {voi_table})" if voi_table
                    else f"visit_source_value IN (SELECT key FROM {key_table})")
    con.execute(f"""
        CREATE OR REPLACE TABLE _cohort_visits AS
        SELECT * FROM parquet_scan('{omop_dir}/VISIT_OCCURRENCE.parquet') WHERE {visit_filter}""")
    con.execute("CREATE OR REPLACE VIEW visit_occurrence AS SELECT * FROM _cohort_visits")
    con.execute(f"""
        CREATE OR REPLACE VIEW person AS
        SELECT * FROM parquet_scan('{omop_dir}/PERSON.parquet')
        WHERE person_id IN (SELECT person_id FROM _cohort_visits)""")
    if os.path.exists(f"{omop_dir}/DEATH.parquet"):
        con.execute(f"""
            CREATE OR REPLACE VIEW death AS
            SELECT * FROM parquet_scan('{omop_dir}/DEATH.parquet')
            WHERE person_id IN (SELECT person_id FROM _cohort_visits)""")
    concepts = pd.DataFrame({"cid": sorted({c for ids in concept_sets(ds).values() for c in ids}  # noqa: F841
                                           | {int(c) for c in RATIO_OF_WBC.get(ds, {}).values()}
                                           | {int(c) for c in label_concepts})})
    con.execute("CREATE OR REPLACE TABLE _dyn_concepts AS SELECT * FROM concepts")
    dropped = "".join(f" AND m.measurement_source_value IS DISTINCT FROM '{v}'" for v in drop_source_values)
    con.execute(f"""
        CREATE OR REPLACE VIEW measurement AS
        SELECT m.person_id, m.visit_occurrence_id, m.measurement_concept_id,
               m.measurement_date, m.measurement_datetime, m.value_as_number,
               m.unit_concept_id, m.unit_source_value
        FROM parquet_scan('{cache_path}') m
        WHERE m.measurement_concept_id IN (SELECT cid FROM _dyn_concepts)
          AND m.visit_occurrence_id IN (SELECT visit_occurrence_id FROM _cohort_visits)
          AND m.value_as_number IS NOT NULL{dropped}""")
    return con.execute("SELECT count(*) FROM _cohort_visits").fetchone()[0]


def setup_stays(con) -> ed.EHRData:
    """EHRData of the scoped cohort's stays, no variables yet: obs ordered by visit_occurrence_id,
    with `stay_id` from the visit_source_value, `los_hours` from the visit's start and end, and the
    person's DEATH row where the scope has a `death` view."""
    has_death = con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = 'death'").fetchone()[0] > 0
    edata = ed.io.omop.setup_obs(backend_handle=con, observation_table="person_visit_occurrence",
                                 death_table=has_death)
    obs = edata.obs
    obs["stay_id"] = pd.to_numeric(
        obs["visit_source_value"].astype(str).str.split("-", n=1).str[1], errors="coerce").astype("Int64")
    obs["voi"] = obs["visit_occurrence_id"].astype(np.int64)
    obs["los_hours"] = ((pd.to_datetime(obs["visit_end_datetime"]) - pd.to_datetime(obs["visit_start_datetime"]))
                        .dt.total_seconds() / 3600.0)
    assert obs["voi"].is_unique, "one row per visit"
    return edata


def unit_conversions() -> pd.DataFrame:
    return pd.DataFrame(unit_conversion_rows(), columns=[
        "variable", "unit_concept_id", "to_unit_concept_id", "multiplier", "offset"])


def dynamic_set(ds: str, var: str, how: str | None = None) -> tuple:
    """One of the 48 variables as a concept set of `setup_arguments`, aggregated per hour with `how` (default
    ricu's: urine summed, else the median), with ricu's bounds and the variable's unit conversions."""
    how = how or ("sum" if var in SUM_VARS else "median")
    return concept_sets(ds)[var], how, RICU_BOUNDS.get(var), var


def setup_arguments(sets: dict[str, tuple], h_lo: int, h_hi: int, unit: str = "h") -> dict:
    """The arguments of `setup_variables` for the concept sets `sets` = {name: (concept ids, hourly
    aggregation, (lower, upper) or None, the variable whose unit conversions apply or None)} over hours
    [h_lo, h_hi) from each stay's start (h_lo <= 0 < h_hi), or intervals of another `unit` such as "min",
    without the edata, backend and output options. With `layer=LAYER, sparse=True` the layer is a sparse.COO
    (n_obs, len(sets), h_hi - h_lo), NaN where unobserved (0 for counts), with the variables in the order of `sets`;
    data after the visit end count."""
    units = pd.DataFrame([(name, u) for name, (*_, u) in sets.items() if u], columns=["name", "variable"])
    conv = units.merge(unit_conversions(), on="variable").drop(columns="variable").rename(columns={"name": "variable"})
    return dict(
        data_tables=["measurement"], data_field_to_keep={"measurement": "value_as_number"},
        interval_length_number=1, interval_length_unit=unit, time_precision="datetime",
        num_intervals=h_hi, num_intervals_before=-h_lo,
        concept_ids={name: [int(c) for c in ids] for name, (ids, *_) in sets.items()},
        aggregation_strategy={name: how for name, (_, how, *_) in sets.items()},
        value_range={name: rng for name, (_, _, rng, _) in sets.items() if rng is not None} or None,
        # as ricu: a grid hour takes its rows even after discharge (the frames cut the grid at
        # floor(LoS) + 1 hours); the visit end date is no bound
        clip_to_end_date=False,
        unit_conversions=conv if len(conv) else None,
    )


def add_variables(edata: ed.EHRData, var, x) -> ed.EHRData:
    """`edata` with variables appended to layer `values`: `var` their names or their `.var` frame, `x`
    their values (n_obs, len(var), n_t), sparse.COO or dense, NaN where unobserved. A second
    `setup_variables` call returns only its own variables, so the calls of one build are joined here."""
    import sparse

    var = var if isinstance(var, pd.DataFrame) else pd.DataFrame(index=pd.Index(var, dtype=str))
    if not isinstance(x, sparse.COO):
        x = sparse.COO.from_numpy(np.asarray(x, dtype=float), fill_value=np.nan)
    return ed.EHRData(obs=edata.obs, var=pd.concat([edata.var, var]), tem=edata.tem, uns=dict(edata.uns),
                      obsm=dict(edata.obsm.items()),
                      layers={LAYER: sparse.concatenate([edata.layers[LAYER], x], axis=1)})


def variable(edata: ed.EHRData, name: str, rounded: bool = False) -> np.ndarray:
    """(n_obs, n_t) dense array of the variable `name` of layer `values`, NaN where unobserved; with `rounded`, to
    VALUE_DECIMALS, as `label_view` gives a measured label input."""
    x = edata.layers[LAYER]
    keep = x.coords[1] == list(edata.var_names).index(name)
    out = np.full((x.shape[0], x.shape[2]), np.nan)
    out[x.coords[0, keep], x.coords[2, keep]] = x.data[keep]
    return np.round(out, VALUE_DECIMALS) if rounded else out


def hours(edata: ed.EHRData) -> tuple[int, int]:
    """The hours [h_lo, h_hi) from each stay's start that the time axis of `edata` covers."""
    step = edata.tem["interval_step"].to_numpy()
    return int(step[0]), int(step[-1]) + 1


def label_view(edata: ed.EHRData, exact=(), chunk: int = LABEL_VIEW_CHUNK) -> None:
    """Add the layer `label_view`, which the ehrapy label tools read: a dask array over layer `values` whose
    blocks of `chunk` stays and one variable become dense, NaN where unobserved, only when a tool reads them.
    Values are rounded to VALUE_DECIMALS, as the label inputs are, except those of the variables `exact`,
    which cohort code computes and the labels read as they are. Add it after the last `add_variables`,
    which keeps only `values`."""
    import dask.array as da

    rounded = ~edata.var_names.isin(list(exact))

    def dense(block, block_info=None):
        lo, hi = block_info[0]["array-location"][1]
        x = block.todense()
        return np.where(rounded[lo:hi, None], np.round(x, VALUE_DECIMALS), x)

    x = da.from_array(edata.layers[LAYER], chunks=(chunk, 1, -1), asarray=False)
    edata.layers[LABEL_VIEW] = x.map_blocks(dense, dtype=np.float64, meta=np.empty((0, 0, 0)))


def _any_time(edata, con, sets: dict[str, tuple], chunk: int):
    """Per chunk of `chunk` stays of `edata`: their positions in `.obs`, the first hour h_lo of their time axis and
    layer `values` of `sets` set up on an axis that holds every row of these stays, at any time. The axis spans the
    rows of the sets' concepts with an hour to spare on each side, and hour 0 (setup_arguments takes
    h_lo <= 0 < h_hi). Hours count from each stay's `visit_start_datetime` in `.obs`."""
    ids = ", ".join(str(c) for c in sorted({int(c) for ids, *_ in sets.values() for c in ids}))
    for lo_row in range(0, edata.n_obs, chunk):
        rows = np.arange(lo_row, min(lo_row + chunk, edata.n_obs))
        starts = pd.DataFrame({"voi": edata.obs["voi"].to_numpy(dtype=np.int64)[rows],
                               "start": pd.to_datetime(edata.obs["visit_start_datetime"]).to_numpy()[rows]})
        con.register("_any_time_starts", starts)
        try:
            lo, hi = con.execute(f"""
                SELECT min(h), max(h) FROM (
                  SELECT floor(date_diff('second', CAST(s.start AS TIMESTAMP),
                                         coalesce(CAST(m.measurement_datetime AS TIMESTAMP),
                                                  CAST(CAST(m.measurement_date AS DATE) AS TIMESTAMP))) / 3600.0) AS h
                  FROM measurement m JOIN _any_time_starts s ON s.voi = m.visit_occurrence_id
                  WHERE m.measurement_concept_id IN ({ids}))""").fetchone()
        finally:
            con.unregister("_any_time_starts")
        if lo is None:
            continue
        h_lo = min(int(lo) - 1, 0)
        e = ed.io.omop.setup_variables(edata[rows], backend_handle=con, layer=LAYER, sparse=True,
                                       **setup_arguments(sets, h_lo, max(int(hi) + 2, 1)))
        yield rows, h_lo, e.layers[LAYER]


def first_last_hours(edata, con, sets: dict[str, tuple], chunk: int = 20_000) -> pd.DataFrame:
    """Per stay of `edata` and set of `sets` (as `setup_arguments` takes them) the first and the last hour with a
    row, at any time: columns `<name>_first` and `<name>_last`, a row per stay in the order of `.obs`, NaN where a
    stay has none. The rows are those ehrdata sets up (binned, converted and bounded), on an axis that holds all of
    them (`_any_time`)."""
    first = np.full((edata.n_obs, len(sets)), np.inf)
    last = np.full((edata.n_obs, len(sets)), -np.inf)
    for rows, h_lo, x in _any_time(edata, con, sets, chunk):
        r, v, t = x.coords[:, np.isfinite(x.data)]
        np.minimum.at(first, (rows[r], v), t + h_lo)
        np.maximum.at(last, (rows[r], v), t + h_lo)
    out = pd.DataFrame(index=pd.RangeIndex(edata.n_obs))
    for j, name in enumerate(sets):
        out[f"{name}_first"] = np.where(np.isfinite(first[:, j]), first[:, j], np.nan)
        out[f"{name}_last"] = np.where(np.isfinite(last[:, j]), last[:, j], np.nan)
    return out


def ratio_of_wbc(edata: ed.EHRData, ds: str) -> ed.EHRData:
    """`edata` with the variables of RATIO_OF_WBC on `ds` as ricu-extensions' `blood_cell_ratio`: 100 x
    the count (variable `<variable>_count`) over the wbc of the same hour (hourly medians, wbc after
    ricu's bounds), inside ricu's bounds of the variable; NaN where either is missing or the wbc is 0. A
    CDM without the count (one that stores the ratio itself, under the variable's concept) gives nothing
    to replace."""
    for var in RATIO_OF_WBC.get(ds, {}):
        count = variable(edata, f"{var}_count")
        if np.isnan(count).all():
            continue
        wbc = variable(edata, "wbc")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(wbc > 0, 100.0 * count / wbc, np.nan)
        lo, hi = RICU_BOUNDS[var]
        ratio[(ratio < lo) | (ratio > hi)] = np.nan
        edata.layers[LAYER] = replace_variable(edata.layers[LAYER], list(edata.var_names).index(var), ratio)
    return edata


def observed_hours(edata, con, ds: str, variables, chunk: int = 20_000) -> pd.DataFrame:
    """The hours, at any time, in which a stay of `edata` (the stays without variables, `setup_stays`) has a row
    of any of `variables` inside ricu's bounds: a frame of (row, h), `row` the position in `edata.obs`, `h` the
    hour from the stay's start, one row per stay and hour (`_any_time`).
    On a grid, `ep.tl.mark_events` gives this, with the first and last marked time in `obs`; the hours at
    any time, outside the label grid, stay here."""
    parts = []
    for rows, h_lo, x in _any_time(edata, con, {v: dynamic_set(ds, v, "count") for v in variables}, chunk):
        r, _, t = x.coords
        keep = x.data > 0
        parts.append(pd.DataFrame({"row": rows[r[keep]], "h": t[keep].astype(np.int64) + h_lo}).drop_duplicates())
    if not parts:
        return pd.DataFrame({"row": np.array([], dtype=np.int64), "h": np.array([], dtype=np.int64)})
    return pd.concat(parts, ignore_index=True)


def take_rows(x, rows: np.ndarray):
    """Rows `rows` of a sparse.COO (n, V, T), in that order."""
    import sparse

    rows = np.asarray(rows, dtype=np.int64)
    new_of = np.full(x.shape[0], -1, dtype=np.int64)
    new_of[rows] = np.arange(len(rows))
    keep = new_of[x.coords[0]] >= 0
    coords = x.coords[:, keep].copy()
    coords[0] = new_of[coords[0]]
    return sparse.COO(coords, x.data[keep], shape=(len(rows), *x.shape[1:]), fill_value=x.fill_value)


def replace_variable(x, j: int, dense: np.ndarray):
    """sparse.COO (n, V, T) with variable `j` replaced by the (n, T) array `dense` (NaN = unobserved)."""
    import sparse

    keep = x.coords[1] != j
    r, t = np.nonzero(np.isfinite(dense))
    coords = np.concatenate([x.coords[:, keep], np.stack([r, np.full(len(r), j), t])], axis=1)
    data = np.concatenate([x.data[keep], dense[r, t]])
    return sparse.COO(coords, data, shape=x.shape, fill_value=x.fill_value)


def dyn_frame(x, stay_ids: np.ndarray, lengths: np.ndarray) -> pd.DataFrame:
    """YAIB `dyn`: for stay i the hours 0..lengths[i]-1, one column per variable, from sparse.COO x."""
    lengths = np.asarray(lengths, dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    n_rows = int(lengths.sum())
    r, v, t = x.coords
    inside = t < lengths[r]
    values = np.full((n_rows, x.shape[1]), np.nan)
    values[offset[r[inside]] + t[inside], v[inside]] = x.data[inside]
    dyn = pd.DataFrame(values, columns=YAIB_DYNAMIC_VARS)
    dyn.insert(0, "time", np.concatenate([np.arange(L, dtype=np.int64) for L in lengths]) if n_rows else
               np.array([], dtype=np.int64))
    dyn.insert(0, "stay_id", np.repeat(np.asarray(stay_ids, dtype=np.int64), lengths))
    return dyn
