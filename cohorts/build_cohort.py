"""The YAIB cohorts from an OMOP CDM: the cohort, statics, labels and the 48 dynamic variables
of a task and dataset, through ehrdata's OMOP IO, ehrapy's cohort filters and duckdb. Writes YAIB's
dyn/sta/outc frames from one EHRData of the build's stays (`build_object`).

Base cohort = 5 exclusions (base_cohort.R): (1) LoS<0, (2) ICU<6h, (3) <4 distinct hours with any
of the 48 variables, (4) a run of >12 empty hourly bins, (5) age<18, computed on the 48 variables
after ricu's bounds. A missing value is never an exclusion. The comparison with ricu, per decision:
../RICU.md. Prints metadata/counts only (no patient values).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings

# the modules next to this one, also when the builder is loaded by its path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cohort_filters as CF  # noqa: E402
import duckdb  # noqa: E402
import ehrapy as ep  # noqa: E402
import ehrdata as ed  # noqa: E402
import ehrdata_dynamics as ED  # noqa: E402
import label_inputs as LI  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sparse  # noqa: E402
from duck_tmp import make_job_tmp  # noqa: E402
from yaib_omop_crosswalk import (  # noqa: E402
    CAI_CONVERT,
    COHORT_CACHE,
    COHORTS_ROOT,
    DATASET_PREFIX,
    FLUID_BALANCE_SOURCE,
    FRAMES,
    GENDER_CONCEPT_TO_SEX,
    HIRID_AIRWAY_SOURCE,
    OMOP_CDM,
    RASS_CONCEPT_ID,
    RATIO_OF_WBC,
    STATIC_UNIT_CONVERSIONS,
    VAR_CONCEPTS,
    YAIB_DYNAMIC_VARS,
    assert_cache_covers_targets,
    require_cdm,
)

OMOP = OMOP_CDM          # environment OMOP_CDM (see yaib_omop_crosswalk.py)
CACHE = COHORT_CACHE      # environment COHORT_CACHE
# FRAMES_OUT redirects a check build so it cannot overwrite the locked outputs.
OUT = os.environ.get("FRAMES_OUT") or FRAMES
# duckdb's spill files (duck_tmp.py), next to the outputs unless DUCK_TMP_BASE says otherwise
DUCK_TMP_BASE = os.environ.get("DUCK_TMP_BASE") or (f"{COHORTS_ROOT}/_duckdb_tmp" if COHORTS_ROOT else "")


def to_parquet_atomic(df, path: str) -> None:
    """Write `df` to a temporary file, flush it to disk and rename it into place, so that a reader never sees a
    partial file."""
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)

TASK_HORIZON = {"mortality24": 24, "kidney_function": 24, "aki": 168, "sepsis": 168, "los": 168}
# ricu's grid is INCLUSIVE of both endpoints. `map_to_grid` uses `ricu::expand`, whose core is
# `seq(min, max, step_size)` -- and R's seq() includes the upper bound -- so a T=24 task yields
# hours 0..24, i.e. T+1 = 25 rows per stay, not 24. Verified in the reference cohorts: every one
# of the 49,523 miiv/mortality24 stays has exactly 25 rows, max time 86400 s = 24 h; los has 169.
# We were producing 0..T-1 and silently dropping each stay's final hour on BOTH routes.
N_BINS = {task: T + 1 for task, T in TASK_HORIZON.items()}

# OBSERVATION row with the age at the stay (one per visit)
AGE_CONCEPT_ID = 44804452
# Height and weight: MEASUREMENT rows of these concepts, in cm and kg after STATIC_UNIT_CONVERSIONS
HEIGHT_CONCEPT_IDS = (3036277,)
WEIGHT_CONCEPT_IDS = (3025315,)
# ricu's source items of height and weight (`measurement_source_value`): miiv 226707 (inches) and 226512
# "Admission Weight (Kg)", hirid 10000450 and 10000400. eicu's ricu source is the admission value, which is the
# ETL's averaged row ('raw_height', 'raw_weight').
RICU_STATIC_SOURCE_ITEMS = {"height": ("226707", "10000450"), "weight": ("226512", "10000400")}
# Experiment: replace hirid's `urine` (real hourly volume) with ricu's definition
# (de-cumulated fluid balance). Off by default -- we deliberately diverge. Set via the environment
# so the comparison needs no code edit: RICU_HIRID_URINE=1
RICU_HIRID_URINE = os.environ.get("RICU_HIRID_URINE", "") == "1"
# COHORT_DIAG=1 writes per-stay label ingredients (diag_*.parquet) next to the outputs
DIAG = os.environ.get("COHORT_DIAG", "") == "1"
# ricu's filter_bounds for these concepts; replicated so we do not hand YAIB a value ricu drops.
HEIGHT_RANGE_CM = (10.0, 230.0)
WEIGHT_RANGE_KG = (1.0, 500.0)
BASE_MAX_H = 168
MIN_MEAS = 4
MAX_GAP_H = 12
MIN_LOS_H = 6
# Start of the label grids before ICU admission (aki creatinine for stays without earlier values;
# the sepsis SOFA written to logs and diagnostics). ricu loads each concept over all times.
# Antibiotics are read at all times, aki creatinine before this grid on a per-stay grid
# (`_aki_onset`), the sepsis onset on a grid around each stay's suspected infection (`_sep3_at_si`).
LABEL_PRE_H = 12 * 24
# ricu kidney_function.R reads creatinine at hours 25..48
KIDNEY_FUNCTION_HOURS = (25, 49)


def object_hours(task: str, T: int) -> tuple[int, int]:
    """Hours [h_lo, h_hi) from ICU admission of a build's EHRData: from LABEL_PRE_H before admission for
    aki and sepsis, whose labels read earlier values, to the end of KIDNEY_FUNCTION_HOURS for
    kidney_function, else the frames' [0, T)."""
    return (-LABEL_PRE_H if task in ("aki", "sepsis") else 0,
            KIDNEY_FUNCTION_HOURS[1] if task == "kidney_function" else T)


def kidney_function_label(crea_hourly):
    """ricu kidney_function.R: hourly crea (median per hour, bounds applied), index in (24, 48],
    i.e. hours 25..48, then the median over those hours. `crea_hourly` holds hours 25..48."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(crea_hourly, axis=1)


def los_label(los_h, t):
    """ricu los.R: `floor(los_icu * 24)` hours, then `pmin(7 * 24, los_icu - start)`."""
    return np.minimum(168.0, np.floor(los_h) - t)

def _statics_height_weight(con, measurement_path, voi_order, ds):
    """(height_cm, weight_kg) aligned to `voi_order`, from the cohort's MEASUREMENT cache: per stay the median of
    ricu's source items, converted to cm and kg and within ricu's ranges. On eicu a stay without these items takes
    the median of its other height and weight rows."""
    h_ids = ",".join(str(c) for c in HEIGHT_CONCEPT_IDS)
    w_ids = ",".join(str(c) for c in WEIGHT_CONCEPT_IDS)
    # the rows in cm and kg (STATIC_UNIT_CONVERSIONS; a row without a listed unit is taken as is)
    conv = [(cid, unit, mult, off) for kind, ids in (("height", HEIGHT_CONCEPT_IDS), ("weight", WEIGHT_CONCEPT_IDS))
            for cid in ids for unit, (mult, off) in STATIC_UNIT_CONVERSIONS[kind][1].items()]
    values = ", ".join(f"({c}, {u}, {m!r}, {o!r})" for c, u, m, o in conv)
    rows_cm_kg = (f"(SELECT m.* REPLACE (m.value_as_number * coalesce(u.mult, 1.0) + coalesce(u.off, 0.0) "
                  f"AS value_as_number) FROM parquet_scan('{measurement_path}') m "
                  f"LEFT JOIN (VALUES {values}) u(cid, unit, mult, off) "
                  f"ON u.cid = m.measurement_concept_id AND u.unit = m.unit_concept_id)")
    # ricu's filter_bounds drops rows outside 10-230 cm / 1-500 kg before the median
    in_range = (f"((measurement_concept_id IN ({h_ids}) AND TRY_CAST(value_as_number AS DOUBLE) "
                f"BETWEEN {HEIGHT_RANGE_CM[0]} AND {HEIGHT_RANGE_CM[1]}) OR "
                f"(measurement_concept_id IN ({w_ids}) AND TRY_CAST(value_as_number AS DOUBLE) "
                f"BETWEEN {WEIGHT_RANGE_KG[0]} AND {WEIGHT_RANGE_KG[1]}))")
    h_items = ",".join(f"'{i}'" for i in RICU_STATIC_SOURCE_ITEMS["height"])
    w_items = ",".join(f"'{i}'" for i in RICU_STATIC_SOURCE_ITEMS["weight"])
    hw = con.execute(
        f"""
        WITH m AS (
          SELECT visit_occurrence_id AS voi, measurement_concept_id AS cid,
                 TRY_CAST(value_as_number AS DOUBLE) AS v,
                 CASE WHEN measurement_concept_id IN ({h_ids})
                        AND measurement_source_value IN ({h_items}) THEN 0
                      WHEN measurement_concept_id IN ({w_ids})
                        AND measurement_source_value IN ({w_items}) THEN 0
                      ELSE 1 END AS tier
          FROM {rows_cm_kg}
          WHERE measurement_concept_id IN ({h_ids},{w_ids}) AND value_as_number IS NOT NULL
            AND {in_range})
        ,
        -- median over the best tier's rows, as ricu does; deterministic on ties
        k AS (SELECT *, CASE WHEN cid IN ({h_ids}) THEN 'h' ELSE 'w' END AS kind FROM m),
        best AS (SELECT voi, kind, MIN(tier) AS bt FROM k GROUP BY voi, kind)
        SELECT voi,
               MIN(tier) FILTER (WHERE kind = 'h') AS _ht,
               MIN(tier) FILTER (WHERE kind = 'w') AS _wt,
               MEDIAN(v) FILTER (WHERE kind = 'h') AS height,
               MEDIAN(v) FILTER (WHERE kind = 'w') AS weight
        FROM k JOIN best USING (voi, kind)
        WHERE tier = bt {"" if ds == "eicu" else "AND tier = 0"}
        GROUP BY voi
        """
    ).df().set_index("voi")
    n_ricu_h = int((hw["_ht"] == 0).sum()); n_ricu_w = int((hw["_wt"] == 0).sum())
    print(f"[statics] ricu-item rows used: height {n_ricu_h}, weight {n_ricu_w} "
          f"of {len(hw)} stays", flush=True)
    hw = hw.drop(columns=["_ht", "_wt"]).reindex(voi_order)
    n_h, n_w = int(hw["height"].notna().sum()), int(hw["weight"].notna().sum())
    print(f"[statics] height {n_h}/{len(hw)} ({100*n_h/max(len(hw),1):.1f}%)  "
          f"weight {n_w}/{len(hw)} ({100*n_w/max(len(hw),1):.1f}%)", flush=True)
    if n_h == 0 and n_w == 0:
        raise RuntimeError("no height or weight in the cohort cache: TARGET_MEASUREMENT_CONCEPT_IDS must hold "
                           f"{HEIGHT_CONCEPT_IDS} and {WEIGHT_CONCEPT_IDS}")
    return hw["height"].to_numpy(), hw["weight"].to_numpy()


def _obs_hours(obs: pd.DataFrame, key: str) -> np.ndarray:
    """The time differences of obs column `key` in hours, NaN where missing."""
    return (obs[key] / pd.Timedelta(hours=1)).to_numpy(dtype=float)


# --- KDIGO / AKI ------------------------------------------------------------------------------
# ricu-extensions `callback-kdigo.R` (the overrides that built the benchmark cohorts) by ep.tl.kdigo on the hourly
# creatinine and ricu's urine rate per row (`urine_rate`), with ricu's windows of 48 h and 7 days for the
# creatinine and these urine criteria (stage, hours, mL/kg/h).
# RICU CHOICE (RICU.md, "Where we follow ricu"): ricu compares the urine per kg of a window with KDIGO's rates
# per hour, so its thresholds as mean rates are KDIGO's divided by the window.
KDIGO_URINE_CRITERIA = ((1, 6, 0.5 / 6), (2, 12, 0.5 / 12), (3, 24, 0.3 / 24))
KDIGO_ANURIA_H = 12                 # no urine over 12 h is stage 3
# RICU CHOICE (RICU.md, "Where we follow ricu"): the thresholds compared exactly in floating point, as ricu, so
# that a rise from 1.1 to 1.4 mg/dL (0.29999999999999982) is no rise of 0.3; ep.tl.kdigo allows 1e-9 by default
KDIGO_TOLERANCE = 0.0
AKI_THRESHOLD = 1                   # ricu: `aki := kdigo >= threshold`, threshold = 1, i.e. any stage but 0
AKI_OUTCOME_WINDOW_H = 6            # aki.R: outcome_window(window = c(6L, 6L)) -- SYMMETRIC
AKI_BASELINE_CREA_MAX = 4.0         # aki.R excl8: baseline creatinine > 4 excludes the stay

# --- Sepsis-3 -----------------------------------------------------------------------------------
# sepsis.R: outcome is sep3_alt (continuous-abx variant), summary_step "first", stop_obs_at +6h,
# then map_to_grid + the same symmetric outcome_window(c(6,6)) as aki. For eicu and hirid it passes
# si_mode="abx" (antibiotics alone, no cultures), for miiv "and" (antibiotics and cultures).
# The antibiotics are ricu's `abx_duration` items (abx_sets.py, LI.infection_tables).
# ricu-extensions `abx_cont`: an administration counts if, over the ABX_WIN_H hours from it (or up to death), no
# gap from the latest end of the administrations so far to the next one or to the window end exceeds ABX_MAX_GAP_H
# (ep.tl.mark_sustained_events).
ABX_WIN_H = 72
ABX_MAX_GAP_H = 24
# ricu `si_and`: an antibiotic time counts if a sample follows within 24 h, a sample time if an antibiotic
# follows within 72 h (ep.tl.mark_paired_events)
SI_ABX_TO_SAMPLE_H = 24
SI_SAMPLE_TO_ABX_H = 72
# The suspected infection runs on chunks of stays, each on a grid from its first to its last antibiotic or
# culture row, of at most this many stays x hours
SI_MAX_CELLS = 50_000_000
SEPSIS_OUTCOME_WINDOW_H = 6
# sep3 looks for the SOFA rise in [si - 48, si + 24] around the first suspected infection, at any
# time of the stay. A SOFA row there depends on inputs up to 48 h earlier (the 24 h worst value
# of urine24, itself a 24 h window), so the onset grid starts that much before the window.
SEP3_SI_LWR_H = 48
SEP3_SI_UPR_H = 24
SEP3_MARGIN_H = 48
# Variables whose rows open SOFA rows; with the GCS components, the urine and the vasopressors they
# give the first SOFA input row at any time (`sofa_first` in obs), from which ep.tl.sofa scores.
SOFA_ROW_VARS = ("po2", "plt", "bili", "map", "crea")
# ep.tl.sofa's inputs as variables of the build's EHRData; the urine is `urine_fb` on hirid
SOFA_INPUTS = dict(pao2="po2_min", fio2="fio2_max", ventilation="vent", platelets="plt_min", bilirubin="bili_max",
                   mean_arterial_pressure="map_min", dopamine="dopa", dobutamine="dobu", epinephrine="epi",
                   norepinephrine="norepi", gcs="gcs", creatinine="crea_max")
# The label inputs computed by cohort code (`add_sofa_inputs`, `kdigo_object`), which `label_view` leaves unrounded
COMPUTED_INPUTS = ("vent", "vent_invasive", "gcs", *LI.VASO, "urine_fb", "urine_rate")
# The organs of ep.tl.sofa by the names of the logs and diagnostics
SOFA_ORGANS = {"resp": "respiration", "coag": "coagulation", "liver": "liver", "cardio": "cardiovascular",
               "cns": "cns", "renal": "renal"}
# Label inputs that aggregate a dynamic's concepts another way, as (variable, hourly aggregation), in
# the build's EHRData as `<variable>_<aggregation>`. A concept may be in one set per setup_variables
# call, so these take a second call.
SECOND_AGGREGATIONS = (("crea", "max"), ("po2", "min"), ("fio2", "max"), ("plt", "min"), ("map", "min"),
                       ("bili", "max"))


def label_sets(ds: str) -> tuple[dict, dict]:
    """The label inputs of the build's EHRData as concept sets of `ED.setup_arguments`: those with concepts
    of their own (the GCS components as `LI.gcs` reads them, eicu's RASS in the sedated range, hirid's
    neutrophil count for `neut`), and the SECOND_AGGREGATIONS."""
    own = {k: ([cid], "min", LI.GCS_BOUNDS[k], None) for k, cid in LI.GCS_CONCEPTS.items()}
    if ds == "eicu":
        own["rass"] = ([RASS_CONCEPT_ID], "min", (-5, LI.RASS_SEDATED), None)
    own.update({f"{var}_count": ([cid], "median", None, None) for var, cid in RATIO_OF_WBC.get(ds, {}).items()})
    second = {f"{var}_{how}": ED.dynamic_set(ds, var, how) for var, how in SECOND_AGGREGATIONS}
    return own, second


def build_object(con, stays: ed.EHRData, ds: str, h_lo: int, h_hi: int) -> ed.EHRData:
    """The build's EHRData, one per task and dataset: the obs of `stays` (the stays without variables) on hours
    [h_lo, h_hi) from each stay's start, with layer `values` holding the 48 dynamics in YAIB order and the label
    inputs of `label_sets`, in two setup_variables calls; hirid's `neut` from its count."""
    t0 = time.time()
    own, second = label_sets(ds)
    sets = {**{v: ED.dynamic_set(ds, v) for v in YAIB_DYNAMIC_VARS}, **own}
    edata = ed.io.omop.setup_variables(stays, backend_handle=con, layer=ED.LAYER, sparse=True,
                                       **ED.setup_arguments(sets, h_lo, h_hi))
    more = ed.io.omop.setup_variables(stays, backend_handle=con, layer=ED.LAYER, sparse=True,
                                      **ED.setup_arguments(second, h_lo, h_hi))
    edata = ED.ratio_of_wbc(ED.add_variables(edata, more.var, more.layers[ED.LAYER]), ds)
    print(f"[object] {edata.n_obs} stays x {edata.n_vars} variables x hours [{h_lo}, {h_hi}): "
          f"nnz={edata.layers[ED.LAYER].nnz:,} in {time.time() - t0:.0f} s", flush=True)
    return edata


def _ricu_hirid_urine(con, measurement_path, voi_order, T):
    """ricu's hirid dyn `urine`: de-cumulated FLUID BALANCE, for the experiment with ricu's urine (RICU_HIRID_URINE).

    ricu's base `urine` concept for hirid reads variableid 30005110 ("Fluid balance", a cumulative
    value) and de-cumulates it per patient: the first value is kept, every later one becomes the
    difference to the one before, and negative differences (counter resets) become 0, in mL. We
    normally DIVERGE from this and use real hourly urine volume (10020000), because fluid balance is
    intake minus output and not urine. This function exists only to reproduce ricu's definition for
    the comparison, and is not used by default.
    """
    df = con.execute(
        f"""
        SELECT m.visit_occurrence_id AS voi,
               CAST(floor(date_diff('second', v.visit_start_datetime, m.measurement_datetime)/3600.0) AS INTEGER) AS h,
               last(m.value_as_number ORDER BY m.measurement_datetime, m.value_as_number) AS v
        FROM parquet_scan('{measurement_path}') m
        JOIN _cohort_visits v ON v.visit_occurrence_id = m.visit_occurrence_id
        WHERE m.measurement_source_value = '{FLUID_BALANCE_SOURCE}' AND m.value_as_number IS NOT NULL
          AND date_diff('second', v.visit_start_datetime, m.measurement_datetime)/3600.0 >= 0
          AND date_diff('second', v.visit_start_datetime, m.measurement_datetime)/3600.0 < {T}
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()
    pos = {v: i for i, v in enumerate(voi_order)}
    cum = np.full((len(voi_order), T), np.nan)
    if len(df):
        idx = df["voi"].map(pos)
        ok = idx.notna()
        cum[idx[ok].to_numpy().astype(int), df["h"][ok].to_numpy().astype(int)] = df["v"][ok].to_numpy()
    out = np.full_like(cum, np.nan)
    for i in range(cum.shape[0]):
        hrs = np.flatnonzero(~np.isnan(cum[i]))
        if hrs.size == 0:
            continue
        vals = cum[i, hrs]
        d = np.empty_like(vals)
        d[0] = vals[0]                      # the first value, then the differences
        d[1:] = np.diff(vals)
        d[d < 0] = 0                        # res[res < 0] <- 0
        out[i, hrs] = d
    return out


def add_sofa_inputs(con, edata: ed.EHRData, ds: str, weight) -> ed.EHRData:
    """`edata` with the SOFA inputs that cohort code computes appended on its hours (COMPUTED_INPUTS):
    ventilation (`vent`, hirid `vent_invasive`), GCS with sedation (`gcs`) from the GCS components and
    eicu's `rass` of the object, the vasopressor rates and hirid's fluid-balance urine (`urine_fb`). Each
    stay's hours count from its `visit_start_datetime` in obs.

    Also writes to obs, as time differences from that start, the first SOFA input row at any time
    (`sofa_first`: SOFA_ROW_VARS, the GCS components, the urine and the first vasopressor piece at
    any rate) and the first and the last urine row (`urine_first`, `urine_last`), from which
    ep.tl.sofa scores and extrapolates the urine, as ricu over the whole timeline."""
    h_lo, h_hi = ED.hours(edata)
    obs = edata.obs
    cache = f"{CACHE}/measurement_{ds}.parquet"
    vent, invasive = LI.ventilation(con, OMOP, cache, ds, obs, h_lo, h_hi, HIRID_AIRWAY_SOURCE)
    gcs = LI.gcs(con, cache, ds, obs, {k: ED.variable(edata, k) for k in LI.GCS_CONCEPTS}, h_lo, h_hi,
                 invasive=invasive, rass=ED.variable(edata, "rass") if ds == "eicu" else None)
    vaso, vaso_first = LI.vasopressors(con, OMOP, ds, obs, weight, h_lo, h_hi)
    grids = {"vent": np.where(vent, 1.0, np.nan), "gcs": gcs, **vaso}
    own, _ = label_sets(ds)
    sets = {v: ED.dynamic_set(ds, v) for v in SOFA_ROW_VARS} | {k: own[k] for k in LI.GCS_CONCEPTS}
    if ds == "hirid":
        grids["vent_invasive"] = np.where(invasive, 1.0, np.nan)
        # RICU CHOICE (RICU.md, "Where we follow ricu"): ricu's hirid urine24 is built on
        # de-cumulated fluid balance, which is intake minus output, not urine output
        grids["urine_fb"], u_first, u_last = LI.hirid_fluid_urine(con, cache, obs, h_lo, h_hi, FLUID_BALANCE_SOURCE)
        rows = ED.first_last_hours(edata, con, sets)
    else:
        rows = ED.first_last_hours(edata, con, sets | {"urine": ED.dynamic_set(ds, "urine")})
        u_first, u_last = rows["urine_first"].to_numpy(), rows["urine_last"].to_numpy()
    # RICU CHOICE (RICU.md, "Where we follow ricu"): a vasopressor row starts the SOFA rows even
    # at rate 0, where ricu's sofa_cardio scores it 0
    first = np.fmin.reduce([rows[f"{k}_first"].to_numpy() for k in sets] + [u_first, vaso_first])
    # each dense grid to sparse on its own, so that the grids are never dense side by side
    x = sparse.concatenate([sparse.COO.from_numpy(np.asarray(g, dtype=float)[:, None, :], fill_value=np.nan)
                            for g in grids.values()], axis=1)
    edata = ED.add_variables(edata, list(grids), x)
    for key, h in (("sofa_first", first), ("urine_first", u_first), ("urine_last", u_last)):
        edata.obs[key] = pd.to_timedelta(np.asarray(h, dtype=float), unit="h")
    return edata


def score_sofa(edata: ed.EHRData, ds: str) -> None:
    """ricu's SOFA (ep.tl.sofa, with ricu's windows of a day that include both ends) on the inputs of
    `add_sofa_inputs`, read from `label_view`: `obsm["sofa"]` and `obsm["sofa_<organ>"]`. The tool takes the times
    of obs as positions on the object's hours."""
    h_lo, _ = ED.hours(edata)
    for key in ("sofa_first", "urine_first", "urine_last"):
        edata.obs[f"{key}_position"] = _obs_hours(edata.obs, key) - h_lo
    ED.label_view(edata, exact=COMPUTED_INPUTS)
    ep.tl.sofa(edata, layer=ED.LABEL_VIEW, **SOFA_INPUTS, urine="urine_fb" if ds == "hirid" else "urine",
               urine_period_keys=("urine_first_position", "urine_last_position"), start_key="sofa_first_position",
               closed="both")


def sofa_object(con, stays: ed.EHRData, ds: str, weight, h_lo: int, h_hi: int) -> ed.EHRData:
    """The build's EHRData of `stays` on hours [h_lo, h_hi) from their start, with the SOFA scored."""
    edata = add_sofa_inputs(con, build_object(con, stays, ds, h_lo, h_hi), ds, weight)
    score_sofa(edata, ds)
    return edata


def _sep3_at_si(con, ds, stays: ed.EHRData, weight) -> None:
    """ricu's Sepsis-3 for the stays `stays` (without variables) with `si_first` in obs, on a grid anchored at each
    stay's first suspected infection: hours [si - 48 - margin, si + 24], where ricu looks for the SOFA rise. Writes to
    obs `sepsis3` and `sepsis3_onset`, a time difference from the stay's start, NaT if none; the onset may lie long
    before admission or after the label grid (which counts for the hospital prevalence). The SOFA comes from the
    build's EHRData on a copy of the stays' obs whose start is moved to each stay's anchor (`origin_h` hours after
    admission)."""
    si = _obs_hours(stays.obs, "si_first")
    has = np.zeros(stays.n_obs, dtype=bool)
    onset_h = np.full(stays.n_obs, np.nan)
    idx = np.flatnonzero(np.isfinite(si))
    if idx.size:
        lead = SEP3_SI_LWR_H + SEP3_MARGIN_H
        anchor = si[idx].astype(np.int64) - lead
        moved = stays[idx].copy()
        moved.obs["visit_start_datetime"] = (pd.to_datetime(moved.obs["visit_start_datetime"])
                                             + pd.to_timedelta(anchor, unit="h").to_numpy())
        moved.obs["origin_h"] = anchor.astype(float)
        # the grid starts at the anchor, so a position is the hours from it
        moved.obs["suspected_infection"] = np.full(idx.size, float(lead))
        grid = sofa_object(con, moved, ds, np.asarray(weight)[idx], 0, lead + SEP3_SI_UPR_H + 1)
        ep.tl.sepsis3(grid, infection_key="suspected_infection", before=SEP3_SI_LWR_H, after=SEP3_SI_UPR_H)
        ok = grid.obs["sepsis3"].to_numpy(dtype=bool)
        onset = grid.obs["sepsis3_onset"].to_numpy(dtype=float)
        has[idx[ok]] = True
        onset_h[idx[ok]] = anchor[ok] + onset[ok].astype(np.int64)
    stays.obs["sepsis3"] = has
    stays.obs["sepsis3_onset"] = pd.to_timedelta(onset_h, unit="h")


def infection_object(con, stays: ed.EHRData, ds: str, n_hours: int) -> ed.EHRData:
    """The antibiotic and culture rows of the stays `stays` (without variables) on hours [0, n_hours) from their
    start, from the tables of `LI.infection_tables`, in a dense layer `values`: `abx`, the longest duration (hours,
    floored) of the administrations starting in each hour, and on miiv `sample`, 1 in each hour with a sample."""
    common = dict(backend_handle=con, layer=ED.LAYER, interval_length_number=1, interval_length_unit="h",
                  time_precision="datetime", clip_to_end_date=False, num_intervals=n_hours)
    abx = ed.io.omop.setup_interval_variables(stays, data_tables=[LI.ABX_TABLE],
                                              data_field_to_keep={LI.ABX_TABLE: "dur_floor"},
                                              aggregation_strategy="max", keep_date="start", **common)
    assert abx.n_vars == 1, "the antibiotic rows are one concept"
    layers, names = [abx.layers[ED.LAYER]], ["abx"]
    if ds == "miiv":
        samples = ed.io.omop.setup_variables(stays, data_tables=[LI.SPECIMEN_TABLE],
                                             data_field_to_keep={LI.SPECIMEN_TABLE: "is_present"},
                                             concept_ids={"sample": [0]}, aggregation_strategy="max", **common)
        layers.append(samples.layers[ED.LAYER])
        names.append("sample")
    return ed.EHRData(obs=abx.obs, var=pd.DataFrame(index=pd.Index(names, dtype=str)), tem=abx.tem,
                      uns=dict(abx.uns), layers={ED.LAYER: np.concatenate(layers, axis=1)})


def _first_infection(edata: ed.EHRData, ds: str) -> np.ndarray:
    """The first suspected infection of each stay of `infection_object`, in hours from its start, NaN if none:
    ricu-extensions' `abx_cont` (ep.tl.mark_sustained_events on the starts of the administrations, each lasting
    its `abx`, up to the death in obs `death`, hours from the start), on miiv with a culture close in time (ricu's
    `si_and`, ep.tl.mark_paired_events). The object's hours start at 0, so positions are hours."""
    ep.tl.mark_events(edata, "abx", layer=ED.LAYER, key_added="abx_starts")
    ep.tl.mark_sustained_events(edata, "abx_starts", duration=ABX_WIN_H, max_gap=ABX_MAX_GAP_H, lengths="abx",
                                end_key="death", layer=ED.LAYER, key_added="abx_cont")
    key = "abx_cont"
    if ds == "miiv":
        ep.tl.mark_events(edata, "sample", layer=ED.LAYER, key_added="samples")
        ep.tl.mark_paired_events(edata, "abx_cont", "samples", a_to_b=SI_ABX_TO_SAMPLE_H,
                                 b_to_a=SI_SAMPLE_TO_ABX_H, layer=ED.LAYER, key_added="si")
        key = "si"
    return edata.obs[f"{key}_first"].to_numpy(dtype=float)


def _suspected_infection(con, ds, stays: ed.EHRData, death_h) -> None:
    """The first suspected infection of the stays `stays` (without variables, in the views' scope) from their
    antibiotic rows and, on miiv, culture rows at all times, so outside the build's label grid. Each stay
    gets a grid from its first to its last row, by a copy of its obs whose start is moved to its first row, and the
    stays run in chunks of similar grid length and at most SI_MAX_CELLS stays x hours. ricu floors the death time
    `death_h` (hours from the start, NaN if none) to the hour, as the starts of the administrations.
    Writes to obs `si_first`, a time difference from the start, NaT if none, and for the diagnostics `n_abx` and
    `abx_first` (hours from the start)."""
    rows = LI.infection_tables(con, OMOP, ds, stays.obs)
    first, last = rows["first"].to_numpy(), rows["last"].to_numpy()
    idx = np.flatnonzero((rows["n_abx"] > 0).to_numpy() & np.isfinite(first))
    n_hours = np.zeros(stays.n_obs, dtype=np.int64)
    n_hours[idx] = np.maximum(last[idx] - first[idx] + 1, 2)   # two hours at least, for ehrapy's regular grid
    death = np.floor(np.asarray(death_h, dtype=float))
    si_first = np.full(stays.n_obs, np.nan)
    try:
        order = idx[np.argsort(n_hours[idx], kind="stable")]
        k = 0
        while k < len(order):
            width = n_hours[order[k:]]
            m = max(1, int(np.searchsorted(np.arange(1, len(width) + 1) * width, SI_MAX_CELLS, side="right")))
            chunk, k = order[k:k + m], k + m
            part = stays[chunk].copy()
            part.obs["visit_start_datetime"] = (pd.to_datetime(part.obs["visit_start_datetime"])
                                                + pd.to_timedelta(first[chunk], unit="h").to_numpy())
            edata = infection_object(con, part, ds, int(width[m - 1]))
            edata.obs["death"] = death[chunk] - first[chunk]
            si_first[chunk] = first[chunk] + _first_infection(edata, ds)
    finally:
        for table in (LI.ABX_TABLE, LI.SPECIMEN_TABLE):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    stays.obs["si_first"] = pd.to_timedelta(si_first, unit="h")
    stays.obs["n_abx"] = rows["n_abx"].to_numpy()
    stays.obs["abx_first"] = rows["abx_first"].to_numpy()


# The obs columns of `_sepsis_onset`
SEPSIS_OBS = ("si_first", "n_abx", "abx_first", "sepsis3", "sepsis3_onset")


def _sepsis_onset(con, ds, stays: ed.EHRData, weight) -> ed.EHRData:
    """sep3_alt for the stays `stays` (without variables), written to their obs (SEPSIS_OBS): the suspected infection
    from all antibiotic and culture rows (`_suspected_infection`), the onset from the SOFA around the first one
    (`_sep3_at_si`). Returns `stays`."""
    _, death_h = _icu_death(stays.obs, _person_visits(con, stays.obs))
    _suspected_infection(con, ds, stays, death_h)
    _sep3_at_si(con, ds, stays, weight)
    return stays


def _write_diag(out_dir, name, df):
    """Per-stay label ingredients for aggregate comparison with ricu (COHORT_DIAG=1 only)."""
    if DIAG:
        to_parquet_atomic(df, f"{out_dir}/diag_{name}.parquet")


AKI_EXT_CHUNK = 500
AKI_EXT_MAX_CELLS = 100_000_000


def urine_rate(con, stays: ed.EHRData, ds: str, T: int) -> np.ndarray:
    """ricu's urine rate, KDIGO's urine input, of the stays `stays` on hours [0, T) from their start, NaN in hours
    without a row, from the urine rows as ehrdata sets them up (the concept set and bounds of the variable `urine`):
    on miiv and eicu per minute from 24 h before the start, where an earlier row still divides, and then
    `LI.ricu_urine_rate`; on hirid, which records the rate, the largest row of an hour."""
    ids, _, bounds, unit = ED.dynamic_set(ds, "urine")
    if ds == "hirid":
        x = ed.io.omop.setup_variables(stays, backend_handle=con, layer=ED.LAYER, sparse=True,
                                       **ED.setup_arguments({"urine": (ids, "max", bounds, unit)}, 0, T))
        return x.layers[ED.LAYER][:, 0, :].todense()
    m_lo = -60 * LI.URINE_RATE_MAX_GAP_H
    top, count = (ed.io.omop.setup_variables(stays, backend_handle=con, layer=ED.LAYER, sparse=True,
                                             **ED.setup_arguments({"urine": (ids, how, bounds, unit)}, m_lo, 60 * T, "min"))
                  .layers[ED.LAYER] for how in ("max", "count"))
    return LI.ricu_urine_rate(top, count, m_lo, T)


def kdigo_object(con, stays: ed.EHRData, ds: str, h_lo: int, T: int, rate,
                 edata: ed.EHRData | None = None) -> ed.EHRData:
    """The KDIGO inputs of the stays `stays` (without variables) on hours [h_lo, T) from ICU admission, h_lo <= 0:
    their `crea` (of `edata`, the build's EHRData on these hours, else set up here) and ricu's urine rate `rate`
    (stays x hours [0, T), `urine_rate`) as `urine_rate`. The urine rate stays empty before admission, where
    KDIGO's urine windows never reach."""
    if edata is None:
        edata = ed.io.omop.setup_variables(stays, backend_handle=con, layer=ED.LAYER, sparse=True,
                                           **ED.setup_arguments({"crea": ED.dynamic_set(ds, "crea")}, h_lo, T))
    assert ED.hours(edata) == (h_lo, T), "the object must cover [h_lo, T)"
    r, t = np.nonzero(np.isfinite(rate))
    x = sparse.COO(np.stack([r, np.zeros_like(r), t - h_lo]), rate[r, t], shape=(edata.n_obs, 1, T - h_lo),
                   fill_value=np.nan)
    return ED.add_variables(edata, ["urine_rate"], x)


def score_kdigo(edata: ed.EHRData, weight) -> None:
    """ricu's KDIGO stage (ep.tl.kdigo) on `crea` and `urine_rate` of `kdigo_object`, read from `label_view`:
    `obsm["kdigo"]`, `obsm["kdigo_creatinine"]` and `obsm["kdigo_urine"]`, and in obs the first hour at stage 1 or
    more of each, as time differences (`aki_first`, `aki_creatinine_first`, `aki_urine_first`, ep.tl.mark_events).

    By creatinine, with the lowest value of the trailing 48 h and of the trailing 7 days as baselines: stage 1 at
    0.3 mg/dL above the first or 1.5 times the second, stage 2 at twice the second, stage 3 at three times the second
    or 4 mg/dL with a stage-1 rise. By urine, from ICU admission and at hours with a urine rate: below 0.5 mL/kg over
    6 h (1), 0.5 mL/kg over 12 h (2), 0.3 mL/kg over 24 h or none over 12 h (3). Dialysis, a stage 3 of KDIGO, ricu
    does not use."""
    # RICU CHOICE (RICU.md, "Where we follow ricu"): a stay without a weight keeps its urine in mL, where
    # ricu's join finds no weight; a weight of 1 kg gives that
    edata.obs["kdigo_weight"] = np.where(np.isfinite(weight) & (weight > 0), weight, 1.0)
    # the tools take positions on the object's hours, which start at h_lo; the ICU admission is hour 0
    h_lo, _ = ED.hours(edata)
    edata.obs["icu_admission"] = np.full(edata.n_obs, float(-h_lo))
    ED.label_view(edata, exact=COMPUTED_INPUTS)
    ep.tl.kdigo(edata, layer=ED.LABEL_VIEW, creatinine="crea", urine="urine_rate", weight_key="kdigo_weight",
                rise_window=48, baseline_window=168, urine_criteria=KDIGO_URINE_CRITERIA,
                anuria_window=KDIGO_ANURIA_H, urine_start_key="icu_admission", tolerance=KDIGO_TOLERANCE)
    for key, added in (("kdigo", "aki"), ("kdigo_creatinine", "aki_creatinine"), ("kdigo_urine", "aki_urine")):
        # AKI is a stage of AKI_THRESHOLD (1) or more, which ignore_zero marks
        ep.tl.mark_events(edata, key, ignore_zero=True, key_added=added)
        for which in ("first", "last"):
            edata.obs[f"{added}_{which}"] = pd.to_timedelta(edata.obs[f"{added}_{which}"] + h_lo, unit="h")


def crea_baseline(edata: ed.EHRData) -> None:
    """The creatinine baseline of YAIB's aki exclusion 8 from `crea` of a `kdigo_object`, in obs: `crea_baseline`,
    the lowest creatinine before admission or the first in the ICU, and `crea_first_icu`; for the diagnostics
    `crea_hours_pre` and `crea_hours_icu`, the hours with creatinine before and from admission."""
    h_lo, _ = ED.hours(edata)
    crea = ED.variable(edata, "crea", rounded=True)
    pre, icu = crea[:, :-h_lo], crea[:, -h_lo:]
    has_icu = ~np.isnan(icu).all(1)
    first_icu = np.where(has_icu, icu[np.arange(len(icu)), np.argmax(~np.isnan(icu), axis=1)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        edata.obs["crea_baseline"] = np.fmin(np.nanmin(pre, axis=1), first_icu)
    edata.obs["crea_first_icu"] = first_icu
    edata.obs["crea_hours_pre"] = (~np.isnan(pre)).sum(1)
    edata.obs["crea_hours_icu"] = (~np.isnan(icu)).sum(1)


# The obs columns of `score_kdigo` and `crea_baseline` that `_aki_onset` gives over the whole creatinine series
AKI_OBS = ("aki_first", "aki_creatinine_first", "aki_urine_first", "crea_baseline", "crea_first_icu",
           "crea_hours_pre", "crea_hours_icu")


def _aki_onset(con, ds, stays: ed.EHRData, weight, T, edata: ed.EHRData | None = None) -> ed.EHRData:
    """KDIGO on the whole creatinine series, as ricu, for the stays `stays` (without variables): on `edata`, their
    build's EHRData on [-LABEL_PRE_H, T) (set up here without it), and for stays with creatinine before that on
    objects from their own first value (outside the build's label grid; in chunks of similar start and
    bounded size, as eicu has offsets of decades), so that the 48 h / 7 d minima, the onset and the baseline see
    every earlier value. Returns the object on [-LABEL_PRE_H, T) with the KDIGO inputs, its stages on its hours in
    obsm, and in obs AKI_OBS over the whole series (`score_kdigo`, `crea_baseline`)."""
    first_h = ED.first_last_hours(stays, con, {"crea": ED.dynamic_set(ds, "crea")})["crea_first"].to_numpy()
    rate = urine_rate(con, stays, ds, T)
    edata = kdigo_object(con, stays, ds, -LABEL_PRE_H, T, rate, edata)
    score_kdigo(edata, weight)
    crea_baseline(edata)
    columns = {c: edata.obs[c].to_numpy().copy() for c in AKI_OBS}
    ext = np.flatnonzero(np.isfinite(first_h) & (first_h < -LABEL_PRE_H))
    ext = ext[np.argsort(first_h[ext], kind="stable")]
    k = 0
    while k < len(ext):
        # the first stay of a chunk has the earliest start; stop where the grid gets too large
        m = max(1, min(AKI_EXT_CHUNK, AKI_EXT_MAX_CELLS // (T - int(first_h[ext[k]]))))
        g = ext[k:k + m]
        part = kdigo_object(con, stays[g], ds, int(first_h[g[0]]), T, rate[g])
        score_kdigo(part, np.asarray(weight)[g])
        crea_baseline(part)
        for c in AKI_OBS:   # times from each stay's start, whatever the hours of the object
            columns[c][g] = part.obs[c].to_numpy()
        k += m
    for c in AKI_OBS:
        edata.obs[c] = columns[c]
    if len(ext):
        print(f"[aki] {len(ext)} stays with creatinine before -{LABEL_PRE_H} h, earliest "
              f"{int(np.nanmin(first_h))} h", flush=True)
    return edata


def _person_visits(con, obs: pd.DataFrame) -> pd.DataFrame:
    """Every visit in VISIT_OCCURRENCE of the persons in `obs`, also those outside the cohort."""
    persons = pd.DataFrame({"person_id": pd.unique(obs["person_id"])})  # noqa: F841 (read by duckdb)
    return con.execute(f"""
        SELECT visit_occurrence_id, person_id, visit_start_datetime, visit_end_datetime
        FROM parquet_scan('{OMOP}/VISIT_OCCURRENCE.parquet')
        WHERE person_id IN (SELECT person_id FROM persons)""").df()


def _icu_death(obs: pd.DataFrame, visits: pd.DataFrame | None = None):
    """(died, death_h) per row of a stays' `.obs`: the ICU death flag (discharged to 'Patient died')
    and the death time in hours from the stay's start (DEATH.death_datetime, may lie after the stay).

    DEATH has one row per person. Its time belongs to the person's visit nearest to it among
    `visits` (all the person's visits, `_person_visits`; default `obs`); a flagged stay that is not
    that visit (a person with two flagged stays) has no death time (NaN).

    The flag alone would do on miiv and hirid, where every flagged stay is its person's visit nearest
    to the death. eicu has a few persons with two stays flagged as ICU deaths (fewer than 10 stays in
    mortality24), and DEATH's one time can belong to only one of them."""
    died = (obs["discharged_to_concept_id"] == LI.ICU_DEATH_CONCEPT_ID).fillna(False).to_numpy(dtype=bool)
    if "death_datetime" not in obs:
        return died, np.full(len(obs), np.nan)
    visits = obs if visits is None else visits
    deaths = obs.drop_duplicates("person_id").set_index("person_id")["death_datetime"]  # one DEATH row per person
    start = pd.to_datetime(visits["visit_start_datetime"])
    end = pd.to_datetime(visits["visit_end_datetime"]).fillna(start)
    death = pd.to_datetime(visits["person_id"].map(deaths))
    distance = np.where(death.isna(), np.nan,
                        np.where(death < start, (start - death).dt.total_seconds(),
                                 np.where(death > end, (death - end).dt.total_seconds(), 0.0)))
    near = pd.DataFrame({"person": visits["person_id"].to_numpy(), "d": distance, "start": start.to_numpy(),
                         "voi": visits["visit_occurrence_id"].to_numpy()})
    nearest = set(near.dropna(subset=["d"]).sort_values(["person", "d", "start", "voi"])
                  .drop_duplicates("person")["voi"].astype(np.int64))
    is_nearest = obs["visit_occurrence_id"].astype(np.int64).isin(nearest).to_numpy()
    death_h = ((pd.to_datetime(obs["death_datetime"]) - pd.to_datetime(obs["visit_start_datetime"]))
               .dt.total_seconds() / 3600.0).to_numpy()
    return died, np.where(died & is_nearest, death_h, np.nan)


def _hospitals(con, vois) -> np.ndarray:
    """The hospital of each visit of the scope: CARE_SITE.care_site_source_value of its care site.
    Read from the visit table: in `.obs`, `care_site_id` is the person's (setup_obs joins PERSON first)."""
    df = con.execute(f"""
        SELECT c.visit_occurrence_id AS voi, cs.care_site_source_value AS hosp FROM _cohort_visits c
        LEFT JOIN parquet_scan('{OMOP}/CARE_SITE.parquet') cs ON cs.care_site_id = c.care_site_id""").df()
    return pd.Series(np.asarray(vois, dtype=np.int64)).map(df.set_index("voi")["hosp"]).to_numpy(dtype=object)


HOSP_CHUNK = 100_000


def _hospital_exclusion(con, ds, voi, onset_fn, out_dir=None, task=""):
    """eicu: stays in hospitals where no stay has the label (aki.R / sepsis.R exclusion 6).
    As ricu, the prevalence is over all eicu stays, so the label is computed for every stay of the views' scope
    (`_scope`: every eicu stay with an end time, which in our CDM is every eicu stay), `onset_fn(stays, weight)`
    on their EHRData without variables in chunks of at most HOSP_CHUNK stays; a hospital without a positive stay
    is a group-by on their `.obs`."""
    if ds != "eicu":
        return np.zeros(len(voi), dtype=bool)
    stays = ED.setup_stays(con)
    everyone = stays.obs
    voi_all = everyone["voi"].to_numpy()
    _, w_all = _statics_height_weight(con, f"{CACHE}/measurement_{ds}.parquet", voi_all, ds)
    positive = np.zeros(stays.n_obs, dtype=bool)
    for rows in np.array_split(np.arange(stays.n_obs), max(1, math.ceil(stays.n_obs / HOSP_CHUNK))):
        positive[rows] = onset_fn(stays[rows].copy(), w_all[rows])
    everyone["positive"] = positive
    everyone["hospital"] = _hospitals(con, voi_all)
    has_positive = everyone.groupby("hospital")["positive"].any()
    everyone["hospital_without_positive"] = everyone["hospital"].map(~has_positive).fillna(False).astype(bool)
    if out_dir is not None:
        _write_diag(out_dir, f"hosp_{task}", pd.DataFrame({
            "stay_id": everyone["stay_id"].astype("int64").to_numpy(), "hosp": everyone["hospital"].to_numpy(),
            "positive": everyone["positive"].to_numpy(), "zero_site": everyone["hospital_without_positive"].to_numpy()}))
    excl = pd.Series(everyone["hospital_without_positive"].to_numpy(), index=voi_all).reindex(
        np.asarray(voi, dtype=np.int64)).fillna(False).to_numpy(dtype=bool)
    print(f"[hosp] {everyone['hospital'].nunique()} hospitals over {len(everyone)} stays, "
          f"{int(everyone['positive'].sum())} positive; {int(everyone['hospital_without_positive'].sum())} stays in "
          f"hospitals without a positive, {int(excl.sum())} of them in this cohort", flush=True)
    return excl


def base_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """base_cohort.R exclusions 1-5 (cohort_filters.BASE_CRITERIA) on a frame with los_hours,
    n_hours, max_empty_run, age; a missing value is not an exclusion, as in ricu."""
    stays = ed.EHRData(obs=df.reset_index(drop=True).set_axis(np.arange(len(df)).astype(str)))
    stays.obs["_row"] = np.arange(len(df))
    for criterion in CF.BASE_CRITERIA:
        ep.pp.filter_observations(stays, query=criterion)
    return df.iloc[stays.obs["_row"].to_numpy()]


def _stay_ages(con, obs: pd.DataFrame) -> np.ndarray:
    """The age at each stay: its OBSERVATION age row; a stay without one (a CDM without age rows)
    from the year of birth. A missing age stays missing, as in ricu."""
    ages = con.execute(
        f"""
        SELECT visit_occurrence_id AS voi, max(value_as_number) AS age FROM observation
        WHERE observation_concept_id = {AGE_CONCEPT_ID}
          AND visit_occurrence_id IN (SELECT visit_occurrence_id FROM _cohort_visits)
        GROUP BY 1
        """
    ).df().set_index("voi")["age"]
    from_row = obs["voi"].map(ages).to_numpy(dtype=float)
    from_birth = (pd.to_datetime(obs["visit_start_datetime"]).dt.year
                  - pd.to_numeric(obs["year_of_birth"], errors="coerce")).to_numpy(dtype=float)
    return np.where(obs["voi"].isin(ages.index).to_numpy(), from_row, from_birth)


def _fluid_balance_hours(con, fluid_cache: str, obs: pd.DataFrame) -> pd.DataFrame:
    """hirid: the hours (row of `obs`, h) with a de-cumulated fluid balance row up to URINE_MAX_ML,
    which ricu's hirid `urine` counts in the base exclusions."""
    df = con.execute(
        f"""
        WITH r AS (
          SELECT m.visit_occurrence_id AS voi, m.measurement_datetime AS t,
                 -- rows at the same time in the order of their value, as `LI.hirid_fluid_urine`
                 greatest(0, m.value_as_number - coalesce(lag(m.value_as_number) OVER (
                     PARTITION BY m.visit_occurrence_id ORDER BY m.measurement_datetime, m.value_as_number), 0)) AS u
          FROM parquet_scan('{fluid_cache}') m
          WHERE m.measurement_source_value = '{FLUID_BALANCE_SOURCE}' AND m.value_as_number IS NOT NULL)
        SELECT DISTINCT r.voi, CAST(floor(date_diff('second', c.visit_start_datetime, r.t) / 3600.0) AS INTEGER) AS h
        FROM r JOIN _cohort_visits c ON r.voi = c.visit_occurrence_id WHERE r.u <= {LI.URINE_MAX_ML}
        """
    ).df()
    row = pd.Series(np.arange(len(obs)), index=obs["voi"].to_numpy()).reindex(df["voi"].to_numpy())
    ok = row.notna().to_numpy()
    return pd.DataFrame({"row": row.to_numpy()[ok].astype(np.int64), "h": df["h"].to_numpy()[ok].astype(np.int64)})


def _longest_empty_runs(cells: pd.DataFrame, los_hours: np.ndarray) -> np.ndarray:
    """ricu's longest run of empty hourly bins on 0..min(floor(LoS), BASE_MAX_H) per stay
    (base_cohort.R `longest_rle`), from the observed (row, h) cells, with sentinels at -1 and
    end + 1; NaN for a stay without a valid length of stay."""
    n = len(los_hours)
    valid = np.nan_to_num(los_hours, nan=-1.0) >= 0
    end_h = np.where(valid, np.minimum(np.floor(np.nan_to_num(los_hours, nan=0.0)), BASE_MAX_H), -1).astype(np.int64)
    rows = np.flatnonzero(valid)
    inside = cells[valid[cells["row"].to_numpy()]]
    inside = inside[(inside["h"] >= 0) & (inside["h"].to_numpy() <= end_h[inside["row"].to_numpy()])]
    g = pd.DataFrame({"row": np.concatenate([inside["row"].to_numpy(), rows, rows]),
                      "h": np.concatenate([inside["h"].to_numpy(), np.full(len(rows), -1), end_h[rows] + 1])})
    g = g.sort_values(["row", "h"])
    run = g.groupby("row")["h"].diff() - 1
    out = np.full(n, np.nan)
    longest = run.groupby(g["row"]).max()
    out[longest.index.to_numpy()] = longest.to_numpy()
    return out


def base_candidates(con, ds: str, fluid_cache: str | None = None) -> ed.EHRData:
    """The scoped stays as an EHRData whose `.obs` carries ricu's base-cohort quantities: los_hours,
    age, gender, sex, n_hours, max_empty_run. No exclusion is applied.

    Exclusions 3 and 4 count the hours with a row of the 48 YAIB variables inside ricu's bounds, at
    any time (ricu's `load_step(dict[dynamic_vars])`, summary "count"). With `fluid_cache` (hirid)
    the hours of de-cumulated fluid balance count as well, since ricu's hirid `urine` is that concept.
    """
    stays = ED.setup_stays(con)
    obs = stays.obs
    obs["age"] = _stay_ages(con, obs)
    obs["gender"] = pd.to_numeric(obs["gender_concept_id"], errors="coerce").astype("float64")
    obs["sex"] = [GENDER_CONCEPT_TO_SEX.get(int(g)) if pd.notna(g) else None for g in obs["gender"]]
    cells = ED.observed_hours(stays, con, ds, YAIB_DYNAMIC_VARS)
    if fluid_cache:
        cells = pd.concat([cells, _fluid_balance_hours(con, fluid_cache, obs)], ignore_index=True).drop_duplicates()
    n_hours = np.bincount(cells["row"].to_numpy(), minlength=stays.n_obs).astype(float)
    obs["n_hours"] = np.where(n_hours > 0, n_hours, np.nan)   # a stay without rows is not counted (ricu)
    obs["max_empty_run"] = _longest_empty_runs(cells, obs["los_hours"].to_numpy(dtype=float))
    n_ex = {
        "los<0": int((obs["los_hours"] < 0).sum()),
        f"los<{MIN_LOS_H}h": int((obs["los_hours"] < MIN_LOS_H).sum()),
        f"<{MIN_MEAS} hours": int((obs["n_hours"] < MIN_MEAS).sum()),
        f"empty run>{MAX_GAP_H}h": int((obs["max_empty_run"] > MAX_GAP_H).sum()),
        "age<18": int((obs["age"] < 18).sum()),
    }
    print(f"[base] candidates={stays.n_obs}; would be excluded (overlapping) "
          + ", ".join(f"{k}={v}" for k, v in n_ex.items()), flush=True)
    print(f"[base] missing age {int(np.isnan(obs['age']).sum())} / sex "
          f"{int((obs['gender'].isna() | (obs['gender'] == 0)).sum())} of {stays.n_obs} candidates", flush=True)
    return stays


def _scope(con, cache_path: str, ds: str, vois) -> int:
    """Scope the views (ehrdata_dynamics.scope_cohort) to the stays `vois`; returns their number. The
    `measurement` view holds the label inputs' concepts too, without the ETL's derived GCS total."""
    # registered under a name of its own: the table cohort_voi of an earlier call would shadow a frame of that name
    con.register("_cohort_voi_frame", pd.DataFrame({"voi": np.asarray(vois, dtype=np.int64)}))
    try:
        con.execute("CREATE OR REPLACE TABLE cohort_voi AS SELECT voi FROM _cohort_voi_frame")
    finally:
        con.unregister("_cohort_voi_frame")
    for view in ("measurement", "person", "visit_occurrence"):
        con.execute(f"DROP VIEW IF EXISTS {view}")
    own, _ = label_sets(ds)
    return ED.scope_cohort(con, OMOP, cache_path, ds, voi_table="cohort_voi",
                           label_concepts=[c for ids, *_ in own.values() for c in ids],
                           drop_source_values=(LI.DERIVED_GCS_TOTAL,))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="miiv", choices=list(DATASET_PREFIX))
    ap.add_argument("--task", default="los", choices=list(TASK_HORIZON))
    ap.add_argument("--mem", default="40GB")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=OUT, help="output root (default: $FRAMES_OUT or <cdm>/../cohorts/frames)")
    args = ap.parse_args()
    require_cdm()
    ds, task = args.dataset, args.task

    prefix = DATASET_PREFIX[ds]
    # T is the GRID LENGTH, i.e. the number of hourly bins, which ricu makes horizon+1 because
    # `expand` is inclusive at both ends. Measured in the reference cohorts: rows per stay ==
    # floor(LoS) + 1, capped at horizon + 1, and max time == floor(LoS) hours.
    T = N_BINS[task]
    # An experiment must never overwrite the locked build: RICU_HIRID_URINE and CAI_CONVERT write to
    # a suffixed tree.
    suffix = (("_ricuurine" if RICU_HIRID_URINE and ds == "hirid" else "")
              + ("_caiconverted" if CAI_CONVERT and ds == "eicu" else ""))
    out_dir = f"{args.out}{suffix}/{task}/{ds}"
    os.makedirs(out_dir, exist_ok=True)
    duck_tmp = make_job_tmp(DUCK_TMP_BASE, ds, task)
    print(f"[env] ehrdata={ed.__version__} duckdb={duckdb.__version__} ds={ds} task={task} T={T} "
          f"cai={'converted' if CAI_CONVERT else 'as ricu'} ehrapy={ep.__file__} out={out_dir}", flush=True)
    tools = {"sepsis": ("sofa", "sepsis3", "mark_events", "mark_sustained_events", "mark_paired_events"),
             "aki": ("kdigo", "mark_events")}.get(task, ())
    if not all(hasattr(ep.tl, tool) for tool in tools):
        raise SystemExit(f"{task} needs an ehrapy with tl.{' and tl.'.join(tools)}, as the environment of "
                         f"../pyproject.toml pins, not {ep.__file__}")

    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{args.mem}'")
    con.execute(f"SET temp_directory='{duck_tmp}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(
        f"CREATE VIEW visit_occurrence AS SELECT * FROM parquet_scan('{OMOP}/VISIT_OCCURRENCE.parquet') "
        f"WHERE visit_source_value LIKE '{prefix}-%'"
    )
    con.execute(f"CREATE VIEW observation AS SELECT * FROM parquet_scan('{OMOP}/OBSERVATION.parquet')")

    # 1. the candidate stays: every stay of the dataset with an end time (ricu's stay_windows drops
    #    the others), one observation each (ehrdata), with ricu's per-stay quantities in `.obs`. The
    #    density exclusions count the 48 variables in YAIB units after ricu's bounds; ricu's hirid
    #    urine is fluid balance, so its rows count as well. The views stay scoped to the candidates:
    #    ehrdata gives a stay only the rows linked to it, and eicu's hospital exclusion reads every stay.
    cache_path = f"{CACHE}/measurement_{ds}.parquet"
    cand_voi = con.execute("SELECT visit_occurrence_id FROM visit_occurrence "
                           "WHERE visit_end_datetime IS NOT NULL").df()["visit_occurrence_id"]
    n_vis = _scope(con, cache_path, ds, cand_voi)
    stays = base_candidates(con, ds, cache_path if ds == "hirid" else None)

    # 2. the base exclusions as tracked filters (ehrapy)
    assert stays.obs["stay_id"].notna().all() and stays.obs["stay_id"].is_unique, "stay ids must be unique"
    stays = stays[np.argsort(stays.obs["stay_id"].to_numpy(dtype=np.int64), kind="stable")].copy()
    tracker = CF.tracker(stays)
    n_candidates = stays.n_obs
    steps = CF.apply(stays, CF.BASE_CRITERIA, tracker)
    print(f"[base] candidates={n_candidates} -> base_cohort={stays.n_obs}", flush=True)

    # 3. the build's EHRData of the base cohort, through ehrdata's OMOP IO: the dynamics and the label
    #    inputs, rows linked to their stay by visit_occurrence_id, concept sets per variable, ricu's
    #    bounds and hourly aggregation (ehrdata_dynamics.py); X holds the 48 dynamics on [0, T) for
    #    the frames.
    print(f"[views] cohort visits={n_vis}", flush=True)
    assert_cache_covers_targets(con, cache_path)
    multi = {v: ids for v, ids in VAR_CONCEPTS.items() if len(ids) > 1}
    print(f"[map] multi-pathway pooled = {sorted(multi)}", flush=True)
    h_lo, h_hi = object_hours(task, T)
    edata = build_object(con, stays, ds, h_lo, h_hi)
    assert list(edata.var_names[:len(YAIB_DYNAMIC_VARS)]) == list(YAIB_DYNAMIC_VARS), "variables out of YAIB order"
    X = edata[:, :len(YAIB_DYNAMIC_VARS), -h_lo:T - h_lo].layers[ED.LAYER]
    aligned = edata.obs
    stay_obs = aligned["stay_id"].to_numpy(dtype=np.int64)
    if RICU_HIRID_URINE and ds == "hirid":
        # The experiment only: swap real hourly urine for ricu's de-cumulated fluid balance.
        j = YAIB_DYNAMIC_VARS.index("urine")
        before = int((X.coords[1] == j).sum())
        X = ED.replace_variable(X, j, _ricu_hirid_urine(
            con, f"{CACHE}/measurement_{ds}.parquet", aligned["voi"].to_numpy(), T))
        after = int((X.coords[1] == j).sum())
        print(f"[ricu urine] hirid urine swapped to ricu's de-cumulated fluid balance: "
              f"{before:,} -> {after:,} non-null cells", flush=True)
    los_h = aligned["los_hours"].to_numpy()        # visit end - visit start (ED.setup_stays)
    age_arr = aligned["age"].to_numpy()
    gender_arr = aligned["gender"].to_numpy()
    height_arr, weight_arr = _statics_height_weight(
        con, f"{CACHE}/measurement_{ds}.parquet", aligned["voi"].to_numpy(), ds)
    print(f"[dyn] stays={len(stay_obs)} observed cells={X.nnz:,}", flush=True)

    # 4. per-task label + task-specific exclusions (obs columns + criteria) -> per-stay grid length, outc pieces
    n = len(stay_obs)
    dynamic = task in ("los", "aki", "sepsis")
    static_label = None
    label_grid = None        # (n, T) label per stay and hour of the dynamic tasks
    if task == "los":
        Lgrid = np.minimum(np.floor(los_h) + 1, T).astype(int)   # ricu: floor(LoS)+1 rows
        keep = Lgrid > 0
        task_obs, criteria = {}, ["grid_hours > 0"]
        label_grid = los_label(los_h[:, None], np.arange(T)[None, :])   # floor(LoS) - t, capped 168
    elif task == "kidney_function":
        # hourly median creatinine (mg/dL, bounded) at hours 25..48 of the object, then the median over hours
        lo, hi = (h - h_lo for h in KIDNEY_FUNCTION_HOURS)
        lbl = kidney_function_label(ED.variable(edata, "crea", rounded=True)[:, lo:hi])
        keep = ~(los_h < 48) & np.isfinite(lbl)                   # excl LoS<48h, no crea in (24,48]
        task_obs = {"kidney_label": lbl}
        criteria = ["~(los_hours < 48)", "kidney_label.notna()"]
        Lgrid = np.full(n, T, dtype=int)
        static_label = lbl
    elif task == "aki":
        # aki.R + ricu-extensions callback-kdigo.R:
        #  - KDIGO on the whole creatinine series (hourly median, bounded), including all
        #    pre-ICU values, so baselines and onsets can be pre-ICU;
        #  - first onset only (summary_step "first"); stay truncated at onset + 6 h
        #    (stop_obs_at); symmetric +/-6 h outcome window;
        #  - excl7: first onset < 6 h, including negative onsets;
        #  - excl8: baseline = cummin of creatinine up to and including the first ICU value
        #    (baseline_candidates + summary "last"), excluded if > 4.
        voi = aligned["voi"].to_numpy()
        edata = _aki_onset(con, ds, stays, weight_arr, T, edata)
        aki = edata.obs
        onset_h = _obs_hours(aki, "aki_first")
        has_onset = np.isfinite(onset_h)
        first_onset = np.where(has_onset, onset_h, T).astype(np.int64)
        baseline = aki["crea_baseline"].to_numpy(dtype=float)
        # map_to_grid: an onset after the stay's grid (e.g. from rows after discharge) labels no hour
        L_stay = np.minimum(np.floor(los_h) + 1, T).astype(int)
        pt = np.zeros((n, T), dtype=bool)
        on_grid = has_onset & (first_onset >= 0) & (first_onset < L_stay)
        pt[np.where(on_grid)[0], first_onset[on_grid]] = True
        lab = pt.copy()
        for k in range(1, AKI_OUTCOME_WINDOW_H + 1):
            lab[:, :-k] |= pt[:, k:]        # align="left"  -> looks forward
            lab[:, k:] |= pt[:, :-k]        # align="right" -> looks backward
        excl7 = has_onset & (first_onset < AKI_OUTCOME_WINDOW_H)
        excl8 = np.isfinite(baseline) & (baseline > AKI_BASELINE_CREA_MAX)

        Lgrid = np.where(has_onset, np.minimum(L_stay, np.maximum(first_onset, -1) + AKI_OUTCOME_WINDOW_H + 1),
                         L_stay)                                         # ricu: floor(LoS)+1 rows
        excl6 = _hospital_exclusion(con, ds, voi, lambda s, w: _aki_onset(con, ds, s, w, T).obs["aki_first"].notna()
                                    .to_numpy(), out_dir, task)
        keep = (Lgrid > 0) & ~excl6 & ~excl7 & ~excl8
        task_obs = {"hospital_without_onset": excl6, "onset_before_6h": excl7, "baseline_crea_above_4": excl8}
        criteria = ["grid_hours > 0", "~hospital_without_onset", "~onset_before_6h", "~baseline_crea_above_4"]
        if DIAG:
            _write_diag(out_dir, "aki", pd.DataFrame({
                "stay_id": stay_obs, "has_onset": has_onset,
                "first_onset": np.where(has_onset, first_onset, np.nan),
                "crea_first": _obs_hours(aki, "aki_creatinine_first"), "urine_first": _obs_hours(aki, "aki_urine_first"),
                "baseline": baseline, "first_icu_crea": aki["crea_first_icu"].to_numpy(dtype=float), "excl6": excl6,
                "excl7": excl7, "excl8": excl8, "n_pre_crea_h": aki["crea_hours_pre"].to_numpy(),
                "n_icu_crea_h": aki["crea_hours_icu"].to_numpy(),
                "has_weight": np.isfinite(weight_arr), "keep": keep}))
        label_grid = lab.astype(np.float64)
        print(f"[aki] KDIGO>={AKI_THRESHOLD} in {int(has_onset.sum())}/{n} stays "
              f"({int((has_onset & (first_onset < 0)).sum())} with a pre-ICU onset); "
              f"excluded {int(excl7.sum())} (onset <{AKI_OUTCOME_WINDOW_H}h) + "
              f"{int(excl8.sum())} (baseline crea >{AKI_BASELINE_CREA_MAX}); "
              f"stays with pre-ICU crea {int((aki['crea_hours_pre'] > 0).sum())}", flush=True)

    elif task == "sepsis":
        # sepsis.R sep3_alt: SOFA by ep.tl.sofa on the build's EHRData (ricu's per-concept hourly
        # aggregation, worst value over [t-24, t], NA as 0, rows from the first component row at any
        # time); suspected infection = first abx_cont time (si_mode "abx" on eicu and hirid, "and"
        # with cultures on miiv); SOFA rise >= 2 within [si-48, si+24] by ep.tl.sepsis3, at any time
        # of the stay, so pre-ICU onsets survive and excl7 removes them.
        voi = aligned["voi"].to_numpy()
        _sepsis_onset(con, ds, stays, weight_arr)
        for column in SEPSIS_OBS:
            edata.obs[column] = stays.obs[column].to_numpy()
        sep = edata.obs
        has_onset = sep["sepsis3"].to_numpy(dtype=bool)
        first = np.where(has_onset, _obs_hours(sep, "sepsis3_onset"), 0).astype(np.int64)   # hours from admission
        si_h = _obs_hours(sep, "si_first")
        has_si = np.isfinite(si_h)

        Lgrid = np.minimum(np.floor(los_h) + 1, T).astype(int)        # ricu: floor(LoS)+1 rows
        # excl7 as sepsis.R: first onset < 6 h, including onsets before admission
        excl7 = has_onset & (first < SEPSIS_OUTCOME_WINDOW_H)
        on_grid = has_onset & (first >= 0) & (first < Lgrid)
        onset = np.zeros((n, T), dtype=bool)
        onset[np.where(on_grid)[0], first[on_grid]] = True
        lab = onset.copy()
        for k in range(1, SEPSIS_OUTCOME_WINDOW_H + 1):
            lab[:, :-k] |= onset[:, k:]
            lab[:, k:] |= onset[:, :-k]
        # stop_obs_at(outc, +6 h): the grid ends 6 h after the onset
        Lgrid = np.where(has_onset & (first >= 0),
                         np.minimum(Lgrid, first + SEPSIS_OUTCOME_WINDOW_H + 1), Lgrid)
        in_stay = np.arange(T)[None, :] < Lgrid[:, None]
        lab &= in_stay
        excl6 = _hospital_exclusion(con, ds, voi, lambda s, w: _sepsis_onset(con, ds, s, w).obs["sepsis3"]
                                    .to_numpy(dtype=bool), out_dir, task)
        keep = (Lgrid > 0) & ~excl6 & ~excl7
        task_obs = {"hospital_without_onset": excl6, "onset_before_6h": excl7}
        criteria = ["grid_hours > 0", "~hospital_without_onset", "~onset_before_6h"]
        label_grid = lab.astype(np.float64)
        n_pos = int((on_grid & keep).sum())
        if DIAG:
            _write_diag(out_dir, "sepsis", pd.DataFrame({
                "stay_id": stay_obs, "has_si": has_si, "si_first": si_h,
                "n_abx": sep["n_abx"].to_numpy(), "abx_first": sep["abx_first"].to_numpy(),
                "has_onset": has_onset, "first": np.where(has_onset, first, np.nan),
                "excl6": excl6, "excl7": excl7, "keep": keep}))
            # the SOFA on the object's hours from LABEL_PRE_H before admission, at the hours where it has a row
            sofa = add_sofa_inputs(con, edata, ds, weight_arr)
            score_sofa(sofa, ds)
            total = sofa.obsm["sofa"]
            vent = ~np.isnan(ED.variable(sofa, "vent"))
            r, c = np.nonzero(~np.isnan(total))
            h = pd.DataFrame({"stay_id": stay_obs[r], "time": c - LABEL_PRE_H, "sofa": total[r, c]})
            for k, organ in SOFA_ORGANS.items():
                h[k] = sofa.obsm[f"sofa_{organ}"][r, c]
            h["vent"] = vent[r, c]
            _write_diag(out_dir, "sofa", h)
            icu = total[:, LABEL_PRE_H:]
            sofa_in_stay = icu[in_stay & ~np.isnan(icu)]
            vaso_hours = sum((np.nan_to_num(ED.variable(sofa, k)[:, LABEL_PRE_H:]) > 0)[in_stay].sum() for k in LI.VASO)
            print(f"[sepsis] SOFA(in-stay rows) p50={float(np.median(sofa_in_stay)) if sofa_in_stay.size else float('nan'):.1f} "
                  f"max={float(np.nanmax(total)) if np.isfinite(total).any() else float('nan'):.0f}; "
                  f"vent {100*vent[:, LABEL_PRE_H:][in_stay].mean():.1f}% of in-stay hours; "
                  f"cardio rate hours {int(vaso_hours)}; "
                  f"gcs rows {int(np.isfinite(ED.variable(sofa, 'gcs')[:, LABEL_PRE_H:])[in_stay].sum())}", flush=True)
            del sofa
        print(f"[sepsis] excl7={int(excl7.sum())} (pre-ICU onsets {int((has_onset & (first < 0)).sum())}); "
              f"abx_cont {100*has_si.mean():.1f}% of stays (pre-ICU first SI {int((has_si & (si_h < 0)).sum())}); "
              f"excl6={int(excl6.sum())}; sep3 in {n_pos}/{n} stays ({100*n_pos/max(n,1):.1f}%)", flush=True)

    elif task == "mortality24":
        # ICU death flag on the stay, real death time; an unknown outcome counts as alive (ricu)
        died, dh = _icu_death(aligned, _person_visits(con, aligned))
        keep = ~(los_h < 30) & ~(died & (dh < 30))               # excl LoS<30h, died<30h
        task_obs = {"icu_death": died, "death_hours": dh}
        criteria = ["~(los_hours < 30)", "~(icu_death & (death_hours < 30))"]
        Lgrid = np.full(n, T, dtype=int)
        static_label = died.astype(np.int64)
    # the labels and the task exclusions on the stays' EHRData; the exclusions as tracked filters
    # after the base exclusions
    if dynamic:
        # on the object's hours [h_lo, h_hi); label_grid holds [0, T)
        edata.obsm["label"] = np.pad(label_grid, ((0, 0), (-h_lo, h_hi - T)), constant_values=np.nan)
    else:
        edata.obs["label"] = static_label
    edata.obs["grid_hours"] = Lgrid
    for column, values in task_obs.items():
        edata.obs[column] = values
    edata.obs["_row"] = np.arange(n)
    cohort = CF.obs_only(edata, ["_row", "grid_hours", *task_obs])
    steps += CF.apply(cohort, criteria, tracker)
    idx = cohort.obs["_row"].to_numpy()
    assert np.array_equal(idx, np.flatnonzero(keep)), "the tracked filters disagree with the exclusions"
    CF.write(out_dir, n_candidates, steps, tracker)
    print(f"[task] {task}: kept {len(idx)}/{n} stays after task exclusions", flush=True)

    # 4. dyn (per kept stay's grid) + outc (dynamic: per hour; static: per stay), from the EHRData
    idx = idx[Lgrid[idx] > 0]
    L = Lgrid[idx].astype(np.int64)
    dyn = ED.dyn_frame(ED.take_rows(X, idx), stay_obs[idx], L)
    if dynamic:
        in_grid = np.arange(T)[None, :] < L[:, None]
        outc = pd.DataFrame({"stay_id": dyn["stay_id"].to_numpy(), "time": dyn["time"].to_numpy(),
                             "label": edata.obsm["label"][idx, -h_lo:T - h_lo][in_grid]})
    else:
        outc = pd.DataFrame({"stay_id": stay_obs[idx].astype(np.int64),
                             "label": edata.obs["label"].to_numpy()[idx]})

    # 5. statics (kept stays)
    sta = pd.DataFrame({
        "stay_id": stay_obs[idx].astype(np.int64),
        "age": age_arr[idx],
        "sex": [GENDER_CONCEPT_TO_SEX.get(int(g)) if pd.notna(g) else None for g in gender_arr[idx]],
        "height": height_arr[idx], "weight": weight_arr[idx],
    })

    to_parquet_atomic(dyn, f"{out_dir}/dyn.parquet")
    to_parquet_atomic(outc, f"{out_dir}/outc.parquet")
    to_parquet_atomic(sta, f"{out_dir}/sta.parquet")
    if dynamic:
        assert len(dyn) == len(outc)
    ld = (f"dynamic label hours[min={outc['label'].min():.1f} max={outc['label'].max():.1f} med={outc['label'].median():.1f}]"
          if dynamic else f"static label mean={float(np.mean(outc['label'])):.4f} n_pos={int(np.nansum(outc['label']))}")
    print(f"[write] {out_dir}  stays={len(sta)} dyn_rows={len(dyn)} outc_rows={len(outc)} "
          f"finite_frac={float(np.isfinite(dyn[YAIB_DYNAMIC_VARS].to_numpy()).mean()):.4f} | {ld}", flush=True)
    print("[OK] build done.", flush=True)


if __name__ == "__main__":
    main()
