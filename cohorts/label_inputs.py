"""Label inputs beyond the 48 variables, as ricu defines them: ventilation windows, vasopressor
rates, GCS with sedation, hirid's de-cumulated fluid balance, and the antibiotic and culture rows of the
suspected infection.

The readers take the stays as `stays`, the `.obs` of the build's EHRData: hours count from each stay's
`visit_start_datetime` there, which is the ICU admission or, for the SOFA around the suspected infection, a start
moved by `origin_h` hours from it. Results are aligned to the stays; hourly grids cover hours [h_lo, h_hi). ricu
sources: `ricu-src/R/callback-itm.R` (rate and duration callbacks, hirid_vent), `callback-cncpt.R` (vaso60,
vent_ind, gcs), ricu-extensions `callback-icu-mortality.R` and `configs/medications/concept-dict.json`
(abx_duration).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from abx_sets import abx_condition

ICU_DEATH_CONCEPT_ID = 4216643
SELF_REPORTED_DRUG_TYPE = 32865
VENT_MIIV = (37158404, 4177224)
VENT_EICU_ON = 4230167
VENT_MATCH_H = 6.0
VENT_MIN_H = 0.5
HIRID_VENT_FINAL_H = 4.0
HIRID_VENT_MAX_H = 12.0
GCS_VALID_H = 6
RASS_SEDATED = -3
RASS_WIN_H = 6.0
ETT_TEXT = "No Response-ETT"
DERIVED_GCS_TOTAL = "derived:gcs_total"
GCS_CONCEPTS = {"egcs": 3016335, "vgcs": 3009094, "mgcs": 3008223, "tgcs": 3007194}
GCS_BOUNDS = {"egcs": (1, 4), "vgcs": (1, 5), "mgcs": (1, 6), "tgcs": (3, 15)}
GCS_MAX = {"egcs": 4, "vgcs": 5, "mgcs": 6}

# ricu's vasopressor sources: miiv itemids, hirid pharmaids, eicu drugname regexes with the
# ml -> mcg factor of eicu_rate_kg; rate bounds from the concept dictionary
VASO = {
    "norepi": {"miiv": [221906], "hirid": [1000462, 1000656, 1000657, 1000658],
               "eicu_rate": r"^norepi.*\(.+\)$", "eicu_dur": r"^norepi", "ml_to_mcg": 32, "max": 3},
    "dopa": {"miiv": [221662], "hirid": [],
             "eicu_rate": r"^dopa.*\(.+\)$", "eicu_dur": r"^dopa", "ml_to_mcg": 1600, "max": 50},
    "dobu": {"miiv": [221653], "hirid": [426],
             "eicu_rate": r"^dobu.*\(.+\)$", "eicu_dur": r"^dobu", "ml_to_mcg": 2000, "max": 50},
    "epi": {"miiv": [221289], "hirid": [71, 1000649, 1000650, 1000655, 1000750],
            "eicu_rate": r"^epi( |n).*\(.+\)$", "eicu_dur": r"^epi( |n)", "ml_to_mcg": 40, "max": 1.5},
}
VASO_MERGE_GAP_H = 5 / 60
VASO_MIN_H = 1.0
EICU_DUR_GAP_H = 5.0
INTERVAL_OVERHANG_H = 1.0
INTERVAL_MAX_H = 6.0
MINUTE_H = 1 / 60


def _pos(voi_order):
    return {int(v): i for i, v in enumerate(voi_order)}


# The table of the stays' starts that the SOFA readers join (`_register_starts`)
STARTS = "_stay_starts"


def _register_starts(con, stays: pd.DataFrame) -> np.ndarray:
    """Write table STARTS (visit_occurrence_id, visit_start_datetime) of the stays `stays` (an `.obs`), from
    which the readers count hours; returns their visit_occurrence_ids."""
    voi = stays["voi"].to_numpy(dtype=np.int64)
    con.register("_stay_starts_frame", pd.DataFrame(
        {"visit_occurrence_id": voi, "visit_start_datetime": pd.to_datetime(stays["visit_start_datetime"]).to_numpy()}))
    try:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {STARTS} AS SELECT visit_occurrence_id, "
                    "CAST(visit_start_datetime AS TIMESTAMP) AS visit_start_datetime FROM _stay_starts_frame")
    finally:
        con.unregister("_stay_starts_frame")
    return voi


def _origins(stays: pd.DataFrame) -> np.ndarray:
    """Per stay the hours from ICU admission to its start, `origin_h` of `stays` (0 without it)."""
    return stays["origin_h"].to_numpy(dtype=float) if "origin_h" in stays else np.zeros(len(stays))


def _hours(lo, hi):
    """Hours floor(lo)..floor(hi), as ricu's expand of a window."""
    return np.arange(int(np.floor(lo)), int(np.floor(hi)) + 1)


def mark_windows(windows, voi_order, h_lo, h_hi, values=None):
    """Boolean (or max-value) grid from per-stay windows [(voi, start_h, end_h[, value])]."""
    pos = _pos(voi_order)
    n, H = len(voi_order), h_hi - h_lo
    out = np.zeros((n, H), dtype=bool) if values is None else np.full((n, H), np.nan)
    for k, (voi, s, e) in enumerate(windows):
        i = pos.get(int(voi))
        if i is None or not np.isfinite(s):
            continue
        hrs = _hours(s, e if np.isfinite(e) else s) - h_lo
        hrs = hrs[(hrs >= 0) & (hrs < H)]
        if values is None:
            out[i, hrs] = True
        else:
            out[i, hrs] = np.fmax(out[i, hrs], values[k])
    return out


# --- ICU death --------------------------------------------------------------------------------

def _visit_distance_sql(t):
    """Seconds from time `t` to the visit window of `v` (0 inside), for the nearest-visit rule."""
    end = "coalesce(v.visit_end_datetime, v.visit_start_datetime)"
    return (f"CASE WHEN {t} < v.visit_start_datetime THEN date_diff('second', {t}, v.visit_start_datetime) "
            f"WHEN {t} > {end} THEN date_diff('second', {end}, {t}) ELSE 0 END")

# --- ventilation ------------------------------------------------------------------------------

def window_hours(s, e, origin=0.0):
    """Hour span (lo, hi) of a ricu win_tbl window after change_interval and expand: the start
    floored to the hour, the end at floor(floored start + duration), an end before ICU admission
    moved to admission. `s`, `e` in hours from the grid origin, `origin` the origin's hour from
    ICU admission.
    RICU CHOICE (RICU.md, "Where we follow ricu"): an end before admission moved to
    admission stretches a window that ended before the stay."""
    lo = np.floor(s)
    return lo, max(np.floor(lo + (e - s)), -origin)


def eicu_vent_windows(starts, ends, match_h=VENT_MATCH_H, min_h=VENT_MIN_H, origin=0.0):
    """ricu vent_ind for eicu, as hour spans (lo, hi) of the expanded windows.

    Each start takes the next end within `match_h` (matched on the end moved 1 min earlier) and
    lasts until that end. Without one, ricu's `calc_dur` returns start + match_win, an end time,
    as the duration: the window is (start + 6 h) long, start counted from ICU admission (`origin`
    is the grid origin's hour from admission), and starts more than 5.5 h before admission fall
    below the 30 min minimum. Start and duration are floored to the hour separately, and an end
    before admission is moved to admission.
    RICU CHOICE (RICU.md, "Where we follow ricu"): the duration start + 6 h is ricu's,
    where match_win (6 h) is meant."""
    e_orig = np.sort(np.asarray(ends, dtype=float))
    e_idx = e_orig - MINUTE_H
    out = []
    for s in np.sort(np.asarray(starts, dtype=float)):
        j = np.searchsorted(e_idx, s, side="left")
        dur = e_orig[j] - s if j < len(e_idx) and e_idx[j] - s <= match_h else s + origin + match_h
        if dur >= min_h:
            lo = np.floor(s)
            out.append((lo, max(lo + np.floor(dur), -origin)))
    return out


def hirid_vent_windows(times, codes):
    """ricu hirid_vent: each airway-code row lasts until the next row, 4 h for the last row or
    when the next is more than 12 h away. Returns (start, end, invasive) for codes 1-6.
    Rows at the same time are ordered by code: the first lasts 0 h, the last until the next time.
    ricu keeps the source row order there, which the CDM does not hold."""
    order = np.lexsort((np.asarray(codes, float), np.asarray(times, float)))
    t, c = np.asarray(times, float)[order], np.asarray(codes, float)[order]
    dur = np.append(np.diff(t), HIRID_VENT_FINAL_H)
    dur[dur > HIRID_VENT_MAX_H] = HIRID_VENT_FINAL_H
    keep = np.isin(c, [1, 2, 3, 4, 5, 6])
    return [(a, a + d, cc in (1, 2)) for a, d, cc in zip(t[keep], dur[keep], c[keep])]


def ventilation(con, omop_dir, cache_path, ds, stays, h_lo, h_hi, airway_source="15001552"):
    """(ventilated, invasive) hourly grids of the stays `stays`, ricu's vent_ind and (hirid) invasive
    mech_vent, with ricu's hours per window (`window_hours`, eicu `eicu_vent_windows`)."""
    voi_order = _register_starts(con, stays)
    org = dict(zip(voi_order.tolist(), _origins(stays)))
    if ds == "hirid":
        df = con.execute(
            f"""
            SELECT m.visit_occurrence_id AS voi,
                   date_diff('second', v.visit_start_datetime, m.measurement_datetime) / 3600.0 AS t,
                   m.value_as_number AS code
            FROM parquet_scan('{cache_path}') m JOIN {STARTS} v ON v.visit_occurrence_id = m.visit_occurrence_id
            WHERE m.measurement_source_value = '{airway_source}' AND m.value_as_number IS NOT NULL
            """
        ).df()
        win, inv = [], []
        for voi, g in df.groupby("voi"):
            for s, e, is_inv in hirid_vent_windows(g["t"].to_numpy(), g["code"].to_numpy()):
                w = (voi, *window_hours(s, e, org.get(int(voi), 0.0)))
                win.append(w)
                if is_inv:
                    inv.append(w)
        return mark_windows(win, voi_order, h_lo, h_hi), mark_windows(inv, voi_order, h_lo, h_hi)
    df = con.execute(
        f"""
        SELECT p.visit_occurrence_id AS voi, p.procedure_concept_id AS cid,
               date_diff('second', v.visit_start_datetime, p.procedure_datetime) / 3600.0 AS s,
               date_diff('second', v.visit_start_datetime, p.procedure_end_datetime) / 3600.0 AS e
        FROM parquet_scan('{omop_dir}/PROCEDURE_OCCURRENCE.parquet') p
        JOIN {STARTS} v ON v.visit_occurrence_id = p.visit_occurrence_id
        """
    ).df()
    if ds == "miiv":
        w = df[df["cid"].isin(VENT_MIIV)]
        win = [(voi, *window_hours(s, e, org.get(int(voi), 0.0)))
               for voi, s, e in zip(w["voi"], w["s"], w["e"].fillna(w["s"]))]
    else:
        win = []
        for voi, g in df.groupby("voi"):
            starts = g.loc[g["cid"] == VENT_EICU_ON, "s"].to_numpy()
            ends = np.concatenate([g["e"].dropna().to_numpy(), g.loc[g["cid"] == 0, "s"].to_numpy()])
            win += [(voi, lo, hi) for lo, hi in eicu_vent_windows(starts, ends, origin=org.get(int(voi), 0.0))]
    grid = mark_windows(win, voi_order, h_lo, h_hi)
    return grid, grid


# --- hirid urine for SOFA ------------------------------------------------------------------------

URINE_MAX_ML = 2000.0


def hirid_fluid_urine(con, cache_path, stays, h_lo, h_hi, source="30005110"):
    """ricu's hirid `urine` for the stays `stays`: per stay the cumulative fluid balance de-cumulated
    row by row (`hirid_urine`: first value, then differences, negatives set to 0), rows outside
    0-2000 mL dropped, summed per hour. ricu's sofa_renal reads urine24 from this concept on hirid.
    Returns the hourly grid and per stay the first and the last hour with a row, at any time, NaN if
    none."""
    voi_order = _register_starts(con, stays)
    df = con.execute(
        f"""
        WITH r AS (
          SELECT m.visit_occurrence_id AS voi, m.measurement_datetime AS t, m.value_as_number AS x,
                 v.visit_start_datetime AS t0
          FROM parquet_scan('{cache_path}') m JOIN {STARTS} v ON v.visit_occurrence_id = m.visit_occurrence_id
          WHERE m.measurement_source_value = '{source}' AND m.value_as_number IS NOT NULL),
        d AS (
          -- rows at the same time in the order of their value
          SELECT voi, CAST(floor(date_diff('second', t0, t) / 3600.0) AS INTEGER) AS h,
                 greatest(0, x - coalesce(lag(x) OVER (PARTITION BY voi ORDER BY t, x), 0)) AS u
          FROM r)
        SELECT voi, h, SUM(u) AS val FROM d
        WHERE u <= {URINE_MAX_ML} GROUP BY 1, 2
        """).df()
    grid = _scatter(df[(df["h"] >= h_lo) & (df["h"] < h_hi)], voi_order, h_lo, h_hi)
    r = df.groupby("voi")["h"].agg(["min", "max"]).reindex(voi_order)
    return grid, r["min"].to_numpy(dtype=float), r["max"].to_numpy(dtype=float)


# --- GCS --------------------------------------------------------------------------------------

def _carry_forward(x, max_gap):
    """The last value of `x` (stays x hours) at most `max_gap` hours before each hour, NaN if none."""
    x = np.asarray(x, dtype=np.float64)
    last = np.maximum.accumulate(np.where(~np.isnan(x), np.arange(x.shape[1])[None, :], -1), axis=1)
    held = np.take_along_axis(x, np.maximum(last, 0), axis=1)
    return np.where((last >= 0) & (np.arange(x.shape[1])[None, :] - last <= max_gap), held, np.nan)


def gcs_ricu(comp, sedated, valid_h=GCS_VALID_H):
    """ricu `gcs()`: rows where any component is charted; each component the last value within
    `valid_h`; sedated rows get a total of 15; missing components their maximum; total = charted
    total, else the sum, else 15. `comp` maps egcs/vgcs/mgcs/tgcs to hourly min grids."""
    rows = np.zeros_like(sedated, dtype=bool)
    for x in comp.values():
        rows |= ~np.isnan(x)
    held = {k: _carry_forward(x, valid_h) for k, x in comp.items()}
    tot = np.where(sedated, 15.0, held["tgcs"])
    parts = sum(np.where(np.isnan(held[k]), GCS_MAX[k], held[k]) for k in GCS_MAX)
    tot = np.where(np.isnan(tot), parts, tot)
    return np.where(rows, tot, np.nan)


def _within(seen, hours):
    """Whether `seen` (stays x hours) holds in the hour or in one of the `hours` hours before it."""
    count = np.cumsum(seen, axis=1)
    k = int(hours) + 1
    before = np.zeros_like(count)
    before[:, k:] = count[:, :-k]
    return count > before


def gcs(con, cache_path, ds, stays, comp, h_lo, h_hi, invasive=None, rass=None):
    """Hourly GCS as ricu's `gcs` concept of the stays `stays` from `comp`, the hourly minima of its
    components (egcs, vgcs, mgcs, tgcs inside ricu's bounds, without the ETL's derived total, which is
    no ricu source), with sedation per dataset: miiv verbal 'No Response-ETT'; eicu a RASS of -5..-3
    (`rass`, its hourly grid) in the hour or the 6 h before, which on the first 6 h of the grid misses
    rows before it; hirid invasive ventilation."""
    n, H = len(stays), h_hi - h_lo
    if ds == "miiv":
        voi_order = _register_starts(con, stays)
        df = con.execute(
            f"""
            SELECT m.visit_occurrence_id AS voi,
                   CAST(floor(date_diff('second', v.visit_start_datetime, m.measurement_datetime) / 3600.0) AS INTEGER) AS h,
                   1.0 AS val
            FROM parquet_scan('{cache_path}') m JOIN {STARTS} v ON v.visit_occurrence_id = m.visit_occurrence_id
            WHERE m.value_source_value = '{ETT_TEXT}'
            GROUP BY 1, 2 HAVING h >= {h_lo} AND h < {h_hi}
            """).df()
        sedated = ~np.isnan(_scatter(df, voi_order, h_lo, h_hi))
    elif ds == "eicu":
        sedated = _within(~np.isnan(rass), RASS_WIN_H)
    else:
        sedated = invasive if invasive is not None else np.zeros((n, H), dtype=bool)
    return gcs_ricu(comp, sedated)


def _scatter(df, voi_order, h_lo, h_hi):
    out = np.full((len(voi_order), h_hi - h_lo), np.nan)
    if len(df):
        idx = df["voi"].map(_pos(voi_order))
        ok = idx.notna().to_numpy()
        out[idx[ok].to_numpy().astype(int), df["h"].to_numpy()[ok].astype(int) - h_lo] = df["val"].to_numpy()[ok]
    return out


# --- vasopressors -----------------------------------------------------------------------------

def _sig(col, key):
    return f"NULLIF(regexp_extract({col}, '(^|;){key}=([^;]*)', 2), '')"


def _item_in(ids):
    return "regexp_extract(d.sig, '(^|;)item=([^;]*)', 2) IN (" + ", ".join(f"'{i}'" for i in ids) + ")"


def _drug_rows(con, omop_dir, where):
    """The DRUG_EXPOSURE rows of the stays in STARTS matching `where`, ordered by stay, start time and then their
    values, so that rows sharing a start (eicu rates, hirid doses of one minute) come in a fixed order. ricu keeps the
    source row order there, which the CDM does not hold."""
    return con.execute(
        f"""
        SELECT d.visit_occurrence_id AS voi, d.drug_source_value AS name,
               date_diff('second', v.visit_start_datetime, d.drug_exposure_start_datetime) / 3600.0 AS s,
               date_diff('second', v.visit_start_datetime, d.drug_exposure_end_datetime) / 3600.0 AS e,
               TRY_CAST({_sig('d.sig', 'rate')} AS DOUBLE) AS rate,
               {_sig('d.sig', 'rate_unit')} AS rate_unit,
               TRY_CAST({_sig('d.sig', 'weight_kg')} AS DOUBLE) AS weight,
               {_sig('d.sig', 'group')} AS grp,
               TRY_CAST({_sig('d.sig', 'item')} AS BIGINT) AS item,
               d.quantity AS dose, d.dose_unit_source_value AS dose_unit
        FROM parquet_scan('{omop_dir}/DRUG_EXPOSURE.parquet') d
        JOIN {STARTS} v ON v.visit_occurrence_id = d.visit_occurrence_id
        WHERE {where}
        ORDER BY voi, s, name, item, rate, rate_unit, weight, grp, e, dose, dose_unit
        """
    ).df()


def eicu_rate_kg(rate, unit, weight, ml_to_mcg):
    """ricu eicu_rate_kg: per kg if the unit is not, then /hr -> /min, mg -> mcg, ml -> mcg
    (drug-specific factor), nanograms -> mcg; units, Unknown and bare ml become NA."""
    rate = np.asarray(rate, dtype=float).copy()
    unit = [u if isinstance(u, str) else None for u in unit]
    weight = np.asarray(weight, dtype=float)
    for i, u in enumerate(unit):
        if u is None:
            continue
        if not re.search("/kg/", u, re.I):
            rate[i] = rate[i] / weight[i]
            u = u.replace("/", "/kg/", 1)
        if re.search("/hr$", u, re.I):
            rate[i] /= 60
            u = re.sub("/hr$", "/min", u, flags=re.I)
        if re.search("^mg/", u, re.I):
            rate[i] *= 1000
            u = re.sub("^mg/", "mcg/", u, flags=re.I)
        if re.search("^units/", u, re.I) or re.search("^Unknown$", u, re.I) or re.search("^ml$", u, re.I):
            rate[i] = np.nan
        if re.search("^ml/", u, re.I):
            rate[i] *= ml_to_mcg
            u = re.sub("^ml/", "mcg/", u, flags=re.I)
        if re.search("^nanograms/", u, re.I):
            rate[i] /= 1000
        unit[i] = u
    return rate


def point_intervals(t, overhang=INTERVAL_OVERHANG_H, max_len=INTERVAL_MAX_H):
    """ricu create_intervals: a point row lasts until the next one (at most 6 h), the last one
    1 h; minus one interval step."""
    t = np.asarray(t, dtype=float)
    d = np.append(np.diff(t), overhang)
    return t + np.clip(d, 0, max_len) - MINUTE_H


def merge_ranges(ranges, max_gap=VASO_MERGE_GAP_H):
    """ricu merge_ranges (ricu-src/R/utils-ts.R): every range is widened by `max_gap`, joins the first range (in
    order of lower, then upper bound) it overlaps (`foverlaps(mult = "first")`), takes the largest upper bound of its
    group, and is narrowed by `max_gap` again. A chain of overlaps can leave two overlapping groups, as in ricu."""
    r = sorted((float(lo), float(hi) + max_gap) for lo, hi in ranges)
    merged = {}
    for lo, hi in r:
        first = next(lo2 for lo2, hi2 in r if lo2 <= hi and hi2 >= lo)
        merged[first] = max(merged.get(first, -np.inf), hi)
    return [[lo, hi - max_gap] for lo, hi in sorted(merged.items())]


def vaso60_stay(rates, durations, min_h=VASO_MIN_H):
    """ricu vaso60 for one stay: rate intervals [(start, end, rate)] kept where they overlap a
    merged administration range of at least 1 h. Returns [(start, end, rate)] pieces."""
    ranges = [r for r in merge_ranges([d for d in durations if d[1] - d[0] > 0]) if r[1] - r[0] >= min_h]
    out = []
    for s, e, r in rates:
        for lo, hi in ranges:
            a, b = max(s, lo), min(e, hi)
            if a <= b:
                out.append((a, b, r))
    return out


def vasopressors(con, omop_dir, ds, stays, weight_arr, h_lo, h_hi):
    """For the stays `stays`: {name: hourly max rate in mcg/kg/min over hours with >= 1 h of administration, NaN in
    the other hours}, and per stay the first hour of any of these rates, at any time and of any rate, 0 included, NaN
    if none: in ricu a row of `vaso60` starts the SOFA rows even at rate 0 (sofa_cardio scores it 0)."""
    voi_order = _register_starts(con, stays)
    weight_of = {int(v): w for v, w in zip(voi_order, weight_arr)}
    first = np.full(len(voi_order), np.nan)
    pos = _pos(voi_order)
    out = {}
    for name, cfg in VASO.items():
        rates, durs = {}, {}
        if ds == "miiv":
            df = _drug_rows(con, omop_dir, _item_in(cfg["miiv"]))
            for voi, g in df.groupby("voi"):
                rates[voi] = list(zip(g["s"], g["e"], g["rate"]))
                durs[voi] = [(x["s"].min(), x["e"].max()) for _, x in g.groupby(g["grp"].fillna("-"))]
        elif ds == "hirid":
            if not cfg["hirid"]:
                out[name] = np.full((len(voi_order), h_hi - h_lo), np.nan)
                continue
            df = _drug_rows(con, omop_dir, _item_in(cfg["hirid"]))
            df = df[df["dose_unit"].isin(["mg", "µg"])].copy()
            df["mcg"] = np.where(df["dose_unit"] == "mg", df["dose"] * 1000, df["dose"])
            df["m"] = np.floor(df["s"] * 60) / 60
            for voi, g in df.groupby("voi"):
                w = weight_of.get(int(voi), np.nan)
                r = []
                for _, x in g.groupby(g["grp"].fillna("-")):
                    agg = x.groupby("m")["mcg"].sum().sort_index()
                    t = agg.index.to_numpy()
                    r += list(zip(t, point_intervals(t), agg.to_numpy() / w))
                rates[voi] = r
                durs[voi] = [(x["s"].min(), x["s"].max()) for _, x in g.groupby(g["grp"].fillna("-"))]
        else:
            df = _drug_rows(con, omop_dir,
                            f"regexp_matches(d.drug_source_value, '{cfg['eicu_dur']}', 'i') AND d.sig LIKE '%table=infusiondrug%'")
            for voi, g in df.groupby("voi"):
                # rates sharing a start in the order of `_drug_rows`: the last one lasts until the next start
                rg = g[g["name"].str.contains(cfg["eicu_rate"], case=False, regex=True)]
                if len(rg):
                    unit = rg["name"].str.replace(r"\)$", "", regex=True).str.replace(r"^.+\(", "", regex=True)
                    w = rg["weight"].fillna(weight_of.get(int(voi), np.nan)).to_numpy()
                    r = eicu_rate_kg(rg["rate"].to_numpy(), unit.tolist(), w, cfg["ml_to_mcg"])
                    t = rg["s"].to_numpy()
                    rates[voi] = list(zip(t, point_intervals(t), r))
                t = g["s"].to_numpy()
                grp = np.concatenate([[0], np.cumsum(np.diff(t) > EICU_DUR_GAP_H)])
                durs[voi] = [(t[grp == k].min(), t[grp == k].max()) for k in np.unique(grp)]
        win, val = [], []
        for voi, r in rates.items():
            r = [(s, e, x) for s, e, x in r if np.isfinite(x) and 0 <= x <= cfg["max"] and np.isfinite(s)]
            for s, e, x in vaso60_stay(r, durs.get(voi, [])):
                win.append((voi, s, e))
                val.append(x)
                i = pos.get(int(voi))
                if i is not None:
                    first[i] = np.fmin(first[i], np.floor(s))
        out[name] = mark_windows(win, voi_order, h_lo, h_hi, values=np.array(val))
    return out, first


# --- urine rate --------------------------------------------------------------------------------

URINE_RATE_MAX_GAP_H = 24


def ricu_urine_rate(top, count, m_lo: int, T: int) -> np.ndarray:
    """ricu-extensions' `urine_rate` (KDIGO's urine input) on miiv and eicu, `combine_callbacks(fwd_concept('urine'),
    urine_rate)` (callbacks/callback-kdigo.R), on hours [0, T) from each stay's start, NaN in hours without a row.
    Every urine row is divided by the hours since the stay's previous row plus 1 (by 1 for its first row and after
    more than 24 h), and an hour takes the largest of its rates (the concept's aggregate "max").
    The rows come per minute from minute `m_lo` of each stay's start, as sparse (stays x 1 x minutes) arrays of the
    largest value of a minute (`top`) and its number of rows (`count`). The rows of a minute share one time in these
    CDMs and come in the order of their value: the first, the smallest, takes the divisor and the others 1, so a
    minute of several rows keeps its largest. `m_lo` lies 24 h or more before the start, so that a row whose previous
    row lies before it is divided by 1, as ricu does.
    RICU CHOICE (RICU.md, "Where we follow ricu"): the urine output of an hour is its volume. ricu halves
    hourly charted volume (gap 1 h + 1), divides the first ICU row by the gap since the last row before admission,
    and takes the largest row of an hour instead of the sum."""
    assert m_lo <= -60 * URINE_RATE_MAX_GAP_H, "the minutes must start 24 h before the start"
    r, _, t = top.coords
    ok = np.isfinite(top.data)
    r, t, x = r[ok], t[ok], top.data[ok]
    order = np.lexsort((t, r))
    r, t, x = r[order], t[order], x[order]
    out = np.full((top.shape[0], T), np.nan)
    if not len(x):
        return out
    width = top.shape[2]
    keys = count.coords[0].astype(np.int64) * width + count.coords[2]
    by_key = np.argsort(keys)
    rows = count.data[by_key][np.searchsorted(keys[by_key], r.astype(np.int64) * width + t)]
    h = (t + m_lo) // 60
    gap = np.empty(len(h))
    gap[1:] = h[1:] - h[:-1] + 1
    gap[np.r_[True, r[1:] != r[:-1]]] = np.inf          # a stay's first row
    gap[gap > URINE_RATE_MAX_GAP_H] = 1.0
    rate = np.where(rows >= 2, x, x / gap)
    keep = (h >= 0) & (h < T)
    np.fmax.at(out, (r[keep], h[keep]), rate[keep])
    return out


# --- suspected infection: antibiotic and culture rows -----------------------------------------------

SPECIMEN_DOMAIN, VISIT_DOMAIN, HAS_ASSOCIATED_VISIT = 36, 8, 33136
# The tables of the antibiotic and the culture rows, under the names of the OMOP tables that ehrdata's setup
# functions read; each holds one concept, 0, so that ehrdata makes one variable of it
ABX_TABLE, SPECIMEN_TABLE = "drug_exposure", "specimen"
# Rows per stay at hour resolution, binned as ehrdata bins them: from the stay's start, floored on microseconds
_HOUR_US = 3_600_000_000


def _hour_sql(t, start):
    us = f"(epoch_us(CAST({t} AS TIMESTAMP)) - epoch_us(CAST({start} AS TIMESTAMP)))"
    return f"((({us}) - ((({us}) % {_HOUR_US}) + {_HOUR_US}) % {_HOUR_US}) // {_HOUR_US})"


def infection_tables(con, omop_dir, ds, stays) -> pd.DataFrame:
    """Write the tables ABX_TABLE and, on miiv, SPECIMEN_TABLE with the rows of the stays `stays` (an `.obs`) for
    ehrdata's setup functions, and return per stay, in the order of `stays`: `n_abx` and `abx_first` (the number of
    antibiotic rows and the hour of the first) and `first` and `last` (the first and the last hour of a row of
    either table), NaN where a stay has none. Hours count from each stay's start. Every row carries the
    visit_occurrence_id and person_id of its stay, so that ehrdata gives it to that stay only.

    ABX_TABLE: ricu's `abx_duration`, rows selected per dataset on drug_source_value (abx_sets.py), at any time,
    with the duration floored to the hour in `dur_floor`. RICU CHOICE (RICU.md, "Where we follow ricu"):
    ricu keeps a source stop before the start as a negative duration, which shortens the course; the CDM records
    it in sig.
    SPECIMEN_TABLE: every sample of SPECIMEN, at any time. SPECIMEN has no visit_occurrence_id; a sample belongs
    to the visit FACT_RELATIONSHIP links it to ('Has associated visit'), or, in a CDM without these links, to its
    person's nearest visit."""
    frame = pd.DataFrame({"row": np.arange(len(stays)), "visit_occurrence_id": stays["voi"].to_numpy(dtype=np.int64),
                          "person_id": stays["person_id"].to_numpy(dtype=np.int64),
                          "start": pd.to_datetime(stays["visit_start_datetime"]).to_numpy()})
    con.register("_infection_stays", frame)
    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE {ABX_TABLE} AS
            SELECT s.person_id, s.visit_occurrence_id, 0 AS drug_concept_id, d.drug_exposure_start_datetime,
                   floor(COALESCE(TRY_CAST({_sig('d.sig', 'source_stop_min')} AS DOUBLE) / 60.0,
                                  GREATEST(0, COALESCE(date_diff('second', d.drug_exposure_start_datetime,
                                                                 d.drug_exposure_end_datetime), 0) / 3600.0))) AS dur_floor
            FROM parquet_scan('{omop_dir}/DRUG_EXPOSURE.parquet') d
            JOIN _infection_stays s ON s.visit_occurrence_id = d.visit_occurrence_id
            WHERE {abx_condition(ds, sig_col="d.sig")}
              AND d.drug_type_concept_id IS DISTINCT FROM {SELF_REPORTED_DRUG_TYPE}""")
        rows = [f"SELECT s.row, TRUE AS abx, {_hour_sql('a.drug_exposure_start_datetime', 's.start')} AS h "
                f"FROM {ABX_TABLE} a JOIN _infection_stays s USING (visit_occurrence_id)"]
        if ds == "miiv":
            _specimen_table(con, omop_dir)
            rows.append(f"SELECT s.row, FALSE, {_hour_sql('p.specimen_datetime', 's.start')} "
                        f"FROM {SPECIMEN_TABLE} p JOIN _infection_stays s USING (visit_occurrence_id)")
        df = con.execute(f"""
            SELECT row, count(*) FILTER (WHERE abx) AS n_abx, min(h) FILTER (WHERE abx) AS abx_first,
                   min(h) AS first, max(h) AS last
            FROM ({" UNION ALL ".join(rows)}) GROUP BY row""").df()
    finally:
        con.unregister("_infection_stays")
    out = df.set_index("row").reindex(np.arange(len(stays)))
    out["n_abx"] = out["n_abx"].fillna(0).astype(np.int64)
    return out[["n_abx", "abx_first", "first", "last"]].astype({c: float for c in ("abx_first", "first", "last")})


def _specimen_table(con, omop_dir):
    """SPECIMEN_TABLE of the samples of the stays in `_infection_stays` (`infection_tables`)."""
    fr = f"{omop_dir}/FACT_RELATIONSHIP.parquet"
    linked = os.path.exists(fr) and con.execute(
        f"SELECT count(*) FROM parquet_scan('{fr}') WHERE domain_concept_id_1 = {SPECIMEN_DOMAIN} "
        f"AND relationship_concept_id = {HAS_ASSOCIATED_VISIT}").fetchone()[0] > 0
    if linked:
        visits = f"""
            SELECT s.specimen_datetime, f.fact_id_2 AS visit_occurrence_id
            FROM parquet_scan('{omop_dir}/SPECIMEN.parquet') s
            JOIN parquet_scan('{fr}') f ON f.domain_concept_id_1 = {SPECIMEN_DOMAIN} AND f.fact_id_1 = s.specimen_id
                 AND f.domain_concept_id_2 = {VISIT_DOMAIN} AND f.relationship_concept_id = {HAS_ASSOCIATED_VISIT}"""
    else:
        visits = f"""
            SELECT specimen_datetime, visit_occurrence_id FROM (
              SELECT s.specimen_datetime, v.visit_occurrence_id,
                     row_number() OVER (PARTITION BY s.specimen_id
                                        ORDER BY {_visit_distance_sql('s.specimen_datetime')} NULLS LAST,
                                                 v.visit_start_datetime, v.visit_occurrence_id) AS rk
              FROM parquet_scan('{omop_dir}/SPECIMEN.parquet') s
              JOIN parquet_scan('{omop_dir}/VISIT_OCCURRENCE.parquet') v ON s.person_id = v.person_id)
            WHERE rk = 1"""
    # unit_concept_id: setup_variables reports the units of the table, which hold none here
    con.execute(f"""
        CREATE OR REPLACE TABLE {SPECIMEN_TABLE} AS
        SELECT s.person_id, s.visit_occurrence_id, 0 AS specimen_concept_id, x.specimen_datetime,
               CAST(NULL AS INTEGER) AS unit_concept_id
        FROM ({visits}) x JOIN _infection_stays s USING (visit_occurrence_id)""")
