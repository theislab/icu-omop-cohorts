"""OMOP concepts, units and bounds of YAIB's 48 dynamic variables, and the paths of the cohort builds.

Read by every step of the cohort builds:
  * VAR_CONCEPTS: the measurement concepts of each variable. ehrdata takes the hourly median over
    all of them (urine summed), as a ricu concept unions its source items.
  * UNIT_CONVERSIONS: the units the CDM records a variable in, converted into YAIB's by ehrdata.
  * RICU_BOUNDS: ricu's plausible ranges, applied after the conversion.
  * TARGET_MEASUREMENT_CONCEPT_IDS: what the prefilter (prefilter.py) keeps of
    MEASUREMENT for a dataset's cohort cache.

Changing VAR_CONCEPTS changes TARGET_MEASUREMENT_CONCEPT_IDS: a concept added since a cohort cache
was built is missing from it, so rebuild the cache (sbatch run_prefilter.sbatch
<dataset>). The builders warn about this (`assert_cache_covers_targets`).
"""

from __future__ import annotations

import os

# Which OMOP CDM the builders read: environment OMOP_CDM (config.env, see ../config.example.env). A CDM written by
# ../omop_etl sits at <out_dir>/cdm; its cohort cache and frames go next to it, into
# <out_dir>/cohorts/{cohort_cache,frames}. COHORT_CACHE and FRAMES_OUT (environment) override them. Without OMOP_CDM
# the paths are empty and the builders stop (`require_cdm`).
OMOP_CDM = os.environ.get("OMOP_CDM", "").rstrip("/")
COHORTS_ROOT = f"{os.path.dirname(OMOP_CDM)}/cohorts" if OMOP_CDM else ""
COHORT_CACHE = os.environ.get("COHORT_CACHE") or (f"{COHORTS_ROOT}/cohort_cache" if OMOP_CDM else "")
FRAMES = f"{COHORTS_ROOT}/frames" if OMOP_CDM else ""


def require_cdm() -> None:
    """Stop a builder that has no CDM to read."""
    if not OMOP_CDM:
        raise SystemExit("set OMOP_CDM to the OMOP CDM to read, a folder with a Parquet file per table "
                         "(e.g. in config.env, see config.example.env)")

# The 48 dynamic variables in YAIB's order (PredictionTaskVariables.gin).
YAIB_DYNAMIC_VARS: list[str] = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
    "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp",
    "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
    "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut",
    "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp",
    "temp", "tnt", "urine", "wbc",
]

# The variables fed by more than one concept; ehrdata pools the concepts of a variable, as a ricu concept
# unions its source items. The variables with one concept are added from CONCEPT_TO_YAIB below.
VAR_CONCEPTS: dict[str, list[int]] = {
    # blood pressure: invasive preferred, non-invasive as fallback
    "sbp": [3004249, 21492239],
    "dbp": [3012888, 21492240],
    "map": [3027598, 21492241],
    # oxygen saturation: pulse oximetry and the blood gas saturation
    "o2sat": [40762499, 3013502],
    # respiratory rate: total and spontaneous
    "resp": [3024171, 1175625],
    # labs: the concept whose property matches the recorded unit, mass (mg/dL; MIMIC-IV, eICU) or
    # moles (SI; HiRID); glucose also by specimen (serum or plasma, blood)
    "lact": [3047181, 40762125],
    "glu": [3020491, 3004501, 3000483],
    "mg": [3033836, 3001420],
    "crea": [40762887, 3016723],
    "ca": [3005162, 3006906],
    "bili": [3006140, 3024128],
    "phos": [3018913, 3011904],
    "bun": [43534100, 3013682],
}

# Per-dataset pathway lists where ricu's concept reads other sources: hirid sbp/dbp are the invasive items 100/120
# only; miiv resp also takes 224688 'Respiratory Rate (Set)', the ventilator's set rate, which the CDM holds as
# respiratory_rate_setting (ricu's union on miiv only; RICU.md).
VAR_CONCEPTS_BY_DATASET: dict[str, dict[str, list[int]]] = {
    "hirid": {"sbp": [3004249], "dbp": [3012888]},
    "miiv": {"resp": [3024171, 1175625, 3007469]},
}


def var_concepts(var: str, dataset: str) -> list[int]:
    """The OMOP concepts feeding YAIB variable `var` on `dataset`."""
    return VAR_CONCEPTS_BY_DATASET.get(dataset, {}).get(var, VAR_CONCEPTS.get(var, []))


# OMOP measurement_concept_id -> YAIB variable; the variables of several concepts appear once per concept. The
# concepts are those of OHDSI's MIMIC-IV ETL (github.com/OHDSI/MIMIC), LOINC, for every dataset; HiRID's labs in
# SI units take the LOINC "[Moles/volume]" concept of the same test (omop_etl/docs/OHDSI-MIMIC.md).
CONCEPT_TO_YAIB: dict[int, str] = {
    3027018: "hr",       # Heart rate
    3020891: "temp",     # Body temperature
    3024171: "resp",     # Respiratory rate
    1175625: "resp",     # Breath rate spontaneous
    3020716: "fio2",     # Inhaled oxygen concentration
    40762499: "o2sat",   # Oxygen saturation in Arterial blood by Pulse oximetry (SpO2)
    3013502: "o2sat",    # Oxygen saturation in Blood (blood gas)
    3047181: "lact",     # Lactate [Moles/volume] in Blood
    40762125: "lact",    # Lactate [Mass/volume] in Blood (no dataset records mg/dL; CDMs before iter5)
    3020491: "glu",      # Glucose [Moles/volume] in Blood (HiRID)
    3004501: "glu",      # Glucose [Mass/volume] in Serum or Plasma
    3000483: "glu",      # Glucose [Mass/volume] in Blood (blood gas, bedside)
    3033836: "mg",       # Magnesium [Moles/volume] in Blood (HiRID)
    3001420: "mg",       # Magnesium [Mass/volume] in Serum or Plasma
    3019550: "na",       # Sodium [Moles/volume] in Serum or Plasma
    40762887: "crea",    # Creatinine [Moles/volume] in Blood (HiRID)
    3016723: "crea",     # Creatinine [Mass/volume] in Serum or Plasma
    3005162: "ca",       # Calcium [Moles/volume] in Blood (HiRID)
    3006906: "ca",       # Calcium [Mass/volume] in Serum or Plasma
    3014576: "cl",       # Chloride [Moles/volume] in Serum or Plasma
    3023103: "k",        # Potassium [Moles/volume] in Serum or Plasma
    3018677: "ptt",      # aPTT in Platelet poor plasma by Coagulation assay
    3006140: "bili",     # Bilirubin.total [Moles/volume] in Serum or Plasma (HiRID)
    3024128: "bili",     # Bilirubin.total [Mass/volume] in Serum or Plasma
    3006923: "alt",      # alanine_aminotransferase
    3013721: "ast",      # aspartate_aminotransferase
    3035995: "alp",      # Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma
    3024561: "alb",      # albumin
    3018913: "phos",     # Phosphate [Moles/volume] in Blood (HiRID)
    3011904: "phos",     # Phosphate [Mass/volume] in Serum or Plasma
    3016293: "bicar",    # Bicarbonate [Moles/volume] in Serum or Plasma
    43534100: "bun",     # Urea [Moles/volume] in Venous blood (HiRID)
    3013682: "bun",      # Urea nitrogen [Mass/volume] in Serum or Plasma
    3010421: "ph",       # pH of Blood
    3027315: "po2",      # Oxygen [Partial pressure] in Blood
    3013290: "pco2",     # Carbon dioxide [Partial pressure] in Blood
    3000963: "hgb",      # hemoglobin
    3000905: "wbc",      # Leukocytes [#/volume] in Blood by Automated count
    3024929: "plt",      # Platelets [#/volume] in Blood by Automated count
    3014315: "urine",    # Urine output
    3012501: "be",       3027597: "bili_dir", 3004809: "bnd",   3021119: "cai",
    3007220: "ck",       3005785: "ckmb",     3020460: "crp",   3016407: "fgn",
    3012030: "mch",      3009744: "mchc",     3023599: "mcv",   3007930: "methb",
    3008342: "neut",     3019800: "tnt",      3037511: "lymph", 3022217: "inr_pt",
    # blood pressure, invasive and non-invasive
    3004249: "sbp",  21492239: "sbp",  # invasive_systolic / noninvasive_systolic
    3012888: "dbp",  21492240: "dbp",  # invasive_diastolic / noninvasive_diastolic
    3027598: "map",  21492241: "map",  # invasive_mean / noninvasive_mean
}

# Measurement concepts that are not among YAIB's 48 dynamic variables and never reach `dyn`. GCS is
# SOFA's CNS input and the ventilation concepts qualify its respiration; vasopressors and antibiotics
# come from DRUG_EXPOSURE.
NON_DYNAMIC_CONCEPTS: dict[int, str] = {
    3007194: "glasgow_coma_score",
    3016335: "glasgow_coma_score_eye",
    3008223: "glasgow_coma_score_motor",
    3009094: "glasgow_coma_score_verbal",
    3004921: "ventilator_mode",
    3022875: "PEEP",
    36303946: "plateau_pressure",
    3007469: "respiratory_rate_setting",
    21490854: "tidal_volume",  # observed and set, one concept as in OHDSI's ETL
}

# Units. The CDM stores values as recorded, with the unit they were recorded in
# (MEASUREMENT.unit_concept_id). The builders convert them into the unit ricu and YAIB use, per row,
# with the table below; ehrdata applies it to the dynamics (`setup_variables(unit_conversions=...)`).
# Variables not listed are in one unit in every dataset and are not converted.
MG_DL, MMOL_L, UMOL_L, MEQ_L, MG_L = 8840, 8753, 8749, 9557, 8751
G_DL, G_L, NG_ML, NG_L, PERCENT = 8713, 8636, 8842, 8725, 8554
CELSIUS, FAHRENHEIT, CM, INCH, KG, LB = 586323, 9289, 8582, 9327, 9529, 8739
# {variable: (YAIB unit, {recorded unit: (multiplier, offset)})}; value in YAIB unit = value * multiplier
# + offset. The mass <-> substance factors follow the molar mass (creatinine 113.12 g/mol -> 88.4,
# urea nitrogen 2 x 14.007 -> 2.8, calcium 40.08 -> 4.0, magnesium 24.305 -> 2.43, phosphorus 30.97 -> 3.1,
# glucose 180.16 -> 18.016, bilirubin 584.7 -> 17.1); ricu's HiRID factors
# (calcium 4.008, magnesium 2.431, phosphate 3.097521) differ by at most 0.2 %. Magnesium in mEq/L is
# half the mmol/L (divalent) and so x 1.215 into mg/dL; ricu divides by 1.215 instead.
#
# RICU CHOICE (RICU.md, "Where we follow ricu"): ricu reads every eICU ionized calcium row
# as mmol/L whatever its unit string, so its 0.5-2 mmol/L bounds drop the rows recorded in mg/dL or
# mEq/L (5.6 % of the rows on eICU 2.0). By default we read them as ricu does. CAI_CONVERT=1
# (environment) converts them by their unit instead (40.08 g/mol: 4.008 mg/dL per mmol/L; divalent:
# 2 mEq/L per mmol/L); the builders then write eICU into a `_caiconverted` tree.
CAI_CONVERT = os.environ.get("CAI_CONVERT", "") == "1"
CAI_AS_RICU = {MMOL_L: (1.0, 0.0), MG_DL: (1.0, 0.0), MEQ_L: (1.0, 0.0)}
CAI_CONVERTED = {MMOL_L: (1.0, 0.0), MG_DL: (1 / 4.008, 0.0), MEQ_L: (0.5, 0.0)}
UNIT_CONVERSIONS: dict[str, tuple[int, dict[int, tuple[float, float]]]] = {
    "crea": (MG_DL, {MG_DL: (1.0, 0.0), UMOL_L: (1 / 88.4, 0.0)}),
    "bun": (MG_DL, {MG_DL: (1.0, 0.0), MMOL_L: (2.8, 0.0)}),
    "ca": (MG_DL, {MG_DL: (1.0, 0.0), MMOL_L: (4.0, 0.0)}),
    "bili": (MG_DL, {MG_DL: (1.0, 0.0), UMOL_L: (1 / 17.1, 0.0)}),
    "bili_dir": (MG_DL, {MG_DL: (1.0, 0.0), UMOL_L: (1 / 17.1, 0.0)}),
    "alb": (G_DL, {G_DL: (1.0, 0.0), G_L: (0.1, 0.0)}),
    "mg": (MG_DL, {MG_DL: (1.0, 0.0), MMOL_L: (2.43, 0.0), MEQ_L: (1.215, 0.0)}),
    "phos": (MG_DL, {MG_DL: (1.0, 0.0), MMOL_L: (3.1, 0.0)}),
    "glu": (MG_DL, {MG_DL: (1.0, 0.0), MMOL_L: (18.016, 0.0)}),
    "hgb": (G_DL, {G_DL: (1.0, 0.0), G_L: (0.1, 0.0)}),
    "mchc": (G_DL, {G_DL: (1.0, 0.0), G_L: (0.1, 0.0)}),  # YAIB writes the g/dL as %
    "fgn": (MG_DL, {MG_DL: (1.0, 0.0), G_L: (100.0, 0.0)}),
    "tnt": (NG_ML, {NG_ML: (1.0, 0.0), NG_L: (0.001, 0.0)}),
    "crp": (MG_L, {MG_L: (1.0, 0.0), MG_DL: (10.0, 0.0)}),
    "cai": (MMOL_L, CAI_CONVERTED if CAI_CONVERT else CAI_AS_RICU),  # RICU CHOICE, see above
    "temp": (CELSIUS, {CELSIUS: (1.0, 0.0), FAHRENHEIT: (1 / 1.8, -32 / 1.8)}),
}
# the statics, read by concept rather than by variable
STATIC_UNIT_CONVERSIONS: dict[str, tuple[int, dict[int, tuple[float, float]]]] = {
    "height": (CM, {CM: (1.0, 0.0), INCH: (2.54, 0.0)}),
    "weight": (KG, {KG: (1.0, 0.0), LB: (0.45359237, 0.0)}),
}


def unit_conversion_rows(table=None) -> list[tuple[str, int, int, float, float]]:
    """(variable, unit_concept_id, to_unit_concept_id, multiplier, offset), one row per recorded unit."""
    table = UNIT_CONVERSIONS if table is None else table
    return [(var, unit, target, mult, off)
            for var, (target, units) in table.items() for unit, (mult, off) in units.items()]


# Measurements identified by SOURCE VALUE, not a standard concept id. HiRID's cumulative fluid
# balance has no standard OMOP concept -- every LOINC "Fluid balance" is a fixed window (1h/8h/24h),
# not a running total -- so it is written with concept_id 0 and its identity lives in
# `measurement_source_value`. The prefilter must name it explicitly; filtering on id 0 would pull in
# every unmapped measurement in the CDM.
#
# Used by the experiment with ricu's urine (RICU_HIRID_URINE): ricu's hirid dyn `urine` is this variable de-cumulated, where
# ours is real hourly urine volume. Carrying it lets that comparison be made without re-extraction.
# The CDM stores the source code in measurement_source_value: hirid fluid balance is
# variableid 30005110 and the ventilator airway code 15001552.
FLUID_BALANCE_SOURCE = "30005110"
HIRID_AIRWAY_SOURCE = "15001552"
SOURCE_VALUE_CONCEPTS: tuple[str, ...] = (FLUID_BALANCE_SOURCE, HIRID_AIRWAY_SOURCE)



# ricu's plausibility bounds, applied to the value AFTER unit conversion (ricu's `report_set_unit`
# converts nothing -- it only tabulates and warns -- so `filter_bounds` sees already-in-unit values).
#
#   filter_bounds(x, col, min, max):
#       x <- rm_na_val_var(x, col)                          # NA first
#       keep <- (is.null(min) | v >= min) & (is.null(max) | v <= max)
#
# so a NULL bound is "no constraint on that side", NOT zero. Values outside are DROPPED, not
# clipped. Merged from base ricu's dict: the ricu-extensions overrides are `sources`-only (ricu's
# `combine_sources` errors otherwise), so taking the extension entry first silently shadows the
# bounds with None -- which is what a first attempt at this table did.
# Only `bnd` and `inr_pt` are genuinely unbounded.
RICU_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "alb": (0, 6),
    "alp": (0, None),
    "alt": (0, None),
    "ast": (0, None),
    "be": (-25, 25),
    "bicar": (5, 50),
    "bili": (0, 100),
    "bili_dir": (0, 50),
    "bun": (0, 200),
    "ca": (4, 20),
    "cai": (0.5, 2),
    "ck": (0, None),
    "ckmb": (0, None),
    "cl": (80, 130),
    "crea": (0, 15),
    "crp": (0, None),
    "dbp": (0, 200),
    "fgn": (0, 1500),
    "fio2": (21, 100),
    "glu": (0, 1000),
    "hgb": (4, 18),
    "hr": (0, 300),
    "k": (0, 10),
    "lact": (0, 50),
    "lymph": (0, 100),
    "map": (0, 250),
    "mch": (0, None),
    "mchc": (20, 50),
    "mcv": (50, 150),
    "methb": (0, 100),
    "mg": (0.5, 5),
    "na": (110, 165),
    "neut": (0, 100),
    "o2sat": (50, 100),
    "pco2": (10, 150),
    "ph": (6.8, 8),
    "phos": (0, 40),
    "plt": (5, 1200),
    "po2": (40, 600),
    "ptt": (0, None),
    "resp": (0, 120),
    "sbp": (0, 300),
    "temp": (32, 42),
    "tnt": (0, None),
    "urine": (0, 2000),
    "wbc": (0, None),
}
# Unbounded in ricu, deliberately absent above: bnd, inr_pt.

# Statics: age and sex from PERSON, height and weight from MEASUREMENT (STATIC_MEASUREMENT_CONCEPT_IDS).
GENDER_CONCEPT_TO_SEX = {8507: "Male", 8532: "Female"}

# Complete VAR_CONCEPTS: the multi-pathway entries above, plus one entry per single-concept var.
for _cid, _var in CONCEPT_TO_YAIB.items():
    if _var not in VAR_CONCEPTS:
        VAR_CONCEPTS[_var] = [_cid]
del _cid, _var

# The distinct measurement_concept_ids to pull from MEASUREMENT.
# Static anthropometrics. These are MEASUREMENT rows (in an early version of the ETL they were
# OBSERVATION rows with an empty `value_as_number` and an invalid weight concept id). They are NOT
# dynamic variables -- they never enter the 48-column tensor -- but they MUST be in the prefilter's
# target list, because prefilter.py filters MEASUREMENT to these ids and anything
# absent is dropped from every cohort_cache. Leaving them out makes the statics silently
# all-NaN, which is exactly the bug #31 fixed.
STATIC_MEASUREMENT_CONCEPT_IDS: dict[int, str] = {
    3036277: "height",   # Body height
    3025315: "weight",   # Body weight
}

# Concepts the SOFA / Sepsis-3 label needs that are NOT dynamic tensor columns. Same trap as
# the statics above: prefilter.py filters MEASUREMENT to TARGET_MEASUREMENT_CONCEPT_IDS,
# so a concept absent from that list never reaches a cohort_cache no matter that it exists in the CDM.
#   gcs             -> sofa_cns   (4 minus the number of the bounds 6, 10, 13, 15 that the GCS reaches)
#   ventilator_mode -> sofa_resp  (pafi is floored at 200 when the patient is NOT ventilated)
SOFA_MEASUREMENT_CONCEPT_IDS: dict[int, str] = {
    3007194: "glasgow_coma_score",
    3004921: "ventilator_mode",
    3016335: "glasgow_coma_score_eye",
    3008223: "glasgow_coma_score_motor",
    3009094: "glasgow_coma_score_verbal",
}
RASS_CONCEPT_ID = 36684829
SOFA_MEASUREMENT_CONCEPT_IDS[RASS_CONCEPT_ID] = "rass"
NON_DYNAMIC_CONCEPTS[RASS_CONCEPT_ID] = "rass"

# Variables computed from other concepts as ricu's callbacks do: HiRID records neutrophils as an
# absolute count, and ricu-extensions' `blood_cell_ratio` makes it the % of the wbc of the same hour.
# {dataset: {variable: concept of the count}}; ehrdata_dynamics.ratio_of_wbc computes it.
NEUTROPHILS_ABSOLUTE = 3013650  # Neutrophils [#/volume] in Blood by Automated count
RATIO_OF_WBC: dict[str, dict[str, int]] = {"hirid": {"neut": NEUTROPHILS_ABSOLUTE}}

TARGET_MEASUREMENT_CONCEPT_IDS: list[int] = sorted(
    set(CONCEPT_TO_YAIB) | set(STATIC_MEASUREMENT_CONCEPT_IDS) | set(SOFA_MEASUREMENT_CONCEPT_IDS)
    | {cid for by_var in VAR_CONCEPTS_BY_DATASET.values() for ids in by_var.values() for cid in ids}
    | {cid for by_var in RATIO_OF_WBC.values() for cid in by_var.values()})

# Dataset prefix used in VISIT_OCCURRENCE.visit_source_value = "{prefix}-{stay_id}".
DATASET_PREFIX = {"miiv": "mimic4", "eicu": "eicu", "hirid": "hirid"}


def assert_cache_covers_targets(con, cache_path: str) -> None:
    """Fail loudly on a cohort_cache pre-dating a VAR_CONCEPTS change.

    A stale cache silently yields an all-NaN column for any newly added pathway, which looks
    exactly like 'this dataset does not record it'. Concepts genuinely absent from a dataset are
    reported but tolerated; the fatal case is a cache built before the concept was targeted.
    """
    present = {r[0] for r in con.execute(
        f"SELECT DISTINCT measurement_concept_id FROM parquet_scan('{cache_path}')"
    ).fetchall()}
    absent = sorted(set(TARGET_MEASUREMENT_CONCEPT_IDS) - present)
    if absent:
        print(f"[cache] WARNING: {len(absent)} target concepts absent from "
              f"{os.path.basename(cache_path)}: {absent}", flush=True)
        print("[cache] if any were added to VAR_CONCEPTS since the cache was built, rebuild it: "
              "sbatch run_prefilter.sbatch <dataset>", flush=True)


def _self_check() -> None:
    covered = set(CONCEPT_TO_YAIB.values())
    assert covered == set(YAIB_DYNAMIC_VARS), (
        f"crosswalk != 48 YAIB vars; missing={set(YAIB_DYNAMIC_VARS) - covered}, extra={covered - set(YAIB_DYNAMIC_VARS)}")

    # VAR_CONCEPTS must be a faithful inversion of CONCEPT_TO_YAIB: every covered var present,
    # every listed concept mapping back to its own var, no concept listed twice for one var.
    assert set(VAR_CONCEPTS) == covered, covered ^ set(VAR_CONCEPTS)
    for v, ids in VAR_CONCEPTS.items():
        assert ids, f"{v} has no concepts"
        assert len(set(ids)) == len(ids), (v, ids)
        assert all(CONCEPT_TO_YAIB.get(i) == v for i in ids), (v, ids)
    assert sum(len(ids) for ids in VAR_CONCEPTS.values()) == len(CONCEPT_TO_YAIB)

    # the multi-pathway vars, i.e. where we union like ricu does
    multi = {v: ids for v, ids in VAR_CONCEPTS.items() if len(ids) > 1}
    assert set(multi) == {"sbp", "dbp", "map", "o2sat", "resp", "lact", "glu", "mg", "crea", "ca", "bili", "phos", "bun"}, \
        sorted(multi)
    assert NON_DYNAMIC_CONCEPTS.keys().isdisjoint(CONCEPT_TO_YAIB), "non-dynamic concept in dyn"
    assert STATIC_MEASUREMENT_CONCEPT_IDS.keys().isdisjoint(CONCEPT_TO_YAIB), "static concept in dyn"
    assert set(STATIC_MEASUREMENT_CONCEPT_IDS).issubset(TARGET_MEASUREMENT_CONCEPT_IDS), \
        "statics missing from the prefilter target list -> cohort_cache would drop them"
    assert set(SOFA_MEASUREMENT_CONCEPT_IDS).issubset(TARGET_MEASUREMENT_CONCEPT_IDS), \
        "SOFA inputs missing from the prefilter target list -> cohort_cache would drop them"
    assert SOFA_MEASUREMENT_CONCEPT_IDS.keys() <= NON_DYNAMIC_CONCEPTS.keys(), \
        "SOFA inputs must stay non-dynamic (they are label inputs, not tensor columns)"


_self_check()
