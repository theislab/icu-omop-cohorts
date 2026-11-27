"""Unit tests for the OMOP<->YAIB crosswalk: every variable has concepts, the prefilter keeps every
concept a build reads, units and bounds. No data access.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from yaib_omop_crosswalk import (  # noqa: E402
    CONCEPT_TO_YAIB,
    NON_DYNAMIC_CONCEPTS,
    SOFA_MEASUREMENT_CONCEPT_IDS,
    STATIC_MEASUREMENT_CONCEPT_IDS,
    TARGET_MEASUREMENT_CONCEPT_IDS,
    VAR_CONCEPTS,
    YAIB_DYNAMIC_VARS,
    var_concepts,
)


def test_every_dynamic_var_has_concepts():
    for var in YAIB_DYNAMIC_VARS:
        assert VAR_CONCEPTS.get(var), var


def test_target_ids_cover_every_pathway():
    """The pre-filter must pull every concept any var can draw on, else it stays empty."""
    for var, ids in VAR_CONCEPTS.items():
        for cid in ids:
            assert cid in TARGET_MEASUREMENT_CONCEPT_IDS, (var, cid)
    # targets = the dynamic concepts PLUS the static anthropometrics. The statics are not tensor
    # columns, but prefilter.py filters MEASUREMENT to this list, so anything absent
    # never reaches the cohort_cache -- which is how height/weight silently became all-NaN before.
    # and the counts the ratio variables are computed from
    # and the per-dataset concepts, such as miiv's set respiratory rate in resp
    from yaib_omop_crosswalk import RATIO_OF_WBC, VAR_CONCEPTS_BY_DATASET
    assert set(TARGET_MEASUREMENT_CONCEPT_IDS) == (set(CONCEPT_TO_YAIB)
                                                 | set(STATIC_MEASUREMENT_CONCEPT_IDS)
                                                 | set(SOFA_MEASUREMENT_CONCEPT_IDS)
                                                 | {c for m in VAR_CONCEPTS_BY_DATASET.values() for ids in m.values()
                                                    for c in ids}
                                                 | {c for m in RATIO_OF_WBC.values() for c in m.values()})
    assert var_concepts("resp", "miiv") == [3024171, 1175625, 3007469] and var_concepts("resp", "eicu") == [3024171, 1175625], \
        "ricu's resp takes the set rate (224688) on miiv only"


def test_ricu_bounds_are_complete_and_correctly_merged():
    """Bounds come from BASE ricu: the ricu-extensions overrides are `sources`-only (ricu's
    `combine_sources` errors otherwise), so reading the extension entry first shadows min/max with
    None. A first attempt at this table did exactly that and produced 32 bounded variables instead
    of 46.
    """
    from yaib_omop_crosswalk import RICU_BOUNDS, YAIB_DYNAMIC_VARS  # noqa: E402

    assert len(RICU_BOUNDS) == 46, f"expected 46 bounded vars, got {len(RICU_BOUNDS)}"
    unbounded = set(YAIB_DYNAMIC_VARS) - set(RICU_BOUNDS)
    assert unbounded == {"bnd", "inr_pt"}, f"unexpected unbounded set: {unbounded}"
    assert set(RICU_BOUNDS) <= set(YAIB_DYNAMIC_VARS), "bounds for a non-YAIB variable"

    # spot-check against the dict, including ones the sources-only shadowing had dropped
    for var, expect in (("glu", (0, 1000)), ("hgb", (4, 18)), ("crea", (0, 15)),
                        ("temp", (32, 42)), ("urine", (0, 2000)), ("bili", (0, 100))):
        assert RICU_BOUNDS[var] == expect, f"{var}: {RICU_BOUNDS[var]} != {expect}"

    # a None bound means "no constraint that side", not zero
    for var, (lo, hi) in RICU_BOUNDS.items():
        assert lo is not None or hi is not None, var
        if lo is not None and hi is not None:
            assert lo < hi, var


# ricu's installed concept dictionary (RICU_LIB_PATH: the installed ricu R package) and the ricu-extensions
# dictionaries YAIB loads on top of it (RICU_EXTENSIONS: ricu-extensions/configs of YAIB-cohorts)
RICU_DICT = os.path.join(os.environ.get("RICU_LIB_PATH", ""), "extdata", "config", "concept-dict.json")
RICU_EXT = os.environ.get("RICU_EXTENSIONS", "")


def test_ricu_bounds_equal_ricus_dictionary():
    """RICU_BOUNDS equals the min/max of every dynamic variable in ricu's dictionary. An extension may only add
    sources to a concept of the base dictionary (ricu's `combine_sources` stops otherwise), so the bounds are the
    base dictionary's, or an extension's for a concept the base does not have."""
    import json

    from yaib_omop_crosswalk import RICU_BOUNDS, YAIB_DYNAMIC_VARS  # noqa: E402

    if not os.path.exists(RICU_DICT):
        pytest.skip(f"no ricu dictionary at {RICU_DICT}")
    with open(RICU_DICT) as fh:
        concepts = json.load(fh)
    for sub in sorted(os.listdir(RICU_EXT)) if os.path.isdir(RICU_EXT) else []:
        path = os.path.join(RICU_EXT, sub, "concept-dict.json")
        if os.path.exists(path):
            with open(path) as fh:
                for name, entry in json.load(fh).items():
                    if name not in concepts:
                        concepts[name] = entry
                    else:
                        assert list(entry) == ["sources"], f"{sub} redefines {name} beyond its sources"
    for var in YAIB_DYNAMIC_VARS:
        lo, hi = concepts[var].get("min"), concepts[var].get("max")
        if lo is None and hi is None:
            assert var not in RICU_BOUNDS, var
        else:
            assert RICU_BOUNDS[var] == (lo, hi), f"{var}: {RICU_BOUNDS[var]} != ricu's {(lo, hi)}"


def test_statics_are_prefiltered_but_never_dynamic():
    assert set(STATIC_MEASUREMENT_CONCEPT_IDS).issubset(TARGET_MEASUREMENT_CONCEPT_IDS)
    assert STATIC_MEASUREMENT_CONCEPT_IDS.keys().isdisjoint(CONCEPT_TO_YAIB)


def test_sofa_inputs_are_prefiltered_but_never_dynamic():
    """gcs and ventilator_mode are LABEL inputs (SOFA), not tensor columns. They must survive the
    prefilter -- otherwise sofa_cns and sofa_resp silently cannot be computed -- while staying out
    of the 48-var dyn frame."""
    assert set(SOFA_MEASUREMENT_CONCEPT_IDS).issubset(TARGET_MEASUREMENT_CONCEPT_IDS)
    assert SOFA_MEASUREMENT_CONCEPT_IDS.keys().isdisjoint(CONCEPT_TO_YAIB)
    assert SOFA_MEASUREMENT_CONCEPT_IDS.keys() <= NON_DYNAMIC_CONCEPTS.keys()


def test_non_dynamic_concepts_stay_out_of_dyn():
    assert NON_DYNAMIC_CONCEPTS.keys().isdisjoint(CONCEPT_TO_YAIB)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


def test_unit_conversions_cover_the_mapped_units():
    """Every unit the ETL mappings can give a row of a converted variable has a conversion, the
    target unit converts as itself, and the factors are the ones ricu uses."""
    import csv
    from pathlib import Path

    from yaib_omop_crosswalk import FAHRENHEIT, MEQ_L, MG_L, MMOL_L, UMOL_L, UNIT_CONVERSIONS, var_concepts
    mappings = Path(__file__).resolve().parents[2] / "omop_etl" / "mappings"
    # The units a row can get beside its mapping row's own unit, per rule (omop_etl/src/omop_etl/
    # units.py). A rule the ETL adds or renames fails here until it is listed.
    unit_strings = {int(r["unit_concept_id"]) for r in csv.DictReader(open(mappings / "units.csv"))}
    per_rule = {"": set(), "unit_from_string": unit_strings, "fahrenheit_above": {FAHRENHEIT}}
    # units.csv units that no row of these variables carries on iter5 (ehrdata's setup_variables raises on
    # an unconverted one); a new unit string for them needs a conversion or an entry here
    not_recorded = {"crp": {MMOL_L, MEQ_L}, "ca": {MG_L, MEQ_L}, "mg": {MG_L}, "cai": {MG_L}}
    for ds, folder in (("miiv", "mimic4"), ("eicu", "eicu"), ("hirid", "hirid")):
        rows = list(csv.DictReader(open(mappings / folder / "measurements.csv")))
        for var, (target, units) in UNIT_CONVERSIONS.items():
            assert units[target] == (1.0, 0.0), var
            ids = set(var_concepts(var, ds))
            for r in rows:
                if int(r["concept_id"]) in ids:
                    assert r["rule"] in per_rule, (ds, var, r["source_code"], r["rule"])
                    recorded = (per_rule[r["rule"]] - not_recorded.get(var, set())) | {int(r["unit_concept_id"])}
                    assert recorded <= set(units), (ds, var, r["source_code"], recorded - set(units))
    assert UNIT_CONVERSIONS["glu"][1][MMOL_L] == (18.016, 0.0)
    assert UNIT_CONVERSIONS["crea"][1][UMOL_L] == (1 / 88.4, 0.0)
    m, o = UNIT_CONVERSIONS["temp"][1][FAHRENHEIT]
    assert abs(98.6 * m + o - 37.0) < 1e-12
