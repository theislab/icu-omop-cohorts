"""The unit of a row whose unit varies per row.

Mapping rows with rule `unit_from_string` take the unit from the row's unit string through
mappings/units.csv (lower case, blanks removed), else the unit of the mapping row. Rows with rule
`fahrenheit_above` are in Fahrenheit above FAHRENHEIT_ABOVE, else in the mapping row's unit.
"""

from __future__ import annotations

from omop_etl import config

FAHRENHEIT = 9289
# A temperature above this under a Celsius item is a Fahrenheit value. No value between ricu's
# upper bound (42 °C) and this is plausible in either unit, so the choice does not change what
# survives the bounds.
FAHRENHEIT_ABOVE = 50
UNIT_STRINGS = config.MAPPINGS / "units.csv"


def load(con) -> None:
    """Table `unit_strings` (unit_string, unit_concept_id)."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE unit_strings AS
                    SELECT unit_string, unit_concept_id FROM read_csv('{UNIT_STRINGS}', header = true,
                        columns = {{'unit_string': 'VARCHAR', 'unit_concept_id': 'INTEGER', 'unit_name': 'VARCHAR'}})""")


def key(unit_string: str) -> str:
    """SQL: the unit_strings key of a unit string column."""
    return f"lower(replace(trim({unit_string}), ' ', ''))"


def unit_sql(rule: str, value: str, unit_from_string: str, default: str) -> str:
    """SQL: the unit concept of a row. `unit_from_string` is the unit_strings match of the row's
    unit string (NULL when none), `default` the mapping row's unit."""
    return f"""CASE {rule}
            WHEN 'unit_from_string' THEN coalesce({unit_from_string}, {default})
            WHEN 'fahrenheit_above' THEN CASE WHEN {value} > {FAHRENHEIT_ABOVE} THEN {FAHRENHEIT} ELSE {default} END
            ELSE {default} END"""
