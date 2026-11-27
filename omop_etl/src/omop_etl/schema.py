"""OMOP CDM v5.4 tables, columns and duckdb types, from the OHDSI field-level spec
(`schema/OMOP_CDMv5.4_Field_Level.csv`).

Types: surrogate keys (`*_id`, not `*_concept_id`) BIGINT, other integers INTEGER, float DOUBLE,
varchar VARCHAR, date DATE, datetime TIMESTAMP (µs). `OVERRIDES` lists the exceptions.
"""

from __future__ import annotations

import csv
from functools import cache

from omop_etl.config import SCHEMA_CSV

OVERRIDES = {
    # concept_id is the vocabulary's own key, an integer
    ("CONCEPT", "concept_id"): "INTEGER",
    # the surrogate keys of the linked rows
    ("FACT_RELATIONSHIP", "fact_id_1"): "BIGINT",
    ("FACT_RELATIONSHIP", "fact_id_2"): "BIGINT",
}


def _duck_type(table: str, column: str, spec_type: str) -> str:
    if (table, column) in OVERRIDES:
        return OVERRIDES[(table, column)]
    t = spec_type.lower()
    if t in ("integer", "bigint"):
        if column.endswith("_id") and "concept_id" not in column:
            return "BIGINT"
        return "BIGINT" if t == "bigint" else "INTEGER"
    if t == "float":
        return "DOUBLE"
    if t.startswith("varchar"):
        return "VARCHAR"
    if t == "date":
        return "DATE"
    if t == "datetime":
        return "TIMESTAMP"
    raise ValueError(f"unknown CDM datatype {spec_type!r} for {table}.{column}")


@cache
def tables() -> dict[str, dict[str, str]]:
    """{TABLE: {column: duckdb type}} in spec order, the 39 tables of CDM v5.4."""
    out: dict[str, dict[str, str]] = {}
    with open(SCHEMA_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            table, column = row["cdmTableName"].upper(), row["cdmFieldName"].lower()
            out.setdefault(table, {})[column] = _duck_type(table, column, row["cdmDatatype"])
    return out


def select_sql(table: str, sql: str, columns_present: set[str] | None = None,
               types: dict[str, str] | None = None) -> str:
    """`sql` wrapped to yield every column of `table` in spec order with its type; columns the
    query does not produce (`columns_present` given) are typed NULLs. `types` overrides types."""
    cols = []
    for c, t in tables()[table].items():
        src = c if columns_present is None or c in columns_present else "NULL"
        cols.append(f"CAST({src} AS {(types or {}).get(c, t)}) AS {c}")
    return "SELECT\n    " + ",\n    ".join(cols) + f"\nFROM (\n{sql}\n)"


def empty_sql(table: str) -> str:
    return "SELECT " + ", ".join(f"CAST(NULL AS {t}) AS {c}" for c, t in tables()[table].items()) + " LIMIT 0"
