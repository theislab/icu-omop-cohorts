"""The staging contract between step 2 (`<dataset>.stage`) and step 3 (`cdm`).

Each dataset writes `<out>/staging/<dataset>/<table>.parquet` with exactly the columns below.
Times are offsets from the stay's ICU admission in microseconds (BIGINT, columns `*_us`; not
INTERVAL, which duckdb stores in parquet at millisecond resolution and not below zero);
`stay_id` is the source stay id, `person_key` the source person id. Step 3 adds dataset prefixes,
ids and datetimes.

A dataset module writes a table with `write(con, ds_dir, table, sql)`, which selects and casts the
columns in contract order, and checks the result with `validate(ds_dir)` at the end of `stage`.
`write` also accepts an INTERVAL column named without the `_us` suffix (`t` for `t_us`) and
converts it to microseconds.
"""

from __future__ import annotations

from pathlib import Path

from omop_etl import db

TABLES: dict[str, dict[str, str]] = {
    "stays": {
        "stay_id": "BIGINT",
        "person_key": "VARCHAR",
        # order of the stay within the dataset: picks the person-level age/sex, and lays a person's
        # stays on the synthetic timeline in this order when the source gives no admission time
        "stay_order": "INTEGER",
        # ICU admission on the source's clock, microseconds since 1970-01-01 (MIMIC-IV and HiRID are
        # date-shifted, eICU is placed within its discharge year); null puts the stay on the
        # synthetic timeline
        "admit_us": "BIGINT",
        "year_of_birth": "DOUBLE",  # null: the year of the admission minus the age
        "los_us": "BIGINT",  # null if unknown
        "age": "DOUBLE",
        "sex": "VARCHAR",  # 'M', 'F' or null
        "hospital": "VARCHAR",
        "unit_type": "VARCHAR",
        "admission_source": "VARCHAR",
        "discharge_location": "VARCHAR",
        "mortality": "BOOLEAN",  # null = unknown outcome
        "icu_death": "BOOLEAN",
        "death_t_us": "BIGINT",  # null if no death time
        "height": "DOUBLE",  # averaged admission value, null if none
        "weight": "DOUBLE",
    },
    "measurement": {
        "stay_id": "BIGINT",
        "t_us": "BIGINT",
        "concept_id": "INTEGER",
        "source_value": "VARCHAR",
        "value": "DOUBLE",
        "value_text": "VARCHAR",
        "unit_source_value": "VARCHAR",
        "unit_concept_id": "INTEGER",
    },
    "drug": {
        "stay_id": "BIGINT",
        "t_start_us": "BIGINT",
        "t_end_us": "BIGINT",
        "source_value": "VARCHAR",
        "quantity": "DOUBLE",
        "dose_unit": "VARCHAR",
        "route": "VARCHAR",
        "sig": "VARCHAR",
        "admission_drug": "BOOLEAN",
    },
    "procedure": {
        "stay_id": "BIGINT",
        "t_start_us": "BIGINT",
        "t_end_us": "BIGINT",
        "concept_id": "INTEGER",
        "source_value": "VARCHAR",
    },
    "specimen": {
        "stay_id": "BIGINT",
        "t_us": "BIGINT",
        "concept_id": "INTEGER",
        "source_value": "VARCHAR",
        "source_id": "VARCHAR",
    },
    "condition": {
        "stay_id": "BIGINT",
        "t_us": "BIGINT",
        "concept_id": "INTEGER",
        "source_value": "VARCHAR",
    },
}

REQUIRED = ("stays", "measurement", "drug")

# columns that may not be null in any row
NOT_NULL = {
    "stays": ("stay_id", "person_key", "stay_order"),
    "measurement": ("stay_id", "t_us", "concept_id", "source_value"),
    "drug": ("stay_id", "t_start_us", "t_end_us"),
    "procedure": ("stay_id", "t_start_us", "concept_id"),
    "specimen": ("stay_id", "t_us", "concept_id"),
    "condition": ("stay_id", "t_us", "concept_id"),
}


def path(ds_dir: Path | str, table: str) -> Path:
    return Path(ds_dir) / f"{table}.parquet"


def interval_us(expr: str) -> str:
    """SQL: an INTERVAL expression as BIGINT microseconds (days count 24 h, no months)."""
    return f"(epoch_us(TIMESTAMP '2000-01-01' + ({expr})) - epoch_us(TIMESTAMP '2000-01-01'))"


def select_sql(table: str, sql: str, intervals: tuple[str, ...] = ()) -> str:
    """`sql` wrapped so that it yields exactly the contract columns, in order, with their types.

    `intervals`: `*_us` columns that the query delivers as an INTERVAL without the suffix.
    """
    def col(c, t):
        if c in intervals:
            return f"CAST({interval_us(c.removesuffix('_us'))} AS {t}) AS {c}"
        return f"CAST({c} AS {t}) AS {c}"
    cols = ",\n    ".join(col(c, t) for c, t in TABLES[table].items())
    return f"SELECT\n    {cols}\nFROM (\n{sql}\n)"


def write(con, ds_dir: Path | str, table: str, sql: str) -> int:
    """Write staging table `table` from a query whose columns carry the contract names."""
    target = path(ds_dir, table)
    target.parent.mkdir(parents=True, exist_ok=True)
    got = {row[0]: row[1] for row in con.sql(f"DESCRIBE {sql}").fetchall()}
    intervals = tuple(c for c in TABLES[table]
                      if c.endswith("_us") and c not in got and got.get(c.removesuffix("_us")) == "INTERVAL")
    return db.copy(con, select_sql(table, sql, intervals), target)


def validate(ds_dir: Path | str, con=None) -> dict[str, int]:
    """Check names, types and key nulls of every staging file in `ds_dir`; returns row counts.

    Raises ValueError listing every problem found.
    """
    ds_dir = Path(ds_dir)
    con = con or db.connect_plain()
    problems, counts = [], {}
    for name in REQUIRED:
        if not path(ds_dir, name).exists():
            problems.append(f"{name}.parquet missing")
    for name, columns in TABLES.items():
        p = path(ds_dir, name)
        if not p.exists():
            continue
        schema = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
        got = {row[0]: row[1] for row in schema}
        if list(got) != list(columns):
            problems.append(f"{name}: columns {list(got)}, expected {list(columns)}")
        for col, typ in columns.items():
            if col in got and got[col] != typ:
                problems.append(f"{name}.{col}: type {got[col]}, expected {typ}")
        checks = ", ".join(f"count(*) FILTER ({c} IS NULL)" for c in NOT_NULL[name] if c in got)
        row = con.sql(f"SELECT count(*), {checks} FROM read_parquet('{p}')").fetchone()
        counts[name] = row[0]
        for col, n_null in zip([c for c in NOT_NULL[name] if c in got], row[1:]):
            if n_null:
                problems.append(f"{name}.{col}: {n_null} nulls")
        if name == "stays":
            dup = con.sql(f"SELECT count(*) - count(DISTINCT stay_id) FROM read_parquet('{p}')").fetchone()[0]
            if dup:
                problems.append(f"stays: {dup} duplicate stay_id")
    if problems:
        raise ValueError(f"staging {ds_dir} violates the contract:\n  " + "\n  ".join(problems))
    return counts
