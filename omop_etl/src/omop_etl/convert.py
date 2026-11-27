"""Step 1 helpers: source files -> one typed parquet per table under `<out>/raw/<dataset>/`.

Only the columns a dataset module needs are converted, each with an explicit type. Columns are
read as text first and cast in SQL, so a value that does not parse fails loudly instead of being
guessed by a type sniffer.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omop_etl import db

log = logging.getLogger("omop_etl")

# Field values that mean missing (pandas.read_csv's default `na_values`) are NULL.
PANDAS_NA = ("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
             "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null")
_NA_LIST = ", ".join("'" + s.replace("'", "''") + "'" for s in PANDAS_NA)


def na(col: str) -> str:
    """SQL: the text column `col` with the missing-value strings as NULL."""
    return f"CASE WHEN {col} IN ({_NA_LIST}) THEN NULL ELSE {col} END"


def typed(col: str, typ: str) -> str:
    """SQL: text column `col` cast to `typ` after the missing-value mapping."""
    if typ.upper() == "VARCHAR":
        return f"{na(col)} AS {col}"
    return f"CAST({na(col)} AS {typ}) AS {col}"


def read_csv_sql(src: Path | str, delim: str = ",") -> str:
    """A `read_csv` call that reads every column as text, header on, quotes `"`."""
    return f"read_csv('{src}', all_varchar = true, header = true, delim = '{delim}', quote = '\"')"


def csv_to_parquet(con, src: Path | str, target: Path | str, columns: dict[str, str],
                   delim: str = ",") -> int:
    """Convert columns {name: duckdb type} of one csv(.gz) file to a parquet file."""
    select = ",\n    ".join(typed(c, t) for c, t in columns.items())
    return db.copy(con, f"SELECT\n    {select}\nFROM {read_csv_sql(src, delim)}", target)


def table_to_parquet(con, from_sql: str, target: Path | str, columns: dict[str, str]) -> int:
    """Convert columns {name: SQL expression or duckdb type} of any duckdb relation (e.g.
    `read_parquet('dir/*.parquet')`) to a parquet file. A value that is a plain type name is a
    CAST of the column of that name; anything else is used as the expression."""
    def expr(c, e):
        return f"CAST({c} AS {e}) AS {c}" if e.replace(" ", "").isalnum() else f"{e} AS {c}"
    select = ",\n    ".join(expr(c, e) for c, e in columns.items())
    return db.copy(con, f"SELECT\n    {select}\nFROM {from_sql}", target)


def run_parallel(con, jobs: dict[str, tuple], workers: int = 4) -> dict[str, int]:
    """Run `fn(cursor, *args)` for each `name: (fn, *args)` in threads, one duckdb cursor each.

    gzip csv is read by one thread per file, so converting tables side by side is what makes the
    step fast. Returns {name: rows}. Re-raises the first failure after the others finished.
    """
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, con.cursor(), *args): name for name, (fn, *args) in jobs.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:  # noqa: BLE001 - reported below
                log.error("converting %s failed: %s", name, e)
                errors[name] = e
    if errors:
        raise RuntimeError(f"conversion failed for {sorted(errors)}") from next(iter(errors.values()))
    return results


def check_raw(con, path: Path | str, columns, dataset: str) -> None:
    """Raise if the raw parquet at `path` lacks columns that staging reads, i.e. it was written by an
    older convert step."""
    have = {row[0] for row in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    missing = [c for c in columns if c not in have]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}: it predates the current convert step. "
                         f"Re-run `python -m omop_etl convert --dataset {dataset}` with this config.")
