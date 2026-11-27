"""The user's Athena download, read at run time. Nothing from it is shipped with the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("omop_etl")

CONCEPT_COLUMNS = {
    "concept_id": "INTEGER", "concept_name": "VARCHAR", "domain_id": "VARCHAR",
    "vocabulary_id": "VARCHAR", "concept_class_id": "VARCHAR", "standard_concept": "VARCHAR",
    "concept_code": "VARCHAR", "valid_start_date": "DATE", "valid_end_date": "DATE",
    "invalid_reason": "VARCHAR",
}


def _athena_csv(vocabulary_dir: Path, name: str) -> str:
    path = Path(vocabulary_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: vocabulary_dir must hold the unzipped Athena download")
    # Athena files are tab-separated without quoting (names contain quote characters)
    return f"read_csv('{path}', delim = '\t', quote = '', escape = '', header = true, all_varchar = true)"


def load_concept(con, vocabulary_dir: Path) -> None:
    """Table `athena_concept` with the typed CONCEPT.csv (dates are YYYYMMDD in Athena)."""
    def col(c, t):
        if t == "DATE":
            return f"CAST(strptime({c}, '%Y%m%d') AS DATE) AS {c}"
        return f"CAST({c} AS {t}) AS {c}"
    cols = ", ".join(col(c, t) for c, t in CONCEPT_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE athena_concept AS SELECT {cols} FROM {_athena_csv(vocabulary_dir, 'CONCEPT')}")
    n = con.sql("SELECT count(*) FROM athena_concept").fetchone()[0]
    log.info("vocabulary: %s concepts from %s", f"{n:,}", vocabulary_dir)


def domain_sql(vocabulary_dir: Path) -> str:
    return (f"SELECT domain_id, domain_name, CAST(domain_concept_id AS INTEGER) AS domain_concept_id "
            f"FROM {_athena_csv(vocabulary_dir, 'DOMAIN')}")
