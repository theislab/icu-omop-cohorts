"""Write the small vocabulary that the tests and the demo run use in place of an Athena download.

The fixture holds the concepts that a demo CDM uses and those of the mapping tables, in the format of Athena's
CONCEPT.csv (tab-separated, dates as YYYYMMDD), and Athena's full DOMAIN.csv. Ids, domains, vocabularies,
classes, standard flags and validity are kept for every concept. Names and codes are kept only for
UCUM and for the vocabularies OMOP generates itself, and left empty for the others (LOINC, RxNorm,
SNOMED, ...), whose licenses govern their redistribution. The ETL reads names only for its three
UCUM units (cm, kg, year), so the CDM it builds from the fixture differs from one built from Athena
only in the CONCEPT table's names and codes.

    python omop_etl/tools/make_vocabulary_fixture.py --vocabulary <athena dir> --cdm <demo cdm dir>

The demo CDM is the one `ci/demo.sh` builds of the MIMIC-IV and the eICU demo (`<work dir>/omop/cdm`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb

from omop_etl import cdm, schema

HERE = Path(__file__).resolve().parent
MAPPINGS = HERE.parent / "mappings"
OUT = HERE.parent / "tests" / "fixtures" / "vocabulary"
COLUMNS = ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id", "standard_concept",
           "concept_code", "valid_start_date", "valid_end_date", "invalid_reason"]
NAMED = "vocabulary_id = 'UCUM' OR vocabulary_id IN (SELECT vocabulary_id FROM vocab WHERE vocabulary_reference = 'OMOP generated')"


def athena(path: Path) -> str:
    return f"read_csv('{path}', delim = '\t', quote = '', escape = '', header = true, all_varchar = true)"


def mapping_ids() -> str:
    """SQL: every concept id in the mapping tables."""
    parts = [f"SELECT concept_id FROM read_csv('{MAPPINGS}/conditions.csv', all_varchar = true)",
             f"SELECT concept_id FROM read_csv('{MAPPINGS}/drug_ingredients.csv', all_varchar = true)",
             f"SELECT unit_concept_id FROM read_csv('{MAPPINGS}/units.csv', all_varchar = true)"]
    for table in sorted(MAPPINGS.glob("*/measurements.csv")):
        parts += [f"SELECT concept_id FROM read_csv('{table}', all_varchar = true)",
                  f"SELECT unit_concept_id FROM read_csv('{table}', all_varchar = true)"]
    return " UNION ".join(parts)


def write_tsv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="") as f:
        out = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None, lineterminator="\n")
        out.writerow(header)
        out.writerows(["" if v is None else v for v in row] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vocabulary", type=Path, required=True, help="the unzipped Athena download")
    parser.add_argument("--cdm", type=Path, required=True, help="the demo CDM, whose tables name the used concepts")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute(f"CREATE TABLE vocab AS SELECT * FROM {athena(args.vocabulary / 'VOCABULARY.csv')}")
    tables = [name for name in schema.tables() if name not in ("CONCEPT", "DOMAIN") and (args.cdm / f"{name}.parquet").exists()]
    con.execute(f"""
        CREATE TABLE wanted AS
        SELECT DISTINCT CAST(concept_id AS VARCHAR) AS concept_id FROM ({cdm.used_concepts_sql(args.cdm, tables)})
        UNION SELECT * FROM ({mapping_ids()})""")
    keep = ", ".join(c if c not in ("concept_name", "concept_code") else f"CASE WHEN {NAMED} THEN {c} END AS {c}"
                     for c in COLUMNS)
    rows = con.sql(f"""
        SELECT {keep} FROM {athena(args.vocabulary / 'CONCEPT.csv')}
        WHERE concept_id IN (SELECT concept_id FROM wanted WHERE concept_id IS NOT NULL)
        ORDER BY CAST(concept_id AS INTEGER)""").fetchall()
    missing = con.sql(f"""SELECT count(*) FROM wanted WHERE concept_id IS NOT NULL AND concept_id NOT IN
                          (SELECT concept_id FROM {athena(args.vocabulary / 'CONCEPT.csv')})""").fetchone()[0]
    domains = con.sql(f"SELECT * FROM {athena(args.vocabulary / 'DOMAIN.csv')} ORDER BY domain_id").fetchall()
    domain_header = [d[0] for d in con.sql(f"DESCRIBE SELECT * FROM {athena(args.vocabulary / 'DOMAIN.csv')}").fetchall()]

    for row in rows + domains:
        if any(isinstance(v, str) and ("\t" in v or "\n" in v) for v in row):
            raise ValueError(f"a tab or newline in {row[0]}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / "CONCEPT.csv", COLUMNS, rows)
    write_tsv(args.out / "DOMAIN.csv", domain_header, domains)
    print(f"{len(rows)} concepts ({missing} wanted ids not in the download), {len(domains)} domains -> {args.out}")


if __name__ == "__main__":
    main()
