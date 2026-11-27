"""Compare two cohort builds (`<root>/<task>/<dataset>/{dyn,sta,outc}.parquet`), aggregates only.

    python compare_cohorts.py <base_frames_root> <cand_frames_root> [--tasks ...] [--datasets ...]
                              [--out <dir>] [--tol 1e-6]

Per (task, dataset):
- stays: in both, only in base, only in cand (stay_id of sta);
- sta joined on stay_id, dyn on (stay_id, time), outc on (stay_id, time) for the hourly tasks and
  stay_id for the static ones: rows in both / only one side, and per column the number of joined
  cells whose null-ness differs (NaN counts as null) and whose |a - b| > --tol (strings: differ);
- label agreement on the joined outc rows: Cohen's kappa when both sides are 0/1, else the mean
  absolute difference; label mean per side;
- duplicate join keys per side (should be 0; duplicates would multiply the join).

Counts between 1 and 9 are printed and written as "<10". The verdict per cell is computed before
that suppression: IDENTICAL means the same stays, the same rows and no differing cell.
The comparison of each cell runs in duckdb; the largest cell (eicu los, ~20 M dyn rows) needs a
few GB.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import duckdb

TASKS = ["mortality24", "kidney_function", "los", "aki", "sepsis"]
DATASETS = ["miiv", "eicu", "hirid"]
FRAMES = ["sta", "dyn", "outc"]
MIN_COUNT = 10


def lit(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sup(n):
    """Suppress small counts for output."""
    if n is None:
        return ""
    if isinstance(n, int) and 0 < n < MIN_COUNT:
        return f"<{MIN_COUNT}"
    return n


class Cell:
    def __init__(self, con, base: Path, cand: Path, task: str, ds: str, tol: float):
        self.con, self.task, self.ds, self.tol = con, task, ds, tol
        self.dirs = {"b": base / task / ds, "c": cand / task / ds}
        self.rows: list[dict] = []      # long format: frame, column, metric, value
        self.differs = False

    def add(self, frame, column, metric, value, counts_as_diff=False):
        if counts_as_diff and value:
            self.differs = True
        self.rows.append({"task": self.task, "dataset": self.ds, "frame": frame, "column": column,
                          "metric": metric, "value": value})

    def schema(self, side, frame):
        f = self.dirs[side] / f"{frame}.parquet"
        return {r[0]: r[1] for r in self.con.execute(f"DESCRIBE SELECT * FROM read_parquet({lit(f)})").fetchall()}

    def run(self) -> None:
        present = {s: all((d / f"{fr}.parquet").exists() for fr in FRAMES) for s, d in self.dirs.items()}
        if not all(present.values()):
            self.add("-", "-", "missing_side", ",".join(s for s, ok in present.items() if not ok) or "-", True)
            return
        b_sta, c_sta = (f"read_parquet({lit(self.dirs[s] / 'sta.parquet')})" for s in "bc")
        n = self.con.execute(f"""
            SELECT count(*) FILTER (WHERE b.stay_id IS NOT NULL AND c.stay_id IS NOT NULL),
                   count(*) FILTER (WHERE c.stay_id IS NULL), count(*) FILTER (WHERE b.stay_id IS NULL)
            FROM (SELECT DISTINCT stay_id FROM {b_sta}) b FULL OUTER JOIN (SELECT DISTINCT stay_id FROM {c_sta}) c
              USING (stay_id)""").fetchone()
        self.add("stays", "stay_id", "both", n[0])
        self.add("stays", "stay_id", "only_base", n[1], True)
        self.add("stays", "stay_id", "only_cand", n[2], True)
        for frame in FRAMES:
            self.frame(frame)

    def frame(self, frame: str) -> None:
        sb, sc = self.schema("b", frame), self.schema("c", frame)
        keys = ["stay_id"] + (["time"] if "time" in sb and "time" in sc else [])
        if ("time" in sb) != ("time" in sc):
            self.add(frame, "time", "key_on_one_side_only", 1, True)
        for side, sch in (("base", sb), ("cand", sc)):
            other = sc if side == "base" else sb
            for col in sch:
                if col not in other:
                    self.add(frame, col, f"column_only_in_{side}", 1, True)
        cols = [c for c in sb if c in sc and c not in keys]
        src = {s: f"read_parquet({lit(self.dirs[s] / f'{frame}.parquet')})" for s in "bc"}
        kq = ", ".join(q(k) for k in keys)
        for s, side in (("b", "base"), ("c", "cand")):
            nrow, nkey = self.con.execute(
                f"SELECT count(*), count(DISTINCT ({kq})) FROM {src[s]}").fetchone()
            self.add(frame, "-", f"rows_{side}", nrow)
            self.add(frame, "-", f"duplicate_keys_{side}", nrow - nkey, True)

        def nullish(alias, col, typ):
            num = any(t in typ for t in ("DOUBLE", "FLOAT", "REAL"))
            return f"({alias}.{q(col)} IS NULL" + (f" OR isnan({alias}.{q(col)}))" if num else ")")

        both = "b._in AND c._in"
        exprs = [f"count(*) FILTER (WHERE {both})", "count(*) FILTER (WHERE c._in IS NULL)",
                 "count(*) FILTER (WHERE b._in IS NULL)"]
        for col in cols:
            nb, nc = nullish("b", col, sb[col]), nullish("c", col, sc[col])
            numeric = all(any(t in typ for t in ("DOUBLE", "FLOAT", "INT", "DECIMAL", "REAL"))
                          for typ in (sb[col], sc[col]))
            exprs.append(f"count(*) FILTER (WHERE {both} AND {nb} <> {nc})")
            if numeric:
                exprs.append(f"count(*) FILTER (WHERE {both} AND NOT {nb} AND NOT {nc} AND "
                             f"abs(b.{q(col)}::DOUBLE - c.{q(col)}::DOUBLE) > {self.tol})")
                exprs.append(f"max(abs(b.{q(col)}::DOUBLE - c.{q(col)}::DOUBLE)) FILTER "
                             f"(WHERE {both} AND NOT {nb} AND NOT {nc})")
            else:
                exprs.append(f"count(*) FILTER (WHERE {both} AND NOT {nb} AND NOT {nc} AND "
                             f"b.{q(col)}::VARCHAR <> c.{q(col)}::VARCHAR)")
                exprs.append("NULL::DOUBLE")
        on = " AND ".join(f"b.{q(k)} = c.{q(k)}" for k in keys)
        res = self.con.execute(f"""
            SELECT {', '.join(exprs)}
            FROM (SELECT *, true AS _in FROM {src['b']}) b FULL OUTER JOIN (SELECT *, true AS _in FROM {src['c']}) c
              ON {on}""").fetchone()
        self.add(frame, "-", "rows_both", res[0])
        self.add(frame, "-", "rows_only_base", res[1], True)
        self.add(frame, "-", "rows_only_cand", res[2], True)
        for i, col in enumerate(cols):
            nn, vd, mx = res[3 + 3 * i: 6 + 3 * i]
            self.add(frame, col, "nullness_differs", nn, True)
            self.add(frame, col, "value_differs", vd, True)
            if mx:
                self.add(frame, col, "max_abs_diff", float(mx))
        if frame == "outc" and "label" in cols:
            self.label(src, on, nullish("b", "label", sb["label"]), nullish("c", "label", sc["label"]))

    def label(self, src, on, nb, nc) -> None:
        r = self.con.execute(f"""
            SELECT count(*), avg(b.label::DOUBLE), avg(c.label::DOUBLE),
                   bool_and(b.label::DOUBLE IN (0, 1) AND c.label::DOUBLE IN (0, 1)),
                   avg((b.label::DOUBLE = c.label::DOUBLE)::INT), avg(abs(b.label::DOUBLE - c.label::DOUBLE)),
                   corr(b.label::DOUBLE, c.label::DOUBLE)
            FROM {src['b']} b JOIN {src['c']} c ON {on}
            WHERE NOT {nb} AND NOT {nc}""").fetchone()
        n, mb, mc, binary, po, mad, cor = r
        self.add("outc", "label", "n_joined_nonnull", n)
        if not n:
            return
        self.add("outc", "label", "mean_base", round(mb, 6))
        self.add("outc", "label", "mean_cand", round(mc, 6))
        if binary:
            pe = mb * mc + (1 - mb) * (1 - mc)
            kappa = 1.0 if pe == 1 and po == 1 else (po - pe) / (1 - pe) if pe != 1 else float("nan")
            self.add("outc", "label", "kappa", round(kappa, 6))
        else:
            self.add("outc", "label", "mad", round(mad, 6))
            if cor is not None:
                self.add("outc", "label", "corr", round(cor, 6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, help="write compare_cohorts.csv / .json here")
    ap.add_argument("--mem", default=os.environ.get("CC_MEMORY", "16GB"))
    ap.add_argument("--only-present", action="store_true",
                    help="skip cells that the candidate does not have (e.g. a partial build)")
    args = ap.parse_args()

    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{args.mem}'")
    con.execute(f"SET threads={int(os.environ.get('SLURM_CPUS_PER_TASK') or 4)}")
    # duckdb's spill, next to the candidate build: it holds patient rows (DUCK_TMP_BASE overrides)
    tmp = Path(os.environ.get("DUCK_TMP_BASE") or args.cand.resolve().parent / "_duckdb_tmp") / \
        f"{os.environ.get('SLURM_JOB_ID') or 'pid' + str(os.getpid())}_compare_cohorts"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={lit(tmp)}")

    rows, verdicts = [], []
    t0 = time.time()
    for task in args.tasks.split(","):
        for ds in args.datasets.split(","):
            if args.only_present and not (args.cand / task / ds / "sta.parquet").exists():
                continue
            if not (args.base / task / ds).exists() and not (args.cand / task / ds).exists():
                continue
            cell = Cell(con, args.base, args.cand, task, ds, args.tol)
            cell.run()
            rows += cell.rows
            verdicts.append((task, ds, "DIFFERENT" if cell.differs else "IDENTICAL"))
    con.close()
    if not verdicts:
        sys.exit(f"no (task, dataset) cell found under {args.base} or {args.cand}")
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()

    print(f"base {args.base}\ncand {args.cand}\n")
    for task, ds, v in verdicts:
        print(f"{task:16} {ds:6} {v}")
    print()
    for task, ds, v in verdicts:
        cell_rows = [r for r in rows if r["task"] == task and r["dataset"] == ds]
        get = {(r["frame"], r["column"], r["metric"]): r["value"] for r in cell_rows}
        print(f"== {task} / {ds}: {v}")
        if ("-", "-", "missing_side") in get:
            print(f"   missing on side(s): {get[('-', '-', 'missing_side')]}")
            continue
        print(f"   stays both {sup(get[('stays', 'stay_id', 'both')])}, only base "
              f"{sup(get[('stays', 'stay_id', 'only_base')])}, only cand {sup(get[('stays', 'stay_id', 'only_cand')])}")
        for fr in FRAMES:
            print(f"   {fr:4} rows base {sup(get[(fr, '-', 'rows_base')])}, cand {sup(get[(fr, '-', 'rows_cand')])}, "
                  f"both {sup(get[(fr, '-', 'rows_both')])}, only base {sup(get[(fr, '-', 'rows_only_base')])}, "
                  f"only cand {sup(get[(fr, '-', 'rows_only_cand')])}, dup keys "
                  f"{sup(get[(fr, '-', 'duplicate_keys_base')])}/{sup(get[(fr, '-', 'duplicate_keys_cand')])}")
            bad = [r for r in cell_rows if r["frame"] == fr and r["column"] != "-"
                   and r["metric"] in ("nullness_differs", "value_differs") and r["value"]]
            for r in bad:
                mx = get.get((fr, r["column"], "max_abs_diff"))
                print(f"        {r['column']:12} {r['metric']:17} {sup(r['value'])}"
                      + (f"  max|a-b| {mx:.6g}" if mx and r["metric"] == "value_differs" else ""))
            for r in cell_rows:
                if r["frame"] == fr and r["metric"].startswith("column_only") or (
                        r["frame"] == fr and r["metric"] == "key_on_one_side_only"):
                    print(f"        {r['column']}: {r['metric']}")
        lab = {m: get.get(("outc", "label", m)) for m in ("n_joined_nonnull", "mean_base", "mean_cand", "kappa", "mad", "corr")}
        print("   label " + ", ".join(f"{k} {sup(v)}" for k, v in lab.items() if v is not None))
    print(f"\n{len(verdicts)} cell(s), {sum(v == 'IDENTICAL' for *_, v in verdicts)} identical, "
          f"{time.time() - t0:.0f} s")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "compare_cohorts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "dataset", "frame", "column", "metric", "value"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "value": sup(r["value"])})
        (args.out / "compare_cohorts.json").write_text(json.dumps(
            {"base": str(args.base), "cand": str(args.cand), "tol": args.tol,
             "verdicts": [{"task": t, "dataset": d, "verdict": v} for t, d, v in verdicts]}, indent=2))
    sys.exit(0 if all(v == "IDENTICAL" for *_, v in verdicts) else 1)


if __name__ == "__main__":
    main()
