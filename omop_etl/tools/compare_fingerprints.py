"""Compare two CDM fingerprints written by fingerprint_cdm.py.

    python compare_fingerprints.py <baseline_dir> <candidate_dir> [--count-tol 0] [--stat-tol 0.01]

Prints a report: summary and verdict first, then tables missing/extra, schema differences, row
counts per table and per dataset, join integrity, concept groups present on one side only, groups
whose counts or distributions differ, and dataset-level metrics.

Counts are compared exactly unless --count-tol (relative) is given. A distribution statistic
(quantile or mean) is within tolerance when |a - b| <= max(stat_tol * max(|a|, |b|), floor), with
floor = --time-tol-h for hour-valued statistics and --value-abs-tol otherwise. "Identical" allows
1e-9 relative, since parallel float sums are not bit-reproducible.

Only counts and relative differences are printed. For a group with fewer than --small rows on both
sides no statistic is printed at all ("small group").

Exit status: 0 when everything is identical or within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["table_name", "dataset", "concept_id", "source_value", "unit_source_value", "unit_concept_id"]
NULL = "<null>"
EXACT_REL = 1e-9
# a violation in the candidate is reported even when the baseline has the same count
VIOLATIONS = ("orphan", "mismatch", "duplicate", "end_before_start", "overlaps", "null",
              "outside", "without", "other_dataset", "multiple")


# --- loading ----------------------------------------------------------------------------------
def load(d: Path) -> dict:
    out = {name: pd.read_parquet(d / f"{name}.parquet")
           for name in ("tables", "columns", "integrity", "groups", "metrics")}
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


def key_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        s = df[c]
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            df[c] = s.astype("Int64").astype(str).replace("<NA>", NULL)
        else:
            df[c] = s.astype(object).where(s.notna(), NULL).astype(str)
    return df


# --- formatting -------------------------------------------------------------------------------
def n(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{int(x):,}"


def rel(a, b) -> float:
    a, b = float(a), float(b)
    if a == b:
        return 0.0
    return math.inf if a == 0 else (b - a) / abs(a)


def pct(r: float) -> str:
    if math.isinf(r):
        return "new"
    return f"{r * 100:+.3g}%"


def delta(a, b) -> str:
    a = 0 if a is None or (isinstance(a, float) and math.isnan(a)) else a
    b = 0 if b is None or (isinstance(b, float) and math.isnan(b)) else b
    return f"{n(a)} -> {n(b)} ({int(b) - int(a):+,}, {pct(rel(a, b))})"


def short(s: str, width: int = 40) -> str:
    s = str(s)
    return s if len(s) <= width else s[: width - 3] + "..."


class Report:
    def __init__(self, max_rows: int):
        self.lines: list[str] = []
        self.max_rows = max_rows

    def h(self, title: str) -> None:
        self.lines += ["", title, "-" * len(title)]

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def rows(self, items: list[str], indent: str = "  ") -> None:
        for line in items[: self.max_rows]:
            self.lines.append(indent + line)
        if len(items) > self.max_rows:
            self.lines.append(f"{indent}... {len(items) - self.max_rows} more (--max-rows)")


# --- comparison -------------------------------------------------------------------------------
class Comparer:
    def __init__(self, base: dict, cand: dict, args):
        self.b, self.c, self.a = base, cand, args
        self.r = Report(args.max_rows)
        self.summary: list[tuple[str, str]] = []
        self.worst = 0   # 0 identical, 1 within tolerance, 2 different
        self.diff_frames: dict[str, pd.DataFrame] = {}

    def grade(self, level: int) -> None:
        self.worst = max(self.worst, level)

    def count_level(self, a, b) -> int:
        a = 0 if pd.isna(a) else float(a)
        b = 0 if pd.isna(b) else float(b)
        if a == b:
            return 0
        return 1 if abs(b - a) <= self.a.count_tol * max(abs(a), 1.0) else 2

    def stat_level(self, a, b, floor: float) -> int:
        na, nb = pd.isna(a), pd.isna(b)
        if na and nb:
            return 0
        if na or nb:
            return 2
        d, scale = abs(b - a), max(abs(a), abs(b))
        if d <= EXACT_REL * scale:
            return 0
        return 1 if d <= max(self.a.stat_tol * scale, floor) else 2

    def floor(self, col: str) -> float:
        return self.a.time_tol_h if ("_h_" in col or col.startswith("los_h")) else self.a.value_abs_tol

    # --- tables, schema -------------------------------------------------------------------------
    def tables(self) -> None:
        tb, tc = self.b["tables"].set_index("table_name"), self.c["tables"].set_index("table_name")
        missing, extra = sorted(set(tb.index) - set(tc.index)), sorted(set(tc.index) - set(tb.index))
        common = sorted(set(tb.index) & set(tc.index))
        self.r.h("Tables")
        self.r.add(f"  missing in candidate: {', '.join(missing) or 'none'}")
        self.r.add(f"  extra in candidate:   {', '.join(extra) or 'none'}")
        # an empty table that is missing or extra is noise for the consumer, a filled one is not
        heavy = [t for t in missing if tb.loc[t, "n_rows"] > 0] + [t for t in extra if tc.loc[t, "n_rows"] > 0]
        light = len(missing) + len(extra) - len(heavy)
        if heavy:
            self.grade(2)
        elif light:
            self.grade(1)
        rows, hash_diff = [], []
        n_same = 0
        for t in common:
            a, b = tb.loc[t], tc.loc[t]
            if a["n_rows"] != b["n_rows"]:
                rows.append(f"{t:<24} rows {delta(a['n_rows'], b['n_rows'])}   "
                            f"bytes {pct(rel(a['file_bytes'], b['file_bytes']))}")
                self.grade(self.count_level(a["n_rows"], b["n_rows"]))
            else:
                n_same += 1
            if (a["key_hash"] or "") != (b["key_hash"] or ""):
                hash_diff.append(t)
                self.grade(2)
        self.r.add(f"  row counts: {n_same} of {len(common)} common tables identical")
        self.r.rows(rows, "    ")
        if hash_diff:
            self.r.add(f"  vocabulary key hash differs: {', '.join(hash_diff)}")
        self.summary.append(("tables", f"{len(common)} common, {len(missing)} missing "
                                       f"({len([t for t in missing if tb.loc[t, 'n_rows'] > 0])} non-empty), "
                                       f"{len(extra)} extra; row counts differ in {len(rows)}"
                                       + (f"; vocab hash differs in {len(hash_diff)}" if hash_diff else "")))
        self.common_tables = common

    def schema(self) -> None:
        cb = self.b["columns"].set_index(["table_name", "column_name"])
        cc = self.c["columns"].set_index(["table_name", "column_name"])
        self.r.h("Schema (common tables)")
        n_missing = n_extra = n_type = 0
        null_rows, distinct_rows = [], []
        for t in self.common_tables:
            colb = set(cb.loc[t].index) if t in cb.index.get_level_values(0) else set()
            colc = set(cc.loc[t].index) if t in cc.index.get_level_values(0) else set()
            miss, ext = sorted(colb - colc), sorted(colc - colb)
            types = [(c, cb.loc[(t, c), "type"], cc.loc[(t, c), "type"]) for c in sorted(colb & colc)
                     if cb.loc[(t, c), "type"] != cc.loc[(t, c), "type"]]
            n_missing, n_extra, n_type = n_missing + len(miss), n_extra + len(ext), n_type + len(types)
            if miss or ext or types:
                self.r.add(f"  {t}")
                if miss:
                    self.r.add(f"    missing columns: {', '.join(miss)}")
                if ext:
                    self.r.add(f"    extra columns:   {', '.join(ext)}")
                for c, x, y in types:
                    self.r.add(f"    type {c}: {x} -> {y}")
            for c in sorted(colb & colc):
                x, y = cb.loc[(t, c)], cc.loc[(t, c)]
                if x["n_null"] != y["n_null"]:
                    null_rows.append(f"{t}.{c}: nulls {delta(x['n_null'], y['n_null'])}")
                    self.grade(self.count_level(x["n_null"], y["n_null"]))
                if not (pd.isna(x["n_distinct"]) and pd.isna(y["n_distinct"])) and x["n_distinct"] != y["n_distinct"]:
                    distinct_rows.append(f"{t}.{c}: distinct {delta(x['n_distinct'], y['n_distinct'])}")
                    self.grade(self.count_level(x["n_distinct"], y["n_distinct"]))
        if n_missing or n_extra or n_type:
            self.grade(2)
        else:
            self.r.add("  columns and types identical")
        self.r.add(f"  null counts differ: {len(null_rows)} columns")
        self.r.rows(null_rows, "    ")
        self.r.add(f"  distinct counts of id columns differ: {len(distinct_rows)} columns")
        self.r.rows(distinct_rows, "    ")
        self.summary.append(("schema", f"{n_missing} columns missing, {n_extra} extra, {n_type} type changes; "
                                       f"null counts differ in {len(null_rows)} columns, id distinct "
                                       f"counts in {len(distinct_rows)}"))

    # --- integrity and per-dataset rows -----------------------------------------------------------
    def integrity(self) -> None:
        k = ["table_name", "dataset", "check_name"]
        m = self.b["integrity"].merge(self.c["integrity"], on=k, how="outer", suffixes=("_b", "_c"))
        m[["n_b", "n_c"]] = m[["n_b", "n_c"]].fillna(0)
        m = m[m["table_name"].isin(self.common_tables)]
        rows = m[m["check_name"] == "rows"].sort_values(["table_name", "dataset"])
        self.r.h("Row counts per table and dataset")
        n_diff = 0
        for _, x in rows.iterrows():
            same = x["n_b"] == x["n_c"]
            n_diff += not same
            self.r.add(f"  {'  ' if same else '! '}{x['table_name']:<22} {x['dataset']:<14} "
                       + (n(x['n_b']) if same else delta(x["n_b"], x["n_c"])))
            self.grade(self.count_level(x["n_b"], x["n_c"]))
        chk = m[m["check_name"] != "rows"]
        changed = chk[chk["n_b"] != chk["n_c"]].sort_values(k)
        self.r.h("Join integrity and consistency checks")
        self.r.add(f"  {len(chk)} check cells, {len(changed)} changed")
        self.r.rows([f"{x['table_name']:<22} {x['dataset']:<14} {x['check_name']:<36} {delta(x['n_b'], x['n_c'])}"
                     for _, x in changed.iterrows()], "    ")
        for _, x in changed.iterrows():
            self.grade(self.count_level(x["n_b"], x["n_c"]))
        bad = chk[(chk["n_c"] > 0) & chk["check_name"].str.contains("|".join(VIOLATIONS))
                  & ~chk["check_name"].str.startswith("concept_")]
        self.r.add(f"  violations present in the candidate ({len(bad)} cells; may be by design, "
                   "compare with the baseline column):")
        self.r.rows([f"{x['table_name']:<22} {x['dataset']:<14} {x['check_name']:<36} "
                     f"candidate {n(x['n_c'])}, baseline {n(x['n_b'])}" for _, x in bad.sort_values(k).iterrows()],
                    "    ")
        self.summary.append(("rows per dataset", f"{len(rows) - n_diff} of {len(rows)} table x dataset cells identical"))
        self.summary.append(("integrity", f"{len(chk) - len(changed)} of {len(chk)} check cells identical; "
                                          f"{len(bad)} nonzero violation cells in candidate"))

    # --- concept groups -------------------------------------------------------------------------
    def groups(self) -> None:
        gb, gc = key_str(self.b["groups"], KEYS), key_str(self.c["groups"], KEYS)
        gb, gc = gb[gb["table_name"].isin(self.common_tables)], gc[gc["table_name"].isin(self.common_tables)]
        counts = [c for c in gb.columns if c.startswith("n_") and c in gc.columns]
        stats = [c for c in gb.columns if not c.startswith("n_") and c not in KEYS and c in gc.columns]
        m = gb.merge(gc, on=KEYS, how="outer", suffixes=("_b", "_c"), indicator=True)
        only_b, only_c = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
        both = m[m["_merge"] == "both"].copy()

        lv_counts = np.zeros(len(both), dtype=int)
        for c in counts:
            a, b = both[f"{c}_b"].astype(float).fillna(0).to_numpy(), both[f"{c}_c"].astype(float).fillna(0).to_numpy()
            d = np.abs(b - a)
            lv = np.where(d == 0, 0, np.where(d <= self.a.count_tol * np.maximum(np.abs(a), 1.0), 1, 2))
            lv_counts = np.maximum(lv_counts, lv)
        lv_stats = np.zeros(len(both), dtype=int)
        for c in stats:
            a, b = both[f"{c}_b"].astype(float).to_numpy(), both[f"{c}_c"].astype(float).to_numpy()
            na, nb = np.isnan(a), np.isnan(b)
            d, scale = np.abs(b - a), np.maximum(np.abs(a), np.abs(b))
            with np.errstate(invalid="ignore"):
                lv = np.where(d <= EXACT_REL * scale, 0,
                              np.where(d <= np.maximum(self.a.stat_tol * scale, self.floor(c)), 1, 2))
            lv = np.where(na & nb, 0, np.where(na | nb, 2, lv))
            lv_stats = np.maximum(lv_stats, lv)
        both["level"] = np.maximum(lv_counts, lv_stats)
        both["count_level"], both["stat_level"] = lv_counts, lv_stats
        n_id, n_within, n_diff = (int((both["level"] == i).sum()) for i in range(3))
        self.grade(both["level"].max() if len(both) else 0)
        if len(only_b) or len(only_c):
            self.grade(2)

        self.r.h("Concept groups (dataset x concept_id x source value x unit)")
        self.r.add(f"  {len(both):,} matched: {n_id:,} identical, {n_within:,} within tolerance, "
                   f"{n_diff:,} different; {len(only_b):,} only in baseline "
                   f"({n(only_b['n_rows_b'].sum())} rows), {len(only_c):,} only in candidate "
                   f"({n(only_c['n_rows_c'].sum())} rows)")

        for t in self.common_tables:
            ob, oc = only_b[only_b["table_name"] == t], only_c[only_c["table_name"] == t]
            bt = both[both["table_name"] == t]
            dt = bt[bt["level"] > 0]
            if not (len(ob) or len(oc) or len(dt)):
                continue
            self.r.add("")
            self.r.add(f"  {t}: {len(bt):,} matched groups, {int((bt['level'] == 0).sum()):,} identical, "
                       f"{int((bt['level'] == 1).sum()):,} within tolerance, {int((bt['level'] == 2).sum()):,} different")
            for label, part, col in (("only in baseline", ob, "n_rows_b"), ("only in candidate", oc, "n_rows_c")):
                if len(part):
                    self.r.add(f"    {label}: {len(part):,} groups, {n(part[col].sum())} rows "
                               f"(per dataset: {', '.join(f'{d} {n(v)}' for d, v in part.groupby('dataset')[col].sum().items())})")
                    part = part.sort_values(col, ascending=False)
                    self.r.rows([f"{self.key_label(x)}  rows {n(x[col])}" for _, x in part.iterrows()], "      ")
            if len(dt):
                dt = dt.assign(_dn=(dt["n_rows_c"].astype(float) - dt["n_rows_b"].astype(float)).abs())
                dt = dt.sort_values(["level", "_dn", "n_rows_b"], ascending=False)
                self.r.add("    differing groups (largest row-count change first):")
                self.r.rows([self.group_line(x, counts, stats) for _, x in dt.iterrows()], "      ")
        self.concept_rollup(gb, gc)
        self.summary.append(("concept groups", f"{n_id:,} identical / {n_within:,} within tolerance / "
                                               f"{n_diff:,} different; {len(only_b):,} only in baseline, "
                                               f"{len(only_c):,} only in candidate"))
        self.diff_frames["groups"] = pd.concat([
            both.drop(columns=["_merge"]).assign(side="both"),
            only_b.drop(columns=["_merge"]).assign(side="baseline_only", level=2),
            only_c.drop(columns=["_merge"]).assign(side="candidate_only", level=2)])

    @staticmethod
    def key_label(x) -> str:
        unit = x["unit_source_value"] if x["unit_concept_id"] == NULL else f"{x['unit_source_value']}/{x['unit_concept_id']}"
        return (f"{x['dataset']:<6} concept {x['concept_id']:<10} src {short(x['source_value'])!r:<42} "
                f"unit {short(unit, 24)!r}")

    def group_line(self, x, counts, stats) -> str:
        parts = [f"rows {delta(x['n_rows_b'], x['n_rows_c'])}"]
        for c in counts:
            if c == "n_rows":
                continue
            a, b = x[f"{c}_b"], x[f"{c}_c"]
            if self.count_level(a, b):
                a0, b0 = (0 if pd.isna(a) else a), (0 if pd.isna(b) else b)
                parts.append(f"{c} {int(b0) - int(a0):+,} ({pct(rel(a0, b0))})")
        small = max(float(x["n_rows_b"] or 0), float(x["n_rows_c"] or 0)) < self.a.small
        shifted = []
        for c in stats:
            a, b = x[f"{c}_b"], x[f"{c}_c"]
            lv = self.stat_level(a, b, self.floor(c))
            if lv:
                if pd.isna(a) or pd.isna(b):
                    shifted.append(f"{c} {'appears' if pd.isna(a) else 'disappears'}")
                else:
                    shifted.append(f"{c} {pct((b - a) / max(abs(a), abs(b)))}" + ("" if lv == 2 else " (tol)"))
        if shifted:
            parts.append("small group, distribution differs" if small else "; ".join(shifted))
        return f"{self.key_label(x)}\n          " + " | ".join(parts)

    def concept_rollup(self, gb: pd.DataFrame, gc: pd.DataFrame) -> None:
        """Row totals per table x dataset x concept_id: separates a changed source or unit string
        (groups move, total unchanged) from rows that appear or vanish."""
        k = ["table_name", "dataset", "concept_id"]
        m = (gb.groupby(k)["n_rows"].sum().rename("b").to_frame()
             .join(gc.groupby(k)["n_rows"].sum().rename("c"), how="outer").fillna(0).reset_index())
        d = m[m["b"] != m["c"]].assign(_d=lambda f: (f["c"] - f["b"]).abs()).sort_values("_d", ascending=False)
        self.r.h("Row totals per table x dataset x concept_id")
        self.r.add(f"  {len(m) - len(d):,} of {len(m):,} identical")
        self.r.rows([f"{x['table_name']:<22} {x['dataset']:<6} concept {x['concept_id']:<10} {delta(x['b'], x['c'])}"
                     for _, x in d.iterrows()], "    ")

    # --- metrics --------------------------------------------------------------------------------
    def metrics(self) -> None:
        k = ["table_name", "dataset", "metric", "key"]
        mb, mc = key_str(self.b["metrics"], ["key"]), key_str(self.c["metrics"], ["key"])
        m = mb.merge(mc, on=k, how="outer", suffixes=("_b", "_c"))
        m = m[m["table_name"].isin(self.common_tables)]
        m["kind"] = m["kind_b"].fillna(m["kind_c"])
        lines, levels = [], []
        for _, x in m.sort_values(k).iterrows():
            if x["kind"] == "count":
                lv = self.count_level(x["value_b"], x["value_c"])
                txt = delta(x["value_b"], x["value_c"])
            else:
                lv = self.stat_level(x["value_b"], x["value_c"], self.floor(x["metric"]))
                a, b = x["value_b"], x["value_c"]
                txt = ("appears" if pd.isna(a) else "disappears") if (pd.isna(a) or pd.isna(b)) \
                    else f"shift {pct((b - a) / max(abs(a), abs(b), 1e-300))}"
            levels.append(lv)
            if lv:
                key = f"[{short(x['key'], 30)}]" if x["key"] not in ("", NULL) else ""
                lines.append(f"{x['table_name']:<22} {x['dataset']:<10} {x['metric']}{key}: {txt}"
                             + (" (tol)" if lv == 1 else ""))
        levels = np.array(levels, dtype=int)
        self.grade(int(levels.max()) if len(levels) else 0)
        self.r.h("Dataset-level metrics (person, visit, death, care site, observation period, categorical, sig)")
        self.r.add(f"  {len(m):,} metrics: {int((levels == 0).sum()):,} identical, {int((levels == 1).sum()):,} "
                   f"within tolerance, {int((levels == 2).sum()):,} different")
        self.r.rows(lines, "    ")
        self.summary.append(("metrics", f"{int((levels == 0).sum()):,} identical / {int((levels == 1).sum()):,} "
                                        f"within tolerance / {int((levels == 2).sum()):,} different"))
        self.diff_frames["metrics"] = m.assign(level=levels)

    # --- driver ---------------------------------------------------------------------------------
    def run(self) -> int:
        self.tables()
        self.schema()
        self.integrity()
        self.groups()
        self.metrics()
        verdict = ["IDENTICAL", "WITHIN TOLERANCE", "DIFFERENT"][self.worst]
        mb, mc = self.b["meta"], self.c["meta"]
        head = ["CDM fingerprint comparison", "==========================",
                f"baseline:  {mb['cdm_dir']}  (fingerprint {mb['created']})",
                f"candidate: {mc['cdm_dir']}  (fingerprint {mc['created']})",
                f"tolerance: counts rel {self.a.count_tol:g}, statistics rel {self.a.stat_tol:g} "
                f"(floor {self.a.time_tol_h:g} h for times, {self.a.value_abs_tol:g} for values), "
                f"small group < {self.a.small} rows"]
        if (mb.get("value_quantiles"), mb.get("time_quantiles")) != (mc.get("value_quantiles"), mc.get("time_quantiles")):
            head.append("warning: the fingerprints use different quantile sets")
        if mb.get("dataset_rule") != mc.get("dataset_rule"):
            head.append("warning: the fingerprints use different dataset rules")
        head += ["", f"VERDICT: {verdict}", ""]
        head += [f"  {k:<18} {v}" for k, v in self.summary]
        print("\n".join(head + self.r.lines))
        if self.a.csv_out:
            out = Path(self.a.csv_out)
            out.mkdir(parents=True, exist_ok=True)
            for name, df in self.diff_frames.items():
                df.to_csv(out / f"diff_{name}.csv", index=False)
            print(f"\nfull diff tables written to {out}")
        return 0 if self.worst < 2 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("candidate_dir", type=Path)
    ap.add_argument("--count-tol", type=float, default=0.0, help="relative tolerance for counts (default exact)")
    ap.add_argument("--stat-tol", type=float, default=0.01, help="relative tolerance for quantiles and means")
    ap.add_argument("--time-tol-h", type=float, default=1 / 60, help="absolute floor for hour-valued statistics")
    ap.add_argument("--value-abs-tol", type=float, default=1e-6, help="absolute floor for other statistics")
    ap.add_argument("--small", type=int, default=20, help="groups below this many rows print no statistics")
    ap.add_argument("--max-rows", type=int, default=25, help="lines per listing")
    ap.add_argument("--csv-out", help="also write the full merged diff tables here")
    args = ap.parse_args()
    sys.exit(Comparer(load(args.baseline_dir), load(args.candidate_dir), args).run())


if __name__ == "__main__":
    main()
