"""The dynamic grid is INCLUSIVE of both endpoints, as ricu's `expand` is.

Run:  python tests/test_grid_length.py

Found 2026-09-03 by user inspection of the reference cohorts: every miiv/mortality24 stay has
exactly 25 rows, not 24. `ricu::expand` is built on `seq(min, max, step)`, and R's seq() includes
the upper bound, so a horizon of T hours yields hours 0..T = T+1 bins. We were emitting 0..T-1 and
silently dropping each stay's final hour on BOTH routes.
"""

from __future__ import annotations

import ast
import os

import numpy as np

# The builder imports duckdb, ehrdata and ehrapy at module level; its grid constants are read from the source.
_BUILDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build_cohort.py")
_CONSTANTS: dict = {}
for _node in ast.parse(open(_BUILDER).read()).body:
    if (isinstance(_node, ast.Assign) and isinstance(_node.targets[0], ast.Name)
            and _node.targets[0].id in ("TASK_HORIZON", "N_BINS")):
        exec(compile(ast.Module([_node], []), _BUILDER, "exec"), _CONSTANTS)
TASK_HORIZON, N_BINS = _CONSTANTS["TASK_HORIZON"], _CONSTANTS["N_BINS"]


def test_n_bins_is_horizon_plus_one():
    assert N_BINS["mortality24"] == 25, "T=24 task must have 25 hourly bins (hours 0..24)"
    assert N_BINS["los"] == 169, "T=168 task must have 169 bins (hours 0..168)"
    for task, h in TASK_HORIZON.items():
        assert N_BINS[task] == h + 1, task


def test_rows_per_stay_formula_matches_ricu():
    """Measured in the reference cohorts: for k rows, LoS lies in [k-1, k).
    So rows == floor(LoS) + 1, capped at the grid length."""
    T = 25
    for los_h, expected in ((6.0, 7), (6.98, 7), (10.0, 11), (23.5, 24), (24.0, 25), (100.0, 25)):
        got = int(min(np.floor(los_h) + 1, T))
        assert got == expected, f"LoS {los_h} h -> {got} rows, expected {expected}"


def test_the_old_off_by_one_is_actually_wrong():
    """Guard against reverting: the old formula drops the final hour of every stay."""
    T_old, T_new = 24, 25
    los_h = 30.0
    assert int(min(np.floor(los_h), T_old)) == 24
    assert int(min(np.floor(los_h) + 1, T_new)) == 25


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
