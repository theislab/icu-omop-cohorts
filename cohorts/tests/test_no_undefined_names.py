"""Static check: no function may load a name that is not a parameter, a local, or module-level.

Why this exists: the same bug shipped twice in one day, both times from an edit that referenced a
variable available in `main()` but not in the helper being edited:

    build_dyn_tensor(con, T)        used `task`  -> NameError  (N_BINS[task])
    build_dyn_tensor(con, T)        used `ds`    -> NameError  (unit_factor(var, ds))

The second one failed all 15 builds of the lock rebuild after they had already run their
prefilters. Python cannot catch this at import time because the name only resolves when the line
executes, and these helpers are deep enough into a job that a failure costs minutes to hours.

Run:  python tests/test_no_undefined_names.py
"""

from __future__ import annotations

import ast
import builtins
import os

TARGETS = ("build_cohort.py", "yaib_omop_crosswalk.py", "label_inputs.py")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def _bound_names(node: ast.AST) -> set[str]:
    """Every name a scope binds: assignments, imports, args, comprehension and for targets,
    with-as, except-as, lambdas, and nested def/class names."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ClassDef):
            out.add(n.name)
        # NB not `elif` for the def/lambda case: a FunctionDef binds BOTH its name and its
        # parameters, and chaining these skipped the parameters entirely -- which made the
        # checker report every function argument as undefined.
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
        if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            out |= {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out |= set(n.names)
    return out


def check_file(path: str) -> list[str]:
    tree = ast.parse(open(path).read())
    top = _bound_names(ast.Module(body=[b for b in tree.body
                                        if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))],
                                  type_ignores=[]))
    top |= {b.name for b in tree.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    problems = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        scope = _bound_names(fn) | top | BUILTINS
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in scope:
                problems.append(f"{os.path.relpath(path, ROOT)}:{n.lineno} "
                                f"{fn.name}() loads undefined '{n.id}'")
    return sorted(set(problems))


def test_no_undefined_names_in_builders():
    all_problems = []
    for t in TARGETS:
        p = os.path.join(ROOT, t)
        if os.path.exists(p):
            all_problems += check_file(p)
    assert not all_problems, "undefined names:\n  " + "\n  ".join(all_problems)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
