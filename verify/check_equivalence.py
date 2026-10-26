#!/usr/bin/env python3
"""Check that two Python files differ only in comments and docstrings.

Usage: python verify/check_equivalence.py <scored_file> <public_file>

Both files are parsed, docstrings are removed from the module, every class and
every function, and the remaining syntax trees are compared. Comments are not
part of the syntax tree, so they never affect the result. Prints EQUIVALENT if
the trees match and DIFFERENT otherwise (exit status 0 or 1).
"""
import ast
import sys


def norm(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        if (
            isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)
            and isinstance(n.body[0].value.value, str)
        ):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(t, include_attributes=False)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        scored = f.read()
    with open(argv[2], encoding="utf-8") as f:
        public = f.read()
    same = norm(scored) == norm(public)
    print("EQUIVALENT" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
