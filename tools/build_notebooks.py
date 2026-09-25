"""
Builds and executes the four project notebooks, storing outputs (tables + charts)
inside the .ipynb files so they render directly on GitHub.

You normally don't need this - just open the notebooks in Jupyter and "Run All".
It exists so the repository ships with already-executed notebooks.

Run from the repo root:  python tools/build_notebooks.py
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
sys.path.insert(0, str(ROOT / "tools"))
from notebook_content import NOTEBOOKS  # noqa: E402


def _src(text: str) -> list[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def run_cell(code: str, ns: dict) -> list[dict]:
    outputs = []
    buf = io.StringIO()
    tree = ast.parse(code)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)
        result = eval(compile(last_expr, "<cell>", "eval"), ns) if last_expr else None
    if buf.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": _src(buf.getvalue())})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        png = io.BytesIO()
        fig.savefig(png, format="png", bbox_inches="tight", dpi=100)
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                 "text/plain": ["<Figure>"]}})
    plt.close("all")
    if result is not None:
        data = {"text/plain": _src(repr(result))}
        if isinstance(result, (pd.DataFrame, pd.Series)):
            frame = result.to_frame() if isinstance(result, pd.Series) else result
            data["text/html"] = _src(frame.to_html(max_rows=25))
        elif hasattr(result, "_repr_html_"):
            data["text/html"] = _src(result._repr_html_())
        outputs.append({"output_type": "execute_result", "execution_count": None,
                        "metadata": {}, "data": data})
    return outputs


def build(name: str, cells: list[tuple[str, str]]) -> None:
    os.chdir(NB_DIR)
    ns: dict = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, text in cells:
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text)})
            continue
        count += 1
        outs = run_cell(text.strip("\n"), ns)
        for o in outs:
            if o["output_type"] == "execute_result":
                o["execution_count"] = count
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                         "outputs": outs, "source": _src(text)})
    nb = {"cells": nb_cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"{name[:2]}-{i:03d}"
    (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"built {name}: {count} code cells")


if __name__ == "__main__":
    for nb_name, nb_cells in NOTEBOOKS.items():
        build(nb_name, nb_cells)
