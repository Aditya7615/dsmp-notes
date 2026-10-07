#!/usr/bin/env python3
"""Convert percent-format .py sources (src/*.py) into Jupyter .ipynb notebooks.

Cell markers (jupytext "percent" style):
    # %% [markdown]      -> markdown cell (each line must start with '# ' or '#')
    # %%                 -> code cell
    # %% tags=["hide"]   -> code cell with tags (ignored extras are fine)

Usage:
    python tools/percent2ipynb.py            # convert every src/*.py
    python tools/percent2ipynb.py src/03_numpy.py
"""
import json
import re
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT

MARKER = re.compile(r"^# %%(?: (?P<meta>\[markdown\].*|tags=.*))?\s*$")


def split_cells(text: str):
    cells = []
    current = None  # (kind, [lines])
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            if current:
                cells.append(current)
            meta = (m.group("meta") or "").strip()
            kind = "markdown" if meta.startswith("[markdown]") else "code"
            current = [kind, []]
        else:
            if current is None:
                current = ["markdown", []]  # preamble before first marker = markdown
            current[1].append(line)
    if current:
        cells.append(current)
    return cells


def clean_md(lines):
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.startswith("#"):
            out.append(ln[1:])
        else:
            out.append(ln)
    # strip leading/trailing blank lines
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def clean_code(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines) + ("\n" if lines else "")


def build(path: Path) -> dict:
    cells_out = []
    for n, (kind, lines) in enumerate(split_cells(path.read_text())):
        if kind == "markdown":
            src = clean_md(lines)
            if not src.strip():
                continue
            cid = hashlib.md5(f"{n}:{src}".encode()).hexdigest()[:8]
            cells_out.append({
                "cell_type": "markdown",
                "id": cid,
                "metadata": {},
                "source": src.splitlines(keepends=True),
            })
        else:
            src = clean_code(lines)
            if not src.strip():
                continue
            cid = hashlib.md5(f"{n}:{src}".encode()).hexdigest()[:8]
            cells_out.append({
                "cell_type": "code",
                "id": cid,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            })
    return {
        "cells": cells_out,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv):
    files = [Path(a) for a in argv] or sorted(SRC.glob("*.py"))
    if not files:
        print("no sources found in", SRC)
        return 1
    for f in files:
        nb = build(f)
        dest = OUT / (f.stem + ".ipynb")
        dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"{f.name} -> {dest.name}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
