#!/usr/bin/env python3
"""Generate README.md — an index mapping every notebook to its playlist sessions.

Session->notebook assignment lives in NOTEBOOK_SESSIONS below; titles/links are
pulled from tools/playlist.json (the authoritative ordered playlist dump).
Run after editing:  python3 tools/make_readme.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAYLIST = json.loads((ROOT / "tools" / "playlist.json").read_text())
BY_INDEX = {v["index"]: v for v in PLAYLIST}
BASE = "https://www.youtube.com/watch?v="

NOTEBOOK_SESSIONS = {
    "01_python_fundamentals": {
        "title": "01 · Python Fundamentals",
        "blurb": "Output, data types, variables, operators, control flow, loops, "
                 "strings, lists, tuples, sets, dictionaries, functions, time complexity.",
        "sessions": [1, 2, 3, 4, 7, 9, 10, 11],
    },
    "02_oop_advanced_python": {
        "title": "02 · OOP & Advanced Python",
        "blurb": "Classes/objects, encapsulation, inheritance, polymorphism, abstraction, "
                 "file handling & serialization, exceptions, decorators, recursion, "
                 "iterators, generators.",
        "sessions": [17, 19, 21, 22, 26, 27, 29, 31, 33, 34, 35],
    },
    "03_numpy": {
        "title": "03 · NumPy",
        "blurb": "Array creation, attributes, indexing/slicing, broadcasting, ufuncs, "
                 "aggregations & axis, sorting/searching, stacking, random, linear algebra, "
                 "copy vs view, tricks.",
        "sessions": [41, 43, 45, 55],
    },
    "04_pandas": {
        "title": "04 · Pandas",
        "blurb": "Series & DataFrame reference, indexing (loc/iloc), all core methods, "
                 "groupby, merge/join/concat, MultiIndex, .str/.dt accessors, pivot_table, "
                 "case-study pipeline.",
        "sessions": [50, 51, 52, 53, 56, 60, 61, 70, 63, 71, 72],
    },
    "05_visualization": {
        "title": "05 · Data Visualization",
        "blurb": "Matplotlib basics + advanced (gridspec, formatters, annotations), "
                 "seaborn statistical plots, Plotly Express & graph_objects, chart-choice guide.",
        "sessions": [73, 75, 76, 77, 78, 80],
    },
    "06_data_analysis_process": {
        "title": "06 · Data Analysis Process",
        "blurb": "Data gathering (files/APIs/scraping with offline fallbacks), assessing, "
                 "cleaning (missing/outliers/types/strings), feature basics, EDA workflow "
                 "and an end-to-end case study.",
        "sessions": [82, 84, 88, 91, 95, 97],
    },
    "07_sql": {
        "title": "07 · SQL for Data Science",
        "blurb": "Runnable via sqlite3: DDL/DML, SELECT, grouping, joins, subqueries/CTEs, "
                 "window functions, views/indexes, transactions, normalization, cleaning & EDA "
                 "in SQL, interview queries.",
        "sessions": [96, 99, 101, 102, 104, 107, 108, 111, 112, 115, 116, 117, 118,
                     124, 127, 128, 132, 133],
    },
    "08_statistics_probability": {
        "title": "08 · Statistics & Probability",
        "blurb": "Descriptive statistics, correlation, probability rules & Bayes, "
                 "PMF/PDF/CDF, discrete & continuous distributions, CLT, confidence "
                 "intervals, hypothesis tests (z/t/χ²/ANOVA), test-selection cheat sheet.",
        "sessions": [123, 125, 126, 129, 131, 134, 135, 137, 138, 139, 140, 189, 190],
    },
    "09_linear_algebra": {
        "title": "09 · Linear Algebra",
        "blurb": "Vectors, dot products & norms, matrices, determinants/inverse, systems of "
                 "equations, eigen-decomposition, SVD, projections — with the PCA/ML links.",
        "sessions": [141, 142, 143, 182, 183, 184],
    },
    "10_ml_fundamentals_regression": {
        "title": "10 · ML Fundamentals & Regression",
        "blurb": "ML workflow & splits, simple/multiple linear regression, gradient descent "
                 "(batch/SGD/mini-batch from scratch), statsmodels OLS, LINE assumptions, "
                 "polynomial regression, multicollinearity/VIF, feature selection, "
                 "bias-variance, Ridge/Lasso/ElasticNet.",
        "sessions": [144, 145, 146, 148, 149, 150, 151, 152, 153, 155, 157, 158, 159,
                     160, 161, 162, 163, 164, 165, 167, 169, 170, 171, 172, 173, 174, 175],
    },
    "11_ml_classification": {
        "title": "11 · Classification Algorithms",
        "blurb": "Confusion matrix & metrics, ROC/PR curves, KNN (from scratch), Naive Bayes "
                 "variants, MLE, logistic & softmax regression, SVM hard/soft margin & kernels, "
                 "model comparison.",
        "sessions": [166, 168, 176, 177, 185, 192, 191, 193, 194, 197, 196, 195, 198, 200],
    },
    "12_ml_dimensionality_model_selection": {
        "title": "12 · Dimensionality Reduction & Model Selection",
        "blurb": "Curse of dimensionality, PCA (math + from-scratch + sklearn), cross-validation "
                 "types, data-leakage catalog, hyperparameter tuning (Grid/Random search), "
                 "feature engineering (imputation, scaling, encoding) and a leakage-safe pipeline.",
        "sessions": [178, 179, 180, 181, 186, 187, 188, 199],
    },
}


def session_rows(indices):
    lines = ["| # | Session | Video |", "|---|---------|-------|"]
    for i in indices:
        v = BY_INDEX[i]
        title = v["title"].replace("|", "\\|")
        lines.append(f"| {i} | {title} | {BASE}{v['id']} |")
    return "\n".join(lines)


def main():
    assigned = [i for s in NOTEBOOK_SESSIONS.values() for i in s["sessions"]]
    total_cells = 0
    nb_rows = []
    for stem, info in NOTEBOOK_SESSIONS.items():
        ipynb = ROOT / f"{stem}.ipynb"
        n_cells = "—"
        if ipynb.exists():
            nb = json.loads(ipynb.read_text())
            n_cells = str(len(nb["cells"]))
            total_cells += len(nb["cells"])
        nb_rows.append(f"| `{stem}.ipynb` | {info['title']} | "
                       f"{len(info['sessions'])} | {n_cells} |")

    out = []
    out.append("# DSMP 2022-23 — Data Science Notes (Jupyter Notebooks)\n")
    out.append(
        "Structured, code-first notes built from the **CampusX Data Science Mentorship "
        "Program (DSMP) 2022-23** playlist — every concept with syntax, parameters, "
        "runnable examples, section notes, gotchas and a link back to the exact session video.\n\n"
        "**Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z\n"
    )
    out.append("## Notebook index\n")
    out.append("| Notebook | Topic | Sessions | Cells |\n|---|---|---|---|")
    out.extend(nb_rows)
    out.append(f"\n**Total: {len(nb_rows)} notebooks, {total_cells} cells** "
               "(every code cell executes top-to-bottom with 0 errors).\n")

    out.append("## How to use\n\n"
               "1. Open any `.ipynb` in Jupyter / VS Code and run all — the outputs are already "
               "saved, so you can also just read them.\n"
               "2. Each notebook starts with a **▶ Sessions covered** table linking to its "
               "source videos, and every section repeats its own video link.\n"
               "3. Read in order 01 → 12 (playlist order), or jump to the topic you need.\n")

    out.append("## Repo layout\n\n"
               "```\n"
               "dsmp-notes/\n"
               "├── NN_*.ipynb      # the 12 finished notebooks (generated + executed)\n"
               "├── src/            # notebook sources (percent-format .py — edit these)\n"
               "└── tools/\n"
               "    ├── percent2ipynb.py  # src/*.py -> .ipynb converter\n"
               "    ├── make_readme.py    # regenerates this file\n"
               "    ├── playlist.json     # full ordered playlist dump (200 videos)\n"
               "    └── FORMAT.md         # authoring style guide\n"
               "```\n\n"
               "Rebuild after editing a source:\n\n"
               "```bash\n"
               "python3 tools/percent2ipynb.py                 # convert all src/*.py\n"
               "jupyter-nbconvert --to notebook --execute --inplace NN_topic.ipynb\n"
               "python3 tools/make_readme.py                   # refresh this index\n"
               "```\n")

    out.append("## Sessions covered (by notebook)\n")
    for stem, info in NOTEBOOK_SESSIONS.items():
        out.append(f"\n### {stem}.ipynb\n\n**{info['title']}**\n")
        out.append(f"{info['blurb']}\n")
        out.append(session_rows(info["sessions"]))

    others = [v for v in PLAYLIST if v["index"] not in assigned]
    out.append(f"\n## Not covered ({len(others)} videos)\n")
    out.append("Paid-member task solutions, doubt-clearance, career/soft-skill sessions, "
               "schedule announcements and tool sessions (Flask, Streamlit, Git, Tableau, "
               "Selenium, AWS, open source) — full ordered list with IDs: "
               f"`tools/playlist.json` ({len(PLAYLIST)} videos).\n")
    out.append(f"**Complete video index (all {len(PLAYLIST)}):**\n\n"
               "| # | Session | Video |\n|---|---------|-------|")
    for v in PLAYLIST:
        out.append(f"| {v['index']} | {v['title'].replace('|', chr(92) + '|')} | "
                   f"{BASE}{v['id']} |")
    out.append("")

    (ROOT / "README.md").write_text("\n".join(out) + "\n")
    print(f"README.md written: {len(out)} blocks, {len(PLAYLIST)} videos indexed")


if __name__ == "__main__":
    main()
