# Notebook source format spec

Every notebook is authored as a percent-format Python file in `src/NN_topic.py`
and converted with `python tools/percent2ipynb.py`.

## Cell markers

```python
# %% [markdown]
# # Title (h1 only for the very first cell)
# Body markdown. EVERY line must start with '# ' (hash + space).
# Lines may be empty ('#').

# %%
import numpy as np
# code cells are plain Python, no '# ' prefix needed
```

Rules:

1. First cell of every file MUST be a markdown title cell starting with `# # Notebook Title`.
2. Markdown lines are prefixed `# ` and the prefix is stripped on conversion.
3. Code cells must be runnable top-to-bottom in a fresh kernel (Python 3.10,
   numpy 2.x / pandas 2.x / matplotlib installed, NO internet).
4. Installed and working: numpy 2.2, pandas 2.3, matplotlib, seaborn 0.13,
   plotly 6.5, scipy 1.14, scikit-learn 1.7, statsmodels, sympy, requests,
   bs4, openpyxl, sqlite3 (stdlib). Import freely.
5. NO internet at execution time: any requests/bs4 example must be wrapped in
   `try/except Exception` with an offline fallback so the cell always ends
   without error.
6. Plotly: never let `fig.show()` open a browser during execution — set
   `import plotly.io as pio; pio.renderers.default = "notebook"` first, or end
   the cell with `fig` / `display(fig)`. Always verify by executing.
7. matplotlib: put `%matplotlib inline` as the first line of the first code
   cell in any plotting notebook.

## Structure per notebook

```text
# # <NN> · <Notebook Title>
   (short intro paragraph + how it maps to the playlist)

# ## ▶ Sessions covered        <- markdown table:
   | # | Session | Video |
   |---|---------|-------|
   | 13 | Session 13 - Numpy Fundamentals | https://www.youtube.com/watch?v=XF6DCrNTzug |

# ## 1. <Section>
   ...content...
```

## Content style (full function reference)

For every concept/function:

```markdown
### `function_name()`
**What it does:** one line.
**Syntax:** `function_name(arg1, arg2, ...)`
**Parameters:** bullet list (name — type — meaning — default).
**Returns:** what it returns.
**Example:**
```

Then a code cell with the example + printed output, and where useful a
`# NOTE:` / `# TIP:` / `# WARNING:` inline comment.

Section-level notes:

- End each major section with a markdown cell `**Section notes:**` containing
  a compact bullet/pipe summary of the functions covered.
- Use "Gotchas" callouts (`> **Gotcha:** ...`) for common interview traps.
- Keep examples small and deterministic (fixed seeds for randomness).

## Reference video links

Every section that comes from a session must carry its video link in markdown:

`▶ Session 13 – Numpy Fundamentals: https://www.youtube.com/watch?v=XF6DCrNTzug`

Only YouTube links. Full ordered video list: `tools/playlist.json`
(`{"index": n, "id": "...", "title": "..."}`).

## Naming

`src/NN_short_name.py` -> `NN_short_name.ipynb` at repo root.
Numbering follows playlist order (see README).
