# %% [markdown]
# # 05 · Data Visualization — Matplotlib, Seaborn & Plotly
#
# "A picture is worth a thousand rows" — EDA without charts is guessing. This
# notebook is the function reference for the three plotting libraries used
# throughout DSMP: **Matplotlib** (the foundation), **Seaborn** (statistical
# defaults on top of Matplotlib) and **Plotly** (interactive, browser-based).
#
# | Library | Output | Best for | Key idea |
# |---|---|---|---|
# | Matplotlib | static PNG/SVG | total control, reports, papers | Figure/Axes **artist** objects |
# | Seaborn | static (Matplotlib) | statistical summaries, pretty defaults | one-line plots from *dataframes* |
# | Plotly | interactive HTML | dashboards, exploration, sharing | declarative figures (`plotly.express`) |
#
# > **Note:** all examples run offline in a fresh kernel with fixed random
# > seeds; figures are written only to `/tmp` when saved.

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 73 | Session 23 - Plotting using Matplotlib | https://www.youtube.com/watch?v=XaKn_cKFlSY |
# | 75 | Session 24 - Advanced Matplotlib | https://www.youtube.com/watch?v=7YDc5xU9CQQ |
# | 76 | Session on Plotly Express | https://www.youtube.com/watch?v=VPqw8soG4ZY |
# | 77 | Session on Plotly Part 2 | https://www.youtube.com/watch?v=_ZsVLh6v29A |
# | 78 | Session 25 - Plotting using Seaborn | https://www.youtube.com/watch?v=DWVLRhnuGqI |
# | 80 | Session 26 - Plotting using Seaborn Part 2 | https://www.youtube.com/watch?v=kLWeKeqc9Ms |

# %%
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# rule: never let fig.show() open a browser during execution
pio.renderers.default = "notebook"

print("matplotlib", matplotlib.__version__,
      "| seaborn", sns.__version__,
      "| plotly", plotly.__version__)
print("style in use:", matplotlib.rcParams["figure.figsize"])

# %% [markdown]
# ---
# ## 1. Matplotlib anatomy — Figure, Axes, Axis & the two APIs
#
# ▶ Session 23 - Plotting using Matplotlib: https://www.youtube.com/watch?v=XaKn_cKFlSY

# %% [markdown]
# ### Artist hierarchy
# **What it does:** *everything* in Matplotlib is an **Artist** — you draw by
# configuring objects, not by painting pixels.
#
# ```text
# Figure  (the canvas you save / show)
#  ├── Axes  (ONE plot area: title, limits, grid, legend lives here)
#  │    ├── Axis-x  (ticks, tick labels, margins, "day" units)
#  │    └── Axis-y  (ticks, tick labels, margins)
#  │         └── Artists: Line2D, Rectangle (bars), Polygon (fills),
#  │             Text (labels), Legend, Colorbar, ...
#  └── other Artists: Figure-level texts, suptitle, ...
# ```
#
# ### Two APIs, same engine
#
# | API | Style | Use when |
# |---|---|---|
# | `pyplot` (`plt.plot`) | stateful, MATLAB-like, implicit "current axes" | quick, single-plot scripts |
# | object-oriented (`fig, ax = plt.subplots()`) | explicit handles | anything multi-panel, reusable, or programmatic |

# %%
# pyplot API — state machine: plt targets the "current" figure/axes
plt.figure(figsize=(5, 3), dpi=100)
plt.plot([1, 2, 3], [1, 4, 9], label="y = x^2")
plt.title("pyplot style")
plt.xlabel("x")
plt.legend()
plt.show()

# OO API — explicit handles; scales to dashboards and functions (RECOMMENDED)
fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
ax.plot([1, 2, 3], [1, 4, 9], label="y = x^2")
ax.set_title("object-oriented style")
ax.set_xlabel("x")
ax.legend()
plt.show()

# %% [markdown]
# ### `plt.figure(figsize=None, dpi=None)` and `plt.subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False, ...)`
# **What they do:** create the canvas (`figure`) and the grid of Axes (`subplots` —
# the most-used OO constructor).
# **Parameters:**
# - `figsize` — `(width, height)` in **inches** (the unit is inches, not pixels)
# - `dpi` — dots per inch; final pixel size = `figsize * dpi`
# - `sharex` / `sharey` — link axes across the subplot grid (good for aligned time series)
# **Returns:** `Figure`; `subplots` → `(fig, ax)` or `(fig, axes_array)`.

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
print(type(fig).__name__, "| axes:", axes.shape, "| one Axes ->", type(axes[0, 0]).__name__)

x = np.arange(5)
rng = np.random.default_rng(0)
y = rng.integers(1, 10, size=5)

axes[0, 0].plot(x, y, marker="o")
axes[0, 0].set_title("(a) line")
axes[0, 1].scatter(x, y, s=80, c=y, cmap="viridis")
axes[0, 1].set_title("(b) scatter")
axes[1, 0].bar(x, y, color="#4C72B0")
axes[1, 0].set_title("(c) bar")
axes[1, 1].hist(rng.normal(size=300), bins=20, edgecolor="black")
axes[1, 1].set_title("(d) hist")

fig.suptitle("one figure, four Axes", fontsize=13)
fig.tight_layout()          # no overlapping labels
plt.show()

# %% [markdown]
# ### `plt.plot(x, y, color=, linestyle=, marker=, linewidth=, label=)`
# **What it does:** draws a line chart (also `ax.plot`).
# **Parameters:**
# - `color` / `c` — name (`"crimson"`), hex (`"#1f77b4"`), rgb tuple `(0.1, 0.4, 0.7)`, single letters (`bgrcm`)
# - `linestyle` / `ls` — `"-"`, `"--"`, `"-."`, `":"`, `"none"`
# - `marker` — `"o"`, `"s"`, `"^"`, `"D"`, `"v"`, `">"`, `"*"`, `"P"`, `"x"`, `"+"`, `"|"`
# - `linewidth` / `lw` — line thickness (points); `markersize` / `ms` — marker size
# - `markevery` — draw markers only every *k*-th point (clean dense lines)
# - `label` — legend text (shown by `ax.legend()`)
# **Returns:** list of `Line2D` artists.

# %%
x = np.linspace(0, 2 * np.pi, 50)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), color="#1f77b4", linestyle="-", linewidth=2,
        marker="o", markevery=5, label="sin(x)")
ax.plot(x, np.cos(x), color="darkorange", ls="--", lw=1.5,
        marker="s", markevery=5, label="cos(x)")
ax.plot(x, np.sin(x / 2), color=(0.2, 0.6, 0.4), ls=":", lw=3,
        marker="^", ms=8, label="sin(x/2)")
ax.legend()
ax.set_title("line() parameter reference")
plt.show()

# %%
# categorical line chart — just give x as strings
months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales = [120, 135, 148, 160, 155]
plt.figure(figsize=(6, 3.5))
plt.plot(months, sales, marker="o", lw=2, color="#55A868")
plt.ylabel("sales (k$)")
plt.title("monthly sales")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Section notes — anatomy:**
#
# | Concept | Code |
# |---|---|
# | canvas | `fig = plt.figure(figsize=(w, h), dpi=100)` |
# | one panel | `fig, ax = plt.subplots()` |
# | panel grid | `fig, axes = plt.subplots(2, 2)` → `axes[0, 1]` |
# | quick style | `plt.plot(...)` (state machine) |
# | robust style | `ax.plot(...)` (object oriented) |
# | line kwargs | `color, ls, lw, marker, ms, markevery, label` |

# %% [markdown]
# ---
# ## 2. Basic charts — full parameter reference
#
# ▶ Session 23 - Plotting using Matplotlib: https://www.youtube.com/watch?v=XaKn_cKFlSY

# %% [markdown]
# ### `plt.scatter(x, y, s=None, c=None, cmap=None, marker="o", alpha=1, edgecolors="face")`
# **What it does:** point cloud for **relationships**; size and colour become extra dimensions.
# **Parameters:**
# - `s` — scalar or array — marker **area** in points² (not diameter!)
# - `c` — colour array mapped through `cmap` (or plain colour when numeric array not intended)
# - `cmap` — colormap name (`"viridis"`, `"coolwarm"`, `"plasma"`)
# - `alpha` — transparency (0–1) — the overplotting antidote
# **Returns:** `PathCollection`.

# %%
rng = np.random.default_rng(42)
n = 150
gx = rng.normal(size=n)
gy = gx * 0.8 + rng.normal(size=n) * 0.6
sizes = rng.uniform(30, 200, n)

sc = plt.scatter(gx, gy, s=sizes, c=gy, cmap="viridis", alpha=0.75, marker="o")
plt.colorbar(sc, label="y value")
plt.xlabel("x"); plt.ylabel("y"); plt.title("scatter: size + colour encode extra dims")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### `plt.bar(x, height, width=0.8, bottom=None, color=, edgecolor=)` · `plt.barh(y, width)`
# **What it does:** categorical **comparison** — vertical (`bar`) or horizontal (`barh`) bars.
# **Parameters:** `width` — bar thickness; `bottom`/`left` — stacked segment start; `color` / `edgecolor`; `align` — `"center"` (default) vs `"edge"`.
# **Returns:** list of `Rectangle` patches.

# %%
cats = ["Math", "Phy", "Chem", "CS"]
a = [85, 72, 90, 78]
b = [65, 88, 70, 92]
xpos = np.arange(len(cats))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))

# grouped
axs[0].bar(xpos - 0.2, a, width=0.4, label="2021", color="#4C72B0")
axs[0].bar(xpos + 0.2, b, width=0.4, label="2022", color="#DD8452")
axs[0].set_xticks(xpos, labels=cats)
axs[0].set_title("grouped"); axs[0].legend()

# stacked
axs[1].bar(xpos, a, label="2021", color="#4C72B0")
axs[1].bar(xpos, b, bottom=a, label="2022", color="#DD8452")
axs[1].set_xticks(xpos, labels=cats)
axs[1].set_title("stacked"); axs[1].legend()

# horizontal
axs[2].barh(cats, a, color="#55A868", edgecolor="black")
axs[2].set_title("barh")

fig.tight_layout(); plt.show()

# %% [markdown]
# ### `plt.hist(x, bins=10, density=False, cumulative=False, histtype="bar", edgecolor=None, alpha=1)`
# **What it does:** distribution of values via bins.
# **Parameters:**
# - `bins` — int (bin count), sequence (edges) or `"auto"`
# - `density` — normalise to a probability density (area = 1)
# - `cumulative` — running count instead of per-bin count
# - `histtype` — `"bar"`, `"barstacked"`, `"step"`, `"stepfilled"`
# - `edgecolor` / `alpha` — outline & transparency (stack several samples)
# **Returns:** `(n, bins, patches)`.

# %%
data = rng.normal(loc=170, scale=8, size=1000)   # heights-ish

fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
axs[0].hist(data, bins=25, edgecolor="black", color="#4C72B0")
axs[0].set_title("counts, bins=25")

axs[1].hist(data, bins=25, density=True, alpha=0.6, color="#C44E52",
            label="density")
axs[1].axvline(data.mean(), color="black", ls="--", label="mean")
axs[1].legend(); axs[1].set_title("density=True + mean line")

axs[2].hist(data, bins=25, cumulative=True, color="#55A868", histtype="stepfilled")
axs[2].set_title("cumulative=True")

fig.tight_layout(); plt.show()

# %% [markdown]
# ### `plt.pie(sizes, labels=, autopct=, explode=, startangle=, shadow=, counterclock=)`
# **What it does:** part-of-whole composition.
# **Parameters:**
# - `autopct` — format string for percentages (`"%1.1f%%"`)
# - `explode` — tuple, pull slices out (`(0, 0, 0.1, 0)`)
# - `startangle` — rotate the whole pie (degrees)
# - `shadow`, `counterclock`, `colors`, `wedgeprops`
# **Returns:** `(wedges, texts, autotexts)`.
# > **Gotcha:** pies only read well with **≤ 5 slices** — otherwise use a bar chart.

# %%
sizes = [45, 30, 15, 10]
labels = ["Product A", "Product B", "Product C", "Other"]

wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct="%1.1f%%", explode=(0, 0, 0.08, 0),
    startangle=90, shadow=True, colors=["#4C72B0", "#55A868", "#DD8452", "#C44E52"],
    counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=1.5))
plt.title("revenue split")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### `plt.stackplot(x, ys, labels=)` · `plt.step(x, y, where="pre")` · `plt.errorbar(x, y, yerr=, xerr=, fmt=, capsize=)`
# **What they do:**
# - `stackplot` — stacked **area** chart (cumulative totals over a continuum)
# - `step` — stair-step line (distributions, discrete changes) with `where="pre"/"post"/"mid"`
# - `errorbar` — values with **uncertainty bars** (`yerr`/`xerr`, `fmt` sets marker+line style)

# %%
t = np.arange(1, 6)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))

axs[0].stackplot(t, [3, 4, 5, 6, 7], [2, 3, 3, 4, 4],
                 labels=["web", "mobile"], colors=["#4C72B0", "#DD8452"])
axs[0].legend(loc="upper left"); axs[0].set_title("stackplot (area)")
axs[0].set_xlabel("month")

axs[1].step(t, [1, 3, 2, 4, 3], where="mid", color="#55A868", lw=2)
axs[1].set_title("step(where='mid')")

means = [10, 12, 11]
yerr = [0.8, 1.2, 0.6]
axs[2].errorbar(range(3), means, yerr=yerr, fmt="o", capsize=6,
                ecolor="red", elinewidth=2, color="black")
axs[2].set_title("errorbar(yerr, capsize)")
axs[2].set_xticks(range(3), labels=["A", "B", "C"])

fig.tight_layout(); plt.show()

# %% [markdown]
# ### `plt.boxplot(x, tick_labels=, showfliers=True, showmeans=False, patch_artist=False)` · `plt.violinplot(dataset, showmeans, showmedians, showextrema)`
# **What they do:** distribution shapes — box (median + IQR + whiskers + outliers)
# vs violin (smoothed density on both sides).
# **Box anatomy:** box = Q1..Q3, line = median, whiskers = 1.5·IQR, dots = fliers.
# **Violin params:** `showmeans`/`showmedians`/`showextrema` add markers inside.
# **Returns:** dict of artists (boxplot) / dict of `PolyCollection`s (violinplot).

# %%
sample = [rng.normal(0, 1, 200), rng.normal(1.2, 1.5, 200), rng.normal(-0.5, 0.7, 200)]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))

bp = axs[0].boxplot(sample, tick_labels=["A", "B", "C"],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red"))
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0")
axs[0].set_title("boxplot (patch_artist, showmeans)")

vp = axs[1].violinplot(sample, showmeans=True, showmedians=True, showextrema=True)
axs[1].set_xticks([1, 2, 3], labels=["A", "B", "C"])
axs[1].set_title("violinplot (density shape)")

fig.tight_layout(); plt.show()

# %% [markdown]
# ### `plt.hexbin(x, y, gridsize=10, cmap=, mincnt=1)` · `plt.imshow(array, cmap=, aspect=, interpolation=)`
# **What they do:** 2-D density / matrix visualisation without overplotting.
# - `hexbin` — hexagonal bins for huge scatter datasets; `mincnt=1` hides empty cells
# - `imshow` — render a 2-D array as an image (heatmaps, correlation matrices, images)
# **Returns:** `QuadMesh`; pair with `plt.colorbar(mappable)`.

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))

hx = rng.normal(size=2000); hy = rng.normal(size=2000)
mesh = axs[0].hexbin(hx, hy, gridsize=18, cmap="viridis", mincnt=1)
fig.colorbar(mesh, ax=axs[0], label="points")
axs[0].set_title("hexbin (2-D density)")

im = axs[1].imshow(rng.normal(size=(8, 12)), cmap="magma", aspect="auto",
                   interpolation="nearest")
fig.colorbar(im, ax=axs[1], shrink=0.8)
axs[1].set_title("imshow (matrix as image)")

# correlation matrix rendered with imshow
corr = pd.DataFrame(rng.normal(size=(5, 5)),
                    columns=list("ABCDE")).corr()
im2 = axs[2].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axs[2].set_xticks(range(5), labels=corr.columns, rotation=45)
axs[2].set_yticks(range(5), labels=corr.columns)
fig.colorbar(im2, ax=axs[2], shrink=0.8)
axs[2].set_title("corr matrix (imshow)")

fig.tight_layout(); plt.show()

# %% [markdown]
# **Section notes — chart menu:**
#
# | Question | Chart | Call |
# |---|---|---|
# | trend over time | line | `plt.plot` |
# | relationship | scatter | `plt.scatter(s=, c=, cmap=, alpha=)` |
# | category comparison | bar / barh | `plt.bar`, `plt.barh` |
# | distribution | hist / box / violin | `plt.hist`, `plt.boxplot`, `plt.violinplot` |
# | share of total | pie (≤5 slices) | `plt.pie(autopct=, explode=)` |
# | stacked totals | area | `plt.stackplot` |
# | uncertainty | errorbar | `plt.errorbar(yerr=, capsize=)` |
# | 2-D density / matrix | hexbin / imshow | `plt.hexbin`, `plt.imshow` |

# %% [markdown]
# ---
# ## 3. Labels, ticks, legend & layout
#
# ▶ Session 23 - Plotting using Matplotlib: https://www.youtube.com/watch?v=XaKn_cKFlSY

# %% [markdown]
# ### Titles & axis labels
# **Syntax:** `ax.set_xlabel(text, loc="left"|"center"|"right", labelpad=)`,
# `ax.set_ylabel(...)`, `ax.set_title(text, loc="left"|"center", pad=10)`,
# `fig.suptitle(text)` (figure-level), `ax.set_xscale("log")`.
# **Note:** pyplot twin = `plt.xlabel(...)`; OO requires the `set_` prefix.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(10), np.arange(10) ** 2, label="growth")
ax.set_xlabel("quarters", loc="left", labelpad=8)
ax.set_ylabel("revenue (Rs lakh)", rotation=0, labelpad=40, loc="center")
ax.set_title("KPI over time", loc="left", pad=12)
ax.legend()
plt.show()

# %% [markdown]
# ### `ax.legend(loc=, bbox_to_anchor=, frameon=, ncol=1, fontsize=)`
# **What it does:** decodes `label=` entries.
# **Parameters:**
# - `loc` — preset corner (`"best"`, `"upper right"`, `"lower left"`, `"center"`, ...)
# - `bbox_to_anchor` — `(x, y)` in Axes coordinates — place legend *outside* the plot (`(1.02, 1)`)
# - `frameon` — draw the box; `ncol` — number of columns (wide legends)
# - `title` — legend header (great for `hue`-style legends)

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
xs = np.arange(6)
for k, c in zip([2, 3, 5], ["#4C72B0", "#DD8452", "#55A868"]):
    axs[0].plot(xs, xs * k, marker="o", color=c, label=f"slope {k}")
axs[0].legend(loc="upper left", frameon=True, ncol=3, fontsize=8,
              title="model", bbox_to_anchor=(0, 1.18))
axs[0].set_title("ncol + title above axes")

for k, c in zip([2, 3, 5], ["#4C72B0", "#DD8452", "#55A868"]):
    axs[1].plot(xs, xs * k, marker="s", color=c, label=f"slope {k}")
axs[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False)
axs[1].set_title("legend outside (bbox_to_anchor)")

fig.tight_layout(); plt.show()

# %% [markdown]
# ### Grid, limits, ticks & margins
# **Syntax:**
# - `ax.grid(True, axis="both", alpha=0.3, linestyle="--", linewidth=0.6)` — background guides
# - `ax.set_xlim(left, right)` / `ax.set_ylim(bottom, top)` — **zoom**
# - `ax.set_xticks(ticks, labels=)` / `ax.set_yticks(...)` — choose tick positions + labels
# - `ax.set_xticklabels(...)`, `ax.set_xticks(ticks, minor=True)` — minor ticks
# - `ax.margins(x=0.02)` — padding around data (so markers don't touch the frame)
# - `ax.invert_yaxis()` — flip (e.g. depths, ranks)

# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(range(6), [3, 9, 4, 12, 6, 15], marker="o", label="metric")
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, 20)
ax.set_xticks(range(6), labels=[f"W{i+1}" for i in range(6)])
ax.set_yticks([0, 5, 10, 15, 20])
ax.margins(x=0.04)
ax.legend()
ax.set_title("grid + limits + custom ticks + margins")
plt.show()

# %% [markdown]
# ### `ax.annotate(text, xy=, xytext=, arrowprops=, ...)` · `ax.text(x, y, s)` · `ax.axvline(x)` / `ax.axhline(y)`
# **What they do:** callouts and reference lines — turn a chart into an explanation.
# - `xy` — data point being pointed at; `xytext` — where the label sits
# - `arrowprops` — `dict(arrowstyle="->", color=, lw=, connectionstyle=...)`
# - `axvline`/`axhline` — full-width/height guides (thresholds, means); `axhspan`/`axvspan` shade regions
# - `ax.text` — free text in data coords (`transform=ax.transAxes` for 0–1 panel coords)

# %%
peak = 5
fig, ax = plt.subplots(figsize=(7, 4))
ys = np.exp(np.linspace(0, 2, 30))
ax.plot(ys, color="#4C72B0")
ax.axhline(ys.mean(), color="gray", ls="--", label="mean")
ax.axvline(peak, color="red", ls=":", lw=1)
ax.axhspan(ys.mean(), ys.max(), alpha=0.12, color="green")
ax.annotate("peak!", xy=(peak, ys[peak]), xytext=(12, ys[peak] * 0.9),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
            fontsize=11, color="crimson")
ax.text(0.98, 0.05, "shaded = above mean", transform=ax.transAxes,
        ha="right", fontsize=9, style="italic")
ax.legend()
ax.set_title("annotate / axhline / axhspan / text")
plt.show()

# %% [markdown]
# ### Colormaps & `colorbar`
# **What they do:** map numbers → colours. Three families:
#
# | Family | Examples | Use |
# |---|---|---|
# | sequential | `viridis`, `plasma`, `magma`, `Blues` | magnitude (low → high) |
# | diverging | `coolwarm`, `RdBu_r`, `bwr` | deviation from a midpoint (e.g. corr ∈ [-1, 1]) |
# | qualitative | `tab10`, `Set2`, `Paired` | categories (use with `cmap` only if categorical) |
#
# **Code:** `cax = plt.scatter(..., cmap="viridis", vmin=, vmax=); plt.colorbar(cax, label=...)`
# or `fig.colorbar(mappable, ax=ax, shrink=0.8)`; reverse a map with a `"_r"` suffix.

# %%
print("colormap catalogue (first 12):", plt.colormaps()[:12])

grad = np.linspace(0, 1, 256).reshape(1, -1)
fig, axs = plt.subplots(1, 3, figsize=(12, 2.2), sharey=True)
for ax, cm, ttl in zip(axs, ["viridis", "coolwarm", "tab10"],
                       ["sequential", "diverging", "qualitative"]):
    ax.imshow(grad if cm != "tab10" else np.arange(10).reshape(1, -1),
              aspect="auto", cmap=cm)
    ax.set_title(cm, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.show()

# %% [markdown]
# ### `plt.tight_layout(pad=1.0)` · `fig.subplots_adjust(...)` · `fig.suptitle(...)`
# **What they do:** spacing hygiene. `tight_layout` auto-adjusts so labels never
# overlap (call it **before** `plt.show()`); `subplots_adjust` is the manual
# version; `suptitle` is the figure-level super-title.

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(range(5), np.arange(5) * (i + 1))
    ax.set_title(f"panel {i}")
    ax.set_xlabel("x label that is quite long")
fig.suptitle("figure-level super title", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Section notes — labels/layout:**
#
# | Need | Code |
# |---|---|
# | axis titles | `ax.set_xlabel(..., loc="left")`, `ax.set_title(..., loc="left")` |
# | legend outside | `ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")` |
# | zoom | `ax.set_xlim()`, `ax.set_ylim()` |
# | ticks | `ax.set_xticks(ticks, labels=...)` |
# | guides | `ax.grid(alpha=0.3)`, `ax.axhline`, `ax.axvline` |
# | callout | `ax.annotate(text, xy=, xytext=, arrowprops=dict(arrowstyle="->"))` |
# | no overlap | `fig.tight_layout()` + `fig.suptitle()` |
# | colour scale | `plt.colorbar(mappable, ax=ax)` |
#
# > **Gotcha:** `plt.title()` vs `ax.set_title()` — after `plt.subplots()` there
# > may be *several* Axes; pyplot calls silently target only the "current" one.
# > In multi-panel figures **always** use `ax.` methods.

# %% [markdown]
# ---
# ## 4. Customization — styles, rcParams, colours, spines & saving
#
# ▶ Session 24 - Advanced Matplotlib: https://www.youtube.com/watch?v=7YDc5xU9CQQ

# %% [markdown]
# ### `plt.style.available` · `plt.style.use(name)` · `plt.rcParams`
# **What they do:**
# - `plt.style.use("seaborn-v0_8-whitegrid")` — swap the whole look in one line
#   (Matplotlib ≥3.6 renamed the seaborn styles to `seaborn-v0_8-*`)
# - `plt.rcParams["..."] = ...` — global defaults: fonts, dpi, line widths;
#   use `matplotlib.rcParamsDefault` to inspect, `plt.style.use("default")` to reset
# - `plt.style.context(name)` — temporary style for one block
# **Returns:** `rcParams` is a live dict-like; mutate in place.

# %%
print("available styles:", plt.style.available)

with plt.style.context("seaborn-v0_8-whitegrid"):
    print("inside context, axes.grid default:",
          matplotlib.rcParams["axes.grid"])

plt.style.use("seaborn-v0_8-whitegrid")     # persistent choice
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["figure.dpi"] = 90
matplotlib.rcParams["lines.linewidth"] = 2
print("rcParams font.size =", matplotlib.rcParams["font.size"],
      "| figure.dpi =", matplotlib.rcParams["figure.dpi"])

# %%
# same data, two styles — style only changes presentation, never the numbers
x = np.linspace(0, 10, 60)
y = np.sin(x) * np.exp(-x / 6)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
with plt.style.context("dark_background"):
    axs[0].plot(x, y, color="#8dd357")
    axs[0].set_title("dark_background")
with plt.style.context("seaborn-v0_8-whitegrid"):
    axs[1].plot(x, y, color="#4C72B0")
    axs[1].set_title("seaborn-v0_8-whitegrid")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### Specifying colours
#
# | Form | Examples |
# |---|---|
# | named | `"crimson"`, `"steelblue"`, `"darkorange"`, `"gray"` |
# | hex | `"#1f77b4"`, `"#C44E52"` (RRGGBB) |
# | rgb tuple | `(0.3, 0.5, 0.8)` in 0–1, or `(..., alpha)` |
# | single letter | `b g r c m y k w` |
# | cycle | `C0`–`C9` (the default property cycle), `color="C2"` |
#
# `alpha=` (0–1) controls transparency everywhere — use it to fix overplotting.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
palette = ["crimson", "#1f77b4", (0.2, 0.6, 0.4, 0.9), "C0", "darkorange"]
for i, col in enumerate(palette):
    ax.plot(np.arange(6), np.arange(6) + i, color=col, label=str(col))
ax.legend(fontsize=8, ncol=2)
ax.set_title("named / hex / rgb / cycle colours  (+ alpha)")
plt.show()

# %% [markdown]
# ### Removing spines · `ax.spines[...]`
# **What it does:** the default frame (spines: `top/right/bottom/left`) can be
# trimmed for a cleaner "data-ink" look (Tufte). Common recipe:
# `ax.spines["top"].set_visible(False)` + `ax.spines["right"].set_visible(False)`
# (or style them: `set_color("gray")`, `set_linewidth(0.6)`).

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
data = rng.normal(size=200)

axs[0].hist(data, bins=20, color="#4C72B0")
axs[0].set_title("default spines")

axs[1].hist(data, bins=20, color="#4C72B0")
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].set_title("top/right removed")

axs[2].hist(data, bins=20, color="#4C72B0")
for side in ["top", "right", "left"]:
    axs[2].spines[side].set_visible(False)
axs[2].tick_params(axis="y", length=0)     # hide y tick marks too
axs[2].grid(axis="y", alpha=0.3)
axs[2].set_title("minimal (Tufte-ish)")

fig.tight_layout(); plt.show()

# %% [markdown]
# ### `matplotlib.ticker` — `MultipleLocator`, `FuncFormatter`, `PercentFormatter`
# **What they do:** take manual control of tick **positions** and label **text**.
# - `MultipleLocator(base)` — a tick every `base` units
# - `FuncFormatter(fn)` — `fn(value, position) -> str` (thousands separators, units)
# - `MaxNLocator(n)` — "at most n nice ticks"
# **Syntax:** `ax.xaxis.set_major_locator(...)` / `ax.xaxis.set_major_formatter(...)`

# %%
from matplotlib.ticker import MultipleLocator, FuncFormatter, PercentFormatter

fig, ax = plt.subplots(figsize=(7, 3.6))
vals = np.array([2_000, 8_000, 15_000, 40_000, 120_000, 250_000])
ax.bar(range(len(vals)), vals, color="#4C72B0")

ax.yaxis.set_major_locator(MultipleLocator(50_000))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))
ax.xaxis.set_major_formatter(PercentFormatter(xmax=len(vals) - 1))
ax.set_title("custom tick text: thousands separators + percent x-axis")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### Twin axes — `ax.twinx()` / `ax.twiny()`
# **What it does:** overlays a second y-axis (different units) on the same panel.
# **Returns:** the new Axes. Always give each axis its own label and colour.

# %%
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue = [120, 135, 148, 160, 155, 170]     # k$
growth = [0.0, 12.5, 9.6, 8.1, -3.1, 9.7]    # %

fig, ax1 = plt.subplots(figsize=(7, 4))
bars = ax1.bar(months, revenue, color="#4C72B0", label="revenue (kRs)")
ax1.set_ylabel("revenue (kRs)", color="#4C72B0")
ax1.set_ylim(0, 200)

ax2 = ax1.twinx()
line = ax2.plot(months, growth, color="crimson", marker="o", label="MoM %")
ax2.set_ylabel("MoM growth (%)", color="crimson")
ax2.axhline(0, color="gray", lw=0.8, ls="--")

# merge both legends into one box
handles = list(bars) + line
ax1.legend(handles, [h.get_label() for h in handles], loc="upper left")
ax1.set_title("twinx(): bars (kRs) + line (%)")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### `fig.savefig(fname, dpi=, bbox_inches="tight", transparent=, facecolor=)`
# **What it does:** export the **figure** (PDF/SVG/PNG — extension picks the format).
# **Parameters:** `dpi` — output resolution; `bbox_inches="tight"` — crop stray white
# space (save the legend outside!); `transparent=True` — no background.
# **Note:** prefer `fig.savefig` over `plt.savefig` (unambiguous which figure).

# %%
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot([1, 2, 3], [2, 1, 3], marker="o")
ax.set_title("saved to /tmp")
fig.savefig("/tmp/dsmp_05_line.png", dpi=150, bbox_inches="tight")
fig.savefig("/tmp/dsmp_05_line.pdf", bbox_inches="tight")
import os
print("PNG bytes:", os.path.getsize("/tmp/dsmp_05_line.png"),
      "| PDF bytes:", os.path.getsize("/tmp/dsmp_05_line.pdf"))
plt.show()

# %% [markdown]
# **Section notes — customization:**
#
# | Want | Code |
# |---|---|
# | theme | `plt.style.use("seaborn-v0_8-whitegrid")` |
# | global defaults | `matplotlib.rcParams["font.size"] = 11` |
# | temporary theme | `with plt.style.context("name"):` |
# | colours | name / hex / rgb tuple / `C0`-`C9` + `alpha` |
# | clean frame | `ax.spines["top"].set_visible(False)` |
# | nice ticks | `MultipleLocator`, `FuncFormatter`, `PercentFormatter` |
# | 2nd y-axis | `ax2 = ax.twinx()` |
# | export | `fig.savefig(path, dpi=150, bbox_inches="tight")` |

# %% [markdown]
# ---
# ## 5. Advanced Matplotlib — layouts, axes placement & 2-D data
#
# ▶ Session 24 - Advanced Matplotlib: https://www.youtube.com/watch?v=7YDc5xU9CQQ

# %% [markdown]
# ### `fig.add_axes([left, bottom, width, height])`
# **What it does:** places an Axes at **figure-fraction** coordinates (0–1) —
# the classic way to build an **inset** (zoom) panel.
# **Returns:** the new Axes.

# %%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.12, 0.12, 0.76, 0.76])
x = np.linspace(0, 10, 200)
ax.plot(x, np.sinc(x / 2), color="#4C72B0")
ax.set_title("main plot with inset")

inset = fig.add_axes([0.62, 0.62, 0.28, 0.28])     # left, bottom, w, h (figure fractions)
mask = (x > 2) & (x < 4)
inset.plot(x[mask], np.sinc(x[mask] / 2), color="crimson", lw=1.5)
inset.set_xlim(2, 4)
inset.set_xticks([]); inset.set_yticks([])
inset.set_title("zoom", fontsize=8)
plt.show()

# %% [markdown]
# ### `gridspec.GridSpec(nrows, ncols, width_ratios=, height_ratios=, wspace=, hspace=)`
# **What it does:** unequal subplot sizes — spans via `gs[a:b, c:d]`.
# **Parameters:** `width_ratios` / `height_ratios` — relative sizes per column/row;
# `wspace` / `hspace` — spacing fractions.
# **Returns:** `GridSpec`; `fig.add_subplot(gs[...])` creates the Axes.

# %%
from matplotlib import gridspec

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 2],
                       wspace=0.35, hspace=0.45)

ax_wide = fig.add_subplot(gs[0, :])          # spans both columns
ax_tl = fig.add_subplot(gs[1, 0])
ax_tall = fig.add_subplot(gs[1, 1])

xs = np.arange(30)
ax_wide.plot(xs, np.cumsum(rng.normal(size=30)), color="#4C72B0")
ax_wide.set_title("gs[0, :] — full width top row")
ax_tl.bar([0, 1, 2], [3, 5, 2], color="#55A868")
ax_tl.set_title("gs[1, 0] — narrow")
ax_tall.plot(xs, np.cumsum(rng.normal(size=30)), color="#DD8452")
ax_tall.set_title("gs[1, 1] — tall")
plt.show()

# %% [markdown]
# ### `plt.subplot2grid(shape, loc, colspan=1, rowspan=1)`
# **What it does:** old-school grid placement — specify the parent grid `shape`
# and where (row, col) the Axes starts, plus how many cells it covers.
# **Returns:** the Axes.

# %%
fig = plt.figure(figsize=(9, 4))
top = plt.subplot2grid((3, 3), (0, 0), colspan=3)   # big header plot
bl = plt.subplot2grid((3, 3), (1, 0), colspan=1)
bm = plt.subplot2grid((3, 3), (1, 1), colspan=1)
br = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
bottom = plt.subplot2grid((3, 3), (2, 0), colspan=2)

xs = np.arange(12)
top.plot(xs, np.cumsum(rng.normal(size=12)) + 50, color="#4C72B0")
top.set_title("subplot2grid: colspan=3 header")
for ax, c in zip([bl, bm, bottom], ["#55A868", "#DD8452", "#C44E52"]):
    ax.plot(xs[:6], rng.normal(size=6), color=c)
br.barh([0, 1], [4, 6], color="#8172B3")
br.set_title("rowspan=2")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### `ax.secondary_yaxis(location, functions=(f, finv))`
# **What it does:** a second scale on the *same* side structure (transformable:
# e.g. °C ↔ °F, linear ↔ log) — different from `twinx` (which adds an independent axis).

# %%
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.linspace(0, 100, 50)
ax.plot(x, x * 2, color="#4C72B0")
ax.set_xlabel("engineer age (years)")
ax.set_ylabel("experience (months)")

ax2 = ax.secondary_yaxis("right",
                         functions=(lambda m: m / 12,      # months -> years
                                    lambda y: y * 12))     # years -> months
ax2.set_ylabel("experience (years)")
ax.set_title("secondary_yaxis with unit conversion")
plt.show()

# %% [markdown]
# ### Broken axis pattern (`sharex` + hidden inner spines)
# **What it does:** when data has a gap (0–5 and 1000–1200), compress it with
# two stacked panels that **share x** and hide the spines between them.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 4),
                               gridspec_kw=dict(height_ratios=[1, 1.4]))
x = np.arange(10)
y = np.where(x < 5, x, 950 + x * 10)          # gap between 5 and 950

ax1.plot(x, y, marker="o", color="#4C72B0")
ax1.set_ylim(960, 1050)
ax2.plot(x, y, marker="o", color="#4C72B0")
ax2.set_ylim(0, 8)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(labelbottom=False, bottom=False)
d = 0.012                                        # diagonal break markers
kw = dict(transform=ax1.transAxes, color="gray", clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kw)
ax1.plot((1 - d, 1 + d), (-d, +d), **kw)
kw.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kw)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

ax2.set_xlabel("x with a broken range")
fig.suptitle("broken-axis pattern", y=0.95)
plt.show()

# %% [markdown]
# ### Animation — pointer only
# **What it exists for:** Matplotlib can render frames and assemble a GIF/MP4 with
# the `matplotlib.animation` module — `FuncAnimation(fig, update, frames=...)`
# re-calls a draw function per frame, `ArtistAnimation` re-displays saved artists.
# Use `blit=True` for speed and `interval=` (ms) for timing; export with
# `anim.save("out.gif", writer="pillow")`.
# The build/run cost is heavy for a reference notebook, so we only record the API
# here — see the Advanced Matplotlib session for a live demo:
# ▶ Session 24 - Advanced Matplotlib: https://www.youtube.com/watch?v=7YDc5xU9CQQ

# %%
# API surface (names only — no frame loop in this notebook)
from matplotlib.animation import FuncAnimation, ArtistAnimation
print("animation classes:", FuncAnimation.__name__, ArtistAnimation.__name__)

# %% [markdown]
# ### 2-D data: `contour` / `contourf` and a annotated correlation matrix
# **What they do:**
# - `contour` — isolines; `contourf` — filled bands (topographic-style surfaces)
# - `imshow(corr)` + text annotations — the classic **correlation heatmap**

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))

X, Y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
Z = np.sin(X ** 2 + Y ** 2) / (X ** 2 + Y ** 2 + 0.2)

cf = axs[0].contourf(X, Y, Z, levels=14, cmap="viridis")
cs = axs[0].contour(X, Y, Z, levels=7, colors="white", linewidths=0.6)
axs[0].clabel(cs, inline=True, fontsize=7)
fig.colorbar(cf, ax=axs[0], label="intensity")
axs[0].set_title("contourf + contour + clabel")

# annotated correlation matrix (values printed inside each cell)
rng2 = np.random.default_rng(1)
df_demo = pd.DataFrame(rng2.normal(size=(6, 80)),
                       index=["bill", "tip", "size", "wait", "age", "freq"])
df_demo.columns = [f"obs{i}" for i in range(80)]
corr = df_demo.T.corr()
im = axs[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axs[1].set_xticks(range(len(corr)), labels=corr.columns, rotation=45, ha="right")
axs[1].set_yticks(range(len(corr)), labels=corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        axs[1].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
fig.colorbar(im, ax=axs[1], shrink=0.85, label="Pearson r")
axs[1].set_title("correlation matrix via imshow")

fig.tight_layout(); plt.show()

# %% [markdown]
# **Section notes — advanced:**
#
# | Technique | Code |
# |---|---|
# | exact placement / inset | `fig.add_axes([l, b, w, h])` |
# | unequal panels | `gridspec.GridSpec(2, 2, width_ratios=[2, 1])` |
# | legacy grid | `plt.subplot2grid((3, 3), (0, 0), colspan=3)` |
# | unit-converted axis | `ax.secondary_yaxis("right", functions=(f, finv))` |
# | broken range | 2× `subplots(..., sharex=True)` + hidden spines |
# | tick text | `FuncFormatter(lambda v, pos: f"{v:,.0f}")` |
# | surfaces | `ax.contourf`, `ax.contour` (+ `clabel`) |
# | matrix image | `ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)` |
# | animation | `matplotlib.animation.FuncAnimation` (session 24) |

# %% [markdown]
# ---
# ## 6. Seaborn — statistical graphics in one line
#
# ▶ Session 25 - Plotting using Seaborn: https://www.youtube.com/watch?v=DWVLRhnuGqI
# ▶ Session 26 - Plotting using Seaborn Part 2: https://www.youtube.com/watch?v=kLWeKeqc9Ms

# %% [markdown]
# ### `sns.set_theme(style=, context=, font_scale=, palette=)`
# **What it does:** Seaborn's global defaults — style (frame/grid), context
# (line widths & font sizes: `"paper"`, `"notebook"`, `"talk"`, `"poster"`),
# `palette` (colour cycle, e.g. `"viridis"`, `"Set2"`, `"deep"`).
# **Returns:** `None` (mutates `rcParams`); `sns.axes_style()` shows the current style dict.
# **Per-plot overrides:** `style=` / `context=` args, or `with sns.axes_style("ticks"):`.

# %%
sns.set_theme(style="whitegrid", context="notebook",
              font_scale=1.05, palette="viridis")
print("styles:", ["darkgrid", "whitegrid", "dark", "white", "ticks"])
print("contexts:", ["paper", "notebook", "talk", "poster"])

# synthetic restaurant-bill dataset (deterministic)
rng = np.random.default_rng(7)
n = 400
tips = pd.DataFrame({
    "total_bill": np.round(rng.gamma(4.0, 8.0, n), 2),
    "size": rng.integers(1, 6, n),
    "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n, p=[0.3, 0.2, 0.3, 0.2]),
    "time": rng.choice(["Lunch", "Dinner"], n),
    "smoker": rng.choice(["Yes", "No"], n),
    "sex": rng.choice(["Female", "Male"], n),
})
tips["tip"] = (tips["total_bill"] * rng.uniform(0.08, 0.28, n)).round(2)
print(tips.head())
print(tips.dtypes)

# %% [markdown]
# ### Relational — `relplot` / `scatterplot` / `lineplot`
# **What they do:** how two variables relate, split by categories.
# **Key params (all three):** `hue` (colour = category), `style` (marker/linestyle),
# `size` (marker size = 4th variable), `col`/`row` (small-multiple grids, `relplot` only),
# `palette`, `alpha`.
# **Level:** `relplot` is **figure-level** (returns `FacetGrid`); `scatterplot`/`lineplot`
# are **axes-level** (draw into an existing `ax`, return the `Axes`).
# **Returns:** `FacetGrid` / `Axes`.

# %%
# figure-level: the whole grid in one call
sns.relplot(data=tips, x="total_bill", y="tip", hue="day", style="time",
            size="size", sizes=(20, 220), height=4.5, aspect=1.3, alpha=0.85)
plt.show()

# %%
# axes-level: fits inside your own subplot grid
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="sex",
                size="size", sizes=(25, 200), alpha=0.8, ax=axs[0])
axs[0].set_title("scatterplot(hue, style, size)")

sns.lineplot(data=tips, x="size", y="tip", hue="time", errorbar=("ci", 95),
             marker="o", ax=axs[1])
axs[1].set_title("lineplot with 95% CI band")
fig.tight_layout(); plt.show()

# %% [markdown]
# ### Distribution — `histplot` / `kdeplot` / `rugplot` / `displot`
# **What they do:** the shape of the data.
# - `histplot(bins=, kde=True, stat="density")` — bins (+ optional density curve)
# - `kdeplot(fill=True, bw_adjust=0.8, common_norm=False)` — smoothed density
# - `rugplot` — one tick per observation (1-D rug)
# - `displot(kind="hist"|"kde"|"ecdf")` — **figure-level** container for all three
#   (`col=`/`row=` make grids; `common_norm` merges densities per hue)
# **Returns:** `Axes` (hist/kde/rug) / `FacetGrid` (displot).

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
sns.histplot(data=tips, x="total_bill", bins=25, kde=True, stat="density",
             color="#4C72B0", ax=axs[0])
axs[0].set_title("histplot(bins, kde, stat)")

sns.kdeplot(data=tips, x="total_bill", hue="time", fill=True, bw_adjust=0.7,
            common_norm=False, alpha=0.4, ax=axs[1])
axs[1].set_title("kdeplot(fill, bw_adjust, hue)")

sns.histplot(data=tips, x="tip", color="#55A868", ax=axs[2])
sns.rugplot(data=tips, x="tip", hue="time", height=0.06, ax=axs[2])
axs[2].set_title("hist + rugplot")

fig.tight_layout(); plt.show()

# %%
# figure-level distribution grids
sns.displot(data=tips, x="total_bill", col="time", row="smoker",
            kind="kde", fill=True, height=3.5, aspect=1.1)
plt.show()
sns.displot(data=tips, x="tip", kind="ecdf", hue="day", height=3.5)
plt.show()

# %% [markdown]
# ### `pairplot` & `jointplot` — the pair explorers
# - `sns.pairplot(df, vars=[...], hue=, corner=True)` — scatter matrix of every
#   variable pair (diagonals = univariate distributions)
# - `sns.jointplot(x=, y=, kind="scatter"|"kde"|"hex"|"reg", hue=)` — the 2-D
#   relationship with both 1-D marginals
# **Returns:** `PairGrid` / `JointGrid` (figure-level objects with `.fig`, `.ax_joint`).

# %%
sns.pairplot(tips, vars=["total_bill", "tip", "size"], hue="day",
             corner=True, height=2.2)
plt.show()

# %%
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", gridsize=14)
plt.show()
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde",
              fill=True, cmap="mako", levels=8)
plt.show()

# %% [markdown]
# ### Categorical — `boxplot` / `violinplot` / `stripplot` / `swarmplot` / `countplot` / `barplot` / `catplot`
# **What they do:** distributions & summaries **per category**.
#
# | Function | Shows | Key params |
# |---|---|---|
# | `boxplot` | quartiles + outliers | `order=`, `hue=`, `showfliers=` |
# | `violinplot` | density shape | `inner="box"/"stick"/"point"`, `split=True` |
# | `stripplot` | raw points | `jitter=True`, `alpha=` |
# | `swarmplot` | raw points, no overlap | `size=` |
# | `countplot` | category counts | `hue=` (no y needed) |
# | `barplot` | mean + CI | `estimator=`, `errorbar=("ci",95)/("sd",1)`, `hue=` |
# | `catplot` | figure-level wrapper | `kind="box"/"violin"/"strip"/"bar"/...`, `col=`, `row=` |
#
# **Returns:** `Axes` (single) / `FacetGrid` (`catplot`).

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
order = ["Thur", "Fri", "Sat", "Sun"]

sns.boxplot(data=tips, x="day", y="tip", hue="time", order=order,
            showfliers=True, ax=axs[0, 0])
axs[0, 0].set_title("boxplot(order, hue, showfliers)")

sns.violinplot(data=tips, x="day", y="tip", hue="time", order=order,
               split=True, inner="box", ax=axs[0, 1])
axs[0, 1].set_title("violinplot(split, inner='box')")

sns.stripplot(data=tips, x="day", y="tip", order=order, jitter=True,
              alpha=0.5, size=4, ax=axs[1, 0])
axs[1, 0].set_title("stripplot(jitter)")

sns.swarmplot(data=tips, x="day", y="tip", order=order, size=3, ax=axs[1, 1])
axs[1, 1].set_title("swarmplot")

fig.tight_layout(); plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=tips, x="day", hue="time", order=["Thur", "Fri", "Sat", "Sun"],
              ax=axs[0])
axs[0].set_title("countplot(x, hue)")

sns.barplot(data=tips, x="day", y="tip", hue="time",
            order=["Thur", "Fri", "Sat", "Sun"],
            estimator="mean", errorbar=("sd", 1), ax=axs[1])
axs[1].set_title("barplot(estimator='mean', errorbar=('sd',1))")
fig.tight_layout(); plt.show()

# figure-level categorical grid (one line, two panels by smoker status)
sns.catplot(data=tips, x="day", y="tip", hue="time", col="smoker",
            order=["Thur", "Fri", "Sat", "Sun"],
            kind="box", height=3.6, aspect=1.1)
plt.show()

# %% [markdown]
# ### Matrix — `heatmap` & `clustermap`
# - `sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm",
#   mask=upper_triangle, square=True, linewidths=0.5, cbar_kws={"label": "r"})`
# - `sns.clustermap(df, ...)` — heatmap **plus** hierarchical clustering of rows/cols
#   (`method="average"`, `z_score=0` to standardise)
# **Returns:** `Axes` (heatmap) / `ClusterGrid` (clustermap).

# %%
corr = tips[["total_bill", "tip", "size"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))     # hide the upper triangle

fig, ax = plt.subplots(figsize=(5.5, 4.2))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap="coolwarm", square=True, linewidths=0.6,
            cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("heatmap(corr, annot, mask)")
plt.show()

sns.clustermap(corr, annot=True, fmt=".2f", cmap="viridis",
               linewidths=0.6, figsize=(5, 4.5), method="average")
plt.show()

# %% [markdown]
# ### Regression — `regplot` & `lmplot`
# **What they do:** scatter + fitted linear regression with a confidence band.
# **Params:** `order=` (polynomial fit), `ci=`/`errorbar=`, `scatter_kws=`,
# `line_kws=`, `truncate=`.
# **Level:** `regplot` → **axes-level**; `lmplot` → **figure-level** (`col=`/`hue=` grids).

# %%
fig, ax = plt.subplots(figsize=(6.5, 4))
sns.regplot(data=tips, x="total_bill", y="tip", order=2,
            scatter_kws={"alpha": 0.4, "s": 25},
            line_kws={"color": "crimson"}, ax=ax)
ax.set_title("regplot(order=2, alpha scatter, red fit line)")
plt.show()

sns.lmplot(data=tips, x="total_bill", y="tip", hue="time", palette="Set2",
           height=4.2, aspect=1.2, scatter_kws={"alpha": 0.5})
plt.show()

# %% [markdown]
# > **Gotcha — figure-level vs axes-level:**
# > Seaborn functions come in two flavours. Getting this wrong is the #1
# > Seaborn confusion.
#
# | Level | Functions | Returns | Can draw on *my* `ax`? |
# |---|---|---|---|
# | axes-level | `scatterplot`, `lineplot`, `boxplot`, `violinplot`, `barplot`, `countplot`, `stripplot`, `swarmplot`, `histplot`, `kdeplot`, `rugplot`, `heatmap`, `regplot` | `Axes` | ✓ pass `ax=` |
# | figure-level | `relplot`, `displot`, `catplot`, `pairplot`, `jointplot`, `lmplot`, `clustermap` | `FacetGrid` / `JointGrid` / `ClusterGrid` | ✗ builds its **own** Figure (`g.fig`) |
#
# Figure-level functions are for `col=`/`row=` small-multiple grids; anything
# custom or combined → use the axes-level twin.

# %%
g = sns.relplot(data=tips, x="total_bill", y="tip", hue="day", height=3.5)
ax = sns.scatterplot(data=tips, x="total_bill", y="tip", s=20)
print("figure-level ->", type(g).__name__, "(owns", type(g.fig).__name__ + ")")
print("axes-level   ->", type(ax).__name__, "(owns", type(ax.figure).__name__ + ")")

# axes-level drops into your own subplot grid:
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.boxplot(data=tips, x="day", y="tip", hue="time", ax=ax)
ax.set_title("axes-level boxplot drawn on MY axes")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Section notes — Seaborn:**
#
# | Task | Code |
# |---|---|
# | theme | `sns.set_theme(style="whitegrid", context="notebook", palette="viridis")` |
# | relationship | `sns.relplot(x, y, hue, style, size)` · `sns.scatterplot/lineplot` |
# | distribution | `sns.histplot(kde=True)` · `sns.kdeplot(fill=True)` · `sns.displot(kind=)` |
# | pairs | `sns.pairplot(vars, hue, corner=True)` · `sns.jointplot(kind="hex")` |
# | categories | `sns.boxplot` `sns.violinplot(split=True)` `sns.stripplot(jitter)` `sns.swarmplot` `sns.countplot` `sns.barplot(estimator, errorbar)` `sns.catplot(kind=, col=)` |
# | matrix | `sns.heatmap(corr, annot=True, mask=)` · `sns.clustermap` |
# | regression | `sns.regplot` (axes) · `sns.lmplot(hue=)` (figure) |
# | small multiples | `col=`/`row=` on figure-level funcs |

# %% [markdown]
# ---
# ## 7. Plotly Express — interactive, declarative figures
#
# ▶ Session on Plotly Express: https://www.youtube.com/watch?v=VPqw8soG4ZY
# ▶ Session on Plotly Part 2: https://www.youtube.com/watch?v=_ZsVLh6v29A

# %% [markdown]
# ### Renderer setup (offline, no browser pop-ups)
# **What it does:** Plotly needs a *renderer* to display figures. For notebooks:
#
# ```python
# import plotly.io as pio
# pio.renderers.default = "notebook"   # inline HTML bundle; never opens a browser
# ```
#
# Then simply make a cell's **last expression** the figure (`fig`) — that is the
# display hook. `fig.show()` would try to open a browser — avoid it in kernels.
# Export instead with `fig.write_html(path)` (works with zero internet).
# **Note:** there is no `px.heatmap` — matrices use `px.imshow`, binned 2-D
# densities use `px.density_heatmap`.

# %%
# shared axis data for this section
rng = np.random.default_rng(11)
months = pd.date_range("2023-01-01", periods=18, freq="MS").strftime("%b %Y")
demo = pd.DataFrame({
    "month": np.tile(months, 4),
    "region": np.repeat(["North", "South", "East", "West"], len(months)),
    "revenue": np.concatenate([rng.normal(120, 25, len(months)),
                               rng.normal(95, 18, len(months)),
                               rng.normal(140, 30, len(months)),
                               rng.normal(110, 22, len(months))]).round(1),
    "units": rng.integers(50, 500, 4 * len(months)),
})
demo.head()

# %% [markdown]
# ### `px.line(data_frame, x=, y=, color=, title=)` — the one-liner API
# **What it does:** Plotly Express is *declarative*: you map **columns → visual
# properties** (`x`, `y`, `color`, `size`, `facet_col`, ...) and it builds a
# `plotly.graph_objects.Figure`.
# **Returns:** `plotly.graph_objects.Figure` (interactive: hover, zoom, pan,
# legend toggle, PNG export in the toolbar).

# %%
fig = px.line(demo, x="month", y="revenue", color="region",
              markers=True, title="revenue by region (hover / zoom!)",
              color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(template="plotly_white", height=380,
                  legend_title_text="Region")
fig.update_xaxes(title="Month", tickangle=-45)
fig.update_yaxes(title="Revenue (k$)", showgrid=True)
fig      # <- last expression = the display hook

# %% [markdown]
# ### `px.scatter(..., color=, hover_data=, size=, log_x=, trendline=)` · `px.bar(..., barmode=)` · `px.histogram(..., nbins=)`
# **What they do:** the interactive equivalents of §2's charts.
# **Common params:** `color` (categorical/continuous legend), `hover_data=[...]`
# (extra tooltip columns), `log_x` / `log_y`, `text` (bar labels via `text_auto`),
# `barmode="group"|"stack"`, `pattern_shape` (second encoding).

# %%
fig = px.scatter(demo, x="units", y="revenue", color="region",
                 hover_data=["month"], size="revenue",
                 opacity=0.75, title="px.scatter: size + hover + colour")
fig.update_layout(template="plotly_white", height=360)
fig

# %%
fig = px.bar(demo, x="region", y="revenue", color="region",
             barmode="group", text_auto=".1f",
             title="px.bar (text_auto prints values on bars)")
fig.update_layout(showlegend=False, template="plotly_white", height=340)
fig

# %%
fig = px.histogram(demo, x="revenue", color="region", nbins=20,
                   barmode="overlay", opacity=0.75,
                   marginal="box",      # adds a box plot strip above
                   title="px.histogram (+ marginal='box')")
fig.update_layout(template="plotly_white", height=380)
fig

# %% [markdown]
# ### `px.box` / `px.violin` · `px.area` · `px.imshow` / `px.density_heatmap`
# **What they do:** distribution, composition-over-time and matrix views.
# **Params:** `box.points="all"` shows every observation; `violin(box=True, points="outliers")`;
# `px.imshow(z, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")`.

# %%
fig = px.box(demo, x="region", y="revenue", color="region",
             points="all", title="px.box(points='all')")
fig.update_layout(template="plotly_white", height=360, showlegend=False)
fig

# %%
fig = px.area(demo, x="month", y="revenue", color="region",
              title="px.area (stacked composition)")
fig.update_layout(template="plotly_white", height=360)
fig

# %%
corr = demo[["revenue", "units"]].corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, title="px.imshow — correlation matrix")
fig.update_layout(height=330, width=380)
fig

# %%
# px.density_heatmap = binned 2-D histogram with a colour bar
fig = px.density_heatmap(demo, x="units", y="revenue", nbinsx=15,
                         color_continuous_scale="Viridis",
                         title="px.density_heatmap — 2-D bin counts")
fig.update_layout(height=360)
fig

# %% [markdown]
# ### Hierarchical: `px.sunburst(path=, values=)` and `px.treemap(path=, values=)`
# **What they do:** part-of-whole with **multiple levels** — `path` is the
# hierarchy from root to leaves (`["continent", "country", "city"]`), `values`
# the size of each node. Click a node to drill down.

# %%
tree = (demo.groupby(["region"])
        .agg(revenue=("revenue", "sum"))
        .reset_index())
tree["category"] = "All"
tree = pd.concat([
    tree,
    (demo.groupby(["region", "month"])
     .agg(revenue=("revenue", "sum")).reset_index()
     .assign(category=lambda d: d["month"])),
], ignore_index=True)
tree = tree[["region", "category", "revenue"]]

fig = px.sunburst(tree, path=["region", "category"], values="revenue",
                  color="region", title="px.sunburst — click to drill down")
fig.update_layout(height=420, margin=dict(t=40, b=10, l=10, r=10))
fig

# %%
fig = px.treemap(tree, path=["region", "category"], values="revenue",
                 title="px.treemap — same hierarchy, rectangular layout")
fig.update_layout(height=400, margin=dict(t=40, b=10, l=10, r=10))
fig

# %% [markdown]
# ### Small multiples: `facet_col=`, `facet_row=` · `animation_frame=`
# **What they do:** Seaborn's `col=`/`row=` equivalent for free grids; and
# `animation_frame` turns a column into a play-button time axis.

# %%
facet = demo.copy()
facet["half"] = np.where(np.arange(len(facet)) % 2 == 0, "H1", "H2")
fig = px.scatter(facet, x="units", y="revenue", color="region",
                 facet_col="region", facet_row="half",
                 height=520, title="px facet_col + facet_row")
fig.update_layout(template="plotly_white", showlegend=False)
fig

# %%
# animation: one frame per month (small categorical x kept numeric here)
anim = demo.copy()
anim["frame"] = np.tile(np.arange(len(months)), 4)
anim["quarter"] = pd.cut(anim["frame"], bins=[-1, 5, 11, 17],
                         labels=["Q1-Q2", "Q3-Q4", "Q5+"]).astype(str)
fig = px.bar(anim, x="region", y="revenue", color="region",
             animation_frame="frame", range_y=[0, demo["revenue"].max() * 1.2],
             title="px animation_frame — press ▶ to advance frames")
fig.update_layout(showlegend=False, template="plotly_white", height=360)
fig

# %% [markdown]
# ### `fig.update_layout(...)` · `fig.update_xaxes(...)` · `add_hline` / `add_vline` · `fig.write_html(path)`
# **What they do:** the styling/export surface shared by *every* Plotly figure.
# - `update_layout(title=, template="plotly_white"|"plotly_dark"|"ggplot2", height=,
#   width=, legend=, font=, margin=)` — the frame
# - `update_xaxes(title_text=, tickangle=, showgrid=, range=)` (and `update_yaxes`)
#   — per-axis polish; `fig.update_layout(...)` for both at once
# - `fig.add_hline(y=0.5, line_dash="dash", annotation_text=...)` /
#   `fig.add_vline(x=...)` — reference lines (also `add_hrect`/`add_vrect` for bands)
# - `fig.write_html("/tmp/out.html")` — standalone HTML (self-contained, offline)

# %%
fig = px.line(demo, x="month", y="revenue", color="region",
              title="styled + annotated + exported")
fig.add_hline(y=demo["revenue"].mean(), line_dash="dash", line_color="gray",
              annotation_text="overall mean", annotation_position="bottom right")
fig.add_vrect(x0="Apr 2023", x1="Jul 2023", fillcolor="Orange",
              opacity=0.15, line_width=0)
fig.update_layout(template="plotly_white", height=380,
                  legend=dict(orientation="h", y=1.12, x=0, xanchor="left"),
                  xaxis_title="Month", yaxis_title="Revenue (k$)")
fig.update_xaxes(tickangle=-45)

fig.write_html("/tmp/dsmp_05_line.html")     # openable in any browser, offline
import os
print("wrote /tmp/dsmp_05_line.html ->", os.path.getsize("/tmp/dsmp_05_line.html"), "bytes")
fig

# %% [markdown]
# ### `plotly.graph_objects` + `make_subplots` — low-level control
# **What they do:** when you need precise traces (or mixed trace types in one
# grid), build the figure trace-by-trace.
# **Syntax:** `make_subplots(rows=, cols=, shared_xaxes=, subplot_titles=(...))` →
# `fig.add_trace(go.Scatter(...), row=1, col=1)`, `fig.add_trace(go.Bar(...), row=2, col=1)`.
# **Familiar API:** `go.Scatter` ≈ `plt.plot` (`mode="lines+markers"`), `go.Bar`,
# `go.Histogram`, `go.Heatmap`, `go.Box`.

# %%
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("line", "bar", "histogram", "area"),
                    shared_xaxes=False, vertical_spacing=0.15,
                    horizontal_spacing=0.1)

xs = np.arange(12)
fig.add_trace(go.Scatter(x=xs, y=np.cumsum(rng.normal(size=12)) + 50,
                         mode="lines+markers", name="cumulative",
                         line=dict(color="#636EFA")), row=1, col=1)
fig.add_trace(go.Bar(x=list("ABCD"), y=[30, 18, 25, 12], name="by channel",
                     marker_color="#EF553B"), row=1, col=2)
fig.add_trace(go.Histogram(x=rng.normal(size=400), nbinsx=20, name="dist",
                           marker_color="#00CC96"), row=2, col=1)
fig.add_trace(go.Scatter(x=xs, y=rng.uniform(5, 15, 12), fill="tozeroy",
                         mode="lines", name="range",
                         line=dict(color="#AB63FA")), row=2, col=2)

fig.add_hline(y=0, line_dash="dot", row=2, col=1)   # reference line on a subplot
fig.update_layout(height=520, showlegend=False,
                  template="plotly_white", title_text="go + make_subplots")
fig

# %% [markdown]
# **Section notes — Plotly:**
#
# | Task | Code |
# |---|---|
# | inline renderer | `pio.renderers.default = "notebook"` then end cell with `fig` |
# | basic charts | `px.line/scatter/bar/histogram/box/violin/area` |
# | matrices | `px.imshow`, `px.density_heatmap` (no `px.heatmap`) |
# | hierarchy | `px.sunburst/treemap(path=[...], values=)` |
# | encodings | `color`, `size`, `hover_data`, `log_x`, `pattern_shape`, `text_auto` |
# | grids | `facet_col=`, `facet_row=` |
# | time animation | `animation_frame=` |
# | style | `fig.update_layout(template=, height=)`, `fig.update_xaxes(title=)` |
# | reference lines | `fig.add_hline(...)`, `fig.add_vline(...)`, `add_hrect/vrect` |
# | export | `fig.write_html("/tmp/out.html")` |
# | precise traces | `make_subplots` + `go.Scatter/go.Bar/...` |

# %% [markdown]
# ---
# ## 8. Which library? Which chart?
#
# ▶ Session on Plotly Express: https://www.youtube.com/watch?v=VPqw8soG4ZY
# ▶ Session 25 - Plotting using Seaborn: https://www.youtube.com/watch?v=DWVLRhnuGqI

# %% [markdown]
# ### Decision table
#
# | Situation | Reach for | Why |
# |---|---|---|
# | Static figure for a report / paper / README | **Matplotlib** | pixel control, `savefig` any format, no JS |
# | Quick statistical look at a dataframe | **Seaborn** | one line = distribution + grouping + CI |
# | Small-multiple statistical grids (`col=`, `hue=`) | **Seaborn** `relplot/displot/catplot` | grids & palettes for free |
# | Interactive exploration / dashboard | **Plotly** | hover, zoom, legend toggling |
# | Web dashboard app | **Plotly + Dash** | callbacks on top of the same figures |
# | Publication-grade custom layout (insets, broken axes) | **Matplotlib** | `add_axes`, `GridSpec` |
#
# ### Chart-choice cheat sheet
#
# | Question | Chart |
# |---|---|
# | comparison across categories | bar (`barh` if long labels) |
# | trend over time | line (or area if cumulative) |
# | distribution of one variable | histogram / KDE / box / violin |
# | relationship between two variables | scatter (+ `hue`/`color` for groups) |
# | correlation between many numeric vars | heatmap of `corr()` / pairplot |
# | composition / share of total | pie (≤ 5 slices), stacked bar, treemap/sunburst |
# | part-to-whole over time | stacked area |
# | uncertainty | errorbar / CI bands (`errorbar=("ci", 95)`) |
# | geographic | choropleth (`px.choropleth`) |
#
# ### Common mistakes
#
# - **Truncated y-axis** — bar charts must start at 0 (a clipped bar axis
#   exaggerates differences; only line charts may zoom).
# - **Overplotting** — thousands of points need `alpha`, `hexbin`,
#   `bins`/`kde`, or 2-D density.
# - **Pie > 5 slices** — humans can't compare angles; use a sorted bar chart.
# - **Rainbow colormaps for magnitude** — use `viridis`/`plasma` (perceptually
#   uniform) or `coolwarm` (diverging ± midpoint); `jet` lies about structure.
# - **Double encoding** — don't use both colour *and* size for the same variable.
# - **Chart junk** — heavy 3-D, shadows, gradients: if it isn't data, drop it.
# - **Forgetting units/labels** — an axis without a label+unit is meaningless.
# - **Cherry-picked bins** — histograms change story with `bins`; show KDE too.

# %% [markdown]
# ---
# ## 9. Mini end-to-end — one dataset, three libraries
#
# ▶ Session 23 - Plotting using Matplotlib: https://www.youtube.com/watch?v=XaKn_cKFlSY
# ▶ Session 26 - Plotting using Seaborn Part 2: https://www.youtube.com/watch?v=kLWeKeqc9Ms
# ▶ Session on Plotly Part 2: https://www.youtube.com/watch?v=_ZsVLh6v29A

# %%
# 0. seeded business data: 24 months x 4 regions x 3 categories
rng = np.random.default_rng(2024)
months = pd.date_range("2023-01-01", periods=24, freq="MS")
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Apparel", "Grocery"]

sales = pd.DataFrame({
    "month": np.tile(np.repeat(months, len(categories)), len(regions)),
    "region": np.repeat(regions, len(months) * len(categories)),
    "category": np.tile(np.tile(categories, len(months)), len(regions)),
})
season = 1 + 0.15 * np.sin(np.arange(len(sales)) % 24 / 24 * 2 * np.pi)
base = {"Electronics": 210, "Apparel": 150, "Grocery": 90}
sales["revenue"] = np.round(
    sales["category"].map(base).values * season
    * rng.uniform(0.75, 1.25, len(sales)), 1)
sales["units"] = (sales["revenue"] / rng.uniform(8, 14, len(sales))).round(0)
print(sales.shape)
sales.head()

# %%
# 1. Matplotlib: a 2x2 dashboard on one figure (annotated)
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# (a) trend — total revenue per month
monthly = sales.groupby("month")["revenue"].sum()
axs[0, 0].plot(monthly.index, monthly.values, marker="o", ms=4,
               color="#4C72B0", lw=2)
axs[0, 0].set_title("(a) monthly revenue — trend")
axs[0, 0].set_ylabel("revenue (k$)")
axs[0, 0].tick_params(axis="x", rotation=45)

# (b) category comparison
cat_rev = sales.groupby("category")["revenue"].sum().sort_values()
axs[0, 1].barh(cat_rev.index, cat_rev.values, color=["#55A868", "#DD8452", "#4C72B0"])
axs[0, 1].set_title("(b) revenue by category")
axs[0, 1].set_xlabel("revenue (k$)")
for y, v in enumerate(cat_rev.values):
    axs[0, 1].text(v, y, f" {v:,.0f}", va="center", fontsize=9)

# (c) distribution of monthly totals
axs[1, 0].hist(monthly.values, bins=12, color="#C44E52", edgecolor="white")
axs[1, 0].axvline(monthly.mean(), ls="--", color="black", lw=1.2,
                  label=f"mean = {monthly.mean():,.0f}")
axs[1, 0].set_title("(c) distribution of monthly totals")
axs[1, 0].legend()

# (d) units vs revenue with a highlight annotation
sc = axs[1, 1].scatter(sales["units"], sales["revenue"],
                       c=sales["revenue"], cmap="viridis", s=28, alpha=0.85)
best = sales.loc[sales["revenue"].idxmax()]
axs[1, 1].annotate(f"peak: {best['revenue']:,.0f}",
                   xy=(best["units"], best["revenue"]),
                   xytext=(best["units"] * 0.75, best["revenue"] * 1.05),
                   arrowprops=dict(arrowstyle="->", color="black"))
axs[1, 1].set_title("(d) units vs revenue")
axs[1, 1].set_xlabel("units"); axs[1, 1].set_ylabel("revenue (k$)")
fig.colorbar(sc, ax=axs[1, 1], shrink=0.85, label="revenue")

fig.suptitle("Sales dashboard — 2023/24", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig("/tmp/dsmp_05_dashboard.png", dpi=120, bbox_inches="tight")
plt.show()

# %%
# 2. Seaborn statistical plots (annotated)
fig, ax = plt.subplots(figsize=(6, 4.5))
corr = sales[["revenue", "units"]].assign(
    month_num=sales["month"].dt.month).corr()
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm",
            square=True, linewidths=0.6,
            cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("S1: correlation heatmap (seaborn)")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=sales, x="category", y="revenue", hue="region",
            showfliers=False, ax=ax, palette="Set2")
ax.set_title("S2: revenue spread by category x region (seaborn)")
ax.set_ylabel("revenue (k$)")
plt.show()

# %%
# 3. Plotly interactive figure (annotated)
monthly_region = (sales.groupby(["month", "region"], as_index=False)["revenue"].sum())
fig = px.line(monthly_region, x="month", y="revenue", color="region",
              markers=True, title="P1: interactive monthly revenue by region")
fig.add_hline(y=monthly_region["revenue"].mean(), line_dash="dash",
              line_color="gray", annotation_text="overall mean")
fig.update_layout(template="plotly_white", height=400,
                  xaxis_title="Month", yaxis_title="Revenue (k$)",
                  legend_title_text="Region")
fig.update_xaxes(tickangle=-45)
fig.write_html("/tmp/dsmp_05_sales.html")
fig

# %% [markdown]
# ---
# ## 10. Quick revision sheet
#
# | Task | Code |
# |---|---|
# | notebook magic | `%matplotlib inline` as the first line |
# | figure + axes | `fig, ax = plt.subplots(figsize=(w, h))` |
# | line / scatter | `ax.plot(x, y, ls, marker, label)` / `ax.scatter(s=, c=, cmap=, alpha=)` |
# | bars | `ax.bar(x, h, width=)` / `ax.barh(y, w)` (stack with `bottom=`) |
# | histogram | `ax.hist(v, bins=, density=, cumulative=, edgecolor=)` |
# | pie (≤5) | `ax.pie(s, labels=, autopct="%1.1f%%", explode=, startangle=)` |
# | annotations | `ax.annotate(t, xy=, xytext=, arrowprops=dict(arrowstyle="->"))` |
# | reference lines | `ax.axhline(y)`, `ax.axvline(x)`, `ax.axhspan()` |
# | tidy labels | `ax.set_xlabel/set_ylabel/set_title(loc="left")`, `fig.suptitle`, `fig.tight_layout()` |
# | legend outside | `ax.legend(bbox_to_anchor=(1.02, 1))` |
# | styles | `plt.style.use("seaborn-v0_8-whitegrid")`, `matplotlib.rcParams[...]` |
# | clean frame | `ax.spines["top"].set_visible(False)` |
# | custom ticks | `FuncFormatter`, `MultipleLocator` |
# | twin axis | `ax.twinx()` |
# | unequal grids | `gridspec.GridSpec(width_ratios=)`, `fig.add_axes([...])` |
# | export | `fig.savefig(path, dpi=150, bbox_inches="tight")` |
# | seaborn theme | `sns.set_theme(style=, context=, palette=)` |
# | seaborn grids | `sns.relplot/displot/catplot(..., col=, hue=)` |
# | seaborn matrix | `sns.heatmap(corr, annot=True, mask=upper, cmap="coolwarm")` |
# | plotly setup | `pio.renderers.default = "notebook"` → end cell with `fig` |
# | plotly chart | `px.line/scatter/bar/histogram/box/...(..., color=, facet_col=)` |
# | plotly style | `fig.update_layout(template=, height=)`, `fig.update_xaxes(title=)` |
# | plotly export | `fig.write_html("/tmp/out.html")` |
#
# > **Interview short-list:** Figure vs Axes vs Artist · pyplot vs OO API ·
# > `tight_layout` vs `bbox_inches="tight"` · figure-level vs axes-level Seaborn ·
# > `hue` vs `col` · diverging vs sequential colormap · when a pie chart is wrong ·
# > truncated y-axis bias · `alpha`/`hexbin` vs overplotting ·
# > renderer setup for Plotly in a kernel.

# %% [markdown]
# ---
# **Next notebook:** `06_data_analysis_process.ipynb` — the EDA process that
# decides *which* of these charts to draw, in what order.
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
