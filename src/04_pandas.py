# %% [markdown]
# # 04 · Pandas — Complete Function Reference
#
# pandas ("panel data") is *the* tabular workhorse of Python data science: a
# labelled, rectangular, heterogeneous table (`DataFrame`) plus a labelled 1-D
# array (`Series`), built on top of NumPy. Almost every EDA pipeline in the
# playlist is *load with pandas → clean → aggregate → visualise*.
#
# **What pandas adds over NumPy:**
#
# | Feature | NumPy `ndarray` | pandas `Series` / `DataFrame` |
# |---|---|---|
# | Labels | integer positions only | arbitrary index (names, dates, strings) |
# | Dtypes | one dtype for the whole array | every column its own dtype |
# | Missing data | just `np.nan`, propagates | first-class `NaN` / `NaT` / `None` |
# | Labeled ops | axis integers | `df.loc[...]`, `groupby`, `merge`, `.str`, `.dt` |
# | IO | manual | `read_csv` / `read_excel` / `read_json` / ... |
#
# ```
# pip install pandas
# import pandas as pd
# ```
#
# > **Note:** every example runs top-to-bottom in a fresh kernel (pandas 2.x).
# > Two pandas 1.x habits are **gone** in 2.x and are flagged where relevant:
# > `df.append()` (→ `pd.concat`) and `fillna(method=...)` (→ `.ffill()` / `.bfill()`).

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 50 | Session 16 - Pandas Series | https://www.youtube.com/watch?v=zCDVUyq8lkw |
# | 51 | Session 17 - Pandas DataFrame | https://www.youtube.com/watch?v=OG31yhRQxPI |
# | 52 | Important Series Methods | https://www.youtube.com/watch?v=80QpbZA38HA |
# | 53 | Session 18 - Important DataFrame Methods | https://www.youtube.com/watch?v=zTa4MIrGTIE |
# | 56 | Session 19 - GroupBy Object in Pandas | https://www.youtube.com/watch?v=LPBjF4_gZnI |
# | 60 | Session 20 - Merging, Joining & Concatenating | https://www.youtube.com/watch?v=Ssy1EfK5S-o |
# | 61 | Session 21 - MultiIndex Series and DataFrames | https://www.youtube.com/watch?v=QzoRUwz8DoM |
# | 70 | Session 22 - Vectorized String Ops / DateTime / Pivot Table | https://www.youtube.com/watch?v=UE6DmRQJ2w8 |
# | 63 | Pandas Case Study 1 / Indian Startup Funding | https://www.youtube.com/watch?v=VGPnRAASARY |
# | 71 | Pandas Case Study / Time Series Analysis | https://www.youtube.com/watch?v=LknR_AVYboc |
# | 72 | Pandas Case Study 2 / Working with textual data | https://www.youtube.com/watch?v=cO7sebFXWT8 |

# %%
import numpy as np
import pandas as pd
import io
import timeit
import warnings

print("pandas", pd.__version__)
# NOTE: everything below uses the alias `pd`

# %% [markdown]
# ---
# ## 1. Series — creation, attributes & indexing
#
# ▶ Session 16 - Pandas Series: https://www.youtube.com/watch?v=zCDVUyq8lkw

# %% [markdown]
# ### `pd.Series(data=None, index=None, dtype=None, name=None, copy=False)`
# **What it does:** one-dimensional labelled array — a NumPy array plus an index
# (and optionally a `name`, which later becomes a DataFrame column name).
# **Syntax:** `pd.Series(data, index=None, dtype=None, name=None, copy=False)`
# **Parameters:**
# - `data` — list, dict, ndarray, scalar, mapping or another Series
# - `index` — array-like of labels; same length as `data` (default `RangeIndex(0..n-1)`)
# - `dtype` — target dtype; omitted → pandas *infers* the narrowest dtype
# - `name` — a label for the Series
# - `copy` — copy input data (default `False`; pandas may still copy)
# **Returns:** `pandas.Series`

# %%
# from list + custom index
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="marks")
print(s, "\n")

# from dict -> index = dict keys (order preserved)
print(pd.Series({"math": 90, "phy": 85, "chem": 88}), "\n")

# from ndarray -> index defaults to RangeIndex
print(pd.Series(np.arange(3)), "\n")

# scalar + index -> pandas broadcasts the scalar to every label
print(pd.Series(7, index=["x", "y", "z"], name="const"), "\n")

# explicit dtype
print(pd.Series([1, 2, 3], dtype="int8"))
print(pd.Series([1, 2], dtype=float).dtype)

# %% [markdown]
# ### Series attributes
# **What they do:** the metadata of a Series — shape/size like NumPy, plus the
# pandas-specific `index`, `name` and `axes`.

# %%
s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="marks")
print("shape  :", s.shape)                 # (3,)  -> like ndarray
print("size   :", s.size)                  # 3
print("ndim   :", s.ndim)                  # 1
print("dtype  :", s.dtype)                 # int64
print("index  :", type(s.index).__name__, s.index.tolist())
print("values :", s.values)                # underlying ndarray
print("name   :", s.name)
print("axes   :", s.axes)                  # list of all index objects
print("empty  :", s.empty, "| has NaN:", s.hasnans)
print("str?   :", pd.api.types.is_string_dtype(s), "| numeric?",
      pd.api.types.is_numeric_dtype(s))

# %% [markdown]
# ### Indexing: `[]` vs `.loc` vs `.iloc`
# **What they do:** three ways to pull data out of a Series.
#
# | Accessor | Keyed by | Slice stop |
# |---|---|---|
# | `s[key]` | label (list/mask also works) | — |
# | `s.loc[label]` | **label** | **inclusive** |
# | `s.iloc[pos]` | **integer position** | **exclusive** (Python style) |
#
# > **Gotcha:** `s[0]` on a *string* index triggers a `FutureWarning` in
# > pandas 2.x (the positional fallback is being removed). `.loc[0]` is the
# > honest test — it raises `KeyError`. Never guess; use `.loc` / `.iloc`.

# %%
ser = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])

print(ser["b"])                      # single label            -> 20
print(ser[["a", "c"]])               # list of labels          -> subset
print(ser > 20)                      # boolean mask            -> bool Series
print(ser[ser > 20])                 # apply mask              -> filtered

try:
    ser.loc[0]                       # .loc wants a label, string index
except KeyError as e:
    print("KeyError:", e)            # no positional fallback in .loc

print(ser.loc["b"])                  # label                   -> 20
print(ser.loc["b":"d"])              # label slice, STOP INCLUSIVE -> 20 30 40
print(ser.iloc[0])                   # position                -> 10
print(ser.iloc[1:3])                 # positional slice, stop EXCLUSIVE -> 20 30
print(ser.iloc[[-1, 0]])             # positional fancy index  -> 40 10
print(ser.loc[["a", "d"]])           # label fancy index       -> 10 40

# %% [markdown]
# ### Series vs list vs dict
# **What it does:** a Series is a hybrid — list-like (ordered, positional ops)
# *and* dict-like (label lookup), plus NumPy-grade vectorised maths.

# %%
lst = [10, 20, 30]
dct = {"a": 10, "b": 20, "c": 30}
sr = pd.Series(dct)

print("list append      :", lst + [40])          # list: only concatenation
print("dict keys        :", list(dct))           # dict: unordered keys only
print("Series + 5       :", (sr + 5).tolist())   # Series: vectorised maths
print("Series > 15      :", (sr > 15).to_dict()) # vectorised comparison
print("Series dict view :", sr.to_dict() == dct) # dict-like round trip
print("len/iter        :", len(sr), list(sr), sr.iloc[0])  # list-like round trip

# %% [markdown]
# **Section notes — Series basics:**
#
# | Task | Code |
# |---|---|
# | create from list/dict/scalar | `pd.Series(data, index=...)` |
# | shape / size / dtype | `s.shape`, `s.size`, `s.dtype` |
# | labels / positions | `s.loc[...]` / `s.iloc[...]` |
# | label slice (inclusive) | `s.loc["b":"d"]` |
# | position slice (exclusive) | `s.iloc[1:3]` |
# | boolean filter | `s[s > 20]` |
# | Series ⇄ dict | `pd.Series(d)`, `s.to_dict()` |

# %% [markdown]
# ---
# ## 2. Series methods — the complete reference
#
# ▶ Important Series Methods: https://www.youtube.com/watch?v=80QpbZA38HA
# ▶ Session 16 - Pandas Series: https://www.youtube.com/watch?v=zCDVUyq8lkw

# %%
# shared Series for every example in this section
ser = pd.Series([12, 18, 7, 24, 15, 7, 30, np.nan],
                index=list("abcdefgh"), name="score")
print(ser)

# %% [markdown]
# ### `head(n=5)` / `tail(n=5)`
# **What they do:** first / last `n` rows — the "does my data look sane?" check.
# **Syntax:** `ser.head(n=5)`, `ser.tail(n=5)`
# **Parameters:** `n` — int — how many rows (negative `n` drops that many from the other end).
# **Returns:** Series (new object, original untouched).

# %%
print(ser.head(3))
print(ser.tail(2))
print(ser.head(-6))          # drops the last 6 -> first 2 rows

# %% [markdown]
# ### `describe(percentiles=None, include=None, exclude=None)`
# **What it does:** instant statistics — count/mean/std/min/quartiles/max for
# numeric data, count/unique/top/freq for objects.
# **Syntax:** `ser.describe(percentiles=[.25, .5, .75])`
# **Parameters:**
# - `percentiles` — list of floats in `[0, 1]` — extra quantiles (default `[.25, .5, .75]`)
# - `include` / `exclude` — dtypes to force in/out (Series: `"all"`)
# **Returns:** Series of summary stats (DataFrame for `DataFrame.describe`).

# %%
print(ser.describe())
print(ser.describe(percentiles=[0.1, 0.9]))
print(pd.Series(list("aabbc")).describe())        # object -> count/unique/top/freq

# %% [markdown]
# ### `info(verbose=None, buf=None, memory_usage=None)`
# **What it does:** dtypes, non-null counts and memory — the first call on any dataset.
# **Syntax:** `ser.info(memory_usage="deep")`
# **Parameters:** `memory_usage` — `True`/`"deep"` — add per-column byte counts; `buf` — file-like to print into.
# **Returns:** `None` (prints a report).

# %%
ser.info(memory_usage="deep")

# %% [markdown]
# ### `value_counts(normalize=False, dropna=True, sort=True)`
# **What it does:** frequency table of unique values — the categorical EDA staple.
# **Syntax:** `ser.value_counts(normalize=False, ascending=False, dropna=True)`
# **Parameters:**
# - `normalize` — bool — return fractions instead of counts
# - `dropna` — bool — exclude `NaN` (default `True`)
# - `sort` / `ascending` — order by frequency
# **Returns:** Series (index = values, values = counts).

# %%
c = pd.Series(["up", "down", "up", "up", None, "down"])
print(c.value_counts())                      # counts
print(c.value_counts(normalize=True).round(2))  # relative frequency
print(c.value_counts(dropna=False))          # keep NaN as a category
print(c.value_counts(ascending=True))        # least frequent first

# %% [markdown]
# ### `sort_values(axis=0, ascending=True, na_position="last", key=None)`
# **What it does:** sorts by *values*.
# **Syntax:** `ser.sort_values(ascending=True, na_position="last")`
# **Parameters:** `ascending` — bool — smallest first; `na_position` — `"last"`/`"first"`; `key` — function applied before sorting.
# **Returns:** sorted Series (new object; use `inplace=True` to modify).

# %%
print(ser.sort_values())                     # NaN always last by default
print(ser.sort_values(ascending=False, na_position="first"))
print(ser.sort_values(key=lambda s: s.abs()))

# %% [markdown]
# ### `sort_index(axis=0, ascending=True)`
# **What it does:** sorts by *labels* — restores index order after a groupby/merge.
# **Syntax:** `ser.sort_index(ascending=True)`
# **Returns:** Series ordered by index.

# %%
shuffled = ser.sort_values()
print(shuffled.sort_index())

# %% [markdown]
# ### `unique()` / `nunique(dropna=True)`
# **What they do:** distinct values (as an ndarray) and their count.
# **Syntax:** `ser.unique()`, `ser.nunique(dropna=True)`
# **Returns:** `unique()` → ndarray of values in first-seen order; `nunique()` → int.

# %%
print(c.unique())                # ndarray, includes NaN (object dtype)
print(c.nunique())               # 2 -> NaN not counted
print(c.nunique(dropna=False))   # 3

# %% [markdown]
# ### `isnull()` / `isna()` / `notna()` / `notnull()`
# **What they do:** element-wise missing-value masks (`isna` and `isnull` are aliases).
# **Syntax:** `ser.isna()`, `ser.notna()`
# **Returns:** boolean Series. `.sum()` on the mask counts missing values.

# %%
print(ser.isna())
print("missing:", int(ser.isna().sum()), "of", ser.size)
print(ser.notna())               # == ~ser.isna()
print(ser[ser.notna()])

# %% [markdown]
# ### `fillna(value=None, method=None, limit=None)`
# **What it does:** fills `NaN` — scalar, per-label dict, or directional propagation.
# **Syntax:** `ser.fillna(0)`, `ser.ffill(limit=None)`, `ser.bfill(limit=None)`
# **Parameters:**
# - `value` — scalar/dict/Series — replacement for `NaN`
# - `method` — **removed in pandas 3.x / deprecated since 2.1** → use `.ffill()` (forward) / `.bfill()` (backward)
# - `limit` — int — max consecutive `NaN` to fill
# **Returns:** filled Series.

# %%
print(ser.fillna(0))
print(ser.ffill())               # carry last value forward
print(ser.bfill())               # carry next value backward
print(ser.ffill(limit=1))        # fill at most 1 consecutive NaN
# NOTE: `ser.fillna(method="ffill")` still runs in pandas 2.x but raises a
# FutureWarning — prefer `.ffill()` / `.bfill()`.

# %% [markdown]
# ### `dropna(axis=0, how="any", thresh=None)`
# **What it does:** removes `NaN` entries.
# **Syntax:** `ser.dropna(how="any")`
# **Parameters:** `how` — `"any"`/`"all"`; `thresh` — keep only entries with at least this many non-null values (Series: mostly for DataFrames).
# **Returns:** Series without missing entries.

# %%
print(ser.dropna())
print("dropped:", ser.size - ser.dropna().size, "row(s)")

# %% [markdown]
# ### `map(arg, na_action=None)` / `apply(func, args=())` / `transform(func)`
# **What they do:** row-by-row transformation of a Series.
# - `map` — dict / Series / function, **vectorised-friendly**, element-wise.
# - `apply` — arbitrary function (aggregation *or* element-wise) — Python-level loop.
# - `transform` — must return something **the same shape** as the input (used inside `groupby`).
# **Syntax:** `ser.map(func)`, `ser.apply(func)`, `ser.transform(func)`
# **Returns:** Series with the same index.

# %%
sq = pd.Series([1, 2, 3, 4])
print(sq.map(lambda x: x ** 2))                 # element-wise
print(sq.map({1: "one", 2: "two", 3: "three", 4: "four"}))  # dict lookup
print(sq.apply(lambda x: x + 10))
print(sq.apply(lambda x: (x, x * 2)))           # value can be any object (tuple)
print(sq.transform("sqrt"))                     # pandas function name string
print(sq.transform(lambda x: x - x.mean()))     # same shape guaranteed

# %% [markdown]
# ### `astype(dtype, copy=True, errors="raise")`
# **What it does:** converts the Series dtype — memory savings or cleanup before maths.
# **Syntax:** `ser.astype("int8")` or `ser.astype({"a": "float32"})` (DataFrames)
# **Parameters:** `dtype` — dtype/spec; `errors` — `"raise"` (default) or `"ignore"`.
# **Returns:** Series with the new dtype.

# %%
small = pd.Series([1, 2, 3])
print(small.astype("int8").dtype, small.astype("float32").dtype)
print(small.astype(str).tolist())               # numbers -> strings
print(pd.Series(["1", "2"]).astype(int))        # string -> int
print(pd.Series([0, 1, 2]).astype(bool).tolist())

# %% [markdown]
# ### `between(left, right, inclusive="both")`
# **What it does:** range membership mask — sugar for `((s >= left) & (s <= right))`.
# **Syntax:** `ser.between(left, right, inclusive="both")`
# **Parameters:** `inclusive` — `"both"`, `"neither"`, `"left"`, `"right"`.
# **Returns:** boolean Series.

# %%
vals = pd.Series([1, 5, 9, 12, 15])
print(vals.between(5, 12))
print(vals.between(5, 12, inclusive="left"))
print(vals[vals.between(9, 15)])

# %% [markdown]
# ### `idxmax()` / `idxmin()`
# **What they do:** **label** of the maximum / minimum value (position of the extremum, not the value).
# **Syntax:** `ser.idxmax(skipna=True)`
# **Returns:** index label. With ties → the first occurrence.

# %%
print(ser.idxmax(), ser.idxmin())               # ignores NaN by default
print(ser.loc[ser.idxmax()])

# %% [markdown]
# ### Aggregations: `max/min/mean/median/std/var/sum/count`
# **What they do:** reduce the whole Series to one number (all `NaN`-safe by skipping `NaN`).
# **Syntax:** `ser.mean()`, `ser.std(ddof=1)`, ...
# **Parameters:** `ddof` — delta degrees of freedom for `std`/`var` (pandas default **1**, sample estimator).
# **Returns:** scalar (`count` returns int, `mean` returns float).

# %%
print("sum   :", ser.sum())
print("mean  :", round(ser.mean(), 3))
print("median:", ser.median())
print("std   :", round(ser.std(), 3), "| var:", round(ser.var(), 3))
print("min/max:", ser.min(), ser.max())
print("count :", ser.count(), "(non-null)")     # size - isna().sum()
print("sem / quantile:", round(ser.sem(), 3), ser.quantile(0.5))

# %% [markdown]
# ### `rank(method="average", ascending=True, pct=False)`
# **What it does:** assigns ranks (1 = lowest) instead of sorting.
# **Syntax:** `ser.rank(method="average", ascending=True)`
# **Parameters:** `method` — `"average"` (default), `"min"`, `"max"`, `"first"`, `"dense"` — tie-breaking rule.
# **Returns:** Series of ranks (float).

# %%
ties = pd.Series([10, 20, 20, 30])
print(ties.rank())                              # average -> 1, 2.5, 2.5, 4
print(ties.rank(method="min"))                  # 1, 2, 2, 4
print(ties.rank(method="dense"))                # 1, 2, 2, 3
print(ties.rank(pct=True).round(2))             # percentile ranks

# %% [markdown]
# ### `shift(periods=1, freq=None, fill_value=None)`
# **What it does:** shifts values down (lag) or up (lead) — the time-series building block.
# **Syntax:** `ser.shift(periods=1)`
# **Parameters:** `periods` — int — positive = shift down (lag), negative = up (lead); `freq` — shift the *index* by a date offset instead of values.
# **Returns:** Series with `NaN` introduced in the gap.

# %%
s5 = pd.Series([10, 20, 30, 40])
print("lag  :", s5.shift(1).tolist())           # [NaN, 10, 20, 30]
print("lead :", s5.shift(-1).tolist())          # [20, 30, 40, NaN]
print("diff :", s5.sub(s5.shift(1)).tolist())   # period-over-period change
print("pct  :", s5.pct_change().round(2).tolist())
# NOTE: when freq is given (DateTimeIndex) the index shifts too:
ts = pd.Series([1, 2], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
print(ts.shift(1, freq="D"))

# %% [markdown]
# ### `pct_change(periods=1, fill_method=None)`
# **What it does:** fractional change between consecutive elements: `(x[t] - x[t-p]) / x[t-p]`.
# **Syntax:** `ser.pct_change(periods=1)`
# **Returns:** Series; first `periods` entries are `NaN`.

# %%
prices = pd.Series([100.0, 110.0, 99.0, 108.9])
print(prices.pct_change().round(3))
print(prices.pct_change(periods=2).round(3))
# NOTE: percentages read better with .mul(100): (prices.pct_change() * 100).round(1)

# %% [markdown]
# ### `cumsum()` / `cumprod()` / `cummax()` / `cummin()`
# **What they do:** running totals / products / extrema.
# **Syntax:** `ser.cumsum()`
# **Returns:** Series of the same length.

# %%
c4 = pd.Series([1, 2, 3, 4])
print("cumsum :", c4.cumsum().tolist())         # [1, 3, 6, 10]
print("cumprod:", c4.cumprod().tolist())        # [1, 2, 6, 24]
print("cummax :", c4.cummax().tolist())         # [1, 2, 3, 4]

# %% [markdown]
# ### `round(decimals=0)`
# **What it does:** rounds to `decimals` places (banker's rounding via numpy).
# **Syntax:** `ser.round(2)`
# **Returns:** Series of floats rounded.

# %%
print(pd.Series([1.234, 5.678, -2.999]).round(2))

# %% [markdown]
# ### `replace(to_replace=None, value=None, regex=False)`
# **What it does:** swaps values by exact match or regex.
# **Syntax:** `ser.replace(to_replace, value, regex=False)`
# **Parameters:** `to_replace` — value/dict/regex; `value` — replacement; `regex` — treat `to_replace` as a regex.
# **Returns:** Series with replacements applied.

# %%
sw = pd.Series(["up", "down", "UP!", "n/a"])
print(sw.replace({"UP!": "up"}))
print(sw.replace(r"n/a", np.nan, regex=True))
print(pd.Series(["2020-Jan", "2021-Feb"]).replace(r"^\d{4}-", "", regex=True))

# %% [markdown]
# ### `explode(ignore_index=False)`
# **What it does:** one row per list element — turns list-valued cells into tidy rows.
# **Syntax:** `ser.explode(ignore_index=False)`
# **Returns:** Series (index repeated; empty lists become `NaN`).

# %%
print(pd.Series([[1, 2], [3], [], [4, 5]]).explode())
print(pd.Series([[1, 2], [3]]).explode(ignore_index=True))

# %% [markdown]
# ### `sample(n=None, frac=None, replace=False, random_state=None)`
# **What it does:** random subset — always pass `random_state` for reproducibility.
# **Syntax:** `ser.sample(n=3, random_state=42)`
# **Parameters:** `n` — count; `frac` — fraction of rows; `replace` — sampling with replacement; `random_state` — int/Generator for deterministic output.
# **Returns:** sampled Series.

# %%
print(ser.sample(3, random_state=42))           # deterministic every run
print(ser.sample(frac=0.5, random_state=0))
print(ser.sample(20, replace=True, random_state=0).size)   # n > len needs replace

# %% [markdown]
# ### `where(cond, other=np.nan)` / `mask(cond, other=np.nan)`
# **What they do:** keep values where `cond` is True (`where`) — or replace them where `cond` is True (`mask` — the logical inverse of `where`).
# **Syntax:** `ser.where(cond, other)`
# **Returns:** Series of the same shape.

# %%
w = pd.Series([1, 2, 3, 4])
print(w.where(w > 1))                           # values <= 1 become NaN
print(w.mask(w > 1))                            # values > 1 become NaN
print(w.where(w > 1, 0))                        # replace the *other* side with 0
print(w.mask(w > 1, -1))

# %% [markdown]
# > **Gotcha — label slicing is inclusive:**
# > `s["b":"d"]` / `s.loc["b":"d"]` **includes** `"d"` (pandas follows
# > `iloc`-style exclusive slicing only for positions). Mixing label and
# > positional semantics is the #1 beginner bug.

# %%
sliced = pd.Series([10, 20, 30, 40], index=list("abcd"))
print(sliced.loc["b":"d"].tolist())             # [20, 30, 40]  stop INCLUSIVE
print(sliced.iloc[1:3].tolist())                # [20, 30]      stop EXCLUSIVE

# %% [markdown]
# ### Arithmetic & index alignment
# **What it does:** operators (`+ - * / // % **` and `add/sub/mul/div/...`) are
# **element-wise on labels**: indexes are unioned, and any label present on only
# one side produces `NaN`.
#
# | Situation | Result |
# |---|---|
# | same labels | aligned element-wise ops |
# | label only on one side | `NaN` (missing label) |
# | scalar operand | broadcast to every label |
# | want no `NaN` | `a.add(b, fill_value=0)` |
# | opposite order | commutative ops fine; division isn't — use `a.div(b)` |

# %%
a = pd.Series([10, 20, 30], index=["x", "y", "z"])
b = pd.Series([100, 200, 300], index=["y", "z", "w"])   # different labels!

print("a + b (union index, misaligned -> NaN):")
print(a + b)
print(a.add(b, fill_value=0))                  # treat missing as 0
print(a * 2)                                   # scalar broadcast
print(a.sub(a.mean()).round(2))                # vectorised de-meaning
print((a > 15).sum())                          # boolean sums as 0/1

# %% [markdown]
# > **Gotcha:** `a + b` aligns on **labels**, `np.array(a) + np.array(b)`
# > aligns on **positions** — mixing the two mental models silently gives wrong
# > answers whenever the indexes differ.

# %% [markdown]
# ### `append()` — removed in pandas 2.x
# **What happened:** `Series.append(other)` and `DataFrame.append(...)` were
# deprecated in 1.4 and **removed in 2.0**. They also copied data and were
# misused as a "push one row" API.
#
# ```python
# # pandas 1.x                       # pandas 2.x
# s1.append(s2)                      pd.concat([s1, s2])
# df.append(row_dict, ignore_index=True)  pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
# ```
# Use `pd.concat` (see §7) — it is the one supported stacking API.

# %%
try:
    ser.append(ser)
except AttributeError as e:
    print("AttributeError:", e)

# %% [markdown]
# **Section notes — Series methods:**
#
# | Group | Methods |
# |---|---|
# | Peek | `head` `tail` `describe` `info` `sample` |
# | Frequencies | `value_counts` `unique` `nunique` |
# | Missing | `isna/isnull` `notna` `fillna` `ffill` `bfill` `dropna` |
# | Order | `sort_values` `sort_index` |
# | Transform | `map` `apply` `transform` `astype` `replace` `explode` |
# | Stats | `mean` `median` `std` `var` `sum` `min` `max` `quantile` `idxmax` `idxmin` `rank` `count` |
# | Time-ish | `shift` `pct_change` `cumsum` `cumprod` `round` `between` `where` `mask` |
# | Removed | `append` → `pd.concat` |

# %% [markdown]
# ---
# ## 3. DataFrame — creation, attributes & columns
#
# ▶ Session 17 - Pandas DataFrame: https://www.youtube.com/watch?v=OG31yhRQxPI

# %% [markdown]
# ### `pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)`
# **What it does:** the labelled 2-D table — a dict of aligned Series sharing one index.
# **Syntax:** `pd.DataFrame(data, index=None, columns=None, dtype=None)`
# **Parameters:**
# - `data` — dict of lists/Series, list of dicts, 2-D ndarray, another DataFrame, scalar
# - `index` / `columns` — row labels / column labels (must cover the data's extent)
# - `dtype` — force one dtype for all columns (rarely what you want on real data)
# **Returns:** `pandas.DataFrame`

# %%
# 1) dict of lists  -> keys become columns (must be equal length)
df1 = pd.DataFrame({"name": ["Rahul", "Sita", "Amir"],
                    "marks": [88, 92, 79],
                    "city": ["Delhi", "Pune", "Patna"]})
print(df1, "\n")

# 2) list of dicts  -> each dict is a row (union of keys, missing -> NaN)
df2 = pd.DataFrame([{"name": "Rahul", "marks": 88},
                    {"name": "Sita", "city": "Pune"}])
print(df2, "\n")

# 3) 2-D ndarray + explicit columns
df3 = pd.DataFrame(np.arange(6).reshape(3, 2), columns=["a", "b"])
print(df3, "\n")

# 4) dict of Series -> aligned on the union of indexes (missing -> NaN)
df4 = pd.DataFrame({"x": pd.Series([1, 2], index=["r1", "r2"]),
                    "y": pd.Series([30, 40], index=["r2", "r3"])})
print(df4, "\n")

# 5) scalar + index/columns -> broadcast
print(pd.DataFrame(0, index=["r1", "r2"], columns=["flag"]))

# %% [markdown]
# ### Reading tabular data — `read_csv(file_or_buffer, ...)`
# **What it does:** the most-used IO function; accepts a path, URL *or an
# in-memory buffer* — so a CSV string works with `io.StringIO` (offline-friendly).
# **Syntax:** `pd.read_csv(io.StringIO(text), sep=",", header=0, ...)` — full option list in §12.
# **Related:** `read_excel` (needs `openpyxl`, installed here), `read_json`, `read_parquet`, `read_sql`.
# **Returns:** DataFrame.

# %%
csv_text = """student,marks,grade
Rahul,88,A
Sita,92,A
Amir,79,B
"""
df = pd.read_csv(io.StringIO(csv_text))
print(df)
print(df.dtypes)

# DataFrame -> file round trip (writes locally, no network)
df.to_csv("/tmp/dsmp_04_demo.csv", index=False)
print(pd.read_csv("/tmp/dsmp_04_demo.csv").head(2))

# %% [markdown]
# ### DataFrame attributes
# **What they do:** shape/dtype introspection — `info()` prints them, these read them.

# %%
df = pd.DataFrame({"name": ["Rahul", "Sita", "Amir"],
                   "marks": [88, 92, 79],
                   "city": ["Delhi", "Pune", "Patna"]})

print("shape  :", df.shape)             # (3, 2) -> (rows, columns)
print("columns:", df.columns.tolist())  # Index object of labels
print("dtypes :\n", df.dtypes)
print("index  :", df.index.tolist())
print("ndim   :", df.ndim)              # 2
print("size   :", df.size)              # rows * cols
print("values :\n", df.values)          # one dtype for everything -> object
print("empty? :", df.empty)

# %% [markdown]
# ### Column selection & assignment
# **What they do:** four interchangeable read styles; `.assign()` is the
# **functional** (non-mutating) way to add derived columns.
#
# | Syntax | Returns |
# |---|---|
# | `df["col"]` | Series |
# | `df[["a", "b"]]` | DataFrame (copy) |
# | `df.col` | Series (fails on spaces / method names) |
# | `df.assign(new=...)` | new DataFrame with `new` appended |

# %%
print(df["marks"])                      # Series
print(df[["name", "city"]])             # sub-DataFrame
print(df.marks)                         # attribute access (fragile!)
# NOTE: df["col"] is preferred — df.query style names with spaces need []

df2 = df.assign(passed=df.marks >= 80,
                marks_pct=df.marks / 100)
print(df2)

df["grade"] = np.where(df.marks >= 80, "A", "B")   # in-place assignment
print(df)

# %% [markdown]
# ### `drop(labels, axis)`, `insert(loc, column, value)`, `rename(mapper, axis)`
# **What they do:** structural edits — remove / place / rename columns (or rows).
# **Syntax:** `df.drop(columns=["c"])` · `df.insert(0, "c", v)` · `df.rename(columns={"old": "new"})`
# **Parameters:**
# - `labels` + `axis` — `axis=1` (or `columns=`) targets columns, `axis=0` targets rows
# - `inplace` — mutate instead of returning (discouraged — hides data flow)
# - `insert` has **no** `inplace`; it always mutates and returns `None`
# - `rename` accepts `index=`/`columns=` dicts or `str`-callable mappers
# **Returns:** new DataFrame (drop/rename), `None` (insert).

# %%
base = df.drop(columns="grade")
print(base.columns.tolist())

tmp = base.copy()
tmp.insert(1, "rank", [1, 2, 3])        # position 0-based among columns
print(tmp)

print(base.rename(columns={"marks": "score", "city": "location"}))
print(base.rename(str.upper, axis=1).columns.tolist())
# NOTE: dropping rows -> df.drop(index=[0]) or df.drop([0], axis=0)

# %% [markdown]
# ### `set_index(keys, drop=True)` / `reset_index(level=None, drop=False)`
# **What they do:** move columns ⇄ index — the door to groupby-on-index and MultiIndex.
# **Syntax:** `df.set_index("name")`, `df.reset_index()`
# **Parameters:** `drop` — keep the column as a column too? (`set_index`); `drop` — discard the index instead of turning it into a column? (`reset_index`)
# **Returns:** new DataFrame.

# %%
ix = df.set_index("name")
print(ix)
print(ix.loc["Sita"])
print(ix.reset_index())                 # index becomes a column again
print(df.set_index(["city", "name"]))   # multi-level — see §8
# TIP: pd.MultiIndex.from_arrays / .from_tuples build index objects directly
print(pd.MultiIndex.from_arrays([["a", "b"], [1, 2]], names=["k", "n"]).tolist())

# %% [markdown]
# **Section notes — DataFrame basics:**
#
# | Task | Code |
# |---|---|
# | create | `pd.DataFrame(dict_of_lists)` |
# | read CSV string | `pd.read_csv(io.StringIO(txt))` |
# | shape / dtypes | `df.shape`, `df.dtypes`, `df.info()` |
# | pick columns | `df["c"]`, `df[["a","b"]]`, `df.c` |
# | add derived col | `df.assign(new=...)` or `df["new"] = ...` |
# | drop/insert/rename | `drop(columns=)`, `insert()`, `rename(columns=)` |
# | index ⇄ column | `set_index()`, `reset_index()` |

# %% [markdown]
# ---
# ## 4. DataFrame indexing — loc, iloc, masks, query
#
# ▶ Session 17 - Pandas DataFrame: https://www.youtube.com/watch?v=OG31yhRQxPI
# ▶ Session 18 - Important DataFrame Methods: https://www.youtube.com/watch?v=zTa4MIrGTIE

# %%
df = pd.DataFrame({"a": [10, 20, 30, 40, 50],
                   "b": ["x", "y", "x", "z", "y"],
                   "c": [1.5, 2.5, 3.5, 4.5, 5.5]},
                  index=["r0", "r1", "r2", "r3", "r4"])
df

# %% [markdown]
# ### `df.loc[row_labels, col_labels]` vs `df.iloc[row_positions, col_positions]`
# **What they do:** the two explicit accessors. Both take `[rows, cols]` with
# slices, lists and masks; `loc` is **label**-based (stop inclusive), `iloc` is
# **position**-based (stop exclusive).
# **Syntax:** `df.loc["r0", "a"]`, `df.loc["r0":"r2", ["a","b"]]`, `df.iloc[0:2, 1:]`
# **Returns:** scalar / Series / DataFrame depending on how much you select.

# %%
print(df.loc["r1", "b"])                  # scalar at (label, label)
print(df.loc["r0":"r2", ["a", "c"]])      # label slice (inclusive) x columns
print(df.loc[:, "a"])                     # all rows, one column
print(df.iloc[0, 1])                      # scalar at (pos, pos)
print(df.iloc[0:2, 1:])                   # positional slice (exclusive)
print(df.iloc[[0, 3]][["a"]])             # fancy rows x fancy columns
print(df.loc[df["a"] > 25, "b"])          # mask + column in one shot

# %% [markdown]
# ### Boolean masking & `query()`
# **What they do:** filter rows by condition — the pandas idiom for `WHERE`.
#
# - `df[mask]` / `df.loc[mask]` — vectorised, no strings to typo
# - `df.query("a > 25 and b == 'x'")` — SQL-like **string** expression (readable, uses `@var` for Python names)

# %%
mask = (df["a"] > 20) & (df["b"] != "z")       # & | ~ element-wise, not and/or!
print(df[mask])
print(df.loc[mask, ["a", "c"]])

print(df.query("a > 25 and b == 'y'"))          # same idea, SQL syntax
threshold = 30
print(df.query("a > @threshold"))               # @ injects a Python variable
print(df.query("b in ['x', 'y']"))

# %% [markdown]
# ### `df.at[label, col]` / `df.iat[row_pos, col_pos]`
# **What they do:** **scalar** fast paths — much quicker than `.loc`/`.iloc`
# for single-cell reads (and `.at`/`.iat` also *assign*: `df.at["r0","a"] = 99`).
# **Syntax:** `df.at["r0", "a"]`, `df.iat[0, 0]`
# **Returns:** the scalar.

# %%
print(df.at["r0", "a"], df.iat[0, 0])
d2 = df.copy()
d2.at["r0", "a"] = 99
d2.iat[1, 2] = 9.9
print(d2.head(2))

# %% [markdown]
# > **Gotcha — chained assignment (`SettingWithCopyWarning`):**
# > `df[mask]["col"] = v` writes to a **temporary copy** — your change may be
# > lost (and pandas warns). Always assign through **one** accessor:
# > `df.loc[mask, "col"] = v`. If you *slice first*, `.copy()` the result to
# > own the memory.

# %%
work = df.copy()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    work[work["a"] > 30]["b"] = "!"        # chained: writes to a copy
    print("warnings raised:", [type(w.message).__name__ for w in caught])

# the fix — single accessor writes in place:
work.loc[work["a"] > 30, "b"] = "!"
print(work["b"].tolist())

# slices that you keep must be explicit copies
subset = df.loc[df["a"] > 20].copy()       # own the memory -> no warning later
subset["a"] = subset["a"] * 2
print(subset["a"].tolist())

# %% [markdown]
# **Section notes — indexing:**
#
# | Need | Code |
# |---|---|
# | label cell / slice | `df.loc["r0", "a"]`, `df.loc["r0":"r2", cols]` |
# | positional cell / slice | `df.iat[0, 0]`, `df.iloc[0:2, 1:]` |
# | filter rows | `df[df.a > 25]`, `df.query("a > 25")` |
# | filter + pick columns | `df.loc[mask, ["a","c"]]` |
# | assign in place | `df.loc[mask, "col"] = v` |
# | own a slice | `df[...].copy()` |

# %% [markdown]
# ---
# ## 5. Core DataFrame methods — full reference
#
# ▶ Session 18 - Important DataFrame Methods: https://www.youtube.com/watch?v=zTa4MIrGTIE

# %%
df = pd.DataFrame({"name": ["Rahul", "Sita", "Amir", "Rahul", "Sita", None],
                   "city": ["Delhi", "Pune", "Patna", "Delhi", "Pune", "Patna"],
                   "marks": [88, np.nan, 79, 88, 92, 79],
                   "year": [2021, 2021, 2022, 2021, 2021, 2022]})
df

# %% [markdown]
# ### `head(n)`, `tail(n)`, `sample(n, random_state)`, `describe(include=None)`, `info()`, `memory_usage(deep)`
# **What they do:** the inspection toolkit.
# **Syntax:** `df.head()`, `df.sample(3, random_state=42)`, `df.describe(include="all")`
# **Parameters:**
# - `describe(include=)` — `"all"` forces stats for object columns too (default: numeric only)
# - `sample(n, random_state=)` — `random_state` makes it reproducible
# - `memory_usage(deep=True)` — exact bytes including Python objects (strings)
# **Returns:** DataFrame/Series for `head/tail/sample/describe`; `None` for `info`.

# %%
print(df.head(2))
print(df.sample(2, random_state=42))          # same rows every run
print(df.describe())                          # numeric columns only
print(df.describe(include="all"))             # + count/unique/top for objects
print("bytes deep:", int(df.memory_usage(deep=True).sum()))
df.info(memory_usage="deep")                  # dtypes + null counts

# %% [markdown]
# ### `sort_values(by, axis=0, ascending=True, na_position="last")` / `sort_index(axis=0)`
# **What they do:** order rows by column values (or several, per-column direction) / by index.
# **Syntax:** `df.sort_values("marks", ascending=False)` · `df.sort_values(["city","marks"], ascending=[True, False])`
# **Parameters:** `by` — column label or list; `ascending` — bool or list of bools; `na_position` — `"last"`/`"first"`.
# **Returns:** ordered DataFrame.

# %%
print(df.sort_values("marks", na_position="first"))
print(df.sort_values(["city", "marks"], ascending=[True, False]))
print(df.sort_values("marks", ascending=False).head(3))
print(df.sort_index(axis=1).columns.tolist())   # order the COLUMNS alphabetically

# %% [markdown]
# ### `duplicated(subset=None, keep="first")` / `drop_duplicates(subset=None, keep="first")`
# **What they do:** flag / remove repeated rows.
# **Syntax:** `df.duplicated(subset=["name","city"])`, `df.drop_duplicates(...)`
# **Parameters:** `subset` — columns defining a "row identity"; `keep` — `"first"`/`"last"`/`False` (all dups marked).
# **Returns:** boolean Series (duplicated) / DataFrame (drop).

# %%
print(df.duplicated())                          # full-row duplicates
print(df.duplicated(subset=["name", "city"]))
print("dup rows:", int(df.duplicated(subset=["name", "city"]).sum()))
print(df.drop_duplicates(subset=["name", "city"], keep="last"))

# %% [markdown]
# ### `dtypes` & `astype()` conversion
# **What they do:** per-column dtypes; `astype` converts several columns at once with a **dict**.
# **Syntax:** `df.astype({"year": "int16", "city": "category"})`
# **Returns:** DataFrame with new dtypes.

# %%
print(df.dtypes)
fixed = df.astype({"year": "int16", "marks": "float32"})
print(fixed.dtypes)
print(df["city"].astype("category").dtype)      # low-cardinality -> category

# %% [markdown]
# ### `dropna(axis=0, how="any", subset=None, thresh=None)`
# **What it does:** removes rows/columns containing `NaN`.
# **Parameters:**
# - `axis` — `0` rows (default), `1` columns
# - `how` — `"any"` (default) / `"all"`
# - `subset` — only consider these columns
# - `thresh` — keep rows with at least `thresh` non-null values
# **Returns:** cleaned DataFrame.

# %%
print(df.dropna())                                  # any NaN in the row
print(df.dropna(how="all"))                         # completely empty rows only
print(df.dropna(subset=["marks"]))                  # NaN only judged on marks
print(df.dropna(thresh=3))                          # >= 3 non-null values
print(df.dropna(axis=1).columns.tolist())           # drop whole columns

# %% [markdown]
# ### `fillna(value=None, method=None, limit=None, axis=0)`
# **What it does:** fills `NaN` — per column via dict, or by propagation.
# **Parameters:** `value` — scalar or `{col: fill}`; `method` — **deprecated** (use `.ffill()` / `.bfill()`); `limit` — max consecutive fills.
# **Returns:** filled DataFrame.

# %%
print(df.fillna({"marks": 0}))            # per-column dict
# df.fillna(method="ffill")              # deprecated in 2.x -> FutureWarning
print(df.ffill())                         # <- the pandas 2.x way
print(df.bfill(limit=1))

# %% [markdown]
# ### `replace(to_replace=None, value=None, regex=False, method="linear")`
# **What it does:** value-level cleaning; with `regex=True` patterns apply per column.
# **Syntax:** `df.replace({"Delhi": "New Delhi"})`, `df.replace(r"\$", "", regex=True)`
# **Returns:** DataFrame with values swapped.

# %%
print(df.replace({"Delhi": "New Delhi"}))
print(pd.DataFrame({"amt": ["$5", "$6"]}).replace(r"[$]", "", regex=True))

# %% [markdown]
# ### `apply(func, axis=0, raw=False, result_type=None)`
# **What it does:** apply a function along an axis — `axis=0` → each **column** (a Series), `axis=1` → each **row**.
# **Syntax:** `df.apply(func, axis=1)`
# **Parameters:** `raw` — pass ndarray instead of Series (speed); `result_type` — `"reduce"`/`"expand"`/`"broadcast"` for `axis=1`.
# **Returns:** Series (if the function returns a scalar) or DataFrame.
# **Note:** for *element-wise* use `df.map(...)` (pandas ≥ 2.1). `df.applymap(...)`
# still exists in 2.x but is **deprecated** — `DataFrame.map` replaced it.

# %%
num = df[["marks", "year"]]
print(num.apply(lambda col: col.max()))            # axis=0: one value per column
print(num.apply(lambda row: row.marks - row.year, axis=1))   # per row
print(num.apply(lambda r: pd.Series([r.marks, r.year * 2]), axis=1, result_type="expand").head(2))
# NOTE: na_action="ignore" keeps NaN cells out of the function (they stay NaN)
print(df[["name", "city"]].map(str.upper, na_action="ignore"))   # was applymap
print(df[["marks"]].apply(lambda c: c - c.mean()).round(1))

# %% [markdown]
# ### `value_counts()` on columns, `nunique()`, `idxmax()`
# **What they do:** quick per-column summaries without a full `describe`.

# %%
print(df["city"].value_counts())
print(df[["name", "city"]].nunique())
print(df["marks"].idxmax())                        # label of max (ignores NaN)
print(df["marks"].idxmin())

# %% [markdown]
# ### `corr(method="pearson", numeric_only=False)` / `cov(numeric_only=False)`
# **What they do:** numeric column-vs-column correlation / covariance matrices.
# **Parameters:** `method` — `"pearson"` (linear), `"spearman"` (rank), `"kendall"`; `numeric_only=True` is **required** in pandas 2.x when the frame has text columns.
# **Returns:** square DataFrame (columns × columns).

# %%
nums = pd.DataFrame({"study": [1, 2, 3, 4, 5],
                     "sleep": [8, 7, 6, 7, 5],
                     "marks": [55, 60, 65, 75, 85]})
print(nums.corr(numeric_only=True).round(2))
print(nums.corr(numeric_only=True, method="spearman").round(2))
print(nums.cov(numeric_only=True).round(1))

# %% [markdown]
# ### `T` (transpose), `melt(id_vars, value_vars)`, `stack()` / `unstack()`, `rename(columns=...)`, `set_axis(labels, axis)`
# **What they do:** the reshaping toolbox.
# - `df.T` — swap rows/columns (dtypes unify to `object` if mixed)
# - `melt` — **wide → long** (un-pivot): id columns stay, value columns fold into `(var, value)` pairs
# - `stack` / `unstack` — move the innermost index level ↔ columns (use `future_stack=True` to skip the 2.x deprecation)
# - `rename(columns=str.lower)` — bulk column renaming with a callable
# - `set_axis` — replace all labels at once (length must match)

# %%
wide = pd.DataFrame({"id": [1, 2],
                     "math": [80, 90],
                     "sci": [70, 95]})
print(wide.T)
long = pd.melt(wide, id_vars=["id"], var_name="subject", value_name="score")
print(long)

tall = long.set_index(["id", "subject"]).sort_index()
print(tall.unstack())                              # long -> wide again
print(tall.stack(future_stack=True).head(3))       # wide-ish -> long (new impl)
print(wide.rename(columns=str.lower).columns.tolist())
print(wide.set_axis(["ID", "MATH", "SCI"], axis=1).columns.tolist())

# %% [markdown]
# ### Updating cells via `.loc`
# **What it does:** conditional edits — the correct way to modify existing values.
# **Syntax:** `df.loc[mask, "col"] = new_value`

# %%
upd = df.copy()
upd.loc[upd["marks"] >= 88, "city"] = "Metro"
upd.loc[0, "year"] = 2020
print(upd)

# %% [markdown]
# **Section notes — DataFrame methods:**
#
# | Task | Code |
# |---|---|
# | peek | `head` `tail` `sample(n, random_state)` `describe(include="all")` `info` |
# | order | `sort_values(by, ascending=[...])` `sort_index(axis=1)` |
# | duplicates | `duplicated(subset)` / `drop_duplicates(subset, keep)` |
# | missing | `dropna(how/subset/thresh)` / `fillna(dict)` / `.ffill()` `.bfill()` |
# | replace | `replace(value_dict)` / `replace(regex, regex=True)` |
# | apply | `apply(axis=0/1)` · element-wise → `df.map` (`applymap` deprecated) |
# | stats | `corr(numeric_only=True, method=)` `cov` `nunique` `idxmax` `value_counts` |
# | reshape | `T` `melt` `stack` `unstack` `rename(columns=str.lower)` `set_axis` |
# | edit | `df.loc[mask, "col"] = v` |

# %% [markdown]
# ---
# ## 6. GroupBy — split-apply-combine, pivot_table & crosstab
#
# ▶ Session 19 - GroupBy Object in Pandas: https://www.youtube.com/watch?v=LPBjF4_gZnI
# ▶ Session 22 - Vectorized String Ops / DateTime / Pivot Table: https://www.youtube.com/watch?v=UE6DmRQJ2w8

# %% [markdown]
# ### The split-apply-combine model
# **What it does:** every `groupby` is Hadley Wickham's three-step recipe:
#
# ```text
#   ┌──────────────────┐   split by key    ┌─────┐ ┌─────┐ ┌─────┐
#  # │  orders DataFrame │ ───────────────► │  G1 │ │  G2 │ │  G3 │
#   └──────────────────┘                   └──┬──┘ └──┬──┘ └──┬──┘
#                                    apply fn │       │       │
#                                             ▼       ▼       ▼
#   ┌─────┐ ┌─────┐ ┌─────┐    combine   ┌────────────────────────┐
#   │ agg │ │ agg │ │ agg │  ─────────► │ one row per group      │
#   └─────┘ └─────┘ └─────┘   (concat)  └────────────────────────┘
# ```
#
# - **split** — `df.groupby("k")` builds a `DataFrameGroupBy` (lazy: nothing computed yet)
# - **apply** — an aggregation (`mean`), a transformation (`transform`, same shape) or a filter (`filter`, whole rows)
# - **combine** — pandas glue the per-group results back along the index

# %%
gdf = pd.DataFrame({"dept": ["sales", "sales", "hr", "hr", "tech", "tech"],
                    "emp": ["a", "b", "c", "d", "e", "f"],
                    "salary": [700, 800, 500, 550, 900, 950],
                    "age": [30, 35, 28, 41, 33, 29]})

# ### `DataFrame.groupby(by, axis=0, level=None, sort=True, observed=False, dropna=True)`
# **What it does:** returns a groupby *object* — call an aggregation to materialise it.
# **Parameters:**
# - `by` — column label, list of labels, Series (aligned), function, or `level=` for index levels
# - `observed` — for categoricals: `True` = only combinations that exist (pandas 2.x will default to `True`)
# - `dropna` — treat `NaN` keys as their own group (`False` keeps them)
# **Returns:** `DataFrameGroupBy` / `SeriesGroupBy` (lazy, hashable, reusable).

print(gdf.groupby("dept").mean(numeric_only=True))
print(gdf.groupby("dept")["salary"].mean())        # SeriesGroupBy
print(gdf.groupby(["dept", "age"]).size().head(3)) # multi-key

# %% [markdown]
# ### `.agg(func, *funcs, **named)` — the Swiss-army aggregation
# **What it does:** run *several* different reductions at once, per column.
# **Syntax:**
# - `g.agg("mean")` / `g.agg(["sum", "mean"])` — same funcs for all columns
# - `g.agg({"salary": "sum", "age": "mean"})` — per-column mapping
# - `g.agg(total=("salary", "sum"), avg=("age", "mean"))` — **named** aggregation → clean column names
# **Returns:** DataFrame indexed by the group keys.

# %%
print(gdf.groupby("dept").agg(["min", "max"]))                    # all numeric cols
print(gdf.groupby("dept").agg({"salary": ["sum", "max"],
                               "age": "mean"}))                   # per-column dict
print(gdf.groupby("dept").agg(total=("salary", "sum"),
                              headcount=("emp", "count"),
                              avg_age=("age", "mean")))           # named agg
print(gdf.groupby("dept").salary.agg("sum", "total"))             # single Series.agg

# %% [markdown]
# ### `.transform(func)` — same-shape results (broadcast back)
# **What it does:** applies a function per group but returns a result with the
# **original index/shape** — perfect for group-wise normalisation and "add a
# group aggregate as a column".
# **Returns:** Series/DataFrame aligned to the input rows.

# %%
print(gdf.assign(dept_mean=gdf.groupby("dept")["salary"].transform("mean")))
gdf["pct_in_dept"] = (gdf["salary"] / gdf.groupby("dept")["salary"].transform("sum")).round(3)
print(gdf[["dept", "salary", "pct_in_dept"]])
print(gdf.groupby("dept")["salary"].transform(lambda s: s - s.mean()).round(1))

# %% [markdown]
# ### `.filter(func)` and `.apply(func)`
# - `filter` — keep **whole groups** where `func(group)` is truthy (row-level conditions that need group context)
# - `apply` — arbitrary per-group function; result is concatenated along the group keys (scalar → Series, DataFrame → DataFrame)
#
# **Returns:** DataFrame for both (filter = subset of rows).

# %%
print(gdf.groupby("dept").filter(lambda g: g["salary"].mean() > 700))
print(gdf.groupby("dept").apply(lambda g: g.nlargest(1, "salary"),
                                include_groups=False))
print(gdf.groupby("dept").apply(lambda g: g["salary"].max(),
                                include_groups=False))

# %% [markdown]
# ### Grouping by an index level or a Series
# **What they do:** keys don't have to be columns — use `level=` after
# `set_index`, or pass any aligned Series (great for a derived key).

# %%
print(gdf.set_index("dept").groupby(level=0)["salary"].mean())
# aligned Series as key (here: a bucketed column)
gdf["band"] = pd.cut(gdf["salary"], bins=[0, 800, 1000], labels=["low", "high"])
print(gdf.groupby("band", observed=True)["salary"].agg(["count", "mean"]))
# NOTE: observed=True skips empty categorical combinations (default flips to True in pandas 3)

# %% [markdown]
# ### `pivot_table(values, index, columns, aggfunc="mean", margins=False, margins_name="All", dropna=True, fill_value=None)`
# **What it does:** a spreadsheet-style cross-tabulation — *the* reshaping
# aggregation: `values` measured at the intersection of `index` (rows) and
# `columns` (column groups).
# **Parameters:**
# - `values` / `index` / `columns` — column names (or arrays) for measures / row keys / column keys
# - `aggfunc` — `"mean"` (default), `"sum"`, `"count"`, list, or dict
# - `margins` — add grand-total row+column; `margins_name` labels it
# - `dropna` — drop columns whose entries are all `NaN`; `fill_value` — replace `NaN` in the result
# **Returns:** DataFrame.

# %%
sales = pd.DataFrame({"region": ["N", "N", "S", "S", "N", "S"],
                      "product": ["A", "B", "A", "B", "A", "B"],
                      "units": [10, 15, 12, 8, 7, 20],
                      "revenue": [100, 75, 120, 40, 70, 100]})

print(pd.pivot_table(sales, values="revenue", index="region",
                     columns="product", aggfunc="sum", fill_value=0))
pt = pd.pivot_table(sales, values=["units", "revenue"], index="region",
                    columns="product", aggfunc=["sum", "mean"],
                    margins=True, margins_name="Total")
print(pt.round(1))
print(pd.pivot_table(sales, values="revenue", index="region",
                     aggfunc=["sum", "count", "mean"]))

# %% [markdown]
# ### `pivot` vs `pivot_table`
#
# | | `DataFrame.pivot` | `pd.pivot_table` |
# |---|---|---|
# | duplicate entries | raises `ValueError: Index contains duplicate entries` | aggregates them (default `mean`) |
# | `aggfunc` | ✗ | ✓ any aggregation |
# | margins / totals | ✗ | ✓ `margins=True` |
# | `fill_value`, `dropna` | ✗ | ✓ |
# | grouped input | one row per key | built-in `groupby` semantics |
#
# > **Rule of thumb:** `pivot` when the data is already unique per cell,
# > `pivot_table` for anything real-world (duplicates are the norm).

# %%
# unique cells -> plain pivot is fine
unique_cells = sales.drop_duplicates(subset=["region", "product"])
print(unique_cells.pivot(index="region", columns="product", values="revenue"))
try:
    sales.pivot(index="region", columns="product", values="units")
except ValueError as e:
    print("ValueError:", e)         # duplicated cell -> pivot fails, use pivot_table
print(pd.pivot_table(sales, values="units", index="region",
                     columns="product", aggfunc="sum"))

# %% [markdown]
# ### `pd.crosstab(index, columns, values=None, aggfunc=None, normalize=False)`
# **What it does:** counts (or aggregated values) at the intersection of two
# categoricals — the statistics-native version of `pivot_table` for frequencies.
# **Syntax:** `pd.crosstab(df.region, df.product, normalize="index")`
# **Returns:** DataFrame of counts (or normalised proportions).

# %%
print(pd.crosstab(sales["region"], sales["product"]))
print(pd.crosstab(sales["region"], sales["product"], values=sales["revenue"], aggfunc="sum"))
print(pd.crosstab(sales["region"], sales["product"], normalize="index").round(2))

# %% [markdown]
# **Section notes — groupby/reshape:**
#
# | Task | Code |
# |---|---|
# | mean per key | `df.groupby("k").mean(numeric_only=True)` |
# | many aggregates | `.agg({"col": ["sum","mean"]})` |
# | named output cols | `.agg(total=("col", "sum"))` |
# | add group stat as column | `.groupby("k")["v"].transform("mean")` |
# | keep whole groups | `.groupby("k").filter(fn)` |
# | arbitrary per group | `.groupby("k").apply(fn, include_groups=False)` |
# | cross-tab | `pd.pivot_table(...)` / `pd.crosstab(...)` |
# | unique cells? | `pivot` — otherwise `pivot_table` |
#
# > **Gotcha:** `df.groupby("k").mean()` raises
# > `TypeError: agg function failed [how->mean,dtype->object]` in pandas 2.x when
# > non-numeric columns remain — select `df.groupby("k")[num_cols].mean()` or pass
# > `numeric_only=True`.

# %% [markdown]
# ---
# ## 7. Merge, Join & Concat
#
# ▶ Session 20 - Merging, Joining & Concatenating: https://www.youtube.com/watch?v=Ssy1EfK5S-o

# %%
left = pd.DataFrame({"id": [1, 2, 3, 4],
                     "name": ["Rahul", "Sita", "Amir", "Zoe"],
                     "score": [88, 92, 79, 85]})
right = pd.DataFrame({"id": [3, 4, 5, 6],
                      "city": ["Patna", "Delhi", "Pune", "Kolkata"],
                      "score": [79, 85, 70, 66]})

# %% [markdown]
# ### `pd.merge(left, right, how="inner", on=None, left_on=None, right_on=None, left_index=False, right_index=False, suffixes=("_x", "_y"), indicator=False, validate=None, sort=False)`
# **What it does:** relational join on **columns** (or indices) — SQL `JOIN` in pandas.
# **Parameters:**
# - `how` — `"inner"` (default), `"left"`, `"right"`, `"outer"`; also `"cross"` for a cartesian product
# - `on` — shared key column(s); `left_on` / `right_on` — differently named keys
# - `left_index` / `right_index` — use the index as join key (mixable with `on`)
# - `suffixes` — added to overlapping **non-key** column names
# - `indicator=True` — adds `_merge` column: `left_only` / `right_only` / `both`
# - `validate` — `"one_to_one"`, `"many_to_one"`, ... → raises if the relationship is violated
# - `sort` — sort the result by join keys
# **Returns:** merged DataFrame.

# %%
print(pd.merge(left, right, on="id", how="inner")["id"].tolist())
print(pd.merge(left, right, on="id", how="left"))
print(pd.merge(left, right, on="id", how="outer").fillna({"score": 0}))
print(pd.merge(left, right, on="id", suffixes=("_l", "_r")))

m = pd.merge(left, right, on="id", how="outer", indicator=True)
print(m["_merge"].value_counts())
print(pd.merge(left, right, on="id", validate="one_to_one").shape)

# different key names:
right2 = right.rename(columns={"id": "student_id"})
print(pd.merge(left, right2, left_on="id", right_on="student_id").shape)

# %% [markdown]
# ### Merging on the index — `left_index=True, right_index=True` and `.join()`
# **What they do:** align on row labels instead of key columns. `DataFrame.join`
# is sugar for an index-based merge (`left.join(other, how=...)`).

# %%
li = left.set_index("id")
ri = right.set_index("id")
print(pd.merge(li, ri, left_index=True, right_index=True, how="inner"))
# NOTE: overlapping columns need suffixes (join defaults to ("", "") -> error)
print(li.join(ri, how="left", lsuffix="_l", rsuffix="_r"))   # keeps left's rows
print(li.join(ri, how="outer", lsuffix="_l", rsuffix="_r"))

# %% [markdown]
# ### `pd.concat(objs, axis=0, join="outer", ignore_index=False, keys=None, verify_integrity=False, sort=False)`
# **What it does:** glue frames along an axis **without matching keys** —
# stacking (axis=0) or column-binding (axis=1).
# **Parameters:**
# - `objs` — list of DataFrames/Series
# - `axis` — `0` rows (default), `1` columns
# - `join` — `"outer"` (union of labels) / `"intersection"` (common labels only)
# - `ignore_index` — renumber `0..n-1` after row-stacking
# - `keys` — label per input frame → **MultiIndex** on the result
# - `verify_integrity` — raise if the resulting index has duplicates
# - `sort` — sort the non-concatenation axis when `axis=1`
# **Returns:** single DataFrame/Series.

# %%
a = pd.DataFrame({"v": [1, 2]})
b = pd.DataFrame({"v": [3]})

print(pd.concat([a, b], ignore_index=True))             # row stack, renumbered
print(pd.concat([a, b], keys=["q1", "q2"]).index.tolist())  # MultiIndex groups
print(pd.concat([a, b], axis=1))                        # column bind (index aligned!)
print(pd.concat([a, b], join="inner").shape)            # common columns only

li2 = left.set_index("id")
ri2 = right.set_index("id")
print(pd.concat([li2, ri2], axis=1, join="inner").shape)  # side-by-side on id

try:
    pd.concat([a, a], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)                              # duplicate labels rejected

# %% [markdown]
# ### When to `concat` vs `merge`
#
# | Question | Tool |
# |---|---|
# | same columns, stack more rows? | `pd.concat([df1, df2])` |
# | different tables, match on a key? | `pd.merge(df1, df2, on="id")` |
# | attach columns row-aligned? | `pd.concat([...], axis=1)` or `join` |
# | SQL-ish join (inner/left/outer) | `pd.merge(how=...)` |
# | no key, every row with every row | `pd.merge(how="cross")` |
#
# > **Gotcha:** `concat(axis=1)` **aligns on index** — mismatched labels produce
# > `NaN`, not a row-by-row pairing. Reset/align indexes first if you meant
# > positional binding.
#
# > **Deprecation:** `df.append(...)` (pandas 1.x) is **removed** in 2.x —
# > `pd.concat([df, new_row])` is the replacement (see §2).

# %%
# overlap demo: concat stacks blindly, merge matches keys
v1 = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
v2 = pd.DataFrame({"id": [2, 3], "v": [999, 30]})
print("concat (stack):\n", pd.concat([v1, v2], ignore_index=True))
print("merge (match id, outer):\n", pd.merge(v1, v2, on="id", how="outer",
                                              suffixes=("_1", "_2")))

# %% [markdown]
# **Section notes — combine:**
#
# | Function | Axis | Matching |
# |---|---|---|
# | `pd.concat(objs, axis=0)` | rows | none (stack) |
# | `pd.concat(objs, axis=1)` | columns | index labels |
# | `pd.merge(l, r, how=..., on=)` | rows | key columns |
# | `df.join(other, how=)` | rows | index labels |
# | `df.append` | — | **removed in 2.x** → `pd.concat` |

# %% [markdown]
# ---
# ## 8. MultiIndex — hierarchical labels
#
# ▶ Session 21 - MultiIndex Series and DataFrames: https://www.youtube.com/watch?v=QzoRUwz8DoM

# %%
# ### `pd.MultiIndex.from_tuples(tuples, names=None)` / `from_arrays(arrays, names=None)` / `from_product(iterables)`
# **What they do:** build a hierarchical index — `from_tuples` from pairs,
# `from_arrays` from one label array per level, `from_product` from the cartesian
# product of level values (compact!).
# **Parameters:** `names` — level names (become `df.index.names`).
# **Returns:** `pandas.MultiIndex`.

mi = pd.MultiIndex.from_tuples([("N", "A"), ("N", "B"), ("S", "A"), ("S", "B")],
                               names=["region", "product"])
print(mi)
print(pd.MultiIndex.from_arrays([["N", "N", "S", "S"], ["A", "B", "A", "B"]],
                                names=["region", "product"]).tolist())
print(pd.MultiIndex.from_product([["N", "S"], ["A", "B"]]).tolist())

dfm = pd.DataFrame({"sales": [10, 20, 30, 40], "profit": [1, 2, 3, 4]}, index=mi)
dfm

# %% [markdown]
# ### `set_index(["a", "b"])`, `swaplevel()`, `sort_index()`, `droplevel()`
# **What they do:** move columns into a two-level index / reorder or drop levels —
# the everyday MultiIndex plumbing.

# %%
flat = pd.DataFrame({"region": ["N", "N", "S", "S"],
                     "product": ["A", "B", "A", "B"],
                     "sales": [10, 20, 30, 40]})
mi_df = flat.set_index(["region", "product"])
print(mi_df.index.names, mi_df.index.nlevels)
print(mi_df.swaplevel().sort_index())       # swap levels then re-sort
print(mi_df.index.droplevel("product").unique().tolist())
print(mi_df.reset_index())                  # back to flat columns

# %% [markdown]
# ### Indexing a MultiIndex: tuples, `.loc`, `.xs()` (cross-section)
# **What they do:**
# - `df.loc[("N", "A")]` — full tuple selects one row
# - `df.loc[("N",)]` / `df.xs("N", level="region")` — partial tuple = whole sub-frame
# - `df.xs("A", axis=1, level="product")` — cross-section on **columns**
# - `stack()/unstack()` — convert between MultiIndex rows ↔ column levels

# %%
print(dfm.loc[("N", "A")])                       # Series for one cell
print(dfm.loc[("N",)])                           # sub-frame for region N
print(dfm.xs("A", level="product"))              # same idea via cross-section
print(dfm.swaplevel().xs("B", level="product"))

# rows <-> columns conversion
wide = dfm.unstack()             # product level -> column level (MultiIndex cols!)
print(wide)
print(wide.columns.tolist())
print(wide["sales"])             # cross-section on a column level
print(wide.xs("A", axis=1, level="product"))
print(wide.stack([0, 1], future_stack=True).head(2))   # back to long form

# %% [markdown]
# > **Gotcha:** indexing with a *list* on a MultiIndex selects **labels at the
# > top level**, not combinations — `df.loc[["N"]]` (rows for N), while
# > `df.loc[[("N", "A")]]` selects exact tuples. Getting the brackets wrong is a
# > classic interview gotcha.

# %%
print(dfm.loc[[("N", "A"), ("S", "B")]])
print(dfm.loc[["N"]].shape)

# %% [markdown]
# **Section notes — MultiIndex:**
#
# | Task | Code |
# |---|---|
# | build | `MultiIndex.from_tuples/from_arrays/from_product` |
# | columns → index | `df.set_index(["a","b"])` |
# | reorder / drop levels | `swaplevel()`, `droplevel()`, `sort_index()` |
# | sub-frame | `df.loc[("N",)]`, `df.xs("N", level="region")` |
# | column level slice | `df["sales"]`, `df.xs("A", axis=1, level="product")` |
# | rows ↔ columns | `unstack()` / `stack(future_stack=True)` |

# %% [markdown]
# ---
# ## 9. Vectorised string operations — the `.str` accessor
#
# ▶ Session 22 - Vectorized String Ops / DateTime / Pivot Table: https://www.youtube.com/watch?v=UE6DmRQJ2w8

# %%
names = pd.Series(["  rahul singh ", "SITA ", "aMir khan", None, "$1,250.50"])
names

# %% [markdown]
# ### Case & length: `.str.lower() / upper() / title() / capitalize() / swapcase()` · `.str.len()`
# **What they do:** element-wise case conversion and character counts — implemented
# in C, **NaN-safe** (missing stays missing).
# **Syntax:** `s.str.lower()`, `s.str.len()`
# **Returns:** Series (same index).

# %%
print(names.str.lower().str.strip())
print(names.str.upper())
print(names.str.title())
print(names.str.capitalize())
print(pd.Series(["ab", "cd"]).str.len())
print(pd.Series(["a", None]).str.len())          # NaN -> NaN (no crash)

# %% [markdown]
# ### `.str.strip(to_strip=None)` / `.lstrip()` / `.rstrip()`
# **What it does:** trims whitespace (or explicit characters) from both/left/right ends.

# %%
print(names.str.strip())
print(names.str.strip("$"))
print(pd.Series(["**hi**"]).str.strip("*"))

# %% [markdown]
# ### `.str.contains(pat, case=True, regex=True, na=None)` · `.str.startswith(prefix)` · `.str.endswith(suffix)`
# **What they do:** element-wise predicates — filters for text columns.
# **Parameters:** `regex` — treat `pat` as a regular expression (default `True`); `case` — case-sensitive; `na` — value for missing (`None` → `NaN`).
# **Returns:** boolean Series.

# %%
print(names.str.contains("rahul", case=False, na=False))
print(names.str.contains(r"\d", na=False))            # has any digit
print(pd.Series(["cat", "dog", "bird"]).str.startswith("c"))
print(pd.Series(["cat", "dog", "bird"]).str.endswith("t"))
print(names[names.str.contains("singh", case=False, na=False)])

# %% [markdown]
# ### `.str.replace(pat, repl, regex=True)` · `.str.split(pat=None, expand=False)` · `.str.join(sep)`
# **What they do:** find-and-replace, splitting into lists or new columns, and
# joining list elements back into strings.
# **Syntax:** `s.str.replace(r"[$,]", "", regex=True)`, `s.str.split("-", expand=True)`

# %%
print(names.str.replace(r"[$,]", "", regex=True))
print(pd.Series(["a-b", "c-d"]).str.split("-"))            # Series of lists
print(pd.Series(["a-b", "c-d"]).str.split("-", expand=True))  # two columns!
print(pd.Series(["a,b", "c,d"]).str.split(",").str.join("|"))
print(pd.Series(["2020-01-05"]).str.split("-").str[1])      # list-style access

# %% [markdown]
# ### `.str.extract(pat, flags=0, expand=True)` / `.str.extractall(pat)` / `.str.findall(pat)`
# **What they do:** pull parts of a pattern out — `extract` takes the **first**
# match, one column per capture group; `extractall` returns *every* match (with a
# MultiIndex); `findall` returns lists.
# **Syntax:** `s.str.extract(r"(\d+)-(\w+)")`
# **Returns:** DataFrame (`expand=True`) or Series of tuples (`expand=False`).

# %%
print(pd.Series(["Order-123", "Order-45X"]).str.extract(r"(\d+)"))
print(pd.Series(["a1b2", "c3"]).str.extract(r"([a-z])(\d)", expand=True))
print(pd.Series(["a1b2"]).str.extractall(r"(\d)"))
print(pd.Series(["a1b2"]).str.findall(r"\d"))

# %% [markdown]
# ### Positional helpers: `.str.get(i)` · `.str.slice(start, stop)` · `.str.find(sub)` · `.str.count(pat)`
# **What they do:** per-string indexing without writing a Python loop.

# %%
print(pd.Series(["cat", "dog"]).str.get(0))               # first char
print(pd.Series(["2020-01", "2021-06"]).str.slice(0, 4))   # year
print(pd.Series(["cat", "dog"]).str.find("a"))            # -1 when absent
print(pd.Series(["cat", "dog"]).str.count("a"))

# %% [markdown]
# ### `.str.repeat(repeats)` · `.str.pad(width, side="left")` · `.str.center(width, fillchar)` · `.str.zfill(width)` · `.str.get_dummies(sep=" ")`
# **What they do:** formatting / expansion helpers — padding for fixed-width
# output, `get_dummies` for one-hot encoding of a delimited string.

# %%
print(pd.Series(["ab"]).str.repeat(3))
print(pd.Series(["7", "42"]).str.pad(4, side="left", fillchar="0"))
print(pd.Series(["a", "b"]).str.center(5, "-"))
print(pd.Series(["7"]).str.zfill(3))
print(pd.Series(["red blue", "blue", "green"]).str.get_dummies())

# %% [markdown]
# ### `.str` vs `.apply` — which one?
#
# | | `.str.func()` | `.apply(f)` / `.map(f)` |
# |---|---|---|
# | engine | vectorised C under the hood | Python function per row |
# | `NaN` handling | propagates `NaN` safely | raises (e.g. `float` has no `.lower`) |
# | regex helpers | `contains/replace/extract/...` built in | you re-implement with `re` |
# | custom logic | limited to the accessor set | anything |
#
# **Rule:** reach for `.str` first; drop to `.apply` only when the logic isn't a
# standard string op.

# %%
text = pd.Series(["  Order #1234 - $56.78  "] * 5000)
t_str = timeit.timeit(lambda: text.str.replace(r"[^0-9.]", "", regex=True), number=10)
t_apply = timeit.timeit(lambda: text.apply(lambda t: t.strip().replace("$", "")), number=10)
print(f".str.replace x10 : {t_str:.4f}s")
print(f".apply       x10 : {t_apply:.4f}s")

# NaN safety demo
dirty = pd.Series(["A", None])
print(dirty.str.lower())                       # ['a', NaN]  -> safe
try:
    dirty.apply(str.lower)
except TypeError as e:
    print("TypeError:", e)                     # apply crashes on NaN

# %% [markdown]
# **Section notes — `.str`:**
#
# | Group | Methods |
# |---|---|
# | case/space | `lower` `upper` `title` `capitalize` `strip` `lstrip` `rstrip` `len` |
# | test | `contains` `startswith` `endswith` `count` `find` |
# | rewrite | `replace` `split(expand)` `join` `repeat` `pad` `center` `zfill` |
# | pull out | `extract` `extractall` `findall` `get` `slice` |
# | encode | `get_dummies` |

# %% [markdown]
# ---
# ## 10. DateTime in pandas
#
# ▶ Session 22 - Vectorized String Ops / DateTime / Pivot Table: https://www.youtube.com/watch?v=UE6DmRQJ2w8

# %% [markdown]
# ### `pd.to_datetime(arg, errors="raise", dayfirst=False, utc=False, format=None, origin="unix")`
# **What it does:** parses strings/numbers into `datetime64[ns]` (a `Timestamp`
# scalar, `DatetimeIndex`/Series vectorised).
# **Parameters:**
# - `errors` — `"raise"` (default), `"coerce"` (bad → `NaT`), `"ignore"` (return input)
# - `format` — explicit format, e.g. `"%d/%m/%Y"` (skips inference, faster + stricter)
# - `dayfirst` — disambiguate `01/02/2020` as 1 Feb (default is month-first)
# - `utc` — return timezone-aware UTC timestamps
# **Returns:** `Timestamp` (scalar) / `DatetimeIndex` / Series of datetimes.

# %%
print(pd.to_datetime("2020-01-15"))
print(pd.to_datetime(["2020-01-01", "2021-06-15"]))
print(pd.to_datetime("01/02/2020", format="%d/%m/%Y"))     # 1 Feb 2020
print(pd.to_datetime(["2020-02-31", "2020-01-01"],          # bad -> NaT
                     format="%Y-%m-%d", errors="coerce"))
print(pd.to_datetime(["2020-01-01 10:00"], utc=True))       # tz-aware
print(pd.Timestamp("2020-01-15 09:30").day_name())

# %% [markdown]
# ### The `.dt` accessor — component extraction
# **What it does:** vectorised datetime fields on a Series (the `.dt` = `.str` of time).
# **Key members:** `year, month, day, hour, minute, second, dayofweek, day_name(),
# month_name(), quarter, date, time, days_in_month (daysinmonth), is_month_start,
# is_month_end, is_year_start, week, dayofyear, tz_localize(tz), tz_convert(tz)`.
# **Returns:** integers / strings / `date` objects / datetimes (series).

# %%
ds = pd.Series(pd.date_range("2022-01-01", periods=7, freq="D"), name="ts")
comp = pd.DataFrame({
    "date": ds,
    "year": ds.dt.year,
    "month": ds.dt.month,
    "day": ds.dt.day,
    "weekday": ds.dt.dayofweek,             # Monday=0
    "day_name": ds.dt.day_name(),
    "quarter": ds.dt.quarter,
    "hour": ds.dt.hour,
    "is_month_start": ds.dt.is_month_start,
    "days_in_month": ds.dt.days_in_month,
})
print(comp)
print(ds.dt.strftime("%d-%b-%Y (%A)"))       # custom format strings
print(ds.dt.date.tolist(), ds.dt.time.tolist()[:2])

# timezone localise (naive -> aware) and convert (aware -> other zone)
print(ds.dt.tz_localize("UTC").head(2))
print(ds.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").head(2))

# %% [markdown]
# ### `pd.date_range(start=None, end=None, periods=None, freq="D", tz=None, inclusive="both")`
# **What it does:** generates a regular `DatetimeIndex` — need any **two** of
# `start/end/periods` (the third is implied by `freq`).
# **Parameters:** `freq` — `"h"`, `"D"`, `"W"`, `"MS"` (month start), `"ME"` (month end), `"QS"`, `"Y"`...; `tz` — timezone; `inclusive` — which endpoints.
# **Returns:** `DatetimeIndex`.

# %%
print(pd.date_range("2022-01-01", periods=5, freq="D"))
print(pd.date_range("2022-01-01", "2022-03-01", freq="MS"))
print(pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC"))
print(pd.date_range("2022-01-01", periods=4, freq="W"))

# %% [markdown]
# ### `Series.dt.to_period(freq)` · `Timestamp.strftime` · `pd.Timedelta`
# **What they do:** period labels ("2022-01") for monthly grouping, display
# formatting, and duration arithmetic (`end - start` → `Timedelta`).
# **Syntax:** `s.dt.to_period("M")` · `ts.strftime("%Y-%m")` · `pd.Timedelta(days=2, hours=3)`

# %%
print(ds.dt.to_period("M").unique())          # month buckets (Period dtype)
print(pd.Timestamp("2022-01-15").strftime("%Y/%m/%d"))
delta = pd.Series(pd.to_datetime("2022-01-03")) - pd.Series(pd.to_datetime("2022-01-01"))
print(delta, delta.dt.days.tolist())
print(pd.Timedelta("2 days 3 hours") * 2)
print(pd.to_timedelta(["1 days", "05:00:00"]))

# %% [markdown]
# ### `Series.resample(rule).agg()` — time-based groupby
# **What it does:** buckets a **time-indexed** Series into calendar windows, then
# aggregates (`mean/sum/max/ffill/...`).
# **Syntax:** `ts.resample("ME").mean()`
# **Parameters:** `rule` — `"D"` daily, `"W"` weekly, `"ME"` month-end, `"MS"` month-start,
# `"h"` hourly, `"Q-DEC"` quarterly, `"Y"` yearly. (pandas 2.2+ deprecates the old
# `"M"` in favour of `"ME"`.)
# **Returns:** Series/DataFrame indexed by the bucket start/end.

# %%
daily = pd.Series(np.arange(30, dtype=float),
                  index=pd.date_range("2022-01-01", periods=30, freq="D"),
                  name="visits")
print(daily.resample("W").mean().round(1))
print(daily.resample("ME").sum())
print(daily.resample("D").ffill().head(3))
print(daily.resample("ME").count().head(3))       # any reduction works

# %% [markdown]
# ### `Series.asfreq(freq, method=None)` · `between_time(start_time, end_time)`
# - `asfreq` — *reindex* to a new frequency (introduces `NaN` at gaps; `method="ffill"` fills them)
# - `between_time` — keep rows whose **time of day** falls in a window (business-hours cuts)
#
# **Slicing a time-indexed Series** uses ISO strings directly: `ts["2022-01-05":"2022-01-10"]` (stop **inclusive**).

# %%
hourly = pd.Series(1.0, index=pd.date_range("2022-01-01", periods=48, freq="h"))
print(hourly.asfreq("6h").head(4))                 # gaps -> NaN
print(hourly.resample("6h").ffill().head(4))       # the usual alternative
print(hourly.between_time("08:00", "12:00").shape) # 08:00-12:00 daily window
print(hourly["2022-01-01 10:00":"2022-01-01 14:00"])   # ISO-string slicing
print(hourly.index.slice_indexer("2022-01-01 10:00", "2022-01-01 14:00"))

# %% [markdown]
# **Section notes — datetimes:**
#
# | Task | Code |
# |---|---|
# | parse | `pd.to_datetime(s, format=, errors="coerce", utc=True)` |
# | components | `s.dt.year / month / dayofweek / day_name() / hour / quarter` |
# | flags | `s.dt.is_month_start`, `s.dt.days_in_month` |
# | tz | `s.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")` |
# | calendar | `pd.date_range(..., freq="D", tz=)` |
# | format | `s.dt.strftime("%Y-%m-%d")` |
# | bucket | `s.resample("ME").sum()` (old `"M"` deprecated) |
# | gaps | `asfreq` vs `resample().ffill()` |
# | window | `s.between_time("09:00", "17:00")`, `s["2022-01-01":"2022-01-31"]` |
# | duration | `t2 - t1` → `Timedelta`, `pd.Timedelta("2 days")` |
#
# > **Gotcha:** `resample` **requires** a `DatetimeIndex`; `groupby` on
# > `s.dt.to_period("M")` gives period buckets without resampling. And never use
# > `"M"`-style aliases blindly in pandas 2.2+ — `"ME"` is the month-end spelling.

# %% [markdown]
# ---
# ## 11. Case-study pipeline — Indian Startup Funding (mini)
#
# ▶ Pandas Case Study 1 / Indian Startup Funding: https://www.youtube.com/watch?v=VGPnRAASARY
# ▶ Pandas Case Study / Time Series Analysis: https://www.youtube.com/watch?v=LknR_AVYboc
# ▶ Pandas Case Study 2 / Working with textual data: https://www.youtube.com/watch?v=cO7sebFXWT8
#
# The full case studies in the playlist work with the real Startup Funding CSV.
# Here is the **same pipeline shape** on a small deterministic (seeded) frame:
# **load → inspect → clean → features → group → pivot → top-N**.

# %%
# 1. LOAD — deliberately messy raw data (fixed seed => reproducible)
rng = np.random.default_rng(42)
n = 60
raw = pd.DataFrame({
    "startup_name": rng.choice(["Flipkart", "Zomato", "Ola", "Swiggy", "Paytm"], n),
    "industry": rng.choice(["E-commerce", "Food", "Fintech", "Transport", "Foodtech"], n),
    "city": rng.choice(["Bangalore ", "Mumbai", " Delhi", "Pune", None], n),
    "date": rng.choice(pd.date_range("2019-01-01", periods=40, freq="D"), n),
    "amount": rng.choice(["$1,000,000", "$2,500,000", "$500,000", "$10,000,000"], n),
    "stage": rng.choice(["Seed", "Series A", "Series B"], n),
})
raw.loc[raw.index[:2], "industry"] = np.nan       # sprinkle missing values
raw = pd.concat([raw, raw.iloc[:4]], ignore_index=True)   # duplicate rows
print(raw.shape)
raw.head()

# %%
# 2. INSPECT
raw.info(memory_usage="deep")
print(raw.isna().sum())
print(raw.duplicated().sum(), "duplicate rows")
print(raw.describe(include="all").loc[["count", "unique"]])

# %%
# 3. CLEAN
clean = raw.copy()

# strip currency symbols & commas, then to float (vectorised!)
clean["amount_usd"] = (clean["amount"]
                       .str.replace(r"[$,]", "", regex=True)
                       .astype(float))

# tidy categorical text: strip stray spaces, title-case
clean["city"] = clean["city"].str.strip().str.title()
clean["industry"] = clean["industry"].str.strip()

# parse dates once -> everything else is .dt accessors
clean["date"] = pd.to_datetime(clean["date"], errors="coerce")

# drop exact duplicates, then rows missing the keys we need
clean = (clean
         .drop_duplicates()
         .dropna(subset=["city", "industry", "amount_usd"]))
print(clean.shape, "rows after cleaning")
print(clean.isna().sum().sum(), "nulls left")

# %%
# 4. FEATURES
clean["year"] = clean["date"].dt.year
clean["month"] = clean["date"].dt.month
clean["month_name"] = clean["date"].dt.strftime("%Y-%b")
clean["is_weekend"] = clean["date"].dt.dayofweek >= 5
print(clean[["date", "year", "month", "month_name", "is_weekend"]].head())

# %%
# 5. GROUPBY INSIGHTS
by_industry = (clean.groupby("industry")
               .agg(deals=("amount_usd", "count"),
                    total_usd=("amount_usd", "sum"),
                    avg_usd=("amount_usd", "mean"),
                    max_usd=("amount_usd", "max"))
               .round(0)
               .sort_values("total_usd", ascending=False))
print(by_industry)

# share of funding per industry (transform keeps row shape)
clean["industry_share"] = (clean["amount_usd"]
                           / clean.groupby("industry")["amount_usd"].transform("sum"))
print(clean["industry_share"].groupby(clean["industry"]).sum().round(2))

# deals per stage per year
print(pd.crosstab(clean["year"], clean["stage"]))

# %%
# 6. PIVOT TABLE
pt = pd.pivot_table(clean, values="amount_usd", index="city",
                    columns="stage", aggfunc="sum", fill_value=0,
                    margins=True, margins_name="Total")
print(pt / 1e6)   # in $ millions

# monthly funding trend (time-series case study shape)
monthly = clean.groupby(clean["date"].dt.to_period("M"))["amount_usd"].agg(["count", "sum"])
print(monthly.head(8))

# %%
# 7. TOP-N & TEXT CLEANUP
print(clean.nlargest(5, "amount_usd")[["startup_name", "city", "stage", "amount_usd"]])
print(clean["startup_name"].str.title().value_counts())
print(clean.groupby("city")["amount_usd"].median().sort_values(ascending=False))

# %% [markdown]
# **Section notes — pipeline checklist:**
#
# | Step | Tools used |
# |---|---|
# | load | `pd.read_csv` / `DataFrame(...)` |
# | inspect | `info`, `describe(include="all")`, `isna().sum()`, `duplicated().sum()` |
# | clean | `str.strip`, `str.replace(regex=True)`, `astype(float)`, `to_datetime(errors="coerce")`, `drop_duplicates`, `dropna(subset=)` |
# | features | `dt.year/month/dayofweek`, `strftime` |
# | aggregate | `groupby().agg(named)`, `transform` (broadcast), `crosstab` |
# | reshape | `pivot_table(..., margins=True)` |
# | rank | `nlargest(n, col)` / `nsmallest` |
#
# > **TIP:** clean with *vectorised* `.str` ops — a 4-step chain on a million rows
# > is still sub-second; the same loop with `.apply` will crawl.

# %% [markdown]
# ---
# ## 12. Efficiency notes — vectorise, downcast, read less

# %%
# 1) vectorised ops vs Python loop (same maths)
data = pd.DataFrame({"val": np.random.default_rng(0).random(100_000)})
vec = timeit.timeit(lambda: (data["val"] * 2).sum(), number=100)
loop = timeit.timeit(lambda: sum(v * 2 for v in data["val"]), number=10)
print(f"pandas/numpy x100 : {vec:8.4f}s   -> {vec/100*1000:.4f} ms/call")
print(f"python loop  x10 : {loop:8.4f}s   -> {loop/10*1000:.4f} ms/call")
print(f"speed-up: {loop/10 / (vec/100):.0f}x per call")

# %%
# 2) .apply vs vectorised column ops
t_apply = timeit.timeit(lambda: data["val"].apply(lambda v: v * 2), number=50)
t_vec = timeit.timeit(lambda: data["val"] * 2, number=50)
print(f"Series.apply : {t_apply:.4f}s   Series*2 : {t_vec:.4f}s")

# %%
# 3) category dtype: low-cardinality columns as `category`
labels = np.random.default_rng(0).choice(
    ["north", "south", "east", "west"], size=100_000)
obj_s = pd.Series(labels, dtype="object")
cat_s = obj_s.astype("category")
print("object  :", round(obj_s.memory_usage(deep=True) / 1e6, 2), "MB")
print("category:", round(cat_s.memory_usage(deep=True) / 1e6, 2), "MB")
cmp_o = timeit.timeit(lambda: (obj_s == "north").sum(), number=20)
cmp_c = timeit.timeit(lambda: (cat_s == "north").sum(), number=20)
print(f"filter object {cmp_o:.4f}s vs category {cmp_c:.4f}s -> "
      f"{cmp_o/max(cmp_c, 1e-9):.0f}x faster")

# %%
# 4) numeric downcasting — same precision class, less RAM
small = pd.Series([1.0, 2.0, 3.0])
print("before:", small.dtype, small.nbytes, "bytes")
print("float32:", small.astype("float32").nbytes, "bytes")
print("int8  :", pd.Series([1, 2, 3]).astype("int8").nbytes, "bytes")
# NOTE: for counting/aggregations on huge frames even int64 -> int32/float32
# halves memory; check precision needs first (int32 max ~2.1e9).

# %%
# 5) read only what you need
sample_csv = """id,company,amount,city,date
1,Flipkart,1000000,Bangalore,2019-01-05
2,Zomato,2500000,Delhi,2019-03-12
3,Ola,500000,Pune,2019-06-20
4,Paytm,10000000,Gurgaon,2019-08-01"""

# usecols: skip columns you never touch
part = pd.read_csv(io.StringIO(sample_csv), usecols=["company", "amount"])
print(part.dtypes.to_dict())

# dtype: pin narrow types at read time (never cast later row-by-row)
part = pd.read_csv(io.StringIO(sample_csv), dtype={"id": "int8", "amount": "int32"})
print(part.dtypes.to_dict())

# nrows: preview / chunked pipelines
print(pd.read_csv(io.StringIO(sample_csv), nrows=2))

# parse_dates: parse at load instead of after
print(pd.read_csv(io.StringIO(sample_csv), parse_dates=["date"]).dtypes.to_dict())
# TIP: for very large files add chunksize= and process with a loop / concat

# %% [markdown]
# **Section notes — efficiency:**
#
# | Anti-pattern | Fast alternative |
# |---|---|
# | `for row in df.itertuples()` | vectorised ops / `groupby` |
# | `df.col.apply(f)` for maths | `df.col * k`, `np.where`, `df.eval` |
# | storing ids as `object` strings | `.astype("category")` |
# | `float64` you don't need | `astype("float32"/"int8")` |
# | reading 40 columns, using 3 | `usecols=[...]` + `dtype={...}` |
# | parse after load | `parse_dates=[...]` at read time |

# %% [markdown]
# ---
# ## 13. Quick revision sheet
#
# | Task | Code |
# |---|---|
# | create Series | `pd.Series(data, index=, name=)` |
# | create DataFrame | `pd.DataFrame({"col": [...]})` |
# | read CSV | `pd.read_csv(io.StringIO(txt), usecols=, dtype=, nrows=, parse_dates=)` |
# | shape / dtypes / nulls | `df.shape`, `df.dtypes`, `df.isna().sum()` |
# | label vs position | `df.loc[...]` / `df.iloc[...]` |
# | filter rows | `df[df.a > 5]`, `df.query("a > 5")` |
# | safe edit | `df.loc[mask, "col"] = v` (+ `.copy()` on slices) |
# | missing | `dropna(subset=)`, `fillna(dict)`, `.ffill()` `.bfill()` |
# | dedupe | `drop_duplicates(subset=, keep=)` |
# | string ops | `s.str.lower()`, `.str.replace(regex=True)`, `.str.extract()` |
# | datetime | `pd.to_datetime(...)` → `.dt.year` / `.resample("ME").sum()` |
# | group + agg | `df.groupby("k").agg(total=("v", "sum"))` |
# | broadcast group stat | `df.groupby("k")["v"].transform("mean")` |
# | cross-tab | `pd.pivot_table(df, values=, index=, columns=, aggfunc=, margins=True)` |
# | join | `pd.merge(l, r, on="id", how="left", indicator=True, validate="many_to_one")` |
# | stack rows | `pd.concat([a, b], ignore_index=True)` |
# | hierarchical index | `df.set_index(["a", "b"])` → `.loc[("a",)]`, `.xs()` |
# | top-N | `df.nlargest(n, "col")` |
# | reproducible sampling | `df.sample(n, random_state=42)` |
# | memory | `df.memory_usage(deep=True)`, `.astype("category")` |
#
# > **Interview short-list:** `loc` vs `iloc` vs `at`/`iat` · label-slice
# > inclusivity · index alignment in arithmetic (`+` → `NaN`) ·
# > split-apply-combine · `pivot` vs `pivot_table` · `merge` `how=` types ·
# > `SettingWithCopyWarning` and the fix · removed APIs in pandas 2.x
# > (`append`, `applymap`, `fillna(method=)`, `"M"` freq) · `.str` vs `.apply` ·
# > `transform` (same shape) vs `agg` (reduce).

# %% [markdown]
# ---
# **Next notebook:** `05_visualization.ipynb` — turning these tables into charts
# with Matplotlib, Seaborn & Plotly.
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
