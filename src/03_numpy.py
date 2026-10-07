# %% [markdown]
# # 03 · NumPy — Complete Function Reference
#
# NumPy (Numerical Python) is the base library of the entire data science stack —
# pandas, matplotlib, scikit-learn are all built on top of it. It gives us the
# `ndarray`: a fixed-type, memory-contiguous, vectorised array that is orders of
# magnitude faster than Python lists.
#
# **Why NumPy over lists?**
#
# | Feature | Python `list` | NumPy `ndarray` |
# |---|---|---|
# | Type | heterogeneous | homogeneous (single dtype) |
# | Speed | slow, interpreted loops | fast, C-level vectorised ops |
# | Memory | pointer per element + object overhead | packed raw bytes |
# | Ops | element-wise needs a loop | `a + b`, `a * 2` broadcast automatically |
# | Functions | one-by-one | whole-array, ufuncs, linear algebra, FFT |
#
# ```
# pip install numpy
# import numpy as np
# ```
#
# > **Note:** All examples below run top-to-bottom in a fresh kernel (NumPy 2.x).

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 41 | Session 13 – Numpy Fundamentals | https://www.youtube.com/watch?v=XF6DCrNTzug |
# | 43 | Session 14 – Advanced Numpy | https://www.youtube.com/watch?v=40xGMygHMDU |
# | 45 | Session 15 – Numpy Tricks | https://www.youtube.com/watch?v=jFBpnd0G09o |
# | 55 | Week 5 – Numpy Interview Questions | https://www.youtube.com/watch?v=wjv01Opo6Bk |

# %%
import numpy as np

np.__version__
# NOTE: everything below uses the alias `np`

# %% [markdown]
# ---
# ## 1. Creating arrays
#
# ▶ Session 13 – Numpy Fundamentals: https://www.youtube.com/watch?v=XF6DCrNTzug

# %% [markdown]
# ### `np.array(object, dtype=None, copy=True, ndmin=0)`
# **What it does:** converts any array-like (list, tuple, nested lists, ndarray) into an ndarray.
# **Parameters:**
# - `object` — array-like data
# - `dtype` — target element type; if omitted, NumPy *infers* the smallest common type
# - `copy` — if `False`, reuse memory when possible (NumPy 2.0+: `copy=None` = copy only if needed)
# - `ndmin` — minimum number of dimensions of the result
# **Returns:** `np.ndarray`

# %%
# 1-D, 2-D, 3-D and inferred dtype
a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a1, a1.ndim, a1.shape, a1.dtype)
print(a2, a2.ndim, a2.shape, a2.dtype)
print(a3, a3.ndim, a3.shape, a3.dtype)

# %%
# dtype promotion: int + float -> float64 ; int + str -> <U21 (string)
print(np.array([1, 2, 3]).dtype)                    # int64 (on most 64-bit machines)
print(np.array([1, 2, 3.0]).dtype)                  # float64
print(np.array([1, 2, "hi"]).dtype)                 # <U21
print(np.array([1, 2, 3], dtype=float))             # explicit cast
print(np.array([[1, 2], [3.5, 4]], dtype=int))      # truncates 3.5 -> 3
print(np.array([1, 2], ndmin=3).shape)              # (1, 1, 2)

# %% [markdown]
# ### `np.zeros(shape, dtype=float)`, `np.ones(shape, dtype=float)`, `np.full(shape, fill_value)`
# **What they do:** build arrays filled with a constant — the workhorses for accumulator buffers and masks.

# %%
print(np.zeros((2, 3)))                 # default float64 zeros
print(np.ones(4, dtype=int))            # int ones
print(np.full((2, 2), 7))               # every element = 7
print(np.zeros_like(a2))                # same shape/dtype as a2, all zeros
print(np.ones_like(a1, dtype=float))    # same shape as a1, float ones

# %% [markdown]
# ### `np.arange(start=0, stop, step=1)`
# **What it does:** half-open range `[start, stop)` as an array — Python's `range()` in array form.
# **Gotcha:** `stop` is *exclusive*; a float `step` can make the length unpredictable (use `np.linspace` for exact counts).

# %%
print(np.arange(5))            # [0 1 2 3 4]
print(np.arange(2, 11, 3))     # [ 2  5  8]
print(np.arange(0, 1, 0.25))   # [0.   0.25 0.5  0.75]

# %% [markdown]
# ### `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`
# **What it does:** `num` *equally spaced* samples over `[start, stop]`.
# **Key difference vs `arange`:** exact number of points; includes `stop` when `endpoint=True`.
# **Perfect for** plotting smooth curves (`x = np.linspace(-5, 5, 200)`).

# %%
x, step = np.linspace(0, 10, 5, retstep=True)
print(x, "step =", step)                      # [ 0.   2.5  5.   7.5 10. ]  step=2.5
print(np.linspace(0, 10, 5, endpoint=False))  # [0. 2. 4. 6. 8.]

# %% [markdown]
# ### `np.eye(N, M=None, k=0)`, `np.identity(n)`, `np.diag(v)`
# **What they do:** identity/diagonal matrices — the foundation of linear-algebra code.

# %%
print(np.eye(3))                      # 3x3 identity
print(np.eye(3, 4))                   # rectangular, 1s on main diagonal
print(np.eye(4, k=1))                 # diagonal shifted up by 1
print(np.identity(4))                 # same as np.eye(4)
print(np.diag([1, 2, 3]))             # diagonal matrix from a vector
print(np.diag(np.eye(3)))             # main diagonal extracted -> [1. 1. 1.]

# %% [markdown]
# ### `np.fromfunction(function, shape, dtype=float)`
# **What it does:** builds an array by evaluating `function(i, j, k, ...)` on index grids — elegant for coordinate matrices.

# %%
print(np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=int))

# %% [markdown]
# **Section notes — creation:**
#
# | Function | Use when |
# |---|---|
# | `np.array` | you already have data |
# | `np.zeros/ones/full` | you need an initialised buffer |
# | `np.arange` | integer steps, count doesn't matter |
# | `np.linspace` | exact count / plotting grids |
# | `np.eye/identity/diag` | linear algebra |
# | `np.fromfunction` | index-driven construction |

# %% [markdown]
# ---
# ## 2. Array attributes & inspecting data
#
# ▶ Session 13 – Numpy Fundamentals: https://www.youtube.com/watch?v=XF6DCrNTzug

# %%
m = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)

print("shape   :", m.shape)     # (2, 3)  -> (rows, cols); a tuple of dimension sizes
print("ndim    :", m.ndim)      # 2       -> number of dimensions (axes)
print("size    :", m.size)      # 6       -> total elements  (prod(shape))
print("dtype   :", m.dtype)     # float64 -> element type
print("itemsize:", m.itemsize)  # 8       -> bytes per element
print("nbytes  :", m.nbytes)    # 48      -> size * itemsize
print("T       :\n", m.T)       # transpose (property, not a method)

# %% [markdown]
# ### `.real`, `.imag`, `.flat`, `.nbytes`
# **`.flat`** is a 1-D *iterator* over elements regardless of shape — handy for loops and `.flat[i]` indexing.

# %%
c = np.array([1 + 2j, 3 - 4j])
print(c.real, c.imag)              # [1. 3.]  [ 2. -4.]
m2 = np.arange(6).reshape(2, 3)
print(list(m2.flat)[:4])           # flatten for iteration: [0, 1, 2, 3]
print(m2.flat[4])                  # positional indexing works on any shape

# %% [markdown]
# **Section notes — attributes:**
#
# | Attribute | Meaning |
# |---|---|
# | `.shape` | tuple of sizes per axis |
# | `.ndim` | number of axes |
# | `.size` | total element count |
# | `.dtype` | element data type |
# | `.itemsize` / `.nbytes` | bytes per element / total bytes |
# | `.T` | transpose view |

# %% [markdown]
# ---
# ## 3. `reshape`, `resize`, `ravel`, `flatten`, `expand_dims`, `squeeze`
#
# ▶ Session 13 – Numpy Fundamentals: https://www.youtube.com/watch?v=XF6DCrNTzug

# %%
a = np.arange(12)
print(a.reshape(3, 4))             # -1 means "infer this axis"
print(a.reshape(2, -1))            # (2, 6)
print(a.reshape(-1, 2, 3))         # (2, 2, 3)

# %%
# ravel (view when possible) vs flatten (always a copy)
b = a.reshape(3, 4)
r = b.ravel()
f = b.flatten()
r[0] = 99
print("after ravel mutation ->\n", b)   # CHANGED (view!)
f[1] = 77
print("after flatten mutation ->\n", b) # unchanged (copy)

# %%
# squeeze / expand_dims: drop or add size-1 axes
s = np.random.default_rng(0).integers(0, 9, size=(1, 3, 1, 2))
print(s.shape, "->", s.squeeze().shape)          # (1,3,1,2) -> (3,2)
print(np.expand_dims(a, axis=0).shape, np.expand_dims(a, axis=1).shape)
print(np.atleast_2d(np.array([1, 2, 3])).shape)  # guarantee >= 2 dims

# %% [markdown]
# > **Gotcha:** `reshape` fails if `size` of source and target differ
# > (`ValueError: cannot reshape array of size 12 into shape (4, 4)`).
# > `newaxis` (`a[:, np.newaxis]`) is another way to insert an axis.

# %%
col = a.reshape(4, 3)[:, np.newaxis]   # (4,3) -> (4,1,3)
print(col.shape)

# %% [markdown]
# ---
# ## 4. Indexing & slicing (the #1 interview topic)
#
# ▶ Session 14 – Advanced Numpy: https://www.youtube.com/watch?v=40xGMygHMDU

# %%
arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120]])

print(arr[1, 2])          # single element: row 1, col 2  -> 70
print(arr[0])             # whole row 0                   -> [10 20 30 40]
print(arr[:, 1])          # whole column 1                -> [ 20  60 100]
print(arr[0:2, 1:3])      # sub-matrix
print(arr[::-1])          # reverse rows
print(arr[:, ::2])        # every 2nd column

# %%
# boolean masking -> 1-D array of matching elements (NOT a sub-matrix!)
mask = arr > 60
print(mask)
print(arr[mask])                # [ 70  80  90 100 110 120]
print(arr[arr > 60])            # same
print(arr[(arr > 40) & (arr < 90)])   # element-wise & on masks (use & | ~, not and/or)

# %%
# fancy indexing: integer index arrays pick elements in that order
print(arr[[0, 2, 1], [1, 3, 0]])   # -> [ 20 120  50]  (pairs: (0,1),(2,3),(1,0))
print(arr[[0, 2]])                 # rows 0 and 2 (copy, not a view)

# %%
# assignment through indexing
arr2 = arr.copy()
arr2[arr2 > 100] = 0
arr2[0, :] = -1
print(arr2)

# %% [markdown]
# > **Gotcha:** a slice like `v = arr[0:2]` returns a **view**.
# > Mutating `v` changes `arr`. Use `.copy()` when you need independence.
# > Fancy (integer-array) indexing always returns a **copy**.

# %%
v = arr[0:2]
v[0, 0] = -999
print("view mutation leaks into original:", arr[0, 0])

# %% [markdown]
# **Section notes — indexing:**
#
# | Syntax | Meaning | View/Copy |
# |---|---|---|
# | `a[i]`, `a[i, j]`, `a[i:j, k:l]` | basic slicing | view |
# | `a[mask]` | boolean filter | copy |
# | `a[[0, 2], [1, 3]]` | fancy index pairs | copy |
# | `a[:, np.newaxis]` | insert axis | view |

# %% [markdown]
# ---
# ## 5. Data types, casting & type safety
#
# ▶ Session 13 – Numpy Fundamentals: https://www.youtube.com/watch?v=XF6DCrNTzug

# %%
i = np.array([1, 2, 3], dtype=np.int32)
print(i, i.dtype, i.itemsize)
print(i.astype(np.float64))      # astype ALWAYS returns a new array (copy)
print(i.astype(bool))            # 0 -> False, non-zero -> True
print(np.can_cast(np.int64, np.float32))   # False (int64 -> float32 loses precision)
print(np.can_cast(np.int32, np.float64))   # True
print(np.promote_types(np.int32, np.float32))  # dtype('float64')

# %%
# common dtypes
for dt in (np.int8, np.int16, np.int32, np.int64,
           np.uint8, np.float16, np.float32, np.float64, np.bool_):
    info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else None
    print(dt.__name__, "->", np.zeros(1, dtype=dt).dtype)

# %% [markdown]
# > **Gotcha:** integer overflow is silent in NumPy —
# > `np.array([127], dtype=np.int8) + 1` wraps to `-128`.
# > Never use `np.uint8` for accumulators.

# %%
print((np.array([127], dtype=np.int8) + 1))   # [-128]  (overflow!)

# %% [markdown]
# ---
# ## 6. Universal functions (ufuncs) — vectorised maths
#
# ▶ Session 14 – Advanced Numpy: https://www.youtube.com/watch?v=40xGMygHMDU

# %%
x = np.array([1.0, 4.0, 9.0, 16.0])
y = np.array([2.0, 3.0, 5.0, 7.0])

# arithmetic ufuncs — element-wise, no Python loop
print(x + y, x - y, x * y, x / y, x // y, x ** 2, np.mod(x, y), -x)

# %%
# universal math functions
print(np.sqrt(x), np.exp(x[:2]), np.log(x), np.log2(x), np.log10(x))
print(np.sin(np.pi / 2), np.cos(0), np.round(np.array([1.2, 1.7]), 0))
print(np.abs(np.array([-1, -2.5])), np.ceil(1.2), np.floor(1.8), np.trunc(1.8))

# %%
# comparison ufuncs return boolean arrays (element-wise)
print(x > y, x == y)
print(np.greater(x, y), np.equal(x, y))
print(np.maximum(x, y), np.minimum(x, y))

# %%
# trigonometry / sign helpers
ang = np.array([0, np.pi / 6, np.pi / 4, np.pi / 2])
print(np.sin(ang), np.degrees(np.pi), np.sign(np.array([-3, 0, 4])))

# %% [markdown]
# ### Floating-point helpers every DS engineer needs
# - `np.isnan` / `np.isfinite` / `np.isinf` — NaN & inf detection
# - `np.nan_to_num(a, nan=0.0)` — replace NaN/inf with numbers
# - `np.allclose(a, b, rtol=1e-5, atol=1e-8)` — tolerant equality (never use `==` on floats)

# %%
bad = np.array([1.0, np.nan, np.inf, -np.inf])
print(np.isnan(bad), np.isfinite(bad), np.isinf(bad))
print(np.nan_to_num(bad, nan=0.0, posinf=999.0, neginf=-999.0))
print(np.allclose([1.0, 2.0], [1.0, 2.0000000001]))

# %% [markdown]
# **Section notes — ufuncs:**
#
# | Group | Functions |
# |---|---|
# | Arithmetic | `+ - * / // % ** np.mod np.negative` |
# | Powers/logs | `np.sqrt np.exp np.log np.log2 np.log10` |
# | Rounding | `np.round np.ceil np.floor np.trunc np.abs` |
# | Compare | `np.greater np.equal np.maximum np.minimum` |
# | Sanity | `np.isnan np.isfinite np.allclose np.nan_to_num` |
#
# > **TIP:** every ufunc accepts `where=` (compute only where a condition holds)
# > and `out=` (write into an existing array → saves memory).

# %%
out = np.zeros(4)
np.add(x, y, out=out, where=(x < 10))   # only compute where mask is True
print(out)

# %% [markdown]
# ---
# ## 7. Aggregations & the `axis` parameter
#
# ▶ Session 14 – Advanced Numpy: https://www.youtube.com/watch?v=40xGMygHMDU

# %%
m = np.arange(1, 13).reshape(3, 4)
print(m)

print("sum all        :", m.sum())
print("sum axis=0     :", m.sum(axis=0))   # collapse ROWS -> one value per COLUMN
print("sum axis=1     :", m.sum(axis=1))   # collapse COLS -> one value per ROW
print("mean, std, var :", m.mean(), m.std(), m.var())
print("min/max        :", m.min(), m.max())
print("argmin/argmax  :", m.argmin(), m.argmax())       # flattened position
print("argmax axis=0  :", m.argmax(axis=0))             # index of max per column
print("cumsum axis=1  :\n", m.cumsum(axis=1))
print("prod           :", m.prod(), np.prod(m, axis=1))
print("median         :", np.median(m))
print("percentile 25/50/90 :", np.percentile(m, [25, 50, 90]))
print("any / all      :", (m > 0).any(), (m > 12).all())

# %%
# nan-safe variants — ALWAYS use these on real (dirty) data
dirty = np.array([1.0, np.nan, 3.0, 4.0])
print("sum ignores nothing :", dirty.sum())           # nan propagates!
print("np.nansum           :", np.nansum(dirty))      # 8.0
print("np.nanmean          :", np.nanmean(dirty))     # 2.666...
print("np.nanmedian/std/min/max also exist")

# %% [markdown]
# ### `keepdims=True` — the shape-saving trick
# **What it does:** aggregation keeps the reduced axis with size 1, so results broadcast
# back against the original array without manual `reshape`.

# %%
print(m.sum(axis=1, keepdims=True))   # shape (3,1) not (3,)
print(m - m.mean(axis=1, keepdims=True))   # row-wise de-meaning in one line

# %% [markdown]
# **Section notes — reductions:**
#
# | Operation | Full array | Along axis |
# |---|---|---|
# | sum/mean/std/var/prod | `a.sum()` | `a.sum(axis=k)` |
# | min/max | `a.min()` | `a.min(axis=k)` |
# | index of min/max | `a.argmin()` | `a.argmin(axis=k)` |
# | median/percentile | `np.median(a, axis=k)` | axis supported |
# | running total | — | `a.cumsum(axis=k)` |
# | NaN-safe | `np.nansum`, `np.nanmean`, ... | `axis=` supported |
#
# > **Gotcha:** for a 2-D array, `axis=0` means *along rows* (down a column),
# > `axis=1` means *along columns* (across a row). Memory order C is row-major,
# > so `axis=1` collapses the "inner" axis.

# %% [markdown]
# ---
# ## 8. Broadcasting
#
# ▶ Session 14 – Advanced Numpy: https://www.youtube.com/watch?v=40xGMygHMDU

# %%
# Rule: align shapes from the RIGHT; each dim must be equal or 1.
a = np.ones((3, 4))
b = np.arange(4)          # shape (4,) -> (1, 4) -> broadcast over rows
print(a + b)

c = np.arange(3).reshape(3, 1)   # (3,1) -> broadcast over columns
print(a + c)

# %%
# row-wise normalisation using broadcasting (no loops!)
data = np.array([[10.0, 20.0, 30.0],
                 [40.0, 50.0, 60.0]])
print(data - data.mean(axis=1, keepdims=True))
print(data / data.max(axis=1, keepdims=True))

# %%
# incompatible shapes -> error
try:
    np.ones((3, 4)) + np.ones((3, 5))
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# > **Gotcha:** broadcasting never copies data (it's a virtual view), but a
# > shape like `(3, 4) + (3,)` is **NOT** valid — the `3` aligns against `4`.
# > Fix by reshaping to `(1, 3)` or `(3, 1)` explicitly.

# %% [markdown]
# ---
# ## 9. Sorting, searching & set operations
#
# ▶ Session 15 – Numpy Tricks: https://www.youtube.com/watch?v=jFBpnd0G09o

# %%
s = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print("np.sort         :", np.sort(s))            # sorted COPY (quicksort)
print("a.sort()        :", s.copy().sort() or np.sort(s))
print("sorted, kind    :", np.sort(s, kind="stable"))
print("argsort         :", np.argsort(s))         # indices that would sort
print("searchsorted    :", np.searchsorted(np.sort(s), 5))  # insert position
print("where           :", np.where(s > 3, s, 0))  # 3-arg where: if/else vectorised
print("np.where(mask)  :", np.where(s > 3))        # tuple of index arrays
print("unique          :", np.unique(s, return_counts=True))
print("nonzero         :", np.nonzero(s % 2))
print("isin            :", np.isin(s, [1, 5, 9]))
print("max/min of two  :", np.maximum(s, 3), np.clip(s, 2, 5))

# %%
# 2-D sorting
m = np.array([[3, 1], [2, 4]])
print(np.sort(m, axis=0))          # sort each column
print(np.sort(m, axis=1))          # sort each row
print(np.sort(m.ravel())[::-1])    # global descending order

# %% [markdown]
# ### Set-like helpers on arrays
# `np.unique` (dedupe), `np.in1d`/`np.isin` (membership), `np.intersect1d`,
# `np.union1d`, `np.setdiff1d` — the array equivalents of Python set ops.

# %%
u = np.array([1, 2, 3, 4])
v = np.array([3, 4, 5])
print("intersect:", np.intersect1d(u, v))
print("union    :", np.union1d(u, v))
print("diff     :", np.setdiff1d(u, v))
print("sym diff :", np.setxor1d(u, v))

# %% [markdown]
# **Section notes — sort/search:**
#
# | Function | Purpose |
# |---|---|
# | `np.sort(a)` / `a.sort()` | copy sort / in-place sort |
# | `np.argsort(a)` | positions to build sorted order |
# | `np.searchsorted(a, v)` | O(log n) lookup in a sorted array |
# | `np.where(cond, x, y)` | vectorised if/else; `np.where(cond)` = indices |
# | `np.unique(a, return_counts=True)` | value counts without pandas |
# | `np.clip(a, lo, hi)` | clamp values into a range |

# %% [markdown]
# ---
# ## 10. Stacking, splitting & tiling
#
# ▶ Session 15 – Numpy Tricks: https://www.youtube.com/watch?v=jFBpnd0G09o

# %%
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])

print(np.stack([p, q]))           # axis=0, new dimension -> (2,3)
print(np.vstack([p, q]))          # vertical (rows)      -> (2,3)
print(np.hstack([p, q]))          # horizontal            -> (6,)
print(np.column_stack([p, q]))    # force 2-D columns     -> (3,2)
# NOTE: np.row_stack was only an alias of vstack and is deprecated in NumPy 2.x

# %%
big = np.arange(12)
print(np.split(big, 3))            # equal chunks
print(np.split(big, [3, 7]))       # split at given indices -> 3 pieces
print(np.array_split(big, 5))      # works when 12 % 5 != 0
m2 = big.reshape(3, 4)
print(np.hsplit(m2, 2))            # split columns
print(np.vsplit(m2, 3))            # split rows

# %%
print(np.tile(p, 3))               # repeat the ARRAY   -> [1 2 3 1 2 3 1 2 3]
print(np.repeat(p, 2))             # repeat EACH ELEMENT -> [1 1 2 2 3 3]
print(np.append(big, [99, 100]))   # flatten + append

# %% [markdown]
# ---
# ## 11. Random module (`np.random`)
#
# ▶ Session 15 – Numpy Tricks: https://www.youtube.com/watch?v=jFBpnd0G09o

# %%
rng = np.random.default_rng(seed=42)      # modern API (NumPy >= 1.17), PCG64

print(rng.random((2, 3)))                 # U(0, 1) floats
print(rng.integers(0, 10, size=5))        # integers in [0, 10)
print(rng.integers(0, 10, size=(2, 4), endpoint=True))   # inclusive upper bound
print(rng.choice([10, 20, 30], size=4))   # sample from an array
print(rng.choice(10, size=5, replace=False))  # sampling WITHOUT replacement
print(rng.permutation([1, 2, 3, 4]))      # shuffle a copy
print(rng.normal(loc=0, scale=1, size=5)) # normal distribution
print(rng.binomial(n=10, p=0.5, size=5))  # binomial
print(rng.uniform(0, 5, size=3))          # uniform [0, 5)
print(rng.standard_normal(4))             # standard normal

# %%
# old global-state API still works but is discouraged
np.random.seed(0)
print(np.random.rand(3))                  # legacy: module-level state

# %% [markdown]
# ### Why `default_rng()`?
# - each `Generator` is **independent** (no global state to corrupt)
# - faster & better statistically than the legacy `RandomState`
# - reproducible: same `seed` → same sequence, but sequences differ from legacy `np.random.seed`
#
# **Reshuffling / bootstrapping recipes:**

# %%
idx = rng.permutation(100)
train, test = idx[:80], idx[80:]          # manual train/test split
print(train[:5], test[:5])
boot = rng.choice(50, size=50, replace=True)   # bootstrap sample
print(boot[:10])

# %% [markdown]
# ---
# ## 12. Linear algebra mini-reference
#
# ▶ Session 15 – Numpy Tricks: https://www.youtube.com/watch?v=jFBpnd0G09o

# %%
A = np.array([[2, 1], [1, 3]])
B = np.array([[1, 2], [3, 4]])

print("element-wise * :\n", A * B)
print("matmul @ :\n", A @ B)
print("np.dot  :\n", np.dot(A, B))
print("np.matmul:\n", np.matmul(A, B))          # same as @ for 2-D

print("transpose      :\n", A.T)
print("det            :", np.linalg.det(A))
print("inv            :\n", np.linalg.inv(A))
print("eigvals        :", np.linalg.eigvals(A))
print("trace          :", np.trace(A))
print("norm (L2)      :", np.linalg.norm(np.array([3, 4])))
print("solve Ax=b     :", np.linalg.solve(A, np.array([5, 10])))
print("matrix power   :\n", np.linalg.matrix_power(A, 2))

# %%
# pseudo-inverse (handles rank-deficient matrices) & decompositions
M = np.array([[1.0, 2.0], [2.0, 4.0]])     # singular!
try:
    np.linalg.inv(M)
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)
print("pinv:\n", np.linalg.pinv(M))          # Moore-Penrose pseudo-inverse

Q, R = np.linalg.qr(A)                       # QR decomposition
U, S, Vt = np.linalg.svd(A)                  # singular value decomposition
print("QR Q:\n", Q, "\nR:\n", R)
print("singular values:", S)

# %%
# vectorised distance helpers used constantly in ML
X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
d = X - X[0]
print("Euclidean row distances:", np.sqrt((d ** 2).sum(axis=1)))
print("same via norm          :", np.linalg.norm(d, axis=1))
print("pairwise distances:\n",
      np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)))

# %% [markdown]
# **Section notes — linear algebra:**
#
# | Function | Meaning |
# |---|---|
# | `A @ B`, `np.dot` | matrix product |
# | `np.linalg.inv/det/trace` | inverse, determinant, trace |
# | `np.linalg.solve(A, b)` | solve linear system (prefer over `inv @ b`) |
# | `np.linalg.norm` | vector/matrix norm (default L2) |
# | `np.linalg.eig` | eigenvalues/vectors |
# | `np.linalg.svd` | singular value decomposition |
# | `np.linalg.pinv` | pseudo-inverse for singular matrices |

# %% [markdown]
# ---
# ## 13. Copy vs view, memory layout & `astype`
#
# ▶ Session 14 – Advanced Numpy: https://www.youtube.com/watch?v=40xGMygHMDU

# %%
base = np.arange(6)
view = base[1:4]          # slice -> VIEW (shares memory)
copy_ = base[1:4].copy()  # explicit COPY

view[0] = 99
print("base after view edit :", base)     # [0 99 2 3 4 5]
print("shares memory?       :", np.shares_memory(base, view))

copy_[0] = 77
print("base after copy edit :", base)     # unchanged
print("shares memory?       :", np.shares_memory(base, copy_))

# %%
# is this a view?
print("base is view.base :", view.base is base)
print("flags:\n", base.flags)

# %% [markdown]
# > **Gotcha — chained assignment trap:**
# > `a[a > 0][0] = -1` does **NOT** modify `a` (the second `[]` works on a copy).
# > Fix: `a[a > 0] = -1` or use `np.putmask`.

# %%
t = np.array([1, -2, 3, -4])
t[t > 0][0] = 0        # silent no-op
print(t)               # [ 1 -2  3 -4]
np.putmask(t, t > 0, 0)
print(t)               # [ 0 -2  0 -4]

# %% [markdown]
# **C vs Fortran order:** NumPy stores C-contiguous (row-major) by default.
# `a.flags['C_CONTIGUOUS']` tells you; `np.ascontiguousarray(a)` forces it.
# `a.flatten(order='F')` flattens column-major instead.

# %%
f = np.arange(6).reshape(2, 3)
print("C flatten:", f.flatten(order="C"))   # row by row
print("F flatten:", f.flatten(order="F"))   # column by column

# %% [markdown]
# ---
# ## 14. NumPy tricks (Session 15 highlights)
#
# ▶ Session 15 – Numpy Tricks: https://www.youtube.com/watch?v=jFBpnd0G09o

# %%
# 1. vectorise arbitrary Python functions over arrays
def is_prime(n):
    if n < 2:
        return False
    return all(n % i for i in range(2, int(n ** 0.5) + 1))

print(np.vectorize(is_prime)(np.arange(10)))

# %%
# 2. meshgrid — coordinate grids for heatmaps / decision boundaries
xs, ys = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 2, 3))
print(xs)   # x coordinate of every grid point
print(ys)   # y coordinate of every grid point
print(xs + ys)

# %%
# 3. structured dtypes — mini "table" inside NumPy
emp = np.array([(1, "HR", 5000.0), (2, "IT", 7000.0)],
               dtype=[("id", "i4"), ("dept", "U10"), ("salary", "f8")])
print(emp["salary"], emp[emp["salary"] > 6000]["dept"])

# %%
# 4. memory-efficient idioms
big = np.random.default_rng(0).random(1_000_000)
print("MB:", big.nbytes / 1e6)
smaller = big.astype(np.float32)          # halve memory
print("MB after downcast:", smaller.nbytes / 1e6)

# %%
# 5. `itertools`-style index tricks from NumPy itself
print(np.ndindex(2, 3))                   # iterator over multi-indices
print(np.unravel_index(np.argmax(big), big.shape))
print(np.ravel_multi_index((1, 2), (3, 4)))   # (row, col) -> flat position

# %%
# 6. one-shot stats for interview answers
sample = np.random.default_rng(1).normal(100, 15, 1000)
print("mean/std/median :", sample.mean(), sample.std(ddof=1), np.median(sample))
print("z-score of 130  :", (130 - sample.mean()) / sample.std(ddof=1))
print("IQR             :", np.percentile(sample, 75) - np.percentile(sample, 25))

# %% [markdown]
# ---
# ## 15. Quick revision sheet
#
# ▶ Week 5 – Numpy Interview Questions: https://www.youtube.com/watch?v=wjv01Opo6Bk
#
# | Task | Code |
# |---|---|
# | create 0..9 | `np.arange(10)` |
# | 5 equal points 0..1 | `np.linspace(0, 1, 5)` |
# | shape / size / dtype | `a.shape`, `a.size`, `a.dtype` |
# | reshape with inference | `a.reshape(3, -1)` |
# | boolean filter | `a[a > 0]` |
# | vectorised if/else | `np.where(c, x, y)` |
# | sum per column | `a.sum(axis=0)` |
# | keep reduced dim | `a.sum(axis=1, keepdims=True)` |
# | reproducible RNG | `np.random.default_rng(42)` |
# | matrix multiply | `A @ B` |
# | solve Ax=b | `np.linalg.solve(A, b)` |
# | safe float compare | `np.allclose(a, b)` |
# | NaN-safe mean | `np.nanmean(a)` |
# | sort indices | `np.argsort(a)` |
#
# > **Interview short-list:** broadcasting rules · view vs copy ·
# > `axis` semantics · `arange` vs `linspace` · `argmax` vs `max` ·
# > why `==` fails on floats · overflow of `int8` ·
# > `np.vectorize` is *not* a performance tool (it's a convenience loop).

# %% [markdown]
# ---
# **Next notebook:** `04_pandas.ipynb` — NumPy arrays under a labelled interface.
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
