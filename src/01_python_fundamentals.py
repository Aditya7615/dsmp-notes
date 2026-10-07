# %% [markdown]
# # 01 · Python Fundamentals — Complete Function Reference
#
# This notebook is the foundation layer of the CampusX DSMP 2022-23 notes.
# It covers everything from `print()` and data types to strings, lists, tuples,
# sets, dictionaries, functions and a first look at **time complexity** —
# written as `What / Syntax / Parameters / Returns / Example` blocks, each
# followed by a runnable code cell, so the notebook doubles as a searchable
# function reference.
#
# | You will master | Section |
# |---|---|
# | Output, comments, variables, literals | 1–3 |
# | Operators, control flow, loops | 4–6 |
# | Strings + classic string problems | 7 |
# | Lists / tuples / sets / dicts + problems | 8–11 |
# | Functions, scope, recursion | 12 |
# | Big-O and timing experiments | 13 |
#
# > **Note:** everything below runs top-to-bottom in a fresh kernel
# > (Python 3.10, standard library only, no internet).

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 1 | Session 1 - Python Fundamentals | https://www.youtube.com/watch?v=1z5-O7-5AXk |
# | 2 | Session 2 - Operators + If-Else + Loops | https://www.youtube.com/watch?v=JCkIrdrZEE8 |
# | 3 | Session 3 - Python Strings | https://www.youtube.com/watch?v=6HAu0Y9BjA4 |
# | 4 | Programming Problems on Strings | https://www.youtube.com/watch?v=7ltjqU5iytY |
# | 7 | Session on Time Complexity | https://www.youtube.com/watch?v=J1vdIcg5IZs |
# | 9 | Session 4 - Lists in Python | https://www.youtube.com/watch?v=WmbU3WBaoR0 |
# | 10 | Session 5 - Tuples + Sets + Dictionary | https://www.youtube.com/watch?v=jcQjp11mn1A |
# | 11 | Session 6 - Functions in Python | https://www.youtube.com/watch?v=OOInK25PoFo |

# %%
import sys

print(sys.version_info[:3], "— Python", sys.version.split()[0])
# NOTE: every code cell below runs in this one persistent kernel state

# %% [markdown]
# ---
# ## 1. Output & comments
#
# ▶ Session 1 - Python Fundamentals: https://www.youtube.com/watch?v=1z5-O7-5AXk

# %% [markdown]
# ### `print(*objects, sep=' ', end='\n', flush=False, file=sys.stdout)`
# **What it does:** writes the string representations of objects to an output stream.
# **Syntax:** `print(*objects, sep=' ', end='\n', flush=False, file=sys.stdout)`
# **Parameters:**
# - `*objects` — zero or more values to print (separated by `sep`)
# - `sep` — separator inserted between objects — `str` — default `' '`
# - `end` — string appended after the last object — `str` — default `'\n'`
# - `flush` — force immediate write (bypass the OS buffer) — `bool` — default `False`
# - `file` — a file-like object with a `.write()` method — default `sys.stdout`
# **Returns:** `None` (it is a convenience wrapper, not a value producer).

# %%
print("Plain")
print("A", "B", "C")                       # default sep=' '
print("A", "B", "C", sep="-")              # sep='-'
print("line one", end="")                  # suppress the newline
print(" — same line")                      # ...continues here
print("flushed", flush=True)               # write now, no buffering
print(*[10, 20, 30], sep=", ")             # * unpacks the list into arguments

# %% [markdown]
# ### `print(file=...)` — sending output somewhere else
# **What it does:** `file` accepts anything with `.write(str)` — a real file, an
# `io.StringIO` buffer, `sys.stderr` — which is how you build loggers and
# capture output in tests.

# %%
import io, os, tempfile

# 1) capture into an in-memory buffer
buf = io.StringIO()
print("warning: disk almost full", file=buf)
print("captured:", repr(buf.getvalue()))

# 2) write to a real file (tempfile keeps the repo clean)
path = os.path.join(tempfile.gettempdir(), "print_demo.txt")
with open(path, "w") as f:
    print("first line", file=f)
    print("second line", file=f)
with open(path) as f:
    print("file contains:", f.read().splitlines())
os.remove(path)                             # TIP: always clean up temp files

# %% [markdown]
# ### Comments & docstrings
# - `# comment` — ignored by the interpreter, never stored on the object.
# - `"""docstring"""` — the *first* statement of a module/class/function;
#   stored in `__doc__`, shown by `help()`, picked up by Sphinx/pydoc.
# - `# NOTE:` / `# TIP:` / `# WARNING:` are just conventions that make reviews easy.

# %%
def add(a, b):
    """Return the sum of two numbers.

    Parameters
    ----------
    a, b : int or float
        Inputs to add.
    """
    return a + b

# this is a full-line comment
x = 5  # trailing comment
# print("this never runs")

print(add(2, 3))
print("stored docstring ->", repr(add.__doc__.splitlines()[0]))
print("no such thing as __comment__:", not hasattr(add, "__comment__"))
print(x)

# %% [markdown]
# **Section notes — output:**
#
# | Piece | Remember |
# |---|---|
# | `print(a, b, sep=)` | control the separator |
# | `print(..., end=)` | suppress/replace the trailing newline |
# | `print(..., file=)` | any object with `.write()` works |
# | `print(..., flush=True)` | force immediate write (logs, prompts) |
# | `# comment` | discarded at compile time |
# | `"""docstring"""` | kept in `__doc__`, visible to `help()` |

# %% [markdown]
# ---
# ## 2. Data types & variables
#
# ▶ Session 1 - Python Fundamentals: https://www.youtube.com/watch?v=1z5-O7-5AXk

# %% [markdown]
# ### `type(object)` and `id(object)`
# **What they do:** `type()` returns the class of an object; `id()` returns the
# object's unique identity (its memory address in CPython) valid for its lifetime.
# **Returns:** the `type` object / an `int`.

# %%
print(type(42), type(3.14), type("hi"), type(True), type(None), type([1]))
print("classes:", type(42).__name__, type(3.14).__name__, type(None).__name__)

n = 1000
print("id type:", type(id(n)).__name__, "-> identity is unique per OBJECT, not per value")
pair = (1, 2)
print("pair:", type(pair).__name__, "has an id:", isinstance(id(pair), int))

# %% [markdown]
# ### Variables, identifiers & keywords
# - A **variable** is just a *name bound to an object* (label → object).
# - **Identifier rules:** letters, digits, underscore; cannot start with a digit;
#   cannot be a keyword; case-sensitive; no length limit.
# - **Keywords** are reserved words (`keyword.kwlist`); `match`/`case`/`_` are
#   *soft* keywords since 3.10 — they act as keywords only inside `match`.

# %%
import keyword

print(len(keyword.kwlist), "keywords:", keyword.kwlist)
print("soft keywords (3.10+):", keyword.softkwlist)
print("iskeyword:", keyword.iskeyword("class"), keyword.iskeyword("Class"),
      keyword.iskeyword("forge"))

valid = userName, _temp, camelCase, UPPER2, __dunder = 1, 2, 3, 4, 5
print("valid identifiers:", valid)

for bad in ("1st = 5", "for = 5", "my-var = 5"):
    try:
        exec(bad)
    except SyntaxError as e:
        print(f"SyntaxError for {bad!r}: {e.msg}")

# %% [markdown]
# ### Literals — the fixed values you write in source
# | Type | Literal examples |
# |---|---|
# | `int` | `42`, `0b1010`, `0o17`, `0xFF`, `1_000_000` |
# | `float` | `3.14`, `1e3`, `.5`, `5.` |
# | `complex` | `3+4j` |
# | `str` | `"hi"`, `'hi'`, `r"C:\new"`, `"""multi\nline"""`, `"हि"` |
# | `bool` | `True`, `False` |
# | `NoneType` | `None` — the single "no value" object |

# %%
print(42, 0b1010, 0o17, 0xFF, 1_000_000)      # ints in every base + underscores
print(3.14, 1e3, .5, 5., 1_000.5)             # floats
print(3 + 4j, (3 + 4j).real, (3 + 4j).imag)   # complex
print(True, False, None)

raw = r"C:\new\docs\readme.txt"      # raw string: backslashes are literal
triple = """line one
line two"""                           # triple-quoted: multiline + docstrings
print(raw)
print(triple)
print("unicode:", "\u0939\u093f", "\N{GREEK SMALL LETTER ALPHA}", "\U0001F600")

# %% [markdown]
# ### Dynamic typing — names are labels, not boxes
# Python infers the type at runtime and one name can be re-bound to a
# completely different type. Two names can also point at the *same* object.

# %%
x = 42
print("1:", type(x).__name__, x)
x = "now a string"              # same NAME, brand-new OBJECT
print("2:", type(x).__name__, repr(x))

a = [1, 2, 3]
b = a                           # b is a second label on the SAME list
b.append(4)
print("aliasing:", a, "-> a is b:", a is b)

small1, small2 = 256, 256
print("small-int cache (CPython -5..256):", small1 is small2)      # True
big1, big2 = 1000, 1000
print("'is' on big ints is an implementation detail:", big1 == big2)

# %% [markdown]
# > **Gotcha:** `==` compares **values**, `is` compares **identity**
# > (`id(a) == id(b)`). Never write `if x == None` — use `if x is None`.

# %%
print("[1,2] == [1,2] ->", [1, 2] == [1, 2])          # equal values
print("[1,2] is [1,2] ->", [1, 2] is [1, 2])          # different objects
print("both look the same here:", None is None, None == None)
print("...but `x is None` is the idiomatic check (and faster)")

# %% [markdown]
# **Section notes — types & variables:**
#
# | Concept | Code |
# |---|---|
# | class of a value | `type(x)` |
# | identity / address | `id(x)` |
# | keyword lists | `keyword.kwlist`, `keyword.softkwlist` |
# | keyword test | `keyword.iskeyword("class")` |
# | raw / triple / unicode literals | `r"\n"`, `"""..."""`, `"é"` |
# | value vs identity | `==` vs `is` |

# %% [markdown]
# ---
# ## 3. Input & type conversion
#
# ▶ Session 1 - Python Fundamentals: https://www.youtube.com/watch?v=1z5-O7-5AXk

# %% [markdown]
# ### `input(prompt='')`
# **What it does:** prints `prompt`, reads one line from stdin and returns it
# **always as a `str`** — you must convert explicitly.
# **Returns:** `str` (never an int or float).

# %%
# NOTE: a headless notebook kernel has no interactive stdin, so we script the
# keystrokes by temporarily replacing builtins.input. In a terminal/Colab the
# plain call `age = input("Age: ")` blocks until you press Enter.
import builtins

_real_input = builtins.input
_typed = iter(["Aditya", "25", "yes"])

def _scripted_input(prompt=""):
    value = next(_typed)
    print(f"{prompt}{value}")        # show the prompt + what the user "typed"
    return value

builtins.input = _scripted_input
try:
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    again = input("Continue? (yes/no): ")
finally:
    builtins.input = _real_input     # restore normal input()

print("name  :", name, type(name).__name__)
print("age   :", age, type(age).__name__)     # NOTE: still a str!
print("age+1 :", int(age) + 1)                # convert before arithmetic
print("again :", again.lower() == "yes")

# %% [markdown]
# ### Conversion functions
#
# | Function | What it does | Fails on |
# |---|---|---|
# | `int(x, base=10)` | string/float → int (floats are truncated) | `"3.9"`, `"abc"` |
# | `float(x)` | string/int → float | `"3,5"`, `"abc"` |
# | `str(x)` | any object → its display string | never |
# | `bool(x)` | truthiness → `True`/`False` | never |
# | `chr(n)` / `ord(c)` | int ↔ unicode code point | n > 0x10FFFF |
# | `oct(x)` / `hex(x)` / `bin(x)` | int → base-8/16/2 string | never |
# | `eval(expr)` | **executes** a Python expression | syntax errors |
# | `ast.literal_eval(s)` | parses literals **safely** | anything non-literal |

# %%
print(int("42"), int("  42  "), int("-7"), int(True))
print(int("0b1010", 2), int("ff", 16), int("777", 8))   # base-aware parsing
print(int(3.99), int(-3.99))                            # truncates toward zero
print(float("3.5"), float("1e3"), float(7), float("nan"))
print(str(42), str(3.5), repr(str(True)), repr(str(None)))
print("bool gotcha:", bool(""), bool(0), bool([]), bool("False"), bool("hi"))
#          NOTE: any non-empty string is truthy — even "False" and "0"

for bad in ("3.9", "abc"):
    try:
        int(bad)
    except ValueError as e:
        print("ValueError:", bad, "->", e)

# %%
print(chr(72), ord("H"), chr(ord("a") + 2))     # 'H', 72, 'c'
print(oct(255), hex(255), bin(10), hex(ord("A")))
print("emoji code point:", ord("\U0001F600"), chr(0x1F600))

# %%
import ast

print(eval("2 + 3 * 4"))                     # arithmetic
print(eval("'dsmp'.upper()"))                # method call
print(eval("sum([1, 2, 3])"))                # function call
print(ast.literal_eval("{'a': 1, 'b': [2, 3]}"))   # dict literal
print(ast.literal_eval("(10, 20)"))                # tuple literal
print(ast.literal_eval("1 + 2j"))                  # complex literal works

try:
    ast.literal_eval("1 + 2")        # plain arithmetic is NOT accepted (3.10)
except ValueError:
    print("literal_eval rejects '1 + 2' — only literals, no operations")

# WARNING: eval runs arbitrary code — never feed it untrusted user input.
print("eval executes anything ->", eval("__import__('os').getcwd() is not None"))
try:
    ast.literal_eval("__import__('os').getcwd()")
except ValueError as e:
    print("literal_eval refuses code:", type(e).__name__, "-", e)

# %% [markdown]
# > **Gotcha:** `bool("False")` and `bool("0")` are `True` because they are
# > non-empty strings. To parse user yes/no text, compare explicitly:
# > `ans.strip().lower() in ("y", "yes", "1")`.
#
# > **Gotcha:** `input()` + `int()` is the classic `ValueError` crash — guard
# > it with `try/except` (covered properly in notebook 02, section 10).

# %%
answer = "No"
print("parsed as bool:", answer.strip().lower() in ("y", "yes", "true", "1"))

# %% [markdown]
# **Section notes — input & conversion:**
#
# | Task | Code |
# |---|---|
# | read a line | `s = input("...")` (always `str`) |
# | string → int/float | `int(s)`, `float(s)` |
# | number → string | `str(n)` |
# | base conversion | `int("ff", 16)`, `bin/hex/oct(n)` |
# | char ↔ code point | `chr(65)`, `ord("A")` |
# | safe literal parsing | `ast.literal_eval(s)` — never `eval(...)` on user data |

# %% [markdown]
# ---
# ## 4. Operators
#
# ▶ Session 2 - Operators + If-Else + Loops: https://www.youtube.com/watch?v=JCkIrdrZEE8

# %%
# arithmetic
print(7 + 3, 7 - 3, 7 * 3, 7 / 3)        # / always returns float
print(7 // 3, 7 % 3, 7 ** 3)             # floor-div, modulo, power
print(divmod(7, 3))                      # (quotient, remainder)
print(-7 // 3, -7 % 3)                   # floor goes toward -inf; % keeps divisor's sign
print(7 / 2 == 3.5, int(7 / 2))          # true division vs truncation

# %%
# comparison — comparisons CHAIN
print(5 == 5, 5 != 6, 5 > 4, 5 <= 5)
print(1 < 2 < 3 < 4)                     # chained: (1<2) and (2<3) and (3<4)
print("ab" < "ac", [1, 2] < [1, 3], (1, 2) < (1, 3))   # lexicographic / element-wise

# %%
# logical operators with SHORT-CIRCUIT evaluation
def left():
    print("  left evaluated")
    return True

def right():
    print("  right evaluated")
    return False

print("or :")
left() or right()        # right() never runs (left is already True)
print("and:")
right() and left()       # left() never runs (right is already False)
# TIP: short-circuit powers idioms like `value or default` and `x is not None and x.foo`

name = ""
print("fallback:", name or "anonymous")                # '' is falsy -> default
cache = {"a": 1}
print("cached  :", cache.get("a") or "compute")        # careful: 0/''/None also fall back

# %%
# assignment operators (augmented assignment desugars to a = a OP b)
count = 10
count += 5;  print("+=", count)
count -= 3;  print("-=", count)
count *= 2;  print("*=", count)
count //= 4; print("//=", count)
count %= 3;  print("%=", count)
count **= 2; print("**=", count)
seq = [1]; seq += [2]; print("list += appends in place:", seq)

# walrus operator := (3.8+) assigns inside an expression
data = [1, 5, 9]
if (n := len(data)) > 2:
    print("walrus: length", n, "computed once, usable in the condition")

# %%
# identity operators  vs  equality operators
a = [1, 2]
b = [1, 2]
c = a
print("a == b:", a == b, "| a is b:", a is b, "| a is c:", a is c)
print("a is not b:", a is not b)
print("None identity:", a is None, a == None)     # both False, but 'is None' is idiomatic

# %%
# membership operators
print(3 in [1, 2, 3], "el" in "hello", "b" in {"a", "b"}, 4 not in [1, 2])
print(("a", 1) in {"a": 1, "b": 2})               # 'in' on a dict checks KEYS

# %%
# bitwise operators — integers as infinite two's-complement bit strings
x, y = 0b1100, 0b1010                  # 12, 10
print(f"x={x:08b}  y={y:08b}")
print("&  AND  :", bin(x & y))         # 0b1000 — bits set in BOTH
print("|  OR   :", bin(x | y))         # 0b1110 — bits set in EITHER
print("^  XOR  :", bin(x ^ y))         # 0b0110 — bits set in exactly one
print("~  NOT  :", bin(~x), "->", ~x)  # -13   — (~n == -(n+1))
print("<< LEFT :", bin(x << 2))        # multiply by 4
print(">> RIGHT:", bin(x >> 1))        # divide by 2 (floor)
print("parity trick (5 ^ 3) & 1 =", (5 ^ 3) & 1)   # 1 = odd result

# %% [markdown]
# ### Operator precedence (highest → lowest)
#
# | Level | Operators |
# |---|---|
# | 1 | `()` parentheses — always win, use them liberally |
# | 2 | `**` exponent (right-associative: `2**3**2 == 2**9`) |
# | 3 | `+x`, `-x`, `~x` unary |
# | 4 | `*`, `@`, `/`, `//`, `%` |
# | 5 | `+`, `-` binary |
# | 6 | `<<`, `>>` |
# | 7 | `&` |
# | 8 | `^` |
# | 9 | `\|` |
# | 10 | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `in`, `not in` |
# | 11 | `not` |
# | 12 | `and` |
# | 13 | `or` |
# | 14 | `if … else` (ternary), `lambda`, assignment forms |
#
# > **Gotcha:** `not a == b` means `not (a == b)`; `a is not b` is **one**
# > operator, not `a is (not b)`.

# %%
print("2**3**2 =", 2 ** 3 ** 2)           # right-assoc -> 512
print("not 1 == 2 ->", not 1 == 2)        # not (1 == 2)
print("a is not b ->", [1] is not [1])    # identity comparison
print("(1+2)*3 =", (1 + 2) * 3, "vs 1+2*3 =", 1 + 2 * 3)

# %% [markdown]
# **Section notes — operators:**
#
# | Group | Operators |
# |---|---|
# | Arithmetic | `+ - * / // % **`, `divmod()` |
# | Comparison | `== != < <= > >=` (chains allowed) |
# | Logical | `and or not` (short-circuit) |
# | Assignment | `= += -= *= /= //= %= **= :=` |
# | Identity | `is`, `is not` |
# | Membership | `in`, `not in` |
# | Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |

# %% [markdown]
# ---
# ## 5. Control flow
#
# ▶ Session 2 - Operators + If-Else + Loops: https://www.youtube.com/watch?v=JCkIrdrZEE8

# %%
# if / elif / else — indentation (4 spaces) IS the block structure
marks = 78
if marks >= 90:
    grade = "A"
elif marks >= 75:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "F"
print("grade:", grade)

# nested if
n = -4
if n != 0:
    if n > 0:
        sign = "positive"
    else:
        sign = "negative"
else:
    sign = "zero"
print("sign:", sign)

# %%
# ternary / conditional expression: value_if_true if condition else value_if_false
age = 20
status = "adult" if age >= 18 else "minor"
print(status)
print("even" if 4 % 2 == 0 else "odd")
print("nested:", ("big" if age > 10 else "small") if age else "unknown")

# %%
# pass — a syntactically required no-op (placeholders, stubs, empty classes)
class TODO:
    pass                                   # body coming later

def todo_function():
    pass

for i in range(3):
    if i != 1:
        pass                               # does nothing, satisfies syntax
print("pass keeps the block valid:", TODO.__name__, callable(todo_function))

# %%
# match / case — structural pattern matching (Python 3.10+)
def http_status(code):
    match code:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500 | 502 | 503:              # or-pattern
            return "Server Error"
        case c if 400 <= c < 500:          # guard
            return f"Client Error ({c})"
        case _:
            return "Unknown"

for code in (200, 404, 503, 418, 100):
    print(code, "->", http_status(code))

# sequence patterns decompose tuples/lists
def classify(point):
    match point:
        case (0, 0):
            return "origin"
        case (x, 0):
            return f"on x-axis at {x}"
        case (0, y):
            return f"on y-axis at {y}"
        case (x, y):
            return f"point ({x}, {y})"

print(classify((0, 0)), "|", classify((3, 0)), "|", classify((1, 2)))

# mapping patterns decompose dicts
info = {"type": "user", "name": "Ada"}
match info:
    case {"type": "user", "name": n}:
        print("mapping pattern -> user", n)

# %% [markdown]
# > **Gotcha:** `match` is *not* a switch statement — it is structural
# > pattern matching. Cases are checked top-to-bottom; `case _:` is the
# > catch-all (a wildcard *name*, not an exception).
#
# **Section notes — control flow:**
#
# | Form | Template |
# |---|---|
# | branch | `if cond: … elif cond: … else: …` |
# | ternary | `a if cond else b` |
# | no-op | `pass` |
# | pattern match | `match value: case pat: … case _:` |
# | guard | `case pat if extra_cond:` |

# %% [markdown]
# ---
# ## 6. Loops
#
# ▶ Session 2 - Operators + If-Else + Loops: https://www.youtube.com/watch?v=JCkIrdrZEE8

# %%
# for loop over any iterable (list, str, dict, set, tuple)
for ch in "DSMP":
    print(ch, end=" ")
print()

fruits = ["apple", "banana", "cherry"]
for f in fruits:
    print(f.upper(), end=" | ")
print()

# while loop + loop control keywords
n, total = 1, 0
while n <= 5:
    total += n
    n += 1
print("while sum 1..5 =", total)

for i in range(1, 8):
    if i % 2 == 0:
        continue            # skip evens
    if i > 5:
        break               # stop entirely
    print("odd:", i, end=" ")
print()

# %%
# range(start, stop, step) — half-open [start, stop)
print(list(range(5)))            # start defaults to 0
print(list(range(2, 11)))        # 2..10
print(list(range(0, 10, 2)))     # positive step
print(list(range(10, 0, -2)))    # negative step
print(len(range(0, 10, 3)))      # range knows its own length: 4
print(7 in range(0, 10, 3))      # membership without building a list
print(list(range(0)))            # empty range -> []

# %%
# for-else: the else block runs ONLY if the loop finished without break
for n in [2, 4, 6]:
    if n % 2 != 0:
        break
else:
    print("for-else: no odd number found (loop never broke)")

for n in [2, 3, 4]:
    if n % 2 != 0:
        print("broke on", n)
        break
else:
    print("this line is skipped when we break")

# %%
# enumerate / zip / reversed — the three loop workhorses
letters = ["a", "b", "c"]

for i, ch in enumerate(letters, start=1):     # index + value at once
    print(f"{i}:{ch}", end=" ")
print()
print("list(enumerate):", list(enumerate(letters)))

sizes = [10, 20, 30]
for ch, size in zip(letters, sizes):          # walks iterables in parallel
    print(f"{ch}={size}", end=" ")
print()
print("list(zip):", list(zip(letters, sizes)))
print("zip stops at the shortest:", list(zip([1, 2, 3], ["a", "b"])))
# TIP: zip(*matrix) transposes; zip(a, b, strict=True) (3.10+) errors on length mismatch

print("reversed :", list(reversed(letters)))  # reverse order, no copy of the source
print("sorted   :", sorted(letters, reverse=True))   # new sorted list

# %%
# iterating over a dict — keys by default, or explicitly items/keys/values
person = {"name": "Ada", "age": 36, "city": "London"}
for k in person:
    print("key  :", k)
for k, v in person.items():
    print("pair :", k, "=", v)
for v in person.values():
    print("value:", v)

# %%
# nested loops — every inner pass runs once per outer iteration (O(n*m))
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i}x{j}={i*j}", end="\t")
    print()                                  # newline after each row

# pattern printing: right triangle of stars
for row in range(1, 5):
    print("*" * row)

# %% [markdown]
# > **Gotcha:** mutating the collection you are looping over (especially with
# > `remove()`) silently skips elements — iterate over a **copy**
# > (`for x in lst[:]`) or build a new list with a comprehension.
#
# **Section notes — loops:**
#
# | Tool | Use when |
# |---|---|
# | `for x in iterable` | you know the elements |
# | `range(a, b, c)` | you need indices / a counted loop |
# | `while cond` | the iteration count depends on a condition |
# | `break` / `continue` / `pass` | stop / skip / placeholder |
# | `for … else` | "did anything break?" checks |
# | `enumerate(seq, start=)` | you need the index too |
# | `zip(a, b)` | walk two (or more) sequences together |
# | `reversed(seq)` | reverse without copying |

# %% [markdown]
# ---
# ## 7. Strings
#
# ▶ Session 3 - Python Strings: https://www.youtube.com/watch?v=6HAu0Y9BjA4
#
# ▶ Programming Problems on Strings: https://www.youtube.com/watch?v=7ltjqU5iytY
#
# A `str` is an **immutable** sequence of Unicode code points: you can read,
# slice and search it, but every "modification" returns a *new* string.

# %%
s = "DataScience"
print("value:", s, "| len:", len(s))

# indexing: 0-based from the left, -1 from the right
print("first:", s[0], "| last:", s[-1], "| second last:", s[-2])
print("index 4:", s[4])

# slicing: s[start:stop:step] — stop EXCLUDED, negative indices allowed
print("s[0:4]  :", s[0:4])       # 'Data'
print("s[4:]   :", s[4:])        # 'Science'
print("s[:4]   :", s[:4])
print("s[::2]  :", s[::2])       # every 2nd char
print("s[::-1] :", s[::-1])      # REVERSE
print("s[-6:-1]:", s[-6:-1])     # 'Scienc'
print("s[1:8:3]:", s[1:8:3])     # start/stop/step together
print("s[20:]  :", repr(s[20:])) # out-of-range SLICES are safe -> ''

try:
    s[0] = "X"
except TypeError as e:
    print("immutable! TypeError:", e)

# %% [markdown]
# ### Case & whitespace — `upper() lower() title() capitalize() swapcase() strip()`

# %%
word = "  Hello World  "
print(repr(word.upper()), repr(word.lower()))
print(repr(word.title()), repr(word.capitalize()), repr(word.swapcase()))
print(repr(word.strip()), repr(word.lstrip()), repr(word.rstrip()))
print(repr("--hi--".strip("-")), repr("**bold**".strip("*")))   # strip any chars
print("'Data Science'.title() ->", "Data Science".title())
print("'hELLO'.capitalize() ->", "hELLO".capitalize())
# NOTE: strip() cleans BOTH ends; lstrip/rstrip only one side
# TIP: canonical form for comparisons: s.strip().lower()

# %%
# splitting & joining
print("a,b,c".split(","))
print("a,b,,c".split(","))                 # empty fields are preserved
print("a b  c".split())                    # None = whitespace, collapses runs
print("a,b,c,d".split(",", 2))             # maxsplit from the LEFT
print("a,b,c,d".rsplit(",", 1))            # maxsplit from the RIGHT
print("l1\nl2\r\nl3".splitlines())         # handles \n, \r\n and \r
print(",".join(["a", "b", "c"]))           # join is THE O(n) way to build strings
print("".join("abc"[::-1]))
try:
    ", ".join([1, 2])
except TypeError as e:
    print("join needs strings:", e)

# %%
# searching & replacing
print("'banana'.find('an')  ->", "banana".find("an"))    # index or -1
print("'banana'.rfind('an') ->", "banana".rfind("an"))   # rightmost match
print("'banana'.find('zz')  ->", "banana".find("zz"))    # not found = -1 (no error)
print("'banana'.count('an') ->", "banana".count("an"))
print("'aXaX'.replace('X', '-')    ->", "aXaX".replace("X", "-"))
print("'aXaX'.replace('X', '-', 1) ->", "aXaX".replace("X", "-", 1))   # count=1
print("'hello world'.replace('world', 'there') ->",
      "hello world".replace("world", "there"))

try:
    "banana".index("zz")
except ValueError as e:
    print("index() RAISES instead of returning -1:", e)
print("'banana'.index('an')  ->", "banana".index("an"))
print("'banana'.rindex('an') ->", "banana".rindex("an"))

# %%
# prefix / suffix checks (accept tuples — great for file extensions)
print("readme.txt".startswith("read"), "readme.txt".startswith("#"))
print("readme.txt".endswith((".txt", ".md", ".rst")))     # tuple of options
print("photo.png".lower().endswith(".png"))

# %%
# character classification — always case-sensitive
tests = ["abc", "a1", "123", "   ", "Title Case", "UPPER", "lower", "mixedCase"]
hdr = f"{'word':<14}{'alpha':>6}{'digit':>7}{'alnum':>7}{'space':>7}{'title':>7}{'upper':>7}{'lower':>7}"
print(hdr)
for w in tests:
    print(f"{w!s:<14}{str(w.isalpha()):>6}{str(w.isdigit()):>7}"
          f"{str(w.isalnum()):>7}{str(w.isspace()):>7}{str(w.istitle()):>7}"
          f"{str(w.isupper()):>7}{str(w.islower()):>7}")

print("'2'.isdigit() ->", "2".isdigit(), "| '²'.isdigit() ->", "²".isdigit())
print("ID check:", "AB12".isalnum() and "AB12".upper() == "AB12")

# %%
# alignment & padding: center / ljust / rjust / zfill
print(repr("hi".center(10, "-")))       # ----hi----
print(repr("hi".ljust(10, ".")))        # hi........
print(repr("hi".rjust(10, ".")))        # ........hi
print(repr("42".zfill(5)))              # 00042
print(repr("hi".center(9)))             # default fill = space
print(f"{'name':<10}{'qty':>5}")         # f-strings do the same for tables
print(f"{'ada':<10}{'7':>5}")

# %%
# partition / rpartition — split into exactly 3 parts (head, sep, tail)
print("name=sheet".partition("="))       # ('name', '=', 'sheet')
print("a=b=c".partition("="))            # first separator wins
print("a=b=c".rpartition("="))           # last separator wins
print("nosplit".partition("="))          # ('nosplit', '', '') — never raises
print("config.yaml".rpartition("."))     # ('config', '.', 'yaml')

# %%
# prefix/suffix editing (3.9+), tabs and encoding
print("TestCLI.py".removeprefix("Test"))
print("config.yaml".removesuffix(".yaml"))
print(repr("no match".removeprefix("zzz")))
print(repr("a\tb\tc".expandtabs(4)))     # tabs -> spaces of width 4
print(repr("a\tb".expandtabs(0)))

data = "héllo 🙂".encode("utf-8")
print("encode ->", data)
print("decode ->", data.decode("utf-8"))
print("bytes are just ints:", list("hi".encode()))
try:
    "héllo".encode("ascii")
except UnicodeEncodeError as e:
    print("UnicodeEncodeError:", e.object, "->", e.reason)

# %% [markdown]
# ### `str.format()` — indexed & named placeholders with format specs
# **Syntax:** `"{:[[fill]align][sign][#][0][width][,][.precision][type]}".format(...)`

# %%
print("{} {} {}".format("a", "b", "c"))
print("{0} {1} {0}".format("x", "y"))                     # indices, reuse
print("{name} is {age} years old".format(name="Ada", age=36))   # keywords
print("{:>8}|{:<8}|{:^8}|{:*^8}".format("hi", "hi", "hi", "hi"))
print("{:.2f}|{:.3f}|{:e}".format(3.14159, 1 / 3, 12345.678))
print("{:,}|{:#x}|{:#o}|{:#b}".format(1234567, 255, 8, 10))
print("{:>10}".format("right"), "|", "{:<10}".format("left"))

# %%
# f-strings (3.6+) — expressions evaluated at runtime, the default choice
name, age = "Ada", 36
print(f"{name} is {age} years old")
print(f"{name:>10}|{name:<10}|{name:^10}|{name:*^10}")
print(f"{age=}")                              # debug specifier -> 'age=36'
print(f"{name.upper()=}")                     # expression + = together
print(f"{3.14159:.2f} | {2 ** 10:,} | {255:#x} | {0.852:.1%}")
print(f"{1234567.891:<15,.2f}|")              # width + thousands + precision
print(f"expr: {2 + 3 * 4}, nested: {[x * 2 for x in range(3)]}")
print(f"{'quoted':{'>10'}}")                  # dynamic spec (built at runtime)

# %%
# %-formatting — the old printf style, still common in legacy code
print("%s is %d years old" % ("Ada", 36))
print("%-10s|%10d|%06.2f|" % ("hi", 42, 3.14159))
print("%(name)s scored %(marks)d" % {"name": "Ada", "marks": 95})
print("%% escapes the percent sign ->", "100%%")

# %% [markdown]
# > **Gotcha:** `"{:.2f}".format(x)` and `f"{x:.2f}"` are equivalent — but
# > f-strings win on readability and speed. `str.format` is still needed when
# > the format string itself comes from data (translation templates).

# %% [markdown]
# ### Classic string problems (Session 4)
# reverse · palindrome · anagram · character frequency · most common character ·
# substring · rotation

# %%
# 1) reverse a string — three ways
s = "campus"
print(s[::-1])                      # slicing with step -1 (idiomatic)
print("".join(reversed(s)))         # reversed() iterator + join
rev = ""
for ch in s:
    rev = ch + rev
print(rev)                          # manual (O(n^2) — avoid in interviews)

# %%
# 2) palindrome check — ignore case and non-alphanumerics
def is_palindrome(text: str) -> bool:
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]

for word in ("level", "Madam", "A man, a plan, a canal: Panama", "hello"):
    print(f"{word!r:35} -> {is_palindrome(word)}")

print("quick check:", "abba" == "abba"[::-1])

# %%
# 3) anagram check — same characters with the same frequencies
def are_anagrams(a: str, b: str) -> bool:
    a = "".join(ch.lower() for ch in a if ch.isalnum())
    b = "".join(ch.lower() for ch in b if ch.isalnum())
    return sorted(a) == sorted(b)                    # O(n log n)

print(are_anagrams("listen", "silent"), are_anagrams("hello", "world"))
print("frequency version:", sorted("triangle") == sorted("integral"))

# %%
# 4) character frequency — manual dict vs collections.Counter
from collections import Counter

text = "banana"
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1                  # get() avoids KeyError
print("manual  :", freq)

print("Counter :", Counter(text))                    # {'a': 3, 'n': 2, 'b': 1}
print("on words:", Counter("the cat sat on the mat".split()))

# %%
# 5) most common character
text = "abracadabra"
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1
most = max(freq, key=freq.get)                      # key= ranks by value
print("most common (max)     :", most, "->", freq[most])
print("most common (Counter) :", Counter(text).most_common(1))
print("top 2                 :", Counter(text).most_common(2))

# %%
# 6) substring & 7) rotation checks
haystack = "DataScience"
print("'ence' in haystack ->", "ence" in haystack)          # membership = substring
print("find ->", haystack.find("ence"), "| rfind ->", haystack.rfind("ence"))
print("case-insensitive:", "data" in haystack.lower())

def is_rotation(a: str, b: str) -> bool:
    """b is a rotation of a iff lengths match and b appears inside a+a."""
    return len(a) == len(b) and b in (a + a)

print(is_rotation("abcde", "cdeab"), is_rotation("abcde", "abced"))
print(is_rotation("waterbottle", "erbottlewat"))

# %% [markdown]
# **Section notes — strings:**
#
# | Category | Methods |
# |---|---|
# | case | `upper lower title capitalize swapcase` |
# | whitespace | `strip lstrip rstrip` |
# | split/join | `split rsplit splitlines join partition rpartition` |
# | search | `find rfind index rindex count startswith endswith` |
# | test | `isalpha isdigit isalnum isspace istitle isupper islower` |
# | pad | `center ljust rjust zfill expandtabs` |
# | edit | `replace removeprefix removesuffix encode` |
# | format | `format()`, f-strings, `%` formatting |
#
# > **Interview short-list:** slicing step `[::-1]` · `find` vs `index` ·
# > `split(maxsplit)` vs `rsplit` · why strings are immutable ·
# > anagram via `sorted` or `Counter` · palindrome with `isalnum()` cleanup.

# %% [markdown]
# ---
# ## 8. Lists
#
# ▶ Session 4 - Lists in Python: https://www.youtube.com/watch?v=WmbU3WBaoR0

# %%
# creation
empty = []
from_iter = list("abc")            # ['a', 'b', 'c']
from_tuple = list((1, 2, 3))
repeated = [0] * 5                  # [0, 0, 0, 0, 0]
ranged = list(range(5))             # [0, 1, 2, 3, 4]
nested = [[1, 2], [3, 4], [5, 6]]
mixed = [1, "two", 3.0, True, None, [4]]
print(empty, from_iter, from_tuple, repeated, ranged)
print(nested, mixed)

# %%
# slicing + mutability — lists ARE modifiable in place
nums = [10, 20, 30, 40, 50, 60]
print("basic   :", nums[1], nums[-1], nums[1:4], nums[::2], nums[::-1])
nums[0] = 99                       # single element
nums[1:3] = [200, 201, 202]        # slice assignment may CHANGE the length
print("modified:", nums)
del nums[0]                        # remove by index
print("del [0] :", nums)
del nums[2:4]                      # delete a slice
print("del 2:4 :", nums)
print("slice tricks:", nums[:], nums[1:], nums[:-1])
copy_slice = nums[:]               # a[:] is a SHALLOW copy
copy_slice.append(0)
print("original untouched:", nums, "| copy:", copy_slice)

# %%
# append vs extend — the #1 list interview question
lst = ["a", "b"]
lst.append(["c", "d"])             # append adds ONE element (the list itself)
print("append(list) ->", lst)

lst = ["a", "b"]
lst.extend(["c", "d"])             # extend splices each element in
print("extend(list) ->", lst)
lst.extend("ef")                   # strings are iterable too!
print("extend(str)  ->", lst)
lst += ["g"]                       # += is extend for lists
print("+=           ->", lst)

# %%
# insert / remove / pop / clear / index / count
fruits = ["apple", "banana", "cherry"]
fruits.insert(1, "blueberry")      # insert(index, item) — O(n), shifts the tail
print("insert :", fruits)
fruits.remove("blueberry")         # remove BY VALUE (first occurrence) — O(n)
print("remove :", fruits)
popped = fruits.pop()              # pop() last element, returns it
popped0 = fruits.pop(0)            # pop(index) by position
print("popped :", popped, popped0, "| now:", fruits)

fruits = ["apple", "banana", "cherry"]   # fresh list for the search demos
print("index  :", fruits.index("cherry"))                  # value -> position
print("count  :", fruits.count("apple"), fruits.count("kiwi"))  # missing = 0
try:
    fruits.remove("kiwi")
except ValueError as e:
    print("remove missing ->", e)
try:
    fruits.index("kiwi")
except ValueError as e:
    print("index missing ->", e)

fruits.clear()
print("cleared:", fruits)

# %%
# sort: in-place list.sort() vs returning sorted()
nums = [5, 2, 9, 1, 5]
result = sorted(nums)              # returns a NEW list
print("sorted() ->", result, "| original:", nums)          # original untouched
print("sort() returns:", nums.sort() is None)              # IN PLACE -> None
print("sort() result :", nums)
nums.sort(reverse=True)
print("descending    :", nums)

words = ["banana", "Apple", "pie", "cherry"]
print("ASCII order  :", sorted(words))            # uppercase before lowercase
print("case-insens  :", sorted(words, key=str.lower))
print("by length    :", sorted(words, key=len))
print("tuples by #2 :", sorted([(1, 5), (2, 1), (3, 3)], key=lambda t: t[1]))
print("stable sort keeps equal keys in input order")

# %%
# reverse / copy
stack = [1, 2, 3]
stack.reverse()                    # in place
print("reverse()  :", stack)
print("reversed() :", list(reversed(stack)))     # iterator, no mutation
original = [1, 2, 3]
clone = original.copy()            # shallow copy == list(original) == original[:]
clone.append(4)
print("copy(): original", original, "| clone", clone)

# %%
# list comprehension — build lists declaratively (faster than append loops)
print("squares  :", [x * x for x in range(8)])
print("odds     :", [x for x in range(10) if x % 2 == 1])
print("uppercase:", [w.upper() for w in ["hi", "there"]])
print("nested   :", [[i * 3 + j for j in range(3)] for i in range(3)])
print("pairs    :", [(i, j) for i in range(3) for j in range(2)])   # flattened double loop
print("if-else  :", ["even" if x % 2 == 0 else "odd" for x in range(5)])
print("from str :", [int(ch) for ch in "4021"])

# %% [markdown]
# > **Gotcha:** comprehension syntax is `expression for x in … [if …]` — the
# > `if` there is a *filter* placed after the loop; a real `if/else` belongs
# > *before* the `for`.

# %%
# unpacking
first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)
head, tail = [1], []               # works with any iterable of the right size
a, b = 1, 2
a, b = b, a                        # swap without a temp variable
print("swapped:", a, b)
only_head, *_ = [9, 8, 7]
print("only_head:", only_head)
print("star in expressions:", [*[1, 2], *[3, 4]], {**{"a": 1}, **{"b": 2}})

# %%
# shallow copy gotcha — nested objects are shared until you deepcopy
import copy

original = [[1, 2], [3, 4]]
shallow = copy.copy(original)      # or original[:] / list(original)
deep = copy.deepcopy(original)     # fully independent clone

shallow[0][0] = 99
print("after shallow edit, original:", original)   # CHANGED — inner list shared!
deep[0][0] = 77
print("after deep edit,   original:", original)    # unchanged
print("shallow:", shallow, "| deep:", deep)

# %% [markdown]
# ### List problems

# %%
# P1) second largest element (distinct values)
nums = [10, 5, 20, 8, 20, 3]
print("sorted unique approach:", sorted(set(nums))[-2])

second = float("-inf")
largest = float("-inf")
for x in nums:                     # single pass, O(n)
    if x > largest:
        second, largest = largest, x
    elif x > second and x != largest:
        second = x
print("single pass approach  :", second)

# %%
# P2) remove duplicates while PRESERVING insertion order
seen = set()
ordered_unique = []
for x in [1, 2, 1, 3, 2, 4, 1]:
    if x not in seen:
        seen.add(x)
        ordered_unique.append(x)
print("order preserved:", ordered_unique)
print("no order need  :", list({1, 2, 1, 3, 2, 4, 1}))                  # set — arbitrary order
print("one-liner      :", list(dict.fromkeys([1, 2, 1, 3, 2, 4, 1])))   # TIP: best idiom

# %%
# P3) flatten ONE level of nesting
nested = [[1, 2], [3, [4, 5]], 6]
flattened = []
for item in nested:
    if isinstance(item, list):
        flattened.extend(item)
    else:
        flattened.append(item)
print("flatten one level:", flattened)
flat_simple = [[1, 2], [3, 4], 5]
print("comprehension    :",
      [x for item in flat_simple for x in (item if isinstance(item, list) else [item])])

# %%
# P4) pair with a given sum — two-pointer technique (input must be sorted)
def pair_sum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return arr[left], arr[right]
        if current < target:
            left += 1               # need a bigger sum
        else:
            right -= 1              # need a smaller sum
    return None

sorted_nums = [1, 3, 4, 6, 8, 9]
print("target 13 ->", pair_sum(sorted_nums, 13))
print("target 7  ->", pair_sum(sorted_nums, 7))
print("target 100->", pair_sum(sorted_nums, 100))

# brute force for comparison: O(n^2)
found = next(((a, b) for i, a in enumerate(sorted_nums)
              for b in sorted_nums[i + 1:] if a + b == 13), None)
print("brute force ->", found)

# %% [markdown]
# **Section notes — lists:**
#
# | Method | Notes |
# |---|---|
# | `lst.append(x)` | one element (even when x is a list) |
# | `lst.extend(iterable)` | splice elements in (`+=` same) |
# | `lst.insert(i, x)` | O(n) shift |
# | `lst.remove(x)` / `lst.pop([i])` | by value / by index, returns the item |
# | `lst.index(x)` / `lst.count(x)` / `lst.clear()` | search / tally / empty |
# | `lst.sort(key=, reverse=)` | **in place**, returns `None` |
# | `sorted(lst, key=)` | new list, works on anything iterable |
# | `lst.copy()` | shallow copy (`copy.deepcopy` for nesting) |
#
# > **Interview short-list:** `append` vs `extend` · in-place `sort` vs
# > `sorted` · shallow vs deep copy · a slice is a copy · comprehension
# > filter-vs-ternary syntax · two-pointer pair sum.

# %% [markdown]
# ---
# ## 9. Tuples
#
# ▶ Session 5 - Tuples + Sets + Dictionary: https://www.youtube.com/watch?v=jcQjp11mn1A

# %%
# creation, packing & unpacking
t = (1, 2, 3)                      # parentheses (optional)
t2 = 1, 2, 3                       # the comma alone makes a tuple
single = (42,)                     # trailing comma is MANDATORY for 1-tuples
print(t, t2, t == t2, type(t).__name__, single, type(single).__name__)
print("bare (42) is just an int:", type((42)).__name__)

a, b, c = t                        # unpacking (packing = building from values)
print("unpacked:", a, b, c)
first, *rest = t                   # star capture
print("first/rest:", first, rest)

# %%
# immutability
try:
    t[0] = 99
except TypeError as e:
    print("TypeError:", e)
try:
    t.append(4)
except AttributeError as e:
    print("AttributeError:", e)
print("slicing still works (returns a tuple):", t[1:], type(t[1:]).__name__)

# only two methods: count & index — tuples never grow or shrink
print("count:", (1, 2, 1, 1).count(1), "| index:", (1, 2, 3).index(3))
try:
    (1, 2, 3).index(9)
except ValueError as e:
    print("index miss ->", e)

# %%
# why tuples matter: hashable -> valid dict keys & set members
coords = {(1, 2): "A", (3, 4): "B"}
print("tuple key lookup:", coords[(3, 4)])
print("tuples are hashable:", hash((1, 2)) is not None)
try:
    hash([1, 2])
except TypeError as e:
    print("lists are NOT hashable:", e)
try:
    {(1, 2): "x", (3, 4): "y"}[()]
except KeyError as e:
    print("KeyError for missing tuple key:", e)

# TIP: return multiple values as a tuple — no class needed
def min_max(values):
    return min(values), max(values)

lo, hi = min_max([3, 1, 4, 1, 5])
print("min/max returned as tuple:", lo, hi)

# %%
# namedtuple — a tuple with readable field names (from collections)
from collections import namedtuple

Point = namedtuple("Point", "x y")          # also accepts ["x", "y"] or "x, y"
p = Point(3, 4)
print(p, "| fields:", p.x, p.y)
print("still a tuple:", p[0] + p[1], len(p), tuple(p))
print("_asdict:", p._asdict(), "| _replace:", p._replace(x=0))
Student = namedtuple("Student", "name marks")
print(Student("Ada", 95)._asdict())

# %% [markdown]
# **Section notes — tuples:**
#
# | Property | Meaning |
# |---|---|
# | immutable | no assign/append/sort — safe for constant data |
# | hashable | valid dict keys / set members (lists are not) |
# | methods | only `count()` and `index()` |
# | unpacking | `a, b, c = t`, `first, *rest = t` |
# | 1-tuple | `(42,)` — comma required |
# | named fields | `collections.namedtuple` |

# %% [markdown]
# ---
# ## 10. Sets & frozenset
#
# ▶ Session 5 - Tuples + Sets + Dictionary: https://www.youtube.com/watch?v=jcQjp11mn1A

# %%
# creation — unordered, NO duplicates, elements must be hashable
s = {1, 2, 3, 2, 1}                # literals dedupe automatically
print("set literal   :", s)
print("from list     :", set([1, 2, 2, 3]))           # dedupe idiom
print("from string   :", set("banana"))               # {'b','a','n'}
print("empty set     :", set(), "| {} is a", type({}).__name__, "not a set!")
print("set comprehension:", {x * x for x in range(6)})  # {0, 1, 4, 9, 16, 25}

# %%
# add / update / remove / discard / pop / clear
s = {1, 2, 3}
s.add(4)                            # single element; duplicates are a no-op
s.add(2)
print("add   :", s)
s.update([5, 6], {7})               # accepts many iterables at once
print("update:", s)

print("pop   :", s.pop())           # removes & returns an ARBITRARY element
s.discard(99)                       # silently ignores missing values
print("discard(99) on a missing key: no error")
try:
    s.remove(99)
except KeyError as e:
    print("remove() raises KeyError for missing:", e)
print("before clear:", sorted(s))
s.clear()
print("after clear :", s)

# %%
# set algebra — four operations, two syntaxes each
a, b = {1, 2, 3, 4}, {3, 4, 5}
print("union        a | b:", a | b, "== a.union(b)        :", a.union(b))
print("intersection a & b:", a & b, "== a.intersection(b) :", a.intersection(b))
print("difference   a - b:", a - b, "== a.difference(b)   :", a.difference(b))
print("symmetric    a ^ b:", a ^ b, "== a.symmetric_difference(b):",
      a.symmetric_difference(b))

# in-place variants
c = {1, 2}
c.update({2, 3})                    # |=
c.intersection_update({2, 3, 4})    # &=
c.difference_update({4})            # -=
c.symmetric_difference_update({3})  # ^=
print("in-place updates:", c)

# %%
# comparisons & predicates
a, b = {1, 2}, {1, 2, 3}
print("issubset   :", a.issubset(b), a <= b)
print("issuperset :", b.issuperset(a), b >= a)
print("isdisjoint :", {1, 2}.isdisjoint({3, 4}), {1, 2}.isdisjoint({2, 3}))
print("equality   :", {1, 2} == {2, 1})           # order never matters
print("membership :", 2 in {1, 2, 3}, 9 not in {1, 2, 3})

# %%
# frozenset — an immutable set: hashable, so usable inside sets/dicts
fs = frozenset([1, 2, 3])
print(fs, type(fs).__name__)
try:
    fs.add(4)
except AttributeError as e:
    print("frozenset is immutable:", e)
print("frozenset ops:", fs | frozenset([3, 4]), fs & frozenset([2, 3]))
cache = {frozenset([1, 2]): "pair"}
print("frozenset as dict key:", cache[frozenset([2, 1])])

# %%
# classic set problems on two lists
a_list = [1, 2, 3, 4, 5]
b_list = [4, 5, 6, 7]
A, B = set(a_list), set(b_list)

print("common to both:", sorted(A & B))
print("only in A     :", sorted(A - B))
print("only in B     :", sorted(B - A))
print("everything    :", sorted(A | B))
print("unique in A   :", sorted(set(a_list)), " (set order is NOT guaranteed)")
print("dupes in A    :", sorted({x for x in a_list if a_list.count(x) > 1}))

# %% [markdown]
# **Section notes — sets:**
#
# | Operation | Method | Symbol |
# |---|---|---|
# | union | `a.union(b)` | `a \| b` |
# | intersection | `a.intersection(b)` | `a & b` |
# | difference | `a.difference(b)` | `a - b` |
# | symmetric difference | `a.symmetric_difference(b)` | `a ^ b` |
# | subset / superset | `issubset` / `issuperset` | `<=` `>=` |
# | disjoint test | `a.isdisjoint(b)` | — |
# | modify | `add update pop discard remove clear` | `\|=` `&=` `-=` `^=` |
# | immutable set | `frozenset(iterable)` | — |
#
# > **Gotcha:** `set.remove` raises `KeyError`, `set.discard` does not;
# > `set.pop()` removes an *arbitrary* element (never rely on which one);
# > `{}` is always a dict — use `set()` for an empty set.

# %% [markdown]
# ---
# ## 11. Dictionaries
#
# ▶ Session 5 - Tuples + Sets + Dictionary: https://www.youtube.com/watch?v=jcQjp11mn1A

# %%
# creation — four idioms
d1 = {"name": "Ada", "age": 36}
d2 = dict([("name", "Ada"), ("age", 36)])
d3 = dict(name="Ada", age=36)
d4 = dict.fromkeys(["name", "age"], "?")     # same default for every key
print(d1, "\n", d2, "\n", d3, "\n", d4)
print("keys must be hashable -> tuple key works:",
      {(1, 2): "tuple-key"}[(1, 2)])

# %%
# reading: [] vs get(key, default)
d = {"a": 1, "b": 2}
print("d['a']       ->", d["a"])
print("d.get('z')   ->", d.get("z"))               # missing -> None, no error
print("d.get('z', 0)->", d.get("z", 0))             # missing -> default
try:
    d["z"]
except KeyError as e:
    print("[] on missing key -> KeyError:", e)
print("'z' in d ->", "z" in d)                     # membership tests KEYS

# %%
# setdefault & update — insert-if-missing / merge
inventory = {"apple": 2}
inventory.setdefault("apple", 99)      # key exists -> returns existing value
inventory.setdefault("banana", 5)      # key missing -> inserts default
print("setdefault:", inventory)

inventory.update({"banana": 7, "cherry": 1})   # merge, overwrite existing keys
inventory.update(pear=3)                       # kwargs work too
print("update    :", inventory)

# %%
# pop / popitem / keys / values / items
d = {"x": 1, "y": 2, "z": 3}
print("pop:", d.pop("y"), "|", d.pop("missing", "default"))  # default avoids KeyError
print("popitem:", d.popitem())        # removes the LAST inserted pair (3.7+ order)
print("remaining:", d)

d = {"x": 1, "y": 2}
print("keys  :", list(d.keys()))
print("values:", list(d.values()))
print("items :", list(d.items()))
for k, v in d.items():                # the canonical iteration
    print("   ", k, "->", v)

# %%
# copy / fromkeys / comprehension / nested dicts
d = {"a": 1, "b": 2}
clone = d.copy()                      # shallow! nested values are shared
print("copy:", clone, "| is the same dict:", clone is d)
print("fromkeys:", dict.fromkeys("abc", 0), dict.fromkeys(["p", "q"]))

squares = {x: x * x for x in range(5)}                # dict comprehension
evens = {x: x for x in range(10) if x % 2 == 0}
inverted = {v: k for k, v in squares.items()}          # invert a dict
print("squares :", squares)
print("evens   :", evens)
print("inverted:", inverted)

nested = {"user": {"name": "Ada", "address": {"city": "London"}}}
print("nested lookup:", nested["user"]["address"]["city"])
print("nested safe   :", nested.get("user", {}).get("phone"))

# %%
# iteration order: insertion-ordered since Python 3.7 (language guarantee)
d = {"z": 1, "a": 2, "m": 3}
print("insertion order preserved:", list(d))
print("keys() view is a live view:", list(d.keys()))

# merging dicts — three equivalent ways
left, right = {"a": 1, "b": 2}, {"b": 3, "c": 4}
print("unpack merge:", {**left, **right})            # 3.5+, right wins
print("| operator  :", left | right)                 # 3.9+, right wins
merged = dict(left)
merged.update(right)
print("update()    :", merged)

# %%
# collections helpers — defaultdict, Counter, OrderedDict (all stdlib)
from collections import defaultdict, Counter, OrderedDict

words = ["apple", "avocado", "banana", "blueberry", "cherry"]
by_letter = defaultdict(list)          # no KeyError — auto-creates an empty list
for w in words:
    by_letter[w[0]].append(w)
print("defaultdict:", dict(by_letter))

counts = Counter(words)                # a dict subclass tuned for tallies
print("Counter:", counts.most_common(2), "| total:", sum(counts.values()))
print("Counter('mississippi') ->", Counter("mississippi"))

od = OrderedDict.fromkeys(["z", "a", "m"])
od.move_to_end("z")                    # explicit reordering (plain dicts keep insertion order now)
print("OrderedDict:", list(od))
# TIP: plain dicts replaced OrderedDict for ordering; Counter/defaultdict still earn their import

# %% [markdown]
# **Section notes — dicts:**
#
# | Task | Code |
# |---|---|
# | safe read | `d.get(key, default)` |
# | insert-if-missing | `d.setdefault(key, default)` |
# | merge | `d.update(other)`, `{**a, **b}`, `a \| b` |
# | remove | `d.pop(k[, default])`, `d.popitem()` |
# | views | `d.keys()`, `d.values()`, `d.items()` |
# | build | `{k: v for …}`, `dict.fromkeys(iterable, v)` |
# | invert | `{v: k for k, v in d.items()}` |
# | tally / group | `Counter(...)`, `defaultdict(list)` |
#
# > **Gotcha:** iterating a dict yields **keys** (`for k in d`); `in` also
# > tests keys; keys must be hashable (no list keys) — values can be anything.

# %% [markdown]
# ---
# ## 12. Functions
#
# ▶ Session 6 - Functions in Python: https://www.youtube.com/watch?v=OOInK25PoFo

# %% [markdown]
# ### `def name(parameters): … return value`
# **What it does:** binds a reusable block of code to a name.
# **Parameters:** positional, keyword, defaults, `*args`, `**kwargs`.
# **Returns:** a value (or `None` when `return` is omitted — every Python
# function returns *something*).

# %%
def greet(name, greeting="Hello"):            # positional + default parameter
    """Greet someone. This docstring becomes greet.__doc__."""
    return f"{greeting}, {name}!"

print(greet("Ada"))                           # default greeting used
print(greet("Bob", "Hi"))                     # positional override
print(greet(greeting="Hey", name="Carol"))    # keywords are order-independent
print(greet("Dan", greeting="Yo"))
print("docstring:", greet.__doc__)
print("implicit None:", print("x") is None)

# %%
# parameter kinds: positional-only / normal / *args / keyword-only / **kwargs
def signature_demo(a, b, /, c, d="D", *args, e="E", **kwargs):
    """a,b positional-only (before /), c required, d default, e after *."""
    print(f"  a={a} b={b} c={c} d={d} args={args} e={e} kwargs={kwargs}")

signature_demo(1, 2, 3)                                  # only a b c are required
signature_demo(1, 2, 3, 4, 5, 6, e="EE", extra=7)
print(signature_demo.__doc__)

# %%
# *args collects EXTRA POSITIONAL arguments into a tuple
# **kwargs collects EXTRA KEYWORD arguments into a dict
def total(*numbers, **options):
    print("  numbers:", numbers, type(numbers).__name__)
    print("  options:", options, type(options).__name__)
    result = sum(numbers)
    if options.get("negative"):
        result = -result
    return result

print("->", total(1, 2, 3))
print("->", total(1, 2, 3, negative=True))

def add_all(values):                       # unpack when CALLING: *iterable
    return sum(values)

print("unpack list  ->", add_all([1, 2, 3]), "| unpack tuple ->", add_all((4, 5)))

# %%
# multiple returns -> tuple unpacking; bare `return` / no return -> None
def stats(values):
    return min(values), max(values), sum(values) / len(values)

lo, hi, avg = stats([2, 4, 6])
print("unpacked:", lo, hi, avg)
print("packed  :", stats([2, 4, 6]))

def nothing():
    return
    print("dead code")
print("returns ->", nothing())

# %%
# mutable default argument GOTCHA — the default is created ONCE, at def time
def add_item_bad(item, bucket=[]):
    bucket.append(item)
    return bucket

print("call 1:", add_item_bad(1))          # [1]
print("call 2:", add_item_bad(2))          # [1, 2]  <-- leaked from call 1!

def add_item_good(item, bucket=None):
    if bucket is None:                     # fresh list per call
        bucket = []
    bucket.append(item)
    return bucket

print("call 1:", add_item_good(1))         # [1]
print("call 2:", add_item_good(2))         # [2]
print("both bad calls shared one list:", add_item_bad.__defaults__[0])

# %%
# scope: LEGB — Local -> Enclosing -> Global -> Built-in
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"                        # creates its own x
        print("  inner  sees:", x)
    inner()
    print("  outer  sees:", x)

outer()
print("module sees:", x)

def show_builtins():
    print("  built-in scope visible:", len([1]), type(1), max(3, 9))

show_builtins()
try:
    print(undefined_name)                  # noqa: F821 — no LEGB hit
except NameError as e:
    print("NameError:", e)

# %%
# global & nonlocal — write to an OUTER scope from inside a function
counter = 0
def bump():
    global counter                          # declare intent to modify the global
    counter += 1

bump(); bump(); bump()
print("global counter:", counter)

def make_counter():
    n = 0
    def inner():
        nonlocal n                          # modify the ENCLOSING function's variable
        n += 1
        return n
    return inner

c = make_counter()
print("nonlocal counter:", c(), c(), c())

# %%
# lambda — small anonymous function (single expression, no statements)
square = lambda x: x ** 2                  # TIP: fine as an inline key; avoid naming lambdas
print(square(7), (lambda a, b: a + b)(3, 4))

names = ["ada", "grace", "alan"]
print(sorted(names, key=lambda s: len(s)))
print(sorted(names, key=lambda s: s[::-1]))

# %%
# map / filter / sorted — functional tools over iterables
nums = [1, 2, 3, 4, 5]
print("map    :", list(map(lambda x: x * 10, nums)))
print("filter :", list(filter(lambda x: x % 2 == 0, nums)))
print("sorted  :", sorted(nums, key=lambda x: -x))
print("map+str :", list(map(str, nums)))            # builtins are already functions
print("no lambda needed:", list(map(str.upper, ["a", "b"])))
print("map is lazy ->", next(iter(map(lambda x: x * 2, nums))))
# NOTE: map/filter return iterators — wrap in list(...) to see or reuse values

# %%
# functions are FIRST-CLASS objects: assign, pass, store and return them
def shout(text):
    return text.upper()

quiet = shout                                  # alias — same function object
print(quiet("hey"), shout is quiet)

def executor(func, value):                     # functions as ARGUMENTS
    return func(value)

print(executor(shout, "polymorphism"), executor(str.strip, "  pad  "))

def decorator_maker():                         # functions as RETURN VALUES
    def inner(msg):
        return f"<<{msg}>>"
    return inner

wrap = decorator_maker()
print(wrap("returned function"))
print("callable:", callable(shout), callable(42))
print("stored in a list:", [str.upper, str.lower], str.upper("mixed"))

# %%
# recursion basics — a function calling itself (deep dive in notebook 02)
def factorial(n):
    if n <= 1:                                 # base case — stops the recursion
        return 1
    return n * factorial(n - 1)                # recursive case

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print("factorial(10) =", factorial(10))
print("fib(15)       =", fib(15))
# WARNING: naive fib is O(2**n) — every level roughly doubles the work
# TIP: memoisation (dict cache / functools.lru_cache) fixes this — see notebook 02
# TIP: recursion depth and complexity are section 13 here + sections 13-14 there

# %% [markdown]
# **Section notes — functions:**
#
# | Concept | Syntax |
# |---|---|
# | define | `def name(params):` + docstring |
# | defaults | `def f(a, b=1):` — never use a **mutable** default |
# | var-positional | `*args` → tuple |
# | var-keyword | `**kwargs` → dict |
# | keyword-only | `def f(*, key=None):` |
# | unpack on call | `f(*iterable, **mapping)` |
# | scope writes | `global x` / `nonlocal x` |
# | anonymous | `lambda x: expr` |
# | functional | `map(f, it)`, `filter(p, it)`, `sorted(it, key=f)` |
# | first-class | assign / pass / return functions like any value |

# %% [markdown]
# ---
# ## 13. Time complexity primer
#
# ▶ Session on Time Complexity: https://www.youtube.com/watch?v=J1vdIcg5IZs
#
# Complexity describes how the **operation count grows with input size `n`** —
# not the seconds on your laptop.
#
# | Bound | Meaning | Question it answers |
# |---|---|---|
# | **O(f(n))** (Big-O) | upper bound, worst case | "how bad can it get?" |
# | **Ω(f(n))** (Omega) | lower bound, best case | "how good can it get?" |
# | **Θ(f(n))** (Theta) | tight bound (both of the above match) | "what is the typical growth?" |
#
# | Class | Name | Example |
# |---|---|---|
# | O(1) | constant | dict/set lookup, `lst.append` |
# | O(log n) | logarithmic | binary search |
# | O(n) | linear | single loop, `lst.index` worst case |
# | O(n log n) | linearithmic | `sorted()`, `list.sort()` |
# | O(n²) | quadratic | nested loops over the same input |
# | O(2ⁿ) | exponential | subsets, naive recursion |
# | O(n!) | factorial | permutations, brute-force tours |

# %%
import timeit
import time

# --- experiment 1: list membership O(n) vs set membership O(1) ---
data = list(range(20_000))
sdata = set(data)
target = 19_999

def per_lookup_us(fn, number):
    return timeit.timeit(fn, number=number) / number * 1e6   # microseconds

t_list = per_lookup_us(lambda: target in data, number=50)
t_set = per_lookup_us(lambda: target in sdata, number=50_000)
print(f"list 'in' (n=20000): {t_list:9.3f} us/lookup   O(n) worst case")
print(f"set  'in' (n=20000): {t_set:9.3f} us/lookup   O(1) average")
print(f"list is ~{t_list / max(t_set, 1e-9):.0f}x slower -> use a set for membership")

# %%
# --- experiment 2: lst.append O(1) amortised vs lst.insert(0, x) O(n) ---
def append_n(n):
    acc = []
    for i in range(n):
        acc.append(i)                   # grows at the end — no shifting

def insert0_n(n):
    acc = []
    for i in range(n):
        acc.insert(0, i)                # shifts every element on each call

n = 5000
t_append = min(timeit.repeat(lambda: append_n(n), number=5, repeat=3))
t_insert = min(timeit.repeat(lambda: insert0_n(n), number=5, repeat=3))
print(f"append  (n={n}): {t_append * 1000:8.2f} ms   O(n) total")
print(f"insert0 (n={n}): {t_insert * 1000:8.2f} ms   O(n^2) total "
      f"({t_insert / max(t_append, 1e-9):.0f}x slower)")

# %%
# --- experiment 3: string concat vs "".join (immutability costs copies) ---
def concat_way(n):
    s = ""
    for i in range(n):
        s = s + str(i)                  # CPython resizes in place when it can

def concat_shared(n):
    """Same loop — but something else keeps the OLD strings alive."""
    s = ""
    keep = []
    for i in range(n):
        s = s + str(i)
        keep.append(s)                  # refcount > 1 -> a real O(n) copy each time
    return keep

def join_way(n):
    return "".join(str(i) for i in range(n))   # one allocation, then append

n = 20_000
t_concat = min(timeit.repeat(lambda: concat_way(n), number=3, repeat=3))
t_shared = min(timeit.repeat(lambda: concat_shared(n), number=1, repeat=3))
t_join = min(timeit.repeat(lambda: join_way(n), number=3, repeat=3))
print(f"s = s + x            : {t_concat * 1000:8.2f} ms   looks linear (optimised)")
print(f"s = s + x (refs held) : {t_shared * 1000:8.2f} ms   "
      f"{t_shared / max(t_join, 1e-9):6.0f}x slower -> the REAL O(n^2)")
print(f"''.join               : {t_join * 1000:8.2f} ms   always O(n)")
print("NOTE: CPython only re-sizes a string when nothing else references it —")
print("      an implementation detail you must not depend on. `''.join(...)` is")
print("      the idiom that is O(n) on every Python, every time.")

# %%
# --- a wall-clock helper (time.perf_counter) ---
def time_call(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000                  # ms

print(f"sum(range(10**6))          : {time_call(lambda: sum(range(10**6))):8.3f} ms")
print(f"sum([i for i in range(10**6)]): {time_call(lambda: sum([i for i in range(10**6)])):8.3f} ms")
print("TIP: perf_counter is monotonic — never use time.time() for benchmarks")

# %%
# --- interview example 1: nested loops are O(n^2) ---
def count_ops(n):
    ops = 0
    for i in range(n):
        for j in range(n):
            ops += 1                     # exactly n*n iterations
    return ops

for n in (10, 100, 1000):
    print(f"n={n:5} -> {count_ops(n):8} iterations (= n^2)")
# 10x input -> 100x work: the quadratic signature

# %%
# --- interview example 2: binary search O(log n) vs linear search O(n) ---
def binary_search(arr, target):
    low, high, steps = 0, len(arr) - 1, 0
    while low <= high:
        steps += 1
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid, steps
        if arr[mid] < target:
            low = mid + 1                # discard left half
        else:
            high = mid - 1               # discard right half
    return -1, steps

def linear_search(arr, target):
    steps = 0
    for i, value in enumerate(arr):
        steps += 1
        if value == target:
            return i, steps
    return -1, steps

big = list(range(100_000))
print("binary  77777 ->", binary_search(big, 77777), " (~17 comparisons = log2 100000)")
print("linear  77777 ->", linear_search(big, 77777), " (77778 comparisons)")
print("binary  missing ->", binary_search(big, -1))
# TIP: binary search needs SORTED input — sorting first costs O(n log n)

# %% [markdown]
# **Section notes — complexity:**
#
# | Operation | Complexity |
# |---|---|
# | dict/set get, list append | O(1) average |
# | list index, set add (worst) | O(n) |
# | list insert/remove | O(n) |
# | sorted / list.sort | O(n log n) |
# | nested loop over same input | O(n²) |
# | naive recursion (fib) | O(2ⁿ) |
# | `s = s + x` in a loop | O(n²) char copies (CPython may hide it — see exp. 3) |
# | `"".join(parts)` | O(n) |
#
# > **Interview short-list:** how do you prove an algorithm is O(log n)? ·
# > why is a hash-map lookup O(1)? · when does `insert(0)` become a problem? ·
# > amortised cost of `list.append` · `set` membership vs `list` membership.

# %% [markdown]
# ---
# ## Quick revision sheet
#
# | Task | Code |
# |---|---|
# | print with separator/newline | `print(a, b, sep="-", end="!")` |
# | check the type / identity | `type(x)` / `id(x)` |
# | keyword list | `keyword.kwlist` |
# | read + convert input | `int(input("..."))` (inside `try`) |
# | safe literal parse | `ast.literal_eval(s)` |
# | short-circuit default | `value or default` |
# | reverse a string | `s[::-1]` |
# | palindrome | `"".join(c.lower() for c in s if c.isalnum()) == ...[::-1]` |
# | anagram | `sorted(a) == sorted(b)` |
# | character counts | `Counter(s)`, `s.count(c)` |
# | dedupe keeping order | `list(dict.fromkeys(iterable))` |
# | append vs extend | one element vs splice |
# | sort in place / copy | `lst.sort(key=)` / `sorted(lst, key=)` |
# | two-pointer pair sum | shrink from the ends on a sorted list |
# | set algebra | `&`, `\\|`, `-`, `^` (and `.union()` etc.) |
# | safe dict read | `d.get(key, default)` |
# | merge dicts | `{**a, **b}` or `a \\| b` |
# | iteration with index | `enumerate(seq, start=1)` |
# | parallel iteration | `zip(a, b)` |
# | build a list | `[expr for x in it if cond]` |
# | default argument | `def f(x, bucket=None):` |
# | big-O cheat sheet | lookup dict/set `O(1)` · search sorted `O(log n)` · sort `O(n log n)` |
#
# > **Interview short-list:** `==` vs `is` · `append` vs `extend` ·
# > `sort` vs `sorted` · `find` vs `index` · `remove` vs `discard` ·
# > mutable default argument · shallow vs deep copy · `for-else` semantics ·
# > why list membership is O(n) but set membership is O(1).

# %% [markdown]
# ---
# **Next notebook:** `02_oop_advanced_python.ipynb` — classes, exceptions,
# decorators, iterators and generators build directly on the functions,
# containers and complexity ideas from this notebook.
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
