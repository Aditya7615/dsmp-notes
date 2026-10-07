# %% [markdown]
# # 07 · SQL for Data Science — Complete Reference (runnable with sqlite3)
#
# SQL is the *other* half of a data analyst's toolkit: gathering, cleaning and
# exploring data **where it lives** — inside the database. Every statement in
# this notebook runs **live** against an in-memory SQLite database
# (`sqlite3.connect(":memory:")`) and prints its result with pandas, so you can
# edit any query and re-run it immediately.
#
# Covers sessions 30-37 plus the Zomato / Laptops / Flights case studies,
# interviews, database design, views, transactions — in playlist order.
#
# > **Note:** SQLite is a *library* database (no server) chosen because it ships
# > with Python. Dialect differences vs MySQL/PostgreSQL are called out
# > **NOTE (dialect):** at each spot that matters.

# %% [markdown]
# ## ▶ Sessions covered
#
# | # | Session | Video |
# |---|---------|-------|
# | 96 | Session 30 - Database Fundamentals | https://www.youtube.com/watch?v=CTLwYjh92xw |
# | 99 | Session 31 - SQL DDL Commands | https://www.youtube.com/watch?v=ny1mh6VUpnQ |
# | 101 | Session 32 - SQL DML Commands | https://www.youtube.com/watch?v=kzMZDlvHFak |
# | 102 | Session 33 - SQL Grouping + Sorting | https://www.youtube.com/watch?v=nsKcmOly0UY |
# | 104 | Session 34 - SQL Joins | https://www.youtube.com/watch?v=s6xAVqYdmJ4 |
# | 107 | Session 35 - Subqueries in SQL | https://www.youtube.com/watch?v=YYq47MN3TZI |
# | 108 | SQL Case Study 1 \| Zomato Dataset | https://www.youtube.com/watch?v=jYwcB0KEFfY |
# | 111 | SQL Interview Questions Part 1 | https://www.youtube.com/watch?v=QlenVBvzrW4 |
# | 112 | Session 36 - Window Functions in SQL | https://www.youtube.com/watch?v=3LJ-yWBQBGs |
# | 115 | Session 37 - Window Functions Part 2 | https://www.youtube.com/watch?v=YgizUxmBFK8 |
# | 116 | Session 37 (Cont..) - Window Functions Part 3 | https://www.youtube.com/watch?v=e0HQVOdfqlQ |
# | 117 | Session on Data Cleaning using SQL \| Laptops | https://www.youtube.com/watch?v=NsOXJdRIamc |
# | 118 | Session on EDA using SQL \| Laptops | https://www.youtube.com/watch?v=dv8TkUTlQvI |
# | 124 | Session on Datetime in SQL | https://www.youtube.com/watch?v=OY9YiALhSR4 |
# | 127 | SQL Datetime Case Study on Flights | https://www.youtube.com/watch?v=zY1UCTko64E |
# | 128 | Session on Database Design \| Normalization | https://www.youtube.com/watch?v=8SOs0URTvr0 |
# | 132 | Session on Views and UDFs in SQL | https://www.youtube.com/watch?v=dMoKTx1bVnY |
# | 133 | Session on Transactions & Stored Procedures | https://www.youtube.com/watch?v=-O_9iR7XsJw |

# %%
import sqlite3
import pandas as pd

print("SQLite version:", sqlite3.sqlite_version)

con = sqlite3.connect(":memory:")     # in-memory DB — lives only for this kernel
con.isolation_level = None            # autocommit: WE drive BEGIN/COMMIT (§12)


def q(sql: str, params=None) -> pd.DataFrame:
    """Run a SQL statement and return the result as a DataFrame."""
    return pd.read_sql_query(sql, con, params=params)


# --- schema (DDL) -------------------------------------------------------
con.executescript("""
CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL UNIQUE,
    location  TEXT
);

CREATE TABLE employees (
    emp_id     INTEGER PRIMARY KEY,
    emp_name   TEXT NOT NULL,
    dept_id    INTEGER NOT NULL REFERENCES departments(dept_id),
    manager_id INTEGER REFERENCES employees(emp_id),   -- self reference
    salary     REAL NOT NULL CHECK(salary > 0),
    bonus      REAL,                                    -- NULLs on purpose
    hire_date  TEXT NOT NULL,                           -- ISO 'YYYY-MM-DD'
    age        INTEGER,
    city       TEXT
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    cust_name   TEXT NOT NULL,
    city        TEXT,
    signup_date TEXT,
    tier        TEXT DEFAULT 'Bronze'
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  TEXT NOT NULL,
    amount      REAL NOT NULL CHECK(amount >= 0),
    status      TEXT DEFAULT 'completed'
);
""")

# --- seed data (DML) ----------------------------------------------------
con.executemany("INSERT INTO departments VALUES(?, ?, ?)", [
    (1, "HR", "Mumbai"), (2, "IT", "Bengaluru"), (3, "Finance", "Delhi"),
    (4, "Sales", "Pune"), (5, "Marketing", "Mumbai"), (6, "Support", "Hyderabad"),
    (7, "Legal", "Chennai"),          # Legal has NO employees (for join demos)
])

con.executemany(
    "INSERT INTO employees VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (1, "Rahul", 2, None, 90000, 5000, "2019-03-15", 32, "Bengaluru"),
        (2, "Priya", 2, 1, 75000, None, "2020-07-01", 28, "Bengaluru"),
        (3, "Amit", 1, None, 62000, 3000, "2017-11-20", 40, "Mumbai"),
        (4, "Sneha", 1, 3, 58000, None, "2021-01-10", 26, "Mumbai"),
        (5, "Vikram", 3, None, 88000, 8000, "2016-05-05", 45, "Delhi"),
        (6, "Neha", 3, 5, 70000, None, "2022-09-12", 30, "Delhi"),
        (7, "Arjun", 4, None, 66000, 4000, "2018-02-18", 35, "Pune"),
        (8, "Kavya", 4, 7, 54000, None, "2023-04-01", 25, "Pune"),
        (9, "Rohan", 2, 1, 90000, 6000, "2021-06-25", 29, "Bengaluru"),
        (10, "Divya", 5, None, 60000, None, "2019-12-30", 33, "Mumbai"),
        (11, "Manish", 6, None, 48000, 2000, "2020-02-14", 31, "Hyderabad"),
        (12, "Pooja", 6, 11, 46000, None, "2024-01-08", 24, "Hyderabad"),
    ])

con.executemany("INSERT INTO customers VALUES(?, ?, ?, ?, ?)", [
    (1, "Amita", "Delhi", "2024-01-02", "Gold"),
    (2, "Sunil", "Mumbai", "2024-01-15", "Silver"),
    (3, "Meera", "Pune", "2024-02-01", "Gold"),
    (4, "Rakesh", "Delhi", "2024-02-10", "Bronze"),
    (5, "Zoya", "Kolkata", "2024-02-14", "Silver"),
    (6, "Karthik", "Chennai", "2024-03-01", "Bronze"),
    (7, "Nisha", "Mumbai", "2024-03-05", "Gold"),
    (8, "Vivek", "Pune", "2024-03-11", "Silver"),
    (9, "Tara", "Delhi", "2024-03-18", "Bronze"),     # places NO orders
    (10, "Imran", "Hyderabad", "2024-03-25", "Bronze"),  # places NO orders
])

con.executemany("INSERT INTO orders VALUES(?, ?, ?, ?, ?)", [
    (1, 1, "2024-01-05", 1200, "completed"),
    (2, 1, "2024-01-06", 800, "completed"),
    (3, 1, "2024-01-20", 1500, "cancelled"),
    (4, 2, "2024-01-06", 2200, "completed"),
    (5, 2, "2024-02-03", 900, "completed"),
    (6, 3, "2024-02-05", 3000, "completed"),
    (7, 3, "2024-02-06", 450, "completed"),
    (8, 3, "2024-02-07", 700, "pending"),
    (9, 4, "2024-02-14", 1800, "completed"),
    (10, 5, "2024-02-14", 600, "cancelled"),
    (11, 5, "2024-03-02", 2500, "completed"),
    (12, 6, "2024-03-08", 1100, "completed"),
    (13, 7, "2024-03-15", 950, "completed"),
    (14, 7, "2024-03-16", 1300, "completed"),
    (15, 8, "2024-03-21", 2000, "completed"),
])

# --- look at what we built ---------------------------------------------
print("tables:", q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"].tolist())
for t in ("departments", "employees", "customers", "orders"):
    df = q(f"SELECT * FROM {t}")
    print(f"\n--- {t}  ({len(df)} rows) ---")
    print(df)

# %% [markdown]
# ---
# ## 1. Database fundamentals
#
# ▶ Session 30 - Database Fundamentals: https://www.youtube.com/watch?v=CTLwYjh92xw
#
# ### DBMS vs RDBMS
#
# | | **DBMS** | **RDBMS** |
# |---|---|---|
# | Storage model | files, documents, graphs, key-value | **tables** with rows & columns |
# | Relations | none required | foreign keys enforce relations |
# | Query language | varies | SQL |
# | Examples | MongoDB, Redis, Neo4j | PostgreSQL, MySQL, SQLite, Oracle |
#
# ### Table / row / column / keys
#
# ```text
#  departments                       employees
# +---------+----------+---------+   +--------+---------+---------+------------+
# | dept_id | dept_name| location|   | emp_id | emp_name| dept_id | manager_id |
# +---------+----------+---------+   +--------+---------+---------+------------+
# | 1       | HR       | Mumbai  |   | 1      | Rahul   | 2       | NULL  <─────┐
# | 2       | IT       | Bengaluru   | 2      | Priya   | 2       | 1  ─────────┘
# +---------+----------+---------+   +--------+---------+---------+
#    PRIMARY KEY ^   UNIQUE ^            PK ^       FK to departments ^
#                                        FK to *same* table (self reference) ^
# ```
#
# - **Primary key (PK)** — uniquely identifies a row; never NULL.
# - **Foreign key (FK)** — a PK of *another* table, pointing at a parent row.
# - **Composite key** — a key made of 2+ columns.
#
# ### Keys
#
# | Key | Definition | In our schema |
# |---|---|---|
# | candidate | any column (set) that *could* be the PK | `emp_name` (unique here) |
# | primary | the chosen candidate key | `employees.emp_id` |
# | unique | allows NULL, forbids repeats | `departments.dept_name` |
# | foreign | references a PK of another (or same) table | `employees.dept_id` |
# | composite | multi-column key | `(order_id, product_id)` style |
#
# ### ER diagram concepts (entity-relationship)
#
# - **Entity** — a noun → table (`customer`, `order`); **attribute** → column.
# - **Relationship** — a verb between entities: `customer --places--> order`.
# - **Cardinality**:
#   - **1:1** — one user has one profile
#   - **1:N** — one department has many employees ← our `departments/employees`
#   - **M:N** — students ↔ courses (needs a *junction* table `enrollments`)
# - **Participation** — total (every row must match) vs optional (may not).
# - **Degree** — arity of the relationship (binary is the common case).
#
# ### OLTP vs OLAP
#
# | | **OLTP** (operations) | **OLAP** (analytics) |
# |---|---|---|
# | Purpose | run the business | understand the business |
# | Queries | short point reads (`WHERE id=?`) | huge scans, aggregates |
# | Writes | thousands/sec, small | batch loads, rarely updated |
# | Schema | normalized (§13) | star/snowflake, denormalized |
# | Rows touched | 1-100 | millions |
# | Example | place an order | "revenue by month by city" |
#
# ### SQLite vs MySQL / PostgreSQL
#
# | Feature | SQLite | MySQL | PostgreSQL |
# |---|---|---|---|
# | architecture | in-process file | client/server | client/server |
# | concurrency | 1 writer | good | excellent |
# | data types | dynamic (types per column are hints) | rigid | strict, rich types |
# | JSON support | `json_each()` | good | excellent |
# | window functions | ✅ (3.25+) | ✅ (8.0+) | ✅ |
# | `RIGHT/FULL OUTER JOIN` | **only 3.39+** (see §7) | ✅ | ✅ |
# | `TRUNCATE` | ❌ (use `DELETE FROM`) | ✅ | ✅ |
# | stored procedures | ❌ | ✅ (SQL/PSM) | ✅ (PL/pgSQL) |
# | `STDDEV()` | ❌ (compute it, §15) | ✅ | ✅ |
# | `GROUP BY GROUPING SETS` | build-dependent (❌ here) | ✅ | ✅ |
# | `ANY/ALL` subqueries | ❌ (rewrite with IN/MAX) | ✅ | ✅ |
# | `AUTOINCREMENT` | `INTEGER PRIMARY KEY` auto-rowid | `AUTOINCREMENT` | `SERIAL` |
# | best for | learning, embedded, tests | web at scale | analytics + correctness |

# %%
print("SQLite:", sqlite3.sqlite_version)
print(q("PRAGMA table_info(employees)"))         # columns, types, PK
print(q("PRAGMA foreign_key_list(employees)"))    # FKs declared on the table
print(q("SELECT * FROM pragma_index_list('employees')"))   # auto indexes (none yet)

# %% [markdown]
# **Section notes — fundamentals:**
#
# | Concept | One-liner |
# |---|---|
# | DBMS / RDBMS | data platform / relational (tables + SQL) |
# | PK | unique, not null identity of a row |
# | FK | pointer to a parent PK — enforces referential integrity |
# | 1:N | one-to-many: put the FK on the "many" side |
# | M:N | needs a junction table |
# | OLTP vs OLAP | write-fast operational vs read-fast analytical |
# | SQLite here | no server; 3.35.5 → emulate RIGHT/FULL joins (§7) |

# %% [markdown]
# ---
# ## 2. DDL — Data Definition Language
#
# ▶ Session 31 - SQL DDL Commands: https://www.youtube.com/watch?v=ny1mh6VUpnQ
#
# DDL defines/changes **structure**: `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`.
# It is auto-committed in most RDBMS (and, with our `isolation_level=None`, here too).

# %%
# CREATE TABLE with every common constraint
con.executescript("""
CREATE TABLE IF NOT EXISTS products (
    product_id  INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL UNIQUE,
    price       REAL NOT NULL DEFAULT 9.99 CHECK(price > 0),
    dept_id     INTEGER REFERENCES departments(dept_id),
    launched    TEXT
);
""")

q1 = "INSERT INTO products(product_id, product_name, price, dept_id, launched) VALUES(?, ?, ?, ?, ?)"
con.execute(q1, (1, "Notebook", 60.0, 4, "2023-06-01"))
con.execute(q1, (2, "Marker", 25.5, 4, "2023-06-01"))
con.execute(q1, (3, "Stapler", 180.0, 6, "2024-01-15"))
print(q("SELECT * FROM products"))

# %%
# constraints reject bad data — Python sees sqlite3.IntegrityError
violations = [
    ("duplicate PK", "INSERT INTO products VALUES(1, 'Pen', 10, 4, '2024-01-01')"),
    ("NULL in NOT NULL", "INSERT INTO products(product_id, product_name) VALUES(4, NULL)"),
    ("CHECK price > 0", "INSERT INTO products VALUES(4, 'Broken', -5, 4, '2024-01-01')"),
    ("bad FK (dept 99)", "INSERT INTO products VALUES(4, 'Ghost', 10, 99, '2024-01-01')"),
]
for label, sql in violations:
    try:
        con.execute(sql)
        print(f"{label}: NO ERROR (unexpected!)")
    except sqlite3.IntegrityError as e:
        print(f"{label:20s} -> IntegrityError: {e}")
print("\nstill only the 3 valid rows:", len(q("SELECT * FROM products")))

# %%
# ### ALTER TABLE — ADD / DROP / RENAME COLUMN
con.execute("ALTER TABLE products ADD COLUMN weight_g REAL DEFAULT 0")
con.execute("UPDATE products SET weight_g = 70 WHERE product_id = 1")
con.execute("ALTER TABLE products RENAME COLUMN weight_g TO weight_grams")
print(q("SELECT * FROM products"))

con.execute("ALTER TABLE products DROP COLUMN launched")     # SQLite >= 3.35
print("after DROP COLUMN:", q("PRAGMA table_info(products)")["name"].tolist())

# %%
# ### DROP vs TRUNCATE vs DELETE — removing rows/table
con.execute("CREATE TABLE scratch(x INTEGER)")   # disposable table
con.executemany("INSERT INTO scratch VALUES(?)", [(i,) for i in range(100)])

print("before          :", len(q("SELECT * FROM scratch")))
con.execute("DELETE FROM scratch")               # TRUNCATE in SQLite (no TRUNCATE stmt!)
print("after DELETE all:", len(q("SELECT * FROM scratch")))

con.execute("DROP TABLE scratch")                # removes the table itself
print("tables now:",
      q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"].tolist())

# %%
# RENAME a table (SQLite: ALTER TABLE old RENAME TO new)
con.execute("CREATE TABLE old_reports(id INTEGER PRIMARY KEY, title TEXT)")
con.execute("ALTER TABLE old_reports RENAME TO reports")
print(q("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%report%'"))
con.execute("DROP TABLE reports")

# %% [markdown]
# ### DDL cheat-sheet
#
# | Task | SQLite | MySQL / PostgreSQL |
# |---|---|---|
# | add column | `ALTER TABLE t ADD COLUMN c TYPE` | same |
# | drop column | `ALTER TABLE t DROP COLUMN c` (3.35+) | same / `DROP COLUMN` |
# | rename column | `ALTER TABLE t RENAME COLUMN a TO b` | MySQL: `CHANGE`/`MODIFY` |
# | rename table | `ALTER TABLE a RENAME TO b` | `RENAME TABLE a TO b` (MySQL) |
# | remove all rows | `DELETE FROM t` (**no TRUNCATE**) | `TRUNCATE t` (faster, DDL) |
# | remove table | `DROP TABLE t` (structure + data) | same |
#
# > **Gotcha:** `DELETE` is DML (row-by-row, logged, where-clause friendly, can
# > roll back); `TRUNCATE` is DDL (whole table, resets identity, usually not
# > rollback-able). SQLite only has the former.
#
# **Section notes — DDL:** `CREATE TABLE (...constraints...)` · `ALTER TABLE
# ADD/DROP/RENAME COLUMN` · `ALTER TABLE ... RENAME TO ...` · `DROP TABLE` ·
# `DELETE FROM` (truncate substitute) · constraints: `PRIMARY KEY`, `NOT NULL`,
# `UNIQUE`, `DEFAULT`, `CHECK`, `REFERENCES`.

# %% [markdown]
# ---
# ## 3. DML — Data Manipulation Language
#
# ▶ Session 32 - SQL DML Commands: https://www.youtube.com/watch?v=kzMZDlvHFak
#
# DML changes **data**: `INSERT`, `UPDATE`, `DELETE` (+ `MERGE`/upsert in other dialects).

# %%
# INSERT — single row, multi row, and INSERT ... SELECT
con.executescript("CREATE TABLE staging(id INTEGER PRIMARY KEY, tag TEXT);")
con.execute("INSERT INTO staging VALUES(10, 'solo')")                       # single
con.executemany("INSERT INTO staging VALUES(?, ?)",                         # multi (driver)
                [(11, 'batch1'), (12, 'batch2')])
con.execute("INSERT INTO staging(id, tag) VALUES(13, 'sql-multi'), (14, 'sql-multi')")  # multi SQL
print(q("SELECT * FROM staging"))

# INSERT ... SELECT: copy a projection of an existing table
con.execute("INSERT INTO staging(id, tag) SELECT emp_id, 'emp:' || emp_name FROM employees WHERE dept_id = 1")
print("\nafter INSERT ... SELECT:\n", q("SELECT * FROM staging ORDER BY id"))

# %%
# ### UPDATE — the missing-WHERE gotcha
con.execute("CREATE TABLE emp_copy AS SELECT * FROM employees")

before = q("SELECT COUNT(*) AS n, SUM(salary) AS total FROM emp_copy")
con.execute("UPDATE emp_copy SET salary = salary + 1000")            # NO WHERE!
after = q("SELECT COUNT(*) AS n, SUM(salary) AS total FROM emp_copy")
print("without WHERE -> every row changed:\n", before, "\n", after)
# NOTE: this is the classic production accident — ALWAYS write the WHERE first.

con.execute("UPDATE emp_copy SET salary = 60000 WHERE emp_id = 4")   # targeted
print("\nspecific row updated:\n", q("SELECT emp_id, emp_name, salary FROM emp_copy WHERE emp_id = 4"))

con.execute("UPDATE emp_copy SET salary = salary * 1.10 WHERE dept_id = 2 AND salary >= 75000")
print("\nmulti-condition update:\n",
      q("SELECT emp_name, salary FROM emp_copy WHERE dept_id = 2 ORDER BY salary DESC"))

# %%
# ### DELETE — always with a WHERE
con.execute("DELETE FROM emp_copy WHERE salary < 50000")
print("rows left after DELETE WHERE salary < 50000:", len(q("SELECT * FROM emp_copy")))

# %%
# ### UPSERT — INSERT ... ON CONFLICT (SQLite) / MERGE (other RDBMS)
try:
    con.execute("INSERT INTO products VALUES(1, 'Notebook XL', 75.0, 4, 80)")
except sqlite3.IntegrityError as e:
    print("plain INSERT on existing PK ->", e)

con.execute("""
    INSERT INTO products(product_id, product_name, price, dept_id, weight_grams)
    VALUES(1, 'Notebook XL', 75.0, 4, 80)
    ON CONFLICT(product_id) DO UPDATE SET
        price = excluded.price,
        weight_grams = excluded.weight_grams
""")
print(q("SELECT * FROM products WHERE product_id = 1"))

# DELETE-if-exists + INSERT, the pre-upsert pattern, in one statement:
con.execute("INSERT OR REPLACE INTO products VALUES(2, 'Marker Pro', 30.0, 4, 45)")
print("\nafter INSERT OR REPLACE:\n", q("SELECT * FROM products ORDER BY product_id"))

# %% [markdown]
# ### DELETE vs TRUNCATE vs DROP
#
# | | `DELETE` | `TRUNCATE` | `DROP` |
# |---|---|---|---|
# | removes | matching **rows** | **all rows** | table **+ structure** |
# | `WHERE` | ✅ | ❌ | ❌ |
# | speed (big table) | slow (logged row-by-row) | fast | fastest |
# | roll back | yes | usually no | no |
# | triggers | fire | usually no | n/a |
# | SQLite | ✅ | ❌ → use `DELETE FROM` | ✅ |
#
# **NOTE (dialect):** other engines implement `MERGE INTO t ... WHEN MATCHED THEN
# UPDATE / WHEN NOT MATCHED THEN INSERT`; SQLite's equivalents are
# `ON CONFLICT ... DO UPDATE` and `INSERT OR REPLACE`.
#
# **Section notes — DML:** `INSERT [VALUES | ...SELECT]` · `UPDATE ... WHERE` ·
# `DELETE ... WHERE` · upsert: `ON CONFLICT` / `INSERT OR REPLACE` ·
# > **Gotcha:** no `WHERE` on UPDATE/DELETE = rewrite of the whole table.

# %% [markdown]
# ---
# ## 4. SELECT basics
#
# ▶ Session 32 - SQL DML Commands: https://www.youtube.com/watch?v=kzMZDlvHFak
# ▶ Session 33 - SQL Grouping + Sorting: https://www.youtube.com/watch?v=nsKcmOly0UY
#
# ```text
# SELECT <columns/expressions>   -- what to show
# FROM <table>                   -- where the data is
# WHERE <filter>                 -- row filter (before grouping)
# GROUP BY <columns>             -- collapse rows into groups
# HAVING <filter on groups>      -- group filter (after grouping)
# ORDER BY <columns> [ASC|DESC]  -- sort the result
# LIMIT n OFFSET m               -- keep only n rows, skip m
# ```

# %%
# projection, aliases, DISTINCT, expressions
print(q("SELECT emp_name, salary * 12 AS annual_ctc FROM employees WHERE salary > 70000"))
print("\nDISTINCT cities:", q("SELECT DISTINCT city FROM employees ORDER BY city")["city"].tolist())
print("DISTINCT depts  :", q("SELECT DISTINCT dept_id FROM employees ORDER BY dept_id")["dept_id"].tolist())

# %%
# WHERE operators: = != <> > < >= <= , BETWEEN , IN , LIKE , IS NULL , AND/OR/NOT
print("salary > 70000            :\n", q("SELECT emp_name, salary FROM employees WHERE salary > 70000"))
print("BETWEEN 60000 AND 80000   :\n", q("SELECT emp_name, salary FROM employees WHERE salary BETWEEN 60000 AND 80000 ORDER BY salary"))
print("IN (HR, IT)               :\n", q("""
    SELECT e.emp_name, d.dept_name FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.dept_name IN ('HR', 'IT') ORDER BY d.dept_name, e.emp_name"""))

# %%
# LIKE: % = any run of chars, _ = exactly one char
print(q("SELECT emp_name FROM employees WHERE emp_name LIKE 'A%'"))     # starts with A
print(q("SELECT emp_name FROM employees WHERE emp_name LIKE '%a%'"))    # contains a
print(q("SELECT emp_name FROM employees WHERE emp_name LIKE '_i%'"))    # 2nd char is i
print("escape gotcha (names with _):", q("SELECT COUNT(*) AS n FROM employees WHERE emp_name LIKE '%'")["n"][0])

# %%
# NULL handling — NULL is UNKNOWN, not 0 or ''
print("bonus IS NULL    :\n", q("SELECT emp_name, bonus FROM employees WHERE bonus IS NULL"))
print("bonus IS NOT NULL:\n", q("SELECT emp_name, bonus FROM employees WHERE bonus IS NOT NULL"))
print("\nNULL comparisons are UNKNOWN (never TRUE):")
print(q("SELECT 1 = NULL AS eq, NULL = NULL AS null_eq, NULL IS NULL AS is_null"))

# %%
# ORDER BY — multiple keys, DESC, expressions; NULL placement
print(q("SELECT emp_name, dept_id, salary FROM employees ORDER BY dept_id ASC, salary DESC"))
print("\nORDER BY expression (name length):",
      q("SELECT emp_name, LENGTH(emp_name) AS len FROM employees ORDER BY len DESC LIMIT 3"))
# NOTE (dialect): SQLite puts NULLs FIRST in ASC; emulate NULLS LAST (PostgreSQL/Oracle style):
print("\nbonus ASC with NULLS LAST emulation:\n",
      q("SELECT emp_name, bonus FROM employees ORDER BY (bonus IS NULL), bonus ASC"))

# %%
# LIMIT / OFFSET — pagination (LIMIT 5 OFFSET 5 = page 2)
print(q("SELECT emp_id, emp_name FROM employees ORDER BY emp_id LIMIT 5"))
print("\npage 2:\n", q("SELECT emp_id, emp_name FROM employees ORDER BY emp_id LIMIT 5 OFFSET 5"))

# %%
# CASE WHEN — vectorised if/else inside SQL
print(q("""
    SELECT emp_name, salary,
           CASE
             WHEN salary >= 85000 THEN 'Senior'
             WHEN salary >= 60000 THEN 'Mid'
             ELSE 'Junior'
           END AS band,
           CASE WHEN bonus IS NULL THEN 'no bonus' ELSE 'has bonus' END AS bonus_flag
    FROM employees
    ORDER BY salary DESC
"""))

# %% [markdown]
# **Section notes — SELECT:**
#
# | Task | Syntax |
# |---|---|
# | filter | `WHERE cond [AND/OR/NOT cond]` |
# | ranges | `BETWEEN a AND b` |
# | sets | `IN (v1, v2)` |
# | patterns | `LIKE 'a%_'` |
# | nulls | `IS NULL` / `IS NOT NULL` |
# | sort | `ORDER BY col [DESC], col2` |
# | page | `LIMIT n OFFSET m` |
# | if/else | `CASE WHEN ... THEN ... ELSE ... END` |
#
# > **Gotcha:** `!=` and `<>` are both "not equal" (standard is `<>`).
# > `WHERE col = NULL` **never** matches — always `IS NULL`.

# %% [markdown]
# ---
# ## 5. Aggregation & grouping
#
# ▶ Session 33 - SQL Grouping + Sorting: https://www.youtube.com/watch?v=nsKcmOly0UY

# %%
# aggregate functions
print(q("""
    SELECT COUNT(*)        AS n_rows,
           COUNT(bonus)    AS n_with_bonus,        -- ignores NULLs!
           COUNT(DISTINCT city) AS n_cities,
           MIN(salary)     AS min_sal,
           MAX(salary)     AS max_sal,
           ROUND(AVG(salary), 1) AS avg_sal,
           SUM(salary)     AS total_sal
    FROM employees
"""))
# COUNT(*) counts rows · COUNT(col) counts non-NULL col · COUNT(DISTINCT col) counts distinct non-NULL

# %%
# GROUP BY — one row per group
print(q("""
    SELECT dept_id, COUNT(*) AS headcount, ROUND(AVG(salary), 0) AS avg_salary
    FROM employees
    GROUP BY dept_id
    ORDER BY dept_id
"""))

print("\ngroup by 2 columns (dept × city):\n", q("""
    SELECT dept_id, city, COUNT(*) AS n, SUM(salary) AS total
    FROM employees
    GROUP BY dept_id, city
    ORDER BY dept_id, city
"""))

# %%
# ### HAVING vs WHERE
print("WHERE  filters ROWS before grouping:\n",
      q("SELECT emp_name FROM employees WHERE salary > 70000"))

print("\nHAVING filters GROUPS after aggregation:\n", q("""
    SELECT dept_id, COUNT(*) AS n, ROUND(AVG(salary)) AS avg_sal
    FROM employees
    GROUP BY dept_id
    HAVING COUNT(*) >= 2 AND AVG(salary) > 60000
    ORDER BY dept_id
"""))

# %%
# ### Execution order — why HAVING exists
# ```text
# FROM  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT  →  ORDER BY  →  LIMIT
#  rows     pre-       collapse    filter     project     sort        cut
#           filter     to groups   groups     columns
# ```
# `SELECT` runs *after* grouping — that's why you cannot use a column alias in
# `WHERE`/`HAVING` (the alias doesn't exist yet), but you CAN in `ORDER BY`.

# %%
# GROUPING SETS / ROLLUP: subtotals in one scan (supported in PG/MySQL & some
# SQLite builds — NOT in this one, so emulate with UNION ALL):
try:
    print(q("SELECT dept_id, SUM(salary) FROM employees GROUP BY GROUPING SETS((dept_id), ())"))
except Exception as e:
    print("GROUPING SETS unsupported here ->", e)
    print("manual rollup emulation:\n", q("""
        SELECT CAST(dept_id AS TEXT) AS grp, SUM(salary) AS total FROM employees GROUP BY dept_id
        UNION ALL
        SELECT 'ALL', SUM(salary) FROM employees
        ORDER BY grp
    """))

# %%
# COUNT(*) vs COUNT(col) vs COUNT(DISTINCT col) — with real NULLs in the data
print(q("""
    SELECT COUNT(*)            AS rows_total,
           COUNT(bonus)        AS non_null_bonus,
           COUNT(*) - COUNT(bonus) AS null_bonus,
           COUNT(DISTINCT city) AS distinct_cities
    FROM employees
"""))

# %% [markdown]
# **Section notes — aggregation:**
#
# | Function | Ignores NULLs? | Notes |
# |---|---|---|
# | `COUNT(*)` | no (counts rows) | fastest row count |
# | `COUNT(col)` | yes | counts non-NULL values |
# | `COUNT(DISTINCT col)` | yes | distinct non-NULL values |
# | `SUM/AVG/MIN/MAX` | yes | `AVG` skips NULLs — mind denominators |
# | `ROUND(x, n)` | n/a | rounding, not aggregation |
# | `GROUP BY` | — | every selected non-agg column must appear here |
# | `HAVING` | — | aggregated filter; `WHERE` cannot use aggregates |
#
# > **Gotcha:** `WHERE salary > AVG(salary)` is illegal (aggregate on a row level).
# > Use a subquery or a window function: `AVG(salary) OVER ()` (§9).

# %% [markdown]
# ---
# ## 6. String & date functions (SQLite flavour)
#
# ▶ Session on Datetime in SQL: https://www.youtube.com/watch?v=OY9YiALhSR4
# ▶ SQL Datetime Case Study on Flights: https://www.youtube.com/watch?v=zY1UCTko64E
#
# **NOTE (dialect):** names shift across engines — MySQL `CONCAT()/DATE_FORMAT()`,
# PostgreSQL `||`/`EXTRACT()/to_char()`, SQL Server `LEN()/GETDATE()`. The
# *concepts* are identical.

# %%
# strings: UPPER LOWER LENGTH SUBSTR TRIM REPLACE and || (concat)
print(q("""
    SELECT emp_name,
           UPPER(emp_name)           AS upper_name,
           LOWER(emp_name)           AS lower_name,
           LENGTH(emp_name)          AS name_len,
           SUBSTR(emp_name, 1, 3)    AS first_3,
           TRIM('   padded   ')      AS trimmed,
           REPLACE(city, 'Bengaluru', 'BLR') AS city_short,
           'Mr. ' || emp_name        AS salutation   -- || is string concat
    FROM employees
    LIMIT 5
"""))
print("\nNULL concat is NULL (SQL 3-valued logic):", q("SELECT 'a' || NULL AS x"))

# %%
# numbers: ROUND, integer division gotcha, CAST
print(q("""
    SELECT ROUND(3.14159, 2) AS rounded,
           7 / 2             AS int_div,      -- 3  (both ints -> integer division in most engines)
           7.0 / 2           AS real_div,     -- 3.5
           CAST('42' AS INT) + 1 AS casted,
           ABS(-45)          AS absolute
"""))
# NOTE: SQLite's / between two INTEGERs IS integer division — cast one side to REAL.

# %%
# dates: strftime is the workhorse
print(q("""
    SELECT hire_date,
           strftime('%Y', hire_date)  AS yr,
           strftime('%m', hire_date)  AS mo,
           strftime('%d', hire_date)  AS dy,
           strftime('%Y-%m', hire_date) AS ym,
           strftime('%w', hire_date)  AS weekday_num,   -- 0 = Sunday
           date(hire_date, '+30 days')  AS plus_30d,
           date(hire_date, '-1 year')   AS minus_1y
    FROM employees
    ORDER BY hire_date
    LIMIT 6
"""))

# %%
# date arithmetic, "now", and EXTRACT equivalents
print(q("""
    SELECT date('now')                    AS today,
           datetime('now', '-2 days')     AS two_days_ago,
           date('2024-01-15', '+1 month') AS plus_a_month,
           julianday('2024-03-10') - julianday('2024-03-01') AS day_gap,
           CAST(strftime('%Y', '2024-03-15') AS INT) AS extract_year
"""))
# NOTE (dialect): other engines use EXTRACT(YEAR FROM d), DATEADD(month, 1, d),
# DATEDIFF(...), NOW()/CURRENT_DATE. SQLite: strftime + date/time modifiers.

# %%
# NULL helpers: COALESCE (first non-null), IFNULL (2-arg SQLite), NULLIF (equal -> NULL)
print(q("""
    SELECT emp_name, bonus,
           COALESCE(bonus, 0)                    AS bonus_or_zero,
           IFNULL(city, 'unknown')               AS city_safe,
           NULLIF(dept_id, 6)                    AS null_if_support,   -- 6 -> NULL
           COALESCE(bonus, 0) * 2                AS bonus_double
    FROM employees
    ORDER BY emp_id
    LIMIT 6
"""))

# %% [markdown]
# **Section notes — functions:**
#
# | Need | SQLite | MySQL | PostgreSQL |
# |---|---|---|---|
# | concat | `\|\|` | `CONCAT()` | `\|\|` |
# | upper/lower/len | `UPPER/LOWER/LENGTH` | same (`CHAR_LENGTH`) | `UPPER/LOWER/LENGTH` |
# | substring | `SUBSTR(s, from, len)` | `SUBSTR`/`SUBSTRING` | `SUBSTRING` |
# | trim | `TRIM(s)` | `TRIM/LTRIM/RTRIM` | `TRIM` |
# | replace | `REPLACE(s, a, b)` | same | `REPLACE` |
# | round | `ROUND(x, n)` | same | same |
# | date parts | `strftime('%Y', d)` | `YEAR(d)`, `DATE_FORMAT` | `EXTRACT(YEAR FROM d)` |
# | now | `datetime('now')` | `NOW()` | `NOW()` |
# | day diff | `julianday(a) - julianday(b)` | `DATEDIFF(a, b)` | `a - b` (date) |
# | nulls | `COALESCE/IFNULL/NULLIF` | `IFNULL` exists | `COALESCE/NULLIF` |

# %% [markdown]
# ---
# ## 7. Joins
#
# ▶ Session 34 - SQL Joins: https://www.youtube.com/watch?v=s6xAVqYdmJ4
#
# | Join | Rows returned | "venn" |
# |---|---|---|
# | `INNER` | matching rows only | A ∩ B |
# | `LEFT` | all of A + matches from B (NULL if none) | A ⊇ (A ∩ B) |
# | `RIGHT` | all of B + matches from A | B ⊇ (A ∩ B) — *swap the sides on SQLite < 3.39* |
# | `FULL OUTER` | all of A and all of B | A ∪ B — *emulated below* |
# | `CROSS` | every pair (n × m rows) | A × B |
# | self | a table joined to **itself** | hierarchy/manager lookup |

# %%
# INNER JOIN — only matches
print(q("""
    SELECT e.emp_name, d.dept_name, d.location
    FROM employees e
    INNER JOIN departments d ON e.dept_id = d.dept_id
    ORDER BY d.dept_name, e.emp_name
"""))

# %%
# LEFT JOIN — keep every employee even if dept were missing (it isn't, but Legal...)
print(q("""
    SELECT d.dept_name, COUNT(e.emp_id) AS headcount
    FROM departments d
    LEFT JOIN employees e ON e.dept_id = d.dept_id
    GROUP BY d.dept_name
    ORDER BY headcount DESC
"""))
# 'Legal' shows 0 — proof the LEFT side is fully preserved.

# %%
# RIGHT JOIN: this SQLite build (3.35.5) predates 3.39 — A RIGHT JOIN B
# is *definitionally* the same as B LEFT JOIN A, so just swap the sides:
print("emulated RIGHT JOIN (employees RIGHT JOIN departments == departments LEFT JOIN employees):\n", q("""
    SELECT e.emp_name, d.dept_name
    FROM departments d
    LEFT JOIN employees e ON e.dept_id = d.dept_id
    ORDER BY d.dept_name, e.emp_name
"""))

# %%
# FULL OUTER JOIN emulation: LEFT JOIN  UNION ALL  anti-join of the right side
print("emulated FULL OUTER JOIN:\n", q("""
    SELECT e.emp_name, d.dept_name, 'matched/employee-side' AS side
    FROM employees e
    LEFT JOIN departments d ON e.dept_id = d.dept_id
    UNION ALL
    SELECT e.emp_name, d.dept_name, 'department-only' AS side
    FROM departments d
    LEFT JOIN employees e ON e.dept_id = d.dept_id
    WHERE e.emp_id IS NULL
    ORDER BY side, d.dept_name
"""))

# %%
# CROSS JOIN (7 depts × 12 employees = 84 pairs) and a non-equi join
print("cross join row count:", q("""
    SELECT COUNT(*) AS n FROM departments CROSS JOIN employees
""")["n"][0])

con.executescript("""
CREATE TABLE salary_bands(band TEXT, lo REAL, hi REAL);
INSERT INTO salary_bands VALUES
 ('Junior',      0, 60000),
 ('Mid',     60000, 85000),
 ('Senior',  85000, 1000000);
""")

print("\nnon-equi join — match employees to a RANGE table (condition is not '='):\n", q("""
    SELECT e.emp_name, e.salary, b.band
    FROM employees e
    JOIN salary_bands b
      ON e.salary >= b.lo AND e.salary < b.hi      -- inequality, no equality involved
    ORDER BY e.salary DESC
"""))

# %%
# SELF JOIN — employees and THEIR managers
print(q("""
    SELECT e.emp_name            AS employee,
           COALESCE(m.emp_name, '(no manager)') AS manager,
           e.dept_id
    FROM employees e
    LEFT JOIN employees m ON e.manager_id = m.emp_id
    ORDER BY manager, employee
"""))

# %%
# join on MULTIPLE keys + join ... WHERE gotcha (filter in ON vs WHERE)
print("join on two keys:\n", q("""
    SELECT o.order_id, c.cust_name, c.city, o.amount
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id AND c.city = 'Delhi'
    ORDER BY o.order_id
"""))

left = "SELECT COUNT(*) AS n FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id"
with_filter = "SELECT COUNT(*) AS n FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id WHERE e.emp_id IS NOT NULL"
print("\nLEFT JOIN rows                :", q(left)["n"][0])
print("LEFT JOIN + WHERE e.emp_id... :", q(with_filter)["n"][0],
      " <- WHERE on the right table silently turns LEFT into INNER!")

# %% [markdown]
# > **Gotcha:** put filters on the **right** table inside the `ON` clause to
# > keep the left table intact; filters in `WHERE` run *after* the join and
# > cancel a LEFT JOIN for any row that failed the match.
#
# **Section notes — joins:**
#
# | Goal | Query |
# |---|---|
# | matches only | `FROM a JOIN b ON a.k = b.k` |
# | keep all of a | `FROM a LEFT JOIN b ON ...` |
# | keep all of b | `FROM b LEFT JOIN a ON ...` (aka RIGHT JOIN) |
# | keep everything | `LEFT JOIN ... UNION ALL ... anti-join` (SQLite < 3.39) |
# | all pairs | `FROM a CROSS JOIN b` |
# | hierarchy | `FROM t x LEFT JOIN t y ON x.manager_id = y.id` |

# %% [markdown]
# ---
# ## 8. Subqueries & CTEs
#
# ▶ Session 35 - Subqueries in SQL: https://www.youtube.com/watch?v=YYq47MN3TZI

# %%
# 1. SCALAR subquery — a single value usable like a constant
print(q("""
    SELECT emp_name, salary,
           (SELECT MAX(salary) FROM employees)          AS company_max,
           ROUND(100.0 * salary / (SELECT MAX(salary) FROM employees), 1) AS pct_of_max
    FROM employees
    ORDER BY salary DESC
    LIMIT 5
"""))

# %%
# 2. IN / NOT IN subquery — customers who never ordered
print("customers with NO orders (NOT IN):\n", q("""
    SELECT cust_name FROM customers
    WHERE customer_id NOT IN (SELECT customer_id FROM orders)
    ORDER BY cust_name
"""))
print("customers WITH orders (IN):\n", q("""
    SELECT cust_name FROM customers
    WHERE customer_id IN (SELECT DISTINCT customer_id FROM orders)
    ORDER BY cust_name
"""))
# NOTE: NOT IN returns NOTHING if the subquery yields any NULL — prefer NOT EXISTS.

# %%
# 3. EXISTS / NOT EXISTS — correlated, short-circuits per row
print("NOT EXISTS version (same answer, safer):\n", q("""
    SELECT cust_name FROM customers c
    WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)
"""))
print("\nEXISTS — departments that have at least one employee:\n", q("""
    SELECT dept_name FROM departments d
    WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id)
"""))

# %%
# 4. CORRELATED subquery — the inner query re-runs ONCE PER OUTER ROW
print("employees earning more than their OWN department average:\n", q("""
    SELECT e.emp_name, e.dept_id, e.salary
    FROM employees e
    WHERE e.salary > (
        SELECT AVG(salary) FROM employees sub WHERE sub.dept_id = e.dept_id
    )
    ORDER BY e.dept_id, e.salary DESC
"""))
# "correlated" = inner query references e.dept_id → cannot run alone;
# think of it as a per-row lookup table.

# %%
# 5. DERIVED TABLE — subquery in FROM (must be aliased)
print(q("""
    SELECT g.dept_name, g.headcount, g.avg_sal
    FROM (
        SELECT d.dept_name,
               COUNT(e.emp_id)   AS headcount,
               ROUND(AVG(e.salary)) AS avg_sal
        FROM employees e
        JOIN departments d ON e.dept_id = d.dept_id
        GROUP BY d.dept_name
    ) AS g                      -- <- alias is mandatory
    WHERE g.headcount >= 2
    ORDER BY g.avg_sal DESC
"""))

# %%
# 6. CTE with WITH — named, readable subquery (may be referenced multiple times)
print(q("""
    WITH dept_stats AS (
        SELECT dept_id, COUNT(*) AS n, AVG(salary) AS avg_sal
        FROM employees
        GROUP BY dept_id
    ),
    ranked AS (
        SELECT dept_id, n, ROUND(avg_sal) AS avg_sal,
               RANK() OVER (ORDER BY avg_sal DESC) AS rnk
        FROM dept_stats
    )
    SELECT r.dept_id, d.dept_name, r.n, r.avg_sal, r.rnk
    FROM ranked r
    JOIN departments d ON d.dept_id = r.dept_id
    ORDER BY r.rnk
"""))

# %%
# 6b. RECURSIVE CTE — org hierarchy (manager → reports → their reports ...)
con.executescript("""
CREATE TABLE org_tree(
    emp_id INTEGER PRIMARY KEY,
    emp_name TEXT,
    manager_id INTEGER
);
INSERT INTO org_tree VALUES
 (1, 'CEO Priya',  NULL),
 (2, 'Rahul',      1),
 (3, 'Vikram',     1),
 (4, 'Priya Eng',  2),
 (5, 'Rohan',      2),
 (6, 'Neha',       3),
 (7, 'Pooja',      6);
""")

print(q("""
    WITH RECURSIVE chain(emp_id, emp_name, manager_id, depth, path) AS (
        SELECT emp_id, emp_name, manager_id, 0, emp_name       -- anchor member
        FROM org_tree WHERE manager_id IS NULL
        UNION ALL
        SELECT c.emp_id, c.emp_name, c.manager_id, ch.depth + 1,
               ch.path || ' > ' || c.emp_name                   -- recursive member
        FROM org_tree c
        JOIN chain ch ON c.manager_id = ch.emp_id
    )
    SELECT printf('%*s%s', depth * 4, '', emp_name) AS org,
           depth, path
    FROM chain ORDER BY path
"""))
# anchor (root) runs once; the recursive part joins back until no new rows appear.

# %%
# 7. ANY / ALL — NOT supported by SQLite; rewrite with aggregates instead:
#    WHERE salary > ANY (SELECT ...)   ==  WHERE salary > (SELECT MIN(...) ...)
#    WHERE salary > ALL (SELECT ...)   ==  WHERE salary > (SELECT MAX(...) ...)
print("ALL rewritten with MAX():\n", q("""
    SELECT emp_name, salary FROM employees
    WHERE salary > (SELECT MAX(salary) FROM employees WHERE dept_id = 1)
"""))
print("\nANY rewritten with MIN():\n", q("""
    SELECT dept_id, emp_name, salary FROM employees e
    WHERE salary > (SELECT MIN(salary) FROM employees WHERE dept_id = e.dept_id)
    ORDER BY dept_id, salary DESC
"""))

# %% [markdown]
# **Section notes — subqueries:**
#
# | Type | Where it sits | Runs |
# |---|---|---|
# | scalar | any expression position | once |
# | `IN (subquery)` | `WHERE` | once (semi-join) |
# | `EXISTS` | `WHERE` | short-circuits per row |
# | correlated | `WHERE` with outer ref | **once per outer row** |
# | derived | `FROM` (aliased) | once |
# | CTE `WITH` | before `SELECT` | named, reusable (may be referenced twice) |
# | recursive CTE | `WITH RECURSIVE` | anchor + loop until empty |
# | `ANY/ALL` | `WHERE` | ❌ SQLite → use `MIN()`/`MAX()` |

# %% [markdown]
# ---
# ## 9. Window functions
#
# ▶ Session 36 - Window Functions in SQL: https://www.youtube.com/watch?v=3LJ-yWBQBGs
# ▶ Session 37 - Window Functions Part 2: https://www.youtube.com/watch?v=YgizUxmBFK8
# ▶ Session 37 (Cont..) - Window Functions Part 3: https://www.youtube.com/watch?v=e0HQVOdfqlQ
#
# A window function computes across a **set of rows related to the current row**
# **without collapsing them** — that is the whole difference from `GROUP BY`.

# %%
# ### ROW_NUMBER vs RANK vs DENSE_RANK (ties break them apart)
print(q("""
    SELECT emp_name, dept_id, salary,
           ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn,
           RANK()       OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk,
           DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dense_rnk
    FROM employees
    WHERE dept_id = 2            -- IT: Rahul & Rohan TIE at 90000
    ORDER BY salary DESC
"""))
# ROW_NUMBER: always 1,2,3 (ties broken arbitrarily)
# RANK:       1,1,3  (gap after ties)
# DENSE_RANK: 1,1,2  (no gap)

# %%
# NTILE — split rows into n balanced buckets
print(q("""
    SELECT emp_name, salary,
           NTILE(4) OVER (ORDER BY salary DESC) AS quartile
    FROM employees
    ORDER BY salary DESC
"""))

# %%
# LAG / LEAD — look at previous / next row (with optional default)
print(q("""
    SELECT emp_name, hire_date, salary,
           LAG(salary)  OVER (ORDER BY hire_date)        AS prev_hire_salary,
           LEAD(salary) OVER (ORDER BY hire_date)        AS next_hire_salary,
           LAG(salary, 1, 0) OVER (ORDER BY hire_date)   AS prev_or_zero
    FROM employees
    ORDER BY hire_date
"""))
# LAG(col, n, default): n = rows back (1 by default), default shown when no row exists.

# %%
# FIRST_VALUE / LAST_VALUE — head/tail of the window (frame matters!)
print(q("""
    SELECT emp_name, dept_id, hire_date,
           FIRST_VALUE(emp_name) OVER (
               PARTITION BY dept_id ORDER BY hire_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
           ) AS first_hired_in_dept,
           LAST_VALUE(emp_name) OVER (
               PARTITION BY dept_id ORDER BY hire_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS most_recent_upto_now
    FROM employees
    ORDER BY dept_id, hire_date
"""))
# WARNING: LAST_VALUE's default frame ends at CURRENT ROW — without an explicit
# frame you silently get the current row, not the last row of the group.

# %%
# RUNNING TOTAL — SUM() OVER (ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
print(q("""
    SELECT order_id, order_date, amount,
           SUM(amount) OVER (
               ORDER BY order_date, order_id
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS running_total
    FROM orders
    ORDER BY order_date, order_id
"""))

# %%
# MOVING AVERAGE (3-row window: current + 2 preceding)
print(q("""
    SELECT order_id, amount,
           ROUND(AVG(amount) OVER (
               ORDER BY order_id
               ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
           ), 1) AS moving_avg_3
    FROM orders
    ORDER BY order_id
"""))

# %%
# RANKING TOP-N PER GROUP — ROW_NUMBER (exactly N) vs RANK (include ties)
print("top 2 earners per department (ROW_NUMBER — exactly 2 rows each):\n", q("""
    SELECT * FROM (
        SELECT emp_name, dept_id, salary,
               ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC, emp_id) AS rn
        FROM employees
    )
    WHERE rn <= 2
    ORDER BY dept_id, rn
"""))

print("\nrank of every IT employee (RANK keeps the tie visible):\n", q("""
    SELECT emp_name, salary,
           RANK()       OVER (ORDER BY salary DESC) AS rnk,
           ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
    WHERE dept_id = 2
"""))

# %%
# AGGREGATE WINDOW vs GROUP BY — same number, rows preserved
print("window: every row kept, group stats attached:\n", q("""
    SELECT emp_name, dept_id, salary,
           COUNT(*)    OVER (PARTITION BY dept_id)          AS dept_size,
           ROUND(AVG(salary) OVER (PARTITION BY dept_id))   AS dept_avg,
           salary - ROUND(AVG(salary) OVER (PARTITION BY dept_id)) AS vs_avg
    FROM employees
    ORDER BY dept_id, salary DESC
"""))

print("\nGROUP BY: one row per group (detail is gone):\n", q("""
    SELECT dept_id, COUNT(*) AS dept_size, ROUND(AVG(salary)) AS dept_avg
    FROM employees
    GROUP BY dept_id
"""))

# %% [markdown]
# ### `OVER()` anatomy
#
# ```text
# function(args) OVER (
#     PARTITION BY col        -- split into groups (like GROUP BY, per row)
#     ORDER BY col [ASC|DESC] -- order inside each group
#     ROWS BETWEEN a AND b    -- physical frame (rows relative to current row)
# )
# ```
#
# | Frame | Meaning |
# |---|---|
# | `UNBOUNDED PRECEDING → CURRENT ROW` | start of partition → now (running) |
# | `2 PRECEDING → CURRENT ROW` | 3-row moving window |
# | `UNBOUNDED PRECEDING → UNBOUNDED FOLLOWING` | whole partition |
# | `BETWEEN CURRENT ROW AND 1 FOLLOWING` | now → next row |
#
# | | `GROUP BY` | window `OVER` |
# |---|---|---|
# | output rows | 1 per group | 1 per **input** row |
# | detail columns | must be grouped/aggregated | fully preserved |
# | can mix group + row values | ❌ | ✅ (`salary - AVG(...) OVER (...)`) |
# | typical use | reports, summaries | rankings, running totals, MoM |
#
# **Section notes — windows:**
#
# | Need | Function |
# |---|---|
# | sequential id in group | `ROW_NUMBER() OVER (PARTITION BY g ORDER BY x)` |
# | dense competition rank | `RANK()` / `DENSE_RANK()` |
# | buckets | `NTILE(n)` |
# | previous/next value | `LAG(col, n, default)` / `LEAD(...)` |
# | first/last in group | `FIRST_VALUE` / `LAST_VALUE` (+ explicit frame!) |
# | running total | `SUM() OVER (ORDER BY d ROWS UNBOUNDED PRECEDING→CURRENT)` |
# | moving average | `AVG() OVER (ORDER BY d ROWS 2 PRECEDING→CURRENT)` |
# | per-group stats | aggregate + `OVER (PARTITION BY g)` |

# %% [markdown]
# ---
# ## 10. Set operations
#
# ▶ Session 34 - SQL Joins: https://www.youtube.com/watch?v=s6xAVqYdmJ4

# %%
# UNION (removes duplicates) vs UNION ALL (keeps them — faster)
print("UNION     :", q("SELECT city FROM customers UNION SELECT city FROM employees ORDER BY city")["city"].tolist())
print("UNION ALL :", len(q("SELECT city FROM customers UNION ALL SELECT city FROM employees")), "rows (with dupes)")

dupes = q("""
    SELECT city FROM customers
    UNION ALL
    SELECT city FROM employees
""")["city"].value_counts()
print("dupes UNION ALL would remove:\n", dupes[dupes > 1])

# %%
# INTERSECT (in both) and EXCEPT (in first, not second)
print("cities in BOTH customers and employees (INTERSECT):\n",
      q("SELECT city FROM customers INTERSECT SELECT city FROM employees ORDER BY city"))

print("\nemployee cities NOT in customers (EXCEPT):\n",
      q("SELECT city FROM employees EXCEPT SELECT city FROM customers ORDER BY city"))

# %% [markdown]
# | Operator | Keeps dupes? | Rows |
# |---|---|---|
# | `UNION` | no (distinct) | distinct rows of A ∪ B |
# | `UNION ALL` | yes | every row of A and B (cheaper) |
# | `INTERSECT` | no | rows in **both** |
# | `EXCEPT` | no | rows in A **not** in B |
#
# > **Gotcha:** all arms must have the **same number of columns** with compatible
# > types; `ORDER BY` applies to the *whole* result and goes at the very end.
# > Set ops compare NULLs as *not equal* in some engines — test before relying on it.

# %% [markdown]
# ---
# ## 11. Views & indexes
#
# ▶ Session on Views and UDFs in SQL: https://www.youtube.com/watch?v=dMoKTx1bVnY

# %%
# CREATE VIEW — a saved query, not a copy of the data
con.execute("""
    CREATE VIEW v_emp_overview AS
    SELECT e.emp_id, e.emp_name, d.dept_name, e.salary, e.hire_date
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
""")
print(q("SELECT * FROM v_emp_overview ORDER BY salary DESC LIMIT 5"))

# views compose: query a view like a table
print("\navg salary per dept, from the view:\n", q("""
    SELECT dept_name, COUNT(*) AS n, ROUND(AVG(salary)) AS avg_sal
    FROM v_emp_overview
    GROUP BY dept_name
    ORDER BY avg_sal DESC
"""))

con.execute("DROP VIEW v_emp_overview")
print("views left:", q("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='view'")["n"][0])
# NOTE: a view stores only its SQL — change the base table and the view follows.

# %%
# EXPLAIN QUERY PLAN — see whether an index is used
plan_before = q("EXPLAIN QUERY PLAN SELECT * FROM employees WHERE dept_id = 2")
print("BEFORE index:\n", plan_before)

con.execute("CREATE INDEX idx_emp_dept ON employees(dept_id)")
plan_after = q("EXPLAIN QUERY PLAN SELECT * FROM employees WHERE dept_id = 2")
print("\nAFTER index:\n", plan_after)
# SCAN TABLE = full table scan (O(n))  →  SEARCH ... USING INDEX (O(log n))

# %%
# UNIQUE index — enforces uniqueness like a constraint
con.execute("CREATE UNIQUE INDEX idx_emp_name ON employees(emp_name)")
try:
    con.execute("INSERT INTO employees VALUES(99, 'Rahul', 1, NULL, 50000, NULL, '2024-01-01', 30, 'Pune')")
except sqlite3.IntegrityError as e:
    print("duplicate name rejected ->", e)

print("\nindexes on employees:",
      q("SELECT name FROM pragma_index_list('employees')")["name"].tolist())
# NOTE (dialect): PostgreSQL adds "USING gin/gin" for JSON/arrays; MySQL has
# FULLTEXT indexes. Indexes speed reads but slow writes (maintained on INSERT).

# %% [markdown]
# **Section notes — views & indexes:**
#
# | Object | Purpose | Cost |
# |---|---|---|
# | `CREATE VIEW v AS SELECT ...` | reusable query, security, simplification | none (SQL only) |
# | `CREATE INDEX i ON t(col)` | fast `WHERE col = ...` lookups | slower writes, disk |
# | `CREATE UNIQUE INDEX` | uniqueness without a table constraint | same |
# | `EXPLAIN QUERY PLAN` | does the planner use my index? | — |
# | `DROP VIEW/INDEX` | remove it | — |
#
# > **Gotcha:** an index on `(dept_id, salary)` can serve `WHERE dept_id = ?`
# > but an index on `(salary, dept_id)` **cannot** — column order matters.

# %% [markdown]
# ---
# ## 12. Transactions & ACID
#
# ▶ Session on Transactions & Stored Procedures in SQL: https://www.youtube.com/watch?v=-O_9iR7XsJw

# %%
con.execute("CREATE TABLE accounts(acc TEXT PRIMARY KEY, bal REAL NOT NULL)")
con.execute("INSERT INTO accounts VALUES('A', 1000), ('B', 500)")
print("start:\n", q("SELECT * FROM accounts"))

# --- ROLLBACK discards the work ---
con.execute("BEGIN")
con.execute("UPDATE accounts SET bal = bal - 200 WHERE acc = 'A'")
con.execute("UPDATE accounts SET bal = bal + 200 WHERE acc = 'B'")
print("\nuncommitted (visible inside the transaction):\n", q("SELECT * FROM accounts"))
con.execute("ROLLBACK")
print("after ROLLBACK:\n", q("SELECT * FROM accounts"))

# --- COMMIT persists the work ---
con.execute("BEGIN")
con.execute("UPDATE accounts SET bal = bal - 200 WHERE acc = 'A'")
con.execute("UPDATE accounts SET bal = bal + 200 WHERE acc = 'B'")
con.execute("COMMIT")
print("\nafter COMMIT:\n", q("SELECT * FROM accounts"))

# --- SAVEPOINT = partial rollback inside a transaction ---
con.execute("BEGIN")
con.execute("UPDATE accounts SET bal = bal - 100 WHERE acc = 'A'")
con.execute("SAVEPOINT sp1")
con.execute("UPDATE accounts SET bal = bal - 100 WHERE acc = 'A'")   # this one gets undone
con.execute("ROLLBACK TO sp1")
con.execute("COMMIT")
print("\nafter SAVEPOINT rollback (A lost only the first 100):\n", q("SELECT * FROM accounts"))

# %% [markdown]
# ### ACID
#
# | Property | Meaning | Example above |
# |---|---|---|
# | **A**tomicity | all-or-nothing: every statement in the tx commits or none does | ROLLBACK removed *both* transfers |
# | **C**onsistency | tx moves the DB from one valid state to another (constraints hold) | `bal` can never violate `CHECK`/PK |
# | **I**solation | concurrent tx don't see each other's uncommitted work | other sessions can't see the pending transfer |
# | **D**urability | once committed, survives crashes (WAL/journal on disk) | COMMIT is permanent |
#
# ### Isolation levels & read anomalies
#
# | Level | Dirty read | Non-repeatable read | Phantom read |
# |---|---|---|---|
# | READ UNCOMMITTED | ✅ possible | ✅ | ✅ |
# | READ COMMITTED | prevented | ✅ | ✅ |
# | REPEATABLE READ | prevented | prevented | ✅ (some engines) |
# | SERIALIZABLE | prevented | prevented | prevented |
#
# - **Dirty read** — you read another tx's *uncommitted* data.
# - **Non-repeatable read** — same query twice → different value (someone committed in between).
# - **Phantom read** — same query twice → different *set of rows* (inserted/deleted).
# - SQLite's default: serializable (a write lock blocks other writers); readers use shared locks.
#
# ### Stored procedures — not supported by SQLite
# SQLite executes statements only; there are no stored procedures/UDF blocks.
# **MySQL syntax for reference** (what you would write on a server):
# ```sql
# -- DELIMITER $$
# -- CREATE PROCEDURE raise_salary(IN p_dept INT, IN pct DOUBLE)
# -- BEGIN
# --     UPDATE employees SET salary = salary * (1 + pct/100) WHERE dept_id = p_dept;
# --     SELECT ROW_COUNT() AS updated;
# -- END $$
# -- DELIMITER ;
# -- CALL raise_salary(2, 10);
# ```
#
# **Section notes — transactions:**
#
# | Task | Statement |
# |---|---|
# | start | `BEGIN` (or `BEGIN TRANSACTION`) |
# | keep | `COMMIT` |
# | undo all | `ROLLBACK` |
# | undo part | `SAVEPOINT sp` → `ROLLBACK TO sp` |
# | other dialects | `START TRANSACTION`, `COMMIT WORK`, `SET TRANSACTION ISOLATION LEVEL ...` |

# %% [markdown]
# ---
# ## 13. Database design & normalization
#
# ▶ Session on Database Design | Normalization: https://www.youtube.com/watch?v=8SOs0URTvr0
#
# ### Functional dependency (FD)
# `A → B` ("A functionally determines B") means each value of `A` maps to exactly
# one value of `B`. Example: `emp_id → emp_name` (an id has one name) but
# `city → pincode` may fail in the real world.
#
# ### The normal forms — worked example below
#
# | Form | Rule (in one line) | Removes |
# |---|---|---|
# | **1NF** | every cell is atomic; no repeating groups/arrays | multi-valued cells |
# | **2NF** | 1NF + no **partial** dependency (non-key col must depend on the *whole* PK) | attributes of a composite key |
# | **3NF** | 2NF + no **transitive** dependency (non-key → non-key) | columns derivable from other columns |
# | **BCNF** | every determinant is a candidate key | exotic FD violations (rare in practice) |

# %%
# UNNORMALIZED source table: repeating customer & product facts in every line
con.executescript("""
CREATE TABLE sales_unnorm (
    order_id    INTEGER,
    cust_name   TEXT,
    cust_city   TEXT,
    product     TEXT,
    qty         INTEGER,
    unit_price  REAL
);
INSERT INTO sales_unnorm VALUES
 (1, 'Amit',  'Delhi',   'Laptop',  1, 55000),
 (1, 'Amit',  'Delhi',   'Mouse',   2,   500),
 (2, 'Priya', 'Pune',    'Laptop',  1, 55000),
 (3, 'Sneha', 'Mumbai',  'Keyboard',1,  1500);
""")
print("UNNORMALIZED (1NF violations waiting to happen — redundant city/price):\n",
      q("SELECT * FROM sales_unnorm"))
# FDs here: order_id -> cust_name, cust_city   |   product -> unit_price
#           cust_city is TRANSITIVELY determined via cust_name  (3NF issue)

# %%
# normalize: customers (3NF) + products (2NF/3NF) + order_lines (1NF/2NF)
con.executescript("""
CREATE TABLE n_customers (
    cust_name TEXT PRIMARY KEY,
    cust_city TEXT NOT NULL
);
CREATE TABLE n_products (
    product    TEXT PRIMARY KEY,
    unit_price REAL NOT NULL CHECK(unit_price > 0)
);
CREATE TABLE n_order_lines (
    order_id   INTEGER,
    cust_name  TEXT REFERENCES n_customers(cust_name),
    product    TEXT REFERENCES n_products(product),
    qty        INTEGER NOT NULL CHECK(qty > 0),
    PRIMARY KEY (order_id, product)
);

INSERT INTO n_customers
    SELECT DISTINCT cust_name, cust_city FROM sales_unnorm;
INSERT INTO n_products
    SELECT DISTINCT product, unit_price FROM sales_unnorm;
INSERT INTO n_order_lines
    SELECT order_id, cust_name, product, qty FROM sales_unnorm;
""")
print("customers (3NF):\n", q("SELECT * FROM n_customers"))
print("\nproducts (3NF):\n", q("SELECT * FROM n_products"))
print("\norder_lines:\n", q("SELECT * FROM n_order_lines"))

# %%
# lossless join — the normalized tables reconstruct the original exactly
rebuilt = q("""
    SELECT l.order_id, c.cust_name, c.cust_city, p.product, l.qty, p.unit_price
    FROM n_order_lines l
    JOIN n_customers c ON c.cust_name = l.cust_name
    JOIN n_products  p ON p.product   = l.product
    ORDER BY l.order_id, p.product
""")
original = q("SELECT * FROM sales_unnorm ORDER BY order_id, product")
print("rebuilt == original:", rebuilt.reset_index(drop=True).equals(original.reset_index(drop=True)))
print(rebuilt)

# %% [markdown]
# ### Denormalization — the deliberate trade-off
#
# | | Normalized (3NF) | Denormalized |
# |---|---|---|
# | writes | one update → one place | updates spread, redundancy risk |
# | reads | joins needed (slower) | single wide table (faster) |
# | anomalies | none (update/insert/delete) | can reappear |
# | used for | OLTP (§1) | OLAP, data warehouses, caches |
#
# ### ER → schema rules of thumb
# - 1:N → FK on the many side (our `employees.dept_id`).
# - M:N → junction table with a composite PK (`order_lines` above).
# - 1:1 → merge into one table, or separate if the part is large/optional.
# - Prefer natural keys (`email`) when truly unique **and** stable; otherwise
#   surrogate keys (`INTEGER PRIMARY KEY`).
#
# **Section notes — design:**
#
# | Step | Question |
# |---|---|
# | 1NF | is every value atomic? |
# | 2NF | does each non-key column depend on the *whole* PK? |
# | 3NF | does any non-key column depend on another non-key column? |
# | BCNF | is every determinant a candidate key? |
# | denormalize? | only when measured read cost > write/anomaly cost |

# %% [markdown]
# ---
# ## 14. Data cleaning with SQL
#
# ▶ Session on Data Cleaning using SQL | Laptops: https://www.youtube.com/watch?v=NsOXJdRIamc

# %%
# a deliberately dirty table (raw scrape-style)
con.executescript("""
CREATE TABLE laptops_raw (
    lid    INTEGER PRIMARY KEY,
    brand  TEXT,
    model  TEXT,
    ram    TEXT,
    os     TEXT,
    rating TEXT,
    added  TEXT
);
INSERT INTO laptops_raw VALUES
 (1, ' Dell ',  'Inspiron 15', '8GB ', 'Windows 11', '4.3', '15/03/2024'),
 (2, 'dell',    'XPS 13',      ' 16 GB', 'windows11', '4.5', '2024-03-16'),
 (3, 'HP ',     'Pavilion',    '8GB',   'Windows 10', 'N/A', '2024-03-17'),
 (4, 'hp',      'EliteBook',   '16 GB', 'win10',      '4.2', '17/03/2024'),
 (5, 'Lenovo',  'IdeaPad',     '8 GB',  'Windows 11', '4.0', '2024-03-18'),
 (6, 'LENOVO',  'ThinkPad',    '32GB',  'Windows 11', '4.6', '18/03/2024'),
 (7, ' Dell ',  'Inspiron 15', '8GB ', 'Windows 11', '4.3', '15/03/2024'),
 (8, 'Asus',    'VivoBook',    NULL,    'Windows 11', NULL,  '19/03/2024');
""")
print("RAW:\n", q("SELECT * FROM laptops_raw"))

# %%
# 1) whitespace + case inconsistencies  ->  TRIM + UPPER + CASE
con.execute("""
    UPDATE laptops_raw SET
        brand = CASE UPPER(TRIM(brand))
                  WHEN 'DELL' THEN 'Dell'
                  WHEN 'HP'   THEN 'HP'
                  WHEN 'LENOVO' THEN 'Lenovo'
                  WHEN 'ASUS' THEN 'Asus'
                  ELSE TRIM(brand)
                END,
        os = CASE
               WHEN UPPER(os) LIKE 'WIN%' THEN 'Windows'
               ELSE os
             END
""")
print("brand/os fixed:\n", q("SELECT lid, brand, os FROM laptops_raw"))

# %%
# 2) recast messy numerics: ' 16 GB' -> 16 ; 'N/A' -> NULL ; rating -> REAL
con.execute("""
    UPDATE laptops_raw SET
        ram = CASE WHEN ram IS NULL THEN NULL
                   ELSE CAST(REPLACE(TRIM(ram), 'GB', '') AS INTEGER)
             END,
        rating = CASE WHEN rating = 'N/A' THEN NULL ELSE CAST(rating AS REAL) END
""")
con.execute("ALTER TABLE laptops_raw RENAME COLUMN ram TO ram_gb")
# 3) missing values -> COALESCE to a defensible default
con.execute("UPDATE laptops_raw SET ram_gb = 8 WHERE ram_gb IS NULL")          # most common
con.execute("UPDATE laptops_raw SET rating = 4.3 WHERE rating IS NULL")        # global median-ish
print("re-cast + filled:\n", q("SELECT lid, brand, ram_gb, rating FROM laptops_raw"))

# %%
# 4) inconsistent dates: 'DD/MM/YYYY' and 'YYYY-MM-DD' -> ISO date
print(q("""
    SELECT lid, added,
           CASE WHEN added LIKE '__/__/____'
                THEN date(substr(added, 7, 4) || '-' || substr(added, 4, 2) || '-' || substr(added, 1, 2))
                ELSE date(added)
           END AS iso_date
    FROM laptops_raw
"""))

# %%
# 5) deduplicate with ROW_NUMBER (keep the lowest lid per brand+model)
con.execute("""
    DELETE FROM laptops_raw
    WHERE lid IN (
        SELECT lid FROM (
            SELECT lid, ROW_NUMBER() OVER (
                       PARTITION BY brand, model ORDER BY lid
                   ) AS rn
            FROM laptops_raw
        ) WHERE rn > 1
    )
""")
print("rows after dedupe:", len(q("SELECT * FROM laptops_raw")))
print(q("SELECT lid, brand, model, ram_gb, os, rating FROM laptops_raw"))

# %%
# 6) fix inconsistent categories with UPDATE + WHERE (targeted, safe)
con.execute("UPDATE laptops_raw SET brand = 'Asus' WHERE brand = 'asus'")
print(q("SELECT DISTINCT brand FROM laptops_raw ORDER BY brand"))

# %% [markdown]
# **Section notes — SQL cleaning:**
#
# | Problem | SQL tool |
# |---|---|
# | whitespace | `TRIM/LTRIM/RTRIM` |
# | case | `UPPER/LOWER/CASE` |
# | recast type | `CAST(col AS TYPE)` / `col * 1` |
# | strip units | `REPLACE(col, 'GB', '')` then `CAST` |
# | missing | `COALESCE(col, default)` / `UPDATE ... WHERE col IS NULL` |
# | junk labels | `UPDATE ... CASE ...` |
# | duplicates | `ROW_NUMBER() OVER (PARTITION BY keys ORDER BY id)` → `DELETE ... rn > 1` |
# | bad dates | `CASE WHEN LIKE '__/__/____' THEN date(substr(...)) ELSE date(col) END` |
#
# > **Gotcha:** always `SELECT` first, wrap `UPDATE/DELETE` in a transaction
# > (`BEGIN; ...; COMMIT;`), and never mutate a raw/bronze table in place.

# %% [markdown]
# ---
# ## 15. EDA with SQL
#
# ▶ Session on EDA using SQL | Laptops: https://www.youtube.com/watch?v=dv8TkUTlQvI
#
# (Same questions as the pandas EDA checklist — answered with `GROUP BY`.)

# %%
# value counts
print("salary bands (value counts):\n", q("""
    SELECT CASE WHEN salary >= 85000 THEN 'Senior'
                WHEN salary >= 60000 THEN 'Mid'
                ELSE 'Junior' END AS band,
           COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM employees), 1) AS pct
    FROM employees
    GROUP BY band
    ORDER BY n DESC
"""))

print("\norder status value counts:\n", q("""
    SELECT status, COUNT(*) AS n, SUM(amount) AS revenue
    FROM orders GROUP BY status ORDER BY n DESC
"""))

# %%
# summary statistics — SQLite has no STDDEV()/VARIANCE(), so compute them:
#   variance = AVG(x*x) - AVG(x)*AVG(x)      (population)
#   stddev   = sqrt(variance)
print(q("""
    SELECT COUNT(*)        AS n,
           ROUND(AVG(salary), 1)   AS mean,
           ROUND(MIN(salary))      AS min,
           ROUND(MAX(salary))      AS max,
           ROUND(SUM(salary))      AS sum,
           ROUND(sqrt(AVG(salary*salary) - AVG(salary)*AVG(salary)), 1) AS stddev,
           (SELECT salary FROM employees ORDER BY salary LIMIT 1 OFFSET
               (SELECT COUNT(*)/2 FROM employees)) AS median_approx
    FROM employees
"""))
# NOTE: that sqrt formula is the POPULATION stddev (÷N). Sample stddev (÷N-1)
# needs an extra factor: sqrt(n/(n-1) * var). MySQL: STDDEV_POP/STDDEV_SAMP;
# PostgreSQL: stddev_pop/stddev_samp.

# %%
# correlation (Pearson) in pure SQL — same identity as in pandas .corr()
print(q("""
    WITH m AS (
        SELECT AVG(age)     AS mx,   AVG(salary) AS my,
               AVG(age*age) AS mxx,  AVG(salary*salary) AS myy,
               AVG(age*salary) AS mxy
        FROM employees
    )
    SELECT ROUND((mxy - mx*my) / (sqrt(mxx - mx*mx) * sqrt(myy - my*my)), 4) AS pearson_r_age_salary
    FROM m
"""))
# r ≈ 0 for age vs salary in this tiny sample — remember: correlation is NOT causation.

# %%
# top-N with a tie: 3 highest salaries (RANK includes the tie -> 4 rows)
print("top 3 salaries with ties kept (RANK):\n", q("""
    SELECT emp_name, salary FROM (
        SELECT emp_name, salary, RANK() OVER (ORDER BY salary DESC) AS rnk
        FROM employees
    ) WHERE rnk <= 3
"""))
print("\ntop 3 exactly (ROW_NUMBER):\n", q("""
    SELECT emp_name, salary FROM (
        SELECT emp_name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC, emp_id) AS rn
        FROM employees
    ) WHERE rn <= 3
"""))

# %%
# time trends: monthly revenue (strftime groups the months)
print(q("""
    SELECT strftime('%Y-%m', order_date) AS month,
           COUNT(*)        AS orders,
           SUM(amount)     AS revenue,
           ROUND(AVG(amount)) AS avg_order_value
    FROM orders
    GROUP BY month
    ORDER BY month
"""))

# cohort-ish grouping: customers by signup month vs how much they ordered
print("\ncohort analysis (signup month → orders & spend):\n", q("""
    SELECT c.cohort, COUNT(DISTINCT c.customer_id) AS customers,
           COUNT(o.order_id) AS orders,
           COALESCE(SUM(o.amount), 0) AS revenue
    FROM (SELECT *, strftime('%Y-%m', signup_date) AS cohort FROM customers) c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    GROUP BY c.cohort
    ORDER BY c.cohort
"""))

# %% [markdown]
# **Section notes — EDA in SQL:**
#
# | Question | Query pattern |
# |---|---|
# | value counts | `SELECT col, COUNT(*) FROM t GROUP BY col ORDER BY 2 DESC` |
# | summary stats | `MIN/MAX/AVG/SUM` + hand-rolled `sqrt(AVG(x*x)-AVG(x)*AVG(x))` |
# | correlation | Pearson identity in a CTE (above) |
# | top-N | `ROW_NUMBER()/RANK() OVER (...)` + outer filter |
# | trends | `GROUP BY strftime('%Y-%m', date_col)` |
# | cohorts | derived table of cohort label + `LEFT JOIN` + `GROUP BY cohort` |
# | share of total | `100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()` (window!) |

# %% [markdown]
# ---
# ## 16. SQL interview questions — 10 classics, all runnable
#
# ▶ SQL Interview Questions Part 1: https://www.youtube.com/watch?v=QlenVBvzrW4
# ▶ SQL Case Study 1 | Zomato Dataset (source of several of these): https://www.youtube.com/watch?v=jYwcB0KEFfY

# %%
# Q1 — 2nd highest salary (two idioms)
print("subquery version:\n", q("""
    SELECT MAX(salary) AS second_highest
    FROM employees
    WHERE salary < (SELECT MAX(salary) FROM employees)
"""))
print("\nwindow version:\n", q("""
    SELECT salary AS second_highest FROM (
        SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
        FROM employees
    ) WHERE rnk = 2
"""))

# %%
# Q2 — highest salary per department
print(q("""
    SELECT d.dept_name, MAX(e.salary) AS max_salary
    FROM employees e
    JOIN departments d ON d.dept_id = e.dept_id
    GROUP BY d.dept_name
    ORDER BY max_salary DESC
"""))
# ...and WHO earns it (the classic follow-up):
print("\nemployees who earn their dept max:\n", q("""
    SELECT e.emp_name, d.dept_name, e.salary
    FROM employees e
    JOIN departments d ON d.dept_id = e.dept_id
    WHERE e.salary = (SELECT MAX(salary) FROM employees x WHERE x.dept_id = e.dept_id)
    ORDER BY e.salary DESC
"""))

# %%
# Q3 — employees earning above the company average
print(q("""
    SELECT emp_name, salary
    FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees)
    ORDER BY salary DESC
"""))

# %%
# Q4 — delete duplicate rows, keep one copy
con.executescript("""
CREATE TABLE contacts(cid INTEGER PRIMARY KEY, name TEXT, email TEXT);
INSERT INTO contacts VALUES
 (1,'Ravi','r@x.com'), (2,'Ravi','r@x.com'), (3,'Sita','s@x.com'),
 (4,'Amit','a@x.com'), (5,'Sita','s@x.com');
""")
print("before:", len(q("SELECT * FROM contacts")), "rows")
con.execute("""
    DELETE FROM contacts
    WHERE cid IN (
        SELECT cid FROM (
            SELECT cid, ROW_NUMBER() OVER (
                       PARTITION BY name, email ORDER BY cid) AS rn
            FROM contacts
        ) WHERE rn > 1
    )
""")
print("after :", q("SELECT * FROM contacts"))

# %%
# Q5 — running total of order amounts
print(q("""
    SELECT order_id, order_date, amount,
           SUM(amount) OVER (ORDER BY order_date, order_id
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
    FROM orders
    ORDER BY order_date, order_id
"""))

# %%
# Q6 — month-over-month revenue growth (%)
print(q("""
    WITH monthly AS (
        SELECT strftime('%Y-%m', order_date) AS ym, SUM(amount) AS revenue
        FROM orders GROUP BY ym
    )
    SELECT ym, revenue,
           LAG(revenue) OVER (ORDER BY ym) AS prev_revenue,
           ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY ym))
                 / LAG(revenue) OVER (ORDER BY ym), 1) AS pct_growth
    FROM monthly
    ORDER BY ym
"""))

# %%
# Q7 — customers who never placed an order (3 ways to the same answer)
print("LEFT JOIN + IS NULL:\n", q("""
    SELECT c.cust_name
    FROM customers c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.order_id IS NULL
"""))
print("\nNOT EXISTS:\n", q("""
    SELECT c.cust_name FROM customers c
    WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)
"""))

# %%
# Q8 — count of orders per day (and days with more than one order)
print(q("""
    SELECT order_date, COUNT(*) AS orders, SUM(amount) AS revenue
    FROM orders
    GROUP BY order_date
    ORDER BY orders DESC, order_date
"""))

# %%
# Q9 — consecutive-day activity streaks (LAG + date difference)
print("orders placed on the previous day too:\n", q("""
    SELECT customer_id, order_date, prev_date,
           CAST(julianday(order_date) - julianday(prev_date) AS INT) AS gap_days
    FROM (
        SELECT customer_id, order_date,
               LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_date
        FROM orders
        WHERE status = 'completed'
    )
    WHERE prev_date IS NOT NULL
      AND julianday(order_date) - julianday(prev_date) = 1
    ORDER BY customer_id, order_date
"""))

# %%
# Q10 — Nth highest salary *per department* (N = 2) with ROW_NUMBER
N = 2
print(f"2nd highest salary in each department:\n", q("""
    SELECT dept_id, emp_name, salary FROM (
        SELECT emp_name, dept_id, salary,
               ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC, emp_id) AS rn
        FROM employees
    ) WHERE rn = 2
    ORDER BY dept_id
"""))

# %% [markdown]
# ### Interview quick-recap
#
# | Question | One-line answer |
# |---|---|
# | 2nd highest salary | `MAX(salary) WHERE salary < MAX(...)` or `DENSE_RANK() = 2` |
# | dept max + who | `GROUP BY` for the value, correlated `WHERE salary = (SELECT MAX ...)` for the name |
# | above average | `WHERE salary > (SELECT AVG(salary) FROM t)` |
# | delete dupes | `ROW_NUMBER() OVER (PARTITION BY keys)` → `DELETE rn > 1` |
# | running total | `SUM() OVER (ORDER BY d ROWS UNBOUNDED PRECEDING→CURRENT)` |
# | MoM growth | aggregate by month + `LAG()` |
# | customers w/o orders | `LEFT JOIN ... WHERE o.id IS NULL` (or `NOT EXISTS`) |
# | count per day | `GROUP BY date_col` |
# | streaks | `LAG(date)` + `julianday(a)-julianday(b) = 1` |
# | Nth per group | `ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC) = N` |
#
# > **Gotcha:** `WHERE` filters rows, `HAVING` filters groups, `ORDER BY` runs
# > last — an alias from `SELECT` can only be reused in `ORDER BY` (and in some
# > engines, `HAVING` via the underlying expression).

# %% [markdown]
# ---
# ## SQL cheat sheet
#
# | Task | Query template |
# |---|---|
# | peek at a table | `SELECT * FROM t LIMIT 10;` |
# | column list | `PRAGMA table_info(t);` |
# | filter | `SELECT ... FROM t WHERE cond AND cond;` |
# | pattern | `WHERE name LIKE 'A%';` |
# | null-safe | `WHERE col IS NOT NULL;` / `COALESCE(col, default)` |
# | sort + page | `ORDER BY col DESC LIMIT 10 OFFSET 20;` |
# | categorise | `CASE WHEN x > 5 THEN 'a' ELSE 'b' END` |
# | group counts | `SELECT g, COUNT(*), AVG(x) FROM t GROUP BY g HAVING COUNT(*) > 1;` |
# | dedupe keys | `SELECT DISTINCT a, b FROM t;` |
# | join | `FROM a JOIN b ON a.id = b.a_id` / `LEFT JOIN` to keep `a` |
# | full outer (SQLite) | `LEFT JOIN ... UNION ALL ... WHERE right.id IS NULL` |
# | subquery in FROM | `FROM (SELECT ...) AS d;` |
# | reusable logic | `WITH cte AS (SELECT ...) SELECT ... FROM cte;` |
# | hierarchy | `WITH RECURSIVE r AS (anchor UNION ALL step) SELECT ...;` |
# | rank in group | `ROW_NUMBER()/RANK() OVER (PARTITION BY g ORDER BY v DESC)` |
# | running total | `SUM(x) OVER (ORDER BY d ROWS UNBOUNDED PRECEDING AND CURRENT ROW)` |
# | moving average | `AVG(x) OVER (ORDER BY d ROWS 2 PRECEDING AND CURRENT ROW)` |
# | prev/next | `LAG(x, 1, default)` / `LEAD(x)` |
# | top-N per group | outer `WHERE rn <= N` over `ROW_NUMBER() OVER (PARTITION BY g ...)` |
# | MoM change | `GROUP BY strftime('%Y-%m', d)` + `LAG()` |
# | set combine | `UNION` (distinct) / `UNION ALL` (keep dupes) / `INTERSECT` / `EXCEPT` |
# | save a query | `CREATE VIEW v AS SELECT ...;` |
# | speed up lookup | `CREATE INDEX i ON t(col);` then `EXPLAIN QUERY PLAN` |
# | atomic change | `BEGIN; ...; COMMIT;` (`ROLLBACK` / `SAVEPOINT` to undo) |
# | dedupe rows | `DELETE WHERE cid IN (SELECT ... WHERE rn > 1)` |
# | stddev (SQLite) | `sqrt(AVG(x*x) - AVG(x)*AVG(x))` |
# | pearson r | `(AVG(xy)-AVG(x)*AVG(y)) / (sd(x)*sd(y))` |
#
# > **Interview short-list:** `WHERE` vs `HAVING` · execution order ·
# > `COUNT(*)` vs `COUNT(col)` · NULL three-valued logic (`= NULL` never true) ·
# > inner vs left vs full outer · correlated subquery runs per row ·
# > `ROW_NUMBER` vs `RANK` vs `DENSE_RANK` · explicit frame for `LAST_VALUE` ·
# > `UNION` vs `UNION ALL` · 1NF/2NF/3NF · ACID & isolation anomalies.

# %% [markdown]
# ---
# **Next notebook:** `08_statistics_probability.ipynb` — the maths behind the
# numbers you just queried (distributions, tests, sampling).
#
# **Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z
