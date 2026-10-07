# DSMP 2022-23 — Data Science Notes (Jupyter Notebooks)

Structured, code-first notes built from the **CampusX Data Science Mentorship Program (DSMP) 2022-23** playlist — every concept with syntax, parameters, runnable examples, section notes, gotchas and a link back to the exact session video.

**Playlist:** https://www.youtube.com/playlist?list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z

## Notebook index

| Notebook | Topic | Sessions | Cells |
|---|---|---|---|
| `01_python_fundamentals.ipynb` | 01 · Python Fundamentals | 8 | 146 |
| `02_oop_advanced_python.ipynb` | 02 · OOP & Advanced Python | 11 | 111 |
| `03_numpy.ipynb` | 03 · NumPy | 4 | 98 |
| `04_pandas.ipynb` | 04 · Pandas | 11 | 212 |
| `05_visualization.ipynb` | 05 · Data Visualization | 6 | 129 |
| `06_data_analysis_process.ipynb` | 06 · Data Analysis Process | 6 | 107 |
| `07_sql.ipynb` | 07 · SQL for Data Science | 18 | 118 |
| `08_statistics_probability.ipynb` | 08 · Statistics & Probability | 13 | 173 |
| `09_linear_algebra.ipynb` | 09 · Linear Algebra | 6 | 101 |
| `10_ml_fundamentals_regression.ipynb` | 10 · ML Fundamentals & Regression | 27 | 91 |
| `11_ml_classification.ipynb` | 11 · Classification Algorithms | 14 | 96 |
| `12_ml_dimensionality_model_selection.ipynb` | 12 · Dimensionality Reduction & Model Selection | 8 | 114 |

**Total: 12 notebooks, 1496 cells** (every code cell executes top-to-bottom with 0 errors).

## How to use

1. Open any `.ipynb` in Jupyter / VS Code and run all — the outputs are already saved, so you can also just read them.
2. Each notebook starts with a **▶ Sessions covered** table linking to its source videos, and every section repeats its own video link.
3. Read in order 01 → 12 (playlist order), or jump to the topic you need.

## Repo layout

```
dsmp-notes/
├── NN_*.ipynb      # the 12 finished notebooks (generated + executed)
├── src/            # notebook sources (percent-format .py — edit these)
└── tools/
    ├── percent2ipynb.py  # src/*.py -> .ipynb converter
    ├── make_readme.py    # regenerates this file
    ├── playlist.json     # full ordered playlist dump (200 videos)
    └── FORMAT.md         # authoring style guide
```

Rebuild after editing a source:

```bash
python3 tools/percent2ipynb.py                 # convert all src/*.py
jupyter-nbconvert --to notebook --execute --inplace NN_topic.ipynb
python3 tools/make_readme.py                   # refresh this index
```

## Sessions covered (by notebook)


### 01_python_fundamentals.ipynb

**01 · Python Fundamentals**

Output, data types, variables, operators, control flow, loops, strings, lists, tuples, sets, dictionaries, functions, time complexity.

| # | Session | Video |
|---|---------|-------|
| 1 | Session 1 - Python Fundamentals \| CampusX Data Science Mentorship Program \| 7th Nov 2022 2 hours, 10 minutes | https://www.youtube.com/watch?v=1z5-O7-5AXk |
| 2 | Session 2 - Operators + If-Else + Loops  \| CampusX Data Science Mentorship Program \| 8th Nov 2022 2 hours, 11 minutes | https://www.youtube.com/watch?v=JCkIrdrZEE8 |
| 3 | Session 3 - Python Strings  \| CampusX Data Science Mentorship Program \| 9th Nov 2022 2 hours, 22 minutes | https://www.youtube.com/watch?v=6HAu0Y9BjA4 |
| 4 | Programming Problems on Strings \| Session 3 Supplementary Session 29 minutes | https://www.youtube.com/watch?v=7ltjqU5iytY |
| 7 | Session on Time Complexity \| Friday Paid Session 1 hour, 42 minutes | https://www.youtube.com/watch?v=J1vdIcg5IZs |
| 9 | Session 4 - Lists in Python \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 7 minutes | https://www.youtube.com/watch?v=WmbU3WBaoR0 |
| 10 | Session 5 - Tuples + Sets + Dictionary \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 16 minutes | https://www.youtube.com/watch?v=jcQjp11mn1A |
| 11 | Session 6 - Functions in Python \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=OOInK25PoFo |

### 02_oop_advanced_python.ipynb

**02 · OOP & Advanced Python**

Classes/objects, encapsulation, inheritance, polymorphism, abstraction, file handling & serialization, exceptions, decorators, recursion, iterators, generators.

| # | Session | Video |
|---|---------|-------|
| 17 | OOP Part 1 \| Class \u0026 Object \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 13 minutes | https://www.youtube.com/watch?v=Trvqq5w7-0Q |
| 19 | OOP Part 2 \| Encapsulation \u0026 Static Keyword \| Static \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=P4xizq3CiJg |
| 21 | OOP Part 3 \| Inheritance \u0026 Polymorphism \| Static \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 14 minutes | https://www.youtube.com/watch?v=bEWwM4hXZg8 |
| 22 | OOP Project \| Friday Paid Session \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=awjzebtHQ_o |
| 26 | What is Abstraction \| OOP Concept \| Python 14 minutes, 53 seconds | https://www.youtube.com/watch?v=O-NrQvtorp0 |
| 27 | Session 10 - File Handling + Serialization \u0026 Deserialization \| DSMP 2022 - 23 2 hours, 5 minutes | https://www.youtube.com/watch?v=o-TAYRMQzIQ |
| 29 | Session 11 - Exception Handling \u0026 Modules and Packages \| DSMP 2022 - 23 1 hour, 29 minutes | https://www.youtube.com/watch?v=rvKR6tciJ2Q |
| 31 | Session 12 - Decorators \u0026 Namespaces \| DSMP 2022 - 23 1 hour, 59 minutes | https://www.youtube.com/watch?v=aMARZGTbULc |
| 33 | Recursion using Python \| Recursion with examples \| Memoization 45 minutes | https://www.youtube.com/watch?v=cNvZK0Wyoik |
| 34 | What is Iterators in Python? 38 minutes | https://www.youtube.com/watch?v=pH7YVRhnpUI |
| 35 | Generators in Python \| Advanced Python Programming 24 minutes | https://www.youtube.com/watch?v=ZfJoU67tG1A |

### 03_numpy.ipynb

**03 · NumPy**

Array creation, attributes, indexing/slicing, broadcasting, ufuncs, aggregations & axis, sorting/searching, stacking, random, linear algebra, copy vs view, tricks.

| # | Session | Video |
|---|---------|-------|
| 41 | Session 13 - Numpy Fundamentals \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 1 minute | https://www.youtube.com/watch?v=XF6DCrNTzug |
| 43 | Session 14 - Advanced Numpy \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 1 hour, 49 minutes | https://www.youtube.com/watch?v=40xGMygHMDU |
| 45 | Session 15 - Numpy Tricks \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 1 hour, 30 minutes | https://www.youtube.com/watch?v=jFBpnd0G09o |
| 55 | Week 5 - Numpy Interview Questions \| Paid Session \| DSMP 2022-23 2 hours, 34 minutes | https://www.youtube.com/watch?v=wjv01Opo6Bk |

### 04_pandas.ipynb

**04 · Pandas**

Series & DataFrame reference, indexing (loc/iloc), all core methods, groupby, merge/join/concat, MultiIndex, .str/.dt accessors, pivot_table, case-study pipeline.

| # | Session | Video |
|---|---------|-------|
| 50 | Session 16 - Pandas Series\| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 7 minutes | https://www.youtube.com/watch?v=zCDVUyq8lkw |
| 51 | Session 17 - Pandas DataFrame \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 8 minutes | https://www.youtube.com/watch?v=OG31yhRQxPI |
| 52 | Important Series Methods \| Supplementary Session \| DSMP 2022-23 25 minutes | https://www.youtube.com/watch?v=80QpbZA38HA |
| 53 | Session 18 - Important DataFrame Methods \| Data Science Mentorship Program (DSMP) 2022-23 2 hours, 12 minutes | https://www.youtube.com/watch?v=zTa4MIrGTIE |
| 56 | Session 19 - GroupBy Object in Pandas \| Data Science Mentorship Program (DSMP) 2022-23 2 hours, 9 minutes | https://www.youtube.com/watch?v=LPBjF4_gZnI |
| 60 | Session 20 - Merging, Joining \u0026 Concatenating \| DSMP 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=Ssy1EfK5S-o |
| 61 | Session 21 - MultiIndex Series and DataFrames \| DSMP 2022-23 1 hour, 57 minutes | https://www.youtube.com/watch?v=QzoRUwz8DoM |
| 70 | Session 22 - Vectorized String Operations \| DateTime in Pandas \| Pivot Table \| DSMP 2022-23 2 hours, 6 minutes | https://www.youtube.com/watch?v=UE6DmRQJ2w8 |
| 63 | Pandas Case Study 1 \| Indian Startup Funding 2 hours, 23 minutes | https://www.youtube.com/watch?v=VGPnRAASARY |
| 71 | Pandas Case Study   Time Series Analysis 2 hours, 27 minutes | https://www.youtube.com/watch?v=LknR_AVYboc |
| 72 | Pandas Case Study 2   Working with textual data 2 hours, 40 minutes | https://www.youtube.com/watch?v=cO7sebFXWT8 |

### 05_visualization.ipynb

**05 · Data Visualization**

Matplotlib basics + advanced (gridspec, formatters, annotations), seaborn statistical plots, Plotly Express & graph_objects, chart-choice guide.

| # | Session | Video |
|---|---------|-------|
| 73 | Session 23 - Plotting using Matplotlib \| DSMP 2022-23 2 hours, 6 minutes | https://www.youtube.com/watch?v=XaKn_cKFlSY |
| 75 | Session 24 - Advanced Matplotlib \| DSMP 2022-23 2 hours, 13 minutes | https://www.youtube.com/watch?v=7YDc5xU9CQQ |
| 76 | Session on Plotly Express 2 hours, 14 minutes | https://www.youtube.com/watch?v=VPqw8soG4ZY |
| 77 | Session on Plotly Part 2 2 hours, 9 minutes | https://www.youtube.com/watch?v=_ZsVLh6v29A |
| 78 | Session 25 - Plotting using Seaborn \| DSMP 2022-23 1 hour, 50 minutes | https://www.youtube.com/watch?v=DWVLRhnuGqI |
| 80 | Session 26 - Plotting using Seaborn Part 2 \| DSMP 2022-23 1 hour, 41 minutes | https://www.youtube.com/watch?v=kLWeKeqc9Ms |

### 06_data_analysis_process.ipynb

**06 · Data Analysis Process**

Data gathering (files/APIs/scraping with offline fallbacks), assessing, cleaning (missing/outliers/types/strings), feature basics, EDA workflow and an end-to-end case study.

| # | Session | Video |
|---|---------|-------|
| 82 | Session 27 - Data Gathering \| Data Analysis Process \| DSMP 2023 2 hours, 50 minutes | https://www.youtube.com/watch?v=dA6ZksRR6aw |
| 84 | Session 28 - Data Assessing and Cleaning \| DSMP 2022-23 2 hours, 26 minutes | https://www.youtube.com/watch?v=-HnN8GBINnc |
| 88 | Session on Data Cleaning Case Study - Smartphone dataset 2 hours, 10 minutes | https://www.youtube.com/watch?v=0j_7j3PNSEw |
| 91 | Session 29 - Exploratory Data Analysis \| Data Analysis Process \| DSMP 2022-23 2 hours, 30 minutes | https://www.youtube.com/watch?v=PPEHpg2RixQ |
| 95 | Session 2 on Data Cleaning - Smartphones Dataset \| Paid Session \| DSMP 2023 2 hours, 28 minutes | https://www.youtube.com/watch?v=o-8KvsSdDiE |
| 97 | Session on EDA Case Study  - Smartphones Dataset \| Paid Session \| DSMP 2023 2 hours, 29 minutes | https://www.youtube.com/watch?v=PxRth8xosqY |

### 07_sql.ipynb

**07 · SQL for Data Science**

Runnable via sqlite3: DDL/DML, SELECT, grouping, joins, subqueries/CTEs, window functions, views/indexes, transactions, normalization, cleaning & EDA in SQL, interview queries.

| # | Session | Video |
|---|---------|-------|
| 96 | Session 30 - Database Fundamentals \| DSMP 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=CTLwYjh92xw |
| 99 | Session 31 - SQL DDL Commands \| DSMP 2023 2 hours, 4 minutes | https://www.youtube.com/watch?v=ny1mh6VUpnQ |
| 101 | Session 32 - SQL DML Commands \| DSMP 2023 2 hours, 4 minutes | https://www.youtube.com/watch?v=kzMZDlvHFak |
| 102 | Session 33 - SQL Grouping + Sorting \| DSMP 2023 2 hours, 5 minutes | https://www.youtube.com/watch?v=nsKcmOly0UY |
| 104 | Session 34 - SQL Joins\| DSMP 2023 2 hours, 10 minutes | https://www.youtube.com/watch?v=s6xAVqYdmJ4 |
| 107 | Session 35 - Subqueries in SQL \| DSMP 2023 2 hours, 18 minutes | https://www.youtube.com/watch?v=YYq47MN3TZI |
| 108 | SQL Case Study 1 \| Zomato Dataset 2 hours, 10 minutes | https://www.youtube.com/watch?v=jYwcB0KEFfY |
| 111 | SQL Interview Questions Part 1 2 hours, 4 minutes | https://www.youtube.com/watch?v=QlenVBvzrW4 |
| 112 | Session 36 - Window Functions in SQL \| DSMP 2023 1 hour, 40 minutes | https://www.youtube.com/watch?v=3LJ-yWBQBGs |
| 115 | Session 37 - Window Functions Part 2 \| SQL Tutorials 47 minutes | https://www.youtube.com/watch?v=YgizUxmBFK8 |
| 116 | Session 37 (Cont..) - Window Functions Part 3 \| SQL Tutorial 53 minutes | https://www.youtube.com/watch?v=e0HQVOdfqlQ |
| 117 | Session on Data Cleaning using SQL \|  Laptops Dataset 2 hours, 18 minutes | https://www.youtube.com/watch?v=NsOXJdRIamc |
| 118 | Session on EDA using SQL \| Laptops Dataset 2 hours, 9 minutes | https://www.youtube.com/watch?v=dv8TkUTlQvI |
| 124 | Session on Datetime in SQL \| Paid Session \| DSMP 2023 2 hours, 14 minutes | https://www.youtube.com/watch?v=OY9YiALhSR4 |
| 127 | SQL Datetime Case Study on Flights Dataset \| Paid Session \| DSMP 2023 1 hour, 52 minutes | https://www.youtube.com/watch?v=zY1UCTko64E |
| 128 | Session on Database Design \| SQL Data Types \| Database Normalization 2 hours, 25 minutes | https://www.youtube.com/watch?v=8SOs0URTvr0 |
| 132 | Session on Views and User Defined Functions in SQL \| Paid Session \| DSMP 2023 1 hour, 54 minutes | https://www.youtube.com/watch?v=dMoKTx1bVnY |
| 133 | Session on Transactions & Stored Procedures 2 hours, 21 minutes | https://www.youtube.com/watch?v=-O_9iR7XsJw |

### 08_statistics_probability.ipynb

**08 · Statistics & Probability**

Descriptive statistics, correlation, probability rules & Bayes, PMF/PDF/CDF, discrete & continuous distributions, CLT, confidence intervals, hypothesis tests (z/t/χ²/ANOVA), test-selection cheat sheet.

| # | Session | Video |
|---|---------|-------|
| 123 | Session 38 - Descriptive Statistics Part 1 \| DSMP 2023 1 hour, 52 minutes | https://www.youtube.com/watch?v=Uv3Blie7F3g |
| 125 | Session 39 - Descriptive Statistics Part 2 \| DSMP 2023 1 hour, 55 minutes | https://www.youtube.com/watch?v=1ndVC500-EU |
| 126 | Session 40 - Probability Distribution Functions - PDF, PMF & CDF \| DSMP 2023 2 hours, 9 minutes | https://www.youtube.com/watch?v=C_QAURbgBqY |
| 129 | Session 41 - Normal Distribution \| DSMP 2023 2 hours, 1 minute | https://www.youtube.com/watch?v=ADqYqSdtyW8 |
| 131 | Session 42 - Non-Gaussian Probability Distributions \| DSMP 2023 2 hours, 16 minutes | https://www.youtube.com/watch?v=U6QCc_3zgUk |
| 134 | Session 43 - Central Limit Theorem \| DSMP 2023 1 hour, 51 minutes | https://www.youtube.com/watch?v=-WmJDYBor7c |
| 135 | Session 44 - Confidence Intervals \| DSMP 2023 2 hours, 10 minutes | https://www.youtube.com/watch?v=X52HK2qkiIE |
| 137 | Session 45 - Hypothesis Testing Part 1 \| DSMP 2023 1 hour, 59 minutes | https://www.youtube.com/watch?v=S94mx6OL7kM |
| 138 | Session 46 - Hypothesis Testing Part 2 \| p-values \| t-tests \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=xHTMjxx14sU |
| 139 | Session on Chi Square Tests 2 hours, 15 minutes | https://www.youtube.com/watch?v=_8DTqznosvY |
| 140 | Session on ANOVA  \| Paid Zoom Session 1 hour, 58 minutes | https://www.youtube.com/watch?v=G9ppvMWyd54 |
| 189 | Master Probability in Data Science: The Ultimate Crash Course! \| Part 1 \| CampusX 1 hour, 34 minutes | https://www.youtube.com/watch?v=DUT4WEUngt0 |
| 190 | Mastering Probability for ML: Joint \| Marginal \| Conditional Probability and Bayes' Theorem 1 hour, 20 minutes | https://www.youtube.com/watch?v=ndHDsvqmbuI |

### 09_linear_algebra.ipynb

**09 · Linear Algebra**

Vectors, dot products & norms, matrices, determinants/inverse, systems of equations, eigen-decomposition, SVD, projections — with the PCA/ML links.

| # | Session | Video |
|---|---------|-------|
| 141 | Supercharge Your ML Journey: Mastering Vectors in Linear Algebra - Part 1 1 hour, 31 minutes | https://www.youtube.com/watch?v=mQewAJb8oJ8 |
| 142 | Linear Algebra Part 2 \| Matrices 2 hours, 26 minutes | https://www.youtube.com/watch?v=Y8UjSNSMeZ4 |
| 143 | Linear Algebra Part 3  \| Matrices  Intuition 2 hours, 10 minutes | https://www.youtube.com/watch?v=7IV56bGTJTQ |
| 182 | Session on Eigen Vectors and Eigen Values \| Week 29 \| DSMP 1 hour, 40 minutes | https://www.youtube.com/watch?v=5HDkRG5gQHM |
| 183 | Session on Eigen Decomposition + PCA Variants \| Week 29 \| DSMP 2 hours, 7 minutes | https://www.youtube.com/watch?v=D5IlO4lzaZ0 |
| 184 | Session on Singular Value Decomposition 2 hours, 13 minutes | https://www.youtube.com/watch?v=NPkMoUNkEtQ |

### 10_ml_fundamentals_regression.ipynb

**10 · ML Fundamentals & Regression**

ML workflow & splits, simple/multiple linear regression, gradient descent (batch/SGD/mini-batch from scratch), statsmodels OLS, LINE assumptions, polynomial regression, multicollinearity/VIF, feature selection, bias-variance, Ridge/Lasso/ElasticNet.

| # | Session | Video |
|---|---------|-------|
| 144 | Introduction to Machine Learning 3 hours, 26 minutes | https://www.youtube.com/watch?v=1ctqJCHMAmc |
| 145 | Simple Linear Regression \| Lecture 49 \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=aEPoLeS6UMM |
| 146 | Session 50 - Multiple Linear Regression \| DSMP 2023 1 hour, 49 minutes | https://www.youtube.com/watch?v=e_9vJXokd-Y |
| 148 | Session on Optimization - The Big Picture 2 hours, 9 minutes | https://www.youtube.com/watch?v=T2f8249569Q |
| 149 | Session on Differential Calculus 2 hours, 25 minutes | https://www.youtube.com/watch?v=-siBIGUREjw |
| 150 | Gradient Descent From Scratch \| End to End Gradient Descent \| Gradient Descent Animation 1 hour, 57 minutes | https://www.youtube.com/watch?v=ORyfPJypKuU |
| 151 | Batch Gradient Descent with Code Demo \| Simple Explanation in Hindi 1 hour, 4 minutes | https://www.youtube.com/watch?v=Jyo53pAyVAM |
| 152 | Stochastic Gradient Descent 49 minutes | https://www.youtube.com/watch?v=V7KBAa_gh4c |
| 153 | Mini-Batch Gradient Descent 22 minutes | https://www.youtube.com/watch?v=_scscQ4HVTY |
| 155 | Session 1 on Regression Analysis 2 hours, 16 minutes | https://www.youtube.com/watch?v=qB5k6A60Unw |
| 157 | Session 2 on Regression Analysis 1 hour, 54 minutes | https://www.youtube.com/watch?v=6QQdhk2_H9Q |
| 158 | Polynomial Regression \| Machine Learning 26 minutes | https://www.youtube.com/watch?v=BNWLf3cKdbQ |
| 159 | Session on Assumptions of Linear Regression 2 hours, 25 minutes | https://www.youtube.com/watch?v=EWV9laVakjc |
| 160 | Session 53 - Session on Multicollinearity \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=XGJLWJ0a6aU |
| 161 | Session 54 - Feature Selection Part 1 \| Filter Methods \| Variance Threshold \| Chi-Square \| DSMP 2023 2 hours, 6 minutes | https://www.youtube.com/watch?v=hRTe7tSkR2k |
| 162 | Session 55 - Feature Selection Part 2 \| Wrapper Methods  \| DSMP 2023 1 hour, 53 minutes | https://www.youtube.com/watch?v=6-1WiITAW10 |
| 163 | Session 3 on Feature Selection \| Embedded Methods 1 hour, 59 minutes | https://www.youtube.com/watch?v=zLB-csjZ9V4 |
| 164 | Regularization Part 1 \| Bias Variance Trade-off 2 hours, 2 minutes | https://www.youtube.com/watch?v=qfWUIUvKi98 |
| 165 | Regularization Part 2 \| What is Regularization? 1 hour, 56 minutes | https://www.youtube.com/watch?v=cVsN5VAioz0 |
| 167 | Doubt Clearance Session on Regularization 1 hour, 9 minutes | https://www.youtube.com/watch?v=7_DDQkLzmRM |
| 169 | Ridge Regression Part 1 \| Geometric Intuition and Code \| Regularized Linear Models 19 minutes | https://www.youtube.com/watch?v=aEow1QoTLo0 |
| 170 | Ridge Regression Part 2 \| Mathematical Formulation & Code from scratch \| Regularized Linear Models 43 minutes | https://www.youtube.com/watch?v=oDlZBQjk_3A |
| 171 | Ridge Regression Part 3 \| Gradient Descent \| Regularized Linear Models 18 minutes | https://www.youtube.com/watch?v=Fci_wwMp8G8 |
| 172 | 5 Key Points - Ridge Regression \| Part 4 \| Regularized Linear Models 30 minutes | https://www.youtube.com/watch?v=8osKeShYVRQ |
| 173 | Lasso Regression \| Intuition and Code Sample \| Regularized Linear Models 28 minutes | https://www.youtube.com/watch?v=HLF4bFbBgwk |
| 174 | Why Lasso Regression creates sparsity? 24 minutes | https://www.youtube.com/watch?v=FN4aZPIAfI4 |
| 175 | ElasticNet Regression \| Intuition and Code Example \| Regularized Linear Models 11 minutes, 41 seconds | https://www.youtube.com/watch?v=2g2DBkFhTTY |

### 11_ml_classification.ipynb

**11 · Classification Algorithms**

Confusion matrix & metrics, ROC/PR curves, KNN (from scratch), Naive Bayes variants, MLE, logistic & softmax regression, SVM hard/soft margin & kernels, model comparison.

| # | Session | Video |
|---|---------|-------|
| 166 | What is K Nearest Neighbors? \| KNN Explained in Hindi \| Simple Overview in 1 Video \| CampusX 52 minutes | https://www.youtube.com/watch?v=abnL_GUGub4 |
| 168 | Session on Advanced KNN 2 hours, 6 minutes | https://www.youtube.com/watch?v=d7xo69q-1H4 |
| 176 | Accuracy and Confusion Matrix \| Type 1 and Type 2 Errors \| Classification Metrics Part 1 34 minutes | https://www.youtube.com/watch?v=c09drtuCS3c |
| 177 | Precision, Recall and F1 Score \| Classification Metrics Part 2 42 minutes | https://www.youtube.com/watch?v=iK-kdhJ-7yI |
| 185 | ROC Curve in Machine Learning \| ROC-AUC in Machine Learning Simplified \| CampusX 1 hour, 11 minutes | https://www.youtube.com/watch?v=gdW6hj9IXaA |
| 192 | Session 1 on Naive Bayes 2 hours | https://www.youtube.com/watch?v=mEjGU4e-WPs |
| 191 | Session 2 on Naive Bayes 2 hours, 13 minutes | https://www.youtube.com/watch?v=mrBut7CPCxE |
| 193 | Session 3 on Naive Bayes 1 hour, 58 minutes | https://www.youtube.com/watch?v=_x61Lut84zI |
| 194 | Session 1 on Logistic Regression 2 hours, 8 minutes | https://www.youtube.com/watch?v=BuAp03q5GAw |
| 197 | Session 3 on Logistic Regression 2 hours, 6 minutes | https://www.youtube.com/watch?v=u2IfZX5q2jY |
| 196 | Session on Maximum Likelihood Estimation 2 hours, 7 minutes | https://www.youtube.com/watch?v=YSur3ESoV_U |
| 195 | Session on Softmax Regression \| Multiclass Classification \| Week 32 \| DSMP 1 hour, 52 minutes | https://www.youtube.com/watch?v=Dg-1TuPxElw |
| 198 | SVM Part 1 \| Hard Margin SVM 1 hour, 42 minutes | https://www.youtube.com/watch?v=HfrKYccB2DQ |
| 200 | SVM Part 2 - Soft Margin SVM 1 hour, 46 minutes | https://www.youtube.com/watch?v=SlsXm-utgFY |

### 12_ml_dimensionality_model_selection.ipynb

**12 · Dimensionality Reduction & Model Selection**

Curse of dimensionality, PCA (math + from-scratch + sklearn), cross-validation types, data-leakage catalog, hyperparameter tuning (Grid/Random search), feature engineering (imputation, scaling, encoding) and a leakage-safe pipeline.

| # | Session | Video |
|---|---------|-------|
| 178 | Curse of Dimensionality 15 minutes | https://www.youtube.com/watch?v=ToGuhynu-No |
| 179 | Principle Component Analysis  (PCA) \| Part 1 \| Geometric Intuition 33 minutes | https://www.youtube.com/watch?v=iRbsBi5W0-c |
| 180 | Principle Component Analysis (PCA) \| Part 2 \| Problem Formulation and Step by Step Solution 56 minutes | https://www.youtube.com/watch?v=tXXnxjj2wM4 |
| 181 | Principle Component Analysis(PCA) \| Part 3 \| Code Example and Visualization 43 minutes | https://www.youtube.com/watch?v=tofVCUDrg4M |
| 186 | Session on Cross Validation 1 hour, 8 minutes | https://www.youtube.com/watch?v=cY_pBnFM7fQ |
| 187 | Session on Data Leakage 2 hours, 25 minutes | https://www.youtube.com/watch?v=KHvcr0Q7jSk |
| 188 | Session on Hyperparameter Tuning 1 hour, 46 minutes | https://www.youtube.com/watch?v=cDmMVCC4dEE |
| 199 | Feature Engineering \| Extra Session \| Session on Handling Missing Values Part 1 2 hours, 33 minutes | https://www.youtube.com/watch?v=e1-5NOwWCmo |

## Not covered (68 videos)

Paid-member task solutions, doubt-clearance, career/soft-skill sessions, schedule announcements and tool sessions (Flask, Streamlit, Git, Tableau, Selenium, AWS, open source) — full ordered list with IDs: `tools/playlist.json` (200 videos).

**Complete video index (all 200):**

| # | Session | Video |
|---|---------|-------|
| 1 | Session 1 - Python Fundamentals \| CampusX Data Science Mentorship Program \| 7th Nov 2022 2 hours, 10 minutes | https://www.youtube.com/watch?v=1z5-O7-5AXk |
| 2 | Session 2 - Operators + If-Else + Loops  \| CampusX Data Science Mentorship Program \| 8th Nov 2022 2 hours, 11 minutes | https://www.youtube.com/watch?v=JCkIrdrZEE8 |
| 3 | Session 3 - Python Strings  \| CampusX Data Science Mentorship Program \| 9th Nov 2022 2 hours, 22 minutes | https://www.youtube.com/watch?v=6HAu0Y9BjA4 |
| 4 | Programming Problems on Strings \| Session 3 Supplementary Session 29 minutes | https://www.youtube.com/watch?v=7ltjqU5iytY |
| 5 | How to Build a Portfolio Website for Data Science 48 minutes | https://www.youtube.com/watch?v=vg7BRC7wd0c |
| 6 | Session on Week 1 Task Solutions \| Paid Members 2 hours, 14 minutes | https://www.youtube.com/watch?v=kDIsjtH69G8 |
| 7 | Session on Time Complexity \| Friday Paid Session 1 hour, 42 minutes | https://www.youtube.com/watch?v=J1vdIcg5IZs |
| 8 | Week 1 Doubt Clearance + Interview Preparation \| Paid Members 1 hour, 29 minutes | https://www.youtube.com/watch?v=lD4nziSXdMk |
| 9 | Session 4 - Lists in Python \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 7 minutes | https://www.youtube.com/watch?v=WmbU3WBaoR0 |
| 10 | Session 5 - Tuples + Sets + Dictionary \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 16 minutes | https://www.youtube.com/watch?v=jcQjp11mn1A |
| 11 | Session 6 - Functions in Python \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=OOInK25PoFo |
| 12 | Week 1 Task 3 Solutions \| CampusX DSMP 2022-23 \| Paid Session 1 hour, 6 minutes | https://www.youtube.com/watch?v=tLer7kI2vF0 |
| 13 | Career QnA Live Session for Paid Members \| Data Science Mentorship Program(DSMP) 2022-23 1 hour, 59 minutes | https://www.youtube.com/watch?v=_PKpXucutB0 |
| 14 | Array Interview Questions \| Paid Members \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 17 minutes | https://www.youtube.com/watch?v=w73JmBsrGsY |
| 15 | Interview Questions + Doubt Clearance \| Paid Members \| Data Science Mentorship Program(DSMP) 2022-23 1 hour, 27 minutes | https://www.youtube.com/watch?v=QR5ORwasxxg |
| 16 | Week 2 - Task 4 Solutions \| Paid Session \| Data Science Mentorship Program \| DSMP 2022-23 46 minutes | https://www.youtube.com/watch?v=uJfX2xkSSNE |
| 17 | OOP Part 1 \| Class \u0026 Object \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 13 minutes | https://www.youtube.com/watch?v=Trvqq5w7-0Q |
| 18 | Task 5 Solutions \| Paid Members \| Data Science Mentorship Program (DSMP) 2022-23 47 minutes | https://www.youtube.com/watch?v=-42QuawhWqw |
| 19 | OOP Part 2 \| Encapsulation \u0026 Static Keyword \| Static \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=P4xizq3CiJg |
| 20 | Task 6 Solutions \| Paid Recording \| DSMP 2022-23 45 minutes | https://www.youtube.com/watch?v=aQpG2H19BlE |
| 21 | OOP Part 3 \| Inheritance \u0026 Polymorphism \| Static \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 14 minutes | https://www.youtube.com/watch?v=bEWwM4hXZg8 |
| 22 | OOP Project \| Friday Paid Session \| Data Science Mentorship Program(DSMP) 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=awjzebtHQ_o |
| 23 | New Website Launch \| DSMP 2022-23 21 minutes | https://www.youtube.com/watch?v=TkeG4Jnd86U |
| 24 | Week 3 - Interview Questions \| Data Science Mentorship Program(DSMP) 2022-23 1 hour, 51 minutes | https://www.youtube.com/watch?v=HQH5FxvJXzI |
| 25 | Task 7 Solutions \| Paid Recording \| DSMP 2022-23 31 minutes | https://www.youtube.com/watch?v=IHoa-zhGjWg |
| 26 | What is Abstraction \| OOP Concept \| Python 14 minutes, 53 seconds | https://www.youtube.com/watch?v=O-NrQvtorp0 |
| 27 | Session 10 - File Handling + Serialization \u0026 Deserialization \| DSMP 2022 - 23 2 hours, 5 minutes | https://www.youtube.com/watch?v=o-TAYRMQzIQ |
| 28 | Task 8 Solutions \| Paid Recording \| DSMP 2022-23 56 minutes | https://www.youtube.com/watch?v=GqfqvkjxOzI |
| 29 | Session 11 - Exception Handling \u0026 Modules and Packages \| DSMP 2022 - 23 1 hour, 29 minutes | https://www.youtube.com/watch?v=rvKR6tciJ2Q |
| 30 | Task 9 Solutions \| Paid Recording \| DSMP 2022-23 25 minutes | https://www.youtube.com/watch?v=RSFlzUbMuuo |
| 31 | Session 12 - Decorators \u0026 Namespaces \| DSMP 2022 - 23 1 hour, 59 minutes | https://www.youtube.com/watch?v=aMARZGTbULc |
| 32 | Schedule for December + Program Improvements \| CampusX Data Science Mentorship Program (DSMP 2022) 21 minutes | https://www.youtube.com/watch?v=5pLs7Mtcw6Y |
| 33 | Recursion using Python \| Recursion with examples \| Memoization 45 minutes | https://www.youtube.com/watch?v=cNvZK0Wyoik |
| 34 | What is Iterators in Python? 38 minutes | https://www.youtube.com/watch?v=pH7YVRhnpUI |
| 35 | Generators in Python \| Advanced Python Programming 24 minutes | https://www.youtube.com/watch?v=ZfJoU67tG1A |
| 36 | Task 10 Solutions \| Paid Member Recording 45 minutes | https://www.youtube.com/watch?v=H68OwEtzMS0 |
| 37 | GUI development using Python \| Friday Paid Session \| DSMP 2022-23 2 hours, 47 minutes | https://www.youtube.com/watch?v=NlVwSDeUd60 |
| 38 | Week 4 - Interview Questions \| Paid Recordings \| DSMP 2022-23 2 hours, 39 minutes | https://www.youtube.com/watch?v=MTZBjMFQ1Bw |
| 39 | Task 11 Solutions \| Paid Recording \| DSMP 2022-23 26 minutes | https://www.youtube.com/watch?v=pNGw5KGQOKo |
| 40 | Task 12 Solutions \| Paid Recording \| DSMP 2022-23 35 minutes | https://www.youtube.com/watch?v=wNNClClOgc4 |
| 41 | Session 13 - Numpy Fundamentals \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 1 minute | https://www.youtube.com/watch?v=XF6DCrNTzug |
| 42 | Resume Building for Data Scientist \| Career Pe Charcha \| DSMP 2022-23 48 minutes | https://www.youtube.com/watch?v=K_2AdybCEMY |
| 43 | Session 14 - Advanced Numpy \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 1 hour, 49 minutes | https://www.youtube.com/watch?v=40xGMygHMDU |
| 44 | Feedback Form for November \| Data Science Mentorship Program(DSMP) 2022-23 3 minutes, 59 seconds | https://www.youtube.com/watch?v=0W_Xcz_wevM |
| 45 | Session 15 - Numpy Tricks \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 1 hour, 30 minutes | https://www.youtube.com/watch?v=jFBpnd0G09o |
| 46 | Session on Web Development using Flask \| Paid Session \| DSMP 2022-23 2 hours, 33 minutes | https://www.youtube.com/watch?v=KZhytm3DiCE |
| 47 | Task 13 Solutions \| Paid Recording \| DSMP 2022-23 50 minutes | https://www.youtube.com/watch?v=RjoijkOxM34 |
| 48 | Task 14 Solutions \| Paid Recording \| DSMP 2022-23 24 minutes | https://www.youtube.com/watch?v=Nzkvhm_hKdE |
| 49 | Task 15 Solutions \| Paid Recording \| DSMP 2022-23 21 minutes | https://www.youtube.com/watch?v=1sm3JYfBbtc |
| 50 | Session 16 - Pandas Series\| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 7 minutes | https://www.youtube.com/watch?v=zCDVUyq8lkw |
| 51 | Session 17 - Pandas DataFrame \| Data Science Mentorship Program (DSMP) 2022-23 \| Free Session 2 hours, 8 minutes | https://www.youtube.com/watch?v=OG31yhRQxPI |
| 52 | Important Series Methods \| Supplementary Session \| DSMP 2022-23 25 minutes | https://www.youtube.com/watch?v=80QpbZA38HA |
| 53 | Session 18 - Important DataFrame Methods \| Data Science Mentorship Program (DSMP) 2022-23 2 hours, 12 minutes | https://www.youtube.com/watch?v=zTa4MIrGTIE |
| 54 | API Development using Flask \| Paid Session \| DSMP 2022-23 2 hours, 43 minutes | https://www.youtube.com/watch?v=-bqSWUHJ1ek |
| 55 | Week 5 - Numpy Interview Questions \| Paid Session \| DSMP 2022-23 2 hours, 34 minutes | https://www.youtube.com/watch?v=wjv01Opo6Bk |
| 56 | Session 19 - GroupBy Object in Pandas \| Data Science Mentorship Program (DSMP) 2022-23 2 hours, 9 minutes | https://www.youtube.com/watch?v=LPBjF4_gZnI |
| 57 | Task 16 Solutions \| Paid Recording \| DSMP 2022-23 13 minutes, 49 seconds | https://www.youtube.com/watch?v=DmOPhX7jE0U |
| 58 | Task 17 Solutions \| Paid Recording \| DSMP 2022-23 16 minutes | https://www.youtube.com/watch?v=NYzbSm8YCag |
| 59 | Task 18 Solutions \| Paid Recording \| DSMP 2022-23 39 minutes | https://www.youtube.com/watch?v=88GawL9n3FY |
| 60 | Session 20 - Merging, Joining \u0026 Concatenating \| DSMP 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=Ssy1EfK5S-o |
| 61 | Session 21 - MultiIndex Series and DataFrames \| DSMP 2022-23 1 hour, 57 minutes | https://www.youtube.com/watch?v=QzoRUwz8DoM |
| 62 | Session on Streamlit \| Paid Session \| DSMP 2022-23 2 hours, 24 minutes | https://www.youtube.com/watch?v=ob8w91O3DbQ |
| 63 | Pandas Case Study 1 \| Indian Startup Funding 2 hours, 23 minutes | https://www.youtube.com/watch?v=VGPnRAASARY |
| 64 | Session 1 on Git Fundamentals 2 hours, 11 minutes | https://www.youtube.com/watch?v=Q3aqtqCZNOQ |
| 65 | Task 19 Solutions \| Paid Recording \| DSMP 2022-23 30 minutes | https://www.youtube.com/watch?v=1IjSVS-4rPg |
| 66 | Task 20 Solutions \| Paid Recording \| DSMP 2022-23 57 minutes | https://www.youtube.com/watch?v=iyiPAsu_6DY |
| 67 | Session on Git and Github Part 2 \| Paid Session \| DSMP 2022-23 2 hours, 9 minutes | https://www.youtube.com/watch?v=C194ihuPM0U |
| 68 | Talking Data Science with Pranjal Yadav \| Guest Session 1 hour, 24 minutes | https://www.youtube.com/watch?v=M1sNi0ipKYk |
| 69 | CampusX DSMP 2023 January Schedule 9 minutes, 57 seconds | https://www.youtube.com/watch?v=kjX6eIuTV5Y |
| 70 | Session 22 - Vectorized String Operations \| DateTime in Pandas \| Pivot Table \| DSMP 2022-23 2 hours, 6 minutes | https://www.youtube.com/watch?v=UE6DmRQJ2w8 |
| 71 | Pandas Case Study   Time Series Analysis 2 hours, 27 minutes | https://www.youtube.com/watch?v=LknR_AVYboc |
| 72 | Pandas Case Study 2   Working with textual data 2 hours, 40 minutes | https://www.youtube.com/watch?v=cO7sebFXWT8 |
| 73 | Session 23 - Plotting using Matplotlib \| DSMP 2022-23 2 hours, 6 minutes | https://www.youtube.com/watch?v=XaKn_cKFlSY |
| 74 | Task 21 Solutions \| Recorded Session \| DSMP 2023 25 minutes | https://www.youtube.com/watch?v=pwTzBScr_K4 |
| 75 | Session 24 - Advanced Matplotlib \| DSMP 2022-23 2 hours, 13 minutes | https://www.youtube.com/watch?v=7YDc5xU9CQQ |
| 76 | Session on Plotly Express 2 hours, 14 minutes | https://www.youtube.com/watch?v=VPqw8soG4ZY |
| 77 | Session on Plotly Part 2 2 hours, 9 minutes | https://www.youtube.com/watch?v=_ZsVLh6v29A |
| 78 | Session 25 - Plotting using Seaborn \| DSMP 2022-23 1 hour, 50 minutes | https://www.youtube.com/watch?v=DWVLRhnuGqI |
| 79 | Task 22 Solutions \| Paid Recordings \| DSMP 2023 42 minutes | https://www.youtube.com/watch?v=iauvbdt572s |
| 80 | Session 26 - Plotting using Seaborn Part 2 \| DSMP 2022-23 1 hour, 41 minutes | https://www.youtube.com/watch?v=kLWeKeqc9Ms |
| 81 | Session on Open Source Software Part 1 1 hour | https://www.youtube.com/watch?v=ve532huOUU8 |
| 82 | Session 27 - Data Gathering \| Data Analysis Process \| DSMP 2023 2 hours, 50 minutes | https://www.youtube.com/watch?v=dA6ZksRR6aw |
| 83 | Session on Open Source Software Part 2 \| Paid Session 1 hour, 9 minutes | https://www.youtube.com/watch?v=SetEc8en33Q |
| 84 | Session 28 - Data Assessing and Cleaning \| DSMP 2022-23 2 hours, 26 minutes | https://www.youtube.com/watch?v=-HnN8GBINnc |
| 85 | Task 23 Solutions \| Paid Recording \| DSMP 2023 43 minutes | https://www.youtube.com/watch?v=pUNmHlY7bfA |
| 86 | Session on ETL using AWS RDS \| Paid Session \| DSMP 2023 2 hours, 13 minutes | https://www.youtube.com/watch?v=bSB5G5Fgkhg |
| 87 | Session on Advanced Web Scraping using Selenium 3 hours, 6 minutes | https://www.youtube.com/watch?v=hkXW_FabPAI |
| 88 | Session on Data Cleaning Case Study - Smartphone dataset 2 hours, 10 minutes | https://www.youtube.com/watch?v=0j_7j3PNSEw |
| 89 | Month of SQL \| CampusX DSMP 2023 \| Feb Schedule 13 minutes, 15 seconds | https://www.youtube.com/watch?v=5v8oTYGbh8M |
| 90 | Build a Data Scientist's Dream LinkedIn Profile with ChatGPT 45 minutes | https://www.youtube.com/watch?v=0m1dXTy8IpE |
| 91 | Session 29 - Exploratory Data Analysis \| Data Analysis Process \| DSMP 2022-23 2 hours, 30 minutes | https://www.youtube.com/watch?v=PPEHpg2RixQ |
| 92 | Task 25 Solutions \| Paid Recording \| DSMP 2023 20 minutes | https://www.youtube.com/watch?v=mRZSmT73JjY |
| 93 | Task 26 Solutions \| Paid Recording \| DSMP 2023 30 minutes | https://www.youtube.com/watch?v=zlU-73jpq3Q |
| 94 | Task 24 Solutions \| Paid Recording \| DSMP 2023 40 minutes | https://www.youtube.com/watch?v=GIkTjDqH4II |
| 95 | Session 2 on Data Cleaning - Smartphones Dataset \| Paid Session \| DSMP 2023 2 hours, 28 minutes | https://www.youtube.com/watch?v=o-8KvsSdDiE |
| 96 | Session 30 - Database Fundamentals \| DSMP 2022-23 2 hours, 15 minutes | https://www.youtube.com/watch?v=CTLwYjh92xw |
| 97 | Session on EDA Case Study  - Smartphones Dataset \| Paid Session \| DSMP 2023 2 hours, 29 minutes | https://www.youtube.com/watch?v=PxRth8xosqY |
| 98 | Task 27 Solutions \| Paid Recording \| DSMP 2023 56 minutes | https://www.youtube.com/watch?v=dmz5CCQSc74 |
| 99 | Session 31 - SQL DDL Commands \| DSMP 2023 2 hours, 4 minutes | https://www.youtube.com/watch?v=ny1mh6VUpnQ |
| 100 | Session 1 on Tableau \| Paid Session \| DSMP 2023 1 hour, 43 minutes | https://www.youtube.com/watch?v=RhA13o7BQUE |
| 101 | Session 32 - SQL DML Commands \| DSMP 2023 2 hours, 4 minutes | https://www.youtube.com/watch?v=kzMZDlvHFak |
| 102 | Session 33 - SQL Grouping + Sorting \| DSMP 2023 2 hours, 5 minutes | https://www.youtube.com/watch?v=nsKcmOly0UY |
| 103 | Career Pe Charcha - Soft Skills Masterclass \| DSMP 2023 1 hour, 12 minutes | https://www.youtube.com/watch?v=DWUtCsFGPr8 |
| 104 | Session 34 - SQL Joins\| DSMP 2023 2 hours, 10 minutes | https://www.youtube.com/watch?v=s6xAVqYdmJ4 |
| 105 | Career QnA for Paid Members 1 hour, 48 minutes | https://www.youtube.com/watch?v=ZEzOEZi8_7I |
| 106 | Session 2 on Tableau - Sales Dataset 2 hours, 16 minutes | https://www.youtube.com/watch?v=6_xWLmCbkDI |
| 107 | Session 35 - Subqueries in SQL \| DSMP 2023 2 hours, 18 minutes | https://www.youtube.com/watch?v=YYq47MN3TZI |
| 108 | SQL Case Study 1 \| Zomato Dataset 2 hours, 10 minutes | https://www.youtube.com/watch?v=jYwcB0KEFfY |
| 109 | Task 32 Solutions 14 minutes, 21 seconds | https://www.youtube.com/watch?v=Nz6AQUj2BU0 |
| 110 | Making a Flights Dashboard using Python and SQL \| Paid Recording \| DSMP 2023 2 hours, 3 minutes | https://www.youtube.com/watch?v=9ofjk2TaQgE |
| 111 | SQL Interview Questions Part 1 2 hours, 4 minutes | https://www.youtube.com/watch?v=QlenVBvzrW4 |
| 112 | Session 36 - Window Functions in SQL \| DSMP 2023 1 hour, 40 minutes | https://www.youtube.com/watch?v=3LJ-yWBQBGs |
| 113 | Task 33 Solutions \| Paid Recording \| DSMP 2023 34 minutes | https://www.youtube.com/watch?v=hi0YTLUraVk |
| 114 | Career Pe Charcha \| Markdown Basics \| How to build a Kickass Github Profile 25 minutes | https://www.youtube.com/watch?v=NMlJD-djTmQ |
| 115 | Session 37 - Window Functions Part 2 \| SQL Tutorials 47 minutes | https://www.youtube.com/watch?v=YgizUxmBFK8 |
| 116 | Session 37 (Cont..) - Window Functions Part 3 \| SQL Tutorial 53 minutes | https://www.youtube.com/watch?v=e0HQVOdfqlQ |
| 117 | Session on Data Cleaning using SQL \|  Laptops Dataset 2 hours, 18 minutes | https://www.youtube.com/watch?v=NsOXJdRIamc |
| 118 | Session on EDA using SQL \| Laptops Dataset 2 hours, 9 minutes | https://www.youtube.com/watch?v=dv8TkUTlQvI |
| 119 | CampusX DSMP March Schedule \| Dher Sara Statistics!!! 7 minutes, 37 seconds | https://www.youtube.com/watch?v=hysPNdlpXpI |
| 120 | Task 34 Solutions \| Paid recording \| DSMP 2023 29 minutes | https://www.youtube.com/watch?v=jnMipVT6SDc |
| 121 | Task 35 Solutions \| Paid Recording \| DSMP 2023 36 minutes | https://www.youtube.com/watch?v=obAhEPrckTQ |
| 122 | CampusX Data Science Mentorship Program 2023 21 minutes | https://www.youtube.com/watch?v=gzMRsbNSSbc |
| 123 | Session 38 - Descriptive Statistics Part 1 \| DSMP 2023 1 hour, 52 minutes | https://www.youtube.com/watch?v=Uv3Blie7F3g |
| 124 | Session on Datetime in SQL \| Paid Session \| DSMP 2023 2 hours, 14 minutes | https://www.youtube.com/watch?v=OY9YiALhSR4 |
| 125 | Session 39 - Descriptive Statistics Part 2 \| DSMP 2023 1 hour, 55 minutes | https://www.youtube.com/watch?v=1ndVC500-EU |
| 126 | Session 40 - Probability Distribution Functions - PDF, PMF & CDF \| DSMP 2023 2 hours, 9 minutes | https://www.youtube.com/watch?v=C_QAURbgBqY |
| 127 | SQL Datetime Case Study on Flights Dataset \| Paid Session \| DSMP 2023 1 hour, 52 minutes | https://www.youtube.com/watch?v=zY1UCTko64E |
| 128 | Session on Database Design \| SQL Data Types \| Database Normalization 2 hours, 25 minutes | https://www.youtube.com/watch?v=8SOs0URTvr0 |
| 129 | Session 41 - Normal Distribution \| DSMP 2023 2 hours, 1 minute | https://www.youtube.com/watch?v=ADqYqSdtyW8 |
| 130 | Task 36 Solutions 35 minutes | https://www.youtube.com/watch?v=ULfLxlzAlrc |
| 131 | Session 42 - Non-Gaussian Probability Distributions \| DSMP 2023 2 hours, 16 minutes | https://www.youtube.com/watch?v=U6QCc_3zgUk |
| 132 | Session on Views and User Defined Functions in SQL \| Paid Session \| DSMP 2023 1 hour, 54 minutes | https://www.youtube.com/watch?v=dMoKTx1bVnY |
| 133 | Session on Transactions & Stored Procedures 2 hours, 21 minutes | https://www.youtube.com/watch?v=-O_9iR7XsJw |
| 134 | Session 43 - Central Limit Theorem \| DSMP 2023 1 hour, 51 minutes | https://www.youtube.com/watch?v=-WmJDYBor7c |
| 135 | Session 44 - Confidence Intervals \| DSMP 2023 2 hours, 10 minutes | https://www.youtube.com/watch?v=X52HK2qkiIE |
| 136 | April Schedule \| CampusX Data Science Mentorship Program 2023 12 minutes | https://www.youtube.com/watch?v=O0omBfGHYfs |
| 137 | Session 45 - Hypothesis Testing Part 1 \| DSMP 2023 1 hour, 59 minutes | https://www.youtube.com/watch?v=S94mx6OL7kM |
| 138 | Session 46 - Hypothesis Testing Part 2 \| p-values \| t-tests \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=xHTMjxx14sU |
| 139 | Session on Chi Square Tests 2 hours, 15 minutes | https://www.youtube.com/watch?v=_8DTqznosvY |
| 140 | Session on ANOVA  \| Paid Zoom Session 1 hour, 58 minutes | https://www.youtube.com/watch?v=G9ppvMWyd54 |
| 141 | Supercharge Your ML Journey: Mastering Vectors in Linear Algebra - Part 1 1 hour, 31 minutes | https://www.youtube.com/watch?v=mQewAJb8oJ8 |
| 142 | Linear Algebra Part 2 \| Matrices 2 hours, 26 minutes | https://www.youtube.com/watch?v=Y8UjSNSMeZ4 |
| 143 | Linear Algebra Part 3  \| Matrices  Intuition 2 hours, 10 minutes | https://www.youtube.com/watch?v=7IV56bGTJTQ |
| 144 | Introduction to Machine Learning 3 hours, 26 minutes | https://www.youtube.com/watch?v=1ctqJCHMAmc |
| 145 | Simple Linear Regression \| Lecture 49 \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=aEPoLeS6UMM |
| 146 | Session 50 - Multiple Linear Regression \| DSMP 2023 1 hour, 49 minutes | https://www.youtube.com/watch?v=e_9vJXokd-Y |
| 147 | Level Up Your Data Science Skills with ChatGPT \| Unlock the Secrets of AI-Assisted Learning 50 minutes | https://www.youtube.com/watch?v=5Y4pWR1P8G0 |
| 148 | Session on Optimization - The Big Picture 2 hours, 9 minutes | https://www.youtube.com/watch?v=T2f8249569Q |
| 149 | Session on Differential Calculus 2 hours, 25 minutes | https://www.youtube.com/watch?v=-siBIGUREjw |
| 150 | Gradient Descent From Scratch \| End to End Gradient Descent \| Gradient Descent Animation 1 hour, 57 minutes | https://www.youtube.com/watch?v=ORyfPJypKuU |
| 151 | Batch Gradient Descent with Code Demo \| Simple Explanation in Hindi 1 hour, 4 minutes | https://www.youtube.com/watch?v=Jyo53pAyVAM |
| 152 | Stochastic Gradient Descent 49 minutes | https://www.youtube.com/watch?v=V7KBAa_gh4c |
| 153 | Mini-Batch Gradient Descent 22 minutes | https://www.youtube.com/watch?v=_scscQ4HVTY |
| 154 | Doubt Clearance Session on Linear Regression 2 hours, 9 minutes | https://www.youtube.com/watch?v=duUyp91EREU |
| 155 | Session 1 on Regression Analysis 2 hours, 16 minutes | https://www.youtube.com/watch?v=qB5k6A60Unw |
| 156 | DSMP 2023 May Schedule \| CampusX Data Science Mentorship Program 5 minutes, 31 seconds | https://www.youtube.com/watch?v=CyFW7wn41qk |
| 157 | Session 2 on Regression Analysis 1 hour, 54 minutes | https://www.youtube.com/watch?v=6QQdhk2_H9Q |
| 158 | Polynomial Regression \| Machine Learning 26 minutes | https://www.youtube.com/watch?v=BNWLf3cKdbQ |
| 159 | Session on Assumptions of Linear Regression 2 hours, 25 minutes | https://www.youtube.com/watch?v=EWV9laVakjc |
| 160 | Session 53 - Session on Multicollinearity \| DSMP 2023 2 hours, 11 minutes | https://www.youtube.com/watch?v=XGJLWJ0a6aU |
| 161 | Session 54 - Feature Selection Part 1 \| Filter Methods \| Variance Threshold \| Chi-Square \| DSMP 2023 2 hours, 6 minutes | https://www.youtube.com/watch?v=hRTe7tSkR2k |
| 162 | Session 55 - Feature Selection Part 2 \| Wrapper Methods  \| DSMP 2023 1 hour, 53 minutes | https://www.youtube.com/watch?v=6-1WiITAW10 |
| 163 | Session 3 on Feature Selection \| Embedded Methods 1 hour, 59 minutes | https://www.youtube.com/watch?v=zLB-csjZ9V4 |
| 164 | Regularization Part 1 \| Bias Variance Trade-off 2 hours, 2 minutes | https://www.youtube.com/watch?v=qfWUIUvKi98 |
| 165 | Regularization Part 2 \| What is Regularization? 1 hour, 56 minutes | https://www.youtube.com/watch?v=cVsN5VAioz0 |
| 166 | What is K Nearest Neighbors? \| KNN Explained in Hindi \| Simple Overview in 1 Video \| CampusX 52 minutes | https://www.youtube.com/watch?v=abnL_GUGub4 |
| 167 | Doubt Clearance Session on Regularization 1 hour, 9 minutes | https://www.youtube.com/watch?v=7_DDQkLzmRM |
| 168 | Session on Advanced KNN 2 hours, 6 minutes | https://www.youtube.com/watch?v=d7xo69q-1H4 |
| 169 | Ridge Regression Part 1 \| Geometric Intuition and Code \| Regularized Linear Models 19 minutes | https://www.youtube.com/watch?v=aEow1QoTLo0 |
| 170 | Ridge Regression Part 2 \| Mathematical Formulation & Code from scratch \| Regularized Linear Models 43 minutes | https://www.youtube.com/watch?v=oDlZBQjk_3A |
| 171 | Ridge Regression Part 3 \| Gradient Descent \| Regularized Linear Models 18 minutes | https://www.youtube.com/watch?v=Fci_wwMp8G8 |
| 172 | 5 Key Points - Ridge Regression \| Part 4 \| Regularized Linear Models 30 minutes | https://www.youtube.com/watch?v=8osKeShYVRQ |
| 173 | Lasso Regression \| Intuition and Code Sample \| Regularized Linear Models 28 minutes | https://www.youtube.com/watch?v=HLF4bFbBgwk |
| 174 | Why Lasso Regression creates sparsity? 24 minutes | https://www.youtube.com/watch?v=FN4aZPIAfI4 |
| 175 | ElasticNet Regression \| Intuition and Code Example \| Regularized Linear Models 11 minutes, 41 seconds | https://www.youtube.com/watch?v=2g2DBkFhTTY |
| 176 | Accuracy and Confusion Matrix \| Type 1 and Type 2 Errors \| Classification Metrics Part 1 34 minutes | https://www.youtube.com/watch?v=c09drtuCS3c |
| 177 | Precision, Recall and F1 Score \| Classification Metrics Part 2 42 minutes | https://www.youtube.com/watch?v=iK-kdhJ-7yI |
| 178 | Curse of Dimensionality 15 minutes | https://www.youtube.com/watch?v=ToGuhynu-No |
| 179 | Principle Component Analysis  (PCA) \| Part 1 \| Geometric Intuition 33 minutes | https://www.youtube.com/watch?v=iRbsBi5W0-c |
| 180 | Principle Component Analysis (PCA) \| Part 2 \| Problem Formulation and Step by Step Solution 56 minutes | https://www.youtube.com/watch?v=tXXnxjj2wM4 |
| 181 | Principle Component Analysis(PCA) \| Part 3 \| Code Example and Visualization 43 minutes | https://www.youtube.com/watch?v=tofVCUDrg4M |
| 182 | Session on Eigen Vectors and Eigen Values \| Week 29 \| DSMP 1 hour, 40 minutes | https://www.youtube.com/watch?v=5HDkRG5gQHM |
| 183 | Session on Eigen Decomposition + PCA Variants \| Week 29 \| DSMP 2 hours, 7 minutes | https://www.youtube.com/watch?v=D5IlO4lzaZ0 |
| 184 | Session on Singular Value Decomposition 2 hours, 13 minutes | https://www.youtube.com/watch?v=NPkMoUNkEtQ |
| 185 | ROC Curve in Machine Learning \| ROC-AUC in Machine Learning Simplified \| CampusX 1 hour, 11 minutes | https://www.youtube.com/watch?v=gdW6hj9IXaA |
| 186 | Session on Cross Validation 1 hour, 8 minutes | https://www.youtube.com/watch?v=cY_pBnFM7fQ |
| 187 | Session on Data Leakage 2 hours, 25 minutes | https://www.youtube.com/watch?v=KHvcr0Q7jSk |
| 188 | Session on Hyperparameter Tuning 1 hour, 46 minutes | https://www.youtube.com/watch?v=cDmMVCC4dEE |
| 189 | Master Probability in Data Science: The Ultimate Crash Course! \| Part 1 \| CampusX 1 hour, 34 minutes | https://www.youtube.com/watch?v=DUT4WEUngt0 |
| 190 | Mastering Probability for ML: Joint \| Marginal \| Conditional Probability and Bayes' Theorem 1 hour, 20 minutes | https://www.youtube.com/watch?v=ndHDsvqmbuI |
| 191 | Session 2 on Naive Bayes 2 hours, 13 minutes | https://www.youtube.com/watch?v=mrBut7CPCxE |
| 192 | Session 1 on Naive Bayes 2 hours | https://www.youtube.com/watch?v=mEjGU4e-WPs |
| 193 | Session 3 on Naive Bayes 1 hour, 58 minutes | https://www.youtube.com/watch?v=_x61Lut84zI |
| 194 | Session 1 on Logistic Regression 2 hours, 8 minutes | https://www.youtube.com/watch?v=BuAp03q5GAw |
| 195 | Session on Softmax Regression \| Multiclass Classification \| Week 32 \| DSMP 1 hour, 52 minutes | https://www.youtube.com/watch?v=Dg-1TuPxElw |
| 196 | Session on Maximum Likelihood Estimation 2 hours, 7 minutes | https://www.youtube.com/watch?v=YSur3ESoV_U |
| 197 | Session 3 on Logistic Regression 2 hours, 6 minutes | https://www.youtube.com/watch?v=u2IfZX5q2jY |
| 198 | SVM Part 1 \| Hard Margin SVM 1 hour, 42 minutes | https://www.youtube.com/watch?v=HfrKYccB2DQ |
| 199 | Feature Engineering \| Extra Session \| Session on Handling Missing Values Part 1 2 hours, 33 minutes | https://www.youtube.com/watch?v=e1-5NOwWCmo |
| 200 | SVM Part 2 - Soft Margin SVM 1 hour, 46 minutes | https://www.youtube.com/watch?v=SlsXm-utgFY |

