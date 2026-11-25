# E-Commerce Customer Analytics

An end-to-end, locally executed Business Intelligence (BI) and Machine Learning (ML) analytics system built in Python on real **UCI Online Retail II** transaction records (`2009-12-01` to `2011-12-09`). The project combines structured data auditing, rule-based cleaning, RFM and behavioral feature engineering, exploratory analysis, four serialized machine learning pipelines (`Linear Regression`, `Logistic Regression`, `Item-Based KNN`, and `K-Means`), and an 11-page interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Technology Stack](#5-technology-stack)
6. [Project Folder Structure](#6-project-folder-structure)
7. [Installation Instructions](#7-installation-instructions)
8. [Environment Setup](#8-environment-setup)
9. [How to Run Notebooks](#9-how-to-run-notebooks)
10. [How to Run the Dashboard](#10-how-to-run-the-dashboard)
11. [Machine Learning Algorithms](#11-machine-learning-algorithms)
12. [Model Evaluation Results](#12-model-evaluation-results)
13. [Limitations](#13-limitations)
14. [Future Improvements](#14-future-improvements)

---

## 1. Project Overview

Online retail businesses accumulate high-volume transactional ledgers containing valid orders, customer cancellations/returns, shipping fees, guest checkouts, and duplicate exports. Without disciplined engineering, raw transaction tables distort revenue KPIs, leak future outcomes into predictive features, and produce unreliable models.

This project implements a reproducible analytics lifecycle across three layers:
- **Notebooks (`notebooks/`):** Eight sequential, documented workflows from raw dataset audit through unsupervised segmentation.
- **Modular Python Package (`src/`):** Tested, reusable pipeline code for path resolution, quality auditing, cleaning, feature engineering, EDA KPIs, ML training/evaluation, and schema-validated inference (`src/models/inference.py`).
- **Interactive Dashboard (`app/`):** An 11-page Streamlit + Plotly web application backed by cached analytics tables (`dataset/processed/analytics/`) and saved `.joblib` artifacts (`models/`).

All analysis and modeling use exclusively the real UCI Online Retail II records provided—no synthetic customers, ratings, or demographics were added.

---

## 2. Problem Statement

E-commerce managers need reliable answers to four operational questions:
1. **What is the true net business performance** across time periods, products, and countries once sheet-overlap duplicates, warehouse write-offs, and customer cancellations are separated accurately?
2. **How much will an active customer spend** in the upcoming quarter (`90` days) without using leaked post-cutoff data?
3. **Which customers are likely to place a repeat purchase** in the next `90` days so retention campaigns can be budgeted cleanly?
4. **Which products should be recommended to each customer**, and how can buyers be segmented by Recency, Frequency, and Monetary (`RFM`) value into actionable personas?

---

## 3. Objectives

- Preserve the original raw CSV read-only (`dataset/Ecommerce.csv`) and verify byte-for-byte integrity.
- Perform a complete data quality audit documenting missingness, duplicates, cancellations (`C`-invoices), accounting adjustments (`A`-invoices), and anomalies.
- Engineer clean transaction-, customer-, product-, country-, and monthly-level analytical tables.
- Conduct quantitative EDA covering seasonality, weekday/hourly behavior, Pareto concentration, customer repeat retention, and market mix.
- Build and benchmark four non-leaking ML components against explicit baselines, serialize trained pipelines to `models/`, and verify error-resilient inference via unit tests.
- Deploy a clean, beginner-friendly interactive dashboard (`app/app.py`) exposing both BI exploration and live ML predictions.

---

## 4. Dataset Description

- **Source File (Immutable):** `dataset/Ecommerce.csv` (`90.46 MB`, UTF-8, SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`)
- **Time Span:** `2009-12-01 07:45:00` to `2011-12-09 12:50:00` (combined `2009–2010` and `2010–2011` UCI Online Retail II periods; `25` monthly snapshots)
- **Raw Schema (`1,067,371` rows × `8` columns):** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- **Cleaning Summary:**
  - Retained **`1,026,678`** valid commercial transactions (`96.19%`), partitioned cleanly into **`1,007,823`** normal positive sales (`£20,434,219.52` gross) and **`18,855`** valid `C`-prefixed cancellations/returns (`-£1,168,287.81`).
  - Removed `40,693` invalid/noise rows (`34,335` exact duplicates—including `22,844` in the Dec 1–9, 2010 sheet overlap window—plus `6` `A`-prefix bad-debt adjustments, `17` `TEST*` rows, `3,393` zero-price non-`C` warehouse write-offs, `2,618` positive `Price <= 0` entries, `1` anomalous positive `C`-row `C496350`, and `323` internal fee/sample rows).
  - Preserved **`228,994`** valid guest/unregistered checkout rows (`has_customer_id = False`) for store-wide revenue/product metrics without fabricating customer IDs.
  - Non-null customer modeling tables cover `5,869` purchasing customers (plus `52` return-only accounts tracked transparently during cleaning).

---

## 5. Technology Stack

- **Language:** Python `3.14` (`3.14.6` in `.venv`)
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `plotly`, `matplotlib`, `seaborn`
- **Machine Learning & Serialization:** `scikit-learn`, `scipy`, `joblib`
- **Interactive Web Dashboard:** `streamlit`
- **Notebooks & Testing:** `jupyter`, `ipykernel`, `nbconvert`, `unittest` (`streamlit.testing.v1.AppTest`)

---

## 6. Project Folder Structure

```text
E-Commerce-Customer-Analytics/
├── .streamlit/                     # Streamlit Light BI theme & telemetry configuration
│   └── config.toml
├── .venv/                          # Local Python virtual environment (excluded from Git)
├── .vscode/                        # Local editor Python interpreter settings
│   └── settings.json
├── app/                            # Multi-page Streamlit interactive dashboard
│   ├── app.py                      # Main entrypoint (st.navigation across 11 pages)
│   ├── components/
│   │   └── ui.py                   # KPI cards, page headers, callouts, CSV download button
│   ├── pages/                      # 11 modular Streamlit dashboard views (p01–p11)
│   │   ├── p01_home.py
│   │   ├── p02_sales.py
│   │   ├── p03_customers.py
│   │   ├── p04_products.py
│   │   ├── p05_countries.py
│   │   ├── p06_future_spending.py
│   │   ├── p07_purchase_pred.py
│   │   ├── p08_recommendations.py
│   │   ├── p09_segmentation.py
│   │   ├── p10_model_perf.py
│   │   └── p11_data_quality.py
│   └── utils/
│       ├── data_loader.py          # Cached analytics, report, and .joblib model loaders
│       └── theme.py                # Consistent palette, CSS styles, Plotly helper
├── dataset/                        # Raw CSV (read-only) & derived analytical tables
│   ├── Ecommerce.csv
│   └── processed/
│       ├── cleaned_online_retail.csv
│       ├── returns_and_cancellations.csv
│       ├── sales_transactions.csv
│       └── analytics/
│           ├── country_analytics.csv
│           ├── customer_analytics.csv
│           ├── customer_segments.csv
│           ├── monthly_sales.csv
│           ├── product_analytics.csv
│           └── transaction_features.csv
├── models/                         # Serialized Scikit-Learn & recommender artifacts
│   ├── kmeans_pipeline.joblib
│   ├── knn_recommender.joblib
│   ├── linear_regression_pipeline.joblib
│   └── logistic_regression_pipeline.joblib
├── notebooks/                      # Executed analytical & modeling notebooks (01–08)
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Exploratory_Data_Analysis.ipynb
│   ├── 05_Linear_Regression.ipynb
│   ├── 06_Logistic_Regression.ipynb
│   ├── 07_KNN_Recommendation.ipynb
│   └── 08_KMeans_Customer_Segmentation.ipynb
├── reports/                        # Structured CSV metrics & Markdown evaluation reports
│   ├── data_cleaning_report.md
│   ├── data_quality_report.csv
│   ├── data_quality_report.md
│   ├── eda_business_insights.md
│   ├── final_project_report.md
│   ├── kmeans_cluster_summary.csv
│   ├── knn_recommender_results.csv
│   ├── linear_regression_results.csv
│   ├── logistic_regression_results.csv
│   └── model_evaluation_summary.md
├── src/                            # Reusable Python source modules
│   ├── analysis/
│   │   └── eda_metrics.py
│   ├── data/
│   │   ├── audit.py
│   │   └── cleaning.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── inference.py
│   │   ├── recommender.py
│   │   ├── regression.py
│   │   └── segmentation.py
│   └── utils/
│       └── paths.py
├── tests/                          # Automated unit & headless Streamlit dashboard tests
│   ├── test_classification.py
│   ├── test_data_cleaning.py
│   ├── test_eda_metrics.py
│   ├── test_feature_engineering.py
│   ├── test_inference.py
│   ├── test_recommender.py
│   ├── test_regression.py
│   ├── test_segmentation.py
│   └── test_streamlit_app.py
├── visualizations/                 # Exported static PNG plots & interactive HTML charts
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 7. Installation Instructions

Open Windows PowerShell inside the project root (`E-Commerce-Customer-Analytics`):

```powershell
# 1. Create a virtual environment (if setting up from scratch)
python -m venv .venv

# 2. Install exact pinned dependencies into .venv (never install globally)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Verify package integrity and automated test suite
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

---

## 8. Environment Setup

### Register the Local Jupyter Kernel

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name ecommerce-analytics --display-name "Python (.venv - E-Commerce Analytics)"
```

### IDE Configuration (VS Code / Antigravity)

- **Python Interpreter:** Set in `.vscode/settings.json` to `${workspaceFolder}/.venv/Scripts/python.exe` (Command Palette → `Python: Select Interpreter`).
- **Jupyter Kernel:** Choose `Python (.venv - E-Commerce Analytics)` via the top-right `Select Kernel` menu.
- **Windows PowerShell Note:** If PowerShell script execution policies or Windows Application Control block pip-generated `.exe` wrappers inside `.venv\Scripts\` (such as `streamlit.exe` or `jupyter.exe`), run tools via Python's `-m` flag instead:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
.\.venv\Scripts\python.exe -m notebook
```

---

## 9. How to Run Notebooks

Execute the notebooks sequentially so upstream cleaned datasets, feature tables, and saved models exist for downstream stages:

1. `notebooks/01_Data_Inspection.ipynb` — Read-only data quality audit → `reports/data_quality_report.{csv,md}`
2. `notebooks/02_Data_Cleaning.ipynb` — Rule-based cleaning & return separation → `dataset/processed/*.csv`, `reports/data_cleaning_report.md`
3. `notebooks/03_Feature_Engineering.ipynb` — Transaction, Customer (RFM), Product, Country, and Monthly analytical tables → `dataset/processed/analytics/*.csv`
4. `notebooks/04_Exploratory_Data_Analysis.ipynb` — Executive KPIs, trends, Pareto & RFM visuals → `reports/eda_business_insights.md`, `visualizations/`
5. `notebooks/05_Linear_Regression.ipynb` — Chronological `275d → 90d` future spend model → `models/linear_regression_pipeline.joblib`
6. `notebooks/06_Logistic_Regression.ipynb` — Chronological `275d → 90d` repeat purchase probability model → `models/logistic_regression_pipeline.joblib`
7. `notebooks/07_KNN_Recommendation.ipynb` — Sparse Item-Based Cosine KNN recommender → `models/knn_recommender.joblib`
8. `notebooks/08_KMeans_Customer_Segmentation.ipynb` — Log-scaled RFM K-Means segmentation (`K=4` + `ClusterID=-1`) → `models/kmeans_pipeline.joblib`, `customer_segments.csv`

To re-execute any notebook cleanly from PowerShell using `nbconvert`:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ecommerce-analytics notebooks/01_Data_Inspection.ipynb
```

---

## 10. How to Run the Dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Open **`http://localhost:8501`** in Chrome/Edge/Firefox to access all **11 interactive pages**:

| Section | Pages | Key Capabilities |
|---|---|---|
| **Overview** | 1. Home (`p01_home.py`) | Executive KPIs (`£19.27M` net revenue, `39,988` orders, `5,869` buyers, `4,746` SKUs), monthly gross vs. net chart, navigation guide |
| **Business Analytics** | 2. Sales Analytics (`p02_sales.py`) | Filters by year, country, and scope (`All Valid`, `Merchandise Only`, `Normal Sales Only`); monthly, day-of-week & hourly demand; CSV export |
| | 3. Customer Analytics (`p03_customers.py`) | Filters by segment, min orders, net spend; top customers, log10 spend & frequency distributions; CSV export |
| | 4. Product Analytics (`p04_products.py`) | SKU/description search, reach threshold, revenue & quantity leaders, ASP bubble scatter; CSV export |
| | 5. Country Analytics (`p05_countries.py`) | UK vs. 42 international markets toggle, horizontal bar comparisons, market-share treemap; CSV export |
| **Predictive & Recommender ML** | 6. Future Spending Prediction (`p06_future_spending.py`) | Real test-cohort scoring + Interactive What-If Feature Simulator (`Linear Regression`) + batch CSV export |
| | 7. Purchase Prediction (`p07_purchase_pred.py`) | Live threshold tuning slider (`0.10–0.90`), confusion metrics, customer propensity bar, score distribution; CSV export |
| | 8. Product Recommendation (`p08_recommendations.py`) | Personalized Item-KNN top-N + popularity cold-start fallback + Similar Product Explorer (`find_similar_products`) |
| | 9. Customer Segmentation (`p09_segmentation.py`) | Segment share & 2D PCA cluster scatter, cluster RFM table, customer lookup & custom RFM live scorer |
| **Evaluation & Governance** | 10. Model Performance (`p10_model_perf.py`) | Independent evaluation tables, charts, Notebook 05–08 diagnostic PNGs, and unified summary report |
| | 11. Data Quality (`p11_data_quality.py`) | Raw missingness & log-scale anomaly audit charts, complete `data_quality_report.csv` table, and cleaning report |

---

## 11. Machine Learning Algorithms

1. **Future Customer Spending Prediction (`src/models/regression.py`):**
   - **Algorithm:** OLS `LinearRegression` inside a Scikit-Learn `Pipeline` (`SimpleImputer(strategy="median")` → `StandardScaler()` → `LinearRegression()`).
   - **Leakage Control:** Strictly chronological `275`-day observation window (`<= cutoff`) predicting subsequent `90`-day Q4 net spending (`Train 2009–2010: n = 3,270`, `Test 2010–2011: n = 3,248`).
2. **Future Purchase Prediction (`src/models/classification.py`):**
   - **Algorithm:** Binary `LogisticRegression` (`SimpleImputer(median)` → signed `log1p` transform → `StandardScaler()` → `LogisticRegression(max_iter=1000)`).
   - **Thresholding:** Evaluates default `0.50` alongside 5-fold Train-CV tuned thresholds (`0.56` balanced precision, `0.34` max-F1/high recall).
3. **Product Recommendation System (`src/models/recommender.py`):**
   - **Algorithm:** Item-Based Collaborative Filtering with **Cosine similarity** over sparse CSR `log1p(total_quantity)` matrices (merchandise SKUs with `>= 5` unique training buyers, `n_neighbors = 25`), selected empirically over User-Based KNN (`38.48%` vs. `35.97%` Hit Rate@10). Includes a popularity fallback for cold-start customers.
4. **Customer Segmentation (`src/models/segmentation.py`):**
   - **Algorithm:** Unsupervised **K-Means clustering** (`FunctionTransformer(np.log1p)` → `StandardScaler()` → `KMeans(n_clusters=4, n_init=20, random_state=42)`). Accounts with `MonetaryValue <= £0` (`40` customers) are isolated into `ClusterID = -1` (`Net-Returned / Zero-Spend`) before `log1p` transformation.

---

## 12. Model Evaluation Results

| Model | Evaluation Split | Key Metrics | Baseline Comparison |
|---|---|---|---|
| **Linear Regression** | Chronological Test (`n = 3,248`) | `MAE = £586.61`, `RMSE = £2,726.22`, `R² = 0.5511` | Beats Mean baseline (`RMSE = £4,069.36`, `R² = -0.0002`) and Median baseline (`MAE = £775.43`) |
| **Logistic Regression (Signed-Log1p)** | Chronological Test (`n = 3,248`, `thr = 0.50`) | `Accuracy = 0.6832`, `Precision = 0.7234`, `Recall = 0.7292`, `F1 = 0.7263`, `ROC-AUC = 0.7471`, `PR-AUC = 0.8178` | Beats Majority baseline (`Accuracy = 0.5764`, `ROC-AUC = 0.5000`, `PR-AUC = 0.5764`) |
| **Item-Based KNN (Cosine)** | Chronological Holdout (`@10`, new items, `n = 2,196`) | `Precision@10 = 0.0668`, `Recall@10 = 0.0355`, `Hit Rate@10 = 38.48%`, `Coverage@10 = 24.72%` (`1,099` SKUs) | Beats Popularity baseline (`Hit Rate@10 = 29.10%`, `Coverage = 1.80%`) and User-Based KNN (`Hit Rate@10 = 35.97%`) |
| **K-Means RFM (`K = 4` + `-1`)** | Positive Spenders (`n = 5,829`) + `40` Isolated | `Silhouette Score = 0.3668`, `Inertia = 4,835.87`, `2D PCA Variance = 95.19%` | Beats `K = 3` (`0.3472`) and `K = 5` (`0.3459`); isolates `4` distinct personas (`74.16%` rev in `High-Value Loyal`) |

---

## 13. Limitations

1. **Guest / Unregistered Transactions:** `228,994` cleaned valid rows (`22.3%` of transactions, `£2.97M` net revenue—including `100%` of Bermuda and Hong Kong orders) have no `Customer ID` and therefore cannot be scored by customer-level ML models or collaborative filtering without fabricating identities.
2. **Wholesale Extreme Outliers in OLS Regression:** `54` extreme high-volume Q4 buyers (`> £5,000`) and sudden reactivations (e.g., Customer `14096`) drive over `93%` of test MSE (`RMSE = £2,726.22` vs. `£718.28` when trimmed to the `99th` target percentile), while `42.6%` of test accounts spend `£0` in Q4.
3. **Partial Calendar Bounds:** `2009` covers December only (`£797.4k`), and `2011` stops on `2011-12-09`, preventing raw annual totals from representing complete 12-month comparisons.
4. **Pure Implicit Co-Purchase Constraints:** Item-Based KNN requires `>= 5` historical customers per SKU and cannot recommend newly launched SKUs with sparse co-purchase edges using content alone.
5. **Linear Classification & Static Segmentation:** Logistic Regression imposes linear log-odds boundaries, and K-Means relies on a single global RFM snapshot (`2011-12-10` reference date) rather than dynamic longitudinal transitions.

---

## 14. Future Improvements

1. **Two-Stage Hurdle Spending Model:** Combine purchase propensity (`P(buy)`) with a regularized or tree-based conditional spend estimator (`Ridge`, `HistGradientBoostingRegressor`, or `log1p`-target regression) to handle zero-inflation and B2B tails.
2. **Gradient-Boosted Purchase Classifiers:** Test `HistGradientBoostingClassifier` / `LightGBM` with inter-purchase interval variance and seasonal indicators, calibrated via isotonic regression.
3. **Hybrid Recommender System:** Blend Item-Based Cosine similarity with TF-IDF product description similarity and recency weighting so new/niche SKUs (`< 5` buyers) receive meaningful cold-start recommendations.
4. **Probabilistic & Longitudinal Segmentation:** Evaluate Gaussian Mixture Models (GMM) for soft membership probabilities and add month-over-month cluster migration tracking.
5. **Scheduled Re-Training & Batch Scoring CLI:** Automate incremental data ingestion, drift checks, and periodic model re-training as new transaction exports arrive.
