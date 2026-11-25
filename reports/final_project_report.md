# E-Commerce Customer Analytics — Final Project Report

**Dataset:** UCI Online Retail II (`2009-12-01` to `2011-12-09`)  
**Tech Stack:** Python `3.14`, Pandas, NumPy, Scikit-Learn, Plotly, Matplotlib, Seaborn, Joblib, Streamlit  

---

## 1. Abstract

This project presents a complete, reproducible Business Intelligence (BI) and Machine Learning (ML) pipeline built exclusively on real transactional records from the combined **UCI Online Retail II** dataset (`1,067,371` raw rows spanning December 2009 to December 2011). After performing a non-destructive read-only quality audit and rule-based preprocessing that removed `40,693` duplicate and operational noise records, `1,026,678` validated commercial rows (`£19,265,931.71` net revenue across `39,988` purchase orders) were structured into five analytical tables covering transactions, customers (`5,869` purchasing buyers), merchandise products (`4,746` SKUs), countries (`43` markets), and monthly aggregates (`25` snapshots). Four machine learning models were designed with strict chronological leakage prevention: **Linear Regression** for 90-day future spending (`Test R² = 0.5511`, `MAE = £586.61`), **Logistic Regression** for 90-day repeat purchase propensity (`ROC-AUC = 0.7471`, `PR-AUC = 0.8178`), **Item-Based Cosine KNN** for product recommendations (`Hit Rate@10 = 38.48%`, `Coverage = 24.72%`), and **K-Means clustering** over `log1p`-scaled RFM metrics (`K = 4` positive clusters with `Silhouette = 0.3668` plus isolated `ClusterID = -1` for zero/net-return accounts). All analytics and serialized `.joblib` pipelines are integrated into a verified 11-page Streamlit web dashboard.

---

## 2. Introduction

E-commerce platforms generate rich event logs containing customer purchases, cancellations, returns, postage fees, manual ledger entries, and unregistered guest checkouts. Turning these raw ledgers into reliable executive reporting and forward-looking predictive systems requires systematic data validation before any visualization or model fitting begins. When sheet overlaps, returns, or future target periods are left mixed inside training tables, dashboards double-count revenue and predictive models suffer severe data leakage.

This project demonstrates an end-to-end engineering workflow structured across notebooks (`notebooks/`), modular tested Python packages (`src/`), serialized models (`models/`), derived analytical tables (`dataset/processed/analytics/`), audit reports (`reports/`), and an interactive Streamlit dashboard (`app/`).

---

## 3. Problem Statement

The project addresses five analytical and predictive challenges observed in real retail transaction logs:
1. **Data Quality & Revenue Reconciliation:** Raw combined UCI Online Retail II exports contain overlapping date windows (`2010-12-01` to `2010-12-09`), negative quantities split between genuine customer cancellations (`C`-invoices) and internal warehouse write-offs (`Price == 0`), accounting bad-debt entries (`A`-invoices), and `22.77%` missing `Customer ID` rows.
2. **Executive BI Visibility:** Leadership requires clear breakdowns of net revenue (net of returns), order cadence across weekdays/hours, SKU-level Pareto concentration, customer loyalty skew, and domestic vs. export market mix.
3. **Future Customer Spending Prediction:** Forecasting net spending over a fixed 90-day future horizon using only historical pre-cutoff features.
4. **Repeat Purchase Classification:** Estimating the probability that an eligible customer purchases again in the next 90 days while supporting precision/recall threshold trade-offs for targeted marketing.
5. **Product Recommendation & Behavioral Segmentation:** Generating personalized SKU recommendations from observed co-purchases and discovering interpretable RFM customer segments without synthetic labels.

---

## 4. Objectives

- Preserve the raw CSV untouched (`dataset/Ecommerce.csv`) and verify cryptographic immutability before and after execution.
- Audit all schema properties, missingness, duplicates, cancellations, prices, quantities, and date ranges in a dedicated inspection notebook and CSV/Markdown report.
- Clean the dataset transparently using documented business rules, keeping customer returns/cancellations and unregistered checkouts distinct rather than blindly deleting valid sales or fabricating IDs.
- Engineer multi-granularity analytical datasets and evaluate KPIs in exploratory visualizations.
- Train, tune, and benchmark four non-leaking ML models against explicit baselines, serialize all inference pipelines, and test edge cases (`Missing model`, `Invalid Customer ID`, `Unknown StockCode`, `Missing features`, negative monetary inputs).
- Deliver an intuitive, responsive 11-page Streamlit application backed directly by real datasets and saved models.

---

## 5. Dataset Description

- **Origin & Integrity:** Combined `2009–2010` and `2010–2011` UCI Online Retail II CSV stored read-only at `dataset/Ecommerce.csv` (`90.46 MB`, SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`).
- **Dimensions:** `1,067,371` rows × `8` columns.
- **Original Columns:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.
- **Observation Window:** `2009-12-01 07:45:00` to `2011-12-09 12:50:00` (`0` unparseable dates).
- **Raw Data Audit Summary (`reports/data_quality_report.csv`):**
  - `Missing Customer ID records`: `243,007` (`22.77%`)
  - `Missing Description records`: `4,382` (`0.41%` — all `4,382` also lacked `Customer ID`)
  - `Exact duplicate rows`: `34,335` (`3.22%`), including `22,844` rows inside the Dec 1–9, 2010 two-sheet overlap
  - `Cancelled invoice records (C-prefix)`: `19,494` rows across `8,292` unique invoices (`1` anomalous positive quantity row `C496350`)
  - `Non-C negative quantity records`: `3,457` (warehouse damages/write-offs; `100%` had `Price == 0` and missing `Customer ID`)
  - `Price anomalies`: `6,202` zero-price rows and `5` negative-price bad-debt adjustment rows (`A`-prefixed invoices)
  - `Other anomalies`: `17` `TEST*` rows, `6,093` non-standard `StockCode` rows, `823` lowercase status-note descriptions, and `756` `Country == 'Unspecified'` rows.
  - `Raw Entity Counts`: `53,628` unique invoices, `5,942` distinct `Customer ID`s, `5,305` raw `StockCode`s (`5,131` stripped/uppercased), and `43` countries.

---

## 6. Data Cleaning

Implemented in `src/data/cleaning.py` and executed in `notebooks/02_Data_Cleaning.ipynb` on a copy of the raw data:

1. **Column Standardization & Formatting:** Columns were standardized to `snake_case` (`invoice`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`, `customer_id`, `country`); `stock_code` was trimmed and uppercased, `invoice_date` parsed to `datetime64`, and `customer_id` kept as nullable integer (`Int64`).
2. **Step-by-Step Exclusion Trail (`40,693` rows removed, `3.81%`):**

| Cleaning Step | Rows Before | Removed | Rows After | Rationale |
|---|---:|---:|---:|---|
| Drop exact duplicates | 1,067,371 | 34,335 | 1,033,036 | Eliminates Dec 1–9, 2010 sheet-overlap duplicates (`22,844`) and exact copies |
| Remove `A`-prefixed bad-debt adjustments | 1,033,036 | 6 | 1,033,030 | Internal ledger entries (including all `5` negative-price rows) |
| Remove `TEST*` codes | 1,033,030 | 17 | 1,033,013 | Internal test items (`TEST001`, `TEST002`) |
| Remove non-`C` negative-quantity rows | 1,033,013 | 3,393 | 1,029,620 | Warehouse write-offs (`Price == 0`, missing `Customer ID`) |
| Remove `Price <= 0` records | 1,029,620 | 2,618 | 1,027,002 | Unpriced entries; resolves 100% of remaining missing `Description`s |
| Remove anomalous positive `C`-row | 1,027,002 | 1 | 1,027,001 | Contradictory positive quantity on `C496350` |
| Remove internal fee/sample codes | 1,027,001 | 323 | 1,026,678 | Non-customer operational codes (`AMAZONFEE`, `BANK CHARGES`, `ADJUST`, `ADJUST2`, `CRUK`, `S`) |

3. **Resulting Partitioned Datasets (`dataset/processed/`):**
   - `cleaned_online_retail.csv`: **`1,026,678`** rows (`£19,265,931.71` net, `0` non-ID nulls, `0` exact duplicates)
   - `sales_transactions.csv`: **`1,007,823`** normal purchase rows (`£20,434,219.52` gross)
   - `returns_and_cancellations.csv`: **`18,855`** `C`-prefixed returns/cancellations (`-£1,168,287.81`)
   - Guest rows (`has_customer_id == False`, `228,994` rows) remain flagged cleanly for store-level totals while customer modeling restricts to `has_customer_id == True` (`797,684` rows across `5,921` unique IDs: `5,869` purchasing accounts + `52` return-only IDs).

---

## 7. Feature Engineering

Implemented in `src/features/engineering.py` (`notebooks/03_Feature_Engineering.ipynb`) under `dataset/processed/analytics/`:

1. **`transaction_features.csv` (`1,026,678` × `25` cols):** Adds `TotalAmount` (`quantity × price`), `Year`, `Month`, `MonthName`, `Quarter`, `Day`, `DayOfWeek`, `DayName`, `Hour`, `IsWeekend`, and `InvoiceYearMonth` alongside cleaning flags.
2. **`customer_analytics.csv` (`5,869` × `16` cols):** Uses fixed reference date **`2011-12-10`** (`max_date + 1 day` = `2011-12-09` + 1d) to compute `TotalSpending` (net), `TotalOrders`, `TotalQuantity`, `UniqueProductsPurchased` (merchandise SKUs), `AverageOrderValue`, `FirstPurchaseDate`, `LastPurchaseDate`, `Recency` (`days`), `Frequency`, `MonetaryValue`, plus audit breakdowns (`GrossSpending`, `ReturnedAmount`, `CancelledOrders`, `GrossQuantity`, `ReturnedQuantity`). Overall customer medians: `Recency = 96.0 days`, `Frequency = 3.0`, `MonetaryValue = £840.24`, `AverageOrderValue = £270.76`, `UniqueProductsPurchased = 45.0` (means: `R=201.24`, `F=6.29`, `M=£2,786.77`, `AOV=£347.16`, `UniqueProducts=82.11`).
3. **`product_analytics.csv` (`4,746` × `13` cols):** Merchandise-only table (excluding service/fee codes `POST`, `DOT`, `C2`, `D`, `M`; net revenue reconciled identically to `£18,929,085.68`), tracking `TotalQuantitySold`, `TotalRevenue`, `TotalInvoices`, `UniqueCustomers`, `AverageSellingPrice`, and gross/return splits.
4. **`country_analytics.csv` (`43` × `9` cols) & `monthly_sales.csv` (`25` × `11` cols):** Market and monthly revenue, order counts, customer counts, and AOV tables reconciling exactly to `£19,265,931.71` total net revenue.

---

## 8. Exploratory Data Analysis

Key findings verified via `src/analysis/eda_metrics.py` (`notebooks/04_Exploratory_Data_Analysis.ipynb`, `reports/eda_business_insights.md`):

- **Overall KPIs:** Net realized revenue is **`£19,265,931.71`** (`£20,434,219.52` gross minus `£1,168,287.81` / `5.72%` returns/cancellations) across **`39,988`** purchase orders (`8,113` cancelled invoices) and **`10,728,456`** net units (`£481.79` net AOV vs. `£511.01` gross AOV).
- **Monthly Seasonality:** Strong Q4 peaks occur in **November 2010** (`£1,416,682.20` net, `2,744` orders) and **November 2011** (`£1,474,273.68` net, `2,767` orders). Comparable full-quarter growth rose `+13.8%` from `2010 Q3` (`£2.14M`) to `2011 Q3` (`£2.44M`). (`2009` covers December only at `£797.4k`; `2011` stops at `2011-12-09`.)
- **Time-of-Week & Hour:** Weekdays (`Mon–Fri`) account for `£17.48M` (**`90.7%`** of net revenue, `35,123` orders) vs. weekends (`£1.79M`, `9.3%`, `4,865` orders). **Thursday** (`£3.96M`, `8,279` orders) and **12:00 noon** (`6,420` orders) lead volume, while **Saturday** sees only `30` orders (`£9,803`)—confirming a predominantly B2B/wholesale Mon–Fri cadence.
- **Product Concentration (Pareto):** **`1,038` of `4,711`** positive-revenue merchandise SKUs (`22.0%`) drive **`80%`** of positive merchandise revenue. Top revenue SKU is `22423` (`REGENCY CAKESTAND 3 TIER`, `£314,045.02` net), and top volume SKU is `84077` (`WORLD WAR 2 GLIDERS ASSTD DESIGNS`, `104,435` net units). Median SKU ASP is `£1.97`.
- **Customer Skew:** **`72.47%`** (`4,253` of `5,869`) buyers made `>= 2` orders. Mean net spending (`£2,786.77`) is **`3.3×`** median (`£840.24`), and the top `20%` of buyers contribute `76.88%` of customer-identified spend (top spender: `18102` at `£570,380.61`; highest frequency: `14911` at `394` orders).
- **Geography:** United Kingdom accounts for **`85.12%`** (`£16,398,388.18`) of net revenue. Top export markets are `EIRE` (`£608,374.51`, `3` buyers), `Netherlands` (`£548,349.08`, `AOV £2,405.04`), `Germany` (`£411,904.37`, `107` buyers), and `France` (`£321,676.66`, `94` buyers).

---

## 9. Machine Learning Methodology

All supervised and recommender models enforce chronological train/test temporal splits to guarantee zero data leakage:

1. **Chronological Feature/Target Framing (Models 1 & 2):**
   - **Train Cohort (`n = 3,270`):** Observation `2009-12-01` to `2010-09-10` (`283` days; all customers with at least 1 valid purchase before cutoff) → Future target `2010-09-11` to `2010-12-09` (`90` Q4 days).
   - **Test Cohort (`n = 3,248`):** Observation `2010-12-10` to `2011-09-10` (`275` days) → Future target `2011-09-11` to `2011-12-09` (`90` Q4 days).
   - **Eight Strict Historical Predictors:** `hist_spending`, `hist_frequency`, `hist_quantity`, `hist_aov`, `hist_unique_products`, `hist_recency`, `hist_tenure`, `hist_cancelled_orders`.
   - **Linear Regression Pipeline:** `SimpleImputer(median)` → `StandardScaler()` → `LinearRegression()`.
   - **Logistic Regression Pipeline:** `SimpleImputer(median)` → `signed_log1p` (`np.sign(x) * np.log1p(np.abs(x))`) → `StandardScaler()` → `LogisticRegression(max_iter=1000)` predicting binary Q4 purchase (`58.35%` positive in Train; `57.64%` positive in Test). Non-purchasers in target window with full 90-day observability are validly labeled `0`.
2. **Product Recommendation (`ItemKNNRecommender`):**
   - Aggregates positive merchandise customer–product quantities (`>= 1` buyer purchase, excludes cancellations and fee codes) dampened via `log1p(quantity)` into a `scipy.sparse.csr_matrix` (`5,851` customers × `4,200` SKUs with `>= 5` distinct customers).
   - Fits Cosine `NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=25)` over item vectors; evaluated chronologically on `Train (< 2011-09-11)` vs. `Holdout (>= 2011-09-11)` (`n = 2,196` new-item customers / `2,272` repeat-included). Falls back cleanly to global bestselling rank on cold-start customers.
3. **Customer Segmentation (`K-Means RFM`):**
   - Evaluates `K = 2..9` on `Recency`, `Frequency`, and `MonetaryValue` transformed with `np.log1p` and `StandardScaler`.
   - Excludes `40` accounts with `MonetaryValue <= £0` before clustering and assigns them deterministically to `ClusterID = -1` (`Net-Return Accounts`) so log-domain math never receives invalid inputs.

---

## 10. Model Evaluation

### 10.1 Linear Regression — 90-Day Future Spending (`reports/linear_regression_results.csv`)

| Model / Evaluation Scope | Split (`n`) | MAE (£) | RMSE (£) | R² |
|---|---|---:|---:|---:|
| Mean Baseline | Test (`3,248`) | 935.35 | 4,069.36 | -0.0002 |
| Median Baseline | Test (`3,248`) | 775.43 | 4,109.58 | -0.0201 |
| Simple OLS (`hist_spending` only) | Test (`3,248`) | 596.71 | 2,732.93 | 0.5489 |
| **Linear Regression (Full Pipeline)** | **Test (`3,248`)** | **586.61** | **2,726.22** | **0.5511** |
| Linear Regression (Full Pipeline) | Train (`3,270`) | 565.76 | 1,565.50 | 0.7132 |
| Full Pipeline (`<= 99th` pct target) | Test trimmed (`3,215`) | 429.23 | 718.28 | 0.3828 |

### 10.2 Logistic Regression — 90-Day Repeat Purchase (`reports/logistic_regression_results.csv`)

| Model / Operating Point | Threshold | Accuracy | Balanced Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority Baseline | 0.50 | 0.5764 | 0.5000 | 0.5764 | 1.0000 | 0.7312 | 0.5000 | 0.5764 |
| Raw Scaled LogReg | 0.50 | 0.6881 | 0.6846 | 0.7398 | 0.7078 | 0.7235 | 0.7475 | 0.8149 |
| **Signed-Log1p LogReg (Default)** | **0.50** | **0.6832** | **0.6749** | **0.7234** | **0.7292** | **0.7263** | **0.7471** | **0.8178** |
| Balanced Precision (`CV-tuned`) | 0.56 | 0.6786 | 0.6824 | 0.7534 | 0.6576 | 0.7022 | 0.7471 | 0.8178 |
| High-Recall / Max-F1 (`CV-tuned`) | 0.34 | 0.6155 | 0.5548 | 0.6059 | 0.9519 | 0.7405 | 0.7471 | 0.8178 |

### 10.3 Product Recommendation — Chronological Holdout (`reports/knn_recommender_results.csv`)

| Model | K | Exclude Seen | Eval Users | Precision@K | Recall@K | Hit Rate@K | Coverage (%) | Unique SKUs |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Popularity Baseline | 10 | True | 2,196 | 0.0412 | 0.0168 | 0.2910 | 1.80% | 80 |
| User-Based KNN (Cosine) | 10 | True | 2,196 | 0.0593 | 0.0304 | 0.3597 | 23.28% | 1,035 |
| **Item-Based KNN (Cosine)** | **5** | **True** | **2,196** | **0.0721** | **0.0197** | **0.2600** | **17.46%** | **776** |
| **Item-Based KNN (Cosine)** | **10** | **True** | **2,196** | **0.0668** | **0.0355** | **0.3848** | **24.72%** | **1,099** |
| Item-Based KNN (Incl. Repeats) | 10 | False | 2,272 | 0.1952 | 0.0681 | 0.6461 | 26.68% | 1,186 |

### 10.4 K-Means Customer Segmentation (`reports/kmeans_cluster_summary.csv`)

`K = 4` achieved a Silhouette Score of **`0.3668`** (`Inertia = 4,835.87`), outperforming `K = 3` (`0.3472`) and `K = 5` (`0.3459`), with `2D PCA` capturing **`95.19%`** of standardized feature variance (`customer_segments.csv` saved for all `5,869` accounts):

| ClusterID | Segment Name | Customers | Share (%) | Median R (d) | Median F | Median M (£) | Revenue (£) | Rev Share (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | VIP Accounts | 1,193 | 20.33% | 17.0 | 13.0 | 4,888.75 | 12,128,760.30 | 74.16% |
| `1` | Lapsing Regulars | 1,446 | 24.64% | 185.0 | 5.0 | 1,422.08 | 2,618,355.68 | 16.01% |
| `2` | Active Growth | 1,256 | 21.40% | 24.5 | 3.0 | 709.28 | 1,036,769.16 | 6.34% |
| `3` | Dormant Buyers | 1,934 | 32.95% | 403.0 | 1.0 | 265.41 | 603,196.45 | 3.69% |
| `-1` | Net-Return Accounts | 40 | 0.68% | 426.5 | 1.0 | 0.00 | -31,515.23 | -0.19% |

---

## 11. Dashboard Implementation

Built cleanly in `app/` without duplicating notebook logic:
- **Navigation & Styling:** `app/app.py` configures wide layout and groups `11` subpages (`app/pages/p01_home.py`–`p11_data_quality.py`) via `st.navigation`, styled consistently with `.streamlit/config.toml` (`base="light"`), `app/utils/theme.py`, and reusable components in `app/components/ui.py` (KPI cards, category pills, beginner-friendly callout banners, and UTF-8 CSV download buttons).
- **Performance Caching:** `app/utils/data_loader.py` uses `@st.cache_data` (compact dtypes for `1,026,678` rows) and `@st.cache_resource` (loading `.joblib` models once per session).
- **Live Verification:** Automated `AppTest` checks (`tests/test_streamlit_app.py`) and live Google Chrome verification across all `11` pages confirmed zero exceptions, zero browser console errors/warnings, responsive filters, working What-If simulators, interactive classification threshold tuning, KNN cold-start / invalid-ID / unknown-SKU error handling, RFM cluster scoring, and clean theme contrast.

---

## 12. Results and Discussion

- **Revenue Concentration Requires Account-Tiered Strategy:** Just `1,193` `VIP Accounts` (`20.33%` of buyers) generate `74.16%` (`£12.13M`) of customer-attributed net revenue, while `1,446` `Lapsing Regulars` (`16.01%` of spend, median Recency `185` days) represent the highest-leverage win-back audience.
- **Chronological Models Beat Baselines Cleanly:** All three supervised/recommender systems outperform non-trivial historical baselines (`+0.551` R² gain over mean predictor, `+0.247` ROC-AUC gain over majority class, and `+9.38` percentage points Hit Rate@10 + `13.7×` higher SKU coverage vs. Popularity baseline).
- **Unified Inference API Readiness:** Every model reloads cleanly from disk through `src/models/inference.py` and gracefully handles invalid types, missing model files (`ModelLoadError`), unseen users (popularity fallback), unknown SKUs (`UnknownProductError`), missing numeric features (saved `SimpleImputer`), and negative RFM values (`ClusterID = -1`).

---

## 13. Limitations

1. **Missing Customer Identifiers (`22.3%` post-cleaning transactions):** Guest orders cannot be tracked across visits for RFM segmentation, spending forecasts, propensity scoring, or personalized collaborative filtering.
2. **Wholesale Outlier Sensitivity in Linear Regression:** Extreme B2B purchasers (`54` accounts above `£5,000` plus unpredictable onboarding surges) create right-tailed residuals (`RMSE = £2,726.22` vs. `£718.28` on the 99th-percentile trimmed cohort), alongside `42.6%` zero-spend outcomes in Q4.
3. **Partial Boundary Years (`2009` and `2011`):** Because the archive covers `2009-12-01` to `2011-12-09`, full-year comparisons should rely on complete quarters (e.g., `2010 Q1–Q3` vs. `2011 Q1–Q3`) or monthly rates rather than raw annual sums.
4. **Pure Co-Purchase Recommender Scope:** `ItemKNNRecommender` excludes `546` sparse catalog items purchased by `< 5` training customers and uses implicit purchase quantity rather than unavailable star ratings or clickstreams.
5. **Static Snapshot & Linear Boundaries:** Logistic Regression captures linear log-odds relationships, and K-Means reflects a fixed `2011-12-10` reference snapshot.

---

## 14. Conclusion

The E-Commerce Customer Analytics project delivers a complete, transparent, and fully tested analytics pipeline from raw audit to interactive web deployment. By preserving `dataset/Ecommerce.csv` immutably, handling cancellations and missing IDs honestly, eliminating temporal leakage across every machine learning benchmark, and consolidating logic into modular Python code (`src/`) plus an 11-page Streamlit application (`app/`), the project satisfies academic rigor and practical BI usability with every metric and finding grounded strictly in observed data.

---

## 15. Future Scope

- Implement a **two-stage Hurdle model** (`P(purchase) × E[spend | purchase]`) and non-linear gradient boosting (`HistGradientBoosting` / `LightGBM`) to better handle zero-inflation and B2B wholesale outliers.
- Add **probability calibration** (isotonic / Platt scaling) for marketing cost-matrix optimization under Logistic Regression.
- Expand collaborative filtering into a **hybrid content + CF recommender** using TF-IDF similarity over `Description` to cover newly launched or low-support SKUs (`< 5` buyers).
- Replace static K-Means with **Gaussian Mixture Models (GMM)** and track month-over-month **RFM migration transitions** (`Recent → Loyal` vs. `Loyal → At-Risk`).
