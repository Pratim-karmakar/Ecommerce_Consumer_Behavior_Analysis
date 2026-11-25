# Exploratory Data Analysis & Business Insights Report — UCI Online Retail II

- **Observation Period:** `2009-12-01` to `2011-12-09` (`25` monthly snapshots; `2009` covers December only, `2011` stops Dec 9)
- **Baseline Scope:** `1,026,678` validated transactions (`dataset/processed/analytics/`)

## 1. Executive KPI Summary

- **Net Realized Revenue:** `£19,265,931.71` (`£20,434,219.52` gross sales minus `£1,168,287.81` / `5.72%` returns & cancellations)
- **Order & Volume Totals:** `39,988` purchase orders (`8,113` cancelled invoices); `10,728,456` net units (`11,205,017` gross)
- **Average Order Value (AOV):** `£481.79` net (`£511.01` gross)

## 2. Key Analytical Findings by Pillar

### Sales & Seasonality
- Strong Q4 holiday surge peaking in **November 2010 (`£1,416,682.20` net, `2,744` orders)** and **November 2011 (`£1,474,273.68` net, `2,767` orders)**.
- Comparable full-quarter growth: `2011 Q3` (`£2.44M`) rose `+13.8%` over `2010 Q3` (`£2.14M`), though raw `2009` (`£797.4k`, Dec only) and `2011` (`£9.19M`, missing Dec 10–31) are not full 12-month totals.

### Time-of-Week & Hourly Behavior
- **Weekday Dominance:** Weekdays (`Mon–Fri`) generate `£17.48M` (`90.7%` of net revenue, `35,123` orders) vs. weekends at `£1.79M` (`9.3%`, `4,865` orders).
- **Peak Activity Window:** **Thursday** (`£3.96M`, `8,279` orders) and **12:00 noon** (`6,420` orders) lead volume, while **Saturday** has only `30` orders (`£9,803`), reflecting a predominantly B2B/wholesale Monday–Friday and Sunday-afternoon purchasing cadence.

### Product Portfolio & Pareto Concentration
- **Pareto 80/20 Pattern:** `1,038` of `4,711` positive-revenue merchandise SKUs (`22.0%`) generate `80%` of positive merchandise revenue (`£18.93M` net across all `4,746` merchandise SKUs).
- **Top Performers:** `22423` (`REGENCY CAKESTAND 3 TIER`) leads net revenue at `£314,045.02`, while `84077` (`WORLD WAR 2 GLIDERS ASSTD DESIGNS`) leads net quantity at `104,435` units (`ASP £0.23`).
- **Pricing & Return Tail:** Median SKU ASP is `£1.97` (`75th percentile £3.95`), and `35` SKUs show net revenue `<= £0` due to returns exceeding in-window sales.

### Customer Behavior & RFM
- **Repeat Retention & Skew:** `72.47%` (`4,253` of `5,869` purchasing customers) placed `>= 2` orders. Mean spending (`£2,786.77`) is `3.3x` median spending (`£840.24`), and the top `20%` of customers account for `76.88%` of customer-identified spend.
- **Key Accounts & Churn Signals:** Customer `18102` leads net spend (`£570,380.61`, `145` orders, `Recency 1` day), while Customer `14911` leads frequency (`394` orders). High-frequency dormant accounts (e.g., `17850` with `155` orders and `Recency = 373` days) indicate churn-recovery opportunities.

### Geographic Markets
- **Domestic Anchor & Export Hubs:** United Kingdom delivers `85.12%` (`£16,398,388.18`) of net revenue. Top export markets are `EIRE` (`£608,374.51`, `3` customers), `Netherlands` (`£548,349.08`, `AOV £2,405.04`), `Germany` (`£411,904.37`, `107` customers), and `France` (`£321,676.66`, `94` customers).
- **Data Caveats:** `22.3%` of transactions (`£2.97M` net) lack `Customer ID` (affecting 100% of `Bermuda` and `Hong Kong`), and `52` return-only `Customer ID`s (`-£60.9k`) lack in-window purchases.

## 3. Generated Visualizations (`visualizations/`)

- `01_sales_monthly_trends.html` (4717.4 KB)
- `01_sales_yearly_quarterly.png` (90.5 KB)
- `02_time_heatmap_day_hour.png` (58.5 KB)
- `02_time_patterns_dow_hour.png` (183.5 KB)
- `03_product_pareto_and_asp.png` (111.8 KB)
- `03_product_top100_interactive.html` (4723.6 KB)
- `03_product_top10_rev_qty.png` (159.7 KB)
- `04_customer_rfm_interactive.html` (5071.4 KB)
- `04_customer_top_and_rfm.png` (307.5 KB)
- `05_country_performance.png` (112.5 KB)
- `05_country_treemap_interactive.html` (4719.3 KB)
