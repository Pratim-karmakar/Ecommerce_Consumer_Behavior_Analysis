# Unified Model Evaluation & Integration Summary — E-Commerce Customer Analytics

**Project:** E-Commerce Customer Analytics  
**Scope:** Independent verification, evaluation, and Streamlit-ready integration of all four production ML components trained strictly on verified UCI Online Retail II transaction data (`2009-12-01` to `2011-12-09`).

---

## 1. Model Verification Checklist

| Component | Saved Artifact | Preprocessing Included | Leakage Prevention & Evaluation Design | Independent Load & Inference Check |
|---|---|---|---|---|
| **1. Linear Regression** (Future Spending) | `models/linear_regression_pipeline.joblib` (`2.06` KB) | `SimpleImputer(median)` + `StandardScaler` | Chronological `275d -> 90d` Q4 split (`Train <= 2010-12-09`, `Test >= 2010-12-10`, zero overlap) | Passed (`predict_future_spending`) |
| **2. Logistic Regression** (Purchase Prob.) | `models/logistic_regression_pipeline.joblib` (`2.55` KB) | `SimpleImputer(median)` + `signed_log1p` + `StandardScaler` | Chronological non-overlapping split + complete 90-day future observability + 5-fold CV threshold tuning strictly on Train | Passed (`predict_purchase_probability`) |
| **3. Item-Based KNN** (Recommendations) | `models/knn_recommender.joblib` (`7.92` MB) | Sparse CSR matrix + `log1p(quantity)` + `min_item_customers=5` + Cosine KNN | Evaluated on chronological holdout (`Train < 2011-09-11` vs `Holdout >= 2011-09-11`); cold-start fallback verified | Passed (`recommend_products`, `find_similar_products`) |
| **4. K-Means RFM** (Segmentation) | `models/kmeans_pipeline.joblib` (`24.28` KB) | `FunctionTransformer(np.log1p)` + `StandardScaler` | Evaluated `K = 2..9` via Elbow & Silhouette; `40` `MonetaryValue <= 0` accounts isolated into `ClusterID = -1` | Passed (`assign_customer_segment`, 100% match with `customer_segments.csv`) |

---

## 2. Evaluation Summary by Model (No Cross-Metric Comparisons)

### 2.1 Linear Regression — 90-Day Future Spending (`reports/linear_regression_results.csv`)

- **Target & Cohort:** Predicts net `future_spending` over 90 Q4 days (`Train n = 3,270`; `Test n = 3,248`).
- **Test Performance vs. Baselines:**
  - **Mean Baseline:** `MAE = £935.35`, `RMSE = £4,069.36`, `R² = -0.0002`
  - **Median Baseline:** `MAE = £775.43`, `RMSE = £4,109.58`, `R² = -0.0201`
  - **Simple 1-Feature OLS (`hist_spending`):** `MAE = £596.71`, `RMSE = £2,732.93`, `R² = 0.5489`
  - **Full Linear Regression Pipeline:** **`MAE = £586.61`**, **`RMSE = £2,726.22`**, **`R² = 0.5511`** (`Train R² = 0.7132`, `RMSE = £1,565.50`)
  - **Outlier Sensitivity (`<= 99th percentile` Test, `n = 3,215`):** `MAE = £429.23`, `RMSE = £718.28`, `R² = 0.3828`
- **Weaknesses & Limitations:**
  - Extreme B2B wholesale accounts (`> £5k`, `54` customers, `MAE = £10,956.10`) and sudden account ramp-ups (e.g., Customer `14096` jumping from `£134.23` historically to `£64,920.11` in Q4) drive over `93%` of squared error.
  - `42.6%` of Test customers spend `£0` in the future 90 days, creating zero-inflation and mild heteroscedasticity.
- **Improvements:** Two-stage hurdle architecture (`P(buy) × E[spend | buy]`), regularized/tree-based regressors (`Ridge`, `HistGradientBoostingRegressor`), or `log1p` target transformation.

### 2.2 Logistic Regression — 90-Day Future Purchase Classification (`reports/logistic_regression_results.csv`)

- **Target & Class Balance:** Binary purchase indicator (`58.35%` positive in Train `n = 3,270`; `57.64%` positive in Test `n = 3,248`).
- **Test Performance vs. Baseline:**
  - **Majority Baseline:** `Accuracy = 0.5764`, `Balanced Acc = 0.5000`, `Precision = 0.5764`, `Recall = 1.0000`, `F1 = 0.7312`, `ROC-AUC = 0.5000`, `PR-AUC = 0.5764`
  - **Raw Scaled LogReg (`thr = 0.50`):** `Accuracy = 0.6881`, `Balanced Acc = 0.6846`, `Precision = 0.7398`, `Recall = 0.7078`, `F1 = 0.7235`, `ROC-AUC = 0.7475`, `PR-AUC = 0.8149`
  - **Signed-Log1p LogReg (`thr = 0.50`, Default):** **`Accuracy = 0.6832`**, **`Balanced Acc = 0.6749`**, **`Precision = 0.7234`**, **`Recall = 0.7292`**, **`F1 = 0.7263`**, **`ROC-AUC = 0.7471`**, **`PR-AUC = 0.8178`** (`TN=854, FP=522, FN=507, TP=1365`)
  - **CV-Tuned Balanced Threshold (`thr = 0.56`):** `Accuracy = 0.6786`, `Balanced Acc = 0.6824`, `Precision = 0.7534`, `Recall = 0.6576`, `F1 = 0.7022` (`TN=973, FP=403, FN=641, TP=1231`)
  - **CV-Tuned Max-F1 Threshold (`thr = 0.34`):** `Accuracy = 0.6155`, `Balanced Acc = 0.5548`, `Precision = 0.6059`, `Recall = 0.9519`, `F1 = 0.7405` (`TN=217, FP=1159, FN=90, TP=1782`)
- **Weaknesses & Limitations:** Linear log-odds boundaries cannot capture non-linear seasonality or sudden reactivation shocks; `22.3%` guest checkouts lack `Customer ID` and cannot be scored.
- **Improvements:** Non-linear gradient boosting (`LightGBM`/`HistGradientBoostingClassifier`), purchase velocity/gap variance features, and isotonic probability calibration.

### 2.3 Item-Based KNN — Product Recommendation (`reports/knn_recommender_results.csv`)

- **Setup:** Cosine similarity over `log1p(total_quantity)` sparse matrix (`min_item_customers=5`, `n_neighbors=25`, merchandise SKUs only).
- **Chronological Holdout (`Train < 2011-09-11` vs `Holdout >= 2011-09-11`):**
  - **Popularity Baseline (`@5` / `@10`, new items, `n = 2,196`):** `Precision = 0.0476 / 0.0412`, `Recall = 0.0106 / 0.0168`, `Hit Rate = 0.1990 / 0.2910`, `Coverage = 1.03% / 1.80%`
  - **User-Based KNN (`@5` / `@10`, new items, `n = 2,196`):** `Precision = 0.0703 / 0.0593`, `Recall = 0.0188 / 0.0304`, `Hit Rate = 0.2609 / 0.3597`, `Coverage = 17.14% / 23.28%`
  - **Item-Based KNN (`@5` / `@10`, new items, `n = 2,196`):** **`Precision = 0.0721 / 0.0668`**, **`Recall = 0.0197 / 0.0355`**, **`Hit Rate = 0.2600 / 0.3848`**, **`Coverage = 17.46% / 24.72%`** (`776 / 1,099` SKUs)
  - **Item-Based KNN (Including Repeats, `@5` / `@10`, `n = 2,272`):** `Precision = 0.2123 / 0.1952`, `Recall = 0.0369 / 0.0681`, `Hit Rate = 0.5163 / 0.6461`, `Coverage = 18.97% / 26.68%`
- **Weaknesses & Limitations:** Pure co-purchase CF cannot recommend newly launched SKUs with `< 5` historical buyers (`cold-start` items) and relies on implicit binary/quantity signals rather than explicit ratings.
- **Improvements:** Hybrid TF-IDF product description similarity + co-purchase blending for new SKUs, and recency-weighted interaction decay.

### 2.4 K-Means — RFM Customer Segmentation (`reports/kmeans_cluster_summary.csv`)

- **Setup:** `K = 4` clusters selected via Elbow inflection (`Inertia = 4,835.87`) and local Silhouette peak (**`0.3668`**, beating `K = 3` at `0.3472` and `K = 5` at `0.3459`) on `np.log1p`-scaled RFM (`n = 5,829` positive spenders; 2D PCA explains **`95.19%`** variance). `40` accounts with `MonetaryValue <= £0` are isolated as `ClusterID = -1`.
- **Cluster Profiles (`n = 5,869` total):**
  - **`0` — VIP Accounts:** `1,193` (`20.33%`) | Medians: `R=17d, F=13, M=£4,888.75` | **`74.16%`** of net customer revenue (`£12.13M`)
  - **`1` — Lapsing Regulars:** `1,446` (`24.64%`) | Medians: `R=185d, F=5, M=£1,422.08` | **`16.01%`** of net customer revenue (`£2.62M`)
  - **`2` — Active Growth:** `1,256` (`21.40%`) | Medians: `R=24.5d, F=3, M=£709.28` | `6.34%` of net customer revenue (`£1.04M`)
  - **`3` — Dormant Buyers:** `1,934` (`32.95%`) | Medians: `R=403d, F=1, M=£265.41` | `3.69%` of net customer revenue (`£0.60M`)
  - **`-1` — Net-Return Accounts:** `40` (`0.68%`) | Medians: `R=426.5d, F=1, M=£0.00` | `-0.19%` of net customer revenue (`-£31.5k`)
- **Weaknesses & Limitations:** Hard spherical boundaries in log-RFM space split borderline customers smoothly rather than capturing multi-modal B2B vs. consumer sub-behaviors; static snapshot requires periodic re-scoring.
- **Improvements:** Gaussian Mixture Models (GMM) for soft segment probabilities and tracking month-over-month cluster migration.

---

## 3. Reusable Inference & Error-Handling API (`src/models/inference.py`)

All Streamlit pages and scripts import unified, validated entry points from `src.models`:

- **Loaders:** `load_linear_regression_model()`, `load_logistic_regression_model()`, `load_knn_recommender()`, `load_kmeans_model()` (raise `ModelLoadError` if missing/corrupt).
- **Predictors:**
  - `predict_future_spending(features)` — imputes missing historical inputs via saved `SimpleImputer`, validates schema/infinity.
  - `predict_purchase_probability(features, threshold=0.50)` — returns probability, class prediction, and propensity tier; validates `threshold` in `[0, 1]`.
  - `recommend_products(customer_id, top_n=10, exclude_purchased=True)` — validates positive integer ID (`validate_customer_id`), returns cold-start popularity fallback transparently for unseen IDs.
  - `find_similar_products(stock_code, top_n=10, strict=True)` — raises `UnknownProductError` (or empty DataFrame when `strict=False`) on unknown SKUs.
  - `assign_customer_segment(rfm_features)` — validates finite non-null RFM (`Recency >= 0`, `Frequency >= 0`), routes `MonetaryValue <= 0` safely to `ClusterID = -1`, and maps positive rows deterministically via recovered centroids to match `customer_segments.csv`.
