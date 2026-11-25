"""Leakage-free temporal feature generation, training, and evaluation for future spending prediction."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: List[str] = [
    "hist_spending",
    "hist_frequency",
    "hist_quantity",
    "hist_aov",
    "hist_unique_products",
    "hist_recency",
    "hist_tenure",
    "hist_cancelled_orders",
]

TARGET_COLUMN = "future_spending"


def build_temporal_cohort(
    tx_df: pd.DataFrame,
    obs_start: str,
    obs_end: str,
    target_start: str,
    target_end: str,
) -> pd.DataFrame:
    """
    Build one customer-level cohort from strictly non-overlapping observation and target windows.

    All boundary comparisons treat `*_start` as inclusive and `*_end` as exclusive, ensuring
    zero future transactions can enter the historical feature calculations.
    """
    if pd.Timestamp(obs_end) > pd.Timestamp(target_start):
        raise ValueError("Observation end date cannot exceed target start date (would cause leakage).")

    identified = tx_df[tx_df["has_customer_id"]].copy()
    obs_mask = (identified["invoice_date"] >= obs_start) & (identified["invoice_date"] < obs_end)
    tgt_mask = (identified["invoice_date"] >= target_start) & (identified["invoice_date"] < target_end)

    obs = identified[obs_mask]
    tgt = identified[tgt_mask]

    obs_sales = obs[~obs["is_cancelled"]]
    obs_returns = obs[obs["is_cancelled"]]

    features = obs_sales.groupby("customer_id", as_index=False).agg(
        hist_gross_spending=("TotalAmount", "sum"),
        hist_frequency=("invoice", "nunique"),
        hist_gross_qty=("quantity", "sum"),
        hist_unique_products=("stock_code", "nunique"),
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
    )

    ret_agg = obs_returns.groupby("customer_id", as_index=False).agg(
        hist_return_amount=("TotalAmount", "sum"),
        hist_return_qty=("quantity", "sum"),
        hist_cancelled_orders=("invoice", "nunique"),
    )

    features = features.merge(ret_agg, on="customer_id", how="left")
    features["hist_return_amount"] = features["hist_return_amount"].fillna(0.0)
    features["hist_return_qty"] = features["hist_return_qty"].fillna(0).astype("int64")
    features["hist_cancelled_orders"] = features["hist_cancelled_orders"].fillna(0).astype("int64")

    features["hist_spending"] = (features["hist_gross_spending"] + features["hist_return_amount"]).round(2)
    features["hist_quantity"] = (features["hist_gross_qty"] + features["hist_return_qty"]).astype("int64")
    features["hist_aov"] = (features["hist_spending"] / features["hist_frequency"]).round(2)

    cutoff_ts = pd.Timestamp(obs_end)
    features["hist_recency"] = (cutoff_ts - features["last_purchase_date"].dt.normalize()).dt.days.astype("int64")
    features["hist_tenure"] = (cutoff_ts - features["first_purchase_date"].dt.normalize()).dt.days.astype("int64")

    tgt_agg = (
        tgt.groupby("customer_id", as_index=False)["TotalAmount"]
        .sum()
        .rename(columns={"TotalAmount": TARGET_COLUMN})
    )
    cohort = features.merge(tgt_agg, on="customer_id", how="left")
    cohort[TARGET_COLUMN] = cohort[TARGET_COLUMN].fillna(0.0).round(2)
    return cohort.sort_values("customer_id").reset_index(drop=True)


def build_linear_regression_pipeline() -> Pipeline:
    """Create the scikit-learn preprocessing and Linear Regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def compute_regression_metrics(
    model_name: str, split: str, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> Dict[str, float | int | str]:
    """Compute MAE, MSE, RMSE, and R2 Score for a model split."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "model_name": model_name,
        "split": split,
        "n_samples": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate_regression_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Fit baselines and Linear Regression strictly on train_df and evaluate chronologically on test_df.

    Returns the fitted full Linear Regression pipeline, an evaluation comparison DataFrame, and
    a DataFrame of test-set predictions/residuals.
    """
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    results = []

    # 1. Naive baselines (mean and median of training target)
    for strategy in ("mean", "median"):
        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(X_train, y_train)
        results.append(compute_regression_metrics(f"Baseline ({strategy.title()})", "test", y_test, dummy.predict(X_test)))

    # 2. Simple 1-feature OLS baseline (historical spending only)
    simple_pipe = build_linear_regression_pipeline()
    simple_pipe.fit(train_df[["hist_spending"]], y_train)
    results.append(
        compute_regression_metrics("Simple OLS (hist_spending only)", "test", y_test, simple_pipe.predict(test_df[["hist_spending"]]))
    )

    # 3. Full Linear Regression pipeline
    full_pipe = build_linear_regression_pipeline()
    full_pipe.fit(X_train, y_train)
    y_train_pred = full_pipe.predict(X_train)
    y_test_pred = full_pipe.predict(X_test)
    results.append(compute_regression_metrics("Linear Regression (Full Pipeline)", "train", y_train, y_train_pred))
    results.append(compute_regression_metrics("Linear Regression (Full Pipeline)", "test", y_test, y_test_pred))

    # 4. Outlier sensitivity checks
    train_q99 = float(y_train.quantile(0.99))
    trimmed_train = train_df[y_train <= train_q99]
    trimmed_pipe = build_linear_regression_pipeline()
    trimmed_pipe.fit(trimmed_train[feature_cols], trimmed_train[TARGET_COLUMN])
    results.append(
        compute_regression_metrics("Linear Regression (Trained w/o Top 1% Target Outliers)", "test", y_test, trimmed_pipe.predict(X_test))
    )

    test_q99 = float(y_test.quantile(0.99))
    non_outlier_mask = y_test <= test_q99
    results.append(
        compute_regression_metrics(
            "Linear Regression (Full Pipeline)",
            "test_excl_top1pct_outliers",
            y_test[non_outlier_mask],
            y_test_pred[non_outlier_mask],
        )
    )

    pred_df = test_df[["customer_id", "hist_spending", "hist_frequency", "hist_recency", TARGET_COLUMN]].copy()
    pred_df["predicted_spending"] = np.round(y_test_pred, 2)
    pred_df["residual"] = (pred_df[TARGET_COLUMN] - pred_df["predicted_spending"]).round(2)
    pred_df["abs_error"] = pred_df["residual"].abs().round(2)

    return full_pipe, pd.DataFrame(results), pred_df
