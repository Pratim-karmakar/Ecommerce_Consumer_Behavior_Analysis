"""Cached data and model loaders reusing modular src/ pipelines."""

from pathlib import Path
import sys
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classification import build_purchase_classification_cohort  # noqa: E402
from src.models.inference import (  # noqa: E402
    load_kmeans_model,
    load_knn_recommender,
    load_linear_regression_model,
    load_logistic_regression_model,
)

ANALYTICS_DIR = PROJECT_ROOT / "dataset" / "processed" / "analytics"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
VIZ_DIR = PROJECT_ROOT / "visualizations"


@st.cache_data(show_spinner="Loading validated transaction records...")
def load_transactions() -> pd.DataFrame:
    """Load processed transaction features with memory-efficient dtypes."""
    path = ANALYTICS_DIR / "transaction_features.csv"
    return pd.read_csv(
        path,
        usecols=[
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "price",
            "customer_id",
            "country",
            "is_cancelled",
            "has_customer_id",
            "is_merchandise",
            "TotalAmount",
            "Year",
            "Month",
            "MonthName",
            "Quarter",
            "DayOfWeek",
            "DayName",
            "Hour",
            "IsWeekend",
            "InvoiceYearMonth",
        ],
        dtype={
            "invoice": "string",
            "stock_code": "string",
            "description": "string",
            "country": "category",
            "MonthName": "category",
            "DayName": "category",
            "InvoiceYearMonth": "category",
            "customer_id": "Int32",
            "quantity": "int32",
            "Year": "int16",
            "Month": "int8",
            "Quarter": "int8",
            "DayOfWeek": "int8",
            "Hour": "int8",
        },
        parse_dates=["invoice_date"],
    )


@st.cache_data
def load_customer_segments() -> pd.DataFrame:
    """Load customer RFM analytics merged with K-Means cluster assignments."""
    return pd.read_csv(
        ANALYTICS_DIR / "customer_segments.csv",
        dtype={"Customer ID": "Int64", "Segment": "string"},
        parse_dates=["FirstPurchaseDate", "LastPurchaseDate"],
    )


@st.cache_data
def load_product_analytics() -> pd.DataFrame:
    """Load merchandise product-level analytics."""
    return pd.read_csv(
        ANALYTICS_DIR / "product_analytics.csv",
        dtype={"StockCode": "string", "Description": "string"},
    )


@st.cache_data
def load_country_analytics() -> pd.DataFrame:
    """Load country-level revenue and order summaries."""
    return pd.read_csv(ANALYTICS_DIR / "country_analytics.csv", dtype={"Country": "string"})


@st.cache_data
def load_monthly_sales() -> pd.DataFrame:
    """Load monthly gross and net sales summary."""
    return pd.read_csv(ANALYTICS_DIR / "monthly_sales.csv")


@st.cache_data
def load_data_quality_report() -> pd.DataFrame:
    """Load structured data quality audit report CSV."""
    return pd.read_csv(REPORTS_DIR / "data_quality_report.csv")


@st.cache_data
def load_text_report(filename: str) -> str:
    """Load a markdown report safely as text."""
    path = REPORTS_DIR / filename
    return path.read_text(encoding="utf-8") if path.is_file() else ""


@st.cache_data
def load_model_metric_tables() -> Dict[str, pd.DataFrame]:
    """Load saved evaluation metric CSVs for all four models."""
    return {
        "linear_regression": pd.read_csv(REPORTS_DIR / "linear_regression_results.csv"),
        "logistic_regression": pd.read_csv(REPORTS_DIR / "logistic_regression_results.csv"),
        "knn_recommender": pd.read_csv(REPORTS_DIR / "knn_recommender_results.csv"),
        "kmeans_summary": pd.read_csv(REPORTS_DIR / "kmeans_cluster_summary.csv"),
    }


@st.cache_data(show_spinner="Preparing chronological test cohort...")
def load_chronological_test_cohort() -> Tuple[pd.DataFrame, str, str]:
    """
    Build the strictly non-overlapping 2010-2011 chronological test cohort for Pages 6 & 7.

    Observation: 2010-12-10 to 2011-09-10 (275 days)
    Prediction Target: 2011-09-11 to 2011-12-09 (90 Q4 days)
    """
    tx = load_transactions()
    cohort = build_purchase_classification_cohort(
        tx, "2010-12-10", "2011-09-11", "2011-09-11", "2011-12-10"
    )
    return cohort, "2010-12-10 to 2011-09-10 (275 days)", "2011-09-11 to 2011-12-09 (90 days)"


@st.cache_resource
def get_linear_regression_pipeline():
    """Load and cache the saved Linear Regression spending pipeline."""
    return load_linear_regression_model(MODELS_DIR)


@st.cache_resource
def get_logistic_regression_pipeline():
    """Load and cache the saved Logistic Regression purchase classifier."""
    return load_logistic_regression_model(MODELS_DIR)


@st.cache_resource
def get_knn_recommender_model():
    """Load and cache the saved Item-Based KNN product recommender."""
    return load_knn_recommender(MODELS_DIR)


@st.cache_resource
def get_kmeans_pipeline():
    """Load and cache the saved K-Means RFM customer segmentation pipeline."""
    return load_kmeans_model(MODELS_DIR)
