"""Feature engineering pipeline generating transaction, customer, product, country, and monthly datasets."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def add_transaction_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and amount features to cleaned transactions without modifying input."""
    df = cleaned_df.copy()
    dt = pd.to_datetime(df["invoice_date"])

    df["TotalAmount"] = (df["quantity"] * df["price"]).round(4)
    df["Year"] = dt.dt.year.astype("int64")
    df["Month"] = dt.dt.month.astype("int64")
    df["MonthName"] = dt.dt.month_name().astype("string")
    df["Quarter"] = dt.dt.quarter.astype("int64")
    df["Day"] = dt.dt.day.astype("int64")
    # Pandas day_of_week convention: 0=Monday ... 6=Sunday.
    df["DayOfWeek"] = dt.dt.day_of_week.astype("int64")
    df["DayName"] = dt.dt.day_name().astype("string")
    df["Hour"] = dt.dt.hour.astype("int64")
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(bool)
    df["InvoiceYearMonth"] = dt.dt.strftime("%Y-%m").astype("string")
    return df


def compute_reference_date(tx_df: pd.DataFrame) -> pd.Timestamp:
    """Define Recency reference date as midnight the day after the final observed transaction."""
    max_dt = pd.to_datetime(tx_df["invoice_date"]).max()
    return max_dt.normalize() + pd.Timedelta(days=1)


def build_customer_analytics(
    tx_df: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> Tuple[pd.DataFrame, pd.Timestamp, Dict[str, int]]:
    """
    Build customer-level RFM and spending features for customers with >= 1 valid sale.

    Business rules:
    - Excludes rows with missing customer_id (never fabricates customer IDs).
    - TotalOrders / Frequency / FirstPurchaseDate / LastPurchaseDate / Recency /
      UniqueProductsPurchased are computed from positive purchase invoices (is_cancelled == False).
    - TotalSpending, MonetaryValue, and TotalQuantity are net of C-prefixed cancellations/returns
      so fully or largely cancelled orders do not inflate customer lifetime value.
    """
    ref_date = reference_date or compute_reference_date(tx_df)
    identified = tx_df[tx_df["has_customer_id"]].copy()
    sales = identified[~identified["is_cancelled"]].copy()
    returns = identified[identified["is_cancelled"]].copy()

    sales_agg = sales.groupby("customer_id", as_index=False).agg(
        GrossSpending=("TotalAmount", "sum"),
        TotalOrders=("invoice", "nunique"),
        GrossQuantity=("quantity", "sum"),
        UniqueProductsPurchased=("stock_code", "nunique"),
        FirstPurchaseDate=("invoice_date", "min"),
        LastPurchaseDate=("invoice_date", "max"),
    )

    ret_agg = returns.groupby("customer_id", as_index=False).agg(
        ReturnedAmount=("TotalAmount", "sum"),
        ReturnedQuantity=("quantity", "sum"),
        CancelledOrders=("invoice", "nunique"),
    )

    cust = sales_agg.merge(ret_agg, on="customer_id", how="left")
    cust["ReturnedAmount"] = cust["ReturnedAmount"].fillna(0.0)
    cust["ReturnedQuantity"] = cust["ReturnedQuantity"].fillna(0).astype("int64")
    cust["CancelledOrders"] = cust["CancelledOrders"].fillna(0).astype("int64")

    cust["TotalSpending"] = (cust["GrossSpending"] + cust["ReturnedAmount"]).round(2)
    cust["GrossSpending"] = cust["GrossSpending"].round(2)
    cust["ReturnedAmount"] = cust["ReturnedAmount"].round(2)
    cust["TotalQuantity"] = (cust["GrossQuantity"] + cust["ReturnedQuantity"]).astype("int64")
    cust["AverageOrderValue"] = (cust["TotalSpending"] / cust["TotalOrders"]).round(2)
    cust["Recency"] = (ref_date - cust["LastPurchaseDate"].dt.normalize()).dt.days.astype("int64")
    cust["Frequency"] = cust["TotalOrders"].astype("int64")
    cust["MonetaryValue"] = cust["TotalSpending"]

    cust = cust.rename(columns={"customer_id": "Customer ID"})
    ordered_cols = [
        "Customer ID",
        "TotalSpending",
        "TotalOrders",
        "TotalQuantity",
        "UniqueProductsPurchased",
        "AverageOrderValue",
        "FirstPurchaseDate",
        "LastPurchaseDate",
        "Recency",
        "Frequency",
        "MonetaryValue",
        "GrossSpending",
        "ReturnedAmount",
        "CancelledOrders",
        "GrossQuantity",
        "ReturnedQuantity",
    ]
    cust = cust[ordered_cols].sort_values("TotalSpending", ascending=False).reset_index(drop=True)

    all_ids = set(identified["customer_id"].unique())
    sales_ids = set(sales["customer_id"].unique())
    meta = {
        "all_identified_customers": len(all_ids),
        "purchasing_customers": len(sales_ids),
        "return_only_customers_excluded": len(all_ids - sales_ids),
    }
    return cust, ref_date, meta


def _most_frequent_description(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return ""
    top_count = counts.iloc[0]
    tied = sorted(counts[counts == top_count].index.tolist())
    return tied[0]


def build_product_analytics(tx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build product-level metrics across merchandise StockCodes (is_merchandise == True).

    TotalQuantitySold and TotalRevenue reflect net units and net revenue after C-prefixed returns,
    while GrossQuantitySold, GrossRevenue, ReturnedQuantity, and ReturnedRevenue expose both sides.
    """
    merch = tx_df[tx_df["is_merchandise"]].copy()
    sales = merch[~merch["is_cancelled"]].copy()
    returns = merch[merch["is_cancelled"]].copy()

    desc_map = merch.groupby("stock_code")["description"].agg(_most_frequent_description)

    net_agg = merch.groupby("stock_code", as_index=False).agg(
        TotalQuantitySold=("quantity", "sum"),
        TotalRevenue=("TotalAmount", "sum"),
        TotalInvoices=("invoice", "nunique"),
        UniqueCustomers=("customer_id", "nunique"),
        MeanTransactionPrice=("price", "mean"),
    )

    sales_agg = sales.groupby("stock_code", as_index=False).agg(
        GrossQuantitySold=("quantity", "sum"),
        GrossRevenue=("TotalAmount", "sum"),
        SalesInvoices=("invoice", "nunique"),
    )

    ret_agg = returns.groupby("stock_code", as_index=False).agg(
        ReturnedQuantity=("quantity", "sum"),
        ReturnedRevenue=("TotalAmount", "sum"),
        ReturnInvoices=("invoice", "nunique"),
    )

    prod = (
        net_agg.merge(sales_agg, on="stock_code", how="left")
        .merge(ret_agg, on="stock_code", how="left")
    )
    prod["Description"] = prod["stock_code"].map(desc_map).astype("string")

    for col in ("GrossQuantitySold", "SalesInvoices", "ReturnedQuantity", "ReturnInvoices"):
        prod[col] = prod[col].fillna(0).astype("int64")
    for col in ("GrossRevenue", "ReturnedRevenue"):
        prod[col] = prod[col].fillna(0.0).round(2)

    prod["TotalRevenue"] = prod["TotalRevenue"].round(2)
    # Quantity-weighted selling price across sales; fall back to mean price for return-only SKUs.
    prod["AverageSellingPrice"] = np.where(
        prod["GrossQuantitySold"] > 0,
        prod["GrossRevenue"] / prod["GrossQuantitySold"],
        prod["MeanTransactionPrice"],
    ).round(4)

    prod = prod.rename(columns={"stock_code": "StockCode"})
    ordered_cols = [
        "StockCode",
        "Description",
        "TotalQuantitySold",
        "TotalRevenue",
        "TotalInvoices",
        "UniqueCustomers",
        "AverageSellingPrice",
        "GrossQuantitySold",
        "GrossRevenue",
        "ReturnedQuantity",
        "ReturnedRevenue",
        "SalesInvoices",
        "ReturnInvoices",
    ]
    return prod[ordered_cols].sort_values("TotalRevenue", ascending=False).reset_index(drop=True)


def build_country_analytics(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Build country-level net and gross sales metrics across all valid commercial transactions."""
    sales = tx_df[~tx_df["is_cancelled"]].copy()
    returns = tx_df[tx_df["is_cancelled"]].copy()

    net_agg = tx_df.groupby("country", as_index=False).agg(
        TotalRevenue=("TotalAmount", "sum"),
        TotalQuantity=("quantity", "sum"),
        TotalCustomers=("customer_id", "nunique"),
    )
    sales_agg = sales.groupby("country", as_index=False).agg(
        TotalOrders=("invoice", "nunique"),
        GrossRevenue=("TotalAmount", "sum"),
    )
    ret_agg = returns.groupby("country", as_index=False).agg(
        CancelledOrders=("invoice", "nunique"),
        ReturnedAmount=("TotalAmount", "sum"),
    )

    ctry = (
        net_agg.merge(sales_agg, on="country", how="left")
        .merge(ret_agg, on="country", how="left")
    )
    for col in ("TotalOrders", "CancelledOrders"):
        ctry[col] = ctry[col].fillna(0).astype("int64")
    for col in ("TotalRevenue", "GrossRevenue", "ReturnedAmount"):
        ctry[col] = ctry[col].fillna(0.0).round(2)

    ctry["AverageOrderValue"] = np.where(
        ctry["TotalOrders"] > 0, ctry["TotalRevenue"] / ctry["TotalOrders"], 0.0
    ).round(2)

    ctry = ctry.rename(columns={"country": "Country"})
    ordered_cols = [
        "Country",
        "TotalRevenue",
        "TotalOrders",
        "TotalCustomers",
        "TotalQuantity",
        "AverageOrderValue",
        "GrossRevenue",
        "ReturnedAmount",
        "CancelledOrders",
    ]
    return ctry[ordered_cols].sort_values("TotalRevenue", ascending=False).reset_index(drop=True)


def build_monthly_sales(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly sales and net revenue summary across the 2009-12 to 2011-12 timeline."""
    sales = tx_df[~tx_df["is_cancelled"]].copy()
    returns = tx_df[tx_df["is_cancelled"]].copy()

    group_cols = ["Year", "Month", "MonthName", "InvoiceYearMonth"]
    net_agg = tx_df.groupby(group_cols, as_index=False).agg(
        TotalRevenue=("TotalAmount", "sum"),
        TotalQuantity=("quantity", "sum"),
        UniqueCustomers=("customer_id", "nunique"),
    )
    sales_agg = sales.groupby(group_cols, as_index=False).agg(
        TotalOrders=("invoice", "nunique"),
        GrossRevenue=("TotalAmount", "sum"),
    )
    ret_agg = returns.groupby(group_cols, as_index=False).agg(
        CancelledOrders=("invoice", "nunique"),
        ReturnedAmount=("TotalAmount", "sum"),
    )

    monthly = (
        net_agg.merge(sales_agg, on=group_cols, how="left")
        .merge(ret_agg, on=group_cols, how="left")
    )
    for col in ("TotalOrders", "CancelledOrders"):
        monthly[col] = monthly[col].fillna(0).astype("int64")
    for col in ("TotalRevenue", "GrossRevenue", "ReturnedAmount"):
        monthly[col] = monthly[col].fillna(0.0).round(2)

    ordered_cols = [
        "Year",
        "Month",
        "MonthName",
        "TotalRevenue",
        "TotalOrders",
        "TotalQuantity",
        "UniqueCustomers",
        "InvoiceYearMonth",
        "GrossRevenue",
        "ReturnedAmount",
        "CancelledOrders",
    ]
    return monthly[ordered_cols].sort_values(["Year", "Month"]).reset_index(drop=True)


def export_analytical_datasets(
    datasets: Dict[str, pd.DataFrame], output_dir: Path
) -> Dict[str, Path]:
    """Save engineered analytical tables under dataset/processed/analytics/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for name, df in datasets.items():
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        saved[name] = out_path
    return saved
