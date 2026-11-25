"""Read-only dataset loading and data quality audit helpers."""

from pathlib import Path
from typing import Any, Dict, Tuple
import re

import pandas as pd

STANDARD_STOCKCODE_PATTERN = re.compile(r"^\d{5}[A-Za-z]*$")
LOWERCASE_NOTE_PATTERN = re.compile(r"^[a-z\s\?\.\-\d/]+$")


def load_raw_transactions(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load the raw transaction CSV while preserving string identifiers."""
    # Force Invoice and StockCode to str so leading zeros and C/A prefixes stay intact.
    dtype_overrides = {"Invoice": "string", "StockCode": "string"}
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding, dtype=dtype_overrides)
            return df, encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {csv_path} with standard encodings.")


def classify_records(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Build boolean masks identifying cancellations, adjustments, and anomalies without altering df."""
    invoice_clean = df["Invoice"].fillna("").astype(str).str.strip()
    stock_clean = df["StockCode"].fillna("").astype(str).str.strip()
    desc_series = df["Description"]
    qty = df["Quantity"]
    price = df["Price"]

    is_c_invoice = invoice_clean.str.upper().str.startswith("C")
    is_a_invoice = invoice_clean.str.upper().str.startswith("A")
    is_digit_invoice = invoice_clean.str.match(r"^\d+$")

    is_non_standard_stock = ~stock_clean.str.match(STANDARD_STOCKCODE_PATTERN)
    is_test_stock = stock_clean.str.upper().str.startswith("TEST")
    is_lowercase_desc = desc_series.notna() & desc_series.astype(str).str.match(LOWERCASE_NOTE_PATTERN)

    return {
        "invoice_clean": invoice_clean,
        "stock_clean": stock_clean,
        "is_c_invoice": is_c_invoice,
        "is_a_invoice": is_a_invoice,
        "is_digit_invoice": is_digit_invoice,
        "is_non_c_negative_qty": (qty < 0) & (~is_c_invoice),
        "is_c_positive_qty": (qty > 0) & is_c_invoice,
        "is_zero_qty": qty == 0,
        "is_negative_qty": qty < 0,
        "is_negative_price": price < 0,
        "is_zero_price": price == 0,
        "is_missing_customer": df["Customer ID"].isna(),
        "is_missing_description": desc_series.isna(),
        "is_non_standard_stock": is_non_standard_stock,
        "is_test_stock": is_test_stock,
        "is_lowercase_desc": is_lowercase_desc,
        "is_unspecified_country": df["Country"].fillna("").astype(str).str.strip().eq("Unspecified"),
    }


def build_data_quality_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute a structured DataFrame and dictionary summarizing all data quality audit metrics."""
    total_rows, total_cols = df.shape
    masks = classify_records(df)
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    exact_dup_count = int(df.duplicated(keep="first").sum())
    overlap_mask = (dates >= "2010-12-01") & (dates < "2010-12-10")
    overlap_dup_count = int(df.duplicated(keep="first")[overlap_mask].sum())

    missing_cust = int(masks["is_missing_customer"].sum())
    missing_desc = int(masks["is_missing_description"].sum())
    missing_desc_no_cust = int((masks["is_missing_description"] & masks["is_missing_customer"]).sum())
    total_missing_cells = int(df.isna().sum().sum())

    cancelled_rows = int(masks["is_c_invoice"].sum())
    cancelled_unique_invoices = int(masks["invoice_clean"][masks["is_c_invoice"]].nunique())
    bad_debt_rows = int(masks["is_a_invoice"].sum())
    bad_debt_unique_invoices = int(masks["invoice_clean"][masks["is_a_invoice"]].nunique())
    digit_unique_invoices = int(masks["invoice_clean"][masks["is_digit_invoice"]].nunique())

    neg_qty_rows = int(masks["is_negative_qty"].sum())
    zero_qty_rows = int(masks["is_zero_qty"].sum())
    non_c_neg_qty_rows = int(masks["is_non_c_negative_qty"].sum())
    non_c_neg_qty_zero_price = int((masks["is_non_c_negative_qty"] & masks["is_zero_price"]).sum())
    non_c_neg_qty_no_cust = int((masks["is_non_c_negative_qty"] & masks["is_missing_customer"]).sum())
    c_pos_qty_rows = int(masks["is_c_positive_qty"].sum())

    neg_price_rows = int(masks["is_negative_price"].sum())
    zero_price_rows = int(masks["is_zero_price"].sum())
    zero_price_pos_qty_rows = int((masks["is_zero_price"] & (df["Quantity"] > 0)).sum())

    non_std_stock_rows = int(masks["is_non_standard_stock"].sum())
    test_stock_rows = int(masks["is_test_stock"].sum())
    lowercase_desc_rows = int(masks["is_lowercase_desc"].sum())
    unspecified_country_rows = int(masks["is_unspecified_country"].sum())
    unparseable_dates = int(dates.isna().sum())

    # Union of rows affected by explicit structural/value anomalies (non-standard codes, test rows,
    # negative/zero prices, non-C negative quantities, bad-debt A-invoices, or lowercase warehouse notes).
    invalid_or_unusual_mask = (
        masks["is_negative_price"]
        | masks["is_zero_price"]
        | masks["is_zero_qty"]
        | masks["is_non_c_negative_qty"]
        | masks["is_c_positive_qty"]
        | masks["is_a_invoice"]
        | masks["is_non_standard_stock"]
        | masks["is_lowercase_desc"]
        | masks["is_unspecified_country"]
    )
    invalid_or_unusual_rows = int(invalid_or_unusual_mask.sum())

    unique_invoices = int(masks["invoice_clean"].nunique())
    unique_customers = int(df["Customer ID"].nunique(dropna=True))
    unique_stockcodes_raw = int(df["StockCode"].nunique(dropna=True))
    unique_stockcodes_upper = int(masks["stock_clean"].str.upper().nunique())
    unique_descriptions = int(df["Description"].dropna().astype(str).str.strip().nunique())
    unique_countries = int(df["Country"].dropna().nunique())

    date_min_str = dates.min().strftime("%Y-%m-%d %H:%M:%S")
    date_max_str = dates.max().strftime("%Y-%m-%d %H:%M:%S")
    date_range_str = f"{date_min_str} to {date_max_str}"

    def pct(count: int) -> float:
        return round((count / total_rows) * 100, 4) if total_rows else 0.0

    rows = [
        ("Total rows", "Dataset Dimensions", total_rows, 100.0, "Combined 2009-2010 and 2010-2011 transactions"),
        ("Total columns", "Dataset Dimensions", total_cols, "", "Expected schema: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country"),
        ("Missing values (total cells)", "Missing Values", total_missing_cells, round((total_missing_cells / (total_rows * total_cols)) * 100, 4), "Sum of null cells across all columns"),
        ("Missing Customer ID records", "Missing Values", missing_cust, pct(missing_cust), "Guest/unregistered checkouts or internal stock adjustments"),
        ("Missing Description records", "Missing Values", missing_desc, pct(missing_desc), f"Rows with null product description ({missing_desc_no_cust:,} also lack Customer ID)"),
        ("Duplicate records (exact rows)", "Duplicates", exact_dup_count, pct(exact_dup_count), f"Includes {overlap_dup_count:,} duplicates in Dec 1-9, 2010 sheet overlap window"),
        ("Invalid or unusual records (union)", "Invalid / Unusual Values", invalid_or_unusual_rows, pct(invalid_or_unusual_rows), "Rows with price<=0, non-C qty<0, A-invoices, non-standard StockCode, lowercase notes, or Unspecified country"),
        ("Cancelled invoice records (C-prefix)", "Cancellations", cancelled_rows, pct(cancelled_rows), f"Across {cancelled_unique_invoices:,} unique C-prefixed invoices ({c_pos_qty_rows} has Quantity > 0)"),
        ("Negative quantity records (all)", "Quantity Anomalies", neg_qty_rows, pct(neg_qty_rows), f"{neg_qty_rows - non_c_neg_qty_rows:,} in C-invoices + {non_c_neg_qty_rows:,} non-C inventory adjustments"),
        ("Non-C negative quantity records", "Quantity Anomalies", non_c_neg_qty_rows, pct(non_c_neg_qty_rows), f"Warehouse write-offs/damages ({non_c_neg_qty_zero_price:,} Price==0; {non_c_neg_qty_no_cust:,} missing Customer ID)"),
        ("Zero quantity records", "Quantity Anomalies", zero_qty_rows, pct(zero_qty_rows), "Rows with Quantity == 0"),
        ("Negative price records", "Price Anomalies", neg_price_rows, pct(neg_price_rows), "Adjust bad debt records (StockCode='B', A-prefixed invoices)"),
        ("Zero price records", "Price Anomalies", zero_price_rows, pct(zero_price_rows), f"Includes {non_c_neg_qty_zero_price:,} non-C negative qty adjustments + {zero_price_pos_qty_rows:,} positive qty zero-price items"),
        ("Bad debt adjustment records (A-prefix)", "Invalid / Unusual Values", bad_debt_rows, pct(bad_debt_rows), f"Across {bad_debt_unique_invoices:,} unique A-prefixed accounting adjustment invoices"),
        ("Non-standard StockCode records", "Invalid / Unusual Values", non_std_stock_rows, pct(non_std_stock_rows), "Fees/postage/manual/gift/test codes not matching 5-digit pattern"),
        ("Test StockCode records (TEST*)", "Invalid / Unusual Values", test_stock_rows, pct(test_stock_rows), "Internal test product entries (TEST001, TEST002)"),
        ("Lowercase/note Description records", "Invalid / Unusual Values", lowercase_desc_rows, pct(lowercase_desc_rows), "Warehouse status notes like 'check', '?', 'damages', 'lost'"),
        ("Unspecified country records", "Invalid / Unusual Values", unspecified_country_rows, pct(unspecified_country_rows), "Rows with Country == 'Unspecified'"),
        ("Unparseable InvoiceDate records", "Date Audit", unparseable_dates, pct(unparseable_dates), "InvoiceDate values failing datetime parsing"),
        ("Date range", "Date Audit", date_range_str, "", "Spans combined 2009-2010 and 2010-2011 sheets"),
        ("Number of unique invoices", "Entity Counts", unique_invoices, "", f"{digit_unique_invoices:,} standard + {cancelled_unique_invoices:,} C-cancelled + {bad_debt_unique_invoices:,} A-adjustments"),
        ("Number of customers", "Entity Counts", unique_customers, "", "Distinct non-null Customer ID values"),
        ("Number of products (StockCode raw)", "Entity Counts", unique_stockcodes_raw, "", f"{unique_stockcodes_upper:,} unique after stripping whitespace and uppercasing"),
        ("Number of product descriptions", "Entity Counts", unique_descriptions, "", "Distinct non-null stripped Description values"),
        ("Number of countries", "Entity Counts", unique_countries, "", "Includes 'Unspecified'"),
    ]

    report_df = pd.DataFrame(
        rows, columns=["metric", "category", "value", "percentage_of_rows", "notes"]
    )
    summary_dict: Dict[str, Any] = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_missing_cells": total_missing_cells,
        "missing_customer_id": missing_cust,
        "missing_description": missing_desc,
        "missing_desc_no_cust": missing_desc_no_cust,
        "exact_duplicates": exact_dup_count,
        "overlap_duplicates": overlap_dup_count,
        "invalid_or_unusual_rows": invalid_or_unusual_rows,
        "cancelled_rows": cancelled_rows,
        "cancelled_unique_invoices": cancelled_unique_invoices,
        "digit_unique_invoices": digit_unique_invoices,
        "bad_debt_unique_invoices": bad_debt_unique_invoices,
        "neg_qty_rows": neg_qty_rows,
        "non_c_neg_qty_rows": non_c_neg_qty_rows,
        "non_c_neg_qty_zero_price": non_c_neg_qty_zero_price,
        "non_c_neg_qty_no_cust": non_c_neg_qty_no_cust,
        "c_pos_qty_rows": c_pos_qty_rows,
        "zero_qty_rows": zero_qty_rows,
        "neg_price_rows": neg_price_rows,
        "zero_price_rows": zero_price_rows,
        "zero_price_pos_qty_rows": zero_price_pos_qty_rows,
        "bad_debt_rows": bad_debt_rows,
        "non_std_stock_rows": non_std_stock_rows,
        "test_stock_rows": test_stock_rows,
        "lowercase_desc_rows": lowercase_desc_rows,
        "unspecified_country_rows": unspecified_country_rows,
        "date_min": date_min_str,
        "date_max": date_max_str,
        "unique_invoices": unique_invoices,
        "unique_customers": unique_customers,
        "unique_stockcodes_raw": unique_stockcodes_raw,
        "unique_stockcodes_upper": unique_stockcodes_upper,
        "unique_descriptions": unique_descriptions,
        "unique_countries": unique_countries,
    }
    return report_df, summary_dict
