"""Modular data cleaning and preprocessing pipeline for UCI Online Retail II."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

COLUMN_RENAME_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Internal accounting/platform codes that do not represent customer purchases or returns.
INTERNAL_NON_COMMERCIAL_CODES = frozenset(
    {"ADJUST", "ADJUST2", "AMAZONFEE", "BANK CHARGES", "CRUK", "S"}
)

# Customer-facing service, postage, discount, and manual entries retained in transaction tables
# but flagged so product-level analytics can exclude them cleanly.
SERVICE_OR_FEE_CODES = frozenset({"POST", "DOT", "C2", "D", "M"})


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized snake_case column names."""
    return df.rename(columns=COLUMN_RENAME_MAP).copy()


def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text identifiers and convert date, quantity, price, and customer_id types."""
    out = df.copy()
    out["invoice"] = out["invoice"].astype("string").str.strip().str.upper()
    out["stock_code"] = out["stock_code"].astype("string").str.strip().str.upper()
    out["description"] = out["description"].astype("string").str.strip()
    out["country"] = out["country"].astype("string").str.strip()

    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").astype("int64")
    out["price"] = pd.to_numeric(out["price"], errors="coerce").astype("float64")
    # Use nullable Int64 so valid IDs drop trailing '.0' while missing IDs stay <NA> (never fabricated).
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    return out


def clean_retail_transactions(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean a copy of raw UCI Online Retail II transactions and record an audit trail of every step.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Validated transaction records (both normal sales and C-prefixed returns/cancellations).
    audit_log_df : pd.DataFrame
        Step-by-step table documenting rows before, rows removed, rows remaining, and business rationale.
    """
    working = standardize_columns(raw_df)
    steps = []

    def record_step(step_name: str, before_count: int, after_count: int, reason: str) -> None:
        steps.append(
            {
                "step": step_name,
                "rows_before": before_count,
                "rows_removed": before_count - after_count,
                "rows_after": after_count,
                "reason": reason,
            }
        )

    # 1. Exact duplicate removal (captures Dec 1-9, 2010 sheet overlap + intra-period repeats).
    n0 = len(working)
    working = working.drop_duplicates(keep="first").copy()
    record_step(
        "Remove exact duplicate rows",
        n0,
        len(working),
        "Eliminates combined-sheet overlap duplicates (Dec 1-9, 2010) and repeated identical rows",
    )

    # 2. Convert types & normalize identifiers/strings on deduplicated copy.
    working = enforce_data_types(working)

    # 3. Remove bad-debt accounting adjustments (A-prefixed invoices, StockCode='B').
    n1 = len(working)
    working = working[~working["invoice"].str.startswith("A", na=False)].copy()
    record_step(
        "Remove 'A'-prefixed bad-debt adjustments",
        n1,
        len(working),
        "Internal ledger bad-debt adjustments (includes all negative Price rows), not customer orders",
    )

    # 4. Remove internal test products (TEST001, TEST002).
    n2 = len(working)
    working = working[~working["stock_code"].str.startswith("TEST", na=False)].copy()
    record_step(
        "Remove TEST* stock codes",
        n2,
        len(working),
        "Internal software test product entries (TEST001/TEST002), not real sales",
    )

    # 5. Remove non-'C' negative-quantity stock write-offs (lost, damages, short, missing Price/Customer).
    n3 = len(working)
    is_c_invoice = working["invoice"].str.startswith("C", na=False)
    is_non_c_neg_qty = (working["quantity"] < 0) & (~is_c_invoice)
    working = working[~is_non_c_neg_qty].copy()
    record_step(
        "Remove non-C negative-quantity write-offs",
        n3,
        len(working),
        "Warehouse inventory adjustments (damages/lost/short) with Price==0 and missing Customer ID",
    )

    # 6. Remove remaining zero/negative price records (unpriced promos/missing descriptions).
    n4 = len(working)
    working = working[working["price"] > 0].copy()
    record_step(
        "Remove Price <= 0 records",
        n4,
        len(working),
        "Unpriced adjustments/promos (also resolves 100% of remaining missing Descriptions)",
    )

    # 7. Remove anomalous C-prefixed invoice with positive quantity (sign contradiction).
    n5 = len(working)
    is_c_pos_qty = working["invoice"].str.startswith("C", na=False) & (working["quantity"] > 0)
    working = working[~is_c_pos_qty].copy()
    record_step(
        "Remove anomalous C-invoice with Quantity > 0",
        n5,
        len(working),
        "Single contradictory manual cancellation row (C496350) with positive quantity",
    )

    # 8. Remove internal platform/accounting fees & sample codes.
    n6 = len(working)
    working = working[~working["stock_code"].isin(INTERNAL_NON_COMMERCIAL_CODES)].copy()
    record_step(
        "Remove internal fee/ledger/sample StockCodes",
        n6,
        len(working),
        "Excludes AMAZONFEE, BANK CHARGES, ADJUST, ADJUST2, CRUK, and S (samples)",
    )

    # 9. Add validated business columns and transparency flags.
    working["transaction_amount"] = (working["quantity"] * working["price"]).round(4)
    working["is_cancelled"] = working["invoice"].str.startswith("C", na=False).astype(bool)
    working["transaction_type"] = (
        working["is_cancelled"]
        .map({True: "cancellation_or_return", False: "sale"})
        .astype("string")
    )
    working["has_customer_id"] = working["customer_id"].notna().astype(bool)
    working["is_merchandise"] = (~working["stock_code"].isin(SERVICE_OR_FEE_CODES)).astype(bool)
    working["is_unspecified_country"] = working["country"].eq("Unspecified").astype(bool)

    working = working.reset_index(drop=True)
    audit_log_df = pd.DataFrame(steps)
    return working, audit_log_df


def split_sales_and_returns(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition cleaned transactions into normal sales and C-prefixed returns/cancellations."""
    sales_df = cleaned_df[~cleaned_df["is_cancelled"]].reset_index(drop=True)
    returns_df = cleaned_df[cleaned_df["is_cancelled"]].reset_index(drop=True)
    return sales_df, returns_df


def export_processed_datasets(
    cleaned_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Save the three documented cleaned CSV datasets under dataset/processed/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cleaned_online_retail": output_dir / "cleaned_online_retail.csv",
        "sales_transactions": output_dir / "sales_transactions.csv",
        "returns_and_cancellations": output_dir / "returns_and_cancellations.csv",
    }
    cleaned_df.to_csv(paths["cleaned_online_retail"], index=False, encoding="utf-8")
    sales_df.to_csv(paths["sales_transactions"], index=False, encoding="utf-8")
    returns_df.to_csv(paths["returns_and_cancellations"], index=False, encoding="utf-8")
    return paths
