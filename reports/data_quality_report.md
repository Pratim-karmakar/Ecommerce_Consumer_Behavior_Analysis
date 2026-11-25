# Data Quality Audit Report — UCI Online Retail II

- **Source Dataset:** `dataset\Ecommerce.csv` (`90.46 MB`, encoding `utf-8`)
- **Audit Status:** Read-only inspection complete (**no records modified or deleted**)

## Executive Summary Table

| Metric | Category | Value | % of Total Rows | Notes |
|---|---|---:|---:|---|
| Total rows | Dataset Dimensions | 1,067,371 | 100.0% | Combined 2009-2010 and 2010-2011 transactions |
| Total columns | Dataset Dimensions | 8 | — | Expected schema: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |
| Missing values (total cells) | Missing Values | 247,389 | 2.8972% | Sum of null cells across all columns |
| Missing Customer ID records | Missing Values | 243,007 | 22.7669% | Guest/unregistered checkouts or internal stock adjustments |
| Missing Description records | Missing Values | 4,382 | 0.4105% | Rows with null product description (4,382 also lack Customer ID) |
| Duplicate records (exact rows) | Duplicates | 34,335 | 3.2168% | Includes 22,844 duplicates in Dec 1-9, 2010 sheet overlap window |
| Invalid or unusual records (union) | Invalid / Unusual Values | 12,963 | 1.2145% | Rows with price<=0, non-C qty<0, A-invoices, non-standard StockCode, lowercase notes, or Unspecified country |
| Cancelled invoice records (C-prefix) | Cancellations | 19,494 | 1.8264% | Across 8,292 unique C-prefixed invoices (1 has Quantity > 0) |
| Negative quantity records (all) | Quantity Anomalies | 22,950 | 2.1501% | 19,493 in C-invoices + 3,457 non-C inventory adjustments |
| Non-C negative quantity records | Quantity Anomalies | 3,457 | 0.3239% | Warehouse write-offs/damages (3,457 Price==0; 3,457 missing Customer ID) |
| Zero quantity records | Quantity Anomalies | 0 | 0.0% | Rows with Quantity == 0 |
| Negative price records | Price Anomalies | 5 | 0.0005% | Adjust bad debt records (StockCode='B', A-prefixed invoices) |
| Zero price records | Price Anomalies | 6,202 | 0.5811% | Includes 3,457 non-C negative qty adjustments + 2,745 positive qty zero-price items |
| Bad debt adjustment records (A-prefix) | Invalid / Unusual Values | 6 | 0.0006% | Across 6 unique A-prefixed accounting adjustment invoices |
| Non-standard StockCode records | Invalid / Unusual Values | 6,093 | 0.5708% | Fees/postage/manual/gift/test codes not matching 5-digit pattern |
| Test StockCode records (TEST*) | Invalid / Unusual Values | 17 | 0.0016% | Internal test product entries (TEST001, TEST002) |
| Lowercase/note Description records | Invalid / Unusual Values | 823 | 0.0771% | Warehouse status notes like 'check', '?', 'damages', 'lost' |
| Unspecified country records | Invalid / Unusual Values | 756 | 0.0708% | Rows with Country == 'Unspecified' |
| Unparseable InvoiceDate records | Date Audit | 0 | 0.0% | InvoiceDate values failing datetime parsing |
| Date range | Date Audit | 2009-12-01 07:45:00 to 2011-12-09 12:50:00 | — | Spans combined 2009-2010 and 2010-2011 sheets |
| Number of unique invoices | Entity Counts | 53,628 | — | 45,330 standard + 8,292 C-cancelled + 6 A-adjustments |
| Number of customers | Entity Counts | 5,942 | — | Distinct non-null Customer ID values |
| Number of products (StockCode raw) | Entity Counts | 5,305 | — | 5,131 unique after stripping whitespace and uppercasing |
| Number of product descriptions | Entity Counts | 5,655 | — | Distinct non-null stripped Description values |
| Number of countries | Entity Counts | 43 | — | Includes 'Unspecified' |

## Key Audit Findings

1. **Dataset Scale & Date Coverage:** `1,067,371` rows x `8` columns spanning `2009-12-01 07:45:00` to `2011-12-09 12:50:00` (`0` unparseable timestamps).
2. **Missing Values:** `Customer ID` is missing on `243,007` rows (`22.77%`), and `Description` is missing on `4,382` rows (`0.41%`, `4,382` of which also lack `Customer ID`).
3. **Duplicates & Sheet Overlap:** `34,335` exact duplicate rows (`3.22%`), of which `22,844` (`66.5%` of duplicates) occur in the `2010-12-01` to `2010-12-09` overlap window between the two combined UCI sheets.
4. **Cancellations vs. Stock Adjustments vs. Bad Debt:** `19,494` rows (`8,292` `C`-invoices) are cancellations (`1` anomalous row with `Quantity > 0`), `3,457` rows are non-`C` negative-quantity warehouse write-offs (all `3,457` with `Price == 0` and `3,457` missing `Customer ID`), and `6` rows are `A`-prefixed bad-debt adjustments (containing all `5` negative-price rows).
5. **Invalid / Non-Product Entries:** `6,202` rows have `Price == 0`, `6,093` rows have non-standard `StockCode`s (postage `POST`/`DOT`, manual `M`/`m`, discounts `D`, bank charges, `TEST001`/`TEST002` across `17` rows), `823` rows have lowercase warehouse notes, and `756` rows have `Country == 'Unspecified'`.

## Recommended Cleaning Strategy (Awaiting User Approval)

1. **Keep Raw File Immutable:** Never overwrite `dataset/Ecommerce.csv`; save cleaned outputs into separate derived files (e.g., `dataset/cleaned_transactions.parquet` or `.csv`).
2. **Remove Exact Duplicates First:** Drop exact duplicate rows (`34,335` rows), which eliminates the `22,844` Dec 1–9, 2010 sheet-overlap duplicates plus `11,491` intra-period duplicates.
3. **Separate Cancellations & Returns (`C`-Invoices) Instead of Blindly Discarding:** Keep a dedicated cancellation flag or return summary so net customer spend / net units properly subtract cancelled orders (including massive paired reversals like `±80,995` and `±74,215`) rather than overstating gross revenue.
4. **Exclude Non-Customer Operational Noise:** Remove `A`-prefixed bad-debt rows (`6` rows), non-`C` negative-quantity warehouse write-offs (`3,457` rows), `TEST*` rows (`17` rows), and `Price <= 0` records (`6,207` rows) from revenue/modeling tables.
5. **Handle Non-Standard Service/Fee `StockCode`s Deliberately:** Exclude fees/admin codes (`POST`, `DOT`, `M`/`m`, `BANK CHARGES`, `C2`, `D`, `ADJUST`, `AMAZONFEE`, `CRUK`, `B`) from product-level analytics and RFM product counts (or track postage separately when reconciling invoice totals).
6. **Normalize Identifiers & Text:** Convert `InvoiceDate` to `datetime`, strip whitespace and normalize `StockCode` casing (`5,305 -> 5,131`), strip `Description`, and standardize non-null `Customer ID` formatting.
7. **Split Scope for Missing `Customer ID`s:** Retain valid sales without `Customer ID` (`243,007` total rows missing ID) only for aggregate store/product revenue views if needed, and filter strictly to non-null `Customer ID` for customer-level RFM segmentation, churn, and CLV modeling.
