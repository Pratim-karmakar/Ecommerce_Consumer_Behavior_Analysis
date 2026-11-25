# Data Cleaning & Preprocessing Report — UCI Online Retail II

- **Source Dataset (Preserved Untouched):** `dataset\Ecommerce.csv` (SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`)
- **Cleaned Rows Retained:** `1,026,678` of `1,067,371` raw rows (`96.19%`)
- **Rows Removed:** `40,693` (`3.81%`)

## 1. Summary of Processed Output Files (`dataset/processed/`)

| File | Definition | Row Count | Customer-ID Rows | Net Amount (£) |
|---|---|---:|---:|---:|
| `cleaned_online_retail.csv` | All valid commercial transactions (normal sales + `C`-prefixed returns/cancellations) | 1,026,678 | 797,684 | £19,265,931.71 |
| `sales_transactions.csv` | Normal positive sales (`is_cancelled == False`, `quantity > 0`, `price > 0`) | 1,007,823 | 779,349 | £20,434,219.52 |
| `returns_and_cancellations.csv` | Valid `C`-prefixed cancellations/returns (`is_cancelled == True`, `quantity < 0`, `price > 0`) | 18,855 | 18,335 | £-1,168,287.81 |

## 2. Step-by-Step Cleaning Audit Trail

| Step | Rows Before | Rows Removed | Rows After | Why This Decision Was Made (Beginner Explanation) |
|---|---:|---:|---:|---|
| Remove exact duplicate rows | 1,067,371 | 34,335 | 1,033,036 | Eliminates combined-sheet overlap duplicates (Dec 1-9, 2010) and repeated identical rows |
| Remove 'A'-prefixed bad-debt adjustments | 1,033,036 | 6 | 1,033,030 | Internal ledger bad-debt adjustments (includes all negative Price rows), not customer orders |
| Remove TEST* stock codes | 1,033,030 | 17 | 1,033,013 | Internal software test product entries (TEST001/TEST002), not real sales |
| Remove non-C negative-quantity write-offs | 1,033,013 | 3,393 | 1,029,620 | Warehouse inventory adjustments (damages/lost/short) with Price==0 and missing Customer ID |
| Remove Price <= 0 records | 1,029,620 | 2,618 | 1,027,002 | Unpriced adjustments/promos (also resolves 100% of remaining missing Descriptions) |
| Remove anomalous C-invoice with Quantity > 0 | 1,027,002 | 1 | 1,027,001 | Single contradictory manual cancellation row (C496350) with positive quantity |
| Remove internal fee/ledger/sample StockCodes | 1,027,001 | 323 | 1,026,678 | Excludes AMAZONFEE, BANK CHARGES, ADJUST, ADJUST2, CRUK, and S (samples) |

## 3. Key Cleaning & Modeling Decisions Explained

1. **Standardized Column Names (`snake_case`):** Renamed `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country` to `invoice`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`, `customer_id`, and `country` so code is clean and consistent.
2. **Removed Exact Duplicates (`34,335` rows):** Stacking the 2009–2010 and 2010–2011 sheets caused `22,844` overlapping duplicate rows in Dec 1–9, 2010, plus `11,491` repeated identical rows elsewhere. Dropping exact duplicates prevents double-counting revenue.
3. **Kept Customer Returns/Cancellations Instead of Deleting All Negatives:** Valid `C`-prefixed invoices (`18,855` rows, `-£1,168,287.81`) represent real cancelled orders and returns. They are saved in both `cleaned_online_retail.csv` (flagged `is_cancelled=True`) and `returns_and_cancellations.csv` so net customer spend stays accurate.
4. **Removed Non-Customer Negative & Unpriced Operational Noise:** Non-`C` negative-quantity rows (`3,393` rows) all had `price == 0` and missing `customer_id` (damaged/lost stock write-offs). `A`-prefixed invoices (`6` rows) were bad-debt ledger adjustments (including all `5` negative prices). Excluding these plus `17` `TEST*` rows, `2,618` zero-price positive rows, `1` contradictory positive `C`-row (`C496350`), and `323` internal fee/sample rows (`AMAZONFEE`, `BANK CHARGES`, `ADJUST`, `ADJUST2`, `CRUK`, `S`) ensures every remaining row is a real customer-facing transaction—and automatically eliminates **100% of missing `description` values**.
5. **Preserved Missing `Customer ID` Honestly (`228,994` rows):** We did not invent customer IDs. Valid guest checkouts are kept in `cleaned_online_retail.csv` for store-wide revenue/product reporting (`has_customer_id == False`), while customer-level modeling (RFM, clustering, churn, CLV) filters strictly to `has_customer_id == True` (`797,684` rows across `5,921` unique customers, £16,294,622.05 net revenue).
6. **Added Transparency Flags:** `transaction_amount` (`quantity * price`), `is_cancelled`, `transaction_type`, `has_customer_id`, `is_merchandise` (`False` for `5,328` customer postage/carriage/discount/manual rows: `POST`, `DOT`, `C2`, `D`, `M`), and `is_unspecified_country` (`751` rows).

## 4. Final Validation Checks

- **Post-cleaning exact duplicates:** `0`
- **Missing values in non-ID columns:** `0` (including `0` missing `description`, `0` invalid dates, `0` invalid quantities/prices)
- **Unique customers (non-null):** `5,921` (`5,869` with sales, `2,553` with returns)
- **Unique StockCodes (normalized):** `4,751` (`1,021,350` merchandise rows, `5,328` customer service/fee rows)
- **Raw CSV Integrity:** Confirmed `dataset/Ecommerce.csv` SHA-256 checksum remained identical before and after processing.
