"""Unit tests for modular feature engineering functions."""

import unittest

import pandas as pd

from src.features.engineering import (
    add_transaction_features,
    build_country_analytics,
    build_customer_analytics,
    build_monthly_sales,
    build_product_analytics,
)


class TestFeatureEngineering(unittest.TestCase):
    """Validate transaction, customer, product, country, and monthly aggregations."""

    def _sample_cleaned_df(self) -> pd.DataFrame:
        rows = [
            # Saturday 2011-12-03 10:15:00 — customer 1001 buys 2 units at 10.0
            ("500001", "85048", "GLASS BALL", 2, "2011-12-03 10:15:00", 10.0, 1001, "United Kingdom", 20.0, False, "sale", True, True, False),
            # Sunday 2011-12-04 14:30:00 — customer 1001 buys postage (non-merchandise) + product
            ("500002", "22041", "RECORD FRAME", 1, "2011-12-04 14:30:00", 5.0, 1001, "United Kingdom", 5.0, False, "sale", True, True, False),
            ("500002", "POST", "POSTAGE", 1, "2011-12-04 14:30:00", 15.0, 1001, "United Kingdom", 15.0, False, "sale", True, False, False),
            # Monday 2011-12-05 09:00:00 — customer 1001 cancels 1 GLASS BALL (-10.0)
            ("C500003", "85048", "GLASS BALL", -1, "2011-12-05 09:00:00", 10.0, 1001, "United Kingdom", -10.0, True, "cancellation_or_return", True, True, False),
            # Guest checkout (missing customer_id) in France
            ("500004", "22041", "RECORD FRAME", 4, "2011-12-05 16:00:00", 5.0, pd.NA, "France", 20.0, False, "sale", False, True, False),
            # Return-only customer 1002
            ("C500005", "22041", "RECORD FRAME", -1, "2011-12-05 17:00:00", 5.0, 1002, "France", -5.0, True, "cancellation_or_return", True, True, False),
        ]
        cols = [
            "invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country", "transaction_amount", "is_cancelled",
            "transaction_type", "has_customer_id", "is_merchandise", "is_unspecified_country",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["customer_id"] = df["customer_id"].astype("Int64")
        return df

    def test_all_feature_tables(self) -> None:
        cleaned = self._sample_cleaned_df()
        tx = add_transaction_features(cleaned)

        self.assertEqual(len(tx), 6)
        self.assertTrue(bool(tx.loc[0, "IsWeekend"]))  # Saturday
        self.assertTrue(bool(tx.loc[1, "IsWeekend"]))  # Sunday
        self.assertFalse(bool(tx.loc[3, "IsWeekend"]))  # Monday
        self.assertEqual(tx.loc[0, "InvoiceYearMonth"], "2011-12")
        self.assertEqual(int(tx.loc[0, "Quarter"]), 4)

        cust, ref_date, meta = build_customer_analytics(tx)
        self.assertEqual(ref_date, pd.Timestamp("2011-12-06"))
        self.assertEqual(meta["return_only_customers_excluded"], 1)
        self.assertEqual(len(cust), 1)
        row = cust.iloc[0]
        self.assertEqual(int(row["Customer ID"]), 1001)
        self.assertEqual(int(row["TotalOrders"]), 2)
        self.assertEqual(int(row["CancelledOrders"]), 1)
        self.assertAlmostEqual(float(row["TotalSpending"]), 30.0)  # 40 gross - 10 return
        self.assertEqual(int(row["TotalQuantity"]), 3)  # 2 + 1 + 1 - 1
        self.assertEqual(int(row["Recency"]), 2)  # 2011-12-06 - 2011-12-04

        prod = build_product_analytics(tx)
        # POST excluded because is_merchandise == False
        self.assertEqual(set(prod["StockCode"]), {"85048", "22041"})
        self.assertAlmostEqual(float(prod["TotalRevenue"].sum()), 30.0)

        ctry = build_country_analytics(tx)
        self.assertEqual(len(ctry), 2)
        self.assertAlmostEqual(float(ctry["TotalRevenue"].sum()), float(tx["TotalAmount"].sum()))

        monthly = build_monthly_sales(tx)
        self.assertEqual(len(monthly), 1)
        self.assertAlmostEqual(float(monthly["TotalRevenue"].iloc[0]), 45.0)
        self.assertEqual(int(monthly["UniqueCustomers"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
