"""Unit tests for the modular data cleaning pipeline."""

import unittest

import pandas as pd

from src.data.cleaning import clean_retail_transactions, split_sales_and_returns


class TestDataCleaningPipeline(unittest.TestCase):
    """Verify business rules without mutating raw files or fabricating data."""

    def _sample_raw_df(self) -> pd.DataFrame:
        rows = [
            # Valid sale + exact duplicate
            ("489434", "85048", "GLASS BALL LIGHTS ", 2, "2009-12-01 07:45:00", 6.95, 13085.0, "United Kingdom"),
            ("489434", "85048", "GLASS BALL LIGHTS ", 2, "2009-12-01 07:45:00", 6.95, 13085.0, "United Kingdom"),
            # Valid guest sale (missing Customer ID preserved, not fabricated)
            ("489435", "22041", "RECORD FRAME", 3, "2009-12-01 08:00:00", 2.10, None, "Unspecified"),
            # Customer-facing postage fee (retained, flagged is_merchandise=False)
            ("489435", "post", "POSTAGE", 1, "2009-12-01 08:00:00", 15.00, 13085.0, "United Kingdom"),
            # Valid C-prefixed cancellation (retained in cleaned + returns_and_cancellations)
            ("C489440", "85048", "GLASS BALL LIGHTS", -1, "2009-12-01 09:00:00", 6.95, 13085.0, "United Kingdom"),
            # Non-C negative quantity warehouse adjustment (excluded)
            ("489464", "21733", "damages", -10, "2009-12-01 10:00:00", 0.0, None, "United Kingdom"),
            # Bad-debt A-invoice with negative price (excluded)
            ("A506401", "B", "Adjust bad debt", 1, "2010-04-29 13:36:00", -500.0, None, "United Kingdom"),
            # Internal test code (excluded)
            ("491725", "TEST001", "This is a test product.", 5, "2009-12-14 08:34:00", 4.5, 12346.0, "United Kingdom"),
            # Internal platform fee (excluded)
            ("550000", "AMAZONFEE", "AMAZON FEE", -1, "2011-04-14 10:00:00", 100.0, None, "United Kingdom"),
            # Contradictory C-invoice with positive quantity (excluded)
            ("C496350", "M", "Manual", 1, "2010-02-01 08:24:00", 373.57, None, "United Kingdom"),
        ]
        cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
        return pd.DataFrame(rows, columns=cols)

    def test_clean_and_partition_rules(self) -> None:
        raw = self._sample_raw_df()
        raw_copy = raw.copy(deep=True)

        cleaned, audit_log = clean_retail_transactions(raw)
        # Ensure input DataFrame was never mutated in place.
        pd.testing.assert_frame_equal(raw, raw_copy)

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(int(audit_log.loc[0, "rows_removed"]), 1)

        sales, returns = split_sales_and_returns(cleaned)
        self.assertEqual(len(sales), 3)
        self.assertEqual(len(returns), 1)
        self.assertTrue((sales["transaction_amount"] > 0).all())
        self.assertTrue((returns["transaction_amount"] < 0).all())

        # Check nullable customer_id preserved without fabrication
        self.assertEqual(int(cleaned["has_customer_id"].sum()), 3)
        self.assertTrue(pd.isna(cleaned.loc[cleaned["invoice"] == "489435", "customer_id"].iloc[0]))

        # Check postage is flagged as non-merchandise while retained
        post_row = cleaned[cleaned["stock_code"] == "POST"].iloc[0]
        self.assertFalse(bool(post_row["is_merchandise"]))
        self.assertAlmostEqual(float(post_row["transaction_amount"]), 15.0)

        # Check Unspecified country flag
        self.assertEqual(int(cleaned["is_unspecified_country"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
