"""Unit tests for leakage-free purchase classification cohorts and Logistic Regression."""

import unittest

import pandas as pd

from src.models.classification import (
    build_purchase_classification_cohort,
    evaluate_classification_suite,
)
from src.models.regression import FEATURE_COLUMNS


class TestPurchaseClassification(unittest.TestCase):
    """Verify positive/negative label creation, completeness guard, and pipeline metrics."""

    def _sample_tx(self) -> pd.DataFrame:
        rows = []
        # Create 12 synthetic in-memory customer histories (6 repeat buyers, 6 non-buyers)
        for cid in range(1, 13):
            amt = 200.0 if cid <= 6 else 25.0
            freq_dates = ["2010-02-01 10:00:00", "2010-06-01 10:00:00"] if cid <= 6 else ["2010-01-15 10:00:00"]
            for idx, d in enumerate(freq_dates):
                rows.append((f"10{cid}{idx}", "A1", 2, amt / 2, d, cid, False, True, amt))
            if cid <= 6:
                rows.append((f"20{cid}0", "A1", 2, 50.0, "2010-10-15 10:00:00", cid, False, True, 100.0))

        # Test cohort (2010-12-10 .. 2011-09-10 -> 2011-09-11 .. 2011-12-09)
        for cid in range(1, 13):
            amt = 220.0 if cid <= 6 else 30.0
            freq_dates = ["2011-02-01 10:00:00", "2011-07-01 10:00:00"] if cid <= 6 else ["2011-01-10 10:00:00"]
            for idx, d in enumerate(freq_dates):
                rows.append((f"30{cid}{idx}", "A1", 2, amt / 2, d, cid, False, True, amt))
            if cid <= 6:
                rows.append((f"40{cid}0", "A1", 2, 55.0, "2011-11-20 10:00:00", cid, False, True, 110.0))

        # Final date marker covering Dec 9, 2011
        rows.append(("49999", "A1", 1, 10.0, "2011-12-09 12:00:00", 1, False, True, 10.0))
        cols = [
            "invoice", "stock_code", "quantity", "price", "invoice_date",
            "customer_id", "is_cancelled", "has_customer_id", "TotalAmount",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["customer_id"] = df["customer_id"].astype("Int64")
        return df

    def test_incomplete_window_guard(self) -> None:
        df = self._sample_tx()
        with self.assertRaises(ValueError):
            # Target ends 2011-12-31, past dataset max 2011-12-09 -> incomplete!
            build_purchase_classification_cohort(df, "2010-12-10", "2011-09-11", "2011-09-11", "2012-01-01")

    def test_classification_suite(self) -> None:
        df = self._sample_tx()
        tr = build_purchase_classification_cohort(df, "2009-12-10", "2010-09-11", "2010-09-11", "2010-12-10")
        te = build_purchase_classification_cohort(df, "2010-12-10", "2011-09-11", "2011-09-11", "2011-12-10")

        self.assertEqual(int(tr["will_purchase"].sum()), 6)
        self.assertEqual(int((tr["will_purchase"] == 0).sum()), 6)
        pipe, eval_df, thresholds, pred_df = evaluate_classification_suite(tr, te, FEATURE_COLUMNS)
        self.assertEqual(len(pred_df), 12)
        test_rows = eval_df[eval_df["split"] == "test"]
        self.assertGreater(float(test_rows["roc_auc"].max()), 0.9)


if __name__ == "__main__":
    unittest.main()
