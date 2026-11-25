"""Unit tests for leakage-free temporal cohorts and Linear Regression pipeline."""

import unittest

import pandas as pd

from src.models.regression import (
    FEATURE_COLUMNS,
    build_temporal_cohort,
    evaluate_regression_suite,
)


class TestTemporalRegression(unittest.TestCase):
    """Ensure strict observation/target separation and valid model training."""

    def _sample_transactions(self) -> pd.DataFrame:
        rows = [
            # Train obs (2010-01 to 2010-09-10)
            ("1001", "A1", 2, 10.0, "2010-03-01 10:00:00", 1, False, True, 20.0),
            ("1002", "A2", 5, 10.0, "2010-05-01 10:00:00", 2, False, True, 50.0),
            ("1003", "A1", 10, 10.0, "2010-06-01 10:00:00", 3, False, True, 100.0),
            ("C1004", "A1", -1, 10.0, "2010-06-05 10:00:00", 3, True, True, -10.0),
            # Train target (2010-09-11 to 2010-12-09)
            ("1005", "A1", 1, 10.0, "2010-10-01 10:00:00", 1, False, True, 10.0),
            ("1006", "A2", 3, 10.0, "2010-11-01 10:00:00", 2, False, True, 30.0),
            ("1007", "A1", 5, 10.0, "2010-11-15 10:00:00", 3, False, True, 50.0),
            # Test obs (2010-12-10 to 2011-09-10)
            ("2001", "A1", 3, 10.0, "2011-03-01 10:00:00", 1, False, True, 30.0),
            ("2002", "A2", 6, 10.0, "2011-05-01 10:00:00", 2, False, True, 60.0),
            ("2003", "A1", 9, 10.0, "2011-06-01 10:00:00", 3, False, True, 90.0),
            # Test target (2011-09-11 to 2011-12-09)
            ("2004", "A1", 2, 10.0, "2011-10-01 10:00:00", 1, False, True, 20.0),
            ("2005", "A2", 4, 10.0, "2011-11-01 10:00:00", 2, False, True, 40.0),
            # Customer 3 makes 0 future purchases in test target -> future_spending == 0.0
            # Guest checkout (ignored for customer prediction)
            ("2006", "A1", 50, 10.0, "2011-11-10 10:00:00", pd.NA, False, False, 500.0),
        ]
        cols = [
            "invoice", "stock_code", "quantity", "price", "invoice_date",
            "customer_id", "is_cancelled", "has_customer_id", "TotalAmount",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["customer_id"] = df["customer_id"].astype("Int64")
        return df

    def test_no_leakage_guard(self) -> None:
        df = self._sample_transactions()
        with self.assertRaises(ValueError):
            build_temporal_cohort(df, "2010-01-01", "2010-10-01", "2010-09-11", "2010-12-10")

    def test_cohort_features_and_evaluation(self) -> None:
        df = self._sample_transactions()
        train_df = build_temporal_cohort(df, "2009-12-10", "2010-09-11", "2010-09-11", "2010-12-10")
        test_df = build_temporal_cohort(df, "2010-12-10", "2011-09-11", "2011-09-11", "2011-12-10")

        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(test_df), 3)
        # Customer 3 had 100 - 10 = 90 historical spending in train obs
        c3_train = train_df[train_df["customer_id"] == 3].iloc[0]
        self.assertAlmostEqual(float(c3_train["hist_spending"]), 90.0)
        self.assertAlmostEqual(float(c3_train["future_spending"]), 50.0)
        # Customer 3 had 0 future spending in test target
        self.assertAlmostEqual(float(test_df.loc[test_df["customer_id"] == 3, "future_spending"].iloc[0]), 0.0)

        pipe, eval_df, pred_df = evaluate_regression_suite(train_df, test_df, FEATURE_COLUMNS)
        self.assertEqual(len(pred_df), 3)
        self.assertIn("mae", eval_df.columns)
        self.assertIn("rmse", eval_df.columns)
        self.assertIn("r2_score", eval_df.columns)


if __name__ == "__main__":
    unittest.main()
