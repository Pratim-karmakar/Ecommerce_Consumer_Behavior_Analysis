"""Unit tests for modular EDA calculation helpers."""

import unittest

import pandas as pd

from src.analysis.eda_metrics import (
    compute_customer_concentration,
    compute_period_summaries,
    compute_product_pareto,
    compute_sales_kpis,
    compute_time_breakdowns,
)


class TestEdaMetrics(unittest.TestCase):
    """Verify EDA metric calculations on deterministic sample records."""

    def _sample_tx(self) -> pd.DataFrame:
        rows = [
            ("5001", "A1", "PROD A", 2, "2010-11-01 10:00:00", 10.0, 101, "United Kingdom", False, 20.0, 2010, 11, "November", 4, 1, 0, "Monday", 10, False, "2010-11"),
            ("5002", "A2", "PROD B", 1, "2010-11-07 14:00:00", 30.0, 102, "Germany", False, 30.0, 2010, 11, "November", 4, 7, 6, "Sunday", 14, True, "2010-11"),
            ("C5003", "A1", "PROD A", -1, "2010-11-08 09:00:00", 10.0, 101, "United Kingdom", True, -10.0, 2010, 11, "November", 4, 8, 0, "Monday", 9, False, "2010-11"),
        ]
        cols = [
            "invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country", "is_cancelled", "TotalAmount", "Year", "Month",
            "MonthName", "Quarter", "Day", "DayOfWeek", "DayName", "Hour", "IsWeekend", "InvoiceYearMonth",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        return df

    def test_kpis_and_periods(self) -> None:
        tx = self._sample_tx()
        kpis = compute_sales_kpis(tx)
        self.assertEqual(kpis["gross_revenue"], 50.0)
        self.assertEqual(kpis["returned_amount"], -10.0)
        self.assertEqual(kpis["net_revenue"], 40.0)
        self.assertEqual(kpis["total_orders"], 2)
        self.assertEqual(kpis["cancelled_invoices"], 1)
        self.assertEqual(kpis["net_aov"], 20.0)

        yearly, quarterly = compute_period_summaries(tx)
        self.assertEqual(len(yearly), 1)
        self.assertEqual(quarterly["YearQuarter"].iloc[0], "2010-Q4")

        tb = compute_time_breakdowns(tx)
        self.assertEqual(int(tb["by_weekend"].loc[tb["by_weekend"]["IsWeekend"], "TotalOrders"].iloc[0]), 1)

    def test_pareto_and_customer_concentration(self) -> None:
        prod = pd.DataFrame([
            ("A1", 80.0),
            ("A2", 15.0),
            ("A3", 5.0),
            ("A4", -2.0),
        ], columns=["StockCode", "TotalRevenue"])
        _, pstats = compute_product_pareto(prod)
        self.assertEqual(pstats["positive_revenue_skus"], 3)
        self.assertEqual(pstats["non_positive_revenue_skus"], 1)
        self.assertEqual(pstats["skus_for_80pct_revenue"], 2)

        cust = pd.DataFrame([
            (1, 500.0, 5, 10),
            (2, 100.0, 2, 20),
            (3, 50.0, 1, 50),
            (4, 20.0, 1, 90),
            (5, -5.0, 1, 100),
        ], columns=["Customer ID", "TotalSpending", "Frequency", "Recency"])
        cstats = compute_customer_concentration(cust)
        self.assertEqual(cstats["repeat_buyers"], 2)
        self.assertEqual(cstats["one_time_buyers"], 3)
        self.assertEqual(cstats["non_positive_spending_customers"], 1)


if __name__ == "__main__":
    unittest.main()
