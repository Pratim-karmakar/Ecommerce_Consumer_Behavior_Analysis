"""Unit tests for RFM K-Means customer segmentation."""

import unittest

import pandas as pd

from src.models.segmentation import evaluate_k_range, fit_and_segment_customers


class TestKMeansSegmentation(unittest.TestCase):
    """Verify RFM scaling, K evaluation, deterministic labeling, and non-positive isolation."""

    def _sample_customers(self) -> pd.DataFrame:
        rows = [
            # High-value loyal cluster
            (1, 5, 20, 5000.0, 5000.0),
            (2, 10, 18, 4500.0, 4500.0),
            (3, 8, 25, 6000.0, 6000.0),
            # Recent growing cluster
            (4, 15, 3, 600.0, 600.0),
            (5, 20, 4, 750.0, 750.0),
            (6, 12, 3, 550.0, 550.0),
            # At-risk established cluster
            (7, 200, 6, 1500.0, 1500.0),
            (8, 220, 5, 1300.0, 1300.0),
            (9, 180, 7, 1700.0, 1700.0),
            # Dormant low-engagement cluster
            (10, 420, 1, 200.0, 200.0),
            (11, 450, 1, 180.0, 180.0),
            (12, 400, 1, 220.0, 220.0),
            # Net-returned / zero-spend edge case
            (13, 300, 1, -50.0, -50.0),
        ]
        return pd.DataFrame(rows, columns=["Customer ID", "Recency", "Frequency", "MonetaryValue", "TotalSpending"])

    def test_k_evaluation_and_segmentation(self) -> None:
        cust = self._sample_customers()
        k_df, _ = evaluate_k_range(cust[cust["MonetaryValue"] > 0], k_values=range(2, 6))
        self.assertEqual(len(k_df), 4)
        self.assertTrue((k_df["silhouette_score"] > 0).all())

        pipe, segmented, summary, pca = fit_and_segment_customers(cust, n_clusters=4)
        self.assertEqual(len(segmented), 13)
        self.assertEqual(int((segmented["ClusterID"] == -1).sum()), 1)
        self.assertEqual(set(summary["Segment"]), {
            "Net-Returned / Zero-Spend",
            "High-Value Loyal",
            "At-Risk Established",
            "Recent / Growing",
            "Dormant / Low-Engagement",
        })
        self.assertEqual(len(pca.explained_variance_ratio_), 2)


if __name__ == "__main__":
    unittest.main()
