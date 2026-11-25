"""Integration and error-handling unit tests across all four serialized ML models."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.inference import (
    InvalidInputError,
    ModelLoadError,
    UnknownProductError,
    assign_customer_segment,
    find_similar_products,
    load_kmeans_model,
    load_knn_recommender,
    load_linear_regression_model,
    load_logistic_regression_model,
    predict_future_spending,
    predict_purchase_probability,
    recommend_products,
    validate_customer_id,
)


class TestModelIntegrationAndErrorHandling(unittest.TestCase):
    """Verify all 4 saved models load independently and handle valid and malformed inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lr_model = load_linear_regression_model()
        cls.clf_model = load_logistic_regression_model()
        cls.knn_model = load_knn_recommender()
        cls.km_model = load_kmeans_model()

    def test_valid_inference_all_four_models(self) -> None:
        valid_hist = {
            "hist_spending": 2400.0,
            "hist_frequency": 6,
            "hist_quantity": 950,
            "hist_aov": 400.0,
            "hist_unique_products": 42,
            "hist_recency": 18,
            "hist_tenure": 210,
            "hist_cancelled_orders": 1,
        }
        reg_out = predict_future_spending(valid_hist, model=self.lr_model)
        self.assertEqual(reg_out.shape, (1, 2))
        self.assertGreater(float(reg_out["predicted_spending"].iloc[0]), 0.0)

        clf_out = predict_purchase_probability(valid_hist, threshold=0.56, model=self.clf_model)
        self.assertEqual(clf_out.shape, (1, 4))
        prob = float(clf_out["purchase_probability"].iloc[0])
        self.assertTrue(0.0 <= prob <= 1.0)

        rec_known = recommend_products(14911, top_n=5, recommender=self.knn_model)
        self.assertFalse(rec_known["is_cold_start"])
        self.assertEqual(len(rec_known["recommendations"]), 5)

        rec_cold = recommend_products(999999, top_n=5, recommender=self.knn_model)
        self.assertTrue(rec_cold["is_cold_start"])
        self.assertEqual(len(rec_cold["recommendations"]), 5)

        sim_df = find_similar_products("22423", top_n=5, recommender=self.knn_model)
        self.assertEqual(len(sim_df), 5)

        seg_vip = assign_customer_segment({"Recency": 15, "Frequency": 15, "MonetaryValue": 5500.0}, model=self.km_model)
        self.assertEqual(seg_vip.iloc[0]["Segment"], "High-Value Loyal")
        self.assertEqual(int(seg_vip.iloc[0]["ClusterID"]), 0)

        seg_zero = assign_customer_segment({"Recency": 200, "Frequency": 1, "MonetaryValue": -25.0}, model=self.km_model)
        self.assertEqual(int(seg_zero.iloc[0]["ClusterID"]), -1)
        self.assertEqual(seg_zero.iloc[0]["Segment"], "Net-Returned / Zero-Spend")

    def test_missing_model_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ModelLoadError):
                load_linear_regression_model(Path(tmpdir))
            with self.assertRaises(ModelLoadError):
                load_knn_recommender(Path(tmpdir))

    def test_invalid_customer_ids(self) -> None:
        self.assertEqual(validate_customer_id("14911"), 14911)
        self.assertEqual(validate_customer_id(14911.0), 14911)
        for bad_id in (None, True, -5, 0, "abc", 12.34, float("nan")):
            with self.assertRaises(InvalidInputError):
                validate_customer_id(bad_id)
        with self.assertRaises(InvalidInputError):
            recommend_products("not_an_id", recommender=self.knn_model)

    def test_unknown_products(self) -> None:
        with self.assertRaises(UnknownProductError):
            find_similar_products("NO_SUCH_SKU_999", strict=True, recommender=self.knn_model)
        empty_df = find_similar_products("NO_SUCH_SKU_999", strict=False, recommender=self.knn_model)
        self.assertTrue(empty_df.empty)

    def test_missing_feature_values_handling(self) -> None:
        # Regression & Classification pipelines impute missing values via SimpleImputer
        partial_hist = {
            "hist_spending": 1200.0,
            "hist_frequency": 3,
            "hist_quantity": np.nan,
            "hist_aov": None,
            "hist_unique_products": 20,
            "hist_recency": 45,
            "hist_tenure": 180,
            "hist_cancelled_orders": 0,
        }
        reg_out = predict_future_spending(partial_hist, model=self.lr_model)
        clf_out = predict_purchase_probability(partial_hist, model=self.clf_model)
        self.assertFalse(reg_out.isna().any().any())
        self.assertFalse(clf_out.isna().any().any())

        # K-Means explicitly rejects missing RFM values
        with self.assertRaises(InvalidInputError):
            assign_customer_segment({"Recency": 20, "Frequency": None, "MonetaryValue": 500.0}, model=self.km_model)

    def test_unexpected_input_formats(self) -> None:
        for bad_input in ("not a dict", [1, 2, 3], {}, pd.DataFrame()):
            with self.assertRaises(InvalidInputError):
                predict_future_spending(bad_input, model=self.lr_model)

        # Missing required column
        with self.assertRaises(InvalidInputError):
            predict_purchase_probability({"hist_spending": 100.0}, model=self.clf_model)

        # Non-numeric string inside numeric feature or inf
        with self.assertRaises(InvalidInputError):
            assign_customer_segment({"Recency": "bad", "Frequency": 2, "MonetaryValue": 300.0}, model=self.km_model)
        with self.assertRaises(InvalidInputError):
            assign_customer_segment({"Recency": -1, "Frequency": 2, "MonetaryValue": 300.0}, model=self.km_model)
        with self.assertRaises(InvalidInputError):
            predict_future_spending(
                {
                    "hist_spending": float("inf"),
                    "hist_frequency": 2,
                    "hist_quantity": 10,
                    "hist_aov": 50.0,
                    "hist_unique_products": 5,
                    "hist_recency": 10,
                    "hist_tenure": 20,
                    "hist_cancelled_orders": 0,
                },
                model=self.lr_model,
            )


if __name__ == "__main__":
    unittest.main()
