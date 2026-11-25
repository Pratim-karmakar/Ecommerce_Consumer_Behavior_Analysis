"""Unit tests for the Item-Based KNN product recommender and holdout evaluation."""

import unittest

import pandas as pd

from src.models.recommender import (
    ItemKNNRecommender,
    UserKNNRecommender,
    build_interaction_table,
    evaluate_recommender_holdout,
)


class TestKNNRecommender(unittest.TestCase):
    """Verify sparse interaction aggregation, exclusion, cold-start, and metrics."""

    def _sample_sales(self) -> pd.DataFrame:
        rows = [
            # Customer 1 & 2 share A and B; Customer 2 also bought C -> C recommended to Customer 1
            ("1001", "SKU_A", "ITEM A", 10, "2011-01-01 10:00:00", 2.0, 1, False, True, True, 20.0),
            ("1001", "SKU_B", "ITEM B", 5, "2011-01-01 10:00:00", 3.0, 1, False, True, True, 15.0),
            ("1002", "SKU_A", "ITEM A", 4, "2011-02-01 10:00:00", 2.0, 2, False, True, True, 8.0),
            ("1002", "SKU_B", "ITEM B", 2, "2011-02-01 10:00:00", 3.0, 2, False, True, True, 6.0),
            ("1002", "SKU_C", "ITEM C", 6, "2011-02-01 10:00:00", 4.0, 2, False, True, True, 24.0),
            # Customer 3 buys D and C -> connects C & D
            ("1003", "SKU_C", "ITEM C", 3, "2011-03-01 10:00:00", 4.0, 3, False, True, True, 12.0),
            ("1003", "SKU_D", "ITEM D", 1, "2011-03-01 10:00:00", 5.0, 3, False, True, True, 5.0),
            # Non-merchandise postage must be excluded from recommendations
            ("1002", "POST", "POSTAGE", 1, "2011-02-01 10:00:00", 15.0, 2, False, True, False, 15.0),
            # Cancelled return must be excluded
            ("C1004", "SKU_X", "ITEM X", -2, "2011-03-05 10:00:00", 5.0, 2, True, True, True, -10.0),
            # Holdout purchases after 2011-09-11
            ("2001", "SKU_C", "ITEM C", 2, "2011-10-01 10:00:00", 4.0, 1, False, True, True, 8.0),
            ("2002", "SKU_D", "ITEM D", 1, "2011-10-05 10:00:00", 5.0, 2, False, True, True, 5.0),
        ]
        cols = [
            "invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "is_cancelled", "has_customer_id", "is_merchandise", "TotalAmount",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["customer_id"] = df["customer_id"].astype("Int64")
        return df

    def test_recommender_and_cold_start(self) -> None:
        df = self._sample_sales()
        train = df[df["invoice_date"] < "2011-09-11"]
        holdout = df[df["invoice_date"] >= "2011-09-11"]

        inter = build_interaction_table(train)
        self.assertNotIn("POST", set(inter["stock_code"]))
        self.assertNotIn("SKU_X", set(inter["stock_code"]))

        rec = ItemKNNRecommender(n_neighbors=5, min_item_customers=1).fit(train)
        recs_c1 = rec.recommend_for_customer(1, top_n=2, exclude_purchased=True)
        self.assertEqual(recs_c1.iloc[0]["stock_code"], "SKU_C")
        self.assertNotIn("SKU_A", set(recs_c1["stock_code"]))

        # Verify min_item_customers filters low-support SKUs cleanly
        rec_filtered = ItemKNNRecommender(n_neighbors=5, min_item_customers=2).fit(train)
        self.assertNotIn("SKU_D", set(rec_filtered.item_ids_))
        self.assertIn("SKU_A", set(rec_filtered.item_ids_))

        # Cold-start unseen customer receives fallback popularity items
        recs_unseen = rec.recommend_for_customer(99999, top_n=2, exclude_purchased=True)
        self.assertEqual(len(recs_unseen), 2)
        self.assertTrue((recs_unseen["source"] == "popularity_fallback").all())

        # Similar items lookup
        sim_a = rec.find_similar_items("SKU_A", top_n=2)
        self.assertEqual(sim_a.iloc[0]["stock_code"], "SKU_B")

        # Holdout evaluation
        eval_df = evaluate_recommender_holdout(train, holdout, rec, "Item-KNN", rec.popular_items_, k_values=(1, 2), min_train_items=2)
        self.assertEqual(len(eval_df), 2)
        self.assertGreater(float(eval_df.loc[eval_df["k"] == 1, "hit_rate_at_k"].iloc[0]), 0.0)

        # User-Based KNN comparison helper
        uknn = UserKNNRecommender(n_neighbors=2).fit(train)
        self.assertIsInstance(uknn.recommend_codes(1, top_n=2), list)


if __name__ == "__main__":
    unittest.main()
