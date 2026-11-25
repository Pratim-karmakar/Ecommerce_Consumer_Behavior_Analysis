"""Sparse Item-Based & User-Based KNN collaborative filtering on real purchase interactions."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_interaction_table(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate valid identified merchandise purchases into unique (customer_id, stock_code) weights.

    Excludes cancellations and non-merchandise service codes (POST, DOT, C2, D, M), then applies
    log1p(total_quantity) so wholesale bulk quantities do not crush cosine similarity.
    """
    valid = sales_df[
        sales_df["has_customer_id"] & (~sales_df["is_cancelled"]) & sales_df["is_merchandise"]
    ].copy()

    agg = valid.groupby(["customer_id", "stock_code"], as_index=False).agg(
        total_quantity=("quantity", "sum"),
        order_count=("invoice", "nunique"),
        total_spent=("TotalAmount", "sum"),
    )
    agg = agg[agg["total_quantity"] > 0].reset_index(drop=True)
    agg["interaction_weight"] = np.log1p(agg["total_quantity"]).astype("float64")
    return agg


def build_sparse_matrix(
    interactions_df: pd.DataFrame,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, Dict[int, int], Dict[str, int]]:
    """Build sparse customer x item CSR matrix and bidirectional ID maps."""
    cust_ids = np.sort(interactions_df["customer_id"].unique())
    item_ids = np.sort(interactions_df["stock_code"].unique())

    cust_to_idx = {int(c): idx for idx, c in enumerate(cust_ids)}
    item_to_idx = {str(s): idx for idx, s in enumerate(item_ids)}

    row_idx = interactions_df["customer_id"].map(cust_to_idx).to_numpy()
    col_idx = interactions_df["stock_code"].map(item_to_idx).to_numpy()
    weights = interactions_df["interaction_weight"].to_numpy(dtype="float64")

    matrix = csr_matrix((weights, (row_idx, col_idx)), shape=(len(cust_ids), len(item_ids)))
    return matrix, cust_ids, item_ids, cust_to_idx, item_to_idx


class ItemKNNRecommender:
    """Item-Based Collaborative Filtering recommender using Cosine KNN over sparse purchases."""

    def __init__(
        self, n_neighbors: int = 25, min_history_for_cf: int = 1, min_item_customers: int = 5
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_history_for_cf = min_history_for_cf
        self.min_item_customers = min_item_customers
        self.matrix_: Optional[csr_matrix] = None
        self.customer_ids_: Optional[np.ndarray] = None
        self.item_ids_: Optional[np.ndarray] = None
        self.cust_to_idx_: Dict[int, int] = {}
        self.item_to_idx_: Dict[str, int] = {}
        self.item_neighbors_: Optional[np.ndarray] = None
        self.item_similarities_: Optional[np.ndarray] = None
        self.item_catalog_: Optional[pd.DataFrame] = None
        self.popular_items_: List[str] = []
        self.metadata_: Dict[str, Any] = {}

    def fit(self, sales_df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> "ItemKNNRecommender":
        """Fit sparse item-item Cosine KNN index and product metadata catalog."""
        interactions_raw = build_interaction_table(sales_df)
        # Filter ultra-rare SKUs (< min_item_customers buyers) to prevent spurious cosine=1.0 ties.
        if self.min_item_customers > 1:
            item_support = interactions_raw.groupby("stock_code")["customer_id"].count()
            supported_skus = set(item_support[item_support >= self.min_item_customers].index)
            interactions = interactions_raw[interactions_raw["stock_code"].isin(supported_skus)].reset_index(drop=True)
        else:
            interactions = interactions_raw

        matrix, cust_ids, item_ids, cust_to_idx, item_to_idx = build_sparse_matrix(interactions)

        # Build item catalog (mode description, reach, average selling price)
        valid_sales = sales_df[
            sales_df["has_customer_id"]
            & (~sales_df["is_cancelled"])
            & sales_df["is_merchandise"]
            & sales_df["stock_code"].isin(set(item_ids))
        ]
        desc_map = (
            valid_sales.groupby("stock_code")["description"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0]))
        )
        cat = valid_sales.groupby("stock_code", as_index=False).agg(
            customer_reach=("customer_id", "nunique"),
            total_units=("quantity", "sum"),
            total_revenue=("TotalAmount", "sum"),
        )
        cat["description"] = cat["stock_code"].map(desc_map).astype(str)
        cat["avg_selling_price"] = (cat["total_revenue"] / cat["total_units"].clip(lower=1)).round(2)
        cat = cat.set_index("stock_code")

        k_eff = min(self.n_neighbors + 1, matrix.shape[1])
        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff)
        nn.fit(matrix.T)
        distances, indices = nn.kneighbors(matrix.T)

        self.matrix_ = matrix
        self.customer_ids_ = cust_ids
        self.item_ids_ = item_ids
        self.cust_to_idx_ = cust_to_idx
        self.item_to_idx_ = item_to_idx
        self.item_neighbors_ = indices
        self.item_similarities_ = np.clip(1.0 - distances, 0.0, 1.0)
        self.item_catalog_ = cat
        self.popular_items_ = cat.sort_values(["customer_reach", "total_revenue"], ascending=False).index.tolist()
        self.metadata_ = metadata or {
            "n_customers": int(matrix.shape[0]),
            "n_items": int(matrix.shape[1]),
            "n_interactions": int(matrix.nnz),
            "sparsity_pct": round((1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100, 4),
        }
        return self

    def get_customer_history(self, customer_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return a customer's top historical purchased items ranked by interaction weight."""
        cid = int(customer_id)
        if cid not in self.cust_to_idx_ or self.matrix_ is None or self.item_catalog_ is None:
            return pd.DataFrame(columns=["stock_code", "description", "interaction_weight", "avg_selling_price"])

        uidx = self.cust_to_idx_[cid]
        row = self.matrix_[uidx]
        records = []
        for idx, wt in zip(row.indices, row.data):
            code = str(self.item_ids_[idx])
            info = self.item_catalog_.loc[code]
            records.append(
                {
                    "stock_code": code,
                    "description": str(info["description"]),
                    "interaction_weight": round(float(wt), 4),
                    "est_quantity": int(round(np.expm1(wt))),
                    "avg_selling_price": float(info["avg_selling_price"]),
                }
            )
        hist_df = pd.DataFrame(records).sort_values("interaction_weight", ascending=False).head(top_n)
        return hist_df.reset_index(drop=True)

    def find_similar_items(self, stock_code: str, top_n: int = 10) -> pd.DataFrame:
        """Return the most co-purchased similar merchandise items for a given StockCode."""
        code = str(stock_code).strip().upper()
        if code not in self.item_to_idx_ or self.item_catalog_ is None:
            return pd.DataFrame(columns=["stock_code", "description", "cosine_similarity", "customer_reach", "avg_selling_price"])

        j = self.item_to_idx_[code]
        nbr_idxs = self.item_neighbors_[j]
        nbr_sims = self.item_similarities_[j]

        rows = []
        for nb_idx, sim in zip(nbr_idxs, nbr_sims):
            nb_code = str(self.item_ids_[nb_idx])
            if nb_code == code or sim <= 0:
                continue
            info = self.item_catalog_.loc[nb_code]
            rows.append(
                {
                    "stock_code": nb_code,
                    "description": str(info["description"]),
                    "cosine_similarity": round(float(sim), 4),
                    "customer_reach": int(info["customer_reach"]),
                    "avg_selling_price": float(info["avg_selling_price"]),
                }
            )
            if len(rows) >= top_n:
                break
        return pd.DataFrame(rows)

    def recommend_for_customer(
        self, customer_id: int, top_n: int = 10, exclude_purchased: bool = True
    ) -> pd.DataFrame:
        """
        Generate top-N product recommendations for a customer with transparent cold-start fallback.

        If a customer is unseen or has fewer items than `min_history_for_cf`, fills remaining
        slots with popular merchandise items not already purchased by that customer.
        """
        if self.matrix_ is None or self.item_catalog_ is None:
            raise RuntimeError("Recommender must be fitted before generating recommendations.")

        cid = int(customer_id)
        seen_codes: Set[str] = set()
        recs: List[Dict[str, Any]] = []

        if cid in self.cust_to_idx_:
            uidx = self.cust_to_idx_[cid]
            user_row = self.matrix_[uidx]
            u_items = user_row.indices
            u_weights = user_row.data
            seen_codes = {str(self.item_ids_[i]) for i in u_items}

            if len(u_items) >= self.min_history_for_cf:
                scores = np.zeros(len(self.item_ids_), dtype=np.float64)
                top_driver: Dict[int, Tuple[float, str]] = {}

                for j, wt in zip(u_items, u_weights):
                    src_code = str(self.item_ids_[j])
                    for nb_idx, sim in zip(self.item_neighbors_[j], self.item_similarities_[j]):
                        if nb_idx == j or sim <= 0:
                            continue
                        contrib = float(wt * sim)
                        scores[nb_idx] += contrib
                        if nb_idx not in top_driver or contrib > top_driver[nb_idx][0]:
                            top_driver[nb_idx] = (contrib, src_code)

                if exclude_purchased:
                    scores[u_items] = -1.0

                candidate_idxs = np.where(scores > 0)[0]
                if len(candidate_idxs) > 0:
                    sorted_idxs = candidate_idxs[np.argsort(scores[candidate_idxs])[::-1]][:top_n]
                    for idx in sorted_idxs:
                        code = str(self.item_ids_[idx])
                        info = self.item_catalog_.loc[code]
                        driver_code = top_driver[idx][1]
                        driver_desc = str(self.item_catalog_.loc[driver_code, "description"])
                        recs.append(
                            {
                                "stock_code": code,
                                "description": str(info["description"]),
                                "score": round(float(scores[idx]), 4),
                                "source": "item_knn",
                                "reason": f"Co-purchased with {driver_code} ({driver_desc[:32]})",
                                "customer_reach": int(info["customer_reach"]),
                                "avg_selling_price": float(info["avg_selling_price"]),
                            }
                        )

        # Cold-start or sparse backfill with global merchandise popularity
        used_codes = {r["stock_code"] for r in recs} | (seen_codes if exclude_purchased else set())
        if len(recs) < top_n:
            reason_msg = (
                "Cold-start popularity fallback (no purchase history)"
                if cid not in self.cust_to_idx_
                else "Popularity backfill (sparse purchase history)"
            )
            for code in self.popular_items_:
                if code in used_codes:
                    continue
                info = self.item_catalog_.loc[code]
                recs.append(
                    {
                        "stock_code": code,
                        "description": str(info["description"]),
                        "score": 0.0,
                        "source": "popularity_fallback",
                        "reason": reason_msg,
                        "customer_reach": int(info["customer_reach"]),
                        "avg_selling_price": float(info["avg_selling_price"]),
                    }
                )
                used_codes.add(code)
                if len(recs) >= top_n:
                    break

        out_df = pd.DataFrame(recs[:top_n])
        out_df.insert(0, "rank", range(1, len(out_df) + 1))
        return out_df


class UserKNNRecommender:
    """Customer-Based Collaborative Filtering recommender for empirical comparison against Item-KNN."""

    def __init__(self, n_neighbors: int = 25) -> None:
        self.n_neighbors = n_neighbors
        self.matrix_: Optional[csr_matrix] = None
        self.item_ids_: Optional[np.ndarray] = None
        self.cust_to_idx_: Dict[int, int] = {}
        self.user_neighbors_: Optional[np.ndarray] = None
        self.user_similarities_: Optional[np.ndarray] = None

    def fit(self, sales_df: pd.DataFrame) -> "UserKNNRecommender":
        """Build the customer-item interaction matrix and fit user cosine neighbors."""
        interactions = build_interaction_table(sales_df)
        matrix, _, item_ids, cust_to_idx, _ = build_sparse_matrix(interactions)
        k_eff = min(self.n_neighbors + 1, matrix.shape[0])
        nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff)
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix)
        self.matrix_ = matrix
        self.item_ids_ = item_ids
        self.cust_to_idx_ = cust_to_idx
        self.user_neighbors_ = indices
        self.user_similarities_ = np.clip(1.0 - distances, 0.0, 1.0)
        return self

    def recommend_codes(self, customer_id: int, top_n: int = 10, exclude_purchased: bool = True) -> List[str]:
        """Return top-N StockCode recommendations ranked by neighbor purchase similarity."""
        cid = int(customer_id)
        if cid not in self.cust_to_idx_ or self.matrix_ is None:
            return []
        uidx = self.cust_to_idx_[cid]
        nb_idxs = self.user_neighbors_[uidx, 1:]
        nb_sims = self.user_similarities_[uidx, 1:]
        scores = np.asarray(self.matrix_[nb_idxs].T.dot(nb_sims)).ravel()
        if exclude_purchased:
            scores[self.matrix_[uidx].indices] = -1.0
        pos_idxs = np.where(scores > 0)[0]
        if len(pos_idxs) == 0:
            return []
        top_idxs = pos_idxs[np.argsort(scores[pos_idxs])[::-1]][:top_n]
        return [str(self.item_ids_[i]) for i in top_idxs]


def evaluate_recommender_holdout(
    train_sales_df: pd.DataFrame,
    holdout_sales_df: pd.DataFrame,
    recommender: ItemKNNRecommender | UserKNNRecommender | None,
    model_name: str,
    popular_rank: List[str],
    k_values: Tuple[int, ...] = (5, 10),
    exclude_seen: bool = True,
    min_train_items: int = 2,
) -> pd.DataFrame:
    """Evaluate Precision@K, Recall@K, Hit Rate@K, and Catalog Coverage on a chronological holdout."""
    train_inter = build_interaction_table(train_sales_df)
    hold_valid = holdout_sales_df[
        holdout_sales_df["has_customer_id"] & (~holdout_sales_df["is_cancelled"]) & holdout_sales_df["is_merchandise"]
    ]
    train_items_map = train_inter.groupby("customer_id")["stock_code"].apply(set).to_dict()
    hold_items_map = hold_valid.groupby("customer_id")["stock_code"].apply(set).to_dict()

    eligible_cids = [
        int(cid)
        for cid, items in train_items_map.items()
        if len(items) >= min_train_items and int(cid) in hold_items_map
    ]
    catalog_size = int(train_inter["stock_code"].nunique())

    max_k = max(k_values)
    # Precompute top max_k recommendations per eligible customer once
    recs_by_cust: Dict[int, List[str]] = {}
    gt_by_cust: Dict[int, Set[str]] = {}

    for cid in eligible_cids:
        seen = train_items_map[cid]
        gt = (hold_items_map[cid] - seen) if exclude_seen else hold_items_map[cid]
        if not gt:
            continue
        if isinstance(recommender, ItemKNNRecommender):
            rec_codes = recommender.recommend_for_customer(cid, top_n=max_k, exclude_purchased=exclude_seen)["stock_code"].tolist()
        elif isinstance(recommender, UserKNNRecommender):
            rec_codes = recommender.recommend_codes(cid, top_n=max_k, exclude_purchased=exclude_seen)
            if len(rec_codes) < max_k:
                used = set(rec_codes) | (seen if exclude_seen else set())
                for s in popular_rank:
                    if s not in used:
                        rec_codes.append(s)
                        if len(rec_codes) >= max_k:
                            break
        else:
            used = seen if exclude_seen else set()
            rec_codes = [s for s in popular_rank if s not in used][:max_k]

        recs_by_cust[cid] = rec_codes
        gt_by_cust[cid] = gt

    rows = []
    for k in k_values:
        precs, recs_list, hits = [], [], []
        covered: Set[str] = set()
        for cid, rec_codes in recs_by_cust.items():
            topk = rec_codes[:k]
            covered.update(topk)
            gt = gt_by_cust[cid]
            n_hits = len(set(topk) & gt)
            precs.append(n_hits / k)
            recs_list.append(n_hits / len(gt))
            hits.append(1.0 if n_hits > 0 else 0.0)

        rows.append(
            {
                "model_name": model_name,
                "k": k,
                "exclude_seen": exclude_seen,
                "n_eval_customers": len(recs_by_cust),
                "precision_at_k": round(float(np.mean(precs)), 4),
                "recall_at_k": round(float(np.mean(recs_list)), 4),
                "hit_rate_at_k": round(float(np.mean(hits)), 4),
                "coverage_pct": round(len(covered) / catalog_size * 100.0, 2),
                "unique_items_recommended": len(covered),
            }
        )
    return pd.DataFrame(rows)
