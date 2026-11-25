"""RFM customer segmentation pipeline using log1p scaling, PCA projection, and K-Means."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RFM_COLUMNS: List[str] = ["Recency", "Frequency", "MonetaryValue"]


def evaluate_k_range(
    rfm_positive_df: pd.DataFrame, k_values: range = range(2, 10)
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Evaluate Elbow inertia and Silhouette scores across candidate cluster counts."""
    X_log = np.log1p(rfm_positive_df[RFM_COLUMNS].to_numpy(dtype="float64"))
    X_scaled = StandardScaler().fit_transform(X_log)

    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": int(k),
                "inertia": round(float(km.inertia_), 2),
                "silhouette_score": round(float(silhouette_score(X_scaled, labels)), 4),
            }
        )
    return pd.DataFrame(rows), X_scaled


def build_kmeans_pipeline(n_clusters: int = 4) -> Pipeline:
    """Create a reproducible preprocessing + K-Means pipeline using built-in np.log1p."""
    return Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=20, random_state=42)),
        ]
    )


def _assign_evidence_based_labels(
    pos_df: pd.DataFrame, raw_cluster_col: str = "raw_cluster"
) -> Dict[int, str]:
    """
    Map arbitrary K-Means integer IDs deterministically to descriptive segment names
    based strictly on each cluster's median MonetaryValue, Frequency, and Recency.
    """
    stats = pos_df.groupby(raw_cluster_col).agg(
        med_recency=("Recency", "median"),
        med_freq=("Frequency", "median"),
        med_mon=("MonetaryValue", "median"),
    )
    # Highest monetary + frequency champion cluster
    champion_id = int(stats["med_mon"].idxmax())
    # Highest recency (most dormant) cluster among remaining
    remaining = stats.drop(index=champion_id)
    dormant_id = int(remaining["med_recency"].idxmax())
    remaining = remaining.drop(index=dormant_id)

    label_map: Dict[int, str] = {
        champion_id: "High-Value Loyal",
        dormant_id: "Dormant / Low-Engagement",
    }
    if len(remaining) == 2:
        # Compare remaining two by Recency: higher Recency & higher Monetary = At-Risk Established;
        # lower Recency = Recent / Growing Buyers.
        ids_sorted_rec = remaining.sort_values("med_recency", ascending=False).index.tolist()
        label_map[int(ids_sorted_rec[0])] = "At-Risk Established"
        label_map[int(ids_sorted_rec[1])] = "Recent / Growing"
    else:
        for idx, cid in enumerate(remaining.sort_values("med_mon", ascending=False).index):
            label_map[int(cid)] = f"Mid-Tier Segment {idx + 1}"
    return label_map


def fit_and_segment_customers(
    customer_df: pd.DataFrame, n_clusters: int = 4
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, PCA]:
    """
    Fit K-Means on positive-MonetaryValue customers and assign clean segments across all customers.

    Customers with MonetaryValue <= 0 (net-returned / zero-net spending) are cleanly separated into
    Cluster -1 ('Net-Returned / Zero-Spend') so they do not distort log1p transformations.
    """
    df = customer_df.copy()
    pos_mask = df["MonetaryValue"] > 0
    pos_df = df[pos_mask].copy()
    non_pos_df = df[~pos_mask].copy()

    pipeline = build_kmeans_pipeline(n_clusters=n_clusters)
    X_pos = pos_df[RFM_COLUMNS].to_numpy(dtype="float64")
    pos_df["raw_cluster"] = pipeline.fit_predict(X_pos)

    label_map = _assign_evidence_based_labels(pos_df, "raw_cluster")
    # Re-index ClusterID cleanly in descending order of median MonetaryValue (0 = top segment)
    order = (
        pos_df.groupby("raw_cluster")["MonetaryValue"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    id_remap = {int(raw_id): new_id for new_id, raw_id in enumerate(order)}

    pos_df["ClusterID"] = pos_df["raw_cluster"].map(id_remap).astype("int64")
    pos_df["Segment"] = pos_df["raw_cluster"].map(label_map).astype("string")
    pos_df = pos_df.drop(columns=["raw_cluster"])

    # Compute 2D PCA projection on scaled log1p features for cluster visualization
    X_pos_scaled = pipeline[:-1].transform(X_pos)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_pos_scaled)
    pos_df["PCA1"] = np.round(coords[:, 0], 4)
    pos_df["PCA2"] = np.round(coords[:, 1], 4)

    if len(non_pos_df) > 0:
        non_pos_df["ClusterID"] = -1
        non_pos_df["Segment"] = "Net-Returned / Zero-Spend"
        non_pos_df["PCA1"] = np.nan
        non_pos_df["PCA2"] = np.nan
        segmented_df = pd.concat([pos_df, non_pos_df], ignore_index=True)
    else:
        segmented_df = pos_df

    segmented_df = segmented_df.sort_values(["ClusterID", "MonetaryValue"], ascending=[True, False]).reset_index(drop=True)

    total_customers = len(segmented_df)
    total_revenue = float(segmented_df["TotalSpending"].sum())
    summary_df = (
        segmented_df.groupby(["ClusterID", "Segment"], as_index=False)
        .agg(
            CustomerCount=("Customer ID", "count"),
            RecencyMedian=("Recency", "median"),
            RecencyMean=("Recency", "mean"),
            FrequencyMedian=("Frequency", "median"),
            FrequencyMean=("Frequency", "mean"),
            MonetaryMedian=("MonetaryValue", "median"),
            MonetaryMean=("MonetaryValue", "mean"),
            TotalSegmentRevenue=("TotalSpending", "sum"),
        )
        .sort_values("ClusterID")
        .reset_index(drop=True)
    )
    summary_df["CustomerSharePct"] = (summary_df["CustomerCount"] / total_customers * 100.0).round(2)
    summary_df["RevenueSharePct"] = (summary_df["TotalSegmentRevenue"] / total_revenue * 100.0).round(2)
    for col in ("RecencyMean", "FrequencyMean", "MonetaryMedian", "MonetaryMean", "TotalSegmentRevenue"):
        summary_df[col] = summary_df[col].round(2)

    return pipeline, segmented_df, summary_df, pca
