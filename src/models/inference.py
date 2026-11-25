"""Unified model loading, validation, error handling, and inference helpers for all 4 ML models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.recommender import ItemKNNRecommender
from src.models.regression import FEATURE_COLUMNS
from src.models.segmentation import RFM_COLUMNS
from src.utils.paths import find_project_root

ModelInput = Union[Dict[str, Any], pd.Series, pd.DataFrame]


class ModelLoadError(FileNotFoundError):
    """Raised when a serialized model artifact is missing or cannot be loaded."""


class InvalidInputError(ValueError):
    """Raised when caller inputs have invalid types, missing required keys, or impossible values."""


class UnknownProductError(KeyError):
    """Raised when a StockCode is not present in the trained recommender catalog."""


def get_default_models_dir() -> Path:
    """Resolve the project models/ directory dynamically."""
    return find_project_root() / "models"


def load_model_artifact(
    model_path: Path, expected_type: Optional[type] = None
) -> Any:
    """Safely load a joblib artifact and validate its existence and object type."""
    path = Path(model_path)
    if not path.is_file():
        raise ModelLoadError(
            f"Required model artifact not found at '{path}'. Run the corresponding training notebook first."
        )
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to deserialize model artifact '{path}': {exc}") from exc

    if expected_type is not None and not isinstance(artifact, expected_type):
        raise ModelLoadError(
            f"Artifact at '{path}' has unexpected type {type(artifact).__name__}; expected {expected_type.__name__}."
        )
    return artifact


def load_linear_regression_model(models_dir: Optional[Path] = None) -> Pipeline:
    """Load the saved Linear Regression spending prediction pipeline."""
    mdir = models_dir or get_default_models_dir()
    return load_model_artifact(mdir / "linear_regression_pipeline.joblib", Pipeline)


def load_logistic_regression_model(models_dir: Optional[Path] = None) -> Pipeline:
    """Load the saved Logistic Regression future purchase probability pipeline."""
    mdir = models_dir or get_default_models_dir()
    return load_model_artifact(mdir / "logistic_regression_pipeline.joblib", Pipeline)


def load_knn_recommender(models_dir: Optional[Path] = None) -> ItemKNNRecommender:
    """Load the saved Item-Based KNN product recommender."""
    mdir = models_dir or get_default_models_dir()
    return load_model_artifact(mdir / "knn_recommender.joblib", ItemKNNRecommender)


def load_kmeans_model(models_dir: Optional[Path] = None) -> Pipeline:
    """Load the saved RFM K-Means segmentation pipeline."""
    mdir = models_dir or get_default_models_dir()
    return load_model_artifact(mdir / "kmeans_pipeline.joblib", Pipeline)


def validate_customer_id(customer_id: Any) -> int:
    """Validate and convert a customer identifier into a strictly positive integer."""
    if customer_id is None or isinstance(customer_id, bool):
        raise InvalidInputError("Customer ID must be a positive integer (got None/bool).")
    if isinstance(customer_id, float):
        if np.isnan(customer_id) or not np.isfinite(customer_id) or customer_id % 1 != 0:
            raise InvalidInputError(f"Customer ID '{customer_id}' is not a valid whole integer.")
        cid = int(customer_id)
    elif isinstance(customer_id, (int, np.integer)):
        cid = int(customer_id)
    elif isinstance(customer_id, str):
        stripped = customer_id.strip()
        if not stripped or not stripped.lstrip("+-").isdigit():
            raise InvalidInputError(f"Customer ID '{customer_id}' is not a valid numeric ID.")
        cid = int(stripped)
    else:
        raise InvalidInputError(f"Unsupported Customer ID type: {type(customer_id).__name__}.")

    if cid <= 0:
        raise InvalidInputError(f"Customer ID must be positive (got {cid}).")
    return cid


def _coerce_feature_dataframe(
    inputs: ModelInput,
    required_columns: List[str],
    allow_missing_values: bool,
) -> pd.DataFrame:
    """Validate and normalize dict, Series, or DataFrame inputs into a clean numeric DataFrame."""
    if isinstance(inputs, dict):
        if not inputs:
            raise InvalidInputError("Input feature dictionary cannot be empty.")
        df = pd.DataFrame([inputs])
    elif isinstance(inputs, pd.Series):
        df = inputs.to_frame().T
    elif isinstance(inputs, pd.DataFrame):
        if inputs.empty:
            raise InvalidInputError("Input DataFrame cannot be empty.")
        df = inputs.copy()
    else:
        raise InvalidInputError(
            f"Expected dict, pd.Series, or pd.DataFrame; got {type(inputs).__name__}."
        )

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise InvalidInputError(f"Missing required feature column(s): {missing_cols}")

    selected = df[required_columns].copy()
    for col in required_columns:
        try:
            # Convert None/pd.NA cleanly to np.nan before numeric coercion.
            selected[col] = pd.to_numeric(selected[col], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise InvalidInputError(
                f"Feature '{col}' contains non-numeric or invalid value(s): {exc}"
            ) from exc

        if np.isinf(selected[col]).any():
            raise InvalidInputError(f"Feature '{col}' contains infinite values.")

    if not allow_missing_values and selected.isna().any().any():
        bad_cols = [c for c in required_columns if selected[c].isna().any()]
        raise InvalidInputError(
            f"Missing (NaN/None) values are not permitted for: {bad_cols}."
        )
    return selected


def predict_future_spending(
    features: ModelInput,
    model: Optional[Pipeline] = None,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Predict 90-day future customer spending using the saved Linear Regression pipeline.

    Missing values inside historical feature columns are imputed automatically by the saved
    SimpleImputer(strategy='median') step.
    """
    X = _coerce_feature_dataframe(features, FEATURE_COLUMNS, allow_missing_values=True)
    pipe = model or load_linear_regression_model(models_dir)
    preds = pipe.predict(X)
    return pd.DataFrame(
        {
            "predicted_spending": np.round(preds, 2),
            "predicted_spending_non_negative": np.round(np.clip(preds, 0.0, None), 2),
        },
        index=X.index,
    )


def predict_purchase_probability(
    features: ModelInput,
    threshold: float = 0.50,
    model: Optional[Pipeline] = None,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Predict the probability of at least one future 90-day purchase using Logistic Regression."""
    if not (0.0 <= float(threshold) <= 1.0):
        raise InvalidInputError(f"Classification threshold must lie in [0, 1]; got {threshold}.")

    X = _coerce_feature_dataframe(features, FEATURE_COLUMNS, allow_missing_values=True)
    pipe = model or load_logistic_regression_model(models_dir)
    probs = pipe.predict_proba(X)[:, 1]
    tiers = pd.cut(
        probs,
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=["Low Propensity", "Medium Propensity", "High Propensity"],
    ).astype(str)

    return pd.DataFrame(
        {
            "purchase_probability": np.round(probs, 4),
            "predicted_will_purchase": (probs >= threshold).astype(int),
            "threshold_used": round(float(threshold), 2),
            "propensity_tier": tiers,
        },
        index=X.index,
    )


def recommend_products(
    customer_id: Any,
    top_n: int = 10,
    exclude_purchased: bool = True,
    recommender: Optional[ItemKNNRecommender] = None,
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate top-N product recommendations for a validated Customer ID.

    Unseen valid customer IDs trigger a cold-start popularity fallback transparently.
    """
    cid = validate_customer_id(customer_id)
    if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or int(top_n) <= 0:
        raise InvalidInputError(f"top_n must be a positive integer (got {top_n}).")

    rec = recommender or load_knn_recommender(models_dir)
    is_known = cid in rec.cust_to_idx_
    history_df = rec.get_customer_history(cid, top_n=int(top_n))
    recs_df = rec.recommend_for_customer(cid, top_n=int(top_n), exclude_purchased=exclude_purchased)

    return {
        "customer_id": cid,
        "is_cold_start": not is_known,
        "history_preview": history_df,
        "recommendations": recs_df,
    }


def find_similar_products(
    stock_code: Any,
    top_n: int = 10,
    strict: bool = True,
    recommender: Optional[ItemKNNRecommender] = None,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Find similar co-purchased products for a StockCode, validating against the trained catalog."""
    if not isinstance(stock_code, str) or not stock_code.strip():
        raise InvalidInputError("stock_code must be a non-empty string.")
    if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or int(top_n) <= 0:
        raise InvalidInputError(f"top_n must be a positive integer (got {top_n}).")

    code = stock_code.strip().upper()
    rec = recommender or load_knn_recommender(models_dir)
    if code not in rec.item_to_idx_:
        if strict:
            raise UnknownProductError(
                f"StockCode '{code}' was not found in the trained collaborative filtering catalog "
                f"(requires valid merchandise with >= {rec.min_item_customers} customers)."
            )
        return pd.DataFrame(
            columns=["stock_code", "description", "cosine_similarity", "customer_reach", "avg_selling_price"]
        )
    return rec.find_similar_items(code, top_n=int(top_n))


def derive_cluster_label_mapping(pipeline: Pipeline) -> Dict[int, Tuple[int, str]]:
    """
    Recover deterministic raw_cluster -> (ClusterID, Segment) mapping directly from saved centroids.
    """
    scaler = pipeline.named_steps["scaler"]
    km = pipeline.named_steps["kmeans"]
    centroids = np.expm1(scaler.inverse_transform(km.cluster_centers_))
    stats = pd.DataFrame(centroids, columns=RFM_COLUMNS)
    stats["raw_id"] = range(len(stats))

    champion_raw = int(stats.loc[stats["MonetaryValue"].idxmax(), "raw_id"])
    remaining = stats[stats["raw_id"] != champion_raw]
    dormant_raw = int(remaining.loc[remaining["Recency"].idxmax(), "raw_id"])
    middle = remaining[remaining["raw_id"] != dormant_raw]

    label_names: Dict[int, str] = {
        champion_raw: "High-Value Loyal",
        dormant_raw: "Dormant / Low-Engagement",
    }
    if len(middle) == 2:
        mid_sorted = middle.sort_values("Recency", ascending=False)
        label_names[int(mid_sorted.iloc[0]["raw_id"])] = "At-Risk Established"
        label_names[int(mid_sorted.iloc[1]["raw_id"])] = "Recent / Growing"
    else:
        for idx, raw_id in enumerate(middle.sort_values("MonetaryValue", ascending=False)["raw_id"]):
            label_names[int(raw_id)] = f"Mid-Tier Segment {idx + 1}"

    ordered_raw = stats.sort_values("MonetaryValue", ascending=False)["raw_id"].tolist()
    return {int(raw_id): (new_id, label_names[int(raw_id)]) for new_id, raw_id in enumerate(ordered_raw)}


def assign_customer_segment(
    rfm_features: ModelInput,
    model: Optional[Pipeline] = None,
    models_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Assign K-Means RFM segments consistently with `customer_segments.csv`.

    Validates non-negative Recency and Frequency, and cleanly assigns ClusterID=-1
    ('Net-Returned / Zero-Spend') when MonetaryValue <= 0 without failing log1p.
    """
    X = _coerce_feature_dataframe(rfm_features, RFM_COLUMNS, allow_missing_values=False)
    if (X["Recency"] < 0).any() or (X["Frequency"] < 0).any():
        raise InvalidInputError("Recency and Frequency cannot be negative.")

    pipe = model or load_kmeans_model(models_dir)
    mapping = derive_cluster_label_mapping(pipe)

    cluster_ids = np.full(len(X), -1, dtype=int)
    segments = np.full(len(X), "Net-Returned / Zero-Spend", dtype=object)

    pos_mask = (X["MonetaryValue"] > 0).to_numpy()
    if pos_mask.any():
        raw_preds = pipe.predict(X.loc[pos_mask, RFM_COLUMNS].to_numpy(dtype="float64"))
        pos_indices = np.where(pos_mask)[0]
        for idx, raw_id in zip(pos_indices, raw_preds):
            cid, seg_name = mapping[int(raw_id)]
            cluster_ids[idx] = cid
            segments[idx] = seg_name

    return pd.DataFrame(
        {
            "ClusterID": cluster_ids,
            "Segment": segments,
        },
        index=X.index,
    )
