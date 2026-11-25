"""Leakage-free temporal cohort creation, training, and evaluation for future purchase prediction."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.models.regression import FEATURE_COLUMNS, build_temporal_cohort

TARGET_CLASS_COLUMN = "will_purchase"


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """Apply sign-preserving log1p transform so skewed spend/counts stay stable with negative values."""
    return np.sign(X) * np.log1p(np.abs(X))


def build_purchase_classification_cohort(
    tx_df: pd.DataFrame,
    obs_start: str,
    obs_end: str,
    target_start: str,
    target_end: str,
) -> pd.DataFrame:
    """
    Build eligible customer classification cohort with complete future target observability.

    Label definition:
    - 1: Customer placed at least one valid non-cancelled purchase invoice in [target_start, target_end).
    - 0: Customer was active in [obs_start, obs_end) and made zero valid purchases in the complete target window.
    """
    cohort = build_temporal_cohort(tx_df, obs_start, obs_end, target_start, target_end)

    # Check completeness of target window against dataset observation end.
    max_observed_date = pd.to_datetime(tx_df["invoice_date"]).max()
    expected_end_inclusive = pd.Timestamp(target_end) - pd.Timedelta(days=1)
    if max_observed_date.normalize() < expected_end_inclusive.normalize():
        raise ValueError(
            f"Target window ending {expected_end_inclusive.date()} is incomplete; dataset ends at {max_observed_date.date()}."
        )

    tgt_sales = tx_df[
        tx_df["has_customer_id"]
        & (~tx_df["is_cancelled"])
        & (tx_df["invoice_date"] >= target_start)
        & (tx_df["invoice_date"] < target_end)
    ]
    purchasers = set(tgt_sales["customer_id"].unique())
    cohort[TARGET_CLASS_COLUMN] = cohort["customer_id"].isin(purchasers).astype("int64")
    return cohort


def build_logistic_regression_pipeline(
    use_log_transform: bool = True, c_value: float = 1.0
) -> Pipeline:
    """Build preprocessing and Logistic Regression pipeline (all importable for safe joblib serialization)."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_log_transform:
        steps.append(("log_transform", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c_value, max_iter=1000, random_state=42)),
        ]
    )
    return Pipeline(steps=steps)


def compute_classification_metrics(
    model_name: str,
    split: str,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float | int | str]:
    """Compute Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion counts."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model_name": model_name,
        "split": split,
        "threshold": round(float(threshold), 2),
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_thresholds_on_train_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> Tuple[float, float, np.ndarray]:
    """Find optimal F1 and Balanced Accuracy thresholds strictly via out-of-fold CV on Train."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    best_f1_thr, best_f1 = 0.50, -1.0
    best_bal_thr, best_bal = 0.50, -1.0
    for thr in np.linspace(0.20, 0.80, 61):
        preds = (oof_probs >= thr).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        bal = balanced_accuracy_score(y_train, preds)
        if f1 > best_f1:
            best_f1, best_f1_thr = f1, float(thr)
        if bal > best_bal:
            best_bal, best_bal_thr = bal, float(thr)

    return round(best_bal_thr, 2), round(best_f1_thr, 2), oof_probs


def evaluate_classification_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Fit baselines and Logistic Regression strictly on Train, tune thresholds via CV, evaluate on Test."""
    X_train, y_train = train_df[feature_cols], train_df[TARGET_CLASS_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_CLASS_COLUMN]

    results = []

    # 1. Majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    results.append(compute_classification_metrics("Baseline (Majority Class)", "test", y_test, dummy_probs, 0.50))

    # 2. Raw StandardScaled Logistic Regression (comparison)
    raw_pipe = build_logistic_regression_pipeline(use_log_transform=False)
    raw_pipe.fit(X_train, y_train)
    results.append(
        compute_classification_metrics(
            "Logistic Regression (Raw Scaled)", "test", y_test, raw_pipe.predict_proba(X_test)[:, 1], 0.50
        )
    )

    # 3. Signed-Log1p + StandardScaled Logistic Regression (primary model)
    log_pipe = build_logistic_regression_pipeline(use_log_transform=True)
    bal_thr, f1_thr, oof_probs = tune_thresholds_on_train_cv(log_pipe, X_train, y_train)
    log_pipe.fit(X_train, y_train)

    train_probs = log_pipe.predict_proba(X_train)[:, 1]
    test_probs = log_pipe.predict_proba(X_test)[:, 1]

    results.append(compute_classification_metrics("Logistic Regression (Signed-Log1p)", "train_oof_cv", y_train, oof_probs, 0.50))
    results.append(compute_classification_metrics("Logistic Regression (Signed-Log1p)", "train_full", y_train, train_probs, 0.50))
    results.append(compute_classification_metrics("Logistic Regression (Signed-Log1p, thr=0.50)", "test", y_test, test_probs, 0.50))
    results.append(compute_classification_metrics(f"Logistic Regression (Balanced thr={bal_thr:.2f})", "test", y_test, test_probs, bal_thr))
    results.append(compute_classification_metrics(f"Logistic Regression (Max-F1 thr={f1_thr:.2f})", "test", y_test, test_probs, f1_thr))

    pred_df = test_df[["customer_id", "hist_spending", "hist_frequency", "hist_recency", TARGET_CLASS_COLUMN]].copy()
    pred_df["pred_probability"] = np.round(test_probs, 4)
    pred_df["pred_default_050"] = (test_probs >= 0.50).astype(int)
    pred_df["pred_balanced"] = (test_probs >= bal_thr).astype(int)
    pred_df["pred_max_f1"] = (test_probs >= f1_thr).astype(int)

    thresholds = {"default": 0.50, "balanced_accuracy": bal_thr, "max_f1": f1_thr}
    return log_pipe, pd.DataFrame(results), thresholds, pred_df
