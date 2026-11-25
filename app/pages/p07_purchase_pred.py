"""Page 7: Purchase Prediction — Chronological Logistic Regression repeat-purchase probability."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import (
    get_logistic_regression_pipeline,
    load_chronological_test_cohort,
    load_model_metric_tables,
)
from app.utils.theme import PALETTE, style_plotly
from src.models.inference import (
    InvalidInputError,
    ModelLoadError,
    predict_purchase_probability,
)
from src.models.regression import FEATURE_COLUMNS


def render() -> None:
    render_page_header(
        "Purchase Prediction",
        "Estimate the probability that an eligible customer places at least one valid purchase during the 90-day Q4 target window.",
        badge="Machine Learning",
    )

    try:
        clf_pipe = get_logistic_regression_pipeline()
        cohort_df, obs_period, pred_period = load_chronological_test_cohort()
        clf_metrics = load_model_metric_tables()["logistic_regression"]
    except (ModelLoadError, FileNotFoundError) as exc:
        st.error(f"Could not load Logistic Regression assets: {exc}")
        return

    base_row = clf_metrics[(clf_metrics["split"] == "test") & (clf_metrics["threshold"] == 0.50) & (clf_metrics["model_name"].str.contains("Signed-Log1p"))].iloc[0]

    render_kpi_row(
        [
            ("Test ROC-AUC", f"{base_row['roc_auc']:.4f}", "vs. 0.5000 majority baseline"),
            ("Test PR-AUC", f"{base_row['pr_auc']:.4f}", f"Positive rate: {cohort_df['will_purchase'].mean()*100:.1f}%"),
            ("Observation Period", "275 Days", obs_period),
            ("Target Window", "90 Days (Q4)", pred_period),
        ]
    )

    render_beginner_note(
        "Why Classification Threshold Matters",
        "Logistic Regression outputs a probability between 0% and 100%. Raising the threshold (e.g., 0.56) increases Precision "
        "for costly campaigns; lowering it (e.g., 0.34) increases Recall for inexpensive digital emails.",
    )

    threshold = st.slider(
        "Classification Decision Threshold (Default 0.50 | Balanced Precision 0.56 | High-Recall 0.34)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.02,
        key="clf_threshold",
    )

    try:
        batch_out = predict_purchase_probability(cohort_df[FEATURE_COLUMNS], threshold=threshold, model=clf_pipe)
    except InvalidInputError as exc:
        st.error(f"Prediction error: {exc}")
        return
    y_true = cohort_df["will_purchase"].to_numpy()
    y_prob = batch_out["purchase_probability"].to_numpy()
    y_pred = batch_out["predicted_will_purchase"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    st.markdown(
        f"**Cohort Performance @ Threshold = `{threshold:.2f}`:** "
        f"Accuracy = `{accuracy_score(y_true, y_pred):.4f}` | "
        f"Precision = `{precision_score(y_true, y_pred, zero_division=0):.4f}` | "
        f"Recall = `{recall_score(y_true, y_pred, zero_division=0):.4f}` | "
        f"F1 = `{f1_score(y_true, y_pred, zero_division=0):.4f}` | "
        f"Confusion (`TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}`)"
    )

    col_left, col_right = st.columns([1.15, 1.0])
    with col_left:
        cids = cohort_df["customer_id"].astype(int).tolist()
        default_idx = cids.index(14911) if 14911 in cids else 0
        sel_cid = st.selectbox("Select Eligible Customer ID", options=cids, index=default_idx, key="clf_cid")
        idx_pos = cohort_df.index[cohort_df["customer_id"] == sel_cid][0]
        cust_row = cohort_df.loc[idx_pos]
        prob_val = float(batch_out.loc[idx_pos, "purchase_probability"])
        pred_cls = int(batch_out.loc[idx_pos, "predicted_will_purchase"])
        tier = str(batch_out.loc[idx_pos, "propensity_tier"])
        actual_cls = int(cust_row["will_purchase"])

        st.metric("Predicted Purchase Probability", f"{prob_val * 100:.1f}%", delta=tier)
        st.progress(min(max(prob_val, 0.0), 1.0))
        st.write(
            f"**Predicted Class (Threshold {threshold:.2f}):** "
            f"{'Will Purchase (1)' if pred_cls == 1 else 'No Purchase (0)'}  |  "
            f"**Actual Observed Outcome (Q4 2011):** "
            f"{'Purchased (1)' if actual_cls == 1 else 'No Purchase (0)'}"
        )

        hist_show = cust_row[FEATURE_COLUMNS].to_frame("Observed Historical Value").reset_index()
        hist_show.columns = ["Feature", "Observed Historical Value"]
        st.dataframe(hist_show, width="stretch", hide_index=True)

    with col_right:
        hist_probs = pd.DataFrame(
            {
                "Probability": y_prob,
                "Actual Outcome": np.where(y_true == 1, "Purchased (1)", "No Purchase (0)"),
            }
        )
        fig_p = px.histogram(
            hist_probs,
            x="Probability",
            color="Actual Outcome",
            barmode="overlay",
            nbins=30,
            color_discrete_map={"Purchased (1)": PALETTE["teal"], "No Purchase (0)": PALETTE["rose"]},
            title="Predicted Probability Distribution by Actual Outcome",
        )
        fig_p.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["amber"], annotation_text=f"Threshold {threshold:.2f}")
        st.plotly_chart(style_plotly(fig_p, height=400), width="stretch")

    scored_clf = cohort_df[["customer_id", "hist_spending", "hist_frequency", "hist_recency", "will_purchase"]].copy()
    scored_clf["purchase_probability"] = batch_out["purchase_probability"].values
    scored_clf["predicted_will_purchase"] = batch_out["predicted_will_purchase"].values
    scored_clf["propensity_tier"] = batch_out["propensity_tier"].values

    with st.expander("View & Download Scored Customer Cohort"):
        st.dataframe(scored_clf.sort_values("purchase_probability", ascending=False).head(300), width="stretch", hide_index=True)
        render_csv_download(scored_clf, "purchase_probability_predictions.csv", key="dl_clf_preds")


render()
