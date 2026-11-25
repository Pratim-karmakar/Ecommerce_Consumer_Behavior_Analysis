"""Page 6: Future Spending Prediction — Chronological Linear Regression 90-day spend forecast."""

import plotly.express as px
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import (
    get_linear_regression_pipeline,
    load_chronological_test_cohort,
    load_model_metric_tables,
)
from app.utils.theme import PALETTE, style_plotly
from src.models.inference import (
    InvalidInputError,
    ModelLoadError,
    predict_future_spending,
)
from src.models.regression import FEATURE_COLUMNS


def render() -> None:
    render_page_header(
        "Future Spending Prediction",
        "Predict customer net spending over the next 90 days using strictly historical pre-cutoff behavior (Linear Regression).",
        badge="Machine Learning",
    )

    try:
        lr_pipe = get_linear_regression_pipeline()
        cohort_df, obs_period, pred_period = load_chronological_test_cohort()
        lr_metrics = load_model_metric_tables()["linear_regression"]
    except (ModelLoadError, FileNotFoundError) as exc:
        st.error(f"Could not load Linear Regression assets: {exc}")
        return

    test_row = lr_metrics[
        (lr_metrics["model_name"] == "Linear Regression (Full Pipeline)")
        & (lr_metrics["split"] == "test")
    ].iloc[0]

    render_kpi_row(
        [
            ("Observation Period", "275 Days", obs_period),
            ("Prediction Horizon", "90 Days (Q4)", pred_period),
            ("Test MAE / RMSE", f"£{test_row['mae']:,.2f}", f"RMSE £{test_row['rmse']:,.2f}"),
            ("Test R² Score", f"{test_row['r2_score']:.4f}", f"Eligible cohort: {len(cohort_df):,} customers"),
        ]
    )

    render_beginner_note(
        "How Data Leakage Was Prevented",
        "All input features (historical spending, order count, quantity, AOV, unique products, recency, tenure, and cancellations) "
        f"come strictly from {obs_period}. The model predicts spending during {pred_period} without seeing any target-period transactions.",
    )

    mode = st.radio(
        "Prediction Mode",
        ["Select Eligible Real Customer (Chronological Test Cohort)", "Interactive What-If Feature Simulator"],
        horizontal=True,
        key="lr_mode",
    )

    if mode.startswith("Select"):
        cids = cohort_df["customer_id"].astype(int).tolist()
        default_idx = cids.index(14911) if 14911 in cids else 0
        sel_cid = st.selectbox("Select Eligible Customer ID", options=cids, index=default_idx, key="lr_cid")
        cust_row = cohort_df[cohort_df["customer_id"] == sel_cid].iloc[0]
        feat_dict = {col: float(cust_row[col]) for col in FEATURE_COLUMNS}

        pred_df = predict_future_spending(feat_dict, model=lr_pipe)
        pred_val = float(pred_df["predicted_spending"].iloc[0])
        pred_clip = float(pred_df["predicted_spending_non_negative"].iloc[0])
        actual_val = float(cust_row["future_spending"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted 90-Day Spend (Raw OLS)", f"£{pred_val:,.2f}")
        c2.metric("Non-Negative Clipped Forecast", f"£{pred_clip:,.2f}")
        c3.metric("Actual Observed Q4 2011 Spend", f"£{actual_val:,.2f}", delta=f"Error: £{actual_val - pred_val:,.2f}")

        st.subheader(f"Historical Input Features for Customer {sel_cid} ({obs_period})")
        feat_view = cust_row[FEATURE_COLUMNS].to_frame("Observed Historical Value").reset_index()
        feat_view.columns = ["Feature", "Observed Historical Value"]
        st.dataframe(feat_view, width="stretch", hide_index=True)
    else:
        st.subheader("Test Custom Historical Behavior with the Trained Pipeline")
        c1, c2, c3, c4 = st.columns(4)
        h_spend = c1.number_input("Historical Net Spend (£)", value=1500.0, step=100.0, key="lr_sim_spend")
        h_freq = c2.number_input("Historical Orders", min_value=1, value=5, step=1, key="lr_sim_freq")
        h_qty = c3.number_input("Historical Units", value=600, step=1, key="lr_sim_qty")
        h_aov = c4.number_input("Historical AOV (£)", value=round(h_spend / max(h_freq, 1), 2), step=25.0, key="lr_sim_aov")

        c5, c6, c7, c8 = st.columns(4)
        h_prod = c5.number_input("Unique Products", min_value=1, value=35, step=1, key="lr_sim_prod")
        h_rec = c6.number_input("Recency (Days)", min_value=1, value=25, step=1, key="lr_sim_rec")
        h_ten = c7.number_input("Tenure (Days)", min_value=1, value=180, step=1, key="lr_sim_ten")
        h_can = c8.number_input("Cancelled Orders", min_value=0, value=0, step=1, key="lr_sim_can")

        try:
            sim_res = predict_future_spending(
                {
                    "hist_spending": h_spend,
                    "hist_frequency": h_freq,
                    "hist_quantity": h_qty,
                    "hist_aov": h_aov,
                    "hist_unique_products": h_prod,
                    "hist_recency": h_rec,
                    "hist_tenure": h_ten,
                    "hist_cancelled_orders": h_can,
                },
                model=lr_pipe,
            )
            st.success(
                f"**Predicted 90-Day Spending:** £{sim_res['predicted_spending'].iloc[0]:,.2f} "
                f"(Non-negative forecast: £{sim_res['predicted_spending_non_negative'].iloc[0]:,.2f})"
            )
        except InvalidInputError as exc:
            st.error(f"Invalid input: {exc}")

    # Batch scoring preview on test cohort
    batch_preds = predict_future_spending(cohort_df[FEATURE_COLUMNS], model=lr_pipe)
    scored_df = cohort_df[["customer_id", "hist_spending", "hist_frequency", "hist_recency", "future_spending"]].copy()
    scored_df["predicted_spending"] = batch_preds["predicted_spending"].values
    scored_df["abs_error"] = (scored_df["future_spending"] - scored_df["predicted_spending"]).abs().round(2)

    zoom = scored_df[(scored_df["future_spending"] <= 15000) & (scored_df["predicted_spending"].between(-500, 15000))]
    fig_ap = px.scatter(
        zoom,
        x="future_spending",
        y="predicted_spending",
        hover_data=["customer_id", "hist_spending", "hist_frequency"],
        opacity=0.45,
        color_discrete_sequence=[PALETTE["accent"]],
        title="Test Cohort Actual vs. Predicted 90-Day Spending (Zoomed <= £15,000)",
        labels={"future_spending": "Actual Q4 2011 Spending (£)", "predicted_spending": "Predicted Spending (£)"},
    )
    fig_ap.add_shape(type="line", x0=0, y0=0, x1=15000, y1=15000, line=dict(color=PALETTE["rose"], dash="dash"))
    st.plotly_chart(style_plotly(fig_ap, height=400), width="stretch")

    with st.expander("View & Download Batch Test Predictions"):
        st.dataframe(scored_df.sort_values("predicted_spending", ascending=False).head(300), width="stretch", hide_index=True)
        render_csv_download(scored_df, "future_spending_predictions.csv", key="dl_lr_preds")


render()
