"""Page 9: Customer Segmentation — K-Means RFM clusters, PCA view, and live customer lookup."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import (
    get_kmeans_pipeline,
    load_customer_segments,
    load_model_metric_tables,
)
from app.utils.theme import PALETTE, SEGMENT_COLORS, style_plotly
from src.models.inference import InvalidInputError, ModelLoadError, assign_customer_segment


def render() -> None:
    render_page_header(
        "Customer Segmentation",
        "Explore behavioral RFM segments discovered via log1p-scaled K-Means clustering (K = 4 + isolated zero/net-return accounts).",
        badge="Machine Learning",
    )

    try:
        km_pipe = get_kmeans_pipeline()
        seg_df = load_customer_segments()
        summary_df = load_model_metric_tables()["kmeans_summary"]
    except (ModelLoadError, FileNotFoundError) as exc:
        st.error(f"Could not load segmentation assets: {exc}")
        return

    loyal_row = summary_df[summary_df["Segment"] == "High-Value Loyal"].iloc[0]
    risk_row = summary_df[summary_df["Segment"] == "At-Risk Established"].iloc[0]

    render_kpi_row(
        [
            ("Clusters + Isolated", "4 + 1", "K=4 RFM clusters | Silhouette 0.3668"),
            ("High-Value Loyal Share", f"{loyal_row['RevenueSharePct']:.1f}% Rev", f"{int(loyal_row['CustomerCount']):,} customers ({loyal_row['CustomerSharePct']:.1f}%)"),
            ("At-Risk Established Pool", f"{int(risk_row['CustomerCount']):,} Buyers", f"Median Recency {risk_row['RecencyMedian']:.0f} days"),
            ("Net-Returned / Zero-Spend", "40 Accounts", "Isolated safely (Monetary <= £0)"),
        ]
    )

    render_beginner_note(
        "How Segments Were Formed",
        "Because Frequency and MonetaryValue are heavily skewed, RFM metrics were log1p-transformed and standardized before clustering. "
        "K=4 achieved the best Silhouette peak (0.3668) above K=3 and K=5 while separating At-Risk Established buyers from Recent Growing buyers.",
    )

    col_l, col_r = st.columns(2)
    with col_l:
        pos_sum = summary_df[summary_df["ClusterID"] >= 0]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=pos_sum["Segment"], y=pos_sum["CustomerSharePct"], name="% of Customers", marker_color=PALETTE["accent"]))
        fig_bar.add_trace(go.Bar(x=pos_sum["Segment"], y=pos_sum["RevenueSharePct"], name="% of Revenue", marker_color=PALETTE["teal"]))
        fig_bar.update_layout(barmode="group", title="Customer Share vs. Net Revenue Share by Segment", yaxis_title="Share (%)")
        st.plotly_chart(style_plotly(fig_bar, height=400), width="stretch")

    with col_r:
        pos_pts = seg_df[seg_df["ClusterID"] >= 0]
        fig_pca = px.scatter(
            pos_pts,
            x="PCA1",
            y="PCA2",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            hover_data=["Customer ID", "Recency", "Frequency", "MonetaryValue"],
            opacity=0.6,
            title="2D PCA Cluster Projection (95.19% Explained Variance)",
        )
        st.plotly_chart(style_plotly(fig_pca, height=400), width="stretch")

    st.subheader("Cluster-Level RFM Summary")
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.subheader("Customer Segment Lookup & Live Scorer")
    lookup_mode = st.radio("Lookup Mode", ["Lookup Real Customer ID", "Score Custom RFM Values"], horizontal=True, key="seg_mode")

    if lookup_mode == "Lookup Real Customer ID":
        cids = seg_df["Customer ID"].astype(int).tolist()
        default_idx = cids.index(14911) if 14911 in cids else 0
        sel_cid = st.selectbox("Select Customer ID", options=cids, index=default_idx, key="seg_cid")
        row = seg_df[seg_df["Customer ID"] == sel_cid].iloc[0]
        st.info(
            f"**Customer `{sel_cid}`** $\\rightarrow$ **Segment: `{row['Segment']}`** (`ClusterID = {int(row['ClusterID'])}`)  |  "
            f"Recency: `{int(row['Recency'])} days` | Frequency: `{int(row['Frequency'])} orders` | "
            f"MonetaryValue: `£{float(row['MonetaryValue']):,.2f}`"
        )
    else:
        c1, c2, c3 = st.columns(3)
        r_val = c1.number_input("Recency (Days Since Last Purchase)", min_value=0.0, value=30.0, step=5.0, key="seg_sim_r")
        f_val = c2.number_input("Frequency (Completed Orders)", min_value=0.0, value=6.0, step=1.0, key="seg_sim_f")
        m_val = c3.number_input("Monetary Value (£)", value=1500.0, step=100.0, key="seg_sim_m")
        try:
            assigned = assign_customer_segment({"Recency": r_val, "Frequency": f_val, "MonetaryValue": m_val}, model=km_pipe)
            st.success(
                f"Assigned Segment: **{assigned['Segment'].iloc[0]}** (`ClusterID = {int(assigned['ClusterID'].iloc[0])}`)"
            )
        except InvalidInputError as exc:
            st.error(f"Invalid RFM input: {exc}")

    with st.expander("Filter & Download Customer Segment Assignments"):
        seg_choices = seg_df["Segment"].unique().tolist()
        chosen_segs = st.multiselect("Select Segments to Export", seg_choices, default=seg_choices, key="seg_export_filter")
        export_df = seg_df[seg_df["Segment"].isin(chosen_segs)]
        st.dataframe(export_df.head(500), width="stretch", hide_index=True)
        render_csv_download(export_df, "customer_segments_filtered.csv", key="dl_segments")


render()
