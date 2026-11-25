"""Page 3: Customer Analytics — Top customers, spending skew, frequency, and RFM behavior."""

import numpy as np
import plotly.express as px
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import load_customer_segments
from app.utils.theme import PALETTE, SEGMENT_COLORS, style_plotly


def render() -> None:
    render_page_header(
        "Customer Analytics",
        "Analyze spending distributions, purchase frequency, repeat loyalty, and top accounts across all 5,869 identified buyers.",
        badge="Analytics",
    )

    cust_df = load_customer_segments()

    with st.expander("Filter Customers", expanded=True):
        r1c1, r1c2 = st.columns([2.4, 1.0])
        segments_avail = cust_df["Segment"].unique().tolist()
        sel_segments = r1c1.multiselect("Segments", segments_avail, default=segments_avail, key="cust_seg_filter")
        r1c2.write("")
        exclude_non_pos = r1c2.checkbox("Exclude Net <= £0 Accounts", value=False, key="cust_excl_neg")

        r2c1, r2c2 = st.columns(2)
        min_orders = r2c1.slider("Minimum Orders", 1, 50, 1, key="cust_min_orders")
        top_n = r2c2.slider("Top N Customers to Chart", 5, 25, 10, key="cust_top_n")

    filtered = cust_df[cust_df["Segment"].isin(sel_segments) & (cust_df["Frequency"] >= min_orders)]
    if exclude_non_pos:
        filtered = filtered[filtered["TotalSpending"] > 0]

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        return

    repeat_pct = (filtered["Frequency"] >= 2).mean() * 100
    render_kpi_row(
        [
            ("Customers Shown", f"{len(filtered):,}", f"{len(filtered)/len(cust_df)*100:.1f}% of purchasing base"),
            ("Mean Net Spending", f"£{filtered['TotalSpending'].mean():,.2f}", f"Median £{filtered['TotalSpending'].median():,.2f}"),
            ("Mean Order Frequency", f"{filtered['Frequency'].mean():.2f}", f"Median {filtered['Frequency'].median():.0f} orders"),
            ("Repeat Buyer Rate", f"{repeat_pct:.1f}%", f"Median Recency {filtered['Recency'].median():.0f} days"),
        ]
    )

    render_beginner_note(
        "Why Mean Spending Exceeds Median Spending",
        "Notice the wide gap between Mean and Median customer spending: wholesale/B2B buyers pull the average upward, "
        "while most individual shoppers sit near the median. All totals reflect Net Spending (Gross Purchases minus Returns).",
    )

    col_l, col_r = st.columns(2)
    with col_l:
        top_spend = filtered.sort_values("TotalSpending", ascending=False).head(top_n).copy()
        top_spend["CustomerLabel"] = "ID " + top_spend["Customer ID"].astype(str)
        fig_s = px.bar(
            top_spend.sort_values("TotalSpending"),
            x="TotalSpending",
            y="CustomerLabel",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title=f"Top {len(top_spend)} Customers by Net Spending",
            labels={"TotalSpending": "Net Total Spending (£)", "CustomerLabel": "Customer"},
        )
        st.plotly_chart(style_plotly(fig_s, height=400), width="stretch")

    with col_r:
        top_freq = filtered.sort_values("Frequency", ascending=False).head(top_n).copy()
        top_freq["CustomerLabel"] = "ID " + top_freq["Customer ID"].astype(str)
        fig_f = px.bar(
            top_freq.sort_values("Frequency"),
            x="Frequency",
            y="CustomerLabel",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title=f"Top {len(top_freq)} Customers by Order Frequency",
            labels={"Frequency": "Completed Orders", "CustomerLabel": "Customer"},
        )
        st.plotly_chart(style_plotly(fig_f, height=400), width="stretch")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        pos_only = filtered[filtered["TotalSpending"] > 0].copy()
        pos_only["Log10Spending"] = np.log10(pos_only["TotalSpending"])
        fig_hist_m = px.histogram(
            pos_only,
            x="Log10Spending",
            nbins=35,
            color_discrete_sequence=[PALETTE["primary"]],
            title="Customer Net Spending Distribution (Log10 Scale)",
            labels={"Log10Spending": "log10(Net Spending £)"},
        )
        st.plotly_chart(style_plotly(fig_hist_m, height=360), width="stretch")

    with col_h2:
        freq_plot = filtered.copy()
        freq_plot["FrequencyClipped"] = freq_plot["Frequency"].clip(upper=30)
        fig_hist_f = px.histogram(
            freq_plot,
            x="FrequencyClipped",
            nbins=30,
            color_discrete_sequence=[PALETTE["teal"]],
            title="Purchase Frequency Distribution (Clipped at 30 Orders)",
            labels={"FrequencyClipped": "Completed Orders (30 = 30+)"},
        )
        st.plotly_chart(style_plotly(fig_hist_f, height=360), width="stretch")

    st.subheader("Filtered Customer Explorer")
    cols_show = [
        "Customer ID", "Segment", "TotalSpending", "GrossSpending", "ReturnedAmount",
        "TotalOrders", "TotalQuantity", "UniqueProductsPurchased", "AverageOrderValue", "Recency",
    ]
    st.dataframe(filtered[cols_show].head(500), width="stretch", hide_index=True)
    render_csv_download(filtered[cols_show], "filtered_customer_analytics.csv", key="dl_customers")


render()
