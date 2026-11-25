"""Page 1: Home — Project overview, dataset summary, key KPIs, and navigation guide."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import render_beginner_note, render_kpi_row, render_page_header
from app.utils.data_loader import (
    load_country_analytics,
    load_customer_segments,
    load_data_quality_report,
    load_monthly_sales,
    load_product_analytics,
)
from app.utils.theme import PALETTE, style_plotly


def render() -> None:
    render_page_header(
        "E-Commerce Customer Analytics",
        "Interactive Business Intelligence & Machine Learning Suite powered by real UCI Online Retail II transactions (2009-12-01 to 2011-12-09).",
        badge="Overview",
    )

    monthly_df = load_monthly_sales()
    country_df = load_country_analytics()
    product_df = load_product_analytics()
    cust_df = load_customer_segments()
    dq_df = load_data_quality_report()

    net_rev = float(monthly_df["TotalRevenue"].sum())
    gross_rev = float(monthly_df["GrossRevenue"].sum())
    ret_amt = float(monthly_df["ReturnedAmount"].sum())
    total_orders = int(monthly_df["TotalOrders"].sum())
    net_aov = net_rev / total_orders if total_orders else 0.0

    render_kpi_row(
        [
            ("Net Realized Revenue", f"£{net_rev:,.0f}", f"Gross £{gross_rev:,.0f} | Returns £{ret_amt:,.0f}"),
            ("Completed Purchase Orders", f"{total_orders:,}", f"Avg Order Value £{net_aov:,.2f}"),
            ("Purchasing Customers", f"{len(cust_df):,}", "With valid Customer ID"),
            ("Merchandise SKUs", f"{len(product_df):,}", f"Across {len(country_df):,} markets"),
        ]
    )

    render_beginner_note(
        "How to Read This Dashboard",
        "All KPIs and charts reflect validated real transactions from dataset/processed/. "
        "Revenue figures report Net Realized Revenue (normal sales minus C-prefixed customer returns/cancellations) "
        "alongside Gross Sales so reversed orders never inflate performance.",
    )

    col_left, col_right = st.columns([1.65, 1.0])

    with col_left:
        m_plot = monthly_df.copy()
        m_plot["MonthDate"] = pd.to_datetime(m_plot["InvoiceYearMonth"] + "-01")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=m_plot["MonthDate"],
                y=m_plot["GrossRevenue"],
                mode="lines+markers",
                name="Gross Sales (£)",
                line=dict(color=PALETTE["slate"], width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=m_plot["MonthDate"],
                y=m_plot["TotalRevenue"],
                mode="lines+markers",
                name="Net Revenue (£)",
                line=dict(color=PALETTE["accent"], width=3),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
            )
        )
        fig.update_layout(
            title="Monthly Gross vs. Net Revenue (Dec 2009 – Dec 2011)",
            xaxis_title="Month",
            yaxis_title="Revenue (£)",
            hovermode="x unified",
        )
        st.plotly_chart(style_plotly(fig, height=390), width="stretch")

    with col_right:
        st.subheader("Dataset & Pipeline Summary")
        raw_rows = int(dq_df.loc[dq_df["metric"] == "Total rows", "value"].iloc[0])
        summary_rows = pd.DataFrame(
            [
                ("Raw Combined CSV Rows", f"{raw_rows:,}"),
                ("Cleaned Valid Transactions", "1,026,678 (96.19%)"),
                ("Normal Sales Transactions", "1,007,823 (£20.43M)"),
                ("Returns & Cancellations ('C')", "18,855 (-£1.17M)"),
                ("Missing Customer ID Rows", "228,994 (Preserved via flag)"),
                ("ML Models Integrated", "4 (LR, LogReg, KNN, K-Means)"),
            ],
            columns=["Project Metric", "Verified Value"],
        )
        st.dataframe(summary_rows, width="stretch", hide_index=True)

    st.subheader("Dashboard Navigation Guide")
    nav_cols = st.columns(3)
    nav_cols[0].info(
        "**Business Analytics (Pages 2–5)**\n\n"
        "- **Sales Analytics:** Filterable revenue, orders, AOV & hourly trends.\n"
        "- **Customer Analytics:** Top accounts, repeat behavior & spending skew.\n"
        "- **Product Analytics:** Best-selling SKUs, pricing & Pareto focus.\n"
        "- **Country Analytics:** UK core vs. high-AOV export hubs."
    )
    nav_cols[1].success(
        "**Predictive & Recommender ML (Pages 6–9)**\n\n"
        "- **Future Spending:** 90-day spending forecast (Linear Regression).\n"
        "- **Purchase Prediction:** Repeat purchase probability + threshold tuning.\n"
        "- **Product Recommendation:** Item-Based KNN + cold-start fallback.\n"
        "- **Customer Segmentation:** 4 RFM K-Means personas + lookup."
    )
    nav_cols[2].warning(
        "**Governance & Validation (Pages 10–11)**\n\n"
        "- **Model Performance:** Evaluation metrics & diagnostic plots across all 4 models.\n"
        "- **Data Quality:** Missingness, Dec 2010 sheet-overlap duplicates & cancellation audit."
    )


render()
