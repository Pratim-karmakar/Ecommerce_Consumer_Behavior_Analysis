"""Page 2: Sales Analytics — Interactive revenue, order, AOV, and time-pattern analysis."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import load_transactions
from app.utils.theme import PALETTE, style_plotly
from src.analysis.eda_metrics import compute_sales_kpis, compute_time_breakdowns


def render() -> None:
    render_page_header(
        "Sales Analytics",
        "Explore revenue, orders, quantity sold, Average Order Value, and temporal demand patterns with interactive filters.",
        badge="Analytics",
    )

    tx_df = load_transactions()

    # Sidebar / Top Filters
    with st.expander("Filter Transactions", expanded=True):
        c1, c2, c3 = st.columns(3)
        years_avail = sorted(tx_df["Year"].unique().tolist())
        selected_years = c1.multiselect("Years", years_avail, default=years_avail, key="sales_years")

        selected_countries = c2.multiselect(
            "Countries (leave empty for All 43)",
            options=sorted(tx_df["country"].unique().tolist()),
            default=[],
            placeholder="All countries (or select specific markets)",
            key="sales_countries",
        )

        scope_choice = c3.selectbox(
            "Transaction Scope",
            ["All Valid Transactions (Net)", "Merchandise Only (Excl. Fees)", "Normal Sales Only (Excl. Returns)"],
            key="sales_scope",
        )

    if not selected_years:
        st.warning("Select at least one year to display sales analytics.")
        return

    filtered = tx_df[tx_df["Year"].isin(selected_years)]
    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]
    if scope_choice == "Merchandise Only (Excl. Fees)":
        filtered = filtered[filtered["is_merchandise"]]
    elif scope_choice == "Normal Sales Only (Excl. Returns)":
        filtered = filtered[~filtered["is_cancelled"]]

    if filtered.empty:
        st.warning("No transactions match the selected filters.")
        return

    kpis = compute_sales_kpis(filtered)
    render_kpi_row(
        [
            ("Net Revenue", f"£{kpis['net_revenue']:,.2f}", f"Gross £{kpis['gross_revenue']:,.2f}"),
            ("Purchase Orders", f"{kpis['total_orders']:,}", f"Cancelled invoices: {kpis['cancelled_invoices']:,}"),
            ("Net Quantity Sold", f"{kpis['net_quantity']:,}", f"Gross units: {kpis['gross_quantity']:,}"),
            ("Net Avg Order Value", f"£{kpis['net_aov']:,.2f}", f"Return rate: {kpis['return_rate_pct']:.2f}%"),
        ]
    )

    render_beginner_note(
        "Interpreting Sales Metrics",
        "Note that 2009 covers only December (31 days), 2010 covers all 12 months, and 2011 runs through December 9. "
        "Net Revenue subtracts valid C-prefixed returns from Gross Sales.",
    )

    # Monthly Trend
    monthly = (
        filtered.groupby(["Year", "Month", "InvoiceYearMonth"], observed=True, as_index=False)
        .agg(
            NetRevenue=("TotalAmount", "sum"),
            NetQuantity=("quantity", "sum"),
            Transactions=("invoice", "count"),
        )
        .sort_values(["Year", "Month"])
    )
    monthly["MonthDate"] = pd.to_datetime(monthly["InvoiceYearMonth"].astype(str) + "-01")
    sales_only = filtered[~filtered["is_cancelled"]]
    monthly_orders = (
        sales_only.groupby("InvoiceYearMonth", observed=True)["invoice"]
        .nunique()
        .rename("Orders")
    )
    monthly = monthly.merge(monthly_orders, on="InvoiceYearMonth", how="left")
    monthly["Orders"] = monthly["Orders"].fillna(0).astype(int)

    fig_m = go.Figure()
    fig_m.add_trace(
        go.Scatter(
            x=monthly["MonthDate"],
            y=monthly["NetRevenue"],
            mode="lines+markers",
            name="Net Revenue (£)",
            line=dict(color=PALETTE["primary"], width=3),
        )
    )
    fig_m.update_layout(
        title="Monthly Net Revenue Trend (Filtered)",
        xaxis_title="Month",
        yaxis_title="Net Revenue (£)",
        hovermode="x unified",
    )
    st.plotly_chart(style_plotly(fig_m, height=390), width="stretch")

    # Day of Week & Hourly Patterns
    tb = compute_time_breakdowns(filtered)
    col_l, col_r = st.columns(2)
    with col_l:
        fig_dow = px.bar(
            tb["by_dow"],
            x="DayName",
            y="NetRevenue",
            hover_data={"TotalOrders": ":,"},
            color_discrete_sequence=[PALETTE["primary"]],
            title="Net Revenue & Orders by Day of Week",
            labels={"DayName": "Day", "NetRevenue": "Net Revenue (£)", "TotalOrders": "Orders"},
        )
        st.plotly_chart(style_plotly(fig_dow, height=370), width="stretch")

    with col_r:
        fig_hr = px.line(
            tb["by_hour"],
            x="Hour",
            y="TotalOrders",
            markers=True,
            title="Order Volume by Hour of Day (24h)",
            labels={"Hour": "Hour of Day", "TotalOrders": "Purchase Orders"},
        )
        fig_hr.update_traces(line_color=PALETTE["teal"], line_width=2.5)
        st.plotly_chart(style_plotly(fig_hr, height=370), width="stretch")

    with st.expander("View Monthly Breakdown Table"):
        display_m = monthly[["InvoiceYearMonth", "NetRevenue", "Orders", "NetQuantity", "Transactions"]].copy()
        display_m["NetRevenue"] = display_m["NetRevenue"].round(2)
        st.dataframe(display_m, width="stretch", hide_index=True)
        render_csv_download(display_m, "filtered_monthly_sales.csv", key="dl_sales_monthly")


render()
