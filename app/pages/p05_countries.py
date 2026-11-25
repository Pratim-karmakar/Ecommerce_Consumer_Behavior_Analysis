"""Page 5: Country Analytics — Geographic revenue, order volume, AOV, and customer distribution."""

import numpy as np
import plotly.express as px
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import load_country_analytics
from app.utils.theme import PALETTE, style_plotly


def render() -> None:
    render_page_header(
        "Country Analytics",
        "Compare domestic UK performance against 42 international export markets by revenue, orders, customers, and Average Order Value.",
        badge="Analytics",
    )

    ctry_df = load_country_analytics()

    with st.expander("Geographic Display Options", expanded=True):
        c1, c2, c3 = st.columns(3)
        include_uk = c1.checkbox("Include United Kingdom (85.1% of total)", value=False, key="ctry_uk")
        excl_unspec = c2.checkbox("Exclude 'Unspecified' Country", value=False, key="ctry_unspec")
        top_n = c3.slider("Top N Countries to Chart", 5, 25, 12, key="ctry_top_n")

    filtered = ctry_df.copy()
    if not include_uk:
        filtered = filtered[filtered["Country"] != "United Kingdom"]
    if excl_unspec:
        filtered = filtered[filtered["Country"] != "Unspecified"]

    if filtered.empty:
        st.warning("No countries match the selected options.")
        return

    render_kpi_row(
        [
            ("Markets Displayed", f"{len(filtered):,}", f"Total dataset: {len(ctry_df):,} markets"),
            ("Net Revenue (Shown)", f"£{filtered['TotalRevenue'].sum():,.2f}", f"Gross £{filtered['GrossRevenue'].sum():,.2f}"),
            ("Purchase Orders (Shown)", f"{int(filtered['TotalOrders'].sum()):,}", f"Cancelled {int(filtered['CancelledOrders'].sum()):,}"),
            ("Identified Customers", f"{int(filtered['TotalCustomers'].sum()):,}", "Distinct non-null IDs per country"),
        ]
    )

    render_beginner_note(
        "Understanding International Market Profiles",
        "Toggle 'Include United Kingdom' off to examine export hubs clearly. Notice EIRE generates over £608k from just 3 wholesale accounts, "
        "whereas Germany (107 customers) and France (94 customers) represent broader adoption. Bermuda and Hong Kong contain guest orders (0 Customer IDs).",
    )

    top_view = filtered.sort_values("TotalRevenue", ascending=False).head(top_n)
    col_l, col_r = st.columns(2)

    with col_l:
        fig_rev = px.bar(
            top_view.sort_values("TotalRevenue"),
            x="TotalRevenue",
            y="Country",
            orientation="h",
            hover_data={"AverageOrderValue": ":.2f", "TotalOrders": ":,"},
            color_discrete_sequence=[PALETTE["primary"]],
            title=f"Top {len(top_view)} Markets by Net Revenue",
            labels={"TotalRevenue": "Net Revenue (£)", "AverageOrderValue": "Net AOV (£)"},
        )
        st.plotly_chart(style_plotly(fig_rev, height=410), width="stretch")

    with col_r:
        fig_ord = px.bar(
            top_view.sort_values("TotalOrders"),
            x="TotalOrders",
            y="Country",
            orientation="h",
            hover_data={"TotalCustomers": ":,"},
            color_discrete_sequence=[PALETTE["teal"]],
            title=f"Top {len(top_view)} Markets by Purchase Orders & Customers",
            labels={"TotalOrders": "Completed Orders", "TotalCustomers": "Identified Customers"},
        )
        st.plotly_chart(style_plotly(fig_ord, height=410), width="stretch")

    pos_ctry = filtered[filtered["TotalRevenue"] > 0].copy()
    if not pos_ctry.empty:
        pos_ctry["MarketType"] = np.where(pos_ctry["Country"] == "United Kingdom", "Domestic (UK)", "International")
        fig_tree = px.treemap(
            pos_ctry,
            path=["MarketType", "Country"],
            values="TotalRevenue",
            color="AverageOrderValue",
            hover_data={"TotalOrders": ":,", "TotalCustomers": ":,", "ReturnedAmount": ":,.2f"},
            color_continuous_scale="Blues",
            title="Market Share & Average Order Value Treemap",
        )
        st.plotly_chart(style_plotly(fig_tree, height=430), width="stretch")

    st.subheader("Country Performance Table")
    st.dataframe(filtered, width="stretch", hide_index=True)
    render_csv_download(filtered, "country_analytics_filtered.csv", key="dl_countries")


render()
