"""Page 4: Product Analytics — Best-selling items, revenue leaders, ASP, and SKU filters."""

import plotly.express as px
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import load_product_analytics
from app.utils.theme import PALETTE, style_plotly


def render() -> None:
    render_page_header(
        "Product Analytics",
        "Inspect net sales performance, unit demand, Average Selling Price, and return impact across 4,746 merchandise SKUs.",
        badge="Analytics",
    )

    prod_df = load_product_analytics()

    with st.expander("Filter Products", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        keyword = c1.text_input("Search StockCode / Description", "", key="prod_kw").strip()
        min_reach = c2.slider("Min Unique Customers", 1, 200, 1, key="prod_reach")
        excl_non_pos = c3.checkbox("Exclude Net Revenue <= £0", value=True, key="prod_excl_neg")
        top_n = c4.slider("Top N Products", 5, 25, 10, key="prod_top_n")

    filtered = prod_df[prod_df["UniqueCustomers"] >= min_reach].copy()
    if excl_non_pos:
        filtered = filtered[filtered["TotalRevenue"] > 0]
    if keyword:
        mask = (
            filtered["StockCode"].str.contains(keyword, case=False, na=False)
            | filtered["Description"].str.contains(keyword, case=False, na=False)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.warning("No merchandise SKUs match the selected filters.")
        return

    render_kpi_row(
        [
            ("Merchandise SKUs Shown", f"{len(filtered):,}", f"Out of {len(prod_df):,} total SKUs"),
            ("Net Merchandise Revenue", f"£{filtered['TotalRevenue'].sum():,.2f}", f"Gross £{filtered['GrossRevenue'].sum():,.2f}"),
            ("Net Units Sold", f"{int(filtered['TotalQuantitySold'].sum()):,}", f"Returned {int(filtered['ReturnedQuantity'].sum()):,}"),
            ("Median Selling Price", f"£{filtered['AverageSellingPrice'].median():,.2f}", f"Mean £{filtered['AverageSellingPrice'].mean():,.2f}"),
        ]
    )

    render_beginner_note(
        "Product Filtering Rule",
        "Non-merchandise fee/service codes (POST, DOT, C2, D, M) are excluded so shipping fees never crowd out retail products. "
        "TotalRevenue and TotalQuantitySold reflect net realized figures after C-prefixed returns.",
    )

    col_l, col_r = st.columns(2)
    with col_l:
        top_rev = filtered.sort_values("TotalRevenue", ascending=False).head(top_n).copy()
        top_rev["Label"] = top_rev["StockCode"] + " — " + top_rev["Description"].str.slice(0, 24)
        fig_r = px.bar(
            top_rev.sort_values("TotalRevenue"),
            x="TotalRevenue",
            y="Label",
            orientation="h",
            hover_data={"AverageSellingPrice": ":.2f", "TotalQuantitySold": ":,"},
            color_discrete_sequence=[PALETTE["primary"]],
            title=f"Top {len(top_rev)} Products by Net Revenue",
            labels={"TotalRevenue": "Net Revenue (£)", "Label": "Product", "AverageSellingPrice": "ASP (£)"},
        )
        st.plotly_chart(style_plotly(fig_r, height=410), width="stretch")

    with col_r:
        top_qty = filtered.sort_values("TotalQuantitySold", ascending=False).head(top_n).copy()
        top_qty["Label"] = top_qty["StockCode"] + " — " + top_qty["Description"].str.slice(0, 24)
        fig_q = px.bar(
            top_qty.sort_values("TotalQuantitySold"),
            x="TotalQuantitySold",
            y="Label",
            orientation="h",
            color_discrete_sequence=[PALETTE["teal"]],
            title=f"Top {len(top_qty)} Products by Net Units Sold",
            labels={"TotalQuantitySold": "Net Units Sold", "Label": "Product"},
        )
        st.plotly_chart(style_plotly(fig_q, height=410), width="stretch")

    scatter_df = filtered[filtered["TotalRevenue"] > 0].sort_values("TotalRevenue", ascending=False).head(150)
    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df,
            x="TotalQuantitySold",
            y="TotalRevenue",
            size="TotalInvoices",
            color="AverageSellingPrice",
            hover_name="Description",
            hover_data={"StockCode": True, "UniqueCustomers": ":,", "AverageSellingPrice": ":.2f"},
            color_continuous_scale="Viridis",
            title="Top 150 Filtered Products: Units Sold vs. Net Revenue (Bubble Size = Invoices)",
            labels={"TotalQuantitySold": "Net Units Sold", "TotalRevenue": "Net Revenue (£)", "AverageSellingPrice": "ASP (£)"},
        )
        st.plotly_chart(style_plotly(fig_sc, height=420), width="stretch")

    st.subheader("Filtered Product Catalog Table")
    st.dataframe(filtered.head(500), width="stretch", hide_index=True)
    render_csv_download(filtered, "filtered_product_analytics.csv", key="dl_products")


render()
