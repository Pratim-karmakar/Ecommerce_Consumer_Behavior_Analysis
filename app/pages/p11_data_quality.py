"""Page 11: Data Quality — Audit metrics, missingness, duplicates, cancellations & cleaning rules."""

import plotly.express as px
import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import load_data_quality_report, load_text_report
from app.utils.theme import PALETTE, style_plotly


def render() -> None:
    render_page_header(
        "Data Quality & Cleaning Audit",
        "Transparent audit of raw UCI Online Retail II records (1,067,371 rows) and rule-based cleaning into 1,026,678 validated transactions.",
        badge="Governance",
    )

    dq_df = load_data_quality_report()

    def get_val(metric_name: str) -> str:
        sub = dq_df.loc[dq_df["metric"] == metric_name, "value"]
        return str(sub.iloc[0]) if not sub.empty else "—"

    raw_rows = int(get_val("Total rows"))
    dup_rows = int(get_val("Duplicate records (exact rows)"))
    miss_cust = int(get_val("Missing Customer ID records"))
    c_rows = int(get_val("Cancelled invoice records (C-prefix)"))

    render_kpi_row(
        [
            ("Raw Dataset Rows", f"{raw_rows:,}", "Cleaned: 1,026,678 (96.19%)"),
            ("Exact Duplicates Removed", f"{dup_rows:,}", "22,844 in Dec 1–9, 2010 sheet overlap"),
            ("Missing Customer ID (Raw)", f"{miss_cust:,}", "22.77% raw | 228,994 preserved in cleaned"),
            ("Cancelled ('C') Records", f"{c_rows:,}", "18,855 valid retained in returns table"),
        ]
    )

    render_beginner_note(
        "Distinguishing Cancellations from Warehouse Write-Offs",
        "Not all negative quantities mean customer returns. C-prefixed invoices represent customer cancellations/returns (retained and netted), "
        "whereas 3,393 post-dedup non-C negative-quantity rows all had Price=0 and missing Customer ID (damaged/lost inventory write-offs) and were excluded.",
    )

    col_l, col_r = st.columns(2)
    with col_l:
        missing_chart_df = dq_df[dq_df["metric"].isin(["Missing Customer ID records", "Missing Description records"])].copy()
        missing_chart_df["Count"] = missing_chart_df["value"].astype(int)
        fig_m = px.bar(
            missing_chart_df,
            x="metric",
            y="Count",
            text="Count",
            color_discrete_sequence=[PALETTE["amber"]],
            title="Raw Missing Values by Column (0 Missing Description After Price <= 0 Filter)",
            labels={"metric": "Column", "Count": "Missing Row Count"},
        )
        fig_m.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(style_plotly(fig_m, height=380), width="stretch")

    with col_r:
        anom_metrics = [
            "Duplicate records (exact rows)",
            "Cancelled invoice records (C-prefix)",
            "Zero price records",
            "Non-standard StockCode records",
            "Non-C negative quantity records",
            "Lowercase/note Description records",
            "Unspecified country records",
        ]
        anom_df = dq_df[dq_df["metric"].isin(anom_metrics)].copy()
        anom_df["Count"] = anom_df["value"].astype(int)
        fig_a = px.bar(
            anom_df.sort_values("Count"),
            x="Count",
            y="metric",
            orientation="h",
            color_discrete_sequence=[PALETTE["primary"]],
            title="Audit Categories Across Raw Dataset (Log Scale)",
            labels={"metric": "Audit Category", "Count": "Row Count"},
        )
        fig_a.update_xaxes(type="log")
        st.plotly_chart(style_plotly(fig_a, height=380), width="stretch")

    st.subheader("Complete Data Quality Audit Table")
    st.dataframe(dq_df, width="stretch", hide_index=True)
    render_csv_download(dq_df, "data_quality_report.csv", key="dl_dq_report")

    with st.expander("Read Beginner-Friendly Data Cleaning Summary Report"):
        st.markdown(load_text_report("data_cleaning_report.md"))


render()
