"""Reusable UI components for KPI cards, page headers, beginner callouts, and CSV exports."""

import html
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st


def render_page_header(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """Render a clean, consistent page header with an optional category pill."""
    if badge:
        st.markdown(f'<span class="page-badge">{html.escape(badge)}</span>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def render_kpi_row(metrics: List[Tuple[str, str, Optional[str]]]) -> None:
    """Render equal-width styled KPI cards from a list of (label, value, delta_note)."""
    cols = st.columns(len(metrics))
    for col, (label, value, note) in zip(cols, metrics):
        note_html = f'<div class="kpi-delta">{html.escape(note)}</div>' if note else ""
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{html.escape(label)}</div>
                <div class="kpi-value">{html.escape(value)}</div>
                {note_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_beginner_note(title: str, body: str) -> None:
    """Display a beginner-friendly explanation callout box."""
    st.markdown(
        f'<div class="info-banner"><strong>{html.escape(title)}</strong><br>{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def render_csv_download(
    df: pd.DataFrame, filename: str, label: str = "Download Data (CSV)", key: Optional[str] = None
) -> None:
    """Render a standardized UTF-8 CSV download button for filtered tables."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
    )
