"""Consistent visual theme, color palette, and CSS helpers for the Streamlit dashboard."""

import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "primary": "#1E3A8A",
    "accent": "#2563EB",
    "teal": "#0F766E",
    "amber": "#D97706",
    "rose": "#DC2626",
    "slate": "#475569",
    "bg_card": "#F8FAFC",
    "border": "#E2E8F0",
}

SEGMENT_COLORS = {
    "VIP Accounts": "#1E3A8A",
    "Lapsing Regulars": "#D97706",
    "Active Growth": "#0F766E",
    "Dormant Buyers": "#64748B",
    "Net-Return Accounts": "#DC2626",
}


def apply_custom_theme() -> None:
    """Inject clean, restrained custom CSS for cards, typography, and spacing."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.75rem;
            padding-bottom: 2.2rem;
            max-width: 1260px;
        }
        .kpi-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            margin-bottom: 0.6rem;
        }
        .kpi-label {
            font-size: 0.80rem;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .kpi-delta {
            font-size: 0.80rem;
            color: #64748B;
            margin-top: 0.25rem;
        }
        .info-banner {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 4px solid #1E3A8A;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1.0rem 0;
            color: #1E293B;
            font-size: 0.92rem;
        }
        .page-badge {
            display: inline-block;
            background-color: #DBEAFE;
            color: #1E3A8A;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.35;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            margin-top: 0.2rem;
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply consistent margins, height, and typography while respecting active Streamlit theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=25, t=55, b=40),
        font=dict(size=12),
        title_font=dict(size=15),
    )
    return fig
