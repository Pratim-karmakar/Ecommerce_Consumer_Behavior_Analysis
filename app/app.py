"""Main Streamlit application entrypoint for E-Commerce Customer Analytics."""

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
APP_DIR_STR = str(APP_DIR)
PROJECT_ROOT_STR = str(PROJECT_ROOT)

# Prevent Streamlit's implicit script-directory sys.path entry (`app/`) from
# shadowing the `app` package via `app/app.py` when running `streamlit run app/app.py`.
while APP_DIR_STR in sys.path:
    sys.path.remove(APP_DIR_STR)
while PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.utils.theme import apply_custom_theme  # noqa: E402


def main() -> None:
    """Configure Streamlit navigation, theme styling, and sidebar project metadata."""
    st.set_page_config(
        page_title="E-Commerce Customer Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_theme()

    pages_dir = Path(__file__).resolve().parent / "pages"

    nav = st.navigation(
        {
            "Overview": [
                st.Page(str(pages_dir / "p01_home.py"), title="Home", icon="🏠", default=True),
            ],
            "Business Analytics": [
                st.Page(str(pages_dir / "p02_sales.py"), title="Sales Analytics", icon="📈"),
                st.Page(str(pages_dir / "p03_customers.py"), title="Customer Analytics", icon="👥"),
                st.Page(str(pages_dir / "p04_products.py"), title="Product Analytics", icon="📦"),
                st.Page(str(pages_dir / "p05_countries.py"), title="Country Analytics", icon="🌍"),
            ],
            "Predictive & Recommender ML": [
                st.Page(str(pages_dir / "p06_future_spending.py"), title="Future Spending Prediction", icon="💷"),
                st.Page(str(pages_dir / "p07_purchase_pred.py"), title="Purchase Prediction", icon="🎯"),
                st.Page(str(pages_dir / "p08_recommendations.py"), title="Product Recommendation", icon="🛍️"),
                st.Page(str(pages_dir / "p09_segmentation.py"), title="Customer Segmentation", icon="🧩"),
            ],
            "Evaluation & Governance": [
                st.Page(str(pages_dir / "p10_model_perf.py"), title="Model Performance", icon="📊"),
                st.Page(str(pages_dir / "p11_data_quality.py"), title="Data Quality", icon="🛡️"),
            ],
        }
    )

    with st.sidebar:
        st.divider()
        st.caption("**Project:** E-Commerce Customer Analytics")
        st.caption("**Dataset:** UCI Online Retail II (`2009-12` to `2011-12`)")
        st.caption("**Data:** `1,026,678` validated transactions | `4` saved `.joblib` models")

    nav.run()


if __name__ == "__main__":
    main()
