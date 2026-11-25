"""Automated headless tests verifying all 11 Streamlit dashboard pages and interactive controls."""

from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAGES_DIR = PROJECT_ROOT / "app" / "pages"


class TestStreamlitDashboard(unittest.TestCase):
    """Verify app entrypoint and every dashboard page executes without unhandled exceptions."""

    def test_main_entrypoint(self) -> None:
        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Main app raised exception: {at.exception}")

    def test_all_eleven_pages_render_and_interact(self) -> None:
        page_files = [
            "p01_home.py",
            "p02_sales.py",
            "p03_customers.py",
            "p04_products.py",
            "p05_countries.py",
            "p06_future_spending.py",
            "p07_purchase_pred.py",
            "p08_recommendations.py",
            "p09_segmentation.py",
            "p10_model_perf.py",
            "p11_data_quality.py",
        ]
        for fname in page_files:
            with self.subTest(page=fname):
                at = AppTest.from_file(str(PAGES_DIR / fname), default_timeout=45)
                at.run()
                self.assertFalse(at.exception, f"{fname} raised exception: {at.exception}")
                if fname != "p01_home.py":
                    self.assertTrue(len(at.get("download_button")) >= 1, f"{fname} missing CSV download button")

    def test_interactive_controls(self) -> None:
        # Page 2: Sales scope switch
        at_sales = AppTest.from_file(str(PAGES_DIR / "p02_sales.py"), default_timeout=45).run()
        at_sales.selectbox(key="sales_scope").select("Merchandise Only (Excl. Fees)").run()
        self.assertFalse(at_sales.exception)

        # Page 4: Product search filter
        at_prod = AppTest.from_file(str(PAGES_DIR / "p04_products.py"), default_timeout=30).run()
        at_prod.text_input(key="prod_kw").set_value("CAKESTAND").run()
        self.assertFalse(at_prod.exception)

        # Page 6: Future Spending simulator mode
        at_lr = AppTest.from_file(str(PAGES_DIR / "p06_future_spending.py"), default_timeout=45).run()
        at_lr.radio(key="lr_mode").set_value("Interactive What-If Feature Simulator").run()
        self.assertFalse(at_lr.exception)
        self.assertTrue(len(at_lr.success) >= 1)

        # Page 7: Purchase Prediction threshold slider
        at_clf = AppTest.from_file(str(PAGES_DIR / "p07_purchase_pred.py"), default_timeout=45).run()
        at_clf.slider(key="clf_threshold").set_value(0.56).run()
        self.assertFalse(at_clf.exception)

        # Page 8: Cold-start Customer ID, Invalid Customer ID & Unknown Product graceful handling
        at_knn = AppTest.from_file(str(PAGES_DIR / "p08_recommendations.py"), default_timeout=45).run()
        at_knn.text_input(key="knn_custom_cid").set_value("999999").run()
        at_knn.text_input(key="knn_custom_sku").set_value("UNKNOWN_SKU_XYZ").run()
        self.assertFalse(at_knn.exception)
        self.assertTrue(len(at_knn.warning) >= 2)
        at_knn.text_input(key="knn_custom_cid").set_value("not_an_int").run()
        self.assertFalse(at_knn.exception)
        self.assertTrue(len(at_knn.error) >= 1)

        # Page 9: Segmentation custom RFM with negative MonetaryValue
        at_seg = AppTest.from_file(str(PAGES_DIR / "p09_segmentation.py"), default_timeout=45).run()
        at_seg.radio(key="seg_mode").set_value("Score Custom RFM Values").run()
        at_seg.number_input(key="seg_sim_m").set_value(-50.0).run()
        self.assertFalse(at_seg.exception)
        self.assertTrue(any("Net-Return Accounts" in s.value for s in at_seg.success))


if __name__ == "__main__":
    unittest.main()
