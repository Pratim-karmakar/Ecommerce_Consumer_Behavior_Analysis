"""Page 8: Product Recommendation — Item-Based KNN collaborative filtering & cold-start handling."""

import streamlit as st

from app.components.ui import (
    render_beginner_note,
    render_csv_download,
    render_kpi_row,
    render_page_header,
)
from app.utils.data_loader import get_knn_recommender_model, load_model_metric_tables
from src.models.inference import (
    InvalidInputError,
    ModelLoadError,
    UnknownProductError,
    find_similar_products,
    recommend_products,
)


def render() -> None:
    render_page_header(
        "Product Recommendation",
        "Generate real-data Item-Based KNN recommendations and co-purchase product matches with transparent cold-start fallback.",
        badge="Machine Learning",
    )

    try:
        knn_model = get_knn_recommender_model()
        knn_metrics = load_model_metric_tables()["knn_recommender"]
    except (ModelLoadError, FileNotFoundError) as exc:
        st.error(f"Could not load KNN recommender assets: {exc}")
        return

    k10_row = knn_metrics[(knn_metrics["model_name"] == "Item-Based KNN (Cosine)") & (knn_metrics["k"] == 10)].iloc[0]

    render_kpi_row(
        [
            ("Customers Indexed", f"{knn_model.matrix_.shape[0]:,}", "With supported merchandise history"),
            ("Catalog SKUs (>=5 Buyers)", f"{knn_model.matrix_.shape[1]:,}", "Cosine KNN item graph"),
            ("Holdout Hit Rate@10", f"{k10_row['hit_rate_at_k']*100:.2f}%", f"vs. {knn_metrics.loc[1, 'hit_rate_at_k']*100:.2f}% popularity baseline"),
            ("Catalog Coverage@10", f"{k10_row['coverage_pct']:.2f}%", f"{int(k10_row['unique_items_recommended']):,} unique SKUs recommended"),
        ]
    )

    render_beginner_note(
        "How Recommendations Are Generated",
        "Recommendations use only real customer–product purchase quantities (dampened with log1p) and Cosine similarity. "
        "No fake reviews or demographic profiles exist. Unseen or low-history customers automatically receive popular bestsellers via cold-start fallback.",
    )

    tab_cust, tab_item = st.tabs(["Customer Recommendations", "Similar Product Explorer"])

    with tab_cust:
        c1, c2, c3, c4 = st.columns([1.3, 1.0, 0.9, 0.9])
        known_cids = [int(c) for c in knn_model.customer_ids_]
        default_idx = known_cids.index(12367) if 12367 in known_cids else 0
        sel_known = c1.selectbox("Select Known Customer ID", options=known_cids, index=default_idx, key="knn_sel_cid")
        custom_cid_str = c2.text_input("Or Enter Custom / Cold-Start ID (e.g., 999999)", "", key="knn_custom_cid").strip()
        top_n = c3.slider("Top N Recommendations", 5, 20, 10, key="knn_top_n")
        exclude_seen = c4.checkbox("Exclude Already Purchased", value=True, key="knn_excl_seen")

        target_cid = custom_cid_str if custom_cid_str else sel_known

        try:
            res = recommend_products(target_cid, top_n=top_n, exclude_purchased=exclude_seen, recommender=knn_model)
            if res["is_cold_start"]:
                st.warning(
                    f"Customer `{res['customer_id']}` has no historical supported merchandise purchases — "
                    "displaying **Cold-Start Popularity Fallback** recommendations."
                )
            else:
                st.success(f"Personalized Item-KNN recommendations for **Customer `{res['customer_id']}`**.")

            col_h, col_r = st.columns([1.0, 1.35])
            with col_h:
                st.subheader("Top Historical Purchases")
                if res["history_preview"].empty:
                    st.info("No purchase history recorded in collaborative filtering matrix.")
                else:
                    st.dataframe(res["history_preview"], width="stretch", hide_index=True)

            with col_r:
                st.subheader(f"Top {len(res['recommendations'])} Recommended Products")
                st.dataframe(res["recommendations"], width="stretch", hide_index=True)
                render_csv_download(
                    res["recommendations"],
                    f"recommendations_customer_{res['customer_id']}.csv",
                    key="dl_knn_recs",
                )
        except InvalidInputError as exc:
            st.error(f"Invalid Customer ID input: {exc}")

    with tab_item:
        item_codes = [str(s) for s in knn_model.item_ids_]
        default_item_idx = item_codes.index("22423") if "22423" in item_codes else 0
        col_i1, col_i2, col_i3 = st.columns([1.4, 1.0, 0.8])
        sel_sku = col_i1.selectbox("Select Seed StockCode", options=item_codes, index=default_item_idx, key="knn_seed_sku")
        custom_sku = col_i2.text_input("Or Type StockCode", "", key="knn_custom_sku").strip()
        item_top_n = col_i3.slider("Similar Items Count", 5, 20, 8, key="knn_item_n")

        query_sku = custom_sku if custom_sku else sel_sku
        try:
            sim_df = find_similar_products(query_sku, top_n=item_top_n, strict=True, recommender=knn_model)
            seed_desc = knn_model.item_catalog_.loc[query_sku.strip().upper(), "description"]
            st.markdown(f"**Seed Product:** `{query_sku.strip().upper()}` — **{seed_desc}**")
            st.dataframe(sim_df, width="stretch", hide_index=True)
            render_csv_download(sim_df, f"similar_products_{query_sku.strip().upper()}.csv", key="dl_knn_sim")
        except (UnknownProductError, InvalidInputError) as exc:
            st.warning(str(exc))


render()
