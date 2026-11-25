"""Page 10: Model Performance — Side-by-side verified evaluation across all 4 ML components."""

import plotly.express as px
import streamlit as st

from app.components.ui import render_beginner_note, render_csv_download, render_page_header
from app.utils.data_loader import VIZ_DIR, load_model_metric_tables, load_text_report
from app.utils.theme import PALETTE, style_plotly


def render() -> None:
    render_page_header(
        "Model Performance & Evaluation",
        "Review verified chronological metrics, baselines, and diagnostic plots independently for each of the four ML models.",
        badge="Evaluation",
    )

    render_beginner_note(
        "Why Metrics Are Kept Separate by Problem Type",
        "Regression (MAE/RMSE/R²), Classification (ROC-AUC/PR-AUC/F1), Recommendation (Precision@K/Hit Rate@K/Coverage), "
        "and Unsupervised Clustering (Silhouette/Inertia) measure fundamentally different tasks and must never be compared on a single scale.",
    )

    tables = load_model_metric_tables()

    tab_lr, tab_clf, tab_knn, tab_km, tab_summary = st.tabs(
        ["Linear Regression", "Logistic Regression", "KNN Recommender", "K-Means Segmentation", "Evaluation Report"]
    )

    with tab_lr:
        st.subheader("1. Linear Regression — 90-Day Future Spending")
        df_lr = tables["linear_regression"]
        st.dataframe(df_lr, width="stretch", hide_index=True)
        render_csv_download(df_lr, "linear_regression_results.csv", key="dl_perf_lr")

        test_lr = df_lr[df_lr["split"].isin(["test", "test_excl_top1pct_outliers"])].copy()
        test_lr["Label"] = test_lr["model_name"] + " (" + test_lr["split"] + ")"
        fig_lr = px.bar(
            test_lr,
            x="Label",
            y="rmse",
            hover_data={"r2_score": ":.4f", "mae": ":,.2f"},
            color_discrete_sequence=[PALETTE["primary"]],
            title="RMSE Comparison Across Baselines & Outlier Checks",
        )
        fig_lr.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(style_plotly(fig_lr, height=380), width="stretch")

        img_path = VIZ_DIR / "05_lr_actual_vs_predicted.png"
        if img_path.is_file():
            st.image(str(img_path), caption="Actual vs. Predicted 90-Day Spending (Notebook 05)", width="stretch")

    with tab_clf:
        st.subheader("2. Logistic Regression — 90-Day Future Purchase Classification")
        df_clf = tables["logistic_regression"]
        st.dataframe(df_clf, width="stretch", hide_index=True)
        render_csv_download(df_clf, "logistic_regression_results.csv", key="dl_perf_clf")

        test_clf = df_clf[df_clf["split"] == "test"]
        fig_clf = px.bar(
            test_clf,
            x="model_name",
            y=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            barmode="group",
            title="Classification Metrics Across Test Operating Points",
        )
        fig_clf.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(style_plotly(fig_clf, height=400), width="stretch")

        img_roc = VIZ_DIR / "06_logreg_roc_and_pr_curves.png"
        if img_roc.is_file():
            st.image(str(img_roc), caption="ROC & Precision-Recall Curves (Notebook 06)", width="stretch")

    with tab_knn:
        st.subheader("3. Item-Based KNN — Chronological Holdout Recommendation")
        df_knn = tables["knn_recommender"]
        st.dataframe(df_knn, width="stretch", hide_index=True)
        render_csv_download(df_knn, "knn_recommender_results.csv", key="dl_perf_knn")

        img_knn = VIZ_DIR / "07_knn_holdout_evaluation.png"
        if img_knn.is_file():
            st.image(str(img_knn), caption="Chronological Holdout @10 Accuracy & Coverage (Notebook 07)", width="stretch")

    with tab_km:
        st.subheader("4. K-Means — RFM Customer Segmentation (K = 4 + Cluster -1)")
        df_km = tables["kmeans_summary"]
        st.dataframe(df_km, width="stretch", hide_index=True)
        render_csv_download(df_km, "kmeans_cluster_summary.csv", key="dl_perf_km")

        c1, c2 = st.columns(2)
        p_elb = VIZ_DIR / "08_kmeans_elbow_silhouette.png"
        p_pca = VIZ_DIR / "08_kmeans_pca_2d.png"
        if p_elb.is_file():
            c1.image(str(p_elb), caption="Elbow & Silhouette Evidence for K = 4", width="stretch")
        if p_pca.is_file():
            c2.image(str(p_pca), caption="2D PCA Cluster Projection (95.19% Variance)", width="stretch")

    with tab_summary:
        st.subheader("Unified Model Evaluation Summary Report")
        md_text = load_text_report("model_evaluation_summary.md")
        st.markdown(md_text)


render()
