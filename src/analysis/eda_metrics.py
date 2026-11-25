"""Reusable business intelligence and EDA aggregation helpers."""

from typing import Any, Dict, Tuple

import pandas as pd


def compute_sales_kpis(tx_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall sales, cancellation, quantity, and AOV KPIs."""
    sales = tx_df[~tx_df["is_cancelled"]]
    returns = tx_df[tx_df["is_cancelled"]]

    net_revenue = float(tx_df["TotalAmount"].sum())
    gross_revenue = float(sales["TotalAmount"].sum())
    returned_amount = float(returns["TotalAmount"].sum())
    total_orders = int(sales["invoice"].nunique())
    cancelled_invoices = int(returns["invoice"].nunique())
    net_quantity = int(tx_df["quantity"].sum())
    gross_quantity = int(sales["quantity"].sum())

    return {
        "net_revenue": round(net_revenue, 2),
        "gross_revenue": round(gross_revenue, 2),
        "returned_amount": round(returned_amount, 2),
        "return_rate_pct": round((abs(returned_amount) / gross_revenue * 100) if gross_revenue else 0.0, 2),
        "total_orders": total_orders,
        "cancelled_invoices": cancelled_invoices,
        "net_quantity": net_quantity,
        "gross_quantity": gross_quantity,
        "net_aov": round((net_revenue / total_orders) if total_orders else 0.0, 2),
        "gross_aov": round((gross_revenue / total_orders) if total_orders else 0.0, 2),
    }


def compute_period_summaries(tx_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute yearly and year-quarter performance tables from transaction features."""
    sales = tx_df[~tx_df["is_cancelled"]]

    def _agg_period(group_cols: list[str]) -> pd.DataFrame:
        net = tx_df.groupby(group_cols, as_index=False).agg(
            NetRevenue=("TotalAmount", "sum"),
            NetQuantity=("quantity", "sum"),
            UniqueCustomers=("customer_id", "nunique"),
        )
        gr = sales.groupby(group_cols, as_index=False).agg(
            GrossRevenue=("TotalAmount", "sum"),
            TotalOrders=("invoice", "nunique"),
        )
        merged = net.merge(gr, on=group_cols, how="left")
        merged["TotalOrders"] = merged["TotalOrders"].fillna(0).astype("int64")
        merged["NetRevenue"] = merged["NetRevenue"].round(2)
        merged["GrossRevenue"] = merged["GrossRevenue"].fillna(0.0).round(2)
        merged["NetAOV"] = (merged["NetRevenue"] / merged["TotalOrders"].replace(0, pd.NA)).round(2)
        return merged

    yearly = _agg_period(["Year"]).sort_values("Year").reset_index(drop=True)
    quarterly = _agg_period(["Year", "Quarter"]).sort_values(["Year", "Quarter"]).reset_index(drop=True)
    quarterly["YearQuarter"] = quarterly["Year"].astype(str) + "-Q" + quarterly["Quarter"].astype(str)
    return yearly, quarterly


def compute_time_breakdowns(tx_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute day-of-week, hourly, weekday-vs-weekend, and calendar-month aggregations."""
    sales = tx_df[~tx_df["is_cancelled"]]

    def _agg(cols: list[str]) -> pd.DataFrame:
        net = tx_df.groupby(cols, as_index=False).agg(
            NetRevenue=("TotalAmount", "sum"),
            NetQuantity=("quantity", "sum"),
            Transactions=("invoice", "count"),
        )
        gr = sales.groupby(cols, as_index=False).agg(TotalOrders=("invoice", "nunique"))
        out = net.merge(gr, on=cols, how="left")
        out["TotalOrders"] = out["TotalOrders"].fillna(0).astype("int64")
        out["NetRevenue"] = out["NetRevenue"].round(2)
        return out

    by_dow = _agg(["DayOfWeek", "DayName"]).sort_values("DayOfWeek").reset_index(drop=True)
    by_hour = _agg(["Hour"]).sort_values("Hour").reset_index(drop=True)
    by_weekend = _agg(["IsWeekend"]).sort_values("IsWeekend").reset_index(drop=True)
    by_weekend["DayType"] = by_weekend["IsWeekend"].map({False: "Weekday (Mon–Fri)", True: "Weekend (Sat–Sun)"})
    by_month_cal = _agg(["Month", "MonthName"]).sort_values("Month").reset_index(drop=True)

    heatmap_pivot = (
        sales.groupby(["DayOfWeek", "DayName", "Hour"], as_index=False)["invoice"]
        .nunique()
        .pivot(index="DayName", columns="Hour", values="invoice")
        .fillna(0)
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    return {
        "by_dow": by_dow,
        "by_hour": by_hour,
        "by_weekend": by_weekend,
        "by_month_cal": by_month_cal,
        "day_hour_heatmap": heatmap_pivot,
    }


def compute_product_pareto(product_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute cumulative revenue share across positive-revenue SKUs (Pareto 80/20)."""
    pos = product_df[product_df["TotalRevenue"] > 0].sort_values("TotalRevenue", ascending=False).copy()
    total_pos_rev = float(pos["TotalRevenue"].sum())
    pos["CumulativeRevenue"] = pos["TotalRevenue"].cumsum()
    pos["CumulativeRevenuePct"] = (pos["CumulativeRevenue"] / total_pos_rev) * 100.0
    pos["Rank"] = range(1, len(pos) + 1)
    pos["ProductSharePct"] = (pos["Rank"] / len(pos)) * 100.0

    n_80 = int((pos["CumulativeRevenuePct"] <= 80.0).sum()) + 1
    top_20_n = max(1, int(round(len(pos) * 0.20)))
    top_20_rev_pct = float(pos.head(top_20_n)["TotalRevenue"].sum() / total_pos_rev * 100.0)

    stats = {
        "total_merchandise_skus": len(product_df),
        "positive_revenue_skus": len(pos),
        "non_positive_revenue_skus": int((product_df["TotalRevenue"] <= 0).sum()),
        "skus_for_80pct_revenue": n_80,
        "pct_skus_for_80pct_revenue": round(n_80 / len(pos) * 100.0, 2),
        "top_20pct_sku_revenue_share": round(top_20_rev_pct, 2),
    }
    return pos, stats


def compute_customer_concentration(customer_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute repeat-buyer rate, mean/median spend, and top-20% revenue concentration."""
    total_customers = len(customer_df)
    repeat_buyers = int((customer_df["Frequency"] >= 2).sum())
    one_time_buyers = total_customers - repeat_buyers

    pos = customer_df[customer_df["TotalSpending"] > 0].sort_values("TotalSpending", ascending=False)
    top_20_n = max(1, int(round(total_customers * 0.20)))
    total_cust_spend = float(customer_df["TotalSpending"].sum())
    top_20_spend_pct = float(customer_df.sort_values("TotalSpending", ascending=False).head(top_20_n)["TotalSpending"].sum() / total_cust_spend * 100.0)

    return {
        "total_purchasing_customers": total_customers,
        "repeat_buyers": repeat_buyers,
        "one_time_buyers": one_time_buyers,
        "repeat_rate_pct": round(repeat_buyers / total_customers * 100.0, 2),
        "mean_spending": round(float(customer_df["TotalSpending"].mean()), 2),
        "median_spending": round(float(customer_df["TotalSpending"].median()), 2),
        "mean_frequency": round(float(customer_df["Frequency"].mean()), 2),
        "median_frequency": round(float(customer_df["Frequency"].median()), 2),
        "median_recency": round(float(customer_df["Recency"].median()), 2),
        "positive_spending_customers": len(pos),
        "non_positive_spending_customers": total_customers - len(pos),
        "top_20pct_customer_spend_share": round(top_20_spend_pct, 2),
    }
