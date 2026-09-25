"""Reusable KPI functions (used by the notebooks and the Streamlit dashboard)."""
from __future__ import annotations

import pandas as pd


def _shipped(orders: pd.DataFrame) -> pd.DataFrame:
    """Delivery KPIs only make sense for orders that were not cancelled."""
    return orders[orders["is_cancelled"] == 0]


def headline_kpis(orders: pd.DataFrame) -> dict:
    s = _shipped(orders)
    return {
        "Total orders": len(orders),
        "Net sales": orders["net_sales"].sum(),
        "Profit": orders["profit"].sum(),
        "Profit margin %": 100 * orders["profit"].sum() / orders["net_sales"].sum(),
        "On-time delivery %": 100 * s["is_on_time"].mean(),
        "Late delivery %": 100 * s["is_late"].mean(),
        "Avg actual lead time (days)": s["actual_days"].mean(),
        "Avg scheduled lead time (days)": s["scheduled_days"].mean(),
        "Avg delay when late (days)": s.loc[s["is_late"] == 1, "delay_days"].mean(),
        "Cancellation / fraud %": 100 * orders["is_cancelled"].mean(),
        "Order fulfilment %": 100 * orders["order_status"].isin(["COMPLETE", "CLOSED"]).mean(),
    }


def delivery_by(orders: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Delivery performance table grouped by any dimension (mode, region, market...)."""
    s = _shipped(orders)
    out = s.groupby(dim).agg(
        orders=("order_id", "count"),
        late_pct=("is_late", "mean"),
        on_time_pct=("is_on_time", "mean"),
        avg_scheduled_days=("scheduled_days", "mean"),
        avg_actual_days=("actual_days", "mean"),
        avg_delay_days=("delay_days", "mean"),
        net_sales=("net_sales", "sum"),
    )
    out[["late_pct", "on_time_pct"]] *= 100
    return out.sort_values("late_pct", ascending=False).round(2)


def monthly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    s = _shipped(orders)
    t = s.groupby("order_month").agg(
        orders=("order_id", "count"),
        late_pct=("is_late", "mean"),
        net_sales=("net_sales", "sum"),
        profit=("profit", "sum"),
    )
    t["late_pct"] *= 100
    return t.round(2)


def abc_classification(items: pd.DataFrame) -> pd.DataFrame:
    """ABC (Pareto) inventory classification by revenue: A=top 80%, B=next 15%, C=last 5%."""
    p = items.groupby(["product", "category"]).agg(
        units=("quantity", "sum"), net_sales=("net_sales", "sum"), profit=("profit", "sum"),
        orders=("order_id", "nunique"),
    ).sort_values("net_sales", ascending=False).reset_index()
    p["revenue_share_%"] = 100 * p["net_sales"] / p["net_sales"].sum()
    p["cumulative_%"] = p["revenue_share_%"].cumsum()
    p["abc_class"] = pd.cut(p["cumulative_%"].shift(fill_value=0), [-1, 80, 95, 101], labels=["A", "B", "C"])
    p["profit_margin_%"] = 100 * p["profit"] / p["net_sales"]
    return p.round(2)
