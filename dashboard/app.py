"""
Supply Chain & Logistics Performance Dashboard (Streamlit)

Run from the repo root:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from cleaning import clean, load_raw, to_orders  # noqa: E402
from config import CLEAN_ITEMS_PATH, CLEAN_ORDERS_PATH  # noqa: E402
from kpis import abc_classification, delivery_by, headline_kpis, monthly_trend  # noqa: E402

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"

st.set_page_config(page_title="Supply Chain Performance", page_icon="🚚", layout="wide")


@st.cache_data
def load_data():
    """Use processed files if they exist, otherwise build them from raw data."""
    if CLEAN_ITEMS_PATH.exists() and CLEAN_ORDERS_PATH.exists():
        items = pd.read_csv(CLEAN_ITEMS_PATH, parse_dates=["order_date", "shipping_date"])
        orders = pd.read_csv(CLEAN_ORDERS_PATH, parse_dates=["order_date", "shipping_date"])
    else:
        raw, _ = load_raw()
        items = clean(raw)
        orders = to_orders(items)
    return items, orders


items, orders = load_data()

# ------------------------------------------------------------------ filters
st.sidebar.header("Filters")
min_d, max_d = orders["order_date"].min().date(), orders["order_date"].max().date()
date_range = st.sidebar.date_input("Order date range", (min_d, max_d), min_value=min_d, max_value=max_d)
markets = st.sidebar.multiselect("Market", sorted(orders["market"].unique()), default=sorted(orders["market"].unique()))
modes = st.sidebar.multiselect("Shipping mode", sorted(orders["shipping_mode"].unique()),
                               default=sorted(orders["shipping_mode"].unique()))
segments = st.sidebar.multiselect("Customer segment", sorted(orders["customer_segment"].unique()),
                                  default=sorted(orders["customer_segment"].unique()))

start, end = (date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else (min_d, max_d))
mask = (
    orders["order_date"].dt.date.between(start, end)
    & orders["market"].isin(markets)
    & orders["shipping_mode"].isin(modes)
    & orders["customer_segment"].isin(segments)
)
o = orders[mask]
i = items[items["order_id"].isin(o["order_id"])]

st.title("🚚 Supply Chain & Logistics Performance")
st.caption(f"{len(o):,} orders selected · {start:%d %b %Y} – {end:%d %b %Y}")

if o.empty:
    st.warning("No orders match the current filters.")
    st.stop()

# ------------------------------------------------------------------ KPI row
k = headline_kpis(o)
c = st.columns(6)
c[0].metric("Orders", f"{k['Total orders']:,}")
c[1].metric("Net sales", f"${k['Net sales'] / 1e6:,.2f}M")
c[2].metric("Profit margin", f"{k['Profit margin %']:.1f}%")
c[3].metric("On-time delivery", f"{k['On-time delivery %']:.1f}%")
c[4].metric("Avg lead time", f"{k['Avg actual lead time (days)']:.2f} days",
            f"{k['Avg actual lead time (days)'] - k['Avg scheduled lead time (days)']:+.2f} vs promise",
            delta_color="inverse")
c[5].metric("Cancelled / fraud", f"{k['Cancellation / fraud %']:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Delivery performance", "Trends", "Products (ABC)", "Data"])

# ------------------------------------------------------------------ delivery
with tab1:
    left, right = st.columns(2)
    mode = delivery_by(o, "shipping_mode").reset_index().sort_values("late_pct")
    fig = px.bar(mode, x="late_pct", y="shipping_mode", orientation="h", text="late_pct",
                 title="Late delivery rate by shipping mode (%)",
                 color_discrete_sequence=[BLUE], labels={"late_pct": "Late %", "shipping_mode": ""})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_range=[0, 110])
    left.plotly_chart(fig, use_container_width=True)

    days = mode.melt(id_vars="shipping_mode", value_vars=["avg_scheduled_days", "avg_actual_days"],
                     var_name="type", value_name="days")
    days["type"] = days["type"].map({"avg_scheduled_days": "Promised", "avg_actual_days": "Actual"})
    fig = px.bar(days, x="days", y="shipping_mode", color="type", barmode="group", orientation="h",
                 title="Promised vs actual shipping days", color_discrete_map={"Promised": MUTED, "Actual": BLUE},
                 labels={"shipping_mode": "", "days": "Days", "type": ""})
    right.plotly_chart(fig, use_container_width=True)

    dim = st.radio("Break down by", ["region", "market", "customer_segment", "payment_type"], horizontal=True)
    tbl = delivery_by(o, dim).reset_index().sort_values("late_pct")
    fig = px.bar(tbl, x="late_pct", y=dim, orientation="h", text="late_pct", color_discrete_sequence=[BLUE],
                 title=f"Late delivery rate by {dim.replace('_', ' ')} (%)", labels={"late_pct": "Late %", dim: ""},
                 height=max(350, 28 * len(tbl)), hover_data=["orders", "avg_delay_days"])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_vline(x=k["Late delivery %"], line_dash="dash", line_color="#52514e",
                  annotation_text=f"overall {k['Late delivery %']:.1f}%")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ trends
with tab2:
    t = monthly_trend(o).reset_index()
    fig = px.line(t, x="order_month", y="net_sales", markers=True, title="Monthly net sales ($)",
                  color_discrete_sequence=[BLUE], labels={"order_month": "", "net_sales": "Net sales"})
    st.plotly_chart(fig, use_container_width=True)
    fig = px.line(t, x="order_month", y="late_pct", markers=True, title="Monthly late delivery rate (%)",
                  color_discrete_sequence=[ORANGE], labels={"order_month": "", "late_pct": "Late %"})
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ products
with tab3:
    abc = abc_classification(i)
    fig = px.bar(abc, x="product", y="net_sales", color="abc_class",
                 color_discrete_map={"A": BLUE, "B": AQUA, "C": MUTED},
                 title="Revenue by product with ABC class", labels={"product": "", "net_sales": "Net sales ($)"},
                 hover_data=["units", "cumulative_%", "profit_margin_%"])
    fig.update_layout(xaxis_tickangle=-60, height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(abc, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ data
with tab4:
    st.dataframe(o.head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download filtered orders (CSV)", o.to_csv(index=False).encode(), "filtered_orders.csv",
                       "text/csv")
