"""
Data loading and cleaning.

Works with both the bundled sample and the real DataCo Kaggle file
(the real file is latin-1 encoded and has ~50 extra columns we don't need).

Run:  python src/cleaning.py
"""
from __future__ import annotations

import pandas as pd

from config import (CLEAN_ITEMS_PATH, CLEAN_ORDERS_PATH, RAW_SAMPLE_PATH,
                    REAL_DATACO_PATH)

# DataCo column -> clean snake_case name
COLUMN_MAP = {
    "Order Id": "order_id",
    "Order Item Id": "order_item_id",
    "order date (DateOrders)": "order_date",
    "shipping date (DateOrders)": "shipping_date",
    "Type": "payment_type",
    "Days for shipping (real)": "actual_days",
    "Days for shipment (scheduled)": "scheduled_days",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "is_late",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Order Region": "region",
    "Order Country": "country",
    "Customer Segment": "customer_segment",
    "Department Name": "department",
    "Category Name": "category",
    "Product Name": "product",
    "Product Price": "product_price",
    "Order Item Quantity": "quantity",
    "Order Item Discount": "discount",
    "Order Item Discount Rate": "discount_rate",
    "Sales": "gross_sales",
    "Order Item Total": "net_sales",
    "Order Profit Per Order": "profit",
    "Order Status": "order_status",
}

CANCELLED_STATUSES = {"CANCELED", "SUSPECTED_FRAUD"}


def load_raw() -> tuple[pd.DataFrame, str]:
    """Load the real DataCo file if present, otherwise the bundled sample."""
    if REAL_DATACO_PATH.exists():
        return pd.read_csv(REAL_DATACO_PATH, encoding="latin-1"), "DataCo (Kaggle)"
    if not RAW_SAMPLE_PATH.exists():
        from data_generator import generate
        RAW_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(RAW_SAMPLE_PATH, index=False)
    return pd.read_csv(RAW_SAMPLE_PATH), "synthetic sample (DataCo schema)"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, item-level table with derived logistics features."""
    df = raw[[c for c in COLUMN_MAP if c in raw.columns]].rename(columns=COLUMN_MAP).copy()

    # 1. Duplicates
    df = df.drop_duplicates()
    if "order_item_id" in df.columns:
        df = df.drop_duplicates(subset="order_item_id")

    # 2. Types
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y %H:%M")
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], format="%m/%d/%Y %H:%M")
    for c in ["actual_days", "scheduled_days", "quantity", "is_late"]:
        df[c] = df[c].astype(int)

    # 3. Missing values
    df["customer_segment"] = df["customer_segment"].fillna("Unknown")

    # 4. Standardise text
    for c in ["shipping_mode", "market", "region", "country", "category", "product", "delivery_status"]:
        df[c] = df[c].astype(str).str.strip()

    # 5. Derived features
    df["delay_days"] = df["actual_days"] - df["scheduled_days"]
    df["is_cancelled"] = df["order_status"].isin(CANCELLED_STATUSES).astype(int)
    df["is_on_time"] = ((df["delay_days"] <= 0) & (df["is_cancelled"] == 0)).astype(int)
    df["is_loss"] = (df["profit"] < 0).astype(int)
    df["profit_margin"] = (df["profit"] / df["net_sales"]).where(df["net_sales"] > 0)
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    df["order_weekday"] = df["order_date"].dt.day_name()
    df["order_hour"] = df["order_date"].dt.hour
    return df.reset_index(drop=True)


def to_orders(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item rows to one row per order (the unit for delivery KPIs)."""
    first = ["order_date", "shipping_date", "payment_type", "actual_days", "scheduled_days",
             "delay_days", "delivery_status", "is_late", "is_on_time", "is_cancelled",
             "shipping_mode", "market", "region", "country", "customer_segment", "order_status",
             "order_year", "order_month", "order_quarter", "order_weekday", "order_hour"]
    agg = items.groupby("order_id").agg(
        **{c: (c, "first") for c in first},
        n_items=("order_item_id", "count"),
        units=("quantity", "sum"),
        gross_sales=("gross_sales", "sum"),
        discount=("discount", "sum"),
        net_sales=("net_sales", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    agg["profit_margin"] = (agg["profit"] / agg["net_sales"]).where(agg["net_sales"] > 0)
    return agg


if __name__ == "__main__":
    raw, source = load_raw()
    items = clean(raw)
    orders = to_orders(items)
    CLEAN_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    items.to_csv(CLEAN_ITEMS_PATH, index=False)
    orders.to_csv(CLEAN_ORDERS_PATH, index=False)
    print(f"Source: {source}")
    print(f"Raw rows: {len(raw):,} -> clean item rows: {len(items):,}, orders: {len(orders):,}")
