"""
Generate a synthetic supply-chain dataset that follows the column schema of the
public "DataCo Smart Supply Chain for Big Data Analysis" dataset (Kaggle).

Why synthetic?  The real DataCo file (~95 MB) is too large to commit comfortably
and must be downloaded from Kaggle. This generator lets the whole project run
out-of-the-box. If you place the real file at
    data/raw/DataCoSupplyChainDataset.csv
the cleaning step will use it automatically instead of the sample.

Run:  python src/data_generator.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RAW_SAMPLE_PATH, SEED

N_ORDERS = 12_000

MARKETS = {
    "LATAM": ["Central America", "South America", "Caribbean"],
    "Europe": ["Western Europe", "Northern Europe", "Southern Europe", "Eastern Europe"],
    "Pacific Asia": ["Southeast Asia", "South Asia", "Oceania", "Eastern Asia", "West Asia"],
    "USCA": ["US Center", "West of USA", "East of USA", "South of USA", "Canada"],
    "Africa": ["West Africa", "North Africa", "East Africa", "Southern Africa"],
}
MARKET_WEIGHTS = {"LATAM": 0.29, "Europe": 0.28, "Pacific Asia": 0.23, "USCA": 0.14, "Africa": 0.06}

COUNTRIES = {
    "Central America": ["Mexico", "Guatemala", "Honduras"],
    "South America": ["Brazil", "Colombia", "Argentina"],
    "Caribbean": ["Dominican Republic", "Cuba", "Haiti"],
    "Western Europe": ["France", "Germany", "Netherlands"],
    "Northern Europe": ["United Kingdom", "Sweden", "Ireland"],
    "Southern Europe": ["Italy", "Spain", "Portugal"],
    "Eastern Europe": ["Poland", "Russia", "Ukraine"],
    "Southeast Asia": ["Indonesia", "Philippines", "Thailand"],
    "South Asia": ["India", "Pakistan", "Bangladesh"],
    "Oceania": ["Australia", "New Zealand"],
    "Eastern Asia": ["China", "Japan", "South Korea"],
    "West Asia": ["Turkey", "Iran", "Saudi Arabia"],
    "US Center": ["United States"],
    "West of USA": ["United States"],
    "East of USA": ["United States"],
    "South of USA": ["United States"],
    "Canada": ["Canada"],
    "West Africa": ["Nigeria", "Ghana"],
    "North Africa": ["Egypt", "Morocco"],
    "East Africa": ["Kenya", "Ethiopia"],
    "Southern Africa": ["South Africa"],
}

# (product, category, department, price)
PRODUCTS = [
    ("Field & Stream Sportsman 16 Gun Fire Safe", "Fishing", "Fan Shop", 399.98),
    ("Perfect Fitness Perfect Rip Deck", "Cleats", "Apparel", 59.99),
    ("Nike Men's Dri-FIT Victory Golf Polo", "Women's Apparel", "Golf", 50.00),
    ("Nike Men's Free 5.0+ Running Shoe", "Cardio Equipment", "Footwear", 99.99),
    ("O'Brien Men's Neoprene Life Vest", "Indoor/Outdoor Games", "Fan Shop", 49.98),
    ("Pelican Sunstream 100 Kayak", "Water Sports", "Fan Shop", 199.99),
    ("Diamondback Women's Serene Classic Comfort Bi", "Camping & Hiking", "Fan Shop", 299.98),
    ("Nike Men's CJ Elite 2 TD Football Cleat", "Men's Footwear", "Apparel", 129.99),
    ("Under Armour Girls' Toddler Spine Surge Runni", "Girls' Apparel", "Apparel", 39.99),
    ("Garmin Forerunner 910XT GPS Watch", "Electronics", "Outdoors", 349.99),
    ("Bowflex SelectTech 1090 Dumbbells", "Fitness Accessories", "Fitness", 599.99),
    ("Team Golf Pittsburgh Steelers Putter Grip", "Golf Gloves", "Golf", 24.99),
    ("Under Armour Hustle Storm Medium Duffle Bag", "Accessories", "Outdoors", 34.99),
    ("Columbia Men's PFG Anchor Tough T-Shirt", "Men's Clothing", "Apparel", 30.00),
    ("adidas Brazuca 2014 Official Match Ball", "Soccer", "Fan Shop", 159.99),
    ("Coleman 8-Person Instant Tent", "Camping & Hiking", "Outdoors", 249.99),
    ("Yakima DoubleDown Ace Hitch Mount 4-Bike Rack", "Accessories", "Outdoors", 499.99),
    ("LIJA Women's Button Golf Dress", "Golf Apparel", "Golf", 80.00),
    ("Smart watch", "Electronics", "Technology", 327.75),
    ("Dell Laptop", "Computers", "Technology", 1500.00),
    ("SOLE E35 Elliptical", "Cardio Equipment", "Fitness", 999.99),
    ("Hirzl Women's Soffft Flex Golf Glove", "Golf Gloves", "Golf", 22.00),
    ("Glove It Women's Imperial Golf Glove", "Golf Gloves", "Golf", 19.99),
    ("Titleist Pro V1 High Numbers Golf Balls", "Golf Balls", "Golf", 47.99),
    ("Clicgear 8.0 Shoe Brush", "Golf Shoes", "Golf", 9.99),
]
# Popularity: cheaper everyday items sell more often
PRODUCT_POP = np.array([6, 14, 12, 12, 13, 8, 3, 9, 5, 3, 2, 4, 5, 8, 4, 3, 2, 3, 2, 1, 1, 3, 3, 3, 2], float)

SHIP_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_MODE_P = [0.60, 0.195, 0.15, 0.055]
SCHEDULED_DAYS = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}

SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SEGMENT_P = [0.52, 0.30, 0.18]
PAY_TYPES = ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
PAY_P = [0.38, 0.28, 0.23, 0.11]

# Regions with systematically weaker last-mile performance (adds realistic signal)
SLOW_REGIONS = {"West Africa": 1, "Central America": 0.5, "South Asia": 0.5, "Caribbean": 0.5}


def _actual_days(mode: str, region: str, month: int, rng: np.random.Generator) -> int:
    """Actual transit days: carrier behaviour per mode + regional and peak-season delays."""
    if mode == "Standard Class":
        base = rng.choice([2, 3, 4, 5, 6], p=[0.20, 0.20, 0.22, 0.20, 0.18])
    elif mode == "Second Class":
        base = rng.choice([2, 3, 4, 5, 6], p=[0.20, 0.20, 0.22, 0.20, 0.18])
    elif mode == "First Class":
        base = rng.choice([1, 2], p=[0.04, 0.96])
    else:  # Same Day
        base = rng.choice([0, 1], p=[0.52, 0.48])
    extra = 0
    if rng.random() < 0.25 * SLOW_REGIONS.get(region, 0):
        extra += 1
    if month in (11, 12) and mode in ("Standard Class", "Second Class") and rng.random() < 0.20:
        extra += 1  # holiday peak congestion
    return int(base + extra)


def generate(n_orders: int = N_ORDERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31 23:59")
    span = (end - start).total_seconds()

    markets = list(MARKETS)
    rows = []
    item_id = 1
    pop = PRODUCT_POP / PRODUCT_POP.sum()

    for order_id in range(1, n_orders + 1):
        # seasonality: more orders in Q4
        while True:
            ts = start + pd.Timedelta(seconds=float(rng.random() * span))
            if rng.random() < (1.0 if ts.month in (10, 11, 12) else 0.8):
                break
        ts = ts.floor("min")
        market = rng.choice(markets, p=[MARKET_WEIGHTS[m] for m in markets])
        region = rng.choice(MARKETS[market])
        country = rng.choice(COUNTRIES[region])
        mode = rng.choice(SHIP_MODES, p=SHIP_MODE_P)
        segment = rng.choice(SEGMENTS, p=SEGMENT_P)
        pay = rng.choice(PAY_TYPES, p=PAY_P)

        # order status
        if pay == "TRANSFER" and rng.random() < 0.09:
            status = "SUSPECTED_FRAUD"
        elif rng.random() < 0.04:
            status = "CANCELED"
        else:
            status = rng.choice(
                ["COMPLETE", "PENDING", "CLOSED", "PENDING_PAYMENT", "PROCESSING", "ON_HOLD", "PAYMENT_REVIEW"],
                p=[0.36, 0.12, 0.12, 0.23, 0.12, 0.04, 0.01],
            )

        sched = SCHEDULED_DAYS[mode]
        real = _actual_days(mode, region, ts.month, rng)
        if status in ("CANCELED", "SUSPECTED_FRAUD"):
            delivery = "Shipping canceled"
        elif real > sched:
            delivery = "Late delivery"
        elif real < sched:
            delivery = "Advance shipping"
        else:
            delivery = "Shipping on time"
        ship_ts = ts + pd.Timedelta(days=real)

        n_items = rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05])
        prod_idx = rng.choice(len(PRODUCTS), size=n_items, replace=False, p=pop)
        for pi in prod_idx:
            name, cat, dept, price = PRODUCTS[pi]
            qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.15, 0.12, 0.10, 0.08])) if price < 100 else 1
            disc_rate = float(rng.choice([0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.09, 0.10, 0.12, 0.15, 0.17, 0.18, 0.20, 0.25]))
            gross = price * qty
            disc = round(gross * disc_rate, 2)
            total = round(gross - disc, 2)
            margin = rng.normal(0.20 - 0.8 * disc_rate, 0.25)  # deeper discounts erode margin
            margin = float(np.clip(margin, -1.5, 0.5))
            profit = round(total * margin, 2)
            rows.append({
                "Type": pay,
                "Days for shipping (real)": real,
                "Days for shipment (scheduled)": sched,
                "Benefit per order": profit,
                "Sales per customer": total,
                "Delivery Status": delivery,
                "Late_delivery_risk": int(delivery == "Late delivery"),
                "Category Name": cat,
                "Customer Segment": segment,
                "Department Name": dept,
                "Market": market,
                "Order City": country,
                "Order Country": country,
                "order date (DateOrders)": f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}",
                "Order Id": order_id,
                "Order Item Id": item_id,
                "Order Item Discount": disc,
                "Order Item Discount Rate": disc_rate,
                "Order Item Product Price": price,
                "Order Item Quantity": qty,
                "Sales": round(gross, 2),
                "Order Item Total": total,
                "Order Profit Per Order": profit,
                "Order Region": region,
                "Order Status": status,
                "Product Name": name,
                "Product Price": price,
                "shipping date (DateOrders)": f"{ship_ts.month}/{ship_ts.day}/{ship_ts.year} {ship_ts.hour}:{ship_ts.minute:02d}",
                "Shipping Mode": mode,
            })
            item_id += 1

    df = pd.DataFrame(rows)
    # Inject a little real-world mess so the cleaning step has work to do
    dup = df.sample(frac=0.005, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    miss = df.sample(frac=0.01, random_state=seed + 1).index
    df.loc[miss, "Customer Segment"] = np.nan
    return df


if __name__ == "__main__":
    RAW_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(RAW_SAMPLE_PATH, index=False)
    print(f"Saved {len(data):,} rows -> {RAW_SAMPLE_PATH}")
