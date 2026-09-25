"""Cell contents for the four project notebooks (consumed by build_notebooks.py)."""

SETUP = """
import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import CLEAN_ITEMS_PATH, CLEAN_ORDERS_PATH, IMAGES
from viz import set_style, label_bars, BLUE, ORANGE, AQUA, MUTED, TEXT_2

set_style()
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 40)
"""

# ---------------------------------------------------------------- 01 cleaning
NB1 = [
    ("md", """
# 01 · Data Cleaning & Preparation

**Project:** Supply Chain & Logistics Performance Analytics

**Goal of this notebook:** load the raw order-item data, check its quality, clean it, and
create the derived logistics features (delay days, on-time flag, time columns) used by the
rest of the analysis.

> **Data source.** The project follows the schema of the public
> [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
> The repo ships with a synthetic sample in the same format so everything runs out of the box.
> Put the real file at `data/raw/DataCoSupplyChainDataset.csv` and re-run - `load_raw()` picks it up automatically.
"""),
    ("code", SETUP + """
from cleaning import load_raw, clean, to_orders, COLUMN_MAP
"""),
    ("md", "## 1. Load raw data"),
    ("code", """
raw, source = load_raw()
print(f"Source : {source}")
print(f"Shape  : {raw.shape[0]:,} rows x {raw.shape[1]} columns")
raw.head()
"""),
    ("code", """
raw[list(COLUMN_MAP)].dtypes.to_frame("dtype").T
"""),
    ("md", "## 2. Data quality checks"),
    ("code", """
quality = pd.DataFrame({
    "missing_values": raw.isna().sum(),
    "missing_%": 100 * raw.isna().mean(),
    "unique_values": raw.nunique(),
})
quality[quality["missing_values"] > 0]
"""),
    ("code", """
print("Fully duplicated rows      :", raw.duplicated().sum())
print("Duplicated order item IDs  :", raw["Order Item Id"].duplicated().sum())
print("Negative shipping days     :", (raw["Days for shipping (real)"] < 0).sum())
print("Negative sales             :", (raw["Sales"] < 0).sum())
print("Order lines with a loss (profit<0):", (raw["Order Profit Per Order"] < 0).sum())
"""),
    ("md", """
**Findings**
- A small number of fully duplicated rows (same order item repeated) - these would double-count sales, so they are dropped.
- `Customer Segment` has ~1% missing values - filled with `"Unknown"` so no orders are lost.
- No impossible values (negative days or sales). Negative profit is valid (loss-making orders) and is kept.
- Dates are stored as text (`m/d/Y H:M`) and need to be parsed.
"""),
    ("md", "## 3. Clean and engineer features"),
    ("code", """
items = clean(raw)
orders = to_orders(items)

print(f"Item rows after cleaning : {len(items):,}  (removed {len(raw) - len(items):,})")
print(f"Unique orders            : {len(orders):,}")
print(f"Date range               : {items['order_date'].min():%d %b %Y} -> {items['order_date'].max():%d %b %Y}")
items.head()
"""),
    ("md", """
**Derived columns**

| Column | Meaning |
|---|---|
| `delay_days` | actual shipping days − scheduled days (positive = late) |
| `is_late` | 1 if delivered after the promised date |
| `is_on_time` | 1 if delivered on or before the promised date (and not cancelled) |
| `is_cancelled` | 1 if order was cancelled or flagged as suspected fraud |
| `profit_margin` | profit ÷ net sales |
| `order_month / quarter / weekday / hour` | time features for trend analysis |
"""),
    ("code", """
items[["actual_days", "scheduled_days", "delay_days", "quantity", "net_sales", "profit", "discount_rate"]].describe().T
"""),
    ("md", "## 4. Save processed data"),
    ("code", """
CLEAN_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
items.to_csv(CLEAN_ITEMS_PATH, index=False)
orders.to_csv(CLEAN_ORDERS_PATH, index=False)
print("Saved:", CLEAN_ITEMS_PATH.name, "and", CLEAN_ORDERS_PATH.name)
"""),
]

# ---------------------------------------------------------------- 02 EDA
NB2 = [
    ("md", """
# 02 · Exploratory Data Analysis

Understanding the business before measuring performance: how much is sold, where, through which
shipping modes, and how delivery times are distributed.
"""),
    ("code", SETUP + """
items = pd.read_csv(CLEAN_ITEMS_PATH, parse_dates=["order_date", "shipping_date"])
orders = pd.read_csv(CLEAN_ORDERS_PATH, parse_dates=["order_date", "shipping_date"])
print(f"{len(orders):,} orders | {len(items):,} order lines")
"""),
    ("md", "## 1. Sales trend over time"),
    ("code", """
monthly = orders.groupby("order_month")[["net_sales", "profit"]].sum()
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(monthly.index, monthly["net_sales"] / 1000, color=BLUE, label="Net sales")
ax.plot(monthly.index, monthly["profit"] / 1000, color=ORANGE, label="Profit")
ax.set_title("Monthly net sales and profit ($ thousands)")
ax.set_xticks(monthly.index[::3])
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}K"))
ax.legend(loc="upper left")
for name, col, c in [("Net sales", "net_sales", BLUE), ("Profit", "profit", ORANGE)]:
    ax.annotate(name, (len(monthly) - 1, monthly[col].iloc[-1] / 1000), xytext=(6, 0),
                textcoords="offset points", color=TEXT_2, va="center", fontsize=9)
plt.savefig(IMAGES / "01_monthly_sales_profit.png")
"""),
    ("code", """
q = orders.groupby(orders["order_date"].dt.quarter)["net_sales"].sum()
(100 * q / q.sum()).round(1).rename("share_of_sales_%").to_frame().rename_axis("quarter").T
"""),
    ("md", "Q4 carries the largest share of annual sales - the peak season that logistics capacity has to be planned for."),
    ("md", "## 2. Where do orders come from?"),
    ("code", """
by_market = orders.groupby("market").agg(orders=("order_id", "count"), net_sales=("net_sales", "sum")).sort_values("net_sales")
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.barh(by_market.index, by_market["net_sales"] / 1e6, color=BLUE, height=0.6)
ax.set_title("Net sales by market ($ millions)")
ax.grid(axis="y", visible=False)
label_bars(ax, fmt="${:.2f}M", pad=0.02)
plt.savefig(IMAGES / "02_sales_by_market.png")
"""),
    ("md", "## 3. Shipping mode mix"),
    ("code", """
mode_mix = orders["shipping_mode"].value_counts(normalize=True).mul(100).sort_values()
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.barh(mode_mix.index, mode_mix.values, color=BLUE, height=0.6)
ax.set_title("Share of orders by shipping mode (%)")
ax.grid(axis="y", visible=False)
label_bars(ax)
plt.savefig(IMAGES / "03_shipping_mode_mix.png")
"""),
    ("md", "## 4. Category performance"),
    ("code", """
cat = items.groupby("category").agg(net_sales=("net_sales", "sum"), profit=("profit", "sum"))
cat["margin_%"] = 100 * cat["profit"] / cat["net_sales"]
cat = cat.sort_values("net_sales", ascending=False)
cat.head(10)
"""),
    ("md", "## 5. Delivery time distribution"),
    ("code", """
s = orders[orders["is_cancelled"] == 0]
dist = s["delay_days"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 4))
colors = [ORANGE if d > 0 else BLUE for d in dist.index]
ax.bar(dist.index, dist.values, color=colors, width=0.7)
ax.set_title("Delay vs promised date (days)  -  blue = on time/early, orange = late")
ax.set_xlabel("Actual days - scheduled days")
ax.set_ylabel("Orders")
ax.grid(axis="x", visible=False)
plt.savefig(IMAGES / "04_delay_distribution.png")
"""),
    ("md", "## 6. Do discounts hurt profit?"),
    ("code", """
disc = items.assign(discount_band=pd.cut(items["discount_rate"], [-0.01, 0.0, 0.05, 0.10, 0.15, 0.25],
                                         labels=["0%", "1-5%", "6-10%", "11-15%", "16-25%"]))
disc_tbl = disc.groupby("discount_band", observed=True).agg(lines=("order_item_id", "count"),
                                            net_sales=("net_sales", "sum"), profit=("profit", "sum"),
                                            loss_lines_pct=("is_loss", "mean"))
disc_tbl["margin_%"] = 100 * disc_tbl["profit"] / disc_tbl["net_sales"]
disc_tbl["loss_lines_pct"] *= 100

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(disc_tbl.index.astype(str), disc_tbl["margin_%"], color=BLUE, width=0.6)
ax.set_title("Profit margin by discount band (%)")
ax.grid(axis="x", visible=False)
label_bars(ax, horizontal=False, pad=0.3)
plt.savefig(IMAGES / "05_margin_by_discount.png")
disc_tbl
"""),
    ("md", """
### EDA summary
- Sales are seasonal, with a Q4 peak.
- LATAM and Europe are the two biggest markets.
- **Standard Class carries the majority of orders**, so its performance drives the overall numbers.
- A large share of orders arrive after the promised date - explored in detail in notebook 03.
- Profit margin falls steadily as discounts rise - deep discounts create loss-making lines.
"""),
]

# ---------------------------------------------------------------- 03 KPIs
NB3 = [
    ("md", """
# 03 · Logistics KPI Analysis

The core of the project: measure delivery, cost and fulfilment performance and find **where** the
supply chain under-performs.

Delivery KPIs are computed on non-cancelled orders only.
"""),
    ("code", SETUP + """
from kpis import headline_kpis, delivery_by, monthly_trend, abc_classification

items = pd.read_csv(CLEAN_ITEMS_PATH, parse_dates=["order_date", "shipping_date"])
orders = pd.read_csv(CLEAN_ORDERS_PATH, parse_dates=["order_date", "shipping_date"])
"""),
    ("md", "## 1. Headline KPIs"),
    ("code", """
k = headline_kpis(orders)
pd.Series(k, name="value").to_frame()
"""),
    ("md", "## 2. Delivery performance by shipping mode"),
    ("code", """
mode = delivery_by(orders, "shipping_mode")
mode
"""),
    ("code", """
m = mode.sort_values("late_pct")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].barh(m.index, m["late_pct"], color=[ORANGE if v > 60 else BLUE for v in m["late_pct"]], height=0.6)
axes[0].set_title("Late delivery rate by shipping mode (%)")
axes[0].set_xlim(0, 110)
axes[0].grid(axis="y", visible=False)
label_bars(axes[0])

y = np.arange(len(m))
axes[1].barh(y + 0.18, m["avg_scheduled_days"], height=0.34, color=MUTED, label="Promised (scheduled)")
axes[1].barh(y - 0.18, m["avg_actual_days"], height=0.34, color=BLUE, label="Actual")
axes[1].set_yticks(y, m.index)
axes[1].set_title("Promised vs actual shipping days")
axes[1].grid(axis="y", visible=False)
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.tight_layout()
plt.savefig(IMAGES / "06_late_rate_by_mode.png")
"""),
    ("md", """
**Insight:** First Class and Second Class are late in the vast majority of cases. The promise
(1 and 2 days) is shorter than what carriers actually achieve, so the problem is an
**unrealistic SLA**, not just slow shipping. Standard Class (4-day promise) performs best.
"""),
    ("md", "## 3. Regional performance"),
    ("code", """
region = delivery_by(orders, "region")
top = region[region["orders"] >= 100].head(10).sort_values("late_pct")
overall = k["Late delivery %"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top.index, top["late_pct"], color=BLUE, height=0.6)
ax.axvline(overall, color=TEXT_2, linestyle="--", linewidth=1)
ax.text(overall + 0.5, len(top) - 0.35, f"overall average {overall:.1f}%", color=TEXT_2, fontsize=9)
ax.set_xlim(0, 75)
ax.set_title("10 regions with the highest late-delivery rate (%)")
ax.grid(axis="y", visible=False)
label_bars(ax)
plt.savefig(IMAGES / "07_late_rate_by_region.png")
"""),
    ("code", """
delivery_by(orders, "market")
"""),
    ("md", "## 4. Monthly trend of late deliveries"),
    ("code", """
trend = monthly_trend(orders)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(trend.index, trend["late_pct"], color=BLUE, marker="o", markersize=4)
ax.axhline(trend["late_pct"].mean(), color=TEXT_2, linestyle="--", linewidth=1)
ax.set_title("Late delivery rate by month (%)")
ax.set_xticks(trend.index[::3])
ax.tick_params(axis="x", rotation=45)
for i, (idx, row) in enumerate(trend.iterrows()):
    if idx.endswith(("-11", "-12")):
        ax.annotate(f"{row['late_pct']:.0f}%", (i, row["late_pct"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8, color=TEXT_2)
plt.savefig(IMAGES / "08_monthly_late_trend.png")
"""),
    ("code", """
peak = orders[orders["is_cancelled"] == 0].assign(peak=lambda d: d["order_date"].dt.month.isin([11, 12]))
peak.groupby(["shipping_mode", "peak"])["is_late"].mean().mul(100).unstack().rename(columns={False: "Jan-Oct", True: "Nov-Dec"}).round(1)
"""),
    ("md", "## 5. Order fulfilment, cancellations & fraud"),
    ("code", """
status = orders["order_status"].value_counts(normalize=True).mul(100).round(2).rename("share_%")
status.to_frame()
"""),
    ("code", """
fraud = orders.groupby("payment_type").agg(orders=("order_id", "count"),
                                           suspected_fraud_pct=("order_status", lambda s: 100 * (s == "SUSPECTED_FRAUD").mean()),
                                           cancelled_pct=("order_status", lambda s: 100 * (s == "CANCELED").mean()))
fraud.sort_values("suspected_fraud_pct", ascending=False).round(2)
"""),
    ("md", "**Insight:** all suspected-fraud orders come through **TRANSFER** payments - a clear place to add payment verification."),
    ("md", "## 6. Inventory prioritisation - ABC analysis"),
    ("code", """
abc = abc_classification(items)
abc[["product", "category", "units", "net_sales", "revenue_share_%", "cumulative_%", "abc_class", "profit_margin_%"]]
"""),
    ("code", """
fig, ax = plt.subplots(figsize=(12, 4.8))
colors = abc["abc_class"].map({"A": BLUE, "B": AQUA, "C": MUTED}).tolist()
ax.bar(range(len(abc)), abc["cumulative_%"], color=colors, width=0.75)
for level in (80, 95):
    ax.axhline(level, color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(-0.4, level + 1.5, f"{level}% of revenue", color=TEXT_2, fontsize=9, ha="left")
ax.set_xticks(range(len(abc)), [p[:18] for p in abc["product"]], rotation=75, fontsize=8)
ax.set_ylim(0, 105)
ax.set_ylabel("Cumulative share of revenue (%)")
ax.set_title("Pareto of revenue by product  -  class A (blue), B (aqua), C (grey)")
ax.grid(axis="x", visible=False)
plt.savefig(IMAGES / "09_abc_pareto.png")
abc.groupby("abc_class", observed=True).agg(products=("product", "count"), revenue_share=("revenue_share_%", "sum"))
"""),
    ("md", """
**Insight:** a handful of **A-class products generate ~80% of revenue**. These deserve the tightest
stock control, safety stock and the most reliable shipping; C-class items can be managed with
lower stock levels.
"""),
    ("md", "## 7. Loss-making orders"),
    ("code", """
loss = orders[orders["profit"] < 0]
print(f"Loss-making orders : {len(loss):,} ({100 * len(loss) / len(orders):.1f}% of orders)")
print(f"Total loss         : ${-loss['profit'].sum():,.0f}")
print(f"Avg discount on loss orders : {100 * loss['discount'].sum() / loss['gross_sales'].sum():.1f}%")
print(f"Avg discount on other orders: {100 * orders.loc[orders['profit'] >= 0, 'discount'].sum() / orders.loc[orders['profit'] >= 0, 'gross_sales'].sum():.1f}%")
"""),
    ("code", """
kpi_tables = {"shipping_mode": mode, "region": region, "market": delivery_by(orders, "market")}
for name, t in kpi_tables.items():
    t.to_csv(f"../reports/kpi_by_{name}.csv")
abc.to_csv("../reports/abc_classification.csv", index=False)
print("KPI tables exported to reports/")
"""),
]

# ---------------------------------------------------------------- 04 ML
NB4 = [
    ("md", """
# 04 · Predicting Late Deliveries (ML)

**Question:** at the moment an order is placed, can we predict whether it will be delivered late?
If yes, operations can upgrade the carrier or warn the customer in advance.

**Avoiding data leakage:** columns that are only known *after* shipping (`actual_days`,
`delay_days`, `delivery_status`, `shipping_date`) are **not** used as features.
"""),
    ("code", SETUP + """
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from config import MODELS, SEED

orders = pd.read_csv(CLEAN_ORDERS_PATH, parse_dates=["order_date"])
df = orders[orders["is_cancelled"] == 0].copy()
df["order_month_num"] = df["order_date"].dt.month
print(f"Modelling rows: {len(df):,} | late share: {100 * df['is_late'].mean():.1f}%")
"""),
    ("md", "## 1. Features and train/test split"),
    ("code", """
categorical = ["shipping_mode", "market", "region", "customer_segment", "payment_type", "order_weekday"]
numeric = ["scheduled_days", "order_month_num", "order_hour", "n_items", "units", "net_sales", "discount"]
X, y = df[categorical + numeric], df["is_late"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
"""),
    ("md", "## 2. Train and compare models"),
    ("code", """
models = {
    "Baseline (always 'late')": None,
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                            random_state=SEED, n_jobs=-1),
}
results, fitted = [], {}
for name, clf in models.items():
    if clf is None:
        pred = np.ones(len(y_test), dtype=int); proba = np.full(len(y_test), 0.5)
    else:
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
        fitted[name] = pipe
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
    results.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred),
                    "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba)})
results = pd.DataFrame(results).set_index("model").round(3)
results
"""),
    ("code", """
best_name = results.drop(index="Baseline (always 'late')")["roc_auc"].idxmax()
best = fitted[best_name]
pred = best.predict(X_test)
print("Best model:", best_name)
print(classification_report(y_test, pred, target_names=["On time", "Late"]))
"""),
    ("code", """
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.grid(False)
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
ax.set_xticks([0, 1], ["On time", "Late"]); ax.set_yticks([0, 1], ["On time", "Late"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix - {best_name}")
plt.savefig(IMAGES / "10_confusion_matrix.png")
"""),
    ("md", "## 3. What drives late deliveries?"),
    ("code", """
rf = fitted["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
imp.index = imp.index.str.replace("cat__", "").str.replace("num__", "")
top = imp.sort_values().tail(10)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(top.index, top.values, color=BLUE, height=0.6)
ax.set_title("Top 10 drivers of late delivery (Random Forest importance)")
ax.grid(axis="y", visible=False)
plt.savefig(IMAGES / "11_feature_importance.png")
"""),
    ("md", """
**Interpretation:** shipping mode (and its promised `scheduled_days`) dominates the prediction -
the same conclusion as the KPI analysis. Order value, time and region add only small signal. So the
most effective fix is operational: **re-set delivery promises for First/Second Class** or change the
carrier for those services.
"""),
    ("code", """
MODELS.mkdir(exist_ok=True)
joblib.dump(best, MODELS / "late_delivery_model.joblib", compress=3)
print("Saved model ->", "models/late_delivery_model.joblib")
"""),
]

NOTEBOOKS = {
    "01_data_cleaning.ipynb": NB1,
    "02_eda.ipynb": NB2,
    "03_kpi_analysis.ipynb": NB3,
    "04_delay_prediction.ipynb": NB4,
}
