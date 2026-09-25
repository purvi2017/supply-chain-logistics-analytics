"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "images"
MODELS = ROOT / "models"

# If you download the real Kaggle file, put it here and it will be used automatically.
REAL_DATACO_PATH = DATA_RAW / "DataCoSupplyChainDataset.csv"
RAW_SAMPLE_PATH = DATA_RAW / "supply_chain_sample.csv"

CLEAN_ITEMS_PATH = DATA_PROCESSED / "order_items_clean.csv"
CLEAN_ORDERS_PATH = DATA_PROCESSED / "orders_clean.csv"

SEED = 42
