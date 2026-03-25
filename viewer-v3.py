from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================
CATEGORY_MAP = {
    "Traditional": 0,
    "Low end": 1,
    "High end": 2,
    "Performance": 3,
    "Size": 4,
}
CATEGORY_NAMES = list(CATEGORY_MAP.keys())

NUM_COLS = [
    "price_pct_of_target_mid",
    "age_pct_of_target",
    "performance_pct_of_target",
    "size_pct_of_target",
    "reliability_pct_of_target_mid",
    "customer_accessibility",
    "customer_awareness",
]

RAW_PRODUCT_COLS = [
    "Name",
    "Price",
    "Age",
    "Performance",
    "Size",
    "Reliability",
    "Acessibility",
    "Awarness",
]

TARGET_COLS = [
    "ExpAge",
    "ExpPriceLow",
    "ExpPriceHigh",
    "ExpPerformance",
    "ExpSize",
    "ExpReliabilityLow",
    "ExpReliabilityHigh",
]


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_latest_keras_model() -> Tuple[tf.keras.Model, str]:
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        model_dir.glob("v*.keras"),
        key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError("No model files found in models/ like v1.keras, v2.keras, ...")
    best_path = candidates[-1]
    model = tf.keras.models.load_model(best_path, compile=False)
    return model, str(best_path)


# ============================================================
# HELPERS
# ============================================================
def safe_mid(low: float, high: float) -> float:
    return (float(low) + float(high)) / 2.0


def to_model_df(raw_df: pd.DataFrame, category_name: str, targets: Dict[str, float]) -> pd.DataFrame:
    exp_mid_price = safe_mid(targets["ExpPriceLow"], targets["ExpPriceHigh"])
    exp_mid_reliability = safe_mid(targets["ExpReliabilityLow"], targets["ExpReliabilityHigh"])

    out = pd.DataFrame()
    out["Company"] = raw_df["Name"].astype(str).str.strip()
    out["Categories"] = CATEGORY_MAP[category_name]
    out["price_pct_of_target_mid"] = (raw_df["Price"].astype(float) / exp_mid_price) - 1.0
    out["age_pct_of_target"] = (raw_df["Age"].astype(float) / float(targets["ExpAge"])) - 1.0
    out["performance_pct_of_target"] = (raw_df["Performance"].astype(float) / float(targets["ExpPerformance"])) - 1.0
    out["size_pct_of_target"] = (raw_df["Size"].astype(float) / float(targets["ExpSize"])) - 1.0
    out["reliability_pct_of_target_mid"] = (raw_df["Reliability"].astype(float) / exp_mid_reliability) - 1.0
    out["customer_accessibility"] = raw_df["Acessibility"].astype(float)
    out["customer_awareness"] = raw_df["Awarness"].astype(float)
    return out


def predict_scores(model: tf.keras.Model, model_df: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    x_num = model_df[NUM_COLS].astype(np.float32).values
    x_cat = model_df[["Categories"]].astype(np.int32).values
    preds = model.predict(
        {
            "numeric_features": x_num,
            "category": x_cat,
        },
        batch_size=batch_size,
        verbose=0,
    )
    return np.asarray(preds, dtype=np.float32).reshape(-1)


def nonnegative_share_scale(x: np.ndarray, zero_threshold: float = 0.0) -> np.ndarray:
    """
    Convert model outputs into market shares without SoftMax.

    Rule:
    - anything <= zero_threshold is treated as zero demand
    - positive values are scaled to sum to 1

    This is better than SoftMax here because the model output is already a
    share-like regression target, not an unbounded logit.
    """
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    z = np.where(z > zero_threshold, z, 0.0)
    total = np.sum(z)
    if total <= 0.0:
        return np.ones_like(z) / len(z)
    return z / total


def run_market_share(
    model: tf.keras.Model,
    raw_df: pd.DataFrame,
    category_name: str,
    targets: Dict[str, float],
    zero_threshold: float = 0.0,
) -> pd.DataFrame:
    model_df = to_model_df(raw_df, category_name, targets)
    raw_scores = predict_scores(model, model_df)
    shares = nonnegative_share_scale(raw_scores, zero_threshold=zero_threshold)
    out = raw_df[["Name"]].copy()
    out["RawScore"] = raw_scores
    out["MarketShare"] = shares
    return out.sort_values("MarketShare", ascending=False).reset_index(drop=True)


def allocate_units(shares: np.ndarray, total_units_next_period: int) -> np.ndarray:
    exact = shares * total_units_next_period
    base = np.floor(exact).astype(int)
    remainder = int(total_units_next_period - base.sum())
    if remainder > 0:
        frac = exact - base
        order = np.argsort(-frac)
        base[order[:remainder]] += 1
    return base


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Capsim Forecaster", layout="wide")
st.title("Capsim Forecaster")

try:
    model, model_path = load_latest_keras_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.caption(f"Loaded model: {model_path}")

st.subheader("Category Setup")
col1, col2, col3, col4 = st.columns(4)
with col1:
    category_name = st.selectbox("Product category", CATEGORY_NAMES)
with col2:
    total_market_size = st.number_input("Total possible market size", min_value=1, value=10000, step=1)
with col3:
    growth_rate_pct = st.number_input("Expected growth next period (%)", value=10.0, step=0.1)
with col4:
    product_count = st.number_input("Number of products in category", min_value=2, value=5, step=1)

st.subheader("Category Targets")
t1, t2, t3, t4 = st.columns(4)
with t1:
    exp_age = st.number_input("Target age", min_value=0.1, value=2.0, step=0.1)
with t2:
    exp_price_low = st.number_input("Target price low", min_value=0.01, value=19.0, step=0.1)
with t3:
    exp_price_high = st.number_input("Target price high", min_value=0.01, value=29.0, step=0.1)
with t4:
    exp_performance = st.number_input("Target performance", min_value=0.01, value=6.4, step=0.1)

t5, t6, t7 = st.columns(3)
with t5:
    exp_size = st.number_input("Target size", min_value=0.01, value=13.6, step=0.1)
with t6:
    exp_reliability_low = st.number_input("Target reliability low", min_value=1, value=14000, step=100)
with t7:
    exp_reliability_high = st.number_input("Target reliability high", min_value=1, value=19000, step=100)

if exp_age < 0.5:
    exp_age = 0.5

targets = {
    "ExpAge": float(exp_age),
    "ExpPriceLow": float(exp_price_low),
    "ExpPriceHigh": float(exp_price_high),
    "ExpPerformance": float(exp_performance),
    "ExpSize": float(exp_size),
    "ExpReliabilityLow": float(exp_reliability_low),
    "ExpReliabilityHigh": float(exp_reliability_high),
}

if targets["ExpPriceLow"] >= targets["ExpPriceHigh"]:
    st.error("Expected price low must be less than expected price high.")
    st.stop()
if targets["ExpReliabilityLow"] >= targets["ExpReliabilityHigh"]:
    st.error("Expected reliability low must be less than expected reliability high.")
    st.stop()

st.subheader("Next-Step Product Inputs")
input_rows: List[Dict[str, float]] = []
for i in range(int(product_count)):
    st.markdown(f"**Product {i + 1}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        name = st.text_input(f"Name {i + 1}", value=f"Product {i + 1}", key=f"name_{i}")
    with c2:
        price = st.number_input(f"Price {i + 1}", min_value=0.01, value=25.0, step=0.1, key=f"price_{i}")
    with c3:
        age = st.number_input(f"Age {i + 1}", min_value=0.1, value=2.0, step=0.1, key=f"age_{i}")
    with c4:
        performance = st.number_input(f"Performance {i + 1}", min_value=0.01, value=6.0, step=0.1, key=f"perf_{i}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        size = st.number_input(f"Size {i + 1}", min_value=0.01, value=14.0, step=0.1, key=f"size_{i}")
    with c6:
        reliability = st.number_input(f"Reliability {i + 1}", min_value=1, value=17000, step=100, key=f"rel_{i}")
    with c7:
        acessibility = st.number_input(f"Accessibility {i + 1}", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"acc_{i}")
    with c8:
        awarness = st.number_input(f"Awareness {i + 1}", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"aware_{i}")

    input_rows.append(
        {
            "Name": str(name).strip() or f"Product {i + 1}",
            "Price": float(price),
            "Age": float(age),
            "Performance": float(performance),
            "Size": float(size),
            "Reliability": float(reliability),
            "Acessibility": float(acessibility),
            "Awarness": float(awarness),
        }
    )

zero_threshold = st.number_input(
    "Zero-sales cutoff",
    value=0.0,
    step=0.001,
    help="Any model output at or below this value is treated as zero market share before the remaining products are scaled.",
)

run_button = st.button("Calculate market shares", type="primary")

if run_button:
    raw_df = pd.DataFrame(input_rows, columns=RAW_PRODUCT_COLS)
    total_units_next_period = int(round(float(total_market_size) * (1.0 + float(growth_rate_pct) / 100.0)))

    result_df = run_market_share(model, raw_df, category_name, targets, zero_threshold=zero_threshold)
    result_df["Market Share %"] = result_df["MarketShare"] * 100.0
    result_df["Expected Units Sold"] = allocate_units(result_df["MarketShare"].values, total_units_next_period)

    display_df = result_df[["Name", "RawScore", "Market Share %", "Expected Units Sold"]].copy()
    display_df["RawScore"] = display_df["RawScore"].round(6)
    display_df["Market Share %"] = display_df["Market Share %"].round(2)
    display_df["Expected Units Sold"] = display_df["Expected Units Sold"].astype(int)

    st.caption("Market shares are computed by clipping any model output at or below the zero-sales cutoff to 0, then proportionally scaling the remaining positive outputs to sum to 100%.")

    st.subheader("Results")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Next-period total units", f"{total_units_next_period:,}")
    with r2:
        st.metric("Products evaluated", f"{len(display_df)}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)
