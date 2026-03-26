from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================
CATEGORY_MAP = {
    "Traditional": 0,
    "Low End": 1,
    "High End": 2,
    "Performance": 3,
    "Size": 4,
}
CATEGORY_ALIASES = {
    "Traditional": "Traditional",
    "Low end": "Low End",
    "Low End": "Low End",
    "High end": "High End",
    "High End": "High End",
    "Performance": "Performance",
    "Size": "Size",
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

EDITABLE_PRODUCT_COLS = [
    "Name",
    "Price",
    "Age",
    "Performance",
    "Size",
    "Reliability",
    "Acessibility",
    "Awarness",
]

DEFAULT_FEATURE_RANGES = {
    "price_pct_of_target_mid": (-0.5, 0.5),
    "age_pct_of_target": (-0.5, 3.5),
    "performance_pct_of_target": (-0.5, 0.5),
    "size_pct_of_target": (-0.5, 0.5),
    "reliability_pct_of_target_mid": (-0.5, 0.5),
    "customer_accessibility": (0.0, 1.0),
    "customer_awareness": (0.0, 1.0),
}

FEATURE_LABELS = {
    "price_pct_of_target_mid": "Price % of target mid",
    "age_pct_of_target": "Age % of target",
    "performance_pct_of_target": "Performance % of target",
    "size_pct_of_target": "Size % of target",
    "reliability_pct_of_target_mid": "Reliability % of target mid",
    "customer_accessibility": "Customer accessibility",
    "customer_awareness": "Customer awareness",
}


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
# EXCEL PARSING
# ============================================================
def _clean_category_name(sheet_name: str) -> str | None:
    return CATEGORY_ALIASES.get(str(sheet_name).strip())


def _to_float(text: object) -> float:
    if pd.isna(text):
        raise ValueError("Missing numeric value")
    s = str(text).strip().replace(",", "")
    s = s.replace("$", "")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if not match:
        raise ValueError(f"Could not parse numeric value from: {text}")
    return float(match.group(0))


def _parse_price_range(text: object) -> Tuple[float, float]:
    s = str(text).replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        raise ValueError(f"Could not parse price range from: {text}")
    return float(nums[0]), float(nums[1])


def _parse_reliability_range(text: object) -> Tuple[float, float]:
    s = str(text).replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) < 2:
        raise ValueError(f"Could not parse reliability range from: {text}")
    return float(nums[0]), float(nums[1])


def _parse_positioning(text: object) -> Tuple[float, float]:
    s = str(text)
    perf_match = re.search(r"Performance\s+(-?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    size_match = re.search(r"Size\s+(-?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if not perf_match or not size_match:
        raise ValueError(f"Could not parse positioning from: {text}")
    return float(perf_match.group(1)), float(size_match.group(1))


def _extract_targets(sheet_df: pd.DataFrame) -> Dict[str, float]:
    row_labels = sheet_df.iloc[:, 0].astype(str).str.strip()
    values = sheet_df.iloc[:, 1]
    lookup = dict(zip(row_labels, values))

    exp_price_low, exp_price_high = _parse_price_range(lookup["Price"])
    exp_reliability_low, exp_reliability_high = _parse_reliability_range(lookup["Reliability"])
    exp_performance, exp_size = _parse_positioning(lookup["Positioning"])
    exp_age = max(_to_float(lookup["Age"]), 0.5)

    total_market_size = int(round(_to_float(lookup[next(k for k in lookup if "Total Market Size" in k)])))
    demand_growth_rate = float(_to_float(lookup[next(k for k in lookup if "Demand Growth Rate" in k)]))

    return {
        "ExpAge": exp_age,
        "ExpPriceLow": exp_price_low,
        "ExpPriceHigh": exp_price_high,
        "ExpPerformance": exp_performance,
        "ExpSize": exp_size,
        "ExpReliabilityLow": exp_reliability_low,
        "ExpReliabilityHigh": exp_reliability_high,
        "TotalMarketSize": total_market_size,
        "GrowthRate": demand_growth_rate * 100.0,
    }


def _extract_products(sheet_df: pd.DataFrame) -> pd.DataFrame:
    header_row_idx = None
    for idx in range(len(sheet_df)):
        row = sheet_df.iloc[idx].tolist()
        row_str = [str(x).strip() for x in row]
        if "Name" in row_str and "Price" in row_str and "Age" in row_str:
            header_row_idx = idx
            break

    if header_row_idx is None:
        raise ValueError("Could not find product table header row.")

    product_df = sheet_df.iloc[header_row_idx + 1 :].copy()
    product_df.columns = [str(x).strip() for x in sheet_df.iloc[header_row_idx].tolist()]
    product_df = product_df.loc[:, ~pd.Index(product_df.columns).duplicated()]

    keep_cols = [
        "Name",
        "Price",
        "Age",
        "Performance",
        "Size",
        "Reliability",
        "Sales Budget",
    ]
    product_df = product_df[[c for c in keep_cols if c in product_df.columns]].copy()
    product_df = product_df[product_df["Name"].notna()].copy()
    product_df["Name"] = product_df["Name"].astype(str).str.strip()
    product_df = product_df[product_df["Name"] != ""]

    for col in ["Price", "Age", "Performance", "Size", "Reliability"]:
        product_df[col] = pd.to_numeric(product_df[col], errors="coerce")

    # Budget in the workbook is a practical default for both accessibility and awareness.
    budget = pd.to_numeric(product_df.get("Sales Budget", pd.Series(index=product_df.index, dtype=float)), errors="coerce")
    budget = budget.fillna(0.0)
    normalized_budget = (budget / 2000.0).clip(lower=0.0, upper=1.0)

    out = pd.DataFrame()
    out["Name"] = product_df["Name"]
    out["Price"] = product_df["Price"].fillna(0.0)
    out["Age"] = product_df["Age"].fillna(0.0)
    out["Performance"] = product_df["Performance"].fillna(0.0)
    out["Size"] = product_df["Size"].fillna(0.0)
    out["Reliability"] = product_df["Reliability"].fillna(0.0)
    out["Acessibility"] = normalized_budget.round(4)
    out["Awarness"] = normalized_budget.round(4)

    return out[EDITABLE_PRODUCT_COLS].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_capsim_workbook(uploaded_bytes: bytes) -> Dict[str, Dict[str, object]]:
    workbook = pd.ExcelFile(uploaded_bytes)
    parsed: Dict[str, Dict[str, object]] = {}

    for raw_sheet_name in workbook.sheet_names:
        category_name = _clean_category_name(raw_sheet_name)
        if not category_name:
            continue

        sheet_df = pd.read_excel(uploaded_bytes, sheet_name=raw_sheet_name, header=None)
        parsed[category_name] = {
            "targets": _extract_targets(sheet_df),
            "products": _extract_products(sheet_df),
            "source_sheet": raw_sheet_name,
        }

    missing = [name for name in CATEGORY_NAMES if name not in parsed]
    if missing:
        raise ValueError(f"Workbook is missing required category tabs: {missing}")

    return parsed


# ============================================================
# FORECAST HELPERS
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
# OPTIMIZATION PLOT
# ============================================================
def generate_single_feature_sweep(
    model: tf.keras.Model,
    raw_df: pd.DataFrame,
    category_name: str,
    targets: Dict[str, float],
    product_name: str,
    feature_name: str,
    feature_range: Tuple[float, float],
    zero_threshold: float,
    num_points: int = 100,
) -> pd.DataFrame:
    base_model_df = to_model_df(raw_df, category_name, targets)
    product_idx = raw_df.index[raw_df["Name"] == product_name]
    if len(product_idx) == 0:
        raise ValueError(f"Product not found: {product_name}")
    product_idx = int(product_idx[0])

    sweep_vals = np.linspace(feature_range[0], feature_range[1], num_points, dtype=np.float32)
    repeated = pd.concat([base_model_df] * num_points, ignore_index=True)

    for i, val in enumerate(sweep_vals):
        row_idx = i * len(base_model_df) + product_idx
        repeated.at[row_idx, feature_name] = float(val)

    preds = predict_scores(model, repeated)
    preds = preds.reshape(num_points, len(base_model_df))

    target_market_share = []
    target_raw_score = []
    for i in range(num_points):
        row_scores = preds[i]
        shares = nonnegative_share_scale(row_scores, zero_threshold=zero_threshold)
        target_raw_score.append(float(row_scores[product_idx]))
        target_market_share.append(float(shares[product_idx]))

    return pd.DataFrame(
        {
            "sweep_value": sweep_vals,
            "predicted_raw_score": target_raw_score,
            "predicted_market_share": target_market_share,
        }
    )


def make_optimization_figure(
    sweep_df: pd.DataFrame,
    feature_name: str,
    current_value: float,
    product_name: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep_df["sweep_value"], sweep_df["predicted_market_share"] * 100.0)
    ax.axvline(current_value, linestyle="--")
    ax.set_title(f"{product_name}: {FEATURE_LABELS[feature_name]} optimization curve")
    ax.set_xlabel(FEATURE_LABELS[feature_name])
    ax.set_ylabel("Predicted market share %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Capsim Forecaster v4", layout="wide")
st.title("Capsim Forecaster v4")
st.caption("Upload a Capsim workbook, choose a category, edit the imported products, forecast market share, and generate a one-feature optimization curve.")

try:
    model, model_path = load_latest_keras_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.caption(f"Loaded model: {model_path}")

uploaded_file = st.file_uploader(
    "Upload Capsim workbook",
    type=["xlsx", "xlsm", "xls"],
    help="The app reads the category tabs and ignores the non-category tabs.",
)

if uploaded_file is None:
    st.info("Upload an Excel workbook to begin.")
    st.stop()

try:
    workbook_data = parse_capsim_workbook(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Could not parse workbook: {e}")
    st.stop()

st.subheader("Category Setup")
left, right = st.columns([1, 2])
with left:
    category_name = st.selectbox("Forecast category", CATEGORY_NAMES)
with right:
    st.caption(f"Source sheet: {workbook_data[category_name]['source_sheet']}")

selected_targets = dict(workbook_data[category_name]["targets"])
selected_products = workbook_data[category_name]["products"].copy()

c1, c2, c3, c4 = st.columns(4)
with c1:
    total_market_size = st.number_input(
        "Total possible market size",
        min_value=1,
        value=int(selected_targets["TotalMarketSize"]),
        step=1,
    )
with c2:
    growth_rate_pct = st.number_input(
        "Expected growth next period (%)",
        value=float(selected_targets["GrowthRate"]),
        step=0.1,
    )
with c3:
    zero_threshold = st.number_input(
        "Zero-sales cutoff",
        value=0.0,
        step=0.001,
        help="Any model output at or below this value is treated as zero market share before the remaining products are scaled.",
    )
with c4:
    num_sweep_points = st.slider("Optimization curve points", min_value=25, max_value=300, value=100, step=25)

st.subheader("Category Targets")
t1, t2, t3, t4 = st.columns(4)
with t1:
    exp_age = st.number_input("Target age", min_value=0.1, value=float(selected_targets["ExpAge"]), step=0.1)
with t2:
    exp_price_low = st.number_input("Target price low", min_value=0.01, value=float(selected_targets["ExpPriceLow"]), step=0.1)
with t3:
    exp_price_high = st.number_input("Target price high", min_value=0.01, value=float(selected_targets["ExpPriceHigh"]), step=0.1)
with t4:
    exp_performance = st.number_input("Target performance", min_value=0.01, value=float(selected_targets["ExpPerformance"]), step=0.1)

t5, t6, t7 = st.columns(3)
with t5:
    exp_size = st.number_input("Target size", min_value=0.01, value=float(selected_targets["ExpSize"]), step=0.1)
with t6:
    exp_reliability_low = st.number_input("Target reliability low", min_value=1, value=int(selected_targets["ExpReliabilityLow"]), step=100)
with t7:
    exp_reliability_high = st.number_input("Target reliability high", min_value=1, value=int(selected_targets["ExpReliabilityHigh"]), step=100)

exp_age = max(exp_age, 0.5)
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

st.subheader("Imported products")
st.caption("The workbook values are preloaded here. You can edit any product before forecasting.")

editable_df = st.data_editor(
    selected_products,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "Name": st.column_config.TextColumn(required=True),
        "Price": st.column_config.NumberColumn(min_value=0.01, step=0.1),
        "Age": st.column_config.NumberColumn(min_value=0.0, step=0.1),
        "Performance": st.column_config.NumberColumn(min_value=0.0, step=0.1),
        "Size": st.column_config.NumberColumn(min_value=0.0, step=0.1),
        "Reliability": st.column_config.NumberColumn(min_value=1.0, step=100.0),
        "Acessibility": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
        "Awarness": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    },
    key="products_editor",
)

for col in EDITABLE_PRODUCT_COLS:
    if col != "Name":
        editable_df[col] = pd.to_numeric(editable_df[col], errors="coerce")
editable_df["Name"] = editable_df["Name"].astype(str).str.strip()
editable_df = editable_df[editable_df["Name"] != ""].copy()
editable_df = editable_df.fillna(0.0)

run_forecast = st.button("Calculate market shares", type="primary")

if run_forecast:
    total_units_next_period = int(round(float(total_market_size) * (1.0 + float(growth_rate_pct) / 100.0)))
    result_df = run_market_share(model, editable_df, category_name, targets, zero_threshold=zero_threshold)
    result_df["Market Share %"] = result_df["MarketShare"] * 100.0
    result_df["Expected Units Sold"] = allocate_units(result_df["MarketShare"].values, total_units_next_period)

    display_df = result_df[["Name", "RawScore", "Market Share %", "Expected Units Sold"]].copy()
    display_df["RawScore"] = display_df["RawScore"].round(6)
    display_df["Market Share %"] = display_df["Market Share %"].round(2)
    display_df["Expected Units Sold"] = display_df["Expected Units Sold"].astype(int)

    st.caption("Market shares are computed by clipping any model output at or below the zero-sales cutoff to 0, then proportionally scaling the remaining positive outputs to sum to 100%.")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Next-period total units", f"{total_units_next_period:,}")
    with r2:
        st.metric("Products evaluated", f"{len(display_df)}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Single-dimension optimization")
st.caption("This does not change the product for the user. It only shows the forecast curve while all other products and values stay fixed.")

if len(editable_df) == 0:
    st.warning("Add at least one product to use the optimization tool.")
    st.stop()

opt1, opt2 = st.columns(2)
with opt1:
    selected_product = st.selectbox("Product to optimize", editable_df["Name"].tolist())
with opt2:
    selected_feature = st.selectbox(
        "Feature to sweep",
        NUM_COLS,
        format_func=lambda x: FEATURE_LABELS[x],
    )

range_col1, range_col2 = st.columns(2)
default_lo, default_hi = DEFAULT_FEATURE_RANGES[selected_feature]
with range_col1:
    sweep_lo = st.number_input("Sweep minimum", value=float(default_lo), step=0.01)
with range_col2:
    sweep_hi = st.number_input("Sweep maximum", value=float(default_hi), step=0.01)

if sweep_lo >= sweep_hi:
    st.error("Sweep minimum must be less than sweep maximum.")
    st.stop()

if st.button("Generate optimization curve"):
    model_df_for_current = to_model_df(editable_df, category_name, targets)
    current_value = float(
        model_df_for_current.loc[editable_df["Name"] == selected_product, selected_feature].iloc[0]
    )

    sweep_df = generate_single_feature_sweep(
        model=model,
        raw_df=editable_df,
        category_name=category_name,
        targets=targets,
        product_name=selected_product,
        feature_name=selected_feature,
        feature_range=(float(sweep_lo), float(sweep_hi)),
        zero_threshold=float(zero_threshold),
        num_points=int(num_sweep_points),
    )

    fig = make_optimization_figure(
        sweep_df=sweep_df,
        feature_name=selected_feature,
        current_value=current_value,
        product_name=selected_product,
    )
    st.pyplot(fig)

    peak_idx = int(sweep_df["predicted_market_share"].idxmax())
    peak_x = float(sweep_df.loc[peak_idx, "sweep_value"])
    peak_y = float(sweep_df.loc[peak_idx, "predicted_market_share"] * 100.0)
    current_y = float(
        sweep_df.iloc[(sweep_df["sweep_value"] - current_value).abs().idxmin()]["predicted_market_share"] * 100.0
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Current forecast share", f"{current_y:.2f}%")
    with s2:
        st.metric("Peak forecast share in sweep", f"{peak_y:.2f}%")
    with s3:
        st.metric("Peak sweep value", f"{peak_x:.4f}")

    st.dataframe(
        sweep_df.assign(predicted_market_share_pct=lambda d: d["predicted_market_share"] * 100.0)
        [["sweep_value", "predicted_raw_score", "predicted_market_share_pct"]]
        .rename(columns={"predicted_market_share_pct": "predicted_market_share_%"})
        .round(6),
        use_container_width=True,
        hide_index=True,
    )
