from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


CATEGORY_MAP = {
    "Traditional": 0,
    "Low end": 1,
    "High end": 2,
    "Performance": 3,
    "Size": 4,
}
CATEGORY_NAMES = list(CATEGORY_MAP.keys())
CATEGORY_CODE_TO_NAME = {v: k for k, v in CATEGORY_MAP.items()}

NUM_COLS = [
    "Price%",
    "Age%",
    "Performance%",
    "Size%",
    "Reliability%",
    "Acessibility%",
    "Awarness%",
]
ALL_COLS = ["Company", "Categories"] + NUM_COLS

MARKET_TARGET_COLS = [
    "PriceMin",
    "PriceMax",
    "AgeYears",
    "PerformanceTarget",
    "SizeTarget",
    "ReliabilityMin",
    "ReliabilityMax",
    "AcessibilityTarget",
    "AwarnessTarget",
]

PRODUCT_INPUT_COLS = [
    "Company",
    "Price",
    "Age",
    "Performance",
    "Size",
    "Reliability",
    "Acessibility",
    "Awarness",
]

DEFAULT_MARKET_TARGETS = {
    "PriceMin": 18.0,
    "PriceMax": 22.0,
    "AgeYears": 2.0,
    "PerformanceTarget": 10.0,
    "SizeTarget": 10.0,
    "ReliabilityMin": 14000.0,
    "ReliabilityMax": 19000.0,
    "AcessibilityTarget": 50.0,
    "AwarnessTarget": 50.0,
}

RAW_OPTIMIZATION_DEFAULTS = {
    "Price": (10.0, 40.0),
    "Age": (0.5, 5.0),
    "Performance": (0.0, 20.0),
    "Size": (0.0, 20.0),
    "Reliability": (8000.0, 28000.0),
    "Acessibility": (0.0, 100.0),
    "Awarness": (0.0, 100.0),
}

PER_PERIOD_DEFAULTS = {
    "Price": None,
    "Age": None,
    "Performance": 1.2,
    "Size": 1.1,
    "Reliability": 1500.0,
    "Acessibility": None,
    "Awarness": None,
}

RAW_TO_MODEL_COL = {
    "Price": "Price%",
    "Age": "Age%",
    "Performance": "Performance%",
    "Size": "Size%",
    "Reliability": "Reliability%",
    "Acessibility": "Acessibility%",
    "Awarness": "Awarness%",
}

DISPLAY_NAMES = {
    "Price": "Price",
    "Age": "Age",
    "Performance": "Performance",
    "Size": "Size",
    "Reliability": "Reliability",
    "Acessibility": "Customer Accessibility",
    "Awarness": "Customer Awareness",
}

OPTIMIZE_OPTIONS = ["Price", "Age", "Performance", "Size", "Reliability", "Acessibility", "Awarness"]


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


def parse_money_to_float(value: object) -> float:
    text = str(value).strip().replace("$", "").replace(",", "")
    text = text.replace("%", "")
    if text == "" or text.lower() == "nan":
        return np.nan
    return float(text)


def parse_percent_to_float(value: object) -> float:
    text = str(value).strip().replace("%", "")
    if text == "" or text.lower() == "nan":
        return np.nan
    return float(text)


def parse_range_pair(text: object) -> Tuple[float, float]:
    raw = str(text).replace("$", "").replace(",", "").strip()
    parts = [p.strip() for p in raw.split("-")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse range: {text}")
    return float(parts[0]), float(parts[1])


def parse_positioning(text: object) -> Tuple[float, float]:
    raw = str(text).replace(",", " ").strip()
    parts = raw.split()
    perf_idx = parts.index("Performance")
    size_idx = parts.index("Size")
    return float(parts[perf_idx + 1]), float(parts[size_idx + 1])


def clean_product_table(product_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Name": "Company",
        "Customer Accessibility": "Acessibility",
        "Customer Awareness": "Awarness",
    }
    out = product_df.rename(columns=rename_map).copy()
    required = [
        "Company",
        "Price",
        "Age",
        "Performance",
        "Size",
        "Reliability",
        "Acessibility",
        "Awarness",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Top Products section is missing columns: {missing}")

    out = out[required].copy()
    out["Company"] = out["Company"].astype(str).str.strip()
    out = out.loc[out["Company"].ne("")].copy()

    out["Price"] = out["Price"].apply(parse_money_to_float)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Performance"] = pd.to_numeric(out["Performance"], errors="coerce")
    out["Size"] = pd.to_numeric(out["Size"], errors="coerce")
    out["Reliability"] = out["Reliability"].apply(parse_money_to_float)
    out["Acessibility"] = out["Acessibility"].apply(parse_percent_to_float)
    out["Awarness"] = out["Awarness"].apply(parse_percent_to_float)
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def parse_capsim_segment_csv(uploaded_file) -> Tuple[str, Dict[str, float], pd.DataFrame]:
    raw = pd.read_csv(uploaded_file, header=None, dtype=str, keep_default_na=False)
    segment_name = str(raw.iat[0, 0]).strip()
    if segment_name not in CATEGORY_MAP:
        raise ValueError(
            f"First cell must be one of {CATEGORY_NAMES}. Got: {segment_name!r}"
        )

    row0 = raw.iloc[:, 0].astype(str).str.strip()
    top_idx_list = raw.index[row0.eq("Top Products")].tolist()
    if not top_idx_list:
        raise ValueError("Could not find 'Top Products' section in CSV.")
    top_idx = top_idx_list[0]

    age_idx = raw.index[row0.eq("Age")].tolist()[0]
    price_idx = raw.index[row0.eq("Price")].tolist()[0]
    positioning_idx = raw.index[row0.eq("Positioning")].tolist()[0]
    reliability_idx = raw.index[row0.eq("Reliability")].tolist()[0]

    price_min, price_max = parse_range_pair(raw.iat[price_idx, 1])
    reliability_min, reliability_max = parse_range_pair(raw.iat[reliability_idx, 1])
    perf_target, size_target = parse_positioning(raw.iat[positioning_idx, 1])

    age_text = str(raw.iat[age_idx, 1]).replace("Years", "").replace("Year", "").strip()
    age_years = float(age_text)

    header_row = raw.iloc[top_idx + 1].tolist()
    data_rows = raw.iloc[top_idx + 2 :].copy()
    data_rows.columns = header_row
    data_rows = data_rows.loc[data_rows.iloc[:, 0].astype(str).str.strip().ne("")].copy()
    product_df = clean_product_table(data_rows)

    market_targets = {
        "PriceMin": price_min,
        "PriceMax": price_max,
        "AgeYears": age_years,
        "PerformanceTarget": perf_target,
        "SizeTarget": size_target,
        "ReliabilityMin": reliability_min,
        "ReliabilityMax": reliability_max,
        "AcessibilityTarget": float(product_df["Acessibility"].mean()),
        "AwarnessTarget": float(product_df["Awarness"].mean()),
    }
    return segment_name, market_targets, product_df


def product_row_template(idx: int) -> Dict[str, object]:
    return {
        "Company": f"Company {idx + 1}",
        **DEFAULT_PRODUCT_VALUES,
    }


def safe_mid(a: float, b: float) -> float:
    return (float(a) + float(b)) / 2.0


def pct_from_mid(actual: float, reference: float) -> float:
    reference = float(reference)
    if abs(reference) < 1e-8:
        return 0.0
    return float((float(actual) - reference) / reference)


def raw_to_model_df(product_df: pd.DataFrame, category_name: str, market_targets: Dict[str, float]) -> pd.DataFrame:
    df = product_df.copy()
    for col in PRODUCT_INPUT_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required input column: {col}")

    exp_mid_price = safe_mid(market_targets["PriceMin"], market_targets["PriceMax"])
    exp_reliability = safe_mid(market_targets["ReliabilityMin"], market_targets["ReliabilityMax"])

    model_rows = []
    for _, row in df.iterrows():
        model_rows.append(
            {
                "Company": str(row["Company"]).strip(),
                "Categories": CATEGORY_MAP[category_name],
                "Price%": (float(row["Price"]) / exp_mid_price) - 1.0,
                "Age%": (float(row["Age"]) / float(market_targets["AgeYears"])) - 1.0,
                "Performance%": (float(row["Performance"]) / float(market_targets["PerformanceTarget"])) - 1.0,
                "Size%": (float(row["Size"]) / float(market_targets["SizeTarget"])) - 1.0,
                "Reliability%": (float(row["Reliability"]) / exp_reliability) - 1.0,
                "Acessibility%": float(row["Acessibility"]),
                "Awarness%": float(row["Awarness"]),
            }
        )
    return pd.DataFrame(model_rows, columns=ALL_COLS)


def dataframe_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = df.copy()
    clean["Company"] = clean["Company"].astype(str)
    clean["Categories"] = pd.to_numeric(clean["Categories"], errors="coerce")
    for col in NUM_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean[["Categories"] + NUM_COLS].isna().any().any():
        bad_rows = clean.loc[clean[["Categories"] + NUM_COLS].isna().any(axis=1)]
        raise ValueError(
            "Non-finite values found in scenario table before inference.\n"
            f"{bad_rows.to_string(index=False)}"
        )

    companies = clean["Company"].values
    categories = clean["Categories"].astype(np.int32).values
    numeric = clean[NUM_COLS].astype(np.float32).values
    return companies, categories, numeric


def predict_raw_scores_arrays(
    model: tf.keras.Model,
    numeric: np.ndarray,
    categories: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    preds = model.predict(
        {
            "numeric_features": numeric.astype(np.float32, copy=False),
            "category": categories.reshape(-1, 1).astype(np.int32, copy=False),
        },
        batch_size=batch_size,
        verbose=0,
    )
    return np.asarray(preds, dtype=np.float32).reshape(-1)


@njit(cache=True)
def softmax_numba(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x.astype(np.float64) / max(temperature, 1e-8)
    max_z = np.max(z)
    out = np.empty_like(z)
    total = 0.0
    for i in range(z.shape[0]):
        out[i] = np.exp(z[i] - max_z)
        total += out[i]
    for i in range(out.shape[0]):
        out[i] /= total
    return out


@njit(cache=True)
def softmax_batch_numba(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    n_rows, n_cols = x.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    temp = max(temperature, 1e-8)
    for r in range(n_rows):
        max_z = -1e30
        for c in range(n_cols):
            v = x[r, c] / temp
            if v > max_z:
                max_z = v
        total = 0.0
        for c in range(n_cols):
            val = np.exp(x[r, c] / temp - max_z)
            out[r, c] = val
            total += val
        for c in range(n_cols):
            out[r, c] /= total
    return out


def next_step_market_share(model: tf.keras.Model, df: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    _, categories, numeric = dataframe_to_arrays(out)
    out["RawScore"] = predict_raw_scores_arrays(model, numeric, categories)
    out["NextStepShare"] = softmax_numba(out["RawScore"].values.astype(np.float64), temperature=temperature)
    return out.sort_values("NextStepShare", ascending=False).reset_index(drop=True)


def scenario_market_shares_batch(
    model: tf.keras.Model,
    scenario_numeric: np.ndarray,
    scenario_categories: np.ndarray,
    temperature: float,
    infer_batch_size: int = 8192,
) -> np.ndarray:
    n_scenarios, n_companies, n_features = scenario_numeric.shape
    flat_numeric = scenario_numeric.reshape(n_scenarios * n_companies, n_features)
    flat_categories = np.repeat(scenario_categories.astype(np.int32), n_scenarios)
    raw = predict_raw_scores_arrays(model, flat_numeric, flat_categories, batch_size=infer_batch_size)
    raw = raw.reshape(n_scenarios, n_companies)
    shares = softmax_batch_numba(raw.astype(np.float64), temperature=temperature)
    return shares.astype(np.float32)


def percent_to_raw_value(feature_name: str, value_pct: float, market_targets: Dict[str, float]) -> float:
    if feature_name == "Price":
        return float(safe_mid(market_targets["PriceMin"], market_targets["PriceMax"]) * (1.0 + value_pct))
    if feature_name == "Age":
        return float(market_targets["AgeYears"] * (1.0 + value_pct))
    if feature_name == "Performance":
        return float(market_targets["PerformanceTarget"] * (1.0 + value_pct))
    if feature_name == "Size":
        return float(market_targets["SizeTarget"] * (1.0 + value_pct))
    if feature_name == "Reliability":
        return float(safe_mid(market_targets["ReliabilityMin"], market_targets["ReliabilityMax"]) * (1.0 + value_pct))
    if feature_name == "Acessibility":
        return float(value_pct)
    if feature_name == "Awarness":
        return float(value_pct)
    return float(value_pct)


def raw_value_to_model_percent(feature_name: str, raw_value: float, market_targets: Dict[str, float]) -> float:
    if feature_name == "Price":
        return (float(raw_value) / safe_mid(market_targets["PriceMin"], market_targets["PriceMax"])) - 1.0
    if feature_name == "Age":
        return (float(raw_value) / float(market_targets["AgeYears"])) - 1.0
    if feature_name == "Performance":
        return (float(raw_value) / float(market_targets["PerformanceTarget"])) - 1.0
    if feature_name == "Size":
        return (float(raw_value) / float(market_targets["SizeTarget"])) - 1.0
    if feature_name == "Reliability":
        return (float(raw_value) / safe_mid(market_targets["ReliabilityMin"], market_targets["ReliabilityMax"])) - 1.0
    if feature_name == "Acessibility":
        return float(raw_value)
    if feature_name == "Awarness":
        return float(raw_value)
    return float(raw_value)


@dataclass
class SearchConfig:
    target_company: str
    optimize_cols: List[str]
    bounds: Dict[str, Tuple[float, float]]
    grid_points: int = 9
    opponent_drift: float = 0.0
    softmax_temperature: float = 1.0
    infer_batch_size: int = 8192
    scenario_chunk_size: int = 4096


def build_candidate_grid_numeric(base_numeric_row: np.ndarray, search_cfg: SearchConfig) -> np.ndarray:
    candidate_matrix = np.repeat(base_numeric_row.reshape(1, -1), search_cfg.grid_points ** len(search_cfg.optimize_cols), axis=0)
    if not search_cfg.optimize_cols:
        return candidate_matrix.astype(np.float32)

    sweep_axes = []
    for col in search_cfg.optimize_cols:
        lo, hi = search_cfg.bounds[col]
        sweep_axes.append(np.linspace(lo, hi, search_cfg.grid_points, dtype=np.float32))

    mesh = np.meshgrid(*sweep_axes, indexing="ij")
    combos = np.stack([m.reshape(-1) for m in mesh], axis=1)

    for i, col in enumerate(search_cfg.optimize_cols):
        col_idx = NUM_COLS.index(col)
        candidate_matrix[:, col_idx] = combos[:, i]

    return candidate_matrix.astype(np.float32, copy=False)


def optimize_single_company(
    model: tf.keras.Model,
    df_category: pd.DataFrame,
    search_cfg: SearchConfig,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, pd.Series]]:
    companies, categories, base_numeric = dataframe_to_arrays(df_category)
    company_to_idx = {c: i for i, c in enumerate(companies)}
    target_idx = company_to_idx[search_cfg.target_company]
    n_companies, n_features = base_numeric.shape

    target_candidates = build_candidate_grid_numeric(base_numeric[target_idx], search_cfg)

    optimized_competitors: Dict[str, pd.Series] = {}
    for company, company_idx in company_to_idx.items():
        if company == search_cfg.target_company:
            continue

        current_comp = base_numeric[company_idx].copy()
        if search_cfg.opponent_drift <= 0.0:
            optimized_competitors[company] = pd.Series(
                {
                    "Company": company,
                    "Categories": int(categories[company_idx]),
                    **{col: float(current_comp[i]) for i, col in enumerate(NUM_COLS)},
                }
            )
            continue

        comp_candidates = build_candidate_grid_numeric(current_comp, search_cfg)
        n_candidates = comp_candidates.shape[0]
        scenario_numeric = np.repeat(base_numeric.reshape(1, n_companies, n_features), n_candidates, axis=0)
        scenario_numeric[:, company_idx, :] = comp_candidates

        shares = scenario_market_shares_batch(
            model,
            scenario_numeric,
            categories,
            temperature=search_cfg.softmax_temperature,
            infer_batch_size=search_cfg.infer_batch_size,
        )
        best_idx = int(np.argmax(shares[:, company_idx]))
        best_comp = comp_candidates[best_idx]
        blended = (1.0 - search_cfg.opponent_drift) * current_comp + search_cfg.opponent_drift * best_comp
        optimized_competitors[company] = pd.Series(
            {
                "Company": company,
                "Categories": int(categories[company_idx]),
                **{col: float(blended[i]) for i, col in enumerate(NUM_COLS)},
            }
        )

    fixed_numeric = base_numeric.copy()
    for company, company_idx in company_to_idx.items():
        if company != search_cfg.target_company:
            fixed_numeric[company_idx] = optimized_competitors[company][NUM_COLS].values.astype(np.float32)

    scenario_numeric = np.repeat(fixed_numeric.reshape(1, n_companies, n_features), target_candidates.shape[0], axis=0)
    scenario_numeric[:, target_idx, :] = target_candidates

    shares = scenario_market_shares_batch(
        model,
        scenario_numeric,
        categories,
        temperature=search_cfg.softmax_temperature,
        infer_batch_size=search_cfg.infer_batch_size,
    )
    target_shares = shares[:, target_idx]

    results_df = pd.DataFrame(target_candidates, columns=NUM_COLS)
    results_df.insert(0, "Categories", int(categories[target_idx]))
    results_df.insert(0, "Company", search_cfg.target_company)
    results_df["TargetNextStepShare"] = target_shares.astype(np.float32)
    results_df = results_df.sort_values("TargetNextStepShare", ascending=False).reset_index(drop=True)
    return results_df.iloc[0], results_df, optimized_competitors


def monte_carlo_surface(
    model: tf.keras.Model,
    df_category: pd.DataFrame,
    target_company: str,
    best_row: pd.Series,
    x_feature: str,
    y_feature: str,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    x_points: int,
    y_points: int,
    noise_std: float,
    passes: int,
    temperature: float,
    infer_batch_size: int = 8192,
    scenario_chunk_size: int = 4096,
) -> pd.DataFrame:
    companies, categories, base_numeric = dataframe_to_arrays(df_category)
    company_to_idx = {c: i for i, c in enumerate(companies)}
    target_idx = company_to_idx[target_company]
    n_companies, n_features = base_numeric.shape

    x_idx = NUM_COLS.index(x_feature)
    y_idx = NUM_COLS.index(y_feature)
    x_vals = np.linspace(x_bounds[0], x_bounds[1], x_points, dtype=np.float32)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], y_points, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid_xy = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    n_grid = grid_xy.shape[0]
    total_scenarios = n_grid * passes

    target_shares_all = np.empty(total_scenarios, dtype=np.float32)
    best_numeric = np.array([best_row[col] for col in NUM_COLS], dtype=np.float32)
    competitor_indices = np.array([i for i in range(n_companies) if i != target_idx], dtype=np.int32)

    start = 0
    while start < total_scenarios:
        end = min(start + scenario_chunk_size, total_scenarios)
        chunk_size = end - start

        scenario_numeric = np.repeat(base_numeric.reshape(1, n_companies, n_features), chunk_size, axis=0)
        scenario_numeric[:, target_idx, :] = best_numeric.reshape(1, -1)

        flat_idx = np.arange(start, end)
        grid_idx = flat_idx // passes
        scenario_numeric[:, target_idx, x_idx] = grid_xy[grid_idx, 0]
        scenario_numeric[:, target_idx, y_idx] = grid_xy[grid_idx, 1]

        if len(competitor_indices) > 0 and noise_std > 0.0:
            noise = np.random.normal(0.0, noise_std, size=(chunk_size, len(competitor_indices), n_features)).astype(np.float32)
            scenario_numeric[:, competitor_indices, :] += noise

        shares = scenario_market_shares_batch(
            model,
            scenario_numeric,
            categories,
            temperature=temperature,
            infer_batch_size=infer_batch_size,
        )
        target_shares_all[start:end] = shares[:, target_idx]
        start = end

    target_shares_matrix = target_shares_all.reshape(n_grid, passes)
    return pd.DataFrame(
        {
            x_feature: grid_xy[:, 0],
            y_feature: grid_xy[:, 1],
            "MeanShare": np.mean(target_shares_matrix, axis=1),
            "StdShare": np.std(target_shares_matrix, axis=1),
            "P10Share": np.percentile(target_shares_matrix, 10, axis=1),
            "P50Share": np.percentile(target_shares_matrix, 50, axis=1),
            "P90Share": np.percentile(target_shares_matrix, 90, axis=1),
        }
    )


def make_surface_figure(surface_df: pd.DataFrame, x_feature: str, y_feature: str, z_col: str) -> go.Figure:
    x_vals = np.sort(surface_df[x_feature].unique())
    y_vals = np.sort(surface_df[y_feature].unique())
    z_pivot = surface_df.pivot_table(index=y_feature, columns=x_feature, values=z_col, aggfunc="mean").reindex(index=y_vals, columns=x_vals)
    z_vals = z_pivot.values.astype(np.float32)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="xy")

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_vals,
            colorscale="Viridis",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"3D Sensitivity Surface: {z_col}",
        scene=dict(xaxis_title=x_feature, yaxis_title=y_feature, zaxis_title=z_col),
        height=700,
        margin=dict(l=10, r=10, b=10, t=60),
    )
    return fig


def build_raw_results_table(results_df: pd.DataFrame, market_targets: Dict[str, float], optimize_features: List[str]) -> pd.DataFrame:
    out = results_df[["Company", "TargetNextStepShare"]].copy()
    for feature in optimize_features:
        model_col = RAW_TO_MODEL_COL[feature]
        out[feature] = results_df[model_col].apply(lambda v: percent_to_raw_value(feature, float(v), market_targets))
    return out


st.set_page_config(page_title="Category Optimization", layout="wide")
st.title("Category Optimization and Sensitivity Analysis")

try:
    model, model_path = load_latest_keras_model()
except Exception as e:
    st.error(f"Could not load model from models/: {e}")
    st.stop()

st.caption(f"Using model: {model_path}")

uploaded_csv = st.file_uploader(
    "Upload the segment CSV export",
    type=["csv"],
    help="Upload the raw segment CSV. The app will parse expectations and the Top Products table automatically.",
)

if uploaded_csv is None:
    st.info("Upload a raw segment CSV to begin.")
    st.stop()

try:
    category_name, market_targets, raw_products_df = parse_capsim_segment_csv(uploaded_csv)
    df_cat = raw_to_model_df(raw_products_df, category_name, market_targets)
except Exception as e:
    st.error(f"CSV parsing error: {e}")
    st.stop()

st.subheader("Parsed segment")
left, right = st.columns([1, 1.4])
with left:
    st.metric("Category", category_name)
    st.metric("Products found", int(len(raw_products_df)))
with right:
    summary_df = pd.DataFrame(
        [
            {
                "Expected age": market_targets["AgeYears"],
                "Price min": market_targets["PriceMin"],
                "Price max": market_targets["PriceMax"],
                "Performance target": market_targets["PerformanceTarget"],
                "Size target": market_targets["SizeTarget"],
                "Reliability min": market_targets["ReliabilityMin"],
                "Reliability max": market_targets["ReliabilityMax"],
            }
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Current products")
st.dataframe(raw_products_df, use_container_width=True, hide_index=True)

temperature = st.number_input("SoftMax temperature", min_value=0.05, max_value=10.0, value=1.0, step=0.05)
base_prediction = next_step_market_share(model, df_cat, temperature=temperature)

baseline_display = raw_products_df[["Company"]].merge(
    base_prediction[["Company", "NextStepShare"]],
    on="Company",
    how="left",
)
baseline_display["NextStepShare"] = baseline_display["NextStepShare"] * 100.0
st.subheader("Baseline next-step market share")
st.dataframe(baseline_display.rename(columns={"NextStepShare": "Next Step Share %"}), use_container_width=True, hide_index=True)

st.subheader("Optimization setup")
left, right = st.columns([1, 1.35])
with left:
    target_company = st.selectbox("Target product", options=list(raw_products_df["Company"]))
    opponent_drift = st.slider("Opponent drift", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    grid_points = st.slider("Search density", min_value=3, max_value=11, value=7, step=2)
    optimize_features = st.multiselect(
        "Optimize these product fields",
        OPTIMIZE_OPTIONS,
        default=["Price", "Performance", "Reliability", "Acessibility", "Awarness"],
    )

with right:
    st.markdown("**Optimization ranges (raw values)**")
    st.caption("Final search bounds are the intersection of the global min/max and the per-period movement limits for the selected target product.")
    raw_bounds: Dict[str, Tuple[float, float]] = {}
    effective_bounds: Dict[str, Tuple[float, float]] = {}
    current_target_row = raw_products_df.loc[raw_products_df["Company"] == target_company].iloc[0]
    cols = st.columns(2)
    for i, feature in enumerate(OPTIMIZE_OPTIONS):
        default_lo, default_hi = RAW_OPTIMIZATION_DEFAULTS[feature]
        current_val = float(current_target_row[feature])
        lo_seed = min(default_lo, current_val)
        hi_seed = max(default_hi, current_val)
        with cols[i % 2]:
            lo_val = st.number_input(f"{DISPLAY_NAMES.get(feature, feature)} global min", value=float(lo_seed), key=f"raw_{feature}_lo")
            hi_val = st.number_input(f"{DISPLAY_NAMES.get(feature, feature)} global max", value=float(hi_seed), key=f"raw_{feature}_hi")

            if feature == "Age":
                period_lo = max(0.5 * current_val, 0.0)
                period_hi = current_val + 1.0
                st.caption(f"Per-period bound: {period_lo:.3f} to {period_hi:.3f}")
            elif PER_PERIOD_DEFAULTS.get(feature) is not None:
                delta = float(PER_PERIOD_DEFAULTS[feature])
                period_lo = current_val - delta
                period_hi = current_val + delta
                st.caption(f"Per-period bound: {period_lo:.3f} to {period_hi:.3f}")
            else:
                period_lo = float(lo_val)
                period_hi = float(hi_val)
                st.caption("Per-period bound: unrestricted")

            global_lo = float(lo_val)
            global_hi = float(hi_val)
            raw_bounds[feature] = (global_lo, global_hi)
            effective_lo = max(global_lo, period_lo)
            effective_hi = min(global_hi, period_hi)
            effective_bounds[feature] = (effective_lo, effective_hi)
            st.caption(f"Effective search bound: {effective_lo:.3f} to {effective_hi:.3f}")

model_bounds: Dict[str, Tuple[float, float]] = {}
invalid_effective_bounds = []
for feature in optimize_features:
    model_col = RAW_TO_MODEL_COL[feature]
    lo_raw, hi_raw = effective_bounds[feature]
    if lo_raw > hi_raw:
        invalid_effective_bounds.append(feature)
        continue
    lo_model = raw_value_to_model_percent(feature, lo_raw, market_targets)
    hi_model = raw_value_to_model_percent(feature, hi_raw, market_targets)
    model_bounds[model_col] = (min(lo_model, hi_model), max(lo_model, hi_model))

if invalid_effective_bounds:
    st.error(
        "These features have impossible search ranges after applying both global and per-period bounds: "
        + ", ".join(invalid_effective_bounds)
    )
    st.stop()

search_cfg = SearchConfig(
    target_company=target_company,
    optimize_cols=[RAW_TO_MODEL_COL[f] for f in optimize_features],
    bounds=model_bounds,
    grid_points=grid_points,
    opponent_drift=opponent_drift,
    softmax_temperature=temperature,
)

run_opt = st.button("Run optimization", type="primary")
if run_opt:
    best_row, results_df, optimized_competitors = optimize_single_company(model, df_cat, search_cfg)

    st.subheader("Best target settings")
    best_settings = {"Company": target_company, "Predicted Next Step Share %": float(best_row["TargetNextStepShare"]) * 100.0}
    for feature in optimize_features:
        raw_best_value = percent_to_raw_value(
            feature,
            float(best_row[RAW_TO_MODEL_COL[feature]]),
            market_targets,
        )
        best_settings[DISPLAY_NAMES.get(feature, feature)] = raw_best_value
        best_settings[f"{DISPLAY_NAMES.get(feature, feature)} change"] = raw_best_value - float(current_target_row[feature])
    st.dataframe(pd.DataFrame([best_settings]), use_container_width=True, hide_index=True)

    final_rows = []
    base_rows = df_cat.set_index("Company")
    for company in base_rows.index:
        if company == target_company:
            row = {"Company": company, "Categories": CATEGORY_MAP[category_name]}
            for col in NUM_COLS:
                row[col] = float(best_row[col])
        else:
            comp_row = optimized_competitors[company]
            row = {"Company": str(comp_row["Company"]), "Categories": int(comp_row["Categories"])}
            for col in NUM_COLS:
                row[col] = float(comp_row[col])
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows, columns=ALL_COLS)
    final_shares = next_step_market_share(model, final_df, temperature=temperature)

    final_share_display = raw_products_df[["Company"]].merge(
        final_shares[["Company", "NextStepShare"]],
        on="Company",
        how="left",
    )
    final_share_display["NextStepShare"] = final_share_display["NextStepShare"] * 100.0
    st.subheader("Final predicted next-step market share")
    st.dataframe(final_share_display.rename(columns={"NextStepShare": "Next Step Share %"}), use_container_width=True, hide_index=True)

    top_candidates = results_df.head(20).copy()
    top_raw = pd.DataFrame({
        "Company": top_candidates["Company"],
        "Predicted Next Step Share %": top_candidates["TargetNextStepShare"] * 100.0,
    })
    for feature in optimize_features:
        raw_vals = top_candidates[RAW_TO_MODEL_COL[feature]].apply(
            lambda v: percent_to_raw_value(feature, float(v), market_targets)
        )
        top_raw[DISPLAY_NAMES.get(feature, feature)] = raw_vals
        top_raw[f"{DISPLAY_NAMES.get(feature, feature)} change"] = raw_vals - float(current_target_row[feature])
    st.subheader("Top optimization candidates")
    st.dataframe(top_raw, use_container_width=True, hide_index=True)

    st.subheader("3D sensitivity analysis")
    x_feature = st.selectbox("X axis", optimize_features, index=0, key="surface_x")
    y_choices = [f for f in optimize_features if f != x_feature]
    y_feature = st.selectbox("Y axis", y_choices, index=0, key="surface_y")
    z_choice = st.selectbox("Surface output", ["MeanShare", "StdShare", "P10Share", "P50Share", "P90Share"], index=0)

    x_points = st.slider("X points", min_value=10, max_value=40, value=20)
    y_points = st.slider("Y points", min_value=10, max_value=40, value=20)
    mc_passes = st.slider("Monte Carlo passes", min_value=5, max_value=100, value=20)
    noise_std = st.number_input("Competitor noise", min_value=0.0, max_value=15.0, value=2.0, step=0.5)

    if st.button("Run 3D sensitivity surface"):
        noise_std_model = 0.0
        if noise_std > 0:
            noise_std_model = noise_std / 100.0

        surface_df = monte_carlo_surface(
            model=model,
            df_category=df_cat,
            target_company=target_company,
            best_row=best_row,
            x_feature=RAW_TO_MODEL_COL[x_feature],
            y_feature=RAW_TO_MODEL_COL[y_feature],
            x_bounds=model_bounds[RAW_TO_MODEL_COL[x_feature]],
            y_bounds=model_bounds[RAW_TO_MODEL_COL[y_feature]],
            x_points=x_points,
            y_points=y_points,
            noise_std=noise_std_model,
            passes=mc_passes,
            temperature=temperature,
        )

        surface_df_display = surface_df.copy()
        surface_df_display[x_feature] = surface_df_display[RAW_TO_MODEL_COL[x_feature]].apply(
            lambda v: percent_to_raw_value(x_feature, float(v), market_targets)
        )
        surface_df_display[y_feature] = surface_df_display[RAW_TO_MODEL_COL[y_feature]].apply(
            lambda v: percent_to_raw_value(y_feature, float(v), market_targets)
        )
        surface_df_display[z_choice] = surface_df_display[z_choice] * 100.0
        surface_df_display = surface_df_display.drop(columns=[RAW_TO_MODEL_COL[x_feature], RAW_TO_MODEL_COL[y_feature]])

        fig = make_surface_figure(surface_df_display, x_feature=x_feature, y_feature=y_feature, z_col=z_choice)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(surface_df_display, use_container_width=True, hide_index=True)
