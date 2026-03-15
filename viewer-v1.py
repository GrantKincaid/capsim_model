from dataclasses import dataclass
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

DEFAULT_BOUNDS = {
    "Price%": (-0.40, 0.40),
    "Age%": (0.0, 2.0),
    "Performance%": (-0.50, 0.50),
    "Size%": (-0.50, 0.50),
    "Reliability%": (-0.50, 0.50),
    "Acessibility%": (0.0, 1.0),
    "Awarness%": (0.0, 1.0),
}

OPTIMIZE_COLS_DEFAULT = [
    "Price%",
    "Performance%",
    "Reliability%",
    "Acessibility%",
    "Awarness%",
]


# ============================================================
# MODEL HELPERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_keras_model(uploaded_file) -> tf.keras.Model:
    """
    Loads a Keras model from an uploaded .keras or .h5 file.
    Caching avoids reloading on every rerun.
    """
    import tempfile
    import os

    suffix = ".keras"
    if uploaded_file.name.lower().endswith(".h5"):
        suffix = ".h5"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        model = tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return model


def dataframe_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = df.copy()
    clean["Company"] = clean["Company"].astype(str)
    clean["Categories"] = pd.to_numeric(clean["Categories"], errors="coerce")
    for col in NUM_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean[["Categories"] + NUM_COLS].isna().any().any():
        bad_rows = clean.loc[clean[["Categories"] + NUM_COLS].isna().any(axis=1)]
        raise ValueError(
            "Non-finite values found in scenario table before inference. "
            f"Problem rows: {bad_rows.to_string(index=False)}"
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


# ============================================================
# OPTIMIZATION LOGIC
# ============================================================
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
    if search_cfg.target_company not in set(df_category["Company"]):
        raise ValueError(f"Target company '{search_cfg.target_company}' not found in current category rows.")

    companies, categories, base_numeric = dataframe_to_arrays(df_category)
    n_companies, n_features = base_numeric.shape
    company_to_idx = {c: i for i, c in enumerate(companies)}
    target_idx = company_to_idx[search_cfg.target_company]

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
        if company == search_cfg.target_company:
            continue
        fixed_numeric[company_idx] = optimized_competitors[company][NUM_COLS].values.astype(np.float32)

    n_target_candidates = target_candidates.shape[0]
    scenario_numeric = np.repeat(fixed_numeric.reshape(1, n_companies, n_features), n_target_candidates, axis=0)
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
    best_row = results_df.iloc[0]
    return best_row, results_df, optimized_competitors


# ============================================================
# SENSITIVITY / MONTE CARLO SURFACE
# ============================================================
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
            noise = np.random.normal(
                0.0,
                noise_std,
                size=(chunk_size, len(competitor_indices), n_features),
            ).astype(np.float32)
            scenario_numeric[:, competitor_indices, :] += noise
            for feat_i, col in enumerate(NUM_COLS):
                lo, hi = DEFAULT_BOUNDS[col]
                scenario_numeric[:, competitor_indices, feat_i] = np.clip(
                    scenario_numeric[:, competitor_indices, feat_i], lo, hi
                )

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
    out = pd.DataFrame(
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
    return out


def make_surface_figure(surface_df: pd.DataFrame, x_feature: str, y_feature: str, z_col: str) -> go.Figure:
    x_vals = np.sort(surface_df[x_feature].unique())
    y_vals = np.sort(surface_df[y_feature].unique())

    z_pivot = (
        surface_df.pivot_table(index=y_feature, columns=x_feature, values=z_col, aggfunc="mean")
        .reindex(index=y_vals, columns=x_vals)
    )
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
            hovertemplate=(
                f"{x_feature}: %{{x:.4f}}<br>"
                f"{y_feature}: %{{y:.4f}}<br>"
                f"{z_col}: %{{z:.4f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"3D Sensitivity Surface: {z_col}",
        scene=dict(
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            zaxis_title=z_col,
            camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
        ),
        height=720,
        margin=dict(l=10, r=10, b=10, t=60),
    )
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Category Optimization", layout="wide")
st.title("Category Optimization and Sensitivity Analysis")
st.caption(
    "Upload a trained Keras model and current category-state rows, then optimize one target company against competitors."
)

with st.sidebar:
    st.header("1) Inputs")
    model_file = st.file_uploader("Upload Keras model (.keras or .h5)", type=["keras", "h5"])
    data_file = st.file_uploader("Upload category-state CSV", type=["csv"])

    st.markdown("CSV must include these columns:")
    st.code(", ".join(ALL_COLS), language="text")

if model_file is None or data_file is None:
    st.info("Upload both the Keras model and the current category-state CSV to begin.")
    st.stop()

model = load_keras_model(model_file)
df = pd.read_csv(data_file)

missing = [c for c in ALL_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean / cast
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Categories"] = pd.to_numeric(df["Categories"], errors="coerce").astype("Int64")
df = df.dropna(subset=["Company", "Categories"] + NUM_COLS).copy()
df["Categories"] = df["Categories"].astype(np.int32)

# One category at a time
st.subheader("Current Input State")
col_a, col_b, col_c = st.columns([1.2, 1.0, 1.0])
with col_a:
    category_name = st.selectbox("Product category", list(CATEGORY_MAP.keys()), index=0)
with col_b:
    category_code = CATEGORY_MAP[category_name]
    st.metric("Category code", category_code)
with col_c:
    temperature = st.number_input("SoftMax temperature", min_value=0.05, max_value=10.0, value=1.0, step=0.05)

df_cat = df.loc[df["Categories"] == category_code, ALL_COLS].copy().reset_index(drop=True)

if df_cat.empty:
    st.warning("No rows found for the selected category.")
    st.stop()

if df_cat["Company"].duplicated().any():
    st.error("Each company may only have one product per category. Duplicate company rows were found in this category.")
    st.stop()

st.dataframe(df_cat, use_container_width=True)

base_prediction = next_step_market_share(model, df_cat, temperature=temperature)
st.subheader("Baseline Next-Step Market Share")
st.dataframe(base_prediction[["Company", "RawScore", "NextStepShare"]], use_container_width=True)


# ------------------------------------------------------------
# Optimization controls
# ------------------------------------------------------------
st.subheader("Optimization Setup")
left, right = st.columns([1.0, 1.4])

with left:
    target_company = st.selectbox("Target company", options=list(df_cat["Company"]))
    opponent_drift = st.slider(
        "Opponent drift scalar",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0 = competitors unchanged, 1 = competitors move to their local optimum from the same search space.",
    )
    grid_points = st.slider("Grid points per optimized feature", min_value=3, max_value=15, value=7, step=2)
    optimize_cols = st.multiselect(
        "Features to optimize",
        options=NUM_COLS,
        default=OPTIMIZE_COLS_DEFAULT,
    )

with right:
    st.markdown("**Optimization bounds**")
    bounds = {}
    bounds_cols = st.columns(2)
    for i, col in enumerate(NUM_COLS):
        with bounds_cols[i % 2]:
            default_lo, default_hi = DEFAULT_BOUNDS[col]
            lo = st.number_input(f"{col} min", value=float(default_lo), key=f"{col}_lo")
            hi = st.number_input(f"{col} max", value=float(default_hi), key=f"{col}_hi")
            if lo >= hi:
                st.error(f"Bounds invalid for {col}: min must be less than max.")
                st.stop()
            bounds[col] = (float(lo), float(hi))

search_cfg = SearchConfig(
    target_company=target_company,
    optimize_cols=optimize_cols,
    bounds=bounds,
    grid_points=grid_points,
    opponent_drift=opponent_drift,
    softmax_temperature=temperature,
)

run_opt = st.button("Run optimization", type="primary")

if run_opt:
    if not optimize_cols:
        st.error("Select at least one feature to optimize.")
        st.stop()

    total_candidates = grid_points ** len(optimize_cols)
    if total_candidates > 150_000:
        st.warning(
            f"Search space is {total_candidates:,} candidates. That may get sludgy. Reduce features or grid points if needed."
        )

    with st.spinner("Optimizing target company product against category competition..."):
        best_row, results_df, optimized_competitors = optimize_single_company(model, df_cat, search_cfg)

    st.success("Optimization complete.")

    st.subheader("Best Target Product Configuration")
    best_display_cols = ["Company", "Categories"] + optimize_cols + ["TargetNextStepShare"]
    st.dataframe(pd.DataFrame([best_row])[best_display_cols], use_container_width=True)

    # Rebuild final table using best target and the exact drifted competitors used during optimization
    base_rows = df_cat.set_index("Company")
    final_rows = []
    for company in base_rows.index:
        if company == target_company:
            row = {
                "Company": company,
                "Categories": int(base_rows.loc[company, "Categories"]),
            }
            for col in NUM_COLS:
                row[col] = float(best_row[col])
        else:
            comp_row = optimized_competitors[company]
            row = {
                "Company": str(comp_row["Company"]),
                "Categories": int(comp_row["Categories"]),
            }
            for col in NUM_COLS:
                row[col] = float(comp_row[col])
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows, columns=ALL_COLS)
    final_shares = next_step_market_share(model, final_df, temperature=temperature)

    st.subheader("Final Predicted Next-Step Market Share")
    st.dataframe(final_shares[["Company", "RawScore", "NextStepShare"]], use_container_width=True)

    st.subheader("Top Candidate Configurations")
    top_n = min(25, len(results_df))
    st.dataframe(results_df.head(top_n), use_container_width=True)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download optimization results CSV",
        data=csv_bytes,
        file_name=f"optimization_results_category_{category_code}_{target_company}.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # Sensitivity analysis
    # --------------------------------------------------------
    st.subheader("3D Sensitivity Analysis")
    s1, s2, s3 = st.columns(3)
    with s1:
        x_feature = st.selectbox("X-axis feature", options=NUM_COLS, index=0)
    with s2:
        y_feature = st.selectbox("Y-axis feature", options=[c for c in NUM_COLS if c != x_feature], index=1)
    with s3:
        z_choice = st.selectbox("Z output", options=["MeanShare", "StdShare", "P10Share", "P50Share", "P90Share"], index=0)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        x_points = st.slider("X grid points", min_value=10, max_value=60, value=25)
    with p2:
        y_points = st.slider("Y grid points", min_value=10, max_value=60, value=25)
    with p3:
        mc_passes = st.slider("Monte Carlo passes per grid point", min_value=5, max_value=200, value=30)
    with p4:
        noise_std = st.number_input("Competitor noise std", min_value=0.0, max_value=0.50, value=0.02, step=0.005)

    run_surface = st.button("Run 3D sensitivity surface")

    if run_surface:
        with st.spinner("Running Monte Carlo sensitivity surface..."):
            surface_df = monte_carlo_surface(
                model=model,
                df_category=df_cat,
                target_company=target_company,
                best_row=best_row,
                x_feature=x_feature,
                y_feature=y_feature,
                x_bounds=bounds[x_feature],
                y_bounds=bounds[y_feature],
                x_points=x_points,
                y_points=y_points,
                noise_std=noise_std,
                passes=mc_passes,
                temperature=temperature,
            )

        fig = make_surface_figure(surface_df, x_feature=x_feature, y_feature=y_feature, z_col=z_choice)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(surface_df, use_container_width=True)

        surface_csv = surface_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download sensitivity surface CSV",
            data=surface_csv,
            file_name=f"surface_category_{category_code}_{target_company}_{x_feature}_{y_feature}.csv",
            mime="text/csv",
        )


st.markdown("---")
st.markdown(
    "**Notes**  \n"
    "- This page assumes the Keras model takes inputs named `numeric_features` and `category`, matching the earlier regression setup.  \n"
    "- SoftMax is applied across all companies in the selected category to estimate next-step share.  \n"
    "- Opponent drift uses the same search space and locally optimizes each competitor before blending by the drift scalar.  \n"
    "- The 3D sensitivity surface perturbs competitors with Gaussian noise and reports a distribution of next-step share outcomes for the target company."
)
