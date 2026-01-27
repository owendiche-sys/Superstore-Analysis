from __future__ import annotations

import io
import os
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# App config
# =========================
st.set_page_config(
    page_title="Superstore Performance Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_FILENAME = "superstore_dataset_final.csv"
FALLBACK_DATA_FILENAME = "data.csv"

COL_ORDER_DATE = "Order Date"
COL_SHIP_DATE = "Ship Date"
COL_SALES = "Sales"
COL_PROFIT = "Profit"
COL_DISCOUNT = "Discount"
COL_QUANTITY = "Quantity"
COL_REGION = "Region"
COL_CATEGORY = "Category"
COL_SUBCATEGORY = "Sub-Category"
COL_SEGMENT = "Segment"
COL_SHIPMODE = "Ship Mode"
COL_STATE = "State"
COL_CITY = "City"


# =========================
# Styling (Light theme + cards)
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root{
            --bg:#ffffff;
            --panel:#ffffff;
            --text:#111827;
            --muted:#6b7280;
            --border:rgba(17,24,39,0.12);
            --shadow:0 10px 25px rgba(17,24,39,0.06);
            --shadow2:0 6px 18px rgba(17,24,39,0.05);
            --radius:18px;
          }

          html, body, [data-testid="stAppViewContainer"]{
            background: var(--bg) !important;
            color: var(--text) !important;
          }

          .block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; }

          [data-testid="stSidebar"]{
            background: #fbfbfd !important;
            border-right: 1px solid var(--border);
          }

          .card{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow2);
            padding: 16px 16px;
          }
          .card h3{
            margin: 0 0 8px 0;
            font-size: 16px;
            font-weight: 700;
            color: var(--text);
          }
          .subtle{
            color: var(--muted);
            font-size: 13px;
            margin: 0;
          }

          .kpi-wrap{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
          }
          @media (max-width: 1100px){
            .kpi-wrap{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 640px){
            .kpi-wrap{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
          }

          .kpi{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 14px 14px;
          }
          .kpi .label{
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 6px;
          }
          .kpi .value{
            font-size: 22px;
            font-weight: 800;
            color: var(--text);
            line-height: 1.2;
          }
          .kpi .delta{
            font-size: 12px;
            color: var(--muted);
            margin-top: 6px;
          }

          .section-title{
            font-size: 18px;
            font-weight: 800;
            margin: 0 0 10px 0;
            color: var(--text);
          }
          .divider{
            height: 1px;
            background: var(--border);
            margin: 10px 0 16px 0;
          }

          .js-plotly-plot .plotly .modebar{ opacity: 0.15; }
          .js-plotly-plot .plotly:hover .modebar{ opacity: 1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================
# Utilities
# =========================
def _safe_read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings. Last error: {last_err}")


def _safe_read_csv_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    return _safe_read_csv_from_bytes(data)


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace("$", "", regex=False).str.replace("£", "", regex=False)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def _df_fingerprint(df: pd.DataFrame) -> str:
    shape = f"{df.shape[0]}x{df.shape[1]}"
    sample = df.head(200).copy()
    h = pd.util.hash_pandas_object(sample, index=True).sum()
    return f"{shape}-{int(h)}"


def _format_currency(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"${x:,.0f}"


# =========================
# Data load + cleaning
# =========================
@st.cache_data(show_spinner=False)
def load_data(use_upload: bool, uploaded_file_bytes: Optional[bytes]) -> Tuple[pd.DataFrame, str]:
    if use_upload and uploaded_file_bytes is not None:
        df = _safe_read_csv_from_bytes(uploaded_file_bytes)
        source = "Uploaded CSV"
    else:
        if os.path.exists(DEFAULT_DATA_FILENAME):
            df = _safe_read_csv_from_path(DEFAULT_DATA_FILENAME)
            source = DEFAULT_DATA_FILENAME
        elif os.path.exists(FALLBACK_DATA_FILENAME):
            df = _safe_read_csv_from_path(FALLBACK_DATA_FILENAME)
            source = FALLBACK_DATA_FILENAME
        else:
            raise FileNotFoundError(
                f"Default dataset not found. Place '{DEFAULT_DATA_FILENAME}' (preferred) "
                f"or '{FALLBACK_DATA_FILENAME}' next to app.py, or upload a CSV in the sidebar."
            )
    return df, source


@st.cache_data(show_spinner=False)
def clean_superstore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    for c in [COL_ORDER_DATE, COL_SHIP_DATE]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", infer_datetime_format=True)

    for c in [COL_SALES, COL_PROFIT, COL_DISCOUNT, COL_QUANTITY]:
        if c in out.columns:
            out[c] = _to_numeric(out[c])

    out = out.drop_duplicates()

    if COL_ORDER_DATE in out.columns:
        out["Order Month"] = out[COL_ORDER_DATE].dt.to_period("M").dt.to_timestamp()

    if _has_cols(out, [COL_SALES, COL_PROFIT]):
        out["Profit Margin"] = np.where(out[COL_SALES].abs() > 0, out[COL_PROFIT] / out[COL_SALES], np.nan)

    if COL_DISCOUNT in out.columns:
        out["Discount Bin"] = pd.cut(
            out[COL_DISCOUNT],
            bins=[-0.01, 0.10, 0.20, 0.30, 0.40, 1.00],
            labels=["0–10%", "10–20%", "20–30%", "30–40%", "40%+"],
            include_lowest=True,
        )

    return out


@st.cache_data(show_spinner=False)
def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    if "Order Month" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("Order Month", dropna=False).agg(
        Sales=(COL_SALES, "sum") if COL_SALES in df.columns else ("Order Month", "size"),
        Profit=(COL_PROFIT, "sum") if COL_PROFIT in df.columns else ("Order Month", "size"),
        Orders=("Order Month", "size"),
    )
    return agg.reset_index().sort_values("Order Month")


@st.cache_data(show_spinner=False)
def group_sum(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df.columns:
        return pd.DataFrame()
    metrics = [c for c in [COL_SALES, COL_PROFIT] if c in df.columns]
    if not metrics:
        return df.groupby(by).size().reset_index(name="Count").sort_values("Count", ascending=False)
    return df.groupby(by)[metrics].sum().reset_index()


# =========================
# Modeling
# =========================
@dataclass
class ModelResult:
    mode: str
    model: Pipeline
    feature_names: List[str]
    metrics: Dict[str, float]
    threshold: Optional[float] = None
    confusion: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    importances: Optional[pd.DataFrame] = None


def _build_preprocess(df: pd.DataFrame, target_col: str, drop_cols: Optional[List[str]] = None) -> ColumnTransformer:
    drop_cols = drop_cols or []
    X = df.drop(columns=[target_col], errors="ignore")

    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _get_feature_names(pre: ColumnTransformer) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def train_model_cached(df: pd.DataFrame, fingerprint: str, target_col: str, mode: str, seed: int) -> ModelResult:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    work = df.copy()
    work = work[~work[target_col].isna()].copy()
    if work.shape[0] < 200:
        raise ValueError("Not enough rows after removing missing target values (need at least 200).")

    drop_cols: List[str] = []
    for c in work.columns:
        if c == target_col:
            continue
        if work[c].dtype == "object" and work[c].nunique(dropna=True) > 2000:
            drop_cols.append(c)

    pre = _build_preprocess(work, target_col=target_col, drop_cols=drop_cols)
    X = work.drop(columns=[target_col], errors="ignore")
    y = work[target_col].copy()

    if mode == "regression":
        y = _to_numeric(y)
        valid = ~y.isna()
        X = X.loc[valid].copy()
        y = y.loc[valid].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

        rf = RandomForestRegressor(
            n_estimators=250,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        pipe = Pipeline(steps=[("pre", pre), ("model", rf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "R2": float(r2_score(y_test, y_pred)),
        }

        feat_names = _get_feature_names(pipe.named_steps["pre"])
        imp = pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)

        return ModelResult(
            mode="regression",
            model=pipe,
            feature_names=feat_names,
            metrics=metrics,
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            importances=imp_df,
        )

    # classification
    y_vals = y.copy()
    if pd.api.types.is_numeric_dtype(y_vals):
        uniq = pd.Series(y_vals.dropna().unique())
        if uniq.nunique() > 2:
            y_vals = (y_vals > 0).astype(int)
        else:
            y_vals = y_vals.astype(int)
    else:
        top = y_vals.astype(str).value_counts(dropna=True).index[0]
        y_vals = (y_vals.astype(str) == top).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vals, test_size=0.25, random_state=seed, stratify=y_vals
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", rf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    thr = 0.5
    y_pred = (proba >= thr).astype(int)

    metrics = {
        "ROC AUC": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred)
    feat_names = _get_feature_names(pipe.named_steps["pre"])
    imp = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)

    return ModelResult(
        mode="classification",
        model=pipe,
        feature_names=feat_names,
        metrics=metrics,
        threshold=thr,
        confusion=cm,
        proba=proba,
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        importances=imp_df,
    )


# =========================
# UI helpers
# =========================
def card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <h3>{escape(title)}</h3>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


from html import escape

def kpi_cards(items):
    blocks = []
    for label, value, delta in items:
        label_e = escape(str(label))
        value_e = escape(str(value))
        delta_e = escape(str(delta)) if delta else ""

        blocks.append(
            "<div class='kpi'>"
            f"<div class='label'>{label_e}</div>"
            f"<div class='value'>{value_e}</div>"
            f"<div class='delta'>{delta_e if delta_e else '&nbsp;'}</div>"
            "</div>"
        )

    html = "<div class='kpi-wrap'>" + "".join(blocks) + "</div>"
    st.markdown(html.strip(), unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{escape(text)}</div>", unsafe_allow_html=True)


def divider() -> None:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# =========================
# Sidebar: Navigation -> Filters -> Data
# =========================
pages = [
    "Summary",
    "Data Explorer",
    "Sales and Profit",
    "Discount and Quantity",
    "Modeling",
    "Insights",
]

# Create sidebar sections in the required order
nav_box = st.sidebar.container()
filters_box = st.sidebar.container()
data_box = st.sidebar.container()

with nav_box:
    st.markdown("### Navigation")
    page = st.radio("Go to", pages, index=0, label_visibility="collapsed")

with filters_box:
    st.markdown("### Filters")

# Data controls must exist before loading data
with data_box:
    st.markdown("### Data")
    use_upload = st.toggle("Upload CSV instead of default file", value=False)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], disabled=not use_upload)

try:
    df_raw, source_label = load_data(use_upload, uploaded.getvalue() if uploaded is not None else None)
except Exception as e:
    st.error(str(e))
    st.stop()

df = clean_superstore(df_raw)

# Now fill the filters section (after data is available, but still rendered above data in sidebar)
with filters_box:
    date_min = None
    date_max = None
    if COL_ORDER_DATE in df.columns:
        valid_dates = df[COL_ORDER_DATE].dropna()
        if not valid_dates.empty:
            date_min = valid_dates.min().date()
            date_max = valid_dates.max().date()

    if date_min and date_max:
        dr = st.date_input("Order date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
        if isinstance(dr, tuple) and len(dr) == 2:
            d0, d1 = dr
        else:
            d0, d1 = date_min, date_max
    else:
        d0, d1 = None, None

    def _multi_filter(label: str, col: str) -> Optional[List[str]]:
        if col not in df.columns:
            return None
        opts = sorted([x for x in df[col].dropna().unique().tolist()])
        if not opts:
            return None
        return st.multiselect(label, options=opts, default=[])

    f_region = _multi_filter("Region", COL_REGION)
    f_category = _multi_filter("Category", COL_CATEGORY)
    f_segment = _multi_filter("Segment", COL_SEGMENT)
    f_shipmode = _multi_filter("Ship mode", COL_SHIPMODE)

# Update Data section info (still at bottom)
with data_box:
    st.caption(f"Source: {source_label}")
    st.caption(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

# Apply filters
filtered = df.copy()

if d0 and d1 and COL_ORDER_DATE in filtered.columns:
    filtered = filtered[(filtered[COL_ORDER_DATE].dt.date >= d0) & (filtered[COL_ORDER_DATE].dt.date <= d1)]

def _apply_in_filter(data: pd.DataFrame, col: str, values: Optional[List[str]]) -> pd.DataFrame:
    if values is None or len(values) == 0 or col not in data.columns:
        return data
    return data[data[col].isin(values)]

filtered = _apply_in_filter(filtered, COL_REGION, f_region)
filtered = _apply_in_filter(filtered, COL_CATEGORY, f_category)
filtered = _apply_in_filter(filtered, COL_SEGMENT, f_segment)
filtered = _apply_in_filter(filtered, COL_SHIPMODE, f_shipmode)


# =========================
# Header
# =========================
st.title("Superstore Performance Dashboard")
st.caption("Sales, profit, discount impact, and optional predictive modeling based on the active filters.")


# =========================
# Pages
# =========================
if page == "Summary":
    section_title("Overview")
    divider()

    total_orders = filtered.shape[0]
    total_sales = float(filtered[COL_SALES].sum()) if COL_SALES in filtered.columns else float("nan")
    total_profit = float(filtered[COL_PROFIT].sum()) if COL_PROFIT in filtered.columns else float("nan")

    profit_margin = (
        float(total_profit / total_sales)
        if (COL_SALES in filtered.columns and total_sales != 0 and not np.isnan(total_sales))
        else float("nan")
    )

    # KPI cards (only on Summary)
    kpis = [
        ("Total orders", f"{total_orders:,}", "Filtered rows"),
        ("Total sales", _format_currency(total_sales) if not np.isnan(total_sales) else "N/A", "Sum of sales"),
        ("Total profit", _format_currency(total_profit) if not np.isnan(total_profit) else "N/A", "Sum of profit"),
        ("Profit margin", f"{profit_margin*100:,.1f}%" if not np.isnan(profit_margin) else "N/A", "Profit divided by sales"),
    ]
    kpi_cards(kpis)

    st.write("")

    col1, col2 = st.columns([1.3, 1.0], gap="large")

    with col1:
        card("Monthly trend", "<p class='subtle'>Monthly sales and profit over the filtered time window.</p>")
        m = monthly_agg(filtered)
        if not m.empty and "Order Month" in m.columns:
            fig = go.Figure()
            if COL_SALES in filtered.columns:
                fig.add_trace(go.Scatter(x=m["Order Month"], y=m["Sales"], mode="lines", name="Sales"))
            if COL_PROFIT in filtered.columns:
                fig.add_trace(go.Scatter(x=m["Order Month"], y=m["Profit"], mode="lines", name="Profit"))
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title="Month",
                yaxis_title="Amount",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly trend requires a valid Order Date column and at least one numeric metric.")

    with col2:
        card("Quick stats", "<p class='subtle'>High-level descriptive stats for core numeric fields.</p>")
        stats_cols = [c for c in [COL_SALES, COL_PROFIT, COL_DISCOUNT, COL_QUANTITY] if c in filtered.columns]
        if stats_cols:
            desc = filtered[stats_cols].describe().T
            desc = desc[["count", "mean", "std", "min", "50%", "max"]]
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("No core numeric fields were found to summarize.")

    st.write("")
    section_title("Top breakdowns")
    divider()

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        card("Sales by region", "<p class='subtle'>Total sales aggregated by region.</p>")
        if COL_REGION in filtered.columns and COL_SALES in filtered.columns:
            g = group_sum(filtered, COL_REGION)
            fig = px.bar(g, x=COL_REGION, y=COL_SALES, title=None)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Region", yaxis_title="Sales")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Requires Region and Sales columns.")

    with c2:
        card("Profit by category", "<p class='subtle'>Total profit aggregated by category.</p>")
        if COL_CATEGORY in filtered.columns and COL_PROFIT in filtered.columns:
            g = group_sum(filtered, COL_CATEGORY)
            fig = px.bar(g, x=COL_CATEGORY, y=COL_PROFIT, title=None)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Category", yaxis_title="Profit")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Requires Category and Profit columns.")

    with c3:
        card("Profit by sub-category", "<p class='subtle'>Top sub-categories by total profit.</p>")
        if COL_SUBCATEGORY in filtered.columns and COL_PROFIT in filtered.columns:
            g = group_sum(filtered, COL_SUBCATEGORY).sort_values(COL_PROFIT, ascending=False).head(15)
            fig = px.bar(g, x=COL_SUBCATEGORY, y=COL_PROFIT, title=None)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Sub-category", yaxis_title="Profit")
            fig.update_xaxes(tickangle=35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Requires Sub-Category and Profit columns.")

elif page == "Data Explorer":
    section_title("Dataset preview")
    divider()

    c1, c2 = st.columns([1.2, 0.8], gap="large")

    with c1:
        card("Filtered data", "<p class='subtle'>Preview of records after applying sidebar filters.</p>")
        preview_rows = st.slider("Rows to display", min_value=10, max_value=200, value=50, step=10)
        st.dataframe(filtered.head(preview_rows), use_container_width=True)

    with c2:
        card("Data quality checks", "<p class='subtle'>Missingness and basic column diagnostics.</p>")
        missing = filtered.isna().mean().sort_values(ascending=False)
        miss_df = pd.DataFrame({"missing_rate": (missing * 100).round(2)})
        st.dataframe(miss_df.head(25), use_container_width=True)

elif page == "Sales and Profit":
    section_title("Sales and profit performance")
    divider()

    c1, c2 = st.columns([1.15, 0.85], gap="large")

    with c1:
        card("Monthly sales and profit", "<p class='subtle'>Trend aggregated by month.</p>")
        m = monthly_agg(filtered)
        if not m.empty and "Order Month" in m.columns:
            fig = go.Figure()
            if COL_SALES in filtered.columns:
                fig.add_trace(go.Scatter(x=m["Order Month"], y=m["Sales"], mode="lines+markers", name="Sales"))
            if COL_PROFIT in filtered.columns:
                fig.add_trace(go.Scatter(x=m["Order Month"], y=m["Profit"], mode="lines+markers", name="Profit"))
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Month",
                yaxis_title="Amount",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly trend requires Order Date and at least one of Sales or Profit.")

    with c2:
        card("Breakdown selector", "<p class='subtle'>Compare totals by a chosen dimension.</p>")
        dims = [c for c in [COL_REGION, COL_CATEGORY, COL_SUBCATEGORY, COL_SEGMENT, COL_SHIPMODE, COL_STATE, COL_CITY] if c in filtered.columns]
        if dims and (COL_SALES in filtered.columns or COL_PROFIT in filtered.columns):
            dim = st.selectbox("Dimension", options=dims, index=0)
            metric_opts = [c for c in [COL_SALES, COL_PROFIT] if c in filtered.columns]
            metric = st.selectbox("Metric", options=metric_opts, index=0)
            top_n = st.slider("Top N", min_value=5, max_value=25, value=10, step=1)

            g = group_sum(filtered, dim).sort_values(metric, ascending=False).head(top_n)
            fig = px.bar(g, x=dim, y=metric, title=None)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title=dim, yaxis_title=metric)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This view needs at least one categorical dimension and Sales or Profit metrics.")

elif page == "Discount and Quantity":
    section_title("Discount and quantity relationships")
    divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        card("Discount vs profit", "<p class='subtle'>Relationship between discount level and profit.</p>")
        if _has_cols(filtered, [COL_DISCOUNT, COL_PROFIT]):
            fig = px.scatter(filtered, x=COL_DISCOUNT, y=COL_PROFIT, title=None, opacity=0.6)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Discount", yaxis_title="Profit")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Requires Discount and Profit columns.")

    with c2:
        card("Average profit by discount band", "<p class='subtle'>Mean profit across discount ranges.</p>")
        if "Discount Bin" in filtered.columns and COL_PROFIT in filtered.columns:
            g = filtered.groupby("Discount Bin", dropna=False)[COL_PROFIT].mean().reset_index()
            fig = px.bar(g, x="Discount Bin", y=COL_PROFIT, title=None)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Discount band", yaxis_title="Average profit")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Requires Discount and Profit columns.")

elif page == "Modeling":
    section_title("Predictive modeling")
    divider()

    card(
        "Model scope",
        """
        <p class="subtle">
          This section trains a model on the filtered dataset.
          It supports (1) Profit regression and (2) Profitability classification.
        </p>
        """,
    )

    if filtered.shape[0] < 300:
        st.warning("Modeling works best with at least 300 rows. Consider widening filters.")
        st.stop()

    if COL_PROFIT not in filtered.columns:
        st.info("Modeling requires a Profit column in the dataset.")
        st.stop()

    mode_label = st.selectbox("Model type", options=["Profit regression", "Profitability classification"], index=0)
    seed = st.number_input("Random seed", min_value=1, max_value=10_000, value=42, step=1)

    target_col = COL_PROFIT
    mode = "regression" if mode_label == "Profit regression" else "classification"

    fingerprint = _df_fingerprint(filtered)
    try:
        res = train_model_cached(filtered, fingerprint, target_col=target_col, mode=mode, seed=int(seed))
    except Exception as e:
        st.error(str(e))
        st.stop()

    c1, c2 = st.columns([0.9, 1.1], gap="large")

    with c1:
        card("Model metrics", "<p class='subtle'>Performance measured on a holdout split.</p>")
        met = pd.DataFrame({"metric": list(res.metrics.keys()), "value": list(res.metrics.values())})
        st.dataframe(met, use_container_width=True)

        if res.mode == "classification" and res.proba is not None and res.y_true is not None:
            st.write("")
            card("Threshold control", "<p class='subtle'>Adjust the probability threshold and review tradeoffs.</p>")
            thr = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)

            y_pred = (res.proba >= thr).astype(int)
            y_true = res.y_true

            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            kpi_cards(
                [
                    ("Accuracy", f"{acc:.3f}", ""),
                    ("Precision", f"{prec:.3f}", ""),
                    ("Recall", f"{rec:.3f}", ""),
                    ("F1", f"{f1:.3f}", ""),
                ]
            )

            st.write("")
            st.caption("Confusion matrix (rows: actual, columns: predicted)")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)

    with c2:
        card("Feature importance", "<p class='subtle'>Top drivers based on model importance.</p>")
        if res.importances is not None and not res.importances.empty:
            top = res.importances.head(20).copy()
            fig = px.bar(top[::-1], x="importance", y="feature", orientation="h", title=None)
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for the current model configuration.")

        if res.mode == "classification" and res.proba is not None:
            st.write("")
            card("Probability bands", "<p class='subtle'>Distribution of predicted probabilities for the positive class.</p>")
            bins = pd.cut(
                pd.Series(res.proba),
                bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
                include_lowest=True,
            )
            band = bins.value_counts().sort_index()
            band_df = pd.DataFrame({"band": band.index.astype(str), "count": band.values})
            fig = px.bar(band_df, x="band", y="count", title=None)
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Probability band", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Insights":
    section_title("Insights")
    divider()

    card(
        "Data-driven insights",
        "<p class='subtle'>Observed patterns and descriptive statistics based on the active filters.</p>",
    )

    bullets: List[str] = []

    if COL_SALES in filtered.columns:
        bullets.append(f"Total sales is {_format_currency(float(filtered[COL_SALES].sum()))} across {filtered.shape[0]:,} orders.")
        bullets.append(f"Median sales per order is {_format_currency(float(filtered[COL_SALES].median(skipna=True)))}.")

    if COL_PROFIT in filtered.columns:
        total_profit = float(filtered[COL_PROFIT].sum())
        bullets.append(f"Total profit is {_format_currency(total_profit)}.")
        bullets.append(f"Median profit per order is {_format_currency(float(filtered[COL_PROFIT].median(skipna=True)))}.")

        if "Profit Margin" in filtered.columns:
            pm = filtered["Profit Margin"].replace([np.inf, -np.inf], np.nan).dropna()
            if not pm.empty:
                bullets.append(f"Average profit margin is {pm.mean()*100:,.1f}% with a median of {pm.median()*100:,.1f}%.")

    if _has_cols(filtered, [COL_REGION, COL_PROFIT]):
        g = filtered.groupby(COL_REGION)[COL_PROFIT].sum().sort_values(ascending=False)
        if g.shape[0] >= 2:
            bullets.append(f"Region performance differs materially: {g.index[0]} leads in total profit, while {g.index[-1]} is lowest within the selected window.")

    if _has_cols(filtered, [COL_CATEGORY, COL_SALES]):
        g = filtered.groupby(COL_CATEGORY)[COL_SALES].sum().sort_values(ascending=False)
        share = float(g.iloc[0] / g.sum()) * 100 if g.sum() != 0 else np.nan
        if not np.isnan(share):
            bullets.append(f"Category concentration is notable: {g.index[0]} contributes {share:,.1f}% of sales among available categories.")

    if _has_cols(filtered, [COL_DISCOUNT, COL_PROFIT]):
        corr = filtered[[COL_DISCOUNT, COL_PROFIT]].corr(numeric_only=True).iloc[0, 1]
        if not np.isnan(corr):
            bullets.append(f"The correlation between discount and profit is {corr:,.2f}, indicating how discounting aligns with profitability in the selected data.")

    m = monthly_agg(filtered)
    if not m.empty and COL_SALES in filtered.columns and m.shape[0] >= 3:
        y = m["Sales"].to_numpy()
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        direction = "increasing" if slope > 0 else "decreasing"
        bullets.append(f"Monthly sales shows an overall {direction} direction over the selected time period.")

    if bullets:
        for b in bullets:
            st.write(f"- {b}")
    else:
        st.info("Not enough structured fields were found to generate data-driven insights for the current dataset.")

    st.write("")
    divider()

    card(
        "Model-driven insights",
        "<p class='subtle'>Feature importance, probability bands, and threshold tradeoffs derived from an optional model.</p>",
    )

    if COL_PROFIT not in filtered.columns or filtered.shape[0] < 300:
        st.info("Model-driven insights are available when a Profit column exists and at least 300 rows are selected.")
    else:
        fingerprint = _df_fingerprint(filtered)
        try:
            res = train_model_cached(filtered, fingerprint, target_col=COL_PROFIT, mode="classification", seed=42)

            notes: List[str] = []
            if res.importances is not None and not res.importances.empty:
                top_feats = ", ".join(res.importances.head(8)["feature"].astype(str).tolist())
                notes.append(f"Top model drivers include: {top_feats}.")

            if res.proba is not None and res.y_true is not None:
                proba = res.proba
                high_conf = float((proba >= 0.8).mean() * 100)
                low_conf = float((proba <= 0.2).mean() * 100)
                notes.append(f"{high_conf:,.1f}% of predictions fall in the high-confidence band (0.8–1.0), while {low_conf:,.1f}% are in the low-confidence band (0.0–0.2).")

                for thr in [0.40, 0.50, 0.60]:
                    y_pred = (proba >= thr).astype(int)
                    y_true = res.y_true
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    notes.append(f"At threshold {thr:.2f}, precision is {prec:.3f} and recall is {rec:.3f}.")

            for n in notes:
                st.write(f"- {n}")

            c1, c2 = st.columns(2, gap="large")
            with c1:
                if res.importances is not None and not res.importances.empty:
                    fig = px.bar(res.importances.head(15)[::-1], x="importance", y="feature", orientation="h", title=None)
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Importance", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if res.proba is not None:
                    bands = pd.cut(
                        pd.Series(res.proba),
                        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
                        include_lowest=True,
                    )
                    band = bands.value_counts().sort_index()
                    band_df = pd.DataFrame({"band": band.index.astype(str), "count": band.values})
                    fig = px.bar(band_df, x="band", y="count", title=None)
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Probability band", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Model-driven insights could not be generated: {e}")
