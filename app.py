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


# =========================================================
# App configuration
# =========================================================
st.set_page_config(
    page_title="Superstore Commercial Performance Dashboard",
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

DISCOUNT_BAND = "Discount Band"
PROFIT_MARGIN = "Profit Margin"
SHIPPING_DAYS = "Shipping Days"
IS_LOSS_ORDER = "Loss Order"
ORDER_MONTH = "Order Month"
ORDER_YEAR = "Order Year"
ORDER_QUARTER = "Order Quarter"
ORDER_MONTH_NUM = "Order Month Number"

MODEL_FEATURE_PRIORITY = [
    COL_SALES,
    COL_DISCOUNT,
    COL_QUANTITY,
    SHIPPING_DAYS,
    ORDER_YEAR,
    ORDER_QUARTER,
    ORDER_MONTH_NUM,
    COL_REGION,
    COL_CATEGORY,
    COL_SUBCATEGORY,
    COL_SEGMENT,
    COL_SHIPMODE,
    COL_STATE,
]


# =========================================================
# Styling
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root{
            --bg:#f7f8fc;
            --panel:#ffffff;
            --panel-soft:#f9fafb;
            --text:#111827;
            --muted:#6b7280;
            --border:rgba(17,24,39,0.10);
            --accent:#2563eb;
            --accent-soft:rgba(37,99,235,0.08);
            --shadow:0 12px 32px rgba(15,23,42,0.06);
            --shadow-soft:0 6px 18px rgba(15,23,42,0.05);
            --radius:18px;
          }

          html, body, [data-testid="stAppViewContainer"]{
            background: var(--bg) !important;
            color: var(--text) !important;
          }

          [data-testid="stHeader"]{
            background: rgba(247,248,252,0.85);
          }

          [data-testid="stSidebar"]{
            background: #fbfcff !important;
            border-right: 1px solid var(--border);
          }

          .block-container{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
          }

          .hero{
            background: linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 24px 24px 18px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
          }

          .hero-title{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text);
            margin: 0 0 8px 0;
          }

          .hero-subtitle{
            margin: 0;
            color: var(--muted);
            font-size: 15px;
            line-height: 1.6;
            max-width: 920px;
          }

          .hero-strip{
            margin-top: 16px;
            padding: 12px 14px;
            border-radius: 16px;
            background: var(--accent-soft);
            border: 1px solid rgba(37,99,235,0.12);
            color: #1e3a8a;
            font-size: 13px;
          }

          .section-label{
            font-size: 18px;
            font-weight: 800;
            color: var(--text);
            margin: 2px 0 10px 0;
          }

          .card{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-soft);
            padding: 16px 16px;
          }

          .card-title{
            margin: 0 0 6px 0;
            color: var(--text);
            font-size: 16px;
            font-weight: 800;
          }

          .card-subtitle{
            margin: 0 0 12px 0;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
          }

          .kpi-grid{
            display:grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 8px;
          }

          @media (max-width: 1280px){
            .kpi-grid{ grid-template-columns: repeat(3, minmax(0, 1fr)); }
          }
          @media (max-width: 760px){
            .kpi-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
          @media (max-width: 540px){
            .kpi-grid{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
          }

          .kpi-card{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 14px 14px 12px 14px;
            min-height: 102px;
          }

          .kpi-label{
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 8px;
          }

          .kpi-value{
            font-size: 24px;
            font-weight: 800;
            color: var(--text);
            line-height: 1.1;
            margin-bottom: 8px;
          }

          .kpi-note{
            font-size: 12px;
            color: var(--muted);
            line-height: 1.4;
          }

          .insight-list{
            margin: 0;
            padding-left: 18px;
            color: var(--text);
          }

          .insight-list li{
            margin-bottom: 8px;
            line-height: 1.55;
          }

          .badge-row{
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top: 8px;
          }

          .badge{
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--panel-soft);
            border: 1px solid var(--border);
            color: var(--text);
            font-size: 12px;
          }

          .divider{
            height:1px;
            background: var(--border);
            margin: 8px 0 16px 0;
          }

          .tabs-wrap [data-baseweb="tab-list"]{
            gap: 8px;
          }

          .tabs-wrap [data-baseweb="tab"]{
            background: rgba(255,255,255,0.75);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 8px 16px;
          }

          .tabs-wrap [aria-selected="true"]{
            background: var(--panel);
            color: var(--text);
            border-color: rgba(37,99,235,0.20);
            box-shadow: var(--shadow-soft);
          }

          .js-plotly-plot .plotly .modebar{
            opacity: 0.08;
          }

          .js-plotly-plot .plotly:hover .modebar{
            opacity: 1;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# Formatting helpers
# =========================================================
def format_currency(value: Optional[float], digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{digits}f}"


def format_number(value: Optional[float], digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def format_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:,.{digits}f}%"


def safe_divide(a, b):
    if np.isscalar(a) and np.isscalar(b):
        if b in [0, None] or pd.isna(b):
            return np.nan
        return a / b

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    valid = (~np.isnan(b_arr)) & (b_arr != 0)
    result = np.divide(
        a_arr,
        b_arr,
        out=np.full(a_arr.shape, np.nan, dtype=float),
        where=valid,
    )
    return result


def escape_html(text: str) -> str:
    return escape(str(text))


# =========================================================
# General utilities
# =========================================================
def _safe_read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read CSV. Last error: {last_err}")


def _safe_read_csv_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as file:
        return _safe_read_csv_from_bytes(file.read())


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(col in df.columns for col in cols)


def _df_fingerprint(df: pd.DataFrame) -> str:
    sample = df.head(500).copy()
    hashed = pd.util.hash_pandas_object(sample, index=True).sum()
    return f"{df.shape[0]}x{df.shape[1]}-{int(hashed)}"


def numeric_or_default(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if col not in df.columns:
        return default
    value = _to_numeric(df[col]).median(skipna=True)
    return float(value) if not pd.isna(value) else default


# =========================================================
# Data loading and preparation
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(use_upload: bool, uploaded_file_bytes: Optional[bytes]) -> Tuple[pd.DataFrame, str]:
    if use_upload and uploaded_file_bytes is not None:
        return _safe_read_csv_from_bytes(uploaded_file_bytes), "Uploaded CSV"

    if os.path.exists(DEFAULT_DATA_FILENAME):
        return _safe_read_csv_from_path(DEFAULT_DATA_FILENAME), DEFAULT_DATA_FILENAME

    if os.path.exists(FALLBACK_DATA_FILENAME):
        return _safe_read_csv_from_path(FALLBACK_DATA_FILENAME), FALLBACK_DATA_FILENAME

    raise FileNotFoundError(
        f"Place '{DEFAULT_DATA_FILENAME}' or '{FALLBACK_DATA_FILENAME}' beside app.py, or upload a CSV in the sidebar."
    )


@st.cache_data(show_spinner=False)
def prepare_superstore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    out = out.drop_duplicates()

    for col in [COL_ORDER_DATE, COL_SHIP_DATE]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in [COL_SALES, COL_PROFIT, COL_DISCOUNT, COL_QUANTITY]:
        if col in out.columns:
            out[col] = _to_numeric(out[col])

    if COL_ORDER_DATE in out.columns:
        out[ORDER_MONTH] = out[COL_ORDER_DATE].dt.to_period("M").dt.to_timestamp()
        out[ORDER_YEAR] = out[COL_ORDER_DATE].dt.year
        out[ORDER_QUARTER] = out[COL_ORDER_DATE].dt.quarter
        out[ORDER_MONTH_NUM] = out[COL_ORDER_DATE].dt.month

    if _has_cols(out, [COL_ORDER_DATE, COL_SHIP_DATE]):
        shipping_days = (out[COL_SHIP_DATE] - out[COL_ORDER_DATE]).dt.days
        out[SHIPPING_DAYS] = shipping_days.clip(lower=0)

    if _has_cols(out, [COL_SALES, COL_PROFIT]):
        out[PROFIT_MARGIN] = np.where(out[COL_SALES].abs() > 0, out[COL_PROFIT] / out[COL_SALES], np.nan)
        out[IS_LOSS_ORDER] = (out[COL_PROFIT] < 0).astype(int)

    if COL_DISCOUNT in out.columns:
        out[DISCOUNT_BAND] = pd.cut(
            out[COL_DISCOUNT].fillna(-0.001),
            bins=[-0.01, 0.00, 0.10, 0.20, 0.30, 1.00],
            labels=[
                "No discount",
                "0–10%",
                "10–20%",
                "20–30%",
                "30%+",
            ],
            include_lowest=True,
        )

    for col in [COL_REGION, COL_CATEGORY, COL_SUBCATEGORY, COL_SEGMENT, COL_SHIPMODE, COL_STATE, COL_CITY]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Unknown")

    return out


# =========================================================
# Filtering
# =========================================================
def multiselect_filter(label: str, df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    values = sorted([str(v) for v in df[col].dropna().unique().tolist()])
    if not values:
        return []
    return st.multiselect(label, values, default=[])


def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    region_values: List[str],
    category_values: List[str],
    segment_values: List[str],
    shipmode_values: List[str],
) -> pd.DataFrame:
    filtered = df.copy()

    if date_range is not None and COL_ORDER_DATE in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered[COL_ORDER_DATE].dt.date >= start_date) &
            (filtered[COL_ORDER_DATE].dt.date <= end_date)
        ]

    for col, selected in [
        (COL_REGION, region_values),
        (COL_CATEGORY, category_values),
        (COL_SEGMENT, segment_values),
        (COL_SHIPMODE, shipmode_values),
    ]:
        if selected and col in filtered.columns:
            filtered = filtered[filtered[col].isin(selected)]

    return filtered


def filter_context_text(
    start_date,
    end_date,
    region_values: List[str],
    category_values: List[str],
    segment_values: List[str],
    shipmode_values: List[str],
) -> str:
    parts = []

    if start_date and end_date:
        parts.append(f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")

    def describe(name: str, values: List[str]) -> Optional[str]:
        if not values:
            return None
        if len(values) == 1:
            return f"{name}: {values[0]}"
        return f"{name}: {len(values)} selected"

    for name, values in [
        ("Region", region_values),
        ("Category", category_values),
        ("Segment", segment_values),
        ("Ship mode", shipmode_values),
    ]:
        text = describe(name, values)
        if text:
            parts.append(text)

    return "Scope: " + " | ".join(parts) if parts else "Scope: full available dataset"


# =========================================================
# Aggregations
# =========================================================
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if ORDER_MONTH not in df.columns:
        return pd.DataFrame()
    agg = (
        df.groupby(ORDER_MONTH, dropna=False)
        .agg(
            Sales=(COL_SALES, "sum") if COL_SALES in df.columns else (ORDER_MONTH, "size"),
            Profit=(COL_PROFIT, "sum") if COL_PROFIT in df.columns else (ORDER_MONTH, "size"),
            Orders=(ORDER_MONTH, "size"),
        )
        .reset_index()
        .sort_values(ORDER_MONTH)
    )
    if COL_SALES in df.columns:
        agg["Average Order Value"] = safe_divide(agg["Sales"], agg["Orders"])
    if _has_cols(agg, ["Profit", "Sales"]):
        agg[PROFIT_MARGIN] = np.where(agg["Sales"].abs() > 0, agg["Profit"] / agg["Sales"], np.nan)
    return agg


def aggregate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    metrics = {}
    if COL_SALES in df.columns:
        metrics[COL_SALES] = (COL_SALES, "sum")
    if COL_PROFIT in df.columns:
        metrics[COL_PROFIT] = (COL_PROFIT, "sum")
    if COL_QUANTITY in df.columns:
        metrics[COL_QUANTITY] = (COL_QUANTITY, "sum")
    metrics["Orders"] = (group_col, "size")
    grouped = df.groupby(group_col, dropna=False).agg(**metrics).reset_index()
    if _has_cols(grouped, [COL_PROFIT, COL_SALES]):
        grouped[PROFIT_MARGIN] = np.where(
            grouped[COL_SALES].abs() > 0,
            grouped[COL_PROFIT] / grouped[COL_SALES],
            np.nan,
        )
    return grouped


def profit_by_region_category(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, [COL_REGION, COL_CATEGORY, COL_PROFIT]):
        return pd.DataFrame()
    pivot = (
        df.pivot_table(
            index=COL_REGION,
            columns=COL_CATEGORY,
            values=COL_PROFIT,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )
    return pivot


def discount_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    if DISCOUNT_BAND not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby(DISCOUNT_BAND, dropna=False, observed=False)
        .agg(
            Orders=(DISCOUNT_BAND, "size"),
            Average_Discount=(COL_DISCOUNT, "mean") if COL_DISCOUNT in df.columns else (DISCOUNT_BAND, "size"),
            Sales=(COL_SALES, "sum") if COL_SALES in df.columns else (DISCOUNT_BAND, "size"),
            Profit=(COL_PROFIT, "sum") if COL_PROFIT in df.columns else (DISCOUNT_BAND, "size"),
            Avg_Profit=(COL_PROFIT, "mean") if COL_PROFIT in df.columns else (DISCOUNT_BAND, "size"),
            Loss_Rate=(IS_LOSS_ORDER, "mean") if IS_LOSS_ORDER in df.columns else (DISCOUNT_BAND, "size"),
        )
        .reset_index()
    )

    if _has_cols(summary, ["Profit", "Sales"]):
        summary[PROFIT_MARGIN] = np.where(
            summary["Sales"].abs() > 0,
            summary["Profit"] / summary["Sales"],
            np.nan,
        )

    return summary


def category_discount_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_cols(df, [COL_CATEGORY, DISCOUNT_BAND, COL_PROFIT]):
        return pd.DataFrame()
    summary = (
        df.groupby([COL_CATEGORY, DISCOUNT_BAND], dropna=False, observed=False)
        .agg(
            Orders=(DISCOUNT_BAND, "size"),
            Profit=(COL_PROFIT, "sum"),
            Avg_Profit=(COL_PROFIT, "mean"),
            Loss_Rate=(IS_LOSS_ORDER, "mean") if IS_LOSS_ORDER in df.columns else (DISCOUNT_BAND, "size"),
        )
        .reset_index()
    )
    return summary


def subcategory_performance(df: pd.DataFrame) -> pd.DataFrame:
    if COL_SUBCATEGORY not in df.columns:
        return pd.DataFrame()
    g = aggregate_by(df, COL_SUBCATEGORY)
    if g.empty or COL_PROFIT not in g.columns:
        return g
    return g.sort_values(COL_PROFIT, ascending=False)


# =========================================================
# Insight generation
# =========================================================
def build_data_driven_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []

    if df.empty:
        return ["No records remain after the current filters. Widen the filter scope to recover dashboard insights."]

    if _has_cols(df, [COL_SALES, COL_PROFIT]):
        total_sales = float(df[COL_SALES].sum())
        total_profit = float(df[COL_PROFIT].sum())
        profit_margin = safe_divide(total_profit, total_sales)
        insights.append(
            f"The active slice generates {format_currency(total_sales)} in sales and {format_currency(total_profit)} in profit, equivalent to a {format_pct(profit_margin)} margin."
        )

    month_df = monthly_summary(df)
    if not month_df.empty and "Sales" in month_df.columns and len(month_df) >= 2:
        peak_row = month_df.loc[month_df["Sales"].idxmax()]
        peak_month = peak_row[ORDER_MONTH].strftime("%b %Y")
        insights.append(
            f"Revenue peaks in {peak_month}, when monthly sales reach {format_currency(float(peak_row['Sales']))}."
        )

    if _has_cols(df, [COL_CATEGORY, COL_SALES, COL_PROFIT]):
        cat = aggregate_by(df, COL_CATEGORY).sort_values(COL_SALES, ascending=False)
        if not cat.empty:
            revenue_leader = cat.iloc[0]
            worst_margin_row = cat.sort_values(PROFIT_MARGIN, ascending=True, na_position="last").iloc[0]
            insights.append(
                f"{revenue_leader[COL_CATEGORY]} is the largest revenue contributor, while {worst_margin_row[COL_CATEGORY]} shows the weakest profit margin among visible categories."
            )

    if _has_cols(df, [COL_SUBCATEGORY, COL_PROFIT]):
        sub = subcategory_performance(df)
        if len(sub) >= 2:
            best = sub.iloc[0]
            worst = sub.iloc[-1]
            insights.append(
                f"At sub-category level, {best[COL_SUBCATEGORY]} contributes the most profit, while {worst[COL_SUBCATEGORY]} is the largest profit drag in the current selection."
            )

    if _has_cols(df, [COL_REGION, COL_PROFIT]):
        region = aggregate_by(df, COL_REGION).sort_values(COL_PROFIT, ascending=False)
        if len(region) >= 2:
            insights.append(
                f"{region.iloc[0][COL_REGION]} leads regional profit, whereas {region.iloc[-1][COL_REGION]} trails the group on profitability."
            )

    if _has_cols(df, [COL_DISCOUNT, COL_PROFIT]):
        corr = df[[COL_DISCOUNT, COL_PROFIT]].corr(numeric_only=True).iloc[0, 1]
        if not pd.isna(corr):
            direction = "negative" if corr < 0 else "positive"
            insights.append(
                f"The relationship between discount and profit is {direction}, with a correlation of {corr:.2f}, showing how pricing pressure is shaping margin outcomes."
            )

    if DISCOUNT_BAND in df.columns and IS_LOSS_ORDER in df.columns:
        band = discount_band_summary(df)
        if not band.empty and "Loss_Rate" in band.columns:
            highest_loss = band.sort_values("Loss_Rate", ascending=False).iloc[0]
            insights.append(
                f"The {highest_loss[DISCOUNT_BAND]} band carries the highest loss rate, making it the riskiest discount tier in the current slice."
            )

    return insights[:6]


# =========================================================
# UI helpers
# =========================================================
def render_kpis(items: List[Tuple[str, str, str]]) -> None:
    cards_html = []

    for label, value, note in items:
        cards_html.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{escape_html(label)}</div>"
            f"<div class='kpi-value'>{escape_html(value)}</div>"
            f"<div class='kpi-note'>{escape_html(note)}</div>"
            "</div>"
        )

    full_html = "<div class='kpi-grid'>" + "".join(cards_html) + "</div>"
    st.markdown(full_html, unsafe_allow_html=True)


def render_section_title(text: str) -> None:
    st.markdown(f"<div class='section-label'>{escape_html(text)}</div>", unsafe_allow_html=True)


def render_insight_list(items: List[str]) -> None:
    html = "".join([f"<li>{escape_html(item)}</li>" for item in items])
    st.markdown(f"<ul class='insight-list'>{html}</ul>", unsafe_allow_html=True)


def render_badges(items: List[str]) -> None:
    badges = "".join([f"<span class='badge'>{escape_html(item)}</span>" for item in items if item])
    st.markdown(f"<div class='badge-row'>{badges}</div>", unsafe_allow_html=True)


# =========================================================
# Modeling
# =========================================================
@dataclass
class ModelResult:
    task: str
    model: Pipeline
    feature_columns: List[str]
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    importances: pd.DataFrame
    positive_rate: Optional[float] = None


def choose_model_features(df: pd.DataFrame) -> List[str]:
    features = [col for col in MODEL_FEATURE_PRIORITY if col in df.columns]
    final_features: List[str] = []

    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            final_features.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 75:
                final_features.append(col)

    return final_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def train_loss_classifier(df: pd.DataFrame, fingerprint: str, seed: int = 42) -> ModelResult:
    if IS_LOSS_ORDER not in df.columns:
        raise ValueError("Loss classification requires a Profit column so that loss-making orders can be labelled.")

    work = df.copy()
    feature_columns = choose_model_features(work)
    if len(feature_columns) < 3:
        raise ValueError("Not enough suitable features were found for loss classification.")

    work = work.dropna(subset=[IS_LOSS_ORDER]).copy()
    if len(work) < 300:
        raise ValueError("At least 300 filtered rows are recommended for stable loss-risk modelling.")

    X = work[feature_columns].copy()
    y = work[IS_LOSS_ORDER].astype(int).copy()

    if y.nunique() < 2:
        raise ValueError("The filtered data contains only one profitability class. Widen the filters to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    pre = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=350,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)

    metrics = {
        "ROC AUC": float(roc_auc_score(y_test, proba)),
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
    }

    feature_names = get_feature_names(pipe.named_steps["preprocessor"])
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return ModelResult(
        task="classification",
        model=pipe,
        feature_columns=feature_columns,
        metrics=metrics,
        y_true=y_test.to_numpy(),
        y_pred=pred,
        y_proba=proba,
        importances=importances,
        positive_rate=float(y.mean()),
    )


@st.cache_resource(show_spinner=False)
def train_profit_regressor(df: pd.DataFrame, fingerprint: str, seed: int = 42) -> ModelResult:
    if COL_PROFIT not in df.columns:
        raise ValueError("Profit regression requires a Profit column.")

    work = df.copy()
    feature_columns = [col for col in choose_model_features(work) if col != COL_PROFIT]
    if len(feature_columns) < 3:
        raise ValueError("Not enough suitable features were found for profit prediction.")

    work = work.dropna(subset=[COL_PROFIT]).copy()
    if len(work) < 300:
        raise ValueError("At least 300 filtered rows are recommended for stable profit prediction.")

    X = work[feature_columns].copy()
    y = _to_numeric(work[COL_PROFIT]).copy()
    valid = ~y.isna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
    )

    pre = build_preprocessor(X_train)
    model = RandomForestRegressor(
        n_estimators=350,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R²": float(r2_score(y_test, pred)),
    }

    feature_names = get_feature_names(pipe.named_steps["preprocessor"])
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return ModelResult(
        task="regression",
        model=pipe,
        feature_columns=feature_columns,
        metrics=metrics,
        y_true=y_test.to_numpy(),
        y_pred=pred,
        y_proba=None,
        importances=importances,
    )


def build_threshold_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "F1": f1_score(y_true, pred, zero_division=0),
                "Accuracy": accuracy_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)


def scenario_defaults(df: pd.DataFrame) -> Dict[str, object]:
    defaults: Dict[str, object] = {}

    numeric_defaults = {
        COL_SALES: numeric_or_default(df, COL_SALES, 200.0),
        COL_DISCOUNT: numeric_or_default(df, COL_DISCOUNT, 0.10),
        COL_QUANTITY: numeric_or_default(df, COL_QUANTITY, 3.0),
        SHIPPING_DAYS: numeric_or_default(df, SHIPPING_DAYS, 4.0),
        ORDER_YEAR: numeric_or_default(df, ORDER_YEAR, 2017.0),
        ORDER_QUARTER: numeric_or_default(df, ORDER_QUARTER, 4.0),
        ORDER_MONTH_NUM: numeric_or_default(df, ORDER_MONTH_NUM, 11.0),
    }
    defaults.update(numeric_defaults)

    for col in [COL_REGION, COL_CATEGORY, COL_SUBCATEGORY, COL_SEGMENT, COL_SHIPMODE, COL_STATE]:
        if col in df.columns and df[col].dropna().nunique() > 0:
            defaults[col] = str(df[col].mode(dropna=True).iloc[0])

    return defaults


def build_scenario_input(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    defaults = scenario_defaults(df)
    payload: Dict[str, object] = {}

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        if COL_SALES in feature_columns:
            payload[COL_SALES] = st.number_input(
                "Scenario sales",
                min_value=0.0,
                value=float(defaults.get(COL_SALES, 200.0)),
                step=25.0,
            )
        if COL_DISCOUNT in feature_columns:
            payload[COL_DISCOUNT] = st.slider(
                "Scenario discount",
                min_value=0.0,
                max_value=0.80,
                value=float(defaults.get(COL_DISCOUNT, 0.10)),
                step=0.01,
            )
        if COL_QUANTITY in feature_columns:
            payload[COL_QUANTITY] = st.number_input(
                "Scenario quantity",
                min_value=1.0,
                value=float(defaults.get(COL_QUANTITY, 3.0)),
                step=1.0,
            )

    with c2:
        if SHIPPING_DAYS in feature_columns:
            payload[SHIPPING_DAYS] = st.number_input(
                "Scenario shipping days",
                min_value=0.0,
                value=float(defaults.get(SHIPPING_DAYS, 4.0)),
                step=1.0,
            )
        if ORDER_YEAR in feature_columns:
            payload[ORDER_YEAR] = int(
                st.number_input(
                    "Scenario order year",
                    min_value=2014,
                    max_value=2030,
                    value=int(defaults.get(ORDER_YEAR, 2017)),
                    step=1,
                )
            )
        if ORDER_QUARTER in feature_columns:
            payload[ORDER_QUARTER] = int(
                st.selectbox(
                    "Scenario quarter",
                    [1, 2, 3, 4],
                    index=int(defaults.get(ORDER_QUARTER, 4)) - 1,
                )
            )

    with c3:
        if ORDER_MONTH_NUM in feature_columns:
            month_options = list(range(1, 13))
            default_month = int(defaults.get(ORDER_MONTH_NUM, 11))
            payload[ORDER_MONTH_NUM] = int(
                st.selectbox(
                    "Scenario month",
                    month_options,
                    index=max(0, min(11, default_month - 1)),
                    format_func=lambda x: pd.Timestamp(year=2024, month=x, day=1).strftime("%B"),
                )
            )

        for col, label in [
            (COL_REGION, "Scenario region"),
            (COL_CATEGORY, "Scenario category"),
            (COL_SUBCATEGORY, "Scenario sub-category"),
            (COL_SEGMENT, "Scenario segment"),
            (COL_SHIPMODE, "Scenario ship mode"),
            (COL_STATE, "Scenario state"),
        ]:
            if col in feature_columns and col in df.columns:
                options = sorted([str(v) for v in df[col].dropna().unique().tolist()])
                if options:
                    default_value = str(defaults.get(col, options[0]))
                    default_index = options.index(default_value) if default_value in options else 0
                    payload[col] = st.selectbox(label, options, index=default_index)

    for col in feature_columns:
        if col not in payload:
            payload[col] = defaults.get(col, np.nan)

    return pd.DataFrame([payload], columns=feature_columns)


# =========================================================
# Load data
# =========================================================
with st.sidebar:
    st.markdown("### Dashboard controls")
    use_upload = st.toggle("Upload CSV instead of packaged file", value=False)
    uploaded_file = st.file_uploader("Upload Superstore CSV", type=["csv"], disabled=not use_upload)

try:
    raw_df, source_label = load_data(
        use_upload=use_upload,
        uploaded_file_bytes=uploaded_file.getvalue() if uploaded_file is not None else None,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = prepare_superstore(raw_df)

# Sidebar filters
with st.sidebar:
    st.markdown("### Filter scope")

    if COL_ORDER_DATE in df.columns and df[COL_ORDER_DATE].notna().any():
        min_date = df[COL_ORDER_DATE].min().date()
        max_date = df[COL_ORDER_DATE].max().date()
        selected_range = st.date_input(
            "Order date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = min_date, max_date
        date_range = (start_date, end_date)
    else:
        start_date, end_date, date_range = None, None, None

    selected_regions = multiselect_filter("Region", df, COL_REGION)
    selected_categories = multiselect_filter("Category", df, COL_CATEGORY)
    selected_segments = multiselect_filter("Segment", df, COL_SEGMENT)
    selected_shipmodes = multiselect_filter("Ship mode", df, COL_SHIPMODE)

    st.markdown("### Data source")
    st.caption(source_label)
    st.caption(f"Prepared records: {len(df):,}")

filtered = apply_filters(
    df=df,
    date_range=date_range,
    region_values=selected_regions,
    category_values=selected_categories,
    segment_values=selected_segments,
    shipmode_values=selected_shipmodes,
)

if filtered.empty:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Superstore Commercial Performance Dashboard</div>
          <p class="hero-subtitle">
            No records remain after the active filters. Widen the scope in the sidebar to restore performance insights.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# =========================================================
# Hero
# =========================================================
headline_insights = build_data_driven_insights(filtered)[:3]
context_text = filter_context_text(
    start_date=start_date,
    end_date=end_date,
    region_values=selected_regions,
    category_values=selected_categories,
    segment_values=selected_segments,
    shipmode_values=selected_shipmodes,
)

hero_html = f"""
<div class="hero">
  <div class="hero-title">Superstore Commercial Performance Dashboard</div>
  <p class="hero-subtitle">
    Insight-led analysis of revenue, profitability, discount discipline, and predictive risk across the active slice of Superstore orders.
  </p>
  <div class="hero-strip">{escape_html(context_text)}</div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


# =========================================================
# Executive KPIs
# =========================================================
total_sales = float(filtered[COL_SALES].sum()) if COL_SALES in filtered.columns else np.nan
total_profit = float(filtered[COL_PROFIT].sum()) if COL_PROFIT in filtered.columns else np.nan
order_count = len(filtered)
avg_order_value = safe_divide(total_sales, order_count)
profit_margin_value = safe_divide(total_profit, total_sales)
loss_rate_value = float(filtered[IS_LOSS_ORDER].mean()) if IS_LOSS_ORDER in filtered.columns else np.nan
avg_discount = float(filtered[COL_DISCOUNT].mean()) if COL_DISCOUNT in filtered.columns else np.nan
avg_shipping_days = float(filtered[SHIPPING_DAYS].median()) if SHIPPING_DAYS in filtered.columns else np.nan

render_kpis(
    [
        ("Revenue", format_currency(total_sales), "Total sales across the active slice"),
        ("Profit", format_currency(total_profit), "Net profit after all discounts"),
        ("Profit margin", format_pct(profit_margin_value), "Profit as a share of revenue"),
        ("Loss-making order rate", format_pct(loss_rate_value), "Orders with negative profit"),
        ("Average order value", format_currency(avg_order_value), "Revenue per order"),
        ("Average discount", format_pct(avg_discount), "Mean discount applied"),
    ]
)

st.markdown("<div class='tabs-wrap'>", unsafe_allow_html=True)
tab_overview, tab_commercial, tab_discount, tab_model, tab_appendix = st.tabs(
    [
        "Executive summary",
        "Commercial performance",
        "Margin and discount",
        "Predictive analytics",
        "Data appendix",
    ]
)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Executive summary tab
# =========================================================
with tab_overview:
    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    with col_left:
        render_section_title("Data-driven insights")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_insight_list(build_data_driven_insights(filtered))
        st.markdown("</div>", unsafe_allow_html=True)

        month_df = monthly_summary(filtered)
        st.write("")
        render_section_title("Sales and profit trajectory")
        if not month_df.empty and _has_cols(month_df, ["Sales", "Profit"]):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=month_df[ORDER_MONTH],
                    y=month_df["Sales"],
                    mode="lines+markers",
                    name="Sales",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=month_df[ORDER_MONTH],
                    y=month_df["Profit"],
                    mode="lines+markers",
                    name="Profit",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Month",
                yaxis_title="Sales",
                yaxis2=dict(title="Profit", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly analysis requires Order Date, Sales, and Profit fields.")

    with col_right:
        render_section_title("Headline takeaways")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_badges(
            [
                f"Orders: {format_number(order_count)}",
                f"Median shipping days: {format_number(avg_shipping_days)}",
                f"Average order value: {format_currency(avg_order_value)}",
            ]
        )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        extra_notes: List[str] = []
        if _has_cols(filtered, [COL_CATEGORY, COL_SALES]):
            top_category = aggregate_by(filtered, COL_CATEGORY).sort_values(COL_SALES, ascending=False).head(1)
            if not top_category.empty:
                share = safe_divide(float(top_category.iloc[0][COL_SALES]), total_sales)
                extra_notes.append(
                    f"{top_category.iloc[0][COL_CATEGORY]} contributes the largest sales share at {format_pct(share)} of visible revenue."
                )

        if _has_cols(filtered, [COL_REGION, COL_PROFIT]):
            top_region = aggregate_by(filtered, COL_REGION).sort_values(COL_PROFIT, ascending=False).head(1)
            if not top_region.empty:
                extra_notes.append(
                    f"{top_region.iloc[0][COL_REGION]} is the strongest regional profit contributor within the current selection."
                )

        if DISCOUNT_BAND in filtered.columns and COL_PROFIT in filtered.columns:
            disc = discount_band_summary(filtered)
            disc = disc.dropna(subset=["Avg_Profit"]) if "Avg_Profit" in disc.columns else disc
            if not disc.empty:
                weakest_band = disc.sort_values("Avg_Profit", ascending=True).iloc[0]
                extra_notes.append(
                    f"{weakest_band[DISCOUNT_BAND]} is the most margin-destructive discount band based on average profit per order."
                )

        render_insight_list(extra_notes[:3] if extra_notes else headline_insights)

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        render_section_title("Sub-category winners and drags")
        sub = subcategory_performance(filtered)
        if not sub.empty and COL_PROFIT in sub.columns:
            display = pd.concat([sub.head(5), sub.tail(5)], axis=0).drop_duplicates()
            display["Direction"] = np.where(display[COL_PROFIT] >= 0, "Profit contributor", "Profit drag")
            fig = px.bar(
                display.sort_values(COL_PROFIT),
                x=COL_PROFIT,
                y=COL_SUBCATEGORY,
                color="Direction",
                orientation="h",
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Profit",
                yaxis_title="Sub-category",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sub-category performance requires Sub-Category and Profit fields.")


# =========================================================
# Commercial performance tab
# =========================================================
with tab_commercial:
    render_section_title("Where revenue and profit are concentrated")
    c1, c2 = st.columns([1.15, 0.85], gap="large")

    with c1:
        month_df = monthly_summary(filtered)
        if not month_df.empty and _has_cols(month_df, ["Sales", "Orders"]):
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=month_df[ORDER_MONTH],
                    y=month_df["Orders"],
                    name="Orders",
                    opacity=0.65,
                )
            )
            if "Average Order Value" in month_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=month_df[ORDER_MONTH],
                        y=month_df["Average Order Value"],
                        mode="lines+markers",
                        name="Average order value",
                        yaxis="y2",
                    )
                )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Month",
                yaxis_title="Orders",
                yaxis2=dict(
                    title="Average order value",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Order cadence requires Order Date and Sales fields.")

    with c2:
        if _has_cols(filtered, [COL_CATEGORY, COL_SALES]):
            cat = aggregate_by(filtered, COL_CATEGORY).sort_values(COL_SALES, ascending=False)
            metric = COL_PROFIT if COL_PROFIT in cat.columns else COL_SALES
            fig = px.bar(
                cat,
                x=COL_CATEGORY,
                y=metric,
                text_auto=".2s" if metric == COL_SALES else False,
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Category",
                yaxis_title=metric,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category performance requires Category plus Sales or Profit.")

    st.write("")
    render_section_title("Regional and product-level profit quality")
    c3, c4 = st.columns([0.95, 1.05], gap="large")

    with c3:
        heatmap_df = profit_by_region_category(filtered)
        if not heatmap_df.empty:
            fig = go.Figure(
                data=go.Heatmap(
                    z=heatmap_df.values,
                    x=heatmap_df.columns.tolist(),
                    y=heatmap_df.index.tolist(),
                    colorbar_title="Profit",
                )
            )
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Category",
                yaxis_title="Region",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This heatmap requires Region, Category, and Profit.")

    with c4:
        if _has_cols(filtered, [COL_SUBCATEGORY, COL_SALES, COL_PROFIT]):
            sub = aggregate_by(filtered, COL_SUBCATEGORY)
            bubble_size = COL_QUANTITY if COL_QUANTITY in sub.columns else "Orders"
            fig = px.scatter(
                sub,
                x=COL_SALES,
                y=COL_PROFIT,
                size=bubble_size,
                hover_name=COL_SUBCATEGORY,
                text=COL_SUBCATEGORY,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Sales",
                yaxis_title="Profit",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sub-category opportunity mapping requires Sales and Profit.")

    st.write("")
    if _has_cols(filtered, [COL_REGION, COL_SALES, COL_PROFIT]):
        render_section_title("Regional scoreboard")
        region = aggregate_by(filtered, COL_REGION).sort_values(COL_PROFIT, ascending=False).copy()
        display_cols = [COL_REGION]
        if COL_SALES in region.columns:
            display_cols.append(COL_SALES)
        if COL_PROFIT in region.columns:
            display_cols.append(COL_PROFIT)
        if PROFIT_MARGIN in region.columns:
            display_cols.append(PROFIT_MARGIN)
        if "Orders" in region.columns:
            display_cols.append("Orders")
        scoreboard = region[display_cols].copy()
        if COL_SALES in scoreboard.columns:
            scoreboard[COL_SALES] = scoreboard[COL_SALES].map(format_currency)
        if COL_PROFIT in scoreboard.columns:
            scoreboard[COL_PROFIT] = scoreboard[COL_PROFIT].map(format_currency)
        if PROFIT_MARGIN in scoreboard.columns:
            scoreboard[PROFIT_MARGIN] = scoreboard[PROFIT_MARGIN].map(format_pct)
        if "Orders" in scoreboard.columns:
            scoreboard["Orders"] = scoreboard["Orders"].map(lambda x: f"{int(x):,}")
        st.dataframe(scoreboard, use_container_width=True, hide_index=True)


# =========================================================
# Margin and discount tab
# =========================================================
with tab_discount:
    render_section_title("Data-driven insights")
    discount_insights: List[str] = []

    if COL_DISCOUNT in filtered.columns:
        discount_insights.append(
            f"Average discount is {format_pct(avg_discount)}, providing a direct read on current pricing pressure."
        )

    if DISCOUNT_BAND in filtered.columns and COL_PROFIT in filtered.columns:
        band = discount_band_summary(filtered)
        if not band.empty:
            lowest_margin_band = band.sort_values(PROFIT_MARGIN, ascending=True, na_position="last").iloc[0]
            highest_loss_band = band.sort_values("Loss_Rate", ascending=False, na_position="last").iloc[0]
            discount_insights.append(
                f"{lowest_margin_band[DISCOUNT_BAND]} shows the weakest profit margin among the visible discount tiers."
            )
            discount_insights.append(
                f"{highest_loss_band[DISCOUNT_BAND]} carries the highest proportion of loss-making orders."
            )

    if _has_cols(filtered, [COL_CATEGORY, COL_DISCOUNT, COL_PROFIT]):
        category_discount = (
            filtered.groupby(COL_CATEGORY)
            .agg(
                Average_Discount=(COL_DISCOUNT, "mean"),
                Profit=(COL_PROFIT, "sum"),
            )
            .reset_index()
            .sort_values("Average_Discount", ascending=False)
        )
        if not category_discount.empty:
            most_discounted = category_discount.iloc[0]
            discount_insights.append(
                f"{most_discounted[COL_CATEGORY]} is the most heavily discounted category in the current view."
            )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    render_insight_list(discount_insights if discount_insights else ["Discount analysis requires Discount and Profit fields."])
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    d1, d2 = st.columns([1.1, 0.9], gap="large")

    with d1:
        render_section_title("Profit quality by discount band")
        band = discount_band_summary(filtered)
        if not band.empty and {"Avg_Profit", "Loss_Rate"}.issubset(set(band.columns)):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=band[DISCOUNT_BAND], y=band["Avg_Profit"], name="Average profit"))
            fig.add_trace(
                go.Scatter(
                    x=band[DISCOUNT_BAND],
                    y=band["Loss_Rate"] * 100,
                    mode="lines+markers",
                    name="Loss rate",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Discount band",
                yaxis_title="Average profit",
                yaxis2=dict(title="Loss rate (%)", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This analysis requires Discount and Profit fields.")

    with d2:
        render_section_title("Discount versus profit distribution")
        if _has_cols(filtered, [COL_DISCOUNT, COL_PROFIT]):
            sample = filtered.copy()
            if len(sample) > 4000:
                sample = sample.sample(4000, random_state=42)
            sample["Profit status"] = np.where(sample[COL_PROFIT] >= 0, "Profitable", "Loss-making")
            fig = px.scatter(
                sample,
                x=COL_DISCOUNT,
                y=COL_PROFIT,
                color="Profit status",
                opacity=0.55,
                hover_data=[COL_CATEGORY] if COL_CATEGORY in sample.columns else None,
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Discount",
                yaxis_title="Profit",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Discount dispersion requires Discount and Profit.")

    st.write("")
    render_section_title("Category sensitivity to discounting")
    category_discount = category_discount_summary(filtered)
    if not category_discount.empty and {"Avg_Profit"}.issubset(set(category_discount.columns)):
        fig = px.bar(
            category_discount,
            x=DISCOUNT_BAND,
            y="Avg_Profit",
            color=COL_CATEGORY,
            barmode="group",
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Discount band",
            yaxis_title="Average profit per order",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category discount sensitivity requires Category, Discount, and Profit.")

    st.write("")
    if _has_cols(filtered, [COL_QUANTITY, COL_SALES, COL_PROFIT]):
        render_section_title("Order size versus commercial return")
        sample = filtered.copy()
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)
        sample["Profit status"] = np.where(sample[COL_PROFIT] >= 0, "Profitable", "Loss-making")
        sample["Absolute Profit"] = sample[COL_PROFIT].abs().clip(lower=1)
        fig = px.scatter(
            sample,
            x=COL_QUANTITY,
            y=COL_SALES,
            size="Absolute Profit",
            color="Profit status",
            hover_data=[COL_CATEGORY, COL_SUBCATEGORY] if _has_cols(sample, [COL_CATEGORY, COL_SUBCATEGORY]) else None,
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Quantity",
            yaxis_title="Sales",
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Predictive analytics tab
# =========================================================
with tab_model:
    render_section_title("Model framing")
    st.markdown(
        """
        <div class="card">
          <ul class="insight-list">
            <li>Data-driven insights come first in this dashboard; modelling is used here as a secondary decision-support layer.</li>
            <li>The classifier estimates whether an order is likely to become loss-making.</li>
            <li>The regressor estimates expected profit, allowing direct scenario testing for pricing and order mix decisions.</li>
            <li>Both models are retrained on the active filter scope, so metrics and drivers should be interpreted in that exact context.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fingerprint = _df_fingerprint(filtered)
    model_error = None
    clf_res: Optional[ModelResult] = None
    reg_res: Optional[ModelResult] = None

    try:
        clf_res = train_loss_classifier(filtered, fingerprint=fingerprint, seed=42)
        reg_res = train_profit_regressor(filtered, fingerprint=fingerprint, seed=42)
    except Exception as exc:
        model_error = str(exc)

    if model_error:
        st.warning(model_error)
    else:
        render_section_title("Model-driven insights")
        model_notes: List[str] = []

        if clf_res is not None:
            model_notes.append(
                f"The loss-risk classifier reaches ROC AUC {clf_res.metrics['ROC AUC']:.3f} and F1 {clf_res.metrics['F1']:.3f} on the holdout split."
            )
            if clf_res.positive_rate is not None:
                model_notes.append(
                    f"Loss-making orders account for {format_pct(clf_res.positive_rate)} of the training population."
                )
            if clf_res.importances is not None and not clf_res.importances.empty:
                top_drivers = ", ".join(clf_res.importances.head(5)["feature"].astype(str).tolist())
                model_notes.append(f"Top loss-risk drivers are {top_drivers}.")

        if reg_res is not None:
            model_notes.append(
                f"The profit model delivers MAE {format_currency(reg_res.metrics['MAE'])} and R² {reg_res.metrics['R²']:.3f} on the holdout split."
            )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_insight_list(model_notes)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        m1, m2 = st.columns([0.95, 1.05], gap="large")

        with m1:
            render_section_title("Loss-risk evaluation")
            render_kpis(
                [
                    ("ROC AUC", f"{clf_res.metrics['ROC AUC']:.3f}", "Ranking quality for loss-risk scores"),
                    ("Precision", f"{clf_res.metrics['Precision']:.3f}", "Share of flagged cases that are truly loss-making"),
                    ("Recall", f"{clf_res.metrics['Recall']:.3f}", "Share of loss-making orders captured"),
                    ("F1", f"{clf_res.metrics['F1']:.3f}", "Balance between precision and recall"),
                    ("Accuracy", f"{clf_res.metrics['Accuracy']:.3f}", "Overall classification hit rate"),
                    ("Loss prevalence", format_pct(clf_res.positive_rate), "Observed base rate in the training data"),
                ]
            )

            threshold_table = build_threshold_table(clf_res.y_true, clf_res.y_proba)
            threshold_display = threshold_table.copy()
            threshold_display["Threshold"] = threshold_display["Threshold"].map(lambda x: f"{x:.2f}")
            for metric_col in ["Precision", "Recall", "F1", "Accuracy"]:
                threshold_display[metric_col] = threshold_display[metric_col].map(lambda x: f"{x:.3f}")
            st.dataframe(threshold_display, use_container_width=True, hide_index=True)

            threshold = st.slider(
                "Interactive loss-risk threshold",
                min_value=0.20,
                max_value=0.80,
                value=0.50,
                step=0.01,
            )
            threshold_pred = (clf_res.y_proba >= threshold).astype(int)
            cm = confusion_matrix(clf_res.y_true, threshold_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual non-loss", "Actual loss"],
                columns=["Predicted non-loss", "Predicted loss"],
            )
            st.dataframe(cm_df, use_container_width=True)

        with m2:
            render_section_title("Model drivers")
            feature_view = clf_res.importances.head(15).sort_values("importance", ascending=True)
            fig = px.bar(
                feature_view,
                x="importance",
                y="feature",
                orientation="h",
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Importance",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)

            render_section_title("Predicted loss-risk distribution")
            probability_band = pd.cut(
                pd.Series(clf_res.y_proba),
                bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
                include_lowest=True,
            )
            band_df = probability_band.value_counts().sort_index().reset_index()
            band_df.columns = ["Probability band", "Orders"]
            fig = px.bar(band_df, x="Probability band", y="Orders")
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Predicted loss-risk band",
                yaxis_title="Orders",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        render_section_title("Scenario testing")
        st.markdown(
            """
            <div class="card">
              <p class="card-subtitle">
                Change the commercial inputs below to estimate both expected profit and the probability that the order becomes loss-making.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scenario_cols = sorted(
            set(clf_res.feature_columns).union(reg_res.feature_columns),
            key=lambda x: MODEL_FEATURE_PRIORITY.index(x) if x in MODEL_FEATURE_PRIORITY else 999
        )
        scenario_df = build_scenario_input(filtered, scenario_cols)

        scenario_loss_prob = float(clf_res.model.predict_proba(scenario_df[clf_res.feature_columns])[0, 1])
        scenario_profit = float(reg_res.model.predict(scenario_df[reg_res.feature_columns])[0])

        scenario_label = "High loss risk" if scenario_loss_prob >= threshold else "Low to moderate loss risk"
        scenario_note = (
            "The scenario is above the active loss-risk threshold and should be reviewed before approval."
            if scenario_loss_prob >= threshold
            else "The scenario remains below the active loss-risk threshold."
        )

        render_kpis(
            [
                ("Predicted profit", format_currency(scenario_profit), "Estimated order-level profit"),
                ("Loss-risk probability", format_pct(scenario_loss_prob), "Predicted chance of negative profit"),
                ("Decision view", scenario_label, scenario_note),
                ("Applied threshold", f"{threshold:.2f}", "Current classifier threshold"),
                ("Scenario discount", format_pct(float(scenario_df.iloc[0].get(COL_DISCOUNT, np.nan))), "Discount entered above"),
                ("Scenario sales", format_currency(float(scenario_df.iloc[0].get(COL_SALES, np.nan))), "Sales entered above"),
            ]
        )

        s1, s2 = st.columns(2, gap="large")
        with s1:
            actual_vs_pred = pd.DataFrame(
                {
                    "Actual profit": reg_res.y_true,
                    "Predicted profit": reg_res.y_pred,
                }
            )
            fig = px.scatter(actual_vs_pred, x="Actual profit", y="Predicted profit", opacity=0.5)
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with s2:
            reg_feature_view = reg_res.importances.head(15).sort_values("importance", ascending=True)
            fig = px.bar(
                reg_feature_view,
                x="importance",
                y="feature",
                orientation="h",
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Importance",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Data appendix
# =========================================================
with tab_appendix:
    render_section_title("Filtered dataset appendix")
    st.markdown(
        """
        <div class="card">
          <ul class="insight-list">
            <li>The main dashboard intentionally prioritises decision-ready insight over raw data preview.</li>
            <li>This appendix remains available for validation, spot-checking, and export.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns([1.0, 1.0], gap="large")

    with a1:
        render_section_title("Missingness summary")
        missing = (
            filtered.isna().mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        missing.columns = ["Column", "Missing rate"]
        missing["Missing rate"] = missing["Missing rate"].map(lambda x: f"{x * 100:.1f}%")
        st.dataframe(missing, use_container_width=True, hide_index=True)

    with a2:
        render_section_title("Filtered export")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_bytes,
            file_name="superstore_filtered_dashboard_extract.csv",
            mime="text/csv",
        )

    st.write("")
    render_section_title("Filtered record preview")
    preview_rows = st.slider("Rows to preview", min_value=10, max_value=150, value=30, step=10)
    st.dataframe(filtered.head(preview_rows), use_container_width=True, hide_index=True)